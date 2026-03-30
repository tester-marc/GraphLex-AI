"""
this is the Weaviate vector store for the regulatory chunks (Retrieval Layer)

It stores the text chunks extracted from the 6 Swiss/EU data protection PDFs as
vector embeddings and exposes the similarity search used by the RAG pipeline

Deployment:
- local dev:    Weaviate in Docker, no auth needed
- production:   Weaviate Cloud, API key auth
"""

# import libraries
from __future__ import annotations
import uuid as _uuid
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.data import DataObject
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
from app.retrieval.config import RetrievalConfig
from app.retrieval.models import SearchResult


def _deterministic_uuid(source_id: str, chunk_index: int) -> str:
    """
    this generates a UUID5 from "source_id" + "chunk_index"

    Parameters:
    source_id:    the document identifier, e.g., "gdpr", "fadp", "edpb_consent"
    chunk_index:  chunk position within the document

    Returns:
    UUID string, e.g., "a3b2c1d4-e5f6-5a7b-8c9d-0e1f2a3b4c5d"
    """
    return str(_uuid.uuid5(_uuid.NAMESPACE_DNS, f"graphlex:{source_id}:{chunk_index}"))


# collection schema, with one property per metadata field + the core content
# this enables filtered retrieval (e.g., "search only GDPR statutes")
_PROPERTIES = [
    # Core content
    Property(
        name="chunk_id", data_type=DataType.TEXT
    ),  # e.g., "gdpr:chapter-3:art-15:para-1"
    Property(
        name="text", data_type=DataType.TEXT
    ),  # this is the regulatory text shown to the LLM
    # Source identification
    # source_id values: "gdpr", "fadp", "edpb_consent", "edpb_article48",
    #                   "edpb_legitimate_interest", "fdpic_technical_measures"
    Property(name="source_id", data_type=DataType.TEXT),
    # instrument_type: "statute" (binding) or "guidance" (authoritative, non-binding)
    Property(name="instrument_type", data_type=DataType.TEXT),
    # jurisdiction: "EU" (GDPR/EDPB) or "CH" (FADP/FDPIC)
    Property(name="jurisdiction", data_type=DataType.TEXT),
    # the document structure hierarchy (Chapter > Section > Article > Paragraph)
    Property(name="chapter", data_type=DataType.TEXT),
    Property(name="section", data_type=DataType.TEXT),
    # article is used heavily for cross-referencing and graph expansion
    Property(name="article", data_type=DataType.TEXT),
    Property(name="paragraph", data_type=DataType.TEXT),
    # structural metadata
    Property(name="page_numbers", data_type=DataType.INT_ARRAY),
    # cross_references: regex detected article references (e.g., ["Article 6", "Article 9"])
    # fed into as "REFERENCES" relationships
    Property(name="cross_references", data_type=DataType.TEXT_ARRAY),
    Property(name="has_table", data_type=DataType.BOOL),
    Property(name="has_footnote", data_type=DataType.BOOL),
    # ingestion metadata
    Property(name="chunk_index", data_type=DataType.INT),
    # extractor_name: always "pymupdf" in production (because it was chosen over olmocr and Mistral)
    Property(name="extractor_name", data_type=DataType.TEXT),
]


class WeaviateStore:
    """
    this manages the RegulatoryChunk collection in Weaviate

    Responsibilities:
    1. connection : cloud or local Docker
    2. schema mgmt : create/delete collection
    3. ingestion : batch insert the chunks with the precomputed embeddings
    4. search : nearest neighbour lookup with optional metadata filters

    Usage:
    config = RetrievalConfig()
    store = WeaviateStore(config)
    store.connect()
    store.create_collection()
    store.ingest_chunks(chunks, vecs)
    results = store.search(query_vec)
    store.close()
    """

    def __init__(self, config: RetrievalConfig) -> None:
        """
        to initialise with config

        Parameters:
        config: RetrievalConfig with weaviate_url, weaviate_api_key,
                collection_name, and embedding_dimensions (3072)
        """
        self.config = config
        self._client: weaviate.WeaviateClient | None = None

    # connection

    def connect(self) -> None:
        """
        to connect to Weaviate Cloud (if the api_key set) or local Docker

        Raises:
        ConnectionError: if Weaviate is reachable but not ready
        """
        if self._client is not None:
            return

        if self.config.weaviate_api_key:
            # for production: Weaviate Cloud with API key auth
            self._client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self.config.weaviate_url,
                auth_credentials=Auth.api_key(self.config.weaviate_api_key),
            )
        else:
            # for development: local Docker (no auth required)
            self._client = weaviate.connect_to_local(
                host=self.config.weaviate_url.replace("http://", "").split(":")[0],
                port=(
                    int(self.config.weaviate_url.split(":")[-1])
                    if ":" in self.config.weaviate_url.rsplit("/", 1)[-1]
                    else 8080
                ),
            )

        if not self._client.is_ready():
            raise ConnectionError("Weaviate is not ready")
        print(f" Connected to Weaviate at {self.config.weaviate_url}")

    def close(self) -> None:
        """this closes the connection and releases resources"""
        if self._client:
            self._client.close()
            self._client = None

    @property
    def client(self) -> weaviate.WeaviateClient:
        """
        this returns the connected client, or raise if "connect()" hasn't been called

        Raises:
        RuntimeError: if "connect()" hasn't been called
        """
        if self._client is None:
            raise RuntimeError("Not connected - please call connect() first")
        return self._client

    # Collection management

    def collection_exists(self) -> bool:
        """this returns True if the RegulatoryChunk collection exists in Weaviate"""
        return self.client.collections.exists(self.config.collection_name)

    def create_collection(self, recreate: bool = False) -> None:
        """
        this creates the RegulatoryChunk collection with BYOV (bring your own vector) configuration

        Key settings:
        - Vectorizer: none(), embeddings are precomputed externally via OpenAI
        - HNSW and cosine distance: standard for semantic search over text embeddings

        Parameters:
        recreate: if true, drop and recreate the collection
        if false (which is default), skip if already exists
        """
        name = self.config.collection_name

        if self.collection_exists():
            if recreate:
                self.client.collections.delete(name)
                print(f" Deleted existing collection '{name}'")
            else:
                print(f" Collection '{name}' already exists")
                return

        self.client.collections.create(
            name=name,
            properties=_PROPERTIES,
            vectorizer_config=Configure.Vectorizer.none(),
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=weaviate.classes.config.VectorDistances.COSINE,
            ),
        )
        print(
            f" Created collection '{name}' (HNSW cosine, {self.config.embedding_dimensions}d)"
        )

    def delete_collection(self) -> None:
        """
        To delete the RegulatoryChunk collection and all of its data

        """
        name = self.config.collection_name
        if self.collection_exists():
            self.client.collections.delete(name)
            print(f" Deleted collection '{name}'")

    def count(self) -> int:
        """
        This returns the number of chunks in the collection

        It is used for post-ingestion verification and the CLI status reporting
        """
        collection = self.client.collections.get(self.config.collection_name)
        result = collection.aggregate.over_all(total_count=True)
        return result.total_count

    # Ingestion

    def ingest_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 200,
    ) -> int:
        """
        In order to batch insert chunks with their precomputed embedding vectors

        It uses Weaviate's batch API for efficiency.

        Parameters:
        chunks:      a list of chunk dicts matching the _PROPERTIES schema
        embeddings:  a list of 3072-float vectors, one per chunk
        batch_size:  objects per batch request

        Returns:
        the number of chunks submitted for insertion
        """
        collection = self.client.collections.get(self.config.collection_name)
        inserted = 0
        errors = 0

        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
                # ".get(key)"" or "" handles both missing keys and None values
                properties = {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source_id": chunk["source_id"],
                    "instrument_type": chunk["instrument_type"],
                    "jurisdiction": chunk["jurisdiction"],
                    "chapter": chunk.get("chapter") or "",
                    "section": chunk.get("section") or "",
                    "article": chunk.get("article") or "",
                    "paragraph": chunk.get("paragraph") or "",
                    "page_numbers": chunk.get("page_numbers", []),
                    "cross_references": chunk.get("cross_references", []),
                    "has_table": chunk.get("has_table", False),
                    "has_footnote": chunk.get("has_footnote", False),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "extractor_name": chunk.get("extractor_name", ""),
                }
                batch.add_object(
                    properties=properties,
                    vector=vector,
                    uuid=_deterministic_uuid(
                        chunk["source_id"], chunk.get("chunk_index", i)
                    ),
                )
                inserted += 1

                if batch.number_errors > 0:
                    errors = batch.number_errors

        if errors:
            failed = collection.batch.failed_objects
            print(f" Warning: {len(failed)} objects failed to insert")
            for obj in failed[:3]:
                print(f"    {obj.message}")

        return inserted

    # Search

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        source_ids: list[str] | None = None,
        jurisdictions: list[str] | None = None,
        instrument_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        For the vector similarity search with the optional metadata prefilters

        this finds the top_k chunks most semantically similar to the query vector

        Parameters:
        query_vector:     3072-float embedding of the user's question
        top_k:            the max results to return (default is 10)
        source_ids:       to filter by source document, e.g., ["gdpr"]
        jurisdictions:    to filter by jurisdiction, e.g., ["EU"], ["CH"]
        instrument_types: to filter by type, e.g., ["statute"], ["guidance"]

        Returns:
        a list of SearchResult instances sorted by relevance (highest score comes first)
        Every result includes chunk text, metadata, distance (cosine),
        and the score (1 - distance)
        """
        collection = self.client.collections.get(self.config.collection_name)

        # to build metadata filters
        filters = []
        if source_ids:
            filters.append(Filter.by_property("source_id").contains_any(source_ids))
        if jurisdictions:
            filters.append(
                Filter.by_property("jurisdiction").contains_any(jurisdictions)
            )
        if instrument_types:
            filters.append(
                Filter.by_property("instrument_type").contains_any(instrument_types)
            )

        # 0 filters -> no filtering, 1 -> use directly, 2+ -> AND combination
        combined_filter = None
        if len(filters) == 1:
            combined_filter = filters[0]
        elif len(filters) > 1:
            combined_filter = Filter.all_of(filters)

        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=combined_filter,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            props = obj.properties
            # default distance to 1.0 (the max uncertainty) if somehow absent
            distance = (
                obj.metadata.distance if obj.metadata.distance is not None else 1.0
            )
            # "or None" converts empty strings to None
            results.append(
                SearchResult(
                    chunk_id=props.get("chunk_id", ""),
                    text=props.get("text", ""),
                    source_id=props.get("source_id", ""),
                    instrument_type=props.get("instrument_type", ""),
                    jurisdiction=props.get("jurisdiction", ""),
                    article=props.get("article") or None,
                    section=props.get("section") or None,
                    paragraph=props.get("paragraph") or None,
                    cross_references=props.get("cross_references", []),
                    distance=round(distance, 4),
                    score=round(1.0 - distance, 4),
                )
            )

        return results
