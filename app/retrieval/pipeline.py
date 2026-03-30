"""
For the Retrieval pipeline, i.e., load chunks, embed, ingest into Weaviate, and query

This orchestrates the Retrieval Layer and ties the following together:
- config.py                          : RetrievalConfig (Weaviate URL, model, paths)
- models.py                          : SearchResult (the data class for each search hit)
- weaviate_store.py                  : WeaviateStore (Weaviate CRUD)
- app/embeddings/openai_embedder.py  : OpenAIEmbedder (OpenAI API calls)

"RetrievalPipeline" provides 2 operations:
  1. Ingestion (offline)
  2. Query (online, per question from the user)

how this file is called:
- for the CLI: "python -m app.retrieval ingest" or query "..." via __main__.py
- Orchestration Layer: "retrieve_node" in "app/orchestration/nodes.py"
  calls ".connect()" then ".query()" for each user question
- Gradio UI: indirectly via the orchestration layer
"""

# import libraries
from __future__ import annotations
import json
import time
from pathlib import Path
from app.embeddings.openai_embedder import OpenAIEmbedder
from app.retrieval.config import RetrievalConfig
from app.retrieval.models import SearchResult
from app.retrieval.weaviate_store import WeaviateStore


class RetrievalPipeline:
    """
    for the end-to-end retrieval, ingestion from cached embeddings and query

    Usage for ingestion (run it once from CLI):
    pipeline = RetrievalPipeline()
    pipeline.ingest(recreate=False)

    Usage for querying (called per user question):
    pipeline = RetrievalPipeline()
    pipeline.connect()
    results = pipeline.query("What does GDPR Article 32 require?")
    print(pipeline.format_results(results))
    pipeline.close()
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        """
        In order to initialise the pipeline with optional custom configuration

        If config is None then a default RetrievalConfig is created, and reads
        Weaviate connection details from env variables (WEAVIATE_URL,
        WEAVIATE_API_KEY) and uses "text-embedding-3-large" at 3072 dimensions

        """
        self.config = config or RetrievalConfig()
        self.store = WeaviateStore(self.config)

        # text-embedding-3-large was ultimately chosen after the 5 model comparison
        self.embedder = OpenAIEmbedder(
            model=self.config.embedding_model,  # "text-embedding-3-large"
            dimensions=self.config.embedding_dimensions,  # 3072
        )

    # Ingestion

    def load_chunks(self) -> list[dict]:
        """
        this loads all PyMuPDF chunks from disk

        It reads every chunks.json under data/output/pymupdf/ (1 subfolder
        per document: edpb_article48, edpb_consent, edpb_legitimate_interest,
        fadp, fdpic_technical_measures, gdpr)

        It returns all chunks as a single flat list, that is processed in
        alphabetical folder order

        """
        chunks = []

        for source_dir in sorted(self.config.chunks_dir.iterdir()):
            chunks_file = source_dir / "chunks.json"
            if not chunks_file.exists():
                continue  # skip the folders without a chunks.json
            with open(chunks_file, encoding="utf-8") as f:
                doc_chunks = json.load(f)
            chunks.extend(doc_chunks)

        return chunks

    def load_cached_embeddings(self) -> tuple[list[list[float]], int]:
        """
        this loads precalculated text-embedding-3-large embeddings from cache

        It reads from "data/output/embeddings/cache/text-embedding-3-large.json"
        (the path set by RetrievalConfig.embedding_cache_path)

        Returns (embeddings, chunk_count)
        Raises FileNotFoundError if the cache doesn't exist
        """
        cache_path = self.config.embedding_cache_path

        if not cache_path.exists():
            raise FileNotFoundError(
                f"No cached embeddings at {cache_path}. "
                "Run python -m app.embeddings --compare first."
            )

        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)

        embeddings = cached["embeddings"]
        chunk_count = cached["chunk_count"]
        print(f" Loaded {chunk_count} cached embeddings ({cached['dimensions']}d)")
        return embeddings, chunk_count

    def ingest(self, recreate: bool = False) -> int:
        """
        The full ingestion pipeline: load chunks and cached embeddings -> Weaviate

        Steps:
        1. loads 817 chunk dicts from PyMuPDF output on disk
        2. loads 817 precomputed embedding vectors from cache
        3. verifies that the counts match (chunks and embeddings must align)
        4. connects to Weaviate
        5. creates the RegulatoryChunk collection
        6. skips if data already exists and recreate=False
        7. batch inserts all the chunks with their vectors

        Parameters:
        recreate : bool, default False
        to delete and recreate the collection before inserting

        Returns:
        int
        the final object count in the Weaviate collection

        Raises:
        ValueError
        if the chunk count and cached embedding count do not match
        """
        print("\n Retrieval Layer: Ingestion ")

        chunks = self.load_chunks()
        print(f" Loaded {len(chunks)} chunks from PyMuPDF output")

        embeddings, cached_count = self.load_cached_embeddings()

        if len(chunks) != cached_count:
            raise ValueError(
                f"Chunk count mismatch: {len(chunks)} chunks vs. {cached_count} cached embeddings. "
                "Rerun embedding comparison to refresh cache."
            )

        self.store.connect()
        self.store.create_collection(recreate=recreate)

        # skips if data already exists and recreate=False
        existing = self.store.count()
        if existing > 0 and not recreate:
            print(f" Collection already has {existing} objects - skipping ingestion")
            print(" Use --recreate to force reingestion")
            return existing

        start = time.perf_counter()
        # ingest_chunks uses deterministic UUIDs (source_id and chunk_index)
        inserted = self.store.ingest_chunks(chunks, embeddings)
        elapsed = time.perf_counter() - start

        final_count = self.store.count()
        print(f" Ingested {inserted} chunks in {elapsed:.1f}s")
        print(f" Collection now has {final_count} objects")
        return final_count

    # Query

    def query(
        self,
        text: str,
        top_k: int | None = None,
        source_ids: list[str] | None = None,
        jurisdictions: list[str] | None = None,
        instrument_types: list[str] | None = None,
    ) -> list[SearchResult]:
        """
        This embeds a text query and searches Weaviate

        Steps:
        1. embeds the query with text-embedding-3-large (3072d vector)
        2. runs approx. nearest neighbour search in Weaviate (HNSW /
           cosine similarity) against all stored chunk vectors
        3. returns the top-k most similar chunks as SearchResult objects

        Parameters:
        text : str
        the natural language question (or Whisper transcript from Layer 2)
        top_k : int or None
        the results to return
        source_ids : list[str] or None
        to filter by document (e.g., ["gdpr"] or ["gdpr", "fadp"])
        Automatically populated by the interpret node when the user names a source
        jurisdictions : list[str] or None
        this filters by jurisdiction ("EU", "CH"), None searches both
        instrument_types : list[str] or None
        this filters by authority type ("statute", "guidance", "commentary"),
        None searches all types

        Returns:
        list[SearchResult]
        The results are sorted by relevance (the highest score comes first)
        """
        if top_k is None:
            top_k = self.config.default_top_k

        query_vector, tokens, embed_ms = self.embedder.embed_query(text)

        results = self.store.search(
            query_vector=query_vector,
            top_k=top_k,
            source_ids=source_ids,
            jurisdictions=jurisdictions,
            instrument_types=instrument_types,
        )

        return results

    # Formatting

    @staticmethod
    def format_results(results: list[SearchResult]) -> str:
        """
        to format search results as a readable string for the terminal output

        every result shows rank, source, location, authority / jurisdiction,
        similarity score, a 120char text preview, and cross-references

        This is a @staticmethod, it does not depend on the pipeline state
        """
        if not results:
            return " No results found."

        lines = []

        for i, r in enumerate(results, 1):
            lines.append(
                f"  {i}. [{r.source_id}] {r.location_label} "
                f"({r.authority_label}, {r.jurisdiction}) "
                f"score={r.score:.3f}"
            )

            preview = r.text[:120].replace("\n", " ")
            if len(r.text) > 120:
                preview += "..."
            lines.append(f"    {preview}")

            # to cap the cross-references at 5 in order to avoid long lines
            if r.cross_references:
                lines.append(f"    refs: {', '.join(r.cross_references[:5])}")

            lines.append("")

        return "\n".join(lines)

    # lifecycle

    def connect(self) -> None:
        """
        this opens the connection to Weaviate

        It detects cloud vs. local Docker from config.
        """
        self.store.connect()

    def close(self) -> None:
        """
        This closes the Weaviate connection and releases resources

        after closing pipeline can be reconnected by calling "connect()" again
        """
        self.store.close()
