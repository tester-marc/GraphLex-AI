"""
this is the configuration for the Retrieval Layer

this centralizes all the settings that the retrieval layer needs to connect to
Weaviate (vector database) and perform the semantic search over the regulatory chunks, i.e. :
- where Weaviate is running (local Docker or in the cloud)
- which embedding model to use
- how many results to return by default
- and where to find chunk data and cached embeddings on disk
"""

# import libraries
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RetrievalConfig:
    """for the Weaviate connection and retrieval settings"""

    # Weaviate connection

    # Local: "http://localhost:8080" (Docker, and no auth needed)
    # Cloud: "https://fvipoltotkwni9w7ps8b4a.c0.europe-west3.gcp.weaviate.cloud"
    # populated from WEAVIATE_URL env variable in "__post_init__" if it's not set
    weaviate_url: str = ""

    # this is required for Weaviate Cloud only, it comes from the WEAVIATE_API_KEY env var
    weaviate_api_key: str = ""

    # Collection settings

    # for a single Weaviate collection that holds all the chunks
    # the properties per object are: chunk_id, text, source_id, jurisdiction, article, etc.
    collection_name: str = "RegulatoryChunk"

    # Embedding model

    # this was selected after a comparison (see report text)
    embedding_model: str = "text-embedding-3-large"

    # "text-embedding-3-large" produces 3072-dim vectors by default
    # Weaviate uses cosine similarity over the vectors for semantic search
    embedding_dimensions: int = 3072

    # Retrieval defaults

    # this is the number of top-k chunks returned per query
    default_top_k: int = 10

    # Paths

    # for the project root, resolved from this file's location:
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    def __post_init__(self) -> None:
        """this fills the Weaviate URL and API key from the environment variables if not provided"""
        if not self.weaviate_url:
            self.weaviate_url = os.getenv("WEAVIATE_URL", "")
        if not self.weaviate_api_key:
            self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY", "")

    # Calculated paths

    @property
    def chunks_dir(self) -> Path:
        """
        For the directory that contains the PyMuPDF extraction outputs

        Structure: "data/output/pymupdf/<source>/chunks.json"
        6 documents in total and 817 chunks in total
        """
        return self.project_root / "data" / "output" / "pymupdf"

    @property
    def embedding_cache_path(self) -> Path:
        """
        the path to the cached embedding vectors for the model chosen

        It resolves to: "data/output/embeddings/cache/<embedding_model>.json"
        Format is { "model": ..., "dimensions": ..., "chunk_count": ..., "embeddings": [[...], ...] }

        The precalculated vectors are loaded at ingestion time and then sent directly to Weaviate
        """
        return (
            self.project_root
            / "data"
            / "output"
            / "embeddings"
            / "cache"
            / f"{self.embedding_model}.json"
        )

    # validation

    def is_configured(self) -> bool:
        """this returns true if the Weaviate URL is set"""
        return bool(self.weaviate_url)
