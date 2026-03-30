"""
Retrieval Layer: this is the Weaviate vector store that was embedded with text-embedding-3-large

This is the entry point for the Retrieval Layer

It re-exports the 4 public classes so callers can write the below instead of importing from specific modules

"from app.retrieval import RetrievalConfig, WeaviateStore"

Classes contained:
RetrievalConfig    : the settings dataclass (Weaviate URL, API key, collection name,
                     the embedding model, dimensions), it reads from .env
SearchResult       : one search hit, i.e., chunk text, metadata, distance / score
                     properties: authority_label, location_label
WeaviateStore      : Weaviate operations: connect, create collection,
                     batch ingest, vector similarity search with the metadata filters
RetrievalPipeline  : this is the orchestrator, it loads chunks and cached embeddings,
                     ingests into Weaviate, exposes "query()"
"""

# import libraries
from app.retrieval.config import RetrievalConfig
from app.retrieval.models import SearchResult
from app.retrieval.weaviate_store import WeaviateStore
from app.retrieval.pipeline import RetrievalPipeline

# controls "from app.retrieval import *" and the doc tooling
__all__ = [
    "RetrievalConfig",
    "SearchResult",
    "WeaviateStore",
    "RetrievalPipeline",
]
