"""This is the pipeline for the embedding model comparison.
It is the package entry point for "app.embeddings", which is part of Layer 3 -
the Retrieval layer - of the GraphLex AI application.

This file here designates this directory as a Python package. It defines the public API
so that external code can use "from app.embeddings import EmbeddingConfig"
instead of it importing from full internal module path.

Package content (8 files in total):
  __init__.py: this file which is the package entry point and its public API
  __main__.py: the CLI entry point ("python -m app.embeddings")
  config.py: for model definitions, the 12 test queries, and the path helpers
  models.py: for the data classes (ModelConfig, TestQuery, EmbeddingResult, and so forth)
  comparison.py: this orchestrates the full evaluation
  metrics.py: for the precision@k & reciprocal rank calculations
  openai_embedder.py: the API wrapper for OpenAI text-embedding-3-small/large
  kanon_embedder.py: for the Kanon-2-embedder via the Isaacus REST API wrapper

What did this package do? It compared 5 embedding model configs across 12 test queries
throughout 4 different categories (article-specific, conceptual, cross-jurisdictional,
negation-sensitive). The model "text-embedding-3-large" (at 3072 dimensions)
was chosen as the embedder for production and its cached embeddings are used by the
Retrieval Layer (app/retrieval/) for the Weaviate vector search.
"""

# this is the path resolver for the pipeline:
# it locates the PyMuPDF chunks (in: data/output/pymupdf/),
# the cached embeddings (in: data/output/embeddings/cache/),
# and the comparison results (in: data/output/embeddings/comparison_results.json)
from app.embeddings.config import EmbeddingConfig

# EmbeddingResult: the per query and per model metrics (60 in total for 12 queries x 5 models)
# EmbeddingComparisonResult: the aggregated metrics per model (5 in total, i.e., 1 per model)
# These are listed in the tables 8 and 9 for the embedding comparisons in section
# "4.4 Embedding and LLM Comparisons" of the final report.
from app.embeddings.models import EmbeddingComparisonResult, EmbeddingResult

# Public API:
# the internal classes (OpenAIEmbedder, KanonEmbedder, EmbeddingComparisonHarness)
# are intentionally excluded here, they can be imported directly from their modules if required
__all__ = ["EmbeddingConfig", "EmbeddingResult", "EmbeddingComparisonResult"]
