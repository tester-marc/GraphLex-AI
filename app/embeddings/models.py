"""This script is for the data models for the pipeline for the embeddings comparison."""

# The classes used in this pipeline:
# RelevanceRule             : the regex based criterion for judging the chunk relevance
# TestQuery                 : a test question and its relevance rules
# ModelConfig               : the embedding model spec (i.e., the API name, dims, cost, provider)
# EmbeddingResult           : metrics per query and per model (P@5, P@10, MRR, cost, latency)
# EmbeddingComparisonResult : the aggregated metrics for one model across all the queries

# import libraries
from __future__ import annotations  # allows for type hint syntax
from dataclasses import (
    dataclass,
    field,
)  # field(default_factory=...) used for mutable defaults (list/dict)


@dataclass
class RelevanceRule:
    """This is a rule to determine chunk relevance to a test query."""

    level: int  # 2: highly relevant, 1: relevant, and unmatched chunks are 0
    source_id: (
        str | None
    )  # the document to match against (e.g., "gdpr", "fadp"), if None that means any source
    match_field: str  # the chunk metadata field to test: "article", "section", "paragraph", "text"
    pattern: str  # the regex applied with re.search() to the match_field value


@dataclass
class TestQuery:
    """This is for a test query with relevance judgments for evaluation of the retrieval."""

    query_id: str
    # a short unique ID used in the logs and the result files
    # the prefix shows the category: a = article_specific, c = conceptual,
    # x = cross_jurisdictional, n = negation_sensitive

    text: str  # this is the question string sent to the embedding API

    category: (
        str  # article_specific, conceptual, cross_jurisdictional, negation_sensitive
    )

    # an ordered list of rules evaluated top down per chunk, the first match wins
    relevance_rules: list[RelevanceRule] = field(default_factory=list)


@dataclass
class ModelConfig:
    """this is the configuration for an embedding model type"""

    name: str  # the API model identifier (e.g., "text-embedding-3-small", "kanon-2-embedder")

    # the unique display name used in the logs, the result tables, and cache file names
    # (data/output/embeddings/cache/{label}.json)
    label: str

    # the output vector length. for the OpenAI models, it is passed as the "dimensions"
    # API parameter in order to enable the MRL truncation
    dimensions: int

    # USD cost per 1 million tokens
    cost_per_million_tokens: float

    provider: str  # "openai": OpenAIEmbedder; "isaacus": KanonEmbedder


@dataclass
class EmbeddingResult:
    """This is the evaluation result per query per model"""

    model_label: str  # matches the ModelConfig.label
    dimensions: int  # this is copied from ModelConfig for self contained records
    query_id: str  # matches the TestQuery.query_id
    query_category: str  # this is copied from TestQuery.category and is used for aggregation per category

    precision_at_5: float  # the fraction of top5 ranked chunks that were relevant
    precision_at_10: float  # the fraction of top10 ranked chunks that were relevant
    reciprocal_rank: (
        float  # 1 / rank_of_first_relevant_chunk (it is 0.0 if none are found)
    )

    query_latency_ms: float  # round trip API and inference time for this single query
    query_tokens: int  # tokens consumed by API for this query
    cost_usd: float  # USD cost for embedding this query


@dataclass
class EmbeddingComparisonResult:
    """This is for the aggregated results for one model config"""

    model_label: str  # matches the ModelConfig.label
    dimensions: int

    avg_precision_at_5: float  # average P@5 across all the test queries
    avg_precision_at_10: float  # average P@10 across all the test queries
    avg_mrr: float  # Mean Reciprocal Rank (MRR) across all the test queries

    avg_query_latency_ms: float  # mean API latency per query
    total_corpus_tokens: int  # the tokens used to embed the whole corpus
    corpus_embed_latency_ms: (
        float  # this is the time to embed the entire corpus in one batch
    )
    avg_cost_per_query: float  # the mean USD cost per query

    # breakdown P@5/P@10/MRR per category
    # structure:
    # { "article_specific": {"precision_at_5": ..., "precision_at_10": ..., "mrr": ...}, ... }
    # This is for population of tables in section "4.4 Embedding and LLM Comparisons" in the final report
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)

    # these are the 12 per query EmbeddingResult objects that were averaged in order to produce this aggregate
    individual_results: list[EmbeddingResult] = field(default_factory=list)
