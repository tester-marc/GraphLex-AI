"""
LLM Comparison Data Models:

This contains 5 dataclasses representing the comparison pipeline stages:
LLMModelConfig        : the model identity and pricing
BenchmarkQuery        : test question with ground truth
LLMResponse           : the raw API response and metadata
LLMEvaluationResult   : per query graded metrics
LLMComparisonResult   : the aggregated results per model

Imported by: app/llm/config.py, app/llm/comparison.py, app/llm/__main__.py
"""

# import libraries
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class LLMModelConfig:
    """the configuration for an LLM to be tested (model ID, display name, pricing)"""

    name: str  # Together AI model ID (e.g., "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    label: str  # a short display label (e.g., "llama-3.3-70b")
    cost_per_million_input: float  # USD cost per 1 million input tokens
    cost_per_million_output: float  # USD cost per 1 million output tokens


@dataclass
class BenchmarkQuery:
    """
    This is for a benchmark query with ground truth annotations

    Well-evidenced queries (w1-w5): the context contains the answer, the model should cite and respond
    Under-evidenced queries (u1-u3): the context lacks the answer, the model should express uncertainty
    """

    query_id: str  # the unique ID, e.g., "w1" or "u3"
    text: str  # the question text sent to the LLM
    category: str  # "well_evidenced" or "under_evidenced"
    expected_citations: list[str]  # articles that should appear in a correct answer
    evidence_sufficient: bool  # True : model should answer, False : model should refuse


@dataclass
class LLMResponse:
    """for the raw API response before evaluation"""

    model_label: str  # the display label of the model
    query_id: str  # which benchmark query this answers
    query_category: str  # this is copied from BenchmarkQuery
    response_text: str  # the full LLM response text
    latency_ms: float  # API call duration (network and inference)
    input_tokens: int  # the tokens in system + user prompt (reported by API)
    output_tokens: int  # the tokens in the generated response (reported by API)
    cost_usd: float  # (input_tokens * cost_per_M_in + output_tokens * cost_per_M_out) / 1_000_000


@dataclass
class LLMEvaluationResult:
    """
    this is the evaluated result for one model on one query

    the citation metrics are computed over well-evidenced queries only
    the calibration metrics apply to all queries
    """

    # identifiers
    model_label: str
    query_id: str
    query_category: str
    evidence_sufficient: bool  # copied from BenchmarkQuery

    # citation metrics
    citations_found: list[str]  # the article refs extracted from the response
    citations_expected: list[str]  # the ground truth article refs (from BenchmarkQuery)
    citation_precision: float  # |found ∩ expected| / |found|
    citation_recall: float  # |found ∩ expected| / |expected|

    # calibration metrics
    expressed_uncertainty: bool  # this is true if model hedged or refused to answer
    calibration_correct: (
        bool  # this is true if model's confidence matched the evidence level
    )

    # performance metrics
    latency_ms: float
    cost_usd: float
    response_text: str


@dataclass
class LLMComparisonResult:
    """
    the aggregated results for one model across all benchmark queries

    averaging subsets:
    citation metrics               : well-evidenced queries only (5)
    calibration / latency / cost   : all queries (8)
    insufficient_evidence_rate     : under-evidenced queries only (3)
    false_refusal_rate             : well-evidenced queries only (5)
    """

    model_label: str

    avg_citation_precision: float  # average over well-evidenced queries
    avg_citation_recall: float  # average over well-evidenced queries
    calibration_accuracy: float  # the fraction of all queries with correct calibration
    insufficient_evidence_rate: (
        float  # under-evidenced: the fraction where model correctly abstained
    )
    false_refusal_rate: (
        float  # well-evidenced: the fraction where model incorrectly refused
    )
    avg_latency_ms: float  # average over all queries
    avg_cost_per_query: float  # average over all queries

    # per query detail
    individual_results: list[LLMEvaluationResult] = field(default_factory=list)
