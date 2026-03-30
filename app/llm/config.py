"""
LLM Comparison Configuration:

This is for the LLM comparison pipeline:
- the models to compare (IDs, display names, pricing)
- the system prompt (shared across both models for a fair test)
- benchmark queries (8 questions with ground truth)
- uncertainty detection phrases
- path helpers (LLMConfig)

Imported by:  app/llm/comparison.py, app/llm/__main__.py
It uses:      app/llm/models.py (BenchmarkQuery, LLMModelConfig)

Final result: Qwen3-Next 80B-A3B was selected (for details see final report)
"""

# import libraries
from __future__ import annotations
from pathlib import Path
from app.llm.models import BenchmarkQuery, LLMModelConfig

# section 1: models to be compared
# Together AI charges separately for input tokens (prompt) and output tokens
# (the generated response)
# Qwen3-Next uses a MoE architecture: 80B total params, but only 3B are active
# per token, this making it faster and cheaper per query

LLM_MODELS: list[LLMModelConfig] = [
    # Llama 3.3 70B Instruct Turbo
    LLMModelConfig(
        name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        label="llama-3.3-70b",
        cost_per_million_input=0.88,
        cost_per_million_output=0.88,
    ),
    # Qwen3-Next 80B-A3B
    LLMModelConfig(
        name="Qwen/Qwen3-Next-80B-A3B-Instruct",
        label="qwen3-80b-a3b",
        cost_per_million_input=0.15,
        cost_per_million_output=1.50,
    ),
]

# section 2: system prompt
# the system prompt is shared across both models to ensure a fair comparison

SYSTEM_PROMPT = """\
You are a regulatory compliance assistant specialising in Swiss and EU data \
protection law (GDPR and Swiss FADP).

Rules:
1. Answer ONLY based on the provided context passages. Do not use prior knowledge.
2. Cite specific article numbers (e.g., "GDPR Article 17", "FADP Art. 25") for every claim.
3. If the provided context does not contain sufficient evidence to answer the \
question, explicitly state: "Insufficient evidence in the provided context to \
answer this question." and explain what information is missing.
4. Do not fabricate or hallucinate provisions, article numbers, or regulatory content.
5. Indicate the source document for each citation (GDPR, FADP, EDPB guidelines, FDPIC guide).
6. Keep your answer concise (200-400 words) and focused on the regulatory provisions."""

# section 3: benchmark queries
#
# well-evidenced (w1-w5): the corpus contains sufficient info, the model should answer
# confidently and cite the correct articles
#
# under-evidenced (u1-u3): the corpus lacks the required info, the model should refuse
# and flag the gap rather than hallucinate!
#

BENCHMARK_QUERIES: list[BenchmarkQuery] = [
    # well-evidenced
    # w1: single well-defined article (right to erasure)
    BenchmarkQuery(
        query_id="w1",
        text="What does GDPR Article 17 provide regarding the right to erasure?",
        category="well_evidenced",
        expected_citations=["Article 17"],
        evidence_sufficient=True,
    ),
    # w2: multiple related articles (consent validity -> Art. 7 + Art. 6(1)(a))
    BenchmarkQuery(
        query_id="w2",
        text="Under what conditions is consent valid for processing personal data under the GDPR?",
        category="well_evidenced",
        expected_citations=["Article 7", "Article 6"],
        evidence_sufficient=True,
    ),
    # w3: Swiss source (FDPIC guide, FADP Art. 8 : data security)
    BenchmarkQuery(
        query_id="w3",
        text="What technical and organisational measures does the FDPIC recommend for protecting personal data?",
        category="well_evidenced",
        expected_citations=["Article 8"],
        evidence_sufficient=True,
    ),
    # w4: single high importance article (all 6 lawful bases)
    BenchmarkQuery(
        query_id="w4",
        text="What are the lawful bases for processing personal data under GDPR Article 6?",
        category="well_evidenced",
        expected_citations=["Article 6"],
        evidence_sufficient=True,
    ),
    # w5: multi-article chain (Art. 44 general principle, 45 adequacy, 46 safeguards)
    BenchmarkQuery(
        query_id="w5",
        text="When can personal data be transferred to a third country under the GDPR?",
        category="well_evidenced",
        expected_citations=["Article 44", "Article 45", "Article 46"],
        evidence_sufficient=True,
    ),
    # under-evidenced
    # u1: enforcement data (FDPIC penalty amounts/cases) not in corpus
    BenchmarkQuery(
        query_id="u1",
        text="What specific penalties has the Swiss FDPIC imposed for FADP violations in 2024?",
        category="under_evidenced",
        expected_citations=[],
        evidence_sufficient=False,
    ),
    # u2: AI training data: no specific provisions in GDPR (adopted 2016)
    BenchmarkQuery(
        query_id="u2",
        text="How does the GDPR specifically regulate the use of personal data for training artificial intelligence models?",
        category="under_evidenced",
        expected_citations=[],
        evidence_sufficient=False,
    ),
    # u3: EDPB Art. 22 profiling guidelines not among the 3 EDPB docs in corpus
    BenchmarkQuery(
        query_id="u3",
        text="What are the EDPB's guidelines on automated individual decision-making and profiling under Article 22?",
        category="under_evidenced",
        expected_citations=[],
        evidence_sufficient=False,
    ),
]

# section 4: uncertainty detection
# if any phrase appears in a response, it is classified
# as "expressed uncertainty"

UNCERTAINTY_PHRASES: list[str] = [
    # explicit refusals (closest to the system prompt wording)
    "insufficient evidence",
    "not enough information",
    "cannot determine",
    "cannot be determined",
    "unable to answer",
    "unable to provide",
    # nothing relevant found
    "no specific provision",
    "not addressed",
    "not specifically address",
    "not contain sufficient",
    "does not contain",
    # topic is outside context scope
    "beyond the scope",
    "not available in",
    "not included in",
    # missing from context
    "no mention",
    "not mentioned",
    "context does not",
    "provided context does not",
    "no information",
    "cannot find",
    "not found in",
    # weaker hedges
    "does not specifically",
    "no direct",
    "not directly",
]


# section 5: pipeline config path helpers
#
# LLMConfig is a class so project_root can be injected
# in the tests (e.g., LLMConfig(project_root=Path("/tmp/test")))


class LLMConfig:
    """for path helpers and configuration for the LLM comparison"""

    def __init__(self, project_root: Path | None = None) -> None:
        """
        Args:
        project_root: the project root directory
        """
        self.project_root = project_root or Path(__file__).resolve().parents[2]

    @property
    def chunks_dir(self) -> Path:
        """data/output/pymupdf/, one subdirectory per source document"""
        return self.project_root / "data" / "output" / "pymupdf"

    @property
    def embeddings_cache_dir(self) -> Path:
        """data/output/embeddings/cache/, the cached corpus embeddings"""
        return self.project_root / "data" / "output" / "embeddings" / "cache"

    @property
    def output_dir(self) -> Path:
        """data/output/llm/, the comparison results directory"""
        return self.project_root / "data" / "output" / "llm"

    def results_path(self) -> Path:
        """data/output/llm/comparison_results.json: aggregated + per-query results"""
        return self.output_dir / "comparison_results.json"

    def ensure_dirs(self) -> None:
        """this creates the output directory if it doesn't exist"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
