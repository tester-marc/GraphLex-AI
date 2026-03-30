"""
This is the configuration script for the pipeline for the embedding comparison, i.e., comparison.py.

It defines:
- EMBEDDING_MODELS: the 5 models to benchmark
- TEST_QUERIES: the 12 test queries throughout the four difficulty categories,
  each one with RelevanceRule objects which map chunks to the relevance levels
  (2 = highly relevant, 1 = relevant, 0 = default resp. no match)
- EmbeddingConfig: the file system paths for the chunks, cache, and the results

GraphLex AI ingests six Swiss/EU data protection documents (GDPR, FADP, as well
as 4 guidance docs) and retrieves the relevant passages through embedding similarity.
Ultimately, the model "text-embedding-3-large" was chosen for production use.

The relevance rules are checked from top down and the first match wins. The Level 2
rules (the exact article matches) are listed before the Level 1 rules (which are
conceptual matches), i.e., the more specific rules take the priority.
This avoids having to manually label all the 817 chunks x 12 queries (9,804 pairs).
"""

# import libraries
from __future__ import annotations
from pathlib import Path

# ModelConfig (name, label, dimensions, cost_per_million_tokens, provider)
# RelevanceRule (level, source_id, match_field, pattern)
# TestQuery (query_id, text, category, relevance_rules)
from app.embeddings.models import ModelConfig, RelevanceRule, TestQuery

# Models to be compared:
#
# Three "text-embedding-3-small" variants test Matryoshka Representation
# Learning (MRL). "text-embedding-3-large" and "kanon-2-embedder" (which is a legal
# domain model that was trained on European legal texts) complete the comparison.
# The result was that "text-embedding-3-large" was chosen for production (because of best P@5 and MRR).

EMBEDDING_MODELS: list[ModelConfig] = [
    # "text-embedding-3-small" at native, 1/3, and 1/6 of its 1536 dimensional output
    ModelConfig(
        "text-embedding-3-small", "text-embedding-3-small", 1536, 0.02, "openai"
    ),
    ModelConfig(
        "text-embedding-3-small", "text-embedding-3-small-512d", 512, 0.02, "openai"
    ),
    ModelConfig(
        "text-embedding-3-small", "text-embedding-3-small-256d", 256, 0.02, "openai"
    ),
    # "text-embedding-3-large" at 3072 dimensions
    ModelConfig(
        "text-embedding-3-large", "text-embedding-3-large", 3072, 0.13, "openai"
    ),
    # "kanon-2-embedder" which is legal domain specific
    ModelConfig("kanon-2-embedder", "kanon-2-embedder", 1792, 0.35, "isaacus"),
]

# The test queries with relevance criteria judged manually
#
# 12 queries across 4 categories (3 each):
# article_specific     - for the retrieval of a named legal article
# conceptual           - for semantic retrieval across multiple documents/articles
# cross_jurisdictional - results from both GDPR and FADP
# negation_sensitive   - for queries about exceptions, exemptions, and exclusions
#
# source_id values: "gdpr", "fadp", "edpb_consent", "edpb_article48",
#                   "edpb_legitimate_interest", "fdpic_technical_measures"
# match_field values: "article", "text", "section", "paragraph"

TEST_QUERIES: list[TestQuery] = [
    # Category 1: article-specific
    # a1: GDPR Article 17 - right to erasure ("right to be forgotten")
    TestQuery(
        query_id="a1",
        text="What does GDPR Article 17 provide regarding the right to erasure?",
        category="article_specific",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 17"),
            RelevanceRule(
                1, "gdpr", "text", r"(?i)\b(erasure|right to be forgotten)\b"
            ),
        ],
    ),
    # a2: GDPR Article 35 - Data Protection Impact Assessment (DPIA)
    TestQuery(
        query_id="a2",
        text="What are the requirements for data protection impact assessments under GDPR Article 35?",
        category="article_specific",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 35"),
            RelevanceRule(1, "gdpr", "text", r"(?i)\b(impact assessment|DPIA)\b"),
            RelevanceRule(1, "fdpic_technical_measures", "text", r"(?i)\bDPIA\b"),
        ],
    ),
    # a3: FADP Article 25 - right of access (note: "Art." vs. GDPR's "Article")
    TestQuery(
        query_id="a3",
        text="What does FADP Article 25 say about the right of access?",
        category="article_specific",
        relevance_rules=[
            RelevanceRule(2, "fadp", "article", r"Art\. 25"),  # literal dot
            RelevanceRule(
                1, "fadp", "text", r"(?i)\b(right of access|right to access)\b"
            ),
            RelevanceRule(
                1, "fdpic_technical_measures", "section", r"(?i)Right of access"
            ),
        ],
    ),
    # Category 2: conceptual
    # c1: Crossborder data transfers - GDPR Chapter V (Arts. 44-49) and EDPB Art. 48 guidance
    TestQuery(
        query_id="c1",
        text="What obligations exist for transferring personal data to third countries?",
        category="conceptual",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 4[4-9]"),
            RelevanceRule(2, "edpb_article48", "text", r"(?i)transfer"),
            RelevanceRule(1, "gdpr", "text", r"(?i)\btransfer\b.*\bthird countr"),
            RelevanceRule(1, "fadp", "text", r"(?i)\b(transfer|abroad|foreign)\b"),
        ],
    ),
    # c2: Valid consent - GDPR Arts. 6+7 and EDPB consent guidelines
    TestQuery(
        query_id="c2",
        text="How must valid consent be obtained for processing personal data?",
        category="conceptual",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 7"),
            RelevanceRule(2, "edpb_consent", "text", r"(?i)\bvalid consent\b"),
            RelevanceRule(1, "edpb_consent", "text", r"(?i)\bconsent\b"),
            RelevanceRule(1, "gdpr", "article", r"Article 6"),
            RelevanceRule(1, "fadp", "text", r"(?i)\bconsent\b"),
        ],
    ),
    # c3: Technical and organisational measures - GDPR Art. 32 and FDPIC guide
    TestQuery(
        query_id="c3",
        text="What technical and organisational measures are required to protect personal data?",
        category="conceptual",
        relevance_rules=[
            RelevanceRule(
                2,
                "fdpic_technical_measures",
                "text",
                r"(?i)\b(technical|organisational) measures\b",
            ),
            RelevanceRule(2, "gdpr", "article", r"Article 32"),
            RelevanceRule(
                1, "fdpic_technical_measures", "text", r"(?i)\b(security|protect)\b"
            ),
            RelevanceRule(1, "gdpr", "text", r"(?i)\btechnical measures\b"),
        ],
    ),
    # Category 3: cross-jurisdictional
    # x1: definition of "personal data" - GDPR Art. 4(1) vs. FADP Art. 5(a)
    TestQuery(
        query_id="x1",
        text="How do Swiss and EU data protection laws define personal data?",
        category="cross_jurisdictional",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 4"),
            RelevanceRule(2, "fadp", "article", r"Art\. 5"),
            RelevanceRule(1, "gdpr", "text", r"(?i)\bpersonal data\b.*\bmeans\b"),
            RelevanceRule(1, "fadp", "text", r"(?i)\bpersonal data\b.*\bmeans\b"),
        ],
    ),
    # x2: data subject rights - GDPR Arts. 15-22 vs. FADP Arts. 25-29
    TestQuery(
        query_id="x2",
        text="What are the data subject rights under both Swiss and EU data protection law?",
        category="cross_jurisdictional",
        relevance_rules=[
            # Art. 19 was left out: it's a notification obligation not a data subject right per se
            RelevanceRule(2, "gdpr", "article", r"Article (15|16|17|18|20|21|22)"),
            RelevanceRule(2, "fadp", "article", r"Art\. (25|26|27|28|29)"),
            RelevanceRule(1, "gdpr", "text", r"(?i)\bdata subject\b.*\bright\b"),
            RelevanceRule(
                1, "fadp", "text", r"(?i)\bright\b.*(access|rectif|eras|portab)"
            ),
            RelevanceRule(1, "fdpic_technical_measures", "section", r"(?i)Right"),
        ],
    ),
    # x3: crossborder transfer rules compared - GDPR Ch. V vs. FADP Arts. 16-17
    TestQuery(
        query_id="x3",
        text="How do cross-border data transfer rules differ between GDPR and FADP?",
        category="cross_jurisdictional",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 4[4-9]"),
            RelevanceRule(2, "fadp", "article", r"Art\. 1[6-7]"),
            RelevanceRule(2, "edpb_article48", "text", r"(?i)transfer"),
            RelevanceRule(1, "gdpr", "text", r"(?i)\btransfer\b"),
            RelevanceRule(1, "fadp", "text", r"(?i)\b(transfer|abroad)\b"),
        ],
    ),
    # Category 4: negation-sensitive
    # n1: DPIA exemptions - Art. 35 para. 10 lists processing activities exempt
    TestQuery(
        query_id="n1",
        text="Which processing activities are exempt from requiring a data protection impact assessment?",
        category="negation_sensitive",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 35"),
            RelevanceRule(
                2, "fdpic_technical_measures", "section", r"(?i)Exception.*DPIA"
            ),
            RelevanceRule(
                1,
                "gdpr",
                "text",
                r"(?i)\b(exempt|not required|shall not apply)\b.*\b(impact assessment|DPIA)\b",
            ),
            RelevanceRule(1, "fdpic_technical_measures", "text", r"(?i)\bDPIA\b"),
        ],
    ),
    # n2: processing without consent - GDPR Art. 6 (five non-consent bases),
    #     FADP Arts. 30-31, and the EDPB legitimate interest guidelines
    TestQuery(
        query_id="n2",
        text="When is data processing permitted without the data subject's consent?",
        category="negation_sensitive",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 6"),
            RelevanceRule(2, "fadp", "article", r"Art\. (30|31)"),
            RelevanceRule(
                2, "edpb_legitimate_interest", "text", r"(?i)\blegitimate interest\b"
            ),
            RelevanceRule(1, "gdpr", "text", r"(?i)\bwithout.*consent\b"),
            # balancing test = core of Art. 6(1)(f) legitimate interest basis
            RelevanceRule(1, "edpb_legitimate_interest", "text", r"(?i)\bbalancing\b"),
        ],
    ),
    # n3: GDPR scope exclusions - Arts. 2+3 define what falls outside the GDPR
    #     (e.g., household activities, national security, law enforcement, and so forth)
    TestQuery(
        query_id="n3",
        text="What personal data processing falls outside the scope of the GDPR?",
        category="negation_sensitive",
        relevance_rules=[
            RelevanceRule(2, "gdpr", "article", r"Article 2"),
            RelevanceRule(2, "gdpr", "article", r"Article 3"),
            RelevanceRule(
                1,
                "gdpr",
                "text",
                r"(?i)\b(not apply|outside the scope|does not cover|excluded)\b",
            ),
            # the recitals 14-27 discuss the scope in detail
            RelevanceRule(1, "gdpr", "paragraph", r"Recital (1[4-9]|2[0-7])"),
        ],
    ),
]


# Pipeline configuration:
#
# Layout of directory:
# data/output/
# pymupdf/                : chunks_dir (1 subdirectory + chunks.json per document)
# embeddings/             : output_dir
# cache/                  : cache_dir (1 JSON per model)
# comparison_results.json : results_path


class EmbeddingConfig:
    """
    This class contains path helpers for the pipleine for the embedding comparison.

    All paths derive from a single project_root in order to make sure
    the configuration is portable. If the project_root is not provided,
    it's automatically detected from the location of this file.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path(__file__).resolve().parents[2]

    @property
    def chunks_dir(self) -> Path:
        """These are the pre-extracted PyMuPDF chunks (input corpus)."""
        return self.project_root / "data" / "output" / "pymupdf"

    @property
    def output_dir(self) -> Path:
        """The root output directory for all the comparison artefacts."""
        return self.project_root / "data" / "output" / "embeddings"

    @property
    def cache_dir(self) -> Path:
        """For the cached corpus embeddings, this avoids calling the API again on repeated runs."""
        return self.output_dir / "cache"

    def cache_path(self, model_label: str) -> Path:
        """
        This is for the cache file path for a given label of a model.
        """
        safe = model_label.replace("/", "_")
        return self.cache_dir / f"{safe}.json"

    def results_path(self) -> Path:
        """For the final comparison results JSON (individual and aggregated rows)."""
        return self.output_dir / "comparison_results.json"

    def ensure_dirs(self) -> None:
        """This creates the output and cache directories if they don't yet exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
