"""
These are the data models for the Graph Layer.

This script defines Python dataclasses that represent entities
and structures stored in the Neo4j graph

Knowledge graph structure:

The graph is organized around the legal instruments with their structure.
- an instrument contains multiple articles, which can reference other articles.
- each article can define specific definitions and impose obligations.
- additionally, relationships exist across the legal frameworks:
  an article from FADP may be marked as equivalent to a corresponding
  article in GDPR

Types of Nodes:
- ArticleNode     : an individual article or provision (e.g., GDPR Article 5)
- DefinitionNode  : legal term defined in a definitions article
- ObligationNode  : an obligation,right,prohibition,permission from an article

Instrument nodes are created directly as dicts from "DOCUMENT_REGISTRY"

Types of relationships (which are modelled by GraphRelationship):
- CONTAINS
- REFERENCES
- DEFINES
- IMPOSES
- CITES
- EQUIVALENT_TO

Utility Classes:
- GraphStats        : the node and relationship counts after a build
- GraphQueryResult  : the nodes and relationships returned by a query
"""

# import libraries
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field


# Node Type: ArticleNode


@dataclass
class ArticleNode:
    """
    This class is for a regulatory article or provision

    Article nodes are the center of most of the graph relationships:
    - Instrument CONTAINS -> Article
    - Article REFERENCES -> Article (cross-references)
    - Article DEFINES -> Definition (only for Art.4 GDPR and Art.5 FADP)
    - Article IMPOSES -> Obligation (extracted by LLM)
    - FADP and GDPR articles are linked by EQUIVALENT_TO
    """

    # values: "gdpr", "fadp", "edpb_legitimate_interest",
    # "edpb_article48", "edpb_consent", and "fdpic_technical_measures"
    # this matches the source_id in "DOCUMENT_REGISTRY" (see: app/ingestion/config.py)
    source_id: str

    # the GDPR uses the full word: "Article 4", "Article 28"
    # the FADP uses the abbreviation: "Art. 5", "Art. 16a"
    article_label: str  # e.g., "Article 4" or "Art. 5"

    # this is truncated to 2000 characters during extraction (see: extractor.py)
    full_text: str = ""

    # e.g., "CHAPTER II - Principles" for GDPR Article 5
    chapter: str = ""

    # e.g., "Section 1 - Transparency and modalities" for some GDPR articles
    section: str = ""

    @property
    def node_id(self) -> str:
        """
        This is for the unique identifier for this article in the graph

        Format: "{source_id}:{article_label}"
        Examples: "gdpr:Article 5", "fadp:Art. 7", "fadp:Art. 16a"

        The "source_id" prefix prevents collisions between the GDPR and FADP articles
        that share the same label (e.g., "gdpr:Article 1" vs. "fadp:Art. 1")
        """
        return f"{self.source_id}:{self.article_label}"


# Node Type: DefinitionNode


@dataclass
class DefinitionNode:
    """
    This class is for the legal term definitions extracted from Art.4 GDPR and Art.5 FADP

    The definitions are extracted with regex in GraphExtractor (see: extractor.py)
    and are linked to their source article via a "DEFINES" relationship, i.e.,
    Article "gdpr:Article 4" DEFINES -> Definition "gdpr:def:personal data"
    """

    # e.g., "personal data", "controller", "processing", "consent"
    term: str

    # this is truncated to 500 chars during the extraction
    definition_text: str

    # "gdpr" or "fadp"
    source_id: str

    # "Article 4" for GDPR, "Art. 5" for FADP
    article_label: str

    @property
    def node_id(self) -> str:
        """
        For the unique identifier for this definition in the graph

        Format: "{source_id}:def:{term_lowercase}"
        Examples: "gdpr:def:personal data", "fadp:def:data processing"

        The term is in lowercase in order to avoid duplicates from inconsistencies
        """
        return f"{self.source_id}:def:{self.term.lower()}"


# Node Type: ObligationNode


@dataclass
class ObligationNode:
    """
    This is an obligation,right,prohibition,permission extracted by the LLM

    The obligations are extracted by sending each statute article to
    the LLM "Qwen3-Next 80B-A3B" (via Together AI) and this returns structured JSON.
    The results are then cached so any rebuild doesn't need to call the LLM again

    There are 4 recognized types:
    - "obligation"  : this is something an entity MUST do
    - "right"       : this is something an entity is ENTITLED to do
    - "prohibition" : this is something an entity MUST NOT do
    - "permission"  : this is something an entity MAY do

    Every obligation is linked to its source article via "IMPOSES":
    Article "gdpr:Article 33" IMPOSES -> Obligation "a1b2c3d4e5f6..."
    """

    # this is truncated to 300 chars during the extraction
    description: str

    obligation_type: str  # "obligation", "right", "prohibition", "permission"

    # only the statutes have obligations and guidance documents
    # don't impose legally binding obligations.
    # values: "gdpr" or "fadp"
    source_id: str

    # e.g., "Article 33" (GDPR) or "Art. 16" (FADP)
    article_label: str

    # e.g., "controller", "processor", "data subject", "supervisory authority"
    subject: str = ""

    @property
    def node_id(self) -> str:
        """
        This is the unique identifier for this obligation (it is generated via SHA-256 hashing)

        It hashes a combination of source_id, article_label, and the first 50
        chars of the description (in order to distinguish obligations in the same
        article)

        Returns:
        a 16-char hex string, e.g., "a1b2c3d4e5f67890"
        """
        key = f"{self.source_id}:{self.article_label}:{self.description[:50]}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


# GraphRelationship class


@dataclass
class GraphRelationship:
    """
    This is for a relationship between two graph nodes

    There are 6 relationship types in the graph:

    CONTAINS      : Instrument -> Article
    REFERENCES    : Article -> Article
    DEFINES       : Article -> Definition
    IMPOSES       : Article -> Obligation
    CITES         : Instrument (guidance) -> Article (statute)
    EQUIVALENT_TO : FADP Article <-> GDPR Article
    """

    source_node_id: str
    target_node_id: str

    rel_type: str  # the relationship types: CONTAINS, REFERENCES, DEFINES, IMPOSES, CITES, EQUIVALENT_TO

    # relationship metadata (optional), e.g.:
    # REFERENCES: {"reference_text": "Article 32(1)"}
    # EQUIVALENT_TO: {"note": "Processing principles"}
    properties: dict = field(default_factory=dict)


# GraphStats class


@dataclass
class GraphStats:
    """
    This is for the node and relationship counts after a graph is built

    It is used for logging, saving to "data/output/graph/build_stats.json", and
    for checking the expected build output
    """

    # node counts
    instruments: int = 0  # 6: GDPR, FADP, 3 EDPB guidelines, 1 FDPIC guide
    articles: int = 0  # 174: 100 GDPR and 74 FADP
    definitions: int = 0  # 33: 23 from GDPR Art.4 and 10 from FADP Art.5
    obligations: int = 0  # 632: extracted from the 171 statute articles

    # relationship counts
    contains: int = 0  # 174 (1 per article)
    references: int = 0  # 237
    defines: int = 0  # 33 (1 per definition)
    imposes: int = 0  # 632 (1 per obligation)
    cites: int = 0  # 100
    equivalent_to: int = 0  # 14

    def __str__(self) -> str:
        """To pretty print the node and the relationship counts"""
        lines = [
            "Nodes:",
            f"  Instrument:  {self.instruments}",
            f"  Article:     {self.articles}",
            f"  Definition:  {self.definitions}",
            f"  Obligation:  {self.obligations}",
            "Relationships:",
            f"  CONTAINS:       {self.contains}",
            f"  REFERENCES:     {self.references}",
            f"  DEFINES:        {self.defines}",
            f"  IMPOSES:        {self.imposes}",
            f"  CITES:          {self.cites}",
            f"  EQUIVALENT_TO:  {self.equivalent_to}",
        ]
        return "\n".join(lines)


# GraphQueryResult class


@dataclass
class GraphQueryResult:
    """
    This is for the result from a graph query

    It is returned by the Graph Layer when the Orchestration Layer (Layer 5)
    queries the graph (e.g., "What articles does GDPR Article 28 reference?").
    It is used by:
    - the expand_graph node (see: nodes.py) to enrich the LLM prompt context
    - the UI Layer to render the graph visualisations (see: graph_viz.py)
    """

    # the node dicts with keys such as: node_id, article_label, source_id, full_text
    # (for articles) or term, definition_text (for definitions)
    nodes: list[dict] = field(default_factory=list)

    # the relationship dicts with keys as: source, target, type, and any
    # relationship properties (e.g., reference_text, note)
    relationships: list[dict] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """
        This is true only if both the nodes and the relationships are empty

        It is used by the orchestration pipeline in order to decide if to include
        the graph context in the generation prompt by the LLM
        """
        return not self.nodes and not self.relationships
