"""
This is the package for the Graph layer (Neo4j regulatory knowledge graph) and the public API of app.graph

This package is imported in order to access the full graph stack
without having to reach into the individual modules, as follows:

from app.graph import GraphPipeline (instead of "app.graph.pipeline")

Architecture:

This builds and queries a Neo4j knowledge graph that contains the following:
- Instrument nodes: 6 regulatory documents (e.g., GDPR, Swiss FADP)
- Article nodes: 174 individual articles and provisions
- Definition nodes: 33 legal term definitions (e.g., GDPR Art.4 / FADP Art.5)
- Obligation nodes: 632 obligations/rights/prohibitions extracted by a LLM

Relationships:
CONTAINS, REFERENCES, DEFINES, IMPOSES, CITES, EQUIVALENT_TO

CLI reference (from the project root):

python -m app.graph build               # builds the structural graph
python -m app.graph build-obligations   # add the LLM extracted obligations
python -m app.graph status              # show the node/relationship counts
python -m app.graph article "Art. 6"    # looks up a specific article
python -m app.graph refs "Article 6"    # finds cross-references
python -m app.graph defs "consent"      # searches for a definition
python -m app.graph guidance "Art. 6"   # finds citing guidance
python -m app.graph delete              # in order to wipe and rebuild
"""

# GraphConfig: these are the Neo4j connection settings (uri, user, password, database)
# This reads from env variables NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD if not passed directly
# It provides the chunks_dir, output_dir, and is_configured()
from app.graph.config import GraphConfig

# Data model classes:
# These are dataclasses that represent the graph entities
#
# ArticleNode       : one regulatory article, e.g., node_id = "gdpr:Article 4"
# DefinitionNode    : a legal term and definition, e.g., node_id = "gdpr:def:personal data"
# ObligationNode    : an obligation/right/prohibition/permission extracted by the LLM, e.g.,
#                     node_id = 16-character SHA-256 hash of source + article + description
# GraphRelationship : this is an edge between 2 nodes (source_node_id -> target_node_id,
#                     rel_type, optional properties dict)
# GraphQueryResult  : this is the container for the Cypher (Cypher is the Neo4j query language)
#                     query results (nodes + relationships), e.g.,
#                     is_empty = is true if both lists are empty
# GraphStats        : these are integer counts of all node and relationship types
from app.graph.models import (
    ArticleNode,
    DefinitionNode,
    GraphQueryResult,
    GraphRelationship,
    GraphStats,
    ObligationNode,
)

# Neo4jStore:
# the low level database layer that handles the connection lifecycle,
# the uniqueness constraints, the batch node and relationship insertion (i.e., UNWIND pattern),
# arbitrary Cypher queries, the stats collection, as well as the full graph wipe.
from app.graph.neo4j_store import Neo4jStore

# GraphPipeline:
# the orchestrator used by the rest of the application. It
# combines extraction, persistence, and querying.
# Methods: connect/close, build, build_obligations, query_article,
# query_references, query_definitions, query_guidance, query_subgraph_for_viz,
# get_stats, delete.
from app.graph.pipeline import GraphPipeline

# this controls what "from app.graph import *" exposes
__all__ = [
    "GraphConfig",
    "ArticleNode",
    "DefinitionNode",
    "GraphQueryResult",
    "GraphRelationship",
    "GraphStats",
    "ObligationNode",
    "Neo4jStore",
    "GraphPipeline",
]
