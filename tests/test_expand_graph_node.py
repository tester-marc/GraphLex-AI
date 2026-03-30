"""
These are unit tests for the "expand_graph" node, i.e., the Neo4j knowledge graph expansion
It tests the "expand_graph" stage of the pipeline.

After the retrieval fetches the relevant chunks from Weaviate, expand_graph enriches the
context by querying the Neo4j graph which contains:
- articles from GDPR (EU) and FADP (Swiss) data protection statutes
- legal definitions (e.g., "personal data", "processing")
- cross-references between articles (e.g., Article 17 references Article 6)
- cross-jurisdictional equivalences (e.g., GDPR Art. 17 <-> FADP Art. 32)
- guidance documents that cite specific statute articles
- and obligations extracted from statute articles by the LLM extraction

For every article reference the node:
1. looks up the article's definitions, obligations, and crossreferences
2. finds equivalent articles in the other jurisdiction
3. finds guidance documents that cite this article

The tests use "unittest.mock" to replace the Neo4j connection.

The test classes are:
1. TestExplicitArticleRefs    : the query mentions specific articles
2. TestCrossJurisdictional    : the article has an equivalent in the other jurisdiction
3. TestGuidanceCitations      : the guidance documents cite the referenced article
4. TestChunkRefFallback       : no explicit refs, falls back to chunk metadata
5. TestEmptyResults           : for the edge cases where no graph data is available
"""

# import libraries
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from app.graph.models import GraphQueryResult
from app.orchestration.config import OrchestrationConfig
from app.orchestration.nodes import expand_graph_node


# the local fixture takes precedence over any fixture in conftest.py with the same name
@pytest.fixture
def config():
    return OrchestrationConfig()


def _make_article_context(node_id: str = "gdpr:Article 17") -> GraphQueryResult:
    """this is a realistic mock GraphQueryResult that simulates a Neo4j article lookup"""
    return GraphQueryResult(
        nodes=[
            {"node_id": node_id, "article_label": "Article 17", "source_id": "gdpr"},
            {
                "node_id": "gdpr:def:personal data",
                "term": "personal data",
                "definition_text": "any information relating to an identified natural person",
            },
        ],
        relationships=[
            {"type": "DEFINES", "from": node_id, "to": "gdpr:def:personal data"},
            {"type": "REFERENCES", "from": node_id, "to": "gdpr:Article 6"},
        ],
    )


# Test Class 1: Graph expansion with explicit references to articles
#
# Explicit article refs means the user's query directly mentioned a specific
# article (e.g., "What does GDPR Article 17 say?"). The interpret node then parses
# this into article_refs=["gdpr:Article 17"] for expand_graph to look up


class TestExplicitArticleRefs:

    @patch("app.orchestration.nodes._get_graph")
    def test_looks_up_explicit_refs(self, mock_get, config):
        mock_graph = MagicMock()
        mock_graph.query_article.return_value = _make_article_context()
        mock_graph.queries.get_equivalents.return_value = []
        mock_graph.queries.get_guidance_for_article.return_value = []
        mock_get.return_value = mock_graph

        state = {
            "article_refs": ["gdpr:Article 17"],
            "retrieved_chunks": [],
            "stages_completed": [],
        }
        result = expand_graph_node(state, config)

        mock_graph.query_article.assert_called_with("gdpr:Article 17")
        assert len(result["graph_context"]) >= 1
        assert result["graph_context"][0]["type"] == "article_context"

    @patch("app.orchestration.nodes._get_graph")
    def test_includes_definitions_and_refs(self, mock_get, config):
        mock_graph = MagicMock()
        mock_graph.query_article.return_value = _make_article_context()
        mock_graph.queries.get_equivalents.return_value = []
        mock_graph.queries.get_guidance_for_article.return_value = []
        mock_get.return_value = mock_graph

        state = {
            "article_refs": ["gdpr:Article 17"],
            "retrieved_chunks": [],
            "stages_completed": [],
        }
        result = expand_graph_node(state, config)

        ctx = result["graph_context"][0]
        rel_types = [r["type"] for r in ctx["relationships"]]

        assert "DEFINES" in rel_types
        assert "REFERENCES" in rel_types


# Test Class 2: Cross-jurisdictional equivalents
#
# GDPR and FADP articles that cover the same topic are linked with "EQUIVALENT_TO"
# relationships in Neo4j (e.g., GDPR Art. 17 <-> FADP Art. 32). When the node processes
# an article, the node should also look up its equivalent and then add it to
# "graph_context" as "equivalent_article" so the generate node can compare both of them


class TestCrossJurisdictional:

    @patch("app.orchestration.nodes._get_graph")
    def test_fetches_equivalents(self, mock_get, config):
        mock_graph = MagicMock()

        # "side_effect" returns different values on successive calls
        # first call is the primary GDPR article, 2nd call is its FADP equivalent
        mock_graph.query_article.side_effect = [
            _make_article_context("gdpr:Article 17"),
            GraphQueryResult(
                nodes=[
                    {
                        "node_id": "fadp:Art. 32",
                        "article_label": "Art. 32",
                        "source_id": "fadp",
                    }
                ],
                relationships=[],
            ),
        ]
        mock_graph.queries.get_equivalents.return_value = [
            {"node_id": "fadp:Art. 32", "label": "Art. 32", "source_id": "fadp"},
        ]
        mock_graph.queries.get_guidance_for_article.return_value = []
        mock_get.return_value = mock_graph

        state = {
            "article_refs": ["gdpr:Article 17"],
            "retrieved_chunks": [],
            "stages_completed": [],
        }
        result = expand_graph_node(state, config)

        equiv_items = [
            c for c in result["graph_context"] if c["type"] == "equivalent_article"
        ]

        assert len(equiv_items) == 1
        assert equiv_items[0]["equivalent_ref"] == "fadp:Art. 32"


# Test Class 3: Guidance citations
#
# The guidance documents (EDPB guidelines, FDPIC guides) cite specific statute
# articles with the "CITES" relationships in Neo4j. The node adds the citing documents
# to "graph_context" so that the generate node can reference the relevant guidance


class TestGuidanceCitations:

    @patch("app.orchestration.nodes._get_graph")
    def test_includes_guidance(self, mock_get, config):
        mock_graph = MagicMock()
        mock_graph.query_article.return_value = _make_article_context()
        mock_graph.queries.get_equivalents.return_value = []
        mock_graph.queries.get_guidance_for_article.return_value = [
            {"source_id": "edpb_consent", "title": "EDPB Guidelines on Consent"},
        ]
        mock_get.return_value = mock_graph

        state = {
            "article_refs": ["gdpr:Article 17"],
            "retrieved_chunks": [],
            "stages_completed": [],
        }
        result = expand_graph_node(state, config)

        guidance_items = [
            c for c in result["graph_context"] if c["type"] == "guidance_citations"
        ]

        assert len(guidance_items) == 1
        assert guidance_items[0]["guidance"][0]["source_id"] == "edpb_consent"


# Test Class 4: Fall back to the retrieved chunk references
#
# if article_refs=[] (for a vague query e.g. "explain data erasure rules") then the node
# falls back to extracting the article references from the retrieved chunks' metadata.
# In order to avoid to many lookups on broad queries, the fallback is capped at 5 refs


class TestChunkRefFallback:

    @patch("app.orchestration.nodes._get_graph")
    def test_extracts_refs_from_chunks_when_no_explicit(self, mock_get, config):
        mock_graph = MagicMock()
        mock_graph.query_article.return_value = _make_article_context()
        mock_graph.queries.get_equivalents.return_value = []
        mock_graph.queries.get_guidance_for_article.return_value = []
        mock_get.return_value = mock_graph

        state = {
            "article_refs": [],
            "retrieved_chunks": [
                {"article": "Article 17", "source_id": "gdpr", "text": "..."},
            ],
            "stages_completed": [],
        }
        result = expand_graph_node(state, config)

        assert len(result["graph_context"]) >= 1

    @patch("app.orchestration.nodes._get_graph")
    def test_limits_chunk_refs_to_five(self, mock_get, config):
        mock_graph = MagicMock()
        mock_graph.query_article.return_value = GraphQueryResult()
        mock_graph.queries.get_equivalents.return_value = []
        mock_graph.queries.get_guidance_for_article.return_value = []
        mock_get.return_value = mock_graph

        # 19 chunks from 19 different articles, only 5 should be looked up
        chunks = [
            {"article": f"Article {i}", "source_id": "gdpr", "text": "..."}
            for i in range(1, 20)
        ]
        state = {
            "article_refs": [],
            "retrieved_chunks": chunks,
            "stages_completed": [],
        }
        expand_graph_node(state, config)

        assert mock_graph.query_article.call_count <= 5


# Test Class 5: Empty or edge case results
#
# The node must never crash when it finds nothing, it should return an empty
# "graph_context" and still mark as completed in "stages_completed"


class TestEmptyResults:

    @patch("app.orchestration.nodes._get_graph")
    def test_no_refs_no_chunks(self, mock_get, config):
        mock_graph = MagicMock()
        mock_get.return_value = mock_graph

        state = {
            "article_refs": [],
            "retrieved_chunks": [],
            "stages_completed": [],
        }
        result = expand_graph_node(state, config)

        assert result["graph_context"] == []
        assert "expand_graph" in result["stages_completed"]

    @patch("app.orchestration.nodes._get_graph")
    def test_article_not_in_graph(self, mock_get, config):
        mock_graph = MagicMock()
        mock_graph.query_article.return_value = (
            GraphQueryResult()
        )  # empty, is_empty = True
        mock_graph.queries.get_equivalents.return_value = []
        mock_graph.queries.get_guidance_for_article.return_value = []
        mock_get.return_value = mock_graph

        state = {
            "article_refs": ["gdpr:Article 999"],  # a non existent article
            "retrieved_chunks": [],
            "stages_completed": [],
        }
        result = expand_graph_node(state, config)

        # the node checks "if not ctx.is_empty" before adding so this should be empty
        article_items = [
            c for c in result["graph_context"] if c["type"] == "article_context"
        ]
        assert article_items == []
