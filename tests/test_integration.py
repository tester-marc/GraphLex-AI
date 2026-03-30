"""These are the integration tests in order to verify that the data flows correctly
between the pipeline nodes.

Every node reads from a shared state dict, then does its work, and writes the results back.
The unit tests cover the individual nodes and verify the connections between the nodes

To run: python -m pytest tests/test_integration.py -v
"""

# import libraries
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from app.graph.models import GraphQueryResult
from app.orchestration.config import OrchestrationConfig
from app.orchestration.nodes import (
    interpret_node,
    retrieve_node,
    expand_graph_node,
    generate_node,
)
from app.orchestration.pipeline import OrchestrationPipeline
from app.retrieval.models import SearchResult


# fixtures


@pytest.fixture
def config():
    """default OrchestrationConfig, takes precedence over "conftest.py" """
    return OrchestrationConfig()


# helper function _make_search_result


def _make_search_result(**overrides) -> SearchResult:
    """for SearchResult with defaults (GDPR Article 17, score=0.92)

    override any field if required:
    _make_search_result(source_id="fadp", article="Art. 25", score=0.80)
    """
    defaults = dict(
        chunk_id="abc123",
        text="Article 17 Right to erasure.\n1. The data subject shall have the right...",
        source_id="gdpr",
        instrument_type="statute",
        jurisdiction="EU",
        article="Article 17",
        section=None,
        paragraph="1",
        cross_references=["Article 6"],
        distance=0.08,
        score=0.92,
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


# Test class 1: the Interpret -> Retrieve data flow


class TestInterpretToRetrieve:
    """this verifies that interpret output feeds correctly into retrieve

    the interpret node extracts "article_refs", "source_filters", and
    "jurisdiction_filters" from the query and the retrieve node passes these
    to Weaviate as query filters

    this test check that chain works properly
    """

    # patch where the function is used ("nodes.py") and not where it is defined
    @patch("app.orchestration.nodes._get_retrieval")
    def test_filters_passed_through(self, mock_get, config):
        """jurisdiction and source filters from interpret should get to Weaviate

        Scenario: "GDPR Article 17 erasure"
        Expected: source_ids=["gdpr"], jurisdictions=["EU"] are passed to Weaviate
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = (
            []
        )  # the results don't matter here, only the call args
        mock_get.return_value = mock_pipeline

        interpret_result = interpret_node(
            {"query_text": "GDPR Article 17 erasure", "stages_completed": []},
            config,
        )

        state = {**interpret_result}
        retrieve_result = retrieve_node(state, config)

        mock_pipeline.query.assert_called_once()
        call_kwargs = mock_pipeline.query.call_args

        assert call_kwargs.kwargs["source_ids"] == ["gdpr"]
        assert call_kwargs.kwargs["jurisdictions"] == ["EU"]


# Test class 2: Retrieve -> Expand Graph data flow


class TestRetrieveToExpandGraph:
    """This verifies retrieved chunks inform the graph expansion when no explicit article references

    For the conceptual queries (e.g., "What is data portability?"),
    interpret finds no "article_refs". "expand_graph" should then fall back
    to deriving references from the chunk metadata ("source_id" and "article label")
    for Neo4j lookups
    """

    # @patch("..._get_graph")      : mock_graph (second parameter)
    # @patch("..._get_retrieval")  : mock_ret (first parameter after self)
    @patch("app.orchestration.nodes._get_graph")
    @patch("app.orchestration.nodes._get_retrieval")
    def test_chunk_articles_used_for_graph_lookup(self, mock_ret, mock_graph, config):
        """the article metadata from the retrieved chunks should trigger Neo4j lookups

        Scenario: "What is data portability?" -> no explicit article reference
        -> Weaviate returns chunks from the GDPR Article 17 and FADP Art. 25
        -> "expand_graph" derives references from the chunk metadata and calls "query_article"
        """
        mock_ret_pipeline = MagicMock()
        mock_ret_pipeline.query.return_value = [
            _make_search_result(source_id="gdpr", article="Article 17"),
            _make_search_result(
                chunk_id="xyz", source_id="fadp", article="Art. 25", score=0.80
            ),
        ]
        mock_ret.return_value = mock_ret_pipeline

        mock_graph_pipeline = MagicMock()
        mock_graph_pipeline.query_article.return_value = GraphQueryResult()
        mock_graph_pipeline.queries.get_equivalents.return_value = []
        mock_graph_pipeline.queries.get_guidance_for_article.return_value = []
        mock_graph.return_value = mock_graph_pipeline

        state = {"query_text": "What is data portability?", "stages_completed": []}
        interpret_result = interpret_node(state, config)
        state.update(interpret_result)
        retrieve_result = retrieve_node(state, config)
        state.update(retrieve_result)
        expand_result = expand_graph_node(state, config)

        assert mock_graph_pipeline.query_article.call_count >= 1


# Test class 3: The full text pipeline flow


class TestFullTextFlow:
    """These are integration tests for the entire interpret -> retrieve -> expand_graph -> generate flow

    It covers: a well evidenced query, an out of corpus query, and a cross-jurisdictional query
    """

    @patch("app.orchestration.nodes._get_llm")
    @patch("app.orchestration.nodes._get_graph")
    @patch("app.orchestration.nodes._get_retrieval")
    def test_end_to_end_text_flow(self, mock_ret, mock_graph, mock_llm, config):
        """a well evidenced query should flow through all 4 stages with enough confidence

        Scenario: "What does GDPR Article 17 say about erasure?"
        Expected: all stages are complete, confidence="sufficient", and answer contains "Article 17"
        """
        mock_ret_pipeline = MagicMock()
        mock_ret_pipeline.query.return_value = [_make_search_result()]
        mock_ret.return_value = mock_ret_pipeline

        mock_graph_pipeline = MagicMock()
        mock_graph_pipeline.query_article.return_value = GraphQueryResult(
            nodes=[
                {
                    "node_id": "gdpr:Article 17",
                    "article_label": "Article 17",
                    "source_id": "gdpr",
                }
            ],
            relationships=[],
        )
        mock_graph_pipeline.queries.get_equivalents.return_value = []
        mock_graph_pipeline.queries.get_guidance_for_article.return_value = []
        mock_graph.return_value = mock_graph_pipeline

        mock_client = MagicMock()
        # no uncertainty markers in this answer -> the confidence will be "sufficient"
        mock_client.generate.return_value = (
            "Under GDPR Article 17, the data subject has the right to erasure.",
            500.0,
            100,
            50,
        )
        mock_llm.return_value = mock_client

        state = {
            "query_text": "What does GDPR Article 17 say about erasure?",
            "stages_completed": [],
            "errors": [],
        }

        interpret_out = interpret_node(state, config)
        state.update(interpret_out)
        retrieve_out = retrieve_node(state, config)
        state.update(retrieve_out)
        expand_out = expand_graph_node(state, config)
        state.update(expand_out)
        generate_out = generate_node(state, config)
        state.update(generate_out)

        assert "interpret" in state["stages_completed"]
        assert "retrieve" in state["stages_completed"]
        assert "expand_graph" in state["stages_completed"]
        assert "generate" in state["stages_completed"]
        assert state["confidence"] == "sufficient"
        assert "Article 17" in state["answer"]
        assert len(state["retrieved_chunks"]) == 1
        assert state["article_refs"] == ["gdpr:Article 17"]

    @patch("app.orchestration.nodes._get_llm")
    @patch("app.orchestration.nodes._get_graph")
    @patch("app.orchestration.nodes._get_retrieval")
    def test_insufficient_evidence_flow(self, mock_ret, mock_graph, mock_llm, config):
        """out of corpus queries should create confidence="insufficient_evidence"

        Scenario: The user asks about the EU AI Act which is not in the corpus.
        The LLM then responds with an "Insufficient evidence" marker, which
        "generate_node" detects to set the confidence="insufficient_evidence"
        """
        mock_ret_pipeline = MagicMock()
        mock_ret_pipeline.query.return_value = [
            _make_search_result(text="Some unrelated GDPR text.", score=0.3),
        ]
        mock_ret.return_value = mock_ret_pipeline

        mock_graph_pipeline = MagicMock()
        mock_graph_pipeline.query_article.return_value = GraphQueryResult()
        mock_graph_pipeline.queries.get_equivalents.return_value = []
        mock_graph_pipeline.queries.get_guidance_for_article.return_value = []
        mock_graph.return_value = mock_graph_pipeline

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "Insufficient evidence in the provided context to answer this question. "
            "The AI Act is not part of the regulatory corpus.",
            300.0,
            80,
            30,
        )
        mock_llm.return_value = mock_client

        state = {
            "query_text": "What does the EU AI Act say about high-risk systems?",
            "stages_completed": [],
            "errors": [],
        }

        interpret_out = interpret_node(state, config)
        state.update(interpret_out)
        retrieve_out = retrieve_node(state, config)
        state.update(retrieve_out)
        expand_out = expand_graph_node(state, config)
        state.update(expand_out)
        generate_out = generate_node(state, config)
        state.update(generate_out)

        assert state["confidence"] == "insufficient_evidence"
        assert "AI Act" in state["answer"]

    @patch("app.orchestration.nodes._get_llm")
    @patch("app.orchestration.nodes._get_graph")
    @patch("app.orchestration.nodes._get_retrieval")
    def test_cross_jurisdictional_flow(self, mock_ret, mock_graph, mock_llm, config):
        """cross-jurisdictional queries should expand the context for both GDPR and FADP.

        Scenario: "Compare GDPR Article 17 and its Swiss equivalent"
        The Neo4j graph has an "EQUIVALENT_TO" edge: GDPR Art.17 <-> FADP Art.32

        "expand_graph" should query both articles and include both in "graph_context"
        as "equivalent_article" entries
        """
        mock_ret_pipeline = MagicMock()
        mock_ret_pipeline.query.return_value = [_make_search_result()]
        mock_ret.return_value = mock_ret_pipeline

        mock_graph_pipeline = MagicMock()
        # "side_effect" returns different values per call:
        # for the first call: GDPR Article 17, for the second call: FADP Art. 32 (via "get_equivalents")
        mock_graph_pipeline.query_article.side_effect = [
            GraphQueryResult(
                nodes=[{"node_id": "gdpr:Article 17", "source_id": "gdpr"}],
                relationships=[],
            ),
            GraphQueryResult(
                nodes=[{"node_id": "fadp:Art. 32", "source_id": "fadp"}],
                relationships=[],
            ),
        ]
        mock_graph_pipeline.queries.get_equivalents.return_value = [
            {"node_id": "fadp:Art. 32", "label": "Art. 32", "source_id": "fadp"},
        ]
        mock_graph_pipeline.queries.get_guidance_for_article.return_value = []
        mock_graph.return_value = mock_graph_pipeline

        mock_client = MagicMock()
        mock_client.generate.return_value = (
            "GDPR Article 17 and its FADP equivalent Art. 32 both provide erasure rights.",
            600.0,
            120,
            60,
        )
        mock_llm.return_value = mock_client

        state = {
            "query_text": "Compare GDPR Article 17 and its Swiss equivalent",
            "stages_completed": [],
            "errors": [],
        }

        interpret_out = interpret_node(state, config)
        state.update(interpret_out)
        retrieve_out = retrieve_node(state, config)
        state.update(retrieve_out)
        expand_out = expand_graph_node(state, config)
        state.update(expand_out)
        generate_out = generate_node(state, config)
        state.update(generate_out)

        equiv_items = [
            c for c in state["graph_context"] if c["type"] == "equivalent_article"
        ]
        assert len(equiv_items) >= 1


# Test class 4: Pipeline routing


class TestPipelineRouting:
    """To test conditional routing at the pipeline entry

    "_route_input()" returns "audio" or "text" based on the state:
    - "audio" only if input_mode=="audio" AND audio_path is not empty
    - otherwise "text" (incl. fall back when "audio_path" is missing)

    LangGraph maps these to {"audio": "transcribe", "text": "interpret"}
    """

    def test_text_input_routes_to_interpret(self):
        """input_mode="text" should route to interpret"""
        result = OrchestrationPipeline._route_input(
            {
                "input_mode": "text",
                "audio_path": "",
            }
        )
        assert result == "text"

    def test_audio_input_routes_to_transcribe(self):
        """input_mode="audio" with a valid "audio_path" should route to transcribe"""
        result = OrchestrationPipeline._route_input(
            {
                "input_mode": "audio",
                "audio_path": "/tmp/query.wav",
            }
        )
        assert result == "audio"

    def test_audio_without_path_routes_to_text(self):
        """input_mode="audio" with an empty "audio_path" should fall back to the text route"""
        result = OrchestrationPipeline._route_input(
            {
                "input_mode": "audio",
                "audio_path": "",
            }
        )
        assert result == "text"

    def test_missing_mode_defaults_to_text(self):
        """the empty state (no "input_mode" key) should default to the text routing"""
        result = OrchestrationPipeline._route_input({})
        assert result == "text"
