"""these are unit tests for the retrieve node: vector search with mock Weaviate

Retrieve node takes the parsed query from the interpret node, then embeds it via
"text-embedding-3-large", searches Weaviate for nearest chunks, applies the metadata
filters (source, jurisdiction), and writes the results to pipeline state

Weaviate and the OpenAI Embeddings API are mocked, in order that the tests are fast
and free of external dependencies

the node receives SearchResult dataclass objects from Weaviate,
but has to convert them to plain dicts before storing in LangGraph state
(must be JSON-serialisable) - several tests verify this serialization

To run: python -m pytest tests/test_retrieve_node.py -v
"""

# import libraries
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from app.orchestration.config import OrchestrationConfig
from app.orchestration.nodes import retrieve_node
from app.retrieval.models import SearchResult


# fixtures


@pytest.fixture
def config():
    """default OrchestrationConfig, retrieve node reads "config.retrieval_top_k" """
    return OrchestrationConfig()


# helper function


def _make_search_result(**overrides) -> SearchResult:
    """factory for SearchResult with defaults (GDPR Article 17 chunk)

    Parameters:
    **overrides: any keyword arguments replace corresponding defaults

    Returns:
    a SearchResult dataclass instance
    """
    defaults = dict(
        chunk_id="abc123",
        text="Article 17 text here",
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


# tests


class TestRetrieveNode:
    """to test the retrieve node's query execution, serialisation, and error handling"""

    @patch("app.orchestration.nodes._get_retrieval")
    def test_returns_serialised_chunks(self, mock_get, config):
        """this verifies that SearchResult objects are correctly serialised to dicts

        node must convert SearchResult instances to plain dicts before storing
        them in pipeline state
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = [
            _make_search_result(),
            _make_search_result(chunk_id="def456", source_id="fadp", score=0.85),
        ]
        mock_get.return_value = mock_pipeline

        state = {
            "search_query": "right to erasure",
            "source_filters": [],
            "jurisdiction_filters": [],
            "stages_completed": [],
        }

        result = retrieve_node(state, config)

        assert len(result["retrieved_chunks"]) == 2
        assert result["retrieved_chunks"][0]["chunk_id"] == "abc123"
        assert result["retrieved_chunks"][1]["source_id"] == "fadp"

    @patch("app.orchestration.nodes._get_retrieval")
    def test_chunk_has_all_required_fields(self, mock_get, config):
        """to verify serialised chunks contain every field that downstream nodes need

        expand_graph reads:  chunk["article"], chunk["source_id"]
        generate reads:      chunk["text"], chunk["source_id"], chunk["article"],
                             chunk["section"], chunk["instrument_type"], chunk["score"]

        """
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = [_make_search_result()]
        mock_get.return_value = mock_pipeline

        state = {
            "search_query": "test",
            "source_filters": [],
            "jurisdiction_filters": [],
            "stages_completed": [],
        }

        result = retrieve_node(state, config)
        chunk = result["retrieved_chunks"][0]

        required_fields = [
            "chunk_id",
            "text",
            "source_id",
            "instrument_type",
            "jurisdiction",
            "article",
            "section",
            "paragraph",
            "cross_references",
            "score",
        ]
        for field in required_fields:
            assert field in chunk, f"Missing field: {field}"

    @patch("app.orchestration.nodes._get_retrieval")
    def test_passes_source_filters(self, mock_get, config):
        """this verifies source and jurisdiction filters are forwarded to Weaviate

        the filters are set by the interpret node when it detects a specific document
        or jurisdiction in the query
        passing them through improves precision by restricting
        the vector search to relevant chunks only
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = []
        mock_get.return_value = mock_pipeline

        state = {
            "search_query": "GDPR consent",
            "source_filters": ["gdpr"],
            "jurisdiction_filters": ["EU"],
            "stages_completed": [],
        }

        retrieve_node(state, config)

        mock_pipeline.query.assert_called_once_with(
            text="GDPR consent",
            top_k=config.retrieval_top_k,
            source_ids=["gdpr"],
            jurisdictions=["EU"],
        )

    @patch("app.orchestration.nodes._get_retrieval")
    def test_empty_filters_passed_as_none(self, mock_get, config):
        """this verifies empty filter lists are converted to None before calling Weaviate

        None means "no filter" (i.e., search everything)
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = []
        mock_get.return_value = mock_pipeline

        state = {
            "search_query": "portability",
            "source_filters": [],
            "jurisdiction_filters": [],
            "stages_completed": [],
        }

        retrieve_node(state, config)

        mock_pipeline.query.assert_called_once_with(
            text="portability",
            top_k=config.retrieval_top_k,
            source_ids=None,
            jurisdictions=None,
        )

    def test_empty_query_returns_error(self, config):
        """this verifies graceful error handling when there's no search query

        Node should return early without calling Weaviate, produce an empty
        retrieved_chunks list, then mark the stage as "retrieve:error", and populate
        errors list
        """
        state = {
            "search_query": "",
            "stages_completed": [],
            "errors": [],
        }
        result = retrieve_node(state, config)

        assert result["retrieved_chunks"] == []
        assert "retrieve:error" in result["stages_completed"]
        assert any("No search query" in e for e in result["errors"])

    @patch("app.orchestration.nodes._get_retrieval")
    def test_records_latency(self, mock_get, config):
        """verifies that retrieval_ms is recorded for the diagnostics display"""
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = []
        mock_get.return_value = mock_pipeline

        state = {
            "search_query": "test",
            "source_filters": [],
            "jurisdiction_filters": [],
            "stages_completed": [],
        }

        result = retrieve_node(state, config)

        assert "retrieval_ms" in result
        assert result["retrieval_ms"] >= 0

    @patch("app.orchestration.nodes._get_retrieval")
    def test_stage_tracking(self, mock_get, config):
        """this verifies "retrieve" is appended to stages_completed without replacing prior stages

        Node uses "state.get("stages_completed", []) + ["retrieve"]" to create
        a new list, thereby preserving LangGraph's shared state semantics
        """
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = []
        mock_get.return_value = mock_pipeline

        state = {
            "search_query": "test",
            "source_filters": [],
            "jurisdiction_filters": [],
            "stages_completed": ["interpret"],
        }

        result = retrieve_node(state, config)

        assert result["stages_completed"] == ["interpret", "retrieve"]
