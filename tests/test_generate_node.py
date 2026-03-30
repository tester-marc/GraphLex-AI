"""
These are the unit tests for the generate node, for LLM answer generation and confidence detection

This tests 2 things:
1. "_build_context_prompt" puts the retrieved chunks and the graph context into an
    LLM prompt. The sections are "Retrieved Passages" (Weaviate) and "Knowledge Graph
     Context" (Neo4j: definitions, cross-refs, equivalences, guidance, obligations)

2. "generate_node" calls the LLM "Qwen3-Next 80B-A3B" (via Together AI), and then scans the answer
     for uncertainty (e.g., "insufficient evidence", "cannot determine") in order to
     classify the confidence as "sufficient" or "insufficient_evidence"

Fixtures from "conftest.py":
sample_retrieved_chunks : 2 Weaviate results (GDPR Art. 17 score 0.92, FADP Art. 25 score 0.85)
sample_graph_context    : 1 "article_context" for GDPR Art. 17 with a "personal data"
                          definition ("DEFINES") and a reference to Art. 6 ("REFERENCES")
"""

# import libraries
from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from app.orchestration.config import OrchestrationConfig
from app.orchestration.nodes import generate_node, _build_context_prompt


@pytest.fixture
def config():
    return OrchestrationConfig()


# Test class 1: Context prompt building


class TestContextPromptBuilding:

    def test_includes_retrieved_passages(self, sample_retrieved_chunks):
        state = {
            "retrieved_chunks": sample_retrieved_chunks,
            "graph_context": [],
        }
        prompt = _build_context_prompt(state)

        assert "Retrieved Passages" in prompt
        assert "GDPR" in prompt  # "source_id" is uppercase in prompt
        assert "Article 17" in prompt

    def test_includes_graph_definitions(
        self, sample_retrieved_chunks, sample_graph_context
    ):
        state = {
            "retrieved_chunks": sample_retrieved_chunks,
            "graph_context": sample_graph_context,
        }
        prompt = _build_context_prompt(state)

        assert "Knowledge Graph Context" in prompt
        assert (
            "personal data" in prompt
        )  # the term from the "DEFINES" relationship in the fixture

    def test_includes_cross_references(
        self, sample_retrieved_chunks, sample_graph_context
    ):
        state = {
            "retrieved_chunks": sample_retrieved_chunks,
            "graph_context": sample_graph_context,  # contains "REFERENCES" -> Article 6
        }
        prompt = _build_context_prompt(state)

        # either the section label or the target article is sufficient
        assert "Cross-references" in prompt or "Article 6" in prompt

    def test_includes_equivalent_articles(self):
        # this uses in line state because "equivalent_article" is not in the shared fixture
        state = {
            "retrieved_chunks": [],
            "graph_context": [
                {
                    "type": "equivalent_article",  # produced by "expand_graph" via the "EQUIVALENT_TO" relationships
                    "source_ref": "gdpr:Article 17",
                    "equivalent_ref": "fadp:Art. 32",
                    "nodes": [],
                    "relationships": [],
                },
            ],
        }
        prompt = _build_context_prompt(state)

        assert "Cross-jurisdictional equivalent" in prompt
        assert "fadp:Art. 32" in prompt

    def test_includes_guidance_citations(self):
        state = {
            "retrieved_chunks": [],
            "graph_context": [
                {
                    "type": "guidance_citations",  # this is produced by "expand_graph" via "get_guidance_for_article()""
                    "ref": "gdpr:Article 6",
                    "guidance": [
                        {
                            "title": "EDPB Guidelines on Legitimate Interest",
                            "source_id": "edpb_legitimate_interest",
                        }
                    ],
                },
            ],
        }
        prompt = _build_context_prompt(state)

        assert "Guidance citing" in prompt
        assert "Legitimate Interest" in prompt

    def test_empty_state_produces_empty_prompt(self):
        # an empty prompt tells "generate_node" to completely skip the LLM call
        state = {"retrieved_chunks": [], "graph_context": []}
        prompt = _build_context_prompt(state)

        assert prompt.strip() == ""

    def test_obligations_included(self):
        state = {
            "retrieved_chunks": [],
            "graph_context": [
                {
                    "type": "article_context",
                    "ref": "gdpr:Article 17",
                    "nodes": [
                        {
                            "node_id": "gdpr:Article 17",
                            "article_label": "Article 17",
                            "source_id": "gdpr",
                        },
                        # obligation_type "right" = granted to the data subjects (vs. controller obligation or prohibition)
                        {
                            "node_id": "ob_001",
                            "obligation_type": "right",
                            "description": "Data subject has right to erasure",
                        },
                    ],
                    "relationships": [
                        # "IMPOSES" is used for all of the obligation types
                        {"type": "IMPOSES", "from": "gdpr:Article 17", "to": "ob_001"},
                    ],
                },
            ],
        }
        prompt = _build_context_prompt(state)

        assert "Obligations" in prompt  # this is formatted as "Obligations (N):"
        assert "erasure" in prompt


# Test Class 2: LLM generation and confidence detection
#
# confidence detection: the node scans the answer for 10 uncertainty
# markers. Any match is "insufficient_evidence" and no match is "sufficient"
#
# markers: "insufficient evidence", "not enough information", "cannot determine",
#          "unable to answer", "no specific provision", "not addressed",
#          "does not contain sufficient", "beyond the scope", "no information",
#          "cannot find"


class TestGenerateNode:

    @patch("app.orchestration.nodes._get_llm")
    def test_sufficient_confidence(
        self, mock_get, config, sample_retrieved_chunks, sample_graph_context
    ):
        mock_client = MagicMock()
        # TogetherClient.generate() returns (answer_text, latency_ms, input_tokens, output_tokens)
        mock_client.generate.return_value = (
            "Based on GDPR Article 17, the data subject has the right to erasure.",
            500.0,
            100,
            50,
        )
        mock_get.return_value = mock_client

        state = {
            "query_text": "What is the right to erasure?",
            "retrieved_chunks": sample_retrieved_chunks,
            "graph_context": sample_graph_context,
            "stages_completed": ["interpret", "retrieve", "expand_graph"],
        }
        result = generate_node(state, config)

        assert result["confidence"] == "sufficient"
        assert "Article 17" in result["answer"]
        assert "generate" in result["stages_completed"]

    @patch("app.orchestration.nodes._get_llm")
    def test_insufficient_evidence_detected(
        self, mock_get, config, sample_retrieved_chunks
    ):
        mock_client = MagicMock()
        # the primary marker that the system prompt instructs LLM to use when it cannot answer
        mock_client.generate.return_value = (
            "Insufficient evidence in the provided context to answer this question.",
            300.0,
            80,
            20,
        )
        mock_get.return_value = mock_client

        state = {
            "query_text": "What does the AI Act say?",  # this is not in the GDPR / FADP corpus
            "retrieved_chunks": sample_retrieved_chunks,
            "graph_context": [],
            "stages_completed": [],
        }
        result = generate_node(state, config)

        assert result["confidence"] == "insufficient_evidence"

    @patch("app.orchestration.nodes._get_llm")
    def test_alternative_uncertainty_markers(
        self, mock_get, config, sample_retrieved_chunks
    ):
        """all of the uncertainty markers should trigger "insufficient_evidence" """
        # 5 representative phrasings mapping to "_UNCERTAINTY_MARKERS" indices 2, 3, 1, 6, 7
        markers = [
            "cannot determine the answer",
            "unable to answer from the context",
            "not enough information provided",
            "does not contain sufficient evidence",
            "this is beyond the scope of the provided context",
        ]

        for marker_text in markers:
            mock_client = MagicMock()
            mock_client.generate.return_value = (marker_text, 100.0, 50, 10)
            mock_get.return_value = mock_client

            state = {
                "query_text": "test",
                "retrieved_chunks": sample_retrieved_chunks,
                "graph_context": [],
                "stages_completed": [],
            }
            result = generate_node(state, config)

            assert (
                result["confidence"] == "insufficient_evidence"
            ), f"Failed for: {marker_text}"

    def test_no_context_returns_error(self, config):
        # if the node incorrectly calls "_get_llm()" it will crash here
        state = {
            "query_text": "test",
            "retrieved_chunks": [],
            "graph_context": [],
            "stages_completed": [],
            "errors": [],
        }
        result = generate_node(state, config)

        assert "No context was retrieved" in result["answer"]
        assert result["confidence"] == "insufficient_evidence"
        assert (
            "generate:error" in result["stages_completed"]
        )  # ":error" suffix vs. "generate"

    @patch("app.orchestration.nodes._get_llm")
    def test_records_latency(self, mock_get, config, sample_retrieved_chunks):
        mock_client = MagicMock()
        mock_client.generate.return_value = ("Answer text.", 450.0, 90, 40)
        mock_get.return_value = mock_client

        state = {
            "query_text": "test",
            "retrieved_chunks": sample_retrieved_chunks,
            "graph_context": [],
            "stages_completed": [],
        }
        result = generate_node(state, config)

        # the node measures the time with "perf_counter()"
        assert result["generation_ms"] > 0
