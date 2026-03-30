"""These are shared pytest fixtures for the test suite for this app

Fixtures that are defined here are available to all the test files.
Tests can use them by listing them as function params, pytest will then inject them automatically
"""

# import libraries
from __future__ import annotations
import pytest
from app.orchestration.config import OrchestrationConfig
from app.orchestration.models import PipelineState


@pytest.fixture
def config() -> OrchestrationConfig:
    """the default OrchestrationConfig (whisper_model_size="small", retrieval_top_k=10,
    llm_model="Qwen/Qwen3-Next-80B-A3B-Instruct", llm_temperature=0.0, and so forth)

    Tests can change specific fields after receiving, e.g., config.retrieval_top_k = 5
    """
    return OrchestrationConfig()


@pytest.fixture
def empty_state() -> PipelineState:
    """This is the minimal PipelineState before any node has run

    Fields:
    input_text         : the raw text query from user
    audio_path         : this is the file path to an uploaded audio file
    input_mode         : "text" or "audio", it controls whether transcription runs
    query_text         : the fnal query string used by nodes downstream
    transcription_ms   : Whisper runtime in ms
    stages_completed   : list that each node appends its name to after it finishes
    errors             : error messages gathered during pipeline run
    """
    return PipelineState(
        input_text="",
        audio_path="",
        input_mode="text",
        query_text="",
        transcription_ms=0.0,
        stages_completed=[],
        errors=[],
    )


@pytest.fixture
def sample_retrieved_chunks() -> list[dict]:
    """these are mock Weaviate results used for testing expand_graph and generate nodes

    2 chunks that mirror the structure from app/retrieval/ :
    chunk_id            : a unique hash (source_id + article + chunk_index)
    text                : regulatory text content that was embedded
    source_id           : "gdpr" or "fadp"
    instrument_type     : "statute", "guidance", or "commentary"
    jurisdiction        : "EU" or "CH"
    article             : the article label (GDPR: "Article N", FADP: "Art. N")
    section             : the section / chapter heading or None
    paragraph           : a paragraph number within the article (if available)
    cross_references    : other articles that are referenced in this chunk's text
    score               : the cosine similarity score (0.0 - 1.0), sorted in descending order
    """
    return [
        # GDPR Article 17 - Right to erasure (EU statute, score 0.92)
        {
            "chunk_id": "abc123",
            "text": "Article 17\n1. The data subject shall have the right to obtain from the controller the erasure of personal data concerning him or her without undue delay.",
            "source_id": "gdpr",
            "instrument_type": "statute",
            "jurisdiction": "EU",
            "article": "Article 17",
            "section": None,
            "paragraph": "1",
            "cross_references": ["Article 6", "Article 9"],
            "score": 0.92,
        },
        # FADP Art. 25 - Right of access (Swiss statute, score 0.85)
        # Swiss statutes use the "Art. N" format, not "Article N" like GDPR
        {
            "chunk_id": "def456",
            "text": "Art. 25\n1. Any person may request the controller to provide information as to whether personal data concerning them is being processed.",
            "source_id": "fadp",
            "instrument_type": "statute",
            "jurisdiction": "CH",
            "article": "Art. 25",
            "section": None,
            "paragraph": "1",
            "cross_references": ["Art. 26"],
            "score": 0.85,
        },
    ]


@pytest.fixture
def sample_graph_context() -> list[dict]:
    """these are mock Neo4j results used for testing the generate node

    Every entry represents a subgraph around an article reference found in
    the retrieved chunks. This is actually what distinguishes GraphLex AI from basic RAG, i.e.,
    structured knowledge (definitions, crossreferences, obligations) is
    layered on top of the vector search results

    Entry structure:
    type             :  "article_context" -> triggered by an article ref in chunks
    ref              :  the graph node ID looked up: "source_id:article_label"
    nodes            :  article, definition, obligation, or other entity nodes
    relationships    :  the edges between the nodes (type, from, to)
    """
    return [
        {
            "type": "article_context",
            "ref": "gdpr:Article 17",
            "nodes": [
                # article node
                {
                    "node_id": "gdpr:Article 17",
                    "article_label": "Article 17",
                    "source_id": "gdpr",
                    "full_text": "Right to erasure ('right to be forgotten')...",
                },
                # definition node - "personal data" as defined in GDPR Article 4
                {
                    "node_id": "gdpr:def:personal data",
                    "term": "personal data",
                    "definition_text": "any information relating to an identified or identifiable natural person",
                },
            ],
            "relationships": [
                # article 17 uses the legally defined term "personal data"
                {
                    "type": "DEFINES",
                    "from": "gdpr:Article 17",
                    "to": "gdpr:def:personal data",
                },
                # article 17 crossreferences Article 6 (lawful bases for processing)
                {
                    "type": "REFERENCES",
                    "from": "gdpr:Article 17",
                    "to": "gdpr:Article 6",
                },
            ],
        },
    ]
