"""this is for the pipeline state and the data models for the orchestration layer

The PipelineState is the shared state dict that flows through the LangGraph pipeline

it is used in: pipeline.py (init and invoke), nodes.py (every node reads and writes),
app.py (render the UI tabs), conftest.py (the test fixtures)
"""

# import libraries
from __future__ import annotations  # for type hints
from typing import TypedDict


class PipelineState(TypedDict, total=False):
    """the state carried through the LangGraph pipeline

    Fields are populated as each node executes
    "total=False" makes all fields optional because the state is built up in increments
    """

    # input
    input_text: str  # the raw text query (empty if it's audio input)
    audio_path: str  # the path to the audio file (empty if it's text input)
    input_mode: str  # "text" -> skip transcribe, "audio" -> run the transcribe first

    # after transcription
    query_text: str  # the final query text used by all the downstream nodes
    transcription_ms: float  # the time for ffmpeg pre-processing and Whisper inference (it is 0.0 for text input)

    # after interpretation
    search_query: str  # the query text sent to Weaviate
    article_refs: list[
        str
    ]  # graph node IDs extracted from the query, e.g., ["gdpr:Article 17"]
    source_filters: list[
        str
    ]  # the source IDs for the Weaviate filter, e.g., ["gdpr"], ["fadp"], []
    jurisdiction_filters: list[
        str
    ]  # the jurisdiction filter for Weaviate, e.g., ["EU"], ["CH"], []

    # after retrieval
    retrieved_chunks: list[dict]  # top-k Weaviate results (plain dicts)
    # Keys: chunk_id, text, source_id, instrument_type,
    #       jurisdiction, article, section, paragraph,
    #       cross_references, score (cosine 0.0–1.0)
    retrieval_ms: float  # the time for OpenAI embedding and the Weaviate search

    # after graph expansion
    graph_context: list[dict]  # Neo4j results, each item has a "type" key:
    # "article_context"     : article + definitions/obligations/refs
    # "equivalent_article"  : cross-jurisdictional GDPR <-> FADP mapping
    # "guidance_citations"  : the guidance docs citing the article
    # this is used in the Evidence tab, Graph tab (pyvis), and the LLM prompt
    graph_ms: float  # the time for the Neo4j Cypher queries

    # after generation
    answer: str  # the LLM generated answer
    # it falls back to "Insufficient evidence" rather than hallucinating
    confidence: str  # "sufficient" or "insufficient_evidence", scanned from answer text
    generation_ms: float  # the round trip time to Together AI (i.e., the prompt and the streamed response)

    # diagnostics
    stages_completed: list[str]  # stage names
    # e.g., ["interpret", "retrieve", "expand_graph", "generate"]
    errors: list[str]  # error messages from each node
    total_ms: float  # end-to-end latency
