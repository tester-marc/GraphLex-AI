"""
LangGraph stateful pipeline for regulatory QA

Conditional routing bypasses transcription for the text input
"""

# import libraries
from __future__ import annotations
import time
from typing import Literal
from langgraph.graph import END, START, StateGraph
from app.orchestration.config import OrchestrationConfig
from app.orchestration.models import PipelineState
from app.orchestration.nodes import (
    close_resources,
    expand_graph_node,
    generate_node,
    interpret_node,
    retrieve_node,
    transcribe_node,
)


class OrchestrationPipeline:
    """this is the LangGraph based orchestration for the full regulatory QA pipeline

    It connects to 3 external services:
    1. Weaviate (vector database) for semantic search over regulation chunks
    2. Neo4j (graph database) for structured legal relationships
    3. Together AI (LLM API) for answer generation via Qwen3-Next LLM

    Usage:
    pipeline = OrchestrationPipeline()
    result = pipeline.run_text("What does GDPR Article 17 say about erasure?")
    result = pipeline.run_audio("/path/to/query.wav")
    pipeline.close()
    """

    def __init__(self, config: OrchestrationConfig | None = None) -> None:
        """initializes the pipeline

        Args:
        config: optional config override, defaults to "OrchestrationConfig()"
        (Whisper small, top-10 retrieval, Qwen3-Next, temperature 0.0)

        This builds and compiles the LangGraph StateGraph
        """
        self.config = config or OrchestrationConfig()
        self._graph = self._build_graph()

    # Graph construction

    def _build_graph(self) -> StateGraph:
        """to build and compile the LangGraph StateGraph

        It defines nodes, edges, and conditional routing, then compiles into
        an executable graph

        Returns:
        a compiled StateGraph ready to process queries

        """
        config = self.config

        # wrappers inject config into each node via closure, since
        # LangGraph node functions must accept only
        def _transcribe(state: PipelineState) -> dict:
            return transcribe_node(state, config)

        def _interpret(state: PipelineState) -> dict:
            return interpret_node(state, config)

        def _retrieve(state: PipelineState) -> dict:
            return retrieve_node(state, config)

        def _expand_graph(state: PipelineState) -> dict:
            return expand_graph_node(state, config)

        def _generate(state: PipelineState) -> dict:
            return generate_node(state, config)

        builder = StateGraph(PipelineState)

        builder.add_node("transcribe", _transcribe)  # audio -> text (Whisper)
        builder.add_node("interpret", _interpret)  # parsea article refs, filters
        builder.add_node("retrieve", _retrieve)  # the vector search in Weaviate
        builder.add_node("expand_graph", _expand_graph)  # Neo4j knowledge graph lookups
        builder.add_node(
            "generate", _generate
        )  # the LLM answer generation (using Qwen3-Next)

        # route at START, the audio goes through transcription, the text skips it
        builder.add_conditional_edges(
            START,
            self._route_input,
            {
                "audio": "transcribe",
                "text": "interpret",
            },
        )

        builder.add_edge("transcribe", "interpret")
        builder.add_edge("interpret", "retrieve")
        builder.add_edge("retrieve", "expand_graph")
        builder.add_edge("expand_graph", "generate")
        builder.add_edge("generate", END)

        return builder.compile()

    @staticmethod
    def _route_input(state: PipelineState) -> Literal["audio", "text"]:
        """routes to transcription for audio input, skips for text

        Args:
        state: the initial pipeline state from run_text() or run_audio()

        Returns:
        "audio" if input_mode is "audio" and audio_path is set, else returns "text"
        """
        if state.get("input_mode") == "audio" and state.get("audio_path"):
            return "audio"
        return "text"

    # Public API

    def run_text(self, query: str) -> PipelineState:
        """to run the pipeline with a text query

        Args:
        query: The user's question, e.g., "What does GDPR Article 17 say about the right to erasure?"

        Returns:
        completed PipelineState with key fields:
        - "answer": LLM generated response with citations
        - "confidence": "sufficient" or "insufficient_evidence"
        - "retrieved_chunks": relevant passages from Weaviate
        - "graph_context": the structured data from Neo4j
        - "stages_completed": the stages that ran successfully
        - "total_ms": end-to-end latency in ms
        - "errors": any errors that occurred
        """
        initial_state: PipelineState = {
            "input_text": query,
            "audio_path": "",
            "input_mode": "text",
            "query_text": query,
            "transcription_ms": 0.0,
            "stages_completed": [],
            "errors": [],
        }

        start = time.perf_counter()
        result = self._graph.invoke(initial_state)
        result["total_ms"] = (time.perf_counter() - start) * 1000
        return result

    def run_audio(self, audio_path: str) -> PipelineState:
        """this runs the pipeline with an audio file

        It supports any format ffmpeg can decode (WAV, MP3, M4A, FLAC, OGG)
        the file is preprocessed to 16kHz mono WAV before Whisper transcription

        Args:
        audio_path: the absolute path to the audio file, e.g., "/tmp/recording.wav"

        Returns:
        the completed PipelineState (same as run_text()), plus additionally:
        - "transcription_ms": the time spent on audio transcription
        - "query_text": the transcribed text from Whisper
        """
        initial_state: PipelineState = {
            "input_text": "",
            "audio_path": audio_path,
            "input_mode": "audio",
            "query_text": "",
            "transcription_ms": 0.0,
            "stages_completed": [],
            "errors": [],
        }

        start = time.perf_counter()
        result = self._graph.invoke(initial_state)
        result["total_ms"] = (time.perf_counter() - start) * 1000
        return result

    def close(self) -> None:
        """to release persistent Weaviate and Neo4j connections

        this should be called on application shutdown
        """
        close_resources()

    # Formatting

    @staticmethod
    def format_result(result: PipelineState) -> str:
        """to format a pipeline result for the console display

        It is used by "__main__.py" for the CLI output

        the Gradio UI handles its own formatting independently

        Args:
        result: a completed PipelineState from run_text() or run_audio()

        Returns:
        multi-line string showing the following:
        - input mode and query text
        - confidence and the generated answer
        - stages completed, chunk/context counts
        - the latency breakdown (transcription, retrieval, graph, generation, total)
        - any errors
        """
        lines: list[str] = []

        mode = result.get("input_mode", "?")
        query = result.get("query_text", "")
        lines.append(f"Mode: {mode}")
        lines.append(f"Query: {query}")
        lines.append("")

        answer = result.get("answer", "(no answer)")
        confidence = result.get("confidence", "?")
        lines.append(f"Confidence: {confidence}")
        lines.append(f"\n{answer}")

        lines.append("\n Diagnostics ")

        # stages_completed entries are "stage_name" on success or "stage_name:error" on failure
        stages = result.get("stages_completed", [])
        lines.append(f"Stages: {' → '.join(stages)}")

        chunks = result.get("retrieved_chunks", [])
        lines.append(f"Retrieved chunks: {len(chunks)}")

        graph_ctx = result.get("graph_context", [])
        lines.append(f"Graph context items: {len(graph_ctx)}")

        lines.append(f"\nTranscription: {result.get('transcription_ms', 0):.0f} ms")
        lines.append(f"Retrieval:     {result.get('retrieval_ms', 0):.0f} ms")
        lines.append(f"Graph:         {result.get('graph_ms', 0):.0f} ms")
        lines.append(f"Generation:    {result.get('generation_ms', 0):.0f} ms")
        lines.append(f"Total:         {result.get('total_ms', 0):.0f} ms")

        errors = result.get("errors", [])
        if errors:
            lines.append(f"\nErrors: {errors}")

        return "\n".join(lines)
