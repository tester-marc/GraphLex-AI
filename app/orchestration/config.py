"""this is the configuration for the GraphLex AI orchestration pipeline

it defines "OrchestrationConfig" which is a dataclass that holds all the settings
for the pipeline, i.e., Transcribe -> Interpret -> Retrieve -> Expand Graph -> Generate
"""

# import libraries
from __future__ import annotations  # for type hints
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OrchestrationConfig:
    """these are the pipeline settings, all of the fields have defaults tuned for production (see report)"""

    # voice settings / "transcribe" node

    # this has been selected after a comparison against other models (see report)
    whisper_model_size: str = "small"

    # gives Whisper a domain vocabulary prompt (GDPR, FADP, EDPB, Art. 17, etc.)
    # before transcription
    enable_context_biasing: bool = True

    # retrieval settings / "retrieve" node

    # the chunks that are returned from Weaviate, per query
    # k=10 is to balance recall vs prompt size
    retrieval_top_k: int = 10

    # graph settings / "expand_graph" node

    # the cross-reference hops to follow in Neo4j
    # depth = 1 returns article itself with its definitions, obligations,
    # cross-references, jurisdictional equivalents, and citing guidance documents
    # depth = 2 adds too much context (numerous obligations) and makes the queries slow
    graph_ref_depth: int = 1

    # LLM settings / "generate" node

    # MoE model (80B total parameters, 3B active parameters) with the Together AI API
    # the model selected was -Instruct, which is the instruction tuned variant structured task usage
    llm_model: str = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    # the cap for output tokens.
    # 1024 provides enough room for complex multi-article answers
    llm_max_tokens: int = 1024

    # the temp here is 0.0, which is also known as greedy or deterministic.
    # this is important for a compliance tool where reproducibility matters
    # and "creativity" in the answers should be at a minimum
    llm_temperature: float = 0.0

    # paths

    # this resolves to repo root
    # it is used by other modules to locate the data files without having the paths hard coded
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    # environment check

    def is_configured(self) -> bool:
        """this returns true if the 3 important env vars are set and otherwise false

        It checks:
        WEAVIATE_URL       : the vector db endpoint (local Docker / Weaviate Cloud)
        NEO4J_PASSWORD     : the graph db password (local or Hugging Face Spaces secret)
        TOGETHER_API_KEY   : the key for Together AI for the LLM Qwen3-Next inference endpoint

        other variables (NEO4J_URI, NEO4J_USER, OPENAI_API_KEY)
        are left out from this quick "health check"
        """
        return bool(
            os.getenv("WEAVIATE_URL")
            and os.getenv("NEO4J_PASSWORD")
            and os.getenv("TOGETHER_API_KEY")
        )
