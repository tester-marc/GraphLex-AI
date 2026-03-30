"""The Orchestration Layer: LangGraph stateful pipeline for regulatory QA"""

# Overview of the package
#
# This is the public API for the "app.orchestration" package (Layer 5)
# It re-exports "OrchestrationPipeline" so that external code can write
# "from app.orchestration import OrchestrationPipeline"
#
# This layer puts the Voice, Retrieval, and Graph Layers and the LLM (Qwen3-Next via Together AI)
# into a stateful LangGraph pipeline. UI layer calls "run_text()" or
# "run_audio()" on the pipeline in order to process queries
#
# Package structure:
# __init__.py   : This is this file here, re-exports OrchestrationPipeline
# __main__.py   : the CLI entry point ("python -m app.orchestration")
# config.py     : OrchestrationConfig dataclass (the pipeline settings)
# models.py     : PipelineState TypedDict (the data shape flowing through the nodes)
# nodes.py      : 5 node functions: transcribe, interpret, retrieve,
#                 expand_graph, generate, as well as shared resource mgmt
# pipeline.py   : OrchestrationPipeline class (builds and runs the LangGraph graph)

# this makes "OrchestrationPipeline" available as
# "app.orchestration.OrchestrationPipeline" for rest of the project
from app.orchestration.pipeline import OrchestrationPipeline

# this limits "from app.orchestration import *"" to only the public API
__all__ = ["OrchestrationPipeline"]
