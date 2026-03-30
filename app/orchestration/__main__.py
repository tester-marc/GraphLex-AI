"""This is the CLI entry point for the "app.orchestration" package

Usage:
python -m app.orchestration query "What does GDPR Article 17 say?"
python -m app.orchestration query --audio path/to/audio.wav
python -m app.orchestration graph # to print the pipeline graph structure
"""

# import libraries
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def _load_env() -> None:
    """This loads .env from the project root if it is present

    Required variables:
    TOGETHER_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY

    In the deployment with Hugging Face Spaces these are set with the Spaces secrets UI.
    """
    # __file__ = app/orchestration/__main__.py -> parents[2] = project root
    env_path = Path(__file__).resolve().parents[2] / ".env"

    if env_path.exists():
        from dotenv import load_dotenv

        # doesn't override vars that are already set in the environment
        load_dotenv(env_path)


def main() -> None:
    """This is the main CLI entry point, it parses arguments and dispatches to handlers"""
    _load_env()

    parser = argparse.ArgumentParser(
        prog="python -m app.orchestration",
        description="GraphLex AI Orchestration Pipeline",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand: "query"
    q_parser = sub.add_parser("query", help="Runs a query through the pipeline")
    q_parser.add_argument("text", nargs="?", default="", help="Text query")
    q_parser.add_argument("--audio", type=str, default="", help="Path to audio file")
    # more chunks give the LLM more context, but also increase latency and the token cost
    q_parser.add_argument("--top-k", type=int, default=10, help="Retrieval top-k")

    # Subcommand: "graph"
    sub.add_parser("graph", help="Prints the pipeline graph structure")

    args = parser.parse_args()

    if args.command == "graph":
        _print_graph()
    elif args.command == "query":
        _run_query(args)


def _print_graph() -> None:
    """This print the pipeline graph structure as ASCII"""
    # in order to fix the Windows console encoding for Unicode chars for box drawing
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    from app.orchestration.pipeline import OrchestrationPipeline

    pipeline = OrchestrationPipeline()
    graph = pipeline._graph
    print("\n Pipeline Graph Structure ")

    try:
        print(graph.get_graph().draw_ascii())
    except Exception:
        print(" START")
        print("   |-- [audio] -> transcribe -> interpret")
        print("   +-- [text]  -> interpret")
        print(" interpret -> retrieve -> expand_graph -> generate")
        print(" generate -> END")
    print()


def _run_query(args: argparse.Namespace) -> None:
    """To run a query through the pipeline and print the result

    Args:
    args: the parsed CLI arguments (args.text, args.audio, args.top_k)
    """
    # in order to fix Windows console encoding for Unicode chars in regulatory texts
    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    # lazy imports
    from app.orchestration.config import OrchestrationConfig
    from app.orchestration.pipeline import OrchestrationPipeline

    config = OrchestrationConfig(retrieval_top_k=args.top_k)
    # Weaviate, Neo4j, and Together AI connections are made on
    # the first query and not at construction time
    pipeline = OrchestrationPipeline(config)

    print("\n GraphLex AI - Regulatory QA Pipeline \n")

    try:
        if args.audio:
            print(f"Input: audio ({args.audio})")
            result = pipeline.run_audio(args.audio)
        elif args.text:
            print(f"Input: text")
            result = pipeline.run_text(args.text)
        else:
            print("Error: please provide a text query or --audio path")
            sys.exit(1)

        print()
        print(pipeline.format_result(result))
    finally:
        # always close Weaviate and Neo4j connections
        pipeline.close()


if __name__ == "__main__":
    main()
