# __main__.py: this is the CLI entry point for the LLM comparison
#
# Usage:
# python -m app.llm --compare
# python -m app.llm --compare --model "Qwen3-Next 80B-A3B"
# python -m app.llm --list-queries
#
# This compares the LLMs "Llama 3.3 70B" vs. "Qwen3-Next 80B-A3B" on eight
# benchmark queries (5 well evidenced and 3 underevidenced), and measures: citation
# precision/recall, calibration accuracy, insufficient evidence detection,
# latency, and the cost
#
# Ultimately, Qwen3-Next was selected for production use
#

"""the CLI entry point: python -m app.llm [--compare] [--model MODEL] [--list-queries]"""

# import libraries
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

# this loads .env file from the project root before any app imports
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from app.llm.config import BENCHMARK_QUERIES, LLM_MODELS, LLMConfig


def main() -> None:
    """This is to parse the CLI arguments and go to the appropriate actions"""

    parser = argparse.ArgumentParser(description="LLM model comparison")
    parser.add_argument(
        "--compare", action="store_true", help="Run the full comparison"
    )
    parser.add_argument("--model", type=str, help="Run only this model label")
    parser.add_argument(
        "--list-queries", action="store_true", help="Show the benchmark queries"
    )

    args = parser.parse_args()
    config = LLMConfig()

    if not any([args.compare, args.list_queries]):
        parser.print_help()
        sys.exit(1)

    if args.list_queries:
        # prints a table of the benchmark queries
        # Y/N indicates whether or not the corpus has sufficient evidence to answer
        print(f"\n{'ID':<5} {'Category':<20} Query")
        print("-" * 80)
        for q in BENCHMARK_QUERIES:
            suff = "Y" if q.evidence_sufficient else "N"
            print(f"{q.query_id:<5} {q.category:<20} [{suff}] {q.text[:55]}")
        return

    if args.compare:
        # LLMComparisonHarness has some heavy dependencies,
        # so it's only imported when "--compare" is actually requested
        from app.llm.comparison import LLMComparisonHarness

        harness = LLMComparisonHarness(config)
        harness.load_corpus()  # the chunks from PyMuPDF extraction
        harness.load_embeddings()  # precalculated text-embedding-3-large vectors

        models = LLM_MODELS
        if args.model:
            models = [m for m in LLM_MODELS if m.label == args.model]
            if not models:
                print(f"Unknown model: {args.model}")
                print(f"Available: {', '.join(m.label for m in LLM_MODELS)}")
                sys.exit(1)

        print("\n LLM Comparison ")
        results = harness.run(models)

        if results:
            aggregated = harness.aggregate(results)
            print("\n Aggregate Results ")
            print(harness.format_results(aggregated))
            output_path = harness.save_results(aggregated)
            print(f"\n Results saved to {output_path}")


if __name__ == "__main__":
    main()
