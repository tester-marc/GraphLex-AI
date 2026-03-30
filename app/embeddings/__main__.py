"""this CLI entry point works as follows:
python -m app.embeddings [--compare] [--model MODEL] [--list-queries]

--compare:            this runs the full embedding comparison for all 5 model configs,
                      it loads 817 chunks, then embeds with each model or uses cache,
                      then evaluates the 12 test queries, and finally saves the results to
                      "data/output/embeddings/comparison_results.json"
--compare --model M:  this runs the comparison for a single model label (see below) only
--list-queries:       this shows the test queries with categories and the relevant chunk counts

the model labels are defined in config.py:
text-embedding-3-small, text-embedding-3-small-512d, text-embedding-3-small-256d,
text-embedding-3-large, kanon-2-embedder
"""

# import libraries
from __future__ import annotations  # for making type hints lazy
import argparse  # python library for parsing command line args
import sys  # provides access to system level functions
from pathlib import Path  # in order to construct the path to the .env file
from dotenv import (
    load_dotenv,
)  # to read the .env file and set the contents as environment variables

# loads the .env file from the project root dir before the creation of any API clients
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# EMBEDDING_MODELS: for the 5 ModelConfig objects for the models to compare (name, label, dims, cost, provider)
# TEST_QUERIES: for the 12 TestQuery objects
# EmbeddingConfig: for the path helper class (where chunks, cache, and output files can be found)
from app.embeddings.config import EMBEDDING_MODELS, TEST_QUERIES, EmbeddingConfig


# main function
def main() -> None:
    """To parse the command line args and then run the operation that is requested"""
    parser = argparse.ArgumentParser(description="Embedding Model Comparison")
    # to define the 3 command line flags
    parser.add_argument(
        "--compare", action="store_true", help="Runs the full comparison"
    )
    parser.add_argument("--model", type=str, help="Runs only this model label")
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Shows the test queries and relevance stats",
    )
    # parses the command line arguments into an args namespace object
    args = parser.parse_args()

    # this creates an EmbeddingConfig instance with the default project root
    config = EmbeddingConfig()

    # in order to require at least 1 action flag
    if not any([args.compare, args.list_queries]):
        parser.print_help()
        sys.exit(1)

    # Mode 1: list the test queries and the relevance statistics
    if args.list_queries:
        # import in order to avoid loading dependencies unless this mode here is actually used
        from app.embeddings.comparison import EmbeddingComparisonHarness

        # creates a harness and loads the corpus of chunks from disk
        harness = EmbeddingComparisonHarness(config)
        harness.load_corpus()

        # prints a header for the table
        print(f"\n{'ID':<5} {'Category':<25} {'Relevant Chunks':>15}  Query")
        print("-" * 90)

        # for each of the 12 test queries this calculates how many corpus chunks are relevant
        for q in TEST_QUERIES:
            # label_relevance() returns a dict: {chunk_id: relevance_level} (2 is highly relevant, 1 is relevant)
            labels = harness.label_relevance(q)
            # counts how many are highly relevant vs. relevant
            lvl2 = sum(1 for v in labels.values() if v == 2)
            lvl1 = sum(1 for v in labels.values() if v == 1)
            # prints one row per query to screen
            print(
                f"{q.query_id:<5} {q.category:<25} {lvl2:>5} hi + {lvl1:>3} rel  {q.text[:50]}"
            )

        return

    # Mode 2: to run full embedding comparison

    if args.compare:
        # import in order to avoid loading dependencies unless this mode here is actually used
        from app.embeddings.comparison import EmbeddingComparisonHarness

        # creates a harness and loads the corpus of chunks from disk
        harness = EmbeddingComparisonHarness(config)
        harness.load_corpus()

        # this determines which models to run (all 5 models by default from config.py)
        models = EMBEDDING_MODELS

        # filter to one model if the user specified it
        if args.model:
            models = [m for m in EMBEDDING_MODELS if m.label == args.model]
            # if no model matches this prints an error with the list of available models
            if not models:
                print(f"Unknown model: {args.model}")
                print(f"Available: {', '.join(m.label for m in EMBEDDING_MODELS)}")
                sys.exit(1)

        print("\n Embedding Comparison: ")
        # runs the embedding model comparison
        results = harness.run(models)

        if results:
            # this collects cached corpus statistics (tokens, embedding time) per model for the aggregation
            # run() only returns the per query results so they need to be read back from cache
            corpus_stats = {}
            for model in models:
                # accesses cache file for this model
                cached = harness._load_cache(model)
                if cached:
                    # stores as a tuple with tokens and latency by model label
                    corpus_stats[model.label] = (
                        cached["total_tokens"],
                        cached["embed_latency_ms"],
                    )

            # in order to aggregate the per query results into summaries per model
            # returns a list of EmbeddingComparisonResult objects per model
            aggregated = harness.aggregate(results, corpus_stats)

            # prints the aggregated results as a table
            print("\n Aggregate Results: ")
            print(harness.format_results(aggregated))

            # saves the per query as well as the aggregated results to JSON file
            output_path = harness.save_results(results, aggregated)
            print(f"\n Results saved to: {output_path}")


# entry point for this script
if __name__ == "__main__":
    main()
