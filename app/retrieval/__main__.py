"""
this is the CLI entry point: "python -m app.retrieval"

it manages Weaviate vector store from the terminal
It is used during development and deployment

Commands:
ingest [--recreate]  : load the PyMuPDF chunks and the cached text-embedding-3-large
                       embeddings and batch insert into Weaviate. "--recreate"
                       drops and then rebuilds the collection from scratch
query <text> [-k N]  : to embed query text and search Weaviate for similar chunks
                       metadata filters:
                       --source gdpr / --jurisdiction EU / --type statute
status               : shows whether RegulatoryChunk exists and its object count
delete               : this deletes the whole RegulatoryChunk collection
"""

# import libraries
from __future__ import annotations
import argparse
import sys

# in order to fix the Unicode display on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from dotenv import load_dotenv
from app.retrieval.config import RetrievalConfig
from app.retrieval.pipeline import RetrievalPipeline


def main() -> None:
    """this function is to parse the CLI arguments, validate the config, and route to the appropriate pipeline method"""

    # to populate the env vars from .env
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="GraphLex AI - Retrieval Layer (Weaviate and text-embedding-3-large)"
    )
    sub = parser.add_subparsers(dest="command")

    # ingest
    ingest_p = sub.add_parser("ingest", help="Ingests the PyMuPDF chunks into Weaviate")
    ingest_p.add_argument(
        "--recreate", action="store_true", help="To delete and recreate the collection"
    )

    # query
    query_p = sub.add_parser("query", help="Searches the regulatory corpus")
    query_p.add_argument("text", help="the query text")
    query_p.add_argument(
        "-k", "--top-k", type=int, default=10, help="number of results (default is 10)"
    )
    # action="append" allows the flag to be repeated; dest stores results as a list.
    query_p.add_argument(
        "--source", action="append", dest="sources", help="Filters by the source_id"
    )
    query_p.add_argument(
        "--jurisdiction",
        action="append",
        dest="jurisdictions",
        help="filters by jurisdiction",
    )
    query_p.add_argument(
        "--type", action="append", dest="types", help="filters by instrument_type"
    )

    # status / delete
    sub.add_parser("status", help="Shows the collection status")
    sub.add_parser("delete", help="deletes the collection")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = RetrievalConfig()

    if not config.is_configured():
        print(
            "Error: WEAVIATE_URL is not set, set WEAVIATE_URL (and WEAVIATE_API_KEY for cloud) in .env"
        )
        sys.exit(1)

    pipeline = RetrievalPipeline(config)

    try:
        if args.command == "ingest":
            # this connects, ensures the schema, loads all the chunks and cached embeddings,
            # inserts with the deterministic UUIDs and prints the final count
            pipeline.ingest(recreate=args.recreate)

        elif args.command == "query":
            pipeline.connect()
            # this embeds the query via the OpenAI API, runs the vector similarity search within Weaviate
            # filter arguments are None when not specified
            results = pipeline.query(
                text=args.text,
                top_k=args.top_k,
                source_ids=args.sources,
                jurisdictions=args.jurisdictions,
                instrument_types=args.types,
            )
            print(f"\n Top {len(results)} results \n")
            print(pipeline.format_results(results))

        elif args.command == "status":
            pipeline.connect()
            if pipeline.store.collection_exists():
                count = pipeline.store.count()  # aggregates COUNT(*)
                print(
                    f" Collection '{config.collection_name}' exists with {count} objects"
                )
            else:
                print(f" Collection '{config.collection_name}' does not exist")

        elif args.command == "delete":
            pipeline.connect()
            pipeline.store.delete_collection()

    finally:
        # always close to release the HTTP connections
        pipeline.close()


if __name__ == "__main__":
    main()
