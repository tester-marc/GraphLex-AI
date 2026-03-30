"""
The CLI entry point is: python -m app.graph

Commands:

build                           : to build knowledge graph from the chunked regulatory documents
build --recreate                : to wipe the graph first and then rebuild from scratch
build-obligations               : to extract legal obligations via a LLM (used: Qwen3-Next 80B)
build-obligations --from-cache  : to load previously extracted obligations from the JSON cache
status                          : prints the node and relationship counts
article "gdpr:Article 17"       : shows article text, cross-references, and obligations
refs "gdpr:Article 17"          : shows which articles reference resp. are referenced by this one
refs "gdpr:Article 17" -d 2     : dito, but follows the references 2 hops deep
defs                            : lists all the legal term definitions
defs --source gdpr              : in order to filter to GDPR definitions only
defs --search "consent"         : to search definitions by keyword
guidance "gdpr:Article 6"       : to find EDPB/FDPIC guidance documents which cite this article
delete                          : deletes everything in the graph (can be used before a rebuild)
"""

# import libraries

from __future__ import annotations
import argparse
import sys

# in order to fix the Windows console encoding so that Unicode characters in legal text do not crash terminal
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from dotenv import load_dotenv
from app.graph.config import GraphConfig
from app.graph.pipeline import GraphPipeline


def main() -> None:
    # this must run before GraphConfig which reads NEO4J_URI,USER,PASSWORD from the env
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="GraphLex AI - Graph Layer (Neo4j regulatory knowledge graph)"
    )

    # dest="command" stores the selected subcommand name in args.command
    sub = parser.add_subparsers(dest="command")

    # this extracts entities and relationships from the chunks and inserts them into Neo4j
    # "--recreate" deletes everything first
    build_p = sub.add_parser("build", help="Build knowledge graph from chunks")
    build_p.add_argument("--recreate", action="store_true", help="Clear and rebuild")

    # "build-obligations" sends every statute article to the LLM "Qwen3-Next 80B" (via Together AI)
    # in order to extract the ObligationNode entities. Then, results are cached to obligations_cache.json
    ob_p = sub.add_parser(
        "build-obligations",
        help="Extract obligations via the LLM (requires the TOGETHER_API_KEY)",
    )
    # "--from-cache" skips the LLM step and loads cache instead
    # (so the way it works is it requires the TOGETHER_API_KEY from the env unless using "--from-cache")
    ob_p.add_argument(
        "--from-cache",
        action="store_true",
        help="Load from cache instead of calling the LLM",
    )

    # this can be used for verifying if the graph has been built correctly
    sub.add_parser("status", help="Show the graph statistics")

    # "ref" format: "source_id:article_label", e.g. "gdpr:Article 17" or "fadp:Art. 25"
    art_p = sub.add_parser("article", help="Look up an article")
    art_p.add_argument("ref", help="Article reference (e.g., 'gdpr:Article 17')")

    # this uses a Cypher path of variable length: MATCH (a)-[:REFERENCES*1..depth]-(b)
    refs_p = sub.add_parser("refs", help="Show the cross-references for an article")
    refs_p.add_argument("ref", help="Article reference (e.g., 'gdpr:Article 17')")
    refs_p.add_argument(
        "-d", "--depth", type=int, default=1, help="Traversal depth (default is 1)"
    )

    # to list the legal term definitions stored in the graph
    defs_p = sub.add_parser("defs", help="List definitions")
    defs_p.add_argument("--source", help="Filters by source_id (GDPR or FADP)")
    defs_p.add_argument("--search", help="Searches by term fragment")

    # to find guidance documents that cite a specific statute article
    guid_p = sub.add_parser("guidance", help="Find guidance for an article")
    guid_p.add_argument("ref", help="Article reference (e.g., 'gdpr:Article 6')")

    # to delete all nodes and relationships from the Neo4j database
    sub.add_parser("delete", help="Clears the graph")

    # in order to parse the command line arguments
    args = parser.parse_args()

    # shows the help text and exits with error code 1 if user ran it with no subcommand
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # creates the config and checks if it's valid
    config = GraphConfig()

    if not config.is_configured():
        print("Error: NEO4J_URI or NEO4J_PASSWORD not set. Please configure in .env")
        sys.exit(1)

    # creates the pipeline and runs the command requested
    pipeline = GraphPipeline(config)

    # finally makes sure that the Neo4j connection is always closed
    try:
        if args.command == "build":
            pipeline.build(recreate=args.recreate)

        elif args.command == "build-obligations":
            pipeline.connect()
            if args.from_cache:
                pipeline.load_cached_obligations()
            else:
                pipeline.build_obligations()

        elif args.command == "status":
            stats = pipeline.status()
            print(f"\n Graph Status \n\n{stats}")

        elif args.command == "article":
            pipeline.connect()
            result = pipeline.query_article(args.ref)
            print(f"\n Article: {args.ref} \n")
            print(pipeline.format_article_context(result))

        elif args.command == "refs":
            pipeline.connect()
            result = pipeline.query_references(args.ref, args.depth)
            if result.is_empty:
                print(f" No references found for {args.ref}")
            else:
                print(f"\n References from {args.ref} (depth={args.depth}) \n")
                # in order to deduplicate from/to pairs which appear via multiple traversal paths
                seen = set()
                for r in result.relationships:
                    pair = (r.get("from"), r.get("to"))
                    if pair not in seen:
                        seen.add(pair)
                        print(f"  {r.get('from')} -> {r.get('to')}")

        elif args.command == "defs":
            pipeline.connect()
            if args.search:
                rows = pipeline.queries.search_definitions(args.search)
            else:
                rows = pipeline.queries.get_definitions(source_id=args.source)
            print(f"\n Definitions ({len(rows)}) \n")
            for row in rows:
                d = dict(row["d"])
                print(f"  [{d.get('source_id', '')}] {d.get('term', '')}")
                text = d.get("definition_text", "")[:120]
                print(f"    {text}...")
                print()

        elif args.command == "guidance":
            pipeline.connect()
            rows = pipeline.queries.get_guidance_for_article(args.ref)
            if not rows:
                print(f" No guidance found for {args.ref}")
            else:
                print(f"\n Guidance citing {args.ref} \n")
                for row in rows:
                    print(f"  [{row['source_id']}] {row['title']}")

        elif args.command == "delete":
            pipeline.connect()
            pipeline.store.clear_graph()

    finally:
        # always closes the database connection
        pipeline.close()


# __main__ guard
if __name__ == "__main__":
    main()
