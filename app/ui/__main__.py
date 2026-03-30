"""This is the CLI entry point "python -m app.ui"

Usage:
python -m app.ui             # launch on the default port (7860)
python -m app.ui --port 8080 # custom port
python -m app.ui --share     # create a public Gradio link

Prerequisites (local development):
- Docker containers running "docker compose up -d"
- Weaviate data has been ingested: "python -m app.retrieval ingest"
- Neo4j data has been built: "python -m app.graph build --recreate"
- a valid ".env" file with the API keys
"""

# import libraries
from __future__ import annotations
import argparse
from pathlib import Path


def _load_env() -> None:
    """This loads .env from the project root if it exists

    It uses "python-dotenv" and does not override vars already set in the shell
    """
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        from dotenv import load_dotenv

        load_dotenv(env_path)


def main() -> None:
    """parses the CLI arguments, builds the Gradio app, and launches the server"""

    # loads .env before "create_app()" so that the API keys are in "os.environ" at init time
    _load_env()

    parser = argparse.ArgumentParser(
        prog="python -m app.ui",
        description="GraphLex AI - Gradio web interface",
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port to serve on (default is 7860)"
    )
    parser.add_argument(
        "--share", action="store_true", help="Creates a public Gradio share link"
    )
    # format is "username:password"
    parser.add_argument(
        "--auth", type=str, default=None, help="username:password for basic auth"
    )
    args = parser.parse_args()

    # deferred imports
    import gradio as gr
    from app.ui.app import create_app

    app = create_app()

    # splits on the first colon only
    auth = None
    if args.auth:
        parts = args.auth.split(":", 1)
        if len(parts) == 2:
            auth = (parts[0], parts[1])

    from app.ui.app import build_theme

    app.launch(
        server_port=args.port,
        share=args.share,
        auth=auth,
        theme=build_theme(),
    )


if __name__ == "__main__":
    main()
