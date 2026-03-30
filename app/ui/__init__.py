"""UI Layer: the Gradio web interface for GraphLex AI

This here is the "__init__.py" for the "app.ui" package (app/ui/)

It re-exports "create_app" from "app.ui.app" so callers can write:

"from app.ui import create_app"

"create_app()" builds and returns a "gr.Blocks" object whcih contains the
full Gradio UI (i.e., inputs, buttons, output tabs, examples, event handlers)
The caller then does "app.launch(...)" in order to start the web server

2 entry points call it:
- "app/ui/__main__.py"
- "main.py" (in the project root)

Ingestion and Embeddings run at ingestion time (this happens offline), and their
outputs are stored in Weaviate as well as Neo4j before the app starts up.
The UI (Layer 6) calls the Orchestration Layer (Layer 5), which then
orchstrates and calls other layers internally.
"""

# re-exports "create_app" at the package level
from app.ui.app import create_app

# __all__ limits "from app.ui import *"" to only this public name
__all__ = ["create_app"]
