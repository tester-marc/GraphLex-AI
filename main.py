"""GraphLex AI: Hugging Face Spaces entry point

This builds and launches the Gradio web application

env variables (i.e., API keys, database URLs) are set as Hugging Face Spaces Secrets
(locally they are loaded from .env through the CLI entry point ("python -m app.ui")
"""

# import libraries
import os
import gradio as gr

# this builds the full Gradio Blocks app (i.e., tabs, inputs, outputs, event handlers)
# It returns a "gr.Blocks" object and does not start the web server.
from app.ui.app import create_app, build_theme

demo = create_app()

# optional HTTP basic auth via "GRADIO_AUTH" env var ("username:password")
auth = None
auth_str = os.getenv("GRADIO_AUTH", "")
if auth_str and ":" in auth_str:
    # maxsplit=1 allows colons inside passwords (e.g., "admin:p@ss:word")
    user, pwd = auth_str.split(":", 1)
    auth = (user, pwd)

# this start the web server
demo.launch(
    auth=auth,
    # theme with coloration
    theme=build_theme(),
)
