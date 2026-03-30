"""
This is the configuration script for the graph layer:

This defines the settings in order to connect to Neo4j (which is Layer 4 in the GraphLex AI architecture),
which stores the regulatory knowledge graph (for instruments, articles, definitions, obligations,
and their relationships).

There are 2 modes of deployment that were used:
- local development: with Neo4j Community Edition in Docker (bolt://localhost:7687)
- and for production (Hugging Face Spaces): with a Neo4j AuraDB cloud instance (neo4j+ssc://...)

Environment variables: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""

# import libraries
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GraphConfig:
    """
    This class is for the Neo4j connection and the graph settings

    It contains the connection details (URI, credentials, database name) and the file system
    paths (for chunked documents, graph layer outputs). The credentials are read in from
    the env variables if they are not passed explicitly.

    config = GraphConfig() # uses the env vars defaults
    config = GraphConfig(neo4j_uri="...", ...) # to override
    """

    # connection fields:
    # these are filled in from env vars in __post_init__ if they are left empty
    neo4j_uri: str = ""  # e.g., "bolt://localhost:7687" or "neo4j+ssc://..."
    neo4j_user: str = ""  # this defaults to "neo4j"
    neo4j_password: str = ""
    database: str = "neo4j"  # the Neo4j Community Edition only supports "neo4j"

    # this is for the absolute path to the project root resolved at runtime
    # Path(__file__).resolve().parents[2]: config.py -> app/graph/ -> app/ -> project root
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    def __post_init__(self) -> None:
        """
        This function fills in any empty connection fields from the env vars after init

        Precedence:
        constructor argument -> env var -> fallback that is hard coded
        """
        if not self.neo4j_uri:
            self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        if not self.neo4j_user:
            self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        if not self.neo4j_password:
            self.neo4j_password = os.getenv("NEO4J_PASSWORD", "")

    @property
    def chunks_dir(self) -> Path:
        """
        This function is for the directory that contains the chunked document output
        from the PyMuPDF extractor

        Every subfolder (i.e., 1 per regulatory document) contains a chunks.json,
        a extraction_summary.json, and a full_text.txt. PyMuPDF was selected in the end
        after a comparison with the VLM-based AI models "olmocr" and "Mistral Document AI"
        because it matched or even exceeded both while being much faster

        Returns:
        the path to <project_root>/data/output/pymupdf/
        """
        return self.project_root / "data" / "output" / "pymupdf"

    @property
    def output_dir(self) -> Path:
        """
        This function is for the directory where the graph layer outputs are saved

        It contains "build_stats.json" and "obligations_cache.json" (obligations from
        the statute articles were extracted via the LLM "Qwen3-Next 80B-A3B" and then
        cached to avoid calling the LLM again for rebuilds)

        Returns:
        the path to <project_root>/data/output/graph/
        """
        return self.project_root / "data" / "output" / "graph"

    def is_configured(self) -> bool:
        """
        This function returns true if the minimum credentials that are required (URI and password) are set

        config = GraphConfig()
        if not config.is_configured():
        print("Neo4j credentials missing - set NEO4J_URI and NEO4J_PASSWORD")
        sys.exit(1)
        """
        return bool(self.neo4j_uri and self.neo4j_password)
