# GraphLex AI - Regulatory Compliance Intelligence Prototype
#
# This is the package marker for the "app/" package.
#

# What is GraphLex AI?
#
# GraphLex AI is a regulatory compliance intelligence prototype for Swiss-domiciled companies.
# It helps DPOs, Risk Managers, Compliance Officers, and Legal Counsel analyse GDPR
# and the Swiss FADP using 2 AI techniques:
#
# - RAG (Retrieval-augmented Generation): It retrieves relevant passages from
# regulatory documents before it generates grounded, cited answers from the source material.
# - Knowledge Graph (Neo4j): It stores regulatory entities (articles, definitions,
# obligations) and relationships (cross-references, and FADP-GDPR equivalences).
#
#
# Six-Layer Pipeline Architecture
#
# The flow passes through 6 layers:
#
# [1] Ingestion (app/ingestion/): This happens offline.
# It parses the regulatory PDFs into structured text chunks preserving the legal
# structure. 3 extractors were evaluated, and PyMuPDF was finally selected for production.
# For the Embeddings (app/embeddings/): this converts text/queries into semantic vectors.
# 5 models were evaluated and finally OpenAI's "text-embedding-3-large" (3,072dim) was
# chosen for production.
#
# [2] Voice Input (app/voice/): This happens at query time if there is audio.
# ffmpeg preprocessing (16kHz mono WAV) and then OpenAI's Whisper model (small, 244M)
# What was evaluated?: Whisper tiny/base/medium as well as Mistral Voxtral Mini
#
# [3] Retrieval (app/retrieval/): This happens at query time.
# A vector similarity search in Weaviate returns the top-k chunks.
# This supports filtering by source, jurisdiction, as well as instrument type
#
# [4] Knowledge Graph (app/graph/): This is offline for the build, and at runtime for the queries
# Neo4j graph: 6 instruments, 174 articles, 33 definitions, and 632
# obligations. This enriches the retrieved evidence with cross-references and any
# FADP/GDPR equivalences.
#
# [5] Orchestration (app/orchestration/): This happens at query time.
# LangGraph stateful pipeline: transcribe -> interpret -> retrieve
# -> expand_graph -> generate (using LLM with citations)
#
# [6] UI (app/ui/): This is the user interface that is always running for the app.
# It is a Gradio app with 4 tabs: Answer, Evidence, Graph (with pyvis), Diagnostics.
# It supports text input as well as microphone / file audio
#
# Why is this init file empty?
#
# This is because each subpackage contains its own __init__.py and exports its public API,
# e.g., from app.ingestion import PyMuPDFExtractor, LegalChunker
# from app.retrieval import WeaviateStore, RetrievalPipeline
#
# AI models that were used (Template 4.1 requires 3 or more models, over different domains)
#
# The models used are as follows:
#
# Model 1: OpenAI Whisper small (244M) for audio to text
# (app/voice/whisper_transcriber.py)
#
# Model 2: PyMuPDF (rule based pipeline tool, not actually an AI model) for document to structured text
# (app/ingestion/pymupdf_extractor.py)
# (the VLM-based AI models "olmocr" and "Mistral Document AI" were tested but rejected:
# because there was no accuracy gain on digitally authored PDFs)
#
# Model 3: OpenAI text-embedding-3-large for text to vector
# (app/embeddings/openai_embedder.py)
#
# Model 4: Qwen3-Next 80B-A3B (MoE, 3B active) for text to text
# (app/orchestration/nodes.py)
# This was performed via the Together AI API. This model was tested against "Llama 3.3 70B".
#
#
# How to Run It
#
# 1. Start the databases:        docker compose up -d
# 2. Ingest the documents:       python -m app.retrieval ingest
# 3. Build the knowledge graph:  python -m app.graph build --recreate
# 4. Extract the obligations:    python -m app.graph build-obligations --from-cache
# 5. Launch the UI:              python -m app.ui
#
# The demonstration of this prototype was deployed to Hugging Face Spaces using
# Weaviate Cloud and Neo4j AuraDB.
#
# File and Folder Structure:
#
# app/
# ├── __init__.py         : this file here
# ├── ingestion/          : Layer 1: PDF extraction and chunking
# ├── embeddings/         : Layer 1: text-to-vector conversion
# ├── voice/              : Layer 2: for audio preprocessing and transcription
# ├── retrieval/          : Layer 3: vector search (Weaviate)
# ├── graph/              : Layer 4: knowledge graph (Neo4j)
# ├── orchestration/      : Layer 5: LangGraph stateful pipeline for orchestration
# └── ui/                 : Layer 6: Web interface (Gradio)
#
# each subfolder contains:
# __init__.py
# __main__.py
# config.py
# models.py
# and impl modules
#
