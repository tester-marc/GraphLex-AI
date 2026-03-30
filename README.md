# GraphLex AI

Regulatory compliance intelligence prototype for Swiss-domiciled companies. Combines Retrieval-Augmented Generation with a regulatory knowledge graph to answer questions about EU GDPR and Swiss FADP data protection law.

---

## What It Does

Ask a question about Swiss or EU data protection regulation — by text or voice — and get a grounded answer that cites its sources, distinguishes statute from guidance, and flags when evidence is insufficient.

**Example queries:**
- "What does GDPR Article 17 say about the right to erasure?"
- "How does the Swiss FADP equivalent of GDPR consent requirements differ?"
- "What technical measures does the FDPIC recommend for data security?"

Every answer traces back to source material. The system will say "insufficient evidence" rather than hallucinate.

---

## Architecture

Seven-layer pipeline orchestrated with LangGraph:

```
Audio ──► ffmpeg ──► Whisper ──┐
                               ├──► Interpret ──► Retrieve (Weaviate) ──► Expand (Neo4j) ──► Generate (Qwen3) ──► Answer
Text ──────────────────────────┘
```

| Layer | Purpose | Technology |
|-------|---------|------------|
| 1. Ingestion | PDF extraction preserving legal structure | PyMuPDF |
| 2. Voice Input | Audio transcription for voice queries | OpenAI Whisper (small) |
| 3. Retrieval | Semantic vector search with metadata filtering | Weaviate + OpenAI text-embedding-3-large |
| 4. Graph | Regulatory knowledge graph (entities, relationships, obligations) | Neo4j |
| 5. Orchestration | Stateful pipeline with conditional routing | LangGraph |
| 6. Generation | Grounded answer generation with citation | Qwen3-Next 80B-A3B (Together AI) |
| 7. UI | Tabbed web interface (Answer, Evidence, Graph, Diagnostics) | Gradio |

### Knowledge Graph

845 nodes and 1,190 relationships covering:

- **6** regulatory instruments (GDPR, FADP, EDPB/FDPIC guidance)
- **174** articles (100 GDPR + 74 FADP)
- **33** legal definitions
- **632** LLM-extracted obligations
- **14** cross-jurisdictional FADP-GDPR equivalence mappings
- **237** cross-reference links between provisions

---

## Model Comparisons

Four comparisons were conducted, each selecting a production model:

| Comparison | Models Tested | Selected | Reason |
|------------|--------------|----------|--------|
| Document understanding | PyMuPDF vs olmocr vs Mistral Document AI | **PyMuPDF** | Matched VLMs on accuracy, 96x faster |
| Voice transcription | Whisper tiny/base/small/medium vs Voxtral Mini | **Whisper small** | Matched medium on all metrics at 3x lower latency |
| Embeddings | OpenAI text-embedding-3-small/large (+ MRL variants) vs kanon-2-embedder | **text-embedding-3-large** | Higher P@5 (0.600 vs 0.583), 5.8x lower latency |
| LLM generation | Llama 3.3 70B vs Qwen3-Next 80B-A3B | **Qwen3-Next 80B-A3B** | Higher citation recall (0.800 vs 0.733), half the cost |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for local Weaviate + Neo4j)
- ffmpeg (for voice input)
- API keys (see [Environment Variables](#environment-variables))

### Install

```bash
git clone https://github.com/tester-marc/GraphLex-AI.git
cd graphlex-ai
pip install -r requirements.txt
```

### Start databases

```bash
docker compose up -d
```

This launches Weaviate (port 8080) and Neo4j (port 7687) with persistent volumes.

### Ingest data

```bash
# Extract and chunk regulatory PDFs
python -m app.ingestion --extract

# Embed chunks and load into Weaviate
python -m app.retrieval ingest

# Build the knowledge graph in Neo4j
python -m app.graph build --recreate
python -m app.graph build-obligations --from-cache
```

### Run

```bash
python -m app.ui
```

Opens Gradio at `http://localhost:7860`. Optional flags: `--port PORT`, `--share`, `--auth user:pass`.

---

## CLI Reference

Each layer has its own CLI entry point:

```bash
# Document extraction and comparison
python -m app.ingestion --extract
python -m app.ingestion --compare

# Voice transcription comparison
python -m app.voice --all

# Embedding model comparison
python -m app.embeddings --compare
python -m app.embeddings --list-queries

# Vector store management
python -m app.retrieval ingest
python -m app.retrieval query "your question"
python -m app.retrieval status
python -m app.retrieval delete

# Knowledge graph management
python -m app.graph build --recreate
python -m app.graph build-obligations --from-cache
python -m app.graph status
python -m app.graph article "GDPR" "Article 17"
python -m app.graph refs "GDPR" "Article 17"
python -m app.graph defs "GDPR"
python -m app.graph guidance "GDPR" "Article 6"
python -m app.graph delete

# Pipeline queries (without UI)
python -m app.orchestration query "What does GDPR Article 17 require?"
python -m app.orchestration query --audio path/to/audio.wav
python -m app.orchestration graph

# Web UI
python -m app.ui
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Required API keys
OPENAI_API_KEY=         # Embeddings (text-embedding-3-large)
TOGETHER_API_KEY=       # LLM generation (Qwen3-Next 80B-A3B)

# Optional API keys (for model comparisons)
MISTRAL_API_KEY=        # Mistral Document AI comparison
HF_TOKEN=              # HuggingFace access
ISAACUS_API_KEY=       # kanon-2-embedder comparison

# Local databases (defaults for Docker Compose)
WEAVIATE_URL=http://localhost:8080
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=graphlex2026
```

---

## Testing

94 unit and integration tests across 7 test files:

```bash
python -m pytest tests/ -v
```

| Test File | Tests | Coverage |
|-----------|-------|----------|
| test_interpret_node.py | 18 | Article ref extraction, jurisdiction/source detection |
| test_retrieve_node.py | 7 | Chunk serialization, filter pass-through, error handling |
| test_expand_graph_node.py | 8 | Graph expansion, cross-jurisdictional, guidance citations |
| test_generate_node.py | 12 | Context prompt building, confidence detection |
| test_chunker.py | 15 | Regex patterns, statute/guidance splitting |
| test_data_models.py | 15 | SearchResult, ArticleNode, ObligationNode, Chunk IDs |
| test_integration.py | 7 | End-to-end flows, routing, cross-jurisdictional pipeline |

---

## Project Structure

```
graphlex-ai/
├── app/
│   ├── ingestion/          # Layer 1: PDF extraction + chunking
│   ├── voice/              # Layer 2: Voice transcription
│   ├── embeddings/         # Layer 3: Embedding comparison
│   ├── retrieval/          # Layer 4: Weaviate vector search
│   ├── graph/              # Layer 5: Neo4j knowledge graph
│   ├── orchestration/      # Layer 6: LangGraph pipeline
│   └── ui/                 # Layer 7: Gradio web interface
├── data/
│   ├── documents/          # 6 regulatory PDFs (GDPR, FADP, EDPB, FDPIC)
│   ├── ground_truth/       # Manual annotations for evaluation
│   └── output/             # Extraction outputs, embeddings cache, graph cache
├── tests/                  # Unit + integration tests (94 tests)
├── docker-compose.yml      # Local Weaviate + Neo4j
├── main.py                 # HF Spaces entry point
├── requirements.txt        # Python dependencies
└── packages.txt            # System packages (ffmpeg) for HF Spaces
```

---

## Regulatory Documents

The system processes six publicly available documents:

| Document | Jurisdiction | Type | Pages |
|----------|-------------|------|-------|
| GDPR Full Text | EU | Statute | 88 |
| Swiss FADP (Revised 2023) | Switzerland | Statute | 42 |
| EDPB Guidelines on Legitimate Interest | EU | Regulator Guidance | 64 |
| EDPB Guidelines on Article 48 Transfers | EU | Regulator Guidance | 38 |
| EDPB Guidelines on Consent | EU | Regulator Guidance | 38 |
| FDPIC Guide on Technical Measures | Switzerland | Regulator Guidance | 12 |

---

## License

This project is released under the MIT License.
The regulatory documents processed (GDPR, Swiss FADP, EDPB and FDPIC guidelines) are publicly available. All third-party libraries are used under their respective licenses.
