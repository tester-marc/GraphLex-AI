"""This is for the pipeline node functions for the LangGraph orchestration

every node takes the full PipelineState, performs one stage, and then returns
a partial state update (only the keys that it modifies)

Each node function does the following:
1. it reads fields it needs from the state
2. does its work (API call, regex, database query, etc.)
3. and returns a dict of only the fields it modifies
"""

# import libraries
from __future__ import annotations
import re
import time
from dataclasses import asdict
from pathlib import Path
from app.orchestration.config import OrchestrationConfig
from app.orchestration.models import PipelineState


# Shared resources

_retrieval_pipeline = (
    None  # RetrievalPipeline: Weaviate + OpenAI embedder (used by retrieve_node)
)
_graph_pipeline = (
    None  # GraphPipeline: Neo4j + Cypher queries (used by expand_graph_node)
)
_together_client = (
    None  # TogetherClient: Together AI LLM inference (used by generate_node)
)


def _get_retrieval():
    """this gets or creates the RetrievalPipeline

    On the first call it: imports RetrievalPipeline, creates an instance, and opens
    the TCP connection to Weaviate (local Docker or Weaviate Cloud)
    """
    global _retrieval_pipeline
    if _retrieval_pipeline is None:
        from app.retrieval.pipeline import RetrievalPipeline

        _retrieval_pipeline = RetrievalPipeline()
        _retrieval_pipeline.connect()
    return _retrieval_pipeline


def _get_graph():
    """this gets or create the GraphPipeline

    On the first call it: imports GraphPipeline, creates an instance, and opens
    the Bolt connection to Neo4j (local Docker or Neo4j AuraDB)
    """
    global _graph_pipeline
    if _graph_pipeline is None:
        from app.graph.pipeline import GraphPipeline

        _graph_pipeline = GraphPipeline()
        _graph_pipeline.connect()
    return _graph_pipeline


def _get_llm():
    """This gets or creates the TogetherClient

    On the first call it: imports and creates a TogetherClient instance

    Together AI uses a stateless REST API so there is no persistent
    connection
    """
    global _together_client
    if _together_client is None:
        from app.llm.together_client import TogetherClient

        _together_client = TogetherClient()
    return _together_client


def close_resources() -> None:
    """this closes persistent connections (Weaviate and Neo4j)

    it resets globals to None so subsequent calls to "_get_retrieval()" or
    "_get_graph()" will create fresh connections if the pipeline restarts

    the Together AI client does not need closing
    """
    global _retrieval_pipeline, _graph_pipeline
    if _retrieval_pipeline is not None:
        _retrieval_pipeline.close()
        _retrieval_pipeline = None
    if _graph_pipeline is not None:
        _graph_pipeline.close()
        _graph_pipeline = None


# Node: transcribe (audio to text via Whisper small)


def transcribe_node(state: PipelineState, config: OrchestrationConfig) -> dict:
    """this transcribes audio input using Whisper small

    steps:
    1. validate that the audio file exists
    2. preprocess with ffmpeg (normalise to -16 LUFS, convert to 16kHz
       mono WAV, apply 80Hz high pass filter)
    3. run Whisper transcription with optional regulatory vocabulary
       context biasing ("FADP", "GDPR", "EDPB", etc.)

    only executed for audio input, the conditional router in "pipeline.py"
    skips this node for text input

    """
    # import libraries
    from app.voice.whisper_transcriber import transcribe
    from app.voice.preprocessing import preprocess_audio
    import tempfile

    audio_path = Path(state["audio_path"])

    if not audio_path.exists():
        return {
            "query_text": "",
            "transcription_ms": 0.0,
            "errors": state.get("errors", []) + [f"Audio file not found: {audio_path}"],
            "stages_completed": state.get("stages_completed", [])
            + ["transcribe:error"],
        }

    start = time.perf_counter()

    preprocessed = Path(tempfile.gettempdir()) / f"graphlex_{audio_path.stem}_pp.wav"

    try:
        preprocess_audio(audio_path, preprocessed)
        target = preprocessed
    except Exception:
        target = audio_path  # fall back to raw audio if ffmpeg fails

    text, whisper_ms = transcribe(
        target,
        model_size=config.whisper_model_size,  # default is "small" (244M params)
        context_biasing=config.enable_context_biasing,  # default is true
    )

    total_ms = (time.perf_counter() - start) * 1000

    return {
        "query_text": text,
        "transcription_ms": total_ms,
        "stages_completed": state.get("stages_completed", []) + ["transcribe"],
    }


# Node: interpret (extract article refs, jurisdiction filters, search query)
#
# this analyses the query via regex and keyword matching
# It runs  and extracts:
# 1. Article references    : the Neo4j node_ids (e.g., "gdpr:Article 17")
# 2. Jurisdiction filters  : Weaviate metadata filter (e.g., "EU")
# 3. Source filters        : Weaviate metadata filter (e.g., "gdpr")
#
# regex over NLP

# 3 patterns cover common ways that users reference articles:
# pattern 1: "GDPR Article 17" (source, number)
# pattern 2: "Article 17 of the GDPR" (number, source)
# pattern 3: "Article 17 GDPR" (number, source)
# all use re.IGNORECASE, \b prevents partial word matches
_ARTICLE_PATTERNS = [
    # "GDPR Article 17", "FADP Art. 6", "GDPR Art 49"
    re.compile(
        r"\b(GDPR|FADP)\s+(?:Article|Art\.?)\s*(\d+)",
        re.IGNORECASE,
    ),
    # "Article 17 of the GDPR", "Article 6 of the FADP"
    re.compile(
        r"\b(?:Article|Art\.?)\s*(\d+)\s+(?:of\s+(?:the\s+)?)?(GDPR|FADP)",
        re.IGNORECASE,
    ),
    # "Article 17 GDPR" (no "of")
    re.compile(
        r"\b(?:Article|Art\.?)\s*(\d+)\s+(GDPR|FADP)\b",
        re.IGNORECASE,
    ),
]

# this maps query keywords to Weaviate "source_ids" for the 6 corpus documents
# EDPB maps to None (multiple EDPB documents exist), i.e., "edpb" in a query
# triggers jurisdiction filtering on "EU" instead of a source_id filter
_SOURCE_MAP = {
    "gdpr": "gdpr",
    "fadp": "fadp",
    "edpb": None,  # multiple EDPB sources, filter it by jurisdiction instead
    "fdpic": "fdpic_technical_measures",
}

# this maps query keywords to jurisdiction codes used in Weaviate metadata
# "EU" : covers GDPR and all 3 EDPB guidelines
# "CH" : covers FADP and FDPIC technical measures guide
_JURISDICTION_KEYWORDS = {
    "gdpr": "EU",
    "eu": "EU",
    "european": "EU",
    "fadp": "CH",
    "swiss": "CH",
    "switzerland": "CH",
    "fdpic": "CH",
    "edpb": "EU",
}


def interpret_node(state: PipelineState, config: OrchestrationConfig) -> dict:
    """this parses the query to extract article references and filters

    outputs are used by the downstream nodes:
    - article_refs          :  expand_graph_node (Neo4j lookups)
    - source_filters        :  retrieve_node (Weaviate metadata filter)
    - jurisdiction_filters  :  retrieve_node (Weaviate metadata filter)
    """
    query = state.get("query_text", "")

    article_refs: list[str] = []
    source_filters: list[str] = []
    jurisdiction_filters: list[str] = []

    # extracts the article references
    for pattern in _ARTICLE_PATTERNS:
        for match in pattern.finditer(query):
            groups = match.groups()
            if len(groups) == 2:
                # detects the group order, pattern 1 gives (source, number)
                # patterns 2 and 3 give (number, source)
                if groups[0].isdigit():
                    num, src = groups
                else:
                    src, num = groups

                src_lower = src.lower()
                source_id = "gdpr" if src_lower == "gdpr" else "fadp"

                # article label format must match Neo4j storage
                # GDPR -> "Article N" (e.g., "Article 17")
                # FADP -> "Art. N" (e.g., "Art. 6")
                label = f"Article {num}" if source_id == "gdpr" else f"Art. {num}"
                ref = f"{source_id}:{label}"

                if ref not in article_refs:
                    article_refs.append(ref)

    # detects jurisdiction from keywords
    query_lower = query.lower()
    for keyword, jurisdiction in _JURISDICTION_KEYWORDS.items():
        if keyword in query_lower and jurisdiction not in jurisdiction_filters:
            jurisdiction_filters.append(jurisdiction)

    # detect source mentions for Weaviate filtering
    for keyword, source_id in _SOURCE_MAP.items():
        if keyword in query_lower and source_id and source_id not in source_filters:
            source_filters.append(source_id)

    # search_query mirrors query_text
    return {
        "search_query": query,
        "article_refs": article_refs,
        "source_filters": source_filters,
        "jurisdiction_filters": jurisdiction_filters,
        "stages_completed": state.get("stages_completed", []) + ["interpret"],
    }


# Node: retrieve (vector search in Weaviate)
#
# this embeds the search query into a 3072-dim vector (OpenAI "text-embedding-3-large")
# and runs top-k cosine similarity search against all chunks in Weaviate
# the filters from interpret_node are applied as Weaviate WHERE clauses


def retrieve_node(state: PipelineState, config: OrchestrationConfig) -> dict:
    """Runs vector search against Weaviate"""

    search_query = state.get("search_query", "")

    if not search_query:
        return {
            "retrieved_chunks": [],
            "retrieval_ms": 0.0,
            "errors": state.get("errors", []) + ["No search query available"],
            "stages_completed": state.get("stages_completed", []) + ["retrieve:error"],
        }

    pipeline = _get_retrieval()
    start = time.perf_counter()

    # "pipeline.query()"" embeds the query then runs a
    # nearVector search in Weaviate
    results = pipeline.query(
        text=search_query,
        top_k=config.retrieval_top_k,  # default is 10 chunks
        source_ids=state.get("source_filters") or None,
        jurisdictions=state.get("jurisdiction_filters") or None,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    # to convert SearchResult dataclasses to plain dicts
    chunks = []
    for r in results:
        chunks.append(
            {
                "chunk_id": r.chunk_id,
                "text": r.text,
                "source_id": r.source_id,
                "instrument_type": r.instrument_type,
                "jurisdiction": r.jurisdiction,
                "article": r.article,
                "section": r.section,
                "paragraph": r.paragraph,
                "cross_references": r.cross_references,
                "score": r.score,
            }
        )

    return {
        "retrieved_chunks": chunks,
        "retrieval_ms": elapsed_ms,
        "stages_completed": state.get("stages_completed", []) + ["retrieve"],
    }


# Node: expand_graph (Neo4j lookup for article context)
#
# this enriches pipeline context with structured knowledge from the Neo4j graph
#
# this node finds structured knowledge:
# - Definitions: from GDPR Art. 4 / FADP Art. 5
# - Obligations: extracted by Qwen3-Next during the graph build
# - Cross-references: articles cited by each article
# - Cross-jurisdictional equivalents: 14 hard coded GDPR <-> FADP mappings
# - Guidance citations: 100 "CITES" relationships from EDPB/FDPIC docs


def expand_graph_node(state: PipelineState, config: OrchestrationConfig) -> dict:
    """this expands the query context using the Neo4j knowledge graph"""

    graph = _get_graph()
    start = time.perf_counter()

    graph_context: list[dict] = []

    # it uses article references from interpret_node if available, otherwise
    # from the top retrieved chunks
    article_refs = state.get("article_refs", [])

    if not article_refs:
        seen = set()
        for chunk in state.get("retrieved_chunks", []):
            art = chunk.get("article")
            src = chunk.get("source_id")
            if art and src:
                ref = f"{src}:{art}"
                if ref not in seen:
                    seen.add(ref)
                    article_refs.append(ref)
                    if len(article_refs) >= 5:  # limit graph lookups
                        break

    for ref in article_refs:

        # 3a: article context, the Cypher query returns the article node plus
        # all the connected definitions, obligations, and cross-references
        ctx = graph.query_article(ref)
        if not ctx.is_empty:
            graph_context.append(
                {
                    "type": "article_context",
                    "ref": ref,
                    "nodes": ctx.nodes,
                    "relationships": ctx.relationships,
                }
            )

        # 3b: cross-jurisdictional equivalents: 14 EQUIVALENT_TO relationships
        # map the FADP articles to their GDPR counterparts (e.g., FADP Art. 32 <->
        # GDPR Art. 17)
        equivs = graph.queries.get_equivalents(ref)
        for eq in equivs:
            eq_ref = eq.get("node_id", "")
            eq_ctx = graph.query_article(eq_ref)
            if not eq_ctx.is_empty:
                graph_context.append(
                    {
                        "type": "equivalent_article",
                        "source_ref": ref,
                        "equivalent_ref": eq_ref,
                        "nodes": eq_ctx.nodes,
                        "relationships": eq_ctx.relationships,
                    }
                )

        # 3c: guidance citations: 100 "CITES" relationships link the 3 EDPB
        # guidelines and 1 FDPIC guide to the statute articles that they reference
        guidance = graph.queries.get_guidance_for_article(ref)
        if guidance:
            graph_context.append(
                {
                    "type": "guidance_citations",
                    "ref": ref,
                    "guidance": guidance,  # a list of {"source_id": ..., "title": ...}
                }
            )

    elapsed_ms = (time.perf_counter() - start) * 1000

    return {
        "graph_context": graph_context,
        "graph_ms": elapsed_ms,
        "stages_completed": state.get("stages_completed", []) + ["expand_graph"],
    }


# Node: generate (LLM answer via Qwen3-Next)
#
# the final node, it combines the retrieved passages and graph context into a single
# prompt and then sends it to Qwen3-Next on Together AI
#
# After generation it scans the response for uncertainty markers
# in order to produce a confidence label displayed in the UI

# System prompt:
# 1. answer only from provided context, prevents hallucinated provisions
# 2. cite article numbers in order for users to verify against source
# 3. say "insufficient evidence" when context doesn't support an answer
# 4. do not fabricate article numbers or provisions
# 5. indicate source document type (statute vs. guidance)
# 6. stay concise in 200-400 words
_SYSTEM_PROMPT = """\
You are a regulatory compliance assistant specialising in Swiss and EU data \
protection law (GDPR and Swiss FADP).

Rules:
1. Answer ONLY based on the provided context passages. Do not use prior knowledge.
2. Cite specific article numbers (e.g., "GDPR Article 17", "FADP Art. 25") for every claim.
3. If the provided context does not contain sufficient evidence to answer the \
question, explicitly state: "Insufficient evidence in the provided context to \
answer this question." and explain what information is missing.
4. Do not fabricate or hallucinate provisions, article numbers, or regulatory content.
5. Indicate the source document for each citation (GDPR, FADP, EDPB guidelines, FDPIC guide).
6. Keep your answer concise (200-400 words) and focused on the regulatory provisions."""

# phrases that indicate the LLM could not answer confidently (as per Rule 3)
_UNCERTAINTY_MARKERS = [
    "insufficient evidence",
    "not enough information",
    "cannot determine",
    "unable to answer",
    "no specific provision",
    "not addressed",
    "does not contain sufficient",
    "beyond the scope",
    "no information",
    "cannot find",
]


def _build_context_prompt(state: PipelineState) -> str:
    """this builds the context section of the user prompt from retrieval and graph results"""
    sections: list[str] = []

    # part 1: Vector retrieval context (from Weaviate)
    chunks = state.get("retrieved_chunks", [])
    if chunks:
        sections.append(" Retrieved Passages ")
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_id", "unknown").upper()
            article = chunk.get("article", "")
            section = chunk.get("section", "")
            loc = " > ".join(filter(None, [article, section]))
            authority = chunk.get("instrument_type", "")
            text = chunk.get("text", "")
            sections.append(
                f"[Passage {i}] Source: {source} | Location: {loc} | "
                f"Type: {authority}\n{text}"
            )

    # part 2: Knowledge graph context (from Neo4j)
    graph_ctx = state.get("graph_context", [])
    if graph_ctx:
        sections.append("\n Knowledge Graph Context ")

        for item in graph_ctx:
            ctx_type = item.get("type", "")
            ref = item.get("ref", "")

            if ctx_type == "article_context":
                nodes = item.get("nodes", [])
                rels = item.get("relationships", [])
                article_node = nodes[0] if nodes else {}
                sections.append(f"\n[Article: {ref}]")

                # definitions: "DEFINES" edges -> Definition nodes
                defs = [r for r in rels if r.get("type") == "DEFINES"]
                if defs:
                    for d_rel in defs:
                        d_node = next(
                            (n for n in nodes if n.get("node_id") == d_rel.get("to")),
                            {},
                        )
                        term = d_node.get("term", "")
                        defn = d_node.get("definition_text", "")
                        if term:
                            sections.append(f" Definition: '{term}' — {defn[:200]}")

                # obligations: "IMPOSES" edges -> Obligation nodes (capped at 5)
                obs = [r for r in rels if r.get("type") == "IMPOSES"]
                if obs:
                    sections.append(f" Obligations ({len(obs)}):")
                    for o_rel in obs[:5]:
                        o_node = next(
                            (n for n in nodes if n.get("node_id") == o_rel.get("to")),
                            {},
                        )
                        sections.append(
                            f"    [{o_node.get('obligation_type', '?')}] "
                            f"{o_node.get('description', '')[:150]}"
                        )

                # cross-references: "REFERENCES" edges (capped at 10)
                refs_out = [r for r in rels if r.get("type") == "REFERENCES"]
                if refs_out:
                    ref_targets = [r.get("to", "") for r in refs_out]
                    sections.append(f" Cross-references: {', '.join(ref_targets[:10])}")

            elif ctx_type == "equivalent_article":
                sections.append(
                    f"\n[Cross-jurisdictional equivalent: "
                    f"{item.get('source_ref', '')} ↔ {item.get('equivalent_ref', '')}]"
                )

            elif ctx_type == "guidance_citations":
                guidance = item.get("guidance", [])
                if guidance:
                    titles = [g.get("title", g.get("source_id", "")) for g in guidance]
                    sections.append(f"\n[Guidance citing {ref}]: {', '.join(titles)}")

    return "\n".join(sections)


def generate_node(state: PipelineState, config: OrchestrationConfig) -> dict:
    """this generates a grounded answer using Qwen3-Next via Together AI

    steps:
    1. builds a context prompt from retrieve and expand_graph results
    2. sends to Qwen3-Next with the system prompt
    3. scans the response for uncertainty markers
    4. returns the answer with a confidence label
    """
    query = state.get("query_text", "")
    context = _build_context_prompt(state)

    # empty context means Weaviate/Neo4j aren't running or populated
    if not context.strip():
        return {
            "answer": "No context was retrieved. Please check that Weaviate and Neo4j are running.",
            "confidence": "insufficient_evidence",
            "generation_ms": 0.0,
            "errors": state.get("errors", []) + ["No context available for generation"],
            "stages_completed": state.get("stages_completed", []) + ["generate:error"],
        }

    client = _get_llm()
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    start = time.perf_counter()

    # "client.generate()"" sends a /v1/chat/completions request to Together AI
    # It returns (answer, latency_ms, in_tok, out_tok)
    # token counts are not stored in state
    answer, latency_ms, in_tok, out_tok = client.generate(
        model=config.llm_model,  # default is "Qwen/Qwen3-Next-80B-A3B-Instruct"
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=config.llm_max_tokens,  # default is 1024 tokens
        temperature=config.llm_temperature,  # default is 0.0
    )

    # Uses time to capture full round trip including network overhead
    elapsed_ms = (time.perf_counter() - start) * 1000

    # confidence detection: scan for uncertainty markers
    # displayed in the UI as green ("Sufficient evidence") or amber
    # ("Insufficient evidence - answer may be incomplete").
    answer_lower = answer.lower()
    confidence = "sufficient"
    for marker in _UNCERTAINTY_MARKERS:
        if marker in answer_lower:
            confidence = "insufficient_evidence"
            break

    return {
        "answer": answer,
        "confidence": confidence,
        "generation_ms": elapsed_ms,
        "stages_completed": state.get("stages_completed", []) + ["generate"],
    }
