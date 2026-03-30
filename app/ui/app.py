"""This is the Gradio web interface for GraphLex AI

It has a tabbed layout with 4 tabs:
1. Answer        : the generated response with confidence indicator
2. Evidence      : the retrieved passages and graph context
3. Graph         : this is for the interactive pyvis network visualization
4. Diagnostics   : for pipeline stage timings and metadata
"""

# import libraries
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import gradio as gr
from app.orchestration.config import OrchestrationConfig
from app.orchestration.pipeline import OrchestrationPipeline
from app.ui.graph_viz import build_graph_html


def build_theme() -> gr.themes.Default:
    """this builds the GraphLex AI Gradio theme"""
    return gr.themes.Default(
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Courier New", "monospace"],
    ).set(
        input_background_fill="#f5f5f5",
        input_background_fill_dark="#2e2e2e",
        input_border_color="#d4d4d4",
        input_border_color_dark="#525252",
    )


# pipeline

# initialised on the first query, it holds persistent connections to
# Weaviate, Neo4j, as well as the LLM API

_pipeline: OrchestrationPipeline | None = None


def _get_pipeline() -> OrchestrationPipeline:
    """this returns the pipeline and creates it on the first call"""
    global _pipeline
    if _pipeline is None:
        _pipeline = OrchestrationPipeline()
    return _pipeline


# Query handler


def _run_query(
    text_input: str,
    audio_input: str | None,
) -> tuple[str, str, str, str, str]:
    """This is to run a user query through the orchestration pipeline

    It returns a tuple of 5:
    (answer_md, evidence_md, graph_html, diagnostics_md, status)
    """
    pipeline = _get_pipeline()

    # if both inputs are provided, audio has priority
    if audio_input:
        result = pipeline.run_audio(audio_input)
    elif text_input and text_input.strip():
        result = pipeline.run_text(text_input.strip())
    else:
        return (
            "Please enter a question or record audio.",
            "",
            "",
            "",
            "No input provided",
        )

    answer_md = _format_answer(result)
    evidence_md = _format_evidence(result)
    graph_html = _format_graph(result)
    diagnostics_md = _format_diagnostics(result)
    status = _format_status(result)

    return answer_md, evidence_md, graph_html, diagnostics_md, status


# Format Answer tab


def _format_answer(result: dict) -> str:
    """This is to format the "Answer" tab: query echo, confidence badge, LLM response, errors"""
    confidence = result.get("confidence", "unknown")
    answer = result.get("answer", "No answer generated.")
    query = result.get("query_text", "")
    mode = result.get("input_mode", "text")

    lines: list[str] = []

    label = "Transcribed query" if mode == "audio" else "Query"
    lines.append(f"**{label}:** {query}\n")

    if confidence == "sufficient":
        lines.append("**Confidence:** Sufficient evidence found\n")
    else:
        lines.append(
            "**Confidence:** Insufficient evidence - answer may be incomplete\n"
        )

    lines.append("---\n")
    lines.append(answer)

    errors = result.get("errors", [])
    if errors:
        lines.append("\n\n---\n")
        lines.append("**Errors:**\n")
        for err in errors:
            lines.append(f"- {err}")

    return "\n".join(lines)


# Format Evidence tab


def _format_evidence(result: dict) -> str:
    """this is format the Evidence tab for retrieved passages and graph context

    It lets the users verify the sources behind the generated answer
    """
    lines: list[str] = []

    # retrieved passages (Weaviate vector search)
    chunks = result.get("retrieved_chunks", [])
    lines.append(f"### Retrieved Passages ({len(chunks)})\n")

    if not chunks:
        lines.append("*No passages retrieved.*\n")
    else:
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source_id", "unknown").upper()
            article = chunk.get("article", "")
            section = chunk.get("section", "")
            loc = " > ".join(filter(None, [article, section]))
            itype = chunk.get("instrument_type", "")
            score = chunk.get("score", 0)
            text = chunk.get("text", "")

            lines.append(
                f"**[{i}] {source}** | {loc} | *{itype}* | score: {score:.3f}\n"
            )

            preview = text[:300].replace("\n", " ")
            if len(text) > 300:
                preview += "..."
            lines.append(f"> {preview}\n")

    # --- Graph Context (Neo4j) ---
    graph_ctx = result.get("graph_context", [])
    lines.append(f"\n### Graph Context ({len(graph_ctx)} items)\n")

    if not graph_ctx:
        lines.append("*No graph context expanded.*\n")
    else:
        for item in graph_ctx:
            ctx_type = item.get("type", "")
            ref = item.get("ref", "")

            if ctx_type == "article_context":
                nodes = item.get("nodes", [])
                rels = item.get("relationships", [])

                lines.append(f"**Article: {ref}**\n")

                defs = [r for r in rels if r.get("type") == "DEFINES"]
                obs = [r for r in rels if r.get("type") == "IMPOSES"]
                refs_out = [r for r in rels if r.get("type") == "REFERENCES"]

                if defs:
                    lines.append(f"- Definitions: {len(defs)}")
                    for d_rel in defs:
                        d_node = next(
                            (n for n in nodes if n.get("node_id") == d_rel.get("to")),
                            {},
                        )
                        term = d_node.get("term", "?")
                        defn = d_node.get("definition_text", "")[:120]
                        lines.append(f"  - **{term}**: {defn}")

                if obs:
                    lines.append(f"- Obligations: {len(obs)}")
                    # this is capped at 5 to keep display manageable
                    for o_rel in obs[:5]:
                        o_node = next(
                            (n for n in nodes if n.get("node_id") == o_rel.get("to")),
                            {},
                        )
                        lines.append(
                            f"  - [{o_node.get('obligation_type', '?')}] "
                            f"{o_node.get('description', '')[:120]}"
                        )

                if refs_out:
                    # this is capped at 8 cross-references
                    targets = [r.get("to", "?") for r in refs_out[:8]]
                    lines.append(f"- Cross-references: {', '.join(targets)}")

                lines.append("")

            elif ctx_type == "equivalent_article":
                # for cross-jurisdictional FADP <-> GDPR mapping (14 known equivalences)
                lines.append(
                    f"**Equivalent:** {item.get('source_ref', '')} "
                    f"<-> {item.get('equivalent_ref', '')}\n"
                )

            elif ctx_type == "guidance_citations":
                guidance = item.get("guidance", [])
                if guidance:
                    titles = [g.get("title", g.get("source_id", "")) for g in guidance]
                    lines.append(f"**Guidance citing {ref}:** {', '.join(titles)}\n")

    return "\n".join(lines)


# Format Graph tab


def _format_graph(result: dict) -> str:
    """This part builds the pyvis interactive graph visualisation

    Node colors: Blue=Instrument, Green=Article, Orange=Definition, Red=Obligation
    Edge colors: Grey=CONTAINS, Blue=REFERENCES, Orange=DEFINES,
                 Red=IMPOSES, Purple=CITES, Teal=EQUIVALENT_TO
    """
    graph_ctx = result.get("graph_context", [])

    all_nodes: list[dict] = []
    all_rels: list[dict] = []

    for item in graph_ctx:
        all_nodes.extend(item.get("nodes", []))
        all_rels.extend(item.get("relationships", []))

    # include the article nodes from retrieved chunks that may not appear in the
    # graph context, this way the graph always shows at least the sources of the answer
    chunks = result.get("retrieved_chunks", [])
    chunk_refs: set[str] = set()
    for chunk in chunks:
        art = chunk.get("article", "")
        src = chunk.get("source_id", "")
        if art and src:
            ref = f"{src}:{art}"
            if ref not in chunk_refs:
                chunk_refs.add(ref)
                existing_ids = {n.get("node_id", "") for n in all_nodes}
                if ref not in existing_ids:
                    all_nodes.append(
                        {
                            "node_id": ref,
                            "article_label": art,
                            "source_id": src,
                            "full_text": chunk.get("text", "")[:200],
                        }
                    )

    return build_graph_html(all_nodes, all_rels)


# Format Diagnostics tab


def _format_diagnostics(result: dict) -> str:
    """This is to format the Diagnostics tab, i.e., pipeline stages, latency breakdown, filters, errors"""
    lines: list[str] = []

    mode = result.get("input_mode", "?")
    lines.append(f"**Input mode:** {mode}\n")

    stages = result.get("stages_completed", [])
    lines.append(f"**Pipeline stages:** {' -> '.join(stages)}\n")

    lines.append("### Latency Breakdown\n")
    lines.append("| Stage | Time (ms) |")
    lines.append("|---|---|")

    trans_ms = result.get("transcription_ms", 0)
    if trans_ms > 0:
        lines.append(f"| Transcription | {trans_ms:.0f} |")
    lines.append(f"| Retrieval | {result.get('retrieval_ms', 0):.0f} |")
    lines.append(f"| Graph expansion | {result.get('graph_ms', 0):.0f} |")
    lines.append(f"| Generation | {result.get('generation_ms', 0):.0f} |")
    lines.append(f"| **Total** | **{result.get('total_ms', 0):.0f}** |")

    chunks = result.get("retrieved_chunks", [])
    graph_ctx = result.get("graph_context", [])
    lines.append(f"\n**Retrieved chunks:** {len(chunks)}")
    lines.append(f"**Graph context items:** {len(graph_ctx)}")

    # to interpret node outputs: the article references and search filters
    article_refs = result.get("article_refs", [])
    if article_refs:
        lines.append(f"**Article references detected:** {', '.join(article_refs)}")

    source_filters = result.get("source_filters", [])
    jurisdiction_filters = result.get("jurisdiction_filters", [])
    if source_filters:
        lines.append(f"**Source filters:** {', '.join(source_filters)}")
    if jurisdiction_filters:
        lines.append(f"**Jurisdiction filters:** {', '.join(jurisdiction_filters)}")

    errors = result.get("errors", [])
    if errors:
        lines.append("\n### Errors\n")
        for err in errors:
            lines.append(f"- {err}")

    return "\n".join(lines)


# Format Status bar


def _format_status(result: dict) -> str:
    """This returns a 1-line summary for: latency, passage / graph counts, confidence"""
    total_ms = result.get("total_ms", 0)
    confidence = result.get("confidence", "?")
    n_chunks = len(result.get("retrieved_chunks", []))
    n_graph = len(result.get("graph_context", []))
    return (
        f"Done in {total_ms:.0f}ms | "
        f"{n_chunks} passages, {n_graph} graph items | "
        f"confidence: {confidence}"
    )


# Example queries
#
# It covers: article lookup, FADP obligations, cross-jurisdictional comparison,
# guidance retrieval, international transfers, and definition lookup

_EXAMPLES = [
    "What does GDPR Article 17 say about the right to erasure?",
    "What are the data controller's obligations under FADP Article 6?",
    "How do GDPR and FADP differ on consent requirements?",
    "What technical measures does the FDPIC recommend for data security?",
    "What does GDPR Article 49 say about international data transfers?",
    "What is the definition of personal data under GDPR?",
]


# Build the Gradio App


def create_app() -> gr.Blocks:
    """This creates and returns the Gradio Blocks app

    It doesn't start the web server
    """

    with gr.Blocks(title="GraphLex AI") as app:

        gr.Markdown(
            "# GraphLex AI\n"
            "*Regulatory compliance intelligence for Swiss and EU data protection law*"
        )

        # inputs: a text box (4/6 of the width) and audio recorder/uploader (2/6 of the width)
        with gr.Row():
            with gr.Column(scale=4):
                text_input = gr.Textbox(
                    label="Ask a regulatory question",
                    placeholder="e.g., What does GDPR Article 17 say about the right to erasure?",
                    lines=2,
                )
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Or record/upload audio",
                    type="filepath",
                    sources=["microphone", "upload"],
                )

        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary", scale=2)
            clear_btn = gr.Button("Clear", scale=1)

        status_box = gr.Textbox(label="Status", interactive=False, max_lines=1)

        with gr.Tabs():
            with gr.TabItem("Answer"):
                answer_output = gr.Markdown(label="Answer")
            with gr.TabItem("Evidence"):
                evidence_output = gr.Markdown(label="Evidence")
            with gr.TabItem("Graph"):
                graph_output = gr.HTML(label="Knowledge Graph")
            with gr.TabItem("Diagnostics"):
                diagnostics_output = gr.Markdown(label="Diagnostics")

        gr.Examples(
            examples=[[ex, None] for ex in _EXAMPLES],
            inputs=[text_input, audio_input],
            label="Example queries",
        )

        # event handlers
        outputs = [
            answer_output,
            evidence_output,
            graph_output,
            diagnostics_output,
            status_box,
        ]

        submit_btn.click(
            fn=_run_query, inputs=[text_input, audio_input], outputs=outputs
        )
        text_input.submit(
            fn=_run_query, inputs=[text_input, audio_input], outputs=outputs
        )

        clear_btn.click(
            fn=lambda: ("", None, "", "", "", "", ""),
            outputs=[text_input, audio_input, *outputs],
        )

    return app
