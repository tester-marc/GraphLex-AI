"""
For the graph visualization with pyvis

This converts "GraphQueryResult" data (i.e., the nodes and relationships) into a
HTML network graph (interactive) which is embeddable in a Gradio HTML component

pyvis generates a full HTML document, it can be embedded via an iframe
with srcdoc in order to isolate it from Gradio's page

Node types:   Instrument, Article, Definition, Obligation
Edge types:   CONTAINS, REFERENCES, DEFINES, IMPOSES, CITES, EQUIVALENT_TO
"""

# import libraries
from __future__ import annotations
import html
import tempfile
from pathlib import Path
from pyvis.network import Network


# colour palettes
#
# blue=#4A90D9 (Instrument / References), Green=#50C878 (Article),
# orange=#F5A623 (Definition / Defines), Red=#D0021B (Obligation / Imposes),
# purple=#9B59B6 (Cites), Teal=#1ABC9C (Equivalent_To)
# the unknown types fall back to grey (#888888)

_NODE_COLOURS = {
    "Instrument": "#4A90D9",
    "Article": "#50C878",
    "Definition": "#F5A623",
    "Obligation": "#D0021B",
}

_EDGE_COLOURS = {
    "CONTAINS": "#AAAAAA",
    "REFERENCES": "#4A90D9",
    "DEFINES": "#F5A623",
    "IMPOSES": "#D0021B",
    "CITES": "#9B59B6",
    "EQUIVALENT_TO": "#1ABC9C",
}


def _node_label(node: dict) -> str:
    """This derives a readable label for a graph node

    Priority :
    1. Article      : "<SOURCE_ID uppercased> <article_label>"
    2. Definition   : the term
    3. Obligation   : the first 40 characters of the description + "..."
    4. Instrument   : the title
    5. Fallback     : node_id / source_id / "?"
    """
    if "article_label" in node:
        src = node.get("source_id", "")
        return f"{src.upper()} {node['article_label']}"
    if "term" in node:
        return node["term"]
    if "description" in node:
        return node["description"][:40] + (
            "..." if len(node.get("description", "")) > 40 else ""
        )
    if "title" in node:
        return node["title"]
    return node.get("node_id", node.get("source_id", "?"))


def _node_type(node: dict) -> str:
    """this determines the node type based on its properties

    the type is detected by these distinctive property combinations:
    - Article:     has "article_label" + "full_text"
    - Definition:  has "term" + "definition_text"
    - Obligation:  has "obligation_type"
    - Instrument:  has "title" or has "source_id" without "article_label"

    It defaults to "Article" if there is no pattern that matches
    """
    if "article_label" in node and "full_text" in node:
        return "Article"
    if "term" in node and "definition_text" in node:
        return "Definition"
    if "obligation_type" in node:
        return "Obligation"
    if "title" in node or ("source_id" in node and "article_label" not in node):
        return "Instrument"
    return "Article"


def build_graph_html(
    nodes: list[dict],
    relationships: list[dict],
    height: str = "500px",
) -> str:
    """This builds an interactive pyvis HTML graph from the nodes and relationships

    Parameters:
    nodes : list[dict]
    for node dicts with properties as "node_id", "article_label",
    "source_id", "full_text", "term", "definition_text",
    "obligation_type", "description", "title"
    relationships : list[dict]
    for dicts with keys "from" (source node_id), "to" (target node_id),
    and "type" (e.g., CONTAINS, REFERENCES)
    height : str
    the CSS height for the pyvis canvas (default is 500 px)

    Returns:
    str
    This is an "<iframe srcdoc="...">" with the interactive graph or
    a placeholder "<p>" if input is empty
    """

    if not nodes and not relationships:
        return "<p style='color:#888; text-align:center; padding:40px;'>No graph context for this query.</p>"

    # a dark themed, directed network
    net = Network(
        height=height,
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="#ffffff",
        cdn_resources="remote",
    )

    # for the physics, i.e., the nodes repel at -3000, central pull, and 150 px edge rest length
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    # adds nodes
    seen_ids: set[str] = set()

    for node in nodes:
        nid = node.get("node_id", node.get("source_id", ""))
        if not nid or nid in seen_ids:
            continue
        seen_ids.add(nid)

        ntype = _node_type(node)
        colour = _NODE_COLOURS.get(ntype, "#888888")
        label = _node_label(node)

        # this builds HTML tooltip with type specific detail
        title_parts = [f"<b>{ntype}</b>: {label}"]
        if "full_text" in node and node["full_text"]:
            preview = node["full_text"][:300].replace("\n", "<br>")
            title_parts.append(f"<br><br>{preview}...")
        if "definition_text" in node and node["definition_text"]:
            title_parts.append(f"<br><br>{node['definition_text'][:200]}")
        if "description" in node and "obligation_type" in node:
            title_parts.append(f"<br>Type: {node['obligation_type']}")
            title_parts.append(f"<br>Subject: {node.get('subject', '?')}")

        # for the hierarchy -> articles largest, instruments medium, and the others are smallest
        size = 25 if ntype == "Article" else 20 if ntype == "Instrument" else 15

        net.add_node(
            nid,
            label=label,
            title="".join(title_parts),
            color=colour,
            size=size,
        )

    # adds the edges
    for rel in relationships:
        src = rel.get("from", "")
        tgt = rel.get("to", "")
        rtype = rel.get("type", "")

        if src and tgt and src in seen_ids and tgt in seen_ids:
            net.add_edge(
                src,
                tgt,
                title=rtype,
                label=rtype,
                color=_EDGE_COLOURS.get(rtype, "#888888"),
                arrows="to",
                font={"size": 10, "color": "#cccccc"},
            )

    # saves it to a tmp file, reads back, escape for srcdoc, and wraps in an iframe
    tmp = Path(tempfile.gettempdir()) / "graphlex_viz.html"
    net.save_graph(str(tmp))
    raw_html = tmp.read_text(encoding="utf-8")
    escaped = html.escape(raw_html, quote=True)

    # iframe is 550 px in height (500px canvas + 50px)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%; height:550px; border:none;"></iframe>'
    )
