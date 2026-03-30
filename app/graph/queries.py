"""
These are Cypher queries (Cypher is the query language for Neo4j)
for the regulatory knowledge graph.

For the separation of concerns:
- neo4j_store.py : for the connection and the raw query execution
- queries.py     : this defines "what" to ask (Cypher queries)
- pipeline.py    : this orchestrates "when" to ask

Graph Schema:
Instrument CONTAINS -> Article
Article    REFERENCES -> Article (cross references)
Article    DEFINES -> Definition (legal term definitions)
Article    IMPOSES -> Obligation (LLM extracted duties or rights)
Instrument CITES -> Article (guidance citing the statute articles)
Article    EQUIVALENT_TO Article (FADP <-> GDPR mappings)

Contains: 845 nodes and 1,190 relationships
"""

# import libraries
from __future__ import annotations
from app.graph.models import GraphQueryResult
from app.graph.neo4j_store import Neo4jStore


class GraphQueries:
    """
    This class is the query interface for the knowledge graph

    All of the methods are read only. Queries that are available:
    - get_article()              : a single article lookup
    - get_article_context()      : an article with all the related entities
    - get_references()           : a multi hop cross reference traversal
    - get_definitions()          : list the legal term definitions
    - search_definitions()       : searches definitions by substring for the term
    - get_guidance_for_article() : the guidance documents citing an article
    - get_equivalents()          : equivalent articles FADP <-> GDPR
    - get_subgraph()             : the neighborhood for the graph visualisation
    - get_article_hierarchy()    : all the articles in an instrument
    """

    def __init__(self, store: Neo4jStore) -> None:
        """
        Parameters:
        store: This is a connected Neo4jStore instance. The connection lifecycle is
        managed by GraphPipeline
        """
        self.store = store

    # Article lookup:

    def get_article(self, node_id: str) -> dict | None:
        """
        This function returns a single article node by the node_id

        Parameters:
        node_id: "{source_id}:{article_label}" (e.g., "gdpr:Article 17")

        Returns:
        a dict of article properties, otherwise None if not found
        """
        rows = self.store.query(
            "MATCH (a:Article {node_id: $nid}) RETURN a",
            nid=node_id,
        )
        return dict(rows[0]["a"]) if rows else None

    def get_article_context(self, node_id: str) -> GraphQueryResult:
        """
        This function returns an article with all the related entities and relationships

        It gets the article itself, its containing instrument, the definitions,
        outgoing and incoming cross references, obligations, and the cross-jurisdictional
        equivalents.

        Parameters:
        node_id: e.g., "gdpr:Article 6"

        Returns:
        GraphQueryResult with the nodes and typed relationships or
        an empty result if article does not exist
        """
        rows = self.store.query(
            """
            MATCH (a:Article {node_id: $nid})
            OPTIONAL MATCH (i:Instrument)-[:CONTAINS]->(a)
            OPTIONAL MATCH (a)-[:DEFINES]->(d:Definition)
            OPTIONAL MATCH (a)-[:REFERENCES]->(ref:Article)
            OPTIONAL MATCH (a)<-[:REFERENCES]-(back:Article)
            OPTIONAL MATCH (a)-[:IMPOSES]->(ob:Obligation)
            OPTIONAL MATCH (a)-[:EQUIVALENT_TO]-(eq:Article)
            RETURN a, i,
                   collect(DISTINCT d) AS definitions,
                   collect(DISTINCT ref) AS refs_out,
                   collect(DISTINCT back) AS refs_in,
                   collect(DISTINCT ob) AS obligations,
                   collect(DISTINCT eq) AS equivalents
            """,
            nid=node_id,
        )

        if not rows:
            return GraphQueryResult()

        row = rows[0]

        # nodes[0] is always the article, this is used by "GraphPipeline.format_article_context()"
        nodes = [dict(row["a"])]
        rels = []

        if row["i"]:
            nodes.append(dict(row["i"]))
            rels.append(
                {
                    "type": "CONTAINS",
                    "from": dict(row["i"]).get("source_id", ""),
                    "to": node_id,
                }
            )

        for d in row["definitions"]:
            nodes.append(dict(d))
            rels.append(
                {"type": "DEFINES", "from": node_id, "to": dict(d).get("node_id", "")}
            )

        for r in row["refs_out"]:
            rd = dict(r)
            nodes.append(rd)
            rels.append(
                {"type": "REFERENCES", "from": node_id, "to": rd.get("node_id", "")}
            )

        for r in row["refs_in"]:
            rd = dict(r)
            nodes.append(rd)
            rels.append(
                {"type": "REFERENCES", "from": rd.get("node_id", ""), "to": node_id}
            )

        for ob in row["obligations"]:
            nodes.append(dict(ob))
            rels.append(
                {"type": "IMPOSES", "from": node_id, "to": dict(ob).get("node_id", "")}
            )

        for eq in row["equivalents"]:
            nodes.append(dict(eq))
            # the direction goes from the queried article to the equivalent in order to be consistent
            rels.append(
                {
                    "type": "EQUIVALENT_TO",
                    "from": node_id,
                    "to": dict(eq).get("node_id", ""),
                }
            )

        return GraphQueryResult(nodes=nodes, relationships=rels)

    # Cross Reference Traversal

    def get_references(self, node_id: str, depth: int = 1) -> GraphQueryResult:
        """
        This function is to traverse "REFERENCES" edges going from a starting article
        and up to N hops in depth

        Parameters:
        node_id: the starting article's node_id, e.g., "gdpr:Article 6"
        depth: number of hops to follow (default is 1, and clamped to [1, 5])

        Returns:
        GraphQueryResult with all the discovered article nodes and "REFERENCES"
        relationships, it is empty if the article is not found or has no references
        """
        # clamps to [1, 5]
        depth = max(1, min(depth, 5))

        rows = self.store.query(
            f"""
            MATCH path = (a:Article {{node_id: $nid}})-[:REFERENCES*1..{depth}]->(b:Article)
            UNWIND nodes(path) AS n
            UNWIND relationships(path) AS r
            RETURN collect(DISTINCT n) AS nodes,
                   collect(DISTINCT {{
                       type: type(r),
                       from: startNode(r).node_id,
                       to: endNode(r).node_id
                   }}) AS rels
            """,
            nid=node_id,
        )

        if not rows:
            return GraphQueryResult()

        row = rows[0]
        return GraphQueryResult(
            nodes=[dict(n) for n in row["nodes"]],
            relationships=row["rels"],
        )

    # Legal Term Definitions

    def get_definitions(self, source_id: str | None = None) -> list[dict]:
        """
        This function lists all the legal term definitions, which can be optionally
        filtered by regulation

        Parameters:
        source_id: "gdpr", "fadp", or None (which is default) for all definitions

        Returns:
        a list of dicts that contain Definition nodes and are sorted by term
        """
        if source_id:
            return self.store.query(
                "MATCH (d:Definition {source_id: $sid}) RETURN d ORDER BY d.term",
                sid=source_id,
            )
        return self.store.query(
            "MATCH (d:Definition) RETURN d ORDER BY d.source_id, d.term"
        )

    def search_definitions(self, term_fragment: str) -> list[dict]:
        """
        This function is to search for definitions through a substring of the term name

        Parameters:
        term_fragment: the substring to search for, e.g., "data" or "consent"

        Returns:
        a list of dicts that contain matching Definition nodes and are sorted by the term
        """
        return self.store.query(
            """
            MATCH (d:Definition)
            WHERE toLower(d.term) CONTAINS toLower($frag)
            RETURN d ORDER BY d.term
            """,
            frag=term_fragment,
        )

    # Guidance -> Article links

    def get_guidance_for_article(self, article_node_id: str) -> list[dict]:
        """
        This function is for finding guidance documents that cite specific statute article

        Parameters:
        article_node_id: e.g., "gdpr:Article 7"

        Returns:
        a list of dicts with "source_id" and "title" which are sorted by the source_id
        or an empty list if no guidance documents have cited this article
        """
        return self.store.query(
            """
            MATCH (i:Instrument)-[:CITES]->(a:Article {node_id: $nid})
            RETURN i.source_id AS source_id, i.title AS title
            ORDER BY i.source_id
            """,
            nid=article_node_id,
        )

    # For cross-jurisdictional equivalents

    def get_equivalents(self, node_id: str) -> list[dict]:
        """
        This method finds articles equivalent to the given article across jurisdictions

        It traverses the "EQUIVALENT_TO" relationships (FADP <-> GDPR mappings)

        Parameters:
        node_id: the article node_id in either of the regulations

        Returns:
        a list of dicts with "node_id", "label", and "source_id"
        or an empty list if no equivalences exist
        """
        return self.store.query(
            """
            MATCH (a:Article {node_id: $nid})-[:EQUIVALENT_TO]-(eq:Article)
            RETURN eq.node_id AS node_id, eq.article_label AS label,
                   eq.source_id AS source_id
            """,
            nid=node_id,
        )

    # Subgraph for the visualization

    def get_subgraph(self, article_node_ids: list[str]) -> GraphQueryResult:
        """
        This function returns neighbourhood graph for a set of seed articles

        It is used by the UI Layer (graph_viz.py) in order to be able to render the interactive pyvis
        visualization in the Gradio UI's "Graph" tab

        Parameters:
        article_node_ids: the seed article node_ids, e.g., ["gdpr:Article 6", "gdpr:Article 7"]

        Returns:
        GraphQueryResult with all the neighbourhood nodes and relationships
        or an empty result if none of the articles given exist
        """
        rows = self.store.query(
            """
            UNWIND $nids AS nid
            MATCH (a:Article {node_id: nid})
            OPTIONAL MATCH (a)-[r1:REFERENCES]->(ref:Article)
            OPTIONAL MATCH (a)<-[r2:REFERENCES]-(back:Article)
            OPTIONAL MATCH (a)-[:DEFINES]->(d:Definition)
            OPTIONAL MATCH (a)-[:EQUIVALENT_TO]-(eq:Article)
            OPTIONAL MATCH (i:Instrument)-[:CONTAINS]->(a)
            WITH collect(DISTINCT a) + collect(DISTINCT ref) + collect(DISTINCT back)
                 + collect(DISTINCT d) + collect(DISTINCT eq) + collect(DISTINCT i) AS all_nodes,
                 collect(DISTINCT {type: 'REFERENCES', from: a.node_id, to: ref.node_id}) +
                 collect(DISTINCT {type: 'REFERENCES', from: back.node_id, to: a.node_id}) +
                 collect(DISTINCT {type: 'DEFINES', from: a.node_id, to: d.node_id}) +
                 collect(DISTINCT {type: 'EQUIVALENT_TO', from: a.node_id, to: eq.node_id}) +
                 collect(DISTINCT {type: 'CONTAINS', from: i.source_id, to: a.node_id}) AS all_rels
            UNWIND all_nodes AS n
            WITH collect(DISTINCT n) AS nodes, all_rels
            UNWIND all_rels AS r
            WITH nodes, collect(DISTINCT r) AS rels
            RETURN nodes, [r IN rels WHERE r.from IS NOT NULL AND r.to IS NOT NULL] AS rels
            """,
            nids=article_node_ids,
        )

        if not rows:
            return GraphQueryResult()

        row = rows[0]
        return GraphQueryResult(
            nodes=[dict(n) for n in row["nodes"]],
            relationships=row["rels"],
        )

    # For Article Hierarchy

    def get_article_hierarchy(self, source_id: str) -> list[dict]:
        """
        This function returns all the articles in an instrument (which are ordered by label)

        It provides a "table of contents" view of a regulation.

        Parameters:
        source_id: the instrument identifier, e.g., "gdpr" or "fadp".

        Returns:
        a list of dicts with "label", "chapter", "section", and "node_id",
        which is sorted by article_label
        """
        return self.store.query(
            """
            MATCH (i:Instrument {source_id: $sid})-[:CONTAINS]->(a:Article)
            RETURN a.article_label AS label, a.chapter AS chapter,
                   a.section AS section, a.node_id AS node_id
            ORDER BY a.article_label
            """,
            sid=source_id,
        )
