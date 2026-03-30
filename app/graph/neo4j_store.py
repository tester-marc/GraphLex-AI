"""
This is for the Neo4j graph database operations for the GraphLex AI regulatory knowledge graph.

The app stores regulatory knowledge in 2 places:
1. in Weaviate (a vector database): for semantic search over the text chunks
2. in Neo4j (a graph database): for structured relationships between the legal entities

This file is for the Neo4j interface. It deals with the following:
- connecting to or disconnecting from Neo4j
- setting up database schema (for constraints and indexes)
- creates nodes (Instruments, Articles, Definitions, Obligations)
- creates relationships (CONTAINS, REFERENCES, DEFINES, IMPOSES, CITES, EQUIVALENT_TO)
- for running queries and retrieving stats

Node types:
Instrument : A source regulation or guidance document (e.g., "GDPR", "FADP")
Article    : A single article/provision within an instrument (e.g., "Art. 6 GDPR")
Definition : A legal term definition (e.g., "'personal data' means ...")
Obligation : A duty, right, or prohibition extracted from an article by the LLM

Relationship types:
CONTAINS:      Instrument -> Article (regulation contains its articles)
REFERENCES:    Article -> Article (cross reference between articles)
DEFINES:       Article -> Definition (an article defines a legal term)
IMPOSES:       Article -> Obligation (an article imposes an obligation or right)
CITES:         Instrument -> Article (a guidance document cites a statute article)
EQUIVALENT_TO: Article <-> Article (FADP article <-> corresponding GDPR article)

The Graph Layer (Layer 4) is called by the Orchestration pipeline (Layer 5) in order to:
- look up which article a user is asking about
- find cross references (e.g., "Article 6 references Article 4")
- retrieve the definitions of legal terms
- find equivalent provisions across jurisdictions (FADP <-> GDPR)
- surface obligations extracted from the articles
- provide subgraphs for the interactive visualization in the Gradio UI (Layer 6)
"""

# import libraries
from __future__ import annotations
from neo4j import Driver, GraphDatabase
from app.graph.config import GraphConfig
from app.graph.models import GraphQueryResult, GraphStats


class Neo4jStore:
    """
    This class manages the knowledge graph in Neo4j

    It is the single point of contact between the Python app and Neo4j and
    all the graph operations, i.e., creating nodes, creating relationships,
    running queries, etc. go through this class

    Usage:
    config = GraphConfig()
    store = Neo4jStore(config)
    store.connect()
    store.create_constraints()
    store.batch_create_articles([...])
    stats = store.get_stats()
    store.close()
    """

    def __init__(self, config: GraphConfig) -> None:
        """
        This initializes the store with a configuration object

        It does not connect to the database, which is done in connect() further below.

        Parameters:
        config: GraphConfig with the Neo4j URI, username, password, and the database name
        """
        self.config = config
        self._driver: Driver | None = None  # this is None until connect() is called

    # Connection

    def connect(self) -> None:
        """
        This connects to Neo4j and verifies that the db is actually reachable

        Raises:
        neo4j.exceptions.ServiceUnavailable: if the database is not reachable
        neo4j.exceptions.AuthError: if the credentials are wrong
        """
        if self._driver is not None:
            return

        self._driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )

        # this raises immediately if the db is down or if the credentials are wrong
        self._driver.verify_connectivity()

        print(f" Connected to Neo4j at {self.config.neo4j_uri}")

    def close(self) -> None:
        """
        This closes Neo4j connection and releases all the resources

        After this is called, connect() has to be called again before queries can be run
        """
        if self._driver:
            self._driver.close()
            self._driver = None

    @property
    def driver(self) -> Driver:
        """
        This returns the active Neo4j driver and raises an error if it's not connected

        Raises:
        RuntimeError: if connect() has not yet been called
        """
        if self._driver is None:
            raise RuntimeError("Not connected please call connect() first")
        return self._driver

    def _run(self, cypher: str, **params) -> list[dict]:
        """
        This executes a Cypher query and returns the results as a list of dicts

        Parameters:
        cypher: a Cypher query string
        **params: query parameters

        Returns:
        a list of dicts with 1 per result row
        """
        records, _, _ = self.driver.execute_query(
            cypher, database_=self.config.database, **params
        )
        return [dict(record) for record in records]

    # Schema

    def create_constraints(self) -> None:
        """
        This creates uniqueness constraints and indexes in Neo4j

        Uniqueness constraints prevent duplicate nodes and indexes
        speed up lookups based on property
        """
        stmts = [
            # uniqueness constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Instrument) REQUIRE i.source_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Definition) REQUIRE d.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (o:Obligation) REQUIRE o.node_id IS UNIQUE",
            # indexes for query performance
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.source_id)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Article) ON (a.article_label)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Definition) ON (d.term)",
            "CREATE INDEX IF NOT EXISTS FOR (o:Obligation) ON (o.source_id)",
        ]

        # every statement is separately executed because Neo4j does not support
        # multiple DDL statements in a single query
        for stmt in stmts:
            self._run(stmt)

        print(" Created constraints and indexes. ")

    # Clear the graph

    def clear_graph(self) -> None:
        """
        This deletes all nodes and relationships from the graph

        It is used by the CLI "--recreate" flag to wipe the graph,
        e.g., before building from scratch
        """
        self._run("MATCH (n) DETACH DELETE n")
        print(" Cleared graph ")

    # Batch Creation of Nodes:
    #
    # All of the node creation methods follow this pattern:
    # 1. returns early (0) if the input list is empty
    # 2. UNWIND list in a single Cypher query
    # 3. MERGE on unique ID
    # 4. SET all the remaining properties
    # 5. returns the count of nodes that were processed
    #

    def batch_create_instruments(self, instruments: list[dict]) -> int:
        """
        This creates or updates Instrument nodes

        Parameters:
        instruments: a list of dicts, each with the following keys:
        - source_id (str): this is a unique identifier, e.g., "gdpr"
        - title (str): a title
        - instrument_type (str): "statute" or "regulator_guidance"
        - jurisdiction (str): "EU" or "CH" (Switzerland)
        - effective_date (str): when the instrument took effect

        Returns:
        the number of Instrument nodes created / updated
        """
        if not instruments:
            return 0

        records = self._run(
            """
            UNWIND $items AS i
            MERGE (n:Instrument {source_id: i.source_id})
            SET n.title = i.title,
                n.instrument_type = i.instrument_type,
                n.jurisdiction = i.jurisdiction,
                n.effective_date = i.effective_date
            RETURN count(n) AS cnt
            """,
            items=instruments,
        )
        return records[0]["cnt"] if records else 0

    def batch_create_articles(self, articles: list[dict]) -> int:
        """
        This creates / updates Article nodes

        Parameters:
        articles: a list of dicts, each with the following keys:
        - node_id (str): a unique ID, format is "source_id:article_label"
        - source_id (str): for which regulation this article belongs to
        - article_label (str): the article number, e.g., "Article 6"
        - chapter (str): the chapter heading (if any at all)
        - section (str): the section heading (if any at all)
        - full_text (str): the complete text of the article

        Returns:
        the number of Article nodes created or updated
        """
        if not articles:
            return 0

        records = self._run(
            """
            UNWIND $items AS a
            MERGE (n:Article {node_id: a.node_id})
            SET n.source_id = a.source_id,
                n.article_label = a.article_label,
                n.chapter = a.chapter,
                n.section = a.section,
                n.full_text = a.full_text
            RETURN count(n) AS cnt
            """,
            items=articles,
        )
        return records[0]["cnt"] if records else 0

    def batch_create_definitions(self, definitions: list[dict]) -> int:
        """
        This creates or updates the Definition nodes

        Parameters:
        definitions: a list of dicts, each with the following keys:
        - node_id (str): a unique ID, e.g., "gdpr:def:personal data"
        - term (str): the defined term
        - definition_text (str): the full definition text
        - source_id (str): which regulation defines it
        - article_label (str): which article contains definition

        Returns:
        the number of Definition nodes created or updated
        """
        if not definitions:
            return 0

        records = self._run(
            """
            UNWIND $items AS d
            MERGE (n:Definition {node_id: d.node_id})
            SET n.term = d.term,
                n.definition_text = d.definition_text,
                n.source_id = d.source_id,
                n.article_label = d.article_label
            RETURN count(n) AS cnt
            """,
            items=definitions,
        )
        return records[0]["cnt"] if records else 0

    def batch_create_obligations(self, obligations: list[dict]) -> int:
        """
        This creates / updates Obligation nodes

        Obligations are duties, rights, prohibitions, or permissions extracted
        from the statute articles by the LLM. Each one of them has a type:
        - "obligation"   : this is something that MUST be done
        - "right"        : this is something a data subject CAN do
        - "prohibition"  : this is something that MUST NOT be done
        - "permission"   : this is something that MAY be done under certain conditions

        Parameters:
        obligations: a list of dicts, each with the following keys:
        - node_id (str): the SHA-256 hash of source + article + description
        - description (str): what the obligation requires
        - obligation_type (str): one of the 4 types above
        - source_id (str): which regulation it comes from
        - article_label (str): which article imposes the obligation
        - subject (str): who bears the obligation (e.g., "controller")

        Returns:
        number of Obligation nodes created or updated
        """
        if not obligations:
            return 0

        records = self._run(
            """
            UNWIND $items AS o
            MERGE (n:Obligation {node_id: o.node_id})
            SET n.description = o.description,
                n.obligation_type = o.obligation_type,
                n.source_id = o.source_id,
                n.article_label = o.article_label,
                n.subject = o.subject
            RETURN count(n) AS cnt
            """,
            items=obligations,
        )
        return records[0]["cnt"] if records else 0

    # Batch Creation of Relationships
    #
    # All the relationship methods: UNWIND the list, MATCH both endpoint nodes,
    # MERGE the relationship. If either of the nodes doesn't exist, the row is
    # skipped.
    #

    def batch_create_contains(self, rels: list[dict]) -> int:
        """
        This creates "CONTAINS" relationships: Instrument -> Article

        Parameters:
        rels: a list of dicts, each with the following keys:
        - source_id (str): the source_id of the Instrument
        - article_node_id (str): the node_id of the Article

        Returns:
        the number of "CONTAINS" relationships created
        """
        if not rels:
            return 0

        records = self._run(
            """
            UNWIND $items AS r
            MATCH (i:Instrument {source_id: r.source_id})
            MATCH (a:Article {node_id: r.article_node_id})
            MERGE (i)-[rel:CONTAINS]->(a)
            RETURN count(rel) AS cnt
            """,
            items=rels,
        )
        return records[0]["cnt"] if records else 0

    def batch_create_references(self, refs: list[dict]) -> int:
        """
        This creates "REFERENCES" relationships: Article -> Article (cross references)

        This is used by the Orchestration pipelines graph expansion step in order to pull in
        the related articles when the user queries are answered

        Parameters:
        refs: a list of dicts, each with the following keys:
        - source_node_id (str): the referencing article's node_id
        - target_node_id (str): the referenced article's node_id
        - reference_text (str): the raw reference text from source

        Returns:
        number of "REFERENCES" relationships that were created
        """
        if not refs:
            return 0

        records = self._run(
            """
            UNWIND $items AS r
            MATCH (src:Article {node_id: r.source_node_id})
            MATCH (tgt:Article {node_id: r.target_node_id})
            MERGE (src)-[rel:REFERENCES]->(tgt)
            ON CREATE SET rel.reference_text = r.reference_text
            RETURN count(rel) AS cnt
            """,
            items=refs,
        )
        return records[0]["cnt"] if records else 0

    def batch_create_defines(self, rels: list[dict]) -> int:
        """
        This creates the "DEFINES" relationships: Article -> Definition

        Parameters:
        rels: a list of dicts, each with the following keys:
        - article_node_id (str): the article's node_id
        - def_node_id (str): the node_id of the definition

        Returns:
        the number of DEFINES relationships that were created
        """
        if not rels:
            return 0

        records = self._run(
            """
            UNWIND $items AS r
            MATCH (a:Article {node_id: r.article_node_id})
            MATCH (d:Definition {node_id: r.def_node_id})
            MERGE (a)-[rel:DEFINES]->(d)
            RETURN count(rel) AS cnt
            """,
            items=rels,
        )
        return records[0]["cnt"] if records else 0

    def batch_create_imposes(self, rels: list[dict]) -> int:
        """
        This creates "IMPOSES" relationships: Article -> Obligation

        Parameters:
        rels: list of dicts, each with the following keys:
        - article_node_id (str): the node_id of the article
        - obligation_node_id (str): the obligation's node_id (the SHA-256 hash)

        Returns:
        number of "IMPOSES" relationships created
        """
        if not rels:
            return 0

        records = self._run(
            """
            UNWIND $items AS r
            MATCH (a:Article {node_id: r.article_node_id})
            MATCH (o:Obligation {node_id: r.obligation_node_id})
            MERGE (a)-[rel:IMPOSES]->(o)
            RETURN count(rel) AS cnt
            """,
            items=rels,
        )
        return records[0]["cnt"] if records else 0

    def batch_create_cites(self, rels: list[dict]) -> int:
        """
        This here creates "CITES" relationships: i.e., Instrument (guidance) -> Article (statute)

        It is different from REFERENCES in that guidance documents do not have their own Article
        nodes, so CITES links at the Instrument level.
        The pipeline uses these to show authoritative commentary when
        a user asks about a statute article.

        Parameters:
        rels: a list of dicts, each with the following keys:
        - source_id (str): citing guidance Instrument's source_id
        - article_node_id (str): the cited statute article's node_id

        Returns:
        number of "CITES" relationships that were created
        """
        if not rels:
            return 0

        records = self._run(
            """
            UNWIND $items AS r
            MATCH (i:Instrument {source_id: r.source_id})
            MATCH (a:Article {node_id: r.article_node_id})
            MERGE (i)-[rel:CITES]->(a)
            RETURN count(rel) AS cnt
            """,
            items=rels,
        )
        return records[0]["cnt"] if records else 0

    def batch_create_equivalent(self, rels: list[dict]) -> int:
        """
        This creates "EQUIVALENT_TO" relationships: Article (FADP) <-> Article (GDPR)

        These are manually curated cross jurisdictional equivalences (e.g., FADP Art. 7
        <-> GDPR Art. 25) and are used by the pipeline to show the corresponding
        article from the other jurisdiction to compare

        Parameters:
        rels: a list of dicts, each with the following keys:
        - fadp_node_id (str): the FADP article's node_id
        - gdpr_node_id (str): the node_id of the GDPR Article
        - note (str): a short description of the equivalence

        Returns:
        number of EQUIVALENT_TO relationships that were created
        """
        if not rels:
            return 0

        records = self._run(
            """
            UNWIND $items AS r
            MATCH (a:Article {node_id: r.fadp_node_id})
            MATCH (b:Article {node_id: r.gdpr_node_id})
            MERGE (a)-[rel:EQUIVALENT_TO]->(b)
            ON CREATE SET rel.note = r.note
            RETURN count(rel) AS cnt
            """,
            items=rels,
        )
        return records[0]["cnt"] if records else 0

    # Queries

    def query(self, cypher: str, **params) -> list[dict]:
        """
        To run an arbitrary Cypher query and return results as a list of dicts

        This is a public interface for ad hoc queries. It delegates to _run()

        It's used by GraphQueries in "queries.py" and by the Orchestration pipeline

        Parameters:
        cypher: a valid Cypher query string
        **params: query parameters

        Returns:
        list of dicts, 1 per result row
        """
        return self._run(cypher, **params)

    def get_stats(self) -> GraphStats:
        """
        This returns counts of each node type and relationship type in the knowledge graph

        It's used by the CLI status command and for the stats of the report.

        Returns:
        GraphStats object with integer counts for the following:
        instruments, articles, definitions, obligations, contains,
        references, defines, imposes, cites, equivalent_to
        """
        stats = GraphStats()

        for label, attr in [
            ("Instrument", "instruments"),
            ("Article", "articles"),
            ("Definition", "definitions"),
            ("Obligation", "obligations"),
        ]:
            rows = self._run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            setattr(stats, attr, rows[0]["cnt"] if rows else 0)

        for rel_type, attr in [
            ("CONTAINS", "contains"),
            ("REFERENCES", "references"),
            ("DEFINES", "defines"),
            ("IMPOSES", "imposes"),
            ("CITES", "cites"),
            ("EQUIVALENT_TO", "equivalent_to"),
        ]:
            rows = self._run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt")
            setattr(stats, attr, rows[0]["cnt"] if rows else 0)

        return stats
