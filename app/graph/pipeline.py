"""
This is the graph pipeline in order to extract entities and relationships and then build the Neo4j knowledge graph

This file is for the orchestration of the Graph Layer (Layer 4 of GraphLex AI).
It coordinates the entire lifecycle as follows:
1. Building   : it extracts entities and relationships from the regulatory chunks into Neo4j
2. Querying   : for article lookups, cross reference traversal, retrieval of results
3. Formatting : it converts raw graph data into text that can be read by users

Key Classes:

GraphConfig    (config.py)       : for the Neo4j connection settings from env variables
GraphExtractor (extractor.py)    : this extracts entities / relationships from the chunk JSON
Neo4jStore     (neo4j_store.py)  : Cypher executions in Neo4j
GraphQueries   (queries.py)      : Cypher queries for common lookups
GraphStats     (models.py)       : Counts for nodes and relationships

CLI entry points:
python -m app.graph build                           # structural extraction
python -m app.graph build --recreate                # to wipe and rebuild
python -m app.graph build-obligations               # for LLM extraction
python -m app.graph build-obligations --from-cache  # loading from cache
python -m app.graph status                          # to show the node/relationship counts
python -m app.graph article gdpr:Article 6          # looks up a specific article
python -m app.graph refs gdpr:Article 6             # to traverse the cross references
python -m app.graph defs                            # to list all definitions
python -m app.graph guidance gdpr:Article 7         # to find guidance that cites an article
python -m app.graph delete                          # to wipe the whole graph
"""

# import libraries
from __future__ import annotations
import json
import time
from app.graph.config import GraphConfig
from app.graph.extractor import GraphExtractor
from app.graph.models import GraphQueryResult, GraphStats
from app.graph.neo4j_store import Neo4jStore
from app.graph.queries import GraphQueries


# LLM prompt for extraction of obligations

# This is the system prompt sent to the LLM "Qwen3-Next 80B-A3B" for JSON only output
# with fixed schema (description, type, subject) for parsing.
_OBLIGATION_SYSTEM_PROMPT = """\
You are a legal analyst. Extract regulatory obligations from the given article text.
For each obligation, return a JSON array of objects with:
- "description": One-sentence summary of the obligation
- "type": One of "obligation", "right", "prohibition", "permission"
- "subject": Who bears this (e.g., "controller", "processor", "data subject", "supervisory authority")
Return ONLY the JSON array, no other text. If no obligations are found, return [].
"""


class GraphPipeline:
    """
    This is for the E2E graph construction and query interface

    It composes GraphExtractor, Neo4jStore, and GraphQueries.

    Usage:
    pipeline = GraphPipeline()
    pipeline.connect()
    pipeline.build(recreate=True)
    pipeline.load_cached_obligations()
    stats = pipeline.status()
    result = pipeline.query_article("gdpr:Article 17")
    pipeline.close()
    """

    def __init__(self, config: GraphConfig | None = None) -> None:
        """
        Parameters:
        config: the Neo4j connection settings, if None, then GraphConfig reads from env:
        NEO4J_URI (default: bolt://localhost:7687), NEO4J_USER (default: neo4j),
        NEO4J_PASSWORD (required)

        This doesn't connect to Neo4j, connect() or build() need to be called first
        """
        self.config = config or GraphConfig()
        self.store = Neo4jStore(self.config)
        # config.chunks_dir points to the PyMuPDF output
        self.extractor = GraphExtractor(self.config.chunks_dir)
        self.queries = GraphQueries(self.store)

    # Lifecycle

    def connect(self) -> None:
        """Opens the Neo4j connection"""
        self.store.connect()

    def close(self) -> None:
        """Closes the Neo4j connection, to reconnect connect() needs to be called"""
        self.store.close()

    # Build the graph (for structural extraction)

    def build(self, recreate: bool = False) -> GraphStats:
        """
        This builds the knowledge graph through structural extraction (regex and metadata)

        The order of build matters, and later steps depend on earlier nodes existing:
        1. Instruments, 2. Articles, 3. CONTAINS, 4. REFERENCES,
        5. CITES, 6. EQUIVALENT_TO, 7. Definitions and DEFINES

        Parameters:
        recreate: if this is true, it wipes the graph before rebuilding; if it's false (which is default),
        then MERGE operations update existing nodes

        Returns:
        GraphStats with the counts of all the node and relationship types
        """
        print("\n Graph Layer: Building Knowledge Graph... ")
        start = time.perf_counter()
        self.store.connect()

        if recreate:
            self.store.clear_graph()

        # this uses IF NOT EXISTS and is safe to call on each build
        self.store.create_constraints()

        # step 1: the six source documents (GDPR, FADP, 4 guidance docs)
        instruments = self.extractor.extract_instruments()
        n = self.store.batch_create_instruments(instruments)
        print(f" Instruments: {n}")

        # step 2: the 174 articles (100 GDPR and 74 FADP)
        articles = self.extractor.extract_articles()
        n = self.store.batch_create_articles(articles)
        print(f" Articles: {n}")

        # step 3: (Instrument)-[:CONTAINS]->(Article)
        contains = self.extractor.extract_contains()
        n = self.store.batch_create_contains(contains)
        print(f" CONTAINS: {n}")

        # step 4: (Article)-[:REFERENCES]->(Article)
        # found with regex ("Article 7", "Art. 35", "Articles 12 to 22") and
        # resolved against "_article_index"
        refs = self.extractor.extract_references()
        n = self.store.batch_create_references(refs)
        print(f" REFERENCES: {n}")

        # step 5: (GuidanceInstrument)-[:CITES]->(Article)
        # this is distinct from "REFERENCES", the source is an Instrument, not an Article
        cites = self.extractor.extract_cites()
        n = self.store.batch_create_cites(cites)
        print(f" CITES: {n}")

        # step 6: with 14 hard coded FADP <-> GDPR equivalences (e.g., FADP Art.7 <-> GDPR Art.25)
        # hard coded by design choice because mappings that are incorrect could produce wrong legal advice
        equivs = self.extractor.extract_equivalents()
        n = self.store.batch_create_equivalent(equivs)
        print(f" EQUIVALENT_TO: {n}")

        # step 7: 33 definitions (23 GDPR Art.4 and 10 FADP Art.5)
        # "extract_definitions()"" returns both the node dicts and relationship dicts
        defs, def_rels = self.extractor.extract_definitions()
        n = self.store.batch_create_definitions(defs)
        print(f" Definitions: {n}")
        n = self.store.batch_create_defines(def_rels)
        print(f" DEFINES: {n}")

        elapsed = time.perf_counter() - start
        stats = self.store.get_stats()
        print(f"\n Built in {elapsed:.1f}s")
        print(f"\n{stats}")
        self._save_stats(stats)
        return stats

    # Build the obligations (LLM based)

    def build_obligations(self) -> int:
        """
        This extracts the obligations from the statute articles via LLM Qwen3-Next (Together AI)

        It requires the TOGETHER_API_KEY and
        it only needs to run once, because load_cached_obligations() are used
        after the first run

        Process:
        1. load TogetherClient and verify the API key
        2. get the obligation candidates
        3. per article: call the LLM, then parse JSON, then gather results
        4. the Obligation nodes + IMPOSES relationships are batch inserted
        5. finally, cache the results to "data/output/graph/obligations_cache.json"

        Returns:
        the number of Obligation nodes that were created
        """
        # this avoids loading the LLM layer when only the graph queries are needed
        from app.llm.together_client import TogetherClient

        client = TogetherClient()
        if not client.is_available():
            print(" Error: TOGETHER_API_KEY is not set, can't extract obligations")
            return 0

        # "Qwen3-Next 80B-A3B" LLM, MoE model (80B total params, 3B active)
        # this was chosen over "Llama 3.3 70B" in the LLM comparison
        model = "Qwen/Qwen3-Next-80B-A3B-Instruct"

        self.store.connect()

        # "_article_index" is populated by "extract_articles()"
        if not self.extractor._article_index:
            self.extractor.extract_articles()

        candidates = self.extractor.get_obligation_candidates()
        print(f"\n Graph Layer: Extracting Obligations ({len(candidates)} articles) ")

        all_obs: list[dict] = []
        all_rels: list[dict] = []
        total_tokens = 0

        for i, art in enumerate(candidates):
            node_id = art["node_id"]
            # it is capped at 1500 chars in order to keep costs predictable and also to avoid context window limits
            text = art["full_text"][:1500]

            try:
                response, latency, in_tok, out_tok = client.generate(
                    model=model,
                    system_prompt=_OBLIGATION_SYSTEM_PROMPT,
                    user_prompt=f"Article: {art['article_label']} ({art['source_id'].upper()})\n\n{text}",
                    max_tokens=512,
                    temperature=0.0,  # temperature is at zero, the same article always yields the same obligations
                )
                total_tokens += in_tok + out_tok

                obs, rels = self.extractor.parse_obligation_response(
                    node_id,
                    art["source_id"],
                    art["article_label"],
                    response,
                )
                all_obs.extend(obs)
                all_rels.extend(rels)

                if (i + 1) % 20 == 0:
                    print(f" Processed {i + 1}/{len(candidates)} articles...")

            except Exception as e:
                # skips the failed articles rather than aborting
                print(f" Warning: Failed on {node_id}: {e}")
                continue

        # batch insert
        n_obs = self.store.batch_create_obligations(all_obs)
        n_rels = self.store.batch_create_imposes(all_rels)
        print(f" Obligations: {n_obs} nodes, {n_rels} IMPOSES relationships")
        print(f" Total tokens: {total_tokens:,}")

        # caches to disk
        cache_path = self.config.output_dir / "obligations_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"obligations": all_obs, "relationships": all_rels}, f, indent=2)
        print(f" Cached to {cache_path}")

        return n_obs

    def load_cached_obligations(self) -> int:
        """
        This loads obligations from the cache instead of calling the LLM again

        the cache file is "data/output/graph/obligations_cache.json"
        Format is: {"obligations": [...632 dicts...], "relationships": [...632 dicts...]}

        Returns:
        the number of Obligation nodes that are loaded (which is 0 if the cache file doesn't exist)
        """
        cache_path = self.config.output_dir / "obligations_cache.json"
        if not cache_path.exists():
            print(" No obligation cache found, run build-obligations first.")
            return 0

        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)

        self.store.connect()
        n_obs = self.store.batch_create_obligations(cached["obligations"])
        n_rels = self.store.batch_create_imposes(cached["relationships"])
        print(f" Loaded from cache: {n_obs} obligations, {n_rels} IMPOSES")
        return n_obs

    # Query methods

    def query_article(self, ref: str) -> GraphQueryResult:
        """
        This looks up an article with its full context (i.e., instrument, definitions,
        references, equivalents, obligations)

        Parameters:
        ref: article node_id, e.g., "gdpr:Article 17" or "fadp:Art. 7"

        Returns:
        GraphQueryResult with the article node, related nodes, and relationships,
        or an empty result if not found
        """
        return self.queries.get_article_context(ref)

    def query_references(self, ref: str, depth: int = 1) -> GraphQueryResult:
        """
        This traverses "REFERENCES" relationships up to "depth" hops away from an article

        Parameters:
        ref: the article node_id to begin from
        depth: the hops to follow (1 to 5, default is 1)

        Returns:
        the GraphQueryResult with all the discovered nodes and "REFERENCES" relationships
        """
        return self.queries.get_references(ref, depth)

    def status(self) -> GraphStats:
        """
        This returns the current node and relationship counts
        """
        self.store.connect()
        return self.store.get_stats()

    # For the formatting

    @staticmethod
    def format_article_context(result: GraphQueryResult) -> str:
        """
        This formats a GraphQueryResult from "query_article()" into human readable string

        It shows the heading, chapter/section, text preview (200 chars), definitions,
        outgoing/incoming references, cross jurisdictional equivalents, and obligations.

        Returns " Article not found. " if result is empty
        """
        if result.is_empty:
            return " Article not found. "

        lines: list[str] = []
        article = result.nodes[0] if result.nodes else {}

        label = article.get("article_label", "")
        source = article.get("source_id", "")
        chapter = article.get("chapter", "")
        section = article.get("section", "")

        lines.append(f"  {source.upper()} {label}")
        if chapter:
            lines.append(f" Chapter: {chapter}")
        if section:
            lines.append(f" Section: {section}")

        text = article.get("full_text", "")
        if text:
            preview = text[:200].replace("\n", " ")
            if len(text) > 200:
                preview += "..."
            lines.append(f" Text: {preview}")

        # categorizes the relationships by type and direction
        defs = [r for r in result.relationships if r.get("type") == "DEFINES"]
        refs_out = [
            r
            for r in result.relationships
            if r.get("type") == "REFERENCES" and r.get("from") == f"{source}:{label}"
        ]
        refs_in = [
            r
            for r in result.relationships
            if r.get("type") == "REFERENCES" and r.get("to") == f"{source}:{label}"
        ]
        equivs = [r for r in result.relationships if r.get("type") == "EQUIVALENT_TO"]
        obligations = [r for r in result.relationships if r.get("type") == "IMPOSES"]

        if defs:
            lines.append(f"\n Definitions ({len(defs)}):")
            for d_rel in defs:
                d_node = next(
                    (n for n in result.nodes if n.get("node_id") == d_rel.get("to")), {}
                )
                lines.append(
                    f" - {d_node.get('term', '?')}: {d_node.get('definition_text', '')[:100]}..."
                )

        if refs_out:
            lines.append(f"\n References OUT ({len(refs_out)}):")
            for r in refs_out:
                lines.append(f"   -> {r.get('to', '?')}")

        if refs_in:
            lines.append(f"\n Referenced by ({len(refs_in)}):")
            for r in refs_in:
                lines.append(f"   <- {r.get('from', '?')}")

        if equivs:
            lines.append(f"\n Cross jurisdictional equivalents:")
            for r in equivs:
                # "EQUIVALENT_TO" is bidirectional
                target = (
                    r.get("to")
                    if r.get("from") == f"{source}:{label}"
                    else r.get("from")
                )
                lines.append(f"   <=> {target}")

        if obligations:
            lines.append(f"\n Obligations ({len(obligations)}):")
            for o_rel in obligations:
                o_node = next(
                    (n for n in result.nodes if n.get("node_id") == o_rel.get("to")), {}
                )
                lines.append(
                    f"   [{o_node.get('obligation_type', '?')}] {o_node.get('description', '')[:100]}"
                )

        return "\n".join(lines)

    # Stats Records

    def _save_stats(self, stats: GraphStats) -> None:
        """
        This saves the build stats to "data/output/graph/build_stats.json"
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.output_dir / "build_stats.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "instruments": stats.instruments,
                    "articles": stats.articles,
                    "definitions": stats.definitions,
                    "obligations": stats.obligations,
                    "contains": stats.contains,
                    "references": stats.references,
                    "defines": stats.defines,
                    "imposes": stats.imposes,
                    "cites": stats.cites,
                    "equivalent_to": stats.equivalent_to,
                },
                f,
                indent=2,
            )
