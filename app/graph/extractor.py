"""
This script handles the entity and relationship extraction from the regulatory chunks.

It reads the text chunks from the Ingestion Layer (Layer 1) and creates structured
graph entities (nodes) and relationships (edges) for Neo4j (Layer 4).

Nodes:
- Instrument  : this is a regulatory document (e.g., "GDPR", "FADP")
- Article     : an individual article or provision in an instrument
- Definition  : legal term which is defined in a definitions article
- Obligation  : for an obligation, right, prohibition, or permission
                extracted from an article's text by the LLM

Relationships:
- CONTAINS      :   Instrument -> Article
- REFERENCES    :   Article -> Article (cross-references)
- CITES         :   Instrument (guidance) -> Article (statute)
- DEFINES       :   Article -> Definition
- IMPOSES       :   Article -> Obligation
- EQUIVALENT_TO : FADP Article <-> GDPR Article (for parallel provisions)

There are 2 extraction strategies:
1. Structural/regex - instruments, articles, definitions, references,
   citations, and equivalences
2. Semantic/LLM - obligations extracted by the LLM "Qwen3-Next 80B-A3B" via
   Together AI. This returns JSON parsed into Obligation nodes.
   The results are cached to data/output/graph/obligations_cache.json
"""

# import libraries
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
from app.ingestion.config import DOCUMENT_REGISTRY


# Constants on the module level

# this collapses whitespace runs (including newlines from the PDF line breaks)
# to a single space, e.g., "Article\n27" will become "Article 27"
_RE_WHITESPACE = re.compile(r"\s+")

# this extracts article number from strings like "Article 28", "Art. 5", "Article 16a"
_RE_ART_NUM = re.compile(r"(?:Article|Art\.)\s+(\d+[a-z]?)")

# this maps every source_id to the statute that its cross-references most likely have as target
# e.g., EDPB guidance interprets the GDPR and FDPIC guidance interprets the FADP
_SOURCE_TO_STATUTE: dict[str, str] = {
    "gdpr": "gdpr",
    "edpb_legitimate_interest": "gdpr",
    "edpb_article48": "gdpr",
    "edpb_consent": "gdpr",
    "fadp": "fadp",
    "fdpic_technical_measures": "fadp",
}

# this is hard coded knowledge, i.e., parallel provisions that are known between FADP and GDPR
# they are stored as "EQUIVALENT_TO" relationships
# each tuple is: (FADP node_id, GDPR node_id, description)
_CROSS_JURISDICTIONAL: list[tuple[str, str, str]] = [
    ("fadp:Art. 1", "gdpr:Article 1", "Purpose"),
    ("fadp:Art. 2", "gdpr:Article 2", "Scope of application"),
    ("fadp:Art. 3", "gdpr:Article 3", "Territorial scope"),
    ("fadp:Art. 5", "gdpr:Article 4", "Definitions"),
    ("fadp:Art. 6", "gdpr:Article 5", "Processing principles"),
    ("fadp:Art. 7", "gdpr:Article 25", "Data protection by design"),
    ("fadp:Art. 8", "gdpr:Article 35", "Data protection impact assessment"),
    ("fadp:Art. 9", "gdpr:Article 28", "Processor obligations"),
    ("fadp:Art. 16", "gdpr:Article 33", "Breach notification to authority"),
    ("fadp:Art. 17", "gdpr:Article 34", "Breach notification to data subject"),
    ("fadp:Art. 19", "gdpr:Article 13", "Information obligation (collection)"),
    ("fadp:Art. 25", "gdpr:Article 15", "Right of access"),
    ("fadp:Art. 28", "gdpr:Article 20", "Data portability"),
    ("fadp:Art. 32", "gdpr:Article 17", "Right to erasure"),
    ("fadp:Art. 16a", "gdpr:Article 37", "Data protection officer"),
]

# GraphExtractor class


class GraphExtractor:
    """
    This class extracts the entities and the relationships from the regulatory chunks

    It is used by GraphPipeline (in: pipeline.py).

    extractor = GraphExtractor(chunks_dir=config.chunks_dir)

    # For structural extraction:
    instruments    : extractor.extract_instruments()
    articles       : extractor.extract_articles()
    contains       : extractor.extract_contains()
    references     : extractor.extract_references()
    cites          : extractor.extract_cites()
    equivalents    : extractor.extract_equivalents()
    defs, def_rels : extractor.extract_definitions()

    # for LLM obligation extraction:
    candidates     : extractor.get_obligation_candidates()
    obs, obs_rels  : GraphExtractor.parse_obligation_response(...)

    All of these methods return plain dicts since Neo4j store's batch
    insert methods expect dicts for UNWIND queries
    """

    def __init__(self, chunks_dir: Path) -> None:
        """
        Args:
        chunks_dir: The directory which contains the PyMuPDF extraction output
        Each subdirectory (e.g., chunks_dir/gdpr/) holds a chunks.json file.
        """
        self.chunks_dir = chunks_dir
        self._chunks: list[dict] | None = None  # loaded cache
        self._article_index: dict[str, dict] = {}  # node_id -> article dict

    # Loading of chunks

    def load_chunks(self) -> list[dict]:
        """
        This function loads all the PyMuPDF chunks from disk

        Every chunk dict has the keys: text, source_id, article, chapter, section,
        cross_references, instrument_type. The function returns all the chunks for all 6
        regulatory documents (817 in total with the corpus for this project)
        """
        if self._chunks is not None:
            return self._chunks

        chunks = []
        for source_dir in sorted(self.chunks_dir.iterdir()):
            chunks_file = source_dir / "chunks.json"
            if not chunks_file.exists():
                continue
            with open(chunks_file, encoding="utf-8") as f:
                doc_chunks = json.load(f)
            chunks.extend(doc_chunks)

        self._chunks = chunks
        return chunks

    # Instruments:

    def extract_instruments(self) -> list[dict]:
        """
        This function creates instrument node dicts from DOCUMENT_REGISTRY

        It returns 1 dict per regulatory document (so 6 in total), with the
        following keys:
        source_id, title, instrument_type, jurisdiction, effective_date
        """
        return [
            {
                "source_id": doc.source_id,
                "title": doc.title,
                "instrument_type": doc.instrument_type,  # "statute" or "guidance"
                "jurisdiction": doc.jurisdiction,  # "EU" or "CH" (= Switzerland)
                "effective_date": doc.effective_date or "",
            }
            for doc in DOCUMENT_REGISTRY
        ]

    # Articles:

    def extract_articles(self) -> list[dict]:
        """
        This function extracts the unique article nodes from the chunk metadata

        It groups the chunks by article, concatenates the article text,
        and builds "self._article_index", which is a lookup dict
        that is used by all extraction methods that follow

        It returns 174 article dicts (100 GDPR and 74 from FADP), each one
        with the following keys:
        node_id, source_id, article_label, chapter, section, full_text.
        """
        chunks = self.load_chunks()
        self._article_index.clear()
        articles: list[dict] = []

        for chunk in chunks:
            label = chunk.get("article")
            if not label:
                continue

            # format for "node_id": "gdpr:Article 5" or "fadp:Art. 7"
            source_id = chunk["source_id"]
            node_id = f"{source_id}:{label}"

            if node_id in self._article_index:
                # if the article goes over multiple chunks, append text
                self._article_index[node_id]["full_text"] += "\n" + chunk["text"]
                continue

            art = {
                "node_id": node_id,
                "source_id": source_id,
                "article_label": label,
                "chapter": chunk.get("chapter") or "",
                "section": chunk.get("section") or "",
                "full_text": chunk["text"],
            }
            articles.append(art)
            self._article_index[node_id] = art

        # it is capped at 2000 chars, but the full text is still available in original chunks
        for art in articles:
            if len(art["full_text"]) > 2000:
                art["full_text"] = art["full_text"][:2000] + "..."

        return articles

    # "CONTAINS" relationships: Instrument -> Article

    def extract_contains(self) -> list[dict]:
        """
        This function creates "CONTAINS" relationships which link each Instrument to its Articles

        It requires "extract_articles()" to be called first
        And it returns 174 relationship dicts with the following keys:
        source_id, article_node_id
        """
        rels = []
        for node_id, art in self._article_index.items():
            rels.append(
                {
                    "source_id": art["source_id"],
                    "article_node_id": node_id,
                }
            )
        return rels

    # "REFERENCES" relationships: Article -> Article

    def extract_references(self) -> list[dict]:
        """
        This function extracts "REFERENCES" relationships from the chunk cross reference metadata

        The chunker stores cross-references that are detected (e.g., "Article 28",
        "Art. 6(1)(f)") in the cross_references list of each chunk. This method
        resolves these strings to Article node_ids within the graph

        This is only for statute to statute references, and guidance citations
        are handled with "extract_cites()"

        This function requires "extract_articles()" first, and
        returns unique relationship dicts with the following keys:
        source_node_id, target_node_id, reference_text
        """
        chunks = self.load_chunks()
        refs: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for chunk in chunks:
            source_article = chunk.get("article")
            if not source_article:
                continue

            source_id = chunk["source_id"]
            source_node_id = f"{source_id}:{source_article}"

            if source_node_id not in self._article_index:
                continue

            for ref_text in chunk.get("cross_references", []):
                target = self._resolve_reference(ref_text, source_id)

                # skips unresolved references as well as self references
                if not target or target == source_node_id:
                    continue

                key = (source_node_id, target)
                if key in seen:
                    continue
                seen.add(key)

                refs.append(
                    {
                        "source_node_id": source_node_id,
                        "target_node_id": target,
                        "reference_text": _RE_WHITESPACE.sub(" ", ref_text).strip(),
                    }
                )

        return refs

    # "CITES" relationships: Instrument (guidance) -> Article (statute)

    def extract_cites(self) -> list[dict]:
        """
        This function extracts "CITES" relationships: i.e., guidance documents that cite statute articles

        This is different from REFERENCES in the following:
        CITES links an Instrument (guidance) to a statute Article, rather than an Article
        to an Article. Guidance documents are linked at instrument level
        because they don't have numbered articles

        It returns unique relationship dicts with the following keys:
        source_id, article_node_id
        """
        chunks = self.load_chunks()
        rels: list[dict] = []
        seen: set[tuple[str, str]] = set()

        for chunk in chunks:
            source_id = chunk["source_id"]

            if chunk.get("instrument_type") != "guidance":
                continue

            for ref_text in chunk.get("cross_references", []):
                target = self._resolve_reference(ref_text, source_id)
                if not target:
                    continue

                key = (source_id, target)
                if key in seen:
                    continue
                seen.add(key)

                rels.append(
                    {
                        "source_id": source_id,
                        "article_node_id": target,
                    }
                )

        return rels

    # "EQUIVALENT_TO" relationships: FADP Article <-> GDPR Article

    def extract_equivalents(self) -> list[dict]:
        """
        This function extracts "EQUIVALENT_TO" relationships from "_CROSS_JURISDICTIONAL"

        It creates an edge only if both articles exist in the index of articles and
        returns relationship dicts with the following keys:
        fadp_node_id, gdpr_node_id, note
        """
        rels = []
        for fadp_id, gdpr_id, note in _CROSS_JURISDICTIONAL:
            if fadp_id in self._article_index and gdpr_id in self._article_index:
                rels.append(
                    {
                        "fadp_node_id": fadp_id,
                        "gdpr_node_id": gdpr_id,
                        "note": note,
                    }
                )
        return rels

    # Extract the definitions (regex based from Art.4 GDPR, Art.5 FADP)

    def extract_definitions(self) -> tuple[list[dict], list[dict]]:
        """
        This function extracts the Definition nodes and the DEFINES relationships

        The articles GDPR Article 4 and FADP Art. 5 are the dedicated articles for definitions
        But they use different formatting and therefore require different regex patterns:
        GDPR Art.4: "1. 'personal data' means ..." : 1 definition per chunk
        FADP Art.5: "a. personal data means ..."   : all in one chunk, annd split by
        the letter markers (a. - k.)

        The function returns a tuple of (definition dicts, DEFINES relationship dicts)
        """
        chunks = self.load_chunks()
        defs: list[dict] = []
        rels: list[dict] = []

        # definitions GDPR Article 4
        gdpr_art4 = [
            c
            for c in chunks
            if c["source_id"] == "gdpr" and c.get("article") == "Article 4"
        ]

        for chunk in gdpr_art4:
            text = chunk["text"]

            # pattern "1. 'personal data' means any information..."
            m = re.match(
                r"\d+\.\s+['\u2018\u2019]([^'\u2018\u2019]+)['\u2018\u2019]\s+means\s+(.+)",
                text,
                re.DOTALL,
            )
            if m:
                term = m.group(1).strip()
                def_text = m.group(2).strip().rstrip(";")
                node_id = f"gdpr:def:{term.lower()}"

                defs.append(
                    {
                        "node_id": node_id,
                        "term": term,
                        "definition_text": def_text[:500],
                        "source_id": "gdpr",
                        "article_label": "Article 4",
                    }
                )
                rels.append(
                    {
                        "article_node_id": "gdpr:Article 4",
                        "def_node_id": node_id,
                    }
                )

        # definitions FADP Art.5
        fadp_art5 = [
            c
            for c in chunks
            if c["source_id"] == "fadp" and c.get("article") == "Art. 5"
        ]

        for chunk in fadp_art5:
            text = chunk["text"]

            # splits by the letter markers (a. - k.) where the line starts
            # "re.split" keeps letter in the result
            parts = re.split(r"\n([a-k])\.\s+", text)

            # parts[0] is header and pairs that follow are (letter, content)
            for i in range(1, len(parts) - 1, 2):
                content = parts[i + 1].strip()
                m2 = re.match(r"(.+?)\s+means\s+(.+)", content, re.DOTALL)
                if m2:
                    term = m2.group(1).strip()
                    def_text = m2.group(2).strip().rstrip(";")
                    node_id = f"fadp:def:{term.lower()}"

                    defs.append(
                        {
                            "node_id": node_id,
                            "term": term,
                            "definition_text": def_text[:500],
                            "source_id": "fadp",
                            "article_label": "Art. 5",
                        }
                    )
                    rels.append(
                        {
                            "article_node_id": "fadp:Art. 5",
                            "def_node_id": node_id,
                        }
                    )

        return defs, rels

    # Obligations extraction (LLM based, and called externally by the pipeline)

    def get_obligation_candidates(self) -> list[dict]:
        """
        This function returns the statute articles that are suited for the LLM obligation extraction

        It excludes:
        - Definition articles (GDPR Art.4, FADP Art.5), because they are already handled above
        - Articles < 100 chars
        - and Guidance documents (because only statutes carry legal weight)

        It returns 171 of the 174 total articles
        """
        candidates = []
        skip_labels = {"Article 4", "Art. 5"}  # definition articles

        for node_id, art in self._article_index.items():
            if art["source_id"] not in ("gdpr", "fadp"):
                continue
            if art["article_label"] in skip_labels:
                continue
            if len(art["full_text"]) < 100:
                continue
            candidates.append(art)

        return candidates

    @staticmethod
    def parse_obligation_response(
        article_node_id: str,
        source_id: str,
        article_label: str,
        llm_response: str,
    ) -> tuple[list[dict], list[dict]]:
        """
        This method parses the LLM JSON output into Obligation dicts as well as "IMPOSES" relationships

        The LLM gives back a JSON array of objects with the following fields:
        description, type ("obligation","right","prohibition","permission"),
        subject (e.g., "controller", "data subject").

        Node IDs are SHA-256 hashes of (source_id, article_label, description[:50]),
        they are truncated to 16 hex chars, and this is enough to avoid collisions here

        Args:
        article_node_id: the Node ID of the article which is sent to the LLM
        (e.g., "gdpr:Article 33").
        source_id:      the source document (e.g., "gdpr")
        article_label:  the article label (e.g., "Article 33")
        llm_response:   The text response from the LLM

        Returns:
        a tuple of (obligation dicts, IMPOSES relationship dicts)
        """
        obligations: list[dict] = []
        rels: list[dict] = []

        # strips any markdown code fences if there are
        clean = llm_response.strip()
        if clean.startswith("```"):
            clean = re.sub(r"^```(?:json)?\s*", "", clean)
            clean = re.sub(r"\s*```$", "", clean)

        try:
            items = json.loads(clean)
            if not isinstance(items, list):
                return [], []
        except (json.JSONDecodeError, TypeError):
            return [], []

        for item in items:
            desc = str(item.get("description", "")).strip()
            ob_type = str(item.get("type", "obligation")).strip().lower()
            subject = str(item.get("subject", "")).strip()

            if not desc:
                continue

            # unexpected type values are defaulted to "obligation"
            if ob_type not in ("obligation", "right", "prohibition", "permission"):
                ob_type = "obligation"

            # hash the ID: same article text gets the same ID
            node_id = hashlib.sha256(
                f"{source_id}:{article_label}:{desc[:50]}".encode()
            ).hexdigest()[:16]

            obligations.append(
                {
                    "node_id": node_id,
                    "description": desc[:300],
                    "obligation_type": ob_type,
                    "source_id": source_id,
                    "article_label": article_label,
                    "subject": subject,
                }
            )
            rels.append(
                {
                    "article_node_id": article_node_id,
                    "obligation_node_id": node_id,
                }
            )

        return obligations, rels

    # Reference resolution

    def _resolve_reference(self, ref_text: str, source_id: str) -> str | None:
        """
        This function resolves a cross reference string to an existing Article's node_id

        What happens:
          1. Whitespaces normalized ("Article\\n32" -> "Article 32")
          2. Article number is extracted via _RE_ART_NUM
          3. Looks up "_SOURCE_TO_STATUTE" for the target statute that is likely
          4. Tries the primary statute first, and falls back to the other

        FADP uses "fadp:Art. N" and GDPR uses "gdpr:Article N"

        The function returns node_id if found in the article index, otherwise None
        """
        clean = _RE_WHITESPACE.sub(" ", ref_text).strip()

        m = _RE_ART_NUM.match(clean)
        if not m:
            return None

        art_num = m.group(1)
        target_statute = _SOURCE_TO_STATUTE.get(source_id, "gdpr")

        if target_statute == "fadp":
            primary = f"fadp:Art. {art_num}"
            fallback = f"gdpr:Article {art_num}"
        else:
            primary = f"gdpr:Article {art_num}"
            fallback = f"fadp:Art. {art_num}"

        if primary in self._article_index:
            return primary
        if fallback in self._article_index:
            return fallback

        return None
