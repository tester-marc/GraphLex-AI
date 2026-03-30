"""
the data models for the Retrieval Layer

This defines "SearchResult" which represents a Weaviate vector similarity
result

this is created by: "weaviate_store.py"
and consumed by: "pipeline.py", "nodes.py", and "app.py" (Gradio tab for the evidence)
"""

# import libraries
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """
    this is a single retrieval result from Weaviate

    It has: chunk content, source / location metadata, cross-references, and
    relevance scores to be used by layers downstream
    """

    # Identification

    # format is "{source_id}:{article_or_section}:{chunk_index}", e.g., "gdpr:Article 32:0"
    chunk_id: str

    # this is the raw text extracted from the regulatory PDF (through PyMuPDF) which is split on
    # the legal structure boundaries (article, section, paragraph)
    text: str

    # metadata for sources

    # a short doc identifier
    # Corpus IDs:
    # "gdpr"                        : General Data Protection Regulation
    # "fadp"                        : Swiss Federal Act on Data Protection
    # "edpb_consent"                : EDPB Guidelines on Consent
    # "edpb_article48"              : EDPB Guidelines on Article 48 Transfers
    # "edpb_legitimate_interest"    : EDPB Guidelines on Legitimate Interest
    # "fdpic_technical_measures"    : FDPIC Guide on Technical Measures
    source_id: str

    # the legal authority tier (which is set in "app/ingestion/config.py"):
    # "statute"     : the binding primary legislation (GDPR, FADP)
    # "guidance"    : regulator guidelines (EDPB, FDPIC)
    # "commentary"  : a secondary commentary analysis (currently not part of the corpus in this prototype)
    instrument_type: str

    # jurisdiction, i.e.,  "EU" (GDPR, EDPB) or "CH" (FADP, FDPIC)
    # this is used by the Graph Layer in order to find cross-jurisdictional equivalences
    jurisdiction: str

    # metadata for location

    # the article within the document, e.g., "Article 32" or "Art. 8"
    # None for guidance docs organized by section rather than article
    # this is used by the graph expansion to look up related nodes in Neo4j
    article: str | None

    # a section heading, e.g., "4.1 Definition of consent", or None for statute
    # articles that don't have subsections
    section: str | None

    # paragraph number, e.g., "1" or None when NA
    paragraph: str | None

    # cross-references

    # for articles referenced within this chunk's text (detected by the regex in
    # "app/ingestion/chunker.py")
    # this is stored in Weaviate (for filtering) and Neo4j
    # (as "REFERENCES" relationships)
    cross_references: list[str]

    # relevance scores

    # the cosine distance from Weaviate (0 is identical, 1 us orthogonal)
    distance: float

    # the convenience score: 1 - distance
    # higher is more relevant
    score: float  # 1 - distance (cosine similarity proxy)

    # calculated display properties

    @property
    def authority_label(self) -> str:
        """this converts "instrument_type" to a more human readable label for the UI and CLI display"""
        labels = {
            "statute": "Statute",
            "guidance": "Regulator Guidance",
            "commentary": "Commentary",
        }
        return labels.get(self.instrument_type, self.instrument_type)

    @property
    def location_label(self) -> str:
        """
        location string, e.g., "Article 32 > 1"
        it joins whichever of article, section, paragraph are not None with ">"
        and returns "(no location)" if all are None
        """
        parts = []
        if self.article:
            parts.append(self.article)
        if self.section:
            parts.append(self.section)
        if self.paragraph:
            parts.append(self.paragraph)
        return " > ".join(parts) if parts else "(no location)"
