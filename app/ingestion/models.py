"""
These are the data models for the ingestion pipeline

This covers the full ingestion flow, i.e.:
1. reading regulatory PDFs (GDPR, Swiss FADP, EDPB guidance, etc.)
2. extracting text/tables/structure via PyMuPDF, olmocr, or Mistral Document AI
3. splitting extracted text into legal structure aware chunks
4. and comparing extractors against ground truth annotations
"""

# import libraries
from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DocumentMetadata:
    """
    This identifies a source regulatory document

    Example from EU GDPR:
        source_id         : "gdpr"
        title             : "General Data Protection Regulation"
        instrument_type   : "statute"
        jurisdiction      : "EU"
        effective_date    : "2018-05-25"
        file_path         : Path("data/documents/gdpr_full_text.pdf")
        total_pages       : 88
    """

    source_id: (
        str  # a short unique key used throughout the pipeline (e.g., "gdpr", "fadp")
    )
    title: str

    # "statute", "guidance", "commentary"
    instrument_type: str

    # "EU" or "CH", this enables jurisdiction-filtered search and cross-jurisdictional graph mapping
    jurisdiction: str

    effective_date: (
        str | None
    )  # ISO date string, this is stored for provenance and not used for filtering
    file_path: Path
    total_pages: int = 0  # this is filled in during extraction after the file is opened


@dataclass
class TextBlock:
    """
    this is a positioned text block from the PDF extraction

    It stores text content and spatial metadata so that the chunker can identify
    headers, footnotes, and columns by their position on the page
    """

    text: str
    bbox: tuple[
        float, float, float, float
    ]  # (x0, y0, x1, y1) in points, top left to bottom right

    font_size: float | None = None
    font_name: str | None = None
    is_bold: bool = (
        False  # this is used to identify headings and article titles in the legal documents
    )


@dataclass
class TableData:
    """
    an extracted table stored as a 2-D grid of strings (rows x columns)
    """

    rows: list[list[str]]
    row_count: int
    col_count: int
    page_number: int  # 1-indexed
    bbox: tuple[float, float, float, float] | None = (
        None  # None if position is not available
    )


@dataclass
class TocEntry:
    """
    This is a table of contents entry from the PDF outline

    It is used by the chunker to understand the hierarchical structure
    (chapters -> sections -> articles)
    """

    level: int  # the nesting depth: 1 = top-level (e.g., "Chapter I"), 2 = subheading, etc.
    title: str
    page_number: int  # 1-indexed


@dataclass
class PageResult:
    """the extraction result for a single PDF page"""

    page_number: int  # 1-indexed
    raw_text: str  # the full page text in reading order, this is what gets chunked
    blocks: list[TextBlock] = field(
        default_factory=list
    )  # positioned blocks, empty for some extractors like olmocr
    tables: list[TableData] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class ExtractionResult:
    """
    the full extraction output for one document with one extractor

    It is passed directly to the chunker, which then splits full_text into Chunk objects
    """

    extractor_name: str  # "pymupdf" or "olmocr" or "mistral"
    metadata: DocumentMetadata
    pages: list[PageResult]
    full_text: str  # all pages concatenated in order
    tables: list[TableData]  # all tables aggregated across pages
    toc_entries: list[TocEntry]
    total_processing_time_ms: float


@dataclass
class Chunk:
    """
    this is a legal structure aware chunk that is ready for embedding and storage in Weaviate

    cross-references (e.g., "see Article 6(1)") are captured so that the graph layer
    can build REFERENCES edges between articles
    """

    # 16char hex ID from SHA-256(source_id + article + chunk_index)
    chunk_id: str

    text: str  # embedded into a vector for semantic search
    source_id: str

    # this is copied from DocumentMetadata so retrieval can filter without document lookup
    instrument_type: str
    jurisdiction: str

    chapter: str | None = None
    section: str | None = None
    article: str | None = (
        None  # key structural tag which links to ArticleNode in the graph layer
    )
    paragraph: str | None = (
        None  # this is used when a long article is split into multiple chunks
    )

    page_numbers: list[int] = field(default_factory=list)
    cross_references: list[str] = field(
        default_factory=list
    )  # detected by regex in the chunker
    has_table: bool = False
    has_footnote: bool = (
        False  # footnotes often contain important qualifications/exceptions
    )
    chunk_index: int = 0
    extractor_name: str = ""

    @staticmethod
    def make_id(source_id: str, article: str | None, chunk_index: int) -> str:
        """
        This generates a 16-character hex ID for a chunk

        It uses SHA-256 hashing so that rerunning the ingestion produces the same ID

        Args:
        source_id:    the document identifier (e.g. "gdpr")
        article:      the article label (e.g., "Article 5"), or None
        chunk_index:  Chunk's position in the document's chunk list

        Returns:
        a 16-character hex string, e.g., "a3f7c2e901b4d8e5"
        """
        # colon separators prevent collisions between adjacent numeric fields
        key = f"{source_id}:{article or 'none'}:{chunk_index}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class ComparisonResult:
    """
    evaluation scores per document, per extractor

    this feeds into the final report's and was used to select PyMuPDF for production
    """

    extractor_name: str  # "pymupdf" or "olmocr" or "mistral"
    document_id: str  # matches DocumentMetadata.source_id

    # this is manually assigned during ground truth annotation:
    # 1 = simple layout, 2 = moderate (some tables/footnotes), 3 = complex (dense tables, nested footnotes)
    complexity_level: int

    # all scores are 0.0–1.0 (higher is better)

    # the fraction of expected structural elements (headings, chapters, articles) correctly identified
    structure_preservation_score: float | None = None

    # TEDS (Tree-Edit-Distance-based Similarity) for tables, None if the document has no tables
    table_teds_score: float | None = None

    # the fraction of cross-references (e.g., "see Article 6(1)") correctly detected
    cross_reference_detection_rate: float | None = None

    # the fraction of footnotes correctly extracted, None if document has no footnotes
    footnote_preservation_rate: float | None = None

    # raw latency metric
    processing_time_per_page_ms: float = 0.0
