"""
This is for the Ingestion pipeline configuration (Layer 1)

It defines the document registry and "IngestionConfig" for GraphLex AI's Ingestion Layer, i.e.,
reading regulatory PDFs, extracting text while it preserves the legal structure, and
splitting it into chunks for embedding and retrieval

Responsibilities:
1. locates project root for path resolution
2. defines the registry of 6 regulatory documents (2 statutes, and 4 guidance docs)
3. provides "IngestionConfig", which consolidates all the pipeline settings
"""

# import libraries
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# for the absolute path to the project root (<project_root>/GraphLex AI/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# Document entry: metadata for each regulatory PDF


@dataclass
class DocumentEntry:
    """
    This is the per document config for one regulatory PDF

    It is used throughout the pipeline:
    - Chunker uses "instrument_type" to choose the splitting strategy
      (article first for statutes, section based for guidance)
    - the retrieval layer uses "jurisdiction" and "instrument_type" as metadata filters
    - the graph layer uses "source_id" to link articles to their parent instrument node
    - and the extractor comparison uses "complexity_level" to analyze the layout difficulty
    """

    # PDF filename in data/documents/, e.g., "gdpr_full_text.pdf"
    filename: str

    # this is a short unique identifier used as a key in the output dirs, chunk IDs, and graph nodes
    # e.g., "gdpr", "fadp", "edpb_consent"
    source_id: str

    # the full legal title
    title: str

    # the legal instrument type which drives the splitting strategy of the chunker:
    # "statute"     : the law enacted by a legislature, it is split at article boundaries
    # "guidance"    : supervisory authority guidance, it is split at section headings
    # "commentary"  : 3rd party analysis (generally supported but not in current corpus for this prototype)
    instrument_type: str  # "statute", "guidance", "commentary"

    # jurisdiction used for metadata filtering and cross-jurisdictional mapping:
    # "EU" : EU law (GDPR and EDPB guidelines)
    # "CH" : Swiss law (FADP and FDPIC guidance)
    jurisdiction: str  # "EU", "CH"

    # ISO 8601 effective date (in thr format YYYY-MM-DD) or None if not specified
    effective_date: str | None

    # for the layout complexity used in the extractor comparison:
    # 1 = simple (flowing text, few tables / footnotes)
    # 2 = moderate (mix of text, tables, numbered lists, footnotes)
    # 3 = complex (dense tables, multilevel numbering, many cross-references)
    complexity_level: int  # 1, 2, or 3


# Document registry (the six regulatory PDFs processed by the GraphLex AI prototype)

# 2 statutes (GDPR, FADP) and 4 guidance docs covering Swiss-EU data
# protection compliance from the EDPB and FDPIC authorities
DOCUMENT_REGISTRY: list[DocumentEntry] = [
    # statute 1: EU General Data Protection Regulation: 99 articles, 173 recitals
    # this is the most complex layout in the corpus (because of the dense numbering, extensive cross-references)
    DocumentEntry(
        filename="gdpr_full_text.pdf",
        source_id="gdpr",
        title="General Data Protection Regulation (EU) 2016/679",
        instrument_type="statute",
        jurisdiction="EU",
        effective_date="2018-05-25",
        complexity_level=3,
    ),
    # statute 2: Swiss Federal Act on Data Protection (revised 2023): 74 articles
    # the layout is more simple than GDPR and 14 articles are mapped to GDPR equivalents in the graph layer
    DocumentEntry(
        filename="fadp_revised_2023.pdf",
        source_id="fadp",
        title="Swiss Federal Act on Data Protection (FADP) 235.1",
        instrument_type="statute",
        jurisdiction="CH",
        effective_date="2023-09-01",
        complexity_level=1,
    ),
    # guidance 1: EDPB on legitimate interest (GDPR Art. 6(1)(f))
    DocumentEntry(
        filename="edpb_guidelines_legitimate_interest.pdf",
        source_id="edpb_legitimate_interest",
        title="EDPB Guidelines 1/2024 on Legitimate Interest (Art. 6(1)(f) GDPR)",
        instrument_type="guidance",
        jurisdiction="EU",
        effective_date="2024-10-08",
        complexity_level=2,
    ),
    # guidance 2: EDPB on Article 48 (international data transfers)
    # this is relevant for Swiss-domiciled companies subject to both GDPR and FADP
    DocumentEntry(
        filename="edpb_guidelines_article48_transfers.pdf",
        source_id="edpb_article48",
        title="EDPB Guidelines 02/2024 on Article 48 GDPR",
        instrument_type="guidance",
        jurisdiction="EU",
        effective_date="2025-06-01",
        complexity_level=2,
    ),
    # guidance 3: EDPB on consent. It defines "freely given", "specific",
    # "informed", and "unambiguous" requirements
    DocumentEntry(
        filename="edpb_guidelines_consent.pdf",
        source_id="edpb_consent",
        title="EDPB Guidelines 05/2020 on Consent under Regulation 2016/679",
        instrument_type="guidance",
        jurisdiction="EU",
        effective_date="2020-05-04",
        complexity_level=2,
    ),
    # guidance 4: FDPIC on technical and organisational measures
    # this is the only Swiss guidance in the corpus, the effective date was not specified
    DocumentEntry(
        filename="fdpic_guide_technical_measures.pdf",
        source_id="fdpic_technical_measures",
        title="FDPIC Guide to Technical and Organisational Data Protection Measures",
        instrument_type="guidance",
        jurisdiction="CH",
        effective_date=None,
        complexity_level=2,
    ),
]


# IngestionConfig: the top level settings for the ingestion pipeline


@dataclass
class IngestionConfig:
    """
    This is the top level ingestion pipeline config
    """

    # the source PDF folder, the default is <project_root>/data/documents/
    documents_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "documents"
    )

    # the output folder for the extraction results, every extractor gets a subfolder
    # (e.g., output/pymupdf/gdpr/), the default is <project_root>/data/output/
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "output")

    # the "ground truth JSON" folder for the extractor evaluation (1 file per document)
    # default is <project_root>/data/ground_truth/
    ground_truth_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data" / "ground_truth"
    )

    # the documents to process. this defaults to all 6
    documents: list[DocumentEntry] = field(
        default_factory=lambda: list(DOCUMENT_REGISTRY)
    )

    # 3 text extractors to run:
    # "pymupdf"   : fast and rule-based, this was ultimately chosen as the production extractor
    # "olmocr"    : VLM-based (Allen AI), this was precomputed on A100 GPU available via HuggingFace
    # "mistral"   : Mistral Document AI cloud API
    active_extractors: list[str] = field(default_factory=lambda: ["pymupdf"])

    # Chunking parameters
    # statutes are split at the article boundaries, the guidance docs at section headings

    # the max tokens per chunk
    max_chunk_tokens: int = 512

    # the token overlap between consecutive chunks, this is important to avoid
    # sentences from being cut at the boundaries and also for cross-references
    # that span a split so that they appear in at least one chunk
    chunk_overlap_tokens: int = 64

    def get_document(self, source_id: str) -> DocumentEntry | None:
        """
        To look up a document by its "source_id"

        Args:
        source_id: a short identifier, e.g., "gdpr", "fadp", "edpb_consent"

        Returns:
        the matching DocumentEntry, or None if it's not found
        """
        for doc in self.documents:
            if doc.source_id == source_id:
                return doc
        return None

    def document_path(self, entry: DocumentEntry) -> Path:
        """
        This returns the full file system path to a document's PDF file

        Args:
        entry: DocumentEntry whose PDF path is needed

        Returns:
        path to PDF file (documents_dir / entry.filename)
        """
        return self.documents_dir / entry.filename
