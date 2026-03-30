# base_extractor.py: the base class for the document understanding extractors
#
# This defines the interface that each extractor in the GraphLex AI ingestion pipeline
# has to follow. It allows the comparison harness and rest of the system to use
# any extractor interchangeably
#
# there are 3 extractors for this interface:
# - pymupdf_extractor.py: this uses PyMuPDF (which was ultimately chosen as the extractor for production)
# - olmocr_extractor.py:  this uses Allen AI's olmocr (a VLM, vision language model)
# - mistral_extractor.py: this uses Mistral's "Document AI" API
#

# import libraries
from __future__ import annotations  # for lazy type hints
from abc import ABC, abstractmethod
from pathlib import Path  # for cross platform handling of paths

# - DocumentMetadata: this identifies a source document (source_id, title,
#   instrument_type, jurisdiction, effective_date, file_path)
# - ExtractionResult: this is the full output from one extractor on 1 document, i.e.,
#   page results, full text, tables, TOC entries, and timing
# - PageResult: this is the output for a single page, i.e., 1-indexed page number, raw text,
#   structured blocks, tables, and extraction latency in milliseconds
from app.ingestion.models import DocumentMetadata, ExtractionResult, PageResult


class BaseExtractor(ABC):
    """This is the interface that all document understanding tools have to implement

    The subclasses have to provide:
    1. "name" property       : a string identifier (e.g., "pymupdf")
    2. "extract" method      : this processes an entire PDF
    3. "extract_page" method : this processes a single page
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """extractor identifier, e.g., 'pymupdf', 'olmocr', or 'mistral'

        This is used in the output paths (data/output/pymupdf/), comparison tables,
        and the log messages
        """

    @abstractmethod
    def extract(self, file_path: Path, metadata: DocumentMetadata) -> ExtractionResult:
        """this extracts ther structured content from a PDF file --
        It returns the page text, tables, TOC entries, and timing information
        """

    @abstractmethod
    def extract_page(self, file_path: Path, page_number: int) -> PageResult:
        """this extracts a single page (1-indexed)
        It is of use for testing, per page comparison, or debugging
        complex layouts without processing the whole document
        """

    def is_available(self) -> bool:
        """whether this extractor can currentlybe run --
        true by default. The comparison harness calls this before each run.
        """
        return True
