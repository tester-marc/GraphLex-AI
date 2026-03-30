"""
the PyMuPDF-based document extractor, which is the non-AI baseline

All 3 implement BaseExtractor (base_extractor.py) which requires:
- name            : a string identifier ("pymupdf")
- extract()       : extracts an entire PDF document
- extract_page()  : extracts a single page
- is_available()  : whether the extractor can run

PyMuPDF matched or exceeded both VLM-based alternatives on every accuracy
metric (for details see final report).

how PyMuPDF works:
PyMuPDF operates on the PDF's internal drawing commands rather than rendering
pages as images (unlike olmocr)
It parses text coordinates, bounding boxes, font metadata, tables,
as well as the document outline directly.

Per page output:
1. Raw text          : the page text in reading order as a single string
2. Text blocks       : regions with bounding boxes and font info (size, name,
                       bold), used by the chunker for heading/footnote detection
3. Tables            : row x column grids with PyMuPDF's built-in table finder
4. Table of contents : PDF outline/bookmarks, extracted once per document

Outputs are packaged into "models.py" classes:
TextBlock, TableData, TocEntry, PageResult, ExtractionResult
"""

# import libraries
from __future__ import annotations
import time
from pathlib import Path
import fitz  # PyMuPDF, installed as "pip install PyMuPDF", imported as "fitz"
from app.ingestion.base_extractor import BaseExtractor
from app.ingestion.models import (
    DocumentMetadata,
    ExtractionResult,
    PageResult,
    TableData,
    TextBlock,
    TocEntry,
)


class PyMuPDFExtractor(BaseExtractor):
    """
    This extracts structured content from PDFs using PyMuPDF

    It operates on the PDF command stream rather than visual rendering
    """

    @property
    def name(self) -> str:
        """
        extractor identifier: "pymupdf"

        This is used by the comparison harness to label the results, by the chunker to
        tag chunks (Chunk.extractor_name), and in output file names
        """
        return "pymupdf"

    # public interface, extract an entire document

    def extract(self, file_path: Path, metadata: DocumentMetadata) -> ExtractionResult:
        """
        This extracts structured content from an entire PDF file

        It opens the PDF, iterates every page, extracts text/blocks/tables,
        collects the TOC, and packages everything into an ExtractionResult

        Args:
        file_path: the Path to the PDF file on disk
        metadata: DocumentMetadata, total_pages is updated with the actual count

        Returns:
        ExtractionResult with all pages, full text, tables, TOC, and timing
        """
        start = time.perf_counter()

        # str(file_path): older PyMuPDF versions don't accept Path objects directly
        doc = fitz.open(str(file_path))
        metadata.total_pages = len(doc)

        # doc.get_toc() -> [[level, title, page_number], ...]
        # level 1 = chapter, level 2 = section, etc.
        toc_entries = [
            TocEntry(level=entry[0], title=entry[1], page_number=entry[2])
            for entry in doc.get_toc()
        ]

        pages: list[PageResult] = []
        all_tables: list[TableData] = []
        text_parts: list[str] = []

        # PyMuPDF uses 0-based page indexing, the models use 1-based
        for page_idx in range(len(doc)):
            page_result = self._extract_page(doc, page_idx)
            pages.append(page_result)
            text_parts.append(page_result.raw_text)
            all_tables.extend(page_result.tables)

        doc.close()

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExtractionResult(
            extractor_name=self.name,
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(text_parts),
            tables=all_tables,
            toc_entries=toc_entries,
            total_processing_time_ms=elapsed_ms,
        )

    # public interface, extract a single page

    def extract_page(self, file_path: Path, page_number: int) -> PageResult:
        """
        this extracts a single page from a PDF file

        Args:
        file_path:   the Path to the PDF file on disk
        page_number: 1-indexed page number (first page = 1)

        Returns:
        a PageResult for the requested page

        Raises:
        ValueError: if page_number is out of range
        """
        doc = fitz.open(str(file_path))

        if page_number < 1 or page_number > len(doc):
            doc.close()  # close before raising
            raise ValueError(f"Page {page_number} out of range (1-{len(doc)})")

        # converts from 1-indexed to 0-indexed (PyMuPDF internal)
        result = self._extract_page(doc, page_number - 1)
        doc.close()
        return result

    # Private methods: extraction logic

    def _extract_page(self, doc: fitz.Document, page_idx: int) -> PageResult:
        """
        this extracts a single page by 0-based index from an open document

        It pulls 3 types of data:
        1. structured text blocks (positions, font info, bold detection)
        2. plain text (reading order string)
        3. tables (structured row x column data)

        Args:
        doc:       An already open PyMuPDF document
        page_idx:  0-based page index

        Returns:
        a PageResult with text, blocks, tables, and timing
        """
        start = time.perf_counter()
        page = doc[page_idx]

        # page.get_text("dict") returns the full block -> line -> span hierarchy
        # with bounding boxes and font metadata per span
        # TEXT_PRESERVE_WHITESPACE keeps indentation, which carries structural
        # meaning in legal text (e.g.. numbered subparagraphs)
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = self._parse_blocks(page_dict)

        # simple reading order string
        raw_text = page.get_text("text")

        # page_idx + 1: converts 0-based to 1-based for TableData.page_number
        tables = self._extract_tables(page, page_idx + 1)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return PageResult(
            page_number=page_idx + 1,
            raw_text=raw_text,
            blocks=blocks,
            tables=tables,
            processing_time_ms=elapsed_ms,
        )

    def _parse_blocks(self, page_dict: dict) -> list[TextBlock]:
        """
        this parses structured blocks from PyMuPDF's dict output

        It traverses the blocks -> lines -> spans hierarchy and then converts each text
        block into a TextBlock, calculating aggregate font metrics:
        - average font size across all spans
        - primary (most frequent) font name
        - bold: this is true if more than half the spans use a bold font

        Args:
        page_dict: Dict returned by page.get_text("dict")

        Returns:
        a list of TextBlock objects, image blocks (type != 0) and empty
        blocks are skipped
        """
        text_blocks: list[TextBlock] = []

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # 0 = text, 1 = image
                continue

            block_text_parts: list[str] = []
            font_sizes: list[float] = []
            font_names: list[str] = []
            bold_flags: list[bool] = []

            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    font_sizes.append(span.get("size", 0))
                    font_names.append(span.get("font", ""))

                    # checks for "bold" in font name
                    bold_flags.append("bold" in span.get("font", "").lower())

                block_text_parts.append(line_text)

            text = "\n".join(block_text_parts).strip()
            if not text:
                continue

            # bbox: [x0, y0, x1, y1] in PDF points pt,
            # origin is at top left of the page
            bbox = (
                block["bbox"][0],
                block["bbox"][1],
                block["bbox"][2],
                block["bbox"][3],
            )

            avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else None

            # mode of font_names
            primary_font = (
                max(set(font_names), key=font_names.count) if font_names else None
            )

            # majority vote bold: this is true if more than half of spans are bold
            is_bold = sum(bold_flags) > len(bold_flags) / 2 if bold_flags else False

            text_blocks.append(
                TextBlock(
                    text=text,
                    bbox=bbox,
                    font_size=avg_font_size,
                    font_name=primary_font,
                    is_bold=is_bold,
                )
            )

        return text_blocks

    def _extract_tables(self, page: fitz.Page, page_number: int) -> list[TableData]:
        """
        this extracts tables from a page using PyMuPDF's built-in table finder

        Args:
        page:         fitz.Page to extract tables from
        page_number:  1-indexed page number for TableData records

        Returns:
        a list of TableData objects, an empty list if none found or on error
        """
        tables: list[TableData] = []
        try:
            found = page.find_tables()

            for table in found.tables:
                extracted = table.extract()
                if not extracted:
                    continue

                # normalize
                rows = [
                    [cell if cell is not None else "" for cell in row]
                    for row in extracted
                ]

                if not rows:
                    continue

                tables.append(
                    TableData(
                        rows=rows,
                        row_count=len(rows),
                        col_count=len(rows[0]) if rows else 0,
                        page_number=page_number,
                        # bbox was introduced in a newer PyMuPDF version
                        bbox=table.bbox if hasattr(table, "bbox") else None,
                    )
                )
        except Exception:
            pass
        return tables
