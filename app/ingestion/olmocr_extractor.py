"""
this is the olmocr extractor, it loads precomputed results from the HF Inference Endpoint run

All 3 extractors implement BaseExtractor (base_extractor.py) which requires:
- name            : a string identifier
- extract()       : extract an entire PDF document
- extract_page()  : extract a single page

Why is this "pre-computed"? :
olmocr is a 7B VLM (vision language model) and I required a GPU (A100) to run it.
It was run once via HuggingFace Inference Endpoint and the results were saved to data/output/olmocr/
This class here simply loads those JSON files, it doesn't run the model itself (which has already
been run).

Pre-computed output format per document:
- pages.json              :  an array of { "page": N, "content": "...", "processing_time_seconds": 2.3 }
- extraction_summary.json :  timing/metadata summary
- full_text.txt           :  the concatenated text
- chunks.json             :  the chunked output (produced by the chunker, not olmocr)
"""

# import libraries
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from app.ingestion.base_extractor import BaseExtractor
from app.ingestion.models import (
    DocumentMetadata,
    ExtractionResult,
    PageResult,
    TableData,
    TextBlock,
    TocEntry,
)


# Constants

# this maps pipeline source_id to filesystem directory name
_SOURCE_TO_DIR = {
    "gdpr": "gdpr",
    "fadp": "fadp",
    "edpb_legitimate_interest": "edpb_legitimate_interest",
    "edpb_article48": "edpb_article48",
    "edpb_consent": "edpb_consent",
    "fdpic_technical_measures": "fdpic_technical_measures",
}

# this resolves to <project_root>/data/output/olmocr/ regardless of where the
# project is cloned
_DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "output" / "olmocr"
)


# OlmOCRExtractor class


class OlmOCRExtractor(BaseExtractor):
    """
    olmocr: fine-tuned 7B VLM that processes pages as images

    It loads pre-computed results from data/output/olmocr/ (extracted via
    HF Inference Endpoint on A100
    """

    def __init__(self, output_dir: Path | None = None):
        """
        Args:
        output_dir: the Path to the pre-computed output directory
                    Each subdirectory (e.g., "gdpr/") holds one document's results
                    This defaults to <project_root>/data/output/olmocr/
        """
        self._output_dir = output_dir or _DEFAULT_OUTPUT_DIR

    @property
    def name(self) -> str:
        """an identifier used to label results and tag chunks"""
        return "olmocr"

    def is_available(self) -> bool:
        """this is true if the pre-computed output directory exists on disk"""
        return self._output_dir.exists()

    def extract(self, file_path: Path, metadata: DocumentMetadata) -> ExtractionResult:
        """
        This loads and returns the full extraction result for one document

        file_path is unused, because this extractor loads from pre-computed JSON
        rather than reading the PDF, but the parameter is required by
        BaseExtractor so that the harness can treat all extractors the same way

        Raises:
        FileNotFoundError: if no pre-computed output exists for source_id
        """
        source_id = metadata.source_id
        # this falls back to source_id itself if not in the map
        dir_name = _SOURCE_TO_DIR.get(source_id, source_id)
        doc_dir = self._output_dir / dir_name

        if not doc_dir.exists():
            raise FileNotFoundError(
                f"No pre-computed olmocr output for '{source_id}' at {doc_dir}"
            )

        # pages.json: [{ "page": N, "content": "...", "processing_time_seconds": 2.1 }, ...]
        pages_path = doc_dir / "pages.json"
        with open(pages_path, encoding="utf-8") as f:
            pages_data = json.load(f)

        # extraction_summary.json: supports two key formats
        summary_path = doc_dir / "extraction_summary.json"
        total_time_ms = 0.0
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            if "total_processing_time_ms" in summary:
                total_time_ms = summary["total_processing_time_ms"]
            else:
                # original run_olmocr.py writes seconds, convert to ms
                total_time_ms = summary.get("total_time_seconds", 0) * 1000

        metadata.total_pages = len(pages_data)

        pages: list[PageResult] = []
        all_tables: list[TableData] = []
        text_parts: list[str] = []

        for page_entry in pages_data:
            page_num = page_entry["page"]
            content = page_entry.get("content", "")
            page_time_ms = page_entry.get("processing_time_seconds", 0) * 1000

            tables = self._extract_tables_from_content(content, page_num)
            all_tables.extend(tables)

            # strip the HTML tables and replace with [TABLE] placeholders so that the
            # chunker can set has_table on the affected chunks
            plain_text = self._strip_html_tables(content)

            pages.append(
                PageResult(
                    page_number=page_num,
                    raw_text=plain_text,
                    blocks=[],  # olmocr outputs a continuous markdown stream, no position blocks
                    tables=tables,
                    processing_time_ms=page_time_ms,
                )
            )
            text_parts.append(plain_text)

        return ExtractionResult(
            extractor_name=self.name,
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(text_parts),
            tables=all_tables,
            toc_entries=[],  # olmocr sees only rendered images
            total_processing_time_ms=total_time_ms,
        )

    def extract_page(self, file_path: Path, page_number: int) -> PageResult:
        """
        this loads and returns the extraction result for a single page

        Raises:
        ValueError: if source_id cannot be determined or page doesn't exist
        """
        stem = file_path.stem
        source_id = None
        for sid, dname in _SOURCE_TO_DIR.items():
            if dname in stem or stem in dname:
                source_id = sid
                break

        if not source_id:
            raise ValueError(f"Cannot determine source_id from {file_path}")

        doc_dir = self._output_dir / _SOURCE_TO_DIR[source_id]
        pages_path = doc_dir / "pages.json"
        with open(pages_path, encoding="utf-8") as f:
            pages_data = json.load(f)

        for page_entry in pages_data:
            if page_entry["page"] == page_number:
                content = page_entry.get("content", "")
                tables = self._extract_tables_from_content(content, page_number)
                plain_text = self._strip_html_tables(content)

                return PageResult(
                    page_number=page_number,
                    raw_text=plain_text,
                    blocks=[],
                    tables=tables,
                    processing_time_ms=page_entry.get("processing_time_seconds", 0)
                    * 1000,
                )

        raise ValueError(
            f"Page {page_number} not found in olmocr output for {source_id}"
        )

    # Private helpers for HTML table parsing
    #
    # olmocr embeds tables as HTML within markdown
    #

    def _extract_tables_from_content(
        self, content: str, page_number: int
    ) -> list[TableData]:
        """
        this parses HTML <table> blocks from olmocr markdown output into TableData objects

        Args:
        content:      the raw markdown/HTML content for one page
        page_number:  passed through to TableData for provenance

        Returns:
        a list of TableData objects, one per <table> block, it is empty if none found
        """
        tables: list[TableData] = []

        # re.DOTALL so '.' matches newlines
        for match in re.finditer(r"<table>(.*?)</table>", content, re.DOTALL):
            table_html = match.group(1)
            rows = self._parse_html_table(table_html)

            if rows:
                tables.append(
                    TableData(
                        rows=rows,
                        row_count=len(rows),
                        col_count=max(len(r) for r in rows) if rows else 0,
                        page_number=page_number,
                        # bbox is None because olmocr doesn't provide position
                    )
                )
        return tables

    def _parse_html_table(self, table_html: str) -> list[list[str]]:
        """
        this parses rows from HTML table content (tr/th/td tags)

        Args:
        table_html: the inner HTML of a <table> element

        Returns:
        a 2D list of cell strings, e.g., [["Name", "Age"], ["Alice", "30"]]
        """
        rows: list[list[str]] = []

        for row_match in re.finditer(r"<tr>(.*?)</tr>", table_html, re.DOTALL):
            row_html = row_match.group(1)
            # <t[hd]> matches both <th> (header) and <td> (data) cells.
            cells = re.findall(r"<t[hd]>(.*?)</t[hd]>", row_html, re.DOTALL)
            cells = [c.strip() for c in cells]
            if cells:
                rows.append(cells)
        return rows

    def _strip_html_tables(self, content: str) -> str:
        """
        this replaces each <table>...</table> block with the "[TABLE]" placeholder

        the chunker uses "[TABLE]" to set the "has_table" flag on chunks

        Example:
        Input:   "Some text. <table><tr><td>A</td></tr></table> More text."
        Output:  "Some text. [TABLE] More text."
        """
        return re.sub(r"<table>.*?</table>", "[TABLE]", content, flags=re.DOTALL)
