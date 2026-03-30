"""
This is the Mistral Document AI extractor (via the Mistral API)

All 3 extractors implement the same BaseExtractor interface (base_extractor.py)

How does it work:
- the PDF is base64-encoded and sent to Mistral's cloud OCR API
- the API returns content per page as markdown
- Markdown/HTML tables are parsed into TableData objects and then replaced with [TABLE] markers

Requires:
- the MISTRAL_API_KEY environment variable
- mistralai Python package (pip install mistralai)
"""

# import libraries
from __future__ import annotations
import base64
import os
import re
import time
from pathlib import Path
from app.ingestion.base_extractor import BaseExtractor
from app.ingestion.models import (
    DocumentMetadata,
    ExtractionResult,
    PageResult,
    TableData,
    TocEntry,
)


class MistralDocumentAIExtractor(BaseExtractor):
    """this is for the Mistral Document AI API service

    Requires: the MISTRAL_API_KEY environment variable
    Uses: client.ocr.process() with model "mistral-ocr-latest"
    """

    def __init__(self):
        # populated on first use via "_get_client()"
        self._client = None

    @property
    def name(self) -> str:
        """this is a unique extractor identifier used to label outputs and saved results"""
        return "mistral"

    def is_available(self) -> bool:
        """this returns true if the MISTRAL_API_KEY is set and not empty"""
        return bool(os.environ.get("MISTRAL_API_KEY"))

    def _get_client(self):
        """
        to initialize and return the Mistral API client

        the mistralai package is only imported on first use, it's not
        required unless this extractor is actually called

        Raises:
        KeyError: if the  MISTRAL_API_KEY is not set
        """
        if self._client is None:
            from mistralai.client import Mistral

            self._client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        return self._client

    def extract(self, file_path: Path, metadata: DocumentMetadata) -> ExtractionResult:
        """
        this extracts structured content from an entire PDF using Mistral's API

        it reads the PDF, base64 encodes it, sends it to Mistral's OCR API, and then
        parses the markdown per page response into PageResult objects, and extracts
        tables

        Args:
        file_path: the Path to the PDF file
        metadata: the document identity info, "total_pages" is updated after the API call

        Returns:
        ExtractionResult with all pages, tables, full text, and timing info

        Raises:
        RuntimeError: if the MISTRAL_API_KEY is not set
        """
        if not self.is_available():
            raise RuntimeError("MISTRAL_API_KEY not set")

        client = self._get_client()

        # Mistral's API expects the PDF as a base64 encoded data URI in JSON
        with open(file_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

        start = time.perf_counter()

        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
            include_image_base64=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        pages: list[PageResult] = []
        all_tables: list[TableData] = []
        text_parts: list[str] = []

        metadata.total_pages = len(response.pages)

        for resp_page in response.pages:
            page_num = resp_page.index + 1  # the API is 0-indexed, models use 1-indexed
            content = resp_page.markdown or ""

            tables = self._extract_tables_from_markdown(content, page_num)
            all_tables.extend(tables)
            plain_text = self._strip_markdown_tables(content)

            pages.append(
                PageResult(
                    page_number=page_num,
                    raw_text=plain_text,
                    blocks=[],  # Mistral does not provide positional blocks
                    tables=tables,
                    # the total API time distributed evenly, breakdown per page is not available
                    processing_time_ms=elapsed_ms / max(len(response.pages), 1),
                )
            )
            text_parts.append(plain_text)

        return ExtractionResult(
            extractor_name=self.name,
            metadata=metadata,
            pages=pages,
            full_text="\n\n".join(text_parts),
            tables=all_tables,
            toc_entries=[],  # Mistral's OCR API doesn't extract a PDF outline
            total_processing_time_ms=elapsed_ms,
        )

    def extract_page(self, file_path: Path, page_number: int) -> PageResult:
        """
        This extracts a single page from a PDF file

        The full PDF is still encoded and sent even for a single page.

        Args:
        file_path: the Path to the PDF file
        page_number: 1-based page number to extract

        Returns:
        a PageResult for the requested page

        Raises:
        RuntimeError: if the MISTRAL_API_KEY is not set
        ValueError:   if the API returns no content for the requested page
        """
        if not self.is_available():
            raise RuntimeError("MISTRAL_API_KEY is not set")

        client = self._get_client()

        with open(file_path, "rb") as f:
            pdf_b64 = base64.b64encode(f.read()).decode("utf-8")

        start = time.perf_counter()

        response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_b64}",
            },
            pages=[page_number - 1],  # API is 0-indexed
            include_image_base64=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not response.pages:
            raise ValueError(f"No content returned for page {page_number}")

        resp_page = response.pages[0]
        content = resp_page.markdown or ""

        tables = self._extract_tables_from_markdown(content, page_number)
        plain_text = self._strip_markdown_tables(content)

        return PageResult(
            page_number=page_number,
            raw_text=plain_text,
            blocks=[],  # no positional blocks from the Mistral API
            tables=tables,
            processing_time_ms=elapsed_ms,
        )

    # private helpers

    def _extract_tables_from_markdown(
        self, content: str, page_number: int
    ) -> list[TableData]:
        """
        This extracts tables from Mistral's markdown output

        It handles 2 formats produced by the API:
        1. pipe-delimited markdown tables
        2. HTML <table> elements

        Args:
        content:      the full markdown text for one page
        page_number:  1-based page number stored in each TableData

        Returns:
        a list of TableData objects, which is empty if no tables are found
        """
        tables: list[TableData] = []

        # pattern 1: pipe-delimited markdown tables
        table_pattern = re.compile(
            r"((?:^\|.+\|$\n?){2,})",
            re.MULTILINE,
        )

        for match in table_pattern.finditer(content):
            table_text = match.group(1).strip()
            rows = self._parse_markdown_table(table_text)
            if rows:
                tables.append(
                    TableData(
                        rows=rows,
                        row_count=len(rows),
                        col_count=max(len(r) for r in rows) if rows else 0,
                        page_number=page_number,
                    )
                )

        # pattern 2: HTML <table> blocks
        for match in re.finditer(r"<table>(.*?)</table>", content, re.DOTALL):
            rows = self._parse_html_table(match.group(1))
            if rows:
                tables.append(
                    TableData(
                        rows=rows,
                        row_count=len(rows),
                        col_count=max(len(r) for r in rows) if rows else 0,
                        page_number=page_number,
                    )
                )

        return tables

    def _parse_markdown_table(self, table_text: str) -> list[list[str]]:
        """
        This parses a pipe-delimited markdown table into a 2D list of cell values

        Returns: [["Name", "Age"], ["Alice", "30"]]
        """
        rows: list[list[str]] = []

        for line in table_text.strip().split("\n"):
            line = line.strip()

            if not line.startswith("|"):
                continue

            # skips separator rows
            if re.match(r"^\|[\s\-:|]+$", line):
                continue

            # splits by | and drops the empty strings at start/end which are caused by leading/trailing pipes
            cells = [c.strip() for c in line.split("|")[1:-1]]

            if cells:
                rows.append(cells)

        return rows

    def _parse_html_table(self, table_html: str) -> list[list[str]]:
        """
        this parses cell values from the inner HTML of a <table> element

        It handles both <th> (header) and <td> (data) cells

        Args:
        table_html: the inner HTML between <table> and </table> tags

        Returns:
        a list of rows, each a list of cell value strings
        """
        rows: list[list[str]] = []

        for row_match in re.finditer(r"<tr>(.*?)</tr>", table_html, re.DOTALL):
            # <t[hd]> matches both <th> and <td>
            cells = re.findall(r"<t[hd]>(.*?)</t[hd]>", row_match.group(1), re.DOTALL)
            cells = [c.strip() for c in cells]
            if cells:
                rows.append(cells)

        return rows

    def _strip_markdown_tables(self, content: str) -> str:
        """
        this replaces table blocks with [TABLE] markers in page text

        It removes pipe-delimited and HTML tables from the plain text so that downstream
        components (e.g., the chunker) aren't cluttered with raw table syntax
        a [TABLE] marker is left to indicate the table's position
        """
        # replaces pipe-delimited markdown tables
        result = re.sub(
            r"((?:^\|.+\|$\n?){2,})",
            "[TABLE]\n",
            content,
            flags=re.MULTILINE,
        )

        # replaces HTML tables
        result = re.sub(r"<table>.*?</table>", "[TABLE]", result, flags=re.DOTALL)

        return result
