# chunker.py: This is for the legal structure aware chunking for the regulatory documents
#
# It sits between the extraction and the embedding in the pipeline.
#
# There are 2 strategies here:
# - for statutes (GDPR, FADP): to split at article boundaries and "subsplit"
#   long articles at paragraphs, as well as handle GDPR recitals separately
# - and for guidance docs (EDPB, FDPIC): to split by PDF table of contents (if available),
#   else by the numbered headings, or fall back to the size based splitting
#
# Chunks that exceed the "max_tokens" are always subsplit with overlapping windows.
#

"""Legal structure aware chunking for the regulatory documents"""
# import libraries
from __future__ import annotations
import re
from dataclasses import dataclass

# a Chunk: text + metadata (source_id, article, jurisdiction, cross-refs, and so forth),
# the chunk_id is a SHA-256 hash of source_id + article + chunk_index
#
# ExtractionResult: this is the full output of one extractor run (i.e., text, TOC, metadata)
from app.ingestion.models import Chunk, ExtractionResult


# regex patterns for legal structure
# this is compiled once at the module load

# Article boundaries
# GDPR: "Article 1", "Article 44"
RE_GDPR_ARTICLE = re.compile(r"^(Article\s+\d+)\b", re.MULTILINE)
# FADP: "Art. 1", "Art. 25a"
RE_FADP_ARTICLE = re.compile(r"^(Art\.\s+\d+[a-z]?)\b", re.MULTILINE)
# the generic fallback that matches both formats
RE_ARTICLE_GENERIC = re.compile(r"^((?:Article|Art\.)\s+\d+[a-z]?)\b", re.MULTILINE)

# for chapter / section headers
RE_CHAPTER = re.compile(r"^(CHAPTER\s+[IVXLCDM]+)", re.MULTILINE)
RE_SECTION = re.compile(r"^(Section\s+\d+)", re.MULTILINE)

# numbered headings in the guidance docs
# this matches "1.2 Scope", "3.1.4 Consent" etc.
RE_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z])", re.MULTILINE)

# for GDPR recitals
RE_RECITAL = re.compile(r"^\((\d+)\)\s", re.MULTILINE)

# for cross references
# this detects in text article references, e.g., "Article 44", "Art. 6(1)",
# "Article 9(2) point (a)"
RE_CROSS_REF = re.compile(
    r"(?:Article|Art\.)\s+\d+[a-z]?"
    r"(?:\s*\(\d+\))?"
    r"(?:\s*(?:point|letter)\s*\([a-z]\))?"
)

# markers for footnotes
RE_FOOTNOTE_MARKER = re.compile(r"(?<=[a-z.,;])\d{1,3}(?=\s)")


# Token estimation


def estimate_tokens(text: str) -> int:
    """This is for the approx. token estimate: word count * 1.3

    Args:
    text: for the text to estimate the tokens for

    Returns:
    est. number of tokens (an int)
    """
    return int(len(text.split()) * 1.3)


# class _ArticleSpan


@dataclass
class _ArticleSpan:
    """for a detected article or section with its text span

    It covers text from one regex match to the beginning of the next one (or the end of doc)

    It is private
    """

    label: str  # e.g., "Article 44" or "3.2"
    start: int  # the character offset in the full text of the document
    end: int  # the character offset where the next span begins
    text: str  # actual text content of this span


# LegalChunker, the main class


class LegalChunker:
    """This splits the results from the extraction into legal structure aware chunks

    Usage:
    chunker = LegalChunker(max_tokens=512, overlap_tokens=64)
    chunks = chunker.chunk(extraction_result)

    chunks is a list[Chunk], and is then ready for embedding and the Weaviate storage
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64):
        """This initializes the chunker with the token limits

        Args:
        max_tokens: The ceiling per each chunk (default is 512). Most articles
        will fit into this and longer ones are subsplit.
        overlap_tokens: This is the overlap between consecutive chunks
        (default is 64)
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, result: ExtractionResult) -> list[Chunk]:
        """This produces chunks from an extraction result

        It routes to either article based chunking for statutes (GDPR, FADP) or to
        section based chunking for the guidance documents

        Args:
        result: ExtractionResult from any extractor (PyMuPDF, etc.)

        Returns:
        list of Chunk objects, each one with text and metadata
        """
        if result.metadata.instrument_type == "statute":
            return self._chunk_statute(result)
        else:
            return self._chunk_guidance(result)

    # Statute chunking for GDPR, FADP
    #
    # The algorithm works as follows:
    # 1. Detects article boundaries by using source specific regex
    # 2. Handles preamble (GDPR: it is split into individual recitals)
    # 3. For each article: one chunk if it's within max_tokens, else split
    # at the paragraph boundaries (1., 2., and so forth), else a size based
    # fallback.
    #

    def _chunk_statute(self, result: ExtractionResult) -> list[Chunk]:
        """This chunks a statute (GDPR or FADP) by the article boundaries

        Args:
        result: ExtractionResult for a statute document

        Returns:
        a list of Chunk objects, with one per article (or a paragraph for long ones)
        """
        text = result.full_text
        source_id = result.metadata.source_id

        # chooses the correct article detection regex for the source
        if source_id == "gdpr":
            pattern = RE_GDPR_ARTICLE
        elif source_id == "fadp":
            pattern = RE_FADP_ARTICLE
        else:
            # tries both formats for unknown statutes
            pattern = RE_ARTICLE_GENERIC

        spans = self._find_spans(text, pattern)

        if not spans:
            # no articles detected, go back to size based splitting
            return self._split_by_size(
                text, source_id, result, article=None, chapter=None
            )

        chunks: list[Chunk] = []
        chunk_idx = 0  # counter used in chunk_id hash

        # handle the preamble
        if spans[0].start > 0:
            preamble = text[: spans[0].start].strip()
            if preamble and estimate_tokens(preamble) > 20:
                if source_id == "gdpr":
                    # GDPR has many numbered recitals and each one gets its own chunk
                    recital_chunks = self._chunk_recitals(preamble, result)
                    for c in recital_chunks:
                        c.chunk_index = chunk_idx
                        c.chunk_id = Chunk.make_id(source_id, c.paragraph, chunk_idx)
                        chunk_idx += 1
                    chunks.extend(recital_chunks)
                else:
                    chunks.append(
                        self._make_chunk(
                            preamble,
                            result,
                            chunk_idx,
                            article=None,
                            paragraph="Preamble",
                        )
                    )
                    chunk_idx += 1

        # process each article
        # this tracks the chapter or section context for chunk metadata (to enable
        # filtered retrieval, e.g., "CHAPTER V" = international transfers)
        current_chapter = None
        current_section = None

        for span in spans:
            # in order to check the 500 chars that come before this article for chapter or section headers
            preceding = text[max(0, span.start - 500) : span.start]
            ch_match = list(RE_CHAPTER.finditer(preceding))
            if ch_match:
                current_chapter = ch_match[-1].group(1)  # most recent match
            sec_match = list(RE_SECTION.finditer(preceding))
            if sec_match:
                current_section = sec_match[-1].group(1)

            article_text = span.text.strip()
            if not article_text:
                continue

            if estimate_tokens(article_text) <= self.max_tokens:
                chunks.append(
                    self._make_chunk(
                        article_text,
                        result,
                        chunk_idx,
                        article=span.label,
                        chapter=current_chapter,
                        section=current_section,
                    )
                )
                chunk_idx += 1
            else:
                # the article is too long, so split it at paragraph boundaries
                sub_chunks = self._split_article_by_paragraphs(
                    article_text,
                    result,
                    span.label,
                    current_chapter,
                    current_section,
                )
                for c in sub_chunks:
                    c.chunk_index = chunk_idx
                    c.chunk_id = Chunk.make_id(source_id, span.label, chunk_idx)
                    chunk_idx += 1
                chunks.extend(sub_chunks)

        return chunks

    def _chunk_recitals(self, preamble: str, result: ExtractionResult) -> list[Chunk]:
        """This function splits the GDPR preamble into individual recitals

        Args:
        preamble: this is the text before the first article in GDPR
        result: ExtractionResult (which is used for metadata)

        Returns:
        1 Chunk per recital, or a single preamble chunk if no markers are found
        """
        chunks: list[Chunk] = []
        matches = list(RE_RECITAL.finditer(preamble))

        if not matches:
            return [self._make_chunk(preamble, result, 0, paragraph="Preamble")]

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(preamble)
            recital_text = preamble[start:end].strip()
            recital_num = match.group(1)
            if recital_text:
                chunks.append(
                    self._make_chunk(
                        recital_text,
                        result,
                        0,
                        paragraph=f"Recital {recital_num}",
                    )
                )
        return chunks

    def _split_article_by_paragraphs(
        self,
        article_text: str,
        result: ExtractionResult,
        article_label: str,
        chapter: str | None,
        section: str | None,
    ) -> list[Chunk]:
        """This splits a long article at the numbered paragraph boundaries (1., 2., etc.)

        It falls back to the size based splitting if no paragraph markers are found,
        or if a single paragraph still exceeds the "max_tokens"

        Args:
        article_text: full text of the article to split
        result: ExtractionResult (for the metadata)
        article_label: article identifier, e.g., "Article 6"
        chapter: the Current chapter context, e.g., "CHAPTER II"
        section: the Current section context, e.g., "Section 1"

        Returns:
        list of Chunk objects, 1 per paragraph (or subsplit)
        """
        para_pattern = re.compile(r"^(\d+)\.\s", re.MULTILINE)
        matches = list(para_pattern.finditer(article_text))

        if not matches:
            return self._split_by_size(
                article_text,
                result.metadata.source_id,
                result,
                article=article_label,
                chapter=chapter,
            )

        chunks: list[Chunk] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(article_text)
            para_text = article_text[start:end].strip()
            para_num = match.group(1)

            if estimate_tokens(para_text) <= self.max_tokens:
                chunks.append(
                    self._make_chunk(
                        para_text,
                        result,
                        0,
                        article=article_label,
                        paragraph=para_num,
                        chapter=chapter,
                        section=section,
                    )
                )
            else:
                # the single paragraph is still too long, therefore a size based fall back
                sub = self._split_by_size(
                    para_text,
                    result.metadata.source_id,
                    result,
                    article=article_label,
                    chapter=chapter,
                )
                for c in sub:
                    c.paragraph = para_num
                chunks.extend(sub)

        return chunks

    # Chunking for guidance documents (EDPB guidelines, FDPIC guide)
    #
    # Approach:
    # 1. PDF TOC (which is most reliable)
    # 2. Numbered headings with regex ("1.2 Scope", "3.1.4 Consent")
    # 3. size based splitting (last option)
    #

    def _chunk_guidance(self, result: ExtractionResult) -> list[Chunk]:
        """This chunks a guidance document by the section boundaries

        Args:
        result: ExtractionResult

        Returns:
        a list of Chunk objects, 1 per section (or a subsplit for long ones)
        """
        text = result.full_text

        if result.toc_entries:
            return self._chunk_by_toc(result)

        spans = self._find_spans(text, RE_NUMBERED_HEADING)

        if not spans:
            return self._split_by_size(
                text,
                result.metadata.source_id,
                result,
                article=None,
                chapter=None,
            )

        chunks: list[Chunk] = []
        chunk_idx = 0

        # handles the text before the first heading (e.g., cover page, abstract, etc.)
        if spans[0].start > 0:
            preamble = text[: spans[0].start].strip()
            if preamble and estimate_tokens(preamble) > 20:
                chunks.append(
                    self._make_chunk(
                        preamble,
                        result,
                        chunk_idx,
                        section="Preamble",
                    )
                )
                chunk_idx += 1

        for span in spans:
            section_text = span.text.strip()
            if not section_text:
                continue

            if estimate_tokens(section_text) <= self.max_tokens:
                chunks.append(
                    self._make_chunk(
                        section_text,
                        result,
                        chunk_idx,
                        section=span.label,
                    )
                )
                chunk_idx += 1
            else:
                sub = self._split_by_size(
                    section_text,
                    result.metadata.source_id,
                    result,
                    article=None,
                    chapter=None,
                )
                for c in sub:
                    c.section = span.label
                    c.chunk_index = chunk_idx
                    c.chunk_id = Chunk.make_id(
                        result.metadata.source_id, span.label, chunk_idx
                    )
                    chunk_idx += 1
                chunks.extend(sub)

        return chunks

    def _chunk_by_toc(self, result: ExtractionResult) -> list[Chunk]:
        """This splits the guidance document using its table of contents (TOC)

        It locates each TOC title in the extracted text and uses those positions
        for section boundaries. It falls back to size based splitting if no TOC
        titles can be found within the text

        Args:
        result: ExtractionResult with non empty "toc_entries"

        Returns:
        a list of Chunk objects, 1 per TOC section
        """
        text = result.full_text
        toc = result.toc_entries

        # finds each TOC title's char offset within the text. Skips titles that
        # are not found
        boundaries: list[tuple[str, int]] = []
        for entry in toc:
            idx = text.find(entry.title)
            if idx >= 0:
                boundaries.append((entry.title, idx))

        if not boundaries:
            return self._split_by_size(
                text,
                result.metadata.source_id,
                result,
                article=None,
                chapter=None,
            )

        # sorts by position
        boundaries.sort(key=lambda x: x[1])

        chunks: list[Chunk] = []
        chunk_idx = 0

        for i, (title, start) in enumerate(boundaries):
            end = boundaries[i + 1][1] if i + 1 < len(boundaries) else len(text)
            section_text = text[start:end].strip()

            # skips very short sections
            if not section_text or estimate_tokens(section_text) < 20:
                continue

            if estimate_tokens(section_text) <= self.max_tokens:
                chunks.append(
                    self._make_chunk(
                        section_text,
                        result,
                        chunk_idx,
                        section=title,
                    )
                )
                chunk_idx += 1
            else:
                sub = self._split_by_size(
                    section_text,
                    result.metadata.source_id,
                    result,
                    article=None,
                    chapter=None,
                )
                for c in sub:
                    c.section = title
                    c.chunk_index = chunk_idx
                    c.chunk_id = Chunk.make_id(
                        result.metadata.source_id, title, chunk_idx
                    )
                    chunk_idx += 1
                chunks.extend(sub)

        return chunks

    # Utility functions

    def _find_spans(self, text: str, pattern: re.Pattern) -> list[_ArticleSpan]:
        """This finds all the spans which are defined by a boundary pattern

        Every span covers text from one regex match to the beginning of the next

        Args:
        text: full document text to scan
        pattern: a compiled regex whose first group is the label

        Returns:
        list of "_ArticleSpan" objects (which is empty if no matches are found)
        """
        matches = list(pattern.finditer(text))
        if not matches:
            return []

        spans: list[_ArticleSpan] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            spans.append(
                _ArticleSpan(
                    label=match.group(1),
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
        return spans

    def _split_by_size(
        self,
        text: str,
        source_id: str,
        result: ExtractionResult,
        article: str | None,
        chapter: str | None,
    ) -> list[Chunk]:
        """fall back scenario: this splits the text into overlapping windows by token limit

        It uses a sliding window, i.e., "max_words" words per chunk, and advances by
        (max_words - overlap_words) each step so the chunks share
        overlap_words of context at boundary

        the token -> word conversion uses 1.3x factor

        Args:
        text: this is the text to split
        source_id: document identifier (e.g., "gdpr")
        result: ExtractionResult (for the metadata)
        article: article label to attach to the chunks
        chapter: chapter label to attach to the chunks

        Returns:
        list of Chunk objects
        """
        words = text.split()
        max_words = int(self.max_tokens / 1.3)
        overlap_words = int(self.overlap_tokens / 1.3)

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(words):
            end = min(start + max_words, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                self._make_chunk(
                    chunk_text,
                    result,
                    idx,
                    article=article,
                    chapter=chapter,
                )
            )
            idx += 1
            start = end - overlap_words if end < len(words) else end

        return chunks

    def _make_chunk(
        self,
        text: str,
        result: ExtractionResult,
        chunk_index: int,
        article: str | None = None,
        paragraph: str | None = None,
        chapter: str | None = None,
        section: str | None = None,
    ) -> Chunk:
        """This creates a Chunk with cross reference detection as well as content flags

        Steps:
        1. Cross reference detection, it scans for article mentions (e.g.,
           "Article 44") and then stores them as metadata for the Neo4j graph edges
        2. Content flags, this detects tables (with pipe chars or with "table" in first
           100 chars) and footnote markers for evaluation

        chunk_id is a 16-char hex hash of "source_id:article:chunk_index"

        Args:
        text: the text content of the chunk
        result: ExtractionResult that this chunk comes from (for the metadata)
        chunk_index: sequential index of this chunk within the doc
        article: the article label, e.g., "Article 44" (None for guidance docs)
        paragraph: paragraph/recital label, e.g., "1" or "Recital 47"
        chapter: the chapter label, e.g., "CHAPTER V" (None if not known)
        section: the section label, e.g., "3.2" (None for statutes)

        Returns:
        a fully populated Chunk dataclass instance
        """
        cross_refs = list(set(RE_CROSS_REF.findall(text)))
        if article:
            cross_refs = [r for r in cross_refs if r.strip() != article.strip()]

        has_table = bool(re.search(r"\|.*\|", text) or "table" in text.lower()[:100])
        has_footnote = bool(RE_FOOTNOTE_MARKER.search(text))

        source_id = result.metadata.source_id

        return Chunk(
            chunk_id=Chunk.make_id(source_id, article or section, chunk_index),
            text=text,
            source_id=source_id,
            instrument_type=result.metadata.instrument_type,
            jurisdiction=result.metadata.jurisdiction,
            chapter=chapter,
            section=section,
            article=article,
            paragraph=paragraph,
            page_numbers=[],  # this is populated if page mapping is available
            cross_references=cross_refs,
            has_table=has_table,
            has_footnote=has_footnote,
            chunk_index=chunk_index,
            extractor_name=result.extractor_name,
        )
