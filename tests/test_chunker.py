"""These are unit tests for the chunker

This tests the script "app/ingestion/chunker.py", which splits the extracted regulatory PDF
text into semantically meaningful chunks for the embedding in the Weaviate vector database

What the tests cover:
1. Token estimation
2. the regex patterns for the article boundaries and the cross-references
3. the chunking of statutes (GDPR, FADP), i.e., article splitting, preamble handling,
   large article splitting, crossreference detection, self reference exclusion,
   and completeness of metadata
4. the chunking for the guidance documents, i.e., section splitting and size based
fall back
5. the chunk ID determinism
"""

# import libraries
from __future__ import annotations
import pytest
from app.ingestion.chunker import (
    LegalChunker,
    estimate_tokens,
    RE_GDPR_ARTICLE,
    RE_FADP_ARTICLE,
    RE_CROSS_REF,
    RE_NUMBERED_HEADING,
)
from app.ingestion.models import Chunk, DocumentMetadata, ExtractionResult, PageResult


# fixture and helper


@pytest.fixture
def chunker():
    """This is the LegalChunker with settings that match production (max. 512 tokens, 64 overlap)"""
    return LegalChunker(max_tokens=512, overlap_tokens=64)


def _make_extraction(
    source_id: str, instrument_type: str, full_text: str
) -> ExtractionResult:
    """To build a minimal ExtractionResult for testing without the real PDF files

    Args:
    source_id       : "gdpr" -> EU/GDPR article patterns, and "fadp" for Swiss patterns
    instrument_type : "statute" or "guidance", this determines the chunking strategy
    full_text       :  raw text content that is to chunk

    Returns:
    ExtractionResult with placeholder values for all the fields not used by the chunker
    """
    return ExtractionResult(
        extractor_name="test",
        metadata=DocumentMetadata(
            source_id=source_id,
            title="Test Document",
            instrument_type=instrument_type,
            jurisdiction="EU" if source_id == "gdpr" else "CH",
            effective_date=None,
            file_path="test.pdf",
            total_pages=1,
        ),
        pages=[PageResult(page_number=1, raw_text=full_text)],
        full_text=full_text,
        tables=[],
        toc_entries=[],  # this forces heading based splitting
        total_processing_time_ms=0.0,
    )


# Token estimation


class TestTokenEstimation:
    """This tests for "estimate_tokens()", it approximates the token count as "word_count" times 1.3"""

    def test_empty_string(self):
        """Empty string is 0 tokens"""
        assert estimate_tokens("") == 0

    def test_single_word(self):
        """single word: 1 token (int(1 x 1.3) = 1)"""
        assert estimate_tokens("hello") == 1

    def test_approximation(self):
        """a 10 word sentence approx. 13 tokens, checks the range 10 - 15 to allow for a minor multiplier drift"""
        text = "This is a test sentence with ten words in total."
        tokens = estimate_tokens(text)
        assert 10 <= tokens <= 15


# article regex patterns


class TestArticlePatterns:
    """This tests for the regex patterns that detect the legal structure"""

    def test_gdpr_article_pattern(self):
        """ "RE_GDPR_ARTICLE" matches "Article N" at the beginning of any line (re.MULTILINE)"""
        matches = RE_GDPR_ARTICLE.findall("Article 17\nThe data subject...")
        assert "Article 17" in matches

    def test_fadp_article_pattern(self):
        """ "RE_FADP_ARTICLE" matches the Swiss abbreviated format "Art. N" """
        matches = RE_FADP_ARTICLE.findall("Art. 25\nAny person may request...")
        assert "Art. 25" in matches

    def test_fadp_article_with_letter(self):
        """ "RE_FADP_ARTICLE" matches bis-articles with letter suffixes (e.g., "Art. 5a")"""
        matches = RE_FADP_ARTICLE.findall("Art. 5a\nThis article...")
        assert "Art. 5a" in matches

    def test_cross_reference_regex(self):
        """ "RE_CROSS_REF" detects the references to other articles in the text body

        It handles the formats: "Article 6", "Article 6(1)", "Article 6(1) point (a)", "Art. 9"
        """
        text = "as provided in Article 6(1) point (a)"
        matches = RE_CROSS_REF.findall(text)
        assert any("Article 6" in m for m in matches)

    def test_numbered_heading_pattern(self):
        """ "RE_NUMBERED_HEADING" matches section headings e.g., "1.2 Scope of application" """
        text = "1.2 Scope of application"
        matches = RE_NUMBERED_HEADING.findall(text)
        assert len(matches) >= 1


# For statute chunking (GDPR, FADP)
# splits at article boundaries, handles preamble, splits large articles at
# paragraph boundaries, detects cross references, excludes self references


class TestStatuteChunking:
    """This tests for splitting statutes (GDPR, FADP) at the article boundaries"""

    def test_splits_gdpr_articles(self, chunker):
        """Every article in the input produces a separate chunk with the correct label"""
        text = (
            "Preamble text here.\n\n"
            "Article 1\nSubject-matter and objectives.\n\n"
            "Article 2\nMaterial scope.\n\n"
            "Article 3\nTerritorial scope.\n"
        )
        result = _make_extraction("gdpr", "statute", text)
        chunks = chunker.chunk(result)

        article_labels = [c.article for c in chunks if c.article]
        assert "Article 1" in article_labels
        assert "Article 2" in article_labels
        assert "Article 3" in article_labels

    def test_fadp_uses_art_dot_format(self, chunker):
        """The FADP "Art. N" format is detected and labels are preserved with the dot prefix"""
        text = "Art. 1\nPurpose.\n\n" "Art. 2\nScope.\n\n"
        result = _make_extraction("fadp", "statute", text)
        chunks = chunker.chunk(result)

        article_labels = [c.article for c in chunks if c.article]
        assert "Art. 1" in article_labels
        assert "Art. 2" in article_labels

    def test_preamble_chunked_separately(self, chunker):
        """The text before the first article becomes its own chunk(s) (article=None)

        It uses a long preamble (approx. 50 words) to exceed the 20token minimum threshold
        below which the preambles are discarded.
        """
        preamble = (
            "This regulation lays down rules relating to the protection of natural "
            "persons with regard to the processing of personal data and rules "
            "relating to the free movement of such data. It protects fundamental "
            "rights and freedoms of natural persons and in particular their right "
            "to the protection of personal data.\n\n"
        )
        text = preamble + "Article 1\nSubject-matter.\n"
        result = _make_extraction("gdpr", "statute", text)
        chunks = chunker.chunk(result)

        preamble_chunks = [c for c in chunks if c.article is None]
        assert len(preamble_chunks) >= 1

    def test_large_article_split_by_paragraphs(self, chunker):
        """articles exceeding "max_tokens" are split at the numbered paragraph boundaries

        It creates approx. 1,625 tokens (5 paragraphs x ~50 words x 1.3) to force splitting
        """
        paras = "\n".join(f"{i}. Paragraph {i} text. " * 50 for i in range(1, 6))
        text = f"Article 1\n{paras}\n"
        result = _make_extraction("gdpr", "statute", text)
        chunks = chunker.chunk(result)

        art1_chunks = [c for c in chunks if c.article == "Article 1"]
        assert len(art1_chunks) > 1

    def test_cross_references_detected(self, chunker):
        """Mentions of other articles populate the cross_references list of the chunk"""
        text = "Article 1\nAs provided in Article 6(1) and Art. 9.\n"
        result = _make_extraction("gdpr", "statute", text)
        chunks = chunker.chunk(result)

        article_chunk = [c for c in chunks if c.article == "Article 1"][0]
        assert len(article_chunk.cross_references) > 0

    def test_self_reference_excluded(self, chunker):
        """the label of an article must not appear in its cross_references list"""
        text = "Article 6\nThis Article 6 applies to processing.\n"
        result = _make_extraction("gdpr", "statute", text)
        chunks = chunker.chunk(result)

        art_chunk = [c for c in chunks if c.article == "Article 6"][0]
        self_refs = [r for r in art_chunk.cross_references if "Article 6" in r]
        assert self_refs == []

    def test_chunk_metadata_complete(self, chunker):
        """each chunk must have a source_id, instrument_type, jurisdiction, chunk_id, and text"""
        text = "Article 1\nSubject-matter.\n"
        result = _make_extraction("gdpr", "statute", text)
        chunks = chunker.chunk(result)

        for chunk in chunks:
            assert chunk.source_id == "gdpr"
            assert chunk.instrument_type == "statute"
            assert chunk.jurisdiction == "EU"
            assert chunk.chunk_id
            assert chunk.text


# Chunking for guidance docs
# this splits at the numbered section headings (e.g., "1.2 Scope")
# and falls back to size based splitting if no headings are found
#


class TestGuidanceChunking:
    """This tests for splitting guidance documents at the section boundaries"""

    def test_splits_numbered_sections(self, chunker):
        """the numbered section headings become the chunk boundaries and labels are stored in chunk.section"""
        text = (
            "1 Introduction\nThis guideline covers consent.\n\n"
            "2 Scope\nThe scope of these guidelines.\n\n"
            "3 Analysis\nDetailed analysis.\n"
        )
        result = _make_extraction("edpb_consent", "guidance", text)
        chunks = chunker.chunk(result)

        sections = [c.section for c in chunks if c.section]
        assert len(sections) >= 2

    def test_fallback_to_size_split(self, chunker):
        """prose with no recognizable headings is split by token limit with an overlap

        Each chunk is checked to be within "max_tokens" + 50 (the tolerance for the word boundary rounding)
        """
        text = "This is a long block of guidance text. " * 200
        result = _make_extraction("edpb_consent", "guidance", text)
        chunks = chunker.chunk(result)

        assert len(chunks) > 1
        for chunk in chunks:
            assert estimate_tokens(chunk.text) <= chunker.max_tokens + 50


# Determinism for Chunk IDs
# the IDs are SHA-256("{source_id}:{article_or_section}:{chunk_index}")[:16]


class TestChunkIds:
    """This tests for deterministic chunk ID generation"""

    def test_make_id_deterministic(self):
        """same inputs always produce the same 16-char hex hash"""
        id1 = Chunk.make_id("gdpr", "Article 17", 0)
        id2 = Chunk.make_id("gdpr", "Article 17", 0)
        assert id1 == id2

    def test_make_id_different_for_different_inputs(self):
        """different chunk_index or source_id produces different IDs"""
        id1 = Chunk.make_id("gdpr", "Article 17", 0)
        id2 = Chunk.make_id("gdpr", "Article 17", 1)  # different chunk_index
        id3 = Chunk.make_id("fadp", "Article 17", 0)  # different source_id
        assert id1 != id2
        assert id1 != id3
