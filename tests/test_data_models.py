"""These are unit tests for the data models: SearchResult, GraphQueryResult, and related graph / ingestion models

This covers 3 layers of the pipeline:
- Retrieval Layer (SearchResult): authority_label, location_label display properties
- Graph Layer (ArticleNode, DefinitionNode, ObligationNode, GraphQueryResult, GraphStats): node_id schemes, is_empty, string formatting
- Ingestion Layer (Chunk): make_id via SHA-256 hashing
"""

# import libraries
from __future__ import annotations
import pytest
from app.retrieval.models import SearchResult
from app.graph.models import (
    ArticleNode,
    DefinitionNode,
    ObligationNode,
    GraphQueryResult,
    GraphStats,
)
from app.ingestion.models import Chunk


# SearchResult


class TestSearchResult:
    """This tests for the computed display properties for SearchResult"""

    def test_authority_label_statute(self):
        """instrument_type="statute" -> authority_label "Statute" """
        r = SearchResult(
            chunk_id="a",
            text="t",
            source_id="gdpr",
            instrument_type="statute",
            jurisdiction="EU",
            article="Article 17",
            section=None,
            paragraph=None,
            cross_references=[],
            distance=0.1,
            score=0.9,
        )
        assert r.authority_label == "Statute"

    def test_authority_label_guidance(self):
        """instrument_type="guidance" -> authority_label "Regulator Guidance" """
        r = SearchResult(
            chunk_id="a",
            text="t",
            source_id="edpb",
            instrument_type="guidance",
            jurisdiction="EU",
            article=None,
            section="1.2",
            paragraph=None,
            cross_references=[],
            distance=0.2,
            score=0.8,
        )
        assert r.authority_label == "Regulator Guidance"

    def test_location_label_full(self):
        """all three levels of location are present -> breadcrumb joined with " > " """
        r = SearchResult(
            chunk_id="a",
            text="t",
            source_id="gdpr",
            instrument_type="statute",
            jurisdiction="EU",
            article="Article 17",
            section="Section 4",
            paragraph="3",
            cross_references=[],
            distance=0.1,
            score=0.9,
        )
        assert r.location_label == "Article 17 > Section 4 > 3"

    def test_location_label_none(self):
        """no location fields set -> "(no location)" placeholder"""
        r = SearchResult(
            chunk_id="a",
            text="t",
            source_id="test",
            instrument_type="statute",
            jurisdiction="EU",
            article=None,
            section=None,
            paragraph=None,
            cross_references=[],
            distance=0.1,
            score=0.9,
        )
        assert r.location_label == "(no location)"


# ArticleNode
# node_id format is: "source_id:article_label", which is used as lookup key
# throughout the pipeline


class TestArticleNode:
    """This tests for ArticleNode's node_id generation"""

    def test_node_id_gdpr(self):
        """GDPR article -> node_id "gdpr:Article 17" """
        node = ArticleNode(source_id="gdpr", article_label="Article 17")
        assert node.node_id == "gdpr:Article 17"

    def test_node_id_fadp(self):
        """FADP article -> node_id "fadp:Art. 25" (abbr. format preserved)

        The interpret node must generate "fadp:Art. N" and not "fadp:Article N"
        to match the keys stored during the graph construction
        """
        node = ArticleNode(source_id="fadp", article_label="Art. 25")
        assert node.node_id == "fadp:Art. 25"


# DefinitionNode
# node_id format is: "source_id:def:term_lowercased"
# the lowercase normalization makes sure lookups are consistent regardless of the source capitalization


class TestDefinitionNode:
    """This tests for DefinitionNode's node_id generation"""

    def test_node_id_lowercase(self):
        """the term is lowercased in node_id: "Personal Data" -> "gdpr:def:personal data" """
        node = DefinitionNode(
            term="Personal Data",
            definition_text="...",
            source_id="gdpr",
            article_label="Article 4",
        )
        assert node.node_id == "gdpr:def:personal data"


# ObligationNode
# node_id: sha256("{source_id}:{article_label}:{description[:50]}")[:16]


class TestObligationNode:
    """This tests for ObligationNode's hashed node_id generation"""

    def test_node_id_is_hash(self):
        """node_id is a 16-char hex string"""
        node = ObligationNode(
            description="Controller must erase data",
            obligation_type="obligation",
            source_id="gdpr",
            article_label="Article 17",
        )
        assert len(node.node_id) == 16
        assert node.node_id.isalnum()

    def test_different_descriptions_different_ids(self):
        """2 obligations from the same article with different descriptions get distinct IDs"""
        n1 = ObligationNode(
            description="A",
            obligation_type="obligation",
            source_id="gdpr",
            article_label="Article 17",
        )
        n2 = ObligationNode(
            description="B",
            obligation_type="obligation",
            source_id="gdpr",
            article_label="Article 17",
        )
        assert n1.node_id != n2.node_id


# GraphQueryResult
# "is_empty" drives whether pipeline includes the graph context for the LLM


class TestGraphQueryResult:
    """This tests for the "is_empty" property of GraphQueryResult"""

    def test_empty_when_no_nodes_or_rels(self):
        """default result is empty (no nodes, no relationships)"""
        r = GraphQueryResult()
        assert r.is_empty

    def test_not_empty_with_nodes(self):
        """result with at least one node is not empty"""
        r = GraphQueryResult(nodes=[{"id": "test"}])
        assert not r.is_empty

    def test_not_empty_with_rels(self):
        """result with at least one relationship is not empty"""
        r = GraphQueryResult(relationships=[{"type": "REF"}])
        assert not r.is_empty


# GraphStats


class TestGraphStats:
    """This tests for GraphStats' string formatting"""

    def test_str_format(self):
        """ "__str__" produces fixed width aligned output for the terminal display"""
        stats = GraphStats(instruments=6, articles=174, definitions=33)
        text = str(stats)
        assert "Instrument:  6" in text
        assert "Article:     174" in text
        assert "Definition:  33" in text


# Chunk
# "make_id" formula: sha256("{source_id}:{article or 'none'}:{chunk_index}")[:16]


class TestChunk:
    """This tests for Chunk.make_id() deterministic ID generation"""

    def test_make_id_with_none_article(self):
        """ "article=None" is handled through substituting "none", the result is a 16-char hex string"""
        id1 = Chunk.make_id("gdpr", None, 0)
        assert len(id1) == 16

    def test_make_id_uses_sha256(self):
        """ "make_id" matches an independently calculated SHA-256 hash of the key string"""
        import hashlib

        expected = hashlib.sha256("gdpr:Article 1:0".encode()).hexdigest()[:16]
        assert Chunk.make_id("gdpr", "Article 1", 0) == expected
