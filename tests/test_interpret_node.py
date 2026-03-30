"""these are unit tests for the interpret node: query parsing, article ref extraction, filters

The interpret node is the first analytical stage in the GraphLex AI pipeline:
It takes a natural language query and extracts structured metadata:

article_refs:          e.g., ["gdpr:Article 17", "fadp:Art. 25"]
source_filters:        e.g., ["gdpr", "fadp"]
jurisdiction_filters:  e.g., ["EU", "CH"]
search_query:          the original query text (passed through as is)

The extraction is regex based (not LLM)

Neo4j format requirements:
- GDPR: "gdpr:Article N"   (full "Article")
- FADP: "fadp:Art. N"      (abbreviated "Art.")

To run: python -m pytest tests/test_interpret_node.py -v
"""

# import libraries
from __future__ import annotations
import pytest
from app.orchestration.config import OrchestrationConfig
from app.orchestration.nodes import interpret_node


# fixtures


@pytest.fixture
def config():
    """this provide a default OrchestrationConfig, interpret_node doesn't use config
    values, but it accepts it for being consistent with other nodes"""
    return OrchestrationConfig()


# helper function


def _run(query: str, config: OrchestrationConfig) -> dict:
    """this runs interpret_node on query string and returns the result dict"""
    state = {"query_text": query, "stages_completed": []}
    return interpret_node(state, config)


# test class 1: Article reference extraction
#
# 3 regex patterns:
# pattern 1: "GDPR Article 17", "FADP Art. 6" : source first
# pattern 2: "Article 17 of the GDPR"         : number first, with "of"
# pattern 3: "Article 17 GDPR"                : number first, no "of"
#
# Output format:
# GDPR -> "gdpr:Article N", FADP -> "fadp:Art. N"


class TestArticleRefExtraction:
    """this verifies regex patterns extract article references in all the supported formats"""

    def test_gdpr_article_n(self, config):
        """pattern 1: "GDPR Article 17" -> "gdpr:Article 17" """
        result = _run("What does GDPR Article 17 say?", config)
        assert "gdpr:Article 17" in result["article_refs"]

    def test_fadp_article_n(self, config):
        """pattern 1 with FADP: the user writes "Article 6" but the output uses "Art. 6" (Neo4j format)"""
        result = _run("Explain FADP Article 6", config)
        assert "fadp:Art. 6" in result["article_refs"]

    def test_article_n_of_the_gdpr(self, config):
        """pattern 2: "Article 49 of the GDPR" -> "gdpr:Article 49" """
        result = _run("Article 49 of the GDPR", config)
        assert "gdpr:Article 49" in result["article_refs"]

    def test_article_n_of_fadp(self, config):
        """pattern 2 with FADP: "Article 25 of the FADP" -> "fadp:Art. 25" """
        result = _run("Article 25 of the FADP", config)
        assert "fadp:Art. 25" in result["article_refs"]

    def test_art_dot_n_gdpr(self, config):
        """pattern 3: "Art. 6 GDPR" (abbr.) -> normalised to "gdpr:Article 6" """
        result = _run("See Art. 6 GDPR", config)
        assert "gdpr:Article 6" in result["article_refs"]

    def test_art_n_fadp(self, config):
        """pattern 3 without a dot: "Art 25 FADP" -> "fadp:Art. 25" (dot is always added)"""
        result = _run("Art 25 FADP", config)
        assert "fadp:Art. 25" in result["article_refs"]

    def test_multiple_refs(self, config):
        """multiple article refs in one query: all should be extracted"""
        result = _run("Compare GDPR Article 17 and FADP Article 25", config)
        assert "gdpr:Article 17" in result["article_refs"]
        assert "fadp:Art. 25" in result["article_refs"]

    def test_no_article_ref(self, config):
        """conceptual query with no article pattern -> empty list"""
        result = _run("What is data portability?", config)
        assert result["article_refs"] == []

    def test_no_duplicate_refs(self, config):
        """same article mentioned twice via different patterns -> deduplicated"""
        result = _run("GDPR Article 17 and Article 17 GDPR", config)
        assert result["article_refs"].count("gdpr:Article 17") == 1

    def test_case_insensitive(self, config):
        """regex uses IGNORECASE, output is always normalized to standard casing"""
        result = _run("gdpr article 5", config)
        assert "gdpr:Article 5" in result["article_refs"]

    def test_fadp_format_matches_neo4j(self, config):
        """regression guard: FADP refs must use ':Art. ' to match Neo4j node_ids"""
        result = _run("FADP Article 8", config)
        fadp_refs = [r for r in result["article_refs"] if r.startswith("fadp:")]
        assert all(":Art. " in r for r in fadp_refs)

    def test_gdpr_format_matches_neo4j(self, config):
        """regression guard: GDPR refs must use ':Article ' (full word) to match Neo4j node_ids"""
        result = _run("GDPR Article 17", config)
        gdpr_refs = [r for r in result["article_refs"] if r.startswith("gdpr:")]
        assert all(":Article " in r for r in gdpr_refs)


# test class 2: Jurisdiction detection
#
# Keyword -> jurisdiction mapping (_JURISDICTION_KEYWORDS in nodes.py):
# "gdpr" / "eu" / "european" / "edpb"         : "EU"
# "fadp" / "swiss" / "switzerland" / "fdpic"  : "CH"
#
# the detected jurisdictions are passed to retrieve as Weaviate metadata filters


class TestJurisdictionDetection:
    """This verifies that jurisdiction keywords produce correct filters"""

    def test_gdpr_implies_eu(self, config):
        result = _run("GDPR obligations for controllers", config)
        assert "EU" in result["jurisdiction_filters"]

    def test_fadp_implies_ch(self, config):
        """FADP -> "CH" """
        result = _run("FADP requirements", config)
        assert "CH" in result["jurisdiction_filters"]

    def test_swiss_keyword(self, config):
        """ "Swiss implies CH even without mentioning FADP"""
        result = _run("Swiss data protection law", config)
        assert "CH" in result["jurisdiction_filters"]

    def test_european_keyword(self, config):
        """ "European implies EU even without the GDPR acronym"""
        result = _run("European data protection regulation", config)
        assert "EU" in result["jurisdiction_filters"]

    def test_both_jurisdictions(self, config):
        """Cross-jurisdictional query: both EU and CH detected"""
        result = _run("Compare GDPR and FADP erasure provisions", config)
        assert "EU" in result["jurisdiction_filters"]
        assert "CH" in result["jurisdiction_filters"]

    def test_no_jurisdiction(self, config):
        """No jurisdiction keywords : empty list, Weaviate searches all jurisdictions"""
        result = _run("What is data portability?", config)
        assert result["jurisdiction_filters"] == []


# test class 3: Source filters
#
# the source filters tell Weaviate which document(s) to search
# _SOURCE_MAP in nodes.py:
# "gdpr"   : "gdpr"
# "fadp"   : "fadp"
# "fdpic"  : "fdpic_technical_measures"
# "edpb"   : None (maps to no filter)


class TestSourceFilters:
    """this verifies source keywords produce the correct Weaviate document filters"""

    def test_gdpr_source(self, config):
        result = _run("GDPR Article 17", config)
        assert "gdpr" in result["source_filters"]

    def test_fadp_source(self, config):
        result = _run("FADP provisions", config)
        assert "fadp" in result["source_filters"]

    def test_fdpic_source(self, config):
        """FDPIC unambiguously maps to "fdpic_technical_measures" """
        result = _run("FDPIC technical measures guide", config)
        assert "fdpic_technical_measures" in result["source_filters"]

    def test_edpb_no_source_filter(self, config):
        """EDPB maps to None : no source filter added. Weaviate searches all the EU docs instead"""
        result = _run("EDPB guidelines on consent", config)
        edpb_sources = [s for s in result["source_filters"] if "edpb" in s]
        assert edpb_sources == []


# test class 4: Stage tracking
#
# each node appends its name to "stages_completed" for diagnostics/debugging
# interpret_node also sets "search_query", the original query passed through
# unchanged for the retrieve node to embed and vector search in Weaviate


class TestStageTracking:
    """this verifies that the interpret node correctly updates pipeline diagnostics"""

    def test_interpret_added_to_stages(self, config):
        """ "interpret should appear in stages_completed after running

        The node does an append (new list) so LangGraph can cleanly
        merge the partial state update
        """
        result = _run("test query", config)
        assert "interpret" in result["stages_completed"]

    def test_search_query_set(self, config):
        """search_query should equal the original query text as is"""
        result = _run("What is consent?", config)
        assert result["search_query"] == "What is consent?"
