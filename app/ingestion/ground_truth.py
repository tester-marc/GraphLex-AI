"""
This is for the ground truth loading for the comparison evaluation:

It handles the loading of manually created annotations that describe the known structure of
each regulatory PDF. It is then used as the benchmark when evaluating how well each text
extractor (i.e., PyMuPDF, olmocr, Mistral Document AI) preserves the legal structure

The comparison harness (app/ingestion/comparison.py) loads the ground truth for
each document and then scores it across these 5 metrics: structure preservation, table
TEDS, crossreference detection, footnote preservation, and latency

Ground truth files are to be found in data/ground_truth/ as JSON files, with 1 per document
(e.g., gdpr.json, fadp.json, edpb_consent.json) and they record the following:
- which articles exist and how many paragraphs each of them has
- what tables appear and on which pages
- which cross-references link one article to another
- which footnotes appear on which pages
"""

# import libraries
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GroundTruthArticle:
    """
    This is for a single article that is known to exist in the document

    Fields:
    number           : the article number as a string (e.g., "1", "4a"), a string is used rather
                       than an int in order to support identifiers that are not numeric
    title            : the official article title (e.g., "Definitions")
    paragraph_count  : the number of numbered paragraphs, it is used to check whether
                       the extractor captured complete article
    """

    number: str
    title: str
    paragraph_count: int


@dataclass
class GroundTruthTable:
    """
    this is for a table known to exist in the document

    It is used with the TEDS metric in order to measure how accurately each extractor
    preserved the table structure (1.0 = perfect match, 0.0 = no match)

    Fields:
    page   :  1-based page number where the table appears
    rows   :  the number of rows including header rows
    cols   :  the number of columns
    cells  :  the table contents as a 2D list of strings (a list of rows)
    """

    page: int
    rows: int
    cols: int
    cells: list[list[str]]


@dataclass
class GroundTruthCrossRef:
    """
    This is for a crossreference between two articles that exist

    these become "REFERENCES" relationships between article nodes in Neo4j
    The comparison harness checks whether the extractor has preserved the
    reference text on the given page

    Fields:
    source_article   : the aticle number that contains the reference (e.g., "5")
    target_article   : the article number being referred to (e.g., "89")
    page             : the PDF page where the crossreference appears
    """

    source_article: str
    target_article: str
    page: int


@dataclass
class GroundTruthFootnote:
    """
    This is for a footnote known to exist in the document

    The comparison harness checks whether the footnote marker
    and text appear in the extractor's output for the given page

    Fields:
    page     : the page number where the footnote appears
    marker   : the footnote marker as shown in the document (e.g., "1", "*")
    text     : the full footnote text (e.g., "OJ C 229, 31.7.2012, p. 90.")
    """

    page: int
    marker: str
    text: str


@dataclass
class GroundTruth:
    """
    For all manual annotations for a single document

    Fields:
    document_id       : this matches a "source_id" in config.py's DOCUMENT_REGISTRY
    articles          : known articles, used to evaluate the structure preservation
    tables            : known tables, used to evaluate the table extraction (TEDS)
    cross_references  : known article to article references
    footnotes         : known footnotes with markers and text
    """

    document_id: str
    articles: list[GroundTruthArticle] = field(default_factory=list)
    tables: list[GroundTruthTable] = field(default_factory=list)
    cross_references: list[GroundTruthCrossRef] = field(default_factory=list)
    footnotes: list[GroundTruthFootnote] = field(default_factory=list)


def load_ground_truth(ground_truth_dir: Path, document_id: str) -> GroundTruth | None:
    """
    To load the ground truth from JSON
    It returns None if the file doesn't exist

    It reads <ground_truth_dir>/<document_id>.json and then converts it into a typed
    GroundTruth object.
    It is called once per document by the comparison harness.

    Args:
    ground_truth_dir : the directory that contains the ground truth JSON files
                       (data/ground_truth/).
    document_id      : the source ID of the document (e.g., "gdpr", "fadp")

    Returns:
    a GroundTruth object or None if the JSON file doesn't exist

    Example:
    gt = load_ground_truth(Path("data/ground_truth"), "gdpr")

    if gt is not None:
    print(f"GDPR has {len(gt.articles)} annotated articles")
    """
    path = ground_truth_dir / f"{document_id}.json"

    if not path.exists():
        return None

    # UTF-8 encoding that is required for legal text (because of §, accented letters, em-dashes)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return GroundTruth(
        document_id=document_id,
        articles=[GroundTruthArticle(**a) for a in data.get("articles", [])],
        tables=[GroundTruthTable(**t) for t in data.get("tables", [])],
        cross_references=[
            GroundTruthCrossRef(**r) for r in data.get("cross_references", [])
        ],
        footnotes=[GroundTruthFootnote(**fn) for fn in data.get("footnotes", [])],
    )
