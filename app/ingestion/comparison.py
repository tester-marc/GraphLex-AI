# comparison.py: 3-way document understanding comparison harness
#
# Purpose: to compare PyMuPDF, olmocr, and Mistral Document AI on regulatory PDFs,
# and to score each against manually created ground truth in order to justify extractor selection
#
# Extractors:
# 1. PyMuPDF
# 2. olmocr
# 3. Mistral
#
# Metrics (per document):
# 1. Structure Preservation     : the proportion of expected article numbers found
# 2. Table TEDS                 : simplified cell level recall vs. ground truth tables
# 3. Cross-Reference Detection  : the proportion of known cross-ref. targets found
# 4. Footnote Preservation      : the proportion of known footnotes found
# 5. Processing Time (ms/page)
#
#
# Decision: PyMuPDF selected, details elaborated on in the final report.
#
# the ground truth annotations are saved in "data/ground_truth/"" as JSON files (e.g., gdpr.json)
# this harness is run once during development, the results feed into the tables in the final report
#

"""the comparison harness, i.e., run the extractors, score against ground truth, and produce metrics"""

# import libraries
from __future__ import annotations
import json
import re
from pathlib import Path
from app.ingestion.base_extractor import BaseExtractor
from app.ingestion.config import DocumentEntry, IngestionConfig
from app.ingestion.ground_truth import GroundTruth, load_ground_truth
from app.ingestion.models import ComparisonResult, DocumentMetadata, ExtractionResult


class ComparisonHarness:
    """this run the extractors on the regulatory PDFs and evaluates against ground truth

    It produces one ComparisonResult per (extractor, document) pair

    Usage:
    config = IngestionConfig(active_extractors=["pymupdf", "olmocr", "mistral"])
    harness = ComparisonHarness(config, [pymupdf, olmocr, mistral])
    results = harness.run()
    harness.save_results(results, Path("data/output/comparison_results.json"))
    print(harness.format_results(results))
    """

    def __init__(self, config: IngestionConfig, extractors: list[BaseExtractor]):
        """
        Args:
        config: specifies documents, active extractors, and file paths
        extractors: for extractor instances, stored as name -> extractor dict
        """
        self.config = config
        self.extractors = {e.name: e for e in extractors}

    def run(self) -> list[ComparisonResult]:
        """this runs all active extractors on all the configured documents

        It skips missing PDF files, unavailable extractors, and extractors not
        listed in "config.active_extractors"
        It prints progress to stdout

        Returns:
        a list of ComparisonResult objects (one per extractor x document)
        """
        results: list[ComparisonResult] = []

        for doc_entry in self.config.documents:
            file_path = self.config.document_path(doc_entry)
            if not file_path.exists():
                print(f" Skip {doc_entry.source_id}: file not found at {file_path}")
                continue

            # this returns None if no ground truth file exists, and the metrics will be None
            gt = load_ground_truth(self.config.ground_truth_dir, doc_entry.source_id)

            for ext_name in self.config.active_extractors:
                extractor = self.extractors.get(ext_name)
                if not extractor:
                    continue
                if not extractor.is_available():
                    print(f" Skip {ext_name}: not available")
                    continue

                print(f" Extracting {doc_entry.source_id} with {ext_name}...")
                result = self.run_single(extractor, doc_entry, file_path, gt)
                results.append(result)

        return results

    def run_single(
        self,
        extractor: BaseExtractor,
        doc_entry: DocumentEntry,
        file_path: Path,
        ground_truth: GroundTruth | None,
    ) -> ComparisonResult:
        """this runs one extractor on one document and computes all the metrics

        quality metrics (1 to 4) are only computed when ground truth exists

        Args:
        extractor:    the extractor to run
        doc_entry:    the document configuration (source_id, title, etc.)
        file_path:    full path to the PDF
        ground_truth: manual annotations, or None

        Returns:
        ComparisonResult with all five metric scores
        """
        metadata = DocumentMetadata(
            source_id=doc_entry.source_id,
            title=doc_entry.title,
            instrument_type=doc_entry.instrument_type,
            jurisdiction=doc_entry.jurisdiction,
            effective_date=doc_entry.effective_date,
            file_path=file_path,
        )

        extraction = extractor.extract(file_path, metadata)

        # max(..., 1) protects against division by zero for empty documents
        time_per_page = extraction.total_processing_time_ms / max(
            len(extraction.pages), 1
        )

        structure_score = None
        teds_score = None
        cross_ref_rate = None
        footnote_rate = None

        if ground_truth:
            structure_score = self._structure_score(extraction, ground_truth)
            teds_score = self._teds_score(extraction, ground_truth)
            cross_ref_rate = self._cross_ref_rate(extraction, ground_truth)
            footnote_rate = self._footnote_rate(extraction, ground_truth)

        return ComparisonResult(
            extractor_name=extractor.name,
            document_id=doc_entry.source_id,
            complexity_level=doc_entry.complexity_level,
            structure_preservation_score=structure_score,
            table_teds_score=teds_score,
            cross_reference_detection_rate=cross_ref_rate,
            footnote_preservation_rate=footnote_rate,
            processing_time_per_page_ms=time_per_page,
        )

    # metrics

    def _structure_score(self, extraction: ExtractionResult, gt: GroundTruth) -> float:
        """the proportion of expected article numbers found in the extracted text

        returns 1.0 if ground truth has no articles
        """
        if not gt.articles:
            return 1.0

        text = extraction.full_text
        found = 0
        for article in gt.articles:
            patterns = [
                f"Article {article.number}",
                f"Art. {article.number}",
            ]
            if any(p in text for p in patterns):
                found += 1

        return found / len(gt.articles)

    def _teds_score(
        self, extraction: ExtractionResult, gt: GroundTruth
    ) -> float | None:
        """simplified TEDS, cell level recall of extracted tables vs. ground truth

        full TEDS uses the tree-edit-distance on HTML DOM trees
        this is a simplified version that flattens tables to cell lists
        and measures recall

        It matches extracted tables to ground truth tables by page number, takes
        the best score per page when multiple extracted tables exist
        Scores are averaged over all ground truth tables.

        Returns None if doc has no ground truth tables
        """
        if not gt.tables:
            return None

        scores: list[float] = []
        for gt_table in gt.tables:
            best_score = 0.0
            for ext_table in extraction.tables:
                if ext_table.page_number == gt_table.page:
                    score = self._table_similarity(ext_table.rows, gt_table.cells)
                    best_score = max(best_score, score)
            scores.append(best_score)

        return sum(scores) / len(scores) if scores else None

    def _table_similarity(
        self, extracted: list[list[str]], expected: list[list[str]]
    ) -> float:
        """this is the cell level recall between 2 tables

        It flattens both tables to normalized (stripped, lowercased) non empty cell
        lists, then returns proportion of expected cells found in extracted.
        """
        if not extracted or not expected:
            return 0.0

        ext_cells = [
            cell.strip().lower() for row in extracted for cell in row if cell.strip()
        ]
        exp_cells = [
            cell.strip().lower() for row in expected for cell in row if cell.strip()
        ]

        if not exp_cells:
            return 1.0

        matched = sum(1 for cell in exp_cells if cell in ext_cells)
        return matched / len(exp_cells)

    def _cross_ref_rate(self, extraction: ExtractionResult, gt: GroundTruth) -> float:
        """the proportion of known cross-reference targets found in the extracted text

        cross-references (e.g., "see Article 44") are crucial for building
        "REFERENCES" relationships in the Graph Layer. This checks both "Article N"
        and "Art. N" formats

        returns 1.0 if ground truth has no cross-references
        """
        if not gt.cross_references:
            return 1.0

        text = extraction.full_text
        found = 0
        for ref in gt.cross_references:
            target = ref.target_article
            patterns = [
                f"Article {target}",
                f"Art. {target}",
            ]
            if any(p in text for p in patterns):
                found += 1

        return found / len(gt.cross_references)

    def _footnote_rate(self, extraction: ExtractionResult, gt: GroundTruth) -> float:
        """for the proportion of known footnotes preserved in extracted text

        It uses a partial match : checks for the first 30chars of each ground
        truth footnote

        returns 1.0 if ground truth has no footnotes
        """
        if not gt.footnotes:
            return 1.0

        text = extraction.full_text
        found = 0
        for fn in gt.footnotes:
            if fn.text[:30] in text:
                found += 1

        return found / len(gt.footnotes)

    # output

    def save_results(self, results: list[ComparisonResult], output_path: Path) -> None:
        """this saves comparison results to JSON at "output_path" """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for r in results:
            data.append(
                {
                    "extractor": r.extractor_name,
                    "document": r.document_id,
                    "complexity_level": r.complexity_level,
                    "structure_preservation_score": r.structure_preservation_score,
                    "table_teds_score": r.table_teds_score,
                    "cross_reference_detection_rate": r.cross_reference_detection_rate,
                    "footnote_preservation_rate": r.footnote_preservation_rate,
                    "processing_time_per_page_ms": round(
                        r.processing_time_per_page_ms, 2
                    ),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def format_results(results: list[ComparisonResult]) -> str:
        """this returns an aligned ASCII table of results for the terminal output

        Columns:
        Extractor, Document, Lvl, Structure, TEDS, XRef, Footnote, ms/page
        """
        if not results:
            return "No results."

        header = (
            f"{'Extractor':<12} {'Document':<30} {'Lvl':>3} "
            f"{'Structure':>10} {'TEDS':>8} {'XRef':>8} {'Footnote':>8} "
            f"{'ms/page':>8}"
        )
        lines = [header, "-" * len(header)]

        for r in results:

            def fmt(v: float | None) -> str:
                return f"{v:.3f}" if v is not None else "N/A"

            lines.append(
                f"{r.extractor_name:<12} {r.document_id:<30} {r.complexity_level:>3} "
                f"{fmt(r.structure_preservation_score):>10} "
                f"{fmt(r.table_teds_score):>8} "
                f"{fmt(r.cross_reference_detection_rate):>8} "
                f"{fmt(r.footnote_preservation_rate):>8} "
                f"{r.processing_time_per_page_ms:>8.1f}"
            )

        return "\n".join(lines)
