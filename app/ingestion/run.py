# run.py: the CLI entry point for the GraphLex AI Document Ingestion Pipeline
#
# Commands:
# python -m app.ingestion --extract                      # to extract all docs with PyMuPDF (default)
# python -m app.ingestion --extract --extractor olmocr   # use olmocr instead
# python -m app.ingestion --extract --document gdpr      # extract only the GDPR
# python -m app.ingestion --compare                      # run the 3-way extractor comparison
#
# this does not handle embedding, vector storage, or graph construction (for that see: app/embeddings/, app/graph/)

"""CLI entry point: python -m app.ingestion.run"""

# import libraries
from __future__ import annotations
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from app.ingestion.chunker import LegalChunker
from app.ingestion.comparison import ComparisonHarness
from app.ingestion.config import IngestionConfig
from app.ingestion.mistral_extractor import MistralDocumentAIExtractor
from app.ingestion.models import DocumentMetadata
from app.ingestion.olmocr_extractor import OlmOCRExtractor
from app.ingestion.pymupdf_extractor import PyMuPDFExtractor


def get_extractors():
    return [PyMuPDFExtractor(), OlmOCRExtractor(), MistralDocumentAIExtractor()]


def run_extract(config: IngestionConfig, source_id: str | None, extractor_name: str):
    """to extract documents and produce chunks"""

    extractors = {e.name: e for e in get_extractors()}
    extractor = extractors.get(extractor_name)

    if not extractor:
        print(f"Unknown extractor: {extractor_name}")
        sys.exit(1)

    # olmocr needs pre-computed files, Mistral needs the MISTRAL_API_KEY, and PyMuPDF is always available
    if not extractor.is_available():
        print(
            f"Extractor '{extractor_name}' is not available: {type(extractor).__doc__}"
        )
        sys.exit(1)

    docs = config.documents

    if source_id:
        entry = config.get_document(source_id)
        if not entry:
            print(f"Unknown document: {source_id}")
            sys.exit(1)
        docs = [entry]

    chunker = LegalChunker(
        max_tokens=config.max_chunk_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )

    for doc_entry in docs:
        file_path = config.document_path(doc_entry)

        if not file_path.exists():
            print(f"Skip {doc_entry.source_id}: {file_path} not found")
            continue

        print(f"Extracting {doc_entry.source_id} with {extractor_name}...")

        metadata = DocumentMetadata(
            source_id=doc_entry.source_id,
            title=doc_entry.title,
            instrument_type=doc_entry.instrument_type,
            jurisdiction=doc_entry.jurisdiction,
            effective_date=doc_entry.effective_date,
            file_path=file_path,
        )

        # step 1: extract the text from the PDF
        result = extractor.extract(file_path, metadata)

        # step 2: split into legal structure aware chunks
        # the statutes split on article boundaries, the guidance docs split on numbered section headings
        chunks = chunker.chunk(result)

        # step 3: save the outputs to data/output/<extractor>/<source_id>/
        out_dir = config.output_dir / extractor_name / doc_entry.source_id
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "extractor": result.extractor_name,
            "source_id": doc_entry.source_id,
            "total_pages": result.metadata.total_pages,
            "toc_entries": len(result.toc_entries),
            "tables_found": len(result.tables),
            "total_processing_time_ms": round(result.total_processing_time_ms, 2),
            "full_text_chars": len(result.full_text),
        }
        with open(out_dir / "extraction_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        chunks_data = [asdict(c) for c in chunks]
        with open(out_dir / "chunks.json", "w", encoding="utf-8") as f:
            # ensure_ascii=False preserves Unicode (e.g., German umlauts, French accents)
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        with open(out_dir / "full_text.txt", "w", encoding="utf-8") as f:
            f.write(result.full_text)

        print(
            f"  {doc_entry.source_id}: {result.metadata.total_pages} pages, "
            f"{len(chunks)} chunks, {len(result.tables)} tables, "
            f"{result.total_processing_time_ms:.0f}ms"
        )

    print("Done.")


def run_compare(config: IngestionConfig):
    """this runs the comparison harness"""

    # this enables all 3 extractors, only "pymupdf" is active by default
    config.active_extractors = ["pymupdf", "olmocr", "mistral"]

    harness = ComparisonHarness(config, get_extractors())

    print("Running comparison harness...")
    results = harness.run()

    print("\n" + harness.format_results(results))

    out_path = config.output_dir / "comparison_results.json"
    harness.save_results(results, out_path)
    print(f"\nResults saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="GraphLex AI Ingestion Pipeline")

    parser.add_argument(
        "--extract", action="store_true", help="Runs extraction on documents"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Runs comparison harness across all available extractors",
    )
    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Processes a single document by source_id",
    )
    # PyMuPDF is the default: it is the fastest extractor
    parser.add_argument(
        "--extractor",
        type=str,
        default="pymupdf",
        help="Extractor to use (default: pymupdf)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override output directory"
    )

    args = parser.parse_args()
    config = IngestionConfig()

    if args.output_dir:
        config.output_dir = args.output_dir

    if args.extract:
        run_extract(config, args.document, args.extractor)
    elif args.compare:
        run_compare(config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
