"""this runs the olmocr extraction via a HuggingFace Inference Endpoint

It extracts text from the regulatory PDF documents using olmOCR (Allen AI VLM).

every PDF page is rendered to PNG and sent to a remote GPU endpoint,
which returns extracted text in Markdown format

this is one of 3 extractors compared in the project (PyMuPDF, olmOCR,
Mistral). The results are already saved in "data/output/olmocr/"

Usage: python scripts/run_olmocr.py
"""

# import libraries
import base64
import json
import os
import time
from pathlib import Path
import fitz  # PyMuPDF, which is used to render PDF pages to PNG images
import requests as http_requests
from dotenv import load_dotenv

# Configuration

load_dotenv()

# HuggingFace Inference Endpoint running olmOCR on an NVIDIA A100 (temporary and was shut down to save costs)
ENDPOINT_URL = "https://bvdz56fgw1tl16pm.us-east4.gcp.endpoints.huggingface.cloud"
HF_TOKEN = os.getenv("HF_TOKEN")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "data" / "documents"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output" / "olmocr"

# this is the instruction sent with every page image
# it is designed for legal documents where article numbers,
# cross-references, and structure must be preserved 1:1
PROMPT = (
    "Below is a PDF page rendered as an image. "
    "Extract all text content faithfully in Markdown format. "
    "Preserve document structure: headings, numbered articles, "
    "bullet points, tables (as HTML), footnotes, and cross-references. "
    "Output ONLY the extracted text, no commentary."
)

# this maps PDF filenames to short output folder names, in line with the
# PyMuPDF extractor so that the comparison harness can locate results
PDF_TO_FOLDER = {
    "gdpr_full_text.pdf": "gdpr",
    "fadp_revised_2023.pdf": "fadp",
    "edpb_guidelines_legitimate_interest.pdf": "edpb_legitimate_interest",
    "edpb_guidelines_article48_transfers.pdf": "edpb_article48",
    "edpb_guidelines_consent.pdf": "edpb_consent",
    "fdpic_guide_technical_measures.pdf": "fdpic_technical_measures",
}

# helper functions


def render_page_to_base64(pdf_path: Path, page_num: int) -> str:
    """to render a 1-indexed PDF page to a base64 PNG string

    Args:
    pdf_path:  the full path to the PDF file
    page_num:  the page number to render (1-indexed)

    Returns:
    base64-encoded PNG string suitable for embedding in JSON payload
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]  # PyMuPDF uses 0-based indexing
    pix = page.get_pixmap(dpi=150)  # 150 DPI
    img_bytes = pix.tobytes("png")
    doc.close()
    return base64.b64encode(img_bytes).decode()


def extract_page(pdf_path: Path, page_num: int, max_retries: int = 3) -> str:
    """this sends a single PDF page to the olmOCR endpoint and returns the extracted text

    it retries up to max_retries times.

    Args:
    pdf_path:     the full path to the PDF file
    page_num:     page number to extract (1-indexed)
    max_retries:  the number of retry attempts on failure (default is 3)

    Returns:
    extracted text in Markdown format

    Raises:
    Exception: if all retry attempts fail
    """
    img_b64 = render_page_to_base64(pdf_path, page_num)

    # OpenAI compatible chat completions payload with multimodal user message
    payload = {
        "model": "allenai/olmOCR-2-7B-1025-FP8",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.1,  # almost deterministic temp
    }

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            resp = http_requests.post(
                f"{ENDPOINT_URL}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,  # VLM inference may take several sec per page
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 10 * (attempt + 1)
                print(f"    Retry {attempt + 1} in {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


# main function


def main():
    """this extracts text from all the regulatory PDFs using olmOCR

    For each document it saves:
    - pages.json:               extracted text per page with timing data
    - full_text.txt:            all the pages concatenated
    - extraction_summary.json:  statistics (page count, timing, success rate)

    """
    if not HF_TOKEN:
        print("Error: HF_TOKEN not found in .env")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {INPUT_DIR}\n")

    # this tests the endpoint with one page before committing to the full run
    print("Testing endpoint with one page...")
    try:
        test = extract_page(pdfs[0], 1)
        print(f"Test OK - got {len(test)} chars\n")
    except Exception as e:
        print(f"Endpoint test FAILED: {e}")
        print("Check that endpoint is in 'Running' state on huggingface.co")
        return

    all_summaries = []

    for pdf_path in pdfs:
        folder_name = PDF_TO_FOLDER.get(pdf_path.name, pdf_path.stem)

        print(f"{'=' * 60}")
        print(f"Processing: {pdf_path.name} -> {folder_name}")

        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        doc.close()
        print(f"Pages: {num_pages}")

        pages_output = []
        full_text_parts = []
        start_time = time.time()

        # the pages are processed sequentially to avoid overloading the GPU endpoint
        for page_num in range(1, num_pages + 1):
            page_start = time.time()
            try:
                content = extract_page(pdf_path, page_num)
                page_time = time.time() - page_start
                pages_output.append(
                    {
                        "page": page_num,
                        "content": content,
                        "processing_time_seconds": round(page_time, 2),
                    }
                )
                full_text_parts.append(content)
                print(
                    f" Page {page_num}/{num_pages} — {page_time:.1f}s — {len(content)} chars"
                )

            except Exception as e:
                # this records the failure but continues with remaining pages
                page_time = time.time() - page_start
                pages_output.append(
                    {
                        "page": page_num,
                        "content": "",
                        "error": str(e),
                        "processing_time_seconds": round(page_time, 2),
                    }
                )
                print(f" Page {page_num}/{num_pages} - error: {e}")

        total_time = time.time() - start_time

        doc_output_dir = OUTPUT_DIR / folder_name
        doc_output_dir.mkdir(parents=True, exist_ok=True)

        with open(doc_output_dir / "pages.json", "w", encoding="utf-8") as f:
            json.dump(pages_output, f, indent=2, ensure_ascii=False)

        with open(doc_output_dir / "full_text.txt", "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text_parts))

        summary = {
            "document": folder_name,
            "source_file": pdf_path.name,
            "total_pages": num_pages,
            "successful_pages": sum(1 for p in pages_output if "error" not in p),
            "failed_pages": sum(1 for p in pages_output if "error" in p),
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_page_seconds": round(total_time / num_pages, 2),
            "total_chars": sum(len(p["content"]) for p in pages_output),
        }
        with open(
            doc_output_dir / "extraction_summary.json", "w", encoding="utf-8"
        ) as f:
            json.dump(summary, f, indent=2)

        all_summaries.append(summary)
        print(
            f"Done: {num_pages} pages in {total_time:.0f}s "
            f"({total_time / num_pages:.1f}s/page)\n"
        )

    with open(OUTPUT_DIR / "extraction_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)

    print("=" * 60)
    print("ALL DONE!")
    print(f"Results saved to: {OUTPUT_DIR}")

    total_pages = sum(s["total_pages"] for s in all_summaries)
    total_ok = sum(s["successful_pages"] for s in all_summaries)
    print(f"Total: {total_ok}/{total_pages} pages extracted successfully")


if __name__ == "__main__":
    main()
