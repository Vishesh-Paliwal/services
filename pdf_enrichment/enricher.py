import json
import logging
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
import fitz  # PyMuPDF

from .detector import detect_visual_pages
from .describer import describe_pages_batch

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent.parent / "logs" / "enrichment"


def _log_descriptions(source_filename: str, descriptions: dict[int, str], visual_pages: list):
    """Write all vision descriptions to a JSON log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = source_filename.replace(" ", "_").replace("/", "_")
    log_path = LOG_DIR / f"{timestamp}_{safe_name}.json"

    # Build page analysis lookup
    analysis_map = {p.page_num: p for p in visual_pages}

    entries = []
    for page_num in sorted(descriptions.keys()):
        analysis = analysis_map.get(page_num)
        entries.append({
            "page": page_num + 1,
            "has_images": analysis.has_images if analysis else False,
            "has_tables": analysis.has_tables if analysis else False,
            "has_drawings": analysis.has_drawings if analysis else False,
            "description": descriptions[page_num],
        })

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "source": source_filename,
        "total_described": len(descriptions),
        "pages": entries,
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    logger.info("Descriptions logged to %s", log_path)


def _extract_all_text(pdf_path: str) -> dict[int, str]:
    """Extract text from all pages. Returns {0-indexed page_num: text}."""
    doc = fitz.open(pdf_path)
    texts = {page.number: page.get_text() for page in doc}
    doc.close()
    return texts


def _build_enriched_text(
    pdf_path: str,
    page_texts: dict[int, str],
    visual_descriptions: dict[int, str],
    source_filename: str,
) -> str:
    """Merge extracted text with visual descriptions into enriched document."""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    parts = [
        f"# {source_filename}",
        f"Source: {source_filename}",
        f"Total pages: {total_pages}\n",
    ]

    for page_num in range(total_pages):
        page_display = page_num + 1
        text = page_texts.get(page_num, "").strip()
        description = visual_descriptions.get(page_num, None)

        parts.append(f"\n{'='*60}")
        parts.append(f"PAGE {page_display}")
        parts.append(f"{'='*60}\n")

        if text:
            parts.append(text)

        if description:
            parts.append(f"\n--- Visual Content on Page {page_display} ---")
            parts.append(description)
            parts.append(f"--- End Visual Content ---\n")

    return "\n".join(parts)


def enrich_pdf(
    client,
    pdf_path: str,
    source_filename: str,
    model: str = "gemini-2.5-flash",
    progress_callback=None,
) -> tuple[str, dict]:
    """Full enrichment pipeline. Returns (enriched_text, stats)."""
    t_start = time.perf_counter()
    logger.info("Starting enrichment for '%s'", source_filename)

    # Step 1: Detect visual pages
    t0 = time.perf_counter()
    visual_pages = detect_visual_pages(pdf_path)
    visual_page_nums = [p.page_num for p in visual_pages]
    t_detect = time.perf_counter() - t0
    logger.info("Detection: %.1fs — %d visual pages found", t_detect, len(visual_page_nums))

    # Step 2: Extract all text
    t0 = time.perf_counter()
    page_texts = _extract_all_text(pdf_path)
    t_extract = time.perf_counter() - t0
    logger.info("Text extraction: %.1fs — %d pages", t_extract, len(page_texts))

    # Step 3: Get visual descriptions from Gemini
    t0 = time.perf_counter()
    descriptions = {}
    failed = 0
    if visual_page_nums:
        descriptions = describe_pages_batch(
            client, pdf_path, visual_page_nums,
            model=model,
            progress_callback=progress_callback,
        )
        failed = sum(1 for d in descriptions.values() if d.startswith("[Description failed"))
    t_describe = time.perf_counter() - t0
    logger.info("Vision descriptions: %.1fs — %d succeeded, %d failed",
                t_describe, len(descriptions) - failed, failed)

    # Step 4: Build enriched text
    enriched = _build_enriched_text(
        pdf_path, page_texts, descriptions, source_filename
    )

    # Step 5: Log descriptions to file
    _log_descriptions(source_filename, descriptions, visual_pages)

    t_total = time.perf_counter() - t_start
    logger.info("Enrichment complete: %.1fs total, %d chars output", t_total, len(enriched))

    stats = {
        "total_pages": len(page_texts),
        "visual_pages": len(visual_page_nums),
        "pages_with_images": sum(1 for p in visual_pages if p.has_images),
        "pages_with_tables": sum(1 for p in visual_pages if p.has_tables),
        "pages_with_drawings": sum(1 for p in visual_pages if p.has_drawings),
        "descriptions_failed": failed,
        "enriched_text_length": len(enriched),
        "time_detect_s": round(t_detect, 1),
        "time_extract_s": round(t_extract, 1),
        "time_describe_s": round(t_describe, 1),
        "time_total_s": round(t_total, 1),
    }

    return enriched, stats


def enrich_pdf_from_bytes(
    client,
    pdf_bytes: bytes,
    source_filename: str,
    model: str = "gemini-2.5-flash",
    progress_callback=None,
) -> tuple[str, dict]:
    """Convenience wrapper accepting bytes (from Streamlit file_uploader)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return enrich_pdf(client, tmp_path, source_filename, model, progress_callback)
    finally:
        os.unlink(tmp_path)


def enrich_pdf_from_local(
    client,
    local_path: str,
    source_filename: str,
    model: str = "gemini-2.5-flash",
    progress_callback=None,
) -> tuple[str, dict]:
    """Enrich a PDF already on the local filesystem (e.g. downloaded from GCS)."""
    return enrich_pdf(client, local_path, source_filename, model, progress_callback)
