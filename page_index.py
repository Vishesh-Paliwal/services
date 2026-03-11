"""
Page index builder: extracts normalized text per page and saves as a
lightweight JSON file for page number lookups at query time.

Uses two sources:
1. PyMuPDF text extraction from the raw PDF (works for text-based PDFs)
2. Enriched text with PAGE N markers (works for image-heavy/scanned PDFs)

The enriched text source is preferred when available since it includes
AI-generated descriptions of visual content.

Storage: indexes are saved locally and optionally synced to GCS for
persistence across Cloud Run instances.
"""

import json
import logging
import os
import re
from pathlib import Path

import fitz  # PyMuPDF

from text_utils import normalize

_PAGE_SPLIT_RE = re.compile(r'={10,}\nPAGE\s+(\d+)\n={10,}')

INDEX_DIR = Path(os.environ.get("PAGE_INDEX_DIR", Path(__file__).parent / "page_indexes"))
GCS_BUCKET = os.environ.get("GCS_PAGE_INDEX_BUCKET", "")
GCS_PREFIX = os.environ.get("GCS_PAGE_INDEX_PREFIX", "page_indexes/")

logger = logging.getLogger(__name__)


def build_page_index_from_pdf(pdf_path: str) -> dict[str, str]:
    """Extract normalized text per page from a PDF. Returns {page_num_str: normalized_text}."""
    doc = fitz.open(pdf_path)
    index = {}
    for page in doc:
        page_num = page.number + 1  # 1-indexed
        text = normalize(page.get_text())
        if text:
            index[str(page_num)] = text
    doc.close()
    return index


def build_page_index_from_enriched(enriched_text: str) -> dict[str, str]:
    """Extract normalized text per page from enriched text using PAGE N markers.
    Works for all PDFs including image-only/scanned pages."""
    parts = _PAGE_SPLIT_RE.split(enriched_text)
    # parts alternates: [header, page_num, page_content, page_num, page_content, ...]
    index = {}
    for i in range(1, len(parts) - 1, 2):
        page_num = parts[i]
        page_content = parts[i + 1]
        normalized = normalize(page_content)
        if normalized:
            index[page_num] = normalized
    return index


def build_page_index(pdf_path: str, enriched_text: str | None = None) -> dict[str, str]:
    """Build page index from best available source.

    If enriched_text is provided, uses it (covers image pages with AI descriptions).
    Falls back to raw PDF text extraction.
    Merges both if enriched text has gaps.
    """
    # Start with PDF extraction
    pdf_index = build_page_index_from_pdf(pdf_path)

    if enriched_text:
        enriched_index = build_page_index_from_enriched(enriched_text)
        # Merge: enriched text wins (has visual descriptions), PDF fills gaps
        merged = {}
        all_pages = set(pdf_index.keys()) | set(enriched_index.keys())
        for page in all_pages:
            enriched = enriched_index.get(page, "")
            pdf = pdf_index.get(page, "")
            # Use whichever is longer (enriched usually has more due to descriptions)
            merged[page] = enriched if len(enriched) >= len(pdf) else pdf
        return merged

    return pdf_index


def _upload_to_gcs(local_path: Path, gcs_key: str) -> None:
    """Upload a local file to GCS. No-op if GCS_BUCKET is not set."""
    if not GCS_BUCKET:
        return
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_key)
        blob.upload_from_filename(str(local_path))
        logger.info("Uploaded page index to gs://%s/%s", GCS_BUCKET, gcs_key)
    except Exception as e:
        logger.warning("GCS upload failed (non-fatal): %s", e)


def _download_from_gcs(gcs_key: str, local_path: Path) -> bool:
    """Download a file from GCS to local path. Returns True on success."""
    if not GCS_BUCKET:
        return False
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(gcs_key)
        if not blob.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        logger.info("Downloaded page index from gs://%s/%s", GCS_BUCKET, gcs_key)
        return True
    except Exception as e:
        logger.warning("GCS download failed: %s", e)
        return False


def _index_filename(display_name: str) -> str:
    """Derive a stable index filename from the original PDF display name."""
    base = display_name.replace("_enriched.txt", "").replace(".pdf", "").replace(".PDF", "")
    return f"{base}_pages.json"


def save_page_index(pdf_path: str, display_name: str,
                    enriched_text: str | None = None,
                    index_dir: str | Path | None = None) -> Path:
    """Build and save page index JSON. Returns path to the index file.

    Args:
        pdf_path: Path to the PDF file (can be a temp file).
        display_name: Original filename of the PDF (used for naming the index).
        enriched_text: Optional enriched text with PAGE N markers.
        index_dir: Optional override for the output directory.
    """
    index = build_page_index(pdf_path, enriched_text)
    out_dir = Path(index_dir) if index_dir else INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_name = _index_filename(display_name)
    index_path = out_dir / idx_name

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    _upload_to_gcs(index_path, f"{GCS_PREFIX}{idx_name}")

    return index_path


def save_page_index_from_bytes(pdf_bytes: bytes, filename: str,
                               enriched_text: str | None = None,
                               index_dir: str | Path | None = None) -> Path:
    """Build and save page index from PDF bytes. Returns path to the index file.

    Uses the original filename (not the temp path) for naming the index.
    """
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return save_page_index(tmp_path, filename, enriched_text, index_dir)
    finally:
        os.unlink(tmp_path)


def load_page_index(index_path: str | Path) -> dict[str, str]:
    """Load a page index JSON file."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_page_index_by_name(display_name: str,
                            index_dir: str | Path | None = None) -> dict[str, str] | None:
    """Load a page index by the original PDF display name.

    Checks local cache first, then falls back to GCS.
    Returns None if not found.
    """
    out_dir = Path(index_dir) if index_dir else INDEX_DIR
    idx_name = _index_filename(display_name)
    local_path = out_dir / idx_name

    # Try local first
    if local_path.exists():
        return load_page_index(local_path)

    # Try GCS download
    gcs_key = f"{GCS_PREFIX}{idx_name}"
    if _download_from_gcs(gcs_key, local_path):
        return load_page_index(local_path)

    return None
