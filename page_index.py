"""
Page index builder: extracts normalized text per page and saves as a
lightweight JSON file for page number lookups at query time.

Uses two sources:
1. PyMuPDF text extraction from the raw PDF (works for text-based PDFs)
2. Enriched text with PAGE N markers (works for image-heavy/scanned PDFs)

The enriched text source is preferred when available since it includes
AI-generated descriptions of visual content.
"""

import json
import re
from pathlib import Path

import fitz  # PyMuPDF

_NON_ASCII_RE = re.compile(r'[^\x20-\x7E]')
_HYPHEN_BREAK_RE = re.compile(r'-\s+')
_PAGE_SPLIT_RE = re.compile(r'={10,}\nPAGE\s+(\d+)\n={10,}')

INDEX_DIR = Path(__file__).parent / "page_indexes"


def _normalize(text: str) -> str:
    """Normalize text for matching: rejoin hyphens, strip non-ASCII, collapse whitespace."""
    text = _HYPHEN_BREAK_RE.sub('', text)
    text = _NON_ASCII_RE.sub('', text)
    return " ".join(text.split())


def build_page_index_from_pdf(pdf_path: str) -> dict[str, str]:
    """Extract normalized text per page from a PDF. Returns {page_num_str: normalized_text}."""
    doc = fitz.open(pdf_path)
    index = {}
    for page in doc:
        page_num = page.number + 1  # 1-indexed
        text = _normalize(page.get_text())
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
        normalized = _normalize(page_content)
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


def save_page_index(pdf_path: str, enriched_text: str | None = None,
                    index_dir: str | Path | None = None) -> Path:
    """Build and save page index JSON. Returns path to the index file."""
    index = build_page_index(pdf_path, enriched_text)
    out_dir = Path(index_dir) if index_dir else INDEX_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_name = Path(pdf_path).stem
    index_path = out_dir / f"{pdf_name}_pages.json"

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    return index_path


def save_page_index_from_bytes(pdf_bytes: bytes, filename: str,
                               enriched_text: str | None = None,
                               index_dir: str | Path | None = None) -> Path:
    """Build and save page index from PDF bytes. Returns path to the index file."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return save_page_index(tmp_path, enriched_text, index_dir)
    finally:
        os.unlink(tmp_path)


def load_page_index(index_path: str | Path) -> dict[str, str]:
    """Load a page index JSON file."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)
