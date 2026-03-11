import logging
import os
import re
from pathlib import Path
import fitz  # PyMuPDF

from page_index import load_page_index, load_page_index_by_name, INDEX_DIR
from text_utils import normalize

logger = logging.getLogger(__name__)

# Cache: filename -> list of normalized page texts
_pdf_cache: dict[str, list[str]] = {}
# Cache: index filename -> {page_num_str: normalized_text}
_index_cache: dict[str, dict[str, str]] = {}

PDF_DIR = Path(os.environ.get("PDF_DIR", Path(__file__).parent / "books"))


def _load_pdf(pdf_path: str) -> list[str]:
    """Extract and cache normalized text per page from a PDF."""
    if pdf_path in _pdf_cache:
        return _pdf_cache[pdf_path]

    doc = fitz.open(pdf_path)
    pages = [normalize(page.get_text()) for page in doc]
    doc.close()
    _pdf_cache[pdf_path] = pages
    return pages


_PAGE_MARKER_RE = re.compile(r'PAGE\s+(\d+)')


def _resolve_pdf_path(title: str) -> str | None:
    """Find a local PDF matching the grounding chunk title."""
    if not title or not PDF_DIR.exists():
        return None
    candidate = PDF_DIR / title
    if candidate.exists() and title.endswith(".pdf"):
        return str(candidate)
    base = title.replace("_enriched.txt", ".pdf")
    candidate = PDF_DIR / base
    if candidate.exists():
        return str(candidate)
    for f in PDF_DIR.iterdir():
        if f.suffix.lower() == ".pdf" and f.stem.lower() in title.lower():
            return str(f)
    return None


def _resolve_page_index(title: str) -> dict[str, str] | None:
    """Find and load a page index JSON matching the grounding chunk title.

    Checks local cache, then local filesystem, then GCS.
    """
    if not title:
        return None

    cache_key = title
    if cache_key in _index_cache:
        return _index_cache[cache_key]

    # Try direct load by display name (checks local then GCS)
    index = load_page_index_by_name(title)
    if index:
        _index_cache[cache_key] = index
        return index

    # Fuzzy match on local files
    base = title.replace("_enriched.txt", "").replace(".pdf", "")
    if INDEX_DIR.exists():
        for f in INDEX_DIR.iterdir():
            if f.suffix == ".json" and base.lower() in f.stem.lower():
                index = load_page_index(f)
                _index_cache[cache_key] = index
                return index

    return None


def _find_page_in_index(chunk_text: str, page_index: dict[str, str]) -> int | None:
    """Find page number by matching chunk text against page index."""
    full = normalize(chunk_text)

    for length in (120, 60, 40):
        snippet = full[:length]
        if len(snippet) < 20:
            continue
        for page_num_str, page_text in page_index.items():
            if snippet in page_text:
                return int(page_num_str)

    return None


def find_page(chunk_text: str, pdf_path: str) -> int | None:
    """Find the page number where chunk_text appears in the PDF. Returns 1-indexed page or None."""
    if not chunk_text or not pdf_path:
        return None

    pages = _load_pdf(pdf_path)
    full = normalize(chunk_text)

    for length in (120, 60, 40):
        snippet = full[:length]
        if len(snippet) < 20:
            continue
        for page_num, page_text in enumerate(pages, 1):
            if snippet in page_text:
                return page_num

    # Fallback: check for PAGE N marker in enriched text chunks
    match = _PAGE_MARKER_RE.search(chunk_text)
    if match:
        return int(match.group(1))

    return None


def find_pages_for_chunks(chunks) -> list[int | None]:
    """Given grounding_chunks from Google's response, return page numbers for each.

    Tries in order:
    1. PAGE N marker in chunk text (fastest, from enriched text)
    2. Text matching against local PDF (if available)
    3. Text matching against page index JSON (if available, no PDF needed)
    """
    results = []
    for chunk in chunks:
        ctx = getattr(chunk, "retrieved_context", None)
        if not ctx:
            results.append(None)
            continue
        title = ctx.title if ctx.title else ""
        text = ctx.text if hasattr(ctx, "text") else ""

        # 1. Try PAGE marker first (cheapest)
        match = _PAGE_MARKER_RE.search(text)
        if match:
            results.append(int(match.group(1)))
            continue

        # 2. Try local PDF
        pdf_path = _resolve_pdf_path(title)
        if pdf_path:
            page = find_page(text, pdf_path)
            if page:
                results.append(page)
                continue

        # 3. Try page index (no PDF needed)
        page_index = _resolve_page_index(title)
        if page_index:
            page = _find_page_in_index(text, page_index)
            if page:
                results.append(page)
                continue

        results.append(None)
    return results
