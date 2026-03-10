import logging
from dataclasses import dataclass
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class PageAnalysis:
    page_num: int  # 0-indexed
    has_images: bool
    image_count: int
    has_drawings: bool
    has_tables: bool
    image_area_ratio: float


def analyze_page(page: fitz.Page) -> PageAnalysis:
    """Analyze a single page for visual content using PyMuPDF heuristics."""
    images = page.get_images(full=True)

    # Detect vector drawings (charts, diagrams)
    drawings = page.get_drawings()
    significant_drawings = [d for d in drawings if len(d.get("items", [])) > 2]

    # Table detection: count horizontal/vertical lines
    h_lines = 0
    v_lines = 0
    for d in drawings:
        for item in d.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 2:
                    h_lines += 1
                elif abs(p1.x - p2.x) < 2:
                    v_lines += 1
    has_tables = h_lines >= 4 and v_lines >= 4

    # Image area ratio
    page_area = page.rect.width * page.rect.height
    image_area = 0.0
    for img in images:
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
            for r in rects:
                image_area += r.width * r.height
        except Exception:
            pass
    ratio = image_area / page_area if page_area > 0 else 0.0

    return PageAnalysis(
        page_num=page.number,
        has_images=len(images) > 0,
        image_count=len(images),
        has_drawings=len(significant_drawings) > 10,
        has_tables=has_tables,
        image_area_ratio=ratio,
    )


def detect_visual_pages(pdf_path: str, min_image_area_ratio: float = 0.05) -> list[PageAnalysis]:
    """Return PageAnalysis for pages that have visual content worth describing."""
    logger.info("Scanning %s for visual content", pdf_path)
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    visual_pages = []
    for page in doc:
        analysis = analyze_page(page)
        if (analysis.has_images and analysis.image_area_ratio > min_image_area_ratio) \
           or analysis.has_drawings \
           or analysis.has_tables:
            visual_pages.append(analysis)
    doc.close()
    logger.info("Found %d visual pages out of %d total", len(visual_pages), total_pages)
    return visual_pages


def render_page_as_image(pdf_path: str, page_num: int, dpi: int = 200) -> bytes:
    """Render a single page as PNG bytes for Gemini Vision."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")
    doc.close()
    return img_bytes
