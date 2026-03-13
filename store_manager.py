import logging
import os
import time
import tempfile
from google import genai

logger = logging.getLogger(__name__)


def create_store(client: genai.Client, display_name: str):
    store = client.file_search_stores.create(
        config={"display_name": display_name}
    )
    return store


def list_stores(client: genai.Client):
    stores = list(client.file_search_stores.list())
    return stores


def get_store_info(client: genai.Client, store_name: str):
    return client.file_search_stores.get(name=store_name)


def delete_store(client: genai.Client, store_name: str, force: bool = True):
    client.file_search_stores.delete(name=store_name, force=force)


def upload_book(client: genai.Client, store_name: str, file_path: str, display_name: str | None = None):
    """Two-step upload: files.upload() then import_file().

    Decouples file transfer from indexing to avoid the known 503
    'Failed to count tokens' bug in upload_to_file_search_store().
    """
    display = display_name or os.path.basename(file_path)

    # Step 1: Upload file to Files API (reliable, completes in seconds)
    uploaded_file = client.files.upload(
        file=file_path,
        config={"display_name": display},
    )
    logger.info("File uploaded to Files API: %s (%s)", uploaded_file.name, display)

    # Step 2: Import into File Search Store (triggers indexing)
    operation = client.file_search_stores.import_file(
        file_search_store_name=store_name,
        file_name=uploaded_file.name,
    )
    while not operation.done:
        time.sleep(2)
        operation = client.operations.get(operation)
    logger.info("File indexed in store: %s", display)
    return operation


def upload_book_from_bytes(client: genai.Client, store_name: str, file_bytes: bytes, file_name: str):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return upload_book(client, store_name, tmp_path, display_name=file_name)
    finally:
        os.unlink(tmp_path)


def _text_to_pdf(text: str) -> bytes:
    """Convert text to a PDF using reportlab. Handles Unicode natively."""
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch)

    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontSize = 8
    style.leading = 10

    story = []
    for line in text.split("\n"):
        if line.strip():
            # Escape XML special chars for reportlab's Paragraph
            safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(safe, style))
        else:
            story.append(Spacer(1, 6))

    doc.build(story)
    return buf.getvalue()


def upload_enriched_text(client: genai.Client, store_name: str, text: str, display_name: str, max_retries: int = 3):
    """Convert enriched text to PDF and upload to the file_search_store.

    PDFs upload reliably — .txt files hit the 503 'Failed to count tokens' bug.
    """
    logger.info("Enriched text: %s (%d chars, %.2f MB)", display_name, len(text), len(text.encode("utf-8")) / (1024 * 1024))

    pdf_name = display_name.replace(".txt", ".pdf")
    logger.info("Converting enriched text to PDF: %s", pdf_name)
    pdf_bytes = _text_to_pdf(text)
    logger.info("PDF generated: %.2f MB", len(pdf_bytes) / (1024 * 1024))

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        for attempt in range(max_retries):
            try:
                return upload_book(client, store_name, tmp_path, display_name=pdf_name)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)
                    logger.warning("Upload attempt %d/%d failed (%s), retrying in %ds...", attempt + 1, max_retries, e, wait)
                    time.sleep(wait)
                else:
                    raise
    finally:
        os.unlink(tmp_path)


def list_documents(client: genai.Client, store_name: str):
    docs = list(client.file_search_stores.documents.list(parent=store_name))
    return docs
