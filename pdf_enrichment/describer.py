import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from google import genai
from google.genai import types

from .prompts import SCIENTIFIC_IMAGE_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 5


def describe_page_image(
    client: genai.Client,
    image_bytes: bytes,
    model: str = "gemini-2.5-flash",
) -> str:
    """Send a page image to Gemini Vision and get a scientific description."""
    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    types.Part(text=SCIENTIFIC_IMAGE_PROMPT),
                ]
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=4096,
        ),
    )
    return response.text


def _describe_one(client, pdf_path, page_num, model):
    """Describe a single page — thread target."""
    from .detector import render_page_as_image
    img_bytes = render_page_as_image(pdf_path, page_num)
    try:
        desc = describe_page_image(client, img_bytes, model)
        if desc is None:
            logger.warning("Page %d: Gemini returned empty response", page_num + 1)
            return page_num, "[Description failed: empty response]"
        logger.info("Page %d described (%d chars)", page_num + 1, len(desc))
        return page_num, desc
    except Exception as e:
        logger.warning("Page %d description failed: %s", page_num + 1, e)
        return page_num, f"[Description failed: {e}]"


def describe_pages_batch(
    client: genai.Client,
    pdf_path: str,
    page_nums: list[int],
    model: str = "gemini-2.5-flash",
    progress_callback=None,
    max_workers: int = DEFAULT_WORKERS,
) -> dict[int, str]:
    """Describe multiple pages in parallel. Returns {page_num: description}."""
    logger.info("Describing %d pages with %d workers", len(page_nums), max_workers)
    descriptions = {}
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_describe_one, client, pdf_path, pn, model): pn
            for pn in page_nums
        }
        for future in as_completed(futures):
            page_num, desc = future.result()
            descriptions[page_num] = desc
            completed += 1
            if progress_callback:
                progress_callback(completed, len(page_nums), page_num)

    return descriptions
