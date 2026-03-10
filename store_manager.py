import os
import time
import tempfile
from google import genai


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
    display = display_name or os.path.basename(file_path)
    operation = client.file_search_stores.upload_to_file_search_store(
        file=file_path,
        file_search_store_name=store_name,
        config={"display_name": display},
    )
    while not operation.done:
        time.sleep(2)
        operation = client.operations.get(operation)
    return operation


def upload_book_from_bytes(client: genai.Client, store_name: str, file_bytes: bytes, file_name: str):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return upload_book(client, store_name, tmp_path, display_name=file_name)
    finally:
        os.unlink(tmp_path)


def upload_enriched_text(client: genai.Client, store_name: str, text: str, display_name: str):
    """Upload enriched text content as a .txt file to the file_search_store."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name
    try:
        return upload_book(client, store_name, tmp_path, display_name=display_name)
    finally:
        os.unlink(tmp_path)


def list_documents(client: genai.Client, store_name: str):
    docs = list(client.file_search_stores.documents.list(parent=store_name))
    return docs
