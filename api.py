"""
FastAPI backend for Bioreactor RAG.

Wraps query_engine, query_enhancer, store_manager, citations, and pdf_enrichment
into REST endpoints.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_client, DEFAULT_TOP_K, MODEL
from store_manager import (
    create_store, list_stores, delete_store,
    upload_book_from_bytes, upload_enriched_text, list_documents,
)
from query_engine import query, query_smart, query_rewrite
from citations import format_citations, format_references_markdown
from page_finder import find_pages_for_chunks
from page_index import save_page_index_from_bytes
from pdf_enrichment.enricher import enrich_pdf_from_bytes
from response_logger import log_response

logger = logging.getLogger(__name__)

# --- Shared client ---
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = get_client()
    return _client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: init client
    _get_client()
    logger.info("API started, model=%s, default_top_k=%d", MODEL, DEFAULT_TOP_K)
    yield


app = FastAPI(title="Bioreactor RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response models ───


class QueryRequest(BaseModel):
    question: str
    store_name: str
    mode: str = "augmented"
    top_k: int = DEFAULT_TOP_K
    smart: bool = True


class ReferenceItem(BaseModel):
    index: int
    title: str
    uri: str
    page_info: str | None = None


class QueryResponse(BaseModel):
    answer: str
    cited_answer: str
    references: list[ReferenceItem]
    token_usage: dict
    enhancer_meta: dict | None = None


class StoreCreateRequest(BaseModel):
    display_name: str


class StoreItem(BaseModel):
    name: str
    display_name: str


class DocumentItem(BaseModel):
    name: str
    display_name: str


# ─── Query endpoints ───


@app.post("/api/query", response_model=QueryResponse)
def api_query(req: QueryRequest):
    client = _get_client()
    enhancer_meta = None

    try:
        if req.smart:
            response, enhancer_meta = query_smart(
                client, req.store_name, req.question,
                mode=req.mode, top_k=req.top_k,
            )
        else:
            response = query(
                client, req.store_name, req.question,
                mode=req.mode, top_k=req.top_k,
            )

        log_response(req.question, req.mode, response)

        answer_text = response.text or ""

        # Format citations — failures here must not block the answer
        cited_text = answer_text
        references = []
        try:
            if enhancer_meta and enhancer_meta.get("decomposed"):
                cited_text = answer_text
                references = []
            else:
                cited_text, refs = format_citations(response)
                references = [
                    ReferenceItem(
                        index=r["index"],
                        title=r["title"],
                        uri=r.get("uri", ""),
                        page_info=r.get("page_info"),
                    )
                    for r in refs
                ]
        except Exception:
            logger.exception("Citation formatting failed, returning answer without citations")

        # Token usage
        usage = getattr(response, "usage_metadata", None)
        token_usage = {}
        if usage:
            token_usage = {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "candidates_tokens": getattr(usage, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(usage, "total_token_count", 0) or 0,
            }

        return QueryResponse(
            answer=answer_text,
            cited_answer=cited_text,
            references=references,
            token_usage=token_usage,
            enhancer_meta=enhancer_meta,
        )

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Store endpoints ───


@app.get("/api/stores", response_model=list[StoreItem])
def api_list_stores():
    client = _get_client()
    try:
        stores = list_stores(client)
        return [
            StoreItem(
                name=s.name,
                display_name=getattr(s, "display_name", "Unnamed"),
            )
            for s in stores
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stores", response_model=StoreItem)
def api_create_store(req: StoreCreateRequest):
    client = _get_client()
    try:
        store = create_store(client, req.display_name)
        return StoreItem(
            name=store.name,
            display_name=getattr(store, "display_name", req.display_name),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/stores/{store_name:path}")
def api_delete_store(store_name: str):
    client = _get_client()
    try:
        delete_store(client, store_name)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stores/{store_name:path}/documents", response_model=list[DocumentItem])
def api_list_documents(store_name: str):
    client = _get_client()
    try:
        docs = list_documents(client, store_name)
        return [
            DocumentItem(
                name=doc.name,
                display_name=getattr(doc, "display_name", doc.name),
            )
            for doc in docs
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Upload endpoint ───


@app.post("/api/upload")
def api_upload(
    file: UploadFile = File(...),
    store_name: str = Form(...),
    enrich: str = Form("true"),
):
    """Upload a PDF to a store. Optionally enriches with visual descriptions.
    Also builds a page index for page number lookups."""
    client = _get_client()
    pdf_bytes = file.file.read()
    filename = file.filename or "upload.pdf"
    enrich = enrich.lower() in ("true", "1", "yes")

    try:
        if enrich:
            enriched_text, stats = enrich_pdf_from_bytes(
                client, pdf_bytes, filename,
            )
            # Build page index from enriched text (covers image-only pages)
            index_path = save_page_index_from_bytes(pdf_bytes, filename, enriched_text=enriched_text)
            logger.info("Page index saved (with enriched text): %s", index_path)

            enriched_name = filename.replace(".pdf", "_enriched.txt")
            upload_enriched_text(client, store_name, enriched_text, enriched_name)
            return {
                "status": "uploaded",
                "filename": enriched_name,
                "enriched": True,
                "stats": stats,
                "page_index": str(index_path),
            }
        else:
            # Build page index from raw PDF only
            index_path = save_page_index_from_bytes(pdf_bytes, filename)
            logger.info("Page index saved (PDF only): %s", index_path)

            upload_book_from_bytes(client, store_name, pdf_bytes, filename)
            return {
                "status": "uploaded",
                "filename": filename,
                "enriched": False,
                "page_index": str(index_path),
            }

    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Health ───


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL}
