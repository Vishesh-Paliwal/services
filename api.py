"""
FastAPI backend for Bioreactor RAG.

Wraps query_engine, query_enhancer, store_manager, citations, and pdf_enrichment
into REST endpoints.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import get_client, DEFAULT_TOP_K, MODEL
from auth import get_current_user
from chat_routes import router as chat_router
from feedback_routes import router as feedback_router
from store_manager import (
    create_store, list_stores, delete_store,
    upload_book_from_bytes, upload_enriched_text, list_documents,
)
from query_engine import query, query_smart, query_rewrite
from citations import format_citations, format_references_markdown
from page_finder import find_pages_for_chunks
from page_index import save_page_index_from_bytes, save_page_index
from pdf_enrichment.enricher import enrich_pdf_from_bytes, enrich_pdf_from_local
from gcs_upload import generate_upload_url, download_from_gcs, delete_from_gcs, save_enrichment_to_gcs, load_enrichment_from_gcs, delete_enrichment_from_gcs
from response_logger import log_response
from memory import resolve_question, get_user_profile, store_conversation
from auth import get_supabase
from query_trace import QueryTrace
from followups import generate_follow_ups

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
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

app.include_router(chat_router)
app.include_router(feedback_router)

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
    conversation_id: str | None = None


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
    follow_ups: list[str] = []


class StoreCreateRequest(BaseModel):
    display_name: str


class StoreItem(BaseModel):
    name: str
    display_name: str


class DocumentItem(BaseModel):
    name: str
    display_name: str


class UploadUrlRequest(BaseModel):
    filename: str


class ProcessRequest(BaseModel):
    gcs_path: str
    store_name: str
    filename: str
    enrich: bool = True


# ─── Query endpoints ───


@app.post("/api/query", response_model=QueryResponse)
def api_query(req: QueryRequest, user: dict = Depends(get_current_user)):
    client = _get_client()
    enhancer_meta = None
    trace = QueryTrace(user["user_id"], req.question)

    try:
        # Step 1: Fetch conversation history from Supabase for follow-up resolution
        trace.start_step("fetch_conversation")
        conversation_messages = []
        if req.conversation_id:
            try:
                sb = get_supabase()
                result = (
                    sb.table("messages")
                    .select("role, content")
                    .eq("conversation_id", req.conversation_id)
                    .order("created_at", desc=False)
                    .execute()
                )
                conversation_messages = result.data or []
            except Exception:
                logger.warning("Failed to fetch conversation history")
        trace.step("fetch_conversation", {
            "conversation_id": req.conversation_id,
            "message_count": len(conversation_messages),
            "messages": conversation_messages,
        })

        # Step 2: Resolve follow-up into standalone question
        trace.start_step("resolve_question")
        resolved_question = resolve_question(
            client, req.question, conversation_messages,
            user_id=user["user_id"],
        )
        trace.step("resolve_question", {
            "original": req.question,
            "resolved": resolved_question,
            "changed": resolved_question != req.question,
        })

        # Step 3: Get user profile for personalization
        trace.start_step("get_user_profile")
        user_profile = get_user_profile(user["user_id"])
        trace.step("get_user_profile", {
            "profile_length": len(user_profile),
            "profile": user_profile,
        })

        # Step 4: Build last_exchange from conversation history (last Q&A pair)
        last_exchange = ""
        if conversation_messages:
            # Find the last user+assistant pair
            last_user = ""
            last_assistant = ""
            for msg in reversed(conversation_messages):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "assistant" and not last_assistant:
                    last_assistant = content
                elif role == "user" and not last_user:
                    last_user = content
                if last_user and last_assistant:
                    break
            if last_user and last_assistant:
                last_exchange = f"User: {last_user}\n\nAssistant: {last_assistant}"

        # Step 5: Get past memories from Supermemory for cross-session context
        trace.start_step("search_memories")
        from memory import _search_memories
        past_memories = _search_memories(user["user_id"], resolved_question)
        trace.step("search_memories", {
            "past_memories": past_memories,
        })

        trace.set("last_exchange", last_exchange)

        # Step 6: Query with resolved question + user profile + last exchange + past memories
        trace.start_step("rag_query")
        trace.set("query_params", {
            "mode": req.mode,
            "smart": req.smart,
            "top_k": req.top_k,
            "store_name": req.store_name,
        })
        if req.smart:
            response, enhancer_meta = query_smart(
                client, req.store_name, resolved_question,
                mode=req.mode, top_k=req.top_k,
                user_profile=user_profile,
                last_exchange=last_exchange,
                past_memories=past_memories,
            )
        else:
            response = query(
                client, req.store_name, resolved_question,
                mode=req.mode, top_k=req.top_k,
                user_profile=user_profile,
                last_exchange=last_exchange,
                past_memories=past_memories,
            )
        trace.end_step("rag_query")

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

        # Generate follow-up suggestions
        follow_ups = []
        try:
            follow_ups = generate_follow_ups(client, req.question, answer_text)
        except Exception:
            logger.warning("Follow-up generation failed")

        # Store conversation in Supermemory for future context (fire-and-forget)
        try:
            store_conversation(user["user_id"], req.question, answer_text)
        except Exception:
            logger.warning("Failed to store conversation in Supermemory")

        # Save trace
        trace.set("enhancer_meta", enhancer_meta)
        trace.set("answer", answer_text)
        trace.set("cited_answer", cited_text)
        trace.set("references", [r.model_dump() for r in references] if references else [])
        trace.set("token_usage", token_usage)
        trace.set("follow_ups", follow_ups)
        trace.save()

        return QueryResponse(
            answer=answer_text,
            cited_answer=cited_text,
            references=references,
            token_usage=token_usage,
            enhancer_meta=enhancer_meta,
            follow_ups=follow_ups,
        )

    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


# ─── Store endpoints ───


@app.get("/api/stores", response_model=list[StoreItem])
def api_list_stores(user: dict = Depends(get_current_user)):
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
def api_create_store(req: StoreCreateRequest, user: dict = Depends(get_current_user)):
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
def api_delete_store(store_name: str, user: dict = Depends(get_current_user)):
    client = _get_client()
    try:
        delete_store(client, store_name)
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stores/{store_name:path}/documents", response_model=list[DocumentItem])
def api_list_documents(store_name: str, user: dict = Depends(get_current_user)):
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


# ─── GCS Upload Flow (for large files) ───


@app.post("/api/get-upload-url")
def api_get_upload_url(req: UploadUrlRequest, request: Request, user: dict = Depends(get_current_user)):
    """Generate a resumable GCS upload URL for the frontend."""
    try:
        origin = request.headers.get("origin", "")
        result = generate_upload_url(req.filename, origin=origin)
        return result
    except Exception as e:
        logger.exception("Failed to generate upload URL")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process")
def api_process(req: ProcessRequest, user: dict = Depends(get_current_user)):
    """Process a PDF already uploaded to GCS. Enriches and indexes it."""
    client = _get_client()
    filename = req.filename

    # Download once from GCS, reuse local file for all steps
    local_path = download_from_gcs(req.gcs_path)
    try:
        if req.enrich:
            # Check for cached enrichment from a previous failed upload attempt
            cached = load_enrichment_from_gcs(filename)
            if cached:
                logger.info("Using cached enrichment for %s", filename)
                enriched_text = cached
                stats = {"cached": True}
            else:
                enriched_text, stats = enrich_pdf_from_local(
                    client, local_path, filename,
                )
                # Cache to GCS before attempting upload — survives failures
                save_enrichment_to_gcs(filename, enriched_text)

            index_path = save_page_index(local_path, filename, enriched_text=enriched_text)
            logger.info("Page index saved (with enriched text): %s", index_path)

            enriched_name = filename.replace(".pdf", "_enriched.txt")
            fresh_client = get_client()
            upload_enriched_text(fresh_client, req.store_name, enriched_text, enriched_name)

            # Upload succeeded — clean up cache
            delete_enrichment_from_gcs(filename)
            return {
                "status": "uploaded",
                "filename": enriched_name,
                "enriched": True,
                "stats": stats,
                "page_index": str(index_path),
            }
        else:
            index_path = save_page_index(local_path, filename)
            logger.info("Page index saved (PDF only): %s", index_path)

            from store_manager import upload_book
            upload_book(client, req.store_name, local_path, display_name=filename)
            return {
                "status": "uploaded",
                "filename": filename,
                "enriched": False,
                "page_index": str(index_path),
            }

    except Exception as e:
        logger.exception("Processing failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up local temp file and GCS upload
        if os.path.exists(local_path):
            os.unlink(local_path)
        delete_from_gcs(req.gcs_path)


# ─── Upload endpoint (legacy, for small files < 32MB) ───


@app.post("/api/upload")
def api_upload(
    file: UploadFile = File(...),
    store_name: str = Form(...),
    enrich: str = Form("true"),
    user: dict = Depends(get_current_user),
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
