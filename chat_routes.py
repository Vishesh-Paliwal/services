"""
Conversation and message CRUD routes.
All endpoints require authentication via get_current_user.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from auth import get_current_user, get_supabase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["chat"])


# --- Request/Response models ---

class ConversationCreate(BaseModel):
    store_name: str
    title: str = "New conversation"


class ConversationUpdate(BaseModel):
    title: str


class MessageCreate(BaseModel):
    role: str
    content: str
    metadata: dict | None = None


# --- Conversation endpoints ---

@router.get("")
def list_conversations(store_name: str, user: dict = Depends(get_current_user)):
    sb = get_supabase()
    result = (
        sb.table("conversations")
        .select("*")
        .eq("user_id", user["user_id"])
        .eq("store_name", store_name)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data


@router.post("")
def create_conversation(req: ConversationCreate, user: dict = Depends(get_current_user)):
    sb = get_supabase()
    result = (
        sb.table("conversations")
        .insert({
            "user_id": user["user_id"],
            "store_name": req.store_name,
            "title": req.title,
        })
        .execute()
    )
    return result.data[0]


@router.get("/{conversation_id}/messages")
def get_messages(conversation_id: str, user: dict = Depends(get_current_user)):
    sb = get_supabase()
    # Verify ownership
    conv = (
        sb.table("conversations")
        .select("id")
        .eq("id", conversation_id)
        .eq("user_id", user["user_id"])
        .execute()
    )
    if not conv.data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = (
        sb.table("messages")
        .select("*")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
        .execute()
    )
    return result.data


@router.post("/{conversation_id}/messages")
def add_message(conversation_id: str, req: MessageCreate, user: dict = Depends(get_current_user)):
    sb = get_supabase()
    # Verify ownership
    conv = (
        sb.table("conversations")
        .select("id")
        .eq("id", conversation_id)
        .eq("user_id", user["user_id"])
        .execute()
    )
    if not conv.data:
        raise HTTPException(status_code=404, detail="Conversation not found")

    result = (
        sb.table("messages")
        .insert({
            "conversation_id": conversation_id,
            "role": req.role,
            "content": req.content,
            "metadata": req.metadata or {},
        })
        .execute()
    )

    # Update conversation timestamp
    sb.table("conversations").update({"updated_at": "now()"}).eq("id", conversation_id).execute()

    return result.data[0]


@router.patch("/{conversation_id}")
def update_conversation(conversation_id: str, req: ConversationUpdate, user: dict = Depends(get_current_user)):
    sb = get_supabase()
    result = (
        sb.table("conversations")
        .update({"title": req.title, "updated_at": "now()"})
        .eq("id", conversation_id)
        .eq("user_id", user["user_id"])
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return result.data[0]


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, user: dict = Depends(get_current_user)):
    sb = get_supabase()
    result = (
        sb.table("conversations")
        .delete()
        .eq("id", conversation_id)
        .eq("user_id", user["user_id"])
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted"}
