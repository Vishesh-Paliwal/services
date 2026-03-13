"""
JWT verification and Supabase client for FastAPI.
"""

import logging

from fastapi import Request, HTTPException
from supabase import create_client, Client

from config import SUPABASE_URL, SUPABASE_SERVICE_KEY

logger = logging.getLogger(__name__)

# --- Supabase admin client (service role) ---
_supabase: Client | None = None


def get_supabase() -> Client:
    """Returns a Supabase client using the service role key."""
    global _supabase
    if _supabase is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


def get_current_user(request: Request) -> dict:
    """FastAPI dependency — extracts Bearer token, verifies via Supabase, returns {user_id, email}."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")

    token = auth_header[7:]

    try:
        sb = get_supabase()
        user_response = sb.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return {
            "user_id": user.id,
            "email": user.email,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Token verification failed")
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
