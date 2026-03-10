import json
import os
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"


def _serialize(obj):
    """Recursively convert API response objects to dicts for JSON serialization."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    # For API objects, try to_dict() first, then __dict__, then str()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    return str(obj)


def log_response(question: str, mode: str, response):
    """Log the full API response with metadata to a JSON file."""
    LOG_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().isoformat()
    filename = datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"

    candidate = response.candidates[0] if response.candidates else None

    # Extract token usage as a clean top-level summary
    usage = getattr(response, "usage_metadata", None)
    token_usage = {}
    if usage:
        token_usage = {
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "candidates_tokens": getattr(usage, "candidates_token_count", None),
            "thoughts_tokens": getattr(usage, "thoughts_token_count", None),
            "tool_use_prompt_tokens": getattr(usage, "tool_use_prompt_token_count", None),
            "cached_tokens": getattr(usage, "cached_content_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }

    log_entry = {
        "timestamp": timestamp,
        "question": question,
        "mode": mode,
        "token_usage": token_usage,
        "response_text": response.text,
        "candidates": _serialize(response.candidates),
        "model_version": getattr(response, "model_version", None),
        "usage_metadata": _serialize(usage),
    }

    # Extract grounding metadata separately for easy inspection
    if candidate:
        grounding = getattr(candidate, "grounding_metadata", None)
        if grounding:
            log_entry["grounding_metadata"] = {
                "grounding_chunks": _serialize(getattr(grounding, "grounding_chunks", None)),
                "grounding_supports": _serialize(getattr(grounding, "grounding_supports", None)),
                "search_entry_point": _serialize(getattr(grounding, "search_entry_point", None)),
                "retrieval_metadata": _serialize(getattr(grounding, "retrieval_metadata", None)),
                "web_search_queries": _serialize(getattr(grounding, "web_search_queries", None)),
            }

    log_path = LOG_DIR / filename
    with open(log_path, "w") as f:
        json.dump(log_entry, f, indent=2, default=str)

    return log_path
