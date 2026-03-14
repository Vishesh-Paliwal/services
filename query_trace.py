"""
Query trace logger — captures the full pipeline for a single query in one JSON file.

Usage in api.py:
    trace = QueryTrace(user_id, original_question)
    trace.set("conversation_messages", messages)
    trace.set("supermemory_search", results)
    trace.set("resolved_question", resolved)
    ...
    trace.save()
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent / "logs" / "traces"
GCS_TRACE_BUCKET = os.environ.get("GCS_TRACE_BUCKET", "")
GCS_TRACE_PREFIX = os.environ.get("GCS_TRACE_PREFIX", "traces/")


class QueryTrace:
    def __init__(self, user_id: str, original_question: str):
        self.start_time = time.time()
        self.data = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "original_question": original_question,
            "steps": {},
            "timings": {},
        }
        self._step_starts: dict[str, float] = {}

    def start_step(self, step: str):
        """Mark the start of a pipeline step for timing."""
        self._step_starts[step] = time.time()

    def end_step(self, step: str):
        """Mark the end of a pipeline step and record duration."""
        if step in self._step_starts:
            self.data["timings"][step] = round(
                (time.time() - self._step_starts[step]) * 1000
            )

    def set(self, key: str, value):
        """Set a value in the trace. Use for intermediate results."""
        self.data["steps"][key] = value

    def step(self, step_name: str, value):
        """Record a step result and its timing in one call (if start_step was called)."""
        self.data["steps"][step_name] = value
        self.end_step(step_name)

    def save(self) -> str | None:
        """Save trace to local disk + GCS (if configured). Returns GCS path or local path."""
        self.data["total_ms"] = round((time.time() - self.start_time) * 1000)

        trace_json = json.dumps(self.data, indent=2, default=str, ensure_ascii=False)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"

        # Always save locally
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        local_path = LOG_DIR / filename
        try:
            with open(local_path, "w") as f:
                f.write(trace_json)
        except Exception:
            pass

        # Save to GCS for persistence
        if GCS_TRACE_BUCKET:
            try:
                client = storage.Client()
                bucket = client.bucket(GCS_TRACE_BUCKET)
                blob = bucket.blob(f"{GCS_TRACE_PREFIX}{filename}")
                blob.upload_from_string(trace_json, content_type="application/json")
                gcs_path = f"gs://{GCS_TRACE_BUCKET}/{GCS_TRACE_PREFIX}{filename}"
                logger.info("Trace saved to %s", gcs_path)
                return gcs_path
            except Exception as e:
                logger.warning("Failed to save trace to GCS: %s", e)

        return str(local_path)
