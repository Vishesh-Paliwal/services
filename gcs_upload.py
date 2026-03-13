"""
GCS upload helpers: resumable upload URL generation and file download.

Uses GCS resumable upload session URIs to bypass Cloud Run's 32MB request
size limit. The backend initiates the upload and returns a session URI that
the frontend can PUT to directly — no signing or extra IAM permissions needed.
"""

import logging
import os
import uuid
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)

GCS_UPLOAD_BUCKET = os.environ.get("GCS_UPLOAD_BUCKET", "")
GCS_UPLOAD_PREFIX = os.environ.get("GCS_UPLOAD_PREFIX", "uploads/")


def _get_storage_client():
    return storage.Client()


def generate_upload_url(filename: str, origin: str = "") -> dict:
    """Initiate a GCS resumable upload and return the session URI.

    The session URI acts as an auth token — the frontend can PUT directly
    to it without any credentials. No IAM signBlob permission needed.
    Expires after 1 week.
    """
    if not GCS_UPLOAD_BUCKET:
        raise ValueError("GCS_UPLOAD_BUCKET is not configured")

    client = _get_storage_client()
    bucket = client.bucket(GCS_UPLOAD_BUCKET)

    # Unique path to avoid collisions
    unique_id = uuid.uuid4().hex[:12]
    safe_name = filename.replace(" ", "_")
    gcs_path = f"{GCS_UPLOAD_PREFIX}{unique_id}_{safe_name}"

    blob = bucket.blob(gcs_path)

    # Initiate resumable upload — origin required so GCS includes CORS headers
    upload_url = blob.create_resumable_upload_session(
        content_type="application/pdf",
        origin=origin,
    )

    logger.info("Created resumable upload session for %s → gs://%s/%s", filename, GCS_UPLOAD_BUCKET, gcs_path)

    return {
        "upload_url": upload_url,
        "gcs_path": gcs_path,
        "bucket": GCS_UPLOAD_BUCKET,
    }


def download_from_gcs(gcs_path: str, local_path: str | None = None) -> str:
    """Download a file from GCS to a local temp path. Returns local path."""
    if not GCS_UPLOAD_BUCKET:
        raise ValueError("GCS_UPLOAD_BUCKET is not configured")

    if local_path is None:
        suffix = Path(gcs_path).suffix or ".pdf"
        local_path = f"/tmp/{uuid.uuid4().hex[:12]}{suffix}"

    client = _get_storage_client()
    bucket = client.bucket(GCS_UPLOAD_BUCKET)
    blob = bucket.blob(gcs_path)

    blob.download_to_filename(local_path)
    file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
    logger.info("Downloaded gs://%s/%s → %s (%.1f MB)", GCS_UPLOAD_BUCKET, gcs_path, local_path, file_size_mb)

    return local_path


GCS_ENRICHMENT_PREFIX = "enrichment_cache/"


def save_enrichment_to_gcs(filename: str, text: str) -> str:
    """Cache enriched text to GCS so it survives upload failures."""
    if not GCS_UPLOAD_BUCKET:
        return ""
    safe_name = filename.replace(" ", "_").replace(".pdf", "_enriched.txt")
    gcs_path = f"{GCS_ENRICHMENT_PREFIX}{safe_name}"
    client = _get_storage_client()
    bucket = client.bucket(GCS_UPLOAD_BUCKET)
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
    logger.info("Cached enrichment to gs://%s/%s (%.2f MB)", GCS_UPLOAD_BUCKET, gcs_path, len(text.encode("utf-8")) / (1024 * 1024))
    return gcs_path


def load_enrichment_from_gcs(filename: str) -> str | None:
    """Load cached enrichment from GCS. Returns None if not found."""
    if not GCS_UPLOAD_BUCKET:
        return None
    safe_name = filename.replace(" ", "_").replace(".pdf", "_enriched.txt")
    gcs_path = f"{GCS_ENRICHMENT_PREFIX}{safe_name}"
    try:
        client = _get_storage_client()
        bucket = client.bucket(GCS_UPLOAD_BUCKET)
        blob = bucket.blob(gcs_path)
        text = blob.download_as_text(encoding="utf-8")
        logger.info("Loaded cached enrichment from gs://%s/%s (%.2f MB)", GCS_UPLOAD_BUCKET, gcs_path, len(text.encode("utf-8")) / (1024 * 1024))
        return text
    except Exception:
        return None


def delete_enrichment_from_gcs(filename: str) -> None:
    """Clean up cached enrichment after successful upload."""
    if not GCS_UPLOAD_BUCKET:
        return
    safe_name = filename.replace(" ", "_").replace(".pdf", "_enriched.txt")
    gcs_path = f"{GCS_ENRICHMENT_PREFIX}{safe_name}"
    try:
        client = _get_storage_client()
        bucket = client.bucket(GCS_UPLOAD_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.delete()
        logger.info("Deleted enrichment cache gs://%s/%s", GCS_UPLOAD_BUCKET, gcs_path)
    except Exception:
        pass


def delete_from_gcs(gcs_path: str) -> None:
    """Delete a file from GCS (cleanup after processing)."""
    if not GCS_UPLOAD_BUCKET:
        return
    try:
        client = _get_storage_client()
        bucket = client.bucket(GCS_UPLOAD_BUCKET)
        blob = bucket.blob(gcs_path)
        blob.delete()
        logger.info("Deleted gs://%s/%s", GCS_UPLOAD_BUCKET, gcs_path)
    except Exception as e:
        logger.warning("GCS cleanup failed (non-fatal): %s", e)
