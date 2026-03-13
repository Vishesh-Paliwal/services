# Bioreactor RAG Backend - GCP Cloud Run Deployment Guide

## Overview

Deployment of the **Bioreactor RAG Backend** (FastAPI + Gemini File Search) to **Google Cloud Run** using Buildpacks (no Docker required).

**Service URL:** `https://bioreactor-rag-746208330214.us-central1.run.app`
**Region:** `us-central1` (required for GCP free tier)
**Project:** `project-688a4c78-5d5b-45b3-b5d`

---

## Architecture

- **FastAPI** backend with Gemini File Search RAG
- **Google Cloud Run** for serverless hosting
- **GCS Bucket** for persistent page index storage (survives container restarts)
- **Secret Manager** for API keys (GOOGLE_API_KEY, PORTKEY_API_KEY)
- **Cloud Build + Buildpacks** auto-builds from source (no Dockerfile needed)

---

## Prerequisites

- `gcloud` CLI installed (`brew install google-cloud-sdk`)
- GCP project access with **Editor** role
- The default compute service account needs **Editor** role at project level

---

## One-Time Setup Steps

### Step 1: Auth & Project Setup

```bash
gcloud auth login
gcloud config set project project-688a4c78-5d5b-45b3-b5d
```

### Step 2: Enable Required APIs

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com storage.googleapis.com
```

### Step 3: Create GCS Bucket for Page Indexes

```bash
gcloud storage buckets create gs://project-688a4c78-5d5b-45b3-b5d-page-indexes --location=us-central1
```

This bucket stores page index JSON files so citations with page numbers persist across container restarts.

### Step 4: Store API Keys in Secret Manager

```bash
echo -n "YOUR_GOOGLE_API_KEY" | gcloud secrets create GOOGLE_API_KEY --data-file=-
echo -n "YOUR_PORTKEY_API_KEY" | gcloud secrets create PORTKEY_API_KEY --data-file=-
echo -n "https://YOUR_PROJECT.supabase.co" | gcloud secrets create SUPABASE_URL --data-file=-
echo -n "YOUR_SUPABASE_SERVICE_ROLE_KEY" | gcloud secrets create SUPABASE_SERVICE_KEY --data-file=-
```

Replace the placeholder values with actual keys.

### Step 5: Grant IAM Permissions

Grant the compute service account access to secrets and GCS:

```bash
gcloud secrets add-iam-policy-binding GOOGLE_API_KEY --member="serviceAccount:746208330214-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

```bash
gcloud secrets add-iam-policy-binding PORTKEY_API_KEY --member="serviceAccount:746208330214-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

```bash
gcloud secrets add-iam-policy-binding SUPABASE_URL --member="serviceAccount:746208330214-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

```bash
gcloud secrets add-iam-policy-binding SUPABASE_SERVICE_KEY --member="serviceAccount:746208330214-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

```bash
gcloud storage buckets add-iam-policy-binding gs://project-688a4c78-5d5b-45b3-b5d-page-indexes --member="serviceAccount:746208330214-compute@developer.gserviceaccount.com" --role="roles/storage.objectUser"
```

---

## Deploy Command

This is the only command needed for deploying or redeploying:

```bash
gcloud run deploy bioreactor-rag --source . --region us-central1 --allow-unauthenticated --memory 2Gi --timeout 900 --concurrency 2 --set-env-vars "GEMINI_MODEL=gemini-2.5-flash,FILE_SEARCH_TOP_K=10,GCS_PAGE_INDEX_BUCKET=project-688a4c78-5d5b-45b3-b5d-page-indexes,GCS_UPLOAD_BUCKET=project-688a4c78-5d5b-45b3-b5d-pdf-uploads" --set-secrets "GOOGLE_API_KEY=GOOGLE_API_KEY:latest,PORTKEY_API_KEY=PORTKEY_API_KEY:latest,SUPABASE_URL=SUPABASE_URL:latest,SUPABASE_SERVICE_KEY=SUPABASE_SERVICE_KEY:latest"
```

Run this from the `bioreactor-rag-backend` directory. Cloud Build auto-detects Python via `Procfile` + `requirements.txt`.

---

## Verify Deployment

```bash
curl https://bioreactor-rag-746208330214.us-central1.run.app/api/health
```

Expected: `{"status":"ok","model":"gemini-2.5-flash"}`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Health check |
| `/api/query` | POST | Query the RAG system |
| `/api/stores` | GET | List file search stores |
| `/api/stores` | POST | Create a new store |
| `/api/stores/{name}` | DELETE | Delete a store |
| `/api/stores/{name}/documents` | GET | List documents in a store |
| `/api/upload` | POST | Upload PDF with optional enrichment |

---

## Environment Variables

| Variable | Source | Description |
|---|---|---|
| `GEMINI_MODEL` | env var | Model name (default: `gemini-2.5-flash`) |
| `FILE_SEARCH_TOP_K` | env var | Chunks to retrieve (default: `10`) |
| `GCS_PAGE_INDEX_BUCKET` | env var | GCS bucket for page indexes |
| `GCS_PAGE_INDEX_PREFIX` | env var | GCS prefix (default: `page_indexes/`) |
| `GOOGLE_API_KEY` | Secret Manager | Gemini API key |
| `PORTKEY_API_KEY` | Secret Manager | Portkey API key |
| `SUPABASE_URL` | Secret Manager | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | Secret Manager | Supabase service role key |

---

## How Page Indexes Work on Cloud Run

1. **Upload time:** When a PDF is uploaded, a page index JSON is built (maps page numbers to normalized text) and saved locally + uploaded to GCS
2. **Query time:** When citations need page numbers, the system checks local cache first, then fetches from GCS if not found locally
3. **Resilience:** If page index lookup fails entirely, the query still returns an answer — just without page numbers in citations

---

## Free Tier Notes

GCP free tier only applies to **us-central1, us-east1, us-west1** regions:

- **Cloud Run:** 2M requests/month, 360K GB-seconds memory, 180K vCPU-seconds
- **Cloud Storage:** 5GB storage, 1GB egress/month
- **Secret Manager:** 6 active secret versions, 10K access operations/month

Using `asia-south1` (Mumbai) would be billed from the first request.

---

## Updating Secrets

To rotate an API key:

```bash
echo -n "NEW_KEY_VALUE" | gcloud secrets versions add GOOGLE_API_KEY --data-file=-
```

Then redeploy to pick up the new version.

---

## Troubleshooting

**Permission errors during deploy:** Ensure the compute service account (`746208330214-compute@developer.gserviceaccount.com`) has Editor role at project level. Go to IAM & Admin > IAM, find the account, and add Editor role.

**Citations missing page numbers:** Check that `GCS_PAGE_INDEX_BUCKET` env var is set and the page index was uploaded during PDF upload.

**View logs:**

```bash
gcloud run services logs read bioreactor-rag --region us-central1
```
