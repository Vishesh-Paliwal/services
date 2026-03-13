import os
from dotenv import load_dotenv
from google import genai
from portkey_ai import Portkey

load_dotenv()

MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_TOP_K = int(os.environ.get("FILE_SEARCH_TOP_K", "10"))

PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")
PORTKEY_MODEL = os.environ.get("PORTKEY_MODEL", "@gcp-prod-gemini-key/gemini-2.5-flash")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def get_client() -> genai.Client:
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY is required")
    return genai.Client(api_key=key)


def get_portkey_client(api_key: str | None = None) -> Portkey:
    key = api_key or PORTKEY_API_KEY
    if not key:
        raise ValueError("PORTKEY_API_KEY is required")
    return Portkey(api_key=key)
