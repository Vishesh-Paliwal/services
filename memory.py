"""
Supermemory integration — resolve follow-ups, retrieve user profile, store conversations.

Architecture:
1. resolve_question() — uses conversation history (Supabase) + Supermemory search
   to turn follow-up questions into standalone questions
2. get_user_profile() — fetches long-term user profile for personalization
3. store_conversation() — stores Q&A for future memory
"""

import logging
from google import genai
from google.genai import types as genai_types
from supermemory import Supermemory

from config import SUPERMEMORY_API_KEY, MODEL

logger = logging.getLogger(__name__)

_client: Supermemory | None = None


def _get_client() -> Supermemory | None:
    global _client
    if _client is None and SUPERMEMORY_API_KEY:
        _client = Supermemory(api_key=SUPERMEMORY_API_KEY)
    return _client


RESOLVER_PROMPT = """You are a question resolver for a bioprocess engineering assistant.

Your job: take a follow-up question that may reference previous conversation, and rewrite it
as a STANDALONE question that can be understood without any conversation history.

Rules:
- If the question is already standalone (no pronouns, no references to "that", "it", "this",
  "the above", "you mentioned", etc.), return it EXACTLY as-is.
- If it references previous conversation, use the provided history to resolve all references
  into explicit terms.
- Keep the rewritten question concise and natural — don't over-explain.
- Preserve the user's intent exactly. Don't add scope they didn't ask for.
- Output ONLY the resolved question, nothing else. No explanation, no prefix.

Examples:
- History: "Q: What is a bioreactor? A: A bioreactor is a chamber..."
  Follow-up: "What are the types?"
  Output: "What are the types of bioreactors?"

- History: "Q: Compare fermentor vs reactor A: ...optimization approaches differ..."
  Follow-up: "elaborate on optimizations"
  Output: "Elaborate on the optimization approaches that differ between fermentors and reactors"

- Follow-up: "What is the Del factor in sterilization?"
  Output: "What is the Del factor in sterilization?"
"""


def _search_memories(user_id: str, query: str) -> str:
    """Search Supermemory for relevant memories from past sessions."""
    client = _get_client()
    if not client:
        return ""

    try:
        result = client.search.memories(
            q=query,
            container_tag=user_id,
            threshold=0.5,
        )

        # Parse results
        results = None
        if hasattr(result, 'results'):
            results = result.results
        elif isinstance(result, dict):
            results = result.get('results', [])

        if not results:
            return ""

        parts = []
        for r in results[:3]:
            mem = r.memory if hasattr(r, 'memory') else (r.get('memory', '') if isinstance(r, dict) else '')
            if mem:
                parts.append(f"- {mem}")

        if not parts:
            return ""

        return "From past sessions:\n" + "\n".join(parts)

    except Exception as e:
        logger.warning("Supermemory search failed: %s", e)
        return ""


def resolve_question(
    gemini_client: genai.Client,
    question: str,
    conversation_messages: list[dict],
    user_id: str = "",
) -> str:
    """
    Resolve a follow-up question into a standalone question using:
    - Full current conversation from Supabase
    - Relevant memories from Supermemory (older sessions)

    Returns the resolved standalone question.
    If resolution fails or question is already standalone, returns original.
    """
    # If no conversation history, question is standalone
    if not conversation_messages:
        return question

    # Build conversation history string
    history_parts = []
    for msg in conversation_messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "")
        history_parts.append(f"{role}: {content}")

    conversation_history = "\n\n".join(history_parts)

    # Get relevant memories from past sessions (if available)
    past_context = ""
    if user_id:
        past_context = _search_memories(user_id, question)

    # Build the resolver input
    resolver_input = f"Conversation history:\n{conversation_history}"
    if past_context:
        resolver_input += f"\n\n{past_context}"
    resolver_input += f"\n\nFollow-up question: {question}"

    try:
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=resolver_input,
            config=genai_types.GenerateContentConfig(
                system_instruction=RESOLVER_PROMPT,
                temperature=0,
                max_output_tokens=512,
            ),
        )
        resolved = response.text.strip()
        if resolved and resolved != question:
            logger.info("Resolved follow-up: '%s' → '%s'", question[:60], resolved[:80])
        return resolved or question
    except Exception as e:
        logger.warning("Question resolution failed (%s), using original", e)
        return question


def get_user_profile(user_id: str) -> str:
    """
    Fetch long-term user profile from Supermemory (static + dynamic facts).
    Used for personalization in the system prompt.
    No search query — just the profile.
    """
    client = _get_client()
    if not client:
        return ""

    try:
        result = client.profile(container_tag=user_id)

        profile = None
        if hasattr(result, 'profile'):
            profile = result.profile
        elif isinstance(result, dict):
            profile = result.get('profile')

        if not profile:
            return ""

        parts = []
        static = profile.static if hasattr(profile, 'static') else (profile.get('static') if isinstance(profile, dict) else [])
        dynamic = profile.dynamic if hasattr(profile, 'dynamic') else (profile.get('dynamic') if isinstance(profile, dict) else [])

        if static:
            parts.append("User background:")
            for fact in static:
                parts.append(f"- {fact}")

        if dynamic:
            parts.append("Recent focus areas:")
            for fact in dynamic[:5]:
                parts.append(f"- {fact}")

        if not parts:
            return ""

        context = "\n".join(parts)
        logger.info("Supermemory profile for user %s (%d chars)", user_id, len(context))
        return context

    except Exception as e:
        logger.warning("Supermemory profile failed: %s", e)
        return ""


def store_conversation(user_id: str, question: str, answer: str):
    """
    Store full Q&A in Supermemory for long-term memory.
    This builds the user profile and enables cross-session search.
    """
    client = _get_client()
    if not client:
        return

    try:
        content = f"User question: {question}\nAssistant answer: {answer}"
        client.add(
            content=content,
            container_tag=user_id,
        )
        logger.info("Stored conversation in Supermemory for user %s", user_id)
    except Exception as e:
        logger.warning("Supermemory store failed: %s", e)
