"""Generate follow-up question suggestions based on the Q&A exchange."""

import json
import logging

from google import genai
from google.genai import types
from config import MODEL

logger = logging.getLogger(__name__)

FOLLOWUP_PROMPT = """You are a bioprocess engineering assistant. Given a question and answer exchange, suggest exactly 3 concise follow-up questions the user might want to ask next.

Rules:
- Each question should be short (under 15 words)
- Questions should explore different angles: deeper detail, related concept, practical application
- Questions should feel natural, like what a curious scientist would ask next
- Output ONLY a JSON array of 3 strings, no markdown, no explanation

Example output:
["How does this affect scale-up?", "What are typical industrial values?", "Can you show the governing equation?"]"""


def generate_follow_ups(client: genai.Client, question: str, answer: str) -> list[str]:
    """Generate 3 follow-up question suggestions. Returns empty list on failure."""
    try:
        prompt = f"Question: {question}\n\nAnswer: {answer}"

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=FOLLOWUP_PROMPT,
                temperature=0.7,
                max_output_tokens=2048,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        raw = response.text.strip()
        logger.info("Follow-up raw response: %s", raw)
        # Extract JSON array from response (may have markdown fences)
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result = json.loads(raw)
        if isinstance(result, list) and len(result) >= 1:
            return [str(q) for q in result[:3]]
        return []
    except Exception as e:
        logger.warning("Follow-up generation failed: %s", e)
        return []
