"""Shared text normalization utilities used by page_index and page_finder."""

import re

_NON_ASCII_RE = re.compile(r'[^\x20-\x7E]')
_HYPHEN_BREAK_RE = re.compile(r'-\s+')


def normalize(text: str) -> str:
    """Normalize text: rejoin hyphenated line breaks, strip non-ASCII, collapse whitespace."""
    text = _HYPHEN_BREAK_RE.sub('', text)
    text = _NON_ASCII_RE.sub('', text)
    return " ".join(text.split())
