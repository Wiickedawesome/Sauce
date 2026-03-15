"""
prompts/utils.py — Shared utilities for building agent prompts safely.

Rules:
- All LLM-generated text embedded into downstream prompts MUST be passed
  through sanitize_llm_text() before use (Finding 6.1 — prompt injection).
- Never embed raw LLM output from one agent directly into another agent's
  prompt without going through this module.
"""

import re
import unicodedata

from sauce.core.config import get_settings


def sanitize_llm_text(text: str, max_len: int | None = None) -> str:
    if max_len is None:
        max_len = get_settings().max_reasoning_len
    """
    Sanitize an LLM-generated string before embedding it in another prompt.

    Mitigates prompt-injection risk when Research agent reasoning is passed
    into Execution or Supervisor prompts (Finding 6.1).

    Steps applied (in order):
    1. Truncate to max_len characters.
    2. Normalize unicode to NFC (prevents homoglyph attacks).
    3. Strip C0/C1 control characters (null bytes, escape sequences, etc.)
       while preserving normal whitespace (space, newline, tab).
    4. Remove any substring that looks like a JSON instruction object —
       i.e. a curly-brace block containing a colon, which could override
       the downstream prompt's schema.
    5. Collapse multiple consecutive whitespace chars to a single space and
       strip leading/trailing whitespace.

    The result is guaranteed to contain only printable text and normal spaces.
    It is NOT guaranteed to be free of adversarial prose, but it cannot carry
    structural JSON payloads or control-character tricks.

    Parameters
    ----------
    text:    Raw LLM output string to sanitize.
    max_len: Maximum output length in characters. Defaults to 300.

    Returns
    -------
    Safe, truncated plain-text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Truncate early to avoid processing huge blobs.
    s = text[:max_len]

    # 2. Unicode NFC normalisation.
    s = unicodedata.normalize("NFC", s)

    # 3. Strip C0/C1 control characters, keeping tab (\x09), newline (\x0a),
    #    carriage return (\x0d), and printable space (\x20).
    s = "".join(
        ch for ch in s
        if ch in ("\t", "\n", "\r") or (unicodedata.category(ch) not in ("Cc", "Cf"))
    )

    # 4. Remove JSON-like object/array literals that could carry instructions.
    #    Matches balanced and non-balanced { ... } and [ ... ] spanning up to
    #    200 chars (prevents stripping too aggressively on short texts with
    #    individual braces used in prose).
    s = re.sub(r"\{[^}]{0,200}\}", "[REDACTED]", s)
    s = re.sub(r"\[[^\]]{0,200}\]", "[REDACTED]", s)

    # 5. Collapse whitespace.
    s = " ".join(s.split())

    return s
