"""
adapters/llm.py — Claude adapter via Anthropic API.

Rules:
- Every call logs an AuditEvent before and after.
- On total failure: raises LLMError with context logged to DB.
- Retries up to 3 times with exponential back-off on 429 / 5xx.
"""

import asyncio
import logging
from datetime import UTC, datetime

from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds — doubles each attempt


def _strip_fences(text: str) -> str:
    """Strip markdown code fences and extract JSON from mixed prose.

    Handles (in order):
      1. ```json\n{...}\n```  or  ```\n{...}\n```
      2. Prose before/after a JSON object — extracts the first { ... } block.
    Returns the inner content stripped of surrounding whitespace.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence line (e.g. ```json or ```)
        stripped = stripped[stripped.index("\n") + 1 :] if "\n" in stripped else stripped[3:]
        # Remove closing fence
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
        return stripped.strip()

    # If the response is already valid JSON, return as-is
    if stripped.startswith("{"):
        return stripped

    # Extract first JSON object from mixed prose (e.g. "Let me analyze...\n{...}")
    brace_start = stripped.find("{")
    if brace_start != -1:
        brace_end = stripped.rfind("}")
        if brace_end > brace_start:
            return stripped[brace_start : brace_end + 1].strip()

    return stripped


# ── Exceptions ────────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Raised when all retries are exhausted or a fatal LLM error occurs."""

    pass


# ── Internal helpers ──────────────────────────────────────────────────────────


async def _call_anthropic(
    system: str,
    user: str,
    model: str,
    api_key: str,
    loop_id: str,
    temperature: float = 0.3,
) -> str:
    """
    Call Anthropic's Messages API with retry logic.
    """
    import anthropic

    from sauce.adapters.db import log_event

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="llm_call",
            payload={
                "provider": "anthropic",
                "model": model,
                "system_chars": len(system),
                "user_chars": len(user),
            },
            timestamp=datetime.now(UTC),
            prompt_version=get_settings().prompt_version,
        )
    )

    # Construct the client once — not inside the retry loop — to avoid creating
    # a new HTTP connection pool on every attempt (Finding 7.6).
    client = anthropic.AsyncAnthropic(api_key=api_key)
    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            message = await client.messages.create(
                model=model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=temperature,
            )
            content = _strip_fences(message.content[0].text)  # type: ignore[union-attr]

            log_event(
                AuditEvent(
                    loop_id=loop_id,
                    event_type="llm_response",
                    payload={
                        "provider": "anthropic",
                        "model": model,
                        "response_chars": len(content),
                        "stop_reason": message.stop_reason,
                    },
                    timestamp=datetime.now(UTC),
                    prompt_version=get_settings().prompt_version,
                )
            )

            return content

        except anthropic.RateLimitError:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Anthropic rate limit (attempt %d/%d). Retrying in %.1fs.",
                attempt,
                MAX_RETRIES,
                delay,
            )
            await asyncio.sleep(delay)
            last_exc = LLMError(f"Anthropic rate limited after {attempt} attempts")

        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                await asyncio.sleep(delay)
                last_exc = LLMError(f"Anthropic server error {exc.status_code}")
            else:
                # 4xx (not 429) — don't retry
                last_exc = LLMError(f"Anthropic API error {exc.status_code}: {exc.message}")
                break

        except anthropic.APIConnectionError as exc:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Anthropic connection error (attempt %d/%d). Retrying in %.1fs: %s",
                attempt,
                MAX_RETRIES,
                delay,
                exc,
            )
            await asyncio.sleep(delay)
            last_exc = LLMError(f"Anthropic connection error: {exc}")

    error_msg = str(last_exc) if last_exc else "Unknown Anthropic error"
    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"provider": "anthropic", "error": error_msg},
            timestamp=datetime.now(UTC),
        )
    )
    raise LLMError(f"Anthropic call failed after {MAX_RETRIES} retries: {error_msg}")


# ── Ollama (local) backend ────────────────────────────────────────────────────


async def _call_ollama(
    system: str,
    user: str,
    loop_id: str,
    temperature: float = 0.3,
) -> str:
    """
    Call local Ollama server via OpenAI-compatible API.
    Returns raw text response (caller handles JSON parsing).
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url}/v1/chat/completions"
    payload = {
        "model": settings.ollama_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
    }

    for attempt in range(1, 4):  # 3 attempts
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
                return _strip_fences(raw)
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Ollama HTTP error (attempt %d/3): %s", attempt, exc
            )
            if attempt == 3:
                raise LLMError(f"Ollama failed after 3 attempts: {exc}") from exc
            await asyncio.sleep(2 ** attempt)
        except httpx.RequestError as exc:
            logger.warning(
                "Ollama connection error (attempt %d/3): %s", attempt, exc
            )
            if attempt == 3:
                raise LLMError(f"Ollama connection failed: {exc}") from exc
            await asyncio.sleep(2 ** attempt)
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Ollama response parse error: {exc}") from exc

    raise LLMError("Ollama call failed unexpectedly")


# ── Public interface ──────────────────────────────────────────────────────────


async def call_claude(
    system: str,
    user: str,
    loop_id: str = "unset",
    model: str | None = None,
    temperature: float = 0.3,
) -> str:
    """
    Call Claude with a system + user prompt and return the raw string response.

    Parameters
    ----------
    system:    The system prompt (role + anti-hallucination instructions).
    user:      The user message (grounded data + output schema).
    loop_id:   The UUID of the current loop run, used to tie audit events.
    model:     Override the model name from config. Defaults to LLM_MODEL.

    Returns
    -------
    Raw string content from Claude. Caller is responsible for parsing JSON.

    Raises
    ------
    LLMError if all retries are exhausted or a fatal error occurs.
    """
    settings = get_settings()
    effective_model = model or settings.llm_model

    if not settings.anthropic_api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set")
    return await _call_anthropic(
        system=system,
        user=user,
        model=effective_model,
        api_key=settings.anthropic_api_key,
        loop_id=loop_id,
        temperature=temperature,
    )


async def call_llm(
    system: str,
    user: str,
    loop_id: str = "unset",
    *,
    provider: str = "anthropic",
    temperature: float = 0.3,
) -> str:
    """
    Unified LLM dispatcher — routes to Anthropic or Ollama based on provider.

    Args:
        system: System prompt
        user: User prompt
        loop_id: Loop ID for audit logging
        provider: 'anthropic' or 'ollama'
        temperature: Sampling temperature

    Returns:
        Raw text response (JSON fence-stripped)
    """
    if provider == "ollama":
        return await _call_ollama(
            system=system,
            user=user,
            loop_id=loop_id,
            temperature=temperature,
        )
    else:
        return await call_claude(
            system=system,
            user=user,
            loop_id=loop_id,
            temperature=temperature,
        )
