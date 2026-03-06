"""
adapters/llm.py — Claude adapter supporting GitHub Models and Anthropic API.

Provider is selected via LLM_PROVIDER env var:
  - "github"    → GitHub Models inference endpoint (free, uses GITHUB_TOKEN)
  - "anthropic" → Anthropic API (paid, uses ANTHROPIC_API_KEY)

Switching providers requires only a one-line .env change — no code changes.

Rules:
- Every call logs an AuditEvent before and after.
- On total failure: raises LLMError with context logged to DB.
- Retries up to 3 times with exponential back-off on 429 / 5xx.
- GITHUB_TOKEN is never logged — masked as ***.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx

from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)

# GitHub Models inference endpoint (OpenAI-compatible)
GITHUB_MODELS_BASE_URL = "https://models.inference.ai.azure.com"
GITHUB_MODELS_PATH = "/chat/completions"

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds — doubles each attempt


def _strip_fences(text: str) -> str:
    """Strip markdown code fences Claude sometimes wraps JSON in.

    Handles:
        ```json\n{...}\n```
        ```\n{...}\n```
    Returns the inner content stripped of surrounding whitespace.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence line (e.g. ```json or ```)
        stripped = stripped[stripped.index("\n") + 1:] if "\n" in stripped else stripped[3:]
        # Remove closing fence
        if stripped.endswith("```"):
            stripped = stripped[: stripped.rfind("```")]
    return stripped.strip()


# ── Exceptions ────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when all retries are exhausted or a fatal LLM error occurs."""
    pass


# ── Internal helpers ──────────────────────────────────────────────────────────

def _mask_token(token: str) -> str:
    """Return a masked version of a token for safe logging."""
    if len(token) <= 8:
        return "***"
    return token[:4] + "***" + token[-4:]


async def _call_github_models(
    system: str,
    user: str,
    model: str,
    token: str,
    loop_id: str,
) -> str:
    """
    POST to the GitHub Models inference endpoint and return the response content.

    Uses OpenAI-compatible /chat/completions format.
    Retries on 429 (rate limit) and 5xx (server error) with exponential back-off.
    """
    from sauce.adapters.db import log_event  # local import to avoid circular deps

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.1,  # low temperature for consistent structured output
    }

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="llm_call",
        payload={
            "provider": "github",
            "model": model,
            "token_masked": _mask_token(token),
            "system_chars": len(system),
            "user_chars": len(user),
        },
        timestamp=datetime.now(timezone.utc),
        prompt_version=get_settings().prompt_version,
    ))

    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await client.post(
                    f"{GITHUB_MODELS_BASE_URL}{GITHUB_MODELS_PATH}",
                    headers=headers,
                    json=payload,
                )

                if response.status_code == 429:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "GitHub Models rate limit hit (attempt %d/%d). "
                        "Retrying in %.1fs.",
                        attempt, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_exc = LLMError(f"Rate limited after {attempt} attempts")
                    continue

                if response.status_code >= 500:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "GitHub Models server error %d (attempt %d/%d). "
                        "Retrying in %.1fs.",
                        response.status_code, attempt, MAX_RETRIES, delay,
                    )
                    await asyncio.sleep(delay)
                    last_exc = LLMError(f"Server error {response.status_code}")
                    continue

                response.raise_for_status()
                data = response.json()
                content: str = _strip_fences(data["choices"][0]["message"]["content"])

                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="llm_response",
                    payload={
                        "provider": "github",
                        "model": model,
                        "response_chars": len(content),
                        "finish_reason": data["choices"][0].get("finish_reason"),
                    },
                    timestamp=datetime.now(timezone.utc),
                    prompt_version=get_settings().prompt_version,
                ))

                return content

            except httpx.HTTPStatusError as exc:
                last_exc = LLMError(f"HTTP error: {exc.response.status_code}")
                logger.error("LLM HTTP error on attempt %d: %s", attempt, exc)
            except httpx.RequestError as exc:
                last_exc = LLMError(f"Request error: {exc}")
                logger.error("LLM request error on attempt %d: %s", attempt, exc)

    # All retries exhausted — log and raise
    error_msg = str(last_exc) if last_exc else "Unknown LLM error"
    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="error",
        payload={"provider": "github", "error": error_msg, "retries": MAX_RETRIES},
        timestamp=datetime.now(timezone.utc),
    ))
    raise LLMError(f"GitHub Models call failed after {MAX_RETRIES} retries: {error_msg}")


async def _call_anthropic(
    system: str,
    user: str,
    model: str,
    api_key: str,
    loop_id: str,
) -> str:
    """
    Call Anthropic API using the official anthropic SDK.

    Used when LLM_PROVIDER=anthropic.
    """
    import anthropic  # imported here so missing key doesn't fail on github mode
    from sauce.adapters.db import log_event

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="llm_call",
        payload={
            "provider": "anthropic",
            "model": model,
            "system_chars": len(system),
            "user_chars": len(user),
        },
        timestamp=datetime.now(timezone.utc),
        prompt_version=get_settings().prompt_version,
    ))

    last_exc: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = anthropic.AsyncAnthropic(api_key=api_key)
            message = await client.messages.create(
                model=model,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            content = _strip_fences(message.content[0].text)  # type: ignore[index]

            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="llm_response",
                payload={
                    "provider": "anthropic",
                    "model": model,
                    "response_chars": len(content),
                    "stop_reason": message.stop_reason,
                },
                timestamp=datetime.now(timezone.utc),
                prompt_version=get_settings().prompt_version,
            ))

            return content

        except anthropic.RateLimitError:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Anthropic rate limit (attempt %d/%d). Retrying in %.1fs.",
                attempt, MAX_RETRIES, delay,
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

    error_msg = str(last_exc) if last_exc else "Unknown Anthropic error"
    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="error",
        payload={"provider": "anthropic", "error": error_msg},
        timestamp=datetime.now(timezone.utc),
    ))
    raise LLMError(f"Anthropic call failed after {MAX_RETRIES} retries: {error_msg}")


# ── Public interface ──────────────────────────────────────────────────────────

async def call_claude(
    system: str,
    user: str,
    loop_id: str = "unset",
    model: str | None = None,
) -> str:
    """
    Call Claude with a system + user prompt and return the raw string response.

    Routing:
    - LLM_PROVIDER=github  → GitHub Models inference endpoint
    - LLM_PROVIDER=anthropic → Anthropic API

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

    if settings.llm_provider == "github":
        if not settings.github_token:
            raise LLMError("GITHUB_TOKEN is not set but LLM_PROVIDER=github")
        return await _call_github_models(
            system=system,
            user=user,
            model=effective_model,
            token=settings.github_token,
            loop_id=loop_id,
        )

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic")
        return await _call_anthropic(
            system=system,
            user=user,
            model=effective_model,
            api_key=settings.anthropic_api_key,
            loop_id=loop_id,
        )

    raise LLMError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
