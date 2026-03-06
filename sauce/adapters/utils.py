"""
adapters/utils.py — Shared retry helpers for synchronous Alpaca SDK calls.

Finding 4.1: broker.py and market_data.py had no retry logic for transient
network failures. This module provides a synchronous exponential-backoff retry
wrapper that mirrors the pattern already used in adapters/llm.py.

Rules:
- NEVER use call_with_retry() on broker.place_order() — submitting a duplicate
  order is worse than a single network failure.
- Only retries on clearly transient errors (network, timeout, rate-limit HTTP
  status codes embedded in exception messages). Auth/validation errors are
  re-raised immediately.
"""

import logging
import time
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BASE_DELAY = 1.0  # seconds; doubles each attempt: 1s → 2s → 4s

T = TypeVar("T")


def _is_transient(exc: Exception) -> bool:
    """
    Return True for errors that are safe to retry.

    Network-level exceptions (ConnectionError, TimeoutError, OSError) are always
    transient. Alpaca SDK wraps HTTP errors as generic exceptions, so the message
    is inspected for recognisable transient patterns.
    """
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    msg = str(exc).lower()
    transient_markers = (
        "timeout",
        "timed out",
        "connection",
        "reset by peer",
        "rate limit",
        "429",  # Too Many Requests
        "502",  # Bad Gateway
        "503",  # Service Unavailable
        "504",  # Gateway Timeout
    )
    return any(marker in msg for marker in transient_markers)


def call_with_retry(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """
    Call fn(*args, **kwargs) with up to _MAX_RETRIES attempts.

    Retries only on transient errors. Non-transient errors (auth failures,
    validation errors, etc.) are re-raised on the first attempt without delay.

    Usage::

        result = call_with_retry(client.get_account)
        quote  = call_with_retry(client.get_stock_latest_quote, request_obj)

    Do NOT use for non-idempotent operations (e.g. place_order).
    """
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_transient(exc):
                raise  # permanent error — no benefit in retrying
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            delay = _BASE_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "call_with_retry: transient error on attempt %d/%d, "
                "retrying in %.1fs: %s",
                attempt, _MAX_RETRIES, delay, exc,
            )
            time.sleep(delay)

    # All retries exhausted — re-raise the last transient exception
    assert last_exc is not None  # loop always sets last_exc before break
    raise last_exc
