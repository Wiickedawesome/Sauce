"""
tests/test_llm.py — Tests for adapters/llm.py.

Mocks httpx.AsyncClient and the anthropic SDK — no real API calls.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sauce.adapters.llm import LLMError, _mask_token, call_claude


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("LLM_PROVIDER", "github")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_testtoken1234")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet")


def set_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LLM_MODEL", "claude-3-5-sonnet")


def make_github_response(content: str, status_code: int = 200) -> MagicMock:
    """Build a mock httpx response for the GitHub Models endpoint."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"content": content},
                "finish_reason": "stop",
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()
    return mock_response


# ── _mask_token ───────────────────────────────────────────────────────────────

def test_mask_token_short() -> None:
    assert _mask_token("abc") == "***"


def test_mask_token_long() -> None:
    masked = _mask_token("ghp_abcdefghij1234")
    assert masked.startswith("ghp_")
    assert "***" in masked
    assert "abcdefghij" not in masked


def test_mask_token_does_not_leak_middle() -> None:
    token = "ghp_" + "X" * 20 + "abcd"
    masked = _mask_token(token)
    assert "X" * 20 not in masked


# ── GitHub Models — happy path ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_claude_github_returns_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: object,
) -> None:
    set_github_env(monkeypatch)
    monkeypatch.setenv("DB_PATH", "data/test_llm.db")
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    expected = '{"side": "hold", "confidence": 0.0}'
    mock_response = make_github_response(expected)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        with patch("sauce.adapters.db.log_event"):
            result = await call_claude(
                system="You are a trading assistant.",
                user="Analyse AAPL.",
                loop_id="test-loop-001",
            )

    assert result == expected
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_call_claude_github_sends_correct_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_github_env(monkeypatch)
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    mock_response = make_github_response("response content")
    captured_payload: dict = {}

    async def capture_post(url: str, headers: dict, json: dict, **kwargs: object) -> MagicMock:
        captured_payload.update(json)
        return mock_response

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = capture_post
        mock_client_class.return_value = mock_client

        with patch("sauce.adapters.db.log_event"):
            await call_claude(system="sys", user="usr", loop_id="loop-001")

    assert captured_payload["model"] == "claude-3-5-sonnet"
    assert captured_payload["messages"][0]["role"] == "system"
    assert captured_payload["messages"][1]["role"] == "user"
    get_settings.cache_clear()


# ── GitHub Models — rate limit retry ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_claude_github_retries_on_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_github_env(monkeypatch)
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    rate_limit_response = make_github_response("", 429)
    success_response = make_github_response('{"side": "hold"}')

    call_count = 0

    async def mock_post(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return rate_limit_response
        return success_response

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = mock_post
        mock_client_class.return_value = mock_client

        with patch("sauce.adapters.db.log_event"):
            with patch("asyncio.sleep", new_callable=AsyncMock):  # skip real delays
                result = await call_claude(system="sys", user="usr", loop_id="loop-retry")

    assert result == '{"side": "hold"}'
    assert call_count == 3
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_call_claude_github_raises_after_all_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_github_env(monkeypatch)
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    always_rate_limited = make_github_response("", 429)

    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=always_rate_limited)
        mock_client_class.return_value = mock_client

        with patch("sauce.adapters.db.log_event"):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMError, match="failed after"):
                    await call_claude(system="sys", user="usr", loop_id="loop-fail")

    get_settings.cache_clear()


# ── Missing credentials ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_claude_github_no_token_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("LLM_PROVIDER", "github")
    monkeypatch.setenv("GITHUB_TOKEN", "")
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    with pytest.raises(LLMError, match="GITHUB_TOKEN"):
        await call_claude(system="sys", user="usr")

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_call_claude_anthropic_no_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        await call_claude(system="sys", user="usr")

    get_settings.cache_clear()


# ── Unknown provider ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_call_claude_unknown_provider_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Settings validator would catch this, but test the LLM adapter guard too."""
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("LLM_PROVIDER", "github")
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    settings = get_settings()
    # Directly patch the provider on the settings object to bypass validator
    object.__setattr__(settings, "llm_provider", "openai")

    with patch("sauce.adapters.llm.get_settings", return_value=settings):
        with pytest.raises(LLMError, match="Unknown LLM_PROVIDER"):
            await call_claude(system="sys", user="usr")

    get_settings.cache_clear()
