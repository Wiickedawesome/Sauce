"""
tests/test_llm.py — Tests for adapters/llm.py.

Mocks the anthropic SDK — no real API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sauce.adapters.llm import LLMError, call_claude

# ── Helpers ───────────────────────────────────────────────────────────────────


def set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")


# ── Happy path ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_claude_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.core.config import get_settings

    get_settings.cache_clear()

    expected = '{"side": "hold", "confidence": 0.0}'
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=expected)]
    mock_message.stop_reason = "end_turn"

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock(return_value=mock_message)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            result = await call_claude(
                system="You are a trading assistant.",
                user="Analyse AAPL.",
                loop_id="test-loop-001",
            )

    assert result == expected
    get_settings.cache_clear()


# ── Rate-limit retry ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_claude_retries_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.core.config import get_settings

    get_settings.cache_clear()

    import anthropic as anthropic_mod

    call_count = 0
    success_message = MagicMock()
    success_message.content = [MagicMock(text='{"side": "hold"}')]
    success_message.stop_reason = "end_turn"

    async def mock_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise anthropic_mod.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429),
                body=None,
            )
        return success_message

    mock_client = AsyncMock()
    mock_client.messages.create = mock_create

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await call_claude(system="sys", user="usr", loop_id="loop-retry")

    assert result == '{"side": "hold"}'
    assert call_count == 3
    get_settings.cache_clear()


# ── Exhausted retries ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_claude_raises_after_all_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.core.config import get_settings

    get_settings.cache_clear()

    import anthropic as anthropic_mod

    async def always_fail(**kwargs: object) -> None:
        raise anthropic_mod.RateLimitError(
            message="rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

    mock_client = AsyncMock()
    mock_client.messages.create = always_fail

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(LLMError, match="failed after"):
                    await call_claude(system="sys", user="usr", loop_id="loop-fail")

    get_settings.cache_clear()


# ── Missing credentials ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_claude_retries_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """APIConnectionError should be retried (network blip), not immediately fatal."""
    set_env(monkeypatch)
    from sauce.core.config import get_settings

    get_settings.cache_clear()

    import anthropic as anthropic_mod

    call_count = 0
    success_message = MagicMock()
    success_message.content = [MagicMock(text='{"side": "hold"}')]
    success_message.stop_reason = "end_turn"

    async def mock_create(**kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise anthropic_mod.APIConnectionError(request=MagicMock())
        return success_message

    mock_client = AsyncMock()
    mock_client.messages.create = mock_create

    with patch("anthropic.AsyncAnthropic", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await call_claude(system="sys", user="usr", loop_id="loop-conn")

    assert result == '{"side": "hold"}'
    assert call_count == 2
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_call_claude_no_api_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from pydantic import ValidationError

    from sauce.core.config import get_settings

    get_settings.cache_clear()

    with pytest.raises(ValidationError, match="API key must not be empty"):
        get_settings()

    get_settings.cache_clear()
