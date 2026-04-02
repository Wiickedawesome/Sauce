"""
test_reflection.py — Tests for the post-trade reflection system.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from sauce.memory import MemoryEntry
from sauce.reflection import reflect_on_trade


class TestReflectOnTrade:
    """Tests for reflect_on_trade() with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_successful_reflection(self):
        """LLM returns a lesson successfully."""
        response = json.dumps({
            "lesson": "BTC bullish setups with low RSI tend to follow through well",
        })

        with patch("sauce.reflection.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = response
            lesson = await reflect_on_trade(
                symbol="BTC/USD",
                situation="BTC bullish RSI=42 MACD positive",
                outcome="profit +5% via profit_target in 12h",
            )

        assert lesson is not None
        assert "RSI" in lesson
        assert mock_claude.call_count == 1

    @pytest.mark.asyncio
    async def test_reflection_with_memories(self):
        """Memories are included in the reflection prompt."""
        response = json.dumps({
            "lesson": "Consistent pattern: low RSI bullish BTC works",
        })
        memories = [
            MemoryEntry(
                situation="BTC RSI=40",
                outcome="profit +3%",
                lesson="Low RSI entries in BTC bullish work",
            ),
        ]

        with patch("sauce.reflection.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = response
            lesson = await reflect_on_trade(
                symbol="BTC/USD",
                situation="BTC RSI=42",
                outcome="profit +5%",
                memories=memories,
            )

        assert lesson is not None
        # Verify memories appear in the prompt
        call_kwargs = mock_claude.call_args
        user_prompt = call_kwargs.kwargs.get("user", call_kwargs[1].get("user", ""))
        assert "past trades" in user_prompt.lower()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none(self):
        """If the LLM call fails, return None (non-critical)."""
        from sauce.adapters.llm import LLMError

        with patch("sauce.reflection.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = LLMError("API down")
            lesson = await reflect_on_trade(
                symbol="BTC/USD",
                situation="test",
                outcome="test",
            )

        assert lesson is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        """If the LLM returns non-JSON, return None."""
        with patch("sauce.reflection.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "This is not JSON at all"
            lesson = await reflect_on_trade(
                symbol="BTC/USD",
                situation="test",
                outcome="test",
            )

        assert lesson is None

    @pytest.mark.asyncio
    async def test_empty_lesson_returns_none(self):
        """If the LLM returns an empty lesson, return None."""
        response = json.dumps({"lesson": ""})

        with patch("sauce.reflection.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = response
            lesson = await reflect_on_trade(
                symbol="BTC/USD",
                situation="test",
                outcome="test",
            )

        assert lesson is None

    @pytest.mark.asyncio
    async def test_no_memories(self):
        """Works fine with no memories."""
        response = json.dumps({
            "lesson": "First trade — be cautious",
        })

        with patch("sauce.reflection.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = response
            lesson = await reflect_on_trade(
                symbol="ETH/USD",
                situation="ETH bearish RSI=70",
                outcome="loss -3%",
                memories=None,
            )

        assert lesson == "First trade — be cautious"
