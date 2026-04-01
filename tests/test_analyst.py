"""
test_analyst.py — Tests for the pre-trade analyst committee.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from sauce.analyst import AnalystVerdict, analyst_committee
from sauce.memory import MemoryEntry


def _run(coro):
    """Helper to run async code in tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAnalystVerdict:
    def test_frozen_dataclass(self):
        v = AnalystVerdict(
            approve=True,
            confidence=80,
            bull_case="good",
            bear_case="bad",
            reasoning="ok",
        )
        assert v.approve is True
        assert v.confidence == 80
        assert v.size_fraction == pytest.approx(1.0)

    def test_reject_verdict(self):
        v = AnalystVerdict(
            approve=False,
            confidence=30,
            bull_case="weak",
            bear_case="strong",
            reasoning="risk too high",
        )
        assert v.approve is False
        assert v.size_fraction == pytest.approx(0.0)


class TestAnalystCommittee:
    """Tests for analyst_committee() with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_approve_flow(self):
        """Both LLM calls succeed and PM approves."""
        analysis_response = json.dumps({
            "bull_case": "Strong momentum and volume",
            "bear_case": "RSI approaching overbought",
        })
        verdict_response = json.dumps({
            "approve": True,
            "confidence": 75,
            "reasoning": "Bull case outweighs bearish concerns",
        })

        with patch("sauce.analyst.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = [analysis_response, verdict_response]
            result = await analyst_committee(
                symbol="BTC/USD",
                strategy_name="crypto_momentum",
                score=70,
                threshold=40,
                regime="bullish",
                current_price=50000.0,
                rsi_14=45.0,
                macd_hist=0.001,
                bb_pct=0.3,
                volume_ratio=1.5,
            )

        assert result.approve is True
        assert result.confidence == 75
        assert result.size_fraction == pytest.approx(0.75)
        assert "momentum" in result.bull_case.lower()
        assert mock_claude.call_count == 2

    @pytest.mark.asyncio
    async def test_reject_flow(self):
        """Both LLM calls succeed and PM rejects."""
        analysis_response = json.dumps({
            "bull_case": "Mild positive momentum",
            "bear_case": "Extreme overbought, volume declining",
        })
        verdict_response = json.dumps({
            "approve": False,
            "confidence": 25,
            "reasoning": "Bear case is stronger — too risky",
        })

        with patch("sauce.analyst.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = [analysis_response, verdict_response]
            result = await analyst_committee(
                symbol="ETH/USD",
                strategy_name="crypto_momentum",
                score=45,
                threshold=40,
                regime="bearish",
                current_price=3000.0,
                rsi_14=72.0,
                macd_hist=-0.002,
                bb_pct=0.9,
                volume_ratio=0.5,
            )

        assert result.approve is False
        assert result.confidence == 25
        assert result.size_fraction == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_analysis_llm_failure_defaults_reject(self):
        """If the first LLM call fails, default to reject."""
        from sauce.adapters.llm import LLMError

        with patch("sauce.analyst.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = LLMError("API down")
            result = await analyst_committee(
                symbol="SOL/USD",
                strategy_name="crypto_momentum",
                score=60,
                threshold=40,
                regime="neutral",
                current_price=100.0,
                rsi_14=50.0,
                macd_hist=0.0,
                bb_pct=0.5,
                volume_ratio=1.0,
            )

        assert result.approve is False
        assert result.confidence == 0
        assert "unavailable" in result.bull_case.lower()
        assert mock_claude.call_count == 1  # Only first call made

    @pytest.mark.asyncio
    async def test_verdict_llm_failure_defaults_reject(self):
        """If the second LLM call fails, default to reject with analysis intact."""
        from sauce.adapters.llm import LLMError

        analysis_response = json.dumps({
            "bull_case": "Good setup",
            "bear_case": "Some risk",
        })

        with patch("sauce.analyst.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = [analysis_response, LLMError("Timeout")]
            result = await analyst_committee(
                symbol="BTC/USD",
                strategy_name="crypto_momentum",
                score=70,
                threshold=40,
                regime="bullish",
                current_price=50000.0,
                rsi_14=45.0,
                macd_hist=0.001,
                bb_pct=0.3,
                volume_ratio=1.5,
            )

        assert result.approve is False
        assert result.confidence == 0
        assert result.bull_case == "Good setup"
        assert result.bear_case == "Some risk"

    @pytest.mark.asyncio
    async def test_with_memories(self):
        """Memories are included in the PM verdict call."""
        analysis_response = json.dumps({
            "bull_case": "Momentum is strong",
            "bear_case": "Slight headwind",
        })
        verdict_response = json.dumps({
            "approve": True,
            "confidence": 85,
            "reasoning": "Past trades confirm pattern",
        })

        memories = [
            MemoryEntry(
                situation="BTC bullish RSI=42",
                outcome="profit +5%",
                lesson="BTC bullish setups with low RSI tend to work",
            ),
        ]

        with patch("sauce.analyst.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = [analysis_response, verdict_response]
            result = await analyst_committee(
                symbol="BTC/USD",
                strategy_name="crypto_momentum",
                score=70,
                threshold=40,
                regime="bullish",
                current_price=50000.0,
                rsi_14=42.0,
                macd_hist=0.001,
                bb_pct=0.3,
                volume_ratio=1.5,
                memories=memories,
            )

        assert result.approve is True
        assert result.confidence == 85
        # Verify the PM call got the memories in its prompt
        pm_call = mock_claude.call_args_list[1]
        assert "past trades" in pm_call.kwargs.get("user", pm_call[1].get("user", "")).lower() or \
               "past trades" in str(pm_call).lower()

    @pytest.mark.asyncio
    async def test_confidence_clamped(self):
        """Confidence is clamped to 0-100 range."""
        analysis_response = json.dumps({
            "bull_case": "x",
            "bear_case": "y",
        })
        verdict_response = json.dumps({
            "approve": True,
            "confidence": 150,  # out of range
            "reasoning": "ok",
        })

        with patch("sauce.analyst.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = [analysis_response, verdict_response]
            result = await analyst_committee(
                symbol="BTC/USD",
                strategy_name="crypto_momentum",
                score=70,
                threshold=40,
                regime="bullish",
                current_price=50000.0,
                rsi_14=50.0,
                macd_hist=0.0,
                bb_pct=0.5,
                volume_ratio=1.0,
            )

        assert result.confidence == 100  # clamped
        assert result.size_fraction == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_none_indicators(self):
        """Works when indicator values are None."""
        analysis_response = json.dumps({
            "bull_case": "x",
            "bear_case": "y",
        })
        verdict_response = json.dumps({
            "approve": True,
            "confidence": 60,
            "reasoning": "ok",
        })

        with patch("sauce.analyst.call_llm", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = [analysis_response, verdict_response]
            result = await analyst_committee(
                symbol="BTC/USD",
                strategy_name="crypto_momentum",
                score=70,
                threshold=40,
                regime="neutral",
                current_price=50000.0,
                rsi_14=None,
                macd_hist=None,
                bb_pct=None,
                volume_ratio=None,
            )

        assert result.approve is True
        assert result.size_fraction == pytest.approx(0.50)
