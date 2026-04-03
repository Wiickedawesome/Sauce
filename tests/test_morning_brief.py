"""
test_morning_brief.py — Tests for regime classification (heuristic + Claude).

Covers:
  - infer_intraday_regime() scoring boundaries
  - get_regime() Claude success path
  - get_regime() LLM failure → heuristic fallback
  - Invalid regime from Claude → defaults to neutral
  - JSON parse error handling
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from sauce.morning_brief import get_regime, infer_intraday_regime

# ── Heuristic regime inference ────────────────────────────────────────────────


class TestInferIntraday:
    def test_strong_bullish(self):
        """All indicators positive → bullish (score >= 2)."""
        regime, reason = infer_intraday_regime(
            btc_change=0.05,   # +1
            eth_change=0.04,   # +1
            spy_change=0.02,   # +1
            vix=14.0,          # +1
            btc_rsi=65.0,      # +1
        )
        assert regime == "bullish"
        assert "heuristic_score=5" in reason

    def test_strong_bearish(self):
        """All indicators negative → bearish (score <= -2)."""
        regime, _ = infer_intraday_regime(
            btc_change=-0.05,  # -1
            eth_change=-0.04,  # -1
            spy_change=-0.02,  # -1
            vix=30.0,          # -1
            btc_rsi=35.0,      # -1
        )
        assert regime == "bearish"

    def test_neutral_mixed(self):
        """Mixed signals → neutral (score between -1 and +1)."""
        regime, _ = infer_intraday_regime(
            btc_change=0.03,   # +1
            eth_change=-0.03,  # -1
            spy_change=0.005,  #  0  (< 0.01)
            vix=20.0,          #  0  (16 < vix < 25)
            btc_rsi=50.0,      #  0  (40 < rsi < 60)
        )
        assert regime == "neutral"

    def test_exact_boundary_bullish(self):
        """Score of exactly 2 → bullish."""
        regime, reason = infer_intraday_regime(
            btc_change=0.03,   # +1
            eth_change=0.03,   # +1
            spy_change=0.005,  #  0
            vix=20.0,          #  0
            btc_rsi=50.0,      #  0
        )
        assert regime == "bullish"
        assert "heuristic_score=2" in reason

    def test_exact_boundary_bearish(self):
        """Score of exactly -2 → bearish."""
        regime, reason = infer_intraday_regime(
            btc_change=-0.03,  # -1
            eth_change=-0.03,  # -1
            spy_change=0.005,  #  0
            vix=20.0,          #  0
            btc_rsi=50.0,      #  0
        )
        assert regime == "bearish"
        assert "heuristic_score=-2" in reason

    def test_reason_contains_all_values(self):
        _, reason = infer_intraday_regime(0.01, -0.01, 0.005, 20.0, 50.0)
        assert "btc=" in reason
        assert "eth=" in reason
        assert "spy=" in reason
        assert "vix=" in reason
        assert "btc_rsi=" in reason


# ── Claude regime classification ──────────────────────────────────────────────


class TestGetRegime:
    @pytest.mark.asyncio
    async def test_claude_returns_valid_regime(self):
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = '{"regime": "bullish", "reasoning": "strong momentum"}'
            regime = await get_regime(0.05, 0.04, 0.02, 14.0, 65.0)
            assert regime == "bullish"
            mock_claude.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_claude_returns_bearish(self):
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = '{"regime": "bearish", "reasoning": "sell-off"}'
            regime = await get_regime(-0.05, -0.04, -0.02, 30.0, 35.0)
            assert regime == "bearish"

    @pytest.mark.asyncio
    async def test_invalid_regime_defaults_to_neutral(self):
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = '{"regime": "apocalyptic", "reasoning": "doom"}'
            regime = await get_regime(0.0, 0.0, 0.0, 20.0, 50.0)
            assert regime == "neutral"

    @pytest.mark.asyncio
    async def test_missing_regime_key_defaults_to_neutral(self):
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = '{"reasoning": "no regime key"}'
            regime = await get_regime(0.0, 0.0, 0.0, 20.0, 50.0)
            assert regime == "neutral"

    @pytest.mark.asyncio
    async def test_json_parse_error_falls_back_to_heuristic(self):
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = "not json at all"
            # Strong bullish inputs → heuristic should return bullish
            regime = await get_regime(0.05, 0.04, 0.02, 14.0, 65.0)
            assert regime == "bullish"

    @pytest.mark.asyncio
    async def test_llm_error_falls_back_to_heuristic(self):
        from sauce.adapters.llm import LLMError

        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.side_effect = LLMError("API down")
            # Strong bearish inputs
            regime = await get_regime(-0.05, -0.04, -0.02, 30.0, 35.0)
            assert regime == "bearish"

    @pytest.mark.asyncio
    async def test_regime_case_insensitive(self):
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = '{"regime": "NEUTRAL", "reasoning": "test"}'
            regime = await get_regime(0.0, 0.0, 0.0, 20.0, 50.0)
            assert regime == "neutral"

    @pytest.mark.asyncio
    async def test_regime_whitespace_trimmed(self):
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock) as mock_claude:
            mock_claude.return_value = '{"regime": "  bullish  ", "reasoning": "test"}'
            regime = await get_regime(0.0, 0.0, 0.0, 20.0, 50.0)
            assert regime == "bullish"
