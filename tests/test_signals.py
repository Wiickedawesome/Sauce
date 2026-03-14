"""Tests for sauce.signals — multi-timeframe analysis and confluence scoring."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sauce.core.schemas import Indicators
from sauce.signals.confluence import (
    TIMEFRAME_WEIGHTS,
    ConfluenceResult,
    SignalTier,
    _vote,
    compute_confluence,
)
from sauce.signals.timeframes import (
    TIMEFRAMES,
    MultiTimeframeContext,
    TimeframeAnalysis,
    _classify_momentum,
    _classify_trend,
    fetch_multi_timeframe,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ind(**overrides) -> Indicators:
    """Create Indicators with given overrides; everything else None."""
    return Indicators(**overrides)


def _analysis(
    label: str = "1h",
    timeframe: str = "1Hour",
    bars: int = 48,
    trend: str = "bullish",
    momentum: str = "bullish",
    **ind_kw,
) -> TimeframeAnalysis:
    return TimeframeAnalysis(
        label=label,
        timeframe=timeframe,
        bars_fetched=bars,
        indicators=_ind(**ind_kw),
        trend=trend,
        momentum=momentum,
    )


def _ohlcv(n: int = 60) -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame with DatetimeIndex."""
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2026-01-01", periods=n, freq="30min")
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.3, n),
            "high": close + rng.uniform(0, 0.5, n),
            "low": close - rng.uniform(0, 0.5, n),
            "close": close,
            "volume": rng.integers(1000, 5000, size=n).astype(float),
        },
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TREND CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyTrend:
    def test_bullish_when_sma20_above_sma50(self):
        assert _classify_trend(_ind(sma_20=110.0, sma_50=100.0)) == "bullish"

    def test_bearish_when_sma20_below_sma50(self):
        assert _classify_trend(_ind(sma_20=90.0, sma_50=100.0)) == "bearish"

    def test_neutral_when_equal(self):
        assert _classify_trend(_ind(sma_20=100.0, sma_50=100.0)) == "neutral"

    def test_unknown_when_sma20_none(self):
        assert _classify_trend(_ind(sma_50=100.0)) == "unknown"

    def test_unknown_when_sma50_none(self):
        assert _classify_trend(_ind(sma_20=100.0)) == "unknown"

    def test_unknown_when_both_none(self):
        assert _classify_trend(_ind()) == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# MOMENTUM CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyMomentum:
    def test_bullish_rsi_and_macd(self):
        assert _classify_momentum(_ind(rsi_14=60.0, macd_histogram=0.5)) == "bullish"

    def test_bearish_rsi_and_macd(self):
        assert _classify_momentum(_ind(rsi_14=40.0, macd_histogram=-0.5)) == "bearish"

    def test_neutral_when_mixed(self):
        # RSI bullish, MACD bearish → tied → neutral
        assert _classify_momentum(_ind(rsi_14=60.0, macd_histogram=-0.5)) == "neutral"

    def test_unknown_when_both_none(self):
        assert _classify_momentum(_ind()) == "unknown"

    def test_bullish_rsi_only(self):
        assert _classify_momentum(_ind(rsi_14=60.0)) == "bullish"

    def test_bearish_macd_only(self):
        assert _classify_momentum(_ind(macd_histogram=-0.3)) == "bearish"

    def test_neutral_rsi_in_middle(self):
        # RSI=50 → neither bull nor bear, MACD=None → unknown
        # 0 bull, 0 bear → neutral
        assert _classify_momentum(_ind(rsi_14=50.0)) == "neutral"


# ═══════════════════════════════════════════════════════════════════════════
# TimeframeAnalysis + MultiTimeframeContext
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeframeAnalysis:
    def test_frozen(self):
        a = _analysis()
        with pytest.raises(AttributeError):
            a.label = "5m"  # type: ignore[misc]


class TestMultiTimeframeContext:
    def test_empty_context(self):
        ctx = MultiTimeframeContext(symbol="AAPL")
        assert ctx.analyses == ()
        d = ctx.to_prompt_dict()
        assert d["symbol"] == "AAPL"
        assert d["timeframes"] == {}

    def test_to_prompt_dict_with_analyses(self):
        a = _analysis(
            label="1h",
            trend="bullish",
            momentum="bearish",
            sma_20=105.0,
            rsi_14=55.5,
        )
        ctx = MultiTimeframeContext(symbol="TSLA", analyses=(a,))
        d = ctx.to_prompt_dict()
        assert d["symbol"] == "TSLA"
        tf = d["timeframes"]["1h"]
        assert tf["trend"] == "bullish"
        assert tf["momentum"] == "bearish"
        assert tf["sma_20"] == 105.0
        assert tf["rsi_14"] == 55.5


# ═══════════════════════════════════════════════════════════════════════════
# fetch_multi_timeframe
# ═══════════════════════════════════════════════════════════════════════════

class TestFetchMultiTimeframe:
    def test_success_all_timeframes(self):
        df = _ohlcv(60)
        with patch("sauce.signals.timeframes.market_data") as md:
            md.get_history.return_value = df
            ctx = fetch_multi_timeframe("AAPL")

        assert ctx.symbol == "AAPL"
        assert len(ctx.analyses) == len(TIMEFRAMES)
        labels = {a.label for a in ctx.analyses}
        assert labels == {"5m", "15m", "1h", "4h", "1d"}

    def test_skips_empty_dataframe(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        with patch("sauce.signals.timeframes.market_data") as md:
            md.get_history.return_value = empty
            ctx = fetch_multi_timeframe("AAPL")

        assert len(ctx.analyses) == 0

    def test_skips_insufficient_bars(self):
        """Fewer than 20 bars → skip that timeframe."""
        small = _ohlcv(15)
        with patch("sauce.signals.timeframes.market_data") as md:
            md.get_history.return_value = small
            ctx = fetch_multi_timeframe("AAPL")

        assert len(ctx.analyses) == 0

    def test_handles_market_data_error(self):
        with patch("sauce.signals.timeframes.market_data") as md:
            md.MarketDataError = type("MarketDataError", (Exception,), {})
            md.get_history.side_effect = md.MarketDataError("API down")
            ctx = fetch_multi_timeframe("AAPL")

        assert len(ctx.analyses) == 0
        assert ctx.symbol == "AAPL"

    def test_handles_unexpected_error(self):
        with patch("sauce.signals.timeframes.market_data") as md:
            md.MarketDataError = type("MarketDataError", (Exception,), {})
            md.get_history.side_effect = ValueError("oops")
            ctx = fetch_multi_timeframe("AAPL")

        assert len(ctx.analyses) == 0

    def test_partial_failure(self):
        """Some timeframes succeed, some fail → partial context."""
        df = _ohlcv(60)
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("boom")
            return df

        with patch("sauce.signals.timeframes.market_data") as md:
            md.MarketDataError = type("MarketDataError", (Exception,), {})
            md.get_history.side_effect = side_effect
            ctx = fetch_multi_timeframe("NVDA")

        # First 2 fail, last 3 succeed
        assert len(ctx.analyses) == 3

    def test_crypto_flag_passed(self):
        """is_crypto propagates to compute_all."""
        df = _ohlcv(60)
        with (
            patch("sauce.signals.timeframes.market_data") as md,
            patch("sauce.signals.timeframes.compute_all") as ca,
        ):
            md.get_history.return_value = df
            ca.return_value = _ind()
            fetch_multi_timeframe("BTC/USD", is_crypto=True)

        # All calls should have is_crypto=True
        for call in ca.call_args_list:
            assert call.kwargs.get("is_crypto") is True


# ═══════════════════════════════════════════════════════════════════════════
# _vote
# ═══════════════════════════════════════════════════════════════════════════

class TestVote:
    def test_all_bullish(self):
        assert _vote("bullish", "bullish") == pytest.approx(1.0)

    def test_all_bearish(self):
        assert _vote("bearish", "bearish") == pytest.approx(-1.0)

    def test_mixed_cancels(self):
        assert _vote("bullish", "bearish") == pytest.approx(0.0)

    def test_neutral_trend_bullish_momentum(self):
        assert _vote("neutral", "bullish") == pytest.approx(0.5)

    def test_unknown_both(self):
        assert _vote("unknown", "unknown") == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# compute_confluence
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeConfluence:
    def test_empty_context_returns_s3(self):
        ctx = MultiTimeframeContext(symbol="AAPL")
        result = compute_confluence(ctx)
        assert result.tier == SignalTier.S3
        assert result.score == 0.0
        assert result.confidence_adjustment == 0.0
        assert result.bullish_count == 0

    def test_all_bullish_gives_s1(self):
        """≥3 aligned + score≥0.5 → S1."""
        analyses = tuple(
            _analysis(label=lbl, trend="bullish", momentum="bullish")
            for lbl in ("5m", "15m", "1h", "4h", "1d")
        )
        ctx = MultiTimeframeContext(symbol="AAPL", analyses=analyses)
        result = compute_confluence(ctx)
        assert result.tier == SignalTier.S1
        assert result.score == pytest.approx(1.0)
        assert result.confidence_adjustment == 0.10
        assert result.bullish_count == 5
        assert result.bearish_count == 0

    def test_all_bearish_gives_s1(self):
        """All bearish → S1 (high alignment, just bearish direction)."""
        analyses = tuple(
            _analysis(label=lbl, trend="bearish", momentum="bearish")
            for lbl in ("5m", "15m", "1h", "4h", "1d")
        )
        ctx = MultiTimeframeContext(symbol="SPY", analyses=analyses)
        result = compute_confluence(ctx)
        assert result.tier == SignalTier.S1
        assert result.score == pytest.approx(-1.0)
        assert result.confidence_adjustment == 0.10
        assert result.bearish_count == 5

    def test_two_bullish_moderate_gives_s2(self):
        """2 aligned + score≥0.3 → S2."""
        analyses = (
            _analysis(label="4h", trend="bullish", momentum="bullish"),
            _analysis(label="1d", trend="bullish", momentum="bullish"),
            _analysis(label="5m", trend="neutral", momentum="neutral"),
        )
        ctx = MultiTimeframeContext(symbol="GOOG", analyses=analyses)
        result = compute_confluence(ctx)
        assert result.tier == SignalTier.S2
        assert result.confidence_adjustment == 0.05

    def test_conflicting_gives_s4(self):
        """Bull+bear with low score → S4."""
        analyses = (
            _analysis(label="5m", trend="bullish", momentum="bullish"),
            _analysis(label="15m", trend="bearish", momentum="bearish"),
            _analysis(label="1h", trend="neutral", momentum="neutral"),
        )
        ctx = MultiTimeframeContext(symbol="META", analyses=analyses)
        result = compute_confluence(ctx)
        assert result.tier == SignalTier.S4
        assert result.confidence_adjustment == -0.10

    def test_mostly_neutral_gives_s3(self):
        """All neutral → S3."""
        analyses = tuple(
            _analysis(label=lbl, trend="neutral", momentum="neutral")
            for lbl in ("5m", "15m", "1h")
        )
        ctx = MultiTimeframeContext(symbol="QQQ", analyses=analyses)
        result = compute_confluence(ctx)
        assert result.tier == SignalTier.S3
        assert result.confidence_adjustment == 0.0

    def test_score_range(self):
        """Score is always in [-1, 1]."""
        analyses = tuple(
            _analysis(label=lbl, trend="bullish", momentum="bullish")
            for lbl in ("5m", "15m", "1h", "4h", "1d")
        )
        ctx = MultiTimeframeContext(symbol="X", analyses=analyses)
        result = compute_confluence(ctx)
        assert -1.0 <= result.score <= 1.0

    def test_summary_contains_tier(self):
        ctx = MultiTimeframeContext(
            symbol="X",
            analyses=(_analysis(label="1h"),),
        )
        result = compute_confluence(ctx)
        assert result.tier.value in result.summary

    def test_to_prompt_dict(self):
        result = ConfluenceResult(
            score=0.75,
            tier=SignalTier.S1,
            bullish_count=4,
            bearish_count=0,
            neutral_count=1,
            confidence_adjustment=0.10,
            summary="Strong bullish",
        )
        d = result.to_prompt_dict()
        assert d["confluence_score"] == 0.75
        assert d["signal_tier"] == "S1"
        assert d["bullish_timeframes"] == 4
        assert d["confidence_adjustment"] == 0.10


# ═══════════════════════════════════════════════════════════════════════════
# TIMEFRAME_WEIGHTS sanity
# ═══════════════════════════════════════════════════════════════════════════

class TestTimeframeWeights:
    def test_weights_sum_to_one(self):
        assert sum(TIMEFRAME_WEIGHTS.values()) == pytest.approx(1.0)

    def test_all_timeframes_have_weights(self):
        labels = {label for label, _, _ in TIMEFRAMES}
        assert labels == set(TIMEFRAME_WEIGHTS.keys())
