"""Tests for sauce.core.regime — market regime classifier + duration tracker."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sauce.core.regime import (
    AGING_THRESHOLD,
    DEAD_ATR_PCT_MAX,
    DEAD_VOLUME_RATIO_MAX,
    MIN_BARS,
    RegimeDuration,
    _atr_pct,
    _compute_vix_proxy,
    _detect_swing_structure,
    _sma_spread_pct,
    _volume_direction_bias,
    classify_regime,
    compute_regime_duration,
)
from sauce.core.schemas import (
    Indicators,
    RegimeLogEntry,
    RegimeTransitionEntry,
)

NOW = datetime(2025, 6, 15, 14, 30, 0, tzinfo=timezone.utc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bars(
    n: int = 20,
    base_close: float = 100.0,
    spread: float = 1.0,
    volume: float = 1000.0,
    trend: float = 0.0,
) -> pd.DataFrame:
    """
    Build a synthetic OHLCV DataFrame.

    trend > 0 → ascending prices (bullish)
    trend < 0 → descending prices (bearish)
    trend = 0 → flat / ranging
    spread controls high-low range (volatility proxy).
    """
    rows: list[dict] = []
    for i in range(n):
        c = base_close + trend * i
        rows.append({
            "open": c - spread * 0.1,
            "high": c + spread,
            "low": c - spread,
            "close": c + spread * 0.1,
            "volume": volume,
            "timestamp": NOW - timedelta(minutes=30 * (n - i)),
        })
    return pd.DataFrame(rows)


def _make_indicators(
    sma_20: float | None = None,
    sma_50: float | None = None,
    rsi_14: float | None = None,
    atr_14: float | None = None,
    volume_ratio: float | None = None,
    **kwargs: float | None,
) -> Indicators:
    return Indicators(
        sma_20=sma_20, sma_50=sma_50, rsi_14=rsi_14,
        atr_14=atr_14, volume_ratio=volume_ratio, **kwargs,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeVixProxy:
    def test_returns_zero_on_too_few_bars(self) -> None:
        df = _make_bars(n=1)
        assert _compute_vix_proxy(df) == 0.0

    def test_returns_positive_for_normal_data(self) -> None:
        df = _make_bars(n=20, base_close=100.0, spread=1.0)
        result = _compute_vix_proxy(df)
        assert result > 0.0

    def test_higher_spread_yields_higher_proxy(self) -> None:
        low = _compute_vix_proxy(_make_bars(spread=0.5))
        high = _compute_vix_proxy(_make_bars(spread=3.0))
        assert high > low


class TestDetectSwingStructure:
    def test_higher_highs_in_uptrend(self) -> None:
        df = _make_bars(n=20, trend=1.0)
        hh, ll = _detect_swing_structure(df)
        assert hh is True
        assert ll is False

    def test_lower_lows_in_downtrend(self) -> None:
        df = _make_bars(n=20, trend=-1.0)
        hh, ll = _detect_swing_structure(df)
        assert hh is False
        assert ll is True

    def test_flat_returns_both_false(self) -> None:
        df = _make_bars(n=20, trend=0.0)
        hh, ll = _detect_swing_structure(df)
        assert hh is False
        assert ll is False

    def test_too_few_bars_returns_both_false(self) -> None:
        df = _make_bars(n=3)
        hh, ll = _detect_swing_structure(df)
        assert hh is False
        assert ll is False


class TestVolumeDirectionBias:
    def test_all_up_bars_yield_positive(self) -> None:
        df = _make_bars(n=10, trend=1.0)
        # All bars close > open due to positive trend + spread
        bias = _volume_direction_bias(df)
        assert bias > 0.0

    def test_too_few_bars(self) -> None:
        df = _make_bars(n=1)
        assert _volume_direction_bias(df) == 0.0

    def test_zero_volume_returns_zero(self) -> None:
        df = _make_bars(n=10, volume=0.0)
        assert _volume_direction_bias(df) == 0.0


class TestSmaSpreadPct:
    def test_none_sma_returns_zero(self) -> None:
        ind = _make_indicators(sma_20=None, sma_50=100.0)
        assert _sma_spread_pct(ind) == 0.0

    def test_equal_sma_returns_zero(self) -> None:
        ind = _make_indicators(sma_20=100.0, sma_50=100.0)
        assert _sma_spread_pct(ind) == 0.0

    def test_diverging_sma_returns_positive(self) -> None:
        ind = _make_indicators(sma_20=105.0, sma_50=95.0)
        result = _sma_spread_pct(ind)
        assert result > 0.0


class TestAtrPct:
    def test_none_atr(self) -> None:
        ind = _make_indicators(atr_14=None)
        assert _atr_pct(ind, 100.0) == 0.0

    def test_positive_atr(self) -> None:
        ind = _make_indicators(atr_14=2.0)
        result = _atr_pct(ind, 100.0)
        assert result == pytest.approx(2.0)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFY REGIME
# ══════════════════════════════════════════════════════════════════════════════


class TestClassifyRegime:
    def test_insufficient_data_returns_dead(self) -> None:
        df = _make_bars(n=3)
        ind = _make_indicators()
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "DEAD"
        assert result.confidence == 0.0
        assert result.market_bias == "insufficient data"

    def test_dead_regime_low_volume_low_atr(self) -> None:
        df = _make_bars(n=20, spread=0.1)
        ind = _make_indicators(volume_ratio=0.1, atr_14=0.1)
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "DEAD"
        assert result.confidence > 0.0
        assert result.vix_proxy is not None

    def test_volatile_regime_high_vix(self) -> None:
        # High spread → high VIX proxy (>3%)
        df = _make_bars(n=20, base_close=100.0, spread=5.0)
        ind = _make_indicators(volume_ratio=1.0, atr_14=3.0)
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "VOLATILE"
        assert result.confidence >= 0.5

    def test_volatile_regime_high_atr_and_volume(self) -> None:
        df = _make_bars(n=20, base_close=100.0, spread=1.0)
        ind = _make_indicators(volume_ratio=3.0, atr_14=3.0)
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "VOLATILE"

    def test_trending_up(self) -> None:
        df = _make_bars(n=20, trend=2.0, base_close=100.0)
        ind = _make_indicators(
            sma_20=120.0, sma_50=100.0, rsi_14=60.0,
            volume_ratio=1.0, atr_14=1.0,
        )
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "TRENDING_UP"
        assert result.confidence >= 0.5
        assert result.market_bias == "bullish momentum"

    def test_trending_down(self) -> None:
        df = _make_bars(n=20, trend=-2.0, base_close=100.0)
        ind = _make_indicators(
            sma_20=80.0, sma_50=100.0, rsi_14=30.0,
            volume_ratio=1.0, atr_14=1.0,
        )
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "TRENDING_DOWN"
        assert result.confidence >= 0.5
        assert result.market_bias == "bearish pressure"

    def test_ranging_default(self) -> None:
        df = _make_bars(n=20, trend=0.0, spread=0.5)
        ind = _make_indicators(
            sma_20=100.0, sma_50=100.0, rsi_14=50.0,
            volume_ratio=0.5, atr_14=0.5,
        )
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "RANGING"
        assert result.market_bias == "sideways consolidation"

    def test_returns_valid_regime_log_entry(self) -> None:
        df = _make_bars(n=20)
        ind = _make_indicators(volume_ratio=1.0, atr_14=1.0)
        result = classify_regime(df, ind, NOW)
        assert isinstance(result, RegimeLogEntry)
        assert result.timestamp == NOW
        assert 0.0 <= result.confidence <= 1.0
        assert result.vix_proxy is not None

    def test_dead_takes_priority_over_trending(self) -> None:
        """Even with trend structure, DEAD wins if volume/ATR are extremely low."""
        df = _make_bars(n=20, trend=1.0, spread=0.05)
        ind = _make_indicators(
            sma_20=110.0, sma_50=100.0,
            volume_ratio=0.05, atr_14=0.05,
        )
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "DEAD"

    def test_volatile_takes_priority_over_trending(self) -> None:
        """Even with trend structure, VOLATILE wins if VIX proxy is extreme."""
        df = _make_bars(n=20, trend=1.0, base_close=100.0, spread=6.0)
        ind = _make_indicators(
            sma_20=110.0, sma_50=100.0,
            volume_ratio=1.0, atr_14=1.0,
        )
        result = classify_regime(df, ind, NOW)
        assert result.regime_type == "VOLATILE"


# ══════════════════════════════════════════════════════════════════════════════
# REGIME DURATION TRACKER
# ══════════════════════════════════════════════════════════════════════════════


class TestComputeRegimeDuration:
    def test_empty_history_returns_none(self) -> None:
        result = compute_regime_duration([], [], NOW)
        assert result is None

    def test_single_entry_duration_zero(self) -> None:
        history = [
            RegimeLogEntry(timestamp=NOW, regime_type="RANGING", confidence=0.7),
        ]
        result = compute_regime_duration(history, [], NOW)
        assert result is not None
        assert result.regime_type == "RANGING"
        assert result.active_minutes == 0.0
        assert result.historical_avg_minutes is None
        assert result.aging_out is False

    def test_same_regime_multi_entry_computes_duration(self) -> None:
        t0 = NOW - timedelta(hours=2)
        t1 = NOW - timedelta(hours=1)
        t2 = NOW - timedelta(minutes=30)
        history = [
            RegimeLogEntry(timestamp=t0, regime_type="RANGING", confidence=0.7),
            RegimeLogEntry(timestamp=t1, regime_type="RANGING", confidence=0.8),
            RegimeLogEntry(timestamp=t2, regime_type="RANGING", confidence=0.75),
        ]
        result = compute_regime_duration(history, [], NOW)
        assert result is not None
        assert result.active_minutes == pytest.approx(120.0, abs=1.0)

    def test_regime_change_resets_start(self) -> None:
        t0 = NOW - timedelta(hours=3)
        t1 = NOW - timedelta(hours=2)
        t2 = NOW - timedelta(hours=1)
        history = [
            RegimeLogEntry(timestamp=t0, regime_type="TRENDING_UP", confidence=0.7),
            RegimeLogEntry(timestamp=t1, regime_type="RANGING", confidence=0.8),
            RegimeLogEntry(timestamp=t2, regime_type="RANGING", confidence=0.7),
        ]
        result = compute_regime_duration(history, [], NOW)
        assert result is not None
        assert result.regime_type == "RANGING"
        # Duration should be from t1 (when RANGING started), not t0
        assert result.active_minutes == pytest.approx(120.0, abs=1.0)

    def test_aging_out_when_above_threshold(self) -> None:
        t0 = NOW - timedelta(hours=2)
        history = [
            RegimeLogEntry(timestamp=t0, regime_type="RANGING", confidence=0.7),
        ]
        # Historical: RANGING → X transitions average 100 minutes
        transitions = [
            RegimeTransitionEntry(
                from_regime="RANGING", to_regime="TRENDING_UP",
                duration_minutes=100.0, count=5,
            ),
        ]
        result = compute_regime_duration(history, transitions, NOW)
        assert result is not None
        # 120 min / 100 min avg = 1.2 > AGING_THRESHOLD (0.8)
        assert result.aging_out is True
        assert result.aging_ratio > AGING_THRESHOLD

    def test_not_aging_out_when_below_threshold(self) -> None:
        t0 = NOW - timedelta(minutes=30)
        history = [
            RegimeLogEntry(timestamp=t0, regime_type="RANGING", confidence=0.7),
        ]
        transitions = [
            RegimeTransitionEntry(
                from_regime="RANGING", to_regime="TRENDING_UP",
                duration_minutes=200.0, count=10,
            ),
        ]
        result = compute_regime_duration(history, transitions, NOW)
        assert result is not None
        # 30 min / 200 min avg = 0.15 < AGING_THRESHOLD (0.8)
        assert result.aging_out is False
        assert result.aging_ratio < AGING_THRESHOLD

    def test_historical_avg_computed_correctly(self) -> None:
        t0 = NOW - timedelta(hours=1)
        history = [
            RegimeLogEntry(timestamp=t0, regime_type="VOLATILE", confidence=0.9),
        ]
        transitions = [
            RegimeTransitionEntry(
                from_regime="VOLATILE", to_regime="RANGING",
                duration_minutes=60.0, count=3,
            ),
            RegimeTransitionEntry(
                from_regime="VOLATILE", to_regime="TRENDING_DOWN",
                duration_minutes=90.0, count=1,
            ),
        ]
        result = compute_regime_duration(history, transitions, NOW)
        assert result is not None
        # Weighted avg: (60*3 + 90*1) / (3+1) = 270/4 = 67.5
        assert result.historical_avg_minutes == pytest.approx(67.5, abs=0.1)

    def test_no_matching_transitions_returns_none_avg(self) -> None:
        t0 = NOW - timedelta(hours=1)
        history = [
            RegimeLogEntry(timestamp=t0, regime_type="DEAD", confidence=0.5),
        ]
        # Only transitions FROM different regimes
        transitions = [
            RegimeTransitionEntry(
                from_regime="RANGING", to_regime="TRENDING_UP",
                duration_minutes=100.0, count=5,
            ),
        ]
        result = compute_regime_duration(history, transitions, NOW)
        assert result is not None
        assert result.historical_avg_minutes is None
        assert result.aging_out is False

    def test_returns_regime_duration_model(self) -> None:
        history = [
            RegimeLogEntry(timestamp=NOW, regime_type="TRENDING_UP", confidence=0.8),
        ]
        result = compute_regime_duration(history, [], NOW)
        assert isinstance(result, RegimeDuration)

    def test_mixed_naive_and_aware_timestamps_are_normalized(self) -> None:
        naive_start = datetime(2025, 6, 15, 13, 30, 0)
        history = [
            RegimeLogEntry(timestamp=naive_start, regime_type="RANGING", confidence=0.7),
        ]

        result = compute_regime_duration(history, [], NOW)

        assert result is not None
        assert result.active_minutes == pytest.approx(60.0, abs=1.0)
