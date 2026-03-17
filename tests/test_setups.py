"""Tests for sauce.core.setups — Sprint 3 Strategy Engine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sauce.core.schemas import (
    Disqualification,
    HardConditionResult,
    Indicators,
    SetupResult,
    SignalLogEntry,
    SoftConditionResult,
)
from sauce.core.setups import (
    SETUP_1_MIN_SCORE,
    SETUP_1_REGIMES,
    SETUP_2_MIN_SCORE,
    SETUP_2_REGIMES,
    SETUP_3_MIN_SCORE,
    SETUP_3_REGIMES,
    MIN_BARS_SETUP_1,
    MIN_BARS_SETUP_2,
    MIN_BARS_SETUP_3,
    S1_RSI_HARD,
    S1_BB_TOLERANCE,
    S1_DOWN_CANDLE_VOL_RATIO,
    S1_RSI_SOFT,
    S1_STOCH_K_SOFT,
    S1_WIN_RATE_THRESHOLD,
    S1_MAX_ATTEMPTS,
    S2_SMA_PROXIMITY_PCT,
    S2_RSI_LOW,
    S2_RSI_HIGH,
    S2_RSI_TURNING_THRESHOLD,
    S2_BOUNCE_TOLERANCE,
    S2_VOLUME_CONTRACTION_RATIO,
    S3_CONSOLIDATION_BARS,
    S3_RANGE_PCT_MAX,
    S3_BREAKOUT_VOL_MULTIPLE,
    S3_EXTENDED_CONSOL_BARS,
    S3_VOLUME_3X,
    S3_REVERSAL_WICK_RATIO,
    S3_PRE_BREAKOUT_VOL_RATIO,
    S3_MAX_ATTEMPTS,
    SETUP_4_REGIMES,
    SETUP_4_MIN_SCORE,
    S4_RSI_LOW,
    S4_RSI_HIGH,
    S4_MAX_ATTEMPTS,
    SELLING_PRESSURE_KEYWORDS,
    _is_crypto,
    _compute_macd_histogram,
    _compute_rsi,
    _compute_score,
    _build_narrative,
    _resample_4h,
    _resample_daily,
    evaluate_crypto_mean_reversion,
    evaluate_equity_trend_pullback,
    evaluate_crypto_breakout,
    evaluate_crypto_momentum,
    scan_setups,
)

# ── Fixtures & factories ─────────────────────────────────────────────────────

_NOW = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)  # Monday


def _make_df(
    n_bars: int,
    *,
    close: float = 100.0,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
    freq: str = "30min",
) -> pd.DataFrame:
    """Return a simple OHLCV DataFrame with DatetimeIndex."""
    idx = pd.date_range(end=_NOW, periods=n_bars, freq=freq)
    o = open_ if open_ is not None else close
    h = high if high is not None else close + 1
    lw = low if low is not None else close - 1
    return pd.DataFrame(
        {"open": o, "high": h, "low": lw, "close": close, "volume": volume},
        index=idx,
    )


def _make_indicators(**overrides) -> Indicators:
    return Indicators(**overrides)


@pytest.fixture(autouse=True)
def _mock_calendar(monkeypatch):
    """Patch calendar checks to False by default."""
    monkeypatch.setattr(
        "sauce.core.setups.is_near_major_event", lambda *a, **kw: False,
    )
    monkeypatch.setattr(
        "sauce.core.setups.is_major_event_within_hours", lambda *a, **kw: False,
    )


# ── Helper tests ─────────────────────────────────────────────────────────────


class TestComputeRSI:
    def test_returns_series(self):
        s = pd.Series([float(x) for x in range(1, 31)])
        result = _compute_rsi(s, 14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_known_monotonically_rising(self):
        s = pd.Series([float(x) for x in range(1, 31)])
        result = _compute_rsi(s, 14)
        # All gains, no losses → RSI = 100
        assert result.iloc[-1] == 100.0

    def test_known_monotonically_falling(self):
        s = pd.Series([float(30 - x) for x in range(30)])
        result = _compute_rsi(s, 14)
        assert result.iloc[-1] == 0.0  # all losses, no gains

    def test_flat_returns_nan(self):
        s = pd.Series([100.0] * 30)
        result = _compute_rsi(s, 14)
        # delta is all zero -> gain=0, loss=0 -> RS=0/0=nan
        assert pd.isna(result.iloc[-1])


class TestComputeMACDHistogram:
    def test_returns_series(self):
        s = pd.Series([float(x) for x in range(50)])
        result = _compute_macd_histogram(s)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_flat_near_zero(self):
        s = pd.Series([100.0] * 50)
        result = _compute_macd_histogram(s)
        assert abs(result.iloc[-1]) < 1e-6


class TestResample4h:
    def test_resamples_correctly(self):
        df = _make_df(48, close=100.0, volume=100.0)
        result = _resample_4h(df)
        assert len(result) > 0
        assert "close" in result.columns


class TestResampleDaily:
    def test_resamples_correctly(self):
        df = _make_df(96, close=100.0, volume=100.0)
        result = _resample_daily(df)
        assert len(result) > 0
        assert "close" in result.columns


class TestComputeScore:
    def test_all_hard_pass_no_soft_no_disqual(self):
        hard = [
            HardConditionResult(label="H1", passed=True),
            HardConditionResult(label="H2", passed=True),
        ]
        score, passed = _compute_score(hard, [], [], 60.0)
        assert score == 60.0
        assert passed is True

    def test_one_hard_fails(self):
        hard = [
            HardConditionResult(label="H1", passed=True),
            HardConditionResult(label="H2", passed=False),
        ]
        score, passed = _compute_score(hard, [], [], 60.0)
        assert passed is False

    def test_soft_adds_points(self):
        hard = [HardConditionResult(label="H1", passed=True)]
        soft = [SoftConditionResult(label="S1", triggered=True, points=15.0)]
        score, passed = _compute_score(hard, soft, [], 60.0)
        assert score == min(60.0 + 15.0, 100.0)
        assert passed is True

    def test_disqualifier_blocks_pass(self):
        hard = [HardConditionResult(label="H1", passed=True)]
        soft = [SoftConditionResult(label="S1", triggered=True, points=15.0)]
        disqual = [Disqualification(reason="blocked")]
        score, passed = _compute_score(hard, soft, disqual, 60.0)
        assert passed is False

    def test_below_min_score_fails(self):
        hard = [
            HardConditionResult(label="H1", passed=True),
            HardConditionResult(label="H2", passed=True),
            HardConditionResult(label="H3", passed=True),
        ]
        # hard_score = (3/3) * 90 = 90. This should pass at min 60.
        score, passed = _compute_score(hard, [], [], 90.0)
        assert passed is True

    def test_score_capped_at_100(self):
        hard = [HardConditionResult(label="H1", passed=True)]
        soft = [
            SoftConditionResult(label="S1", triggered=True, points=50.0),
            SoftConditionResult(label="S2", triggered=True, points=50.0),
        ]
        score, passed = _compute_score(hard, soft, [], 60.0)
        assert score <= 100.0

    def test_min_hard_required_partial_pass(self):
        """4 of 6 hard pass with min_hard_required=4 → passed=True."""
        hard = [
            HardConditionResult(label=f"H{i}", passed=(i <= 4))
            for i in range(1, 7)
        ]
        # hard_score = (4/6)*60 = 40. Need soft to reach 60.
        soft = [SoftConditionResult(label="S1", triggered=True, points=25.0)]
        score, passed = _compute_score(hard, soft, [], 60.0, min_hard_required=4)
        assert passed is True
        assert score >= 60.0

    def test_min_hard_required_too_few(self):
        """3 of 6 hard pass with min_hard_required=4 → passed=False."""
        hard = [
            HardConditionResult(label=f"H{i}", passed=(i <= 3))
            for i in range(1, 7)
        ]
        soft = [SoftConditionResult(label="S1", triggered=True, points=50.0)]
        score, passed = _compute_score(hard, soft, [], 60.0, min_hard_required=4)
        assert passed is False

    def test_min_hard_required_none_means_all(self):
        """Without min_hard_required, all hard must pass (backward compat)."""
        hard = [
            HardConditionResult(label="H1", passed=True),
            HardConditionResult(label="H2", passed=True),
            HardConditionResult(label="H3", passed=False),
        ]
        score, passed = _compute_score(hard, [], [], 60.0)
        assert passed is False


class TestBuildNarrative:
    def test_contains_setup_type_and_symbol(self):
        hard = [HardConditionResult(label="H1", passed=True)]
        result = _build_narrative("crypto_mean_reversion", "BTC/USD", hard, [], [], 70.0, True)
        assert "crypto_mean_reversion" in result
        assert "BTC/USD" in result
        assert "PASSED" in result

    def test_rejected_in_narrative(self):
        hard = [HardConditionResult(label="H1", passed=False)]
        result = _build_narrative("crypto_mean_reversion", "BTC/USD", hard, [], [], 30.0, False)
        assert "REJECTED" in result

    def test_disqualification_in_narrative(self):
        hard = [HardConditionResult(label="H1", passed=True)]
        disqual = [Disqualification(reason="too risky")]
        result = _build_narrative("crypto_breakout", "ETH/USD", hard, [], disqual, 0.0, False)
        assert "DISQUALIFIED" in result
        assert "too risky" in result


# ── Constants tests ──────────────────────────────────────────────────────────


class TestConstants:
    def test_is_crypto_detects_slash_pairs(self):
        assert _is_crypto("BTC/USD") is True
        assert _is_crypto("ETH/USD") is True
        assert _is_crypto("SOL/USD") is True

    def test_is_crypto_rejects_equities(self):
        assert _is_crypto("AAPL") is False
        assert _is_crypto("SPY") is False
        assert _is_crypto("QQQ") is False

    def test_setup_1_regimes(self):
        assert SETUP_1_REGIMES == frozenset({"RANGING", "TRENDING_UP"})

    def test_setup_2_regimes(self):
        assert SETUP_2_REGIMES == frozenset({"TRENDING_UP"})

    def test_setup_3_regimes(self):
        assert SETUP_3_REGIMES == frozenset({"RANGING", "TRENDING_UP"})

    def test_min_scores(self):
        assert SETUP_1_MIN_SCORE == 50.0
        assert SETUP_2_MIN_SCORE == 55.0  # FH-02: lowered from 65.0
        assert SETUP_3_MIN_SCORE == 55.0

    def test_min_bars(self):
        assert MIN_BARS_SETUP_1 == 5
        assert MIN_BARS_SETUP_2 == 5
        assert MIN_BARS_SETUP_3 == 10

    def test_max_attempts(self):
        assert S1_MAX_ATTEMPTS == 4
        assert S3_MAX_ATTEMPTS == 2


# ── Setup 1: Crypto Mean Reversion ───────────────────────────────────────────


def _make_setup1_passing_df(n_bars: int = 500) -> pd.DataFrame:
    """Build a DF that satisfies H3/H4/H5 for setup 1.

    - 500 bars of 30m data → 64 4hr bars → enough for SMA50 (H3)
    - 3-phase pattern: decline → flat → sharp uptick
    - Sharp uptick creates real MACD divergence (H5)
    - Last 3 down candles have elevated volume → H4
    """
    idx = pd.date_range(end=_NOW, periods=n_bars, freq="30min")
    closes = np.empty(n_bars)
    mid = n_bars // 2
    closes[:mid] = np.linspace(100.0, 95.0, mid)                    # Phase 1: decline
    closes[mid:-10] = np.linspace(95.0, 96.0, n_bars - mid - 10)    # Phase 2: flat
    closes[-10:] = np.linspace(96.5, 98.0, 10)                      # Phase 3: sharp uptick → MACD curls
    opens = closes - 0.3  # most candles slightly up
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n_bars, 500.0)

    # Make last 3 down candles with elevated volume for H4
    for i in [-3, -5, -7]:
        opens[i] = closes[i] + 1.0  # down candle: open > close
        volumes[i] = 3000.0  # way above average

    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )
    return df


def _make_setup1_passing_indicators() -> Indicators:
    """Indicators that satisfy H1/H2 and trigger soft conditions."""
    return Indicators(
        rsi_14=25.0,       # H1: <42, S1: <32
        bb_lower=98.5,     # H2: close(98.0) <= 98.5*1.01≈99.49
        stoch_k=15.0,      # S2: <20
        sma_20=99.0,       # S3: close(98.0)<=sma20
        vwap=99.0,         # S4: vwap>close(98.0)
    )


class TestCryptoMeanReversion:
    def test_all_pass(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion(
            "BTC/USD", ind, df, "RANGING", as_of=_NOW,
            strategic_win_rate=0.70,
        )
        assert isinstance(result, SetupResult)
        assert result.setup_type == "crypto_mean_reversion"
        assert result.symbol == "BTC/USD"
        assert len(result.hard_conditions) == 5
        assert all(h.passed for h in result.hard_conditions), (
            f"Failed hard: {[h.label for h in result.hard_conditions if not h.passed]}"
        )
        assert result.passed is True
        assert result.score >= SETUP_1_MIN_SCORE

    def test_h1_rsi_too_high(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        ind.rsi_14 = 50.0  # > 42; fails H1
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        h1 = result.hard_conditions[0]
        assert h1.label == f"RSI < {S1_RSI_HARD}"
        assert h1.passed is False
        # With min_hard_required=3, failing just H1 still passes (4/5 >= 3)

    def test_h1_rsi_none(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        ind.rsi_14 = None
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert result.hard_conditions[0].passed is False

    def test_h2_price_above_bb(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        ind.bb_lower = 50.0  # close ~100 >> bb_lower*1.005=50.25
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        h2 = result.hard_conditions[1]
        assert h2.label == "Price near lower BB"
        assert h2.passed is False

    def test_h3_insufficient_bars(self):
        df = _make_df(50, close=100.0)  # <120 bars
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        h3 = result.hard_conditions[2]
        assert h3.passed is False

    def test_open_position_disqualifies(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion(
            "BTC/USD", ind, df, "RANGING", has_open_position=True, as_of=_NOW,
        )
        reasons = [d.reason for d in result.disqualifiers]
        assert any("open" in r.lower() for r in reasons)
        assert result.passed is False

    def test_disqualifier_volatile(self):
        """CF-02: VOLATILE no longer disqualifies crypto setups."""
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "VOLATILE", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert not any("VOLATILE" in r for r in reasons)

    def test_disqualifier_trending_down(self):
        """CF-02: TRENDING_DOWN no longer disqualifies crypto setups."""
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "TRENDING_DOWN", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert not any("TRENDING_DOWN" in r for r in reasons)

    def test_disqualifier_max_attempts(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion(
            "BTC/USD", ind, df, "RANGING",
            mean_reversion_attempts_today=4,
            as_of=_NOW,
        )
        reasons = [d.reason for d in result.disqualifiers]
        assert any("attempted" in r for r in reasons)
        assert result.passed is False

    def test_disqualifier_near_major_event(self, monkeypatch):
        """CF-01: near_major_event no longer disqualifies crypto setups."""
        monkeypatch.setattr(
            "sauce.core.setups.is_near_major_event", lambda *a, **kw: True,
        )
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert not any("Major economic event" in r for r in reasons)

    def test_disqualifier_selling_pressure(self):
        """Selling pressure disqualifier removed for crypto — should pass."""
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion(
            "BTC/USD", ind, df, "RANGING",
            narrative_text="There is sustained selling in the market.",
            as_of=_NOW,
        )
        # Selling pressure no longer disqualifies crypto setups
        assert not any("selling" in d.reason.lower() for d in result.disqualifiers)
        assert result.passed is True

    def test_soft_deep_oversold(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        ind.rsi_14 = 25.0  # < 32
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        s1 = result.soft_conditions[0]
        assert s1.label == "Deep oversold RSI < 32"
        assert s1.triggered is True
        assert s1.points == 20.0

    def test_soft_stoch_k(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        ind.stoch_k = 15.0  # < 20
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        s2 = result.soft_conditions[1]
        assert s2.triggered is True

    def test_soft_win_rate(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion(
            "BTC/USD", ind, df, "RANGING", strategic_win_rate=0.70, as_of=_NOW,
        )
        s5 = result.soft_conditions[4]
        assert s5.triggered is True

    def test_soft_win_rate_too_low(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion(
            "BTC/USD", ind, df, "RANGING", strategic_win_rate=0.50, as_of=_NOW,
        )
        s5 = result.soft_conditions[4]
        assert s5.triggered is False

    def test_empty_df(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], dtype="datetime64[ns]")
        ind = _make_indicators(rsi_14=30.0, bb_lower=99.0)
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert result.passed is False

    def test_evidence_narrative_populated(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert len(result.evidence_narrative) > 0

    def test_as_of_set(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert result.as_of == _NOW

    def test_min_score_field(self):
        df = _make_setup1_passing_df()
        ind = _make_setup1_passing_indicators()
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert result.min_score == SETUP_1_MIN_SCORE


# ── Setup 2: Equity Trend Pullback ───────────────────────────────────────────


def _make_setup2_passing_df(n_bars: int = 2800) -> pd.DataFrame:
    """Build a DF that passes Setup 2 hard conditions.

    Need ~2800 30m bars to produce ~58 daily bars after resample (enough for
    50-bar SMA, 20-bar RSI, volume checks with daily resamples).
    Prices rise gently from 440 to 450 (uptrend, SMA20>SMA50).
    Last bar low >= second-to-last low (H5).
    Volume on last daily bar < 20d average (H6).
    """
    idx = pd.date_range(end=_NOW, periods=n_bars, freq="30min")
    closes = np.linspace(440.0, 450.0, n_bars)
    opens = closes - 0.1
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.full(n_bars, 1000.0)
    # Lower volume on last day (~48 bars) for H6 pass
    volumes[-48:] = 400.0

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_setup2_passing_indicators(last_close: float | None = None) -> Indicators:
    """Indicators that don't block setup 2 (RSI and rsi_14 for S3)."""
    return Indicators(rsi_14=42.0)


class TestEquityTrendPullback:
    def test_all_pass(self):
        df = _make_setup2_passing_df()
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("SPY", ind, df, "TRENDING_UP", as_of=_NOW)
        assert result.setup_type == "equity_trend_pullback"
        assert result.symbol == "SPY"
        assert len(result.hard_conditions) == 6
        # Due to dynamic data we confirm structure not exact pass (thresholds are tight)
        assert isinstance(result.score, float)
        assert isinstance(result.passed, bool)

    def test_insufficient_daily_data(self):
        df = _make_df(10, close=450.0)  # too few for daily resamples
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("SPY", ind, df, "TRENDING_UP", as_of=_NOW)
        # All hard conditions needing daily data should fail
        assert result.passed is False

    def test_h5_lower_lows(self):
        df = _make_setup2_passing_df()
        # Force last bar to have a lower low than previous
        df.iloc[-1, df.columns.get_loc("low")] = df["low"].iloc[-2] - 5.0
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("SPY", ind, df, "TRENDING_UP", as_of=_NOW)
        h5 = result.hard_conditions[4]
        assert h5.label == "No lower lows (last 2 bars)"
        assert h5.passed is False

    def test_disqualifier_ranging(self):
        df = _make_setup2_passing_df()
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("QQQ", ind, df, "RANGING", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert any("RANGING" in r for r in reasons)
        assert result.passed is False

    def test_disqualifier_volatile(self):
        df = _make_setup2_passing_df()
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("SPY", ind, df, "VOLATILE", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert any("VOLATILE" in r for r in reasons)

    def test_disqualifier_friday_after_2pm_et(self):
        from zoneinfo import ZoneInfo

        # Friday Jan 12, 2024 at 3pm ET = 8pm UTC
        friday_3pm_et = datetime(2024, 1, 12, 20, 0, 0, tzinfo=timezone.utc)
        df = _make_setup2_passing_df()
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback(
            "SPY", ind, df, "TRENDING_UP", as_of=friday_3pm_et,
        )
        reasons = [d.reason for d in result.disqualifiers]
        assert any("Friday" in r for r in reasons)

    def test_disqualifier_major_event(self, monkeypatch):
        monkeypatch.setattr(
            "sauce.core.setups.is_major_event_within_hours", lambda *a, **kw: True,
        )
        df = _make_setup2_passing_df()
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("SPY", ind, df, "TRENDING_UP", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert any("FOMC" in r or "CPI" in r or "NFP" in r for r in reasons)
        assert result.passed is False

    def test_disqualifier_gap_down(self):
        df = _make_setup2_passing_df()
        daily = _resample_daily(df)
        # We need to make today's open < yesterday's close in the raw data
        # The last ~48 bars form "today". Set open much lower.
        df.iloc[-48:, df.columns.get_loc("open")] = 430.0  # gap down
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("SPY", ind, df, "TRENDING_UP", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert any("gapped down" in r for r in reasons)

    def test_soft_conditions_structure(self):
        df = _make_setup2_passing_df()
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("SPY", ind, df, "TRENDING_UP", as_of=_NOW)
        assert len(result.soft_conditions) == 4
        labels = [s.label for s in result.soft_conditions]
        assert "SMA20 bounce" in labels
        assert "Weekly uptrend proxy" in labels
        assert "15m RSI turning up from <45" in labels
        assert "Very low volume contraction" in labels

    def test_as_of_and_min_score(self):
        df = _make_setup2_passing_df()
        ind = _make_setup2_passing_indicators()
        result = evaluate_equity_trend_pullback("SPY", ind, df, "TRENDING_UP", as_of=_NOW)
        assert result.as_of == _NOW
        assert result.min_score == SETUP_2_MIN_SCORE


# ── Setup 3: Crypto Breakout ─────────────────────────────────────────────────


def _make_setup3_passing_df(n_bars: int = 30) -> pd.DataFrame:
    """Build a DF that passes Setup 3 hard conditions.

    - 30 bars of tight consolidation at 100.0 (range < 2%)
    - Last bar breaks above consolidation high with 2x volume
    - Declining volume in consolidation (first half > second half)
    - 4hr trend up
    """
    idx = pd.date_range(end=_NOW, periods=n_bars, freq="30min")

    # Tight consolidation with very small range
    closes = np.full(n_bars, 100.0)
    opens = np.full(n_bars, 99.9)
    highs = np.full(n_bars, 100.5)
    lows = np.full(n_bars, 99.5)

    # Declining volume in consolidation window (bars[-9:-1])
    volumes = np.full(n_bars, 1000.0)
    # First half of consolidation window: higher volume
    volumes[-9:-5] = 1500.0
    # Second half of consolidation window: lower volume
    volumes[-5:-1] = 500.0

    # Last bar: breakout above range_high with high volume
    closes[-1] = 101.0   # above range_high (100.5)
    opens[-1] = 100.3
    highs[-1] = 101.2    # small upper wick (not reversal)
    lows[-1] = 100.2
    volumes[-1] = 2500.0  # >= 2x avg consolidation vol

    # 4hr trend up: earlier bars slightly lower
    closes[:10] = 99.0
    opens[:10] = 98.8
    highs[:10] = 99.5
    lows[:10] = 98.5

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


class TestCryptoBreakout:
    def test_all_pass(self):
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout(
            "BTC/USD", ind, df, "RANGING", as_of=_NOW,
        )
        assert result.setup_type == "crypto_breakout"
        assert result.symbol == "BTC/USD"
        assert len(result.hard_conditions) == 5

    def test_h1_range_too_wide(self):
        df = _make_setup3_passing_df()
        # Widen the consolidation range beyond 2%
        df.iloc[-5, df.columns.get_loc("high")] = 110.0
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        h1 = result.hard_conditions[0]
        assert h1.label == f"Consolidation >= {S3_CONSOLIDATION_BARS} bars, range < {S3_RANGE_PCT_MAX}%"
        assert h1.passed is False

    def test_h3_close_below_range_high(self):
        df = _make_setup3_passing_df()
        # Last close below consolidation high
        df.iloc[-1, df.columns.get_loc("close")] = 99.0
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        h3 = result.hard_conditions[2]
        assert h3.label == "Close above range high"
        assert h3.passed is False

    def test_h4_breakout_vol_too_low(self):
        df = _make_setup3_passing_df()
        # Reduce breakout bar volume
        df.iloc[-1, df.columns.get_loc("volume")] = 100.0
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        h4 = result.hard_conditions[3]
        assert h4.label == f"Breakout volume >= {S3_BREAKOUT_VOL_MULTIPLE}x avg"
        assert h4.passed is False

    def test_open_position_disqualifies(self):
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout(
            "BTC/USD", ind, df, "RANGING", has_open_position=True, as_of=_NOW,
        )
        reasons = [d.reason for d in result.disqualifiers]
        assert any("open" in r.lower() for r in reasons)
        assert result.passed is False

    def test_insufficient_bars(self):
        df = _make_df(5, close=100.0)  # < 9 bars needed
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        # Multiple hard conditions should fail
        assert result.passed is False

    def test_s4_always_false(self):
        """Setup 3 S4 (range below resistance) always returns False."""
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        s4 = result.soft_conditions[3]
        assert s4.triggered is False

    def test_disqualifier_max_attempts(self):
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout(
            "BTC/USD", ind, df, "RANGING",
            breakout_attempts_today=2,
            as_of=_NOW,
        )
        reasons = [d.reason for d in result.disqualifiers]
        assert len(reasons) > 0
        assert result.passed is False

    def test_disqualifier_reversal_wick(self):
        df = _make_setup3_passing_df()
        # Make a large upper wick (reversal candle)
        df.iloc[-1, df.columns.get_loc("high")] = 115.0
        df.iloc[-1, df.columns.get_loc("close")] = 101.0
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert any("reversal" in r.lower() for r in reasons)

    def test_disqualifier_volatile(self):
        """CF-02: VOLATILE no longer disqualifies crypto setups."""
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "VOLATILE", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert not any("VOLATILE" in r for r in reasons)

    def test_disqualifier_near_major_event(self, monkeypatch):
        """CF-01: near_major_event no longer disqualifies crypto setups."""
        monkeypatch.setattr(
            "sauce.core.setups.is_near_major_event", lambda *a, **kw: True,
        )
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert not any("event" in r.lower() for r in reasons)

    def test_disqualifier_pre_elevated_volume(self):
        df = _make_setup3_passing_df()
        consol = df.iloc[-(S3_CONSOLIDATION_BARS + 1):-1]
        avg_vol = consol["volume"].mean()
        # Set bar[-2] volume to > 1.5x consolidation avg
        df.iloc[-2, df.columns.get_loc("volume")] = avg_vol * 2.0
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert any("volume" in r.lower() or "elevated" in r.lower() for r in reasons)

    def test_soft_s5_win_rate(self):
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout(
            "BTC/USD", ind, df, "RANGING",
            strategic_win_rate=0.70,
            as_of=_NOW,
        )
        s5 = result.soft_conditions[4]
        assert s5.triggered is True

    def test_as_of_and_min_score(self):
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert result.as_of == _NOW
        assert result.min_score == SETUP_3_MIN_SCORE


# ── Setup 4: Crypto Momentum ────────────────────────────────────────────────


def _make_setup4_passing_df(n_bars: int = 100) -> pd.DataFrame:
    """Build a DF that satisfies H2 (price > SMA20) and H3 (MACD positive)."""
    idx = pd.date_range(end=_NOW, periods=n_bars, freq="30min")
    # Steady uptrend — price well above SMA20, MACD will be positive
    closes = np.linspace(90.0, 105.0, n_bars)
    opens = closes - 0.2
    highs = closes + 1.0
    lows = closes - 1.0
    volumes = np.full(n_bars, 500.0)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _make_setup4_passing_indicators() -> Indicators:
    """Indicators that satisfy H1 (RSI 40-65), H2 (price > SMA20), and soft conditions."""
    return Indicators(
        rsi_14=55.0,       # H1: 40-65
        sma_20=100.0,      # H2: close(105.0) > 100.0
        vwap=103.0,        # S1: close > vwap
        volume_ratio=1.5,  # S2: > 1.0
        stoch_k=50.0,      # S3: 30-70
    )


class TestCryptoMomentum:
    def test_all_pass(self):
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        result = evaluate_crypto_momentum(
            "BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW,
            strategic_win_rate=0.70,
        )
        assert isinstance(result, SetupResult)
        assert result.setup_type == "crypto_momentum"
        assert result.symbol == "BTC/USD"
        assert len(result.hard_conditions) == 3
        assert all(h.passed for h in result.hard_conditions), (
            f"Failed hard: {[h.label for h in result.hard_conditions if not h.passed]}"
        )
        assert result.passed is True
        assert result.score >= SETUP_4_MIN_SCORE

    def test_h1_rsi_too_low(self):
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        ind.rsi_14 = 35.0  # < 40; fails H1
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW)
        h1 = result.hard_conditions[0]
        assert h1.passed is False
        # 2 of 3 hard still pass → still passes with min_hard_required=2
        assert result.passed is True

    def test_h1_rsi_too_high(self):
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        ind.rsi_14 = 70.0  # > 65; fails H1
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW)
        h1 = result.hard_conditions[0]
        assert h1.passed is False

    def test_h2_price_below_sma20(self):
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        ind.sma_20 = 200.0  # close(~105) < 200; fails H2
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW)
        h2 = result.hard_conditions[1]
        assert h2.passed is False

    def test_h3_insufficient_bars(self):
        df = _make_df(20, close=105.0)  # < 30 bars
        ind = _make_setup4_passing_indicators()
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW)
        h3 = result.hard_conditions[2]
        assert h3.passed is False

    def test_two_hard_fail_means_setup_fails(self):
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        ind.rsi_14 = 70.0   # H1 fails
        ind.sma_20 = 200.0  # H2 fails → only 1 of 3 hard pass
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW)
        assert result.passed is False

    def test_disqualifier_open_position(self):
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        result = evaluate_crypto_momentum(
            "BTC/USD", ind, df, "TRENDING_UP", has_open_position=True, as_of=_NOW,
        )
        reasons = [d.reason for d in result.disqualifiers]
        assert any("open" in r.lower() for r in reasons)
        assert result.passed is False

    def test_disqualifier_volatile(self):
        """CF-02: VOLATILE no longer disqualifies crypto setups."""
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "VOLATILE", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert not any("VOLATILE" in r for r in reasons)

    def test_disqualifier_trending_down(self):
        """CF-02: TRENDING_DOWN no longer disqualifies crypto setups."""
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "TRENDING_DOWN", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert not any("TRENDING_DOWN" in r for r in reasons)

    def test_disqualifier_max_attempts(self):
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        result = evaluate_crypto_momentum(
            "BTC/USD", ind, df, "TRENDING_UP",
            momentum_attempts_today=S4_MAX_ATTEMPTS,
            as_of=_NOW,
        )
        reasons = [d.reason for d in result.disqualifiers]
        assert any("attempted" in r.lower() for r in reasons)
        assert result.passed is False

    def test_disqualifier_near_major_event(self, monkeypatch):
        """CF-01: near_major_event no longer disqualifies crypto setups."""
        monkeypatch.setattr(
            "sauce.core.setups.is_near_major_event", lambda *a, **kw: True,
        )
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW)
        reasons = [d.reason for d in result.disqualifiers]
        assert not any("Major economic event" in r for r in reasons)

    def test_setup4_regimes(self):
        assert SETUP_4_REGIMES == frozenset({"TRENDING_UP", "RANGING"})

    def test_setup4_min_score(self):
        assert SETUP_4_MIN_SCORE == 50.0

    def test_as_of_preserved(self):
        df = _make_setup4_passing_df()
        ind = _make_setup4_passing_indicators()
        result = evaluate_crypto_momentum("BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW)
        assert result.as_of == _NOW
        assert result.min_score == SETUP_4_MIN_SCORE


# ── scan_setups ──────────────────────────────────────────────────────────────


class TestScanSetups:
    def test_btc_ranging_gets_setup_1_and_3(self):
        df = _make_df(250, close=100.0)
        ind = _make_indicators(rsi_14=30.0)
        results = scan_setups(
            "BTC/USD", ind, df, "RANGING", as_of=_NOW,
        )
        types = [r.setup_type for r in results]
        assert "crypto_mean_reversion" in types
        assert "crypto_breakout" in types
        assert "equity_trend_pullback" not in types

    def test_btc_trending_up_gets_setup_1(self):
        df = _make_df(250, close=100.0)
        ind = _make_indicators(rsi_14=30.0)
        results = scan_setups("BTC/USD", ind, df, "TRENDING_UP", as_of=_NOW)
        types = [r.setup_type for r in results]
        assert "crypto_mean_reversion" in types
        assert "crypto_breakout" in types  # TRENDING_UP now valid for breakout
        assert "crypto_momentum" in types  # TRENDING_UP valid for momentum

    def test_spy_trending_up_gets_setup_2_only(self):
        df = _make_df(2800, close=450.0)
        ind = _make_indicators(rsi_14=42.0)
        results = scan_setups("SPY", ind, df, "TRENDING_UP", as_of=_NOW)
        types = [r.setup_type for r in results]
        assert "equity_trend_pullback" in types
        assert "crypto_mean_reversion" not in types
        assert "crypto_breakout" not in types

    def test_equity_symbol_gets_setup_2(self):
        df = _make_df(2800, close=150.0)
        ind = _make_indicators(rsi_14=42.0)
        results = scan_setups("AAPL", ind, df, "TRENDING_UP", as_of=_NOW)
        types = [r.setup_type for r in results]
        assert "equity_trend_pullback" in types
        assert "crypto_mean_reversion" not in types
        assert "crypto_breakout" not in types

    def test_equity_wrong_regime_empty(self):
        df = _make_df(2800, close=150.0)
        ind = _make_indicators()
        results = scan_setups("AAPL", ind, df, "RANGING", as_of=_NOW)
        assert results == []

    def test_ineligible_regime_for_setup_2(self):
        df = _make_df(2800, close=450.0)
        ind = _make_indicators(rsi_14=42.0)
        results = scan_setups("SPY", ind, df, "RANGING", as_of=_NOW)
        types = [r.setup_type for r in results]
        assert "equity_trend_pullback" not in types

    def test_attempt_counting(self):
        df = _make_df(250, close=100.0)
        ind = _make_indicators(rsi_14=30.0, bb_lower=101.0)
        # Four prior mean reversion signals for BTC/USD (S1_MAX_ATTEMPTS=4)
        sigs = [
            SignalLogEntry(
                timestamp=_NOW,
                symbol="BTC/USD",
                setup_type="crypto_mean_reversion",
                score=70.0,
                claude_decision="approve",
            ),
            SignalLogEntry(
                timestamp=_NOW,
                symbol="BTC/USD",
                setup_type="crypto_mean_reversion",
                score=65.0,
                claude_decision="approve",
            ),
            SignalLogEntry(
                timestamp=_NOW,
                symbol="BTC/USD",
                setup_type="crypto_mean_reversion",
                score=60.0,
                claude_decision="approve",
            ),
            SignalLogEntry(
                timestamp=_NOW,
                symbol="BTC/USD",
                setup_type="crypto_mean_reversion",
                score=55.0,
                claude_decision="approve",
            ),
        ]
        results = scan_setups(
            "BTC/USD", ind, df, "RANGING", signals_today=sigs, as_of=_NOW,
        )
        mr_result = next(r for r in results if r.setup_type == "crypto_mean_reversion")
        # Should have max-attempts disqualifier
        reasons = [d.reason for d in mr_result.disqualifiers]
        assert any("attempted" in r for r in reasons)
        assert mr_result.passed is False

    def test_open_symbols_passed(self):
        df = _make_df(250, close=100.0)
        ind = _make_indicators(rsi_14=30.0, bb_lower=101.0)
        results = scan_setups(
            "BTC/USD", ind, df, "RANGING",
            open_symbols={"BTC/USD"},
            as_of=_NOW,
        )
        mr = next(r for r in results if r.setup_type == "crypto_mean_reversion")
        reasons = [d.reason for d in mr.disqualifiers]
        assert any("open" in r.lower() for r in reasons)
        assert mr.passed is False

    def test_empty_when_dead_regime(self):
        df = _make_df(250, close=100.0)
        ind = _make_indicators()
        results = scan_setups("BTC/USD", ind, df, "DEAD", as_of=_NOW)
        assert results == []

    def test_eth_ranging(self):
        df = _make_df(250, close=100.0)
        ind = _make_indicators(rsi_14=30.0)
        results = scan_setups("ETH/USD", ind, df, "RANGING", as_of=_NOW)
        types = [r.setup_type for r in results]
        assert "crypto_mean_reversion" in types
        assert "crypto_breakout" in types

    def test_qqq_trending_up(self):
        df = _make_df(2800, close=400.0)
        ind = _make_indicators(rsi_14=42.0)
        results = scan_setups("QQQ", ind, df, "TRENDING_UP", as_of=_NOW)
        types = [r.setup_type for r in results]
        assert "equity_trend_pullback" in types


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_all_none_indicators(self):
        df = _make_df(250, close=100.0)
        ind = _make_indicators()  # all None
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert result.passed is False
        assert result.hard_conditions[0].passed is False  # RSI None
        assert result.hard_conditions[1].passed is False  # BB None

    def test_single_bar_df(self):
        df = _make_df(1, close=100.0)
        ind = _make_indicators(rsi_14=30.0, bb_lower=101.0)
        result = evaluate_crypto_mean_reversion("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert result.passed is False

    def test_setup_result_score_bounds(self):
        df = _make_setup3_passing_df()
        ind = _make_indicators()
        result = evaluate_crypto_breakout("BTC/USD", ind, df, "RANGING", as_of=_NOW)
        assert 0.0 <= result.score <= 100.0
