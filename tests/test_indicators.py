"""
tests/test_indicators.py — Indicator library tests.

Verifies pure indicator functions return correct values, handle edge cases
(empty DataFrames, insufficient bars, constant price), and that compute_all
produces a valid Indicators schema.
"""

import numpy as np
import pandas as pd

from sauce.core.schemas import Indicators
from sauce.indicators.core import (
    _last_float,
    compute_all,
    compute_atr,
    compute_bbands,
    compute_macd,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    compute_volume_1d_avg,
    compute_volume_ratio,
    compute_vwap,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ohlcv(n: int = 100, *, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with *n* bars and DatetimeIndex."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    opn = close + rng.standard_normal(n) * 0.2
    volume = rng.uniform(1000, 5000, n)
    idx = pd.date_range("2026-01-01", periods=n, freq="30min")
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


EMPTY_DF = pd.DataFrame(
    columns=["open", "high", "low", "close", "volume"],
    index=pd.DatetimeIndex([], name="timestamp"),
)
TINY_DF = _ohlcv(5)  # Too few bars for most indicators
STD_DF = _ohlcv(100)


# ── _last_float ──────────────────────────────────────────────────────────────


class TestLastFloat:
    def test_valid_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        assert _last_float(s) == 3.0

    def test_series_with_nans(self):
        s = pd.Series([1.0, 2.0, float("nan")])
        assert _last_float(s) == 2.0

    def test_all_nan(self):
        s = pd.Series([float("nan"), float("nan")])
        assert _last_float(s) is None

    def test_empty_series(self):
        assert _last_float(pd.Series(dtype=float)) is None

    def test_none_input(self):
        assert _last_float(None) is None


# ── Individual indicators ────────────────────────────────────────────────────


class TestComputeSma:
    def test_returns_float(self):
        result = compute_sma(STD_DF["close"], 20)
        assert isinstance(result, float)

    def test_insufficient_bars(self):
        assert compute_sma(TINY_DF["close"], 20) is None

    def test_empty(self):
        assert compute_sma(pd.Series(dtype=float), 20) is None


class TestComputeRsi:
    def test_returns_float_in_range(self):
        result = compute_rsi(STD_DF["close"], 14)
        assert isinstance(result, float)
        assert 0.0 <= result <= 100.0

    def test_insufficient_bars(self):
        assert compute_rsi(TINY_DF["close"], 14) is None


class TestComputeAtr:
    def test_returns_positive_float(self):
        result = compute_atr(STD_DF["high"], STD_DF["low"], STD_DF["close"], 14)
        assert isinstance(result, float)
        assert result > 0

    def test_insufficient_bars(self):
        result = compute_atr(TINY_DF["high"], TINY_DF["low"], TINY_DF["close"], 14)
        assert result is None


class TestComputeVolumeRatio:
    def test_returns_positive_float(self):
        result = compute_volume_ratio(STD_DF["volume"], 20)
        assert isinstance(result, float)
        assert result > 0

    def test_short_series_still_works(self):
        # Even with 5 bars it uses available data
        result = compute_volume_ratio(TINY_DF["volume"], 20)
        assert isinstance(result, float)

    def test_empty(self):
        assert compute_volume_ratio(pd.Series(dtype=float), 20) is None


class TestComputeVolume1dAvg:
    def test_equity(self):
        result = compute_volume_1d_avg(STD_DF["volume"], 100, 13)
        assert isinstance(result, float)
        assert result > 0

    def test_crypto(self):
        result = compute_volume_1d_avg(STD_DF["volume"], 100, 48)
        assert isinstance(result, float)
        assert result > 0


class TestComputeMacd:
    def test_returns_three_floats(self):
        line, signal, hist = compute_macd(STD_DF["close"])
        assert isinstance(line, float)
        assert isinstance(signal, float)
        assert isinstance(hist, float)

    def test_insufficient_bars(self):
        line, signal, hist = compute_macd(TINY_DF["close"])
        assert all(v is None for v in (line, signal, hist))


class TestComputeBbands:
    def test_returns_three_floats(self):
        lower, mid, upper = compute_bbands(STD_DF["close"])
        assert isinstance(lower, float)
        assert isinstance(mid, float)
        assert isinstance(upper, float)
        assert lower < upper

    def test_insufficient_bars(self):
        lower, mid, upper = compute_bbands(TINY_DF["close"])
        assert all(v is None for v in (lower, mid, upper))


class TestComputeStochastic:
    def test_returns_two_floats(self):
        k, d = compute_stochastic(STD_DF["high"], STD_DF["low"], STD_DF["close"])
        assert isinstance(k, float)
        assert isinstance(d, float)
        assert 0.0 <= k <= 100.0
        assert 0.0 <= d <= 100.0

    def test_insufficient_bars(self):
        k, d = compute_stochastic(TINY_DF["high"], TINY_DF["low"], TINY_DF["close"])
        assert all(v is None for v in (k, d))


class TestComputeVwap:
    def test_returns_float(self):
        result = compute_vwap(
            STD_DF["high"],
            STD_DF["low"],
            STD_DF["close"],
            STD_DF["volume"],
        )
        assert isinstance(result, float)

    def test_empty(self):
        e = pd.Series(dtype=float)
        assert compute_vwap(e, e, e, e) is None


# ── compute_all ──────────────────────────────────────────────────────────────


class TestComputeAll:
    def test_returns_indicators_schema(self):
        result = compute_all(STD_DF)
        assert isinstance(result, Indicators)

    def test_all_fields_populated(self):
        result = compute_all(STD_DF)
        for field in Indicators.model_fields:
            assert getattr(result, field) is not None, f"{field} is None"

    def test_crypto_flag_affects_volume_1d_avg(self):
        equity = compute_all(STD_DF, is_crypto=False)
        crypto = compute_all(STD_DF, is_crypto=True)
        # Same raw data but different bars_per_day → different daily avg
        assert equity.volume_1d_avg != crypto.volume_1d_avg

    def test_empty_df_returns_all_none(self):
        result = compute_all(EMPTY_DF)
        # All indicator fields are None except volume_1d_avg which is 0.0
        # (empty sum / 1.0 = 0.0, a valid degenerate result)
        for field in Indicators.model_fields:
            if field == "volume_1d_avg":
                continue
            assert getattr(result, field) is None, f"{field} should be None"

    def test_tiny_df_graceful(self):
        """With only 5 bars, most indicators are None but no crash."""
        result = compute_all(TINY_DF)
        assert isinstance(result, Indicators)
        # Volume ratio should still work with 5 bars
        assert result.volume_ratio is not None

    def test_constant_price(self):
        """Constant OHLCV should not crash — RSI and ATR may be None/0."""
        df = pd.DataFrame(
            {
                "open": [100.0] * 50,
                "high": [100.0] * 50,
                "low": [100.0] * 50,
                "close": [100.0] * 50,
                "volume": [1000.0] * 50,
            },
            index=pd.date_range("2026-01-01", periods=50, freq="30min"),
        )
        result = compute_all(df)
        assert isinstance(result, Indicators)
        assert result.sma_20 == 100.0
