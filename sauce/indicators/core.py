"""
indicators/core.py — Pure technical indicator computations.

All functions accept a pandas DataFrame with standard OHLCV columns
(open, high, low, close, volume) and return typed results.

None is returned when data is insufficient or computation fails.
"""

from __future__ import annotations

import pandas as pd
import pandas_ta as ta  # type: ignore[import-untyped]

from sauce.core.schemas import Indicators

# 30-min bar counts per trading day.
# Crypto: 24h × 2 bars/h = 48.  Equity: 6.5h × 2 bars/h = 13.
BARS_PER_DAY_CRYPTO: int = 48
BARS_PER_DAY_EQUITY: int = 13


def _last_float(series: object) -> float | None:
    """Return the last non-NaN value from a pandas Series, or None."""
    try:
        val = series.dropna()  # type: ignore[union-attr]
        if val.empty:  # type: ignore[union-attr]
            return None
        f = float(val.iloc[-1])  # type: ignore[union-attr]
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError, AttributeError):
        return None


# ── Individual indicator functions ────────────────────────────────────────────


def compute_sma(close: pd.Series, length: int = 20) -> float | None:
    """Simple Moving Average of *close* over *length* periods."""
    return _last_float(ta.sma(close, length=length))


def compute_rsi(close: pd.Series, length: int = 14) -> float | None:
    """Relative Strength Index."""
    return _last_float(ta.rsi(close, length=length))


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14,
) -> float | None:
    """Average True Range."""
    return _last_float(ta.atr(high, low, close, length=length))


def compute_volume_ratio(volume: pd.Series, lookback: int = 20) -> float | None:
    """Current bar volume divided by trailing *lookback*-bar average."""
    try:
        vol_mean = float(volume.iloc[-lookback:].mean())
        if vol_mean <= 0:
            return None
        return float(volume.iloc[-1]) / vol_mean
    except (TypeError, ValueError, IndexError):
        return None


def compute_volume_1d_avg(
    volume: pd.Series, n_bars: int, bars_per_day: int,
) -> float | None:
    """Estimated average daily volume from intraday bar data."""
    try:
        estimated_days = max(n_bars / bars_per_day, 1.0)
        return float(volume.sum()) / estimated_days
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float | None, float | None, float | None]:
    """MACD line, signal line, histogram."""
    macd_df = ta.macd(close, fast=fast, slow=slow, signal=signal)
    if macd_df is None or macd_df.empty:
        return None, None, None
    return (
        _last_float(macd_df.iloc[:, 0]),
        _last_float(macd_df.iloc[:, 1]),
        _last_float(macd_df.iloc[:, 2]),
    )


def compute_bbands(
    close: pd.Series, length: int = 20, std: float = 2.0,
) -> tuple[float | None, float | None, float | None]:
    """Bollinger Bands: (lower, middle, upper)."""
    bb_df = ta.bbands(close, length=length, std=std)
    if bb_df is None or bb_df.empty:
        return None, None, None
    return (
        _last_float(bb_df.iloc[:, 0]),
        _last_float(bb_df.iloc[:, 1]),
        _last_float(bb_df.iloc[:, 2]),
    )


def compute_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
) -> tuple[float | None, float | None]:
    """Stochastic Oscillator: (%K, %D)."""
    stoch_df = ta.stoch(high, low, close, k=k, d=d, smooth_k=smooth_k)
    if stoch_df is None or stoch_df.empty:
        return None, None
    return _last_float(stoch_df.iloc[:, 0]), _last_float(stoch_df.iloc[:, 1])


def compute_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
) -> float | None:
    """Volume-Weighted Average Price."""
    result = ta.vwap(high, low, close, volume)
    return _last_float(result) if result is not None else None


# ── Aggregate: compute everything at once ─────────────────────────────────────


def compute_all(
    df: pd.DataFrame,
    *,
    is_crypto: bool = False,
) -> Indicators:
    """
    Compute all standard indicators from an OHLCV DataFrame.

    Returns a populated Indicators schema.  Missing / insufficient data
    fields are left as ``None``.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    n_bars = len(df)

    sma_20 = compute_sma(close, 20)
    sma_50 = compute_sma(close, 50)
    rsi_14 = compute_rsi(close, 14)
    atr_14 = compute_atr(high, low, close, 14)
    vol_ratio = compute_volume_ratio(volume, 20)
    macd_line, macd_signal, macd_histogram = compute_macd(close)
    bb_lower, bb_middle, bb_upper = compute_bbands(close)
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    vwap_val = compute_vwap(high, low, close, volume)

    bars_per_day = BARS_PER_DAY_CRYPTO if is_crypto else BARS_PER_DAY_EQUITY
    volume_1d_avg = compute_volume_1d_avg(volume, n_bars, bars_per_day)

    return Indicators(
        sma_20=sma_20,
        sma_50=sma_50,
        rsi_14=rsi_14,
        atr_14=atr_14,
        volume_ratio=vol_ratio,
        volume_1d_avg=volume_1d_avg,
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        vwap=vwap_val,
    )
