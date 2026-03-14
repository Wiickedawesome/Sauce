"""
indicators — Pure-function technical indicator library.

Wraps pandas-ta into typed helpers that accept a DataFrame and return
either a single float value (latest reading) or a full pd.Series.
All functions are side-effect free.
"""

from sauce.indicators.core import (
    compute_all,
    compute_atr,
    compute_bbands,
    compute_macd,
    compute_rsi,
    compute_sma,
    compute_stochastic,
    compute_volume_ratio,
    compute_vwap,
)

__all__ = [
    "compute_all",
    "compute_atr",
    "compute_bbands",
    "compute_macd",
    "compute_rsi",
    "compute_sma",
    "compute_stochastic",
    "compute_volume_ratio",
    "compute_vwap",
]
