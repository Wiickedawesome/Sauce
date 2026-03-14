"""
signals/timeframes.py — Multi-timeframe OHLCV fetch + indicator computation.

Fetches OHLCV at 5m, 15m, 1h, 4h, 1d timeframes for a single symbol,
computes indicators per timeframe, and packages results into a structured
MultiTimeframeContext for downstream confluence scoring and prompt injection.

All indicator computation delegates to sauce.indicators.core.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sauce.adapters import market_data
from sauce.core.schemas import Indicators
from sauce.indicators.core import compute_all

logger = logging.getLogger(__name__)

# Timeframes to analyse, from shortest to longest.
# Each entry: (label, Alpaca timeframe string, bars to fetch).
# Bars chosen so each timeframe covers roughly 2-5 days for context.
TIMEFRAMES: list[tuple[str, str, int]] = [
    ("5m", "5Min", 120),    # ~10h of 5-min bars
    ("15m", "15Min", 80),   # ~20h of 15-min bars
    ("1h", "1Hour", 48),    # ~2 days of hourly bars
    ("4h", "4Hour", 30),    # ~5 days of 4-hour bars
    ("1d", "1Day", 50),     # ~50 trading days
]


@dataclass(frozen=True)
class TimeframeAnalysis:
    """Indicators plus trend classification for a single timeframe."""

    label: str               # e.g. "5m", "1h", "1d"
    timeframe: str           # Alpaca string, e.g. "5Min"
    bars_fetched: int
    indicators: Indicators
    trend: str               # "bullish" | "bearish" | "neutral" | "unknown"
    momentum: str            # "bullish" | "bearish" | "neutral" | "unknown"


@dataclass(frozen=True)
class MultiTimeframeContext:
    """Aggregated multi-timeframe analysis for a single symbol."""

    symbol: str
    analyses: tuple[TimeframeAnalysis, ...] = field(default_factory=tuple)

    def to_prompt_dict(self) -> dict:
        """Serialize to a dict suitable for JSON injection into the prompt."""
        result: dict = {"symbol": self.symbol, "timeframes": {}}
        for a in self.analyses:
            result["timeframes"][a.label] = {
                "bars": a.bars_fetched,
                "trend": a.trend,
                "momentum": a.momentum,
                "sma_20": _r(a.indicators.sma_20),
                "sma_50": _r(a.indicators.sma_50),
                "rsi_14": _r(a.indicators.rsi_14),
                "macd_histogram": _r(a.indicators.macd_histogram),
                "bb_upper": _r(a.indicators.bb_upper),
                "bb_lower": _r(a.indicators.bb_lower),
                "stoch_k": _r(a.indicators.stoch_k),
                "volume_ratio": _r(a.indicators.volume_ratio),
            }
        return result


def _r(v: float | None) -> float | None:
    return round(v, 4) if v is not None else None


def _classify_trend(ind: Indicators) -> str:
    """Classify trend from SMA relationship and price position."""
    sma20 = ind.sma_20
    sma50 = ind.sma_50
    if sma20 is None:
        return "unknown"
    if sma50 is None:
        # Only have SMA20 — use it alone
        return "unknown"
    if sma20 > sma50:
        return "bullish"
    if sma20 < sma50:
        return "bearish"
    return "neutral"


def _classify_momentum(ind: Indicators) -> str:
    """Classify momentum from RSI + MACD histogram."""
    rsi = ind.rsi_14
    hist = ind.macd_histogram

    if rsi is None and hist is None:
        return "unknown"

    bullish_count = 0
    bearish_count = 0

    if rsi is not None:
        if rsi > 55:
            bullish_count += 1
        elif rsi < 45:
            bearish_count += 1

    if hist is not None:
        if hist > 0:
            bullish_count += 1
        elif hist < 0:
            bearish_count += 1

    if bullish_count > bearish_count:
        return "bullish"
    if bearish_count > bullish_count:
        return "bearish"
    return "neutral"


def fetch_multi_timeframe(
    symbol: str,
    *,
    is_crypto: bool = False,
) -> MultiTimeframeContext:
    """
    Fetch OHLCV at multiple timeframes and compute indicators for each.

    Failures on individual timeframes are logged and skipped — the result
    will contain only the timeframes that succeeded.  This is intentional:
    partial multi-TF context is better than none.
    """
    analyses: list[TimeframeAnalysis] = []

    for label, tf_str, bars in TIMEFRAMES:
        try:
            df = market_data.get_history(symbol, timeframe=tf_str, bars=bars)
            if df.empty or len(df) < 20:
                logger.debug(
                    "multi-tf[%s/%s]: insufficient bars (%d), skipping",
                    symbol, label, len(df),
                )
                continue

            indicators = compute_all(df, is_crypto=is_crypto)
            trend = _classify_trend(indicators)
            momentum = _classify_momentum(indicators)

            analyses.append(TimeframeAnalysis(
                label=label,
                timeframe=tf_str,
                bars_fetched=len(df),
                indicators=indicators,
                trend=trend,
                momentum=momentum,
            ))
        except market_data.MarketDataError as exc:
            logger.warning("multi-tf[%s/%s]: fetch failed: %s", symbol, label, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("multi-tf[%s/%s]: unexpected error: %s", symbol, label, exc)

    return MultiTimeframeContext(symbol=symbol, analyses=tuple(analyses))
