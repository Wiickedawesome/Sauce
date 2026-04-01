"""
indicators/timeframes.py — Multi-timeframe indicator scoring.

Provides confirmation across multiple timeframes to filter out noise.
Higher timeframes add weight to signals on lower timeframes.

Timeframe sets:
- Crypto: 1h (primary), 4h (confirmation), 1d (trend)
- Equity: 1d (primary), 1w (confirmation)

Usage:
    mtf_score = score_multi_timeframe(symbol, is_crypto=True)
    if mtf_score.trend_aligned:
        # Higher timeframes confirm the direction
        total_score += mtf_score.bonus_points
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from sauce.adapters.market_data import get_history
from sauce.core.schemas import Indicators
from sauce.indicators.core import compute_all

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MultiTimeframeScore:
    """Result of multi-timeframe analysis."""

    symbol: str
    primary_tf: str
    secondary_tf: str
    tertiary_tf: str | None

    # Trend direction per timeframe
    primary_trend: Literal["bullish", "bearish", "neutral"]
    secondary_trend: Literal["bullish", "bearish", "neutral"]
    tertiary_trend: Literal["bullish", "bearish", "neutral"] | None

    # Are all timeframes aligned?
    trend_aligned: bool
    # Bonus points for confluence (0-30)
    bonus_points: int


def _classify_trend(
    indicators: Indicators, price: float
) -> Literal["bullish", "bearish", "neutral"]:
    """Classify trend direction from indicators."""
    bullish_signals = 0
    bearish_signals = 0

    # SMA trend
    if indicators.sma_20 is not None and indicators.sma_50 is not None:
        if indicators.sma_20 > indicators.sma_50:
            bullish_signals += 1
        elif indicators.sma_20 < indicators.sma_50:
            bearish_signals += 1

    # Price vs SMA20
    if indicators.sma_20 is not None:
        if price > indicators.sma_20:
            bullish_signals += 1
        elif price < indicators.sma_20:
            bearish_signals += 1

    # MACD direction
    if indicators.macd_histogram is not None:
        if indicators.macd_histogram > 0:
            bullish_signals += 1
        elif indicators.macd_histogram < 0:
            bearish_signals += 1

    # RSI momentum
    if indicators.rsi_14 is not None:
        if indicators.rsi_14 > 55:
            bullish_signals += 1
        elif indicators.rsi_14 < 45:
            bearish_signals += 1

    # Classify
    if bullish_signals >= 3:
        return "bullish"
    elif bearish_signals >= 3:
        return "bearish"
    else:
        return "neutral"


def _fetch_and_classify(
    symbol: str, timeframe: str, is_crypto: bool
) -> tuple[Indicators | None, Literal["bullish", "bearish", "neutral"]]:
    """Fetch indicators and classify trend for a timeframe."""
    try:
        df = get_history(symbol, timeframe=timeframe, bars=100)
        if df is None or df.empty:
            return None, "neutral"

        indicators = compute_all(df, is_crypto=is_crypto)
        if indicators is None:
            return None, "neutral"

        # Get latest price
        price = float(df["close"].iloc[-1])
        trend = _classify_trend(indicators, price)

        return indicators, trend

    except Exception as exc:
        logger.warning("MTF fetch failed for %s %s: %s", symbol, timeframe, exc)
        return None, "neutral"


def score_multi_timeframe(
    symbol: str,
    is_crypto: bool = True,
) -> MultiTimeframeScore:
    """
    Analyze symbol across multiple timeframes and return alignment score.

    Crypto uses: 1Hour (primary), 4Hour (secondary), 1Day (tertiary)
    Equity uses: 1Day (primary), 1Week (secondary), no tertiary
    """
    if is_crypto:
        primary_tf = "1Hour"
        secondary_tf = "4Hour"
        tertiary_tf = "1Day"
    else:
        primary_tf = "1Day"
        secondary_tf = "1Week"
        tertiary_tf = None

    # Fetch and classify each timeframe
    _, primary_trend = _fetch_and_classify(symbol, primary_tf, is_crypto)
    _, secondary_trend = _fetch_and_classify(symbol, secondary_tf, is_crypto)

    tertiary_trend: Literal["bullish", "bearish", "neutral"] | None = None
    if tertiary_tf:
        _, tertiary_trend = _fetch_and_classify(symbol, tertiary_tf, is_crypto)

    # Check alignment
    trends = [primary_trend, secondary_trend]
    if tertiary_trend is not None:
        trends.append(tertiary_trend)

    all_bullish = all(t == "bullish" for t in trends)
    all_bearish = all(t == "bearish" for t in trends)
    trend_aligned = all_bullish or all_bearish

    # Calculate bonus points
    bonus_points = 0
    if trend_aligned:
        # Full alignment across all timeframes
        bonus_points = 20 if tertiary_tf else 15

    elif primary_trend == secondary_trend and primary_trend != "neutral":
        # Primary and secondary aligned (not tertiary)
        bonus_points = 10

    elif secondary_trend != "neutral" and primary_trend != "neutral":
        # Mixed signals — no bonus, but not a penalty either
        bonus_points = 0

    logger.info(
        "MTF %s: %s=%s %s=%s %s=%s aligned=%s bonus=%d",
        symbol,
        primary_tf,
        primary_trend,
        secondary_tf,
        secondary_trend,
        tertiary_tf or "N/A",
        tertiary_trend or "N/A",
        trend_aligned,
        bonus_points,
    )

    return MultiTimeframeScore(
        symbol=symbol,
        primary_tf=primary_tf,
        secondary_tf=secondary_tf,
        tertiary_tf=tertiary_tf,
        primary_trend=primary_trend,
        secondary_trend=secondary_trend,
        tertiary_trend=tertiary_trend,
        trend_aligned=trend_aligned,
        bonus_points=bonus_points,
    )


def get_mtf_confirmation(
    symbol: str,
    direction: Literal["call", "put", "buy", "sell"],
    is_crypto: bool = True,
) -> tuple[bool, int]:
    """
    Check if multi-timeframe analysis confirms the intended direction.

    Args:
        symbol: Ticker symbol
        direction: Intended trade direction
        is_crypto: Whether this is a crypto pair

    Returns:
        (confirmed, bonus_points) tuple
    """
    mtf = score_multi_timeframe(symbol, is_crypto=is_crypto)

    # Map direction to required trend
    bullish_directions = {"call", "buy"}
    bearish_directions = {"put", "sell"}

    if direction in bullish_directions:
        required_trend = "bullish"
    elif direction in bearish_directions:
        required_trend = "bearish"
    else:
        return False, 0

    # Check if primary matches required trend
    if mtf.primary_trend != required_trend:
        return False, 0

    # Return alignment status and bonus
    if mtf.trend_aligned:
        return True, mtf.bonus_points
    elif mtf.secondary_trend == required_trend:
        return True, mtf.bonus_points // 2
    else:
        return True, 0
