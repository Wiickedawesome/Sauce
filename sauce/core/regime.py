"""
core/regime.py — Deterministic market regime classifier.

Pure Python, no LLM. Classifies 30-minute OHLCV data into one of five regimes:
  TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE, DEAD

Priority order (first match wins):
  1. DEAD — volume extremely low, price barely moving → no trades
  2. VOLATILE — VIX proxy elevated, wide swings → confidence -20%, sizes halved
  3. TRENDING_UP — higher highs/lows, SMA20 > SMA50, volume on up bars
  4. TRENDING_DOWN — lower highs/lows, SMA20 < SMA50
  5. RANGING — default (SMAs converging, price bouncing)
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from pydantic import Field

from sauce.core.schemas import (
    Indicators,
    MarketRegime,
    RegimeLogEntry,
    RegimeTransitionEntry,
    StrictModel,
)

logger = logging.getLogger(__name__)


class RegimeDuration(StrictModel):
    """Result of regime duration analysis."""

    regime_type: MarketRegime
    active_minutes: float = Field(ge=0.0)
    historical_avg_minutes: float | None = None
    aging_out: bool = False
    aging_ratio: float = Field(ge=0.0, default=0.0)

# ── Thresholds (module-level constants) ───────────────────────────────────────
# These govern regime classification sensitivity. Tuned for 30-minute bars.

# DEAD regime
DEAD_VOLUME_RATIO_MAX: float = 0.25      # volume_ratio below this = "extremely low"
DEAD_ATR_PCT_MAX: float = 0.3            # ATR/close % below this = "barely moving"

# VOLATILE regime
VOLATILE_VIX_PROXY_MIN: float = 3.0      # VIX proxy above this = "elevated"
VOLATILE_ATR_PCT_MIN: float = 2.0        # ATR/close % above this with high volume = volatile
VOLATILE_VOLUME_RATIO_MIN: float = 2.0   # volume_ratio above this with high ATR = volatile

# TRENDING thresholds
TREND_SMA_DIVERGENCE_MIN: float = 0.5    # SMA20-SMA50 spread % needed for trend signal
TREND_VOLUME_BIAS_MIN: float = 0.1       # volume bias threshold for directional confirmation
TREND_RSI_WEAK: float = 40.0             # RSI below this adds to bearish confidence

# RANGING thresholds
RANGE_SMA_CONVERGENCE_MAX: float = 1.0   # SMA spread % below this → converging
RANGE_VOLUME_RATIO_MAX: float = 0.8      # volume_ratio below this → declining

# Higher-highs/lows detection
HH_HL_LOOKBACK: int = 10                 # bars to analyze for swing structure
HH_HL_SPLIT: int = 5                     # split point for first vs second half

# Minimum bars for meaningful classification
MIN_BARS: int = 5


# ── Helper functions ──────────────────────────────────────────────────────────

def _compute_vix_proxy(df: pd.DataFrame) -> float:
    """
    ATR-based volatility proxy from raw OHLCV bars.

    Computes average true range as a percentage of average close price
    over the last 14 bars (or available bars if fewer).
    Returns 0.0 on insufficient data.
    """
    n = min(len(df), 14)
    if n < 2:
        return 0.0

    recent = df.tail(n)
    highs = recent["high"]
    lows = recent["low"]
    closes = recent["close"]

    avg_range = (highs - lows).mean()
    avg_close = closes.mean()

    if avg_close <= 0:
        return 0.0

    return float((avg_range / avg_close) * 100)


def _detect_swing_structure(
    df: pd.DataFrame,
    lookback: int = HH_HL_LOOKBACK,
) -> tuple[bool, bool]:
    """
    Detect higher-highs/higher-lows vs lower-highs/lower-lows.

    Splits the last ``lookback`` bars into two halves and compares
    the max high and min low of each half.

    Returns:
        (higher_highs_and_lows, lower_highs_and_lows)
    """
    n = min(len(df), lookback)
    if n < 4:
        return False, False

    recent = df.tail(n)
    mid = n // 2
    first_half = recent.iloc[:mid]
    second_half = recent.iloc[mid:]

    first_high = first_half["high"].max()
    second_high = second_half["high"].max()
    first_low = first_half["low"].min()
    second_low = second_half["low"].min()

    higher_trend = bool(second_high > first_high and second_low > first_low)
    lower_trend = bool(second_high < first_high and second_low < first_low)

    return higher_trend, lower_trend


def _volume_direction_bias(df: pd.DataFrame, lookback: int = 10) -> float:
    """
    Compute volume-weighted directional bias.

    Positive → volume concentrated on up bars (bullish).
    Negative → volume concentrated on down bars (bearish).
    Range: [-1.0, 1.0]. Returns 0.0 on insufficient data.
    """
    n = min(len(df), lookback)
    if n < 2:
        return 0.0

    recent = df.tail(n)
    closes = recent["close"].values
    opens = recent["open"].values
    volumes = recent["volume"].values

    up_mask = closes >= opens
    up_volume = float(volumes[up_mask].sum())
    down_volume = float(volumes[~up_mask].sum())

    total = up_volume + down_volume
    if total == 0:
        return 0.0

    return (up_volume - down_volume) / total


def _sma_spread_pct(indicators: Indicators) -> float:
    """
    Distance between SMA20 and SMA50 as a percentage of their average.

    Low values → SMAs converging (RANGING signal).
    High values → SMAs diverging (TRENDING signal).
    Returns 0.0 if either SMA is None.
    """
    if indicators.sma_20 is None or indicators.sma_50 is None:
        return 0.0

    avg = (indicators.sma_20 + indicators.sma_50) / 2
    if avg <= 0:
        return 0.0

    return abs(indicators.sma_20 - indicators.sma_50) / avg * 100


def _atr_pct(indicators: Indicators, last_close: float) -> float:
    """ATR as a percentage of the latest close price."""
    if indicators.atr_14 is None or last_close <= 0:
        return 0.0
    return (indicators.atr_14 / last_close) * 100


# ── Main classifier ──────────────────────────────────────────────────────────

def classify_regime(
    df: pd.DataFrame,
    indicators: Indicators,
    as_of: datetime,
) -> RegimeLogEntry:
    """
    Classify the current market regime from OHLCV bars and indicators.

    Pure Python, deterministic, no LLM call. Priority-based classification:
    DEAD → VOLATILE → TRENDING_UP → TRENDING_DOWN → RANGING (default).

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
            Must have at least MIN_BARS rows for meaningful classification.
        indicators: Pre-computed technical indicators for the symbol.
        as_of: Timestamp from the data API (UTC). Never use datetime.now().

    Returns:
        RegimeLogEntry with classified regime, confidence, VIX proxy, and bias.
    """
    # Insufficient data → safest default
    if len(df) < MIN_BARS:
        logger.warning("regime: only %d bars available (need %d) — defaulting to DEAD", len(df), MIN_BARS)
        return RegimeLogEntry(
            timestamp=as_of,
            regime_type="DEAD",
            confidence=0.0,
            vix_proxy=0.0,
            market_bias="insufficient data",
        )

    # Pre-compute all signals once
    vix_proxy = _compute_vix_proxy(df)
    higher_trend, lower_trend = _detect_swing_structure(df)
    vol_bias = _volume_direction_bias(df)
    sma_spread = _sma_spread_pct(indicators)
    volume_ratio = indicators.volume_ratio if indicators.volume_ratio is not None else 0.0
    last_close = float(df["close"].iloc[-1])
    atr_pct_val = _atr_pct(indicators, last_close)

    # ── Priority 1: DEAD ─────────────────────────────────────────────────
    if volume_ratio < DEAD_VOLUME_RATIO_MAX and atr_pct_val < DEAD_ATR_PCT_MAX:
        # Both conditions met — confidence based on how far below thresholds
        vol_score = (DEAD_VOLUME_RATIO_MAX - volume_ratio) / DEAD_VOLUME_RATIO_MAX
        atr_score = (DEAD_ATR_PCT_MAX - atr_pct_val) / DEAD_ATR_PCT_MAX
        confidence = min(1.0, vol_score * 0.5 + atr_score * 0.5)
        logger.info("regime: DEAD (vol_ratio=%.3f, atr_pct=%.3f)", volume_ratio, atr_pct_val)
        return RegimeLogEntry(
            timestamp=as_of,
            regime_type="DEAD",
            confidence=round(confidence, 3),
            vix_proxy=round(vix_proxy, 4),
            market_bias="no significant activity",
        )

    # ── Priority 2: VOLATILE ─────────────────────────────────────────────
    vix_elevated = vix_proxy > VOLATILE_VIX_PROXY_MIN
    atr_high = atr_pct_val > VOLATILE_ATR_PCT_MIN
    vol_high = volume_ratio > VOLATILE_VOLUME_RATIO_MIN

    if vix_elevated or (atr_high and vol_high):
        signals = [vix_elevated, atr_high, vol_high]
        confidence = min(1.0, max(0.5, sum(signals) / len(signals)))
        logger.info(
            "regime: VOLATILE (vix_proxy=%.3f, atr_pct=%.3f, vol_ratio=%.3f)",
            vix_proxy, atr_pct_val, volume_ratio,
        )
        return RegimeLogEntry(
            timestamp=as_of,
            regime_type="VOLATILE",
            confidence=round(confidence, 3),
            vix_proxy=round(vix_proxy, 4),
            market_bias="elevated volatility",
        )

    # ── Priority 3: TRENDING_UP ──────────────────────────────────────────
    sma_bullish = (
        indicators.sma_20 is not None
        and indicators.sma_50 is not None
        and indicators.sma_20 > indicators.sma_50
    )

    if higher_trend and sma_bullish and vol_bias > TREND_VOLUME_BIAS_MIN:
        signals = [
            higher_trend,
            sma_bullish,
            vol_bias > TREND_VOLUME_BIAS_MIN,
            sma_spread > TREND_SMA_DIVERGENCE_MIN,
        ]
        confidence = min(1.0, max(0.5, sum(signals) / len(signals)))
        logger.info(
            "regime: TRENDING_UP (sma_spread=%.3f, vol_bias=%.3f)",
            sma_spread, vol_bias,
        )
        return RegimeLogEntry(
            timestamp=as_of,
            regime_type="TRENDING_UP",
            confidence=round(confidence, 3),
            vix_proxy=round(vix_proxy, 4),
            market_bias="bullish momentum",
        )

    # ── Priority 4: TRENDING_DOWN ────────────────────────────────────────
    sma_bearish = (
        indicators.sma_20 is not None
        and indicators.sma_50 is not None
        and indicators.sma_20 < indicators.sma_50
    )
    rsi_weak = (indicators.rsi_14 or 50) < TREND_RSI_WEAK

    if lower_trend and sma_bearish:
        signals = [
            lower_trend,
            sma_bearish,
            rsi_weak,
            vol_bias < -TREND_VOLUME_BIAS_MIN,
        ]
        confidence = min(1.0, max(0.5, sum(signals) / len(signals)))
        logger.info(
            "regime: TRENDING_DOWN (sma_spread=%.3f, rsi=%.1f, vol_bias=%.3f)",
            sma_spread, indicators.rsi_14 or 0.0, vol_bias,
        )
        return RegimeLogEntry(
            timestamp=as_of,
            regime_type="TRENDING_DOWN",
            confidence=round(confidence, 3),
            vix_proxy=round(vix_proxy, 4),
            market_bias="bearish pressure",
        )

    # ── Default: RANGING ─────────────────────────────────────────────────
    sma_converging = sma_spread < RANGE_SMA_CONVERGENCE_MAX
    vol_declining = volume_ratio < RANGE_VOLUME_RATIO_MAX
    no_swing = not higher_trend and not lower_trend
    vol_neutral = abs(vol_bias) < 0.3

    signals = [sma_converging, vol_declining, no_swing, vol_neutral]
    confidence = min(1.0, max(0.4, sum(signals) / len(signals)))

    logger.info(
        "regime: RANGING (sma_spread=%.3f, vol_ratio=%.3f)",
        sma_spread, volume_ratio,
    )
    return RegimeLogEntry(
        timestamp=as_of,
        regime_type="RANGING",
        confidence=round(confidence, 3),
        vix_proxy=round(vix_proxy, 4),
        market_bias="sideways consolidation",
    )


# ── Regime duration tracker ──────────────────────────────────────────────────

# When the active duration reaches this fraction of the historical average,
# the regime is flagged as "aging out" (approaching typical transition point).
AGING_THRESHOLD: float = 0.8


def _ensure_utc(dt: datetime) -> datetime:
    """Return a UTC-aware datetime for downstream duration math."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_regime_duration(
    regime_history: list[RegimeLogEntry],
    historical_transitions: list[RegimeTransitionEntry],
    as_of: datetime,
) -> RegimeDuration | None:
    """
    Track how long the current regime has been active and whether it is aging out.

    Uses session memory (today's regime history) to determine the current regime
    and its start time. Compares against strategic memory (historical transitions)
    to compute the aging ratio.

    Args:
        regime_history: Today's regime classifications, ordered by timestamp.
            Must not be empty.
        historical_transitions: Historical regime transition records from
            strategic memory. Used to compute average duration.
        as_of: Current time (UTC) from the data API.

    Returns:
        RegimeDuration with active minutes, historical average, and aging flag.
        None if regime_history is empty.
    """
    if not regime_history:
        return None

    as_of = _ensure_utc(as_of)

    # Sort by timestamp to ensure correct ordering
    sorted_history = sorted(regime_history, key=lambda r: _ensure_utc(r.timestamp))

    current_regime = sorted_history[-1].regime_type

    # Walk backward to find when this regime started
    regime_start = _ensure_utc(sorted_history[-1].timestamp)
    for entry in reversed(sorted_history[:-1]):
        if entry.regime_type == current_regime:
            regime_start = _ensure_utc(entry.timestamp)
        else:
            break

    active_minutes = max(0.0, (as_of - regime_start).total_seconds() / 60.0)

    # Compute historical average duration for this regime type
    matching = [
        t for t in historical_transitions
        if t.from_regime == current_regime
    ]

    historical_avg: float | None = None
    aging_ratio = 0.0
    aging_out = False

    if matching:
        total_duration = sum(t.duration_minutes * t.count for t in matching)
        total_count = sum(t.count for t in matching)
        if total_count > 0:
            historical_avg = total_duration / total_count
            if historical_avg > 0:
                aging_ratio = active_minutes / historical_avg
                aging_out = aging_ratio >= AGING_THRESHOLD

    logger.info(
        "regime duration: %s active %.1f min (avg %.1f, ratio %.2f, aging=%s)",
        current_regime,
        active_minutes,
        historical_avg or 0.0,
        aging_ratio,
        aging_out,
    )

    return RegimeDuration(
        regime_type=current_regime,
        active_minutes=round(active_minutes, 1),
        historical_avg_minutes=round(historical_avg, 1) if historical_avg is not None else None,
        aging_out=aging_out,
        aging_ratio=round(aging_ratio, 3),
    )
