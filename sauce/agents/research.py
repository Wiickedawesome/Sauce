"""
agents/research.py — Research agent: generates trading signals from market data.

Phase 5 IMPLEMENTATION.

Sequence:
  1. Fetch 60 bars of 30-min OHLCV history via market_data.get_history().
  2. Compute SMA_20, SMA_50, RSI_14, ATR_14, volume_ratio via pandas-ta.
  3. Build a grounded prompt with price + indicator data.
  4. Call llm.call_claude() with the Research system prompt.
  5. Parse JSON response into Signal.
  6. On any error (parse, validation, LLM, stale data) → return safe hold Signal.
"""

import json
import logging
from datetime import datetime, timezone

import pandas_ta as ta  # type: ignore[import-untyped]
from pydantic import ValidationError

from sauce.adapters import llm, market_data
from sauce.adapters.db import log_event
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, Evidence, Indicators, PriceReference, Signal
from sauce.prompts import research as research_prompts

logger = logging.getLogger(__name__)

# Minimum bars needed before indicators are reliable.
# SMA_50 needs 50 bars, so we require at least 55 as a buffer.
_MIN_BARS_REQUIRED = 55


async def run(
    symbol: str,
    quote: PriceReference,
    loop_id: str,
) -> Signal:
    """
    Generate a trading signal for a single symbol.

    Fetches OHLCV history, computes indicators, calls Claude, and parses
    the response into a Signal. Any failure returns a safe hold signal.

    Parameters
    ----------
    symbol:   Trading symbol (equity ticker or crypto pair).
    quote:    Latest price reference from the market data adapter (must be fresh).
    loop_id:  Current loop run UUID for audit correlation.

    Returns
    -------
    Signal with real side/confidence (Phase 5), or side="hold" on any failure.
    """
    settings = get_settings()
    db_path = str(settings.db_path)
    as_of = datetime.now(timezone.utc)

    def _safe_hold(reason: str) -> Signal:
        """Return a safe hold signal with an attached audit log entry."""
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="signal",
                symbol=symbol,
                payload={"agent": "research", "side": "hold", "reason": reason},
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return Signal(
            symbol=symbol,
            side="hold",
            confidence=0.0,
            evidence=Evidence(
                symbol=symbol,
                price_reference=quote,
                indicators=Indicators(),
                as_of=quote.as_of,
            ),
            reasoning=f"Hold (safe default): {reason}",
            as_of=as_of,
            prompt_version=settings.prompt_version,
        )

    # ── Step 1: Fetch OHLCV history ───────────────────────────────────────────
    try:
        df = market_data.get_history(symbol, timeframe="30Min", bars=60)
    except market_data.MarketDataError as exc:
        logger.warning("research[%s]: get_history failed: %s", symbol, exc)
        return _safe_hold(f"market data unavailable: {exc}")

    if df.empty or len(df) < _MIN_BARS_REQUIRED:
        logger.info(
            "research[%s]: insufficient history (%d bars, need %d)",
            symbol, len(df), _MIN_BARS_REQUIRED,
        )
        return _safe_hold(f"insufficient history ({len(df)} bars)")

    # ── Step 2: Compute indicators ────────────────────────────────────────────
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

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

    sma_20 = _last_float(ta.sma(close, length=20))
    sma_50 = _last_float(ta.sma(close, length=50))
    rsi_14 = _last_float(ta.rsi(close, length=14))
    atr_result = ta.atr(high, low, close, length=14)
    atr_14 = _last_float(atr_result)

    # Volume ratio: current bar volume vs 20-bar mean
    try:
        vol_mean = float(volume.iloc[-20:].mean())
        volume_ratio: float | None = (
            float(volume.iloc[-1]) / vol_mean
            if vol_mean > 0
            else None
        )
    except (TypeError, ValueError, IndexError):
        volume_ratio = None

    # Average daily volume estimate: sum all bars, divide by estimated trading days.
    # 60 bars of 30-min data ≈ 4–5 trading days for equities (13 bars/day).
    # Crypto trades 24/7 (48 bars/day), giving a slightly tighter estimate.
    # Using 13 bars/day is conservative for equities and reasonable for crypto.
    _BARS_PER_DAY = 13
    try:
        estimated_days = max(len(df) / _BARS_PER_DAY, 1.0)
        volume_1d_avg: float | None = float(volume.sum()) / estimated_days
    except (TypeError, ValueError, ZeroDivisionError):
        volume_1d_avg = None

    # ── Step 3: Build prompt ──────────────────────────────────────────────────
    user_prompt = research_prompts.build_user_prompt(
        symbol=symbol,
        mid=quote.mid,
        bid=quote.bid,
        ask=quote.ask,
        sma_20=sma_20,
        sma_50=sma_50,
        rsi_14=rsi_14,
        atr_14=atr_14,
        volume_ratio=volume_ratio,
        prompt_version=settings.prompt_version,
        as_of_utc=as_of,
    )

    # ── Step 4: Call Claude ───────────────────────────────────────────────────
    try:
        raw_response = await llm.call_claude(
            system=research_prompts.SYSTEM_PROMPT,
            user=user_prompt,
            loop_id=loop_id,
        )
    except llm.LLMError as exc:
        logger.error("research[%s]: LLM call failed: %s", symbol, exc)
        return _safe_hold(f"LLM error: {exc}")

    # ── Step 5: Parse response ────────────────────────────────────────────────
    try:
        data = json.loads(raw_response.strip())
        side = str(data.get("side", "hold")).lower()
        if side not in ("buy", "sell", "hold"):
            side = "hold"
        confidence = float(data.get("confidence", 0.0))
        # Clamp to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(data.get("reasoning", ""))
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        logger.warning(
            "research[%s]: failed to parse LLM response: %s | raw=%r",
            symbol, exc, raw_response[:200],
        )
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=symbol,
                payload={
                    "agent": "research",
                    "error": f"JSON parse failed: {exc}",
                    "raw_response_preview": raw_response[:200],
                },
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return _safe_hold("LLM response parse error")

    # ── Step 6: Build validated Signal ───────────────────────────────────────
    indicators = Indicators(
        sma_20=sma_20,
        sma_50=sma_50,
        rsi_14=rsi_14,
        atr_14=atr_14,
        volume_ratio=volume_ratio,
        volume_1d_avg=volume_1d_avg,
    )
    evidence = Evidence(
        symbol=symbol,
        price_reference=quote,
        indicators=indicators,
        as_of=quote.as_of,
    )
    try:
        signal = Signal(
            symbol=symbol,
            side=side,  # type: ignore[arg-type]
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            as_of=as_of,
            prompt_version=settings.prompt_version,
        )
    except ValidationError as exc:
        logger.error("research[%s]: Signal validation failed: %s", symbol, exc)
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=symbol,
                payload={"agent": "research", "error": f"Signal validation: {exc}"},
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return _safe_hold("Signal schema validation error")

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="signal",
            symbol=symbol,
            payload={
                "agent": "research",
                "side": signal.side,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "indicators": {
                    "sma_20": sma_20,
                    "sma_50": sma_50,
                    "rsi_14": rsi_14,
                    "atr_14": atr_14,
                    "volume_ratio": volume_ratio,
                },
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    return signal
