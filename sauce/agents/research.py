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
from sauce.adapters.db import get_recent_signals, log_event
from sauce.memory.db import get_session_context, get_strategic_context
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, Evidence, Indicators, PriceReference, Signal
from sauce.core.setups import scan_setups
from sauce.prompts import research as research_prompts
from sauce.prompts.context import build_session_paragraph, build_strategic_paragraph

logger = logging.getLogger(__name__)

# Minimum bars needed before indicators are reliable.
# SMA_50 needs 50 bars, so we require at least 55 as a buffer.
_MIN_BARS_REQUIRED = 55


async def run(
    symbol: str,
    quote: PriceReference,
    loop_id: str,
    *,
    regime: str | None = None,
    positions: list[dict] | None = None,
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
    # Crypto trades 24/7 → 48 bars/day for 30-min timeframe.
    _BARS_PER_DAY_EQUITY = 13
    _BARS_PER_DAY_CRYPTO = 48
    _bars_per_day = _BARS_PER_DAY_CRYPTO if market_data._is_crypto(symbol) else _BARS_PER_DAY_EQUITY
    try:
        estimated_days = max(len(df) / _bars_per_day, 1.0)
        volume_1d_avg: float | None = float(volume.sum()) / estimated_days
    except (TypeError, ValueError, ZeroDivisionError):
        volume_1d_avg = None

    # MACD (12, 26, 9)
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        macd_line = _last_float(macd_df.iloc[:, 0])
        macd_signal = _last_float(macd_df.iloc[:, 1])
        macd_histogram = _last_float(macd_df.iloc[:, 2])
    else:
        macd_line = macd_signal = macd_histogram = None

    # Bollinger Bands (20, 2)
    bb_df = ta.bbands(close, length=20, std=2)
    if bb_df is not None and not bb_df.empty:
        bb_lower = _last_float(bb_df.iloc[:, 0])
        bb_middle = _last_float(bb_df.iloc[:, 1])
        bb_upper = _last_float(bb_df.iloc[:, 2])
    else:
        bb_upper = bb_middle = bb_lower = None

    # Stochastic Oscillator (14, 3, 3)
    stoch_df = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
    if stoch_df is not None and not stoch_df.empty:
        stoch_k = _last_float(stoch_df.iloc[:, 0])
        stoch_d = _last_float(stoch_df.iloc[:, 1])
    else:
        stoch_k = stoch_d = None

    # VWAP
    vwap_series = ta.vwap(high, low, close, volume)
    vwap_val = _last_float(vwap_series) if vwap_series is not None else None

    # ── Step 2b: Fetch daily bars for higher-timeframe trend context ──────────
    df_daily = None  # declared here so step 2e (setup scoring) can reference it
    daily_trend: dict | None = None
    try:
        df_daily = market_data.get_history(symbol, timeframe="1Day", bars=50)
        if not df_daily.empty and len(df_daily) >= 20:
            d_close = df_daily["close"]
            d_sma_20 = _last_float(ta.sma(d_close, length=20))
            d_sma_50 = (
                _last_float(ta.sma(d_close, length=50))
                if len(df_daily) >= 50
                else None
            )
            d_rsi_14 = _last_float(ta.rsi(d_close, length=14))
            d_close_price = (
                float(d_close.iloc[-1]) if not d_close.empty else None
            )

            # Classify daily trend
            if d_sma_20 and d_sma_50 and d_close_price:
                if d_close_price > d_sma_20 > d_sma_50:
                    trend = "bullish"
                elif d_close_price < d_sma_20 < d_sma_50:
                    trend = "bearish"
                else:
                    trend = "neutral"
            elif d_sma_20 and d_close_price:
                trend = "bullish" if d_close_price > d_sma_20 else "bearish"
            else:
                trend = "unknown"

            daily_trend = {
                "daily_sma_20": round(d_sma_20, 4) if d_sma_20 else None,
                "daily_sma_50": round(d_sma_50, 4) if d_sma_50 else None,
                "daily_rsi_14": round(d_rsi_14, 4) if d_rsi_14 else None,
                "daily_close": round(d_close_price, 4) if d_close_price else None,
                "trend_classification": trend,
            }
    except market_data.MarketDataError:
        pass  # Daily data unavailable — proceed with 30min only

    # ── Step 2c: Fetch recent signal history for feedback loop ─────────────
    signal_history: list[dict] | None = None
    try:
        db_path = str(settings.db_path)
        recent = get_recent_signals(symbol=symbol, days=7, db_path=db_path)
        if recent:
            signal_history = recent
    except Exception as exc:  # noqa: BLE001
        logger.debug("research[%s]: signal history unavailable: %s", symbol, exc)

    # ── Step 2d: Fetch memory context for Claude ──────────────────────────
    session_ctx = None
    strategic_ctx = None
    session_context_text = ""
    strategic_context_text = ""
    try:
        session_ctx = get_session_context(
            db_path=str(settings.session_memory_db_path),
        )
        session_context_text = build_session_paragraph(session_ctx)
    except Exception as exc:  # noqa: BLE001
        logger.debug("research[%s]: session context unavailable: %s", symbol, exc)

    try:
        strategic_ctx = get_strategic_context(
            db_path=str(settings.strategic_memory_db_path),
            symbol=symbol,
        )
        strategic_context_text = build_strategic_paragraph(
            strategic_ctx, symbol=symbol,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("research[%s]: strategic context unavailable: %s", symbol, exc)

    # ── Step 2e: Deterministic setup scoring ────────────────────────────────
    # The v2 SYSTEM_PROMPT tells Claude it is an "auditor reviewing a pre-scored
    # thesis" — so we MUST provide that thesis.  If no setup passes, skip the
    # LLM call entirely: saves cost and prevents phantom buy/sell signals.
    _open_symbols: set[str] = {
        str(p.get("symbol", "")).replace("/", "").upper()
        for p in (positions or [])
    }
    _signals_today = getattr(session_ctx, "signals_today", []) or []
    _narrative_text = getattr(session_ctx, "narrative", "") or ""

    # Build per-setup win-rate lookup from strategic memory.
    _strategic_win_rates: dict[str, float] = {}
    if strategic_ctx:
        from collections import defaultdict
        _perf_by_setup: dict[str, list[bool]] = defaultdict(list)
        for _entry in strategic_ctx.setup_performance:
            _perf_by_setup[str(_entry.setup_type)].append(_entry.win)
        for _stype, _wins in _perf_by_setup.items():
            if _wins:
                _strategic_win_rates[_stype] = sum(_wins) / len(_wins)

    indicators = Indicators(
        sma_20=sma_20,
        sma_50=sma_50,
        rsi_14=rsi_14,
        atr_14=atr_14,
        volume_ratio=volume_ratio,
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

    _setup_results = []
    try:
        _setup_results = scan_setups(
            symbol=symbol,
            indicators=indicators,
            df=df,
            regime=regime,  # type: ignore[arg-type]
            open_symbols=_open_symbols,
            signals_today=_signals_today,
            strategic_win_rates=_strategic_win_rates,  # type: ignore[arg-type]
            narrative_text=_narrative_text,
            df_daily=df_daily if df_daily is not None and not df_daily.empty else None,
            as_of=as_of,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("research[%s]: setup scan failed: %s", symbol, exc)

    _passed = [r for r in _setup_results if r.passed]
    if not _passed:
        _reg = regime or "unknown"
        logger.info(
            "research[%s]: no qualifying setup (regime=%s) — skipping LLM call",
            symbol, _reg,
        )
        return _safe_hold(f"No qualifying setup for {symbol} (regime={_reg})")

    # Best-scoring passed setup is what Claude will audit.
    _best_setup = max(_passed, key=lambda r: r.score)
    setup_result_dict = _best_setup.model_dump(mode="json")

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
        macd_line=macd_line,
        macd_signal=macd_signal,
        macd_histogram=macd_histogram,
        bb_upper=bb_upper,
        bb_middle=bb_middle,
        bb_lower=bb_lower,
        stoch_k=stoch_k,
        stoch_d=stoch_d,
        vwap=vwap_val,
        daily_trend_context=daily_trend,
        signal_history=signal_history,
        prompt_version=settings.prompt_version,
        as_of_utc=as_of,
        is_crypto=market_data._is_crypto(symbol),
        session_context_text=session_context_text,
        strategic_context_text=strategic_context_text,
        setup_result=setup_result_dict,
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
                    "macd_line": macd_line,
                    "macd_signal": macd_signal,
                    "macd_histogram": macd_histogram,
                    "bb_upper": bb_upper,
                    "bb_middle": bb_middle,
                    "bb_lower": bb_lower,
                    "stoch_k": stoch_k,
                    "stoch_d": stoch_d,
                    "vwap": vwap_val,
                },
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    return signal
