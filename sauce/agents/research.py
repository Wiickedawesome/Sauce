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

from pydantic import ValidationError

from sauce.adapters import llm, market_data
from sauce.indicators.core import compute_all, compute_sma, compute_rsi
from sauce.adapters.db import get_recent_signals, log_event
from sauce.memory.db import get_session_context, get_strategic_context
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, Evidence, Indicators, PriceReference, Signal
from sauce.core.setups import scan_setups
from sauce.prompts import research as research_prompts
from sauce.prompts.context import build_session_paragraph, build_strategic_paragraph
from sauce.signals.confluence import compute_confluence
from sauce.signals.timeframes import fetch_multi_timeframe

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
    allowed_setups: list[str] | None = None,
    score_offset: float = 0.0,
    crypto_regime_filter: list[str] | None = None,
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

    # Detect asset class early — needed to choose correct bar count.
    _is_crypto = market_data.is_crypto(symbol)

    def _safe_hold(reason: str, indicators: Indicators | None = None) -> Signal:
        """Return a safe hold signal with an attached audit log entry."""
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="signal",
                symbol=symbol,
                payload={"agent": "research", "side": "hold", "confidence": 0.0, "reasoning": reason},
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
                indicators=indicators or Indicators(),
                as_of=quote.as_of,
            ),
            reasoning=f"Hold (safe default): {reason}",
            as_of=as_of,
            prompt_version=settings.prompt_version,
        )

    # ── Step 1: Fetch OHLCV history ───────────────────────────────────────────
    # Crypto needs 500 bars so Setup 1 H3 (4hr SMA50) can compute:
    # 500 / 16 = ~31 4hr bars from 15-min data; use 1000 for 62+ 4hr bars.
    _bars = 1000 if _is_crypto else 120
    try:
        df = market_data.get_history(symbol, timeframe="15Min", bars=_bars)
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
    indicators = compute_all(df, is_crypto=_is_crypto)

    sma_20 = indicators.sma_20
    sma_50 = indicators.sma_50
    rsi_14 = indicators.rsi_14
    atr_14 = indicators.atr_14
    volume_ratio = indicators.volume_ratio
    macd_line = indicators.macd_line
    macd_signal = indicators.macd_signal
    macd_histogram = indicators.macd_histogram
    bb_upper = indicators.bb_upper
    bb_middle = indicators.bb_middle
    bb_lower = indicators.bb_lower
    stoch_k = indicators.stoch_k
    stoch_d = indicators.stoch_d
    vwap_val = indicators.vwap

    # ── Step 2b: Fetch daily bars for higher-timeframe trend context ──────────
    df_daily = None  # declared here so step 2e (setup scoring) can reference it
    daily_trend: dict | None = None
    try:
        df_daily = market_data.get_history(symbol, timeframe="1Day", bars=50)
        if not df_daily.empty and len(df_daily) >= 20:
            d_close = df_daily["close"]
            d_sma_20 = compute_sma(d_close, 20)
            d_sma_50 = (
                compute_sma(d_close, 50)
                if len(df_daily) >= 50
                else None
            )
            d_rsi_14 = compute_rsi(d_close, 14)
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
        pass  # Daily data unavailable — proceed with 15min only

    # ── Step 2c: Fetch recent signal history for feedback loop ─────────────
    signal_history: list[dict] | None = None
    try:
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

    # ── Step 2d2: Retrieve similar past trades (RAG) ──────────────────────
    similar_trades_text = ""
    try:
        from sauce.memory.db import get_similar_trades
        from sauce.prompts.context import build_similar_trades_paragraph

        similar = get_similar_trades(
            db_path=str(settings.strategic_memory_db_path),
            symbol=symbol,
            regime=regime,
        )
        if similar:
            similar_trades_text = build_similar_trades_paragraph(similar)
            logger.info(
                "research[%s]: retrieved %d similar past trades",
                symbol, len(similar),
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("research[%s]: similar trades unavailable: %s", symbol, exc)

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
            allowed_setups=allowed_setups,
            score_offset=score_offset,
            crypto_regime_filter=crypto_regime_filter,
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
        return _safe_hold(f"No qualifying setup for {symbol} (regime={_reg})", indicators)

    # Send ALL passed setups to Claude so it can weigh competing theses.
    # Sorted by score descending so the strongest thesis appears first.
    _passed_sorted = sorted(_passed, key=lambda r: r.score, reverse=True)
    all_setup_results = [r.model_dump(mode="json") for r in _passed_sorted]

    # ── Step 2f: Multi-timeframe analysis + confluence scoring ────────────
    # Runs AFTER the setup gate so we only spend 5 API calls on symbols that
    # actually passed a setup (saves ~400 calls/loop for symbols that hold).
    mtf_context = None
    confluence = None
    try:
        mtf_context = fetch_multi_timeframe(symbol, is_crypto=_is_crypto)
        if mtf_context.analyses:
            confluence = compute_confluence(mtf_context)
            logger.info(
                "research[%s]: multi-TF %s (score=%.2f, tier=%s)",
                symbol, confluence.summary, confluence.score, confluence.tier.value,
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("research[%s]: multi-TF analysis unavailable: %s", symbol, exc)

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
        is_crypto=market_data.is_crypto(symbol),
        session_context_text=session_context_text,
        strategic_context_text=strategic_context_text,
        setup_results=all_setup_results,
        positions=positions,
        multi_timeframe_context=(
            mtf_context.to_prompt_dict() if mtf_context and mtf_context.analyses else None
        ),
        confluence_result=(
            confluence.to_prompt_dict() if confluence else None
        ),
        similar_trades_text=similar_trades_text,
    )

    # ── Step 4: Call Claude ───────────────────────────────────────────────────
    try:
        raw_response = await llm.call_claude(
            system=research_prompts.SYSTEM_PROMPT,
            user=user_prompt,
            loop_id=loop_id,
            temperature=settings.research_temperature,
        )
    except llm.LLMError as exc:
        logger.error("research[%s]: LLM call failed: %s", symbol, exc)
        return _safe_hold(f"LLM error: {exc}", indicators)

    # ── Step 5: Parse response ────────────────────────────────────────────────
    try:
        data = json.loads(raw_response.strip())
        side = str(data.get("side", "hold")).lower()
        if side not in ("buy", "sell", "hold"):
            side = "hold"
        confidence = float(data.get("confidence", 0.0))
        # Clamp to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))
        # Apply multi-timeframe confluence adjustment
        if confluence:
            raw_conf = confidence
            confidence = max(0.0, min(1.0, confidence + confluence.confidence_adjustment))
            if raw_conf != confidence:
                logger.info(
                    "research[%s]: confidence adjusted %.2f → %.2f (tier=%s)",
                    symbol, raw_conf, confidence, confluence.tier.value,
                )
        reasoning = str(data.get("reasoning", ""))
        bear_case = str(data.get("bear_case", ""))
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
        return _safe_hold("LLM response parse error", indicators)

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
            bear_case=bear_case,
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
        return _safe_hold("Signal schema validation error", indicators)

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
                "confluence": confluence.to_prompt_dict() if confluence else None,
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    return signal
