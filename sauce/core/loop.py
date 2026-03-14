"""
core/loop.py — Main orchestration loop for the Sauce trading system.

This module is the single entry point for a complete trading cycle.
It is called by `scripts/run_loop.sh` via cron every 30 minutes.

Loop sequence (per run):
  1.  Log loop_start
  2.  Safety pre-flight: paused? daily loss?
  3.  Fetch account + current positions
  4.  Fetch universe snapshot (quotes for all symbols)
  5.  For each symbol: check market hours + data freshness → call Research agent
  6.  For each non-hold signal above min_confidence: call Risk agent
  7.  For each approved risk result: check freshness → call Execution agent
  8.  Portfolio agent review (suggestions only, no orders)
  9.  Supervisor agent: final arbitration
  10. For each approved order in SupervisorDecision: place_order()
  11. Ops agent: write audit trail
  12. Log loop_end (always — even on error, via finally)

Rules:
  - TRADING_PAUSE is checked first. If paused → immediately return.
  - A single bad symbol must NOT abort the whole run.
  - Any exception in the main body aborts remaining work for this run.
  - broker.place_order() is NEVER called without Supervisor approval.
  - AuditEvent is logged on every stage entry.
"""

import asyncio
import logging
import logging.handlers
import signal
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sauce.adapters.broker import BrokerError, cancel_stale_orders, get_account, get_positions, place_order
from sauce.adapters.db import OrderRow, has_recent_submitted_order, log_event, log_order
from sauce.adapters.market_data import get_universe_snapshot
from sauce.agents import execution, exit_research, market_context, ops, portfolio, research, risk, session_boot, supervisor
from sauce.agents.debate import run_debate
from sauce.core.capital import detect_tier_transition, get_tier_parameters
from sauce.core.config import get_settings
from sauce.core.safety import (
    check_daily_loss,
    check_market_hours,
    has_earnings_risk,
    is_data_fresh,
    is_trading_paused,
)
from sauce.core.schemas import (
    AuditEvent,
    BootContext,
    ExitSignal,
    Indicators,
    MarketContext,
    Order,
    PortfolioReview,
    RiskCheckResult,
    Signal,
    SupervisorDecision,
)
from sauce.memory.learning import record_trade_outcome

logger = logging.getLogger(__name__)

# Graceful shutdown: set by SIGTERM/SIGINT handler, checked between pipeline stages.
_shutdown_requested = False

# Track open positions across loop iterations to detect closures.
# Maps normalized symbol → last-seen position dict snapshot.
_previous_positions: dict[str, dict[str, Any]] = {}


def _detect_closed_positions(
    current_positions: list[dict[str, Any]],
    loop_id: str,
    regime: str,
    db_path: str,
) -> None:
    """Compare current positions with previous snapshot and record outcomes for closures."""
    global _previous_positions  # noqa: PLW0603
    from sauce.core.schemas import SetupPerformanceEntry

    current_symbols = {
        _canonicalize_symbol(str(p.get("symbol", ""))): p
        for p in current_positions
    }

    for sym, prev in _previous_positions.items():
        if sym not in current_symbols:
            # Position was closed since last loop
            pnl = float(prev.get("unrealized_pl", 0.0))
            try:
                entry = SetupPerformanceEntry(
                    setup_type="equity_trend_pullback",  # best-effort default
                    symbol=sym,
                    regime_at_entry=regime or "unknown",  # type: ignore[arg-type]
                    time_of_day_bucket=datetime.now(timezone.utc).strftime("%H:00"),
                    win=pnl > 0,
                    pnl=round(pnl, 2),
                    hold_duration_minutes=0.0,  # unknown from position data alone
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                )
                record_trade_outcome(entry, db_path)
                logger.info(
                    "Recorded closed position outcome: %s pnl=%.2f win=%s [loop_id=%s]",
                    sym, pnl, pnl > 0, loop_id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to record trade outcome for %s: %s [loop_id=%s]",
                    sym, exc, loop_id,
                )

    _previous_positions = current_symbols


def _canonicalize_symbol(symbol: str) -> str:
    """Normalize broker and strategy symbols for reconciliation lookups."""
    return symbol.replace("/", "").upper()


def configure_logging() -> None:
    """Set up structured logging with rotation for the Sauce trading system.

    Called once at the start of main(). Configures:
    - Root logger at INFO level so all module loggers emit their messages.
    - Structured format with timestamp, level, logger name, and message.
    - Rotating file handler: 5 MB max per file, 3 backups, in data/logs/.
    - Stream handler for stdout (captured by cron redirect).
    """
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = lambda *_: datetime.now(timezone.utc).timetuple()

    root = logging.getLogger()
    # Avoid adding duplicate handlers on repeated calls (e.g. tests)
    if root.handlers:
        return
    root.setLevel(logging.INFO)

    # Rotating file handler — keeps disk usage bounded
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "sauce.log",
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Stream handler for stdout (picked up by cron.log)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


async def main() -> None:
    """
    Execute one full trading cycle.

    Entry point for `python -m sauce.core.loop` and cron.
    All exceptions are caught, logged, and re-raised so cron gets a non-zero exit.
    """
    loop_id = str(uuid.uuid4())
    settings = get_settings()
    configure_logging()

    # Register graceful shutdown handler for SIGTERM (Docker stop) and SIGINT (Ctrl+C)
    def _handle_shutdown(signum: int, frame: object) -> None:
        global _shutdown_requested
        sig_name = signal.Signals(signum).name
        logger.warning("Received %s — will shut down after current stage [loop_id=%s]", sig_name, loop_id)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="loop_start",
        payload={
            "universe": settings.full_universe,
            "paper": settings.alpaca_paper,
            "prompt_version": settings.prompt_version,
        },
    ))

    logger.info("Loop started [loop_id=%s]", loop_id)

    # ── Agent 0: Session Boot ─────────────────────────────────────────────
    boot_ctx = await session_boot.run(loop_id=loop_id)
    logger.info("Session boot complete [loop_id=%s, suppressed=%s]", loop_id, boot_ctx.is_suppressed)

    try:
        await asyncio.wait_for(
            _run_loop(loop_id=loop_id, settings=settings, boot_ctx=boot_ctx),
            timeout=settings.loop_timeout_seconds,
        )
    except TimeoutError:
        logger.critical(
            "Loop timed out after %ds [loop_id=%s]",
            settings.loop_timeout_seconds, loop_id,
        )
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={
                "stage": "main",
                "error": f"Loop timed out after {settings.loop_timeout_seconds}s",
                "type": "TimeoutError",
            },
        ))
    except Exception as exc:
        logger.exception("Unhandled exception in loop [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "main", "error": str(exc), "type": type(exc).__name__},
        ))
        raise
    finally:
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="loop_end",
            payload={"timestamp": datetime.now(timezone.utc).isoformat()},
        ))
        # Lightweight DB maintenance (integrity check, audit pruning, VACUUM)
        try:
            from sauce.adapters.db import run_maintenance
            run_maintenance()
        except Exception:  # noqa: BLE001
            logger.warning("DB maintenance skipped due to error", exc_info=True)
        logger.info("Loop ended [loop_id=%s]", loop_id)


async def _run_loop(loop_id: str, settings: Any, boot_ctx: BootContext) -> None:
    """
    Inner loop body. Separated so main() can guarantee loop_end is always logged.
    """

    # ── Step 1b: Safety pre-flight ────────────────────────────────────────────
    if is_trading_paused(loop_id=loop_id):
        logger.warning("Trading is paused — aborting loop [loop_id=%s]", loop_id)
        return

    if boot_ctx.is_suppressed:
        logger.warning(
            "Macro-event suppression active — aborting loop before research [loop_id=%s]",
            loop_id,
        )
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="safety_check",
            payload={
                "check": "macro_suppression",
                "result": True,
                "reason": "major economic event suppression window active",
            },
        ))
        return

    # ── Agent 1: Market Context ───────────────────────────────────────────────
    # Cancel any unfilled orders from previous runs before new cycle begins
    try:
        stale_cancelled = await asyncio.to_thread(
            cancel_stale_orders,
            max_age_minutes=settings.stale_order_cancel_minutes,
            loop_id=loop_id,
        )
        if stale_cancelled:
            logger.info(
                "Cancelled %d stale order(s) [loop_id=%s]", stale_cancelled, loop_id,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Stale order cancellation failed: %s [loop_id=%s]", exc, loop_id)

    mkt_ctx = await market_context.run(loop_id=loop_id, boot_ctx=boot_ctx)
    logger.info(
        "Market context ready [loop_id=%s, regime=%s, dead=%s]",
        loop_id, mkt_ctx.regime.regime_type, mkt_ctx.is_dead,
    )

    # ── Step 3: Account + positions ───────────────────────────────────────────
    try:
        account = await asyncio.to_thread(get_account, loop_id)
    except BrokerError as exc:
        logger.error("Cannot fetch account — aborting [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "get_account", "error": str(exc)},
        ))
        return

    # ── Step 3b: Capital tier detection ────────────────────────────────────
    try:
        _equity = float(account.get("equity") or 0.0)
        tier_params = get_tier_parameters(_equity)
    except (TypeError, ValueError):
        _equity = 0.0
        tier_params = get_tier_parameters(500.0)  # fallback to seed tier

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="tier_transition",
        payload={
            "tier": tier_params.tier,
            "equity": _equity,
            "max_position_pct": tier_params.max_position_pct,
            "max_daily_loss_pct": tier_params.max_daily_loss_pct,
            "max_positions": tier_params.max_positions,
        },
    ))
    logger.info(
        "Capital tier: %s (equity=$%.2f, max_pos=%.1f%%, max_loss=%.1f%%)",
        tier_params.tier, _equity,
        tier_params.max_position_pct * 100,
        tier_params.max_daily_loss_pct * 100,
    )

    if not check_daily_loss(account, loop_id=loop_id, max_daily_loss_pct=tier_params.max_daily_loss_pct):
        logger.warning("Daily loss limit breached — aborting [loop_id=%s]", loop_id)
        return

    try:
        positions = await asyncio.to_thread(get_positions, loop_id)
    except BrokerError as exc:
        logger.error("Cannot fetch positions — aborting [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "get_positions", "error": str(exc)},
        ))
        return

    # Detect positions closed since last loop and record trade outcomes (Gap 4).
    _detect_closed_positions(
        positions,
        loop_id=loop_id,
        regime=mkt_ctx.regime.regime_type,
        db_path=str(settings.db_path),
    )

    # ── Step 4: Universe quote snapshot ──────────────────────────────────────
    try:
        quotes = await asyncio.to_thread(get_universe_snapshot, settings.full_universe)
    except Exception as exc:
        logger.error("Cannot fetch universe snapshot — aborting [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "get_universe_snapshot", "error": str(exc)},
        ))
        return

    # ── Step 5: Research agent — one signal per symbol ────────────────────────
    if _shutdown_requested:
        logger.warning("Shutdown requested — exiting before research stage [loop_id=%s]", loop_id)
        return
    signals: list[Signal] = []

    # First pass: synchronously filter eligible symbols (market hours + quote freshness).
    _eligible: list[tuple[str, Any]] = []
    for symbol in settings.full_universe:

        # Skip if market is closed for this symbol
        if not check_market_hours(symbol=symbol, loop_id=loop_id):
            logger.debug("Market closed for %s — skipping", symbol)
            continue

        # Skip if quote is missing or stale
        quote = quotes.get(symbol)
        if quote is None:
            logger.warning("No quote for %s — skipping", symbol)
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=symbol,
                payload={"stage": "research_preflight", "error": "no quote"},
            ))
            continue

        if not is_data_fresh(quote.as_of, settings.data_ttl_seconds):
            logger.warning("Stale quote for %s (as_of=%s) — skipping", symbol, quote.as_of)
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=symbol,
                payload={
                    "stage": "research_preflight",
                    "error": "stale quote",
                    "as_of": quote.as_of.isoformat(),
                },
            ))
            continue

        # Skip symbols within the earnings blackout window (Finding 2.6).
        if has_earnings_risk(symbol, loop_id=loop_id):
            logger.info("Earnings proximity detected for %s — skipping this run", symbol)
            continue

        _eligible.append((symbol, quote))

    # Second pass: run all eligible research calls in parallel (Finding 7.1).
    # return_exceptions=True ensures one failure doesn’t cancel the others.
    if _eligible:
        _research_results = await asyncio.gather(
            *[
                research.run(
                    symbol=sym,
                    quote=q,
                    loop_id=loop_id,
                    regime=mkt_ctx.regime.regime_type,
                    positions=positions,
                )
                for sym, q in _eligible
            ],
            return_exceptions=True,
        )
        for (sym, _), result in zip(_eligible, _research_results):
            if isinstance(result, BaseException):
                logger.error("Research failed for %s [loop_id=%s]: %s", sym, loop_id, result)
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="error",
                    symbol=sym,
                    payload={"stage": "research", "error": str(result)},
                ))
            else:
                signal: Signal = result
                signals.append(signal)
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="signal",
                    symbol=sym,
                    payload={
                        "side": signal.side,
                        "confidence": signal.confidence,
                        "reasoning": signal.reasoning[:200],  # truncate for DB
                    },
                ))

    # ── Step 5b: Bull/Bear debate — deterministic opposing analysis ───────────
    debate_results: dict[str, object] = {}  # symbol → DebateResult
    for signal in signals:
        if signal.side == "hold":
            continue
        if signal.confidence < settings.min_confidence:
            continue
        try:
            debate = run_debate(signal)
            debate_results[signal.symbol] = debate
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="debate",
                symbol=signal.symbol,
                payload={
                    "verdict": debate.verdict,
                    "bull_score": debate.bull_score,
                    "bear_score": debate.bear_score,
                    "confidence_adjustment": debate.confidence_adjustment,
                    "summary": debate.summary()[:300],
                },
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Debate failed for %s [loop_id=%s]: %s", signal.symbol, loop_id, exc)

    # ── Step 6: Risk agent — checks per actionable signal ────────────────────
    risk_results: list[RiskCheckResult] = []

    # Track buying power consumed by buy orders approved so far this run.
    # Each approved buy depletes this pool so subsequent signals cannot
    # double-claim the same cash (Finding 1.3).
    try:
        _loop_buying_power: float = float(account.get("buying_power") or 0.0)
    except (TypeError, ValueError):
        _loop_buying_power = 0.0

    # Symbols that actually went through the risk check (non-hold, above
    # min_confidence). Used by ops.py to compute the veto-rate anomaly.
    _symbols_attempted: list[str] = []

    for signal in signals:
        # Short-circuit: hold signals and low-confidence signals skip risk
        if signal.side == "hold":
            continue
        if signal.confidence < settings.min_confidence:
            logger.debug(
                "Signal for %s below min_confidence (%.2f < %.2f) — skipping",
                signal.symbol, signal.confidence, settings.min_confidence,
            )
            continue

        _symbols_attempted.append(signal.symbol)
        try:
            risk_result = await risk.run(
                signal=signal,
                account=account,
                positions=positions,
                loop_id=loop_id,
                remaining_buying_power=_loop_buying_power,
                max_position_pct=tier_params.max_position_pct,
                max_daily_loss_pct=tier_params.max_daily_loss_pct,
            )
            risk_results.append(risk_result)
            # Deduct committed notional from the running buying-power pool so
            # the next buy signal in this loop run sees the correct headroom.
            if not risk_result.veto and risk_result.qty and signal.side == "buy":
                committed = risk_result.qty * signal.evidence.price_reference.mid
                _loop_buying_power = max(0.0, _loop_buying_power - committed)
                logger.debug(
                    "Buying power after %s approval: $%.2f (committed $%.2f)",
                    signal.symbol, _loop_buying_power, committed,
                )
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="risk_check",
                symbol=signal.symbol,
                payload={
                    "veto": risk_result.veto,
                    "reason": risk_result.reason,
                    "qty": risk_result.qty,
                    "remaining_buying_power": _loop_buying_power,
                },
            ))
        except Exception as exc:
            logger.error("Risk check failed for %s [loop_id=%s]: %s", signal.symbol, loop_id, exc)
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=signal.symbol,
                payload={"stage": "risk", "error": str(exc)},
            ))

    # ── Step 7: Execution agent — build orders for approved signals ───────────
    if _shutdown_requested:
        logger.warning("Shutdown requested — exiting before execution stage [loop_id=%s]", loop_id)
        return
    orders: list[Order] = []

    # Build a lookup: symbol → signal (needed to pass to execution agent)
    signal_by_symbol: dict[str, Signal] = {s.symbol: s for s in signals}

    for risk_result in risk_results:
        if risk_result.veto:
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="veto",
                symbol=risk_result.symbol,
                payload={"reason": risk_result.reason, "stage": "risk"},
            ))
            continue

        matching_signal = signal_by_symbol.get(risk_result.symbol)
        if matching_signal is None:
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=risk_result.symbol,
                payload={"stage": "execution_preflight", "error": "signal not found"},
            ))
            continue

        quote = quotes.get(risk_result.symbol)
        if quote is None or not is_data_fresh(quote.as_of, settings.data_ttl_seconds):
            logger.warning(
                "Stale/missing quote at execution stage for %s — vetoing",
                risk_result.symbol,
            )
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="veto",
                symbol=risk_result.symbol,
                payload={"reason": "stale or missing quote at execution stage"},
            ))
            continue

        try:
            order = await execution.run(
                signal=matching_signal,
                risk_result=risk_result,
                quote=quote,
                loop_id=loop_id,
            )
            if order is not None:
                # ── Attach ATR-based stop-loss / take-profit ──────────────
                _atr = matching_signal.evidence.indicators.atr_14
                if _atr and _atr > 0 and order.side == "buy":
                    _ref = quote.mid
                    order.stop_loss_price = round(
                        _ref - _atr * settings.stop_loss_atr_multiple, 4,
                    )
                    order.take_profit_price = round(
                        _ref + _atr * settings.profit_target_atr_multiple, 4,
                    )
                orders.append(order)
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="order",
                    symbol=order.symbol,
                    payload={
                        "side": order.side,
                        "qty": order.qty,
                        "order_type": order.order_type,
                        "limit_price": order.limit_price,
                    },
                ))
        except Exception as exc:
            logger.error(
                "Execution failed for %s [loop_id=%s]: %s",
                risk_result.symbol, loop_id, exc,
            )
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=risk_result.symbol,
                payload={"stage": "execution", "error": str(exc)},
            ))

    # ── Step 8: Portfolio agent — suggestions only ───────────────────────────
    # Capture the review so the Supervisor can see current exposure (Finding 2.2).
    portfolio_review: PortfolioReview | None = None
    try:
        equity_usd = float(account.get("equity") or 0.0)
        portfolio_review = await portfolio.run(
            symbols=settings.full_universe,
            positions=positions,
            signals=signals,
            loop_id=loop_id,
            equity=equity_usd,
            max_position_pct=tier_params.max_position_pct,
        )
    except Exception as exc:
        logger.error("Portfolio agent failed [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "portfolio", "error": str(exc)},
        ))

    # ── Step 8b: Exit research — evaluate open positions for exits ──────────
    # Runs after portfolio review so we have context, before supervisor so exit
    # orders go through the same approval gate as buy orders.
    exit_orders: list[Order] = []
    if positions:
        # Build canonicalized lookups — broker returns "BTCUSD" but
        # universe quotes/signals use "BTC/USD".
        canon_quotes: dict[str, PriceReference] = {
            _canonicalize_symbol(sym): q for sym, q in quotes.items()
        }
        canon_signals: dict[str, Signal] = {
            _canonicalize_symbol(s.symbol): s for s in signals
        }
        canon_to_universe: dict[str, str] = {
            _canonicalize_symbol(sym): sym for sym in quotes
        }
        for pos in positions:
            pos_symbol = str(pos.get("symbol", "")).upper()
            pos_canon = _canonicalize_symbol(pos_symbol)
            pos_qty = float(pos.get("qty") or 0.0)
            if pos_qty <= 0:
                continue  # skip zero or short positions (no short selling yet)

            pos_quote = canon_quotes.get(pos_canon)
            if pos_quote is None:
                continue

            # Extract indicators from this run's signal if available
            sig = canon_signals.get(pos_canon)
            pos_indicators = sig.evidence.indicators if sig else Indicators()

            # Try to extract entry time for stale-hold check
            entry_time = None
            for _ts_key in ("created_at", "avg_entry_timestamp"):
                _ts_val = pos.get(_ts_key)
                if _ts_val is not None:
                    try:
                        entry_time = datetime.fromisoformat(
                            str(_ts_val).replace("Z", "+00:00")
                        )
                    except (ValueError, TypeError):
                        pass
                    break

            try:
                exit_sig: ExitSignal = await exit_research.run(
                    position=pos,
                    quote=pos_quote,
                    indicators=pos_indicators,
                    regime=mkt_ctx.regime.regime_type,
                    loop_id=loop_id,
                    entry_time=entry_time,
                )
            except Exception as exc:
                logger.error(
                    "Exit research failed for %s [loop_id=%s]: %s",
                    pos_symbol, loop_id, exc,
                )
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="error",
                    symbol=pos_symbol,
                    payload={"stage": "exit_research", "error": str(exc)},
                ))
                continue

            if exit_sig.action == "exit":
                # Use universe symbol (BTC/USD) so broker detects crypto
                # and auto-converts time_in_force from day→gtc.
                universe_sym = canon_to_universe.get(pos_canon, pos_symbol)
                _exit_order = Order(
                    symbol=universe_sym,
                    side="sell",
                    qty=pos_qty,
                    order_type="limit",
                    time_in_force="day",
                    limit_price=pos_quote.bid,
                    as_of=exit_sig.as_of,
                    prompt_version=exit_sig.prompt_version,
                )
                exit_orders.append(_exit_order)
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="order",
                    symbol=universe_sym,
                    payload={
                        "side": "sell",
                        "qty": pos_qty,
                        "order_type": "limit",
                        "limit_price": pos_quote.bid,
                        "source": "exit_research",
                        "exit_reason": exit_sig.reason,
                        "exit_urgency": exit_sig.urgency,
                    },
                ))

    # Merge exit orders into the main orders list for supervisor approval
    orders.extend(exit_orders)

    # ── Step 9: Supervisor — final arbitration ────────────────────────────────
    decision: SupervisorDecision = _make_abort_decision(
        reason="Supervisor not yet called",
        settings=settings,
    )

    try:
        decision = await supervisor.run(
            orders=orders,
            signals=signals,
            risk_results=risk_results,
            account=account,
            loop_id=loop_id,
            portfolio_review=portfolio_review,
            debate_results=debate_results,
        )
    except Exception as exc:
        logger.error("Supervisor failed [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "supervisor", "error": str(exc)},
        ))
        # decision stays as the safe abort default above

    # ── Step 10: Place approved orders ───────────────────────────────────────
    if _shutdown_requested:
        logger.warning("Shutdown requested — skipping order placement [loop_id=%s]", loop_id)
        return
    _orders_placed_count: int = 0
    _db_path = str(settings.db_path)
    if decision.action == "execute":
        for order in decision.final_orders:
            # ── Idempotency guard (Finding 3.2) ───────────────────────────────
            # If this loop crashed after submitting this order and cron restarted
            # it, an OrderRow with status='submitted' will already exist for
            # this symbol+side within the last 30 minutes. Detect and skip it
            # rather than sending a duplicate order to the broker.
            if has_recent_submitted_order(
                symbol=order.symbol,
                side=order.side,
                minutes=30,
                db_path=_db_path,
            ):
                logger.warning(
                    "Idempotency: skipping %s %s — order already submitted "
                    "within last 30 min [loop_id=%s]",
                    order.side, order.symbol, loop_id,
                )
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="veto",
                    symbol=order.symbol,
                    payload={
                        "reason": "idempotency: recent submitted order exists",
                        "stage": "place_order",
                    },
                ))
                continue

            try:
                result = await asyncio.to_thread(place_order, order, loop_id)
                _orders_placed_count += 1
                broker_order_id = str(
                    result.get("id") or result.get("order_id") or "unknown"
                )
                broker_status = str(result.get("status") or "submitted")

                # ── Persist order record (Finding 3.1) ─────────────────────────
                # Write a queryable OrderRow so we can reconcile against the
                # broker and detect duplicates on restart (Finding 3.2).
                log_order(
                    OrderRow(
                        loop_id=loop_id,
                        symbol=order.symbol,
                        side=order.side,
                        qty=order.qty,
                        order_type=order.order_type,
                        time_in_force=order.time_in_force,
                        limit_price=order.limit_price,
                        stop_price=order.stop_price,
                        status="submitted",
                        broker_order_id=broker_order_id,
                        prompt_version=order.prompt_version,
                    ),
                    db_path=_db_path,
                )

                # NOTE: place_order() confirms SUBMISSION, not fill.
                # event_type='order_submitted' is intentional (Finding 5.4).
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="order_submitted",
                    symbol=order.symbol,
                    payload={
                        "broker_order_id": broker_order_id,
                        "broker_status": broker_status,
                        "side": order.side,
                        "qty": order.qty,
                        "order_type": order.order_type,
                        "limit_price": order.limit_price,
                        "stop_price": order.stop_price,
                    },
                ))

                # ── Companion stop-loss order (P0-1) ──────────────────────
                if order.side == "buy" and order.stop_loss_price:
                    _sl_order = Order(
                        symbol=order.symbol,
                        side="sell",
                        qty=order.qty,
                        order_type="stop",
                        time_in_force="gtc",
                        stop_price=order.stop_loss_price,
                        as_of=order.as_of,
                        prompt_version=order.prompt_version,
                    )
                    try:
                        _sl_result = await asyncio.to_thread(
                            place_order, _sl_order, loop_id,
                        )
                        _sl_broker_id = str(
                            _sl_result.get("id")
                            or _sl_result.get("order_id")
                            or "unknown"
                        )
                        log_order(
                            OrderRow(
                                loop_id=loop_id,
                                symbol=_sl_order.symbol,
                                side=_sl_order.side,
                                qty=_sl_order.qty,
                                order_type=_sl_order.order_type,
                                time_in_force=_sl_order.time_in_force,
                                limit_price=None,
                                stop_price=_sl_order.stop_price,
                                status="submitted",
                                broker_order_id=_sl_broker_id,
                                prompt_version=_sl_order.prompt_version,
                            ),
                            db_path=_db_path,
                        )
                        log_event(AuditEvent(
                            loop_id=loop_id,
                            event_type="order_submitted",
                            symbol=_sl_order.symbol,
                            payload={
                                "broker_order_id": _sl_broker_id,
                                "side": "sell",
                                "qty": _sl_order.qty,
                                "order_type": "stop",
                                "stop_price": _sl_order.stop_price,
                                "purpose": "stop_loss",
                            },
                        ))
                        logger.info(
                            "Stop-loss submitted for %s @ %.4f [loop_id=%s]",
                            order.symbol, order.stop_loss_price, loop_id,
                        )
                    except BrokerError as sl_exc:
                        logger.error(
                            "Stop-loss order failed for %s [loop_id=%s]: %s",
                            order.symbol, loop_id, sl_exc,
                        )
                        log_event(AuditEvent(
                            loop_id=loop_id,
                            event_type="error",
                            symbol=order.symbol,
                            payload={
                                "stage": "stop_loss",
                                "error": str(sl_exc),
                                "stop_loss_price": order.stop_loss_price,
                            },
                        ))
            except BrokerError as exc:
                logger.error(
                    "Order placement failed for %s [loop_id=%s]: %s",
                    order.symbol, loop_id, exc,
                )
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="error",
                    symbol=order.symbol,
                    payload={"stage": "place_order", "error": str(exc)},
                ))
                # Never retry automatically — fail and log

        # ── Step 10b: Position reconciliation (Finding 3.3) ────────────────────────
        # Re-fetch positions immediately after submission and log what the broker
        # reports for each submitted symbol. Discrepancies surface broker rejects
        # and aid post-run analysis without blocking the hot path.
        if _orders_placed_count > 0:
            try:
                _reconciled = await asyncio.to_thread(get_positions, loop_id)
                _reconciled_by_sym = {
                    _canonicalize_symbol(str(p.get("symbol", ""))): p for p in _reconciled
                }
                for _order in decision.final_orders:
                    _rp = _reconciled_by_sym.get(_canonicalize_symbol(_order.symbol))
                    log_event(AuditEvent(
                        loop_id=loop_id,
                        event_type="reconciliation",
                        symbol=_order.symbol,
                        payload={
                            "stage": "position_reconciliation",
                            "order_side": _order.side,
                            "order_qty": _order.qty,
                            "position_confirmed": _rp is not None,
                            "position_qty": _rp.get("qty") if _rp else None,
                            "position_market_value": _rp.get("market_value") if _rp else None,
                        },
                    ))
                logger.info(
                    "Reconciliation: %d symbols checked after %d orders [loop_id=%s]",
                    len(decision.final_orders), _orders_placed_count, loop_id,
                )
            except BrokerError as exc:
                logger.warning("Position reconciliation failed [loop_id=%s]: %s", loop_id, exc)
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="error",
                    payload={"stage": "reconciliation", "error": str(exc)},
                ))
    else:
        logger.info(
            "Supervisor decision: abort — no orders placed [loop_id=%s]. Reason: %s",
            loop_id, decision.reason,
        )

    # ── Step 11: Ops agent — write audit trail ────────────────────────────────
    # Build veto_by_symbol so ops.py anomaly detection can flag persistently
    # problematic symbols (Finding 2.1).
    _veto_by_symbol: dict[str, int] = {}
    for _rr in risk_results:
        if _rr.veto:
            _veto_by_symbol[_rr.symbol] = _veto_by_symbol.get(_rr.symbol, 0) + 1

    try:
        await ops.run(
            loop_id=loop_id,
            summary={
                # Keys must exactly match ops.py field names (Finding 2.1).
                "signals_total": len(signals),
                "signals_buy_sell": sum(1 for s in signals if s.side != "hold"),
                "risk_vetoes": sum(1 for r in risk_results if r.veto),
                "orders_prepared": len(orders),
                "orders_placed": _orders_placed_count,
                "supervisor_action": decision.action,
                "symbols_attempted": _symbols_attempted,
                "veto_by_symbol": _veto_by_symbol,
                # Pass equity so ops.py can compute NAV and write daily_stats
                # (Finding 1.1, Finding 5.1).
                "equity_usd": float(account.get("equity") or 0.0),
            },
        )
    except Exception as exc:
        logger.error("Ops agent failed [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "ops", "error": str(exc)},
        ))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_abort_decision(reason: str, settings: Any) -> SupervisorDecision:
    """Build a safe abort SupervisorDecision for use as a default."""
    return SupervisorDecision(
        action="abort",
        final_orders=[],
        vetoes=["Fallback abort — supervisor did not complete"],
        reason=reason,
        as_of=datetime.now(timezone.utc),
        prompt_version=settings.prompt_version,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(main())
