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
import uuid
from datetime import datetime, timezone
from typing import Any

from sauce.adapters.broker import BrokerError, get_account, get_positions, place_order
from sauce.adapters.db import log_event
from sauce.adapters.market_data import get_universe_snapshot
from sauce.agents import execution, ops, portfolio, research, risk, supervisor
from sauce.core.config import get_settings
from sauce.core.safety import (
    check_daily_loss,
    check_market_hours,
    is_data_fresh,
    is_trading_paused,
)
from sauce.core.schemas import (
    AuditEvent,
    Order,
    RiskCheckResult,
    Signal,
    SupervisorDecision,
)

logger = logging.getLogger(__name__)


async def main() -> None:
    """
    Execute one full trading cycle.

    Entry point for `python -m sauce.core.loop` and cron.
    All exceptions are caught, logged, and re-raised so cron gets a non-zero exit.
    """
    loop_id = str(uuid.uuid4())
    settings = get_settings()

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

    try:
        await _run_loop(loop_id=loop_id, settings=settings)
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
        logger.info("Loop ended [loop_id=%s]", loop_id)


async def _run_loop(loop_id: str, settings: Any) -> None:
    """
    Inner loop body. Separated so main() can guarantee loop_end is always logged.
    """

    # ── Step 2: Safety pre-flight ─────────────────────────────────────────────
    if is_trading_paused(loop_id=loop_id):
        logger.warning("Trading is paused — aborting loop [loop_id=%s]", loop_id)
        return

    # ── Step 3: Account + positions ───────────────────────────────────────────
    try:
        account = get_account(loop_id=loop_id)
    except BrokerError as exc:
        logger.error("Cannot fetch account — aborting [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "get_account", "error": str(exc)},
        ))
        return

    if not check_daily_loss(account, loop_id=loop_id):
        logger.warning("Daily loss limit breached — aborting [loop_id=%s]", loop_id)
        return

    try:
        positions = get_positions(loop_id=loop_id)
    except BrokerError as exc:
        logger.error("Cannot fetch positions — aborting [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "get_positions", "error": str(exc)},
        ))
        return

    # ── Step 4: Universe quote snapshot ──────────────────────────────────────
    try:
        quotes = get_universe_snapshot(settings.full_universe)
    except Exception as exc:
        logger.error("Cannot fetch universe snapshot — aborting [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "get_universe_snapshot", "error": str(exc)},
        ))
        return

    # ── Step 5: Research agent — one signal per symbol ────────────────────────
    signals: list[Signal] = []

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

        try:
            signal = await research.run(symbol=symbol, quote=quote, loop_id=loop_id)
            signals.append(signal)
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="signal",
                symbol=symbol,
                payload={
                    "side": signal.side,
                    "confidence": signal.confidence,
                    "reasoning": signal.reasoning[:200],  # truncate for DB
                },
            ))
        except Exception as exc:
            logger.error("Research failed for %s [loop_id=%s]: %s", symbol, loop_id, exc)
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=symbol,
                payload={"stage": "research", "error": str(exc)},
            ))
            # Continue — a single symbol failure must not abort the whole run

    # ── Step 6: Risk agent — checks per actionable signal ────────────────────
    risk_results: list[RiskCheckResult] = []

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

        try:
            risk_result = await risk.run(
                signal=signal,
                account=account,
                positions=positions,
                loop_id=loop_id,
            )
            risk_results.append(risk_result)
            log_event(AuditEvent(
                loop_id=loop_id,
                event_type="risk_check",
                symbol=signal.symbol,
                payload={
                    "veto": risk_result.veto,
                    "reason": risk_result.reason,
                    "qty": risk_result.qty,
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
    try:
        await portfolio.run(
            symbols=settings.full_universe,
            positions=positions,
            signals=signals,
            loop_id=loop_id,
        )
    except Exception as exc:
        logger.error("Portfolio agent failed [loop_id=%s]: %s", loop_id, exc)
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"stage": "portfolio", "error": str(exc)},
        ))

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
    if decision.action == "execute":
        for order in decision.final_orders:
            try:
                result = place_order(order=order, loop_id=loop_id)
                log_event(AuditEvent(
                    loop_id=loop_id,
                    event_type="fill",
                    symbol=order.symbol,
                    payload={"broker_response": result},
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
    else:
        logger.info(
            "Supervisor decision: abort — no orders placed [loop_id=%s]. Reason: %s",
            loop_id, decision.reason,
        )

    # ── Step 11: Ops agent — write audit trail ────────────────────────────────
    try:
        await ops.run(
            loop_id=loop_id,
            summary={
                "signals_total": len(signals),
                "signals_actionable": sum(1 for s in signals if s.side != "hold"),
                "risk_results": len(risk_results),
                "risk_vetoes": sum(1 for r in risk_results if r.veto),
                "orders_prepared": len(orders),
                "supervisor_action": decision.action,
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
