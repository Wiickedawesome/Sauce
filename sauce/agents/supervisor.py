"""
agents/supervisor.py — Supervisor agent: final gate before broker submission.

Phase 5 IMPLEMENTATION.

The Supervisor is the LAST check before orders are sent to the broker.
It first performs four hard pre-flight checks (independent of Claude),
then calls Claude for a final holistic sanity review.

Pre-flight checks (abort immediately if any fail):
  1. Orders list is non-empty.
  2. Trading is not paused (TRADING_PAUSE flag).
  3. Every order’s underlying signal has a fresh quote.
  4. Every order has an approved (non-vetoed) RiskCheckResult.

Claude gate:
  - If pre-flight passes: prompt Claude with orders, signals summary, account.
  - Claude must return {action: "execute" | "abort", vetoes: [...], reason: "..."}.
  - Any parse/validation failure → abort (safe default).
  - Supervisor can NEVER upgrade an abort to execute.
"""

import json
import logging
from datetime import datetime, timezone

from sauce.adapters import llm
from sauce.adapters.db import log_event
from sauce.core.config import get_settings
from sauce.core.safety import is_data_fresh, is_trading_paused
from sauce.core.schemas import AuditEvent, Order, RiskCheckResult, Signal, SupervisorDecision
from sauce.prompts import supervisor as supervisor_prompts

logger = logging.getLogger(__name__)


async def run(
    orders: list[Order],
    signals: list[Signal],
    risk_results: list[RiskCheckResult],
    account: dict,
    loop_id: str,
) -> SupervisorDecision:
    """
    Perform pre-flight checks then ask Claude for a final execute/abort decision.

    Returns SupervisorDecision(action="execute", final_orders=orders) if all
    checks pass and Claude approves. Otherwise returns action="abort".

    Parameters
    ----------
    orders:       Candidate orders prepared by the Execution agent.
    signals:      All signals from the Research agent (same loop run).
    risk_results: All risk results from the Risk agent (same loop run).
    account:      Account dict from broker.get_account().
    loop_id:      Current loop run UUID for audit correlation.
    """
    settings = get_settings()
    db_path = str(settings.db_path)
    as_of = datetime.now(timezone.utc)

    def _abort(reason: str, vetoes: list[str] | None = None) -> SupervisorDecision:
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="supervisor_decision",
                symbol=None,
                payload={
                    "agent": "supervisor",
                    "action": "abort",
                    "reason": reason,
                    "vetoes": vetoes or [],
                },
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return SupervisorDecision(
            action="abort",
            final_orders=[],
            vetoes=vetoes or [],
            reason=reason,
            as_of=as_of,
            prompt_version=settings.prompt_version,
        )

    # ── Pre-flight 1: Orders list must be non-empty ───────────────────────────────
    if not orders:
        return _abort("No orders to evaluate — nothing to approve.")

    # ── Pre-flight 2: TRADING_PAUSE flag ──────────────────────────────────────
    if is_trading_paused(loop_id=loop_id):
        return _abort(
            "TRADING_PAUSE is active — all orders blocked.",
            vetoes=[o.symbol for o in orders],
        )

    # ── Pre-flight 3: Quote freshness for each order ────────────────────────────
    signal_by_symbol: dict[str, Signal] = {
        s.symbol.upper(): s for s in signals
    }
    stale_symbols: list[str] = []
    for order in orders:
        sig = signal_by_symbol.get(order.symbol.upper())
        if sig is None:
            stale_symbols.append(order.symbol)
            continue
        if not is_data_fresh(
            sig.evidence.price_reference.as_of,
            ttl_sec=settings.data_ttl_seconds,
        ):
            stale_symbols.append(order.symbol)

    if stale_symbols:
        return _abort(
            f"Stale quotes for {stale_symbols} — aborting all orders.",
            vetoes=stale_symbols,
        )

    # ── Pre-flight 4: Risk approval for each order ─────────────────────────────
    approved_risk: dict[str, RiskCheckResult] = {
        r.symbol.upper(): r for r in risk_results if not r.veto
    }
    unapproved: list[str] = [
        o.symbol for o in orders if o.symbol.upper() not in approved_risk
    ]
    if unapproved:
        return _abort(
            f"Orders for {unapproved} have no approved risk check — aborting all.",
            vetoes=unapproved,
        )

    # ── Claude gate ─────────────────────────────────────────────────────────────
    orders_dicts = [o.model_dump(mode="json") for o in orders]
    signals_summary = [
        {
            "symbol": s.symbol,
            "side": s.side,
            "confidence": s.confidence,
            "reasoning": s.reasoning,
        }
        for s in signals
        if s.side != "hold"
    ]
    user_prompt = supervisor_prompts.build_user_prompt(
        orders=orders_dicts,
        signals_summary=signals_summary,
        account=account,
        prompt_version=settings.prompt_version,
        as_of_utc=as_of,
    )

    try:
        raw_response = await llm.call_claude(
            system=supervisor_prompts.SYSTEM_PROMPT,
            user=user_prompt,
            loop_id=loop_id,
        )
    except llm.LLMError as exc:
        logger.error("supervisor: LLM call failed: %s", exc)
        return _abort(f"LLM error: {exc}", vetoes=[o.symbol for o in orders])

    # ── Parse Claude’s response ───────────────────────────────────────────────
    try:
        data = json.loads(raw_response.strip())
        claude_action = str(data.get("action", "abort")).lower()
        claude_vetoes: list[str] = [
            str(v) for v in (data.get("vetoes") or [])
        ]
        claude_reason = str(data.get("reason", ""))
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        logger.warning(
            "supervisor: failed to parse LLM response: %s | raw=%r",
            exc, raw_response[:200],
        )
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=None,
                payload={
                    "agent": "supervisor",
                    "error": f"JSON parse failed: {exc}",
                    "raw_response_preview": raw_response[:200],
                },
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return _abort("LLM response parse error")

    # Supervisor can NEVER upgrade an abort to execute.
    if claude_action != "execute":
        return _abort(
            claude_reason or "Claude returned non-execute action.",
            vetoes=claude_vetoes,
        )

    # ── Approved: build final decision ───────────────────────────────────────────
    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="supervisor_decision",
            symbol=None,
            payload={
                "agent": "supervisor",
                "action": "execute",
                "order_count": len(orders),
                "vetoes": claude_vetoes,
                "reason": claude_reason,
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    return SupervisorDecision(
        action="execute",
        final_orders=orders,
        vetoes=claude_vetoes,
        reason=claude_reason,
        as_of=as_of,
        prompt_version=settings.prompt_version,
    )
