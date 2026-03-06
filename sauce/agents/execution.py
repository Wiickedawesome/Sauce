"""
agents/execution.py — Execution agent: translates an approved signal into an Order.

Phase 5 IMPLEMENTATION.

Sequence:
  1. Pre-condition checks: veto flag, qty > 0, side in {buy, sell}.
  2. Freshness re-check: quote must not be stale at this point.
  3. Price deviation check: if quote moved > max_price_deviation vs. evidence → abort.
  4. Ask Claude for order-type and limit-price recommendation.
  5. Sanity-check the returned limit price against the live bid/ask.
  6. Construct and return a validated Order, or None on any failure.

Notes:
  - NEVER places market orders. limit or stop_limit only.
  - On ANY failure, logs a veto AuditEvent and returns None.
  - Returns None (not an exception) so the loop can continue with other symbols.
"""

import json
import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from sauce.adapters import llm
from sauce.adapters.db import log_event
from sauce.core.config import get_settings
from sauce.core.safety import is_data_fresh
from sauce.core.schemas import AuditEvent, Order, PriceReference, RiskCheckResult, Signal
from sauce.prompts import execution as execution_prompts

logger = logging.getLogger(__name__)


async def run(
    signal: Signal,
    risk_result: RiskCheckResult,
    quote: PriceReference,
    loop_id: str,
) -> Order | None:
    """
    Translate an approved signal into an Order.

    Returns an Order if all checks pass, or None if any check fails.

    Parameters
    ----------
    signal:      Approved Signal from Research.
    risk_result: Approved RiskCheckResult (veto must be False).
    quote:       Fresh price reference fetched immediately before this call.
    loop_id:     Current loop run UUID for audit correlation.
    """
    settings = get_settings()
    db_path = str(settings.db_path)
    as_of = datetime.now(timezone.utc)

    def _abort(reason: str) -> None:
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="veto",
                symbol=signal.symbol,
                payload={"agent": "execution", "reason": reason},
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return None

    # ── Pre-condition checks ───────────────────────────────────────────────
    if risk_result.veto:
        return _abort("Risk agent vetoed this signal — should not have reached Execution.")

    if risk_result.qty is None or risk_result.qty <= 0:
        return _abort(f"Risk-approved qty is {risk_result.qty} — cannot size order.")

    if signal.side not in ("buy", "sell"):
        return _abort(f"Signal side is '{signal.side}' — only buy/sell proceed to Execution.")

    # ── Step 1: Freshness re-check ────────────────────────────────────────────
    if not is_data_fresh(quote.as_of, ttl_sec=settings.data_ttl_seconds):
        return _abort(
            f"Quote is stale for {signal.symbol} "
            f"(as_of={quote.as_of.isoformat()}, ttl={settings.data_ttl_seconds}s)"
        )

    # ── Step 2: Price deviation check ─────────────────────────────────────────
    evidence_mid = signal.evidence.price_reference.mid
    if evidence_mid > 0:
        deviation = abs(quote.mid - evidence_mid) / evidence_mid
        if deviation > settings.max_price_deviation:
            return _abort(
                f"Price moved {deviation:.2%} from evidence mid "
                f"(evidence_mid={evidence_mid}, live_mid={quote.mid}, "
                f"max_allowed={settings.max_price_deviation:.2%})"
            )

    # ── Step 3: Ask Claude for order parameters ─────────────────────────────────
    atr_14 = signal.evidence.indicators.atr_14
    user_prompt = execution_prompts.build_user_prompt(
        symbol=signal.symbol,
        side=signal.side,
        qty=risk_result.qty,
        bid=quote.bid,
        ask=quote.ask,
        mid=quote.mid,
        atr_14=atr_14,
        signal_reasoning=signal.reasoning,
        prompt_version=settings.prompt_version,
        as_of_utc=as_of,
    )

    try:
        raw_response = await llm.call_claude(
            system=execution_prompts.SYSTEM_PROMPT,
            user=user_prompt,
            loop_id=loop_id,
        )
    except llm.LLMError as exc:
        logger.error("execution[%s]: LLM call failed: %s", signal.symbol, exc)
        return _abort(f"LLM error: {exc}")

    # ── Step 4: Parse LLM response ─────────────────────────────────────────────
    try:
        data = json.loads(raw_response.strip())
        order_type = str(data.get("order_type", "limit")).lower()
        time_in_force = str(data.get("time_in_force", "day")).lower()
        raw_limit_price: float | None = (
            float(data["limit_price"]) if data.get("limit_price") is not None else None
        )
        raw_stop_price: float | None = (
            float(data["stop_price"]) if data.get("stop_price") is not None else None
        )
    except (json.JSONDecodeError, TypeError, ValueError, KeyError) as exc:
        logger.warning(
            "execution[%s]: failed to parse LLM response: %s | raw=%r",
            signal.symbol, exc, raw_response[:200],
        )
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=signal.symbol,
                payload={
                    "agent": "execution",
                    "error": f"JSON parse failed: {exc}",
                    "raw_response_preview": raw_response[:200],
                },
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return _abort("LLM response parse error")

    # ── Step 5: Sanity-check price ───────────────────────────────────────────────
    # Limit price must stay within ±max_limit_price_premium of bid/ask (Finding 1.9).
    # Default 0.1% keeps prices from straying far from the market while still
    # tolerating tiny floating-point rounding in Claude's output.
    limit_price = raw_limit_price
    if limit_price is not None:
        premium = settings.max_limit_price_premium
        lo_bound = quote.bid * (1.0 - premium)
        hi_bound = quote.ask * (1.0 + premium)
        if not (lo_bound <= limit_price <= hi_bound):
            logger.warning(
                "execution[%s]: Claude limit_price %.4f out of sane range [%.4f, %.4f]; "
                "falling back to bid/ask.",
                signal.symbol, limit_price, lo_bound, hi_bound,
            )
            limit_price = quote.ask if signal.side == "buy" else quote.bid
    else:
        # Claude returned null — use ask for buys, bid for sells
        limit_price = quote.ask if signal.side == "buy" else quote.bid

    # Validate order_type; only allow limit and stop_limit
    if order_type not in ("limit", "stop_limit"):
        order_type = "limit"
    if time_in_force not in ("day", "gtc", "ioc", "fok"):
        time_in_force = "day"

    # ── Step 6: Construct validated Order ───────────────────────────────────────
    try:
        order = Order(
            symbol=signal.symbol,
            side=signal.side,  # type: ignore[arg-type]
            qty=risk_result.qty,
            order_type=order_type,  # type: ignore[arg-type]
            time_in_force=time_in_force,  # type: ignore[arg-type]
            limit_price=limit_price,
            stop_price=raw_stop_price,
            as_of=as_of,
            prompt_version=settings.prompt_version,
        )
    except ValidationError as exc:
        logger.error("execution[%s]: Order validation failed: %s", signal.symbol, exc)
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="error",
                symbol=signal.symbol,
                payload={"agent": "execution", "error": f"Order validation: {exc}"},
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return None

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="order_prepared",
            symbol=signal.symbol,
            payload={
                "agent": "execution",
                "order_type": order_type,
                "side": signal.side,
                "qty": risk_result.qty,
                "limit_price": limit_price,
                "stop_price": raw_stop_price,
                "time_in_force": time_in_force,
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    return order
