"""
agents/execution.py — Execution agent: translates an approved signal into an Order.

Phase 5 IMPLEMENTATION — Deterministic Engine (no LLM call).

Sequence:
  1. Pre-condition checks: veto flag, qty > 0, side in {buy, sell}.
  2. Freshness re-check: quote must not be stale at this point.
  3. Price deviation check: if quote moved > max_price_deviation vs. evidence → abort.
  4. Deterministic limit-price calculation from bid/ask spread.
  5. Construct and return a validated Order, or None on any failure.

Notes:
  - NEVER places market orders. limit only.
  - On ANY failure, logs a veto AuditEvent and returns None.
  - Returns None (not an exception) so the loop can continue with other symbols.
"""

import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from sauce.adapters.db import log_event
from sauce.core.config import get_settings
from sauce.core.safety import is_data_fresh
from sauce.core.schemas import AuditEvent, Order, PriceReference, RiskCheckResult, Signal

logger = logging.getLogger(__name__)

# Slight premium/discount from the top-of-book to improve fill probability
# while keeping execution close to the market.
_LIMIT_PRICE_OFFSET = 0.0005


async def run(
    signal: Signal,
    risk_result: RiskCheckResult,
    quote: PriceReference,
    loop_id: str,
) -> Order | None:
    """
    Translate an approved signal into an Order using deterministic logic.

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

    # ── Step 3: Deterministic limit-price calculation ─────────────────────────
    # Buy: slightly above ask to improve fill probability.
    # Sell: slightly below bid for the same reason.
    if signal.side == "buy":
        limit_price = round(quote.ask * (1.0 + _LIMIT_PRICE_OFFSET), 4)
    else:
        limit_price = round(quote.bid * (1.0 - _LIMIT_PRICE_OFFSET), 4)

    # Sanity-check: clamp within configurable premium band around bid/ask
    premium = settings.max_limit_price_premium
    lo_bound = quote.bid * (1.0 - premium)
    hi_bound = quote.ask * (1.0 + premium)
    if not (lo_bound <= limit_price <= hi_bound):
        logger.warning(
            "execution[%s]: calculated limit_price %.4f out of range [%.4f, %.4f]; "
            "falling back to bid/ask.",
            signal.symbol, limit_price, lo_bound, hi_bound,
        )
        limit_price = quote.ask if signal.side == "buy" else quote.bid

    # ── Step 4: Construct validated Order ───────────────────────────────────────
    try:
        order = Order(
            symbol=signal.symbol,
            side=signal.side,  # type: ignore[arg-type]
            qty=risk_result.qty,
            order_type="limit",
            time_in_force="day",
            limit_price=limit_price,
            stop_price=None,
            as_of=as_of,
            prompt_version=settings.prompt_version,
            source="execution",
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
                "order_type": "limit",
                "side": signal.side,
                "qty": risk_result.qty,
                "limit_price": limit_price,
                "time_in_force": "day",
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    return order
