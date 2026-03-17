"""
agents/options_execution.py — Options execution agent.

Deterministic engine (no LLM call). Translates an OptionsSignal into an
OptionsOrder for entry, or an ExitDecision + OptionsOrder for exit.

Rules:
  - NEVER market orders. Limit only.
  - Freshness re-check: reject stale quotes.
  - Spread check: reject if bid/ask spread is too wide.
  - Returns None on any failure — never raises to the loop.
"""

import logging
from datetime import datetime, timezone

from sauce.adapters.db import log_event
from sauce.core.options_config import get_options_settings
from sauce.core.options_schemas import (
    OptionsOrder,
    OptionsQuote,
    OptionsSignal,
)
from sauce.core.safety import is_data_fresh
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)

# Slight offset from mid to improve fill probability
_LIMIT_PRICE_OFFSET = 0.01  # options spreads wider than equities


async def build_entry_order(
    signal: OptionsSignal,
    quote: OptionsQuote,
    loop_id: str = "unset",
) -> OptionsOrder | None:
    """
    Build a limit buy order from an OptionsSignal.

    Returns None on any precondition failure.
    """
    cfg = get_options_settings()

    # ── Freshness ──────────────────────────────────────────────────────
    from sauce.core.config import get_settings
    settings = get_settings()

    if not is_data_fresh(quote.as_of, ttl_sec=settings.data_ttl_seconds):
        _log_veto(loop_id, signal, f"Quote stale: {quote.as_of.isoformat()}")
        return None

    # ── Spread check ───────────────────────────────────────────────────
    if quote.mid > 0:
        spread_pct = (quote.ask - quote.bid) / quote.mid
        if spread_pct > cfg.option_max_spread_pct:
            _log_veto(
                loop_id, signal,
                f"Spread too wide: {spread_pct:.2%} > {cfg.option_max_spread_pct:.2%}",
            )
            return None

    # ── Limit price: slightly above mid for buys ───────────────────────
    limit_price = round(quote.mid * (1.0 + _LIMIT_PRICE_OFFSET), 2)
    # Clamp to ask — never pay more than the ask
    limit_price = min(limit_price, quote.ask)
    if limit_price <= 0:
        _log_veto(loop_id, signal, f"Invalid limit price: {limit_price}")
        return None

    # ── Qty from affordable sizing ─────────────────────────────────────
    cost_per_contract = limit_price * 100  # options = 100 shares
    if signal.max_position_cost > 0 and cost_per_contract > 0:
        qty = max(1, min(
            cfg.option_max_contracts,
            int(signal.max_position_cost / cost_per_contract),
        ))
    else:
        qty = 1

    order = OptionsOrder(
        contract_symbol=signal.contract.contract_symbol,
        underlying=signal.symbol,
        qty=qty,
        side="buy",
        limit_price=limit_price,
        source="options_entry",
        as_of=datetime.now(timezone.utc),
        prompt_version=signal.prompt_version,
    )

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="options_order",
        symbol=signal.symbol,
        payload={
            "agent": "options_execution",
            "action": "entry_order_prepared",
            "contract": order.contract_symbol,
            "side": "buy",
            "qty": qty,
            "limit_price": limit_price,
        },
    ))

    return order


async def build_exit_order(
    contract_symbol: str,
    underlying: str,
    qty: int,
    quote: OptionsQuote,
    exit_type: str = "",
    loop_id: str = "unset",
    prompt_version: str = "exit-engine-v2",
) -> OptionsOrder | None:
    """
    Build a limit sell order for an exit decision.

    Returns None on any precondition failure.
    """
    from sauce.core.config import get_settings
    settings = get_settings()

    if qty <= 0:
        return None

    if not is_data_fresh(quote.as_of, ttl_sec=settings.data_ttl_seconds):
        logger.warning(
            "options_execution: stale quote for exit order %s", contract_symbol,
        )
        return None

    # Limit price: slightly below mid for sells
    limit_price = round(quote.mid * (1.0 - _LIMIT_PRICE_OFFSET), 2)
    # Floor to bid — never sell below the bid
    limit_price = max(limit_price, quote.bid)
    if limit_price <= 0:
        return None

    source = "options_stop" if exit_type == "hard_stop" else "options_exit"

    order = OptionsOrder(
        contract_symbol=contract_symbol,
        underlying=underlying,
        qty=qty,
        side="sell",
        limit_price=limit_price,
        source=source,
        exit_type=exit_type,
        as_of=datetime.now(timezone.utc),
        prompt_version=prompt_version,
    )

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="options_order",
        symbol=underlying,
        payload={
            "agent": "options_execution",
            "action": "exit_order_prepared",
            "contract": contract_symbol,
            "side": "sell",
            "qty": qty,
            "limit_price": limit_price,
            "exit_type": exit_type,
        },
    ))

    return order


def _log_veto(loop_id: str, signal: OptionsSignal, reason: str) -> None:
    """Log an execution veto for an options signal."""
    logger.info(
        "options_execution[%s]: veto — %s", signal.symbol, reason,
    )
    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="veto",
        symbol=signal.symbol,
        payload={"agent": "options_execution", "reason": reason},
    ))
