"""
agents/options_exit.py — Momentum Snipe exit engine for options positions.

Implements a 7-condition deterministic exit evaluator (no LLM).

Priority order (first match wins):
  1. hard_stop      — loss exceeds max_loss_pct
  2. regime_stop    — market regime turned hostile (caller signals via kwarg)
  3. dte_stop       — remaining DTE below option_dte_exit_days
  4. time_stop      — held > option_time_stop_days with gain < min threshold
  5. trailing_stop  — price fell below trailing stop (activated at +20%)
  6. stretch_target — gain >= option_stretch_target_pct → FULL_CLOSE
  7. profit_target  — gain >= option_profit_target_pct → PARTIAL/FULL_CLOSE

If none match → HOLD.  High-water mark and trailing activation are updated
on every evaluation so the position tracks its peak.
"""

import logging
from datetime import date, datetime, timezone

from sauce.adapters.db import log_event
from sauce.core.options_config import get_options_settings
from sauce.core.options_schemas import (
    ExitDecision,
    OptionsPosition,
    OptionsQuote,
)
from sauce.core.options_safety import check_max_loss
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)


def evaluate_position(
    position: OptionsPosition,
    quote: OptionsQuote,
    loop_id: str = "unset",
    *,
    regime_hostile: bool = False,
) -> ExitDecision:
    """
    Evaluate an open options position and return an exit decision.

    Parameters
    ----------
    position:        The tracked options position.
    quote:           Live quote for the position's contract.
    loop_id:         Current loop UUID for audit correlation.
    regime_hostile:  True if the market regime engine signals danger.

    Returns
    -------
    ExitDecision: HOLD, PARTIAL_CLOSE, or FULL_CLOSE.
    """
    cfg = get_options_settings()
    current_price = quote.mid

    if position.remaining_qty <= 0:
        return ExitDecision(action="HOLD", reason="No remaining quantity")

    if current_price <= 0:
        return ExitDecision(action="HOLD", reason="Quote mid price is zero")

    gain_pct = (current_price - position.entry_price) / position.entry_price

    # ── 1. Hard stop: max loss ────────────────────────────────────────────
    passed, reason = check_max_loss(position, current_price, loop_id)
    if not passed:
        _log_exit(loop_id, position, "FULL_CLOSE", reason, "hard_stop")
        return ExitDecision(
            action="FULL_CLOSE",
            qty=position.remaining_qty,
            reason=reason,
            exit_type="hard_stop",
        )

    # ── 2. Regime stop ────────────────────────────────────────────────────
    if regime_hostile:
        reason = "Market regime turned hostile — closing options position"
        _log_exit(loop_id, position, "FULL_CLOSE", reason, "regime_stop")
        return ExitDecision(
            action="FULL_CLOSE",
            qty=position.remaining_qty,
            reason=reason,
            exit_type="regime_stop",
        )

    # ── 3. DTE stop ──────────────────────────────────────────────────────
    try:
        expiration = date.fromisoformat(position.contract.expiration)
        remaining_dte = (expiration - date.today()).days
        if remaining_dte <= cfg.option_dte_exit_days:
            reason = (
                f"DTE stop: {remaining_dte} days remaining "
                f"<= {cfg.option_dte_exit_days} threshold"
            )
            _log_exit(loop_id, position, "FULL_CLOSE", reason, "dte_stop")
            return ExitDecision(
                action="FULL_CLOSE",
                qty=position.remaining_qty,
                reason=reason,
                exit_type="dte_stop",
            )
    except (ValueError, TypeError):
        pass  # If expiration is invalid/placeholder, skip DTE check

    # ── 4. Time stop ─────────────────────────────────────────────────────
    if position.entry_time is not None:
        entry = position.entry_time
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
        days_held = (datetime.now(timezone.utc) - entry).days
        if days_held >= cfg.option_time_stop_days and gain_pct < cfg.option_time_stop_min_gain_pct:
            reason = (
                f"Time stop: held {days_held} days with gain "
                f"{gain_pct:.1%} < {cfg.option_time_stop_min_gain_pct:.0%} threshold"
            )
            _log_exit(loop_id, position, "FULL_CLOSE", reason, "time_stop")
            return ExitDecision(
                action="FULL_CLOSE",
                qty=position.remaining_qty,
                reason=reason,
                exit_type="time_stop",
            )

    # ── 5. Trailing stop check (if active) ────────────────────────────────
    if position.trailing_active and position.trailing_stop_price is not None:
        if current_price <= position.trailing_stop_price:
            reason = (
                f"Trailing stop hit: {current_price:.4f} <= "
                f"stop {position.trailing_stop_price:.4f}"
            )
            _log_exit(loop_id, position, "FULL_CLOSE", reason, "trailing_stop")
            return ExitDecision(
                action="FULL_CLOSE",
                qty=position.remaining_qty,
                reason=reason,
                exit_type="trailing_stop",
            )

    # ── Update high-water mark & activate trailing if threshold met ───────
    if position.high_water_price is None or current_price > position.high_water_price:
        position.high_water_price = current_price

    if not position.trailing_active and gain_pct >= cfg.option_trail_activation_pct:
        position.trailing_active = True
        position.trailing_stop_price = current_price * (1 - cfg.option_trail_pct)

    if position.trailing_active and position.high_water_price is not None:
        new_stop = position.high_water_price * (1 - cfg.option_trail_pct)
        if position.trailing_stop_price is None or new_stop > position.trailing_stop_price:
            position.trailing_stop_price = new_stop

    # ── 6. Stretch target: close everything at +60% ──────────────────────
    if gain_pct >= cfg.option_stretch_target_pct:
        reason = (
            f"Stretch target hit: gain {gain_pct:.1%} >= "
            f"{cfg.option_stretch_target_pct:.0%}"
        )
        _log_exit(loop_id, position, "FULL_CLOSE", reason, "stretch_target")
        return ExitDecision(
            action="FULL_CLOSE",
            qty=position.remaining_qty,
            reason=reason,
            exit_type="stretch_target",
        )

    # ── 7. Profit target: partial if qty>=2, else activate trail ─────────
    if gain_pct >= cfg.option_profit_target_pct:
        if position.remaining_qty >= 2:
            sell_qty = position.remaining_qty // 2
            reason = (
                f"Profit target hit: gain {gain_pct:.1%} >= "
                f"{cfg.option_profit_target_pct:.0%}, selling {sell_qty} of "
                f"{position.remaining_qty}"
            )
            _log_exit(loop_id, position, "PARTIAL_CLOSE", reason, "profit_target")
            return ExitDecision(
                action="PARTIAL_CLOSE",
                qty=sell_qty,
                reason=reason,
                exit_type="profit_target",
                set_trailing_stop=True,
                trailing_stop_pct=cfg.option_trail_pct,
            )
        else:
            # qty=1: just activate trailing stop, don't close yet
            if not position.trailing_active:
                position.trailing_active = True
                position.trailing_stop_price = current_price * (1 - cfg.option_trail_pct)
            return ExitDecision(
                action="HOLD",
                reason=(
                    f"Profit target hit at {gain_pct:.1%} with qty=1 "
                    "— activating trailing stop instead of partial close"
                ),
            )

    return ExitDecision(action="HOLD", reason="No exit condition met")


def build_exit_order(
    position: OptionsPosition,
    decision: ExitDecision,
    limit_price: float,
) -> "OptionsOrder | None":
    """
    Build an OptionsOrder from an exit decision.

    Returns None if decision is HOLD or qty is 0.
    """
    from sauce.core.options_schemas import OptionsOrder

    if decision.action == "HOLD" or decision.qty <= 0:
        return None

    source = "options_stop" if decision.exit_type == "hard_stop" else "options_exit"

    return OptionsOrder(
        contract_symbol=position.contract.contract_symbol,
        underlying=position.contract.underlying,
        qty=decision.qty,
        side="sell",
        limit_price=limit_price,
        source=source,
        exit_type=decision.exit_type,
        as_of=datetime.now(timezone.utc),
        prompt_version="exit-engine-v2",
    )


def _log_exit(
    loop_id: str,
    position: OptionsPosition,
    action: str,
    reason: str,
    exit_type: str,
) -> None:
    """Log an exit decision as an AuditEvent."""
    event_type = "options_exit_stop" if exit_type == "hard_stop" else "options_exit"
    log_event(AuditEvent(
        loop_id=loop_id,
        event_type=event_type,
        symbol=position.contract.underlying,
        payload={
            "position_id": position.position_id,
            "contract": position.contract.contract_symbol,
            "action": action,
            "reason": reason,
            "exit_type": exit_type,
            "remaining_qty": position.remaining_qty,
        },
    ))
