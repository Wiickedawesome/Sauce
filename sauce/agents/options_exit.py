"""
agents/options_exit.py — Compounding exit engine for options positions.

Implements the "Double Up & Take Gains" strategy:
  Entry → 2x → sell 50% → 4x → sell 50% of rest → 8x → sell/trail.

This is a DETERMINISTIC engine (no LLM). All exit decisions are based on
the compound stage ladder defined at entry time.

Sequence for each open options position:
  1. Fetch live quote for the contract.
  2. Check hard stop (max loss).
  3. Walk the compound stage ladder — trigger next stage if reached.
  4. If all stages completed, manage trailing stop.
  5. Return an ExitDecision (HOLD / PARTIAL_CLOSE / FULL_CLOSE).
"""

import logging
from datetime import datetime, timezone

from sauce.adapters.db import log_event
from sauce.core.options_config import get_options_settings
from sauce.core.options_schemas import (
    CompoundStage,
    ExitDecision,
    OptionsOrder,
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
) -> ExitDecision:
    """
    Evaluate an open options position and return an exit decision.

    Parameters
    ----------
    position:  The tracked options position with its compound stage ladder.
    quote:     Live quote for the position's contract.
    loop_id:   Current loop UUID for audit correlation.

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

    # ── Hard stop: max loss ───────────────────────────────────────────────
    passed, reason = check_max_loss(position, current_price, loop_id)
    if not passed:
        _log_exit(loop_id, position, "FULL_CLOSE", reason, 0)
        return ExitDecision(
            action="FULL_CLOSE",
            qty=position.remaining_qty,
            reason=reason,
            stage=0,
        )

    # ── Trailing stop check (if active) ───────────────────────────────────
    if position.trailing_stop_price is not None:
        if current_price <= position.trailing_stop_price:
            reason = (
                f"Trailing stop hit: {current_price:.4f} <= "
                f"stop {position.trailing_stop_price:.4f}"
            )
            _log_exit(loop_id, position, "FULL_CLOSE", reason, position.stages_completed)
            return ExitDecision(
                action="FULL_CLOSE",
                qty=position.remaining_qty,
                reason=reason,
                stage=position.stages_completed,
            )

        # Update trailing stop if price has risen
        # trailing_stop_pct comes from the last completed stage
        last_stage = _get_last_completed_stage(position)
        if last_stage:
            new_stop = current_price * (1 - last_stage.trailing_stop_pct)
            if new_stop > position.trailing_stop_price:
                position.trailing_stop_price = new_stop

    # ── Walk compound stages ──────────────────────────────────────────────
    for stage in position.compound_stages:
        if stage.completed:
            continue

        trigger_price = position.entry_price * stage.trigger_multiplier

        if current_price >= trigger_price:
            # Stage triggered!
            sell_qty = max(1, int(position.remaining_qty * stage.sell_fraction))

            # Don't sell more than remaining
            sell_qty = min(sell_qty, position.remaining_qty)

            # Is this the last stage or would we have 0 remaining?
            remaining_after = position.remaining_qty - sell_qty
            if remaining_after <= 0:
                # Full close on final stage
                _log_exit(
                    loop_id, position, "FULL_CLOSE",
                    f"Stage {stage.stage_num} triggered at {current_price:.4f} "
                    f"(target {trigger_price:.4f}), closing all",
                    stage.stage_num,
                )
                return ExitDecision(
                    action="FULL_CLOSE",
                    qty=position.remaining_qty,
                    reason=f"Stage {stage.stage_num}: {stage.trigger_multiplier}x target hit, closing remaining",
                    stage=stage.stage_num,
                    set_trailing_stop=False,
                )

            # Partial close
            _log_exit(
                loop_id, position, "PARTIAL_CLOSE",
                f"Stage {stage.stage_num} triggered at {current_price:.4f} "
                f"(target {trigger_price:.4f}), selling {sell_qty}",
                stage.stage_num,
            )
            return ExitDecision(
                action="PARTIAL_CLOSE",
                qty=sell_qty,
                reason=(
                    f"Stage {stage.stage_num}: {stage.trigger_multiplier}x target hit, "
                    f"selling {sell_qty} of {position.remaining_qty}"
                ),
                stage=stage.stage_num,
                set_trailing_stop=True,
                trailing_stop_pct=stage.trailing_stop_pct,
            )

        # Price hasn't reached this stage yet — stop checking further stages
        break

    return ExitDecision(action="HOLD", reason="No exit condition met")


def apply_exit_decision(
    position: OptionsPosition,
    decision: ExitDecision,
) -> OptionsPosition:
    """
    Apply an ExitDecision to a position, updating its state.

    Returns the updated OptionsPosition (mutated in place).
    This is called after the broker confirms the exit order.
    """
    if decision.action == "HOLD":
        return position

    if decision.action in ("PARTIAL_CLOSE", "FULL_CLOSE"):
        sold_qty = decision.qty
        position.remaining_qty = max(0, position.remaining_qty - sold_qty)

        # Mark the stage as completed
        if decision.stage > 0:
            for stage in position.compound_stages:
                if stage.stage_num == decision.stage and not stage.completed:
                    stage.completed = True
                    position.stages_completed += 1
                    break

        # Set trailing stop after partial close
        if decision.set_trailing_stop and decision.trailing_stop_pct > 0:
            # We don't have current_price here, so the caller must set it
            # after this method returns. We store the percentage for reference.
            pass

    return position


def build_exit_order(
    position: OptionsPosition,
    decision: ExitDecision,
    limit_price: float,
) -> OptionsOrder | None:
    """
    Build an OptionsOrder from an exit decision.

    Returns None if decision is HOLD or qty is 0.
    """
    if decision.action == "HOLD" or decision.qty <= 0:
        return None

    source = "options_stop" if decision.stage == 0 else "options_exit"

    return OptionsOrder(
        contract_symbol=position.contract.contract_symbol,
        underlying=position.contract.underlying,
        qty=decision.qty,
        side="sell",
        limit_price=limit_price,
        stage=decision.stage,
        source=source,
        as_of=datetime.now(timezone.utc),
        prompt_version="exit-engine-v1",
    )


def _get_last_completed_stage(
    position: OptionsPosition,
) -> CompoundStage | None:
    """Return the most recently completed stage, or None."""
    completed = [s for s in position.compound_stages if s.completed]
    if not completed:
        return None
    return max(completed, key=lambda s: s.stage_num)


def _log_exit(
    loop_id: str,
    position: OptionsPosition,
    action: str,
    reason: str,
    stage: int,
) -> None:
    """Log an exit decision as an AuditEvent."""
    event_type = "options_exit_stop" if stage == 0 else "options_exit_stage"
    log_event(AuditEvent(
        loop_id=loop_id,
        event_type=event_type,
        symbol=position.contract.underlying,
        payload={
            "position_id": position.position_id,
            "contract": position.contract.contract_symbol,
            "action": action,
            "reason": reason,
            "stage": stage,
            "remaining_qty": position.remaining_qty,
        },
    ))
