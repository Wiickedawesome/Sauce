"""
exit_monitor.py — Eight-condition exit engine for Sauce.

Called every loop cycle for each open position. Returns the first triggered
exit condition (priority order) or None if no exit.

Priority order (first match wins):
  1. Hard stop       — price ≤ entry × (1 − stop_loss_pct)
  1b. ATR stop       — price ≤ entry - (ATR × atr_stop_multiple) [optional]
  2. Trailing stop   — price ≤ trailing_stop_price (once activated)
  3. Trail activate  — price ≥ entry × (1 + trail_activation_pct) → arm trailing
  4. Trail ratchet   — new high water mark → tighten trailing stop
  5. Profit target   — price ≥ entry × (1 + profit_target_pct)
  5b. ATR target     — price ≥ entry + (ATR × atr_target_multiple) [optional]
  6. RSI exhaustion  — rsi_14 ≥ rsi_exhaustion_threshold
  7. Time stop       — held > max_hold_hours AND gain < time_stop_min_gain
  8. Regime stop     — regime flipped to bearish (if plan.regime_stop=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sauce.strategy import ExitPlan, Position


@dataclass(frozen=True, slots=True)
class ExitSignal:
    """Describes why and how a position should be closed."""

    trigger: str  # one of the exit condition names
    symbol: str
    position_id: str
    side: str  # always "sell" for long exits
    current_price: float
    reason: str


def evaluate_exit(
    position: Position,
    plan: ExitPlan,
    current_price: float,
    rsi_14: float | None,
    now: datetime | None = None,
    atr_14: float | None = None,
    atr_stop_multiple: float = 2.0,
    atr_target_multiple: float = 3.0,
    regime: str | None = None,
) -> tuple[ExitSignal | None, Position]:
    """Check all exit conditions for one position.

    Args:
        position: Current position
        plan: Exit plan from strategy
        current_price: Current market price
        rsi_14: Current RSI(14) value
        now: Current timestamp (defaults to utcnow)
        atr_14: Current ATR(14) value for ATR-scaled stops
        atr_stop_multiple: ATR multiple for stop loss (default 2.0)
        atr_target_multiple: ATR multiple for profit target (default 3.0)
        regime: Current market regime for regime stop

    Returns (exit_signal_or_none, updated_position).
    The position is returned with possibly updated high_water_price,
    trailing_active, and trailing_stop_price fields.
    """
    if now is None:
        now = datetime.now(UTC)

    entry = position.entry_price
    symbol = position.symbol
    pid = position.id

    # ── 1a. Hard stop (percentage-based) ─────────────────────────────────
    hard_stop_price = entry * (1 - plan.stop_loss_pct)
    if current_price <= hard_stop_price:
        return (
            ExitSignal(
                trigger="hard_stop",
                symbol=symbol,
                position_id=pid,
                side="sell",
                current_price=current_price,
                reason=f"Price {current_price:.4f} hit hard stop {hard_stop_price:.4f} "
                f"({plan.stop_loss_pct:.1%} below entry {entry:.4f})",
            ),
            position,
        )

    # ── 1b. ATR stop (volatility-scaled) ─────────────────────────────────
    if atr_14 is not None and atr_14 > 0:
        atr_stop_price = entry - (atr_14 * atr_stop_multiple)
        if current_price <= atr_stop_price:
            return (
                ExitSignal(
                    trigger="atr_stop",
                    symbol=symbol,
                    position_id=pid,
                    side="sell",
                    current_price=current_price,
                    reason=f"Price {current_price:.4f} hit ATR stop {atr_stop_price:.4f} "
                    f"({atr_stop_multiple}× ATR {atr_14:.4f} below entry {entry:.4f})",
                ),
                position,
            )

    # ── 2. Trailing stop (if armed) ──────────────────────────────────────
    if (
        position.trailing_active
        and position.trailing_stop_price is not None
        and current_price <= position.trailing_stop_price
    ):
        return (
            ExitSignal(
                trigger="trailing_stop",
                symbol=symbol,
                position_id=pid,
                side="sell",
                current_price=current_price,
                reason=f"Price {current_price:.4f} hit trailing stop "
                f"{position.trailing_stop_price:.4f}",
            ),
            position,
        )

    # ── 3. Trail activation ──────────────────────────────────────────────
    activation_price = entry * (1 + plan.trail_activation_pct)
    if not position.trailing_active and current_price >= activation_price:
        position.trailing_active = True
        position.high_water_price = current_price
        position.trailing_stop_price = current_price * (1 - plan.trail_pct)

    # ── 4. Trail ratchet (new high water mark) ───────────────────────────
    if position.trailing_active and current_price > position.high_water_price:
        position.high_water_price = current_price
        position.trailing_stop_price = current_price * (1 - plan.trail_pct)

    # ── 5. Profit target ─────────────────────────────────────────────────
    target_price = entry * (1 + plan.profit_target_pct)
    if current_price >= target_price:
        return (
            ExitSignal(
                trigger="profit_target",
                symbol=symbol,
                position_id=pid,
                side="sell",
                current_price=current_price,
                reason=f"Price {current_price:.4f} hit target {target_price:.4f} "
                f"({plan.profit_target_pct:.1%} above entry {entry:.4f})",
            ),
            position,
        )

    # ── 5b. ATR profit target (volatility-scaled) ────────────────────────
    if atr_14 is not None and atr_14 > 0:
        atr_target_price = entry + (atr_14 * atr_target_multiple)
        if current_price >= atr_target_price:
            return (
                ExitSignal(
                    trigger="atr_target",
                    symbol=symbol,
                    position_id=pid,
                    side="sell",
                    current_price=current_price,
                    reason=f"Price {current_price:.4f} hit ATR target {atr_target_price:.4f} "
                    f"({atr_target_multiple}× ATR {atr_14:.4f} above entry {entry:.4f})",
                ),
                position,
            )

    # ── 6. RSI exhaustion ────────────────────────────────────────────────
    if rsi_14 is not None and rsi_14 >= plan.rsi_exhaustion_threshold:
        return (
            ExitSignal(
                trigger="rsi_exhaustion",
                symbol=symbol,
                position_id=pid,
                side="sell",
                current_price=current_price,
                reason=f"RSI {rsi_14:.1f} ≥ exhaustion threshold "
                f"{plan.rsi_exhaustion_threshold:.0f}",
            ),
            position,
        )

    # ── 7. Time stop ────────────────────────────────────────────────────
    held_hours = (now - position.entry_time).total_seconds() / 3600
    if held_hours > plan.max_hold_hours:
        gain_pct = (current_price - entry) / entry if entry > 0 else 0
        if gain_pct < plan.time_stop_min_gain:
            return (
                ExitSignal(
                    trigger="time_stop",
                    symbol=symbol,
                    position_id=pid,
                    side="sell",
                    current_price=current_price,
                    reason=f"Held {held_hours:.1f}h (>{plan.max_hold_hours}h) "
                    f"with gain {gain_pct:.2%} < min {plan.time_stop_min_gain:.2%}",
                ),
                position,
            )

    # ── 8. Regime stop ───────────────────────────────────────────────────
    if plan.regime_stop and regime == "bearish":
        gain_pct = (current_price - entry) / entry if entry > 0 else 0
        # Only exit on regime flip if we're not already in profit
        if gain_pct < 0.05:
            return (
                ExitSignal(
                    trigger="regime_stop",
                    symbol=symbol,
                    position_id=pid,
                    side="sell",
                    current_price=current_price,
                    reason=f"Regime flipped to bearish with gain {gain_pct:.2%} < 5%",
                ),
                position,
            )

    return None, position
