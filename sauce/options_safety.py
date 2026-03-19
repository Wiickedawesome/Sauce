"""
options_safety.py — Risk and exit checks for options positions.

Five exit conditions:
1. Hard stop — premium dropped 50%+ from entry
2. Profit target — premium doubled (100% gain)
3. DTE threshold — approaching expiration (≤2 DTE)
4. Underlying reversal — underlying's RSI flipped against position
5. Time stop — holding >24h with no gains

Returns OptionsExitSignal with reason and urgency.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Literal

from sauce.core.config import get_settings
from sauce.core.options_schemas import OptionsExitSignal, OptionsPosition
from sauce.core.schemas import Indicators

logger = logging.getLogger(__name__)


def check_options_position(
    position: OptionsPosition,
    current_price: float,
    current_dte: int,
    underlying_indicators: Indicators | None = None,
) -> OptionsExitSignal | None:
    """
    Check if an options position should be exited.

    Args:
        position: The current options position
        current_price: Current premium (bid price)
        current_dte: Days to expiration remaining
        underlying_indicators: Current indicators for the underlying

    Returns:
        OptionsExitSignal if exit triggered, None otherwise
    """
    settings = get_settings()

    entry_price = position.entry_price
    pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
    entry_time = position.entry_time
    hours_held = (datetime.now(UTC) - entry_time).total_seconds() / 3600

    # === CONDITION 1: HARD STOP (50% loss) ===
    if pnl_pct <= -0.50:
        logger.warning(
            "OPTIONS HARD STOP: %s down %.1f%% (entry=%.2f current=%.2f)",
            position.contract_symbol,
            pnl_pct * 100,
            entry_price,
            current_price,
        )
        return OptionsExitSignal(
            position_id=position.position_id,
            contract_symbol=position.contract_symbol,
            reason="hard_stop",
            current_price=current_price,
            entry_price=entry_price,
            pnl_pct=pnl_pct,
            dte_remaining=current_dte,
            urgency="immediate",
        )

    # === CONDITION 2: PROFIT TARGET (100% gain) ===
    if pnl_pct >= 1.00:
        logger.info(
            "OPTIONS PROFIT TARGET: %s up %.1f%% (entry=%.2f current=%.2f)",
            position.contract_symbol,
            pnl_pct * 100,
            entry_price,
            current_price,
        )
        return OptionsExitSignal(
            position_id=position.position_id,
            contract_symbol=position.contract_symbol,
            reason="profit_target",
            current_price=current_price,
            entry_price=entry_price,
            pnl_pct=pnl_pct,
            dte_remaining=current_dte,
            urgency="immediate",
        )

    # === CONDITION 3: DTE THRESHOLD ===
    dte_threshold = settings.options_dte_exit_threshold
    if current_dte <= dte_threshold:
        logger.warning(
            "OPTIONS DTE EXIT: %s has %d DTE (threshold=%d)",
            position.contract_symbol,
            current_dte,
            dte_threshold,
        )
        # Exit by end of day to avoid pin risk / rapid theta decay
        urgency: Literal["immediate", "eod", "next_open"] = (
            "immediate" if current_dte <= 1 else "eod"
        )
        return OptionsExitSignal(
            position_id=position.position_id,
            contract_symbol=position.contract_symbol,
            reason="dte_threshold",
            current_price=current_price,
            entry_price=entry_price,
            pnl_pct=pnl_pct,
            dte_remaining=current_dte,
            urgency=urgency,
        )

    # === CONDITION 4: UNDERLYING REVERSAL ===
    if underlying_indicators is not None and underlying_indicators.rsi_14 is not None:
        rsi = underlying_indicators.rsi_14

        # If we have calls and RSI is now oversold (bearish shift)
        if position.option_type == "call" and rsi > 70:
            logger.info(
                "OPTIONS REVERSAL: %s call with underlying RSI=%.1f (overbought → reversal risk)",
                position.contract_symbol,
                rsi,
            )
            return OptionsExitSignal(
                position_id=position.position_id,
                contract_symbol=position.contract_symbol,
                reason="underlying_reversal",
                current_price=current_price,
                entry_price=entry_price,
                pnl_pct=pnl_pct,
                dte_remaining=current_dte,
                urgency="eod",
            )

        # If we have puts and RSI is now overbought (bullish shift)
        if position.option_type == "put" and rsi < 30:
            logger.info(
                "OPTIONS REVERSAL: %s put with underlying RSI=%.1f (oversold → reversal risk)",
                position.contract_symbol,
                rsi,
            )
            return OptionsExitSignal(
                position_id=position.position_id,
                contract_symbol=position.contract_symbol,
                reason="underlying_reversal",
                current_price=current_price,
                entry_price=entry_price,
                pnl_pct=pnl_pct,
                dte_remaining=current_dte,
                urgency="eod",
            )

    # === CONDITION 5: TIME STOP ===
    # If held >24h with no gains, exit stale position
    if hours_held > 24 and pnl_pct < 0.05:
        logger.info(
            "OPTIONS TIME STOP: %s held %.1fh with %.1f%% gain (stale)",
            position.contract_symbol,
            hours_held,
            pnl_pct * 100,
        )
        return OptionsExitSignal(
            position_id=position.position_id,
            contract_symbol=position.contract_symbol,
            reason="time_stop",
            current_price=current_price,
            entry_price=entry_price,
            pnl_pct=pnl_pct,
            dte_remaining=current_dte,
            urgency="next_open",
        )

    # No exit triggered
    return None


def check_options_allocation(
    total_options_value: float,
    equity: float,
) -> tuple[bool, str]:
    """
    Check if options allocation is within limits.

    Returns:
        (passed, reason) tuple
    """
    settings = get_settings()
    max_allocation = settings.max_options_allocation_pct

    if equity <= 0:
        return False, "zero equity"

    current_pct = total_options_value / equity
    if current_pct >= max_allocation:
        return False, f"options allocation {current_pct:.1%} >= limit {max_allocation:.1%}"

    return True, "ok"


def validate_options_order(
    premium_total: float,
    equity: float,
    existing_options_value: float,
) -> tuple[bool, str]:
    """
    Validate a new options order before submission.

    Args:
        premium_total: Total premium for the new order (contracts × price × 100)
        equity: Account equity
        existing_options_value: Current value of all options positions

    Returns:
        (passed, reason) tuple
    """
    settings = get_settings()

    # Check position size limit
    max_premium_pct = settings.options_max_premium_pct
    position_pct = premium_total / equity if equity > 0 else 1.0
    if position_pct > max_premium_pct:
        return False, f"premium {position_pct:.1%} exceeds limit {max_premium_pct:.1%}"

    # Check total allocation limit
    new_total = existing_options_value + premium_total
    max_allocation = settings.max_options_allocation_pct
    new_allocation_pct = new_total / equity if equity > 0 else 1.0
    if new_allocation_pct > max_allocation:
        return (
            False,
            f"new allocation {new_allocation_pct:.1%} would exceed limit {max_allocation:.1%}",
        )

    return True, "ok"
