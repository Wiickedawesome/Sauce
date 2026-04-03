"""
risk.py — Five-rule risk gate for Sauce.

Before every entry order, the loop calls `check_risk()`. If ANY rule fails,
the symbol is skipped (the loop continues to the next symbol).

Rules:
  1. Daily P&L   — today's realized + unrealized loss < tier.daily_loss_limit
  2. Position count — open positions < tier.max_concurrent
  3. Buying power — order value < available buying power
  4. Minimum order — order value must be at least $1 (prevents dust trades)
  5. Max portfolio exposure — total open positions × avg size < 80% of equity

Returns a RiskVerdict with pass/fail and the specific reason if failed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """Result of risk gate check."""

    passed: bool
    rule: str  # which rule was checked last (or "all" if passed)
    reason: str  # human-readable reason if failed, empty if passed


def check_risk(
    daily_pnl: float,
    equity: float,
    open_position_count: int,
    buying_power: float,
    order_value: float,
    daily_loss_limit: float,
    max_concurrent: int,
    total_existing_exposure: float = 0.0,
    max_portfolio_exposure: float = 0.80,
) -> RiskVerdict:
    """Run the five risk rules. Returns immediately on first failure.

    Parameters
    ----------
    daily_pnl : float
        Today's realized + unrealized P&L as a fraction of equity (negative = loss).
        Example: -0.04 means down 4%.
    equity : float
        Current account equity.
    open_position_count : int
        Number of currently open positions.
    buying_power : float
        Available buying power from the broker.
    order_value : float
        Dollar value of the proposed order (qty × limit_price).
    daily_loss_limit : float
        Maximum allowed daily loss as a fraction (from TierParams). Example: 0.08.
    max_concurrent : int
        Maximum allowed open positions (from TierParams).
    total_existing_exposure : float
        Sum of actual market values for existing open positions.
    max_portfolio_exposure : float
        Maximum fraction of equity that may be committed after the new order.
    """
    # Rule 1: Daily P&L gate
    if daily_pnl <= -daily_loss_limit:
        return RiskVerdict(
            passed=False,
            rule="daily_pnl",
            reason=f"Daily loss {daily_pnl:.2%} exceeds limit {-daily_loss_limit:.2%}",
        )

    # Rule 2: Position count gate
    if open_position_count >= max_concurrent:
        return RiskVerdict(
            passed=False,
            rule="position_count",
            reason=f"Open positions ({open_position_count}) at max ({max_concurrent})",
        )

    # Rule 3: Buying power gate
    if order_value > buying_power:
        return RiskVerdict(
            passed=False,
            rule="buying_power",
            reason=f"Order value ${order_value:,.2f} exceeds buying power ${buying_power:,.2f}",
        )

    # Rule 4: Minimum order size (prevent dust trades)
    if order_value < 1.0:
        return RiskVerdict(
            passed=False,
            rule="min_order_size",
            reason=f"Order value ${order_value:.2f} below $1.00 minimum",
        )

    # Rule 5: Max portfolio exposure using actual position market values.
    if equity <= 0:
        return RiskVerdict(
            passed=False,
            rule="max_exposure",
            reason="Equity is zero or negative — cannot compute exposure",
        )
    exposure_after = (total_existing_exposure + order_value) / equity
    if exposure_after > max_portfolio_exposure:
        return RiskVerdict(
            passed=False,
            rule="max_exposure",
            reason=(
                f"Total exposure {exposure_after:.0%} would exceed "
                f"{max_portfolio_exposure:.0%} of equity"
            ),
        )

    return RiskVerdict(passed=True, rule="all", reason="")


def check_consecutive_loss_circuit(
    recent_realized_pnls: list[float],
    max_consecutive_losses: int,
) -> RiskVerdict:
    """Trip when the most recent closed trades are all losses.

    `recent_realized_pnls` should be ordered newest-first.
    """
    if max_consecutive_losses <= 0:
        return RiskVerdict(passed=True, rule="all", reason="")

    if len(recent_realized_pnls) < max_consecutive_losses:
        return RiskVerdict(passed=True, rule="all", reason="")

    recent_window = recent_realized_pnls[:max_consecutive_losses]
    if all(pnl < 0 for pnl in recent_window):
        loss_total = sum(recent_window)
        return RiskVerdict(
            passed=False,
            rule="consecutive_losses",
            reason=(
                f"Last {max_consecutive_losses} closed trades were losses "
                f"(${loss_total:,.2f} total)"
            ),
        )

    return RiskVerdict(passed=True, rule="all", reason="")
