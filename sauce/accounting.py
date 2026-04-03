"""Trade accounting helpers for gross and net P&L.

The live trading system records net realized P&L as the primary value while
retaining gross P&L, fees, and slippage estimates for attribution.
"""

from __future__ import annotations

from dataclasses import dataclass

from sauce.core.config import get_settings


@dataclass(frozen=True, slots=True)
class ExecutionCosts:
    """Estimated transaction costs for one side of a trade."""

    notional: float
    fees_paid: float
    slippage_paid: float

    @property
    def total_cost(self) -> float:
        return self.fees_paid + self.slippage_paid


@dataclass(frozen=True, slots=True)
class TradeAccounting:
    """Gross and net accounting for a completed round-trip trade."""

    asset_class: str
    qty: float
    multiplier: float
    entry_notional: float
    exit_notional: float
    gross_pnl: float
    fees_paid: float
    slippage_paid: float
    realized_pnl: float


# Notional-tiered market-impact multipliers (proxy for ADV-fraction model).
# Larger orders have higher price impact relative to available liquidity.
# Tiers align with capital scale: SEED (<$1k), SCALING ($1k-$10k),
# GROWTH ($10k-$50k), INSTITUTIONAL (>$50k).
# Fees are not scaled — exchange/broker fees are flat per-trade.
_IMPACT_TIERS: tuple[tuple[float, float], ...] = (
    (1_000.0,    1.0),
    (10_000.0,   1.5),
    (50_000.0,   2.0),
    (float("inf"), 3.0),
)


def _impact_multiplier(notional: float) -> float:
    """Return a slippage scaling factor based on order notional size."""
    for threshold, multiplier in _IMPACT_TIERS:
        if notional < threshold:
            return multiplier
    return _IMPACT_TIERS[-1][1]  # unreachable but satisfies type checker


def _normalize_asset_class(asset_class: str) -> str:
    normalized = asset_class.strip().lower()
    if normalized in {"options", "option"}:
        return "option"
    if normalized not in {"crypto", "equity", "option"}:
        return "equity"
    return normalized


def _fee_bps(asset_class: str) -> float:
    settings = get_settings()
    normalized = _normalize_asset_class(asset_class)
    if normalized == "crypto":
        return settings.crypto_fee_bps
    if normalized == "option":
        return settings.option_fee_bps
    return settings.equity_fee_bps


def _slippage_bps(asset_class: str) -> float:
    settings = get_settings()
    normalized = _normalize_asset_class(asset_class)
    if normalized == "crypto":
        return settings.crypto_slippage_bps
    if normalized == "option":
        return settings.option_slippage_bps
    return settings.equity_slippage_bps


def estimate_side_costs(asset_class: str, notional: float) -> ExecutionCosts:
    """Estimate fees and slippage for one side of a trade.

    Slippage is scaled by a notional-tiered impact multiplier to model
    market impact at larger order sizes. Fees are flat (exchange-rate based).
    """
    normalized = _normalize_asset_class(asset_class)
    safe_notional = max(notional, 0.0)
    impact = _impact_multiplier(safe_notional)
    fees_paid = safe_notional * (_fee_bps(normalized) / 10_000)
    slippage_paid = safe_notional * (_slippage_bps(normalized) / 10_000) * impact
    return ExecutionCosts(
        notional=safe_notional,
        fees_paid=fees_paid,
        slippage_paid=slippage_paid,
    )


def estimate_trade_accounting(
    asset_class: str,
    qty: float,
    entry_price: float,
    exit_price: float,
    *,
    contract_multiplier: float | None = None,
) -> TradeAccounting:
    """Estimate round-trip gross and net P&L for a completed trade."""
    normalized = _normalize_asset_class(asset_class)
    multiplier = contract_multiplier if contract_multiplier is not None else (100.0 if normalized == "option" else 1.0)

    safe_qty = max(qty, 0.0)
    entry_notional = max(entry_price, 0.0) * safe_qty * multiplier
    exit_notional = max(exit_price, 0.0) * safe_qty * multiplier
    gross_pnl = (exit_price - entry_price) * safe_qty * multiplier

    entry_costs = estimate_side_costs(normalized, entry_notional)
    exit_costs = estimate_side_costs(normalized, exit_notional)
    fees_paid = entry_costs.fees_paid + exit_costs.fees_paid
    slippage_paid = entry_costs.slippage_paid + exit_costs.slippage_paid
    realized_pnl = gross_pnl - fees_paid - slippage_paid

    return TradeAccounting(
        asset_class=normalized,
        qty=safe_qty,
        multiplier=multiplier,
        entry_notional=entry_notional,
        exit_notional=exit_notional,
        gross_pnl=gross_pnl,
        fees_paid=fees_paid,
        slippage_paid=slippage_paid,
        realized_pnl=realized_pnl,
    )
