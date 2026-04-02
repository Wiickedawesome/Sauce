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
    """Estimate fees and slippage for one side of a trade."""
    normalized = _normalize_asset_class(asset_class)
    safe_notional = max(notional, 0.0)
    fees_paid = safe_notional * (_fee_bps(normalized) / 10_000)
    slippage_paid = safe_notional * (_slippage_bps(normalized) / 10_000)
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
