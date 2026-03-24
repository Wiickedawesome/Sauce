"""
core/options_schemas.py — Pydantic v2 data models for options trading.

Simple calls and puts only. Targeting aggressive day/swing trades on high-volume
underlyings (SPY, QQQ, TSLA, NVDA, AMD).

Design:
- DTE 7-45 for swing, 0-7 for day trades
- Delta 0.25-0.40 for leverage with defined risk
- Single leg only (no spreads)
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base for all options models. Forbids extra fields."""

    model_config = ConfigDict(extra="forbid", frozen=False)


# ── Contract ───────────────────────────────────────────────────────────────────


class OptionContract(StrictModel):
    """Represents a single options contract."""

    underlying: str = Field(..., description="Underlying ticker (e.g., 'SPY', 'TSLA')")
    contract_symbol: str = Field(
        ...,
        description="OCC symbol (e.g., 'SPY250321C00550000' for SPY Mar 21 2025 $550 Call)",
    )
    option_type: Literal["call", "put"]
    strike: float = Field(..., gt=0.0)
    expiration: date
    dte: int = Field(..., ge=0, description="Days to expiration")
    delta: float | None = Field(default=None, ge=-1.0, le=1.0)
    implied_volatility: float | None = Field(default=None, ge=0.0)
    bid: float | None = Field(default=None, ge=0.0)
    ask: float | None = Field(default=None, ge=0.0)
    mid: float | None = Field(default=None, ge=0.0)
    open_interest: int | None = Field(default=None, ge=0)
    volume: int | None = Field(default=None, ge=0)


# ── Order ──────────────────────────────────────────────────────────────────────


class OptionsOrder(StrictModel):
    """
    Options order passed to broker.place_option_order().

    Single-leg only. Market orders are forbidden for risk control.
    """

    underlying: str
    contract_symbol: str
    option_type: Literal["call", "put"]
    side: Literal["buy", "sell"]
    qty: int = Field(..., gt=0, description="Number of contracts")
    limit_price: float = Field(..., gt=0.0)
    stop_loss_price: float | None = Field(
        default=None,
        ge=0.0,
        description="Mental stop — close position if premium drops below this",
    )
    take_profit_price: float | None = Field(
        default=None,
        ge=0.0,
        description="Take profit price for the premium",
    )
    time_in_force: Literal["day", "gtc"] = "day"
    stage: Literal["entry", "exit", "stop", "target"] = "entry"
    source: Literal["options_entry", "options_exit", "options_stop"] = "options_entry"
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Position ───────────────────────────────────────────────────────────────────


class OptionsPosition(StrictModel):
    """Tracks an open options position."""

    position_id: str = Field(..., description="Unique ID for this position")
    underlying: str
    contract_symbol: str
    option_type: Literal["call", "put"]
    qty: int = Field(..., gt=0)
    entry_price: float = Field(..., gt=0.0, description="Premium paid per contract")
    entry_time: datetime
    expiration: date
    high_water_price: float = Field(..., ge=0.0, description="Highest premium since entry")
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    dte_at_entry: int = Field(..., ge=0)
    strategy_name: str = "options_momentum"
    broker_order_id: str | None = None


# ── Signal ─────────────────────────────────────────────────────────────────────


class OptionsSignalResult(StrictModel):
    """Result of scoring an options entry opportunity."""

    underlying: str
    option_type: Literal["call", "put"]
    score: int = Field(..., ge=0, le=100)
    threshold: int
    fired: bool
    # Selected contract details
    contract_symbol: str | None = None
    strike: float | None = None
    expiration: date | None = None
    dte: int | None = None
    delta: float | None = None
    bid: float | None = None
    ask: float | None = None
    # Underlying indicators
    rsi_14: float | None = None
    macd_hist: float | None = None
    atr_14: float | None = None
    regime: str
    strategy_name: str = "options_momentum"


# ── Exit ───────────────────────────────────────────────────────────────────────


class OptionsExitSignal(StrictModel):
    """
    Exit signal for an options position.

    Conditions checked:
    1. Hard stop — premium fell below stop_loss_price
    2. Profit target — premium hit take_profit_price
    3. DTE threshold — approaching expiration (close at 2 DTE)
    4. Underlying reversal — underlying's RSI flipped against position
    5. Time stop — holding too long without gains
    """

    position_id: str
    contract_symbol: str
    reason: Literal[
        "hard_stop",
        "profit_target",
        "dte_threshold",
        "underlying_reversal",
        "time_stop",
        "manual",
    ]
    current_price: float
    entry_price: float
    pnl_pct: float
    dte_remaining: int
    urgency: Literal["immediate", "eod", "next_open"] = "immediate"
