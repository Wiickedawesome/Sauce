"""
core/options_schemas.py — Pydantic v2 models for the options trading module.

All models inherit from StrictModel (extra="forbid").
All include as_of: datetime and prompt_version: str where they cross agent boundaries.

These schemas support the "Double Up & Take Gains" compounding strategy:
  Entry → Nx profit → close fraction → let rest ride → repeat.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import Field, field_validator

from sauce.core.schemas import StrictModel


# ── Compound Stage Ladder ─────────────────────────────────────────────────────

class CompoundStage(StrictModel):
    """
    One rung of the take-profit ladder.

    stage_num=1, trigger_multiplier=2.0 → sell sell_fraction at 2x entry price.
    """

    stage_num: int = Field(..., ge=1)
    trigger_multiplier: float = Field(
        ..., gt=1.0,
        description="Price multiplier that triggers this stage (e.g. 2.0 = 100% gain).",
    )
    sell_fraction: float = Field(
        ..., gt=0.0, le=1.0,
        description="Fraction of remaining qty to sell at this stage.",
    )
    trailing_stop_pct: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Trailing stop as a fraction of current price after this stage triggers.",
    )
    completed: bool = Field(default=False)


# ── Contract / Position ───────────────────────────────────────────────────────

class OptionsContract(StrictModel):
    """Represents a single options contract."""

    contract_symbol: str = Field(..., description="OCC symbol, e.g. AAPL250418C00200000")
    underlying: str
    expiration: str = Field(..., description="YYYY-MM-DD expiration date")
    strike: float = Field(..., gt=0.0)
    option_type: Literal["call", "put"]
    # Greeks snapshot at selection time
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    iv: float | None = Field(default=None, ge=0.0, description="Implied volatility")
    bid: float | None = Field(default=None, ge=0.0)
    ask: float | None = Field(default=None, ge=0.0)
    mid: float | None = Field(default=None, ge=0.0)
    open_interest: int | None = Field(default=None, ge=0)
    volume: int | None = Field(default=None, ge=0)


class OptionsPosition(StrictModel):
    """Tracks an open options position through the compounding ladder."""

    position_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contract: OptionsContract
    entry_price: float = Field(..., gt=0.0, description="Average fill price per contract")
    qty: int = Field(..., ge=1, description="Total contracts entered")
    remaining_qty: int = Field(..., ge=0, description="Contracts still held")
    direction: Literal["long_call", "long_put"]
    strategy_type: Literal["directional", "debit_spread", "credit_spread"] = "directional"
    compound_stages: list[CompoundStage] = Field(default_factory=list)
    stages_completed: int = Field(default=0, ge=0)
    realized_pnl: float = Field(default=0.0)
    trailing_stop_price: float | None = None
    entry_time: datetime | None = None
    as_of: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Signals ───────────────────────────────────────────────────────────────────

class OptionsBias(StrictModel):
    """Directional bias derived from equity confluence + IV/momentum filters."""

    symbol: str
    direction: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    iv_rank: float | None = Field(default=None, ge=0.0, le=1.0)
    regime_ok: bool = True
    momentum_aligned: bool = True
    reasoning: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: object) -> float:
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        import math
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return max(0.0, min(1.0, f))


class OptionsSignal(StrictModel):
    """Output of the options research agent — a recommended trade."""

    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    contract: OptionsContract
    direction: Literal["long_call", "long_put"]
    strategy_type: Literal["directional", "debit_spread", "credit_spread"] = "directional"
    target_multiplier: float = Field(
        default=2.0, gt=1.0,
        description="Profit multiplier target per compound stage.",
    )
    compound_stages: int = Field(default=3, ge=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    bear_case: str = ""
    bias: OptionsBias
    as_of: datetime
    prompt_version: str


# ── Orders ────────────────────────────────────────────────────────────────────

class OptionsOrder(StrictModel):
    """Order to be submitted to the broker for an options trade."""

    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contract_symbol: str
    underlying: str
    qty: int = Field(..., ge=1)
    side: Literal["buy", "sell"]
    order_type: Literal["limit"] = "limit"  # NEVER market orders on options
    limit_price: float = Field(..., gt=0.0)
    time_in_force: Literal["day", "gtc"] = "day"
    stage: int = Field(
        default=0, ge=0,
        description="0 = initial entry, 1+ = compounding exit stage.",
    )
    source: Literal["options_entry", "options_exit", "options_stop"] = "options_entry"
    as_of: datetime
    prompt_version: str


# ── Exit Decision ─────────────────────────────────────────────────────────────

class ExitDecision(StrictModel):
    """Output of the compounding exit engine for a single position."""

    action: Literal["HOLD", "PARTIAL_CLOSE", "FULL_CLOSE"]
    qty: int = Field(default=0, ge=0)
    reason: str = ""
    set_trailing_stop: bool = False
    trailing_stop_pct: float = Field(default=0.0, ge=0.0, le=1.0)
    stage: int = Field(default=0, ge=0)


# ── Options Quote (from broker) ───────────────────────────────────────────────

class OptionsQuote(StrictModel):
    """Live quote snapshot for an options contract."""

    contract_symbol: str
    bid: float = Field(..., ge=0.0)
    ask: float = Field(..., ge=0.0)
    mid: float = Field(..., ge=0.0)
    last: float | None = Field(default=None, ge=0.0)
    volume: int = Field(default=0, ge=0)
    open_interest: int = Field(default=0, ge=0)
    iv: float | None = Field(default=None, ge=0.0)
    delta: float | None = None
    theta: float | None = None
    as_of: datetime
