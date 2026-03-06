"""
core/schemas.py — All Pydantic v2 data models for the Sauce trading system.

Rules:
- All models use extra="forbid" — unknown fields raise ValidationError immediately.
- All models include as_of (datetime, UTC) and prompt_version (str).
- No raw dicts cross agent boundaries — always use these models.
- ValidationError anywhere in the pipeline = abort signal, log AuditEvent, continue.
"""

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Shared base ───────────────────────────────────────────────────────────────

class StrictModel(BaseModel):
    """Base for all Sauce models. Forbids extra fields globally."""

    model_config = ConfigDict(extra="forbid", frozen=False)


# ── Market Data ───────────────────────────────────────────────────────────────

class PriceReference(StrictModel):
    """Latest bid/ask/mid for a symbol at a given moment."""

    symbol: str
    bid: float = Field(..., ge=0.0)
    ask: float = Field(..., ge=0.0)
    mid: float = Field(..., ge=0.0)
    as_of: datetime = Field(..., description="Timestamp from the data API (UTC). Never datetime.now().")


# ── Research Agent ────────────────────────────────────────────────────────────

class Indicators(StrictModel):
    """Technical indicators computed from OHLCV history via pandas-ta."""

    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None
    atr_14: float | None = None
    volume_ratio: float | None = None  # today's volume / 20-day avg volume


class Evidence(StrictModel):
    """Grounded evidence passed to Research agent and embedded in Signal."""

    symbol: str
    price_reference: PriceReference
    indicators: Indicators
    volatility: float | None = None
    volume: float | None = None
    as_of: datetime


class Signal(StrictModel):
    """
    Output of the Research agent.

    confidence < 0.5 is treated as hold by the system regardless of side.
    side="hold" short-circuits the rest of the pipeline immediately.
    """

    symbol: str
    side: Literal["buy", "sell", "hold"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence
    reasoning: str = Field(default="", description="Claude's reasoning (for audit purposes)")
    as_of: datetime
    prompt_version: str

    @field_validator("confidence")
    @classmethod
    def clamp_hold_on_low_confidence(cls, v: float) -> float:
        """Confidence below 0.5 is valid but the system will treat it as hold."""
        return v


# ── Risk Agent ────────────────────────────────────────────────────────────────

class RiskChecks(StrictModel):
    """Individual boolean results for each risk rule."""

    max_position_pct_ok: bool
    max_exposure_ok: bool
    daily_loss_ok: bool
    volatility_ok: bool
    confidence_ok: bool


class RiskCheckResult(StrictModel):
    """
    Output of the Risk agent per signal.

    veto=True means the signal must not proceed to Execution.
    qty is only set when veto=False.
    """

    symbol: str
    side: Literal["buy", "sell", "hold"]
    veto: bool
    reason: str | None = None
    qty: float | None = Field(default=None, ge=0.0)
    checks: RiskChecks
    as_of: datetime
    prompt_version: str


# ── Execution Agent ───────────────────────────────────────────────────────────

class Order(StrictModel):
    """
    Output of the Execution agent.

    This is the final order object passed to broker.place_order().
    It must originate from an approved RiskCheckResult — never created directly.
    """

    symbol: str
    side: Literal["buy", "sell"]
    qty: float = Field(..., gt=0.0)
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "limit"
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"
    limit_price: float | None = Field(default=None, ge=0.0)
    stop_price: float | None = Field(default=None, ge=0.0)
    as_of: datetime
    prompt_version: str


# ── Supervisor Agent ──────────────────────────────────────────────────────────

class SupervisorDecision(StrictModel):
    """
    Final arbitration output from the Supervisor agent.

    action="execute" means all final_orders are safe to send to the broker.
    action="abort" means nothing is sent — vetoes contains the reasons.
    final_orders is always empty when action="abort".
    """

    action: Literal["execute", "abort"]
    final_orders: list[Order] = Field(default_factory=list)
    vetoes: list[str] = Field(default_factory=list)
    reason: str
    as_of: datetime
    prompt_version: str

    @field_validator("final_orders")
    @classmethod
    def orders_empty_on_abort(cls, v: list[Order], info: object) -> list[Order]:
        # Validated post-init in model_post_init for cross-field checks
        return v

    def model_post_init(self, __context: object) -> None:
        if self.action == "abort" and self.final_orders:
            raise ValueError("final_orders must be empty when action='abort'")
        if self.action == "execute" and not self.final_orders:
            raise ValueError("final_orders cannot be empty when action='execute'")


# ── Audit / Ops ───────────────────────────────────────────────────────────────

class AuditEvent(StrictModel):
    """
    Immutable audit log entry written by every agent and adapter.

    All writes to the DB use this model. Nothing is ever updated or deleted.
    loop_id ties all events from one cron run together.
    """

    loop_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal[
        "loop_start",
        "loop_end",
        "signal",
        "risk_check",
        "order",
        "order_prepared",
        "fill",
        "veto",
        "error",
        "stub_called",
        "llm_call",
        "llm_response",
        "broker_call",
        "broker_response",
        "safety_check",
        "supervisor_decision",
        "portfolio_review",
        "ops_summary",
    ]
    symbol: str | None = None
    payload: dict = Field(default_factory=dict)  # serialised model or error detail
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_version: str | None = None


# ── Portfolio Agent ───────────────────────────────────────────────────────────

class PositionNote(StrictModel):
    """Suggestion from the Portfolio agent for a single position."""

    symbol: str
    current_qty: float
    current_value_usd: float
    suggested_stop_price: float | None = None
    suggested_target_price: float | None = None
    over_concentrated: bool = False
    note: str = ""


class PortfolioReview(StrictModel):
    """Full output of the Portfolio agent for a single loop run."""

    positions: list[PositionNote] = Field(default_factory=list)
    total_exposure_pct: float = Field(..., ge=0.0, le=2.0)
    rebalance_needed: bool = False
    uncovered_symbols: list[str] = Field(default_factory=list)
    as_of: datetime
    prompt_version: str


# ── Daily Stats ───────────────────────────────────────────────────────────────

class DailyStats(StrictModel):
    """Aggregated stats for a single trading day. Written by Ops agent."""

    date: str  # YYYY-MM-DD
    loop_runs: int = 0
    signals_generated: int = 0
    signals_vetoed: int = 0
    orders_placed: int = 0
    realized_pnl_usd: float = 0.0
    starting_nav_usd: float = 0.0
    ending_nav_usd: float = 0.0
    trading_paused: bool = False
