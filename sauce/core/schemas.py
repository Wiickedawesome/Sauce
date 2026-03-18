"""
core/schemas.py — Pydantic v2 data models for the Sauce trading system.

Rules:
- All models use extra="forbid" — unknown fields raise ValidationError immediately.
- ValidationError anywhere in the pipeline = abort signal, log AuditEvent, continue.
"""

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


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


# ── Technical Indicators ──────────────────────────────────────────────────────

class Indicators(StrictModel):
    """Technical indicators computed from OHLCV history via pandas-ta."""

    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None
    atr_14: float | None = None
    volume_ratio: float | None = None  # today's volume / 20-day avg volume
    volume_1d_avg: float | None = None  # estimated average daily volume in shares
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    stoch_k: float | None = None
    stoch_d: float | None = None
    vwap: float | None = None


# ── Order ─────────────────────────────────────────────────────────────────────

class Order(StrictModel):
    """
    Order object passed to broker.place_order().

    Must originate from an approved risk check — never created directly.
    """

    symbol: str
    side: Literal["buy", "sell"]
    qty: float = Field(..., gt=0.0)
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "limit"
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"
    limit_price: float | None = Field(default=None, ge=0.0)
    stop_price: float | None = Field(default=None, ge=0.0)
    stop_loss_price: float | None = Field(
        default=None, ge=0.0,
        description="Companion stop-loss price (ATR-based). Used to submit a "
                    "protective stop order after entry fill.",
    )
    take_profit_price: float | None = Field(
        default=None, ge=0.0,
        description="Companion take-profit price (ATR-based). Informational — "
                    "logged for audit but not automatically submitted.",
    )
    as_of: datetime
    prompt_version: str
    source: Literal["execution", "exit_research", "stop_loss", "options_entry", "options_exit", "options_stop"] | None = Field(
        default=None,
        description="Provenance tag: which agent/stage created this order.",
    )


# ── Audit ─────────────────────────────────────────────────────────────────────

class AuditEvent(StrictModel):
    """
    Immutable audit log entry written by adapters.

    All writes to the DB use this model. Nothing is ever updated or deleted.
    loop_id ties all events from one trading loop run together.
    """

    loop_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal[
        "broker_call",
        "broker_response",
        "error",
        "llm_call",
        "llm_response",
        "options_order_submitted",
    ]
    symbol: str | None = None
    payload: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_version: str | None = None
