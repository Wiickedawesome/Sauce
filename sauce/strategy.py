"""
strategy.py — Protocol interface and core dataclasses for Sauce.

All strategies implement the Strategy protocol. The loop iterates registered
strategies and calls eligible() → score() → build_order() → build_exit_plan().

Dataclasses:
  - SignalResult: output of score(), 0–100 integer plus metadata
  - ExitPlan: per-position exit parameters (stop, trail, target, RSI, time)
  - TierParams: risk parameters keyed to account equity
  - Position: open position with trailing stop state
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from sauce.core.schemas import Indicators, Order


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SignalResult:
    """Output of Strategy.score(). Every scoring cycle produces one per symbol."""

    symbol: str
    side: str  # "buy" | "sell" | "hold"
    score: int  # 0–100
    threshold: int  # effective threshold for this cycle
    fired: bool  # score >= threshold
    rsi_14: float | None
    macd_hist: float | None
    bb_pct: float | None  # how close price is to lower BB (0 = at lower, 1 = at upper)
    volume_ratio: float | None
    regime: str
    strategy_name: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class Position:
    """An open position with trailing-stop state."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    asset_class: str = "crypto"  # crypto | equity | option
    qty: float = 0.0
    entry_price: float = 0.0
    high_water_price: float = 0.0
    trailing_stop_price: float | None = None
    trailing_active: bool = False
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    broker_order_id: str | None = None
    strategy_name: str = ""
    stop_loss_price: float = 0.0
    profit_target_price: float = 0.0


@dataclass(frozen=True, slots=True)
class ExitPlan:
    """Exit parameters for a position. Built by strategy.build_exit_plan()."""

    stop_loss_pct: float
    trail_activation_pct: float
    trail_pct: float
    profit_target_pct: float
    rsi_exhaustion_threshold: float  # e.g. 72
    max_hold_hours: float
    time_stop_min_gain: float
    # Options extras (None for non-options)
    dte_exit_threshold: int | None = None
    regime_stop: bool = False  # close on bearish regime?


@dataclass(frozen=True, slots=True)
class TierParams:
    """Risk parameters derived from account equity. Set by get_tier_params()."""

    tier: str  # "seed" | "building" | "scaling" | "operating"
    max_position_pct: float
    max_concurrent: int
    daily_loss_limit: float
    stop_loss_pct: float
    trail_activation_pct: float
    trail_pct: float
    profit_target_pct: float
    rsi_exhaustion_threshold: float
    max_hold_hours: float
    time_stop_min_gain: float
    # Sector limits (building+)
    max_crypto_pct: float = 1.0  # 100% at seed
    max_equity_sector_pct: float = 1.0


# ── Tier Table ────────────────────────────────────────────────────────────────

SEED_PARAMS = TierParams(
    tier="seed",
    max_position_pct=0.30,
    max_concurrent=2,
    daily_loss_limit=0.08,
    stop_loss_pct=0.03,
    trail_activation_pct=0.03,
    trail_pct=0.02,
    profit_target_pct=0.06,
    rsi_exhaustion_threshold=72,
    max_hold_hours=48,
    time_stop_min_gain=0.01,
)

BUILDING_PARAMS = TierParams(
    tier="building",
    max_position_pct=0.15,
    max_concurrent=4,
    daily_loss_limit=0.05,
    stop_loss_pct=0.03,
    trail_activation_pct=0.03,
    trail_pct=0.02,
    profit_target_pct=0.06,
    rsi_exhaustion_threshold=72,
    max_hold_hours=48,
    time_stop_min_gain=0.01,
    max_crypto_pct=0.40,
    max_equity_sector_pct=0.35,
)

SCALING_PARAMS = TierParams(
    tier="scaling",
    max_position_pct=0.08,
    max_concurrent=8,
    daily_loss_limit=0.03,
    stop_loss_pct=0.03,
    trail_activation_pct=0.03,
    trail_pct=0.02,
    profit_target_pct=0.06,
    rsi_exhaustion_threshold=72,
    max_hold_hours=48,
    time_stop_min_gain=0.01,
    max_crypto_pct=0.30,
    max_equity_sector_pct=0.30,
)

OPERATING_PARAMS = TierParams(
    tier="operating",
    max_position_pct=0.05,
    max_concurrent=12,
    daily_loss_limit=0.02,
    stop_loss_pct=0.03,
    trail_activation_pct=0.03,
    trail_pct=0.02,
    profit_target_pct=0.06,
    rsi_exhaustion_threshold=72,
    max_hold_hours=48,
    time_stop_min_gain=0.01,
    max_crypto_pct=0.30,
    max_equity_sector_pct=0.25,
)


def get_tier_params(equity: float) -> TierParams:
    """Return risk parameters for the given account equity."""
    if equity < 10_000:
        return SEED_PARAMS
    elif equity < 50_000:
        return BUILDING_PARAMS
    elif equity < 100_000:
        return SCALING_PARAMS
    else:
        return OPERATING_PARAMS


# ── Strategy Protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class Strategy(Protocol):
    """
    Every trading strategy must implement this interface.

    The loop iterates registered strategies and calls:
      1. eligible(symbol, regime) — can this strategy fire here?
      2. score(indicators, symbol, regime, price) — rate the setup 0–100
      3. build_order(signal, account) — construct the entry order
      4. build_exit_plan(position) — define exit rules for this position
    """

    name: str
    instruments: list[str]

    def eligible(self, instrument: str, regime: str) -> bool:
        """Can this strategy fire on this instrument given today's regime?"""
        ...

    def score(self, indicators: Indicators, instrument: str, regime: str, current_price: float) -> SignalResult:
        """Score the signal. Returns score 0–100 and metadata."""
        ...

    def build_order(self, signal: SignalResult, account: dict, tier: TierParams) -> Order:
        """Construct the entry order. Handles sizing per tier."""
        ...

    def build_exit_plan(self, position: Position, tier: TierParams) -> ExitPlan:
        """Define trailing stop, profit target, time stop for this position."""
        ...
