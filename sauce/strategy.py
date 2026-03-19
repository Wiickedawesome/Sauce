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
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

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
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


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
    entry_time: datetime = field(default_factory=lambda: datetime.now(UTC))
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
# AGGRESSIVE CONFIGURATION: Optimized for day/swing trading with 15-25% risk tolerance

SEED_PARAMS = TierParams(
    tier="seed",
    max_position_pct=0.40,  # 40% per position for conviction plays
    max_concurrent=4,  # 4 concurrent positions
    daily_loss_limit=0.20,  # 20% daily drawdown ceiling (aggressive)
    stop_loss_pct=0.05,  # 5% stop (wider for swing holds)
    trail_activation_pct=0.06,  # Trail kicks in at 6% gain
    trail_pct=0.03,  # 3% trailing stop
    profit_target_pct=0.12,  # 12% target (2.4:1 R/R)
    rsi_exhaustion_threshold=75,  # Higher RSI exit (let winners run)
    max_hold_hours=336,  # 14 days for swing positions
    time_stop_min_gain=0.02,  # Exit stale positions at 2% gain minimum
)

BUILDING_PARAMS = TierParams(
    tier="building",
    max_position_pct=0.25,  # 25% per position ($10K-$50K)
    max_concurrent=6,  # 6 concurrent positions
    daily_loss_limit=0.15,  # 15% daily drawdown
    stop_loss_pct=0.04,  # 4% stop
    trail_activation_pct=0.05,  # Trail at 5%
    trail_pct=0.025,  # 2.5% trail
    profit_target_pct=0.10,  # 10% target
    rsi_exhaustion_threshold=75,
    max_hold_hours=336,
    time_stop_min_gain=0.02,
    max_crypto_pct=0.50,  # 50% crypto allocation
    max_equity_sector_pct=0.40,  # 40% per equity sector
)

SCALING_PARAMS = TierParams(
    tier="scaling",
    max_position_pct=0.15,  # 15% per position ($50K-$100K)
    max_concurrent=8,
    daily_loss_limit=0.10,  # 10% daily drawdown
    stop_loss_pct=0.035,  # 3.5% stop
    trail_activation_pct=0.04,
    trail_pct=0.02,
    profit_target_pct=0.08,
    rsi_exhaustion_threshold=75,
    max_hold_hours=336,
    time_stop_min_gain=0.015,
    max_crypto_pct=0.40,
    max_equity_sector_pct=0.35,
)

OPERATING_PARAMS = TierParams(
    tier="operating",
    max_position_pct=0.10,  # 10% per position ($100K+)
    max_concurrent=12,
    daily_loss_limit=0.08,  # 8% daily drawdown (more conservative at scale)
    stop_loss_pct=0.03,
    trail_activation_pct=0.035,
    trail_pct=0.02,
    profit_target_pct=0.06,
    rsi_exhaustion_threshold=75,
    max_hold_hours=336,
    time_stop_min_gain=0.01,
    max_crypto_pct=0.35,
    max_equity_sector_pct=0.30,
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

    def score(
        self, indicators: Indicators, instrument: str, regime: str, current_price: float
    ) -> SignalResult:
        """Score the signal. Returns score 0–100 and metadata."""
        ...

    def build_order(self, signal: SignalResult, account: dict[str, Any], tier: TierParams) -> Order:
        """Construct the entry order. Handles sizing per tier."""
        ...

    def build_exit_plan(self, position: Position, tier: TierParams) -> ExitPlan:
        """Define trailing stop, profit target, time stop for this position."""
        ...
