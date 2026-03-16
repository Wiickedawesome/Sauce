"""
backtest/options_models.py — Data models for options backtesting.

Follows the equity backtest models pattern: frozen dataclasses, enums for
exit reasons, and a result container with aggregate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class OptionsExitReason(str, Enum):
    STAGE_1 = "stage_1"
    STAGE_2 = "stage_2"
    STAGE_3 = "stage_3"
    HARD_STOP = "hard_stop"
    TRAILING_STOP = "trailing_stop"
    EXPIRY = "expiry"
    MAX_DTE = "max_dte"


@dataclass(frozen=True)
class OptionsBacktestTrade:
    """Record of a single simulated options trade."""

    symbol: str
    direction: str  # long_call / long_put
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    pnl_pct: float
    exit_reason: OptionsExitReason
    stages_completed: int
    bars_held: int


@dataclass
class OptionsBacktestConfig:
    """Configuration for the options backtest engine."""

    initial_capital: float = 10_000.0
    position_size_pct: float = 0.05
    max_total_exposure_pct: float = 0.20
    profit_multiplier: float = 2.0
    compound_stages: int = 3
    sell_fraction: float = 0.5
    max_loss_pct: float = 0.50
    trailing_stop_pct: float = 0.15
    min_delta: float = 0.30
    max_delta: float = 0.60
    min_dte: int = 7
    max_dte: int = 45
    commission_per_contract: float = 0.65
    slippage_pct: float = 0.01
    theta_decay_annual: float = 0.05  # daily theta as fraction of option value


@dataclass
class OptionsBacktestResult:
    """Aggregate results of an options backtest run."""

    symbol: str
    config: OptionsBacktestConfig
    trades: list[OptionsBacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    # Populated by _compute_metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_stages_completed: float = 0.0
    avg_bars_held: float = 0.0
