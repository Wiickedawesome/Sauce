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
    PROFIT_TARGET = "profit_target"
    STRETCH_TARGET = "stretch_target"
    TRAILING_STOP = "trailing_stop"
    HARD_STOP = "hard_stop"
    TIME_STOP = "time_stop"
    DTE_STOP = "dte_stop"
    REGIME_STOP = "regime_stop"
    EXPIRY = "expiry"


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
    bars_held: int


@dataclass
class OptionsBacktestConfig:
    """Configuration for the options backtest engine."""

    initial_capital: float = 5_000.0
    position_size_pct: float = 0.10
    max_total_exposure_pct: float = 0.20
    max_contract_cost: float = 500.0
    max_contracts: int = 5
    profit_target_pct: float = 0.35
    stretch_target_pct: float = 0.60
    trail_activation_pct: float = 0.20
    trail_pct: float = 0.12
    max_loss_pct: float = 0.25
    time_stop_days: int = 5
    time_stop_min_gain_pct: float = 0.10
    dte_exit_days: int = 5
    min_delta: float = 0.30
    max_delta: float = 0.60
    min_dte: int = 14
    max_dte: int = 35
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
    avg_bars_held: float = 0.0
