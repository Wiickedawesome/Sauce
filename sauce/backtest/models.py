"""
backtest/models.py — Data models for the backtesting engine.

All models are plain dataclasses (not Pydantic) since they are internal to
the backtest subsystem and never cross agent boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"


class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    PROFIT_TARGET = "profit_target"
    END_OF_DATA = "end_of_data"


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    initial_capital: float = 100_000.0
    max_position_pct: float = 0.08       # max % of capital per position
    stop_loss_atr_multiple: float = 2.0
    profit_target_atr_multiple: float = 3.0
    min_confidence: float = 0.40         # below this → hold
    commission_per_trade: float = 0.0    # flat $ per side
    slippage_pct: float = 0.001          # 0.1% slippage per fill
    lookback_bars: int = 60              # bars fed to indicator engine
    bar_step: int = 1                    # bars to advance per step


@dataclass(frozen=True)
class BacktestTrade:
    """Record of a single completed trade."""

    symbol: str
    direction: TradeDirection
    setup_type: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: float
    pnl: float                # net P&L after costs
    pnl_pct: float            # as fraction of entry notional
    exit_reason: ExitReason
    confidence: float         # setup score / 100
    bars_held: int


@dataclass
class BacktestResult:
    """Aggregate output of a backtest run."""

    symbol: str
    config: BacktestConfig
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)  # NAV at each bar
    timestamps: list[datetime] = field(default_factory=list)  # timestamp per bar

    # ── computed fields (populated by engine) ────────────────────────
    total_return: float = 0.0
    sharpe_ratio: float | None = None
    max_drawdown: float = 0.0       # negative fraction
    win_rate: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_pnl: float = 0.0
    avg_bars_held: float = 0.0
    profit_factor: float | None = None  # gross profit / gross loss
