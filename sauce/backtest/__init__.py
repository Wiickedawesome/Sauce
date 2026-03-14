"""
sauce.backtest — Bar-replay backtesting engine.

Usage:
    from sauce.backtest import run_backtest, BacktestConfig, BacktestResult

    result = run_backtest("BTC/USD", ohlcv_df, regime="RANGING")
    print(f"Sharpe: {result.sharpe_ratio}, Trades: {result.total_trades}")
"""

from sauce.backtest.engine import run_backtest
from sauce.backtest.models import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    ExitReason,
    TradeDirection,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestTrade",
    "ExitReason",
    "TradeDirection",
    "run_backtest",
]
