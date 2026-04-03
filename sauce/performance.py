"""Performance reporting helpers for Sauce trades and research backtests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import sqrt

import pandas as pd


@dataclass(frozen=True, slots=True)
class TradePerformanceRecord:
    symbol: str
    asset_class: str
    strategy_name: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    qty: float
    gross_pnl: float
    realized_pnl: float
    fees_paid: float
    slippage_paid: float
    hold_hours: float
    exit_trigger: str


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    trade_count: int
    gross_pnl: float
    net_pnl: float
    fees_paid: float
    slippage_paid: float
    win_rate: float
    profit_factor: float | None
    average_hold_hours: float
    max_drawdown_pct: float
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None
    total_return_pct: float | None
    ending_equity: float | None
    max_consecutive_losses: int


def _profit_factor(net_pnls: pd.Series) -> float | None:
    gross_wins = float(net_pnls[net_pnls > 0].sum())
    gross_losses = abs(float(net_pnls[net_pnls < 0].sum()))
    if gross_losses == 0:
        return None if gross_wins == 0 else float("inf")
    return gross_wins / gross_losses


def _max_consecutive_losses(net_pnls: pd.Series) -> int:
    streak = 0
    max_streak = 0
    for pnl in net_pnls:
        if pnl < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


def _risk_adjusted_metrics(daily_returns: pd.Series, risk_free_rate: float) -> tuple[float | None, float | None]:
    if daily_returns.empty:
        return None, None

    daily_rf = risk_free_rate / 252
    excess_returns = daily_returns - daily_rf
    std = float(excess_returns.std(ddof=0))
    sharpe = None if std <= 0 else float(excess_returns.mean() / std * sqrt(252))

    downside = daily_returns[daily_returns < daily_rf] - daily_rf
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    sortino = None if downside_std <= 0 else float(excess_returns.mean() / downside_std * sqrt(252))
    return sharpe, sortino


def summarize_performance(
    trades: list[TradePerformanceRecord],
    *,
    starting_equity: float | None = None,
    risk_free_rate: float = 0.05,
) -> PerformanceSummary:
    """Compute net performance metrics for a list of completed trades."""
    if not trades:
        return PerformanceSummary(
            trade_count=0,
            gross_pnl=0.0,
            net_pnl=0.0,
            fees_paid=0.0,
            slippage_paid=0.0,
            win_rate=0.0,
            profit_factor=None,
            average_hold_hours=0.0,
            max_drawdown_pct=0.0,
            sharpe_ratio=None,
            sortino_ratio=None,
            calmar_ratio=None,
            total_return_pct=None,
            ending_equity=starting_equity,
            max_consecutive_losses=0,
        )

    frame = pd.DataFrame(
        {
            "exit_time": [pd.Timestamp(trade.exit_time) for trade in trades],
            "gross_pnl": [trade.gross_pnl for trade in trades],
            "net_pnl": [trade.realized_pnl for trade in trades],
            "fees_paid": [trade.fees_paid for trade in trades],
            "slippage_paid": [trade.slippage_paid for trade in trades],
            "hold_hours": [trade.hold_hours for trade in trades],
        }
    ).sort_values("exit_time")

    gross_pnl = float(frame["gross_pnl"].sum())
    net_pnl = float(frame["net_pnl"].sum())
    fees_paid = float(frame["fees_paid"].sum())
    slippage_paid = float(frame["slippage_paid"].sum())
    win_rate = float((frame["net_pnl"] > 0).mean())
    profit_factor = _profit_factor(frame["net_pnl"])
    average_hold_hours = float(frame["hold_hours"].mean())
    max_loss_streak = _max_consecutive_losses(frame["net_pnl"])

    ending_equity = None if starting_equity is None else starting_equity + net_pnl
    total_return_pct = None
    max_drawdown_pct = 0.0
    calmar_ratio = None
    sharpe_ratio = None
    sortino_ratio = None

    if starting_equity is not None and starting_equity > 0:
        assert ending_equity is not None  # guaranteed: ending_equity = starting_equity + net_pnl
        equity_curve = starting_equity + frame["net_pnl"].cumsum()
        peaks = equity_curve.cummax().replace(0, pd.NA)
        drawdowns = ((equity_curve - peaks) / peaks).fillna(0.0)
        max_drawdown_pct = abs(float(drawdowns.min()))
        total_return_pct = float((ending_equity - starting_equity) / starting_equity)

        daily_pnl = frame.groupby(frame["exit_time"].dt.floor("D"))["net_pnl"].sum().sort_index()
        prior_equity = (starting_equity + daily_pnl.cumsum().shift(fill_value=0)).replace(0, pd.NA)
        daily_returns = (daily_pnl / prior_equity).dropna()
        sharpe_ratio, sortino_ratio = _risk_adjusted_metrics(daily_returns, risk_free_rate)

        if daily_returns.empty:
            annual_return = None
        else:
            periods = max(len(daily_returns), 1)
            annual_return = float((ending_equity / starting_equity) ** (252 / periods) - 1)
        if annual_return is not None and max_drawdown_pct > 0:
            calmar_ratio = annual_return / max_drawdown_pct

    return PerformanceSummary(
        trade_count=len(trades),
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        fees_paid=fees_paid,
        slippage_paid=slippage_paid,
        win_rate=win_rate,
        profit_factor=profit_factor,
        average_hold_hours=average_hold_hours,
        max_drawdown_pct=max_drawdown_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        calmar_ratio=calmar_ratio,
        total_return_pct=total_return_pct,
        ending_equity=ending_equity,
        max_consecutive_losses=max_loss_streak,
    )


def summary_as_dict(summary: PerformanceSummary) -> dict[str, float | int | None]:
    return asdict(summary)
