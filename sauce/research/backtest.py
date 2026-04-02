"""Minimal backtest engine for deterministic signal and exit validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC
from typing import Any

import pandas as pd

from sauce.accounting import estimate_side_costs, estimate_trade_accounting
from sauce.exit_monitor import evaluate_exit
from sauce.indicators.core import compute_all
from sauce.performance import PerformanceSummary, TradePerformanceRecord, summarize_performance
from sauce.strategy import Position, Strategy, get_tier_params


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    starting_equity: float = 10_000.0
    warmup_bars: int = 60
    regime: str = "neutral"
    risk_free_rate: float = 0.05
    allow_static_universe: bool = False


@dataclass(frozen=True, slots=True)
class BacktestResult:
    strategy_name: str
    instrument: str
    asset_class: str
    trade_count: int
    ending_equity: float
    trades: list[TradePerformanceRecord]
    metrics: PerformanceSummary


def _coerce_index_timestamp(index_value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(index_value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def _entry_account(cash: float, current_price: float) -> dict[str, str]:
    return {
        "equity": f"{cash:.8f}",
        "buying_power": f"{cash:.8f}",
        "_ask": f"{current_price:.8f}",
    }


def backtest_strategy(
    strategy: Strategy,
    instrument: str,
    df: pd.DataFrame,
    *,
    is_crypto: bool,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a deterministic backtest over a single OHLCV DataFrame.

    Indicators are computed on history up to the prior bar so the engine does
    not use the current bar's close as future information.
    """
    if df.empty:
        raise ValueError("Backtest requires a non-empty DataFrame")

    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Backtest DataFrame missing required columns: {sorted(missing_columns)}")

    resolved_config = config or BacktestConfig()
    history = df.sort_index().copy()
    cash = resolved_config.starting_equity
    trades: list[TradePerformanceRecord] = []
    open_position: Position | None = None
    asset_class = "crypto" if is_crypto else "equity"

    for bar_index in range(resolved_config.warmup_bars, len(history)):
        current_row = history.iloc[bar_index]
        historical_window = history.iloc[:bar_index]
        indicators = compute_all(historical_window, is_crypto=is_crypto)
        if indicators is None:
            continue

        current_time = _coerce_index_timestamp(history.index[bar_index])
        current_price = float(current_row["close"])
        tier = get_tier_params(cash if cash > 0 else resolved_config.starting_equity)

        if open_position is not None:
            exit_plan = strategy.build_exit_plan(open_position, tier)
            exit_signal, managed_position = evaluate_exit(
                open_position,
                exit_plan,
                current_price,
                indicators.rsi_14,
                now=current_time.to_pydatetime(),
                atr_14=indicators.atr_14,
                regime=resolved_config.regime,
            )
            open_position = managed_position
            if exit_signal is not None:
                accounting = estimate_trade_accounting(
                    asset_class=asset_class,
                    qty=open_position.qty,
                    entry_price=open_position.entry_price,
                    exit_price=current_price,
                )
                exit_side = estimate_side_costs(asset_class, accounting.exit_notional)
                cash += accounting.exit_notional - exit_side.total_cost
                trades.append(
                    TradePerformanceRecord(
                        symbol=instrument,
                        asset_class=asset_class,
                        strategy_name=strategy.name,
                        entry_time=pd.Timestamp(open_position.entry_time),
                        exit_time=current_time,
                        entry_price=open_position.entry_price,
                        exit_price=current_price,
                        qty=open_position.qty,
                        gross_pnl=accounting.gross_pnl,
                        realized_pnl=accounting.realized_pnl,
                        fees_paid=accounting.fees_paid,
                        slippage_paid=accounting.slippage_paid,
                        hold_hours=(current_time.to_pydatetime() - open_position.entry_time).total_seconds() / 3600,
                        exit_trigger=exit_signal.trigger,
                    )
                )
                open_position = None
            continue

        signal = strategy.score(indicators, instrument, resolved_config.regime, current_price)
        if not signal.fired:
            continue

        order = strategy.build_order(signal, _entry_account(cash, current_price), tier)
        if order.qty <= 0:
            continue

        entry_price = order.limit_price or current_price
        entry_accounting = estimate_trade_accounting(
            asset_class=asset_class,
            qty=order.qty,
            entry_price=entry_price,
            exit_price=entry_price,
        )
        entry_side = estimate_side_costs(asset_class, entry_accounting.entry_notional)
        total_entry_debit = entry_accounting.entry_notional + entry_side.total_cost
        if total_entry_debit > cash:
            continue

        cash -= total_entry_debit
        open_position = Position(
            symbol=instrument,
            asset_class=asset_class,
            qty=order.qty,
            entry_price=entry_price,
            high_water_price=entry_price,
            entry_time=current_time.to_pydatetime(),
            strategy_name=strategy.name,
            stop_loss_price=order.stop_loss_price or 0.0,
            profit_target_price=order.take_profit_price or 0.0,
        )

    if open_position is not None:
        final_time = _coerce_index_timestamp(history.index[-1])
        final_price = float(history.iloc[-1]["close"])
        accounting = estimate_trade_accounting(
            asset_class=asset_class,
            qty=open_position.qty,
            entry_price=open_position.entry_price,
            exit_price=final_price,
        )
        exit_side = estimate_side_costs(asset_class, accounting.exit_notional)
        cash += accounting.exit_notional - exit_side.total_cost
        trades.append(
            TradePerformanceRecord(
                symbol=instrument,
                asset_class=asset_class,
                strategy_name=strategy.name,
                entry_time=pd.Timestamp(open_position.entry_time),
                exit_time=final_time,
                entry_price=open_position.entry_price,
                exit_price=final_price,
                qty=open_position.qty,
                gross_pnl=accounting.gross_pnl,
                realized_pnl=accounting.realized_pnl,
                fees_paid=accounting.fees_paid,
                slippage_paid=accounting.slippage_paid,
                hold_hours=(final_time.to_pydatetime() - open_position.entry_time).total_seconds() / 3600,
                exit_trigger="end_of_backtest",
            )
        )

    metrics = summarize_performance(
        trades,
        starting_equity=resolved_config.starting_equity,
        risk_free_rate=resolved_config.risk_free_rate,
    )
    ending_equity = metrics.ending_equity if metrics.ending_equity is not None else cash
    return BacktestResult(
        strategy_name=strategy.name,
        instrument=instrument,
        asset_class=asset_class,
        trade_count=len(trades),
        ending_equity=ending_equity,
        trades=trades,
        metrics=metrics,
    )
