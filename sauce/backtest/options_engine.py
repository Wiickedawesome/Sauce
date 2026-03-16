"""
backtest/options_engine.py — Bar-replay backtesting engine for options.

Simulates the "Double Up & Take Gains" compounding strategy on historical
underlying price data. Uses a simplified options pricing model (delta-based
proxy + theta decay) since full option chain history is rarely available.

No LLM calls — purely deterministic simulation.

Usage:
    from sauce.backtest.options_engine import run_options_backtest
    from sauce.backtest.options_models import OptionsBacktestConfig
    result = run_options_backtest("SPY", df, config=OptionsBacktestConfig())
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pandas as pd

from sauce.backtest.options_models import (
    OptionsBacktestConfig,
    OptionsBacktestResult,
    OptionsBacktestTrade,
    OptionsExitReason,
)

logger = logging.getLogger(__name__)


class _OptionsPosition:
    """Mutable in-flight options position during simulation."""

    __slots__ = (
        "symbol", "direction", "entry_time", "entry_price", "underlying_entry",
        "qty", "remaining_qty", "delta", "stages_completed", "realized_pnl",
        "trailing_stop_price", "stage_triggers", "entry_bar_idx",
    )

    def __init__(
        self,
        symbol: str,
        direction: str,
        entry_time: datetime,
        entry_price: float,
        underlying_entry: float,
        qty: int,
        delta: float,
        stage_triggers: list[float],
        entry_bar_idx: int,
    ) -> None:
        self.symbol = symbol
        self.direction = direction
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.underlying_entry = underlying_entry
        self.qty = qty
        self.remaining_qty = qty
        self.delta = delta
        self.stages_completed = 0
        self.realized_pnl = 0.0
        self.trailing_stop_price: float | None = None
        self.stage_triggers = stage_triggers
        self.entry_bar_idx = entry_bar_idx


def _estimate_option_price(
    entry_option_price: float,
    underlying_current: float,
    underlying_entry: float,
    delta: float,
    bars_held: int,
    theta_daily: float,
    bars_per_day: int,
) -> float:
    """Estimate current option price using delta approximation + theta decay.

    price ≈ entry + delta * (underlying_change) - theta * time
    """
    underlying_change = underlying_current - underlying_entry
    delta_pnl = delta * underlying_change
    days_held = bars_held / max(bars_per_day, 1)
    theta_cost = theta_daily * entry_option_price * days_held
    estimated = entry_option_price + delta_pnl - theta_cost
    return max(estimated, 0.01)  # options can't go below ~0


def _compute_metrics(result: OptionsBacktestResult) -> None:
    """Populate aggregate metrics on an OptionsBacktestResult in-place."""
    trades = result.trades
    result.total_trades = len(trades)

    if not trades:
        return

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    result.winning_trades = len(wins)
    result.losing_trades = len(losses)
    result.win_rate = len(wins) / len(trades)
    result.avg_pnl = sum(t.pnl for t in trades) / len(trades)
    result.avg_bars_held = sum(t.bars_held for t in trades) / len(trades)
    result.avg_stages_completed = sum(t.stages_completed for t in trades) / len(trades)

    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    if gross_loss > 0:
        result.profit_factor = gross_profit / gross_loss

    # Equity curve metrics
    curve = result.equity_curve
    if len(curve) >= 2:
        start = curve[0]
        end = curve[-1]
        if start > 0:
            result.total_return = (end - start) / start

        # Sharpe from bar-to-bar returns
        returns = []
        for i in range(1, len(curve)):
            if curve[i - 1] > 0:
                returns.append((curve[i] - curve[i - 1]) / curve[i - 1])
        if len(returns) >= 2:
            mean_r = sum(returns) / len(returns)
            var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = math.sqrt(var) if var > 0 else 0.0
            if std_r > 0:
                result.sharpe_ratio = round(mean_r / std_r * math.sqrt(252), 4)

        # Max drawdown
        peak = curve[0]
        max_dd = 0.0
        for nav in curve[1:]:
            if nav > peak:
                peak = nav
            elif peak > 0:
                dd = (nav - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        result.max_drawdown = round(max_dd, 6)


def run_options_backtest(
    symbol: str,
    df: pd.DataFrame,
    config: OptionsBacktestConfig | None = None,
) -> OptionsBacktestResult:
    """
    Run a bar-replay options backtest on historical underlying OHLCV data.

    Parameters
    ----------
    symbol : str
        Underlying symbol (e.g. "SPY").
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex. Must have: open, high, low, close, volume.
    config : OptionsBacktestConfig
        Backtest parameters. Uses defaults if None.

    Returns
    -------
    OptionsBacktestResult with trades, equity curve, and metrics.
    """
    if config is None:
        config = OptionsBacktestConfig()

    result = OptionsBacktestResult(symbol=symbol, config=config)

    if df.empty or len(df) < 20:
        logger.warning("Insufficient data for options backtest on %s", symbol)
        return result

    # Normalize columns
    cols = {c.lower(): c for c in df.columns}
    close_col = cols.get("close", "close")
    high_col = cols.get("high", "high")
    low_col = cols.get("low", "low")

    capital = config.initial_capital
    equity_curve: list[float] = [capital]
    trades: list[OptionsBacktestTrade] = []
    positions: list[_OptionsPosition] = []

    bars_per_day = 26  # ~15 min bars for a 6.5h trading day

    # Build stage trigger multipliers
    stage_multipliers = [
        config.profit_multiplier ** (i + 1)
        for i in range(config.compound_stages)
    ]

    # Simple entry signal: RSI-based (compute inline to avoid heavy deps)
    closes = df[close_col].values
    rsi = _simple_rsi(closes, 14)

    for bar_idx in range(20, len(df)):
        bar_time = df.index[bar_idx]
        if not isinstance(bar_time, datetime):
            bar_time = pd.Timestamp(bar_time).to_pydatetime()
        if bar_time.tzinfo is None:
            bar_time = bar_time.replace(tzinfo=timezone.utc)

        current_close = float(closes[bar_idx])
        current_high = float(df[high_col].iloc[bar_idx])
        current_low = float(df[low_col].iloc[bar_idx])

        # ── Evaluate exits ────────────────────────────────────────────
        closed_positions: list[_OptionsPosition] = []
        for pos in positions:
            opt_price = _estimate_option_price(
                pos.entry_price, current_close, pos.underlying_entry,
                pos.delta, bar_idx - pos.entry_bar_idx,
                config.theta_decay_annual / 252, bars_per_day,
            )

            # Hard stop
            loss_pct = (opt_price - pos.entry_price) / pos.entry_price
            if loss_pct <= -config.max_loss_pct:
                pnl = (opt_price - pos.entry_price) * pos.remaining_qty * 100
                pnl -= config.commission_per_contract * pos.remaining_qty
                capital += pnl
                trades.append(OptionsBacktestTrade(
                    symbol=symbol, direction=pos.direction,
                    entry_time=pos.entry_time, exit_time=bar_time,
                    entry_price=pos.entry_price, exit_price=opt_price,
                    qty=pos.remaining_qty, pnl=round(pnl, 2),
                    pnl_pct=round(loss_pct, 4),
                    exit_reason=OptionsExitReason.HARD_STOP,
                    stages_completed=pos.stages_completed,
                    bars_held=bar_idx - pos.entry_bar_idx,
                ))
                closed_positions.append(pos)
                continue

            # Trailing stop
            if pos.trailing_stop_price is not None and opt_price <= pos.trailing_stop_price:
                pnl = (opt_price - pos.entry_price) * pos.remaining_qty * 100
                pnl -= config.commission_per_contract * pos.remaining_qty
                capital += pnl
                pnl_pct = (opt_price - pos.entry_price) / pos.entry_price
                trades.append(OptionsBacktestTrade(
                    symbol=symbol, direction=pos.direction,
                    entry_time=pos.entry_time, exit_time=bar_time,
                    entry_price=pos.entry_price, exit_price=opt_price,
                    qty=pos.remaining_qty, pnl=round(pnl, 2),
                    pnl_pct=round(pnl_pct, 4),
                    exit_reason=OptionsExitReason.TRAILING_STOP,
                    stages_completed=pos.stages_completed,
                    bars_held=bar_idx - pos.entry_bar_idx,
                ))
                closed_positions.append(pos)
                continue

            # Walk compound stages
            for stage_idx in range(pos.stages_completed, len(pos.stage_triggers)):
                trigger = pos.entry_price * pos.stage_triggers[stage_idx]
                if opt_price >= trigger:
                    sell_qty = max(1, int(pos.remaining_qty * config.sell_fraction))
                    sell_qty = min(sell_qty, pos.remaining_qty)

                    pnl = (opt_price - pos.entry_price) * sell_qty * 100
                    pnl -= config.commission_per_contract * sell_qty
                    capital += pnl
                    pos.remaining_qty -= sell_qty
                    pos.stages_completed += 1
                    pos.realized_pnl += pnl

                    # Set trailing stop
                    pos.trailing_stop_price = opt_price * (1 - config.trailing_stop_pct)

                    if pos.remaining_qty <= 0:
                        pnl_pct = (opt_price - pos.entry_price) / pos.entry_price
                        reason_map = {
                            1: OptionsExitReason.STAGE_1,
                            2: OptionsExitReason.STAGE_2,
                            3: OptionsExitReason.STAGE_3,
                        }
                        trades.append(OptionsBacktestTrade(
                            symbol=symbol, direction=pos.direction,
                            entry_time=pos.entry_time, exit_time=bar_time,
                            entry_price=pos.entry_price, exit_price=opt_price,
                            qty=pos.qty, pnl=round(pos.realized_pnl, 2),
                            pnl_pct=round(pnl_pct, 4),
                            exit_reason=reason_map.get(
                                pos.stages_completed, OptionsExitReason.STAGE_3,
                            ),
                            stages_completed=pos.stages_completed,
                            bars_held=bar_idx - pos.entry_bar_idx,
                        ))
                        closed_positions.append(pos)
                        break
                else:
                    break

            # Update trailing stop if price rose
            if pos.trailing_stop_price is not None and pos not in closed_positions:
                new_stop = opt_price * (1 - config.trailing_stop_pct)
                if new_stop > pos.trailing_stop_price:
                    pos.trailing_stop_price = new_stop

        for cp in closed_positions:
            positions.remove(cp)

        # ── Entry signal: RSI oversold (call) / overbought (put) ──────
        if bar_idx < len(rsi) and not math.isnan(rsi[bar_idx]):
            current_rsi = rsi[bar_idx]
            total_exposure = sum(
                p.entry_price * p.remaining_qty * 100 for p in positions
            )
            max_exposure = capital * config.max_total_exposure_pct
            position_cost = capital * config.position_size_pct

            if total_exposure < max_exposure and position_cost > 0:
                direction = None
                delta = 0.0

                if current_rsi < 35:
                    direction = "long_call"
                    delta = 0.45
                elif current_rsi > 65:
                    direction = "long_put"
                    delta = -0.45

                if direction is not None:
                    # Estimate initial option price as fraction of underlying
                    opt_entry_price = current_close * abs(delta) * 0.02
                    opt_entry_price = max(opt_entry_price, 0.10)
                    # Apply slippage
                    opt_entry_price *= (1 + config.slippage_pct)
                    qty = max(1, int(position_cost / (opt_entry_price * 100)))
                    cost = opt_entry_price * qty * 100
                    cost += config.commission_per_contract * qty

                    if cost <= capital * config.position_size_pct * 1.5:
                        capital -= cost
                        positions.append(_OptionsPosition(
                            symbol=symbol,
                            direction=direction,
                            entry_time=bar_time,
                            entry_price=opt_entry_price,
                            underlying_entry=current_close,
                            qty=qty,
                            delta=abs(delta),
                            stage_triggers=stage_multipliers,
                            entry_bar_idx=bar_idx,
                        ))

        # Mark-to-market for equity curve
        mtm = capital
        for pos in positions:
            opt_price = _estimate_option_price(
                pos.entry_price, current_close, pos.underlying_entry,
                pos.delta, bar_idx - pos.entry_bar_idx,
                config.theta_decay_annual / 252, bars_per_day,
            )
            mtm += opt_price * pos.remaining_qty * 100
        equity_curve.append(round(mtm, 2))

    # Close any remaining positions at last bar
    if positions:
        last_close = float(closes[-1])
        last_time = df.index[-1]
        if not isinstance(last_time, datetime):
            last_time = pd.Timestamp(last_time).to_pydatetime()
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)

        for pos in positions:
            opt_price = _estimate_option_price(
                pos.entry_price, last_close, pos.underlying_entry,
                pos.delta, len(df) - 1 - pos.entry_bar_idx,
                config.theta_decay_annual / 252, bars_per_day,
            )
            pnl = (opt_price - pos.entry_price) * pos.remaining_qty * 100
            pnl -= config.commission_per_contract * pos.remaining_qty
            pnl += pos.realized_pnl
            pnl_pct = (opt_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0.0
            trades.append(OptionsBacktestTrade(
                symbol=symbol, direction=pos.direction,
                entry_time=pos.entry_time, exit_time=last_time,
                entry_price=pos.entry_price, exit_price=opt_price,
                qty=pos.qty, pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 4),
                exit_reason=OptionsExitReason.EXPIRY,
                stages_completed=pos.stages_completed,
                bars_held=len(df) - 1 - pos.entry_bar_idx,
            ))

    result.trades = trades
    result.equity_curve = equity_curve
    _compute_metrics(result)
    return result


def _simple_rsi(closes: object, period: int = 14) -> list[float]:
    """Compute RSI without external dependencies."""
    rsi = [float("nan")] * len(closes)
    if len(closes) < period + 1:
        return rsi

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi
