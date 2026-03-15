"""
backtest/engine.py — Vectorized bar-replay backtesting engine.

Walks a historical OHLCV DataFrame bar-by-bar, runs the indicator library +
setup scanner at each step (using only data visible up to that bar), and
simulates entry / exit logic with ATR-based stop-loss and profit targets.

No LLM calls — the backtest assumes Claude always approves passing setups
with the deterministic setup score as the confidence value (score / 100).

Usage:
    from sauce.backtest import run_backtest, BacktestConfig
    result = run_backtest("BTC/USD", df, regime="RANGING", config=BacktestConfig())
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

import pandas as pd

from sauce.backtest.models import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    ExitReason,
    TradeDirection,
)
from sauce.adapters.market_data import is_crypto as _is_crypto
from sauce.indicators.core import compute_all

logger = logging.getLogger(__name__)

_VALID_REGIMES = frozenset({
    "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "DEAD",
})


def _compute_metrics(result: BacktestResult) -> None:
    """Populate aggregate metrics on a BacktestResult in-place."""
    trades = result.trades
    result.total_trades = len(trades)

    if trades:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(trades)
        result.avg_pnl = sum(t.pnl for t in trades) / len(trades)
        result.avg_bars_held = sum(t.bars_held for t in trades) / len(trades)

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
                # Annualise assuming 30-min bars, ~13 bars/day equity, ~48/day crypto
                bars_per_day = 48 if _is_crypto(result.symbol) else 13
                result.sharpe_ratio = round(
                    mean_r / std_r * math.sqrt(bars_per_day * 252), 4,
                )

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


# ── Position tracking ────────────────────────────────────────────────────────

class _Position:
    """Mutable in-flight position used during simulation."""

    __slots__ = (
        "symbol", "direction", "setup_type", "entry_time", "entry_price",
        "qty", "stop_price", "target_price", "confidence", "entry_bar_idx",
    )

    def __init__(
        self,
        symbol: str,
        direction: TradeDirection,
        setup_type: str,
        entry_time: datetime,
        entry_price: float,
        qty: float,
        stop_price: float,
        target_price: float,
        confidence: float,
        entry_bar_idx: int,
    ) -> None:
        self.symbol = symbol
        self.direction = direction
        self.setup_type = setup_type
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.qty = qty
        self.stop_price = stop_price
        self.target_price = target_price
        self.confidence = confidence
        self.entry_bar_idx = entry_bar_idx


def _apply_slippage(price: float, direction: TradeDirection, pct: float) -> float:
    """Adjust price for slippage (worse fill)."""
    if direction == TradeDirection.LONG:
        return price * (1.0 + pct)  # buy higher
    return price * (1.0 - pct)  # sell lower


def _close_position(
    pos: _Position,
    exit_price: float,
    exit_time: datetime,
    exit_reason: ExitReason,
    bar_idx: int,
    config: BacktestConfig,
) -> BacktestTrade:
    """Close a position and return a BacktestTrade record."""
    # Slippage on exit is adverse: selling a LONG fills lower, covering a
    # SHORT fills higher.  We pass the *exit* trade direction (opposite of the
    # position direction) so _apply_slippage applies the correct penalty.
    exit_direction = (
        TradeDirection.SHORT if pos.direction == TradeDirection.LONG
        else TradeDirection.LONG
    )
    slipped_exit = _apply_slippage(exit_price, exit_direction, config.slippage_pct)
    notional = pos.entry_price * pos.qty
    if pos.direction == TradeDirection.LONG:
        raw_pnl = (slipped_exit - pos.entry_price) * pos.qty
    else:
        raw_pnl = (pos.entry_price - slipped_exit) * pos.qty

    net_pnl = raw_pnl - 2 * config.commission_per_trade  # entry + exit
    pnl_pct = net_pnl / notional if notional > 0 else 0.0

    return BacktestTrade(
        symbol=pos.symbol,
        direction=pos.direction,
        setup_type=pos.setup_type,
        entry_time=pos.entry_time,
        exit_time=exit_time,
        entry_price=pos.entry_price,
        exit_price=slipped_exit,
        qty=pos.qty,
        pnl=round(net_pnl, 2),
        pnl_pct=round(pnl_pct, 6),
        exit_reason=exit_reason,
        confidence=pos.confidence,
        bars_held=bar_idx - pos.entry_bar_idx,
    )


# ── Main engine ──────────────────────────────────────────────────────────────

def run_backtest(
    symbol: str,
    df: pd.DataFrame,
    regime: str = "RANGING",
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """
    Run a bar-replay backtest on historical OHLCV data.

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. "BTC/USD", "SPY").
    df : pd.DataFrame
        Full OHLCV DataFrame with DatetimeIndex. Must have columns:
        open, high, low, close, volume.
    regime : str
        Market regime to use for setup scanning (constant throughout backtest).
    config : BacktestConfig, optional
        Backtest parameters. Defaults to BacktestConfig().

    Returns
    -------
    BacktestResult
        Complete backtest output including trades, equity curve, and metrics.
    """
    if config is None:
        config = BacktestConfig()

    # Validate regime
    if regime not in _VALID_REGIMES:
        raise ValueError(f"{regime!r} is not a valid MarketRegime")

    result = BacktestResult(symbol=symbol, config=config)
    crypto = _is_crypto(symbol)

    if len(df) < config.lookback_bars + 10:
        logger.warning(
            "backtest[%s]: insufficient data (%d bars, need %d+10)",
            symbol, len(df), config.lookback_bars,
        )
        return result

    capital = config.initial_capital
    position: _Position | None = None

    # Walk forward from lookback_bars to end of data
    start_idx = config.lookback_bars
    timestamps = []
    equity_curve = []

    for i in range(start_idx, len(df), config.bar_step):
        window = df.iloc[max(0, i - config.lookback_bars) : i + 1]
        bar = df.iloc[i]
        bar_time = _bar_timestamp(df, i)

        timestamps.append(bar_time)

        # ── Check exit conditions on open position ────────────────────
        if position is not None:
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])

            exit_reason: ExitReason | None = None
            exit_price = close

            if position.direction == TradeDirection.LONG:
                if low <= position.stop_price:
                    exit_reason = ExitReason.STOP_LOSS
                    exit_price = position.stop_price
                elif high >= position.target_price:
                    exit_reason = ExitReason.PROFIT_TARGET
                    exit_price = position.target_price
            else:  # SHORT
                if high >= position.stop_price:
                    exit_reason = ExitReason.STOP_LOSS
                    exit_price = position.stop_price
                elif low <= position.target_price:
                    exit_reason = ExitReason.PROFIT_TARGET
                    exit_price = position.target_price

            if exit_reason is not None:
                trade = _close_position(
                    position, exit_price, bar_time, exit_reason, i, config,
                )
                result.trades.append(trade)
                capital += trade.pnl
                position = None

        # ── Record equity ─────────────────────────────────────────────
        unrealised = 0.0
        if position is not None:
            close = float(bar["close"])
            if position.direction == TradeDirection.LONG:
                unrealised = (close - position.entry_price) * position.qty
            else:
                unrealised = (position.entry_price - close) * position.qty
        equity_curve.append(round(capital + unrealised, 2))

        # ── Try to enter a new position ───────────────────────────────
        if position is not None:
            continue  # already in a trade

        # Compute indicators on the visible window
        indicators = compute_all(window, is_crypto=crypto)

        # Run setup scanner
        from sauce.core.setups import scan_setups  # local import to avoid circulars

        setups = scan_setups(
            symbol=symbol,
            indicators=indicators,
            df=window,
            regime=regime,
            as_of=bar_time,
        )

        # Take the first passing setup
        passing = [s for s in setups if s.passed]
        if not passing:
            continue

        best = max(passing, key=lambda s: s.score)
        confidence = best.score / 100.0

        if confidence < config.min_confidence:
            continue

        # Size the position
        entry_price = float(bar["close"])
        slipped_entry = _apply_slippage(
            entry_price, TradeDirection.LONG, config.slippage_pct,
        )
        max_notional = capital * config.max_position_pct
        qty = max_notional / slipped_entry if slipped_entry > 0 else 0.0
        if qty <= 0:
            continue

        # ATR-based stops
        atr = indicators.atr_14
        if atr is None or atr <= 0:
            continue  # can't set stops without ATR

        stop_price = slipped_entry - atr * config.stop_loss_atr_multiple
        target_price = slipped_entry + atr * config.profit_target_atr_multiple

        position = _Position(
            symbol=symbol,
            # Short-selling P&L math exists in _close_position but is
            # unreachable until setups emit short signals.
            direction=TradeDirection.LONG,
            setup_type=best.setup_type,
            entry_time=bar_time,
            entry_price=slipped_entry,
            qty=qty,
            stop_price=stop_price,
            target_price=target_price,
            confidence=confidence,
            entry_bar_idx=i,
        )

    # ── Close any remaining position at last bar's close ──────────────
    if position is not None:
        last_bar = df.iloc[-1]
        trade = _close_position(
            position,
            float(last_bar["close"]),
            _bar_timestamp(df, len(df) - 1),
            ExitReason.END_OF_DATA,
            len(df) - 1,
            config,
        )
        result.trades.append(trade)
        capital += trade.pnl

    result.equity_curve = equity_curve
    result.timestamps = timestamps
    _compute_metrics(result)
    return result


def _bar_timestamp(df: pd.DataFrame, idx: int) -> datetime:
    """Extract a timezone-aware UTC datetime from the DataFrame index."""
    ts = df.index[idx]
    if hasattr(ts, "to_pydatetime"):
        dt = ts.to_pydatetime()
    else:
        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
