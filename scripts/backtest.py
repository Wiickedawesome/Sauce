#!/usr/bin/env python3
"""Run a simple backtest for a supported Sauce strategy and instrument."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sauce.adapters.market_data import get_history
from sauce.research.backtest import BacktestConfig, backtest_strategy
from sauce.strategies.crypto_momentum import CryptoMomentumReversion
from sauce.strategies.equity_momentum import EquityMomentum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=["crypto_momentum", "equity_momentum"])
    parser.add_argument("instrument", help="Symbol or crypto pair to backtest")
    parser.add_argument("--bars", type=int, default=300, help="Number of historical bars to fetch")
    parser.add_argument("--starting-equity", type=float, default=10_000.0)
    args = parser.parse_args()

    if args.strategy == "crypto_momentum":
        strategy = CryptoMomentumReversion()
        timeframe = "1Hour"
        is_crypto = True
    else:
        strategy = EquityMomentum()
        timeframe = "1Day"
        is_crypto = False

    history = get_history(args.instrument, timeframe=timeframe, bars=args.bars)
    if history is None or history.empty:
        raise SystemExit(f"No history returned for {args.instrument}")

    result = backtest_strategy(
        strategy,
        args.instrument,
        history,
        is_crypto=is_crypto,
        config=BacktestConfig(starting_equity=args.starting_equity),
    )

    print(f"Strategy      : {result.strategy_name}")
    print(f"Instrument    : {result.instrument}")
    print(f"Trades        : {result.trade_count}")
    print(f"Ending Equity : {result.ending_equity:,.2f}")
    print(f"Net P&L       : {result.metrics.net_pnl:,.2f}")
    print(f"Sharpe        : {result.metrics.sharpe_ratio}")
    print(f"Drawdown      : {result.metrics.max_drawdown_pct:.2%}")


if __name__ == "__main__":
    main()
