#!/usr/bin/env python3
"""Generate a net-performance report from completed Sauce trades."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sauce.adapters.db import get_session
from sauce.core.config import get_settings
from sauce.db import DailySummaryRow, load_completed_trades
from sauce.performance import summarize_performance


def _default_starting_equity() -> float | None:
    session = get_session()
    try:
        row = session.query(DailySummaryRow).order_by(DailySummaryRow.date.asc()).first()
        if row is None:
            return None
        return float(row.starting_equity or 0.0) or None
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only report on the newest N trades")
    parser.add_argument(
        "--starting-equity",
        type=float,
        default=None,
        help="Override the equity baseline used for return and drawdown metrics",
    )
    args = parser.parse_args()

    settings = get_settings()
    trades = load_completed_trades(limit=args.limit)
    starting_equity = args.starting_equity if args.starting_equity is not None else _default_starting_equity()
    summary = summarize_performance(
        trades,
        starting_equity=starting_equity,
        risk_free_rate=settings.performance_risk_free_rate,
    )

    print("=" * 72)
    print("SAUCE PERFORMANCE REPORT")
    print("=" * 72)
    print(f"Trades              : {summary.trade_count}")
    print(f"Gross P&L           : ${summary.gross_pnl:,.2f}")
    print(f"Net P&L             : ${summary.net_pnl:,.2f}")
    print(f"Fees Paid           : ${summary.fees_paid:,.2f}")
    print(f"Slippage Paid       : ${summary.slippage_paid:,.2f}")
    print(f"Win Rate            : {summary.win_rate:.2%}")
    print(f"Profit Factor       : {summary.profit_factor if summary.profit_factor is not None else 'n/a'}")
    print(f"Average Hold Hours  : {summary.average_hold_hours:.2f}")
    print(f"Max Drawdown        : {summary.max_drawdown_pct:.2%}")
    print(f"Sharpe Ratio        : {summary.sharpe_ratio if summary.sharpe_ratio is not None else 'n/a'}")
    print(f"Sortino Ratio       : {summary.sortino_ratio if summary.sortino_ratio is not None else 'n/a'}")
    print(f"Calmar Ratio        : {summary.calmar_ratio if summary.calmar_ratio is not None else 'n/a'}")
    print(f"Total Return        : {summary.total_return_pct if summary.total_return_pct is not None else 'n/a'}")
    print(f"Ending Equity       : {summary.ending_equity if summary.ending_equity is not None else 'n/a'}")
    print(f"Max Loss Streak     : {summary.max_consecutive_losses}")


if __name__ == "__main__":
    main()
