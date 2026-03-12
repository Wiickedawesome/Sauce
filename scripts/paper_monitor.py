#!/usr/bin/env python3
"""
scripts/paper_monitor.py — CLI dashboard for paper-trading validation.

Usage:
    python scripts/paper_monitor.py          # summary view
    python scripts/paper_monitor.py --detail  # per-criterion history
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select
from sqlalchemy.orm import Session

from sauce.core.config import get_settings
from sauce.core.validation import (
    REQUIRED_CONSECUTIVE_DAYS,
    check_claude_calibration,
    check_expectancy,
    check_max_drawdown,
    check_max_single_day_loss,
    check_sharpe_ratio,
    check_win_rate,
)
from sauce.memory.db import ValidationResultRow, get_session as strategic_get_session


def _latest_results(strategic_db_path: str, limit: int = 30) -> list[ValidationResultRow]:
    with strategic_get_session(strategic_db_path) as session:
        stmt = (
            select(ValidationResultRow)
            .order_by(ValidationResultRow.date.desc())
            .limit(limit)
        )
        return list(session.execute(stmt).scalars().all())


def _print_summary(settings) -> None:
    db_path = settings.db_path
    strategic_db_path = str(settings.strategic_memory_db_path)

    criteria = [
        ("Win Rate (>52%)", *check_win_rate(strategic_db_path)),
        ("Expectancy (+, ≥50 trades)", *check_expectancy(strategic_db_path)),
        ("Max Drawdown (<8%)", *check_max_drawdown(db_path)),
        ("Sharpe Ratio (>0.8)", *check_sharpe_ratio(db_path)),
        ("Max Single-Day Loss (<3%)", *check_max_single_day_loss(db_path)),
        ("Claude Calibration (>0.60)", *check_claude_calibration(strategic_db_path)),
    ]

    rows = _latest_results(strategic_db_path, limit=1)
    consecutive = rows[0].consecutive_days if rows else 0

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║          SAUCE Paper-Trading Validation              ║")
    print("╠══════════════════════════════════════════════════════╣")
    for name, passed, value in criteria:
        status = "✅" if passed else "❌"
        print(f"║  {status} {name:<35} {value:>8.4f} ║")
    print("╠══════════════════════════════════════════════════════╣")
    pct = (consecutive / REQUIRED_CONSECUTIVE_DAYS) * 100
    bar_len = 30
    filled = int(bar_len * consecutive / REQUIRED_CONSECUTIVE_DAYS)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"║  Consecutive Days: {consecutive:>3}/{REQUIRED_CONSECUTIVE_DAYS:<3}  [{bar}] {pct:5.1f}% ║")
    if consecutive >= REQUIRED_CONSECUTIVE_DAYS:
        print("║  🎉  VALIDATION PASSED — Ready for live trading!    ║")
    print("╚══════════════════════════════════════════════════════╝\n")


def _print_detail(settings) -> None:
    strategic_db_path = str(settings.strategic_memory_db_path)
    rows = _latest_results(strategic_db_path, limit=30)

    if not rows:
        print("No validation results recorded yet.")
        return

    print(f"\n{'Date':<12} {'WR':>6} {'Exp':>8} {'DD':>7} {'SR':>6} {'MDL':>7} {'Cal':>6} {'Pass':>5} {'Streak':>7}")
    print("─" * 72)
    for r in reversed(rows):
        p = "✅" if r.all_passed else "❌"
        print(
            f"{r.date:<12} "
            f"{r.win_rate:>6.3f} "
            f"{r.expectancy:>8.4f} "
            f"{r.max_drawdown_pct:>7.4f} "
            f"{r.sharpe_ratio:>6.3f} "
            f"{r.max_single_day_loss_pct:>7.4f} "
            f"{r.calibration_score:>6.3f} "
            f"{p:>5} "
            f"{r.consecutive_days:>7}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SAUCE paper-trading validation monitor")
    parser.add_argument("--detail", action="store_true", help="Show per-day history table")
    args = parser.parse_args()

    settings = get_settings()

    if args.detail:
        _print_detail(settings)
    else:
        _print_summary(settings)


if __name__ == "__main__":
    main()
