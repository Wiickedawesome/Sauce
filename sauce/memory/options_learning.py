"""
memory/options_learning.py — Learning loop for options trades.

Mirrors the equity learning.py pattern:
  - Record options trade outcomes to strategic memory.
  - Detect options-specific win rate drift.
  - Analyze performance by IV rank bucket, delta bucket, and stage reached.
"""

import json
import logging
import math
from datetime import datetime, timezone
from typing import Any

from sauce.adapters.db import (
    close_options_position,
    get_session,
    load_open_options_positions,
    OptionsPositionRow,
)

logger = logging.getLogger(__name__)


def record_options_trade_outcome(
    position_id: str,
    exit_price: float,
    exit_reason: str,
    stages_completed: int,
    db_path: str | None = None,
) -> dict[str, Any] | None:
    """Compute and log outcome stats for a closed options position.

    Returns a summary dict or None if position not found.
    """
    session = get_session(db_path)
    try:
        row = (
            session.query(OptionsPositionRow)
            .filter(OptionsPositionRow.position_id == position_id)
            .first()
        )
        if row is None:
            return None

        pnl = (exit_price - row.entry_price) * row.qty * 100  # per-contract multiplier
        pnl_pct = (exit_price - row.entry_price) / row.entry_price if row.entry_price > 0 else 0.0
        win = pnl > 0

        entry_time = row.entry_time
        hold_minutes = 0.0
        if entry_time:
            if entry_time.tzinfo is None:
                entry_time = entry_time.replace(tzinfo=timezone.utc)
            hold_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60.0

        outcome = {
            "position_id": position_id,
            "underlying": row.underlying,
            "direction": row.direction,
            "strategy_type": row.strategy_type,
            "entry_price": row.entry_price,
            "exit_price": exit_price,
            "qty": row.qty,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "win": win,
            "exit_reason": exit_reason,
            "stages_completed": stages_completed,
            "hold_duration_minutes": round(hold_minutes, 1),
        }

        logger.info(
            "Options trade outcome: %s %s pnl=%.2f (%.1f%%) stages=%d reason=%s",
            row.underlying, row.direction, pnl, pnl_pct * 100,
            stages_completed, exit_reason,
        )

        return outcome
    except Exception as exc:  # noqa: BLE001
        logger.critical("Failed to record options trade outcome for %s: %s", position_id, exc)
        return None
    finally:
        session.close()


def detect_options_win_rate_drift(
    db_path: str | None = None,
    window: int = 10,
    threshold: float = 0.40,
) -> dict[str, Any] | None:
    """Return drift details if options win rate over last *window* closed trades < *threshold*.

    Returns None if healthy or insufficient data.
    """
    session = get_session(db_path)
    try:
        rows = (
            session.query(OptionsPositionRow)
            .filter(OptionsPositionRow.status == "closed")
            .order_by(OptionsPositionRow.id.desc())
            .limit(window)
            .all()
        )
        if len(rows) < window:
            return None

        wins = sum(1 for r in rows if r.realized_pnl > 0)
        win_rate = wins / len(rows)

        if win_rate < threshold:
            return {
                "win_rate": round(win_rate, 4),
                "window": window,
                "threshold": threshold,
                "wins": wins,
                "losses": len(rows) - wins,
                "asset_class": "options",
            }
        return None
    finally:
        session.close()


def get_options_performance_by_stage(
    db_path: str | None = None,
) -> dict[str, Any]:
    """Analyze closed options trades grouped by max stages_completed.

    Returns e.g. {0: {count: 5, avg_pnl: -42.0}, 1: {count: 3, avg_pnl: 120.0}, ...}
    """
    session = get_session(db_path)
    try:
        rows = (
            session.query(OptionsPositionRow)
            .filter(OptionsPositionRow.status == "closed")
            .all()
        )
        if not rows:
            return {"by_stage": {}, "total": 0}

        by_stage: dict[int, list[float]] = {}
        for r in rows:
            by_stage.setdefault(r.stages_completed, []).append(r.realized_pnl)

        report: dict[str, Any] = {"by_stage": {}, "total": len(rows)}
        for stage, pnls in sorted(by_stage.items()):
            avg = sum(pnls) / len(pnls) if pnls else 0.0
            wins = sum(1 for p in pnls if p > 0)
            report["by_stage"][stage] = {
                "count": len(pnls),
                "avg_pnl": round(avg, 2),
                "win_rate": round(wins / len(pnls), 4) if pnls else 0.0,
            }

        return report
    finally:
        session.close()


def get_options_performance_by_underlying(
    db_path: str | None = None,
) -> dict[str, Any]:
    """Analyze closed options trades grouped by underlying symbol."""
    session = get_session(db_path)
    try:
        rows = (
            session.query(OptionsPositionRow)
            .filter(OptionsPositionRow.status == "closed")
            .all()
        )
        if not rows:
            return {"by_symbol": {}, "total": 0}

        by_sym: dict[str, list[float]] = {}
        for r in rows:
            by_sym.setdefault(r.underlying, []).append(r.realized_pnl)

        report: dict[str, Any] = {"by_symbol": {}, "total": len(rows)}
        for sym, pnls in sorted(by_sym.items()):
            avg = sum(pnls) / len(pnls) if pnls else 0.0
            wins = sum(1 for p in pnls if p > 0)
            report["by_symbol"][sym] = {
                "count": len(pnls),
                "avg_pnl": round(avg, 2),
                "total_pnl": round(sum(pnls), 2),
                "win_rate": round(wins / len(pnls), 4) if pnls else 0.0,
            }

        return report
    finally:
        session.close()
