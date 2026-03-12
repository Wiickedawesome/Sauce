"""
core/learning.py — Learning Loop: post-trade analysis and drift detection.

Sprint 6 — analyzes trade outcomes, detects win-rate drift, reports weekly
performance, tracks Claude calibration accuracy, and updates symbol behavior.
"""

import logging
import math
from datetime import date as date_cls, datetime, timedelta, timezone
from typing import Any

from sauce.memory.db import (
    ClaudeCalibrationRow,
    SetupPerformanceRow,
    get_session,
    write_claude_calibration,
    write_setup_performance,
    write_symbol_behavior,
    write_weekly_performance,
)
from sauce.core.schemas import (
    ClaudeCalibrationEntry,
    SetupPerformanceEntry,
    SetupType,
    SymbolLearnedBehaviorEntry,
    WeeklyPerformanceEntry,
)

logger = logging.getLogger(__name__)


# ── Trade Outcome Writer ─────────────────────────────────────────────────────


def record_trade_outcome(
    perf_entry: SetupPerformanceEntry,
    db_path: str,
    calibration_entry: ClaudeCalibrationEntry | None = None,
) -> None:
    """Persist a completed trade and optional calibration data to strategic memory."""
    write_setup_performance(perf_entry, db_path)
    if calibration_entry is not None:
        write_claude_calibration(calibration_entry, db_path)


# ── Win Rate Drift Detector ──────────────────────────────────────────────────


def detect_win_rate_drift(
    db_path: str,
    window: int = 20,
    threshold: float = 0.45,
) -> dict[str, Any] | None:
    """
    Return drift details if win rate over the last *window* trades < *threshold*.

    Returns None if healthy or insufficient data.
    """
    session = get_session(db_path)
    try:
        rows = (
            session.query(SetupPerformanceRow)
            .order_by(SetupPerformanceRow.id.desc())
            .limit(window)
            .all()
        )
        if len(rows) < window:
            return None

        wins = sum(1 for r in rows if r.win)
        win_rate = wins / len(rows)

        if win_rate < threshold:
            return {
                "win_rate": round(win_rate, 4),
                "window": window,
                "threshold": threshold,
                "wins": wins,
                "losses": len(rows) - wins,
            }
        return None
    finally:
        session.close()


# ── Claude Calibration Analyzer ──────────────────────────────────────────────


def analyze_claude_calibration(db_path: str) -> dict[str, Any]:
    """
    Group calibration entries into confidence buckets and compute actual win rate
    per bucket.  A well-calibrated model has actual ≈ expected per bucket.

    Returns ``{buckets: {label: {count, actual_win_rate, expected_midpoint}},
    total_entries: N}``.
    """
    session = get_session(db_path)
    try:
        rows = session.query(ClaudeCalibrationRow).all()
        if not rows:
            return {"buckets": {}, "total_entries": 0}

        bucket_defs = [
            ("0.50-0.60", 0.50, 0.60),
            ("0.60-0.70", 0.60, 0.70),
            ("0.70-0.80", 0.70, 0.80),
            ("0.80-0.90", 0.80, 0.90),
            ("0.90-1.00", 0.90, 1.01),  # 1.01 upper bound to include 1.0
        ]

        buckets: dict[str, list[bool]] = {label: [] for label, _, _ in bucket_defs}

        for row in rows:
            is_win = row.outcome == "win"
            for label, lo, hi in bucket_defs:
                if lo <= row.confidence_stated < hi:
                    buckets[label].append(is_win)
                    break

        report: dict[str, Any] = {}
        for label, lo, hi_raw in bucket_defs:
            outcomes = buckets[label]
            if not outcomes:
                continue
            hi = min(hi_raw, 1.0)
            report[label] = {
                "count": len(outcomes),
                "actual_win_rate": round(sum(outcomes) / len(outcomes), 4),
                "expected_midpoint": round((lo + hi) / 2, 2),
            }

        return {"buckets": report, "total_entries": len(rows)}
    finally:
        session.close()


# ── Symbol Learned Behavior Updater ──────────────────────────────────────────


def update_symbol_learned_behavior(
    symbol: str,
    setup_type: SetupType,
    db_path: str,
) -> SymbolLearnedBehaviorEntry | None:
    """
    Recompute aggregated behavior for *symbol*/*setup_type* from trade history.

    Returns the updated entry, or None if fewer than 3 trades exist.
    """
    session = get_session(db_path)
    try:
        rows = (
            session.query(SetupPerformanceRow)
            .filter(
                SetupPerformanceRow.symbol == symbol,
                SetupPerformanceRow.setup_type == setup_type,
            )
            .all()
        )
        if len(rows) < 3:
            return None

        losses = [abs(r.pnl) for r in rows if not r.win and r.pnl < 0]
        avg_reversion = round(sum(losses) / len(losses), 6) if losses else None

        wins_pnl = [r.pnl for r in rows if r.win and r.pnl > 0]
        avg_bounce = round(sum(wins_pnl) / len(wins_pnl), 6) if wins_pnl else None

        entry = SymbolLearnedBehaviorEntry(
            symbol=symbol,
            setup_type=setup_type,
            avg_reversion_depth=avg_reversion,
            avg_bounce_magnitude=avg_bounce,
            sample_size=len(rows),
        )
        write_symbol_behavior(entry, db_path)
        return entry
    finally:
        session.close()


# ── Weekly Performance Reporter ──────────────────────────────────────────────


def generate_weekly_report(
    week: str,
    db_path: str,
) -> list[WeeklyPerformanceEntry]:
    """
    Aggregate SetupPerformanceRow for the given ISO week (``YYYY-Www``) into
    per-setup WeeklyPerformanceEntry records, persist them, and return them.
    """
    session = get_session(db_path)
    try:
        year = int(week.split("-W")[0])
        week_num = int(week.split("-W")[1])
        monday = date_cls.fromisocalendar(year, week_num, 1)
        sunday = date_cls.fromisocalendar(year, week_num, 7)
        date_start = monday.isoformat()
        date_end = sunday.isoformat()

        rows = (
            session.query(SetupPerformanceRow)
            .filter(
                SetupPerformanceRow.date >= date_start,
                SetupPerformanceRow.date <= date_end,
            )
            .all()
        )
        if not rows:
            return []

        by_setup: dict[str, list] = {}
        for r in rows:
            by_setup.setdefault(r.setup_type, []).append(r)

        entries: list[WeeklyPerformanceEntry] = []
        for setup_type_val, setup_rows in by_setup.items():
            n = len(setup_rows)
            wins = sum(1 for r in setup_rows if r.win)
            pnls = [r.pnl for r in setup_rows]
            avg_pnl = sum(pnls) / n
            win_rate = wins / n

            if n >= 2:
                variance = sum((p - avg_pnl) ** 2 for p in pnls) / (n - 1)
                std = math.sqrt(variance) if variance > 0 else 0.0
                sharpe = avg_pnl / std if std > 0 else 0.0
            else:
                sharpe = 0.0

            entry = WeeklyPerformanceEntry(
                week=week,
                setup_type=setup_type_val,
                trades=n,
                win_rate=round(win_rate, 4),
                avg_pnl=round(avg_pnl, 4),
                sharpe=round(sharpe, 4),
            )
            write_weekly_performance(entry, db_path)
            entries.append(entry)

        return entries
    finally:
        session.close()


# ── Learning Cycle Orchestrator ──────────────────────────────────────────────


def run_learning_cycle(
    loop_id: str,
    strategic_db_path: str,
    run_weekly: bool = False,
) -> dict[str, Any]:
    """
    High-level entry point called from ops.py after daily stats.

    1. Detect win-rate drift → return alert dict if triggered.
    2. If *run_weekly*, generate the previous-week report and calibration analysis.
    """
    results: dict[str, Any] = {}

    # 1. Drift detection (every cycle)
    try:
        drift = detect_win_rate_drift(strategic_db_path)
        if drift is not None:
            results["drift_alert"] = drift
            logger.warning(
                "learning[%s]: win-rate drift — %.1f%% over last %d trades",
                loop_id, drift["win_rate"] * 100, drift["window"],
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("learning[%s]: drift detection failed: %s", loop_id, exc)

    # 2. Weekly report + calibration (on trigger)
    if run_weekly:
        last_week = datetime.now(timezone.utc) - timedelta(days=7)
        year, wk, _ = last_week.isocalendar()
        week_str = f"{year}-W{wk:02d}"

        try:
            weekly = generate_weekly_report(week_str, strategic_db_path)
            results["weekly_report"] = {"week": week_str, "entries": len(weekly)}
        except Exception as exc:  # noqa: BLE001
            logger.error("learning[%s]: weekly report failed: %s", loop_id, exc)

        try:
            calibration = analyze_claude_calibration(strategic_db_path)
            results["calibration_report"] = calibration
        except Exception as exc:  # noqa: BLE001
            logger.error("learning[%s]: calibration analysis failed: %s", loop_id, exc)

    return results
