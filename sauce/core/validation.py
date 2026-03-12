"""
core/validation.py — Paper-trading validation engine (Sprint 7).

Evaluates 6 criteria daily. ALL 6 must pass for 30 consecutive days
before the system is cleared for live trading.

Criteria:
  1. Win rate > 52%
  2. Expectancy positive (min 50 trades per setup)
  3. Max drawdown < 8% of starting capital
  4. Sharpe ratio > 0.8 on daily returns
  5. Max single-day loss < 3%
  6. Claude calibration score > 0.60

Write helpers NEVER raise — on DB error, log critical + print to stderr.
"""

import logging
import math
import sys
from datetime import datetime, timezone

from sqlalchemy import func

from sauce.adapters.db import (
    DailyStatsRow,
    get_session as audit_get_session,
    log_event,
)
from sauce.core.schemas import AuditEvent
from sauce.memory.db import (
    ClaudeCalibrationRow,
    SetupPerformanceRow,
    ValidationResultRow,
    get_session as strategic_get_session,
)

logger = logging.getLogger(__name__)

REQUIRED_CONSECUTIVE_DAYS = 30


# ── Individual Criterion Checks ───────────────────────────────────────────────


def check_win_rate(
    strategic_db_path: str,
    min_rate: float = 0.52,
) -> tuple[bool, float]:
    """Return (passed, win_rate) across all trades in strategic memory."""
    session = strategic_get_session(strategic_db_path)
    try:
        total = session.query(func.count(SetupPerformanceRow.id)).scalar() or 0
        if total == 0:
            return False, 0.0
        wins = (
            session.query(func.count(SetupPerformanceRow.id))
            .filter(SetupPerformanceRow.win.is_(True))
            .scalar()
            or 0
        )
        rate = wins / total
        return rate >= min_rate, round(rate, 4)
    finally:
        session.close()


def check_expectancy(
    strategic_db_path: str,
    min_trades: int = 50,
) -> tuple[bool, float]:
    """Return (passed, expectancy_value).

    Expectancy = (win_rate × avg_win) - (loss_rate × avg_loss).
    Requires at least *min_trades* total trades to pass.
    """
    session = strategic_get_session(strategic_db_path)
    try:
        rows = session.query(SetupPerformanceRow.win, SetupPerformanceRow.pnl).all()
        total = len(rows)
        if total < min_trades:
            return False, 0.0

        wins = [r.pnl for r in rows if r.win]
        losses = [r.pnl for r in rows if not r.win]

        win_rate = len(wins) / total if total else 0.0
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = abs(sum(losses) / len(losses)) if losses else 0.0

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        return expectancy > 0, round(expectancy, 4)
    finally:
        session.close()


def check_max_drawdown(
    audit_db_path: str,
    max_dd_pct: float = 0.08,
) -> tuple[bool, float]:
    """Return (passed, max_drawdown_pct) from daily NAV series.

    Drawdown = (peak - trough) / peak.
    """
    session = audit_get_session(audit_db_path)
    try:
        rows = (
            session.query(DailyStatsRow.ending_nav_usd)
            .order_by(DailyStatsRow.date)
            .all()
        )
        if not rows:
            return True, 0.0

        navs = [r.ending_nav_usd for r in rows]
        peak = navs[0]
        max_dd = 0.0
        for nav in navs:
            if nav > peak:
                peak = nav
            dd = (peak - nav) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        return max_dd <= max_dd_pct, round(max_dd, 6)
    finally:
        session.close()


def check_sharpe_ratio(
    audit_db_path: str,
    min_sharpe: float = 0.8,
) -> tuple[bool, float]:
    """Return (passed, sharpe_ratio) from daily NAV returns.

    Annualised Sharpe = mean(daily_returns) / std(daily_returns) * sqrt(252).
    Uses n-1 variance (sample std). Requires ≥ 2 days.
    """
    session = audit_get_session(audit_db_path)
    try:
        rows = (
            session.query(DailyStatsRow.ending_nav_usd)
            .order_by(DailyStatsRow.date)
            .all()
        )
        navs = [r.ending_nav_usd for r in rows]
        if len(navs) < 2:
            return False, 0.0

        returns = []
        for i in range(1, len(navs)):
            if navs[i - 1] > 0:
                returns.append((navs[i] - navs[i - 1]) / navs[i - 1])

        if not returns:
            return False, 0.0

        mean_r = sum(returns) / len(returns)
        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0

        if std_r == 0:
            return False, 0.0

        sharpe = (mean_r / std_r) * math.sqrt(252)
        return sharpe >= min_sharpe, round(sharpe, 4)
    finally:
        session.close()


def check_max_single_day_loss(
    audit_db_path: str,
    max_loss_pct: float = 0.03,
) -> tuple[bool, float]:
    """Return (passed, worst_day_loss_pct).

    Loss = (starting_nav - ending_nav) / starting_nav for the worst day.
    """
    session = audit_get_session(audit_db_path)
    try:
        rows = (
            session.query(
                DailyStatsRow.starting_nav_usd,
                DailyStatsRow.ending_nav_usd,
            )
            .all()
        )
        if not rows:
            return True, 0.0

        worst = 0.0
        for r in rows:
            if r.starting_nav_usd > 0:
                loss = (r.starting_nav_usd - r.ending_nav_usd) / r.starting_nav_usd
                if loss > worst:
                    worst = loss

        return worst <= max_loss_pct, round(worst, 6)
    finally:
        session.close()


def check_claude_calibration(
    strategic_db_path: str,
    min_score: float = 0.60,
) -> tuple[bool, float]:
    """Return (passed, calibration_score).

    Score = 1 - mean(|confidence_stated - actual_outcome|).
    Outcome is mapped: 'win' → 1.0, 'loss' → 0.0.
    """
    session = strategic_get_session(strategic_db_path)
    try:
        rows = session.query(
            ClaudeCalibrationRow.confidence_stated,
            ClaudeCalibrationRow.outcome,
        ).all()

        if not rows:
            return False, 0.0

        errors = []
        for r in rows:
            actual = 1.0 if r.outcome == "win" else 0.0
            errors.append(abs(r.confidence_stated - actual))

        avg_error = sum(errors) / len(errors)
        score = 1.0 - avg_error
        return score >= min_score, round(score, 4)
    finally:
        session.close()


# ── Consecutive Days Tracker ──────────────────────────────────────────────────


def _count_consecutive_pass_days(
    strategic_db_path: str,
    today: str,
) -> int:
    """Count contiguous days ending at *today* where all_passed=True."""
    session = strategic_get_session(strategic_db_path)
    try:
        rows = (
            session.query(ValidationResultRow.date, ValidationResultRow.all_passed)
            .filter(ValidationResultRow.date <= today)
            .order_by(ValidationResultRow.date.desc())
            .all()
        )
        count = 0
        for r in rows:
            if r.all_passed:
                count += 1
            else:
                break
        return count
    finally:
        session.close()


# ── Persist Result ────────────────────────────────────────────────────────────


def _save_validation_result(
    strategic_db_path: str,
    date: str,
    results: dict,
    all_passed: bool,
    consecutive_days: int,
) -> None:
    """Upsert today's validation result into strategic memory."""
    session = strategic_get_session(strategic_db_path)
    try:
        existing = (
            session.query(ValidationResultRow)
            .filter_by(date=date)
            .first()
        )
        if existing is not None:
            existing.win_rate = results["win_rate"][1]
            existing.expectancy = results["expectancy"][1]
            existing.max_drawdown_pct = results["max_drawdown"][1]
            existing.sharpe_ratio = results["sharpe_ratio"][1]
            existing.max_single_day_loss_pct = results["max_single_day_loss"][1]
            existing.calibration_score = results["calibration"][1]
            existing.all_passed = all_passed
            existing.consecutive_days = consecutive_days
        else:
            row = ValidationResultRow(
                date=date,
                win_rate=results["win_rate"][1],
                expectancy=results["expectancy"][1],
                max_drawdown_pct=results["max_drawdown"][1],
                sharpe_ratio=results["sharpe_ratio"][1],
                max_single_day_loss_pct=results["max_single_day_loss"][1],
                calibration_score=results["calibration"][1],
                all_passed=all_passed,
                consecutive_days=consecutive_days,
            )
            session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.critical("Failed to save validation result: %s", exc)
        print(f"[validation] CRITICAL: save failed: {exc}", file=sys.stderr)
    finally:
        session.close()


# ── Orchestrator ──────────────────────────────────────────────────────────────


def run_validation(
    loop_id: str,
    db_path: str,
    strategic_db_path: str,
) -> dict:
    """Run all 6 validation criteria and persist the result.

    Returns a dict with per-criterion results, all_passed, and
    consecutive_days count.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    results: dict[str, tuple[bool, float]] = {}
    try:
        results["win_rate"] = check_win_rate(strategic_db_path)
        results["expectancy"] = check_expectancy(strategic_db_path)
        results["max_drawdown"] = check_max_drawdown(db_path)
        results["sharpe_ratio"] = check_sharpe_ratio(db_path)
        results["max_single_day_loss"] = check_max_single_day_loss(db_path)
        results["calibration"] = check_claude_calibration(strategic_db_path)
    except Exception as exc:  # noqa: BLE001
        logger.critical("Validation criteria evaluation failed: %s", exc)
        print(f"[validation] CRITICAL: criteria eval failed: {exc}", file=sys.stderr)
        return {"error": str(exc), "all_passed": False, "consecutive_days": 0}

    all_passed = all(passed for passed, _ in results.values())

    # Persist first so the consecutive-day query includes today.
    _save_validation_result(
        strategic_db_path=strategic_db_path,
        date=today,
        results=results,
        all_passed=all_passed,
        consecutive_days=0,  # placeholder, updated below
    )

    consecutive_days = _count_consecutive_pass_days(strategic_db_path, today)

    # Update with correct count.
    _save_validation_result(
        strategic_db_path=strategic_db_path,
        date=today,
        results=results,
        all_passed=all_passed,
        consecutive_days=consecutive_days,
    )

    # ── Degradation guard ─────────────────────────────────────────────────
    if consecutive_days >= REQUIRED_CONSECUTIVE_DAYS:
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="validation_passed",
                payload={
                    "agent": "validation",
                    "consecutive_days": consecutive_days,
                    "message": "30-day validation PASSED — system cleared for live trading.",
                },
            ),
            db_path=db_path,
        )

    payload = {
        "criteria": {k: {"passed": v[0], "value": v[1]} for k, v in results.items()},
        "all_passed": all_passed,
        "consecutive_days": consecutive_days,
        "date": today,
    }
    return payload
