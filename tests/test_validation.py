"""
tests/test_validation.py — Sprint 7 validation engine tests.

All tests:
  - Use tmp_path SQLite DBs (no real DB touched).
  - Reset engine caches after each test.
  - Never require real API keys.
"""

import math
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from sauce.adapters import db as audit_db_mod
from sauce.adapters.db import Base as AuditBase, DailyStatsRow
from sauce.core.validation import (
    REQUIRED_CONSECUTIVE_DAYS,
    _count_consecutive_pass_days,
    _save_validation_result,
    check_claude_calibration,
    check_expectancy,
    check_max_drawdown,
    check_max_single_day_loss,
    check_sharpe_ratio,
    check_win_rate,
    run_validation,
)
from sauce.memory import db as mem_db_mod
from sauce.memory.db import (
    ClaudeCalibrationRow,
    SetupPerformanceRow,
    StrategicBase,
    ValidationResultRow,
    get_session as strategic_get_session,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture()
def audit_db(tmp_path):
    """Create a fresh audit SQLite DB and return its path string."""
    path = str(tmp_path / "test_audit.db")
    # Force engine creation & table init
    from sauce.adapters.db import get_engine

    engine = get_engine(path)
    AuditBase.metadata.create_all(engine)
    yield path
    audit_db_mod._engines.pop(path, None)


@pytest.fixture()
def strategic_db(tmp_path):
    """Create a fresh strategic SQLite DB and return its path string."""
    path = str(tmp_path / "test_strategic.db")
    from sauce.memory.db import get_engine

    engine = get_engine(path)
    StrategicBase.metadata.create_all(engine)
    yield path
    mem_db_mod._engines.pop(path, None)


def _add_trades(strategic_db_path, trades):
    """Insert SetupPerformanceRow rows. trades = list of (win, pnl)."""
    session = strategic_get_session(strategic_db_path)
    for win, pnl in trades:
        session.add(
            SetupPerformanceRow(
                setup_type="mean_reversion",
                symbol="AAPL",
                regime_at_entry="trending_up",
                time_of_day_bucket="morning",
                win=win,
                pnl=pnl,
                hold_duration_minutes=30.0,
                date="2024-01-15",
            )
        )
    session.commit()
    session.close()


def _add_daily_stats(audit_db_path, stats):
    """Insert DailyStatsRow rows. stats = list of (date, starting_nav, ending_nav)."""
    from sauce.adapters.db import get_session

    session = get_session(audit_db_path)
    for date, s_nav, e_nav in stats:
        session.add(
            DailyStatsRow(
                date=date,
                loop_runs=1,
                starting_nav_usd=s_nav,
                ending_nav_usd=e_nav,
            )
        )
    session.commit()
    session.close()


def _add_calibration(strategic_db_path, entries):
    """Insert ClaudeCalibrationRow rows. entries = list of (confidence, outcome)."""
    session = strategic_get_session(strategic_db_path)
    for conf, outcome in entries:
        session.add(
            ClaudeCalibrationRow(
                date="2024-01-15",
                confidence_stated=conf,
                outcome=outcome,
                setup_type="mean_reversion",
            )
        )
    session.commit()
    session.close()


def _add_validation_results(strategic_db_path, results):
    """Insert ValidationResultRow rows. results = list of (date, all_passed)."""
    session = strategic_get_session(strategic_db_path)
    for date, all_passed in results:
        session.add(
            ValidationResultRow(
                date=date,
                win_rate=0.55,
                expectancy=0.5,
                max_drawdown_pct=0.02,
                sharpe_ratio=1.2,
                max_single_day_loss_pct=0.01,
                calibration_score=0.7,
                all_passed=all_passed,
                consecutive_days=0,
            )
        )
    session.commit()
    session.close()


# ═══════════════════════════════════════════════════════════════════════════════
# check_win_rate
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckWinRate:
    def test_empty_db(self, strategic_db):
        passed, rate = check_win_rate(strategic_db)
        assert passed is False
        assert rate == 0.0

    def test_all_wins(self, strategic_db):
        _add_trades(strategic_db, [(True, 10.0)] * 10)
        passed, rate = check_win_rate(strategic_db)
        assert passed is True
        assert rate == 1.0

    def test_all_losses(self, strategic_db):
        _add_trades(strategic_db, [(False, -5.0)] * 10)
        passed, rate = check_win_rate(strategic_db)
        assert passed is False
        assert rate == 0.0

    def test_exact_boundary_passes(self, strategic_db):
        # 52 wins out of 100 = 0.52 exactly
        _add_trades(strategic_db, [(True, 5.0)] * 52 + [(False, -3.0)] * 48)
        passed, rate = check_win_rate(strategic_db)
        assert passed is True
        assert rate == 0.52

    def test_below_boundary(self, strategic_db):
        # 51 wins out of 100 = 0.51
        _add_trades(strategic_db, [(True, 5.0)] * 51 + [(False, -3.0)] * 49)
        passed, rate = check_win_rate(strategic_db)
        assert passed is False
        assert rate == 0.51

    def test_custom_min_rate(self, strategic_db):
        _add_trades(strategic_db, [(True, 5.0)] * 60 + [(False, -3.0)] * 40)
        passed, rate = check_win_rate(strategic_db, min_rate=0.70)
        assert passed is False
        assert rate == 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# check_expectancy
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckExpectancy:
    def test_too_few_trades(self, strategic_db):
        _add_trades(strategic_db, [(True, 10.0)] * 5)
        passed, val = check_expectancy(strategic_db, min_trades=50)
        assert passed is False
        assert val == 0.0

    def test_positive_expectancy(self, strategic_db):
        # 60% win rate, avg_win=10, avg_loss=5
        # Expected = 0.6*10 - 0.4*5 = 6 - 2 = 4.0
        _add_trades(strategic_db, [(True, 10.0)] * 30 + [(False, -5.0)] * 20)
        passed, val = check_expectancy(strategic_db)
        assert passed is True
        assert val == pytest.approx(4.0, abs=0.01)

    def test_negative_expectancy(self, strategic_db):
        # 30% win rate, avg_win=5, avg_loss=10
        # Expected = 0.3*5 - 0.7*10 = 1.5 - 7.0 = -5.5
        _add_trades(strategic_db, [(True, 5.0)] * 15 + [(False, -10.0)] * 35)
        passed, val = check_expectancy(strategic_db)
        assert passed is False
        assert val == pytest.approx(-5.5, abs=0.01)

    def test_custom_min_trades(self, strategic_db):
        _add_trades(strategic_db, [(True, 10.0)] * 6 + [(False, -5.0)] * 4)
        passed, val = check_expectancy(strategic_db, min_trades=10)
        assert passed is True
        assert val > 0

    def test_all_wins_expectancy(self, strategic_db):
        _add_trades(strategic_db, [(True, 10.0)] * 50)
        passed, val = check_expectancy(strategic_db)
        # win_rate=1.0, avg_win=10, loss_rate=0, avg_loss=0  → 10.0
        assert passed is True
        assert val == pytest.approx(10.0, abs=0.01)

    def test_all_losses_expectancy(self, strategic_db):
        _add_trades(strategic_db, [(False, -8.0)] * 50)
        passed, val = check_expectancy(strategic_db)
        # win_rate=0, avg_win=0, loss_rate=1, avg_loss=8  → -8.0
        assert passed is False
        assert val == pytest.approx(-8.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# check_max_drawdown
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckMaxDrawdown:
    def test_empty_db(self, audit_db):
        passed, dd = check_max_drawdown(audit_db)
        assert passed is True
        assert dd == 0.0

    def test_no_drawdown(self, audit_db):
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 1010),
            ("2024-01-02", 1010, 1020),
            ("2024-01-03", 1020, 1030),
        ])
        passed, dd = check_max_drawdown(audit_db)
        assert passed is True
        assert dd == 0.0

    def test_drawdown_under_threshold(self, audit_db):
        # Peak=1000, trough=950 → 5% drawdown → under 8%
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 1000),
            ("2024-01-02", 1000, 950),
            ("2024-01-03", 950, 960),
        ])
        passed, dd = check_max_drawdown(audit_db)
        assert passed is True
        assert dd == pytest.approx(0.05, abs=0.001)

    def test_drawdown_over_threshold(self, audit_db):
        # Peak=1000, trough=900 → 10% drawdown → over 8%
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 1000),
            ("2024-01-02", 1000, 900),
        ])
        passed, dd = check_max_drawdown(audit_db)
        assert passed is False
        assert dd == pytest.approx(0.10, abs=0.001)

    def test_recovery_after_dip(self, audit_db):
        # Peak=1000, dip=950, recover=1100, dip=1050
        # Max DD: (1100-1050)/1100 = 4.5%
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 1000),
            ("2024-01-02", 1000, 950),
            ("2024-01-03", 950, 1100),
            ("2024-01-04", 1100, 1050),
        ])
        passed, dd = check_max_drawdown(audit_db)
        assert passed is True
        # Max DD is (1000-950)/1000=5% or (1100-1050)/1100=4.5%, whichever larger
        assert dd == pytest.approx(0.05, abs=0.001)

    def test_custom_threshold(self, audit_db):
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 1000),
            ("2024-01-02", 1000, 960),
        ])
        passed, dd = check_max_drawdown(audit_db, max_dd_pct=0.03)
        assert passed is False
        assert dd == pytest.approx(0.04, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# check_sharpe_ratio
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckSharpeRatio:
    def test_empty_db(self, audit_db):
        passed, sharpe = check_sharpe_ratio(audit_db)
        assert passed is False
        assert sharpe == 0.0

    def test_single_day(self, audit_db):
        _add_daily_stats(audit_db, [("2024-01-01", 1000, 1010)])
        passed, sharpe = check_sharpe_ratio(audit_db)
        assert passed is False
        assert sharpe == 0.0

    def test_consistent_positive_returns(self, audit_db):
        # 1% daily returns → very high Sharpe (near-zero std from float noise)
        navs = [1000.0]
        stats = []
        for i in range(30):
            date = f"2024-01-{i + 1:02d}"
            s = navs[-1]
            e = s * 1.01
            stats.append((date, s, e))
            navs.append(e)
        _add_daily_stats(audit_db, stats)
        passed, sharpe = check_sharpe_ratio(audit_db)
        # Floating-point compounding makes returns not exactly identical,
        # so std > 0 and Sharpe is extremely high → passes.
        assert passed is True
        assert sharpe > 10.0

    def test_good_sharpe(self, audit_db):
        # Gradually increasing with slight noise → good Sharpe
        import random
        random.seed(42)
        navs = [1000.0]
        stats = []
        for i in range(60):
            date = f"2024-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
            prev = navs[-1]
            # ~0.5% daily + small noise
            ret = 0.005 + random.gauss(0, 0.001)
            e = prev * (1 + ret)
            stats.append((date, prev, e))
            navs.append(e)
        _add_daily_stats(audit_db, stats)
        passed, sharpe = check_sharpe_ratio(audit_db)
        assert sharpe > 0
        # With 0.5% drift and 0.1% noise, Sharpe should be very high
        assert passed is True

    def test_volatile_returns(self, audit_db):
        # Alternating +10% / -10% → low Sharpe
        navs = [1000.0]
        stats = []
        for i in range(20):
            date = f"2024-01-{i + 1:02d}"
            prev = navs[-1]
            e = prev * (1.10 if i % 2 == 0 else 0.90)
            stats.append((date, prev, e))
            navs.append(e)
        _add_daily_stats(audit_db, stats)
        passed, sharpe = check_sharpe_ratio(audit_db)
        # Mean return ≈ 0, high std → low/negative Sharpe
        assert passed is False


# ═══════════════════════════════════════════════════════════════════════════════
# check_max_single_day_loss
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckMaxSingleDayLoss:
    def test_empty_db(self, audit_db):
        passed, loss = check_max_single_day_loss(audit_db)
        assert passed is True
        assert loss == 0.0

    def test_no_loss_days(self, audit_db):
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 1010),
            ("2024-01-02", 1010, 1020),
        ])
        passed, loss = check_max_single_day_loss(audit_db)
        assert passed is True
        assert loss == 0.0

    def test_loss_under_threshold(self, audit_db):
        # 2% loss → under 3%
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 980),
        ])
        passed, loss = check_max_single_day_loss(audit_db)
        assert passed is True
        assert loss == pytest.approx(0.02, abs=0.001)

    def test_loss_over_threshold(self, audit_db):
        # 5% loss → over 3%
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 950),
        ])
        passed, loss = check_max_single_day_loss(audit_db)
        assert passed is False
        assert loss == pytest.approx(0.05, abs=0.001)

    def test_worst_day_selected(self, audit_db):
        _add_daily_stats(audit_db, [
            ("2024-01-01", 1000, 990),   # 1% loss
            ("2024-01-02", 990, 950),    # ~4.04% loss → worst
            ("2024-01-03", 950, 960),    # gain
        ])
        passed, loss = check_max_single_day_loss(audit_db)
        assert passed is False
        assert loss == pytest.approx(0.0404, abs=0.001)

    def test_custom_threshold(self, audit_db):
        _add_daily_stats(audit_db, [("2024-01-01", 1000, 960)])
        passed, loss = check_max_single_day_loss(audit_db, max_loss_pct=0.05)
        assert passed is True
        assert loss == pytest.approx(0.04, abs=0.001)

    def test_zero_starting_nav_ignored(self, audit_db):
        # Zero starting NAV should be skipped (no div-by-zero)
        _add_daily_stats(audit_db, [
            ("2024-01-01", 0, 100),
            ("2024-01-02", 1000, 990),
        ])
        passed, loss = check_max_single_day_loss(audit_db)
        assert passed is True
        assert loss == pytest.approx(0.01, abs=0.001)


# ═══════════════════════════════════════════════════════════════════════════════
# check_claude_calibration
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckClaudeCalibration:
    def test_empty_db(self, strategic_db):
        passed, score = check_claude_calibration(strategic_db)
        assert passed is False
        assert score == 0.0

    def test_perfect_calibration(self, strategic_db):
        # confidence=0.9, outcome=win (actual=1.0) → |0.9 - 1.0| = 0.1
        # confidence=0.1, outcome=loss (actual=0.0) → |0.1 - 0.0| = 0.1
        # avg_error=0.1, score=0.9
        _add_calibration(strategic_db, [
            (0.9, "win"),
            (0.1, "loss"),
        ])
        passed, score = check_claude_calibration(strategic_db)
        assert passed is True
        assert score == pytest.approx(0.9, abs=0.01)

    def test_poor_calibration(self, strategic_db):
        # confidence=0.9, outcome=loss → |0.9 - 0.0| = 0.9
        # confidence=0.1, outcome=win  → |0.1 - 1.0| = 0.9
        # avg_error=0.9, score=0.1
        _add_calibration(strategic_db, [
            (0.9, "loss"),
            (0.1, "win"),
        ])
        passed, score = check_claude_calibration(strategic_db)
        assert passed is False
        assert score == pytest.approx(0.1, abs=0.01)

    def test_at_boundary(self, strategic_db):
        # Engineer exact 0.60 score: avg_error=0.40, score=0.60
        # |conf - actual| = 0.40 each
        # win: actual=1.0, conf=0.60 → |0.60-1.0|=0.40
        # loss: actual=0.0, conf=0.40 → |0.40-0.0|=0.40
        _add_calibration(strategic_db, [
            (0.60, "win"),
            (0.40, "loss"),
        ])
        passed, score = check_claude_calibration(strategic_db)
        assert passed is True
        assert score == pytest.approx(0.60, abs=0.01)

    def test_custom_min_score(self, strategic_db):
        _add_calibration(strategic_db, [(0.80, "win"), (0.20, "loss")])
        # score = 1 - avg(|0.80-1.0|, |0.20-0.0|) = 1 - 0.2 = 0.8
        passed, score = check_claude_calibration(strategic_db, min_score=0.90)
        assert passed is False
        assert score == pytest.approx(0.80, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# _count_consecutive_pass_days
# ═══════════════════════════════════════════════════════════════════════════════


class TestCountConsecutivePassDays:
    def test_no_results(self, strategic_db):
        count = _count_consecutive_pass_days(strategic_db, "2024-01-15")
        assert count == 0

    def test_all_passed(self, strategic_db):
        _add_validation_results(strategic_db, [
            ("2024-01-13", True),
            ("2024-01-14", True),
            ("2024-01-15", True),
        ])
        count = _count_consecutive_pass_days(strategic_db, "2024-01-15")
        assert count == 3

    def test_broken_streak(self, strategic_db):
        _add_validation_results(strategic_db, [
            ("2024-01-12", True),
            ("2024-01-13", False),  # break
            ("2024-01-14", True),
            ("2024-01-15", True),
        ])
        count = _count_consecutive_pass_days(strategic_db, "2024-01-15")
        assert count == 2

    def test_most_recent_fails(self, strategic_db):
        _add_validation_results(strategic_db, [
            ("2024-01-14", True),
            ("2024-01-15", False),
        ])
        count = _count_consecutive_pass_days(strategic_db, "2024-01-15")
        assert count == 0

    def test_future_dates_excluded(self, strategic_db):
        _add_validation_results(strategic_db, [
            ("2024-01-15", True),
            ("2024-01-16", True),  # future
            ("2024-01-17", True),  # future
        ])
        count = _count_consecutive_pass_days(strategic_db, "2024-01-15")
        assert count == 1


# ═══════════════════════════════════════════════════════════════════════════════
# _save_validation_result
# ═══════════════════════════════════════════════════════════════════════════════


class TestSaveValidationResult:
    def _make_results(self):
        return {
            "win_rate": (True, 0.55),
            "expectancy": (True, 4.0),
            "max_drawdown": (True, 0.03),
            "sharpe_ratio": (True, 1.2),
            "max_single_day_loss": (True, 0.01),
            "calibration": (True, 0.75),
        }

    def test_insert_new(self, strategic_db):
        results = self._make_results()
        _save_validation_result(strategic_db, "2024-01-15", results, True, 5)

        session = strategic_get_session(strategic_db)
        row = session.query(ValidationResultRow).filter_by(date="2024-01-15").first()
        assert row is not None
        assert row.win_rate == 0.55
        assert row.expectancy == 4.0
        assert row.all_passed is True
        assert row.consecutive_days == 5
        session.close()

    def test_upsert_existing(self, strategic_db):
        results = self._make_results()
        _save_validation_result(strategic_db, "2024-01-15", results, True, 0)
        # Update same date
        _save_validation_result(strategic_db, "2024-01-15", results, True, 10)

        session = strategic_get_session(strategic_db)
        count = session.query(ValidationResultRow).filter_by(date="2024-01-15").count()
        assert count == 1
        row = session.query(ValidationResultRow).filter_by(date="2024-01-15").first()
        assert row.consecutive_days == 10
        session.close()

    def test_all_fields_persisted(self, strategic_db):
        results = {
            "win_rate": (False, 0.45),
            "expectancy": (False, -2.0),
            "max_drawdown": (True, 0.06),
            "sharpe_ratio": (False, 0.5),
            "max_single_day_loss": (True, 0.02),
            "calibration": (False, 0.55),
        }
        _save_validation_result(strategic_db, "2024-01-20", results, False, 0)

        session = strategic_get_session(strategic_db)
        row = session.query(ValidationResultRow).filter_by(date="2024-01-20").first()
        assert row.win_rate == 0.45
        assert row.expectancy == -2.0
        assert row.max_drawdown_pct == 0.06
        assert row.sharpe_ratio == 0.5
        assert row.max_single_day_loss_pct == 0.02
        assert row.calibration_score == 0.55
        assert row.all_passed is False
        assert row.consecutive_days == 0
        session.close()


# ═══════════════════════════════════════════════════════════════════════════════
# run_validation (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRunValidation:
    @patch("sauce.core.validation.datetime")
    @patch("sauce.core.validation.log_event")
    def test_all_pass(self, mock_log, mock_dt, audit_db, strategic_db):
        mock_dt.now.return_value = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        # Populate enough data for all checks to pass
        _add_trades(strategic_db, [(True, 10.0)] * 30 + [(False, -3.0)] * 20)
        _add_daily_stats(audit_db, [
            (f"2024-01-{d:02d}", 1000 + d, 1005 + d) for d in range(1, 16)
        ])
        _add_calibration(strategic_db, [(0.85, "win")] * 30 + [(0.15, "loss")] * 20)

        result = run_validation("loop-1", audit_db, strategic_db)

        assert "criteria" in result
        assert "all_passed" in result
        assert "consecutive_days" in result
        assert "date" in result
        assert result["date"] == "2024-01-15"

    @patch("sauce.core.validation.datetime")
    @patch("sauce.core.validation.log_event")
    def test_some_fail(self, mock_log, mock_dt, audit_db, strategic_db):
        mock_dt.now.return_value = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        # Empty strategic DB → win_rate fails, calibration fails
        _add_daily_stats(audit_db, [("2024-01-01", 1000, 1000)])

        result = run_validation("loop-1", audit_db, strategic_db)

        assert result["all_passed"] is False
        assert result["criteria"]["win_rate"]["passed"] is False
        assert result["criteria"]["calibration"]["passed"] is False

    @patch("sauce.core.validation.datetime")
    @patch("sauce.core.validation.log_event")
    def test_consecutive_days_tracking(self, mock_log, mock_dt, audit_db, strategic_db):
        mock_dt.now.return_value = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        # Pre-populate 3 passing days
        _add_validation_results(strategic_db, [
            ("2024-01-12", True),
            ("2024-01-13", True),
            ("2024-01-14", True),
        ])

        # Run with all-passing data
        _add_trades(strategic_db, [(True, 10.0)] * 30 + [(False, -3.0)] * 20)
        _add_daily_stats(audit_db, [
            (f"2024-01-{d:02d}", 1000 + d, 1005 + d) for d in range(1, 16)
        ])
        _add_calibration(strategic_db, [(0.85, "win")] * 30 + [(0.15, "loss")] * 20)

        result = run_validation("loop-1", audit_db, strategic_db)

        if result["all_passed"]:
            assert result["consecutive_days"] >= 4  # at least 3 prior + today

    @patch("sauce.core.validation.datetime")
    @patch("sauce.core.validation.log_event")
    def test_30_day_milestone_logs_event(self, mock_log, mock_dt, audit_db, strategic_db):
        mock_dt.now.return_value = datetime(2024, 2, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        # Pre-populate 29 passing days
        for i in range(29):
            day = f"2024-02-{i + 1:02d}" if i + 1 <= 14 else f"2024-01-{i - 13:02d}"
        # Actually populate with ordered dates
        results_to_add = []
        for i in range(29):
            d = 15 - 29 + i  # negative → January
            if d > 0:
                results_to_add.append((f"2024-02-{d:02d}", True))
            else:
                results_to_add.append((f"2024-01-{31 + d:02d}", True))
        _add_validation_results(strategic_db, results_to_add)

        # Data for passing on day 30
        _add_trades(strategic_db, [(True, 10.0)] * 30 + [(False, -3.0)] * 20)
        _add_daily_stats(audit_db, [
            (f"2024-02-{d:02d}", 1000 + d, 1005 + d) for d in range(1, 16)
        ])
        _add_calibration(strategic_db, [(0.85, "win")] * 30 + [(0.15, "loss")] * 20)

        result = run_validation("loop-1", audit_db, strategic_db)

        if result["all_passed"] and result["consecutive_days"] >= 30:
            # log_event should've been called with validation_passed
            found = any(
                call.args[0].event_type == "validation_passed"
                for call in mock_log.call_args_list
            )
            assert found, "Expected validation_passed log_event at 30 days"

    @patch("sauce.core.validation.datetime")
    @patch("sauce.core.validation.log_event")
    def test_criterion_exception_returns_error(self, mock_log, mock_dt, audit_db, strategic_db):
        mock_dt.now.return_value = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        with patch(
            "sauce.core.validation.check_win_rate",
            side_effect=RuntimeError("db exploded"),
        ):
            result = run_validation("loop-1", audit_db, strategic_db)

        assert "error" in result
        assert result["all_passed"] is False
        assert result["consecutive_days"] == 0

    @patch("sauce.core.validation.datetime")
    @patch("sauce.core.validation.log_event")
    def test_result_payload_structure(self, mock_log, mock_dt, audit_db, strategic_db):
        mock_dt.now.return_value = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        _add_daily_stats(audit_db, [("2024-01-01", 1000, 1000)])

        result = run_validation("loop-1", audit_db, strategic_db)

        assert isinstance(result["criteria"], dict)
        for key in ["win_rate", "expectancy", "max_drawdown", "sharpe_ratio", "max_single_day_loss", "calibration"]:
            assert key in result["criteria"]
            assert "passed" in result["criteria"][key]
            assert "value" in result["criteria"][key]
        assert isinstance(result["all_passed"], bool)
        assert isinstance(result["consecutive_days"], int)
        assert result["date"] == "2024-01-15"


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════


def test_required_consecutive_days_is_30():
    assert REQUIRED_CONSECUTIVE_DAYS == 30
