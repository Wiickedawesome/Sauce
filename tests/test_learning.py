"""
tests/test_learning.py — Sprint 6 Learning Loop tests.

50 mock trades with known outcomes, plus targeted unit tests for each
learning function.  Verifies win-rate computation, drift detection,
calibration scoring, symbol behavior updates, and weekly report generation.
"""

import math
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from sauce.memory import db as memory_db
from sauce.memory.db import (
    ClaudeCalibrationRow,
    SetupPerformanceRow,
    StrategicBase,
    SymbolLearnedBehaviorRow,
    WeeklyPerformanceRow,
    get_engine,
    get_session,
    write_claude_calibration,
)
from sauce.memory.learning import (
    analyze_claude_calibration,
    detect_win_rate_drift,
    generate_weekly_report,
    record_trade_outcome,
    run_learning_cycle,
    update_symbol_learned_behavior,
)
from sauce.core.schemas import (
    ClaudeCalibrationEntry,
    SetupPerformanceEntry,
    SymbolLearnedBehaviorEntry,
    WeeklyPerformanceEntry,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_strategic_db(tmp_path):
    """Fresh strategic memory DB for each test."""
    db = str(tmp_path / "test_strategic.db")
    memory_db._engines = {}
    memory_db._last_reset_date = None
    engine = get_engine(db)
    StrategicBase.metadata.create_all(engine)
    yield db
    memory_db._engines = {}
    memory_db._last_reset_date = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_perf_entry(
    *,
    setup_type="crypto_mean_reversion",
    symbol="BTC-USD",
    regime="RANGING",
    bucket="09:30-12:00",
    win=True,
    pnl=10.0,
    hold_mins=30.0,
    date="2025-06-01",
):
    return SetupPerformanceEntry(
        setup_type=setup_type,
        symbol=symbol,
        regime_at_entry=regime,
        time_of_day_bucket=bucket,
        win=win,
        pnl=pnl,
        hold_duration_minutes=hold_mins,
        date=date,
    )


def _make_cal_entry(
    *,
    date="2025-06-01",
    confidence=0.75,
    outcome="win",
    setup_type="crypto_mean_reversion",
):
    return ClaudeCalibrationEntry(
        date=date,
        confidence_stated=confidence,
        outcome=outcome,
        setup_type=setup_type,
    )


# ── record_trade_outcome ────────────────────────────────────────────────────


class TestRecordTradeOutcome:
    def test_writes_performance_entry(self, tmp_strategic_db):
        entry = _make_perf_entry()
        record_trade_outcome(entry, tmp_strategic_db)

        session = get_session(tmp_strategic_db)
        try:
            rows = session.query(SetupPerformanceRow).all()
            assert len(rows) == 1
            assert rows[0].symbol == "BTC-USD"
            assert rows[0].win is True
            assert rows[0].pnl == 10.0
        finally:
            session.close()

    def test_writes_with_calibration(self, tmp_strategic_db):
        perf = _make_perf_entry()
        cal = _make_cal_entry()
        record_trade_outcome(perf, tmp_strategic_db, calibration_entry=cal)

        session = get_session(tmp_strategic_db)
        try:
            assert session.query(SetupPerformanceRow).count() == 1
            cal_rows = session.query(ClaudeCalibrationRow).all()
            assert len(cal_rows) == 1
            assert cal_rows[0].confidence_stated == 0.75
            assert cal_rows[0].outcome == "win"
        finally:
            session.close()

    def test_writes_without_calibration(self, tmp_strategic_db):
        record_trade_outcome(_make_perf_entry(), tmp_strategic_db)

        session = get_session(tmp_strategic_db)
        try:
            assert session.query(SetupPerformanceRow).count() == 1
            assert session.query(ClaudeCalibrationRow).count() == 0
        finally:
            session.close()

    def test_multiple_trades(self, tmp_strategic_db):
        for i in range(5):
            record_trade_outcome(
                _make_perf_entry(pnl=float(i), win=i % 2 == 0),
                tmp_strategic_db,
            )

        session = get_session(tmp_strategic_db)
        try:
            assert session.query(SetupPerformanceRow).count() == 5
        finally:
            session.close()

    def test_different_setup_types(self, tmp_strategic_db):
        for st in ("crypto_mean_reversion", "equity_trend_pullback", "crypto_breakout"):
            record_trade_outcome(_make_perf_entry(setup_type=st), tmp_strategic_db)

        session = get_session(tmp_strategic_db)
        try:
            types = {r.setup_type for r in session.query(SetupPerformanceRow).all()}
            assert types == {"crypto_mean_reversion", "equity_trend_pullback", "crypto_breakout"}
        finally:
            session.close()


# ── detect_win_rate_drift ────────────────────────────────────────────────────


class TestWinRateDrift:
    def test_no_drift_when_above_threshold(self, tmp_strategic_db):
        # 12/20 = 60% > 45%
        for i in range(20):
            record_trade_outcome(
                _make_perf_entry(win=i < 12, pnl=10.0 if i < 12 else -5.0),
                tmp_strategic_db,
            )
        assert detect_win_rate_drift(tmp_strategic_db) is None

    def test_drift_detected_below_threshold(self, tmp_strategic_db):
        # 8/20 = 40% < 45%
        for i in range(20):
            record_trade_outcome(
                _make_perf_entry(win=i < 8, pnl=10.0 if i < 8 else -5.0),
                tmp_strategic_db,
            )
        result = detect_win_rate_drift(tmp_strategic_db)
        assert result is not None
        assert result["win_rate"] == 0.4
        assert result["wins"] == 8
        assert result["losses"] == 12

    def test_drift_at_exact_threshold(self, tmp_strategic_db):
        # 9/20 = 45% — exactly at threshold → no drift (< not <=)
        for i in range(20):
            record_trade_outcome(
                _make_perf_entry(win=i < 9, pnl=10.0 if i < 9 else -5.0),
                tmp_strategic_db,
            )
        assert detect_win_rate_drift(tmp_strategic_db) is None

    def test_not_enough_data(self, tmp_strategic_db):
        for i in range(5):
            record_trade_outcome(_make_perf_entry(win=False, pnl=-5.0), tmp_strategic_db)
        assert detect_win_rate_drift(tmp_strategic_db) is None

    def test_empty_db(self, tmp_strategic_db):
        assert detect_win_rate_drift(tmp_strategic_db) is None

    def test_custom_window_and_threshold(self, tmp_strategic_db):
        # 3/10 = 30% < 50%
        for i in range(10):
            record_trade_outcome(
                _make_perf_entry(win=i < 3, pnl=10.0 if i < 3 else -5.0),
                tmp_strategic_db,
            )
        result = detect_win_rate_drift(tmp_strategic_db, window=10, threshold=0.50)
        assert result is not None
        assert result["win_rate"] == 0.3

    def test_uses_most_recent_trades(self, tmp_strategic_db):
        # First 10 = all losses, last 20 = 15 wins → last-20 = 75% → no drift
        for _ in range(10):
            record_trade_outcome(_make_perf_entry(win=False, pnl=-5.0), tmp_strategic_db)
        for i in range(20):
            record_trade_outcome(
                _make_perf_entry(win=i < 15, pnl=10.0 if i < 15 else -5.0),
                tmp_strategic_db,
            )
        assert detect_win_rate_drift(tmp_strategic_db) is None

    def test_all_wins_no_drift(self, tmp_strategic_db):
        for _ in range(20):
            record_trade_outcome(_make_perf_entry(win=True, pnl=10.0), tmp_strategic_db)
        assert detect_win_rate_drift(tmp_strategic_db) is None


# ── analyze_claude_calibration ───────────────────────────────────────────────


class TestClaudeCalibration:
    def test_empty_db(self, tmp_strategic_db):
        result = analyze_claude_calibration(tmp_strategic_db)
        assert result == {"buckets": {}, "total_entries": 0}

    def test_single_bucket(self, tmp_strategic_db):
        # 3 trades at 0.75: 2 wins, 1 loss → 66.7%
        for i in range(3):
            write_claude_calibration(
                _make_cal_entry(confidence=0.75, outcome="win" if i < 2 else "loss"),
                tmp_strategic_db,
            )
        result = analyze_claude_calibration(tmp_strategic_db, min_bucket_size=1)
        assert result["total_entries"] == 3
        bucket = result["buckets"]["0.70-0.80"]
        assert bucket["count"] == 3
        assert abs(bucket["actual_win_rate"] - 0.6667) < 0.001

    def test_multiple_buckets(self, tmp_strategic_db):
        # High: 4 at 0.85, 3 wins → 75%
        for i in range(4):
            write_claude_calibration(
                _make_cal_entry(confidence=0.85, outcome="win" if i < 3 else "loss"),
                tmp_strategic_db,
            )
        # Low: 6 at 0.55, 2 wins → 33.3%
        for i in range(6):
            write_claude_calibration(
                _make_cal_entry(confidence=0.55, outcome="win" if i < 2 else "loss"),
                tmp_strategic_db,
            )
        result = analyze_claude_calibration(tmp_strategic_db, min_bucket_size=1)
        assert result["total_entries"] == 10
        assert result["buckets"]["0.80-0.90"]["actual_win_rate"] == 0.75
        assert abs(result["buckets"]["0.50-0.60"]["actual_win_rate"] - 0.3333) < 0.001

    def test_expected_midpoint(self, tmp_strategic_db):
        write_claude_calibration(
            _make_cal_entry(confidence=0.95, outcome="win"), tmp_strategic_db
        )
        result = analyze_claude_calibration(tmp_strategic_db)
        assert result["buckets"]["0.90-1.00"]["expected_midpoint"] == 0.95

    def test_confidence_at_1_point_0(self, tmp_strategic_db):
        write_claude_calibration(
            _make_cal_entry(confidence=1.0, outcome="win"), tmp_strategic_db
        )
        result = analyze_claude_calibration(tmp_strategic_db, min_bucket_size=1)
        assert "0.90-1.00" in result["buckets"]

    def test_min_bucket_size_guard(self, tmp_strategic_db):
        """Buckets with fewer than min_bucket_size entries report insufficient_data (IMP-31)."""
        for i in range(3):
            write_claude_calibration(
                _make_cal_entry(confidence=0.75, outcome="win" if i < 2 else "loss"),
                tmp_strategic_db,
            )
        # Default min_bucket_size=5, so 3 entries should be insufficient
        result = analyze_claude_calibration(tmp_strategic_db)
        bucket = result["buckets"]["0.70-0.80"]
        assert bucket["count"] == 3
        assert bucket["insufficient_data"] is True
        assert "actual_win_rate" not in bucket


# ── update_symbol_learned_behavior ───────────────────────────────────────────


class TestSymbolLearnedBehavior:
    def test_not_enough_data(self, tmp_strategic_db):
        for _ in range(2):
            record_trade_outcome(_make_perf_entry(symbol="BTC-USD"), tmp_strategic_db)
        result = update_symbol_learned_behavior(
            "BTC-USD", "crypto_mean_reversion", tmp_strategic_db
        )
        assert result is None

    def test_computes_behavior(self, tmp_strategic_db):
        # 3 wins at +20, +30, +40;  2 losses at -10, -15
        for pnl in [20.0, 30.0, 40.0]:
            record_trade_outcome(
                _make_perf_entry(symbol="ETH-USD", win=True, pnl=pnl), tmp_strategic_db
            )
        for pnl in [-10.0, -15.0]:
            record_trade_outcome(
                _make_perf_entry(symbol="ETH-USD", win=False, pnl=pnl), tmp_strategic_db
            )

        result = update_symbol_learned_behavior(
            "ETH-USD", "crypto_mean_reversion", tmp_strategic_db
        )
        assert result is not None
        assert result.sample_size == 5
        assert result.avg_bounce_magnitude == 30.0  # (20+30+40)/3
        assert result.avg_reversion_depth == 12.5  # (10+15)/2

    def test_writes_to_db(self, tmp_strategic_db):
        for _ in range(5):
            record_trade_outcome(
                _make_perf_entry(symbol="SOL-USD", win=True, pnl=15.0), tmp_strategic_db
            )
        update_symbol_learned_behavior("SOL-USD", "crypto_mean_reversion", tmp_strategic_db)

        session = get_session(tmp_strategic_db)
        try:
            rows = session.query(SymbolLearnedBehaviorRow).all()
            assert len(rows) == 1
            assert rows[0].symbol == "SOL-USD"
            assert rows[0].sample_size == 5
        finally:
            session.close()

    def test_filters_by_symbol_and_setup(self, tmp_strategic_db):
        for _ in range(5):
            record_trade_outcome(
                _make_perf_entry(symbol="BTC-USD", win=True, pnl=10.0), tmp_strategic_db
            )
        for _ in range(3):
            record_trade_outcome(
                _make_perf_entry(symbol="ETH-USD", win=True, pnl=20.0), tmp_strategic_db
            )

        btc = update_symbol_learned_behavior(
            "BTC-USD", "crypto_mean_reversion", tmp_strategic_db
        )
        eth = update_symbol_learned_behavior(
            "ETH-USD", "crypto_mean_reversion", tmp_strategic_db
        )
        assert btc is not None and btc.sample_size == 5
        assert eth is not None and eth.sample_size == 3

    def test_no_losses_leaves_reversion_none(self, tmp_strategic_db):
        for _ in range(3):
            record_trade_outcome(
                _make_perf_entry(symbol="BTC-USD", win=True, pnl=10.0), tmp_strategic_db
            )
        result = update_symbol_learned_behavior(
            "BTC-USD", "crypto_mean_reversion", tmp_strategic_db
        )
        assert result is not None
        assert result.avg_reversion_depth is None
        assert result.avg_bounce_magnitude == 10.0

    def test_no_wins_leaves_bounce_none(self, tmp_strategic_db):
        for _ in range(3):
            record_trade_outcome(
                _make_perf_entry(symbol="BTC-USD", win=False, pnl=-10.0), tmp_strategic_db
            )
        result = update_symbol_learned_behavior(
            "BTC-USD", "crypto_mean_reversion", tmp_strategic_db
        )
        assert result is not None
        assert result.avg_bounce_magnitude is None
        assert result.avg_reversion_depth == 10.0


# ── generate_weekly_report ───────────────────────────────────────────────────


class TestWeeklyReport:
    def test_empty_week(self, tmp_strategic_db):
        assert generate_weekly_report("2025-W25", tmp_strategic_db) == []

    def test_single_setup_type(self, tmp_strategic_db):
        # 5 trades in W01: 3 wins(+10), 2 losses(-5) → avg_pnl = 4.0
        for i in range(5):
            record_trade_outcome(
                _make_perf_entry(
                    setup_type="crypto_mean_reversion",
                    win=i < 3,
                    pnl=10.0 if i < 3 else -5.0,
                    date="2025-01-02",
                ),
                tmp_strategic_db,
            )
        result = generate_weekly_report("2025-W01", tmp_strategic_db)
        assert len(result) == 1
        assert result[0].trades == 5
        assert result[0].win_rate == 0.6
        assert result[0].avg_pnl == 4.0

    def test_multiple_setup_types(self, tmp_strategic_db):
        for _ in range(3):
            record_trade_outcome(
                _make_perf_entry(
                    setup_type="crypto_mean_reversion", win=True, pnl=10.0, date="2025-01-02"
                ),
                tmp_strategic_db,
            )
        for _ in range(2):
            record_trade_outcome(
                _make_perf_entry(
                    setup_type="equity_trend_pullback", win=False, pnl=-5.0, date="2025-01-03"
                ),
                tmp_strategic_db,
            )
        result = generate_weekly_report("2025-W01", tmp_strategic_db)
        assert len(result) == 2
        types = {e.setup_type for e in result}
        assert types == {"crypto_mean_reversion", "equity_trend_pullback"}

    def test_excludes_other_weeks(self, tmp_strategic_db):
        record_trade_outcome(
            _make_perf_entry(date="2025-01-02", win=True, pnl=10.0), tmp_strategic_db
        )
        record_trade_outcome(
            _make_perf_entry(date="2025-01-09", win=False, pnl=-5.0), tmp_strategic_db
        )
        result = generate_weekly_report("2025-W01", tmp_strategic_db)
        assert len(result) == 1
        assert result[0].trades == 1

    def test_sharpe_computation(self, tmp_strategic_db):
        # PnLs: [10, -5, 10, -5] → mean=2.5, sample_var=75, std≈8.66, sharpe≈0.2887
        for pnl in [10.0, -5.0, 10.0, -5.0]:
            record_trade_outcome(
                _make_perf_entry(win=pnl > 0, pnl=pnl, date="2025-01-02"),
                tmp_strategic_db,
            )
        result = generate_weekly_report("2025-W01", tmp_strategic_db)
        expected_sharpe = 2.5 / math.sqrt(75)
        assert abs(result[0].sharpe - round(expected_sharpe, 4)) < 0.001

    def test_writes_to_db(self, tmp_strategic_db):
        for _ in range(3):
            record_trade_outcome(
                _make_perf_entry(date="2025-01-02", win=True, pnl=10.0), tmp_strategic_db
            )
        generate_weekly_report("2025-W01", tmp_strategic_db)

        session = get_session(tmp_strategic_db)
        try:
            rows = session.query(WeeklyPerformanceRow).all()
            assert len(rows) == 1
            assert rows[0].week == "2025-W01"
        finally:
            session.close()

    def test_single_trade_sharpe_zero(self, tmp_strategic_db):
        record_trade_outcome(
            _make_perf_entry(win=True, pnl=10.0, date="2025-01-02"), tmp_strategic_db
        )
        result = generate_weekly_report("2025-W01", tmp_strategic_db)
        assert result[0].sharpe == 0.0

    def test_idempotent_rerun(self, tmp_strategic_db):
        for _ in range(3):
            record_trade_outcome(
                _make_perf_entry(date="2025-01-02", win=True, pnl=10.0), tmp_strategic_db
            )
        generate_weekly_report("2025-W01", tmp_strategic_db)
        generate_weekly_report("2025-W01", tmp_strategic_db)  # re-run: upsert, not dup

        session = get_session(tmp_strategic_db)
        try:
            assert session.query(WeeklyPerformanceRow).count() == 1
        finally:
            session.close()


# ── run_learning_cycle ───────────────────────────────────────────────────────


class TestRunLearningCycle:
    def test_returns_empty_when_no_data(self, tmp_strategic_db):
        result = run_learning_cycle("test-loop", tmp_strategic_db)
        assert result == {}

    def test_detects_drift(self, tmp_strategic_db):
        # 7/20 = 35% < 45%
        for i in range(20):
            record_trade_outcome(
                _make_perf_entry(win=i < 7, pnl=10.0 if i < 7 else -5.0),
                tmp_strategic_db,
            )
        result = run_learning_cycle("test-loop", tmp_strategic_db)
        assert "drift_alert" in result
        assert result["drift_alert"]["win_rate"] == 0.35

    def test_no_drift(self, tmp_strategic_db):
        for i in range(20):
            record_trade_outcome(
                _make_perf_entry(win=i < 15, pnl=10.0 if i < 15 else -5.0),
                tmp_strategic_db,
            )
        result = run_learning_cycle("test-loop", tmp_strategic_db)
        assert "drift_alert" not in result

    def test_weekly_report_when_flagged(self, tmp_strategic_db):
        for _ in range(5):
            record_trade_outcome(
                _make_perf_entry(date="2025-01-02", win=True, pnl=10.0), tmp_strategic_db
            )
        # Patch datetime.now → Jan 10 so last_week lands in W01
        fake_now = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        with patch("sauce.memory.learning.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = run_learning_cycle("test-loop", tmp_strategic_db, run_weekly=True)
        assert "weekly_report" in result
        assert result["weekly_report"]["week"] == "2025-W01"

    def test_calibration_when_weekly(self, tmp_strategic_db):
        write_claude_calibration(
            _make_cal_entry(confidence=0.80, outcome="win"), tmp_strategic_db
        )
        fake_now = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        with patch("sauce.memory.learning.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = run_learning_cycle("test-loop", tmp_strategic_db, run_weekly=True)
        assert "calibration_report" in result
        assert result["calibration_report"]["total_entries"] == 1

    def test_weekly_not_generated_without_flag(self, tmp_strategic_db):
        for _ in range(5):
            record_trade_outcome(
                _make_perf_entry(date="2025-01-02", win=True, pnl=10.0), tmp_strategic_db
            )
        result = run_learning_cycle("test-loop", tmp_strategic_db)
        assert "weekly_report" not in result
        assert "calibration_report" not in result

    def test_drift_and_weekly_together(self, tmp_strategic_db):
        # 5/20 = 25% drift + calibration data in W01
        for i in range(20):
            record_trade_outcome(
                _make_perf_entry(
                    win=i < 5, pnl=10.0 if i < 5 else -5.0, date="2025-01-02"
                ),
                tmp_strategic_db,
                calibration_entry=_make_cal_entry(
                    confidence=0.75, outcome="win" if i < 5 else "loss", date="2025-01-02"
                ),
            )
        fake_now = datetime(2025, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
        with patch("sauce.memory.learning.datetime") as mock_dt:
            mock_dt.now.return_value = fake_now
            result = run_learning_cycle("test-loop", tmp_strategic_db, run_weekly=True)
        assert "drift_alert" in result
        assert "weekly_report" in result
        assert "calibration_report" in result


# ── 50 Mock Trades Integration Test ─────────────────────────────────────────


class TestFiftyMockTrades:
    """
    Write 50 mock trades with known outcomes, verify win rates and
    calibration scores compute correctly.

    Distribution:
      crypto_mean_reversion / BTC-USD:  15 trades (10W 5L)
      crypto_mean_reversion / ETH-USD:  15 trades (10W 5L)
      equity_trend_pullback / AAPL:     15 trades ( 6W 9L)
      crypto_breakout       / SOL-USD:   5 trades ( 4W 1L)
      ──────────────────────────────────────────────────────
      Total:  50 trades, 30 wins → 60 %

    All trades on 2025-06-02 (Monday of ISO W23).
    """

    @pytest.fixture
    def loaded_db(self, tmp_strategic_db):
        """Write 50 trades with known distributions and return the db path."""
        trades = self._generate_fifty_trades()
        for perf, cal in trades:
            record_trade_outcome(perf, tmp_strategic_db, calibration_entry=cal)
        return tmp_strategic_db

    @staticmethod
    def _generate_fifty_trades():
        """
        Returns 50 (perf_entry, cal_entry) tuples.

        Calibration distribution:
            0.90 confidence: 10 entries (10W  0L) → 100 %
            0.75 confidence: 10 entries (10W  0L) → 100 %
            0.65 confidence: 13 entries (10W  3L) →  76.9 %
            0.55 confidence: 17 entries ( 0W 17L) →   0 %
        """
        trades = []
        date_base = "2025-06-02"  # Monday of ISO W23

        # crypto_mean_reversion / BTC-USD: 10W 5L
        for i in range(15):
            win = i < 10
            pnl = 25.0 if win else -15.0
            conf = 0.90 if i < 5 else (0.75 if i < 10 else 0.55)
            trades.append((
                _make_perf_entry(
                    setup_type="crypto_mean_reversion", symbol="BTC-USD",
                    win=win, pnl=pnl, date=date_base,
                ),
                _make_cal_entry(
                    confidence=conf, outcome="win" if win else "loss",
                    setup_type="crypto_mean_reversion", date=date_base,
                ),
            ))

        # crypto_mean_reversion / ETH-USD: 10W 5L
        for i in range(15):
            win = i < 10
            pnl = 20.0 if win else -12.0
            conf = 0.75 if i < 5 else (0.65 if i < 10 else 0.55)
            trades.append((
                _make_perf_entry(
                    setup_type="crypto_mean_reversion", symbol="ETH-USD",
                    win=win, pnl=pnl, date=date_base,
                ),
                _make_cal_entry(
                    confidence=conf, outcome="win" if win else "loss",
                    setup_type="crypto_mean_reversion", date=date_base,
                ),
            ))

        # equity_trend_pullback / AAPL: 6W 9L
        for i in range(15):
            win = i < 6
            pnl = 30.0 if win else -10.0
            conf = 0.90 if i < 3 else (0.65 if i < 8 else 0.55)
            trades.append((
                _make_perf_entry(
                    setup_type="equity_trend_pullback", symbol="AAPL",
                    win=win, pnl=pnl, date=date_base,
                ),
                _make_cal_entry(
                    confidence=conf, outcome="win" if win else "loss",
                    setup_type="equity_trend_pullback", date=date_base,
                ),
            ))

        # crypto_breakout / SOL-USD: 4W 1L
        for i in range(5):
            win = i < 4
            pnl = 50.0 if win else -20.0
            conf = 0.90 if i < 2 else 0.65
            trades.append((
                _make_perf_entry(
                    setup_type="crypto_breakout", symbol="SOL-USD",
                    win=win, pnl=pnl, date=date_base,
                ),
                _make_cal_entry(
                    confidence=conf, outcome="win" if win else "loss",
                    setup_type="crypto_breakout", date=date_base,
                ),
            ))

        assert len(trades) == 50
        return trades

    # ── Verification tests ───────────────────────────────────────────────

    def test_total_trades_written(self, loaded_db):
        session = get_session(loaded_db)
        try:
            assert session.query(SetupPerformanceRow).count() == 50
            assert session.query(ClaudeCalibrationRow).count() == 50
        finally:
            session.close()

    def test_overall_win_rate(self, loaded_db):
        session = get_session(loaded_db)
        try:
            total = session.query(SetupPerformanceRow).count()
            wins = (
                session.query(SetupPerformanceRow)
                .filter(SetupPerformanceRow.win == True)  # noqa: E712
                .count()
            )
            assert total == 50
            assert wins == 30
            assert wins / total == 0.6
        finally:
            session.close()

    def test_per_setup_win_rates(self, loaded_db):
        session = get_session(loaded_db)
        try:
            # crypto_mean_reversion: 20W / 30T = 66.7%
            cmr_t = session.query(SetupPerformanceRow).filter(
                SetupPerformanceRow.setup_type == "crypto_mean_reversion"
            ).count()
            cmr_w = session.query(SetupPerformanceRow).filter(
                SetupPerformanceRow.setup_type == "crypto_mean_reversion",
                SetupPerformanceRow.win == True,  # noqa: E712
            ).count()
            assert cmr_t == 30 and cmr_w == 20
            assert abs(cmr_w / cmr_t - 0.6667) < 0.001

            # equity_trend_pullback: 6W / 15T = 40%
            etp_t = session.query(SetupPerformanceRow).filter(
                SetupPerformanceRow.setup_type == "equity_trend_pullback"
            ).count()
            etp_w = session.query(SetupPerformanceRow).filter(
                SetupPerformanceRow.setup_type == "equity_trend_pullback",
                SetupPerformanceRow.win == True,  # noqa: E712
            ).count()
            assert etp_t == 15 and etp_w == 6
            assert etp_w / etp_t == 0.4

            # crypto_breakout: 4W / 5T = 80%
            cb_t = session.query(SetupPerformanceRow).filter(
                SetupPerformanceRow.setup_type == "crypto_breakout"
            ).count()
            cb_w = session.query(SetupPerformanceRow).filter(
                SetupPerformanceRow.setup_type == "crypto_breakout",
                SetupPerformanceRow.win == True,  # noqa: E712
            ).count()
            assert cb_t == 5 and cb_w == 4
            assert cb_w / cb_t == 0.8
        finally:
            session.close()

    def test_drift_not_triggered_at_default(self, loaded_db):
        # Last 20 by ID: SOL(4W 1L) + AAPL-last-15 → 10W 10L = 50% > 45%
        assert detect_win_rate_drift(loaded_db) is None

    def test_drift_triggered_at_higher_threshold(self, loaded_db):
        # 50% < 55%
        result = detect_win_rate_drift(loaded_db, window=20, threshold=0.55)
        assert result is not None
        assert result["win_rate"] == 0.5

    def test_calibration_buckets(self, loaded_db):
        result = analyze_claude_calibration(loaded_db)
        assert result["total_entries"] == 50
        b = result["buckets"]

        # 0.90-1.00: BTC(5W) + AAPL(3W) + SOL(2W) = 10 entries, 10W → 100%
        assert b["0.90-1.00"]["count"] == 10
        assert b["0.90-1.00"]["actual_win_rate"] == 1.0

        # 0.70-0.80: BTC(5W) + ETH(5W) = 10 entries, 10W → 100%
        assert b["0.70-0.80"]["count"] == 10
        assert b["0.70-0.80"]["actual_win_rate"] == 1.0

        # 0.60-0.70: ETH(5W) + AAPL(3W+2L) + SOL(2W+1L) = 13 entries, 10W → 76.9%
        assert b["0.60-0.70"]["count"] == 13
        assert abs(b["0.60-0.70"]["actual_win_rate"] - 0.7692) < 0.001

        # 0.50-0.60: BTC(5L) + ETH(5L) + AAPL(7L) = 17 entries, 0W → 0%
        assert b["0.50-0.60"]["count"] == 17
        assert b["0.50-0.60"]["actual_win_rate"] == 0.0

    def test_calibration_all_buckets_present(self, loaded_db):
        result = analyze_claude_calibration(loaded_db)
        assert set(result["buckets"].keys()) == {
            "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.90-1.00"
        }

    def test_symbol_behavior_btc(self, loaded_db):
        result = update_symbol_learned_behavior(
            "BTC-USD", "crypto_mean_reversion", loaded_db
        )
        assert result is not None
        assert result.sample_size == 15
        assert result.avg_bounce_magnitude == 25.0
        assert result.avg_reversion_depth == 15.0

    def test_symbol_behavior_eth(self, loaded_db):
        result = update_symbol_learned_behavior(
            "ETH-USD", "crypto_mean_reversion", loaded_db
        )
        assert result is not None
        assert result.sample_size == 15
        assert result.avg_bounce_magnitude == 20.0
        assert result.avg_reversion_depth == 12.0

    def test_symbol_behavior_aapl(self, loaded_db):
        result = update_symbol_learned_behavior(
            "AAPL", "equity_trend_pullback", loaded_db
        )
        assert result is not None
        assert result.sample_size == 15
        assert result.avg_bounce_magnitude == 30.0
        assert result.avg_reversion_depth == 10.0

    def test_symbol_behavior_sol(self, loaded_db):
        result = update_symbol_learned_behavior(
            "SOL-USD", "crypto_breakout", loaded_db
        )
        assert result is not None
        assert result.sample_size == 5
        assert result.avg_bounce_magnitude == 50.0
        assert result.avg_reversion_depth == 20.0

    def test_weekly_report_for_fifty_trades(self, loaded_db):
        # All 50 on 2025-06-02 → ISO W23
        result = generate_weekly_report("2025-W23", loaded_db)
        assert len(result) == 3  # 3 setup types

        by_type = {e.setup_type: e for e in result}

        cmr = by_type["crypto_mean_reversion"]
        assert cmr.trades == 30
        assert abs(cmr.win_rate - 0.6667) < 0.001
        assert cmr.avg_pnl == 10.5  # (10×25+10×20+5×(-15)+5×(-12))/30 = 315/30

        etp = by_type["equity_trend_pullback"]
        assert etp.trades == 15
        assert etp.win_rate == 0.4
        assert etp.avg_pnl == 6.0  # (6×30+9×(-10))/15 = 90/15

        cb = by_type["crypto_breakout"]
        assert cb.trades == 5
        assert cb.win_rate == 0.8
        assert cb.avg_pnl == 36.0  # (4×50+1×(-20))/5 = 180/5


# ── F-02: update_symbol_learned_behavior wired into run_learning_cycle ──────


class TestSymbolBehaviorInLearningCycle:
    """F-02: run_learning_cycle should call update_symbol_learned_behavior
    for each distinct (symbol, setup_type) pair with trade data."""

    def test_symbol_behavior_updated_during_cycle(self, tmp_strategic_db):
        """F-02: With sufficient trades, symbol behavior should be updated."""
        # Need at least 5 trades per (symbol, setup_type) for meaningful stats
        for i in range(10):
            record_trade_outcome(
                _make_perf_entry(
                    symbol="BTC-USD",
                    setup_type="crypto_mean_reversion",
                    win=i < 7,
                    pnl=20.0 if i < 7 else -10.0,
                ),
                tmp_strategic_db,
            )
        for i in range(8):
            record_trade_outcome(
                _make_perf_entry(
                    symbol="AAPL",
                    setup_type="equity_trend_pullback",
                    win=i < 4,
                    pnl=15.0 if i < 4 else -5.0,
                ),
                tmp_strategic_db,
            )

        result = run_learning_cycle("test-loop", tmp_strategic_db)
        assert "symbol_behavior_updated" in result
        assert result["symbol_behavior_updated"] == 2

        # Verify rows were written to symbol_learned_behavior table
        session = get_session(tmp_strategic_db)
        try:
            rows = session.query(SymbolLearnedBehaviorRow).all()
            symbols_updated = {(r.symbol, r.setup_type) for r in rows}
            assert ("BTC-USD", "crypto_mean_reversion") in symbols_updated
            assert ("AAPL", "equity_trend_pullback") in symbols_updated
        finally:
            session.close()

    def test_symbol_behavior_not_updated_when_no_data(self, tmp_strategic_db):
        """F-02: With no trades, symbol_behavior_updated should be empty or absent."""
        result = run_learning_cycle("test-loop", tmp_strategic_db)
        # Either key is absent (no data at all → empty dict) or is empty list
        assert result.get("symbol_behavior_updated", []) == []
