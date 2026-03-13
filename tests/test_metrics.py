"""
tests/test_metrics.py — Tests for core/metrics.py trailing performance metrics.

Uses a real temp SQLite DB to exercise the SQL query and pure-math computation.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from sauce.core.metrics import compute_trailing_metrics


@pytest.fixture
def metrics_db(tmp_path: Path) -> str:
    """Create a fresh temp DB with daily_stats table."""
    import sauce.adapters.db as db_mod

    db_path = str(tmp_path / "metrics_test.db")
    db_mod._engines.pop(db_path, None)
    db_mod.get_engine(db_path)
    return db_path


def _insert_nav_series(db_path: str, navs: list[tuple[str, float]]) -> None:
    """Insert (date, ending_nav_usd) rows into the DB."""
    from sauce.adapters.db import DailyStatsRow, get_session

    session = get_session(db_path)
    for date_str, nav_val in navs:
        session.add(DailyStatsRow(
            date=date_str,
            ending_nav_usd=nav_val,
            starting_nav_usd=nav_val,
            updated_at=datetime.now(timezone.utc),
        ))
    session.commit()
    session.close()


class TestComputeTrailingMetrics:
    def test_empty_db_returns_none_metrics(self, metrics_db: str) -> None:
        result = compute_trailing_metrics(days=30, db_path=metrics_db)
        assert result["sharpe_ratio"] is None
        assert result["max_drawdown"] is None
        assert result["total_return"] is None
        assert result["days_computed"] == 0

    def test_single_row_insufficient(self, metrics_db: str) -> None:
        _insert_nav_series(metrics_db, [("2026-03-01", 10_000.0)])
        result = compute_trailing_metrics(days=30, db_path=metrics_db)
        assert result["total_return"] is None

    def test_flat_series(self, metrics_db: str) -> None:
        """Flat NAV series → 0 return, 0 drawdown, Sharpe is None (zero vol)."""
        navs = [(f"2026-03-{str(d).zfill(2)}", 10_000.0) for d in range(1, 11)]
        _insert_nav_series(metrics_db, navs)

        result = compute_trailing_metrics(days=30, db_path=metrics_db)
        assert result["total_return"] == 0.0
        assert result["max_drawdown"] == 0.0
        assert result["sharpe_ratio"] is None  # zero std → no Sharpe
        assert result["start_nav"] == 10_000.0
        assert result["end_nav"] == 10_000.0

    def test_upward_series(self, metrics_db: str) -> None:
        """Monotonically rising NAV → positive return, zero drawdown."""
        navs = [(f"2026-03-{str(d).zfill(2)}", 10_000.0 + d * 100) for d in range(1, 11)]
        _insert_nav_series(metrics_db, navs)

        result = compute_trailing_metrics(days=30, db_path=metrics_db)
        assert result["total_return"] is not None
        assert result["total_return"] > 0
        assert result["max_drawdown"] == 0.0

    def test_drawdown(self, metrics_db: str) -> None:
        """Peak → trough → recovery gives negative max_drawdown."""
        navs = [
            ("2026-03-01", 100.0),
            ("2026-03-02", 110.0),  # peak
            ("2026-03-03", 99.0),   # trough: dd = (99-110)/110 ≈ -0.1
            ("2026-03-04", 105.0),
        ]
        _insert_nav_series(metrics_db, navs)

        result = compute_trailing_metrics(days=30, db_path=metrics_db)
        assert result["max_drawdown"] is not None
        assert result["max_drawdown"] < 0
        assert abs(result["max_drawdown"] - (-0.1)) < 0.001

    def test_days_limit(self, metrics_db: str) -> None:
        """With days=3 we get at most 4 rows (days+1 for LIMIT)."""
        navs = [(f"2026-03-{str(d).zfill(2)}", 10_000.0 + d * 50) for d in range(1, 21)]
        _insert_nav_series(metrics_db, navs)

        result = compute_trailing_metrics(days=3, db_path=metrics_db)
        assert result["days_computed"] <= 4
        assert result["total_return"] is not None

    def test_as_of_present(self, metrics_db: str) -> None:
        result = compute_trailing_metrics(days=30, db_path=metrics_db)
        assert "as_of" in result
        assert "T" in result["as_of"]  # ISO format check
