"""
tests/test_nav.py — Tests for core/nav.py NAV and fee calculations.

Pure math tests for the formulas + DB integration for get_high_water_mark.
No real broker calls.
"""

from pathlib import Path

import pytest

from sauce.core.nav import (
    _TRADING_DAYS_PER_YEAR,
    calc_daily_management_fee,
    calc_gross_nav,
    calc_performance_fee,
    compute_nav_and_fees,
    get_high_water_mark,
)


# ── calc_gross_nav ────────────────────────────────────────────────────────────


class TestCalcGrossNav:
    def test_positive_equity(self) -> None:
        assert calc_gross_nav(10_000.0) == 10_000.0

    def test_zero_equity(self) -> None:
        assert calc_gross_nav(0.0) == 0.0

    def test_negative_equity_clamped(self) -> None:
        assert calc_gross_nav(-500.0) == 0.0


# ── calc_daily_management_fee ─────────────────────────────────────────────────


class TestCalcDailyManagementFee:
    def test_normal(self) -> None:
        fee = calc_daily_management_fee(100_000.0, 0.01)
        expected = 100_000.0 * 0.01 / _TRADING_DAYS_PER_YEAR
        assert abs(fee - expected) < 0.01

    def test_zero_nav(self) -> None:
        assert calc_daily_management_fee(0.0, 0.01) == 0.0

    def test_zero_rate(self) -> None:
        assert calc_daily_management_fee(100_000.0, 0.0) == 0.0

    def test_negative_nav(self) -> None:
        assert calc_daily_management_fee(-100.0, 0.01) == 0.0

    def test_negative_rate(self) -> None:
        assert calc_daily_management_fee(100_000.0, -0.01) == 0.0


# ── calc_performance_fee ──────────────────────────────────────────────────────


class TestCalcPerformanceFee:
    def test_above_hwm(self) -> None:
        fee = calc_performance_fee(net_nav=110_000.0, high_water_mark=100_000.0, performance_rate=0.20)
        assert fee == 2_000.0

    def test_at_hwm(self) -> None:
        fee = calc_performance_fee(net_nav=100_000.0, high_water_mark=100_000.0, performance_rate=0.20)
        assert fee == 0.0

    def test_below_hwm(self) -> None:
        fee = calc_performance_fee(net_nav=90_000.0, high_water_mark=100_000.0, performance_rate=0.20)
        assert fee == 0.0

    def test_zero_rate(self) -> None:
        fee = calc_performance_fee(net_nav=110_000.0, high_water_mark=100_000.0, performance_rate=0.0)
        assert fee == 0.0

    def test_negative_rate(self) -> None:
        fee = calc_performance_fee(net_nav=110_000.0, high_water_mark=100_000.0, performance_rate=-0.1)
        assert fee == 0.0


# ── get_high_water_mark ───────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Provide a fresh temp DB with daily_stats table created."""
    import sauce.adapters.db as db_mod

    db_path = str(tmp_path / "test_nav.db")
    db_mod._engines.pop(db_path, None)
    db_mod.get_engine(db_path)  # creates tables
    return db_path


class TestGetHighWaterMark:
    def test_empty_db_returns_zero(self, tmp_db: str) -> None:
        assert get_high_water_mark(tmp_db) == 0.0

    def test_single_row(self, tmp_db: str) -> None:
        from sauce.adapters.db import get_session
        session = get_session(tmp_db)
        from sauce.adapters.db import DailyStatsRow
        from datetime import datetime, timezone

        session.add(DailyStatsRow(
            date="2026-03-01", ending_nav_usd=10_500.0,
            starting_nav_usd=10_000.0, updated_at=datetime.now(timezone.utc),
        ))
        session.commit()
        session.close()

        assert get_high_water_mark(tmp_db) == 10_500.0

    def test_picks_max(self, tmp_db: str) -> None:
        from sauce.adapters.db import DailyStatsRow, get_session
        from datetime import datetime, timezone

        session = get_session(tmp_db)
        for i, nav in enumerate([9_800.0, 10_200.0, 10_100.0]):
            session.add(DailyStatsRow(
                date=f"2026-03-0{i+1}", ending_nav_usd=nav,
                starting_nav_usd=10_000.0, updated_at=datetime.now(timezone.utc),
            ))
        session.commit()
        session.close()

        assert get_high_water_mark(tmp_db) == 10_200.0

    def test_ignores_zero_nav(self, tmp_db: str) -> None:
        from sauce.adapters.db import DailyStatsRow, get_session
        from datetime import datetime, timezone

        session = get_session(tmp_db)
        session.add(DailyStatsRow(
            date="2026-03-01", ending_nav_usd=0.0,
            starting_nav_usd=0.0, updated_at=datetime.now(timezone.utc),
        ))
        session.commit()
        session.close()

        assert get_high_water_mark(tmp_db) == 0.0


# ── compute_nav_and_fees ──────────────────────────────────────────────────────


class TestComputeNavAndFees:
    def test_returns_all_keys(self, tmp_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        from sauce.core.config import Settings
        settings = Settings(_env_file=None)

        result = compute_nav_and_fees(
            equity=10_000.0, date="2026-03-12",
            db_path=tmp_db, settings=settings, loop_id="test",
        )
        required_keys = {
            "gross_nav", "daily_mgmt_fee", "net_nav",
            "high_water_mark", "performance_fee", "fully_adjusted_nav",
            "annual_mgmt_fee_pct", "performance_fee_pct",
        }
        assert required_keys.issubset(result.keys())

    def test_fees_deducted_correctly(self, tmp_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        from sauce.core.config import Settings
        settings = Settings(_env_file=None)

        result = compute_nav_and_fees(
            equity=100_000.0, date="2026-03-12",
            db_path=tmp_db, settings=settings, loop_id="test",
        )
        assert result["gross_nav"] == 100_000.0
        assert result["daily_mgmt_fee"] > 0
        assert result["net_nav"] < result["gross_nav"]
        assert result["fully_adjusted_nav"] <= result["net_nav"]

    def test_zero_equity(self, tmp_db: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "k")
        monkeypatch.setenv("ALPACA_SECRET_KEY", "s")
        from sauce.core.config import Settings
        settings = Settings(_env_file=None)

        result = compute_nav_and_fees(
            equity=0.0, date="2026-03-12",
            db_path=tmp_db, settings=settings, loop_id="test",
        )
        assert result["gross_nav"] == 0.0
        assert result["daily_mgmt_fee"] == 0.0
        assert result["performance_fee"] == 0.0
