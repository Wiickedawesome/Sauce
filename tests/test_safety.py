"""
tests/test_safety.py — Tests for core/safety.py.

Covers every safety guard per Testing Rule 6.3:
- TRADING_PAUSE=true
- DB-level pause and resume
- Stale data (is_data_fresh)
- Daily loss breach
- Market hours (open, closed, weekend, crypto)
"""

import importlib
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

import sauce.adapters.db as db_module

_ET = ZoneInfo("America/New_York")


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_db_and_settings(tmp_path, monkeypatch):
    """
    Isolate each test: use a fresh in-memory DB and clear the settings cache.
    """
    db_path = str(tmp_path / "test_safety.db")
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("LLM_PROVIDER", "github")
    monkeypatch.setenv("DB_PATH", db_path)
    # Do NOT set TRADING_PAUSE here — each test sets it as needed

    # Reset engine so it picks up new DB path
    db_module._engine = None

    from sauce.core.config import get_settings
    get_settings.cache_clear()

    yield

    db_module._engine = None
    get_settings.cache_clear()


# ── is_trading_paused — config level ─────────────────────────────────────────

def test_paused_when_config_flag_true(monkeypatch):
    monkeypatch.setenv("TRADING_PAUSE", "true")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import is_trading_paused
    assert is_trading_paused() is True


def test_not_paused_when_config_flag_false(monkeypatch):
    monkeypatch.setenv("TRADING_PAUSE", "false")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import is_trading_paused
    assert is_trading_paused() is False


# ── is_trading_paused — DB level ──────────────────────────────────────────────

def test_paused_after_pause_trading_written_to_db(monkeypatch):
    monkeypatch.setenv("TRADING_PAUSE", "false")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import is_trading_paused, pause_trading

    assert is_trading_paused() is False  # starts unpaused

    pause_trading("test reason", loop_id="test-loop")

    assert is_trading_paused() is True


def test_not_paused_after_resume_trading(monkeypatch):
    monkeypatch.setenv("TRADING_PAUSE", "false")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import is_trading_paused, pause_trading, resume_trading

    pause_trading("test reason")
    assert is_trading_paused() is True

    resume_trading()
    assert is_trading_paused() is False


def test_paused_after_pause_then_remains_when_config_is_true(monkeypatch):
    """Config-level pause overrides DB-level resume."""
    monkeypatch.setenv("TRADING_PAUSE", "true")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import is_trading_paused, resume_trading

    resume_trading()  # DB-level resume
    # But config is still paused — should remain paused
    assert is_trading_paused() is True


def test_pause_trading_writes_audit_event(monkeypatch):
    monkeypatch.setenv("TRADING_PAUSE", "false")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.adapters.db import get_session
    from sauce.core.safety import pause_trading

    pause_trading("abnormal P&L", loop_id="audit-test")

    session = get_session()
    try:
        from sqlalchemy import text
        row = session.execute(
            text(
                "SELECT event_type, payload FROM audit_events "
                "WHERE event_type = 'safety_check' LIMIT 1"
            )
        ).fetchone()
    finally:
        session.close()

    assert row is not None
    assert row[0] == "safety_check"


def test_resume_trading_writes_audit_event(monkeypatch):
    monkeypatch.setenv("TRADING_PAUSE", "false")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.adapters.db import get_session
    from sauce.core.safety import pause_trading, resume_trading

    pause_trading("testing")
    resume_trading(loop_id="resume-test")

    session = get_session()
    try:
        from sqlalchemy import text
        count = session.execute(
            text(
                "SELECT COUNT(*) FROM audit_events WHERE event_type = 'safety_check'"
            )
        ).scalar()
    finally:
        session.close()

    # At least 2 safety_check events (pause + resume); is_trading_paused() may
    # also write diagnostic events so we assert >=2 rather than ==2
    assert count >= 2


# ── is_data_fresh ─────────────────────────────────────────────────────────────

def test_data_fresh_within_ttl():
    from sauce.core.safety import is_data_fresh

    recent = datetime.now(timezone.utc) - timedelta(seconds=30)
    assert is_data_fresh(recent, ttl_sec=120) is True


def test_data_stale_exceeds_ttl():
    from sauce.core.safety import is_data_fresh

    old = datetime.now(timezone.utc) - timedelta(seconds=200)
    assert is_data_fresh(old, ttl_sec=120) is False


def test_data_fresh_exactly_at_ttl_boundary():
    from sauce.core.safety import is_data_fresh

    # Use ttl_sec=121 so the boundary cannot slip past us in the tiny time
    # between capturing the timestamp and the function running.
    now = datetime.now(timezone.utc) - timedelta(seconds=120)
    assert is_data_fresh(now, ttl_sec=121) is True


def test_data_fresh_naive_datetime_treated_as_utc():
    """Naive datetimes (no tzinfo) should be coerced to UTC without crashing.

    datetime.now() returns LOCAL time and must NOT be used — the real
    data path always provides UTC-aware datetimes. We use datetime.utcnow()
    to produce a naive datetime that, after replace(tzinfo=UTC), is correct.
    """
    from sauce.core.safety import is_data_fresh
    import warnings

    # datetime.utcnow() → naive datetime numerically equal to UTC
    # After replace(tzinfo=utc) inside is_data_fresh it is treated as UTC correctly
    naive_utc_recent = datetime.utcnow() - timedelta(seconds=10)
    assert is_data_fresh(naive_utc_recent, ttl_sec=120) is True


def test_data_stale_very_old():
    from sauce.core.safety import is_data_fresh

    # Data from 1 hour ago — always stale
    hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
    assert is_data_fresh(hour_ago, ttl_sec=120) is False


# ── check_daily_loss ──────────────────────────────────────────────────────────

def test_check_daily_loss_within_limit(monkeypatch):
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.02")  # 2%
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import check_daily_loss

    account = {
        "equity": "10000.00",
        "last_equity": "10050.00",  # -0.5% loss — within 2% limit
    }
    assert check_daily_loss(account) is True


def test_check_daily_loss_breach(monkeypatch):
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.02")  # 2%
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import check_daily_loss

    account = {
        "equity": "9700.00",
        "last_equity": "10000.00",  # -3% loss — exceeds 2% limit
    }
    assert check_daily_loss(account) is False


def test_check_daily_loss_breach_triggers_pause(monkeypatch):
    """A breached loss limit must automatically write a DB pause."""
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.02")
    monkeypatch.setenv("TRADING_PAUSE", "false")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import check_daily_loss, is_trading_paused

    account = {"equity": "9700.00", "last_equity": "10000.00"}
    check_daily_loss(account, loop_id="loss-test")

    # Should now be paused via DB
    assert is_trading_paused() is True


def test_check_daily_loss_gain_is_ok(monkeypatch):
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.02")
    from sauce.core.config import get_settings
    get_settings.cache_clear()
    from sauce.core.safety import check_daily_loss

    account = {
        "equity": "10200.00",
        "last_equity": "10000.00",  # +2% gain
    }
    assert check_daily_loss(account) is True


def test_check_daily_loss_missing_last_equity_blocks(monkeypatch):
    """Missing last_equity → fail safe → return False."""
    from sauce.core.safety import check_daily_loss

    account = {"equity": "10000.00"}  # no last_equity
    assert check_daily_loss(account) is False


def test_check_daily_loss_zero_last_equity_blocks(monkeypatch):
    """last_equity=0 → cannot compute → fail safe → return False."""
    from sauce.core.safety import check_daily_loss

    account = {"equity": "10000.00", "last_equity": "0"}
    assert check_daily_loss(account) is False


def test_check_daily_loss_invalid_values_blocks(monkeypatch):
    """Non-numeric values → fail safe → return False (never raises)."""
    from sauce.core.safety import check_daily_loss

    account = {"equity": "N/A", "last_equity": "N/A"}
    assert check_daily_loss(account) is False


# ── check_market_hours ────────────────────────────────────────────────────────

def test_market_open_tuesday_10am():
    from sauce.core import safety
    # Tuesday 10:00 AM ET — market open
    tuesday_10am = datetime(2024, 1, 2, 10, 0, 0, tzinfo=_ET)  # Jan 2 2024 = Tuesday

    with patch.object(safety, "_now_et", return_value=tuesday_10am):
        assert safety.check_market_hours("AAPL") is True


def test_market_closed_before_open():
    from sauce.core import safety
    # Tuesday 8:00 AM ET — before market open
    tuesday_8am = datetime(2024, 1, 2, 8, 0, 0, tzinfo=_ET)

    with patch.object(safety, "_now_et", return_value=tuesday_8am):
        assert safety.check_market_hours("AAPL") is False


def test_market_closed_after_close():
    from sauce.core import safety
    # Tuesday 16:00 ET exactly — market closed (close is exclusive)
    tuesday_4pm = datetime(2024, 1, 2, 16, 0, 0, tzinfo=_ET)

    with patch.object(safety, "_now_et", return_value=tuesday_4pm):
        assert safety.check_market_hours("AAPL") is False


def test_market_closed_saturday():
    from sauce.core import safety
    # Saturday 11:00 AM ET — weekend
    saturday = datetime(2024, 1, 6, 11, 0, 0, tzinfo=_ET)  # Jan 6 2024 = Saturday

    with patch.object(safety, "_now_et", return_value=saturday):
        assert safety.check_market_hours("AAPL") is False


def test_market_closed_sunday():
    from sauce.core import safety
    sunday = datetime(2024, 1, 7, 11, 0, 0, tzinfo=_ET)  # Jan 7 2024 = Sunday

    with patch.object(safety, "_now_et", return_value=sunday):
        assert safety.check_market_hours("AAPL") is False


def test_crypto_always_open_weekday():
    from sauce.core import safety
    tuesday_10am = datetime(2024, 1, 2, 10, 0, 0, tzinfo=_ET)

    with patch.object(safety, "_now_et", return_value=tuesday_10am):
        assert safety.check_market_hours("BTC/USD") is True


def test_crypto_always_open_weekend():
    from sauce.core import safety
    saturday = datetime(2024, 1, 6, 11, 0, 0, tzinfo=_ET)

    # _now_et should NOT be called for crypto — but no error either way
    with patch.object(safety, "_now_et", return_value=saturday):
        assert safety.check_market_hours("BTC/USD") is True


def test_crypto_always_open_after_hours():
    from sauce.core import safety
    tuesday_9pm = datetime(2024, 1, 2, 21, 0, 0, tzinfo=_ET)

    with patch.object(safety, "_now_et", return_value=tuesday_9pm):
        assert safety.check_market_hours("ETH/USD") is True


def test_market_open_at_930_boundary():
    from sauce.core import safety
    # Exactly at market open — should be True
    tuesday_930 = datetime(2024, 1, 2, 9, 30, 0, tzinfo=_ET)

    with patch.object(safety, "_now_et", return_value=tuesday_930):
        assert safety.check_market_hours("MSFT") is True
