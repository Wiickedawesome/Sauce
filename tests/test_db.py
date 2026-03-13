"""
tests/test_db.py — Tests for adapters/db.py SQLite adapter.

Uses an in-memory / temp-file SQLite DB for isolation.
No real network calls or production DB touched.
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sauce.adapters.db import (
    AuditEventRow,
    OrderRow,
    SignalRow,
    count_orders_today,
    get_daily_stats,
    get_engine,
    get_session,
    log_event,
    log_order,
    log_signal,
)
from sauce.core.schemas import AuditEvent, DailyStats


NOW = datetime.now(timezone.utc)


@pytest.fixture
def tmp_db(tmp_path: Path) -> str:
    """Return a path to a fresh temp SQLite DB for each test."""
    # Reset the module-level engine so each test gets a clean DB
    import sauce.adapters.db as db_module
    db_module._engines = {}
    db_path = str(tmp_path / "test_sauce.db")
    return db_path


# ── Engine & Table Creation ───────────────────────────────────────────────────

def test_engine_creates_db_file(tmp_db: str) -> None:
    engine = get_engine(tmp_db)
    assert Path(tmp_db).exists()
    engine.dispose()


def test_engine_creates_all_tables(tmp_db: str) -> None:
    from sqlalchemy import inspect
    engine = get_engine(tmp_db)
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    assert "audit_events" in tables
    assert "orders" in tables
    assert "signals" in tables
    assert "daily_stats" in tables
    engine.dispose()


def test_engine_is_cached(tmp_db: str) -> None:
    e1 = get_engine(tmp_db)
    e2 = get_engine(tmp_db)
    assert e1 is e2
    e1.dispose()


# ── log_event ─────────────────────────────────────────────────────────────────

def test_log_event_writes_to_db(tmp_db: str) -> None:
    event = AuditEvent(
        event_type="loop_start",
        payload={"test": True},
        prompt_version="v1",
    )
    log_event(event, db_path=tmp_db)

    session = get_session(tmp_db)
    rows = session.query(AuditEventRow).all()
    session.close()

    assert len(rows) == 1
    assert rows[0].event_type == "loop_start"
    assert rows[0].loop_id == event.loop_id


def test_log_event_is_append_only(tmp_db: str) -> None:
    """Two events create two rows — no upsert or deduplication."""
    e1 = AuditEvent(event_type="loop_start", payload={"run": 1})
    e2 = AuditEvent(event_type="loop_end", payload={"run": 1})
    log_event(e1, db_path=tmp_db)
    log_event(e2, db_path=tmp_db)

    session = get_session(tmp_db)
    rows = session.query(AuditEventRow).all()
    session.close()

    assert len(rows) == 2


def test_log_event_stores_symbol(tmp_db: str) -> None:
    event = AuditEvent(event_type="signal", symbol="AAPL", payload={})
    log_event(event, db_path=tmp_db)

    session = get_session(tmp_db)
    row = session.query(AuditEventRow).filter_by(symbol="AAPL").first()
    session.close()

    assert row is not None
    assert row.symbol == "AAPL"


def test_log_event_stores_json_payload(tmp_db: str) -> None:
    import json
    event = AuditEvent(
        event_type="order",
        payload={"symbol": "NVDA", "qty": 5, "side": "buy"},
    )
    log_event(event, db_path=tmp_db)

    session = get_session(tmp_db)
    row = session.query(AuditEventRow).first()
    session.close()

    assert row is not None
    payload = json.loads(row.payload)
    assert payload["symbol"] == "NVDA"
    assert payload["qty"] == 5


# ── log_signal ────────────────────────────────────────────────────────────────

def test_log_signal_writes_row(tmp_db: str) -> None:
    # Reset engine
    import sauce.adapters.db as db_module
    db_module._engines = {}

    row = SignalRow(
        loop_id="loop-001",
        symbol="MSFT",
        side="buy",
        confidence=0.72,
        reasoning="strong RSI",
        vetoed=False,
        as_of=NOW,
        prompt_version="v1",
    )
    log_signal(row, db_path=tmp_db)

    session = get_session(tmp_db)
    result = session.query(SignalRow).filter_by(symbol="MSFT").first()
    session.close()

    assert result is not None
    assert result.confidence == 0.72
    assert result.vetoed is False


# ── log_order ─────────────────────────────────────────────────────────────────

def test_log_order_writes_row(tmp_db: str) -> None:
    import sauce.adapters.db as db_module
    db_module._engines = {}

    row = OrderRow(
        loop_id="loop-002",
        symbol="AAPL",
        side="buy",
        qty=10.0,
        order_type="limit",
        time_in_force="day",
        limit_price=150.05,
        status="pending",
        prompt_version="v1",
    )
    log_order(row, db_path=tmp_db)

    session = get_session(tmp_db)
    result = session.query(OrderRow).filter_by(symbol="AAPL").first()
    session.close()

    assert result is not None
    assert result.qty == 10.0
    assert result.side == "buy"


# ── get_daily_stats ───────────────────────────────────────────────────────────

def test_get_daily_stats_returns_none_for_missing_date(tmp_db: str) -> None:
    import sauce.adapters.db as db_module
    db_module._engines = {}

    result = get_daily_stats("2026-01-01", db_path=tmp_db)
    assert result is None


# ── Audit immutability ────────────────────────────────────────────────────────

def test_no_update_method_exists_on_log_event() -> None:
    """Ensure there is no update_event or delete_event function exported."""
    import sauce.adapters.db as db_module
    assert not hasattr(db_module, "update_event")
    assert not hasattr(db_module, "delete_event")
    assert not hasattr(db_module, "update_order")
    assert not hasattr(db_module, "delete_order")
