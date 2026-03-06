"""
adapters/db.py — SQLite adapter via SQLAlchemy.

Rules:
- ALL writes are append-only. No UPDATE or DELETE on audit tables — ever.
- Every broker and LLM action must call log_event() before and after.
- get_engine() is the single source of the DB connection.
- Tables are created automatically on first run (create_all).
"""

import json
import os
from datetime import datetime
from pathlib import Path

from datetime import timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sauce.core.schemas import AuditEvent, DailyStats


# ── ORM Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── Table Definitions ─────────────────────────────────────────────────────────

class AuditEventRow(Base):
    """Immutable audit log. One row per event. Never updated. Never deleted."""

    __tablename__ = "audit_events"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    loop_id: str = Column(String(36), nullable=False, index=True)
    event_type: str = Column(String(64), nullable=False, index=True)
    symbol: str | None = Column(String(20), nullable=True, index=True)
    payload: str = Column(Text, nullable=False, default="{}")  # JSON string
    timestamp: datetime = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    prompt_version: str | None = Column(String(32), nullable=True)


class OrderRow(Base):
    """Every order prepared by the Execution agent, approved or not."""

    __tablename__ = "orders"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    loop_id: str = Column(String(36), nullable=False, index=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    side: str = Column(String(8), nullable=False)
    qty: float = Column(Float, nullable=False)
    order_type: str = Column(String(16), nullable=False)
    time_in_force: str = Column(String(8), nullable=False)
    limit_price: float | None = Column(Float, nullable=True)
    stop_price: float | None = Column(Float, nullable=True)
    status: str = Column(String(16), nullable=False, default="pending")
    broker_order_id: str | None = Column(String(64), nullable=True)
    created_at: datetime = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    prompt_version: str = Column(String(32), nullable=False, default="v1")


class SignalRow(Base):
    """Every signal emitted by the Research agent."""

    __tablename__ = "signals"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    loop_id: str = Column(String(36), nullable=False, index=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    side: str = Column(String(8), nullable=False)
    confidence: float = Column(Float, nullable=False)
    reasoning: str = Column(Text, nullable=False, default="")
    vetoed: bool = Column(Boolean, nullable=False, default=False)
    veto_reason: str | None = Column(Text, nullable=True)
    as_of: datetime = Column(DateTime, nullable=False)
    created_at: datetime = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    prompt_version: str = Column(String(32), nullable=False, default="v1")


class DailyStatsRow(Base):
    """One row per trading day, written by the Ops agent."""

    __tablename__ = "daily_stats"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: str = Column(String(10), nullable=False, unique=True, index=True)  # YYYY-MM-DD
    loop_runs: int = Column(Integer, nullable=False, default=0)
    signals_generated: int = Column(Integer, nullable=False, default=0)
    signals_vetoed: int = Column(Integer, nullable=False, default=0)
    orders_placed: int = Column(Integer, nullable=False, default=0)
    realized_pnl_usd: float = Column(Float, nullable=False, default=0.0)
    starting_nav_usd: float = Column(Float, nullable=False, default=0.0)
    ending_nav_usd: float = Column(Float, nullable=False, default=0.0)
    trading_paused: bool = Column(Boolean, nullable=False, default=False)
    updated_at: datetime = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


# ── Engine Factory ────────────────────────────────────────────────────────────

_engine: Engine | None = None


def get_engine(db_path: str = "data/sauce.db") -> Engine:
    """
    Return the cached SQLAlchemy engine, creating it on first call.

    Creates the DB file and all tables if they don't exist.
    """
    global _engine
    if _engine is None:
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(_engine)
    return _engine


def get_session(db_path: str = "data/sauce.db") -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal()


# ── Public Write Helpers ──────────────────────────────────────────────────────

def log_event(event: AuditEvent, db_path: str = "data/sauce.db") -> None:
    """
    Append an AuditEvent to the audit_events table.

    This is the primary write path for every agent, adapter, and safety check.
    Never raises — on DB error, prints to stderr to avoid masking the original error.
    """
    session = get_session(db_path)
    try:
        row = AuditEventRow(
            loop_id=event.loop_id,
            event_type=event.event_type,
            symbol=event.symbol,
            payload=json.dumps(event.payload),
            timestamp=event.timestamp,
            prompt_version=event.prompt_version,
        )
        session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001 — last-resort catch so logging never crashes the loop
        import sys
        print(f"[db] CRITICAL: failed to write AuditEvent: {exc}", file=sys.stderr)
    finally:
        session.close()


def log_signal(
    signal_row: SignalRow,
    db_path: str = "data/sauce.db",
) -> None:
    """Append a SignalRow to the signals table."""
    session = get_session(db_path)
    try:
        session.add(signal_row)
        session.commit()
    finally:
        session.close()


def log_order(order_row: OrderRow, db_path: str = "data/sauce.db") -> None:
    """Append an OrderRow to the orders table."""
    session = get_session(db_path)
    try:
        session.add(order_row)
        session.commit()
    finally:
        session.close()


# ── Public Read Helpers ───────────────────────────────────────────────────────

def get_daily_stats(date: str, db_path: str = "data/sauce.db") -> DailyStats | None:
    """
    Return DailyStats for a given date string (YYYY-MM-DD), or None if not found.

    Used by safety.py to check daily loss limits.
    """
    session = get_session(db_path)
    try:
        row = session.query(DailyStatsRow).filter_by(date=date).first()
        if row is None:
            return None
        return DailyStats(
            date=row.date,
            loop_runs=row.loop_runs,
            signals_generated=row.signals_generated,
            signals_vetoed=row.signals_vetoed,
            orders_placed=row.orders_placed,
            realized_pnl_usd=row.realized_pnl_usd,
            starting_nav_usd=row.starting_nav_usd,
            ending_nav_usd=row.ending_nav_usd,
            trading_paused=row.trading_paused,
        )
    finally:
        session.close()


def is_trading_paused_in_db(db_path: str = "data/sauce.db") -> bool:
    """
    Check if a pause flag has been written to the DB by a previous loop run.

    A DB-level pause persists across restarts. Only resume_trading() clears it.
    """
    session = get_session(db_path)
    try:
        result = session.execute(
            text(
                "SELECT COUNT(*) FROM audit_events "
                "WHERE event_type = 'safety_check' "
                "AND json_extract(payload, '$.paused') = 1 "
                "ORDER BY timestamp DESC LIMIT 1"
            )
        ).scalar()
        return bool(result and result > 0)
    finally:
        session.close()


def count_orders_today(db_path: str = "data/sauce.db") -> int:
    """Return count of orders placed today. Used by Ops agent."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    session = get_session(db_path)
    try:
        count = (
            session.query(OrderRow)
            .filter(OrderRow.created_at >= today)
            .count()
        )
        return int(count)
    finally:
        session.close()
