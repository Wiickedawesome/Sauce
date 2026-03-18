"""
adapters/db.py — SQLite adapter via SQLAlchemy.

Provides the shared ORM Base, engine factory, session factory, and audit
logging used by both the trading engine (sauce/db.py) and adapters.

Rules:
- ALL writes to audit_events are append-only. No UPDATE or DELETE — ever.
- Every broker and LLM action must call log_event() before and after.
- get_engine() is the single source of the DB connection.
- Tables are created automatically on first run (create_all).
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)


def _default_db_path() -> str:
    """Resolve the DB path from settings (config / env var), not a hardcoded string."""
    try:
        from sauce.core.config import get_settings
        return str(get_settings().db_path)
    except Exception:  # noqa: BLE001 — during early init or test isolation
        return os.environ.get("DB_PATH", "data/sauce.db")


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


# ── Engine Factory ────────────────────────────────────────────────────────────

# Cache engines keyed by db_path so tests can use isolated DBs without the
# singleton silently returning the first-call engine (Finding 7.4).
_engines: dict[str, Engine] = {}


def get_engine(db_path: str | None = None) -> Engine:
    """
    Return the cached SQLAlchemy engine for db_path, creating it on first call.

    Each distinct db_path gets its own engine so that test DBs are fully isolated
    from the production DB even within the same process (Finding 7.4).
    """
    if db_path is None:
        db_path = _default_db_path()
    global _engines
    if db_path not in _engines:
        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False, "timeout": 30},
            echo=False,
        )
        # Enable WAL mode for concurrent read safety (consistent with memory/db.py).
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
        Base.metadata.create_all(engine)
        _engines[db_path] = engine
    return _engines[db_path]


# Cache session factories keyed by db_path (consistent with memory/db.py).
_session_factories: dict[str, sessionmaker] = {}


def get_session(db_path: str | None = None) -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    if db_path is None:
        db_path = _default_db_path()
    if db_path not in _session_factories:
        engine = get_engine(db_path)
        _session_factories[db_path] = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _session_factories[db_path]()


def cleanup_engines() -> None:
    """Dispose all cached engines and clear the cache."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()
    _session_factories.clear()


# ── Public Write Helpers ──────────────────────────────────────────────────────

def log_event(event: AuditEvent, db_path: str | None = None) -> None:
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
        # Log to Python logging system so any log aggregator captures this (Finding 4.2).
        logger.critical(
            "DB write FAILED for AuditEvent [loop_id=%s event_type=%s]: %s",
            event.loop_id, event.event_type, exc,
        )
    finally:
        session.close()

