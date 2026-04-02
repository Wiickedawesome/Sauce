"""
adapters/db.py — Database adapter supporting SQLite and PostgreSQL (Supabase).

Provides the shared ORM Base, engine factory, session factory, and audit
logging used by both the trading engine (sauce/db.py) and adapters.

Database Selection:
- If SUPABASE_DB_URL is set → PostgreSQL via Supabase
- Otherwise → SQLite fallback at DB_PATH

Rules:
- ALL writes to audit_events are append-only. No UPDATE or DELETE — ever.
- Every broker and LLM action must call log_event() before and after.
- get_engine() is the single source of the DB connection.
- Tables are created automatically on first run (create_all) for SQLite only.
  For PostgreSQL/Supabase, run migrations via `supabase db push`.
"""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)


def _get_db_config() -> tuple[str, bool]:
    """
    Determine database URL and type from settings.

    Returns:
        (db_url, is_postgres): URL string and True if PostgreSQL, False if SQLite
    """
    try:
        from sauce.core.config import get_settings

        settings = get_settings()
        if settings.use_supabase and settings.supabase_db_url:
            return settings.supabase_db_url, True
        return f"sqlite:///{settings.db_path}", False
    except Exception:  # noqa: BLE001 — during early init or test isolation
        # Fallback to env vars for test isolation
        supabase_url = os.environ.get("SUPABASE_DB_URL", "")
        if supabase_url:
            return supabase_url, True
        db_path = os.environ.get("DB_PATH", "data/sauce.db")
        return f"sqlite:///{db_path}", False


def _default_db_url() -> str:
    """Resolve the DB URL from settings."""
    url, _ = _get_db_config()
    return url


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
    timestamp: datetime = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    prompt_version: str | None = Column(String(32), nullable=True)


# ── Engine Factory ────────────────────────────────────────────────────────────

# Cache engines keyed by db_url so tests can use isolated DBs without the
# singleton silently returning the first-call engine (Finding 7.4).
_engines: dict[str, Engine] = {}


def get_engine(db_url: str | None = None) -> Engine:
    """
    Return the cached SQLAlchemy engine for db_url, creating it on first call.

    Supports both SQLite and PostgreSQL:
    - SQLite: sqlite:///path/to/db.db
    - PostgreSQL: postgresql://user:pass@host:port/dbname

    Each distinct db_url gets its own engine so that test DBs are fully isolated
    from the production DB even within the same process (Finding 7.4).
    """
    if db_url is None:
        db_url = _default_db_url()

    global _engines
    if db_url not in _engines:
        is_postgres = db_url.startswith("postgresql")

        if is_postgres:
            # PostgreSQL via Supabase
            engine = create_engine(
                db_url,
                pool_pre_ping=True,  # Check connection health before use
                pool_size=5,
                max_overflow=10,
                pool_recycle=300,  # Recycle connections after 5 minutes
                echo=False,
            )
            logger.info("Connected to PostgreSQL (Supabase)")
            # Tables should be created via migrations for PostgreSQL
        else:
            # SQLite fallback
            # Extract path from sqlite:/// URL
            db_path = db_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            engine = create_engine(
                db_url,
                connect_args={"check_same_thread": False, "timeout": 30},
                echo=False,
            )
            # Enable WAL mode for concurrent read safety (consistent with memory/db.py).
            with engine.connect() as conn:
                conn.execute(text("PRAGMA journal_mode=WAL"))
                conn.commit()
            # Auto-create tables for SQLite (dev/test environments)
            Base.metadata.create_all(engine)
            logger.info("Connected to SQLite at %s", db_path)

        _engines[db_url] = engine
    return _engines[db_url]


# Cache session factories keyed by db_url (consistent with memory/db.py).
_session_factories: dict[str, sessionmaker[Session]] = {}


def get_session(db_url: str | None = None) -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    if db_url is None:
        db_url = _default_db_url()
    if db_url not in _session_factories:
        engine = get_engine(db_url)
        _session_factories[db_url] = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return _session_factories[db_url]()


def cleanup_engines() -> None:
    """Dispose all cached engines and clear the cache."""
    for engine in _engines.values():
        engine.dispose()
    _engines.clear()
    _session_factories.clear()


# ── Public Write Helpers ──────────────────────────────────────────────────────


def log_event(event: AuditEvent, db_url: str | None = None) -> None:
    """
    Append an AuditEvent to the audit_events table.

    This is the primary write path for every agent, adapter, and safety check.
    Never raises — on DB error, prints to stderr to avoid masking the original error.
    """
    session = get_session(db_url)
    try:
        row = AuditEventRow(
            loop_id=event.loop_id,
            event_type=event.event_type,
            symbol=event.symbol,
            payload=json.dumps(event.payload, default=str),
            timestamp=event.timestamp,
            prompt_version=event.prompt_version,
        )
        session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001 — last-resort catch so logging never crashes the loop
        # Log to Python logging system so any log aggregator captures this (Finding 4.2).
        logger.critical(
            "DB write FAILED for AuditEvent [loop_id=%s event_type=%s]: %s",
            event.loop_id,
            event.event_type,
            exc,
        )
    finally:
        session.close()
