"""
adapters/db.py — SQLite adapter via SQLAlchemy.

Rules:
- ALL writes are append-only. No UPDATE or DELETE on audit tables — ever.
- Every broker and LLM action must call log_event() before and after.
- get_engine() is the single source of the DB connection.
- Tables are created automatically on first run (create_all).
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

from datetime import timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sauce.core.schemas import AuditEvent, DailyStats

logger = logging.getLogger(__name__)


def _default_db_path() -> str:
    """Resolve the DB path from settings (config / env var), not a hardcoded string.

    This ensures all callers — including loop.py, agents, and adapters — write
    to the DB pointed to by the ``DB_PATH`` env var / ``get_settings().db_path``
    without having to thread the path explicitly through every call.
    """
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
        Base.metadata.create_all(engine)
        _engines[db_path] = engine
    return _engines[db_path]


def get_session(db_path: str | None = None) -> Session:
    """Return a new SQLAlchemy session. Caller is responsible for closing it."""
    engine = get_engine(db_path)
    SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return SessionLocal()


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
        import sys
        print(f"[db] CRITICAL: failed to write AuditEvent: {exc}", file=sys.stderr)
    finally:
        session.close()


def log_signal(
    signal_row: SignalRow,
    db_path: str | None = None,
) -> None:
    """Append a SignalRow to the signals table."""
    session = get_session(db_path)
    try:
        session.add(signal_row)
        session.commit()
    finally:
        session.close()


def log_order(order_row: OrderRow, db_path: str | None = None) -> None:
    """Append an OrderRow to the orders table."""
    session = get_session(db_path)
    try:
        session.add(order_row)
        session.commit()
    finally:
        session.close()


# ── Public Read Helpers ───────────────────────────────────────────────────────

def get_daily_stats(date: str, db_path: str | None = None) -> DailyStats | None:
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


def count_orders_today(db_path: str | None = None) -> int:
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


def upsert_daily_stats(
    date: str,
    db_path: str | None = None,
    **fields: object,
) -> None:
    """
    Insert or update a DailyStatsRow for the given date (YYYY-MM-DD).

    Uses merge() so multiple loop runs per day accumulate counts rather than
    overwriting.  Additive fields (loop_runs, signals_generated, etc.) are
    incremented; snapshot fields (starting_nav_usd, ending_nav_usd) are
    overwritten with the latest value (Finding 5.1).

    Parameters
    ----------
    date:    Trading day in YYYY-MM-DD format.
    db_path: Path to the SQLite database.
    **fields: Any subset of DailyStatsRow columns (excluding id, date).
             Unset fields keep their existing DB values (or default to 0).
    """
    session = get_session(db_path)
    try:
        existing = session.query(DailyStatsRow).filter_by(date=date).first()
        if existing is None:
            row = DailyStatsRow(date=date)
            session.add(row)
        else:
            row = existing

        # Additive integer counters — accumulate across loop runs per day.
        additive_int_fields = {
            "loop_runs", "signals_generated", "signals_vetoed", "orders_placed",
        }
        for key, value in fields.items():
            if not hasattr(row, key):
                continue
            if key in additive_int_fields and existing is not None:
                current = getattr(row, key, 0) or 0
                setattr(row, key, int(current) + int(value))  # type: ignore[arg-type]
            else:
                setattr(row, key, value)

        row.updated_at = datetime.now(timezone.utc)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        import sys
        print(f"[db] CRITICAL: upsert_daily_stats failed: {exc}", file=sys.stderr)
    finally:
        session.close()


# ── Maintenance ───────────────────────────────────────────────────────────────

def run_maintenance(
    db_path: str | None = None,
    *,
    retention_days: int = 90,
    vacuum: bool = True,
) -> dict:
    """
    Run periodic DB maintenance: integrity check, old audit event pruning,
    and optional VACUUM.

    Returns a summary dict with keys: integrity_ok, rows_pruned, vacuumed.
    Non-raising — logs errors and returns partial results.
    """
    result: dict = {"integrity_ok": None, "rows_pruned": 0, "vacuumed": False}
    engine = get_engine(db_path)

    with engine.connect() as conn:
        # 1. Integrity check
        try:
            check = conn.execute(text("PRAGMA integrity_check")).scalar()
            result["integrity_ok"] = check == "ok"
            if check != "ok":
                logger.error("db maintenance: integrity_check returned: %s", check)
        except Exception as exc:  # noqa: BLE001
            logger.error("db maintenance: integrity_check failed: %s", exc)

        # 2. Prune old audit_events beyond retention window
        try:
            cutoff = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            # SQLite date arithmetic: subtract retention_days
            r = conn.execute(
                text(
                    "DELETE FROM audit_events "
                    "WHERE timestamp < datetime(:cutoff, :offset)"
                ),
                {"cutoff": cutoff, "offset": f"-{retention_days} days"},
            )
            result["rows_pruned"] = r.rowcount
            conn.commit()
            if result["rows_pruned"] > 0:
                logger.info("db maintenance: pruned %d audit_events older than %d days",
                            result["rows_pruned"], retention_days)
        except Exception as exc:  # noqa: BLE001
            logger.error("db maintenance: audit_events pruning failed: %s", exc)

    # 3. VACUUM (reclaim space after pruning) — must run outside a transaction
    if vacuum:
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as ac_conn:
                ac_conn.execute(text("VACUUM"))
            result["vacuumed"] = True
            logger.info("db maintenance: VACUUM completed")
        except Exception as exc:  # noqa: BLE001
            logger.error("db maintenance: VACUUM failed: %s", exc)

    return result


def has_recent_submitted_order(
    symbol: str,
    side: str,
    minutes: int = 30,
    db_path: str | None = None,
) -> bool:
    """
    Return True if a 'submitted' order for this symbol+side was written to the
    orders table within the last `minutes` minutes.

    Used as an idempotency guard before calling place_order(). If the loop
    crashes mid-run and cron restarts it, orders already submitted to the
    broker will be detected here and skipped rather than re-submitted
    (Finding 3.2).

    Fails OPEN (returns False) on any DB error so a transient read failure
    never silently blocks a new order from going through.
    """
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    session = get_session(db_path)
    try:
        count = (
            session.query(OrderRow)
            .filter(
                OrderRow.symbol == symbol.upper(),
                OrderRow.side == side.lower(),
                OrderRow.status == "submitted",
                OrderRow.created_at >= cutoff,
            )
            .count()
        )
        return count > 0
    except Exception:  # noqa: BLE001 — fail open, never block a legitimate order
        return False
    finally:
        session.close()


def get_recent_signals(
    symbol: str,
    days: int = 7,
    db_path: str | None = None,
) -> list[dict]:
    """
    Return recent signals for a symbol from the signals table.

    Each dict has: side, confidence, vetoed, created_at.
    Returns an empty list on any error (fail open).
    """
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    session = get_session(db_path)
    try:
        rows = (
            session.query(SignalRow)
            .filter(
                SignalRow.symbol == symbol.upper(),
                SignalRow.created_at >= cutoff,
            )
            .order_by(SignalRow.created_at.desc())
            .limit(50)
            .all()
        )
        return [
            {
                "side": r.side,
                "confidence": r.confidence,
                "vetoed": r.vetoed,
                "created_at": str(r.created_at),
            }
            for r in rows
        ]
    except Exception:  # noqa: BLE001
        return []
    finally:
        session.close()


def get_supervisor_abort_rate(
    days: int = 7,
    db_path: str | None = None,
) -> dict:
    """
    Return supervisor decision stats over the given window.

    Returns dict with: total_decisions, aborts, executes, abort_rate.
    On error returns a safe default dict.
    """
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    session = get_session(db_path)
    try:
        rows = (
            session.query(AuditEventRow)
            .filter(
                AuditEventRow.event_type == "supervisor_decision",
                AuditEventRow.timestamp >= cutoff,
            )
            .all()
        )
        total = len(rows)
        aborts = 0
        executes = 0
        for r in rows:
            try:
                payload = json.loads(r.payload) if isinstance(r.payload, str) else r.payload
                action = payload.get("action", "")
                if action == "abort":
                    aborts += 1
                elif action == "execute":
                    executes += 1
            except (json.JSONDecodeError, AttributeError):
                pass
        return {
            "total_decisions": total,
            "aborts": aborts,
            "executes": executes,
            "abort_rate": round(aborts / total, 4) if total > 0 else 0.0,
        }
    except Exception:  # noqa: BLE001
        return {"total_decisions": 0, "aborts": 0, "executes": 0, "abort_rate": 0.0}
    finally:
        session.close()
