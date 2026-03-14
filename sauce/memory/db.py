"""
adapters/memory_db.py — SQLite adapters for session and strategic memory.

Rules:
- Session memory (session_memory.db) is wiped at the start of each trading day.
- Strategic memory (strategic_memory.db) is append-only and never wiped.
- Write helpers NEVER raise — on DB error, log critical + print to stderr.
- Read helpers return Pydantic models, not raw ORM rows.
- Engine caching uses the same pattern as db.py (_engines dict keyed by path).
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

from datetime import timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from sauce.core.schemas import (
    ClaudeCalibrationEntry,
    IntradayNarrativeEntry,
    MarketRegime,
    PositionPeakPnL,
    RegimeLogEntry,
    RegimeTransitionEntry,
    SessionContext,
    SetupPerformanceEntry,
    SetupType,
    SignalLogEntry,
    StrategicContext,
    SymbolCharacterEntry,
    SymbolLearnedBehaviorEntry,
    TradeLogEntry,
    VetoPatternEntry,
    WeeklyPerformanceEntry,
)

logger = logging.getLogger(__name__)


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize SQLite datetimes to UTC-aware datetimes."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ── ORM Bases (one per database) ─────────────────────────────────────────────

class SessionBase(DeclarativeBase):
    pass


class StrategicBase(DeclarativeBase):
    pass


# ══════════════════════════════════════════════════════════════════════════════
# SESSION MEMORY TABLES (data/session_memory.db — wiped daily)
# ══════════════════════════════════════════════════════════════════════════════

class RegimeLogRow(SessionBase):
    """Market regime snapshot per cycle."""

    __tablename__ = "regime_log"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, nullable=False)
    regime_type: str = Column(String(20), nullable=False)
    confidence: float = Column(Float, nullable=False)
    vix_proxy: float | None = Column(Float, nullable=True)
    market_bias: str | None = Column(String(32), nullable=True)


class SignalLogRow(SessionBase):
    """Every signal generated during the trading day."""

    __tablename__ = "signal_log"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, nullable=False)
    symbol: str = Column(String(20), nullable=False, index=True)
    setup_type: str = Column(String(40), nullable=False)
    score: float = Column(Float, nullable=False)
    claude_decision: str = Column(String(10), nullable=False)
    reason: str | None = Column(Text, nullable=True)


class TradeLogRow(SessionBase):
    """Intraday trade records."""

    __tablename__ = "trade_log"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, nullable=False)
    symbol: str = Column(String(20), nullable=False, index=True)
    entry_price: float = Column(Float, nullable=False)
    direction: str = Column(String(8), nullable=False)
    status: str = Column(String(16), nullable=False)
    unrealized_pnl: float = Column(Float, nullable=False, default=0.0)


class IntradayNarrativeRow(SessionBase):
    """Running narrative entries for the day."""

    __tablename__ = "intraday_narrative"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    timestamp: datetime = Column(DateTime, nullable=False)
    narrative_text: str = Column(Text, nullable=False)


class SymbolCharacterRow(SessionBase):
    """Per-symbol intraday behavior profile."""

    __tablename__ = "symbol_character"
    __table_args__ = (UniqueConstraint("symbol", name="uq_symbol_character"),)

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    signal_count_today: int = Column(Integer, nullable=False, default=0)
    direction_consistency: float = Column(Float, nullable=False, default=0.0)
    last_signal_result: str = Column(String(10), nullable=False, default="none")


class PositionPeakPnLRow(SessionBase):
    """High-water mark of unrealized P&L per symbol for trailing stop logic."""

    __tablename__ = "position_peak_pnl"
    __table_args__ = (UniqueConstraint("symbol", name="uq_position_peak_pnl"),)

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    peak_unrealized_pnl: float = Column(Float, nullable=False, default=0.0)
    peak_at: datetime = Column(DateTime, nullable=False)


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIC MEMORY TABLES (data/strategic_memory.db — never wipes)
# ══════════════════════════════════════════════════════════════════════════════

class SetupPerformanceRow(StrategicBase):
    """Historical performance record per setup execution."""

    __tablename__ = "setup_performance"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    setup_type: str = Column(String(40), nullable=False, index=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    regime_at_entry: str = Column(String(20), nullable=False)
    time_of_day_bucket: str = Column(String(20), nullable=False)
    win: bool = Column(Boolean, nullable=False)
    pnl: float = Column(Float, nullable=False)
    hold_duration_minutes: float = Column(Float, nullable=False)
    date: str = Column(String(10), nullable=False, index=True)


class RegimeTransitionRow(StrategicBase):
    """Observed regime transitions and durations."""

    __tablename__ = "regime_transitions"
    __table_args__ = (UniqueConstraint("from_regime", "to_regime", name="uq_regime_transition"),)

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    from_regime: str = Column(String(20), nullable=False)
    to_regime: str = Column(String(20), nullable=False)
    duration_minutes: float = Column(Float, nullable=False)
    count: int = Column(Integer, nullable=False, default=1)


class VetoPatternRow(StrategicBase):
    """Recurring veto reasons by setup type."""

    __tablename__ = "veto_patterns"
    __table_args__ = (UniqueConstraint("veto_reason", "setup_type", name="uq_veto_pattern"),)

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    veto_reason: str = Column(Text, nullable=False)
    setup_type: str = Column(String(40), nullable=False, index=True)
    count: int = Column(Integer, nullable=False, default=1)
    last_seen: datetime = Column(DateTime, nullable=False)


class WeeklyPerformanceRow(StrategicBase):
    """Weekly aggregated performance by setup type."""

    __tablename__ = "weekly_performance"
    __table_args__ = (UniqueConstraint("week", "setup_type", name="uq_weekly_performance"),)

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    week: str = Column(String(10), nullable=False, index=True)
    setup_type: str = Column(String(40), nullable=False)
    trades: int = Column(Integer, nullable=False, default=0)
    win_rate: float = Column(Float, nullable=False, default=0.0)
    avg_pnl: float = Column(Float, nullable=False, default=0.0)
    sharpe: float = Column(Float, nullable=False, default=0.0)


class SymbolLearnedBehaviorRow(StrategicBase):
    """Learned indicator thresholds per symbol/setup."""

    __tablename__ = "symbol_learned_behavior"
    __table_args__ = (UniqueConstraint("symbol", "setup_type", name="uq_symbol_behavior"),)

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    setup_type: str = Column(String(40), nullable=False, index=True)
    optimal_rsi_entry: float | None = Column(Float, nullable=True)
    avg_reversion_depth: float | None = Column(Float, nullable=True)
    avg_bounce_magnitude: float | None = Column(Float, nullable=True)
    sample_size: int = Column(Integer, nullable=False, default=0)


class ClaudeCalibrationRow(StrategicBase):
    """Claude confidence vs actual outcome tracking."""

    __tablename__ = "claude_calibration"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: str = Column(String(10), nullable=False, index=True)
    confidence_stated: float = Column(Float, nullable=False)
    outcome: str = Column(String(8), nullable=False)
    setup_type: str = Column(String(40), nullable=False)


class ValidationResultRow(StrategicBase):
    """Daily paper-trading validation result (30-day gate)."""

    __tablename__ = "validation_results"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: str = Column(String(10), nullable=False, unique=True, index=True)
    win_rate: float = Column(Float, nullable=False)
    expectancy: float = Column(Float, nullable=False)
    max_drawdown_pct: float = Column(Float, nullable=False)
    sharpe_ratio: float = Column(Float, nullable=False)
    max_single_day_loss_pct: float = Column(Float, nullable=False)
    calibration_score: float = Column(Float, nullable=False)
    all_passed: bool = Column(Boolean, nullable=False)
    consecutive_days: int = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


# ── Engine Factory ────────────────────────────────────────────────────────────

_engines: dict[str, Engine] = {}


def get_engine(db_path: str) -> Engine:
    """Return cached engine for db_path, creating on first call."""
    global _engines
    if db_path not in _engines:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        # Enable WAL mode for better concurrent read/write performance
        with engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
        # Create tables for the correct base based on db_path
        if "session_memory" in db_path:
            SessionBase.metadata.create_all(engine)
        elif "strategic_memory" in db_path:
            StrategicBase.metadata.create_all(engine)
        else:
            # Fallback: create both
            SessionBase.metadata.create_all(engine)
            StrategicBase.metadata.create_all(engine)
        # Ensure unique indexes exist on pre-existing databases
        _ensure_unique_indexes(engine, db_path)
        _engines[db_path] = engine
    return _engines[db_path]


def _ensure_unique_indexes(engine: Engine, db_path: str) -> None:
    """Create unique indexes on existing tables that may lack them."""
    indexes: list[str] = []
    if "session_memory" in db_path:
        indexes = [
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_symbol_character ON symbol_character (symbol)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_position_peak_pnl ON position_peak_pnl (symbol)",
        ]
    elif "strategic_memory" in db_path:
        indexes = [
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_regime_transition ON regime_transitions (from_regime, to_regime)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_veto_pattern ON veto_patterns (veto_reason, setup_type)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_weekly_performance ON weekly_performance (week, setup_type)",
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_symbol_behavior ON symbol_learned_behavior (symbol, setup_type)",
        ]
    if not indexes:
        return
    with engine.connect() as conn:
        for idx_sql in indexes:
            try:
                conn.execute(text(idx_sql))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not create unique index (duplicates may exist): %s — %s", idx_sql, exc)
        conn.commit()


def get_session(db_path: str) -> Session:
    """Return a new SQLAlchemy session. Caller must close."""
    engine = get_engine(db_path)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return factory()


# ── Session Reset (wipe session_memory.db at start of each day) ──────────────

_last_reset_date: str | None = None


def reset_session_memory_if_new_day(db_path: str) -> bool:
    """
    Truncate all session memory tables if the current UTC date differs from
    the last reset date. Returns True if a reset was performed.

    Called at the top of each loop run before any reads or writes.
    """
    global _last_reset_date
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if _last_reset_date == today:
        return False

    session = get_session(db_path)
    try:
        for table_cls in [
            RegimeLogRow,
            SignalLogRow,
            TradeLogRow,
            IntradayNarrativeRow,
            SymbolCharacterRow,
            PositionPeakPnLRow,
        ]:
            session.query(table_cls).delete()
        session.commit()
        _last_reset_date = today
        logger.info("Session memory reset for new day: %s", today)
        return True
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to reset session memory: %s", exc)
        print(f"[memory_db] CRITICAL: session reset failed: {exc}", file=sys.stderr)
        return False
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# WRITE HELPERS — Never raise
# ══════════════════════════════════════════════════════════════════════════════

# ── Session Memory Writes ─────────────────────────────────────────────────────

def write_regime_log(entry: RegimeLogEntry, db_path: str) -> None:
    """Append a regime observation to session memory."""
    session = get_session(db_path)
    try:
        row = RegimeLogRow(
            timestamp=entry.timestamp,
            regime_type=entry.regime_type,
            confidence=entry.confidence,
            vix_proxy=entry.vix_proxy,
            market_bias=entry.market_bias,
        )
        session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write regime_log: %s", exc)
        print(f"[memory_db] CRITICAL: regime_log write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_signal_log(entry: SignalLogEntry, db_path: str) -> None:
    """Append a signal record to session memory."""
    session = get_session(db_path)
    try:
        row = SignalLogRow(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            setup_type=entry.setup_type,
            score=entry.score,
            claude_decision=entry.claude_decision,
            reason=entry.reason,
        )
        session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write signal_log: %s", exc)
        print(f"[memory_db] CRITICAL: signal_log write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_trade_log(entry: TradeLogEntry, db_path: str) -> None:
    """Append a trade record to session memory."""
    session = get_session(db_path)
    try:
        row = TradeLogRow(
            timestamp=entry.timestamp,
            symbol=entry.symbol,
            entry_price=entry.entry_price,
            direction=entry.direction,
            status=entry.status,
            unrealized_pnl=entry.unrealized_pnl,
        )
        session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write trade_log: %s", exc)
        print(f"[memory_db] CRITICAL: trade_log write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_narrative(entry: IntradayNarrativeEntry, db_path: str) -> None:
    """Append a narrative entry to session memory."""
    session = get_session(db_path)
    try:
        row = IntradayNarrativeRow(
            timestamp=entry.timestamp,
            narrative_text=entry.narrative_text,
        )
        session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write intraday_narrative: %s", exc)
        print(f"[memory_db] CRITICAL: narrative write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_symbol_character(entry: SymbolCharacterEntry, db_path: str) -> None:
    """Upsert a symbol character profile in session memory."""
    session = get_session(db_path)
    try:
        stmt = sqlite_insert(SymbolCharacterRow).values(
            symbol=entry.symbol,
            signal_count_today=entry.signal_count_today,
            direction_consistency=entry.direction_consistency,
            last_signal_result=entry.last_signal_result,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "signal_count_today": stmt.excluded.signal_count_today,
                "direction_consistency": stmt.excluded.direction_consistency,
                "last_signal_result": stmt.excluded.last_signal_result,
            },
        )
        session.execute(stmt)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write symbol_character for %s: %s", entry.symbol, exc)
        print(f"[memory_db] CRITICAL: symbol_character write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_peak_pnl(entry: PositionPeakPnL, db_path: str) -> None:
    """Upsert the high-water mark of unrealized P&L for a symbol."""
    session = get_session(db_path)
    try:
        stmt = sqlite_insert(PositionPeakPnLRow).values(
            symbol=entry.symbol,
            peak_unrealized_pnl=entry.peak_unrealized_pnl,
            peak_at=entry.peak_at,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol"],
            set_={
                "peak_unrealized_pnl": stmt.excluded.peak_unrealized_pnl,
                "peak_at": stmt.excluded.peak_at,
            },
        )
        session.execute(stmt)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write peak_pnl for %s: %s", entry.symbol, exc)
        print(f"[memory_db] CRITICAL: peak_pnl write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def get_peak_pnl(symbol: str, db_path: str) -> PositionPeakPnL | None:
    """Read the peak P&L record for a symbol, or None if not tracked yet."""
    session = get_session(db_path)
    try:
        row = (
            session.query(PositionPeakPnLRow)
            .filter(PositionPeakPnLRow.symbol == symbol)
            .first()
        )
        if row is None:
            return None
        return PositionPeakPnL(
            symbol=row.symbol,
            peak_unrealized_pnl=row.peak_unrealized_pnl,
            peak_at=_ensure_utc(row.peak_at),
        )
    except Exception as exc:  # noqa: BLE001
        logger.critical("Failed to read peak_pnl for %s: %s", symbol, exc)
        print(f"[memory_db] CRITICAL: peak_pnl read failed: {exc}", file=sys.stderr)
        return None
    finally:
        session.close()


# ── Strategic Memory Writes ───────────────────────────────────────────────────

def write_setup_performance(entry: SetupPerformanceEntry, db_path: str) -> None:
    """Append a setup performance record to strategic memory."""
    session = get_session(db_path)
    try:
        row = SetupPerformanceRow(
            setup_type=entry.setup_type,
            symbol=entry.symbol,
            regime_at_entry=entry.regime_at_entry,
            time_of_day_bucket=entry.time_of_day_bucket,
            win=entry.win,
            pnl=entry.pnl,
            hold_duration_minutes=entry.hold_duration_minutes,
            date=entry.date,
        )
        session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write setup_performance: %s", exc)
        print(f"[memory_db] CRITICAL: setup_performance write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_regime_transition(entry: RegimeTransitionEntry, db_path: str) -> None:
    """Record or increment a regime transition in strategic memory."""
    session = get_session(db_path)
    try:
        stmt = sqlite_insert(RegimeTransitionRow).values(
            from_regime=entry.from_regime,
            to_regime=entry.to_regime,
            duration_minutes=entry.duration_minutes,
            count=entry.count,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["from_regime", "to_regime"],
            set_={
                "count": RegimeTransitionRow.__table__.c.count + stmt.excluded.count,
                "duration_minutes": stmt.excluded.duration_minutes,
            },
        )
        session.execute(stmt)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write regime_transition: %s", exc)
        print(f"[memory_db] CRITICAL: regime_transition write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_veto_pattern(entry: VetoPatternEntry, db_path: str) -> None:
    """Record or increment a veto pattern in strategic memory."""
    session = get_session(db_path)
    try:
        stmt = sqlite_insert(VetoPatternRow).values(
            veto_reason=entry.veto_reason,
            setup_type=entry.setup_type,
            count=entry.count,
            last_seen=entry.last_seen,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["veto_reason", "setup_type"],
            set_={
                "count": VetoPatternRow.__table__.c.count + stmt.excluded.count,
                "last_seen": stmt.excluded.last_seen,
            },
        )
        session.execute(stmt)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write veto_pattern: %s", exc)
        print(f"[memory_db] CRITICAL: veto_pattern write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_weekly_performance(entry: WeeklyPerformanceEntry, db_path: str) -> None:
    """Upsert weekly performance for a setup type in strategic memory."""
    session = get_session(db_path)
    try:
        stmt = sqlite_insert(WeeklyPerformanceRow).values(
            week=entry.week,
            setup_type=entry.setup_type,
            trades=entry.trades,
            win_rate=entry.win_rate,
            avg_pnl=entry.avg_pnl,
            sharpe=entry.sharpe,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["week", "setup_type"],
            set_={
                "trades": stmt.excluded.trades,
                "win_rate": stmt.excluded.win_rate,
                "avg_pnl": stmt.excluded.avg_pnl,
                "sharpe": stmt.excluded.sharpe,
            },
        )
        session.execute(stmt)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write weekly_performance: %s", exc)
        print(f"[memory_db] CRITICAL: weekly_performance write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_symbol_behavior(entry: SymbolLearnedBehaviorEntry, db_path: str) -> None:
    """Upsert learned symbol behavior in strategic memory."""
    session = get_session(db_path)
    try:
        stmt = sqlite_insert(SymbolLearnedBehaviorRow).values(
            symbol=entry.symbol,
            setup_type=entry.setup_type,
            optimal_rsi_entry=entry.optimal_rsi_entry,
            avg_reversion_depth=entry.avg_reversion_depth,
            avg_bounce_magnitude=entry.avg_bounce_magnitude,
            sample_size=entry.sample_size,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["symbol", "setup_type"],
            set_={
                "optimal_rsi_entry": stmt.excluded.optimal_rsi_entry,
                "avg_reversion_depth": stmt.excluded.avg_reversion_depth,
                "avg_bounce_magnitude": stmt.excluded.avg_bounce_magnitude,
                "sample_size": stmt.excluded.sample_size,
            },
        )
        session.execute(stmt)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write symbol_behavior for %s: %s", entry.symbol, exc)
        print(f"[memory_db] CRITICAL: symbol_behavior write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


def write_claude_calibration(entry: ClaudeCalibrationEntry, db_path: str) -> None:
    """Append a Claude calibration record to strategic memory."""
    session = get_session(db_path)
    try:
        row = ClaudeCalibrationRow(
            date=entry.date,
            confidence_stated=entry.confidence_stated,
            outcome=entry.outcome,
            setup_type=entry.setup_type,
        )
        session.add(row)
        session.commit()
    except Exception as exc:  # noqa: BLE001
        session.rollback()
        logger.critical("Failed to write claude_calibration: %s", exc)
        print(f"[memory_db] CRITICAL: claude_calibration write failed: {exc}", file=sys.stderr)
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# READ HELPERS — Return Pydantic models
# ══════════════════════════════════════════════════════════════════════════════

def get_session_context(db_path: str) -> SessionContext:
    """
    Build a complete SessionContext from today's session memory.

    Aggregates regime history, signals, trades, narrative, and symbol
    characters into a single Pydantic model for prompt injection.
    """
    now = datetime.now(timezone.utc)
    session = get_session(db_path)
    try:
        regime_rows = session.query(RegimeLogRow).order_by(RegimeLogRow.timestamp).all()
        regime_history = [
            RegimeLogEntry(
                timestamp=_ensure_utc(r.timestamp),
                regime_type=r.regime_type,
                confidence=r.confidence,
                vix_proxy=r.vix_proxy,
                market_bias=r.market_bias,
            )
            for r in regime_rows
        ]

        signal_rows = session.query(SignalLogRow).order_by(SignalLogRow.timestamp).all()
        signals_today = [
            SignalLogEntry(
                timestamp=_ensure_utc(s.timestamp),
                symbol=s.symbol,
                setup_type=s.setup_type,
                score=s.score,
                claude_decision=s.claude_decision,
                reason=s.reason,
            )
            for s in signal_rows
        ]

        trade_rows = session.query(TradeLogRow).order_by(TradeLogRow.timestamp).all()
        trades_today = [
            TradeLogEntry(
                timestamp=_ensure_utc(t.timestamp),
                symbol=t.symbol,
                entry_price=t.entry_price,
                direction=t.direction,
                status=t.status,
                unrealized_pnl=t.unrealized_pnl,
            )
            for t in trade_rows
        ]

        narrative_rows = (
            session.query(IntradayNarrativeRow)
            .order_by(IntradayNarrativeRow.timestamp.desc())
            .first()
        )
        narrative = narrative_rows.narrative_text if narrative_rows else ""

        char_rows = session.query(SymbolCharacterRow).all()
        symbol_characters = [
            SymbolCharacterEntry(
                symbol=c.symbol,
                signal_count_today=c.signal_count_today,
                direction_consistency=c.direction_consistency,
                last_signal_result=c.last_signal_result,
            )
            for c in char_rows
        ]

        return SessionContext(
            regime_history=regime_history,
            signals_today=signals_today,
            trades_today=trades_today,
            narrative=narrative,
            symbol_characters=symbol_characters,
            as_of=now,
        )
    finally:
        session.close()


def get_strategic_context(
    db_path: str,
    setup_type: SetupType | None = None,
    symbol: str | None = None,
    regime: MarketRegime | None = None,
) -> StrategicContext:
    """
    Build a StrategicContext from strategic memory, optionally filtered.

    Filters setup_performance and related tables by the given setup_type,
    symbol, and/or regime to return only relevant historical data.
    """
    now = datetime.now(timezone.utc)
    session = get_session(db_path)
    try:
        # Setup performance — filtered
        perf_query = session.query(SetupPerformanceRow)
        if setup_type is not None:
            perf_query = perf_query.filter_by(setup_type=setup_type)
        if symbol is not None:
            perf_query = perf_query.filter_by(symbol=symbol)
        if regime is not None:
            perf_query = perf_query.filter_by(regime_at_entry=regime)
        perf_rows = perf_query.order_by(SetupPerformanceRow.date.desc()).limit(100).all()
        setup_performance = [
            SetupPerformanceEntry(
                setup_type=p.setup_type,
                symbol=p.symbol,
                regime_at_entry=p.regime_at_entry,
                time_of_day_bucket=p.time_of_day_bucket,
                win=p.win,
                pnl=p.pnl,
                hold_duration_minutes=p.hold_duration_minutes,
                date=p.date,
            )
            for p in perf_rows
        ]

        # Regime transitions — all
        trans_rows = session.query(RegimeTransitionRow).all()
        regime_transitions = [
            RegimeTransitionEntry(
                from_regime=t.from_regime,
                to_regime=t.to_regime,
                duration_minutes=t.duration_minutes,
                count=t.count,
            )
            for t in trans_rows
        ]

        # Veto patterns — filtered by setup_type if given
        veto_query = session.query(VetoPatternRow)
        if setup_type is not None:
            veto_query = veto_query.filter_by(setup_type=setup_type)
        veto_rows = veto_query.order_by(VetoPatternRow.count.desc()).limit(20).all()
        relevant_veto_patterns = [
            VetoPatternEntry(
                veto_reason=v.veto_reason,
                setup_type=v.setup_type,
                count=v.count,
                last_seen=_ensure_utc(v.last_seen),
            )
            for v in veto_rows
        ]

        # Weekly performance — filtered by setup_type, last 12 weeks
        weekly_query = session.query(WeeklyPerformanceRow)
        if setup_type is not None:
            weekly_query = weekly_query.filter_by(setup_type=setup_type)
        weekly_rows = (
            weekly_query.order_by(WeeklyPerformanceRow.week.desc()).limit(12).all()
        )
        weekly_trend = [
            WeeklyPerformanceEntry(
                week=w.week,
                setup_type=w.setup_type,
                trades=w.trades,
                win_rate=w.win_rate,
                avg_pnl=w.avg_pnl,
                sharpe=w.sharpe,
            )
            for w in weekly_rows
        ]

        # Symbol behavior — single match
        symbol_behavior = None
        if symbol is not None and setup_type is not None:
            sb_row = (
                session.query(SymbolLearnedBehaviorRow)
                .filter_by(symbol=symbol, setup_type=setup_type)
                .first()
            )
            if sb_row is not None:
                symbol_behavior = SymbolLearnedBehaviorEntry(
                    symbol=sb_row.symbol,
                    setup_type=sb_row.setup_type,
                    optimal_rsi_entry=sb_row.optimal_rsi_entry,
                    avg_reversion_depth=sb_row.avg_reversion_depth,
                    avg_bounce_magnitude=sb_row.avg_bounce_magnitude,
                    sample_size=sb_row.sample_size,
                )

        # Claude calibration — filtered by setup_type, last 50
        cal_query = session.query(ClaudeCalibrationRow)
        if setup_type is not None:
            cal_query = cal_query.filter_by(setup_type=setup_type)
        cal_rows = cal_query.order_by(ClaudeCalibrationRow.date.desc()).limit(50).all()
        claude_calibration = [
            ClaudeCalibrationEntry(
                date=c.date,
                confidence_stated=c.confidence_stated,
                outcome=c.outcome,
                setup_type=c.setup_type,
            )
            for c in cal_rows
        ]

        return StrategicContext(
            setup_performance=setup_performance,
            regime_transitions=regime_transitions,
            relevant_veto_patterns=relevant_veto_patterns,
            weekly_trend=weekly_trend,
            symbol_behavior=symbol_behavior,
            claude_calibration=claude_calibration,
            as_of=now,
        )
    finally:
        session.close()


def get_similar_trades(
    db_path: str,
    symbol: str,
    regime: str | None = None,
    setup_type: str | None = None,
    top_k: int = 5,
) -> list[SetupPerformanceEntry]:
    """
    Retrieve the most relevant past trades for RAG-style context injection.

    Scoring heuristic (higher = more relevant):
      - Same symbol:     +3
      - Same regime:     +2
      - Same setup_type: +1

    Returns at most *top_k* entries, sorted by relevance then recency.
    """
    session = get_session(db_path)
    try:
        # Fetch candidates: same symbol OR same regime (at least one must match)
        query = session.query(SetupPerformanceRow)
        if regime is not None:
            query = query.filter(
                (SetupPerformanceRow.symbol == symbol)
                | (SetupPerformanceRow.regime_at_entry == regime)
            )
        else:
            query = query.filter(SetupPerformanceRow.symbol == symbol)

        # Limit to a reasonable candidate pool before scoring
        rows = query.order_by(SetupPerformanceRow.date.desc()).limit(200).all()

        if not rows:
            return []

        # Score and sort
        scored: list[tuple[int, str, SetupPerformanceRow]] = []
        for row in rows:
            score = 0
            if row.symbol == symbol:
                score += 3
            if regime is not None and row.regime_at_entry == regime:
                score += 2
            if setup_type is not None and row.setup_type == setup_type:
                score += 1
            scored.append((score, row.date, row))

        # Sort by score desc, then date desc
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

        results: list[SetupPerformanceEntry] = []
        for _, _, row in scored[:top_k]:
            results.append(
                SetupPerformanceEntry(
                    setup_type=row.setup_type,
                    symbol=row.symbol,
                    regime_at_entry=row.regime_at_entry,
                    time_of_day_bucket=row.time_of_day_bucket,
                    win=row.win,
                    pnl=row.pnl,
                    hold_duration_minutes=row.hold_duration_minutes,
                    date=row.date,
                )
            )
        return results
    except Exception as exc:  # noqa: BLE001
        logger.critical("Failed to get_similar_trades for %s: %s", symbol, exc)
        print(f"[memory_db] CRITICAL: get_similar_trades failed: {exc}", file=sys.stderr)
        return []
    finally:
        session.close()
