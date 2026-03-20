"""
db.py — Sauce database layer (5-table schema).

Tables:
  1. trades       — completed trades with P&L (append-only)
  2. positions    — open positions with trailing state (mutable)
  3. signals      — every scoring result (append-only)
  4. daily_stats  — daily aggregates (upsert)
  5. instrument_meta — per-instrument config/regime cache

Uses the existing engine/session infrastructure from adapters.db.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from sauce.adapters.db import Base, get_session
from sauce.strategy import Position, SignalResult

logger = logging.getLogger(__name__)


# ── Table Definitions ─────────────────────────────────────────────────────────


class TradeRow(Base):
    """Completed trade record. Append-only."""

    __tablename__ = "trades"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    trade_id: str = Column(String(36), nullable=False, unique=True, index=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    side: str = Column(String(8), nullable=False)
    qty: float = Column(Float, nullable=False)
    entry_price: float = Column(Float, nullable=False)
    exit_price: float = Column(Float, nullable=False)
    realized_pnl: float = Column(Float, nullable=False)
    strategy_name: str = Column(String(64), nullable=False)
    exit_trigger: str = Column(String(32), nullable=False)  # e.g. "hard_stop", "profit_target"
    entry_time: datetime = Column(DateTime, nullable=False)
    exit_time: datetime = Column(DateTime, nullable=False)
    hold_hours: float = Column(Float, nullable=False)
    broker_order_id: str | None = Column(String(64), nullable=True)


class PositionRow(Base):
    """Open position with trailing-stop state. Mutable — updated each cycle."""

    __tablename__ = "positions"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    position_id: str = Column(String(36), nullable=False, unique=True, index=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    asset_class: str = Column(String(16), nullable=False, default="crypto")
    qty: float = Column(Float, nullable=False)
    entry_price: float = Column(Float, nullable=False)
    high_water_price: float = Column(Float, nullable=False, default=0.0)
    trailing_stop_price: float | None = Column(Float, nullable=True)
    trailing_active: bool = Column(Boolean, nullable=False, default=False)
    entry_time: datetime = Column(DateTime, nullable=False)
    broker_order_id: str | None = Column(String(64), nullable=True)
    strategy_name: str = Column(String(64), nullable=False)
    stop_loss_price: float = Column(Float, nullable=False, default=0.0)
    profit_target_price: float = Column(Float, nullable=False, default=0.0)
    status: str = Column(String(12), nullable=False, default="open", index=True)


class SignalLogRow(Base):
    """Every scoring result. Append-only."""

    __tablename__ = "signal_log"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(20), nullable=False, index=True)
    side: str = Column(String(8), nullable=False)
    score: int = Column(Integer, nullable=False)
    threshold: int = Column(Integer, nullable=False)
    fired: bool = Column(Boolean, nullable=False)
    rsi_14: float | None = Column(Float, nullable=True)
    macd_hist: float | None = Column(Float, nullable=True)
    bb_pct: float | None = Column(Float, nullable=True)
    volume_ratio: float | None = Column(Float, nullable=True)
    regime: str = Column(String(16), nullable=False)
    strategy_name: str = Column(String(64), nullable=False)
    timestamp: datetime = Column(DateTime, nullable=False)


class DailySummaryRow(Base):
    """Daily aggregates. Upserted once per day."""

    __tablename__ = "daily_summary"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    date: str = Column(String(10), nullable=False, unique=True, index=True)
    loop_runs: int = Column(Integer, nullable=False, default=0)
    signals_fired: int = Column(Integer, nullable=False, default=0)
    signals_skipped: int = Column(Integer, nullable=False, default=0)
    orders_placed: int = Column(Integer, nullable=False, default=0)
    trades_closed: int = Column(Integer, nullable=False, default=0)
    realized_pnl_usd: float = Column(Float, nullable=False, default=0.0)
    starting_equity: float = Column(Float, nullable=False, default=0.0)
    ending_equity: float = Column(Float, nullable=False, default=0.0)
    regime: str = Column(String(16), nullable=False, default="neutral")
    updated_at: datetime = Column(DateTime, nullable=False)


class InstrumentMetaRow(Base):
    """Per-instrument configuration and cached data."""

    __tablename__ = "instrument_meta"

    id: int = Column(Integer, primary_key=True, autoincrement=True)
    symbol: str = Column(String(20), nullable=False, unique=True, index=True)
    asset_class: str = Column(String(16), nullable=False)  # crypto | equity
    strategy_name: str = Column(String(64), nullable=False)
    last_signal_score: int | None = Column(Integer, nullable=True)
    last_signal_time: datetime | None = Column(DateTime, nullable=True)
    extra: str = Column(Text, nullable=False, default="{}")  # JSON for ad-hoc metadata


# ── Signal Persistence ────────────────────────────────────────────────────────


def log_signal(signal: SignalResult, db_path: str | None = None) -> None:
    """Append a signal scoring result. Never raises."""
    session = get_session(db_path)
    try:
        row = SignalLogRow(
            symbol=signal.symbol,
            side=signal.side,
            score=signal.score,
            threshold=signal.threshold,
            fired=signal.fired,
            rsi_14=signal.rsi_14,
            macd_hist=signal.macd_hist,
            bb_pct=signal.bb_pct,
            volume_ratio=signal.volume_ratio,
            regime=signal.regime,
            strategy_name=signal.strategy_name,
            timestamp=signal.timestamp,
        )
        session.add(row)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to log signal for %s: %s", signal.symbol, exc)
    finally:
        session.close()


# ── Position Persistence ──────────────────────────────────────────────────────


def save_position(position: Position, db_path: str | None = None) -> None:
    """Insert a new open position."""
    session = get_session(db_path)
    try:
        row = PositionRow(
            position_id=position.id,
            symbol=position.symbol,
            asset_class=position.asset_class,
            qty=position.qty,
            entry_price=position.entry_price,
            high_water_price=position.high_water_price,
            trailing_stop_price=position.trailing_stop_price,
            trailing_active=position.trailing_active,
            entry_time=position.entry_time,
            broker_order_id=position.broker_order_id,
            strategy_name=position.strategy_name,
            stop_loss_price=position.stop_loss_price,
            profit_target_price=position.profit_target_price,
            status="open",
        )
        session.add(row)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to save position %s: %s", position.id, exc)
    finally:
        session.close()


def update_position(position: Position, db_path: str | None = None) -> None:
    """Update trailing-stop state for an open position."""
    session = get_session(db_path)
    try:
        row = session.query(PositionRow).filter_by(position_id=position.id, status="open").first()
        if row is None:
            logger.warning("Position %s not found for update", position.id)
            return
        row.high_water_price = position.high_water_price
        row.trailing_stop_price = position.trailing_stop_price
        row.trailing_active = position.trailing_active
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to update position %s: %s", position.id, exc)
    finally:
        session.close()


def close_position(position_id: str, db_path: str | None = None) -> None:
    """Mark a position as closed."""
    session = get_session(db_path)
    try:
        row = session.query(PositionRow).filter_by(position_id=position_id, status="open").first()
        if row is None:
            return
        row.status = "closed"
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to close position %s: %s", position_id, exc)
    finally:
        session.close()


def load_open_positions(db_path: str | None = None) -> list[Position]:
    """Load all open positions from DB into Position dataclasses."""
    session = get_session(db_path)
    try:
        rows = session.query(PositionRow).filter_by(status="open").all()
        return [
            Position(
                id=row.position_id,
                symbol=row.symbol,
                asset_class=row.asset_class,
                qty=row.qty,
                entry_price=row.entry_price,
                high_water_price=row.high_water_price,
                trailing_stop_price=row.trailing_stop_price,
                trailing_active=row.trailing_active,
                entry_time=row.entry_time.replace(tzinfo=UTC) if row.entry_time.tzinfo is None else row.entry_time,
                broker_order_id=row.broker_order_id,
                strategy_name=row.strategy_name,
                stop_loss_price=row.stop_loss_price,
                profit_target_price=row.profit_target_price,
            )
            for row in rows
        ]
    finally:
        session.close()


# ── Trade Persistence ─────────────────────────────────────────────────────────


def log_trade(
    position: Position,
    exit_price: float,
    exit_trigger: str,
    exit_time: datetime | None = None,
    db_path: str | None = None,
) -> None:
    """Record a completed trade. Append-only."""
    if exit_time is None:
        exit_time = datetime.now(UTC)

    hold_hours = (exit_time - position.entry_time).total_seconds() / 3600
    realized_pnl = (exit_price - position.entry_price) * position.qty

    session = get_session(db_path)
    try:
        row = TradeRow(
            trade_id=position.id,
            symbol=position.symbol,
            side="sell",
            qty=position.qty,
            entry_price=position.entry_price,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            strategy_name=position.strategy_name,
            exit_trigger=exit_trigger,
            entry_time=position.entry_time,
            exit_time=exit_time,
            hold_hours=hold_hours,
            broker_order_id=position.broker_order_id,
        )
        session.add(row)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to log trade for %s: %s", position.symbol, exc)
    finally:
        session.close()


# ── Daily Stats ───────────────────────────────────────────────────────────────


def upsert_daily_stats(
    date: str,
    db_path: str | None = None,
    **fields: object,
) -> None:
    """Insert or update daily stats. Additive counters accumulate."""
    session = get_session(db_path)
    try:
        existing = session.query(DailySummaryRow).filter_by(date=date).first()
        if existing is None:
            row = DailySummaryRow(date=date, updated_at=datetime.now(UTC))
            session.add(row)
        else:
            row = existing

        additive = {
            "loop_runs",
            "signals_fired",
            "signals_skipped",
            "orders_placed",
            "trades_closed",
        }
        for key, value in fields.items():
            if not hasattr(row, key):
                continue
            if key in additive and existing is not None:
                current = getattr(row, key, 0) or 0
                # Both current and value come from dynamic sources; cast via str/int
                setattr(row, key, int(str(current)) + int(str(value)))
            else:
                setattr(row, key, value)

        row.updated_at = datetime.now(UTC)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to upsert daily stats for %s: %s", date, exc)
    finally:
        session.close()


def get_daily_pnl(date: str, db_path: str | None = None) -> float:
    """Return today's realized P&L in USD."""
    session = get_session(db_path)
    try:
        row = session.query(DailySummaryRow).filter_by(date=date).first()
        if row is None:
            return 0.0
        return float(row.realized_pnl_usd)
    finally:
        session.close()


def get_daily_regime(date: str, db_path: str | None = None) -> str | None:
    """Return the regime cached in daily_summary for the given date, or None.

    Returns None only when no cycle has run today yet (no row or loop_runs == 0).
    """
    session = get_session(db_path)
    try:
        row = session.query(DailySummaryRow).filter_by(date=date).first()
        if row is None or (row.loop_runs or 0) == 0:
            return None  # no cycle has run today yet
        return str(row.regime)
    finally:
        session.close()


# ── Instrument Meta ───────────────────────────────────────────────────────────


def upsert_instrument_meta(
    symbol: str,
    asset_class: str,
    strategy_name: str,
    last_signal_score: int | None = None,
    last_signal_time: datetime | None = None,
    extra: dict[str, Any] | None = None,
    db_path: str | None = None,
) -> None:
    """Insert or update instrument metadata."""
    session = get_session(db_path)
    try:
        row = session.query(InstrumentMetaRow).filter_by(symbol=symbol).first()
        if row is None:
            row = InstrumentMetaRow(
                symbol=symbol,
                asset_class=asset_class,
                strategy_name=strategy_name,
            )
            session.add(row)
        row.last_signal_score = last_signal_score
        row.last_signal_time = last_signal_time
        if extra is not None:
            row.extra = json.dumps(extra)
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error("Failed to upsert instrument meta for %s: %s", symbol, exc)
    finally:
        session.close()
