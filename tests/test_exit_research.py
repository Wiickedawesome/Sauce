"""
tests/test_exit_research.py — Tests for the exit research agent.

Covers all 5 exit conditions, peak P&L tracking, and hold behavior.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from sauce.agents import exit_research
from sauce.agents.exit_research import (
    TRAILING_STOP_PULLBACK_PCT,
    RSI_OVERBOUGHT_THRESHOLD,
    STALE_HOLD_HOURS,
    STALE_HOLD_MIN_GAIN_PCT,
    run,
)
from sauce.core.schemas import (
    ExitSignal,
    Indicators,
    PositionPeakPnL,
    PriceReference,
)
from sauce.memory.db import get_peak_pnl, write_peak_pnl

_NOW = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
_LOOP_ID = "test-loop-001"


def _quote(symbol: str = "AAPL", mid: float = 100.0) -> PriceReference:
    return PriceReference(
        symbol=symbol,
        bid=mid - 0.05,
        ask=mid + 0.05,
        mid=mid,
        as_of=_NOW,
    )


def _position(
    symbol: str = "AAPL",
    qty: float = 10.0,
    avg_entry_price: float = 95.0,
    unrealized_pl: float = 50.0,
) -> dict:
    return {
        "symbol": symbol,
        "qty": qty,
        "avg_entry_price": avg_entry_price,
        "unrealized_pl": unrealized_pl,
        "market_value": avg_entry_price * qty + unrealized_pl,
    }


# ─── Hold — no conditions met ────────────────────────────────────────────────


class TestHoldBehavior:
    def test_hold_when_no_conditions_met(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=50.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"
        assert result.symbol == "AAPL"

    def test_hold_on_zero_qty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        result = asyncio.run(
            run(
                position=_position(qty=0.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"
        assert "Zero-qty" in result.reason


# ─── Condition 1: Trailing stop ──────────────────────────────────────────────


class TestTrailingStop:
    def test_exit_on_trailing_stop(self, tmp_path, monkeypatch):
        session_db = str(tmp_path / "session.db")
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", session_db)

        # Seed a peak P&L of 100
        write_peak_pnl(
            PositionPeakPnL(symbol="AAPL", peak_unrealized_pnl=100.0, peak_at=_NOW),
            session_db,
        )

        # Current P&L is 60 → 40% pullback from 100 (> 30% threshold)
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=60.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "exit"
        assert "Trailing stop" in result.reason
        assert result.urgency == "high"

    def test_hold_when_pullback_below_threshold(self, tmp_path, monkeypatch):
        session_db = str(tmp_path / "session.db")
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", session_db)

        # Seed a peak P&L of 100
        write_peak_pnl(
            PositionPeakPnL(symbol="AAPL", peak_unrealized_pnl=100.0, peak_at=_NOW),
            session_db,
        )

        # Current P&L is 80 → only 20% pullback (< 30% threshold)
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=80.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"

    def test_no_trailing_stop_when_peak_is_zero(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))

        # First call with negative P&L — peak stays 0, no trailing stop
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=-10.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"


# ─── Condition 2: Regime flip ────────────────────────────────────────────────


class TestRegimeFlip:
    @pytest.mark.parametrize("regime", ["TRENDING_DOWN", "VOLATILE"])
    def test_exit_on_adverse_regime(self, tmp_path, monkeypatch, regime):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        result = asyncio.run(
            run(
                position=_position(),
                quote=_quote(),
                indicators=Indicators(),
                regime=regime,
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "exit"
        assert "Regime flip" in result.reason
        assert regime in result.reason

    @pytest.mark.parametrize("regime", ["TRENDING_UP", "RANGING", "DEAD"])
    def test_hold_on_neutral_regime(self, tmp_path, monkeypatch, regime):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        result = asyncio.run(
            run(
                position=_position(),
                quote=_quote(),
                indicators=Indicators(),
                regime=regime,
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"


# ─── Condition 3: RSI overbought ─────────────────────────────────────────────


class TestRSIOverbought:
    def test_exit_on_high_rsi(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        result = asyncio.run(
            run(
                position=_position(),
                quote=_quote(),
                indicators=Indicators(rsi_14=80.0),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "exit"
        assert "RSI overbought" in result.reason

    def test_hold_on_normal_rsi(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        result = asyncio.run(
            run(
                position=_position(),
                quote=_quote(),
                indicators=Indicators(rsi_14=60.0),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"

    def test_hold_when_rsi_is_none(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        result = asyncio.run(
            run(
                position=_position(),
                quote=_quote(),
                indicators=Indicators(rsi_14=None),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"


# ─── Condition 4: Stale position ─────────────────────────────────────────────


class TestStalePosition:
    def test_exit_on_stale_with_low_gain(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        now = datetime.now(timezone.utc)
        entry = now - timedelta(hours=72)  # 72 hours > 48 threshold
        # 10 shares at $95 = $950 cost basis, $1 gain = 0.105% < 1%
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=1.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
                entry_time=entry,
            )
        )
        assert result.action == "exit"
        assert "Stale position" in result.reason

    def test_hold_when_recent_position(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        now = datetime.now(timezone.utc)
        entry = now - timedelta(hours=12)  # 12 hours < 48 threshold
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=1.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
                entry_time=entry,
            )
        )
        assert result.action == "hold"

    def test_hold_when_good_gain(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        now = datetime.now(timezone.utc)
        entry = now - timedelta(hours=72)
        # 10 shares at $95 = $950, $50 gain = 5.26% > 1%
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=50.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
                entry_time=entry,
            )
        )
        assert result.action == "hold"

    def test_hold_when_no_entry_time(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        # No entry_time → stale check skipped
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=1.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
                entry_time=None,
            )
        )
        assert result.action == "hold"


# ─── Condition 5: ATR stop-loss breach ───────────────────────────────────────


class TestATRStop:
    def test_exit_on_atr_stop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        # entry=95, stop_loss_atr_multiple=2.0, ATR=1.5 → stop=95-3.0=92.0
        # mid=91.0 < 92.0 → exit
        result = asyncio.run(
            run(
                position=_position(avg_entry_price=95.0, unrealized_pl=-40.0),
                quote=_quote(mid=91.0),
                indicators=Indicators(atr_14=1.5),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "exit"
        assert "ATR stop hit" in result.reason
        assert result.urgency == "high"

    def test_hold_when_above_atr_stop(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        # entry=95, stop_loss_atr_multiple=2.0, ATR=1.5 → stop=92.0
        # mid=93.0 > 92.0 → hold
        result = asyncio.run(
            run(
                position=_position(avg_entry_price=95.0, unrealized_pl=-20.0),
                quote=_quote(mid=93.0),
                indicators=Indicators(atr_14=1.5),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"

    def test_atr_stop_skipped_when_no_atr(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", str(tmp_path / "session.db"))
        result = asyncio.run(
            run(
                position=_position(avg_entry_price=95.0, unrealized_pl=-40.0),
                quote=_quote(mid=91.0),
                indicators=Indicators(atr_14=None),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "hold"


# ─── Peak P&L tracking ──────────────────────────────────────────────────────


class TestPeakTracking:
    def test_peak_updated_when_new_high(self, tmp_path, monkeypatch):
        session_db = str(tmp_path / "session.db")
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", session_db)

        # Run with unrealized_pl=50 → should set peak to 50
        asyncio.run(
            run(
                position=_position(unrealized_pl=50.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        peak = get_peak_pnl("AAPL", session_db)
        assert peak is not None
        assert peak.peak_unrealized_pnl == 50.0

    def test_peak_not_downgraded(self, tmp_path, monkeypatch):
        session_db = str(tmp_path / "session.db")
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", session_db)

        # Seed peak at 100
        write_peak_pnl(
            PositionPeakPnL(symbol="AAPL", peak_unrealized_pnl=100.0, peak_at=_NOW),
            session_db,
        )

        # Run with lower P&L of 80 — peak should stay at 100
        asyncio.run(
            run(
                position=_position(unrealized_pl=80.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="TRENDING_UP",
                loop_id=_LOOP_ID,
            )
        )
        peak = get_peak_pnl("AAPL", session_db)
        assert peak is not None
        assert peak.peak_unrealized_pnl == 100.0


# ─── Condition priority (first match wins) ──────────────────────────────────


class TestConditionPriority:
    def test_trailing_stop_fires_before_regime_flip(self, tmp_path, monkeypatch):
        """Trailing stop (condition 1) is checked before regime flip (condition 2)."""
        session_db = str(tmp_path / "session.db")
        monkeypatch.setenv("SESSION_MEMORY_DB_PATH", session_db)

        write_peak_pnl(
            PositionPeakPnL(symbol="AAPL", peak_unrealized_pnl=100.0, peak_at=_NOW),
            session_db,
        )

        # Both trailing stop AND regime flip conditions met
        result = asyncio.run(
            run(
                position=_position(unrealized_pl=60.0),
                quote=_quote(),
                indicators=Indicators(),
                regime="VOLATILE",
                loop_id=_LOOP_ID,
            )
        )
        assert result.action == "exit"
        assert "Trailing stop" in result.reason
