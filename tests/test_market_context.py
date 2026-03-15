"""
tests/test_market_context.py — Tests for Agent 1 (Market Context Builder).

Verifies:
  - Happy-path returns valid MarketContext with correct fields.
  - Degraded context on MarketDataError or insufficient bars.
  - Regime transition detection and recording.
  - is_dead / is_suppressed flag logic.
  - AuditEvent with event_type="market_context" is logged.
  - Helper functions: _last_float, _degraded_context.

All external dependencies are mocked — no real DB, market data, or indicators.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sauce.adapters.market_data import MarketDataError
from sauce.core.regime import RegimeDuration
from sauce.core.schemas import (
    AuditEvent,
    BootContext,
    EconomicEvent,
    Indicators,
    IntradayNarrativeEntry,
    MarketContext,
    RegimeLogEntry,
    RegimeTransitionEntry,
    SessionContext,
    StrategicContext,
)

_NOW = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
_LOOP_ID = "loop-mc-001"


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _make_boot_ctx(
    *,
    is_suppressed: bool = False,
    calendar_events: list | None = None,
) -> BootContext:
    return BootContext(
        was_reset=True,
        calendar_events=calendar_events or [],
        strategic_context=StrategicContext(as_of=_NOW),
        is_suppressed=is_suppressed,
        as_of=_NOW,
    )


def _make_regime(
    regime_type: str = "TRENDING_UP",
    confidence: float = 0.85,
    timestamp: datetime = _NOW,
) -> RegimeLogEntry:
    return RegimeLogEntry(
        timestamp=timestamp,
        regime_type=regime_type,
        confidence=confidence,
    )


def _make_session_ctx(
    regime_history: list[RegimeLogEntry] | None = None,
) -> SessionContext:
    return SessionContext(
        regime_history=regime_history or [],
        signals_today=[],
        trades_today=[],
        narrative="",
        symbol_characters=[],
        as_of=_NOW,
    )


def _make_duration(
    regime_type: str = "TRENDING_UP",
    active_minutes: float = 45.0,
    aging_out: bool = False,
) -> RegimeDuration:
    return RegimeDuration(
        regime_type=regime_type,
        active_minutes=active_minutes,
        aging_out=aging_out,
    )


def _make_narrative() -> IntradayNarrativeEntry:
    return IntradayNarrativeEntry(
        timestamp=_NOW,
        narrative_text="Market is trending up with moderate volume.",
    )


def _make_spy_df(rows: int = 20) -> pd.DataFrame:
    """Build a minimal SPY OHLCV DataFrame with *rows* entries."""
    return pd.DataFrame({
        "open": [400.0 + i for i in range(rows)],
        "high": [401.0 + i for i in range(rows)],
        "low": [399.0 + i for i in range(rows)],
        "close": [400.5 + i for i in range(rows)],
        "volume": [1_000_000 + i * 1000 for i in range(rows)],
        "timestamp": [_NOW + timedelta(minutes=30 * i) for i in range(rows)],
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_settings(tmp_path):
    s = MagicMock()
    s.session_memory_db_path = str(tmp_path / "session_memory.db")
    s.strategic_memory_db_path = str(tmp_path / "strategic_memory.db")
    s.db_path = str(tmp_path / "audit.db")
    return s


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MC = "sauce.agents.market_context"


def _run(coro):
    """Run a coroutine blocking."""
    return asyncio.run(coro)


# ===========================================================================
# Tests — helper functions
# ===========================================================================

class TestDegradedContext:
    """Verify _degraded_context returns correct structure."""

    def test_degraded_fields(self):
        from sauce.agents.market_context import _degraded_context

        boot_ctx = _make_boot_ctx(is_suppressed=True)
        result = _degraded_context(boot_ctx, _NOW)

        assert isinstance(result, MarketContext)
        assert result.is_dead is True
        assert result.regime.regime_type == "DEAD"
        assert result.regime.confidence == 0.0
        assert result.is_suppressed is True
        assert result.calendar_events == []
        assert "degraded" in result.narrative.narrative_text.lower()

    def test_calendar_events_passthrough(self):
        from sauce.agents.market_context import _degraded_context

        ev = EconomicEvent(date=_NOW, event_type="FOMC", description="Rate decision")
        boot_ctx = _make_boot_ctx(calendar_events=[ev])
        result = _degraded_context(boot_ctx, _NOW)

        assert len(result.calendar_events) == 1
        assert result.calendar_events[0].event_type == "FOMC"


# ===========================================================================
# Tests — run() happy path
# ===========================================================================

class TestMarketContextHappyPath:
    """Agent 1 returns valid MarketContext under normal conditions."""

    def test_returns_market_context(self, mock_settings):
        regime = _make_regime()
        session_ctx = _make_session_ctx(regime_history=[regime])
        duration = _make_duration()
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=regime),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=duration),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.log_event") as mock_log,
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            result = _run(run(_LOOP_ID, _make_boot_ctx()))

        assert isinstance(result, MarketContext)
        assert result.regime.regime_type == "TRENDING_UP"
        assert result.regime_duration_minutes == 45.0
        assert result.regime_aging_out is False
        assert result.is_dead is False
        assert result.is_suppressed is False
        assert result.narrative is narrative

        mock_log.assert_called_once()
        event = mock_log.call_args[0][0]
        assert isinstance(event, AuditEvent)
        assert event.event_type == "market_context"
        assert event.payload["status"] == "ok"

    def test_duration_none_when_missing(self, mock_settings):
        regime = _make_regime()
        session_ctx = _make_session_ctx(regime_history=[regime])
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=regime),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=None),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.log_event"),
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            result = _run(run(_LOOP_ID, _make_boot_ctx()))

        assert result.regime_duration_minutes is None
        assert result.regime_aging_out is False


# ===========================================================================
# Tests — degraded path
# ===========================================================================

class TestMarketContextDegraded:
    """Agent 1 returns degraded context when SPY data is unavailable."""

    def test_degraded_on_market_data_error(self, mock_settings):
        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}.log_event") as mock_log,
        ):
            md.MarketDataError = MarketDataError
            md.get_history.side_effect = MarketDataError("API down")
            from sauce.agents.market_context import run

            result = _run(run(_LOOP_ID, _make_boot_ctx()))

        assert isinstance(result, MarketContext)
        assert result.is_dead is True
        assert result.regime.regime_type == "DEAD"
        assert result.regime.confidence == 0.0

        event = mock_log.call_args[0][0]
        assert event.payload["status"] == "degraded"
        assert event.payload["reason"] == "spy_data_unavailable"

    def test_degraded_on_insufficient_bars(self, mock_settings):
        short_df = _make_spy_df(rows=5)  # < _MIN_BARS_REQUIRED (10)

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}.log_event") as mock_log,
        ):
            md.MarketDataError = MarketDataError
            md.get_history.return_value = short_df
            from sauce.agents.market_context import run

            result = _run(run(_LOOP_ID, _make_boot_ctx()))

        assert result.is_dead is True
        assert result.regime.regime_type == "DEAD"

        event = mock_log.call_args[0][0]
        assert event.payload["status"] == "degraded"
        assert event.payload["reason"] == "insufficient_bars"
        assert event.payload["bars"] == 5


# ===========================================================================
# Tests — regime transition
# ===========================================================================

class TestRegimeTransition:
    """Regime transition detection and write logic."""

    def test_transition_detected_and_written(self, mock_settings):
        prev_regime = _make_regime("RANGING", timestamp=_NOW - timedelta(minutes=30))
        curr_regime = _make_regime("TRENDING_UP", timestamp=_NOW)
        session_ctx = _make_session_ctx(
            regime_history=[prev_regime, curr_regime],
        )
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=curr_regime),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=_make_duration()),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.write_regime_transition") as mock_wrt,
            patch(f"{_MC}.log_event"),
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            _run(run(_LOOP_ID, _make_boot_ctx()))

        mock_wrt.assert_called_once()
        transition = mock_wrt.call_args[0][0]
        assert isinstance(transition, RegimeTransitionEntry)
        assert transition.from_regime == "RANGING"
        assert transition.to_regime == "TRENDING_UP"
        assert transition.duration_minutes == 30.0
        assert transition.count == 1

    def test_no_transition_when_same_regime(self, mock_settings):
        r1 = _make_regime("TRENDING_UP", timestamp=_NOW - timedelta(minutes=30))
        r2 = _make_regime("TRENDING_UP", timestamp=_NOW)
        session_ctx = _make_session_ctx(regime_history=[r1, r2])
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=r2),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=_make_duration()),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.write_regime_transition") as mock_wrt,
            patch(f"{_MC}.log_event"),
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            _run(run(_LOOP_ID, _make_boot_ctx()))

        mock_wrt.assert_not_called()

    def test_no_transition_when_single_regime_entry(self, mock_settings):
        regime = _make_regime()
        session_ctx = _make_session_ctx(regime_history=[regime])
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=regime),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=_make_duration()),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.write_regime_transition") as mock_wrt,
            patch(f"{_MC}.log_event"),
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            _run(run(_LOOP_ID, _make_boot_ctx()))

        mock_wrt.assert_not_called()


# ===========================================================================
# Tests — flags
# ===========================================================================

class TestMarketContextFlags:
    """Verify is_dead and is_suppressed flag logic."""

    def test_is_dead_when_regime_dead(self, mock_settings):
        regime = _make_regime("DEAD", confidence=0.0)
        session_ctx = _make_session_ctx(regime_history=[regime])
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=regime),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=None),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.log_event"),
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            result = _run(run(_LOOP_ID, _make_boot_ctx()))

        assert result.is_dead is True

    def test_suppressed_passthrough(self, mock_settings):
        regime = _make_regime()
        session_ctx = _make_session_ctx(regime_history=[regime])
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=regime),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=_make_duration()),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.log_event"),
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            result = _run(run(_LOOP_ID, _make_boot_ctx(is_suppressed=True)))

        assert result.is_suppressed is True

    def test_calendar_events_passthrough(self, mock_settings):
        ev = EconomicEvent(date=_NOW, event_type="CPI", description="CPI Release")
        regime = _make_regime()
        session_ctx = _make_session_ctx(regime_history=[regime])
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=regime),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=_make_duration()),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.log_event"),
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            result = _run(run(_LOOP_ID, _make_boot_ctx(calendar_events=[ev])))

        assert len(result.calendar_events) == 1
        assert result.calendar_events[0].event_type == "CPI"


# ===========================================================================
# Tests — audit logging
# ===========================================================================

class TestMarketContextAudit:
    """Verify audit event payload structure."""

    def test_audit_payload_keys(self, mock_settings):
        regime = _make_regime()
        session_ctx = _make_session_ctx(regime_history=[regime])
        duration = _make_duration()
        narrative = _make_narrative()
        spy_df = _make_spy_df()

        with (
            patch(f"{_MC}.get_settings", return_value=mock_settings),
            patch(f"{_MC}.market_data") as md,
            patch(f"{_MC}._compute_spy_indicators", return_value=Indicators()),
            patch(f"{_MC}.classify_regime", return_value=regime),
            patch(f"{_MC}.write_regime_log"),
            patch(f"{_MC}.get_session_context", return_value=session_ctx),
            patch(f"{_MC}.compute_regime_duration", return_value=duration),
            patch(f"{_MC}.build_narrative", return_value=narrative),
            patch(f"{_MC}.write_narrative"),
            patch(f"{_MC}.log_event") as mock_log,
        ):
            md.get_history.return_value = spy_df
            from sauce.agents.market_context import run

            _run(run(_LOOP_ID, _make_boot_ctx()))

        event = mock_log.call_args[0][0]
        assert event.loop_id == _LOOP_ID
        assert event.event_type == "market_context"
        payload = event.payload
        expected_keys = {
            "status", "regime", "confidence",
            "duration_minutes", "aging_out",
            "is_dead", "is_suppressed",
        }
        assert expected_keys <= set(payload.keys())
        assert payload["status"] == "ok"
        assert payload["regime"] == "TRENDING_UP"
        assert payload["confidence"] == 0.85
        assert payload["duration_minutes"] == 45.0
        assert payload["aging_out"] is False
        assert payload["is_dead"] is False
        assert payload["is_suppressed"] is False

        # db_path passed correctly
        assert mock_log.call_args[0][1] == mock_settings.db_path
