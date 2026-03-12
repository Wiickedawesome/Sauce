"""
tests/test_session_boot.py — Tests for Agent 0 (Session Boot).

Verifies:
  - BootContext is returned with correct fields.
  - Session memory is reset (or not) based on day boundary.
  - Calendar events and suppression are wired through.
  - AuditEvent with event_type="session_boot" is logged.

All external dependencies are mocked — no real DB, calendar, or config.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from sauce.core.schemas import (
    AuditEvent,
    BootContext,
    EconomicEvent,
    StrategicContext,
)

_NOW = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)


def _make_strategic_context() -> StrategicContext:
    return StrategicContext(as_of=_NOW)


def _make_calendar_event() -> EconomicEvent:
    return EconomicEvent(
        date=_NOW,
        event_type="FOMC",
        description="FOMC Rate Decision",
    )


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
# Tests — happy path
# ---------------------------------------------------------------------------

class TestSessionBootHappyPath:
    """Agent 0 returns valid BootContext under normal conditions."""

    def test_returns_boot_context(self, mock_settings):
        strat_ctx = _make_strategic_context()
        with (
            patch("sauce.agents.session_boot.get_settings", return_value=mock_settings),
            patch("sauce.agents.session_boot.reset_session_memory_if_new_day", return_value=True),
            patch("sauce.agents.session_boot.get_events_for_date", return_value=[]),
            patch("sauce.agents.session_boot.get_strategic_context", return_value=strat_ctx),
            patch("sauce.agents.session_boot.is_suppressed", return_value=False),
            patch("sauce.agents.session_boot.log_event") as mock_log,
        ):
            from sauce.agents.session_boot import run
            result = asyncio.run(run("loop-001"))

        assert isinstance(result, BootContext)
        assert result.was_reset is True
        assert result.calendar_events == []
        assert result.strategic_context is strat_ctx
        assert result.is_suppressed is False
        assert result.as_of is not None

        # Audit event logged
        mock_log.assert_called_once()
        event = mock_log.call_args[0][0]
        assert isinstance(event, AuditEvent)
        assert event.event_type == "session_boot"
        assert event.loop_id == "loop-001"

    def test_no_reset_when_same_day(self, mock_settings):
        strat_ctx = _make_strategic_context()
        with (
            patch("sauce.agents.session_boot.get_settings", return_value=mock_settings),
            patch("sauce.agents.session_boot.reset_session_memory_if_new_day", return_value=False),
            patch("sauce.agents.session_boot.get_events_for_date", return_value=[]),
            patch("sauce.agents.session_boot.get_strategic_context", return_value=strat_ctx),
            patch("sauce.agents.session_boot.is_suppressed", return_value=False),
            patch("sauce.agents.session_boot.log_event"),
        ):
            from sauce.agents.session_boot import run
            result = asyncio.run(run("loop-001"))

        assert result.was_reset is False


# ---------------------------------------------------------------------------
# Tests — calendar events
# ---------------------------------------------------------------------------

class TestSessionBootCalendar:
    """Calendar events are wired through to BootContext."""

    def test_calendar_events_passed_through(self, mock_settings):
        strat_ctx = _make_strategic_context()
        ev = _make_calendar_event()
        with (
            patch("sauce.agents.session_boot.get_settings", return_value=mock_settings),
            patch("sauce.agents.session_boot.reset_session_memory_if_new_day", return_value=False),
            patch("sauce.agents.session_boot.get_events_for_date", return_value=[ev]),
            patch("sauce.agents.session_boot.get_strategic_context", return_value=strat_ctx),
            patch("sauce.agents.session_boot.is_suppressed", return_value=False),
            patch("sauce.agents.session_boot.log_event"),
        ):
            from sauce.agents.session_boot import run
            result = asyncio.run(run("loop-001"))

        assert len(result.calendar_events) == 1
        assert result.calendar_events[0].event_type == "FOMC"


# ---------------------------------------------------------------------------
# Tests — suppression
# ---------------------------------------------------------------------------

class TestSessionBootSuppression:
    """Suppression flag is wired through to BootContext."""

    def test_suppressed_true(self, mock_settings):
        strat_ctx = _make_strategic_context()
        with (
            patch("sauce.agents.session_boot.get_settings", return_value=mock_settings),
            patch("sauce.agents.session_boot.reset_session_memory_if_new_day", return_value=False),
            patch("sauce.agents.session_boot.get_events_for_date", return_value=[]),
            patch("sauce.agents.session_boot.get_strategic_context", return_value=strat_ctx),
            patch("sauce.agents.session_boot.is_suppressed", return_value=True),
            patch("sauce.agents.session_boot.log_event"),
        ):
            from sauce.agents.session_boot import run
            result = asyncio.run(run("loop-001"))

        assert result.is_suppressed is True


# ---------------------------------------------------------------------------
# Tests — audit payload correctness
# ---------------------------------------------------------------------------

class TestSessionBootAudit:
    """AuditEvent payload includes correct boot metadata."""

    def test_audit_payload_fields(self, mock_settings):
        strat_ctx = _make_strategic_context()
        ev = _make_calendar_event()
        with (
            patch("sauce.agents.session_boot.get_settings", return_value=mock_settings),
            patch("sauce.agents.session_boot.reset_session_memory_if_new_day", return_value=True),
            patch("sauce.agents.session_boot.get_events_for_date", return_value=[ev]),
            patch("sauce.agents.session_boot.get_strategic_context", return_value=strat_ctx),
            patch("sauce.agents.session_boot.is_suppressed", return_value=True),
            patch("sauce.agents.session_boot.log_event") as mock_log,
        ):
            from sauce.agents.session_boot import run
            asyncio.run(run("loop-audit"))

        event = mock_log.call_args[0][0]
        assert event.payload["was_reset"] is True
        assert event.payload["calendar_event_count"] == 1
        assert event.payload["is_suppressed"] is True

        # db_path passed correctly
        db_path_arg = mock_log.call_args[0][1]
        assert db_path_arg == mock_settings.db_path
