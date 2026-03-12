"""Tests for sauce.core.calendar — economic calendar loader."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sauce.core.calendar import (
    EVENTS,
    get_events_for_date,
    get_upcoming_events,
    is_major_event_within_hours,
    is_near_major_event,
    is_suppressed,
)
from sauce.core.schemas import EconomicEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


# Known event times (from hardcoded _EVENTS list)
FOMC_JAN31 = _utc(2024, 1, 31, 19, 0)   # FOMC 2024-01-31 19:00 UTC
CPI_JAN11 = _utc(2024, 1, 11, 13, 30)    # CPI  2024-01-11 13:30 UTC
NFP_JAN05 = _utc(2024, 1, 5, 13, 30)     # NFP  2024-01-05 13:30 UTC
CPI_JUN12 = _utc(2024, 6, 12, 12, 30)    # CPI  2024-06-12 12:30 UTC
FOMC_JUN12 = _utc(2024, 6, 12, 18, 0)    # FOMC 2024-06-12 18:00 UTC
CPI_MAR12 = _utc(2024, 3, 12, 12, 30)    # CPI  2024-03-12 12:30 UTC


# ===== EVENTS list validation =====

class TestEventsListIntegrity:
    """Validate the pre-parsed EVENTS list."""

    def test_total_event_count(self) -> None:
        assert len(EVENTS) == 64  # 16 FOMC + 24 CPI + 24 NFP

    def test_fomc_count(self) -> None:
        fomc = [e for e in EVENTS if e.event_type == "FOMC"]
        assert len(fomc) == 16  # 8 per year × 2 years

    def test_cpi_count(self) -> None:
        cpi = [e for e in EVENTS if e.event_type == "CPI"]
        assert len(cpi) == 24  # 12 per year × 2 years

    def test_nfp_count(self) -> None:
        nfp = [e for e in EVENTS if e.event_type == "NFP"]
        assert len(nfp) == 24  # 12 per year × 2 years

    def test_events_sorted_by_date(self) -> None:
        dates = [e.date for e in EVENTS]
        assert dates == sorted(dates)

    def test_all_events_are_economic_event(self) -> None:
        for ev in EVENTS:
            assert isinstance(ev, EconomicEvent)

    def test_all_events_have_utc_timezone(self) -> None:
        for ev in EVENTS:
            assert ev.date.tzinfo is not None
            assert ev.date.utcoffset() == timedelta(0)

    def test_event_types_are_valid(self) -> None:
        valid = {"FOMC", "CPI", "NFP"}
        for ev in EVENTS:
            assert ev.event_type in valid


# ===== get_events_for_date =====

class TestGetEventsForDate:
    """Tests for get_events_for_date()."""

    def test_single_event_on_date(self) -> None:
        # 2024-01-31 has one FOMC event
        result = get_events_for_date(FOMC_JAN31)
        assert len(result) == 1
        assert result[0].event_type == "FOMC"
        assert result[0].date == FOMC_JAN31

    def test_multiple_events_on_same_date(self) -> None:
        # 2024-06-12 has both CPI (12:30) and FOMC (18:00)
        result = get_events_for_date(CPI_JUN12)
        assert len(result) == 2
        types = {e.event_type for e in result}
        assert types == {"CPI", "FOMC"}

    def test_no_events_on_date(self) -> None:
        # Pick a date with no known events
        no_event = _utc(2024, 2, 15, 10, 0)
        result = get_events_for_date(no_event)
        assert result == []

    def test_date_matching_ignores_time(self) -> None:
        # Passing midnight of a known event date should still find
        midnight = _utc(2024, 1, 31, 0, 0)
        result = get_events_for_date(midnight)
        assert len(result) == 1
        assert result[0].event_type == "FOMC"

    def test_nfp_event_found(self) -> None:
        result = get_events_for_date(NFP_JAN05)
        assert len(result) == 1
        assert result[0].event_type == "NFP"

    def test_naive_datetime_treated_correctly(self) -> None:
        # Pass a naive datetime — function should still work
        naive = datetime(2024, 1, 11, 10, 0)
        result = get_events_for_date(naive)
        assert len(result) == 1
        assert result[0].event_type == "CPI"


# ===== get_upcoming_events =====

class TestGetUpcomingEvents:
    """Tests for get_upcoming_events()."""

    def test_window_includes_event(self) -> None:
        # 1 hour before FOMC, window = 2 hours → should find it
        as_of = FOMC_JAN31 - timedelta(hours=1)
        result = get_upcoming_events(as_of, timedelta(hours=2))
        fomc_hits = [e for e in result if e.date == FOMC_JAN31]
        assert len(fomc_hits) == 1

    def test_window_excludes_event(self) -> None:
        # 3 hours before FOMC, window = 1 hour → should NOT find it
        as_of = FOMC_JAN31 - timedelta(hours=3)
        result = get_upcoming_events(as_of, timedelta(hours=1))
        fomc_hits = [e for e in result if e.date == FOMC_JAN31]
        assert len(fomc_hits) == 0

    def test_event_at_exact_as_of(self) -> None:
        # as_of exactly at event time → event should be included (as_of <= ev.date)
        result = get_upcoming_events(FOMC_JAN31, timedelta(hours=1))
        fomc_hits = [e for e in result if e.date == FOMC_JAN31]
        assert len(fomc_hits) == 1

    def test_event_at_exact_cutoff(self) -> None:
        # as_of such that FOMC is exactly at cutoff boundary
        as_of = FOMC_JAN31 - timedelta(hours=2)
        result = get_upcoming_events(as_of, timedelta(hours=2))
        fomc_hits = [e for e in result if e.date == FOMC_JAN31]
        assert len(fomc_hits) == 1

    def test_large_window_captures_multiple(self) -> None:
        # From start of 2024, window covers many events
        as_of = _utc(2024, 1, 1, 0, 0)
        result = get_upcoming_events(as_of, timedelta(days=365))
        assert len(result) > 10

    def test_past_events_excluded(self) -> None:
        # as_of after all 2024-01-05 NFP → it should not appear
        as_of = NFP_JAN05 + timedelta(hours=1)
        result = get_upcoming_events(as_of, timedelta(hours=1))
        nfp_hits = [e for e in result if e.date == NFP_JAN05]
        assert len(nfp_hits) == 0

    def test_zero_window(self) -> None:
        # Zero-length window at event time → should include exact match
        result = get_upcoming_events(FOMC_JAN31, timedelta(0))
        fomc_hits = [e for e in result if e.date == FOMC_JAN31]
        assert len(fomc_hits) == 1


# ===== is_near_major_event =====

class TestIsNearMajorEvent:
    """Tests for is_near_major_event()."""

    def test_within_default_window(self) -> None:
        # 89 minutes before FOMC → within 90min default
        as_of = FOMC_JAN31 - timedelta(minutes=89)
        assert is_near_major_event(as_of) is True

    def test_outside_default_window(self) -> None:
        # 91 minutes before FOMC → outside 90min default
        as_of = FOMC_JAN31 - timedelta(minutes=91)
        assert is_near_major_event(as_of) is False

    def test_exactly_at_event(self) -> None:
        assert is_near_major_event(FOMC_JAN31) is True

    def test_custom_window(self) -> None:
        # 30 min before, with 60 min window → True
        as_of = CPI_JAN11 - timedelta(minutes=30)
        assert is_near_major_event(as_of, window_minutes=60) is True

    def test_custom_window_too_narrow(self) -> None:
        # 30 min before, with 20 min window → False
        as_of = CPI_JAN11 - timedelta(minutes=30)
        assert is_near_major_event(as_of, window_minutes=20) is False

    def test_after_event_still_near(self) -> None:
        # 30 min after FOMC → function only looks at future events, so False
        as_of = FOMC_JAN31 + timedelta(minutes=30)
        assert is_near_major_event(as_of) is False

    def test_no_events_nearby(self) -> None:
        # Mid-February, no events nearby
        as_of = _utc(2024, 2, 15, 12, 0)
        assert is_near_major_event(as_of) is False


# ===== is_major_event_within_hours =====

class TestIsMajorEventWithinHours:
    """Tests for is_major_event_within_hours()."""

    def test_within_default_48h(self) -> None:
        # 47 hours before FOMC → within 48h
        as_of = FOMC_JAN31 - timedelta(hours=47)
        assert is_major_event_within_hours(as_of) is True

    def test_outside_default_48h(self) -> None:
        # 49 hours before FOMC → outside 48h
        as_of = FOMC_JAN31 - timedelta(hours=49)
        assert is_major_event_within_hours(as_of) is False

    def test_exactly_at_event(self) -> None:
        assert is_major_event_within_hours(FOMC_JAN31) is True

    def test_custom_hours(self) -> None:
        # 5 hours before with 6h window → True
        as_of = CPI_MAR12 - timedelta(hours=5)
        assert is_major_event_within_hours(as_of, hours=6) is True

    def test_custom_hours_too_narrow(self) -> None:
        # 5 hours before with 4h window → False
        as_of = CPI_MAR12 - timedelta(hours=5)
        assert is_major_event_within_hours(as_of, hours=4) is False

    def test_no_event_nearby(self) -> None:
        as_of = _utc(2024, 2, 20, 12, 0)
        assert is_major_event_within_hours(as_of) is False


# ===== is_suppressed =====

class TestIsSuppressed:
    """Tests for is_suppressed()."""

    def test_within_default_suppression(self) -> None:
        # 119 minutes before FOMC → within 120min
        as_of = FOMC_JAN31 - timedelta(minutes=119)
        assert is_suppressed(as_of) is True

    def test_outside_default_suppression(self) -> None:
        # 121 minutes before FOMC → outside 120min
        as_of = FOMC_JAN31 - timedelta(minutes=121)
        assert is_suppressed(as_of) is False

    def test_exactly_at_event(self) -> None:
        assert is_suppressed(FOMC_JAN31) is True

    def test_custom_suppression_minutes(self) -> None:
        # 45 min before NFP with 60 min window → True
        as_of = NFP_JAN05 - timedelta(minutes=45)
        assert is_suppressed(as_of, suppression_minutes=60) is True

    def test_custom_suppression_too_narrow(self) -> None:
        # 45 min before NFP with 30 min window → False
        as_of = NFP_JAN05 - timedelta(minutes=45)
        assert is_suppressed(as_of, suppression_minutes=30) is False

    def test_after_event_suppressed(self) -> None:
        # 60 minutes after FOMC → function only looks at future events, so False
        as_of = FOMC_JAN31 + timedelta(minutes=60)
        assert is_suppressed(as_of) is False

    def test_no_suppression_far_from_events(self) -> None:
        as_of = _utc(2024, 2, 20, 12, 0)
        assert is_suppressed(as_of) is False


# ===== Edge cases =====

class TestCalendarEdgeCases:
    """Cross-cutting edge cases."""

    def test_end_of_year_boundary(self) -> None:
        # Last event of 2024 should exist and be before 2025-01-01
        events_2024 = [e for e in EVENTS if e.date.year == 2024]
        assert len(events_2024) > 0
        last_2024 = max(e.date for e in events_2024)
        assert last_2024 < _utc(2025, 1, 1)

    def test_start_of_year_boundary(self) -> None:
        # First event of 2025 should exist and be after 2024-12-31
        events_2025 = [e for e in EVENTS if e.date.year == 2025]
        assert len(events_2025) > 0
        first_2025 = min(e.date for e in events_2025)
        assert first_2025 > _utc(2024, 12, 31)

    def test_same_day_events_returned_ordered(self) -> None:
        # 2024-06-12 has CPI 12:30 and FOMC 18:00
        result = get_events_for_date(CPI_JUN12)
        assert len(result) == 2
        assert result[0].date <= result[1].date

    def test_events_list_is_not_mutated(self) -> None:
        original_len = len(EVENTS)
        _ = get_events_for_date(FOMC_JAN31)
        _ = get_upcoming_events(FOMC_JAN31, timedelta(hours=1))
        assert len(EVENTS) == original_len
