"""Economic calendar loader — hardcoded FOMC, CPI, NFP dates.

Pure Python, no external API calls, fully deterministic.
Used by Agent 0 (session boot) to load today's events, and by
the setup disqualifier logic to suppress signals near major releases.

Plan references
───────────────
- Section 3, Layer 3:  "2-hour window before major releases"
- Setup 1 disqualifiers: "Major economic event in next 90 minutes"
- Setup 2 disqualifiers: "FOMC, CPI, or NFP within 48 hours"
- Agent 0 spec:          "Loads the economic calendar for the day"
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sauce.core.schemas import EconomicEvent, EconomicEventType

logger = logging.getLogger(__name__)

# ── Hardcoded event schedule ──────────────────────────────────────────────────
# All times are UTC.  FOMC decisions typically at 18:00 UTC (14:00 ET).
# CPI and NFP releases are typically at 12:30 UTC (08:30 ET).
# Dates sourced from the official Federal Reserve, BLS, and BLS Employment
# release calendars.  Expand annually.

_EVENTS: list[dict[str, str]] = [
    # ── 2024 FOMC ──
    {"date": "2024-01-31T19:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2024-03-20T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2024-05-01T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2024-06-12T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2024-07-31T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2024-09-18T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2024-11-07T19:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2024-12-18T19:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    # ── 2025 FOMC ──
    {"date": "2025-01-29T19:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2025-03-19T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2025-05-07T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2025-06-18T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2025-07-30T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2025-09-17T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2025-10-29T18:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    {"date": "2025-12-17T19:00:00Z", "type": "FOMC", "desc": "FOMC Rate Decision"},
    # ── 2024 CPI ──
    {"date": "2024-01-11T13:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-02-13T13:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-03-12T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-04-10T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-05-15T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-06-12T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-07-11T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-08-14T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-09-11T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-10-10T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-11-13T13:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2024-12-11T13:30:00Z", "type": "CPI", "desc": "CPI Release"},
    # ── 2025 CPI ──
    {"date": "2025-01-15T13:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-02-12T13:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-03-12T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-04-10T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-05-13T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-06-11T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-07-15T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-08-12T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-09-10T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-10-14T12:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-11-12T13:30:00Z", "type": "CPI", "desc": "CPI Release"},
    {"date": "2025-12-10T13:30:00Z", "type": "CPI", "desc": "CPI Release"},
    # ── 2024 NFP ──
    {"date": "2024-01-05T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-02-02T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-03-08T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-04-05T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-05-03T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-06-07T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-07-05T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-08-02T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-09-06T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-10-04T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-11-01T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2024-12-06T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    # ── 2025 NFP ──
    {"date": "2025-01-10T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-02-07T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-03-07T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-04-04T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-05-02T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-06-06T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-07-03T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-08-01T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-09-05T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-10-03T12:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-11-07T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
    {"date": "2025-12-05T13:30:00Z", "type": "NFP", "desc": "Nonfarm Payrolls"},
]

# Pre-parsed, sorted list of EconomicEvent objects (built once at import time).
EVENTS: list[EconomicEvent] = sorted(
    [
        EconomicEvent(
            date=datetime.fromisoformat(e["date"]),
            event_type=e["type"],  # type: ignore[arg-type]
            description=e["desc"],
        )
        for e in _EVENTS
    ],
    key=lambda ev: ev.date,
)


# ── Public API ────────────────────────────────────────────────────────────────


def get_events_for_date(as_of: datetime) -> list[EconomicEvent]:
    """Return all economic events scheduled on the same calendar day (UTC).

    Used by Agent 0 at session boot to load today's calendar.
    """
    target_date = as_of.astimezone(timezone.utc).date()
    return [ev for ev in EVENTS if ev.date.date() == target_date]


def get_upcoming_events(
    as_of: datetime,
    window: timedelta,
) -> list[EconomicEvent]:
    """Return events within *window* of *as_of* (forward-looking only)."""
    as_of_utc = as_of.astimezone(timezone.utc)
    cutoff = as_of_utc + window
    return [ev for ev in EVENTS if as_of_utc <= ev.date <= cutoff]


def is_near_major_event(as_of: datetime, window_minutes: int = 90) -> bool:
    """True if a major event is within *window_minutes* of *as_of*.

    Default 90 minutes — Setup 1 (mean reversion) disqualifier.
    """
    return len(get_upcoming_events(as_of, timedelta(minutes=window_minutes))) > 0


def is_major_event_within_hours(as_of: datetime, hours: int = 48) -> bool:
    """True if a major event is within *hours* of *as_of*.

    Default 48 hours — Setup 2 (equity trend pullback) disqualifier.
    """
    return len(get_upcoming_events(as_of, timedelta(hours=hours))) > 0


def is_suppressed(as_of: datetime, suppression_minutes: int = 120) -> bool:
    """True if a major event is within the suppression window.

    Plan Layer 3: "suppress certain setups in the 2-hour window before
    major releases."  General-purpose guard — broader than per-setup vetos.
    """
    return len(get_upcoming_events(as_of, timedelta(minutes=suppression_minutes))) > 0
