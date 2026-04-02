"""US equity market calendar helpers.

Provides lightweight NYSE-style trading day logic without an external
calendar dependency. The holiday set is intentionally conservative and is
used for eligibility checks and historical lookback sizing.
"""

from __future__ import annotations

import math
from calendar import monthrange
from datetime import date, timedelta
from functools import cache


def _nth_weekday_of_month(year: int, month: int, weekday: int, occurrence: int) -> date:
    first_day = date(year, month, 1)
    offset = (weekday - first_day.weekday()) % 7
    day = 1 + offset + ((occurrence - 1) * 7)
    return date(year, month, day)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    last_day = date(year, month, monthrange(year, month)[1])
    offset = (last_day.weekday() - weekday) % 7
    return last_day - timedelta(days=offset)


def _observed_holiday(day: date) -> date:
    if day.weekday() == 5:
        return day - timedelta(days=1)
    if day.weekday() == 6:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    offset = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * offset) // 451
    month = (h + offset - 7 * m + 114) // 31
    day = ((h + offset - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@cache
def us_equity_market_holidays(year: int) -> frozenset[date]:
    good_friday = _easter_sunday(year) - timedelta(days=2)
    holidays = {
        _observed_holiday(date(year, 1, 1)),
        _nth_weekday_of_month(year, 1, 0, 3),
        _nth_weekday_of_month(year, 2, 0, 3),
        good_friday,
        _last_weekday_of_month(year, 5, 0),
        _observed_holiday(date(year, 6, 19)),
        _observed_holiday(date(year, 7, 4)),
        _nth_weekday_of_month(year, 9, 0, 1),
        _nth_weekday_of_month(year, 11, 3, 4),
        _observed_holiday(date(year, 12, 25)),
    }
    return frozenset(holidays)


def is_us_equity_trading_day(day: date) -> bool:
    return day.weekday() < 5 and day not in us_equity_market_holidays(day.year)


def calendar_days_for_equity_bars(
    as_of: date,
    bars: int,
    timeframe_minutes: int,
    *,
    extra_sessions: int = 2,
) -> int:
    """Return a calendar-day lookback that respects weekends and holidays."""
    trading_sessions = max(1, math.ceil((bars * timeframe_minutes) / 390))
    target_sessions = trading_sessions + max(extra_sessions, 0)

    cursor = as_of
    sessions_seen = 0
    while sessions_seen < target_sessions:
        if is_us_equity_trading_day(cursor):
            sessions_seen += 1
        cursor -= timedelta(days=1)

    return max((as_of - cursor).days + 1, 5)
