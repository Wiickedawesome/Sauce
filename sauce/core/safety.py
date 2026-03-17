"""
core/safety.py — Global safety guards for the Sauce trading system.

Rules:
- TRADING_PAUSE is checked FIRST in every loop run. No exceptions.
- is_data_fresh() must be called before using ANY market data value.
- All safety decisions are logged as AuditEvent(event_type="safety_check").
- pause_trading() persists to DB — survives restarts. Only resume_trading() clears it.
- resume_trading() is manual only — it is never called automatically.
- check_daily_loss() returns False on ANY failure (fail-safe default).
- has_earnings_risk() returns True on API failure (fail-closed default).

Market hours note:
  US equities trade 09:30–16:00 Eastern Time, Monday–Friday.
  Crypto pairs ("/" in symbol) have no market-hours restriction.
"""

import logging
import sqlite3
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)

# SQLite json_extract requires 3.9.0+ — warn once at import time.
_sqlite_version = tuple(int(x) for x in sqlite3.sqlite_version.split("."))
if _sqlite_version < (3, 9, 0):
    logger.warning(
        "SQLite %s lacks json_extract (requires 3.9.0+). "
        "_is_paused_in_db() will always return False.",
        sqlite3.sqlite_version,
    )

# Eastern timezone used for US market-hours checks
_ET = ZoneInfo("America/New_York")

# US equity market open/close times (Eastern)
_MARKET_OPEN = time(9, 30, 0)
_MARKET_CLOSE = time(16, 0, 0)

# Minimum time-of-day before trusting last_equity for daily-loss calc (Finding 1.7).
# Alpaca resets last_equity at market open; before ~06:30 ET it may still carry
# the prior session’s value and produce a misleading daily P&L figure.
_MIN_DAILY_LOSS_CHECK_TIME = time(6, 30, 0)


def _now_et() -> datetime:
    """Return the current wall-clock time in US Eastern. Injectable for tests."""
    return datetime.now(_ET)


# ── Pause / Resume ────────────────────────────────────────────────────────────

def is_trading_paused(loop_id: str = "unset") -> bool:
    """
    Return True if trading is currently paused.

    Checks (in order, stops at first True):
      1. TRADING_PAUSE env var / config field (immediate, no DB needed).
      2. DB flag — looks for the most recent 'safety_check' AuditEvent
         with payload.paused set, and returns whether it was a pause or resume.
    """
    from sauce.adapters.db import log_event

    settings = get_settings()
    db_path = str(settings.db_path)

    # 1. Config-level pause (set via env var or .env file)
    if settings.trading_pause:
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="safety_check",
                payload={"check": "trading_pause", "result": True, "source": "config"},
            ),
            db_path=db_path,
        )
        return True

    # 2. DB-level pause (persists across restarts)
    if _is_paused_in_db(db_path=db_path):
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="safety_check",
                payload={"check": "trading_pause", "result": True, "source": "db"},
            ),
            db_path=db_path,
        )
        return True

    # Log pass-through so the audit trail always shows the check ran (IMP-01)
    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="safety_check",
            payload={"check": "trading_pause", "result": False},
        ),
        db_path=db_path,
    )
    return False


def _is_paused_in_db(db_path: str | None = None) -> bool:
    """
    Query the most recent safety_check event that has a 'paused' payload key.
    Returns True only if that most recent event records paused=1.

    This correctly handles resume_trading(): when resume writes paused=0,
    the most recent record shows 0 → unpaused.

    db_path: pass explicitly to avoid defaulting to the on-disk production DB
             during tests.  Callers should always obtain this from get_settings().
    """
    from sqlalchemy import text as sa_text

    from sauce.adapters.db import get_session

    if _sqlite_version < (3, 9, 0):
        return False

    resolved_path = db_path or str(get_settings().db_path)
    session = get_session(resolved_path)
    try:
        row = session.execute(
            sa_text(
                "SELECT json_extract(payload, '$.paused') "
                "FROM audit_events "
                "WHERE event_type = 'safety_check' "
                "AND json_extract(payload, '$.paused') IS NOT NULL "
                "ORDER BY timestamp DESC "
                "LIMIT 1"
            )
        ).fetchone()
    finally:
        session.close()

    if row is None:
        return False
    return bool(row[0])


def pause_trading(reason: str, loop_id: str = "manual") -> None:
    """
    Write a pause flag to the DB and log the reason.

    This persists across restarts. Only resume_trading() clears it.
    Never call this automatically — it requires human review to resume.
    """
    from sauce.adapters.db import log_event

    db_path = str(get_settings().db_path)
    logger.warning("Trading PAUSED. Reason: %s (loop_id=%s)", reason, loop_id)

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="safety_check",
            payload={"paused": True, "reason": reason, "action": "pause"},
        ),
        db_path=db_path,
    )


def resume_trading(loop_id: str = "manual") -> None:
    """
    Clear the DB-level pause flag. Manual operation only — never called
    by the loop automatically.

    After this call, is_trading_paused() will return False (assuming
    TRADING_PAUSE env var is also false).
    """
    from sauce.adapters.db import log_event

    db_path = str(get_settings().db_path)
    logger.info("Trading RESUMED (loop_id=%s)", loop_id)

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="safety_check",
            payload={"paused": False, "action": "resume"},
        ),
        db_path=db_path,
    )


# ── Data Freshness ────────────────────────────────────────────────────────────

def is_data_fresh(as_of: datetime, ttl_sec: int = 120) -> bool:
    """
    Return True if as_of is within ttl_sec seconds of now (UTC).

    Rule 2.4: Never use datetime.now() as a proxy for as_of.
    This function is the correct check — call it before using any market data.

    Parameters
    ----------
    as_of:   Timestamp from the data API (must be UTC-aware).
    ttl_sec: Maximum allowable age in seconds. Defaults to 120s (2 min).

    Returns
    -------
    True  → data is fresh enough to trust.
    False → data is stale; do not trade on it.
    """
    now = datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    return (now - as_of).total_seconds() <= ttl_sec


# ── Daily P&L Guard ───────────────────────────────────────────────────────────

def check_daily_loss(
    account: dict,
    loop_id: str = "unset",
    max_daily_loss_pct: float | None = None,
) -> bool:
    """
    Return True if today's loss is within the configured limit.
    Return False (block trading) if the daily loss limit is breached.

    Fails SAFE: any missing field or parse error returns False.

    The Alpaca account dict contains 'equity' and 'last_equity' as strings.
    Daily P&L = (equity - last_equity) / last_equity × 100%.

    Parameters
    ----------
    max_daily_loss_pct: Per-tier override for daily loss limit. When None,
                        falls back to settings.max_daily_loss_pct.
    """
    from sauce.adapters.db import log_event

    settings = get_settings()
    effective_max_daily_loss_pct = (
        max_daily_loss_pct if max_daily_loss_pct is not None else settings.max_daily_loss_pct
    )
    db_path = str(settings.db_path)

    # Finding 1.7: Skip daily loss check before 06:30 ET.
    # Alpaca resets last_equity at market open; checking before it settles
    # can produce a spurious large loss and block legitimate trading.
    now_et = _now_et()
    if now_et.time() < _MIN_DAILY_LOSS_CHECK_TIME:
        logger.info(
            "check_daily_loss: before %s ET (%s) — skipping stale last_equity check [loop_id=%s]",
            _MIN_DAILY_LOSS_CHECK_TIME.strftime("%H:%M"),
            now_et.strftime("%H:%M %Z"),
            loop_id,
        )
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="safety_check",
                payload={
                    "check": "daily_loss",
                    "result": True,
                    "reason": "pre-06:30 ET — last_equity not yet settled",
                },
            ),
            db_path=db_path,
        )
        return True

    try:
        equity = float(account.get("equity") or 0.0)
        last_equity = float(account.get("last_equity") or 0.0)

        if last_equity <= 0.0:
            # Cannot compute — default to blocking if we have no reference point
            logger.warning(
                "check_daily_loss: last_equity is 0 or missing — blocking as precaution"
            )
            log_event(
                AuditEvent(
                    loop_id=loop_id,
                    event_type="safety_check",
                    payload={
                        "check": "daily_loss",
                        "result": False,
                        "reason": "last_equity unavailable",
                    },
                ),
                db_path=db_path,
            )
            return False

        daily_pnl_pct = (equity - last_equity) / last_equity
        limit = -abs(effective_max_daily_loss_pct)  # e.g. -0.02

        ok = daily_pnl_pct >= limit

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="safety_check",
                payload={
                    "check": "daily_loss",
                    "result": ok,
                    "daily_pnl_pct": round(daily_pnl_pct * 100, 4),
                    "limit_pct": round(abs(effective_max_daily_loss_pct) * 100, 4),
                    "equity": equity,
                    "last_equity": last_equity,
                },
            ),
            db_path=db_path,
        )

        if not ok:
            pause_trading(
                f"Daily loss limit breached: {daily_pnl_pct:.2%} (limit: {limit:.2%})",
                loop_id=loop_id,
            )

        return ok

    except (TypeError, ValueError, KeyError) as exc:
        logger.error("check_daily_loss: error parsing account data: %s", exc)
        db_path = str(get_settings().db_path)
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="safety_check",
                payload={"check": "daily_loss", "result": False, "error": str(exc)},
            ),
            db_path=db_path,
        )
        return False


# ── NYSE holiday helpers (Finding 7.7) ───────────────────────────────────────

def _nyse_holidays(year: int) -> frozenset[date]:
    """
    Return the set of NYSE market holidays for the given year.

    Computed algorithmically from the official NYSE holiday schedule:
    New Year’s, MLK Day, Presidents’ Day, Good Friday, Memorial Day,
    Juneteenth, Independence Day, Labor Day, Thanksgiving, Christmas.

    Saturday holidays are observed the preceding Friday; Sunday holidays
    are observed the following Monday — matching NYSE practice.
    """
    import calendar
    from datetime import timedelta

    def _observed(d: date) -> date:
        """Shift weekend holidays to their NYSE-observed weekday."""
        if d.weekday() == 5:   # Saturday → Friday
            return d - timedelta(days=1)
        if d.weekday() == 6:   # Sunday → Monday
            return d + timedelta(days=1)
        return d

    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
        """Return the n-th occurrence (1-based) of weekday in the given month/year."""
        # weekday: 0=Mon, 6=Sun
        first = date(year, month, 1)
        delta = (weekday - first.weekday()) % 7
        first_occurrence = first + timedelta(days=delta)
        return first_occurrence + timedelta(weeks=n - 1)

    def _last_weekday(year: int, month: int, weekday: int) -> date:
        """Return the last occurrence of weekday in the given month."""
        last_day = date(year, month, calendar.monthrange(year, month)[1])
        delta = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=delta)

    def _good_friday(year: int) -> date:
        """Compute Good Friday via the anonymous Gregorian algorithm."""
        a = year % 19
        b, c = divmod(year, 100)
        d, e = divmod(b, 4)
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i, k = divmod(c, 4)
        l = (32 + 2 * e + 2 * i - h - k) % 7  # noqa: E741
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter_sunday = date(year, month, day)
        return easter_sunday - timedelta(days=2)

    holidays = {
        _observed(date(year, 1, 1)),                      # New Year’s Day
        _nth_weekday(year, 1, 0, 3),                      # MLK Day (3rd Mon Jan)
        _nth_weekday(year, 2, 0, 3),                      # Presidents’ Day (3rd Mon Feb)
        _good_friday(year),                               # Good Friday
        _last_weekday(year, 5, 0),                        # Memorial Day (last Mon May)
        _observed(date(year, 6, 19)),                     # Juneteenth
        _observed(date(year, 7, 4)),                      # Independence Day
        _nth_weekday(year, 9, 0, 1),                      # Labor Day (1st Mon Sep)
        _nth_weekday(year, 11, 3, 4),                     # Thanksgiving (4th Thu Nov)
        _observed(date(year, 12, 25)),                    # Christmas Day
    }
    return frozenset(holidays)


def _is_nyse_holiday(d: date) -> bool:
    """Return True if d is a NYSE market holiday."""
    return d in _nyse_holidays(d.year)


# ── Market Hours ──────────────────────────────────────────────────────────────

def check_market_hours(symbol: str = "", loop_id: str = "unset") -> bool:
    """
    Return True if the given symbol can be traded right now.

    - Crypto pairs (containing "/") → always tradeable (24/7).
    - Equity tickers → must be within US market hours (09:30–16:00 ET, Mon–Fri).

    Parameters
    ----------
    symbol:   Symbol to check. Empty string defaults to equity check.
    loop_id:  For audit logging.

    Returns
    -------
    True  → market is open for this symbol.
    False → market is closed; skip this symbol for this run.
    """
    from sauce.adapters.db import log_event

    # Crypto trades 24/7 on Alpaca
    if "/" in symbol:
        return True

    now_et = _now_et()
    reason = ""

    # Weekend check (0=Monday, 6=Sunday)
    if now_et.weekday() >= 5:
        reason = "weekend"
    # NYSE market holiday check (Finding 7.7)
    elif _is_nyse_holiday(now_et.date()):
        reason = "nyse_holiday"
    else:
        current_time = now_et.time()
        if not (_MARKET_OPEN <= current_time < _MARKET_CLOSE):
            reason = "outside_hours"

    if reason:
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="safety_check",
                symbol=symbol,
                payload={
                    "check": "market_hours",
                    "result": False,
                    "reason": reason,
                    "time_et": now_et.strftime("%H:%M %Z"),
                },
            ),
        )
        return False

    # Log pass-through so audit trail confirms the check ran
    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="safety_check",
            symbol=symbol,
            payload={
                "check": "market_hours",
                "result": True,
                "time_et": now_et.strftime("%H:%M %Z"),
            },
        ),
    )
    return True


# ── Earnings Proximity Guard (Finding 2.6) ────────────────────────────────────

# Keywords that indicate an earnings event in a news headline.
_EARNINGS_KEYWORDS = frozenset({
    "earnings",
    "quarterly results",
    "q1 results", "q2 results", "q3 results", "q4 results",
    "annual results",
    "eps",
    "profit report",
    "reports earnings",
    "beats estimates",
    "misses estimates",
})


def has_earnings_risk(symbol: str, loop_id: str = "unset") -> bool:
    """
    Return True if the symbol has a scheduled or recently-announced earnings
    event within the configured blackout window (earnings_blackout_days).

    Detection strategy: query Alpaca News for the symbol over the blackout
    window and scan headlines for earnings-related keywords. If the news
    API call fails at runtime, fails CLOSED (returns True) so uncertain
    situations block the trade rather than risking an earnings surprise.

    Called per-symbol in the loop eligibility filter BEFORE research.run().

    Parameters
    ----------
    symbol:   Trading symbol to check.
    loop_id:  For audit correlation.

    Returns
    -------
    True  → earnings risk detected; suppress signal for this window.
    False → no earnings risk detected (or News API unavailable).
    """
    from datetime import timedelta

    from sauce.adapters.db import log_event

    settings = get_settings()
    db_path = str(settings.db_path)
    blackout_days = settings.earnings_blackout_days

    if blackout_days <= 0:
        return False  # operator has disabled earnings blackout

    # Crypto pairs (e.g. BTC/USD) don't have earnings — skip the check entirely.
    if "/" in symbol:
        return False

    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(days=blackout_days)
    window_end = now_utc + timedelta(days=blackout_days)

    try:
        from alpaca.data.historical import NewsClient  # type: ignore[import-untyped]
        from alpaca.data.requests import NewsRequest  # type: ignore[import-untyped]

        client = NewsClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        request = NewsRequest(
            symbols=symbol,
            start=window_start,
            end=window_end,
            limit=20,
        )
        response = client.get_news(request)
        articles = response.news if hasattr(response, "news") else []

        for article in articles:
            headline = str(getattr(article, "headline", "")).lower()
            summary = str(getattr(article, "summary", "")).lower()
            combined = headline + " " + summary
            if any(kw in combined for kw in _EARNINGS_KEYWORDS):
                logger.info(
                    "has_earnings_risk[%s]: earnings signal detected in news — vetoing. "
                    "headline=%r [loop_id=%s]",
                    symbol, headline[:120], loop_id,
                )
                log_event(
                    AuditEvent(
                        loop_id=loop_id,
                        event_type="safety_check",
                        symbol=symbol,
                        payload={
                            "check": "earnings_proximity",
                            "result": True,
                            "headline_preview": headline[:120],
                            "blackout_days": blackout_days,
                        },
                    ),
                    db_path=db_path,
                )
                return True

        return False

    except ImportError:
        # alpaca-py NewsClient not available in this SDK version — fail closed.
        # Missing dependency means we cannot verify earnings safety, so block
        # the symbol as a precaution until the SDK is installed.
        logger.warning(
            "has_earnings_risk[%s]: NewsClient not available — blocking symbol as precaution [loop_id=%s]",
            symbol, loop_id,
        )
        return True
    except Exception as exc:
        # Any infrastructure failure → fail closed (block trading until check succeeds)
        logger.warning(
            "has_earnings_risk[%s]: news API error — blocking symbol as precaution: %s [loop_id=%s]",
            symbol, exc, loop_id,
        )
        return True
