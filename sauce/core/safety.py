"""
core/safety.py — Global safety guards for the Sauce trading system.

Rules:
- TRADING_PAUSE is checked FIRST in every loop run. No exceptions.
- is_data_fresh() must be called before using ANY market data value.
- All safety decisions are logged as AuditEvent(event_type="safety_check").
- pause_trading() persists to DB — survives restarts. Only resume_trading() clears it.
- resume_trading() is manual only — it is never called automatically.
- check_daily_loss() returns False on ANY failure (fail-safe default).

Market hours note:
  US equities trade 09:30–16:00 Eastern Time, Monday–Friday.
  Crypto pairs ("/" in symbol) have no market-hours restriction.
"""

import logging
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)

# Eastern timezone used for US market-hours checks
_ET = ZoneInfo("America/New_York")

# US equity market open/close times (Eastern)
_MARKET_OPEN = time(9, 30, 0)
_MARKET_CLOSE = time(16, 0, 0)


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

    # Normalise naive datetimes to UTC (defensive — API should always return tz-aware)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)

    age_seconds = (now - as_of).total_seconds()
    return age_seconds <= ttl_sec


# ── Daily P&L Guard ───────────────────────────────────────────────────────────

def check_daily_loss(account: dict, loop_id: str = "unset") -> bool:
    """
    Return True if today's loss is within the configured limit.
    Return False (block trading) if the daily loss limit is breached.

    Fails SAFE: any missing field or parse error returns False.

    The Alpaca account dict contains 'equity' and 'last_equity' as strings.
    Daily P&L = (equity - last_equity) / last_equity × 100%.
    """
    from sauce.adapters.db import log_event

    settings = get_settings()

    try:
        equity = float(account.get("equity") or 0.0)
        last_equity = float(account.get("last_equity") or 0.0)

        db_path = str(settings.db_path)

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
        limit = -abs(settings.max_daily_loss_pct)  # e.g. -0.02

        ok = daily_pnl_pct >= limit

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="safety_check",
                payload={
                    "check": "daily_loss",
                    "result": ok,
                    "daily_pnl_pct": round(daily_pnl_pct * 100, 4),
                    "limit_pct": round(abs(settings.max_daily_loss_pct) * 100, 4),
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
    # Crypto trades 24/7 on Alpaca
    if "/" in symbol:
        return True

    now_et = _now_et()

    # Weekend check (0=Monday, 6=Sunday)
    if now_et.weekday() >= 5:
        return False

    current_time = now_et.time()
    return _MARKET_OPEN <= current_time < _MARKET_CLOSE
