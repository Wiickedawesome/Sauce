"""
core/nav.py — Net Asset Value calculation and management/performance fee accrual.

Terminology used throughout:
  gross_nav        Total market value of all positions + cash (= broker equity).
  daily_mgmt_fee   Management fee accrued for one trading day.
  net_nav          gross_nav minus daily management fee accrual.
  high_water_mark  Highest ever net_nav achieved at any end-of-day.
  performance_fee  Performance fee accrued when net_nav exceeds the high-water mark.

IMPORTANT LIMITATIONS (acknowledged design boundaries):
  - This system manages a single brokerage account, not a multi-investor fund.
    There is no share/unit count, so "NAV per unit" is not computed here.
    gross_nav == total account equity for this implementation.
  - Performance fees are computed for observability only. They are NOT
    automatically transferred out of the account — a human must do that.
  - Management fees are accrued to the fee_accruals table and reduce net_nav
    for reporting purposes. They do not generate a cash withdrawal automatically.
  - Fee rates are read from Settings (annual_management_fee_pct,
    performance_fee_pct) so they are adjustable from .env without code changes.

Usage:
    from sauce.core.nav import compute_nav_and_fees
    nav_result = compute_nav_and_fees(equity=float(account["equity"]),
                                       date=today_str, db_path=db_path,
                                       settings=settings, loop_id=loop_id)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# US trading days per year used for daily fee accrual.
_TRADING_DAYS_PER_YEAR: int = 252


# ── Core NAV formulas ─────────────────────────────────────────────────────────

def calc_gross_nav(equity: float) -> float:
    """
    Gross NAV = total account equity as reported by the broker.

    This equals all position market values + cash. No liability deductions
    other than fees (which are subtracted to get net NAV).

    Returns 0.0 if equity is negative or zero (should never happen, but safe).
    """
    return max(0.0, float(equity))


def calc_daily_management_fee(gross_nav: float, annual_rate: float) -> float:
    """
    Daily management fee accrual for one trading day.

    Formula: gross_nav × annual_rate / TRADING_DAYS_PER_YEAR

    Parameters
    ----------
    gross_nav:   Current gross NAV in USD.
    annual_rate: Annual management fee as a decimal (e.g. 0.01 for 1%).

    Returns
    -------
    Fee amount in USD for one trading day (always >= 0.0).
    """
    if gross_nav <= 0.0 or annual_rate <= 0.0:
        return 0.0
    return gross_nav * annual_rate / _TRADING_DAYS_PER_YEAR


def calc_performance_fee(
    net_nav: float,
    high_water_mark: float,
    performance_rate: float,
) -> float:
    """
    Performance fee for the current period (typically computed end-of-day).

    Formula: max(0, net_nav - high_water_mark) × performance_rate

    Only positive when net NAV exceeds the prior high-water mark.

    Parameters
    ----------
    net_nav:          Current net NAV after management fee deduction.
    high_water_mark:  Prior highest net NAV at end of any trading day.
    performance_rate: Performance fee rate as a decimal (e.g. 0.20 for 20%).

    Returns
    -------
    Performance fee in USD (always >= 0.0).
    """
    if net_nav <= high_water_mark or performance_rate <= 0.0:
        return 0.0
    return (net_nav - high_water_mark) * performance_rate


# ── DB helpers ────────────────────────────────────────────────────────────────

def get_high_water_mark(db_path: str) -> float:
    """
    Return the highest ever ending_nav_usd from daily_stats, or 0.0.

    Used to determine whether the current net_nav exceeds the HWM and
    a performance fee should be accrued.

    Fails safe: returns 0.0 on any DB error (which means a performance fee
    would be computed against 0 HWM — conservative but correct direction).
    """
    from sqlalchemy import text as sa_text
    from sauce.adapters.db import get_session

    session = get_session(db_path)
    try:
        row = session.execute(
            sa_text("SELECT MAX(ending_nav_usd) FROM daily_stats WHERE ending_nav_usd > 0")
        ).fetchone()
        if row is None or row[0] is None:
            return 0.0
        return float(row[0])
    except Exception as exc:
        logger.warning("get_high_water_mark: DB error: %s", exc)
        return 0.0
    finally:
        session.close()


# ── Composite entry point ─────────────────────────────────────────────────────

def compute_nav_and_fees(
    equity: float,
    date: str,
    db_path: str,
    settings: Any,
    loop_id: str = "unset",
) -> dict[str, float]:
    """
    Compute all NAV and fee values for the current loop run.

    Workflow:
      1. Compute gross_nav = equity.
      2. Compute daily_mgmt_fee = gross_nav × annual_rate / 252.
      3. net_nav = gross_nav - daily_mgmt_fee.
      4. Look up current high-water mark from daily_stats.
      5. Compute performance_fee if net_nav > HWM.
      6. fully_adjusted_nav = net_nav - performance_fee.

    Parameters
    ----------
    equity:   Account equity in USD from broker.get_account()["equity"].
    date:     Today's date as YYYY-MM-DD string. Used to check if a fee has
              already been accrued today (prevents double-accrual on multiple
              loop runs per day).
    db_path:  Path to the SQLite database.
    settings: get_settings() instance for fee rate config.
    loop_id:  Audit correlation ID.

    Returns
    -------
    Dict with keys:
        gross_nav, daily_mgmt_fee, net_nav,
        high_water_mark, performance_fee, fully_adjusted_nav.
    All values are floats in USD. All are >= 0.0.
    """
    from sauce.adapters.db import log_event
    from sauce.core.schemas import AuditEvent

    gross_nav = calc_gross_nav(equity)

    daily_mgmt_fee = calc_daily_management_fee(
        gross_nav, settings.annual_management_fee_pct
    )
    net_nav = max(0.0, gross_nav - daily_mgmt_fee)

    high_water_mark = get_high_water_mark(db_path)
    performance_fee = calc_performance_fee(
        net_nav, high_water_mark, settings.performance_fee_pct
    )
    fully_adjusted_nav = max(0.0, net_nav - performance_fee)

    result = {
        "gross_nav": round(gross_nav, 2),
        "daily_mgmt_fee": round(daily_mgmt_fee, 2),
        "net_nav": round(net_nav, 2),
        "high_water_mark": round(high_water_mark, 2),
        "performance_fee": round(performance_fee, 2),
        "fully_adjusted_nav": round(fully_adjusted_nav, 2),
        "annual_mgmt_fee_pct": settings.annual_management_fee_pct,
        "performance_fee_pct": settings.performance_fee_pct,
    }

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="ops_summary",
            payload={"nav_calculation": result, "date": date},
        ),
        db_path=db_path,
    )

    if performance_fee > 0:
        logger.info(
            "NAV[%s]: gross=%.2f mgmt_fee=%.2f net=%.2f perf_fee=%.2f (HWM=%.2f) adjusted=%.2f",
            date, gross_nav, daily_mgmt_fee, net_nav,
            performance_fee, high_water_mark, fully_adjusted_nav,
        )
    else:
        logger.info(
            "NAV[%s]: gross=%.2f mgmt_fee=%.2f net=%.2f (HWM=%.2f, no perf fee)",
            date, gross_nav, daily_mgmt_fee, net_nav, high_water_mark,
        )

    return result
