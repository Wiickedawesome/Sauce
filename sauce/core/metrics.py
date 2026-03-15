"""
core/metrics.py — Trailing performance metrics computed from daily NAV history.

Finding 1.4: provides Sharpe ratio, max drawdown, and total return so that
ops.py and any monitoring tooling can surface them without custom DB queries.

All computations are pure Python / pandas and read-only against the DB.
"""

import logging
import math
from datetime import datetime, timezone

from sqlalchemy import text

logger = logging.getLogger(__name__)


def compute_trailing_metrics(
    days: int = 252,
    db_path: str = "data/sauce.db",
) -> dict:
    """
    Compute annualised Sharpe ratio, maximum drawdown, and total return from
    the ``daily_stats`` table over the last ``days`` trading days.

    Returns a dict with:
        {
            "sharpe_ratio": float | None,
            "max_drawdown": float | None,   # expressed as a negative fraction
            "total_return": float | None,   # e.g. 0.05 = 5%
            "start_nav": float | None,
            "end_nav": float | None,
            "days_computed": int,
            "as_of": str,                   # ISO 8601 UTC timestamp
        }

    Any field is None when there is insufficient data (< 2 data points).
    This function is deliberately non-raising; on any DB / computation error
    it logs a warning and returns all-None metrics so callers stay healthy.
    """
    result: dict = {
        "sharpe_ratio": None,
        "max_drawdown": None,
        "total_return": None,
        "start_nav": None,
        "end_nav": None,
        "days_computed": 0,
        "as_of": datetime.now(timezone.utc).isoformat(),
    }

    try:
        from sauce.adapters.db import get_session

        session = get_session(db_path)
        try:
            rows = session.execute(
                text(
                    "SELECT date, ending_nav_usd "
                    "FROM daily_stats "
                    "ORDER BY date DESC "
                    "LIMIT :limit"
                ),
                {"limit": days + 1},
            ).fetchall()
        finally:
            session.close()

        if len(rows) < 2:
            logger.debug("metrics: fewer than 2 daily_stats rows — skipping computation")
            return result

        # Rows come newest-first; reverse so chronological order for diff.
        navs = [float(r[1]) for r in reversed(rows)]
        result["days_computed"] = len(navs)
        result["start_nav"] = navs[0]
        result["end_nav"] = navs[-1]

        # Daily returns: r_t = (NAV_t - NAV_{t-1}) / NAV_{t-1}
        daily_returns: list[float] = []
        for i in range(1, len(navs)):
            if navs[i - 1] > 0:
                daily_returns.append((navs[i] - navs[i - 1]) / navs[i - 1])

        if len(daily_returns) < 2:
            return result

        # Total return
        if navs[0] > 0:
            result["total_return"] = round((navs[-1] - navs[0]) / navs[0], 6)

        # Annualised Sharpe ratio (assuming 252 trading days, risk-free rate = 0)
        n = len(daily_returns)
        mean_r = sum(daily_returns) / n
        variance = sum((r - mean_r) ** 2 for r in daily_returns) / (n - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.0
        if std_r > 0:
            result["sharpe_ratio"] = round(mean_r / std_r * math.sqrt(252), 4)

        # Maximum drawdown: (peak - trough) / peak, expressed as negative fraction
        peak = navs[0]
        max_dd = 0.0
        for nav in navs[1:]:
            if nav > peak:
                peak = nav
            elif peak > 0:
                dd = (nav - peak) / peak  # negative value
                if dd < max_dd:
                    max_dd = dd
        result["max_drawdown"] = round(max_dd, 6)

    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics: failed to compute trailing metrics: %s", exc)

    return result
