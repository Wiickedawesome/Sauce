"""
agents/ops.py — Ops agent: anomaly detection, daily log, audit summary.

Phase 5 IMPLEMENTATION — no LLM call.

Responsibilities:
  1. Parse loop-run summary to extract key metrics.
  2. Detect anomalies in this run:
     a. No signals generated at all (market data missing?).
     b. 100% veto rate across ≥3 symbols (systematic problem?) → pause_trading().
     c. Any symbol vetoed 3+ times within the summary.
     d. Supervisor aborted with prepared orders (LLM gate activated).
  3. Write a daily log line to data/logs/daily_YYYY-MM-DD.txt.
  4. Log a final AuditEvent for the run with anomaly flags.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sauce.adapters.db import log_event
from sauce.core.config import get_settings
from sauce.core.safety import pause_trading
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)

# Minimum symbol count before triggering the "all vetoed" auto-pause.
_ANOMALY_VETO_SYMBOL_THRESHOLD = 3
# How many times the same symbol must be vetoed to flag it as anomalous.
_PER_SYMBOL_VETO_THRESHOLD = 3


async def run(
    loop_id: str,
    summary: dict[str, Any],
) -> None:
    """
    Write audit trail, detect anomalies, and update daily log file.

    Parameters
    ----------
    loop_id:  Current loop run UUID.
    summary:  Dict with counts and metadata for this run.
              Expected keys (all optional, default to safe values if missing):
                signals_total, signals_hold, signals_buy_sell,
                risk_vetoes, risk_approved,
                orders_prepared, supervisor_action, orders_placed,
                symbols_attempted, veto_by_symbol.

    Side effects
    ------------
    - Writes AuditEvent(s) to the DB.
    - Appends a line to data/logs/daily_YYYY-MM-DD.txt.
    - May call pause_trading() if systematic anomaly detected.
    """
    settings = get_settings()
    db_path = str(settings.db_path)
    now = datetime.now(timezone.utc)

    # ── Extract summary fields safely ─────────────────────────────────────────────
    signals_total: int = int(summary.get("signals_total", 0))
    signals_buy_sell: int = int(summary.get("signals_buy_sell", 0))
    risk_vetoes: int = int(summary.get("risk_vetoes", 0))
    orders_prepared: int = int(summary.get("orders_prepared", 0))
    supervisor_action: str = str(summary.get("supervisor_action", "abort"))
    orders_placed: int = int(summary.get("orders_placed", 0))
    symbols_attempted: list[str] = list(summary.get("symbols_attempted", []))
    veto_by_symbol: dict[str, int] = dict(summary.get("veto_by_symbol", {}))

    # ── Anomaly detection ──────────────────────────────────────────────────────
    anomalies: list[str] = []

    # Anomaly 1: No signals at all
    if signals_total == 0:
        anomalies.append("no_signals_generated")
        logger.warning("ops[%s]: no signals generated this run", loop_id)

    # Anomaly 2: 100% veto rate across enough symbols → systematic problem
    if (
        signals_buy_sell > 0
        and risk_vetoes == signals_buy_sell
        and len(symbols_attempted) >= _ANOMALY_VETO_SYMBOL_THRESHOLD
    ):
        anomalies.append("all_signals_vetoed")
        reason = (
            f"100% veto rate: {risk_vetoes} vetoes / {signals_buy_sell} signals "
            f"across {len(symbols_attempted)} symbols"
        )
        logger.error("ops[%s]: auto-pause triggered — %s", loop_id, reason)
        try:
            pause_trading(reason=reason, loop_id=loop_id)
        except Exception as exc:  # noqa: BLE001
            logger.error("ops[%s]: pause_trading() failed: %s", loop_id, exc)
            log_event(
                AuditEvent(
                    loop_id=loop_id,
                    event_type="error",
                    symbol=None,
                    payload={"agent": "ops", "error": f"pause_trading failed: {exc}"},
                    prompt_version=settings.prompt_version,
                ),
                db_path=db_path,
            )

    # Anomaly 3: any single symbol vetoed many times
    high_veto_symbols = [
        sym
        for sym, count in veto_by_symbol.items()
        if count >= _PER_SYMBOL_VETO_THRESHOLD
    ]
    if high_veto_symbols:
        anomalies.append(f"high_veto_symbols:{high_veto_symbols}")
        logger.warning(
            "ops[%s]: symbols vetoed ≥%d times: %s",
            loop_id, _PER_SYMBOL_VETO_THRESHOLD, high_veto_symbols,
        )

    # Anomaly 4: supervisor aborted with orders ready
    if supervisor_action == "abort" and orders_prepared > 0:
        anomalies.append("supervisor_aborted_with_orders")
        logger.warning(
            "ops[%s]: supervisor aborted %d prepared orders",
            loop_id, orders_prepared,
        )

    # ── Log final AuditEvent ─────────────────────────────────────────────────────
    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="ops_summary",
            symbol=None,
            payload={
                "agent": "ops",
                "summary": summary,
                "anomalies": anomalies,
                "timestamp": now.isoformat(),
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    # ── Write daily log file ──────────────────────────────────────────────────────
    _write_daily_log(
        loop_id=loop_id,
        db_path=db_path,
        now=now,
        signals_total=signals_total,
        signals_buy_sell=signals_buy_sell,
        risk_vetoes=risk_vetoes,
        orders_prepared=orders_prepared,
        supervisor_action=supervisor_action,
        orders_placed=orders_placed,
        anomalies=anomalies,
    )


def _write_daily_log(
    loop_id: str,
    db_path: str,
    now: datetime,
    signals_total: int,
    signals_buy_sell: int,
    risk_vetoes: int,
    orders_prepared: int,
    supervisor_action: str,
    orders_placed: int,
    anomalies: list[str],
) -> None:
    """Append one line per loop run to the daily log file."""
    try:
        log_dir = Path(db_path).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = now.strftime("%Y-%m-%d")
        log_file = log_dir / f"daily_{date_str}.txt"
        line = (
            f"{now.isoformat()} "
            f"loop={loop_id[:8]} "
            f"signals={signals_total}(active={signals_buy_sell}) "
            f"vetoes={risk_vetoes} "
            f"orders_prepared={orders_prepared} "
            f"supervisor={supervisor_action} "
            f"placed={orders_placed} "
            f"anomalies={anomalies or 'none'}\n"
        )
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError as exc:
        logger.error("ops: failed to write daily log: %s", exc)
