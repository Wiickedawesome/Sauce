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

from sauce.adapters.db import log_event, upsert_daily_stats
from sauce.adapters.notify import send_alert
from sauce.core.config import get_settings
from sauce.memory.learning import run_learning_cycle
from sauce.core.nav import compute_nav_and_fees
from sauce.core.safety import pause_trading
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)

# Minimum symbol count before triggering the "all vetoed" auto-pause.
_ANOMALY_VETO_SYMBOL_THRESHOLD = 3
# How many times the same symbol must be vetoed to flag it as anomalous.
_PER_SYMBOL_VETO_THRESHOLD = 3
# How many consecutive loop-level errors before triggering circuit breaker (Finding 4.3).
_CIRCUIT_BREAKER_THRESHOLD = 3


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
    # Anomaly 5: circuit breaker — consecutive loop-level errors (Finding 4.3)
    # Count error events at loop/infrastructure scope (not per-symbol) in recent
    # runs. If the last N loop_end events all had preceding stage errors, the
    # infrastructure is likely broken — pause and alert.
    try:
        _recent_error_count = _count_recent_loop_errors(
            db_path=db_path,
            threshold=_CIRCUIT_BREAKER_THRESHOLD,
        )
        if _recent_error_count >= _CIRCUIT_BREAKER_THRESHOLD:
            _cb_reason = (
                f"Circuit breaker: {_recent_error_count} consecutive loop-level "
                "errors detected — auto-pausing."
            )
            anomalies.append("circuit_breaker_triggered")
            logger.critical("ops[%s]: %s", loop_id, _cb_reason)
            send_alert("CRITICAL", _cb_reason, loop_id=loop_id)
            try:
                pause_trading(reason=_cb_reason, loop_id=loop_id)
            except Exception as _pause_exc:  # noqa: BLE001
                logger.error("ops[%s]: circuit-breaker pause failed: %s", loop_id, _pause_exc)
    except Exception as exc:  # noqa: BLE001
        logger.warning("ops[%s]: circuit-breaker check failed: %s", loop_id, exc)

    # Send alert for any high-severity anomalies found this run.
    if anomalies:
        send_alert(
            "WARNING",
            f"Anomalies detected: {', '.join(anomalies)}",
            loop_id=loop_id,
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
    # ── NAV calculation and daily stats (Finding 1.1, Finding 5.1) ─────────────
    date_str = now.strftime("%Y-%m-%d")
    equity_usd: float = float(summary.get("equity_usd") or 0.0)
    nav_result: dict[str, float] = {}

    if equity_usd > 0:
        try:
            nav_result = compute_nav_and_fees(
                equity=equity_usd,
                date=date_str,
                db_path=db_path,
                settings=settings,
                loop_id=loop_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("ops[%s]: NAV computation failed: %s", loop_id, exc)
            log_event(
                AuditEvent(
                    loop_id=loop_id,
                    event_type="error",
                    payload={"agent": "ops", "error": f"NAV computation: {exc}"},
                    prompt_version=settings.prompt_version,
                ),
                db_path=db_path,
            )

    # Persist daily stats so DailyStatsRow is actually populated (Finding 5.1).
    try:
        upsert_daily_stats(
            date=date_str,
            db_path=db_path,
            loop_runs=1,
            signals_generated=signals_total,
            signals_vetoed=risk_vetoes,
            orders_placed=int(summary.get("orders_placed", 0)),
            ending_nav_usd=nav_result.get("fully_adjusted_nav", equity_usd),
            trading_paused=supervisor_action == "abort",
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("ops[%s]: upsert_daily_stats failed: %s", loop_id, exc)

    # ── Learning loop (Sprint 6) ──────────────────────────────────────────────────
    run_weekly = bool(summary.get("run_weekly", False))
    try:
        learning_result = run_learning_cycle(
            loop_id=loop_id,
            strategic_db_path=str(settings.strategic_memory_db_path),
            run_weekly=run_weekly,
        )
        if learning_result.get("drift_alert"):
            anomalies.append("learning_drift_detected")
            send_alert(
                "WARNING",
                f"Win-rate drift detected: {learning_result['drift_alert']}",
                loop_id=loop_id,
            )
            log_event(
                AuditEvent(
                    loop_id=loop_id,
                    event_type="learning_drift_detected",
                    payload={"agent": "ops", "drift": learning_result["drift_alert"]},
                    prompt_version=settings.prompt_version,
                ),
                db_path=db_path,
            )
        if learning_result.get("weekly_report"):
            log_event(
                AuditEvent(
                    loop_id=loop_id,
                    event_type="learning_weekly_report",
                    payload={"agent": "ops", "report_count": len(learning_result["weekly_report"])},
                    prompt_version=settings.prompt_version,
                ),
                db_path=db_path,
            )
        if learning_result.get("calibration"):
            log_event(
                AuditEvent(
                    loop_id=loop_id,
                    event_type="learning_calibration_analysis",
                    payload={"agent": "ops", "total_entries": learning_result["calibration"].get("total_entries", 0)},
                    prompt_version=settings.prompt_version,
                ),
                db_path=db_path,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("ops[%s]: run_learning_cycle failed: %s", loop_id, exc)

    # ── Validation check ─────────────────────────────────────────────────────────
    try:
        from sauce.core.validation import run_validation

        validation_result = run_validation(
            loop_id=loop_id,
            db_path=db_path,
            strategic_db_path=str(settings.strategic_memory_db_path),
        )
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="validation_daily_check",
                payload={"agent": "ops", **validation_result},
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("ops[%s]: run_validation failed: %s", loop_id, exc)

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
    """Append one JSON line per loop run to the daily log file (jsonlines format, Finding 5.3)."""
    import json
    try:
        log_dir = Path(db_path).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        date_str = now.strftime("%Y-%m-%d")
        log_file = log_dir / f"daily_{date_str}.jsonl"
        record = {
            "timestamp": now.isoformat(),
            "loop_id": loop_id,
            "signals_total": signals_total,
            "signals_buy_sell": signals_buy_sell,
            "risk_vetoes": risk_vetoes,
            "orders_prepared": orders_prepared,
            "supervisor_action": supervisor_action,
            "orders_placed": orders_placed,
            "anomalies": anomalies,
        }
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.error("ops: failed to write daily log: %s", exc)


def _count_recent_loop_errors(
    db_path: str,
    threshold: int = 3,
) -> int:
    """
    Count how many of the most recent loop runs ended with a stage-level error
    event (e.g. stage in get_account, get_positions, supervisor, etc.).

    Returns the count of such loop IDs found in the last threshold*2 loop_end
    events.  Used by the circuit breaker (Finding 4.3).
    """
    try:
        from sqlalchemy import text
        from sauce.adapters.db import get_session

        session = get_session(db_path)
        try:
            # Fetch the most recent (threshold * 2) loop IDs from loop_end events.
            loop_rows = session.execute(
                text(
                    "SELECT loop_id FROM audit_events "
                    "WHERE event_type = 'loop_end' "
                    "ORDER BY timestamp DESC "
                    "LIMIT :limit"
                ),
                {"limit": threshold * 2},
            ).fetchall()

            if not loop_rows:
                return 0

            recent_loop_ids = [str(r[0]) for r in loop_rows]

            # For each loop ID, check if it has a stage-level error event.
            # Stage-level errors: stage in (get_account, get_positions,
            # get_universe_snapshot, supervisor, portfolio, ops, main).
            error_loop_ids: set[str] = set()
            for lid in recent_loop_ids:
                count = session.execute(
                    text(
                        "SELECT COUNT(*) FROM audit_events "
                        "WHERE loop_id = :lid "
                        "AND event_type = 'error' "
                        "AND json_extract(payload, '$.stage') IN ("
                        "  'get_account', 'get_positions', 'get_universe_snapshot',"
                        "  'supervisor', 'portfolio', 'ops', 'main'"
                        ")"
                    ),
                    {"lid": lid},
                ).scalar()
                if count and count > 0:
                    error_loop_ids.add(lid)
        finally:
            session.close()

        # Count consecutive runs from the most recent that had errors.
        consecutive = 0
        for lid in recent_loop_ids:
            if lid in error_loop_ids:
                consecutive += 1
            else:
                break  # First clean run resets the streak.

        return consecutive

    except Exception as exc:  # noqa: BLE001
        logger.warning("ops: _count_recent_loop_errors failed: %s", exc)
        return 0
