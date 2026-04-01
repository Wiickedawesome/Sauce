#!/usr/bin/env python3
"""
scripts/diagnose.py — Query database for current system health state.

Usage: python scripts/diagnose.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from sauce.adapters.db import get_session
from sauce.core.config import get_settings


def run() -> None:
    settings = get_settings()
    session = get_session()

    db_type = "Supabase (PostgreSQL)" if settings.use_supabase else f"SQLite ({settings.db_path})"
    print("=" * 70)
    print("SAUCE SYSTEM DIAGNOSTIC")
    print(f"DB: {db_type}")
    print("=" * 70)

    # ── 1. Pause state ────────────────────────────────────────────────────────
    print("\n[PAUSE STATE]")
    current_pause = settings.trading_pause
    status = "*** PAUSED ***" if current_pause else "NOT PAUSED"
    print(f"  status   : {status}")
    print(f"  source   : env/config")
    print(f"  TRADING_PAUSE env/config: {current_pause}")

    row = session.execute(
        text(
            "SELECT payload->>'reason', "
            "       payload->>'action', "
            "       timestamp "
            "FROM audit_events "
            "WHERE event_type = 'safety_check' "
            "AND payload->>'action' = 'halt' "
            "ORDER BY timestamp DESC LIMIT 1"
        )
    ).fetchone()
    if row:
        reason, action, ts = row
        print(f"  last_halt_action: {action}")
        print(f"  last_halt_reason: {reason!r}")
        print(f"  last_halt_at    : {ts}")
    else:
        print("  last_halt_action: none recorded")

    # ── 2. Recent ops summaries ───────────────────────────────────────────────
    print("\n[TODAY SUMMARY]")
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    row = session.execute(
        text(
            "SELECT loop_runs, signals_fired, signals_skipped, orders_placed, trades_closed, "
            "       realized_pnl_usd, ending_equity, regime, updated_at "
            "FROM daily_summary WHERE date = :today LIMIT 1"
        ),
        {"today": today},
    ).fetchone()
    if not row:
        print("  No daily summary row found for today")
    else:
        (
            loop_runs,
            signals_fired,
            signals_skipped,
            orders_placed,
            trades_closed,
            realized_pnl_usd,
            ending_equity,
            regime,
            updated_at,
        ) = row
        print(f"  loop_runs      : {loop_runs}")
        print(f"  signals_fired  : {signals_fired}")
        print(f"  signals_skipped: {signals_skipped}")
        print(f"  orders_placed  : {orders_placed}")
        print(f"  trades_closed  : {trades_closed}")
        print(f"  realized_pnl   : ${float(realized_pnl_usd or 0):,.2f}")
        print(f"  ending_equity  : ${float(ending_equity or 0):,.2f}")
        print(f"  regime         : {regime}")
        print(f"  updated_at     : {updated_at}")

    # ── 3. Recent supervisor decisions ───────────────────────────────────────
    print("\n[LAST 10 SUPERVISOR DECISIONS]")
    rows = session.execute(
        text(
            "SELECT timestamp, "
            "       payload->>'action', "
            "       payload->>'reason' "
            "FROM audit_events "
            "WHERE event_type = 'supervisor_decision' "
            "ORDER BY timestamp DESC LIMIT 10"
        )
    ).fetchall()
    if not rows:
        print("  (none found)")
    for ts, action, reason in rows:
        print(f"  {ts}  action={action!s:7s}  reason={reason!r}")

    # ── 4. Recent loop errors ─────────────────────────────────────────────────
    print("\n[LAST 10 LOOP-LEVEL ERRORS]")
    rows = session.execute(
        text(
            "SELECT timestamp, symbol, "
            "       payload->>'stage', "
            "       payload->>'error' "
            "FROM audit_events "
            "WHERE event_type = 'error' "
            "ORDER BY timestamp DESC LIMIT 10"
        )
    ).fetchall()
    if not rows:
        print("  (none)")
    for ts, sym, stage, err in rows:
        print(f"  {ts}  symbol={sym!s:12s}  stage={stage!s:25s}  error={str(err)[:80]!r}")

    # ── 5. Recent signals ─────────────────────────────────────────────────────
    print("\n[LAST 15 RESEARCH SIGNALS]")
    rows = session.execute(
        text(
            "SELECT timestamp, symbol, side, score, threshold, strategy_name "
            "FROM signal_log "
            "ORDER BY timestamp DESC LIMIT 15"
        )
    ).fetchall()
    if not rows:
        print("  (none found)")
    for ts, sym, side, score, threshold, strategy_name in rows:
        print(
            f"  {ts}  symbol={sym!s:12s}  side={side!s:6s}  "
            f"score={int(score):3d}/{int(threshold):3d}  strategy={strategy_name!s}"
        )

    # ── 6. Latest loop_end timestamps ─────────────────────────────────────────
    print("\n[LAST 5 LOOP RUNS]")
    rows = session.execute(
        text(
            "SELECT s.loop_id, s.timestamp, e.timestamp, e.payload->>'status' "
            "FROM audit_events s "
            "LEFT JOIN audit_events e "
            "  ON e.loop_id = s.loop_id AND e.event_type = 'loop_end' "
            "WHERE s.event_type = 'loop_start' "
            "ORDER BY s.timestamp DESC LIMIT 5"
        )
    ).fetchall()
    if not rows:
        print("  (none found)")
    for loop_id, start_ts, end_ts, status in rows:
        print(f"  start={start_ts}  end={end_ts}  status={status or 'missing'}  loop_id={loop_id}")

    session.close()
    print("\n" + "=" * 70)
    print("To resume trading:  set TRADING_PAUSE=false in .env and restart")
    print("=" * 70)


if __name__ == "__main__":
    run()
