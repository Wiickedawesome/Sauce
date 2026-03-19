#!/usr/bin/env python3
"""
scripts/diagnose.py — Query sauce.db for current system health state.

Run locally:    python scripts/diagnose.py
Run on VPS:     docker exec sauce python scripts/diagnose.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from sauce.adapters.db import get_session
from sauce.core.config import get_settings


def run() -> None:
    settings = get_settings()
    db_path = str(settings.db_path)
    session = get_session(db_path)

    print("=" * 70)
    print("SAUCE SYSTEM DIAGNOSTIC")
    print(f"DB: {db_path}")
    print("=" * 70)

    # ── 1. Pause state ────────────────────────────────────────────────────────
    row = session.execute(
        text(
            "SELECT json_extract(payload, '$.paused'), "
            "       json_extract(payload, '$.reason'), "
            "       json_extract(payload, '$.action'), "
            "       timestamp "
            "FROM audit_events "
            "WHERE event_type = 'safety_check' "
            "AND json_extract(payload, '$.paused') IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT 1"
        )
    ).fetchone()
    print("\n[PAUSE STATE]")
    if row:
        paused_val, reason, action, ts = row
        is_paused = bool(paused_val)
        status = "*** PAUSED ***" if is_paused else "NOT PAUSED"
        print(f"  status   : {status}")
        print(f"  action   : {action}")
        print(f"  reason   : {reason!r}")
        print(f"  timestamp: {ts}")
    else:
        print("  No pause record found — assume not paused")

    # Also check config-level pause
    print(f"  TRADING_PAUSE env/config: {settings.trading_pause}")

    # ── 2. Recent ops summaries ───────────────────────────────────────────────
    print("\n[LAST 10 OPS SUMMARIES]")
    rows = session.execute(
        text(
            "SELECT timestamp, "
            "       json_extract(payload, '$.anomalies'), "
            "       json_extract(payload, '$.summary') "
            "FROM audit_events "
            "WHERE event_type = 'ops_summary' "
            "ORDER BY timestamp DESC LIMIT 10"
        )
    ).fetchall()
    if not rows:
        print("  (none found)")
    for ts, anomalies_json, summary_json in rows:
        s = json.loads(summary_json or "{}")
        a = json.loads(anomalies_json or "[]")
        print(
            f"  {ts}  signals={s.get('signals_total', 0):2d} "
            f"buy/sell={s.get('signals_buy_sell', 0):2d} "
            f"vetoes={s.get('risk_vetoes', 0):2d} "
            f"orders={s.get('orders_placed', 0):2d} "
            f"supervisor={s.get('supervisor_action', '?'):7s} "
            f"anomalies={a}"
        )

    # ── 3. Recent supervisor decisions ───────────────────────────────────────
    print("\n[LAST 10 SUPERVISOR DECISIONS]")
    rows = session.execute(
        text(
            "SELECT timestamp, "
            "       json_extract(payload, '$.action'), "
            "       json_extract(payload, '$.reason') "
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
            "       json_extract(payload, '$.stage'), "
            "       json_extract(payload, '$.error') "
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
            "SELECT timestamp, symbol, "
            "       json_extract(payload, '$.side'), "
            "       json_extract(payload, '$.confidence'), "
            "       COALESCE(json_extract(payload, '$.reasoning'), json_extract(payload, '$.reason')) "
            "FROM audit_events "
            "WHERE event_type = 'signal' "
            "ORDER BY timestamp DESC LIMIT 15"
        )
    ).fetchall()
    if not rows:
        print("  (none found)")
    for ts, sym, side, conf, reason in rows:
        print(
            f"  {ts}  symbol={sym!s:12s}  side={side!s:6s}  "
            f"conf={str(conf):5s}  reason={str(reason or '')[:60]!r}"
        )

    # ── 6. Latest loop_end timestamps ─────────────────────────────────────────
    print("\n[LAST 5 LOOP RUNS]")
    rows = session.execute(
        text(
            "SELECT loop_id, timestamp "
            "FROM audit_events "
            "WHERE event_type = 'loop_end' "
            "ORDER BY timestamp DESC LIMIT 5"
        )
    ).fetchall()
    if not rows:
        print("  (none found)")
    for loop_id, ts in rows:
        print(f"  {ts}  loop_id={loop_id}")

    session.close()
    print("\n" + "=" * 70)
    print("To resume trading:  set TRADING_PAUSE=false in .env and restart")
    print("=" * 70)


if __name__ == "__main__":
    run()
