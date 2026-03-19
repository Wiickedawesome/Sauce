#!/usr/bin/env python3
"""Docker HEALTHCHECK script — verifies cron + recent loop completion."""

import os
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta

DB_PATH = os.environ.get("DB_PATH", "/app/data/sauce.db")
_INTERVAL = int(os.environ.get("LOOP_INTERVAL_MINUTES", "30"))
MAX_AGE_MINUTES = (_INTERVAL * 2) + 5  # 2 cron cycles + grace


def main() -> int:
    # 1. Check cron is running
    result = subprocess.run(["pgrep", "cron"], capture_output=True)
    if result.returncode != 0:
        print("UNHEALTHY: cron process not found")
        return 1

    # 2. Check last loop_end timestamp in audit DB
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute(
            "SELECT MAX(timestamp) FROM audit_events WHERE event_type = 'loop_end'"
        ).fetchone()
    except Exception:
        # DB doesn't exist yet (first boot) — cron is running, that's enough
        return 0

    ts = row[0] if row else None
    if ts is None:
        # No loop has run yet — cron is running, acceptable during startup
        return 0

    try:
        if "+" in ts or "Z" in ts:
            last = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            last = datetime.fromisoformat(ts).replace(tzinfo=UTC)
    except ValueError:
        print(f"UNHEALTHY: cannot parse timestamp: {ts}")
        return 1

    age = datetime.now(UTC) - last
    age_minutes = age.total_seconds() / 60

    if age_minutes > MAX_AGE_MINUTES:
        print(f"UNHEALTHY: last loop_end was {age_minutes:.0f} min ago (limit: {MAX_AGE_MINUTES})")
        return 1

    # 3. Check for orphaned loop_start (started but never finished — crash indicator)
    # Only flag orphans from within the MAX_AGE window; older ones are historical.
    try:
        cutoff = (datetime.now(UTC) - timedelta(minutes=MAX_AGE_MINUTES)).isoformat()
        orphan = conn.execute(
            "SELECT s.loop_id, s.timestamp FROM audit_events s "
            "WHERE s.event_type = 'loop_start' "
            "AND s.timestamp >= ? "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM audit_events e "
            "  WHERE e.event_type = 'loop_end' AND e.loop_id = s.loop_id"
            ") ORDER BY s.timestamp DESC LIMIT 1",
            (cutoff,),
        ).fetchone()
        if orphan:
            print(
                f"UNHEALTHY: orphaned loop_start detected (loop_id={orphan[0]}, started={orphan[1]})"
            )
            return 1
    except Exception:
        pass  # Non-fatal — primary checks already passed

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
