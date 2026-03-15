#!/usr/bin/env python3
"""Docker HEALTHCHECK script — verifies cron + recent loop completion."""

import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get("DB_PATH", "/app/data/sauce.db")
MAX_AGE_MINUTES = 70  # 2 cron cycles + grace


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
        conn.close()
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
            last = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"UNHEALTHY: cannot parse timestamp: {ts}")
        return 1

    age = datetime.now(timezone.utc) - last
    age_minutes = age.total_seconds() / 60

    if age_minutes > MAX_AGE_MINUTES:
        print(f"UNHEALTHY: last loop_end was {age_minutes:.0f} min ago (limit: {MAX_AGE_MINUTES})")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
