#!/usr/bin/env python3
"""Log a container_start audit event to the DB on every container startup.

Called from the Dockerfile CMD before cron starts. Enables detection of
unexpected restarts (Finding F-13 / IMP-15).
"""

import os
import sqlite3
from datetime import datetime, timezone
from json import dumps


def main() -> int:
    db_path = os.environ.get("DB_PATH", "data/sauce.db")
    now = datetime.now(timezone.utc).isoformat()

    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO audit_events (loop_id, event_type, timestamp, payload) "
            "VALUES (?, ?, ?, ?)",
            (
                "container",
                "container_start",
                now,
                dumps({"timestamp": now, "hostname": os.environ.get("HOSTNAME", "unknown")}),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        # Non-fatal — don't prevent container from starting
        print(f"[log_container_start] Could not log event: {exc}")
        return 0

    print(f"[log_container_start] Logged container_start at {now}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
