"""Agent 0 — Session Boot.

Runs once at market open (or loop start).  Pure Python — no LLM, no broker.

Responsibilities:
  1. Reset session memory if new trading day.
  2. Load today's economic calendar.
  3. Build strategic context from historical memory.
  4. Determine calendar suppression status.
  5. Log an AuditEvent and return BootContext.
"""

from datetime import datetime, timezone

from sauce.adapters.db import log_event
from sauce.memory.db import (
    get_strategic_context,
    reset_session_memory_if_new_day,
)
from sauce.core.calendar import get_events_for_date, is_suppressed
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, BootContext


async def run(loop_id: str) -> BootContext:
    """Execute session boot sequence and return boot context."""
    settings = get_settings()
    now = datetime.now(timezone.utc)

    # Step 1 — wipe session memory if new day
    was_reset = reset_session_memory_if_new_day(settings.session_memory_db_path)

    # Step 2 — load today's calendar
    calendar_events = get_events_for_date(now)

    # Step 3 — load strategic context (unfiltered at boot)
    strategic_ctx = get_strategic_context(settings.strategic_memory_db_path)

    # Step 4 — check suppression
    suppressed = is_suppressed(now)

    boot_ctx = BootContext(
        was_reset=was_reset,
        calendar_events=calendar_events,
        strategic_context=strategic_ctx,
        is_suppressed=suppressed,
        as_of=now,
    )

    # Step 5 — audit
    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="session_boot",
            payload={
                "was_reset": was_reset,
                "calendar_event_count": len(calendar_events),
                "is_suppressed": suppressed,
            },
        ),
        settings.db_path,
    )

    return boot_ctx
