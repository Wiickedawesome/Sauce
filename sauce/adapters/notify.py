"""
adapters/notify.py — Critical alert notifications (Finding 5.2).

Sends alerts to a configured webhook (Slack-compatible) when severity
thresholds are crossed. If alert_webhook_url is empty, falls back to
Python logging so no alert is silently discarded.

Usage:
    from sauce.adapters.notify import send_alert
    send_alert("CRITICAL", "Daily loss limit breached", loop_id=loop_id)
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def send_alert(
    severity: str,
    message: str,
    loop_id: str = "unset",
) -> None:
    """
    Fire-and-forget alert delivery.

    Sends a JSON POST to settings.alert_webhook_url when configured.
    Always logs to Python logging regardless of webhook status so the
    message is never silently dropped.

    Parameters
    ----------
    severity:  Alert severity label, e.g. "CRITICAL", "WARNING", "INFO".
    message:   Human-readable description of the alert condition.
    loop_id:   Loop run UUID for correlation with audit trail.
    """
    from sauce.core.config import get_settings

    settings = get_settings()
    timestamp = datetime.now(timezone.utc).isoformat()
    text = f"[{severity}] [{loop_id}] {message} (at {timestamp})"

    # Always emit to Python logging — captured by any log aggregator.
    log_fn = logger.critical if severity.upper() in ("CRITICAL", "ERROR") else logger.warning
    log_fn("ALERT %s: %s [loop_id=%s]", severity, message, loop_id)

    webhook_url = settings.alert_webhook_url.strip()
    if not webhook_url:
        return  # Logging-only mode; nothing else to do.

    try:
        import httpx

        payload = {"text": text}
        # Use a short timeout — alert delivery must not block the loop.
        with httpx.Client(timeout=5.0) as client:
            response = client.post(webhook_url, json=payload)
            if response.status_code >= 400:
                logger.error(
                    "notify: webhook returned HTTP %s for alert '%s'",
                    response.status_code, message[:100],
                )
    except Exception as exc:  # noqa: BLE001
        # Never let notification failure crash the calling code.
        logger.error("notify: failed to send webhook alert: %s", exc)
