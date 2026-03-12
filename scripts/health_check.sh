#!/usr/bin/env bash
# scripts/health_check.sh — Sauce system health check
#
# Runs on the droplet (SSH'd in by GitHub Actions monitor.yml).
# Exits 0 if everything looks healthy; exits 1 with a clear message if not.
#
# Checks:
#   1. Docker container 'sauce' is in "running" state.
#   2. cron.log was written within the last 70 minutes (just over 2 cron cycles).
#
# Usage: bash scripts/health_check.sh
# Env:   DO_APP_PATH or script-local APP_PATH default below.

set -euo pipefail

APP_PATH="${DO_APP_PATH:-/root/Sauce}"
CONTAINER_NAME="sauce"
LOG_FILE="${APP_PATH}/data/logs/cron.log"
LOG_STALE_MINUTES=70   # 2 × 30-min cron cycles + 10-min grace

FAIL=0

# ── 1. Container state ────────────────────────────────────────────────────────
STATUS=$(docker inspect --format='{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null || echo "not_found")

if [[ "${STATUS}" != "running" ]]; then
  echo "FAIL: container '${CONTAINER_NAME}' is '${STATUS}' (expected 'running')"
  FAIL=1
else
  echo "OK:   container '${CONTAINER_NAME}' is running"
fi

# ── 2. Cron log freshness ─────────────────────────────────────────────────────
if [[ ! -f "${LOG_FILE}" ]]; then
  echo "FAIL: log file not found: ${LOG_FILE}"
  FAIL=1
else
  # find exits 0 and prints the path when the file is newer than -mmin minutes.
  RECENT=$(find "${LOG_FILE}" -mmin "-${LOG_STALE_MINUTES}" 2>/dev/null || true)
  if [[ -z "${RECENT}" ]]; then
    LAST_MODIFIED=$(stat -c '%y' "${LOG_FILE}" 2>/dev/null || echo "unknown")
    echo "FAIL: cron.log has not been updated in ${LOG_STALE_MINUTES}+ minutes (last: ${LAST_MODIFIED})"
    FAIL=1
  else
    echo "OK:   cron.log updated within last ${LOG_STALE_MINUTES} minutes"
  fi
fi

# ── Result ────────────────────────────────────────────────────────────────────
if [[ "${FAIL}" -ne 0 ]]; then
  echo "HEALTH CHECK FAILED"
  exit 1
fi

echo "HEALTH CHECK PASSED"
exit 0
