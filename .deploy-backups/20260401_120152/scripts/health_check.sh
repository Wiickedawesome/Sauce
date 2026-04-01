#!/usr/bin/env bash
# scripts/health_check.sh — Sauce system health check
#
# Runs on the droplet (SSH'd in by GitHub Actions monitor.yml).
# Exits 0 if everything looks healthy; exits 1 with a clear message if not.
#
# Checks:
#   1. Docker container 'sauce' is in "running" state.
#   2. cron.log was written within the last 70 minutes (just over 2 cron cycles).
#   3. Latest loop_end status is not failed.
#   4. Broker account auth works from inside the live container.
#   5. Market data auth works from inside the live container.
#
# Usage: bash scripts/health_check.sh
# Env:   VPS_APP_PATH or script-local APP_PATH default below.

set -euo pipefail

APP_PATH="${VPS_APP_PATH:-/root/Sauce}"
CONTAINER_NAME="sauce"
LOG_FILE="${APP_PATH}/data/logs/cron.log"

DEFAULT_LOOP_INTERVAL_MINUTES=5
LOOP_INTERVAL="${LOOP_INTERVAL_MINUTES:-}"
if [[ -z "${LOOP_INTERVAL}" && -f "${APP_PATH}/.env" ]]; then
  LOOP_INTERVAL=$(awk -F= '/^LOOP_INTERVAL_MINUTES=/{print $2; exit}' "${APP_PATH}/.env" | tr -d '[:space:]')
fi
if [[ ! "${LOOP_INTERVAL}" =~ ^[0-9]+$ ]]; then
  LOOP_INTERVAL="${DEFAULT_LOOP_INTERVAL_MINUTES}"
fi
LOG_STALE_MINUTES=$(( LOOP_INTERVAL * 2 + 10 ))  # 2 cycles + 10-min grace

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

# ── 3. Broker + Market Data auth inside container ────────────────────────────
if [[ "${STATUS}" == "running" ]]; then
  LAST_LOOP_STATUS=$(docker exec "${CONTAINER_NAME}" sqlite3 /app/data/sauce.db \
    "SELECT json_extract(payload, '$.status') FROM audit_events WHERE event_type = 'loop_end' ORDER BY timestamp DESC LIMIT 1;" \
    2>/tmp/sauce_health_check_status_error.txt || true)

  if [[ -z "${LAST_LOOP_STATUS}" ]]; then
    echo "FAIL: could not determine latest loop_end status"
    cat /tmp/sauce_health_check_status_error.txt 2>/dev/null || true
    FAIL=1
  elif [[ "${LAST_LOOP_STATUS}" == "failed" ]]; then
    echo "FAIL: latest loop_end status is failed"
    FAIL=1
  else
    echo "OK:   latest loop_end status is ${LAST_LOOP_STATUS}"
  fi

  if docker exec "${CONTAINER_NAME}" /app/.venv/bin/python -c '
from sauce.adapters.broker import get_account
from sauce.adapters.market_data import get_history

account = get_account(loop_id="health-check")
equity = account.get("equity", "unknown")
history = get_history("SPY", timeframe="30Min", bars=1)

print(f"BROKER_OK equity={equity}")
print(f"MARKET_DATA_OK bars={len(history)}")
' >/tmp/sauce_health_check_output.txt 2>/tmp/sauce_health_check_error.txt; then
    cat /tmp/sauce_health_check_output.txt
    echo "OK:   broker and market data auth succeeded inside container"
  else
    echo "FAIL: live API probe failed inside container"
    cat /tmp/sauce_health_check_error.txt
    FAIL=1
  fi
fi

# ── Result ────────────────────────────────────────────────────────────────────
if [[ "${FAIL}" -ne 0 ]]; then
  echo "HEALTH CHECK FAILED"
  exit 1
fi

echo "HEALTH CHECK PASSED"
exit 0
