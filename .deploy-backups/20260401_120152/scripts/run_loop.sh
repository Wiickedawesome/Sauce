#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/run_loop.sh — Cron entrypoint for the Sauce trading loop.
#
# Invoked by cron at LOOP_INTERVAL_MINUTES cadence (default 5).
#
# Contract:
#   - Logs a timestamped START and END line on every run.
#   - A non-zero exit from the Python process is logged and preserved for cron/health checks.
#   - Activates the virtual environment before running Python.
#   - All stdout/stderr from Python is captured by the cron redirect (>> cron.log).
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV="${APP_DIR}/.venv"
LOGS_DIR="${APP_DIR}/data/logs"

# ── Ensure log directory exists ───────────────────────────────────────────────
mkdir -p "${LOGS_DIR}"

# ── Timestamp helper ──────────────────────────────────────────────────────────
ts() {
    date -u +"%Y-%m-%dT%H:%M:%SZ"
}

echo "----------------------------------------"
echo "[$(ts)] SAUCE LOOP START"

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -f "${VENV}/bin/python" ]]; then
    echo "[$(ts)] ERROR: virtual environment not found at ${VENV}" >&2
    echo "[$(ts)] Run: python -m venv .venv && .venv/bin/pip install -e ." >&2
    exit 1
fi

if [[ ! -f "${APP_DIR}/.env" ]] && [[ ! -f /etc/environment.sauce ]]; then
    echo "[$(ts)] ERROR: no credentials found — expected ${APP_DIR}/.env or /etc/environment.sauce" >&2
    echo "[$(ts)] Copy .env.example to .env and fill in credentials." >&2
    exit 1
fi

# ── Activate virtual environment ──────────────────────────────────────────────
# shellcheck source=/dev/null
source "${VENV}/bin/activate"

# ── Change to project root so relative imports work ───────────────────────────
cd "${APP_DIR}"

# ── Load environment variables ────────────────────────────────────────────────
# Cron jobs do NOT inherit Docker env vars. We must explicitly load them.
# First try the cron env dump (written by Docker CMD), then fall back to .env.
if [[ -f /etc/environment.sauce ]]; then
    # shellcheck source=/dev/null
    set -a; source /etc/environment.sauce; set +a
elif [[ -f "${APP_DIR}/.env" ]]; then
    # shellcheck source=/dev/null
    set -a; source "${APP_DIR}/.env"; set +a
fi

# ── Run the loop ──────────────────────────────────────────────────────────────
# Preserve the Python exit code so cron, Docker health checks, and heartbeat
# monitors can distinguish a healthy cycle from a failed one.
exit_code=0
python -m sauce.loop || exit_code=$?

if [[ "${exit_code}" -ne 0 ]]; then
    echo "[$(ts)] WARNING: loop exited with code ${exit_code} — check audit DB for details"
fi

echo "[$(ts)] SAUCE LOOP END (exit=${exit_code})"
echo "----------------------------------------"

# ── Heartbeat (optional) ─────────────────────────────────────────────────────
# Set HEARTBEAT_URL in .env to ping a UptimeRobot/BetterUptime heartbeat
# monitor after every loop run. Use a 35-minute interval on the monitor side
# so one missed cron cycle doesn't immediately fire an alert.
# shellcheck source=/dev/null
if [[ "${exit_code}" -eq 0 ]]; then
    [[ -f "${APP_DIR}/.env" ]] && source "${APP_DIR}/.env" || true
    [[ -n "${HEARTBEAT_URL:-}" ]] && curl -fsS --retry 3 "${HEARTBEAT_URL}" > /dev/null 2>&1 || true
fi

exit "${exit_code}"
