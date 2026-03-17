# ──────────────────────────────────────────────────────────────────────────────
# Sauce — Trading System Docker Image
#
# Build:  docker build -f docker/Dockerfile -t sauce .
# Run:    docker compose -f docker/docker-compose.yml up -d
#
# The image installs cron and sets up the loop schedule from
# LOOP_INTERVAL_MINUTES (default 30) at container start.
# All output goes to /app/data/logs/cron.log (mounted volume).
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        cron \
        procps \
        sqlite3 \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to UTC so all datetime calls are unambiguous
ENV TZ=UTC

# ── Project setup ─────────────────────────────────────────────────────────────
WORKDIR /app

# Copy dependency manifests first for better layer caching
COPY pyproject.toml requirements.lock ./

# Create and populate virtual environment using the lockfile for reproducibility
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --upgrade pip --quiet \
    && /app/.venv/bin/pip install -r requirements.lock --quiet

# Copy the full project (after deps — layer cache benefit)
COPY . .

# Install the project itself (deps already satisfied by lockfile)
RUN /app/.venv/bin/pip install -e . --no-deps --quiet

# Ensure the data directories exist
RUN mkdir -p /app/data/logs

# ── Cron setup ────────────────────────────────────────────────────────────────
# The cron schedule is written at container start from LOOP_INTERVAL_MINUTES
# (see CMD below). We create the file here so cron has it at the right path.
RUN touch /etc/cron.d/sauce && chmod 0644 /etc/cron.d/sauce

# Ensure run_loop.sh is executable
RUN chmod +x /app/scripts/run_loop.sh

# ── Health check ──────────────────────────────────────────────────────────────
# Confirm cron is running AND that a loop has completed within the last 35
# minutes (just over 2 cron cycles). Queries the audit DB for the most recent
# loop_end event.
HEALTHCHECK --interval=5m --timeout=10s --start-period=5m --retries=3 \
    CMD /app/.venv/bin/python /app/scripts/docker_healthcheck.py

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Cron does NOT inherit the container's environment variables. We dump them to
# a file that run_loop.sh sources before running the Python process.
# Only whitelisted variable names are exported to avoid leaking shell internals.
CMD ["sh", "-c", "env | grep -E '^(ALPACA_|LLM_|ANTHROPIC_|DB_PATH|TRADING_|DATA_|HEARTBEAT_|WEBHOOK_|LOOP_INTERVAL|TZ=|MAX_|MIN_|SCREENER_|PROMPT_|CONFIRM_|ALLOW_|STOP_|PROFIT_|OVER_|ANNUAL_|PERFORMANCE_|EARNINGS_|OPTIONS_)' > /etc/environment.sauce && INTERVAL=${LOOP_INTERVAL_MINUTES:-30} && printf \"*/${INTERVAL} * * * * root /app/scripts/run_loop.sh >> /app/data/logs/cron.log 2>&1\n\" > /etc/cron.d/sauce && chmod 0644 /etc/cron.d/sauce && /app/.venv/bin/python /app/scripts/log_container_start.py && cron && tail -F /app/data/logs/cron.log"]
