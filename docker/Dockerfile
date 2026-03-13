# ──────────────────────────────────────────────────────────────────────────────
# Sauce — Trading System Docker Image
#
# Build:  docker build -f docker/Dockerfile -t sauce .
# Run:    docker compose -f docker/docker-compose.yml up -d
#
# The image installs cron and configures it to fire run_loop.sh every 30
# minutes. Cron runs as the foreground process (PID 1 via exec).
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
# Write the crontab entry. Output is appended to /app/data/logs/cron.log
# which is on the mounted volume and survives container restarts.
RUN printf "*/30 * * * * root /app/scripts/run_loop.sh >> /app/data/logs/cron.log 2>&1\n" \
        > /etc/cron.d/sauce \
    && chmod 0644 /etc/cron.d/sauce

# Ensure run_loop.sh is executable
RUN chmod +x /app/scripts/run_loop.sh

# ── Health check ──────────────────────────────────────────────────────────────
# Confirm cron is running AND that a loop has completed within the last 70
# minutes (just over 2 cron cycles). Queries the audit DB for the most recent
# loop_end event.
HEALTHCHECK --interval=5m --timeout=10s --start-period=5m --retries=3 \
    CMD /app/.venv/bin/python /app/scripts/docker_healthcheck.py

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Run cron in foreground so Docker can manage the process lifecycle.
# Logs flow to stdout via tail so `docker compose logs -f` works.
CMD ["sh", "-c", "cron && tail -F /app/data/logs/cron.log"]
