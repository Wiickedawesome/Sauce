# Deployment

Sauce is deployed as a Docker container on a VPS, running on a 15-minute cron schedule.

---

## Prerequisites

- Docker and Docker Compose
- Alpaca account (paper or live)
- Anthropic API key
- A VPS or local machine with persistent storage

---

## Quick Start

```bash
# Clone and configure
git clone https://github.com/Wiickedawesome/Sauce.git
cd Sauce
cp .env.example .env
# Edit .env with your API keys

# Deploy
cd docker
docker compose up -d --build

# Verify
docker compose logs -f
```

---

## Docker Architecture

```
docker/
├── Dockerfile           Python 3.13-slim + cron, UTC timezone
└── docker-compose.yml   Service definition, volume mounts
```

- `.env` is mounted **read-only** — secrets never bake into the image
- `./data` is mounted **read-write** — SQLite DB and logs survive restarts
- Cron fires every 15 minutes via `scripts/run_loop.sh`
- Container restarts automatically (`restart: unless-stopped`)

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ALPACA_API_KEY` | Yes | — | Alpaca key ID |
| `ALPACA_SECRET_KEY` | Yes | — | Alpaca secret |
| `ALPACA_PAPER` | No | `true` | Set `false` for live trading |
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `LLM_MODEL` | No | `claude-sonnet-4-6` | Anthropic model name |
| `TRADING_PAUSE` | No | `false` | Emergency kill switch |
| `OPTIONS_ENABLED` | No | `false` | Enable options trading pipeline |
| `DB_PATH` | No | `data/sauce.db` | SQLite database path |

See `sauce/core/config.py` for the complete list of configuration options.

---

## Operations

```bash
# Tail live logs
docker compose logs -f

# Emergency pause (no restart needed)
# Edit .env: TRADING_PAUSE=true
docker compose restart

# Health check
docker exec sauce python scripts/docker_healthcheck.py

# Resume after pause
python scripts/resume_trading.py

# Diagnostics
python scripts/diagnose.py
```

---

## Cron Environment

Cron doesn't inherit container environment variables. The Dockerfile handles this by:
1. Dumping whitelisted env vars to `/etc/environment.sauce` at container start
2. `run_loop.sh` sources this file before running Python

This ensures API keys and configuration reach the trading loop.

---

## Data Persistence

The SQLite database (`data/sauce.db`) and logs (`data/logs/`) are stored on a Docker volume mapped to the host. Database backups are timestamped in the `data/` directory.
