# Deployment

Sauce is deployed as a Docker container on a VPS (Contabo / DigitalOcean / any Linux host),
running on a configurable cron schedule. Pushes to `main` auto-deploy via GitHub Actions.

---

## Prerequisites

- Docker and Docker Compose
- Alpaca account (paper or live)
- Anthropic API key
- A VPS or local machine with persistent storage

---

## Quick Start (fresh server)

```bash
# 1. Bootstrap the VPS (run from your LOCAL machine)
ssh root@YOUR_VPS_IP 'bash -s' < scripts/bootstrap_server.sh

# 2. Copy your .env to the server
scp .env root@YOUR_VPS_IP:/root/Sauce/.env

# 3. Set GitHub repo secrets (Settings → Secrets → Actions):
#    VPS_HOST     = your server IP
#    VPS_USER     = root
#    VPS_SSH_KEY  = the private key printed by bootstrap
#    VPS_APP_PATH = /root/Sauce

# 4. Push to main — GitHub Actions builds, deploys, and verifies.
git push origin main
```

### Manual start (without CI/CD)

```bash
ssh root@YOUR_VPS_IP
cd /root/Sauce
cp .env.example .env   # edit with your API keys
docker compose -f docker/docker-compose.yml up -d --build
docker compose -f docker/docker-compose.yml logs -f
```

---

## CI/CD Pipeline

Every push to `main` triggers `.github/workflows/deploy.yml`:

1. **Test** — runs `pytest` on GitHub-hosted runner
2. **Deploy** — SSHes to VPS, `git pull`, `docker compose up --build -d`, prune old images
3. **Verify** — SSHes in to confirm container is `running`

A separate **Monitor** workflow (`.github/workflows/monitor.yml`) runs every 2 hours
and triggers a GitHub notification if the container is down or the loop is stale.

### Required secrets

| Secret | Example | Description |
|---|---|---|
| `VPS_HOST` | `123.45.67.89` | Server IP or hostname |
| `VPS_USER` | `root` | SSH user |
| `VPS_SSH_KEY` | (PEM content) | Deploy private key |
| `VPS_APP_PATH` | `/root/Sauce` | Absolute path to repo on server |

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
| `OPTIONS_ENABLED` | No | `false` | Enable the integrated options pipeline explicitly |
| `TRADING_PAUSE` | No | `false` | Emergency kill switch |
| `DB_PATH` | No | `data/sauce.db` | SQLite database path |

See `sauce/core/config.py` for the complete list of configuration options.

---

## Operations

```bash
# Tail live logs (via SSH)
ssh root@YOUR_VPS_IP "docker compose -f /root/Sauce/docker/docker-compose.yml logs -f"

# Emergency pause (no restart needed)
# Edit .env on server: TRADING_PAUSE=true
ssh root@YOUR_VPS_IP "docker compose -f /root/Sauce/docker/docker-compose.yml restart"

# Health check
ssh root@YOUR_VPS_IP "docker exec sauce /app/.venv/bin/python scripts/docker_healthcheck.py"

# Diagnostics
ssh root@YOUR_VPS_IP "docker exec sauce /app/.venv/bin/python scripts/diagnose.py"
```

---

## Migration between providers

To move to a new VPS:
1. Run `scripts/bootstrap_server.sh` on the new server
2. Copy `.env` and `data/` directory (SQLite DBs + logs) to the new server
3. Update GitHub secrets (`VPS_HOST`) to point to new IP
4. Push to `main` — auto-deploys to the new server

```bash
# Copy data from old server to new
scp -r root@OLD_IP:/root/Sauce/data root@NEW_IP:/root/Sauce/data
scp root@OLD_IP:/root/Sauce/.env root@NEW_IP:/root/Sauce/.env
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
