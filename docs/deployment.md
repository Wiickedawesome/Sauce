# Deployment

Sauce uses **Supabase** (PostgreSQL) for persistent storage. The trading loop runs locally
or can be scheduled via cron/systemd on any machine with Python 3.13+.

---

## Prerequisites

- Python 3.13+
- Alpaca account (paper or live)
- Anthropic API key
- Supabase project (free tier works)

---

## Quick Start

### 1. Create Supabase Project

1. Go to [supabase.com](https://supabase.com) → New Project
2. Note your project reference ID (e.g., `gomtjyldzaelbhqboons`)
3. Get credentials from **Settings → API**:
   - `SUPABASE_URL` — Project URL
   - `SUPABASE_SERVICE_ROLE_KEY` — Service role key (keep secret!)
4. Get database URL from **Settings → Database → Connection string (URI)**

### 2. Deploy Database Schema

```bash
# Install Supabase CLI (if needed)
brew install supabase/tap/supabase  # macOS
# or: npm install -g supabase

# Link to your project
supabase link --project-ref YOUR_PROJECT_REF

# Push migrations
supabase db push
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials:
#   ALPACA_API_KEY, ALPACA_SECRET_KEY
#   ANTHROPIC_API_KEY
#   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_DB_URL
```

### 4. Install & Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run once
python -m sauce.loop

# Or schedule via cron (every 15 min)
# */15 * * * * cd /path/to/Sauce && .venv/bin/python -m sauce.loop >> data/logs/cron.log 2>&1
```

---

## Database Schema

Tables are defined in `supabase/migrations/`:

| Table | Purpose |
|-------|---------|
| `trades` | Completed trades with P&L |
| `positions` | Open positions |
| `signal_log` | Every scoring result (auto-purged after 90 days) |
| `daily_summary` | Daily aggregates |
| `instrument_meta` | Per-instrument config/regime cache |
| `trade_memories` | BM25 trade reflections |
| `options_positions` | Open options positions |
| `options_trades` | Completed options trades |
| `audit_events` | System events (auto-purged after 90 days) |

Helpful views: `v_today_signals`, `v_open_positions`, `v_recent_trades`, `v_strategy_performance`

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `ALPACA_API_KEY` | Yes | — | Alpaca key ID |
| `ALPACA_SECRET_KEY` | Yes | — | Alpaca secret |
| `ALPACA_PAPER` | No | `true` | Set `false` for live trading |
| `ANTHROPIC_API_KEY` | Yes | — | Anthropic API key |
| `LLM_MODEL` | No | `claude-sonnet-4-6` | Anthropic model name |
| `SUPABASE_URL` | No | — | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | No | — | Supabase service role key |
| `SUPABASE_DB_URL` | No | — | PostgreSQL connection string |
| `OPTIONS_ENABLED` | No | `false` | Enable options trading |
| `TRADING_PAUSE` | No | `false` | Emergency kill switch |

See `sauce/core/config.py` for the complete list.

---

## Operations

```bash
# Run diagnostics
python scripts/diagnose.py

# Query Supabase directly
supabase db query "SELECT COUNT(*) FROM signal_log"

# View recent signals
supabase db query "SELECT * FROM v_today_signals LIMIT 10"

# Check open positions
supabase db query "SELECT * FROM v_open_positions"

# Emergency pause — edit .env
TRADING_PAUSE=true
```

---

## Data Retention

Signal logs and audit events are automatically purged after 90 days via the
`purge_old_records()` PostgreSQL function. Run manually if needed:

```sql
SELECT purge_old_records();
```

---

## Local Development

```bash
# Run tests (uses isolated temp DBs)
pytest

# Type check
mypy sauce/

# Lint
ruff check sauce/
```

---

## Backup & Recovery

### Export data from Supabase

```bash
# Dump entire database
pg_dump "$SUPABASE_DB_URL" > backup.sql

# Export specific tables
supabase db query "SELECT * FROM trades" --csv > trades.csv
```

### Restore to new project

```bash
# Create new Supabase project, then:
supabase link --project-ref NEW_PROJECT_REF
supabase db push
psql "$NEW_SUPABASE_DB_URL" < backup.sql
```

---

## Scheduling (GitHub Actions)

Sauce runs on **GitHub Actions** — no VPS needed. The workflow is in `.github/workflows/trading-loop.yml`.

### Setup

1. Go to **Settings → Secrets → Actions** in your GitHub repo
2. Add these repository secrets:
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
   - `ANTHROPIC_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_ROLE_KEY`
   - `SUPABASE_DB_URL`
3. Push to `main` — the workflow auto-enables

### Schedule

| Market | Schedule | Notes |
|--------|----------|-------|
| Equities | Weekdays 9am-4pm ET | Every 15 min |
| Crypto | 24/7 | Every 15 min (optional — comment out to save minutes) |

### Manual trigger

Go to **Actions → Trading Loop → Run workflow** for an immediate cycle.

### Costs

GitHub free tier = 2000 minutes/month. Each cycle ~2-3 min → ~96 runs/day = ~6000 min/month.
For heavy usage, consider GitHub Pro or a $5/mo Railway deployment.

The loop handles market hours internally, so running outside hours is safe (exits early).
