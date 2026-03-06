<div align="center">

# SAUCE

**Personal multi-agent AI-driven trading system**

<br/>

![Python](https://img.shields.io/badge/Python-3.13-3776ab?style=flat-square&logo=python&logoColor=white)
![Broker](https://img.shields.io/badge/Broker-Alpaca-ffde57?style=flat-square&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-Claude-cc785c?style=flat-square)
![DB](https://img.shields.io/badge/Database-SQLite-003b57?style=flat-square&logo=sqlite&logoColor=white)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ed?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-165%20passed-22c55e?style=flat-square)
![Paper](https://img.shields.io/badge/Mode-Paper%20First-f59e0b?style=flat-square)

<br/>

</div>

---

Sauce is a fully autonomous, multi-agent trading system that runs on a VPS on a 30-minute cron cadence. It uses Claude (via GitHub Models API) as the reasoning engine for every agent, Alpaca as the broker for US equities and crypto, and SQLite as an append-only audit database. No machine learning. No training. No external SaaS dependencies.

---

## Architecture

```
                         ┌─────────────────────────────────────────┐
                         │              core/loop.py               │
                         │   Orchestrates one full trading cycle   │
                         └──────────────────┬──────────────────────┘
                                            │
          ┌─────────────────────────────────▼──────────────────────────────────┐
          │                          Safety Checks                             │
          │  TRADING_PAUSE · daily loss limit · market hours · data freshness  │
          └─────────────────────────────────┬──────────────────────────────────┘
                                            │ per symbol
                    ┌───────────────────────▼────────────────────────┐
                    │             agents/research.py                  │
                    │  Reads bars + indicators, calls Claude, returns  │
                    │  Signal(side, confidence, evidence, rationale)   │
                    └───────────────────────┬────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────┐
                    │               agents/risk.py                    │
                    │  Checks position limits, portfolio exposure,    │
                    │  volatility, daily loss — returns RiskResult    │
                    └───────────────────────┬────────────────────────┘
                                            │ approved signals only
                    ┌───────────────────────▼────────────────────────┐
                    │             agents/execution.py                 │
                    │  Fetches live quote, validates price deviation, │
                    │  calls Claude — returns Order(type, qty, price) │
                    └───────────────────────┬────────────────────────┘
                                            │
                    ┌───────────────────────▼────────────────────────┐
                    │             agents/supervisor.py                │
                    │  Final veto gate. Claude reviews all orders     │
                    │  before any broker call is made                 │
                    └───────────────────────┬────────────────────────┘
                                            │ execute / abort
                    ┌───────────────────────▼────────────────────────┐
                    │              adapters/broker.py                 │
                    │  Places orders via alpaca-py. Every call is     │
                    │  wrapped in pre/post AuditEvents                │
                    └───────────────────────┬────────────────────────┘
                                            │
          ┌─────────────────────────────────▼──────────────────────────────────┐
          │                         adapters/db.py                             │
          │          Append-only SQLite audit log. No UPDATE. No DELETE.       │
          └────────────────────────────────────────────────────────────────────┘
```

### Supporting Agents

| Agent | Role |
|---|---|
| `agents/portfolio.py` | Monitors exposure, stop/target levels, concentration |
| `agents/ops.py` | Writes daily ops log, detects anomalies, raises alerts |

---

## Project Layout

```
sauce/
  adapters/
    broker.py         Alpaca order placement + account queries
    market_data.py    Bars, quotes, snapshots via Alpaca Data API
    llm.py            Claude via GitHub Models (+ Anthropic fallback)
    db.py             SQLite engine, audit log, signal/order writers
  agents/
    research.py       Signal generation
    risk.py           Risk checks + position sizing
    execution.py      Order construction + price validation
    supervisor.py     Final approval gate
    portfolio.py      Portfolio analysis
    ops.py            Operational health checks
  core/
    config.py         Pydantic v2 settings — single source of truth
    schemas.py        All typed models: Signal, Order, AuditEvent, ...
    safety.py         TRADING_PAUSE, staleness, daily loss, market hours
    loop.py           Main orchestration loop
  prompts/
    research.py       Research agent system + user prompt templates
    execution.py      Execution agent prompt templates
    supervisor.py     Supervisor agent prompt templates

tests/               165 tests, zero real API calls
scripts/
  run_loop.sh        Cron entrypoint
docker/
  Dockerfile         Python 3.13-slim + cron, venv, UTC timezone
  docker-compose.yml VPS deployment config
```

---

## Setup

**Requirements:** Python 3.13, Docker

```bash
# Clone
git clone https://github.com/Wiickedawesome/Sauce.git
cd Sauce

# Environment
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Secrets
cp .env.example .env
# Fill in ALPACA_API_KEY, ALPACA_SECRET_KEY, GITHUB_TOKEN
```

---

## Configuration

All configuration lives in `.env`. No value is hardcoded anywhere in the agent or adapter code.

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | — | Required. Alpaca key ID |
| `ALPACA_SECRET_KEY` | — | Required. Alpaca secret |
| `ALPACA_PAPER` | `true` | `false` to go live. Default is always paper |
| `LLM_PROVIDER` | `github` | `github` or `anthropic` |
| `GITHUB_TOKEN` | — | Bearer token for GitHub Models API |
| `LLM_MODEL` | `claude-3-5-sonnet` | Model name on the endpoint |
| `ANTHROPIC_API_KEY` | — | Required only when `LLM_PROVIDER=anthropic` |
| `TRADING_UNIVERSE_EQUITIES` | `AAPL,MSFT,GOOGL,...` | Comma-separated tickers |
| `TRADING_UNIVERSE_CRYPTO` | `BTC/USD,ETH/USD` | Alpaca crypto pairs |
| `MAX_POSITION_PCT` | `0.05` | Max allocation per symbol (fraction of NAV) |
| `MAX_PORTFOLIO_EXPOSURE` | `1.0` | Max total exposure (1.0 = no leverage) |
| `MAX_DAILY_LOSS_PCT` | `0.02` | Auto-pause threshold (2% daily drawdown) |
| `MIN_CONFIDENCE` | `0.5` | Signals below this are treated as hold |
| `DATA_TTL_SECONDS` | `120` | Max age of market data before it is considered stale |
| `MAX_PRICE_DEVIATION` | `0.01` | Execution vetoes if live quote deviates > 1% |
| `TRADING_PAUSE` | `false` | Set `true` to halt all trading immediately |
| `PROMPT_VERSION` | `v1` | Stamped on every LLM call for auditability |

---

## Running Locally

```bash
# Run one loop cycle manually
python -m sauce.core.loop

# Run tests
pytest

# Lint
ruff check sauce/ tests/

# Type check
mypy sauce/
```

---

## Deployment

```bash
# On the VPS — build and start
cd docker
docker compose up -d --build

# Live log tail (cron fires every 30 minutes)
docker compose logs -f

# Emergency pause (no restart needed)
# Edit .env: TRADING_PAUSE=true
docker compose restart
```

The container runs cron as PID 1. `run_loop.sh` fires every 30 minutes, activates the venv, and executes `python -m sauce.core.loop`. All output is written to `data/logs/cron.log`, which is on the host-mounted volume and survives container restarts.

---

## Safety Model

Every loop run passes through a layered set of guards before any order can reach the broker:

```
1. TRADING_PAUSE flag          -- hard stop, checked first, no exceptions
2. Market hours check          -- equities: 09:30-16:00 ET weekdays only
3. Daily loss limit            -- auto-pause when drawdown exceeds threshold
4. Data freshness check        -- stale data blocks the signal, not just the order
5. Confidence floor            -- sub-threshold signals become hold before risk
6. Risk agent veto             -- position limits, exposure, volatility
7. Price deviation check       -- execution aborts if quote moved since signal
8. Supervisor veto             -- Claude reviews all orders before broker call
```

No order reaches `broker.place_order()` without passing all eight layers. Every layer writes an `AuditEvent` to SQLite before and after. The audit table is append-only — no `UPDATE` or `DELETE` is ever issued.

---

## Security

- Credentials are loaded exclusively through `core/config.py` — never via `os.environ` directly in agent code
- `.env` is in `.gitignore` and is never committed
- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are masked as `***` in all log output
- `GITHUB_TOKEN` is treated as a password — masked in every log line
- `ALPACA_PAPER` defaults to `true` even if the env var is missing or empty

---

<div align="center">

![Paper Trading](https://img.shields.io/badge/Always%20start-paper%20trading-f59e0b?style=flat-square)
![No ML](https://img.shields.io/badge/No%20local%20ML-Claude%20only-cc785c?style=flat-square)
![Append Only](https://img.shields.io/badge/Audit%20log-append%20only-3776ab?style=flat-square)

</div>
