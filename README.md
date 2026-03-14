<div align="center">

# SAUCE

**Personal multi-agent AI-driven trading system**

<br/>

![Python](https://img.shields.io/badge/Python-3.13-3776ab?style=flat-square&logo=python&logoColor=white)
![Broker](https://img.shields.io/badge/Broker-Alpaca-ffde57?style=flat-square&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-Claude-cc785c?style=flat-square)
![DB](https://img.shields.io/badge/Database-SQLite-003b57?style=flat-square&logo=sqlite&logoColor=white)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ed?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-GPLv3-22c55e?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-856%20passed-22c55e?style=flat-square)
![Paper](https://img.shields.io/badge/Mode-Paper%20First-f59e0b?style=flat-square)

<br/>

</div>

---

Sauce is a fully autonomous, multi-agent trading system that runs on a VPS on a 30-minute cron cadence. It uses Claude (via GitHub Models API) as the reasoning engine, Alpaca as the broker for US equities and crypto, and SQLite as an append-only audit database. No machine learning. No training. No external SaaS dependencies.

---

## Key Features

- **8-layer safety gauntlet** — data freshness, market hours, regime gate, setup scanner, Claude audit, risk check, bull/bear debate, supervisor veto
- **Claude as auditor** — the rule engine generates trade theses; Claude reviews them for contradictions
- **Multi-timeframe signal engine** — confluence scoring across 5m, 15m, 1h, 4h, 1d with weighted S1–S4 signal tiers
- **Technical indicator library** — RSI, MACD, Bollinger Bands, ATR, VWAP, SMA, Stochastic, volume ratio (via pandas-ta)
- **Bull/Bear debate layer** — deterministic adversarial arguments for/against every trade, forwarded to the Supervisor
- **Backtesting engine** — vectorized bar-replay with ATR-based stop-loss and take-profit
- **Memory & RAG** — session memory (resets daily) + strategic memory (persistent). Top-K similar past trades retrieved and injected into Claude's prompt for confidence calibration
- **Exit management** — active position monitoring with trailing stops and regime-flip detection
- **856 tests, zero real API calls** — full test coverage with no external dependencies
- **Paper-first default** — live trading requires explicit opt-in
- **Append-only SQLite audit log** — full forensic trail of every decision

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       SAUCE LOOP (every 30min)                      │
├──────────────┬───────────────────────────────────────────────────────┤
│  Session Boot│  Init day, load economic calendar, macro suppression  │
│  Mkt Context │  SPY bars → regime classification → narrative         │
│  Research    │  Indicators → multi-TF confluence → RAG retrieval →   │
│              │  setup scanner → Claude audit (temp 0.3)              │
│  Risk        │  Position limits, exposure, daily loss checks         │
│  Debate      │  Deterministic bull/bear arguments, confidence adj    │
│  Execution   │  Live quote → price validation → order construction   │
│  Supervisor  │  Hard pre-flight + Claude veto (temp 0.2)             │
│  Exit Mgmt   │  Open position evaluation, trailing stops             │
│  Broker      │  Place orders + companion stop-loss                   │
│  Ops         │  Audit trail, anomaly detection, circuit breakers     │
└──────────────┴───────────────────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full pipeline breakdown.

---

## Project Layout

```
sauce/
  adapters/
    broker.py          Alpaca order placement + account queries
    market_data.py     Bars, quotes, snapshots via Alpaca Data API
    llm.py             Claude via GitHub Models (+ Anthropic fallback)
    db.py              SQLite engine, audit log, signal/order writers
    notify.py          Alert dispatch
  agents/
    session_boot.py    Day-start initialization + calendar suppression
    market_context.py  Regime classification + intraday narrative
    research.py        Signal generation (Claude auditor role)
    risk.py            Risk checks + position sizing
    execution.py       Order construction + price validation
    debate.py          Deterministic bull/bear debate engine
    exit_research.py   Exit signal evaluation for open positions
    supervisor.py      Final approval gate (Claude)
    portfolio.py       Portfolio analysis
    ops.py             Operational health checks
  core/
    config.py          Pydantic v2 settings — single source of truth
    schemas.py         All typed models: Signal, Order, AuditEvent, ...
    safety.py          TRADING_PAUSE, staleness, daily loss, market hours
    loop.py            Main orchestration loop
    setups.py          Deterministic setup scanner (hard/soft scoring)
    regime.py          Market regime classifier
    capital.py         Capital tier system
  indicators/
    core.py            RSI, MACD, BB, ATR, VWAP, SMA, Stochastic, vol ratio
  signals/
    timeframes.py      Multi-TF OHLCV fetch + indicators (5m–1d)
    confluence.py      Weighted confluence scoring + signal tiers (S1–S4)
  backtest/
    models.py          BacktestConfig, BacktestResult, BacktestTrade
    engine.py          Vectorized bar-replay backtesting
  memory/
    db.py              Session + strategic memory ORM, RAG retrieval
    learning.py        Post-trade outcome recording
    narrative.py       Intraday narrative builder
  prompts/
    context.py         Memory → plain English context builders
    research.py        Research agent system + user prompt templates
    execution.py       Execution agent prompt templates
    supervisor.py      Supervisor agent prompt templates

tests/                 856 tests, zero real API calls
scripts/               Cron entry, health checks, diagnostics
docker/                Dockerfile + docker-compose.yml
docs/                  Architecture, features, deployment guides
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
# Fill in ALPACA_API_KEY, ALPACA_SECRET_KEY, ANTHROPIC_API_KEY
```

---

## Configuration

All configuration lives in `.env`. No value is hardcoded in agent or adapter code.

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | — | Required. Alpaca key ID |
| `ALPACA_SECRET_KEY` | — | Required. Alpaca secret |
| `ALPACA_PAPER` | `true` | `false` to go live. Default is always paper |
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key |
| `LLM_MODEL` | `claude-sonnet-4-6` | Anthropic model name |
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

## Running

```bash
# Run one loop cycle
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
# On the VPS
cd docker
docker compose up -d --build
docker compose logs -f

# Emergency pause (no restart needed)
# Edit .env: TRADING_PAUSE=true
docker compose restart
```

See [docs/deployment.md](docs/deployment.md) for the full deployment guide.

---

## Documentation

| Doc | Description |
|---|---|
| [Architecture](docs/architecture.md) | Full pipeline, module map, design principles |
| [Indicators](docs/indicators.md) | Technical indicator library reference |
| [Signals](docs/signals.md) | Multi-timeframe engine, confluence scoring, signal tiers |
| [Backtesting](docs/backtesting.md) | Bar-replay engine, configuration, exit reasons |
| [Debate](docs/debate.md) | Bull/bear debate layer, arguments, integration |
| [Memory & RAG](docs/memory.md) | Session/strategic memory, similar trade retrieval |
| [Safety & Risk](docs/safety.md) | 8-layer gauntlet, risk parameters, anomaly detection |
| [Deployment](docs/deployment.md) | Docker setup, environment variables, operations |

---

## License

[GNU General Public License v3.0](LICENSE)

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
- `ALPACA_PAPER` defaults to `true` even if the env var is missing or empty

---

<div align="center">

![Paper Trading](https://img.shields.io/badge/Always%20start-paper%20trading-f59e0b?style=flat-square)
![No ML](https://img.shields.io/badge/No%20local%20ML-Claude%20only-cc785c?style=flat-square)
![Append Only](https://img.shields.io/badge/Audit%20log-append%20only-3776ab?style=flat-square)

</div>
