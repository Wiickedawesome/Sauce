<div align="center">

# SAUCE

**Personal AI-driven trading system**

<br/>

![Python](https://img.shields.io/badge/Python-3.13-3776ab?style=flat-square&logo=python&logoColor=white)
![Broker](https://img.shields.io/badge/Broker-Alpaca-ffde57?style=flat-square&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-Claude-cc785c?style=flat-square)
![DB](https://img.shields.io/badge/Database-SQLite-003b57?style=flat-square&logo=sqlite&logoColor=white)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ed?style=flat-square&logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-GPLv3-22c55e?style=flat-square)
![Tests](https://img.shields.io/badge/Tests-168%20passed-22c55e?style=flat-square)
![Paper](https://img.shields.io/badge/Mode-Paper%20First-f59e0b?style=flat-square)

<br/>

</div>

---

Sauce is an autonomous trading system that runs on a VPS on a cron cadence. It uses Claude (via the Anthropic API) as the reasoning engine, Alpaca as the broker for US equities and crypto, and SQLite for trade logging and audit. No machine learning. No training.

---

## Key Features

- **Technical indicator library** — RSI, MACD, Bollinger Bands, ATR, VWAP, SMA, Stochastic, volume ratio (via pandas-ta)
- **Strategy framework** — pluggable strategies with signal generation and LLM-assisted analysis
- **Risk management** — position limits, portfolio exposure caps, daily loss circuit breakers, ATR-based volatility gates, spread/liquidity checks
- **Exit monitoring** — active position monitoring with trailing stops, time stops, RSI exhaustion, profit targets
- **Morning brief** — LLM-generated market context at session start
- **168 tests, zero real API calls** — full test coverage with no external dependencies
- **Paper-first default** — live trading requires explicit opt-in
- **Append-only SQLite audit log** — full forensic trail of every decision

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                       SAUCE LOOP (cron cadence)                     │
├──────────────┬───────────────────────────────────────────────────────┤
│  Morning Brief  Market context via LLM                              │
│  Strategy       Indicators → signal generation → LLM analysis       │
│  Risk           Position limits, exposure, volatility, liquidity    │
│  Execution      Live quote → price validation → order construction  │
│  Exit Monitor   Trailing stops, time stops, profit targets          │
│  Broker         Place orders via Alpaca                             │
│  Audit          Append-only event log                               │
└──────────────┴───────────────────────────────────────────────────────┘
```

---

## Project Layout

```
sauce/
  loop.py              Main orchestration loop
  strategy.py          Strategy runner + signal generation
  risk.py              Risk checks + position sizing
  exit_monitor.py      Exit signal evaluation for open positions
  morning_brief.py     LLM-generated market context
  db.py                Trade/position/signal tables (new engine)
  strategies/
    crypto_momentum.py Crypto momentum strategy implementation
  adapters/
    broker.py          Alpaca order placement + account queries
    market_data.py     Bars, quotes, snapshots via Alpaca Data API
    llm.py             Claude via Anthropic API (with retry + back-off)
    db.py              SQLite engine, audit log
    utils.py           Shared adapter utilities
  core/
    config.py          Pydantic v2 settings — single source of truth
    schemas.py         Typed models: Order, AuditEvent, Indicators, ...
  indicators/
    core.py            RSI, MACD, BB, ATR, VWAP, SMA, Stochastic, vol ratio

tests/                 168 tests, zero real API calls
scripts/               Cron entry, health checks, diagnostics
docker/                Dockerfile + docker-compose.yml
docs/                  Deployment guide, indicator reference
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

All configuration lives in `.env`. No value is hardcoded in source code.

| Variable | Default | Description |
|---|---|---|
| `ALPACA_API_KEY` | — | Required. Alpaca key ID |
| `ALPACA_SECRET_KEY` | — | Required. Alpaca secret |
| `ALPACA_PAPER` | `true` | `false` to go live. Default is always paper |
| `ANTHROPIC_API_KEY` | — | Required. Anthropic API key |
| `LLM_MODEL` | `claude-sonnet-4-6` | Anthropic model name |
| `TRADING_UNIVERSE_EQUITIES` | `AAPL,MSFT,GOOGL,...` | Comma-separated tickers |
| `TRADING_UNIVERSE_CRYPTO` | `BTC/USD,ETH/USD,...` | Alpaca crypto pairs |
| `MAX_POSITION_PCT` | `0.08` | Max allocation per symbol |
| `MAX_PORTFOLIO_EXPOSURE` | `1.0` | Max total exposure (1.0 = no leverage) |
| `MAX_DAILY_LOSS_PCT` | `0.03` | Auto-pause threshold (3% daily drawdown) |
| `TRADING_PAUSE` | `false` | Set `true` to halt all trading immediately |
| `OPTIONS_ENABLED` | `false` | Enable the options entry/exit pipeline explicitly |

See `sauce/core/config.py` for the complete list of configuration options.

---

## Running

```bash
# Run one loop cycle
python -m sauce.loop

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
| [Indicators](docs/indicators.md) | Technical indicator library reference |
| [Deployment](docs/deployment.md) | Docker setup, environment variables, operations |

---

## License

[GNU General Public License v3.0](LICENSE)

The container runs cron as PID 1. `run_loop.sh` fires every 15 minutes, activates the venv, and executes `python -m sauce.core.loop`. All output is written to `data/logs/cron.log`, which is on the host-mounted volume and survives container restarts.

---

## Safety Model

Every loop run passes through a layered set of guards before any order can reach the broker:

```
1. TRADING_PAUSE flag          -- hard stop, checked first, no exceptions
2. Market hours check          -- equities: 09:30-16:00 ET weekdays only
3. Daily loss limit            -- cycle halts when drawdown exceeds threshold
4. Data freshness check        -- stale quotes block the symbol before execution
5. Confidence floor            -- analyst approvals below configured confidence are rejected
6. Risk gate                   -- daily loss, position count, buying power, exposure
7. Price deviation check       -- execution aborts if the live quote drifted since signal
8. Supervisor preflight        -- final veto before broker.place_order()
```
No order reaches `broker.place_order()` without passing all eight layers. Loop boundaries, safety checks, broker actions, and supervisor decisions are written to append-only SQLite audit events.

Options trading is integrated into `sauce.loop`, including contract selection, order submission, local position persistence, and exit scanning. It still defaults to `false` so options remain an explicit opt-in.
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
