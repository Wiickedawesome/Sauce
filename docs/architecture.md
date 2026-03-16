# Architecture

Sauce is a fully autonomous, multi-agent trading system that runs on a 15-minute cron cadence. It uses Claude as the reasoning engine, Alpaca as the broker for US equities, crypto, and options, and SQLite as an append-only audit database.

---

## Pipeline Overview

Every 15 minutes, the loop orchestrator (`core/loop.py`) executes this sequence:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          SAUCE LOOP (every 15min)                       │
├──────────────┬───────────────────────────────────────────────────────────┤
│  Step 0      │  Session Boot (Pure Python)                              │
│              │  Wipe session memory if new day, load economic calendar, │
│              │  suppress trading if FOMC/CPI imminent                   │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 1      │  Market Context (Pure Python)                            │
│              │  Fetch SPY bars → classify regime → build narrative      │
│              │  Regimes: TRENDING_UP / TRENDING_DOWN / RANGING /        │
│              │           VOLATILE / DEAD                                │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 2      │  Research Agent (Claude — temp 0.3) per symbol           │
│              │  Compute indicators → multi-TF confluence → retrieve     │
│              │  similar past trades (RAG) → run setup scanner →         │
│              │  if setup passes: Claude audits thesis                   │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 3      │  Risk Agent (Pure Python) per signal                     │
│              │  Position limits, exposure, daily loss, buying power     │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 4      │  Execution Agent (Claude) per approved signal            │
│              │  Fresh quote, price deviation check, order construction  │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 5      │  Bull/Bear Debate (Deterministic — no LLM)              │
│              │  Both sides argue for/against using indicators;          │
│              │  confidence adjustment -0.10 to +0.05                    │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 6      │  Portfolio Agent (suggestions only)                      │
│              │  Reviews positions, generates rebalance suggestions      │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 7      │  Supervisor Agent (Claude — temp 0.2, final gate)       │
│              │  Hard pre-flight checks + Claude veto. Can only          │
│              │  downgrade, NEVER upgrade. Receives debate transcript    │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 8      │  Exit Research — for each open position                  │
│              │  Evaluate trailing stops, regime flips, exit signals     │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Step 8c     │  Options Pipeline (feature-flagged via OPTIONS_ENABLED)  │
│              │  Research → safety gates (IV, DTE, spread, delta,        │
│              │  exposure, max loss) → entry/exit → compound stage mgmt  │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Broker      │  Place orders only if Supervisor says "execute"          │
│              │  Companion stop-loss placed immediately                  │
├──────────────┼───────────────────────────────────────────────────────────┤
│  Ops         │  Audit trail, anomaly detection, circuit breakers        │
└──────────────┴───────────────────────────────────────────────────────────┘
```

---

## Module Map

```
sauce/
├── adapters/          External interface layer
│   ├── broker.py        Alpaca order placement + account queries
│   ├── market_data.py   OHLCV bars, quotes, snapshots via Alpaca Data API
│   ├── options_data.py  Options chains, quotes, greeks via Alpaca Options API
│   ├── llm.py           Claude via GitHub Models API (+ Anthropic fallback)
│   ├── db.py            SQLite engine, audit log, signal/order/options writers
│   ├── notify.py        Alert dispatch
│   └── utils.py         Adapter utilities
│
├── agents/            Agent implementations
│   ├── session_boot.py  Day-start initialization + calendar suppression
│   ├── market_context.py  Regime classification + intraday narrative
│   ├── research.py      Signal generation (Claude auditor)
│   ├── risk.py          Position sizing + exposure checks
│   ├── execution.py     Order construction + price validation
│   ├── debate.py        Deterministic bull/bear debate engine
│   ├── exit_research.py Exit signal evaluation for open positions
│   ├── options_research.py  Options signal generation (Claude)
│   ├── options_execution.py Options order construction
│   ├── options_exit.py  Options compound-stage exit engine
│   ├── portfolio.py     Portfolio analysis + rebalance suggestions
│   ├── supervisor.py    Final veto gate (Claude)
│   └── ops.py           Operational health + anomaly detection
│
├── core/              Business logic
│   ├── config.py        Pydantic v2 settings — single source of truth
│   ├── schemas.py       Typed models: Signal, Order, AuditEvent, etc.
│   ├── loop.py          Main orchestration loop
│   ├── setups.py        Deterministic setup scanner (hard/soft scoring)
│   ├── safety.py        TRADING_PAUSE, staleness, daily loss, market hours
│   ├── regime.py        Market regime classifier
│   ├── capital.py       Capital tier system
│   ├── metrics.py       Performance metrics
│   ├── nav.py           Net asset value tracking
│   ├── calendar.py      Economic calendar integration
│   ├── options_schemas.py Pydantic models for options (Position, ExitDecision, ...)
│   ├── options_config.py  Options settings (IV, DTE, delta, stages)
│   ├── options_safety.py  7 safety gates: IV rank, DTE, spread, delta, exposure
│   └── validation.py    Input validation
│
├── indicators/        Technical analysis
│   └── core.py          Pure-function indicator library (RSI, MACD, BB,
│                        ATR, VWAP, SMA, Stochastic, volume ratio)
│
├── signals/           Multi-timeframe signal engine
│   ├── timeframes.py    Fetch OHLCV at 5m/15m/1h/4h/1d + compute indicators
│   └── confluence.py    Weighted confluence scoring + signal tiers (S1–S4)
│
├── backtest/          Backtesting engine
│   ├── models.py        BacktestConfig, BacktestResult, BacktestTrade
│   ├── engine.py        Vectorized bar-replay with ATR-based exits
│   ├── options_models.py  Options backtest dataclasses + exit reasons
│   └── options_engine.py  Options bar-replay engine (delta+theta pricing)
│
├── memory/            Session + strategic memory (SQLite)
│   ├── db.py            ORM adapters, get_similar_trades() RAG retrieval
│   ├── learning.py      Post-trade outcome recording
│   ├── options_learning.py  Options post-trade analytics + win rate drift
│   └── narrative.py     Intraday narrative builder
│
└── prompts/           Prompt templates for Claude
    ├── context.py       Context builders (memory → plain English)
    ├── research.py      Research agent system + user prompts
    ├── options_research.py  Options research agent prompts
    ├── execution.py     Execution agent prompts
    ├── supervisor.py    Supervisor agent prompts
    └── utils.py         Prompt sanitization + injection protection
```

---

## Design Principles

1. **Claude is the auditor, not the analyst.** The rule engine scores setups and presents them to Claude for review. Claude finds contradictions, not trade ideas.

2. **Safety is non-negotiable.** The 8-layer gauntlet (data freshness → regime gate → setup scanner → research audit → risk check → debate → supervisor veto → broker validation) means no trade executes without passing every layer.

3. **Paper-first by default.** `ALPACA_PAPER=true` is the default. Live trading requires explicit opt-in.

4. **Append-only audit trail.** No UPDATE, no DELETE on the audit database. Full forensic trail of every decision.

5. **Zero external SaaS dependencies.** All indicators computed locally. Memory stored in SQLite. No cloud services beyond the broker and LLM APIs.
