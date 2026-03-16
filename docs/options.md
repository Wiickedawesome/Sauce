# Options Trading

Sauce includes a feature-flagged options trading module that runs alongside the equity/crypto pipeline. It implements a "Double Up & Take Gains" compounding strategy with multi-stage profit targets and a dedicated safety gauntlet.

**Status:** Feature-flagged. Set `OPTIONS_ENABLED=true` in `.env` to activate.

---

## Strategy: Double Up & Take Gains

The compounding strategy works through profit stages:

```
Entry → 2x gain → sell 50% → 4x gain → sell 50% of remainder → 8x gain → sell/trail

Stage 1:  Position doubles  →  sell 50%  →  activate trailing stop
Stage 2:  Remaining doubles →  sell 50%  →  tighten trailing stop
Stage 3:  Remaining doubles →  sell rest  →  or ride with tight trail
```

Each stage's trigger multiplier, sell fraction, and trailing stop tightness are configurable. The system tracks compound stages per position in the database and evaluates exits every loop iteration.

---

## Architecture

The options pipeline runs as **Step 8c** in the main loop, gated by `OPTIONS_ENABLED`:

```
Step 8c — Options Pipeline (runs only if OPTIONS_ENABLED=true)
├── 1. Load open options positions from DB
├── 2. Evaluate each position for exit (compound stage triggers, hard stop, trailing stop)
├── 3. Execute exits through Supervisor approval
├── 4. Run options research (Claude) for new entry signals
├── 5. Run 7-layer safety gauntlet on each signal
├── 6. Build entry orders and submit through Supervisor
└── 7. Persist new positions to DB
```

All options orders go through the same Supervisor approval gate as equity orders.

---

## Module Map

```
sauce/
├── core/
│   ├── options_schemas.py    8 Pydantic models (Position, ExitDecision, ...)
│   ├── options_config.py     OptionsSettings (all thresholds from .env)
│   └── options_safety.py     7 entry safety gates
├── agents/
│   ├── options_research.py   Signal generation via Claude
│   ├── options_execution.py  Limit order construction
│   └── options_exit.py       Compound-stage exit evaluation engine
├── adapters/
│   ├── options_data.py       Options chain + quote fetcher (Alpaca)
│   └── db.py                 OptionsPositionRow ORM + CRUD
├── signals/
│   └── options_confluence.py Options signal confluence scoring
├── memory/
│   └── options_learning.py   Post-trade analytics + win rate drift detection
├── backtest/
│   ├── options_models.py     Backtest dataclasses + exit reasons
│   └── options_engine.py     Bar-replay engine (delta+theta pricing model)
└── prompts/
    └── options_research.py   Options research agent prompt templates
```

---

## Safety Gates

Every options entry passes through 7 sequential safety checks. The pipeline stops at the first failure:

| # | Gate | What it checks |
|---|---|---|
| 1 | `check_options_enabled` | Master switch — `OPTIONS_ENABLED` must be `true` |
| 2 | `check_iv_rank` | IV rank must be below `OPTION_MAX_IV_RANK` (default 0.70) |
| 3 | `check_dte` | Days to expiry within `[OPTION_MIN_DTE, OPTION_MAX_DTE]` range |
| 4 | `check_spread` | Bid-ask spread < `OPTION_MAX_SPREAD_PCT` of mid price |
| 5 | `check_delta` | Absolute delta within `[OPTION_MIN_DELTA, OPTION_MAX_DELTA]` |
| 6 | `check_exposure` | Total options exposure < `OPTION_MAX_TOTAL_EXPOSURE` of NAV |
| 7 | `check_max_loss` | Computed max loss below `OPTION_MAX_LOSS_PCT` threshold |

Each gate returns a typed `(passed, reason)` result. All results are logged to the audit trail.

---

## Exit Engine

The exit engine (`options_exit.py`) evaluates each open position every loop:

1. **Compound stage check** — Has the position reached the next profit target? If so, sell the configured fraction and advance the stage counter.
2. **Hard stop** — Has the position lost more than `OPTION_MAX_LOSS_PCT` from entry? Exit immediately.
3. **Trailing stop** — After any compound stage triggers, a trailing stop activates. It ratchets up with new highs and never moves down. Tightness increases with each stage.

Exit decisions are one of: `HOLD`, `PARTIAL_CLOSE` (compound stage), `FULL_CLOSE` (stop hit), or `CLOSE` (expiry/DTE).

---

## Persistence

Options positions are persisted to SQLite via `OptionsPositionRow` in `adapters/db.py`:

- **On entry:** Position saved with compound stages serialized as JSON
- **On partial exit:** `remaining_qty`, `stages_completed`, `trailing_stop_price` updated
- **On full exit:** Position marked `status="closed"` with realized PnL and timestamp

This allows the exit engine to track positions across loop iterations and container restarts.

---

## Configuration

All options settings live in `.env` and are loaded via `OptionsSettings` (Pydantic v2):

| Variable | Default | Description |
|---|---|---|
| `OPTIONS_ENABLED` | `false` | Master switch for entire options module |
| `OPTIONS_UNIVERSE` | `SPY,QQQ,AAPL,TSLA,NVDA` | Eligible tickers for options |
| `OPTION_PROFIT_MULTIPLIER` | `2.0` | Gain trigger per stage (2.0 = 100% gain) |
| `OPTION_COMPOUND_STAGES` | `3` | Number of take-profit stages |
| `OPTION_SELL_FRACTION` | `0.5` | Fraction to sell at each stage |
| `OPTION_MAX_DTE` | `45` | Max days to expiry at entry |
| `OPTION_MIN_DTE` | `7` | Min days to expiry at entry |
| `OPTION_MAX_IV_RANK` | `0.70` | Max IV rank (above = IV too expensive) |
| `OPTION_MIN_DELTA` | `0.30` | Min absolute delta |
| `OPTION_MAX_DELTA` | `0.60` | Max absolute delta |
| `OPTION_MAX_POSITION_PCT` | `0.05` | Max NAV fraction per position (5%) |
| `OPTION_MAX_TOTAL_EXPOSURE` | `0.20` | Max NAV across all options (20%) |
| `OPTION_MAX_SPREAD_PCT` | `0.05` | Max bid-ask spread as pct of mid |
| `OPTION_MAX_LOSS_PCT` | `0.50` | Hard stop: max loss per position |
| `OPTION_TRAILING_STOP_STAGE1` | `0.20` | Trailing stop after stage 1 |
| `OPTION_TRAILING_STOP_STAGE2` | `0.15` | Trailing stop after stage 2 |
| `OPTION_TRAILING_STOP_STAGE3` | `0.10` | Trailing stop after stage 3 |

---

## Backtesting

The options backtest engine (`backtest/options_engine.py`) provides bar-replay simulation:

- **Pricing model:** Delta approximation + theta decay (no full Black-Scholes needed)
- **Entry signals:** RSI-based (RSI < 35 → long call, RSI > 65 → long put)
- **Exit simulation:** Full compound stage logic, hard stops, trailing stops, theta decay
- **Metrics:** Win rate, Sharpe ratio, max drawdown, profit factor, equity curve

```python
from sauce.backtest.options_engine import run_options_backtest
from sauce.backtest.options_models import OptionsBacktestConfig

result = run_options_backtest("SPY", df, OptionsBacktestConfig())
print(f"Win rate: {result.win_rate:.1%}, Sharpe: {result.sharpe_ratio:.2f}")
```

---

## Learning & Analytics

Post-trade analytics (`memory/options_learning.py`) tracks:

- **Trade outcomes:** PnL, hold duration, exit reason per position
- **Win rate drift:** Detects when recent win rate drops below threshold
- **Stage performance:** Groups closed trades by stages_completed (how far through the compounding ladder)
- **Underlying performance:** Aggregates results by ticker symbol
