# Options Trading

Sauce includes a feature-flagged options trading module that runs alongside the equity/crypto pipeline. It implements a **Momentum Snipe** strategy — quick-strike directional calls with defined profit targets, a trailing stop, and 7 priority-ordered exit conditions.

**Status:** Feature-flagged. Set `OPTIONS_ENABLED=true` in `.env` to activate.

---

## Strategy: Momentum Snipe

The strategy targets short-duration momentum plays on high-liquidity names with affordable contracts (≤ $500 per contract). Positions are sized dynamically based on `max_position_cost` and closed through a structured exit ladder:

```
Entry → monitor 7 exit conditions every loop iteration

  +35% gain, qty ≥ 2  →  sell half (PARTIAL_CLOSE)  →  activate trailing stop on remainder
  +35% gain, qty = 1  →  can't split → activate trailing stop instead
  +60% gain           →  sell all  (FULL_CLOSE — stretch target)
  Trailing stop hit   →  sell all  (FULL_CLOSE — trailing stop)
  -25% from entry     →  sell all  (FULL_CLOSE — hard stop)
  Regime turns hostile →  sell all  (FULL_CLOSE — regime stop)
  DTE ≤ 5             →  sell all  (FULL_CLOSE — DTE stop)
  Held > 5 days with   →  sell all  (FULL_CLOSE — time stop)
    < 10% gain
```

The trailing stop activates at +20% gain and trails at 12% below the high-water mark. Once active, it ratchets up with new highs and never moves down.

---

## Architecture

The options pipeline runs as **Step 8c** in the main loop, gated by `OPTIONS_ENABLED`:

```
Step 8c — Options Pipeline (runs only if OPTIONS_ENABLED=true)
├── 1. Load open options positions from DB
├── 2. Evaluate each position for exit (7-condition priority check)
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
│   ├── options_schemas.py    Pydantic models (Position, ExitDecision, ...)
│   ├── options_config.py     OptionsSettings (all thresholds from .env)
│   └── options_safety.py     7 entry safety gates
├── agents/
│   ├── options_research.py   Signal generation via Claude + affordability filter
│   ├── options_execution.py  Limit order construction + dynamic qty sizing
│   └── options_exit.py       Momentum Snipe 7-condition exit engine
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
    └── options_research.py   Options research agent prompt (v2 — Momentum Snipe)
```

---

## Safety Gates

Every options entry passes through 7 sequential safety checks. The pipeline stops at the first failure:

| # | Gate | What it checks |
|---|---|---|
| 1 | `check_options_enabled` | Master switch — `OPTIONS_ENABLED` must be `true` |
| 2 | `check_iv_rank` | IV rank must be below `OPTION_MAX_IV_RANK` (default 0.60) |
| 3 | `check_dte` | Days to expiry within `[OPTION_MIN_DTE, OPTION_MAX_DTE]` range |
| 4 | `check_spread` | Bid-ask spread < `OPTION_MAX_SPREAD_PCT` of mid price |
| 5 | `check_delta` | Absolute delta within `[OPTION_MIN_DELTA, OPTION_MAX_DELTA]` |
| 6 | `check_exposure` | Total options exposure < `OPTION_MAX_TOTAL_EXPOSURE` of NAV |
| 7 | `check_max_loss` | Computed max loss below `OPTION_MAX_LOSS_PCT` threshold |

Each gate returns a typed `(passed, reason)` result. All results are logged to the audit trail.

---

## Exit Engine

The exit engine (`options_exit.py`) evaluates each open position every loop iteration using 7 priority-ordered conditions. The first triggered condition wins:

| Priority | Condition | Exit Type | Action |
|---|---|---|---|
| 1 | Loss ≥ 25% from entry | `hard_stop` | FULL_CLOSE |
| 2 | Regime turned hostile | `regime_stop` | FULL_CLOSE |
| 3 | DTE ≤ 5 days | `dte_stop` | FULL_CLOSE |
| 4 | Held > 5 days with < 10% gain | `time_stop` | FULL_CLOSE |
| 5 | Trailing stop hit (after activation) | `trailing_stop` | FULL_CLOSE |
| 6 | Gain ≥ 60% | `stretch_target` | FULL_CLOSE |
| 7 | Gain ≥ 35% (qty ≥ 2) | `profit_target` | PARTIAL_CLOSE (sell half) |
| 7 | Gain ≥ 35% (qty = 1) | `profit_target` | Activate trailing stop |

If no condition triggers, the decision is `HOLD`. On HOLD, the engine still updates the high-water mark and checks whether the trailing stop should activate (gain ≥ 20%).

Exit decisions are one of: `HOLD`, `PARTIAL_CLOSE` (profit target), or `FULL_CLOSE` (any stop or stretch target).

---

## Trailing Stop Mechanics

- **Activation:** Trailing stop activates when unrealized gain reaches +20% (`trail_activation_pct`)
- **Trail distance:** 12% below the high-water mark (`trail_pct`)
- **Ratcheting:** High-water mark updates on every HOLD when mid price exceeds the previous high
- **One-way:** Trailing stop price only moves up, never down
- **Trigger:** Position closed when mid price falls below the trailing stop price

---

## Persistence

Options positions are persisted to SQLite via `OptionsPositionRow` in `adapters/db.py`:

- **On entry:** Position saved with `high_water_price` (= entry price), `trailing_active=False`
- **On partial exit:** `remaining_qty` updated, `trailing_active` may be set to `True`
- **On HOLD:** `high_water_price`, `trailing_active`, `trailing_stop_price` updated if changed
- **On full exit:** Position marked `status="closed"` with realized PnL, `exit_type`, and timestamp

This allows the exit engine to track positions across loop iterations and container restarts.

---

## Configuration

All options settings live in `.env` and are loaded via `OptionsSettings` (Pydantic v2):

| Variable | Default | Description |
|---|---|---|
| `OPTIONS_ENABLED` | `false` | Master switch for entire options module |
| `OPTIONS_UNIVERSE` | `AAPL,AMD,PLTR,SOFI,COIN,MARA` | Eligible tickers for options |
| `OPTION_PROFIT_TARGET_PCT` | `0.35` | Profit target — partial close at +35% |
| `OPTION_STRETCH_TARGET_PCT` | `0.60` | Stretch target — full close at +60% |
| `OPTION_TRAIL_ACTIVATION_PCT` | `0.20` | Trailing stop activates at +20% gain |
| `OPTION_TRAIL_PCT` | `0.12` | Trail distance: 12% below high-water |
| `OPTION_MAX_LOSS_PCT` | `0.25` | Hard stop: max loss per position (25%) |
| `OPTION_TIME_STOP_DAYS` | `5` | Time stop: close if held > N days with weak gain |
| `OPTION_DTE_EXIT_DAYS` | `5` | DTE stop: close if ≤ N days to expiry |
| `OPTION_MAX_CONTRACT_COST` | `500.0` | Max cost per contract (affordability filter) |
| `OPTION_MAX_CONTRACTS` | `5` | Max contracts per position |
| `OPTION_MAX_DTE` | `35` | Max days to expiry at entry |
| `OPTION_MIN_DTE` | `14` | Min days to expiry at entry |
| `OPTION_MAX_IV_RANK` | `0.60` | Max IV rank (above = IV too expensive) |
| `OPTION_MIN_DELTA` | `0.30` | Min absolute delta |
| `OPTION_MAX_DELTA` | `0.60` | Max absolute delta |
| `OPTION_MAX_POSITION_PCT` | `0.10` | Max NAV fraction per position (10%) |
| `OPTION_MAX_TOTAL_EXPOSURE` | `0.20` | Max NAV across all options (20%) |
| `OPTION_MAX_SPREAD_PCT` | `0.05` | Max bid-ask spread as pct of mid |

---

## Backtesting

The options backtest engine (`backtest/options_engine.py`) provides bar-replay simulation:

- **Initial capital:** $5,000 (matches sub-$5K account target)
- **Pricing model:** Delta approximation + theta decay (no full Black-Scholes needed)
- **Entry signals:** RSI-based (RSI < 35 → long call, RSI > 65 → long put)
- **Exit simulation:** Full Momentum Snipe logic — 5 exit conditions (hard stop, trailing stop, profit/stretch targets, expiry), affordability filter, dynamic qty sizing
- **Metrics:** Win rate, Sharpe ratio, max drawdown, profit factor, equity curve

```python
from sauce.backtest.options_engine import run_options_backtest
from sauce.backtest.options_models import OptionsBacktestConfig

result = run_options_backtest("AAPL", df, OptionsBacktestConfig())
print(f"Win rate: {result.win_rate:.1%}, Sharpe: {result.sharpe_ratio:.2f}")
```

---

## Learning & Analytics

Post-trade analytics (`memory/options_learning.py`) tracks:

- **Trade outcomes:** PnL, hold duration, exit reason per position
- **Win rate drift:** Detects when recent win rate drops below threshold
- **Exit type performance:** Groups closed trades by `exit_type` (profit_target, stretch_target, trailing_stop, hard_stop, time_stop, dte_stop, regime_stop)
- **Underlying performance:** Aggregates results by ticker symbol
