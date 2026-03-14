# Backtesting Engine

The backtesting engine (`sauce/backtest/`) provides vectorized bar-replay simulation of Sauce's trading strategies on historical data.

---

## Overview

The engine walks a historical OHLCV DataFrame bar-by-bar, running the indicator library and setup scanner at each step using only data visible up to that bar (no lookahead). It simulates entry/exit logic with ATR-based stop-loss and profit targets.

No LLM calls are made during backtesting — the engine assumes Claude always approves passing setups with the deterministic setup score as the confidence value.

---

## Configuration

```python
from sauce.backtest.models import BacktestConfig

config = BacktestConfig(
    initial_capital=100_000.0,   # Starting capital
    position_pct=0.05,           # Fraction of capital per trade
    atr_stop_mult=2.0,           # ATR multiplier for stop-loss
    atr_target_mult=3.0,         # ATR multiplier for take-profit
    max_hold_bars=20,            # Max bars before forced exit
    min_bars_warmup=50,          # Bars needed before first signal
)
```

---

## Usage

```python
from sauce.backtest import run_backtest, BacktestConfig

result = run_backtest(
    symbol="BTC/USD",
    df=ohlcv_dataframe,        # pandas DataFrame with OHLCV columns
    regime="RANGING",           # Market regime to simulate
    config=BacktestConfig(),
)

# Result fields:
#   result.total_trades    → int
#   result.wins / losses   → int
#   result.win_rate        → float
#   result.total_pnl       → float
#   result.sharpe_ratio    → float
#   result.max_drawdown    → float
#   result.trades          → list[BacktestTrade]
```

---

## Exit Reasons

Each trade records why it was closed:

| Exit Reason | Description |
|---|---|
| `STOP_LOSS` | Price hit ATR-based stop |
| `TAKE_PROFIT` | Price hit ATR-based target |
| `MAX_HOLD` | Held for max allowed bars |
| `END_OF_DATA` | Data ended with position open |

---

## Supported Regimes

The engine validates regime against: `TRENDING_UP`, `TRENDING_DOWN`, `RANGING`, `VOLATILE`, `DEAD`.

---

## Limitations

- No slippage modeling — fills assumed at bar close
- No commission/fee deduction
- Single position per symbol at a time
- Deterministic only — no Claude interaction
