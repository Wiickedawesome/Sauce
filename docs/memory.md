# Memory & RAG

Sauce's memory system (`sauce/memory/`) provides two layers of persistent context that make Claude an experienced trader rather than an amnesiac analyst.

---

## Memory Layers

### Layer 1 — Session Memory (Working Memory)

Resets at the start of each trading day. Stores intraday context:

- Market regime at each cycle throughout the day
- Every setup detected today (approved or rejected, and why)
- Every trade placed today with current status
- Symbol activity patterns (hot vs noisy)
- Running intraday narrative

**Purpose:** When BTC triggers a mean reversion signal at 2:30 PM, Claude can see that the same setup triggered at 10:00 AM and 12:00 PM and both failed. That context changes the confidence assessment.

### Layer 2 — Strategic Memory (Persistent)

Never resets. Accumulated experience across the system's lifetime:

| Table | What It Stores |
|---|---|
| `SetupPerformance` | Win/loss, P&L, hold duration by setup type, symbol, regime, time bucket |
| `SymbolLearnedBehavior` | Optimal entry RSI, avg reversion depth per symbol |
| `ClaudeCalibration` | How well Claude's confidence predictions match outcomes |
| `VetoPattern` | Patterns that lead to repeated supervisor vetoes |
| `WeeklyPerformance` | Weekly Sharpe, win rate, max drawdown |

---

## RAG Retrieval

Before each research cycle, the system retrieves the most relevant past trades for the current signal context:

```python
from sauce.memory.db import get_similar_trades

# Retrieve top-5 similar past trades
trades = get_similar_trades(
    db_path="data/sauce.db",
    symbol="BTC/USD",
    regime="RANGING",
    setup_type="crypto_mean_reversion",
    top_k=5,
)
```

### Scoring Heuristic

Candidates are scored by relevance:

| Match | Score |
|---|---|
| Same symbol | +3 |
| Same regime | +2 |
| Same setup type | +1 |

Ties are broken by recency (most recent first). Up to 200 candidates are evaluated per query.

### Prompt Injection

Retrieved trades are formatted into plain English and injected into the Research Agent's Claude prompt as the `past_trade_memory` field:

```
SIMILAR PAST TRADES (5 most relevant):
Record: 3W / 2L.
  2025-06-01 BTC/USD crypto_mean_reversion in RANGING (09:30-12:00): WIN +150.00 held 45min.
  2025-05-28 BTC/USD crypto_mean_reversion in VOLATILE (12:00-14:00): LOSS -80.00 held 30min.
  ...
Avg P&L: +42.00.
```

Claude is instructed to use these outcomes for confidence calibration — if similar trades have a poor track record, it should be more skeptical.

---

## Write Path

After each closed trade, `sauce/memory/learning.py` records:
- Trade outcome (win/loss, P&L, hold duration)
- Setup type, symbol, regime at entry, time of day
- Claude's confidence prediction vs actual outcome

This feeds the strategic memory tables, closing the learning loop.
