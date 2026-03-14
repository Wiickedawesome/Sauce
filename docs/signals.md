# Multi-Timeframe Signals

The signal engine (`sauce/signals/`) fetches OHLCV data across multiple timeframes, computes indicators for each, and produces a weighted confluence score with signal tier classification.

---

## Timeframes

| Label | Alpaca String | Bars | Coverage |
|---|---|---|---|
| 5m | `5Min` | 120 | ~10 hours |
| 15m | `15Min` | 80 | ~20 hours |
| 1h | `1Hour` | 48 | ~2 days |
| 4h | `4Hour` | 30 | ~5 days |
| 1d | `1Day` | 50 | ~50 trading days |

---

## Confluence Scoring

Each timeframe casts a directional vote (bullish / bearish / neutral) based on its indicators. Votes are weighted:

| Timeframe | Weight |
|---|---|
| 5m | 0.10 |
| 15m | 0.15 |
| 1h | 0.25 |
| 4h | 0.25 |
| 1d | 0.25 |

Higher timeframes carry more weight — the 1h, 4h, and 1d together account for 75% of the total score.

The weighted votes produce a confluence score from -1.0 (fully bearish) to +1.0 (fully bullish).

---

## Signal Tiers

Based on the confluence score, each signal is assigned a tier that integrates with the research agent's confidence adjustment:

| Tier | Meaning | Confidence Adjustment |
|---|---|---|
| **S1** | Strong alignment across ≥3 timeframes | +0.10 |
| **S2** | Moderate alignment (2 timeframes agree) | +0.05 |
| **S3** | Mixed or insufficient data | 0 |
| **S4** | Conflicting signals across timeframes | -0.10 |

---

## Usage

```python
from sauce.signals.timeframes import fetch_multi_timeframe
from sauce.signals.confluence import compute_confluence

# Fetch indicators across all timeframes
mtf_context = fetch_multi_timeframe("BTC/USD", is_crypto=True)

# Compute confluence score and tier
result = compute_confluence(mtf_context)
# result.score  → float (-1.0 to +1.0)
# result.tier   → SignalTier.S1 / S2 / S3 / S4
# result.votes  → per-timeframe vote breakdown
```

---

## Integration

The multi-timeframe context is computed in the Research Agent (Step 2) and:
1. Confluence score adjusts research confidence before Claude audits the thesis
2. Signal tier is included in the supervisor prompt for final review
3. Per-timeframe indicators provide Claude with cross-timeframe context
