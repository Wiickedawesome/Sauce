# Indicators

The indicator library (`sauce/indicators/core.py`) provides pure-function technical indicator computations built on top of [pandas-ta](https://github.com/twopirllc/pandas-ta). All functions accept a pandas DataFrame with standard OHLCV columns and return typed results via the `Indicators` Pydantic schema.

---

## Available Indicators

| Indicator | Field(s) | Description |
|---|---|---|
| SMA 20 | `sma_20` | Simple Moving Average, 20 periods |
| SMA 50 | `sma_50` | Simple Moving Average, 50 periods |
| RSI 14 | `rsi_14` | Relative Strength Index |
| ATR 14 | `atr_14` | Average True Range (volatility) |
| MACD | `macd_line`, `macd_signal`, `macd_histogram` | Moving Average Convergence Divergence |
| Bollinger Bands | `bb_upper`, `bb_middle`, `bb_lower` | 20-period, 2 std dev |
| Stochastic | `stoch_k`, `stoch_d` | Stochastic oscillator K and D |
| VWAP | `vwap` | Volume Weighted Average Price |
| Volume Ratio | `volume_ratio` | Current volume vs 20-period average |

---

## Usage

```python
from sauce.indicators.core import compute_all

# df = pandas DataFrame with: open, high, low, close, volume columns
indicators = compute_all(df, is_crypto=False)
# Returns: Indicators schema or None if data insufficient
```

The `compute_all()` function is the primary entry point. It computes all indicators in a single pass and returns the `Indicators` Pydantic model. Individual indicator functions are also available for standalone use.

The `is_crypto` flag controls VWAP computation — crypto uses 24h sessions vs equity regular hours.

---

## Integration Points

- **Research Agent** — indicators are computed per symbol and packaged into the Claude prompt as structured JSON
- **Multi-Timeframe Engine** — `compute_all()` is called per timeframe (5m, 15m, 1h, 4h, 1d)
- **Debate Layer** — bull/bear arguments reference indicator values directly
- **Setup Scanner** — hard/soft scoring thresholds reference specific indicator values (e.g., RSI < 38 for crypto mean reversion)
