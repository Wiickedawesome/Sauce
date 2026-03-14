# Safety & Risk

Sauce's 8-layer safety gauntlet is the system's core differentiator. No trade executes without passing every layer.

---

## The 8 Layers

| Layer | Type | Description |
|---|---|---|
| 1. Data Freshness | Code | Market data must be < `DATA_TTL_SECONDS` old. Stale data = no trade. |
| 2. Market Hours | Code | Economic calendar integration. Suppresses trading near FOMC, CPI, and other macro events. |
| 3. Regime Gate | Code | Each setup type only fires in eligible regimes (e.g., equity pullback requires `TRENDING_UP`). |
| 4. Setup Scanner | Code | Hard/soft scoring model with minimum thresholds. No setup pass = no Claude call (cost savings). |
| 5. Research Audit | Claude | Claude reviews the scored thesis as an auditor, looking for contradictions. Returns confidence 0–1. |
| 6. Risk Check | Code | Position limits, portfolio exposure, daily loss limit, buying power depletion tracking. |
| 7. Debate | Code | Deterministic bull/bear debate adjusts confidence. Counter-trend signals penalized. |
| 8. Supervisor Veto | Claude | Final gate. Can only downgrade (abort), NEVER upgrade. Hard pre-flight checks before Claude even sees the orders. |

---

## Supervisor Invariant

The Supervisor has a hardcoded invariant that cannot be bypassed by Claude:

```python
# Can NEVER upgrade an abort to execute
if claude_action != "execute":
    return _abort(claude_reason)
```

The Supervisor can only abort. It receives the debate transcript, portfolio review, and all signals — but it cannot modify orders or override risk rejections.

---

## Risk Parameters

| Parameter | Default | Description |
|---|---|---|
| `MAX_POSITION_PCT` | 5% | Max allocation per symbol as fraction of NAV |
| `MAX_PORTFOLIO_EXPOSURE` | 1.0 | Max total exposure (1.0 = no leverage) |
| `MAX_DAILY_LOSS_PCT` | 2% | Auto-pause if daily drawdown exceeds this |
| `MIN_CONFIDENCE` | 0.5 | Signals below this are treated as hold |
| `MAX_PRICE_DEVIATION` | 1% | Execution vetoes if live quote deviates more |
| `TRADING_PAUSE` | false | Emergency kill switch — halts all trading |

---

## Anomaly Detection

The Ops agent monitors:
- 100% veto rate (all signals rejected)
- Zero signals generated
- Consecutive loss streaks
- Setup-level circuit breakers

---

## Prompt Security

- `sanitize_llm_text()` strips injection attempts from all LLM inputs
- Sentinel markers bound Claude's output parsing
- Claude's output is always validated against Pydantic schemas before any action is taken
