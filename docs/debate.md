# Bull/Bear Debate Layer

The debate engine (`sauce/agents/debate.py`) adds adversarial pressure to every trade signal before it reaches the Supervisor. Both sides receive the same signal and indicators and produce structured arguments.

---

## Design

The debate layer is **fully deterministic** — no LLM calls. This is intentional: it adds zero latency and zero cost while combating single-model confirmation bias. The debate transcript is forwarded to the Supervisor (Claude) who uses it in the final veto decision.

---

## How It Works

1. **Bull case** — argues FOR the trade using supporting indicator values (trend alignment, momentum, volume confirmation)
2. **Bear case** — argues AGAINST using contradicting signals (divergences, overextension, volume weakness)
3. **Verdict** — classifies the debate outcome as `bull_wins`, `bear_wins`, or `contested`
4. **Confidence adjustment** — modifies the signal's confidence score:

| Verdict | Adjustment |
|---|---|
| `bull_wins` (buy signal) | +0.05 |
| `bear_wins` (buy signal) | -0.10 |
| `contested` | -0.03 |
| `bull_wins` (sell signal) | -0.10 |
| `bear_wins` (sell signal) | +0.05 |

For sell signals, the bull/bear semantics are swapped — a "bear wins" verdict supports the sell thesis.

---

## Arguments Structure

Each side produces a list of `Argument` objects:

```python
@dataclass(frozen=True, slots=True)
class Argument:
    label: str      # e.g. "RSI oversold", "MACD bearish cross"
    detail: str     # Human-readable explanation
    weight: float   # 0.0–1.0 importance
```

Arguments are scored by summing weights. The side with the higher total wins.

---

## Integration

The debate runs in the main loop (Step 5b) after risk approval:

```
Research → Risk → Debate → Supervisor
```

- `supervisor.py` receives `debate_results` parameter
- `prompts/supervisor.py` formats debate summaries into the Supervisor's Claude prompt
- The Supervisor sees bull arguments, bear arguments, and the verdict when making the final execute/abort decision
