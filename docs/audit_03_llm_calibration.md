# Audit 03 — LLM Calibration & Claude Reasoning Quality

**Version:** 1.0  
**Date:** 2026-03-14  
**Auditor:** Automated (Copilot)  
**System:** Sauce — Autonomous Multi-Agent Trading System  
**Overall Verdict:** INSUFFICIENT DATA  
**Core Question:** *Is Claude's stated confidence a meaningful probability estimate, or is it decorative?*

---

## Executive Summary

The calibration table (`claude_calibration` in `data/strategic_memory.db`) contains **0 rows**. No LLM calls (`llm_call` / `llm_response` events) exist in any of the three audit databases. Zero trades have been executed end-to-end through the production pipeline, so no outcome data exists to evaluate calibration quality.

Per the anti-hallucination protocol: **"If the calibration table has fewer than 30 rows, mark this audit as INSUFFICIENT DATA."** This audit therefore cannot produce statistical findings on calibration accuracy. Instead, it provides a **code-level readiness assessment** — verifying that the calibration pipeline, prompt architecture, RAG retrieval, and confidence-gated gating logic are correctly wired and will produce meaningful data once the system begins live trading.

---

## Pre-Audit Evidence Summary

| Data Source | Evidence |
|---|---|
| `data/strategic_memory.db` → `claude_calibration` | **0 rows** |
| `data/strategic_memory.db` → `setup_performance` | **0 rows** |
| `data/strategic_memory.db` → `validation_results` | **1 row** (2026-03-12, all zeroes, `all_passed=False`) |
| `data/sauce.db` → `llm_call` / `llm_response` events | **0** |
| `data/sauce.db.bak.20260307_215539` → LLM events | **0** (24 signals, all stubs) |
| `data/sauce.db.bak.20260313_190553` → LLM events | **0** (275 signals: 238 stubs, 37 safe-hold failures) |
| Signals with `confidence > 0` (newest backup) | **21** — all `side=buy, conf=0.75, reasoning=stub, symbol=None` (test artifacts) |
| Signals with `prompt_version` set | **0** across all DBs |
| `data/session_memory.db` → `signal_log` | **0 rows** |

**Anti-hallucination check:** Every claim above is derived from direct SQL queries against the production and backup databases. No inferences.

---

## Section 1 — Calibration Table Analysis

### 1.1 Schema Review

**Verdict: PASS (schema)**

`ClaudeCalibrationRow` in [sauce/memory/db.py](sauce/memory/db.py#L233-L240):

| Column | Type | Purpose |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `date` | String(10), indexed | Trade date (YYYY-MM-DD) |
| `confidence_stated` | Float, NOT NULL | Claude's output confidence [0.0, 1.0] |
| `outcome` | String(8), NOT NULL | `"win"` or `"loss"` |
| `setup_type` | String(40), NOT NULL | Which setup type was active |

`ClaudeCalibrationEntry` Pydantic model in [sauce/core/schemas.py](sauce/core/schemas.py#L478-L483):
- `confidence_stated: float = Field(ge=0.0, le=1.0)` — bounded
- `outcome: Literal["win", "loss"]` — enum-constrained
- `setup_type: SetupType` — typed to valid setup types

**Assessment:** Schema is well-designed. Columns capture the minimum necessary for calibration analysis: stated confidence, actual outcome, and the setup type that generated the signal. The `date` index enables time-windowed queries.

### 1.2 Data Population Pipeline

**Verdict: WARN — calibration entries are never written in production**

The calibration write path is:

1. `record_trade_outcome(entry, db_path, calibration_entry=None)` in [sauce/memory/learning.py](sauce/memory/learning.py#L36-L44)
2. If `calibration_entry is not None`, calls `write_claude_calibration(entry, db_path)`

**Problem:** The sole production call site in [sauce/core/loop.py](sauce/core/loop.py#L125):
```python
record_trade_outcome(entry, db_path)
```
...passes **only** `entry` and `db_path`. The `calibration_entry` parameter is **never supplied**. This means even when a trade closes with a known P&L outcome, the system never constructs a `ClaudeCalibrationEntry` linking the original Claude confidence to the win/loss result.

The pipeline to construct the `ClaudeCalibrationEntry` (associating the signal's `confidence` at entry time with the closed position's `win/loss` at exit time) does not exist in the production code path.

**Finding F-01:** `record_trade_outcome()` in `loop.py` never passes `calibration_entry`. The calibration table will remain permanently empty even after live trading begins.

### 1.3 Calibration Analyzer

**Verdict: PASS (code quality)**

`analyze_claude_calibration()` in [sauce/memory/learning.py](sauce/memory/learning.py#L90-L133):
- Buckets: `0.50–0.60`, `0.60–0.70`, `0.70–0.80`, `0.80–0.90`, `0.90–1.00`
- Computes `actual_win_rate` vs `expected_midpoint` per bucket
- Returns `{"buckets": {...}, "total_entries": N}`
- Handles empty table gracefully (returns `{buckets: {}, total_entries: 0}`)
- Called by `run_learning_cycle()` on weekly trigger

**Assessment:** The analyzer is correctly implemented. The bucket boundaries are sensible. The only issue is that it will never receive data due to F-01.

### 1.4 Validation Integration

**Verdict: PASS (code quality)**

`check_claude_calibration()` in [sauce/core/validation.py](sauce/core/validation.py#L206-L231):
- Score formula: `1 - mean(|confidence_stated - actual_outcome|)` where `win→1.0, loss→0.0`
- Default threshold: `min_score=0.60`
- Used in the 30-day validation gate — system cannot graduate from paper to live without calibration score ≥ 0.60
- Returns `(False, 0.0)` when table is empty (prevents false pass)

**Assessment:** The scoring formula is a proper Brier-like score. The empty-table guard prevents graduating without calibration data.

### 1.5 Context Injection

**Verdict: PASS (code quality)**

`build_strategic_paragraph()` in [sauce/prompts/context.py](sauce/prompts/context.py#L202-L222):
- When `claude_calibration` list is non-empty, injects average stated confidence, actual win rate, and an overconfidence/underconfidence warning (threshold: |avg_conf - actual_rate| > 0.15) into Claude's prompt
- This creates a feedback loop: Claude sees its own historical accuracy

**Assessment:** Well-designed self-correcting mechanism. Will function correctly once data exists.

---

## Section 2 — Reasoning Quality Audit

### 2.1 Signal Reasoning in Production

**Verdict: CANNOT ASSESS (no LLM signals exist)**

All 299 signals across both backup databases are either stubs or safe-hold failures:

| Category | Count | Source |
|---|---|---|
| `reasoning="stub"` | 256 | Test harness artifacts |
| `reasoning="Hold (safe default): market data unavailable..."` | 29 | Alpaca 401 errors |
| `reasoning="Hold (safe default): No qualifying setup..."` | 7 | Setup gate (correct behavior) |
| `reasoning=NULL` | 7 | Missing field |

Zero signals originated from an actual Claude LLM call. No reasoning text can be assessed for quality, specificity, or hallucination.

### 2.2 Bear Case Field

**Verdict: CANNOT ASSESS**

The `bear_case` field was added in the v2 prompt architecture (see [sauce/prompts/research.py](sauce/prompts/research.py#L271-L277)). The prompt instructs Claude:

> *"REQUIRED — 1 to 2 sentences. The strongest argument AGAINST this trade. Play devil's advocate: what could go wrong?"*

No signal in any database contains a populated `bear_case` field. All are `None`.

### 2.3 Prompt Architecture Quality (Code Review)

**Verdict: PASS**

The v2 `build_user_prompt()` in [sauce/prompts/research.py](sauce/prompts/research.py#L72-L340) is well-structured:

- **Grounding:** All indicators are explicitly provided — Claude is told "Base ONLY on the indicators provided above. Do not reference any external data."
- **Calibration guidance:** Explicit confidence ranges: `0.55–0.80` for approved setups, with per-scenario guidance (e.g., "conflicting timeframes → reduce by 0.10–0.15")
- **Anti-inflation:** "Values below 0.40 are treated as hold by the system. Hold is a valid and expected outcome — do not inflate confidence to cross the threshold."
- **Required output schema:** `side`, `confidence`, `reasoning`, `bear_case` — all REQUIRED
- **Asset-type awareness:** Separate `volume_ratio` interpretation for crypto vs equity
- **Multi-timeframe context injection:** Confluence scoring from 5m to 1D timeframes

**Assessment:** The prompt design follows calibration best practices. Explicit ranges, anti-inflation language, and grounding constraints reduce the risk of decorative confidence scores. This can only be validated with live data.

---

## Section 3 — Prompt Versioning Review

### 3.1 Version Tracking

**Verdict: PASS**

- `PROMPT_VERSION = "v2"` is defined in [sauce/prompts/research.py](sauce/prompts/research.py#L20)
- `prompt_version` is propagated through: `Signal`, `RiskCheckResult`, `Order`, `SupervisorDecision`, `AuditEvent`, `PortfolioReview`, `ExitSignal`
- Every `log_event()` call includes `prompt_version=settings.prompt_version`
- The `build_user_prompt()` function includes `prompt_version` in the JSON payload sent to Claude

### 3.2 Version History

| Version | System Prompt Role | Status |
|---|---|---|
| v1 | "You are a quantitative trading analyst" — raw analyst generating ideas | Preserved in `_SYSTEM_PROMPT_V1` |
| v2 | "You are an auditor inside a live algorithmic trading system. You do NOT generate trade ideas." — auditor reviewing pre-scored theses | **Active** |

### 3.3 Prompt Version in Signal Data

**Verdict: WARN**

All 275 signals in the newest backup have `prompt_version=None`. This is because:
1. The 238 stubs and 37 safe-hold signals come from loop runs where the LLM was never called
2. Although `_safe_hold()` now sets `prompt_version` (per Audit 02 fix), these signals predate that fix

**Assessment:** Post-fix, prompt version will be recorded on all signals. No action needed.

---

## Section 4 — RAG Retrieval Effectiveness

### 4.1 Memory Context Pipeline

**Verdict: PASS (architecture)**

Three RAG sources feed into Claude's prompt:

1. **Session memory** ([sauce/agents/research.py](sauce/agents/research.py#L192-L200)): Regime history, signals today, trades today, symbol characters → `build_session_paragraph()`
2. **Strategic memory** ([sauce/agents/research.py](sauce/agents/research.py#L202-L209)): Setup performance, weekly trends, learned symbol behavior, calibration data → `build_strategic_paragraph()`
3. **Similar past trades** ([sauce/agents/research.py](sauce/agents/research.py#L212-L226)): `get_similar_trades()` retrieves matching trades by symbol + regime → `build_similar_trades_paragraph()`

All three are wrapped in try/except blocks with graceful degradation — Claude operates without memory if retrieval fails.

### 4.2 Calibration Feedback Loop (RAG → Claude)

The strategic memory paragraph includes the calibration self-correction logic:

```
Claude calibration (N recent trades): stated avg confidence X%, actual win rate Y%.
NOTE: Claude is overconfident — stated confidence exceeds actual outcomes.
```

This means Claude sees its own historical accuracy and is prompted to adjust. The threshold for the overconfidence/underconfidence warning is a 15-point gap (|avg_conf - actual_rate| > 0.15).

### 4.3 Data Availability for RAG

**Verdict: INSUFFICIENT DATA**

| RAG Source | Available Data |
|---|---|
| Session memory | 1 regime log entry, 1 narrative entry, 0 signals, 0 trades |
| Strategic memory | 0 setup_performance, 0 weekly_performance, 0 symbol_behavior, 0 calibration |
| Similar trades | 0 (no trades to retrieve) |

No RAG context can be evaluated for retrieval quality because the source tables are empty.

---

## Section 5 — Confidence-Gated Action Review

### 5.1 Confidence Threshold Configuration

**Verdict: PASS**

- `.env`: `MIN_CONFIDENCE=0.5`
- Research prompt instructs Claude: "Values below 0.40 are treated as hold by the system"
- Risk agent checks `confidence_ok` as part of `RiskChecks`

### 5.2 Confidence Adjustment Pipeline

**Verdict: PASS (code quality)**

The research agent applies multi-timeframe confluence adjustment:
```python
confidence = max(0.0, min(1.0, confidence + confluence.confidence_adjustment))
```
This modifies Claude's raw confidence based on cross-timeframe agreement before the signal is emitted. The clamping to [0.0, 1.0] prevents out-of-range values.

### 5.3 Calibration-Gated Validation

**Verdict: PASS**

The 30-day validation gate (`check_claude_calibration()`) requires `calibration_score ≥ 0.60` before transitioning from paper to live. With 0 calibration entries, the function returns `(False, 0.0)` — correctly blocking graduation.

### 5.4 Gap: Calibration Entry Not Created on Position Close

**Same as F-01.** When a position closes in the main loop, the `SetupPerformanceEntry` is persisted but no `ClaudeCalibrationEntry` is constructed. The signal's original `confidence` value is not carried forward to the trade outcome record.

---

## Findings Register

| ID | Section | Severity | Title | Status |
|---|---|---|---|---|
| F-01 | 1.2 | FAIL | `record_trade_outcome()` never receives `calibration_entry` — calibration table permanently empty | OPEN |

### F-01 Detail

**What:** The sole production call to `record_trade_outcome()` in [sauce/core/loop.py](sauce/core/loop.py#L125) does not pass a `calibration_entry`. The function signature accepts it as an optional parameter, and `write_claude_calibration()` is correctly wired, but the caller never constructs a `ClaudeCalibrationEntry`.

**Why this matters:** Without calibration data, the system cannot:
1. Evaluate whether Claude's confidence scores are meaningful probabilities
2. Self-correct via the RAG feedback loop (strategic memory calibration paragraph)
3. Pass the validation gate (`check_claude_calibration()` returns `(False, 0.0)`)
4. Graduate from paper to live trading

**Root cause:** When a position closes, the loop has access to the P&L outcome and setup type, but it does not retain the original signal's `confidence` value. The confidence is logged in the audit event payload but not extracted for calibration.

**Fix approach:** When recording a closed position outcome, resolve the original signal's confidence from the signal log (session memory) or audit events, construct a `ClaudeCalibrationEntry`, and pass it to `record_trade_outcome()`.

---

## Improvement Recommendations

| ID | Title | Priority | Description |
|---|---|---|---|
| IMP-29 | Wire calibration entry on position close | HIGH | Resolve the original signal confidence (from `signal_log` or audit events) when a position closes. Construct `ClaudeCalibrationEntry(date, confidence_stated, outcome, setup_type)` and pass it to `record_trade_outcome()`. Without this, the calibration table will remain empty permanently. |
| IMP-30 | Prompt version in signal log | LOW | Ensure `prompt_version` is stored in `signal_log` entries so calibration analysis can be segmented by prompt version. Currently the `SignalLogRow` schema does not include a `prompt_version` column. |
| IMP-31 | Calibration minimum sample guard | LOW | `analyze_claude_calibration()` produces per-bucket statistics even with 1 entry. Add a minimum sample size (e.g., 5 per bucket) before reporting win rates to avoid misleading statistics from small samples. |

---

## Cross-Audit Dependencies

| Dependency | Status | Impact on Audit 03 |
|---|---|---|
| Audit 01 (Safety) | COMPLETE — all IMPs implemented | No impact |
| Audit 02 (Signal Quality) | COMPLETE — F-01 through F-04 resolved | F-05 (no production data) directly causes INSUFFICIENT DATA here |
| Audit 02 IMP-27 (validation run) | OPEN | A production run would generate the LLM calls needed for this audit |

---

## Sign-Off

| Field | Value |
|---|---|
| Databases examined | `data/strategic_memory.db`, `data/session_memory.db`, `data/sauce.db`, 2 backups |
| Source files reviewed | `sauce/memory/learning.py`, `sauce/memory/db.py`, `sauce/prompts/research.py`, `sauce/prompts/context.py`, `sauce/agents/research.py`, `sauce/adapters/llm.py`, `sauce/core/validation.py`, `sauce/core/schemas.py`, `sauce/core/loop.py`, `.env` |
| Total `claude_calibration` rows | 0 |
| Total LLM call/response events | 0 across all databases |
| Minimum required for full audit | 30 calibration rows |
| Calibration code quality | PASS — schema, analyzer, validation gate, RAG feedback all correctly wired |
| Blocking finding count | 1 (F-01: calibration entry never created) |
| Overall verdict | **INSUFFICIENT DATA** — code architecture is sound; 0 production data prevents statistical assessment |

---

*Re-audit trigger: Run this audit after ≥30 trades have been executed through the full pipeline (LLM call → signal → order → position close → calibration entry). F-01 must be fixed first.*
