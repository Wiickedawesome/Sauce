# Audit 02 — Signal Quality & Research Agent Verification

**Audit Date:** 2026-03-15  
**Auditor:** AI Audit Agent (Claude Opus 4.6)  
**Scope:** Signal pipeline quality, setup scanner, multi-TF confluence, LLM calibration, regime alignment  
**Databases Examined:** 3 (production + 2 backups)  
**Status:** COMPLETE

---

## Executive Summary

**Overall Verdict: CANNOT ASSESS — NO GENUINE LLM-GENERATED SIGNAL DATA**

Across all three available databases, the Sauce system has **never completed a full signal pipeline execution through Claude**. Zero LLM calls exist in any database. The setup scanner gate has correctly prevented LLM invocation because no qualifying setup was found, but this means the downstream pipeline (multi-TF confluence, Claude reasoning, confidence calibration) has never been exercised in production.

Of the 275 signals in the most data-rich backup database:
- **238** are test-contaminated (sub-second loop duration, `reasoning="stub"`)
- **30** are error fallbacks from Alpaca 401 authentication failures
- **7** are genuine production holds from a single loop that correctly gated on regime

Code review of the full signal pipeline confirms the architecture is **well-designed and defensively coded** — every error path returns a safe hold, the setup scanner applies hard/soft scoring correctly, and the confluence engine has proper tier logic. The code is ready to produce real signals; it simply hasn't had the conditions to do so.

Three code-level bugs were discovered during the audit that affect observability and diagnostics.

---

## Pre-Audit Checklist

| Check | Status | Evidence |
|-------|--------|----------|
| DB accessible | PASS | `data/sauce.db` (prod), 2 backups accessible |
| Signal pipeline code present | PASS | `research.py` (452 lines), `setups.py` (903 lines), `confluence.py` (163 lines), `timeframes.py` (180 lines) |
| Schema intact | PASS | `Signal` model with `side`, `confidence`, `reasoning`, `evidence`, `bear_case` |
| Prior audit complete | PASS | Audit 01 completed; all IMPs (01–08, 10) implemented; 873 tests passing |

---

## Database Inventory

| Database | Total Events | Signals | Orders | LLM Calls | Date Range |
|----------|-------------|---------|--------|-----------|------------|
| `data/sauce.db` (production) | 20 | 0 | 0 | 0 | Mar 14–15 |
| `data/sauce.db.bak.20260307_215539` | 290 | 24 | 0 | 0 | Mar 7–8 |
| `data/sauce.db.bak.20260313_190553` | 1,507 | 275 | 21 | 0 | Mar 8–13 |

**All subsequent analysis uses the backup DB `20260313_190553` as it contains the most data.**

### Signal Decomposition (Backup DB2: 275 signals)

| Category | Count | Identification Method |
|----------|-------|----------------------|
| Test contamination | 238 | Loop duration < 0.05s, `reasoning="stub"` |
| Alpaca auth error fallback | 30 | Loop duration 0.3–0.64s, `reasoning` contains "market data unavailable" |
| Genuine production hold | 7 | Loop `10013b33`, duration 6.69s, `reasoning` mentions "No qualifying setup" |
| LLM-generated signal | **0** | — |

---

## Section 1: Signal Distribution & Hold Rate Analysis

**Verdict: FAIL — No actionable signal data exists**

### Raw Distribution

| Side | Count | Percentage |
|------|-------|------------|
| hold | 254 | 92.36% |
| buy | 21 | 7.64% |
| sell | 0 | 0.00% |

### Interpretation

These numbers are **meaningless for quality assessment**:
- All 21 buy signals are test-contaminated (identical: BTC/USD, confidence=0.75, reasoning="stub")
- 254 holds decompose to: 217 test stubs + 30 auth error fallbacks + 7 genuine
- 0 sell signals across entire history

### Hold Reason Analysis (genuine signals only)

Only 7 genuine hold signals exist, all from loop `10013b33` (the only genuine production loop that ran the full pipeline):

| Symbol | Reasoning |
|--------|-----------|
| AAPL | No qualifying setup for AAPL (regime=RANGING) |
| MSFT | No qualifying setup for MSFT (regime=RANGING) |
| GOOGL | No qualifying setup for GOOGL (regime=RANGING) |
| AMZN | No qualifying setup for AMZN (regime=RANGING) |
| NVDA | No qualifying setup for NVDA (regime=RANGING) |
| SPY | No qualifying setup for SPY (regime=RANGING) |
| QQQ | No qualifying setup for QQQ (regime=RANGING) |

All 7 equities were correctly blocked by the setup scanner: `equity_trend_pullback` requires `TRENDING_UP` regime, and the market context was `RANGING`.

### ULTRATHINK Assessment

The 92.36% hold rate appears alarming on the surface. It is not. The real concern is simpler: **the system has never produced a real trading signal**. Whether this is because market conditions genuinely never qualified, or because the setup scanner is too restrictive, or because the system was never run long enough — cannot be determined from available data.

---

## Section 2: Confidence Calibration

**Verdict: CANNOT ASSESS — No LLM-generated confidence values**

### Confidence Distribution

| Confidence | Count | Source |
|------------|-------|--------|
| 0.00 | 247 | Hold signals (safe default) |
| 0.75 | 21 | Test stubs (hardcoded in test fixtures) |
| NULL | 7 | Regime-blocked holds (research.py `_safe_hold()` path) |

### Analysis

- `confidence=0.0` is the hardcoded default for `_safe_hold()` returns — these are not calibrated values
- `confidence=0.75` appears solely in test fixtures (`tests/test_loop.py` line 119)
- No LLM has ever assigned a confidence value in any production context
- The `Signal` schema correctly clamps confidence to [0.0, 1.0] via Pydantic validator

### Code Review: Confidence Pipeline

The confidence pipeline is well-structured but untested in production:

1. **Claude assigns initial confidence** in JSON response (research.py parses `conf` field)
2. **Confluence adjustment** modifies confidence ±0.10 based on multi-TF tier (confluence.py)
3. **Schema validator** clamps to [0.0, 1.0]

**Finding:** The pipeline is correctly designed. No assessment of calibration quality is possible without LLM-generated data.

---

## Section 3: Setup Scanner Gate

**Verdict: PASS (code review) — Gate logic correct but never tested with qualifying setup**

### Evidence

```
Total signals:          275
Test contaminated:      238  (bypassed scanner entirely)
Auth error fallbacks:    30  (failed before scanner)
Reached scanner:          7  (all blocked — regime mismatch)
Passed scanner:           0
Reached LLM:              0
```

### Code Review: Setup Scanner (`sauce/core/setups.py`, 903 lines)

Three setup strategies implemented:

| Setup | Eligible Regimes | Key Hard Conditions | Min Score |
|-------|-----------------|---------------------|-----------|
| `crypto_mean_reversion` | RANGING, TRENDING_UP | RSI<38, price near lower BB, MACD curling, down-candle volume surge | 60 |
| `equity_trend_pullback` | TRENDING_UP only | 20SMA > 50SMA, RSI 30–50, price near 20SMA, volume below avg | 55 |
| `crypto_breakout` | RANGING only | Price near upper BB, volume >2x avg, RSI 55–75, MACD positive | 65 |

**Architecture assessment:**
- Hard/soft condition separation is clean — hard conditions are binary gates, soft conditions contribute to a 0–100 score
- Disqualifiers (e.g., earnings within 5 days, extreme RSI, excessive spread) are checked independently
- Scoring uses weighted sum with configurable weights
- `scan_setups()` correctly filters by symbol type (crypto vs equity) and regime

**Concern:** The `equity_trend_pullback` setup requires `TRENDING_UP` regime. In the only genuine production run, the regime was `RANGING`. If `RANGING` is the predominant market regime, equity setups will *never* fire. This is by design (conservative), but it means the system may go weeks without generating equity signals.

### SUPERTHINK Assessment

The setup scanner gate is the primary reason no LLM calls have been made. This is **correct defensive behavior** — the scanner prevents Claude calls when no technical setup qualifies, saving API cost and avoiding hallucinated signals. However, the restrictiveness means the full downstream pipeline (multi-TF, Claude, debate, supervisor) remains unexercised.

---

## Section 4: Multi-Timeframe Confluence

**Verdict: CANNOT ASSESS — Code exists but never executed in production**

### Evidence

```sql
-- Signals with confluence data:
SELECT COUNT(*) FROM audit_events 
WHERE event_type='signal' AND json_extract(payload,'$.confluence_tier') IS NOT NULL;
-- Result: 0
```

Zero signals have confluence data because no signal ever passed the setup scanner gate.

### Code Review: Confluence Engine (`sauce/signals/confluence.py`, 163 lines)

| Tier | Requirements | Confidence Adjustment |
|------|-------------|----------------------|
| S1 | ≥3 timeframes aligned, weighted score ≥0.5 | +0.10 |
| S2 | ≥2 timeframes aligned, weighted score ≥0.3 | +0.05 |
| S3 | Mixed signals | 0.00 |
| S4 | Conflicting signals, weighted score < -0.3 | -0.10 |

**Timeframe weights:**

| Timeframe | Weight |
|-----------|--------|
| 5m | 0.10 |
| 15m | 0.15 |
| 1h | 0.25 |
| 4h | 0.30 |
| 1d | 0.20 |

**Architecture assessment:**
- Higher timeframes (4h, 1h) get appropriately higher weights
- S4 tier correctly applies negative confidence adjustment for conflicting signals
- Per-timeframe data is fetched in parallel via `ThreadPoolExecutor` (timeframes.py)
- Each timeframe independently classifies trend (SMA20 vs SMA50) and momentum (RSI + MACD histogram)

**Finding:** The confluence engine is well-structured. Cannot assess output quality without production execution data.

---

## Section 5: Regime Alignment

**Verdict: PARTIAL PASS — Logic correct for the one regime observed**

### Evidence

```
Market context events:      1 (RANGING)
Genuine hold signals:       7 (all cite regime=RANGING)
Regimes observed:           1 (RANGING only)
```

### Code Review: Regime Filtering

The regime filter operates at two levels:

1. **Loop-level pre-filter** (loop.py): Equities are skipped entirely when regime ≠ TRENDING_UP
2. **Setup-level filter** (setups.py): Each setup declares eligible regimes — `scan_setups()` skips ineligible setups

**Observation:** In the only genuine production loop, all 7 equity symbols were correctly gated by the regime filter. The 7 genuine hold signals confirm this with reasoning: "No qualifying setup for {SYMBOL} (regime=RANGING)".

**Limitation:** Only `RANGING` regime has been observed. Behavior under `TRENDING_UP`, `TRENDING_DOWN`, `VOLATILE`, or `CALM` regimes is untested in production.

---

## Code-Level Findings

### F-01: Double-Logging of Signal Events (MEDIUM) — RESOLVED

**Evidence:**

Signal events were logged in **three** code locations:

| Location | Event Logged | Payload Key | When |
|----------|-------------|-------------|------|
| `research.py` line 76 (`_safe_hold()`) | Hold signal | `"reason"` | Every hold |
| `research.py` line 423 (post-LLM) | LLM signal | `"reasoning"` | After Claude call |
| `loop.py` line 541 | All signals | `"reasoning"` | After research returns |

**Impact:** Every signal produced **2 audit events** — one from research.py and one from loop.py. In the genuine production loop `10013b33`, 7 symbols generated 14 signal events (7 from `_safe_hold()` + 7 from loop.py).

**Resolution:** Removed the `log_event(AuditEvent(event_type="signal"))` call from `loop.py`. Research.py is now the single source of signal audit events (`_safe_hold()` for holds, post-LLM log for Claude signals). Loop.py now calls `log_signal(SignalRow(...))` instead, populating the typed `signals` table (see F-04). 873 tests pass.

---

### F-02: Field Name Inconsistency — `"reason"` vs `"reasoning"` (MEDIUM) — RESOLVED

**Evidence:**

`_safe_hold()` in research.py used `"reason"` while its sibling post-LLM log and loop.py both used `"reasoning"`. Supervisor events correctly use `"reason"` consistently (different event type, no conflict).

**Resolution:** Changed `_safe_hold()` payload from `{"reason": reason}` to `{"reasoning": reason, "confidence": 0.0}`. All signal events now consistently use `"reasoning"`. 873 tests pass.

---

### F-03: `diagnose.py` Queries Wrong Field Name for Signals (LOW-MEDIUM) — RESOLVED

**Evidence:**

`scripts/diagnose.py` line 125 queried signal events using `json_extract(payload, '$.reason')`, which missed signals using `"reasoning"`.

**Resolution:** Changed to `COALESCE(json_extract(payload, '$.reasoning'), json_extract(payload, '$.reason'))` for backward compatibility with any historical data that used the old field name. Supervisor query at line 90 left unchanged (correctly uses `$.reason`). 873 tests pass.

---

### F-04: `SignalRow` Table / `log_signal()` Never Called (LOW) — RESOLVED

**Evidence:**

- `sauce/adapters/db.py` defined `SignalRow` and `log_signal()` but `log_signal()` was never called
- `get_recent_signals()` queries the `signals` table and IS called in production (`research.py` line 177) to provide signal history context to Claude — this was returning empty, breaking the feedback loop

**Resolution:** Wired `log_signal(SignalRow(...))` into `loop.py` at the signal processing point (replacing the removed `log_event` from F-01). The `signals` table is now populated with every signal, and `get_recent_signals()` will return actual history for the Claude prompt. 873 tests pass.

---

### F-05: No Genuine Production Trading Data Exists (HIGH)

**Evidence:**

Across all three databases:
- **0** LLM calls made
- **0** signals passed the setup scanner
- **0** debate evaluations run
- **0** supervisor approvals on genuine signals
- **0** live orders placed from genuine signals
- **1** genuine production loop completed the full pipeline (to setup scanner gate)

**Impact:** The system has code-level readiness for production trading but has never executed a complete signal-to-order pipeline. All downstream components (debate, supervisor, execution, risk management) remain unexercised in production.

**Root Causes:**
1. Alpaca API authentication errors (401) prevented market data fetch in most loops
2. When market data was available, the market regime was RANGING — blocking equity setups
3. System was not run for sufficient duration with valid API credentials

---

## Signal Pipeline Architecture Review

### Pipeline Flow (Verified via Code Review)

```
Symbol → fetch OHLCV (500 bars crypto / 60 equity)
       → compute_all indicators (15 indicators)
       → fetch daily bars (higher-TF context)
       → fetch memory context (session + strategic + similar trades)
       → scan_setups() [GATE]
           ├── No qualifying setup → _safe_hold() → STOP
           └── Setup qualifies → fetch multi-TF data (5m/15m/1h/4h/1d)
                              → compute confluence score + tier
                              → build Claude prompt (with memory + confluence)
                              → call Claude API
                              → parse JSON response
                              → validate against Signal schema
                              → return Signal
       → [on any error at any stage] → _safe_hold() → STOP
```

### Architecture Quality Assessment

| Aspect | Assessment |
|--------|-----------|
| Error handling | EXCELLENT — every stage has try/except → _safe_hold() |
| Schema validation | GOOD — Pydantic v2 with `extra="forbid"`, confidence clamped |
| Setup gate design | GOOD — deterministic, no LLM waste on weak setups |
| Memory integration | GOOD — session + strategic + similar trade context fed to Claude |
| Multi-TF design | GOOD — parallel fetch, weighted tier logic |
| Defensive defaults | EXCELLENT — system holds by default, never acts on error |

---

## Improvement Recommendations

### IMP-05: Signal Reasoning Quality Scoring

**Status:** Implemented (code exists) but **not exercisable** — no LLM signals have been generated.

When genuine LLM signals arrive, re-audit to verify:
- Reasoning contains specific indicator references (not vague language)
- Bear case is substantive (not a restatement of bull case negated)
- Confidence correlates with reasoning strength

### IMP-06: Cross-Symbol Signal Correlation Detection

**Status:** Not implementable yet — requires multiple simultaneous buy signals to detect correlation.

### IMP-07: Claude Refusal Rate Tracking

**Status:** Implemented (error paths track refusals). No data to evaluate — 0 Claude calls.

### IMP-08: Reasoning Injection Audit

**Status:** Implemented (prompt construction reviewed — no injection vectors found in code). Cannot test with live data.

---

## New Recommendations from This Audit

### IMP-25: Standardize Signal Payload Field Names — IMPLEMENTED

Standardized all signal events to use `"reasoning"` consistently. `_safe_hold()` payload updated, `diagnose.py` uses COALESCE for backward compat. Implemented as part of F-02/F-03 fixes.

### IMP-26: Eliminate Signal Double-Logging — IMPLEMENTED

Chose Option B: removed loop.py's `log_event(AuditEvent(event_type="signal"))` call. Research.py retained as single source of signal audit events (has richer payload with indicator detail). Loop.py now calls `log_signal(SignalRow(...))` instead, activating the typed signals table.

### IMP-27: Production Readiness Validation Run

Run a controlled production loop with:
1. Valid Alpaca API credentials (resolve 401 errors)
2. A symbol universe that includes crypto (for regime-agnostic setups like `crypto_mean_reversion`)
3. Sufficient market volatility for at least one setup to qualify
4. Logging verification that the full pipeline executes through Claude

This is a **prerequisite** for meaningful re-audit of Sections 1–4.

### IMP-28: Activate or Remove `SignalRow` / `log_signal()` — IMPLEMENTED

Wired `log_signal(SignalRow(...))` into `loop.py`. The `signals` table is now active and `get_recent_signals()` will return data for Claude's signal history context.

---

## Anti-Hallucination Verification

Every finding in this audit is backed by specific evidence:

| Finding | Evidence Source |
|---------|---------------|
| 0 LLM calls | `SELECT COUNT(*) FROM audit_events WHERE event_type IN ('llm_call','claude_call')` → 0 across all 3 DBs |
| 238 test stubs | Loop duration analysis: all stub loops < 0.05s; "stub" string exists only in test files |
| 7 genuine holds | Loop `10013b33` duration = 6.69s; reasoning contains "No qualifying setup" with regime |
| 30 auth errors | Reasoning contains "market data unavailable"; associated with Alpaca 401 errors in error events |
| Double-logging | Loop `10013b33`: 14 signal events for 7 symbols; 268 total from loop.py + 7 from research.py |
| Field name inconsistency | Direct code read: research.py:78 uses `"reason"`, loop.py:544 uses `"reasoning"` |
| diagnose.py bug | Direct code read: diagnose.py:125 uses `$.reason` for signal events |

---

## Verdict Summary

| Section | Verdict | Reason |
|---------|---------|--------|
| 1. Signal Distribution | FAIL | No genuine signal data; 100% test/error contamination |
| 2. Confidence Calibration | CANNOT ASSESS | No LLM-generated confidence values exist |
| 3. Setup Scanner Gate | PASS (code) | Gate logic correct; correctly prevented LLM calls when no setup qualified |
| 4. Multi-TF Confluence | CANNOT ASSESS | Code is well-structured but never executed in production |
| 5. Regime Alignment | PARTIAL PASS | Correct for RANGING; only 1 of 5 regimes observed |

| Finding | Severity | Status |
|---------|----------|--------|
| F-01: Double-logging | MEDIUM | **RESOLVED** — loop.py signal audit_event removed; replaced with `log_signal()` |
| F-02: Field name inconsistency | MEDIUM | **RESOLVED** — `_safe_hold()` now uses `"reasoning"` consistently |
| F-03: diagnose.py wrong field | LOW-MEDIUM | **RESOLVED** — uses `COALESCE($.reasoning, $.reason)` |
| F-04: SignalRow dead code | LOW | **RESOLVED** — `log_signal()` wired into loop.py; signals table now active |
| F-05: No production data | HIGH | Open — requires ops intervention |

---

## Next Steps

1. ~~**Fix F-01 through F-03** before the next production run (IMP-25, IMP-26)~~ — **DONE**
2. **Resolve Alpaca 401 authentication** to enable genuine market data fetch
3. **Run IMP-27 validation loop** to generate real signal data
4. **Re-audit Sections 1, 2, 4** once LLM-generated signals exist
5. Proceed to **Audit 03 (LLM Calibration)** — note: will face similar data limitations

---

*Audit 02 complete. All findings evidence-backed. Confidence in code-level assessments: HIGH. Confidence in signal quality assessment: N/A (no data).*
