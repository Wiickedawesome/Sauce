# Sauce Audit 05 — Setup Performance & Strategy Engine Analysis

**Audit Date:** 2026-03-15  
**Classification:** MEDIUM-HIGH PRIORITY  
**Period Covered:** All available data (system inception through 2026-03-15)  
**Auditor:** Strategy Performance Auditor (AI-assisted)  
**Databases Examined:** `data/strategic_memory.db`, `data/sauce.db`, `data/sauce.db.bak.20260313_190553`, `data/sauce.db.bak.20260307_215539`

---

## Executive Summary

**OVERALL VERDICT: CANNOT ASSESS — ZERO TRADE DATA**

All six query-based sections return empty result sets. The system has never completed a production trading loop (established in Audits 01 and 04), so zero rows exist in `setup_performance`, `veto_patterns`, `weekly_performance`, `symbol_learned_behavior`, debate audit events, or orders.

However, a **code-path verification audit** of the strategy pipeline reveals **five critical findings** — three are dead code paths that will prevent the learning system from functioning even after production deployment, and two are data quality bugs that will produce misleading analytics. These must be fixed before production data can be trusted.

**Finding Summary:**

| ID | Severity | Title |
|----|----------|-------|
| F-01 | CRITICAL | `veto_patterns` write path is dead — never called from production code |
| F-02 | CRITICAL | `symbol_learned_behavior` write path is dead — `update_symbol_learned_behavior()` never invoked |
| F-03 | CRITICAL | Weekly learning cycle unreachable — `run_weekly` flag never set to True |
| F-04 | WARN | `regime_at_entry` records exit-time regime, not entry-time regime |
| F-05 | WARN | `time_of_day_bucket` records exit-time hour, not entry-time hour; format mismatches schema docs |

---

## Section 1 — Overall Setup Performance Summary

### Evidence

```
sqlite3 data/strategic_memory.db "SELECT COUNT(*) FROM setup_performance;"
→ 0
```

All query results for 1a (top-level performance), 1b (expectancy), and 1c (trend) return **zero rows**.

### Verdict: **CANNOT ASSESS — no data**

No closed trades have been recorded. The `setup_performance` table is structurally sound (schema verified, write path from `_detect_closed_positions()` → `record_trade_outcome()` → `write_setup_performance()` is correctly wired and passes `settings.strategic_memory_db_path`).

**Code verification of the write path:**

- `_detect_closed_positions()` at [sauce/core/loop.py](sauce/core/loop.py#L87-L161) correctly compares previous position snapshot against current positions
- Position disappearance triggers `SetupPerformanceEntry` construction
- `record_trade_outcome()` at [sauce/memory/learning.py](sauce/memory/learning.py#L36-L44) delegates to `write_setup_performance()` at [sauce/memory/db.py](sauce/memory/db.py#L617-L636)
- `db_path` is correctly set to `settings.strategic_memory_db_path` (verified at [loop.py call site](sauce/core/loop.py#L453-L461))
- The earlier bug (using `settings.db_path` instead of `strategic_memory_db_path`) has been **fixed**

**Expectancy, profit factor, Kelly criterion:** Not computable — n=0.

---

## Section 2 — Regime Segmentation Analysis

### Evidence

```
sqlite3 data/strategic_memory.db "SELECT COUNT(*) FROM setup_performance;"
→ 0
```

Queries 2a (win rate by regime), 2b (regime violations), 2c (best/worst regimes) all return **zero rows**.

### Verdict: **CANNOT ASSESS — no data**

**Code verification of regime gates:**

Regime gates are correctly implemented in `scan_setups()` at [sauce/core/setups.py](sauce/core/setups.py#L872-L902):

| Setup | Constant | Allowed Regimes | Verified |
|-------|----------|-----------------|----------|
| Setup 1: `crypto_mean_reversion` | `SETUP_1_REGIMES` | `{"RANGING", "TRENDING_UP"}` | YES — gate at line 872 |
| Setup 2: `equity_trend_pullback` | `SETUP_2_REGIMES` | `{"TRENDING_UP"}` | YES — gate at line 884 |
| Setup 3: `crypto_breakout` | `SETUP_3_REGIMES` | `{"RANGING"}` | YES — gate at line 896 |

Additional per-setup disqualifiers within evaluation functions:
- Setup 1 disqualifies `VOLATILE` and `TRENDING_DOWN` (line 355-362)
- Setup 2 disqualifies `RANGING` and `VOLATILE` (line 564)
- Setup 3 disqualifies `VOLATILE` (line 831)

**Note [F-04]:** The `regime_at_entry` field in `SetupPerformanceEntry` does NOT record the regime at trade entry. It records the regime at position-close detection time. See Finding F-04 below.

---

## Section 3 — Time-of-Day Analysis

### Evidence

```
sqlite3 data/strategic_memory.db "SELECT COUNT(*) FROM setup_performance;"
→ 0
```

Queries 3a (win rate by time bucket) and 3b (hold duration by bucket) return **zero rows**.

### Verdict: **CANNOT ASSESS — no data**

**Code verification:**

- `time_of_day_bucket` is assigned at [loop.py line 126](sauce/core/loop.py#L126): `now.strftime("%H:00")`
- This produces single-hour strings like `"14:00"`, `"15:00"` — **not** the range format documented in the Pydantic schema docstring (which suggests `"09:30-12:00"`, `"12:00-14:00"`, `"14:00-16:00"`)
- The value uses **exit time (UTC)**, not entry time — see Finding F-05
- No `time_of_day_bucket` disqualifier exists in `scan_setups()` — time-based filtering is done via explicit checks (e.g., Setup 2 Friday afternoon disqualifier at setups.py line 578), not via bucket-based logic

---

## Section 4 — Symbol-Level Learned Behavior

### Evidence

```
sqlite3 data/strategic_memory.db "SELECT COUNT(*) FROM symbol_learned_behavior;"
→ 0
```

Queries 4a (completeness), 4b (symbol performance with learned data), 4c (consistent losers) return **zero rows**.

### Verdict: **CANNOT ASSESS — no data | CRITICAL code path dead (F-02)**

Even if `setup_performance` contained trade data, the `symbol_learned_behavior` table would **never be populated** because `update_symbol_learned_behavior()` is never called from production code (see Finding F-02).

**Code verification:**

- `update_symbol_learned_behavior()` at [learning.py lines 152-191](sauce/memory/learning.py#L152-L191) correctly queries `setup_performance`, requires ≥3 trades, computes `avg_reversion_depth` and `avg_bounce_magnitude`
- `optimal_rsi_entry` is **always None** — the field exists in the schema but is never computed
- `write_symbol_behavior()` at [db.py lines 722-749](sauce/memory/db.py#L722-L749) correctly does an upsert
- **BUT:** No caller exists in the production code path. Not in `run_learning_cycle()`, not in `ops.run()`, not in `_detect_closed_positions()`
- Only callers are in `tests/test_learning.py`

The RAG system (`get_strategic_context()` at [db.py line 853](sauce/memory/db.py#L853)) correctly reads from this table and would inject symbol-specific calibration into Claude's prompts — but since the table is never populated, this context is always empty.

---

## Section 5 — Consecutive Loss Streak Analysis

### Evidence

```
sqlite3 data/strategic_memory.db "SELECT COUNT(*) FROM setup_performance;"
→ 0

sqlite3 data/strategic_memory.db "SELECT COUNT(*) FROM veto_patterns;"
→ 0
```

Queries 5a (last 20 trades), 5b (max streak), 5c (veto patterns) return **zero rows**.

### Verdict: **CANNOT ASSESS — no data | CRITICAL code path dead (F-01, F-03)**

The ops agent's circuit breaker check at [sauce/agents/ops.py](sauce/agents/ops.py#L234) calls `run_learning_cycle()`, which calls `detect_win_rate_drift()`. This function checks the last 20 `setup_performance` rows and alerts if win rate < 45%. This path is correctly wired and will function once trades are recorded.

However:
- **Veto patterns** (query 5c) will never be populated — see Finding F-01
- **Weekly aggregation** will never run — see Finding F-03

---

## Section 6 — Debate Layer Correlation with Outcomes

### Evidence

```
sqlite3 data/sauce.db "SELECT COUNT(*) FROM audit_events WHERE event_type='debate';"
→ 0

sqlite3 data/sauce.db.bak.20260313_190553 "SELECT COUNT(*) FROM audit_events WHERE event_type='debate';"
→ 0

sqlite3 data/sauce.db.bak.20260307_215539 "SELECT COUNT(*) FROM audit_events WHERE event_type='debate';"
→ 0
```

Query 6a (debate verdict vs outcome correlation) returns **zero rows** — no debate events exist in any database.

### Verdict: **CANNOT ASSESS — no data**

**Code verification of debate pipeline:**

- `run_debate()` at [sauce/agents/debate.py](sauce/agents/debate.py#L287-L327) is purely deterministic (no LLM)
- Verdict logic: bull/bear score margin > 0.5 → `"bull_wins"`/`"bear_wins"`, else `"contested"`
- Confidence adjustment is asymmetric: max bonus +0.05, max penalty -0.10
- Debate events are logged at [loop.py lines 604-618](sauce/core/loop.py#L604-L618) with `event_type="debate"`, payload includes `verdict`, `bull_score`, `bear_score`, `confidence_adjustment`, `summary`
- The adjustment is passed to the supervisor prompt but does NOT mutate the signal's `.confidence` field

**When data exists**, query 6a's JOIN on `DATE(ae.timestamp) = sp.date AND ae.symbol = sp.symbol` should correctly correlate debate verdicts with trade outcomes, though a date+symbol join may produce false matches if multiple debates occur for the same symbol on the same day.

---

## Findings

### F-01 [CRITICAL → RESOLVED] — `veto_patterns` Write Path is Dead Code

**Evidence:** `grep -rn "write_veto_pattern" sauce/` returns only the function definition at [sauce/memory/db.py line 666](sauce/memory/db.py#L666). Zero production callers. Only test callers in `tests/test_memory_db.py`.

**Impact:** The `veto_patterns` table will never be populated. The supervisor's veto decisions are not recorded for strategic learning. The RAG context (`get_strategic_context()`) queries this table and injects veto history into Claude's prompts, but will always return empty results. The system cannot learn from its own veto patterns.

**Root cause:** The write function was implemented and tested, but never wired into the supervisor veto path.

**Resolution:** Wired `write_veto_pattern()` into the supervisor abort branch in `loop.py`. When the supervisor vetoes a signal, the code now iterates all actionable signals and writes a `VetoPatternEntry` for each, recording `setup_type`, `veto_reason` (from supervisor reasoning), and `symbol`. Tests: `TestVetoPatternRecording` (2 tests) verify correct recording on abort and no recording for hold signals. Verified in full suite (888 passed).

---

### F-02 [CRITICAL → RESOLVED] — `symbol_learned_behavior` Write Path is Dead Code

**Evidence:** `grep -rn "update_symbol_learned_behavior" sauce/` returns only the function definition at [learning.py line 152](sauce/memory/learning.py#L152). Zero production callers. Only test callers.

**Impact:** The `symbol_learned_behavior` table will never be populated. Per-symbol calibration data (`optimal_rsi_entry`, `avg_reversion_depth`, `avg_bounce_magnitude`) will never be computed from trade outcomes. The RAG system will never provide Claude with symbol-specific historical context, even after hundreds of trades.

**Root cause:** `update_symbol_learned_behavior()` exists as a standalone function but was never added to `run_learning_cycle()` or any other production invocation point.

**Resolution:** Added a symbol behavior update block to `run_learning_cycle()` in `learning.py`. The block queries distinct `(symbol, setup_type)` pairs from `setup_performance`, then calls `update_symbol_learned_behavior()` for each pair. Returns the count in `results["symbol_behavior_updated"]`. Tests: `TestSymbolBehaviorInLearningCycle` (2 tests) verify correct update with data and no-op with empty data. Verified in full suite (888 passed).

**Additional note:** The `optimal_rsi_entry` field is **never computed** — it is always `None`. Either implement the computation (e.g., median RSI at entry for winning trades) or remove the field to avoid confusion.

---

### F-03 [CRITICAL → RESOLVED] — Weekly Learning Cycle is Unreachable

**Evidence:** The summary dict built at [loop.py lines 1240-1255](sauce/core/loop.py#L1240-L1255) never includes a `"run_weekly"` key. `ops.run()` reads `summary.get("run_weekly", False)`, which is always `False`. Therefore `run_learning_cycle(run_weekly=True)` is never called. `generate_weekly_report()` and `analyze_claude_calibration()` are dead code.

**Impact:** The `weekly_performance` table will never be populated. Weekly aggregated metrics (trades, win_rate, avg_pnl, sharpe per setup type per ISO week) will never be computed. The RAG system's `weekly_trend` context will always be empty. Claude will never see performance trends in its prompts.

**Root cause:** The loop summary dict was designed to carry all metrics to `ops.run()`, but no logic was added to determine when it's a "weekly" run. There is no day-of-week check, no weekly scheduler, and no flag-setting code.

**Resolution:** Added `"run_weekly": datetime.now(timezone.utc).weekday() == 6` to the summary dict in `loop.py`. This sets the flag to `True` on Sundays (ISO weekday 6), triggering the weekly learning cycle. Tests: `TestRunWeeklyFlag` (3 tests) verify True on Sunday, False on Mon-Sat, and the key's presence in source code. Verified in full suite (888 passed).

---

### F-04 [WARN → RESOLVED] — `regime_at_entry` Records Exit-Time Regime, Not Entry-Time Regime

**Evidence:** At [loop.py line 129](sauce/core/loop.py#L129), `regime_at_entry` is set to the `regime` parameter passed to `_detect_closed_positions()`. This parameter comes from [loop.py line 457](sauce/core/loop.py#L457): `regime=mkt_ctx.regime.regime_type`, which is the market regime **at position-close detection time**, not at trade entry.

**Impact:** Regime segmentation analysis (Section 2) will be misleading. A trade entered during RANGING that exits during TRENDING_UP will be classified as a TRENDING_UP trade. Regime violation detection (query 2b) will produce false positives and false negatives. The audit's regime gate verification will be unreliable.

**Root cause:** The true entry-time regime is not persisted anywhere. The `signal_log` in session memory may contain it, but session memory is wiped daily (by design), and positions may be held across days.

**Severity rationale:** WARN rather than CRITICAL because: (a) most Sauce trades are intraday with short holds (target < 90 minutes), so regime drift during the hold period is uncommon; (b) the system has zero data now, so this is a design bug to fix before data accumulates rather than an active data corruption.

**Resolution:** Added `_entry_regime` tagging to the `_previous_positions` global dict in `loop.py`. When a new position is detected, the current regime is stored as `_entry_regime` on the position snapshot. This tag is carried forward across loop iterations. When a position closes, `regime_at_entry` now reads from `prev.get("_entry_regime", regime)`, falling back to current regime only if no tag exists. Tests: `TestDetectClosedPositions` (4 tests) verify stored regime is used over current regime, new positions get tagged, and tags carry forward. Verified in full suite (888 passed).

---

### F-05 [WARN → RESOLVED] — `time_of_day_bucket` Uses Exit Time and Mismatched Format

**Evidence:** At [loop.py line 126](sauce/core/loop.py#L126): `time_of_day_bucket=now.strftime("%H:00")`. This uses `datetime.now(timezone.utc)` — the **exit time** in UTC, not the entry time.

**Two issues:**

1. **Exit time vs entry time:** Time-of-day analysis is meaningful at entry time (when the trading decision was made), not exit time. A trade entered at 09:35 (morning open momentum) that exits at 14:00 will be bucketed as an afternoon trade, obscuring the actual pattern.

2. **Format mismatch:** The code produces single-hour strings like `"14:00"`. The `SetupPerformanceEntry` Pydantic schema docstring suggests range-format values like `"09:30-12:00"`, `"12:00-14:00"`, `"14:00-16:00"`. The hourly format is actually more granular and arguably better, but the inconsistency with documentation may cause confusion in future analysis.

**Root cause:** Same as F-04 — entry context is not persisted for position-close detection.

**Resolution:** `time_of_day_bucket` now uses `entry_time` when available: `_bucket_time = entry_time if entry_time else now`, then `_bucket_time.strftime("%H:00")`. This ensures the bucket reflects the hour of the trading decision, not the exit. Test: `test_time_of_day_bucket_uses_entry_time` verifies that entry hour (09:00) is used instead of exit hour (15:00). Verified in full suite (888 passed).

---

## Improvement Recommendations Status

| ID | Title | Status |
|----|-------|--------|
| IMP-17 | Dynamic Universe Pruning Based on Setup Performance | NOT IMPLEMENTED — blocked by F-01/F-02 (no data accumulation) |
| IMP-18 | Regime-Aware Setup Score Adjustment | NOT IMPLEMENTED — design recommendation |
| IMP-19 | Hold Duration Optimization per Setup | NOT IMPLEMENTED — blocked by F-02 (symbol learned behavior dead) |
| IMP-20 | Time Bucket Disqualifier Auto-Tuning | NOT IMPLEMENTED — blocked by F-05 (time bucket data unreliable) |

**Note:** All four IMPs are unimplementable until F-01 through F-05 are resolved and sufficient trade data accumulates.

---

## Code-Path Verification Summary

The audit could not perform any quantitative analysis due to zero trade data. Instead, a full code-path trace was performed on the strategy pipeline. Results:

| Pipeline Component | Status | Notes |
|-------------------|--------|-------|
| `_detect_closed_positions()` → `record_trade_outcome()` | **FUNCTIONAL** | Correctly wired, correct DB path |
| `setup_performance` schema and write | **FUNCTIONAL** | Will record data when positions close |
| `veto_patterns` write | **FIXED (F-01)** | `write_veto_pattern()` now called on supervisor abort |
| `symbol_learned_behavior` write | **FIXED (F-02)** | `update_symbol_learned_behavior()` now called in `run_learning_cycle()` |
| `weekly_performance` write | **FIXED (F-03)** | `run_weekly` flag set True on Sundays |
| `run_learning_cycle()` — drift detection | **FUNCTIONAL** | `detect_win_rate_drift()` correctly wired |
| `get_strategic_context()` — RAG reads | **FUNCTIONAL** | All tables correctly queried; returns empty when tables empty |
| Regime gates in `scan_setups()` | **CORRECT** | All 3 setup types correctly gated |
| Debate layer logging | **FUNCTIONAL** | Correct event_type, complete payload |
| `regime_at_entry` data quality | **FIXED (F-04)** | Now uses entry-time regime via `_entry_regime` tag |
| `time_of_day_bucket` data quality | **FIXED (F-05)** | Now uses entry time when available |
| `optimal_rsi_entry` computation | **NEVER COMPUTED** | Field exists but always NULL |

---

## Audit Sign-Off

```
AUDIT DATE:              2026-03-15
PERIOD COVERED:          System inception  to  2026-03-15
TOTAL CLOSED TRADES:     0

SETUP 1 (CMR) TRADES:    0   Win Rate: N/A
SETUP 2 (ETP) TRADES:    0   Win Rate: N/A
SETUP 3 (CB)  TRADES:    0   Win Rate: N/A

SETUP EXPECTANCY:
  crypto_mean_reversion: N/A (n=0)
  equity_trend_pullback: N/A (n=0)
  crypto_breakout:       N/A (n=0)

REGIME VIOLATIONS:       N/A (no trades)
SYMBOLS FOR REVIEW:      N/A (no trades)
ACTIVE LOSS STREAKS:     N/A (no trades)

SECTION 1 (OVERALL PERF):    CANNOT ASSESS — no data
SECTION 2 (REGIME SEG):      CANNOT ASSESS — no data (code gates PASS)
SECTION 3 (TIME BUCKET):     CANNOT ASSESS — no data
SECTION 4 (SYMBOL LEARNED):  CANNOT ASSESS — no data | CRITICAL: write path dead (F-02)
SECTION 5 (STREAKS):         CANNOT ASSESS — no data | CRITICAL: veto/weekly dead (F-01, F-03)
SECTION 6 (DEBATE CORR):     CANNOT ASSESS — no data

OPEN FINDINGS:
  F-01 [CRITICAL → RESOLVED] — veto_patterns write path wired into supervisor abort
  F-02 [CRITICAL → RESOLVED] — symbol_learned_behavior wired into run_learning_cycle
  F-03 [CRITICAL → RESOLVED] — run_weekly flag set on Sundays
  F-04 [WARN → RESOLVED]     — regime_at_entry uses stored entry regime
  F-05 [WARN → RESOLVED]     — time_of_day_bucket uses entry time

OVERALL VERDICT: CANNOT ASSESS — ZERO TRADE DATA
                 All 5 code-path findings RESOLVED (3 CRITICAL, 2 WARN)
                 12 new tests added (888 total passing)
                 Remaining: optimal_rsi_entry field never computed (cosmetic)
```

**Required before meaningful re-run:**
1. ~~Fix F-01, F-02, F-03 (wire dead code paths)~~ DONE
2. ~~Fix F-04, F-05 (data quality corrections)~~ DONE
3. Deploy to production (Audit 04 dependency)
4. Accumulate ≥30 trades per setup type (minimum for expectancy conclusions)
5. Re-run this audit against production data
