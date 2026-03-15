# Audit 06 — Data Integrity & Database Health

**Version:** 1.0  
**Date:** 2026-03-15  
**Auditor:** AI (Claude Opus 4.6)  
**System:** Sauce — Autonomous Multi-Agent Trading System  
**Databases Audited:** `data/sauce.db`, `data/strategic_memory.db`, `data/session_memory.db`

---

## Overall Verdict: NEEDS MAINTENANCE

| Section | Verdict | Notes |
|---------|---------|-------|
| S1 — Integrity Checks (GATE) | PASS | All three DBs pass `integrity_check` and `quick_check` |
| S2 — Audit DB Structure | PASS | 22 audit events, all structurally valid |
| S3 — Cross-Table Consistency | PASS | Vacuously — no orders, signals, loops, or daily_stats |
| S4 — Strategic Memory | WARN | Test contamination: 1 setup_performance + 2 validation_results rows |
| S5 — Daily Stats Completeness | PASS | Vacuously — 0 daily_stats rows |
| S6 — DB Sizes & Maintenance | PASS | All databases < 0.1 MB, no WAL files |
| S7 — Session Memory Reset | WARN | Stale data from 2026-03-12, expected (no run since then) |

**Findings:** 1 code-path finding (F-01 CRITICAL)

---

## Section 1 — SQLite Integrity Checks (GATE)

Gate result: **CLEARED**

| Database | `PRAGMA integrity_check` | `PRAGMA quick_check` |
|----------|--------------------------|----------------------|
| `data/sauce.db` | ok | ok |
| `data/strategic_memory.db` | ok | ok |
| `data/session_memory.db` | ok | ok |

All three databases pass both physical integrity checks. No B-tree corruption, no page-level errors.

---

## Section 2 — Audit DB Structural Validation

### 2a — Row Counts

| Table | Row Count |
|-------|-----------|
| `audit_events` | 22 |
| `orders` | 0 |
| `signals` | 0 |
| `daily_stats` | 0 |

System has not executed trading operations. The 22 audit_events are exclusively broker connectivity checks.

### 2b — Event Type Distribution

| Event Type | Count | First Seen | Last Seen |
|------------|-------|------------|-----------|
| `broker_call` | 11 | 2026-03-14 14:17:19 | 2026-03-15 07:35:49 |
| `broker_response` | 11 | 2026-03-14 14:17:19 | 2026-03-15 07:35:49 |

All events are broker call/response pairs from diagnostic scripts. No trading events.

### 2c — Invalid Confidence Values

```sql
SELECT COUNT(*) FROM signals WHERE confidence NOT BETWEEN 0 AND 1;
-- Result: 0 rows in signals table (vacuous pass)
```

### 2d — Impossible Order Combinations

```sql
SELECT COUNT(*) FROM orders WHERE side='BUY' AND pnl > 0 AND status='FILLED';
-- Result: 0 rows in orders table (vacuous pass)
```

### 2e — Malformed Payload JSON

```sql
SELECT COUNT(*) FROM audit_events WHERE json_valid(payload) = 0;
-- Result: 0 (all 22 payloads are valid JSON)
```

**Verdict: PASS** — All 22 audit events are structurally valid. Remaining tables are empty.

---

## Section 3 — Cross-Table Consistency

### 3a — Orders Without Supervisor Decision

```
No orders exist — vacuous pass
```

### 3b — Incomplete Loops (Start Without End)

```
No loop_start or loop_end events exist — vacuous pass
```

### 3c — Signals Without Audit Event

```
No signals exist — vacuous pass
```

### 3d — daily_stats loop_runs Mismatch

```
No daily_stats exist — vacuous pass
```

**Verdict: PASS** — No data exists to be inconsistent. When trading begins, re-run this section.

---

## Section 4 — Strategic Memory Validation

### 4a — Row Counts

| Table | Row Count |
|-------|-----------|
| `setup_performance` | 1 |
| `claude_calibration` | 0 |
| `regime_transitions` | 0 |
| `veto_patterns` | 0 |
| `weekly_performance` | 0 |
| `symbol_learned_behavior` | 0 |
| `validation_results` | 2 |

**UNEXPECTED:** 1 `setup_performance` row and 2 `validation_results` rows exist despite no trading activity.

### 4a — Contamination Evidence

**setup_performance row (id=1):**

| Field | Value | Concern |
|-------|-------|---------|
| `symbol` | `BTCUSD` | Broker format, not canonical `BTC/USD` |
| `setup_type` | `equity_trend_pullback` | Equity setup on crypto symbol |
| `regime_at_entry` | `RANGING` | — |
| `time_of_day_bucket` | `19:00` | — |
| `win` | `1` | — |
| `pnl` | `20.0` | Round number, looks synthetic |
| `hold_duration_minutes` | `0.0` | Impossible for real trade |
| `date` | `2026-03-15` | Today — coincides with test runs |

**validation_results rows:**

| Date | Total Signals | Wins | Losses | All Passed | Consecutive Days |
|------|--------------|------|--------|------------|-----------------|
| 2026-03-12 | 0 | 0 | 0 | 0 | 0 |
| 2026-03-15 | 0 | 0 | 0 | 0 | 0 |

All-zero metrics with `all_passed=0` — typical of validation stub calls.

### 4b — Value Validation

The single `setup_performance` row has no NULL fields and no invalid values (confidence and date within range). However, the content itself is test data, not production data.

### 4c — claude_calibration

```
0 rows — no validation issues
```

### 4d — Win/Loss P&L Inconsistencies

```
No inconsistencies — the single row (win=1, pnl=20.0) is consistent
```

**Verdict: WARN — Test contamination confirmed (see F-01)**

---

## Section 5 — Daily Stats Completeness

### 5a — Impossible Values

```
0 daily_stats rows — vacuous pass
```

### 5b — Orders Placed but NAV=0

```
No matching rows — vacuous pass
```

### 5c — Validation Consecutive Days Accuracy

The 2 `validation_results` rows both have `consecutive_days=0`, which is correct (all metrics are zero, `all_passed=0`).

**Verdict: PASS** — No data to validate. Re-run after trading begins.

---

## Section 6 — DB Sizes & Maintenance

### 6a — Database Sizes

| Database | Size (MB) | Threshold | Status |
|----------|-----------|-----------|--------|
| `data/sauce.db` | 0.051 | 50 MB | OK |
| `data/strategic_memory.db` | 0.086 | 10 MB | OK |
| `data/session_memory.db` | 0.055 | 1 MB | OK |

### 6b — WAL/SHM Files

```
No .wal or .shm files present for any database
```

All databases are in rollback journal mode with no pending transactions.

### 6c — Maintenance Assessment

All databases are < 0.1 MB. No fragmentation concerns. `VACUUM` not needed. `run_maintenance()` has not been invoked (no loop runs have completed), which is fine given the database sizes.

**Verdict: PASS** — All databases are healthy and well within size thresholds.

---

## Section 7 — Session Memory Reset Verification

### 7a — Current Session Memory Contents

| Table | Row Count | Oldest Entry |
|-------|-----------|-------------|
| `regime_log` | 1 | 2026-03-12 |
| `signal_log` | 0 | — |
| `trade_log` | 0 | — |
| `intraday_narrative` | 1 | 2026-03-12 |

### 7b — Stale Data Assessment

Data from 2026-03-12 persists (3 days old). This is **expected behavior**:

- `reset_session_memory_if_new_day()` uses a module-global `_last_reset_date` variable
- On process restart, `_last_reset_date` is `None`, which always triggers reset
- No process has run since 2026-03-12, so no reset has been triggered
- On next `run_loop.sh` invocation, the reset will fire automatically

### 7c — Reset Mechanism Analysis

The reset function (`sauce/memory/db.py` lines 347-395) truncates: `regime_log`, `signal_log`, `trade_log`, `intraday_narrative`, `symbol_character`. It intentionally preserves `position_peak_pnl` for multi-day positions. This design is correct for the cron-based architecture.

**Verdict: WARN — Stale data present but expected. Will auto-clear on next run. Not actionable.**

---

## Findings

### F-01 [CRITICAL] — Test Contamination of Production Strategic Memory DB

**Category:** Data Integrity  
**Severity:** CRITICAL  
**Impact:** Production `data/strategic_memory.db` contains test data that would corrupt learning signals and validation metrics if trading were active.

**Evidence:**
- 1 row in `setup_performance` with synthetic characteristics: `BTCUSD` (broker format), `hold_duration_minutes=0.0`, `equity_trend_pullback` on crypto, `pnl=20.0` (round number)
- 2 rows in `validation_results` with all-zero metrics
- Dates align with test runs (2026-03-12, 2026-03-15)

**Root Cause:**

`tests/conftest.py` `_isolate_db` fixture (lines 17-50):
- Sets `DB_PATH` env var to `tmp_path` → isolates `sauce.db` ✓
- Does **NOT** set `STRATEGIC_MEMORY_DB_PATH` → defaults to `data/strategic_memory.db` ✗
- Does **NOT** set `SESSION_MEMORY_DB_PATH` → defaults to `data/session_memory.db` ✗
- IMP-07 teardown assertion only checks `db_module._engines` keys, **NOT** `memory_db_module._engines` ✗

`sauce/core/config.py` (lines 210-212):
```python
db_path: str = Field(default="data/sauce.db")
session_memory_db_path: str = Field(default="data/session_memory.db")
strategic_memory_db_path: str = Field(default="data/strategic_memory.db")
```

Any test that calls code using `get_settings().strategic_memory_db_path` or `get_settings().session_memory_db_path` without passing an explicit `db_path` parameter writes to the production databases.

**Resolution:**
1. Add `STRATEGIC_MEMORY_DB_PATH` and `SESSION_MEMORY_DB_PATH` monkeypatch to `_isolate_db` fixture
2. Extend IMP-07 teardown assertion to check `memory_db_module._engines` against production memory DB paths
3. Clean contaminated rows from `data/strategic_memory.db`

**Status:** FIXED (see Resolution section below)

---

## IMP Verification — Code-Path Analysis

### IMP-21 — WAL Checkpointing

**Status:** NOT IMPLEMENTED  
`run_maintenance()` (`sauce/adapters/db.py` lines 343-403) does not include `PRAGMA wal_checkpoint`. Currently, all databases run in rollback journal mode (no WAL files present), so this is not immediately needed. Would become important if WAL mode is enabled for concurrent reads.

### IMP-22 — Append-Only Constraint Verification

**Status:** NOT IMPLEMENTED  
No programmatic verification exists to confirm append-only behavior. Additionally, `run_maintenance()` itself runs `DELETE FROM audit_events WHERE timestamp < :cutoff` for data pruning, which contradicts any append-only design claim. The pruning is intentional (retention_days=90), but there is no audit trail for deletions.

### IMP-23 — Transactional Grouping for Related Writes

**Status:** NOT IMPLEMENTED  
`log_order()` and `log_event()` each create separate SQLAlchemy sessions. A crash between a `log_order()` call and its corresponding `log_event()` call would create orphaned records. For the current cron-based single-threaded architecture, this risk is low but not zero.

### IMP-24 — Automated Integrity Check on Startup

**Status:** NOT IMPLEMENTED  
`main()` (`sauce/core/loop.py`) does not run `PRAGMA integrity_check` at startup. `run_maintenance()` only runs at the end of a successful loop iteration. `scripts/run_loop.sh` has no pre-flight DB checks. A corrupted database would only be detected when a query fails at runtime.

---

## Resolution — F-01 Fix Applied

### 1. conftest.py Updated

Added `STRATEGIC_MEMORY_DB_PATH` and `SESSION_MEMORY_DB_PATH` env var isolation to `_isolate_db` fixture. Extended IMP-07 teardown assertion to check `memory_db_module._engines` against both production memory DB paths.

### 2. Production Data Cleaned

Deleted contaminated rows:
- 1 row from `setup_performance` (id=1)
- 2 rows from `validation_results`

### 3. Test Suite Verification

Full test suite re-run after fix — all tests pass. No strategic or session memory engine leaks detected.

---

## Sign-Off

| Field | Value |
|-------|-------|
| Audit completed | 2026-03-15 |
| Sections completed | 7 / 7 |
| Findings | 1 (F-01 — CRITICAL, FIXED) |
| Recommendations | IMP-21 through IMP-24 documented, not implemented |
| Overall verdict | NEEDS MAINTENANCE → PASS (after F-01 fix applied) |
| Next audit due | After first live trading session, or monthly, whichever is sooner |
| Blocking issues | None after F-01 resolution |
