-- Sauce Trading Bot — Initial PostgreSQL Schema for Supabase
-- Migrated from SQLite on 2026-04-01
--
-- Tables:
--   1. audit_events    — Immutable audit log (append-only)
--   2. trades          — Completed trades with P&L (append-only)
--   3. positions       — Open positions with trailing state (mutable)
--   4. signal_log      — Every scoring result (append-only)
--   5. daily_summary   — Daily aggregates (upsert)
--   6. instrument_meta — Per-instrument config/regime cache
--   7. trade_memories  — BM25 trade reflections (append-only)
--   8. options_positions — Open options positions (mutable)
--   9. options_trades  — Completed options trades (append-only)

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ══════════════════════════════════════════════════════════════════════════════
-- 1. AUDIT EVENTS — Immutable log. Never updated. Never deleted.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS audit_events (
    id              BIGSERIAL PRIMARY KEY,
    loop_id         UUID NOT NULL,
    event_type      VARCHAR(64) NOT NULL,
    symbol          VARCHAR(20),
    payload         JSONB NOT NULL DEFAULT '{}',
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    prompt_version  VARCHAR(32),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_events_loop_id ON audit_events(loop_id);
CREATE INDEX idx_audit_events_event_type ON audit_events(event_type);
CREATE INDEX idx_audit_events_symbol ON audit_events(symbol) WHERE symbol IS NOT NULL;
CREATE INDEX idx_audit_events_timestamp ON audit_events(timestamp);

-- Partitioning by month for better performance (optional, for future scaling)
-- COMMENT: Consider partitioning this table if it grows beyond 10M rows

-- ══════════════════════════════════════════════════════════════════════════════
-- 2. TRADES — Completed trade record. Append-only.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        UUID NOT NULL UNIQUE,
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(8) NOT NULL,
    qty             DECIMAL(18, 8) NOT NULL,
    entry_price     DECIMAL(18, 8) NOT NULL,
    exit_price      DECIMAL(18, 8) NOT NULL,
    realized_pnl    DECIMAL(18, 8) NOT NULL,
    gross_realized_pnl DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    fees_paid       DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    slippage_paid   DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    strategy_name   VARCHAR(64) NOT NULL,
    exit_trigger    VARCHAR(32) NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ NOT NULL,
    hold_hours      DECIMAL(10, 2) NOT NULL,
    broker_order_id VARCHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_exit_time ON trades(exit_time);
CREATE INDEX idx_trades_strategy ON trades(strategy_name);

-- ══════════════════════════════════════════════════════════════════════════════
-- 3. POSITIONS — Open positions with trailing-stop state. Mutable.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS positions (
    id                  BIGSERIAL PRIMARY KEY,
    position_id         UUID NOT NULL UNIQUE,
    symbol              VARCHAR(20) NOT NULL,
    asset_class         VARCHAR(16) NOT NULL DEFAULT 'crypto',
    qty                 DECIMAL(18, 8) NOT NULL,
    entry_price         DECIMAL(18, 8) NOT NULL,
    high_water_price    DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    trailing_stop_price DECIMAL(18, 8),
    trailing_active     BOOLEAN NOT NULL DEFAULT FALSE,
    entry_time          TIMESTAMPTZ NOT NULL,
    broker_order_id     VARCHAR(64),
    strategy_name       VARCHAR(64) NOT NULL,
    stop_loss_price     DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    profit_target_price DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    status              VARCHAR(12) NOT NULL DEFAULT 'open',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_positions_symbol ON positions(symbol);
CREATE INDEX idx_positions_status ON positions(status);
CREATE INDEX idx_positions_open ON positions(status) WHERE status = 'open';

-- ══════════════════════════════════════════════════════════════════════════════
-- 4. SIGNAL LOG — Every scoring result. Append-only.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS signal_log (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL,
    side            VARCHAR(8) NOT NULL,
    score           INTEGER NOT NULL,
    threshold       INTEGER NOT NULL,
    fired           BOOLEAN NOT NULL,
    rsi_14          DECIMAL(10, 4),
    macd_hist       DECIMAL(18, 8),
    bb_pct          DECIMAL(10, 4),
    volume_ratio    DECIMAL(10, 4),
    regime          VARCHAR(16) NOT NULL,
    strategy_name   VARCHAR(64) NOT NULL,
    timestamp       TIMESTAMPTZ NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_signal_log_symbol ON signal_log(symbol);
CREATE INDEX idx_signal_log_timestamp ON signal_log(timestamp);
CREATE INDEX idx_signal_log_fired ON signal_log(fired) WHERE fired = TRUE;

-- ══════════════════════════════════════════════════════════════════════════════
-- 5. DAILY SUMMARY — Daily aggregates. Upserted once per day.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS daily_summary (
    id              BIGSERIAL PRIMARY KEY,
    date            DATE NOT NULL UNIQUE,
    loop_runs       INTEGER NOT NULL DEFAULT 0,
    signals_fired   INTEGER NOT NULL DEFAULT 0,
    signals_skipped INTEGER NOT NULL DEFAULT 0,
    orders_placed   INTEGER NOT NULL DEFAULT 0,
    trades_closed   INTEGER NOT NULL DEFAULT 0,
    realized_pnl_usd DECIMAL(18, 2) NOT NULL DEFAULT 0.0,
    gross_realized_pnl_usd DECIMAL(18, 2) NOT NULL DEFAULT 0.0,
    fees_paid_usd   DECIMAL(18, 2) NOT NULL DEFAULT 0.0,
    slippage_paid_usd DECIMAL(18, 2) NOT NULL DEFAULT 0.0,
    starting_equity DECIMAL(18, 2) NOT NULL DEFAULT 0.0,
    ending_equity   DECIMAL(18, 2) NOT NULL DEFAULT 0.0,
    regime          VARCHAR(16) NOT NULL DEFAULT 'neutral',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_daily_summary_date ON daily_summary(date);

-- ══════════════════════════════════════════════════════════════════════════════
-- 6. INSTRUMENT META — Per-instrument configuration and cached data.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS instrument_meta (
    id              BIGSERIAL PRIMARY KEY,
    symbol          VARCHAR(20) NOT NULL UNIQUE,
    asset_class     VARCHAR(16) NOT NULL,
    strategy_name   VARCHAR(64) NOT NULL,
    last_signal_score INTEGER,
    last_signal_time TIMESTAMPTZ,
    extra           JSONB NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_instrument_meta_symbol ON instrument_meta(symbol);

-- ══════════════════════════════════════════════════════════════════════════════
-- 7. TRADE MEMORIES — Post-trade reflection memory. Append-only.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS trade_memories (
    id          BIGSERIAL PRIMARY KEY,
    situation   TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    lesson      TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_trade_memories_created ON trade_memories(created_at);

-- ══════════════════════════════════════════════════════════════════════════════
-- 8. OPTIONS POSITIONS — Open options position. Mutable until closed.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS options_positions (
    id              BIGSERIAL PRIMARY KEY,
    position_id     UUID NOT NULL UNIQUE,
    underlying      VARCHAR(20) NOT NULL,
    contract_symbol VARCHAR(64) NOT NULL UNIQUE,
    option_type     VARCHAR(8) NOT NULL,
    qty             INTEGER NOT NULL,
    entry_price     DECIMAL(18, 8) NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    expiration      DATE NOT NULL,
    high_water_price DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    stop_loss_price DECIMAL(18, 8),
    take_profit_price DECIMAL(18, 8),
    dte_at_entry    INTEGER NOT NULL,
    strategy_name   VARCHAR(64) NOT NULL,
    broker_order_id VARCHAR(64),
    status          VARCHAR(12) NOT NULL DEFAULT 'open',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_options_positions_underlying ON options_positions(underlying);
CREATE INDEX idx_options_positions_status ON options_positions(status);
CREATE INDEX idx_options_positions_open ON options_positions(status) WHERE status = 'open';

-- ══════════════════════════════════════════════════════════════════════════════
-- 9. OPTIONS TRADES — Completed options trade record. Append-only.
-- ══════════════════════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS options_trades (
    id              BIGSERIAL PRIMARY KEY,
    trade_id        UUID NOT NULL UNIQUE,
    underlying      VARCHAR(20) NOT NULL,
    contract_symbol VARCHAR(64) NOT NULL,
    option_type     VARCHAR(8) NOT NULL,
    qty             INTEGER NOT NULL,
    entry_price     DECIMAL(18, 8) NOT NULL,
    exit_price      DECIMAL(18, 8) NOT NULL,
    realized_pnl    DECIMAL(18, 8) NOT NULL,
    gross_realized_pnl DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    fees_paid       DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    slippage_paid   DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    strategy_name   VARCHAR(64) NOT NULL,
    exit_trigger    VARCHAR(32) NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ NOT NULL,
    hold_hours      DECIMAL(10, 2) NOT NULL,
    broker_order_id VARCHAR(64),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_options_trades_underlying ON options_trades(underlying);
CREATE INDEX idx_options_trades_exit_time ON options_trades(exit_time);

-- ══════════════════════════════════════════════════════════════════════════════
-- TRIGGERS — Auto-update timestamps
-- ══════════════════════════════════════════════════════════════════════════════
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER positions_updated_at
    BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER instrument_meta_updated_at
    BEFORE UPDATE ON instrument_meta
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER options_positions_updated_at
    BEFORE UPDATE ON options_positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ══════════════════════════════════════════════════════════════════════════════
-- ROW LEVEL SECURITY — Enable for all tables (service role bypasses)
-- ══════════════════════════════════════════════════════════════════════════════
-- Note: RLS is enabled but with permissive policies for service role.
-- The trading bot uses the service role key, which bypasses RLS.

ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE signal_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_summary ENABLE ROW LEVEL SECURITY;
ALTER TABLE instrument_meta ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE options_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE options_trades ENABLE ROW LEVEL SECURITY;

-- Service role policy (full access for the trading bot)
CREATE POLICY "Service role has full access on audit_events" ON audit_events
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role has full access on trades" ON trades
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role has full access on positions" ON positions
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role has full access on signal_log" ON signal_log
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role has full access on daily_summary" ON daily_summary
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role has full access on instrument_meta" ON instrument_meta
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role has full access on trade_memories" ON trade_memories
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role has full access on options_positions" ON options_positions
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Service role has full access on options_trades" ON options_trades
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Authenticated users denied on audit_events" ON audit_events
    FOR ALL TO authenticated USING (false) WITH CHECK (false);
CREATE POLICY "Authenticated users denied on trades" ON trades
    FOR ALL TO authenticated USING (false) WITH CHECK (false);
CREATE POLICY "Authenticated users denied on positions" ON positions
    FOR ALL TO authenticated USING (false) WITH CHECK (false);
CREATE POLICY "Authenticated users denied on signal_log" ON signal_log
    FOR ALL TO authenticated USING (false) WITH CHECK (false);
CREATE POLICY "Authenticated users denied on daily_summary" ON daily_summary
    FOR ALL TO authenticated USING (false) WITH CHECK (false);
CREATE POLICY "Authenticated users denied on instrument_meta" ON instrument_meta
    FOR ALL TO authenticated USING (false) WITH CHECK (false);
CREATE POLICY "Authenticated users denied on trade_memories" ON trade_memories
    FOR ALL TO authenticated USING (false) WITH CHECK (false);
CREATE POLICY "Authenticated users denied on options_positions" ON options_positions
    FOR ALL TO authenticated USING (false) WITH CHECK (false);
CREATE POLICY "Authenticated users denied on options_trades" ON options_trades
    FOR ALL TO authenticated USING (false) WITH CHECK (false);

-- ══════════════════════════════════════════════════════════════════════════════
-- DATA RETENTION — Auto-purge old signal_log and audit_events (90 days)
-- ══════════════════════════════════════════════════════════════════════════════
-- This function is called daily via pg_cron (set up in Supabase dashboard)
CREATE OR REPLACE FUNCTION purge_old_records()
RETURNS void AS $$
DECLARE
    signal_count INTEGER;
    audit_count INTEGER;
BEGIN
    -- Purge signal_log older than 90 days
    DELETE FROM signal_log 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS signal_count = ROW_COUNT;
    
    -- Purge audit_events older than 90 days
    DELETE FROM audit_events 
    WHERE created_at < NOW() - INTERVAL '90 days';
    GET DIAGNOSTICS audit_count = ROW_COUNT;
    
    -- Log the cleanup
    IF signal_count > 0 OR audit_count > 0 THEN
        RAISE NOTICE 'Data retention: purged % signal_log rows and % audit_events rows', 
            signal_count, audit_count;
    END IF;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION purge_old_records() TO service_role;

-- ══════════════════════════════════════════════════════════════════════════════
-- HELPFUL VIEWS
-- ══════════════════════════════════════════════════════════════════════════════

-- Today's signals summary
CREATE OR REPLACE VIEW v_today_signals AS
SELECT 
    symbol,
    side,
    score,
    threshold,
    fired,
    regime,
    strategy_name,
    timestamp
FROM signal_log
WHERE timestamp >= CURRENT_DATE
ORDER BY timestamp DESC;

-- Open positions with unrealized metrics
CREATE OR REPLACE VIEW v_open_positions AS
SELECT 
    position_id,
    symbol,
    asset_class,
    qty,
    entry_price,
    high_water_price,
    trailing_stop_price,
    trailing_active,
    stop_loss_price,
    profit_target_price,
    strategy_name,
    entry_time,
    EXTRACT(EPOCH FROM (NOW() - entry_time)) / 3600 AS hold_hours
FROM positions
WHERE status = 'open'
ORDER BY entry_time DESC;

-- Recent trades (last 30 days)
CREATE OR REPLACE VIEW v_recent_trades AS
SELECT 
    trade_id,
    symbol,
    side,
    qty,
    entry_price,
    exit_price,
    realized_pnl,
    strategy_name,
    exit_trigger,
    entry_time,
    exit_time,
    hold_hours
FROM trades
WHERE exit_time >= NOW() - INTERVAL '30 days'
ORDER BY exit_time DESC;

-- Performance by strategy
CREATE OR REPLACE VIEW v_strategy_performance AS
SELECT 
    strategy_name,
    COUNT(*) AS total_trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) AS losses,
    ROUND(100.0 * SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS win_rate_pct,
    SUM(realized_pnl) AS total_net_pnl,
    SUM(gross_realized_pnl) AS total_gross_pnl,
    SUM(fees_paid) AS total_fees_paid,
    SUM(slippage_paid) AS total_slippage_paid,
    AVG(realized_pnl) AS avg_net_pnl,
    AVG(hold_hours) AS avg_hold_hours
FROM trades
GROUP BY strategy_name
ORDER BY total_net_pnl DESC;

COMMENT ON TABLE audit_events IS 'Immutable audit log. Append-only. Never update or delete.';
COMMENT ON TABLE trades IS 'Completed trades with P&L. Append-only.';
COMMENT ON TABLE positions IS 'Open positions with trailing-stop state. Mutable.';
COMMENT ON TABLE signal_log IS 'Every signal scoring result. Append-only. Auto-purged after 90 days.';
COMMENT ON TABLE daily_summary IS 'Daily trading summary. Upserted once per day.';
COMMENT ON TABLE trade_memories IS 'BM25 post-trade reflection memories. Append-only.';
