-- Harden RLS policies for existing deployments.
-- The initial migration now scopes policies correctly for fresh installs, but
-- deployed databases need an explicit migration to replace permissive rules.

DROP POLICY IF EXISTS "Service role has full access on audit_events" ON audit_events;
DROP POLICY IF EXISTS "Service role has full access on trades" ON trades;
DROP POLICY IF EXISTS "Service role has full access on positions" ON positions;
DROP POLICY IF EXISTS "Service role has full access on signal_log" ON signal_log;
DROP POLICY IF EXISTS "Service role has full access on daily_summary" ON daily_summary;
DROP POLICY IF EXISTS "Service role has full access on instrument_meta" ON instrument_meta;
DROP POLICY IF EXISTS "Service role has full access on trade_memories" ON trade_memories;
DROP POLICY IF EXISTS "Service role has full access on options_positions" ON options_positions;
DROP POLICY IF EXISTS "Service role has full access on options_trades" ON options_trades;

DROP POLICY IF EXISTS "Authenticated users denied on audit_events" ON audit_events;
DROP POLICY IF EXISTS "Authenticated users denied on trades" ON trades;
DROP POLICY IF EXISTS "Authenticated users denied on positions" ON positions;
DROP POLICY IF EXISTS "Authenticated users denied on signal_log" ON signal_log;
DROP POLICY IF EXISTS "Authenticated users denied on daily_summary" ON daily_summary;
DROP POLICY IF EXISTS "Authenticated users denied on instrument_meta" ON instrument_meta;
DROP POLICY IF EXISTS "Authenticated users denied on trade_memories" ON trade_memories;
DROP POLICY IF EXISTS "Authenticated users denied on options_positions" ON options_positions;
DROP POLICY IF EXISTS "Authenticated users denied on options_trades" ON options_trades;

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