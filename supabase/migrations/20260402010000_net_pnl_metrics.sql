-- Add net-P&L attribution fields to existing tables and refresh performance views.

ALTER TABLE trades
    ADD COLUMN IF NOT EXISTS gross_realized_pnl DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS fees_paid DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS slippage_paid DECIMAL(18, 8) NOT NULL DEFAULT 0.0;

ALTER TABLE options_trades
    ADD COLUMN IF NOT EXISTS gross_realized_pnl DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS fees_paid DECIMAL(18, 8) NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS slippage_paid DECIMAL(18, 8) NOT NULL DEFAULT 0.0;

ALTER TABLE daily_summary
    ADD COLUMN IF NOT EXISTS gross_realized_pnl_usd DECIMAL(18, 2) NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS fees_paid_usd DECIMAL(18, 2) NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS slippage_paid_usd DECIMAL(18, 2) NOT NULL DEFAULT 0.0;

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