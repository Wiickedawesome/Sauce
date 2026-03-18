"""
loop.py — Sauce main trading loop.

5-minute cadence, deterministic signal scoring, no per-trade LLM calls.

Loop cycle:
  1. Morning brief (once/day) → regime classification via Claude
  2. Fetch account state + open positions from broker
  3. For each strategy instrument:
     a. Fetch indicators (history → compute_all)
     b. Score signal (deterministic, 0–100)
     c. Risk gate (3 rules: daily P&L, position count, buying power)
     d. Place order if signal fires and risk passes
  4. For each open position:
     a. Fetch current price + RSI
     b. Evaluate 7 exit conditions
     c. Place sell order if exit triggers
  5. Update daily stats
  6. Sleep 5 minutes → repeat
"""

from __future__ import annotations

import asyncio
import logging
import signal
import uuid
from datetime import datetime, timezone

from sauce.adapters.broker import BrokerError, get_account, get_positions, place_order
from sauce.adapters.market_data import get_history, get_quote
from sauce.core.config import get_settings
from sauce.core.schemas import Order
from sauce.exit_monitor import evaluate_exit
from sauce.indicators.core import compute_all
from sauce.morning_brief import get_regime
from sauce.risk import check_risk
from sauce.strategies.crypto_momentum import CryptoMomentumReversion
from sauce.strategy import Position, get_tier_params
from sauce.db import (
    close_position,
    load_open_positions,
    log_signal,
    log_trade,
    save_position,
    update_position,
    upsert_daily_stats,
)

logger = logging.getLogger(__name__)

# Graceful shutdown
_shutdown = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _shutdown
    _shutdown = True
    logger.info("Shutdown requested (signal %d)", signum)


# ── Strategies ────────────────────────────────────────────────────────────────

STRATEGIES = [CryptoMomentumReversion()]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _hour_et() -> int:
    """Current hour in US/Eastern (approximate: UTC−5 for EST, UTC−4 for EDT)."""
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York")).hour


def _fetch_indicators(symbol: str, is_crypto: bool):
    """Fetch history and compute indicators for a symbol."""
    bars = 100  # enough for SMA50 + MACD
    timeframe = "1Hour" if is_crypto else "1Day"
    df = get_history(symbol, timeframe=timeframe, bars=bars)
    if df is None or df.empty:
        return None
    return compute_all(df, is_crypto=is_crypto)


# ── Entry Scan ────────────────────────────────────────────────────────────────


def _scan_entries(regime: str, account: dict, open_positions: list[Position]) -> None:
    """Score each strategy instrument for entry signals."""
    settings = get_settings()
    equity = float(account.get("equity", "0"))
    buying_power = float(account.get("buying_power", "0"))
    tier = get_tier_params(equity)
    today = _today()

    # Calculate daily P&L fraction
    starting = float(account.get("last_equity", equity))
    daily_pnl = (equity - starting) / starting if starting > 0 else 0.0

    open_symbols = {p.symbol for p in open_positions}

    for strategy in STRATEGIES:
        for instrument in strategy.instruments:
            if _shutdown:
                return
            if instrument in open_symbols:
                continue  # already have a position
            if not strategy.eligible(instrument, regime):
                continue

            try:
                indicators = _fetch_indicators(instrument, is_crypto=True)
                if indicators is None:
                    logger.warning("No indicators for %s, skipping", instrument)
                    continue

                # Get current price for BB scoring
                quote = get_quote(instrument)
                if quote is None:
                    continue
                current_price = float(quote.mid) if hasattr(quote, "mid") else 0.0
                ask = float(quote.ask) if hasattr(quote, "ask") else current_price

                signal = strategy.score(indicators, instrument, regime, current_price)
                log_signal(signal)

                if not signal.fired:
                    continue

                # Risk gate
                verdict = check_risk(
                    daily_pnl=daily_pnl,
                    equity=equity,
                    open_position_count=len(open_positions),
                    buying_power=buying_power,
                    order_value=equity * tier.max_position_pct,
                    daily_loss_limit=tier.daily_loss_limit,
                    max_concurrent=tier.max_concurrent,
                )
                if not verdict.passed:
                    logger.info("Risk gate blocked %s: %s", instrument, verdict.reason)
                    continue

                # Build and place order
                account_with_ask = {**account, "_ask": str(ask)}
                order = strategy.build_order(signal, account_with_ask, tier)
                broker_result = place_order(order)
                broker_order_id = getattr(broker_result, "id", None)

                # Track position locally
                position = Position(
                    symbol=instrument,
                    asset_class="crypto",
                    qty=order.qty,
                    entry_price=order.limit_price or ask,
                    high_water_price=order.limit_price or ask,
                    entry_time=datetime.now(timezone.utc),
                    broker_order_id=str(broker_order_id) if broker_order_id else None,
                    strategy_name=strategy.name,
                    stop_loss_price=order.stop_loss_price or 0.0,
                    profit_target_price=order.take_profit_price or 0.0,
                )
                save_position(position)
                upsert_daily_stats(today, orders_placed=1)
                logger.info(
                    "ENTRY %s: score=%d threshold=%d qty=%.4f limit=%.2f",
                    instrument, signal.score, signal.threshold, order.qty, order.limit_price,
                )

            except BrokerError as exc:
                logger.error("Broker error for %s: %s", instrument, exc)
            except Exception as exc:
                logger.error("Unexpected error scanning %s: %s", instrument, exc)


# ── Exit Scan ─────────────────────────────────────────────────────────────────


def _scan_exits(open_positions: list[Position]) -> None:
    """Check exit conditions for each open position."""
    today = _today()

    for position in open_positions:
        if _shutdown:
            return
        try:
            quote = get_quote(position.symbol)
            if quote is None:
                continue
            current_price = float(quote.mid) if hasattr(quote, "mid") else 0.0

            # Fetch current RSI for exhaustion check
            indicators = _fetch_indicators(position.symbol, is_crypto=True)
            rsi_14 = indicators.rsi_14 if indicators else None

            # Get exit plan from strategy
            strategy = _find_strategy(position.strategy_name)
            if strategy is None:
                continue
            tier = get_tier_params(0)  # use seed defaults for exit params
            plan = strategy.build_exit_plan(position, tier)

            exit_signal, updated_pos = evaluate_exit(
                position, plan, current_price, rsi_14,
            )

            # Always persist trailing state updates
            if (updated_pos.trailing_active != position.trailing_active or
                    updated_pos.high_water_price != position.high_water_price):
                update_position(updated_pos)

            if exit_signal is None:
                continue

            # Place sell order
            sell_order = Order(
                symbol=position.symbol,
                side="sell",
                qty=position.qty,
                order_type="market",
                time_in_force="gtc",
                as_of=datetime.now(timezone.utc),
                prompt_version="v2",
                source="execution",
            )
            place_order(sell_order)
            log_trade(position, current_price, exit_signal.trigger)
            close_position(position.id)
            upsert_daily_stats(today, trades_closed=1)
            logger.info(
                "EXIT %s: trigger=%s price=%.4f reason=%s",
                position.symbol, exit_signal.trigger, current_price, exit_signal.reason,
            )

        except BrokerError as exc:
            logger.error("Broker error exiting %s: %s", position.symbol, exc)
        except Exception as exc:
            logger.error("Unexpected error on exit scan for %s: %s", position.symbol, exc)


def _find_strategy(name: str):
    """Find a registered strategy by name."""
    for s in STRATEGIES:
        if s.name == name:
            return s
    return None


# ── Main Loop ─────────────────────────────────────────────────────────────────


async def run_loop() -> None:
    """Run the Sauce trading loop indefinitely."""
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    settings = get_settings()
    interval = getattr(settings, "loop_interval_minutes", 5) * 60

    regime = "neutral"
    last_brief_date = ""

    logger.info("Sauce loop starting — interval=%ds, strategies=%d",
                interval, len(STRATEGIES))

    while not _shutdown:
        cycle_id = str(uuid.uuid4())[:8]
        today = _today()

        try:
            # Morning brief: once per day
            if today != last_brief_date and _hour_et() >= 7:
                regime = await get_regime(
                    btc_change=0.0, eth_change=0.0, spy_change=0.0,
                    vix=20.0, btc_rsi=50.0,
                    loop_id=cycle_id,
                )
                last_brief_date = today
                upsert_daily_stats(today, regime=regime)
                logger.info("Morning brief: regime=%s", regime)

            # Account state
            account = get_account()
            equity = float(account.get("equity", "0"))
            upsert_daily_stats(today, loop_runs=1, ending_equity=equity)

            if equity <= 0:
                logger.error("Account equity is $0 — pausing")
                await asyncio.sleep(interval)
                continue

            # Load positions
            open_positions = load_open_positions()

            # Entry scan
            _scan_entries(regime, account, open_positions)

            # Exit scan
            if open_positions:
                _scan_exits(open_positions)

            logger.info(
                "Cycle %s complete — equity=$%.2f, positions=%d, regime=%s",
                cycle_id, equity, len(open_positions), regime,
            )

        except Exception as exc:
            logger.error("Loop cycle %s failed: %s", cycle_id, exc, exc_info=True)

        await asyncio.sleep(interval)

    logger.info("Sauce loop stopped")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_loop())


if __name__ == "__main__":
    main()
