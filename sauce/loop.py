"""
loop.py — Sauce single-cycle trading engine.

Cron fires run_loop.sh every N minutes — each invocation runs ONE cycle and exits.
No internal while-loop or sleep. Cron handles scheduling.

Cycle:
  1. Morning brief (once/day, cached in DB) → regime classification via Claude
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
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime

from sauce.adapters.broker import BrokerError, get_account, get_positions, place_order
from sauce.adapters.market_data import MarketDataError, get_history, get_quote
from sauce.core.config import get_settings
from sauce.core.schemas import Indicators, Order
from sauce.db import (
    close_position,
    get_daily_regime,
    load_open_positions,
    log_signal,
    log_trade,
    save_position,
    update_position,
    upsert_daily_stats,
)
from sauce.exit_monitor import evaluate_exit
from sauce.indicators.core import compute_all
from sauce.morning_brief import get_regime
from sauce.risk import check_risk
from sauce.strategies.crypto_momentum import CryptoMomentumReversion
from sauce.strategies.equity_momentum import EquityMomentum
from sauce.strategy import Position, Strategy, get_tier_params

logger = logging.getLogger(__name__)


# ── Strategies ────────────────────────────────────────────────────────────────

STRATEGIES: list[Strategy] = [CryptoMomentumReversion(), EquityMomentum()]


def _is_crypto(symbol: str) -> bool:
    """Detect if symbol is crypto (contains '/' like BTC/USD)."""
    return "/" in symbol


# ── Helpers ───────────────────────────────────────────────────────────────────


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _hour_et() -> int:
    """Current hour in US/Eastern (approximate: UTC−5 for EST, UTC−4 for EDT)."""
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).hour


def _fetch_indicators(symbol: str, is_crypto: bool) -> Indicators | None:
    """Fetch history and compute indicators for a symbol."""
    bars = 100  # enough for SMA50 + MACD
    timeframe = "15Min" if is_crypto else "1Day"
    df = get_history(symbol, timeframe=timeframe, bars=bars)
    if df is None or df.empty:
        return None
    return compute_all(df, is_crypto=is_crypto)


def _gather_brief_data() -> dict[str, float]:
    """Fetch real market data for the morning brief.

    Returns dict with keys: btc_change, eth_change, spy_change, vix, btc_rsi.
    Each value has a sensible fallback if its specific API call fails.
    """
    result: dict[str, float] = {
        "btc_change": 0.0,
        "eth_change": 0.0,
        "spy_change": 0.0,
        "vix": 0.0,
        "btc_rsi": 50.0,
    }

    # BTC 24h % change
    try:
        df = get_history("BTC/USD", timeframe="1Hour", bars=24)
        if df is not None and len(df) >= 2:
            first_close = float(df["close"].iloc[0])
            last_close = float(df["close"].iloc[-1])
            if first_close > 0:
                result["btc_change"] = (last_close - first_close) / first_close
    except MarketDataError:
        logger.warning("Failed to fetch BTC history for morning brief")

    # ETH 24h % change
    try:
        df = get_history("ETH/USD", timeframe="1Hour", bars=24)
        if df is not None and len(df) >= 2:
            first_close = float(df["close"].iloc[0])
            last_close = float(df["close"].iloc[-1])
            if first_close > 0:
                result["eth_change"] = (last_close - first_close) / first_close
    except MarketDataError:
        logger.warning("Failed to fetch ETH history for morning brief")

    # SPY daily % change (equity — may fail outside market hours, that's okay)
    try:
        df = get_history("SPY", timeframe="1Day", bars=2)
        if df is not None and len(df) >= 2:
            prev_close = float(df["close"].iloc[-2])
            last_close = float(df["close"].iloc[-1])
            if prev_close > 0:
                result["spy_change"] = (last_close - prev_close) / prev_close
    except MarketDataError:
        logger.warning("Failed to fetch SPY history for morning brief")

    # VIX — use CBOE VIX ETN (VIXY) mid-price as proxy if direct VIX unavailable
    try:
        quote = get_quote("VIXY")
        if quote and quote.mid > 0:
            result["vix"] = quote.mid
    except MarketDataError:
        logger.warning("Failed to fetch VIX proxy for morning brief")

    # BTC RSI(14) from hourly bars
    try:
        indicators = _fetch_indicators("BTC/USD", is_crypto=True)
        if indicators and indicators.rsi_14 is not None:
            result["btc_rsi"] = indicators.rsi_14
    except MarketDataError:
        logger.warning("Failed to fetch BTC RSI for morning brief")

    return result


# ── Entry Scan ────────────────────────────────────────────────────────────────


def _scan_entries(regime: str, account: dict[str, str], open_positions: list[Position]) -> None:
    """Score each strategy instrument for entry signals."""
    get_settings()
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
            if instrument in open_symbols:
                continue  # already have a position
            if not strategy.eligible(instrument, regime):
                continue

            try:
                is_crypto = _is_crypto(instrument)
                indicators = _fetch_indicators(instrument, is_crypto=is_crypto)
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
                order_value = min(equity * tier.max_position_pct, buying_power * 0.95)
                verdict = check_risk(
                    daily_pnl=daily_pnl,
                    equity=equity,
                    open_position_count=len(open_positions),
                    buying_power=buying_power,
                    order_value=order_value,
                    daily_loss_limit=tier.daily_loss_limit,
                    max_concurrent=tier.max_concurrent,
                )
                if not verdict.passed:
                    logger.info("Risk gate blocked %s: %s", instrument, verdict.reason)
                    continue

                # Skip orders below broker minimum ($10)
                if order_value < 10.0:
                    logger.info("Skipping %s: order value $%.2f below $10 minimum", instrument, order_value)
                    continue

                # Build and place order
                account_with_ask = {**account, "_ask": str(ask)}
                order = strategy.build_order(signal, account_with_ask, tier)
                broker_result = place_order(order)
                broker_order_id = getattr(broker_result, "id", None)

                # Use broker's filled_qty when available (accounts for
                # fees / fractional rounding), otherwise fall back to
                # the qty we requested.
                filled_qty_raw = broker_result.get("filled_qty") if isinstance(broker_result, dict) else getattr(broker_result, "filled_qty", None)
                filled_qty = float(str(filled_qty_raw)) if filled_qty_raw and float(str(filled_qty_raw)) > 0 else order.qty

                filled_price_raw = broker_result.get("filled_avg_price") if isinstance(broker_result, dict) else getattr(broker_result, "filled_avg_price", None)
                filled_price = float(str(filled_price_raw)) if filled_price_raw and float(str(filled_price_raw)) > 0 else (order.limit_price or ask)

                # Track position locally
                position = Position(
                    symbol=instrument,
                    asset_class="crypto" if is_crypto else "equity",
                    qty=filled_qty,
                    entry_price=filled_price,
                    high_water_price=filled_price,
                    entry_time=datetime.now(UTC),
                    broker_order_id=str(broker_order_id) if broker_order_id else None,
                    strategy_name=strategy.name,
                    stop_loss_price=order.stop_loss_price or 0.0,
                    profit_target_price=order.take_profit_price or 0.0,
                )
                save_position(position)
                open_positions.append(position)
                upsert_daily_stats(today, orders_placed=1)

                # Deduct order cost so subsequent iterations see reduced buying power
                order_cost = filled_qty * filled_price
                buying_power = max(buying_power - order_cost, 0.0)
                account["buying_power"] = str(buying_power)

                logger.info(
                    "ENTRY %s: score=%d threshold=%d qty=%.4f (filled=%.4f) price=%.2f",
                    instrument,
                    signal.score,
                    signal.threshold,
                    order.qty,
                    filled_qty,
                    filled_price,
                )

            except BrokerError as exc:
                logger.error("Broker error for %s: %s", instrument, exc)
            except Exception as exc:
                logger.error("Unexpected error scanning %s: %s", instrument, exc)


# ── Exit Scan ─────────────────────────────────────────────────────────────────


def _scan_exits(open_positions: list[Position], equity: float, regime: str) -> None:
    """Check exit conditions for each open position."""
    today = _today()
    tier = get_tier_params(equity)

    # Fetch broker positions once to get actual qty (may differ from DB due to fees)
    broker_qty_map: dict[str, float] = {}
    try:
        broker_positions = get_positions()
        for bp in broker_positions:
            sym = str(bp.get("symbol", ""))
            qty_val = bp.get("qty") or bp.get("qty_available") or 0
            broker_qty_map[sym] = float(str(qty_val))
    except BrokerError as exc:
        logger.warning("Could not fetch broker positions for exit scan: %s", exc)

    for position in open_positions:
        try:
            quote = get_quote(position.symbol)
            if quote is None:
                continue
            current_price = float(quote.mid) if hasattr(quote, "mid") else 0.0

            # Fetch current indicators for exit checks
            is_crypto = _is_crypto(position.symbol)
            indicators = _fetch_indicators(position.symbol, is_crypto=is_crypto)
            rsi_14 = indicators.rsi_14 if indicators else None
            atr_14 = indicators.atr_14 if indicators else None

            # Get exit plan from strategy
            strategy = _find_strategy(position.strategy_name)
            if strategy is None:
                continue
            plan = strategy.build_exit_plan(position, tier)

            exit_signal, updated_pos = evaluate_exit(
                position,
                plan,
                current_price,
                rsi_14,
                atr_14=atr_14,
                regime=regime,
            )

            # Always persist trailing state updates
            if (
                updated_pos.trailing_active != position.trailing_active
                or updated_pos.high_water_price != position.high_water_price
            ):
                update_position(updated_pos)

            if exit_signal is None:
                continue

            # Use broker's actual qty (accounts for fees/rounding),
            # fall back to DB qty if broker lookup unavailable.
            broker_symbol = position.symbol.replace("/", "")
            sell_qty = broker_qty_map.get(broker_symbol, position.qty)
            if sell_qty <= 0:
                logger.warning(
                    "Broker reports 0 qty for %s, skipping sell", position.symbol
                )
                continue
            if sell_qty != position.qty:
                logger.info(
                    "Qty reconcile %s: DB=%.10f broker=%.10f (diff=%.10f)",
                    position.symbol,
                    position.qty,
                    sell_qty,
                    position.qty - sell_qty,
                )

            # Place sell order
            sell_order = Order(
                symbol=position.symbol,
                side="sell",
                qty=sell_qty,
                order_type="market",
                time_in_force="gtc",
                as_of=datetime.now(UTC),
                prompt_version="v2",
                source="execution",
            )
            place_order(sell_order)
            log_trade(position, current_price, exit_signal.trigger)
            close_position(position.id)
            upsert_daily_stats(today, trades_closed=1)
            logger.info(
                "EXIT %s: trigger=%s price=%.4f reason=%s",
                position.symbol,
                exit_signal.trigger,
                current_price,
                exit_signal.reason,
            )

        except BrokerError as exc:
            logger.error("Broker error exiting %s: %s", position.symbol, exc)
        except Exception as exc:
            logger.error("Unexpected error on exit scan for %s: %s", position.symbol, exc)


def _find_strategy(name: str) -> Strategy | None:
    """Find a registered strategy by name."""
    for s in STRATEGIES:
        if s.name == name:
            return s
    return None


# ── Main Cycle ─────────────────────────────────────────────────────────────


async def run_cycle() -> None:
    """Run a single trading cycle. Called by cron — no internal loop."""
    cycle_id = str(uuid.uuid4())[:8]
    today = _today()
    regime = "neutral"

    try:
        # Morning brief: check DB cache first (avoids duplicate LLM calls
        # if cron fires multiple times before the first cycle finishes).
        cached_regime = get_daily_regime(today)
        if cached_regime:
            regime = cached_regime
        elif _hour_et() >= 7:
            brief_data = _gather_brief_data()
            regime = await get_regime(
                btc_change=brief_data["btc_change"],
                eth_change=brief_data["eth_change"],
                spy_change=brief_data["spy_change"],
                vix=brief_data["vix"],
                btc_rsi=brief_data["btc_rsi"],
                loop_id=cycle_id,
            )
            upsert_daily_stats(today, regime=regime)
            logger.info("Morning brief: regime=%s", regime)

        # Account state
        account = get_account()
        equity = float(account.get("equity", "0"))
        upsert_daily_stats(today, loop_runs=1, ending_equity=equity)

        if equity <= 0:
            logger.error("Account equity is $0 — skipping cycle")
            return

        # Load positions
        open_positions = load_open_positions()

        # Entry scan
        _scan_entries(regime, account, open_positions)

        # Exit scan
        if open_positions:
            _scan_exits(open_positions, equity, regime)

        logger.info(
            "Cycle %s complete — equity=$%.2f, positions=%d, regime=%s",
            cycle_id,
            equity,
            len(open_positions),
            regime,
        )

    except Exception as exc:
        logger.error("Cycle %s failed: %s", cycle_id, exc, exc_info=True)


def main() -> None:
    """CLI entry point — runs one cycle then exits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_cycle())


if __name__ == "__main__":
    main()
