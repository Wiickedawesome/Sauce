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
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sauce.adapters.broker import (
    BrokerError,
    cancel_stale_orders,
    get_account,
    get_option_positions,
    get_order_by_id,
    get_positions,
    get_recent_orders,
    place_option_order,
    place_order,
)
from sauce.adapters.db import log_event
from sauce.adapters.market_data import (
    MarketDataError,
    get_history,
    get_option_chain,
    get_option_quotes,
    get_quote,
    get_snapshot_candidates,
    get_universe_snapshot,
)
from sauce.analyst import analyst_committee
from sauce.core.config import get_settings
from sauce.core.options_schemas import OptionsOrder, OptionsPosition
from sauce.core.schemas import AuditEvent, Indicators, Order, PriceReference
from sauce.db import (
    close_option_position,
    close_position,
    get_daily_regime,
    load_all_memories,
    load_open_option_positions,
    load_open_positions,
    log_option_trade,
    log_signal,
    log_trade,
    save_memory,
    save_option_position,
    save_position,
    update_option_position,
    update_position,
    upsert_daily_stats,
)
from sauce.exit_monitor import evaluate_exit
from sauce.indicators.core import compute_all
from sauce.memory import (
    MemoryEntry,
    TradeMemory,
    build_outcome_description,
    build_situation_description,
)
from sauce.morning_brief import get_regime
from sauce.options_safety import check_options_position, validate_options_order
from sauce.reflection import reflect_on_trade
from sauce.risk import check_risk
from sauce.strategies.crypto_momentum import CryptoMomentumReversion
from sauce.strategies.equity_momentum import EquityMomentum
from sauce.strategies.options_momentum import OptionsMomentum
from sauce.strategy import Position, Strategy, TierParams, get_tier_params

logger = logging.getLogger(__name__)

_PENDING_ORDER_STATUSES = {
    "accepted",
    "accepted_for_bidding",
    "held",
    "new",
    "partially_filled",
    "pending_new",
    "pending_replace",
}
_FILLED_ORDER_STATUSES = {"filled", "partially_filled"}
_TERMINAL_ORDER_STATUSES = {"filled", "canceled", "expired", "rejected", "suspended"}
_ENTRY_STAGE_TARGETS = (0.35, 0.65, 1.0)
_SCALE_IN_CONFIRM_GAINS = (0.0, 0.0075, 0.015)
_CRYPTO_MIN_NOTIONAL = 10.0
_EXIT_POLL_ATTEMPTS = 4
_EXIT_POLL_INTERVAL = 1.0  # seconds between polls


# ── Strategies ────────────────────────────────────────────────────────────────

STRATEGIES: list[Strategy] = [CryptoMomentumReversion(), EquityMomentum()]
OPTIONS_STRATEGY = OptionsMomentum()


@dataclass(frozen=True, slots=True)
class SupervisorDecision:
    action: str
    reason: str


@dataclass(frozen=True, slots=True)
class EntrySizingPlan:
    order_value_usd: float
    current_position_value_usd: float
    target_stage: int
    current_gain_pct: float
    is_scale_in: bool


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


def _audit_event(loop_id: str, event_type: str, payload: dict[str, object], symbol: str | None = None) -> None:
    settings = get_settings()
    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type=event_type,
            symbol=symbol,
            payload=payload,
            timestamp=datetime.now(UTC),
            prompt_version=settings.prompt_version,
        )
    )


def _is_quote_fresh(quote: PriceReference, ttl_seconds: int, now: datetime | None = None) -> bool:
    ref_now = now or datetime.now(UTC)
    quote_time = quote.as_of if quote.as_of.tzinfo is not None else quote.as_of.replace(tzinfo=UTC)
    return (ref_now - quote_time).total_seconds() <= ttl_seconds


def _pending_order_symbols(recent_orders: list[dict[str, object]]) -> set[str]:
    pending_symbols: set[str] = set()
    for order in recent_orders:
        status = str(order.get("status", "")).strip().lower()
        symbol = str(order.get("symbol", "")).strip().upper()
        if status in _PENDING_ORDER_STATUSES and symbol:
            pending_symbols.add(symbol)
    return pending_symbols


def _should_persist_position(status: str, filled_qty: float) -> bool:
    return status in _FILLED_ORDER_STATUSES and filled_qty > 0


def _symbol_aliases(symbol: str) -> tuple[str, ...]:
    upper_symbol = symbol.strip().upper()
    normalized_symbol = upper_symbol.replace("/", "")
    if normalized_symbol == upper_symbol:
        return (upper_symbol,)
    return (upper_symbol, normalized_symbol)


def _price_precision(symbol: str) -> int:
    return 8 if _is_crypto(symbol) else 2


def _round_position_price(value: float, symbol: str) -> float:
    return round(value, _price_precision(symbol))


def _broker_result_field(result: dict[str, object] | object, field_name: str) -> object | None:
    if isinstance(result, dict):
        return result.get(field_name)
    return getattr(result, field_name, None)


def _parse_positive_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _extract_fill_details(result: dict[str, object] | object) -> tuple[str, float, float | None]:
    status = str(_broker_result_field(result, "status") or "").lower()
    filled_qty = _parse_positive_float(_broker_result_field(result, "filled_qty")) or 0.0
    filled_price = _parse_positive_float(_broker_result_field(result, "filled_avg_price"))
    return status, filled_qty, filled_price


def _poll_order_fill(
    order_id: str,
    symbol: str,
    loop_id: str,
) -> tuple[str, float, float | None]:
    """Poll broker for order fill status when initial response is PENDING_NEW.

    Makes up to _EXIT_POLL_ATTEMPTS checks with _EXIT_POLL_INTERVAL second
    delays. Returns the final (status, filled_qty, filled_price) tuple.
    """
    for attempt in range(1, _EXIT_POLL_ATTEMPTS + 1):
        time.sleep(_EXIT_POLL_INTERVAL)
        try:
            refreshed = get_order_by_id(order_id, loop_id)
            status, filled_qty, filled_price = _extract_fill_details(refreshed)
            logger.info(
                "Poll %d/%d for %s order %s: status=%s filled=%.4f",
                attempt,
                _EXIT_POLL_ATTEMPTS,
                symbol,
                order_id,
                status,
                filled_qty,
            )
            if status in _TERMINAL_ORDER_STATUSES:
                return status, filled_qty, filled_price
        except BrokerError as exc:
            logger.warning("Poll %d/%d for %s failed: %s", attempt, _EXIT_POLL_ATTEMPTS, symbol, exc)
    return status, filled_qty, filled_price


def _find_broker_position(symbol: str, broker_positions: list[dict[str, object]]) -> dict[str, object] | None:
    wanted_aliases = set(_symbol_aliases(symbol))
    for broker_position in broker_positions:
        broker_symbol = str(broker_position.get("symbol", ""))
        if wanted_aliases.intersection(_symbol_aliases(broker_symbol)):
            return broker_position
    return None


def _current_position_cost_basis_value(
    symbol: str,
    position: Position | None,
    broker_positions: list[dict[str, object]],
    fallback_price: float,
) -> float:
    if position is None:
        return 0.0

    broker_position = _find_broker_position(symbol, broker_positions)
    if broker_position is not None:
        broker_qty = _parse_positive_float(
            broker_position.get("qty") or broker_position.get("qty_available")
        )
        broker_entry_price = _parse_positive_float(broker_position.get("avg_entry_price"))
        if broker_qty is not None:
            anchor_price = broker_entry_price or position.entry_price or fallback_price
            return broker_qty * anchor_price

    anchor_price = position.entry_price or fallback_price
    return position.qty * anchor_price


def _plan_entry_sizing(
    symbol: str,
    position: Position | None,
    broker_positions: list[dict[str, object]],
    current_price: float,
    equity: float,
    tier: TierParams,
    analyst_size_fraction: float,
) -> EntrySizingPlan | None:
    analyst_fraction = max(0.0, min(1.0, analyst_size_fraction))
    max_position_value = equity * tier.max_position_pct
    if analyst_fraction <= 0.0 or max_position_value <= 0.0:
        return None

    current_position_value = _current_position_cost_basis_value(
        symbol,
        position,
        broker_positions,
        current_price,
    )
    current_ratio = current_position_value / max_position_value if max_position_value > 0 else 0.0
    next_stage_idx = next(
        (idx for idx, target in enumerate(_ENTRY_STAGE_TARGETS) if current_ratio + 1e-6 < target),
        None,
    )
    if next_stage_idx is None:
        return None

    current_gain_pct = 0.0
    if position is not None:
        if position.trailing_active or position.profit_target_price <= 0:
            return None
        if position.entry_price <= 0:
            return None
        current_gain_pct = (current_price - position.entry_price) / position.entry_price
        required_gain = _SCALE_IN_CONFIRM_GAINS[next_stage_idx]
        if current_gain_pct < required_gain:
            return None

    previous_target = _ENTRY_STAGE_TARGETS[next_stage_idx - 1] if next_stage_idx > 0 else 0.0
    stage_target_value = max_position_value * _ENTRY_STAGE_TARGETS[next_stage_idx]
    stage_band_value = max_position_value * (_ENTRY_STAGE_TARGETS[next_stage_idx] - previous_target)
    remaining_to_stage = max(stage_target_value - current_position_value, 0.0)
    desired_order_value = min(remaining_to_stage, stage_band_value * analyst_fraction)
    if desired_order_value <= 0.0:
        return None

    return EntrySizingPlan(
        order_value_usd=desired_order_value,
        current_position_value_usd=current_position_value,
        target_stage=next_stage_idx + 1,
        current_gain_pct=current_gain_pct,
        is_scale_in=position is not None,
    )


def _planned_exit_qty(
    position: Position,
    total_qty: float,
    current_price: float,
    exit_fraction: float,
) -> float:
    planned_fraction = max(0.0, min(1.0, exit_fraction))
    if planned_fraction >= 1.0:
        return total_qty

    planned_qty = total_qty * planned_fraction
    if position.asset_class == "equity":
        planned_qty = float(int(planned_qty))
    else:
        planned_qty = round(planned_qty, 8)

    if planned_qty <= 0:
        return total_qty

    if position.asset_class == "crypto":
        sell_notional = planned_qty * current_price
        remaining_notional = max(total_qty - planned_qty, 0.0) * current_price
        if sell_notional < _CRYPTO_MIN_NOTIONAL or remaining_notional < _CRYPTO_MIN_NOTIONAL:
            return total_qty

    return min(planned_qty, total_qty)


def _reconcile_entry_fill(
    symbol: str,
    loop_id: str,
    initial_qty: float,
    initial_price: float | None,
) -> tuple[float, float | None]:
    try:
        broker_position = _find_broker_position(symbol, get_positions(loop_id))
    except BrokerError as exc:
        logger.warning("Could not reconcile broker fill for %s: %s", symbol, exc)
        return initial_qty, initial_price

    if broker_position is None:
        return initial_qty, initial_price

    broker_qty = _parse_positive_float(
        broker_position.get("qty") or broker_position.get("qty_available")
    )
    broker_price = _parse_positive_float(broker_position.get("avg_entry_price"))
    return broker_qty or initial_qty, broker_price or initial_price


def _audit_missing_quotes(loop_id: str, stage: str, requested_symbols: list[str], quotes_map: dict[str, PriceReference]) -> None:
    missing_symbols = sorted(set(requested_symbols) - set(quotes_map))
    if not missing_symbols:
        return

    logger.warning(
        "Batch snapshot missing %d/%d symbols during %s: %s",
        len(missing_symbols),
        len(requested_symbols),
        stage,
        ", ".join(missing_symbols),
    )
    _audit_event(
        loop_id,
        "error",
        {
            "stage": stage,
            "error": "batch snapshot missing symbols",
            "requested_symbols": requested_symbols,
            "missing_symbols": missing_symbols,
        },
    )


def _safe_get_universe_snapshot(
    symbols: list[str],
    *,
    loop_id: str,
    stage: str,
    respect_suppression: bool = True,
) -> dict[str, PriceReference]:
    """Fetch a batch quote snapshot without letting a transient outage fail the whole cycle.

    Falls back to individual get_quote() calls for any symbols missing from
    the batch response (e.g. LINK/USD, UNI/USD occasionally absent from
    Alpaca's snapshot endpoint).
    """
    if not symbols:
        return {}

    try:
        result = get_universe_snapshot(symbols, respect_suppression=respect_suppression)
    except MarketDataError as exc:
        logger.error("Batch quote fetch failed during %s: %s", stage, exc)
        _audit_event(
            loop_id,
            "error",
            {
                "stage": stage,
                "error": str(exc),
                "requested_symbols": symbols,
            },
        )
        result = {}

    # Fall back to individual quotes for missing symbols.
    missing = set(symbols) - set(result)
    for sym in missing:
        try:
            result[sym] = get_quote(sym)
        except MarketDataError:
            logger.debug("Individual quote fallback also failed for %s", sym)

    return result


def _safe_get_option_quotes(
    symbols: list[str],
    *,
    loop_id: str,
    stage: str,
) -> dict[str, PriceReference]:
    """Fetch option quotes without letting a batch failure abort the cycle."""
    if not symbols:
        return {}

    try:
        return get_option_quotes(symbols)
    except MarketDataError as exc:
        logger.error("Option quote fetch failed during %s: %s", stage, exc)
        _audit_event(
            loop_id,
            "error",
            {
                "stage": stage,
                "error": str(exc),
                "requested_symbols": symbols,
            },
        )
        return {}


def _pending_option_underlyings(recent_orders: list[dict[str, object]], underlyings: list[str]) -> set[str]:
    pending: set[str] = set()
    ordered_underlyings = sorted({underlying.upper() for underlying in underlyings}, key=len, reverse=True)
    for order in recent_orders:
        status = str(order.get("status", "")).strip().lower()
        symbol = str(order.get("symbol", "")).strip().upper()
        if status not in _PENDING_ORDER_STATUSES or not symbol:
            continue
        for underlying in ordered_underlyings:
            if symbol.startswith(underlying):
                pending.add(underlying)
                break
    return pending


def _broker_position_exposure(broker_positions: list[dict[str, object]]) -> float:
    total = 0.0
    for position in broker_positions:
        market_value = position.get("market_value") or 0
        try:
            total += abs(float(str(market_value)))
        except (TypeError, ValueError):
            continue
    return total


def _supervisor_review(
    symbol: str,
    order: Order,
    snapshot_quote: PriceReference,
    fresh_quote: PriceReference,
    buying_power: float,
    loop_id: str,
) -> SupervisorDecision:
    settings = get_settings()

    if settings.trading_pause:
        decision = SupervisorDecision(action="abort", reason="TRADING_PAUSE enabled")
    elif not _is_quote_fresh(fresh_quote, settings.data_ttl_seconds):
        decision = SupervisorDecision(action="abort", reason="fresh quote is stale")
    elif order.qty <= 0:
        decision = SupervisorDecision(action="abort", reason="order quantity is zero")
    else:
        baseline = snapshot_quote.mid if snapshot_quote.mid > 0 else fresh_quote.mid
        deviation = abs(fresh_quote.mid - baseline) / baseline if baseline > 0 else 1.0
        order_price = order.limit_price or order.stop_price or fresh_quote.ask or fresh_quote.mid
        order_value = order.qty * order_price
        # Crypto is volatile 24/7; allow 2.5x wider price deviation tolerance
        # to survive normal movement during the ~5-10 s LLM analysis window.
        max_dev = settings.max_price_deviation * 2.5 if _is_crypto(symbol) else settings.max_price_deviation
        if deviation > max_dev:
            decision = SupervisorDecision(
                action="abort",
                reason=(
                    f"price deviation {deviation:.2%} exceeds "
                    f"limit {max_dev:.2%}"
                ),
            )
        elif order_value > buying_power:
            decision = SupervisorDecision(
                action="abort",
                reason=f"order value ${order_value:,.2f} exceeds buying power ${buying_power:,.2f}",
            )
        else:
            decision = SupervisorDecision(
                action="execute",
                reason="risk approved and fresh quote within deviation threshold",
            )

    _audit_event(
        loop_id,
        "supervisor_decision",
        {
            "action": decision.action,
            "reason": decision.reason,
            "order_type": order.order_type,
            "qty": order.qty,
            "limit_price": order.limit_price,
            "quote_mid": fresh_quote.mid,
        },
        symbol=symbol,
    )
    return decision


def _supervisor_review_option(
    underlying: str,
    contract_symbol: str,
    order: OptionsOrder,
    snapshot_quote: PriceReference,
    fresh_quote: PriceReference,
    buying_power: float,
    loop_id: str,
) -> SupervisorDecision:
    settings = get_settings()
    baseline = snapshot_quote.mid if snapshot_quote.mid > 0 else fresh_quote.mid
    deviation = abs(fresh_quote.mid - baseline) / baseline if baseline > 0 else 1.0
    order_value = order.qty * order.limit_price * 100

    if settings.trading_pause:
        decision = SupervisorDecision(action="abort", reason="TRADING_PAUSE enabled")
    elif not _is_quote_fresh(fresh_quote, settings.data_ttl_seconds):
        decision = SupervisorDecision(action="abort", reason="fresh option quote is stale")
    elif order.qty <= 0:
        decision = SupervisorDecision(action="abort", reason="options order quantity is zero")
    elif deviation > settings.max_price_deviation:
        decision = SupervisorDecision(
            action="abort",
            reason=f"option price deviation {deviation:.2%} exceeds limit {settings.max_price_deviation:.2%}",
        )
    elif order_value > buying_power:
        decision = SupervisorDecision(
            action="abort",
            reason=f"option premium ${order_value:,.2f} exceeds buying power ${buying_power:,.2f}",
        )
    else:
        decision = SupervisorDecision(action="execute", reason="options preflight checks passed")

    _audit_event(
        loop_id,
        "supervisor_decision",
        {
            "action": decision.action,
            "reason": decision.reason,
            "contract_symbol": contract_symbol,
            "qty": order.qty,
            "limit_price": order.limit_price,
            "quote_mid": fresh_quote.mid,
        },
        symbol=underlying,
    )
    return decision


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


async def _scan_entries(
    regime: str,
    account: dict[str, str],
    open_positions: list[Position],
    broker_positions: list[dict[str, object]],
    recent_orders: list[dict[str, object]],
    trade_memory: TradeMemory,
    loop_id: str = "entry",
) -> None:
    """Score each strategy instrument for entry signals."""
    settings = get_settings()
    equity = float(account.get("equity", "0"))
    buying_power = float(account.get("buying_power", "0"))
    tier = get_tier_params(equity)
    today = _today()

    # Calculate daily P&L fraction
    starting = float(account.get("last_equity", equity))
    daily_pnl = (equity - starting) / starting if starting > 0 else 0.0

    open_positions_by_symbol = {p.symbol: p for p in open_positions}
    pending_symbols = _pending_order_symbols(recent_orders)
    actual_open_position_count = len(broker_positions)
    total_existing_exposure = _broker_position_exposure(broker_positions)

    # ── Pre-filter eligible instruments ──
    eligible_instruments: list[tuple[Strategy, str]] = []
    for strategy in STRATEGIES:
        for instrument in strategy.instruments:
            if instrument in pending_symbols:
                continue
            if not strategy.eligible(instrument, regime):
                continue
            eligible_instruments.append((strategy, instrument))

    if not eligible_instruments:
        logger.info("No eligible instruments this cycle")
        return

    # ── Batch-fetch quotes (1–2 API calls instead of N) ──
    all_symbols = list({inst for _, inst in eligible_instruments})
    snapshot_symbols = get_snapshot_candidates(all_symbols)
    quotes_map = _safe_get_universe_snapshot(
        snapshot_symbols,
        loop_id=loop_id,
        stage="entry_batch_quotes",
    )
    _audit_missing_quotes(loop_id, "entry_batch_quotes", snapshot_symbols, quotes_map)
    logger.info("Batch quotes fetched for %d symbols (%d returned)", len(snapshot_symbols), len(quotes_map))

    for strategy, instrument in eligible_instruments:
        existing_position = open_positions_by_symbol.get(instrument)
        quote = quotes_map.get(instrument)
        if quote is None:
            continue
        if not _is_quote_fresh(quote, settings.data_ttl_seconds):
            _audit_event(
                loop_id,
                "safety_check",
                {
                    "paused": False,
                    "action": "skip_symbol",
                    "reason": "stale quote",
                    "quote_as_of": quote.as_of.isoformat(),
                    "ttl_seconds": settings.data_ttl_seconds,
                },
                symbol=instrument,
            )
            logger.info("Skipping %s: stale snapshot quote", instrument)
            continue
        current_price = float(quote.mid) if hasattr(quote, "mid") else 0.0

        # Skip if price looks stale or zero
        if current_price <= 0:
            continue

        try:
            is_crypto = _is_crypto(instrument)
            indicators = _fetch_indicators(instrument, is_crypto=is_crypto)
            if indicators is None:
                logger.warning("No indicators for %s, skipping", instrument)
                continue

            signal = strategy.score(indicators, instrument, regime, current_price)
            log_signal(signal)

            if not signal.fired:
                continue

            upsert_daily_stats(today, signals_fired=1)

            # ── Analyst committee (2 LLM calls) ──
            situation_desc = build_situation_description(
                symbol=instrument,
                regime=regime,
                score=signal.score,
                threshold=signal.threshold,
                rsi_14=signal.rsi_14,
                macd_hist=signal.macd_hist,
                bb_pct=signal.bb_pct,
                volume_ratio=signal.volume_ratio,
                current_price=current_price,
                strategy_name=strategy.name,
            )
            recalled = trade_memory.recall(situation_desc)
            verdict = await analyst_committee(
                symbol=instrument,
                strategy_name=strategy.name,
                score=signal.score,
                threshold=signal.threshold,
                regime=regime,
                current_price=current_price,
                rsi_14=signal.rsi_14,
                macd_hist=signal.macd_hist,
                bb_pct=signal.bb_pct,
                volume_ratio=signal.volume_ratio,
                memories=recalled,
                loop_id=loop_id,
            )
            if not verdict.approve or verdict.size_fraction <= 0.0:
                logger.info(
                    "ANALYST REJECTED %s: confidence=%d size=%.2f — %s",
                    instrument,
                    verdict.confidence,
                    verdict.size_fraction,
                    verdict.reasoning,
                )
                upsert_daily_stats(today, signals_skipped=1)
                continue

            logger.info(
                "ANALYST APPROVED %s: confidence=%d size=%.2f — %s",
                instrument,
                verdict.confidence,
                verdict.size_fraction,
                verdict.reasoning,
            )

            fresh_quote = get_quote(instrument)
            if not _is_quote_fresh(fresh_quote, settings.data_ttl_seconds):
                _audit_event(
                    loop_id,
                    "supervisor_decision",
                    {
                        "action": "abort",
                        "reason": "fresh quote is stale",
                        "quote_as_of": fresh_quote.as_of.isoformat(),
                    },
                    symbol=instrument,
                )
                logger.info("Supervisor aborted %s: fresh quote is stale", instrument)
                continue

            entry_sizing = _plan_entry_sizing(
                instrument,
                existing_position,
                broker_positions,
                fresh_quote.mid,
                equity,
                tier,
                verdict.size_fraction,
            )
            if entry_sizing is None:
                continue

            if not _is_crypto(instrument) and entry_sizing.order_value_usd < fresh_quote.ask:
                logger.info(
                    "Skipping %s: staged order budget $%.2f cannot afford one share at $%.2f",
                    instrument,
                    entry_sizing.order_value_usd,
                    fresh_quote.ask,
                )
                continue

            account_with_ask = {
                **account,
                "_ask": str(fresh_quote.ask or fresh_quote.mid),
                "_target_order_value": entry_sizing.order_value_usd,
                "_position_size_fraction": verdict.size_fraction,
            }
            order = strategy.build_order(signal, account_with_ask, tier)
            order_value = order.qty * (order.limit_price or order.stop_price or fresh_quote.ask or fresh_quote.mid)

            # Risk gate
            risk_verdict = check_risk(
                daily_pnl=daily_pnl,
                equity=equity,
                open_position_count=(
                    actual_open_position_count if existing_position is None else max(actual_open_position_count - 1, 0)
                ),
                buying_power=buying_power,
                order_value=order_value,
                daily_loss_limit=tier.daily_loss_limit,
                max_concurrent=tier.max_concurrent,
                total_existing_exposure=total_existing_exposure,
                max_portfolio_exposure=settings.max_portfolio_exposure,
            )
            if not risk_verdict.passed:
                _audit_event(
                    loop_id,
                    "supervisor_decision",
                    {
                        "action": "abort",
                        "reason": risk_verdict.reason,
                        "rule": risk_verdict.rule,
                        "order_value": order_value,
                    },
                    symbol=instrument,
                )
                logger.info("Risk gate blocked %s: %s", instrument, risk_verdict.reason)
                continue

            # Skip orders below broker minimum ($10)
            if order_value < 10.0:
                _audit_event(
                    loop_id,
                    "supervisor_decision",
                    {
                        "action": "abort",
                        "reason": f"order value ${order_value:.2f} below $10 minimum",
                        "rule": "min_order_value",
                        "order_value": order_value,
                    },
                    symbol=instrument,
                )
                logger.info("Skipping %s: order value $%.2f below $10 minimum", instrument, order_value)
                continue

            decision = _supervisor_review(
                instrument,
                order,
                quote,
                fresh_quote,
                buying_power,
                loop_id,
            )
            if decision.action != "execute":
                logger.info("Supervisor aborted %s: %s", instrument, decision.reason)
                continue

            broker_result = place_order(order, loop_id)
            broker_order_id = _broker_result_field(broker_result, "id")
            status, filled_qty, filled_price = _extract_fill_details(broker_result)

            if not _should_persist_position(status, filled_qty):
                pending_symbols.add(instrument)
                _audit_event(
                    loop_id,
                    "supervisor_decision",
                    {
                        "action": "pending",
                        "reason": f"order submitted but status={status or 'unknown'}, filled_qty={filled_qty:.4f}",
                        "broker_order_id": str(broker_order_id) if broker_order_id else None,
                    },
                    symbol=instrument,
                )
                logger.info(
                    "Order for %s status=%s filled_qty=%.4f — not persisting position until broker reports a fill",
                    instrument,
                    status or "unknown",
                    filled_qty,
                )
                continue

            fallback_entry_price = order.limit_price or fresh_quote.ask or fresh_quote.mid
            filled_qty, filled_price = _reconcile_entry_fill(
                instrument,
                loop_id,
                filled_qty,
                filled_price,
            )
            if filled_price is None:
                filled_price = fallback_entry_price
                logger.warning(
                    "Filled order for %s missing avg price; using fallback entry price %.4f",
                    instrument,
                    filled_price,
                )
                _audit_event(
                    loop_id,
                    "error",
                    {
                        "stage": "entry_fill_reconciliation",
                        "error": "filled order missing avg entry price",
                        "fallback_entry_price": filled_price,
                    },
                    symbol=instrument,
                )

            # Track position locally
            is_crypto = _is_crypto(instrument)
            if existing_position is None:
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
                open_positions_by_symbol[instrument] = position
                actual_open_position_count += 1
            else:
                existing_position.qty = filled_qty
                existing_position.entry_price = filled_price
                existing_position.high_water_price = max(existing_position.high_water_price, fresh_quote.mid)
                existing_position.trailing_active = False
                existing_position.trailing_stop_price = None
                existing_position.broker_order_id = (
                    str(broker_order_id) if broker_order_id else existing_position.broker_order_id
                )
                existing_position.stop_loss_price = _round_position_price(
                    filled_price * (1 - tier.stop_loss_pct),
                    instrument,
                )
                existing_position.profit_target_price = _round_position_price(
                    filled_price * (1 + tier.profit_target_pct),
                    instrument,
                )
                update_position(existing_position)
                position = existing_position

            upsert_daily_stats(today, orders_placed=1)

            # Deduct order cost so subsequent iterations see reduced buying power
            order_cost = order_value
            buying_power = max(buying_power - order_cost, 0.0)
            account["buying_power"] = str(buying_power)
            total_existing_exposure += order_cost

            logger.info(
                "ENTRY %s: stage=%d scale_in=%s score=%d threshold=%d qty=%.4f (filled=%.4f) price=%.2f",
                instrument,
                entry_sizing.target_stage,
                entry_sizing.is_scale_in,
                signal.score,
                signal.threshold,
                order.qty,
                filled_qty,
                filled_price,
            )

        except BrokerError as exc:
            logger.error("Broker error for %s: %s", instrument, exc)
            _audit_event(
                loop_id,
                "supervisor_decision",
                {"action": "error", "stage": "entry_order", "reason": str(exc)[:500]},
                symbol=instrument,
            )
        except Exception as exc:
            logger.error("Unexpected error scanning %s: %s", instrument, exc)
            _audit_event(loop_id, "supervisor_decision", {"action": "error", "stage": "entry_scan", "reason": str(exc)}, symbol=instrument)


async def _scan_option_entries(
    regime: str,
    account: dict[str, str],
    open_option_positions: list[OptionsPosition],
    broker_positions: list[dict[str, object]],
    recent_orders: list[dict[str, object]],
    trade_memory: TradeMemory,
    loop_id: str = "options-entry",
) -> None:
    """Scan approved underlyings for options entries and place option orders."""
    settings = get_settings()
    if not settings.options_enabled:
        return

    equity = float(account.get("equity", "0"))
    buying_power = float(account.get("buying_power", "0"))
    tier = get_tier_params(equity)
    today = _today()
    starting = float(account.get("last_equity", equity))
    daily_pnl = (equity - starting) / starting if starting > 0 else 0.0

    open_underlyings = {position.underlying for position in open_option_positions}
    pending_underlyings = _pending_option_underlyings(recent_orders, OPTIONS_STRATEGY.instruments)
    actual_open_position_count = len(broker_positions)
    total_existing_exposure = _broker_position_exposure(broker_positions)
    existing_options_value = sum(position.entry_price * position.qty * 100 for position in open_option_positions)

    eligible_underlyings = [
        underlying
        for underlying in OPTIONS_STRATEGY.instruments
        if underlying not in open_underlyings
        and underlying not in pending_underlyings
        and OPTIONS_STRATEGY.eligible(underlying, regime)
    ]
    if not eligible_underlyings:
        return

    snapshot_underlyings = get_snapshot_candidates(eligible_underlyings)
    quotes_map = _safe_get_universe_snapshot(
        snapshot_underlyings,
        loop_id=loop_id,
        stage="options_entry_batch_quotes",
    )
    _audit_missing_quotes(loop_id, "options_entry_batch_quotes", snapshot_underlyings, quotes_map)
    for underlying in eligible_underlyings:
        quote = quotes_map.get(underlying)
        if quote is None or not _is_quote_fresh(quote, settings.data_ttl_seconds):
            continue

        try:
            indicators = _fetch_indicators(underlying, is_crypto=False)
            if indicators is None:
                continue

            signal = OPTIONS_STRATEGY.score(indicators, underlying, regime, quote.mid)
            if not signal.fired:
                continue

            situation_desc = build_situation_description(
                symbol=underlying,
                regime=regime,
                score=signal.score,
                threshold=signal.threshold,
                rsi_14=signal.rsi_14,
                macd_hist=signal.macd_hist,
                bb_pct=None,
                volume_ratio=None,
                current_price=quote.mid,
                strategy_name=OPTIONS_STRATEGY.name,
            )
            recalled = trade_memory.recall(situation_desc)
            verdict = await analyst_committee(
                symbol=underlying,
                strategy_name=OPTIONS_STRATEGY.name,
                score=signal.score,
                threshold=signal.threshold,
                regime=regime,
                current_price=quote.mid,
                rsi_14=signal.rsi_14,
                macd_hist=signal.macd_hist,
                bb_pct=None,
                volume_ratio=None,
                memories=recalled,
                loop_id=loop_id,
            )
            if not verdict.approve or verdict.size_fraction <= 0.0:
                continue

            contracts = get_option_chain(underlying, quote.mid, signal.option_type)
            contract = OPTIONS_STRATEGY.select_contract(signal, contracts, quote.mid)
            if contract is None:
                continue

            order = OPTIONS_STRATEGY.build_order(
                signal,
                contract,
                {**account, "_position_size_fraction": verdict.size_fraction},
                tier,
            )
            premium_total = order.qty * (contract.ask or order.limit_price or contract.mid or 0.0) * 100
            is_valid, reason = validate_options_order(
                premium_total=premium_total,
                equity=equity,
                existing_options_value=existing_options_value,
            )
            if not is_valid:
                _audit_event(
                    loop_id,
                    "supervisor_decision",
                    {"action": "abort", "reason": reason, "contract_symbol": contract.contract_symbol},
                    symbol=underlying,
                )
                continue

            order_value = order.qty * order.limit_price * 100
            risk_verdict = check_risk(
                daily_pnl=daily_pnl,
                equity=equity,
                open_position_count=actual_open_position_count,
                buying_power=buying_power,
                order_value=order_value,
                daily_loss_limit=tier.daily_loss_limit,
                max_concurrent=tier.max_concurrent,
                total_existing_exposure=total_existing_exposure,
                max_portfolio_exposure=settings.max_portfolio_exposure,
            )
            if not risk_verdict.passed:
                continue

            fresh_option_quote = get_option_quotes([contract.contract_symbol]).get(contract.contract_symbol)
            if fresh_option_quote is None:
                continue
            snapshot_option_quote = PriceReference(
                symbol=contract.contract_symbol,
                bid=contract.bid or 0.0,
                ask=contract.ask or 0.0,
                mid=contract.mid or max(contract.bid or 0.0, contract.ask or 0.0),
                as_of=fresh_option_quote.as_of,
            )
            decision = _supervisor_review_option(
                underlying,
                contract.contract_symbol,
                order,
                snapshot_option_quote,
                fresh_option_quote,
                buying_power,
                loop_id,
            )
            if decision.action != "execute":
                continue

            broker_result = place_option_order(order, loop_id)
            status = str(
                broker_result.get("status") if isinstance(broker_result, dict) else getattr(broker_result, "status", "")
            ).lower()
            filled_qty_raw = broker_result.get("filled_qty") if isinstance(broker_result, dict) else getattr(broker_result, "filled_qty", None)
            filled_qty = int(float(str(filled_qty_raw))) if filled_qty_raw and float(str(filled_qty_raw)) > 0 else 0
            if not _should_persist_position(status, float(filled_qty)):
                pending_underlyings.add(underlying)
                continue

            filled_price_raw = broker_result.get("filled_avg_price") if isinstance(broker_result, dict) else getattr(broker_result, "filled_avg_price", None)
            filled_price = float(str(filled_price_raw)) if filled_price_raw and float(str(filled_price_raw)) > 0 else order.limit_price
            broker_order_id = broker_result.get("id") if isinstance(broker_result, dict) else getattr(broker_result, "id", None)
            option_position = OptionsPosition(
                position_id=str(uuid.uuid4()),
                underlying=underlying,
                contract_symbol=contract.contract_symbol,
                option_type=signal.option_type,
                qty=filled_qty,
                entry_price=filled_price,
                entry_time=datetime.now(UTC),
                expiration=contract.expiration,
                high_water_price=filled_price,
                stop_loss_price=order.stop_loss_price,
                take_profit_price=order.take_profit_price,
                dte_at_entry=contract.dte,
                strategy_name=OPTIONS_STRATEGY.name,
                broker_order_id=str(broker_order_id) if broker_order_id else None,
            )
            save_option_position(option_position)
            open_option_positions.append(option_position)
            upsert_daily_stats(today, orders_placed=1)
            cost = filled_qty * filled_price * 100
            existing_options_value += cost
            buying_power = max(buying_power - cost, 0.0)
            total_existing_exposure += cost
            actual_open_position_count += 1
        except BrokerError as exc:
            logger.error("Options broker error for %s: %s", underlying, exc)
        except Exception as exc:
            logger.error("Unexpected error scanning options for %s: %s", underlying, exc)


# ── Exit Scan ─────────────────────────────────────────────────────────────────


async def _scan_exits(
    open_positions: list[Position],
    equity: float,
    regime: str,
    trade_memory: TradeMemory,
    loop_id: str = "exit",
) -> None:
    """Check exit conditions for each open position."""
    settings = get_settings()
    today = _today()
    tier = get_tier_params(equity)

    # Fetch broker positions once to get actual qty (may differ from DB due to fees)
    broker_qty_map: dict[str, float] = {}
    try:
        broker_positions = get_positions(loop_id)
        for bp in broker_positions:
            sym = str(bp.get("symbol", ""))
            qty_val = bp.get("qty") or bp.get("qty_available") or 0
            parsed_qty = float(str(qty_val))
            for alias in _symbol_aliases(sym):
                broker_qty_map[alias] = parsed_qty
    except BrokerError as exc:
        logger.warning("Could not fetch broker positions for exit scan: %s", exc)

    # Batch-fetch quotes for all open positions (1–2 API calls instead of N)
    exit_symbols = [p.symbol for p in open_positions]
    exit_quotes_map = _safe_get_universe_snapshot(
        exit_symbols,
        loop_id=loop_id,
        stage="exit_batch_quotes",
        respect_suppression=False,
    )
    _audit_missing_quotes(loop_id, "exit_batch_quotes", exit_symbols, exit_quotes_map)

    for position in open_positions:
        try:
            # Skip positions that don't exist at broker (stale DB records).
            # Use alias matching to handle slash differences (AAVE/USD vs AAVEUSD).
            pos_aliases = set(_symbol_aliases(position.symbol))
            broker_qty = None
            for alias in pos_aliases:
                if alias in broker_qty_map:
                    broker_qty = broker_qty_map[alias]
                    break
            if broker_qty is not None and broker_qty <= 0:
                logger.warning(
                    "Broker reports 0 qty for %s, skipping exit scan", position.symbol
                )
                continue

            quote = exit_quotes_map.get(position.symbol)
            if quote is None:
                continue
            if not _is_quote_fresh(quote, settings.data_ttl_seconds):
                _audit_event(
                    loop_id,
                    "safety_check",
                    {
                        "paused": False,
                        "action": "skip_symbol",
                        "reason": "stale exit quote",
                        "quote_as_of": quote.as_of.isoformat(),
                        "ttl_seconds": settings.data_ttl_seconds,
                    },
                    symbol=position.symbol,
                )
                logger.info("Skipping exit scan for %s: stale snapshot quote", position.symbol)
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
                logger.warning(
                    "EXIT SKIPPED %s: strategy '%s' not registered — "
                    "position cannot be managed by exit engine",
                    position.symbol,
                    position.strategy_name,
                )
                continue
            plan = strategy.build_exit_plan(position, tier)

            previous_managed_state = (
                position.trailing_active,
                position.high_water_price,
                position.trailing_stop_price,
                position.profit_target_price,
            )
            exit_signal, updated_pos = evaluate_exit(
                position,
                plan,
                current_price,
                rsi_14,
                atr_14=atr_14,
                regime=regime,
            )
            managed_position = updated_pos

            # Always persist trailing state updates
            if (
                managed_position.trailing_active,
                managed_position.high_water_price,
                managed_position.trailing_stop_price,
                managed_position.profit_target_price,
            ) != previous_managed_state and exit_signal is None:
                update_position(managed_position)

            if exit_signal is None:
                continue

            # Use broker's actual qty (accounts for fees/rounding),
            # fall back to DB qty if broker lookup unavailable.
            position_qty_before_exit = broker_qty if broker_qty is not None else managed_position.qty
            sell_qty = position_qty_before_exit
            if sell_qty <= 0:
                logger.warning(
                    "Broker reports 0 qty for %s, skipping sell", position.symbol
                )
                continue
            if sell_qty != managed_position.qty:
                logger.info(
                    "Qty reconcile %s: DB=%.10f broker=%.10f (diff=%.10f)",
                    position.symbol,
                    managed_position.qty,
                    sell_qty,
                    managed_position.qty - sell_qty,
                )

            sell_qty = _planned_exit_qty(
                managed_position,
                sell_qty,
                current_price,
                exit_signal.exit_fraction,
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
            broker_result = place_order(sell_order, loop_id)
            status, executed_qty, executed_price = _extract_fill_details(broker_result)

            # Poll broker if the order is still pending (market orders usually
            # fill within seconds but the initial response can be PENDING_NEW).
            if not _should_persist_position(status, executed_qty):
                order_id = str(_broker_result_field(broker_result, "id") or "")
                if order_id and status in _PENDING_ORDER_STATUSES:
                    status, executed_qty, executed_price = _poll_order_fill(
                        order_id, position.symbol, loop_id,
                    )

            if not _should_persist_position(status, executed_qty):
                logger.info(
                    "Exit order for %s status=%s filled_qty=%.4f — keeping position open",
                    position.symbol,
                    status or "unknown",
                    executed_qty,
                )
                continue

            executed_qty = min(executed_qty, sell_qty)
            if executed_qty <= 0:
                logger.warning("Exit order for %s reported no executable quantity", position.symbol)
                continue

            exit_price = executed_price or current_price
            if exit_signal.trigger in {"profit_target_partial", "atr_target_partial"}:
                managed_position.trailing_active = True
                managed_position.high_water_price = max(managed_position.high_water_price, exit_price)
                managed_position.trailing_stop_price = max(
                    exit_price * (1 - plan.trail_pct),
                    managed_position.entry_price,
                )
                managed_position.profit_target_price = -1.0
            exited_position = Position(
                id=managed_position.id,
                symbol=managed_position.symbol,
                asset_class=managed_position.asset_class,
                qty=executed_qty,
                entry_price=managed_position.entry_price,
                high_water_price=managed_position.high_water_price,
                trailing_stop_price=managed_position.trailing_stop_price,
                trailing_active=managed_position.trailing_active,
                entry_time=managed_position.entry_time,
                broker_order_id=managed_position.broker_order_id,
                strategy_name=managed_position.strategy_name,
                stop_loss_price=managed_position.stop_loss_price,
                profit_target_price=managed_position.profit_target_price,
            )
            log_trade(exited_position, exit_price, exit_signal.trigger)
            realized_pnl = (exit_price - managed_position.entry_price) * executed_qty

            remaining_qty = max(position_qty_before_exit - executed_qty, 0.0)
            if remaining_qty > 0:
                managed_position.qty = remaining_qty
                update_position(managed_position)
                upsert_daily_stats(today, realized_pnl_usd=realized_pnl)
                logger.info(
                    "PARTIAL EXIT %s: filled=%.4f remaining=%.4f price=%.4f reason=%s",
                    position.symbol,
                    executed_qty,
                    remaining_qty,
                    exit_price,
                    exit_signal.reason,
                )
                continue

            close_position(position.id)
            upsert_daily_stats(today, trades_closed=1, realized_pnl_usd=realized_pnl)
            logger.info(
                "EXIT %s: trigger=%s price=%.4f reason=%s",
                position.symbol,
                exit_signal.trigger,
                exit_price,
                exit_signal.reason,
            )

            # ── Post-trade reflection (1 LLM call) ──
            try:
                hold_hrs = (datetime.now(UTC) - position.entry_time).total_seconds() / 3600
                realized_pnl = (exit_price - position.entry_price) * executed_qty
                sit = build_situation_description(
                    symbol=position.symbol,
                    regime=regime,
                    score=0,
                    threshold=0,
                    rsi_14=rsi_14,
                    macd_hist=None,
                    bb_pct=None,
                    volume_ratio=None,
                    current_price=exit_price,
                    strategy_name=position.strategy_name,
                )
                out = build_outcome_description(
                    symbol=position.symbol,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    exit_trigger=exit_signal.trigger,
                    hold_hours=hold_hrs,
                    realized_pnl=realized_pnl,
                )
                recalled = trade_memory.recall(sit)
                lesson = await reflect_on_trade(
                    symbol=position.symbol,
                    situation=sit,
                    outcome=out,
                    memories=recalled,
                    loop_id=loop_id,
                )
                if lesson:
                    mem_entry = MemoryEntry(situation=sit, outcome=out, lesson=lesson)
                    trade_memory.store(mem_entry)
                    save_memory(mem_entry)
            except Exception as exc:
                logger.warning("Reflection failed for %s: %s", position.symbol, exc)

        except BrokerError as exc:
            logger.error("Broker error exiting %s: %s", position.symbol, exc)
        except Exception as exc:
            logger.error("Unexpected error on exit scan for %s: %s", position.symbol, exc)


async def _scan_option_exits(
    open_option_positions: list[OptionsPosition],
    loop_id: str = "options-exit",
) -> None:
    """Check open options positions for exit conditions and place exit orders."""
    if not open_option_positions:
        return

    settings = get_settings()
    today = _today()
    broker_option_qty_map: dict[str, int] = {}
    try:
        for broker_position in get_option_positions(loop_id):
            symbol = str(broker_position.get("symbol", ""))
            qty_val = broker_position.get("qty") or broker_position.get("qty_available") or 0
            broker_option_qty_map[symbol] = int(float(str(qty_val)))
    except Exception as exc:
        logger.warning("Could not fetch broker option positions for exit scan: %s", exc)

    quote_symbols = [position.contract_symbol for position in open_option_positions]
    quotes_map = _safe_get_option_quotes(
        quote_symbols,
        loop_id=loop_id,
        stage="options_exit_batch_quotes",
    )
    _audit_missing_quotes(
        loop_id,
        "options_exit_batch_quotes",
        quote_symbols,
        quotes_map,
    )
    for position in open_option_positions:
        try:
            quote = quotes_map.get(position.contract_symbol)
            if quote is None or not _is_quote_fresh(quote, settings.data_ttl_seconds):
                continue

            current_bid = quote.bid if quote.bid > 0 else quote.mid
            if current_bid <= 0:
                continue
            current_dte = max((position.expiration - datetime.now(UTC).date()).days, 0)
            indicators = _fetch_indicators(position.underlying, is_crypto=False)

            if current_bid > position.high_water_price:
                position = position.model_copy(update={"high_water_price": current_bid})
                update_option_position(position)

            exit_signal = check_options_position(position, current_bid, current_dte, indicators)
            if exit_signal is None:
                continue

            sell_qty = broker_option_qty_map.get(position.contract_symbol, position.qty)
            if sell_qty <= 0:
                continue

            exit_order = OPTIONS_STRATEGY.build_exit_order(position.model_copy(update={"qty": sell_qty}), current_bid, exit_signal.reason)
            broker_result = place_option_order(exit_order, loop_id)
            status, filled_qty, filled_price = _extract_fill_details(broker_result)
            executed_qty = int(min(float(sell_qty), filled_qty, float(position.qty)))
            if not _should_persist_position(status, float(executed_qty)):
                continue

            exit_price = filled_price or exit_order.limit_price
            settled_position = position.model_copy(update={"qty": executed_qty})
            log_option_trade(settled_position, exit_price, exit_signal.reason)

            remaining_qty = position.qty - executed_qty
            if remaining_qty > 0:
                update_option_position(position.model_copy(update={"qty": remaining_qty}))
                logger.info(
                    "PARTIAL OPTION EXIT %s: filled=%d remaining=%d price=%.4f reason=%s",
                    position.contract_symbol,
                    executed_qty,
                    remaining_qty,
                    exit_price,
                    exit_signal.reason,
                )
                continue

            close_option_position(position.position_id)
            upsert_daily_stats(today, trades_closed=1)
            logger.info(
                "OPTION EXIT %s: trigger=%s price=%.4f reason=%s",
                position.contract_symbol,
                exit_signal.reason,
                exit_price,
                exit_signal.reason,
            )
        except BrokerError as exc:
            logger.error("Broker error exiting option %s: %s", position.contract_symbol, exc)
        except Exception as exc:
            logger.error("Unexpected error on options exit scan for %s: %s", position.contract_symbol, exc)


def _find_strategy(name: str) -> Strategy | None:
    """Find a registered strategy by name."""
    for s in STRATEGIES:
        if s.name == name:
            return s
    return None


def _reconcile_broker_positions(
    open_positions: list[Position],
    broker_positions: list[dict[str, object]],
    tier: TierParams,
) -> None:
    """Create DB records for broker positions not yet tracked locally.

    Handles positions opened outside the current DB (manual orders, prior code
    versions, DB migrations). Saves a minimal record with tier-based stops so
    the exit monitor can manage them on the very next cycle.

    Respects max_concurrent: will not import positions that would push the
    total count above the tier limit.
    """
    settings = get_settings()
    for bp in broker_positions:
        broker_symbol = str(bp.get("symbol", "")).strip().upper()
        if not broker_symbol:
            continue

        # Robust alias matching handles broker's slash-free form (DOGEUSD vs DOGE/USD).
        already_tracked = any(
            bool(set(_symbol_aliases(broker_symbol)) & set(_symbol_aliases(p.symbol)))
            for p in open_positions
        )
        if already_tracked:
            continue

        # Enforce max_concurrent — don't import if already at limit.
        if len(open_positions) >= tier.max_concurrent:
            logger.warning(
                "RECONCILE SKIP %s: open positions (%d) already at max_concurrent (%d)",
                broker_symbol,
                len(open_positions),
                tier.max_concurrent,
            )
            continue

        qty = _parse_positive_float(bp.get("qty") or bp.get("qty_available"))
        entry_price = _parse_positive_float(bp.get("avg_entry_price"))
        if not qty or not entry_price:
            continue

        # Prefer broker's explicit asset_class; fall back to symbol heuristic.
        asset_class_raw = str(bp.get("asset_class", "")).lower()
        is_crypto_pos = "crypto" in asset_class_raw or _is_crypto(broker_symbol)

        # Broker sends crypto without slash (DOGEUSD). Restore canonical form.
        if is_crypto_pos and "/" not in broker_symbol and broker_symbol.endswith("USD"):
            canonical = broker_symbol[:-3] + "/USD"
            if canonical not in settings.crypto_universe:
                canonical = broker_symbol  # unknown pair — keep broker form
        else:
            canonical = broker_symbol

        asset_class = "crypto" if is_crypto_pos else "equity"
        strategy_name = "crypto_momentum" if is_crypto_pos else "equity_momentum"

        position = Position(
            symbol=canonical,
            asset_class=asset_class,
            qty=qty,
            entry_price=entry_price,
            high_water_price=entry_price,
            entry_time=datetime.now(UTC),
            broker_order_id=None,
            strategy_name=strategy_name,
            stop_loss_price=round(entry_price * (1 - tier.stop_loss_pct), 8),
            profit_target_price=round(entry_price * (1 + tier.profit_target_pct), 8),
        )
        save_position(position)
        open_positions.append(position)
        logger.warning(
            "RECONCILED orphan broker position %s: qty=%.8f entry=%.6f "
            "stop=%.6f target=%.6f strategy=%s",
            canonical,
            qty,
            entry_price,
            position.stop_loss_price,
            position.profit_target_price,
            strategy_name,
        )


def _reconcile_stale_positions(
    open_positions: list[Position],
    broker_positions: list[dict[str, object]],
    loop_id: str,
) -> None:
    """Close DB positions that no longer exist at the broker.

    Builds a set of broker-known symbols (with aliases for slash-free forms)
    and closes any DB position whose symbol has no broker counterpart.
    This prevents the exit scan from repeatedly trying to sell phantom positions.
    """
    # Build the set of symbols the broker actually holds.
    broker_symbols: set[str] = set()
    for bp in broker_positions:
        sym = str(bp.get("symbol", "")).strip().upper()
        if sym:
            for alias in _symbol_aliases(sym):
                broker_symbols.add(alias)

    stale: list[Position] = []
    for position in open_positions:
        pos_aliases = set(_symbol_aliases(position.symbol))
        if not pos_aliases.intersection(broker_symbols):
            stale.append(position)

    for position in stale:
        # Log the trade with entry_price as exit price (best approximation
        # when broker already sold the position outside our control).
        log_trade(position, position.entry_price, "broker_reconciliation")
        close_position(position.id)
        open_positions.remove(position)
        logger.warning(
            "RECONCILED stale DB position %s (id=%s): broker does not hold this symbol — "
            "logging trade and closing (qty=%.8f entry=%.6f)",
            position.symbol,
            position.id,
            position.qty,
            position.entry_price,
        )
        _audit_event(
            loop_id,
            "supervisor_decision",
            {
                "action": "broker_reconciliation",
                "reason": "DB position not found at broker — closing stale record",
                "symbol": position.symbol,
                "position_id": position.id,
                "qty": position.qty,
                "entry_price": position.entry_price,
            },
            symbol=position.symbol,
        )


# ── Main Cycle ─────────────────────────────────────────────────────────────


async def run_cycle() -> None:
    """Run a single trading cycle. Called by cron — no internal loop."""
    settings = get_settings()
    cycle_id = str(uuid.uuid4())
    today = _today()
    regime = "neutral"
    loop_status = "ok"
    loop_error: str | None = None
    fatal_exc: Exception | None = None

    _audit_event(
        cycle_id,
        "loop_start",
        {
            "paper": settings.alpaca_paper,
            "prompt_version": settings.prompt_version,
            "strategy_count": len(STRATEGIES),
            "options_enabled": settings.options_enabled,
        },
    )

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

        # Account state (needed for both exit and entry sizing)
        account = get_account(cycle_id)
        equity = float(account.get("equity", "0"))
        upsert_daily_stats(today, loop_runs=1, ending_equity=equity)

        if equity <= 0:
            logger.error("Account equity is $0 — skipping cycle")
            loop_status = "halted"
            return

        starting = float(account.get("last_equity", equity))
        daily_pnl = (equity - starting) / starting if starting > 0 else 0.0
        tier = get_tier_params(equity)
        daily_loss_limit = min(settings.max_daily_loss_pct, tier.daily_loss_limit)

        broker_positions = get_positions(cycle_id)
        stale_orders_cancelled = cancel_stale_orders(loop_id=cycle_id)
        if stale_orders_cancelled:
            logger.info("Cancelled %d stale broker orders before entry scan", stale_orders_cancelled)
        recent_orders = get_recent_orders(cycle_id)

        # Load positions
        open_positions = load_open_positions()
        open_option_positions = load_open_option_positions()

        # Reconcile: import any broker positions not yet in our DB.
        # Handles positions entered outside the current DB (manual orders,
        # prior code version, DB migration) so the exit engine can see them.
        _reconcile_broker_positions(open_positions, broker_positions, tier)

        # Reverse reconciliation: close DB positions that no longer exist at broker.
        # Prevents phantom sell attempts and false position-count inflation.
        _reconcile_stale_positions(open_positions, broker_positions, cycle_id)

        # Load trade memory (BM25 index for past reflections)
        memory_entries = load_all_memories()
        trade_memory = TradeMemory(memory_entries)
        logger.info("Trade memory loaded: %d entries", trade_memory.size)

        # Exit scan runs BEFORE the TRADING_PAUSE check.
        # Open positions must always be monitored for stops and profit targets
        # even when new entries are disabled.
        if open_positions:
            await _scan_exits(open_positions, equity, regime, trade_memory, cycle_id)
        if open_option_positions:
            await _scan_option_exits(open_option_positions, cycle_id)

        # TRADING_PAUSE halts new entries only — exits already ran above.
        if settings.trading_pause:
            _audit_event(
                cycle_id,
                "safety_check",
                {
                    "paused": True,
                    "action": "halt",
                    "reason": "TRADING_PAUSE enabled",
                },
            )
            logger.warning("Cycle %s: new entries halted — TRADING_PAUSE enabled", cycle_id)
            loop_status = "halted"
            return

        if daily_pnl <= -daily_loss_limit:
            _audit_event(
                cycle_id,
                "safety_check",
                {
                    "paused": True,
                    "action": "halt",
                    "reason": "daily loss limit breached",
                    "daily_pnl": daily_pnl,
                    "daily_loss_limit": daily_loss_limit,
                },
            )
            logger.warning(
                "Cycle %s: new entries halted — daily loss %.2f%% exceeds limit %.2f%%",
                cycle_id,
                daily_pnl * 100,
                daily_loss_limit * 100,
            )
            loop_status = "halted"
            return

        _audit_event(
            cycle_id,
            "safety_check",
            {
                "paused": False,
                "action": "continue",
                "reason": "preflight checks passed",
                "daily_pnl": daily_pnl,
                "daily_loss_limit": daily_loss_limit,
            },
        )

        # Entry scan (async — includes analyst committee)
        await _scan_entries(regime, account, open_positions, broker_positions, recent_orders, trade_memory, cycle_id)
        await _scan_option_entries(regime, account, open_option_positions, broker_positions, recent_orders, trade_memory, cycle_id)

        logger.info(
            "Cycle %s complete — equity=$%.2f, positions=%d, options=%d, regime=%s",
            cycle_id,
            equity,
            len(open_positions),
            len(open_option_positions),
            regime,
        )

    except Exception as exc:
        loop_status = "failed"
        loop_error = str(exc)
        fatal_exc = exc
        logger.error("Cycle %s failed: %s", cycle_id, exc, exc_info=True)
    finally:
        _audit_event(
            cycle_id,
            "loop_end",
            {
                "status": loop_status,
                "regime": regime,
                "error": loop_error,
            },
        )

    if fatal_exc is not None:
        raise fatal_exc


def main() -> None:
    """CLI entry point — runs one cycle then exits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(run_cycle())


if __name__ == "__main__":
    main()
