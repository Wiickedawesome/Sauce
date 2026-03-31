"""
adapters/broker.py — Alpaca broker adapter.

Wraps alpaca-py TradingClient for account queries and order placement.
Supports both US equities (ticker symbols) and crypto pairs (e.g. BTC/USD).

Rules:
- ALPACA_PAPER defaults to True — never default to live.
- Every broker call logs AuditEvent before and after.
- place_order() requires a fully validated Order schema — never raw dicts.
- On any broker error: log to DB, raise BrokerError — never retry silently.

NOTE: Verify alpaca-py method signatures against https://alpaca.markets/docs/api-references/
      before going live. SDK versions may change field names.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from sauce.adapters.utils import call_with_retry
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, Order, PriceReference

if TYPE_CHECKING:
    from sauce.core.options_schemas import OptionsOrder

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────


class BrokerError(Exception):
    """Raised on any unrecoverable Alpaca API error."""

    pass


# ── Client factory ────────────────────────────────────────────────────────────


def _get_trading_client() -> Any:
    """
    Return an Alpaca TradingClient configured for paper or live trading.

    Paper/live is determined exclusively by the ALPACA_PAPER env var.
    """
    from alpaca.trading.client import TradingClient

    settings = get_settings()
    return TradingClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
        paper=settings.alpaca_paper,
    )


# ── Account ───────────────────────────────────────────────────────────────────


def get_account(loop_id: str = "unset") -> dict[str, Any]:
    """
    Fetch current account details from Alpaca.

    Returns a dict with keys: id, status, currency, buying_power,
    equity, last_equity, portfolio_value, pattern_day_trader, etc.

    All values are strings as returned by the Alpaca API.
    """
    from sauce.adapters.db import log_event

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="broker_call",
            payload={"action": "get_account"},
            timestamp=datetime.now(UTC),
        )
    )

    try:
        client = _get_trading_client()
        account = call_with_retry(client.get_account)
        result: dict[str, Any] = account.__dict__

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="broker_response",
                payload={
                    "action": "get_account",
                    "equity": str(getattr(account, "equity", "unknown")),
                    "buying_power": str(getattr(account, "buying_power", "unknown")),
                    "portfolio_value": str(getattr(account, "portfolio_value", "unknown")),
                },
                timestamp=datetime.now(UTC),
            )
        )

        return result

    except Exception as exc:
        _log_broker_error(loop_id, "get_account", exc)
        raise BrokerError(f"get_account failed: {exc}") from exc


# ── Positions ─────────────────────────────────────────────────────────────────


def get_positions(loop_id: str = "unset") -> list[dict[str, Any]]:
    """
    Fetch all open positions from Alpaca.

    Returns a list of position dicts. Each dict contains at minimum:
    symbol, qty, side, avg_entry_price, market_value, unrealized_pl.
    """
    from sauce.adapters.db import log_event

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="broker_call",
            payload={"action": "get_positions"},
            timestamp=datetime.now(UTC),
        )
    )

    try:
        client = _get_trading_client()
        positions = call_with_retry(client.get_all_positions)
        result: list[dict[str, Any]] = [p.__dict__ for p in positions]

        # Validate expected fields are present (catches Alpaca SDK changes early).
        _EXPECTED_FIELDS = {
            "symbol",
            "qty",
            "side",
            "avg_entry_price",
            "market_value",
            "current_price",
        }
        for pos_dict in result:
            missing = _EXPECTED_FIELDS - pos_dict.keys()
            if missing:
                logger.warning(
                    "Position for %s missing expected fields: %s — Alpaca SDK may have changed",
                    pos_dict.get("symbol", "???"),
                    missing,
                )

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="broker_response",
                payload={
                    "action": "get_positions",
                    "count": len(result),
                    "symbols": [str(getattr(p, "symbol", "?")) for p in positions],
                },
                timestamp=datetime.now(UTC),
            )
        )

        return result

    except Exception as exc:
        _log_broker_error(loop_id, "get_positions", exc)
        raise BrokerError(f"get_positions failed: {exc}") from exc


# ── Order placement ───────────────────────────────────────────────────────────


def _build_stop_loss_request(order: Order) -> Any:
    """
    Build a StopLossRequest for bracket/OTO orders.

    Supports stop-limit on SL leg when stop_loss_limit_price is provided,
    giving better slippage protection vs pure market-on-trigger.
    """
    from alpaca.trading.requests import StopLossRequest

    assert order.stop_loss_price is not None, "stop_loss_price required for StopLossRequest"

    if order.stop_loss_limit_price is not None:
        # Stop-limit: when stop triggers, place limit order (slippage protection)
        return StopLossRequest(
            stop_price=order.stop_loss_price,
            limit_price=order.stop_loss_limit_price,
        )
    else:
        # Pure stop: when stop triggers, market order (fills immediately, may slip)
        return StopLossRequest(stop_price=order.stop_loss_price)


def _determine_order_class(order: Order, is_crypto: bool) -> str:
    """
    Determine effective order class based on explicit field or auto-detection.

    Order class hierarchy:
    - simple: Single order, no attached legs
    - bracket: Entry + SL + TP (requires both prices)
    - oto: Entry triggers single dependent order (SL only, no TP)
    - oco: Two linked exit orders — one cancels other (SL XOR TP)

    Crypto always returns 'simple' — Alpaca doesn't support advanced order classes.
    """
    if is_crypto:
        return "simple"

    # Explicit order_class takes precedence
    if order.order_class is not None:
        return order.order_class

    # Auto-detect from stop_loss_price / take_profit_price
    has_sl = order.stop_loss_price is not None
    has_tp = order.take_profit_price is not None

    if has_sl and has_tp:
        return "bracket"
    elif has_sl and not has_tp:
        return "oto"
    elif has_tp and not has_sl:
        # TP only — this is unusual but valid as simple limit order
        return "simple"
    else:
        return "simple"


def place_order(order: Order, loop_id: str = "unset") -> dict[str, Any]:
    """
    Submit an order to Alpaca. Requires a fully-validated Order schema.

    Supports multi-leg order classes for equities:
    - bracket: Entry + stop-loss + take-profit (server-side SL/TP)
    - oto: Entry triggers stop-loss only (One-Triggers-Other)
    - oco: Two linked exit orders (One-Cancels-Other) — no entry

    Stop-limit on SL leg: Set stop_loss_limit_price for slippage protection.
    Without it, SL triggers a market order which may slip in fast markets.

    This function must only be called from core/loop.py after Supervisor approval.
    Never call this directly from an agent.

    Returns a dict with Alpaca's order response (id, status, filled_qty, etc.).
    On any error: raises BrokerError — never retries automatically.
    """
    from alpaca.trading.enums import (
        OrderClass,
        OrderSide,
        TimeInForce,
    )
    from alpaca.trading.requests import (
        LimitOrderRequest,
        MarketOrderRequest,
        StopLimitOrderRequest,
        StopOrderRequest,
        TakeProfitRequest,
        TrailingStopOrderRequest,
    )

    from sauce.adapters.db import log_event

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="broker_call",
            payload={
                "action": "place_order",
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "order_type": order.order_type,
                "order_class": order.order_class,
                "time_in_force": order.time_in_force,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "stop_loss_price": order.stop_loss_price,
                "stop_loss_limit_price": order.stop_loss_limit_price,
                "take_profit_price": order.take_profit_price,
            },
            timestamp=datetime.now(UTC),
        )
    )

    try:
        client = _get_trading_client()

        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
        tif_map = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        # Alpaca crypto pairs only support gtc/ioc/fok — never day.
        # Force gtc for any symbol containing "/" (e.g. BTC/USD, ETH/USD).
        is_crypto = "/" in order.symbol
        effective_tif = "gtc" if is_crypto and order.time_in_force == "day" else order.time_in_force
        tif = tif_map[effective_tif]

        # Determine order class: explicit, or auto-detect from SL/TP presence
        effective_class = _determine_order_class(order, is_crypto)
        {
            "simple": None,  # Don't specify — Alpaca default
            "bracket": OrderClass.BRACKET,
            "oto": OrderClass.OTO,
            "oco": OrderClass.OCO,
        }.get(effective_class)

        order_request: Any
        if order.order_type == "trailing_stop":
            # Native trailing stop order — Alpaca handles the ratchet server-side
            trail_pct = order.trail_percent if order.trail_percent is not None else 0.05
            order_request = TrailingStopOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                trail_percent=trail_pct,
            )
            logger.info(
                "TRAILING STOP %s: qty=%.4f, trail=%.1f%%",
                order.symbol,
                order.qty,
                trail_pct * 100,
            )
        elif order.order_type == "market":
            if effective_class == "bracket":
                # Bracket: entry + stop-loss + take-profit
                assert order.take_profit_price is not None
                order_request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
                    stop_loss=_build_stop_loss_request(order),
                )
                sl_type = "stop-limit" if order.stop_loss_limit_price else "stop"
                logger.info(
                    "BRACKET order %s: entry=market, TP=%.2f, SL=%.2f (%s)",
                    order.symbol,
                    order.take_profit_price,
                    order.stop_loss_price,
                    sl_type,
                )
            elif effective_class == "oto":
                # OTO: entry triggers stop-loss only (no take-profit)
                order_request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                    order_class=OrderClass.OTO,
                    stop_loss=_build_stop_loss_request(order),
                )
                sl_type = "stop-limit" if order.stop_loss_limit_price else "stop"
                logger.info(
                    "OTO order %s: entry=market, SL=%.2f (%s)",
                    order.symbol,
                    order.stop_loss_price,
                    sl_type,
                )
            elif effective_class == "oco":
                # OCO: Two linked exit orders — one cancels other
                # This is for exits only, no entry. Requires SL + TP.
                if order.stop_loss_price is None or order.take_profit_price is None:
                    raise BrokerError("OCO requires both stop_loss_price and take_profit_price")
                order_request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                    order_class=OrderClass.OCO,
                    take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
                    stop_loss=_build_stop_loss_request(order),
                )
                logger.info(
                    "OCO order %s: TP=%.2f OR SL=%.2f",
                    order.symbol,
                    order.take_profit_price,
                    order.stop_loss_price,
                )
            else:
                # Simple market order
                order_request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                )
        elif order.order_type == "limit":
            if order.limit_price is None:
                raise BrokerError("limit_price is required for limit orders")
            if effective_class == "bracket":
                assert order.take_profit_price is not None
                order_request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    order_class=OrderClass.BRACKET,
                    take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
                    stop_loss=_build_stop_loss_request(order),
                )
                sl_type = "stop-limit" if order.stop_loss_limit_price else "stop"
                logger.info(
                    "BRACKET order %s: entry=%.2f, TP=%.2f, SL=%.2f (%s)",
                    order.symbol,
                    order.limit_price,
                    order.take_profit_price,
                    order.stop_loss_price,
                    sl_type,
                )
            elif effective_class == "oto":
                order_request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    order_class=OrderClass.OTO,
                    stop_loss=_build_stop_loss_request(order),
                )
                sl_type = "stop-limit" if order.stop_loss_limit_price else "stop"
                logger.info(
                    "OTO order %s: entry=%.2f, SL=%.2f (%s)",
                    order.symbol,
                    order.limit_price,
                    order.stop_loss_price,
                    sl_type,
                )
            elif effective_class == "oco":
                # OCO with limit entry
                if order.stop_loss_price is None or order.take_profit_price is None:
                    raise BrokerError("OCO requires both stop_loss_price and take_profit_price")
                order_request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                    order_class=OrderClass.OCO,
                    take_profit=TakeProfitRequest(limit_price=order.take_profit_price),
                    stop_loss=_build_stop_loss_request(order),
                )
                logger.info(
                    "OCO order %s: TP=%.2f OR SL=%.2f",
                    order.symbol,
                    order.take_profit_price,
                    order.stop_loss_price,
                )
            else:
                order_request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.qty,
                    side=side,
                    time_in_force=tif,
                    limit_price=order.limit_price,
                )
        elif order.order_type == "stop":
            if order.stop_price is None:
                raise BrokerError("stop_price is required for stop orders")
            order_request = StopOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                stop_price=order.stop_price,
            )
        elif order.order_type == "stop_limit":
            if order.limit_price is None or order.stop_price is None:
                raise BrokerError("Both limit_price and stop_price required for stop_limit orders")
            order_request = StopLimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
            )
        else:
            raise BrokerError(f"Unsupported order_type: {order.order_type!r}")

        submitted = client.submit_order(order_data=order_request)
        result: dict[str, Any] = submitted.__dict__

        # Ensure 'id' is present — Pydantic v2 model __dict__ may not
        # include mapped attributes reliably across SDK versions.
        if "id" not in result or result["id"] is None:
            raw_id = getattr(submitted, "id", None)
            if raw_id is not None:
                result["id"] = raw_id

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="broker_response",
                payload={
                    "action": "place_order",
                    "symbol": order.symbol,
                    "broker_order_id": str(getattr(submitted, "id", "unknown")),
                    "status": str(getattr(submitted, "status", "unknown")),
                },
                timestamp=datetime.now(UTC),
            )
        )

        return result

    except BrokerError:
        raise
    except Exception as exc:
        _log_broker_error(loop_id, "place_order", exc)
        raise BrokerError(f"place_order failed for {order.symbol}: {exc}") from exc


# ── Quote (convenience, single symbol) ───────────────────────────────────────


def get_latest_quote(symbol: str, loop_id: str = "unset") -> PriceReference:
    """
    Fetch the latest quote for a single symbol from the Alpaca trading API.

    Supports both equity tickers (AAPL) and crypto pairs (BTC/USD).
    Returns a PriceReference with a real as_of timestamp from the API.

    NOTE: Market data should primarily come from market_data.py for bulk
    snapshot fetches. This function is used for freshness re-checks in
    the Execution agent.
    """
    from sauce.adapters.db import log_event

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="broker_call",
            payload={"action": "get_latest_quote", "symbol": symbol},
            timestamp=datetime.now(UTC),
        )
    )

    try:
        # Delegate to market_data.get_quote() which enforces strict timestamp
        # handling (no datetime.now() fallback) — Findings 6.2 and 7.5.
        from sauce.adapters.market_data import MarketDataError, get_quote

        price_ref = get_quote(symbol)

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="broker_response",
                payload={
                    "action": "get_latest_quote",
                    "symbol": symbol,
                    "mid": price_ref.mid,
                    "as_of": price_ref.as_of.isoformat(),
                },
                timestamp=datetime.now(UTC),
            )
        )

        return price_ref

    except MarketDataError as exc:
        _log_broker_error(loop_id, f"get_latest_quote({symbol})", exc)
        raise BrokerError(f"get_latest_quote failed for {symbol}: {exc}") from exc
    except BrokerError:
        raise
    except Exception as exc:
        _log_broker_error(loop_id, f"get_latest_quote({symbol})", exc)
        raise BrokerError(f"get_latest_quote failed for {symbol}: {exc}") from exc


# ── Recent orders query ──────────────────────────────────────────────────────


def get_recent_orders(loop_id: str = "unset") -> list[dict[str, Any]]:
    """
    Return today's orders from the broker as a list of dicts.

    Each dict has keys: broker_order_id, symbol, side, qty, status, created_at.
    Never raises — logs errors and returns [] on failure.
    """
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    try:
        client = _get_trading_client()
        request = GetOrdersRequest(status=QueryOrderStatus.ALL)
        raw = client.get_orders(filter=request)
        result: list[dict[str, Any]] = []
        for o in raw:
            result.append(
                {
                    "broker_order_id": str(getattr(o, "id", "")),
                    "symbol": str(getattr(o, "symbol", "")),
                    "side": str(getattr(o, "side", "")),
                    "qty": str(getattr(o, "qty", "")),
                    "status": str(getattr(o, "status", "")),
                    "created_at": str(getattr(o, "created_at", "")),
                }
            )
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("get_recent_orders failed [loop_id=%s]: %s", loop_id, exc)
        return []


# ── Stale order cancellation ─────────────────────────────────────────────────


def cancel_stale_orders(max_age_minutes: int = 30, loop_id: str = "unset") -> int:
    """
    Cancel any open (unfilled) orders older than max_age_minutes.

    Returns the count of orders cancelled. Never raises — logs errors
    and returns 0 on failure.
    """
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    from sauce.adapters.db import log_event

    try:
        client = _get_trading_client()
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = client.get_orders(filter=request)

        if not open_orders:
            return 0

        now = datetime.now(UTC)
        cutoff = now - timedelta(minutes=max_age_minutes)
        cancelled = 0

        for order in open_orders:
            created_at = getattr(order, "created_at", None)
            if created_at is None:
                continue
            # Normalise naive timestamps to UTC
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=UTC)
            if created_at < cutoff:
                try:
                    order_id = str(getattr(order, "id", ""))
                    client.cancel_order_by_id(order_id)
                    cancelled += 1
                    logger.info(
                        "Cancelled stale order %s for %s (age: %d min) [loop_id=%s]",
                        order_id,
                        getattr(order, "symbol", "?"),
                        int((now - created_at).total_seconds() / 60),
                        loop_id,
                    )
                    log_event(
                        AuditEvent(
                            loop_id=loop_id,
                            event_type="broker_call",
                            payload={
                                "action": "cancel_stale_order",
                                "order_id": order_id,
                                "symbol": str(getattr(order, "symbol", "?")),
                                "age_minutes": int((now - created_at).total_seconds() / 60),
                            },
                            timestamp=now,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to cancel stale order %s: %s [loop_id=%s]",
                        getattr(order, "id", "?"),
                        exc,
                        loop_id,
                    )

        return cancelled

    except Exception as exc:  # noqa: BLE001
        logger.warning("cancel_stale_orders failed: %s [loop_id=%s]", exc, loop_id)
        return 0


# ── Error logging helper ──────────────────────────────────────────────────────

# ── Options: order placement ──────────────────────────────────────────────────


def place_option_order(
    order: "OptionsOrder",
    loop_id: str = "unset",
) -> dict[str, Any]:
    """
    Submit an options order to Alpaca. Limit orders only.

    Requires a fully-validated OptionsOrder schema.
    Returns a dict with Alpaca's order response.
    Raises BrokerError on failure.
    """
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest

    from sauce.adapters.db import log_event

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="options_order_submitted",
            symbol=order.underlying,
            payload={
                "action": "place_option_order",
                "contract": order.contract_symbol,
                "side": order.side,
                "qty": order.qty,
                "limit_price": order.limit_price,
                "stage": order.stage,
                "source": order.source,
            },
            timestamp=datetime.now(UTC),
        )
    )

    try:
        client = _get_trading_client()
        side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL

        order_request = LimitOrderRequest(
            symbol=order.contract_symbol,
            qty=order.qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=order.limit_price,
        )

        submitted = client.submit_order(order_data=order_request)
        result: dict[str, Any] = submitted.__dict__

        # Ensure 'id' is present — Pydantic v2 model __dict__ may not
        # include mapped attributes reliably across SDK versions.
        if "id" not in result or result["id"] is None:
            raw_id = getattr(submitted, "id", None)
            if raw_id is not None:
                result["id"] = raw_id

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="broker_response",
                symbol=order.underlying,
                payload={
                    "action": "place_option_order",
                    "contract": order.contract_symbol,
                    "broker_order_id": str(getattr(submitted, "id", "unknown")),
                    "status": str(getattr(submitted, "status", "unknown")),
                },
                timestamp=datetime.now(UTC),
            )
        )

        return result

    except Exception as exc:
        _log_broker_error(loop_id, f"place_option_order({order.contract_symbol})", exc)
        raise BrokerError(
            f"place_option_order failed for {order.contract_symbol}: {exc}",
        ) from exc


def get_option_positions(loop_id: str = "unset") -> list[dict[str, Any]]:
    """
    Fetch open options positions from Alpaca.

    Returns a list of position dicts filtered to asset_class == 'options'.
    Never raises — logs errors and returns [] on failure.
    """
    from sauce.adapters.db import log_event

    try:
        client = _get_trading_client()
        positions = call_with_retry(client.get_all_positions)
        result: list[dict[str, Any]] = []

        for p in positions:
            # Alpaca marks options positions with asset_class
            asset_class = str(getattr(p, "asset_class", "")).lower()
            if "option" in asset_class:
                result.append(p.__dict__)

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="broker_response",
                payload={
                    "action": "get_option_positions",
                    "count": len(result),
                },
                timestamp=datetime.now(UTC),
            )
        )

        return result

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "get_option_positions failed [loop_id=%s]: %s",
            loop_id,
            exc,
        )
        return []


# ── Error logging helper ──────────────────────────────────────────────────────


def _log_broker_error(loop_id: str, action: str, exc: Exception) -> None:
    """Log a broker error to the audit DB. Never raises."""
    try:
        from sauce.adapters.db import log_event

        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="error",
                payload={"source": "broker", "action": action, "error": str(exc)},
                timestamp=datetime.now(UTC),
            )
        )
    except Exception as log_exc:  # noqa: BLE001
        logger.error("Failed to log broker error to DB: %s", log_exc)
