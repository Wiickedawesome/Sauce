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
from datetime import datetime, timezone
from typing import Any

from sauce.adapters.utils import call_with_retry
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, Order, PriceReference

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
    from alpaca.trading.client import TradingClient  # type: ignore[import-untyped]

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

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="broker_call",
        payload={"action": "get_account"},
        timestamp=datetime.now(timezone.utc),
    ))

    try:
        client = _get_trading_client()
        account = call_with_retry(client.get_account)
        result: dict[str, Any] = account.__dict__

        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="broker_response",
            payload={
                "action": "get_account",
                "equity": str(getattr(account, "equity", "unknown")),
                "buying_power": str(getattr(account, "buying_power", "unknown")),
                "portfolio_value": str(getattr(account, "portfolio_value", "unknown")),
            },
            timestamp=datetime.now(timezone.utc),
        ))

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

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="broker_call",
        payload={"action": "get_positions"},
        timestamp=datetime.now(timezone.utc),
    ))

    try:
        client = _get_trading_client()
        positions = call_with_retry(client.get_all_positions)
        result: list[dict[str, Any]] = [p.__dict__ for p in positions]

        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="broker_response",
            payload={
                "action": "get_positions",
                "count": len(result),
                "symbols": [str(getattr(p, "symbol", "?")) for p in positions],
            },
            timestamp=datetime.now(timezone.utc),
        ))

        return result

    except Exception as exc:
        _log_broker_error(loop_id, "get_positions", exc)
        raise BrokerError(f"get_positions failed: {exc}") from exc


# ── Order placement ───────────────────────────────────────────────────────────

def place_order(order: Order, loop_id: str = "unset") -> dict[str, Any]:
    """
    Submit an order to Alpaca. Requires a fully-validated Order schema.

    This function must only be called from core/loop.py after Supervisor approval.
    Never call this directly from an agent.

    Returns a dict with Alpaca's order response (id, status, filled_qty, etc.).
    On any error: raises BrokerError — never retries automatically.
    """
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce  # type: ignore[import-untyped]
    from alpaca.trading.requests import (  # type: ignore[import-untyped]
        LimitOrderRequest,
        MarketOrderRequest,
        StopLimitOrderRequest,
        StopOrderRequest,
    )
    from sauce.adapters.db import log_event

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="broker_call",
        payload={
            "action": "place_order",
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "order_type": order.order_type,
            "time_in_force": order.time_in_force,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
        },
        timestamp=datetime.now(timezone.utc),
    ))

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
        effective_tif = "gtc" if "/" in order.symbol and order.time_in_force == "day" else order.time_in_force
        tif = tif_map[effective_tif]

        order_request: Any
        if order.order_type == "market":
            order_request = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=tif,
            )
        elif order.order_type == "limit":
            if order.limit_price is None:
                raise BrokerError("limit_price is required for limit orders")
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

        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="broker_response",
            payload={
                "action": "place_order",
                "symbol": order.symbol,
                "broker_order_id": str(getattr(submitted, "id", "unknown")),
                "status": str(getattr(submitted, "status", "unknown")),
            },
            timestamp=datetime.now(timezone.utc),
        ))

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

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="broker_call",
        payload={"action": "get_latest_quote", "symbol": symbol},
        timestamp=datetime.now(timezone.utc),
    ))

    try:
        # Delegate to market_data.get_quote() which enforces strict timestamp
        # handling (no datetime.now() fallback) — Findings 6.2 and 7.5.
        from sauce.adapters.market_data import MarketDataError, get_quote

        price_ref = get_quote(symbol)

        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="broker_response",
            payload={
                "action": "get_latest_quote",
                "symbol": symbol,
                "mid": price_ref.mid,
                "as_of": price_ref.as_of.isoformat(),
            },
            timestamp=datetime.now(timezone.utc),
        ))

        return price_ref

    except MarketDataError as exc:
        _log_broker_error(loop_id, f"get_latest_quote({symbol})", exc)
        raise BrokerError(f"get_latest_quote failed for {symbol}: {exc}") from exc
    except BrokerError:
        raise
    except Exception as exc:
        _log_broker_error(loop_id, f"get_latest_quote({symbol})", exc)
        raise BrokerError(f"get_latest_quote failed for {symbol}: {exc}") from exc


# ── Error logging helper ──────────────────────────────────────────────────────

def _log_broker_error(loop_id: str, action: str, exc: Exception) -> None:
    """Log a broker error to the audit DB. Never raises."""
    try:
        from sauce.adapters.db import log_event
        log_event(AuditEvent(
            loop_id=loop_id,
            event_type="error",
            payload={"source": "broker", "action": action, "error": str(exc)},
            timestamp=datetime.now(timezone.utc),
        ))
    except Exception as log_exc:  # noqa: BLE001
        logger.error("Failed to log broker error to DB: %s", log_exc)
