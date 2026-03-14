"""
tests/test_broker.py — Tests for adapters/broker.py.

Mocks alpaca-py TradingClient — no real API calls.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from sauce.core.schemas import Order


# ── Env setup ─────────────────────────────────────────────────────────────────

def set_env(monkeypatch: pytest.MonkeyPatch, paper: str = "true") -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ALPACA_PAPER", paper)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from sauce.core.config import get_settings
    get_settings.cache_clear()


# ── Build a minimal Order ─────────────────────────────────────────────────────

_TS = datetime(2024, 1, 2, 15, 30, 0, tzinfo=timezone.utc)
_PVER = "v1-test"


def make_limit_order(
    symbol: str = "AAPL",
    side: str = "buy",
    qty: int = 5,
    limit_price: float = 150.0,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        order_type="limit",
        qty=qty,
        limit_price=limit_price,
        as_of=_TS,
        prompt_version=_PVER,
    )


def make_market_order(
    symbol: str = "AAPL",
    side: str = "sell",
    qty: int = 3,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        order_type="market",
        qty=qty,
        as_of=_TS,
        prompt_version=_PVER,
    )


# ── Fake Alpaca account ───────────────────────────────────────────────────────

def make_fake_account() -> MagicMock:
    acct = MagicMock()
    acct.id = "acct-001"
    acct.cash = "10000.00"
    acct.portfolio_value = "25000.00"
    acct.buying_power = "10000.00"
    acct.equity = "25000.00"
    return acct


def make_fake_position(symbol: str = "AAPL") -> MagicMock:
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = "10"
    pos.market_value = "1800.00"
    pos.avg_entry_price = "175.00"
    pos.unrealized_pl = "50.00"
    return pos


def make_fake_order_response() -> MagicMock:
    resp = MagicMock()
    resp.id = "order-abc-123"
    resp.symbol = "AAPL"
    resp.qty = "5"
    resp.status = "accepted"
    return resp


# ── get_account ───────────────────────────────────────────────────────────────

def test_get_account_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch)
    from sauce.adapters import broker

    mock_client = MagicMock()
    mock_client.get_account.return_value = make_fake_account()

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            result = broker.get_account(loop_id="loop-001")

    assert isinstance(result, dict)
    assert "id" in result
    assert result["id"] == "acct-001"
    get_settings = __import__(
        "sauce.core.config", fromlist=["get_settings"]
    ).get_settings
    get_settings.cache_clear()


def test_get_account_logs_audit_events(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch)
    from sauce.adapters import broker

    mock_client = MagicMock()
    mock_client.get_account.return_value = make_fake_account()

    logged_events: list[str] = []

    def capture_log(event: object) -> None:
        from sauce.core.schemas import AuditEvent
        if isinstance(event, AuditEvent):
            logged_events.append(event.event_type)

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event", side_effect=capture_log):
            broker.get_account(loop_id="loop-001")

    assert "broker_call" in logged_events
    assert "broker_response" in logged_events


def test_get_account_raises_broker_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters.broker import BrokerError, get_account

    mock_client = MagicMock()
    mock_client.get_account.side_effect = Exception("API down")

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            with pytest.raises(BrokerError, match="get_account"):
                get_account(loop_id="fail-loop")


# ── get_positions ─────────────────────────────────────────────────────────────

def test_get_positions_returns_list(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch)
    from sauce.adapters import broker

    mock_client = MagicMock()
    mock_client.get_all_positions.return_value = [
        make_fake_position("AAPL"),
        make_fake_position("TSLA"),
    ]

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            result = broker.get_positions(loop_id="loop-001")

    assert isinstance(result, list)
    assert len(result) == 2


def test_get_positions_raises_broker_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters.broker import BrokerError, get_positions

    mock_client = MagicMock()
    mock_client.get_all_positions.side_effect = Exception("Timeout")

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            with pytest.raises(BrokerError, match="get_positions"):
                get_positions(loop_id="fail-loop")


# ── place_order ───────────────────────────────────────────────────────────────

def test_place_order_limit_calls_submit_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import broker

    order = make_limit_order()
    mock_client = MagicMock()
    mock_client.submit_order.return_value = make_fake_order_response()

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            result = broker.place_order(order=order, loop_id="loop-001")

    assert mock_client.submit_order.called
    assert result["id"] == "order-abc-123"


def test_place_order_market_calls_submit_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import broker

    order = make_market_order()
    mock_client = MagicMock()
    mock_client.submit_order.return_value = make_fake_order_response()

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            result = broker.place_order(order=order, loop_id="loop-001")

    assert mock_client.submit_order.called
    assert isinstance(result, dict)


def test_place_order_raises_on_api_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters.broker import BrokerError, place_order

    order = make_limit_order()
    mock_client = MagicMock()
    mock_client.submit_order.side_effect = Exception("Order rejected")

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            with pytest.raises(BrokerError, match="place_order"):
                place_order(order=order, loop_id="fail-loop")


def test_place_order_limit_missing_price_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A limit order with no limit_price should raise BrokerError before hitting Alpaca."""
    set_env(monkeypatch)
    from sauce.adapters.broker import BrokerError, place_order

    # Build an order with no limit_price
    order = Order(
        symbol="AAPL",
        side="buy",
        order_type="limit",
        qty=5,
        limit_price=None,
        as_of=_TS,
        prompt_version=_PVER,
    )

    mock_client = MagicMock()

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            with pytest.raises(BrokerError, match="limit_price"):
                place_order(order=order, loop_id="loop-noprice")


def test_place_order_logs_audit_events(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch)
    from sauce.adapters import broker

    order = make_limit_order()
    mock_client = MagicMock()
    mock_client.submit_order.return_value = make_fake_order_response()

    logged_events: list[str] = []

    def capture_log(event: object) -> None:
        from sauce.core.schemas import AuditEvent
        if isinstance(event, AuditEvent):
            logged_events.append(event.event_type)

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event", side_effect=capture_log):
            broker.place_order(order=order, loop_id="loop-001")

    assert "broker_call" in logged_events
    assert "broker_response" in logged_events


# ── paper=True is enforced ────────────────────────────────────────────────────

def test_trading_client_uses_paper_true_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch, paper="true")

    with patch("alpaca.trading.client.TradingClient") as mock_tc_class:
        mock_tc_class.return_value = MagicMock()
        import sauce.adapters.broker as broker_mod
        broker_mod._get_trading_client()

    _, kwargs = mock_tc_class.call_args
    assert kwargs.get("paper") is True


def test_trading_client_paper_false_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch, paper="false")
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", "LIVE-TRADING-CONFIRMED")

    with patch("alpaca.trading.client.TradingClient") as mock_tc_class:
        mock_tc_class.return_value = MagicMock()
        import sauce.adapters.broker as broker_mod
        broker_mod._get_trading_client()

    _, kwargs = mock_tc_class.call_args
    assert kwargs.get("paper") is False


# ── Crypto order dispatch ─────────────────────────────────────────────────────

def test_place_order_crypto_symbol_dispatches_correctly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crypto symbol (contains '/') should still reach submit_order."""
    set_env(monkeypatch)
    from sauce.adapters import broker

    order = Order(
        symbol="BTC/USD",
        side="buy",
        order_type="market",
        qty=1,
        as_of=_TS,
        prompt_version=_PVER,
    )
    mock_client = MagicMock()
    resp = MagicMock()
    resp.id = "crypto-order-001"
    resp.symbol = "BTC/USD"
    resp.qty = "1"
    resp.status = "accepted"
    mock_client.submit_order.return_value = resp

    with patch("sauce.adapters.broker._get_trading_client", return_value=mock_client):
        with patch("sauce.adapters.db.log_event"):
            result = broker.place_order(order=order, loop_id="loop-crypto")

    assert mock_client.submit_order.called
    assert result["symbol"] == "BTC/USD"
