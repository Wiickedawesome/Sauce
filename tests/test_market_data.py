"""
tests/test_market_data.py — Tests for adapters/market_data.py.

Mocks alpaca-py data clients — no real API calls.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sauce.core.schemas import PriceReference

# ── Env setup ─────────────────────────────────────────────────────────────────


def set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    from sauce.core.config import get_settings

    get_settings.cache_clear()


def clear_cache() -> None:
    from sauce.core.config import get_settings

    get_settings.cache_clear()


# ── Helpers ────────────────────────────────────────────────────────────────────


def make_bar_df(
    symbol: str = "AAPL",
    rows: int = 5,
    index_type: str = "single",
) -> pd.DataFrame:
    """
    Build a fake OHLCV DataFrame.

    index_type="single"  → simple DatetimeIndex (single symbol)
    index_type="multi"   → MultiIndex (symbol, timestamp) as alpaca-py returns
                           when requesting multiple symbols but we isolate one
    """
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="30min", tz="UTC")
    data = {
        "open": [100.0 + i for i in range(rows)],
        "high": [101.0 + i for i in range(rows)],
        "low": [99.0 + i for i in range(rows)],
        "close": [100.5 + i for i in range(rows)],
        "volume": [1000 + i * 10 for i in range(rows)],
    }

    if index_type == "multi":
        idx = pd.MultiIndex.from_product([[symbol], timestamps], names=["symbol", "timestamp"])
        return pd.DataFrame(data, index=idx)
    else:
        df = pd.DataFrame(data, index=timestamps)
        df.index.name = "timestamp"
        return df


def make_quote_mock(
    symbol: str = "AAPL",
    bid: float = 149.5,
    ask: float = 150.5,
    ts: datetime | None = None,
) -> MagicMock:
    """Build a mock quote object as returned by alpaca-py latest-quote calls."""
    q = MagicMock()
    q.bid_price = bid
    q.ask_price = ask
    q.timestamp = ts or datetime(2024, 1, 2, 15, 30, 0, tzinfo=UTC)
    return q


# ── get_quote — equity ────────────────────────────────────────────────────────


def test_get_quote_equity_returns_price_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    ts = datetime(2024, 1, 2, 15, 30, 0, tzinfo=UTC)
    mock_quote = make_quote_mock("AAPL", ts=ts)

    mock_client = MagicMock()
    mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    with patch("sauce.adapters.market_data._get_stock_client", return_value=mock_client):
        result = market_data.get_quote("AAPL")

    assert isinstance(result, PriceReference)
    assert result.symbol == "AAPL"
    assert result.as_of == ts
    clear_cache()


def test_get_quote_equity_as_of_comes_from_api_not_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """as_of must come from the quote object — never datetime.now() (Rule 2.4)."""
    set_env(monkeypatch)
    from sauce.adapters import market_data

    # Historical timestamp well in the past, proving it isn't datetime.now()
    historical_ts = datetime(2023, 6, 15, 14, 0, 0, tzinfo=UTC)
    mock_quote = make_quote_mock("AAPL", ts=historical_ts)

    mock_client = MagicMock()
    mock_client.get_stock_latest_quote.return_value = {"AAPL": mock_quote}

    with patch("sauce.adapters.market_data._get_stock_client", return_value=mock_client):
        result = market_data.get_quote("AAPL")

    assert result.as_of == historical_ts
    clear_cache()


# ── get_quote — crypto ────────────────────────────────────────────────────────


def test_get_quote_crypto_uses_crypto_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    ts = datetime(2024, 1, 2, 15, 30, 0, tzinfo=UTC)
    mock_quote = make_quote_mock("BTC/USD", bid=42000.0, ask=42100.0, ts=ts)

    stock_client = MagicMock()
    crypto_client = MagicMock()
    crypto_client.get_crypto_latest_quote.return_value = {"BTC/USD": mock_quote}

    with patch("sauce.adapters.market_data._get_stock_client", return_value=stock_client):
        with patch("sauce.adapters.market_data._get_crypto_client", return_value=crypto_client):
            result = market_data.get_quote("BTC/USD")

    stock_client.get_stock_latest_quote.assert_not_called()
    crypto_client.get_crypto_latest_quote.assert_called_once()
    assert isinstance(result, PriceReference)
    assert result.symbol == "BTC/USD"
    clear_cache()


# ── _parse_timestamp — None must raise, not use datetime.now() ────────────────


def test_parse_timestamp_raises_on_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Rule 2.4: Never substitute datetime.now() for a missing timestamp.
    If the API returns no timestamp, MarketDataError must be raised.
    """
    set_env(monkeypatch)
    from sauce.adapters.market_data import MarketDataError, _parse_timestamp

    with pytest.raises(MarketDataError, match="timestamp"):
        _parse_timestamp(None)
    clear_cache()


def test_parse_timestamp_returns_datetime_for_pd_timestamp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters.market_data import _parse_timestamp

    ts = pd.Timestamp("2024-01-02 15:30:00+00:00")
    result = _parse_timestamp(ts)
    assert isinstance(result, datetime)
    assert result.tzinfo is not None
    clear_cache()


def test_parse_timestamp_returns_datetime_for_iso_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters.market_data import _parse_timestamp

    result = _parse_timestamp("2024-01-02T15:30:00Z")
    assert isinstance(result, datetime)
    assert result.year == 2024
    clear_cache()


def test_parse_timestamp_raises_on_invalid_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters.market_data import MarketDataError, _parse_timestamp

    with pytest.raises(MarketDataError):
        _parse_timestamp("not-a-timestamp")
    clear_cache()


# ── get_history ───────────────────────────────────────────────────────────────


def test_get_history_equity_returns_dataframe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    mock_df = make_bar_df("AAPL", rows=10)
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value.df = mock_df

    with patch("sauce.adapters.market_data._get_stock_client", return_value=mock_client):
        result = market_data.get_history("AAPL", timeframe="30Min", bars=10)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
    assert "close" in result.columns
    clear_cache()


def test_get_history_crypto_uses_crypto_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    mock_df = make_bar_df("ETH/USD", rows=5)
    stock_client = MagicMock()
    crypto_client = MagicMock()
    crypto_client.get_crypto_bars.return_value.df = mock_df

    with patch("sauce.adapters.market_data._get_stock_client", return_value=stock_client):
        with patch("sauce.adapters.market_data._get_crypto_client", return_value=crypto_client):
            result = market_data.get_history("ETH/USD", timeframe="30Min", bars=5)

    stock_client.get_stock_bars.assert_not_called()
    crypto_client.get_crypto_bars.assert_called_once()
    assert isinstance(result, pd.DataFrame)
    clear_cache()


def test_get_history_returns_required_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    mock_df = make_bar_df("AAPL", rows=8)
    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value.df = mock_df

    with patch("sauce.adapters.market_data._get_stock_client", return_value=mock_client):
        result = market_data.get_history("AAPL")

    for col in ("open", "high", "low", "close", "volume"):
        assert col in result.columns
    clear_cache()


def test_get_history_raises_market_data_error_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters.market_data import MarketDataError, get_history

    mock_client = MagicMock()
    mock_client.get_stock_bars.side_effect = Exception("API timeout")

    with patch("sauce.adapters.market_data._get_stock_client", return_value=mock_client):
        with pytest.raises(MarketDataError, match="get_history"):
            get_history("AAPL")
    clear_cache()


# ── get_universe_snapshot ─────────────────────────────────────────────────────


def make_batch_quote(symbol: str, ts: datetime) -> MagicMock:
    q = MagicMock()
    q.bid_price = 100.0
    q.ask_price = 101.0
    q.timestamp = ts
    return q


def test_get_universe_snapshot_merges_equity_and_crypto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    ts = datetime(2024, 1, 2, 15, 30, 0, tzinfo=UTC)

    equity_snap = {
        "AAPL": make_batch_quote("AAPL", ts),
        "MSFT": make_batch_quote("MSFT", ts),
    }
    crypto_snap = {"BTC/USD": make_batch_quote("BTC/USD", ts)}

    mock_stock_client = MagicMock()
    mock_stock_client.get_stock_latest_quote.return_value = equity_snap
    mock_crypto_client = MagicMock()
    mock_crypto_client.get_crypto_latest_quote.return_value = crypto_snap

    with (
        patch("sauce.adapters.market_data._get_stock_client", return_value=mock_stock_client),
        patch("sauce.adapters.market_data._get_crypto_client", return_value=mock_crypto_client),
    ):
        result = market_data.get_universe_snapshot(["AAPL", "MSFT", "BTC/USD"])

    assert "AAPL" in result
    assert "MSFT" in result
    assert "BTC/USD" in result
    assert all(isinstance(v, PriceReference) for v in result.values())
    clear_cache()


def test_get_universe_snapshot_partial_failure_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If crypto snapshot raises, equity results should still be returned."""
    set_env(monkeypatch)
    from sauce.adapters import market_data

    ts = datetime(2024, 1, 2, 15, 30, 0, tzinfo=UTC)
    equity_snap = {"AAPL": make_batch_quote("AAPL", ts)}

    mock_stock_client = MagicMock()
    mock_stock_client.get_stock_latest_quote.return_value = equity_snap
    mock_crypto_client = MagicMock()
    mock_crypto_client.get_crypto_latest_quote.side_effect = Exception("Crypto API down")

    with (
        patch("sauce.adapters.market_data._get_stock_client", return_value=mock_stock_client),
        patch("sauce.adapters.market_data._get_crypto_client", return_value=mock_crypto_client),
    ):
        result = market_data.get_universe_snapshot(["AAPL", "BTC/USD"])

    assert "AAPL" in result
    assert isinstance(result["AAPL"], PriceReference)
    clear_cache()


def test_get_universe_snapshot_empty_list_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    result = market_data.get_universe_snapshot([])
    assert result == {}
    clear_cache()


def test_get_universe_snapshot_equity_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    ts = datetime(2024, 1, 2, 15, 30, 0, tzinfo=UTC)
    equity_snap = {"AAPL": make_batch_quote("AAPL", ts)}

    mock_stock_client = MagicMock()
    mock_stock_client.get_stock_latest_quote.return_value = equity_snap
    mock_crypto_client = MagicMock()

    with (
        patch("sauce.adapters.market_data._get_stock_client", return_value=mock_stock_client),
        patch("sauce.adapters.market_data._get_crypto_client", return_value=mock_crypto_client),
    ):
        result = market_data.get_universe_snapshot(["AAPL"])

    mock_crypto_client.get_crypto_latest_quote.assert_not_called()
    assert "AAPL" in result
    clear_cache()


def test_get_option_quotes_returns_price_references(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    ts = datetime(2024, 1, 2, 15, 30, 0, tzinfo=UTC)
    mock_quote = make_quote_mock("SPY250321C00550000", bid=5.0, ask=5.5, ts=ts)
    mock_client = MagicMock()
    mock_client.get_option_latest_quote.return_value = {"SPY250321C00550000": mock_quote}

    with patch("sauce.adapters.market_data._get_option_client", return_value=mock_client):
        result = market_data.get_option_quotes(["SPY250321C00550000"])

    assert "SPY250321C00550000" in result
    assert isinstance(result["SPY250321C00550000"], PriceReference)
    clear_cache()


def test_get_option_chain_normalizes_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch)
    from sauce.adapters import market_data

    contract = MagicMock()
    contract.symbol = "SPY250321C00550000"
    contract.type = "call"
    contract.strike_price = 550.0
    contract.expiration_date = datetime(2026, 4, 17, tzinfo=UTC).date()
    contract.open_interest = 250

    greeks = MagicMock()
    greeks.delta = 0.31
    quote = make_quote_mock(
        "SPY250321C00550000",
        bid=5.0,
        ask=5.4,
        ts=datetime(2026, 3, 24, 15, 30, 0, tzinfo=UTC),
    )
    snapshot = MagicMock()
    snapshot.latest_quote = quote
    snapshot.greeks = greeks
    snapshot.implied_volatility = 0.22

    contract_client = MagicMock()
    contract_client.get_option_contracts.return_value.option_contracts = [contract]
    option_client = MagicMock()
    option_client.get_option_chain.return_value = {"SPY250321C00550000": snapshot}

    with (
        patch("sauce.adapters.market_data._get_option_contract_client", return_value=contract_client),
        patch("sauce.adapters.market_data._get_option_client", return_value=option_client),
    ):
        result = market_data.get_option_chain("SPY", 540.0, "call")

    assert len(result) == 1
    assert result[0].contract_symbol == "SPY250321C00550000"
    assert result[0].delta == pytest.approx(0.31)
    assert result[0].bid == pytest.approx(5.0)
    clear_cache()
