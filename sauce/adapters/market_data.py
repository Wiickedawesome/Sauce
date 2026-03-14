"""
adapters/market_data.py — Alpaca Data API adapter.

Fetches OHLCV history, latest quotes, and universe snapshots for both
US equities (ticker symbols) and crypto pairs (e.g. BTC/USD).

Rules:
- NEVER use datetime.now() as a substitute for a real API timestamp.
- All returned PriceReference objects include a real as_of from the API.
- is_data_fresh() must be called by the consumer before using any returned data.
- Both equities and crypto are supported — asset type detected by "/" in symbol.

NOTE: Verify alpaca-py field names against https://alpaca.markets/docs/api-references/market-data-api/
      before going live. SDK versions may change.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from sauce.adapters.utils import call_with_retry
from sauce.core.config import get_settings
from sauce.core.schemas import PriceReference

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class MarketDataError(Exception):
    """Raised on unrecoverable Alpaca data API errors."""
    pass


# ── Client factories ──────────────────────────────────────────────────────────

def _get_stock_client() -> Any:
    from alpaca.data.historical import StockHistoricalDataClient  # type: ignore[import-untyped]
    s = get_settings()
    return StockHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)


def _get_crypto_client() -> Any:
    from alpaca.data.historical import CryptoHistoricalDataClient  # type: ignore[import-untyped]
    s = get_settings()
    return CryptoHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)


def _is_crypto(symbol: str) -> bool:
    return "/" in symbol


# ── get_quote ─────────────────────────────────────────────────────────────────

def get_quote(symbol: str) -> PriceReference:
    """
    Fetch the latest bid/ask/mid for a symbol.

    Works for both equities and crypto pairs.
    The as_of field is set from the API timestamp — never datetime.now().

    Raises MarketDataError on failure.
    """
    try:
        if _is_crypto(symbol):
            return _crypto_quote(symbol)
        return _equity_quote(symbol)
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"get_quote({symbol}) failed: {exc}") from exc


def _equity_quote(symbol: str) -> PriceReference:
    from alpaca.data.requests import StockLatestQuoteRequest  # type: ignore[import-untyped]

    s = get_settings()
    client = _get_stock_client()
    response = call_with_retry(
        client.get_stock_latest_quote,
        StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=s.data_feed),
    )
    quote = response[symbol]

    bid = float(quote.bid_price or 0.0)
    ask = float(quote.ask_price or 0.0)
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
    as_of: datetime = _parse_timestamp(getattr(quote, "timestamp", None))

    return PriceReference(symbol=symbol, bid=bid, ask=ask, mid=mid, as_of=as_of)


def _crypto_quote(symbol: str) -> PriceReference:
    from alpaca.data.requests import CryptoLatestQuoteRequest  # type: ignore[import-untyped]

    client = _get_crypto_client()
    response = call_with_retry(
        client.get_crypto_latest_quote,
        CryptoLatestQuoteRequest(symbol_or_symbols=symbol),
    )
    quote = response[symbol]

    bid = float(quote.bid_price or 0.0)
    ask = float(quote.ask_price or 0.0)
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
    as_of = _parse_timestamp(getattr(quote, "timestamp", None))

    return PriceReference(symbol=symbol, bid=bid, ask=ask, mid=mid, as_of=as_of)


# ── get_history ───────────────────────────────────────────────────────────────

def get_history(
    symbol: str,
    timeframe: str = "30Min",
    bars: int = 100,
) -> pd.DataFrame:
    """
    Fetch OHLCV price history for a symbol.

    Parameters
    ----------
    symbol:    Equity ticker or crypto pair (BTC/USD).
    timeframe: Alpaca TimeFrame string. Common values:
               "1Min", "5Min", "15Min", "30Min", "1Hour", "1Day".
    bars:      Number of bars to fetch (most recent N bars).

    Returns
    -------
    pandas DataFrame with columns: open, high, low, close, volume, timestamp.
    Indexed by timestamp (UTC, timezone-aware).
    Empty DataFrame if no data is available.

    Raises MarketDataError on API failure.
    """
    try:
        if _is_crypto(symbol):
            return _crypto_history(symbol, timeframe, bars)
        return _equity_history(symbol, timeframe, bars)
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"get_history({symbol}) failed: {exc}") from exc


def _equity_history(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    from alpaca.data.requests import StockBarsRequest  # type: ignore[import-untyped]
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore[import-untyped]

    tf = _parse_timeframe(timeframe)
    start = datetime.now(timezone.utc) - timedelta(days=_days_back_for_bars(timeframe, bars))

    s = get_settings()
    client = _get_stock_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        limit=bars,
        adjustment="raw",
        feed=s.data_feed,
    )
    bars_response = call_with_retry(client.get_stock_bars, request)
    df = bars_response.df

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])

    return _normalise_bars_df(df, symbol)


def _crypto_history(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    from alpaca.data.requests import CryptoBarsRequest  # type: ignore[import-untyped]

    tf = _parse_timeframe(timeframe)
    start = datetime.now(timezone.utc) - timedelta(days=_days_back_for_bars(timeframe, bars))

    client = _get_crypto_client()
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        limit=bars,
    )
    bars_response = call_with_retry(client.get_crypto_bars, request)
    df = bars_response.df

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])

    return _normalise_bars_df(df, symbol)


# ── get_universe_snapshot ─────────────────────────────────────────────────────

def get_universe_snapshot(symbols: list[str]) -> dict[str, PriceReference]:
    """
    Fetch latest quotes for a batch of symbols in one API call (where possible).

    Equities and crypto symbols are split and fetched separately, then merged.
    Returns a dict mapping symbol → PriceReference.

    Symbols that fail to fetch are logged and omitted from the result.
    This means a partial result is possible — consumers must check presence.
    """
    equity_symbols = [s for s in symbols if not _is_crypto(s)]
    crypto_symbols = [s for s in symbols if _is_crypto(s)]

    result: dict[str, PriceReference] = {}

    if equity_symbols:
        try:
            equity_quotes = _equity_snapshot(equity_symbols)
            result.update(equity_quotes)
        except Exception as exc:
            logger.error("Equity snapshot failed for %s: %s", equity_symbols, exc)

    if crypto_symbols:
        try:
            crypto_quotes = _crypto_snapshot(crypto_symbols)
            result.update(crypto_quotes)
        except Exception as exc:
            logger.error("Crypto snapshot failed for %s: %s", crypto_symbols, exc)

    return result


def _equity_snapshot(symbols: list[str]) -> dict[str, PriceReference]:
    from alpaca.data.requests import StockLatestQuoteRequest  # type: ignore[import-untyped]

    s = get_settings()
    client = _get_stock_client()
    response = call_with_retry(
        client.get_stock_latest_quote,
        StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=s.data_feed),
    )

    result: dict[str, PriceReference] = {}
    for symbol, quote in response.items():
        bid = float(quote.bid_price or 0.0)
        ask = float(quote.ask_price or 0.0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
        as_of = _parse_timestamp(getattr(quote, "timestamp", None))
        result[symbol] = PriceReference(symbol=symbol, bid=bid, ask=ask, mid=mid, as_of=as_of)

    return result


def _crypto_snapshot(symbols: list[str]) -> dict[str, PriceReference]:
    from alpaca.data.requests import CryptoLatestQuoteRequest  # type: ignore[import-untyped]

    client = _get_crypto_client()
    response = call_with_retry(
        client.get_crypto_latest_quote,
        CryptoLatestQuoteRequest(symbol_or_symbols=symbols),
    )

    result: dict[str, PriceReference] = {}
    for symbol, quote in response.items():
        bid = float(quote.bid_price or 0.0)
        ask = float(quote.ask_price or 0.0)
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
        as_of = _parse_timestamp(getattr(quote, "timestamp", None))
        result[symbol] = PriceReference(symbol=symbol, bid=bid, ask=ask, mid=mid, as_of=as_of)

    return result


def get_active_equity_assets() -> list[dict[str, Any]]:
    """
    Fetch all active, tradeable US equity assets from Alpaca.

    Returns a list of dicts with keys: symbol, name, exchange, status, tradable.
    Uses the TradingClient (1 API call).
    """
    from alpaca.trading.client import TradingClient  # type: ignore[import-untyped]
    from alpaca.trading.requests import GetAssetsRequest  # type: ignore[import-untyped]
    from alpaca.trading.enums import AssetClass, AssetStatus  # type: ignore[import-untyped]

    s = get_settings()
    client = TradingClient(
        api_key=s.alpaca_api_key,
        secret_key=s.alpaca_secret_key,
        paper=s.alpaca_paper,
    )
    request = GetAssetsRequest(asset_class=AssetClass.US_EQUITY, status=AssetStatus.ACTIVE)
    assets = call_with_retry(client.get_all_assets, request)
    return [
        {
            "symbol": str(a.symbol),
            "name": str(getattr(a, "name", "")),
            "exchange": str(getattr(a, "exchange", "")),
            "tradable": bool(getattr(a, "tradable", False)),
        }
        for a in assets
        if getattr(a, "tradable", False)
    ]


def get_bulk_equity_bars(
    symbols: list[str],
    timeframe: str = "1Day",
    bars: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV bars for multiple equity symbols in a single API call.

    Returns a dict mapping symbol → DataFrame (same format as get_history).
    Symbols without data are silently omitted.
    """
    from alpaca.data.requests import StockBarsRequest  # type: ignore[import-untyped]

    if not symbols:
        return {}

    tf = _parse_timeframe(timeframe)
    start = datetime.now(timezone.utc) - timedelta(days=_days_back_for_bars(timeframe, bars))

    s = get_settings()
    client = _get_stock_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start,
        limit=bars,
        adjustment="raw",
        feed=s.data_feed,
    )
    bars_response = call_with_retry(client.get_stock_bars, request)
    df = bars_response.df

    if df.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(df.index, pd.MultiIndex):
        for sym in df.index.get_level_values(0).unique():
            result[sym] = _normalise_bars_df(df, sym)
    else:
        # Single symbol returned without MultiIndex
        if symbols:
            result[symbols[0]] = _normalise_bars_df(df, symbols[0])

    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_timestamp(ts: Any) -> datetime:
    """
    Parse an API timestamp into a UTC-aware datetime.

    If ts is None or unparseable, raises MarketDataError — never silently
    substitutes datetime.now() which would mask stale data.
    """
    if ts is None:
        raise MarketDataError(
            "API returned no timestamp. Treating data as stale — no trade."
        )
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts
    if isinstance(ts, str):
        try:
            parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            return parsed
        except ValueError as exc:
            raise MarketDataError(f"Unparseable timestamp '{ts}': {exc}") from exc
    raise MarketDataError(f"Unexpected timestamp type {type(ts).__name__}: {ts!r}")


def _parse_timeframe(timeframe: str) -> Any:
    """
    Convert a human-readable timeframe string to an alpaca-py TimeFrame object.

    Supported: "1Min", "5Min", "15Min", "30Min", "1Hour", "4Hour", "1Day".
    """
    from alpaca.data.timeframe import TimeFrame  # type: ignore[import-untyped]

    mapping: dict[str, Any] = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrame.Minute.unit),
        "15Min": TimeFrame(15, TimeFrame.Minute.unit),
        "30Min": TimeFrame(30, TimeFrame.Minute.unit),
        "1Hour": TimeFrame.Hour,
        "4Hour": TimeFrame(4, TimeFrame.Hour.unit),
        "1Day": TimeFrame.Day,
    }
    tf = mapping.get(timeframe)
    if tf is None:
        raise MarketDataError(
            f"Unsupported timeframe '{timeframe}'. "
            f"Use one of: {list(mapping.keys())}"
        )
    return tf


def _days_back_for_bars(timeframe: str, bars: int) -> int:
    """
    Estimate how many calendar days we need to fetch `bars` bars of `timeframe`.

    Adds a 2x buffer for weekends/holidays. Minimum of 5 days.
    """
    minutes_per_bar = {
        "1Min": 1, "5Min": 5, "15Min": 15, "30Min": 30,
        "1Hour": 60, "4Hour": 240, "1Day": 390,
    }
    mins = minutes_per_bar.get(timeframe, 30)
    # US market: ~390 trading minutes/day → ~6.5 hours
    trading_minutes_per_day = 390
    days_needed = max(5, int((bars * mins / trading_minutes_per_day) * 2) + 5)
    return days_needed


def _normalise_bars_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Normalise an alpaca-py bars DataFrame to a consistent column set.

    alpaca-py returns a MultiIndex DataFrame when multiple symbols are requested.
    This function extracts a single symbol and renames columns to lowercase.
    """
    # If MultiIndex (symbol, timestamp), select the symbol level
    if isinstance(df.index, pd.MultiIndex):
        if symbol in df.index.get_level_values(0):
            df = df.loc[symbol]
        else:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    # Normalise column names to lowercase
    df = df.rename(columns=str.lower)

    # Ensure required columns exist
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("History for %s missing columns: %s", symbol, missing)

    return df
