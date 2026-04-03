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
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, cast

import pandas as pd

from sauce.adapters.utils import call_with_retry
from sauce.core.config import get_settings
from sauce.core.options_schemas import OptionContract
from sauce.core.schemas import PriceReference
from sauce.db import load_instrument_meta_extra, merge_instrument_meta_extra
from sauce.market_calendar import calendar_days_for_equity_bars

logger = logging.getLogger(__name__)

_SNAPSHOT_FAILURE_THRESHOLD = 3
_SNAPSHOT_SUPPRESS_FOR = timedelta(hours=6)
_SNAPSHOT_FAILURES_KEY = "snapshot_failures"
_SNAPSHOT_SUPPRESS_UNTIL_KEY = "snapshot_suppress_until"
_snapshot_state_cache: dict[str, tuple[int, datetime | None]] = {}


# ── Exceptions ────────────────────────────────────────────────────────────────


class MarketDataError(Exception):
    """Raised on unrecoverable Alpaca data API errors."""

    pass


# ── Client factories ──────────────────────────────────────────────────────────


def _get_stock_client() -> Any:
    from alpaca.data.historical import StockHistoricalDataClient

    s = get_settings()
    return StockHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)


def _get_crypto_client() -> Any:
    from alpaca.data.historical import CryptoHistoricalDataClient

    s = get_settings()
    return CryptoHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)


def _get_option_client() -> Any:
    from alpaca.data.historical.option import OptionHistoricalDataClient

    s = get_settings()
    return OptionHistoricalDataClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key)


def _get_option_contract_trading_client() -> Any:
    from alpaca.trading.client import TradingClient

    s = get_settings()
    return TradingClient(api_key=s.alpaca_api_key, secret_key=s.alpaca_secret_key, paper=s.alpaca_paper)


def _get_option_contract_client() -> Any:
    return _get_option_contract_trading_client()


def is_crypto(symbol: str) -> bool:
    """Return True for Alpaca crypto pairs (e.g. 'BTC/USD')."""
    return "/" in symbol


# Backward-compatible alias for any external callers.
_is_crypto = is_crypto


def _quote_is_fresh(quote: PriceReference, ttl_seconds: int, now: datetime | None = None) -> bool:
    ref_now = now or datetime.now(UTC)
    quote_time = quote.as_of if quote.as_of.tzinfo is not None else quote.as_of.replace(tzinfo=UTC)
    return (ref_now - quote_time).total_seconds() <= ttl_seconds


def _quote_ttl_seconds(symbol: str) -> int:
    settings = get_settings()
    return settings.crypto_data_ttl_seconds if _is_crypto(symbol) else settings.equity_data_ttl_seconds


def clear_snapshot_state_cache() -> None:
    _snapshot_state_cache.clear()


def _load_snapshot_state(symbol: str) -> tuple[int, datetime | None]:
    cached = _snapshot_state_cache.get(symbol)
    if cached is not None:
        return cached

    extra = load_instrument_meta_extra(symbol)
    failures_raw = extra.get(_SNAPSHOT_FAILURES_KEY, 0)
    suppress_raw = extra.get(_SNAPSHOT_SUPPRESS_UNTIL_KEY)

    try:
        failures = int(failures_raw)
    except (TypeError, ValueError):
        failures = 0

    suppressed_until: datetime | None = None
    if isinstance(suppress_raw, str) and suppress_raw:
        try:
            suppressed_until = datetime.fromisoformat(suppress_raw.replace("Z", "+00:00"))
            if suppressed_until.tzinfo is None:
                suppressed_until = suppressed_until.replace(tzinfo=UTC)
        except ValueError:
            suppressed_until = None

    state = (failures, suppressed_until)
    _snapshot_state_cache[symbol] = state
    return state


def _save_snapshot_state(symbol: str, failures: int, suppressed_until: datetime | None) -> None:
    state = (max(failures, 0), suppressed_until)
    _snapshot_state_cache[symbol] = state
    merge_instrument_meta_extra(
        symbol=symbol,
        asset_class="crypto" if _is_crypto(symbol) else "equity",
        extra_updates={
            _SNAPSHOT_FAILURES_KEY: state[0],
            _SNAPSHOT_SUPPRESS_UNTIL_KEY: suppressed_until.isoformat() if suppressed_until else None,
        },
    )


def _is_snapshot_suppressed(symbol: str, now: datetime) -> bool:
    failures, suppressed_until = _load_snapshot_state(symbol)
    if suppressed_until is None:
        return False
    if suppressed_until <= now:
        _save_snapshot_state(symbol, 0, None)
        return False
    return True


def _mark_snapshot_available(symbol: str) -> None:
    _save_snapshot_state(symbol, 0, None)


def _record_snapshot_failure(symbol: str, now: datetime) -> None:
    failures, _suppressed_until = _load_snapshot_state(symbol)
    failures += 1
    if failures < _SNAPSHOT_FAILURE_THRESHOLD:
        _save_snapshot_state(symbol, failures, None)
        return

    suppressed_until = now + _SNAPSHOT_SUPPRESS_FOR
    _save_snapshot_state(symbol, failures, suppressed_until)
    logger.warning(
        "Suppressing %s from batch snapshots until %s after %d consecutive quote failures",
        symbol,
        suppressed_until.isoformat(),
        failures,
    )


def get_snapshot_candidates(symbols: list[str], now: datetime | None = None) -> list[str]:
    """Return symbols that are not currently suppressed for entry batch scans."""
    ref_now = now or datetime.now(UTC)
    return [symbol for symbol in symbols if not _is_snapshot_suppressed(symbol, ref_now)]


def _recover_batch_quotes(
    symbols: list[str],
    result: dict[str, PriceReference],
    now: datetime,
) -> None:
    if not symbols:
        return

    missing_symbols = [symbol for symbol in symbols if symbol not in result]
    stale_symbols = [
        symbol for symbol, quote in result.items()
        if symbol in symbols and not _quote_is_fresh(quote, _quote_ttl_seconds(symbol), now)
    ]
    recovery_symbols = sorted(set(missing_symbols + stale_symbols))

    for symbol in recovery_symbols:
        try:
            recovered_quote = get_quote(symbol)
        except MarketDataError:
            _record_snapshot_failure(symbol, now)
            continue

        result[symbol] = recovered_quote
        if _quote_is_fresh(recovered_quote, _quote_ttl_seconds(symbol), now):
            _mark_snapshot_available(symbol)
        else:
            _record_snapshot_failure(symbol, now)


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
    from alpaca.data.requests import StockLatestQuoteRequest

    s = get_settings()
    client = _get_stock_client()
    response = call_with_retry(
        client.get_stock_latest_quote,
        StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=s.data_feed),  # type: ignore[arg-type]
    )
    quote = response[symbol]

    bid = float(quote.bid_price or 0.0)
    ask = float(quote.ask_price or 0.0)
    mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
    as_of: datetime = _parse_timestamp(getattr(quote, "timestamp", None))

    return PriceReference(symbol=symbol, bid=bid, ask=ask, mid=mid, as_of=as_of)


def _crypto_quote(symbol: str) -> PriceReference:
    from alpaca.data.requests import CryptoLatestQuoteRequest

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
    timeframe: str = "15Min",
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
    from alpaca.data.requests import StockBarsRequest

    tf = _parse_timeframe(timeframe)
    start = datetime.now(UTC) - timedelta(days=_days_back_for_bars(timeframe, bars, is_crypto=False))

    s = get_settings()
    client = _get_stock_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
        adjustment="raw",  # type: ignore[arg-type]
        feed=s.data_feed,  # type: ignore[arg-type]
    )
    bars_response = call_with_retry(client.get_stock_bars, request)
    df = bars_response.df

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])

    df = _normalise_bars_df(df, symbol)
    return df.tail(bars)


def _crypto_history(symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    from alpaca.data.requests import CryptoBarsRequest

    tf = _parse_timeframe(timeframe)
    start = datetime.now(UTC) - timedelta(days=_days_back_for_bars(timeframe, bars, is_crypto=True))

    client = _get_crypto_client()
    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
    )
    bars_response = call_with_retry(client.get_crypto_bars, request)
    df = bars_response.df

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "timestamp"])

    df = _normalise_bars_df(df, symbol)
    return df.tail(bars)


# ── get_universe_snapshot ─────────────────────────────────────────────────────


def get_universe_snapshot(
    symbols: list[str],
    *,
    respect_suppression: bool = True,
) -> dict[str, PriceReference]:
    """
    Fetch latest quotes for a batch of symbols in one API call (where possible).

    Equities and crypto symbols are split and fetched separately, then merged.
    Returns a dict mapping symbol → PriceReference.

    Symbols that fail to fetch are logged and omitted from the result.
    This means a partial result is possible — consumers must check presence.
    """
    now = datetime.now(UTC)
    active_symbols = (
        [s for s in symbols if not _is_snapshot_suppressed(s, now)]
        if respect_suppression
        else list(symbols)
    )

    if not active_symbols:
        logger.info("All %d requested symbols are temporarily suppressed from snapshot retries", len(symbols))
        return {}

    equity_symbols = [s for s in active_symbols if not _is_crypto(s)]
    crypto_symbols = [s for s in active_symbols if _is_crypto(s)]

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

    _recover_batch_quotes(active_symbols, result, now)

    if active_symbols and not result:
        raise MarketDataError(
            f"get_universe_snapshot({active_symbols}) returned no quotes for {len(active_symbols)} requested symbols"
        )

    return result


def _equity_snapshot(symbols: list[str]) -> dict[str, PriceReference]:
    from alpaca.data.requests import StockLatestQuoteRequest

    s = get_settings()
    client = _get_stock_client()
    response = call_with_retry(
        client.get_stock_latest_quote,
        StockLatestQuoteRequest(symbol_or_symbols=symbols, feed=s.data_feed),  # type: ignore[arg-type]
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
    from alpaca.data.requests import CryptoLatestQuoteRequest

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
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import AssetClass, AssetStatus
    from alpaca.trading.requests import GetAssetsRequest

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
            "symbol": str(a.symbol),  # type: ignore[union-attr]
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
    from alpaca.data.requests import StockBarsRequest

    if not symbols:
        return {}

    tf = _parse_timeframe(timeframe)
    start = datetime.now(UTC) - timedelta(days=_days_back_for_bars(timeframe, bars, is_crypto=False))

    s = get_settings()
    client = _get_stock_client()
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=tf,
        start=start,
        adjustment="raw",  # type: ignore[arg-type]
        feed=s.data_feed,  # type: ignore[arg-type]
    )
    bars_response = call_with_retry(client.get_stock_bars, request)
    df = bars_response.df

    if df.empty:
        return {}

    result: dict[str, pd.DataFrame] = {}
    if isinstance(df.index, pd.MultiIndex):
        for sym in df.index.get_level_values(0).unique():
            result[sym] = _normalise_bars_df(df, sym).tail(bars)
    else:
        # Single symbol returned without MultiIndex
        if symbols:
            result[symbols[0]] = _normalise_bars_df(df, symbols[0]).tail(bars)

    return result


def get_option_chain(
    underlying: str,
    current_price: float,
    option_type: str | None = None,
) -> list[OptionContract]:
    """Fetch option contracts for an underlying and enrich them with quotes and greeks."""
    from alpaca.data.requests import OptionChainRequest
    from alpaca.trading.enums import ContractType
    from alpaca.trading.requests import GetOptionContractsRequest

    settings = get_settings()
    option_type_enum = None
    if option_type == "call":
        option_type_enum = ContractType.CALL
    elif option_type == "put":
        option_type_enum = ContractType.PUT

    contract_client = _get_option_contract_client()
    option_client = _get_option_client()
    today = datetime.now(UTC).date()
    strike_low = max(current_price * 0.75, 0.01)
    strike_high = max(current_price * 1.25, strike_low + 1)

    contracts_response = call_with_retry(
        contract_client.get_option_contracts,
        GetOptionContractsRequest(
            underlying_symbols=[underlying],
            type=option_type_enum,
            expiration_date_gte=today + timedelta(days=settings.options_dte_min),
            expiration_date_lte=today + timedelta(days=settings.options_dte_max),
            strike_price_gte=f"{strike_low:.2f}",
            strike_price_lte=f"{strike_high:.2f}",
            limit=200,
        ),
    )
    contracts = getattr(contracts_response, "option_contracts", None) or []

    snapshots = call_with_retry(
        option_client.get_option_chain,
        OptionChainRequest(
            underlying_symbol=underlying,
            type=option_type_enum,
            strike_price_gte=strike_low,
            strike_price_lte=strike_high,
            expiration_date_gte=today + timedelta(days=settings.options_dte_min),
            expiration_date_lte=today + timedelta(days=settings.options_dte_max),
        ),
    )

    result: list[OptionContract] = []
    for contract in contracts:
        symbol = str(getattr(contract, "symbol", ""))
        if not symbol:
            continue
        snapshot = snapshots.get(symbol) if isinstance(snapshots, dict) else None
        latest_quote = getattr(snapshot, "latest_quote", None)
        greeks = getattr(snapshot, "greeks", None)
        bid = float(getattr(latest_quote, "bid_price", 0.0) or 0.0)
        ask = float(getattr(latest_quote, "ask_price", 0.0) or 0.0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)
        expiration = getattr(contract, "expiration_date", today)
        dte = max((expiration - today).days, 0)
        result.append(
            OptionContract(
                underlying=underlying,
                contract_symbol=symbol,
                option_type=cast(Literal["call", "put"], str(getattr(contract, "type", "")).lower()),
                strike=float(getattr(contract, "strike_price", 0.0) or 0.0),
                expiration=expiration,
                dte=dte,
                delta=(
                    float(getattr(greeks, "delta", 0.0))
                    if greeks and getattr(greeks, "delta", None) is not None
                    else None
                ),
                implied_volatility=(
                    float(getattr(snapshot, "implied_volatility", 0.0))
                    if snapshot and getattr(snapshot, "implied_volatility", None) is not None
                    else None
                ),
                bid=bid if bid > 0 else None,
                ask=ask if ask > 0 else None,
                mid=mid if mid > 0 else None,
                open_interest=int(getattr(contract, "open_interest", 0) or 0),
                volume=None,
            )
        )

    return result


def get_option_quotes(symbols: list[str]) -> dict[str, PriceReference]:
    """Fetch latest bid/ask/mid quotes for option contracts."""
    from alpaca.data.requests import OptionLatestQuoteRequest

    if not symbols:
        return {}

    client = _get_option_client()
    response = call_with_retry(
        client.get_option_latest_quote,
        OptionLatestQuoteRequest(symbol_or_symbols=symbols),
    )

    result: dict[str, PriceReference] = {}
    for symbol, quote in response.items():
        bid = float(getattr(quote, "bid_price", 0.0) or 0.0)
        ask = float(getattr(quote, "ask_price", 0.0) or 0.0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)
        as_of = _parse_timestamp(getattr(quote, "timestamp", None))
        result[symbol] = PriceReference(symbol=symbol, bid=bid, ask=ask, mid=mid, as_of=as_of)

    return result


def get_option_quote(symbol: str) -> PriceReference:
    """Fetch the latest quote for a single option contract."""
    quotes = get_option_quotes([symbol])
    if symbol not in quotes:
        raise MarketDataError(f"No option quote returned for {symbol}")
    return quotes[symbol]


# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_timestamp(ts: Any) -> datetime:
    """
    Parse an API timestamp into a UTC-aware datetime.

    If ts is None or unparseable, raises MarketDataError — never silently
    substitutes datetime.now() which would mask stale data.
    """
    if ts is None:
        raise MarketDataError("API returned no timestamp. Treating data as stale — no trade.")
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=UTC)
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
    from alpaca.data.timeframe import TimeFrame

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
            f"Unsupported timeframe '{timeframe}'. Use one of: {list(mapping.keys())}"
        )
    return tf


def _days_back_for_bars(timeframe: str, bars: int, *, is_crypto: bool = False) -> int:
    """
    Estimate how many calendar days we need to fetch `bars` bars of `timeframe`.

    Crypto trades 24/7 (1440 min/day). Equities trade ~6.5h/day (390 min/day)
    with weekends off, so we add a 2x buffer. Minimum of 2 days.
    """
    minutes_per_bar = {
        "1Min": 1,
        "5Min": 5,
        "15Min": 15,
        "30Min": 30,
        "1Hour": 60,
        "4Hour": 240,
        "1Day": 1440 if is_crypto else 390,
    }
    mins = minutes_per_bar.get(timeframe, 30)
    if is_crypto:
        # Crypto: 24h/day, 7 days/week — exact calculation + small buffer
        days_needed = max(2, int(bars * mins / 1440) + 2)
    else:
        days_needed = calendar_days_for_equity_bars(
            datetime.now(UTC).date(),
            bars,
            mins if timeframe != "1Day" else 390,
        )
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
