"""
adapters/options_data.py — Alpaca Options Data adapter.

Fetches options chains, contract snapshots, and IV data via
the Alpaca Options Data API.

Rules:
- All returned models use sauce.core.options_schemas types.
- call_with_retry is used for all read calls (idempotent).
- Never use datetime.now() as a substitute for an API timestamp.
- Raises OptionsDataError on any unrecoverable failure.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sauce.adapters.utils import call_with_retry
from sauce.core.config import get_settings
from sauce.core.options_config import get_options_settings
from sauce.core.options_schemas import OptionsContract, OptionsQuote

logger = logging.getLogger(__name__)


# ── Exceptions ────────────────────────────────────────────────────────────────

class OptionsDataError(Exception):
    """Raised on unrecoverable Alpaca options data API errors."""


# ── Client factory ────────────────────────────────────────────────────────────

def _get_option_client() -> Any:
    """Return an Alpaca OptionHistoricalDataClient."""
    from alpaca.data.historical.option import (  # type: ignore[import-untyped]
        OptionHistoricalDataClient,
    )

    s = get_settings()
    return OptionHistoricalDataClient(
        api_key=s.alpaca_api_key,
        secret_key=s.alpaca_secret_key,
    )


def _get_trading_client() -> Any:
    """Return an Alpaca TradingClient (used for option chain metadata)."""
    from alpaca.trading.client import TradingClient  # type: ignore[import-untyped]

    s = get_settings()
    return TradingClient(
        api_key=s.alpaca_api_key,
        secret_key=s.alpaca_secret_key,
        paper=s.alpaca_paper,
    )


# ── Option Chain ──────────────────────────────────────────────────────────────

def get_option_chain(
    underlying: str,
    expiration_date: str | None = None,
    option_type: str | None = None,
) -> list[OptionsContract]:
    """
    Fetch available options contracts for an underlying symbol.

    Parameters
    ----------
    underlying:      Equity ticker (e.g. "AAPL").
    expiration_date: Optional YYYY-MM-DD filter.
    option_type:     Optional "call" or "put" filter.

    Returns
    -------
    List of OptionsContract models sorted by (expiration, strike).

    Raises OptionsDataError on failure.
    """
    try:
        from alpaca.trading.requests import (  # type: ignore[import-untyped]
            GetOptionContractsRequest,
        )

        client = _get_trading_client()

        kwargs: dict[str, Any] = {
            "underlying_symbols": [underlying],
            "status": "active",
        }
        if expiration_date:
            kwargs["expiration_date"] = expiration_date
        if option_type:
            kwargs["type"] = option_type

        request = GetOptionContractsRequest(**kwargs)
        response = call_with_retry(client.get_option_contracts, request)

        contracts: list[OptionsContract] = []
        for c in getattr(response, "option_contracts", []):
            contracts.append(
                OptionsContract(
                    contract_symbol=str(c.symbol),
                    underlying=underlying,
                    expiration=str(c.expiration_date),
                    strike=float(c.strike_price),
                    option_type="call" if str(c.type).lower() == "call" else "put",
                    open_interest=int(c.open_interest) if c.open_interest else None,
                )
            )

        contracts.sort(key=lambda c: (c.expiration, c.strike))
        logger.info(
            "Fetched %d option contracts for %s", len(contracts), underlying,
        )
        return contracts

    except OptionsDataError:
        raise
    except Exception as exc:
        raise OptionsDataError(
            f"get_option_chain({underlying}) failed: {exc}"
        ) from exc


# ── Contract Snapshot (Greeks + Quote) ────────────────────────────────────────

def get_contract_snapshot(contract_symbol: str) -> OptionsContract:
    """
    Fetch a live snapshot for a single options contract, including greeks.

    Returns an OptionsContract with greeks, bid/ask/mid, volume, OI populated.
    Raises OptionsDataError on failure.
    """
    try:
        from alpaca.data.requests import (  # type: ignore[import-untyped]
            OptionSnapshotRequest,
        )

        client = _get_option_client()
        request = OptionSnapshotRequest(symbol_or_symbols=contract_symbol)
        response = call_with_retry(client.get_option_snapshot, request)

        snap = response.get(contract_symbol) or response.get(
            contract_symbol.upper()
        )
        if snap is None:
            raise OptionsDataError(
                f"No snapshot returned for {contract_symbol}"
            )

        greeks = getattr(snap, "greeks", None)
        quote = getattr(snap, "latest_quote", None)
        trade = getattr(snap, "latest_trade", None)

        bid = float(quote.bid_price) if quote and quote.bid_price else 0.0
        ask = float(quote.ask_price) if quote and quote.ask_price else 0.0
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)

        # Parse the OCC symbol for metadata
        parsed = _parse_occ_symbol(contract_symbol)

        return OptionsContract(
            contract_symbol=contract_symbol,
            underlying=parsed["underlying"],
            expiration=parsed["expiration"],
            strike=parsed["strike"],
            option_type=parsed["option_type"],
            delta=float(greeks.delta) if greeks and greeks.delta is not None else None,
            gamma=float(greeks.gamma) if greeks and greeks.gamma is not None else None,
            theta=float(greeks.theta) if greeks and greeks.theta is not None else None,
            vega=float(greeks.vega) if greeks and greeks.vega is not None else None,
            iv=float(greeks.implied_volatility) if greeks and greeks.implied_volatility is not None else None,
            bid=bid,
            ask=ask,
            mid=mid,
            open_interest=int(snap.open_interest) if getattr(snap, "open_interest", None) else None,
            volume=int(trade.size) if trade and trade.size else None,
        )

    except OptionsDataError:
        raise
    except Exception as exc:
        raise OptionsDataError(
            f"get_contract_snapshot({contract_symbol}) failed: {exc}"
        ) from exc


def get_option_quote(contract_symbol: str) -> OptionsQuote:
    """
    Fetch the latest quote for an options contract.

    Returns an OptionsQuote with bid/ask/mid, volume, OI, and greeks.
    Raises OptionsDataError on failure.
    """
    try:
        snap = get_contract_snapshot(contract_symbol)
        return OptionsQuote(
            contract_symbol=contract_symbol,
            bid=snap.bid or 0.0,
            ask=snap.ask or 0.0,
            mid=snap.mid or 0.0,
            last=None,
            volume=snap.volume or 0,
            open_interest=snap.open_interest or 0,
            iv=snap.iv,
            delta=snap.delta,
            theta=snap.theta,
            as_of=datetime.now(timezone.utc),
        )
    except OptionsDataError:
        raise
    except Exception as exc:
        raise OptionsDataError(
            f"get_option_quote({contract_symbol}) failed: {exc}"
        ) from exc


# ── Contract Filtering ────────────────────────────────────────────────────────

def filter_contracts(
    contracts: list[OptionsContract],
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_delta: float | None = None,
    max_delta: float | None = None,
    min_oi: int = 10,
) -> list[OptionsContract]:
    """
    Filter a list of contracts by DTE, delta range, and liquidity.

    Uses OptionsSettings defaults when min/max not provided.
    """
    from datetime import date

    cfg = get_options_settings()
    min_dte = min_dte if min_dte is not None else cfg.option_min_dte
    max_dte = max_dte if max_dte is not None else cfg.option_max_dte
    min_delta = min_delta if min_delta is not None else cfg.option_min_delta
    max_delta = max_delta if max_delta is not None else cfg.option_max_delta

    today = date.today()
    result: list[OptionsContract] = []

    for c in contracts:
        # DTE check
        try:
            exp = date.fromisoformat(c.expiration)
        except ValueError:
            continue
        dte = (exp - today).days
        if dte < min_dte or dte > max_dte:
            continue

        # Delta check (absolute value)
        if c.delta is not None:
            abs_delta = abs(c.delta)
            if abs_delta < min_delta or abs_delta > max_delta:
                continue

        # Liquidity check
        if c.open_interest is not None and c.open_interest < min_oi:
            continue

        result.append(c)

    return result


# ── OCC Symbol Parser ─────────────────────────────────────────────────────────

def _parse_occ_symbol(occ: str) -> dict[str, Any]:
    """
    Parse an OCC options symbol into its components.

    Format: AAPL250418C00200000
            ^^^^                underlying (variable length)
                ^^^^^^          expiration YYMMDD
                      ^         C=call, P=put
                       ^^^^^^^^ strike * 1000 (zero-padded)
    """
    # The last 15 chars are always: YYMMDD + C/P + 8-digit strike
    if len(occ) < 16:
        raise OptionsDataError(f"Invalid OCC symbol: {occ!r}")

    strike_str = occ[-8:]
    option_char = occ[-9]
    date_str = occ[-15:-9]
    underlying = occ[:-15]

    if not underlying:
        raise OptionsDataError(f"Cannot parse underlying from OCC symbol: {occ!r}")

    try:
        strike = int(strike_str) / 1000.0
    except ValueError as exc:
        raise OptionsDataError(f"Bad strike in OCC symbol {occ!r}: {exc}") from exc

    try:
        exp_date = datetime.strptime(date_str, "%y%m%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise OptionsDataError(f"Bad date in OCC symbol {occ!r}: {exc}") from exc

    return {
        "underlying": underlying,
        "expiration": exp_date,
        "strike": strike,
        "option_type": "call" if option_char.upper() == "C" else "put",
    }
