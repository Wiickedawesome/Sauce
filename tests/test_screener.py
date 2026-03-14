"""
tests/test_screener.py — Tests for core/screener.py dynamic equity screener.

All Alpaca API calls are mocked — no real network calls.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sauce.core.config import get_settings


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_quote(symbol: str, bid: float, ask: float) -> MagicMock:
    """Return a mock quote object with bid/ask/mid."""
    q = MagicMock()
    q.bid = bid
    q.ask = ask
    q.mid = (bid + ask) / 2
    return q


def _set_wide_spread(monkeypatch):
    """Set MAX_SPREAD_PCT high enough so test quotes pass the spread filter."""
    monkeypatch.setenv("MAX_SPREAD_PCT", "0.10")


def _make_assets(*symbols: str) -> list[dict]:
    return [
        {"symbol": s, "name": s, "exchange": "NASDAQ", "tradable": True}
        for s in symbols
    ]


def _make_bars_df(close: float, volume: float, rows: int = 5) -> pd.DataFrame:
    """Return a DataFrame mimicking Alpaca bars with close/volume columns."""
    return pd.DataFrame({
        "open": [close] * rows,
        "high": [close] * rows,
        "low": [close] * rows,
        "close": [close] * rows,
        "volume": [volume] * rows,
    })


# ── Patch prefixes ───────────────────────────────────────────────────────────

_P = "sauce.core.screener"


# ── Basic screening pipeline ─────────────────────────────────────────────────

def test_screen_equities_returns_top_candidates(monkeypatch):
    """Full happy-path: 3 stocks, all pass filters, top-2 returned by max_candidates."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("SCREENER_MAX_CANDIDATES", "2")
    monkeypatch.setenv("SCREENER_MIN_DOLLAR_VOLUME", "1000")
    monkeypatch.setenv("SCREENER_PRICE_MIN", "1")
    monkeypatch.setenv("SCREENER_PRICE_MAX", "9999")
    _set_wide_spread(monkeypatch)
    get_settings.cache_clear()

    assets = _make_assets("AAPL", "MSFT", "GOOG")
    quotes = {
        "AAPL": _make_quote("AAPL", 149.0, 151.0),  # mid=150, spread=1.33%
        "MSFT": _make_quote("MSFT", 299.0, 301.0),  # mid=300, spread=0.67%
        "GOOG": _make_quote("GOOG", 99.0, 101.0),   # mid=100, spread=2.0%
    }
    bars = {
        "AAPL": _make_bars_df(150.0, 50_000),  # dv = 150*50000 = 7.5M
        "MSFT": _make_bars_df(300.0, 80_000),  # dv = 300*80000 = 24M
        "GOOG": _make_bars_df(100.0, 20_000),  # dv = 100*20000 = 2M
    }

    with patch(f"{_P}.get_active_equity_assets", return_value=assets):
        with patch(f"{_P}.get_universe_snapshot", return_value=quotes):
            with patch(f"{_P}.get_bulk_equity_bars", return_value=bars):
                from sauce.core.screener import screen_equities
                result = screen_equities(regime="TRENDING_UP")

    # MSFT should rank highest (most $ volume), AAPL second
    assert result[0] == "MSFT"
    assert result[1] == "AAPL"
    # GOOG should still appear because it's in always-include (.env list)
    # (the default equity_universe includes it if set, otherwise it won't)


def test_screen_equities_price_filter(monkeypatch):
    """Stocks outside price range are excluded."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("SCREENER_MAX_CANDIDATES", "10")
    monkeypatch.setenv("SCREENER_MIN_DOLLAR_VOLUME", "0")
    monkeypatch.setenv("SCREENER_PRICE_MIN", "10")
    monkeypatch.setenv("SCREENER_PRICE_MAX", "200")
    _set_wide_spread(monkeypatch)
    get_settings.cache_clear()

    assets = _make_assets("CHEAP", "OK", "EXPENSIVE")
    quotes = {
        "CHEAP": _make_quote("CHEAP", 2.0, 3.0),        # mid=2.5 — below min
        "OK": _make_quote("OK", 49.0, 51.0),             # mid=50 — in range
        "EXPENSIVE": _make_quote("EXPENSIVE", 999, 1001), # mid=1000 — above max
    }
    bars = {
        "OK": _make_bars_df(50.0, 100_000),
    }

    with patch(f"{_P}.get_active_equity_assets", return_value=assets):
        with patch(f"{_P}.get_universe_snapshot", return_value=quotes):
            with patch(f"{_P}.get_bulk_equity_bars", return_value=bars):
                from sauce.core.screener import screen_equities
                result = screen_equities(regime="TRENDING_UP")

    screened_set = set(result)
    assert "OK" in screened_set
    assert "CHEAP" not in screened_set or "CHEAP" in {s for s in get_settings().equity_universe}
    assert "EXPENSIVE" not in screened_set or "EXPENSIVE" in {s for s in get_settings().equity_universe}


def test_screen_equities_dollar_volume_filter(monkeypatch):
    """Stocks below min dollar volume are excluded (unless always-include)."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("SCREENER_MAX_CANDIDATES", "10")
    monkeypatch.setenv("SCREENER_MIN_DOLLAR_VOLUME", "1000000")
    monkeypatch.setenv("SCREENER_PRICE_MIN", "1")
    monkeypatch.setenv("SCREENER_PRICE_MAX", "9999")
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "")
    _set_wide_spread(monkeypatch)
    get_settings.cache_clear()

    assets = _make_assets("LIQUID", "ILLIQUID")
    quotes = {
        "LIQUID": _make_quote("LIQUID", 99.0, 101.0),
        "ILLIQUID": _make_quote("ILLIQUID", 49.0, 51.0),
    }
    bars = {
        "LIQUID": _make_bars_df(100.0, 50_000),   # dv = 5M — passes
        "ILLIQUID": _make_bars_df(50.0, 100),      # dv = 5K — fails
    }

    with patch(f"{_P}.get_active_equity_assets", return_value=assets):
        with patch(f"{_P}.get_universe_snapshot", return_value=quotes):
            with patch(f"{_P}.get_bulk_equity_bars", return_value=bars):
                from sauce.core.screener import screen_equities
                result = screen_equities(regime="TRENDING_UP")

    assert "LIQUID" in result
    assert "ILLIQUID" not in result


def test_screen_equities_always_includes_env_list(monkeypatch):
    """Symbols from .env equity_universe always appear even if not screened."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("SCREENER_MAX_CANDIDATES", "1")
    monkeypatch.setenv("SCREENER_MIN_DOLLAR_VOLUME", "1000000")
    monkeypatch.setenv("SCREENER_PRICE_MIN", "1")
    monkeypatch.setenv("SCREENER_PRICE_MAX", "9999")
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "WATCHME")
    _set_wide_spread(monkeypatch)
    get_settings.cache_clear()

    # WATCHME not in the screened assets at all — but it's in .env
    assets = _make_assets("BIGSTOCK")
    quotes = {
        "BIGSTOCK": _make_quote("BIGSTOCK", 99, 101),
    }
    bars = {
        "BIGSTOCK": _make_bars_df(100.0, 80_000),
    }

    with patch(f"{_P}.get_active_equity_assets", return_value=assets):
        with patch(f"{_P}.get_universe_snapshot", return_value=quotes):
            with patch(f"{_P}.get_bulk_equity_bars", return_value=bars):
                from sauce.core.screener import screen_equities
                result = screen_equities(regime="TRENDING_UP")

    assert "BIGSTOCK" in result
    assert "WATCHME" in result


def test_screen_equities_always_includes_open_positions(monkeypatch):
    """Open position symbols are force-included."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("SCREENER_MAX_CANDIDATES", "1")
    monkeypatch.setenv("SCREENER_MIN_DOLLAR_VOLUME", "1000000")
    monkeypatch.setenv("SCREENER_PRICE_MIN", "1")
    monkeypatch.setenv("SCREENER_PRICE_MAX", "9999")
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "")
    _set_wide_spread(monkeypatch)
    get_settings.cache_clear()

    assets = _make_assets("AAA")
    quotes = {"AAA": _make_quote("AAA", 99, 101)}
    bars = {"AAA": _make_bars_df(100.0, 80_000)}

    with patch(f"{_P}.get_active_equity_assets", return_value=assets):
        with patch(f"{_P}.get_universe_snapshot", return_value=quotes):
            with patch(f"{_P}.get_bulk_equity_bars", return_value=bars):
                from sauce.core.screener import screen_equities
                result = screen_equities(
                    regime="TRENDING_UP",
                    open_symbols={"HELD"},
                )

    assert "HELD" in result


def test_screen_equities_fallback_on_asset_fetch_failure(monkeypatch):
    """If get_active_equity_assets fails, fall back to .env list."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "FALLBACK")
    get_settings.cache_clear()

    with patch(f"{_P}.get_active_equity_assets", side_effect=RuntimeError("API down")):
        from sauce.core.screener import screen_equities
        result = screen_equities(regime="TRENDING_UP")

    assert "FALLBACK" in result


def test_screen_equities_fallback_on_empty_quotes(monkeypatch):
    """If all quote chunks fail, fall back to .env list."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "FALLBACK")
    get_settings.cache_clear()

    assets = _make_assets("X")

    with patch(f"{_P}.get_active_equity_assets", return_value=assets):
        with patch(f"{_P}.get_universe_snapshot", return_value={}):
            from sauce.core.screener import screen_equities
            result = screen_equities(regime="TRENDING_UP")

    assert "FALLBACK" in result


def test_screen_equities_skips_crypto_symbols_in_quotes(monkeypatch):
    """Crypto symbols in the quote feed are ignored by the screener."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("SCREENER_MAX_CANDIDATES", "10")
    monkeypatch.setenv("SCREENER_MIN_DOLLAR_VOLUME", "0")
    monkeypatch.setenv("SCREENER_PRICE_MIN", "1")
    monkeypatch.setenv("SCREENER_PRICE_MAX", "999999")
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "")
    _set_wide_spread(monkeypatch)
    get_settings.cache_clear()

    assets = _make_assets("STOCK")
    quotes = {
        "STOCK": _make_quote("STOCK", 49, 51),
        "BTC/USD": _make_quote("BTC/USD", 60000, 61000),  # crypto — should be skipped
    }
    bars = {"STOCK": _make_bars_df(50.0, 100_000)}

    with patch(f"{_P}.get_active_equity_assets", return_value=assets):
        with patch(f"{_P}.get_universe_snapshot", return_value=quotes):
            with patch(f"{_P}.get_bulk_equity_bars", return_value=bars):
                from sauce.core.screener import screen_equities
                result = screen_equities(regime="TRENDING_UP")

    assert "STOCK" in result
    assert "BTC/USD" not in result


def test_screen_equities_deduplicates_always_include(monkeypatch):
    """If an always-include symbol is also top-screened, it only appears once."""
    monkeypatch.setenv("SCREENER_ENABLED", "true")
    monkeypatch.setenv("SCREENER_MAX_CANDIDATES", "10")
    monkeypatch.setenv("SCREENER_MIN_DOLLAR_VOLUME", "0")
    monkeypatch.setenv("SCREENER_PRICE_MIN", "1")
    monkeypatch.setenv("SCREENER_PRICE_MAX", "9999")
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "AAPL")
    _set_wide_spread(monkeypatch)
    get_settings.cache_clear()

    assets = _make_assets("AAPL")
    quotes = {"AAPL": _make_quote("AAPL", 149, 151)}
    bars = {"AAPL": _make_bars_df(150.0, 50_000)}

    with patch(f"{_P}.get_active_equity_assets", return_value=assets):
        with patch(f"{_P}.get_universe_snapshot", return_value=quotes):
            with patch(f"{_P}.get_bulk_equity_bars", return_value=bars):
                from sauce.core.screener import screen_equities
                result = screen_equities(regime="TRENDING_UP")

    assert result.count("AAPL") == 1
