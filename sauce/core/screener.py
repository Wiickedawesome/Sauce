"""
core/screener.py — Dynamic equity screener for full-market scanning.

Replaces the static .env equity symbol list with a data-driven screener that:
  1. Fetches all active, tradeable US equities from Alpaca (1 API call).
  2. Gets bulk latest quotes for price + spread filtering (chunked API calls).
  3. Gets bulk daily bars for volume estimation (chunked API calls).
  4. Ranks candidates by a composite screening score.
  5. Returns the top-N symbols for the research pipeline.

The .env equity list is kept as an "always include" watchlist — those symbols
are always passed to research regardless of screening results.  Open positions
are also force-included so exit signals are never missed.

Crypto is NOT screened — it uses the .env list as-is (user confirmed).
"""

import logging
from dataclasses import dataclass

from sauce.adapters.market_data import (
    is_crypto as _is_crypto,
    get_active_equity_assets,
    get_bulk_equity_bars,
    get_universe_snapshot,
)
from sauce.core.config import get_settings

logger = logging.getLogger(__name__)

# Maximum symbols to quote in a single bulk request (Alpaca limit safety).
_QUOTE_CHUNK_SIZE = 500
_BARS_CHUNK_SIZE = 200


@dataclass
class ScreenResult:
    """Single screened equity candidate."""
    symbol: str
    mid: float
    spread_pct: float
    est_dollar_volume: float
    score: float = 0.0


def screen_equities(
    regime: str,
    open_symbols: set[str] | None = None,
) -> list[str]:
    """
    Screen the full Alpaca equity universe and return the top-N symbols.

    Parameters
    ----------
    regime:        Current market regime (e.g. "TRENDING_UP", "RANGING").
    open_symbols:  Symbols with open positions (force-included).

    Returns
    -------
    List of equity ticker strings, length <= screener_max_candidates + len(always_include).
    """
    settings = get_settings()
    open_symbols = open_symbols or set()

    # ── Step 1: Get all tradeable equities ────────────────────────────────
    try:
        all_assets = get_active_equity_assets()
    except Exception as exc:
        logger.error("Screener: failed to fetch assets, falling back to .env list: %s", exc)
        return settings.equity_universe

    tradeable_symbols = [a["symbol"] for a in all_assets]
    logger.info("Screener: %d tradeable equities from Alpaca", len(tradeable_symbols))

    # ── Step 2: Get bulk quotes for price + spread filtering ──────────────
    all_quotes: dict = {}
    for i in range(0, len(tradeable_symbols), _QUOTE_CHUNK_SIZE):
        chunk = tradeable_symbols[i : i + _QUOTE_CHUNK_SIZE]
        try:
            quotes = get_universe_snapshot(chunk)
            all_quotes.update(quotes)
        except Exception as exc:
            logger.warning("Screener: quote chunk %d failed: %s", i, exc)

    if not all_quotes:
        logger.error("Screener: no quotes returned, falling back to .env list")
        return settings.equity_universe

    # ── Step 3: Price + spread pre-filter ─────────────────────────────────
    price_min = settings.screener_price_min
    price_max = settings.screener_price_max
    max_spread = settings.max_spread_pct

    price_passed: list[str] = []
    price_data: dict[str, dict] = {}

    for sym, quote in all_quotes.items():
        if _is_crypto(sym):
            continue
        mid = quote.mid
        if mid <= 0:
            continue
        if mid < price_min or mid > price_max:
            continue
        spread_pct = (quote.ask - quote.bid) / mid if mid > 0 else 1.0
        if spread_pct > max_spread:
            continue
        price_passed.append(sym)
        price_data[sym] = {"mid": mid, "spread_pct": spread_pct}

    logger.info("Screener: %d equities passed price/spread filter", len(price_passed))

    if not price_passed:
        logger.warning("Screener: no equities passed price filter, falling back")
        return settings.equity_universe

    # ── Step 4: Bulk daily bars for volume estimation ─────────────────────
    volume_data: dict[str, float] = {}
    for i in range(0, len(price_passed), _BARS_CHUNK_SIZE):
        chunk = price_passed[i : i + _BARS_CHUNK_SIZE]
        try:
            bars_map = get_bulk_equity_bars(chunk, timeframe="1Day", bars=5)
            for sym, df in bars_map.items():
                if df.empty:
                    continue
                avg_vol = float(df["volume"].mean()) if "volume" in df.columns else 0.0
                close = float(df["close"].iloc[-1]) if "close" in df.columns and len(df) > 0 else 0.0
                volume_data[sym] = avg_vol * close  # estimated daily $ volume
        except Exception as exc:
            logger.warning("Screener: bars chunk %d failed: %s", i, exc)

    # ── Step 5: Dollar volume filter ──────────────────────────────────────
    min_dv = settings.screener_min_dollar_volume
    candidates: list[ScreenResult] = []

    for sym in price_passed:
        dv = volume_data.get(sym, 0.0)
        if dv < min_dv:
            continue
        pd_ = price_data[sym]
        candidates.append(ScreenResult(
            symbol=sym,
            mid=pd_["mid"],
            spread_pct=pd_["spread_pct"],
            est_dollar_volume=dv,
        ))

    logger.info("Screener: %d equities passed dollar-volume filter ($%.0fM min)",
                len(candidates), min_dv / 1_000_000)

    # ── Step 6: Score and rank ────────────────────────────────────────────
    # Simple composite: favor high liquidity and tight spreads.
    if candidates:
        max_dv = max(c.est_dollar_volume for c in candidates)
        for c in candidates:
            liquidity_score = c.est_dollar_volume / max_dv if max_dv > 0 else 0.0
            tightness_score = max(0.0, 1.0 - (c.spread_pct / max_spread)) if max_spread > 0 else 0.5
            c.score = 0.6 * liquidity_score + 0.4 * tightness_score

    candidates.sort(key=lambda c: c.score, reverse=True)

    # ── Step 7: Build final list ──────────────────────────────────────────
    max_n = settings.screener_max_candidates
    top_symbols = [c.symbol for c in candidates[:max_n]]

    # Always-include: .env equity watchlist + open positions
    always_include = set(settings.equity_universe)
    for sym in open_symbols:
        if not _is_crypto(sym):
            always_include.add(sym)

    # Merge: screened + always-include (deduplicated, screened first)
    seen = set(top_symbols)
    for sym in always_include:
        if sym not in seen:
            top_symbols.append(sym)
            seen.add(sym)

    logger.info(
        "Screener: returning %d equities (%d screened + %d always-include)",
        len(top_symbols),
        min(max_n, len(candidates)),
        len(top_symbols) - min(max_n, len(candidates)),
    )

    return top_symbols
