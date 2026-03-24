import sys
import types
from datetime import UTC, datetime, timedelta

from sauce.core.schemas import PriceReference

fake_rank_bm25 = types.ModuleType("rank_bm25")
fake_rank_bm25.BM25Plus = object
sys.modules.setdefault("rank_bm25", fake_rank_bm25)

from sauce.loop import _is_quote_fresh, _pending_order_symbols, _should_persist_position


def test_is_quote_fresh_respects_ttl() -> None:
    now = datetime(2026, 3, 24, 12, 0, tzinfo=UTC)
    fresh = PriceReference(symbol="AAPL", bid=100.0, ask=101.0, mid=100.5, as_of=now - timedelta(seconds=30))
    stale = PriceReference(symbol="AAPL", bid=100.0, ask=101.0, mid=100.5, as_of=now - timedelta(seconds=300))

    assert _is_quote_fresh(fresh, ttl_seconds=120, now=now)
    assert not _is_quote_fresh(stale, ttl_seconds=120, now=now)


def test_pending_order_symbols_tracks_only_open_like_statuses() -> None:
    recent_orders = [
        {"symbol": "AAPL", "status": "accepted"},
        {"symbol": "MSFT", "status": "filled"},
        {"symbol": "NVDA", "status": "partially_filled"},
    ]

    pending = _pending_order_symbols(recent_orders)

    assert pending == {"AAPL", "NVDA"}


def test_should_persist_position_requires_actual_fill() -> None:
    assert _should_persist_position("filled", 1.0)
    assert _should_persist_position("partially_filled", 0.5)
    assert not _should_persist_position("accepted", 0.0)
    assert not _should_persist_position("new", 0.0)