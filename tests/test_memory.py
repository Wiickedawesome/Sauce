"""
test_memory.py — Tests for the BM25 trade memory system.
"""

from sauce.memory import (
    MemoryEntry,
    TradeMemory,
    build_outcome_description,
    build_situation_description,
)


class TestMemoryEntry:
    def test_frozen(self):
        entry = MemoryEntry(situation="s", outcome="o", lesson="l")
        assert entry.situation == "s"
        try:
            entry.situation = "x"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestTradeMemory:
    def test_empty_recall(self):
        tm = TradeMemory()
        assert tm.recall("anything") == []
        assert tm.size == 0

    def test_store_and_recall(self):
        tm = TradeMemory()
        e1 = MemoryEntry(
            situation="BTC bullish RSI=45 MACD positive volume high",
            outcome="profit +5%",
            lesson="Bullish BTC with high volume works",
        )
        e2 = MemoryEntry(
            situation="ETH bearish RSI=70 MACD negative volume low",
            outcome="loss -3%",
            lesson="Avoid ETH in bearish low volume",
        )
        tm.store(e1)
        tm.store(e2)
        assert tm.size == 2

        # Querying for BTC bullish should return e1 first
        results = tm.recall("BTC bullish volume high", n=2)
        assert len(results) >= 1
        assert results[0].situation == e1.situation

    def test_recall_limits_to_n(self):
        tm = TradeMemory()
        for i in range(10):
            tm.store(MemoryEntry(
                situation=f"trade {i} with RSI and MACD data",
                outcome=f"outcome {i}",
                lesson=f"lesson {i}",
            ))
        results = tm.recall("trade with RSI", n=3)
        assert len(results) <= 3

    def test_recall_skips_zero_score(self):
        """Completely irrelevant queries shouldn't match."""
        tm = TradeMemory()
        tm.store(MemoryEntry(
            situation="BTC bullish momentum",
            outcome="profit",
            lesson="works",
        ))
        # Query with entirely different words
        results = tm.recall("completely unrelated gibberish xyzzy")
        assert len(results) == 0

    def test_init_with_entries(self):
        entries = [
            MemoryEntry(situation="a b c", outcome="o1", lesson="l1"),
            MemoryEntry(situation="d e f", outcome="o2", lesson="l2"),
        ]
        tm = TradeMemory(entries)
        assert tm.size == 2
        results = tm.recall("a b c")
        assert len(results) >= 1


class TestBuildDescriptions:
    def test_situation_description(self):
        desc = build_situation_description(
            symbol="BTC/USD",
            regime="bullish",
            score=75,
            threshold=40,
            rsi_14=45.0,
            macd_hist=0.0012,
            bb_pct=0.35,
            volume_ratio=1.5,
            current_price=50000.0,
            strategy_name="crypto_momentum",
        )
        assert "BTC/USD" in desc
        assert "bullish" in desc
        assert "RSI=45.0" in desc
        assert "crypto_momentum" in desc

    def test_situation_description_none_indicators(self):
        desc = build_situation_description(
            symbol="ETH/USD",
            regime="neutral",
            score=50,
            threshold=40,
            rsi_14=None,
            macd_hist=None,
            bb_pct=None,
            volume_ratio=None,
            current_price=3000.0,
            strategy_name="crypto_momentum",
        )
        assert "ETH/USD" in desc
        assert "RSI" not in desc  # None values excluded

    def test_outcome_description(self):
        desc = build_outcome_description(
            symbol="BTC/USD",
            entry_price=50000.0,
            exit_price=52000.0,
            exit_trigger="profit_target",
            hold_hours=12.5,
            realized_pnl=200.0,
        )
        assert "BTC/USD" in desc
        assert "profit_target" in desc
        assert "+200.00" in desc
        assert "+4.00%" in desc
