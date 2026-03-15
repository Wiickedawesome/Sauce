"""Tests for sauce.memory.db — session and strategic memory adapters."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect

import sauce.memory.db as mem
from sauce.memory.db import (
    ClaudeCalibrationRow,
    IntradayNarrativeRow,
    RegimeLogRow,
    RegimeTransitionRow,
    SetupPerformanceRow,
    SignalLogRow,
    SymbolCharacterRow,
    SymbolLearnedBehaviorRow,
    TradeLogRow,
    VetoPatternRow,
    WeeklyPerformanceRow,
    get_engine,
    get_session,
    get_session_context,
    get_similar_trades,
    get_strategic_context,
    reset_session_memory_if_new_day,
    write_claude_calibration,
    write_narrative,
    write_regime_log,
    write_regime_transition,
    write_setup_performance,
    write_signal_log,
    write_symbol_behavior,
    write_symbol_character,
    write_trade_log,
    write_veto_pattern,
    write_weekly_performance,
)
from sauce.core.schemas import (
    ClaudeCalibrationEntry,
    IntradayNarrativeEntry,
    RegimeLogEntry,
    RegimeTransitionEntry,
    SetupPerformanceEntry,
    SignalLogEntry,
    SymbolCharacterEntry,
    SymbolLearnedBehaviorEntry,
    TradeLogEntry,
    VetoPatternEntry,
    WeeklyPerformanceEntry,
)

NOW = datetime.now(timezone.utc)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_session_db(tmp_path: Path) -> str:
    """Provide a temporary session_memory DB, resetting module state."""
    mem._engines = {}
    mem._last_reset_date = None
    return str(tmp_path / "test_session_memory.db")


@pytest.fixture()
def tmp_strategic_db(tmp_path: Path) -> str:
    """Provide a temporary strategic_memory DB, resetting module state."""
    mem._engines = {}
    mem._last_reset_date = None
    return str(tmp_path / "test_strategic_memory.db")


# ══════════════════════════════════════════════════════════════════════════════
# ENGINE / TABLE CREATION
# ══════════════════════════════════════════════════════════════════════════════


def test_session_engine_creates_db_file(tmp_session_db: str) -> None:
    get_engine(tmp_session_db)
    assert Path(tmp_session_db).exists()


def test_strategic_engine_creates_db_file(tmp_strategic_db: str) -> None:
    get_engine(tmp_strategic_db)
    assert Path(tmp_strategic_db).exists()


def test_session_engine_creates_all_tables(tmp_session_db: str) -> None:
    engine = get_engine(tmp_session_db)
    tables = set(inspect(engine).get_table_names())
    expected = {"regime_log", "signal_log", "trade_log", "intraday_narrative", "symbol_character"}
    assert expected.issubset(tables)


def test_strategic_engine_creates_all_tables(tmp_strategic_db: str) -> None:
    engine = get_engine(tmp_strategic_db)
    tables = set(inspect(engine).get_table_names())
    expected = {
        "setup_performance",
        "regime_transitions",
        "veto_patterns",
        "weekly_performance",
        "symbol_learned_behavior",
        "claude_calibration",
    }
    assert expected.issubset(tables)


def test_engine_is_cached(tmp_session_db: str) -> None:
    e1 = get_engine(tmp_session_db)
    e2 = get_engine(tmp_session_db)
    assert e1 is e2


# ══════════════════════════════════════════════════════════════════════════════
# SESSION WRITE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def test_write_regime_log(tmp_session_db: str) -> None:
    entry = RegimeLogEntry(
        timestamp=NOW,
        regime_type="TRENDING_UP",
        confidence=0.85,
        vix_proxy=18.5,
        market_bias="bullish",
    )
    write_regime_log(entry, tmp_session_db)

    session = get_session(tmp_session_db)
    row = session.query(RegimeLogRow).first()
    session.close()

    assert row is not None
    assert row.regime_type == "TRENDING_UP"
    assert row.confidence == pytest.approx(0.85)
    assert row.vix_proxy == pytest.approx(18.5)
    assert row.market_bias == "bullish"


def test_write_signal_log(tmp_session_db: str) -> None:
    entry = SignalLogEntry(
        timestamp=NOW,
        symbol="AAPL",
        setup_type="equity_trend_pullback",
        score=72.5,
        claude_decision="approve",
        reason="Strong pullback to support",
    )
    write_signal_log(entry, tmp_session_db)

    session = get_session(tmp_session_db)
    row = session.query(SignalLogRow).filter_by(symbol="AAPL").first()
    session.close()

    assert row is not None
    assert row.setup_type == "equity_trend_pullback"
    assert row.score == pytest.approx(72.5)
    assert row.claude_decision == "approve"
    assert row.reason == "Strong pullback to support"


def test_write_trade_log(tmp_session_db: str) -> None:
    entry = TradeLogEntry(
        timestamp=NOW,
        symbol="BTC/USD",
        entry_price=65000.0,
        direction="buy",
        status="open",
        unrealized_pnl=120.50,
    )
    write_trade_log(entry, tmp_session_db)

    session = get_session(tmp_session_db)
    row = session.query(TradeLogRow).filter_by(symbol="BTCUSD").first()
    session.close()

    assert row is not None
    assert row.entry_price == pytest.approx(65000.0)
    assert row.direction == "buy"
    assert row.status == "open"
    assert row.unrealized_pnl == pytest.approx(120.50)


def test_write_narrative(tmp_session_db: str) -> None:
    entry = IntradayNarrativeEntry(
        timestamp=NOW,
        narrative_text="Market opened strong, SPY up 0.3%.",
    )
    write_narrative(entry, tmp_session_db)

    session = get_session(tmp_session_db)
    row = session.query(IntradayNarrativeRow).first()
    session.close()

    assert row is not None
    assert row.narrative_text == "Market opened strong, SPY up 0.3%."


def test_write_symbol_character_insert(tmp_session_db: str) -> None:
    entry = SymbolCharacterEntry(
        symbol="TSLA",
        signal_count_today=3,
        direction_consistency=0.8,
        last_signal_result="win",
    )
    write_symbol_character(entry, tmp_session_db)

    session = get_session(tmp_session_db)
    row = session.query(SymbolCharacterRow).filter_by(symbol="TSLA").first()
    session.close()

    assert row is not None
    assert row.signal_count_today == 3
    assert row.direction_consistency == pytest.approx(0.8)
    assert row.last_signal_result == "win"


def test_write_symbol_character_upsert(tmp_session_db: str) -> None:
    first = SymbolCharacterEntry(
        symbol="TSLA",
        signal_count_today=1,
        direction_consistency=0.5,
        last_signal_result="pending",
    )
    write_symbol_character(first, tmp_session_db)

    updated = SymbolCharacterEntry(
        symbol="TSLA",
        signal_count_today=4,
        direction_consistency=-0.3,
        last_signal_result="loss",
    )
    write_symbol_character(updated, tmp_session_db)

    session = get_session(tmp_session_db)
    rows = session.query(SymbolCharacterRow).filter_by(symbol="TSLA").all()
    session.close()

    assert len(rows) == 1
    assert rows[0].signal_count_today == 4
    assert rows[0].direction_consistency == pytest.approx(-0.3)
    assert rows[0].last_signal_result == "loss"


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGIC WRITE HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def test_write_setup_performance(tmp_strategic_db: str) -> None:
    entry = SetupPerformanceEntry(
        setup_type="crypto_mean_reversion",
        symbol="BTC/USD",
        regime_at_entry="VOLATILE",
        time_of_day_bucket="09:30-12:00",
        win=True,
        pnl=350.0,
        hold_duration_minutes=45.0,
        date="2025-06-01",
    )
    write_setup_performance(entry, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    row = session.query(SetupPerformanceRow).first()
    session.close()

    assert row is not None
    assert row.setup_type == "crypto_mean_reversion"
    assert row.symbol == "BTCUSD"
    assert row.win is True
    assert row.pnl == pytest.approx(350.0)


def test_write_regime_transition_insert(tmp_strategic_db: str) -> None:
    entry = RegimeTransitionEntry(
        from_regime="RANGING",
        to_regime="TRENDING_UP",
        duration_minutes=120.0,
        count=1,
    )
    write_regime_transition(entry, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    row = session.query(RegimeTransitionRow).first()
    session.close()

    assert row is not None
    assert row.from_regime == "RANGING"
    assert row.to_regime == "TRENDING_UP"
    assert row.count == 1


def test_write_regime_transition_upsert_increments_count(tmp_strategic_db: str) -> None:
    entry = RegimeTransitionEntry(
        from_regime="RANGING",
        to_regime="TRENDING_UP",
        duration_minutes=120.0,
    )
    write_regime_transition(entry, tmp_strategic_db)
    write_regime_transition(entry, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    rows = session.query(RegimeTransitionRow).filter_by(
        from_regime="RANGING", to_regime="TRENDING_UP"
    ).all()
    session.close()

    assert len(rows) == 1
    assert rows[0].count == 2


def test_write_veto_pattern_insert(tmp_strategic_db: str) -> None:
    entry = VetoPatternEntry(
        veto_reason="RSI too high",
        setup_type="crypto_breakout",
        count=1,
        last_seen=NOW,
    )
    write_veto_pattern(entry, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    row = session.query(VetoPatternRow).first()
    session.close()

    assert row is not None
    assert row.veto_reason == "RSI too high"
    assert row.count == 1


def test_write_veto_pattern_upsert_increments_count(tmp_strategic_db: str) -> None:
    entry = VetoPatternEntry(
        veto_reason="RSI too high",
        setup_type="crypto_breakout",
        count=1,
        last_seen=NOW,
    )
    write_veto_pattern(entry, tmp_strategic_db)
    write_veto_pattern(entry, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    row = session.query(VetoPatternRow).filter_by(
        veto_reason="RSI too high", setup_type="crypto_breakout"
    ).first()
    session.close()

    assert row is not None
    assert row.count == 2


def test_write_weekly_performance_insert(tmp_strategic_db: str) -> None:
    entry = WeeklyPerformanceEntry(
        week="2025-W22",
        setup_type="equity_trend_pullback",
        trades=5,
        win_rate=0.6,
        avg_pnl=120.0,
        sharpe=1.2,
    )
    write_weekly_performance(entry, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    row = session.query(WeeklyPerformanceRow).first()
    session.close()

    assert row is not None
    assert row.week == "2025-W22"
    assert row.trades == 5
    assert row.win_rate == pytest.approx(0.6)


def test_write_weekly_performance_upsert(tmp_strategic_db: str) -> None:
    first = WeeklyPerformanceEntry(
        week="2025-W22",
        setup_type="equity_trend_pullback",
        trades=5,
        win_rate=0.6,
        avg_pnl=120.0,
        sharpe=1.2,
    )
    write_weekly_performance(first, tmp_strategic_db)

    updated = WeeklyPerformanceEntry(
        week="2025-W22",
        setup_type="equity_trend_pullback",
        trades=8,
        win_rate=0.75,
        avg_pnl=180.0,
        sharpe=1.5,
    )
    write_weekly_performance(updated, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    rows = session.query(WeeklyPerformanceRow).filter_by(
        week="2025-W22", setup_type="equity_trend_pullback"
    ).all()
    session.close()

    assert len(rows) == 1
    assert rows[0].trades == 8
    assert rows[0].win_rate == pytest.approx(0.75)
    assert rows[0].sharpe == pytest.approx(1.5)


def test_write_symbol_behavior_insert(tmp_strategic_db: str) -> None:
    entry = SymbolLearnedBehaviorEntry(
        symbol="ETH/USD",
        setup_type="crypto_mean_reversion",
        optimal_rsi_entry=32.0,
        avg_reversion_depth=0.04,
        avg_bounce_magnitude=0.06,
        sample_size=15,
    )
    write_symbol_behavior(entry, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    row = session.query(SymbolLearnedBehaviorRow).filter_by(symbol="ETHUSD").first()
    session.close()

    assert row is not None
    assert row.optimal_rsi_entry == pytest.approx(32.0)
    assert row.sample_size == 15


def test_write_symbol_behavior_upsert(tmp_strategic_db: str) -> None:
    first = SymbolLearnedBehaviorEntry(
        symbol="ETH/USD",
        setup_type="crypto_mean_reversion",
        optimal_rsi_entry=32.0,
        sample_size=15,
    )
    write_symbol_behavior(first, tmp_strategic_db)

    updated = SymbolLearnedBehaviorEntry(
        symbol="ETH/USD",
        setup_type="crypto_mean_reversion",
        optimal_rsi_entry=29.5,
        avg_reversion_depth=0.05,
        sample_size=25,
    )
    write_symbol_behavior(updated, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    rows = session.query(SymbolLearnedBehaviorRow).filter_by(
        symbol="ETHUSD", setup_type="crypto_mean_reversion"
    ).all()
    session.close()

    assert len(rows) == 1
    assert rows[0].optimal_rsi_entry == pytest.approx(29.5)
    assert rows[0].sample_size == 25


def test_write_claude_calibration(tmp_strategic_db: str) -> None:
    entry = ClaudeCalibrationEntry(
        date="2025-06-01",
        confidence_stated=0.78,
        outcome="win",
        setup_type="crypto_breakout",
    )
    write_claude_calibration(entry, tmp_strategic_db)

    session = get_session(tmp_strategic_db)
    row = session.query(ClaudeCalibrationRow).first()
    session.close()

    assert row is not None
    assert row.confidence_stated == pytest.approx(0.78)
    assert row.outcome == "win"


# ══════════════════════════════════════════════════════════════════════════════
# SESSION RESET
# ══════════════════════════════════════════════════════════════════════════════


def test_reset_session_memory_truncates_tables(tmp_session_db: str) -> None:
    """After reset, all 5 session tables should be empty."""
    write_regime_log(
        RegimeLogEntry(timestamp=NOW, regime_type="RANGING", confidence=0.5),
        tmp_session_db,
    )
    write_signal_log(
        SignalLogEntry(
            timestamp=NOW, symbol="AAPL", setup_type="equity_trend_pullback",
            score=50.0, claude_decision="hold",
        ),
        tmp_session_db,
    )
    write_trade_log(
        TradeLogEntry(
            timestamp=NOW, symbol="AAPL", entry_price=150.0,
            direction="buy", status="open",
        ),
        tmp_session_db,
    )
    write_narrative(
        IntradayNarrativeEntry(timestamp=NOW, narrative_text="test"),
        tmp_session_db,
    )
    write_symbol_character(
        SymbolCharacterEntry(symbol="AAPL"),
        tmp_session_db,
    )

    # Force reset by clearing _last_reset_date
    mem._last_reset_date = None
    result = reset_session_memory_if_new_day(tmp_session_db)
    assert result is True

    session = get_session(tmp_session_db)
    assert session.query(RegimeLogRow).count() == 0
    assert session.query(SignalLogRow).count() == 0
    assert session.query(TradeLogRow).count() == 0
    assert session.query(IntradayNarrativeRow).count() == 0
    assert session.query(SymbolCharacterRow).count() == 0
    session.close()


def test_reset_session_memory_returns_false_same_day(tmp_session_db: str) -> None:
    """Second call on the same day should be a no-op returning False."""
    mem._last_reset_date = None
    first = reset_session_memory_if_new_day(tmp_session_db)
    second = reset_session_memory_if_new_day(tmp_session_db)
    assert first is True
    assert second is False


# ══════════════════════════════════════════════════════════════════════════════
# READ HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def test_get_session_context_empty(tmp_session_db: str) -> None:
    ctx = get_session_context(tmp_session_db)
    assert ctx.regime_history == []
    assert ctx.signals_today == []
    assert ctx.trades_today == []
    assert ctx.narrative == ""
    assert ctx.symbol_characters == []
    assert ctx.as_of is not None


def test_get_session_context_populated(tmp_session_db: str) -> None:
    write_regime_log(
        RegimeLogEntry(timestamp=NOW, regime_type="VOLATILE", confidence=0.9, vix_proxy=25.0),
        tmp_session_db,
    )
    write_signal_log(
        SignalLogEntry(
            timestamp=NOW, symbol="AAPL", setup_type="equity_trend_pullback",
            score=80.0, claude_decision="approve", reason="Momentum strong",
        ),
        tmp_session_db,
    )
    write_trade_log(
        TradeLogEntry(
            timestamp=NOW, symbol="AAPL", entry_price=175.0,
            direction="buy", status="open", unrealized_pnl=50.0,
        ),
        tmp_session_db,
    )
    write_narrative(
        IntradayNarrativeEntry(timestamp=NOW, narrative_text="SPY rallying hard"),
        tmp_session_db,
    )
    write_symbol_character(
        SymbolCharacterEntry(symbol="AAPL", signal_count_today=2, direction_consistency=0.9),
        tmp_session_db,
    )

    ctx = get_session_context(tmp_session_db)

    assert len(ctx.regime_history) == 1
    assert ctx.regime_history[0].regime_type == "VOLATILE"
    assert len(ctx.signals_today) == 1
    assert ctx.signals_today[0].symbol == "AAPL"
    assert len(ctx.trades_today) == 1
    assert ctx.trades_today[0].entry_price == pytest.approx(175.0)
    assert ctx.narrative == "SPY rallying hard"
    assert len(ctx.symbol_characters) == 1
    assert ctx.symbol_characters[0].symbol == "AAPL"
    assert ctx.regime_history[0].timestamp.tzinfo == timezone.utc
    assert ctx.signals_today[0].timestamp.tzinfo == timezone.utc
    assert ctx.trades_today[0].timestamp.tzinfo == timezone.utc


def test_get_strategic_context_empty(tmp_strategic_db: str) -> None:
    ctx = get_strategic_context(tmp_strategic_db)
    assert ctx.setup_performance == []
    assert ctx.regime_transitions == []
    assert ctx.relevant_veto_patterns == []
    assert ctx.weekly_trend == []
    assert ctx.symbol_behavior is None
    assert ctx.claude_calibration == []


def test_get_strategic_context_filtered(tmp_strategic_db: str) -> None:
    # Populate with two setup types
    write_setup_performance(
        SetupPerformanceEntry(
            setup_type="crypto_mean_reversion", symbol="BTC/USD",
            regime_at_entry="VOLATILE", time_of_day_bucket="09:30-12:00",
            win=True, pnl=200.0, hold_duration_minutes=30.0, date="2025-06-01",
        ),
        tmp_strategic_db,
    )
    write_setup_performance(
        SetupPerformanceEntry(
            setup_type="equity_trend_pullback", symbol="AAPL",
            regime_at_entry="TRENDING_UP", time_of_day_bucket="12:00-14:00",
            win=False, pnl=-50.0, hold_duration_minutes=60.0, date="2025-06-01",
        ),
        tmp_strategic_db,
    )
    write_veto_pattern(
        VetoPatternEntry(
            veto_reason="Volume too low", setup_type="crypto_mean_reversion",
            count=1, last_seen=NOW,
        ),
        tmp_strategic_db,
    )
    write_symbol_behavior(
        SymbolLearnedBehaviorEntry(
            symbol="BTC/USD", setup_type="crypto_mean_reversion",
            optimal_rsi_entry=30.0, sample_size=10,
        ),
        tmp_strategic_db,
    )

    # Filter by setup_type and symbol
    ctx = get_strategic_context(
        tmp_strategic_db,
        setup_type="crypto_mean_reversion",
        symbol="BTC/USD",
    )

    assert len(ctx.setup_performance) == 1
    assert ctx.setup_performance[0].symbol == "BTCUSD"
    assert len(ctx.relevant_veto_patterns) == 1
    assert ctx.relevant_veto_patterns[0].veto_reason == "Volume too low"
    assert ctx.symbol_behavior is not None
    assert ctx.symbol_behavior.optimal_rsi_entry == pytest.approx(30.0)


def test_get_strategic_context_regime_filter(tmp_strategic_db: str) -> None:
    write_setup_performance(
        SetupPerformanceEntry(
            setup_type="crypto_mean_reversion", symbol="BTC/USD",
            regime_at_entry="VOLATILE", time_of_day_bucket="09:30-12:00",
            win=True, pnl=200.0, hold_duration_minutes=30.0, date="2025-06-01",
        ),
        tmp_strategic_db,
    )
    write_setup_performance(
        SetupPerformanceEntry(
            setup_type="crypto_mean_reversion", symbol="BTC/USD",
            regime_at_entry="RANGING", time_of_day_bucket="12:00-14:00",
            win=False, pnl=-80.0, hold_duration_minutes=45.0, date="2025-06-02",
        ),
        tmp_strategic_db,
    )

    ctx = get_strategic_context(tmp_strategic_db, regime="VOLATILE")
    assert len(ctx.setup_performance) == 1
    assert ctx.setup_performance[0].regime_at_entry == "VOLATILE"


# ══════════════════════════════════════════════════════════════════════════════
# SIMILAR TRADES RETRIEVAL (RAG)
# ══════════════════════════════════════════════════════════════════════════════


def _write_perf(db: str, **overrides) -> None:
    """Helper to write a SetupPerformanceEntry with defaults."""
    defaults = dict(
        setup_type="equity_trend_pullback",
        symbol="SPY",
        regime_at_entry="TRENDING_UP",
        time_of_day_bucket="09:30-12:00",
        win=True,
        pnl=50.0,
        hold_duration_minutes=30.0,
        date="2025-06-01",
    )
    defaults.update(overrides)
    write_setup_performance(SetupPerformanceEntry(**defaults), db)


def test_get_similar_trades_empty_db(tmp_strategic_db: str) -> None:
    result = get_similar_trades(tmp_strategic_db, symbol="SPY")
    assert result == []


def test_get_similar_trades_exact_symbol(tmp_strategic_db: str) -> None:
    _write_perf(tmp_strategic_db, symbol="SPY", date="2025-06-01")
    _write_perf(tmp_strategic_db, symbol="AAPL", date="2025-06-02")

    result = get_similar_trades(tmp_strategic_db, symbol="SPY")
    assert len(result) == 1
    assert result[0].symbol == "SPY"


def test_get_similar_trades_regime_widens_pool(tmp_strategic_db: str) -> None:
    _write_perf(tmp_strategic_db, symbol="AAPL", regime_at_entry="VOLATILE", date="2025-06-01")
    _write_perf(tmp_strategic_db, symbol="MSFT", regime_at_entry="TRENDING_UP", date="2025-06-02")

    # Without regime, only exact symbol matches
    result = get_similar_trades(tmp_strategic_db, symbol="SPY")
    assert len(result) == 0

    # With regime, picks up AAPL which shares VOLATILE
    result = get_similar_trades(tmp_strategic_db, symbol="SPY", regime="VOLATILE")
    assert len(result) == 1
    assert result[0].symbol == "AAPL"


def test_get_similar_trades_scores_correctly(tmp_strategic_db: str) -> None:
    """Same symbol + same regime should rank higher than regime-only match."""
    _write_perf(tmp_strategic_db, symbol="SPY", regime_at_entry="VOLATILE", date="2025-06-01")
    _write_perf(tmp_strategic_db, symbol="AAPL", regime_at_entry="VOLATILE", date="2025-06-02")

    result = get_similar_trades(tmp_strategic_db, symbol="SPY", regime="VOLATILE")
    assert len(result) == 2
    # SPY match (score=3+2=5) ranks above AAPL match (score=0+2=2)
    assert result[0].symbol == "SPY"
    assert result[1].symbol == "AAPL"


def test_get_similar_trades_setup_type_bonus(tmp_strategic_db: str) -> None:
    _write_perf(tmp_strategic_db, symbol="SPY", setup_type="equity_trend_pullback", date="2025-06-01")
    _write_perf(tmp_strategic_db, symbol="SPY", setup_type="crypto_mean_reversion", date="2025-06-02")

    result = get_similar_trades(
        tmp_strategic_db, symbol="SPY", setup_type="equity_trend_pullback",
    )
    assert len(result) == 2
    # Both have symbol match (3), but first also has setup_type match (+1)
    assert result[0].setup_type == "equity_trend_pullback"


def test_get_similar_trades_top_k(tmp_strategic_db: str) -> None:
    for i in range(10):
        _write_perf(tmp_strategic_db, symbol="SPY", date=f"2025-06-{i+1:02d}")

    result = get_similar_trades(tmp_strategic_db, symbol="SPY", top_k=3)
    assert len(result) == 3


def test_get_similar_trades_recency_tiebreaker(tmp_strategic_db: str) -> None:
    _write_perf(tmp_strategic_db, symbol="SPY", date="2025-06-01")
    _write_perf(tmp_strategic_db, symbol="SPY", date="2025-06-10")

    result = get_similar_trades(tmp_strategic_db, symbol="SPY", top_k=2)
    assert len(result) == 2
    # Same score, so more recent date comes first
    assert result[0].date == "2025-06-10"
    assert result[1].date == "2025-06-01"


def test_get_similar_trades_returns_pydantic_models(tmp_strategic_db: str) -> None:
    _write_perf(tmp_strategic_db, symbol="SPY", pnl=42.5, win=True)

    result = get_similar_trades(tmp_strategic_db, symbol="SPY")
    assert len(result) == 1
    assert isinstance(result[0], SetupPerformanceEntry)
    assert result[0].pnl == 42.5
    assert result[0].win is True
