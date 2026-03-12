"""
tests/test_context_builders.py — Sprint 4 context builder tests.

Covers build_session_paragraph() and build_strategic_paragraph() from
sauce/prompts/context.py.
"""

from datetime import datetime, timezone

import pytest

from sauce.core.schemas import (
    ClaudeCalibrationEntry,
    IntradayNarrativeEntry,
    RegimeLogEntry,
    RegimeTransitionEntry,
    SessionContext,
    SetupPerformanceEntry,
    SignalLogEntry,
    StrategicContext,
    SymbolCharacterEntry,
    SymbolLearnedBehaviorEntry,
    TradeLogEntry,
    VetoPatternEntry,
    WeeklyPerformanceEntry,
)
from sauce.prompts.context import build_session_paragraph, build_strategic_paragraph

_NOW = datetime(2024, 6, 10, 14, 30, 0, tzinfo=timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# build_session_paragraph
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSessionParagraph:
    """Tests for build_session_paragraph."""

    def _empty_ctx(self) -> SessionContext:
        return SessionContext(as_of=_NOW)

    # ── Empty context ────────────────────────────────────────────────────────

    def test_empty_context_returns_empty_string(self):
        assert build_session_paragraph(self._empty_ctx()) == ""

    # ── Regime history ───────────────────────────────────────────────────────

    def test_single_regime_entry(self):
        ctx = SessionContext(
            as_of=_NOW,
            regime_history=[
                RegimeLogEntry(
                    timestamp=_NOW,
                    regime_type="TRENDING_UP",
                    confidence=0.82,
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "Regime has been TRENDING_UP since session start" in result
        assert "(confidence 82%)" in result

    def test_multiple_regime_entries_shows_transitions(self):
        ctx = SessionContext(
            as_of=_NOW,
            regime_history=[
                RegimeLogEntry(
                    timestamp=datetime(2024, 6, 10, 10, 0, tzinfo=timezone.utc),
                    regime_type="RANGING",
                    confidence=0.50,
                ),
                RegimeLogEntry(
                    timestamp=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
                    regime_type="TRENDING_UP",
                    confidence=0.75,
                ),
                RegimeLogEntry(
                    timestamp=datetime(2024, 6, 10, 14, 0, tzinfo=timezone.utc),
                    regime_type="VOLATILE",
                    confidence=0.90,
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "RANGING → TRENDING_UP at 12:00" in result
        assert "TRENDING_UP → VOLATILE at 14:00" in result
        assert "Current regime: VOLATILE" in result
        assert "(confidence 90%)" in result

    # ── Signals today ────────────────────────────────────────────────────────

    def test_signals_today_summary_and_detail(self):
        ctx = SessionContext(
            as_of=_NOW,
            signals_today=[
                SignalLogEntry(
                    timestamp=datetime(2024, 6, 10, 10, 30, tzinfo=timezone.utc),
                    symbol="AAPL",
                    setup_type="equity_trend_pullback",
                    score=72.0,
                    claude_decision="approve",
                    reason="Strong trend",
                ),
                SignalLogEntry(
                    timestamp=datetime(2024, 6, 10, 11, 0, tzinfo=timezone.utc),
                    symbol="BTC-USD",
                    setup_type="crypto_mean_reversion",
                    score=55.0,
                    claude_decision="reject",
                    reason="Weak bounce",
                ),
                SignalLogEntry(
                    timestamp=datetime(2024, 6, 10, 12, 0, tzinfo=timezone.utc),
                    symbol="ETH-USD",
                    setup_type="crypto_breakout",
                    score=40.0,
                    claude_decision="hold",
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "3 signals evaluated today" in result
        assert "1 approved" in result
        assert "1 rejected" in result
        assert "1 held" in result
        # Detail lines
        assert "10:30 AAPL equity_trend_pullback (score 72.0): approve — Strong trend." in result
        assert "11:00 BTC-USD crypto_mean_reversion (score 55.0): reject — Weak bounce." in result
        assert "12:00 ETH-USD crypto_breakout (score 40.0): hold." in result

    def test_signal_without_reason_omits_dash(self):
        ctx = SessionContext(
            as_of=_NOW,
            signals_today=[
                SignalLogEntry(
                    timestamp=_NOW,
                    symbol="AAPL",
                    setup_type="equity_trend_pullback",
                    score=60.0,
                    claude_decision="hold",
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "hold." in result
        assert "—" not in result

    # ── Trades today ─────────────────────────────────────────────────────────

    def test_open_trades_shown(self):
        ctx = SessionContext(
            as_of=_NOW,
            trades_today=[
                TradeLogEntry(
                    timestamp=_NOW,
                    symbol="AAPL",
                    entry_price=150.1234,
                    direction="buy",
                    status="open",
                    unrealized_pnl=12.50,
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "Open position: AAPL buy at 150.1234" in result
        assert "(unrealized P&L: +12.50)" in result

    def test_no_open_positions_message(self):
        ctx = SessionContext(
            as_of=_NOW,
            trades_today=[
                TradeLogEntry(
                    timestamp=_NOW,
                    symbol="AAPL",
                    entry_price=150.0,
                    direction="buy",
                    status="closed",
                    unrealized_pnl=5.00,
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "No open positions." in result

    def test_closed_trades_net_pnl(self):
        ctx = SessionContext(
            as_of=_NOW,
            trades_today=[
                TradeLogEntry(
                    timestamp=_NOW,
                    symbol="AAPL",
                    entry_price=150.0,
                    direction="buy",
                    status="closed",
                    unrealized_pnl=10.00,
                ),
                TradeLogEntry(
                    timestamp=_NOW,
                    symbol="MSFT",
                    entry_price=300.0,
                    direction="sell",
                    status="closed",
                    unrealized_pnl=-3.50,
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "2 closed trade(s) today" in result
        assert "net P&L: +6.50" in result

    # ── Symbol characters ────────────────────────────────────────────────────

    def test_symbol_characters(self):
        ctx = SessionContext(
            as_of=_NOW,
            symbol_characters=[
                SymbolCharacterEntry(
                    symbol="AAPL",
                    signal_count_today=3,
                    direction_consistency=0.85,
                    last_signal_result="win",
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "AAPL character: 3 signals today" in result
        assert "direction consistency +0.85" in result
        assert "last result: win" in result

    def test_negative_direction_consistency(self):
        ctx = SessionContext(
            as_of=_NOW,
            symbol_characters=[
                SymbolCharacterEntry(
                    symbol="BTC-USD",
                    signal_count_today=1,
                    direction_consistency=-0.40,
                    last_signal_result="loss",
                ),
            ],
        )
        result = build_session_paragraph(ctx)
        assert "direction consistency -0.40" in result

    # ── Narrative ────────────────────────────────────────────────────────────

    def test_narrative_appended_verbatim(self):
        ctx = SessionContext(
            as_of=_NOW,
            narrative="Market opened strong but faded after CPI.",
        )
        result = build_session_paragraph(ctx)
        assert "Market opened strong but faded after CPI." in result

    # ── Combined context ─────────────────────────────────────────────────────

    def test_combined_context_joins_all_parts(self):
        ctx = SessionContext(
            as_of=_NOW,
            regime_history=[
                RegimeLogEntry(
                    timestamp=_NOW,
                    regime_type="TRENDING_UP",
                    confidence=0.80,
                ),
            ],
            signals_today=[
                SignalLogEntry(
                    timestamp=_NOW,
                    symbol="AAPL",
                    setup_type="equity_trend_pullback",
                    score=70.0,
                    claude_decision="approve",
                ),
            ],
            trades_today=[
                TradeLogEntry(
                    timestamp=_NOW,
                    symbol="AAPL",
                    entry_price=100.0,
                    direction="buy",
                    status="open",
                    unrealized_pnl=2.00,
                ),
            ],
            symbol_characters=[
                SymbolCharacterEntry(
                    symbol="AAPL",
                    signal_count_today=1,
                    direction_consistency=1.0,
                    last_signal_result="pending",
                ),
            ],
            narrative="Good session so far.",
        )
        result = build_session_paragraph(ctx)
        # All sections present
        assert "Regime has been TRENDING_UP" in result
        assert "1 signals evaluated today" in result
        assert "Open position: AAPL" in result
        assert "AAPL character:" in result
        assert "Good session so far." in result


# ─────────────────────────────────────────────────────────────────────────────
# build_strategic_paragraph
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildStrategicParagraph:
    """Tests for build_strategic_paragraph."""

    def _empty_ctx(self) -> StrategicContext:
        return StrategicContext(as_of=_NOW)

    # ── Empty context ────────────────────────────────────────────────────────

    def test_empty_context_returns_empty_string(self):
        assert build_strategic_paragraph(self._empty_ctx()) == ""

    # ── Setup performance ────────────────────────────────────────────────────

    def _perf_entries(self) -> list[SetupPerformanceEntry]:
        return [
            SetupPerformanceEntry(
                setup_type="crypto_mean_reversion",
                symbol="BTC-USD",
                regime_at_entry="TRENDING_UP",
                time_of_day_bucket="morning",
                win=True,
                pnl=50.0,
                hold_duration_minutes=25.0,
                date="2024-06-01",
            ),
            SetupPerformanceEntry(
                setup_type="crypto_mean_reversion",
                symbol="BTC-USD",
                regime_at_entry="TRENDING_UP",
                time_of_day_bucket="morning",
                win=False,
                pnl=-20.0,
                hold_duration_minutes=15.0,
                date="2024-06-02",
            ),
            SetupPerformanceEntry(
                setup_type="crypto_mean_reversion",
                symbol="BTC-USD",
                regime_at_entry="RANGING",
                time_of_day_bucket="afternoon",
                win=True,
                pnl=30.0,
                hold_duration_minutes=20.0,
                date="2024-06-03",
            ),
        ]

    def test_setup_performance_with_setup_and_symbol(self):
        ctx = StrategicContext(as_of=_NOW, setup_performance=self._perf_entries())
        result = build_strategic_paragraph(
            ctx, setup_type="crypto_mean_reversion", symbol="BTC-USD"
        )
        assert "Historical performance for BTC-USD crypto_mean_reversion:" in result
        assert "3 occurrences" in result
        assert "67% win rate" in result
        # avg win = (50 + 30) / 2 = 40.0
        assert "avg win +40.00" in result
        # avg loss = -20 / 1 = -20.0
        assert "avg loss -20.00" in result
        # avg pnl = (50 - 20 + 30) / 3 = 20.0
        assert "avg P&L +20.00" in result
        # avg hold = (25 + 15 + 20) / 3 = 20.0
        assert "avg hold 20 min" in result

    def test_setup_performance_label_setup_only(self):
        ctx = StrategicContext(as_of=_NOW, setup_performance=self._perf_entries())
        result = build_strategic_paragraph(ctx, setup_type="crypto_mean_reversion")
        assert "Historical performance for crypto_mean_reversion:" in result

    def test_setup_performance_label_symbol_only(self):
        ctx = StrategicContext(as_of=_NOW, setup_performance=self._perf_entries())
        result = build_strategic_paragraph(ctx, symbol="BTC-USD")
        assert "Historical performance for BTC-USD setups:" in result

    def test_setup_performance_label_neither(self):
        ctx = StrategicContext(as_of=_NOW, setup_performance=self._perf_entries())
        result = build_strategic_paragraph(ctx)
        assert "Historical performance for this setup:" in result

    def test_setup_performance_all_wins(self):
        entries = [
            SetupPerformanceEntry(
                setup_type="equity_trend_pullback",
                symbol="AAPL",
                regime_at_entry="TRENDING_UP",
                time_of_day_bucket="morning",
                win=True,
                pnl=10.0,
                hold_duration_minutes=30.0,
                date="2024-06-01",
            ),
        ]
        ctx = StrategicContext(as_of=_NOW, setup_performance=entries)
        result = build_strategic_paragraph(ctx)
        assert "100% win rate" in result
        assert "avg loss +0.00" in result  # no losers → avg_loss = 0

    # ── Regime transitions ───────────────────────────────────────────────────

    def test_regime_transitions(self):
        ctx = StrategicContext(
            as_of=_NOW,
            regime_transitions=[
                RegimeTransitionEntry(
                    from_regime="RANGING",
                    to_regime="TRENDING_UP",
                    duration_minutes=45.0,
                    count=3,
                ),
                RegimeTransitionEntry(
                    from_regime="TRENDING_UP",
                    to_regime="VOLATILE",
                    duration_minutes=120.0,
                    count=1,
                ),
            ],
        )
        result = build_strategic_paragraph(ctx)
        assert "Regime transition patterns:" in result
        assert "RANGING → TRENDING_UP (avg 45 min, 3x)" in result
        assert "TRENDING_UP → VOLATILE (avg 120 min, 1x)" in result
        # Semicolon-separated
        assert "; " in result

    # ── Veto patterns ────────────────────────────────────────────────────────

    def test_veto_patterns(self):
        ctx = StrategicContext(
            as_of=_NOW,
            relevant_veto_patterns=[
                VetoPatternEntry(
                    veto_reason="RSI overbought rejection",
                    setup_type="crypto_breakout",
                    count=4,
                    last_seen=datetime(2024, 6, 9, 16, 0, tzinfo=timezone.utc),
                ),
            ],
        )
        result = build_strategic_paragraph(ctx)
        assert "WARNING:" in result
        assert "'RSI overbought rejection' has vetoed crypto_breakout 4 time(s)" in result
        assert "last seen 2024-06-09" in result

    # ── Weekly trend ─────────────────────────────────────────────────────────

    def test_weekly_trend_uses_latest_entry(self):
        ctx = StrategicContext(
            as_of=_NOW,
            weekly_trend=[
                WeeklyPerformanceEntry(
                    week="2024-W22",
                    setup_type="crypto_mean_reversion",
                    trades=8,
                    win_rate=0.625,
                    avg_pnl=15.30,
                    sharpe=1.45,
                ),
                WeeklyPerformanceEntry(
                    week="2024-W23",
                    setup_type="crypto_mean_reversion",
                    trades=12,
                    win_rate=0.75,
                    avg_pnl=22.10,
                    sharpe=2.10,
                ),
            ],
        )
        result = build_strategic_paragraph(ctx)
        # Latest week is W23
        assert "Recent week (2024-W23):" in result
        assert "12 trades" in result
        assert "75% win rate" in result
        assert "avg P&L +22.10" in result
        assert "Sharpe 2.10" in result

    # ── Symbol learned behavior ──────────────────────────────────────────────

    def test_symbol_behavior_all_fields(self):
        ctx = StrategicContext(
            as_of=_NOW,
            symbol_behavior=SymbolLearnedBehaviorEntry(
                symbol="BTC-USD",
                setup_type="crypto_mean_reversion",
                optimal_rsi_entry=32.5,
                avg_reversion_depth=0.025,
                avg_bounce_magnitude=0.04,
                sample_size=18,
            ),
        )
        result = build_strategic_paragraph(ctx)
        assert "BTC-USD crypto_mean_reversion learned behavior:" in result
        assert "optimal RSI entry 32.5" in result
        assert "avg reversion depth 2.50%" in result
        assert "avg bounce 4.00%" in result
        assert "(sample size: 18)" in result

    def test_symbol_behavior_partial_fields(self):
        ctx = StrategicContext(
            as_of=_NOW,
            symbol_behavior=SymbolLearnedBehaviorEntry(
                symbol="AAPL",
                setup_type="equity_trend_pullback",
                optimal_rsi_entry=None,
                avg_reversion_depth=None,
                avg_bounce_magnitude=0.015,
                sample_size=5,
            ),
        )
        result = build_strategic_paragraph(ctx)
        assert "optimal RSI entry" not in result
        assert "avg reversion depth" not in result
        assert "avg bounce 1.50%" in result
        assert "(sample size: 5)" in result

    # ── Claude calibration ───────────────────────────────────────────────────

    def test_calibration_basic(self):
        cal = [
            ClaudeCalibrationEntry(
                date="2024-06-0{}".format(i),
                confidence_stated=0.70,
                outcome="win" if i <= 2 else "loss",
                setup_type="crypto_mean_reversion",
            )
            for i in range(1, 4)
        ]
        ctx = StrategicContext(as_of=_NOW, claude_calibration=cal)
        result = build_strategic_paragraph(ctx)
        # 3 trades, < 5 so no NOTE
        assert "Claude calibration (3 recent trades):" in result
        assert "stated avg confidence 70%" in result
        assert "actual win rate 67%" in result
        assert "NOTE:" not in result

    def test_calibration_overconfident(self):
        # 6 trades: avg conf 80%, actual win rate 33% (2 wins of 6)
        cal = [
            ClaudeCalibrationEntry(
                date=f"2024-06-0{i}",
                confidence_stated=0.80,
                outcome="win" if i <= 2 else "loss",
                setup_type="crypto_mean_reversion",
            )
            for i in range(1, 7)
        ]
        ctx = StrategicContext(as_of=_NOW, claude_calibration=cal)
        result = build_strategic_paragraph(ctx)
        assert "NOTE: Claude is overconfident" in result
        assert "stated confidence exceeds actual outcomes" in result

    def test_calibration_underconfident(self):
        # 5 trades: avg conf 30%, actual win rate 80% (4 wins of 5)
        cal = [
            ClaudeCalibrationEntry(
                date=f"2024-06-0{i}",
                confidence_stated=0.30,
                outcome="win" if i <= 4 else "loss",
                setup_type="crypto_mean_reversion",
            )
            for i in range(1, 6)
        ]
        ctx = StrategicContext(as_of=_NOW, claude_calibration=cal)
        result = build_strategic_paragraph(ctx)
        assert "NOTE: Claude is underconfident" in result
        assert "actual outcomes exceed stated confidence" in result

    def test_calibration_no_note_when_gap_small(self):
        # 5 trades: avg conf 60%, actual win rate 60% (3 wins of 5), gap = 0
        cal = [
            ClaudeCalibrationEntry(
                date=f"2024-06-0{i}",
                confidence_stated=0.60,
                outcome="win" if i <= 3 else "loss",
                setup_type="crypto_mean_reversion",
            )
            for i in range(1, 6)
        ]
        ctx = StrategicContext(as_of=_NOW, claude_calibration=cal)
        result = build_strategic_paragraph(ctx)
        assert "NOTE:" not in result

    # ── Combined strategic context ───────────────────────────────────────────

    def test_combined_strategic_context(self):
        ctx = StrategicContext(
            as_of=_NOW,
            setup_performance=[
                SetupPerformanceEntry(
                    setup_type="crypto_mean_reversion",
                    symbol="BTC-USD",
                    regime_at_entry="TRENDING_UP",
                    time_of_day_bucket="morning",
                    win=True,
                    pnl=50.0,
                    hold_duration_minutes=25.0,
                    date="2024-06-01",
                ),
            ],
            regime_transitions=[
                RegimeTransitionEntry(
                    from_regime="RANGING",
                    to_regime="TRENDING_UP",
                    duration_minutes=45.0,
                    count=2,
                ),
            ],
            relevant_veto_patterns=[
                VetoPatternEntry(
                    veto_reason="Low volume",
                    setup_type="crypto_mean_reversion",
                    count=1,
                    last_seen=datetime(2024, 6, 8, tzinfo=timezone.utc),
                ),
            ],
            weekly_trend=[
                WeeklyPerformanceEntry(
                    week="2024-W23",
                    setup_type="crypto_mean_reversion",
                    trades=5,
                    win_rate=0.60,
                    avg_pnl=10.0,
                    sharpe=1.0,
                ),
            ],
        )
        result = build_strategic_paragraph(
            ctx, setup_type="crypto_mean_reversion", symbol="BTC-USD"
        )
        assert "Historical performance for BTC-USD crypto_mean_reversion:" in result
        assert "Regime transition patterns:" in result
        assert "WARNING:" in result
        assert "Recent week (2024-W23):" in result
