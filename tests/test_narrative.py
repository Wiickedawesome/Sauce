"""Tests for sauce.memory.narrative — intraday narrative builder."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from sauce.memory.narrative import (
    _build_notable_observations,
    _build_open_momentum,
    _build_pnl_sentence,
    _build_regime_sentence,
    _build_setup_summary,
    _format_minutes,
    _format_pct,
    _regime_label,
    build_narrative,
)
from sauce.core.regime import RegimeDuration
from sauce.core.schemas import (
    IntradayNarrativeEntry,
    RegimeLogEntry,
    SignalLogEntry,
    TradeLogEntry,
)

NOW = datetime(2024, 6, 15, 16, 0, 0, tzinfo=timezone.utc)


# ── helpers ──────────────────────────────────────────────────────────────


def _make_spy_df(opens: list[float], closes: list[float]) -> pd.DataFrame:
    """Build a tiny SPY DataFrame with open/close columns."""
    return pd.DataFrame({"open": opens, "close": closes})


def _regime_entry(
    regime: str, confidence: float = 0.8, minutes_ago: int = 0
) -> RegimeLogEntry:
    ts = datetime(2024, 6, 15, 16, 0, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_ago)
    return RegimeLogEntry(
        timestamp=ts,
        regime_type=regime,  # type: ignore[arg-type]
        confidence=confidence,
    )


def _signal(
    decision: str,
    reason: str | None = None,
    symbol: str = "AAPL",
) -> SignalLogEntry:
    return SignalLogEntry(
        timestamp=NOW,
        symbol=symbol,
        setup_type="equity_trend_pullback",
        score=50.0,
        claude_decision=decision,  # type: ignore[arg-type]
        reason=reason,
    )


def _trade(
    status: str = "open",
    pnl: float = 0.0,
    symbol: str = "AAPL",
) -> TradeLogEntry:
    return TradeLogEntry(
        timestamp=NOW,
        symbol=symbol,
        entry_price=150.0,
        direction="buy",
        status=status,  # type: ignore[arg-type]
        unrealized_pnl=pnl,
    )


# ── _format_pct ──────────────────────────────────────────────────────────


class TestFormatPct:
    def test_positive(self) -> None:
        assert _format_pct(1.5) == "+1.5%"

    def test_negative(self) -> None:
        assert _format_pct(-2.3) == "-2.3%"

    def test_zero(self) -> None:
        assert _format_pct(0.0) == "+0.0%"

    def test_large(self) -> None:
        assert _format_pct(12.345) == "+12.3%"


# ── _regime_label ────────────────────────────────────────────────────────


class TestRegimeLabel:
    def test_trending_up(self) -> None:
        assert _regime_label("TRENDING_UP") == "Trending Up"

    def test_trending_down(self) -> None:
        assert _regime_label("TRENDING_DOWN") == "Trending Down"

    def test_ranging(self) -> None:
        assert _regime_label("RANGING") == "Ranging"

    def test_volatile(self) -> None:
        assert _regime_label("VOLATILE") == "Volatile"

    def test_dead(self) -> None:
        assert _regime_label("DEAD") == "Dead"

    def test_unknown_fallback(self) -> None:
        assert _regime_label("SOMETHING_ELSE") == "SOMETHING_ELSE"  # type: ignore[arg-type]


# ── _format_minutes ──────────────────────────────────────────────────────


class TestFormatMinutes:
    def test_under_hour(self) -> None:
        assert _format_minutes(45) == "45min"

    def test_exact_hour(self) -> None:
        assert _format_minutes(120) == "2h"

    def test_hour_and_minutes(self) -> None:
        assert _format_minutes(90) == "1h 30min"

    def test_zero(self) -> None:
        assert _format_minutes(0) == "0min"

    def test_just_over_hour(self) -> None:
        assert _format_minutes(61) == "1h 1min"


# ── _build_open_momentum ────────────────────────────────────────────────


class TestBuildOpenMomentum:
    def test_none_df(self) -> None:
        result = _build_open_momentum(None)
        assert "insufficient data" in result.lower()

    def test_too_short_df(self) -> None:
        df = _make_spy_df([100.0], [100.5])
        result = _build_open_momentum(df)
        assert "insufficient data" in result.lower()

    def test_first_open_zero(self) -> None:
        df = _make_spy_df([0.0, 100.0], [100.0, 101.0])
        result = _build_open_momentum(df)
        assert "insufficient data" in result.lower()

    def test_flat(self) -> None:
        # change < 0.1%: 100 → 100.05 = 0.05%
        df = _make_spy_df([100.0, 100.0], [100.05, 100.05])
        result = _build_open_momentum(df)
        assert "flat" in result.lower()

    def test_slight_upward(self) -> None:
        # change ~0.2%: 100 → 100.2
        df = _make_spy_df([100.0, 100.0], [100.2, 100.2])
        result = _build_open_momentum(df)
        assert "slight" in result.lower()
        assert "upward" in result.lower()

    def test_slight_downward(self) -> None:
        # change ~-0.2%: 100 → 99.8
        df = _make_spy_df([100.0, 100.0], [99.8, 99.8])
        result = _build_open_momentum(df)
        assert "slight" in result.lower()
        assert "downward" in result.lower()

    def test_moderate_upward(self) -> None:
        # change ~0.5%: 100 → 100.5
        df = _make_spy_df([100.0, 100.0], [100.5, 100.5])
        result = _build_open_momentum(df)
        assert "moderate" in result.lower()
        assert "upward" in result.lower()

    def test_strong_upward(self) -> None:
        # change ~1.0%: 100 → 101.0
        df = _make_spy_df([100.0, 100.0], [101.0, 101.0])
        result = _build_open_momentum(df)
        assert "strong" in result.lower()
        assert "upward" in result.lower()

    def test_strong_downward(self) -> None:
        # change ~-1.0%: 100 → 99.0
        df = _make_spy_df([100.0, 100.0], [99.0, 99.0])
        result = _build_open_momentum(df)
        assert "strong" in result.lower()
        assert "downward" in result.lower()

    def test_contains_pct_format(self) -> None:
        df = _make_spy_df([100.0, 100.0], [101.0, 101.0])
        result = _build_open_momentum(df)
        assert "%" in result
        assert "SPY" in result


# ── _build_regime_sentence ──────────────────────────────────────────────


class TestBuildRegimeSentence:
    def test_no_data(self) -> None:
        result = _build_regime_sentence(None, [])
        assert result == "No regime data available."

    def test_with_duration_not_aging(self) -> None:
        duration = RegimeDuration(
            regime_type="TRENDING_UP",
            active_minutes=60.0,
            aging_out=False,
        )
        result = _build_regime_sentence(duration, [])
        assert "Trending Up" in result
        assert "1h" in result
        assert "imminent" not in result

    def test_with_duration_aging(self) -> None:
        duration = RegimeDuration(
            regime_type="VOLATILE",
            active_minutes=180.0,
            aging_out=True,
        )
        result = _build_regime_sentence(duration, [])
        assert "Volatile" in result
        assert "imminent" in result

    def test_no_duration_but_history(self) -> None:
        history = [_regime_entry("RANGING")]
        result = _build_regime_sentence(None, history)
        assert "Ranging" in result
        assert "duration not computed" in result


# ── _build_setup_summary ────────────────────────────────────────────────


class TestBuildSetupSummary:
    def test_no_signals(self) -> None:
        result = _build_setup_summary([], [])
        assert result == "No setups detected today."

    def test_approved_closed_profitable(self) -> None:
        signals = [_signal("approve")]
        trades = [_trade("closed", pnl=100.0)]
        result = _build_setup_summary(signals, trades)
        assert "1 setup detected" in result
        assert "1 approved" in result
        assert "profitable" in result.lower() or "profit" in result.lower()

    def test_approved_closed_at_loss(self) -> None:
        signals = [_signal("approve")]
        trades = [_trade("closed", pnl=-50.0)]
        result = _build_setup_summary(signals, trades)
        assert "1 approved" in result
        assert "loss" in result.lower()

    def test_approved_open_trade(self) -> None:
        signals = [_signal("approve")]
        trades = [_trade("open", pnl=25.0)]
        result = _build_setup_summary(signals, trades)
        assert "1 approved" in result
        assert "open" in result.lower()

    def test_rejected_with_reason(self) -> None:
        signals = [_signal("reject", reason="position size too large")]
        result = _build_setup_summary(signals, [])
        assert "1 rejected" in result
        assert "position size too large" in result

    def test_held(self) -> None:
        signals = [_signal("hold")]
        result = _build_setup_summary(signals, [])
        assert "1 held" in result

    def test_mixed(self) -> None:
        signals = [
            _signal("approve"),
            _signal("reject", reason="risk too high"),
            _signal("hold"),
        ]
        trades = [_trade("open", pnl=10.0)]
        result = _build_setup_summary(signals, trades)
        assert "3 setups detected" in result
        assert "1 approved" in result
        assert "1 rejected" in result
        assert "1 held" in result


# ── _build_pnl_sentence ─────────────────────────────────────────────────


class TestBuildPnlSentence:
    def test_no_trades(self) -> None:
        result = _build_pnl_sentence([])
        assert result == "No trades today."

    def test_positive_pnl(self) -> None:
        trades = [_trade("open", pnl=1.5), _trade("closed", pnl=0.5)]
        result = _build_pnl_sentence(trades)
        assert "+2.0%" in result
        assert "1 open" in result
        assert "1 closed" in result

    def test_negative_pnl(self) -> None:
        trades = [_trade("closed", pnl=-3.0)]
        result = _build_pnl_sentence(trades)
        assert "-3.0%" in result
        assert "1 closed" in result


# ── _build_notable_observations ──────────────────────────────────────────


class TestBuildNotableObservations:
    def test_empty_history(self) -> None:
        result = _build_notable_observations([], None)
        assert result == ""

    def test_stable_regime(self) -> None:
        duration = RegimeDuration(
            regime_type="TRENDING_UP",
            active_minutes=150.0,
            aging_out=False,
        )
        history = [_regime_entry("TRENDING_UP")]
        result = _build_notable_observations(history, duration)
        assert "Notable:" in result
        assert "stable" in result.lower()

    def test_frequent_regime_changes(self) -> None:
        # Need ≥ 3 entries total, with ≥ 3 unique regimes in last 4
        history = [
            _regime_entry("TRENDING_UP", minutes_ago=30),
            _regime_entry("VOLATILE", minutes_ago=20),
            _regime_entry("RANGING", minutes_ago=10),
        ]
        result = _build_notable_observations(history, None)
        assert "Notable:" in result
        assert "indecisive" in result.lower()

    def test_declining_confidence(self) -> None:
        # Strictly decreasing over VOLUME_TREND_LOOKBACK(4) entries
        history = [
            _regime_entry("TRENDING_UP", confidence=0.9, minutes_ago=40),
            _regime_entry("TRENDING_UP", confidence=0.8, minutes_ago=30),
            _regime_entry("TRENDING_UP", confidence=0.7, minutes_ago=20),
            _regime_entry("TRENDING_UP", confidence=0.6, minutes_ago=10),
        ]
        result = _build_notable_observations(history, None)
        assert "Notable:" in result
        assert "declining" in result.lower()

    def test_no_observations_when_not_triggered(self) -> None:
        # Short stable regime, not enough history for patterns
        duration = RegimeDuration(
            regime_type="TRENDING_UP",
            active_minutes=30.0,
            aging_out=False,
        )
        history = [_regime_entry("TRENDING_UP")]
        result = _build_notable_observations(history, duration)
        assert result == ""


# ── build_narrative (integration) ────────────────────────────────────────


class TestBuildNarrative:
    def test_returns_intraday_narrative_entry(self) -> None:
        spy_df = _make_spy_df([100.0, 100.0], [101.0, 101.0])
        history = [_regime_entry("TRENDING_UP")]
        duration = RegimeDuration(
            regime_type="TRENDING_UP",
            active_minutes=60.0,
        )
        signals = [_signal("approve")]
        trades = [_trade("open", pnl=1.0)]

        result = build_narrative(spy_df, history, duration, signals, trades, NOW)

        assert isinstance(result, IntradayNarrativeEntry)
        assert result.timestamp == NOW
        assert len(result.narrative_text) > 0

    def test_narrative_contains_key_sections(self) -> None:
        spy_df = _make_spy_df([100.0, 100.0], [101.0, 101.0])
        history = [_regime_entry("TRENDING_UP")]
        duration = RegimeDuration(
            regime_type="TRENDING_UP",
            active_minutes=60.0,
        )
        signals = [_signal("approve")]
        trades = [_trade("open", pnl=1.0)]

        result = build_narrative(spy_df, history, duration, signals, trades, NOW)
        text = result.narrative_text

        # Must contain momentum section
        assert "momentum" in text.lower() or "flat" in text.lower()
        # Must contain regime info
        assert "Trending Up" in text
        # Must contain setup info
        assert "setup" in text.lower()
        # Must contain PnL info
        assert "%" in text

    def test_narrative_with_no_data(self) -> None:
        result = build_narrative(None, [], None, [], [], NOW)

        assert isinstance(result, IntradayNarrativeEntry)
        assert "insufficient data" in result.narrative_text.lower()
        assert "No regime data" in result.narrative_text
        assert "No setups detected" in result.narrative_text
        assert "No trades today" in result.narrative_text
