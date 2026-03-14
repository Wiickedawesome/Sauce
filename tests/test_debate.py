"""Tests for sauce.agents.debate — Bull/Bear debate layer."""

from datetime import datetime, timezone

import pytest

from sauce.agents.debate import (
    Argument,
    DebateResult,
    _bear_case,
    _bull_case,
    _confidence_adjustment,
    _score,
    _verdict,
    run_debate,
)
from sauce.core.schemas import Evidence, Indicators, PriceReference, Signal


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_signal(
    symbol: str = "SPY",
    side: str = "buy",
    confidence: float = 0.65,
    bear_case: str = "",
    *,
    sma_20: float | None = 450.0,
    sma_50: float | None = 440.0,
    rsi_14: float | None = 55.0,
    atr_14: float | None = 3.0,
    volume_ratio: float | None = 1.3,
    macd_histogram: float | None = 0.5,
    macd_line: float | None = 1.0,
    macd_signal: float | None = 0.8,
    bb_lower: float | None = 430.0,
    bb_upper: float | None = 460.0,
    stoch_k: float | None = 50.0,
    mid: float = 450.0,
) -> Signal:
    return Signal(
        symbol=symbol,
        side=side,
        confidence=confidence,
        bear_case=bear_case,
        evidence=Evidence(
            symbol=symbol,
            price_reference=PriceReference(
                symbol=symbol,
                bid=mid - 0.1,
                ask=mid + 0.1,
                mid=mid,
                as_of=datetime.now(timezone.utc),
            ),
            indicators=Indicators(
                sma_20=sma_20,
                sma_50=sma_50,
                rsi_14=rsi_14,
                atr_14=atr_14,
                volume_ratio=volume_ratio,
                macd_histogram=macd_histogram,
                macd_line=macd_line,
                macd_signal=macd_signal,
                bb_lower=bb_lower,
                bb_upper=bb_upper,
                stoch_k=stoch_k,
            ),
            as_of=datetime.now(timezone.utc),
        ),
        reasoning="test reasoning",
        as_of=datetime.now(timezone.utc),
        prompt_version="v1",
    )


# ── Argument dataclass ─────────────────────────────────────────────────────


class TestArgument:
    def test_creation(self):
        a = Argument(label="test", detail="detail", weight=0.5)
        assert a.label == "test"
        assert a.weight == 0.5

    def test_frozen(self):
        a = Argument(label="test", detail="detail", weight=0.5)
        with pytest.raises(AttributeError):
            a.label = "other"  # type: ignore[misc]


# ── DebateResult ───────────────────────────────────────────────────────────


class TestDebateResult:
    def test_summary(self):
        result = DebateResult(
            symbol="SPY",
            side="buy",
            bull_arguments=(
                Argument(label="Trend up", detail="d", weight=0.8),
            ),
            bear_arguments=(
                Argument(label="Volume low", detail="d", weight=0.5),
            ),
            bull_score=0.8,
            bear_score=0.5,
            verdict="contested",
            confidence_adjustment=-0.02,
        )
        s = result.summary()
        assert "CONTESTED" in s
        assert "Trend up" in s
        assert "Volume low" in s

    def test_frozen(self):
        result = DebateResult(
            symbol="X", side="buy",
            bull_arguments=(), bear_arguments=(),
            bull_score=0, bear_score=0,
            verdict="contested", confidence_adjustment=0,
        )
        with pytest.raises(AttributeError):
            result.symbol = "Y"  # type: ignore[misc]


# ── Bull case ──────────────────────────────────────────────────────────────


class TestBullCase:
    def test_golden_cross(self):
        sig = _make_signal(sma_20=450, sma_50=440)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("Golden cross" in l for l in labels)

    def test_no_golden_cross_when_death_cross(self):
        sig = _make_signal(sma_20=430, sma_50=440)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert not any("Golden cross" in l for l in labels)

    def test_rsi_healthy(self):
        sig = _make_signal(rsi_14=55)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("RSI in healthy" in l for l in labels)

    def test_rsi_oversold(self):
        sig = _make_signal(rsi_14=28)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("Oversold" in l for l in labels)

    def test_above_average_volume(self):
        sig = _make_signal(volume_ratio=1.5)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("volume" in l.lower() for l in labels)

    def test_low_volume_no_bull_arg(self):
        sig = _make_signal(volume_ratio=0.5)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert not any("Above-average volume" in l for l in labels)

    def test_macd_positive(self):
        sig = _make_signal(macd_histogram=0.5)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("MACD histogram positive" in l for l in labels)

    def test_high_confidence(self):
        sig = _make_signal(confidence=0.75)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("High research confidence" in l for l in labels)

    def test_stoch_oversold(self):
        sig = _make_signal(stoch_k=20)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("Stochastic oversold" in l for l in labels)

    def test_price_near_lower_bb(self):
        sig = _make_signal(bb_lower=449, bb_upper=460, mid=449.5)
        args = _bull_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("lower Bollinger" in l for l in labels)

    def test_no_args_with_empty_indicators(self):
        sig = _make_signal(
            sma_20=None, sma_50=None, rsi_14=None, atr_14=None,
            volume_ratio=None, macd_histogram=None, macd_line=None,
            macd_signal=None, bb_lower=None, bb_upper=None,
            stoch_k=None, confidence=0.55,
        )
        args = _bull_case(sig, sig.evidence.indicators)
        assert len(args) == 0


# ── Bear case ──────────────────────────────────────────────────────────────


class TestBearCase:
    def test_death_cross(self):
        sig = _make_signal(sma_20=430, sma_50=440)
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("Death cross" in l for l in labels)

    def test_rsi_overbought(self):
        sig = _make_signal(rsi_14=75)
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("RSI overbought" in l for l in labels)

    def test_weak_volume(self):
        sig = _make_signal(volume_ratio=0.5)
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("Below-average volume" in l for l in labels)

    def test_macd_negative(self):
        sig = _make_signal(macd_histogram=-0.3)
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("MACD histogram negative" in l for l in labels)

    def test_macd_below_signal(self):
        sig = _make_signal(macd_line=0.5, macd_signal=0.8)
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("MACD below signal" in l for l in labels)

    def test_low_confidence(self):
        sig = _make_signal(confidence=0.45)
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("Low research confidence" in l for l in labels)

    def test_stoch_overbought(self):
        sig = _make_signal(stoch_k=85)
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("Stochastic overbought" in l for l in labels)

    def test_price_near_upper_bb(self):
        sig = _make_signal(bb_lower=430, bb_upper=460, mid=459)
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("upper Bollinger" in l for l in labels)

    def test_high_volatility(self):
        sig = _make_signal(atr_14=20, mid=450)  # ATR/price = 0.044
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("High volatility" in l for l in labels)

    def test_bear_case_text_included(self):
        sig = _make_signal(bear_case="Earnings risk this week")
        args = _bear_case(sig, sig.evidence.indicators)
        labels = [a.label for a in args]
        assert any("Research bear case" in l for l in labels)

    def test_no_args_with_neutral_indicators(self):
        sig = _make_signal(
            sma_20=450, sma_50=440,  # golden cross — no death cross
            rsi_14=55,  # neutral
            volume_ratio=1.0,  # neutral
            macd_histogram=0.1,  # positive
            macd_line=1.0, macd_signal=0.8,  # above signal
            bb_lower=430, bb_upper=460,  # price at 450 — middle
            stoch_k=50,  # neutral
            atr_14=3.0, mid=450,  # ATR/price = 0.0067 — low
            confidence=0.65,  # not low
            bear_case="",
        )
        args = _bear_case(sig, sig.evidence.indicators)
        assert len(args) == 0


# ── Scoring ────────────────────────────────────────────────────────────────


class TestScoring:
    def test_empty_arguments(self):
        assert _score([]) == 0.0

    def test_single_argument(self):
        assert _score([Argument(label="x", detail="d", weight=0.7)]) == 0.7

    def test_multiple_arguments(self):
        args = [
            Argument(label="a", detail="d", weight=0.5),
            Argument(label="b", detail="d", weight=0.3),
        ]
        assert abs(_score(args) - 0.8) < 1e-9


# ── Verdict ────────────────────────────────────────────────────────────────


class TestVerdict:
    def test_bull_wins(self):
        assert _verdict(2.0, 1.0) == "bull_wins"

    def test_bear_wins(self):
        assert _verdict(0.5, 1.5) == "bear_wins"

    def test_contested(self):
        assert _verdict(1.0, 1.0) == "contested"

    def test_contested_narrow_margin(self):
        assert _verdict(1.2, 0.9) == "contested"  # margin 0.3 < 0.5


# ── Confidence adjustment ─────────────────────────────────────────────────


class TestConfidenceAdjustment:
    def test_bull_wins_positive(self):
        adj = _confidence_adjustment("bull_wins", 2.0, 1.0)
        assert adj > 0
        assert adj <= 0.05

    def test_bear_wins_negative(self):
        adj = _confidence_adjustment("bear_wins", 0.5, 2.0)
        assert adj < 0
        assert adj >= -0.10

    def test_contested_small_penalty(self):
        adj = _confidence_adjustment("contested", 1.0, 1.0)
        assert adj == -0.02

    def test_bull_cap(self):
        adj = _confidence_adjustment("bull_wins", 10.0, 1.0)
        assert adj <= 0.05

    def test_bear_floor(self):
        adj = _confidence_adjustment("bear_wins", 0.0, 10.0)
        assert adj >= -0.10


# ── run_debate integration ─────────────────────────────────────────────────


class TestRunDebate:
    def test_buy_signal_bullish_indicators(self):
        """Bullish indicators should produce bull_wins or contested."""
        sig = _make_signal(
            sma_20=450, sma_50=440,
            rsi_14=55, volume_ratio=1.5,
            macd_histogram=0.5, macd_line=1.0, macd_signal=0.8,
            stoch_k=50,
        )
        result = run_debate(sig)
        assert isinstance(result, DebateResult)
        assert result.symbol == "SPY"
        assert result.side == "buy"
        assert result.bull_score > 0
        assert len(result.bull_arguments) > 0
        assert result.verdict in ("bull_wins", "contested")

    def test_buy_signal_bearish_indicators(self):
        """Bearish indicators should produce bear_wins or contested."""
        sig = _make_signal(
            sma_20=430, sma_50=440,
            rsi_14=75, volume_ratio=0.5,
            macd_histogram=-0.5, macd_line=0.5, macd_signal=0.8,
            stoch_k=85, confidence=0.45,
            atr_14=20, mid=450,
        )
        result = run_debate(sig)
        assert result.bear_score > result.bull_score
        assert result.verdict == "bear_wins"
        assert result.confidence_adjustment < 0

    def test_sell_signal_swaps_sides(self):
        """For sell signals, bull/bear args are swapped."""
        sig = _make_signal(side="sell", sma_20=430, sma_50=440)
        result = run_debate(sig)
        # In a sell, death cross (bearish indicator) should be a bull arg
        # (supporting the sell), not a bear arg (opposing it)
        bull_labels = [a.label for a in result.bull_arguments]
        assert any("Death cross" in l for l in bull_labels)

    def test_summary_format(self):
        sig = _make_signal()
        result = run_debate(sig)
        s = result.summary()
        assert "DEBATE" in s
        assert result.verdict.upper() in s

    def test_empty_indicators(self):
        """Should not crash with empty indicators — just few/no arguments."""
        sig = _make_signal(
            sma_20=None, sma_50=None, rsi_14=None, atr_14=None,
            volume_ratio=None, macd_histogram=None, macd_line=None,
            macd_signal=None, bb_lower=None, bb_upper=None,
            stoch_k=None, confidence=0.55,
        )
        result = run_debate(sig)
        assert isinstance(result, DebateResult)
        assert result.bull_score == 0.0
        assert result.bear_score == 0.0
        assert result.verdict == "contested"

    def test_confidence_adjustment_range(self):
        """Adjustment should always be in [-0.10, +0.05]."""
        for rsi in [20, 40, 55, 75, 90]:
            for vol in [0.3, 0.8, 1.5]:
                sig = _make_signal(rsi_14=rsi, volume_ratio=vol)
                result = run_debate(sig)
                assert -0.10 <= result.confidence_adjustment <= 0.05

    def test_crypto_symbol(self):
        sig = _make_signal(symbol="BTC/USD")
        result = run_debate(sig)
        assert result.symbol == "BTC/USD"
