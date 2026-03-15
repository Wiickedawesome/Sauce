"""
agents/debate.py — Bull/Bear debate layer for trade quality.

Deterministic (no LLM calls). Both sides receive the same signal + indicators
and produce structured arguments for/against the trade. The debate transcript
is forwarded to the Supervisor for final arbitration.

Inspired by: QuantDinger (7-agent Bull vs Bear debate) + NOFX (AI competition).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sauce.core.schemas import Indicators, Signal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Argument:
    """Single argument from bull or bear side."""

    label: str
    detail: str
    weight: float  # 0.0–1.0 importance


@dataclass(frozen=True, slots=True)
class DebateResult:
    """Complete debate transcript for one signal."""

    symbol: str
    side: str  # original signal side
    bull_arguments: tuple[Argument, ...]
    bear_arguments: tuple[Argument, ...]
    bull_score: float  # weighted sum of bull arguments
    bear_score: float  # weighted sum of bear arguments
    verdict: str  # "bull_wins", "bear_wins", "contested"
    confidence_adjustment: float  # suggested adjustment to signal confidence

    def summary(self) -> str:
        """One-paragraph summary for the supervisor prompt."""
        bull_pts = "; ".join(a.label for a in self.bull_arguments)
        bear_pts = "; ".join(a.label for a in self.bear_arguments)
        return (
            f"DEBATE [{self.verdict.upper()}] bull={self.bull_score:.2f} "
            f"bear={self.bear_score:.2f} | "
            f"Bull: {bull_pts} | Bear: {bear_pts}"
        )


# ── Bull arguments ─────────────────────────────────────────────────────────


def _bull_case(signal: Signal, ind: Indicators) -> list[Argument]:
    """Build bullish arguments from indicators and signal evidence."""
    args: list[Argument] = []

    # Trend alignment
    if ind.sma_20 and ind.sma_50 and ind.sma_20 > ind.sma_50:
        args.append(Argument(
            label="Golden cross (SMA20 > SMA50)",
            detail=f"SMA20={ind.sma_20:.2f} above SMA50={ind.sma_50:.2f}",
            weight=0.8,
        ))

    # RSI not overbought and showing strength
    if ind.rsi_14 is not None:
        if 40 <= ind.rsi_14 <= 65:
            args.append(Argument(
                label="RSI in healthy range",
                detail=f"RSI={ind.rsi_14:.1f} — not overbought, room to run",
                weight=0.6,
            ))
        elif ind.rsi_14 < 35:
            args.append(Argument(
                label="Oversold bounce setup",
                detail=f"RSI={ind.rsi_14:.1f} — potential reversal to the upside",
                weight=0.7,
            ))

    # Volume confirmation
    if ind.volume_ratio is not None and ind.volume_ratio > 1.2:
        args.append(Argument(
            label="Above-average volume",
            detail=f"Volume ratio={ind.volume_ratio:.2f}x — strong participation",
            weight=0.7,
        ))

    # MACD bullish
    if ind.macd_histogram is not None and ind.macd_histogram > 0:
        args.append(Argument(
            label="MACD histogram positive",
            detail=f"Histogram={ind.macd_histogram:.4f} — momentum expanding",
            weight=0.6,
        ))
    if (ind.macd_line is not None and ind.macd_signal is not None
            and ind.macd_line > ind.macd_signal):
        args.append(Argument(
            label="MACD above signal line",
            detail=f"MACD={ind.macd_line:.4f} > Signal={ind.macd_signal:.4f}",
            weight=0.5,
        ))

    # Bollinger Band position (price near lower band = value)
    if ind.bb_lower is not None and ind.bb_upper is not None:
        mid = signal.evidence.price_reference.mid
        if mid and mid <= ind.bb_lower * 1.02:
            args.append(Argument(
                label="Price near lower Bollinger Band",
                detail=f"Price={mid:.2f} near BB_lower={ind.bb_lower:.2f} — value zone",
                weight=0.6,
            ))

    # High confidence from research
    if signal.confidence >= 0.70:
        args.append(Argument(
            label="High research confidence",
            detail=f"Confidence={signal.confidence:.2f} — strong conviction",
            weight=0.5,
        ))

    # ATR-based stop is tight (good risk/reward)
    if ind.atr_14 is not None:
        mid = signal.evidence.price_reference.mid
        if mid and mid > 0 and ind.atr_14 / mid < 0.02:
            args.append(Argument(
                label="Low volatility relative to price",
                detail=f"ATR/Price={ind.atr_14/mid:.4f} — tight stop possible",
                weight=0.4,
            ))

    # Stochastic oversold
    if ind.stoch_k is not None and ind.stoch_k < 25:
        args.append(Argument(
            label="Stochastic oversold",
            detail=f"Stoch_K={ind.stoch_k:.1f} — potential upward reversal",
            weight=0.5,
        ))

    return args


# ── Bear arguments ─────────────────────────────────────────────────────────


def _bear_case(signal: Signal, ind: Indicators) -> list[Argument]:
    """Build bearish arguments from indicators and signal evidence."""
    args: list[Argument] = []

    # Trend misalignment
    if ind.sma_20 and ind.sma_50 and ind.sma_20 < ind.sma_50:
        args.append(Argument(
            label="Death cross (SMA20 < SMA50)",
            detail=f"SMA20={ind.sma_20:.2f} below SMA50={ind.sma_50:.2f}",
            weight=0.8,
        ))

    # RSI overbought
    if ind.rsi_14 is not None and ind.rsi_14 > 70:
        args.append(Argument(
            label="RSI overbought",
            detail=f"RSI={ind.rsi_14:.1f} — potential reversal or exhaustion",
            weight=0.7,
        ))

    # Weak volume
    if ind.volume_ratio is not None and ind.volume_ratio < 0.7:
        args.append(Argument(
            label="Below-average volume",
            detail=f"Volume ratio={ind.volume_ratio:.2f}x — weak conviction",
            weight=0.6,
        ))

    # MACD bearish
    if ind.macd_histogram is not None and ind.macd_histogram < 0:
        args.append(Argument(
            label="MACD histogram negative",
            detail=f"Histogram={ind.macd_histogram:.4f} — momentum fading",
            weight=0.6,
        ))
    if (ind.macd_line is not None and ind.macd_signal is not None
            and ind.macd_line < ind.macd_signal):
        args.append(Argument(
            label="MACD below signal line",
            detail=f"MACD={ind.macd_line:.4f} < Signal={ind.macd_signal:.4f}",
            weight=0.5,
        ))

    # Price near upper Bollinger Band (overextended)
    if ind.bb_lower is not None and ind.bb_upper is not None:
        mid = signal.evidence.price_reference.mid
        if mid and mid >= ind.bb_upper * 0.98:
            args.append(Argument(
                label="Price near upper Bollinger Band",
                detail=f"Price={mid:.2f} near BB_upper={ind.bb_upper:.2f} — overextended",
                weight=0.6,
            ))

    # Low confidence from research
    if signal.confidence < 0.55:
        args.append(Argument(
            label="Low research confidence",
            detail=f"Confidence={signal.confidence:.2f} — borderline conviction",
            weight=0.5,
        ))

    # High ATR relative to price (wide stop = poor risk/reward)
    if ind.atr_14 is not None:
        mid = signal.evidence.price_reference.mid
        if mid and mid > 0 and ind.atr_14 / mid > 0.04:
            args.append(Argument(
                label="High volatility relative to price",
                detail=f"ATR/Price={ind.atr_14/mid:.4f} — wide stop required",
                weight=0.5,
            ))

    # Stochastic overbought
    if ind.stoch_k is not None and ind.stoch_k > 80:
        args.append(Argument(
            label="Stochastic overbought",
            detail=f"Stoch_K={ind.stoch_k:.1f} — potential downward reversal",
            weight=0.5,
        ))

    # Bear case from research itself (Claude flagged risks)
    if signal.bear_case:
        args.append(Argument(
            label="Research bear case",
            detail=signal.bear_case[:150],
            weight=0.4,
        ))

    return args


# ── Scoring & verdict ──────────────────────────────────────────────────────


def _score(arguments: list[Argument]) -> float:
    """Compute weighted score for a side's arguments."""
    if not arguments:
        return 0.0
    return sum(a.weight for a in arguments)


def _verdict(bull_score: float, bear_score: float) -> str:
    """Determine debate outcome."""
    margin = bull_score - bear_score
    if margin > 0.5:
        return "bull_wins"
    if margin < -0.5:
        return "bear_wins"
    return "contested"


def _confidence_adjustment(verdict: str, bull_score: float, bear_score: float) -> float:
    """Suggest a confidence adjustment based on debate outcome.

    Returns a value in [-0.10, +0.05] — bearish debate has more penalty
    power than bullish has bonus, to bias toward caution.
    """
    if verdict == "bull_wins":
        return min(0.05, (bull_score - bear_score) * 0.02)
    if verdict == "bear_wins":
        return max(-0.10, (bull_score - bear_score) * 0.03)
    # contested — small penalty for uncertainty
    return -0.02


# ── Public API ─────────────────────────────────────────────────────────────


def run_debate(signal: Signal) -> DebateResult:
    """
    Run a deterministic bull/bear debate for a signal.

    Deliberately synchronous: this is pure computation with no I/O,
    so ``async def`` would add coroutine overhead for zero benefit.

    Parameters
    ----------
    signal : Signal
        The research signal to debate. Must have side != "hold".

    Returns
    -------
    DebateResult
        Complete debate transcript with arguments, scores, and verdict.
    """
    ind = signal.evidence.indicators
    bull_args = _bull_case(signal, ind)
    bear_args = _bear_case(signal, ind)

    # For sell signals, swap the framing: bear arguments support the sell,
    # bull arguments argue against it
    if signal.side == "sell":
        bull_args, bear_args = bear_args, bull_args

    bull_score = _score(bull_args)
    bear_score = _score(bear_args)
    v = _verdict(bull_score, bear_score)
    adj = _confidence_adjustment(v, bull_score, bear_score)

    result = DebateResult(
        symbol=signal.symbol,
        side=signal.side,
        bull_arguments=tuple(bull_args),
        bear_arguments=tuple(bear_args),
        bull_score=round(bull_score, 4),
        bear_score=round(bear_score, 4),
        verdict=v,
        confidence_adjustment=round(adj, 4),
    )

    logger.info(
        "debate[%s]: %s (bull=%.2f bear=%.2f adj=%.4f)",
        signal.symbol, v, bull_score, bear_score, adj,
    )
    return result
