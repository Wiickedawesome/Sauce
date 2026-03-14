"""
signals/confluence.py — Weighted confluence scoring + signal tiers.

Takes a MultiTimeframeContext and computes:
  1. Per-timeframe directional votes (bullish / bearish / neutral)
  2. Weighted confluence score (-1.0 to +1.0)
  3. Signal tier assignment (S1=strongest … S4=weakest)

Signal tiers integrate with the existing confidence floor in the research agent:
  S1 → confidence bonus +0.10  (strong multi-TF alignment)
  S2 → confidence bonus +0.05  (moderate alignment)
  S3 → no adjustment           (mixed / weak alignment)
  S4 → confidence penalty -0.10 (counter-trend, conflicting)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sauce.signals.timeframes import MultiTimeframeContext


class SignalTier(str, Enum):
    """Multi-timeframe signal quality tier."""

    S1 = "S1"  # Strong alignment across ≥3 timeframes
    S2 = "S2"  # Moderate alignment (2 timeframes agree)
    S3 = "S3"  # Mixed or insufficient data
    S4 = "S4"  # Conflicting signals across timeframes


# Weights for each timeframe label.  Higher timeframes get more weight
# because they represent stronger, more durable trends.
TIMEFRAME_WEIGHTS: dict[str, float] = {
    "5m": 0.10,
    "15m": 0.15,
    "1h": 0.25,
    "4h": 0.25,
    "1d": 0.25,
}


@dataclass(frozen=True)
class ConfluenceResult:
    """Output of multi-timeframe confluence analysis."""

    score: float              # -1.0 (bearish) to +1.0 (bullish)
    tier: SignalTier
    bullish_count: int        # timeframes with bullish trend+momentum
    bearish_count: int        # timeframes with bearish trend+momentum
    neutral_count: int
    confidence_adjustment: float  # delta to apply to Claude's confidence
    summary: str              # human-readable one-liner

    def to_prompt_dict(self) -> dict:
        """Serialize for prompt injection."""
        return {
            "confluence_score": round(self.score, 3),
            "signal_tier": self.tier.value,
            "bullish_timeframes": self.bullish_count,
            "bearish_timeframes": self.bearish_count,
            "neutral_timeframes": self.neutral_count,
            "confidence_adjustment": self.confidence_adjustment,
            "summary": self.summary,
        }


def _vote(trend: str, momentum: str) -> float:
    """Convert trend+momentum into a directional vote: +1, -1, or 0."""
    t_score = {"bullish": 1.0, "bearish": -1.0}.get(trend, 0.0)
    m_score = {"bullish": 1.0, "bearish": -1.0}.get(momentum, 0.0)
    # Average of trend and momentum
    combined = (t_score + m_score) / 2.0
    return combined


def compute_confluence(ctx: MultiTimeframeContext) -> ConfluenceResult:
    """
    Compute weighted confluence score and signal tier from multi-TF context.

    Returns a ConfluenceResult with:
      - score: weighted directional score (-1.0 to +1.0)
      - tier: S1-S4 quality tier
      - confidence_adjustment: delta for Claude's output confidence
    """
    if not ctx.analyses:
        return ConfluenceResult(
            score=0.0,
            tier=SignalTier.S3,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            confidence_adjustment=0.0,
            summary="No multi-timeframe data available",
        )

    weighted_sum = 0.0
    total_weight = 0.0
    bullish = 0
    bearish = 0
    neutral = 0

    for analysis in ctx.analyses:
        weight = TIMEFRAME_WEIGHTS.get(analysis.label, 0.15)
        vote = _vote(analysis.trend, analysis.momentum)
        weighted_sum += vote * weight
        total_weight += weight

        if vote > 0.25:
            bullish += 1
        elif vote < -0.25:
            bearish += 1
        else:
            neutral += 1

    score = weighted_sum / total_weight if total_weight > 0 else 0.0
    # Clamp to [-1, 1]
    score = max(-1.0, min(1.0, score))

    n_aligned = max(bullish, bearish)
    n_total = bullish + bearish + neutral

    # ── Tier assignment ───────────────────────────────────────────────────
    if n_aligned >= 3 and abs(score) >= 0.5:
        tier = SignalTier.S1
        confidence_adj = 0.10
    elif n_aligned >= 2 and abs(score) >= 0.3:
        tier = SignalTier.S2
        confidence_adj = 0.05
    elif bullish > 0 and bearish > 0 and abs(score) < 0.2:
        tier = SignalTier.S4
        confidence_adj = -0.10
    else:
        tier = SignalTier.S3
        confidence_adj = 0.0

    # ── Summary ───────────────────────────────────────────────────────────
    direction = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
    summary = (
        f"{direction.capitalize()} confluence ({score:+.2f}) across "
        f"{n_total} timeframes: {bullish} bullish, {bearish} bearish, "
        f"{neutral} neutral → {tier.value}"
    )

    return ConfluenceResult(
        score=score,
        tier=tier,
        bullish_count=bullish,
        bearish_count=bearish,
        neutral_count=neutral,
        confidence_adjustment=confidence_adj,
        summary=summary,
    )
