"""
analyst.py — Pre-trade analyst committee (debate-lite).

Before every entry order placement, the committee runs two focused LLM calls:

  1. **Dual Analysis** — A single prompt asks Claude to produce BOTH the bull
     case and the bear case for the proposed trade. This replaces separate
     bull/bear researcher agents with one efficient call.

  2. **PM Verdict** — A portfolio manager prompt receives the bull/bear analysis
     plus any relevant past trade memories (BM25 recall) and renders a final
     approve/reject with confidence score.

If both calls succeed and the PM approves, the trade proceeds to the risk gate.
If the committee rejects or either call fails, the trade is skipped gracefully
(the deterministic signal still logged — we just don't act on it).

Cost: 2 LLM calls per signal that fires. Most cycles fire 0-1 signals.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sauce.adapters.llm import LLMError, call_llm
from sauce.core.config import get_settings
from sauce.memory import MemoryEntry

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalystVerdict:
    """Output of the analyst committee."""

    approve: bool
    confidence: int  # 0–100
    bull_case: str
    bear_case: str
    reasoning: str
    size_fraction: float = 0.0  # 0.0–1.0 fraction of the planned tranche to deploy

    def __post_init__(self) -> None:
        try:
            normalized_size = float(self.size_fraction)
        except (TypeError, ValueError):
            normalized_size = 0.0
        normalized_size = max(0.0, min(1.0, normalized_size))
        if not self.approve:
            normalized_size = 0.0
        elif normalized_size <= 0.0:
            normalized_size = 1.0
        object.__setattr__(self, "size_fraction", normalized_size)


# ── Prompts ───────────────────────────────────────────────────────────────────

DUAL_ANALYSIS_SYSTEM = """\
You are two financial analysts debating a proposed trade.

First, present the BULL CASE — reasons this trade will succeed.
Then, present the BEAR CASE — reasons this trade will fail or underperform.

Be specific and grounded in the data provided. Consider technical indicators,
market regime, volume, momentum, and any risk factors.

Respond with ONLY a JSON object:
{
  "bull_case": "2-3 sentences arguing FOR the trade",
  "bear_case": "2-3 sentences arguing AGAINST the trade"
}

No other text. No markdown fences."""

DUAL_ANALYSIS_USER = """\
Proposed trade: BUY {symbol}
Strategy: {strategy_name}
Signal score: {score}/{threshold}

Current market data:
- Price: ${current_price:.4f}
- RSI(14): {rsi_14}
- MACD histogram: {macd_hist}
- Bollinger Band position: {bb_pct} (0=lower band, 1=upper band)
- Volume ratio (vs 20d avg): {volume_ratio}
- Market regime: {regime}"""

PM_VERDICT_SYSTEM = """\
You are a portfolio manager making the final decision on a proposed trade.
You have received analysis from two analysts (bull and bear cases) and
may have lessons from similar past trades.

Your job: weigh the evidence and decide whether to APPROVE or REJECT this trade,
and if approved, how large the position should be.

Consider:
1. Strength of bull vs bear arguments
2. Risk/reward given current regime
3. Lessons from past similar trades (if provided)
4. Whether the signal score justifies the entry

Sizing rules:
- Prefer reducing size over rejecting when the setup is mixed but still plausible.
- Similar-trade lessons are context, not hard laws.
- CRITICAL: If only 1-5 past trades are available, the sample is too small for \
statistical conclusions. Weight the current technical setup and bull/bear analysis \
MORE heavily than a handful of past outcomes. A few losses do not prove a pattern \
— they are normal variance.
- Do NOT extrapolate rigid entry criteria (e.g. "only enter if RSI > X") from a \
small sample. Every trade has unique context.
- Use smaller starter sizes for neutral or mixed setups and reserve full size for the strongest alignment only.

Respond with ONLY a JSON object:
{
  "approve": true or false,
    "size_fraction": 0.0 to 1.0,
  "confidence": 0-100,
  "reasoning": "1-2 sentences explaining your decision"
}

Interpret size_fraction as:
- 0.0 = reject / no capital
- 0.25 = tiny experimental starter
- 0.50 = normal starter
- 0.75 = strong starter or add-on
- 1.0 = highest-conviction size only

No other text. No markdown fences."""

PM_VERDICT_USER = """\
Proposed trade: BUY {symbol} (score {score}/{threshold}, regime={regime})

Bull case: {bull_case}
Bear case: {bear_case}

{memory_section}"""

MEMORY_HEADER = "Lessons from similar past trades (treat as anecdotal context, NOT rules):"
NO_MEMORIES = "No similar past trades found — this is a fresh setup. Judge on current data alone."


def _coerce_size_fraction(raw_value: object) -> float:
    try:
        size_fraction = float(str(raw_value))
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, size_fraction))


def _default_size_fraction_from_confidence(confidence: int) -> float:
    if confidence >= 85:
        return 1.0
    if confidence >= 70:
        return 0.75
    if confidence >= 55:
        return 0.50
    return 0.25


# ── Public Interface ──────────────────────────────────────────────────────────


async def analyst_committee(
    symbol: str,
    strategy_name: str,
    score: int,
    threshold: int,
    regime: str,
    current_price: float,
    rsi_14: float | None,
    macd_hist: float | None,
    bb_pct: float | None,
    volume_ratio: float | None,
    memories: list[MemoryEntry] | None = None,
    loop_id: str = "analyst",
) -> AnalystVerdict:
    """Run the two-call analyst committee.

    Returns an AnalystVerdict. On any LLM failure, returns a default
    REJECT verdict (fail-safe: don't open positions on broken analysis).
    """
    settings = get_settings()

    # ── Call 1: Dual Analysis ──
    try:
        # Paper-trading volume data from Alpaca is unreliable (near-zero
        # for crypto).  Flag it so the LLM doesn't reject on that alone.
        if settings.alpaca_paper:
            vol_display = "N/A (paper-trading — volume data unreliable, ignore volume)"
        else:
            vol_display = f"{volume_ratio:.2f}" if volume_ratio is not None else "N/A"

        analysis_user = DUAL_ANALYSIS_USER.format(
            symbol=symbol,
            strategy_name=strategy_name,
            score=score,
            threshold=threshold,
            current_price=current_price,
            rsi_14=f"{rsi_14:.1f}" if rsi_14 is not None else "N/A",
            macd_hist=f"{macd_hist:.4f}" if macd_hist is not None else "N/A",
            bb_pct=f"{bb_pct:.2f}" if bb_pct is not None else "N/A",
            volume_ratio=vol_display,
            regime=regime,
        )

        raw_analysis = await call_llm(
            system=DUAL_ANALYSIS_SYSTEM,
            user=analysis_user,
            loop_id=loop_id,
            provider=settings.dual_analysis_provider,
            temperature=settings.research_temperature,
        )
        analysis = json.loads(raw_analysis)
        bull_case = analysis.get("bull_case", "No bull case provided")
        bear_case = analysis.get("bear_case", "No bear case provided")

    except (LLMError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Analyst dual analysis failed (%s), defaulting to reject", exc)
        return AnalystVerdict(
            approve=False,
            confidence=0,
            bull_case="Analysis unavailable",
            bear_case="Analysis unavailable",
            reasoning=f"Committee skipped due to LLM error: {exc}",
            size_fraction=0.0,
        )

    # ── Call 2: PM Verdict ──
    try:
        # Build memory section
        if memories:
            n = len(memories)
            sample_note = (
                f"(⚠ Only {n} past trade{'s' if n != 1 else ''} — "
                f"too few for statistical weight. Prioritize current setup.)"
                if n <= 5
                else f"({n} past trades recalled.)"
            )
            memory_lines = [f"{MEMORY_HEADER} {sample_note}"]
            for i, mem in enumerate(memories, 1):
                memory_lines.append(f"  {i}. {mem.outcome} — Lesson: {mem.lesson}")
            memory_section = "\n".join(memory_lines)
        else:
            memory_section = NO_MEMORIES

        verdict_user = PM_VERDICT_USER.format(
            symbol=symbol,
            score=score,
            threshold=threshold,
            regime=regime,
            bull_case=bull_case,
            bear_case=bear_case,
            memory_section=memory_section,
        )

        raw_verdict = await call_llm(
            system=PM_VERDICT_SYSTEM,
            user=verdict_user,
            loop_id=loop_id,
            provider=settings.pm_verdict_provider,
            temperature=settings.supervisor_temperature,
        )
        verdict = json.loads(raw_verdict)

        approve = bool(verdict.get("approve", False))
        confidence = int(verdict.get("confidence", 0))
        confidence = max(0, min(100, confidence))
        reasoning = str(verdict.get("reasoning", "No reasoning provided"))
        size_fraction = _coerce_size_fraction(verdict.get("size_fraction", 0.0))

        if approve and size_fraction <= 0.0:
            size_fraction = _default_size_fraction_from_confidence(confidence)

        if not approve:
            size_fraction = 0.0

        # Reject low-confidence approvals (PM uncertain = skip)
        min_confidence_pct = int(settings.min_confidence * 100)
        if approve and confidence < min_confidence_pct:
            logger.info(
                "ANALYST %s: approve overridden to REJECT — confidence %d < %d",
                symbol,
                confidence,
                min_confidence_pct,
            )
            approve = False
            size_fraction = 0.0
            reasoning = f"Low confidence ({confidence}) — {reasoning}"

        logger.info(
            "ANALYST %s: approve=%s confidence=%d size=%.2f — %s",
            symbol,
            approve,
            confidence,
            size_fraction,
            reasoning,
        )

        return AnalystVerdict(
            approve=approve,
            confidence=confidence,
            bull_case=bull_case,
            bear_case=bear_case,
            reasoning=reasoning,
            size_fraction=size_fraction,
        )

    except (LLMError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("PM verdict failed (%s), defaulting to reject", exc)
        return AnalystVerdict(
            approve=False,
            confidence=0,
            bull_case=bull_case,
            bear_case=bear_case,
            reasoning=f"PM verdict skipped due to LLM error: {exc}",
            size_fraction=0.0,
        )
