"""
reflection.py — Post-trade reflection loop.

After every position exit, a single LLM call reflects on what happened:
what the market conditions were when the trade was entered, what the
outcome was, and what lesson can be drawn for future trades.

The generated lesson is stored in the BM25 trade memory for recall
during future analyst committee deliberations.

Cost: 1 LLM call per closed trade (typically 0-2 per cycle).
"""

from __future__ import annotations

import json
import logging

from sauce.adapters.llm import LLMError, call_claude
from sauce.core.config import get_settings
from sauce.memory import MemoryEntry

logger = logging.getLogger(__name__)

# ── Prompt ────────────────────────────────────────────────────────────────────

REFLECTION_SYSTEM = """\
You are a trading analyst conducting a post-trade review.

Given the entry situation and outcome, write a concise, actionable lesson
that would help a portfolio manager make better decisions on similar
future trades.

Focus on:
- What signals or conditions predicted the outcome
- Whether the entry timing was good or bad
- What could have been done differently
- Any pattern worth remembering for similar setups

Respond with ONLY a JSON object:
{
  "lesson": "1-3 sentences with the key actionable insight"
}

No other text. No markdown fences."""

REFLECTION_USER = """\
Trade completed for {symbol}:

Entry situation: {situation}

Outcome: {outcome}

{memory_section}"""

MEMORY_HEADER = "Lessons from similar past trades:"
NO_MEMORIES = "No similar past trades for comparison."


# ── Public Interface ──────────────────────────────────────────────────────────


async def reflect_on_trade(
    symbol: str,
    situation: str,
    outcome: str,
    memories: list[MemoryEntry] | None = None,
    loop_id: str = "reflection",
) -> str | None:
    """Generate a lesson from a completed trade via Claude.

    Returns the lesson string, or None if the LLM call fails.
    Failures are logged but never raised — reflection is non-critical.
    """
    settings = get_settings()

    # Build memory context
    if memories:
        memory_lines = [MEMORY_HEADER]
        for i, mem in enumerate(memories, 1):
            memory_lines.append(f"  {i}. {mem.outcome} — Lesson: {mem.lesson}")
        memory_section = "\n".join(memory_lines)
    else:
        memory_section = NO_MEMORIES

    user_prompt = REFLECTION_USER.format(
        symbol=symbol,
        situation=situation,
        outcome=outcome,
        memory_section=memory_section,
    )

    try:
        raw = await call_claude(
            system=REFLECTION_SYSTEM,
            user=user_prompt,
            loop_id=loop_id,
            temperature=settings.reflection_temperature,
        )
        parsed = json.loads(raw)
        lesson = str(parsed.get("lesson", ""))
        if not lesson:
            logger.warning("Reflection returned empty lesson for %s", symbol)
            return None

        logger.info("REFLECTION %s: %s", symbol, lesson)
        return lesson

    except (LLMError, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Reflection failed for %s (%s), skipping memory storage", symbol, exc)
        return None
