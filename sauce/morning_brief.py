"""
morning_brief.py — Single daily Claude call for regime classification.

Called once at ~7 AM ET (or first loop of the day). Classifies the market
regime as one of: bullish, neutral, bearish.

The regime affects signal thresholds:
  - bullish  → threshold base − 10 (easier to fire)
  - neutral  → threshold base + 0
  - bearish  → threshold base + 10 (harder to fire)

If Claude fails, defaults to "neutral" (no bias). This is the ONLY
LLM call in the entire system per day.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from sauce.adapters.llm import LLMError, call_claude

logger = logging.getLogger(__name__)

VALID_REGIMES = {"bullish", "neutral", "bearish"}

SYSTEM_PROMPT = """\
You are a market regime classifier. Given recent market data, classify the
current regime as exactly one of: bullish, neutral, bearish.

Respond with ONLY a JSON object:
{"regime": "bullish"|"neutral"|"bearish", "reasoning": "one sentence"}

No other text. No markdown fences."""

USER_TEMPLATE = """\
Date: {date}
BTC 24h change: {btc_change:.2%}
ETH 24h change: {eth_change:.2%}
SPY daily change: {spy_change:.2%}
VIX level: {vix:.1f}
BTC RSI(14): {btc_rsi:.1f}"""


def infer_intraday_regime(
    btc_change: float,
    eth_change: float,
    spy_change: float,
    vix: float,
    btc_rsi: float,
) -> tuple[str, str]:
    """Heuristic regime refresh used between the daily LLM classifications."""
    score = 0
    if btc_change >= 0.02:
        score += 1
    elif btc_change <= -0.02:
        score -= 1

    if eth_change >= 0.02:
        score += 1
    elif eth_change <= -0.02:
        score -= 1

    if spy_change >= 0.01:
        score += 1
    elif spy_change <= -0.01:
        score -= 1

    if vix <= 16:
        score += 1
    elif vix >= 25:
        score -= 1

    if btc_rsi >= 60:
        score += 1
    elif btc_rsi <= 40:
        score -= 1

    if score >= 2:
        regime = "bullish"
    elif score <= -2:
        regime = "bearish"
    else:
        regime = "neutral"

    return regime, f"heuristic_score={score} btc={btc_change:.2%} eth={eth_change:.2%} spy={spy_change:.2%} vix={vix:.1f} btc_rsi={btc_rsi:.1f}"


async def get_regime(
    btc_change: float,
    eth_change: float,
    spy_change: float,
    vix: float,
    btc_rsi: float,
    loop_id: str = "morning_brief",
) -> str:
    """Classify market regime via a single Claude call.

    Returns one of: "bullish", "neutral", "bearish".
    Falls back to "neutral" on any error.
    """
    user_msg = USER_TEMPLATE.format(
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        btc_change=btc_change,
        eth_change=eth_change,
        spy_change=spy_change,
        vix=vix,
        btc_rsi=btc_rsi,
    )

    try:
        raw = await call_claude(
            system=SYSTEM_PROMPT,
            user=user_msg,
            loop_id=loop_id,
            temperature=0.1,
        )
        parsed = json.loads(raw)
        regime: str = parsed.get("regime", "neutral").lower().strip()

        if regime not in VALID_REGIMES:
            logger.warning("Invalid regime from Claude: %r, defaulting to neutral", regime)
            return "neutral"

        logger.info("Morning brief regime: %s — %s", regime, parsed.get("reasoning", ""))
        return regime

    except (LLMError, json.JSONDecodeError, KeyError) as exc:
        fallback_regime, fallback_reason = infer_intraday_regime(
            btc_change,
            eth_change,
            spy_change,
            vix,
            btc_rsi,
        )
        logger.warning(
            "Morning brief failed (%s), falling back to %s (%s)",
            exc,
            fallback_regime,
            fallback_reason,
        )
        return fallback_regime
