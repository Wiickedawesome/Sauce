"""
prompts/options_research.py — System and user prompt templates for the Options Research agent.

Prompt version: v1
Rules:
- Prompt strings are module-level constants.
- The user prompt is built at call time via build_user_prompt().
- Anti-hallucination instructions are embedded.
- Claude returns only valid JSON.
"""

import json
from datetime import datetime, timezone


PROMPT_VERSION = "options-v2"


SYSTEM_PROMPT = """\
You are an options contract selector inside a live algorithmic trading system. \
The equity signal pipeline has already identified a directional bias for the \
underlying. Your job is to SELECT the best options contract to express that \
bias, or recommend NO TRADE if no suitable contract exists.

CRITICAL RULES — FOLLOW EXACTLY:
- Only use the data provided in this prompt. Do not invent or extrapolate.
- If no contract meets the criteria, return action="no_trade" with confidence=0.0.
- NEVER recommend a contract with DTE < 14 or DTE > 35.
- NEVER recommend a contract with absolute delta outside [0.30, 0.60].
- NEVER recommend a contract with bid/ask spread > 5% of mid price.
- NEVER recommend market orders on options. All orders are LIMIT only.
- confidence is a strict float between 0.0 and 1.0.
- Return ONLY valid JSON. No prose, no explanation, no markdown fences.

STRATEGY: "Momentum Snipe" — single-leg directional options.
- Profit target at +35%: sell half (if qty >= 2) and activate trailing stop.
- Stretch target at +60%: close all remaining contracts.
- Trailing stop: activated at +20% gain, trails at 12% below high-water mark.
- Hard stop: close at -25% loss.
- Time stop: close after 5 days if gain < 10%.
- DTE stop: close when <= 5 DTE remaining.
- Prefer affordable contracts under max_position_cost to allow multi-contract entries.
- Your job is ONLY contract selection, not exit management.

OUTPUT SCHEMA:
{
    "action": "buy_call" | "buy_put" | "no_trade",
    "contract_symbol": "<OCC symbol or null>",
    "confidence": <0.0-1.0>,
    "reasoning": "<1-2 sentence justification>",
    "bear_case": "<1-2 sentence risk/bear case>",
    "limit_price": <suggested limit price or null>,
    "qty": <number of contracts or 0>
}
"""


def build_user_prompt(
    *,
    symbol: str,
    direction: str,
    bias_confidence: float,
    iv_rank: float | None,
    contracts: list[dict],
    nav: float,
    max_position_cost: float,
    regime: str | None = None,
) -> str:
    """
    Build the user prompt for options contract selection.

    Parameters
    ----------
    symbol:             Underlying ticker.
    direction:          "bullish" or "bearish" from OptionsBias.
    bias_confidence:    Confidence from the equity bias (0.0-1.0).
    iv_rank:            IV rank for the underlying (0.0-1.0 or None).
    contracts:          List of candidate contract dicts with greeks/quotes.
    nav:                Current net asset value.
    max_position_cost:  Maximum $ amount for this position.
    regime:             Current market regime string.
    """
    data = {
        "underlying": symbol,
        "directional_bias": direction,
        "bias_confidence": round(bias_confidence, 3),
        "iv_rank": round(iv_rank, 3) if iv_rank is not None else None,
        "market_regime": regime or "unknown",
        "account": {
            "nav": round(nav, 2),
            "max_position_cost": round(max_position_cost, 2),
        },
        "candidate_contracts": contracts[:15],  # cap to avoid token blow-up
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return (
        "Below is the options analysis context. Select the best contract "
        "or recommend no_trade.\n\n"
        + json.dumps(data, indent=2, default=str)
    )
