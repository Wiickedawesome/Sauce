"""
prompts/research.py — System and user prompt templates for the Research agent.

Prompt version: v1
Rules:
- Prompt strings are module-level constants (Rule 4.4).
- The user prompt is built at call time via build_user_prompt().
- Every prompt must include anti-hallucination instructions (Rule 5.3).
- Every prompt must instruct Claude to return only valid JSON (Rule 5.2).
- If prompt changes: increment PROMPT_VERSION and keep old version as suffix (Rule 5.5).
"""

import json
from datetime import datetime, timezone

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are a quantitative trading analyst operating inside a live algorithmic \
trading system. Financial accuracy is critical.

CRITICAL RULES — FOLLOW EXACTLY:
- Only use the data provided in this prompt. Do not invent, estimate, or extrapolate values \
not present in the input.
- If the data is insufficient to make a decision, return side="hold" with confidence=0.0.
- Never return a buy or sell signal unless the evidence clearly supports it based on the \
provided indicators.
- Do not fabricate indicator values. If an indicator is null in the input, treat it as \
unavailable and do not use it in your reasoning.
- confidence is a strict float between 0.0 and 1.0. Values below 0.5 will be automatically \
treated as hold by the system — Claude should know this.
- Prefer hold when uncertain rather than inflating confidence.
- Your output will be parsed by a strict JSON schema validator. Deviating from the schema \
will block the trade.
- Return ONLY valid JSON. No prose, no explanation, no markdown fences, no extra keys.

You are operating on a 30-minute trading cadence. Signals are for the NEXT 30 minutes only."""


def build_user_prompt(
    symbol: str,
    mid: float,
    bid: float,
    ask: float,
    sma_20: float | None,
    sma_50: float | None,
    rsi_14: float | None,
    atr_14: float | None,
    volume_ratio: float | None,
    prompt_version: str,
    as_of_utc: datetime,
    *,
    is_crypto: bool = False,
) -> str:
    """
    Build the grounded user prompt for the Research agent.

    All indicator values passed explicitly — Claude must never be asked to
    look up or derive data not present in the input.
    """
    timestamp_str = as_of_utc.replace(tzinfo=timezone.utc).isoformat()

    indicators = {
        "sma_20": round(sma_20, 4) if sma_20 is not None else None,
        "sma_50": round(sma_50, 4) if sma_50 is not None else None,
        "rsi_14": round(rsi_14, 4) if rsi_14 is not None else None,
        "atr_14": round(atr_14, 4) if atr_14 is not None else None,
        "volume_ratio": round(volume_ratio, 4) if volume_ratio is not None else None,
    }

    payload = {
        "task": (
            "Analyze this symbol and generate a trading signal for the next 30 minutes. "
            "Base your decision ONLY on the indicators provided."
        ),
        "timestamp_utc": timestamp_str,
        "prompt_version": prompt_version,
        "symbol": symbol,
        "price_reference": {
            "bid": round(bid, 4),
            "ask": round(ask, 4),
            "mid": round(mid, 4),
        },
        "indicators": indicators,
        "indicator_interpretation_guide": {
            "sma_20_vs_sma_50": (
                "price above both SMAs = bullish trend; "
                "price below both = bearish trend"
            ),
            "rsi_14": (
                "below 30 = oversold (potential bullish reversal); "
                "above 70 = overbought (potential bearish reversal); "
                "30-70 = neutral"
            ),
            "volume_ratio": (
                "above 1.5 = elevated volume (confirms moves); "
                "below 0.5 = low volume (reduces conviction but does NOT by itself mean hold). "
                "Low volume should lower confidence slightly, not force a hold when "
                "trend and momentum indicators are clearly aligned."
            ) if not is_crypto else (
                "above 1.5 = elevated volume (confirms moves); "
                "below 0.5 = low relative volume. For crypto, low volume_ratio is common "
                "outside peak hours and on weekends — it should NOT be treated as a reason "
                "to hold. Focus on trend alignment (SMAs) and momentum (RSI) instead."
            ),
            "price_vs_sma20": "price above SMA_20 = short-term bullish bias",
            "atr_14_interpretation": (
                "ATR measures recent volatility in price units. "
                "High ATR = high risk/high reward. "
                "If ATR is null, volatility is unknown."
            ),
        },
        "asset_type": "crypto" if is_crypto else "equity",
        "confidence_calibration": (
            "Confidence below 0.5 will be treated as hold by the system. "
            "Use hold (confidence=0.0) if indicators genuinely conflict or are insufficient. "
            "When trend (SMAs) and momentum (RSI) align in the same direction, "
            "a confidence of 0.5-0.7 is appropriate even with low volume. "
            "Do not inflate confidence, but do not default to hold when evidence is clear."
        ),
        "required_output_schema": {
            "description": "Return ONLY this JSON object. No other text.",
            "side": "REQUIRED — exactly 'buy', 'sell', or 'hold'",
            "confidence": "REQUIRED — float 0.0 to 1.0",
            "reasoning": (
                "REQUIRED — 1 to 3 sentences. "
                "Base ONLY on the indicators provided above. "
                "Do not reference any external data."
            ),
        },
    }

    return json.dumps(payload, indent=2)
