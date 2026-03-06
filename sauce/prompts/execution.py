"""
prompts/execution.py — System and user prompt templates for the Execution agent.

Prompt version: v1
The execution prompt asks Claude to determine the optimal order parameters
(order_type, time_in_force, limit_price, stop_price) for an approved signal.

Rules per Section 5 of trading instructions:
- Every prompt includes anti-hallucination instructions.
- Claude is given all constraints explicitly.
- Output is strictly validated against the Order schema.
"""

import json
from datetime import datetime, timezone

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are an order execution specialist inside a live algorithmic trading system.
Your job is to determine optimal order parameters for a pre-approved trading signal.

CRITICAL RULES:
- Only use the data provided. Do not invent prices or market conditions.
- Always use limit or stop_limit orders. Never recommend market orders.
- Default time_in_force is 'day' (expires at market close). Use 'gtc' only if \
the signal reasoning clearly justifies it.
- For a BUY limit order: limit_price must be at or slightly above the current ask \
(to improve fill probability). Do not set limit_price more than 0.1% above ask.
- For a SELL limit order: limit_price must be at or slightly below the current bid. \
Do not set limit_price more than 0.1% below bid.
- stop_price is only required for stop_limit orders. Set to null otherwise.
- Your output will be parsed by a strict JSON validator. Deviate from the schema and \
the order will be blocked.
- Return ONLY valid JSON. No prose, no explanation, no markdown fences."""


def build_user_prompt(
    symbol: str,
    side: str,
    qty: float,
    bid: float,
    ask: float,
    mid: float,
    atr_14: float | None,
    signal_reasoning: str,
    prompt_version: str,
    as_of_utc: datetime,
) -> str:
    """
    Build the grounded user prompt for the Execution agent.
    """
    timestamp_str = as_of_utc.replace(tzinfo=timezone.utc).isoformat()

    # Calculate reference candidate prices to give Claude a starting point
    if side == "buy":
        candidate_limit = round(ask * 1.0005, 4)  # 0.05% above ask
    else:
        candidate_limit = round(bid * 0.9995, 4)  # 0.05% below bid

    payload = {
        "task": "Determine optimal order parameters for this pre-approved trading signal.",
        "timestamp_utc": timestamp_str,
        "prompt_version": prompt_version,
        "symbol": symbol,
        "side": side,
        "approved_qty": round(qty, 6),
        "current_quote": {
            "bid": round(bid, 4),
            "ask": round(ask, 4),
            "mid": round(mid, 4),
        },
        "atr_14": round(atr_14, 4) if atr_14 is not None else None,
        "signal_reasoning": signal_reasoning,
        "constraints": {
            "allowed_order_types": ["limit", "stop_limit"],
            "allowed_time_in_force": ["day", "gtc", "ioc", "fok"],
            "default_order_type": "limit",
            "default_tif": "day",
            "buy_limit_price_max": round(ask * 1.001, 4),
            "sell_limit_price_min": round(bid * 0.999, 4),
            "candidate_limit_price": candidate_limit,
            "note": (
                "Use candidate_limit_price unless there is a clear reason for stop_limit. "
                "stop_limit is only appropriate when the signal reasoning mentions "
                "a breakout or momentum entry."
            ),
        },
        "output_schema": {
            "description": "Return ONLY this JSON object. No other text.",
            "order_type": "REQUIRED — 'limit' or 'stop_limit'",
            "time_in_force": "REQUIRED — 'day', 'gtc', 'ioc', or 'fok'",
            "limit_price": "REQUIRED — float, the limit price for the order",
            "stop_price": (
                "REQUIRED for stop_limit orders, null for limit orders"
            ),
        },
    }

    return json.dumps(payload, indent=2)
