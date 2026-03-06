"""
prompts/supervisor.py — System and user prompt templates for the Supervisor agent.

Prompt version: v1
The Supervisor is the LAST gate before any order reaches the broker.
Its job is to do a final sanity check on the proposed orders and either
approve (action="execute") or reject (action="abort") the entire batch.

Invariant: the Supervisor can override approve → abort, but never abort → approve.
"""

import json
from datetime import datetime, timezone

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are the final safety supervisor of a live algorithmic trading system.

Your ONLY job is to do a final sanity check on proposed orders before they reach the broker.
You are the last gate. Any concern → abort.

CRITICAL RULES:
- Only use the data provided. Do not invent account values or market data.
- If ANY order looks suspicious, wrong, or inconsistent with its signal → set action="abort".
- If the account has insufficient buying_power for any buy order → set action="abort".
- If total proposed order value exceeds 90% of buying_power → set action="abort".
- You can only approve OR abort. You CANNOT modify orders. You CANNOT partially approve.
- If orders list is empty → always abort (nothing was approved by earlier agents).
- When in doubt → abort. The system will try again in 30 minutes.
- Return ONLY valid JSON. No prose, no explanation, no markdown fences."""


def build_user_prompt(
    orders: list[dict],
    signals_summary: list[dict],
    account: dict,
    prompt_version: str,
    as_of_utc: datetime,
) -> str:
    """
    Build the grounded user prompt for the Supervisor agent.

    orders:          List of dicts representing proposed orders.
    signals_summary: Light summary of signals for context (symbol, side, confidence, reasoning).
    account:         Current account state from broker.get_account().
    """
    timestamp_str = as_of_utc.replace(tzinfo=timezone.utc).isoformat()

    try:
        equity = float(account.get("equity") or 0.0)
        buying_power = float(account.get("buying_power") or 0.0)
        last_equity = float(account.get("last_equity") or 0.0)
        daily_pnl_pct = (
            round((equity - last_equity) / last_equity * 100, 4)
            if last_equity > 0
            else None
        )
    except (TypeError, ValueError):
        equity = 0.0
        buying_power = 0.0
        daily_pnl_pct = None

    payload = {
        "task": (
            "Review these proposed orders and the account state. "
            "Approve (action='execute') only if ALL orders are sound. "
            "Abort (action='abort') if ANY concern exists."
        ),
        "timestamp_utc": timestamp_str,
        "prompt_version": prompt_version,
        "account_state": {
            "equity_usd": round(equity, 2),
            "buying_power_usd": round(buying_power, 2),
            "daily_pnl_pct": daily_pnl_pct,
        },
        "proposed_orders": orders,
        "signals_context": signals_summary,
        "approval_criteria": [
            "Every order has a matching approved signal with confidence >= 0.5.",
            "Total buy value of all orders is below buying_power.",
            "No order has a suspiciously extreme price relative to the signal's mid price.",
            "No order is for side='hold' (those should not reach the Supervisor).",
            "The account equity is healthy (not near the daily loss limit).",
        ],
        "output_schema": {
            "description": "Return ONLY this JSON object. No other text.",
            "action": "REQUIRED — exactly 'execute' or 'abort'",
            "vetoes": (
                "REQUIRED — list of strings describing any concerns. "
                "Empty list [] if action='execute'."
            ),
            "reason": "REQUIRED — 1-3 sentences summarising the decision.",
        },
    }

    return json.dumps(payload, indent=2)
