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

PROMPT_VERSION = "v2"

# ── v1 prompt (preserved per Rule 5.5) ────────────────────────────────────────
_SYSTEM_PROMPT_V1 = """You are a quantitative trading analyst operating inside a live algorithmic \
trading system. Financial accuracy is critical.

CRITICAL RULES — FOLLOW EXACTLY:
- Only use the data provided in this prompt. Do not invent, estimate, or extrapolate values \
not present in the input.
- If the data is insufficient to make a decision, return side="hold" with confidence=0.0.
- Never return a buy or sell signal unless the evidence clearly supports it based on the \
provided indicators.
- Do not fabricate indicator values. If an indicator is null in the input, treat it as \
unavailable and do not use it in your reasoning.
- confidence is a strict float between 0.0 and 1.0. Values below 0.40 will be automatically \
treated as hold by the system — Claude should know this.
- Prefer hold when uncertain rather than inflating confidence.
- Your output will be parsed by a strict JSON schema validator. Deviating from the schema \
will block the trade.
- Return ONLY valid JSON. No prose, no explanation, no markdown fences, no extra keys.

You are operating on a 30-minute trading cadence. Signals are for the NEXT 30 minutes only."""

# ── v2 prompt — auditor role ──────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an auditor inside a live algorithmic trading system. \
You do NOT generate trade ideas. The rule engine scores setups and presents them to you. \
Your job is to review the evidence and either approve or reject the thesis.

CRITICAL RULES — FOLLOW EXACTLY:
- Only use the data provided in this prompt. Do not invent, estimate, or extrapolate values \
not present in the input.
- If the data is insufficient to make a decision, return side="hold" with confidence=0.0.
- Do not fabricate indicator values. If an indicator is null in the input, treat it as \
unavailable and do not use it in your reasoning.
- confidence is a strict float between 0.0 and 1.0. Values below 0.40 will be automatically \
treated as hold by the system.
- Your output will be parsed by a strict JSON schema validator. Deviating from the schema \
will block the trade.
- Return ONLY valid JSON. No prose, no explanation, no markdown fences, no extra keys.

YOUR ROLE:
You are auditing a pre-scored thesis. The rules have been evaluated. \
Find contradictions. Find reasons this specific thesis is wrong TODAY. \
Use the session and strategic memory — that data exists for this exact purpose. \
If the thesis withstands scrutiny: approve with calibrated confidence. \
If you find a genuine contradiction: reject with specific reason. \
Do not approve because the score is high. Do not reject because you are uncertain. \
Approve or reject based on whether the evidence coheres.

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
    macd_line: float | None = None,
    macd_signal: float | None = None,
    macd_histogram: float | None = None,
    bb_upper: float | None = None,
    bb_middle: float | None = None,
    bb_lower: float | None = None,
    stoch_k: float | None = None,
    stoch_d: float | None = None,
    vwap: float | None = None,
    daily_trend_context: dict | None = None,
    signal_history: list[dict] | None = None,
    prompt_version: str = "",
    as_of_utc: datetime | None = None,
    *,
    is_crypto: bool = False,
    session_context_text: str = "",
    strategic_context_text: str = "",
    setup_results: list[dict] | None = None,
    positions: list[dict] | None = None,
    multi_timeframe_context: dict | None = None,
    confluence_result: dict | None = None,
    similar_trades_text: str = "",
) -> str:
    """
    Build the grounded user prompt for the Research agent.

    All indicator values passed explicitly — Claude must never be asked to
    look up or derive data not present in the input.
    """
    if as_of_utc is None:
        as_of_utc = datetime.now(timezone.utc)
    timestamp_str = as_of_utc.replace(tzinfo=timezone.utc).isoformat()

    indicators = {
        "sma_20": round(sma_20, 4) if sma_20 is not None else None,
        "sma_50": round(sma_50, 4) if sma_50 is not None else None,
        "rsi_14": round(rsi_14, 4) if rsi_14 is not None else None,
        "atr_14": round(atr_14, 4) if atr_14 is not None else None,
        "volume_ratio": round(volume_ratio, 4) if volume_ratio is not None else None,
        "macd_line": round(macd_line, 4) if macd_line is not None else None,
        "macd_signal": round(macd_signal, 4) if macd_signal is not None else None,
        "macd_histogram": round(macd_histogram, 4) if macd_histogram is not None else None,
        "bb_upper": round(bb_upper, 4) if bb_upper is not None else None,
        "bb_middle": round(bb_middle, 4) if bb_middle is not None else None,
        "bb_lower": round(bb_lower, 4) if bb_lower is not None else None,
        "stoch_k": round(stoch_k, 4) if stoch_k is not None else None,
        "stoch_d": round(stoch_d, 4) if stoch_d is not None else None,
        "vwap": round(vwap, 4) if vwap is not None else None,
    }

    payload = {
        "task": (
            "You are auditing one or more pre-scored setup theses presented in 'setup_results'. "
            "Review the rule engine's evidence for each setup and decide which (if any) to approve. "
            "If multiple setups passed, compare them — the strongest thesis should drive your decision. "
            "If the indicators cohere with the thesis, return the appropriate side "
            "with calibrated confidence. If you find a genuine contradiction in the data, "
            "return side='hold'. Do not generate trade ideas — only audit the ones presented."
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
            "macd": (
                "MACD line crossing above signal line = bullish momentum; "
                "crossing below = bearish momentum. "
                "Histogram > 0 and growing = strengthening bullish momentum; "
                "histogram < 0 and shrinking = weakening bearish momentum."
            ),
            "bollinger_bands": (
                "Price near bb_upper = potentially overbought / strong uptrend; "
                "price near bb_lower = potentially oversold / strong downtrend; "
                "price at bb_middle = neutral. "
                "Bandwidth squeeze (bb_upper close to bb_lower) = low volatility, "
                "potential breakout incoming."
            ),
            "stochastic": (
                "stoch_k > 80 = overbought; stoch_k < 20 = oversold. "
                "stoch_k crossing above stoch_d = bullish signal; "
                "stoch_k crossing below stoch_d = bearish signal."
            ),
            "vwap": (
                "Price above VWAP = bullish intraday bias (buyers in control); "
                "price below VWAP = bearish intraday bias (sellers in control). "
                "VWAP acts as dynamic intraday support/resistance."
            ),
        },
        "asset_type": "crypto" if is_crypto else "equity",
        "setup_results": setup_results,
        "current_positions": (
            {
                "description": (
                    "Open positions the system currently holds. Consider existing "
                    "exposure when auditing: avoid recommending a buy if already long "
                    "the same symbol, and factor unrealized P&L into conviction — "
                    "a losing position may warrant caution while a winning one "
                    "suggests the thesis is playing out."
                ),
                "positions": positions,
            }
            if positions
            else None
        ),
        "confidence_calibration": (
            "If the strongest setup thesis in 'setup_results' is sound and the indicators "
            "cohere with it, approve with 0.55–0.80 confidence. A score near 100 with strong "
            "soft bonuses warrants higher confidence (up to 0.80). If multiple setups passed, "
            "use the comparison to increase or decrease conviction. If you find a genuine "
            "contradiction (e.g., setup claims oversold but RSI is 75), reject with "
            "side='hold'. Do not hold because you are uncertain about your role — you are "
            "an auditor of pre-scored evidence. Values below 0.40 are treated as hold "
            "by the system. Hold is a valid and expected outcome — do not inflate "
            "confidence to cross the threshold."
        ),
        "daily_trend_context": daily_trend_context,
        "signal_history": (
            {
                "description": (
                    "Recent signals generated for this symbol over the past 7 days. "
                    "Use this to identify patterns: repeated holds (maybe you are being "
                    "too cautious), repeated buys that reversed (maybe confidence was too "
                    "high), or missed opportunities. Adjust your current analysis "
                    "accordingly — learn from recent outcomes."
                ),
                "recent_signals": signal_history,
            }
            if signal_history
            else None
        ),
        "daily_trend_interpretation": (
            {
                "usage": (
                    "The daily_trend_context provides the bigger picture. "
                    "Use it to confirm or contradict 30-minute signals."
                ),
                "bullish_daily_plus_30m_oversold": (
                    "Daily trend bullish + 30min RSI oversold = buy-the-dip opportunity. "
                    "High conviction setup — confidence 0.60-0.75."
                ),
                "bearish_daily_plus_30m_overbought": (
                    "Daily trend bearish + 30min RSI overbought = potential sell/short. "
                    "Confidence 0.55-0.70."
                ),
                "conflicting_timeframes": (
                    "If daily and 30min disagree, reduce confidence by 0.10-0.15 "
                    "but do NOT automatically hold. The 30min signal still has value."
                ),
            }
            if daily_trend_context is not None
            else None
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
            "bear_case": (
                "REQUIRED — 1 to 2 sentences. The strongest argument AGAINST this trade. "
                "Play devil's advocate: what could go wrong? What contradicts the thesis? "
                "If side='hold', state why the thesis failed. "
                "This field combats confirmation bias in the auditor role."
            ),
        },
        "multi_timeframe_analysis": (
            {
                "description": (
                    "Indicators computed across multiple timeframes (5m to 1d). "
                    "Use this to validate the 30-min thesis: if higher timeframes "
                    "agree, the signal is stronger. If they conflict, reduce confidence."
                ),
                **multi_timeframe_context,
            }
            if multi_timeframe_context
            else None
        ),
        "confluence_scoring": (
            {
                "description": (
                    "Weighted confluence score across all available timeframes. "
                    "The signal tier (S1=strongest, S4=conflicting) reflects how well "
                    "different timeframes agree.  Use this as a meta-signal: "
                    "S1/S2 agreement should increase your conviction; "
                    "S4 conflict should decrease it or push you toward hold."
                ),
                **confluence_result,
            }
            if confluence_result
            else None
        ),
    }

    if session_context_text:
        payload["session_memory"] = session_context_text
    if strategic_context_text:
        payload["strategic_memory"] = strategic_context_text
    if similar_trades_text:
        payload["past_trade_memory"] = {
            "description": (
                "The most similar past trades from strategic memory, ranked by "
                "relevance (same symbol, regime, setup type). Use these outcomes "
                "to calibrate your confidence: if similar setups consistently lost, "
                "lower confidence; if they consistently won, the thesis has precedent."
            ),
            "similar_trades": similar_trades_text,
        }

    return json.dumps(payload, indent=2)
