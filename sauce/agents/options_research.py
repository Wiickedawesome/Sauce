"""
agents/options_research.py — Options Research agent: selects contracts.

Sequence:
  1. Check OPTIONS_ENABLED gate.
  2. Use existing equity confluence to derive directional bias.
  3. Fetch option chain for the underlying.
  4. Enrich candidates with snapshots (greeks, quotes).
  5. Run entry safety checks.
  6. Call Claude to select the best contract.
  7. Parse JSON response into OptionsSignal.
  8. On any error → return None (no options trade).

This agent runs AFTER the equity research agent and reuses its signals.
It NEVER generates its own directional thesis.
"""

import json
import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from sauce.adapters import llm
from sauce.adapters.db import log_event
from sauce.adapters.options_data import (
    OptionsDataError,
    filter_contracts,
    get_contract_snapshot,
    get_option_chain,
)
from sauce.core.config import get_settings
from sauce.core.options_config import get_options_settings
from sauce.core.options_schemas import (
    OptionsBias,
    OptionsContract,
    OptionsSignal,
)
from sauce.core.options_safety import check_options_enabled, run_entry_safety
from sauce.core.schemas import AuditEvent
from sauce.prompts import options_research as prompts

logger = logging.getLogger(__name__)


async def run(
    symbol: str,
    bias: OptionsBias,
    nav: float,
    current_options_value: float,
    loop_id: str,
) -> OptionsSignal | None:
    """
    Select an options contract for a given directional bias.

    Returns an OptionsSignal if a suitable contract is found,
    or None if no trade should be taken.

    Parameters
    ----------
    symbol:                Underlying ticker.
    bias:                  Directional bias from equity confluence.
    nav:                   Current net asset value.
    current_options_value: Total market value of all current options positions.
    loop_id:               Current loop UUID for audit correlation.
    """
    settings = get_settings()
    cfg = get_options_settings()
    db_path = str(settings.db_path)

    # ── Gate check ────────────────────────────────────────────────────────
    passed, reason = check_options_enabled(loop_id)
    if not passed:
        return None

    # ── Bias check ────────────────────────────────────────────────────────
    if bias.direction == "neutral" or bias.confidence < settings.min_confidence:
        logger.info(
            "options_research[%s]: skipping — bias=%s conf=%.3f",
            symbol, bias.direction, bias.confidence,
        )
        return None

    # ── Fetch option chain ────────────────────────────────────────────────
    option_type = "call" if bias.direction == "bullish" else "put"
    try:
        chain = get_option_chain(
            underlying=symbol,
            option_type=option_type,
        )
    except OptionsDataError as exc:
        logger.warning("options_research[%s]: chain fetch failed: %s", symbol, exc)
        return None

    if not chain:
        logger.info("options_research[%s]: no contracts in chain", symbol)
        return None

    # ── Filter by DTE / delta / liquidity ─────────────────────────────────
    candidates = filter_contracts(chain)
    if not candidates:
        logger.info("options_research[%s]: no contracts pass filters", symbol)
        return None

    # ── Enrich top candidates with snapshots ──────────────────────────────
    enriched: list[OptionsContract] = []
    for c in candidates[:20]:  # cap API calls
        try:
            snap = get_contract_snapshot(c.contract_symbol)
            enriched.append(snap)
        except OptionsDataError:
            continue

    if not enriched:
        logger.info("options_research[%s]: no snapshots available", symbol)
        return None

    # ── Dynamic affordability filter ──────────────────────────────────────
    affordable = [
        c for c in enriched
        if (c.mid or 0) * 100 <= cfg.option_max_contract_cost
    ]
    if not affordable:
        logger.info(
            "options_research[%s]: no contracts under $%.0f cost limit",
            symbol, cfg.option_max_contract_cost,
        )
        return None
    enriched = affordable

    # ── Position sizing ───────────────────────────────────────────────────
    max_position_cost = nav * cfg.option_max_position_pct

    # ── Build contract dicts for prompt ───────────────────────────────────
    contract_dicts = []
    for c in enriched:
        contract_dicts.append({
            "contract_symbol": c.contract_symbol,
            "strike": c.strike,
            "expiration": c.expiration,
            "delta": c.delta,
            "gamma": c.gamma,
            "theta": c.theta,
            "iv": c.iv,
            "bid": c.bid,
            "ask": c.ask,
            "mid": c.mid,
            "open_interest": c.open_interest,
            "volume": c.volume,
        })

    # ── Call Claude ───────────────────────────────────────────────────────
    user_prompt = prompts.build_user_prompt(
        symbol=symbol,
        direction=bias.direction,
        bias_confidence=bias.confidence,
        iv_rank=bias.iv_rank,
        contracts=contract_dicts,
        nav=nav,
        max_position_cost=max_position_cost,
    )

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="llm_call",
        symbol=symbol,
        payload={"agent": "options_research", "candidates": len(enriched)},
        prompt_version=prompts.PROMPT_VERSION,
    ), db_path=db_path)

    try:
        raw_response = await llm.call_claude(
            system=prompts.SYSTEM_PROMPT,
            user=user_prompt,
            loop_id=loop_id,
            temperature=settings.research_temperature,
        )
    except llm.LLMError as exc:
        logger.warning("options_research[%s]: LLM call failed: %s", symbol, exc)
        return None

    # ── Parse response ────────────────────────────────────────────────────
    try:
        parsed = json.loads(llm._strip_fences(raw_response))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "options_research[%s]: JSON parse failed: %s", symbol, exc,
        )
        return None

    action = parsed.get("action", "no_trade")
    if action == "no_trade":
        logger.info("options_research[%s]: Claude recommends no_trade", symbol)
        return None

    contract_symbol = parsed.get("contract_symbol")
    if not contract_symbol:
        return None

    # Find the enriched contract
    selected = next(
        (c for c in enriched if c.contract_symbol == contract_symbol), None,
    )
    if selected is None:
        logger.warning(
            "options_research[%s]: Claude selected unknown contract %s",
            symbol, contract_symbol,
        )
        return None

    confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))

    # ── Run entry safety on the selected contract ─────────────────────────
    from sauce.core.options_safety import run_entry_safety
    from sauce.adapters.options_data import get_option_quote

    try:
        quote = get_option_quote(selected.contract_symbol)
    except OptionsDataError:
        return None

    cost = (selected.mid or 0) * 100 * max(1, int(parsed.get("qty", 1)))
    passed, reason = run_entry_safety(
        contract=selected,
        quote=quote,
        iv_rank=bias.iv_rank,
        nav=nav,
        current_options_value=current_options_value,
        new_position_cost=cost,
        loop_id=loop_id,
    )
    if not passed:
        logger.info(
            "options_research[%s]: safety rejected — %s", symbol, reason,
        )
        return None

    direction_str = "long_call" if action == "buy_call" else "long_put"

    signal = OptionsSignal(
        symbol=symbol,
        contract=selected,
        direction=direction_str,
        max_position_cost=max_position_cost,
        confidence=confidence,
        reasoning=parsed.get("reasoning", ""),
        bear_case=parsed.get("bear_case", ""),
        bias=bias,
        as_of=datetime.now(timezone.utc),
        prompt_version=prompts.PROMPT_VERSION,
    )

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="options_signal",
        symbol=symbol,
        payload={
            "contract": selected.contract_symbol,
            "direction": direction_str,
            "confidence": confidence,
            "action": action,
        },
        prompt_version=prompts.PROMPT_VERSION,
    ), db_path=db_path)

    return signal

