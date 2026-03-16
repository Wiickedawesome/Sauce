"""
core/options_safety.py — Safety guards specific to the options module.

These are the 4 additional safety layers on top of the existing 8-layer
safety gauntlet. All checks run BEFORE any options trade can be placed.

Layers:
  S1. OPTIONS_ENABLED gate   — master switch must be True
  S2. IV rank ceiling        — reject if IV rank > config threshold
  S3. DTE window             — reject if DTE outside min/max range
  S4. Spread width check     — reject if bid/ask spread > max %
  S5. Delta range check      — reject if abs(delta) outside bounds
  S6. Exposure limit         — reject if total options exposure > max fraction
  S7. Max loss stop          — exit if position has lost > max_loss_pct

Rules:
- All checks return (pass: bool, reason: str).
- Every check is logged as AuditEvent(event_type="options_safety_check").
- Fail-safe default: reject trade on any exception.
"""

import logging
from datetime import date, datetime, timezone

from sauce.core.options_config import get_options_settings
from sauce.core.options_schemas import OptionsContract, OptionsPosition, OptionsQuote
from sauce.core.schemas import AuditEvent

logger = logging.getLogger(__name__)

SafetyResult = tuple[bool, str]


def _log_check(
    loop_id: str,
    check_name: str,
    passed: bool,
    details: dict | None = None,
) -> None:
    """Log an options safety check as an AuditEvent."""
    from sauce.adapters.db import log_event

    payload: dict = {
        "check": check_name,
        "result": passed,
    }
    if details:
        payload.update(details)

    log_event(AuditEvent(
        loop_id=loop_id,
        event_type="options_safety_check",
        payload=payload,
        timestamp=datetime.now(timezone.utc),
    ))


# ── S1: Master Switch ────────────────────────────────────────────────────────

def check_options_enabled(loop_id: str = "unset") -> SafetyResult:
    """Reject if OPTIONS_ENABLED is False."""
    cfg = get_options_settings()
    passed = cfg.options_enabled
    reason = "" if passed else "OPTIONS_ENABLED is False"
    _log_check(loop_id, "options_enabled", passed)
    return passed, reason


# ── S2: IV Rank Ceiling ───────────────────────────────────────────────────────

def check_iv_rank(
    iv_rank: float | None,
    loop_id: str = "unset",
) -> SafetyResult:
    """
    Reject if IV rank exceeds the configured ceiling.

    IV rank = None → fail-closed (assume expensive).
    """
    cfg = get_options_settings()

    if iv_rank is None:
        _log_check(loop_id, "iv_rank", False, {"iv_rank": None})
        return False, "IV rank unavailable — fail-closed"

    passed = iv_rank <= cfg.option_max_iv_rank
    reason = "" if passed else (
        f"IV rank {iv_rank:.2f} > max {cfg.option_max_iv_rank:.2f}"
    )
    _log_check(loop_id, "iv_rank", passed, {"iv_rank": iv_rank})
    return passed, reason


# ── S3: DTE Window ────────────────────────────────────────────────────────────

def check_dte(
    contract: OptionsContract,
    loop_id: str = "unset",
) -> SafetyResult:
    """
    Reject if DTE is outside the configured min/max range.

    Expired contracts (DTE <= 0) are always rejected.
    """
    cfg = get_options_settings()

    try:
        exp = date.fromisoformat(contract.expiration)
    except ValueError:
        _log_check(loop_id, "dte_window", False, {"reason": "bad expiration"})
        return False, f"Cannot parse expiration: {contract.expiration!r}"

    dte = (exp - date.today()).days

    if dte <= 0:
        _log_check(loop_id, "dte_window", False, {"dte": dte})
        return False, f"Contract expired (DTE={dte})"

    passed = cfg.option_min_dte <= dte <= cfg.option_max_dte
    reason = "" if passed else (
        f"DTE {dte} outside [{cfg.option_min_dte}, {cfg.option_max_dte}]"
    )
    _log_check(loop_id, "dte_window", passed, {"dte": dte})
    return passed, reason


# ── S4: Spread Width ──────────────────────────────────────────────────────────

def check_spread(
    quote: OptionsQuote,
    loop_id: str = "unset",
) -> SafetyResult:
    """
    Reject if bid/ask spread exceeds the configured max percentage of mid.

    Zero mid → fail-closed (no meaningful price).
    """
    cfg = get_options_settings()

    if quote.mid <= 0:
        _log_check(loop_id, "spread_width", False, {"mid": 0})
        return False, "Mid price is zero — cannot assess spread"

    spread_pct = (quote.ask - quote.bid) / quote.mid
    passed = spread_pct <= cfg.option_max_spread_pct
    reason = "" if passed else (
        f"Spread {spread_pct:.1%} > max {cfg.option_max_spread_pct:.1%}"
    )
    _log_check(loop_id, "spread_width", passed, {
        "bid": quote.bid, "ask": quote.ask, "spread_pct": round(spread_pct, 4),
    })
    return passed, reason


# ── S5: Delta Range ───────────────────────────────────────────────────────────

def check_delta(
    contract: OptionsContract,
    loop_id: str = "unset",
) -> SafetyResult:
    """
    Reject if absolute delta is outside the configured range.

    Delta = None → fail-closed.
    """
    cfg = get_options_settings()

    if contract.delta is None:
        _log_check(loop_id, "delta_range", False, {"delta": None})
        return False, "Delta unavailable — fail-closed"

    abs_delta = abs(contract.delta)
    passed = cfg.option_min_delta <= abs_delta <= cfg.option_max_delta
    reason = "" if passed else (
        f"Delta {abs_delta:.2f} outside [{cfg.option_min_delta}, {cfg.option_max_delta}]"
    )
    _log_check(loop_id, "delta_range", passed, {"delta": contract.delta})
    return passed, reason


# ── S6: Total Exposure Limit ──────────────────────────────────────────────────

def check_exposure(
    nav: float,
    current_options_value: float,
    new_position_cost: float,
    loop_id: str = "unset",
) -> SafetyResult:
    """
    Reject if adding this position would exceed the total options exposure cap.

    nav:                   Current net asset value.
    current_options_value: Sum of all existing options positions' market value.
    new_position_cost:     Cost of the proposed new position.
    """
    cfg = get_options_settings()

    if nav <= 0:
        _log_check(loop_id, "exposure_limit", False, {"nav": nav})
        return False, "NAV is zero or negative"

    total = current_options_value + new_position_cost
    exposure_pct = total / nav
    passed = exposure_pct <= cfg.option_max_total_exposure
    reason = "" if passed else (
        f"Options exposure {exposure_pct:.1%} would exceed "
        f"max {cfg.option_max_total_exposure:.1%}"
    )
    _log_check(loop_id, "exposure_limit", passed, {
        "nav": nav, "total_options": total, "exposure_pct": round(exposure_pct, 4),
    })
    return passed, reason


# ── S7: Max Loss Stop ────────────────────────────────────────────────────────

def check_max_loss(
    position: OptionsPosition,
    current_price: float,
    loop_id: str = "unset",
) -> SafetyResult:
    """
    Return False (fail = must exit) if unrealized loss exceeds the hard stop.

    This is checked in the exit engine loop, not at entry.
    """
    cfg = get_options_settings()

    if position.entry_price <= 0:
        return True, ""

    loss_pct = (position.entry_price - current_price) / position.entry_price
    if loss_pct >= cfg.option_max_loss_pct:
        reason = (
            f"Loss {loss_pct:.1%} >= hard stop {cfg.option_max_loss_pct:.1%}"
        )
        _log_check(loop_id, "max_loss_stop", False, {
            "entry": position.entry_price,
            "current": current_price,
            "loss_pct": round(loss_pct, 4),
        })
        return False, reason

    return True, ""


# ── Run All Entry Checks ─────────────────────────────────────────────────────

def run_entry_safety(
    contract: OptionsContract,
    quote: OptionsQuote,
    iv_rank: float | None,
    nav: float,
    current_options_value: float,
    new_position_cost: float,
    loop_id: str = "unset",
) -> SafetyResult:
    """
    Run all entry-time safety checks in sequence. Stops at first failure.

    Returns (True, "") if all pass, or (False, reason) on first failure.
    """
    checks = [
        lambda: check_options_enabled(loop_id),
        lambda: check_iv_rank(iv_rank, loop_id),
        lambda: check_dte(contract, loop_id),
        lambda: check_spread(quote, loop_id),
        lambda: check_delta(contract, loop_id),
        lambda: check_exposure(nav, current_options_value, new_position_cost, loop_id),
    ]

    for check_fn in checks:
        passed, reason = check_fn()
        if not passed:
            return False, reason

    return True, ""
