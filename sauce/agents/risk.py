"""
agents/risk.py — Risk agent: validates signals against all configured risk limits.

Phase 5 IMPLEMENTATION — pure rule-based, no LLM call.

Rule-based approach rationale:
  - Rule evaluation is deterministic and fast.
  - LLM opinion on "is 5% of NAV too much?" adds latency/cost with no edge.
  - Claude is reserved for Research (signal generation) and Supervisor (final sanity).

Checks performed:
  1. Confidence floor: signal.confidence >= settings.min_confidence.
  2. Max position per symbol: existing + proposed <= equity * max_position_pct.
  3. Max portfolio exposure: total exposure + proposed <= equity * max_portfolio_exposure.
  4. Volatility (ATR) guard: ATR/price < 5% (avoid trading in extreme conditions).
  5. Daily loss already ok (was checked in loop pre-flight — double-check here).

Position sizing:
  - If all checks pass, qty = (equity * max_position_pct - existing_value) / mid_price.
  - Rounded down to avoid overshoot.
  - If qty <= 0 → veto (already at max).
"""

import logging
from datetime import datetime, timezone

from sauce.adapters.db import log_event
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, RiskCheckResult, RiskChecks, Signal

logger = logging.getLogger(__name__)

# Volatility guard: ATR/price above this threshold = too volatile to trade.
_MAX_ATR_RATIO = 0.05  # 5% of price


async def run(
    signal: Signal,
    account: dict,
    positions: list[dict],
    loop_id: str,
) -> RiskCheckResult:
    """
    Evaluate a signal against all configured risk limits.

    Returns RiskCheckResult with veto=False and a qty if the signal is approved,
    or veto=True (safe default) if any check fails.

    Parameters
    ----------
    signal:    Signal from the Research agent.
    account:   Account dict from broker.get_account().
    positions: Current open positions from broker.get_positions().
    loop_id:   Current loop run UUID for audit correlation.
    """
    settings = get_settings()
    db_path = str(settings.db_path)
    as_of = datetime.now(timezone.utc)

    def _veto(reason: str, checks: RiskChecks) -> RiskCheckResult:
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="risk_check",
                symbol=signal.symbol,
                payload={
                    "agent": "risk",
                    "veto": True,
                    "reason": reason,
                    "checks": checks.model_dump(),
                },
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return RiskCheckResult(
            symbol=signal.symbol,
            side=signal.side,
            veto=True,
            reason=reason,
            qty=None,
            checks=checks,
            as_of=as_of,
            prompt_version=settings.prompt_version,
        )

    # Safety: only evaluate non-hold signals
    if signal.side == "hold":
        checks = RiskChecks(
            max_position_pct_ok=True,
            max_exposure_ok=True,
            daily_loss_ok=True,
            volatility_ok=True,
            confidence_ok=False,
        )
        return _veto("Signal is hold — no risk check needed.", checks)

    # ── Parse account data ────────────────────────────────────────────────────
    try:
        equity = float(account.get("equity") or 0.0)
        last_equity = float(account.get("last_equity") or equity)
        buying_power = float(account.get("buying_power") or 0.0)
    except (TypeError, ValueError):
        checks = RiskChecks(
            max_position_pct_ok=False,
            max_exposure_ok=False,
            daily_loss_ok=False,
            volatility_ok=False,
            confidence_ok=False,
        )
        return _veto("Cannot parse account data (equity/last_equity/buying_power).", checks)

    if equity <= 0:
        checks = RiskChecks(
            max_position_pct_ok=False,
            max_exposure_ok=False,
            daily_loss_ok=False,
            volatility_ok=False,
            confidence_ok=False,
        )
        return _veto("Account equity is zero — cannot size position.", checks)

    mid_price = signal.evidence.price_reference.mid
    if mid_price <= 0:
        checks = RiskChecks(
            max_position_pct_ok=False,
            max_exposure_ok=False,
            daily_loss_ok=False,
            volatility_ok=False,
            confidence_ok=False,
        )
        return _veto("Mid price is zero — cannot size position.", checks)

    # ── Check 1: Confidence floor ─────────────────────────────────────────────
    confidence_ok = signal.confidence >= settings.min_confidence

    # ── Check 2: Max position per symbol ──────────────────────────────────────
    # Find existing position value for this symbol (if any)
    existing_qty = 0.0
    existing_value = 0.0
    for pos in positions:
        if str(pos.get("symbol", "")).upper() == signal.symbol.upper():
            try:
                existing_qty = float(pos.get("qty", 0.0) or 0.0)
                existing_value = abs(existing_qty) * mid_price
            except (TypeError, ValueError):
                existing_value = 0.0
            break

    max_position_value = equity * settings.max_position_pct
    remaining_capacity = max_position_value - existing_value
    max_position_pct_ok = remaining_capacity > 0

    # ── Check 3: Max portfolio exposure ───────────────────────────────────────
    # Sum all open position market values
    total_existing_exposure = 0.0
    for pos in positions:
        try:
            market_value = float(pos.get("market_value") or pos.get("current_price", 0.0))
            if market_value == 0.0:
                qty_val = float(pos.get("qty") or 0.0)
                total_existing_exposure += abs(qty_val) * mid_price
            else:
                total_existing_exposure += abs(market_value)
        except (TypeError, ValueError):
            pass

    max_exposure_value = equity * settings.max_portfolio_exposure
    proposed_addition = min(remaining_capacity, equity * settings.max_position_pct)
    max_exposure_ok = (total_existing_exposure + proposed_addition) <= max_exposure_value

    # ── Check 4: Volatility (ATR) guard ───────────────────────────────────────
    atr_14 = signal.evidence.indicators.atr_14
    if atr_14 is not None and mid_price > 0:
        atr_ratio = atr_14 / mid_price
        volatility_ok = atr_ratio < _MAX_ATR_RATIO
    else:
        # ATR unavailable — treat as acceptable (don't block on missing data)
        volatility_ok = True

    # ── Check 5: Daily loss ───────────────────────────────────────────────────
    # loop.py already checked this as a pre-flight. Re-confirm here.
    try:
        if last_equity > 0:
            daily_pnl = (equity - last_equity) / last_equity
            daily_loss_ok = daily_pnl >= -abs(settings.max_daily_loss_pct)
        else:
            daily_loss_ok = False
    except (TypeError, ValueError, ZeroDivisionError):
        daily_loss_ok = False

    checks = RiskChecks(
        max_position_pct_ok=max_position_pct_ok,
        max_exposure_ok=max_exposure_ok,
        daily_loss_ok=daily_loss_ok,
        volatility_ok=volatility_ok,
        confidence_ok=confidence_ok,
    )

    # ── Consolidate ───────────────────────────────────────────────────────────
    failed: list[str] = []
    if not confidence_ok:
        failed.append(
            f"confidence {signal.confidence:.2f} < min {settings.min_confidence}"
        )
    if not max_position_pct_ok:
        failed.append(
            f"already at max position for {signal.symbol} "
            f"(existing ≈ ${existing_value:.0f}, max ${max_position_value:.0f})"
        )
    if not max_exposure_ok:
        failed.append(
            f"total exposure would exceed limit "
            f"(current ${total_existing_exposure:.0f}, "
            f"max ${max_exposure_value:.0f})"
        )
    if not volatility_ok:
        failed.append(
            f"ATR/price too high ({atr_14 / mid_price:.2%} > {_MAX_ATR_RATIO:.0%})"  # type: ignore[operator]
        )
    if not daily_loss_ok:
        failed.append("daily loss limit already breached")

    if failed:
        return _veto("; ".join(failed), checks)

    # ── Compute approved qty ──────────────────────────────────────────────────
    approved_qty = remaining_capacity / mid_price
    # Also cap by buying power for buy orders
    if signal.side == "buy":
        max_by_buying_power = buying_power / mid_price
        approved_qty = min(approved_qty, max_by_buying_power)
    # For sells: cap by existing qty (cannot short sell)
    elif signal.side == "sell":
        approved_qty = min(approved_qty, abs(existing_qty))

    # Round down to 6 decimal places (crypto fractional shares)
    approved_qty = float(int(approved_qty * 1_000_000) / 1_000_000)

    if approved_qty <= 0:
        return _veto(
            f"Computed qty is zero after sizing (equity=${equity:.0f}, "
            f"max_position_pct={settings.max_position_pct:.0%})",
            checks,
        )

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="risk_check",
            symbol=signal.symbol,
            payload={
                "agent": "risk",
                "veto": False,
                "approved_qty": approved_qty,
                "checks": checks.model_dump(),
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    return RiskCheckResult(
        symbol=signal.symbol,
        side=signal.side,
        veto=False,
        reason=None,
        qty=approved_qty,
        checks=checks,
        as_of=as_of,
        prompt_version=settings.prompt_version,
    )
