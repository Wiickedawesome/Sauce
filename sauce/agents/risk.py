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
  3b. Asset-class allocation cap: per-class exposure <= equity * max_{crypto,equity}_allocation_pct.
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
from sauce.adapters.market_data import is_crypto
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, RiskCheckResult, RiskChecks, Signal

logger = logging.getLogger(__name__)


async def run(
    signal: Signal,
    account: dict,
    positions: list[dict],
    loop_id: str,
    remaining_buying_power: float | None = None,
    max_position_pct: float | None = None,
    max_daily_loss_pct: float | None = None,
) -> RiskCheckResult:
    """
    Evaluate a signal against all configured risk limits.

    Returns RiskCheckResult with veto=False and a qty if the signal is approved,
    or veto=True (safe default) if any check fails.

    Parameters
    ----------
    signal:               Signal from the Research agent.
    account:              Account dict from broker.get_account().
    positions:            Current open positions from broker.get_positions().
    loop_id:              Current loop run UUID for audit correlation.
    remaining_buying_power: Buying power still uncommitted in this loop run.
                          Decremented by the loop after each approved buy order
                          so that two simultaneous buy signals cannot both claim
                          the full available cash.  When None, falls back to the
                          raw buying_power from the account dict (safe default).
    max_position_pct:     Per-tier override for max position sizing. When None,
                          falls back to settings.max_position_pct.
    max_daily_loss_pct:   Per-tier override for daily loss limit. When None,
                          falls back to settings.max_daily_loss_pct.
    """
    settings = get_settings()
    effective_max_position_pct = (
        max_position_pct if max_position_pct is not None else settings.max_position_pct
    )
    effective_max_daily_loss_pct = (
        max_daily_loss_pct if max_daily_loss_pct is not None else settings.max_daily_loss_pct
    )
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
            asset_class_ok=True,
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
        # Use caller-supplied remaining_buying_power when available. This
        # prevents multiple buy orders approved in the same loop run from each
        # claiming the full raw buying_power (Finding 1.3).
        effective_buying_power = (
            remaining_buying_power
            if remaining_buying_power is not None
            else buying_power
        )
    except (TypeError, ValueError):
        checks = RiskChecks(
            max_position_pct_ok=False,
            max_exposure_ok=False,
            asset_class_ok=False,
            daily_loss_ok=False,
            volatility_ok=False,
            confidence_ok=False,
        )
        return _veto("Cannot parse account data (equity/last_equity/buying_power).", checks)

    if equity <= 0:
        checks = RiskChecks(
            max_position_pct_ok=False,
            max_exposure_ok=False,
            asset_class_ok=False,
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
            asset_class_ok=False,
            daily_loss_ok=False,
            volatility_ok=False,
            confidence_ok=False,
        )
        return _veto("Mid price is zero — cannot size position.", checks)

    # ── Check 0: Bid-ask spread guard (Finding 2.5) ────────────────────────────
    _is_crypto = is_crypto(signal.symbol)
    price_ref = signal.evidence.price_reference
    spread_too_wide = False
    bid_ask_spread = 0.0
    effective_max_spread = (
        settings.max_spread_pct_crypto if _is_crypto else settings.max_spread_pct
    )
    if price_ref.bid > 0 and price_ref.ask > 0 and price_ref.mid > 0:
        bid_ask_spread = (price_ref.ask - price_ref.bid) / price_ref.mid
        spread_too_wide = bid_ask_spread > effective_max_spread

    # ── Check 0b: Volume / liquidity guard (Finding 2.5) ─────────────────────
    # Estimate the maximum proposed order size (shares) and compare against
    # the average daily volume supplied by the Research agent. A proposed order
    # exceeding max_volume_participation × daily_volume indicates excessive
    # market impact and should be vetoed.
    # NOTE: Skip for crypto pairs — Alpaca-reported crypto volume does not
    # reflect actual exchange liquidity (orders are routed to real exchanges
    # with far deeper books). The check remains active for equities.
    volume_too_low = False
    volume_1d_avg = signal.evidence.indicators.volume_1d_avg
    if not _is_crypto and volume_1d_avg is not None and volume_1d_avg > 0 and mid_price > 0:
        # Conservative upper-bound estimate: full remaining position capacity.
        max_proposed_qty = (equity * effective_max_position_pct) / mid_price
        volume_too_low = max_proposed_qty > volume_1d_avg * settings.max_volume_participation

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

    max_position_value = equity * effective_max_position_pct
    remaining_capacity = max_position_value - existing_value
    max_position_pct_ok = remaining_capacity > 0

    # ── Check 3: Max portfolio exposure ───────────────────────────────────────
    # Sum all open position market values
    total_existing_exposure = 0.0
    for pos in positions:
        try:
            raw_mv = pos.get("market_value")
            if raw_mv is not None:
                market_value = float(raw_mv)
            else:
                market_value = 0.0
            if market_value == 0.0:
                qty_val = float(pos.get("qty") or 0.0)
                total_existing_exposure += abs(qty_val) * float(pos.get("current_price") or 0.0)
            else:
                total_existing_exposure += abs(market_value)
        except (TypeError, ValueError):
            pass  # Unparseable position value — skip this position in exposure sum (conservative)

    max_exposure_value = equity * settings.max_portfolio_exposure
    proposed_addition = min(remaining_capacity, equity * effective_max_position_pct)
    max_exposure_ok = (total_existing_exposure + proposed_addition) <= max_exposure_value

    # ── Check 3b: Asset-class allocation cap ──────────────────────────────────
    # Prevent any single asset class (crypto or equities) from consuming all
    # buying power.  Without this guard the 24/7 crypto market can exhaust
    # capital before equities market opens.
    crypto_exposure = 0.0
    equity_exposure = 0.0
    for pos in positions:
        try:
            mv = float(pos.get("market_value") or 0.0)
            if mv == 0.0:
                mv = abs(float(pos.get("qty") or 0.0)) * float(pos.get("current_price") or 0.0)
            mv = abs(mv)
        except (TypeError, ValueError):
            mv = 0.0
        sym = str(pos.get("symbol", ""))
        if "/" in sym:
            crypto_exposure += mv
        else:
            equity_exposure += mv

    if signal.side == "buy":
        if _is_crypto:
            asset_class_cap = equity * settings.max_crypto_allocation_pct
            asset_class_ok = (crypto_exposure + proposed_addition) <= asset_class_cap
        else:
            asset_class_cap = equity * settings.max_equity_allocation_pct
            asset_class_ok = (equity_exposure + proposed_addition) <= asset_class_cap
    else:
        # Sells always OK — reducing exposure is never blocked by this cap.
        asset_class_ok = True
        asset_class_cap = 0.0  # not used

    # ── Check 4: Volatility (ATR) guard ───────────────────────────────────────
    atr_14 = signal.evidence.indicators.atr_14
    atr_ratio: float | None = None
    effective_max_atr = (
        settings.max_atr_ratio_crypto if _is_crypto else settings.max_atr_ratio
    )
    if atr_14 is not None and mid_price > 0:
        atr_ratio = atr_14 / mid_price
        volatility_ok = atr_ratio < effective_max_atr
    else:
        # ATR unavailable — fail-closed unless operator has explicitly opted out
        # (Finding 2.4). Instruments with too-short history or data outages are
        # vetoed by default to prevent trading in unmeasurable volatility regimes.
        volatility_ok = settings.allow_no_atr

    # ── Check 5: Daily loss ───────────────────────────────────────────────────
    # loop.py already checked this as a pre-flight. Re-confirm here.
    try:
        if last_equity > 0:
            daily_pnl = (equity - last_equity) / last_equity
            daily_loss_ok = daily_pnl >= -abs(effective_max_daily_loss_pct)
        else:
            daily_loss_ok = False
    except (TypeError, ValueError, ZeroDivisionError):
        daily_loss_ok = False

    checks = RiskChecks(
        max_position_pct_ok=max_position_pct_ok,
        max_exposure_ok=max_exposure_ok,
        asset_class_ok=asset_class_ok,
        daily_loss_ok=daily_loss_ok,
        volatility_ok=volatility_ok,
        confidence_ok=confidence_ok,
    )

    # ── Consolidate ───────────────────────────────────────────────────────────
    failed: list[str] = []
    if spread_too_wide:
        failed.append(
            f"bid-ask spread too wide ({bid_ask_spread:.2%} > {effective_max_spread:.2%})"
        )
    if volume_too_low:
        max_proposed_qty_display = (equity * effective_max_position_pct) / mid_price
        failed.append(
            f"order size ({max_proposed_qty_display:.0f} shares) would exceed "
            f"{settings.max_volume_participation:.0%} of est. daily volume "
            f"({volume_1d_avg:.0f} shares) — excessive market impact"
        )
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
    if not asset_class_ok:
        class_label = "crypto" if _is_crypto else "equity"
        current_class_exp = crypto_exposure if _is_crypto else equity_exposure
        failed.append(
            f"{class_label} allocation would exceed cap "
            f"(current ${current_class_exp:.0f} + proposed ${proposed_addition:.0f} "
            f"> cap ${asset_class_cap:.0f})"
        )
    if not volatility_ok:
        if atr_ratio is not None:
            failed.append(
                f"ATR/price too high ({atr_ratio:.2%} > {effective_max_atr:.0%})"
            )
        else:
            failed.append("ATR unavailable — cannot assess volatility (allow_no_atr=False)")
    if not daily_loss_ok:
        failed.append("daily loss limit already breached")

    if failed:
        return _veto("; ".join(failed), checks)

    # ── Compute approved qty ──────────────────────────────────────────────────
    approved_qty = remaining_capacity / mid_price
    # Also cap by buying power for buy orders. Use effective_buying_power which
    # already has previously committed notional subtracted by the caller.
    if signal.side == "buy":
        if effective_buying_power <= 0:
            return _veto(
                "No remaining buying power for additional buy orders this run.",
                checks,
            )
        max_by_buying_power = effective_buying_power / mid_price
        approved_qty = min(approved_qty, max_by_buying_power)
    # For sells: cap by existing qty (cannot short sell)
    elif signal.side == "sell":
        approved_qty = min(approved_qty, abs(existing_qty))

    # Round down to 6 decimal places (crypto fractional shares)
    approved_qty = float(int(approved_qty * 1_000_000) / 1_000_000)

    if approved_qty <= 0:
        return _veto(
            f"Computed qty is zero after sizing (equity=${equity:.0f}, "
            f"max_position_pct={effective_max_position_pct:.0%})",
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
