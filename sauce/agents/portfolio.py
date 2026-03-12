"""
agents/portfolio.py — Portfolio agent: reviews open positions, suggests stops/targets.

Phase 5 IMPLEMENTATION — pure rule-based, no LLM call.

Responsibilities:
  - For each open position that has a corresponding Signal this run:
    * Compute a suggested stop-loss price (ATR-based).
    * Compute a suggested profit target (ATR-based).
    * Flag if the position is over-concentrated.
  - Detect symbols in the universe with no coverage this run.
  - Compute total portfolio exposure as a fraction of equity.
  - Flag rebalance_needed if any position is over-concentrated.

IMPORTANT: The Portfolio agent makes SUGGESTIONS only. It never places orders.
All position changes must go through Research → Risk → Execution → Supervisor.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sauce.adapters.db import log_event
from sauce.core.config import get_settings
from sauce.core.schemas import AuditEvent, PortfolioReview, PositionNote, Signal

logger = logging.getLogger(__name__)


async def run(
    symbols: list[str],
    positions: list[dict[str, Any]],
    signals: list[Signal],
    loop_id: str,
    equity: float = 0.0,
    max_position_pct: float | None = None,
) -> PortfolioReview:
    """
    Review current portfolio and generate position notes.

    Parameters
    ----------
    symbols:    Full trading universe for this run.
    positions:  Current open positions from broker.get_positions().
    signals:    All signals generated this run (for stop/target computation).
    loop_id:    Current loop run UUID for audit correlation.
    equity:     Account equity in USD, used for accurate exposure and
                concentration calculations (Finding 1.5, 1.6). When 0 the
                agent falls back to position-count approximation.
    max_position_pct: Per-tier override for max position sizing. When None,
                      falls back to settings.max_position_pct.

    Returns
    -------
    PortfolioReview with suggested levels, exposure, and over-concentration flags.
    """
    settings = get_settings()
    effective_max_position_pct = (
        max_position_pct if max_position_pct is not None else settings.max_position_pct
    )
    db_path = str(settings.db_path)
    as_of = datetime.now(timezone.utc)

    signal_by_symbol: dict[str, Signal] = {
        s.symbol.upper(): s for s in signals
    }
    universe_upper: set[str] = {s.upper() for s in symbols}

    # ── Compute total position value and effective equity ───────────────────────
    # Use market_value if available, otherwise qty * mid from signal.
    total_pos_value = 0.0
    for pos in positions:
        try:
            mv = float(pos.get("market_value") or 0.0)
        except (TypeError, ValueError):
            mv = 0.0
        total_pos_value += abs(mv)

    # For concentration checks prefer caller-supplied equity; fall back to
    # total position value only when equity is unavailable (Finding 1.6).
    effective_equity = equity if equity > 0 else total_pos_value
    max_single_position_value = (
        effective_equity * effective_max_position_pct * settings.over_concentration_multiplier
        if effective_equity > 0
        else 0.0
    )

    # ── Build position reviews ────────────────────────────────────────────────
    position_reviews: list[PositionNote] = []
    rebalance_needed = False

    for pos in positions:
        try:
            pos_symbol = str(pos.get("symbol", "")).upper()
            qty = float(pos.get("qty") or 0.0)
            current_value = abs(float(pos.get("market_value") or 0.0))
        except (TypeError, ValueError):
            continue

        if not pos_symbol:
            continue

        sig = signal_by_symbol.get(pos_symbol)

        # Determine mid price and ATR from the current-run signal (if available)
        mid_for_stops: float | None = None
        atr_14: float | None = None
        if sig is not None:
            mid_for_stops = sig.evidence.price_reference.mid
            atr_14 = sig.evidence.indicators.atr_14

        # Compute stop and target prices
        suggested_stop: float | None = None
        suggested_target: float | None = None
        if mid_for_stops is not None and atr_14 is not None and atr_14 > 0:
            if qty > 0:  # Long position
                suggested_stop = mid_for_stops - settings.stop_loss_atr_multiple * atr_14
                suggested_target = mid_for_stops + settings.profit_target_atr_multiple * atr_14
            else:  # Short position
                suggested_stop = mid_for_stops + settings.stop_loss_atr_multiple * atr_14
                suggested_target = mid_for_stops - settings.profit_target_atr_multiple * atr_14

        # Over-concentration check
        over_concentrated = (
            max_single_position_value > 0
            and current_value > max_single_position_value
        )
        if over_concentrated:
            rebalance_needed = True

        position_reviews.append(
            PositionNote(
                symbol=pos_symbol,
                current_qty=qty,
                current_value_usd=current_value,
                suggested_stop_price=suggested_stop,
                suggested_target_price=suggested_target,
                over_concentrated=over_concentrated,
            )
        )

    # ── Total exposure fraction (Finding 1.5) ───────────────────────────────
    # Prefer real equity as denominator; fall back to position-count estimate.
    if equity > 0:
        total_exposure_pct = min(1.0, total_pos_value / equity)
    else:
        total_exposure_pct = min(1.0, len(positions) * settings.max_position_pct)

    # ── Uncovered symbols ──────────────────────────────────────────────────────
    covered_symbols = set(signal_by_symbol.keys())
    uncovered = sorted(universe_upper - covered_symbols)

    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="portfolio_review",
            symbol=None,
            payload={
                "agent": "portfolio",
                "open_positions": len(positions),
                "positions_reviewed": len(position_reviews),
                "rebalance_needed": rebalance_needed,
                "total_exposure_pct": total_exposure_pct,
                "uncovered_symbols": uncovered,
            },
            prompt_version=settings.prompt_version,
        ),
        db_path=db_path,
    )

    return PortfolioReview(
        positions=position_reviews,
        total_exposure_pct=total_exposure_pct,
        rebalance_needed=rebalance_needed,
        uncovered_symbols=uncovered,
        as_of=as_of,
        prompt_version=settings.prompt_version,
    )
