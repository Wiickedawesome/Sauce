"""
agents/exit_research.py — Exit research agent: evaluates open positions for exit signals.

Pure rule-based (no LLM call). Runs once per open position per loop cycle.

Exit conditions evaluated (any triggers an exit):
  1. Trailing stop — unrealized P&L has pulled back ≥ 30% from peak.
  2. Regime flip — market regime shifted to TRENDING_DOWN or VOLATILE.
  3. RSI overbought — RSI(14) > 72 on a long position.
  4. Stale position — held > 48 hours with < 1% unrealized gain.
  5. ATR stop-loss hit — current price breached the ATR-based stop.
  6. Profit target hit — current price reached entry + profit_target_atr_multiple × ATR.

Peak P&L tracking:
  - Each call reads the current peak from session memory.
  - If current unrealized P&L exceeds the stored peak, updates it.
  - Trailing stop fires when current P&L drops 30%+ from peak.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from sauce.adapters.db import log_event
from sauce.core.config import get_settings
from sauce.core.schemas import (
    AuditEvent,
    ExitSignal,
    Indicators,
    MarketRegime,
    PositionPeakPnL,
    PriceReference,
)
from sauce.memory.db import get_peak_pnl, write_peak_pnl

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
TRAILING_STOP_PULLBACK_PCT = 0.30   # exit if P&L drops 30% from peak
RSI_OVERBOUGHT_THRESHOLD = 72.0    # exit long if RSI > 72
STALE_HOLD_HOURS = 48.0            # exit if held > 48h with < 1% gain
STALE_HOLD_MIN_GAIN_PCT = 0.01     # minimum 1% gain to justify holding


async def run(
    position: dict[str, Any],
    quote: PriceReference,
    indicators: Indicators,
    regime: MarketRegime,
    loop_id: str,
    entry_time: datetime | None = None,
) -> ExitSignal:
    """
    Evaluate a single open position for exit conditions.

    Parameters
    ----------
    position:    Broker position dict (symbol, qty, market_value, avg_entry_price, unrealized_pl, etc.)
    quote:       Current price reference for the symbol.
    indicators:  Current technical indicators for the symbol.
    regime:      Current market regime.
    loop_id:     Current loop run UUID for audit correlation.
    entry_time:  When the position was opened (if known). Used for stale-hold check.

    Returns
    -------
    ExitSignal with action="exit" if any condition fires, otherwise action="hold".
    """
    settings = get_settings()
    session_memory_path = str(settings.session_memory_db_path)
    db_path = str(settings.db_path)
    as_of = datetime.now(timezone.utc)

    symbol = str(position.get("symbol", "")).upper()
    qty = float(position.get("qty") or 0.0)
    avg_entry_price = float(position.get("avg_entry_price") or 0.0)
    unrealized_pnl = float(position.get("unrealized_pl") or position.get("unrealized_pnl") or 0.0)

    def _exit(reason: str, urgency: str = "normal") -> ExitSignal:
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="exit_signal_generated",
                symbol=symbol,
                payload={
                    "agent": "exit_research",
                    "action": "exit",
                    "reason": reason,
                    "urgency": urgency,
                    "unrealized_pnl": unrealized_pnl,
                    "qty": qty,
                },
                prompt_version=settings.prompt_version,
            ),
            db_path=db_path,
        )
        return ExitSignal(
            symbol=symbol,
            action="exit",
            reason=reason,
            urgency=urgency,
            as_of=as_of,
            prompt_version=settings.prompt_version,
        )

    def _hold(reason: str = "No exit conditions met") -> ExitSignal:
        return ExitSignal(
            symbol=symbol,
            action="hold",
            reason=reason,
            urgency="normal",
            as_of=as_of,
            prompt_version=settings.prompt_version,
        )

    # Skip if position has no meaningful size
    if qty == 0:
        return _hold("Zero-qty position — nothing to exit")

    # ── Update peak P&L tracking ──────────────────────────────────────────────
    peak = get_peak_pnl(symbol, session_memory_path)
    current_peak = peak.peak_unrealized_pnl if peak else 0.0

    if unrealized_pnl > current_peak:
        write_peak_pnl(
            PositionPeakPnL(
                symbol=symbol,
                peak_unrealized_pnl=unrealized_pnl,
                peak_at=as_of,
            ),
            session_memory_path,
        )
        current_peak = unrealized_pnl

    # ── Condition 1: Trailing stop ────────────────────────────────────────────
    # Only fire if we've been profitable (peak > 0) and pulled back significantly
    if current_peak > 0 and unrealized_pnl < current_peak * (1.0 - TRAILING_STOP_PULLBACK_PCT):
        return _exit(
            f"Trailing stop: P&L pulled back to ${unrealized_pnl:.2f} from "
            f"peak ${current_peak:.2f} (>{TRAILING_STOP_PULLBACK_PCT:.0%} drawdown)",
            urgency="high",
        )

    # ── Condition 2: Regime flip ──────────────────────────────────────────────
    if regime in ("TRENDING_DOWN", "VOLATILE") and qty > 0:
        return _exit(
            f"Regime flip: market is {regime} while holding long position",
            urgency="normal",
        )

    # ── Condition 3: RSI overbought ───────────────────────────────────────────
    if indicators.rsi_14 is not None and indicators.rsi_14 > RSI_OVERBOUGHT_THRESHOLD and qty > 0:
        return _exit(
            f"RSI overbought: RSI(14)={indicators.rsi_14:.1f} > {RSI_OVERBOUGHT_THRESHOLD}",
            urgency="normal",
        )

    # ── Condition 4: Stale position ───────────────────────────────────────────
    if entry_time is not None and avg_entry_price > 0:
        hold_hours = (as_of - entry_time).total_seconds() / 3600.0
        gain_pct = unrealized_pnl / (avg_entry_price * abs(qty)) if abs(qty) > 0 else 0.0
        if hold_hours > STALE_HOLD_HOURS and gain_pct < STALE_HOLD_MIN_GAIN_PCT:
            return _exit(
                f"Stale position: held {hold_hours:.1f}h with {gain_pct:.2%} gain "
                f"(threshold: >{STALE_HOLD_HOURS}h, <{STALE_HOLD_MIN_GAIN_PCT:.0%})",
                urgency="normal",
            )

    # ── Condition 5: ATR stop-loss breach ─────────────────────────────────────
    if avg_entry_price > 0 and indicators.atr_14 is not None and indicators.atr_14 > 0:
        stop_price = avg_entry_price - settings.stop_loss_atr_multiple * indicators.atr_14
        if qty > 0 and quote.mid <= stop_price:
            return _exit(
                f"ATR stop hit: mid={quote.mid:.4f} <= stop={stop_price:.4f} "
                f"(entry={avg_entry_price:.4f}, ATR={indicators.atr_14:.4f})",
                urgency="high",
            )

    # ── Condition 6: Profit target hit ────────────────────────────────────────
    if avg_entry_price > 0 and indicators.atr_14 is not None and indicators.atr_14 > 0:
        target_price = avg_entry_price + settings.profit_target_atr_multiple * indicators.atr_14
        if qty > 0 and quote.mid >= target_price:
            return _exit(
                f"Profit target hit: mid={quote.mid:.4f} >= target={target_price:.4f} "
                f"(entry={avg_entry_price:.4f}, ATR={indicators.atr_14:.4f}, "
                f"multiple={settings.profit_target_atr_multiple}x)",
                urgency="normal",
            )

    return _hold()
