"""
strategies/equity_momentum.py — Disciplined equity momentum strategy.

Day/swing strategy for high-beta equities. Trades RTH only (09:30-16:00 ET).

Six scoring conditions (max 115 points):
  1. Price > SMA(20)             → 20 pts  (trend confirmation)
  2. SMA(20) > SMA(50)           → 15 pts  (uptrend structure)
  3. RSI(14) between 50-65       → 20 pts  (bullish momentum, not exhausted)
  4. RSI(14) below 30            → 25 pts  (oversold bounce setup — alternative path)
  5. MACD histogram positive     → 20 pts  (momentum confirmed)
  6. Volume ratio above 1.5×     → 15 pts  (institutional participation)

Threshold: base 65, shifted by regime (bullish -5, bearish +15).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sauce.core.config import get_settings
from sauce.core.schemas import Indicators, Order
from sauce.strategy import ExitPlan, Position, SignalResult, TierParams

logger = logging.getLogger(__name__)

# Eastern timezone for market hours
ET = ZoneInfo("America/New_York")


class EquityMomentum:
    """Disciplined momentum strategy for US equities during RTH."""

    name: str = "equity_momentum"

    @property
    def instruments(self) -> list[str]:
        return get_settings().equity_universe

    # Trend confirmation conditions
    TREND_ABOVE_SMA20_POINTS = 20
    UPTREND_STRUCTURE_POINTS = 15  # SMA20 > SMA50

    # Momentum conditions
    RSI_MOMENTUM_MIN = 50
    RSI_MOMENTUM_MAX = 65  # Tighter ceiling (was 70) — avoid exhaustion
    RSI_MOMENTUM_POINTS = 20

    # Mean-reversion alternative (oversold bounce)
    RSI_OVERSOLD = 30  # True oversold (was 40)
    RSI_OVERSOLD_POINTS = 25

    # MACD momentum
    MACD_MOMENTUM_POINTS = 20

    # Volume confirmation
    VOLUME_RATIO_MIN = 1.5
    VOLUME_POINTS = 15

    BASE_THRESHOLD = 65  # Require 65 pts = strong multi-indicator confluence
    # CORRECT: Bearish = HARDER to enter (capital preservation)
    REGIME_SHIFT = {"bullish": -5, "neutral": 0, "bearish": 15}

    def eligible(self, instrument: str, regime: str) -> bool:
        """Only trade during regular trading hours (09:30-16:00 ET)."""
        if instrument not in self.instruments:
            return False

        now_et = datetime.now(ET)
        # Check weekday (0=Mon, 6=Sun)
        if now_et.weekday() >= 5:
            return False

        # Check RTH: 09:30-16:00 ET
        hour, minute = now_et.hour, now_et.minute
        if hour < 9 or (hour == 9 and minute < 30):
            return False
        return not hour >= 16

    def score(
        self, indicators: Indicators, instrument: str, regime: str, current_price: float
    ) -> SignalResult:
        """Score the signal from 0–115 based on six conditions.

        Two MUTUALLY EXCLUSIVE entry paths — only the stronger path scores:
        1. Momentum breakout: price > SMA20, SMA20 > SMA50, RSI 50-65, MACD+
        2. Oversold bounce: RSI < 30, volume spike
        Trend and volume confirmation adds to whichever path is active.
        """
        rsi_14 = indicators.rsi_14
        macd_hist = indicators.macd_histogram
        vol_ratio = indicators.volume_ratio
        sma_20 = indicators.sma_20
        sma_50 = indicators.sma_50

        # === TREND CONFIRMATION (shared) ===
        trend_points = 0
        if sma_20 is not None and current_price > sma_20:
            trend_points += self.TREND_ABOVE_SMA20_POINTS
        if sma_20 is not None and sma_50 is not None and sma_20 > sma_50:
            trend_points += self.UPTREND_STRUCTURE_POINTS

        # === MOMENTUM PATH ===
        mo_points = trend_points
        if rsi_14 is not None and self.RSI_MOMENTUM_MIN <= rsi_14 <= self.RSI_MOMENTUM_MAX:
            mo_points += self.RSI_MOMENTUM_POINTS
        if macd_hist is not None and macd_hist > 0:
            mo_points += self.MACD_MOMENTUM_POINTS

        # === MEAN REVERSION PATH ===
        mr_points = 0
        if rsi_14 is not None and rsi_14 < self.RSI_OVERSOLD:
            mr_points += self.RSI_OVERSOLD_POINTS
        if macd_hist is not None and macd_hist > 0:
            mr_points += self.MACD_MOMENTUM_POINTS

        # Take the stronger path (mutually exclusive)
        points = max(mo_points, mr_points)

        # === VOLUME CONFIRMATION (adds to either path) ===
        if vol_ratio is not None and vol_ratio >= self.VOLUME_RATIO_MIN:
            points += self.VOLUME_POINTS

        # Apply regime threshold shift
        shift = self.REGIME_SHIFT.get(regime, 0)
        threshold = self.BASE_THRESHOLD + shift

        fired = points >= threshold
        side = "buy" if fired else "hold"

        logger.info(
            "SCORE %s: %d/%d (RSI=%.1f SMA20=%.2f SMA50=%.2f MACD=%.2f Vol=%.2f) regime=%s",
            instrument,
            points,
            threshold,
            rsi_14 or 0,
            sma_20 or 0,
            sma_50 or 0,
            macd_hist or 0,
            vol_ratio or 0,
            regime,
        )

        return SignalResult(
            symbol=instrument,
            side=side,
            score=points,
            threshold=threshold,
            fired=fired,
            rsi_14=rsi_14,
            macd_hist=macd_hist,
            bb_pct=None,  # Not used for equities
            volume_ratio=vol_ratio,
            regime=regime,
            strategy_name=self.name,
        )

    def build_order(self, signal: SignalResult, account: dict[str, Any], tier: TierParams) -> Order:
        """Construct a limit buy order sized by tier parameters, clamped to buying power."""
        equity = float(account.get("equity", "0"))
        buying_power = float(account.get("buying_power", "0"))
        ask = float(account.get("_ask", "0"))

        # Size: min(tier target, available buying power × 95% safety buffer)
        target_value = equity * tier.max_position_pct
        position_value = min(target_value, buying_power * 0.95)
        qty = position_value / ask if ask > 0 else 0.0

        # Equities: round to whole shares
        qty = int(qty)

        # Limit price: ask + 0.05% for faster fills (tighter than crypto)
        limit_price = round(ask * 1.0005, 2)

        return Order(
            symbol=signal.symbol,
            side="buy",
            qty=float(qty),
            order_type="limit",
            time_in_force="day",  # Day order for RTH
            limit_price=limit_price,
            stop_loss_price=round(limit_price * (1 - tier.stop_loss_pct), 2),
            take_profit_price=round(limit_price * (1 + tier.profit_target_pct), 2),
            as_of=datetime.now(UTC),
            prompt_version="v2",
            source="execution",
        )

    def build_exit_plan(self, position: Position, tier: TierParams) -> ExitPlan:
        """Equity exit plan with tighter stops during RTH."""
        return ExitPlan(
            stop_loss_pct=tier.stop_loss_pct,
            trail_activation_pct=tier.trail_activation_pct,
            trail_pct=tier.trail_pct,
            profit_target_pct=tier.profit_target_pct,
            rsi_exhaustion_threshold=tier.rsi_exhaustion_threshold,
            max_hold_hours=tier.max_hold_hours,
            time_stop_min_gain=tier.time_stop_min_gain,
            regime_stop=True,  # Close on bearish regime flip (equities more vulnerable)
        )
