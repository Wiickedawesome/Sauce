"""
strategies/crypto_momentum.py — Crypto Momentum Reversion strategy.

Phase 1 strategy ($1K–$10K). Trades BTC/USD, ETH/USD, SOL/USD on 5-minute cadence.

Five scoring conditions (max 100 points):
  1. RSI(14) below 35              → 25 pts  (oversold bounce setup)
  2. Price in lower 20% of BB(2.0) → 20 pts  (statistical support zone)
  3. MACD histogram negative       → 15 pts  (confirms dip — aligned with mean reversion)
  4. MACD momentum shift           → 20 pts  (MACD line > signal, or histogram > 0)
  5. Volume ratio above 1.5×       → 20 pts  (above-average participation)

Threshold: base 60, shifted by regime (bullish -5, bearish +10).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sauce.core.schemas import Indicators, Order
from sauce.strategy import ExitPlan, Position, SignalResult, TierParams

logger = logging.getLogger(__name__)


class CryptoMomentumReversion:
    """Aggressive momentum + mean-reversion strategy for 24/7 crypto markets.

    DISCIPLINED CONFIG: Higher thresholds require multi-indicator confluence.
    Regime shifts make bearish markets harder to enter (capital preservation).
    Combines mean-reversion (buy dips) with momentum breakout conditions.
    """

    name: str = "crypto_momentum"
    instruments: list[str] = ["BTC/USD", "ETH/USD", "SOL/USD"]

    # Scoring condition thresholds — DISCIPLINED
    RSI_OVERSOLD = 35  # True oversold: RSI must be < 35
    RSI_POINTS = 25

    BB_PROXIMITY = 0.20  # Lower 20% of BB range (tight)
    BB_POINTS = 20

    MACD_DIP_POINTS = 15  # histogram negative (confirms oversold dip)
    MACD_MOMENTUM_POINTS = 20  # histogram positive (reversal confirmed)

    VOLUME_RATIO_MIN = 1.5  # 1.5x avg volume (meaningful participation)
    VOLUME_POINTS = 20

    # Momentum breakout conditions
    MOMENTUM_RSI_MIN = 55  # RSI 55-70 = bullish momentum
    MOMENTUM_RSI_MAX = 70
    MOMENTUM_POINTS = 15

    BASE_THRESHOLD = 60  # Require 60% confluence (3+ conditions)
    # CORRECT: Bearish = HARDER to enter (capital preservation)
    REGIME_SHIFT = {"bullish": -5, "neutral": 0, "bearish": 10}

    def eligible(self, instrument: str, regime: str) -> bool:
        """Crypto trades 24/7 in all regimes."""
        return instrument in self.instruments

    def score(
        self, indicators: Indicators, instrument: str, regime: str, current_price: float
    ) -> SignalResult:
        """Score the signal from 0–100 based on conditions.

        Two MUTUALLY EXCLUSIVE entry paths — only the stronger path scores:
        1. Mean-reversion: RSI oversold + BB low + MACD negative (dip buying)
        2. Momentum breakout: MACD positive + RSI in momentum zone (trend following)
        Volume confirmation adds to whichever path is active.
        """
        rsi_14 = indicators.rsi_14
        macd_hist = indicators.macd_histogram
        vol_ratio = indicators.volume_ratio
        bb_pct = _compute_bb_pct(indicators, current_price)

        # === MEAN REVERSION PATH ===
        mr_points = 0
        if rsi_14 is not None and rsi_14 < self.RSI_OVERSOLD:
            mr_points += self.RSI_POINTS
        if bb_pct is not None and bb_pct <= self.BB_PROXIMITY:
            mr_points += self.BB_POINTS
        if macd_hist is not None and macd_hist < 0:
            mr_points += self.MACD_DIP_POINTS

        # === MOMENTUM BREAKOUT PATH ===
        mo_points = 0
        if macd_hist is not None and macd_hist > 0:
            mo_points += self.MACD_MOMENTUM_POINTS
        if rsi_14 is not None and self.MOMENTUM_RSI_MIN <= rsi_14 <= self.MOMENTUM_RSI_MAX:
            mo_points += self.MOMENTUM_POINTS

        # Take the stronger path (mutually exclusive — no double-counting)
        points = max(mr_points, mo_points)

        # === VOLUME CONFIRMATION (adds to either path) ===
        if vol_ratio is not None and vol_ratio >= self.VOLUME_RATIO_MIN:
            points += self.VOLUME_POINTS

        # Apply regime threshold shift (bearish = easier entry)
        shift = self.REGIME_SHIFT.get(regime, 0)
        threshold = self.BASE_THRESHOLD + shift

        fired = points >= threshold
        side = "buy" if fired else "hold"

        logger.info(
            "SCORE %s: %d/%d (RSI=%.1f BB=%.2f MACD=%.2f Vol=%.2f) regime=%s",
            instrument,
            points,
            threshold,
            rsi_14 or 0,
            bb_pct if bb_pct is not None else -1,
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
            bb_pct=bb_pct,
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

        # Limit price: ask + 0.1% for faster fills
        limit_price = round(ask * 1.001, 2)

        return Order(
            symbol=signal.symbol,
            side="buy",
            qty=round(qty, 8),  # crypto supports fractional
            order_type="limit",
            time_in_force="gtc",
            limit_price=limit_price,
            stop_loss_price=round(limit_price * (1 - tier.stop_loss_pct), 2),
            take_profit_price=round(limit_price * (1 + tier.profit_target_pct), 2),
            as_of=datetime.now(UTC),
            prompt_version="v2",
            source="execution",
        )

    def build_exit_plan(self, position: Position, tier: TierParams) -> ExitPlan:
        """Standard crypto exit plan from tier parameters."""
        return ExitPlan(
            stop_loss_pct=tier.stop_loss_pct,
            trail_activation_pct=tier.trail_activation_pct,
            trail_pct=tier.trail_pct,
            profit_target_pct=tier.profit_target_pct,
            rsi_exhaustion_threshold=tier.rsi_exhaustion_threshold,
            max_hold_hours=tier.max_hold_hours,
            time_stop_min_gain=tier.time_stop_min_gain,
        )


def _compute_bb_pct(indicators: Indicators, current_price: float) -> float | None:
    """Compute where price sits relative to Bollinger Bands.

    Returns 0.0 when price is at the lower band, 1.0 at the upper band.
    Values below 0.0 mean price is below the lower band.
    """
    lower = indicators.bb_lower
    upper = indicators.bb_upper

    if lower is None or upper is None:
        return None

    band_width = upper - lower
    if band_width <= 0:
        return None

    return (current_price - lower) / band_width
