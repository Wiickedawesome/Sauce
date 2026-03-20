"""
strategies/crypto_momentum.py — Crypto Momentum Reversion strategy.

Phase 1 strategy ($1K–$10K). Trades BTC/USD, ETH/USD, SOL/USD on 5-minute cadence.

Five scoring conditions (max 100 points):
  1. RSI(14) below 42              → 25 pts  (oversold bounce setup)
  2. Price in lower 25% of BB(2.0) → 20 pts  (statistical support zone)
  3. MACD histogram negative       → 15 pts  (confirms dip — aligned with mean reversion)
  4. MACD momentum shift           → 20 pts  (MACD line > signal, or histogram > 0)
  5. Volume ratio above 1.3×       → 20 pts  (above-average participation)

Threshold: base 50, shifted by regime (bullish -5, bearish +10).
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

    AGGRESSIVE CONFIG: Lower thresholds, inverted regime scoring (bearish = opportunity).
    Combines mean-reversion (buy dips) with momentum breakout conditions.
    """

    name: str = "crypto_momentum"
    instruments: list[str] = ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]

    # Scoring condition thresholds — AGGRESSIVE
    RSI_OVERSOLD = 45  # 45 RSI = more entry opportunities (was 42)
    RSI_POINTS = 25

    BB_PROXIMITY = 0.30  # Lower 30% of BB range (was 25%)
    BB_POINTS = 20

    MACD_DIP_POINTS = 15  # histogram negative (confirms oversold dip)
    MACD_MOMENTUM_POINTS = 20  # histogram positive (reversal confirmed)

    VOLUME_RATIO_MIN = 1.2  # 1.2x avg volume (was 1.3x — more triggers)
    VOLUME_POINTS = 20

    # Momentum breakout conditions (new)
    MOMENTUM_RSI_MIN = 55  # RSI 55-75 = bullish momentum
    MOMENTUM_RSI_MAX = 75
    MOMENTUM_POINTS = 15

    BASE_THRESHOLD = 40  # Fire on 40% confluence (was 50%)
    # INVERTED: Bearish = EASIER to enter (volatility = opportunity)
    REGIME_SHIFT = {"bullish": 0, "neutral": 0, "bearish": -10}

    def eligible(self, instrument: str, regime: str) -> bool:
        """Crypto trades 24/7 in all regimes."""
        return instrument in self.instruments

    def score(
        self, indicators: Indicators, instrument: str, regime: str, current_price: float
    ) -> SignalResult:
        """Score the signal from 0–100 based on six conditions.

        Combines mean-reversion (oversold bounce) with momentum breakout.
        """
        points = 0

        rsi_14 = indicators.rsi_14
        macd_hist = indicators.macd_histogram
        vol_ratio = indicators.volume_ratio
        bb_pct = _compute_bb_pct(indicators, current_price)

        # === MEAN REVERSION PATH ===
        # Condition 1: RSI(14) below oversold threshold
        if rsi_14 is not None and rsi_14 < self.RSI_OVERSOLD:
            points += self.RSI_POINTS

        # Condition 2: Price in lower portion of Bollinger Bands
        if bb_pct is not None and bb_pct <= self.BB_PROXIMITY:
            points += self.BB_POINTS

        # Condition 3: MACD histogram negative (confirms dip)
        if macd_hist is not None and macd_hist < 0:
            points += self.MACD_DIP_POINTS

        # === MOMENTUM BREAKOUT PATH ===
        # Condition 4: MACD histogram positive (momentum confirmed)
        if macd_hist is not None and macd_hist > 0:
            points += self.MACD_MOMENTUM_POINTS

        # Condition 5: RSI in momentum zone (55-75 = bullish but not exhausted)
        if rsi_14 is not None and self.MOMENTUM_RSI_MIN <= rsi_14 <= self.MOMENTUM_RSI_MAX:
            points += self.MOMENTUM_POINTS

        # === VOLUME CONFIRMATION ===
        # Condition 6: Volume ratio above average
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
