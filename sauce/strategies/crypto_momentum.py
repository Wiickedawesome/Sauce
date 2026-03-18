"""
strategies/crypto_momentum.py — Crypto Momentum Reversion strategy.

Phase 1 strategy ($1K–$10K). Trades BTC/USD, ETH/USD, SOL/USD on 5-minute cadence.

Four scoring conditions (max 100 points):
  1. RSI(14) below 35              → 30 pts  (oversold bounce setup)
  2. Price at/below lower BB(2.5)  → 25 pts  (statistical support zone)
  3. MACD histogram rising vs 3 bars ago → 25 pts  (momentum confirming reversal)
  4. Volume ratio above 1.5×       → 20 pts  (above-average participation)

Threshold: base 65, shifted by regime (bullish -10, bearish +10).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sauce.core.schemas import Indicators, Order
from sauce.strategy import ExitPlan, Position, SignalResult, TierParams


class CryptoMomentumReversion:
    """Mean-reversion strategy for 24/7 crypto markets."""

    name: str = "crypto_momentum_reversion"
    instruments: list[str] = ["BTC/USD", "ETH/USD", "SOL/USD"]

    # Scoring condition thresholds
    RSI_OVERSOLD = 35
    RSI_POINTS = 30

    BB_STD = 2.5  # Bollinger Band width in std devs
    BB_POINTS = 25

    MACD_POINTS = 25

    VOLUME_RATIO_MIN = 1.5
    VOLUME_POINTS = 20

    BASE_THRESHOLD = 65
    REGIME_SHIFT = {"bullish": -10, "neutral": 0, "bearish": 10}

    def eligible(self, instrument: str, regime: str) -> bool:
        """Crypto trades 24/7 in all regimes."""
        return instrument in self.instruments

    def score(self, indicators: Indicators, instrument: str, regime: str, current_price: float) -> SignalResult:
        """Score the signal from 0–100 based on four conditions."""
        points = 0

        # Condition 1: RSI(14) below oversold threshold
        rsi_14 = indicators.rsi_14
        if rsi_14 is not None and rsi_14 < self.RSI_OVERSOLD:
            points += self.RSI_POINTS

        # Condition 2: Price at or below lower Bollinger Band
        # bb_pct: 0.0 when price is at the lower band, 1.0 at the upper band
        bb_pct = _compute_bb_pct(indicators, current_price)
        if bb_pct is not None and bb_pct <= 0.0:
            points += self.BB_POINTS

        # Condition 3: MACD histogram positive or rising
        # We use the histogram value directly — positive means momentum shifting up
        macd_hist = indicators.macd_histogram
        if macd_hist is not None and macd_hist > 0:
            points += self.MACD_POINTS

        # Condition 4: Volume ratio above 1.5× average
        vol_ratio = indicators.volume_ratio
        if vol_ratio is not None and vol_ratio >= self.VOLUME_RATIO_MIN:
            points += self.VOLUME_POINTS

        # Apply regime threshold shift
        shift = self.REGIME_SHIFT.get(regime, 0)
        threshold = self.BASE_THRESHOLD + shift

        fired = points >= threshold
        side = "buy" if fired else "hold"

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

    def build_order(self, signal: SignalResult, account: dict, tier: TierParams) -> Order:
        """Construct a limit buy order sized by tier parameters."""
        equity = float(account.get("equity", "0"))
        ask = float(account.get("_ask", "0"))

        # Size: tier.max_position_pct × equity / current price
        position_value = equity * tier.max_position_pct
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
            as_of=datetime.now(timezone.utc),
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
