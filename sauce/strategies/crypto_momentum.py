"""
strategies/crypto_momentum.py — Crypto Momentum Reversion strategy.

Phase 1 strategy ($1K–$10K). Trades BTC/USD, ETH/USD, SOL/USD on 5-minute cadence.

Seven scoring conditions (max 70 points):
  1. Price > SMA(20)              → 15 pts  (trend confirmation)
  2. SMA(20) > SMA(50)            → 10 pts  (uptrend structure)
  3. RSI(14) below 35             → 25 pts  (oversold bounce setup)
  4. Price in lower 20% of BB(2.0)→ 20 pts  (statistical support zone)
  5. MACD histogram negative      → 15 pts  (confirms dip — mean reversion)
  6. MACD histogram positive      → 20 pts  (momentum confirmed)
  7. RSI(14) 55-70                → 15 pts  (bullish momentum zone)

Volume bonus (10 pts) when vol_ratio >= 0.5× (lowered for Alpaca paper crypto).
Threshold: base 50, shifted by regime (bullish -5, bearish +10).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sauce.core.config import get_settings
from sauce.core.schemas import Indicators, Order
from sauce.research.profiles import DEFAULT_STRATEGY_PROFILES, get_strategy_profile
from sauce.strategy import ExitPlan, Position, SignalResult, TierParams

logger = logging.getLogger(__name__)


class CryptoMomentumReversion:
    """Aggressive momentum + mean-reversion strategy for 24/7 crypto markets.

    DISCIPLINED CONFIG: Higher thresholds require multi-indicator confluence.
    Regime shifts make bearish markets harder to enter (capital preservation).
    Combines mean-reversion (buy dips) with momentum breakout conditions.
    Trend confirmation (price vs SMA) adds scoring depth like equity strategy.
    """

    name: str = "crypto_momentum"
    DEFAULT_PROFILE = DEFAULT_STRATEGY_PROFILES["crypto_momentum"]

    def __init__(self, profile_override: dict[str, Any] | None = None) -> None:
        self._profile_override = profile_override

    @property
    def instruments(self) -> list[str]:
        return get_settings().crypto_universe

    def _profile(self) -> dict[str, Any]:
        return get_strategy_profile(self.name, self.DEFAULT_PROFILE, self._profile_override)

    def eligible(self, instrument: str, regime: str) -> bool:
        """Crypto trades 24/7 in all regimes."""
        return instrument in self.instruments

    def score(
        self, indicators: Indicators, instrument: str, regime: str, current_price: float
    ) -> SignalResult:
        """Score the signal from 0–70 based on conditions.

        Two MUTUALLY EXCLUSIVE entry paths — only the stronger path scores:
        1. Mean-reversion: RSI oversold + BB low + MACD negative (dip buying)
        2. Momentum breakout: trend + MACD positive + RSI momentum zone (trend following)
        Volume confirmation adds to whichever path is active.
        """
        rsi_14 = indicators.rsi_14
        macd_hist = indicators.macd_histogram
        vol_ratio = indicators.volume_ratio
        bb_pct = _compute_bb_pct(indicators, current_price)
        sma_20 = indicators.sma_20
        sma_50 = indicators.sma_50
        stoch_k = indicators.stoch_k
        vwap = indicators.vwap
        profile = self._profile()

        # === TREND CONFIRMATION ===
        trend_points = 0
        if sma_20 is not None and current_price > sma_20:
            trend_points += int(profile["trend_above_sma20_points"])
        if sma_20 is not None and sma_50 is not None and sma_20 > sma_50:
            trend_points += int(profile["uptrend_structure_points"])

        # === MEAN REVERSION PATH ===
        mr_points = 0
        if rsi_14 is not None and rsi_14 < float(profile["rsi_oversold"]):
            mr_points += int(profile["rsi_points"])
        if bb_pct is not None and bb_pct <= float(profile["bb_proximity"]):
            mr_points += int(profile["bb_points"])
        if macd_hist is not None and macd_hist < 0:
            mr_points += int(profile["macd_dip_points"])
        if stoch_k is not None and stoch_k < 20:
            mr_points += int(profile["stoch_confirm_points"])
        if vwap is not None and current_price <= vwap:
            mr_points += int(profile["vwap_confirm_points"])

        # === MOMENTUM BREAKOUT PATH (includes trend) ===
        mo_points = trend_points
        if macd_hist is not None and macd_hist > 0:
            mo_points += int(profile["macd_momentum_points"])
        if rsi_14 is not None and float(profile["momentum_rsi_min"]) <= rsi_14 <= float(profile["momentum_rsi_max"]):
            mo_points += int(profile["momentum_points"])
        if vwap is not None and current_price >= vwap:
            mo_points += int(profile["vwap_confirm_points"])

        # Take the stronger path (mutually exclusive — no double-counting)
        points = max(mr_points, mo_points)

        # === VOLUME CONFIRMATION (adds to either path) ===
        if vol_ratio is not None and vol_ratio >= float(profile["volume_ratio_min"]):
            points += int(profile["volume_points"])

        # Apply regime threshold shift (bearish = easier entry)
        shift = int(profile["regime_shift"].get(regime, 0))
        threshold = int(profile["base_threshold"]) + shift

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
        target_value = float(account.get("_target_order_value", equity * tier.max_position_pct))
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
