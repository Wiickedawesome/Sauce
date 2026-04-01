"""
strategies/options_momentum.py — Aggressive options momentum strategy.

Simple calls/puts on high-beta underlyings. Day/swing trades.

Entry logic:
1. Score underlying using equity-style momentum scoring
2. Select contract: delta 0.25-0.40, DTE 7-45, liquid strikes
3. Direction: calls for bullish, puts for bearish setups

Exit conditions:
1. Hard stop — premium dropped 50%
2. Profit target — premium gained 100%+ (doubles)
3. DTE threshold — close at 2 DTE regardless
4. Underlying reversal — RSI flipped against position
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sauce.core.config import get_settings
from sauce.core.options_schemas import (
    OptionContract,
    OptionsOrder,
    OptionsPosition,
    OptionsSignalResult,
)
from sauce.core.schemas import Indicators
from sauce.strategy import TierParams

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class OptionsMomentum:
    """Aggressive momentum strategy for options on liquid underlyings."""

    name: str = "options_momentum"

    # Scoring weights (underlying analysis)
    RSI_OVERSOLD_THRESHOLD = 35  # Buy calls
    RSI_OVERBOUGHT_THRESHOLD = 65  # Buy puts
    RSI_POINTS = 25

    TREND_ABOVE_SMA20_POINTS = 15
    MACD_MOMENTUM_POINTS = 20
    VOLUME_RATIO_MIN = 1.2
    VOLUME_POINTS = 15

    BASE_THRESHOLD = 40  # Aggressive
    REGIME_SHIFT = {"bullish": -5, "neutral": 0, "bearish": -5}  # Easier entry in volatility

    # Contract selection
    MIN_OPEN_INTEREST = 100
    MAX_SPREAD_PCT = 0.10  # Max 10% bid-ask spread

    @property
    def _settings(self):
        return get_settings()

    @property
    def instruments(self) -> list[str]:
        """Underlyings approved for options trading."""
        return self._settings.options_universe

    def eligible(self, underlying: str, regime: str) -> bool:
        """Check if we can trade options on this underlying right now."""
        if not self._settings.options_enabled:
            return False
        if underlying not in self.instruments:
            return False

        # Options only during RTH
        now_et = datetime.now(ET)
        if now_et.weekday() >= 5:
            return False
        hour, minute = now_et.hour, now_et.minute
        if hour < 9 or (hour == 9 and minute < 30):
            return False
        return not hour >= 16

    def score(
        self,
        indicators: Indicators,
        underlying: str,
        regime: str,
        current_price: float,
    ) -> OptionsSignalResult:
        """
        Score the underlying for options entry.

        Returns OptionsSignalResult with direction (call/put) based on momentum.
        """
        points = 0
        option_type: str = "call"  # Default

        rsi_14 = indicators.rsi_14
        macd_hist = indicators.macd_histogram
        vol_ratio = indicators.volume_ratio
        sma_20 = indicators.sma_20
        atr_14 = indicators.atr_14

        # === DIRECTION DETERMINATION ===
        # RSI oversold → buy calls (bullish reversal)
        if rsi_14 is not None and rsi_14 < self.RSI_OVERSOLD_THRESHOLD:
            points += self.RSI_POINTS
            option_type = "call"
        # RSI overbought → buy puts (bearish reversal)
        elif rsi_14 is not None and rsi_14 > self.RSI_OVERBOUGHT_THRESHOLD:
            points += self.RSI_POINTS
            option_type = "put"

        # === TREND CONFIRMATION ===
        if sma_20 is not None and (
            (option_type == "call" and current_price > sma_20)
            or (option_type == "put" and current_price < sma_20)
        ):
            points += self.TREND_ABOVE_SMA20_POINTS

        # === MACD MOMENTUM ===
        if macd_hist is not None and (
            (option_type == "call" and macd_hist > 0)
            or (option_type == "put" and macd_hist < 0)
        ):
            points += self.MACD_MOMENTUM_POINTS

        # === VOLUME CONFIRMATION ===
        if vol_ratio is not None and vol_ratio >= self.VOLUME_RATIO_MIN:
            points += self.VOLUME_POINTS

        # Apply regime shift
        shift = self.REGIME_SHIFT.get(regime, 0)
        threshold = self.BASE_THRESHOLD + shift

        fired = points >= threshold

        logger.info(
            "OPTIONS SCORE %s: %d/%d type=%s (RSI=%.1f MACD=%.2f Vol=%.2f) regime=%s",
            underlying,
            points,
            threshold,
            option_type,
            rsi_14 or 0,
            macd_hist or 0,
            vol_ratio or 0,
            regime,
        )

        return OptionsSignalResult(
            underlying=underlying,
            option_type=option_type,  # type: ignore[arg-type]
            score=points,
            threshold=threshold,
            fired=fired,
            rsi_14=rsi_14,
            macd_hist=macd_hist,
            atr_14=atr_14,
            regime=regime,
            strategy_name=self.name,
            # Contract details filled in by select_contract()
            contract_symbol=None,
            strike=None,
            expiration=None,
            dte=None,
            delta=None,
            bid=None,
            ask=None,
        )

    def select_contract(
        self,
        signal: OptionsSignalResult,
        available_contracts: list[OptionContract],
        current_price: float,
    ) -> OptionContract | None:
        """
        Select the best contract from available options chain.

        Criteria:
        - Correct type (call/put)
        - DTE within range
        - Delta within range
        - Sufficient open interest
        - Tight bid-ask spread
        """
        settings = self._settings

        # Filter by type
        candidates = [c for c in available_contracts if c.option_type == signal.option_type]

        # Filter by DTE
        date.today()
        candidates = [
            c for c in candidates if settings.options_dte_min <= c.dte <= settings.options_dte_max
        ]

        # Filter by delta (use absolute value)
        candidates = [
            c
            for c in candidates
            if c.delta is not None
            and settings.options_delta_min <= abs(c.delta) <= settings.options_delta_max
        ]

        # Filter by open interest
        candidates = [
            c
            for c in candidates
            if c.open_interest is not None and c.open_interest >= self.MIN_OPEN_INTEREST
        ]

        # Filter by spread
        candidates = [
            c
            for c in candidates
            if c.bid is not None
            and c.ask is not None
            and c.bid > 0
            and (c.ask - c.bid) / c.bid <= self.MAX_SPREAD_PCT
        ]

        if not candidates:
            logger.warning(
                "No suitable contract found for %s %s (checked %d options)",
                signal.underlying,
                signal.option_type,
                len(available_contracts),
            )
            return None

        # Sort by:
        # 1. Delta closest to target (0.30-0.35 sweet spot)
        # 2. Highest open interest (liquidity)
        target_delta = 0.32

        def score_contract(c: OptionContract) -> tuple[float, int]:
            delta_distance = abs(abs(c.delta or 0) - target_delta)
            oi = c.open_interest or 0
            return (delta_distance, -oi)  # Lower is better

        candidates.sort(key=score_contract)

        best = candidates[0]
        logger.info(
            "Selected contract: %s strike=%.2f exp=%s delta=%.2f OI=%d",
            best.contract_symbol,
            best.strike,
            best.expiration,
            best.delta or 0,
            best.open_interest or 0,
        )
        return best

    def build_order(
        self,
        signal: OptionsSignalResult,
        contract: OptionContract,
        account: dict[str, Any],
        tier: TierParams,
    ) -> OptionsOrder:
        """Build an options order sized by tier and premium limits."""
        settings = self._settings
        equity = float(account.get("equity", "0"))
        size_fraction = float(account.get("_position_size_fraction", 1.0))

        # Max premium per position
        max_premium = equity * settings.options_max_premium_pct * max(0.0, min(1.0, size_fraction))
        # Premium per contract (use ask for buying)
        premium_per_contract = (contract.ask or 0) * 100  # 100 shares per contract

        qty = 1 if premium_per_contract <= 0 else int(max_premium / premium_per_contract)

        # Cap by max contracts setting
        qty = min(qty, settings.options_max_contracts_per_position)
        if qty < 1:
            raise ValueError("analyst-sized premium budget is too small for one contract")

        # Limit price: midpoint of bid-ask
        limit_price = ((contract.bid or 0) + (contract.ask or 0)) / 2
        limit_price = round(limit_price, 2)

        # Stop loss: 50% of premium
        stop_loss = round(limit_price * 0.50, 2)

        # Take profit: 100% gain (double)
        take_profit = round(limit_price * 2.0, 2)

        return OptionsOrder(
            underlying=signal.underlying,
            contract_symbol=contract.contract_symbol,
            option_type=signal.option_type,
            side="buy",
            qty=qty,
            limit_price=limit_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            time_in_force="day",
            stage="entry",
            source="options_entry",
        )

    def build_exit_order(
        self,
        position: OptionsPosition,
        current_bid: float,
        reason: str,
    ) -> OptionsOrder:
        """Build a sell order to exit an options position."""
        # Use bid for selling
        limit_price = round(current_bid * 0.995, 2)  # Slight discount for fill

        return OptionsOrder(
            underlying=position.underlying,
            contract_symbol=position.contract_symbol,
            option_type=position.option_type,
            side="sell",
            qty=position.qty,
            limit_price=limit_price,
            time_in_force="day",
            stage="exit",
            source="options_exit",
        )
