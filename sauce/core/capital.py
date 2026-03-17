"""
core/capital.py — Capital tier system for dynamic position sizing.

Maps account equity to a tier, each with its own risk parameters:
  - max_position_pct (position sizing)
  - max_daily_loss_pct (daily drawdown guard)
  - max_positions (concurrent open positions)
  - cash_reserve_pct (minimum cash buffer)
  - allowed_setups (which strategy setups may fire)

Tier boundaries:
  seed       $500  – $2,499
  building   $2,500 – $4,999
  growing    $5,000 – $9,999
  scaling    $10,000 – $24,999
  operating  $25,000+
"""

from __future__ import annotations

from pydantic import model_validator

from sauce.core.schemas import CapitalTier, SetupType, StrictModel


class TierParameters(StrictModel):
    """Risk parameters for a single capital tier."""

    tier: CapitalTier
    equity_min: float
    equity_max: float | None = None  # None → no upper bound (top tier)
    allowed_setups: list[SetupType]
    max_positions: int
    max_position_pct: float
    cash_reserve_pct: float
    max_daily_loss_pct: float

    # ── Tier-aware aggression fields ──────────────────────────────────────
    min_confidence: float = 0.30
    stop_loss_atr_multiple: float = 2.0
    profit_target_atr_multiple: float = 3.0
    min_setup_score_offset: float = 0.0
    crypto_regime_filter: list[str] | None = None  # None → use hardcoded defaults
    stale_hold_hours: float = 48.0
    trailing_stop_pct: float = 0.30

    @model_validator(mode="after")
    def _enforce_safety_floors(self) -> "TierParameters":
        """Clamp aggressive values to prevent account-destroying settings."""
        if self.min_confidence < 0.15:
            object.__setattr__(self, "min_confidence", 0.15)
        if self.stop_loss_atr_multiple < 1.0:
            object.__setattr__(self, "stop_loss_atr_multiple", 1.0)
        if self.min_setup_score_offset < -15.0:
            object.__setattr__(self, "min_setup_score_offset", -15.0)
        if self.max_daily_loss_pct > 0.10:
            object.__setattr__(self, "max_daily_loss_pct", 0.10)
        return self


TIER_TABLE: list[TierParameters] = [
    TierParameters(
        tier="seed",
        equity_min=500.0,
        equity_max=2_499.99,
        allowed_setups=["crypto_mean_reversion", "equity_trend_pullback", "crypto_momentum"],
        max_positions=20,
        max_position_pct=0.50,
        cash_reserve_pct=0.10,
        max_daily_loss_pct=0.08,
        min_confidence=0.20,
        stop_loss_atr_multiple=1.5,
        profit_target_atr_multiple=4.0,
        min_setup_score_offset=-10.0,
        crypto_regime_filter=["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"],
        stale_hold_hours=24.0,
        trailing_stop_pct=0.40,
    ),
    TierParameters(
        tier="building",
        equity_min=2_500.0,
        equity_max=4_999.99,
        allowed_setups=["crypto_mean_reversion", "equity_trend_pullback", "crypto_breakout", "crypto_momentum"],
        max_positions=10,
        max_position_pct=0.35,
        cash_reserve_pct=0.15,
        max_daily_loss_pct=0.05,
        min_confidence=0.25,
        stop_loss_atr_multiple=1.5,
        profit_target_atr_multiple=3.5,
        min_setup_score_offset=-5.0,
        crypto_regime_filter=["TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE"],
        stale_hold_hours=36.0,
        trailing_stop_pct=0.35,
    ),
    TierParameters(
        tier="growing",
        equity_min=5_000.0,
        equity_max=9_999.99,
        allowed_setups=[
            "crypto_mean_reversion",
            "equity_trend_pullback",
            "crypto_breakout",
            "crypto_momentum",
        ],
        max_positions=6,
        max_position_pct=0.25,
        cash_reserve_pct=0.20,
        max_daily_loss_pct=0.03,
        min_confidence=0.25,
        stop_loss_atr_multiple=2.0,
        profit_target_atr_multiple=3.0,
        min_setup_score_offset=0.0,
        crypto_regime_filter=["TRENDING_UP", "RANGING"],
        stale_hold_hours=48.0,
        trailing_stop_pct=0.30,
    ),
    TierParameters(
        tier="scaling",
        equity_min=10_000.0,
        equity_max=24_999.99,
        allowed_setups=[
            "crypto_mean_reversion",
            "equity_trend_pullback",
            "crypto_breakout",
            "crypto_momentum",
        ],
        max_positions=8,
        max_position_pct=0.15,
        cash_reserve_pct=0.20,
        max_daily_loss_pct=0.02,
        min_confidence=0.30,
        stop_loss_atr_multiple=2.0,
        profit_target_atr_multiple=3.0,
        min_setup_score_offset=0.0,
        crypto_regime_filter=["TRENDING_UP", "RANGING"],
        stale_hold_hours=48.0,
        trailing_stop_pct=0.30,
    ),
    TierParameters(
        tier="operating",
        equity_min=25_000.0,
        equity_max=None,
        allowed_setups=[
            "crypto_mean_reversion",
            "equity_trend_pullback",
            "crypto_breakout",
            "crypto_momentum",
        ],
        max_positions=12,
        max_position_pct=0.10,
        cash_reserve_pct=0.20,
        max_daily_loss_pct=0.02,
        min_confidence=0.30,
        stop_loss_atr_multiple=2.5,
        profit_target_atr_multiple=3.0,
        min_setup_score_offset=5.0,
        crypto_regime_filter=["TRENDING_UP", "RANGING"],
        stale_hold_hours=72.0,
        trailing_stop_pct=0.25,
    ),
]


def get_tier(equity: float) -> CapitalTier:
    """Return the capital tier for the given equity value.

    Raises ``ValueError`` if equity is below the minimum tier threshold.
    """
    for params in TIER_TABLE:
        if params.equity_max is None:
            # Top tier — no upper bound
            if equity >= params.equity_min:
                return params.tier
        elif params.equity_min <= equity <= params.equity_max:
            return params.tier
    raise ValueError(
        f"Equity ${equity:,.2f} is below the minimum tier threshold "
        f"(${TIER_TABLE[0].equity_min:,.2f})"
    )


def get_tier_parameters(equity: float) -> TierParameters:
    """Return the full ``TierParameters`` for the given equity value.

    Raises ``ValueError`` if equity is below the minimum tier threshold.
    """
    for params in TIER_TABLE:
        if params.equity_max is None:
            if equity >= params.equity_min:
                return params
        elif params.equity_min <= equity <= params.equity_max:
            return params
    raise ValueError(
        f"Equity ${equity:,.2f} is below the minimum tier threshold "
        f"(${TIER_TABLE[0].equity_min:,.2f})"
    )


def detect_tier_transition(
    previous_tier: CapitalTier,
    current_equity: float,
) -> dict | None:
    """Detect whether equity has moved to a different tier.

    Returns a dict with transition details if the tier changed, or ``None``
    if the tier is the same.
    """
    current_tier = get_tier(current_equity)
    if current_tier == previous_tier:
        return None
    current_params = get_tier_parameters(current_equity)
    return {
        "from_tier": previous_tier,
        "to_tier": current_tier,
        "equity": current_equity,
        "new_max_position_pct": current_params.max_position_pct,
        "new_max_daily_loss_pct": current_params.max_daily_loss_pct,
        "new_max_positions": current_params.max_positions,
    }
