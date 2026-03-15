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


TIER_TABLE: list[TierParameters] = [
    TierParameters(
        tier="seed",
        equity_min=500.0,
        equity_max=2_499.99,
        allowed_setups=["crypto_mean_reversion"],
        max_positions=1,
        max_position_pct=0.35,
        cash_reserve_pct=0.25,
        max_daily_loss_pct=0.03,
    ),
    TierParameters(
        tier="building",
        equity_min=2_500.0,
        equity_max=4_999.99,
        allowed_setups=["crypto_mean_reversion", "equity_trend_pullback"],
        max_positions=2,
        max_position_pct=0.25,
        cash_reserve_pct=0.25,
        max_daily_loss_pct=0.03,
    ),
    TierParameters(
        tier="growing",
        equity_min=5_000.0,
        equity_max=9_999.99,
        allowed_setups=[
            "crypto_mean_reversion",
            "equity_trend_pullback",
            "crypto_breakout",
        ],
        max_positions=3,
        max_position_pct=0.18,
        cash_reserve_pct=0.25,
        max_daily_loss_pct=0.025,
    ),
    TierParameters(
        tier="scaling",
        equity_min=10_000.0,
        equity_max=24_999.99,
        allowed_setups=[
            "crypto_mean_reversion",
            "equity_trend_pullback",
            "crypto_breakout",
        ],
        max_positions=5,
        max_position_pct=0.12,
        cash_reserve_pct=0.20,
        max_daily_loss_pct=0.02,
    ),
    TierParameters(
        tier="operating",
        equity_min=25_000.0,
        equity_max=None,
        allowed_setups=[
            "crypto_mean_reversion",
            "equity_trend_pullback",
            "crypto_breakout",
        ],
        max_positions=8,
        max_position_pct=0.10,
        cash_reserve_pct=0.20,
        max_daily_loss_pct=0.02,
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
