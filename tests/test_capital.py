"""
tests/test_capital.py — Tests for core/capital.py.

Covers the capital tier system per Sprint 5:
- get_tier()        — correct tier for each equity range + boundary values
- get_tier_parameters() — returns correct TierParameters for each tier
- detect_tier_transition() — returns transition dict or None
- Error handling    — ValueError for equity below minimum
"""

import pytest

from sauce.core.capital import (
    TIER_TABLE,
    TierParameters,
    detect_tier_transition,
    get_tier,
    get_tier_parameters,
)


# ── get_tier — maps equity to tier name ──────────────────────────────────────


class TestGetTier:

    def test_seed_minimum(self):
        assert get_tier(500.0) == "seed"

    def test_seed_mid(self):
        assert get_tier(1_500.0) == "seed"

    def test_seed_upper_boundary(self):
        assert get_tier(2_499.99) == "seed"

    def test_building_lower_boundary(self):
        assert get_tier(2_500.0) == "building"

    def test_building_upper_boundary(self):
        assert get_tier(4_999.99) == "building"

    def test_growing_lower_boundary(self):
        assert get_tier(5_000.0) == "growing"

    def test_growing_upper_boundary(self):
        assert get_tier(9_999.99) == "growing"

    def test_scaling_lower_boundary(self):
        assert get_tier(10_000.0) == "scaling"

    def test_scaling_upper_boundary(self):
        assert get_tier(24_999.99) == "scaling"

    def test_operating_lower_boundary(self):
        assert get_tier(25_000.0) == "operating"

    def test_operating_large_equity(self):
        assert get_tier(1_000_000.0) == "operating"

    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match="below the minimum tier threshold"):
            get_tier(499.99)

    def test_zero_equity_raises(self):
        with pytest.raises(ValueError, match="below the minimum tier threshold"):
            get_tier(0.0)

    def test_negative_equity_raises(self):
        with pytest.raises(ValueError, match="below the minimum tier threshold"):
            get_tier(-100.0)


# ── get_tier_parameters — returns full TierParameters ────────────────────────


class TestGetTierParameters:

    def test_returns_tier_parameters_type(self):
        result = get_tier_parameters(500.0)
        assert isinstance(result, TierParameters)

    def test_seed_parameters(self):
        p = get_tier_parameters(1_000.0)
        assert p.tier == "seed"
        assert p.max_positions == 20
        assert p.max_position_pct == 0.50
        assert p.max_daily_loss_pct == 0.08
        assert p.cash_reserve_pct == 0.10
        assert set(p.allowed_setups) == {"crypto_mean_reversion", "equity_trend_pullback", "crypto_momentum"}
        assert p.min_confidence == 0.20
        assert p.stop_loss_atr_multiple == 1.5
        assert p.profit_target_atr_multiple == 4.0
        assert p.min_setup_score_offset == -10.0
        assert p.stale_hold_hours == 24.0
        assert p.trailing_stop_pct == 0.40

    def test_building_parameters(self):
        p = get_tier_parameters(3_000.0)
        assert p.tier == "building"
        assert p.max_positions == 10
        assert p.max_position_pct == 0.35
        assert p.max_daily_loss_pct == 0.05
        assert p.min_confidence == 0.25
        assert p.stop_loss_atr_multiple == 1.5
        assert p.min_setup_score_offset == -5.0

    def test_growing_parameters(self):
        p = get_tier_parameters(7_000.0)
        assert p.tier == "growing"
        assert p.max_positions == 6
        assert p.max_position_pct == 0.25
        assert p.max_daily_loss_pct == 0.03
        assert p.min_confidence == 0.25
        assert p.min_setup_score_offset == 0.0
        assert p.crypto_regime_filter == ["TRENDING_UP", "RANGING"]

    def test_scaling_parameters(self):
        p = get_tier_parameters(15_000.0)
        assert p.tier == "scaling"
        assert p.max_positions == 8
        assert p.max_position_pct == 0.15
        assert p.max_daily_loss_pct == 0.02
        assert p.min_confidence == 0.30

    def test_operating_parameters(self):
        p = get_tier_parameters(50_000.0)
        assert p.tier == "operating"
        assert p.max_positions == 12
        assert p.max_position_pct == 0.10
        assert p.max_daily_loss_pct == 0.02
        assert p.min_confidence == 0.30
        assert p.trailing_stop_pct == 0.25
        assert p.stale_hold_hours == 72.0
        assert p.min_setup_score_offset == 5.0

    def test_operating_has_all_setups(self):
        p = get_tier_parameters(50_000.0)
        assert "crypto_mean_reversion" in p.allowed_setups
        assert "equity_trend_pullback" in p.allowed_setups
        assert "crypto_breakout" in p.allowed_setups
        assert "crypto_momentum" in p.allowed_setups

    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match="below the minimum tier threshold"):
            get_tier_parameters(100.0)

    def test_equity_min_max_correct(self):
        p = get_tier_parameters(1_000.0)
        assert p.equity_min == 500.0
        assert p.equity_max == 2_499.99

    def test_operating_has_no_upper_bound(self):
        p = get_tier_parameters(25_000.0)
        assert p.equity_max is None


# ── detect_tier_transition — returns dict or None ────────────────────────────


class TestDetectTierTransition:

    def test_no_transition_same_tier(self):
        result = detect_tier_transition("seed", 1_000.0)
        assert result is None

    def test_transition_seed_to_building(self):
        result = detect_tier_transition("seed", 2_500.0)
        assert result is not None
        assert result["from_tier"] == "seed"
        assert result["to_tier"] == "building"
        assert result["equity"] == 2_500.0
        assert result["new_max_position_pct"] == 0.35
        assert result["new_max_daily_loss_pct"] == 0.05
        assert result["new_max_positions"] == 10

    def test_transition_building_to_growing(self):
        result = detect_tier_transition("building", 5_000.0)
        assert result is not None
        assert result["from_tier"] == "building"
        assert result["to_tier"] == "growing"

    def test_transition_growing_to_scaling(self):
        result = detect_tier_transition("growing", 10_000.0)
        assert result is not None
        assert result["from_tier"] == "growing"
        assert result["to_tier"] == "scaling"

    def test_transition_scaling_to_operating(self):
        result = detect_tier_transition("scaling", 25_000.0)
        assert result is not None
        assert result["from_tier"] == "scaling"
        assert result["to_tier"] == "operating"

    def test_transition_downward(self):
        """Equity dropped from building back to seed."""
        result = detect_tier_transition("building", 1_000.0)
        assert result is not None
        assert result["from_tier"] == "building"
        assert result["to_tier"] == "seed"

    def test_transition_skip_tier(self):
        """Jump from seed straight to growing."""
        result = detect_tier_transition("seed", 7_500.0)
        assert result is not None
        assert result["from_tier"] == "seed"
        assert result["to_tier"] == "growing"

    def test_transition_below_minimum_raises(self):
        with pytest.raises(ValueError, match="below the minimum tier threshold"):
            detect_tier_transition("seed", 100.0)


# ── TIER_TABLE integrity ─────────────────────────────────────────────────────


class TestTierTable:

    def test_five_tiers_defined(self):
        assert len(TIER_TABLE) == 5

    def test_tier_names(self):
        names = [t.tier for t in TIER_TABLE]
        assert names == ["seed", "building", "growing", "scaling", "operating"]

    def test_ascending_equity_min(self):
        mins = [t.equity_min for t in TIER_TABLE]
        assert mins == sorted(mins)

    def test_boundaries_contiguous(self):
        """Upper bound of tier N + 0.01 == lower bound of tier N+1."""
        for i in range(len(TIER_TABLE) - 1):
            assert TIER_TABLE[i].equity_max is not None
            gap = TIER_TABLE[i + 1].equity_min - TIER_TABLE[i].equity_max
            assert abs(gap - 0.01) < 0.001

    def test_only_top_tier_has_no_upper(self):
        for t in TIER_TABLE[:-1]:
            assert t.equity_max is not None
        assert TIER_TABLE[-1].equity_max is None


class TestTierParametersSafetyFloors:
    """Verify model_validator clamps dangerous values to safe floors."""

    def test_min_confidence_floor(self):
        p = TierParameters(
            tier="seed", equity_min=500.0, equity_max=2_499.99,
            allowed_setups=["crypto_mean_reversion"],
            max_positions=2, max_position_pct=0.50, cash_reserve_pct=0.10,
            max_daily_loss_pct=0.05, min_confidence=0.05,
        )
        assert p.min_confidence == 0.15

    def test_stop_loss_floor(self):
        p = TierParameters(
            tier="seed", equity_min=500.0, equity_max=2_499.99,
            allowed_setups=["crypto_mean_reversion"],
            max_positions=2, max_position_pct=0.50, cash_reserve_pct=0.10,
            max_daily_loss_pct=0.05, stop_loss_atr_multiple=0.5,
        )
        assert p.stop_loss_atr_multiple == 1.0

    def test_score_offset_floor(self):
        p = TierParameters(
            tier="seed", equity_min=500.0, equity_max=2_499.99,
            allowed_setups=["crypto_mean_reversion"],
            max_positions=2, max_position_pct=0.50, cash_reserve_pct=0.10,
            max_daily_loss_pct=0.05, min_setup_score_offset=-20.0,
        )
        assert p.min_setup_score_offset == -15.0

    def test_max_daily_loss_ceiling(self):
        p = TierParameters(
            tier="seed", equity_min=500.0, equity_max=2_499.99,
            allowed_setups=["crypto_mean_reversion"],
            max_positions=2, max_position_pct=0.50, cash_reserve_pct=0.10,
            max_daily_loss_pct=0.15,
        )
        assert p.max_daily_loss_pct == 0.10

    def test_valid_values_unchanged(self):
        p = TierParameters(
            tier="seed", equity_min=500.0, equity_max=2_499.99,
            allowed_setups=["crypto_mean_reversion"],
            max_positions=2, max_position_pct=0.50, cash_reserve_pct=0.10,
            max_daily_loss_pct=0.05, min_confidence=0.30,
            stop_loss_atr_multiple=2.0, min_setup_score_offset=-5.0,
        )
        assert p.min_confidence == 0.30
        assert p.stop_loss_atr_multiple == 2.0
        assert p.min_setup_score_offset == -5.0
        assert p.max_daily_loss_pct == 0.05
