"""
test_risk.py — Comprehensive tests for risk.check_risk() and check_consecutive_loss_circuit().

Covers all 5 risk rules plus edge cases identified in audit:
  - equity=0 rejection (the fix from prior session)
  - negative equity handling
  - boundary conditions for each rule
  - consecutive loss circuit breaker
"""

from __future__ import annotations

import pytest

from sauce.risk import RiskVerdict, check_consecutive_loss_circuit, check_risk


def _passing_kwargs(**overrides) -> dict:
    """Base kwargs that pass all 5 rules."""
    defaults = dict(
        daily_pnl=0.0,
        equity=10_000,
        open_position_count=0,
        buying_power=5_000,
        order_value=1_000,
        daily_loss_limit=0.08,
        max_concurrent=3,
        total_existing_exposure=0,
        max_portfolio_exposure=0.80,
    )
    defaults.update(overrides)
    return defaults


# ── Rule 1: Daily P&L gate ────────────────────────────────────────────────────


class TestDailyPnl:
    def test_within_limit_passes(self):
        v = check_risk(**_passing_kwargs(daily_pnl=-0.05))
        assert v.passed

    def test_beyond_limit_fails(self):
        v = check_risk(**_passing_kwargs(daily_pnl=-0.09))
        assert not v.passed and v.rule == "daily_pnl"

    def test_at_limit_boundary_fails(self):
        """Exactly at daily_loss_limit → fail (<= check)."""
        v = check_risk(**_passing_kwargs(daily_pnl=-0.08))
        assert not v.passed and v.rule == "daily_pnl"

    def test_positive_pnl_passes(self):
        v = check_risk(**_passing_kwargs(daily_pnl=0.05))
        assert v.passed

    def test_zero_pnl_passes(self):
        v = check_risk(**_passing_kwargs(daily_pnl=0.0))
        assert v.passed


# ── Rule 2: Position count gate ───────────────────────────────────────────────


class TestPositionCount:
    def test_below_max_passes(self):
        v = check_risk(**_passing_kwargs(open_position_count=1, max_concurrent=3))
        assert v.passed

    def test_at_max_fails(self):
        v = check_risk(**_passing_kwargs(open_position_count=3, max_concurrent=3))
        assert not v.passed and v.rule == "position_count"

    def test_above_max_fails(self):
        v = check_risk(**_passing_kwargs(open_position_count=5, max_concurrent=3))
        assert not v.passed and v.rule == "position_count"


# ── Rule 3: Buying power gate ─────────────────────────────────────────────────


class TestBuyingPower:
    def test_sufficient_passes(self):
        v = check_risk(**_passing_kwargs(buying_power=5000, order_value=1000))
        assert v.passed

    def test_exact_match_passes(self):
        v = check_risk(**_passing_kwargs(buying_power=1000, order_value=1000))
        assert v.passed

    def test_insufficient_fails(self):
        v = check_risk(**_passing_kwargs(buying_power=500, order_value=1000))
        assert not v.passed and v.rule == "buying_power"


# ── Rule 4: Minimum order size ────────────────────────────────────────────────


class TestMinOrderSize:
    def test_above_minimum_passes(self):
        v = check_risk(**_passing_kwargs(order_value=10.0))
        assert v.passed

    def test_dust_trade_fails(self):
        v = check_risk(**_passing_kwargs(order_value=0.50))
        assert not v.passed and v.rule == "min_order_size"

    def test_exactly_one_dollar_passes(self):
        v = check_risk(**_passing_kwargs(order_value=1.0))
        assert v.passed


# ── Rule 5: Max portfolio exposure (the audit fix) ───────────────────────────


class TestMaxExposure:
    def test_within_limit_passes(self):
        v = check_risk(**_passing_kwargs(
            equity=10_000, total_existing_exposure=5_000, order_value=1_000,
        ))
        assert v.passed

    def test_over_limit_fails(self):
        v = check_risk(**_passing_kwargs(
            equity=1_000, total_existing_exposure=650, order_value=200,
        ))
        assert not v.passed and v.rule == "max_exposure"

    def test_zero_equity_rejects(self):
        """The audit fix: equity=0 must reject, not silently skip."""
        v = check_risk(**_passing_kwargs(equity=0))
        assert not v.passed
        assert v.rule == "max_exposure"
        assert "zero or negative" in v.reason.lower()

    def test_negative_equity_rejects(self):
        v = check_risk(**_passing_kwargs(equity=-500))
        assert not v.passed
        assert v.rule == "max_exposure"

    def test_at_boundary_passes(self):
        """Exposure exactly at 80% → passes (not >)."""
        v = check_risk(**_passing_kwargs(
            equity=10_000, total_existing_exposure=7_000, order_value=1_000,
            max_portfolio_exposure=0.80,
        ))
        assert v.passed

    def test_just_over_boundary_fails(self):
        v = check_risk(**_passing_kwargs(
            equity=10_000, total_existing_exposure=7_000, order_value=1_001,
            max_portfolio_exposure=0.80,
        ))
        assert not v.passed and v.rule == "max_exposure"


# ── Rule priority (first failure wins) ────────────────────────────────────────


class TestRulePriority:
    def test_first_failure_wins(self):
        """When multiple rules fail, daily_pnl (Rule 1) is reported."""
        v = check_risk(
            daily_pnl=-0.10,
            equity=10_000,
            open_position_count=5,
            buying_power=0,
            order_value=1_000,
            daily_loss_limit=0.08,
            max_concurrent=2,
            total_existing_exposure=9_000,
            max_portfolio_exposure=0.80,
        )
        assert not v.passed and v.rule == "daily_pnl"

    def test_all_pass_returns_rule_all(self):
        v = check_risk(**_passing_kwargs())
        assert v.passed and v.rule == "all" and v.reason == ""


# ── Consecutive loss circuit breaker ──────────────────────────────────────────


class TestConsecutiveLossCircuit:
    def test_trips_on_all_losses(self):
        v = check_consecutive_loss_circuit([-100.0, -25.0, -5.0], 3)
        assert not v.passed and v.rule == "consecutive_losses"

    def test_interrupted_streak_passes(self):
        v = check_consecutive_loss_circuit([-100.0, 10.0, -5.0], 3)
        assert v.passed

    def test_too_few_trades_passes(self):
        v = check_consecutive_loss_circuit([-100.0, -50.0], 3)
        assert v.passed

    def test_empty_list_passes(self):
        v = check_consecutive_loss_circuit([], 3)
        assert v.passed

    def test_zero_max_always_passes(self):
        v = check_consecutive_loss_circuit([-100.0, -50.0, -25.0], 0)
        assert v.passed

    def test_single_loss_threshold(self):
        v = check_consecutive_loss_circuit([-100.0, 50.0], 1)
        assert not v.passed

    def test_reason_includes_total(self):
        v = check_consecutive_loss_circuit([-100.0, -50.0], 2)
        assert not v.passed
        assert "$-150.00" in v.reason
