"""
test_exit_monitor.py — Comprehensive tests for exit_monitor.evaluate_exit().

Covers all 8 exit conditions plus edge cases identified in audit:
  - ATR stop vs hard stop precedence
  - Regime stop only exits losers (gain_pct < 0.0 fix)
  - exit_fraction propagation on profit target partials
  - profit_target_price sentinel values (-1, 0, positive)
  - ATR profit target
  - Entry price zero guard
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sauce.exit_monitor import ExitSignal, evaluate_exit
from sauce.strategy import ExitPlan, Position


# ── Helpers ───────────────────────────────────────────────────────────────────


def _plan(**overrides) -> ExitPlan:
    defaults = dict(
        stop_loss_pct=0.03,
        trail_activation_pct=0.03,
        trail_pct=0.02,
        profit_target_pct=0.06,
        rsi_exhaustion_threshold=72,
        max_hold_hours=48,
        time_stop_min_gain=0.01,
        profit_take_fraction=0.40,
        regime_stop=False,
    )
    defaults.update(overrides)
    return ExitPlan(**defaults)


def _pos(**overrides) -> Position:
    defaults = dict(
        symbol="BTC/USD",
        entry_price=100.0,
        qty=1.0,
        entry_time=datetime.now(UTC),
    )
    defaults.update(overrides)
    return Position(**defaults)


# ── 1a. Hard stop ─────────────────────────────────────────────────────────────


class TestHardStop:
    def test_triggers_below_stop(self):
        sig, _ = evaluate_exit(_pos(), _plan(stop_loss_pct=0.03), 96.5, rsi_14=50.0)
        assert sig is not None and sig.trigger == "hard_stop"

    def test_triggers_at_boundary(self):
        sig, _ = evaluate_exit(_pos(), _plan(stop_loss_pct=0.03), 97.0, rsi_14=50.0)
        assert sig is not None and sig.trigger == "hard_stop"

    def test_no_trigger_above_stop(self):
        sig, _ = evaluate_exit(_pos(), _plan(stop_loss_pct=0.03), 97.5, rsi_14=50.0)
        assert sig is None


# ── 1b. ATR stop ──────────────────────────────────────────────────────────────


class TestATRStop:
    def test_atr_tightens_when_more_conservative(self):
        # ATR stop = 100 - (0.5 * 2.0) = 99.0  vs  hard stop = 97.0
        sig, _ = evaluate_exit(
            _pos(), _plan(stop_loss_pct=0.03), 98.5, rsi_14=50.0,
            atr_14=0.5, atr_stop_multiple=2.0,
        )
        assert sig is not None and sig.trigger == "atr_stop"

    def test_hard_stop_wins_when_tighter(self):
        # ATR stop = 100 - (5.0 * 2.0) = 90.0  vs  hard stop = 97.0
        sig, _ = evaluate_exit(
            _pos(), _plan(stop_loss_pct=0.03), 96.5, rsi_14=50.0,
            atr_14=5.0, atr_stop_multiple=2.0,
        )
        assert sig is not None and sig.trigger == "hard_stop"

    def test_atr_none_falls_to_hard_stop(self):
        sig, _ = evaluate_exit(
            _pos(), _plan(stop_loss_pct=0.03), 96.5, rsi_14=50.0,
            atr_14=None,
        )
        assert sig is not None and sig.trigger == "hard_stop"

    def test_atr_zero_ignored(self):
        sig, _ = evaluate_exit(
            _pos(), _plan(stop_loss_pct=0.03), 96.5, rsi_14=50.0,
            atr_14=0.0,
        )
        assert sig is not None and sig.trigger == "hard_stop"


# ── 2. Trailing stop ─────────────────────────────────────────────────────────


class TestTrailingStop:
    def test_armed_trailing_triggers(self):
        pos = _pos(
            trailing_active=True,
            high_water_price=110.0,
            trailing_stop_price=107.8,
        )
        sig, _ = evaluate_exit(pos, _plan(), 107.0, rsi_14=50.0)
        assert sig is not None and sig.trigger == "trailing_stop"

    def test_unarmed_trailing_no_trigger(self):
        pos = _pos(trailing_active=False, trailing_stop_price=99.0)
        sig, _ = evaluate_exit(pos, _plan(), 98.0, rsi_14=50.0)
        assert sig is None  # hard stop at 97.0, so 98 is above it


# ── 3. Trail activation ──────────────────────────────────────────────────────


class TestTrailActivation:
    def test_activation_arms_trailing(self):
        sig, pos_out = evaluate_exit(
            _pos(), _plan(trail_activation_pct=0.03, trail_pct=0.02, profit_target_pct=0.20),
            103.5, rsi_14=50.0,
        )
        assert sig is None
        assert pos_out.trailing_active
        assert pos_out.trailing_stop_price == pytest.approx(103.5 * 0.98)

    def test_no_activation_below_threshold(self):
        sig, pos_out = evaluate_exit(
            _pos(), _plan(trail_activation_pct=0.03), 102.0, rsi_14=50.0,
        )
        assert not pos_out.trailing_active


# ── 4. Trail ratchet ─────────────────────────────────────────────────────────


class TestTrailRatchet:
    def test_new_high_tightens_stop(self):
        pos = _pos(
            trailing_active=True,
            high_water_price=105.0,
            trailing_stop_price=105.0 * 0.98,
        )
        sig, pos_out = evaluate_exit(
            pos, _plan(profit_target_pct=0.20), 108.0, rsi_14=50.0,
        )
        assert sig is None
        assert pos_out.high_water_price == 108.0
        assert pos_out.trailing_stop_price == pytest.approx(108.0 * 0.98)


# ── 5. Profit target ─────────────────────────────────────────────────────────


class TestProfitTarget:
    def test_triggers_above_target(self):
        sig, _ = evaluate_exit(_pos(), _plan(profit_target_pct=0.06), 106.5, rsi_14=50.0)
        assert sig is not None and sig.trigger == "profit_target_partial"

    def test_exit_fraction_propagated(self):
        sig, _ = evaluate_exit(
            _pos(), _plan(profit_target_pct=0.06, profit_take_fraction=0.50),
            106.5, rsi_14=50.0,
        )
        assert sig is not None
        assert sig.exit_fraction == pytest.approx(0.50)

    def test_default_exit_fraction(self):
        sig, _ = evaluate_exit(_pos(), _plan(profit_target_pct=0.06), 106.5, rsi_14=50.0)
        assert sig is not None
        assert sig.exit_fraction == pytest.approx(0.40)

    def test_negative_sentinel_disables_target(self):
        """profit_target_price = -1 means runner mode — no profit target exit."""
        pos = _pos(
            profit_target_price=-1.0,
            trailing_active=True,
            high_water_price=106.5,
            trailing_stop_price=104.37,
        )
        sig, _ = evaluate_exit(pos, _plan(profit_target_pct=0.06), 106.5, rsi_14=50.0)
        assert sig is None

    def test_zero_sentinel_uses_plan_pct(self):
        """profit_target_price = 0 falls through to plan.profit_target_pct."""
        pos = _pos(profit_target_price=0.0)
        sig, _ = evaluate_exit(pos, _plan(profit_target_pct=0.06), 106.5, rsi_14=50.0)
        assert sig is not None and sig.trigger == "profit_target_partial"

    def test_positive_sentinel_uses_stored_price(self):
        """profit_target_price > 0 is used directly."""
        pos = _pos(profit_target_price=105.0)
        sig, _ = evaluate_exit(pos, _plan(profit_target_pct=0.20), 105.5, rsi_14=50.0)
        assert sig is not None and sig.trigger == "profit_target_partial"


# ── 5b. ATR profit target ────────────────────────────────────────────────────


class TestATRTarget:
    def test_atr_target_triggers(self):
        # ATR target = 100 + (1.0 * 3.0) = 103.0
        sig, _ = evaluate_exit(
            _pos(), _plan(profit_target_pct=0.06), 103.5, rsi_14=50.0,
            atr_14=1.0, atr_target_multiple=3.0,
        )
        assert sig is not None and sig.trigger == "atr_target_partial"

    def test_atr_target_requires_positive_atr(self):
        sig, _ = evaluate_exit(
            _pos(), _plan(profit_target_pct=0.20), 103.5, rsi_14=50.0,
            atr_14=None,
        )
        assert sig is None


# ── 6. RSI exhaustion ─────────────────────────────────────────────────────────


class TestRSIExhaustion:
    def test_rsi_above_threshold(self):
        sig, _ = evaluate_exit(
            _pos(), _plan(rsi_exhaustion_threshold=72, profit_target_pct=0.20),
            103.0, rsi_14=75.0,
        )
        assert sig is not None and sig.trigger == "rsi_exhaustion"

    def test_rsi_none_no_trigger(self):
        sig, _ = evaluate_exit(
            _pos(), _plan(profit_target_pct=0.20), 103.0, rsi_14=None,
        )
        assert sig is None

    def test_rsi_below_threshold(self):
        sig, _ = evaluate_exit(
            _pos(), _plan(rsi_exhaustion_threshold=72, profit_target_pct=0.20),
            103.0, rsi_14=60.0,
        )
        assert sig is None


# ── 7. Time stop ──────────────────────────────────────────────────────────────


class TestTimeStop:
    def test_held_too_long_with_small_gain(self):
        entry_time = datetime.now(UTC) - timedelta(hours=50)
        sig, _ = evaluate_exit(
            _pos(entry_time=entry_time),
            _plan(max_hold_hours=48, time_stop_min_gain=0.01, profit_target_pct=0.20),
            100.5, rsi_14=50.0,
        )
        assert sig is not None and sig.trigger == "time_stop"

    def test_sufficient_gain_prevents_time_stop(self):
        entry_time = datetime.now(UTC) - timedelta(hours=50)
        sig, _ = evaluate_exit(
            _pos(entry_time=entry_time),
            _plan(max_hold_hours=48, time_stop_min_gain=0.01, profit_target_pct=0.20),
            102.0, rsi_14=50.0,
        )
        assert sig is None

    def test_within_hold_window_no_trigger(self):
        entry_time = datetime.now(UTC) - timedelta(hours=10)
        sig, _ = evaluate_exit(
            _pos(entry_time=entry_time),
            _plan(max_hold_hours=48, time_stop_min_gain=0.01),
            100.5, rsi_14=50.0,
        )
        assert sig is None


# ── 8. Regime stop ────────────────────────────────────────────────────────────


class TestRegimeStop:
    def test_bearish_exits_loser(self):
        """Regime bearish + losing position → exit."""
        sig, _ = evaluate_exit(
            _pos(),
            _plan(regime_stop=True, profit_target_pct=0.20),
            99.0, rsi_14=50.0, regime="bearish",
        )
        assert sig is not None and sig.trigger == "regime_stop"

    def test_bearish_spares_winner(self):
        """Regime bearish but position is profitable → no regime exit (the fix)."""
        sig, _ = evaluate_exit(
            _pos(),
            _plan(regime_stop=True, profit_target_pct=0.20),
            101.0, rsi_14=50.0, regime="bearish",
        )
        assert sig is None

    def test_bearish_spares_breakeven(self):
        """gain_pct == 0 is NOT < 0 → no regime exit."""
        sig, _ = evaluate_exit(
            _pos(),
            _plan(regime_stop=True, profit_target_pct=0.20),
            100.0, rsi_14=50.0, regime="bearish",
        )
        assert sig is None

    def test_regime_stop_disabled(self):
        """regime_stop=False → no regime exit even if bearish and losing."""
        sig, _ = evaluate_exit(
            _pos(),
            _plan(regime_stop=False, profit_target_pct=0.20),
            99.0, rsi_14=50.0, regime="bearish",
        )
        assert sig is None

    def test_non_bearish_regime_no_trigger(self):
        sig, _ = evaluate_exit(
            _pos(),
            _plan(regime_stop=True, profit_target_pct=0.20),
            99.0, rsi_14=50.0, regime="bullish",
        )
        assert sig is None

    def test_none_regime_no_trigger(self):
        sig, _ = evaluate_exit(
            _pos(),
            _plan(regime_stop=True, profit_target_pct=0.20),
            99.0, rsi_14=50.0, regime=None,
        )
        assert sig is None


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_entry_price_zero_no_crash(self):
        """entry_price=0 must not divide-by-zero in gain_pct calculations."""
        pos = _pos(entry_price=0.0)
        plan = _plan(profit_target_pct=0.20, max_hold_hours=1, time_stop_min_gain=0.01)
        entry_time = datetime.now(UTC) - timedelta(hours=2)
        pos = _pos(entry_price=0.0, entry_time=entry_time)
        sig, _ = evaluate_exit(pos, plan, 50.0, rsi_14=50.0)
        # Should not crash — hard stop at 0*(1-0.03)=0 which is 50>0 so no stop
        # Time stop: gain_pct = 0 (guarded), < min_gain → triggers
        assert sig is not None and sig.trigger == "time_stop"

    def test_exit_signal_fields(self):
        sig, _ = evaluate_exit(_pos(), _plan(stop_loss_pct=0.03), 96.5, rsi_14=50.0)
        assert sig is not None
        assert sig.symbol == "BTC/USD"
        assert sig.side == "sell"
        assert sig.current_price == 96.5
        assert sig.exit_fraction == 1.0  # default for non-partial exits

    def test_mid_range_no_exit(self):
        sig, _ = evaluate_exit(_pos(), _plan(), 101.0, rsi_14=50.0)
        assert sig is None
