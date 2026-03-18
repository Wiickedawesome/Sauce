"""
test_engine.py — Tests for Sauce core components.

Covers:
  - Signal scoring (CryptoMomentumReversion)
  - Risk gate (3 rules)
  - Exit monitor (7 conditions)
  - Morning brief (fallback behaviour)
  - Database (CRUD operations)
  - Strategy protocol & tier table
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from sauce.core.schemas import Indicators, Order
from sauce.exit_monitor import ExitSignal, evaluate_exit
from sauce.risk import RiskVerdict, check_risk
from sauce.strategies.crypto_momentum import CryptoMomentumReversion, _compute_bb_pct
from sauce.strategy import (
    BUILDING_PARAMS,
    OPERATING_PARAMS,
    SCALING_PARAMS,
    SEED_PARAMS,
    ExitPlan,
    Position,
    SignalResult,
    TierParams,
    get_tier_params,
)


# ══════════════════════════════════════════════════════════════════════════════
# Tier Table
# ══════════════════════════════════════════════════════════════════════════════


class TestTierTable:
    def test_seed_tier(self):
        tier = get_tier_params(1_000.0)
        assert tier.tier == "seed"
        assert tier.max_concurrent == 2
        assert tier.max_position_pct == 0.30

    def test_building_tier(self):
        tier = get_tier_params(25_000.0)
        assert tier.tier == "building"
        assert tier.max_concurrent == 4
        assert tier.max_position_pct == 0.15

    def test_scaling_tier(self):
        tier = get_tier_params(75_000.0)
        assert tier.tier == "scaling"
        assert tier.max_concurrent == 8

    def test_operating_tier(self):
        tier = get_tier_params(150_000.0)
        assert tier.tier == "operating"
        assert tier.max_concurrent == 12
        assert tier.max_position_pct == 0.05

    def test_boundary_building(self):
        """Exactly $10K should be building tier."""
        tier = get_tier_params(10_000.0)
        assert tier.tier == "building"

    def test_boundary_scaling(self):
        """Exactly $50K should be scaling tier."""
        tier = get_tier_params(50_000.0)
        assert tier.tier == "scaling"

    def test_boundary_operating(self):
        """Exactly $100K should be operating tier."""
        tier = get_tier_params(100_000.0)
        assert tier.tier == "operating"


# ══════════════════════════════════════════════════════════════════════════════
# Signal Scoring — CryptoMomentumReversion
# ══════════════════════════════════════════════════════════════════════════════


def _make_indicators(**overrides) -> Indicators:
    """Build an Indicators instance with sensible defaults."""
    defaults = dict(
        sma_20=100.0,
        sma_50=98.0,
        rsi_14=50.0,
        atr_14=2.5,
        volume_ratio=1.0,
        macd_line=0.0,
        macd_signal=0.0,
        macd_histogram=0.0,
        bb_upper=110.0,
        bb_middle=100.0,
        bb_lower=90.0,
    )
    defaults.update(overrides)
    return Indicators(**defaults)


class TestCryptoMomentumReversion:
    strategy = CryptoMomentumReversion()

    def test_eligible_btc(self):
        assert self.strategy.eligible("BTC/USD", "neutral")

    def test_eligible_eth(self):
        assert self.strategy.eligible("ETH/USD", "bearish")

    def test_not_eligible_random(self):
        assert not self.strategy.eligible("AAPL", "neutral")

    def test_no_conditions_met(self):
        """No scoring conditions met → score 0, not fired."""
        ind = _make_indicators(rsi_14=50, volume_ratio=1.0, macd_histogram=-1.0)
        result = self.strategy.score(ind, "BTC/USD", "neutral", 100.0)
        assert result.score == 0
        assert not result.fired
        assert result.side == "hold"

    def test_rsi_oversold_only(self):
        """RSI < 35 → 30 pts, below threshold → not fired."""
        ind = _make_indicators(rsi_14=30, volume_ratio=1.0, macd_histogram=-1.0)
        result = self.strategy.score(ind, "BTC/USD", "neutral", 100.0)
        assert result.score == 30
        assert not result.fired

    def test_all_conditions_met(self):
        """All 4 conditions → 100 pts, definitely fired."""
        ind = _make_indicators(
            rsi_14=20,
            bb_upper=110.0,
            bb_lower=95.0,
            macd_histogram=0.5,
            volume_ratio=2.0,
        )
        # Price at lower band → bb_pct ≤ 0
        result = self.strategy.score(ind, "ETH/USD", "neutral", 90.0)
        assert result.score == 100
        assert result.fired
        assert result.side == "buy"

    def test_regime_bullish_lowers_threshold(self):
        """Bullish regime → threshold 55, easier to fire."""
        ind = _make_indicators(
            rsi_14=30,          # 30 pts
            macd_histogram=0.5,  # 25 pts → total 55
            volume_ratio=1.0,
        )
        result = self.strategy.score(ind, "BTC/USD", "bullish", 100.0)
        assert result.threshold == 55
        assert result.score == 55
        assert result.fired

    def test_regime_bearish_raises_threshold(self):
        """Bearish regime → threshold 75, harder to fire."""
        ind = _make_indicators(
            rsi_14=30,          # 30 pts
            macd_histogram=0.5,  # 25 pts → total 55
            volume_ratio=1.0,
        )
        result = self.strategy.score(ind, "BTC/USD", "bearish", 100.0)
        assert result.threshold == 75
        assert result.score == 55
        assert not result.fired

    def test_bb_pct_at_lower(self):
        """Price exactly at lower BB → bb_pct 0.0 → 25 pts."""
        ind = _make_indicators(bb_lower=90.0, bb_upper=110.0)
        pct = _compute_bb_pct(ind, 90.0)
        assert pct == pytest.approx(0.0)

    def test_bb_pct_below_lower(self):
        """Price below lower BB → bb_pct < 0."""
        ind = _make_indicators(bb_lower=90.0, bb_upper=110.0)
        pct = _compute_bb_pct(ind, 85.0)
        assert pct < 0.0

    def test_bb_pct_at_upper(self):
        """Price at upper BB → bb_pct 1.0."""
        ind = _make_indicators(bb_lower=90.0, bb_upper=110.0)
        pct = _compute_bb_pct(ind, 110.0)
        assert pct == pytest.approx(1.0)

    def test_bb_pct_none_when_no_bands(self):
        ind = _make_indicators(bb_lower=None, bb_upper=None)
        assert _compute_bb_pct(ind, 100.0) is None

    def test_signal_result_metadata(self):
        """SignalResult includes all required metadata fields."""
        ind = _make_indicators(rsi_14=25, macd_histogram=0.3, volume_ratio=2.0)
        result = self.strategy.score(ind, "SOL/USD", "neutral", 100.0)
        assert result.symbol == "SOL/USD"
        assert result.regime == "neutral"
        assert result.strategy_name == "crypto_momentum_reversion"
        assert result.rsi_14 == 25
        assert result.volume_ratio == 2.0

    def test_build_order_sizing(self):
        """Order size = tier.max_position_pct × equity / ask."""
        signal = SignalResult(
            symbol="BTC/USD", side="buy", score=80, threshold=65,
            fired=True, rsi_14=30, macd_hist=0.5, bb_pct=-0.1,
            volume_ratio=2.0, regime="neutral",
            strategy_name="crypto_momentum_reversion",
        )
        account = {"equity": "10000", "_ask": "50000"}
        order = self.strategy.build_order(signal, account, SEED_PARAMS)
        # 30% of $10K = $3K / $50K ask = 0.06 BTC
        assert order.qty == pytest.approx(0.06, rel=0.01)
        assert order.side == "buy"
        assert order.order_type == "limit"
        assert order.limit_price == pytest.approx(50000 * 1.001, rel=0.001)

    def test_build_exit_plan(self):
        pos = Position(symbol="BTC/USD", entry_price=50000)
        plan = self.strategy.build_exit_plan(pos, SEED_PARAMS)
        assert plan.stop_loss_pct == 0.03
        assert plan.trail_activation_pct == 0.03
        assert plan.profit_target_pct == 0.06


# ══════════════════════════════════════════════════════════════════════════════
# Risk Gate
# ══════════════════════════════════════════════════════════════════════════════


class TestRiskGate:
    def test_all_pass(self):
        v = check_risk(
            daily_pnl=0.0, equity=10000, open_position_count=1,
            buying_power=5000, order_value=1000,
            daily_loss_limit=0.08, max_concurrent=2,
        )
        assert v.passed
        assert v.rule == "all"

    def test_daily_pnl_fail(self):
        v = check_risk(
            daily_pnl=-0.09, equity=10000, open_position_count=0,
            buying_power=5000, order_value=1000,
            daily_loss_limit=0.08, max_concurrent=2,
        )
        assert not v.passed
        assert v.rule == "daily_pnl"

    def test_daily_pnl_at_limit(self):
        """Exactly at the limit → fail (<=)."""
        v = check_risk(
            daily_pnl=-0.08, equity=10000, open_position_count=0,
            buying_power=5000, order_value=1000,
            daily_loss_limit=0.08, max_concurrent=2,
        )
        assert not v.passed
        assert v.rule == "daily_pnl"

    def test_position_count_fail(self):
        v = check_risk(
            daily_pnl=0.0, equity=10000, open_position_count=2,
            buying_power=5000, order_value=1000,
            daily_loss_limit=0.08, max_concurrent=2,
        )
        assert not v.passed
        assert v.rule == "position_count"

    def test_buying_power_fail(self):
        v = check_risk(
            daily_pnl=0.0, equity=10000, open_position_count=0,
            buying_power=500, order_value=1000,
            daily_loss_limit=0.08, max_concurrent=2,
        )
        assert not v.passed
        assert v.rule == "buying_power"

    def test_first_failure_wins(self):
        """When multiple rules fail, the first one (daily_pnl) is reported."""
        v = check_risk(
            daily_pnl=-0.10, equity=10000, open_position_count=5,
            buying_power=0, order_value=1000,
            daily_loss_limit=0.08, max_concurrent=2,
        )
        assert not v.passed
        assert v.rule == "daily_pnl"


# ══════════════════════════════════════════════════════════════════════════════
# Exit Monitor
# ══════════════════════════════════════════════════════════════════════════════


def _make_plan(**overrides) -> ExitPlan:
    defaults = dict(
        stop_loss_pct=0.03,
        trail_activation_pct=0.03,
        trail_pct=0.02,
        profit_target_pct=0.06,
        rsi_exhaustion_threshold=72,
        max_hold_hours=48,
        time_stop_min_gain=0.01,
    )
    defaults.update(overrides)
    return ExitPlan(**defaults)


def _make_position(**overrides) -> Position:
    defaults = dict(
        id="test-pos",
        symbol="BTC/USD",
        entry_price=100.0,
        qty=1.0,
        entry_time=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return Position(**defaults)


class TestExitMonitor:
    def test_no_exit(self):
        """Mid-range price, no conditions met."""
        pos = _make_position(entry_price=100.0)
        plan = _make_plan()
        sig, pos_out = evaluate_exit(pos, plan, 101.0, rsi_14=50.0)
        assert sig is None

    def test_hard_stop(self):
        """Price drops below entry × (1−0.03) = 97."""
        pos = _make_position(entry_price=100.0)
        plan = _make_plan(stop_loss_pct=0.03)
        sig, _ = evaluate_exit(pos, plan, 96.5, rsi_14=50.0)
        assert sig is not None
        assert sig.trigger == "hard_stop"

    def test_hard_stop_at_boundary(self):
        """Price exactly at stop → triggers (<=)."""
        pos = _make_position(entry_price=100.0)
        plan = _make_plan(stop_loss_pct=0.03)
        sig, _ = evaluate_exit(pos, plan, 97.0, rsi_14=50.0)
        assert sig is not None
        assert sig.trigger == "hard_stop"

    def test_trailing_stop_triggers(self):
        """Trailing stop is already armed and price drops below it."""
        pos = _make_position(
            entry_price=100.0,
            trailing_active=True,
            high_water_price=110.0,
            trailing_stop_price=107.8,  # 110 * (1 - 0.02)
        )
        plan = _make_plan()
        sig, _ = evaluate_exit(pos, plan, 107.0, rsi_14=50.0)
        assert sig is not None
        assert sig.trigger == "trailing_stop"

    def test_trail_activation(self):
        """Price rises 3% above entry → trailing gets armed."""
        pos = _make_position(entry_price=100.0)
        plan = _make_plan(trail_activation_pct=0.03, trail_pct=0.02)
        sig, pos_out = evaluate_exit(pos, plan, 103.5, rsi_14=50.0)
        assert sig is None  # no exit yet
        assert pos_out.trailing_active
        assert pos_out.high_water_price == 103.5
        assert pos_out.trailing_stop_price == pytest.approx(103.5 * 0.98)

    def test_trail_ratchet(self):
        """New high water mark tightens the trailing stop."""
        pos = _make_position(
            entry_price=100.0,
            trailing_active=True,
            high_water_price=105.0,
            trailing_stop_price=105.0 * 0.98,
        )
        plan = _make_plan()
        sig, pos_out = evaluate_exit(pos, plan, 108.0, rsi_14=50.0)
        # 108 > 105 (old high) → ratchet up. But 108 >= 106 (profit target) → hits profit target
        # Let's use a higher profit target to isolate the ratchet test
        plan_high_target = _make_plan(profit_target_pct=0.20)
        sig, pos_out = evaluate_exit(pos, plan_high_target, 108.0, rsi_14=50.0)
        assert sig is None
        assert pos_out.high_water_price == 108.0
        assert pos_out.trailing_stop_price == pytest.approx(108.0 * 0.98)

    def test_profit_target(self):
        """Price hits +6% → profit target exit."""
        pos = _make_position(entry_price=100.0)
        plan = _make_plan(profit_target_pct=0.06)
        sig, _ = evaluate_exit(pos, plan, 106.5, rsi_14=50.0)
        assert sig is not None
        assert sig.trigger == "profit_target"

    def test_rsi_exhaustion(self):
        """RSI 75 ≥ threshold 72 → exhaustion exit."""
        pos = _make_position(entry_price=100.0)
        plan = _make_plan(rsi_exhaustion_threshold=72, profit_target_pct=0.20)
        sig, _ = evaluate_exit(pos, plan, 103.0, rsi_14=75.0)
        assert sig is not None
        assert sig.trigger == "rsi_exhaustion"

    def test_rsi_none_no_exit(self):
        """RSI is None → RSI exhaustion doesn't fire."""
        pos = _make_position(entry_price=100.0)
        plan = _make_plan(profit_target_pct=0.20)
        sig, _ = evaluate_exit(pos, plan, 103.0, rsi_14=None)
        assert sig is None

    def test_time_stop(self):
        """Held 50 hours with 0.5% gain < 1% min → time stop."""
        entry_time = datetime.now(timezone.utc) - timedelta(hours=50)
        pos = _make_position(entry_price=100.0, entry_time=entry_time)
        plan = _make_plan(max_hold_hours=48, time_stop_min_gain=0.01, profit_target_pct=0.20)
        sig, _ = evaluate_exit(pos, plan, 100.5, rsi_14=50.0)
        assert sig is not None
        assert sig.trigger == "time_stop"

    def test_time_stop_not_triggered_with_gain(self):
        """Held 50 hours but gain 2% > 1% min → no time stop."""
        entry_time = datetime.now(timezone.utc) - timedelta(hours=50)
        pos = _make_position(entry_price=100.0, entry_time=entry_time)
        plan = _make_plan(max_hold_hours=48, time_stop_min_gain=0.01, profit_target_pct=0.20)
        sig, _ = evaluate_exit(pos, plan, 102.0, rsi_14=50.0)
        assert sig is None

    def test_priority_hard_stop_over_trailing(self):
        """Hard stop is checked before trailing stop."""
        pos = _make_position(
            entry_price=100.0,
            trailing_active=True,
            trailing_stop_price=98.0,
        )
        plan = _make_plan(stop_loss_pct=0.03)
        # Price 96.5 hits both hard stop (97) and trailing (98)
        sig, _ = evaluate_exit(pos, plan, 96.5, rsi_14=50.0)
        assert sig is not None
        assert sig.trigger == "hard_stop"


# ══════════════════════════════════════════════════════════════════════════════
# Morning Brief
# ══════════════════════════════════════════════════════════════════════════════


class TestMorningBrief:
    def test_bullish_regime(self):
        from sauce.morning_brief import get_regime

        mock_response = json.dumps({"regime": "bullish", "reasoning": "BTC up big"})
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(get_regime(0.05, 0.03, 0.01, 15.0, 55.0))
        assert result == "bullish"

    def test_bearish_regime(self):
        from sauce.morning_brief import get_regime

        mock_response = json.dumps({"regime": "bearish", "reasoning": "All red"})
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(get_regime(-0.05, -0.03, -0.02, 30.0, 25.0))
        assert result == "bearish"

    def test_fallback_on_error(self):
        """LLM error → falls back to neutral."""
        from sauce.adapters.llm import LLMError
        from sauce.morning_brief import get_regime

        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock, side_effect=LLMError("API down")):
            result = asyncio.run(get_regime(0.0, 0.0, 0.0, 20.0, 50.0))
        assert result == "neutral"

    def test_fallback_on_invalid_regime(self):
        """Claude returns invalid regime → falls back to neutral."""
        from sauce.morning_brief import get_regime

        mock_response = json.dumps({"regime": "sideways", "reasoning": "mixed"})
        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(get_regime(0.0, 0.0, 0.0, 20.0, 50.0))
        assert result == "neutral"

    def test_fallback_on_bad_json(self):
        """Claude returns non-JSON → falls back to neutral."""
        from sauce.morning_brief import get_regime

        with patch("sauce.morning_brief.call_claude", new_callable=AsyncMock, return_value="not valid json"):
            result = asyncio.run(get_regime(0.0, 0.0, 0.0, 20.0, 50.0))
        assert result == "neutral"


# ══════════════════════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════════════════════


class TestDatabase:
    """Tests DB CRUD operations against an isolated temp DB.

    The autouse _isolate_db fixture from conftest.py sets DB_PATH to a temp file.
    """

    def test_save_and_load_position(self):
        from sauce.db import close_position, load_open_positions, save_position

        pos = Position(
            id="pos-1", symbol="BTC/USD", qty=0.5, entry_price=50000.0,
            strategy_name="crypto_momentum_reversion",
        )
        save_position(pos)
        loaded = load_open_positions()
        assert len(loaded) == 1
        assert loaded[0].id == "pos-1"
        assert loaded[0].symbol == "BTC/USD"
        assert loaded[0].qty == 0.5

    def test_close_position(self):
        from sauce.db import close_position, load_open_positions, save_position

        pos = Position(id="pos-2", symbol="ETH/USD", qty=1.0, entry_price=3000.0,
                       strategy_name="test")
        save_position(pos)
        close_position("pos-2")
        loaded = load_open_positions()
        assert len(loaded) == 0

    def test_update_position(self):
        from sauce.db import load_open_positions, save_position, update_position

        pos = Position(id="pos-3", symbol="SOL/USD", qty=10.0, entry_price=150.0,
                       strategy_name="test")
        save_position(pos)

        pos.trailing_active = True
        pos.high_water_price = 160.0
        pos.trailing_stop_price = 156.8
        update_position(pos)

        loaded = load_open_positions()
        assert len(loaded) == 1
        assert loaded[0].trailing_active is True
        assert loaded[0].high_water_price == 160.0
        assert loaded[0].trailing_stop_price == pytest.approx(156.8)

    def test_log_signal(self):
        from sauce.adapters.db import get_session
        from sauce.db import SignalLogRow, log_signal

        signal = SignalResult(
            symbol="BTC/USD", side="buy", score=80, threshold=65,
            fired=True, rsi_14=30.0, macd_hist=0.5, bb_pct=-0.1,
            volume_ratio=2.0, regime="neutral",
            strategy_name="crypto_momentum_reversion",
        )
        log_signal(signal)

        session = get_session()
        try:
            rows = session.query(SignalLogRow).all()
            assert len(rows) == 1
            assert rows[0].symbol == "BTC/USD"
            assert rows[0].score == 80
            assert rows[0].fired is True
        finally:
            session.close()

    def test_log_trade(self):
        from sauce.adapters.db import get_session
        from sauce.db import TradeRow, log_trade

        pos = Position(
            id="trade-1", symbol="BTC/USD", qty=0.5, entry_price=50000.0,
            strategy_name="test",
            entry_time=datetime.now(timezone.utc) - timedelta(hours=24),
        )
        log_trade(pos, exit_price=52000.0, exit_trigger="profit_target")

        session = get_session()
        try:
            rows = session.query(TradeRow).all()
            assert len(rows) == 1
            assert rows[0].realized_pnl == pytest.approx(1000.0)  # (52000-50000)*0.5
            assert rows[0].exit_trigger == "profit_target"
        finally:
            session.close()

    def test_daily_stats_upsert(self):
        from sauce.db import get_daily_pnl, upsert_daily_stats

        upsert_daily_stats("2026-03-15", loop_runs=1, orders_placed=2,
                          realized_pnl_usd=150.0)
        upsert_daily_stats("2026-03-15", loop_runs=1, orders_placed=1)

        # Additive fields should accumulate
        from sauce.adapters.db import get_session
        from sauce.db import DailySummaryRow

        session = get_session()
        try:
            row = session.query(DailySummaryRow).filter_by(date="2026-03-15").first()
            assert row is not None
            assert row.loop_runs == 2
            assert row.orders_placed == 3
        finally:
            session.close()

    def test_get_daily_pnl_zero_when_missing(self):
        from sauce.db import get_daily_pnl
        assert get_daily_pnl("2099-01-01") == 0.0

    def test_instrument_meta_upsert(self):
        from sauce.adapters.db import get_session
        from sauce.db import InstrumentMetaRow, upsert_instrument_meta

        upsert_instrument_meta(
            symbol="BTC/USD",
            asset_class="crypto",
            strategy_name="crypto_momentum_reversion",
            last_signal_score=80,
            extra={"note": "test"},
        )

        session = get_session()
        try:
            row = session.query(InstrumentMetaRow).filter_by(symbol="BTC/USD").first()
            assert row is not None
            assert row.last_signal_score == 80
            assert json.loads(row.extra) == {"note": "test"}
        finally:
            session.close()
