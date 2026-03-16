"""
tests/test_options.py — Comprehensive test suite for options trading module.

Covers:
  - Options schemas (construction, validation, edge cases)
  - Options safety checks (all 7 guards)
  - Options exit engine (stage triggers, trailing stops, hard stop)
  - Options execution (entry/exit order building)
  - Options config (defaults, overrides)
  - Options position persistence (save/load/update/close round-trip)
  - Options learning (drift detection, performance analytics)
  - Options backtest (engine, models)
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta, timezone

import pytest

from sauce.core.options_schemas import (
    CompoundStage,
    ExitDecision,
    OptionsContract,
    OptionsBias,
    OptionsOrder,
    OptionsPosition,
    OptionsQuote,
    OptionsSignal,
)


# ═══════════════════════════════════════════════════════════════════════════════
# region: Schema Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCompoundStage:
    def test_basic_construction(self):
        s = CompoundStage(stage_num=1, trigger_multiplier=2.0, sell_fraction=0.5)
        assert s.stage_num == 1
        assert s.trigger_multiplier == 2.0
        assert s.sell_fraction == 0.5
        assert s.trailing_stop_pct == 0.15
        assert s.completed is False

    def test_rejects_trigger_below_1(self):
        with pytest.raises(Exception):
            CompoundStage(stage_num=1, trigger_multiplier=0.5, sell_fraction=0.5)

    def test_rejects_sell_fraction_above_1(self):
        with pytest.raises(Exception):
            CompoundStage(stage_num=1, trigger_multiplier=2.0, sell_fraction=1.5)

    def test_rejects_stage_num_zero(self):
        with pytest.raises(Exception):
            CompoundStage(stage_num=0, trigger_multiplier=2.0, sell_fraction=0.5)


class TestOptionsContract:
    def _make(self, **overrides):
        defaults = dict(
            contract_symbol="SPY250418C00500000",
            underlying="SPY",
            expiration="2025-04-18",
            strike=500.0,
            option_type="call",
        )
        defaults.update(overrides)
        return OptionsContract(**defaults)

    def test_basic_construction(self):
        c = self._make()
        assert c.underlying == "SPY"
        assert c.option_type == "call"
        assert c.delta is None

    def test_with_greeks(self):
        c = self._make(delta=0.45, theta=-0.02, iv=0.25)
        assert c.delta == 0.45
        assert c.theta == -0.02
        assert c.iv == 0.25

    def test_rejects_negative_strike(self):
        with pytest.raises(Exception):
            self._make(strike=-10.0)

    def test_rejects_invalid_option_type(self):
        with pytest.raises(Exception):
            self._make(option_type="straddle")


class TestOptionsPosition:
    def _make_contract(self):
        return OptionsContract(
            contract_symbol="SPY250418C00500000",
            underlying="SPY",
            expiration="2025-04-18",
            strike=500.0,
            option_type="call",
        )

    def test_basic_construction(self):
        pos = OptionsPosition(
            contract=self._make_contract(),
            entry_price=5.0,
            qty=2,
            remaining_qty=2,
            direction="long_call",
        )
        assert pos.entry_price == 5.0
        assert pos.qty == 2
        assert pos.remaining_qty == 2
        assert pos.stages_completed == 0
        assert pos.realized_pnl == 0.0
        assert pos.trailing_stop_price is None

    def test_position_id_auto_generated(self):
        p1 = OptionsPosition(
            contract=self._make_contract(), entry_price=5.0,
            qty=1, remaining_qty=1, direction="long_call",
        )
        p2 = OptionsPosition(
            contract=self._make_contract(), entry_price=5.0,
            qty=1, remaining_qty=1, direction="long_call",
        )
        assert p1.position_id != p2.position_id

    def test_with_compound_stages(self):
        stages = [
            CompoundStage(stage_num=1, trigger_multiplier=2.0, sell_fraction=0.5),
            CompoundStage(stage_num=2, trigger_multiplier=4.0, sell_fraction=0.5),
        ]
        pos = OptionsPosition(
            contract=self._make_contract(), entry_price=5.0,
            qty=4, remaining_qty=4, direction="long_call",
            compound_stages=stages,
        )
        assert len(pos.compound_stages) == 2

    def test_rejects_negative_entry_price(self):
        with pytest.raises(Exception):
            OptionsPosition(
                contract=self._make_contract(), entry_price=-1.0,
                qty=1, remaining_qty=1, direction="long_call",
            )


class TestOptionsBias:
    def test_confidence_clamping(self):
        b = OptionsBias(symbol="SPY", direction="bullish", confidence=1.5)
        assert b.confidence == 1.0

    def test_nan_confidence(self):
        b = OptionsBias(symbol="SPY", direction="bullish", confidence=float("nan"))
        assert b.confidence == 0.0

    def test_neutral_direction(self):
        b = OptionsBias(symbol="SPY", direction="neutral", confidence=0.5)
        assert b.direction == "neutral"


class TestOptionsQuote:
    def test_basic(self):
        q = OptionsQuote(
            contract_symbol="SPY250418C00500000",
            bid=4.50, ask=4.80, mid=4.65,
            as_of=datetime.now(timezone.utc),
        )
        assert q.bid == 4.50
        assert q.ask == 4.80

    def test_rejects_negative_bid(self):
        with pytest.raises(Exception):
            OptionsQuote(
                contract_symbol="SPY250418C00500000",
                bid=-0.01, ask=4.80, mid=4.65,
                as_of=datetime.now(timezone.utc),
            )


class TestExitDecision:
    def test_hold(self):
        d = ExitDecision(action="HOLD")
        assert d.qty == 0
        assert d.stage == 0

    def test_partial_close(self):
        d = ExitDecision(action="PARTIAL_CLOSE", qty=2, stage=1, set_trailing_stop=True,
                         trailing_stop_pct=0.15)
        assert d.action == "PARTIAL_CLOSE"
        assert d.set_trailing_stop is True


class TestOptionsOrder:
    def test_basic(self):
        o = OptionsOrder(
            contract_symbol="SPY250418C00500000", underlying="SPY",
            qty=1, side="buy", limit_price=5.0,
            as_of=datetime.now(timezone.utc), prompt_version="v1",
        )
        assert o.order_type == "limit"
        assert o.source == "options_entry"

    def test_rejects_market_order(self):
        with pytest.raises(Exception):
            OptionsOrder(
                contract_symbol="SPY250418C00500000", underlying="SPY",
                qty=1, side="buy", order_type="market", limit_price=5.0,
                as_of=datetime.now(timezone.utc), prompt_version="v1",
            )


# endregion


# ═══════════════════════════════════════════════════════════════════════════════
# region: Config Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionsConfig:
    def test_defaults(self):
        from sauce.core.options_config import OptionsSettings
        s = OptionsSettings()
        assert s.options_enabled is False
        assert s.option_profit_multiplier == 2.0
        assert s.option_compound_stages == 3
        assert s.option_sell_fraction == 0.5
        assert s.option_max_dte == 45
        assert s.option_min_dte == 7
        assert s.option_max_loss_pct == 0.50

    def test_universe_parsed(self):
        from sauce.core.options_config import OptionsSettings
        s = OptionsSettings(options_universe="SPY, QQQ , aapl")
        assert s.universe == ["SPY", "QQQ", "AAPL"]

    def test_override_via_constructor(self):
        from sauce.core.options_config import OptionsSettings
        s = OptionsSettings(option_profit_multiplier=3.0, option_compound_stages=5)
        assert s.option_profit_multiplier == 3.0
        assert s.option_compound_stages == 5


# endregion


# ═══════════════════════════════════════════════════════════════════════════════
# region: Safety Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionsSafety:
    """Test each of the 7 options safety checks independently."""

    def _make_contract(self, **kw):
        defaults = dict(
            contract_symbol="SPY250418C00500000",
            underlying="SPY",
            expiration=(date.today() + timedelta(days=30)).isoformat(),
            strike=500.0,
            option_type="call",
            delta=0.45,
        )
        defaults.update(kw)
        return OptionsContract(**defaults)

    def _make_quote(self, **kw):
        defaults = dict(
            contract_symbol="SPY250418C00500000",
            bid=4.50, ask=4.60, mid=4.55,
            as_of=datetime.now(timezone.utc),
        )
        defaults.update(kw)
        return OptionsQuote(**defaults)

    def _make_position(self, entry_price=5.0, remaining_qty=2):
        return OptionsPosition(
            contract=self._make_contract(),
            entry_price=entry_price, qty=2,
            remaining_qty=remaining_qty, direction="long_call",
        )

    # S1: Options enabled gate
    def test_check_options_enabled_pass(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")

        from sauce.core.options_safety import check_options_enabled
        passed, reason = check_options_enabled()
        assert passed is True
        get_options_settings.cache_clear()

    def test_check_options_enabled_fail(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "false")

        from sauce.core.options_safety import check_options_enabled
        passed, reason = check_options_enabled()
        assert passed is False
        assert "OPTIONS_ENABLED" in reason
        get_options_settings.cache_clear()

    # S2: IV rank
    def test_check_iv_rank_pass(self):
        from sauce.core.options_safety import check_iv_rank
        passed, _ = check_iv_rank(0.30)
        assert passed is True

    def test_check_iv_rank_fail_high(self):
        from sauce.core.options_safety import check_iv_rank
        passed, reason = check_iv_rank(0.90)
        assert passed is False
        assert "IV rank" in reason

    def test_check_iv_rank_fail_none(self):
        from sauce.core.options_safety import check_iv_rank
        passed, reason = check_iv_rank(None)
        assert passed is False
        assert "unavailable" in reason

    # S3: DTE
    def test_check_dte_pass(self):
        from sauce.core.options_safety import check_dte
        c = self._make_contract(expiration=(date.today() + timedelta(days=20)).isoformat())
        passed, _ = check_dte(c)
        assert passed is True

    def test_check_dte_fail_expired(self):
        from sauce.core.options_safety import check_dte
        c = self._make_contract(expiration=(date.today() - timedelta(days=1)).isoformat())
        passed, reason = check_dte(c)
        assert passed is False
        assert "expired" in reason.lower()

    def test_check_dte_fail_too_far(self):
        from sauce.core.options_safety import check_dte
        c = self._make_contract(expiration=(date.today() + timedelta(days=100)).isoformat())
        passed, reason = check_dte(c)
        assert passed is False
        assert "DTE" in reason

    def test_check_dte_fail_too_close(self):
        from sauce.core.options_safety import check_dte
        c = self._make_contract(expiration=(date.today() + timedelta(days=3)).isoformat())
        passed, reason = check_dte(c)
        assert passed is False

    # S4: Spread
    def test_check_spread_pass(self):
        from sauce.core.options_safety import check_spread
        q = self._make_quote(bid=4.50, ask=4.55, mid=4.525)
        passed, _ = check_spread(q)
        assert passed is True

    def test_check_spread_fail_wide(self):
        from sauce.core.options_safety import check_spread
        q = self._make_quote(bid=4.00, ask=5.00, mid=4.50)
        passed, reason = check_spread(q)
        assert passed is False
        assert "Spread" in reason

    def test_check_spread_fail_zero_mid(self):
        from sauce.core.options_safety import check_spread
        q = self._make_quote(bid=0.0, ask=0.0, mid=0.0)
        passed, reason = check_spread(q)
        assert passed is False

    # S5: Delta
    def test_check_delta_pass(self):
        from sauce.core.options_safety import check_delta
        c = self._make_contract(delta=0.45)
        passed, _ = check_delta(c)
        assert passed is True

    def test_check_delta_fail_too_low(self):
        from sauce.core.options_safety import check_delta
        c = self._make_contract(delta=0.10)
        passed, reason = check_delta(c)
        assert passed is False
        assert "Delta" in reason

    def test_check_delta_fail_too_high(self):
        from sauce.core.options_safety import check_delta
        c = self._make_contract(delta=0.90)
        passed, reason = check_delta(c)
        assert passed is False

    def test_check_delta_fail_none(self):
        from sauce.core.options_safety import check_delta
        c = self._make_contract(delta=None)
        passed, reason = check_delta(c)
        assert passed is False
        assert "unavailable" in reason

    # S6: Exposure
    def test_check_exposure_pass(self):
        from sauce.core.options_safety import check_exposure
        passed, _ = check_exposure(nav=10000.0, current_options_value=500.0, new_position_cost=500.0)
        assert passed is True

    def test_check_exposure_fail(self):
        from sauce.core.options_safety import check_exposure
        passed, reason = check_exposure(nav=10000.0, current_options_value=1500.0, new_position_cost=800.0)
        assert passed is False
        assert "exposure" in reason.lower()

    def test_check_exposure_fail_zero_nav(self):
        from sauce.core.options_safety import check_exposure
        passed, _ = check_exposure(nav=0.0, current_options_value=0.0, new_position_cost=100.0)
        assert passed is False

    # S7: Max loss
    def test_check_max_loss_pass(self):
        from sauce.core.options_safety import check_max_loss
        pos = self._make_position(entry_price=5.0)
        passed, _ = check_max_loss(pos, current_price=4.0)
        assert passed is True

    def test_check_max_loss_fail(self):
        from sauce.core.options_safety import check_max_loss
        pos = self._make_position(entry_price=5.0)
        # 60% loss (max_loss_pct default = 0.50)
        passed, reason = check_max_loss(pos, current_price=2.0)
        assert passed is False
        assert "hard stop" in reason.lower() or "Loss" in reason

    # run_entry_safety
    def test_run_entry_safety_all_pass(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")

        from sauce.core.options_safety import run_entry_safety
        c = self._make_contract()
        q = self._make_quote()
        passed, reason = run_entry_safety(
            contract=c, quote=q, iv_rank=0.30,
            nav=10000.0, current_options_value=0.0, new_position_cost=400.0,
        )
        assert passed is True
        assert reason == ""
        get_options_settings.cache_clear()

    def test_run_entry_safety_stops_at_first_failure(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "false")

        from sauce.core.options_safety import run_entry_safety
        c = self._make_contract()
        q = self._make_quote()
        passed, reason = run_entry_safety(
            contract=c, quote=q, iv_rank=0.30,
            nav=10000.0, current_options_value=0.0, new_position_cost=400.0,
        )
        assert passed is False
        assert "OPTIONS_ENABLED" in reason
        get_options_settings.cache_clear()


# endregion


# ═══════════════════════════════════════════════════════════════════════════════
# region: Exit Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionsExitEngine:
    """Test the deterministic exit engine."""

    def _make_position(self, entry_price=5.0, qty=4, stages=None):
        contract = OptionsContract(
            contract_symbol="SPY250418C00500000",
            underlying="SPY",
            expiration=(date.today() + timedelta(days=30)).isoformat(),
            strike=500.0,
            option_type="call",
        )
        if stages is None:
            stages = [
                CompoundStage(stage_num=1, trigger_multiplier=2.0, sell_fraction=0.5),
                CompoundStage(stage_num=2, trigger_multiplier=4.0, sell_fraction=0.5),
                CompoundStage(stage_num=3, trigger_multiplier=8.0, sell_fraction=1.0),
            ]
        return OptionsPosition(
            contract=contract, entry_price=entry_price,
            qty=qty, remaining_qty=qty, direction="long_call",
            compound_stages=stages,
        )

    def _make_quote(self, mid):
        return OptionsQuote(
            contract_symbol="SPY250418C00500000",
            bid=mid * 0.98, ask=mid * 1.02, mid=mid,
            as_of=datetime.now(timezone.utc),
        )

    def test_hold_when_no_exit_condition(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import evaluate_position

        pos = self._make_position(entry_price=5.0)
        quote = self._make_quote(mid=6.0)  # Up 20%, below 2x trigger
        decision = evaluate_position(pos, quote)
        assert decision.action == "HOLD"
        get_options_settings.cache_clear()

    def test_stage_1_trigger(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import evaluate_position

        pos = self._make_position(entry_price=5.0, qty=4)
        quote = self._make_quote(mid=10.0)  # 2x entry → stage 1
        decision = evaluate_position(pos, quote)
        assert decision.action == "PARTIAL_CLOSE"
        assert decision.stage == 1
        assert decision.qty == 2  # 50% of 4
        get_options_settings.cache_clear()

    def test_hard_stop_exit(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import evaluate_position

        pos = self._make_position(entry_price=5.0)
        quote = self._make_quote(mid=2.0)  # 60% loss > 50% hard stop
        decision = evaluate_position(pos, quote)
        assert decision.action == "FULL_CLOSE"
        assert decision.stage == 0  # hard stop = stage 0
        get_options_settings.cache_clear()

    def test_trailing_stop_exit(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import evaluate_position

        pos = self._make_position(entry_price=5.0)
        pos.trailing_stop_price = 8.0
        pos.stages_completed = 1
        pos.compound_stages[0].completed = True
        quote = self._make_quote(mid=7.5)  # Below trailing stop of 8.0
        decision = evaluate_position(pos, quote)
        assert decision.action == "FULL_CLOSE"
        get_options_settings.cache_clear()

    def test_trailing_stop_updates_upward(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import evaluate_position

        pos = self._make_position(entry_price=5.0)
        pos.trailing_stop_price = 8.0
        pos.stages_completed = 1
        pos.compound_stages[0].completed = True
        quote = self._make_quote(mid=12.0)  # Well above trailing stop
        decision = evaluate_position(pos, quote)
        # Should HOLD but update the trailing stop
        assert decision.action == "HOLD"
        # The trailing stop should have moved up
        assert pos.trailing_stop_price > 8.0
        get_options_settings.cache_clear()

    def test_zero_remaining_qty_returns_hold(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import evaluate_position

        pos = self._make_position(entry_price=5.0)
        pos.remaining_qty = 0
        quote = self._make_quote(mid=10.0)
        decision = evaluate_position(pos, quote)
        assert decision.action == "HOLD"
        get_options_settings.cache_clear()

    def test_apply_exit_decision_partial(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import apply_exit_decision

        pos = self._make_position(entry_price=5.0, qty=4)
        decision = ExitDecision(action="PARTIAL_CLOSE", qty=2, stage=1,
                                set_trailing_stop=True, trailing_stop_pct=0.20)
        result = apply_exit_decision(pos, decision)
        assert result.remaining_qty == 2
        assert result.stages_completed == 1
        assert result.compound_stages[0].completed is True
        get_options_settings.cache_clear()

    def test_apply_exit_decision_hold_no_mutation(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import apply_exit_decision

        pos = self._make_position(entry_price=5.0, qty=4)
        decision = ExitDecision(action="HOLD")
        result = apply_exit_decision(pos, decision)
        assert result.remaining_qty == 4
        assert result.stages_completed == 0
        get_options_settings.cache_clear()

    def test_build_exit_order_returns_order(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import build_exit_order

        pos = self._make_position(entry_price=5.0)
        decision = ExitDecision(action="PARTIAL_CLOSE", qty=2, stage=1)
        order = build_exit_order(pos, decision, limit_price=9.80)
        assert order is not None
        assert order.side == "sell"
        assert order.qty == 2
        assert order.source == "options_exit"
        get_options_settings.cache_clear()

    def test_build_exit_order_none_on_hold(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import build_exit_order

        pos = self._make_position()
        decision = ExitDecision(action="HOLD")
        order = build_exit_order(pos, decision, limit_price=5.0)
        assert order is None
        get_options_settings.cache_clear()

    def test_build_exit_order_stop_source(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTIONS_ENABLED", "true")
        from sauce.agents.options_exit import build_exit_order

        pos = self._make_position()
        decision = ExitDecision(action="FULL_CLOSE", qty=4, stage=0)
        order = build_exit_order(pos, decision, limit_price=2.0)
        assert order is not None
        assert order.source == "options_stop"
        get_options_settings.cache_clear()


# endregion


# ═══════════════════════════════════════════════════════════════════════════════
# region: Execution Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionsExecution:
    """Test entry/exit order building."""

    def _make_signal(self):
        now = datetime.now(timezone.utc)
        contract = OptionsContract(
            contract_symbol="SPY250418C00500000", underlying="SPY",
            expiration="2025-04-18", strike=500.0, option_type="call",
            delta=0.45,
        )
        bias = OptionsBias(symbol="SPY", direction="bullish", confidence=0.8)
        return OptionsSignal(
            symbol="SPY", contract=contract, direction="long_call",
            confidence=0.8, bias=bias, as_of=now, prompt_version="v1",
        )

    def _make_quote(self, mid=5.0):
        return OptionsQuote(
            contract_symbol="SPY250418C00500000",
            bid=mid * 0.98, ask=mid * 1.02, mid=mid,
            as_of=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_build_entry_order_success(self):
        from sauce.agents.options_execution import build_entry_order
        signal = self._make_signal()
        quote = self._make_quote(mid=5.0)
        order = await build_entry_order(signal, quote, loop_id="test")
        assert order is not None
        assert order.side == "buy"
        assert order.qty == 1
        assert order.limit_price > 0
        assert order.limit_price <= quote.ask

    @pytest.mark.asyncio
    async def test_build_entry_order_rejects_stale_quote(self):
        from sauce.agents.options_execution import build_entry_order
        signal = self._make_signal()
        quote = self._make_quote()
        # Make quote stale
        quote = OptionsQuote(
            contract_symbol="SPY250418C00500000",
            bid=4.90, ask=5.10, mid=5.0,
            as_of=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        order = await build_entry_order(signal, quote, loop_id="test")
        assert order is None

    @pytest.mark.asyncio
    async def test_build_entry_order_rejects_wide_spread(self, monkeypatch):
        from sauce.core.options_config import get_options_settings
        get_options_settings.cache_clear()
        monkeypatch.setenv("OPTION_MAX_SPREAD_PCT", "0.02")

        from sauce.agents.options_execution import build_entry_order
        signal = self._make_signal()
        # 20% spread
        quote = OptionsQuote(
            contract_symbol="SPY250418C00500000",
            bid=4.0, ask=5.0, mid=4.5,
            as_of=datetime.now(timezone.utc),
        )
        order = await build_entry_order(signal, quote, loop_id="test")
        assert order is None
        get_options_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_build_exit_order_success(self):
        from sauce.agents.options_execution import build_exit_order
        quote = self._make_quote(mid=10.0)
        order = await build_exit_order(
            contract_symbol="SPY250418C00500000", underlying="SPY",
            qty=2, quote=quote, stage=1, loop_id="test",
        )
        assert order is not None
        assert order.side == "sell"
        assert order.qty == 2
        assert order.limit_price >= quote.bid

    @pytest.mark.asyncio
    async def test_build_exit_order_zero_qty_returns_none(self):
        from sauce.agents.options_execution import build_exit_order
        quote = self._make_quote()
        order = await build_exit_order(
            contract_symbol="SPY250418C00500000", underlying="SPY",
            qty=0, quote=quote, stage=1,
        )
        assert order is None


# endregion


# ═══════════════════════════════════════════════════════════════════════════════
# region: Position Persistence Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionsPersistence:
    """Test save/load/update/close round-trip for OptionsPositionRow."""

    def _make_position(self):
        contract = OptionsContract(
            contract_symbol="SPY250418C00500000", underlying="SPY",
            expiration="2025-04-18", strike=500.0, option_type="call",
        )
        return OptionsPosition(
            contract=contract, entry_price=5.0,
            qty=4, remaining_qty=4, direction="long_call",
            compound_stages=[
                CompoundStage(stage_num=1, trigger_multiplier=2.0, sell_fraction=0.5),
                CompoundStage(stage_num=2, trigger_multiplier=4.0, sell_fraction=0.5),
            ],
        )

    def test_save_and_load(self, tmp_path):
        from sauce.adapters.db import save_options_position, load_open_options_positions
        db = str(tmp_path / "test.db")

        pos = self._make_position()
        save_options_position(pos, loop_id="L1", db_path=db)

        loaded = load_open_options_positions(db_path=db)
        assert len(loaded) == 1
        row = loaded[0]
        assert row["underlying"] == "SPY"
        assert row["entry_price"] == 5.0
        assert row["qty"] == 4
        assert row["remaining_qty"] == 4
        assert row["direction"] == "long_call"
        assert row["position_id"] == pos.position_id

        # Compound stages deserialized correctly
        stages = json.loads(row["compound_stages_json"])
        assert len(stages) == 2
        assert stages[0]["trigger_multiplier"] == 2.0

    def test_update_position(self, tmp_path):
        from sauce.adapters.db import save_options_position, update_options_position, load_open_options_positions
        db = str(tmp_path / "test.db")

        pos = self._make_position()
        save_options_position(pos, loop_id="L1", db_path=db)

        update_options_position(
            pos.position_id, db_path=db,
            remaining_qty=2, stages_completed=1, trailing_stop_price=9.0,
        )

        loaded = load_open_options_positions(db_path=db)
        assert len(loaded) == 1
        assert loaded[0]["remaining_qty"] == 2
        assert loaded[0]["stages_completed"] == 1
        assert loaded[0]["trailing_stop_price"] == 9.0

    def test_close_position(self, tmp_path):
        from sauce.adapters.db import save_options_position, close_options_position, load_open_options_positions
        db = str(tmp_path / "test.db")

        pos = self._make_position()
        save_options_position(pos, loop_id="L1", db_path=db)

        close_options_position(pos.position_id, realized_pnl=250.0, db_path=db)

        # Should NOT appear in open positions
        loaded = load_open_options_positions(db_path=db)
        assert len(loaded) == 0

    def test_multiple_positions(self, tmp_path):
        from sauce.adapters.db import save_options_position, load_open_options_positions, close_options_position
        db = str(tmp_path / "test.db")

        p1 = self._make_position()
        p2 = self._make_position()
        save_options_position(p1, loop_id="L1", db_path=db)
        save_options_position(p2, loop_id="L1", db_path=db)

        # Both open
        loaded = load_open_options_positions(db_path=db)
        assert len(loaded) == 2

        # Close one
        close_options_position(p1.position_id, realized_pnl=100.0, db_path=db)
        loaded = load_open_options_positions(db_path=db)
        assert len(loaded) == 1
        assert loaded[0]["position_id"] == p2.position_id

    def test_update_nonexistent_position_no_crash(self, tmp_path):
        from sauce.adapters.db import update_options_position
        db = str(tmp_path / "test.db")
        # Should not raise
        update_options_position("nonexistent-id", db_path=db, remaining_qty=0)


# endregion


# ═══════════════════════════════════════════════════════════════════════════════
# region: Learning Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionsLearning:
    """Test options learning analytics."""

    def _seed_closed_positions(self, db_path: str, n: int = 10, win_ratio: float = 0.5):
        from sauce.adapters.db import save_options_position, close_options_position

        for i in range(n):
            contract = OptionsContract(
                contract_symbol=f"SPY{i:06d}C00500000",
                underlying="SPY", expiration="2025-04-18",
                strike=500.0, option_type="call",
            )
            pos = OptionsPosition(
                contract=contract, entry_price=5.0,
                qty=2, remaining_qty=2, direction="long_call",
            )
            save_options_position(pos, loop_id="L1", db_path=db_path)
            pnl = 100.0 if i < int(n * win_ratio) else -50.0
            close_options_position(pos.position_id, realized_pnl=pnl, db_path=db_path)

    def test_record_trade_outcome(self, tmp_path):
        from sauce.adapters.db import save_options_position
        from sauce.memory.options_learning import record_options_trade_outcome

        db = str(tmp_path / "test.db")
        contract = OptionsContract(
            contract_symbol="SPY250418C00500000", underlying="SPY",
            expiration="2025-04-18", strike=500.0, option_type="call",
        )
        pos = OptionsPosition(
            contract=contract, entry_price=5.0,
            qty=2, remaining_qty=2, direction="long_call",
        )
        save_options_position(pos, loop_id="L1", db_path=db)

        outcome = record_options_trade_outcome(
            pos.position_id, exit_price=10.0,
            exit_reason="stage_1", stages_completed=1, db_path=db,
        )
        assert outcome is not None
        assert outcome["pnl"] == 1000.0  # (10 - 5) * 2 * 100
        assert outcome["win"] is True
        assert outcome["underlying"] == "SPY"

    def test_record_trade_outcome_not_found(self, tmp_path):
        from sauce.memory.options_learning import record_options_trade_outcome
        db = str(tmp_path / "test.db")
        # Force table creation
        from sauce.adapters.db import get_engine
        get_engine(db)

        outcome = record_options_trade_outcome(
            "nonexistent", exit_price=10.0,
            exit_reason="stage_1", stages_completed=1, db_path=db,
        )
        assert outcome is None

    def test_detect_win_rate_drift_healthy(self, tmp_path):
        from sauce.memory.options_learning import detect_options_win_rate_drift
        db = str(tmp_path / "test.db")
        self._seed_closed_positions(db, n=10, win_ratio=0.7)

        result = detect_options_win_rate_drift(db_path=db, window=10, threshold=0.40)
        assert result is None  # Healthy

    def test_detect_win_rate_drift_unhealthy(self, tmp_path):
        from sauce.memory.options_learning import detect_options_win_rate_drift
        db = str(tmp_path / "test.db")
        self._seed_closed_positions(db, n=10, win_ratio=0.2)

        result = detect_options_win_rate_drift(db_path=db, window=10, threshold=0.40)
        assert result is not None
        assert result["win_rate"] < 0.40
        assert result["asset_class"] == "options"

    def test_detect_drift_insufficient_data(self, tmp_path):
        from sauce.memory.options_learning import detect_options_win_rate_drift
        db = str(tmp_path / "test.db")
        self._seed_closed_positions(db, n=3, win_ratio=0.0)

        result = detect_options_win_rate_drift(db_path=db, window=10, threshold=0.40)
        assert result is None  # Not enough data

    def test_performance_by_stage(self, tmp_path):
        from sauce.memory.options_learning import get_options_performance_by_stage
        from sauce.adapters.db import save_options_position, close_options_position, get_session, OptionsPositionRow

        db = str(tmp_path / "test.db")

        # Create positions with different stages_completed
        for stages in [0, 0, 1, 1, 2]:
            contract = OptionsContract(
                contract_symbol=f"SPY{stages:06d}C00500000",
                underlying="SPY", expiration="2025-04-18",
                strike=500.0, option_type="call",
            )
            pos = OptionsPosition(
                contract=contract, entry_price=5.0,
                qty=2, remaining_qty=2, direction="long_call",
            )
            save_options_position(pos, loop_id="L1", db_path=db)
            pnl = 100.0 if stages > 0 else -50.0
            close_options_position(pos.position_id, realized_pnl=pnl, db_path=db)
            # Set stages_completed on the DB row
            session = get_session(db)
            try:
                row = session.query(OptionsPositionRow).filter_by(position_id=pos.position_id).first()
                row.stages_completed = stages
                session.commit()
            finally:
                session.close()

        report = get_options_performance_by_stage(db_path=db)
        assert report["total"] == 5
        assert 0 in report["by_stage"]
        assert report["by_stage"][0]["count"] == 2

    def test_performance_by_underlying(self, tmp_path):
        from sauce.memory.options_learning import get_options_performance_by_underlying
        from sauce.adapters.db import save_options_position, close_options_position

        db = str(tmp_path / "test.db")

        for sym in ["SPY", "SPY", "QQQ"]:
            contract = OptionsContract(
                contract_symbol=f"{sym}250418C00500000",
                underlying=sym, expiration="2025-04-18",
                strike=500.0, option_type="call",
            )
            pos = OptionsPosition(
                contract=contract, entry_price=5.0,
                qty=2, remaining_qty=2, direction="long_call",
            )
            save_options_position(pos, loop_id="L1", db_path=db)
            close_options_position(pos.position_id, realized_pnl=100.0, db_path=db)

        report = get_options_performance_by_underlying(db_path=db)
        assert report["total"] == 3
        assert "SPY" in report["by_symbol"]
        assert report["by_symbol"]["SPY"]["count"] == 2


# endregion


# ═══════════════════════════════════════════════════════════════════════════════
# region: Backtest Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOptionsBacktestModels:
    def test_config_defaults(self):
        from sauce.backtest.options_models import OptionsBacktestConfig
        cfg = OptionsBacktestConfig()
        assert cfg.initial_capital == 10_000.0
        assert cfg.compound_stages == 3
        assert cfg.profit_multiplier == 2.0
        assert cfg.commission_per_contract == 0.65

    def test_result_init(self):
        from sauce.backtest.options_models import OptionsBacktestConfig, OptionsBacktestResult
        cfg = OptionsBacktestConfig()
        r = OptionsBacktestResult(symbol="SPY", config=cfg)
        assert r.trades == []
        assert r.total_trades == 0
        assert r.win_rate == 0.0

    def test_trade_frozen(self):
        from sauce.backtest.options_models import OptionsBacktestTrade, OptionsExitReason
        t = OptionsBacktestTrade(
            symbol="SPY", direction="long_call",
            entry_time=datetime.now(timezone.utc),
            exit_time=datetime.now(timezone.utc),
            entry_price=5.0, exit_price=10.0, qty=1,
            pnl=500.0, pnl_pct=1.0,
            exit_reason=OptionsExitReason.STAGE_1,
            stages_completed=1, bars_held=50,
        )
        with pytest.raises(Exception):
            t.pnl = 999.0  # frozen


class TestOptionsBacktestEngine:
    """Test the options backtest engine."""

    def _make_df(self, n=200, trend="up"):
        """Create synthetic OHLCV data."""
        import numpy as np
        import pandas as pd

        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        base = 100.0
        if trend == "up":
            noise = np.random.RandomState(42).randn(n).cumsum() * 0.5
            closes = base + np.linspace(0, 30, n) + noise
        elif trend == "down":
            noise = np.random.RandomState(42).randn(n).cumsum() * 0.5
            closes = base + np.linspace(0, -30, n) + noise
        else:  # flat
            noise = np.random.RandomState(42).randn(n).cumsum() * 0.3
            closes = base + noise

        highs = closes + abs(np.random.RandomState(43).randn(n) * 0.5)
        lows = closes - abs(np.random.RandomState(44).randn(n) * 0.5)
        opens = closes + np.random.RandomState(45).randn(n) * 0.2
        volumes = np.random.RandomState(46).randint(100000, 1000000, n)

        return pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": volumes,
        }, index=dates)

    def test_basic_run(self):
        from sauce.backtest.options_engine import run_options_backtest
        df = self._make_df(200, "up")
        result = run_options_backtest("SPY", df)
        assert result.symbol == "SPY"
        assert len(result.equity_curve) > 0
        assert result.total_trades >= 0

    def test_empty_dataframe(self):
        import pandas as pd
        from sauce.backtest.options_engine import run_options_backtest
        df = pd.DataFrame()
        result = run_options_backtest("SPY", df)
        assert result.total_trades == 0
        assert result.trades == []

    def test_short_dataframe(self):
        from sauce.backtest.options_engine import run_options_backtest
        df = self._make_df(10)
        result = run_options_backtest("SPY", df)
        assert result.total_trades == 0

    def test_custom_config(self):
        from sauce.backtest.options_engine import run_options_backtest
        from sauce.backtest.options_models import OptionsBacktestConfig
        cfg = OptionsBacktestConfig(
            initial_capital=50_000, position_size_pct=0.10,
            compound_stages=2, profit_multiplier=3.0,
        )
        df = self._make_df(200, "up")
        result = run_options_backtest("SPY", df, config=cfg)
        assert result.config.initial_capital == 50_000
        assert result.config.compound_stages == 2

    def test_metrics_computed(self):
        from sauce.backtest.options_engine import run_options_backtest
        df = self._make_df(500, "up")
        result = run_options_backtest("SPY", df)
        if result.total_trades > 0:
            assert result.winning_trades + result.losing_trades == result.total_trades
            assert 0.0 <= result.win_rate <= 1.0

    def test_equity_curve_starts_at_capital(self):
        from sauce.backtest.options_engine import run_options_backtest
        from sauce.backtest.options_models import OptionsBacktestConfig
        cfg = OptionsBacktestConfig(initial_capital=10_000)
        df = self._make_df(200)
        result = run_options_backtest("SPY", df, config=cfg)
        assert result.equity_curve[0] == 10_000.0

    def test_rsi_simple(self):
        """Test the internal RSI computation."""
        from sauce.backtest.options_engine import _simple_rsi
        import numpy as np
        # Create a series that should produce known RSI values
        closes = list(np.linspace(100, 130, 50))  # pure uptrend
        rsi = _simple_rsi(closes, 14)
        assert math.isnan(rsi[0])  # first values are NaN
        assert not math.isnan(rsi[14])  # first valid RSI at period
        # Pure uptrend should have RSI > 50
        assert rsi[-1] > 50


# endregion
