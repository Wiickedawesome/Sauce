from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from sauce.analyst import AnalystVerdict
from sauce.core.config import get_settings
from sauce.core.options_schemas import OptionContract, OptionsOrder, OptionsPosition, OptionsSignalResult
from sauce.core.schemas import AuditEvent, Order, PriceReference
from sauce.exit_monitor import ExitSignal
from sauce.memory import TradeMemory
from sauce.risk import RiskVerdict
from sauce.strategy import ExitPlan, Position, SignalResult
from sauce.loop import _scan_entries, _scan_exits, _scan_option_entries, _scan_option_exits, main, run_cycle


def _capture_events(events: list[AuditEvent]):
    def _capture(event: AuditEvent) -> None:
        events.append(event)

    return _capture


@pytest.mark.asyncio
async def test_run_cycle_paused_before_external_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_PAUSE", "true")
    get_settings.cache_clear()

    events: list[AuditEvent] = []
    mock_get_regime = AsyncMock()
    mock_get_account = MagicMock()

    monkeypatch.setattr("sauce.loop.log_event", _capture_events(events))
    monkeypatch.setattr("sauce.loop.get_regime", mock_get_regime)
    monkeypatch.setattr("sauce.loop.get_account", mock_get_account)

    await run_cycle()

    assert mock_get_regime.await_count == 0
    assert mock_get_account.call_count == 0

    safety_events = [event for event in events if event.event_type == "safety_check"]
    assert safety_events
    assert safety_events[-1].payload["reason"] == "TRADING_PAUSE enabled"

    loop_end_events = [event for event in events if event.event_type == "loop_end"]
    assert loop_end_events
    assert loop_end_events[-1].payload["status"] == "halted"


@pytest.mark.asyncio
async def test_run_cycle_reraises_after_logging_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_PAUSE", "false")
    get_settings.cache_clear()

    events: list[AuditEvent] = []
    monkeypatch.setattr("sauce.loop.log_event", _capture_events(events))
    monkeypatch.setattr("sauce.loop._hour_et", lambda: 0)
    monkeypatch.setattr("sauce.loop.get_daily_regime", lambda _today: None)
    monkeypatch.setattr("sauce.loop.get_account", MagicMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await run_cycle()

    loop_end_events = [event for event in events if event.event_type == "loop_end"]
    assert loop_end_events
    assert loop_end_events[-1].payload["status"] == "failed"
    assert loop_end_events[-1].payload["error"] == "boom"


def test_main_propagates_cycle_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _boom() -> None:
        raise RuntimeError("cycle-failed")

    monkeypatch.setattr("sauce.loop.run_cycle", _boom)

    with pytest.raises(RuntimeError, match="cycle-failed"):
        main()


@pytest.mark.asyncio
async def test_scan_exits_persists_mutated_trailing_state(monkeypatch: pytest.MonkeyPatch) -> None:
    position = Position(
        id="pos-trail",
        symbol="BTC/USD",
        qty=1.0,
        entry_price=100.0,
        high_water_price=100.0,
        strategy_name="fake",
    )
    quote = PriceReference(symbol="BTC/USD", bid=104.0, ask=104.0, mid=104.0, as_of=position.entry_time)
    fake_strategy = SimpleNamespace(
        build_exit_plan=lambda _position, _tier: ExitPlan(
            stop_loss_pct=0.03,
            trail_activation_pct=0.03,
            trail_pct=0.02,
            profit_target_pct=0.20,
            rsi_exhaustion_threshold=72,
            max_hold_hours=48,
            time_stop_min_gain=0.01,
        )
    )

    def _mutating_exit(position_arg, *args, **kwargs):
        position_arg.trailing_active = True
        position_arg.high_water_price = 104.0
        position_arg.trailing_stop_price = 101.92
        return None, position_arg

    update_position = MagicMock()
    monkeypatch.setattr("sauce.loop.get_positions", MagicMock(return_value=[]))
    monkeypatch.setattr("sauce.loop.get_universe_snapshot", MagicMock(return_value={"BTC/USD": quote}))
    monkeypatch.setattr("sauce.loop._fetch_indicators", MagicMock(return_value=None))
    monkeypatch.setattr("sauce.loop._find_strategy", lambda _name: fake_strategy)
    monkeypatch.setattr("sauce.loop.evaluate_exit", _mutating_exit)
    monkeypatch.setattr("sauce.loop.update_position", update_position)

    await _scan_exits([position], equity=1_000.0, regime="neutral", trade_memory=TradeMemory())

    assert update_position.call_count == 1
    persisted = update_position.call_args.args[0]
    assert persisted.trailing_active is True
    assert persisted.trailing_stop_price == pytest.approx(101.92)


@pytest.mark.asyncio
async def test_scan_exits_keeps_position_open_on_partial_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    position = Position(
        id="pos-exit",
        symbol="BTC/USD",
        qty=1.0,
        entry_price=100.0,
        high_water_price=100.0,
        strategy_name="fake",
    )
    quote = PriceReference(symbol="BTC/USD", bid=110.0, ask=110.0, mid=110.0, as_of=position.entry_time)
    fake_strategy = SimpleNamespace(
        build_exit_plan=lambda _position, _tier: ExitPlan(
            stop_loss_pct=0.03,
            trail_activation_pct=0.03,
            trail_pct=0.02,
            profit_target_pct=0.06,
            rsi_exhaustion_threshold=72,
            max_hold_hours=48,
            time_stop_min_gain=0.01,
        )
    )
    exit_signal = ExitSignal(
        trigger="profit_target",
        symbol="BTC/USD",
        position_id="pos-exit",
        side="sell",
        current_price=110.0,
        reason="target hit",
    )

    log_trade = MagicMock()
    close_position = MagicMock()
    update_position = MagicMock()

    monkeypatch.setattr("sauce.loop.get_positions", MagicMock(return_value=[{"symbol": "BTC/USD", "qty": "1.0"}]))
    monkeypatch.setattr("sauce.loop.get_universe_snapshot", MagicMock(return_value={"BTC/USD": quote}))
    monkeypatch.setattr("sauce.loop._fetch_indicators", MagicMock(return_value=None))
    monkeypatch.setattr("sauce.loop._find_strategy", lambda _name: fake_strategy)
    monkeypatch.setattr("sauce.loop.evaluate_exit", lambda *args, **kwargs: (exit_signal, position))
    monkeypatch.setattr(
        "sauce.loop.place_order",
        MagicMock(return_value={"status": "partially_filled", "filled_qty": "0.4", "filled_avg_price": "108.0"}),
    )
    monkeypatch.setattr("sauce.loop.log_trade", log_trade)
    monkeypatch.setattr("sauce.loop.close_position", close_position)
    monkeypatch.setattr("sauce.loop.update_position", update_position)

    await _scan_exits([position], equity=1_000.0, regime="neutral", trade_memory=TradeMemory())

    assert log_trade.call_count == 1
    traded_position, traded_exit_price, _trigger = log_trade.call_args.args
    assert traded_position.qty == pytest.approx(0.4)
    assert traded_exit_price == pytest.approx(108.0)
    assert close_position.call_count == 0
    assert update_position.call_count == 1
    persisted = update_position.call_args.args[0]
    assert persisted.qty == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_scan_entries_uses_broker_avg_entry_price_when_fill_price_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStrategy:
        name = "fake_strategy"
        instruments = ["BTC/USD"]

        def eligible(self, instrument: str, regime: str) -> bool:
            return True

        def score(self, indicators, instrument: str, regime: str, current_price: float) -> SignalResult:
            return SignalResult(
                symbol=instrument,
                side="buy",
                score=80,
                threshold=60,
                fired=True,
                rsi_14=30.0,
                macd_hist=0.2,
                bb_pct=0.1,
                volume_ratio=1.5,
                regime=regime,
                strategy_name=self.name,
            )

        def build_order(self, signal: SignalResult, account: dict[str, object], tier) -> Order:
            return Order(
                symbol=signal.symbol,
                side="buy",
                qty=1.0,
                order_type="limit",
                time_in_force="gtc",
                limit_price=100.0,
                as_of=quote.as_of,
                prompt_version="v2",
            )

    quote = PriceReference(symbol="BTC/USD", bid=100.0, ask=100.0, mid=100.0, as_of=datetime.now(UTC))
    save_position = MagicMock()

    monkeypatch.setattr("sauce.loop.STRATEGIES", [FakeStrategy()])
    monkeypatch.setattr("sauce.loop.get_universe_snapshot", MagicMock(return_value={"BTC/USD": quote}))
    monkeypatch.setattr("sauce.loop._fetch_indicators", MagicMock(return_value=object()))
    monkeypatch.setattr(
        "sauce.loop.analyst_committee",
        AsyncMock(
            return_value=AnalystVerdict(
                approve=True,
                confidence=90,
                bull_case="bull",
                bear_case="bear",
                reasoning="good",
            )
        ),
    )
    monkeypatch.setattr("sauce.loop.get_quote", MagicMock(return_value=quote))
    monkeypatch.setattr("sauce.loop.check_risk", MagicMock(return_value=RiskVerdict(True, "all", "")))
    monkeypatch.setattr("sauce.loop._supervisor_review", MagicMock(return_value=SimpleNamespace(action="execute", reason="ok")))
    monkeypatch.setattr("sauce.loop.place_order", MagicMock(return_value={"id": "ord-1", "status": "filled", "filled_qty": "1.0"}))
    monkeypatch.setattr("sauce.loop.get_positions", MagicMock(return_value=[{"symbol": "BTC/USD", "qty": "1.0", "avg_entry_price": "101.5"}]))
    monkeypatch.setattr("sauce.loop.save_position", save_position)
    monkeypatch.setattr("sauce.loop.log_signal", MagicMock())
    monkeypatch.setattr("sauce.loop.upsert_daily_stats", MagicMock())

    await _scan_entries(
        regime="neutral",
        account={"equity": "1000", "buying_power": "1000", "last_equity": "1000"},
        open_positions=[],
        broker_positions=[],
        recent_orders=[],
        trade_memory=TradeMemory(),
    )

    assert save_position.call_count == 1
    saved_position = save_position.call_args.args[0]
    assert saved_position.entry_price == pytest.approx(101.5)


@pytest.mark.asyncio
async def test_scan_entries_overrides_analyst_rejection_in_paper_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_PAPER", "true")
    get_settings.cache_clear()

    class FakeStrategy:
        name = "fake_strategy"
        instruments = ["BTC/USD"]

        def eligible(self, instrument: str, regime: str) -> bool:
            return True

        def score(self, indicators, instrument: str, regime: str, current_price: float) -> SignalResult:
            return SignalResult(
                symbol=instrument,
                side="buy",
                score=60,
                threshold=60,
                fired=True,
                rsi_14=55.0,
                macd_hist=0.2,
                bb_pct=0.1,
                volume_ratio=1.5,
                regime=regime,
                strategy_name=self.name,
            )

        def build_order(self, signal: SignalResult, account: dict[str, object], tier) -> Order:
            return Order(
                symbol=signal.symbol,
                side="buy",
                qty=1.0,
                order_type="limit",
                time_in_force="gtc",
                limit_price=100.0,
                as_of=quote.as_of,
                prompt_version="v2",
            )

    quote = PriceReference(symbol="BTC/USD", bid=100.0, ask=100.0, mid=100.0, as_of=datetime.now(UTC))
    save_position = MagicMock()

    monkeypatch.setattr("sauce.loop.STRATEGIES", [FakeStrategy()])
    monkeypatch.setattr("sauce.loop.get_snapshot_candidates", lambda symbols: symbols)
    monkeypatch.setattr("sauce.loop.get_universe_snapshot", MagicMock(return_value={"BTC/USD": quote}))
    monkeypatch.setattr("sauce.loop._fetch_indicators", MagicMock(return_value=object()))
    monkeypatch.setattr(
        "sauce.loop.analyst_committee",
        AsyncMock(
            return_value=AnalystVerdict(
                approve=False,
                confidence=90,
                bull_case="bull",
                bear_case="bear",
                reasoning="llm veto",
            )
        ),
    )
    monkeypatch.setattr("sauce.loop.get_quote", MagicMock(return_value=quote))
    monkeypatch.setattr("sauce.loop.check_risk", MagicMock(return_value=RiskVerdict(True, "all", "")))
    monkeypatch.setattr("sauce.loop._supervisor_review", MagicMock(return_value=SimpleNamespace(action="execute", reason="ok")))
    monkeypatch.setattr("sauce.loop.place_order", MagicMock(return_value={"id": "ord-1", "status": "filled", "filled_qty": "1.0", "filled_avg_price": "100.0"}))
    monkeypatch.setattr("sauce.loop.get_positions", MagicMock(return_value=[{"symbol": "BTC/USD", "qty": "1.0", "avg_entry_price": "100.0"}]))
    monkeypatch.setattr("sauce.loop.save_position", save_position)
    monkeypatch.setattr("sauce.loop.log_signal", MagicMock())
    monkeypatch.setattr("sauce.loop.upsert_daily_stats", MagicMock())
    monkeypatch.setattr("sauce.loop._audit_event", MagicMock())

    await _scan_entries(
        regime="bearish",
        account={"equity": "1000", "buying_power": "1000", "last_equity": "1000"},
        open_positions=[],
        broker_positions=[],
        recent_orders=[],
        trade_memory=TradeMemory(),
    )

    assert save_position.call_count == 1


@pytest.mark.asyncio
async def test_scan_option_entries_rejects_oversized_actual_premium(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIONS_ENABLED", "true")
    get_settings.cache_clear()

    quote = PriceReference(symbol="SPY", bid=100.0, ask=100.0, mid=100.0, as_of=datetime.now(UTC))
    contract = OptionContract(
        underlying="SPY",
        contract_symbol="SPY250417C00100000",
        option_type="call",
        strike=100.0,
        expiration=quote.as_of.date(),
        dte=20,
        delta=0.3,
        bid=5.5,
        ask=6.0,
        mid=5.75,
        open_interest=1000,
    )

    class FakeOptionsStrategy:
        name = "options_momentum"
        instruments = ["SPY"]

        def eligible(self, underlying: str, regime: str) -> bool:
            return True

        def score(self, indicators, underlying: str, regime: str, current_price: float) -> OptionsSignalResult:
            return OptionsSignalResult(
                underlying=underlying,
                option_type="call",
                score=90,
                threshold=40,
                fired=True,
                regime=regime,
                strategy_name=self.name,
            )

        def select_contract(self, signal, available_contracts, current_price):
            return contract

        def build_order(self, signal, chosen_contract, account, tier) -> OptionsOrder:
            return OptionsOrder(
                underlying=signal.underlying,
                contract_symbol=chosen_contract.contract_symbol,
                option_type=signal.option_type,
                side="buy",
                qty=1,
                limit_price=6.0,
            )

    monkeypatch.setattr("sauce.loop.OPTIONS_STRATEGY", FakeOptionsStrategy())
    monkeypatch.setattr("sauce.loop.get_universe_snapshot", MagicMock(return_value={"SPY": quote}))
    monkeypatch.setattr("sauce.loop._fetch_indicators", MagicMock(return_value=object()))
    monkeypatch.setattr(
        "sauce.loop.analyst_committee",
        AsyncMock(
            return_value=AnalystVerdict(
                approve=True,
                confidence=90,
                bull_case="bull",
                bear_case="bear",
                reasoning="good",
            )
        ),
    )
    monkeypatch.setattr("sauce.loop.get_option_chain", MagicMock(return_value=[contract]))
    place_option_order = MagicMock()
    monkeypatch.setattr("sauce.loop.place_option_order", place_option_order)
    monkeypatch.setattr("sauce.loop._audit_event", MagicMock())

    await _scan_option_entries(
        regime="neutral",
        account={"equity": "10000", "buying_power": "10000", "last_equity": "10000"},
        open_option_positions=[],
        broker_positions=[],
        recent_orders=[],
        trade_memory=TradeMemory(),
    )

    assert place_option_order.call_count == 0


@pytest.mark.asyncio
async def test_scan_option_exits_keeps_contract_open_on_partial_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    option_position = OptionsPosition(
        position_id="opt-pos-exit",
        underlying="SPY",
        contract_symbol="SPY250417C00100000",
        option_type="call",
        qty=2,
        entry_price=5.0,
        entry_time=datetime.now(UTC),
        expiration=datetime(2026, 4, 17, tzinfo=UTC).date(),
        high_water_price=6.0,
        stop_loss_price=4.0,
        take_profit_price=7.0,
        dte_at_entry=20,
    )
    quote = PriceReference(
        symbol=option_position.contract_symbol,
        bid=6.5,
        ask=6.7,
        mid=6.6,
        as_of=datetime.now(UTC),
    )
    exit_signal = SimpleNamespace(reason="profit_target")
    fake_strategy = SimpleNamespace(
        build_exit_order=lambda pos, current_bid, reason: OptionsOrder(
            underlying=pos.underlying,
            contract_symbol=pos.contract_symbol,
            option_type=pos.option_type,
            side="sell",
            qty=pos.qty,
            limit_price=6.45,
            stage="exit",
            source="options_exit",
        )
    )

    log_option_trade = MagicMock()
    close_option_position = MagicMock()
    update_option_position = MagicMock()

    monkeypatch.setattr("sauce.loop.OPTIONS_STRATEGY", fake_strategy)
    monkeypatch.setattr(
        "sauce.loop.get_option_positions",
        MagicMock(return_value=[{"symbol": option_position.contract_symbol, "qty": "2"}]),
    )
    monkeypatch.setattr(
        "sauce.loop.get_option_quotes",
        MagicMock(return_value={option_position.contract_symbol: quote}),
    )
    monkeypatch.setattr("sauce.loop._fetch_indicators", MagicMock(return_value=None))
    monkeypatch.setattr("sauce.loop.check_options_position", MagicMock(return_value=exit_signal))
    monkeypatch.setattr(
        "sauce.loop.place_option_order",
        MagicMock(return_value={"status": "partially_filled", "filled_qty": "1", "filled_avg_price": "6.3"}),
    )
    monkeypatch.setattr("sauce.loop.log_option_trade", log_option_trade)
    monkeypatch.setattr("sauce.loop.close_option_position", close_option_position)
    monkeypatch.setattr("sauce.loop.update_option_position", update_option_position)
    monkeypatch.setattr("sauce.loop.upsert_daily_stats", MagicMock())

    await _scan_option_exits([option_position])

    assert log_option_trade.call_count == 1
    traded_position, traded_exit_price, traded_trigger = log_option_trade.call_args.args
    assert traded_position.qty == 1
    assert traded_exit_price == pytest.approx(6.3)
    assert traded_trigger == "profit_target"
    assert close_option_position.call_count == 0
    assert update_option_position.call_count >= 1
    persisted = update_option_position.call_args.args[0]
    assert persisted.qty == 1
