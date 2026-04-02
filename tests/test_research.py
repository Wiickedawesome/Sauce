from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from sauce.core.schemas import Indicators, Order
from sauce.morning_brief import infer_intraday_regime
from sauce.performance import TradePerformanceRecord, summarize_performance
from sauce.research.backtest import BacktestConfig, backtest_strategy
from sauce.research.universe import get_equity_universe_as_of
from sauce.research.walk_forward import walk_forward_optimize
from sauce.strategy import ExitPlan, Position, SignalResult, TierParams


class DummyStrategy:
    name = "dummy"
    instruments = ["BTC/USD"]

    def eligible(self, instrument: str, regime: str) -> bool:
        return True

    def score(self, indicators: Indicators, instrument: str, regime: str, current_price: float) -> SignalResult:
        return SignalResult(
            symbol=instrument,
            side="buy",
            score=100,
            threshold=1,
            fired=True,
            rsi_14=indicators.rsi_14,
            macd_hist=indicators.macd_histogram,
            bb_pct=None,
            volume_ratio=indicators.volume_ratio,
            regime=regime,
            strategy_name=self.name,
        )

    def build_order(self, signal: SignalResult, account: dict[str, str], tier: TierParams) -> Order:
        ask = float(account["_ask"])
        return Order(
            symbol=signal.symbol,
            side="buy",
            qty=1.0,
            order_type="limit",
            time_in_force="gtc",
            limit_price=ask,
            stop_loss_price=ask * 0.95,
            take_profit_price=ask * 1.02,
            as_of=datetime.now(UTC),
            prompt_version="test",
            source="execution",
        )

    def build_exit_plan(self, position: Position, tier: TierParams) -> ExitPlan:
        return ExitPlan(
            stop_loss_pct=0.05,
            trail_activation_pct=0.50,
            trail_pct=0.05,
            profit_target_pct=0.02,
            rsi_exhaustion_threshold=90,
            max_hold_hours=240,
            time_stop_min_gain=-1.0,
        )


class ProfiledDummyStrategy(DummyStrategy):
    name = "dummy_profiled"

    def __init__(self, profile: dict[str, float]) -> None:
        self.profile = profile

    def build_exit_plan(self, position: Position, tier: TierParams) -> ExitPlan:
        return ExitPlan(
            stop_loss_pct=0.05,
            trail_activation_pct=0.50,
            trail_pct=0.05,
            profit_target_pct=float(self.profile["profit_target_pct"]),
            rsi_exhaustion_threshold=90,
            max_hold_hours=240,
            time_stop_min_gain=-1.0,
        )


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC")
    closes = [100 + (idx * 0.2) for idx in range(len(index))]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value + 0.5 for value in closes],
            "low": [value - 0.5 for value in closes],
            "close": closes,
            "volume": [1_000 + (idx * 10) for idx in range(len(index))],
        },
        index=index,
    )


def test_infer_intraday_regime_bullish() -> None:
    regime, reasoning = infer_intraday_regime(0.03, 0.025, 0.015, 14.0, 65.0)
    assert regime == "bullish"
    assert "heuristic_score" in reasoning


def test_summarize_performance_reports_net_metrics() -> None:
    trades = [
        TradePerformanceRecord(
            symbol="BTC/USD",
            asset_class="crypto",
            strategy_name="test",
            entry_time=datetime.now(UTC) - timedelta(days=2),
            exit_time=datetime.now(UTC) - timedelta(days=1),
            entry_price=100.0,
            exit_price=110.0,
            qty=1.0,
            gross_pnl=10.0,
            realized_pnl=8.0,
            fees_paid=1.0,
            slippage_paid=1.0,
            hold_hours=12.0,
            exit_trigger="target",
        ),
        TradePerformanceRecord(
            symbol="BTC/USD",
            asset_class="crypto",
            strategy_name="test",
            entry_time=datetime.now(UTC) - timedelta(days=1),
            exit_time=datetime.now(UTC),
            entry_price=110.0,
            exit_price=105.0,
            qty=1.0,
            gross_pnl=-5.0,
            realized_pnl=-6.0,
            fees_paid=0.5,
            slippage_paid=0.5,
            hold_hours=6.0,
            exit_trigger="stop",
        ),
    ]

    summary = summarize_performance(trades, starting_equity=1000.0, risk_free_rate=0.0)

    assert summary.trade_count == 2
    assert summary.gross_pnl == 5.0
    assert summary.net_pnl == 2.0
    assert summary.fees_paid == 1.5
    assert summary.slippage_paid == 1.5
    assert summary.win_rate == 0.5


def test_backtest_strategy_runs_end_to_end() -> None:
    result = backtest_strategy(
        DummyStrategy(),
        "BTC/USD",
        _frame(),
        is_crypto=True,
        config=BacktestConfig(starting_equity=10_000.0, warmup_bars=60, risk_free_rate=0.0),
    )

    assert result.trade_count >= 1
    assert result.metrics.trade_count == result.trade_count


def test_walk_forward_optimize_returns_best_profile() -> None:
    base_profile = {"profit_target_pct": 0.02}
    result = walk_forward_optimize(
        lambda profile: ProfiledDummyStrategy(profile),
        "BTC/USD",
        _frame(),
        is_crypto=True,
        base_profile=base_profile,
        parameter_grid={"profit_target_pct": [0.01, 0.02, 0.03]},
        train_bars=60,
        test_bars=30,
        step_bars=30,
        config=BacktestConfig(starting_equity=10_000.0, warmup_bars=20, risk_free_rate=0.0),
    )

    assert result.best_profile["profit_target_pct"] in {0.01, 0.02, 0.03}
    assert result.windows


def test_get_equity_universe_as_of(tmp_path) -> None:
    universe_file = tmp_path / "equity_universe_history.json"
    universe_file.write_text(
        '{"snapshots":[{"effective_date":"2026-01-01","symbols":["AAPL","MSFT"]},{"effective_date":"2026-02-01","symbols":["NVDA"]}]}'
    )

    universe = get_equity_universe_as_of(datetime(2026, 2, 15, tzinfo=UTC).date(), str(universe_file))

    assert universe == ["NVDA"]
