"""Tests for sauce.backtest — bar-replay backtesting engine."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sauce.backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestTrade,
    ExitReason,
    TradeDirection,
    run_backtest,
)
from sauce.backtest.engine import (
    _Position,
    _apply_slippage,
    _bar_timestamp,
    _close_position,
    _compute_metrics,
    _is_crypto,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ohlcv(n: int = 200, trend: float = 0.05) -> pd.DataFrame:
    """Generate synthetic OHLCV with a mild uptrend + noise."""
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(trend, 0.3, n))
    close = np.maximum(close, 10.0)  # floor at 10
    idx = pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "open": close - rng.uniform(0, 0.2, n),
            "high": close + rng.uniform(0.1, 0.5, n),
            "low": close - rng.uniform(0.1, 0.5, n),
            "close": close,
            "volume": rng.integers(1000, 10000, size=n).astype(float),
        },
        index=idx,
    )


def _flat_ohlcv(n: int = 200, price: float = 100.0) -> pd.DataFrame:
    """Flat-price OHLCV — no setups should trigger."""
    idx = pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC")
    rng = np.random.default_rng(99)
    return pd.DataFrame(
        {
            "open": np.full(n, price),
            "high": np.full(n, price + 0.01),
            "low": np.full(n, price - 0.01),
            "close": np.full(n, price),
            "volume": rng.integers(1000, 5000, size=n).astype(float),
        },
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# UNIT: helper functions
# ═══════════════════════════════════════════════════════════════════════════

class TestIsCrypto:
    def test_crypto(self):
        assert _is_crypto("BTC/USD") is True

    def test_equity(self):
        assert _is_crypto("AAPL") is False


class TestApplySlippage:
    def test_long_slips_up(self):
        result = _apply_slippage(100.0, TradeDirection.LONG, 0.001)
        assert result == pytest.approx(100.1)

    def test_short_slips_down(self):
        result = _apply_slippage(100.0, TradeDirection.SHORT, 0.001)
        assert result == pytest.approx(99.9)


class TestBarTimestamp:
    def test_extracts_utc_datetime(self):
        df = _ohlcv(10)
        ts = _bar_timestamp(df, 0)
        assert isinstance(ts, datetime)
        assert ts.tzinfo is not None

    def test_naive_index_gets_utc(self):
        idx = pd.date_range("2025-01-01", periods=5, freq="30min")
        df = pd.DataFrame(
            {"open": [1]*5, "high": [2]*5, "low": [0]*5, "close": [1]*5, "volume": [100]*5},
            index=idx,
        )
        ts = _bar_timestamp(df, 0)
        assert ts.tzinfo == timezone.utc


class TestClosePosition:
    def test_long_profit(self):
        cfg = BacktestConfig(slippage_pct=0.0, commission_per_trade=0.0)
        pos = _Position(
            symbol="AAPL", direction=TradeDirection.LONG, setup_type="test",
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            entry_price=100.0, qty=10.0, stop_price=95.0, target_price=110.0,
            confidence=0.8, entry_bar_idx=0,
        )
        trade = _close_position(
            pos, 110.0,
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            ExitReason.PROFIT_TARGET, 5, cfg,
        )
        assert trade.pnl == pytest.approx(100.0)  # (110-100)*10
        assert trade.bars_held == 5
        assert trade.exit_reason == ExitReason.PROFIT_TARGET

    def test_long_loss(self):
        cfg = BacktestConfig(slippage_pct=0.0, commission_per_trade=0.0)
        pos = _Position(
            symbol="AAPL", direction=TradeDirection.LONG, setup_type="test",
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            entry_price=100.0, qty=10.0, stop_price=95.0, target_price=110.0,
            confidence=0.8, entry_bar_idx=0,
        )
        trade = _close_position(
            pos, 95.0,
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            ExitReason.STOP_LOSS, 3, cfg,
        )
        assert trade.pnl == pytest.approx(-50.0)  # (95-100)*10
        assert trade.exit_reason == ExitReason.STOP_LOSS

    def test_commission_deducted(self):
        cfg = BacktestConfig(slippage_pct=0.0, commission_per_trade=5.0)
        pos = _Position(
            symbol="AAPL", direction=TradeDirection.LONG, setup_type="test",
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            entry_price=100.0, qty=10.0, stop_price=95.0, target_price=110.0,
            confidence=0.8, entry_bar_idx=0,
        )
        trade = _close_position(
            pos, 110.0,
            datetime(2025, 1, 2, tzinfo=timezone.utc),
            ExitReason.PROFIT_TARGET, 5, cfg,
        )
        # raw pnl = 100, minus 2*5 commission = 90
        assert trade.pnl == pytest.approx(90.0)


# ═══════════════════════════════════════════════════════════════════════════
# UNIT: _compute_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def test_no_trades(self):
        result = BacktestResult(symbol="X", config=BacktestConfig())
        _compute_metrics(result)
        assert result.total_trades == 0
        assert result.win_rate == 0.0

    def test_with_trades(self):
        result = BacktestResult(
            symbol="X",
            config=BacktestConfig(),
            trades=[
                BacktestTrade(
                    symbol="X", direction=TradeDirection.LONG, setup_type="t",
                    entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
                    exit_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
                    entry_price=100.0, exit_price=110.0, qty=10, pnl=100.0,
                    pnl_pct=0.10, exit_reason=ExitReason.PROFIT_TARGET,
                    confidence=0.8, bars_held=5,
                ),
                BacktestTrade(
                    symbol="X", direction=TradeDirection.LONG, setup_type="t",
                    entry_time=datetime(2025, 1, 3, tzinfo=timezone.utc),
                    exit_time=datetime(2025, 1, 4, tzinfo=timezone.utc),
                    entry_price=100.0, exit_price=95.0, qty=10, pnl=-50.0,
                    pnl_pct=-0.05, exit_reason=ExitReason.STOP_LOSS,
                    confidence=0.7, bars_held=3,
                ),
            ],
            equity_curve=[100000, 100050, 100100, 100050, 100000],
        )
        _compute_metrics(result)
        assert result.total_trades == 2
        assert result.winning_trades == 1
        assert result.losing_trades == 1
        assert result.win_rate == pytest.approx(0.5)
        assert result.profit_factor == pytest.approx(2.0)  # 100/50
        assert result.avg_bars_held == pytest.approx(4.0)

    def test_equity_curve_metrics(self):
        result = BacktestResult(
            symbol="X",
            config=BacktestConfig(),
            equity_curve=[100000, 101000, 102000, 101500, 103000],
        )
        _compute_metrics(result)
        assert result.total_return == pytest.approx(0.03)
        assert result.max_drawdown < 0  # should be negative (drawdown from 102000 to 101500)


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION: run_backtest
# ═══════════════════════════════════════════════════════════════════════════

class TestRunBacktest:
    def test_insufficient_data(self):
        df = _ohlcv(30)  # less than lookback_bars + 10
        result = run_backtest("SPY", df, regime="TRENDING_UP")
        assert result.total_trades == 0
        assert result.equity_curve == []

    def test_invalid_regime_raises(self):
        df = _ohlcv(200)
        with pytest.raises(ValueError, match="not a valid"):
            run_backtest("SPY", df, regime="INVALID_REGIME")

    def test_flat_market_no_trades(self):
        """Flat price → indicators won't trigger setups → no trades."""
        df = _flat_ohlcv(200)
        result = run_backtest("SPY", df, regime="TRENDING_UP")
        assert result.total_trades == 0
        # Equity curve should be flat at initial capital
        if result.equity_curve:
            assert all(v == pytest.approx(100_000.0) for v in result.equity_curve)

    def test_returns_backtest_result(self):
        df = _ohlcv(200)
        result = run_backtest("SPY", df, regime="TRENDING_UP")
        assert isinstance(result, BacktestResult)
        assert result.symbol == "SPY"
        assert len(result.equity_curve) > 0
        assert len(result.timestamps) == len(result.equity_curve)

    def test_equity_curve_starts_at_initial_capital(self):
        df = _ohlcv(200)
        cfg = BacktestConfig(initial_capital=50_000.0)
        result = run_backtest("SPY", df, regime="TRENDING_UP", config=cfg)
        if result.equity_curve:
            assert result.equity_curve[0] == pytest.approx(50_000.0)

    def test_custom_config(self):
        df = _ohlcv(200)
        cfg = BacktestConfig(
            initial_capital=200_000.0,
            max_position_pct=0.05,
            stop_loss_atr_multiple=1.5,
            profit_target_atr_multiple=4.0,
        )
        result = run_backtest("SPY", df, regime="TRENDING_UP", config=cfg)
        assert result.config.initial_capital == 200_000.0
        assert result.config.max_position_pct == 0.05

    def test_trades_have_valid_fields(self):
        """If any trades generated, verify field consistency."""
        df = _ohlcv(300)
        result = run_backtest("SPY", df, regime="TRENDING_UP")
        for trade in result.trades:
            assert trade.qty > 0
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.bars_held >= 0
            assert trade.exit_reason in ExitReason
            assert 0.0 <= trade.confidence <= 1.0

    def test_end_of_data_closes_position(self):
        """Force a setup to pass, then run to end of data."""
        from sauce.core.schemas import Indicators, SetupResult

        # Make scan_setups always return a passing setup
        fake_setup = SetupResult(
            setup_type="equity_trend_pullback",
            symbol="SPY",
            score=80.0,
            min_score=65.0,
            passed=True,
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        df = _ohlcv(100)
        cfg = BacktestConfig(lookback_bars=20)

        with patch("sauce.core.setups.scan_setups", return_value=[fake_setup]):
            # Also ensure compute_all returns an ATR so stops are valid
            ind = Indicators(atr_14=1.0, sma_20=100.0, sma_50=99.0, rsi_14=50.0)
            with patch("sauce.backtest.engine.compute_all", return_value=ind):
                result = run_backtest("SPY", df, regime="TRENDING_UP", config=cfg)

        # Should have at least one trade, and the last should be END_OF_DATA
        # (unless stop/target triggered first)
        assert result.total_trades >= 1

    def test_stop_loss_triggers(self):
        """Force entry then drop price below stop → stop loss exit."""
        from sauce.core.schemas import Indicators, SetupResult

        fake_setup = SetupResult(
            setup_type="equity_trend_pullback",
            symbol="SPY",
            score=80.0,
            min_score=65.0,
            passed=True,
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        # Build data: first 25 bars flat, then sharp drop
        n = 50
        idx = pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC")
        close = np.full(n, 100.0)
        close[26:] = 90.0  # massive drop after entry
        high = close + 0.5
        low = close.copy()
        low[26:] = 85.0  # low goes well below stop
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close,
             "volume": np.full(n, 5000.0)},
            index=idx,
        )

        cfg = BacktestConfig(
            lookback_bars=20,
            stop_loss_atr_multiple=2.0,
            slippage_pct=0.0,
            commission_per_trade=0.0,
        )
        ind = Indicators(atr_14=2.0, sma_20=100.0, sma_50=99.0, rsi_14=50.0)

        call_count = 0

        def mock_scan(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Only pass setup on first call to get one entry
            if call_count == 1:
                return [fake_setup]
            return []

        with (
            patch("sauce.core.setups.scan_setups", side_effect=mock_scan),
            patch("sauce.backtest.engine.compute_all", return_value=ind),
        ):
            result = run_backtest("SPY", df, regime="TRENDING_UP", config=cfg)

        assert result.total_trades >= 1
        stop_trades = [t for t in result.trades if t.exit_reason == ExitReason.STOP_LOSS]
        assert len(stop_trades) >= 1

    def test_profit_target_triggers(self):
        """Force entry then price rises above target → profit target exit."""
        from sauce.core.schemas import Indicators, SetupResult

        fake_setup = SetupResult(
            setup_type="equity_trend_pullback",
            symbol="SPY",
            score=80.0,
            min_score=65.0,
            passed=True,
            as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )

        n = 50
        idx = pd.date_range("2025-01-01", periods=n, freq="30min", tz="UTC")
        close = np.full(n, 100.0)
        close[26:] = 120.0  # big jump
        high = close.copy()
        high[26:] = 125.0
        low = close - 0.5
        df = pd.DataFrame(
            {"open": close, "high": high, "low": low, "close": close,
             "volume": np.full(n, 5000.0)},
            index=idx,
        )

        cfg = BacktestConfig(
            lookback_bars=20,
            stop_loss_atr_multiple=2.0,
            profit_target_atr_multiple=3.0,
            slippage_pct=0.0,
            commission_per_trade=0.0,
        )
        ind = Indicators(atr_14=2.0, sma_20=100.0, sma_50=99.0, rsi_14=50.0)

        call_count = 0

        def mock_scan(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [fake_setup]
            return []

        with (
            patch("sauce.core.setups.scan_setups", side_effect=mock_scan),
            patch("sauce.backtest.engine.compute_all", return_value=ind),
        ):
            result = run_backtest("SPY", df, regime="TRENDING_UP", config=cfg)

        assert result.total_trades >= 1
        profit_trades = [t for t in result.trades if t.exit_reason == ExitReason.PROFIT_TARGET]
        assert len(profit_trades) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# MODEL tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBacktestModels:
    def test_trade_frozen(self):
        t = BacktestTrade(
            symbol="X", direction=TradeDirection.LONG, setup_type="t",
            entry_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
            exit_time=datetime(2025, 1, 2, tzinfo=timezone.utc),
            entry_price=100.0, exit_price=110.0, qty=10, pnl=100.0,
            pnl_pct=0.10, exit_reason=ExitReason.PROFIT_TARGET,
            confidence=0.8, bars_held=5,
        )
        with pytest.raises(AttributeError):
            t.pnl = 999  # type: ignore[misc]

    def test_config_defaults(self):
        cfg = BacktestConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.lookback_bars == 60
        assert cfg.slippage_pct == 0.001

    def test_result_defaults(self):
        r = BacktestResult(symbol="X", config=BacktestConfig())
        assert r.trades == []
        assert r.total_trades == 0
        assert r.sharpe_ratio is None

    def test_exit_reason_enum(self):
        assert ExitReason.STOP_LOSS.value == "stop_loss"
        assert ExitReason.PROFIT_TARGET.value == "profit_target"
        assert ExitReason.END_OF_DATA.value == "end_of_data"

    def test_trade_direction_enum(self):
        assert TradeDirection.LONG.value == "long"
        assert TradeDirection.SHORT.value == "short"
