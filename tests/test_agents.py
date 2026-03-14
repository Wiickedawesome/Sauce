"""
tests/test_agents.py — Phase 5 agent tests.

All tests:
  - Mock all external calls (market_data.get_history, llm.call_claude).
  - Use a tmp_path SQLite DB (no real DB touched).
  - Never require real API keys.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from sauce.signals.timeframes import MultiTimeframeContext

# Patch target for fetch_multi_timeframe returning no analyses (no confidence adjustment).
_PATCH_NO_MTF = patch(
    "sauce.agents.research.fetch_multi_timeframe",
    return_value=MultiTimeframeContext(symbol="X"),
)

import pandas as pd
import pytest

from sauce.core.schemas import (
    AuditEvent,
    Evidence,
    Indicators,
    Order,
    PortfolioReview,
    PositionNote,
    PriceReference,
    RiskCheckResult,
    RiskChecks,
    Signal,
    SupervisorDecision,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

_NOW = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)


def _fresh_quote(
    symbol: str = "AAPL",
    bid: float = 99.9,
    ask: float = 100.1,
    mid: float = 100.0,
) -> PriceReference:
    return PriceReference(
        symbol=symbol,
        bid=bid,
        ask=ask,
        mid=mid,
        as_of=datetime.now(timezone.utc),
    )


def _stale_quote(symbol: str = "AAPL") -> PriceReference:
    """Quote timestamped well in the past."""
    return PriceReference(
        symbol=symbol,
        bid=99.0,
        ask=101.0,
        mid=100.0,
        as_of=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )


def _buy_signal(
    symbol: str = "AAPL",
    confidence: float = 0.8,
    atr_14: float | None = 1.0,
    quote: PriceReference | None = None,
) -> Signal:
    q = quote or _fresh_quote(symbol)
    return Signal(
        symbol=symbol,
        side="buy",
        confidence=confidence,
        evidence=Evidence(
            symbol=symbol,
            price_reference=q,
            indicators=Indicators(atr_14=atr_14),
            as_of=q.as_of,
        ),
        reasoning="Test signal",
        as_of=datetime.now(timezone.utc),
        prompt_version="test-v1",
    )


def _hold_signal(symbol: str = "AAPL") -> Signal:
    q = _fresh_quote(symbol)
    return Signal(
        symbol=symbol,
        side="hold",
        confidence=0.0,
        evidence=Evidence(
            symbol=symbol,
            price_reference=q,
            indicators=Indicators(),
            as_of=q.as_of,
        ),
        reasoning="Stub hold",
        as_of=datetime.now(timezone.utc),
        prompt_version="test-v1",
    )


def _approved_risk(symbol: str = "AAPL", qty: float = 5.0) -> RiskCheckResult:
    return RiskCheckResult(
        symbol=symbol,
        side="buy",
        veto=False,
        reason=None,
        qty=qty,
        checks=RiskChecks(
            max_position_pct_ok=True,
            max_exposure_ok=True,
            asset_class_ok=True,
            daily_loss_ok=True,
            volatility_ok=True,
            confidence_ok=True,
        ),
        as_of=datetime.now(timezone.utc),
        prompt_version="test-v1",
    )


def _vetoed_risk(symbol: str = "AAPL") -> RiskCheckResult:
    return RiskCheckResult(
        symbol=symbol,
        side="buy",
        veto=True,
        reason="Test veto",
        qty=None,
        checks=RiskChecks(
            max_position_pct_ok=False,
            max_exposure_ok=False,
            asset_class_ok=False,
            daily_loss_ok=True,
            volatility_ok=True,
            confidence_ok=False,
        ),
        as_of=datetime.now(timezone.utc),
        prompt_version="test-v1",
    )


def _limit_order(
    symbol: str = "AAPL",
    side: str = "buy",
    qty: float = 5.0,
    limit_price: float = 101.0,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,  # type: ignore[arg-type]
        qty=qty,
        order_type="limit",
        time_in_force="day",
        limit_price=limit_price,
        stop_price=None,
        as_of=datetime.now(timezone.utc),
        prompt_version="test-v1",
    )


def _make_ohlcv(rows: int = 60) -> pd.DataFrame:
    """Make a minimal OHLCV DataFrame with {rows} rows."""
    return pd.DataFrame(
        {
            "open": [100.0] * rows,
            "high": [102.0] * rows,
            "low": [98.0] * rows,
            "close": [101.0] * rows,
            "volume": [1_000_000.0] * rows,
        }
    )


def _settings_patch(tmp_path: Path) -> dict:
    """Return a dict of settings fields for patching get_settings()."""
    db_file = tmp_path / "data" / "sauce.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    session_mem_db = tmp_path / "data" / "session_memory.db"
    strategic_mem_db = tmp_path / "data" / "strategic_memory.db"
    return {
        "db_path": db_file,
        "session_memory_db_path": str(session_mem_db),
        "strategic_memory_db_path": str(strategic_mem_db),
        "prompt_version": "test-v1",
        "min_confidence": 0.5,
        "max_position_pct": 0.05,
        "max_portfolio_exposure": 0.5,
        "max_daily_loss_pct": 0.02,
        "max_crypto_allocation_pct": 0.40,
        "max_equity_allocation_pct": 0.70,
        "data_ttl_seconds": 300,
        "max_price_deviation": 0.01,
        "trading_universe": ["AAPL", "MSFT"],
        "alpaca_paper": True,
        "max_atr_ratio": 0.05,
        "allow_no_atr": False,
        "stop_loss_atr_multiple": 2.0,
        "profit_target_atr_multiple": 3.0,
        "over_concentration_multiplier": 2.0,
        "max_limit_price_premium": 0.001,
        "max_spread_pct": 0.005,
        "alert_webhook_url": "",
    }


def _mock_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    for k, v in _settings_patch(tmp_path).items():
        setattr(s, k, v)
    return s


def _passing_mock_setup() -> MagicMock:
    """Return a mock SetupResult (passed=True) for tests that need Claude called.

    Added when scan_setups was integrated into research.run(): symbols not in
    any setup (like 'AAPL') now return safe-hold before the LLM call.  Tests
    that want to exercise the Claude path must mock scan_setups out.
    """
    m = MagicMock(passed=True, score=75.0)
    m.model_dump.return_value = {
        "setup_type": "equity_trend_pullback",
        "passed": True,
        "score": 75.0,
        "evidence_narrative": "Mock passing setup for unit test.",
    }
    return m


# ─── Research agent ───────────────────────────────────────────────────────────

class TestResearchAgent:
    def test_returns_hold_on_empty_history(self, tmp_path: Path) -> None:
        """Empty DataFrame → safe hold."""
        from sauce.agents import research

        settings = _mock_settings(tmp_path)
        with (
            patch("sauce.agents.research.get_settings", return_value=settings),
            patch("sauce.agents.research.market_data") as md,
            patch("sauce.agents.research.log_event"),
        ):
            md.get_history.return_value = pd.DataFrame()
            sig = asyncio.run(
                research.run("AAPL", _fresh_quote(), loop_id="test-loop")
            )

        assert sig.side == "hold"
        assert sig.confidence == 0.0

    def test_returns_hold_on_insufficient_bars(self, tmp_path: Path) -> None:
        """30 rows < 55 required → hold."""
        from sauce.agents import research

        settings = _mock_settings(tmp_path)
        with (
            patch("sauce.agents.research.get_settings", return_value=settings),
            patch("sauce.agents.research.market_data") as md,
            patch("sauce.agents.research.log_event"),
        ):
            md.get_history.return_value = _make_ohlcv(30)
            sig = asyncio.run(
                research.run("AAPL", _fresh_quote(), loop_id="test-loop")
            )

        assert sig.side == "hold"

    def test_returns_hold_on_market_data_error(self, tmp_path: Path) -> None:
        """MarketDataError → hold."""
        from sauce.adapters.market_data import MarketDataError
        from sauce.agents import research

        settings = _mock_settings(tmp_path)
        with (
            patch("sauce.agents.research.get_settings", return_value=settings),
            patch("sauce.agents.research.market_data") as md,
            patch("sauce.agents.research.log_event"),
        ):
            md.MarketDataError = MarketDataError
            md.get_history.side_effect = MarketDataError("API down")
            sig = asyncio.run(
                research.run("AAPL", _fresh_quote(), loop_id="test-loop")
            )

        assert sig.side == "hold"
        assert "market data unavailable" in sig.reasoning

    def test_calls_claude_and_parses_signal(self, tmp_path: Path) -> None:
        """Full flow: 60 rows + Claude returns buy at 0.85 → Signal(side=buy)."""
        from sauce.agents import research

        settings = _mock_settings(tmp_path)
        claude_response = json.dumps(
            {"side": "buy", "confidence": 0.85, "reasoning": "Strong uptrend."}
        )
        with (
            patch("sauce.agents.research.get_settings", return_value=settings),
            patch("sauce.agents.research.market_data") as md,
            patch("sauce.agents.research.llm") as mock_llm,
            patch("sauce.agents.research.scan_setups", return_value=[_passing_mock_setup()]),
            patch("sauce.agents.research.log_event"),
            _PATCH_NO_MTF,
        ):
            md.get_history.return_value = _make_ohlcv(60)
            mock_llm.call_claude = AsyncMock(return_value=claude_response)
            mock_llm.LLMError = Exception  # won't be raised

            sig = asyncio.run(
                research.run("AAPL", _fresh_quote(), loop_id="test-loop")
            )

        assert sig.side == "buy"
        assert abs(sig.confidence - 0.85) < 1e-9
        assert sig.reasoning == "Strong uptrend."
        assert sig.symbol == "AAPL"

    def test_returns_hold_on_llm_error(self, tmp_path: Path) -> None:
        """LLMError → safe hold."""
        from sauce.agents import research

        class _FakeLLMError(Exception):
            pass

        settings = _mock_settings(tmp_path)
        with (
            patch("sauce.agents.research.get_settings", return_value=settings),
            patch("sauce.agents.research.market_data") as md,
            patch("sauce.agents.research.llm") as mock_llm,
            patch("sauce.agents.research.scan_setups", return_value=[_passing_mock_setup()]),
            patch("sauce.agents.research.log_event"),
            _PATCH_NO_MTF,
        ):
            md.get_history.return_value = _make_ohlcv(60)
            mock_llm.LLMError = _FakeLLMError
            mock_llm.call_claude = AsyncMock(side_effect=_FakeLLMError("timeout"))

            sig = asyncio.run(
                research.run("AAPL", _fresh_quote(), loop_id="test-loop")
            )

        assert sig.side == "hold"
        assert "LLM error" in sig.reasoning

    def test_returns_hold_on_bad_json(self, tmp_path: Path) -> None:
        """Corrupted JSON → safe hold."""
        from sauce.agents import research

        settings = _mock_settings(tmp_path)
        with (
            patch("sauce.agents.research.get_settings", return_value=settings),
            patch("sauce.agents.research.market_data") as md,
            patch("sauce.agents.research.llm") as mock_llm,
            patch("sauce.agents.research.scan_setups", return_value=[_passing_mock_setup()]),
            patch("sauce.agents.research.log_event"),
            _PATCH_NO_MTF,
        ):
            md.get_history.return_value = _make_ohlcv(60)
            mock_llm.LLMError = Exception
            mock_llm.call_claude = AsyncMock(return_value="NOT JSON {{{")

            sig = asyncio.run(
                research.run("AAPL", _fresh_quote(), loop_id="test-loop")
            )

        assert sig.side == "hold"

    def test_confidence_clamped_to_1(self, tmp_path: Path) -> None:
        """Claude returns confidence > 1.0 → clamped to 1.0."""
        from sauce.agents import research

        settings = _mock_settings(tmp_path)
        claude_response = json.dumps(
            {"side": "sell", "confidence": 99.0, "reasoning": "Extreme."}
        )
        with (
            patch("sauce.agents.research.get_settings", return_value=settings),
            patch("sauce.agents.research.market_data") as md,
            patch("sauce.agents.research.llm") as mock_llm,
            patch("sauce.agents.research.scan_setups", return_value=[_passing_mock_setup()]),
            patch("sauce.agents.research.log_event"),
            _PATCH_NO_MTF,
        ):
            md.get_history.return_value = _make_ohlcv(60)
            mock_llm.LLMError = Exception
            mock_llm.call_claude = AsyncMock(return_value=claude_response)

            sig = asyncio.run(
                research.run("AAPL", _fresh_quote(), loop_id="test-loop")
            )

        assert sig.confidence == 1.0


# ─── Risk agent ───────────────────────────────────────────────────────────────

class TestRiskAgent:
    def test_vetoes_hold_signal(self, tmp_path: Path) -> None:
        from sauce.agents import risk

        settings = _mock_settings(tmp_path)
        account = {"equity": "10000", "last_equity": "10000", "buying_power": "5000"}
        with (
            patch("sauce.agents.risk.get_settings", return_value=settings),
            patch("sauce.agents.risk.log_event"),
        ):
            result = asyncio.run(
                risk.run(_hold_signal(), account, [], loop_id="test-loop")
            )

        assert result.veto is True
        assert "hold" in result.reason.lower()

    def test_vetoes_low_confidence(self, tmp_path: Path) -> None:
        from sauce.agents import risk

        settings = _mock_settings(tmp_path)
        # min_confidence=0.5, signal has 0.3
        sig = _buy_signal(confidence=0.3)
        account = {"equity": "10000", "last_equity": "10000", "buying_power": "5000"}
        with (
            patch("sauce.agents.risk.get_settings", return_value=settings),
            patch("sauce.agents.risk.log_event"),
        ):
            result = asyncio.run(risk.run(sig, account, [], loop_id="test-loop"))

        assert result.veto is True
        assert result.checks.confidence_ok is False

    def test_approves_valid_signal(self, tmp_path: Path) -> None:
        """confidence=0.8, equity=10000, mid=100 → approved with qty ≈5."""
        from sauce.agents import risk

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(confidence=0.8, atr_14=1.0)
        account = {
            "equity": "10000",
            "last_equity": "10000",
            "buying_power": "10000",
        }
        with (
            patch("sauce.agents.risk.get_settings", return_value=settings),
            patch("sauce.agents.risk.log_event"),
        ):
            result = asyncio.run(risk.run(sig, account, [], loop_id="test-loop"))

        assert result.veto is False
        assert result.qty is not None
        assert result.qty > 0
        # max_position_pct=0.05; equity=10000 → max $500; mid=100 → qty≈5
        assert result.qty == pytest.approx(5.0, abs=0.01)

    def test_vetoes_at_max_position(self, tmp_path: Path) -> None:
        """Existing position value already at max → veto."""
        from sauce.agents import risk

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(confidence=0.8)
        # equity=10000, max_position_pct=0.05 → max $500
        # existing = 501 → over limit
        account = {
            "equity": "10000",
            "last_equity": "10000",
            "buying_power": "10000",
        }
        existing_position = {"symbol": "AAPL", "qty": "5.01", "market_value": "501"}
        with (
            patch("sauce.agents.risk.get_settings", return_value=settings),
            patch("sauce.agents.risk.log_event"),
        ):
            result = asyncio.run(
                risk.run(sig, account, [existing_position], loop_id="test-loop")
            )

        assert result.veto is True
        assert result.checks.max_position_pct_ok is False

    def test_vetoes_high_volatility(self, tmp_path: Path) -> None:
        """ATR/price > 5% → veto for volatility."""
        from sauce.agents import risk

        settings = _mock_settings(tmp_path)
        # ATR=6, mid=100 → ratio=6% > 5%
        sig = _buy_signal(confidence=0.8, atr_14=6.0)
        account = {
            "equity": "10000",
            "last_equity": "10000",
            "buying_power": "10000",
        }
        with (
            patch("sauce.agents.risk.get_settings", return_value=settings),
            patch("sauce.agents.risk.log_event"),
        ):
            result = asyncio.run(risk.run(sig, account, [], loop_id="test-loop"))

        assert result.veto is True
        assert result.checks.volatility_ok is False

    def test_vetoes_daily_loss_breach(self, tmp_path: Path) -> None:
        """equity < last_equity * (1 - max_daily_loss_pct) → veto."""
        from sauce.agents import risk

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(confidence=0.8)
        # daily loss = (9700 - 10000) / 10000 = -3% > max 2%
        account = {
            "equity": "9700",
            "last_equity": "10000",
            "buying_power": "10000",
        }
        with (
            patch("sauce.agents.risk.get_settings", return_value=settings),
            patch("sauce.agents.risk.log_event"),
        ):
            result = asyncio.run(risk.run(sig, account, [], loop_id="test-loop"))

        assert result.veto is True
        assert result.checks.daily_loss_ok is False


# ─── Execution agent ──────────────────────────────────────────────────────────

class TestExecutionAgent:
    def test_returns_none_on_stale_quote(self, tmp_path: Path) -> None:
        from sauce.agents import execution

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(confidence=0.8)
        risk_r = _approved_risk()
        stale = _stale_quote()

        with (
            patch("sauce.agents.execution.get_settings", return_value=settings),
            patch("sauce.agents.execution.log_event"),
            patch("sauce.agents.execution.is_data_fresh", return_value=False),
        ):
            result = asyncio.run(
                execution.run(sig, risk_r, stale, loop_id="test-loop")
            )

        assert result is None

    def test_returns_none_on_veto(self, tmp_path: Path) -> None:
        from sauce.agents import execution

        settings = _mock_settings(tmp_path)
        sig = _buy_signal()
        risk_r = _vetoed_risk()

        with (
            patch("sauce.agents.execution.get_settings", return_value=settings),
            patch("sauce.agents.execution.log_event"),
        ):
            result = asyncio.run(
                execution.run(sig, risk_r, _fresh_quote(), loop_id="test-loop")
            )

        assert result is None

    def test_returns_none_on_price_deviation(self, tmp_path: Path) -> None:
        """Quote moved 2% from evidence mid → abort."""
        from sauce.agents import execution

        settings = _mock_settings(tmp_path)
        # evidence mid=100, live mid=103 → 3% deviation > 1% max
        sig = _buy_signal(quote=_fresh_quote(mid=100.0))
        risk_r = _approved_risk()
        live_quote = _fresh_quote(mid=103.0)

        with (
            patch("sauce.agents.execution.get_settings", return_value=settings),
            patch("sauce.agents.execution.log_event"),
            patch("sauce.agents.execution.is_data_fresh", return_value=True),
        ):
            result = asyncio.run(
                execution.run(sig, risk_r, live_quote, loop_id="test-loop")
            )

        assert result is None

    def test_deterministic_buy_order(self, tmp_path: Path) -> None:
        """Full flow: all checks pass → deterministic limit order for buy."""
        from sauce.agents import execution

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(confidence=0.8, quote=_fresh_quote(mid=100.0))
        risk_r = _approved_risk(qty=5.0)
        live_quote = _fresh_quote(bid=99.5, ask=100.5, mid=100.0)

        with (
            patch("sauce.agents.execution.get_settings", return_value=settings),
            patch("sauce.agents.execution.log_event"),
            patch("sauce.agents.execution.is_data_fresh", return_value=True),
        ):
            result = asyncio.run(
                execution.run(sig, risk_r, live_quote, loop_id="test-loop")
            )

        assert result is not None
        assert isinstance(result, Order)
        assert result.order_type == "limit"
        assert result.qty == pytest.approx(5.0)
        # buy limit = ask * (1 + 0.0005)
        expected_limit = round(100.5 * 1.0005, 4)
        assert result.limit_price == pytest.approx(expected_limit)
        assert result.time_in_force == "day"
        assert result.stop_price is None

    def test_deterministic_sell_order(self, tmp_path: Path) -> None:
        """Sell side → limit price slightly below bid."""
        from sauce.agents import execution

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(confidence=0.8, quote=_fresh_quote(mid=100.0))
        sig = sig.model_copy(update={"side": "sell"})
        risk_r = _approved_risk(qty=3.0)
        live_quote = _fresh_quote(bid=99.5, ask=100.5, mid=100.0)

        with (
            patch("sauce.agents.execution.get_settings", return_value=settings),
            patch("sauce.agents.execution.log_event"),
            patch("sauce.agents.execution.is_data_fresh", return_value=True),
        ):
            result = asyncio.run(
                execution.run(sig, risk_r, live_quote, loop_id="test-loop")
            )

        assert result is not None
        assert result.side == "sell"
        # sell limit = bid * (1 - 0.0005)
        expected_limit = round(99.5 * (1.0 - 0.0005), 4)
        assert result.limit_price == pytest.approx(expected_limit)


# ─── Supervisor agent ─────────────────────────────────────────────────────────

class TestSupervisorAgent:
    def test_aborts_on_empty_orders(self, tmp_path: Path) -> None:
        from sauce.agents import supervisor

        settings = _mock_settings(tmp_path)
        with (
            patch("sauce.agents.supervisor.get_settings", return_value=settings),
            patch("sauce.agents.supervisor.log_event"),
            patch("sauce.agents.supervisor.is_trading_paused", return_value=False),
        ):
            decision = asyncio.run(
                supervisor.run(
                    orders=[],
                    signals=[],
                    risk_results=[],
                    account={"equity": "10000"},
                    loop_id="test-loop",
                )
            )

        assert decision.action == "abort"
        assert decision.final_orders == []

    def test_aborts_on_trading_paused(self, tmp_path: Path) -> None:
        from sauce.agents import supervisor

        settings = _mock_settings(tmp_path)
        order = _limit_order()
        sig = _buy_signal()
        risk_r = _approved_risk()

        with (
            patch("sauce.agents.supervisor.get_settings", return_value=settings),
            patch("sauce.agents.supervisor.log_event"),
            patch("sauce.agents.supervisor.is_trading_paused", return_value=True),
        ):
            decision = asyncio.run(
                supervisor.run(
                    orders=[order],
                    signals=[sig],
                    risk_results=[risk_r],
                    account={"equity": "10000"},
                    loop_id="test-loop",
                )
            )

        assert decision.action == "abort"
        assert "TRADING_PAUSE" in decision.reason

    def test_aborts_on_stale_quote(self, tmp_path: Path) -> None:
        from sauce.agents import supervisor

        settings = _mock_settings(tmp_path)
        order = _limit_order()
        sig = _buy_signal(quote=_stale_quote())  # stale evidence
        risk_r = _approved_risk()

        with (
            patch("sauce.agents.supervisor.get_settings", return_value=settings),
            patch("sauce.agents.supervisor.log_event"),
            patch("sauce.agents.supervisor.is_trading_paused", return_value=False),
            patch("sauce.agents.supervisor.is_data_fresh", return_value=False),
        ):
            decision = asyncio.run(
                supervisor.run(
                    orders=[order],
                    signals=[sig],
                    risk_results=[risk_r],
                    account={"equity": "10000"},
                    loop_id="test-loop",
                )
            )

        assert decision.action == "abort"

    def test_aborts_on_missing_risk_approval(self, tmp_path: Path) -> None:
        from sauce.agents import supervisor

        settings = _mock_settings(tmp_path)
        order = _limit_order()
        sig = _buy_signal()
        risk_r = _vetoed_risk()  # vetoed → not in approved_risk

        with (
            patch("sauce.agents.supervisor.get_settings", return_value=settings),
            patch("sauce.agents.supervisor.log_event"),
            patch("sauce.agents.supervisor.is_trading_paused", return_value=False),
            patch("sauce.agents.supervisor.is_data_fresh", return_value=True),
        ):
            decision = asyncio.run(
                supervisor.run(
                    orders=[order],
                    signals=[sig],
                    risk_results=[risk_r],
                    account={"equity": "10000"},
                    loop_id="test-loop",
                )
            )

        assert decision.action == "abort"

    def test_calls_claude_returns_execute(self, tmp_path: Path) -> None:
        from sauce.agents import supervisor

        settings = _mock_settings(tmp_path)
        order = _limit_order()
        sig = _buy_signal()
        risk_r = _approved_risk()

        claude_response = json.dumps(
            {"action": "execute", "vetoes": [], "reason": "All checks pass."}
        )

        with (
            patch("sauce.agents.supervisor.get_settings", return_value=settings),
            patch("sauce.agents.supervisor.log_event"),
            patch("sauce.agents.supervisor.is_trading_paused", return_value=False),
            patch("sauce.agents.supervisor.is_data_fresh", return_value=True),
            patch("sauce.agents.supervisor.llm") as mock_llm,
        ):
            mock_llm.LLMError = Exception
            mock_llm.call_claude = AsyncMock(return_value=claude_response)
            decision = asyncio.run(
                supervisor.run(
                    orders=[order],
                    signals=[sig],
                    risk_results=[risk_r],
                    account={"equity": "10000", "buying_power": "5000"},
                    loop_id="test-loop",
                )
            )

        assert decision.action == "execute"
        assert len(decision.final_orders) == 1

    def test_aborts_if_claude_returns_abort(self, tmp_path: Path) -> None:
        from sauce.agents import supervisor

        settings = _mock_settings(tmp_path)
        order = _limit_order()
        sig = _buy_signal()
        risk_r = _approved_risk()

        claude_response = json.dumps(
            {
                "action": "abort",
                "vetoes": ["AAPL"],
                "reason": "Unexpected market conditions.",
            }
        )

        with (
            patch("sauce.agents.supervisor.get_settings", return_value=settings),
            patch("sauce.agents.supervisor.log_event"),
            patch("sauce.agents.supervisor.is_trading_paused", return_value=False),
            patch("sauce.agents.supervisor.is_data_fresh", return_value=True),
            patch("sauce.agents.supervisor.llm") as mock_llm,
        ):
            mock_llm.LLMError = Exception
            mock_llm.call_claude = AsyncMock(return_value=claude_response)
            decision = asyncio.run(
                supervisor.run(
                    orders=[order],
                    signals=[sig],
                    risk_results=[risk_r],
                    account={"equity": "10000"},
                    loop_id="test-loop",
                )
            )

        assert decision.action == "abort"
        assert decision.final_orders == []


# ─── Portfolio agent ──────────────────────────────────────────────────────────

class TestPortfolioAgent:
    def test_empty_positions_return_zero_exposure(self, tmp_path: Path) -> None:
        from sauce.agents import portfolio

        settings = _mock_settings(tmp_path)
        with (
            patch("sauce.agents.portfolio.get_settings", return_value=settings),
            patch("sauce.agents.portfolio.log_event"),
        ):
            review = asyncio.run(
                portfolio.run(
                    symbols=["AAPL"],
                    positions=[],
                    signals=[],
                    loop_id="test-loop",
                )
            )

        assert isinstance(review, PortfolioReview)
        assert review.total_exposure_pct == 0.0
        assert review.rebalance_needed is False

    def test_computes_stop_and_target_from_atr(self, tmp_path: Path) -> None:
        """Long AAPL, ATR=2, mid=100 → stop=96, target=106."""
        from sauce.agents import portfolio

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(symbol="AAPL", atr_14=2.0)
        position = {"symbol": "AAPL", "qty": "5", "market_value": "500"}

        with (
            patch("sauce.agents.portfolio.get_settings", return_value=settings),
            patch("sauce.agents.portfolio.log_event"),
        ):
            review = asyncio.run(
                portfolio.run(
                    symbols=["AAPL"],
                    positions=[position],
                    signals=[sig],
                    loop_id="test-loop",
                )
            )

        assert len(review.positions) == 1
        pos = review.positions[0]
        assert pos.symbol == "AAPL"
        assert pos.suggested_stop_price == pytest.approx(96.0)   # 100 - 2*2
        assert pos.suggested_target_price == pytest.approx(106.0)  # 100 + 3*2

    def test_flags_over_concentrated_position(self, tmp_path: Path) -> None:
        """One position is >2x max_position_pct * total → over_concentrated=True."""
        from sauce.agents import portfolio

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(symbol="AAPL", atr_14=1.0)
        # max_position_pct=0.05, multiplier=2x
        # total_pos_value for concentration calc: 500+50 = 550
        # threshold = 550 * 0.05 * 2 = 55
        # AAPL market_value=500 > 55 → over-concentrated
        positions = [
            {"symbol": "AAPL", "qty": "5", "market_value": "500"},
            {"symbol": "MSFT", "qty": "0.5", "market_value": "50"},
        ]
        with (
            patch("sauce.agents.portfolio.get_settings", return_value=settings),
            patch("sauce.agents.portfolio.log_event"),
        ):
            review = asyncio.run(
                portfolio.run(
                    symbols=["AAPL", "MSFT"],
                    positions=positions,
                    signals=[sig],
                    loop_id="test-loop",
                )
            )

        aapl_pos = next((p for p in review.positions if p.symbol == "AAPL"), None)
        assert aapl_pos is not None
        assert aapl_pos.over_concentrated is True
        assert review.rebalance_needed is True

    def test_detects_uncovered_symbols(self, tmp_path: Path) -> None:
        """MSFT in universe but no signal → appears in uncovered_symbols."""
        from sauce.agents import portfolio

        settings = _mock_settings(tmp_path)
        sig = _buy_signal(symbol="AAPL")  # only AAPL has a signal

        with (
            patch("sauce.agents.portfolio.get_settings", return_value=settings),
            patch("sauce.agents.portfolio.log_event"),
        ):
            review = asyncio.run(
                portfolio.run(
                    symbols=["AAPL", "MSFT"],
                    positions=[],
                    signals=[sig],
                    loop_id="test-loop",
                )
            )

        assert "MSFT" in review.uncovered_symbols


# ─── Ops agent ────────────────────────────────────────────────────────────────

class TestOpsAgent:
    def test_writes_daily_log_file(self, tmp_path: Path) -> None:
        """run() should create data/logs/daily_YYYY-MM-DD.txt."""
        from sauce.agents import ops

        settings = _mock_settings(tmp_path)
        summary = {
            "signals_total": 2,
            "signals_buy_sell": 1,
            "risk_vetoes": 0,
            "risk_approved": 1,
            "orders_prepared": 1,
            "supervisor_action": "execute",
            "orders_placed": 1,
            "symbols_attempted": ["AAPL"],
            "veto_by_symbol": {},
        }

        with (
            patch("sauce.agents.ops.get_settings", return_value=settings),
            patch("sauce.agents.ops.log_event"),
            patch("sauce.agents.ops.pause_trading"),
        ):
            asyncio.run(ops.run("test-loop", summary))

        log_dir = Path(str(settings.db_path)).parent / "logs"
        log_files = list(log_dir.glob("daily_*.jsonl"))
        assert len(log_files) == 1
        content = log_files[0].read_text()
        assert "test-loop" in content
        assert "execute" in content

    def test_detects_100pct_veto_anomaly(self, tmp_path: Path) -> None:
        """All signals vetoed across 3 symbols → pause_trading called."""
        from sauce.agents import ops

        settings = _mock_settings(tmp_path)
        summary = {
            "signals_total": 3,
            "signals_buy_sell": 3,
            "risk_vetoes": 3,
            "risk_approved": 0,
            "orders_prepared": 0,
            "supervisor_action": "abort",
            "orders_placed": 0,
            "symbols_attempted": ["AAPL", "MSFT", "TSLA"],
            "veto_by_symbol": {"AAPL": 1, "MSFT": 1, "TSLA": 1},
        }

        with (
            patch("sauce.agents.ops.get_settings", return_value=settings),
            patch("sauce.agents.ops.send_alert"),
            patch("sauce.agents.ops.log_event"),
            patch("sauce.agents.ops.pause_trading") as mock_pause,
        ):
            asyncio.run(ops.run("test-loop", summary))

        mock_pause.assert_called_once()
        call_kwargs = mock_pause.call_args
        assert "veto" in call_kwargs.kwargs.get("reason", "").lower() or \
               "veto" in str(call_kwargs.args).lower()

    def test_no_pause_when_below_symbol_threshold(self, tmp_path: Path) -> None:
        """Only 2 symbols vetoed < threshold 3 → no pause."""
        from sauce.agents import ops

        settings = _mock_settings(tmp_path)
        summary = {
            "signals_total": 2,
            "signals_buy_sell": 2,
            "risk_vetoes": 2,
            "risk_approved": 0,
            "orders_prepared": 0,
            "supervisor_action": "abort",
            "orders_placed": 0,
            "symbols_attempted": ["AAPL", "MSFT"],
            "veto_by_symbol": {},
        }

        with (
            patch("sauce.agents.ops.get_settings", return_value=settings),
            patch("sauce.agents.ops.log_event"),
            patch("sauce.agents.ops.pause_trading") as mock_pause,
        ):
            asyncio.run(ops.run("test-loop", summary))

        mock_pause.assert_not_called()

    def test_logs_ops_summary_event(self, tmp_path: Path) -> None:
        """run() calls log_event with event_type='ops_summary'."""
        from sauce.agents import ops

        settings = _mock_settings(tmp_path)
        summary = {
            "signals_total": 1,
            "signals_buy_sell": 0,
            "risk_vetoes": 0,
            "risk_approved": 0,
            "orders_prepared": 0,
            "supervisor_action": "abort",
            "orders_placed": 0,
            "symbols_attempted": ["AAPL"],
            "veto_by_symbol": {},
        }
        logged: list[AuditEvent] = []

        with (
            patch("sauce.agents.ops.get_settings", return_value=settings),
            patch(
                "sauce.agents.ops.log_event",
                side_effect=lambda evt, **kw: logged.append(evt),
            ),
            patch("sauce.agents.ops.pause_trading"),
        ):
            asyncio.run(ops.run("test-loop", summary))

        assert any(e.event_type == "ops_summary" for e in logged)
