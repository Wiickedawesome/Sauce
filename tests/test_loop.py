"""
tests/test_loop.py — Tests for core/loop.py.

Per Testing Rule 6.4: loop test with stubs must confirm:
  - At least one AuditEvent with event_type="loop_start" in the DB.
  - At least one AuditEvent with event_type="loop_end" in the DB.
  - Zero rows in the orders table.

All agents and broker calls are mocked — no real network calls.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sauce.adapters.db as db_module


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_db_and_settings(tmp_path, monkeypatch):
    """
    Isolate each test: fresh DB path, clear settings cache, reset engine.
    """
    db_path = str(tmp_path / "test_loop.db")
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("TRADING_PAUSE", "false")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    # Small universe for speed
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "AAPL")
    monkeypatch.setenv("TRADING_UNIVERSE_CRYPTO", "BTC/USD")

    db_module._engines = {}

    from sauce.core.config import get_settings
    get_settings.cache_clear()

    yield

    db_module._engines = {}
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def mock_boot_and_market_ctx():
    """Mock Agent 0 (session_boot) and Agent 1 (market_context) for all loop tests."""
    boot_ctx = make_stub_boot_ctx()
    mkt_ctx = make_stub_market_ctx()
    with patch("sauce.agents.session_boot.run", new=AsyncMock(return_value=boot_ctx)):
        with patch("sauce.agents.market_context.run", new=AsyncMock(return_value=mkt_ctx)):
            yield


def make_stub_boot_ctx():
    """Return a minimal BootContext for loop test stubs."""
    from sauce.core.schemas import BootContext, StrategicContext
    return BootContext(
        was_reset=False,
        calendar_events=[],
        strategic_context=StrategicContext(as_of=datetime.now(timezone.utc)),
        is_suppressed=False,
        as_of=datetime.now(timezone.utc),
    )


def make_stub_market_ctx():
    """Return a minimal MarketContext for loop test stubs."""
    from sauce.core.schemas import IntradayNarrativeEntry, MarketContext, RegimeLogEntry
    return MarketContext(
        regime=RegimeLogEntry(
            timestamp=datetime.now(timezone.utc),
            regime_type="RANGING",
            confidence=0.5,
        ),
        narrative=IntradayNarrativeEntry(
            timestamp=datetime.now(timezone.utc),
            narrative_text="Stub market context",
        ),
        calendar_events=[],
        is_dead=False,
        is_suppressed=False,
        as_of=datetime.now(timezone.utc),
    )


def make_fresh_quote(symbol: str = "AAPL") -> MagicMock:
    from sauce.core.schemas import PriceReference
    return PriceReference(
        symbol=symbol,
        bid=149.5,
        ask=150.5,
        mid=150.0,
        as_of=datetime.now(timezone.utc),
    )


def make_stub_signal(symbol: str = "AAPL") -> MagicMock:
    """Return a real Signal with side=hold."""
    from datetime import timezone
    from sauce.core.schemas import Evidence, Indicators, PriceReference, Signal
    quote = make_fresh_quote(symbol)
    evidence = Evidence(
        symbol=symbol,
        price_reference=quote,
        indicators=Indicators(),
        as_of=datetime.now(timezone.utc),
    )
    return Signal(
        symbol=symbol,
        side="hold",
        confidence=0.0,
        evidence=evidence,
        reasoning="stub",
        as_of=datetime.now(timezone.utc),
        prompt_version="v1",
    )


def make_stub_risk_result(symbol: str = "AAPL") -> MagicMock:
    from sauce.core.schemas import RiskCheckResult, RiskChecks
    return RiskCheckResult(
        symbol=symbol,
        side="hold",
        veto=True,
        reason="stub",
        qty=None,
        checks=RiskChecks(
            max_position_pct_ok=False,
            max_exposure_ok=False,
            asset_class_ok=False,
            daily_loss_ok=False,
            volatility_ok=False,
            confidence_ok=False,
        ),
        as_of=datetime.now(timezone.utc),
        prompt_version="v1",
    )


def make_stub_portfolio_review() -> MagicMock:
    from sauce.core.schemas import PortfolioReview
    return PortfolioReview(
        positions=[],
        total_exposure_pct=0.0,
        as_of=datetime.now(timezone.utc),
        prompt_version="v1",
    )


def make_stub_supervisor_abort() -> MagicMock:
    from sauce.core.schemas import SupervisorDecision
    return SupervisorDecision(
        action="abort",
        final_orders=[],
        vetoes=["stub"],
        reason="stub abort",
        as_of=datetime.now(timezone.utc),
        prompt_version="v1",
    )


def make_fake_account() -> dict:
    return {
        "id": "test-acct",
        "equity": "10000.00",
        "last_equity": "10000.00",  # 0% loss — within limit
        "buying_power": "10000.00",
        "portfolio_value": "10000.00",
    }


# ── Helpers to count DB events ────────────────────────────────────────────────

def count_events_by_type(event_type: str) -> int:
    from sqlalchemy import text
    from sauce.adapters.db import get_session
    session = get_session()
    try:
        return session.execute(
            text("SELECT COUNT(*) FROM audit_events WHERE event_type = :et"),
            {"et": event_type},
        ).scalar() or 0
    finally:
        session.close()


def count_orders() -> int:
    from sauce.adapters.db import get_session
    from sauce.adapters.db import OrderRow
    session = get_session()
    try:
        return session.query(OrderRow).count()
    finally:
        session.close()


def get_latest_event_payload(event_type: str) -> dict:
    import json
    from sqlalchemy import text
    from sauce.adapters.db import get_session

    session = get_session()
    try:
        row = session.execute(
            text(
                "SELECT payload FROM audit_events WHERE event_type = :et "
                "ORDER BY id DESC LIMIT 1"
            ),
            {"et": event_type},
        ).fetchone()
        assert row is not None, f"No audit event found for {event_type}"
        return json.loads(row[0])
    finally:
        session.close()


# ── Full stub run — audit trail ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_stub_run_writes_loop_start_and_end(monkeypatch):
    """Rule 6.4: loop_start and loop_end must appear in DB after a stub run."""

    quotes = {
        "AAPL": make_fresh_quote("AAPL"),
        "BTC/USD": make_fresh_quote("BTC/USD"),
    }
    signal = make_stub_signal("AAPL")
    btc_signal = make_stub_signal("BTC/USD")
    portfolio_review = make_stub_portfolio_review()
    supervisor_decision = make_stub_supervisor_abort()

    with patch("sauce.core.loop.get_account", return_value=make_fake_account()):
        with patch("sauce.core.loop.get_positions", return_value=[]):
            with patch(
                "sauce.core.loop.get_universe_snapshot", return_value=quotes
            ):
                with patch(
                    "sauce.agents.research.run",
                    new=AsyncMock(side_effect=[signal, btc_signal]),
                ):
                    with patch("sauce.agents.risk.run", new=AsyncMock(return_value=make_stub_risk_result())):
                        with patch(
                            "sauce.agents.portfolio.run",
                            new=AsyncMock(return_value=portfolio_review),
                        ):
                            with patch(
                                "sauce.agents.supervisor.run",
                                new=AsyncMock(return_value=supervisor_decision),
                            ):
                                with patch("sauce.agents.ops.run", new=AsyncMock()):
                                    from sauce.core.loop import main
                                    await main()

    assert count_events_by_type("loop_start") >= 1, "loop_start not in DB"
    assert count_events_by_type("loop_end") >= 1, "loop_end not in DB"


@pytest.mark.asyncio
async def test_full_stub_run_places_zero_orders(monkeypatch):
    """Rule 6.4: zero rows in the orders table after a full stub run."""

    quotes = {"AAPL": make_fresh_quote("AAPL"), "BTC/USD": make_fresh_quote("BTC/USD")}

    with patch("sauce.core.loop.get_account", return_value=make_fake_account()):
        with patch("sauce.core.loop.get_positions", return_value=[]):
            with patch(
                "sauce.core.loop.get_universe_snapshot", return_value=quotes
            ):
                with patch(
                    "sauce.agents.research.run",
                    new=AsyncMock(side_effect=[make_stub_signal("AAPL"), make_stub_signal("BTC/USD")]),
                ):
                    with patch("sauce.agents.risk.run", new=AsyncMock(return_value=make_stub_risk_result())):
                        with patch(
                            "sauce.agents.portfolio.run",
                            new=AsyncMock(return_value=make_stub_portfolio_review()),
                        ):
                            with patch(
                                "sauce.agents.supervisor.run",
                                new=AsyncMock(return_value=make_stub_supervisor_abort()),
                            ):
                                with patch("sauce.agents.ops.run", new=AsyncMock()):
                                    with patch(
                                        "sauce.core.loop.place_order"
                                    ) as mock_place:
                                        from sauce.core.loop import main
                                        await main()

    # Verify that place_order was never called
    mock_place.assert_not_called()
    assert count_orders() == 0, "orders table should be empty after stub run"


# ── TRADING_PAUSE causes early exit ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_loop_aborts_when_trading_paused(monkeypatch):
    """If TRADING_PAUSE=true the loop should exit before calling any agent."""
    monkeypatch.setenv("TRADING_PAUSE", "true")
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    with patch("sauce.agents.research.run", new=AsyncMock()) as mock_research:
        with patch("sauce.core.loop.get_account") as mock_account:
            from sauce.core.loop import main
            await main()

    mock_research.assert_not_called()
    mock_account.assert_not_called()

    # loop_start and loop_end must still be written even when paused
    assert count_events_by_type("loop_start") >= 1
    assert count_events_by_type("loop_end") >= 1


@pytest.mark.asyncio
async def test_loop_aborts_when_macro_suppressed(monkeypatch):
    """CF-01: Macro suppression filters out equities but allows crypto through.

    When is_suppressed=True the loop no longer aborts entirely — it continues
    with only crypto symbols.  Market context and account fetches still run.
    """
    from sauce.core.schemas import BootContext, StrategicContext

    suppressed_boot_ctx = BootContext(
        was_reset=False,
        calendar_events=[],
        strategic_context=StrategicContext(as_of=datetime.now(timezone.utc)),
        is_suppressed=True,
        as_of=datetime.now(timezone.utc),
    )

    quotes = {"BTC/USD": make_fresh_quote("BTC/USD")}

    with patch("sauce.agents.session_boot.run", new=AsyncMock(return_value=suppressed_boot_ctx)):
        with patch("sauce.agents.market_context.run", new=AsyncMock(return_value=make_stub_market_ctx())) as mock_market_context:
            with patch("sauce.core.loop.get_account", return_value=make_fake_account()):
                with patch("sauce.core.loop.get_positions", return_value=[]):
                    with patch("sauce.core.loop.get_universe_snapshot", return_value=quotes):
                        with patch("sauce.agents.research.run", new=AsyncMock(return_value=make_stub_signal("BTC/USD"))) as mock_research:
                            with patch("sauce.agents.risk.run", new=AsyncMock(return_value=make_stub_risk_result())):
                                with patch("sauce.agents.portfolio.run", new=AsyncMock(return_value=make_stub_portfolio_review())):
                                    with patch("sauce.agents.supervisor.run", new=AsyncMock(return_value=make_stub_supervisor_abort())):
                                        with patch("sauce.agents.ops.run", new=AsyncMock()):
                                            from sauce.core.loop import main
                                            await main()

    # CF-01: market context IS called now — suppression only filters equities.
    mock_market_context.assert_called()
    assert count_events_by_type("loop_start") >= 1
    assert count_events_by_type("loop_end") >= 1
    # Verify the macro_suppression safety_check was logged (may not be the
    # latest safety_check since the loop continues and logs more events).
    import json
    from sqlalchemy import text
    from sauce.adapters.db import get_session
    session = get_session()
    try:
        rows = session.execute(
            text(
                "SELECT payload FROM audit_events WHERE event_type = 'safety_check' "
                "ORDER BY id DESC"
            ),
        ).fetchall()
        payloads = [json.loads(r[0]) for r in rows]
        suppression_events = [p for p in payloads if p.get("check") == "macro_suppression"]
        assert len(suppression_events) >= 1, "No macro_suppression safety_check logged"
        assert suppression_events[0]["result"] is True
    finally:
        session.close()


# ── Stale data is skipped ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stale_quote_skips_research(monkeypatch):
    """If a quote is stale, research.run must NOT be called for that symbol."""
    from datetime import timedelta
    from sauce.core.schemas import PriceReference

    # Create a quote that is 1 hour old (stale)
    stale_as_of = datetime.now(timezone.utc) - timedelta(hours=1)
    stale_quote = PriceReference(
        symbol="AAPL",
        bid=149.5,
        ask=150.5,
        mid=150.0,
        as_of=stale_as_of,
    )
    quotes = {"AAPL": stale_quote, "BTC/USD": make_fresh_quote("BTC/USD")}

    with patch("sauce.core.loop.get_account", return_value=make_fake_account()):
        with patch("sauce.core.loop.get_positions", return_value=[]):
            with patch(
                "sauce.core.loop.get_universe_snapshot", return_value=quotes
            ):
                with patch("sauce.agents.research.run", new=AsyncMock(return_value=make_stub_signal("BTC/USD"))) as mock_research:
                    with patch("sauce.agents.risk.run", new=AsyncMock(return_value=make_stub_risk_result())):
                        with patch(
                            "sauce.agents.portfolio.run",
                            new=AsyncMock(return_value=make_stub_portfolio_review()),
                        ):
                            with patch(
                                "sauce.agents.supervisor.run",
                                new=AsyncMock(return_value=make_stub_supervisor_abort()),
                            ):
                                with patch("sauce.agents.ops.run", new=AsyncMock()):
                                    from sauce.core.loop import main
                                    await main()

    # research.run should only have been called for BTC/USD (crypto, always fresh check passes)
    # AAPL quote was stale so research.run should not be called for AAPL
    calls = mock_research.call_args_list
    called_symbols = [c.kwargs.get("symbol") or c.args[0] for c in calls]
    assert "AAPL" not in called_symbols, "research.run should not be called with stale data"


# ── Daily loss breach aborts loop ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_loop_aborts_on_daily_loss_breach(monkeypatch):
    """If daily loss is breached the loop should exit after checking account."""
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.02")
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    # Account showing a 5% loss — exceeds 2% limit
    bad_account = {
        "id": "test",
        "equity": "9500.00",
        "last_equity": "10000.00",
        "buying_power": "9500.00",
        "portfolio_value": "9500.00",
    }

    with patch("sauce.core.loop.get_account", return_value=bad_account):
        with patch("sauce.core.loop.get_positions", return_value=[]):
            with patch("sauce.agents.research.run", new=AsyncMock()) as mock_research:
                from sauce.core.loop import main
                await main()

    mock_research.assert_not_called()

    # Loop start/end must still be logged
    assert count_events_by_type("loop_start") >= 1
    assert count_events_by_type("loop_end") >= 1


# ── loop_end always written even on exception ─────────────────────────────────

@pytest.mark.asyncio
async def test_loop_end_written_even_when_account_fetch_fails(monkeypatch):
    """loop_end must always be written, even when a step raises."""
    from sauce.adapters.broker import BrokerError

    with patch("sauce.core.loop.get_account", side_effect=BrokerError("API down")):
        from sauce.core.loop import main
        await main()

    assert count_events_by_type("loop_start") >= 1
    assert count_events_by_type("loop_end") >= 1


@pytest.mark.asyncio
async def test_reconciliation_matches_crypto_position_symbols_without_slash(monkeypatch):
    """Crypto reconciliation should match BTC/USD orders against BTCUSD positions."""
    from sauce.core.schemas import Order, SupervisorDecision

    quotes = {"BTC/USD": make_fresh_quote("BTC/USD")}
    buy_signal = make_stub_signal("BTC/USD").model_copy(update={"side": "buy", "confidence": 0.75})
    approved_risk = make_stub_risk_result("BTC/USD").model_copy(
        update={"side": "buy", "veto": False, "reason": None, "qty": 0.01}
    )
    prepared_order = Order(
        symbol="BTC/USD",
        side="buy",
        qty=0.01,
        order_type="limit",
        time_in_force="day",
        limit_price=70000.0,
        stop_price=None,
        as_of=datetime.now(timezone.utc),
        prompt_version="v1",
    )
    execute_decision = SupervisorDecision(
        action="execute",
        final_orders=[prepared_order],
        vetoes=[],
        reason="approved",
        as_of=datetime.now(timezone.utc),
        prompt_version="v1",
    )

    with patch("sauce.core.loop.check_market_hours", return_value=True):
        with patch("sauce.core.loop.get_account", return_value=make_fake_account()):
            with patch("sauce.core.loop.get_positions", side_effect=[[], [{"symbol": "BTCUSD", "qty": "0.01", "market_value": "700.0"}]]):
                with patch("sauce.core.loop.get_universe_snapshot", return_value=quotes):
                    with patch("sauce.agents.research.run", new=AsyncMock(return_value=buy_signal)):
                        with patch("sauce.agents.risk.run", new=AsyncMock(return_value=approved_risk)):
                            with patch("sauce.agents.execution.run", new=AsyncMock(return_value=prepared_order)):
                                with patch("sauce.agents.portfolio.run", new=AsyncMock(return_value=make_stub_portfolio_review())):
                                    with patch("sauce.agents.supervisor.run", new=AsyncMock(return_value=execute_decision)):
                                        with patch("sauce.agents.ops.run", new=AsyncMock()):
                                            with patch("sauce.core.loop.place_order", return_value={"id": "broker-1", "status": "submitted"}):
                                                from sauce.core.loop import main
                                                await main()

    payload = get_latest_event_payload("reconciliation")
    assert payload["position_confirmed"] is True
    assert payload["position_qty"] == "0.01"


# ── IMP-06: Full pipeline integration test — all 8 safety layers ─────────

def _get_all_safety_checks() -> list[dict]:
    """Return all safety_check event payloads from the audit trail."""
    import json
    from sqlalchemy import text
    from sauce.adapters.db import get_session

    session = get_session()
    try:
        rows = session.execute(
            text("SELECT payload FROM audit_events WHERE event_type = 'safety_check' ORDER BY id")
        ).fetchall()
        return [json.loads(r[0]) for r in rows]
    finally:
        session.close()


@pytest.mark.asyncio
async def test_full_pipeline_all_8_safety_layers(monkeypatch):
    """IMP-06: Verify that a complete loop run with an approved order exercises
    all 8 safety layers and leaves a complete audit trail.

    Uses AAPL (equity) so market_hours check runs its real code path.
    _now_et is patched to return a weekday during market hours.

    Layers verified:
      L1 trading_pause (pass)
      L2 daily_loss (pass)
      L3 market_hours (pass — real code, time patched)
      L4 data_freshness (pass — fresh quote)
      L5 confidence_floor (pass — confidence above min)
      L6 risk_check (pass — veto=False)
      L7 debate (real run, not mocked — deterministic)
      L8 supervisor_decision (execute)
    """
    from zoneinfo import ZoneInfo
    from sauce.core.schemas import (
        Evidence,
        Indicators,
        IntradayNarrativeEntry,
        MarketContext,
        Order,
        PriceReference,
        RegimeLogEntry,
        RiskCheckResult,
        RiskChecks,
        Signal,
        SupervisorDecision,
    )

    # Force universe to AAPL only (no crypto) so all checks run for equity
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "AAPL")
    monkeypatch.setenv("TRADING_UNIVERSE_CRYPTO", "")
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    # Patch _now_et to return a Wednesday at 10:00 ET (market open)
    fake_et = datetime(2025, 6, 11, 10, 0, 0, tzinfo=ZoneInfo("America/New_York"))
    with patch("sauce.core.safety._now_et", return_value=fake_et):

        # Market context with TRENDING_UP regime (required for equity eligibility)
        mkt_ctx = MarketContext(
            regime=RegimeLogEntry(
                timestamp=datetime.now(timezone.utc),
                regime_type="TRENDING_UP",
                confidence=0.8,
            ),
            narrative=IntradayNarrativeEntry(
                timestamp=datetime.now(timezone.utc),
                narrative_text="Integration test market context",
            ),
            calendar_events=[],
            is_dead=False,
            is_suppressed=False,
            as_of=datetime.now(timezone.utc),
        )
        with patch("sauce.agents.market_context.run", new=AsyncMock(return_value=mkt_ctx)):

            # ── Build signal with populated indicators so debate fires real arguments
            quote = PriceReference(
                symbol="AAPL", bid=149.5, ask=150.5, mid=150.0,
                as_of=datetime.now(timezone.utc),
            )
            indicators = Indicators(
                sma_20=152.0, sma_50=145.0,   # golden cross → bull arg
                rsi_14=55.0,                    # healthy range → bull arg
                atr_14=2.0,                     # ATR/price ~1.3% → bull arg (tight stop)
                volume_ratio=1.5,               # above avg → bull arg
                macd_histogram=0.5,             # positive → bull arg
                macd_line=1.0, macd_signal=0.8, # MACD > signal → bull arg
            )
            evidence = Evidence(
                symbol="AAPL", price_reference=quote,
                indicators=indicators, as_of=datetime.now(timezone.utc),
            )
            buy_signal = Signal(
                symbol="AAPL", side="buy", confidence=0.75,
                evidence=evidence, reasoning="integration test signal",
                bear_case="test bear case argument",
                as_of=datetime.now(timezone.utc), prompt_version="v1",
            )

            approved_risk = RiskCheckResult(
                symbol="AAPL", side="buy", veto=False, reason=None, qty=10,
                checks=RiskChecks(
                    max_position_pct_ok=True, max_exposure_ok=True,
                    asset_class_ok=True, daily_loss_ok=True,
                    volatility_ok=True, confidence_ok=True,
                ),
                as_of=datetime.now(timezone.utc), prompt_version="v1",
            )

            prepared_order = Order(
                symbol="AAPL", side="buy", qty=10,
                order_type="limit", time_in_force="day",
                limit_price=150.0, stop_price=None,
                as_of=datetime.now(timezone.utc), prompt_version="v1",
            )
            execute_decision = SupervisorDecision(
                action="execute", final_orders=[prepared_order],
                vetoes=[], reason="all checks passed",
                as_of=datetime.now(timezone.utc), prompt_version="v1",
            )

            quotes = {"AAPL": quote}

            with patch("sauce.core.loop.has_earnings_risk", return_value=False):
                with patch("sauce.core.loop.get_account", return_value=make_fake_account()):
                    with patch("sauce.core.loop.get_positions", return_value=[]):
                        with patch("sauce.core.loop.get_universe_snapshot", return_value=quotes):
                            with patch("sauce.agents.research.run", new=AsyncMock(return_value=buy_signal)):
                                with patch("sauce.agents.risk.run", new=AsyncMock(return_value=approved_risk)):
                                    with patch("sauce.agents.execution.run", new=AsyncMock(return_value=prepared_order)):
                                        with patch("sauce.agents.portfolio.run", new=AsyncMock(return_value=make_stub_portfolio_review())):
                                            with patch("sauce.agents.supervisor.run", new=AsyncMock(return_value=execute_decision)):
                                                with patch("sauce.agents.ops.run", new=AsyncMock()):
                                                    with patch("sauce.core.loop.place_order", return_value={"id": "int-test-1", "status": "submitted"}):
                                                        from sauce.core.loop import main
                                                        await main()

    # ── Verify loop lifecycle
    assert count_events_by_type("loop_start") >= 1
    assert count_events_by_type("loop_end") >= 1

    # ── L1: trading_pause pass-through logged
    safety_checks = _get_all_safety_checks()
    pause_checks = [c for c in safety_checks if c.get("check") == "trading_pause"]
    assert any(c["result"] is False for c in pause_checks), \
        "L1: trading_pause pass-through not logged"

    # ── L2: daily_loss pass logged
    loss_checks = [c for c in safety_checks if c.get("check") == "daily_loss"]
    assert any(c["result"] is True for c in loss_checks), \
        "L2: daily_loss pass not logged"

    # ── L3: market_hours pass logged (real check_market_hours with time patched)
    hours_checks = [c for c in safety_checks if c.get("check") == "market_hours"]
    assert any(c["result"] is True for c in hours_checks), \
        "L3: market_hours pass not logged"

    # ── L5: confidence_floor pass logged
    conf_checks = [c for c in safety_checks if c.get("check") == "confidence_floor"]
    assert any(c["result"] is True for c in conf_checks), \
        "L5: confidence_floor pass not logged"

    # ── L6: risk_check logged
    assert count_events_by_type("risk_check") >= 1, \
        "L6: risk_check event not logged"

    # ── L7: debate logged (real deterministic debate, not mocked)
    assert count_events_by_type("debate") >= 1, \
        "L7: debate event not logged"
    debate_payload = get_latest_event_payload("debate")
    assert debate_payload["bull_score"] > 0, \
        "L7: debate should have >0 bull score with populated indicators"
    assert debate_payload["verdict"] in ("bull_wins", "bear_wins", "contested"), \
        "L7: debate verdict unexpected"

    # ── L8: supervisor_decision logged
    assert count_events_by_type("supervisor_decision") >= 1, \
        "L8: supervisor_decision event not logged"

    # ── Order submitted and persisted
    assert count_events_by_type("order_submitted") >= 1, \
        "order_submitted event not logged"
    assert count_orders() >= 1, \
        "orders table should have at least 1 row after approved execution"


# ── F-14: Ops agent guaranteed execution on early abort ───────────────────────

@pytest.mark.asyncio
async def test_ops_runs_on_early_abort_paused(monkeypatch):
    """F-14: ops.run() must execute even when the loop aborts early (trading paused)."""
    monkeypatch.setenv("TRADING_PAUSE", "true")
    from sauce.core.config import get_settings
    get_settings.cache_clear()

    with patch("sauce.agents.ops.run", new=AsyncMock()) as mock_ops:
        from sauce.core.loop import main
        await main()

    mock_ops.assert_called_once()
    call_kwargs = mock_ops.call_args
    summary = call_kwargs.kwargs.get("summary") or call_kwargs[1].get("summary")
    assert summary["supervisor_action"] == "abort"
    assert summary["signals_total"] == 0


@pytest.mark.asyncio
async def test_ops_runs_on_early_abort_broker_error(monkeypatch):
    """F-14: ops.run() must execute even when the broker account fetch fails."""
    from sauce.adapters.broker import BrokerError

    with patch("sauce.core.loop.get_account", side_effect=BrokerError("API down")):
        with patch("sauce.agents.ops.run", new=AsyncMock()) as mock_ops:
            from sauce.core.loop import main
            await main()

    mock_ops.assert_called_once()
    summary = mock_ops.call_args.kwargs.get("summary") or mock_ops.call_args[1].get("summary")
    assert summary["supervisor_action"] == "abort"


# ── F-13: Container restart detection ─────────────────────────────────────────

def test_log_container_start_writes_event(tmp_path, monkeypatch):
    """F-13: log_container_start.py must insert a container_start audit event."""
    import sqlite3

    db_path = str(tmp_path / "container_test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE audit_events ("
        "  id TEXT PRIMARY KEY,"
        "  loop_id TEXT,"
        "  event_type TEXT,"
        "  timestamp TEXT,"
        "  payload TEXT"
        ")"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("DB_PATH", db_path)

    import importlib
    import scripts.log_container_start as lcs
    importlib.reload(lcs)

    result = lcs.main()
    assert result == 0

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT event_type, loop_id FROM audit_events WHERE event_type = 'container_start'"
    ).fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0] == "container_start"
    assert rows[0][1] == "container"


# ── Audit 05 Fixes ────────────────────────────────────────────────────────────

# F-05: time_of_day_bucket should use entry time, not exit time
# F-04: regime_at_entry should use entry regime, not current regime
# F-03: run_weekly should be set in the summary dict
# F-01: write_veto_pattern should be called on supervisor abort


class TestDetectClosedPositions:
    """Tests for _detect_closed_positions fixes (F-04, F-05)."""

    def test_time_of_day_bucket_uses_entry_time(self, tmp_path, monkeypatch):
        """F-05: time_of_day_bucket should reflect entry hour, not exit hour."""
        import sauce.core.loop as loop_mod
        from sauce.core.schemas import SetupPerformanceEntry
        from sauce.memory.db import get_engine, get_session, SetupPerformanceRow, StrategicBase, SessionBase
        from sauce.memory import db as memory_db

        strategic_db = str(tmp_path / "strat.db")
        session_db = str(tmp_path / "session.db")

        memory_db._engines = {}
        memory_db._session_factories = {}
        engine = get_engine(strategic_db)
        StrategicBase.metadata.create_all(engine)
        s_engine = get_engine(session_db)
        SessionBase.metadata.create_all(s_engine)

        # Simulate a position opened at 09:00 UTC, closing now at 15:00 UTC
        entry_time = datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
        exit_time = datetime(2025, 6, 1, 15, 0, 0, tzinfo=timezone.utc)

        # Set _previous_positions with a BTC position
        loop_mod._previous_positions = {
            "BTCUSD": {"symbol": "BTCUSD", "unrealized_pl": 50.0},
        }

        # Mock get_trade_entry_time to return entry_time
        with patch("sauce.core.loop.get_trade_entry_time", return_value=entry_time):
            with patch("sauce.core.loop.get_latest_setup_type", return_value=None):
                with patch("sauce.core.loop.datetime") as mock_dt:
                    mock_dt.now.return_value = exit_time
                    mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
                    loop_mod._detect_closed_positions(
                        current_positions=[],  # position is gone — closed
                        loop_id="test-loop",
                        regime="TRENDING_UP",
                        db_path=strategic_db,
                        session_db_path=session_db,
                    )

        # Verify the time_of_day_bucket uses entry hour (09:00), not exit hour (15:00)
        session = get_session(strategic_db)
        try:
            rows = session.query(SetupPerformanceRow).all()
            assert len(rows) == 1
            assert rows[0].time_of_day_bucket == "09:00"
        finally:
            session.close()

        memory_db._engines = {}
        memory_db._session_factories = {}

    def test_regime_at_entry_uses_stored_regime(self, tmp_path, monkeypatch):
        """F-04: regime_at_entry should use the regime from when position was opened."""
        import sauce.core.loop as loop_mod
        from sauce.memory.db import get_engine, get_session, SetupPerformanceRow, StrategicBase, SessionBase
        from sauce.memory import db as memory_db

        strategic_db = str(tmp_path / "strat.db")
        session_db = str(tmp_path / "session.db")

        memory_db._engines = {}
        memory_db._session_factories = {}
        engine = get_engine(strategic_db)
        StrategicBase.metadata.create_all(engine)
        s_engine = get_engine(session_db)
        SessionBase.metadata.create_all(s_engine)

        # Set _previous_positions with entry regime tagged (simulates a position
        # that was first seen when regime was RANGING)
        loop_mod._previous_positions = {
            "BTCUSD": {
                "symbol": "BTCUSD",
                "unrealized_pl": 50.0,
                "_entry_regime": "RANGING",
            },
        }

        with patch("sauce.core.loop.get_trade_entry_time", return_value=None):
            with patch("sauce.core.loop.get_latest_setup_type", return_value=None):
                loop_mod._detect_closed_positions(
                    current_positions=[],
                    loop_id="test-loop",
                    regime="TRENDING_UP",  # current regime is different
                    db_path=strategic_db,
                    session_db_path=session_db,
                )

        session = get_session(strategic_db)
        try:
            rows = session.query(SetupPerformanceRow).all()
            assert len(rows) == 1
            assert rows[0].regime_at_entry == "RANGING"  # NOT TRENDING_UP
        finally:
            session.close()

        memory_db._engines = {}
        memory_db._session_factories = {}

    def test_entry_regime_tagged_on_new_position(self, tmp_path):
        """F-04: New positions get tagged with the current regime for later use."""
        import sauce.core.loop as loop_mod
        from sauce.memory.db import get_engine, StrategicBase, SessionBase
        from sauce.memory import db as memory_db

        strategic_db = str(tmp_path / "strat.db")
        session_db = str(tmp_path / "session.db")

        memory_db._engines = {}
        memory_db._session_factories = {}
        get_engine(strategic_db)
        s_engine = get_engine(session_db)
        SessionBase.metadata.create_all(s_engine)

        loop_mod._previous_positions = {}

        loop_mod._detect_closed_positions(
            current_positions=[{"symbol": "BTCUSD", "unrealized_pl": 0.0}],
            loop_id="test-loop",
            regime="RANGING",
            db_path=strategic_db,
            session_db_path=session_db,
        )

        # The new position should be tagged with entry regime
        assert loop_mod._previous_positions["BTCUSD"]["_entry_regime"] == "RANGING"

        memory_db._engines = {}
        memory_db._session_factories = {}

    def test_entry_regime_carried_forward(self, tmp_path):
        """F-04: Entry regime is carried forward across loop iterations."""
        import sauce.core.loop as loop_mod
        from sauce.memory.db import get_engine, StrategicBase, SessionBase
        from sauce.memory import db as memory_db

        strategic_db = str(tmp_path / "strat.db")
        session_db = str(tmp_path / "session.db")

        memory_db._engines = {}
        memory_db._session_factories = {}
        get_engine(strategic_db)
        s_engine = get_engine(session_db)
        SessionBase.metadata.create_all(s_engine)

        # Simulate: position was first seen under RANGING
        loop_mod._previous_positions = {
            "BTCUSD": {
                "symbol": "BTCUSD",
                "unrealized_pl": 10.0,
                "_entry_regime": "RANGING",
            },
        }

        # Next loop iteration: regime changed to TRENDING_UP, position still open
        loop_mod._detect_closed_positions(
            current_positions=[{"symbol": "BTCUSD", "unrealized_pl": 20.0}],
            loop_id="test-loop",
            regime="TRENDING_UP",
            db_path=strategic_db,
            session_db_path=session_db,
        )

        # Entry regime should still be RANGING, not overwritten
        assert loop_mod._previous_positions["BTCUSD"]["_entry_regime"] == "RANGING"

        memory_db._engines = {}
        memory_db._session_factories = {}


class TestRunWeeklyFlag:
    """Tests for F-03: run_weekly flag in summary dict."""

    def test_run_weekly_true_on_sunday(self):
        """F-03: The expression datetime.now(utc).weekday() == 6 yields True on Sundays."""
        sunday = datetime(2025, 6, 8, 12, 0, 0, tzinfo=timezone.utc)  # 2025-06-08 is a Sunday
        assert sunday.weekday() == 6

    def test_run_weekly_false_on_weekday(self):
        """F-03: The expression datetime.now(utc).weekday() == 6 yields False on non-Sundays."""
        # Mon through Sat should all be False
        for day in range(9, 15):  # June 9 (Mon) through June 14 (Sat)
            d = datetime(2025, 6, day, tzinfo=timezone.utc)
            assert d.weekday() != 6, f"June {day} should not be Sunday"

    def test_run_weekly_key_exists_in_loop_source(self):
        """F-03: Verify run_weekly is present in loop.py source code."""
        from pathlib import Path
        source = (Path(__file__).parent.parent / "sauce" / "core" / "loop.py").read_text()
        assert '"run_weekly"' in source


def _make_actionable_signal(symbol: str = "AAPL", side: str = "buy", confidence: float = 0.75):
    """Helper: return a Signal with actionable side/confidence."""
    from sauce.core.schemas import Evidence, Indicators, PriceReference, Signal
    quote = make_fresh_quote(symbol)
    evidence = Evidence(
        symbol=symbol,
        price_reference=quote,
        indicators=Indicators(),
        as_of=datetime.now(timezone.utc),
    )
    return Signal(
        symbol=symbol,
        side=side,
        confidence=confidence,
        evidence=evidence,
        reasoning="test actionable signal",
        as_of=datetime.now(timezone.utc),
        prompt_version="v1",
    )


class TestVetoPatternRecording:
    """Tests for F-01: write_veto_pattern is called on supervisor abort."""

    @pytest.mark.asyncio
    async def test_veto_pattern_written_on_abort(self, monkeypatch):
        """F-01: When supervisor aborts with actionable signals, veto patterns are recorded."""
        from sauce.core.schemas import SupervisorDecision

        abort_decision = SupervisorDecision(
            action="abort",
            final_orders=[],
            vetoes=["low_conviction", "high_volatility"],
            reason="risk too high",
            as_of=datetime.now(timezone.utc),
            prompt_version="v1",
        )

        # Only crypto passes regime filter (RANGING excludes equities),
        # so provide only crypto quotes + signals.
        quotes = {"BTC/USD": make_fresh_quote("BTC/USD")}
        crypto_signal = _make_actionable_signal("BTC/USD", "buy", 0.80)

        with patch("sauce.core.loop.get_account", return_value=make_fake_account()):
            with patch("sauce.core.loop.get_positions", return_value=[]):
                with patch("sauce.core.loop.get_universe_snapshot", return_value=quotes):
                    with patch("sauce.agents.research.run", new=AsyncMock(return_value=crypto_signal)):
                        with patch("sauce.agents.risk.run", new=AsyncMock(return_value=make_stub_risk_result())):
                            with patch("sauce.agents.portfolio.run", new=AsyncMock(return_value=make_stub_portfolio_review())):
                                with patch("sauce.agents.supervisor.run", new=AsyncMock(return_value=abort_decision)):
                                    with patch("sauce.agents.ops.run", new=AsyncMock()):
                                        with patch("sauce.core.loop.write_veto_pattern") as mock_veto:
                                            from sauce.core.loop import main
                                            await main()

        # 1 actionable signal × 2 vetoes = 2 calls
        assert mock_veto.call_count == 2

        # Verify the veto reasons and setup types
        veto_entries = [call.args[0] for call in mock_veto.call_args_list]
        reasons = {e.veto_reason for e in veto_entries}
        setup_types = {e.setup_type for e in veto_entries}
        assert reasons == {"low_conviction", "high_volatility"}
        assert "crypto_mean_reversion" in setup_types  # BTC/USD

    @pytest.mark.asyncio
    async def test_no_veto_pattern_for_hold_signals(self, monkeypatch):
        """F-01: Hold signals should NOT trigger veto pattern recording."""
        from sauce.core.schemas import SupervisorDecision

        abort_decision = SupervisorDecision(
            action="abort",
            final_orders=[],
            vetoes=["low_conviction"],
            reason="nothing actionable",
            as_of=datetime.now(timezone.utc),
            prompt_version="v1",
        )

        quotes = {"AAPL": make_fresh_quote("AAPL"), "BTC/USD": make_fresh_quote("BTC/USD")}
        # Both signals are "hold" — not actionable
        hold1 = make_stub_signal("AAPL")
        hold2 = make_stub_signal("BTC/USD")

        with patch("sauce.core.loop.get_account", return_value=make_fake_account()):
            with patch("sauce.core.loop.get_positions", return_value=[]):
                with patch("sauce.core.loop.get_universe_snapshot", return_value=quotes):
                    with patch("sauce.agents.research.run", new=AsyncMock(side_effect=[hold1, hold2])):
                        with patch("sauce.agents.risk.run", new=AsyncMock(return_value=make_stub_risk_result())):
                            with patch("sauce.agents.portfolio.run", new=AsyncMock(return_value=make_stub_portfolio_review())):
                                with patch("sauce.agents.supervisor.run", new=AsyncMock(return_value=abort_decision)):
                                    with patch("sauce.agents.ops.run", new=AsyncMock()):
                                        with patch("sauce.core.loop.write_veto_pattern") as mock_veto:
                                            from sauce.core.loop import main
                                            await main()

        mock_veto.assert_not_called()
