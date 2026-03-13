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
    monkeypatch.setenv("LLM_PROVIDER", "github")
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
    """If macro suppression is active the loop should exit before market/account work."""
    from sauce.core.schemas import BootContext, StrategicContext

    suppressed_boot_ctx = BootContext(
        was_reset=False,
        calendar_events=[],
        strategic_context=StrategicContext(as_of=datetime.now(timezone.utc)),
        is_suppressed=True,
        as_of=datetime.now(timezone.utc),
    )

    with patch("sauce.agents.session_boot.run", new=AsyncMock(return_value=suppressed_boot_ctx)):
        with patch("sauce.agents.market_context.run", new=AsyncMock()) as mock_market_context:
            with patch("sauce.core.loop.get_account") as mock_account:
                from sauce.core.loop import main
                await main()

    mock_market_context.assert_not_called()
    mock_account.assert_not_called()
    assert count_events_by_type("loop_start") >= 1
    assert count_events_by_type("loop_end") >= 1
    payload = get_latest_event_payload("safety_check")
    assert payload["check"] == "macro_suppression"
    assert payload["result"] is True


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
