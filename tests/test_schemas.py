"""
tests/test_schemas.py — Validation tests for core/schemas.py Pydantic models.

Every schema must:
- Accept valid input.
- Reject invalid input with ValidationError.
- Reject extra fields.
- Enforce field constraints (types, ranges, literals).
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from sauce.core.schemas import (
    AuditEvent,
    DailyStats,
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


NOW = datetime.now(timezone.utc)
PROMPT_V = "v1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_price_ref(**overrides: object) -> PriceReference:
    base = {"symbol": "AAPL", "bid": 150.0, "ask": 150.10, "mid": 150.05, "as_of": NOW}
    return PriceReference(**{**base, **overrides})  # type: ignore[arg-type]


def make_evidence(**overrides: object) -> Evidence:
    base = {
        "symbol": "AAPL",
        "price_reference": make_price_ref(),
        "indicators": Indicators(sma_20=148.0, rsi_14=55.0),
        "as_of": NOW,
    }
    return Evidence(**{**base, **overrides})  # type: ignore[arg-type]


def make_signal(**overrides: object) -> Signal:
    base = {
        "symbol": "AAPL",
        "side": "buy",
        "confidence": 0.75,
        "evidence": make_evidence(),
        "as_of": NOW,
        "prompt_version": PROMPT_V,
    }
    return Signal(**{**base, **overrides})  # type: ignore[arg-type]


def make_risk_checks(**overrides: object) -> RiskChecks:
    base = {
        "max_position_pct_ok": True,
        "max_exposure_ok": True,
        "daily_loss_ok": True,
        "volatility_ok": True,
        "confidence_ok": True,
    }
    return RiskChecks(**{**base, **overrides})  # type: ignore[arg-type]


def make_risk_result(**overrides: object) -> RiskCheckResult:
    base = {
        "symbol": "AAPL",
        "side": "buy",
        "veto": False,
        "qty": 10.0,
        "checks": make_risk_checks(),
        "as_of": NOW,
        "prompt_version": PROMPT_V,
    }
    return RiskCheckResult(**{**base, **overrides})  # type: ignore[arg-type]


def make_order(**overrides: object) -> Order:
    base = {
        "symbol": "AAPL",
        "side": "buy",
        "qty": 10.0,
        "order_type": "limit",
        "time_in_force": "day",
        "limit_price": 150.05,
        "as_of": NOW,
        "prompt_version": PROMPT_V,
    }
    return Order(**{**base, **overrides})  # type: ignore[arg-type]


# ── PriceReference ────────────────────────────────────────────────────────────

def test_price_reference_valid() -> None:
    ref = make_price_ref()
    assert ref.symbol == "AAPL"
    assert ref.mid == 150.05


def test_price_reference_negative_price_rejected() -> None:
    with pytest.raises(ValidationError):
        make_price_ref(bid=-1.0)


def test_price_reference_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        PriceReference(
            symbol="AAPL", bid=150.0, ask=150.1, mid=150.05, as_of=NOW,
            unexpected_field="boom",  # type: ignore[call-arg]
        )


def test_price_reference_missing_as_of_rejected() -> None:
    with pytest.raises(ValidationError):
        PriceReference(symbol="AAPL", bid=150.0, ask=150.1, mid=150.05)  # type: ignore[call-arg]


# ── Signal ────────────────────────────────────────────────────────────────────

def test_signal_valid_buy() -> None:
    s = make_signal()
    assert s.side == "buy"
    assert s.confidence == 0.75


def test_signal_hold_valid() -> None:
    s = make_signal(side="hold", confidence=0.0)
    assert s.side == "hold"


def test_signal_invalid_side_rejected() -> None:
    with pytest.raises(ValidationError):
        make_signal(side="short")  # type: ignore[arg-type]


def test_signal_confidence_above_1_rejected() -> None:
    with pytest.raises(ValidationError):
        make_signal(confidence=1.1)


def test_signal_confidence_below_0_rejected() -> None:
    with pytest.raises(ValidationError):
        make_signal(confidence=-0.1)


def test_signal_missing_prompt_version_rejected() -> None:
    with pytest.raises(ValidationError):
        Signal(
            symbol="AAPL", side="buy", confidence=0.7,
            evidence=make_evidence(), as_of=NOW,
        )  # type: ignore[call-arg]


def test_signal_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        make_signal(hallucinated_field="bad")  # type: ignore[arg-type]


# ── RiskCheckResult ───────────────────────────────────────────────────────────

def test_risk_result_approved_valid() -> None:
    r = make_risk_result(veto=False, qty=10.0)
    assert r.veto is False
    assert r.qty == 10.0


def test_risk_result_veto_valid() -> None:
    r = make_risk_result(veto=True, qty=None, reason="exposure too high")
    assert r.veto is True
    assert r.qty is None


def test_risk_result_negative_qty_rejected() -> None:
    with pytest.raises(ValidationError):
        make_risk_result(qty=-5.0)


def test_risk_result_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        make_risk_result(fake_key="oops")  # type: ignore[arg-type]


# ── Order ─────────────────────────────────────────────────────────────────────

def test_order_valid() -> None:
    o = make_order()
    assert o.symbol == "AAPL"
    assert o.qty == 10.0
    assert o.order_type == "limit"


def test_order_zero_qty_rejected() -> None:
    with pytest.raises(ValidationError):
        make_order(qty=0.0)


def test_order_invalid_order_type_rejected() -> None:
    with pytest.raises(ValidationError):
        make_order(order_type="twap")  # type: ignore[arg-type]


def test_order_invalid_tif_rejected() -> None:
    with pytest.raises(ValidationError):
        make_order(time_in_force="opg")  # type: ignore[arg-type]


def test_order_invalid_side_rejected() -> None:
    with pytest.raises(ValidationError):
        make_order(side="short")  # type: ignore[arg-type]


# ── SupervisorDecision ────────────────────────────────────────────────────────

def test_supervisor_execute_valid() -> None:
    decision = SupervisorDecision(
        action="execute",
        final_orders=[make_order()],
        reason="all checks passed",
        as_of=NOW,
        prompt_version=PROMPT_V,
    )
    assert decision.action == "execute"
    assert len(decision.final_orders) == 1


def test_supervisor_abort_valid() -> None:
    decision = SupervisorDecision(
        action="abort",
        final_orders=[],
        vetoes=["risk agent vetoed"],
        reason="veto",
        as_of=NOW,
        prompt_version=PROMPT_V,
    )
    assert decision.action == "abort"


def test_supervisor_abort_with_orders_rejected() -> None:
    """abort + non-empty final_orders must raise — this is a core safety invariant."""
    with pytest.raises(ValidationError):
        SupervisorDecision(
            action="abort",
            final_orders=[make_order()],  # NOT allowed on abort
            reason="broken",
            as_of=NOW,
            prompt_version=PROMPT_V,
        )


def test_supervisor_execute_empty_orders_rejected() -> None:
    """execute + empty final_orders must raise."""
    with pytest.raises(ValidationError):
        SupervisorDecision(
            action="execute",
            final_orders=[],  # NOT allowed on execute
            reason="should fail",
            as_of=NOW,
            prompt_version=PROMPT_V,
        )


# ── AuditEvent ────────────────────────────────────────────────────────────────

def test_audit_event_valid() -> None:
    event = AuditEvent(
        event_type="loop_start",
        payload={"loop_id": "abc"},
        prompt_version=PROMPT_V,
    )
    assert event.event_type == "loop_start"
    assert event.loop_id  # auto-generated UUID


def test_audit_event_invalid_type_rejected() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(event_type="made_up_event")  # type: ignore[arg-type]


def test_audit_event_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(event_type="loop_start", surprise="field")  # type: ignore[call-arg]


def test_audit_event_loop_id_auto_generated() -> None:
    e1 = AuditEvent(event_type="loop_start")
    e2 = AuditEvent(event_type="loop_end")
    assert e1.loop_id != e2.loop_id  # each gets a unique UUID


# ── DailyStats ────────────────────────────────────────────────────────────────

def test_daily_stats_valid() -> None:
    ds = DailyStats(date="2026-03-04", starting_nav_usd=10000.0, ending_nav_usd=10100.0)
    assert ds.loop_runs == 0
    assert ds.trading_paused is False
