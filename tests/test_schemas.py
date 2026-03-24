"""
tests/test_schemas.py — Validation tests for core/schemas.py Pydantic models.

Every schema must:
- Accept valid input.
- Reject invalid input with ValidationError.
- Reject extra fields.
- Enforce field constraints (types, ranges, literals).
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from sauce.core.schemas import (
    AuditEvent,
    Order,
    PriceReference,
)

NOW = datetime.now(UTC)
PROMPT_V = "v1"


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_price_ref(**overrides: object) -> PriceReference:
    base = {"symbol": "AAPL", "bid": 150.0, "ask": 150.10, "mid": 150.05, "as_of": NOW}
    return PriceReference(**{**base, **overrides})  # type: ignore[arg-type]


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
            symbol="AAPL",
            bid=150.0,
            ask=150.1,
            mid=150.05,
            as_of=NOW,
            unexpected_field="boom",  # type: ignore[call-arg]
        )


def test_price_reference_missing_as_of_rejected() -> None:
    with pytest.raises(ValidationError):
        PriceReference(symbol="AAPL", bid=150.0, ask=150.1, mid=150.05)  # type: ignore[call-arg]


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


# ── AuditEvent ────────────────────────────────────────────────────────────────


def test_audit_event_valid() -> None:
    event = AuditEvent(
        event_type="error",
        payload={"detail": "test"},
        prompt_version=PROMPT_V,
    )
    assert event.event_type == "error"
    assert event.loop_id  # auto-generated UUID


def test_audit_event_invalid_type_rejected() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(event_type="made_up_event")  # type: ignore[arg-type]


def test_audit_event_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        AuditEvent(event_type="error", surprise="field")  # type: ignore[call-arg]


def test_audit_event_loop_id_auto_generated() -> None:
    e1 = AuditEvent(event_type="error")
    e2 = AuditEvent(event_type="llm_call")
    assert e1.loop_id != e2.loop_id  # each gets a unique UUID


def test_audit_event_accepts_loop_boundaries_and_safety_events() -> None:
    start = AuditEvent(event_type="loop_start")
    safety = AuditEvent(event_type="safety_check")
    supervisor = AuditEvent(event_type="supervisor_decision")
    end = AuditEvent(event_type="loop_end")
    assert start.event_type == "loop_start"
    assert safety.event_type == "safety_check"
    assert supervisor.event_type == "supervisor_decision"
    assert end.event_type == "loop_end"
