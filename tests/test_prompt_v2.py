"""
tests/test_prompt_v2.py — Tests for the v2 (auditor-role) research prompt.

Covers:
- PROMPT_VERSION is "v2"
- SYSTEM_PROMPT contains auditor-role language
- _SYSTEM_PROMPT_V1 preserves old analyst language
- build_user_prompt memory context injection
- build_user_prompt returns valid JSON with all required keys
"""

import json
from datetime import datetime, timezone

import pytest

from sauce.prompts.research import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    _SYSTEM_PROMPT_V1,
    build_user_prompt,
)

_NOW = datetime(2024, 6, 10, 14, 30, 0, tzinfo=timezone.utc)

# Minimal kwargs for build_user_prompt so every call is valid.
_BASE_KWARGS: dict = dict(
    symbol="BTC-USD",
    mid=65000.0,
    bid=64990.0,
    ask=65010.0,
    sma_20=64500.0,
    sma_50=63000.0,
    rsi_14=55.0,
    atr_14=800.0,
    volume_ratio=1.2,
    as_of_utc=_NOW,
    prompt_version="v2",
)


# ── Prompt version ────────────────────────────────────────────────────────────


class TestPromptVersion:
    """PROMPT_VERSION constant."""

    def test_version_is_v2(self) -> None:
        assert PROMPT_VERSION == "v2"


# ── System prompts ────────────────────────────────────────────────────────────


class TestSystemPrompts:
    """SYSTEM_PROMPT (v2) and _SYSTEM_PROMPT_V1 content checks."""

    # v2 auditor language present
    def test_v2_contains_auditor(self) -> None:
        assert "auditor" in SYSTEM_PROMPT.lower()

    def test_v2_contains_do_not_generate(self) -> None:
        assert "do NOT generate trade ideas" in SYSTEM_PROMPT

    def test_v2_contains_approve_or_reject(self) -> None:
        assert "approve" in SYSTEM_PROMPT.lower()
        assert "reject" in SYSTEM_PROMPT.lower()

    def test_v2_contains_session_and_strategic(self) -> None:
        assert "session" in SYSTEM_PROMPT.lower()
        assert "strategic" in SYSTEM_PROMPT.lower()

    def test_v2_contains_anti_hallucination(self) -> None:
        assert "Do not invent" in SYSTEM_PROMPT

    def test_v2_contains_json_only(self) -> None:
        assert "Return ONLY valid JSON" in SYSTEM_PROMPT

    # v1 preserved
    def test_v1_contains_analyst(self) -> None:
        assert "analyst" in _SYSTEM_PROMPT_V1.lower()

    def test_v1_does_not_contain_auditor(self) -> None:
        assert "auditor" not in _SYSTEM_PROMPT_V1.lower()


# ── build_user_prompt — memory injection ──────────────────────────────────────


class TestMemoryInjection:
    """session_context_text and strategic_context_text injection."""

    def test_no_context_no_memory_keys(self) -> None:
        raw = build_user_prompt(**_BASE_KWARGS)
        payload = json.loads(raw)
        assert "session_memory" not in payload
        assert "strategic_memory" not in payload

    def test_session_context_injected(self) -> None:
        raw = build_user_prompt(**_BASE_KWARGS, session_context_text="sess text")
        payload = json.loads(raw)
        assert payload["session_memory"] == "sess text"
        assert "strategic_memory" not in payload

    def test_strategic_context_injected(self) -> None:
        raw = build_user_prompt(**_BASE_KWARGS, strategic_context_text="strat text")
        payload = json.loads(raw)
        assert payload["strategic_memory"] == "strat text"
        assert "session_memory" not in payload

    def test_both_contexts_injected(self) -> None:
        raw = build_user_prompt(
            **_BASE_KWARGS,
            session_context_text="sess",
            strategic_context_text="strat",
        )
        payload = json.loads(raw)
        assert payload["session_memory"] == "sess"
        assert payload["strategic_memory"] == "strat"

    def test_empty_string_context_not_injected(self) -> None:
        """Empty strings are falsy — keys should NOT appear."""
        raw = build_user_prompt(
            **_BASE_KWARGS,
            session_context_text="",
            strategic_context_text="",
        )
        payload = json.loads(raw)
        assert "session_memory" not in payload
        assert "strategic_memory" not in payload


# ── build_user_prompt — payload structure ─────────────────────────────────────


class TestPayloadStructure:
    """Ensure all required keys are present and values are correct."""

    @pytest.fixture()
    def payload(self) -> dict:
        raw = build_user_prompt(**_BASE_KWARGS)
        return json.loads(raw)

    def test_returns_valid_json(self) -> None:
        raw = build_user_prompt(**_BASE_KWARGS)
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_task_key(self, payload: dict) -> None:
        assert "task" in payload
        assert "trading signal" in payload["task"].lower()

    def test_timestamp_utc(self, payload: dict) -> None:
        assert payload["timestamp_utc"] == _NOW.isoformat()

    def test_prompt_version(self, payload: dict) -> None:
        assert payload["prompt_version"] == "v2"

    def test_symbol(self, payload: dict) -> None:
        assert payload["symbol"] == "BTC-USD"

    def test_price_reference(self, payload: dict) -> None:
        pr = payload["price_reference"]
        assert pr["bid"] == 64990.0
        assert pr["ask"] == 65010.0
        assert pr["mid"] == 65000.0

    def test_indicators_present(self, payload: dict) -> None:
        ind = payload["indicators"]
        assert ind["sma_20"] == 64500.0
        assert ind["sma_50"] == 63000.0
        assert ind["rsi_14"] == 55.0
        assert ind["atr_14"] == 800.0
        assert ind["volume_ratio"] == 1.2

    def test_null_indicators_are_none(self) -> None:
        """Optional indicators default to None/null in JSON."""
        raw = build_user_prompt(**_BASE_KWARGS)
        payload = json.loads(raw)
        assert payload["indicators"]["macd_line"] is None
        assert payload["indicators"]["vwap"] is None

    def test_asset_type_crypto(self) -> None:
        raw = build_user_prompt(**_BASE_KWARGS, is_crypto=True)
        payload = json.loads(raw)
        assert payload["asset_type"] == "crypto"

    def test_asset_type_equity(self) -> None:
        raw = build_user_prompt(**_BASE_KWARGS, is_crypto=False)
        payload = json.loads(raw)
        assert payload["asset_type"] == "equity"

    def test_required_output_schema(self, payload: dict) -> None:
        schema = payload["required_output_schema"]
        assert "side" in schema
        assert "confidence" in schema
        assert "reasoning" in schema

    def test_indicator_interpretation_guide(self, payload: dict) -> None:
        guide = payload["indicator_interpretation_guide"]
        assert "rsi_14" in guide
        assert "macd" in guide

    def test_confidence_calibration(self, payload: dict) -> None:
        assert "confidence_calibration" in payload

    def test_daily_trend_context_default_none(self, payload: dict) -> None:
        assert payload["daily_trend_context"] is None

    def test_signal_history_default_none(self, payload: dict) -> None:
        assert payload["signal_history"] is None

    def test_daily_trend_interpretation_none_when_no_context(
        self, payload: dict
    ) -> None:
        assert payload["daily_trend_interpretation"] is None

    def test_daily_trend_provided(self) -> None:
        trend = {"direction": "bullish", "strength": 0.7}
        raw = build_user_prompt(**_BASE_KWARGS, daily_trend_context=trend)
        payload = json.loads(raw)
        assert payload["daily_trend_context"] == trend
        assert payload["daily_trend_interpretation"] is not None

    def test_signal_history_provided(self) -> None:
        history = [{"side": "buy", "confidence": 0.6}]
        raw = build_user_prompt(**_BASE_KWARGS, signal_history=history)
        payload = json.loads(raw)
        sh = payload["signal_history"]
        assert sh["recent_signals"] == history

    def test_crypto_volume_ratio_guide(self) -> None:
        """Crypto flag changes volume_ratio interpretation text."""
        raw = build_user_prompt(**_BASE_KWARGS, is_crypto=True)
        payload = json.loads(raw)
        guide_text = payload["indicator_interpretation_guide"]["volume_ratio"]
        assert "crypto" in guide_text.lower()

    def test_equity_volume_ratio_guide(self) -> None:
        raw = build_user_prompt(**_BASE_KWARGS, is_crypto=False)
        payload = json.loads(raw)
        guide_text = payload["indicator_interpretation_guide"]["volume_ratio"]
        assert "crypto" not in guide_text.lower()
