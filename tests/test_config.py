"""
tests/test_config.py — Tests for core/config.py Settings.

pydantic-settings reads from environment variables. Tests inject env vars via
monkeypatch.setenv, which is the correct and realistic way to test Settings.
No real API calls. No live credentials required.
"""

import pytest
from pydantic import ValidationError

from sauce.core.config import Settings, get_settings

# ── Helper ────────────────────────────────────────────────────────────────────


def set_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the minimum required env vars so Settings() can construct."""
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


# ── Required field validation ─────────────────────────────────────────────────


def test_settings_loads_with_required_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.alpaca_api_key == "test_key"
    assert s.alpaca_secret_key == "test_secret"


def test_settings_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # bypass .env on disk; rely solely on monkeypatched env


def test_settings_missing_secret_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # bypass .env on disk; rely solely on monkeypatched env


# ── Paper trading default ─────────────────────────────────────────────────────


def test_alpaca_paper_defaults_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """CRITICAL: paper must default to True. Never default to live."""
    set_required(monkeypatch)
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    s = Settings(_env_file=None)
    assert s.alpaca_paper is True


def test_alpaca_paper_can_be_set_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("CONFIRM_LIVE_TRADING", "LIVE-TRADING-CONFIRMED")
    s = Settings(_env_file=None)
    assert s.alpaca_paper is False


def test_live_trading_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting ALPACA_PAPER=false without CONFIRM_LIVE_TRADING must raise."""
    set_required(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.delenv("CONFIRM_LIVE_TRADING", raising=False)
    with pytest.raises(ValidationError, match="LIVE-TRADING-CONFIRMED"):
        Settings(_env_file=None)


def test_alpaca_paper_empty_string_defaults_to_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard: if ALPACA_PAPER is empty, default to paper (safe)."""
    set_required(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER", "")
    s = Settings(_env_file=None)
    assert s.alpaca_paper is True


# ── LLM config ─────────────────────────────────────────────────────────────


def test_anthropic_api_key_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_llm_model_defaults_to_claude_sonnet(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.llm_model == "claude-sonnet-4-6"


# ── Risk limit bounds ─────────────────────────────────────────────────────────


def test_max_daily_loss_pct_default(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.max_daily_loss_pct == 0.03


def test_min_confidence_default(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.min_confidence == 0.3


def test_asset_specific_ttl_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.equity_data_ttl_seconds == 30
    assert s.crypto_data_ttl_seconds == 120
    assert s.option_data_ttl_seconds == 60


def test_stale_order_cancel_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.stale_order_cancel_equity_minutes == 15
    assert s.stale_order_cancel_crypto_minutes == 30


def test_intraday_regime_and_delayed_equity_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.intraday_regime_refresh_hours == 4
    assert s.allow_delayed_equity_entries is False


def test_net_pnl_assumption_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    s = Settings(_env_file=None)
    assert s.crypto_fee_bps == 15
    assert s.crypto_slippage_bps == 15
    assert s.option_fee_bps == 50
    assert s.option_slippage_bps == 50


def test_options_enabled_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    monkeypatch.delenv("OPTIONS_ENABLED", raising=False)
    s = Settings(_env_file=None)
    assert s.options_enabled is False


# ── Trading universe parsing ──────────────────────────────────────────────────


def test_equity_universe_parses_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "AAPL,MSFT, NVDA ")
    s = Settings(_env_file=None)
    assert s.equity_universe == ["AAPL", "MSFT", "NVDA"]


def test_crypto_universe_parses_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("TRADING_UNIVERSE_CRYPTO", "BTC/USD,ETH/USD")
    s = Settings(_env_file=None)
    assert s.crypto_universe == ["BTC/USD", "ETH/USD"]


def test_full_universe_combines_both(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("TRADING_UNIVERSE_EQUITIES", "AAPL")
    monkeypatch.setenv("TRADING_UNIVERSE_CRYPTO", "BTC/USD")
    s = Settings(_env_file=None)
    assert s.full_universe == ["AAPL", "BTC/USD"]


# ── Safety flag ───────────────────────────────────────────────────────────────


def test_trading_pause_defaults_to_false(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    monkeypatch.delenv("TRADING_PAUSE", raising=False)
    s = Settings(_env_file=None)
    assert s.trading_pause is False


def test_trading_pause_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    monkeypatch.setenv("TRADING_PAUSE", "true")
    s = Settings(_env_file=None)
    assert s.trading_pause is True


# ── Singleton cache ───────────────────────────────────────────────────────────


def test_get_settings_returns_same_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    set_required(monkeypatch)
    get_settings.cache_clear()
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    get_settings.cache_clear()
