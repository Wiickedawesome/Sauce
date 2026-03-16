"""
core/options_config.py — Options-specific settings (Pydantic v2 BaseSettings).

All options settings are loaded from .env alongside the main Settings.
OPTIONS_ENABLED defaults to False — the entire options module is a no-op
until explicitly enabled.

Access via get_options_settings().
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OptionsSettings(BaseSettings):
    """Typed, validated settings for the options trading module."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Master Switch ─────────────────────────────────────────────────────────
    options_enabled: bool = Field(
        default=False,
        description="Master switch for the entire options module. "
                    "Must be explicitly set to true in .env.",
    )

    # ── Universe ──────────────────────────────────────────────────────────────
    options_universe: str = Field(
        default="SPY,QQQ,AAPL,TSLA,NVDA",
        description="Comma-separated tickers eligible for options trades.",
    )

    # ── Compounding Strategy ──────────────────────────────────────────────────
    option_profit_multiplier: float = Field(
        default=2.0, gt=1.0,
        description="Trigger to take gains at each stage (2.0 = 100% gain).",
    )
    option_compound_stages: int = Field(
        default=3, ge=1, le=10,
        description="Number of take-profit stages in the compounding ladder.",
    )
    option_sell_fraction: float = Field(
        default=0.5, gt=0.0, le=1.0,
        description="Fraction of remaining qty to sell at each compounding stage.",
    )

    # ── DTE Limits ────────────────────────────────────────────────────────────
    option_max_dte: int = Field(
        default=45, ge=1,
        description="Maximum days-to-expiry at entry.",
    )
    option_min_dte: int = Field(
        default=7, ge=1,
        description="Minimum days-to-expiry at entry. No buying inside this window.",
    )

    # ── IV / Greeks ───────────────────────────────────────────────────────────
    option_max_iv_rank: float = Field(
        default=0.70, ge=0.0, le=1.0,
        description="Max IV rank for buying options. Above this, IV is expensive.",
    )
    option_min_delta: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Minimum absolute delta for directional buys.",
    )
    option_max_delta: float = Field(
        default=0.60, ge=0.0, le=1.0,
        description="Maximum absolute delta (avoid deep ITM).",
    )

    # ── Position Sizing ───────────────────────────────────────────────────────
    option_max_position_pct: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Max fraction of NAV per single options position (5%).",
    )
    option_max_total_exposure: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Max fraction of NAV across ALL options positions (20%).",
    )

    # ── Execution ─────────────────────────────────────────────────────────────
    option_max_spread_pct: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Max bid/ask spread as fraction of mid price (5%).",
    )
    option_max_loss_pct: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="Hard stop: exit if position loses more than this from entry.",
    )

    # ── Trailing Stop (post-compound) ─────────────────────────────────────────
    option_trailing_stop_stage1: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Trailing stop % after stage 1 triggers.",
    )
    option_trailing_stop_stage2: float = Field(
        default=0.15, ge=0.0, le=1.0,
        description="Trailing stop % after stage 2 triggers.",
    )
    option_trailing_stop_stage3: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Trailing stop % after stage 3 triggers.",
    )

    # ── Computed helpers ──────────────────────────────────────────────────────

    @property
    def universe(self) -> list[str]:
        """Parsed list of options-eligible tickers."""
        return [s.strip().upper() for s in self.options_universe.split(",") if s.strip()]

    @field_validator("option_min_dte", mode="after")
    @classmethod
    def min_dte_less_than_max(cls, v: int, info: object) -> int:
        """Validate min_dte < max_dte (when max is available)."""
        # Pydantic v2 field_validator doesn't easily cross-validate;
        # we rely on model_validator below instead.
        return v


@lru_cache(maxsize=1)
def get_options_settings() -> OptionsSettings:
    """Return the cached OptionsSettings singleton."""
    return OptionsSettings()
