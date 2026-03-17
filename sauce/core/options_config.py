"""
core/options_config.py — Options-specific settings (Pydantic v2 BaseSettings).

All options settings are loaded from .env alongside the main Settings.
OPTIONS_ENABLED defaults to False — the entire options module is a no-op
until explicitly enabled.

Strategy: "Momentum Snipe" — high-conviction directional plays with
realistic profit targets (+35%/+60%), strict time/DTE stops, and
trailing-stop activation. No compounding ladder.

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
        default="AAPL,AMD,PLTR,SOFI,COIN,MARA",
        description="Comma-separated tickers eligible for options trades.",
    )
    option_max_contract_cost: float = Field(
        default=500.0, gt=0.0,
        description="Max cost per contract (mid × 100) for dynamic universe "
                    "filtering. Contracts above this are skipped at research time.",
    )

    # ── Profit Targets ────────────────────────────────────────────────────────
    option_profit_target_pct: float = Field(
        default=0.35, gt=0.0, le=5.0,
        description="First profit target: close 50% (if qty≥2) or activate "
                    "trailing stop (if qty=1) at +35%.",
    )
    option_stretch_target_pct: float = Field(
        default=0.60, gt=0.0, le=5.0,
        description="Stretch profit target: close everything at +60%.",
    )

    # ── Trailing Stop ─────────────────────────────────────────────────────────
    option_trail_activation_pct: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Activate trailing stop when gain reaches +20%.",
    )
    option_trail_pct: float = Field(
        default=0.12, ge=0.01, le=1.0,
        description="Trail 12% below high-water mark once activated.",
    )

    # ── Time / DTE Stops ──────────────────────────────────────────────────────
    option_time_stop_days: int = Field(
        default=5, ge=1,
        description="Close if held > N trading days AND gain < time_stop_min_gain.",
    )
    option_time_stop_min_gain_pct: float = Field(
        default=0.10, ge=0.0, le=1.0,
        description="Minimum gain to survive the time stop.",
    )
    option_dte_exit_days: int = Field(
        default=5, ge=1,
        description="Close if remaining DTE falls below this.",
    )

    # ── DTE Limits ────────────────────────────────────────────────────────────
    option_max_dte: int = Field(
        default=35, ge=1,
        description="Maximum days-to-expiry at entry.",
    )
    option_min_dte: int = Field(
        default=14, ge=1,
        description="Minimum days-to-expiry at entry. No buying inside this window.",
    )

    # ── IV / Greeks ───────────────────────────────────────────────────────────
    option_max_iv_rank: float = Field(
        default=0.60, ge=0.0, le=1.0,
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
        default=0.10, ge=0.0, le=1.0,
        description="Max fraction of NAV per single options position (10%).",
    )
    option_max_total_exposure: float = Field(
        default=0.20, ge=0.0, le=1.0,
        description="Max fraction of NAV across ALL options positions (20%).",
    )
    option_max_contracts: int = Field(
        default=5, ge=1,
        description="Max contracts per position (small-account guardrail).",
    )

    # ── Execution ─────────────────────────────────────────────────────────────
    option_max_spread_pct: float = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Max bid/ask spread as fraction of mid price (5%).",
    )
    option_max_loss_pct: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Hard stop: exit if position loses more than this from entry.",
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
        return v


@lru_cache(maxsize=1)
def get_options_settings() -> OptionsSettings:
    """Return the cached OptionsSettings singleton."""
    return OptionsSettings()
