"""
core/config.py — Settings loaded from .env via Pydantic v2 BaseSettings.

All configuration must be accessed through get_settings(). Never read env
vars directly in agent or adapter code.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Typed, validated settings for the entire Sauce system.

    All fields map 1-to-1 to keys in .env / .env.example.
    Missing required keys will raise a ValidationError at startup — fail fast.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars (OS noise, etc.)
        case_sensitive=False,
    )

    # ── Broker ────────────────────────────────────────────────────────────────
    alpaca_api_key: str = Field(..., description="Alpaca API key")
    alpaca_secret_key: str = Field(..., description="Alpaca secret key")
    alpaca_paper: bool = Field(default=True, description="True = paper trading. Default MUST be True.")

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: str = Field(default="github", description="'github' or 'anthropic'")
    github_token: str = Field(default="", description="GitHub token for GitHub Models API")
    llm_model: str = Field(default="claude-3-5-sonnet", description="Model name on LLM endpoint")
    anthropic_api_key: str = Field(default="", description="Anthropic API key (fallback provider)")

    # ── Trading Universe ──────────────────────────────────────────────────────
    trading_universe_equities: str = Field(
        default="AAPL,MSFT,GOOGL,AMZN,NVDA,SPY,QQQ",
        description="Comma-separated equity tickers",
    )
    trading_universe_crypto: str = Field(
        default="BTC/USD,ETH/USD",
        description="Comma-separated Alpaca crypto pairs",
    )

    # ── Risk Limits ───────────────────────────────────────────────────────────
    max_position_pct: float = Field(default=0.05, ge=0.0, le=1.0)
    max_portfolio_exposure: float = Field(default=1.0, ge=0.0, le=2.0)
    max_daily_loss_pct: float = Field(default=0.02, ge=0.0, le=1.0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    data_ttl_seconds: int = Field(default=120, ge=1)
    max_price_deviation: float = Field(default=0.01, ge=0.0, le=1.0)

    # ── Safety ────────────────────────────────────────────────────────────────
    trading_pause: bool = Field(default=False)

    # ── Database ──────────────────────────────────────────────────────────────
    db_path: str = Field(default="data/sauce.db")

    # ── Prompt Versioning ─────────────────────────────────────────────────────
    prompt_version: str = Field(default="v1")

    # ── Computed helpers ──────────────────────────────────────────────────────

    @property
    def equity_universe(self) -> list[str]:
        """Parsed list of equity tickers from the comma-separated env var."""
        return [s.strip().upper() for s in self.trading_universe_equities.split(",") if s.strip()]

    @property
    def crypto_universe(self) -> list[str]:
        """Parsed list of crypto pairs from the comma-separated env var."""
        return [s.strip().upper() for s in self.trading_universe_crypto.split(",") if s.strip()]

    @property
    def full_universe(self) -> list[str]:
        """Combined equity + crypto trading universe."""
        return self.equity_universe + self.crypto_universe

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"github", "anthropic"}
        if v.lower() not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("alpaca_paper", mode="before")
    @classmethod
    def default_paper_to_true(cls, v: object) -> object:
        """Guard: if ALPACA_PAPER is somehow empty/None, default to True (paper)."""
        if v is None or v == "":
            return True
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Loaded once at first call. All code must use this function — never
    instantiate Settings() directly in production code.
    """
    return Settings()
