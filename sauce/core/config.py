"""
core/config.py — Settings loaded from .env via Pydantic v2 BaseSettings.

All configuration must be accessed through get_settings(). Never read env
vars directly in agent or adapter code.
"""

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
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
    alpaca_api_key: str = Field(..., repr=False, description="Alpaca API key")
    alpaca_secret_key: str = Field(..., repr=False, description="Alpaca secret key")
    alpaca_paper: bool = Field(default=True, description="True = paper trading. Default MUST be True.")

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., repr=False, description="Anthropic API key")
    llm_model: str = Field(default="claude-sonnet-4-6", description="Anthropic model name")
    research_temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="LLM temperature for research agent")
    supervisor_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM temperature for supervisor agent")

    # ── Trading Universe ──────────────────────────────────────────────────────
    trading_universe_equities: str = Field(
        default=(
            # Mega-cap tech
            "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,AVGO,CRM,ORCL,ADBE,AMD,INTC,QCOM,CSCO,"
            # Financials
            "JPM,BAC,GS,MS,V,MA,AXP,BLK,SCHW,C,"
            # Healthcare
            "UNH,JNJ,LLY,PFE,ABBV,MRK,TMO,ABT,BMY,AMGN,"
            # Consumer / Retail
            "WMT,COST,HD,MCD,NKE,SBUX,TGT,LOW,PG,KO,PEP,"
            # Energy / Industrials
            "XOM,CVX,COP,NEE,CAT,DE,HON,UNP,BA,GE,"
            # ETFs
            "SPY,QQQ,IWM,DIA,XLF,XLE,XLK,XLV,ARKK,GLD,SLV,TLT"
        ),
        description="Comma-separated equity tickers",
    )
    trading_universe_crypto: str = Field(
        default=(
            "BTC/USD,ETH/USD,SOL/USD,LINK/USD,AVAX/USD,DOGE/USD,XRP/USD,"
            "ADA/USD,DOT/USD,MATIC/USD,UNI/USD,AAVE/USD,SHIB/USD,LTC/USD,"
            "BCH/USD,ALGO/USD,ATOM/USD,FIL/USD,NEAR/USD,APT/USD,"
            "ARB/USD,OP/USD,MKR/USD,GRT/USD,RENDER/USD,INJ/USD,SUI/USD,"
            "SEI/USD,PEPE/USD,FET/USD"
        ),
        description="Comma-separated Alpaca crypto pairs",
    )

    # ── Risk Limits ───────────────────────────────────────────────────────────
    max_position_pct: float = Field(default=0.08, ge=0.0, le=1.0)
    max_portfolio_exposure: float = Field(default=1.0, ge=0.0, le=2.0)
    max_daily_loss_pct: float = Field(default=0.03, ge=0.0, le=1.0)

    # ── Asset-Class Allocation Caps ───────────────────────────────────────────
    # Independent caps — they do not need to sum to 1.0 (remainder is cash).
    max_crypto_allocation_pct: float = Field(
        default=0.40, ge=0.0, le=1.0,
        description="Maximum fraction of equity that may be allocated to crypto "
                    "positions combined (e.g. 0.40 = 40%).",
    )
    max_equity_allocation_pct: float = Field(
        default=0.70, ge=0.0, le=1.0,
        description="Maximum fraction of equity that may be allocated to equity "
                    "positions combined (e.g. 0.70 = 70%).",
    )
    min_confidence: float = Field(default=0.40, ge=0.0, le=1.0)
    data_ttl_seconds: int = Field(default=120, ge=1)
    max_price_deviation: float = Field(default=0.01, ge=0.0, le=1.0)

    # ── Volatility / ATR parameters (Finding 1.8 / 2.4) ─────────────────────
    max_atr_ratio: float = Field(
        default=0.08, ge=0.0,
        description="Maximum ATR/price ratio permitted before vetoing a trade (8%).",
    )
    allow_no_atr: bool = Field(
        default=False,
        description="If True, approve trades when ATR data is unavailable. "
                    "Defaults to False (fail-closed) to prevent trading illiquid/new instruments.",
    )
    stop_loss_atr_multiple: float = Field(
        default=2.0, ge=0.0,
        description="Stop-loss distance = ATR × this multiple.",
    )
    profit_target_atr_multiple: float = Field(
        default=3.0, ge=0.0,
        description="Profit-target distance = ATR × this multiple.",
    )
    over_concentration_multiplier: float = Field(
        default=2.0, ge=1.0,
        description="A single position exceeding equity × max_position_pct × this "
                    "value is flagged as over-concentrated.",
    )

    # ── Order Execution parameters (Finding 1.9) ─────────────────────────────
    max_limit_price_premium: float = Field(
        default=0.001, ge=0.0, le=0.05,
        description="Maximum deviation of limit_price from ask/bid as a fraction "
                    "(e.g. 0.001 = 0.1%). Buy orders must not exceed ask × (1 + premium).",
    )

    # ── Liquidity / Spread parameters (Finding 2.5) ──────────────────────────
    max_spread_pct: float = Field(
        default=0.005, ge=0.0, le=1.0,
        description="Maximum permissible bid-ask spread as a fraction of mid price "
                    "(e.g. 0.005 = 0.5%). Trades on wider instruments are vetoed.",
    )
    max_volume_participation: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Maximum fraction of estimated daily volume the proposed order "
                    "may represent (e.g. 0.01 = 1%). Orders exceeding this are vetoed "
                    "to avoid excessive market impact (Finding 2.5).",
    )

    # ── Earnings blackout parameters (Finding 2.6) ────────────────────────────
    earnings_blackout_days: int = Field(
        default=1, ge=0,
        description="Number of calendar days before and after a scheduled earnings "
                    "announcement to suppress trading signals for that symbol (Finding 2.6).",
    )

    # ── Fund Fee Parameters ───────────────────────────────────────────────────
    annual_management_fee_pct: float = Field(
        default=0.01, ge=0.0, le=0.10,
        description="Annual management fee as a decimal (e.g. 0.01 = 1%). "
                    "Accrued daily as equity × rate / 252.",
    )
    performance_fee_pct: float = Field(
        default=0.20, ge=0.0, le=0.50,
        description="Performance fee rate above the high-water mark (e.g. 0.20 = 20%). "
                    "Applied only when ending NAV exceeds the prior high-water mark.",
    )

    # ── Data Feed ─────────────────────────────────────────────────────────────
    data_feed: str = Field(
        default="iex",
        description="Alpaca market data feed for equities: 'iex' (free) or 'sip' (paid). "
                    "Free-tier accounts MUST use 'iex'. Default is 'iex'.",
    )

    # ── Safety ────────────────────────────────────────────────────────────────
    trading_pause: bool = Field(default=False)
    confirm_live_trading: str = Field(
        default="",
        description="Must be set to 'LIVE-TRADING-CONFIRMED' when alpaca_paper=False. "
                    "Prevents accidental live-money trading.",
    )
    loop_timeout_seconds: int = Field(
        default=1500, ge=60,
        description="Maximum seconds a single loop iteration may run before being "
                    "cancelled. Default 1500 (25 min) for a 30-min cron cadence.",
    )
    stale_order_cancel_minutes: int = Field(
        default=30, ge=1,
        description="Cancel any open (unfilled) orders older than this many minutes "
                    "at the start of each loop run. Default 30 (one cron cycle).",
    )

    # ── Screener ──────────────────────────────────────────────────────────────
    screener_enabled: bool = Field(
        default=False,
        description="Enable dynamic equity screening from the full Alpaca market. "
                    "When True, the screener runs at the start of each loop to find "
                    "the top-N equity candidates. Crypto uses the .env list as-is.",
    )
    screener_max_candidates: int = Field(
        default=30, ge=1,
        description="Maximum number of equity symbols the screener passes to research.",
    )
    screener_min_dollar_volume: float = Field(
        default=5_000_000.0, ge=0.0,
        description="Minimum estimated daily dollar volume to qualify for screening.",
    )
    screener_price_min: float = Field(
        default=5.0, ge=0.0,
        description="Minimum last-trade price for screener eligibility.",
    )
    screener_price_max: float = Field(
        default=5000.0, ge=0.0,
        description="Maximum last-trade price for screener eligibility.",
    )

    # ── Prompt Tuning ─────────────────────────────────────────────────────────
    max_reasoning_len: int = Field(
        default=500, ge=50, le=5000,
        description="Maximum character length for LLM reasoning text embedded "
                    "in downstream prompts (sanitize_llm_text truncation limit).",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    db_path: str = Field(default="data/sauce.db")
    session_memory_db_path: str = Field(default="data/session_memory.db")
    strategic_memory_db_path: str = Field(default="data/strategic_memory.db")

    # ── Alerting (Finding 5.2) ────────────────────────────────────────────────
    alert_webhook_url: str = Field(
        default="",
        repr=False,
        description="Slack or generic webhook URL for critical alert notifications. "
                    "If empty, alerts are written to Python logging only.",
    )

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

    @field_validator("alpaca_api_key", "alpaca_secret_key", "anthropic_api_key")
    @classmethod
    def reject_blank_api_keys(cls, v: str) -> str:
        """Reject API keys that are empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("API key must not be empty or whitespace-only")
        return v

    @field_validator("data_feed")
    @classmethod
    def validate_data_feed(cls, v: str) -> str:
        allowed = {"iex", "sip"}
        if v.lower() not in allowed:
            raise ValueError(f"data_feed must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("alpaca_paper", mode="before")
    @classmethod
    def default_paper_to_true(cls, v: object) -> object:
        """Guard: if ALPACA_PAPER is somehow empty/None, default to True (paper)."""
        if v is None or v == "":
            return True
        return v

    @model_validator(mode="after")
    def require_live_trading_confirmation(self) -> "Settings":
        """Prevent accidental live trading without explicit confirmation."""
        if not self.alpaca_paper and self.confirm_live_trading != "LIVE-TRADING-CONFIRMED":
            raise ValueError(
                "Live trading (ALPACA_PAPER=false) requires "
                "CONFIRM_LIVE_TRADING='LIVE-TRADING-CONFIRMED' in .env. "
                "This safeguard prevents accidental real-money execution."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Loaded once at first call. All code must use this function — never
    instantiate Settings() directly in production code.
    """
    return Settings()
