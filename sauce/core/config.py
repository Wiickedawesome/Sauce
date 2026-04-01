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
    alpaca_paper: bool = Field(
        default=True, description="True = paper trading. Default MUST be True."
    )

    # ── LLM ───────────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(..., repr=False, description="Anthropic API key")
    llm_model: str = Field(default="claude-sonnet-4-6", description="Anthropic model name")

    # ── LLM Provider Routing (hybrid local/API) ───────────────────────────────
    dual_analysis_provider: str = Field(
        default="ollama", description="Provider for dual analysis calls: 'anthropic' or 'ollama'"
    )
    pm_verdict_provider: str = Field(
        default="anthropic", description="Provider for PM verdict calls: 'anthropic' or 'ollama'"
    )
    reflection_provider: str = Field(
        default="ollama", description="Provider for reflection calls: 'anthropic' or 'ollama'"
    )
    ollama_model: str = Field(
        default="qwen2.5:7b-instruct-q4_K_M", description="Ollama model name"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Ollama server base URL"
    )

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
            "ADA/USD,DOT/USD,UNI/USD,AAVE/USD,SHIB/USD,LTC/USD,"
            "BCH/USD,FIL/USD,"
            "ARB/USD,GRT/USD,RENDER/USD,PEPE/USD"
        ),
        description="Comma-separated Alpaca crypto pairs",
    )

    # ── Risk Limits ───────────────────────────────────────────────────────────
    max_portfolio_exposure: float = Field(default=1.0, ge=0.0, le=2.0)
    max_daily_loss_pct: float = Field(default=0.03, ge=0.0, le=1.0)

    # ── Asset-Class Allocation Caps ───────────────────────────────────────────
    max_options_allocation_pct: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of equity that may be allocated to options "
        "positions combined (e.g. 0.20 = 20%). Keep lower due to leverage.",
    )

    # ── Options Parameters ────────────────────────────────────────────────────
    options_enabled: bool = Field(
        default=False,
        description="Enable the options entry and exit pipeline. Defaults to False so "
        "options trading is an explicit opt-in.",
    )
    options_dte_min: int = Field(
        default=7,
        ge=0,
        le=90,
        description="Minimum days to expiration for new positions.",
    )
    options_dte_max: int = Field(
        default=45,
        ge=1,
        le=365,
        description="Maximum days to expiration for new positions.",
    )
    options_dte_exit_threshold: int = Field(
        default=2,
        ge=0,
        le=14,
        description="Close positions at this DTE to avoid expiration risk.",
    )
    options_delta_min: float = Field(
        default=0.25,
        ge=0.05,
        le=0.50,
        description="Minimum absolute delta for contract selection.",
    )
    options_delta_max: float = Field(
        default=0.40,
        ge=0.20,
        le=0.80,
        description="Maximum absolute delta for contract selection.",
    )
    options_max_contracts_per_position: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum contracts per options position.",
    )
    options_max_premium_pct: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Maximum premium as % of equity per position (5%).",
    )
    options_underlyings: str = Field(
        default="SPY,QQQ,TSLA,NVDA,AMD,AAPL,AMZN,META,GOOGL",
        description="Comma-separated underlyings approved for options trading.",
    )

    min_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    data_ttl_seconds: int = Field(default=120, ge=1)
    max_price_deviation: float = Field(default=0.01, ge=0.0, le=1.0)

    # ── Volatility / ATR parameters (Finding 1.8 / 2.4) ─────────────────────
    max_atr_ratio: float = Field(
        default=0.08,
        ge=0.0,
        description="Maximum ATR/price ratio permitted before vetoing a trade (8%).",
    )
    max_atr_ratio_crypto: float = Field(
        default=0.15,
        ge=0.0,
        description="Maximum ATR/price ratio for crypto pairs (15%). Crypto is "
        "inherently more volatile than equities.",
    )
    allow_no_atr: bool = Field(
        default=True,
        description="If True, approve trades when ATR data is unavailable. "
        "Defaults to True to avoid vetoing crypto pairs with short history.",
    )
    stop_loss_atr_multiple: float = Field(
        default=2.0,
        ge=0.0,
        description="Stop-loss distance = ATR × this multiple.",
    )
    profit_target_atr_multiple: float = Field(
        default=3.0,
        ge=0.0,
        description="Profit-target distance = ATR × this multiple.",
    )
    # ── Order Execution parameters (Finding 1.9) ─────────────────────────────
    max_limit_price_premium: float = Field(
        default=0.001,
        ge=0.0,
        le=0.05,
        description="Maximum deviation of limit_price from ask/bid as a fraction "
        "(e.g. 0.001 = 0.1%). Buy orders must not exceed ask × (1 + premium).",
    )

    # ── Liquidity / Spread parameters (Finding 2.5) ──────────────────────────
    max_spread_pct: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Maximum permissible bid-ask spread as a fraction of mid price "
        "(e.g. 0.005 = 0.5%). Trades on wider instruments are vetoed.",
    )
    max_spread_pct_crypto: float = Field(
        default=0.015,
        ge=0.0,
        le=1.0,
        description="Maximum permissible bid-ask spread for crypto pairs "
        "(e.g. 0.015 = 1.5%). Crypto spreads are wider than equities.",
    )
    max_volume_participation: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of estimated daily volume the proposed order "
        "may represent (e.g. 0.01 = 1%). Orders exceeding this are vetoed "
        "to avoid excessive market impact (Finding 2.5).",
    )

    # ── Data Feed ─────────────────────────────────────────────────────────────
    data_feed: str = Field(
        default="iex",
        description="Alpaca market data feed for equities: 'iex' (free) or 'sip' (paid). "
        "Free-tier accounts MUST use 'iex'. Default is 'iex'. "
        "NOTE: IEX provides 15-min delayed quotes for equities. "
        "Crypto quotes are real-time regardless of data_feed setting. "
        "SIP provides real-time NBBO but requires a paid market data subscription.",
    )

    # ── Loop Cadence ──────────────────────────────────────────────────────────
    loop_interval_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Loop cadence in minutes. Default is 5.",
    )

    # ── Signal Scoring ────────────────────────────────────────────────────────
    signal_threshold_base: int = Field(
        default=65,
        ge=0,
        le=100,
        description="Base threshold for signal scoring (0–100). "
        "Regime shift adjusts this: bullish −10, bearish +10.",
    )
    stop_loss_pct: float = Field(
        default=0.03,
        ge=0.0,
        le=0.50,
        description="Hard stop-loss as a fraction of entry price (3%).",
    )
    trail_activation_pct: float = Field(
        default=0.03,
        ge=0.0,
        le=0.50,
        description="Gain required to activate trailing stop (3%).",
    )
    trail_pct: float = Field(
        default=0.02,
        ge=0.0,
        le=0.50,
        description="Trailing stop distance below high water mark (2%).",
    )
    profit_target_pct: float = Field(
        default=0.06,
        ge=0.0,
        le=1.0,
        description="Take-profit target as a fraction of entry price (6%).",
    )
    rsi_exhaustion_threshold: float = Field(
        default=72.0,
        ge=50.0,
        le=100.0,
        description="RSI level at which to exit for momentum exhaustion.",
    )
    max_hold_hours: float = Field(
        default=48.0,
        ge=1.0,
        description="Maximum hours to hold a position before time-stop check.",
    )
    time_stop_min_gain: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum gain required to survive a time stop (1%).",
    )

    # ── Analyst Committee LLM temperatures ───────────────────────────────────
    research_temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Temperature for the analyst dual-analysis call (bull/bear).",
    )
    supervisor_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Temperature for the PM verdict call (approve/reject).",
    )
    reflection_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for the post-trade reflection call.",
    )

    # ── Safety ────────────────────────────────────────────────────────────────
    trading_pause: bool = Field(default=False)
    confirm_live_trading: str = Field(
        default="",
        description="Must be set to 'LIVE-TRADING-CONFIRMED' when alpaca_paper=False. "
        "Prevents accidental live-money trading.",
    )

    # ── Database ──────────────────────────────────────────────────────────────
    # SQLite fallback (local development)
    db_path: str = Field(default="data/sauce.db")

    # Supabase PostgreSQL (production)
    supabase_url: str = Field(
        default="",
        description="Supabase project URL (e.g., https://xxx.supabase.co). "
        "If empty, falls back to SQLite at db_path.",
    )
    supabase_service_role_key: str = Field(
        default="",
        repr=False,
        description="Supabase service role key (bypasses RLS). Required for DB writes.",
    )
    supabase_db_url: str = Field(
        default="",
        repr=False,
        description="Direct PostgreSQL connection URL for SQLAlchemy. "
        "Format: postgresql://user:pass@host:port/dbname. "
        "Use pooler URL (port 6543) for serverless deployments.",
    )

    @property
    def use_supabase(self) -> bool:
        """True if Supabase is configured (all required fields present)."""
        return bool(self.supabase_url and self.supabase_db_url)

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
    def options_universe(self) -> list[str]:
        """Parsed list of underlyings approved for options trading."""
        return [s.strip().upper() for s in self.options_underlyings.split(",") if s.strip()]

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
    return Settings()  # type: ignore[call-arg]
