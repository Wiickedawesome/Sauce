"""
core/schemas.py — All Pydantic v2 data models for the Sauce trading system.

Rules:
- All models use extra="forbid" — unknown fields raise ValidationError immediately.
- All models include as_of (datetime, UTC) and prompt_version (str).
- No raw dicts cross agent boundaries — always use these models.
- ValidationError anywhere in the pipeline = abort signal, log AuditEvent, continue.
"""

import math
import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Shared base ───────────────────────────────────────────────────────────────

class StrictModel(BaseModel):
    """Base for all Sauce models. Forbids extra fields globally."""

    model_config = ConfigDict(extra="forbid", frozen=False)


# ── Market Data ───────────────────────────────────────────────────────────────

class PriceReference(StrictModel):
    """Latest bid/ask/mid for a symbol at a given moment."""

    symbol: str
    bid: float = Field(..., ge=0.0)
    ask: float = Field(..., ge=0.0)
    mid: float = Field(..., ge=0.0)
    as_of: datetime = Field(..., description="Timestamp from the data API (UTC). Never datetime.now().")


# ── Research Agent ────────────────────────────────────────────────────────────

class Indicators(StrictModel):
    """Technical indicators computed from OHLCV history via pandas-ta."""

    sma_20: float | None = None
    sma_50: float | None = None
    rsi_14: float | None = None
    atr_14: float | None = None
    volume_ratio: float | None = None  # today's volume / 20-day avg volume
    volume_1d_avg: float | None = None  # estimated average daily volume in shares (Finding 2.5)
    macd_line: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    stoch_k: float | None = None
    stoch_d: float | None = None
    vwap: float | None = None


class Evidence(StrictModel):
    """Grounded evidence passed to Research agent and embedded in Signal."""

    symbol: str
    price_reference: PriceReference
    indicators: Indicators
    volatility: float | None = None
    volume: float | None = None
    as_of: datetime


class Signal(StrictModel):
    """
    Output of the Research agent.

    confidence < 0.5 is treated as hold by the system regardless of side.
    side="hold" short-circuits the rest of the pipeline immediately.
    """

    symbol: str
    side: Literal["buy", "sell", "hold"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: Evidence
    reasoning: str = Field(default="", description="Claude's reasoning (for audit purposes)")
    bear_case: str = Field(default="", description="Devil's advocate argument against the trade")
    as_of: datetime
    prompt_version: str

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: object) -> float:
        """Clamp confidence to [0.0, 1.0]. LLM may return out-of-range values."""
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(f) or math.isinf(f):
            return 0.0
        return max(0.0, min(1.0, f))


# ── Risk Agent ────────────────────────────────────────────────────────────────

class RiskChecks(StrictModel):
    """Individual boolean results for each risk rule."""

    max_position_pct_ok: bool
    max_exposure_ok: bool
    asset_class_ok: bool
    daily_loss_ok: bool
    volatility_ok: bool
    confidence_ok: bool


class RiskCheckResult(StrictModel):
    """
    Output of the Risk agent per signal.

    veto=True means the signal must not proceed to Execution.
    qty is only set when veto=False.
    """

    symbol: str
    side: Literal["buy", "sell", "hold"]
    veto: bool
    reason: str | None = None
    qty: float | None = Field(default=None, ge=0.0)
    checks: RiskChecks
    as_of: datetime
    prompt_version: str


# ── Execution Agent ───────────────────────────────────────────────────────────

class Order(StrictModel):
    """
    Output of the Execution agent.

    This is the final order object passed to broker.place_order().
    It must originate from an approved RiskCheckResult — never created directly.
    """

    symbol: str
    side: Literal["buy", "sell"]
    qty: float = Field(..., gt=0.0)
    order_type: Literal["market", "limit", "stop", "stop_limit"] = "limit"
    time_in_force: Literal["day", "gtc", "ioc", "fok"] = "day"
    limit_price: float | None = Field(default=None, ge=0.0)
    stop_price: float | None = Field(default=None, ge=0.0)
    stop_loss_price: float | None = Field(
        default=None, ge=0.0,
        description="Companion stop-loss price (ATR-based). Used to submit a "
                    "protective stop order after entry fill.",
    )
    take_profit_price: float | None = Field(
        default=None, ge=0.0,
        description="Companion take-profit price (ATR-based). Informational — "
                    "logged for audit but not automatically submitted.",
    )
    as_of: datetime
    prompt_version: str
    source: Literal["execution", "exit_research", "stop_loss"] | None = Field(
        default=None,
        description="Provenance tag: which agent/stage created this order.",
    )

# ── Supervisor Agent ──────────────────────────────────────────────────────────

class SupervisorDecision(StrictModel):
    """
    Final arbitration output from the Supervisor agent.

    action="execute" means all final_orders are safe to send to the broker.
    action="abort" means nothing is sent — vetoes contains the reasons.
    final_orders is always empty when action="abort".
    """

    action: Literal["execute", "abort"]
    final_orders: list[Order] = Field(default_factory=list)
    vetoes: list[str] = Field(default_factory=list)
    reason: str
    as_of: datetime
    prompt_version: str

    def model_post_init(self, __context: object) -> None:
        if self.action == "abort" and self.final_orders:
            raise ValueError("final_orders must be empty when action='abort'")
        if self.action == "execute" and not self.final_orders:
            raise ValueError("final_orders cannot be empty when action='execute'")


# ── Audit / Ops ───────────────────────────────────────────────────────────────

class AuditEvent(StrictModel):
    """
    Immutable audit log entry written by every agent and adapter.

    All writes to the DB use this model. Nothing is ever updated or deleted.
    loop_id ties all events from one cron run together.
    """

    loop_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal[
        "loop_start",
        "loop_end",
        "signal",
        "risk_check",
        "order",
        "order_prepared",
        "order_submitted",   # broker confirmed submission (not fill) — Finding 5.4
        "fill",              # kept for legacy compatibility; prefer order_submitted
        "veto",
        "error",
        "stub_called",
        "llm_call",
        "llm_response",
        "broker_call",
        "broker_response",
        "safety_check",
        "supervisor_decision",
        "portfolio_review",
        "ops_summary",
        "reconciliation",
        "session_boot",
        "market_context",
        "debate",
        "regime_transition",
        "tier_transition",
        "tier_check",
        "learning_drift_detected",
        "learning_weekly_report",
        "learning_calibration_analysis",
        "learning_behavior_updated",
        "validation_daily_check",
        "validation_passed",
        "validation_degradation",
        "exit_signal_generated",
    ]
    symbol: str | None = None
    payload: dict = Field(default_factory=dict)  # serialised model or error detail
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    prompt_version: str | None = None


# ── Portfolio Agent ───────────────────────────────────────────────────────────

class PositionNote(StrictModel):
    """Suggestion from the Portfolio agent for a single position."""

    symbol: str
    current_qty: float
    current_value_usd: float
    suggested_stop_price: float | None = None
    suggested_target_price: float | None = None
    over_concentrated: bool = False
    note: str = ""


class PortfolioReview(StrictModel):
    """Full output of the Portfolio agent for a single loop run."""

    positions: list[PositionNote] = Field(default_factory=list)
    total_exposure_pct: float = Field(..., ge=0.0, le=2.0)
    rebalance_needed: bool = False
    uncovered_symbols: list[str] = Field(default_factory=list)
    as_of: datetime
    prompt_version: str


# ── Exit Research Agent ───────────────────────────────────────────────────────

class ExitSignal(StrictModel):
    """Output of the exit research agent for a single open position."""

    symbol: str
    action: Literal["hold", "exit"]
    reason: str
    urgency: Literal["normal", "high"] = "normal"
    as_of: datetime
    prompt_version: str


class PositionPeakPnL(StrictModel):
    """Tracks the high-water mark of unrealized P&L for trailing stop logic."""

    symbol: str
    peak_unrealized_pnl: float
    peak_at: datetime


# ── Daily Stats ───────────────────────────────────────────────────────────────

class DailyStats(StrictModel):
    """Aggregated stats for a single trading day. Written by Ops agent."""

    date: str  # YYYY-MM-DD
    loop_runs: int = 0
    signals_generated: int = 0
    signals_vetoed: int = 0
    orders_placed: int = 0
    realized_pnl_usd: float = 0.0
    starting_nav_usd: float = 0.0
    ending_nav_usd: float = 0.0
    trading_paused: bool = False


# ── Type Aliases ──────────────────────────────────────────────────────────────

MarketRegime = Literal[
    "TRENDING_UP", "TRENDING_DOWN", "RANGING", "VOLATILE", "DEAD"
]

SetupType = Literal[
    "crypto_mean_reversion", "equity_trend_pullback", "crypto_breakout"
]

CapitalTier = Literal[
    "seed", "building", "growing", "scaling", "operating"
]

EconomicEventType = Literal["FOMC", "CPI", "NFP"]


# ── Economic Calendar ─────────────────────────────────────────────────────────

class EconomicEvent(StrictModel):
    """A scheduled major economic event (FOMC, CPI, NFP)."""

    date: datetime = Field(..., description="Scheduled release time in UTC.")
    event_type: EconomicEventType
    description: str = Field("", description="Short label, e.g. 'FOMC Rate Decision'.")


# ── Strategy Scanner (Sprint 3) ───────────────────────────────────────────────

class HardConditionResult(StrictModel):
    """Result of a single hard condition check."""

    label: str
    passed: bool
    detail: str = ""


class SoftConditionResult(StrictModel):
    """Result of a single soft condition check with its point contribution."""

    label: str
    triggered: bool
    points: float = Field(ge=0.0)


class Disqualification(StrictModel):
    """An automatic disqualifier that killed a setup."""

    reason: str


class SetupResult(StrictModel):
    """
    Output of the strategy scanner for one setup evaluation.

    If all hard conditions pass and no disqualifiers fire, score >= min_score
    means the setup is viable and should proceed to Claude for audit.
    """

    setup_type: SetupType
    symbol: str
    hard_conditions: list[HardConditionResult] = Field(default_factory=list)
    soft_conditions: list[SoftConditionResult] = Field(default_factory=list)
    disqualifiers: list[Disqualification] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=100.0)
    min_score: float = Field(ge=0.0, le=100.0)
    passed: bool = False
    evidence_narrative: str = ""
    as_of: datetime


# ── Session Memory Models (data/session_memory.db — wiped daily) ─────────────

class RegimeLogEntry(StrictModel):
    """Snapshot of the detected market regime at a given cycle."""

    timestamp: datetime
    regime_type: MarketRegime
    confidence: float = Field(ge=0.0, le=1.0)
    vix_proxy: float | None = None
    market_bias: str | None = None


class SignalLogEntry(StrictModel):
    """Record of every signal generated during the day, approved or not."""

    timestamp: datetime
    symbol: str
    setup_type: SetupType
    score: float = Field(ge=0.0, le=100.0)
    claude_decision: Literal["approve", "reject", "hold"]
    reason: str | None = None


class TradeLogEntry(StrictModel):
    """Intraday trade record including live P&L tracking."""

    timestamp: datetime
    symbol: str
    entry_price: float = Field(gt=0.0)
    direction: Literal["buy", "sell"]
    status: Literal["open", "closed", "cancelled"]
    unrealized_pnl: float = 0.0


class IntradayNarrativeEntry(StrictModel):
    """Running plain-English summary of market action during the day."""

    timestamp: datetime
    narrative_text: str


class SymbolCharacterEntry(StrictModel):
    """Per-symbol intraday behavior profile."""

    symbol: str
    signal_count_today: int = Field(ge=0, default=0)
    direction_consistency: float = Field(ge=-1.0, le=1.0, default=0.0)
    last_signal_result: Literal["win", "loss", "pending", "none"] = "none"


# ── Strategic Memory Models (data/strategic_memory.db — never wipes) ─────────

class SetupPerformanceEntry(StrictModel):
    """Historical performance record for a specific setup execution."""

    setup_type: SetupType
    symbol: str
    regime_at_entry: MarketRegime
    time_of_day_bucket: str  # e.g. "09:30-12:00", "12:00-14:00", "14:00-16:00"
    win: bool
    pnl: float
    hold_duration_minutes: float = Field(ge=0.0)
    date: str  # YYYY-MM-DD


class RegimeTransitionEntry(StrictModel):
    """Tracks how regimes change over time to detect patterns."""

    from_regime: MarketRegime
    to_regime: MarketRegime
    duration_minutes: float = Field(ge=0.0)
    count: int = Field(ge=1, default=1)


class VetoPatternEntry(StrictModel):
    """Tracks recurring veto reasons by setup type."""

    veto_reason: str
    setup_type: SetupType
    count: int = Field(ge=1, default=1)
    last_seen: datetime


class WeeklyPerformanceEntry(StrictModel):
    """Aggregated weekly performance by setup type."""

    week: str  # YYYY-Www (ISO week)
    setup_type: SetupType
    trades: int = Field(ge=0, default=0)
    win_rate: float = Field(ge=0.0, le=1.0, default=0.0)
    avg_pnl: float = 0.0
    sharpe: float = 0.0


class SymbolLearnedBehaviorEntry(StrictModel):
    """Learned indicator thresholds per symbol/setup from trade history."""

    symbol: str
    setup_type: SetupType
    optimal_rsi_entry: float | None = None
    avg_reversion_depth: float | None = None
    avg_bounce_magnitude: float | None = None
    sample_size: int = Field(ge=0, default=0)


class ClaudeCalibrationEntry(StrictModel):
    """Tracks Claude's stated confidence vs actual trade outcome."""

    date: str  # YYYY-MM-DD
    confidence_stated: float = Field(ge=0.0, le=1.0)
    outcome: Literal["win", "loss"]
    setup_type: SetupType


# ── Memory Context Aggregation Models ─────────────────────────────────────────

class SessionContext(StrictModel):
    """Aggregated session memory passed to Claude prompts."""

    regime_history: list[RegimeLogEntry] = Field(default_factory=list)
    signals_today: list[SignalLogEntry] = Field(default_factory=list)
    trades_today: list[TradeLogEntry] = Field(default_factory=list)
    narrative: str = ""
    symbol_characters: list[SymbolCharacterEntry] = Field(default_factory=list)
    as_of: datetime


class StrategicContext(StrictModel):
    """Aggregated strategic memory relevant to the current setup."""

    setup_performance: list[SetupPerformanceEntry] = Field(default_factory=list)
    regime_transitions: list[RegimeTransitionEntry] = Field(default_factory=list)
    relevant_veto_patterns: list[VetoPatternEntry] = Field(default_factory=list)
    weekly_trend: list[WeeklyPerformanceEntry] = Field(default_factory=list)
    symbol_behavior: SymbolLearnedBehaviorEntry | None = None
    claude_calibration: list[ClaudeCalibrationEntry] = Field(default_factory=list)
    as_of: datetime


# ── Boot & Market Context Models ─────────────────────────────────────────────

class BootContext(StrictModel):
    """Output of Agent 0 — session boot context assembled at market open."""

    was_reset: bool
    calendar_events: list[EconomicEvent] = Field(default_factory=list)
    strategic_context: StrategicContext
    is_suppressed: bool = False
    as_of: datetime


class MarketContext(StrictModel):
    """Output of Agent 1 — market context assembled each cycle."""

    regime: RegimeLogEntry
    regime_duration_minutes: float | None = None
    regime_aging_out: bool = False
    narrative: IntradayNarrativeEntry
    calendar_events: list[EconomicEvent] = Field(default_factory=list)
    is_dead: bool = False
    is_suppressed: bool = False
    as_of: datetime
