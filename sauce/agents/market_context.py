"""Agent 1 — Market Context Builder.

Assembles the current market context each loop cycle:
1. Fetches SPY OHLCV data and computes indicators.
2. Classifies the current market regime.
3. Detects regime transitions and records them.
4. Builds the intraday narrative.
5. Returns a fully-typed MarketContext for downstream agents.

No LLM calls, no broker calls — pure data assembly.
"""

import logging
from datetime import datetime, timezone

from sauce.adapters import market_data
from sauce.adapters.db import log_event
from sauce.indicators.core import compute_all
from sauce.memory.db import (
    get_session_context,
    write_narrative,
    write_regime_log,
    write_regime_transition,
)
from sauce.core.config import get_settings
from sauce.memory.narrative import build_narrative
from sauce.core.regime import classify_regime, compute_regime_duration
from sauce.core.schemas import (
    AuditEvent,
    BootContext,
    Indicators,
    IntradayNarrativeEntry,
    MarketContext,
    RegimeLogEntry,
    RegimeTransitionEntry,
)

logger = logging.getLogger(__name__)

_MIN_BARS_REQUIRED = 10


def _compute_spy_indicators(spy_df: "pd.DataFrame") -> Indicators:
    """Compute the full indicator set for SPY via the canonical library."""
    return compute_all(spy_df, is_crypto=False)


# ---------------------------------------------------------------------------
# Degraded context — returned when SPY data is unavailable
# ---------------------------------------------------------------------------

def _degraded_context(
    boot_ctx: BootContext,
    as_of: datetime,
) -> MarketContext:
    """Return a minimal MarketContext when SPY data cannot be obtained."""
    regime = RegimeLogEntry(
        timestamp=as_of,
        regime_type="DEAD",
        confidence=0.0,
    )
    narrative = IntradayNarrativeEntry(
        timestamp=as_of,
        narrative_text="SPY data unavailable — market context degraded.",
    )
    return MarketContext(
        regime=regime,
        regime_duration_minutes=None,
        regime_aging_out=False,
        narrative=narrative,
        calendar_events=boot_ctx.calendar_events,
        is_dead=True,
        is_suppressed=boot_ctx.is_suppressed,
        as_of=as_of,
    )


# ---------------------------------------------------------------------------
# Agent entry-point
# ---------------------------------------------------------------------------

async def run(loop_id: str, boot_ctx: BootContext) -> MarketContext:
    """Execute Agent 1 — assemble market context for this loop cycle.

    Parameters
    ----------
    loop_id:
        Unique identifier for the current loop iteration.
    boot_ctx:
        Output of Agent 0 (session boot).

    Returns
    -------
    MarketContext
        Fully-typed market context for downstream agents.
    """
    settings = get_settings()
    session_db = settings.session_memory_db_path
    strategic_db = settings.strategic_memory_db_path
    as_of = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Step 1 — Fetch SPY OHLCV data
    # ------------------------------------------------------------------
    try:
        spy_df = market_data.get_history("SPY", timeframe="30Min", bars=60)
    except market_data.MarketDataError:
        logger.exception("Failed to fetch SPY data")
        ctx = _degraded_context(boot_ctx, as_of)
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="market_context",
                timestamp=as_of,
                payload={"status": "degraded", "reason": "spy_data_unavailable"},
            ),
            settings.db_path,
        )
        return ctx

    if len(spy_df) < _MIN_BARS_REQUIRED:
        logger.warning(
            "SPY data insufficient: %d bars (need %d)",
            len(spy_df),
            _MIN_BARS_REQUIRED,
        )
        ctx = _degraded_context(boot_ctx, as_of)
        log_event(
            AuditEvent(
                loop_id=loop_id,
                event_type="market_context",
                timestamp=as_of,
                payload={
                    "status": "degraded",
                    "reason": "insufficient_bars",
                    "bars": len(spy_df),
                },
            ),
            settings.db_path,
        )
        return ctx

    # ------------------------------------------------------------------
    # Step 2 — Compute SPY indicators
    # ------------------------------------------------------------------
    indicators = _compute_spy_indicators(spy_df)

    # ------------------------------------------------------------------
    # Step 3 — Classify regime & persist
    # ------------------------------------------------------------------
    regime = classify_regime(spy_df, indicators, as_of)
    write_regime_log(regime, session_db)

    # ------------------------------------------------------------------
    # Step 4 — Retrieve session context (regime history, signals, trades)
    # ------------------------------------------------------------------
    session_ctx = get_session_context(session_db)

    # ------------------------------------------------------------------
    # Step 5 — Compute regime duration
    # ------------------------------------------------------------------
    duration = compute_regime_duration(
        session_ctx.regime_history,
        boot_ctx.strategic_context.regime_transitions,
        as_of,
    )

    # ------------------------------------------------------------------
    # Step 6 — Detect regime transition
    # ------------------------------------------------------------------
    if len(session_ctx.regime_history) >= 2:
        prev = session_ctx.regime_history[-2]
        if prev.regime_type != regime.regime_type:
            transition_minutes = (
                regime.timestamp - prev.timestamp
            ).total_seconds() / 60.0
            transition = RegimeTransitionEntry(
                from_regime=prev.regime_type,
                to_regime=regime.regime_type,
                duration_minutes=max(transition_minutes, 0.0),
                count=1,
            )
            write_regime_transition(transition, strategic_db)

    # ------------------------------------------------------------------
    # Step 7 — Build narrative & persist
    # ------------------------------------------------------------------
    narrative = build_narrative(
        spy_df,
        session_ctx.regime_history,
        duration,
        session_ctx.signals_today,
        session_ctx.trades_today,
        as_of,
    )
    write_narrative(narrative, session_db)

    # ------------------------------------------------------------------
    # Step 8 — Assemble MarketContext
    # ------------------------------------------------------------------
    market_ctx = MarketContext(
        regime=regime,
        regime_duration_minutes=duration.active_minutes if duration else None,
        regime_aging_out=duration.aging_out if duration else False,
        narrative=narrative,
        calendar_events=boot_ctx.calendar_events,
        is_dead=(regime.regime_type == "DEAD"),
        is_suppressed=boot_ctx.is_suppressed,
        as_of=as_of,
    )

    # ------------------------------------------------------------------
    # Step 9 — Audit log
    # ------------------------------------------------------------------
    log_event(
        AuditEvent(
            loop_id=loop_id,
            event_type="market_context",
            timestamp=as_of,
            payload={
                "status": "ok",
                "regime": regime.regime_type,
                "confidence": regime.confidence,
                "duration_minutes": market_ctx.regime_duration_minutes,
                "aging_out": market_ctx.regime_aging_out,
                "is_dead": market_ctx.is_dead,
                "is_suppressed": market_ctx.is_suppressed,
            },
        ),
        settings.db_path,
    )

    return market_ctx
