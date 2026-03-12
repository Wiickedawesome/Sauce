"""
core/narrative.py — Intraday narrative builder.

Pure Python, no LLM. Builds a one-paragraph plain-English description of
what the market has been doing today. Stored in session memory. Injected
into Claude's context for situational awareness.

Built programmatically from objective data — regime status, setup counts,
trade outcomes, account P&L, and notable observations.
"""

import logging
from datetime import datetime, timezone

import pandas as pd

from sauce.core.regime import RegimeDuration
from sauce.core.schemas import (
    IntradayNarrativeEntry,
    MarketRegime,
    RegimeLogEntry,
    SignalLogEntry,
    TradeLogEntry,
)

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

# Number of bars from session open used to measure opening momentum.
# At 30-min cadence, 1 bar = first 30 minutes of the session.
OPEN_MOMENTUM_BARS: int = 1

# Volume decline lookback (number of regime log entries to compare).
VOLUME_TREND_LOOKBACK: int = 4


def _format_pct(value: float) -> str:
    """Format a percentage with sign prefix."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _regime_label(regime: MarketRegime) -> str:
    """Human-readable regime label."""
    labels: dict[MarketRegime, str] = {
        "TRENDING_UP": "Trending Up",
        "TRENDING_DOWN": "Trending Down",
        "RANGING": "Ranging",
        "VOLATILE": "Volatile",
        "DEAD": "Dead",
    }
    return labels.get(regime, regime)


def _format_minutes(minutes: float) -> str:
    """Format minutes as 'Xh Ym' or 'Xm'."""
    total = int(minutes)
    if total >= 60:
        h = total // 60
        m = total % 60
        return f"{h}h {m}min" if m > 0 else f"{h}h"
    return f"{total}min"


def _build_open_momentum(spy_df: pd.DataFrame | None) -> str:
    """
    Describe the opening momentum from SPY (or primary benchmark) data.

    Returns a sentence fragment like "moderate upward momentum (+0.3% SPY first 30min)".
    If no data is available, returns a fallback.
    """
    if spy_df is None or len(spy_df) < 2:
        return "insufficient data for opening momentum"

    first_open = float(spy_df["open"].iloc[0])
    # Use close of the first bar(s) for momentum calculation.
    end_idx = min(OPEN_MOMENTUM_BARS, len(spy_df) - 1)
    end_close = float(spy_df["close"].iloc[end_idx])

    if first_open == 0.0:
        return "insufficient data for opening momentum"

    change_pct = ((end_close - first_open) / first_open) * 100.0

    if abs(change_pct) < 0.1:
        direction = "flat"
    elif abs(change_pct) < 0.3:
        direction = "slight upward" if change_pct > 0 else "slight downward"
    elif abs(change_pct) < 0.7:
        direction = "moderate upward" if change_pct > 0 else "moderate downward"
    else:
        direction = "strong upward" if change_pct > 0 else "strong downward"

    return f"{direction} momentum ({_format_pct(change_pct)} SPY first 30min)"


def _build_regime_sentence(
    regime_duration: RegimeDuration | None,
    regime_history: list[RegimeLogEntry],
) -> str:
    """
    Describe the current regime status and how long it has been active.

    Example: "Regime has been Ranging since 10:15 ET (3h 27min)"
    """
    if regime_duration is None and not regime_history:
        return "No regime data available."

    if regime_duration is not None:
        label = _regime_label(regime_duration.regime_type)
        duration_str = _format_minutes(regime_duration.active_minutes)
        aging = ""
        if regime_duration.aging_out:
            aging = " — approaching historical average duration, regime change may be imminent"
        return f"Regime has been {label} for {duration_str}{aging}."

    # Fallback: use latest regime log entry if no duration computed.
    latest = regime_history[-1]
    label = _regime_label(latest.regime_type)
    return f"Current regime: {label} (duration not computed)."


def _build_setup_summary(
    signals: list[SignalLogEntry],
    trades: list[TradeLogEntry],
) -> str:
    """
    Summarize setups detected, approvals, rejections, and outcomes.

    Example: "4 setups detected today: 2 approved (1 hit target +1.4%,
    1 currently +0.6% unrealized), 2 rejected."
    """
    if not signals:
        return "No setups detected today."

    total = len(signals)
    approved = [s for s in signals if s.claude_decision == "approve"]
    rejected = [s for s in signals if s.claude_decision == "reject"]
    held = [s for s in signals if s.claude_decision == "hold"]

    parts: list[str] = [f"{total} setup{'s' if total != 1 else ''} detected today"]

    if approved:
        # Match approved signals to trades by symbol and approximate time.
        open_trades = [t for t in trades if t.status == "open"]
        closed_trades = [t for t in trades if t.status == "closed"]
        sub: list[str] = []
        if closed_trades:
            winners = [t for t in closed_trades if t.unrealized_pnl > 0]
            losers = [t for t in closed_trades if t.unrealized_pnl <= 0]
            if winners:
                sub.append(f"{len(winners)} closed profitable")
            if losers:
                sub.append(f"{len(losers)} closed at loss")
        if open_trades:
            total_unreal = sum(t.unrealized_pnl for t in open_trades)
            sub.append(
                f"{len(open_trades)} open ({_format_pct(total_unreal)} unrealized)"
            )
        approved_detail = f" ({', '.join(sub)})" if sub else ""
        parts.append(f"{len(approved)} approved{approved_detail}")

    if rejected:
        # Include most common rejection reason if available.
        reasons = [r.reason for r in rejected if r.reason]
        reason_note = ""
        if reasons:
            reason_note = f" (most recent reason: {reasons[-1]})"
        parts.append(f"{len(rejected)} rejected{reason_note}")

    if held:
        parts.append(f"{len(held)} held")

    return ": ".join(parts[:1]) + " — " + ", ".join(parts[1:]) + "." if len(parts) > 1 else parts[0] + "."


def _build_pnl_sentence(trades: list[TradeLogEntry]) -> str:
    """Summarize account P&L for the day from trade records."""
    if not trades:
        return "No trades today."

    total_pnl = sum(t.unrealized_pnl for t in trades)
    open_count = sum(1 for t in trades if t.status == "open")
    closed_count = sum(1 for t in trades if t.status == "closed")

    status_parts: list[str] = []
    if open_count:
        status_parts.append(f"{open_count} open")
    if closed_count:
        status_parts.append(f"{closed_count} closed")

    status_str = f" ({', '.join(status_parts)})" if status_parts else ""
    return f"Account {_format_pct(total_pnl)} today{status_str}."


def _build_notable_observations(
    regime_history: list[RegimeLogEntry],
    regime_duration: RegimeDuration | None,
) -> str:
    """
    Add notable observations: volume trends, regime stability, etc.

    Observations are derived from regime log entries over the session.
    """
    observations: list[str] = []

    # Check regime stability — if the same regime has held for many entries.
    if regime_duration and regime_duration.active_minutes > 120:
        label = _regime_label(regime_duration.regime_type)
        observations.append(f"regime very stable ({label} for {_format_minutes(regime_duration.active_minutes)})")

    # Check for regime transitions (instability).
    if len(regime_history) >= 3:
        recent_types = [r.regime_type for r in regime_history[-4:]]
        unique_recent = len(set(recent_types))
        if unique_recent >= 3:
            observations.append("frequent regime changes — market indecisive")

    # Check for declining confidence trend.
    if len(regime_history) >= VOLUME_TREND_LOOKBACK:
        recent = regime_history[-VOLUME_TREND_LOOKBACK:]
        confidences = [r.confidence for r in recent]
        if all(
            confidences[i] > confidences[i + 1]
            for i in range(len(confidences) - 1)
        ):
            observations.append("regime confidence declining consistently")

    if not observations:
        return ""

    return "Notable: " + "; ".join(observations) + "."


def build_narrative(
    spy_df: pd.DataFrame | None,
    regime_history: list[RegimeLogEntry],
    regime_duration: RegimeDuration | None,
    signals: list[SignalLogEntry],
    trades: list[TradeLogEntry],
    as_of: datetime,
) -> IntradayNarrativeEntry:
    """
    Build the intraday narrative — one paragraph of plain English describing
    today's market action.

    Parameters
    ----------
    spy_df : pd.DataFrame | None
        SPY OHLCV data for the current session (used for opening momentum).
    regime_history : list[RegimeLogEntry]
        All regime classifications from today's session.
    regime_duration : RegimeDuration | None
        Current regime duration analysis from compute_regime_duration().
    signals : list[SignalLogEntry]
        All signals generated today (approved, rejected, held).
    trades : list[TradeLogEntry]
        All trades opened/closed today.
    as_of : datetime
        Current UTC timestamp.

    Returns
    -------
    IntradayNarrativeEntry
        Timestamped narrative text ready for session memory storage.
    """
    sentences: list[str] = []

    # 1. Session open momentum.
    momentum = _build_open_momentum(spy_df)
    sentences.append(f"Session opened with {momentum}.")

    # 2. Regime status + duration.
    regime_sentence = _build_regime_sentence(regime_duration, regime_history)
    sentences.append(regime_sentence)

    # 3. Setup counts and outcomes.
    setup_summary = _build_setup_summary(signals, trades)
    sentences.append(setup_summary)

    # 4. Account P&L.
    pnl_sentence = _build_pnl_sentence(trades)
    sentences.append(pnl_sentence)

    # 5. Notable observations.
    notable = _build_notable_observations(regime_history, regime_duration)
    if notable:
        sentences.append(notable)

    narrative_text = " ".join(sentences)

    logger.info("narrative_built | length=%d chars", len(narrative_text))

    return IntradayNarrativeEntry(
        timestamp=as_of,
        narrative_text=narrative_text,
    )
