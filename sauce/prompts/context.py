"""
prompts/context.py — Session and strategic context builders for Claude prompts.

Sprint 4: Claude Repositioning.

Converts structured memory data (SessionContext, StrategicContext) into
plain-English paragraphs that Claude can reason about. These paragraphs are
injected into the research prompt as [SESSION MEMORY] and [STRATEGIC MEMORY].
"""

from sauce.core.schemas import SessionContext, StrategicContext


def build_session_paragraph(ctx: SessionContext) -> str:
    """
    Convert session memory into a plain-English paragraph.

    Covers: regime evolution, signals today, trades today, symbol characters,
    and any existing narrative text.

    Returns empty string if the context contains no meaningful data.
    """
    parts: list[str] = []

    # ── Regime history ────────────────────────────────────────────────────────
    if ctx.regime_history:
        latest = ctx.regime_history[-1]
        if len(ctx.regime_history) == 1:
            parts.append(
                f"Regime has been {latest.regime_type} since session start "
                f"(confidence {latest.confidence:.0%})."
            )
        else:
            transitions = []
            for i in range(1, len(ctx.regime_history)):
                prev = ctx.regime_history[i - 1]
                curr = ctx.regime_history[i]
                t_str = curr.timestamp.strftime("%H:%M")
                transitions.append(f"{prev.regime_type} → {curr.regime_type} at {t_str}")
            parts.append(
                f"Regime transitions today: {', '.join(transitions)}. "
                f"Current regime: {latest.regime_type} "
                f"(confidence {latest.confidence:.0%})."
            )

    # ── Signals today ─────────────────────────────────────────────────────────
    if ctx.signals_today:
        approved = [s for s in ctx.signals_today if s.claude_decision == "approve"]
        rejected = [s for s in ctx.signals_today if s.claude_decision == "reject"]
        held = [s for s in ctx.signals_today if s.claude_decision == "hold"]
        signal_parts = []
        if approved:
            signal_parts.append(f"{len(approved)} approved")
        if rejected:
            signal_parts.append(f"{len(rejected)} rejected")
        if held:
            signal_parts.append(f"{len(held)} held")
        parts.append(
            f"{len(ctx.signals_today)} signals evaluated today: "
            f"{', '.join(signal_parts)}."
        )
        # Detail each signal
        for sig in ctx.signals_today:
            t_str = sig.timestamp.strftime("%H:%M")
            reason_str = f" — {sig.reason}" if sig.reason else ""
            parts.append(
                f"  {t_str} {sig.symbol} {sig.setup_type} "
                f"(score {sig.score}): {sig.claude_decision}{reason_str}."
            )

    # ── Trades today ──────────────────────────────────────────────────────────
    if ctx.trades_today:
        open_trades = [t for t in ctx.trades_today if t.status == "open"]
        closed_trades = [t for t in ctx.trades_today if t.status == "closed"]
        if open_trades:
            for t in open_trades:
                parts.append(
                    f"Open position: {t.symbol} {t.direction} "
                    f"at {t.entry_price:.4f} "
                    f"(unrealized P&L: {t.unrealized_pnl:+.2f})."
                )
        else:
            parts.append("No open positions.")
        if closed_trades:
            total_pnl = sum(t.unrealized_pnl for t in closed_trades)
            parts.append(
                f"{len(closed_trades)} closed trade(s) today, "
                f"net P&L: {total_pnl:+.2f}."
            )

    # ── Symbol characters ─────────────────────────────────────────────────────
    if ctx.symbol_characters:
        for sc in ctx.symbol_characters:
            parts.append(
                f"{sc.symbol} character: {sc.signal_count_today} signals today, "
                f"direction consistency {sc.direction_consistency:+.2f}, "
                f"last result: {sc.last_signal_result}."
            )

    # ── Narrative ─────────────────────────────────────────────────────────────
    if ctx.narrative:
        parts.append(ctx.narrative)

    return " ".join(parts)


def build_strategic_paragraph(
    ctx: StrategicContext,
    setup_type: str | None = None,
    symbol: str | None = None,
) -> str:
    """
    Convert strategic memory into a plain-English paragraph.

    Covers: setup performance stats, veto patterns, weekly trend,
    symbol learned behavior, and Claude calibration history.

    Parameters
    ----------
    ctx:        Aggregated strategic context from memory_db.
    setup_type: Current setup being evaluated (filters relevance narrative).
    symbol:     Current symbol being evaluated.

    Returns empty string if the context contains no meaningful data.
    """
    parts: list[str] = []

    # ── Setup performance ─────────────────────────────────────────────────────
    if ctx.setup_performance:
        total = len(ctx.setup_performance)
        wins = sum(1 for sp in ctx.setup_performance if sp.win)
        win_rate = wins / total if total > 0 else 0.0
        avg_pnl = sum(sp.pnl for sp in ctx.setup_performance) / total if total else 0.0
        avg_hold = (
            sum(sp.hold_duration_minutes for sp in ctx.setup_performance) / total
            if total
            else 0.0
        )
        # Compute expectancy
        winning = [sp.pnl for sp in ctx.setup_performance if sp.win]
        losing = [sp.pnl for sp in ctx.setup_performance if not sp.win]
        avg_win = sum(winning) / len(winning) if winning else 0.0
        avg_loss = sum(losing) / len(losing) if losing else 0.0

        label = "this setup"
        if setup_type and symbol:
            label = f"{symbol} {setup_type}"
        elif setup_type:
            label = setup_type
        elif symbol:
            label = f"{symbol} setups"

        parts.append(
            f"Historical performance for {label}: "
            f"{total} occurrences, {win_rate:.0%} win rate, "
            f"avg win {avg_win:+.2f}, avg loss {avg_loss:+.2f}, "
            f"avg P&L {avg_pnl:+.2f}, "
            f"avg hold {avg_hold:.0f} min."
        )

    # ── Regime transitions ────────────────────────────────────────────────────
    if ctx.regime_transitions:
        transition_strs = []
        for rt in ctx.regime_transitions:
            transition_strs.append(
                f"{rt.from_regime} → {rt.to_regime} "
                f"(avg {rt.duration_minutes:.0f} min, {rt.count}x)"
            )
        parts.append(
            f"Regime transition patterns: {'; '.join(transition_strs)}."
        )

    # ── Veto patterns ─────────────────────────────────────────────────────────
    if ctx.relevant_veto_patterns:
        for vp in ctx.relevant_veto_patterns:
            parts.append(
                f"WARNING: '{vp.veto_reason}' has vetoed {vp.setup_type} "
                f"{vp.count} time(s), last seen {vp.last_seen.strftime('%Y-%m-%d')}."
            )

    # ── Weekly trend ──────────────────────────────────────────────────────────
    if ctx.weekly_trend:
        latest_week = ctx.weekly_trend[-1]
        parts.append(
            f"Recent week ({latest_week.week}): {latest_week.setup_type} — "
            f"{latest_week.trades} trades, {latest_week.win_rate:.0%} win rate, "
            f"avg P&L {latest_week.avg_pnl:+.2f}, Sharpe {latest_week.sharpe:.2f}."
        )

    # ── Symbol learned behavior ───────────────────────────────────────────────
    if ctx.symbol_behavior:
        sb = ctx.symbol_behavior
        behavior_parts = [f"{sb.symbol} {sb.setup_type} learned behavior:"]
        if sb.optimal_rsi_entry is not None:
            behavior_parts.append(f"optimal RSI entry {sb.optimal_rsi_entry:.1f}")
        if sb.avg_reversion_depth is not None:
            behavior_parts.append(f"avg reversion depth {sb.avg_reversion_depth:.2%}")
        if sb.avg_bounce_magnitude is not None:
            behavior_parts.append(f"avg bounce {sb.avg_bounce_magnitude:.2%}")
        behavior_parts.append(f"(sample size: {sb.sample_size})")
        parts.append(" ".join(behavior_parts))

    # ── Claude calibration ────────────────────────────────────────────────────
    if ctx.claude_calibration:
        cal_total = len(ctx.claude_calibration)
        cal_wins = sum(1 for c in ctx.claude_calibration if c.outcome == "win")
        cal_rate = cal_wins / cal_total if cal_total else 0.0
        avg_conf = (
            sum(c.confidence_stated for c in ctx.claude_calibration) / cal_total
            if cal_total
            else 0.0
        )
        parts.append(
            f"Claude calibration ({cal_total} recent trades): "
            f"stated avg confidence {avg_conf:.0%}, "
            f"actual win rate {cal_rate:.0%}."
        )
        # Flag miscalibration
        if cal_total >= 5 and abs(avg_conf - cal_rate) > 0.15:
            if avg_conf > cal_rate:
                parts.append(
                    "NOTE: Claude is overconfident — "
                    "stated confidence exceeds actual outcomes."
                )
            else:
                parts.append(
                    "NOTE: Claude is underconfident — "
                    "actual outcomes exceed stated confidence."
                )

    return " ".join(parts)
