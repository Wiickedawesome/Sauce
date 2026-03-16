"""
signals/options_confluence.py — Directional bias for options from equity signals.

Takes existing equity confluence data (multi-timeframe scores, setup results,
regime) and computes an OptionsBias for a given underlying.  This bridges the
existing equity signal pipeline with the options module.

The options module does NOT invent its own directional thesis — it reuses the
equity signal system, adding IV + momentum filters on top.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sauce.core.options_config import get_options_settings
from sauce.core.options_schemas import OptionsBias
from sauce.signals.confluence import ConfluenceResult, SignalTier

logger = logging.getLogger(__name__)


def compute_options_bias(
    symbol: str,
    confluence: ConfluenceResult,
    iv_rank: float | None = None,
    regime: str | None = None,
) -> OptionsBias:
    """
    Derive an OptionsBias from equity confluence signals.

    Parameters
    ----------
    symbol:      Underlying ticker.
    confluence:  Result from the equity multi-timeframe confluence engine.
    iv_rank:     Current IV rank for the symbol (0.0-1.0). None if unavailable.
    regime:      Current market regime string (e.g. "trending", "volatile").

    Returns
    -------
    OptionsBias with direction, confidence, and filter flags.
    """
    cfg = get_options_settings()

    # Direction from confluence score
    if confluence.score > 0.25:
        direction = "bullish"
    elif confluence.score < -0.25:
        direction = "bearish"
    else:
        direction = "neutral"

    # Base confidence from confluence
    confidence = abs(confluence.score)

    # Apply confluence tier adjustment
    confidence += confluence.confidence_adjustment

    # Clamp
    confidence = max(0.0, min(1.0, confidence))

    # IV rank filter
    iv_ok = True
    if iv_rank is not None and iv_rank > cfg.option_max_iv_rank:
        iv_ok = False
        confidence *= 0.5  # penalize but don't zero out

    # Regime filter — suppress in extreme regimes
    regime_ok = True
    if regime and regime.lower() in ("crisis", "extreme_volatility"):
        regime_ok = False
        confidence *= 0.3

    # Momentum alignment — strong tier = aligned
    momentum_aligned = confluence.tier in (SignalTier.S1, SignalTier.S2)

    # Neutral direction should not produce options signals
    if direction == "neutral":
        confidence = 0.0

    reasoning_parts = [
        f"Confluence score={confluence.score:.3f}, tier={confluence.tier.value}",
        f"IV rank={'N/A' if iv_rank is None else f'{iv_rank:.2f}'}",
        f"Regime={'N/A' if regime is None else regime}",
        f"Direction={direction}, confidence={confidence:.3f}",
    ]

    return OptionsBias(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        iv_rank=iv_rank,
        regime_ok=regime_ok,
        momentum_aligned=momentum_aligned,
        reasoning=" | ".join(reasoning_parts),
    )
