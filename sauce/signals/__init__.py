"""signals package — Multi-timeframe signal analysis and confluence scoring."""

from sauce.signals.timeframes import MultiTimeframeContext, TimeframeAnalysis, fetch_multi_timeframe
from sauce.signals.confluence import ConfluenceResult, SignalTier, compute_confluence

__all__ = [
    "MultiTimeframeContext",
    "TimeframeAnalysis",
    "fetch_multi_timeframe",
    "ConfluenceResult",
    "SignalTier",
    "compute_confluence",
]
