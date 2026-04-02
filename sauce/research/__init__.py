"""Research and calibration tooling for Sauce."""

from sauce.research.backtest import BacktestConfig, BacktestResult, backtest_strategy
from sauce.research.profiles import (
    DEFAULT_STRATEGY_PROFILES,
    clear_strategy_profile_cache,
    get_strategy_profile,
    save_strategy_profiles,
)
from sauce.research.universe import HistoricalUniverseError, get_equity_universe_as_of
from sauce.research.walk_forward import WalkForwardResult, walk_forward_optimize

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "HistoricalUniverseError",
    "WalkForwardResult",
    "DEFAULT_STRATEGY_PROFILES",
    "backtest_strategy",
    "clear_strategy_profile_cache",
    "get_equity_universe_as_of",
    "get_strategy_profile",
    "save_strategy_profiles",
    "walk_forward_optimize",
]
