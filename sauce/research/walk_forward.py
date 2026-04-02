"""Walk-forward optimization helpers for calibrated strategy profiles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from sauce.research.backtest import BacktestConfig, BacktestResult, backtest_strategy


@dataclass(frozen=True, slots=True)
class WalkForwardWindowResult:
    window_index: int
    profile: dict[str, Any]
    train_result: BacktestResult
    test_result: BacktestResult
    objective_score: float


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    best_profile: dict[str, Any]
    windows: list[WalkForwardWindowResult]
    mean_test_objective: float


def _expand_grid(parameter_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not parameter_grid:
        return [{}]

    keys = sorted(parameter_grid)
    values = [parameter_grid[key] for key in keys]
    return [dict(zip(keys, combination, strict=False)) for combination in product(*values)]


def _objective(result: BacktestResult) -> float:
    metrics = result.metrics
    sharpe_component = 0.0 if metrics.sharpe_ratio is None else metrics.sharpe_ratio
    return_component = 0.0 if metrics.total_return_pct is None else metrics.total_return_pct
    return return_component + (0.25 * sharpe_component) - metrics.max_drawdown_pct


def walk_forward_optimize(
    strategy_factory: Callable[[dict[str, Any]], Any],
    instrument: str,
    df: pd.DataFrame,
    *,
    is_crypto: bool,
    base_profile: dict[str, Any],
    parameter_grid: dict[str, list[Any]],
    train_bars: int,
    test_bars: int,
    step_bars: int,
    config: BacktestConfig | None = None,
) -> WalkForwardResult:
    """Evaluate candidate profiles using rolling train/test windows."""
    if train_bars <= 0 or test_bars <= 0 or step_bars <= 0:
        raise ValueError("train_bars, test_bars, and step_bars must all be positive")

    history = df.sort_index().copy()
    candidates = _expand_grid(parameter_grid)
    if not candidates:
        candidates = [{}]

    best_windows: list[WalkForwardWindowResult] = []
    mean_scores: list[float] = []
    max_start = len(history) - (train_bars + test_bars)
    if max_start < 0:
        raise ValueError("Not enough bars for the requested walk-forward windows")

    for window_index, start in enumerate(range(0, max_start + 1, step_bars), start=1):
        train_df = history.iloc[start : start + train_bars]
        test_df = history.iloc[start + train_bars : start + train_bars + test_bars]
        if train_df.empty or test_df.empty:
            continue

        candidate_results: list[WalkForwardWindowResult] = []
        for candidate in candidates:
            profile = dict(base_profile)
            profile.update(candidate)
            strategy = strategy_factory(profile)
            train_result = backtest_strategy(
                strategy,
                instrument,
                train_df,
                is_crypto=is_crypto,
                config=config,
            )
            test_result = backtest_strategy(
                strategy_factory(profile),
                instrument,
                test_df,
                is_crypto=is_crypto,
                config=config,
            )
            candidate_results.append(
                WalkForwardWindowResult(
                    window_index=window_index,
                    profile=profile,
                    train_result=train_result,
                    test_result=test_result,
                    objective_score=_objective(test_result),
                )
            )

        best_window = max(candidate_results, key=lambda result: result.objective_score)
        best_windows.append(best_window)
        mean_scores.append(best_window.objective_score)

    if not best_windows:
        raise ValueError("Walk-forward optimization produced no valid windows")

    overall_best = max(best_windows, key=lambda result: result.objective_score)
    mean_test_objective = sum(mean_scores) / len(mean_scores)
    return WalkForwardResult(
        best_profile=overall_best.profile,
        windows=best_windows,
        mean_test_objective=mean_test_objective,
    )
