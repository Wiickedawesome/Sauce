#!/usr/bin/env python3
"""Walk-forward calibrate a Sauce strategy profile and write it to disk."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from sauce.adapters.market_data import get_history
from sauce.core.config import get_settings
from sauce.research.profiles import (
    DEFAULT_STRATEGY_PROFILES,
    save_strategy_profiles,
)
from sauce.research.universe import HistoricalUniverseError, get_equity_universe_as_of
from sauce.research.walk_forward import walk_forward_optimize
from sauce.strategies.crypto_momentum import CryptoMomentumReversion
from sauce.strategies.equity_momentum import EquityMomentum

DEFAULT_PARAMETER_GRIDS: dict[str, dict[str, list[Any]]] = {
    "crypto_momentum": {
        "base_threshold": [40, 45, 50],
        "rsi_oversold": [35, 40, 45],
        "bb_proximity": [0.15, 0.20, 0.25],
        "volume_ratio_min": [0.5, 0.75, 1.0],
    },
    "equity_momentum": {
        "base_threshold": [55, 60, 65],
        "rsi_momentum_max": [60, 65, 70],
        "rsi_oversold": [25, 30, 35],
        "volume_ratio_min": [1.0, 1.5, 2.0],
    },
}


def _factory(strategy_name: str):
    if strategy_name == "crypto_momentum":
        return lambda profile: CryptoMomentumReversion(profile_override=profile)
    return lambda profile: EquityMomentum(profile_override=profile)


def _symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        return [symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()]

    if args.strategy == "crypto_momentum":
        return get_settings().crypto_universe[:3]

    if args.allow_static_universe:
        return get_settings().equity_universe

    try:
        return get_equity_universe_as_of(datetime.now(UTC).date())
    except HistoricalUniverseError as exc:
        raise SystemExit(str(exc)) from exc


def _aggregate_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    if not profiles:
        raise ValueError("No profiles available to aggregate")

    aggregated: dict[str, Any] = {}
    for key, sample_value in profiles[0].items():
        if isinstance(sample_value, dict):
            aggregated[key] = {
                nested_key: int(round(median(float(profile[key][nested_key]) for profile in profiles)))
                for nested_key in sample_value
            }
        elif isinstance(sample_value, int):
            aggregated[key] = int(round(median(float(profile[key]) for profile in profiles)))
        elif isinstance(sample_value, float):
            aggregated[key] = float(median(float(profile[key]) for profile in profiles))
        else:
            aggregated[key] = sample_value
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", choices=["crypto_momentum", "equity_momentum"])
    parser.add_argument("--symbols", default="", help="Comma-separated instruments to calibrate")
    parser.add_argument("--bars", type=int, default=600)
    parser.add_argument("--train-bars", type=int, default=240)
    parser.add_argument("--test-bars", type=int, default=120)
    parser.add_argument("--step-bars", type=int, default=120)
    parser.add_argument("--allow-static-universe", action="store_true")
    args = parser.parse_args()

    is_crypto = args.strategy == "crypto_momentum"
    timeframe = "1Hour" if is_crypto else "1Day"
    strategy_factory = _factory(args.strategy)
    base_profile = DEFAULT_STRATEGY_PROFILES[args.strategy]

    symbol_profiles: list[dict[str, Any]] = []
    symbol_scores: list[float] = []
    for symbol in _symbols(args):
        history = get_history(symbol, timeframe=timeframe, bars=args.bars)
        if history is None or history.empty:
            continue
        result = walk_forward_optimize(
            strategy_factory,
            symbol,
            history,
            is_crypto=is_crypto,
            base_profile=base_profile,
            parameter_grid=DEFAULT_PARAMETER_GRIDS[args.strategy],
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.step_bars,
        )
        symbol_profiles.append(result.best_profile)
        symbol_scores.append(result.mean_test_objective)

    if not symbol_profiles:
        raise SystemExit("Calibration produced no usable profiles")

    aggregated_profile = _aggregate_profiles(symbol_profiles)
    output_path = save_strategy_profiles(
        {args.strategy: aggregated_profile},
        metadata={
            "generated_at": datetime.now(UTC).isoformat(),
            "strategy": args.strategy,
            "symbols": _symbols(args),
            "mean_test_objective": sum(symbol_scores) / len(symbol_scores),
        },
    )
    print(f"Wrote calibrated profile to {output_path}")


if __name__ == "__main__":
    main()
