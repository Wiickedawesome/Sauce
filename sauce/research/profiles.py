"""Strategy profile loading for calibrated scoring weights and thresholds."""

from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from sauce.core.config import get_settings

DEFAULT_STRATEGY_PROFILES: dict[str, dict[str, Any]] = {
    "crypto_momentum": {
        "trend_above_sma20_points": 15,
        "uptrend_structure_points": 10,
        "rsi_oversold": 40,
        "rsi_points": 25,
        "bb_proximity": 0.20,
        "bb_points": 20,
        "macd_dip_points": 15,
        "macd_momentum_points": 20,
        "volume_ratio_min": 0.5,
        "volume_points": 10,
        "momentum_rsi_min": 50,
        "momentum_rsi_max": 70,
        "momentum_points": 15,
        "stoch_confirm_points": 10,
        "vwap_confirm_points": 5,
        "base_threshold": 45,
        "regime_shift": {"bullish": -5, "neutral": 0, "bearish": 10},
    },
    "equity_momentum": {
        "trend_above_sma20_points": 20,
        "uptrend_structure_points": 15,
        "rsi_momentum_min": 50,
        "rsi_momentum_max": 65,
        "rsi_momentum_points": 20,
        "rsi_oversold": 30,
        "rsi_oversold_points": 25,
        "macd_momentum_points": 20,
        "volume_ratio_min": 1.5,
        "volume_points": 15,
        "base_threshold": 60,
        "regime_shift": {"bullish": -5, "neutral": 0, "bearish": 15},
    },
    "options_momentum": {
        "rsi_oversold_threshold": 35,
        "rsi_overbought_threshold": 65,
        "rsi_points": 25,
        "trend_above_sma20_points": 15,
        "macd_momentum_points": 20,
        "volume_ratio_min": 1.2,
        "volume_points": 15,
        "base_threshold": 40,
        "regime_shift": {"bullish": -5, "neutral": 0, "bearish": -5},
        "min_open_interest": 100,
        "max_spread_pct": 0.10,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _strategy_profile_path(path: str | None = None) -> str:
    if path:
        return path
    settings = get_settings()
    return settings.strategy_profile_path


@lru_cache(maxsize=8)
def load_strategy_profiles(path: str | None = None) -> dict[str, dict[str, Any]]:
    """Load calibrated strategy profiles and merge them with hard defaults."""
    resolved_path = _strategy_profile_path(path)
    merged = deepcopy(DEFAULT_STRATEGY_PROFILES)
    profile_file = Path(resolved_path)
    if not profile_file.exists():
        return merged

    raw_payload = json.loads(profile_file.read_text(encoding="utf-8"))
    raw_profiles = raw_payload.get("profiles", raw_payload)
    if not isinstance(raw_profiles, dict):
        return merged

    for strategy_name, overrides in raw_profiles.items():
        if strategy_name not in merged or not isinstance(overrides, dict):
            continue
        merged[strategy_name] = _deep_merge(merged[strategy_name], overrides)
    return merged


def clear_strategy_profile_cache() -> None:
    load_strategy_profiles.cache_clear()


def get_strategy_profile(
    strategy_name: str,
    default_profile: dict[str, Any],
    override_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the effective profile for a strategy."""
    resolved = deepcopy(load_strategy_profiles().get(strategy_name, default_profile))
    if override_profile:
        resolved = _deep_merge(resolved, override_profile)
    return resolved


def save_strategy_profiles(
    profiles: dict[str, dict[str, Any]],
    *,
    path: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Persist calibrated profiles to the configured research path."""
    resolved_path = Path(_strategy_profile_path(path))
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"profiles": profiles}
    if metadata:
        payload["metadata"] = metadata
    resolved_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    clear_strategy_profile_cache()
    return resolved_path
