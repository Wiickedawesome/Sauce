"""Point-in-time research universe helpers.

Live trading can use a static universe, but research code should require an
explicit historical universe source for equities to avoid survivorship bias.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from sauce.core.config import get_settings


class HistoricalUniverseError(RuntimeError):
    """Raised when a point-in-time research universe is required but unavailable."""


@dataclass(frozen=True, slots=True)
class HistoricalUniverseSnapshot:
    effective_date: date
    symbols: tuple[str, ...]


def _universe_path(path: str | None = None) -> Path:
    settings = get_settings()
    return Path(path or settings.research_equity_universe_path)


def load_historical_equity_universe(path: str | None = None) -> list[HistoricalUniverseSnapshot]:
    universe_path = _universe_path(path)
    if not universe_path.exists():
        raise HistoricalUniverseError(
            f"Historical equity universe file not found: {universe_path}. "
            "Populate a point-in-time universe before running equity research, "
            "or explicitly opt into a static universe override."
        )

    raw_payload = json.loads(universe_path.read_text(encoding="utf-8"))
    raw_snapshots = raw_payload.get("snapshots", [])
    snapshots: list[HistoricalUniverseSnapshot] = []
    for raw_snapshot in raw_snapshots:
        if not isinstance(raw_snapshot, dict):
            continue
        effective_raw = raw_snapshot.get("effective_date")
        symbols_raw = raw_snapshot.get("symbols", [])
        if not isinstance(effective_raw, str) or not isinstance(symbols_raw, list):
            continue
        effective_date = datetime.strptime(effective_raw, "%Y-%m-%d").date()
        symbols = tuple(sorted({str(symbol).upper() for symbol in symbols_raw if str(symbol).strip()}))
        snapshots.append(HistoricalUniverseSnapshot(effective_date=effective_date, symbols=symbols))

    snapshots.sort(key=lambda snapshot: snapshot.effective_date)
    if not snapshots:
        raise HistoricalUniverseError(
            f"Historical equity universe file {universe_path} contains no usable snapshots"
        )
    return snapshots


def get_equity_universe_as_of(as_of: date, path: str | None = None) -> list[str]:
    snapshots = load_historical_equity_universe(path)
    eligible = [snapshot for snapshot in snapshots if snapshot.effective_date <= as_of]
    if not eligible:
        raise HistoricalUniverseError(
            f"No historical universe snapshot exists on or before {as_of.isoformat()}"
        )
    return list(eligible[-1].symbols)
