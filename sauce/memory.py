"""
memory.py — BM25-based trade memory for Sauce.

Stores (situation, outcome, lesson) tuples from post-trade reflections.
Recalls the most similar past situations using BM25 lexical matching —
no API calls required for retrieval.

Inspired by TradingAgents' FinancialSituationMemory but adapted for
Sauce's live trading context (positions, indicators, P&L).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

try:
    from rank_bm25 import BM25Plus
except ModuleNotFoundError:
    class BM25Plus:  # type: ignore[no-redef]
        """Minimal BM25-style fallback when rank_bm25 is unavailable."""

        def __init__(self, corpus: list[list[str]]) -> None:
            self._corpus = corpus
            self._doc_count = len(corpus)
            self._avgdl = (
                sum(len(document) for document in corpus) / self._doc_count if self._doc_count else 0.0
            )
            self._doc_freq: dict[str, int] = {}
            for document in corpus:
                for token in set(document):
                    self._doc_freq[token] = self._doc_freq.get(token, 0) + 1

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            if not self._corpus:
                return []

            unique_query = [token for token in dict.fromkeys(query_tokens) if token]
            scores: list[float] = []
            for document in self._corpus:
                doc_len = len(document)
                score = 0.0
                for token in unique_query:
                    term_freq = document.count(token)
                    if term_freq == 0:
                        continue
                    doc_freq = self._doc_freq.get(token, 0)
                    idf = math.log(1 + ((self._doc_count - doc_freq + 0.5) / (doc_freq + 0.5)))
                    norm = 1.2 * (1 - 0.75 + 0.75 * (doc_len / self._avgdl if self._avgdl else 1.0))
                    score += idf * ((term_freq * 2.2) / (term_freq + norm))
                scores.append(score)
            return scores

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """A single stored trade reflection."""

    situation: str  # market context at entry time
    outcome: str  # what happened (P&L, exit trigger, hold time)
    lesson: str  # LLM-generated insight from reflection


class TradeMemory:
    """BM25-based trade memory for recalling similar past situations.

    Memory entries are loaded from the DB on init. New entries are
    added via store() which updates both the in-memory index and the DB.
    The BM25 index is rebuilt whenever entries change.
    """

    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        self._entries: list[MemoryEntry] = list(entries) if entries else []
        self._index: BM25Plus | None = None
        if not hasattr(BM25Plus, "__module__") or BM25Plus.__module__ == __name__:
            logger.warning("rank_bm25 not installed; TradeMemory is using the built-in fallback scorer")
        if self._entries:
            self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from all stored entries."""
        if not self._entries:
            self._index = None
            return
        corpus = [entry.situation.lower().split() for entry in self._entries]
        self._index = BM25Plus(corpus)

    def store(self, entry: MemoryEntry) -> None:
        """Add a new memory entry and rebuild the index."""
        self._entries.append(entry)
        self._rebuild_index()

    def recall(self, current_situation: str, n: int = 3) -> list[MemoryEntry]:
        """Find the n most similar past situations using BM25.

        Returns up to n entries, sorted by relevance (best match first).
        Returns an empty list if no memories exist.
        """
        if not self._entries or self._index is None:
            return []

        query_tokens = current_situation.lower().split()
        scores = self._index.get_scores(query_tokens)

        # Pair scores with entries and sort descending
        scored = sorted(zip(scores, self._entries), key=lambda x: x[0], reverse=True)

        # Return top n with non-zero scores
        return [entry for score, entry in scored[:n] if score > 0]

    @property
    def size(self) -> int:
        """Number of stored memories."""
        return len(self._entries)


def build_situation_description(
    symbol: str,
    regime: str,
    score: int,
    threshold: int,
    rsi_14: float | None,
    macd_hist: float | None,
    bb_pct: float | None,
    volume_ratio: float | None,
    current_price: float,
    strategy_name: str,
) -> str:
    """Build a text description of the current market situation for BM25 matching."""
    parts = [
        f"symbol={symbol}",
        f"strategy={strategy_name}",
        f"regime={regime}",
        f"score={score}/{threshold}",
    ]
    if rsi_14 is not None:
        parts.append(f"RSI={rsi_14:.1f}")
    if macd_hist is not None:
        parts.append(f"MACD_hist={macd_hist:.4f}")
    if bb_pct is not None:
        parts.append(f"BB_pct={bb_pct:.2f}")
    if volume_ratio is not None:
        parts.append(f"volume_ratio={volume_ratio:.2f}")
    parts.append(f"price={current_price:.2f}")

    return " ".join(parts)


def build_outcome_description(
    symbol: str,
    entry_price: float,
    exit_price: float,
    exit_trigger: str,
    hold_hours: float,
    realized_pnl: float,
) -> str:
    """Build a text description of the trade outcome for storage."""
    pnl_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
    return (
        f"symbol={symbol} entry={entry_price:.4f} exit={exit_price:.4f} "
        f"pnl={realized_pnl:+.2f} ({pnl_pct:+.2f}%) "
        f"trigger={exit_trigger} held={hold_hours:.1f}h"
    )
