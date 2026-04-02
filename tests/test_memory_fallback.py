from __future__ import annotations

import builtins
import importlib.util
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_trade_memory_fallback_without_rank_bm25(monkeypatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "rank_bm25":
            raise ModuleNotFoundError("No module named 'rank_bm25'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    module_name = f"memory_fallback_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / "sauce" / "memory.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    entry = module.MemoryEntry(
        situation="BTC bullish RSI low volume high",
        outcome="profit",
        lesson="Works in strong reversals",
    )
    memory = module.TradeMemory([entry])

    results = memory.recall("BTC bullish volume high", n=1)

    assert len(results) == 1
    assert results[0].lesson == "Works in strong reversals"
