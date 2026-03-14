"""
tests/conftest.py — Project-wide test fixtures.

Guarantees every test uses an isolated temp DB, never the production sauce.db.
"""

import pytest

import sauce.adapters.db as db_module
import sauce.memory.db as memory_db_module
from sauce.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect all DB writes to a per-test temp directory.

    This fires for EVERY test automatically (autouse=True).  It:
      1. Points DB_PATH to a temp file so ``_default_db_path()`` resolves there.
      2. Clears the engine cache so no leftover connections leak between tests.
      3. Clears the settings LRU cache so the new env var is picked up.
      4. Also clears the memory DB engine cache (sauce.memory.db).
    """
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    db_module._engines = {}
    memory_db_module._engines = {}
    get_settings.cache_clear()

    yield

    db_module._engines = {}
    memory_db_module._engines = {}
    get_settings.cache_clear()
