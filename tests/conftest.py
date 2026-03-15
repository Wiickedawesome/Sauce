"""
tests/conftest.py — Project-wide test fixtures.

Guarantees every test uses an isolated temp DB, never the production sauce.db.
"""

import os

import pytest

import sauce.adapters.db as db_module
import sauce.memory.db as memory_db_module
from sauce.core.config import get_settings


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect all DB writes to a per-test temp directory.

    This fires for EVERY test automatically (autouse=True).  It:
      1. Points DB_PATH, STRATEGIC_MEMORY_DB_PATH, and SESSION_MEMORY_DB_PATH
         to temp files so no production DB is touched.
      2. Clears the engine cache so no leftover connections leak between tests.
      3. Clears the settings LRU cache so the new env vars are picked up.
      4. Also clears the memory DB engine cache (sauce.memory.db).
      5. After each test, asserts that no SQLAlchemy engine was opened against
         any production DB path (IMP-07).
    """
    db_path = str(tmp_path / "test.db")
    strategic_db_path = str(tmp_path / "strategic_memory.db")
    session_db_path = str(tmp_path / "session_memory.db")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("STRATEGIC_MEMORY_DB_PATH", strategic_db_path)
    monkeypatch.setenv("SESSION_MEMORY_DB_PATH", session_db_path)
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    db_module._engines = {}
    memory_db_module._engines = {}
    get_settings.cache_clear()

    yield

    # IMP-07: Verify no engine was opened against any production DB path
    _prod_db = os.path.join(os.getcwd(), "data", "sauce.db")
    _prod_strategic = os.path.join(os.getcwd(), "data", "strategic_memory.db")
    _prod_session = os.path.join(os.getcwd(), "data", "session_memory.db")
    for engine_path in list(db_module._engines.keys()):
        assert os.path.abspath(engine_path) != os.path.abspath(_prod_db), (
            f"Test leaked a connection to production DB: {engine_path}"
        )
    for engine_path in list(memory_db_module._engines.keys()):
        abs_path = os.path.abspath(engine_path)
        assert abs_path != os.path.abspath(_prod_strategic), (
            f"Test leaked a connection to production strategic memory DB: {engine_path}"
        )
        assert abs_path != os.path.abspath(_prod_session), (
            f"Test leaked a connection to production session memory DB: {engine_path}"
        )

    db_module._engines = {}
    memory_db_module._engines = {}
    get_settings.cache_clear()
