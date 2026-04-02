"""
tests/conftest.py — Project-wide test fixtures.

Guarantees every test uses an isolated temp DB, never the production sauce.db.
"""

import os

import pytest

import sauce.adapters.db as db_module
import sauce.adapters.market_data as market_data_module
from sauce.core.config import get_settings
from sauce.research.profiles import clear_strategy_profile_cache


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Redirect all DB writes to a per-test temp directory.

    This fires for EVERY test automatically (autouse=True).  It:
      1. Points DB_PATH to a temp file so no production DB is touched.
      2. Clears the engine cache so no leftover connections leak between tests.
      3. Clears the settings LRU cache so the new env vars are picked up.
      4. After each test, asserts that no SQLAlchemy engine was opened against
         any production DB path (IMP-07).
    """
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("DB_PATH", db_path)
    monkeypatch.setenv("SUPABASE_URL", "")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "")
    monkeypatch.setenv("SUPABASE_DB_URL", "")
    monkeypatch.setenv("STRATEGY_PROFILE_PATH", str(tmp_path / "strategy_profiles.test.json"))
    monkeypatch.setenv("RESEARCH_EQUITY_UNIVERSE_PATH", str(tmp_path / "equity_universe_history.test.json"))
    monkeypatch.setenv("ALPACA_API_KEY", "test-key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    db_module.cleanup_engines()
    market_data_module.clear_snapshot_state_cache()
    clear_strategy_profile_cache()
    get_settings.cache_clear()

    yield

    # IMP-07: Verify no engine was opened against any production DB path
    _prod_db = os.path.join(os.getcwd(), "data", "sauce.db")
    for engine_path in list(db_module._engines.keys()):
        assert os.path.abspath(engine_path) != os.path.abspath(_prod_db), (
            f"Test leaked a connection to production DB: {engine_path}"
        )

    db_module.cleanup_engines()
    market_data_module.clear_snapshot_state_cache()
    clear_strategy_profile_cache()
    get_settings.cache_clear()
