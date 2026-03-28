from __future__ import annotations

import importlib.util
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_healthcheck_module(db_path: Path, interval: str = "30"):
    import os

    os.environ["DB_PATH"] = str(db_path)
    os.environ["LOOP_INTERVAL_MINUTES"] = interval

    script_path = REPO_ROOT / "scripts" / "docker_healthcheck.py"
    module_name = f"docker_healthcheck_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_loop_end(db_path: Path, timestamp: str, payload: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE audit_events (event_type TEXT, timestamp TEXT, payload TEXT, loop_id TEXT)")
    conn.execute(
        "INSERT INTO audit_events (event_type, timestamp, payload, loop_id) VALUES (?, ?, ?, ?)",
        ("loop_end", timestamp, payload, "loop-1"),
    )
    conn.commit()
    conn.close()


def test_docker_healthcheck_fails_on_failed_loop(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "health.db"
    _write_loop_end(
        db_path,
        "2026-03-27T12:00:00+00:00",
        '{"status": "failed", "error": "boom"}',
    )
    module = _load_healthcheck_module(db_path)
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    monkeypatch.setattr(module, "datetime", SimpleNamespace(now=lambda tz=None: module.datetime.fromisoformat("2026-03-27T12:05:00+00:00"), fromisoformat=module.datetime.fromisoformat))

    assert module.main() == 1
