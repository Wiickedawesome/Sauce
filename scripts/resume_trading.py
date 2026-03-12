#!/usr/bin/env python3
"""
scripts/resume_trading.py — Clear the DB-level trading pause flag.

Run locally:    python scripts/resume_trading.py
Run on VPS:     docker exec sauce python scripts/resume_trading.py

This writes a paused=False safety_check event to the DB, which is the only
way to unblock a pause set by the circuit breaker or 100%-veto anomaly.
After running, also verify with:  python scripts/diagnose.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sauce.core.safety import is_trading_paused, resume_trading
from sauce.core.config import get_settings

settings = get_settings()

print("Checking pause state...")

if settings.trading_pause:
    print(
        "\n[WARNING] TRADING_PAUSE=True is set in your .env / environment variables.\n"
        "  This is a config-level pause that overrides DB state.\n"
        "  Set TRADING_PAUSE=False in your .env file on the VPS, then restart the container.\n"
        "  DB resume alone will NOT unblock a config-level pause."
    )
else:
    print("  Config-level TRADING_PAUSE: False (OK)")

if is_trading_paused(loop_id="resume-script"):
    print("\nDB pause flag is ACTIVE. Writing resume record...")
    resume_trading(loop_id="manual-resume")
    print("Done. Trading RESUMED via DB flag.")
    print("\nVerify with: python scripts/diagnose.py")
    print("Then trigger a loop: python -m sauce.core.loop")
else:
    print("\nDB pause flag is NOT active — system should be running.")
    print("If the loop is still not trading, check:")
    print("  1. scripts/diagnose.py — look at recent supervisor abort reasons")
    print("  2. data/logs/daily_<date>.jsonl — check anomaly fields")
    print("  3. Cron health: scripts/health_check.sh")
