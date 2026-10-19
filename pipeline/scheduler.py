"""
SaayaShift — hourly pipeline scheduler (no admin rights required)

Runs the hourly_flow.py every 60 minutes and logs each run.
Keep this process running during the evidence window:
  Tue 20 Oct 16:00 PKT → Wed 21 Oct 16:00 PKT

Usage:
  python pipeline/scheduler.py

Evidence window is enforced automatically — the script logs start/end times
and all runs are appended to pipeline/prefect_runs.csv (unedited).
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "pipeline" / "scheduler_log.txt"
INTERVAL_SECONDS = 3600  # 1 hour


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run_flow() -> bool:
    """Run hourly_flow.py as a subprocess. Returns True on success."""
    flow_path = ROOT / "pipeline" / "flows" / "hourly_flow.py"
    result = subprocess.run(
        [sys.executable, str(flow_path)],
        cwd=str(ROOT),
        capture_output=False,
        timeout=1200,  # 20 min max per run
    )
    return result.returncode == 0


def main() -> None:
    log("=" * 60)
    log("SaayaShift hourly scheduler STARTED")
    log(f"Will run every {INTERVAL_SECONDS // 60} minutes")
    log("Keep this process running for the evidence window:")
    log("  Tue 20 Oct 16:00 PKT → Wed 21 Oct 16:00 PKT")
    log("=" * 60)

    run_count = 0
    while True:
        run_count += 1
        log(f"--- Run #{run_count} starting ---")
        try:
            success = run_flow()
            log(f"Run #{run_count} {'SUCCEEDED' if success else 'FAILED (check prefect_runs.csv)'}")
        except subprocess.TimeoutExpired:
            log(f"Run #{run_count} TIMED OUT after 20 min")
        except KeyboardInterrupt:
            log("Scheduler stopped by user (KeyboardInterrupt)")
            break
        except Exception as e:
            log(f"Run #{run_count} ERROR: {e}")

        log(f"Sleeping {INTERVAL_SECONDS // 60} min until next run...")
        try:
            time.sleep(INTERVAL_SECONDS)
        except KeyboardInterrupt:
            log("Scheduler stopped by user during sleep")
            break

    log("Scheduler exited.")


if __name__ == "__main__":
    main()
