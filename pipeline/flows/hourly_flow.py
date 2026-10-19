"""
SaayaShift — Prefect Hourly Pipeline Flow

8-step idempotent, retry-safe flow:
  1. Fetch      — new Open-Meteo data
  2. Bronze     — append to raw store (never overwrite)
  3. Gate 1     — Silver (fail = quarantine + alert + STOP)
  4. Gate 2     — Gold  (fail = alert + keep last good version)
  5. Retrain    — if drift detected in Gate 2
  6. Re-index   — RAG corpus with new docs/tables
  7. Briefs     — regenerate heat-safety briefs (placeholder)
  8. Dashboard  — refresh freshness metadata

Evidence window: Tue 20 Oct 16:00 PKT → Wed 21 Oct 16:00 PKT
Run log written to: pipeline/prefect_runs.csv (unedited)
"""

from __future__ import annotations

import csv
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Ensure project root is on sys.path so src.* imports work
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PREFECT_CSV = ROOT / "pipeline" / "prefect_runs.csv"
DRIFT_FLAG  = ROOT / "data" / "gold" / ".drift_detected"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── CSV run logger (unedited — do not post-process) ──────────────────────────

def _log_run(run_id: str, step: str, status: str, note: str = "") -> None:
    PREFECT_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not PREFECT_CSV.exists() or PREFECT_CSV.stat().st_size == 0
    with open(PREFECT_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["run_id", "step", "status", "timestamp_utc", "note"])
        writer.writerow([run_id, step, status, datetime.now(timezone.utc).isoformat(), note])


# ── Step implementations ──────────────────────────────────────────────────────

def step_fetch(run_id: str) -> bool:
    """Step 1: Fetch new Open-Meteo forecast data."""
    try:
        from src.ingest.open_meteo import load_config, fetch_forecast
        cfg = load_config()
        path = fetch_forecast(cfg, days=7)
        _log_run(run_id, "1_fetch", "success", f"saved={path.name}")
        return True
    except Exception as e:
        _log_run(run_id, "1_fetch", "failed", str(e))
        log.error("Step 1 (fetch) failed: %s", e)
        return False


def step_bronze(run_id: str) -> bool:
    """Step 2: Verify raw data landed in Bronze (append-only, never overwrite)."""
    raw_dir = ROOT / "data" / "raw" / "open_meteo"
    files = list(raw_dir.glob("*.json"))
    if not files:
        _log_run(run_id, "2_bronze", "failed", "no raw files found")
        return False
    _log_run(run_id, "2_bronze", "success", f"files={len(files)}")
    return True


def step_gate1_silver(run_id: str) -> tuple[bool, bool]:
    """
    Step 3: Gate 1 — Bronze → Silver.
    Returns (passed, drift_detected).
    On failure: quarantines rows and stops the flow.
    """
    try:
        from src.dq_agent.silver import run_silver_pipeline
        run_silver_pipeline()
        _log_run(run_id, "3_gate1_silver", "pass")
        return True, False
    except ValueError as e:
        _log_run(run_id, "3_gate1_silver", "fail_quarantine", str(e))
        log.error("Gate 1 FAILED — flow stopping: %s", e)
        return False, False
    except Exception as e:
        _log_run(run_id, "3_gate1_silver", "error", str(e))
        log.error("Gate 1 unexpected error: %s", traceback.format_exc())
        return False, False


def step_gate2_gold(run_id: str) -> tuple[bool, bool]:
    """
    Step 4: Gate 2 — Silver → Gold.
    Returns (passed, drift_detected).
    On failure: keeps last-good Gold, does not stop flow.
    """
    try:
        from src.dq_agent.gold import run_gold_pipeline
        result = run_gold_pipeline()
        if result is None:
            _log_run(run_id, "4_gate2_gold", "fail_kept_last_good")
            return False, False

        # Check for drift flag written by gold.py
        drift = DRIFT_FLAG.exists()
        _log_run(run_id, "4_gate2_gold", "pass", f"drift={drift}")
        return True, drift
    except Exception as e:
        _log_run(run_id, "4_gate2_gold", "error", str(e))
        log.error("Gate 2 error: %s", e)
        return False, False


def step_retrain(run_id: str, drift_detected: bool) -> None:
    """Step 5: Retrain model only if drift was detected in Gate 2."""
    if not drift_detected:
        _log_run(run_id, "5_retrain", "skipped", "no_drift")
        return
    try:
        from src.model.train import run_training
        run_training()
        _log_run(run_id, "5_retrain", "success")
    except Exception as e:
        _log_run(run_id, "5_retrain", "error", str(e))
        log.warning("Retrain failed (non-critical): %s", e)


def step_reindex_rag(run_id: str) -> None:
    """Step 6: Re-index RAG corpus with any new documents or Gold tables."""
    try:
        from src.rag.corpus import build_index
        build_index()
        _log_run(run_id, "6_reindex_rag", "success")
    except Exception as e:
        _log_run(run_id, "6_reindex_rag", "error", str(e))
        log.warning("RAG re-index failed (non-critical): %s", e)


def step_regenerate_briefs(run_id: str) -> None:
    """Step 7: Regenerate heat-safety briefs (summary of current conditions)."""
    try:
        brief_path = ROOT / "data" / "gold" / "current_brief.json"
        brief_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        gold_path = ROOT / "data" / "gold" / "weather_gold.parquet"
        if gold_path.exists():
            import pandas as pd
            df = pd.read_parquet(gold_path)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            recent = df.nlargest(24, "time")
            brief = {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "latest_apparent_temp_c": float(recent["apparent_temperature"].iloc[0])
                    if "apparent_temperature" in recent.columns else None,
                "24h_mean_c": float(recent["apparent_temperature"].mean())
                    if "apparent_temperature" in recent.columns else None,
                "data_as_of": str(df["time"].max()),
            }
            brief_path.write_text(json.dumps(brief, indent=2))
        _log_run(run_id, "7_briefs", "success")
    except Exception as e:
        _log_run(run_id, "7_briefs", "error", str(e))
        log.warning("Brief generation failed (non-critical): %s", e)


def step_refresh_dashboard(run_id: str) -> None:
    """Step 8: Update freshness metadata file used by the dashboard."""
    try:
        import json
        meta_path = ROOT / "data" / "gold" / "pipeline_meta.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "last_successful_run_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        _log_run(run_id, "8_dashboard", "success")
    except Exception as e:
        _log_run(run_id, "8_dashboard", "error", str(e))
        log.warning("Dashboard refresh failed (non-critical): %s", e)


# ── Main flow ─────────────────────────────────────────────────────────────────

def run_hourly_flow() -> bool:
    """
    Execute the full 8-step hourly flow.
    Steps 1-3 are hard failures (stop on fail).
    Steps 4-8 are soft failures (log and continue).
    Returns True if the full flow completed successfully.
    """
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    log.info("=== SaayaShift hourly flow starting: %s ===", run_id)
    _log_run(run_id, "flow_start", "running")

    # Step 1: Fetch
    if not step_fetch(run_id):
        _log_run(run_id, "flow_end", "failed", "step=1_fetch")
        log.error("Flow %s FAILED at step 1", run_id)
        return False

    # Step 2: Bronze
    if not step_bronze(run_id):
        _log_run(run_id, "flow_end", "failed", "step=2_bronze")
        return False

    # Step 3: Gate 1 — hard failure
    gate1_passed, _ = step_gate1_silver(run_id)
    if not gate1_passed:
        _log_run(run_id, "flow_end", "failed", "step=3_gate1_silver")
        log.error("Flow %s STOPPED at Gate 1", run_id)
        return False

    # Step 4: Gate 2 — soft failure (keep last-good Gold)
    _, drift_detected = step_gate2_gold(run_id)

    # Step 5–8: all soft failures
    step_retrain(run_id, drift_detected)
    step_reindex_rag(run_id)
    step_regenerate_briefs(run_id)
    step_refresh_dashboard(run_id)

    _log_run(run_id, "flow_end", "success")
    log.info("=== Flow %s completed successfully ===", run_id)
    return True


# ── Scheduler ─────────────────────────────────────────────────────────────────

def run_with_prefect() -> None:
    """Run using Prefect's scheduling — wraps flow with @flow decorator."""
    try:
        from prefect import flow, task
        from prefect.schedules import IntervalSchedule
        from datetime import timedelta

        @flow(name="saayashift-hourly", retries=1, retry_delay_seconds=120)
        def prefect_hourly_flow():
            return run_hourly_flow()

        prefect_hourly_flow.serve(
            name="saayashift-hourly-deployment",
            interval=timedelta(hours=1),
        )
    except ImportError:
        log.warning("Prefect not installed — running flow directly once")
        run_hourly_flow()


if __name__ == "__main__":
    # Run once directly (for testing); use run_with_prefect() for scheduled mode
    run_hourly_flow()
