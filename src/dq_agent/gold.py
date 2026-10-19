"""
Data Quality Agent — Silver → Gold transformation (Gate 2)

Gate 2 checks:
  - Completeness:  < 2% missing in key columns
  - Consistency:   apparent_temperature in plausible Karachi range
  - Freshness:     latest record within 26 hours of now
  - Drift:         mean apparent_temperature within ±3σ of rolling window

On PASS:  writes data/gold/weather_gold.parquet
On FAIL:  alerts, keeps last-good Gold version (does not overwrite)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "karachi.yaml"
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR = ROOT / "data" / "gold"
DQ_LOG_PATH = SILVER_DIR / "dq_log.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _log_decision(action: str, reason: str, row_count: int, notes: str = "") -> None:
    from datetime import datetime, timezone
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not DQ_LOG_PATH.exists() or DQ_LOG_PATH.stat().st_size == 0
    with open(DQ_LOG_PATH, "a") as f:
        if write_header:
            f.write("timestamp_utc,action,reason,row_count,notes\n")
        ts = datetime.now(timezone.utc).isoformat()
        notes_clean = notes.replace(",", ";")
        f.write(f"{ts},{action},{reason},{row_count},{notes_clean}\n")
    log.info("[DQ Gold] %s | %s | rows=%d", action, reason, row_count)


KEY_COLUMNS = ["apparent_temperature", "temperature_2m", "relative_humidity_2m"]
MISSING_THRESHOLD = 0.02   # 2% max missing
FRESHNESS_HOURS = 26       # latest record must be within 26 hours — for LIVE forecast data
APPARENT_TEMP_RANGE = (-5.0, 60.0)


# ── Gate 2 checks ─────────────────────────────────────────────────────────────

def check_completeness(df: pd.DataFrame) -> bool:
    """Fail if missing rate in any key column exceeds 2%."""
    passed = True
    for col in KEY_COLUMNS:
        if col not in df.columns:
            _log_decision("alert", f"Gate 2 FAIL: key column {col} missing from Silver", 0)
            passed = False
            continue
        missing_rate = df[col].isna().mean()
        if missing_rate > MISSING_THRESHOLD:
            _log_decision(
                "alert",
                f"Gate 2 FAIL: {col} missing rate {missing_rate:.2%} > {MISSING_THRESHOLD:.0%}",
                int(df[col].isna().sum()),
            )
            passed = False
        else:
            _log_decision("pass", f"completeness OK: {col} missing rate={missing_rate:.4%}", len(df))
    return passed


def check_consistency(df: pd.DataFrame) -> bool:
    """Fail if apparent_temperature has values outside the plausible Karachi range."""
    col = "apparent_temperature"
    if col not in df.columns:
        return True
    lo, hi = APPARENT_TEMP_RANGE
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    if bad:
        _log_decision("alert", f"Gate 2 FAIL: {bad} apparent_temperature values outside [{lo},{hi}]", int(bad))
        return False
    _log_decision("pass", f"consistency OK: apparent_temperature in [{lo},{hi}]", len(df))
    return True


def check_freshness(df: pd.DataFrame, skip_for_historical: bool = False) -> bool:
    """
    Fail if the latest timestamp is older than FRESHNESS_HOURS from now.
    For the initial historical load, skip this check (historical data ends 2025-12-31
    and is expected to be older than 26h relative to the current time during the hackathon).
    Freshness is enforced on the live forecast data in the hourly pipeline.
    """
    if skip_for_historical:
        _log_decision("pass", "freshness check SKIPPED for historical data load (expected to be older than 26h)", 0)
        return True

    if "time" not in df.columns:
        _log_decision("alert", "Gate 2 FAIL: time column missing", 0)
        return False
    latest = pd.to_datetime(df["time"]).max()
    if latest.tzinfo is None:
        latest = latest.tz_localize("UTC")
    now = datetime.now(timezone.utc)
    age_h = (now - latest).total_seconds() / 3600
    if age_h > FRESHNESS_HOURS:
        _log_decision(
            "alert",
            f"Gate 2 FAIL: latest record is {age_h:.1f}h old (limit={FRESHNESS_HOURS}h)",
            1,
        )
        return False
    _log_decision("pass", f"freshness OK: latest record is {age_h:.1f}h old", len(df))
    return True


def check_drift(df: pd.DataFrame, window_days: int = 30, sigma: float = 3.0) -> bool:
    """
    Detect distribution drift: flag if recent mean apparent_temperature
    deviates from the rolling window mean by more than sigma standard deviations.
    """
    col = "apparent_temperature"
    if col not in df.columns or len(df) < 24 * window_days:
        _log_decision("pass", "drift check skipped: insufficient data for rolling window", len(df))
        return True

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    cutoff = df["time"].max() - timedelta(days=1)
    recent = df[df["time"] >= cutoff][col].dropna()
    historical = df[df["time"] < cutoff][col].dropna()

    if len(recent) == 0 or len(historical) == 0:
        return True

    hist_mean = historical.mean()
    hist_std = historical.std()
    recent_mean = recent.mean()

    z = abs(recent_mean - hist_mean) / max(hist_std, 0.01)
    if z > sigma:
        _log_decision(
            "alert",
            f"Gate 2 drift DETECTED: recent mean={recent_mean:.2f}°C vs hist mean={hist_mean:.2f}°C (z={z:.2f})",
            len(recent),
        )
        return False
    _log_decision(
        "pass",
        f"drift OK: z={z:.2f} < {sigma} (recent={recent_mean:.2f}°C hist={hist_mean:.2f}°C)",
        len(df),
    )
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def run_gold_pipeline(skip_freshness: bool = False) -> pd.DataFrame | None:
    """
    Run Gate 2 on Silver data.
    On pass: write gold/weather_gold.parquet and return DataFrame.
    On fail: keep last-good Gold version, return None.

    skip_freshness: set True when loading historical data (which is expected to be
    older than 26h). The live hourly pipeline passes skip_freshness=False.
    """
    silver_path = SILVER_DIR / "weather_silver.parquet"
    if not silver_path.exists():
        raise FileNotFoundError("Silver table not found. Run `make data` (silver step) first.")

    df = pd.read_parquet(silver_path)
    log.info("Loaded Silver: %d rows", len(df))

    checks = [
        check_completeness(df),
        check_consistency(df),
        check_freshness(df, skip_for_historical=skip_freshness),
        check_drift(df),
    ]

    all_passed = all(checks)

    if not all_passed:
        log.error("Gate 2 FAILED — keeping last-good Gold version (if exists).")
        _log_decision("alert", "Gate 2 FAILED — Gold not updated", 0)
        gold_path = GOLD_DIR / "weather_gold.parquet"
        if gold_path.exists():
            log.info("Serving last-good Gold: %s", gold_path)
            return pd.read_parquet(gold_path)
        return None

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold_path = GOLD_DIR / "weather_gold.parquet"

    df["_gold_written_utc"] = datetime.now(timezone.utc).isoformat()
    df["_source"] = "open_meteo"
    df.to_parquet(gold_path, index=False)

    _log_decision("pass", "Gate 2 PASSED — Gold layer written", len(df), notes=str(gold_path))
    log.info("Gold layer written: %s (%d rows)", gold_path, len(df))
    return df


def main() -> None:
    # On initial/historical load, skip freshness check
    result = run_gold_pipeline(skip_freshness=True)
    if result is None:
        log.error("Gold pipeline failed — check DQ log at %s", DQ_LOG_PATH)
    else:
        log.info("Gold pipeline complete. %d rows available.", len(result))


if __name__ == "__main__":
    main()
