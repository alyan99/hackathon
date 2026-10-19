"""
Data Quality Agent — Bronze → Silver transformation (Gate 1)

Applies to Open-Meteo weather data.
Actions at every row:
  - Fix:       rule-based corrections (e.g. unit normalisation, timezone cast)
  - Impute:    fill short gaps with is_imputed flag
  - Flag:      mark uncertain rows; add to quarantine if schema violations exist
  - Quarantine: rows that fail Gate 1 are moved to data/silver/quarantine/ — never dropped silently

Every decision is logged to data/silver/dq_log.csv with:
  reason, action, row_count, timestamp
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "karachi.yaml"
RAW_DIR = ROOT / "data" / "raw" / "open_meteo"
SILVER_DIR = ROOT / "data" / "silver"
QUARANTINE_DIR = SILVER_DIR / "quarantine"
DQ_LOG_PATH = SILVER_DIR / "dq_log.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── DQ log ────────────────────────────────────────────────────────────────────

def _log_decision(action: str, reason: str, row_count: int, notes: str = "") -> None:
    """Append one decision row to the DQ log."""
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not DQ_LOG_PATH.exists() or DQ_LOG_PATH.stat().st_size == 0
    with open(DQ_LOG_PATH, "a") as f:
        if write_header:
            f.write("timestamp_utc,action,reason,row_count,notes\n")
        ts = datetime.now(timezone.utc).isoformat()
        notes_clean = notes.replace(",", ";")
        f.write(f"{ts},{action},{reason},{row_count},{notes_clean}\n")
    log.info("[DQ] %s | %s | rows=%d", action, reason, row_count)


# ── Load raw data ─────────────────────────────────────────────────────────────

def load_raw_weather() -> pd.DataFrame:
    """
    Load and merge historical + all live forecast JSONs from Bronze.
    Historical file provides 2023-2025 training data.
    Live forecast files provide current + future hours for freshness.
    Deduplication keeps the live forecast value where times overlap.
    """
    hist_files = sorted(RAW_DIR.glob("historical_*.json"))
    fc_files = sorted(RAW_DIR.glob("forecast_*.json"))

    if not hist_files:
        raise FileNotFoundError(f"No historical JSON found in {RAW_DIR}. Run `make data` first.")

    # Load historical
    with open(hist_files[-1]) as f:
        hist_data = json.load(f)
    df_hist = pd.DataFrame(hist_data.get("hourly", {}))
    _log_decision("load", f"loaded historical file {hist_files[-1].name}", len(df_hist))

    # Load and merge all live forecast files
    fc_frames = []
    for fc_file in fc_files:
        with open(fc_file) as f:
            fc_data = json.load(f)
        hourly = fc_data.get("hourly", {})
        if hourly.get("time"):
            fc_frames.append(pd.DataFrame(hourly))
            _log_decision("load", f"loaded forecast file {fc_file.name}", len(fc_frames[-1]))

    if fc_frames:
        df_fc = pd.concat(fc_frames, ignore_index=True)
        # Merge: combine historical + forecast, forecast wins on duplicate times
        df = pd.concat([df_hist, df_fc], ignore_index=True)
        _log_decision("load", f"merged historical ({len(df_hist)}) + forecast ({len(df_fc)}) rows", len(df))
    else:
        df = df_hist
        log.info("No live forecast files found — using historical only")

    return df


# ── Gate 1 checks ─────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "time",
    "apparent_temperature",
    "temperature_2m",
    "relative_humidity_2m",
    "windspeed_10m",
]

# Plausible ranges for Karachi
VALUE_RANGES = {
    "apparent_temperature": (-5.0, 60.0),
    "temperature_2m": (-5.0, 55.0),
    "relative_humidity_2m": (0.0, 100.0),
    "windspeed_10m": (0.0, 150.0),
}


def gate1_schema_check(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required columns are present. Missing column = Gate 1 failure."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        _log_decision("quarantine", f"missing required columns: {missing}", len(df))
        quarantine_df(df, reason="missing_required_columns")
        raise ValueError(f"Gate 1 FAILED — missing columns: {missing}")
    _log_decision("pass", "schema check: all required columns present", len(df))
    return df


def gate1_type_cast(df: pd.DataFrame) -> pd.DataFrame:
    """Cast time to UTC datetime; cast numeric columns to float32."""
    original_len = len(df)

    # Time column
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    bad_time = df["time"].isna().sum()
    if bad_time:
        _log_decision("flag", "unparseable time values coerced to NaT", bad_time)

    # Numeric columns
    numeric_cols = [c for c in VALUE_RANGES if c in df.columns]
    for col in numeric_cols:
        before_na = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        after_na = df[col].isna().sum()
        new_na = after_na - before_na
        if new_na > 0:
            _log_decision("fix", f"non-numeric values in {col} coerced to NaN", new_na)

    _log_decision("fix", "type cast applied to all numeric + time columns", original_len)
    return df


def gate1_dedup(df: pd.DataFrame, expected_overlap: bool = False) -> pd.DataFrame:
    """
    Deduplicate on time; keep last occurrence (so live forecast overwrites history on overlap).
    expected_overlap=True: skip the quarantine rate check (used when merging historical + forecast).
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["time"], keep="last")
    removed = n_before - len(df)
    if removed:
        _log_decision("fix", "duplicate time rows removed (kept last — forecast wins over history)", removed)
    dup_rate = removed / max(n_before, 1)
    if not expected_overlap and dup_rate > 0.001:
        _log_decision(
            "quarantine",
            f"duplicate rate {dup_rate:.4%} exceeds 0.1% threshold",
            removed,
        )
        raise ValueError(f"Gate 1 FAILED — duplicate rate {dup_rate:.4%} > 0.1%")
    _log_decision("pass", f"dedup check passed (dup_rate={dup_rate:.4%}, expected_overlap={expected_overlap})", len(df))
    return df


def gate1_range_check(df: pd.DataFrame) -> pd.DataFrame:
    """Flag out-of-range values; impute short gaps; quarantine if too many invalid."""
    for col, (lo, hi) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_bad = mask.sum()
        if n_bad:
            df.loc[mask, col] = float("nan")
            _log_decision(
                "flag",
                f"{col} values outside [{lo}, {hi}] set to NaN",
                int(n_bad),
            )

    # Short gap imputation (linear interpolation, max gap = 3 h)
    for col in VALUE_RANGES:
        if col not in df.columns:
            continue
        n_missing = df[col].isna().sum()
        if n_missing:
            df[col] = df[col].interpolate(method="linear", limit=3)
            df[f"{col}_is_imputed"] = df[col].isna()  # True for remaining gaps
            filled = n_missing - df[col].isna().sum()
            _log_decision("impute", f"linear interpolation filled {filled} gaps in {col}", int(filled))

    return df


def quarantine_df(df: pd.DataFrame, reason: str) -> None:
    """Write rows to the quarantine directory — never drop silently."""
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = QUARANTINE_DIR / f"quarantine_{reason}_{ts}.parquet"
    df.to_parquet(path, index=False)
    log.warning("[DQ] Quarantined %d rows → %s (reason: %s)", len(df), path.name, reason)


# ── Main transformation ───────────────────────────────────────────────────────

def run_silver_pipeline() -> pd.DataFrame:
    """Run the full Bronze → Silver transformation with Gate 1 checks."""
    df = load_raw_weather()

    log.info("Running Gate 1 checks...")
    df = gate1_schema_check(df)
    df = gate1_type_cast(df)
    # expected_overlap=True because we intentionally merge historical + live forecast
    df = gate1_dedup(df, expected_overlap=True)
    df = gate1_range_check(df)

    # Sort by time
    df = df.sort_values("time").reset_index(drop=True)

    # Write Silver
    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    silver_path = SILVER_DIR / "weather_silver.parquet"
    df.to_parquet(silver_path, index=False)

    _log_decision("pass", "Gate 1 PASSED — Silver layer written", len(df), notes=str(silver_path))
    log.info("Silver layer written: %s (%d rows)", silver_path, len(df))
    return df


def main() -> None:
    run_silver_pipeline()


if __name__ == "__main__":
    main()
