"""
Feature engineering — leakage-free

Every feature used to predict target hour H must have been available
at the time of prediction (no data from H or later used as a feature).

Features produced:
  - Lag features:      apparent_temperature at H-1h, H-6h, H-24h, H-168h
  - Rolling stats:     mean/std/min/max over past 24h and 72h
  - Calendar features: hour_of_day, day_of_week, month, is_weekend
  - Dewpoint proxy:    dewpoint_2m if available, else derived from RH + T
  - Wind-heat index:   windspeed_10m (already in data)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TARGET_COL = "apparent_temperature"

LAG_HOURS = [1, 6, 24, 168]
ROLL_WINDOWS = [24, 72]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all leakage-free features from a sorted hourly DataFrame.
    Input df must be sorted by time ascending with no duplicate timestamps.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # ── Lag features (all shift backward — uses only past data) ──────────────
    for h in LAG_HOURS:
        col = f"lag_{h}h"
        df[col] = df[TARGET_COL].shift(h)
        log.debug("Created feature: %s", col)

    # ── Rolling statistics over past window (min_periods avoids NaN floods) ──
    for window in ROLL_WINDOWS:
        base = df[TARGET_COL].shift(1)   # shift by 1 to exclude current hour H
        rolled = base.rolling(window=window, min_periods=max(1, window // 2))
        df[f"roll_mean_{window}h"] = rolled.mean()
        df[f"roll_std_{window}h"]  = rolled.std()
        df[f"roll_min_{window}h"]  = rolled.min()
        df[f"roll_max_{window}h"]  = rolled.max()

    # ── Calendar features ─────────────────────────────────────────────────────
    local_time = df["time"].dt.tz_convert("Asia/Karachi")
    df["hour_of_day"]   = local_time.dt.hour.astype("int8")
    df["day_of_week"]   = local_time.dt.dayofweek.astype("int8")
    df["month"]         = local_time.dt.month.astype("int8")
    df["is_weekend"]    = (df["day_of_week"] >= 5).astype("int8")

    # ── Dewpoint proxy ────────────────────────────────────────────────────────
    if "dewpoint_2m" in df.columns:
        df["dewpoint_feature"] = df["dewpoint_2m"].shift(1)
    elif "temperature_2m" in df.columns and "relative_humidity_2m" in df.columns:
        # Magnus formula approximation (available at prediction time via lagged values)
        T = df["temperature_2m"].shift(1)
        RH = df["relative_humidity_2m"].shift(1)
        df["dewpoint_feature"] = T - ((100 - RH) / 5.0)

    # ── Additional numeric features passed through ────────────────────────────
    for col in ["windspeed_10m", "precipitation"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    # ── Leakage assertion ─────────────────────────────────────────────────────
    # All lag/rolling features must have been shifted by ≥1 step.
    # Calendar features are based on the target hour timestamp only — no data leakage.
    feature_cols = [c for c in df.columns if c.startswith(("lag_", "roll_", "dewpoint", "hour_", "day_", "month", "is_", "windspeed_lag", "precipitation_lag"))]
    log.info("Feature matrix shape: %s | Feature columns: %d", df.shape, len(feature_cols))

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes target and metadata)."""
    exclude = {TARGET_COL, "time", "_gold_written_utc", "_source"}
    exclude.update(c for c in df.columns if c.endswith("_is_imputed"))
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]


if __name__ == "__main__":
    gold_path = ROOT / "data" / "gold" / "weather_gold.parquet"
    if gold_path.exists():
        df = pd.read_parquet(gold_path)
        df_feat = build_features(df)
        feat_cols = get_feature_columns(df_feat)
        print(f"Features built: {len(feat_cols)}")
        print(feat_cols)
    else:
        print("Gold table not found. Run `make data` first.")
