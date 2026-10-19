"""
Frozen Baseline — Lag-24h predictor

For each target hour H, predict apparent_temperature = value at H-24h.

Training / development: 2023-01-01 through 2024-12-31
Holdout (test):         2025-01-01 through 2025-12-31

Metric: MAE in degrees Celsius

THIS BASELINE IS FROZEN. Do not retrain, re-run on the holdout
after seeing scores, or change the definition after evaluation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "weather_gold.parquet"
DOCS_PATH = ROOT / "docs" / "DATASET_DECLARATION.md"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END   = "2025-12-31"
TARGET_COL = "apparent_temperature"


def load_gold() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        raise FileNotFoundError("Gold table not found. Run `make data` first.")
    df = pd.read_parquet(GOLD_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def make_baseline_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift target column by 24 positions (hours) to create the lag-24h predictor.
    Only requires that rows are sorted by time with hourly frequency.
    """
    df = df.copy()
    df["baseline_pred"] = df[TARGET_COL].shift(24)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) using chronological split."""
    train = df[df["time"] <= TRAIN_END].copy()
    test  = df[(df["time"] >= TEST_START) & (df["time"] <= TEST_END)].copy()
    log.info("Train rows: %d | Test rows: %d", len(train), len(test))
    return train, test


def compute_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    mask = y_true.notna() & y_pred.notna()
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def evaluate_baseline() -> dict:
    """
    Evaluate the frozen lag-24h baseline on the 2025 holdout.
    Returns a dict of results; prints a summary.
    """
    df = load_gold()
    df = make_baseline_predictions(df)
    _, test = split_data(df)

    overall_mae = compute_mae(test[TARGET_COL], test["baseline_pred"])
    log.info("Baseline MAE (2025 holdout): %.4f °C", overall_mae)

    # Error slices
    test["month"] = test["time"].dt.month
    test["hour"]  = test["time"].dt.hour

    by_month = (
        test.groupby("month")
        .apply(lambda g: compute_mae(g[TARGET_COL], g["baseline_pred"]))
        .rename("mae_c")
    )
    by_hour = (
        test.groupby("hour")
        .apply(lambda g: compute_mae(g[TARGET_COL], g["baseline_pred"]))
        .rename("mae_c")
    )

    print("\n" + "=" * 50)
    print(f"FROZEN BASELINE — Lag-24h")
    print(f"Test period: {TEST_START} → {TEST_END}")
    print(f"Overall MAE: {overall_mae:.4f} °C")
    print("\nMAE by month:")
    print(by_month.to_string())
    print("\nMAE by hour-of-day:")
    print(by_hour.to_string())
    print("=" * 50 + "\n")

    results = {
        "baseline": "lag_24h",
        "test_start": TEST_START,
        "test_end": TEST_END,
        "overall_mae_c": overall_mae,
        "by_month": by_month.to_dict(),
        "by_hour": by_hour.to_dict(),
    }

    # Persist results
    out_path = ROOT / "docs" / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Baseline results saved → %s", out_path)

    return results


def main() -> None:
    evaluate_baseline()


if __name__ == "__main__":
    main()
