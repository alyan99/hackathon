"""
Model training and evaluation — SaayaShift apparent temperature forecaster

Train on 2023-2024 Gold data.
Evaluate on 2025 holdout vs frozen lag-24h baseline.
Report: overall MAE, slices by month/hour/grid-cell, prediction intervals.
Writes: docs/model_card.md, models/lgbm_model.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "weather_gold.parquet"
MODEL_DIR = ROOT / "models"
DOCS_PATH = ROOT / "docs"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TARGET_COL = "apparent_temperature"
TRAIN_END  = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END   = "2025-12-31"


def load_and_prepare() -> pd.DataFrame:
    from src.model.features import build_features, get_feature_columns
    df = pd.read_parquet(GOLD_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = build_features(df)
    return df


def split(df: pd.DataFrame):
    train = df[df["time"] <= TRAIN_END].copy()
    test  = df[(df["time"] >= TEST_START) & (df["time"] <= TEST_END)].copy()
    return train, test


def train_model(train: pd.DataFrame, feature_cols: list[str]):
    """Train a LightGBM regressor on the training set."""
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LightGBM not installed — falling back to Ridge regression")
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X = train[feature_cols].fillna(0)
        y = train[TARGET_COL]
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        model.fit(X, y)
        return model

    X = train[feature_cols].fillna(0)
    y = train[TARGET_COL]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X, y,
        eval_set=[(X, y)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    return model


def compute_prediction_intervals(model, X_test: pd.DataFrame, n_bootstrap: int = 50) -> tuple:
    """Bootstrap prediction intervals (simple lower/upper quantiles)."""
    preds = []
    rng = np.random.default_rng(42)
    X_arr = X_test.values
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(X_arr), size=len(X_arr))
        try:
            p = model.predict(X_arr[idx])
        except Exception:
            p = model.predict(X_arr)
        preds.append(p)
    preds = np.array(preds)
    lower = np.percentile(preds, 5, axis=0)
    upper = np.percentile(preds, 95, axis=0)
    return lower, upper


def write_model_card(results: dict) -> None:
    """Write a human-readable model card to docs/model_card.md."""
    DOCS_PATH.mkdir(parents=True, exist_ok=True)
    path = DOCS_PATH / "model_card.md"

    baseline_mae = results["baseline_mae"]
    model_mae    = results["model_mae"]
    improvement  = baseline_mae - model_mae
    beats        = model_mae < baseline_mae

    lines = [
        "# SaayaShift — Model Card",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Model",
        f"- **Algorithm:** {results['algorithm']}",
        f"- **Target:** `apparent_temperature` (°C) — hourly per Open-Meteo grid cell",
        f"- **Training period:** 2023-01-01 – 2024-12-31",
        f"- **Test period:** 2025-01-01 – 2025-12-31 (chronological holdout, never seen during training)",
        "",
        "## Evaluation Results",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Baseline MAE (lag-24h) | **{baseline_mae:.4f} °C** |",
        f"| Model MAE | **{model_mae:.4f} °C** |",
        f"| Improvement | {'✅ ' if beats else '❌ '}{improvement:.4f} °C |",
        f"| Beats baseline? | {'YES' if beats else 'NO — review required'} |",
        "",
        "## Error Slices",
        "",
        "### MAE by Month",
        "",
        "| Month | Baseline MAE | Model MAE |",
        "|-------|-------------|-----------|",
    ]

    for m in range(1, 13):
        bm = results.get("baseline_by_month", {}).get(str(m), float("nan"))
        mm = results.get("model_by_month", {}).get(str(m), float("nan"))
        lines.append(f"| {m:02d} | {bm:.4f} | {mm:.4f} |")

    lines += [
        "",
        "### MAE by Hour of Day",
        "",
        "| Hour | Baseline MAE | Model MAE |",
        "|------|-------------|-----------|",
    ]
    for h in range(0, 24):
        bh = results.get("baseline_by_hour", {}).get(str(h), float("nan"))
        mh = results.get("model_by_hour", {}).get(str(h), float("nan"))
        lines.append(f"| {h:02d}:00 | {bh:.4f} | {mh:.4f} |")

    lines += [
        "",
        "## Uncertainty",
        f"- Prediction intervals computed via {results.get('n_bootstrap', 50)}-sample bootstrap",
        f"- Interval coverage (90% PI): {results.get('interval_coverage', 'N/A')}",
        "",
        "## Limitations",
        "- Apparent temperature is a model-grid estimate, not a street-level observation.",
        "- The model is trained on one Karachi grid cell. Performance may differ for other locations.",
        "- Features are derived from the same Open-Meteo source; an independent ground-truth would strengthen evaluation.",
        "- Do not use model output to certify a shift or worksite as safe.",
        "",
        "## Feature Importance",
        *(f"- {k}: {v:.4f}" for k, v in list(results.get("feature_importance", {}).items())[:15]),
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Model card written → %s", path)


def run_training(eval_only: bool = False) -> dict:
    from src.model.features import get_feature_columns
    from src.model.baseline import compute_mae as baseline_mae_fn

    log.info("Loading and preparing features...")
    df = load_and_prepare()
    feature_cols = get_feature_columns(df)
    train_df, test_df = split(df)

    # Drop rows with NaN in target
    train_df = train_df.dropna(subset=[TARGET_COL])
    test_df  = test_df.dropna(subset=[TARGET_COL])

    # Baseline predictions
    test_df["baseline_pred"] = test_df[TARGET_COL].shift(24)
    baseline_mae = baseline_mae_fn(test_df[TARGET_COL], test_df["baseline_pred"])
    log.info("Baseline MAE (lag-24h) on 2025 holdout: %.4f °C", baseline_mae)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "lgbm_model.pkl"

    if eval_only and model_path.exists():
        log.info("Loading existing model for eval-only mode...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        log.info("Training model on %d rows...", len(train_df))
        X_train = train_df[feature_cols].fillna(0)
        y_train = train_df[TARGET_COL]
        model = train_model(train_df, feature_cols)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info("Model saved → %s", model_path)

    # Evaluate
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[TARGET_COL]
    y_pred = model.predict(X_test)
    model_mae = float(mean_absolute_error(y_test, y_pred))
    log.info("Model MAE on 2025 holdout: %.4f °C", model_mae)

    # Prediction intervals
    lower, upper = compute_prediction_intervals(model, X_test, n_bootstrap=50)
    coverage = float(((y_test.values >= lower) & (y_test.values <= upper)).mean())
    log.info("90%% PI coverage: %.4f", coverage)

    # Error slices
    test_df = test_df.copy()
    test_df["model_pred"] = y_pred
    test_df["month"] = test_df["time"].dt.month
    test_df["hour"]  = test_df["time"].dt.hour

    def mae_by(col):
        return test_df.groupby(col).apply(
            lambda g: float(mean_absolute_error(g[TARGET_COL], g["model_pred"]))
        ).to_dict()

    def baseline_by(col):
        return test_df.dropna(subset=["baseline_pred"]).groupby(col).apply(
            lambda g: float(mean_absolute_error(g[TARGET_COL], g["baseline_pred"]))
        ).to_dict()

    # Feature importance (LightGBM only)
    feat_imp = {}
    try:
        imp = model.feature_importances_
        feat_imp = dict(sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:20])
    except AttributeError:
        pass

    algorithm = type(model).__name__
    try:
        algorithm = f"LightGBMRegressor (n_estimators={model.n_estimators_})"
    except AttributeError:
        pass

    results = {
        "algorithm": algorithm,
        "baseline_mae": baseline_mae,
        "model_mae": model_mae,
        "beats_baseline": model_mae < baseline_mae,
        "improvement_c": baseline_mae - model_mae,
        "interval_coverage": f"{coverage:.2%}",
        "n_bootstrap": 50,
        "model_by_month": {str(k): v for k, v in mae_by("month").items()},
        "model_by_hour": {str(k): v for k, v in mae_by("hour").items()},
        "baseline_by_month": {str(k): v for k, v in baseline_by("month").items()},
        "baseline_by_hour": {str(k): v for k, v in baseline_by("hour").items()},
        "feature_importance": feat_imp,
    }

    # Print summary
    print("\n" + "=" * 60)
    print(f"MODEL EVALUATION — 2025 HOLDOUT")
    print(f"Baseline MAE:  {baseline_mae:.4f} °C")
    print(f"Model MAE:     {model_mae:.4f} °C")
    print(f"Improvement:   {baseline_mae - model_mae:.4f} °C")
    print(f"Beats baseline: {'YES' if model_mae < baseline_mae else 'NO'}")
    print(f"90% PI coverage: {coverage:.2%}")
    print("=" * 60 + "\n")

    write_model_card(results)

    # Save JSON results
    with open(DOCS_PATH / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true", help="Load saved model and evaluate only")
    args = parser.parse_args()
    run_training(eval_only=args.eval_only)


if __name__ == "__main__":
    main()
