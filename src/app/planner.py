"""
Shift Planner — SaayaShift

Given a work zone, planned duration, and operating constraints,
generates candidate windows and ranks them by forecast heat exposure.

The ranking is:
  - Deterministic: same input always produces the same output
  - Explainable: every score is broken down by contributing hours
  - Transparent: all thresholds come from cited guidance (WHO/ILO/OSHA)

SAFETY NOTE: This planner does not certify any shift as medically safe.
It does not diagnose heat illness or replace official occupational-health guidance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "weather_gold.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Heat thresholds — from cited WHO/ILO/OSHA guidance ───────────────────────
# Source: WHO Heat and Health Fact Sheet (2023); OSHA Heat Illness Prevention
# These are reference values only; they do not certify a shift as safe.
HEAT_LEVELS = {
    "normal":   (float("-inf"), 32.0),
    "caution":  (32.0, 38.0),
    "high":     (38.0, 43.0),
    "extreme":  (43.0, float("inf")),
}

GUIDANCE = {
    "normal":  "Standard hydration and rest breaks sufficient (WHO/ILO guidance).",
    "caution": "Increase water intake; schedule rest breaks every 45–60 min (ILO Technical Brief 2018).",
    "high":    "Mandatory rest breaks every 30 min; buddy system; monitor for symptoms (OSHA Heat Illness Prevention).",
    "extreme": "Avoid non-essential outdoor work if possible; continuous monitoring required (WHO 2023).",
}


def classify_heat(apparent_temp_c: float) -> str:
    for level, (lo, hi) in HEAT_LEVELS.items():
        if lo <= apparent_temp_c < hi:
            return level
    return "extreme"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ShiftWindow:
    start_hour: datetime
    end_hour: datetime
    duration_h: int
    hours: list[datetime]
    temps: list[float]
    mean_temp_c: float
    min_temp_c: float
    max_temp_c: float
    heat_level: str
    guidance: str
    score: float          # lower = better (lower mean apparent temp)
    rank: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "start": self.start_hour.strftime("%Y-%m-%d %H:%M"),
            "end": self.end_hour.strftime("%Y-%m-%d %H:%M"),
            "duration_h": self.duration_h,
            "mean_apparent_temp_c": round(self.mean_temp_c, 1),
            "min_apparent_temp_c": round(self.min_temp_c, 1),
            "max_apparent_temp_c": round(self.max_temp_c, 1),
            "heat_level": self.heat_level,
            "guidance": self.guidance,
            "score": round(self.score, 2),
            "reason": self.reason,
        }


# ── Planner ───────────────────────────────────────────────────────────────────

def load_forecast(forecast_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load apparent temperature forecast from the Gold table.
    If forecast_date is None, uses the most recent 7 days available.
    """
    if not GOLD_PATH.exists():
        raise FileNotFoundError("Gold table not found. Run `make data` first.")

    df = pd.read_parquet(GOLD_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")

    if forecast_date:
        target = pd.Timestamp(forecast_date, tz="UTC")
        df = df[df["time"].dt.date == target.date()]
    else:
        latest = df["time"].max()
        df = df[df["time"] >= latest - pd.Timedelta(days=7)]

    return df[["time", "apparent_temperature"]].dropna()


def generate_windows(
    df: pd.DataFrame,
    duration_h: int,
    op_start_h: int = 6,
    op_end_h: int = 20,
) -> list[ShiftWindow]:
    """
    Enumerate all candidate shift windows within operating hours.
    Windows slide by 1-hour increments.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"]).dt.tz_convert("Asia/Karachi")
    df = df.set_index("time").sort_index()

    windows = []
    times = df.index.tolist()

    for i, start in enumerate(times):
        # Only consider windows that start within operating hours
        if start.hour < op_start_h or start.hour > op_end_h - duration_h:
            continue

        end = start + timedelta(hours=duration_h - 1)
        window_times = [t for t in times if start <= t <= end]

        if len(window_times) < duration_h:
            continue   # incomplete window

        temps = [float(df.loc[t, "apparent_temperature"]) for t in window_times]
        mean_t = sum(temps) / len(temps)
        min_t  = min(temps)
        max_t  = max(temps)
        level  = classify_heat(mean_t)

        # Score = mean apparent temperature (deterministic; lower = better)
        score = mean_t

        reason = (
            f"Ranked by mean apparent temperature ({mean_t:.1f}°C). "
            f"Hours used: {start.strftime('%H:%M')}–{end.strftime('%H:%M')}. "
            f"Heat level: {level.upper()} (range {min_t:.1f}–{max_t:.1f}°C). "
            f"{GUIDANCE[level]}"
        )

        windows.append(ShiftWindow(
            start_hour=start,
            end_hour=end,
            duration_h=duration_h,
            hours=window_times,
            temps=temps,
            mean_temp_c=mean_t,
            min_temp_c=min_t,
            max_temp_c=max_t,
            heat_level=level,
            guidance=GUIDANCE[level],
            score=score,
            reason=reason,
        ))

    return windows


def rank_windows(windows: list[ShiftWindow]) -> list[ShiftWindow]:
    """Sort windows by score (ascending = lower heat = better rank)."""
    sorted_windows = sorted(windows, key=lambda w: w.score)
    for i, w in enumerate(sorted_windows):
        w.rank = i + 1
    return sorted_windows


def plan_shift(
    duration_h: int = 4,
    forecast_date: Optional[str] = None,
    op_start_h: int = 6,
    op_end_h: int = 20,
    top_n: int = 5,
) -> list[dict]:
    """
    Main entry point: return top_n ranked shift windows as a list of dicts.

    Args:
        duration_h:     Length of shift in hours
        forecast_date:  Date string YYYY-MM-DD or None for most recent data
        op_start_h:     Earliest start hour (local time, 24h)
        op_end_h:       Latest end hour (local time, 24h)
        top_n:          Number of ranked windows to return
    """
    df = load_forecast(forecast_date)
    if df.empty:
        return [{"error": "No forecast data available. Run `make data` first."}]

    windows = generate_windows(df, duration_h, op_start_h, op_end_h)
    if not windows:
        return [{"error": f"No complete {duration_h}-hour windows found within operating hours {op_start_h}:00–{op_end_h}:00."}]

    ranked = rank_windows(windows)
    return [w.to_dict() for w in ranked[:top_n]]


if __name__ == "__main__":
    results = plan_shift(duration_h=4, top_n=5)
    for r in results:
        print(r)
