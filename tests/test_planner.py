"""Tests for the shift planner — determinism and heat ranking."""

import pandas as pd
import pytest


def make_forecast_df():
    """24 hours of hourly apparent temperature data."""
    import numpy as np
    times = pd.date_range("2025-06-15 00:00", periods=24, freq="h", tz="UTC")
    # Make early morning cool, midday hot
    temps = [28, 27, 27, 26, 26, 27, 29, 31, 33, 36, 38, 40,
             41, 42, 41, 40, 38, 36, 34, 32, 31, 30, 29, 28]
    return pd.DataFrame({"time": times, "apparent_temperature": temps})


def test_planner_deterministic():
    """Same input → same ranking every time."""
    from src.app.planner import generate_windows, rank_windows
    df = make_forecast_df()
    windows1 = rank_windows(generate_windows(df, duration_h=4, op_start_h=6, op_end_h=20))
    windows2 = rank_windows(generate_windows(df, duration_h=4, op_start_h=6, op_end_h=20))
    scores1 = [w.score for w in windows1]
    scores2 = [w.score for w in windows2]
    assert scores1 == scores2


def test_planner_lower_heat_ranks_higher():
    """Window with lower mean temp must get rank 1."""
    from src.app.planner import generate_windows, rank_windows
    df = make_forecast_df()
    windows = rank_windows(generate_windows(df, duration_h=4, op_start_h=6, op_end_h=20))
    assert len(windows) > 1
    # Rank 1 should have lower or equal score than rank 2
    assert windows[0].score <= windows[1].score


def test_planner_reason_not_empty():
    """Every window must have a non-empty reason string."""
    from src.app.planner import generate_windows, rank_windows
    df = make_forecast_df()
    windows = rank_windows(generate_windows(df, duration_h=4, op_start_h=6, op_end_h=20))
    for w in windows:
        assert w.reason.strip() != ""


def test_planner_windows_within_operating_hours():
    """All windows must start and end within operating hours."""
    from src.app.planner import generate_windows
    df = make_forecast_df()
    windows = generate_windows(df, duration_h=4, op_start_h=6, op_end_h=20)
    for w in windows:
        start_h = w.start_hour.astimezone(
            __import__("zoneinfo").ZoneInfo("Asia/Karachi")
        ).hour
        assert 6 <= start_h <= 16   # 4h window ending by 20:00
