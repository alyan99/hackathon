"""Tests for the Data Quality Agent — Silver and Gold layers."""

import pandas as pd
import pytest
from pathlib import Path


# ── Silver / Gate 1 tests ─────────────────────────────────────────────────────

def make_weather_df(n=100, bad_temp=False, bad_dup=False, missing_col=False):
    """Helper: make a minimal weather DataFrame."""
    import numpy as np
    times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    data = {
        "time": times,
        "apparent_temperature": np.random.uniform(25, 40, n).astype("float32"),
        "temperature_2m": np.random.uniform(25, 38, n).astype("float32"),
        "relative_humidity_2m": np.random.uniform(40, 90, n).astype("float32"),
        "windspeed_10m": np.random.uniform(0, 30, n).astype("float32"),
    }
    if bad_temp:
        data["apparent_temperature"][0] = 999.0   # out of range
    if bad_dup:
        data["time"] = pd.concat([pd.Series(times), pd.Series(times[:10])]).reset_index(drop=True)[:n]
        data["apparent_temperature"] = data["apparent_temperature"][:n]
    if missing_col:
        del data["temperature_2m"]
    return pd.DataFrame(data)


def test_schema_check_passes():
    from src.dq_agent.silver import gate1_schema_check
    df = make_weather_df()
    result = gate1_schema_check(df)
    assert result is not None


def test_schema_check_fails_on_missing_column():
    from src.dq_agent.silver import gate1_schema_check
    df = make_weather_df(missing_col=True)
    with pytest.raises(ValueError, match="Gate 1 FAILED"):
        gate1_schema_check(df)


def test_dedup_removes_duplicates():
    from src.dq_agent.silver import gate1_dedup
    # Use 1000 rows with only 1 duplicate — stays under 0.1% threshold
    times = pd.date_range("2025-01-01", periods=1000, freq="h", tz="UTC")
    data = {
        "time": list(times) + [times[0]],  # 1 duplicate in 1001 rows = 0.1%
        "apparent_temperature": [30.0] * 1001,
        "temperature_2m": [29.0] * 1001,
        "relative_humidity_2m": [60.0] * 1001,
        "windspeed_10m": [10.0] * 1001,
    }
    df = pd.DataFrame(data)
    result = gate1_dedup(df)
    assert len(result) == 1000


def test_range_check_nullifies_outliers():
    from src.dq_agent.silver import gate1_range_check
    import numpy as np
    times = pd.date_range("2025-01-01", periods=50, freq="h", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "apparent_temperature": ([999.0] + [30.0] * 49),  # one outlier
        "temperature_2m": [29.0] * 50,
        "relative_humidity_2m": [60.0] * 50,
        "windspeed_10m": [10.0] * 50,
    })
    df["apparent_temperature"] = df["apparent_temperature"].astype("float32")
    result = gate1_range_check(df)
    # The 999.0 should become NaN (or be imputed if within limit=3 gap)
    assert result["apparent_temperature"].max() < 999.0


# ── Gold / Gate 2 tests ───────────────────────────────────────────────────────

def test_drift_no_false_positive():
    """Stable data should not trigger drift."""
    from src.dq_agent.gold import check_drift
    import numpy as np
    n = 24 * 35   # 35 days of hourly data
    times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({
        "time": times,
        "apparent_temperature": np.random.normal(32, 2, n).astype("float32"),
    })
    assert check_drift(df) is True


def test_drift_detects_anomaly():
    """Extreme shift in recent 24h should trigger drift flag."""
    from src.dq_agent.gold import check_drift
    import numpy as np
    n = 24 * 35
    times = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    temps = np.random.normal(32, 2, n).astype("float32")
    temps[-24:] = 60.0   # sudden extreme spike
    df = pd.DataFrame({"time": times, "apparent_temperature": temps})
    assert check_drift(df) is False
