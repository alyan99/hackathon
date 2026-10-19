"""
Open-Meteo ingest — Bronze layer
Fetches historical forecast data (2023-2025) and the live forecast
for the Karachi grid cell. Saves raw JSON unchanged and records
provenance in docs/manifest.csv.

Attribution: Data from Open-Meteo.com (CC BY 4.0)
Docs: https://open-meteo.com/en/docs/historical-forecast-api
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "karachi.yaml"
RAW_DIR = ROOT / "data" / "raw" / "open_meteo"
MANIFEST_PATH = ROOT / "docs" / "manifest.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _append_manifest(row: dict) -> None:
    """Append a single row to manifest.csv."""
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Write header if file is empty
    write_header = not MANIFEST_PATH.exists() or MANIFEST_PATH.stat().st_size == 0
    with open(MANIFEST_PATH, "a") as f:
        if write_header:
            f.write("source,url,fetch_time_utc,sha256,licence,osm_base,notes\n")
        osm_base = row.get("osm_base", "")
        notes = row.get("notes", "").replace(",", ";")
        f.write(
            f"{row['source']},{row['url']},{row['fetch_time_utc']},"
            f"{row['sha256']},{row['licence']},{osm_base},{notes}\n"
        )


def fetch_historical(cfg: dict) -> Path:
    """
    Fetch historical hourly forecast data from Open-Meteo Historical Forecast API.
    Covers 2023-01-01 through 2025-12-31 (~26,304 rows).
    Returns path to saved raw JSON file.
    """
    lat = cfg["coordinates"]["latitude"]
    lon = cfg["coordinates"]["longitude"]
    start = cfg["open_meteo"]["historical_start"]
    end = cfg["open_meteo"]["historical_end"]
    variables = ",".join(cfg["open_meteo"]["variables"])

    url = (
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={variables}"
        f"&start_date={start}&end_date={end}"
        "&timezone=Asia%2FKarachi"
    )

    log.info("Fetching Open-Meteo historical data: %s → %s", start, end)
    log.info("URL: %s", url)

    fetch_time = datetime.now(timezone.utc).isoformat()

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    raw_bytes = resp.content

    sha = _sha256(raw_bytes)
    out_path = RAW_DIR / f"historical_{start}_{end}.json"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(raw_bytes)

    log.info("Saved historical data → %s (%d bytes, sha256=%s)", out_path.name, len(raw_bytes), sha[:16])

    _append_manifest({
        "source": "open_meteo_historical",
        "url": url,
        "fetch_time_utc": fetch_time,
        "sha256": sha,
        "licence": "CC BY 4.0",
        "notes": f"historical_forecast_api start={start} end={end} model={cfg['open_meteo']['model']}",
    })

    return out_path


def fetch_forecast(cfg: dict, days: int = 7) -> Path:
    """
    Fetch the current hourly forecast from Open-Meteo Forecast API.
    Records selected model, request URL, and fetch time.
    Does NOT imply a new model run occurs on every hourly request.
    """
    lat = cfg["coordinates"]["latitude"]
    lon = cfg["coordinates"]["longitude"]
    variables = ",".join(cfg["open_meteo"]["variables"])
    model = cfg["open_meteo"]["model"]

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={variables}"
        f"&forecast_days={days}"
        f"&models={model}"
        "&timezone=Asia%2FKarachi"
    )

    log.info("Fetching Open-Meteo live forecast (%d days)", days)
    log.info("URL: %s", url)

    fetch_time = datetime.now(timezone.utc).isoformat()

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    raw_bytes = resp.content

    # Extract model actually used from response
    data = json.loads(raw_bytes)
    model_used = data.get("model", model)

    sha = _sha256(raw_bytes)
    timestamp = fetch_time.replace(":", "-").replace("+", "Z")[:19]
    out_path = RAW_DIR / f"forecast_{timestamp}.json"
    out_path.write_bytes(raw_bytes)

    log.info("Saved forecast → %s (%d bytes, model=%s)", out_path.name, len(raw_bytes), model_used)

    _append_manifest({
        "source": "open_meteo_forecast",
        "url": url,
        "fetch_time_utc": fetch_time,
        "sha256": sha,
        "licence": "CC BY 4.0",
        "notes": f"forecast_api days={days} model_used={model_used}",
    })

    return out_path


def count_rows(json_path: Path) -> int:
    """Return number of hourly records in a raw Open-Meteo JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    times = data.get("hourly", {}).get("time", [])
    return len(times)


def main() -> None:
    cfg = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    hist_path = fetch_historical(cfg)
    rows = count_rows(hist_path)
    log.info("Historical rows: %d (expected ≥26,000)", rows)
    if rows < 26_000:
        log.warning("Row count %d is below the expected 26,000 — check API response", rows)

    # Small pause to respect API rate limits
    time.sleep(2)

    fetch_forecast(cfg)
    log.info("Open-Meteo ingest complete.")


if __name__ == "__main__":
    main()
