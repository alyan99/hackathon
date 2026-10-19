# SaayaShift — Dataset Declaration

**Committed:** 2026-10-19 (Mon 19 October 2026, before 12:00 PM PKT)
**Project:** SaayaShift — Heat-Aware Shift Planner

---

## 1. Forecasting Target

**Target variable:** `apparent_temperature` (°C) — hourly, per Open-Meteo grid cell.

Apparent temperature is the "feels-like" temperature computed by Open-Meteo from
air temperature, relative humidity, and wind speed. It is a model-grid estimate,
not a street-level measurement.

---

## 2. Frozen Baseline

> **For each target hour H, predict apparent temperature as the value for the same
> local hour on the previous day (H − 24 hours).**

- Training / development period: **2023-01-01 through 2024-12-31** (2 full years)
- Holdout / test period: **2025-01-01 through 2025-12-31** (chronological; never used during development)
- Metric: **MAE in degrees Celsius**

This baseline is declared and frozen **before** any model evaluation.
It will not be changed after scores are seen.
The trained model must beat this MAE on the 2025 holdout to claim improvement.

---

## 3. Data Sources

### Source 1 — Open-Meteo (weather data)

| Field | Value |
|-------|-------|
| Publisher | Open-Meteo (open-meteo.com) |
| APIs used | Historical Forecast API · Forecast API |
| Licence | CC BY 4.0 (non-commercial free tier) |
| Attribution | "Data from Open-Meteo.com" required wherever data is displayed |
| Historical period | 2023-01-01 through 2025-12-31 |
| Expected row count | ~26,304 hourly records per distinct grid cell |
| Grid cell | Karachi: 24.8607°N, 67.0011°E (deduplicated; same coordinates map to one cell) |
| Variables | apparent_temperature, temperature_2m, relative_humidity_2m, windspeed_10m, precipitation, dewpoint_2m |
| Fetch rule | Raw JSON responses saved unchanged; request URL and fetch time recorded in manifest.csv |
| Model | best_match (Open-Meteo selects best available NWP model for location) |
| Note | Historical Forecast and Forecast API share the same publisher; they count as **one source** |

Docs: [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) ·
[Forecast API](https://open-meteo.com/en/docs) ·
[Licence](https://open-meteo.com/en/licence) ·
[Terms](https://open-meteo.com/en/terms) ·
[Model updates](https://open-meteo.com/en/docs/model-updates)

---

### Source 2 — OpenStreetMap via Overpass API (geographic context)

| Field | Value |
|-------|-------|
| Publisher | OpenStreetMap Foundation and contributors |
| API | Overpass API (overpass-api.de) |
| Licence | ODbL — share-alike applies to derived databases |
| Attribution | "© OpenStreetMap contributors" required wherever map data is displayed |
| Coverage | Karachi bounding box: 24.74°N–25.10°N, 66.85°E–67.20°E |
| Snapshot | Collected once during the event; `osm_base` timestamp recorded in manifest.csv |
| Features | highways, parks (leisure=park), water (natural=water), key facilities |
| Role | Geographic context only — not measurements of shade, temperature, or safety |
| Note | OSM data may be incomplete for some Karachi areas |

Docs: [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API) ·
[OSM Copyright](https://www.openstreetmap.org/copyright) ·
[ODbL](https://opendatacommons.org/licenses/odbl/)

---

## 4. RAG Reference Corpus

RAG documents are guidance references only. They are **not** model-training labels
and do not count toward the minimum dataset volume.

Candidate sources (ingest only after confirming reuse terms):
- WHO heat-health action plans and guidance
- ILO heat-stress factsheets
- PMD Sindh heatwave bulletins (confirm reuse terms before ingesting)
- NDMA heat-wave guide (confirm reuse terms before ingesting)

Each ingested document must have confirmed reuse permission and its full citation
preserved in the RAG corpus metadata (title, publisher, URL, date, page/section, chunk_id).

---

## 5. Data Volume

| Source | Expected rows | Layer |
|--------|---------------|-------|
| Open-Meteo historical | ~26,304 hourly records per grid cell | Bronze → Silver → Gold |
| OSM Karachi snapshot | ≥1,000 geographic features | Bronze → Silver → Gold |
| **Total** | **>50,000 rows combined** | |

---

## 6. Collection and Provenance Rules

- All data collected during the hackathon event.
- Raw responses preserved unchanged in `data/raw/`.
- `docs/manifest.csv` records: source, exact URL/query, fetch_time (UTC), SHA-256 hash, licence, `osm_base` for OSM.
- No Kaggle/GitHub/blog re-uploads used as substitutes for original sources.
- No synthetic or AI-generated data used for training or evaluation.
  Synthetic stress-test data (if any) is clearly labelled as such.
- No personal information collected.
- Pipeline reproducible via `make data`.

---

## 7. Time Split Summary

| Period | Role |
|--------|------|
| 2023-01-01 – 2024-12-31 | Development (training + cross-validation) |
| 2025-01-01 – 2025-12-31 | **Chronological holdout — not used during development** |

Target leakage prevention: every feature used to predict hour H must be
available at the time of prediction (no future data leaks into features).

---

## 8. Evaluation Protocol

1. Compute baseline MAE: lag-24h prediction on 2025 holdout rows.
2. Compute model MAE: trained model predictions on the same 2025 holdout rows.
3. Model beats baseline if model MAE < baseline MAE.
4. Report additional slices: by month, hour-of-day, and grid cell.
5. Report prediction-interval coverage (conformal or bootstrap).
6. Do not re-run or change anything after viewing 2025 holdout scores.

---

*This declaration is frozen at commit time. Any deviation must be explained and
documented in `AI_Use_Log.md` before implementation.*
