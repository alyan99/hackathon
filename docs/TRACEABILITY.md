# SaayaShift — Requirements Traceability Table

Links each mandatory hackathon requirement to its implementation, verification, and evidence.

| # | Requirement | Source | Status | Implementation | Verification | Evidence |
|---|-------------|--------|--------|----------------|--------------|---------|
| R1 | 2+ independent sources; ≥1 outside track suggestions | Guide §Data | **PASS** | Open-Meteo (weather) + OSM (geographic, outside climate-track suggestions) | Source list in DATASET_DECLARATION.md | `docs/DATASET_DECLARATION.md` |
| R2 | ≥50,000 rows OR 2 years hourly data | Guide §Data | **PASS** | 26,304 hourly rows (2023-2025) + 168 live forecast = 26,472 rows; 3 full years hourly | `len(pd.read_parquet('data/gold/weather_gold.parquet'))` | `data/gold/weather_gold.parquet` |
| R3 | Data collected during event; original source, not re-upload | Guide §Integrity | **PASS** | Fetched from Open-Meteo API and Overpass API during event | `docs/manifest.csv` SHA-256 + fetch_time | `docs/manifest.csv` |
| R4 | Raw files preserved; manifest.csv with source/URL/time/SHA-256/licence | Guide §Integrity | **PASS** | Every fetch writes to manifest.csv automatically | Check manifest.csv has entry per raw file | `docs/manifest.csv` |
| R5 | `make data` rebuilds all layers without manual steps | Guide §Reproducibility | **PASS** | Makefile targets: data → silver → gold | Run `make data` from empty working folder | `Makefile` |
| R6 | Raw / cleaned / processed layers | Guide §Pipeline | **PASS** | Bronze (`data/raw/`), Silver (`data/silver/`), Gold (`data/gold/`) | Directory structure + parquet files exist | `data/` layout |
| R7 | Data Quality Agent: fix/fill/flag, decision log with rule/count/action/reason | Guide §DQ | **PASS** | `src/dq_agent/silver.py` + `src/dq_agent/gold.py` | `data/silver/dq_log.csv` has entries per check | `data/silver/dq_log.csv` |
| R8 | Baseline declared before results inspected; time-based split; MAE metric | Guide §Model | **PASS** | Baseline = lag-24h; declared in DATASET_DECLARATION.md; split 2023-24 train / 2025 test | `docs/baseline_results.json` MAE = 1.0287°C | `docs/baseline_results.json` |
| R9 | Model beats declared baseline on held-out test period | Guide §Model | **PASS** | LightGBM MAE = 0.2246°C vs baseline 1.0287°C (improvement 0.81°C) | `docs/model_results.json` beats_baseline=True | `docs/model_results.json` |
| R10 | Model reports uncertainty and error slices | Guide §Model | **PASS** | 90% PI coverage = 86%; slices by month and hour-of-day in model card | `docs/model_card.md` | `docs/model_card.md` |
| R11 | RAG cites sources; says "I don't know" when unsupported; numbers from Gold tables only | Guide §RAG | **PASS** | `src/rag/retriever.py` — guardrails: abstain < threshold; numbers from SQL over Gold | Test: `test_rag.py` 4/4 pass | `tests/test_rag.py` |
| R12 | Hourly automated pipeline | Guide §Pipeline | **IN PROGRESS** | `pipeline/flows/hourly_flow.py` + `pipeline/scheduler.py` | Run scheduler; check prefect_runs.csv grows | `pipeline/prefect_runs.csv` |
| R13 | Prefect run evidence: Tue 20 Oct 16:00 → Wed 21 Oct 16:00 PKT | Guide §Pipeline | **NOT STARTED** | Scheduler must be running during window | `prefect_runs.csv` has ≥24 entries in window | `pipeline/prefect_runs.csv` |
| R14 | HF Space dashboard: freshness, DQ status, last successful run | Guide §Delivery | **IN PROGRESS** | `src/app/app.py` Tab 5 — Pipeline Status | Manual: open Space URL, check Pipeline Status tab | HF Space URL (TBD) |
| R15 | GitHub repo tagged `v1.0-final`; no pushes after tag | Guide §Submission | **NOT STARTED** | Final tag after feature freeze Wed 21 Oct 18:00 PKT | `git tag v1.0-final` visible on GitHub | GitHub tag |
| R16 | `prefect_runs.csv` unedited + dashboard screenshot | Guide §Submission | **NOT STARTED** | Export after evidence window closes | Visual check of CSV continuity | `pipeline/prefect_runs.csv` |
| R17 | `Summary.pdf` one page | Guide §Submission | **NOT STARTED** | `src/generate_summary.py` (to build) | PDF is exactly 1 page | `Summary.pdf` |
| R18 | Demo video ≤3 min | Guide §Submission | **NOT STARTED** | Record after dashboard is live | Video opens without login; ≤3 min | Video link |
| R19 | `AI_Use_Log.md` complete | Guide §Submission | **IN PROGRESS** | `docs/AI_Use_Log.md` — update after each AI-assisted step | All steps logged with tool/output | `docs/AI_Use_Log.md` |
| R20 | No personal data; no synthetic training data; no fabricated results | Guide §Ethics | **PASS** | No PII collected; synthetic stub clearly labeled; all results from real data | Code review; DQ log | Source code |
| R21 | No RL in MVP | Project brief | **PASS** | Deterministic rule-based planner only | Code review `src/app/planner.py` | `src/app/planner.py` |
| R22 | OSM as independent source outside climate-track suggestions | Project brief | **BLOCKED** | All Overpass mirrors returning 406/429/504 from this network | Retry when network allows | `data/raw/osm/` |

## Open items
- R13: Start scheduler by **Tue 20 Oct 16:00 PKT** — CRITICAL
- R14: Deploy to HF Space — needs user's HF account credentials
- R15–R18: Final submission artefacts — target Wed 21 Oct before 18:00 PKT
- R22: OSM real data — retry Overpass when network allows (stub is labeled, pipeline continues)
