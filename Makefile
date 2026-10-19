# SaayaShift — Makefile
# Usage:
#   make data     — fetch, clean, and build Gold tables
#   make train    — train the forecasting model
#   make eval     — evaluate model vs frozen baseline on 2025 holdout
#   make app      — launch the Gradio dashboard locally
#   make test     — run the full test suite
#   make pipeline — start the Prefect hourly flow locally
#   make clean    — remove generated artefacts (not raw data)

PYTHON = python
SRC    = src

.PHONY: data train eval app test pipeline clean

## ── Data ──────────────────────────────────────────────────────────────────────
data:
	@echo "==> [1/3] Fetching Open-Meteo data..."
	$(PYTHON) -m $(SRC).ingest.open_meteo
	@echo "==> [2/3] Fetching OSM data..."
	$(PYTHON) -m $(SRC).ingest.osm
	@echo "==> [3/3] Running DQ agent: Bronze → Silver → Gold..."
	$(PYTHON) -m $(SRC).dq_agent.silver
	$(PYTHON) -m $(SRC).dq_agent.gold
	@echo "==> Data pipeline complete. Check data/gold/ and data/silver/dq_log.csv"

## ── Model ─────────────────────────────────────────────────────────────────────
train:
	@echo "==> Training forecasting model on 2023-2024 Gold data..."
	$(PYTHON) -m $(SRC).model.train

eval:
	@echo "==> Evaluating model vs frozen baseline on 2025 holdout..."
	$(PYTHON) -m $(SRC).model.baseline
	$(PYTHON) -m $(SRC).model.train --eval-only
	@echo "==> Results written to docs/model_card.md"

## ── App ───────────────────────────────────────────────────────────────────────
app:
	@echo "==> Launching SaayaShift dashboard..."
	$(PYTHON) -m $(SRC).app.app

## ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=$(SRC) --cov-report=term-missing

## ── Pipeline ──────────────────────────────────────────────────────────────────
pipeline:
	@echo "==> Starting Prefect hourly flow (runs every hour)..."
	$(PYTHON) pipeline/flows/hourly_flow.py

## ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/silver/*.parquet data/gold/*.parquet
	@echo "==> Cleaned generated files. Raw data in data/raw/ is preserved."
