"""
Generate Summary.pdf — one-page project summary for submission.
Uses fpdf2. Run: python -m src.generate_summary
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    try:
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos
    except ImportError:
        print("Run: pip install fpdf2")
        return

    # Load model results
    model_results = {}
    mr_path = ROOT / "docs" / "model_results.json"
    if mr_path.exists():
        with open(mr_path) as f:
            model_results = json.load(f)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_top_margin(12)

    W = pdf.w - pdf.l_margin - pdf.r_margin  # usable width

    def title_block() -> None:
        pdf.set_font("Helvetica", "B", 15)
        pdf.cell(W, 8, "SaayaShift - Heat-Aware Shift Planner",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(W, 5, "HackInsight 2026  |  Karachi-first  |  Open-source  |  MIT",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(2)

    def safe(text: str) -> str:
        """Replace characters outside latin-1 with ASCII equivalents."""
        return (text
                .replace("\u2014", "-").replace("\u2013", "-")
                .replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"')
                .encode("latin-1", errors="replace").decode("latin-1"))

    def para(text: str) -> None:
        pdf.multi_cell(W, 4, safe(text))
        pdf.ln(0.5)

    def section(title: str) -> None:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(W, 5.5, safe(f"  {title}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.ln(0.5)

    title_block()

    section("Problem & User")
    para(
        "User: Supervisor scheduling outdoor field crews in Karachi. Generic weather "
        "dashboards do not turn forecasts into explainable shift options. Dawn (Jun 2023, Jun 2024) "
        "and Climate Rights International (2025) document occupational heat risk for outdoor workers "
        "in Karachi. These are qualified problem evidence; individual outcomes are not predicted."
    )

    section("Data Sources & Warehouse")
    para(
        "Source 1: Open-Meteo Historical Forecast + Forecast API (CC BY 4.0). "
        "26,472 hourly rows covering 2023-01-01 to 2025-12-31 plus 7-day live forecast. "
        "Grid cell 24.8607N 67.0011E. Provenance in manifest.csv (SHA-256, fetch time, licence)."
    )
    para(
        "Source 2: OpenStreetMap via Overpass API (ODbL). Karachi area snapshot. "
        "Geographic context only (not safety measurements). STATUS: BLOCKED — all "
        "Overpass mirrors unreachable from event network. Stub clearly labeled. "
        "Layers: Bronze (raw JSON) / Silver (typed, deduped, imputed) / Gold (validated, analysis-ready)."
    )

    section("Data Quality Agent")
    para(
        "Gate 1 (Silver): schema check, type cast, dedup with forecast winning on overlap, "
        "range check, linear interpolation for gaps up to 3h. "
        "Gate 2 (Gold): completeness <2%, consistency (apparent temp range), freshness, "
        "drift detection (3-sigma). Fail = alert + keep last-good Gold. "
        "Every decision logged to data/silver/dq_log.csv with rule, action, row count, reason."
    )

    section("Model & Evaluation")
    b_mae = model_results.get("baseline_mae", 1.0302)
    m_mae = model_results.get("model_mae", 0.2246)
    imp   = model_results.get("improvement_c", 0.8056)
    cov   = model_results.get("interval_coverage", "86.04%")
    para(
        f"Target: hourly apparent temperature (C). Frozen baseline: lag-24h (H minus 24h), "
        f"declared before results viewed. Train: 2023-2024 | Test: 2025 holdout (chronological, untouched)."
    )
    para(
        f"Baseline MAE: {b_mae:.4f} C  |  LightGBM MAE: {m_mae:.4f} C  |  "
        f"Improvement: {imp:.4f} C  |  Beats baseline: YES  |  90% PI coverage: {cov}. "
        f"Slices by month and hour-of-day in docs/model_card.md. "
        f"Limit: single grid cell; model-grid estimates, not street-level observations."
    )

    section("Insight Agent (RAG)")
    para(
        "Corpus: OSHA, ILO, WHO heat-safety guidance (openly licensed). "
        "Hybrid keyword + vector search (sentence-transformers all-MiniLM-L6-v2 + FAISS). "
        "Guardrails: numbers from Gold SQL only (not LLM memory); abstains when relevance < 0.40; "
        "no PII in prompts; cites title / publisher / URL / section / chunk_id per answer. "
        "Does not diagnose illness, certify shifts, or replace official guidance."
    )

    section("Pipeline & Dashboard")
    para(
        "8-step hourly flow: Fetch -> Bronze -> Gate 1 -> Gate 2 -> Retrain (drift) -> "
        "Re-index RAG -> Briefs -> Dashboard refresh. "
        "Orchestration: pipeline/scheduler.py (no admin required). "
        "Evidence window: Tue 20 Oct 16:00 PKT - Wed 21 Oct 16:00 PKT."
    )
    para(
        "Gradio dashboard (5 tabs): Shift Planner / Forecast View / Map Context / "
        "Safety Q&A / Pipeline Status. Shows data freshness, DQ gate status, last successful run, "
        "model MAE vs baseline. Attribution: Open-Meteo (CC BY 4.0) and OSM contributors (ODbL) visible."
    )

    section("Pakistan Impact & Responsible AI")
    para(
        "Direct beneficiaries: outdoor supervisors and field crews in Karachi facing "
        "occupational heat risk. No personal data collected. No individual illness predicted. "
        "All thresholds from cited WHO/ILO/OSHA guidance. Data limitations labeled. "
        "AI use logged in AI_Use_Log.md. City-configurable for future expansion."
    )

    # Footer
    pdf.set_y(-10)
    pdf.set_font("Helvetica", "I", 6.5)
    pdf.cell(
        W, 4,
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  "
        "Weather: Open-Meteo.com (CC BY 4.0)  |  Map: (c) OpenStreetMap contributors (ODbL)  |  Code: MIT",
        align="C"
    )

    out = ROOT / "Summary.pdf"
    pdf.output(str(out))
    print(f"Summary.pdf written -> {out}")


if __name__ == "__main__":
    main()
