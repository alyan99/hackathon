"""
SaayaShift — Gradio Dashboard

5 tabs:
  1. Shift Planner    — ranked windows with explanations
  2. Forecast View    — hourly apparent temp chart + freshness
  3. Map Context      — OSM map with disclaimer
  4. Safety Q&A       — RAG with citations + "I don't know"
  5. Pipeline Status  — DQ gate status, last successful run

Attribution:
  Weather data: Open-Meteo.com (CC BY 4.0)
  Map data: © OpenStreetMap contributors (ODbL)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH     = ROOT / "data" / "gold" / "weather_gold.parquet"
DQ_LOG_PATH   = ROOT / "data" / "silver" / "dq_log.csv"
PREFECT_CSV   = ROOT / "pipeline" / "prefect_runs.csv"
MODEL_RESULTS = ROOT / "docs" / "model_results.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gold() -> pd.DataFrame | None:
    if not GOLD_PATH.exists():
        return None
    df = pd.read_parquet(GOLD_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time")


def pipeline_status() -> dict:
    """Read pipeline status from Gold metadata and DQ log."""
    status = {
        "gold_freshness": "unknown",
        "gate1_status": "unknown",
        "gate2_status": "unknown",
        "last_run": "unknown",
        "total_rows": 0,
    }

    df = load_gold()
    if df is not None:
        latest = df["time"].max()
        now = pd.Timestamp.now(tz="UTC")
        age_h = (now - latest).total_seconds() / 3600
        status["gold_freshness"] = f"{age_h:.1f}h ago ({latest.strftime('%Y-%m-%d %H:%M UTC')})"
        status["total_rows"] = len(df)

    if DQ_LOG_PATH.exists():
        dq = pd.read_csv(DQ_LOG_PATH)
        passes = dq[dq["action"] == "pass"]
        alerts = dq[dq["action"] == "alert"]
        status["gate1_status"] = "✅ PASS" if not alerts.empty or not passes.empty else "⚠️ No runs yet"
        status["gate2_status"] = "✅ PASS" if len(passes) > 5 else "⚠️ Check log"
        if not dq.empty:
            status["last_run"] = dq["timestamp_utc"].iloc[-1]

    if PREFECT_CSV.exists():
        pf = pd.read_csv(PREFECT_CSV)
        if not pf.empty:
            status["last_run"] = pf.iloc[-1].get("end_time", status["last_run"])

    return status


def load_model_results() -> dict:
    if MODEL_RESULTS.exists():
        with open(MODEL_RESULTS) as f:
            return json.load(f)
    return {}


# ── Tab 1 — Shift Planner ─────────────────────────────────────────────────────

def run_planner(duration_h: int, op_start: int, op_end: int, top_n: int) -> str:
    from src.app.planner import plan_shift
    results = plan_shift(
        duration_h=int(duration_h),
        op_start_h=int(op_start),
        op_end_h=int(op_end),
        top_n=int(top_n),
    )
    if not results:
        return "No windows found. Check that Gold data is available."
    if "error" in results[0]:
        return f"Error: {results[0]['error']}"

    lines = [
        f"## Top {len(results)} Shift Windows — Ranked by Heat Exposure",
        "",
        "| Rank | Start | End | Duration | Mean Temp | Heat Level | Score |",
        "|------|-------|-----|----------|-----------|------------|-------|",
    ]
    for r in results:
        lines.append(
            f"| {r['rank']} | {r['start']} | {r['end']} | {r['duration_h']}h "
            f"| {r['mean_apparent_temp_c']}°C | {r['heat_level'].upper()} | {r['score']} |"
        )
    lines.append("")
    lines.append("### Explanations")
    for r in results:
        lines.append(f"**Rank {r['rank']}:** {r['reason']}")
        lines.append("")

    return "\n".join(lines)


# ── Tab 2 — Forecast View ─────────────────────────────────────────────────────

def forecast_chart():
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None, "plotly not installed"

    df = load_gold()
    if df is None:
        return None, "Gold table not found. Run `make data` first."

    # Show last 7 days + next available forecast
    recent = df.tail(24 * 7)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent["time"],
        y=recent["apparent_temperature"],
        mode="lines",
        name="Apparent Temperature",
        line=dict(color="#e63946", width=2),
    ))

    # Heat level bands
    fig.add_hrect(y0=32, y1=38, fillcolor="yellow", opacity=0.08, annotation_text="Caution", annotation_position="top left")
    fig.add_hrect(y0=38, y1=43, fillcolor="orange", opacity=0.10, annotation_text="High", annotation_position="top left")
    fig.add_hrect(y0=43, y1=60, fillcolor="red", opacity=0.10, annotation_text="Extreme", annotation_position="top left")

    latest = df["time"].max().strftime("%Y-%m-%d %H:%M UTC")
    source_note = "Data: Open-Meteo.com (CC BY 4.0) · Grid cell resolution (~1km) · Not street-level measurements"

    fig.update_layout(
        title=f"Karachi Apparent Temperature — last 7 days<br><sup>{source_note}</sup>",
        xaxis_title="Time (UTC)",
        yaxis_title="Apparent Temperature (°C)",
        height=420,
        margin=dict(l=60, r=20, t=80, b=60),
        hovermode="x unified",
    )
    freshness = f"Data as of: {latest}  |  Source: Open-Meteo (CC BY 4.0)"
    return fig, freshness


# ── Tab 3 — Map ───────────────────────────────────────────────────────────────

def get_map_html_tab() -> str:
    from src.app.map_layer import get_map_html
    return get_map_html()


# ── Tab 4 — Safety Q&A ────────────────────────────────────────────────────────

def answer_question(question: str) -> str:
    if not question.strip():
        return "Please enter a question."
    from src.rag.retriever import retrieve, format_response
    result = retrieve(question)
    return format_response(result)


# ── Tab 5 — Pipeline Status ───────────────────────────────────────────────────

def get_pipeline_status() -> str:
    s = pipeline_status()
    mr = load_model_results()

    lines = [
        "## Pipeline Status",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Data freshness | {s['gold_freshness']} |",
        f"| Gate 1 (Silver) | {s['gate1_status']} |",
        f"| Gate 2 (Gold) | {s['gate2_status']} |",
        f"| Last successful run | {s['last_run']} |",
        f"| Total Gold rows | {s['total_rows']:,} |",
    ]

    if mr:
        lines += [
            "",
            "## Model Metrics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Algorithm | {mr.get('algorithm', 'N/A')} |",
            f"| Baseline MAE (lag-24h) | {mr.get('baseline_mae', 'N/A')} °C |",
            f"| Model MAE | {mr.get('model_mae', 'N/A')} °C |",
            f"| Beats baseline? | {'✅ YES' if mr.get('beats_baseline') else '❌ NO'} |",
            f"| 90% PI coverage | {mr.get('interval_coverage', 'N/A')} |",
        ]

    lines += [
        "",
        "---",
        "_Weather data: [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0) · "
        "Map data: [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright) (ODbL)_",
    ]

    return "\n".join(lines)


# ── App layout ────────────────────────────────────────────────────────────────

def build_app():
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Run: pip install gradio")

    with gr.Blocks(title="SaayaShift — Heat-Aware Shift Planner", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# 🌡️ SaayaShift — Heat-Aware Shift Planner
**Karachi · Decision support for outdoor work supervisors**

> This tool helps compare shift windows using heat forecasts. It does not diagnose heat illness,
> predict individual outcomes, or certify any shift or worksite as safe.

_Weather: [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0) · Map: [© OpenStreetMap contributors](https://www.openstreetmap.org/copyright) (ODbL)_
            """
        )

        with gr.Tabs():
            # ── Tab 1: Shift Planner ──────────────────────────────────────────
            with gr.Tab("🗓️ Shift Planner"):
                gr.Markdown("### Plan outdoor shifts using heat forecasts")
                with gr.Row():
                    duration_in = gr.Slider(1, 12, value=4, step=1, label="Shift duration (hours)")
                    op_start_in = gr.Slider(0, 12, value=6, step=1, label="Earliest start hour (local)")
                    op_end_in   = gr.Slider(12, 24, value=20, step=1, label="Latest end hour (local)")
                    top_n_in    = gr.Slider(1, 10, value=5, step=1, label="Number of windows to show")
                plan_btn = gr.Button("🔍 Find Best Windows", variant="primary")
                plan_out = gr.Markdown()
                plan_btn.click(run_planner, inputs=[duration_in, op_start_in, op_end_in, top_n_in], outputs=plan_out)

            # ── Tab 2: Forecast View ──────────────────────────────────────────
            with gr.Tab("📈 Forecast View"):
                gr.Markdown("### Hourly apparent temperature — Karachi")
                refresh_btn = gr.Button("🔄 Refresh Chart")
                freshness_lbl = gr.Markdown()
                chart_out = gr.Plot()
                def refresh_chart():
                    fig, freshness = forecast_chart()
                    return fig, freshness
                refresh_btn.click(refresh_chart, outputs=[chart_out, freshness_lbl])
                demo.load(refresh_chart, outputs=[chart_out, freshness_lbl])

            # ── Tab 3: Map Context ────────────────────────────────────────────
            with gr.Tab("🗺️ Map Context"):
                gr.Markdown(
                    "### Karachi work zone — geographic context only\n\n"
                    "⚠️ **OSM features shown here are geographic context only — "
                    "not measurements of shade, temperature, or safety.**"
                )
                map_out = gr.HTML()
                load_map_btn = gr.Button("🗺️ Load Map")
                load_map_btn.click(get_map_html_tab, outputs=map_out)

            # ── Tab 4: Safety Q&A ─────────────────────────────────────────────
            with gr.Tab("💬 Safety Q&A"):
                gr.Markdown(
                    "### Heat-safety questions — answered with cited sources\n\n"
                    "Answers come from approved guidance only. "
                    "If no supporting source exists, the system says **'I don't know'**. "
                    "This does not replace official occupational-health or emergency guidance."
                )
                q_in  = gr.Textbox(label="Your question", placeholder="e.g. How often should workers drink water in extreme heat?")
                ask_btn = gr.Button("Ask", variant="primary")
                q_out = gr.Markdown()
                ask_btn.click(answer_question, inputs=q_in, outputs=q_out)
                q_in.submit(answer_question, inputs=q_in, outputs=q_out)

            # ── Tab 5: Pipeline Status ────────────────────────────────────────
            with gr.Tab("⚙️ Pipeline Status"):
                status_refresh = gr.Button("🔄 Refresh Status")
                status_out = gr.Markdown()
                status_refresh.click(get_pipeline_status, outputs=status_out)
                demo.load(get_pipeline_status, outputs=status_out)

    return demo


def main() -> None:
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
