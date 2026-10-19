---
title: SaayaShift Heat-Aware Shift Planner
emoji: 🌡️
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 4.42.0
app_file: app.py
pinned: false
license: mit
---

# SaayaShift — Heat-Aware Shift Planner

**Karachi-first · Open-source · HackInsight 2026**

SaayaShift helps outdoor-work supervisors in Karachi compare shift windows using
hourly heat-exposure forecasts. It ranks feasible windows, explains every recommendation,
and answers heat-safety questions with cited sources.

> **Decision-support tool only.** Does not diagnose heat illness, predict individual
> outcomes, or certify any shift or worksite as safe.

## Features
- Shift Planner — ranked windows with heat-exposure explanations
- Forecast View — hourly apparent temperature chart with source freshness
- Map Context — OSM geographic context (not safety measurements)
- Safety Q&A — cited answers with "I don't know" abstention
- Pipeline Status — live data freshness, DQ gate status, last successful run

## Attribution
Weather data: [Open-Meteo.com](https://open-meteo.com) (CC BY 4.0)
Map data: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) (ODbL)
