# SaayaShift — Model Card

**Generated:** 2026-10-19T09:12:34.090366+00:00

## Model
- **Algorithm:** LightGBMRegressor (n_estimators=500)
- **Target:** `apparent_temperature` (°C) — hourly per Open-Meteo grid cell
- **Training period:** 2023-01-01 – 2024-12-31
- **Test period:** 2025-01-01 – 2025-12-31 (chronological holdout, never seen during training)

## Evaluation Results

| Metric | Value |
|--------|-------|
| Baseline MAE (lag-24h) | **1.0302 °C** |
| Model MAE | **0.2246 °C** |
| Improvement | ✅ 0.8056 °C |
| Beats baseline? | YES |

## Error Slices

### MAE by Month

| Month | Baseline MAE | Model MAE |
|-------|-------------|-----------|
| 01 | 1.5775 | 0.2290 |
| 02 | 1.2876 | 0.2534 |
| 03 | 1.5835 | 0.2746 |
| 04 | 0.7840 | 0.2146 |
| 05 | 0.7887 | 0.2131 |
| 06 | 0.9917 | 0.2260 |
| 07 | 0.6258 | 0.1922 |
| 08 | 0.7862 | 0.2063 |
| 09 | 0.8144 | 0.1870 |
| 10 | 0.8793 | 0.2278 |
| 11 | 1.0450 | 0.2533 |
| 12 | 1.2319 | 0.2203 |

### MAE by Hour of Day

| Hour | Baseline MAE | Model MAE |
|------|-------------|-----------|
| 00:00 | 0.9593 | 0.1568 |
| 01:00 | 0.9686 | 0.1524 |
| 02:00 | 1.0209 | 0.1505 |
| 03:00 | 1.0727 | 0.1545 |
| 04:00 | 1.0846 | 0.1439 |
| 05:00 | 1.0868 | 0.1569 |
| 06:00 | 1.1259 | 0.2336 |
| 07:00 | 1.1573 | 0.1754 |
| 08:00 | 1.1719 | 0.2639 |
| 09:00 | 1.1645 | 0.2304 |
| 10:00 | 1.1581 | 0.2579 |
| 11:00 | 1.1835 | 0.3322 |
| 12:00 | 1.1567 | 0.3203 |
| 13:00 | 1.0926 | 0.3290 |
| 14:00 | 1.0711 | 0.3496 |
| 15:00 | 1.0218 | 0.3114 |
| 16:00 | 0.9096 | 0.2887 |
| 17:00 | 0.8926 | 0.2505 |
| 18:00 | 0.8851 | 0.2643 |
| 19:00 | 0.8667 | 0.2190 |
| 20:00 | 0.8912 | 0.1719 |
| 21:00 | 0.9116 | 0.1562 |
| 22:00 | 0.9284 | 0.1539 |
| 23:00 | 0.9438 | 0.1680 |

## Uncertainty
- Prediction intervals computed via 50-sample bootstrap
- Interval coverage (90% PI): 86.04%

## Limitations
- Apparent temperature is a model-grid estimate, not a street-level observation.
- The model is trained on one Karachi grid cell. Performance may differ for other locations.
- Features are derived from the same Open-Meteo source; an independent ground-truth would strengthen evaluation.
- Do not use model output to certify a shift or worksite as safe.

## Feature Importance
- temperature_2m: 4019.0000
- dewpoint_2m: 3435.0000
- windspeed_10m: 3198.0000
- lag_1h: 3061.0000
- hour_of_day: 2522.0000
- windspeed_10m_lag1: 2227.0000
- dewpoint_feature: 1858.0000
- lag_24h: 1816.0000
- lag_6h: 1034.0000
- relative_humidity_2m: 958.0000
- roll_std_24h: 951.0000
- roll_std_72h: 877.0000
- lag_168h: 751.0000
- roll_min_24h: 637.0000
- roll_mean_24h: 591.0000