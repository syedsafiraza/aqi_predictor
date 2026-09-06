# 10Pearls AQI Predictor

A 100% serverless, end-to-end AQI forecasting system for five Pakistani
cities — **Karachi, Lahore, Islamabad, Multan, and Quetta** (used as the
representative city for Balochistan, since the province itself has no
single coordinate). Predicts AQI 1, 2, and 3 days ahead with uncertainty
ranges, and includes a live dashboard, a REST API, and full automation.

## Requirement coverage

| Brief requirement | Implementation |
|---|---|
| Feature pipeline (raw data → features → feature store) | `feature_pipeline.py`, hourly |
| Time-based + derived features (hour/day/month, AQI change rate) | `utils/features.py` — plus AQI momentum and a wind-dispersion interaction feature |
| Historical backfill | `backfill_historical.py` |
| Feature store | SQLite (`data/aqi_store.db`) — portable, zero external service |
| Model training: Random Forest / Ridge / TensorFlow | `training_pipeline.py` — Ridge, Extra Trees, and an optional Keras MLP (auto-skipped if TensorFlow isn't installed) |
| RMSE / MAE / R² evaluation | Computed and stored per city per horizon |
| Model registry | `utils/model_registry.py` |
| CI/CD (hourly feature job, daily training job) | `.github/workflows/` — three scheduled GitHub Actions workflows |
| Dashboard (Streamlit) | `app.py` |
| Flask/FastAPI | `api.py` — REST endpoints for forecasts, current AQI, and city comparison |
| EDA | Historical trend, day-of-week distribution, pollutant breakdown, all in the dashboard |
| SHAP / LIME | SHAP feature-importance plot per city, shown in the dashboard |
| Hazard alerts | Threshold-based alert banner + optional webhook (Slack/Discord/Telegram) |
| Statistical → deep learning model range | Ridge (statistical) → Extra Trees (ensemble) → MLP (deep learning) |

## Bonus features

- **Prediction intervals** — every forecast includes a low/high range (10th/90th percentile quantile regression), not just a single number
- **Forecast accuracy tracker** — the system logs its own forecasts and checks them against real outcomes once known, so the dashboard shows a live rolling accuracy, not just a training-time metric
- **Clean Air Score** — a 0–100 composite index blending current AQI with its short-term trend
- **Punjab smog-season alert** — a seasonal (Oct–Feb) warning specific to Lahore and Multan, where crop burning and temperature inversions reliably worsen air quality
- **Interactive map view** of all five cities colour-coded by live AQI
- **REST API** for programmatic access alongside the dashboard
- **City comparison** — cleanest/most polluted city at a glance

## Architecture

```
Open-Meteo (weather + air quality)
        |
        v
feature_pipeline.py --hourly-->  SQLite (features, model metadata, forecast log)
        |
        v
training_pipeline.py --daily--> model registry (point + interval models, per city/horizon)
        |
        v
accuracy_tracker.py --daily--> resolves past forecasts, logs new ones
        |
        v
app.py (Streamlit dashboard)  +  api.py (FastAPI)
```

## Setup

```bash
pip install -r requirements.txt

python backfill_historical.py --days 300
python training_pipeline.py
python accuracy_tracker.py

streamlit run app.py          # dashboard
uvicorn api:app --reload      # REST API, http://localhost:8000/docs
```

Going forward, `feature_pipeline.py`, `training_pipeline.py`, and
`accuracy_tracker.py` are meant to run hourly/daily via the included
GitHub Actions workflows.

## Configuration

- `config.py` — cities, forecast horizons, hazard threshold, smog-season rule
- `ALERT_WEBHOOK_URL` (env var) — optional Slack/Discord/Telegram webhook for hazard alerts

## Testing without live internet access

`test_smoke.py` runs the full pipeline — aggregation, feature engineering,
SQLite storage, training (including the TensorFlow/SHAP path), and
accuracy logging — against synthetic data, so the system can be verified
end to end in an offline environment. Run with `python test_smoke.py`.

## Deployment

- Dashboard: push to GitHub, deploy `app.py` on Streamlit Community Cloud (free)
- API: any ASGI-compatible host (Render, Railway, Fly.io) running `uvicorn api:app`

## File structure

```
config.py
feature_pipeline.py
backfill_historical.py
training_pipeline.py
accuracy_tracker.py
app.py
api.py
test_smoke.py
utils/
    data_fetch.py
    features.py
    db.py
    model_registry.py
    aqi_utils.py
    scoring.py
    alerts.py
.github/workflows/
data/
models/
```
