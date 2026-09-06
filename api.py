from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException

import config
from utils import db
from utils.aqi_utils import aqi_category
from utils.features import FEATURE_COLUMNS, make_supervised
from utils.model_registry import load_model
from utils.scoring import clean_air_score, score_label

app = FastAPI(title="10Pearls AQI Predictor API", version="1.0")


@app.get("/")
def root():
    return {"service": "10Pearls AQI Predictor", "cities": list(config.CITIES.keys())}


@app.get("/cities")
def cities():
    return config.CITIES


@app.get("/aqi/{city}")
def current_aqi(city: str):
    city = city.title()
    if city not in config.CITIES:
        raise HTTPException(404, f"Unknown city '{city}'")
    df = db.read_features(city)
    if df.empty:
        raise HTTPException(404, f"No data for {city} yet")
    latest = df.iloc[-1]
    label, _, advice = aqi_category(latest["aqi"])
    score = clean_air_score(latest["aqi"], latest.get("aqi_momentum", 0))
    return {
        "city": city,
        "date": latest["date"].strftime("%Y-%m-%d"),
        "aqi": round(float(latest["aqi"]), 1),
        "category": label,
        "advice": advice,
        "clean_air_score": score,
        "clean_air_label": score_label(score),
    }


@app.get("/forecast/{city}")
def forecast(city: str):
    city = city.title()
    if city not in config.CITIES:
        raise HTTPException(404, f"Unknown city '{city}'")
    df = db.read_features(city)
    if df.empty:
        raise HTTPException(404, f"No data for {city} yet")
    df = make_supervised(df)
    row = df.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    if row.empty:
        raise HTTPException(404, "Not enough history to forecast yet")

    X = row[FEATURE_COLUMNS]
    results = []
    for h in config.FORECAST_HORIZONS:
        point = load_model(city, h, "point")
        if point is None:
            continue
        lower = load_model(city, h, "lower")
        upper = load_model(city, h, "upper")
        target_date = row["date"].iloc[0] + pd.Timedelta(days=h)
        pred = max(0, float(point.predict(X)[0]))
        results.append({
            "horizon_days": h,
            "date": target_date.strftime("%Y-%m-%d"),
            "predicted_aqi": round(pred, 1),
            "lower_bound": round(max(0, float(lower.predict(X)[0])), 1) if lower else None,
            "upper_bound": round(float(upper.predict(X)[0]), 1) if upper else None,
        })

    if not results:
        raise HTTPException(404, "No trained models yet for this city")

    return {"city": city, "generated_at": datetime.utcnow().isoformat(), "forecast": results}


@app.get("/compare")
def compare_cities():
    output = {}
    for city in config.CITIES:
        df = db.read_features(city)
        if df.empty:
            continue
        latest = df.iloc[-1]
        output[city] = round(float(latest["aqi"]), 1)
    if not output:
        raise HTTPException(404, "No data available yet")
    return {
        "readings": output,
        "cleanest": min(output, key=output.get),
        "most_polluted": max(output, key=output.get),
    }
