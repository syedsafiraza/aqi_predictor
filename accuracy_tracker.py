import logging

import pandas as pd

import config
from utils import db
from utils.features import FEATURE_COLUMNS, make_supervised
from utils.model_registry import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("accuracy_tracker")


def log_current_forecasts():
    for city in config.CITIES:
        df = db.read_features(city)
        if df.empty:
            continue
        df = make_supervised(df)
        latest_row = df.dropna(subset=FEATURE_COLUMNS).iloc[-1:]
        if latest_row.empty:
            continue

        latest_date = latest_row["date"].iloc[0]
        X = latest_row[FEATURE_COLUMNS]

        for horizon in config.FORECAST_HORIZONS:
            point_model = load_model(city, horizon, kind="point")
            if point_model is None:
                continue
            lower_model = load_model(city, horizon, kind="lower")
            upper_model = load_model(city, horizon, kind="upper")

            predicted = float(point_model.predict(X)[0])
            lower = float(lower_model.predict(X)[0]) if lower_model else None
            upper = float(upper_model.predict(X)[0]) if upper_model else None
            target_date = (latest_date + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d")

            db.log_forecast(city, horizon, target_date, predicted, lower, upper)

    log.info("Logged current forecasts for all cities.")


def compute_rolling_accuracy(city, horizon, window=14):
    log_df = db.read_forecast_log(city)
    if log_df.empty:
        return None, 0
    sub = log_df[(log_df["horizon"] == horizon) & log_df["actual_aqi"].notna()]
    if sub.empty:
        return None, 0
    sub = sub.sort_values("target_date").tail(window)
    mae = (sub["predicted_aqi"] - sub["actual_aqi"]).abs().mean()
    return float(mae), len(sub)


def main():
    db.backfill_actuals()
    log_current_forecasts()
    for city in config.CITIES:
        for horizon in config.FORECAST_HORIZONS:
            mae, n = compute_rolling_accuracy(city, horizon)
            if n:
                log.info(f"{city} h{horizon}: rolling MAE over last {n} forecasts = {mae:.1f}")


if __name__ == "__main__":
    main()
