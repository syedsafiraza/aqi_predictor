"""
Runs the real aggregation -> feature engineering -> SQLite storage ->
training -> accuracy-logging code paths on synthetic data, so the whole
pipeline can be verified without live network access.
"""
import numpy as np
import pandas as pd

import config
from utils.data_fetch import merge_and_aggregate_daily
from utils.features import build_feature_table, make_supervised, FEATURE_COLUMNS
from utils import db
from training_pipeline import train_one_horizon, try_import_tf
from utils.model_registry import save_model, load_model
from accuracy_tracker import log_current_forecasts, compute_rolling_accuracy

np.random.seed(0)


def synthetic_hourly(n_days=350):
    hours = pd.date_range("2024-01-01", periods=n_days * 24, freq="h")
    n = len(hours)
    seasonal = 90 + 40 * np.sin(np.arange(n) / (24 * 30) * 2 * np.pi)
    aqi = np.clip(seasonal + np.random.normal(0, 15, n), 5, 400)

    air_df = pd.DataFrame({"time": hours})
    air_df["pm10"] = aqi * 0.6 + np.random.normal(0, 5, n)
    air_df["pm2_5"] = aqi * 0.4 + np.random.normal(0, 3, n)
    air_df["carbon_monoxide"] = np.random.uniform(200, 800, n)
    air_df["nitrogen_dioxide"] = np.random.uniform(10, 60, n)
    air_df["sulphur_dioxide"] = np.random.uniform(5, 40, n)
    air_df["ozone"] = np.random.uniform(10, 90, n)
    air_df["us_aqi"] = aqi

    weather_df = pd.DataFrame({"time": hours})
    weather_df["temperature_2m"] = 25 + 10 * np.sin(np.arange(n) / (24 * 365) * 2 * np.pi) + np.random.normal(0, 2, n)
    weather_df["relative_humidity_2m"] = np.random.uniform(20, 80, n)
    weather_df["wind_speed_10m"] = np.random.uniform(0, 20, n)
    weather_df["surface_pressure"] = np.random.uniform(1000, 1020, n)
    return air_df, weather_df


def main():
    config.DB_PATH = "/tmp/tenpearls_test.db"
    config.MODEL_DIR = "/tmp/tenpearls_test_models"

    print("1) Synthetic hourly data...")
    air_df, weather_df = synthetic_hourly()

    print("2) Daily aggregation + feature engineering (with Punjab smog flag)...")
    daily = merge_and_aggregate_daily(air_df, weather_df)
    featured = build_feature_table(daily, province="Punjab")
    assert "is_smog_season" in featured.columns
    assert "aqi_momentum" in featured.columns
    assert "wind_dispersion" in featured.columns
    print(f"   -> {len(featured)} rows, new features present")

    print("3) SQLite upsert + read-back...")
    db.upsert_features("Lahore", featured)
    read_back = db.read_features("Lahore")
    assert len(read_back) == len(featured), "SQLite round-trip row count mismatch!"
    print(f"   -> round-trip OK, {len(read_back)} rows")

    print("4) Supervised targets...")
    sup = make_supervised(read_back)
    missing = [c for c in FEATURE_COLUMNS if c not in sup.columns]
    assert not missing, f"missing columns: {missing}"

    print("5) Training (point + quantile interval models) for horizon=1...")
    tf_module = try_import_tf()
    result = train_one_horizon("Lahore", sup, 1, tf_module)
    assert result is not None
    print(f"   -> best={result['best_name']} metrics={result['best_metrics']} "
          f"coverage={result['coverage']:.0%}")

    print("6) Save/load models + metadata...")
    save_model("Lahore", 1, result["best_model"], "point")
    save_model("Lahore", 1, result["lower_model"], "lower")
    save_model("Lahore", 1, result["upper_model"], "upper")
    import json
    db.save_model_metadata("Lahore", 1, result["best_name"], result["best_metrics"],
                            json.dumps(result["leaderboard"]))
    loaded = load_model("Lahore", 1, "point")
    assert loaded is not None
    meta = db.read_model_metadata("Lahore", 1)
    assert meta is not None
    print(f"   -> metadata round-trip OK: {meta}")

    print("7) Accuracy tracker: log forecasts + backfill actuals + rolling MAE...")
    log_current_forecasts()
    db.backfill_actuals()
    mae, n = compute_rolling_accuracy("Lahore", 1)
    print(f"   -> logged forecast(s); resolved so far: {n} (expected 0 on a single run — "
          f"actuals only resolve once real future dates appear in the feature table)")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
