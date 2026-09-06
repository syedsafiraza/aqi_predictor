import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config
from utils import db
from utils.features import FEATURE_COLUMNS, make_supervised
from utils.model_registry import save_model
from utils.aqi_utils import is_hazardous
from utils.alerts import send_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("training_pipeline")

TEST_FRACTION = 0.2


def try_import_tf():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        return tf
    except ImportError:
        return None


def build_mlp(tf, n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_point_models(X_train, y_train, X_test, y_test, tf_module):
    candidates = {}

    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    candidates["ridge"] = (ridge, evaluate(y_test, ridge.predict(X_test)))

    et = ExtraTreesRegressor(n_estimators=300, max_depth=10, random_state=42, n_jobs=-1).fit(X_train, y_train)
    candidates["extra_trees"] = (et, evaluate(y_test, et.predict(X_test)))

    if tf_module is not None:
        try:
            mlp = build_mlp(tf_module, X_train.shape[1])
            mlp.fit(X_train.values, y_train.values, epochs=80, verbose=0, validation_split=0.1)
            preds = mlp.predict(X_test.values, verbose=0).flatten()
            candidates["deep_mlp"] = (mlp, evaluate(y_test, preds))
        except Exception as e:
            log.warning(f"MLP training skipped: {e}")

    best_name, (best_model, best_metrics) = min(candidates.items(), key=lambda kv: kv[1][1]["rmse"])
    leaderboard = {name: m[1] for name, m in candidates.items()}
    return best_name, best_model, best_metrics, leaderboard


def train_interval_models(X_train, y_train, X_test, y_test):
    lower = HistGradientBoostingRegressor(loss="quantile", quantile=0.1, random_state=42).fit(X_train, y_train)
    upper = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, random_state=42).fit(X_train, y_train)
    lo_pred, hi_pred = lower.predict(X_test), upper.predict(X_test)
    coverage = float(np.mean((y_test >= lo_pred) & (y_test <= hi_pred)))
    return lower, upper, coverage


def save_shap_plot(city, horizon, model, model_name, X_test):
    if model_name != "extra_trees":
        return None
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, show=False, plot_type="bar")
        d = os.path.join(config.MODEL_DIR, city.lower())
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"h{horizon}_shap.png")
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close()
        return path
    except Exception as e:
        log.warning(f"{city} h{horizon}: SHAP skipped ({e})")
        return None


def train_one_horizon(city, df, horizon, tf_module):
    target_col = f"target_h{horizon}"
    data = df.dropna(subset=FEATURE_COLUMNS + [target_col]).reset_index(drop=True)
    if len(data) < 20:
        log.warning(f"{city} h{horizon}: only {len(data)} usable rows, skipping")
        return None

    split = int(len(data) * (1 - TEST_FRACTION))
    train, test = data.iloc[:split], data.iloc[split:]
    X_train, y_train = train[FEATURE_COLUMNS], train[target_col]
    X_test, y_test = test[FEATURE_COLUMNS], test[target_col]

    best_name, best_model, best_metrics, leaderboard = train_point_models(X_train, y_train, X_test, y_test, tf_module)
    lower_model, upper_model, coverage = train_interval_models(X_train, y_train, X_test, y_test)
    shap_path = save_shap_plot(city, horizon, best_model, best_name, X_test)

    log.info(f"{city} h{horizon}: best={best_name} rmse={best_metrics['rmse']:.2f} "
             f"r2={best_metrics['r2']:.2f} coverage={coverage:.0%}")

    return {
        "best_name": best_name, "best_model": best_model, "best_metrics": best_metrics,
        "leaderboard": leaderboard, "lower_model": lower_model, "upper_model": upper_model,
        "coverage": coverage, "shap_path": shap_path,
    }


def main():
    tf_module = try_import_tf()
    if tf_module is None:
        log.info("TensorFlow not found — training Ridge + Extra Trees only.")

    for city, coords in config.CITIES.items():
        df = db.read_features(city)
        if df.empty:
            log.warning(f"{city}: no features found, run backfill_historical.py first")
            continue
        df = make_supervised(df)

        latest_aqi = df["aqi"].iloc[-1]
        if is_hazardous(latest_aqi):
            send_alert(f"{city}: latest AQI is {latest_aqi:.0f} (hazardous)")

        for horizon in config.FORECAST_HORIZONS:
            result = train_one_horizon(city, df, horizon, tf_module)
            if result is None:
                continue

            save_model(city, horizon, result["best_model"], kind="point")
            save_model(city, horizon, result["lower_model"], kind="lower")
            save_model(city, horizon, result["upper_model"], kind="upper")

            leaderboard = dict(result["leaderboard"])
            leaderboard["interval_coverage"] = result["coverage"]
            leaderboard["shap_plot"] = result["shap_path"]
            db.save_model_metadata(city, horizon, result["best_name"], result["best_metrics"], json.dumps(leaderboard))

    log.info("Training pipeline complete.")


if __name__ == "__main__":
    main()
