import os

import joblib

import config


def _dir(city, horizon, kind):
    d = os.path.join(config.MODEL_DIR, city.lower())
    os.makedirs(d, exist_ok=True)
    suffix = "" if kind == "point" else f"_{kind}"
    return d, f"h{horizon}{suffix}"


def _is_keras(model):
    return model.__class__.__module__.startswith("keras") or model.__class__.__module__.startswith("tensorflow")


def save_model(city, horizon, model, kind="point"):
    d, name = _dir(city, horizon, kind)
    if _is_keras(model):
        model.save(os.path.join(d, f"{name}.keras"))
    else:
        joblib.dump(model, os.path.join(d, f"{name}.joblib"))


def load_model(city, horizon, kind="point"):
    d, name = _dir(city, horizon, kind)
    keras_path = os.path.join(d, f"{name}.keras")
    joblib_path = os.path.join(d, f"{name}.joblib")
    if os.path.exists(keras_path):
        import tensorflow as tf
        return KerasRegressorWrapper(tf.keras.models.load_model(keras_path))
    if os.path.exists(joblib_path):
        return joblib.load(joblib_path)
    return None


def model_exists(city, horizon, kind="point"):
    d, name = _dir(city, horizon, kind)
    return os.path.exists(os.path.join(d, f"{name}.keras")) or os.path.exists(os.path.join(d, f"{name}.joblib"))


class KerasRegressorWrapper:
    """Gives a Keras model the same .predict(df) -> 1D array interface as
    the sklearn models, so the rest of the codebase doesn't need to care
    which library trained a given model."""

    def __init__(self, keras_model):
        self.keras_model = keras_model

    def predict(self, X):
        values = X.values if hasattr(X, "values") else X
        return self.keras_model.predict(values, verbose=0).flatten()
