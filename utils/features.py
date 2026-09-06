import pandas as pd

import config

FEATURE_COLUMNS = [
    "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "surface_pressure", "day", "month", "day_of_week", "is_weekend",
    "is_smog_season", "aqi_change_rate", "aqi_lag1", "aqi_lag2", "aqi_lag3",
    "pm2_5_lag1", "aqi_roll_mean3", "aqi_roll_mean7", "aqi_momentum",
    "wind_dispersion",
]


def add_time_features(df, province):
    df = df.copy()
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    in_season_month = df["month"].isin(config.SMOG_SEASON_MONTHS)
    df["is_smog_season"] = (in_season_month & (province in config.SMOG_SEASON_PROVINCES)).astype(int)
    return df


def add_derived_features(df):
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["aqi_change_rate"] = df["aqi"].diff()
    for lag in (1, 2, 3):
        df[f"aqi_lag{lag}"] = df["aqi"].shift(lag)
    df["pm2_5_lag1"] = df["pm2_5"].shift(1)
    df["aqi_roll_mean3"] = df["aqi"].shift(1).rolling(3).mean()
    df["aqi_roll_mean7"] = df["aqi"].shift(1).rolling(7).mean()
    df["aqi_momentum"] = df["aqi_roll_mean3"] - df["aqi_roll_mean7"]
    df["wind_dispersion"] = df["wind_speed_10m"] * df["pm2_5"]
    return df


def build_feature_table(daily_df, province):
    df = add_time_features(daily_df, province)
    df = add_derived_features(df)
    return df


def make_supervised(df, horizons=config.FORECAST_HORIZONS):
    df = df.copy()
    for h in horizons:
        df[f"target_h{h}"] = df["aqi"].shift(-h)
    return df
