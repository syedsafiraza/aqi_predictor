import pandas as pd
import requests

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
FORECAST_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

POLLUTANT_VARS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
                   "sulphur_dioxide", "ozone", "us_aqi"]
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m",
                "wind_speed_10m", "surface_pressure"]


def _get(url, params, timeout=30):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_recent_air_quality(lat, lon, past_days=5, forecast_days=4):
    data = _get(AIR_QUALITY_URL, {
        "latitude": lat, "longitude": lon, "hourly": ",".join(POLLUTANT_VARS),
        "past_days": past_days, "forecast_days": forecast_days, "timezone": "auto",
    })
    return pd.DataFrame(data["hourly"])


def fetch_recent_weather(lat, lon, past_days=5, forecast_days=4):
    data = _get(FORECAST_WEATHER_URL, {
        "latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days, "forecast_days": forecast_days, "timezone": "auto",
    })
    return pd.DataFrame(data["hourly"])


def fetch_historical_air_quality(lat, lon, start_date, end_date):
    data = _get(AIR_QUALITY_URL, {
        "latitude": lat, "longitude": lon, "hourly": ",".join(POLLUTANT_VARS),
        "start_date": start_date, "end_date": end_date, "timezone": "auto",
    })
    return pd.DataFrame(data["hourly"])


def fetch_historical_weather(lat, lon, start_date, end_date):
    data = _get(ARCHIVE_WEATHER_URL, {
        "latitude": lat, "longitude": lon, "hourly": ",".join(WEATHER_VARS),
        "start_date": start_date, "end_date": end_date, "timezone": "auto",
    })
    return pd.DataFrame(data["hourly"])


def merge_and_aggregate_daily(air_df, weather_df):
    air_df = air_df.copy()
    weather_df = weather_df.copy()
    air_df["time"] = pd.to_datetime(air_df["time"])
    weather_df["time"] = pd.to_datetime(weather_df["time"])

    merged = pd.merge(air_df, weather_df, on="time", how="inner")
    merged["date"] = merged["time"].dt.date

    agg = {v: "mean" for v in POLLUTANT_VARS + WEATHER_VARS}
    daily = merged.groupby("date").agg(agg).reset_index()
    daily["aqi_max"] = merged.groupby("date")["us_aqi"].max().values
    daily = daily.rename(columns={"us_aqi": "aqi"})
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)
