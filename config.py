import os

CITIES = {
    "Karachi":   {"lat": 24.8607, "lon": 67.0011, "province": "Sindh"},
    "Lahore":    {"lat": 31.5497, "lon": 74.3436, "province": "Punjab"},
    "Islamabad": {"lat": 33.6844, "lon": 73.0479, "province": "Federal"},
    "Multan":    {"lat": 30.1575, "lon": 71.5249, "province": "Punjab"},
    "Quetta":    {"lat": 30.1798, "lon": 66.9750, "province": "Balochistan"},
}

FORECAST_HORIZONS = [1, 2, 3]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "aqi_store.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")

HAZARDOUS_AQI_THRESHOLD = 200

SMOG_SEASON_MONTHS = {10, 11, 12, 1, 2}
SMOG_SEASON_PROVINCES = {"Punjab"}

ALERT_WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")
