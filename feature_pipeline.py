import logging
import sys

import config
from utils.data_fetch import fetch_recent_air_quality, fetch_recent_weather, merge_and_aggregate_daily
from utils.features import build_feature_table
from utils import db
from utils.aqi_utils import is_hazardous, is_smog_season, aqi_category
from utils.alerts import send_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("feature_pipeline")


def run_for_city(city, coords):
    air_df = fetch_recent_air_quality(coords["lat"], coords["lon"])
    weather_df = fetch_recent_weather(coords["lat"], coords["lon"])
    daily = merge_and_aggregate_daily(air_df, weather_df)
    featured = build_feature_table(daily, coords["province"])

    db.upsert_features(city, featured)
    log.info(f"{city}: upserted {len(featured)} daily rows.")

    latest = featured.iloc[-1]
    label, _, advice = aqi_category(latest["aqi"])
    log.info(f"{city}: latest AQI={latest['aqi']:.0f} ({label})")

    if is_hazardous(latest["aqi"]):
        send_alert(f"Hazardous AQI in {city}: {latest['aqi']:.0f} ({label}). {advice}")

    if is_smog_season(int(latest["month"]), coords["province"]):
        log.info(f"{city}: currently in smog season window.")


def main():
    failures = []
    for city, coords in config.CITIES.items():
        try:
            run_for_city(city, coords)
        except Exception as e:
            log.error(f"{city}: pipeline failed — {e}")
            failures.append(city)
    if failures:
        log.error(f"Failures: {failures}")
        sys.exit(1)
    log.info("Feature pipeline done for all 5 cities.")


if __name__ == "__main__":
    main()
