import argparse
import logging
from datetime import date, timedelta

import config
from utils.data_fetch import fetch_historical_air_quality, fetch_historical_weather, merge_and_aggregate_daily
from utils.features import build_feature_table
from utils import db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("backfill")


def backfill_city(city, coords, start_date, end_date):
    air_df = fetch_historical_air_quality(coords["lat"], coords["lon"], start_date, end_date)
    weather_df = fetch_historical_weather(coords["lat"], coords["lon"], start_date, end_date)
    daily = merge_and_aggregate_daily(air_df, weather_df)
    featured = build_feature_table(daily, coords["province"])
    db.upsert_features(city, featured)
    log.info(f"{city}: backfilled {len(featured)} rows.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=300)
    args = parser.parse_args()

    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=args.days)

    for city, coords in config.CITIES.items():
        backfill_city(city, coords, start.isoformat(), end.isoformat())

    log.info("Backfill complete for all 5 cities.")


if __name__ == "__main__":
    main()
