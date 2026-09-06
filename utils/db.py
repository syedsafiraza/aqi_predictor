import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

import config
from utils.features import FEATURE_COLUMNS

RAW_COLUMNS = ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide",
               "sulphur_dioxide", "ozone", "aqi", "aqi_max"]
ENGINEERED_ONLY = [c for c in FEATURE_COLUMNS if c not in RAW_COLUMNS]
ALL_FEATURE_COLS = RAW_COLUMNS + ENGINEERED_ONLY


def _connect():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db():
    conn = _connect()
    cols_sql = ", ".join(f'"{c}" REAL' for c in ALL_FEATURE_COLS)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS features (
            city TEXT NOT NULL,
            date TEXT NOT NULL,
            {cols_sql},
            PRIMARY KEY (city, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            city TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            model_name TEXT,
            rmse REAL, mae REAL, r2 REAL,
            leaderboard_json TEXT,
            trained_at TEXT,
            PRIMARY KEY (city, horizon)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS forecast_log (
            city TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            target_date TEXT NOT NULL,
            predicted_aqi REAL,
            lower_bound REAL,
            upper_bound REAL,
            actual_aqi REAL,
            logged_at TEXT,
            PRIMARY KEY (city, horizon, target_date)
        )
    """)
    conn.commit()
    conn.close()


def upsert_features(city, df):
    init_db()
    conn = _connect()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    cols = ["city", "date"] + ALL_FEATURE_COLS
    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(f'"{c}"' for c in cols)
    sql = f"INSERT OR REPLACE INTO features ({col_names}) VALUES ({placeholders})"

    rows = []
    for _, row in df.iterrows():
        values = [city, row["date"]] + [
            float(row[c]) if c in df.columns and pd.notna(row[c]) else None
            for c in ALL_FEATURE_COLS
        ]
        rows.append(values)

    conn.executemany(sql, rows)
    conn.commit()
    conn.close()


def read_features(city):
    init_db()
    conn = _connect()
    df = pd.read_sql_query("SELECT * FROM features WHERE city = ? ORDER BY date", conn, params=(city,))
    conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def save_model_metadata(city, horizon, model_name, metrics, leaderboard_json):
    init_db()
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO model_metadata
            (city, horizon, model_name, rmse, mae, r2, leaderboard_json, trained_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (city, horizon, model_name, metrics["rmse"], metrics["mae"], metrics["r2"],
          leaderboard_json, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def read_model_metadata(city, horizon):
    init_db()
    conn = _connect()
    row = conn.execute(
        "SELECT model_name, rmse, mae, r2, leaderboard_json, trained_at FROM model_metadata "
        "WHERE city = ? AND horizon = ?", (city, horizon)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    keys = ["model_name", "rmse", "mae", "r2", "leaderboard_json", "trained_at"]
    return dict(zip(keys, row))


def log_forecast(city, horizon, target_date, predicted_aqi, lower_bound, upper_bound):
    init_db()
    conn = _connect()
    conn.execute("""
        INSERT OR REPLACE INTO forecast_log
            (city, horizon, target_date, predicted_aqi, lower_bound, upper_bound, actual_aqi, logged_at)
        VALUES (?, ?, ?, ?, ?, ?,
                COALESCE((SELECT actual_aqi FROM forecast_log
                          WHERE city=? AND horizon=? AND target_date=?), NULL),
                ?)
    """, (city, horizon, target_date, predicted_aqi, lower_bound, upper_bound,
          city, horizon, target_date, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def backfill_actuals():
    init_db()
    conn = _connect()
    conn.execute("""
        UPDATE forecast_log
        SET actual_aqi = (
            SELECT aqi FROM features
            WHERE features.city = forecast_log.city
              AND features.date = forecast_log.target_date
        )
        WHERE actual_aqi IS NULL
          AND EXISTS (
              SELECT 1 FROM features
              WHERE features.city = forecast_log.city
                AND features.date = forecast_log.target_date
          )
    """)
    conn.commit()
    conn.close()


def read_forecast_log(city):
    init_db()
    conn = _connect()
    df = pd.read_sql_query("SELECT * FROM forecast_log WHERE city = ? ORDER BY target_date", conn, params=(city,))
    conn.close()
    return df
