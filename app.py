import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from accuracy_tracker import compute_rolling_accuracy
from utils import db
from utils.aqi_utils import aqi_category, is_hazardous, is_smog_season
from utils.features import FEATURE_COLUMNS, make_supervised
from utils.model_registry import load_model
from utils.scoring import clean_air_score, score_label

st.set_page_config(page_title="10Pearls AQI Predictor", page_icon="🌫️", layout="wide")

CUSTOM_CSS = """
<style>
    .main > div {padding-top: 1.2rem;}
    .block-container {padding-top: 1.5rem;}
    h1, h2, h3 {font-family: 'Segoe UI', sans-serif;}
    .stMetric {background: rgba(255,255,255,0.03); border-radius: 10px; padding: 8px;}
    .city-card {
        border-radius: 16px; padding: 18px 16px; text-align: left;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15); transition: transform .15s ease;
    }
    .city-card:hover {transform: translateY(-3px);}
    .badge {
        display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:0.75em; font-weight:600; color:white; margin-left:6px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=1800)
def load_city_features(city):
    df = db.read_features(city)
    return make_supervised(df) if not df.empty else df


@st.cache_resource
def load_city_models(city):
    models = {}
    for h in config.FORECAST_HORIZONS:
        point = load_model(city, h, "point")
        if point is None:
            continue
        models[h] = {
            "point": point,
            "lower": load_model(city, h, "lower"),
            "upper": load_model(city, h, "upper"),
            "meta": db.read_model_metadata(city, h),
        }
    return models


def forecast_next_days(df, models):
    if df.empty or not models:
        return []
    row = df.dropna(subset=FEATURE_COLUMNS).iloc[[-1]]
    if row.empty:
        return []
    X = row[FEATURE_COLUMNS]
    results = []
    for h in config.FORECAST_HORIZONS:
        if h not in models:
            continue
        m = models[h]
        pred = max(0, float(m["point"].predict(X)[0]))
        lo = max(0, float(m["lower"].predict(X)[0])) if m["lower"] else None
        hi = float(m["upper"].predict(X)[0]) if m["upper"] else None
        results.append({
            "horizon": h, "date": row["date"].iloc[0] + pd.Timedelta(days=h),
            "predicted_aqi": pred, "lower": lo, "upper": hi,
            "model": m["meta"]["model_name"] if m["meta"] else "n/a",
            "rmse": m["meta"]["rmse"] if m["meta"] else None,
        })
    return results


def aqi_gauge(value, title):
    label, color, _ = aqi_category(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 400]}, "bar": {"color": color},
            "steps": [
                {"range": [0, 50], "color": "#00e40033"},
                {"range": [50, 100], "color": "#ffff0033"},
                {"range": [100, 150], "color": "#ff7e0033"},
                {"range": [150, 200], "color": "#ff000033"},
                {"range": [200, 300], "color": "#8f3f9733"},
                {"range": [300, 400], "color": "#7e002333"},
            ],
        },
    ))
    fig.update_layout(height=200, margin=dict(t=40, b=10, l=20, r=20))
    return fig


st.sidebar.title("🌫️ 10Pearls AQI Predictor")
st.sidebar.caption("Karachi · Lahore · Islamabad · Multan · Quetta")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data:** Open-Meteo (live, no API key)\n\n"
    "**Models:** Ridge / Extra Trees / Deep Learning MLP + quantile intervals\n\n"
    "**Explainability:** SHAP\n\n"
    "**API:** FastAPI (`api.py`) for programmatic access"
)
st.sidebar.markdown("---")
selected_cities = st.sidebar.multiselect(
    "Cities to display", list(config.CITIES.keys()), default=list(config.CITIES.keys())
)
view_mode = st.sidebar.radio("View", ["Overview", "City Deep-Dive", "Map"], index=0)

st.title("Pakistan AQI Forecast Dashboard")
st.caption("3-day AQI forecasts with uncertainty ranges, live model leaderboard, and forecast-accuracy tracking.")

city_data = {}
for city in selected_cities:
    df = load_city_features(city)
    models = load_city_models(city)
    forecasts = forecast_next_days(df, models)
    city_data[city] = {"df": df, "models": models, "forecasts": forecasts}

hazard_msgs, smog_msgs = [], []
for city, d in city_data.items():
    if d["df"].empty:
        continue
    latest = d["df"].iloc[-1]
    if is_hazardous(latest["aqi"]):
        hazard_msgs.append(f"**{city}**: {latest['aqi']:.0f}")
    for f in d["forecasts"]:
        if is_hazardous(f["predicted_aqi"]):
            hazard_msgs.append(f"**{city}** (+{f['horizon']}d): {f['predicted_aqi']:.0f}")
    if is_smog_season(int(latest["month"]), config.CITIES[city]["province"]):
        smog_msgs.append(city)

if hazard_msgs:
    st.error("🚨 **Hazardous AQI Alert** — " + " | ".join(hazard_msgs))
if smog_msgs:
    st.warning(f"🍂 **Smog season active** in: {', '.join(smog_msgs)} — crop-burning and winter inversions typically raise AQI Oct–Feb.")

if view_mode == "Map":
    st.subheader("Live AQI Map")
    map_rows = []
    for city, d in city_data.items():
        if d["df"].empty:
            continue
        latest = d["df"].iloc[-1]
        map_rows.append({"city": city, "lat": config.CITIES[city]["lat"], "lon": config.CITIES[city]["lon"],
                          "aqi": latest["aqi"], "province": config.CITIES[city]["province"]})
    if map_rows:
        map_df = pd.DataFrame(map_rows)
            # ...existing code...
        fig = px.scatter_map(
            map_df,
            lat="lat",
            lon="lon",
            color="aqi",
            size="aqi",
            hover_name="city",
            zoom=4,
            center={"lat": 31.5, "lon": 73.5},
            map_style="carto-positron",
        )
        # ...existing code...
        fig.update_layout(map_style="carto-positron", margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet — run the backfill and feature pipeline first.")

elif view_mode == "Overview":
    st.subheader("City comparison — right now")
    cols = st.columns(len(selected_cities) if selected_cities else 1)
    current_aqis = {}
    for i, city in enumerate(selected_cities):
        df = city_data[city]["df"]
        with cols[i]:
            if df.empty:
                st.warning(f"**{city}**\nNo data yet.")
                continue
            latest = df.iloc[-1]
            label, color, _ = aqi_category(latest["aqi"])
            score = clean_air_score(latest["aqi"], latest.get("aqi_momentum", 0))
            current_aqis[city] = latest["aqi"]
            st.markdown(
                f"""
                <div class="city-card" style="background:linear-gradient(135deg,{color}22,{color}08);border:1px solid {color}55;">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <b style="font-size:1.1em;">{city}</b>
                    <span style="font-size:0.75em;color:gray;">{config.CITIES[city]['province']}</span>
                  </div>
                  <h1 style="margin:6px 0;color:{color};">{latest['aqi']:.0f}</h1>
                  <span class="badge" style="background:{color};">{label}</span>
                  <div style="margin-top:8px;font-size:0.85em;">Clean Air Score: <b>{score}</b> · {score_label(score)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if current_aqis:
        worst = max(current_aqis, key=current_aqis.get)
        best = min(current_aqis, key=current_aqis.get)
        st.info(f"🥇 Cleanest right now: **{best}** ({current_aqis[best]:.0f})   |   "
                f"⚠️ Most polluted right now: **{worst}** ({current_aqis[worst]:.0f})")

        st.markdown("###")
        fig = px.bar(
            x=list(current_aqis.keys()), y=list(current_aqis.values()),
            color=list(current_aqis.values()), color_continuous_scale=["#00e400", "#ff7e00", "#7e0023"],
            labels={"x": "City", "y": "Current AQI"}, title="Current AQI across all cities",
        )
        fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

else:
    if not selected_cities:
        st.info("Select at least one city from the sidebar.")
    tabs = st.tabs(selected_cities)
    for tab, city in zip(tabs, selected_cities):
        with tab:
            df = city_data[city]["df"]
            forecasts = city_data[city]["forecasts"]
            if df.empty:
                st.warning("No data for this city yet.")
                continue

            latest = df.iloc[-1]
            g1, g2 = st.columns([1, 2])
            with g1:
                st.plotly_chart(aqi_gauge(latest["aqi"], f"{city} — Today"), use_container_width=True)
            with g2:
                st.subheader("3-day forecast")
                hist = df.tail(14)[["date", "aqi"]]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist["date"], y=hist["aqi"], mode="lines+markers",
                                          name="Historical", line=dict(color="#1f77b4")))
                if forecasts:
                    fc_dates = [f["date"] for f in forecasts]
                    fc_preds = [f["predicted_aqi"] for f in forecasts]
                    fc_lo = [f["lower"] for f in forecasts]
                    fc_hi = [f["upper"] for f in forecasts]
                    bx = [hist["date"].iloc[-1]] + fc_dates
                    by = [hist["aqi"].iloc[-1]] + fc_preds
                    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines+markers", name="Forecast",
                                              line=dict(color="#ff7e00", dash="dash")))
                    if all(v is not None for v in fc_lo + fc_hi):
                        fig.add_trace(go.Scatter(x=fc_dates + fc_dates[::-1], y=fc_hi + fc_lo[::-1],
                                                  fill="toself", fillcolor="rgba(255,126,0,0.15)",
                                                  line=dict(width=0), name="80% interval"))
                else:
                    st.warning("No trained model yet — run training_pipeline.py.")
                fig.update_layout(height=320, margin=dict(t=10), xaxis_title="", yaxis_title="US AQI")
                st.plotly_chart(fig, use_container_width=True)

            if forecasts:
                fc_cols = st.columns(len(forecasts))
                for c, f in zip(fc_cols, forecasts):
                    label, color, _ = aqi_category(f["predicted_aqi"])
                    with c:
                        st.metric(f"Day +{f['horizon']} ({f['date'].strftime('%b %d')})",
                                  f"{f['predicted_aqi']:.0f} AQI", label, delta_color="off")
                        if f["lower"] is not None:
                            st.caption(f"Range: {f['lower']:.0f}–{f['upper']:.0f} · {f['model']}")
                st.download_button("⬇️ Download forecast (CSV)",
                                    pd.DataFrame(forecasts).to_csv(index=False).encode(),
                                    f"{city}_forecast.csv", "text/csv")

            st.subheader("Forecast accuracy (self-monitoring)")
            acc_cols = st.columns(len(config.FORECAST_HORIZONS))
            for acol, h in zip(acc_cols, config.FORECAST_HORIZONS):
                mae, n = compute_rolling_accuracy(city, h)
                with acol:
                    if n:
                        st.metric(f"Day +{h} rolling MAE", f"{mae:.1f} AQI pts", f"last {n} checked")
                    else:
                        st.caption(f"Day +{h}: no resolved forecasts yet.")

            st.subheader("Exploratory data analysis")
            e1, e2 = st.columns(2)
            with e1:
                st.plotly_chart(px.line(df, x="date", y="aqi", title="Full AQI history")
                                 .update_layout(height=290, margin=dict(t=40)), use_container_width=True)
            with e2:
                st.plotly_chart(px.box(df, x="day_of_week", y="aqi", title="AQI by day of week")
                                 .update_layout(height=290, margin=dict(t=40)), use_container_width=True)
            st.plotly_chart(
                px.line(df.tail(60), x="date", y=["pm2_5", "pm10", "nitrogen_dioxide", "ozone"],
                        title="Pollutant breakdown (last 60 days)").update_layout(height=300, margin=dict(t=40)),
                use_container_width=True,
            )

            meta = city_data[city]["models"].get(1, {}).get("meta")
            shap_path = None
            if meta and meta.get("leaderboard_json"):
                import json
                shap_path = json.loads(meta["leaderboard_json"]).get("shap_plot")
            if shap_path:
                import os
                if os.path.exists(shap_path):
                    st.subheader("What's driving the +1 day forecast (SHAP)")
                    st.image(shap_path, use_container_width=False, width=500)

            with st.expander("Model leaderboard"):
                rows = []
                for h in config.FORECAST_HORIZONS:
                    m = city_data[city]["models"].get(h, {}).get("meta")
                    if m:
                        rows.append({"horizon": f"+{h}d", "selected_model": m["model_name"],
                                     "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"]})
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.markdown("---")
st.caption("10Pearls AQI Predictor — Open-Meteo · SQLite · scikit-learn · TensorFlow · SHAP · Streamlit · FastAPI")
