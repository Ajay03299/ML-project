"""Streamlit UI wired to the real pipeline.   Owner: Bhavesh (UI), Anirudh (model loading).

Run from the repo root:   streamlit run app/streamlit_app.py

The user picks a sensor, a date and a time, and sees the model's forecast for one hour
ahead next to the recent traffic. The app rebuilds the same features with the same
`build_dataset` code the experiments use, and loads the models saved in models/, so the
UI and the benchmark cannot drift apart.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_panel  # noqa: E402
from src.features import build_dataset  # noqa: E402

MODELS_DIR = ROOT / "models"
st.set_page_config(page_title="Traffic Flow Forecaster", layout="wide")
st.title("Traffic flow forecaster")

meta_path = MODELS_DIR / "meta.json"
if not meta_path.exists():
    st.error("No trained models found. Run `python -m experiments.run_benchmark` first.")
    st.stop()
info = json.loads(meta_path.read_text())


@st.cache_resource
def load_everything(source, n_sensors, horizon):
    panel = load_panel(source, n_sensors=n_sensors)
    X, y, meta = build_dataset(panel, horizon=horizon)
    models = {p.stem: joblib.load(p) for p in sorted(MODELS_DIR.glob("*.joblib"))}
    X = X[info["columns"]]
    return panel, X, y, meta, models


panel, X, y, meta, models = load_everything(info["source"], info["n_sensors"], info["horizon"])
step = info["step_min"]
horizon_min = info["horizon"] * step   # minutes ahead

tab_fc, tab_board = st.tabs(["Forecast", "Model leaderboard"])

with tab_fc:
    c = st.columns(4)
    model_name = c[0].selectbox("Model", [m for m in models if m != "Persistence"] + ["Persistence"])
    sensor = c[1].selectbox("Sensor", list(panel.sensors))
    last_day = meta["origin_time"].max().normalize() - pd.Timedelta(days=3)
    date = c[2].date_input("Date", last_day.date(),
                           min_value=meta["origin_time"].min().date(), max_value=meta["origin_time"].max().date())
    time_of_day = c[3].time_input("Time now", pd.Timestamp("2000-01-01 08:00").time(), step=step * 60)
    origin = pd.Timestamp.combine(date, time_of_day)

    sel = (meta["sensor"] == sensor).to_numpy()
    day_rows = sel & (meta["origin_time"].dt.date == date).to_numpy()
    cols = info["columns"]
    model = models[model_name]

    this = sel & (meta["origin_time"] == origin).to_numpy()
    if not this.any():
        st.warning("No forecast possible at this time (history too short, or the sensor was faulty).")
    else:
        pred = float(model.predict(X.loc[this, cols])[0])
        actual = float(y[this].iloc[0])
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Forecast for {origin + pd.Timedelta(minutes=horizon_min):%H:%M}", f"{pred:,.0f} {info.get('unit', '')}")
        m2.metric("What actually happened", f"{actual:,.0f}", delta=f"{pred - actual:+.0f} error", delta_color="off")
        j = list(panel.sensors).index(sensor)
        now = panel.index.get_loc(origin)
        recent = pd.DataFrame({"flow": panel.flow[max(0, now - 96): now + 1, j]},
                              index=panel.index[max(0, now - 96): now + 1])
        m3.metric("Flow right now", f"{recent['flow'].iloc[-1]:,.0f}")

        st.subheader("Recent traffic (last 24 h) at this sensor")
        st.line_chart(recent)

    if day_rows.any():
        st.subheader(f"Forecast vs actual, whole day, sensor {sensor} ({horizon_min} min ahead)")
        d = pd.DataFrame({
            "actual": y[day_rows].to_numpy(),
            "forecast": model.predict(X.loc[day_rows, cols]),
        }, index=meta.loc[day_rows, "target_time"])
        st.line_chart(d)

with tab_board:
    results = ROOT / "results" / "benchmark.csv"
    if results.exists():
        st.dataframe(pd.read_csv(results).sort_values("test_MAE"))
    else:
        st.info("results/benchmark.csv not found yet.")
# TODO(Bhavesh): sensor map, compare-models view, CSV export, SHAP explanation of one forecast (extra-effort marks).
