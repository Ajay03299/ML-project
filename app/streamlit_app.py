"""Traffic forecast explorer. Run: python -m streamlit run app/streamlit_app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.forecast_data import day_forecasts, sensor_map, serving_dataset
from src.artifacts import file_hash
from src.data import load_panel
from src.models import BASELINES

MODELS_DIR = ROOT / "models"
st.set_page_config(page_title="Traffic Flow Forecaster", layout="wide")
st.markdown("<style>.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}</style>", unsafe_allow_html=True)
st.title("Traffic flow forecaster")
st.caption("Inspect a held-out forecast, recent sensor readings and a day of model comparisons.")

meta_path = MODELS_DIR / "meta.json"
if not meta_path.exists():
    st.info("No trained models yet. Run python -m experiments.run_benchmark --quick --source synthetic "
            "--only Persistence Ridge DecisionTree HistGradientBoosting to try the demo.")
    st.stop()
try:
    info = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {"run_id", "dataset_id", "columns", "train_end", "val_end", "model_runs", "pipeline_hashes"}
    if not required <= set(info):
        raise ValueError("These saved models predate provenance checks. Re-run experiments.run_benchmark.")
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.stop()


@st.cache_resource
def load_everything(metadata_json):
    metadata = json.loads(metadata_json)
    panel = load_panel(metadata["source"], n_sensors=metadata["n_sensors"])
    X, y, meta = serving_dataset(panel, metadata)
    for name, record in metadata["model_runs"].items():
        if record["run_id"] != metadata["run_id"]:
            raise ValueError(f"Saved {name} differs from its training manifest. Re-run benchmark.")
    return panel, X, y, meta, list(metadata["model_runs"])


@st.cache_resource(max_entries=4)
def load_model(name, expected_hash, file_version):
    path = MODELS_DIR / f"{name}.joblib"
    if file_hash(path) != expected_hash:
        raise ValueError(f"Saved {name} differs from its training manifest. Re-run benchmark.")
    return joblib.load(path)


def model_for(name):
    try:
        path = MODELS_DIR / f"{name}.joblib"
        version = (path.stat().st_mtime_ns, path.stat().st_size)
        return load_model(name, info["model_runs"][name]["sha256"], version)
    except (ValueError, FileNotFoundError, ImportError, OSError) as exc:
        st.error(f"Cannot load {name}: {exc}")
        st.stop()


try:
    panel, X, y, meta, model_names = load_everything(json.dumps(info, sort_keys=True))
except (ValueError, FileNotFoundError, KeyError, ImportError, OSError) as exc:
    st.error(f"Cannot load this forecast setup: {exc}")
    st.stop()
if not model_names or meta.empty:
    st.warning("No saved models or usable held-out forecasts. Re-run benchmark or check missing observations.")
    st.stop()

unit = info["unit"]
horizon_min = info["horizon_min"]
if info["source"] == "synthetic":
    st.warning("SYNTHETIC DEMO - tests the software only. These forecasts are not Madrid research results.")
st.info(f"{horizon_min}-minute forecast | unit: {unit} | {len(panel.sensors)} sensors | source: {info['source']}")
st.caption("Historical replay on the test period. Each forecast uses readings available at its own origin; "
           "the day chart is a sequence of rolling forecasts, not a whole-day forecast made at midnight.")

forecast_tab, map_tab, board_tab = st.tabs(["Forecast & compare", "Sensor map", "Frozen benchmark"])
with forecast_tab:
    show_baselines = st.checkbox("Show baselines", value=False)
    names = [m for m in model_names if show_baselines or m not in BASELINES]
    if not names:
        st.info("Only baselines are saved. Enable Show baselines or train a tree/boosting model.")
    else:
        controls = st.columns(4)
        previous_model = st.session_state.get("model", names[0])
        initial_model = names.index(previous_model) if previous_model in names else 0
        model_name = controls[0].selectbox("Model", names, index=initial_model, key="model")
        sensor = controls[1].selectbox("Sensor", sorted(meta["sensor"].unique().tolist()), key="sensor")
        sensor_meta = meta[meta["sensor"] == sensor]
        dates = sorted(sensor_meta["target_time"].dt.date.unique())
        date = controls[2].date_input("Forecast target date", dates[-2] if len(dates) > 1 else dates[-1],
                                      min_value=dates[0], max_value=dates[-1], key="date")
        target_clock = controls[3].time_input("Forecast target time", pd.Timestamp("2000-01-01 08:00").time(),
                                               step=info["step_min"] * 60, key="time")
        if show_baselines:
            st.caption("Baselines are included in the model selector.")
        target = pd.Timestamp.combine(date, target_clock)
        origin = target - pd.Timedelta(minutes=horizon_min)
        st.caption(f"Last observed: {origin:%Y-%m-%d %H:%M} | forecast target: {target:%Y-%m-%d %H:%M}. "
                   "Timestamps use the dataset clock; no timezone conversion is assumed.")
        selected = (meta["sensor"] == sensor) & (meta["target_time"] == target)
        if not selected.any():
            st.warning("No valid forecast at this target time: missing/faulty readings, insufficient history, "
                       "or an origin outside the held-out period. Choose another time.")
        else:
            prediction = float(model_for(model_name).predict(X.loc[selected])[0])
            actual = float(y.loc[selected].iloc[0])
            a, b, c = st.columns(3)
            a.metric(f"Forecast ({unit})", f"{prediction:,.1f}")
            b.metric(f"Observed at target ({unit})", f"{actual:,.1f}")
            c.metric(f"Absolute error ({unit})", f"{abs(prediction - actual):,.1f}")
            if prediction < 0:
                st.warning("This model predicted negative flow. The raw output is shown, as in the benchmark.")
            pos = int(np.flatnonzero(panel.sensors == sensor)[0])
            recent_mask = (panel.index >= origin - pd.Timedelta(hours=24)) & (panel.index <= origin)
            recent = pd.DataFrame({f"Observed flow ({unit})": panel.flow[recent_mask, pos]},
                                  index=panel.index[recent_mask])
            recent.index.name = "Observation time"
            st.subheader("Recent traffic - last 24 hours before forecast origin")
            st.line_chart(recent)
            with st.expander("Inputs behind this forecast"):
                st.caption("These are the same feature columns and training-period statistics used by the saved model.")
                st.dataframe(X.loc[selected].T.rename(columns={X.loc[selected].index[0]: "value"}))
        comparison = st.multiselect("Compare models on the target day", names,
                                    default=[model_name], key=f"comparison_{show_baselines}_{model_name}")
        comparison = list(dict.fromkeys([model_name] + comparison))
        day = day_forecasts(X, y, meta, {name: model_for(name) for name in comparison}, sensor, date, comparison)
        if day.empty:
            st.info("No usable forecasts on this target day. Missing intervals are not filled.")
        else:
            mae = float((day[model_name] - day["actual"]).abs().mean())
            st.subheader(f"Target day {date} - sensor {sensor}")
            st.metric(f"{model_name} day MAE ({unit})", f"{mae:,.2f}")
            st.caption(f"{len(day)} usable intervals out of {1440 // info['step_min']} expected. "
                       "Error uses only the displayed forecasts; gaps are not zero traffic.")
            # Reindex the chart only, so missing time intervals stay visibly missing.
            grid = pd.date_range(pd.Timestamp(date), periods=1440 // info["step_min"], freq=f"{info['step_min']}min")
            chart = day.set_index("target_time")[["actual"] + comparison].reindex(grid)
            chart.index.name = f"Target time | flow ({unit})"
            st.line_chart(chart)
            metrics = [{"model": name, "MAE": float((day[name] - day["actual"]).abs().mean()),
                        "RMSE": float(np.sqrt(((day[name] - day["actual"]) ** 2).mean()))} for name in comparison]
            st.dataframe(pd.DataFrame(metrics), hide_index=True)
            export = day.copy()
            export["unit"] = unit
            export["horizon_min"] = horizon_min
            export["source"] = info["source"]
            export["run_id"] = info["run_id"]
            st.download_button("Download day's forecasts (CSV)", export.to_csv(index=False).encode("utf-8"),
                               file_name=f"traffic_{info['source']}_{sensor}_{date}_h{horizon_min}.csv", mime="text/csv")
with map_tab:
    points = sensor_map(panel, info["train_end"])
    if points.empty:
        st.info("No valid sensor coordinates in this dataset. No map positions have been assumed.")
    else:
        st.caption(f"Larger markers indicate higher mean training-period flow ({unit}). "
                   "Coordinates and values come from this dataset.")
        st.map(points, latitude="latitude", longitude="longitude", size="marker_size")
        st.caption("If map tiles are unavailable, use the coordinate and flow table below.")
        st.dataframe(points.rename_axis("sensor").reset_index(), hide_index=True)
with board_tab:
    results = ROOT / "results/benchmark.csv"
    if results.exists():
        board = pd.read_csv(results)
        if "run_id" in board:
            board = board[(board["run_id"] == info["run_id"]) & (board["tag"] == "all")]
            if not board.empty:
                st.caption(f"Same dataset, features, training rows and {horizon_min}-minute horizon. MAE/RMSE in {unit}.")
                st.dataframe(board.sort_values("test_MAE"), hide_index=True)
            else:
                st.info("No benchmark rows match the currently saved models.")
        else:
            st.info("The benchmark needs to be regenerated with provenance metadata.")
    else:
        st.info("No benchmark CSV available.")
    with st.expander("Reproducibility details"):
        if info["source"] == "madrid" and panel.weather is None:
            st.caption("Weather is excluded because the shared panel does not establish past-only filling. "
                       "Every displayed model uses the same traffic, calendar and road feature set.")
        st.json({key: info[key] for key in ("source", "run_id", "train_end", "val_end", "n_train", "environment")})
