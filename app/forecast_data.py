"""Forecast presentation using exactly the training pipeline and saved model schema."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.artifacts import dataset_hash, file_hash
from src.features import build_dataset


def validate_panel(panel, info):
    if dataset_hash(panel) != info["dataset_id"]:
        raise ValueError("Traffic data changed since training. Re-run the benchmark before serving forecasts.")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "src"
    for name, expected in info["pipeline_hashes"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"{name} changed since training. Re-run the benchmark.")


def serving_dataset(panel, info):
    validate_panel(panel, info)
    X, y, meta = build_dataset(panel, horizon=info["horizon"], train_end=info["train_end"])
    missing = set(info["columns"]) - set(X.columns)
    if missing:
        raise ValueError(f"Saved model features are unavailable: {sorted(missing)}")
    # Use held-out origins by default. Train-period aggregate features are not an
    # honest retrospective forecast for origins inside the training window.
    keep = meta["origin_time"] > pd.Timestamp(info["val_end"])
    return X.loc[keep, info["columns"]], y.loc[keep], meta.loc[keep]


def day_forecasts(X, y, meta, models, sensor, date, names):
    """Selected TARGET day, including previous-day origins near midnight."""
    keep = (meta["sensor"] == sensor) & (meta["target_time"].dt.date == date)
    selected = meta.loc[keep].copy()
    selected["actual"] = y.loc[keep].to_numpy()
    for name in names:
        selected[name] = models[name].predict(X.loc[keep]) if keep.any() else []
    return selected.sort_values("target_time")


def sensor_map(panel, train_end):
    """Training-period mean flow sets marker area; never invent coordinates."""
    if panel.static is None or not {"latitude", "longitude"} <= set(panel.static):
        return pd.DataFrame()
    result = panel.static.reindex(panel.sensors)[["latitude", "longitude"]].copy()
    history = pd.DataFrame(panel.flow[panel.index <= pd.Timestamp(train_end)])
    result["mean_flow"] = history.mean().to_numpy()
    result = result.dropna()
    valid = result["latitude"].between(-90, 90) & result["longitude"].between(-180, 180)
    result = result.loc[valid]
    if len(result):
        maximum = max(float(result["mean_flow"].max()), 1.0)
        result["marker_size"] = 30 + 170 * np.sqrt(result["mean_flow"].clip(lower=0) / maximum)
    return result
