"""Chronological split with an embargo, plus an optional sensor-level holdout.   Owner: Ajay.

The split dates come from the data: about 70% train, 15% validation, 15% test by time.
Rows whose target time would fall past a boundary are dropped, so no training label overlaps
the validation or test window (an embargo of `horizon` steps). Ajay may replace the
fractions with fixed dates once the real data has been explored, and must document them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def split_dates(index: pd.DatetimeIndex, train_frac: float = 0.70, val_frac: float = 0.15):
    """Last timestamp of the training period and of the validation period."""
    n = len(index)
    return index[int(n * train_frac) - 1], index[int(n * (train_frac + val_frac)) - 1]


def temporal_masks(meta: pd.DataFrame, train_end, val_end) -> dict:
    o, t = meta["origin_time"], meta["target_time"]
    return {
        "train": (t <= train_end).to_numpy(),
        "val": ((o > train_end) & (t <= val_end)).to_numpy(),
        "test": (o > val_end).to_numpy(),
    }


def pick_holdout_sensors(sensors, frac: float = 0.15, seed: int = 42) -> np.ndarray:
    """Sensors that are never seen in training (for the sensor-level generalisation test)."""
    sensors = np.unique(sensors)
    rng = np.random.default_rng(seed)
    return rng.choice(sensors, size=max(1, int(len(sensors) * frac)), replace=False)


def split_xy(X, y, meta, train_end, val_end, holdout_sensors=None,
             max_train_rows: int | None = None, seed: int = 42) -> dict:
    """Return the dict that src.evaluate.evaluate_model expects.

    With holdout_sensors, those sensors are removed from train and val, and kept as the
    extra split "unseen" (test period, held-out sensors only).
    """
    masks = temporal_masks(meta, train_end, val_end)
    if holdout_sensors is not None:
        held = meta["sensor"].isin(holdout_sensors).to_numpy()
        masks["unseen"] = masks["test"] & held
        for k in ("train", "val", "test"):
            masks[k] = masks[k] & ~held
    out = {}
    for name, m in masks.items():
        idx = np.flatnonzero(m)
        if name == "train" and max_train_rows and len(idx) > max_train_rows:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_train_rows, replace=False))
        out[f"X_{name}"], out[f"y_{name}"], out[f"meta_{name}"] = X.iloc[idx], y.iloc[idx], meta.iloc[idx]
    return out
