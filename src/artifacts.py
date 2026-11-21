"""Identify the data and feature setup behind an experiment or saved UI model."""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from src import config
from src.split import split_dates


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_hash(panel):
    """Includes values, time grid, sensor order, weather and static attributes."""
    digest = hashlib.sha256()
    for values in (panel.flow, panel.index.asi8, panel.sensors, panel.occupancy, panel.speed):
        if values is None:
            digest.update(b"None")
            continue
        arr = np.ascontiguousarray(values)
        digest.update(str((arr.shape, arr.dtype)).encode())
        digest.update(arr.tobytes())
    for frame in (panel.weather, panel.static):
        if frame is not None:
            digest.update(frame.to_csv().encode())
    digest.update(str(panel.holidays).encode())
    digest.update(str((panel.step_min, panel.unit)).encode())
    return digest.hexdigest()


def experiment_info(panel, splits, args):
    train_end, val_end = split_dates(panel.index)
    root = Path(__file__).resolve().parent
    info = {
        "source": args.source, "horizon": args.horizon,
        "horizon_min": args.horizon * panel.step_min, "step_min": panel.step_min,
        "unit": panel.unit, "n_sensors": args.n_sensors,
        "sensor_ids": [int(s) for s in panel.sensors],
        "columns": list(splits["X_train"].columns),
        "train_end": str(train_end), "val_end": str(val_end),
        "n_train": len(splits["X_train"]), "seed": config.RANDOM_STATE,
        "dataset_id": dataset_hash(panel),
        "pipeline_hashes": {p: file_hash(root / p) for p in ("data.py", "features.py", "split.py")},
    }
    for part in ("train", "val", "test"):
        frame = splits[f"X_{part}"]
        info[f"{part}_rows_hash"] = hashlib.sha256(
            np.asarray(frame.index, dtype="int64").tobytes()).hexdigest()
        info[f"n_{part}"] = len(frame)
    info["run_id"] = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    info["environment"] = {
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
    }
    return info


def output_dir(args, panel):
    """Smoke outputs cannot overwrite the team's real-data parameter files."""
    mode = "smoke" if args.quick or args.source == "synthetic" else "development"
    requested = getattr(args, "output_dir", None)
    if requested is not None:
        path = Path(requested)
        # A quick or synthetic run must not overwrite a real-data rerun either.
        if mode == "smoke":
            path = path / f"{args.source}_smoke"
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = Path(__file__).resolve().parents[1] / "results" / "bhavesh" / (
        f"{args.source}_h{args.horizon * panel.step_min}_{mode}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
