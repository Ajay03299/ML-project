"""Shared evaluation helper so every model is scored and timed the same way.   Owner: Anirudh.

Every model, from every person, goes through `evaluate_model`, which appends one row to
results/benchmark.csv. That file feeds the final benchmark table, the efficiency
analysis (C) and the UI leaderboard.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS_CSV = ROOT / "results" / "benchmark.csv"


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        # WAPE is safe when flows are near zero at night, unlike plain MAPE.
        "WAPE": float(np.abs(y_true - y_pred).sum() / denom) if denom else float("nan"),
    }


def evaluate_model(name, model, splits: dict, tag: str = "all", fit: bool = True, owner: str = "",
                   horizon_min: int = 60, results_path: Path = RESULTS_CSV, extra: dict | None = None) -> dict:
    """Fit (optionally), score on val and test, time everything, upsert into the results CSV.

    `splits` is the dict from src.split.split_xy. `tag` names the experiment (for example
    "all", "no_neighbors", "pca95") so ablation rows do not overwrite the main benchmark.
    """
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    row = {"model": name, "tag": tag, "owner": owner, "horizon_min": horizon_min,
           "n_train": len(X_tr), "n_features": X_tr.shape[1]}

    if fit:
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        row["train_time_s"] = time.perf_counter() - t0

    for part in ("val", "test", "unseen"):
        if f"X_{part}" not in splits:
            continue
        X, y = splits[f"X_{part}"], splits[f"y_{part}"]
        t0 = time.perf_counter()
        pred = model.predict(X)
        elapsed = time.perf_counter() - t0
        for k, v in regression_metrics(y, pred).items():
            row[f"{part}_{k}"] = v
        if part == "test":
            row["predict_ms_per_1000"] = elapsed / max(len(X), 1) * 1_000_000

    # Disk-backed serialization avoids holding a second full copy of a large
    # forest in RAM just to measure its uncompressed joblib size.
    with tempfile.TemporaryFile() as buf:
        joblib.dump(model, buf)
        row["model_size_kb"] = buf.tell() / 1024
    row.update(extra or {})

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if results_path.exists():
        old = pd.read_csv(results_path)
        same = (old["model"] == name) & (old["tag"] == tag) & (old["horizon_min"] == horizon_min)
        new = pd.concat([old[~same], new], ignore_index=True)
    new.to_csv(results_path, index=False)
    return row
