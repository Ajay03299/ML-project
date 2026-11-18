"""Helpers every experiment script uses: same data, same splits, same tuning loop, same saving.

Run any experiment from the repo root, for example
    python -m experiments.pragun_random_forest            # settings from src/config.py
    python -m experiments.pragun_random_forest --quick    # tiny grid, for a 1-minute smoke test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import ParameterGrid, ParameterSampler  # noqa: E402

from src import config  # noqa: E402
from src.data import load_panel  # noqa: E402
from src.evaluate import evaluate_model, regression_metrics  # noqa: E402
from src.features import GROUPS, build_dataset  # noqa: E402
from src.models import OWNERS, make_model, save_best_params  # noqa: E402
from src.split import pick_holdout_sensors, split_dates, split_xy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"
TUNE_DIR = ROOT / "results" / "tuning"
MODELS_DIR = ROOT / "models"


def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--source", default=config.SOURCE, choices=["synthetic", "madrid", "pems04"])
    ap.add_argument("--n-sensors", type=int, default=config.N_SENSORS)
    ap.add_argument("--horizon", type=int, default=config.HORIZON, help="steps ahead (4 x 15 min = 60 min)")
    ap.add_argument("--max-train-rows", type=int, default=config.MAX_TRAIN_ROWS)
    ap.add_argument("--tune-rows", type=int, default=config.TUNE_ROWS)
    ap.add_argument("--quick", action="store_true", help="tiny grids and samples, to check the script runs")
    return ap


def get_data(args, groups=GROUPS, holdout: bool = False, neighbor_method: str = "corr"):
    """Panel plus the split dict that evaluate_model expects. Identical for everyone."""
    panel = load_panel(args.source, n_sensors=args.n_sensors)
    train_end, val_end = split_dates(panel.index)
    X, y, meta = build_dataset(panel, horizon=args.horizon, groups=groups, train_end=train_end,
                               neighbor_method=neighbor_method)
    held = pick_holdout_sensors(panel.sensors, config.HOLDOUT_FRAC, config.RANDOM_STATE) if holdout else None
    rows = min(args.max_train_rows, 20_000) if args.quick else args.max_train_rows
    splits = split_xy(X, y, meta, train_end, val_end, holdout_sensors=held, max_train_rows=rows,
                      seed=config.RANDOM_STATE)
    print(f"[data] {args.source}: {len(panel.sensors)} sensors, unit {panel.unit}, horizon "
          f"{args.horizon * panel.step_min} min | train {len(splits['X_train']):,}  val {len(splits['X_val']):,}  "
          f"test {len(splits['X_test']):,}" + (f"  unseen {len(splits['X_unseen']):,}" if holdout else "")
          + f" | {X.shape[1]} features")
    return panel, splits


def horizon_min(args, panel) -> int:
    return args.horizon * panel.step_min


def subsample(splits: dict, n: int, seed: int = config.RANDOM_STATE):
    """A smaller training sample for hyper-parameter search (time order does not matter inside train)."""
    X, y = splits["X_train"], splits["y_train"]
    if len(X) <= n:
        return X, y
    idx = np.sort(np.random.default_rng(seed).choice(len(X), n, replace=False))
    return X.iloc[idx], y.iloc[idx]


def tune(name: str, grid: dict, splits: dict, tune_rows: int, n_iter: int | None = None,
         build=None, quick: bool = False) -> tuple[dict, pd.DataFrame]:
    """Time-ordered validation: fit on (a sample of) the training period, score on the validation period.

    grid    dict of parameter -> list of values. All combinations, or n_iter random ones.
    build   function(params) -> unfitted model; default make_model(name, tuned=False, **params)
    Writes results/tuning/<name>.csv and results/best_params/<name>.json, returns (best, table).
    Never looks at the test period.
    """
    build = build or (lambda p: make_model(name, tuned=False, **p))
    combos = list(ParameterSampler(grid, n_iter, random_state=config.RANDOM_STATE)) if n_iter \
        else list(ParameterGrid(grid))
    if quick:
        combos = combos[:2]
    Xt, yt = subsample(splits, 5_000 if quick else tune_rows)
    rows = []
    for i, p in enumerate(combos, 1):
        model = build(dict(p))
        t0 = time.perf_counter()
        model.fit(Xt, yt)
        fit_s = time.perf_counter() - t0
        m_val = regression_metrics(splits["y_val"], model.predict(splits["X_val"]))
        m_tr = regression_metrics(yt, model.predict(Xt))
        rows.append({"combo": i - 1, **p, "val_MAE": m_val["MAE"], "val_RMSE": m_val["RMSE"], "val_R2": m_val["R2"],
                     "train_MAE": m_tr["MAE"], "fit_s": fit_s})
        print(f"  [{name} {i}/{len(combos)}] {p} -> val MAE {m_val['MAE']:.2f} (train {m_tr['MAE']:.2f}, {fit_s:.1f}s)")
    table = pd.DataFrame(rows).sort_values("val_MAE").reset_index(drop=True)
    TUNE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TUNE_DIR / f"{name}.csv", index=False)
    best = {k: _py(v) for k, v in combos[int(table.iloc[0]["combo"])].items()}   # exact original types
    save_best_params(name, best)
    print(f"  best {name}: {best}  (val MAE {table.iloc[0]['val_MAE']:.2f})")
    return best, table


def _py(v):
    """numpy scalars -> plain Python, so params round-trip through JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.str_):
        return str(v)
    return v


def finalize(name: str, splits: dict, panel, args, model=None, tag: str = "all", save: bool = True) -> dict:
    """Fit the tuned model on the full training sample, score it, record it, save it for the UI."""
    model = model if model is not None else make_model(name)
    row = evaluate_model(name, model, splits, tag=tag, owner=OWNERS.get(name, ""),
                         horizon_min=horizon_min(args, panel))
    print(f"[result] {name:22s} tag={tag:14s} val MAE {row['val_MAE']:.2f} | test MAE {row['test_MAE']:.2f} "
          f"RMSE {row['test_RMSE']:.2f} R2 {row['test_R2']:.3f} WAPE {row['test_WAPE']:.3f}")
    if save and tag == "all":
        save_for_ui(name, model, splits, panel, args)
    return row


def save_for_ui(name, model, splits, panel, args) -> None:
    """Only models trained on the main feature set go to models/, so the UI can load all of them."""
    MODELS_DIR.mkdir(exist_ok=True)
    cols = list(splits["X_train"].columns)
    info = {"columns": cols, "horizon": args.horizon, "n_sensors": args.n_sensors,
            "source": args.source, "step_min": panel.step_min, "unit": panel.unit}
    meta = MODELS_DIR / "meta.json"
    if meta.exists():
        old = json.loads(meta.read_text())
        same = all(old.get(k) == info[k] for k in ("columns", "horizon", "n_sensors", "source"))
        if not same:
            print("  NOTE: models/ held models for a different setup; clearing it so the UI stays consistent")
            for f in MODELS_DIR.glob("*.joblib"):
                f.unlink()
    meta.write_text(json.dumps(info, indent=2))
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")


def savefig(fig, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  figure -> {path.relative_to(ROOT)}")
    return path
