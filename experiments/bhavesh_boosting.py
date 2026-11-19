"""Bhavesh's boosters: chronological validation, including round selection.

The test split is never used here. Run the team's final benchmark after tuning.
Synthetic/quick runs write only to results/bhavesh/<setup>/.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import AdaBoostRegressor

from experiments.common import get_data, parser, subsample
from src.artifacts import experiment_info, output_dir, write_json
from src.config import RANDOM_STATE
from src.models import make_model, save_best_params

GRIDS = {
    "AdaBoost": {"base_depth": [3, 6, 10], "learning_rate": [0.05, 0.3, 1.0],
                 "loss": ["linear", "square"], "n_estimators": [200]},
    "GradientBoosting": {"max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1, 0.2],
                         "subsample": [0.8, 1.0], "min_samples_leaf": [10, 30],
                         "n_estimators": [300]},
    "HistGradientBoosting": {"learning_rate": [0.03, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63],
                             "l2_regularization": [0.0, 1.0, 10.0],
                             "min_samples_leaf": [20, 50], "max_iter": [300, 600],
                             "early_stopping": [False]},
}


def round_parameter(name):
    return "max_iter" if name == "HistGradientBoosting" else "n_estimators"


def staged_mae(model, X, y, chunk_rows=4096):
    """Exact staged MAE, caching AdaBoost weak-tree predictions in bounded blocks.

    sklearn's AdaBoost staged_predict re-predicts all earlier trees at every
    stage. The weighted-median calculation below is identical, but each tree
    predicts each row once. All rows and rounds are still evaluated.
    """
    target = np.asarray(y)
    if not isinstance(model, AdaBoostRegressor):
        return np.array([np.mean(np.abs(target - pred)) for pred in model.staged_predict(X)])
    rounds = len(model.estimators_)
    error_sums = np.zeros(rounds, dtype=np.float64)
    for start in range(0, len(X), chunk_rows):
        stop = min(start + chunk_rows, len(X))
        values = X.iloc[start:stop].to_numpy(dtype=np.float32)
        predictions = np.column_stack([tree.predict(values) for tree in model.estimators_])
        row = np.arange(len(values))
        # Sorting weak predictions once is enough: future trees carry zero
        # weight until their round. This avoids sorting every growing prefix.
        order = np.argsort(predictions, axis=1)
        position = np.argsort(order, axis=1)
        sorted_predictions = np.take_along_axis(predictions, order, axis=1)
        active_weights = np.zeros_like(predictions)
        for count in range(1, rounds + 1):
            active_weights[row, position[:, count - 1]] = model.estimator_weights_[count - 1]
            cumulative = np.cumsum(active_weights, axis=1, dtype=np.float64)
            threshold = 0.5 * cumulative[:, -1]
            median = (cumulative >= threshold[:, None]).argmax(axis=1)
            prediction = sorted_predictions[row, median].copy()
            # At a floating-point half-weight boundary, reproduce sklearn's
            # original prefix ordering exactly rather than choosing another tie.
            tolerance = 4 * np.finfo(float).eps * rounds * cumulative[:, -1]
            previous = np.where(median > 0, cumulative[row, np.maximum(median - 1, 0)], 0)
            ambiguous = ((np.abs(cumulative[row, median] - threshold) <= tolerance) |
                         (np.abs(previous - threshold) <= tolerance))
            if ambiguous.any():
                prefix = predictions[ambiguous, :count]
                prefix_order = np.argsort(prefix, axis=1)
                prefix_cdf = np.cumsum(model.estimator_weights_[prefix_order], axis=1, dtype=np.float64)
                prefix_median = (prefix_cdf >= 0.5 * prefix_cdf[:, -1, None]).argmax(axis=1)
                prediction[ambiguous] = prefix[np.arange(len(prefix)), prefix_order[np.arange(len(prefix)), prefix_median]]
            error_sums[count - 1] += np.abs(target[start:stop] - prediction).sum()
    return error_sums / len(X)


def validation_curve(model, X_train, y_train, X_val, y_val):
    train = staged_mae(model, X_train, y_train)
    validation = staged_mae(model, X_val, y_val)
    return pd.DataFrame({"round": np.arange(1, len(validation) + 1),
                         "train_MAE": train, "val_MAE": validation})


def search_model(name, splits, *, tune_rows, candidates, quick=False):
    X_train, y_train = subsample(splits, min(1500, tune_rows) if quick else tune_rows)
    if len(X_train) == 0 or len(splits["X_val"]) == 0:
        raise ValueError("Training and validation periods must both contain usable rows.")
    params = list(ParameterSampler(GRIDS[name], n_iter=candidates, random_state=RANDOM_STATE))
    if quick:
        params = params[:2]
        for p in params:
            p[round_parameter(name)] = 12
    rows, curves = [], []
    for candidate, settings in enumerate(params):
        settings = {k: v.item() if isinstance(v, np.generic) else v for k, v in settings.items()}
        model = make_model(name, tuned=False, **settings)
        start = time.perf_counter()
        model.fit(X_train, y_train)
        fit_s = time.perf_counter() - start
        print(f"[{name} {candidate + 1}/{len(params)}] fit done ({fit_s:.2f}s); scoring all validation stages", flush=True)
        curve = validation_curve(model, X_train, y_train, splits["X_val"], splits["y_val"])
        best = curve.loc[curve["val_MAE"].idxmin()]
        selected = dict(settings, **{round_parameter(name): int(best["round"])})
        rows.append({"candidate": candidate, **settings, "selected_rounds": int(best["round"]),
                     "val_MAE": best["val_MAE"], "train_MAE": best["train_MAE"],
                     "fit_s": fit_s, "n_tune_train": len(X_train),
                     "n_val": len(splits["X_val"]), "params": selected})
        curves.append(curve.assign(candidate=candidate, model=name))
        print(f"[{name} {candidate + 1}/{len(params)}] validation MAE={best['val_MAE']:.3f}; "
              f"rounds={int(best['round'])}; fit={fit_s:.2f}s", flush=True)
    table = pd.DataFrame(rows).sort_values(["val_MAE", "candidate"]).reset_index(drop=True)
    return table.iloc[0]["params"], table, pd.concat(curves, ignore_index=True)


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--candidates", type=int, default=10, help="sampled configurations per model")
    ap.add_argument("--models", nargs="+", choices=list(GRIDS), default=list(GRIDS),
                    help="independent model families to tune; useful for resuming a completed family")
    ap.add_argument("--publish-params", action="store_true",
                    help="copy real, non-quick Madrid settings into the team's best_params folder")
    args = ap.parse_args(argv)
    if min(args.candidates, args.tune_rows, args.max_train_rows, args.n_sensors) <= 0:
        ap.error("candidate count, row limits, sensors and horizon must be positive")
    if args.publish_params and (args.quick or args.source != "madrid"):
        ap.error("--publish-params requires --source madrid without --quick")
    panel, splits = get_data(args)
    out = output_dir(args, panel)
    info = experiment_info(panel, splits, args)
    info.update({"purpose": "code smoke test" if args.quick or args.source == "synthetic" else "validation tuning",
                 "test_used": False, "tune_rows": args.tune_rows, "candidates": args.candidates,
                 "grids": GRIDS, "quick": args.quick})
    info["models_requested"] = args.models
    manifest_name = "tuning_manifest.json" if len(args.models) == 3 else "_".join(args.models) + "_tuning_manifest.json"
    write_json(out / manifest_name, info)
    fig, axes = plt.subplots(1, len(args.models), figsize=(4.7 * len(args.models), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, args.models):
        best, table, curves = search_model(name, splits, tune_rows=args.tune_rows,
                                          candidates=args.candidates, quick=args.quick)
        (out / "tuning").mkdir(exist_ok=True)
        table.drop(columns="params").to_csv(out / "tuning" / f"{name}.csv", index=False)
        curves.to_csv(out / "tuning" / f"{name}_rounds.csv", index=False)
        write_json(out / "best_params" / f"{name}.json", best)
        if args.publish_params:
            save_best_params(name, best)
            canonical = out.parents[1] / "tuning"
            canonical.mkdir(exist_ok=True)
            table.drop(columns="params").to_csv(canonical / f"{name}.csv", index=False)
        winner = curves[curves["candidate"] == table.iloc[0]["candidate"]]
        ax.plot(winner["round"], winner["train_MAE"], label="training sample")
        ax.plot(winner["round"], winner["val_MAE"], label="chronological validation")
        ax.axvline(best[round_parameter(name)], color="black", linestyle=":", label="selected round")
        ax.set(title=name, xlabel="Boosting rounds", ylabel=f"MAE ({panel.unit})")
    axes[0].legend(fontsize=10)
    fig.suptitle(f"{args.source} | {args.horizon * panel.step_min} min ahead | "
                 + ("SMOKE TEST ONLY" if args.quick or args.source == "synthetic" else "validation tuning"))
    fig.tight_layout()
    figure_name = "boosting_rounds.png" if len(args.models) == 3 else "boosting_rounds_" + "_".join(args.models) + ".png"
    fig.savefig(out / figure_name, dpi=160)
    plt.close(fig)
    print(f"Artifacts: {out}. Test period untouched. Run final benchmark separately.")


if __name__ == "__main__":
    main()
