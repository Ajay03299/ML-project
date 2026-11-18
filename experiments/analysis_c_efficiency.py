"""Analysis C: model efficiency (2 marks).   Owner: Bhavesh.

    python -m experiments.analysis_c_efficiency

1. from results/benchmark.csv (tag "all"): test MAE vs training time, prediction latency and model size,
   with the Pareto front (models nobody beats on both error and cost)
2. scaling: training time and MAE as the training set grows, per model
3. results/efficiency.csv + a sentence per plot: is a simpler model good enough for real-time use?
Run after the final benchmark (24 Nov).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import ROOT, get_data, parser, savefig
from src.evaluate import RESULTS_CSV, evaluate_model
from src.models import make_model

SCALING_MODELS = ["Ridge", "DecisionTree", "RandomForest", "HistGradientBoosting", "XGBoost", "LightGBM"]


def pareto(df, x, y="test_MAE"):
    d = df.sort_values(x)
    best, keep = np.inf, []
    for _, r in d.iterrows():
        keep.append(r[y] < best)
        best = min(best, r[y])
    return d[keep]


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    bench = pd.read_csv(RESULTS_CSV)
    b = bench[bench["tag"] == "all"].copy()
    b["train_time_s"] = b["train_time_s"].clip(lower=1e-3)
    cols = ["model", "owner", "test_MAE", "test_RMSE", "train_time_s", "predict_ms_per_1000", "model_size_kb", "n_train"]
    b[cols].sort_values("test_MAE").round(3).to_csv(ROOT / "results" / "efficiency.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (x, label) in zip(axes, [("train_time_s", "training time (s, log)"),
                                     ("predict_ms_per_1000", "prediction time per 1,000 rows (ms, log)"),
                                     ("model_size_kb", "saved model size (KB, log)")]):
        ax.scatter(b[x].clip(lower=1e-3), b["test_MAE"])
        for _, r in b.iterrows():
            ax.annotate(r["model"], (max(r[x], 1e-3), r["test_MAE"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        p = pareto(b.assign(**{x: b[x].clip(lower=1e-3)}), x)
        ax.step(p[x], p["test_MAE"], where="post", color="C3", alpha=0.6, label="Pareto front")
        ax.set(xscale="log", xlabel=label, ylabel="test MAE")
        ax.legend(fontsize=8)
    savefig(fig, "analysis_c_cost_vs_error")

    # scaling with training-set size
    panel, s = get_data(args)
    sizes = [2_000, 5_000] if args.quick else [12_500, 25_000, 50_000, 100_000, 200_000]
    rows = []
    models = SCALING_MODELS[:2] if args.quick else SCALING_MODELS
    for n in sizes:
        if n > len(s["X_train"]):
            continue
        idx = np.sort(np.random.default_rng(42).choice(len(s["X_train"]), n, replace=False))
        sub = dict(s, X_train=s["X_train"].iloc[idx], y_train=s["y_train"].iloc[idx])
        for m in models:
            try:
                r = evaluate_model(m, make_model(m), sub, tag=f"scaling_{n}", results_path=ROOT / "results" / "scaling.csv")
            except ImportError:
                continue
            rows.append({"model": m, "n_train": n, "train_time_s": r["train_time_s"], "test_MAE": r["test_MAE"]})
    sc = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for m, g in sc.groupby("model"):
        ax[0].plot(g["n_train"], g["train_time_s"], "o-", label=m)
        ax[1].plot(g["n_train"], g["test_MAE"], "o-", label=m)
    ax[0].set(xscale="log", yscale="log", xlabel="training rows", ylabel="training time (s)")
    ax[1].set(xscale="log", xlabel="training rows", ylabel="test MAE")
    ax[1].legend(fontsize=8)
    savefig(fig, "analysis_c_scaling")
    print(b[cols].sort_values("test_MAE").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
