"""Boosting part 1 (shares the 2 boosting marks with Anirudh): AdaBoost and Gradient Boosting.   Owner: Bhavesh.

    python -m experiments.bhavesh_boosting

1. AdaBoost: base-tree depth, learning rate, loss; staged validation curve (error vs rounds)
2. Gradient Boosting (classic sklearn): depth, learning rate, subsample; staged validation curve
3. HistGradientBoosting (histogram-based, much faster): learning rate, leaves, regularisation
4. final models -> results/benchmark.csv, models/*.joblib
Explain in the report HOW the variants differ: re-weighting samples (AdaBoost) vs fitting
residuals with gradients (GBM) vs binned histograms (HistGB).
"""
import matplotlib.pyplot as plt
import numpy as np

from experiments.common import finalize, get_data, parser, savefig, subsample, tune
from src.models import make_model


def staged_curve(model, s, label, ax):
    errs = [np.mean(np.abs(s["y_val"].to_numpy() - p)) for p in model.staged_predict(s["X_val"])]
    ax.plot(range(1, len(errs) + 1), errs, label=label)
    return int(np.argmin(errs)) + 1


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    panel, s = get_data(args)
    q = args.quick

    tune("AdaBoost", {"base_depth": [3, 6, 10], "learning_rate": [0.05, 0.3, 1.0], "loss": ["linear", "square"],
                      "n_estimators": [100]}, s, args.tune_rows, n_iter=8, quick=q)
    tune("GradientBoosting", {"max_depth": [3, 5, 7], "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8, 1.0],
                              "n_estimators": [200]}, s, args.tune_rows, n_iter=8, quick=q)
    tune("HistGradientBoosting", {"learning_rate": [0.03, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63, 127],
                                  "l2_regularization": [0.0, 1.0], "max_iter": [300, 600]},
         s, args.tune_rows, n_iter=10, quick=q)

    fig, ax = plt.subplots(figsize=(7, 4))
    Xt, yt = subsample(s, 5_000 if q else args.tune_rows)
    for name in ("AdaBoost", "GradientBoosting"):
        best_n = staged_curve(make_model(name).fit(Xt, yt), s, name, ax)
        print(f"  {name}: best number of rounds on validation = {best_n}")
    ax.set(xlabel="boosting rounds", ylabel="validation MAE", title="Error vs boosting rounds")
    ax.legend()
    savefig(fig, "bhavesh_boosting_rounds")

    for name in ("AdaBoost", "GradientBoosting", "HistGradientBoosting"):
        finalize(name, s, panel, args)


if __name__ == "__main__":
    main()
