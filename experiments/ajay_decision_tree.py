"""Decision Tree (1 mark) + Ridge baseline tuning.   Owner: Ajay.

    python -m experiments.ajay_decision_tree

1. depth curve: train vs validation MAE as the tree grows (shows over-fitting)
2. grid over depth and leaf size on the validation period -> best params saved
3. cost-complexity pruning path on the best depth
4. final model on the full training sample -> results/benchmark.csv, models/DecisionTree.joblib
5. feature importances and the top 3 levels of the tree (for the report and the viva)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

from experiments.common import finalize, get_data, parser, savefig, subsample, tune
from src.evaluate import regression_metrics
from src.models import make_model, save_best_params


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    panel, s = get_data(args)
    Xt, yt = subsample(s, 5_000 if args.quick else args.tune_rows)

    # 1 depth curve
    depths = [2, 4, 8] if args.quick else [2, 4, 6, 8, 10, 12, 14, 16, 20, 25]
    tr, va = [], []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=42).fit(Xt, yt)
        tr.append(regression_metrics(yt, m.predict(Xt))["MAE"])
        va.append(regression_metrics(s["y_val"], m.predict(s["X_val"]))["MAE"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(depths, tr, "o-", label="train")
    ax.plot(depths, va, "o-", label="validation")
    ax.set(xlabel="max_depth", ylabel=f"MAE ({panel.unit})", title="Decision Tree: depth vs error")
    ax.legend()
    savefig(fig, "ajay_dt_depth_curve")

    # 2 grid search on the validation period
    grid = {"max_depth": [8, 12, 16, 20, None], "min_samples_leaf": [1, 5, 20, 50, 100]}
    best, _ = tune("DecisionTree", grid, s, args.tune_rows, quick=args.quick)

    # 3 cost-complexity pruning on top of the best tree
    path = DecisionTreeRegressor(random_state=42, **best).cost_complexity_pruning_path(Xt, yt)
    alphas = np.unique(np.quantile(path.ccp_alphas, np.linspace(0, 0.99, 6 if args.quick else 15)))
    rows = []
    for a in alphas:
        m = DecisionTreeRegressor(random_state=42, ccp_alpha=a, **best).fit(Xt, yt)
        rows.append({"ccp_alpha": a, "leaves": m.get_n_leaves(),
                     "val_MAE": regression_metrics(s["y_val"], m.predict(s["X_val"]))["MAE"]})
    prune = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(prune["leaves"], prune["val_MAE"], "o-")
    ax.set(xscale="log", xlabel="number of leaves (after pruning)", ylabel="validation MAE",
           title="Cost-complexity pruning")
    savefig(fig, "ajay_dt_pruning")
    b = prune.loc[prune["val_MAE"].idxmin()]
    if b["val_MAE"] < prune.iloc[0]["val_MAE"] - 1e-9:
        best["ccp_alpha"] = float(b["ccp_alpha"])
        save_best_params("DecisionTree", best)
        print(f"  pruning helps: ccp_alpha={b['ccp_alpha']:.4g} ({int(b['leaves'])} leaves)")

    # Ridge alpha (a baseline, but tune it so the comparison is fair)
    tune("Ridge", {"alpha": [0.1, 1.0, 10.0, 100.0]}, s, args.tune_rows, quick=args.quick)

    # 4 final fit + record
    for name in ["Persistence", "SeasonalNaive_day", "SeasonalNaive_week", "HistoricalAverage", "Ridge"]:
        finalize(name, s, panel, args)
    model = make_model("DecisionTree")
    finalize("DecisionTree", s, panel, args, model=model)

    # 5 interpretation
    imp = pd.Series(model.feature_importances_, index=s["X_train"].columns).sort_values().tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    imp.plot.barh(ax=ax)
    ax.set(title="Decision Tree: feature importance (impurity)")
    savefig(fig, "ajay_dt_importance")
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(model, max_depth=2, feature_names=list(s["X_train"].columns), filled=True, ax=ax, fontsize=8)
    savefig(fig, "ajay_dt_top_levels")


if __name__ == "__main__":
    main()
