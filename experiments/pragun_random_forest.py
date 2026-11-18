"""Random Forest (1 mark).   Owner: Pragun.

    python -m experiments.pragun_random_forest

1. how many trees are enough: validation MAE as trees are added (warm start)
2. random search over depth, leaf size and max_features on the validation period
3. final model -> results/benchmark.csv, models/RandomForest.joblib
4. impurity importance vs permutation importance on the validation period (they can disagree;
   explain why in the report: impurity favours high-cardinality features)
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from experiments.common import finalize, get_data, parser, savefig, subsample, tune
from src.evaluate import regression_metrics
from src.models import make_model


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    panel, s = get_data(args)
    Xt, yt = subsample(s, 5_000 if args.quick else args.tune_rows)

    # 1 number of trees
    rf = RandomForestRegressor(n_estimators=10, max_depth=16, min_samples_leaf=5, warm_start=True,
                               n_jobs=-1, random_state=42)
    counts = [10, 25] if args.quick else [10, 25, 50, 100, 200, 300]
    curve = []
    for n in counts:
        rf.set_params(n_estimators=n).fit(Xt, yt)
        curve.append(regression_metrics(s["y_val"], rf.predict(s["X_val"]))["MAE"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(counts, curve, "o-")
    ax.set(xlabel="number of trees", ylabel="validation MAE", title="Random Forest: trees vs error")
    savefig(fig, "pragun_rf_trees")

    # 2 random search
    grid = {"n_estimators": [100, 200], "max_depth": [10, 16, 24, None], "min_samples_leaf": [1, 5, 20],
            "max_features": [1.0, 0.5, 0.3, "sqrt"]}
    tune("RandomForest", grid, s, args.tune_rows, n_iter=12, quick=args.quick)

    # 3 final
    model = make_model("RandomForest")
    finalize("RandomForest", s, panel, args, model=model)

    # 4 importances
    cols = s["X_train"].columns
    imp = pd.DataFrame({"impurity": model.feature_importances_}, index=cols)
    Xv, yv = s["X_val"].iloc[:3000 if args.quick else 20000], s["y_val"].iloc[:3000 if args.quick else 20000]
    perm = permutation_importance(model, Xv, yv, n_repeats=3, random_state=42, scoring="neg_mean_absolute_error",
                                  n_jobs=-1)
    imp["permutation"] = perm.importances_mean
    imp = imp.sort_values("permutation").tail(15)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    imp["impurity"].plot.barh(ax=ax[0], title="impurity importance")
    imp["permutation"].plot.barh(ax=ax[1], title="permutation importance (val, MAE increase)")
    savefig(fig, "pragun_rf_importance")


if __name__ == "__main__":
    main()
