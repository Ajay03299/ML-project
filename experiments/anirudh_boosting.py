"""Boosting part 2 (shares the 2 boosting marks with Bhavesh): XGBoost and LightGBM.   Owner: Anirudh.

    python -m experiments.anirudh_boosting

1. random search for each library on the validation period
2. early stopping: the number of rounds is picked on the validation period, never on test
3. validation curves (error vs rounds) for both
4. final models -> results/benchmark.csv, models/*.joblib
Explain in the report how XGBoost (level-wise trees, second-order gradients) and LightGBM
(leaf-wise trees, histogram + GOSS) differ, and whether the difference shows in the results.
"""
import matplotlib.pyplot as plt

from experiments.common import finalize, get_data, parser, savefig, subsample, tune
from src.models import make_model, save_best_params, best_params


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    panel, s = get_data(args)
    q = args.quick
    tune("XGBoost", {"max_depth": [4, 6, 8, 10], "learning_rate": [0.03, 0.1, 0.2], "subsample": [0.7, 0.9],
                     "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 10], "reg_lambda": [1.0, 5.0],
                     "n_estimators": [400]}, s, args.tune_rows, n_iter=12, quick=q)
    tune("LightGBM", {"num_leaves": [15, 31, 63, 127], "learning_rate": [0.03, 0.1, 0.2],
                      "min_child_samples": [10, 50, 200], "colsample_bytree": [0.6, 0.8, 1.0],
                      "subsample": [0.7, 0.9], "subsample_freq": [1], "n_estimators": [400]},
         s, args.tune_rows, n_iter=12, quick=q)

    # early stopping on the validation period -> best number of rounds
    Xt, yt = subsample(s, 5_000 if q else args.max_train_rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("XGBoost", "LightGBM"):
        p = best_params(name)
        if name == "XGBoost":
            m = make_model(name, n_estimators=1500, early_stopping_rounds=50, eval_metric="mae")
            m.fit(Xt, yt, eval_set=[(s["X_val"], s["y_val"])], verbose=False)
            curve = m.evals_result()["validation_0"]["mae"]
            best_n = int(m.best_iteration) + 1
        else:
            import lightgbm as lgb
            m = make_model(name, n_estimators=1500)
            m.fit(Xt, yt, eval_set=[(s["X_val"], s["y_val"])], eval_metric="l1",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
            curve = m.evals_result_["valid_0"]["l1"]
            best_n = int(m.best_iteration_ or len(curve))
        ax.plot(range(1, len(curve) + 1), curve, label=f"{name} (best {best_n})")
        p["n_estimators"] = best_n
        save_best_params(name, p)
        print(f"  {name}: early stopping picked {best_n} rounds")
    ax.set(xlabel="boosting rounds", ylabel="validation MAE", title="Early stopping on the validation period")
    ax.legend()
    savefig(fig, "anirudh_boosting_rounds")

    for name in ("XGBoost", "LightGBM"):
        finalize(name, s, panel, args)


if __name__ == "__main__":
    main()
