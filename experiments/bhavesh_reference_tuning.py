"""Validation-only supporting tuning so efficiency comparators are not weak defaults.

Uses grids from the team starter; does not replace teammates' dedicated analyses.
No test scoring or model serving occurs here.
"""

from threadpoolctl import threadpool_limits
from experiments.common import get_data, parser, tune
from experiments.analysis_c_efficiency import set_threads
from src.artifacts import output_dir
from src.models import make_model, save_best_params

GRIDS = {
    "Ridge": ({"alpha": [0.1, 1.0, 10.0, 100.0]}, None),
    "DecisionTree": (
        {"max_depth": [8, 12, 16, 20, None], "min_samples_leaf": [1, 5, 20, 50, 100]},
        None,
    ),
    "RandomForest": (
        {
            "n_estimators": [100, 200],
            "max_depth": [10, 16, 24, None],
            "min_samples_leaf": [1, 5, 20],
            "max_features": [1.0, 0.5, 0.3, "sqrt"],
        },
        8,
    ),
    "PCA_Ridge": (
        {"n_components": [2, 3, 5, 8, 10, 15, 20, 0.90, 0.95, 0.99], "alpha": [1.0]},
        None,
    ),
    "PCA_RandomForest": (
        {
            "n_components": [5, 10, 15, 20, 0.95, 0.99],
            "n_estimators": [100],
            "max_depth": [16],
            "min_samples_leaf": [5],
        },
        None,
    ),
    "XGBoost": (
        {
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.03, 0.1, 0.2],
            "subsample": [0.7, 0.9],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 10],
            "reg_lambda": [1.0, 5.0],
            "n_estimators": [400],
        },
        8,
    ),
    "LightGBM": (
        {
            "num_leaves": [15, 31, 63, 127],
            "learning_rate": [0.03, 0.1, 0.2],
            "min_child_samples": [10, 50, 200],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "subsample": [0.7, 0.9],
            "subsample_freq": [1],
            "n_estimators": [400],
        },
        8,
    ),
}


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--publish-params", action="store_true",
                    help="explicitly publish selected settings; shared tuning tables are never overwritten")
    args = ap.parse_args(argv)
    if args.quick or args.source != "madrid":
        ap.error("Supporting reference tuning requires real Madrid data without --quick.")
    if args.threads < 1:
        ap.error("threads must be positive")
    panel, s = get_data(args)
    reference_dir = output_dir(args, panel) / "references"
    s = {k: v for k, v in s.items() if k.endswith(("_train", "_val"))}
    with threadpool_limits(limits=args.threads):
        for name, (grid, n_iter) in GRIDS.items():
            print(f"[supporting comparator tuning] {name}", flush=True)
            best, _ = tune(
                name,
                grid,
                s,
                args.tune_rows,
                n_iter=n_iter,
                quick=args.quick,
                out_dir=reference_dir,
                build=lambda params, name=name: set_threads(
                    make_model(name, tuned=False, **params), args.threads
                ),
            )
            if args.publish_params:
                save_best_params(name, best)


if __name__ == "__main__":
    main()
