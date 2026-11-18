"""Consolidated benchmark: every model (tuned settings if saved), same data, same splits.   Owner: Anirudh.

    python -m experiments.run_benchmark                  # all models
    python -m experiments.run_benchmark --only Ridge DecisionTree LightGBM

Writes results/benchmark.csv (tag "all") and models/*.joblib for the UI.
Run it again after everyone has tuned (Tue 24 Nov, models freeze) to get the final table.
"""
from experiments.common import finalize, get_data, parser
from src.models import BASELINES, ML_MODELS

SLOW = {"GradientBoosting", "PCA_RandomForest", "AdaBoost", "RandomForest"}


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="model names, space separated")
    ap.add_argument("--skip-slow", action="store_true", help=f"skip {sorted(SLOW)}")
    args = ap.parse_args(argv)
    panel, splits = get_data(args)
    for name in BASELINES + ML_MODELS:
        if args.only and name not in args.only:
            continue
        if args.skip_slow and name in SLOW:
            continue
        try:
            finalize(name, splits, panel, args)
        except ImportError as e:
            print(f"  skip {name}: {e}")


if __name__ == "__main__":
    main()
