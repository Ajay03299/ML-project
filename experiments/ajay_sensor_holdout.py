"""Sensor-level holdout: can the models forecast sensors they never saw in training?   Owner: Ajay.

    python -m experiments.ajay_sensor_holdout
    python -m experiments.ajay_sensor_holdout --only DecisionTree LightGBM

About 15% of sensors are removed from training and validation. Every model (tuned settings) is
retrained without them and scored on the test period twice: seen sensors vs unseen sensors.
Rows go to results/benchmark.csv with tag "holdout". Run after models freeze (24 Nov).
"""
import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import ROOT, finalize, get_data, parser, savefig
from src.models import BASELINES, ML_MODELS

SLOW = {"GradientBoosting", "PCA_RandomForest"}


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args(argv)
    panel, s = get_data(args, holdout=True)
    names = args.only or [m for m in BASELINES + ML_MODELS if not (args.quick and m in SLOW)]
    rows = []
    for name in names:
        try:
            r = finalize(name, s, panel, args, tag="holdout", save=False)
        except ImportError as e:
            print(f"  skip {name}: {e}")
            continue
        rows.append({"model": name, "seen sensors": r["test_MAE"], "unseen sensors": r["unseen_MAE"]})
    df = pd.DataFrame(rows).set_index("model")
    df["gap_%"] = (df["unseen sensors"] / df["seen sensors"] - 1) * 100
    df.round(2).to_csv(ROOT / "results" / "sensor_holdout.csv")
    print(df.round(2))
    fig, ax = plt.subplots(figsize=(9, 4))
    df[["seen sensors", "unseen sensors"]].plot.bar(ax=ax)
    ax.set(ylabel=f"test MAE ({panel.unit})", title="Seen vs never-seen sensors (test period)")
    savefig(fig, "ajay_sensor_holdout")


if __name__ == "__main__":
    main()
