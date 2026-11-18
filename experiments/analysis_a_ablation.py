"""Analysis A: feature analysis and ablation (2 marks).   Owner: Pragun.

    python -m experiments.analysis_a_ablation
    python -m experiments.analysis_a_ablation --models DecisionTree RandomForest LightGBM

For every model and every feature group: retrain WITHOUT that group (tuned settings) and record
the change in test MAE vs the full feature set. Also "lags only" as a minimal model, and
correlation vs distance neighbours. Rows go to results/benchmark.csv with tag "ablation_*".
Run after models freeze (24 Nov) so the tuned parameters are used.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import ROOT, finalize, get_data, parser, savefig
from src.features import GROUPS
from src.models import make_model

DEFAULT_MODELS = ["Ridge", "DecisionTree", "RandomForest", "HistGradientBoosting", "XGBoost", "LightGBM"]
REMOVABLE = [g for g in GROUPS if g != "lags"] + ["lags"]


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args(argv)
    models = args.models or (["Ridge", "DecisionTree", "LightGBM"] if args.quick else DEFAULT_MODELS)
    variants = {"full": list(GROUPS)}
    variants.update({f"no_{g}": [x for x in GROUPS if x != g] for g in REMOVABLE})
    variants["lags_only"] = ["lags"]
    variants["neighbors_by_distance"] = list(GROUPS)

    rows = []
    for tag, groups in variants.items():
        panel, s = get_data(args, groups=groups,
                            neighbor_method="distance" if tag == "neighbors_by_distance" else "corr")
        if tag not in ("full", "neighbors_by_distance") and s["X_train"].shape[1] == variants_n.get("full", -1):
            print(f"  {tag}: group not present in this dataset, skipped")
            continue
        variants_n[tag] = s["X_train"].shape[1]
        for m in models:
            try:
                r = finalize(m, s, panel, args, model=make_model(m), tag=f"ablation_{tag}", save=False)
            except ImportError as e:
                print(f"  skip {m}: {e}")
                continue
            rows.append({"variant": tag, "model": m, "test_MAE": r["test_MAE"], "n_features": r["n_features"]})

    df = pd.DataFrame(rows)
    full = df[df["variant"] == "full"].set_index("model")["test_MAE"]
    df["delta_MAE_%"] = (df["test_MAE"] / df["model"].map(full) - 1) * 100
    df.to_csv(ROOT / "results" / "ablation.csv", index=False)
    pivot = df[df["variant"] != "full"].pivot(index="variant", columns="model", values="delta_MAE_%")
    print(pivot.round(1))

    fig, ax = plt.subplots(figsize=(1.6 * len(pivot.columns) + 3, 0.45 * len(pivot) + 2))
    lim = np.nanmax(np.abs(pivot.to_numpy())) or 1
    im = ax.imshow(pivot, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set(xticks=range(len(pivot.columns)), xticklabels=pivot.columns, yticks=range(len(pivot)),
           yticklabels=pivot.index, title="Change in test MAE vs full features (%)\nred = removing it hurts")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iat[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(fig, "analysis_a_ablation_heatmap")


variants_n: dict = {}

if __name__ == "__main__":
    main()
