"""PCA (1 mark): what the components capture, and whether PCA helps the models.   Owner: Pragun.

    python -m experiments.pragun_pca

1. explained variance (scree + cumulative) on standardised training features
2. loadings of the first components (which features drive them; name them in the report)
3. 2-D projection coloured by hour of day (shows the daily cycle)
4. PCA in front of Ridge and Random Forest for several numbers of components -> validation MAE
5. best PCA_Ridge and PCA_RandomForest -> results/benchmark.csv (same table as all models)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.common import finalize, get_data, parser, savefig, subsample, tune


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    panel, s = get_data(args)
    Xt, yt = subsample(s, 5_000 if args.quick else args.tune_rows)
    cols = list(Xt.columns)
    Z = StandardScaler().fit_transform(Xt)
    pca = PCA(random_state=42).fit(Z)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(ev) + 1), ev, label="per component")
    ax.plot(range(1, len(ev) + 1), np.cumsum(ev), "o-", color="C1", label="cumulative")
    ax.axhline(0.95, ls="--", color="grey")
    ax.set(xlabel="component", ylabel="explained variance", title="PCA on standardised features")
    ax.legend()
    savefig(fig, "pragun_pca_scree")
    print(f"  components for 90% / 95% / 99% variance: {np.searchsorted(np.cumsum(ev), [0.9, 0.95, 0.99]) + 1}")

    load = pd.DataFrame(pca.components_[:4].T, index=cols, columns=[f"PC{i}" for i in range(1, 5)])
    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(load, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set(yticks=range(len(cols)), yticklabels=cols, xticks=range(4), xticklabels=load.columns, title="Loadings")
    fig.colorbar(im, ax=ax)
    savefig(fig, "pragun_pca_loadings")

    P = pca.transform(Z)[:, :2]
    hour = (Xt["slot"].to_numpy() * panel.step_min / 60) if "slot" in Xt else None
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(P[:, 0], P[:, 1], c=hour, cmap="twilight", s=3, alpha=0.5)
    fig.colorbar(sc, ax=ax, label="hour of target")
    ax.set(xlabel="PC1", ylabel="PC2", title="Training rows in the first two components")
    savefig(fig, "pragun_pca_projection")

    ks = [2, 5, 10, 0.95] if args.quick else [2, 3, 5, 8, 10, 15, 20, 0.90, 0.95, 0.99]
    _, t_ridge = tune("PCA_Ridge", {"n_components": ks, "alpha": [1.0]}, s, args.tune_rows, quick=args.quick)
    _, t_rf = tune("PCA_RandomForest", {"n_components": ks if not args.quick else ks[:2], "n_estimators": [100],
                                        "max_depth": [16], "min_samples_leaf": [5]}, s, args.tune_rows, quick=args.quick)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, t in (("PCA + Ridge", t_ridge), ("PCA + Random Forest", t_rf)):
        t = t.assign(k=t["n_components"].astype(str)).set_index("k")["val_MAE"]
        t.plot(ax=ax, marker="o", ls="", label=name)
    ax.set(xlabel="n_components (int = count, float = variance kept)", ylabel="validation MAE",
           title="Does PCA help?")
    ax.legend()
    savefig(fig, "pragun_pca_vs_k")

    finalize("PCA_Ridge", s, panel, args)
    finalize("PCA_RandomForest", s, panel, args)


if __name__ == "__main__":
    main()
