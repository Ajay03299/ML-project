"""Analysis E: error analysis across all models (2 marks).   Owner: Anirudh.

    python -m experiments.analysis_e_errors

Loads every model saved in models/ (trained by run_benchmark or each person's script) and breaks
the test-period errors down by: time-of-day class (peak / off-peak / night), weekday / weekend /
holiday, rain vs dry, sudden changes, and sensor. Writes results/error_breakdown.csv and figures.
Run after the final benchmark (24 Nov).
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import MODELS_DIR, ROOT, get_data, parser, savefig


def segments(X: pd.DataFrame, meta: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    t = meta["target_time"]
    hour = t.dt.hour
    weekend = t.dt.day_of_week >= 5
    seg = pd.DataFrame(index=X.index)
    seg["time_of_day"] = np.select([hour.between(0, 5), (~weekend) & (hour.between(7, 9) | hour.between(17, 19))],
                                   ["night (0-6h)", "peak (7-10h, 17-20h)"], "off-peak")
    day = np.where(weekend, "weekend", "weekday")
    if "is_holiday" in X:
        day = np.where(X["is_holiday"].to_numpy() == 1, "holiday", day)
    seg["day_type"] = day
    if "wx_precip" in X:
        seg["weather"] = np.where(X["wx_precip"].to_numpy() > 0, "rain", "dry")
    change = (y - X["flow_lag0"]).abs()
    seg["change"] = np.where(change > change.quantile(0.9), "sudden change (top 10%)", "normal")
    seg["hour"] = hour.to_numpy()
    seg["sensor"] = meta["sensor"].to_numpy()
    return seg


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    info = json.loads((MODELS_DIR / "meta.json").read_text())
    args.source, args.n_sensors, args.horizon = info["source"], info["n_sensors"], info["horizon"]
    panel, s = get_data(args)
    X, y, meta = s["X_test"][info["columns"]], s["y_test"], s["meta_test"]
    seg = segments(X, meta, y)
    errs = {}
    for f in sorted(MODELS_DIR.glob("*.joblib")):
        errs[f.stem] = np.abs(y.to_numpy() - joblib.load(f).predict(X))
    E = pd.DataFrame(errs, index=X.index)
    print(f"  {len(E.columns)} models, {len(E):,} test rows")

    rows = []
    for dim in [c for c in ("time_of_day", "day_type", "weather", "change") if c in seg]:
        g = E.groupby(seg[dim]).mean()
        for level, r in g.iterrows():
            rows.append({"dimension": dim, "level": level, "share_of_rows": (seg[dim] == level).mean(), **r.to_dict()})
    table = pd.DataFrame(rows)
    table.round(3).to_csv(ROOT / "results" / "error_breakdown.csv", index=False)
    print(table.round(2).to_string(index=False))

    order = E.mean().sort_values().index
    dims = [c for c in ("time_of_day", "day_type", "weather", "change") if c in seg]
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4.5), sharey=True)
    for ax, dim in zip(np.atleast_1d(axes), dims):
        E[order].groupby(seg[dim]).mean().T.plot.bar(ax=ax)
        ax.set(title=f"MAE by {dim}", ylabel=f"MAE ({panel.unit})")
        ax.legend(fontsize=7)
    savefig(fig, "analysis_e_by_segment")

    fig, ax = plt.subplots(figsize=(9, 4))
    for m in order[:6]:
        E[m].groupby(seg["hour"]).mean().plot(ax=ax, label=m)
    ax.set(xlabel="hour of target", ylabel="MAE", title="Error by hour of day (best 6 models)")
    ax.legend(fontsize=8)
    savefig(fig, "analysis_e_by_hour")

    best = order[0]
    per_sensor = E.groupby(seg["sensor"]).mean()
    rel = per_sensor[best] / y.groupby(seg["sensor"]).mean()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    rel.sort_values().plot.bar(ax=ax[0], color="C0")
    ax[0].set(xticks=[], xlabel="sensor (sorted)", ylabel="MAE / mean flow", title=f"{best}: relative error per sensor")
    base = "HistoricalAverage" if "HistoricalAverage" in per_sensor else per_sensor.columns[-1]
    win = (per_sensor[order].lt(per_sensor[base], axis=0)).mean().drop(base, errors="ignore")
    win.plot.bar(ax=ax[1], color="C2")
    ax[1].set(ylabel="share of sensors", title=f"Share of sensors where the model beats {base}")
    savefig(fig, "analysis_e_by_sensor")

    pred = joblib.load(MODELS_DIR / f"{best}.joblib").predict(X)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hexbin(y, pred, gridsize=60, mincnt=1, bins="log")
    lim = [0, float(np.nanpercentile(y, 99.5))]
    ax[0].plot(lim, lim, "r--")
    ax[0].set(xlabel="actual", ylabel="predicted", title=f"{best}: predicted vs actual")
    ax[1].hist(pred - y, bins=100)
    ax[1].set(xlabel="prediction - actual", title="Residuals")
    savefig(fig, "analysis_e_residuals")
    worst = rel.sort_values().tail(5)
    print(f"\n  worst sensors for {best} (MAE / mean flow): {worst.round(2).to_dict()}")


if __name__ == "__main__":
    main()
