"""EDA (2 marks): every figure and number the TA's EDA section asks for.   Owner: Ajay.

    python -m experiments.ajay_eda

Writes results/figures/eda_*.png and results/eda_summary.md. Put the figures in the report with
one sentence each saying what they show and why it is plausible for traffic.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.common import ROOT, parser, savefig
from src.data import load_panel


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    p = load_panel(args.source, n_sensors=args.n_sensors)
    f = pd.DataFrame(p.flow, index=p.index, columns=p.sensors)
    lines = ["# EDA summary\n",
             f"- source: {args.source}; sensors: {len(p.sensors)}; steps: {len(p.index):,} of {p.step_min} min",
             f"- period: {p.index[0]} to {p.index[-1]} ({(p.index[-1] - p.index[0]).days} days); unit: {p.unit}",
             f"- missing or faulty share (NaN after cleaning): {np.isnan(p.flow).mean():.2%}",
             f"- exact zeros: {(p.flow == 0).mean():.2%}; holidays: {0 if p.holidays is None else len(p.holidays)}",
             f"- flow distribution: median {np.nanmedian(p.flow):.0f}, p99 {np.nanpercentile(p.flow, 99):.0f}, "
             f"max {np.nanmax(p.flow):.0f}"]

    # 1 missingness per sensor and over time
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    f.isna().mean().sort_values().plot.bar(ax=ax[0], color="#4C72B0")
    ax[0].set(title="Share missing per sensor", xticks=[], xlabel="sensor (sorted)", ylabel="share")
    f.isna().mean(axis=1).resample("D").mean().plot(ax=ax[1], color="#C44E52")
    ax[1].set(title="Share of sensors missing, per day", ylabel="share")
    savefig(fig, "eda_missingness")

    # 2 daily profile: weekday vs weekend vs holiday
    hol = f.index.normalize().isin(p.holidays) if p.holidays is not None else np.zeros(len(f), bool)
    kind = np.where(hol, "holiday", np.where(f.index.day_of_week >= 5, "weekend", "weekday"))
    prof = f.mean(axis=1).groupby([kind, f.index.hour + f.index.minute / 60]).mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    prof.plot(ax=ax)
    ax.set(title="Average flow by time of day", xlabel="hour", ylabel=p.unit)
    savefig(fig, "eda_daily_profile")

    # 3 hour x weekday heatmap
    hm = f.mean(axis=1).groupby([f.index.day_of_week, f.index.hour]).mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(hm, aspect="auto", cmap="viridis")
    ax.set(yticks=range(7), yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], xlabel="hour",
           title="Mean flow, weekday x hour")
    fig.colorbar(im, ax=ax, label=p.unit)
    savefig(fig, "eda_heatmap_week_hour")

    # 4 differences between sensors
    stats = pd.DataFrame({"mean": f.mean(), "cv": f.std() / f.mean()})
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    stats["mean"].sort_values().plot.bar(ax=ax[0], color="#55A868")
    ax[0].set(title="Mean flow per sensor", xticks=[], xlabel="sensor (sorted)", ylabel=p.unit)
    if p.static is not None and {"latitude", "longitude"} <= set(p.static.columns):
        sc = ax[1].scatter(p.static["longitude"], p.static["latitude"], c=stats["mean"].to_numpy(), cmap="magma", s=18)
        fig.colorbar(sc, ax=ax[1], label=f"mean flow ({p.unit})")
        ax[1].set(title="Sensor map", xlabel="longitude", ylabel="latitude")
    savefig(fig, "eda_sensors")

    # 5 long-term trend and seasonality
    fig, ax = plt.subplots(figsize=(10, 3.2))
    f.mean(axis=1).resample("D").mean().plot(ax=ax, lw=0.8, label="daily mean")
    f.mean(axis=1).resample("D").mean().rolling(28, min_periods=7).mean().plot(ax=ax, lw=2, label="28-day mean")
    ax.set(title="Network-average flow over time", ylabel=p.unit)
    ax.legend()
    savefig(fig, "eda_trend")

    # 6 autocorrelation: why lags and "same time yesterday / last week" help
    s = f.mean(axis=1).interpolate(limit=8)
    per_day = 1440 // p.step_min
    lags = np.arange(1, 7 * per_day + 1)
    ac = [s.autocorr(int(l)) for l in lags[::2]]
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(lags[::2] / per_day, ac)
    ax.set(title="Autocorrelation of network flow", xlabel="lag (days)", ylabel="corr")
    savefig(fig, "eda_autocorrelation")

    # 7 weather effect
    if p.weather is not None and "precip" in p.weather:
        rain = p.weather["precip"].to_numpy() > 0
        day = (f.index.day_of_week < 5) & (f.index.hour >= 7) & (f.index.hour < 20)
        r = f[rain & day].mean().mean()
        d = f[~rain & day].mean().mean()
        lines.append(f"- weekday daytime mean flow: rain {r:.0f} vs dry {d:.0f} ({(r / d - 1):+.1%}); "
                     f"rainy share of steps {rain.mean():.1%}")

    # 8 neighbouring sensors: correlation vs distance
    if p.static is not None and {"latitude", "longitude"} <= set(p.static.columns):
        lat, lon = np.radians(p.static["latitude"].to_numpy(float)), np.radians(p.static["longitude"].to_numpy(float))
        h = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
             + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
        dist = 2 * 6371 * np.arcsin(np.sqrt(h))
        corr = f.corr(min_periods=500).to_numpy()
        iu = np.triu_indices(len(p.sensors), 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(dist[iu], corr[iu], s=4, alpha=0.4)
        ax.set(title="Flow correlation vs distance between sensors", xlabel="km", ylabel="correlation")
        savefig(fig, "eda_corr_vs_distance")

    out = ROOT / "results" / "eda_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
