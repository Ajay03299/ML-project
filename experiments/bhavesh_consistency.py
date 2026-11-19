"""Describe frozen booster gains over a validation-selected baseline.

No fitting or retuning. Recomputed fixed predictions must match the benchmark.
Weekly moving-block intervals are descriptive within this one held-out period.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_limits

from experiments.common import get_data, parser, ROOT
from experiments.analysis_c_efficiency import compatible_benchmark, set_threads
from src.artifacts import experiment_info, output_dir, file_hash
from src.evaluate import RESULTS_CSV
from src.models import BASELINES


def weekly_gain_interval(gains, seed=42, repetitions=2000, block_days=7):
    gains = np.asarray(gains, dtype=float)
    if len(gains) < 2 * block_days or not np.isfinite(gains).all():
        raise ValueError(
            "At least two complete finite blocks of daily gains are required."
        )
    rng = np.random.default_rng(seed)
    starts = rng.integers(
        0, len(gains), size=(repetitions, int(np.ceil(len(gains) / block_days)))
    )
    indices = (starts[:, :, None] + np.arange(block_days)) % len(gains)
    sampled = gains[indices.reshape(repetitions, -1)[:, : len(gains)]].mean(axis=1)
    return np.quantile(sampled, [0.025, 0.975])


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    panel, splits = get_data(args)
    info = experiment_info(panel, splits, args)
    bench = compatible_benchmark(pd.read_csv(RESULTS_CSV), info)
    baseline = (
        bench[bench["model"].isin(BASELINES)].sort_values("val_MAE").iloc[0]["model"]
    )
    names = [baseline, "AdaBoost", "GradientBoosting", "HistGradientBoosting"]
    errors = splits["meta_test"].copy()
    errors["date"] = errors["target_time"].dt.normalize()
    metadata = json.loads((ROOT / "models/meta.json").read_text())
    if metadata["run_id"] != info["run_id"]:
        raise ValueError("Saved models and benchmark differ.")
    target = splits["y_test"].to_numpy(dtype=float)
    with threadpool_limits(limits=2):
        for name in names:
            path = ROOT / "models" / f"{name}.joblib"
            if file_hash(path) != metadata["model_runs"][name]["sha256"]:
                raise ValueError(f"Changed artifact: {name}")
            model = set_threads(joblib.load(ROOT / "models" / f"{name}.joblib"), 2)
            prediction = np.asarray(model.predict(splits["X_test"]), dtype=float)
            errors[name] = np.abs(target - prediction)
            measured = float(errors[name].mean())
            expected = float(bench.loc[bench["model"] == name, "test_MAE"].iloc[0])
            if not np.isclose(measured, expected, rtol=1e-7):
                raise ValueError(
                    f"{name} predictions do not reproduce its frozen score."
                )
            del model
    out = output_dir(args, panel)
    daily = errors.groupby("date")[names].mean()
    sensors = errors.groupby("sensor")[names].mean()
    daily.to_csv(out / "frozen_daily_mae.csv")
    sensors.to_csv(out / "frozen_sensor_mae.csv")
    rows = []
    for name in names[1:]:
        gains = daily[baseline] - daily[name]
        low, high = weekly_gain_interval(gains)
        rows.append(
            {
                "model": name,
                "run_id": info["run_id"],
                "baseline_selected_on_validation": baseline,
                "test_MAE": float(errors[name].mean()),
                "relative_MAE_improvement": 1
                - float(errors[name].mean() / errors[baseline].mean()),
                "days_improved": int((gains > 0).sum()),
                "n_days": len(daily),
                "sensors_improved": int((sensors[name] < sensors[baseline]).sum()),
                "n_sensors": len(sensors),
                "equal_day_mean_MAE_gain": float(gains.mean()),
                "weekly_block_gain_low": low,
                "weekly_block_gain_high": high,
                "block_days": 7,
                "bootstrap_repetitions": 2000,
                "seed": 42,
            }
        )
    pd.DataFrame(rows).to_csv(out / "frozen_gain_consistency.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    for name in names[1:]:
        ax.plot(daily.index, daily[baseline] - daily[name], label=name, linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(title=f"Frozen test: daily gain over validation-selected {baseline}",
           xlabel="Target date", ylabel=f"Baseline MAE minus booster MAE ({info['unit']})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "frozen_daily_gains.png", dpi=160)
    plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
