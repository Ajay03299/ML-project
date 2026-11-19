"""Bhavesh: controlled efficiency audit of the team's frozen benchmark.

Reuses recorded test scores; scaling uses validation only. Models and benchmark
must match the current data, split, horizon and features. No mixed-run plots.
"""
from __future__ import annotations

import json
import copy
import gc
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from experiments.common import ROOT, get_data, parser
from src.artifacts import experiment_info, file_hash, output_dir, write_json
from src.config import RANDOM_STATE
from src.evaluate import RESULTS_CSV, evaluate_model
from src.models import OWNERS

COSTS = {"train_time_s": "Training time (s)", "single_predict_ms": "Single-row prediction (ms)",
         "predict_ms_per_1000": "1,000-row prediction (ms)", "model_size_kb": "Serialized model (KiB)"}
SCALING_MODELS = ["Ridge", "DecisionTree", "RandomForest", "AdaBoost", "GradientBoosting",
                  "HistGradientBoosting", "XGBoost", "LightGBM"]


def pareto(df, x, y="test_MAE"):
    """Minimise both axes. Equal points remain; ties never keep a dominated row."""
    values = df[[x, y]].to_numpy(float)
    if not np.isfinite(values).all() or (values[:, 0] < 0).any():
        raise ValueError("Pareto costs/errors must be finite; costs must be non-negative.")
    keep = []
    for point in values:
        dominates = (values <= point).all(axis=1) & (values < point).any(axis=1)
        keep.append(not dominates.any())
    return df.loc[keep].sort_values([x, y])


def compatible_benchmark(bench, info):
    required = {"model", "tag", "horizon_min", "run_id", "source", "n_train", "n_features", "test_MAE"}
    if not required <= set(bench):
        raise ValueError("Benchmark has no complete provenance. Re-run experiments.run_benchmark.")
    selected = bench[(bench["tag"] == "all") & (bench["run_id"] == info["run_id"]) &
                     (bench["horizon_min"] == info["horizon_min"]) &
                     (bench["source"] == info["source"]) & (bench["n_train"] == info["n_train"]) &
                     (bench["n_features"] == len(info["columns"]))].copy()
    if selected.empty:
        raise ValueError("No benchmark matches this source, sensors, horizon, features and training sample.")
    if selected["model"].duplicated().any():
        raise ValueError("Duplicate compatible model rows in benchmark; rebuild it.")
    return selected


def set_threads(model, threads):
    if hasattr(model, "get_params"):
        settings = {k: threads for k in model.get_params(deep=True) if k.endswith("n_jobs")}
        if settings:
            model.set_params(**settings)
    return model


def time_prediction(model, X, repeats, batch_size):
    if X.empty:
        raise ValueError("Cannot measure latency on an empty validation split.")
    batch_size = min(batch_size, len(X))
    # Warm both paths; include DataFrame/estimator validation, exclude feature construction.
    model.predict(X.iloc[:1])
    model.predict(X.iloc[:batch_size])
    single, batch = [], []
    rng = np.random.default_rng(RANDOM_STATE)
    single_rows = rng.choice(len(X), size=repeats, replace=repeats > len(X))
    batch_starts = rng.integers(0, len(X) - batch_size + 1, size=repeats)
    for repeat in range(repeats):
        start_row = int(batch_starts[repeat])
        start = time.perf_counter_ns()
        model.predict(X.iloc[[int(single_rows[repeat])]])
        single.append((time.perf_counter_ns() - start) / 1e6)
        start = time.perf_counter_ns()
        model.predict(X.iloc[start_row:start_row + batch_size])
        batch.append((time.perf_counter_ns() - start) / 1e6)
    return single, batch, batch_size


def nested_indices(total, sizes, seed=RANDOM_STATE):
    permutation = np.random.default_rng(seed).permutation(total)
    return {n: np.sort(permutation[:n]) for n in sorted(set(sizes)) if 0 < n <= total}


def plot_costs(summary, unit, title, path):
    """A shared legend stays readable even where many model errors are similar."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    palette = plt.get_cmap("tab20")
    markers = ["o", "s", "^", "v", "D", "P", "X"]
    handles = []
    for ax_index, (ax, (cost, label)) in enumerate(zip(axes.ravel(), COSTS.items())):
        for index, (_, row) in enumerate(summary.iterrows()):
            point = ax.scatter(max(row[cost], 1e-6), row["test_MAE"], color=palette(index),
                               marker=markers[index % len(markers)], s=48, label=row["model"],
                               edgecolors="black", linewidths=0.35, zorder=3)
            if ax_index == 0:
                handles.append(point)
        front = pareto(summary, cost)
        ax.plot(front[cost].clip(lower=1e-6), front["test_MAE"], "--", color="black",
                alpha=0.65, label="Pareto front")
        ax.set(xscale="log", xlabel=label + " (log)", ylabel=f"Frozen test MAE ({unit})")
        ax.legend(handles=[ax.lines[-1]], fontsize=10)
        ax.grid(alpha=0.15)
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--repeats", type=int, default=3, help="fresh training fits per size")
    ap.add_argument("--latency-repeats", type=int, default=20)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reuse-benchmark-fit", action="store_true",
                    help="use one controlled frozen benchmark fit as training cost; requires --repeats 1")
    ap.add_argument("--scaling-models", nargs="+", default=None)
    args = ap.parse_args(argv)
    if min(args.repeats, args.latency_repeats, args.threads, args.max_train_rows, args.n_sensors) < 1:
        ap.error("repetitions, threads, row limits, horizon and sensor count must be positive")
    if args.reuse_benchmark_fit and args.repeats != 1:
        ap.error("--reuse-benchmark-fit requires --repeats 1; it is a single observed fit time")
    if not RESULTS_CSV.exists() or not (ROOT / "models/meta.json").exists():
        ap.error("Run experiments.run_benchmark for this exact setup first.")
    panel, splits = get_data(args)
    info = experiment_info(panel, splits, args)
    saved_info = json.loads((ROOT / "models/meta.json").read_text(encoding="utf-8"))
    if saved_info.get("run_id") != info["run_id"]:
        ap.error("Saved models have a different setup. Re-run benchmark with the same arguments.")
    bench = compatible_benchmark(pd.read_csv(RESULTS_CSV), info)
    out = output_dir(args, panel)
    if args.reuse_benchmark_fit:
        freeze_path = out / "benchmark_freeze.json"
        if not freeze_path.exists():
            ap.error("A controlled bhavesh_final_benchmark freeze is required to reuse training times")
        freeze = json.loads(freeze_path.read_text())
        if (freeze["run_id"] != info["run_id"] or freeze["threads"] != args.threads
                or freeze["environment"] != info["environment"]):
            ap.error("Benchmark data/thread settings differ from the requested efficiency audit")
    info.update({"threads": args.threads, "repeats": args.repeats,
                 "training_time_source": "controlled frozen benchmark" if args.reuse_benchmark_fit else "fresh refits",
                 "missing_benchmark_models": sorted(set(OWNERS) - set(bench["model"])),
                 "latency_repeats": args.latency_repeats, "scaling_score": "validation",
                 "latency_sampling": "seed-42 uniform validation rows and batch starts, identical for every model",
                 "test_scores": "reused from frozen benchmark; never re-evaluated by Analysis C",
                 "measurement": "warm estimator predict, including pandas slicing; feature building excluded"})
    write_json(out / "efficiency_manifest.json", info)
    # Deliberately omit X_test/y_test so fit/scaling cannot inspect test labels.
    validation_splits = {k: v for k, v in splits.items() if k.endswith(("_train", "_val"))}
    raw, summaries, templates = [], [], {}
    full_fit_times = {}
    with threadpool_limits(limits=args.threads):
        for _, row in bench.iterrows():
            name = row["model"]
            path = ROOT / "models" / f"{name}.joblib"
            record = saved_info.get("model_runs", {}).get(name, {})
            if record.get("run_id") != info["run_id"] or not path.exists() or record.get("sha256") != file_hash(path):
                raise ValueError(f"Missing or altered model artifact: {name}. Re-run benchmark.")
            model = set_threads(joblib.load(path), args.threads)
            templates[name] = clone(model) if hasattr(model, "get_params") else copy.deepcopy(model)
            single, batch, batch_n = time_prediction(model, splits["X_val"], args.latency_repeats, 1000)
            del model
            gc.collect()
            times = []
            for repeat in range(args.repeats):
                if args.reuse_benchmark_fit:
                    elapsed = float(row["train_time_s"])
                    times.append(elapsed)
                    raw.append({"model": name, "phase": "fit", "repeat": repeat,
                                "seconds": elapsed, "source": "controlled frozen benchmark"})
                    continue
                template = templates[name]
                fresh = clone(template) if hasattr(template, "get_params") else copy.deepcopy(template)
                fresh = set_threads(fresh, args.threads)
                start = time.perf_counter()
                fresh.fit(splits["X_train"], splits["y_train"])
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                raw.append({"model": name, "phase": "fit", "repeat": repeat, "seconds": elapsed})
                del fresh
                gc.collect()
            full_fit_times[name] = times
            for repeat, (one, many) in enumerate(zip(single, batch)):
                raw.append({"model": name, "phase": "predict", "repeat": repeat,
                            "single_ms": one, "batch_ms": many, "batch_rows": batch_n})
            summaries.append({"model": name, "test_MAE": row["test_MAE"], "val_MAE": row["val_MAE"],
                              "train_time_s": np.median(times),
                              "train_time_p95_s": np.percentile(times, 95) if len(times) > 1 else np.nan,
                              "single_predict_ms": np.median(single), "single_p95_ms": np.percentile(single, 95),
                              "predict_ms_per_1000": np.median(batch) * 1000 / batch_n,
                              "batch_p95_ms": np.percentile(batch, 95), "batch_rows": batch_n,
                              "model_size_kb": path.stat().st_size / 1024, "n_train": len(splits["X_train"]),
                              "horizon_min": info["horizon_min"], "run_id": info["run_id"]})
            print(f"[efficiency] {name}: {np.median(single):.3f} ms single, fit {np.median(times):.3f}s", flush=True)
            pd.DataFrame(summaries).to_csv(out / "efficiency_progress.csv", index=False)
            pd.DataFrame(raw).to_csv(out / "efficiency_measurements.csv", index=False)
        summary = pd.DataFrame(summaries).sort_values("test_MAE")
        for cost in COSTS:
            summary[f"pareto_{cost}"] = summary.index.isin(pareto(summary, cost).index)
        summary.to_csv(out / "efficiency.csv", index=False)
        pd.DataFrame(raw).to_csv(out / "efficiency_measurements.csv", index=False)
        label = "SMOKE TEST ONLY" if args.quick or args.source == "synthetic" else "frozen benchmark audit"
        plot_costs(summary, panel.unit,
                   f"{args.source} | {info['horizon_min']} min | {args.threads} thread(s) | {label}",
                   out / "analysis_c_cost_vs_error.png")
        total = len(splits["X_train"])
        fractions = [0.25, 0.5, 1.0] if args.quick else [0.0625, 0.125, 0.25, 0.5, 1.0]
        samples = nested_indices(total, [max(1, int(total * f)) for f in fractions])
        names = args.scaling_models or (["Ridge", "DecisionTree"] if args.quick else SCALING_MODELS)
        missing = set(names) - set(templates)
        if args.scaling_models and missing:
            ap.error(f"Scaling models are missing from this benchmark: {sorted(missing)}")
        scaling = []
        for n, idx in samples.items():
            sub = dict(validation_splits, X_train=splits["X_train"].iloc[idx], y_train=splits["y_train"].iloc[idx])
            for name in names:
                if name not in templates:
                    continue
                for repeat in range(args.repeats):
                    # The full-size fits were already measured above. Reuse
                    # their times and frozen validation score; smaller sizes
                    # get new fits. All parameters and the training rows match.
                    if n == total:
                        score = float(bench.loc[bench["model"] == name, "val_MAE"].iloc[0])
                        scaling.append({"model": name, "n_train": n, "repeat": repeat,
                                        "train_time_s": full_fit_times[name][repeat], "val_MAE": score,
                                        "evaluation_source": "full_fit_timing_and_frozen_validation"})
                        continue
                    template = templates[name]
                    model = clone(template) if hasattr(template, "get_params") else copy.deepcopy(template)
                    model = set_threads(model, args.threads)
                    result = evaluate_model(name, model, sub, tag=f"scale_{n}_{repeat}",
                                            horizon_min=info["horizon_min"],
                                            results_path=out / "scaling_evaluations.csv",
                                            extra={"run_id": info["run_id"], "source": args.source})
                    scaling.append({"model": name, "n_train": n, "repeat": repeat,
                                    "train_time_s": result["train_time_s"], "val_MAE": result["val_MAE"],
                                    "evaluation_source": "fresh_validation_fit"})
                    print(f"[scaling] {name}: rows={n}, repeat={repeat + 1}, val MAE={result['val_MAE']:.3f}", flush=True)
                    pd.DataFrame(scaling).to_csv(out / "scaling_progress.csv", index=False)
                    del model
                    gc.collect()
        sc = pd.DataFrame(scaling)
        if sc.empty:
            ap.error("None of the scaling models was benchmarked. Include Ridge/DecisionTree or specify models.")
        sc.to_csv(out / "scaling.csv", index=False)
        plot_scaling(sc, panel.unit, label, args.repeats, out / "analysis_c_scaling.png")
    write_interpretation(out, summary, sc, info, label, missing)
    print(f"Artifacts: {out}")


def plot_scaling(sc, unit, label, repeats, path):
    """Render saved observations without refitting; keep the legend off the data."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for name, group in sc.groupby("model"):
        g = group.groupby("n_train").agg(train_time_s=("train_time_s", "median"),
                                        min_time=("train_time_s", "min"), max_time=("train_time_s", "max"),
                                        val_MAE=("val_MAE", "median"))
        axes[0].plot(g.index, g["train_time_s"], "o-", label=name)
        if repeats > 1:
            axes[0].fill_between(g.index, g["min_time"], g["max_time"], alpha=0.12)
        axes[1].plot(g.index, g["val_MAE"], "o-", label=name)
    timing_label = "Fit seconds (single run)" if repeats == 1 else "Fit seconds (median, min-max band)"
    axes[0].set(xscale="log", yscale="log", xlabel="Nested training sample rows", ylabel=timing_label)
    axes[1].set(xscale="log", xlabel="Nested training sample rows", ylabel=f"Validation MAE ({unit})")
    for ax in axes:
        ax.set_xticks(sorted(sc.n_train.unique()))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}k"))
        ax.minorticks_off()
    handles, names = axes[1].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, fontsize=9, frameon=False)
    corrected = "original_train_time_s" in sc and sc.original_train_time_s.notna().any()
    title = f"Scaling: same chronological coverage, fixed validation, {label}"
    if corrected:
        title += "\n25k classic GB timing repeated after documented Windows sleep; original retained"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.15, 1, 0.88 if corrected else 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_interpretation(out, summary, scaling, info, label, missing):
    best = summary.sort_values("test_MAE").iloc[0]
    eligible = summary[summary["test_MAE"] <= best["test_MAE"] * 1.05]
    candidate = eligible.sort_values("single_predict_ms").iloc[0]
    qualification = (
        "Synthetic figures are software checks only and do not support Madrid findings."
        if info["source"] == "synthetic" else
        f"These findings apply to the frozen {len(info['sensor_ids'])}-sensor {info['source']} experiment, "
        f"at {info['step_min']}-minute sampling and a {info['horizon_min']}-minute target. "
        "Other sensor sets, cities and forecast horizons require separate evaluation."
    )
    training_description = (
        f"Training times are single-run observations from the {info['training_time_source']}; "
        "no training-time confidence interval or p95 is estimated."
        if info["repeats"] == 1 else
        f"Fit times are medians of {info['repeats']} identical-seed fresh fits; variation is "
        "machine timing noise, not independent estimates of generalisation."
    )
    text = f"""# Analysis C - {label}

Source: {info['source']}; horizon: {info['horizon_min']} minutes; unit: {info['unit']}.
Run: {info['run_id']}. Fixed training rows: {info['n_train']}; threads: {info['threads']}.

Observed lowest frozen test MAE: {best['model']} ({best['test_MAE']:.3f}).
Within 5% of that error, the fastest measured median single-row predictor was
{candidate['model']} ({candidate['single_predict_ms']:.3f} ms; p95 {candidate['single_p95_ms']:.3f} ms;
{candidate['model_size_kb']:.1f} KiB). This is a descriptive candidate under that
explicit error tolerance, not a universal deployment recommendation. Check the
separate Pareto fronts for training time, latency and storage before choosing.

{training_description} Warm latency has {info['latency_repeats']} repetitions;
p95 is descriptive with this sample size.
Batch time is measured on at most 1,000 validation rows; normalised per-1,000
throughput is not single-request latency. Model size is uncompressed joblib file
size, not peak RAM. The recorded environment is in efficiency_manifest.json.

Scaling uses nested, fixed-seed samples spanning the same training period and a
fixed validation period. It measures row-count scaling, not shorter-history
robustness. Parameters stay frozen. The largest-size timing repetitions and frozen validation score are reused
from the full-size audit to avoid duplicate fits; smaller sizes receive new fits.
Test scores are reused from the benchmark;
no scaling run reads test labels. Models absent from scaling: {sorted(missing)}.
Models absent from the complete team benchmark: {info['missing_benchmark_models']}.

Feature construction, data ingestion, network overhead, concurrent requests and
cold model loading are excluded. No measured estimator latency establishes an
end-to-end real-time SLA. {qualification}
"""
    (out / "interpretation.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
