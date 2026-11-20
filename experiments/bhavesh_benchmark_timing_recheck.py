"""Repeat one interrupted benchmark fit without retuning or rescoring test labels.

Keep the original observations and update the derived cost plots only after the
same training rows/parameters reproduce the frozen validation score.
"""
import ctypes
import json
import os
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from experiments.common import get_data, parser
from experiments.analysis_c_efficiency import plot_costs, plot_scaling, set_threads, pareto, COSTS, write_interpretation
from src.artifacts import experiment_info, file_hash, output_dir, write_json
from src.evaluate import RESULTS_CSV
from src.models import BEST_DIR, ML_MODELS, make_model


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--model", choices=ML_MODELS, required=True)
    args = ap.parse_args(argv)
    if args.quick or args.source != "madrid":
        ap.error("A timing correction requires the frozen real Madrid setup.")
    panel, splits = get_data(args)
    out = output_dir(args, panel)
    freeze = json.loads((out / "benchmark_freeze.json").read_text())
    info = experiment_info(panel, splits, args)
    name = args.model
    if (info["run_id"] != freeze["run_id"] or info["environment"] != freeze["environment"]):
        ap.error("The data, rows, features and environment must match the frozen run.")
    if name not in freeze["parameter_file_hashes"] or name not in freeze["best_params"]:
        ap.error(f"{name} was not included in this frozen comparison.")
    if file_hash(BEST_DIR / f"{name}.json") != freeze["parameter_file_hashes"][name]:
        ap.error("Parameters changed; no timing data was replaced.")
    events = json.loads((out / "system_resume_audit.json").read_text(encoding="utf-8-sig"))
    if not isinstance(events, list) or not any(e["Id"] in (1, 107) for e in events):
        ap.error("Documented system sleep/clock events are required.")
    bench = pd.read_csv(out / "frozen_benchmark.csv")
    selected = bench.model.eq(name) & bench.run_id.eq(info["run_id"])
    if selected.sum() != 1:
        ap.error("Expected exactly one frozen model observation.")
    original = bench.loc[selected].iloc[0].copy()
    audit_path = out / f"{name}_benchmark_timing_recheck.json"
    if audit_path.exists():
        ap.error("A timing repeat is already recorded; preserve it rather than overwrite it.")
    shared = pd.read_csv(RESULTS_CSV)
    summary = pd.read_csv(out / "efficiency.csv")
    scaling = pd.read_csv(out / "scaling.csv")
    raw = pd.read_csv(out / "efficiency_measurements.csv")
    manifest = json.loads((out / "efficiency_manifest.json").read_text())
    if manifest["run_id"] != info["run_id"]:
        ap.error("The efficiency audit belongs to a different run.")
    if ((shared.model.eq(name) & shared.run_id.eq(info["run_id"])).sum() != 1
            or (summary.model.eq(name) & summary.run_id.eq(info["run_id"])).sum() != 1
            or (scaling.model.eq(name) & scaling.n_train.eq(info["n_train"])).sum() != 1):
        ap.error("Expected one matching benchmark, efficiency and full-size scaling observation.")
    # Omit test labels from the fit and validation check.
    data = {k: v for k, v in splits.items() if k.endswith(("_train", "_val"))}
    model = set_threads(make_model(name, tuned=False, **freeze["best_params"][name]), freeze["threads"])
    previous = None
    if os.name == "nt":
        request = ctypes.windll.kernel32.SetThreadExecutionState
        request.argtypes, request.restype = [ctypes.c_uint32], ctypes.c_uint32
        previous = request(0x80000001)
        if not previous:
            raise OSError("Could not inhibit idle sleep; no timing data was replaced.")
    try:
        with threadpool_limits(limits=freeze["threads"]):
            print(f"[timing repeat] {name}: {len(data['X_train']):,} fixed training rows", flush=True)
            started = datetime.now(timezone.utc)
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            model.fit(data["X_train"], data["y_train"])
            wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
            finished = datetime.now(timezone.utc)
            score = float(mean_absolute_error(data["y_val"], model.predict(data["X_val"])))
    finally:
        if previous is not None:
            request(previous)
    if wall > 2 * cpu + 10:
        raise RuntimeError("The repeat was heavily interrupted; primary tables are unchanged.")
    if abs((finished - started).total_seconds() - wall) > 10:
        raise RuntimeError("System clock/suspend gap detected; primary tables are unchanged.")
    if not np.isclose(score, float(original.val_MAE), rtol=1e-9):
        raise ValueError("The timing repeat did not reproduce the frozen validation score.")
    source = "same-setup timing repeat after documented system interruption"
    audit = {"model": name, "run_id": info["run_id"], "n_train": len(data["X_train"]),
             "threads": freeze["threads"], "original_observation": original.to_dict(),
             "repeat_wall_s": wall, "repeat_process_cpu_s": cpu, "repeat_val_MAE": score,
             "started_at_utc": started.isoformat(), "finished_at_utc": finished.isoformat(),
             "same_validation_score": True, "test_used": False, "saved_model_replaced": False,
             "parameter_file_sha256": freeze["parameter_file_hashes"][name],
             "reason": "Original fit spanned Windows sleep and hardware-clock corrections",
             "system_events": events, "automatic_sleep_inhibition": "current timed process; previous state restored"}
    write_json(audit_path, audit)
    for path, frame in ((out / "frozen_benchmark.csv", bench), (RESULTS_CSV, shared)):
        keep = frame.model.eq(name) & frame.run_id.eq(info["run_id"])
        if keep.sum() != 1:
            raise ValueError(f"Expected one compatible row in {path}")
        backup = out / (path.stem + "_before_timing_recheck.csv")
        if not backup.exists():
            shutil.copy2(path, backup)
        frame.loc[keep, "train_time_s"] = wall
        frame["train_time_source"] = frame.get("train_time_source", "controlled frozen benchmark")
        frame.loc[keep, "train_time_source"] = source
        frame.to_csv(path, index=False)
    summary.loc[summary.model.eq(name), "train_time_s"] = wall
    summary["train_time_source"] = summary.get("train_time_source", "controlled frozen benchmark")
    summary.loc[summary.model.eq(name), "train_time_source"] = source
    for cost in COSTS:
        summary[f"pareto_{cost}"] = summary.index.isin(pareto(summary, cost).index)
    if not (out / "efficiency_before_timing_recheck.csv").exists():
        shutil.copy2(out / "efficiency.csv", out / "efficiency_before_timing_recheck.csv")
    summary.to_csv(out / "efficiency.csv", index=False)
    keep = scaling.model.eq(name) & scaling.n_train.eq(info["n_train"])
    if keep.sum() != 1:
        raise ValueError("Expected one full-size scaling observation.")
    if not (out / "scaling_before_timing_recheck.csv").exists():
        shutil.copy2(out / "scaling.csv", out / "scaling_before_timing_recheck.csv")
    scaling.loc[keep, "train_time_s"] = wall
    scaling.loc[keep, "evaluation_source"] = "full_fit_repeated_after_documented_interruption"
    scaling.to_csv(out / "scaling.csv", index=False)
    pd.concat([raw, pd.DataFrame([{"model": name, "phase": "fit_recheck", "repeat": 0,
                                  "seconds": wall, "source": source}])], ignore_index=True).to_csv(
        out / "efficiency_measurements.csv", index=False)
    manifest["training_time_source"] = "single controlled fits with documented timing repeats"
    manifest.setdefault("timing_rechecks", {})[name] = audit_path.name
    freeze.setdefault("timing_rechecks", {})[name] = audit_path.name
    write_json(out / "efficiency_manifest.json", manifest)
    write_json(out / "benchmark_freeze.json", freeze)
    label = "frozen benchmark audit"
    plot_costs(summary, panel.unit, f"madrid | {info['horizon_min']} min | {freeze['threads']} thread(s) | {label}",
               out / "analysis_c_cost_vs_error.png")
    plot_scaling(scaling, panel.unit, label, 1, out / "analysis_c_scaling.png")
    write_interpretation(out, summary, scaling, manifest, label, set())
    with (out / "interpretation.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\nTiming audit: {name}'s original fit spanned documented Windows sleep/clock corrections. "
                     f"Its original {float(original.train_time_s):.3f} s observation is retained in "
                     f"{audit_path.name} and the *_before_timing_recheck.csv backups. "
                     f"The same-row/seed/parameter repeat took {wall:.3f} s ({cpu:.3f} CPU s) "
                     "and reproduced the frozen validation score. Only its fit cost and derived cost plots "
                     "were replaced. Test scores and saved models are unchanged.\n")
    print(f"[timing repeat] {name}: wall {wall:.3f}s, CPU {cpu:.3f}s, validation MAE {score:.9f}", flush=True)


if __name__ == "__main__":
    main()
