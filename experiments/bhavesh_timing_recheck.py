"""Repeat the one scaling fit interrupted by documented Windows system sleep.

Preserves the original observation; changes no settings, rows or test scores.
"""

import ctypes
import json
import os
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

from experiments.common import get_data, parser
from experiments.analysis_c_efficiency import nested_indices, plot_scaling, set_threads
from src.artifacts import experiment_info, file_hash, output_dir, write_json
from src.models import BEST_DIR, make_model


def main(argv=None):
    ap = parser(__doc__)
    args = ap.parse_args(argv)
    panel, splits = get_data(args)
    out = output_dir(args, panel)
    freeze = json.loads((out / "benchmark_freeze.json").read_text())
    info = experiment_info(panel, splits, args)
    name, count = "GradientBoosting", 25000
    if info["run_id"] != freeze["run_id"] or freeze["threads"] != 2:
        ap.error("This correction belongs to the original two-thread frozen run.")
    if file_hash(BEST_DIR / f"{name}.json") != freeze["parameter_file_hashes"][name]:
        ap.error("The selected parameters changed; do not mix timing studies.")
    resume = json.loads((out / "system_resume_audit.json").read_text(encoding="utf-8-sig"))
    events = resume if isinstance(resume, list) else [resume]
    if not any(event["Id"] == 1 and "low power state" in event["Message"] for event in events):
        ap.error("The documented system-resume event is required for this correction.")
    scaling = pd.read_csv(out / "scaling.csv")
    selected = scaling.model.eq(name) & scaling.n_train.eq(count)
    if selected.sum() != 1:
        ap.error("Expected one original 25,000-row GradientBoosting observation.")
    original = scaling.loc[selected].iloc[0]
    if original.evaluation_source != "fresh_validation_fit":
        ap.error("This observation has already been corrected.")
    backup = out / "scaling_original.csv"
    if not backup.exists():
        shutil.copy2(out / "scaling.csv", backup)
    indices = nested_indices(len(splits["X_train"]), [count])[count]
    X, y = splits["X_train"].iloc[indices], splits["y_train"].iloc[indices]
    model = set_threads(make_model(name, tuned=False, **freeze["best_params"][name]), 2)
    previous_state = None
    if os.name == "nt":
        keep_awake = ctypes.windll.kernel32.SetThreadExecutionState
        keep_awake.argtypes, keep_awake.restype = [ctypes.c_uint32], ctypes.c_uint32
        previous_state = keep_awake(0x80000001)
        if not previous_state:
            raise OSError("Could not keep this timed process awake; no primary data changed.")
    try:
        with threadpool_limits(limits=2):
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            model.fit(X, y)
            wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
            score = mean_absolute_error(splits["y_val"], model.predict(splits["X_val"]))
    finally:
        if previous_state is not None:
            keep_awake(previous_state)
    if wall > 2 * cpu + 10:
        raise RuntimeError("The repeat was heavily interrupted; primary data is unchanged.")
    if not np.isclose(score, float(original.val_MAE), rtol=1e-9):
        raise ValueError("The repeat did not reproduce the original validation score.")
    audit = {"model": name, "n_train": count, "run_id": info["run_id"],
             "reason": "Windows system sleep contaminated the original wall-clock fit time",
             "original_observation": original.to_dict(), "repeat_wall_s": wall,
             "repeat_process_cpu_s": cpu, "repeat_val_MAE": score,
             "same_validation_score": True, "parameter_file_sha256": freeze["parameter_file_hashes"][name],
             "system_resume_events": events, "test_used": False,
             "automatic_sleep_inhibition": "current timed process only; previous state restored"}
    write_json(out / "timing_recheck.json", audit)
    scaling.loc[selected, "original_train_time_s"] = float(original.train_time_s)
    scaling.loc[selected, "train_time_s"] = wall
    scaling.loc[selected, "evaluation_source"] = "remeasured_after_documented_system_sleep"
    scaling.to_csv(out / "scaling.csv", index=False)
    plot_scaling(scaling, panel.unit, "frozen benchmark audit", 1, out / "analysis_c_scaling.png")
    note = (f"\nTiming audit: the 25,000-row classic GB fit included a documented Windows sleep. "
            f"Its original {original.train_time_s:.3f} s observation is retained in scaling_original.csv "
            f"and scaling_evaluations.csv. One exact-row/seed/parameter repeat took {wall:.3f} s "
            f"({cpu:.3f} process CPU s) and reproduced validation MAE {score:.6f}. "
            "The primary scaling table/figure uses that repeat. No test scores or settings changed. "
            "See timing_recheck.json and system_resume_audit.json.\n")
    with (out / "interpretation.md").open("a", encoding="utf-8") as stream:
        stream.write(note)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
