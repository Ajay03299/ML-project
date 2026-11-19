"""Finish the real-data benchmark and efficiency run; keep exact command logs.

This is a local sequential runner, not a scheduler or remote service.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run_module(module, arguments, log_dir):
    command = [sys.executable, "-u", "-m", module] + arguments
    path = log_dir / (module.split(".")[-1] + ".log")
    print("[pipeline] " + " ".join(command), flush=True)
    environment = dict(
        os.environ,
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
    )
    with path.open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(command) + "\n")
        with subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as child:
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if child.wait() != 0:
                raise SystemExit(f"{module} failed; inspect {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-tuning", action="store_true")
    args = parser.parse_args()
    out = ROOT / "results/bhavesh/madrid_h60_development"
    marker = out / "boosting_rounds.png"
    if args.wait_for_tuning:
        deadline = time.monotonic() + 4 * 3600
        print(
            "[pipeline] Waiting for the validation-only boosting search to finish.",
            flush=True,
        )
        while not marker.exists():
            if time.monotonic() > deadline:
                raise SystemExit(
                    "Tuning did not finish within four hours; no benchmark was started."
                )
            time.sleep(5)
    elif not marker.exists():
        parser.error(
            "Complete experiments.bhavesh_boosting before starting the frozen benchmark."
        )
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    shared = [
        "--source",
        "madrid",
        "--horizon-min",
        "60",
        "--n-sensors",
        "60",
        "--max-train-rows",
        "200000",
    ]
    run_module("experiments.bhavesh_final_benchmark", shared + ["--threads", "2"], logs)
    run_module(
        "experiments.analysis_c_efficiency",
        shared + ["--repeats", "1", "--reuse-benchmark-fit", "--latency-repeats", "30", "--threads", "2"],
        logs,
    )
    print("[pipeline] Frozen benchmark and Analysis C completed.", flush=True)


if __name__ == "__main__":
    main()
