"""Freeze validation-selected settings, then score each model once on the test period.

Controlled thread budget, shared evaluate_model and shared save_for_ui are used.
"""

import json
from pathlib import Path
from threadpoolctl import threadpool_limits

from experiments.common import get_data, parser, finalize
from experiments.analysis_c_efficiency import set_threads
from src.artifacts import experiment_info, output_dir, write_json, file_hash
from src.models import BASELINES, ML_MODELS, make_model, BEST_DIR, best_params


def main(argv=None):
    ap = parser(__doc__)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--models", nargs="+", choices=BASELINES + ML_MODELS,
                    default=BASELINES + ML_MODELS,
                    help="frozen contributor subset; the default remains the complete team benchmark")
    args = ap.parse_args(argv)
    if args.threads < 1:
        ap.error("threads must be positive")
    names = list(dict.fromkeys(args.models))
    required = [name for name in names if name not in BASELINES or name == "Ridge"]
    missing = [name for name in required if not (BEST_DIR / f"{name}.json").exists()]
    if missing:
        ap.error(
            f"Validation tuning must finish before opening test results: {missing}"
        )
    panel, splits = get_data(args)
    out = output_dir(args, panel)
    freeze = experiment_info(panel, splits, args)
    freeze.update(
        {
            "threads": args.threads,
            "purpose": "frozen real-data comparison; no test-based retuning",
            "models_requested": names,
            "models_excluded": sorted(set(BASELINES + ML_MODELS) - set(names)),
            "best_params": {name: best_params(name) for name in names},
            "parameter_file_hashes": {
                name: file_hash(BEST_DIR / f"{name}.json") for name in required
            },
        }
    )
    write_json(out / "benchmark_freeze.json", freeze)
    with threadpool_limits(limits=args.threads):
        for name in names:
            print(f"[frozen benchmark] {name}", flush=True)
            model = set_threads(make_model(name), args.threads)
            finalize(name, splits, panel, args, model=model)
    # A stable snapshot belongs to this contribution; the team's shared CSV stays ignored.
    import pandas as pd
    from src.evaluate import RESULTS_CSV

    frame = pd.read_csv(RESULTS_CSV)
    frame = frame[(frame["run_id"] == freeze["run_id"]) & (frame["tag"] == "all")
                  & frame["model"].isin(names)]
    frame.sort_values("test_MAE").to_csv(out / "frozen_benchmark.csv", index=False)
    print(f"Frozen benchmark complete: {out}", flush=True)


if __name__ == "__main__":
    main()
