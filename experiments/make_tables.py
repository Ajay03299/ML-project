"""Final consolidated results table for the report.   Owner: Anirudh.

    python -m experiments.make_tables

Reads results/benchmark.csv (tag "all") and writes results/benchmark_table.md: every model, same
test period, MAE / RMSE / R2 / WAPE, improvement over the best baseline, and training cost.
"""
import pandas as pd

from experiments.common import ROOT
from src.evaluate import RESULTS_CSV
from src.models import BASELINES


def main():
    b = pd.read_csv(RESULTS_CSV)
    b = b[b["tag"] == "all"].copy()
    base = b[b["model"].isin(BASELINES)].sort_values("test_MAE").iloc[0]
    b["vs_best_baseline_%"] = (1 - b["test_MAE"] / base["test_MAE"]) * 100
    b["kind"] = b["model"].map(lambda m: "baseline" if m in BASELINES else "ML")
    cols = ["model", "kind", "owner", "test_MAE", "test_RMSE", "test_R2", "test_WAPE", "val_MAE",
            "vs_best_baseline_%", "train_time_s", "predict_ms_per_1000", "model_size_kb"]
    t = b[cols].sort_values("test_MAE").round(3)
    try:
        table = t.to_markdown(index=False)
    except ImportError:                       # tabulate not installed
        table = t.to_string(index=False)
    md = [f"Best baseline: **{base['model']}** (test MAE {base['test_MAE']:.2f}). Positive "
          "`vs_best_baseline_%` = better than that baseline.\n", table]
    out = ROOT / "results" / "benchmark_table.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
