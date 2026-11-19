# Bhavesh real-data contribution - start here

The Madrid panel is now merged in main at commit 358297a. This worktree,
`traffic-volume-prediction-real`, integrates your implementation on branch
`bhavesh/real-data-evaluation`; the earlier synthetic working copy is preserved.

Your assigned work is AdaBoost, classic Gradient Boosting, HistGradientBoosting,
Analysis C (efficiency), Streamlit UI and those report sections. Teammates retain
their EDA, Random Forest/PCA/ablation, XGBoost/LightGBM/error-analysis ownership.
Supporting tuning of their model families is recorded as comparator preparation,
not authorship/completion of their separate analyses.

## Critical corrections from the starter

- The actual panel is **hourly**, 300 sensors, June 2022 to October 2023. The
  main target is **60 minutes = one step**. A 15-minute experiment cannot be
  evaluated from these observations and is explicitly rejected. Do not run
  `--horizon 4` expecting one hour: on this data that means four hours.
- The shared weather array has no fill mask, and the old loader allowed backward
  fill. Since raw CSV/Parquet is unavailable, every compared model excludes
  unverified weather. Traffic, calendar, sensor summaries and road features
  remain available. No dataset bytes are edited or relabelled as verified.
- There are 29 features, 60 selected sensors, 200,000 supervised training rows,
  106,160 validation rows and 105,765 held-out test rows. Feature statistics use
  the full training period; supervised model fitting uses the row budget.
- The inherited long-zero cleanup is retrospective archive quality screening
  with a 12-step threshold (12 hours here), not an online fault detector.

The full reasoning, fixed search budgets, split dates and limitations are in
[BHAVESH_REAL_RUN_PROTOCOL.md](BHAVESH_REAL_RUN_PROTOCOL.md).

## Reproduce

On this computer the installed interpreter is
`../traffic-volume-prediction/.venv/Scripts/python.exe`. On another computer,
create a virtual environment and install requirements.txt. All commands run
from the real-data repository root.

```powershell
python -m pytest -q
python -m experiments.bhavesh_boosting --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000 --tune-rows 30000 --candidates 10 --publish-params
python -m experiments.bhavesh_reference_tuning --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000 --tune-rows 30000 --threads 2
python -m experiments.bhavesh_final_benchmark --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000 --threads 2
python -m experiments.analysis_c_efficiency --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000 --repeats 1 --reuse-benchmark-fit --latency-repeats 30 --threads 2
python -m experiments.bhavesh_consistency --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000
python -m streamlit run app/streamlit_app.py
```

Limit BLAS/OpenMP to two threads as described in the run protocol when comparing
costs. The runner `python scripts/finish_bhavesh_real.py` logs the controlled
benchmark and Analysis C commands after tuning. Long fits are expected: a
training-only pilot estimates over 3 GiB for the selected 200-tree full forest.
The UI loads models lazily; the evaluator uses disk-backed size measurement;
Analysis C keeps one large fitted model at a time.

## What to review

- `experiments/bhavesh_boosting.py`: validation-only configuration and round
  selection; full validation stage curves, cached exact AdaBoost medians.
- `experiments/analysis_c_efficiency.py`: single controlled training times,
  repeated warm latency, serialized size, Pareto fronts and nested
  validation-only scaling. Optional fresh repeated fits are also supported.
- `app/streamlit_app.py`, `app/forecast_data.py`: saved-model forecasts, recent
  history, model comparison, day errors, map and CSV export.
- `src/artifacts.py`, `experiments/common.py`: artifact identity/schema/horizon
  checks. Existing callers remain compatible; tune accepts an optional out_dir
  to isolate supporting searches from the shared owners' tables.
- `src/data.py`, `src/evaluate.py`: safe legacy-weather exclusion, bounded
  causal filling for new panels, disk-backed serialization.
- `experiments/bhavesh_reference_tuning.py`, `bhavesh_final_benchmark.py` and
  `bhavesh_consistency.py`: explicit reference tuning, freeze and descriptive
  post-freeze gain consistency. No test-based retuning.

## Evidence locations

`results/bhavesh/madrid_h60_development/` contains the real tuning CSVs, exact
round curves, selected settings, freeze manifest, frozen benchmark, efficiency
measurements and figures. Its historical folder suffix does not mean synthetic:
the manifests record source=madrid and quick=false.

`results/best_params/` contains the canonical parameter files to review in the
PR. The shared results/benchmark.csv and models/ stay out of Git. The full forest
can be several GiB; do not add it to a commit. The already-tracked merged data
is preserved, not added anew by this contribution.

The measured report sections are in `output/pdf/Bhavesh_RealData_Report.pdf`
and editable `docs/BHAVESH_REPORT_SECTIONS.md`. The separate evidence PDF
contains full tables/figures and real UI screenshots. The screen demonstration
is `output/demo/Bhavesh_Madrid_Demo.webm`. Browser CSVs and verification records
are under `docs/screenshots/`.

To regenerate browser evidence, install `requirements-artifacts.txt` and run
`python -m playwright install chromium`. Start Streamlit on port 8507, then run
`python scripts/verify_bhavesh_real_ui.py --record`. Export the recording to an
approximately two-minute `output/demo/Bhavesh_Madrid_Demo.webm`, then run
`python scripts/build_bhavesh_real_report.py`. Finally,
`python scripts/package_bhavesh_real.py` creates source/evidence and compact
demo ZIPs in the parent workspace's `output/` folder. Report/package builders
reject missing or mismatched browser evidence.

The application is honest held-out historical replay, not live ingestion. Each
day-chart point uses its own origin history; midnight targets correctly use a
previous-day origin. Missing rows stay missing. Map tiles depend on network
access, with a coordinate/flow table available alongside the map. The feature
inspector shows inputs, not causal SHAP attribution. Optional SHAP is not needed
to claim the implemented mandatory UI extras.

Pragun's RF/PCA results were merged into main at 0281cf1 after this experiment
finished. His three selected configurations match this frozen study. His shared
tuning tables/figures are retained; this contribution's seven supporting searches
are under `results/bhavesh/madrid_h60_development/references/`. Reference searches
now write there by default; --publish-params is an explicit option when changing
the shared selected settings. Scores from separate runs must not be combined.

## Individual work still required from you

Read the measured findings and practise the [viva guide](BHAVESH_VIVA.md).
You must personally explain the choices in the individual viva. The team must
review and approve the PR before merge, and integrate these sections with the
other members' analyses A/E and full report; this contribution does not claim
to replace their work.
