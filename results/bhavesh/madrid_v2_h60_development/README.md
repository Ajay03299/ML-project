# Madrid v2 contributor rerun

Base data: Ajay's v2 cache merged in PR #5, at main commit
`63e9b4025cca149c76343ee0e5dd6bfc62bd10bc` after PR #6. The cache declares
`preprocessing_version=2`. Raw CSV/Parquet files were unavailable here, so this
contribution checks the shared cache and loader regressions rather than independently
certifying the raw rebuild.

The traffic matrix, timestamps, sensor ordering, holidays and split row hashes
match the earlier no-weather contribution. The weather array also has identical
values; its v2 provenance marker allows the loader to include `wx_temp`,
`wx_precip` and `wx_wind`, which were excluded from the legacy-cache experiments.
There are 32 features instead of 29. See `data_audit.json` for input hashes and
the exact feature schema. A comparison with the old results therefore concerns
restored weather features and renewed validation tuning, not a new independent
test period or an isolated causal estimate of weather's effect.

The setup remains 60 evenly spaced sensors, an hourly grid, a 60-minute target,
seed 42, 200,000 fitting rows, 106,160 validation rows and 105,765 test rows.
Each booster samples the same ten configurations as the previous search and
selects its settings and round count on validation, using 30,000 training rows.
The test period is not used for tuning. Ridge uses Ajay's v2 validation-selected
parameters already on main; teammates' parameter files are not modified.

Validation selection on the 30,000-row tuning sample:

| Model | Selected rounds | v2 validation MAE | Earlier no-weather validation MAE |
|---|---:|---:|---:|
| AdaBoost | 200 | 44.346 | 44.275 |
| GradientBoosting | 300 | 43.588 | 43.607 |
| HistGradientBoosting | 600 | 40.618 | 40.371 |

All three winners reach their searched round limit. AdaBoost and HistGB retain
their previous selected settings; classic GB changes from 298 to 300 rounds.
The refreshed validation scores do not establish an improvement from weather.
The grids and candidate counts are unchanged; no search was expanded after
seeing test results. Exact values are in `tuning_summary.csv`.

The frozen contributor comparison includes the five baselines and Bhavesh's
three boosters. It is separate from Anirudh's complete team benchmark; omitted
models must not be ranked against these results using older feature setups.
Parameters are frozen before test scoring. No test-based retuning follows.

Run from the repository root with the runtime versions recorded in the manifests.
Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
`NUMEXPR_NUM_THREADS` to `2` before launching Python. All data options are explicit:

```bash
python -m experiments.bhavesh_boosting --source madrid --n-sensors 60 --horizon-min 60 --max-train-rows 200000 --tune-rows 30000 --candidates 10 --publish-params --output-dir results/bhavesh/madrid_v2_h60_development
python -m experiments.bhavesh_final_benchmark --source madrid --n-sensors 60 --horizon-min 60 --max-train-rows 200000 --threads 2 --models Persistence SeasonalNaive_day SeasonalNaive_week HistoricalAverage Ridge AdaBoost GradientBoosting HistGradientBoosting --output-dir results/bhavesh/madrid_v2_h60_development
python -m experiments.analysis_c_efficiency --source madrid --n-sensors 60 --horizon-min 60 --max-train-rows 200000 --threads 2 --repeats 1 --reuse-benchmark-fit --latency-repeats 30 --scaling-models Ridge AdaBoost GradientBoosting HistGradientBoosting --output-dir results/bhavesh/madrid_v2_h60_development
python -m experiments.bhavesh_consistency --source madrid --n-sensors 60 --horizon-min 60 --max-train-rows 200000 --output-dir results/bhavesh/madrid_v2_h60_development
streamlit run app/streamlit_app.py
```

Timing fits run sequentially on this machine. Analysis C reuses one controlled
frozen fit per model and measures 30 warm prediction repetitions. Training costs
have no repeated-fit confidence interval. Scaling uses five nested training sizes
and validation scores only. Estimator prediction latency excludes feature
construction, cold loading, network traffic and concurrent requests.

Full per-round CSVs, raw timing observations, command logs and fitted models are
generated locally by these commands. The PRs retain compact result tables,
parameter files, provenance manifests and figures. No dataset, model binaries or
shared `results/benchmark.csv` is added. Historical no-weather artifacts stay in
`../madrid_h60_development/`; the old demo/screenshots describe that older run.
