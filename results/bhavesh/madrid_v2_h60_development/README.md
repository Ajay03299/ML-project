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

Frozen test scores after fitting the selected models on 200,000 training rows:

| Model | Test MAE (veh/h) | Earlier no-weather test MAE |
|---|---:|---:|
| Ridge | 65.107 | 65.020 |
| AdaBoost | 51.341 | 51.397 |
| GradientBoosting | 45.269 | 44.691 |
| HistGradientBoosting | 43.035 | 42.520 |

HistGB has the lowest test MAE in this contributor subset, 33.9% below Ridge.
The v2 test scores do not demonstrate a general improvement from restoring
weather: AdaBoost improves slightly while classic GB and HistGB regress.
These differences are descriptive within the same historical test period.

Run from the repository root with the runtime versions in `environment.json`.
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

The full-size classic GB fit crossed documented Windows sleep and hardware-clock
corrections. Its original cost observation is retained in
`frozen_benchmark_before_timing_recheck.csv` and the timing audit. The guarded
`experiments.bhavesh_benchmark_timing_recheck` entry point repeats training with
the same frozen rows, parameters, seed and thread budget, checks validation-score
reproduction and records wall/CPU/UTC timing before replacing that cost. Test
scores and saved serving models are unchanged. Tuning CSV `fit_s` values are raw
observations; the controlled Analysis C audit supplies reported fit costs.

The repeat measured 666.250 wall seconds and 659.344 process CPU seconds and
reproduced validation MAE 40.568040101. The recorded correction command was:

```bash
python -m experiments.bhavesh_benchmark_timing_recheck --source madrid --n-sensors 60 --horizon-min 60 --max-train-rows 200000 --model GradientBoosting --output-dir results/bhavesh/madrid_v2_h60_development
```

This guarded repair requires the interruption audit and a completed Analysis C
run. Use a fresh output directory for another experiment so the archived audit
is preserved. The 50k/100k GB and HistGB timing observations are non-monotonic;
they are retained as measured. These single-fit curves do not establish a
scaling law, and CPU frequency/background activity were not controlled.
SciPy emitted an ill-conditioning warning (`rcond=3.15021e-08`) for the 100k
Ridge scaling fit. Its finite validation score is retained with the original
baseline solver and parameters; coefficient stability was not assessed.

`ui_verification.json` records Streamlit AppTest checks on all three saved
boosters and two sensors: displayed forecasts and day MAE match direct inference,
comparison/baseline controls work, and invalid hourly times are explained.
The CSV check covers direct serialization roundtrips; a browser download was not
exercised. Saved-model hashes are recorded in that verification file.

Final source verification: 46 tests passed in the isolated runtime, and dependency,
bytecode-compilation and diff checks passed. `artifact_verification.json` checks
the eight-model snapshot, all 20 scaling points, input/parameter/model hashes and
preservation of test metrics through the timing repair.

Full per-round CSVs, raw timing observations, command logs and fitted models are
generated locally by these commands. The PRs retain compact result tables,
parameter files, provenance manifests and figures. No dataset, model binaries or
shared `results/benchmark.csv` is added. Historical no-weather artifacts stay in
`../madrid_h60_development/`; the old demo/screenshots describe that older run.
