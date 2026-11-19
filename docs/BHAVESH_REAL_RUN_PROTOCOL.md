# Bhavesh real-data protocol (frozen before test evaluation)

Upstream: Ajay03299/ML-project main 358297a, incorporating Madrid data PRs 1/2.
The earlier local working copy is preserved; this branch is bhavesh/real-data-evaluation.
GitHub identity verified as Bhhavveshh with push access to the team repository.

## Dataset and methodological decisions

The merged file is a 12,432 by 300 hourly panel, June 2022 through October 2023.
SHA-256: 2e438da6caa45da811e8e5c08dd509db5e1c5034eb9038f7381f0755b368daab.
Its tracked bytes are preserved. The experiment uses the team's default 60
sensor positions, selected evenly across sensor IDs, 200,000 training rows,
seed 42, and chronological 70/15/15 split with horizon embargo.

- Main target: 60 minutes, exactly one hourly step, unit veh/h.
- Fifteen-minute evaluation is impossible from this hourly target. The code
  rejects it; there is no resampling or invented intermediate target.
- The shared legacy cache contains weather but no missing/imputed mask. Its
  upstream loader used backward fill. The original raw CSV/Parquet is not
  available locally. All models therefore exclude that weather array. We
  cannot establish whether backward fill actually changed any shared values;
  we conservatively avoid using them. Causal version-2 caches retain weather.
- Lag offsets are now hours: 0, 1, 2, 3, 4, 8 and 12. Rolling windows cover
  4 and 12 hourly observations; neighbour lags are 0 and 4 hours. Calendar
  describes the target time. Road attributes are static. Sensor summaries
  and correlation neighbours use training data only.
- The team's existing zero-run threshold remains 12 steps, therefore 12 hours
  for this panel. It is not silently relabelled as three hours. This archival
  cleaning rule retrospectively marks an entire long-zero run; it is a quality
  screening assumption, not a validated online fault detector. Historical
  scores are conditional on this mask. Across all 300 sensors, it adds 195
  missing cells to the 30,292 already missing in the merged panel. Report this
  limitation rather than claiming the complete ingestion pipeline is online-causal.

Resulting shared design has 29 features. The training budget is 200,000 rows;
validation has 106,160 rows and test has 105,765. Validation origins run from
29 May 2023 14:00 to 15 August 2023 05:00; test begins 15 August 2023 07:00.
The gap is the one-step boundary embargo.

## Validation-only selection

Bhavesh: ten fixed-seed candidates each for AdaBoost, classic Gradient Boosting
and HistGradientBoosting, using 30,000 training rows and all validation rows.
All rounds are scored, and the minimum validation MAE selects the configuration
and exact round. HistGB's internal random early stopping is disabled. No test
scores enter this selection. Cached AdaBoost tree predictions and pre-sorting
avoid redundant inference/sorting; regression checks compare all stages with
sklearn's weighted medians, including floating-point tie handling.

Supporting reference tuning uses the team starter grids, to avoid comparing
our tuned boosters only with weak defaults: Ridge (4), Decision Tree (25),
Random Forest (8 sampled), PCA+Ridge (10), PCA+RandomForest (6), XGBoost (8
sampled) and LightGBM (8 sampled). This is supporting comparator tuning, not
completion or authorship of teammates' dedicated analyses. All searches use
30,000 training rows and the same full chronological validation set. The
XGBoost/LightGBM grids fix 400 rounds; their separate 1,500-round early-stopping
experiments are not claimed. PCA preprocessing fits only on training rows.

Two threads are used for training experiments. Search fit times are diagnostic;
they are not the controlled Analysis C comparison because tuning jobs may run
concurrently. Canonical best-parameter JSONs are written only after validation.

## Frozen benchmark and Analysis C

After every search completes, freeze every configuration and record parameter
file hashes. Fit each of all 14 registry models on the same 200,000 rows and
score the held-out test period once. Preserve a per-contribution snapshot of
the benchmark; do not commit the shared results/benchmark.csv or trained models.

Analysis C then reuses these frozen test scores. It reuses one controlled full-size fit time, repeats warm single-row and batch
predictions with a common thread budget, measures serialized
size, and computes separate Pareto fronts. Scaling keeps parameters and validation
fixed and uses nested training samples; no scaling run scores test labels.
No test-based hyperparameter revision is allowed after the freeze.

The post-freeze consistency check selects the comparison baseline on validation
MAE, reproduces its and the three boosters' fixed test predictions, and reports
day/sensor win counts. Its optional descriptive interval uses seed 42, 2,000
circular seven-day moving-block resamples of daily mean MAE gains. Equal-day
mean gain differs from the row-weighted aggregate test MAE reduction. This
resampling describes variation inside one held-out period, not generalisation
to a new city or independent model-training trials.

A training-only resource pilot found 249,179 nodes in one full-data forest
member, implying roughly 3,422 MiB for the selected 200-tree forest. This is
an estimate, not the final measured size. To avoid changing the selected
configuration just to fit the machine, serialization is disk-backed, the UI
loads selected models lazily, and efficiency refits keep only one large
fitted model in memory. The all-size scaling plot reuses the controlled full-size benchmark fit time
and its frozen validation score. Every smaller size gets one new fit and new
validation predictions. Training times are single-run observations, not
confidence intervals; warm latency retains 30 repetitions. This
reuse avoids duplicate full-size fits and is marked in scaling.csv.

## Reproduction

From this worktree, use the project's installed Python environment (or create
one from requirements.txt). On this computer its executable is ../traffic-volume-prediction/.venv/Scripts/python.exe.

```powershell
python -m pytest -q
python -m experiments.bhavesh_boosting --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000 --tune-rows 30000 --candidates 10 --publish-params
python -m experiments.bhavesh_reference_tuning --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000 --tune-rows 30000 --threads 2
python -m experiments.bhavesh_final_benchmark --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000 --threads 2
python -m experiments.analysis_c_efficiency --source madrid --horizon-min 60 --n-sensors 60 --max-train-rows 200000 --repeats 1 --reuse-benchmark-fit --latency-repeats 30 --threads 2
python -m streamlit run app/streamlit_app.py
```

Results and reports must separate findings on these 60 sensors from claims
about all 300, real-time ingestion, different forecast horizons and raw weather.

## Pre-test timing-budget revision

Before opening test scores, the real-data search showed an AdaBoost candidate
requiring 133 seconds to fit only 30,000 rows, while the full forest resource
pilot implied over 3 GiB. Repeating identical full-size fits three times would
dominate the contribution's runtime without independent accuracy evidence.
The actual run therefore uses the controlled frozen benchmark fit once for
training cost, one fit at each smaller scaling size, and 30 warm repetitions
for single-row/batch inference. No training-time confidence or spread is
claimed. The CLI still supports repeated fresh fits for a later cost study.

## Original-source unit verification

The Madrid City Council's historical traffic data dictionary identifies
intensidad as a rate expressed in vehicles/hour; the original quarter-hour
vehicle count is the published intensity divided by four. Our shared archive
is hourly and retains the intensity unit, so its target is a rate, not a count
that should be multiplied by four. The original source dictionary is:
https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico/resource/208627-81-transporte-ptomedida-historico/download/208627-81-transporte-ptomedida-historico.pdf
Original source dataset: https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico
Derived dataset: https://data.mendeley.com/datasets/697ht4f65b/4

Raw aggregation details and timezone/DST reconstruction of the derived panel
are not independently verified. The UI uses the archive's clock and performs
held-out replay, not live wall-clock ingestion. Latency samples use seed-42
rows/batches spanning validation, identical for every model.

Road attributes are assumed time-invariant within this archive, and calendar
holidays are assumed known in advance. The historical vintage of the provided
road metadata was not reconstructed. These assumptions do not establish a
point-in-time road-information feed for a deployed service.

## Observed zero-run screening scope in this run

A direct before/after audit of the selected 60 sensors found 27 newly flagged
readings, all from sensor 10801 between 2022-08-07 13:00 and 2022-08-08 15:00.
This is entirely before the training boundary (2023-05-29 13:00). The filter
adds no flags within this run's validation or test periods. The rule is still
an archival screening assumption, not a confirmed sensor-fault label or an
online detection policy. The data-audit JSON records these counts and dates.

## Timing correction after an observed system suspension

Visual review of the completed scaling figure exposed an anomalous 25,000-row
classic GB wall time (1,287.874 seconds). Windows System Power-Troubleshooter
Event 1 recorded a sleep from 2026-11-19 01:44:43 to 02:04:30 local time, during
that fit. The full benchmark and latency audit had completed before this sleep.
The event query since 18 November 20:00 returned this one resume event.

Only the affected fit is repeated, using identical nested indices, seed,
parameters, validation data and two-thread budget. Test results are untouched.
The original scaling table is preserved as scaling_original.csv, the original
scaling_evaluations.csv remains unchanged, and timing_recheck.json records both
observations, process CPU time and the system event. The primary scaling table
and annotated figure use the uninterrupted repeat. This is correction of a
known invalid wall-clock observation, not additional hyperparameter selection
or a training-time confidence interval. Small ordinary timing fluctuations
elsewhere are retained. The timed repeat temporarily inhibits automatic sleep
for its own process and restores the previous execution state afterward.

## Integration with the later RF/PCA merge

Main advanced to 0281cf1 while this contribution was being verified. The update
adds Pragun's RF/PCA settings, tuning tables and figures, with no source/data
changes. All three selected parameter dictionaries and normalised Git blobs
match this study's selections. No model refit or metric change is required.
His shared tuning tables/figures are preserved. This contribution's seven
supporting search tables and parameter snapshots are retained under
madrid_h60_development/references/. Their scores belong to this study, not his
separate tuning run. Shared tune now accepts an optional out_dir (default
behavior unchanged); the reference helper isolates outputs and requires an
explicit --publish-params to replace shared selected settings. The original
fixed-budget searches were not rerun during this file-location integration.
