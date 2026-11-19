# Bhavesh - real-data report sections

BOOSTING VARIANTS, ANALYSIS C AND INTEGRATED INTERFACE | REAL-DATA RESULTS

## 1. Actual data and evaluation design

The merged Madrid archive contains 300 sensors and 12,432 hourly timestamps from June 2022 to October 2023. This common experiment uses 60 evenly selected sensor IDs and forecasts one hour ahead (one step), in veh/h. A 15-minute target is unavailable and is rejected rather than interpolated. The original city dictionary confirms the rate unit [1,2].

All models share 29 features: hourly lags, 4/12-reading rolling statistics, seasonal lags, target calendar, training-fitted sensor/neighbour summaries and static roads. Unverified legacy weather is excluded for every model. The chronological split and embargo give 486,400 usable training-period rows, sampled to 200,000 for final fitting; validation has 106,160 rows and test 105,765. Test origins span 15 August to 31 October 2023.

## 2. Boosting search and measured results

AdaBoost.R2 reweights difficult samples and predicts by a weighted median. Classic GB adds residual-fitting trees under squared error; HistGB bins features for split search. Ten seed-42 configurations per variant use 30,000 training rows and all validation rows. Every stage is scored; validation MAE selects both settings and rounds. HistGB random internal early stopping is disabled. The configuration freeze precedes all test scoring.

| Model                | Validation-selected settings                           |   Tune val MAE |   Test MAE |   Test RMSE |
|:---------------------|:-------------------------------------------------------|---------------:|-----------:|------------:|
| AdaBoost             | 200 rounds; depth 10; rate 0.05; square loss           |          44.28 |      51.4  |       88.16 |
| GradientBoosting     | 298 rounds; depth 5; rate 0.1; leaf min 10; sample 1.0 |          43.61 |      44.69 |       77.38 |
| HistGradientBoosting | 600 rounds; 63 leaves; rate 0.03; leaf min 20; L2=0.0  |          40.37 |      42.52 |       74.96 |

Tuning scores use the 30,000-row fit; test scores use a fresh 200,000-row fit. All errors are veh/h. Supporting reference families were also validation-tuned; the appendix contains all 14 models.

Late-round detail; full train/validation curves for every candidate are retained. AdaBoost (200) and HistGB (600) hit their search caps: these are bounded-search selections, not proven global optima.

Among these boosters, HistGradientBoosting has the lowest frozen test MAE (42.52), a 34.6% reduction over Ridge (65.02), the baseline selected on validation. It improves on 78/78 target days and 60/60 sensors. These are observations within one held-out period; no test-based retuning was performed.

## 3. Error, latency, training cost and storage

On an Intel Core i5-1035G1 with a two-thread budget, training time is one controlled frozen fit per model. Inference uses 30 warm single-row and batch timings on identical seed-42 validation cases. Size is uncompressed joblib storage, not peak RAM. Feature construction, ingestion, cold loading and concurrency are excluded.

| Model                |   Test MAE |   Fit s |   Single ms |   p95 ms |   Size MiB |
|:---------------------|-----------:|--------:|------------:|---------:|-----------:|
| XGBoost              |      39.99 |   37.07 |      10.262 |   14.039 |      8.468 |
| LightGBM             |      41.45 |   18.76 |       3.206 |    5.714 |      4.228 |
| HistGradientBoosting |      42.52 |   37.85 |      21.748 |   33.635 |      4.156 |
| AdaBoost             |      51.4  | 2934.6  |      47.26  |   63.748 |     23.955 |
| GradientBoosting     |      44.69 | 1175.91 |       3.479 |    5.302 |      1.175 |
| RandomForest         |      41.07 |  419.94 |     101.58  |  148.968 |   3427.42  |

XGBoost gives the lowest aggregate test MAE (39.99). Within an explicit 5% error tolerance, LightGBM is the fastest measured single-row predictor (3.206 ms median; 5.714 ms p95). HistGB fits 31.1 times faster than classic GB, yet its single-row prediction is 6.3 times slower here. Classic GB's 1,000-row median is 15.31 ms versus LightGBM's 66.48 ms: batching changes the trade-off. These estimator timings establish no end-to-end SLA.

Eight model families use nested 12,500/25,000/50,000/100,000/200,000-row samples and fixed validation/settings. HistGB validation MAE changes from 44.15 to 38.30, while fit time changes from 10.62 to 37.85 s. Full-size timings/validation scores are reused from the benchmark; smaller sizes get new fits. This measures row count, not reduced history. Single fit times have no confidence interval.

Timing audit: one 25,000-row GB fit included Windows sleep. Only that timing was repeated with identical rows/settings; original measurements and the system event are retained. No test scores changed.

## 4. Integrated UI and verification

Streamlit uses the shared features and hash-checked models. Sensor, target date/time and model controls drive forecasts, recent history, multi-model day charts, daily MAE/RMSE, sensor map and provenance-rich CSVs. Models load lazily; gaps stay missing. Each point is a rolling historical forecast from its own origin, not a whole day predicted at midnight.

Regression tests cover leakage, horizon/weather safeguards, cached AdaBoost medians, feature identity, time alignment, Pareto ties and UI controls. Browser exports are checked against direct inference. Screenshots, a two-minute demo and full cost/scaling figures accompany the evidence. Existing experiment call sites remain compatible.

## 5. Limits and interpretation

Evidence covers 60 seen sensors and one temporal holdout, with a smaller tuning sample and finite round caps. Zero-run screening changes 27 training readings and no validation/test readings; it is not an online detector. Raw aggregation, road-metadata vintage and timezone remain unverified. Timing is machine-specific and not OS-isolated. All-300-sensor, 15-minute and live-feed claims are unsupported. Teammates retain analyses A/E.

[1] Gomez & Ilarri, Enriched Traffic Datasets for Madrid, Mendeley Data V4, CC BY 4.0, DOI 10.17632/697ht4f65b.4. [2] Madrid City Council, Historical Traffic Data Dictionary, October 2023, section 3.3. [3] Related article: DOI 10.1109/OJITS.2025.3637305. Full URLs, run hashes and environment are in the accompanying protocol/evidence.

## All frozen model scores

| model                |   val_MAE |   test_MAE |   test_RMSE |   test_R2 |   test_WAPE |
|:---------------------|----------:|-----------:|------------:|----------:|------------:|
| XGBoost              |   36.0243 |    39.9925 |     72.9257 |  0.983241 |   0.0654562 |
| RandomForest         |   36.9672 |    41.0733 |     74.5577 |  0.982482 |   0.0672251 |
| LightGBM             |   37.3794 |    41.452  |     73.479  |  0.982986 |   0.067845  |
| HistGradientBoosting |   38.298  |    42.5204 |     74.9588 |  0.982293 |   0.0695937 |
| GradientBoosting     |   40.2672 |    44.6907 |     77.3769 |  0.981133 |   0.0731458 |
| PCA_RandomForest     |   44.873  |    47.6713 |     84.0747 |  0.977725 |   0.0780242 |
| AdaBoost             |   44.2887 |    51.3967 |     88.1574 |  0.975509 |   0.0841215 |
| DecisionTree         |   46.8624 |    53.213  |     93.8289 |  0.972256 |   0.0870944 |
| Ridge                |   56.0732 |    65.0189 |    106.52   |  0.964244 |   0.106417  |
| PCA_Ridge            |   56.9907 |    68.1097 |    113.032  |  0.959738 |   0.111476  |
| SeasonalNaive_week   |   56.484  |    75.3664 |    146.112  |  0.932724 |   0.123353  |
| HistoricalAverage    |   95.7924 |   105.338  |    179.001  |  0.899028 |   0.172409  |
| Persistence          |  102.318  |   109.947  |    185.263  |  0.89184  |   0.179952  |
| SeasonalNaive_day    |  107.033  |   117.227  |    234.556  |  0.826626 |   0.191866  |
