# Understand and explain your contribution

## Project execution path

`load_panel` (`src/data.py`) returns arrays indexed by timestamp and sensor,
plus weather, holidays and static road attributes. `build_dataset`
(`src/features.py`) converts them into one row per forecast origin/sensor.
`split_dates` and `split_xy` (`src/split.py`) define chronological training,
validation and test periods. `make_model` (`src/models.py`) builds the estimator.
Your boosting search selects settings using validation. The team's benchmark
fits those settings, calls `evaluate_model`, saves a model with joblib, and
records its data/feature identity. Streamlit calls the same feature builder
and those saved estimators. Analysis C audits the frozen models and table.

Example: with hourly sampling and horizon 1, a forecast origin of 07:00
predicts 08:00. `flow_lag0` is the 07:00 reading. `flow_lag4` is 03:00 (four hours earlier).
`flow_yesterday` is yesterday at 08:00, which is already observed. Target-time
hour/day features are known in advance; target-time observed weather is not.
Weather is excluded from this real run because its fill provenance is unverified.
For a verified raw rebuild, weather features use the origin observation.
A 4-step rolling window covers four hourly readings and ends at 07:00.

## Three boosting variants

**AdaBoostRegressor implements AdaBoost.R2.** It fits successive weak regression
trees using weighted training samples. Examples with larger normalised errors
receive more emphasis for subsequent rounds. The final prediction is a weighted
median of weak-regressor predictions, not simply their average. Base depth
controls individual-tree flexibility. `loss=linear` versus `square` controls
how errors affect sample weighting. Square loss can emphasise unusual/noisy
sensor observations strongly. Do not confuse this regression procedure with
the binary classification version of AdaBoost.

**Classic GradientBoostingRegressor** adds small trees to the current
prediction. Under its squared-error default, the next tree approximately fits
the residual (actual minus prediction). The learning rate shrinks each update.
Depth and minimum leaf size limit complexity; subsampling below 1 introduces
stochastic fitting. Small learning rates typically need more rounds. Training
may be expensive because splits examine feature values rather than using the
histogram scheme used by HistGB. Whether it is slower here must be measured.

**HistGradientBoostingRegressor** bins feature values and finds splits using
histogram statistics, reducing split-search cost. Leaf count, minimum leaf
size and L2 regularisation control complexity. `max_iter` counts boosting
iterations. Native missing-value support exists, but this project's common
feature builder drops incomplete rows for fair comparison, so do not claim
we measured an advantage from native missing-value handling. We disable the
estimator's random internal early stopping and select rounds on the team's
chronological validation period.

## Why not shuffle all the data?

Traffic repeats daily and weekly. If training sees future observations while
validation contains their earlier neighbours, scores can be optimistic and
will not describe forecasting the future. We split by time before tuning.
Random subsampling occurs only inside the training window, so validation and
test remain later. The horizon embargo removes labels crossing boundaries.
Training-only statistics are legitimate for held-out future origins, but not
for a claimed historical forecast inside that same training period.

## How are rounds selected?

For each candidate, fit the maximum allowed rounds. At every stage compute
training-sample MAE and chronological validation MAE. Find the lowest validation
MAE; ties choose the earliest round. Compare candidates by that value, save
the candidate plus selected round, then benchmark a fresh fit with those
settings. The final benchmark sample may be larger than the tuning sample,
which is a disclosed limitation. Test error never chooses the best round.

## What do the metrics mean?

MAE is the average absolute error in traffic units. RMSE squares errors before
averaging and taking a square root, so large misses matter more. R2 compares
the squared error with predicting the target mean; it can be negative. WAPE is
sum(abs(error))/sum(abs(actual)), stored as a ratio; it is undefined when all
actual flows are zero. Unlike per-row MAPE, individual zero-flow observations
do not cause division by zero. No single metric tells the entire story.

## What makes an efficiency comparison fair?

Use the same dataset, sensor subset, target horizon, feature columns, training
rows, fixed validation/test split, fitted settings, machine and thread budget.
Separate fit time from feature construction. Warm up prediction before timing.
Report the number of repetitions and their spread. A forest can be larger but
faster in a large batch with parallelism, while a small tree can have lower
single-request overhead. Measure rather than assume these outcomes.

A Pareto front keeps models for which no alternative has both no worse error
and no greater cost, with a strict improvement in at least one. If a tree has
MAE 20 and latency 1 ms and a booster has MAE 15 and latency 5 ms, neither
necessarily dominates: the choice depends on the application's error tolerance.
Those are explanatory values, not our project results.

## Why do we hash the setup?

A saved model can accept columns in the wrong order or be loaded with another
horizon's data and still produce plausible numbers. The manifest records
columns, split dates, training indices, data content, feature source files and
model file hashes. The UI checks the manifest before serving. This catches
stale files; it is not a security sandbox for untrusted pickle/joblib files.
Only load locally trained/trusted model artifacts.

## What is the day chart actually showing?

The date is the TARGET day. At midnight, the one-hour forecast originates on
the previous day at 23:00. Every plotted point uses the observations available
at its own origin. We are not forecasting an entire day from its first reading.
Actual observations shown later are for retrospective checking, not input
features. Daily MAE uses only available paired rows; missing intervals are
neither filled with zeros nor included in the denominator.

## Questions you should answer from the final real run

1. Which booster wins at 60 minutes, and does it also win at 15 minutes?
2. How much does it improve over persistence and yesterday/last-week baselines?
3. Does its validation curve flatten or worsen before the round cap?
4. How large is the training/validation error gap? Could it be overfitting?
5. Which models are on each cost/error Pareto front, and why do the fronts differ?
6. Does more training data materially lower validation MAE, and at what fit cost?
7. Which real sensor/day in the UI exposes a failure case? Is it a peak, holiday,
   weather gap or sudden change? Coordinate with Anirudh's error analysis.
8. What portion of real rows is lost to missing history/weather/target labels?
9. Why is your proposed deployment model preferable under a stated latency,
   storage and error budget? What end-to-end costs have not been measured?
10. What did YOU implement, what was starter code, and what did teammates own?

Use the frozen Madrid result tables and the measured report for numerical
answers. Practise tracing a forecast through the actual implementation rather
than memorising a model ranking without its evaluation conditions.


## Real-run points you must explain correctly

- Hourly Madrid data means horizon=1 for 60 minutes. There are no observed
  15-minute targets to evaluate; interpolated labels would not fix that.
- The canonical choices are AdaBoost: 200/depth 10/rate .05/square;
  Gradient Boosting: 298/depth 5/rate .1/min-leaf 10/subsample 1;
  HistGB: 600/63 leaves/rate .03/min-leaf 20/L2 0/no internal early stop.
  AdaBoost and HistGB hit their tested caps, so do not call them globally optimal.
- The search scores all 106,160 validation rows at every round; supervised
  tuning uses 30,000 training rows and the final fit uses 200,000. The full
  training period also supplies sensor statistics and correlation neighbours.
- In these 60 sensors, zero-run screening flags 27 readings from sensor 10801
  in August 2022, entirely inside training. No validation/test readings are
  newly flagged by that rule. The rule is not a confirmed fault label.
- Best AdaBoost here uses square loss. Do not say the experiment found linear
  loss always better just because square loss is theoretically outlier-sensitive.
- GB training-sample MAE can exceed validation MAE because these are different
  time periods/distributions. The sign of that gap alone does not diagnose
  leakage or establish an absence of overfitting. Our chronological split,
  feature audit and omitted unverified weather address separate questions.
- Training cost is a single controlled fit. Inference has 30 warm repetitions;
  repeated latency samples are not independent model-generalisation trials.
  The full-size scaling point reuses the same fit/validation score.
- The current forest's large storage is a measured deployment concern, not a
  reason to silently change its validation-selected configuration after seeing
  test results. Size in the table is joblib storage, not peak process memory.
- PCA pipelines fit preprocessing on training only. Their comparison with a
  differently configured plain forest is not an isolated causal PCA experiment.
- LightGBM uses GBDT here. GOSS was not enabled or evaluated.
- The app demonstrates rolling historical forecasts. Each point has its own
  origin; it is not a whole day predicted at midnight or a live traffic feed.
- Read the frozen result table and Analysis C interpretation for the actual
  measured ranking. Do not infer a winner from theory or synthetic checks.

## Measured answers from the frozen run (19 November 2026)

Use `results/bhavesh/madrid_h60_development/frozen_benchmark.csv` and
`efficiency.csv` as the numerical source of truth:

- Your three test MAEs are AdaBoost 51.40, classic GB 44.69, HistGB 42.52 veh/h.
  HistGB is best among your variants. Ridge was the validation-selected baseline;
  its test MAE is 65.02, so HistGB reduces aggregate MAE by about 34.6%.
- XGBoost has the lowest overall test MAE (39.99). RF is 41.07 and LightGBM
  41.45. Do not call HistGB the overall winner or hide the stronger comparators.
- HistGB fits in 37.85 s versus classic GB's 1,175.91 s and AdaBoost's 2,934.60 s.
  These are one-fit observations on this laptop, with a two-thread budget.
- Fast fitting does not guarantee fast prediction: HistGB's warm single-row
  median is 21.748 ms versus classic GB's 3.479 ms. Their selected ensemble
  structures and library prediction overhead differ; this experiment measures
  the effect but does not isolate each internal cause.
- Classic GB is about 1.175 MiB, HistGB 4.156 MiB, AdaBoost 23.955 MiB and the
  full forest about 3,427 MiB. Size and error form a trade-off; do not assume
  the best training speed also means the smallest artifact.
- With an explicit 5% tolerance over the best test MAE, LightGBM is the fastest
  measured median single-row predictor (3.206 ms; p95 5.714 ms). This is a
  descriptive choice on the measured holdout, not an independently validated
  live-service decision. HistGB lies just outside that error tolerance.
- You must still read the day/sensor consistency and scaling tables for their
  actual observations. A low aggregate MAE alone does not establish that every
  sensor or day improves.

Also avoid saying R2=0.983 means 98.3% accuracy. It is an aggregate squared-error
comparison against the held-out target mean, across many sensor scales. Report
MAE/RMSE in veh/h and WAPE as a ratio or clearly labelled percent. This run
forecasts future times for seen sensors; it does not demonstrate cold-start
performance at unseen sensors.

Batching changes the result: classic GB takes 15.31 ms per 1,000 rows versus
XGBoost 32.37 ms, LightGBM 66.48 ms and HistGB 108.12 ms in this audit. Classic
GB lies on the measured batch-cost and storage Pareto fronts despite its long
training time and higher test MAE. Do not divide a batch time by 1,000 and call
that the latency of an individual API request.

## The rest of the project: minimum understanding for the individual viva

- **Decision Tree:** recursively chooses feature thresholds, then predicts a leaf
  mean. The selected comparator uses maximum depth 20 and at least 50 samples
  per leaf; this limits a single tree's flexibility. Test MAE is 53.21 veh/h.
- **Random Forest:** averages 200 bootstrap-trained trees, considering a random
  feature subset at splits. This selected forest has no depth cap, minimum leaf
  size 1 and max_features=0.5. Its 41.07 test MAE comes with a roughly 3.35 GiB
  saved artifact; averaging trees does not automatically make storage cheap.
- **PCA:** the pipelines standardise features, fit principal directions on
  training rows and transform later rows with those fixed directions. Selecting
  n_components=0.99 retains at least 99% of training feature variance, not 99%
  of target information. The PCA forest has 100 trees, depth 16 and leaf minimum
  5, unlike the plain forest. Their score difference therefore combines PCA
  and changed forest settings; it cannot isolate PCA's causal contribution.
- **XGBoost and LightGBM:** these are teammates' additional boosting variants.
  The supporting comparison fixes 400 rounds. XGBoost's selected depth is 10
  with rate 0.1; LightGBM uses 127 leaves and rate 0.03 with GBDT. They achieve
  test MAE 39.99 and 41.45 respectively. Separate early-stopping/GOSS experiments
  were not performed by this contribution.
- **Analysis A:** compare feature-group removals with the same evaluation setup.
  The current builder always keeps sensor_id, slot and is_weekend. Removing an
  engineered group does not remove every signal related to that concept.
- **Analysis E:** compare errors across times/sensors and other available
  conditions. Weather-based segments cannot be claimed for this non-weather
  matrix. Ask the owner to explain their actual segmentation and results.

For EDA, do not invent rush-hour/holiday findings from general traffic intuition.
Read Ajay's actual figures and missingness analysis when the team assembles the
complete report. This contribution supplies a data audit, not authorship of his
full EDA or the other members' dedicated analyses.

## Consistency and timing audit: observed details

HistGB and classic GB improve on Ridge's mean MAE on all 78 target days and all
60 sensors. AdaBoost improves on 74/78 days and all 60 sensors. These are group
averages: they do not mean every individual forecast improves. In the actual
UI example (sensor 3397, 30 October), AdaBoost's day MAE is 41.21 versus classic
GB 42.72 and HistGB 43.54, although HistGB wins on the aggregate test set.

The first 25,000-row GB timing contained Windows system sleep and is invalid as
compute cost. Its exact-row/seed/parameter repeat took 74.89 seconds (74.52 CPU
seconds), reproducing validation MAE 42.816477. Original measurements and the
system event are retained; this is not selective hyperparameter retuning.
