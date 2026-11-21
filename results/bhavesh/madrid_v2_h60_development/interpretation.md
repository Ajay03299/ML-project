# Analysis C - frozen benchmark audit

Source: madrid; horizon: 60 minutes; unit: veh/h.
Run: 9adb1c2116ffbf69ac1e3fbc35f1e454b4d7d2d0df8ad5eff001396ccc98ff51. Fixed training rows: 200000; threads: 2.

Observed lowest frozen test MAE: HistGradientBoosting (43.035).
Within 5% of that error, the fastest measured median single-row predictor was
HistGradientBoosting (24.448 ms; p95 31.801 ms;
4256.6 KiB). This is a descriptive candidate under that
explicit error tolerance, not a universal deployment recommendation. Check the
separate Pareto fronts for training time, latency and storage before choosing.

Training times are single-run observations from the single controlled fits with documented timing repeats; no training-time confidence interval or p95 is estimated. Warm latency has 30 repetitions;
p95 is descriptive with this sample size.
Batch time is measured on at most 1,000 validation rows; normalised per-1,000
throughput is not single-request latency. Model size is uncompressed joblib file
size, not peak RAM. The recorded environment is in efficiency_manifest.json.

Scaling uses nested, fixed-seed samples spanning the same training period and a
fixed validation period. It measures row-count scaling, not shorter-history
robustness. Parameters stay frozen. The largest-size timing repetitions and frozen validation score are reused
from the full-size audit to avoid duplicate fits; smaller sizes receive new fits.
Test scores are reused from the benchmark;
no scaling run reads test labels. Scaling models: ['AdaBoost', 'GradientBoosting', 'HistGradientBoosting', 'Ridge'].
Requested scaling models unavailable in this benchmark: [].
Registered models not covered by this benchmark: ['DecisionTree', 'LightGBM', 'PCA_RandomForest', 'PCA_Ridge', 'RandomForest', 'XGBoost'].

Measurement limits: the 50k/100k GB and HistGB fit times are non-monotonic and
retained as measured. Single-fit curves do not establish a scaling law; CPU
frequency and background activity were not controlled. SciPy reported an
ill-conditioned matrix (rcond=3.15021e-08) for the 100k Ridge fit. Its finite
validation score is retained without changing the baseline solver or parameters.

Feature construction, data ingestion, network overhead, concurrent requests and
cold model loading are excluded. No measured estimator latency establishes an
end-to-end real-time SLA. These findings apply to the frozen 60-sensor madrid experiment, at 60-minute sampling and a 60-minute target. Other sensor sets, cities and forecast horizons require separate evaluation.

Timing audit: GradientBoosting's original fit spanned documented Windows sleep/clock corrections. Its original 22222.749 s observation is retained in GradientBoosting_benchmark_timing_recheck.json and the *_before_timing_recheck.csv backups. The same-row/seed/parameter repeat took 666.250 s (659.344 CPU s) and reproduced the frozen validation score. Only its fit cost and derived cost plots were replaced. Test scores and saved models are unchanged.
