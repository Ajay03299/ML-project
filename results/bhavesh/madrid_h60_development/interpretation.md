# Analysis C - frozen benchmark audit

Source: madrid; horizon: 60 minutes; unit: veh/h.
Run: a8d53e5b4f04b224e5c554d45e72435cc84b153695ea5b6512c95d63dab03740. Fixed training rows: 200000; threads: 2.

Observed lowest frozen test MAE: XGBoost (39.992).
Within 5% of that error, the fastest measured median single-row predictor was
LightGBM (3.206 ms; p95 5.714 ms;
4329.3 KiB). This is a descriptive candidate under that
explicit error tolerance, not a universal deployment recommendation. Check the
separate Pareto fronts for training time, latency and storage before choosing.

Training times are single-run observations from the controlled frozen benchmark; no training-time confidence interval or p95 is estimated. Warm latency has 30 repetitions;
p95 is descriptive with this sample size.
Batch time is measured on at most 1,000 validation rows; normalised per-1,000
throughput is not single-request latency. Model size is uncompressed joblib file
size, not peak RAM. The recorded environment is in efficiency_manifest.json.

Scaling uses nested, fixed-seed samples spanning the same training period and a
fixed validation period. It measures row-count scaling, not shorter-history
robustness. Parameters stay frozen. The largest-size timing repetitions and frozen validation score are reused
from the full-size audit to avoid duplicate fits; smaller sizes receive new fits.
Test scores are reused from the benchmark;
no scaling run reads test labels. Models absent from scaling: [].
Models absent from the complete team benchmark: [].

Feature construction, data ingestion, network overhead, concurrent requests and
cold model loading are excluded. No measured estimator latency establishes an
end-to-end real-time SLA. These findings apply to the frozen 60-sensor madrid experiment, at 60-minute sampling and a 60-minute target. Other sensor sets, cities and forecast horizons require separate evaluation.

Timing audit: the 25,000-row classic GB fit included a documented Windows sleep. Its original 1287.874 s observation is retained in scaling_original.csv and scaling_evaluations.csv. One exact-row/seed/parameter repeat took 74.886 s (74.516 process CPU s) and reproduced validation MAE 42.816477. The primary scaling table/figure uses that repeat. No test scores or settings changed. See timing_recheck.json and system_resume_audit.json.
