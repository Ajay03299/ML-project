# Traffic Flow Forecasting (ML course project, Topic 4)

Team: **Ajay, Pragun, Bhavesh, Anirudh**

The boosting search tunes on validation only; run the final benchmark separately.
Synthetic/quick tuning is isolated
from the team's canonical best-parameter files. Causal weather preprocessing
uses forward fill only; the merged Madrid v2 cache enables weather features.
Legacy shared panels retain traffic/calendar/road data
but exclude weather whose past-only filling cannot be verified from the shared cache.

The Madrid v2, 60-sensor, one-hour contributor comparison gives HistGB test MAE
**43.04 veh/h**, a **33.9%** reduction over the validation-selected Ridge baseline
(65.11). Classic GB scores **45.27** and AdaBoost **51.34**. This comparison covers
the five baselines and Bhavesh's three boosters; Anirudh owns the complete team
benchmark after the remaining model families' v2 tuning. See the
[v2 benchmark](results/bhavesh/madrid_v2_h60_development/frozen_benchmark.csv) and
[rerun methodology](results/bhavesh/madrid_v2_h60_development/README.md).
The earlier no-weather comparison is retained in the
[historical archive](results/bhavesh/madrid_h60_development/frozen_benchmark.csv).

Forecast the traffic flow of a road sensor **60 minutes ahead** (one step in the merged hourly Madrid panel; 15-minute targets are unavailable) from past
observations only, and compare Decision Tree, Random Forest, PCA-based models and five boosting variants (AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM)
against simple baselines. A Streamlit app lets you pick a sensor, a date and a time and see the forecast
next to recent traffic.

## Task

At origin time `t` (the last observed hourly reading in the merged panel), predict the flow of one sensor at `t + h`.
Regression. Metrics: MAE, RMSE, R2 and WAPE (plain MAPE is avoided: night-time flows are near zero).
Flow unit: vehicles per hour as reported by the Madrid portal; see [data attribution](DATA_ATTRIBUTION.md) for the original source.

## Dataset

**Madrid Traffic Dataset (MTD), subset version.** Gómez and Ilarri, *Enriched Traffic Datasets for Madrid*,
Mendeley Data, V4, DOI 10.17632/697ht4f65b.4, licence CC BY 4.0: https://data.mendeley.com/datasets/697ht4f65b/4
Original traffic source: Madrid Open Data Portal. Related article: Gómez and Ilarri, IEEE Open Journal of
Intelligent Transportation Systems, 2025, DOI 10.1109/OJITS.2025.3637305. Cite all three in the report.

- Subset: 300 sensors, June 2022 to October 2023, hourly readings (the complete version has 554 sensors, 30 months). Use the files inside the extracted `.rar` folder (300-sensor adjacency); ignore `temporal_sequences/` (pre-scaled deep-learning windows)
- One long CSV (`id, date, traffic_intensity, day_type, wind, temperature, precipitation`, plus road attributes:
  lanes, highway class, maxspeed, oneway, length, coordinates), a coordinates CSV and an adjacency matrix (`.npy`)
- Flow unit: `traffic_intensity` as in the Madrid portal (vehicles per hour, reported hourly in the shared panel).
  Raw-file aggregation and timezone reconstruction could not be independently checked from the merged panel.
- Known quirks (handled in code, discussed in the report): weather is city-level (one value per timestamp);
  `temperature` is an encoded value, not degrees C; few rows are labelled "Saturday", so weekends are taken
  from the timestamp and only "Holiday" is used from `day_type`

**Access.** Open the Mendeley page, click *Download All*, unzip into `data/raw/madrid/` (any sub-folders are fine;
the loader finds the files). Then once:

```bash
python scripts/check_madrid.py      # data gate report -> docs/data_gate.md
python -m src.data madrid           # CSV -> Parquet -> data/processed/madrid_panel.npz (a few minutes, once)
```

The team has already merged `madrid_panel.npz` and `madrid_sensors.csv` into
`data/processed/`; a current checkout includes them. The raw-file gate above
requires the original download, which was unavailable for this contribution.

**Fallback:** PEMS04 (Caltrans PeMS District 4, 307 sensors, Jan to Feb 2018, 5-minute data aggregated to 15):
put `PEMS04.npz` in `data/raw/` and set `SOURCE = "pems04"` in `src/config.py`.
**Synthetic:** `SOURCE = "synthetic"` runs everything with no download (for testing code only; never report it).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest                             # all tests should pass
```

## Reproduce the experiments

Settings shared by every script live in `src/config.py` (source, number of sensors, horizon in minutes, sample sizes).
Family experiments support `--quick` for smoke checks. Shared data options are `--source`, `--n-sensors`, `--horizon-min`, and `--max-train-rows`; the frozen reference/recheck entry points require their recorded real-data setup. `--horizon` explicitly overrides the horizon in steps. Horizons must be positive exact multiples of the observed interval.

```bash
# 1. EDA
python -m experiments.ajay_eda
# 2. tune each model family on the validation period (writes results/best_params/*.json)
python -m experiments.ajay_decision_tree          # baselines, Ridge, Decision Tree
python -m experiments.pragun_random_forest
python -m experiments.pragun_pca
python -m experiments.bhavesh_boosting --publish-params  # AdaBoost, Gradient Boosting, HistGradientBoosting
python -m experiments.anirudh_boosting            # XGBoost, LightGBM
# 3. final benchmark with the tuned settings (results/benchmark.csv, models/ for the UI) and the table
python -m experiments.run_benchmark
python -m experiments.make_tables                 # results/benchmark_table.md
# 4. analyses
python -m experiments.analysis_a_ablation         # A: feature groups
python -m experiments.analysis_c_efficiency       # C: cost vs error
python -m experiments.analysis_e_errors           # E: error breakdown
python -m experiments.ajay_sensor_holdout         # generalisation to unseen sensors
# 5. user interface
streamlit run app/streamlit_app.py
```

Figures go to `results/figures/`, tables to `results/`.

For a rebuilt panel, keep Bhavesh's artifacts separate from the historical
no-weather run with `--output-dir results/bhavesh/madrid_v2_h60_development`.
Use that same option for boosting tuning, the frozen comparison, Analysis C and
the consistency check. Quick/synthetic outputs get a separate smoke subdirectory.
The frozen comparison accepts `--models` to evaluate a contributor subset, for example:

```bash
python -m experiments.bhavesh_boosting --tune-rows 30000 --publish-params --output-dir results/bhavesh/madrid_v2_h60_development
python -m experiments.bhavesh_final_benchmark --threads 2 --models Persistence SeasonalNaive_day SeasonalNaive_week HistoricalAverage Ridge AdaBoost GradientBoosting HistGradientBoosting --output-dir results/bhavesh/madrid_v2_h60_development
```

The subset and omitted models are recorded in `benchmark_freeze.json`. This is a
contributor comparison; Anirudh still owns the final complete team benchmark.
Data/feature signatures, shared configuration and teammate parameter files are unchanged.

## Forecast-origin features and evaluation safeguards

| File | Role |
|---|---|
| `src/config.py` | Shared settings: data source, sensors, horizon, sample sizes, seed |
| `src/data.py` | Any dataset -> a `Panel` (flow by time by sensor, detected 15-minute or hourly grid, weather, holidays, road attributes). Negative values and long zero runs become NaN |
| `src/features.py` | Leakage-safe features in switchable groups: `lags, seasonal, rolling, calendar, weather, neighbors, covariates, sensor, road`. Only data at or before `t`; "same time yesterday / last week" is legal because the horizon is under one day; neighbour sets and sensor statistics are fitted on the training period only |
| `src/split.py` | Chronological 70/15/15 split with an embargo, and a sensor-level holdout (15% of sensors never used in training) |
| `src/models.py` | Every model, its owner and defaults; tuned settings are loaded from `results/best_params/` |
| `src/evaluate.py` | Scores, times and sizes any model; one row per (model, tag, horizon) in `results/benchmark.csv` |
| `experiments/` | One script per person and per analysis; `common.py` holds the shared tuning loop |
| `scripts/check_madrid.py` | Day-1 data gate |
| `app/streamlit_app.py` | UI: pick sensor, date and time; forecast next to the last 24 h; leaderboard |
| `tests/` | Includes a test that rewrites the future and checks earlier features do not change. **Keep it passing** |

Hyper-parameters are chosen on the validation period only; the test period is used after the configuration freeze.
The v2 cache includes forecast-origin weather; legacy caches exclude weather
because fill provenance is unavailable.
The inherited long-zero-run cleanup is retrospective archival quality screening,
not a validated online fault detector. Its scope is recorded in the [data audit](results/bhavesh/madrid_data_audit.json).

