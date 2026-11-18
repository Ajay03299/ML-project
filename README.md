# Traffic Flow Forecasting (ML course project, Topic 4)

Team: **Ajay, Pragun, Bhavesh, Anirudh**

Forecast the traffic flow of a road sensor **60 minutes ahead** (15 minutes as a second horizon) from past
observations only, and compare Decision Tree, Random Forest, PCA-based models and five boosting variants (AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM)
against simple baselines. A Streamlit app lets you pick a sensor, a date and a time and see the forecast
next to recent traffic.

## Task

At origin time `t` (the last observed 15-minute step), predict the flow of one sensor at `t + h`.
Regression. Metrics: MAE, RMSE, R2 and WAPE (plain MAPE is avoided: night-time flows are near zero).
Flow unit: vehicles per hour as reported by the Madrid portal (confirmed in `docs/data_gate.md`).

## Dataset

**Madrid Traffic Dataset (MTD), subset version.** Gómez and Ilarri, *Enriched Traffic Datasets for Madrid*,
Mendeley Data, V4, DOI 10.17632/697ht4f65b.4, licence CC BY 4.0: https://data.mendeley.com/datasets/697ht4f65b
Original traffic source: Madrid Open Data Portal. Related article: Gómez and Ilarri, IEEE Open Journal of
Intelligent Transportation Systems, 2025, DOI 10.1109/OJITS.2025.3637305. Cite all three in the report.

- Subset: 300 sensors, June 2022 to October 2023, 15-minute readings (the complete version has 554 sensors, 30 months)
- One long CSV (`id, date, traffic_intensity, day_type, wind, temperature, precipitation`, plus road attributes:
  lanes, highway class, maxspeed, oneway, length, coordinates), a coordinates CSV and an adjacency matrix (`.npy`)
- Flow unit: `traffic_intensity` as in the Madrid portal (vehicles per hour, reported every 15 minutes);
  confirmed by `scripts/check_madrid.py` -> `docs/data_gate.md`
- Known quirks (handled in code, discussed in the report): weather is city-level (one value per timestamp);
  `temperature` is an encoded value, not degrees C; few rows are labelled "Saturday", so weekends are taken
  from the timestamp and only "Holiday" is used from `day_type`

**Access.** Open the Mendeley page, click *Download All*, unzip into `data/raw/madrid/` (any sub-folders are fine;
the loader finds the files). Then once:

```bash
python scripts/check_madrid.py      # data gate report -> docs/data_gate.md
python -m src.data madrid           # CSV -> Parquet -> data/processed/madrid_panel.npz (a few minutes, once)
```

Team members can skip the raw download: copy `madrid_panel.npz` and `madrid_sensors.csv` (shared on the team
Drive) into `data/processed/`.

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

Settings shared by every script live in `src/config.py` (source, number of sensors, horizon, sample sizes).
Every script accepts `--quick` for a one-minute check, and `--source`, `--n-sensors`, `--horizon`, `--max-train-rows`.

```bash
# 1. EDA
python -m experiments.ajay_eda
# 2. tune each model family on the validation period (writes results/best_params/*.json)
python -m experiments.ajay_decision_tree          # baselines, Ridge, Decision Tree
python -m experiments.pragun_random_forest
python -m experiments.pragun_pca
python -m experiments.bhavesh_boosting            # AdaBoost, Gradient Boosting, HistGradientBoosting
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

## Design (why nothing leaks from the future)

| File | Role |
|---|---|
| `src/config.py` | Shared settings: data source, sensors, horizon, sample sizes, seed |
| `src/data.py` | Any dataset -> a `Panel` (flow by time by sensor, 15-min grid, weather, holidays, road attributes). Negative values and long zero runs become NaN |
| `src/features.py` | Leakage-safe features in switchable groups: `lags, seasonal, rolling, calendar, weather, neighbors, covariates, sensor, road`. Only data at or before `t`; "same time yesterday / last week" is legal because the horizon is under one day; neighbour sets and sensor statistics are fitted on the training period only |
| `src/split.py` | Chronological 70/15/15 split with an embargo, and a sensor-level holdout (15% of sensors never used in training) |
| `src/models.py` | Every model, its owner and defaults; tuned settings are loaded from `results/best_params/` |
| `src/evaluate.py` | Scores, times and sizes any model; one row per (model, tag, horizon) in `results/benchmark.csv` |
| `experiments/` | One script per person and per analysis; `common.py` holds the shared tuning loop |
| `scripts/check_madrid.py` | Day-1 data gate |
| `app/streamlit_app.py` | UI: pick sensor, date and time; forecast next to the last 24 h; leaderboard |
| `tests/` | Includes a test that rewrites the future and checks earlier features do not change. **Keep it passing** |

Hyper-parameters are chosen on the validation period only; the test period is used once, for the final table.

## Who does what

| Person | ML models | Analysis | Other ownership | Reviewer |
|---|---|---|---|---|
| **Ajay** | Baselines (persistence, seasonal, historical average, Ridge), **Decision Tree** | Sensor-holdout test | Data gate, loading and cleaning, EDA, features, split, report assembly | Pragun |
| **Pragun** | **Random Forest**, **PCA** (explained variance, loadings, PCA before models) | **A** Feature analysis and ablation | RF, PCA and ablation report sections; viva slides | Bhavesh |
| **Bhavesh** | **AdaBoost**, **Gradient Boosting**, **HistGradientBoosting** | **C** Model efficiency | Streamlit UI; report sections: these boosters, efficiency, UI | Anirudh |
| **Anirudh** | **XGBoost**, **LightGBM** | **E** Error analysis | Evaluation code, model saving and serving, final benchmark table, README and repo hygiene | Ajay |

Review rotation: Ajay reviews Pragun, Pragun reviews Bhavesh, Bhavesh reviews Anirudh, Anirudh reviews Ajay.
Each person's brief, with commands and deadlines: `docs/tasks/<NAME>.md`. Git workflow: `CONTRIBUTING.md`.

## Milestones

Models freeze **Tue 24 Nov**, viva swap session **Fri 27 Nov**, code freeze and tag `v1.0-submission` **Sat 28 Nov**,
submit by **Mon 30 Nov (EOD)**. Viva 1-6 Dec.

## Limitations and results

To be filled in after the real run (results table, analyses A, C, E, caveats such as the seasonal shift between train and test).
