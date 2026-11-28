# Traffic Flow Forecasting in Madrid

ML course project, Topic 4 · **Team:** Ajay, Pragun, Bhavesh, Anirudh
**Repo:** https://github.com/Ajay03299/ML-project (submitted version: tag `v1.0-submission`) · **Report:** [`report/Traffic_Flow_Forecasting_Report.pdf`](report/Traffic_Flow_Forecasting_Report.pdf)

We predict the traffic flow of a Madrid road sensor **one hour ahead**, using only data available at the time of the
forecast. Every model uses the same features, the same time-ordered train/validation/test split and the same metrics,
and a Streamlit app shows the forecasts.

## Results

Test period 15 Aug - 31 Oct 2023, 60 sensors. MAE is in vehicles per hour (lower is better).

| Model | Test MAE | WAPE |
|---|---|---|
| **XGBoost** | **39.9** | 6.5% |
| LightGBM | 40.2 | 6.6% |
| Random Forest | 41.9 | 6.9% |
| HistGradientBoosting | 43.0 | 7.0% |
| Gradient Boosting | 45.3 | 7.4% |
| PCA + Random Forest | 48.6 | 8.0% |
| AdaBoost | 51.3 | 8.4% |
| Decision Tree | 53.3 | 8.7% |
| Ridge (baseline) | 65.1 | 10.7% |
| Same hour last week (baseline) | 75.4 | 12.3% |
| PCA + Ridge | 76.8 | 12.6% |
| Historical average (baseline) | 105.3 | 17.2% |
| Flow now (baseline) | 109.9 | 18.0% |
| Same hour yesterday (baseline) | 117.2 | 19.2% |

XGBoost is 39% better than Ridge and 47% better than "same hour last week". Full table:
[`results/benchmark_table.md`](results/benchmark_table.md). Analyses: [feature ablation](results/ablation.csv),
[error breakdown](results/error_breakdown.csv), [sensor holdout](results/sensor_holdout.csv),
[efficiency](results/bhavesh/).

## Dataset

**Madrid Traffic Dataset (subset)**, Gomez and Ilarri, Mendeley Data V4,
[DOI 10.17632/697ht4f65b.4](https://data.mendeley.com/datasets/697ht4f65b/4), licence
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Original data from the
[Madrid City Council Open Data Portal](https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico);
dataset paper: [doi.org/10.1109/OJITS.2025.3637305](https://doi.org/10.1109/OJITS.2025.3637305). Details in
[`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md).

300 sensors, hourly, June 2022 - October 2023, flow in vehicles per hour, plus weather, holidays, road attributes and
sensor locations. We use 60 evenly spaced sensors.

**Access:** nothing to download. The processed data is in `data/processed/`. To rebuild it from the original files,
download the subset `.rar` from the Mendeley page, extract it into `data/raw/madrid/`, then run
`python scripts/check_madrid.py` and `python -m src.data madrid`.

## Setup

Python 3.11-3.13. On macOS run `brew install libomp` first (needed by XGBoost and LightGBM).

```bash
git clone https://github.com/Ajay03299/ML-project.git
cd ML-project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                             # all tests should pass
```

## Run the user interface

```bash
python -m experiments.run_benchmark --skip-slow   # trains the models the app loads (skips the 4 slowest)
python -m streamlit run app/streamlit_app.py      # opens http://localhost:8501
```

Pick a model, sensor, date and time to see the one-hour forecast next to the last 24 hours of traffic. The app also
compares models on a day, shows a sensor map and the leaderboard, and exports forecasts as CSV. A static showcase of
the results is in `web_gui/index.html` (open it in a browser).

## Reproduce the experiments

The tuned settings are already saved in `results/best_params/`, so steps 1 and 3 reproduce the reported numbers.
Shared settings (60 sensors, 60-minute horizon, seed 42) are in `src/config.py`; add `--quick` to any script for a
one-minute check.

```bash
# 1. EDA
python -m experiments.ajay_eda

# 2. Tuning (optional; writes results/best_params/)
python -m experiments.ajay_decision_tree                  # baselines, Ridge, Decision Tree
python -m experiments.pragun_random_forest
python -m experiments.pragun_pca
python -m experiments.bhavesh_boosting --publish-params   # AdaBoost, Gradient Boosting, HistGradientBoosting
python -m experiments.anirudh_boosting                    # XGBoost, LightGBM

# 3. Final benchmark (about 1 hour) and results table
python -m experiments.run_benchmark
python -m experiments.make_tables

# 4. Analyses
python -m experiments.analysis_a_ablation --max-train-rows 100000   # A: feature ablation
python -m experiments.analysis_c_efficiency                         # C: efficiency
python -m experiments.analysis_e_errors                             # E: error analysis
python -m experiments.ajay_sensor_holdout --max-train-rows 100000   # sensor holdout
```

## Code layout

| Path | What it does |
|---|---|
| `src/` | Data loading and cleaning, features, split, models, evaluation |
| `experiments/` | One script per model family and per analysis |
| `app/streamlit_app.py` | The user interface |
| `tests/` | Tests, including one that proves features never use future data |
| `results/` | Tuned settings, benchmark, analyses and figures |

## Team

| Member | Models | Analysis | Also |
|---|---|---|---|
| Ajay | Baselines, Ridge, Decision Tree | Sensor holdout | Dataset, data checks, EDA, features, split, report |
| Pragun | Random Forest, PCA | A: feature ablation | Web showcase |
| Bhavesh | AdaBoost, Gradient Boosting, HistGradientBoosting | C: efficiency | Streamlit UI |
| Anirudh | XGBoost, LightGBM | E: error analysis | Final benchmark |
