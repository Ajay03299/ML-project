# Bhavesh: AdaBoost, Gradient Boosting, Analysis C (efficiency), Streamlit UI

**Marks you own:** boosting variants (2, shared with Anirudh), Analysis C (2), UI (2). Plus your report sections.
**Reviewer of your PRs:** Pragun. **You review:** Anirudh.

## 1. Set up (today, 30 min)

1. Accept the GitHub invite from Ajay (check your email).
2. Install Python 3.11 or 3.12 and VS Code with the Python extension.
3. In the VS Code terminal:

```bash
git clone https://github.com/<ajay-username>/<repo-name>.git
cd <repo-name>
python -m venv .venv
.venv\Scripts\activate             # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest                             # must say "passed"
git checkout -b bhavesh/boosting
```

4. In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → pick `.venv`.

## 2. Run your scripts on synthetic data (today)

```bash
python -m experiments.bhavesh_boosting --quick
python -m experiments.run_benchmark --skip-slow        # trains all models so the UI has something to show
python -m experiments.analysis_c_efficiency --quick
streamlit run app/streamlit_app.py                     # opens the UI in your browser
```

Read `experiments/bhavesh_boosting.py`, `experiments/analysis_c_efficiency.py` and
`app/streamlit_app.py`. Synthetic numbers mean nothing; never report them.

## 3. Switch to the real data (Thu 19 / Fri 20, when Ajay says go)

1. `git checkout main && git pull` (Ajay changes `SOURCE = "madrid"` in `src/config.py`).
2. Download `madrid_panel.npz` and `madrid_sensors.csv` from the team Google Drive link Ajay
   shares, into `data/processed/`.
3. `git checkout bhavesh/boosting && git merge main`

## 4. Your work, in order

| When | What | Command |
|---|---|---|
| Fri 20 – Sun 22 | AdaBoost, Gradient Boosting, HistGradientBoosting: search + error-vs-rounds curves | `python -m experiments.bhavesh_boosting` |
| Sat 21 – Fri 27 | UI improvements (branch `bhavesh/ui`), list below | `streamlit run app/streamlit_app.py` |
| **Tue 24** | **Models freeze.** PR with `results/best_params/AdaBoost.json`, `GradientBoosting.json`, `HistGradientBoosting.json`, tuning CSVs, figures | |
| Wed 25 – Thu 26 | Analysis C after Anirudh's final benchmark run | `python -m experiments.analysis_c_efficiency` |
| Fri 27 | Report sections; viva swap session; record a 2-minute screen video of the UI as a demo backup | |

Classic `GradientBoosting` is slow on big data (it is not histogram-based); that slowness is a
finding for Analysis C. Use `--tune-rows 30000` if a search takes too long.

## 5. UI checklist (the TA wants: pick sensor + date/time → forecast next to recent trends, plus extras)

Already working: model, sensor, date and time pickers; forecast vs actual; last 24 h chart; day
view; leaderboard. Add, in this order:

- [ ] Show the unit and the horizon clearly; hide the baselines behind a "show baselines" checkbox
- [ ] Day view with several models on one chart (multiselect)
- [ ] Error of the chosen model on the chosen day (MAE) next to the chart
- [ ] Sensor map from `panel.static` latitude/longitude (`st.map`), coloured or sized by mean flow
- [ ] CSV export of the day's forecasts (`st.download_button`)
- [ ] Optional extra: SHAP explanation of one forecast for a tree model (`shap.TreeExplainer`)

The UI must keep calling `build_dataset` and the saved models; never hard-code numbers.

## 6. Done means

- [ ] Three boosting variants tuned on validation only, best params committed
- [ ] Error-vs-rounds figure; one paragraph on how AdaBoost (re-weights samples), Gradient Boosting (fits residuals) and HistGB (binned features) differ, and what the results show
- [ ] Analysis C: cost-vs-error figure with Pareto front, scaling figure, conclusion on which model you would deploy in real time
- [ ] UI runs from a fresh clone after `run_benchmark`, with the checklist items done

## 7. Report sections you write (about 2 pages)

AdaBoost / Gradient Boosting / HistGB (setup, grids, best params, curves), Analysis C, UI (screenshots).

## 8. Be ready to answer in the viva

- What a learning rate does in boosting, and why smaller rates need more rounds
- Why AdaBoost is sensitive to outliers (square loss especially)
- Why histogram-based boosting trains so much faster
- How you measured latency and model size, and why model size matters for a city deployment
- How the UI guarantees it uses the same features as the experiments

## Rules

Work on your branch, small commits, PR to `main`. Never commit `data/`, `models/` or
`results/benchmark.csv`. Do not edit `src/` signatures without telling the team. `pytest` must pass.
