# Anirudh: XGBoost, LightGBM, Analysis E (errors), final benchmark, repo hygiene

**Marks you own:** boosting variants (2, shared with Bhavesh), Analysis E (2), the consolidated
results table, README. Plus your report sections.
**Reviewer of your PRs:** Bhavesh. **You review:** Ajay.

## 1. Set up (today, 30 min)

1. Accept the GitHub invite from Ajay (check your email).
2. Install Python 3.11 or 3.12 and VS Code with the Python extension.
3. In the VS Code terminal:

```bash
git clone https://github.com/Ajay03299/ML-project.git
cd ML-project
python -m venv .venv
.venv\Scripts\activate             # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest                             # must say "passed"
git checkout -b anirudh/xgb-lgbm
```

4. In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → pick `.venv`.

## 2. Run your scripts on synthetic data (today)

```bash
python -m experiments.anirudh_boosting --quick
python -m experiments.run_benchmark --skip-slow
python -m experiments.analysis_e_errors --quick
python -m experiments.make_tables
```

Read those four scripts plus `src/evaluate.py` and `experiments/common.py` (you own the shared
evaluation code). Synthetic numbers mean nothing; never report them.

## 3. Switch to the real data (Thu 19 / Fri 20, when Ajay says go)

1. `git checkout main && git pull` (Ajay changes `SOURCE = "madrid"` in `src/config.py`).
2. Download `madrid_panel.npz` and `madrid_sensors.csv` from the team Google Drive link Ajay
   shares, into `data/processed/`.
3. `git checkout anirudh/xgb-lgbm && git merge main`

## 4. Your work, in order

| When | What | Command |
|---|---|---|
| Fri 20 – Sun 22 | XGBoost and LightGBM: random search, early stopping on validation, curves | `python -m experiments.anirudh_boosting` |
| Mon 23 | Refine grids; check the early-stopping round counts are sensible | |
| **Tue 24** | **Models freeze.** Merge everyone's best params, then run the FINAL benchmark on `main` | `python -m experiments.run_benchmark` then `python -m experiments.make_tables` |
| Wed 25 – Thu 26 | Analysis E on all saved models | `python -m experiments.analysis_e_errors` |
| Thu 26 | Share `results/benchmark.csv` with the team (Drive) so Bhavesh (C) and Pragun use the same numbers | |
| Fri 27 | Report sections; viva swap session; fill the README results section | |
| **Sat 28** | **Code freeze.** Fresh clone, follow the README exactly, fix anything that fails; Ajay tags `v1.0-submission` | |

## 5. Done means

- [ ] XGBoost and LightGBM tuned; rounds chosen by early stopping on validation; best params committed
- [ ] `results/benchmark_table.md`: every model, same test period, MAE / RMSE / R² / WAPE, % vs best baseline
- [ ] Paragraph: where ML beats the baselines, by how much, and whether the gain holds per sensor (the per-sensor win-rate figure)
- [ ] Analysis E: errors by peak / off-peak / night, weekday / weekend / holiday, rain / dry, sudden changes, sensor; same pattern across models?
- [ ] README: results table, exact reproduction commands, dataset access; works from a fresh clone

## 6. Report sections you write (about 2 pages)

XGBoost and LightGBM (setup, grids, early stopping, curves), the consolidated results table and its
discussion, Analysis E.

## 7. Be ready to answer in the viva

- Level-wise (XGBoost) vs leaf-wise (LightGBM) tree growth; why leaf-wise can overfit small data
- Why early stopping uses validation, never test
- Why MAPE is avoided here and WAPE used instead
- Why errors are larger at peaks and after sudden changes, and what features could help
- How `evaluate_model` makes the comparison fair (same rows, same splits, same metrics, timed the same way)

## Rules

Work on your branch, small commits, PR to `main`. Never commit `data/` or `models/`. Only you run
the final benchmark. `pytest` must pass on `main` at all times.
