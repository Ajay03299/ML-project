# Pragun: Random Forest, PCA, Analysis A (ablation)

**Marks you own:** Random Forest (1), PCA (1), Analysis A (2). Plus your report sections and viva slides.
**Reviewer of your PRs:** Ajay. **You review:** Bhavesh.

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
git checkout -b pragun/random-forest
```

4. In VS Code: Ctrl+Shift+P → "Python: Select Interpreter" → pick `.venv`.

## 2. Run your scripts on synthetic data (today)

```bash
python -m experiments.pragun_random_forest --quick     # 1 minute, checks everything runs
python -m experiments.pragun_pca --quick
python -m experiments.analysis_a_ablation --quick
```

Figures appear in `results/figures/`, tuned settings in `results/best_params/`. Read the three
scripts top to bottom; they are short. Synthetic numbers mean nothing; never report them.

## 3. Switch to the real data (Thu 19 / Fri 20, when Ajay says go)

1. `git checkout main && git pull` (Ajay changes `SOURCE = "madrid"` in `src/config.py`).
2. Download `madrid_panel.npz` and `madrid_sensors.csv` from the team Google Drive link Ajay
   shares, into `data/processed/`. (No need to download the 5+ GB raw data.)
3. `git checkout pragun/random-forest && git merge main`

## 4. Your work, in order

| When | What | Command |
|---|---|---|
| Fri 20 – Sat 21 | Random Forest: trees curve, random search, importances | `python -m experiments.pragun_random_forest` |
| Sat 21 – Sun 22 | PCA: scree, loadings, projection, PCA before Ridge and RF | `python -m experiments.pragun_pca` |
| Mon 23 | Widen or narrow the grids where the best value sits at an edge; rerun | edit the `grid = {...}` lines |
| **Tue 24** | **Models freeze.** Open a PR with `results/best_params/RandomForest.json`, `PCA_*.json`, `results/tuning/*.csv`, your figures | |
| Wed 25 – Thu 26 | Analysis A on ALL models, after everyone's best params are merged | `git pull`, then `python -m experiments.analysis_a_ablation` |
| Fri 27 | Report sections + viva slides; viva swap session | |

Slow? Add `--max-train-rows 100000` or `--tune-rows 30000`. Random Forest and ablation are the
slowest scripts in the project; start them before lunch or overnight.

## 5. Done means

- [ ] RF tuned on the validation period only (the script never touches test); best params committed
- [ ] RF beats every baseline in `results/benchmark.csv`, or you can explain why not
- [ ] PCA: how many components keep 90/95/99% of variance; what PC1 and PC2 represent (from loadings)
- [ ] PCA + Ridge and PCA + RF compared with the same models without PCA, with a conclusion
- [ ] Ablation heatmap across all models; one paragraph: which feature groups matter, and is it the same for every model?
- [ ] Every figure in the report has a sentence saying what it shows and why it makes sense for traffic

## 6. Report sections you write (about 2 pages)

Random Forest (setup, tuning grid, best params, trees curve, importances), PCA (variance,
loadings, effect on models), Analysis A (method, heatmap, interpretation). Plus the viva slides.

## 7. Be ready to answer in the viva

- Why bagging + random feature subsets reduce variance compared with one tree
- Why impurity importance and permutation importance can disagree
- Why PCA can hurt tree models (rotations destroy axis-aligned splits) and help linear ones
- Why "same time yesterday" is not leakage when the horizon is 60 minutes
- Why ablation must retrain the model rather than just zeroing a column

## Rules

Work on your branch, small commits, PR to `main`. Never commit `data/`, `models/` or
`results/benchmark.csv`. Do not edit `src/` signatures without telling the team. `pytest` must pass.
