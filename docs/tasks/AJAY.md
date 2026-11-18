# Ajay: data, EDA, baselines, Decision Tree, sensor holdout, report assembly

**Marks you own:** EDA and preprocessing (2), Decision Tree (1), baselines, the sensor-holdout test,
leakage prevention and split design in the report, and assembling the final PDF.
**Reviewer of your PRs:** Anirudh. **You review:** Pragun.

## Today (Wed 18 Nov)

```bash
# 1. put the starter in the repo
git clone https://github.com/Ajay03299/ML-project.git
cd ML-project
#    copy the CONTENTS of the starter folder here, then:
python -m venv .venv
.venv\Scripts\activate             # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
pytest                             # 11 passed
git add . && git commit -m "Add project starter" && git push origin main
#    send Pragun, Bhavesh and Anirudh their start-pack zips

# 2. data gate
#    Mendeley page -> "Download All" -> unzip into data/raw/madrid/
python scripts/check_madrid.py     # writes docs/data_gate.md, prints PASS / CHECK lines
```

## Thu 19 Nov: go / no-go by 12:00

If the gate passes:

```bash
git checkout -b ajay/madrid-data
python -m src.data madrid          # one time: CSV -> Parquet -> data/processed/madrid_panel.npz
#    edit src/config.py: SOURCE = "madrid"; set MADRID_UNIT in src/data.py if the gate says otherwise
pytest
git add src/config.py src/data.py docs/data_gate.md
git commit -m "Switch to Madrid data after data gate"   # PR, Anirudh reviews, merge today
```

Upload `data/processed/madrid_panel.npz` and `data/processed/madrid_sensors.csv` to a team
Google Drive folder and post the link. They are small; the raw download is not needed by the others.

If the gate fails: tell the team, download PEMS04, set `SOURCE = "pems04"`, and follow the same steps.

## Your work, in order

| When | What | Command |
|---|---|---|
| Thu 19 – Fri 20 | EDA figures and summary; decide and justify cleaning (zero-run threshold, negatives, gaps) | `python -m experiments.ajay_eda` |
| Fri 20 | Fix the split dates (look at the EDA trend; avoid splitting inside a holiday week) and document them | `src/split.py` |
| Sat 21 – Mon 23 | Baselines + Decision Tree: depth curve, grid, pruning, importances, tree plot | `python -m experiments.ajay_decision_tree` |
| **Tue 24** | **Models freeze.** PR with `results/best_params/DecisionTree.json`, `Ridge.json`, figures | |
| Wed 25 – Thu 26 | Sensor holdout on all models (after the final benchmark) | `python -m experiments.ajay_sensor_holdout` |
| Wed 25 – Sun 29 | Assemble the report from everyone's sections; final proofread; export PDF | |
| **Sat 28** | Code freeze: `git tag v1.0-submission && git push --tags` | |

## Done means

- [ ] `docs/data_gate.md` committed; unit of flow confirmed and written in README and every plot
- [ ] EDA covers sensors, period, frequency, units, missingness, rush hours, weekday vs weekend, sensor differences, weather
- [ ] Every cleaning rule justified with a number from the EDA
- [ ] Split dates documented, embargo explained, leakage test passing
- [ ] Decision Tree tuned on validation; depth curve shows where it starts to overfit
- [ ] Sensor holdout table: seen vs unseen sensors per model, with an explanation (HistoricalAverage fails on unseen sensors by design)

## Report sections you write

Overview and task, dataset and source (cite MTD on Mendeley and the Madrid Open Data Portal), EDA,
preprocessing and features, leakage prevention, split design, baselines, Decision Tree, sensor
holdout, limitations, conclusions. Then assemble and proofread the whole report.

## Be ready to answer in the viva

- Every leakage guard in `src/features.py` and why each is needed
- Why a chronological split, not a random one; what the embargo removes
- Why the depth curve shows overfitting; what cost-complexity pruning does
- Why the test period is a different season from training, and what that does to errors
