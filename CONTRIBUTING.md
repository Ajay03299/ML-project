# Team workflow

1. Never commit directly to `main`. One branch per task, named after the owner: `pragun/random-forest`, `ajay/madrid-loader`.
2. Start each day with `git pull --rebase origin main`.
3. Commit small and often, with messages that say what changed ("Add PCA scree plot", not "update").
4. Open a Pull Request and request your reviewer (Ajay reviews Pragun, Pragun reviews Bhavesh, Bhavesh reviews Anirudh,
   Anirudh reviews Ajay). Merge after one approval. Run `pytest` before opening the PR.
5. One owner per notebook. Shared logic goes in `src/*.py`, not in notebooks.
6. Always call `src.evaluate.evaluate_model` so every model lands in `results/benchmark.csv`. Ablation and robustness
   runs use their own `tag`.
7. Do not change the signatures in `src/data.py`, `src/features.py`, `src/split.py`, `src/models.py`, `src/evaluate.py`
   or `experiments/common.py`, or the settings in `src/config.py`, without telling the team.
8. Fixed `random_state=42`. No large files, virtual environments, models or secrets in git.
   Commit your `results/best_params/<Model>.json`, `results/tuning/<Model>.csv` and your figures;
   do NOT commit `results/benchmark.csv` (everyone's runs write to it). Anirudh runs the final benchmark.
9. Do not weaken the leakage test in `tests/test_pipeline.py`.


