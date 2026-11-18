"""Model registry: every model in the project, who owns it, and its default settings.

Each person tunes their own models in experiments/<name>_*.py. The tuned settings are saved to
results/best_params/<Model>.json, and `make_model(name)` picks them up automatically, so the
ablation (A), efficiency (C), error analysis (E) and sensor-holdout scripts all use the
TUNED version of every model without copying code.
"""
from __future__ import annotations

import json
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.baselines import ColumnBaseline, HistoricalAverage
from src.config import RANDOM_STATE

ROOT = Path(__file__).resolve().parents[1]
BEST_DIR = ROOT / "results" / "best_params"
RS = RANDOM_STATE

OWNERS = {
    "Persistence": "Ajay", "SeasonalNaive_day": "Ajay", "SeasonalNaive_week": "Ajay",
    "HistoricalAverage": "Ajay", "Ridge": "Ajay", "DecisionTree": "Ajay",
    "RandomForest": "Pragun", "PCA_Ridge": "Pragun", "PCA_RandomForest": "Pragun",
    "AdaBoost": "Bhavesh", "GradientBoosting": "Bhavesh", "HistGradientBoosting": "Bhavesh",
    "XGBoost": "Anirudh", "LightGBM": "Anirudh",
}
BASELINES = ["Persistence", "SeasonalNaive_day", "SeasonalNaive_week", "HistoricalAverage", "Ridge"]
ML_MODELS = ["DecisionTree", "RandomForest", "PCA_Ridge", "PCA_RandomForest", "AdaBoost",
             "GradientBoosting", "HistGradientBoosting", "XGBoost", "LightGBM"]

DEFAULTS = {
    "Ridge": {"alpha": 1.0},
    "DecisionTree": {"max_depth": 14, "min_samples_leaf": 20},
    "RandomForest": {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5, "max_features": 1.0},
    "PCA_Ridge": {"n_components": 0.95, "alpha": 1.0},
    "PCA_RandomForest": {"n_components": 0.95, "n_estimators": 100, "max_depth": 16, "min_samples_leaf": 5},
    "AdaBoost": {"n_estimators": 100, "learning_rate": 0.1, "base_depth": 6, "loss": "linear"},
    "GradientBoosting": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 4, "subsample": 0.8},
    "HistGradientBoosting": {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "l2_regularization": 0.0},
    "XGBoost": {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8},
    "LightGBM": {"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 31, "subsample": 0.8,
                 "subsample_freq": 1, "colsample_bytree": 0.8},
}


def best_params(name: str) -> dict:
    """Defaults, overwritten by the tuned values in results/best_params/<name>.json if it exists."""
    p = dict(DEFAULTS.get(name, {}))
    f = BEST_DIR / f"{name}.json"
    if f.exists():
        p.update(json.loads(f.read_text()))
    return p


def save_best_params(name: str, params: dict) -> Path:
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    f = BEST_DIR / f"{name}.json"
    f.write_text(json.dumps(params, indent=2, default=str))
    return f


def make_model(name: str, tuned: bool = True, **override):
    """Build a fresh, unfitted model. tuned=True starts from the saved best parameters."""
    p = best_params(name) if tuned else dict(DEFAULTS.get(name, {}))
    p.update(override)
    if name == "Persistence":
        return ColumnBaseline("flow_lag0")
    if name == "SeasonalNaive_day":
        return ColumnBaseline("flow_yesterday")
    if name == "SeasonalNaive_week":
        return ColumnBaseline("flow_lastweek")
    if name == "HistoricalAverage":
        return HistoricalAverage()
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(**p))
    if name == "DecisionTree":
        return DecisionTreeRegressor(random_state=RS, **p)
    if name == "RandomForest":
        return RandomForestRegressor(n_jobs=-1, random_state=RS, **p)
    if name == "PCA_Ridge":
        return make_pipeline(StandardScaler(), PCA(n_components=p.pop("n_components"), random_state=RS), Ridge(**p))
    if name == "PCA_RandomForest":
        k = p.pop("n_components")
        return make_pipeline(StandardScaler(), PCA(n_components=k, random_state=RS),
                             RandomForestRegressor(n_jobs=-1, random_state=RS, **p))
    if name == "AdaBoost":
        depth = p.pop("base_depth")
        return AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth, random_state=RS), random_state=RS, **p)
    if name == "GradientBoosting":
        return GradientBoostingRegressor(random_state=RS, **p)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=RS, **p)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_jobs=-1, random_state=RS, tree_method="hist", **p)
    if name == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_jobs=-1, random_state=RS, verbose=-1, **p)
    raise KeyError(f"unknown model {name!r}; known: {list(OWNERS)}")
