"""Leakage-safe feature engineering for multi-sensor traffic forecasting.   Owner: Ajay.

Task: at origin time t (the last observed step) predict the flow of the same sensor at
t + horizon steps (default 4 steps = 60 minutes with 15-minute data).

Every feature uses only information available at time t:
  * lags, rolling statistics, occupancy, speed and weather use steps <= t
  * "seasonal" features read the value at the TARGET time of yesterday / last week, which is
    in the past because the horizon is shorter than a day
  * calendar features describe the target time (known in advance)
  * neighbour sets and sensor statistics are fitted on the TRAINING period only
Feature groups can be switched on and off; that is what the ablation (analysis A) needs.
  road        static attributes of the sensor's road (lanes, speed limit, road class); these help
              most on the sensor-holdout split, where the model has never seen the sensor
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.data import Panel
from src.split import split_dates

GROUPS = ("lags", "seasonal", "rolling", "calendar", "weather", "neighbors", "covariates", "sensor", "road")
LAGS = (0, 1, 2, 3, 4, 8, 12)      # steps back from the origin


def nearest_by_correlation(flow_train: np.ndarray, k: int) -> np.ndarray:
    """For each sensor, the k other sensors whose flow correlates most (training data only)."""
    corr = pd.DataFrame(flow_train).corr(min_periods=50).to_numpy()
    corr = np.nan_to_num(corr, nan=-1.0)
    np.fill_diagonal(corr, -2.0)
    return np.argsort(-corr, axis=1)[:, :k]


def nearest_by_distance(lat: np.ndarray, lon: np.ndarray, k: int) -> np.ndarray:
    """For each sensor, the k geographically closest other sensors (haversine)."""
    lat, lon = np.radians(lat), np.radians(lon)
    h = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    d = 2 * 6371.0 * np.arcsin(np.sqrt(h))
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def build_dataset(panel: Panel, horizon: int = 4, groups=GROUPS, train_end=None,
                  k_neighbors: int = 5, dropna: bool = True, neighbor_method: str = "corr"):
    """Return (X, y, meta). meta has origin_time, target_time, sensor; one row per X row."""
    flow, idx = panel.flow, panel.index
    T, N = flow.shape
    per_day = 1440 // panel.step_min
    per_week = 7 * per_day
    if horizon >= per_day:
        raise ValueError("horizon must be shorter than one day, otherwise seasonal features leak")
    origins = np.arange(per_week, T - horizon)
    n = len(origins)
    if train_end is None:
        train_end = split_dates(idx)[0]
    train_end_idx = int(idx.searchsorted(pd.Timestamp(train_end), side="right"))
    cols: dict[str, np.ndarray] = {}

    def at(a, back):                    # value `back` steps before each origin, shape (n, N)
        return a[origins - back]

    def add(name, arr):
        cols[name] = np.asarray(arr, dtype=np.float32).ravel()    # origin-major, sensor-minor

    target_time = idx[origins + horizon]
    slot = np.asarray((target_time.hour * 60 + target_time.minute) // panel.step_min)
    weekend = (np.asarray(target_time.day_of_week) >= 5).astype(int)
    # always kept: needed by the historical-average baseline and for error analysis
    add("slot", np.repeat(slot, N))
    add("is_weekend", np.repeat(weekend, N))
    add("sensor_id", np.tile(panel.sensors, n))

    if "lags" in groups:
        for lag in LAGS:
            add(f"flow_lag{lag}", at(flow, lag))
    if "seasonal" in groups:
        add("flow_yesterday", flow[origins + horizon - per_day])
        add("flow_lastweek", flow[origins + horizon - per_week])
    if "rolling" in groups:
        for w in (4, 12):
            r = pd.DataFrame(flow).rolling(w, min_periods=w)
            add(f"flow_mean{w}", r.mean().to_numpy()[origins])
            add(f"flow_std{w}", r.std().to_numpy()[origins])
    if "calendar" in groups:
        hour = np.asarray(target_time.hour + target_time.minute / 60)
        add("tod_sin", np.repeat(np.sin(2 * np.pi * hour / 24), N))
        add("tod_cos", np.repeat(np.cos(2 * np.pi * hour / 24), N))
        add("dayofweek", np.repeat(np.asarray(target_time.day_of_week), N))
        if panel.holidays is not None:
            is_hol = np.asarray(target_time.normalize().isin(pd.DatetimeIndex(panel.holidays).normalize()))
            add("is_holiday", np.repeat(is_hol.astype(int), N))
    if "weather" in groups and panel.weather is not None:
        for c in panel.weather.columns:                      # weather observed at the origin
            add(f"wx_{c}", np.repeat(panel.weather[c].to_numpy()[origins], N))
    if "neighbors" in groups and N > 1:
        k = min(k_neighbors, N - 1)
        st = panel.static
        if neighbor_method == "distance" and st is not None and {"latitude", "longitude"} <= set(st.columns):
            nb = nearest_by_distance(st["latitude"].to_numpy(float), st["longitude"].to_numpy(float), k)
        else:                                     # "corr": fitted on the training period only
            nb = nearest_by_correlation(flow[:train_end_idx], k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for lag in (0, 4):
                add(f"nb_flow_lag{lag}", np.nanmean(at(flow, lag)[:, nb], axis=2))
    if "covariates" in groups:
        for name, arr in (("occ", panel.occupancy), ("speed", panel.speed)):
            if arr is not None and not np.isnan(arr).all():
                for lag in (0, 4):
                    add(f"{name}_lag{lag}", at(arr, lag))
    if "sensor" in groups:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            add("sensor_mean", np.tile(np.nanmean(flow[:train_end_idx], axis=0), n))
            add("sensor_std", np.tile(np.nanstd(flow[:train_end_idx], axis=0), n))

    if "road" in groups and panel.static is not None:
        for c in ("lanes", "maxspeed", "road_rank", "oneway", "length"):
            if c in panel.static.columns:
                add(f"road_{c}", np.tile(panel.static[c].fillna(panel.static[c].median()).to_numpy(float), n))

    X = pd.DataFrame(cols)
    y = pd.Series(flow[origins + horizon].ravel(), name="flow_target", dtype="float32")
    meta = pd.DataFrame({
        "origin_time": idx[origins].repeat(N),
        "target_time": target_time.repeat(N),
        "sensor": np.tile(panel.sensors, n),
    })
    if dropna:
        ok = (~X.isna().any(axis=1)) & y.notna()
        X, y, meta = X[ok].reset_index(drop=True), y[ok].reset_index(drop=True), meta[ok].reset_index(drop=True)
    return X, y, meta
