"""Synthetic traffic panel with weather, holidays and a stuck sensor.

For development, tests and demos ONLY: it lets everybody write and test code before the
real data is ready. Never report results from it. It has the same structure as a real Panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import Panel, mark_faults


def make_synthetic_panel(n_sensors: int = 24, days: int = 150, seed: int = 0,
                         start: str = "2022-06-01", step_min: int = 15) -> Panel:
    rng = np.random.default_rng(seed)
    steps_per_day = 1440 // step_min
    T = days * steps_per_day
    index = pd.date_range(start, periods=T, freq=f"{step_min}min")
    hour = np.asarray(index.hour + index.minute / 60)
    weekend = np.asarray(index.day_of_week) >= 5
    holidays = pd.to_datetime([index[0] + pd.Timedelta(days=d) for d in (20, 55, 90)]).normalize()
    holiday = np.asarray(index.normalize().isin(holidays))

    precip = np.clip(rng.gamma(0.2, 2.0, T) * (rng.random(T) < 0.08), 0, None)
    weather = pd.DataFrame({
        "temp": 15 + 10 * np.sin(2 * np.pi * (np.arange(T) / steps_per_day - 120) / 365) + rng.normal(0, 1, T),
        "precip": precip,
        "wind": np.abs(rng.normal(8, 3, T)),
    }, index=index)

    profile = 0.25 + np.exp(-((hour - 8) ** 2) / 6) + 0.9 * np.exp(-((hour - 17) ** 2) / 8)
    profile = profile * np.where(weekend | holiday, 0.55, 1.0) * (1 - 0.04 * np.clip(precip, 0, 5))
    shared = np.cumsum(rng.normal(0, 0.05, T)) * 0.02
    scale = rng.uniform(150, 600, n_sensors)
    flow = np.empty((T, n_sensors), dtype=np.float32)
    for j in range(n_sensors):
        flow[:, j] = scale[j] * (profile + shared) + rng.normal(0, 12, T)
    flow = np.clip(flow, 0, None).round()
    flow[300:420, 3 % n_sensors] = 0            # a stuck sensor: about 30 hours of zeros
    flow = mark_faults(flow, 12)
    occupancy = np.clip(flow / 900, 0, 1).astype(np.float32)
    sensors = np.arange(1000, 1000 + n_sensors)
    static = pd.DataFrame({
        "latitude": 40.42 + rng.normal(0, 0.03, n_sensors), "longitude": -3.70 + rng.normal(0, 0.04, n_sensors),
        "lanes": rng.integers(1, 5, n_sensors).astype(float), "maxspeed": rng.choice([30.0, 50.0, 70.0], n_sensors),
        "road_rank": rng.integers(0, 4, n_sensors).astype(float),
    }, index=sensors)
    return Panel(flow.astype(np.float32), index, sensors, step_min,
                 occupancy=occupancy, speed=None, weather=weather, holidays=holidays,
                 static=static, unit="veh / 15 min (synthetic)")
