"""Synthetic PEMS04-shaped array (time, sensors, [flow, occupancy, speed]) to test the PEMS loader.

For tests only. 16,992 five-minute steps from 2018-01-01 = 59 days.
"""
import numpy as np


def make_pems_array(n_sensors: int = 24, n_steps: int = 16992, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)
    hour = (t % 288) / 12.0
    weekend = ((t // 288) % 7) >= 5          # 1 Jan 2018 was a Monday
    profile = (0.25 + np.exp(-((hour - 8) ** 2) / 6) + 0.9 * np.exp(-((hour - 17) ** 2) / 8)) * np.where(weekend, 0.55, 1.0)
    scale = rng.uniform(20, 70, n_sensors)
    flow = np.clip(scale[None, :] * profile[:, None] + rng.normal(0, 2.0, (n_steps, n_sensors)), 0, None).round()
    flow[1000:1200, 3] = 0                   # a stuck sensor
    occ = np.clip(flow / 400, 0, 1)
    speed = np.clip(70 - 120 * occ + rng.normal(0, 1, flow.shape), 5, 80)
    return np.stack([flow, occ, speed], axis=2).astype(np.float32)
