import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from src import data as data_mod
from src.baselines import ColumnBaseline, HistoricalAverage
from src.evaluate import evaluate_model
from src.features import GROUPS, build_dataset
from src.split import pick_holdout_sensors, split_dates, split_xy, temporal_masks
from src.synthetic import make_synthetic_panel
from tests.helpers import make_pems_array


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_panel(n_sensors=20, days=120)


@pytest.fixture(scope="module")
def built(panel):
    train_end, val_end = split_dates(panel.index)
    X, y, meta = build_dataset(panel, horizon=4, train_end=train_end)
    return X, y, meta, train_end, val_end


def test_synthetic_panel_has_faults_weather_and_regular_grid(panel):
    assert panel.index[1] - panel.index[0] == pd.Timedelta("15min")
    assert np.isnan(panel.flow[:, 3]).sum() > 0 and np.isnan(panel.flow[:, 0]).sum() == 0
    assert list(panel.weather.columns) == ["temp", "precip", "wind"]


def test_dataset_has_no_nans_and_all_groups(built):
    X, y, meta, *_ = built
    assert not X.isna().any().any() and len(X) == len(y) == len(meta)
    for col in ["flow_lag0", "flow_yesterday", "flow_mean4", "tod_sin", "is_holiday", "wx_temp",
                "nb_flow_lag0", "occ_lag0", "sensor_mean"]:
        assert col in X.columns, col
    assert (meta["target_time"] - meta["origin_time"] == pd.Timedelta("60min")).all()


def test_groups_can_be_removed_for_ablation(panel):
    X, _, _ = build_dataset(panel, groups=[g for g in GROUPS if g not in ("neighbors", "weather")])
    assert not any(c.startswith(("nb_", "wx_")) for c in X.columns)


def test_no_future_information_leaks_into_features(panel):
    """Rewrite everything after time T0 and check that features of earlier origins do not move."""
    train_end, _ = split_dates(panel.index)
    X1, _, m1 = build_dataset(panel, horizon=4, train_end=train_end)
    T0 = int(panel.index.searchsorted(train_end)) + 300           # a point after the training period
    after = (np.arange(len(panel.index)) > T0)[:, None]
    rng = np.random.default_rng(1)
    wx = panel.weather.copy()
    wx[after[:, 0]] = rng.uniform(0, 50, (after.sum(), wx.shape[1]))
    changed = dataclasses.replace(
        panel,
        flow=np.where(after, rng.uniform(0, 999, panel.flow.shape), panel.flow).astype("float32"),
        occupancy=np.where(after, rng.uniform(0, 1, panel.flow.shape), panel.occupancy).astype("float32"),
        weather=wx,
    )
    X2, _, m2 = build_dataset(changed, horizon=4, train_end=train_end)
    before = (m1["origin_time"] <= panel.index[T0]).to_numpy()
    assert before.sum() > 1000
    pd.testing.assert_frame_equal(X1[before].reset_index(drop=True), X2[before].reset_index(drop=True))


def test_temporal_masks_do_not_overlap_and_respect_embargo(built):
    _, _, meta, train_end, val_end = built
    m = temporal_masks(meta, train_end, val_end)
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any() and not (m["train"] & m["test"]).any()
    assert meta.loc[m["train"], "target_time"].max() <= train_end
    assert meta.loc[m["val"], "origin_time"].min() > train_end
    assert meta.loc[m["val"], "target_time"].max() <= val_end
    assert meta.loc[m["test"], "origin_time"].min() > val_end


def test_sensor_holdout_removes_sensors_from_training(panel, built):
    X, y, meta, train_end, val_end = built
    held = pick_holdout_sensors(panel.sensors, 0.25)
    s = split_xy(X, y, meta, train_end, val_end, holdout_sensors=held)
    assert not s["meta_train"]["sensor"].isin(held).any()
    assert s["meta_unseen"]["sensor"].isin(held).all() and len(s["X_unseen"]) > 0


def test_baselines_and_model_evaluate_and_write_results(built, tmp_path):
    X, y, meta, train_end, val_end = built
    s = split_xy(X, y, meta, train_end, val_end, max_train_rows=20000)
    out = tmp_path / "bench.csv"
    persistence = evaluate_model("Persistence", ColumnBaseline("flow_lag0"), s, results_path=out)
    hist = evaluate_model("HA", HistoricalAverage(), s, results_path=out)
    tree = evaluate_model("DT", DecisionTreeRegressor(max_depth=10, random_state=0), s, results_path=out)
    saved = pd.read_csv(out)
    assert len(saved) == 3 and {"test_MAE", "train_time_s", "predict_ms_per_1000", "model_size_kb"} <= set(saved.columns)
    assert tree["test_MAE"] < persistence["test_MAE"]
    assert hist["test_R2"] > 0.5


def test_pems04_loader_on_synthetic_file(tmp_path, monkeypatch):
    path = tmp_path / "PEMS04.npz"
    np.savez_compressed(path, data=make_pems_array())
    monkeypatch.setattr(data_mod, "PEMS_NPZ", path)
    monkeypatch.setattr(data_mod, "load_pems_raw", lambda p=path: data_mod.np.load(p)["data"].astype("float32"))
    p = data_mod.load_pems04(n_sensors=10)
    assert p.flow.shape == (16992 // 3, 10) and p.step_min == 15
    assert p.index[0] == pd.Timestamp("2018-01-01") and p.holidays is not None
    X, y, meta = build_dataset(p)
    assert len(X) > 0 and not X.isna().any().any()


def test_every_registered_model_builds_and_fits(built):
    from src.models import OWNERS, make_model
    X, y, meta, train_end, val_end = built
    s = split_xy(X, y, meta, train_end, val_end, max_train_rows=2000)
    for name in OWNERS:
        m = make_model(name, tuned=False)
        if hasattr(m, "n_estimators"):
            m.set_params(n_estimators=10)
        pred = m.fit(s["X_train"], s["y_train"]).predict(s["X_val"].iloc[:50])
        assert len(pred) == 50 and np.isfinite(pred).all(), name
