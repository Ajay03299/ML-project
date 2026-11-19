"""The Madrid loader and the data gate, on a tiny fake download with the real file layout."""
import numpy as np
import pandas as pd
import pytest

from src import data as data_mod
from src.features import build_dataset


def make_fake_download(root, n_sensors=6, days=21, freq="15min"):
    folder = root / "Enriched Traffic Datasets for Madrid" / "Subset dataset" / "abc123hash"
    folder.mkdir(parents=True)
    per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    idx = pd.date_range("2022-06-01", periods=days * per_day, freq=freq)
    rng = np.random.default_rng(0)
    hour = np.asarray(idx.hour + idx.minute / 60)
    prof = 200 + 600 * np.exp(-((hour - 8) ** 2) / 6) + 500 * np.exp(-((hour - 18) ** 2) / 8)
    rows = []
    for k, sid in enumerate(range(3400, 3400 + n_sensors)):
        ti = np.round(prof * (0.6 + 0.15 * k) + rng.normal(0, 20, len(idx))).astype(int)
        ti[50:60] = -1                                   # "no data" convention
        df = pd.DataFrame({"id": sid, "date": idx.strftime("%Y-%m-%d %H:%M:%S"), "longitude": -3.7 + 0.01 * k,
                           "latitude": 40.4 + 0.005 * k, "traffic_intensity": ti,
                           "day_type": np.where(np.asarray(idx.normalize() == pd.Timestamp("2022-06-15")), "Holiday",
                                                np.where(np.asarray(idx.day_of_week) == 6, "Sunday", "Working day")),
                           "wind": 3.0, "temperature": 5.0 + np.sin(np.arange(len(idx)) / 96), "precipitation": 0.0,
                           "original_point": "POINT", "closest_point": "POINT", "oneway": k % 2 == 0,
                           "lanes": 2.0 + k % 3, "name": "Calle", "highway": ["primary", "tertiary"][k % 2],
                           "maxspeed": 50.0, "length": 120.0})
        rows.append(df.iloc[np.r_[0:200, 205:len(df)]])  # a few missing timestamps
    pd.concat(rows).to_csv(folder / "MTD_complete_data.csv", index=False)
    pd.DataFrame({"id": range(3400, 3400 + n_sensors), "longitude": 0.0, "latitude": 0.0}).to_csv(
        folder / "MTD_id_longitude_latitude.csv", index=False)
    np.save(folder / "MTD_adj_matrix.npy", np.eye(n_sensors))
    return root


@pytest.fixture()
def fake(tmp_path, monkeypatch):
    raw = make_fake_download(tmp_path / "raw")
    proc = tmp_path / "processed"
    monkeypatch.setattr(data_mod, "MADRID_DIR", raw)
    monkeypatch.setattr(data_mod, "MADRID_PARQUET", proc / "m.parquet")
    monkeypatch.setattr(data_mod, "MADRID_SENSORS", proc / "s.csv")
    monkeypatch.setattr(data_mod, "MADRID_PANEL", proc / "p.npz")
    return raw


def test_find_files(fake):
    f = data_mod.find_madrid_files(fake)
    assert f["data"].name == "MTD_complete_data.csv" and f["adj"].suffix == ".npy" and f["coords"] is not None


def test_load_madrid_end_to_end(fake):
    data_mod.convert_madrid_csv(data_mod.find_madrid_files(fake)["data"])
    p = data_mod.load_madrid(n_sensors=None, fault_min_run=None)
    assert p.flow.shape == (21 * 96, 6) and p.step_min == 15 and p.unit == "veh/h"
    assert np.isnan(p.flow[50:60, 0]).all()                     # negative -> NaN
    assert np.isnan(p.flow[200:205, 0]).all()                   # missing timestamps -> NaN
    assert list(p.weather.columns) == ["temp", "precip", "wind"] and not p.weather.isna().any().any()
    assert list(pd.DatetimeIndex(p.holidays).strftime("%Y-%m-%d")) == ["2022-06-15"]
    assert {"lanes", "road_rank", "latitude"} <= set(p.static.columns)
    X, y, meta = build_dataset(p, horizon=4)
    assert len(X) > 1000 and not X.isna().any().any()
    assert {"wx_temp", "is_holiday", "road_lanes", "nb_flow_lag0"} <= set(X.columns)
    sub = data_mod.load_madrid(n_sensors=3)
    assert len(sub.sensors) == 3


def test_data_gate_passes_on_good_data(fake, tmp_path):
    from scripts.check_madrid import main
    out = tmp_path / "gate.md"
    code = main(["--dir", str(fake), "--out", str(out)])
    text = out.read_text()
    assert "PASS  regular step of 15 or 60 minutes" in text and "rows:" in text
    assert code in (0, 1)             # fake data has only 6 sensors and 3 weeks, so some checks say CHECK


def test_hourly_download_is_detected(tmp_path, monkeypatch):
    raw = make_fake_download(tmp_path / "raw", days=40, freq="1h")
    proc = tmp_path / "processed"
    monkeypatch.setattr(data_mod, "MADRID_DIR", raw)
    monkeypatch.setattr(data_mod, "MADRID_PARQUET", proc / "m.parquet")
    monkeypatch.setattr(data_mod, "MADRID_SENSORS", proc / "s.csv")
    monkeypatch.setattr(data_mod, "MADRID_PANEL", proc / "p.npz")
    p = data_mod.load_madrid(n_sensors=None, fault_min_run=None)
    assert p.step_min == 60 and p.flow.shape == (40 * 24, 6)
    X, y, meta = build_dataset(p, horizon=1)                       # 1 step = 60 minutes
    assert (meta["target_time"] - meta["origin_time"] == pd.Timedelta("60min")).all()
    assert "flow_yesterday" in X and len(X) > 500 and not X.isna().any().any()


def test_weather_gaps_do_not_read_a_future_observation(tmp_path):
    index = pd.date_range("2022-06-01", periods=14, freq="15min")
    weather = [np.nan] * 10 + [25.0] * 4
    frame = pd.DataFrame({"id": 1, "date": index, "traffic_intensity": 100.0,
                          "temperature": weather})
    parquet = tmp_path / "input.parquet"
    output = tmp_path / "panel.npz"
    frame.to_parquet(parquet)
    data_mod.build_madrid_panel(parquet, output)
    with np.load(output) as panel:
        assert np.isnan(panel["weather"][:10, 0]).all()
        assert panel["weather"][10, 0] == 25.0


def test_old_cached_panel_cannot_silently_reuse_backward_filled_weather(tmp_path, monkeypatch):
    old = tmp_path / "old_panel.npz"
    np.savez_compressed(old, flow=np.ones((20, 2)), sensors=[1, 2],
                        index=pd.date_range("2022-01-01", periods=20, freq="1h").astype(str).to_numpy(dtype="U19"),
                        weather=np.ones((20, 1)), weather_cols=["temperature"], holidays=[], step_min=60)
    monkeypatch.setattr(data_mod, "MADRID_PANEL", old)
    with pytest.warns(UserWarning, match="excluding weather"):
        panel = data_mod.load_madrid()
    assert panel.weather is None and panel.step_min == 60
    np.testing.assert_allclose(panel.flow, 1)
