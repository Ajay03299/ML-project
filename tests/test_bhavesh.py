"""Regression tests for Bhavesh's validation, efficiency and serving contribution."""
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeRegressor

from app.forecast_data import day_forecasts, sensor_map, serving_dataset
from experiments.analysis_c_efficiency import compatible_benchmark, nested_indices, pareto, time_prediction
from experiments.bhavesh_boosting import GRIDS, main as boosting_main, round_parameter, search_model
from experiments.bhavesh_boosting import staged_mae
from src.artifacts import experiment_info, output_dir
from src.data import load_panel
from src.features import build_dataset
from src.models import make_model
from src.split import split_dates, split_xy


@pytest.fixture(scope="module")
def setup():
    args = SimpleNamespace(source="synthetic", n_sensors=4, horizon=4, quick=True)
    panel = load_panel("synthetic", n_sensors=4)
    train_end, val_end = split_dates(panel.index)
    X, y, meta = build_dataset(panel, train_end=train_end)
    splits = split_xy(X, y, meta, train_end, val_end, max_train_rows=600)
    info = experiment_info(panel, splits, args)
    return panel, splits, info


@pytest.mark.parametrize("name", list(GRIDS))
def test_round_selection_is_reproducible_and_never_needs_test(name, setup):
    _, splits, _ = setup
    # No test keys exist: any accidental test scoring fails immediately.
    validation_only = {k: v for k, v in splits.items() if k.endswith(("_train", "_val"))}
    best, table, curves = search_model(name, validation_only, tune_rows=600, candidates=2, quick=True)
    assert best[round_parameter(name)] == table.iloc[0]["selected_rounds"]
    assert 1 <= best[round_parameter(name)] <= 12
    if name == "HistGradientBoosting":
        assert best["early_stopping"] is False
    model = make_model(name, tuned=False, **best).fit(splits["X_train"], splits["y_train"])
    mae = np.mean(np.abs(model.predict(splits["X_val"]) - splits["y_val"].to_numpy()))
    assert mae == pytest.approx(table.iloc[0]["val_MAE"], rel=1e-6)
    assert {"round", "candidate", "train_MAE", "val_MAE"} <= set(curves)


def test_smoke_run_cannot_publish_team_parameters():
    with pytest.raises(SystemExit) as exc:
        boosting_main(["--source", "synthetic", "--publish-params"])
    assert exc.value.code == 2


@pytest.mark.parametrize("source,quick,suffix", [
    ("madrid", False, None),
    ("madrid", True, "madrid_smoke"),
    ("synthetic", False, "synthetic_smoke"),
])
def test_explicit_rerun_directory_isolates_smoke_outputs(tmp_path, setup, source, quick, suffix):
    panel, _, _ = setup
    destination = tmp_path / "madrid_v2"
    args = SimpleNamespace(source=source, quick=quick, output_dir=destination, horizon=4)
    expected = destination if suffix is None else destination / suffix
    assert output_dir(args, panel) == expected
    assert expected.is_dir()


def test_custom_artifact_directory_still_publishes_to_team_tuning(tmp_path, setup, monkeypatch):
    from experiments import bhavesh_boosting as boosting
    panel, splits, _ = setup
    destination = tmp_path / "reports" / "rerun"
    canonical = tmp_path / "team_tuning"
    monkeypatch.setattr(boosting, "get_data", lambda args: (panel, splits))
    monkeypatch.setattr(boosting, "TUNE_DIR", canonical)
    published = []
    monkeypatch.setattr(boosting, "save_best_params", lambda name, params: published.append(name))
    best = {"n_estimators": 2}
    table = pd.DataFrame([{"candidate": 0, "selected_rounds": 2, "val_MAE": 1.0, "params": best}])
    curves = pd.DataFrame({"round": [1, 2], "candidate": [0, 0],
                           "train_MAE": [2.0, 1.0], "val_MAE": [3.0, 1.0]})
    monkeypatch.setattr(boosting, "search_model", lambda *args, **kwargs: (best, table, curves))
    boosting.main(["--source", "madrid", "--horizon", "4", "--n-sensors", "4",
                   "--models", "AdaBoost", "--publish-params", "--output-dir", str(destination)])
    assert published == ["AdaBoost"]
    assert (destination / "tuning/AdaBoost.csv").exists()
    pd.testing.assert_frame_equal(pd.read_csv(canonical / "AdaBoost.csv"), table.drop(columns="params"))
    assert not (tmp_path / "tuning/AdaBoost.csv").exists()


def test_subset_freeze_excludes_unrequested_models(tmp_path, monkeypatch, setup):
    from experiments import bhavesh_final_benchmark as benchmark
    from src import evaluate
    panel, splits, info = setup
    destination = tmp_path / "rerun"
    best = tmp_path / "best"
    best.mkdir()
    # No teammate parameter files exist: a subset must require only its own.
    (best / "Ridge.json").write_text('{"alpha": 0.1}')
    monkeypatch.setattr(benchmark, "BEST_DIR", best)
    monkeypatch.setattr(benchmark, "get_data", lambda args: (panel, splits))
    monkeypatch.setattr(benchmark, "best_params", lambda name: {"alpha": 0.1})
    monkeypatch.setattr(benchmark, "make_model", lambda name: object())
    monkeypatch.setattr(benchmark, "set_threads", lambda model, threads: model)
    called = []
    def record(name, splits, panel, args, model):
        called.append(name)
    monkeypatch.setattr(benchmark, "finalize", record)
    csv = tmp_path / "benchmark.csv"
    monkeypatch.setattr(evaluate, "RESULTS_CSV", csv)
    args = ["--source", "synthetic", "--horizon", "4", "--n-sensors", "4",
            "--output-dir", str(destination), "--models", "Ridge", "Ridge"]
    pd.DataFrame([
        {"run_id": info["run_id"], "tag": "all", "model": "Ridge", "test_MAE": 1},
        {"run_id": info["run_id"], "tag": "all", "model": "DecisionTree", "test_MAE": 2},
    ]).to_csv(csv, index=False)
    benchmark.main(args)
    out = destination / "synthetic_smoke"
    freeze = json.loads((out / "benchmark_freeze.json").read_text())
    assert called == freeze["models_requested"] == ["Ridge"]
    assert "DecisionTree" in freeze["models_excluded"]
    assert pd.read_csv(out / "frozen_benchmark.csv")["model"].tolist() == ["Ridge"]


def test_pareto_handles_equal_cost_and_error_ties_independent_of_order():
    df = pd.DataFrame({"model": ["bad-tie", "best-tie", "same", "fast", "dominated"],
                       "cost": [2, 2, 2, 1, 4], "test_MAE": [6, 3, 3, 5, 8]})
    expected = {"best-tie", "same", "fast"}
    assert set(pareto(df, "cost")["model"]) == expected
    assert set(pareto(df.iloc[::-1], "cost")["model"]) == expected


@pytest.mark.parametrize("cost", [np.nan, np.inf, -1])
def test_pareto_rejects_invalid_measurements(cost):
    with pytest.raises(ValueError):
        pareto(pd.DataFrame({"cost": [cost], "test_MAE": [3]}), "cost")


def test_mixed_horizons_sources_and_samples_cannot_enter_efficiency_plot(setup):
    _, _, info = setup
    good = {"model": "Ridge", "tag": "all", "run_id": info["run_id"], "source": "synthetic",
            "horizon_min": 60, "n_train": info["n_train"], "n_features": len(info["columns"]), "test_MAE": 1}
    rows = [good, dict(good, model="other-horizon", horizon_min=15),
            dict(good, model="other-source", source="madrid"),
            dict(good, model="other-data", run_id="old"), dict(good, model="other-size", n_train=99)]
    assert compatible_benchmark(pd.DataFrame(rows), info)["model"].tolist() == ["Ridge"]
    with pytest.raises(ValueError, match="provenance"):
        compatible_benchmark(pd.DataFrame(rows).drop(columns="run_id"), info)


def test_nested_scaling_samples_are_reproducible_and_actually_nested():
    first = nested_indices(100, [10, 25, 100, 1000])
    second = nested_indices(100, [10, 25, 100])
    assert set(first[10]) < set(first[25]) < set(first[100])
    for n in second:
        np.testing.assert_array_equal(first[n], second[n])


def test_serving_features_equal_held_out_experiment_features(setup):
    panel, splits, info = setup
    X, y, meta = serving_dataset(panel, info)
    pd.testing.assert_frame_equal(X, splits["X_test"])
    pd.testing.assert_series_equal(y, splits["y_test"])
    pd.testing.assert_frame_equal(meta, splits["meta_test"])


def test_changed_traffic_and_static_attributes_are_rejected(setup):
    panel, _, info = setup
    flow = panel.flow.copy()
    flow[-1, 0] += 1
    with pytest.raises(ValueError, match="changed since training"):
        serving_dataset(dataclasses.replace(panel, flow=flow), info)
    static = panel.static.copy()
    static.iloc[0, 0] += 0.01
    with pytest.raises(ValueError, match="changed since training"):
        serving_dataset(dataclasses.replace(panel, static=static), info)
    weather = panel.weather.copy()
    weather.iloc[-1, 0] += 1
    with pytest.raises(ValueError, match="changed since training"):
        serving_dataset(dataclasses.replace(panel, weather=weather), info)


def test_target_day_includes_previous_day_origins_and_aligned_actuals(setup):
    _, splits, _ = setup
    X, y, meta = splits["X_test"], splits["y_test"], splits["meta_test"]
    sensor = meta["sensor"].iloc[0]
    date = sorted(meta["target_time"].dt.date.unique())[2]
    model = make_model("Persistence", tuned=False).fit(X, y)
    day = day_forecasts(X, y, meta, {"Persistence": model}, sensor, date, ["Persistence"])
    assert len(day) == 96
    assert day.iloc[0]["target_time"] == pd.Timestamp(date)
    assert day.iloc[0]["origin_time"].date() < date
    np.testing.assert_allclose(day["actual"], y.loc[day.index])
    assert day_forecasts(X, y, meta, {}, -999, date, []).empty


def test_map_uses_training_flow_and_handles_no_coordinates(setup):
    panel, _, info = setup
    points = sensor_map(panel, info["train_end"])
    assert len(points) == len(panel.sensors)
    expected = np.nanmean(panel.flow[panel.index <= pd.Timestamp(info["train_end"])], axis=0)
    np.testing.assert_allclose(points["mean_flow"], expected, rtol=1e-5)
    assert (points["marker_size"] > 0).all()
    assert sensor_map(dataclasses.replace(panel, static=None), info["train_end"]).empty


def test_latency_reports_actual_batch_size_without_assuming_1000_rows(setup):
    _, splits, _ = setup
    model = DecisionTreeRegressor(max_depth=2).fit(splits["X_train"], splits["y_train"])
    single, batch, n = time_prediction(model, splits["X_val"].iloc[:13], repeats=3, batch_size=1000)
    assert n == 13 and len(single) == len(batch) == 3
    assert min(single + batch) >= 0


def test_saved_model_manifest_and_streamlit_interactions(setup, tmp_path, monkeypatch):
    from experiments import common
    from streamlit.testing.v1 import AppTest
    panel, splits, info = setup
    args = SimpleNamespace(source="synthetic", n_sensors=4, horizon=4)
    model_dir = tmp_path / "models"
    monkeypatch.setattr(common, "MODELS_DIR", model_dir)
    models = {"Persistence": make_model("Persistence", tuned=False),
              "DecisionTree": DecisionTreeRegressor(max_depth=3, random_state=42)}
    for name, model in models.items():
        common.save_for_ui(name, model.fit(splits["X_train"], splits["y_train"]), splits, panel, args)
    saved = json.loads((model_dir / "meta.json").read_text())
    assert saved["run_id"] == info["run_id"]
    assert set(saved["model_runs"]) == set(models)
    script = (common.ROOT / "app/streamlit_app.py").read_text(encoding="utf-8")
    script = script.replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(tmp_path)!r})')
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert app.selectbox(key="model").options == ["DecisionTree"]
    assert any("SYNTHETIC DEMO" in warning.value for warning in app.warning)
    assert any("day MAE" in metric.label for metric in app.metric)
    assert len(app.get("download_button")) == 1
    app.checkbox[0].check().run()
    assert not app.exception
    assert "Persistence" in app.selectbox(key="model").options
    assert app.selectbox(key="model").value == "DecisionTree"
    app.multiselect[0].set_value(["DecisionTree", "Persistence"]).run()
    assert not app.exception
    assert len(app.dataframe) >= 2
    app.time_input(key="time").set_value(pd.Timestamp("2000-01-01 08:07").time()).run()
    assert not app.exception
    assert any("No valid forecast" in warning.value for warning in app.warning)
    app.selectbox(key="model").set_value("Persistence").run()
    app.checkbox[0].uncheck().run()
    assert not app.exception
    assert app.selectbox(key="model").value == "DecisionTree"
    assert "Persistence" not in app.selectbox(key="model").options


def test_ui_with_no_saved_models_explains_setup(tmp_path):
    from experiments.common import ROOT
    from streamlit.testing.v1 import AppTest
    script = (ROOT / "app/streamlit_app.py").read_text(encoding="utf-8")
    script = script.replace('ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(tmp_path)!r})')
    app = AppTest.from_string(script).run()
    assert not app.exception
    assert any("No trained models" in message.value for message in app.info)


def test_hourly_data_rejects_unobservable_fifteen_minute_horizon(monkeypatch, setup):
    from experiments import common
    panel, _, _ = setup
    monkeypatch.setattr(common, "load_panel", lambda *a, **kw: dataclasses.replace(panel, step_min=60))
    args = common.parser("test").parse_args(["--horizon-min", "15"])
    with pytest.raises(ValueError, match="cannot be evaluated"):
        common.get_data(args)


def test_cached_adaboost_stages_equal_sklearn_weighted_medians(setup):
    _, splits, _ = setup
    model = make_model("AdaBoost", tuned=False, n_estimators=12, base_depth=3)
    model.fit(splits["X_train"], splits["y_train"])
    X, y = splits["X_val"].iloc[:53], splits["y_val"].iloc[:53]
    expected = [np.mean(np.abs(y.to_numpy() - p)) for p in model.staged_predict(X)]
    np.testing.assert_allclose(staged_mae(model, X, y, chunk_rows=7), expected, rtol=1e-12)


def test_weekly_gain_interval_is_reproducible_and_preserves_constant_gain():
    from experiments.bhavesh_consistency import weekly_gain_interval
    np.testing.assert_allclose(weekly_gain_interval(np.full(28, 5.0)), [5.0, 5.0])
    first = weekly_gain_interval(np.arange(28, dtype=float), seed=42)
    np.testing.assert_array_equal(first, weekly_gain_interval(np.arange(28, dtype=float), seed=42))
    with pytest.raises(ValueError, match="two complete"):
        weekly_gain_interval([1.0, 2.0])


def test_reused_training_time_cannot_be_reported_as_repeated_fits():
    from experiments.analysis_c_efficiency import main
    with pytest.raises(SystemExit) as exc:
        main(["--reuse-benchmark-fit", "--repeats", "3"])
    assert exc.value.code == 2


def test_reference_tuning_rejects_smoke_runs_before_publishing():
    from experiments.bhavesh_reference_tuning import main
    for arguments in (["--quick"], ["--source", "synthetic"], ["--threads", "0"]):
        with pytest.raises(SystemExit) as exc:
            main(arguments)
        assert exc.value.code == 2


def test_supporting_tuning_keeps_shared_owner_outputs_untouched(setup, tmp_path, monkeypatch):
    from experiments import common
    _, splits, _ = setup
    shared = tmp_path / "shared"
    shared.mkdir()
    sentinel = shared / "Ridge.csv"
    sentinel.write_text("owner results\n")
    monkeypatch.setattr(common, "TUNE_DIR", shared)
    def reject_shared_publish(*args):
        raise AssertionError("A supporting search must not publish shared parameters.")
    monkeypatch.setattr(common, "save_best_params", reject_shared_publish)
    destination = tmp_path / "references"
    best, table = common.tune("Ridge", {"alpha": [1.0]}, splits, 500, out_dir=destination)
    assert sentinel.read_text() == "owner results\n"
    assert json.loads((destination / "best_params/Ridge.json").read_text()) == best
    assert len(pd.read_csv(destination / "tuning/Ridge.csv")) == len(table) == 1
