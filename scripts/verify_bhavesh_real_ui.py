"""Verify real UI downloads against frozen inference and optionally record a tour.

Optional dependency: playwright with Chromium installed. Start Streamlit first.
"""

import argparse
import json
import re
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_panel
from app.forecast_data import serving_dataset, day_forecasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8507")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    output = ROOT / "docs/screenshots"
    output.mkdir(parents=True, exist_ok=True)
    demo = ROOT / "output/demo"
    demo.mkdir(parents=True, exist_ok=True)
    info = json.loads((ROOT / "models/meta.json").read_text())
    if (
        info["source"] != "madrid"
        or info["step_min"] != 60
        or info["horizon_min"] != 60
    ):
        raise ValueError(
            "This verification expects the frozen hourly Madrid experiment."
        )

    def caption(page, text, seconds):
        if not args.record:
            return
        page.evaluate(
            """text => {
            document.getElementById('demo-caption')?.remove();
            const el = document.createElement('div');
            el.id = 'demo-caption'; el.textContent = text;
            Object.assign(el.style, {position:'fixed', bottom:'12px', left:'20px', right:'20px',
                zIndex:'99999', padding:'12px 18px', background:'#123448', color:'white',
                fontSize:'17px', borderRadius:'8px', pointerEvents:'none'});
            document.body.appendChild(el);
        }""",
            text,
        )
        page.wait_for_timeout(seconds * 1000)
        page.evaluate("document.getElementById('demo-caption')?.remove()")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        options = {"viewport": {"width": 1024, "height": 900}, "device_scale_factor": 1}
        if args.record:
            options.update(
                record_video_dir=str(demo / "raw"),
                record_video_size={"width": 1024, "height": 900},
            )
        context = browser.new_context(**options)
        page = context.new_page()
        page.goto(args.url, wait_until="networkidle")
        page.get_by_text("Target day", exact=False).first.wait_for(timeout=120000)
        page.get_by_test_id("stSelectbox").first.get_by_role("combobox").click()
        page.get_by_role("option", name="HistGradientBoosting", exact=True).click()
        expect(
            page.get_by_test_id("stMetricLabel").filter(
                has_text="HistGradientBoosting day MAE"
            )
        ).to_be_visible(timeout=60000)
        expect(page.get_by_test_id("stMultiSelect")).to_have_count(1, timeout=60000)
        page.get_by_test_id("stMain").evaluate("el => el.scrollTop = 0")
        page.screenshot(path=str(output / "madrid_forecast.png"))
        page.get_by_test_id("stSelectbox").first.evaluate(
            "el => el.scrollIntoView({block: 'start'})"
        )
        page.get_by_test_id("stMain").evaluate("el => el.scrollTop -= 65")
        page.screenshot(path=str(output / "madrid_forecast_detail.png"))
        caption(
            page,
            "1/7 - One-hour Madrid forecasts in veh/h. The date is a held-out historical target, not a live feed.",
            16,
        )
        page.get_by_text("Inputs behind this forecast", exact=True).click()
        page.get_by_text(
            "Inputs behind this forecast", exact=True
        ).scroll_into_view_if_needed()
        caption(
            page,
            "2/7 - Actual lag, calendar and road inputs from the shared feature pipeline. Unverified legacy weather is excluded.",
            12,
        )
        page.get_by_text("Inputs behind this forecast", exact=True).click()
        for name in ["GradientBoosting", "AdaBoost"]:
            control = page.get_by_test_id("stMultiSelect").get_by_role("combobox")
            control.click()
            page.get_by_role("option", name=name, exact=True).click()
            control.press("Escape")
            expect(
                page.get_by_test_id("stVegaLiteChart").last.get_by_text(
                    name, exact=True
                )
            ).to_be_visible(timeout=60000)
            expect(page.get_by_test_id("stMultiSelect")).to_have_count(1, timeout=60000)
        page.get_by_role("heading", name=re.compile(r"^Target day ")).evaluate(
            "el => el.scrollIntoView({block: 'start'})"
        )
        page.get_by_test_id("stMain").evaluate("el => el.scrollTop -= 60")
        page.screenshot(path=str(output / "madrid_comparison.png"))
        caption(
            page,
            "3/7 - Compare the three boosters on the same target day. MAE/RMSE use available paired forecasts only.",
            16,
        )
        with page.expect_download() as event:
            page.get_by_role("button", name="Download day's forecasts (CSV)").click()
        event.value.save_as(str(output / "madrid_export.csv"))
        heading_box = page.get_by_role("heading", name=re.compile(r"^Target day ")).bounding_box()
        button_box = page.get_by_role("button", name="Download day's forecasts (CSV)").bounding_box()
        top = max(0, heading_box["y"] - 8)
        left = max(0, heading_box["x"] - 16)
        bottom = min(page.viewport_size["height"], button_box["y"] + button_box["height"] + 8)
        page.screenshot(path=str(output / "madrid_comparison.png"), clip={
            "x": left, "y": top, "width": min(page.viewport_size["width"] - left, heading_box["width"] + 32),
            "height": bottom - top})
        caption(
            page,
            "4/7 - Export the actual forecasts, observations, origin/target times, units, horizon and run ID as CSV.",
            4,
        )
        page.get_by_role("tab", name="Sensor map").click()
        page.get_by_text("Larger markers", exact=False).wait_for()
        page.get_by_text("Larger markers", exact=False).evaluate(
            "el => el.scrollIntoView({block: 'start'})"
        )
        page.get_by_test_id("stMain").evaluate("el => el.scrollTop -= 60")
        page.screenshot(path=str(output / "madrid_sensor_map.png"))
        caption(
            page,
            "5/7 - Sensor positions come from the dataset; marker size uses training-period flow. A coordinate table remains available.",
            16,
        )
        page.get_by_role("tab", name="Frozen benchmark").click()
        page.get_by_text("Same dataset, features", exact=False).wait_for()
        page.screenshot(path=str(output / "madrid_leaderboard.png"))
        caption(
            page,
            "6/7 - Frozen 14-model comparison: identical data, feature schema, training sample and forecast horizon.",
            16,
        )
        page.get_by_role("tab", name="Forecast & compare").click()
        page.get_by_text("Show baselines", exact=True).click()
        expect(page.get_by_role("checkbox", name="Show baselines")).to_be_checked()
        page.get_by_text("Baselines are included in the model selector.", exact=True).wait_for(timeout=60000)
        expect(page.get_by_test_id("stSelectbox").first.get_by_role("combobox")).to_have_attribute(
            "aria-label", "Selected HistGradientBoosting. Model", timeout=60000)
        page.get_by_test_id("stSelectbox").first.get_by_role("combobox").click()
        expect(
            page.get_by_role("option", name="Persistence", exact=True)
        ).to_be_visible(timeout=60000)
        page.get_by_role("option", name="HistGradientBoosting", exact=True).click()
        expect(
            page.get_by_test_id("stMetricLabel").filter(
                has_text="HistGradientBoosting day MAE"
            )
        ).to_be_visible(timeout=60000)
        expect(page.get_by_test_id("stMultiSelect")).to_have_count(1, timeout=60000)
        alternative_sensor = str(info["sensor_ids"][1])
        page.get_by_test_id("stSelectbox").nth(1).get_by_role("combobox").click()
        page.get_by_role("option", name=alternative_sensor, exact=True).click()
        date_control = page.get_by_test_id("stDateInput").locator("input")
        date_control.fill("2023/10/12")
        date_control.press("Enter")
        page.get_by_text(
            f"Target day 2023-10-12 - sensor {alternative_sensor}", exact=True
        ).wait_for(timeout=60000)
        expect(page.get_by_test_id("stMultiSelect")).to_have_count(1, timeout=60000)
        with page.expect_download() as changed_event:
            page.get_by_role("button", name="Download day's forecasts (CSV)").click()
        changed_event.value.save_as(str(output / "madrid_export_changed_controls.csv"))
        changed = pd.read_csv(output / "madrid_export_changed_controls.csv")
        target_times = pd.to_datetime(changed.target_time)
        chosen = int(np.argmin(np.abs(target_times.dt.hour.to_numpy() - 17)))
        target_clock = target_times.iloc[chosen].strftime("%H:%M")
        time_control = page.get_by_test_id("stTimeInput").locator("input")
        time_control.fill(target_clock)
        time_control.press("Enter")
        page.get_by_text(f"forecast target: 2023-10-12 {target_clock}", exact=False).wait_for(timeout=60000)
        expected_value = float(changed.iloc[chosen]["HistGradientBoosting"])
        expect(page.get_by_test_id("stMetricValue").first).to_have_text(
            f"{expected_value:,.1f}", timeout=60000
        )
        page.get_by_test_id("stMain").evaluate("el => el.scrollTop = 0")
        caption(
            page,
            "7/7 - Change sensor, target date and time. The displayed prediction matches direct frozen-model inference; baselines remain available.",
            12,
        )
        if page.get_by_test_id("stException").count():
            raise AssertionError("The browser displayed an application exception.")
        video = page.video
        context.close()
        if args.record:
            video.save_as(str(demo / "Bhavesh_Madrid_Demo_Raw.webm"))
        browser.close()
    panel = load_panel("madrid", n_sensors=info["n_sensors"])
    X, y, meta = serving_dataset(panel, info)
    names = ["HistGradientBoosting", "GradientBoosting", "AdaBoost"]
    models = {name: joblib.load(ROOT / "models" / f"{name}.joblib") for name in names}
    required = {
        "origin_time",
        "target_time",
        "sensor",
        "actual",
        "unit",
        "horizon_min",
        "source",
        "run_id",
    }
    cases = []
    for filename in ["madrid_export.csv", "madrid_export_changed_controls.csv"]:
        frame = pd.read_csv(output / filename)
        present = [name for name in names if name in frame]
        assert required | set(present) <= set(frame)
        assert present
        if filename == "madrid_export.csv":
            assert set(present) == set(names)
        assert frame.source.eq("madrid").all() and frame.run_id.eq(info["run_id"]).all()
        assert frame.horizon_min.eq(60).all() and frame.unit.eq("veh/h").all()
        origins, targets = pd.to_datetime(frame.origin_time), pd.to_datetime(
            frame.target_time
        )
        assert (targets - origins).eq(pd.Timedelta("60min")).all()
        expected = day_forecasts(
            X,
            y,
            meta,
            models,
            int(frame.sensor.iloc[0]),
            targets.iloc[0].date(),
            present,
        )
        assert len(frame) == len(expected) and len(frame) > 0
        for name in present:
            np.testing.assert_allclose(frame[name], expected[name], rtol=1e-10)
        np.testing.assert_allclose(frame.actual, expected.actual)
        cases.append(
            {
                "file": filename,
                "sensor": int(frame.sensor.iloc[0]),
                "day": str(targets.iloc[0].date()),
                "rows": len(frame),
                "models": present,
                "checked_predictions": len(frame) * len(present),
            }
        )
    result = {
        "status": "pass",
        "run_id": info["run_id"],
        "rows": sum(case["rows"] for case in cases),
        "checked_predictions": sum(case["checked_predictions"] for case in cases),
        "cases": cases,
        "models": names,
        "sensor": int(frame.sensor.iloc[0]),
        "target_day": str(targets.iloc[0].date()),
        "horizon_min": 60,
        "unit": "veh/h",
        "checks": [
            "model switching",
            "sensor/date/time change matches displayed prediction",
            "three-model chart",
            "baseline toggle preserves the selected model",
            "real CSV download",
            "direct prediction equality",
            "actual/origin/target alignment",
            "map and leaderboard tabs",
        ],
        "map_tiles": "inspect screenshots; external tile service is not guaranteed",
        "video_recorded": args.record,
    }
    (output / "madrid_browser_verification.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
