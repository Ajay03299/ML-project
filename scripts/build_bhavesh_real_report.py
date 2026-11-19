"""Generate Bhavesh's measured report sections directly from frozen artifacts.

Optional export dependency: reportlab. Figures and screenshots must already exist.
"""

from pathlib import Path
import json
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
)
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/bhavesh/madrid_h60_development"
PDF_DIR = ROOT / "output/pdf"
BOOSTERS = ["AdaBoost", "GradientBoosting", "HistGradientBoosting"]


def load_inputs():
    bench = pd.read_csv(OUT / "frozen_benchmark.csv").set_index("model")
    efficiency = pd.read_csv(OUT / "efficiency.csv").set_index("model")
    scaling = pd.read_csv(OUT / "scaling.csv")
    gains = pd.read_csv(OUT / "frozen_gain_consistency.csv").set_index("model")
    freeze = json.loads((OUT / "benchmark_freeze.json").read_text())
    if len(bench) != 14 or len(efficiency) != 14 or len(scaling) != 40:
        raise ValueError(
            "Expected 14 frozen models and 40 scaling observations; finish the pipeline first."
        )
    if (
        not bench.run_id.eq(freeze["run_id"]).all()
        or not efficiency.run_id.eq(freeze["run_id"]).all()
    ):
        raise ValueError("Report artifacts come from different runs.")
    tuning_info = json.loads((OUT / "tuning_manifest.json").read_text())
    if tuning_info["run_id"] != freeze["run_id"] or not gains.run_id.eq(freeze["run_id"]).all():
        raise ValueError("Tuning or gain consistency belongs to another setup.")
    browser = json.loads((ROOT / "docs/screenshots/madrid_browser_verification.json").read_text())
    if browser["status"] != "pass" or browser["run_id"] != freeze["run_id"]:
        raise ValueError("Real browser verification must pass for this experiment.")
    if not (ROOT / "output/demo/Bhavesh_Madrid_Demo.webm").exists():
        raise FileNotFoundError("Finish the demonstration video before generating the final report.")
    return bench, efficiency, scaling, gains, freeze


def validation_figure():
    fig, ax = plt.subplots(figsize=(6.8, 2.0))
    for name in BOOSTERS:
        table = pd.read_csv(OUT / "tuning" / f"{name}.csv")
        rounds = pd.read_csv(OUT / "tuning" / f"{name}_rounds.csv")
        winner = rounds[
            (rounds.candidate == table.iloc[0].candidate) & (rounds["round"] >= 100)
        ]
        ax.plot(winner["round"], winner.val_MAE, label=name)
        selected = winner[winner["round"] == int(table.iloc[0].selected_rounds)].iloc[0]
        ax.scatter(selected["round"], selected.val_MAE, s=12)
    ax.set(
        xlabel="Boosting rounds (detail from round 100)",
        ylabel="Validation MAE (veh/h)",
    )
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(OUT / "boosting_validation_late.png", dpi=180)
    plt.close(fig)


def report():
    bench, efficiency, scaling, gains, freeze = load_inputs()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    validation_figure()
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="RBody",
            fontName="Helvetica",
            fontSize=9.4,
            leading=12.5,
            spaceAfter=7,
            textColor=colors.HexColor("#263447"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="RTitle",
            fontName="Helvetica-Bold",
            fontSize=21,
            leading=25,
            spaceAfter=7,
            textColor=colors.HexColor("#123448"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="RHead",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=7,
            spaceAfter=5,
            textColor=colors.HexColor("#087d86"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="RSmall",
            fontName="Helvetica",
            fontSize=7.8,
            leading=10.3,
            spaceAfter=5,
            textColor=colors.HexColor("#586779"),
        )
    )
    story = []
    paragraphs = []

    def p(text):
        paragraphs.append(text)
        story.append(Paragraph(text, styles["RBody"]))

    def h(text):
        paragraphs.append("## " + text)
        story.append(Paragraph(text, styles["RHead"]))

    def small(text):
        paragraphs.append(text)
        story.append(Paragraph(text, styles["RSmall"]))

    def table(rows, widths):
        paragraphs.append(pd.DataFrame(rows[1:], columns=rows[0]).to_markdown(index=False))
        value = Table(
            [
                [Paragraph(escape(str(cell)), styles["RSmall"]) for cell in row]
                for row in rows
            ],
            colWidths=widths,
            hAlign="LEFT",
        )
        value.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f4f4")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#087d86")),
                    ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#dce3e8")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.extend([value, Spacer(1, 7)])

    def footer(c, doc):
        c.setStrokeColor(colors.HexColor("#dce3e8"))
        c.line(40, 32, A4[0] - 40, 32)
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#586779"))
        c.drawString(40, 21, "Bhavesh | frozen Madrid experiment | 19 November 2026")
        c.drawRightString(A4[0] - 40, 21, str(doc.page))

    baseline = gains.iloc[0].baseline_selected_on_validation
    best_booster = bench.loc[BOOSTERS].test_MAE.idxmin()
    best_overall = bench.test_MAE.idxmin()
    best_error = float(efficiency.test_MAE.min())
    candidate = efficiency[
        efficiency.test_MAE <= best_error * 1.05
    ].single_predict_ms.idxmin()
    gain = gains.loc[best_booster]
    story.append(
        Paragraph("Traffic prediction: Bhavesh's contribution", styles["RTitle"])
    )
    small("BOOSTING VARIANTS, ANALYSIS C AND INTEGRATED INTERFACE | REAL-DATA RESULTS")
    h("1. Actual data and evaluation design")
    p(
        "The merged Madrid archive contains 300 sensors and 12,432 hourly timestamps from June 2022 to October 2023. "
        "This common experiment uses 60 evenly selected sensor IDs and forecasts one hour ahead (one step), in veh/h. "
        "A 15-minute target is unavailable and is rejected rather than interpolated. The original city dictionary confirms the rate unit [1,2]."
    )
    p(
        "All models share 29 features: hourly lags, 4/12-reading rolling statistics, seasonal lags, target calendar, "
        "training-fitted sensor/neighbour summaries and static roads. Unverified legacy weather is excluded for every model. "
        "The chronological split and embargo give 486,400 usable training-period rows, sampled to 200,000 for final fitting; "
        "validation has 106,160 rows and test 105,765. Test origins span 15 August to 31 October 2023."
    )
    h("2. Boosting search and measured results")
    p(
        "AdaBoost.R2 reweights difficult samples and predicts by a weighted median. Classic GB adds residual-fitting trees "
        "under squared error; HistGB bins features for split search. Ten seed-42 configurations per variant use 30,000 "
        "training rows and all validation rows. Every stage is scored; validation MAE selects both settings and rounds. "
        "HistGB random internal early stopping is disabled. The configuration freeze precedes all test scoring."
    )
    rows = [
        [
            "Model",
            "Validation-selected settings",
            "Tune val MAE",
            "Test MAE",
            "Test RMSE",
        ]
    ]
    for name in BOOSTERS:
        params = json.loads((OUT / "best_params" / f"{name}.json").read_text())
        tune = pd.read_csv(OUT / "tuning" / f"{name}.csv").iloc[0]
        if name == "AdaBoost":
            setting = f"{params['n_estimators']} rounds; depth {params['base_depth']}; rate {params['learning_rate']}; {params['loss']} loss"
        elif name == "GradientBoosting":
            setting = f"{params['n_estimators']} rounds; depth {params['max_depth']}; rate {params['learning_rate']}; leaf min {params['min_samples_leaf']}; sample {params['subsample']}"
        else:
            setting = f"{params['max_iter']} rounds; {params['max_leaf_nodes']} leaves; rate {params['learning_rate']}; leaf min {params['min_samples_leaf']}; L2={params['l2_regularization']}"
        rows.append(
            [
                name,
                setting,
                f"{tune.val_MAE:.2f}",
                f"{bench.loc[name,'test_MAE']:.2f}",
                f"{bench.loc[name,'test_RMSE']:.2f}",
            ]
        )
    table(rows, [105, 190, 70, 70, 80])
    small(
        "Tuning scores use the 30,000-row fit; test scores use a fresh 200,000-row fit. All errors are veh/h. Supporting reference families were also validation-tuned; the appendix contains all 14 models."
    )
    story.append(
        Image(str(OUT / "boosting_validation_late.png"), width=500, height=147)
    )
    small(
        "Late-round detail; full train/validation curves for every candidate are retained. AdaBoost (200) and HistGB (600) hit their search caps: these are bounded-search selections, not proven global optima."
    )
    p(
        f"Among these boosters, {best_booster} has the lowest frozen test MAE ({bench.loc[best_booster,'test_MAE']:.2f}), "
        f"a {100*gain.relative_MAE_improvement:.1f}% reduction over {baseline} ({bench.loc[baseline,'test_MAE']:.2f}), the baseline selected on validation. "
        f"It improves on {int(gain.days_improved)}/{int(gain.n_days)} target days and {int(gain.sensors_improved)}/{int(gain.n_sensors)} sensors. "
        "These are observations within one held-out period; no test-based retuning was performed."
    )

    story.append(PageBreak())
    story.append(Paragraph("Analysis C and the forecast interface", styles["RTitle"]))
    h("3. Error, latency, training cost and storage")
    p(
        "On an Intel Core i5-1035G1 with a two-thread budget, training time is one controlled frozen fit per model. "
        "Inference uses 30 warm single-row and batch timings on identical seed-42 validation cases. Size is uncompressed "
        "joblib storage, not peak RAM. Feature construction, ingestion, cold loading and concurrency are excluded."
    )
    chosen = list(
        dict.fromkeys(
            [
                best_overall,
                candidate,
                best_booster,
                "AdaBoost",
                "GradientBoosting",
                "RandomForest",
            ]
        )
    )
    rows = [["Model", "Test MAE", "Fit s", "Single ms", "p95 ms", "Size MiB"]]
    for name in chosen:
        r = efficiency.loc[name]
        rows.append(
            [
                name,
                f"{r.test_MAE:.2f}",
                f"{r.train_time_s:.2f}",
                f"{r.single_predict_ms:.3f}",
                f"{r.single_p95_ms:.3f}",
                f"{r.model_size_kb/1024:.3f}",
            ]
        )
    table(rows, [125, 66, 73, 73, 73, 105])
    r = efficiency.loc[candidate]
    p(
        f"{best_overall} gives the lowest aggregate test MAE ({best_error:.2f}). Within an explicit 5% error tolerance, "
        f"{candidate} is the fastest measured single-row predictor ({r.single_predict_ms:.3f} ms median; {r.single_p95_ms:.3f} ms p95). "
        f"HistGB fits {efficiency.loc['GradientBoosting','train_time_s']/efficiency.loc['HistGradientBoosting','train_time_s']:.1f} times faster than classic GB, "
        f"yet its single-row prediction is {efficiency.loc['HistGradientBoosting','single_predict_ms']/efficiency.loc['GradientBoosting','single_predict_ms']:.1f} times slower here. "
        f"Classic GB's 1,000-row median is {efficiency.loc['GradientBoosting','predict_ms_per_1000']:.2f} ms versus "
        f"LightGBM's {efficiency.loc['LightGBM','predict_ms_per_1000']:.2f} ms: batching changes the trade-off. "
        "These estimator timings establish no end-to-end SLA."
    )
    g = scaling[scaling.model == "HistGradientBoosting"].sort_values("n_train")
    first, last = g.iloc[0], g.iloc[-1]
    p(
        f"Eight model families use nested 12,500/25,000/50,000/100,000/200,000-row samples and fixed validation/settings. "
        f"HistGB validation MAE changes from {first.val_MAE:.2f} to {last.val_MAE:.2f}, while fit time changes from "
        f"{first.train_time_s:.2f} to {last.train_time_s:.2f} s. Full-size timings/validation scores are reused from the benchmark; "
        "smaller sizes get new fits. This measures row count, not reduced history. Single fit times have no confidence interval."
    )
    if (OUT / "timing_recheck.json").exists():
        small("Timing audit: one 25,000-row GB fit included Windows sleep. Only that timing was repeated with identical rows/settings; original measurements and the system event are retained. No test scores changed.")
    h("4. Integrated UI and verification")
    p(
        "Streamlit uses the shared features and hash-checked models. Sensor, target date/time and model controls drive forecasts, "
        "recent history, multi-model day charts, daily MAE/RMSE, sensor map and provenance-rich CSVs. Models load lazily; "
        "gaps stay missing. Each point is a rolling historical forecast from its own origin, not a whole day predicted at midnight."
    )
    p(
        "Regression tests cover leakage, horizon/weather safeguards, cached AdaBoost medians, feature identity, time alignment, "
        "Pareto ties and UI controls. Browser exports are checked against direct inference. Screenshots, a two-minute demo and full "
        "cost/scaling figures accompany the evidence. Existing experiment call sites remain compatible."
    )
    h("5. Limits and interpretation")
    p(
        "Evidence covers 60 seen sensors and one temporal holdout, with a smaller tuning sample and finite round caps. "
        "Zero-run screening changes 27 training readings and no validation/test readings; it is not an online detector. "
        "Raw aggregation, road-metadata vintage and timezone remain unverified. Timing is machine-specific and not OS-isolated. "
        "All-300-sensor, 15-minute and live-feed claims are unsupported. Teammates retain analyses A/E."
    )
    small(
        "[1] Gomez & Ilarri, Enriched Traffic Datasets for Madrid, Mendeley Data V4, CC BY 4.0, DOI 10.17632/697ht4f65b.4. "
        "[2] Madrid City Council, Historical Traffic Data Dictionary, October 2023, section 3.3. "
        "[3] Related article: DOI 10.1109/OJITS.2025.3637305. Full URLs, run hashes and environment are in the accompanying protocol/evidence."
    )
    document = PDF_DIR / "Bhavesh_RealData_Report.pdf"
    SimpleDocTemplate(
        str(document),
        pagesize=A4,
        leftMargin=40,
        rightMargin=40,
        topMargin=34,
        bottomMargin=42,
        title="Bhavesh - Traffic Prediction Real-Data Results",
        author="Bhavesh",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    import re

    markdown = "# Bhavesh - real-data report sections\n\n" + "\n\n".join(
        re.sub(r"<[^>]+>", "", t) for t in paragraphs
    )
    markdown += (
        "\n\n## All frozen model scores\n\n"
        + bench[
            ["val_MAE", "test_MAE", "test_RMSE", "test_R2", "test_WAPE"]
        ].to_markdown()
        + "\n"
    )
    (ROOT / "docs/BHAVESH_REPORT_SECTIONS.md").write_text(markdown, encoding="utf-8")
    print(document)
    evidence(bench, efficiency, freeze)


def evidence(bench, efficiency, freeze):
    path = PDF_DIR / "Bhavesh_RealData_Evidence.pdf"
    c = canvas.Canvas(str(path), pagesize=landscape(A4))
    c.setAuthor("Bhavesh")
    c.setTitle("Traffic Prediction - Frozen Madrid Evidence")
    page = 0

    def start(title, portrait=False):
        nonlocal page
        page += 1
        size = A4 if portrait else landscape(A4)
        c.setPageSize(size)
        c.setFont("Helvetica-Bold", 17)
        c.setFillColor(colors.HexColor("#123448"))
        c.drawString(35, size[1] - 35, title)
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#586779"))
        c.drawString(
            35, 20, "Frozen hourly Madrid experiment | " + freeze["run_id"][:16]
        )
        c.drawRightString(size[0] - 35, 20, str(page))
        return size

    def figure(path, portrait=False):
        size = A4 if portrait else landscape(A4)
        image = ImageReader(str(path))
        w, h = image.getSize()
        scale = min((size[0] - 70) / w, (size[1] - 95) / h)
        c.drawImage(
            image,
            (size[0] - w * scale) / 2,
            40 + (size[1] - 95 - h * scale) / 2,
            width=w * scale,
            height=h * scale,
        )

    start("All 14 frozen models - same target, features and split")
    data = [
        ["Model", "Val MAE", "Test MAE", "RMSE", "R2", "WAPE %", "Fit seconds", "MiB"]
    ]
    for name, r in bench.sort_values("test_MAE").iterrows():
        data.append(
            [
                name,
                f"{r.val_MAE:.3f}",
                f"{r.test_MAE:.3f}",
                f"{r.test_RMSE:.3f}",
                f"{r.test_R2:.4f}",
                f"{100*r.test_WAPE:.3f}",
                f"{r.train_time_s:.3f}",
                f"{r.model_size_kb/1024:.3f}",
            ]
        )
    t = Table(data, colWidths=[155, 80, 80, 85, 60, 75, 110, 127], rowHeights=25)
    t.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f4f4")),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#f5f8fa")],
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    t.wrapOn(c, 780, 450)
    t.drawOn(c, 35, landscape(A4)[1] - 80 - 25 * len(data))
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#263447"))
    c.drawString(
        35,
        75,
        "MAE/RMSE: veh/h. WAPE is shown as percent. Model storage is uncompressed joblib, not RAM.",
    )
    c.drawString(
        35,
        59,
        "200,000 fit rows; 106,160 validation; 105,765 test. Shared unverified weather excluded. No test-based retuning.",
    )
    c.showPage()
    for title, name in [
        ("Boosting - all rounds, all validation rows", "boosting_rounds.png"),
        (
            "Boosting - late rounds; independent vertical scales",
            "boosting_rounds_tail.png",
        ),
        (
            "Frozen benchmark - daily gains over the validation-selected baseline",
            "frozen_daily_gains.png",
        ),
        (
            "Analysis C - four cost/error Pareto comparisons",
            "analysis_c_cost_vs_error.png",
        ),
        (
            "Analysis C - nested training-row scaling on validation",
            "analysis_c_scaling.png",
        ),
    ]:
        start(title)
        figure(OUT / name)
        c.showPage()
    for title, name in [
        ("Real-data forecast and recent history", "madrid_forecast_detail.png"),
        ("Three-model comparison, daily errors and CSV", "madrid_comparison.png"),
        ("Sensor map and training-period flow", "madrid_sensor_map.png"),
    ]:
        image = ROOT / "docs/screenshots" / name
        if not image.exists():
            raise FileNotFoundError(
                f"Capture the real UI before delivering the report: {image}"
            )
        start(title, portrait=True)
        figure(image, portrait=True)
        c.showPage()
    c.save()
    print(path)


if __name__ == "__main__":
    report()
