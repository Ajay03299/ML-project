const DATA = window.SHOWCASE_DATA;

const FIGURE_CATALOG = [
  ["figures", "eda_daily_profile.png", "Daily Traffic Profile", "EDA", "Ajay", "Average weekday vs. weekend diurnal curves showing morning/evening commuter peaks."],
  ["figures", "eda_heatmap_week_hour.png", "Day-of-Week × Hour Heatmap", "EDA", "Ajay", "Traffic intensity across 7 days and 24 hours highlighting weekday rush-hour bands."],
  ["figures", "eda_autocorrelation.png", "Temporal Autocorrelation", "EDA", "Ajay", "Strong short-term lag correlation alongside 24-hour and 168-hour (weekly) seasonality."],
  ["figures", "eda_corr_vs_distance.png", "Sensor Correlation vs. Distance", "EDA", "Ajay", "Pairwise flow correlation vs. geographic distance across the 60 Madrid sensors."],
  ["figures", "eda_sensors.png", "Sensor Flow Distributions", "EDA", "Ajay", "Cross-sensor volume heterogeneity across motorway, primary, secondary, and residential roads."],
  ["figures", "eda_trend.png", "Long-Term Volume Trend", "EDA", "Ajay", "Multi-month flow evolution from June 2022 to October 2023 including summer dip and holidays."],
  ["figures", "eda_missingness.png", "Missingness & Fault Screening", "EDA", "Ajay", "Data completeness profile showing 0.87% missing/faulty share after cleaning."],
  ["figures", "ajay_dt_depth_curve.png", "Decision Tree Depth Curve", "Decision Tree", "Ajay", "Train vs. validation MAE across tree depths showing bias–variance trade-off."],
  ["figures", "ajay_dt_pruning.png", "Cost-Complexity Pruning", "Decision Tree", "Ajay", "Validation error under post-pruning ccp_alpha regularization."],
  ["figures", "ajay_dt_importance.png", "Decision Tree Feature Importance", "Decision Tree", "Ajay", "Dominant split features for the tuned single Decision Tree."],
  ["figures", "ajay_dt_top_levels.png", "Top Decision Tree Splits", "Decision Tree", "Ajay", "Interpretable top levels of the fitted Decision Tree showing primary flow thresholds."],
  ["figures", "pragun_rf_trees.png", "Random Forest Trees vs. Val MAE", "Random Forest & PCA", "Pragun", "Validation MAE drops steeply up to 100 trees (~41.0 veh/h) and flattens past 100–200 trees."],
  ["figures", "pragun_rf_importance.png", "RF Impurity vs. Permutation Importance", "Random Forest & PCA", "Pragun", "Permutation importance confirms flow_lag0 (~265 veh/h) dominates over flow_lastweek (~80)."],
  ["figures", "pragun_pca_scree.png", "PCA Scree & Cumulative Variance", "Random Forest & PCA", "Pragun", "14 / 18 / 24 components (of 32) capture 90% / 95% / 99% of standardized feature variance."],
  ["figures", "pragun_pca_loadings.png", "PCA Top Component Loadings", "Random Forest & PCA", "Pragun", "PC1 tracks overall volume, PC2 diurnal phase, PC3 day/night ramp, and PC4 weekday/weekend regime."],
  ["figures", "pragun_pca_projection.png", "PCA 2D Traffic Projection", "Random Forest & PCA", "Pragun", "Low-dimensional projection of Madrid traffic states colored by target flow intensity."],
  ["figures", "pragun_pca_vs_k.png", "PCA Downstream Error vs. Components", "Random Forest & PCA", "Pragun", "Even at 99% variance, PCA hurts both Ridge (+18.0%) and Random Forest (+17.8%)."],
  ["figures", "anirudh_boosting_rounds.png", "XGBoost & LightGBM Boosting Rounds", "Boosting", "Anirudh", "Validation early-stopping convergence curves for tuned XGBoost and LightGBM."],
  ["bhavesh", "boosting_rounds.png", "Boosting Validation Curves (v2)", "Boosting", "Bhavesh", "Validation MAE progression across boosting iterations for AdaBoost, GB, and HistGB."],
  ["bhavesh", "boosting_rounds_tail.png", "Boosting Tail Convergence", "Boosting", "Bhavesh", "Zoomed late-stage boosting convergence verifying stability without overfitting."],
  ["bhavesh", "frozen_daily_gains.png", "Daily MAE Gains Over Baseline", "Boosting", "Bhavesh", "Consistent day-by-day test MAE improvements of boosted models over the Ridge baseline."],
  ["figures", "analysis_a_ablation_heatmap.png", "Analysis A: Feature Group Ablation", "Analyses", "Pragun", "Test MAE % change when removing each feature group across 6 tuned models (100k train rows)."],
  ["bhavesh", "analysis_c_cost_vs_error.png", "Analysis C: Compute Cost vs. Error", "Analyses", "Bhavesh", "Pareto frontier of test MAE against training time, inference latency, and model footprint."],
  ["bhavesh", "analysis_c_scaling.png", "Analysis C: Training Data Scaling", "Analyses", "Bhavesh", "Test MAE and training cost scaling as training sample size increases."],
];

const state = {
  leaderboard: DATA.leaderboard,
  familyFilter: "All",
  sortKey: "test_MAE",
  sortAsc: true,
  selectedModel: null,
  selectedSensorId: DATA.sensors[0]?.id,
  galleryFilter: "All",
};

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const fmtSize = (kb) => {
  if (kb === null || kb === undefined) return "—";
  if (kb >= 1024 * 1024) return `${(kb / (1024 * 1024)).toFixed(2)} GB`;
  if (kb >= 1024) return `${(kb / 1024).toFixed(1)} MB`;
  return `${Number(kb).toFixed(1)} KB`;
};

/* ---------- Navigation ---------- */
function switchTab(tabId) {
  document.querySelectorAll(".nav-btn").forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.id === `tab-${tabId}`));
}

document.getElementById("main-nav").addEventListener("click", (e) => {
  const btn = e.target.closest(".nav-btn");
  if (btn) switchTab(btn.dataset.tab);
});

document.addEventListener("click", (e) => {
  const jump = e.target.closest("[data-jump]");
  if (jump) switchTab(jump.dataset.jump);
  const img = e.target.closest(".clickable-img");
  if (img) openLightbox(img.src, img.dataset.title || img.alt, img.dataset.caption || "");
});

/* ---------- Lightbox ---------- */
const lightbox = document.getElementById("lightbox");
function openLightbox(src, title, caption) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-title").textContent = title || "";
  document.getElementById("lightbox-caption").textContent = caption || "";
  lightbox.classList.add("open");
  lightbox.setAttribute("aria-hidden", "false");
}
function closeLightbox() {
  lightbox.classList.remove("open");
  lightbox.setAttribute("aria-hidden", "true");
}
document.getElementById("lightbox-close").addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLightbox(); });

/* ---------- Overview ---------- */
function renderOverview() {
  const lb = state.leaderboard;
  const best = lb[0];
  const ridge = lb.find((r) => r.model === "Ridge");
  const kpis = [
    { label: "Best Test MAE", val: `${fmt(best.test_MAE, 2)} veh/h`, sub: `${best.model} (${best.owner})` },
    { label: "Gain vs. Best Baseline", val: `+${fmt(best.vs_ridge_pct, 1)}%`, sub: `vs. Ridge (${fmt(ridge?.test_MAE, 2)} veh/h)` },
    { label: "Top Test R² / WAPE", val: `${fmt(best.test_R2, 3)} / ${fmt(best.test_WAPE * 100, 1)}%`, sub: "60-minute ahead forecast" },
    { label: "Models Benchmarked", val: `${lb.length}`, sub: "5 baselines + 9 ML pipelines" },
    { label: "Madrid Panel Scale", val: "60 × 12,432", sub: "0.87% missing · 32 features" },
  ];
  document.getElementById("overview-kpis").innerHTML = kpis
    .map((k) => `<div class="kpi-card"><div class="kpi-label">${k.label}</div><div class="kpi-val">${k.val}</div><div class="kpi-sub">${k.sub}</div></div>`)
    .join("");

  renderHorizontalBarChart("overview-chart", lb);
}

function renderHorizontalBarChart(containerId, rows) {
  const maxVal = Math.max(...rows.map((r) => r.test_MAE || 0), 1);
  const rowH = 30;
  const padL = 170;
  const padR = 74;
  const width = 920;
  const height = rows.length * rowH + 26;
  const barW = width - padL - padR;

  const colors = {
    Boosting: "#0f766e",
    "Trees & RF": "#1d4ed8",
    PCA: "#7c3aed",
    Baseline: "#9ca3af",
  };

  const bars = rows
    .map((r, i) => {
      const y = 10 + i * rowH;
      const w = ((r.test_MAE || 0) / maxVal) * barW;
      const fill = colors[r.family] || "#4b5563";
      return `
        <g>
          <text x="${padL - 12}" y="${y + 16}" text-anchor="end" font-size="12" fill="#111827" font-weight="${i < 3 ? 600 : 400}">${r.model}</text>
          <rect x="${padL}" y="${y + 4}" width="${Math.max(w, 2)}" height="16" rx="4" fill="${fill}" opacity="0.9"></rect>
          <text x="${padL + w + 8}" y="${y + 16}" font-size="12" fill="#374151" font-family="JetBrains Mono, monospace">${fmt(r.test_MAE, 2)}</text>
        </g>`;
    })
    .join("");

  document.getElementById(containerId).innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" style="max-width:${width}px;display:block;">
      ${bars}
    </svg>`;
}

/* ---------- Leaderboard & Tuning ---------- */
function renderLeaderboard() {
  let rows = [...state.leaderboard];
  if (state.familyFilter !== "All") {
    rows = rows.filter((r) => r.family === state.familyFilter);
  }
  const key = state.sortKey;
  rows.sort((a, b) => {
    const va = a[key] ?? (state.sortAsc ? 1e12 : -1e12);
    const vb = b[key] ?? (state.sortAsc ? 1e12 : -1e12);
    if (typeof va === "string") return state.sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
    return state.sortAsc ? va - vb : vb - va;
  });

  const tbody = document.querySelector("#leaderboard-table tbody");
  tbody.innerHTML = rows
    .map((r) => {
      const vs = r.vs_ridge_pct;
      const vsCls = vs > 0.05 ? "badge-pos" : vs < -0.05 ? "badge-neg" : "badge-zero";
      const vsTxt = vs === null ? "—" : `${vs > 0 ? "+" : ""}${fmt(vs, 1)}%`;
      return `
        <tr data-model="${r.model}" class="${state.selectedModel === r.model ? "selected" : ""}">
          <td><strong>${r.model}</strong></td>
          <td><span class="pill">${r.family}</span></td>
          <td>${r.owner}</td>
          <td class="num"><strong>${fmt(r.test_MAE, 2)}</strong></td>
          <td class="num">${fmt(r.test_RMSE, 2)}</td>
          <td class="num">${fmt(r.test_R2, 3)}</td>
          <td class="num">${r.test_WAPE != null ? `${fmt(r.test_WAPE * 100, 1)}%` : "—"}</td>
          <td class="num">${fmt(r.val_MAE, 2)}</td>
          <td class="num ${vsCls}">${vsTxt}</td>
          <td class="num">${fmt(r.train_time_s, 2)}</td>
          <td class="num">${fmtSize(r.model_size_kb)}</td>
        </tr>`;
    })
    .join("");
}

document.getElementById("family-filters").addEventListener("click", (e) => {
  const btn = e.target.closest(".filter-btn");
  if (!btn) return;
  state.familyFilter = btn.dataset.family;
  document.querySelectorAll("#family-filters .filter-btn").forEach((b) => b.classList.toggle("active", b === btn));
  renderLeaderboard();
});

document.querySelector("#leaderboard-table thead").addEventListener("click", (e) => {
  const th = e.target.closest("th[data-sort]");
  if (!th) return;
  const k = th.dataset.sort;
  if (state.sortKey === k) state.sortAsc = !state.sortAsc;
  else { state.sortKey = k; state.sortAsc = k !== "test_R2" && k !== "vs_ridge_pct"; }
  renderLeaderboard();
});

document.querySelector("#leaderboard-table tbody").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-model]");
  if (!tr) return;
  state.selectedModel = tr.dataset.model;
  renderLeaderboard();
  renderTuningInspector(state.selectedModel);
});

function renderTuningInspector(model) {
  const box = document.getElementById("tuning-inspector");
  const item = state.leaderboard.find((r) => r.model === model);
  const rows = DATA.tuning[model] || [];
  box.style.display = "block";
  document.getElementById("tuning-title").textContent = `${model} — Hyperparameters & Tuning`;
  document.getElementById("tuning-owner").textContent = `Owner: ${item?.owner || "Team"}`;
  document.getElementById("tuning-params").textContent = item?.best_params
    ? JSON.stringify(item.best_params, null, 2)
    : "No separate best_params JSON (heuristic baseline).";

  const wrap = document.getElementById("tuning-table-wrap");
  if (!rows.length) {
    wrap.innerHTML = `<p class="muted">No tuning grid CSV recorded for ${model}.</p>`;
    return;
  }
  const cols = Object.keys(rows[0]);
  wrap.innerHTML = `
    <table class="data-table">
      <thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows
          .map(
            (r) =>
              `<tr>${cols
                .map((c) => `<td>${typeof r[c] === "number" ? fmt(r[c], Number.isInteger(r[c]) ? 0 : 3) : r[c] ?? "null"}</td>`)
                .join("")}</tr>`
          )
          .join("")}
      </tbody>
    </table>`;
}

/* ---------- Sensor Explorer ---------- */
function roadColor(hw = "") {
  if (hw.includes("motorway") || hw.includes("trunk")) return "#dc2626";
  if (hw.includes("primary")) return "#0f766e";
  if (hw.includes("secondary")) return "#2563eb";
  return "#6b7280";
}

function renderSensorMap() {
  const sensors = DATA.sensors.filter((s) => s.lat && s.lon);
  if (!sensors.length) return;
  const lons = sensors.map((s) => s.lon);
  const lats = sensors.map((s) => s.lat);
  const minLon = Math.min(...lons), maxLon = Math.max(...lons);
  const minLat = Math.min(...lats), maxLat = Math.max(...lats);
  const W = 320, H = 280, pad = 22;

  const circles = sensors
    .map((s) => {
      const x = pad + ((s.lon - minLon) / (maxLon - minLon || 1)) * (W - 2 * pad);
      const y = H - pad - ((s.lat - minLat) / (maxLat - minLat || 1)) * (H - 2 * pad);
      const isSel = s.id === state.selectedSensorId;
      return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${isSel ? 7 : 4.5}"
        fill="${roadColor(s.highway)}" stroke="${isSel ? "#111827" : "#ffffff"}" stroke-width="${isSel ? 2.2 : 1.2}"
        style="cursor:pointer" data-sid="${s.id}">
        <title>Sensor ${s.id} (${s.highway}, mean ${s.mean_flow} veh/h)</title>
      </circle>`;
    })
    .join("");

  const box = document.getElementById("sensor-map");
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;">${circles}</svg>`;
  box.querySelectorAll("circle[data-sid]").forEach((c) => {
    c.addEventListener("click", () => {
      state.selectedSensorId = Number(c.dataset.sid);
      document.getElementById("sensor-select").value = String(state.selectedSensorId);
      renderSensorMap();
      renderSensorSeries();
    });
  });
}

function initExplorer() {
  const sSel = document.getElementById("sensor-select");
  sSel.innerHTML = DATA.sensors
    .map((s) => `<option value="${s.id}">Sensor ${s.id} · ${s.highway || "road"} (${s.mean_flow} veh/h)</option>`)
    .join("");

  // Dates with full 168h lookback (indices >= 168 and < len-1)
  const validDates = [...new Set(DATA.timestamps.slice(168, -1).map((t) => t.slice(0, 10)))];
  const dSel = document.getElementById("date-select");
  dSel.innerHTML = validDates.map((d) => `<option value="${d}">${d}</option>`).join("");
  dSel.value = validDates[Math.floor(validDates.length / 2)] || validDates[0];

  const hSel = document.getElementById("hour-select");
  hSel.innerHTML = Array.from({ length: 24 }, (_, h) => `<option value="${h}">${String(h).padStart(2, "0")}:00</option>`).join("");
  hSel.value = "14";

  sSel.addEventListener("change", () => {
    state.selectedSensorId = Number(sSel.value);
    renderSensorMap();
    renderSensorSeries();
  });
  dSel.addEventListener("change", renderSensorSeries);
  hSel.addEventListener("change", renderSensorSeries);

  renderSensorMap();
  renderSensorSeries();
}

function renderSensorSeries() {
  const sid = Number(document.getElementById("sensor-select").value);
  const date = document.getElementById("date-select").value;
  const hour = String(document.getElementById("hour-select").value).padStart(2, "0");
  const sensor = DATA.sensors.find((s) => s.id === sid) || DATA.sensors[0];
  const targetPrefix = `${date} ${hour}:00`;
  let pos = DATA.timestamps.indexOf(targetPrefix);
  if (pos < 168) pos = 168;
  if (pos >= DATA.timestamps.length - 1) pos = DATA.timestamps.length - 2;

  const history = [];
  for (let t = pos - 23; t <= pos; t++) {
    history.push({ time: DATA.timestamps[t], label: DATA.timestamps[t].slice(11), flow: sensor.series[t] });
  }

  const actual = sensor.series[pos + 1];
  const lag0 = sensor.series[pos];
  const yesterday = sensor.series[pos + 1 - 24];
  const lastweek = sensor.series[pos + 1 - 168];
  const blend = lag0 != null && lastweek != null && yesterday != null
    ? 0.5 * lag0 + 0.35 * lastweek + 0.15 * yesterday
    : lag0;

  const forecasts = {
    "Persistence (t)": lag0,
    "Seasonal Naive (Yesterday)": yesterday,
    "Seasonal Naive (Last Week)": lastweek,
    "Temporal Anchor Blend": blend,
  };

  document.getElementById("map-selected-pill").textContent = `Sensor ${sensor.id}`;
  document.getElementById("sensor-meta-badges").innerHTML = [
    `Road: ${sensor.highway || "—"}`,
    `Lanes: ${sensor.lanes ?? "—"}`,
    `Speed Limit: ${sensor.maxspeed ?? "—"} km/h`,
    `Mean Flow: ${sensor.mean_flow} veh/h`,
    `Target: ${DATA.timestamps[pos + 1]}`,
  ]
    .map((t) => `<span class="pill">${t}</span>`)
    .join("");

  renderSeriesChart(history, actual, blend);

  const cards = [
    `<div class="forecast-card actual"><div class="f-label">Actual Target (t + 60m)</div><div class="f-val">${fmt(actual, 1)}</div><div class="f-err">veh/h at ${DATA.timestamps[pos + 1].slice(11)}</div></div>`,
    ...Object.entries(forecasts).map(([name, val]) => {
      const err = actual != null && val != null ? Math.abs(val - actual) : null;
      return `<div class="forecast-card"><div class="f-label">${name}</div><div class="f-val">${fmt(val, 1)}</div><div class="f-err">${err != null ? `Abs err: ${fmt(err, 1)} veh/h` : "—"}</div></div>`;
    }),
  ];
  document.getElementById("forecast-cards").innerHTML = cards.join("");
}

function renderSeriesChart(pts, actual, blend) {
  const W = 720, H = 220, padL = 48, padR = 24, padT = 16, padB = 28;
  const allVals = pts.map((p) => p.flow).filter((v) => v != null);
  if (actual != null) allVals.push(actual);
  if (blend != null) allVals.push(blend);
  const maxY = Math.max(...allVals, 10) * 1.1;
  const totalSlots = pts.length + 1;

  const xOf = (i) => padL + (i / (totalSlots - 1)) * (W - padL - padR);
  const yOf = (v) => padT + (1 - v / (maxY || 1)) * (H - padT - padB);

  const pathPts = pts
    .map((p, i) => (p.flow == null ? null : `${xOf(i).toFixed(1)},${yOf(p.flow).toFixed(1)}`))
    .filter(Boolean)
    .join(" ");

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const val = maxY * f;
    const y = yOf(val);
    return `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#f3f4f6" />
            <text x="${padL - 8}" y="${y + 4}" text-anchor="end" font-size="10" fill="#9ca3af">${Math.round(val)}</text>`;
  });

  const xTicks = pts
    .map((p, i) => (i % 4 === 0 ? `<text x="${xOf(i)}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#9ca3af">${p.label}</text>` : ""))
    .join("");

  const tx = xOf(pts.length);
  const actualDot = actual != null
    ? `<circle cx="${tx}" cy="${yOf(actual)}" r="5.5" fill="#0f766e" stroke="#ffffff" stroke-width="2"><title>Actual target: ${fmt(actual, 1)}</title></circle>`
    : "";
  const blendDot = blend != null
    ? `<circle cx="${tx}" cy="${yOf(blend)}" r="4.5" fill="#1d4ed8" stroke="#ffffff" stroke-width="1.5"><title>Temporal Anchor Blend: ${fmt(blend, 1)}</title></circle>`
    : "";

  document.getElementById("series-chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;">
      ${yTicks.join("")}
      <line x1="${xOf(pts.length - 1)}" y1="${padT}" x2="${xOf(pts.length - 1)}" y2="${H - padB}" stroke="#d1d5db" stroke-dasharray="4 3" />
      <polyline fill="none" stroke="#111827" stroke-width="2" points="${pathPts}" />
      ${actualDot}
      ${blendDot}
      ${xTicks}
      <text x="${tx}" y="${H - 8}" text-anchor="middle" font-size="10" fill="#0f766e" font-weight="600">+1h</text>
    </svg>`;
}

/* ---------- Ablation ---------- */
function heatColor(delta) {
  if (delta > 0) {
    const a = Math.min(0.82, 0.08 + (delta / 45) * 0.75);
    return `rgba(185, 28, 28, ${a.toFixed(2)})`;
  }
  const a = Math.min(0.45, 0.08 + (Math.abs(delta) / 5) * 0.35);
  return `rgba(15, 118, 110, ${a.toFixed(2)})`;
}

function renderAblation() {
  const { models, full_mae, matrix } = DATA.ablation;
  const header = `<tr><th>Variant</th><th>Feats</th>${models.map((m) => `<th>${m}<br/><span style="font-weight:400;color:#6b7280">full ${fmt(full_mae[m], 1)}</span></th>`).join("")}</tr>`;
  const body = matrix
    .map((row) => {
      const cells = models
        .map((m) => {
          const c = row.cells[m];
          if (!c) return "<td>—</td>";
          const bg = heatColor(c.delta_pct);
          const fg = c.delta_pct > 25 ? "#ffffff" : "#111827";
          return `<td style="background:${bg};color:${fg}" title="${m} (${row.variant}): test MAE ${c.test_MAE} veh/h (${c.delta_pct > 0 ? "+" : ""}${c.delta_pct}%)">${c.delta_pct > 0 ? "+" : ""}${fmt(c.delta_pct, 1)}%</td>`;
        })
        .join("");
      return `<tr><td class="variant-col">${row.variant}</td><td>${row.n_features}</td>${cells}</tr>`;
    })
    .join("");
  document.getElementById("ablation-matrix-wrap").innerHTML = `<table class="heat-table"><thead>${header}</thead><tbody>${body}</tbody></table>`;
}

/* ---------- PCA Chart ---------- */
function renderPcaChart() {
  const order = [2, 3, 5, 8, 10, 0.9, 15, 0.95, 20, 0.99];
  const labelOf = (k) => (k < 1 ? `${Math.round(k * 100)}%` : `${k}`);
  const ridgeMap = new Map(DATA.pca_ridge.map((r) => [Number(r.n_components), r.val_MAE]));
  const rfMap = new Map(DATA.pca_rf.map((r) => [Number(r.n_components), r.val_MAE]));

  const W = 520, H = 220, padL = 46, padR = 20, padT = 18, padB = 34;
  const maxY = 195;
  const xOf = (i) => padL + (i / (order.length - 1)) * (W - padL - padR);
  const yOf = (v) => padT + (1 - v / maxY) * (H - padT - padB);

  const ptsRidge = order.map((k, i) => `${xOf(i)},${yOf(ridgeMap.get(k) || 0)}`).join(" ");
  const ptsRf = order.map((k, i) => `${xOf(i)},${yOf(rfMap.get(k) || 0)}`).join(" ");

  document.getElementById("pca-chart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;">
      ${[0, 50, 100, 150].map((v) => `<line x1="${padL}" y1="${yOf(v)}" x2="${W - padR}" y2="${yOf(v)}" stroke="#f3f4f6" /><text x="${padL - 8}" y="${yOf(v) + 4}" text-anchor="end" font-size="10" fill="#9ca3af">${v}</text>`).join("")}
      <polyline fill="none" stroke="#7c3aed" stroke-width="2" points="${ptsRidge}" />
      <polyline fill="none" stroke="#0f766e" stroke-width="2" points="${ptsRf}" />
      ${order.map((k, i) => `<text x="${xOf(i)}" y="${H - 10}" text-anchor="middle" font-size="10" fill="#6b7280">${labelOf(k)}</text>`).join("")}
      <text x="${W - 140}" y="22" font-size="11" fill="#7c3aed" font-weight="600">● PCA_Ridge (val MAE)</text>
      <text x="${W - 140}" y="38" font-size="11" fill="#0f766e" font-weight="600">● PCA_RandomForest</text>
    </svg>`;
}

/* ---------- Gallery ---------- */
function renderGallery() {
  const items = state.galleryFilter === "All"
    ? FIGURE_CATALOG
    : FIGURE_CATALOG.filter((f) => f[3] === state.galleryFilter);

  document.getElementById("gallery-grid").innerHTML = items
    .map(([bucket, fname, title, _cat, owner, caption]) => {
      const url = `./assets/${bucket}/${fname}`;
      return `
        <div class="gallery-card">
          <img src="${url}" alt="${title}" class="clickable-img" data-title="${title}" data-caption="${caption}" loading="lazy" />
          <div class="gallery-meta">
            <div class="gallery-title">${title}</div>
            <span class="pill">${owner}</span>
          </div>
          <p class="gallery-cap">${caption}</p>
        </div>`;
    })
    .join("");
}

document.getElementById("gallery-filters").addEventListener("click", (e) => {
  const btn = e.target.closest(".filter-btn");
  if (!btn) return;
  state.galleryFilter = btn.dataset.cat;
  document.querySelectorAll("#gallery-filters .filter-btn").forEach((b) => b.classList.toggle("active", b === btn));
  renderGallery();
});

/* ---------- Boot ---------- */
renderOverview();
renderLeaderboard();
initExplorer();
renderAblation();
renderPcaChart();
renderGallery();
