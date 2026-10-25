"use strict";
// driftwatch dashboard: plain DOM and SVG, no build step, no external resources.
// Every value from the API is inserted with textContent, never as HTML.

const SVG = "http://www.w3.org/2000/svg";
const FEATURES = [
  "age", "monthly_income", "loan_amount", "debt_to_income", "credit_utilization", "bureau_score",
  "credit_history_months", "loan_term_months", "num_open_accounts", "delinquencies_2y", "inquiries_6m",
  "employment_type", "home_ownership", "loan_purpose", "channel", "region",
];
const SEVERITY_RANK = { critical: 4, high: 3, medium: 2 };
const SEVERITY_COLOR = { critical: "var(--critical)", high: "var(--serious)", medium: "var(--warning)" };
const tooltip = document.getElementById("tooltip");

function store(key, value) {
  try {
    if (value === undefined) return localStorage.getItem(key);
    localStorage.setItem(key, value);
  } catch (_) { return null; }
  return value;
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  const key = document.getElementById("api-key").value;
  if (key) headers["X-API-Key"] = key;
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmt(value, digits = 3) {
  return value === null || value === undefined || Number.isNaN(value) ? "n/a" : Number(value).toFixed(digits);
}

function showTooltip(event, rows) {
  tooltip.replaceChildren();
  for (const row of rows) {
    const line = document.createElement("div");
    line.className = "row";
    if (row.color) {
      const key = document.createElement("span");
      key.className = "line";
      key.style.background = row.color;
      line.append(key);
    }
    const v = document.createElement("span");
    v.className = "v";
    v.textContent = row.value;
    const k = document.createElement("span");
    k.className = "k";
    k.textContent = row.label;
    line.append(v, k);
    tooltip.append(line);
  }
  tooltip.hidden = false;
  const x = Math.min(event.clientX + 14, window.innerWidth - tooltip.offsetWidth - 8);
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
}

function hideTooltip() { tooltip.hidden = true; }

function scales(width, height, margin, xs, yDomain) {
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const sx = (x) => margin.left + ((x - x0) / Math.max(x1 - x0, 1)) * (width - margin.left - margin.right);
  const sy = (y) => height - margin.bottom - ((y - yDomain[0]) / (yDomain[1] - yDomain[0])) * (height - margin.top - margin.bottom);
  return { sx, sy, x0, x1 };
}

function lineChart(container, series, options) {
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 240;
  const margin = { top: 12, right: 92, bottom: 26, left: 44 };
  const all = series.flatMap((s) => s.points);
  container.replaceChildren();
  if (all.length < 2) {
    container.textContent = "Waiting for data.";
    return;
  }
  const xs = all.map((p) => p.x);
  const { sx, sy, x0, x1 } = scales(width, height, margin, xs, options.yDomain);
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}` });
  for (const tick of options.yTicks) {
    svg.append(el("line", { class: "grid", x1: margin.left, x2: width - margin.right, y1: sy(tick), y2: sy(tick) }));
    svg.append(el("text", { x: margin.left - 6, y: sy(tick) + 4, "text-anchor": "end" }, options.yFormat(tick)));
  }
  svg.append(el("line", { class: "axis", x1: margin.left, x2: width - margin.right, y1: height - margin.bottom, y2: height - margin.bottom }));
  for (const t of [x0, Math.round((x0 + x1) / 2), x1]) {
    svg.append(el("text", { x: sx(t), y: height - 8, "text-anchor": "middle" }, `batch ${t}`));
  }
  if (options.refLine !== undefined && options.refLine !== null) {
    svg.append(el("line", { class: "crosshair", x1: margin.left, x2: width - margin.right, y1: sy(options.refLine), y2: sy(options.refLine) }));
  }
  for (const s of series) {
    if (s.points.length < 1) continue;
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("");
    svg.append(el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const last = s.points[s.points.length - 1];
    svg.append(el("circle", { cx: sx(last.x), cy: sy(last.y), r: 4, fill: s.color, stroke: "var(--surface-1)", "stroke-width": 2 }));
    svg.append(el("text", { class: "label-strong", x: sx(last.x) + 8, y: sy(last.y) + 4 }, `${s.name} ${options.yFormat(last.y)}`));
  }
  const cross = el("line", { class: "crosshair", y1: margin.top, y2: height - margin.bottom, visibility: "hidden" });
  svg.append(cross);
  const hit = el("rect", { x: margin.left, y: margin.top, width: width - margin.left - margin.right, height: height - margin.top - margin.bottom, fill: "transparent" });
  hit.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const px = event.clientX - box.left;
    const target = x0 + ((px - margin.left) / (width - margin.left - margin.right)) * (x1 - x0);
    const nearest = xs.reduce((a, b) => (Math.abs(b - target) < Math.abs(a - target) ? b : a));
    cross.setAttribute("x1", sx(nearest));
    cross.setAttribute("x2", sx(nearest));
    cross.setAttribute("visibility", "visible");
    const rows = [{ value: `batch ${nearest}`, label: "" }];
    for (const s of series) {
      const p = s.points.find((q) => q.x === nearest);
      rows.push({ value: p ? options.yFormat(p.y) : "n/a", label: s.name, color: s.color });
    }
    showTooltip(event, rows);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTooltip(); });
  svg.append(hit);
  container.append(svg);
}

function legend(container, series) {
  if (series.length < 2) return;
  const box = document.createElement("div");
  box.className = "legend";
  for (const s of series) {
    const item = document.createElement("span");
    const key = document.createElement("span");
    key.className = "key";
    key.style.background = s.color;
    item.append(key, document.createTextNode(s.name));
    box.append(item);
  }
  container.prepend(box);
}

function seqColor(value) {
  const steps = ["var(--seq-100)", "var(--seq-250)", "var(--seq-400)", "var(--seq-550)", "var(--seq-700)"];
  const scaled = Math.min(Math.max(value, 0) / 0.5, 0.9999);
  return steps[Math.floor(scaled * steps.length)];
}

function worstSeverity(alerts) {
  return alerts.reduce((best, a) => (SEVERITY_RANK[a.severity] > SEVERITY_RANK[best] ? a.severity : best), "medium");
}

function heatmap(container, timeline, alertsByBatch) {
  const rows = timeline.slice(-120);
  container.replaceChildren();
  if (!rows.length) { container.textContent = "Waiting for data."; return; }
  const left = 150;
  const width = container.clientWidth || 900;
  const cell = Math.max(3, Math.floor((width - left - 8) / rows.length));
  const rowH = 14;
  const height = (FEATURES.length + 1) * rowH + 24;
  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}` });
  svg.append(el("text", { x: left - 8, y: rowH - 3, "text-anchor": "end" }, "alerts"));
  FEATURES.forEach((name, i) => svg.append(el("text", { x: left - 8, y: (i + 2) * rowH - 3, "text-anchor": "end" }, name)));
  rows.forEach((b, j) => {
    const x = left + j * cell;
    const alerts = alertsByBatch.get(b.index) || [];
    if (alerts.length) {
      svg.append(el("rect", { x, y: 1, width: Math.max(cell - 1, 1), height: rowH - 2, rx: 2, fill: SEVERITY_COLOR[worstSeverity(alerts)] }));
    }
    FEATURES.forEach((name, i) => {
      const v = (b.psi || {})[name];
      if (v === undefined) return;
      svg.append(el("rect", { x, y: (i + 1) * rowH + 1, width: Math.max(cell - 1, 1), height: rowH - 2, fill: seqColor(v) }));
    });
  });
  svg.append(el("text", { x: left, y: height - 6 }, `batch ${rows[0].index}`));
  svg.append(el("text", { x: left + (rows.length - 1) * cell, y: height - 6, "text-anchor": "end" }, `batch ${rows[rows.length - 1].index}`));
  svg.addEventListener("pointermove", (event) => {
    const box = svg.getBoundingClientRect();
    const j = Math.floor((event.clientX - box.left - left) / cell);
    const i = Math.floor((event.clientY - box.top) / rowH) - 1;
    if (j < 0 || j >= rows.length || i < -1 || i >= FEATURES.length) { hideTooltip(); return; }
    const b = rows[j];
    if (i === -1) {
      const alerts = alertsByBatch.get(b.index) || [];
      showTooltip(event, [{ value: `batch ${b.index}`, label: b.timestamp || "" }, ...alerts.map((a) => ({ value: a.severity, label: `${a.category}: ${a.scope}` }))]);
      return;
    }
    const name = FEATURES[i];
    showTooltip(event, [
      { value: fmt((b.psi || {})[name]), label: `PSI, ${name}` },
      { value: fmt((b.drift_p || {})[name], 4), label: "test p-value" },
      { value: `batch ${b.index}`, label: b.timestamp || "" },
    ]);
  });
  svg.addEventListener("pointerleave", hideTooltip);
  container.append(svg);
  const scale = document.getElementById("heat-scale");
  scale.replaceChildren(document.createTextNode("PSI 0"), Object.assign(document.createElement("span"), { className: "ramp" }), document.createTextNode("0.5 or more"));
}

function table(container, header, rows) {
  const t = document.createElement("table");
  const head = t.createTHead().insertRow();
  for (const h of header) { const th = document.createElement("th"); th.textContent = h; head.append(th); }
  const body = t.createTBody();
  for (const r of rows) {
    const tr = body.insertRow();
    for (const c of r) {
      const td = tr.insertCell();
      if (c instanceof Node) td.append(c); else td.textContent = c;
      if (typeof c === "string" && /^-?[0-9.]+$/.test(c)) td.className = "num";
    }
  }
  container.replaceChildren(t);
}

function badge(text, color) {
  const span = document.createElement("span");
  span.className = "badge";
  const dot = document.createElement("span");
  dot.className = "dot";
  dot.style.background = color;
  span.append(dot, document.createTextNode(text));
  return span;
}

function renderTiles(state, timeline) {
  document.getElementById("t-champion").textContent = `v${state.champion_version}`;
  const r = state.rollout;
  document.getElementById("t-rollout").textContent = r && r.status === "running" ? `v${r.candidate_version} in ${r.step}, ${Math.round(r.traffic * 100)}% traffic` : r ? `v${r.candidate_version} ${r.status.replace("_", " ")}` : "none";
  document.getElementById("t-alerts").textContent = String(state.open_alerts);
  document.getElementById("t-batches").textContent = `${state.batches} batches, ${state.labelled_batches} labelled`;
  const last = timeline[timeline.length - 1];
  const lastReal = [...timeline].reverse().find((b) => b.realized_auc !== null && b.realized_auc !== undefined);
  document.getElementById("t-est").textContent = last ? fmt(last.estimated_auc) : "n/a";
  document.getElementById("t-real").textContent = lastReal ? fmt(lastReal.realized_auc) : "n/a";
}

function renderRollout(state) {
  const panel = document.getElementById("rollout-panel");
  const r = state.rollout;
  document.getElementById("abort").hidden = !(r && r.status === "running");
  if (!r) { panel.textContent = "No rollout has run yet."; return; }
  panel.replaceChildren();
  const head = document.createElement("p");
  head.textContent = `Candidate v${r.candidate_version}: ${r.status.replace("_", " ")}, step ${r.step}, ${r.batches} batches, ${r.labelled_batches} labelled, ${r.exposure} applications decided by it.`;
  panel.append(head);
  if (r.reason) { const reason = document.createElement("p"); reason.textContent = `Reason: ${r.reason}`; panel.append(reason); }
  const list = document.createElement("ul");
  for (const g of r.guardrails || []) {
    const li = document.createElement("li");
    li.append(badge(g.passed ? "pass" : "fail", g.passed ? "var(--good)" : "var(--critical)"), document.createTextNode(` ${g.name}: ${g.detail}`));
    list.append(li);
  }
  panel.append(list);
}

function renderRegistry(models) {
  const stageColor = { production: "var(--good)", rolled_back: "var(--critical)", rejected: "var(--serious)", shadow: "var(--warning)", canary: "var(--warning)" };
  table(document.getElementById("registry"), ["Version", "Stage", "Parent", "Challenger", "Gate", "Created"], [...models].reverse().map((m) => [
    `v${m.version}`,
    badge(m.stage.replace("_", " "), stageColor[m.stage] || "var(--text-muted)"),
    m.parent_version ? `v${m.parent_version}` : "",
    m.lineage.challenger || m.lineage.source || "",
    m.metrics.reason || (m.metrics.holdout_auc ? `holdout AUC ${fmt(m.metrics.holdout_auc)}` : ""),
    m.created_at,
  ]));
}

function renderAlerts(alerts) {
  const container = document.getElementById("alerts");
  if (!alerts.length) { container.textContent = "No alerts."; return; }
  table(container, ["Batch", "Severity", "Category", "Scope", "Message", ""], alerts.map((a) => {
    const ack = document.createElement("button");
    ack.type = "button";
    ack.textContent = a.status === "open" ? "Acknowledge" : "acknowledged";
    ack.disabled = a.status !== "open";
    ack.addEventListener("click", async () => { await api(`/v1/alerts/${a.id}/ack`, { method: "POST" }); refresh(); });
    return [String(a.batch_index), badge(a.severity, SEVERITY_COLOR[a.severity] || "var(--text-muted)"), a.category, a.scope, a.message, ack];
  }));
}

function renderAucTable(timeline) {
  table(document.getElementById("auc-table"), ["Batch", "Estimated AUC", "Realised AUC", "Approval rate"], timeline.slice(-30).reverse().map((b) => [
    String(b.index), fmt(b.estimated_auc), fmt(b.realized_auc), fmt(b.approval_rate),
  ]));
}

async function refresh() {
  try {
    const [state, timeline, alerts, models] = await Promise.all([
      api("/v1/rollout"), api("/v1/timeline?limit=400"), api("/v1/alerts?limit=200"), api("/v1/registry/models"),
    ]);
    const alertsByBatch = new Map();
    for (const a of alerts) alertsByBatch.set(a.batch_index, [...(alertsByBatch.get(a.batch_index) || []), a]);
    renderTiles(state, timeline);
    const pts = (key) => timeline.filter((b) => b[key] !== null && b[key] !== undefined).map((b) => ({ x: b.index, y: b[key] }));
    const aucSeries = [
      { name: "estimated", color: "var(--series-1)", points: pts("estimated_auc") },
      { name: "realised", color: "var(--series-2)", points: pts("realized_auc") },
    ];
    const aucBox = document.getElementById("auc-chart");
    lineChart(aucBox, aucSeries, { yDomain: [0.6, 0.9], yTicks: [0.6, 0.7, 0.8, 0.9], yFormat: (v) => fmt(v, 3), refLine: state.cbpe_lower_bound });
    legend(aucBox, aucSeries);
    lineChart(document.getElementById("approval-chart"), [{ name: "approved", color: "var(--series-1)", points: pts("approval_rate") }], {
      yDomain: [0.4, 1.0], yTicks: [0.4, 0.6, 0.8, 1.0], yFormat: (v) => `${Math.round(v * 100)}%`,
    });
    heatmap(document.getElementById("heatmap"), timeline, alertsByBatch);
    renderRollout(state);
    renderRegistry(models);
    renderAlerts(alerts);
    renderAucTable(timeline);
  } catch (error) {
    document.getElementById("t-rollout").textContent = `API error: ${error.message}`;
  }
}

document.getElementById("api-key").value = store("driftwatch-api-key") || "";
document.getElementById("api-key").addEventListener("change", (e) => store("driftwatch-api-key", e.target.value));
document.getElementById("theme").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
});
document.getElementById("toggle-table").addEventListener("click", (e) => {
  const box = document.getElementById("auc-table");
  box.hidden = !box.hidden;
  e.target.setAttribute("aria-expanded", String(!box.hidden));
  e.target.textContent = box.hidden ? "Show table" : "Hide table";
});
document.getElementById("abort").addEventListener("click", async () => {
  await api("/v1/rollout/abort", { method: "POST", body: JSON.stringify({ reason: "operator rollback from the dashboard" }) });
  refresh();
});
refresh();
setInterval(refresh, 5000);
