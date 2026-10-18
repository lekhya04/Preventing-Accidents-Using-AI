// SafeDrive AI dashboard
const $ = s => document.querySelector(s);
let S = null;                      // summary from /api/summary
const api = (p, body) => fetch(p, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {}).then(r => r.json());

// ------------------------------------------------------------------ navigation
function go(page) {
  document.querySelectorAll(".page").forEach(p => p.classList.toggle("on", p.id === "page-" + page));
  document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("on", a.dataset.page === page));
  window.scrollTo({ top: 0 });
  if (page === "routes") setTimeout(drawMap, 30);
  if (page === "live") setTimeout(drawSignal, 30);
  history.replaceState(null, "", "#" + page);
}
document.querySelectorAll("#nav a").forEach(a => a.onclick = () => go(a.dataset.page));

function toast(msg) {
  const d = document.createElement("div"); d.className = "toast"; d.innerHTML = msg; $("#toast").appendChild(d);
  setTimeout(() => d.remove(), 4800);
}

// ------------------------------------------------------------------ OVERVIEW
const MOD = {
  behaviour: { page: "behavior", name: "Driver Behaviour", color: "#22d3ee" },
  crash: { page: "crash", name: "Crash Severity", color: "#8b5cf6" },
  alcohol: { page: "alcohol", name: "Alcohol Detection", color: "#f472b6" },
  fusion: { page: "live", name: "AI Fusion & Routes", color: "#a3e635" },
};

function renderOverview() {
  const B = S.behavior, A = S.alcohol, C = S.crash;
  const best = C.standard_paper.comparison.find(r => r.model === C.standard_paper.deployed);
  const bm = B.models[B.deployed];
  $("#heroKpis").innerHTML = [
    { v: C.dataset.n_records.toLocaleString(), l: "real accident records (US-Accidents)", tag: "Real data" },
    { v: B.dataset.n_events + " events", l: `real labelled driving events · ${B.dataset.trips.length} trips`, tag: "Real data" },
    { v: pct(bm.aggressive_vs_safe_accuracy, 0), l: "aggressive-vs-safe detection (unseen trips)", tag: "Behaviour model" },
    { v: f3(best.roc_auc), l: "crash risk ROC-AUC (" + C.standard_paper.deployed + ")", tag: "Crash model" },
    { v: A.n, l: "alcohol sensor samples", tag: A.source.real ? "Real IEEE DataPort" : "Simulated – awaiting IEEE DataPort file", sim: !A.source.real },
    { v: "3 + 1", l: "ML modules + AI fusion layer", tag: "Integrated" },
  ].map(k => `<div class="kpi"><div class="v">${k.v}</div><div class="l">${k.l}</div><span class="tag ${k.sim ? "sim" : ""}">${k.tag}</span></div>`).join("");

  $("#arch").innerHTML = archSVG();
  $("#team").innerHTML = S.team.members.map((m, i) => `
    <div class="card member">
      <div class="av" style="background:${["linear-gradient(135deg,#22d3ee,#8b5cf6)", "linear-gradient(135deg,#f472b6,#fb923c)", "linear-gradient(135deg,#a3e635,#22d3ee)"][i % 3]}">${esc(m.name[0])}</div>
      <h3>${esc(m.name)}</h3><div class="role">${esc(m.role)}</div>
      <ul>${m.work.map(w => `<li>${esc(w)}</li>`).join("")}</ul>
      <div class="mods">${m.modules.map(k => `<span class="chip" onclick="go('${MOD[k].page}')">${MOD[k].name} →</span>`).join("")}</div>
    </div>`).join("");
  $("#behKicker").textContent = memberFor("behaviour") + " · Sub-area 1";
  $("#crKicker").textContent = memberFor("crash") + " · Sub-area 2";
  $("#alcKicker").textContent = memberFor("alcohol") + " · Sub-area 3";
  document.querySelectorAll("#page-live .kicker, #page-routes .kicker").forEach(k => k.textContent = memberFor("fusion") + " · AI Integration");
  $("#sideFoot").innerHTML = `${esc(S.team.institution)}<br><span>${esc(S.team.course)}</span>`;
}
const memberFor = mod => (S.team.members.find(m => m.modules.includes(mod)) || { name: "Team" }).name;

function archSVG() {
  const box = (x, y, w, h, t, sub, c, page) => `<g style="cursor:pointer" onclick="go('${page}')"><rect x="${x}" y="${y}" width="${w}" height="${h}" rx="14" fill="rgba(2,6,15,.6)" stroke="${c}" stroke-opacity=".6"/>
    <text x="${x + 16}" y="${y + 26}" fill="#e8eefc" font-size="14" font-weight="600" font-family="Space Grotesk">${t}</text>
    ${sub.map((s, i) => `<text x="${x + 16}" y="${y + 46 + i * 16}" fill="#7c8aa6" font-size="11.5">${s}</text>`).join("")}</g>`;
  const src = (y, t, s) => `<g><rect x="10" y="${y}" width="190" height="56" rx="12" fill="rgba(148,163,184,.06)" stroke="rgba(148,163,184,.2)"/><text x="24" y="${y + 24}" fill="#e8eefc" font-size="12.5" font-weight="600">${t}</text><text x="24" y="${y + 42}" fill="#7c8aa6" font-size="11">${s}</text></g>`;
  const ln = (x1, y1, x2, y2, c) => `<path class="flow" d="M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}" stroke="${c}" stroke-width="2" fill="none" stroke-opacity=".8"/>`;
  return `<svg viewBox="0 0 1100 330">
    ${src(18, "Smartphone IMU", "accel · gyro · 50 Hz")}${src(92, "In-cabin sensors", "MQ-3 · thermal · camera")}${src(166, "Accident history", "314k records + STATS19")}${src(240, "Live weather / GPS", "Open-Meteo API · location")}
    ${box(270, 14, 250, 76, "Driver Behaviour", [memberFor("behaviour") + " · SVM / RF + FFT", "aggressive-event detection"], "#22d3ee", "behavior")}
    ${box(270, 120, 250, 76, "Alcohol Detection", [memberFor("alcohol") + " · sensor classifier", "impairment probability"], "#f472b6", "alcohol")}
    ${box(270, 226, 250, 76, "Crash Severity", [memberFor("crash") + " · HGB · CCP · MT-FTT", "context risk + severity"], "#8b5cf6", "crash")}
    ${box(600, 100, 230, 112, "AI Fusion Engine", [memberFor("fusion"), "noisy-OR risk fusion", "Low / Medium / High"], "#a3e635", "live")}
    ${box(900, 30, 190, 70, "Driver alert", ["audio-visual warning"], "#fbbf24", "live")}
    ${box(900, 122, 190, 70, "Emergency contact", ["SMS + call (simulated)"], "#f43f5e", "live")}
    ${box(900, 214, 190, 70, "Safer route", ["hotspot-aware routing"], "#22d3ee", "routes")}
    ${ln(200, 46, 270, 52, "#22d3ee")}${ln(200, 120, 270, 158, "#f472b6")}${ln(200, 194, 270, 264, "#8b5cf6")}${ln(200, 268, 270, 264, "#8b5cf6")}
    ${ln(520, 52, 600, 140, "#22d3ee")}${ln(520, 158, 600, 156, "#f472b6")}${ln(520, 264, 600, 172, "#8b5cf6")}
    ${ln(830, 140, 900, 65, "#a3e635")}${ln(830, 156, 900, 157, "#a3e635")}${ln(830, 172, 900, 249, "#a3e635")}
  </svg>`;
}

// ------------------------------------------------------------------ BEHAVIOUR
function renderBehavior() {
  const B = S.behavior, names = Object.keys(B.models), dep = B.models[B.deployed];
  const labs = B.labels;
  $("#behBody").innerHTML = `
    <div class="stat-row">
      <div class="stat"><div class="v">${B.dataset.n_events}</div><div class="l">labelled events · ${B.dataset.trips.length} real trips</div></div>
      <div class="stat"><div class="v">${B.dataset.n_windows.toLocaleString()}</div><div class="l">2-second signal windows</div></div>
      <div class="stat"><div class="v">${pct(dep.aggressive_vs_safe_accuracy)}</div><div class="l">aggressive vs safe · unseen trips</div></div>
      <div class="stat"><div class="v">${pct(dep.event.accuracy)}</div><div class="l">event-type accuracy, 8 classes (${esc(B.deployed)})</div></div>
    </div>
    <div class="card"><div class="ch">Pipeline <span class="badge real">Real dataset</span></div>
      <div class="pipeline">
        <div class="st"><b>1 · Sensors</b><small>${esc(B.dataset.sensors)}</small></div>
        <div class="st"><b>2 · Windowing</b><small>${esc(B.dataset.window)}; events labelled from synchronised video</small></div>
        <div class="st"><b>3 · Features (${B.dataset.n_features})</b><small>mean, std, min, max, range, RMS, jerk + FFT band energies</small><span class="src">FFT branch ← Review-1 Paper 2</span></div>
        <div class="st"><b>4 · Classifier</b><small>Gaussian RBF-SVM vs Random Forest</small><span class="src">RBF-SVM ← Review-1 Paper 1</span></div>
        <div class="st"><b>5 · Evaluation</b><small>leave-one-trip-out CV; window + event level</small></div>
      </div>
      <div class="note info">Dataset: ${esc(B.dataset.name)} – the public benchmark also used by Review-1 Paper 2. 2011 Honda Civic, phone fixed on the windscreen, 2 drivers.</div>
    </div>
    <div class="grid2" style="margin-top:18px">
      <div class="card"><div class="ch">Model comparison (leave-one-trip-out)</div>
        <table><tr><th>Model</th><th class="n">Window acc</th><th class="n">Window F1</th><th class="n">Event acc</th><th class="n">Aggr-vs-safe</th></tr>
        ${names.map(n => { const m = B.models[n]; return `<tr class="${n === B.deployed ? "best" : ""}"><td>${esc(n)}${n === B.deployed ? " ★" : ""}</td><td class="n">${pct(m.window.accuracy)}</td><td class="n">${pct(m.window.macro_f1)}</td><td class="n">${pct(m.event.accuracy)}</td><td class="n">${pct(m.aggressive_vs_safe_accuracy)}</td></tr>`; }).join("")}</table>
        <div class="ch" style="margin-top:18px">Most informative features</div>${hbar(B.top_features.slice(0, 10).map(f => ({ label: f.feature, value: f.importance })), { fmt: v => (v * 100).toFixed(1) + "%", labelW: 170 })}
      </div>
      <div class="card"><div class="ch">Confusion matrix – events (${esc(B.deployed)})</div>${confusion(dep.event.confusion, labs)}
        <div class="note">Only 69 real events exist, so per-class numbers are noisy; the deployed model is judged mainly on aggressive-vs-safe detection on trips it never saw.</div></div>
    </div>
    <div class="card" style="margin-top:18px"><div class="ch">Trip replay – model output vs ground truth <select id="behTrip">${B.dataset.trips.map(t => `<option>${t}</option>`).join("")}</select></div><div id="behTimeline"></div></div>`;
  $("#behTrip").onchange = e => timeline(+e.target.value);
  timeline(B.dataset.trips[0]);
}

const EVC = { "Aggressive acceleration": "#fb923c", "Aggressive braking": "#f43f5e", "Aggressive left turn": "#8b5cf6", "Aggressive right turn": "#a78bfa", "Aggressive left lane change": "#22d3ee", "Aggressive right lane change": "#67e8f9", "Non-aggressive": "#34d399", "Normal cruising": "rgba(148,163,184,.25)" };
async function timeline(trip) {
  const R = await api("/api/replay?trip=" + trip), w = 1100, h = 150, X = t => 10 + (w - 20) * t / R.duration;
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%"><text x="10" y="18" fill="#7c8aa6" font-size="11">Ground truth</text><text x="10" y="78" fill="#7c8aa6" font-size="11">Model (deployed)</text>`;
  R.events.forEach(e => s += `<rect x="${X(e.start)}" y="26" width="${Math.max(3, X(e.end) - X(e.start))}" height="30" rx="3" fill="${EVC[e.label]}"><title>${e.label} ${e.start}–${e.end}s</title></rect>`);
  R.stream.forEach(p => { if (p.pred !== "Normal cruising") s += `<rect x="${X(p.t) - 1}" y="86" width="3" height="30" fill="${EVC[p.pred]}" opacity="${0.35 + 0.65 * p.conf}"><title>${p.t}s ${p.pred} (${Math.round(p.conf * 100)}%)</title></rect>`; });
  for (let t = 0; t <= R.duration; t += 60) s += `<text x="${X(t)}" y="138" fill="#7c8aa6" font-size="10" text-anchor="middle">${Math.round(t / 60)}m</text>`;
  s += "</svg>";
  const leg = Object.entries(EVC).filter(([k]) => k !== "Normal cruising").map(([k, c]) => `<span style="margin-right:14px;font-size:12px"><i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${c};margin-right:5px"></i>${k}</span>`).join("");
  $("#behTimeline").innerHTML = s + `<div style="margin-top:6px">${leg}</div>`;
}

// ------------------------------------------------------------------ ALCOHOL
const AF = ["alcohol_ml_l", "cabin_temp_c", "face_temp_min_c", "face_temp_max_c", "pupil_ratio"];
function renderAlcohol() {
  const A = S.alcohol, names = Object.keys(A.models);
  const real = A.source.real;
  $("#alcBody").innerHTML = `
    ${real ? "" : `<div class="note"><b>Simulated data.</b> The IEEE DataPort file “Alcohol Detection in Drivers by Sensors and Computer Vision” (DOI 10.21227/mnja-3553) needs a login. This page runs on a stand-in with the <i>same 5 attributes and the same 217 / 173 class split</i>. Put the real CSV in <code>data/alcohol/</code> and run <code>python train_all.py</code> – everything here updates automatically.</div>`}
    <div class="stat-row">
      <div class="stat"><div class="v">${A.n}</div><div class="l">samples <span class="badge ${real ? "real" : "sim"}">${real ? "Real" : "Simulated"}</span></div></div>
      <div class="stat"><div class="v">${A.class_counts["No alcohol"]} / ${A.class_counts["Alcohol present"]}</div><div class="l">no-alcohol / alcohol</div></div>
      <div class="stat"><div class="v">${pct(A.models[A.deployed].accuracy)}</div><div class="l">${esc(A.deployed)} · ${esc(A.cv)}</div></div>
      <div class="stat"><div class="v">${f3(A.models[A.deployed].roc_auc)}</div><div class="l">ROC-AUC</div></div>
    </div>
    <div class="grid2">
      <div class="card"><div class="ch">Classifier comparison</div>
        <table><tr><th>Model</th><th class="n">Accuracy</th><th class="n">Macro F1</th><th class="n">ROC-AUC</th></tr>
        ${names.map(n => `<tr class="${n === A.deployed ? "best" : ""}"><td>${esc(n)}${n === A.deployed ? " ★" : ""}</td><td class="n">${pct(A.models[n].accuracy)}</td><td class="n">${pct(A.models[n].macro_f1)}</td><td class="n">${f3(A.models[n].roc_auc)}</td></tr>`).join("")}</table>
        <div class="ch" style="margin-top:18px">Sensor importance</div>${hbar(Object.entries(A.feature_importance).map(([k, v]) => ({ label: k, value: v, color: "#f472b6" })), { labelW: 190 })}
      </div>
      <div class="card"><div class="ch">Try it – sensor readings</div><div class="sliders" id="alcLab"></div><div id="alcGauge" style="text-align:center;margin-top:8px"></div></div>
    </div>`;
  buildAlcSliders("#alcLab", async vals => {
    const r = await api("/api/alcohol", vals);
    $("#alcGauge").innerHTML = gauge(r.p_alcohol, { size: 240, label: "P(alcohol)" }) + `<div class="level ${r.p_alcohol >= .5 ? "High" : "Low"}">${r.label}</div>`;
  });
}

function buildAlcSliders(sel, cb, preset) {
  const A = S.alcohol, st = A.feature_stats;
  const box = $(sel);
  box.innerHTML = AF.map(f => { const lo = st[f].min, hi = st[f].max, v = preset ? preset[f] : st[f].sober_mean, stp = (hi - lo) / 200;
    return `<div class="s"><span>${esc(A.nice[f])}</span><input type="range" data-f="${f}" min="${lo}" max="${hi}" step="${stp}" value="${v}"><span>${(+v).toFixed(f === "pupil_ratio" || f === "alcohol_ml_l" ? 3 : 1)}</span></div>`; }).join("");
  const read = () => { const o = {}; box.querySelectorAll("input").forEach(i => { o[i.dataset.f] = +i.value; i.nextElementSibling.textContent = (+i.value).toFixed(i.dataset.f === "pupil_ratio" || i.dataset.f === "alcohol_ml_l" ? 3 : 1); }); return o; };
  box.querySelectorAll("input").forEach(i => i.oninput = () => cb(read()));
  cb(read());
  return { set: p => { box.querySelectorAll("input").forEach(i => i.value = p[i.dataset.f]); cb(read()); }, read };
}

// ------------------------------------------------------------------ CRASH
const PAPER_T1 = { "Logistic Regression": [0.8156, 0.715, 0.9118], "Random Forest": [0.8979, 0.7966, 0.9339], "XGBoost": [0.8702, 0.7796, 0.9385], "Gradient Boosting": [0.8948, 0.7955, 0.9377], "AdaBoost": [0.8756, 0.7481, 0.8963], "Extra Trees": [0.8925, 0.7905, 0.9327], "LightGBM": [0.7741, 0.696, 0.9317] };
function renderCrash(tab = "predict") {
  document.querySelectorAll("#crTabs button").forEach(b => b.classList.toggle("on", b.dataset.t === tab));
  const C = S.crash, body = $("#crBody");
  if (tab === "std") {
    const comp = C.standard_paper.comparison;
    body.innerHTML = `<div class="grid2"><div class="card"><div class="ch">Our replication of the standard paper's Table I <span class="badge real">Real data</span></div>
      <table><tr><th>Classifier</th><th class="n">Accuracy</th><th class="n">Macro F1</th><th class="n">ROC-AUC</th><th class="n">Train s</th></tr>
      ${comp.map(r => `<tr class="${r.model === C.standard_paper.deployed ? "best" : ""}"><td>${esc(r.model)}${r.model === C.standard_paper.deployed ? " ★ deployed" : ""}</td><td class="n">${pct(r.accuracy)}</td><td class="n">${f3(r.macro_f1)}</td><td class="n">${f3(r.roc_auc)}</td><td class="n">${r.train_seconds}</td></tr>`).join("")}</table>
      ${C.standard_paper.grid_search ? `<div class="note info">Grid search on the deployed model (as in the paper): ${esc(JSON.stringify(C.standard_paper.grid_search.best_params))}</div>` : ""}
      <div class="note">XGBoost / LightGBM rows appear automatically when those packages are installed (<code>pip install xgboost lightgbm</code>). Hist Gradient Boosting is scikit-learn's LightGBM-style booster.</div></div>
      <div class="card"><div class="ch">Standard paper (their data) vs ours (US-Accidents)</div>
      ${gbar(comp.map(r => ({ label: r.model.replace(" Gradient Boosting", " GB").replace("Logistic Regression", "Log. Reg."), values: [r.macro_f1, r.roc_auc, PAPER_T1[r.model] ? PAPER_T1[r.model][1] : null, PAPER_T1[r.model] ? PAPER_T1[r.model][2] : null] })), ["Our macro-F1", "Our ROC-AUC", "Paper macro-F1", "Paper ROC-AUC"], { w: 600, h: 280 })}
      <div class="note info">Same protocol as the paper: 80/20 split, Low / Medium / High risk, accuracy + macro-F1 + ROC-AUC + time. Our numbers are lower because our 3 classes are strongly imbalanced (High = ${pct(C.class_distribution.High / Object.values(C.class_distribution).reduce((a, b) => a + b))}) and we use only conditions known <i>before</i> a crash (no post-crash fields).</div></div></div>
      <div class="grid2" style="margin-top:18px"><div class="card"><div class="ch">What drives risk (deployed model)</div>${hbar(Object.entries(C.standard_paper.feature_importance || {}).slice(0, 10).map(([k, v]) => ({ label: k, value: v, color: "#8b5cf6" })))}</div>
      <div class="card"><div class="ch">Confusion matrix (${esc(C.standard_paper.deployed)})</div>${confusion(comp.find(r => r.model === C.standard_paper.deployed).confusion, C.standard_paper.labels, { w: 400 })}</div></div>`;
  } else if (tab === "ccp") {
    const E = C.enhanced_ccp;
    body.innerHTML = `<div class="stat-row"><div class="stat"><div class="v">${f3(E.test.macro_f1)}</div><div class="l">Enhanced CCP test macro-F1</div></div><div class="stat"><div class="v">${f3(E.unpruned_tree_test.macro_f1)}</div><div class="l">unpruned tree (baseline)</div></div><div class="stat"><div class="v">${E.leaves}</div><div class="l">leaves after pruning</div></div><div class="stat"><div class="v">${E.selected_features.length}</div><div class="l">features kept by FSS</div></div></div>
      <div class="grid2"><div class="card"><div class="ch">Step 1 · Forward Stepwise Selection (wrapper)</div>${line([{ name: "validation macro-F1", points: E.fss_trace.map((r, i) => [i + 1, r.val_macro_f1]) }], { xfmt: v => E.fss_trace[v - 1] ? E.fss_trace[v - 1].feature : v, yfmt: v => v.toFixed(3), w: 560 })}</div>
      <div class="card"><div class="ch">Step 2 · Cost-complexity pruning path</div>${line([{ name: "validation macro-F1", points: E.path.map(p => [p.leaves, p.val_macro_f1]) }], { xl: "number of leaves", yfmt: v => v.toFixed(3), xfmt: v => v })}
      <div class="note info">R<sub>α</sub>(T) = R(T) + α|T|. Among subtrees within 0.005 macro-F1 of the best, the S-score S(t) = 0.5·a(t) + 0.5·c(t) (leaf accuracy + coverage) picks the final tree: α = ${E.alpha.toExponential(2)}, S = ${E.s_score.toFixed(3)}.</div></div></div>
      <div class="card" style="margin-top:18px"><div class="ch">Readable decision rules (top of the pruned tree)</div><pre class="rules">${esc(E.rules_preview).replace(/class: (\d)/g, (m, k) => "→ " + (S.crash.standard_paper.labels[+k] || k).toUpperCase() + " RISK")}</pre></div>`;
  } else if (tab === "mtftt") {
    const M = C.mt_ftt, tasks = Object.keys(M.tasks);
    body.innerHTML = `<div class="card"><div class="ch">Architecture (implemented from Paper 4 in NumPy, gradient-checked)</div><div class="pipeline">
        <div class="st"><b>Feature tokenizer</b><small>E<sub>i</sub> = b<sub>i</sub> + x<sub>i</sub>W<sub>i</sub> (num) · E<sub>i</sub> = b<sub>i</sub> + e<sub>i</sub><sup>T</sup>W<sub>i</sub> (cat) · d = ${M.config.d_token}</small></div>
        <div class="st"><b>[CLS] + Transformer</b><small>Pre-Norm, ${M.config.heads}-head self-attention + FFN, ${M.config.layers} layer</small></div>
        <div class="st"><b>3 task heads</b><small>LayerNorm → ReLU → Linear → Softmax</small></div>
        <div class="st"><b>Uncertainty loss</b><small>L = Σ<sub>k</sub> L<sub>k</sub>/σ<sub>k</sub><sup>2</sup> + log σ<sub>k</sub> · learned σ = ${M.sigma.map(v => v.toFixed(2)).join(" / ")}</small></div></div></div>
      <div class="grid2" style="margin-top:18px"><div class="card"><div class="ch">Multi-task model vs single-task boosting (test set)</div>
      <table><tr><th>Task</th><th class="n">MT-FTT acc</th><th class="n">MT-FTT F1</th><th class="n">Single-task acc</th><th class="n">Single-task F1</th></tr>
      ${tasks.map(t => `<tr><td>${esc(M.tasks[t].title)}<br><span class="muted" style="font-size:11px">${esc(M.tasks[t].classes.join(" / "))}</span></td><td class="n">${pct(M.test[t].accuracy)}</td><td class="n">${f3(M.test[t].macro_f1)}</td><td class="n">${pct(M.single_task_baseline[t].accuracy)}</td><td class="n">${f3(M.single_task_baseline[t].macro_f1)}</td></tr>`).join("")}</table>
      <div class="note info">One shared transformer predicts all three outcomes in a single pass (${M.train_seconds}s training on CPU). The single-task column trains one Hist-GB model per outcome.</div></div>
      <div class="card"><div class="ch">Training curve</div>${line([{ name: "train loss", points: M.history.map(h => [h.epoch, h.train_loss]) }, { name: "val loss", points: M.history.map(h => [h.epoch, h.val_loss]) }], { xl: "epoch", yfmt: v => v.toFixed(3) })}</div></div>
      <div class="card" style="margin-top:18px"><div class="ch">Feature attention ([CLS] → feature tokens)</div>${hbar(Object.entries(M.attention_importance).map(([k, v]) => ({ label: k, value: v, color: "#22d3ee" })), { labelW: 140, w: 700 })}</div>`;
  } else if (tab === "data") {
    const D = C.dataset, cd = C.class_distribution, tot = Object.values(cd).reduce((a, b) => a + b);
    body.innerHTML = `<div class="grid2"><div class="card"><div class="ch">Dataset in use <span class="badge real">Real</span></div><p><b>${esc(D.name)}</b></p><p class="muted">${esc(D.source)}</p>
      <table><tr><th>Split</th><th class="n">Rows</th></tr><tr><td>Sampled for training (stratified)</td><td class="n">${(C.split.train + C.split.val + C.split.test).toLocaleString()}</td></tr><tr><td>Train / validation / test</td><td class="n">${C.split.train.toLocaleString()} / ${C.split.val.toLocaleString()} / ${C.split.test.toLocaleString()}</td></tr></table>
      <div class="ch" style="margin-top:16px">Risk-level distribution</div>${hbar(Object.entries(cd).map(([k, v], i) => ({ label: k, value: v / tot, color: ["#34d399", "#fbbf24", "#f43f5e"][i] })), { labelW: 90 })}
      <div class="ch" style="margin-top:16px">Prediction targets</div><table>${Object.values(D.tasks).map(t => `<tr><td><b>${esc(t.title)}</b></td><td>${esc(t.classes.join(" / "))}</td><td class="muted">${esc(t.rule)}</td></tr>`).join("")}</table></div>
      <div class="card"><div class="ch">IEEE DataPort dataset (STATS19) ${S.crash_stats19 ? '<span class="badge real">Loaded</span>' : '<span class="badge sim">Waiting for file</span>'}</div>
      <p><b>Uncovering Influential Mechanisms of Traffic Accident Severity through Explainable Machine Learning</b><br><span class="muted">Chen Ma · IEEE DataPort 2026 · DOI 10.21227/79s6-qr16 · UK STATS19 2005–2014</span></p>
      ${S.crash_stats19 ? `<table><tr><th>Model</th><th class="n">Acc</th><th class="n">Macro F1</th></tr>${S.crash_stats19.standard_paper.comparison.map(r => `<tr><td>${esc(r.model)}</td><td class="n">${pct(r.accuracy)}</td><td class="n">${f3(r.macro_f1)}</td></tr>`).join("")}</table>` :
        `<div class="note">The file needs an IEEE DataPort login. The loader is ready: put <code>Accidents, Casualties, Vehicles.xlsx</code> (or the Accidents/Casualties CSVs) into <code>data/crash/stats19_ieee_dataport/</code> and run <code>python train_all.py</code>. All three model families are then trained on STATS19 too, with targets: accident severity (Slight / Serious / Fatal), casualty count, pedestrian involved.</div>`}
      <div class="ch" style="margin-top:14px">Input features used</div><div class="mods" style="display:flex;flex-wrap:wrap;gap:6px">${D.cat.concat(D.num).map(f => `<span class="chip">${esc(f)}</span>`).join("")}</div></div></div>`;
  } else {
    body.innerHTML = `<div class="grid2"><div class="card"><div class="ch">Road & weather conditions</div><div class="formgrid" id="crForm"></div><button class="btn primary" id="crGo">Predict crash risk</button><div class="note info" style="margin-top:12px">Inputs are only things known <i>before</i> a crash. Location defaults to Minneapolis (the city with the most records).</div></div>
      <div class="card" id="crOut"><div class="ch">Prediction</div><div class="muted">Set conditions and press Predict.</div></div></div>`;
    ctxForm("#crForm", "cr");
    $("#crGo").onclick = async () => {
      const r = await api("/api/crash", readCtx("cr"));
      const tasks = Object.values(r.mt_ftt);
      $("#crOut").innerHTML = `<div class="ch">Prediction <span class="muted">${esc(r.deployed_model)}</span></div>
        <div style="display:grid;grid-template-columns:220px 1fr;gap:16px;align-items:center"><div style="text-align:center">${gauge(r.risk_score, { size: 220, label: "context risk score" })}<div class="level ${r.risk_level}">${r.risk_level} risk</div></div>
        <div>${hbar(r.risk_labels.map((l, i) => ({ label: "P(" + l + ")", value: r.risk_proba[i], color: ["#34d399", "#fbbf24", "#f43f5e"][i] })), { max: 1, labelW: 110, w: 420 })}
        <div class="muted" style="font-size:12px;margin-top:4px">Enhanced CCP tree: ${r.ccp_proba.map((p, i) => r.risk_labels[i] + " " + pct(p, 0)).join(" · ")}</div></div></div>
        <div class="ch" style="margin-top:14px">Multi-Task FT-Transformer – all outcomes at once</div>
        <table>${tasks.map(t => { const k = t.proba.indexOf(Math.max(...t.proba)); return `<tr><td><b>${esc(t.title)}</b></td><td>${esc(t.classes[k])}</td><td class="n">${pct(t.proba[k], 0)}</td></tr>`; }).join("")}</table>
        <div class="ch" style="margin-top:14px">Why? (change in risk score if this factor were typical)</div>
        ${hbar(Object.entries(r.attribution).slice(0, 6).map(([k, v]) => ({ label: k, value: Math.abs(v), color: v > 0 ? "#f43f5e" : "#34d399" })), { fmt: v => v.toFixed(3), labelW: 120, w: 460 })}
        <div class="muted" style="font-size:11.5px">red = pushes risk up · green = pulls risk down</div>`;
    };
  }
}
document.querySelectorAll("#crTabs button").forEach(b => b.onclick = () => renderCrash(b.dataset.t));

const WEATHERS = ["Clear", "Fair", "Partly Cloudy", "Mostly Cloudy", "Overcast", "Light Rain", "Rain", "Heavy Rain", "Light Drizzle", "Fog", "Haze", "Light Snow", "Snow", "Heavy Snow", "Light Freezing Rain", "Thunderstorm"];
function ctxForm(sel, p, v = {}) {
  const o = (id, opts, val) => `<select id="${p}_${id}">${opts.map(x => `<option ${x === val ? "selected" : ""}>${x}</option>`).join("")}</select>`;
  $(sel).innerHTML = `
    <label>Weather ${o("weather", WEATHERS, v.weather || "Light Snow")}</label>
    <label>Light ${o("light", ["Day", "Night"], v.light || "Night")}</label>
    <label>Road type ${o("road_type", ["Freeway", "Highway/Arterial", "Local street"], v.road_type || "Freeway")}</label>
    <label>Junction ${o("traffic_signal", ["No signal", "Signal"], v.traffic_signal || "No signal")}</label>
    <label>Temperature °F <input type="number" id="${p}_temperature_f" value="${v.temperature_f ?? 24}"></label>
    <label>Visibility (mi) <input type="number" id="${p}_visibility_mi" step="0.5" value="${v.visibility_mi ?? 3}"></label>`;
}
function readCtx(p, extra = {}) {
  const g = id => $(`#${p}_${id}`).value;
  return Object.assign({ weather: g("weather"), light: g("light"), road_type: g("road_type"), traffic_signal: g("traffic_signal"), temperature_f: +g("temperature_f"), wind_chill_f: +g("temperature_f") - 4, visibility_mi: +g("visibility_mi"), lat: 44.98, lng: -93.27, state: "MN", wind_dir: "NW" }, extra);
}

// ------------------------------------------------------------------ LIVE MONITOR
const CITIES = [["Minneapolis, MN", 44.98, -93.27, "MN"], ["Chicago, IL", 41.88, -87.63, "IL"], ["Dallas, TX", 32.78, -96.80, "TX"], ["Charlotte, NC", 35.23, -80.84, "NC"], ["New York, NY", 40.71, -74.0, "NY"], ["Coimbatore, India (outside training data)", 10.90, 76.90, "Other"]];
const WMO = c => c === 0 ? "Clear" : c === 1 ? "Fair" : c === 2 ? "Partly Cloudy" : c === 3 ? "Overcast" : c <= 48 ? "Fog" : c <= 55 ? "Light Drizzle" : c <= 57 ? "Light Freezing Rain" : c === 61 ? "Light Rain" : c === 63 ? "Rain" : c === 65 ? "Heavy Rain" : c <= 67 ? "Light Freezing Rain" : c === 71 ? "Light Snow" : c === 73 || c === 77 ? "Snow" : c === 75 ? "Heavy Snow" : c === 80 ? "Light Rain" : c <= 82 ? "Rain" : c <= 86 ? "Snow" : "Thunderstorm";
let L = { stream: null, i: 0, timer: null, aggr: 0, alcP: 0, lastLevel: null, alc: null, ctxRisk: 0, lastFuse: 0 };

function renderLive() {
  $("#tripSel").innerHTML = S.behavior.dataset.trips.map(t => `<option value="${t}">Real trip #${t}</option>`).join("");
  $("#citySel").innerHTML = CITIES.map((c, i) => `<option value="${i}">${c[0]}</option>`).join("");
  ctxForm("#ctxForm", "lv", { weather: "Light Snow", light: "Night", road_type: "Freeway", temperature_f: 24, visibility_mi: 3 });
  $("#ctxForm").insertAdjacentHTML("beforeend", `<label>Speed km/h <input type="number" id="lv_speed" value="95"></label><label>Speed limit <input type="number" id="lv_limit" value="90"></label>`);
  $("#ctxForm").querySelectorAll("select,input").forEach(e => e.onchange = () => fuseNow(true));
  $("#alcBadge").innerHTML = `<span class="badge ${S.alcohol.source.real ? "real" : "sim"}">${S.alcohol.source.real ? "Real data" : "Simulated data"}</span>`;
  const st = S.alcohol.feature_stats;
  const presets = {
    sober: Object.fromEntries(AF.map(f => [f, st[f].sober_mean])),
    mild: Object.fromEntries(AF.map(f => [f, (st[f].sober_mean + st[f].alcohol_mean) / 2])),
    drunk: Object.fromEntries(AF.map(f => [f, st[f].alcohol_mean + (st[f].alcohol_mean - st[f].sober_mean) * 0.4])),
  };
  L.alc = buildAlcSliders("#alcSliders", vals => { L.alcVals = vals; fuseNow(true); }, presets.sober);
  document.querySelectorAll("#alcPreset button").forEach(b => b.onclick = () => { document.querySelectorAll("#alcPreset button").forEach(x => x.classList.toggle("on", x === b)); L.alc.set(presets[b.dataset.p]); });
  $("#wxBtn").onclick = liveWeather;
  $("#playBtn").onclick = togglePlay;
  $("#tripSel").onchange = () => { stop(); loadTrip(); };
  $("#gaugeMain").innerHTML = gauge(0, { size: 260, label: "risk score" });
  loadTrip();
}
async function loadTrip() { L.stream = await api("/api/replay?trip=" + $("#tripSel").value); L.i = 0; drawSignal(); }
function stop() { clearInterval(L.timer); L.timer = null; $("#playBtn").textContent = "▶ Start trip"; }
function togglePlay() {
  if (L.timer) return stop();
  $("#playBtn").textContent = "❚❚ Pause";
  addAlert("Low", "Trip started – streaming real sensor data from trip #" + $("#tripSel").value);
  L.timer = setInterval(tick, 500 / (+$("#speedSel").value / 4));
}
function tick() {
  const s = L.stream.stream; if (L.i >= s.length) { stop(); addAlert("Low", "Trip finished"); return; }
  const p = s[L.i++];
  L.aggr = 0.6 * L.aggr + 0.4 * p.aggr;
  $("#nowPred").textContent = p.pred; $("#nowConf").textContent = " " + Math.round(p.conf * 100) + "%";
  $("#nowPred").style.color = ["Normal cruising", "Non-aggressive"].includes(p.pred) ? "#34d399" : "#fb7185";
  $("#nowTruth").textContent = p.truth || "—";
  $("#nowAggr").textContent = Math.round(L.aggr * 100) + "%";
  $("#tripClock").textContent = `t = ${Math.floor(p.t / 60)}:${String(Math.floor(p.t % 60)).padStart(2, "0")} / ${Math.floor(L.stream.duration / 60)} min`;
  if (p.aggr > 0.6 && p.pred !== L.lastEvt) { addAlert("Medium", `${p.pred} detected (${Math.round(p.conf * 100)}% confidence)` + (p.truth ? ` · ground truth: ${p.truth}` : "")); L.lastEvt = p.pred; }
  if (p.aggr < 0.3) L.lastEvt = null;
  drawSignal();
  fuseNow();
}
async function fuseNow(force) {
  const now = Date.now(); if (!force && now - L.lastFuse < 700) return; L.lastFuse = now;
  if (!L.alcVals) return;
  const c = CITIES[+$("#citySel").value || 0];
  const r = await api("/api/fuse", { aggr: L.aggr, alcohol: L.alcVals, context: readCtx("lv", { lat: c[1], lng: c[2], state: c[3] }), speed: +$("#lv_speed").value, speed_limit: +$("#lv_limit").value });
  $("#gaugeMain").innerHTML = gauge(r.risk, { size: 260, label: "fused risk score" });
  const lb = $("#levelBadge"); lb.className = "level " + r.level; lb.textContent = r.level.toUpperCase() + " RISK";
  const cols = { behaviour: "#22d3ee", alcohol: "#f472b6", context: "#8b5cf6", speeding: "#fbbf24" };
  $("#contribBars").innerHTML = `<div class="lbl" style="text-align:left;margin-bottom:6px">Share of the risk</div>` + Object.entries(r.contributions).map(([k, v]) => `<div class="cb"><span>${k}</span><div class="bar"><i style="width:${v * 100}%;background:${cols[k]}"></i></div><span>${Math.round(v * 100)}%</span></div>`).join("") +
    `<div class="mini-out" style="text-align:left;margin-top:14px"><span class="lbl">Actions</span>${r.actions.map(a => "• " + esc(a)).join("<br>")}</div>`;
  $("#alcOut").innerHTML = `<b>${r.alcohol.label}</b> · P(alcohol) = ${pct(r.alcohol.p_alcohol, 0)} <span class="muted">(${esc(r.alcohol.model)})</span>`;
  $("#ctxOut").innerHTML = `Context crash risk: <b style="color:${riskColor(r.context.risk_score)}">${r.context.risk_level}</b> · score ${r.context.risk_score.toFixed(2)} <span class="muted">(${esc(r.context.deployed_model)})</span><br><span class="muted">Likely severity outcome: ${esc(Object.values(r.context.mt_ftt)[0].classes[Object.values(r.context.mt_ftt)[0].proba.indexOf(Math.max(...Object.values(r.context.mt_ftt)[0].proba))])} · clearance ${esc(Object.values(r.context.mt_ftt)[2].classes[Object.values(r.context.mt_ftt)[2].proba.indexOf(Math.max(...Object.values(r.context.mt_ftt)[2].proba))])}</span>`;
  if (r.level !== L.lastLevel && L.lastLevel !== null) {
    addAlert(r.level, `Risk level changed to <b>${r.level}</b> (${Math.round(r.risk * 100)}) → ${r.actions.slice(0, 2).join("; ")}`);
    if (r.level === "High") toast(`🚨 <b>HIGH RISK</b> – SMS + call sent to emergency contact (simulated)<br><span style="opacity:.8">“Driver risk ${Math.round(r.risk * 100)}/100 at ${c[0]}. ${r.alcohol.p_alcohol >= .5 ? "Alcohol detected. " : ""}Please check on the driver.”</span>`);
  }
  L.lastLevel = r.level;
}
function addAlert(level, msg) {
  const f = $("#alertFeed"); if (f.querySelector(".muted")) f.innerHTML = "";
  const t = new Date().toLocaleTimeString();
  f.insertAdjacentHTML("afterbegin", `<div class="alert ${level}"><span class="t">${t}</span><b style="color:var(--${level === "High" ? "high" : level === "Medium" ? "med" : "low"})">${level}</b><span>${msg}</span></div>`);
}
async function liveWeather() {
  const c = CITIES[+$("#citySel").value];
  $("#wxBtn").textContent = "…";
  try {
    const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${c[1]}&longitude=${c[2]}&current=temperature_2m,weather_code,visibility,is_day&temperature_unit=fahrenheit`).then(r => r.json());
    const cur = r.current;
    $("#lv_weather").value = WMO(cur.weather_code); $("#lv_temperature_f").value = Math.round(cur.temperature_2m);
    $("#lv_visibility_mi").value = Math.min(10, (cur.visibility || 16000) / 1609).toFixed(1); $("#lv_light").value = cur.is_day ? "Day" : "Night";
    addAlert("Low", `Live weather for ${c[0]}: ${WMO(cur.weather_code)}, ${Math.round(cur.temperature_2m)}°F (Open-Meteo)`);
    fuseNow(true);
  } catch (e) { toast("Live weather unavailable (offline?) – using the manual values."); }
  $("#wxBtn").textContent = "⟳ Live weather";
}
function drawSignal() {
  const cv = $("#sigCanvas"); if (!cv || !L.stream) return;
  const W = cv.clientWidth || 600, H = 170; cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio;
  const g = cv.getContext("2d"); g.scale(devicePixelRatio, devicePixelRatio); g.clearRect(0, 0, W, H);
  const s = L.stream.stream, n = 120, a = Math.max(0, L.i - n), seg = s.slice(a, Math.max(L.i, 1));
  g.strokeStyle = "rgba(148,163,184,.12)"; for (let y = 0; y <= 4; y++) { g.beginPath(); g.moveTo(0, y * H / 4); g.lineTo(W, y * H / 4); g.stroke(); }
  seg.forEach((p, i) => { if (p.truth) { g.fillStyle = "rgba(251,191,36,.10)"; g.fillRect(i * W / n, 0, W / n + 1, H); } if (p.aggr > .5) { g.fillStyle = "rgba(244,63,94,.18)"; g.fillRect(i * W / n, H - 10, W / n + 1, 10); } });
  [["ax", "#22d3ee", 6], ["ay", "#a3e635", 6], ["gz", "#f472b6", 1.2]].forEach(([k, c, sc]) => {
    g.strokeStyle = c; g.lineWidth = 2; g.beginPath();
    seg.forEach((p, i) => { const x = i * W / n, y = H / 2 - p[k] / sc * H * 0.42; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke();
  });
  g.fillStyle = "#7c8aa6"; g.font = "11px Inter"; g.fillText("— lin. accel x   — lin. accel y   — yaw rate      ▮ labelled event   ▮ model: aggressive", 8, 14);
}

// ------------------------------------------------------------------ ROUTES
let MAP = { h: null, start: null, goal: null, res: null };
async function drawMap() {
  if (!MAP.h) MAP.h = await api("/api/hotspots");
  const h = MAP.h, cv = $("#mapCanvas"), [R, C] = h.grid;
  const W = cv.clientWidth || 800, H = Math.round(W * R / C * 1.05); cv.width = W * devicePixelRatio; cv.height = H * devicePixelRatio; cv.style.height = H + "px";
  const g = cv.getContext("2d"); g.scale(devicePixelRatio, devicePixelRatio);
  const cw = W / C, chh = H / R, mx = Math.max(...h.count.flat());
  g.fillStyle = "#050913"; g.fillRect(0, 0, W, H);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    const v = Math.log1p(h.count[r][c]) / Math.log1p(mx);
    g.fillStyle = v > 0 ? riskColor(v).replace("rgb", "rgba").replace(")", `,${0.12 + 0.55 * v})`) : "rgba(148,163,184,.03)";
    g.fillRect(c * cw + 1, (R - 1 - r) * chh + 1, cw - 2, chh - 2);
  }
  const [la0, la1, lo0, lo1] = h.box;
  g.fillStyle = "rgba(244,114,182,.75)";
  h.points.forEach(([la, lo, sv]) => { g.beginPath(); g.arc((lo - lo0) / (lo1 - lo0) * W, (1 - (la - la0) / (la1 - la0)) * H, sv >= 3 ? 2.2 : 1.3, 0, 7); g.fill(); });
  const cen = ([r, c]) => [c * cw + cw / 2, (R - 1 - r) * chh + chh / 2];
  const path = (p, col, wd) => { g.strokeStyle = col; g.lineWidth = wd; g.lineJoin = "round"; g.shadowColor = col; g.shadowBlur = 12; g.beginPath(); p.forEach((q, i) => { const [x, y] = cen(q); i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); g.shadowBlur = 0; };
  if (MAP.res) { path(MAP.res.fastest.path, "#22d3ee", 4); path(MAP.res.safest.path, "#a3e635", 4); }
  [[MAP.start, "#22d3ee", "A"], [MAP.goal, "#a3e635", "B"]].forEach(([p, c, t]) => { if (!p) return; const [x, y] = cen(p); g.fillStyle = c; g.beginPath(); g.arc(x, y, 11, 0, 7); g.fill(); g.fillStyle = "#050913"; g.font = "bold 12px Inter"; g.fillText(t, x - 4, y + 4); });
  g.fillStyle = "#e8eefc"; g.font = "600 13px Space Grotesk"; g.fillText(h.city + " · " + h.n_accidents.toLocaleString() + " real accidents", 12, 20);
  cv.onclick = e => { const b = cv.getBoundingClientRect(); const c = Math.floor((e.clientX - b.left) / cw), r = R - 1 - Math.floor((e.clientY - b.top) / chh);
    if (!MAP.start || MAP.goal) { MAP.start = [r, c]; MAP.goal = null; MAP.res = null; $("#mapHint").textContent = "Click to set DESTINATION"; drawMap(); }
    else { MAP.goal = [r, c]; $("#mapHint").textContent = "Click to start a new route"; computeRoute(); } };
}
async function computeRoute() {
  if (!MAP.start || !MAP.goal) return;
  MAP.res = await api("/api/route", { start: MAP.start, goal: MAP.goal, safety_weight: +$("#swt").value, context_multiplier: +$("#rtWx").value });
  const r = MAP.res, row = (k, a, b) => `<div class="kv"><span class="muted">${k}</span><span>${a}</span></div>`;
  $("#routeStats").innerHTML = `<div class="ch">Route comparison</div>
    <div class="rs"><h4 style="color:#22d3ee">Fastest route</h4>${row("Distance", r.fastest.km + " km")}${row("Est. time", r.fastest.minutes + " min")}${row("Risk exposure", r.fastest.exposure)}${row("Hotspot cells crossed", r.fastest.hotspots_crossed)}</div>
    <div class="rs"><h4 style="color:#a3e635">Safest route (recommended)</h4>${row("Distance", r.safest.km + " km")}${row("Est. time", r.safest.minutes + " min")}${row("Risk exposure", r.safest.exposure)}${row("Hotspot cells crossed", r.safest.hotspots_crossed)}</div>
    <div class="big-green">−${r.risk_reduction_pct}%</div><div class="muted">risk exposure for +${r.extra_minutes} min travel time</div>
    <div class="note info" style="margin-top:12px">Cost per cell = distance × (1 + w × hotspot risk × weather factor). Hotspot risk comes from real accident counts and the share of severe (level ≥ 3) accidents.</div>`;
  drawMap();
}
$("#swt").oninput = e => { $("#swtv").textContent = e.target.value; computeRoute(); };
$("#rtWx").onchange = computeRoute;
$("#rtReset").onclick = () => { MAP.start = MAP.goal = MAP.res = null; $("#mapHint").textContent = "Click to set START"; $("#routeStats").innerHTML = `<div class="ch">Route comparison</div><div class="muted">Pick two points on the map.</div>`; drawMap(); };

// ------------------------------------------------------------------ PAPER MAPPING
function renderPaper() {
  const P = S.team.standard_paper;
  $("#paperRef").innerHTML = `<b>${esc(P.title)}</b><br>${esc(P.authors)} · ${esc(P.venue)} · DOI ${esc(P.doi)}`;
  const rows = [
    ["Problem", "Predict accident risk in real time and recommend safer routes using multi-source data", "Same goal: real-time risk + safer route", "Adds the driver's own state – driving behaviour and alcohol – which the paper does not model"],
    ["Data sources", "Accident history, weather, traffic, GPS, road features, driver demographics", "Real accident history (US-Accidents, 314k) with weather, visibility, road type, signals, location; live weather API", "Real smartphone IMU trips; in-cabin alcohol sensors (IEEE DataPort schema); STATS19 loader (IEEE DataPort)"],
    ["Pre-processing", "Imputation, one-hot / label encoding, normalisation, feature engineering (risk index, hotspots)", "One-hot encoding, standardisation, engineered road type from street names, geo-cells, hotspot grid", "Signal resampling to 50 Hz, windowing, FFT band features"],
    ["Target", "Risk level Low / Medium / High", "Risk level Low / Medium / High (Severity 1–2 / 3 / 4)", "Also road length affected and clearance time (multi-task)"],
    ["Models", "LR, RF, SVM, NN, XGBoost, GB, AdaBoost, Extra Trees, LightGBM – 80/20 split, grid search", "Same comparison: LR, RF, Extra Trees, GB, AdaBoost, Hist-GB (+XGBoost / LightGBM when installed), 80/20, grid search", "Enhanced CCP tree (Paper 3) and Multi-Task FT-Transformer (Paper 4)"],
    ["Evaluation", "Accuracy, macro-F1, ROC-AUC (OvR), confusion matrix, feature importance, processing time", "Identical metric set + train time", "Balanced accuracy; leave-one-trip-out CV for behaviour; event-level scoring"],
    ["Deployment", "REST API + dashboard (Streamlit), Flutter app, Firebase location, heat maps", "REST API + web dashboard, live weather (Open-Meteo), hotspot heat map", "Live replay of real trips through all models at once"],
    ["Alerts", "Notification by message / call to the registered number when risk is high", "Threshold alerts + emergency-contact notification (simulated)", "Graded actions: warning, coaching, interlock advice"],
    ["Route recommendation", "Weighs safety against travel time and distance", "Dijkstra on a risk-weighted grid; fastest vs safest", "Weather factor scales hotspot risk; user-tunable safety weight"],
  ];
  const C = S.crash.standard_paper.comparison;
  $("#paperBody").innerHTML = `<div class="card"><table class="map-table"><tr><th>Aspect</th><th>Standard paper</th><th><span class="tick">✓</span> What we took</th><th><span class="ext">＋</span> What we extended</th></tr>
    ${rows.map(r => `<tr><td>${r[0]}</td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`).join("")}</table></div>
    <div class="grid2" style="margin-top:18px"><div class="card"><div class="ch">Their Table I (their dataset)</div><table><tr><th>Classifier</th><th class="n">Acc</th><th class="n">Macro F1</th><th class="n">ROC-AUC</th></tr>${Object.entries(PAPER_T1).map(([k, v]) => `<tr><td>${k}</td><td class="n">${v[0].toFixed(4)}</td><td class="n">${v[1].toFixed(3)}</td><td class="n">${v[2].toFixed(4)}</td></tr>`).join("")}</table></div>
    <div class="card"><div class="ch">Our Table I (real US-Accidents)</div><table><tr><th>Classifier</th><th class="n">Acc</th><th class="n">Macro F1</th><th class="n">ROC-AUC</th></tr>${C.map(r => `<tr class="${r.model === S.crash.standard_paper.deployed ? "best" : ""}"><td>${r.model}</td><td class="n">${r.accuracy.toFixed(4)}</td><td class="n">${r.macro_f1.toFixed(3)}</td><td class="n">${(r.roc_auc || 0).toFixed(4)}</td></tr>`).join("")}</table>
    <div class="note info">Like the paper, the boosted-tree model gives the best ROC-AUC / macro-F1 balance and is the one deployed.</div></div></div>`;
}

// ------------------------------------------------------------------ FUTURE
function renderFuture() {
  const F = [
    ["📱", "Mobile app + real phone sensors", "Stream the live phone IMU directly instead of replaying recorded trips (Flutter, as in the standard paper).", memberFor("behaviour")],
    ["🧪", "Real IEEE DataPort alcohol data", "Swap the simulated alcohol set for the IEEE DataPort file and add breath-sensor hardware (MQ-3 + ESP32).", memberFor("alcohol")],
    ["🇬🇧", "STATS19 training + Indian data", "Train all crash models on the IEEE DataPort STATS19 file, then validate on Indian state accident records.", memberFor("crash")],
    ["⏱", "Real-time crash prediction", "Add the ST BiLSTM-Transformer (Review-2 Paper 2) once connected-vehicle traffic data is available.", memberFor("crash")],
    ["🗺", "Real road-network routing", "Replace the city grid with OpenStreetMap roads and live traffic for turn-by-turn safer routes.", memberFor("fusion")],
    ["📞", "Real SMS / call alerts", "Twilio / Firebase Cloud Messaging for actual emergency-contact notifications.", memberFor("fusion")],
    ["🧠", "Learned fusion", "Replace the noisy-OR rule with a model trained on jointly-labelled trips (behaviour + impairment + outcome).", memberFor("fusion")],
    ["😴", "Drowsiness & distraction camera", "Add a driver-facing camera model (fatigue, phone use) as a fourth risk source.", "Team"],
    ["🔍", "SHAP explanations", "Per-prediction SHAP plots (as in Papers 2 and 4) once the shap package is part of the build.", "Team"],
  ];
  $("#futureBody").innerHTML = F.map(f => `<div class="card"><div class="ic">${f[0]}</div><h3>${f[1]}</h3><p style="margin:0 0 8px">${f[2]}</p><div class="who">${esc(f[3])}</div></div>`).join("");
}

// ------------------------------------------------------------------ boot
(async () => {
  S = await api("/api/summary");
  renderOverview(); renderBehavior(); renderAlcohol(); renderCrash(); renderLive(); renderPaper(); renderFuture();
  const h = location.hash.slice(1); if (h && $("#page-" + h)) go(h);
  window.addEventListener("resize", () => { drawSignal(); if ($("#page-routes").classList.contains("on")) drawMap(); });
})();
