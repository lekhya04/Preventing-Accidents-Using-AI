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
  const B = S.behavior, bm = B.models[B.deployed];
  $("#heroKpis").innerHTML = [
    { v: pct(bm.aggressive_vs_safe_accuracy, 0), l: "aggressive-vs-safe driving detection" },
    { v: "3", l: "risk levels: Low · Medium · High" },
    { v: "3 + 1", l: "ML modules + AI fusion layer" },
    { v: "Real-time", l: "fused risk score and alerts" },
  ].map(k => `<div class="kpi"><div class="v">${k.v}</div><div class="l">${k.l}</div></div>`).join("");

  $("#arch").innerHTML = archSVG();
  $("#team").innerHTML = S.team.members.map((m, i) => `
    <div class="card member">
      <div class="av" style="background:${["#22d3ee", "#f472b6", "#a3e635"][i % 3]}">${esc(m.name[0])}</div>
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
    ${src(18, "Smartphone IMU", "accel · gyro · 50 Hz")}${src(92, "In-cabin sensors", "MQ-3 · thermal · camera")}${src(166, "Accident history", "road · weather · location")}${src(240, "Live weather / GPS", "live weather · location")}
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
      <div class="stat"><div class="v">${pct(dep.aggressive_vs_safe_accuracy)}</div><div class="l">aggressive vs safe detection</div></div>
      <div class="stat"><div class="v">${pct(dep.event.accuracy)}</div><div class="l">event-type accuracy (${esc(B.deployed)})</div></div>
    </div>
    <div class="card"><div class="ch">Pipeline</div>
      <div class="pipeline">
        <div class="st"><b>1 · Sensors</b><small>Smartphone accelerometer and gyroscope</small></div>
        <div class="st"><b>2 · Windowing</b><small>2-second sliding windows</small></div>
        <div class="st"><b>3 · Features</b><small>mean, std, min, max, range, RMS, jerk + FFT band energies</small></div>
        <div class="st"><b>4 · Classifier</b><small>Gaussian RBF-SVM / Random Forest</small></div>
      </div>
    </div>
    <div class="grid2" style="margin-top:18px">
      <div class="card"><div class="ch">Most informative features</div>${hbar(B.top_features.slice(0, 10).map(f => ({ label: f.feature, value: f.importance })), { fmt: v => (v * 100).toFixed(1) + "%", labelW: 170 })}</div>
      <div class="card"><div class="ch">Confusion matrix – events (${esc(B.deployed)})</div>${confusion(dep.event.confusion, labs)}</div>
    </div>
    <div class="card" style="margin-top:18px"><div class="ch">Trip replay – model output vs ground truth <select id="behTrip">${B.dataset.trips.map(t => `<option>${t}</option>`).join("")}</select></div><div id="behTimeline"></div></div>`;
  $("#behTrip").onchange = e => timeline(+e.target.value);
  timeline(B.dataset.trips[0]);
}

const EVC = { "Aggressive acceleration": "#fb923c", "Aggressive braking": "#f43f5e", "Aggressive left turn": "#8b5cf6", "Aggressive right turn": "#a78bfa", "Aggressive left lane change": "#22d3ee", "Aggressive right lane change": "#67e8f9", "Non-aggressive": "#34d399", "Speed bump": "#facc15", "Pothole": "#eab308", "Normal cruising": "rgba(148,163,184,.25)" };
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
  const A = S.alcohol;
  $("#alcBody").innerHTML = `
    <div class="stat-row">
      <div class="stat"><div class="v">${pct(A.models[A.deployed].accuracy)}</div><div class="l">${esc(A.deployed)} accuracy</div></div>
      <div class="stat"><div class="v">${f3(A.models[A.deployed].roc_auc)}</div><div class="l">ROC-AUC</div></div>
    </div>
    <div class="grid2">
      <div class="card"><div class="ch">Sensor importance</div>${hbar(Object.entries(A.feature_importance).map(([k, v]) => ({ label: k, value: v, color: "#f472b6" })), { labelW: 190 })}</div>
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
function renderCrash() {
  const body = $("#crBody");
  {
    body.innerHTML = `<div class="grid2"><div class="card"><div class="ch">Road & weather conditions</div><div class="formgrid" id="crForm"></div><button class="btn primary" id="crGo">Predict crash risk</button></div>
      <div class="card" id="crOut"><div class="ch">Prediction</div><div class="muted">Set conditions and press Predict.</div></div></div>`;
    ctxForm("#crForm", "cr");
    $("#crGo").onclick = async () => {
      const r = await api("/api/crash", readCtx("cr"));
      const tasks = Object.values(r.mt_ftt);
      $("#crOut").innerHTML = `<div class="ch">Prediction <span class="muted">${esc(r.deployed_model)}</span></div>
        <div style="display:grid;grid-template-columns:220px 1fr;gap:16px;align-items:center"><div style="text-align:center">${gauge(r.risk_score, { size: 220, label: "context risk score" })}<div class="level ${r.risk_level}">${r.risk_level} risk</div></div>
        <div>${hbar(r.risk_labels.map((l, i) => ({ label: "P(" + l + ")", value: r.risk_proba[i], color: ["#34d399", "#fbbf24", "#f43f5e"][i] })), { max: 1, labelW: 110, w: 420 })}
        <div class="muted" style="font-size:12px;margin-top:4px">Enhanced CCP tree: ${r.ccp_proba.map((p, i) => r.risk_labels[i] + " " + pct(p, 0)).join(" · ")}</div></div></div>
        <div class="ch" style="margin-top:14px">Expected outcomes</div>
        <table>${tasks.map(t => { const k = t.proba.indexOf(Math.max(...t.proba)); return `<tr><td><b>${esc(t.title)}</b></td><td>${esc(t.classes[k])}</td><td class="n">${pct(t.proba[k], 0)}</td></tr>`; }).join("")}</table>
        <div class="ch" style="margin-top:14px">Why? (change in risk score if this factor were typical)</div>
        ${hbar(Object.entries(r.attribution).slice(0, 6).map(([k, v]) => ({ label: k, value: Math.abs(v), color: v > 0 ? "#f43f5e" : "#34d399" })), { fmt: v => v.toFixed(3), labelW: 120, w: 460 })}
        <div class="muted" style="font-size:11.5px">red = pushes risk up · green = pulls risk down</div>`;
    };
  }
}

const WEATHERS = ["Clear", "Partly Cloudy", "Overcast", "Haze", "Fog", "Drizzle", "Light Rain", "Rain", "Heavy Rain", "Thunderstorm"];
function ctxForm(sel, p, v = {}) {
  const o = (id, opts, val) => `<select id="${p}_${id}">${opts.map(x => `<option ${x === val ? "selected" : ""}>${x}</option>`).join("")}</select>`;
  $(sel).innerHTML = `
    <label>Weather ${o("weather", WEATHERS, v.weather || "Clear")}</label>
    <label>Light ${o("light", ["Day", "Night"], v.light || "Day")}</label>
    <label>Road type ${o("road_type", ["Expressway/Bypass", "Highway/Arterial", "Local street"], v.road_type || "Highway/Arterial")}</label>
    <label>Junction ${o("traffic_signal", ["No signal", "Signal"], v.traffic_signal || "No signal")}</label>
    <label>Temperature °C <input type="number" id="${p}_temperature_c" value="${v.temperature_c ?? 30}"></label>
    <label>Visibility (km) <input type="number" id="${p}_visibility_km" step="0.5" value="${v.visibility_km ?? 10}"></label>
    <label>Vehicle ${o("vehicle", ["two_wheeler", "car", "auto_rickshaw", "bus", "goods_vehicle"], v.vehicle || "car")}</label>
    <label>Speed limit (km/h) <input type="number" id="${p}_speed_limit_kmh" step="10" value="${v.speed_limit_kmh ?? 50}"></label>`;
}
function readCtx(p, extra = {}) {
  const g = id => $(`#${p}_${id}`).value;
  return Object.assign({ weather: g("weather"), light: g("light"), road_type: g("road_type"), traffic_signal: g("traffic_signal"), temperature_c: +g("temperature_c"), visibility_km: +g("visibility_km"), vehicle: g("vehicle"), speed_limit_kmh: +g("speed_limit_kmh"), lat: 10.99, lng: 76.97, wind_dir: "W" }, extra);
}

// ------------------------------------------------------------------ LIVE MONITOR
const CITIES = [["Coimbatore, India", 10.99, 76.97], ["Chennai, India", 13.08, 80.27], ["Bengaluru, India", 12.97, 77.59], ["Hyderabad, India", 17.39, 78.49], ["Mumbai, India", 19.08, 72.88]];
const WMO = c => c === 0 ? "Clear" : c === 1 ? "Clear" : c === 2 ? "Partly Cloudy" : c === 3 ? "Overcast" : c <= 48 ? "Fog" : c <= 55 ? "Drizzle" : c <= 57 ? "Drizzle" : c === 61 ? "Light Rain" : c === 63 ? "Rain" : c === 65 ? "Heavy Rain" : c <= 67 ? "Rain" : c <= 77 ? "Overcast" : c === 80 ? "Light Rain" : c <= 82 ? "Rain" : c <= 86 ? "Overcast" : "Thunderstorm";
let L = { stream: null, i: 0, timer: null, aggr: 0, alcP: 0, lastLevel: null, alc: null, ctxRisk: 0, lastFuse: 0 };

function renderLive() {
  $("#tripSel").innerHTML = S.behavior.dataset.trips.map(t => `<option value="${t}">Trip #${t}</option>`).join("");
  $("#citySel").innerHTML = CITIES.map((c, i) => `<option value="${i}">${c[0]}</option>`).join("");
  ctxForm("#ctxForm", "lv", {});
  $("#ctxForm").insertAdjacentHTML("beforeend", `<label>Speed km/h <input type="number" id="lv_speed" value="45"></label>`);
  const CTX_PRESETS = {
    normal: { weather: "Clear", light: "Day", road_type: "Highway/Arterial", traffic_signal: "No signal", temperature_c: 30, visibility_km: 10, vehicle: "car", speed_limit_kmh: 50, speed: 45 },
    rain: { weather: "Heavy Rain", light: "Night", road_type: "Expressway/Bypass", traffic_signal: "No signal", temperature_c: 24, visibility_km: 2, vehicle: "two_wheeler", speed_limit_kmh: 80, speed: 95 },
    fog: { weather: "Fog", light: "Day", road_type: "Highway/Arterial", traffic_signal: "No signal", temperature_c: 23, visibility_km: 1, vehicle: "car", speed_limit_kmh: 50, speed: 50 },
  };
  $("#ctxForm").insertAdjacentHTML("afterbegin", `<label>Scenario <select id="ctxPreset"><option value="normal">Normal day (default)</option><option value="rain">Hazardous: heavy rain, night, speeding</option><option value="fog">Foggy day</option></select></label>`);
  $("#ctxPreset").addEventListener("change", e => { Object.entries(CTX_PRESETS[e.target.value]).forEach(([k, v]) => { $("#lv_" + k).value = v; }); fuseNow(true); });
  $("#ctxForm").querySelectorAll("select,input").forEach(e => e.onchange = () => fuseNow(true));
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
  addAlert("Low", "Trip started – streaming sensor data from trip #" + $("#tripSel").value);
  L.timer = setInterval(tick, 500 / (+$("#speedSel").value / 4));
}
function tick() {
  const s = L.stream.stream; if (L.i >= s.length) { stop(); addAlert("Low", "Trip finished"); return; }
  const p = s[L.i++];
  L.aggr = 0.6 * L.aggr + 0.4 * p.aggr;
  $("#nowPred").textContent = p.pred; $("#nowConf").textContent = " " + Math.round(p.conf * 100) + "%";
  $("#nowPred").style.color = ["Normal cruising", "Non-aggressive", "Speed bump", "Pothole"].includes(p.pred) ? "#34d399" : "#fb7185";
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
  const r = await api("/api/fuse", { aggr: L.aggr, alcohol: L.alcVals, context: readCtx("lv", { lat: c[1], lng: c[2], }), speed: +$("#lv_speed").value, speed_limit: +$("#lv_speed_limit_kmh").value });
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
    const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${c[1]}&longitude=${c[2]}&current=temperature_2m,weather_code,visibility,is_day`).then(r => r.json());
    const cur = r.current;
    $("#lv_weather").value = WMO(cur.weather_code); $("#lv_temperature_c").value = Math.round(cur.temperature_2m);
    $("#lv_visibility_km").value = Math.min(10, (cur.visibility || 10000) / 1000).toFixed(1); $("#lv_light").value = cur.is_day ? "Day" : "Night";
    addAlert("Low", `Live weather for ${c[0]}: ${WMO(cur.weather_code)}, ${Math.round(cur.temperature_2m)}°C (Open-Meteo)`);
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
  g.fillStyle = "#e8eefc"; g.font = "600 13px Space Grotesk"; g.fillText(h.city + " · " + h.n_accidents.toLocaleString() + " accidents", 12, 20);
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
    `;
  drawMap();
}
$("#swt").oninput = e => { $("#swtv").textContent = e.target.value; computeRoute(); };
$("#rtWx").onchange = computeRoute;
$("#rtReset").onclick = () => { MAP.start = MAP.goal = MAP.res = null; $("#mapHint").textContent = "Click to set START"; $("#routeStats").innerHTML = `<div class="ch">Route comparison</div><div class="muted">Pick two points on the map.</div>`; drawMap(); };

// ------------------------------------------------------------------ boot
(async () => {
  S = await api("/api/summary");
  renderOverview(); renderBehavior(); renderAlcohol(); renderCrash(); renderLive();
  const h = location.hash.slice(1); if (h && $("#page-" + h)) go(h);
  window.addEventListener("resize", () => { drawSignal(); if ($("#page-routes").classList.contains("on")) drawMap(); });
})();
