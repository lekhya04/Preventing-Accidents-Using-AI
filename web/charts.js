// Tiny dependency-free SVG chart helpers for the SafeDrive AI dashboard
const NS = "http://www.w3.org/2000/svg";
const PAL = ["#22d3ee", "#8b5cf6", "#a3e635", "#fbbf24", "#f472b6", "#34d399", "#60a5fa", "#fb923c"];
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (v, d = 1) => v == null ? "—" : (v * 100).toFixed(d) + "%";
const f3 = v => v == null ? "—" : Number(v).toFixed(3);

function riskColor(v) { // 0..1 -> green..amber..red
  const stops = [[0, [52, 211, 153]], [0.45, [251, 191, 36]], [1, [244, 63, 94]]];
  for (let i = 1; i < stops.length; i++) if (v <= stops[i][0]) {
    const [a, ca] = stops[i - 1], [b, cb] = stops[i], t = (v - a) / (b - a);
    return `rgb(${ca.map((c, k) => Math.round(c + (cb[k] - c) * t)).join(",")})`;
  }
  return "rgb(244,63,94)";
}

// Horizontal bar chart: items [{label, value, color?, note?}]
function hbar(items, { max = null, fmt = pct, w = 520, barH = 22, gap = 9, labelW = 190, color = PAL[0] } = {}) {
  const m = max ?? Math.max(...items.map(i => i.value), 1e-9);
  const h = items.length * (barH + gap) + 4;
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%" style="max-width:${w}px">`;
  items.forEach((it, i) => {
    const y = i * (barH + gap), bw = Math.max(2, (w - labelW - 70) * it.value / m);
    s += `<text x="${labelW - 10}" y="${y + barH / 2 + 4}" text-anchor="end" fill="#b4c0d8" font-size="12">${esc(it.label)}</text>
      <rect x="${labelW}" y="${y}" width="${w - labelW - 70}" height="${barH}" rx="6" fill="rgba(148,163,184,.08)"/>
      <rect class="growx" x="${labelW}" y="${y}" width="${bw}" height="${barH}" rx="6" fill="${it.color || color}"/>
      <text x="${labelW + bw + 8}" y="${y + barH / 2 + 4}" fill="#e8eefc" font-size="12" font-family="JetBrains Mono">${fmt(it.value)}</text>`;
  });
  return s + "</svg>";
}

// Grouped vertical bars: groups [{label, values:[..]}], series names
function gbar(groups, series, { w = 640, h = 260, max = 1, fmt = v => (v * 100).toFixed(0) } = {}) {
  const pl = 34, pb = 46, pt = 24, cw = (w - pl - 10) / groups.length, bw = Math.min(26, (cw - 14) / series.length);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
  for (let g = 0; g <= 4; g++) {
    const y = pt + (h - pt - pb) * (1 - g / 4);
    s += `<line x1="${pl}" x2="${w - 6}" y1="${y}" y2="${y}" stroke="rgba(148,163,184,.12)"/><text x="${pl - 6}" y="${y + 4}" text-anchor="end" fill="#7c8aa6" font-size="10">${fmt(max * g / 4)}</text>`;
  }
  groups.forEach((gr, i) => {
    const x0 = pl + i * cw + (cw - bw * series.length) / 2;
    gr.values.forEach((v, k) => {
      if (v == null) return;
      const bh = (h - pt - pb) * v / max, x = x0 + k * bw, y = h - pb - bh;
      s += `<rect class="growy" x="${x + 1}" y="${y}" width="${bw - 3}" height="${bh}" rx="3" fill="${PAL[k % PAL.length]}"><title>${esc(series[k])}: ${fmt(v)}</title></rect>`;
    });
    const lbl = gr.label.length > 16 ? gr.label.slice(0, 15) + "…" : gr.label;
    s += `<text x="${pl + i * cw + cw / 2}" y="${h - pb + 16}" text-anchor="middle" fill="#b4c0d8" font-size="11">${esc(lbl)}</text>`;
  });
  let lx = pl;
  series.forEach((n, k) => { s += `<rect x="${lx}" y="4" width="10" height="10" rx="2" fill="${PAL[k % PAL.length]}"/><text x="${lx + 15}" y="13" fill="#b4c0d8" font-size="11">${esc(n)}</text>`; lx += 24 + n.length * 6.6; });
  return s + "</svg>";
}

// Confusion-matrix heatmap
function confusion(mat, labels, { w = 470 } = {}) {
  const n = labels.length, lw = 150, top = 104, cell = Math.min(46, (w - lw) / n), h = top + cell * n + 6;
  const rows = mat.map(r => r.reduce((a, b) => a + b, 0) || 1);
  let s = `<svg viewBox="0 0 ${lw + cell * n + 8} ${h}" width="100%" style="max-width:${lw + cell * n + 8}px">`;
  labels.forEach((l, j) => s += `<text transform="translate(${lw + j * cell + cell / 2 + 4},${top - 8}) rotate(-50)" fill="#7c8aa6" font-size="10.5">${esc(l)}</text>`);
  mat.forEach((r, i) => {
    s += `<text x="${lw - 8}" y="${top + i * cell + cell / 2 + 4}" text-anchor="end" fill="#b4c0d8" font-size="11">${esc(labels[i])}</text>`;
    r.forEach((v, j) => {
      const p = v / rows[i];
      const fill = i === j ? `rgba(34,211,238,${0.12 + 0.8 * p})` : `rgba(244,63,94,${p * 0.85})`;
      s += `<rect x="${lw + j * cell}" y="${top + i * cell}" width="${cell - 3}" height="${cell - 3}" rx="5" fill="${fill}" stroke="rgba(148,163,184,.1)"><title>${esc(labels[i])} → ${esc(labels[j])}: ${v}</title></rect>
        <text x="${lw + j * cell + cell / 2 - 1}" y="${top + i * cell + cell / 2 + 3}" text-anchor="middle" fill="${p > .5 ? "#fff" : "#b4c0d8"}" font-size="11" font-family="JetBrains Mono">${v}</text>`;
    });
  });
  return s + `<text x="4" y="${top - 8}" fill="#7c8aa6" font-size="10">true ↓ / predicted →</text></svg>`;
}

// Line chart: series [{name, points:[[x,y]..]}]
function line(series, { w = 560, h = 230, xl = "", yl = "", ylim = null, xfmt = v => v, yfmt = v => v.toFixed(2) } = {}) {
  const pl = 44, pb = 34, pt = 26, pr = 12;
  const xs = series.flatMap(s => s.points.map(p => p[0])), ys = series.flatMap(s => s.points.map(p => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = ylim ? ylim[0] : Math.min(...ys), y1 = ylim ? ylim[1] : Math.max(...ys);
  const X = v => pl + (w - pl - pr) * (v - x0) / ((x1 - x0) || 1), Y = v => h - pb - (h - pb - pt) * (v - y0) / ((y1 - y0) || 1);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
  for (let g = 0; g <= 4; g++) { const v = y0 + (y1 - y0) * g / 4; s += `<line x1="${pl}" x2="${w - pr}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(148,163,184,.12)"/><text x="${pl - 6}" y="${Y(v) + 4}" text-anchor="end" fill="#7c8aa6" font-size="10">${yfmt(v)}</text>`; }
  series.forEach((sr, k) => {
    const d = sr.points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join("");
    const c = sr.color || PAL[k % PAL.length];
    s += `<path d="${d}" fill="none" stroke="${c}" stroke-width="2.4" stroke-linejoin="round"/>`;
    sr.points.forEach(p => s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.2" fill="${c}"><title>${sr.name}: ${xfmt(p[0])} → ${yfmt(p[1])}</title></circle>`);
  });
  const ux = [...new Set(xs)], stepx = Math.ceil(ux.length / 9);
  ux.filter((v, i) => i % stepx === 0 || i === ux.length - 1).forEach(v => s += `<text x="${X(v)}" y="${h - pb + 15}" text-anchor="middle" fill="#7c8aa6" font-size="10">${xfmt(v)}</text>`);
  let lx = pl;
  series.forEach((sr, k) => { s += `<rect x="${lx}" y="5" width="10" height="3" fill="${sr.color || PAL[k % PAL.length]}"/><text x="${lx + 14}" y="11" fill="#b4c0d8" font-size="11">${esc(sr.name)}</text>`; lx += 26 + sr.name.length * 6.5; });
  if (xl) s += `<text x="${w - pr}" y="${h - 4}" text-anchor="end" fill="#7c8aa6" font-size="10.5">${esc(xl)}</text>`;
  return s + "</svg>";
}

// Semi-circular gauge
function gauge(v, { size = 250, label = "" } = {}) {
  const r = size * 0.38, cx = size / 2, cy = size * 0.55, a0 = Math.PI, a = a0 - Math.PI * Math.max(0, Math.min(1, v));
  const P = t => [cx + r * Math.cos(t), cy - r * Math.sin(t)];
  const arc = (t0, t1, col, wdt) => { const [x0, y0] = P(t0), [x1, y1] = P(t1); return `<path d="M${x0},${y0} A${r},${r} 0 0 1 ${x1},${y1}" stroke="${col}" stroke-width="${wdt}" fill="none" stroke-linecap="round"/>`; };
  let s = `<svg viewBox="0 0 ${size} ${size * 0.72}" width="100%" style="max-width:${size}px">
    <defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="#34d399"/><stop offset=".5" stop-color="#fbbf24"/><stop offset="1" stop-color="#f43f5e"/></linearGradient></defs>`;
  s += arc(Math.PI, 0.0001, "rgba(148,163,184,.12)", 18) + arc(Math.PI, a + 0.0001, "url(#gg)", 18);
  const [nx, ny] = P(a);
  s += `<circle cx="${nx}" cy="${ny}" r="9" fill="#0b1020" stroke="${riskColor(v)}" stroke-width="4"/>`;
  s += `<text x="${cx}" y="${cy - 4}" text-anchor="middle" fill="#e8eefc" font-size="${size * 0.17}" font-family="Space Grotesk" font-weight="700">${Math.round(v * 100)}</text>
        <text x="${cx}" y="${cy + 18}" text-anchor="middle" fill="#7c8aa6" font-size="11">${esc(label)}</text></svg>`;
  return s;
}
