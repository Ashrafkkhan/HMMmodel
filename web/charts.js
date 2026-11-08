/* Dependency-free chart helpers: SVG strings and a canvas scatter. */
const PALETTE = ['#6ea8fe', '#f59e8b', '#7ee0b0', '#e6c36a', '#c39bf5', '#6fd6e0', '#f08fc0', '#a8b4c4'];
const colorOf = (c) => PALETTE[c % PALETTE.length];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const fmt = {
  int: (n) => (n == null ? '–' : Math.round(n).toLocaleString('en-US')),
  num: (n, d = 2) => (n == null ? '–' : Number(n).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d })),
  pct: (n, d = 1) => (n == null ? '–' : Number(n).toFixed(d) + '%'),
  compact: (n) => {
    if (n == null) return '–';
    const a = Math.abs(n);
    if (a >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (a >= 1e3) return (n / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'k';
    return Number(n).toFixed(a < 10 && a % 1 ? 2 : 0);
  },
};

function niceTicks(min, max, n = 5) {
  if (!(max > min)) { max = min + 1; }
  const raw = (max - min) / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw);
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

const M = { l: 54, r: 14, t: 14, b: 34 };
const frame = (W, H, inner) => `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img">${inner}</svg>`;

function yAxis(ticks, sy, W, f) {
  return ticks.map((t) => `<line class="grid-l" x1="${M.l}" x2="${W - M.r}" y1="${sy(t)}" y2="${sy(t)}"/>` +
    `<text x="${M.l - 8}" y="${sy(t) + 3}" text-anchor="end">${f(t)}</text>`).join('');
}

function barChart({ labels, values, colors, f = fmt.compact, W = 620, H = 260, valueLabels = true }) {
  const max = Math.max(...values, 1e-9), ticks = niceTicks(0, max);
  const top = Math.max(ticks[ticks.length - 1], max * 1.1);
  const sy = (v) => H - M.b - (v / top) * (H - M.b - M.t);
  const bw = (W - M.l - M.r) / values.length;
  let s = yAxis(ticks, sy, W, f);
  values.forEach((v, i) => {
    const x = M.l + i * bw + bw * 0.2, w = bw * 0.6, y = sy(v);
    s += `<rect class="bar" x="${x}" y="${y}" width="${w}" height="${H - M.b - y}" rx="5" fill="${colors[i]}"><title>${esc(labels[i])}: ${f(v)}</title></rect>`;
    if (valueLabels) s += `<text x="${x + w / 2}" y="${y - 5}" text-anchor="middle" style="fill:var(--text)">${f(v)}</text>`;
    s += `<text x="${x + w / 2}" y="${H - M.b + 16}" text-anchor="middle">${esc(labels[i])}</text>`;
  });
  return frame(W, H, s);
}

function lineChart({ xs, series, f = fmt.compact, W = 620, H = 260, mark = null, xLabel = '' }) {
  const all = series.flatMap((s) => s.ys).filter((v) => v != null);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.1 || 1; lo -= pad; hi += pad;
  const ticks = niceTicks(lo, hi), a = lo, b = hi;
  const sy = (v) => H - M.b - ((v - a) / (b - a)) * (H - M.b - M.t);
  const sx = (x) => M.l + ((x - xs[0]) / (xs[xs.length - 1] - xs[0] || 1)) * (W - M.l - M.r);
  let s = yAxis(ticks, sy, W, f);
  const skip = Math.ceil(xs.length / 10);
  xs.forEach((x, i) => { if (i % skip === 0 || i === xs.length - 1) s += `<text x="${sx(x)}" y="${H - M.b + 16}" text-anchor="middle">${x}</text>`; });
  if (xLabel) s += `<text x="${(W + M.l) / 2}" y="${H - 2}" text-anchor="middle">${esc(xLabel)}</text>`;
  series.forEach((ser) => {
    const pts = xs.map((x, i) => (ser.ys[i] == null ? null : [sx(x), sy(ser.ys[i])]));
    const d = pts.map((p, i) => (p ? `${i && pts[i - 1] ? 'L' : 'M'}${p[0]},${p[1]}` : '')).join(' ');
    s += `<path d="${d}" fill="none" stroke="${ser.color}" stroke-width="2" stroke-linejoin="round"/>`;
    pts.forEach((p, i) => { if (p) s += `<circle cx="${p[0]}" cy="${p[1]}" r="${mark === xs[i] ? 6 : 3}" fill="${mark === xs[i] ? ser.color : 'var(--tile)'}" stroke="${ser.color}" stroke-width="2"><title>${esc(ser.name)} @ ${xs[i]}: ${f(ser.ys[i])}</title></circle>`; });
  });
  if (series.length > 1) s += series.map((ser, i) => `<text x="${M.l + 8 + i * 70}" y="${M.t + 2}" style="fill:${ser.color}">● ${esc(ser.name)}</text>`).join('');
  return frame(W, H, s);
}

function varianceChart(explained, thr, nSel, W = 620, H = 260) {
  const n = explained.length, bw = (W - M.l - M.r) / n;
  const sy = (v) => H - M.b - v * (H - M.b - M.t);
  let s = yAxis([0, .25, .5, .75, 1], sy, W, (t) => (t * 100) + '%');
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${sy(thr)}" y2="${sy(thr)}" stroke="var(--warn)" stroke-dasharray="4 4"/><text x="${W - M.r}" y="${sy(thr) - 5}" text-anchor="end" style="fill:var(--warn)">${Math.round(thr * 100)}% threshold</text>`;
  explained.forEach((e, i) => {
    const x = M.l + i * bw + bw * .2, y = sy(e.ExplainedVariance), on = i < nSel;
    s += `<rect class="bar" x="${x}" y="${y}" width="${bw * .6}" height="${H - M.b - y}" rx="5" fill="${on ? 'var(--accent)' : 'var(--line-2)'}"><title>${e.Component}: ${fmt.pct(e.ExplainedVariance * 100)}</title></rect>`;
    s += `<text x="${x + bw * .3}" y="${H - M.b + 16}" text-anchor="middle">${e.Component}</text>`;
  });
  const pts = explained.map((e, i) => `${M.l + i * bw + bw / 2},${sy(e.Cumulative)}`);
  s += `<polyline points="${pts.join(' ')}" fill="none" stroke="var(--good)" stroke-width="2"/>`;
  explained.forEach((e, i) => { s += `<circle cx="${M.l + i * bw + bw / 2}" cy="${sy(e.Cumulative)}" r="3.5" fill="var(--good)"><title>Cumulative to ${e.Component}: ${fmt.pct(e.Cumulative * 100)}</title></circle>`; });
  return frame(W, H, s);
}

function stackedBars(rows, W = 620) {
  const rh = 34, H = rows.length * rh + 24, lw = 190, max = Math.max(...rows.map((r) => r.parts.reduce((a, p) => a + p.value, 0)));
  let s = '';
  rows.forEach((r, i) => {
    let x = lw; const y = 8 + i * rh;
    s += `<text x="${lw - 10}" y="${y + 17}" text-anchor="end">${esc(r.label)}</text>`;
    r.parts.forEach((p) => {
      const w = (p.value / max) * (W - lw - 10);
      s += `<rect class="bar" x="${x}" y="${y}" width="${Math.max(w - 2, 1)}" height="22" rx="4" fill="${p.color}"><title>${esc(p.title)}</title></rect>`;
      x += w;
    });
  });
  return frame(W, H, s);
}

function heatmap({ rows, cols, vals, W = 620 }) {
  const lw = 230, cw = (W - lw - 10) / cols.length, rh = 34, H = rows.length * rh + 30;
  const color = (v) => {
    if (v == null) return 'var(--tile-2)';
    const t = Math.max(-1, Math.min(1, Math.log2(Math.max(v, 1e-3)) / 2));
    return t >= 0 ? `rgba(245,158,139,${(0.12 + 0.7 * t).toFixed(2)})` : `rgba(110,168,254,${(0.12 + 0.7 * -t).toFixed(2)})`;
  };
  const short = { TotalQuantity: 'Quantity', UniqueProducts: 'Products', AverageOrderValue: 'Avg order' };
  let s = cols.map((c, j) => `<text x="${lw + j * cw + cw / 2}" y="12" text-anchor="middle">${esc(short[c] || c)}</text>`).join('');
  rows.forEach((r, i) => {
    s += `<text x="${lw - 10}" y="${30 + i * rh + 19}" text-anchor="end">${esc(r)}</text>`;
    cols.forEach((c, j) => {
      const v = vals[i][j];
      s += `<rect x="${lw + j * cw + 2}" y="${24 + i * rh + 2}" width="${cw - 4}" height="${rh - 4}" rx="6" fill="${color(v)}"><title>${esc(r)} · ${esc(c)}: ${v == null ? 'n/a' : v.toFixed(2)}× overall median</title></rect>`;
      s += `<text x="${lw + j * cw + cw / 2}" y="${24 + i * rh + rh / 2 + 4}" text-anchor="middle" style="fill:var(--text)">${v == null ? '–' : v.toFixed(2)}</text>`;
    });
  });
  return frame(W, H, s);
}

function boxChart(boxes, W = 620, H = 280) {
  const lo = Math.min(...boxes.map((b) => b.lo)), hi = Math.max(...boxes.map((b) => b.hi));
  const a = Math.floor(Math.log10(Math.max(lo, 0.01))), b = Math.ceil(Math.log10(Math.max(hi, 1)));
  const sy = (v) => H - M.b - ((Math.log10(Math.max(v, 1e-6)) - a) / (b - a || 1)) * (H - M.b - M.t);
  let s = '';
  for (let p = a; p <= b; p++) s += `<line class="grid-l" x1="${M.l}" x2="${W - M.r}" y1="${sy(10 ** p)}" y2="${sy(10 ** p)}"/><text x="${M.l - 8}" y="${sy(10 ** p) + 3}" text-anchor="end">${fmt.compact(10 ** p)}</text>`;
  const bw = (W - M.l - M.r) / boxes.length;
  boxes.forEach((bx, i) => {
    const cx = M.l + i * bw + bw / 2, w = Math.min(bw * .5, 70), col = colorOf(bx.cluster);
    s += `<line x1="${cx}" x2="${cx}" y1="${sy(bx.lo)}" y2="${sy(bx.hi)}" stroke="${col}" stroke-width="1.5"/>`;
    s += `<rect x="${cx - w / 2}" y="${sy(bx.q3)}" width="${w}" height="${Math.max(sy(bx.q1) - sy(bx.q3), 1)}" rx="5" fill="${col}" fill-opacity=".25" stroke="${col}"><title>Cluster ${bx.cluster}: Q1 ${fmt.compact(bx.q1)} · median ${fmt.compact(bx.med)} · Q3 ${fmt.compact(bx.q3)} · ${bx.outliers} outliers beyond whiskers</title></rect>`;
    s += `<line x1="${cx - w / 2}" x2="${cx + w / 2}" y1="${sy(bx.med)}" y2="${sy(bx.med)}" stroke="${col}" stroke-width="2.5"/>`;
    s += `<text x="${cx}" y="${H - M.b + 16}" text-anchor="middle">Cluster ${bx.cluster}</text>`;
  });
  return frame(W, H, s);
}

function histChart(values, { logX = true, bins = 40, W = 620, H = 220 } = {}) {
  const v = values.filter((x) => x != null && (!logX || x > 0)).map((x) => (logX ? Math.log10(x) : x));
  const lo = Math.min(...v), hi = Math.max(...v), bw = (hi - lo) / bins || 1, counts = new Array(bins).fill(0);
  v.forEach((x) => { counts[Math.min(bins - 1, Math.floor((x - lo) / bw))]++; });
  const ticks = niceTicks(0, Math.max(...counts)), top = Math.max(ticks[ticks.length - 1], Math.max(...counts) * 1.08);
  const sy = (c) => H - M.b - (c / top) * (H - M.b - M.t), w = (W - M.l - M.r) / bins;
  let s = yAxis(ticks, sy, W, fmt.compact);
  counts.forEach((c, i) => { s += `<rect class="bar" x="${M.l + i * w + 1}" y="${sy(c)}" width="${w - 2}" height="${H - M.b - sy(c)}" rx="2" fill="var(--accent)" fill-opacity=".8"><title>${fmt.compact(logX ? 10 ** (lo + i * bw) : lo + i * bw)} – ${fmt.compact(logX ? 10 ** (lo + (i + 1) * bw) : lo + (i + 1) * bw)}: ${c} customers</title></rect>`; });
  niceTicks(lo, hi, 5).forEach((t) => { s += `<text x="${M.l + ((t - lo) / (hi - lo || 1)) * (W - M.l - M.r)}" y="${H - M.b + 16}" text-anchor="middle">${fmt.compact(logX ? 10 ** t : t)}</text>`; });
  return frame(W, H, s);
}

function dendrogramChart({ icoord, dcoord }, W = 900, H = 320) {
  const xs = icoord.flat(), ys = dcoord.flat(), x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys);
  const sx = (x) => 10 + ((x - x0) / (x1 - x0 || 1)) * (W - 20), sy = (y) => H - 14 - (y / y1) * (H - 30);
  const paths = icoord.map((ic, i) => `<polyline points="${ic.map((x, j) => `${sx(x)},${sy(dcoord[i][j])}`).join(' ')}" fill="none" stroke="var(--accent)" stroke-opacity=".7" stroke-width="1"/>`).join('');
  return frame(W, H, paths);
}

/* Canvas scatter with nearest-point tooltip. opts: xs, ys, colors[], logY, tip(i) */
function drawScatter(canvas, { xs, ys, colors, logY = false, tip, xLabel = '', yLabel = '' }) {
  const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = Math.round(W * 0.58);
  canvas.width = W * dpr; canvas.height = H * dpr; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr);
  const pad = { l: 52, r: 12, t: 10, b: 30 }, ty = (v) => (logY ? Math.log10(Math.max(v, 1e-6)) : v);
  const yv = ys.map(ty); let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...yv), y1 = Math.max(...yv);
  const px = (x1 - x0) * .04 || 1, py = (y1 - y0) * .04 || 1; x0 -= px; x1 += px; y0 -= py; y1 += py;
  const sx = (x) => pad.l + ((x - x0) / (x1 - x0)) * (W - pad.l - pad.r), sy = (y) => H - pad.b - ((y - y0) / (y1 - y0)) * (H - pad.t - pad.b);
  const css = getComputedStyle(document.documentElement), line = css.getPropertyValue('--line').trim(), muted = css.getPropertyValue('--muted').trim();
  ctx.font = '10.5px sans-serif'; ctx.fillStyle = muted; ctx.strokeStyle = line; ctx.lineWidth = 1;
  const yt = logY ? niceTicks(Math.ceil(y0), Math.floor(y1), 4) : niceTicks(y0, y1, 5);
  yt.forEach((t) => { const y = sy(t); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); ctx.textAlign = 'right'; ctx.fillText(logY ? fmt.compact(10 ** t) : fmt.compact(t), pad.l - 8, y + 3); });
  ctx.textAlign = 'center';
  niceTicks(x0, x1, 6).forEach((t) => ctx.fillText(fmt.compact(t), sx(t), H - pad.b + 16));
  if (xLabel) ctx.fillText(xLabel, (W + pad.l) / 2, H - 2);
  ctx.globalAlpha = .62;
  xs.forEach((x, i) => { ctx.fillStyle = colors[i]; ctx.beginPath(); ctx.arc(sx(x), sy(yv[i]), 2.3, 0, 6.283); ctx.fill(); });
  ctx.globalAlpha = 1;
  const tipEl = document.getElementById('tip');
  canvas.onmousemove = (e) => {
    const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bd = 100;
    for (let i = 0; i < xs.length; i++) { const dx = sx(xs[i]) - mx, dy = sy(yv[i]) - my, d = dx * dx + dy * dy; if (d < bd) { bd = d; best = i; } }
    if (best < 0) { tipEl.style.display = 'none'; return; }
    tipEl.textContent = tip(best); tipEl.style.display = 'block';
    tipEl.style.left = e.clientX + 14 + 'px'; tipEl.style.top = e.clientY + 14 + 'px';
  };
  canvas.onmouseleave = () => { tipEl.style.display = 'none'; };
}

/* ---------- HMM charts ---------- */
const OBS_COLORS = ['#26262e', '#4a6bb5', '#9ec1ff'];
const stateColor = (i) => PALETTE[i % PALETTE.length];

/* Probability matrix: rows -> cols, blue intensity = probability. */
function probMatrix({ rows, cols, vals, W = 620, lw = 150 }) {
  const cw = (W - lw - 10) / cols.length, rh = 38, H = rows.length * rh + 30;
  let s = cols.map((c, j) => `<text x="${lw + j * cw + cw / 2}" y="12" text-anchor="middle">${esc(c)}</text>`).join('');
  rows.forEach((r, i) => {
    s += `<text x="${lw - 10}" y="${30 + i * rh + 21}" text-anchor="end">${esc(r)}</text>`;
    cols.forEach((c, j) => {
      const v = vals[i][j];
      s += `<rect x="${lw + j * cw + 2}" y="${24 + i * rh + 2}" width="${cw - 4}" height="${rh - 4}" rx="7" fill="rgba(138,180,255,${(0.06 + 0.8 * v).toFixed(2)})"><title>${esc(r)} → ${esc(c)}: ${(v * 100).toFixed(1)}%</title></rect>`;
      s += `<text x="${lw + j * cw + cw / 2}" y="${24 + i * rh + rh / 2 + 4}" text-anchor="middle" style="fill:var(--text)">${v.toFixed(2)}</text>`;
    });
  });
  return frame(W, H, s);
}

/* 100% stacked columns. series: [{name, color, values[]}] */
function stackedColumns({ labels, series, W = 620, H = 280 }) {
  const n = labels.length, bw = (W - M.l - M.r) / n, sy = (v) => H - M.b - v * (H - M.b - M.t);
  let s = yAxis([0, .25, .5, .75, 1], sy, W, (t) => t * 100 + '%');
  labels.forEach((lab, t) => {
    const tot = series.reduce((a, x) => a + x.values[t], 0) || 1; let acc = 0;
    series.forEach((x) => {
      const v = x.values[t] / tot, y = sy(acc + v);
      s += `<rect class="bar" x="${M.l + t * bw + bw * .12}" y="${y}" width="${bw * .76}" height="${Math.max(sy(acc) - y, 0)}" fill="${x.color}"><title>${lab} · ${esc(x.name)}: ${x.values[t]} customers (${(v * 100).toFixed(0)}%)</title></rect>`;
      acc += v;
    });
    s += `<text x="${M.l + t * bw + bw / 2}" y="${H - M.b + 16}" text-anchor="middle">${lab.slice(2)}</text>`;
  });
  return frame(W, H, s);
}

/* One customer's sequence: observed symbols, Viterbi path, smoothed posteriors, scaling factors. */
function pathTimeline(d, names, symbols, W = 900) {
  const L = d.obs.length, lw = 120, cw = (W - lw - 10) / L, K = names.length, rh = 34, ph = 70;
  const cell = (x, y, fill, txt, title) => `<rect x="${x + 2}" y="${y}" width="${cw - 4}" height="${rh - 6}" rx="6" fill="${fill}"><title>${esc(title)}</title></rect><text x="${x + cw / 2}" y="${y + 19}" text-anchor="middle" style="fill:var(--text)">${txt}</text>`;
  let s = d.months.map((m, t) => `<text x="${lw + t * cw + cw / 2}" y="12" text-anchor="middle">${m.slice(2)}</text>`).join('');
  s += `<text x="${lw - 10}" y="${38 + 14}" text-anchor="end">Observed</text><text x="${lw - 10}" y="${38 + rh + 14}" text-anchor="end">Viterbi state</text>`;
  d.obs.forEach((o, t) => { s += cell(lw + t * cw, 26, OBS_COLORS[o], ['–', 'L', 'H'][o], `${d.months[t]}: ${symbols[o]}`); });
  d.viterbi.forEach((z, t) => { s += cell(lw + t * cw, 26 + rh, stateColor(z), z, `${d.months[t]}: most likely state ${names[z]}`); });
  const y0 = 26 + 2 * rh + 6;
  s += `<text x="${lw - 10}" y="${y0 + ph / 2}" text-anchor="end">Smoothed P(state)</text>`;
  d.smoothed.forEach((g, t) => {
    let acc = 0;
    g.forEach((p, k) => { const h = p * ph; s += `<rect x="${lw + t * cw + 2}" y="${y0 + ph - acc - h}" width="${cw - 4}" height="${h}" fill="${stateColor(k)}"><title>${d.months[t]} · ${esc(names[k])}: ${(p * 100).toFixed(0)}%</title></rect>`; acc += h; });
  });
  const y1 = y0 + ph + 22;
  s += `<text x="${lw - 10}" y="${y1}" text-anchor="end">Scaling c<tspan dy="3" font-size="8">t</tspan></text>`;
  d.scaling.forEach((c, t) => { s += `<text x="${lw + t * cw + cw / 2}" y="${y1}" text-anchor="middle">${c.toFixed(2)}</text>`; });
  return frame(W, y1 + 8, s);
}
