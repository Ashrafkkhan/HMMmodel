/* Customer segmentation dashboard (vanilla JS). Data comes from /api/results. */
const KM = 'K-Means (full data)';
const FEATS = ['Recency', 'Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'AverageOrderValue'];
const TABS = [['overview', 'Overview'], ['features', 'Customer features'], ['pca', 'PCA'], ['clustering', 'Clustering'],
  ['compare', 'Model comparison'], ['hmm', 'Sequences (HMM)'], ['explorer', 'Customer explorer'], ['downloads', 'Downloads']];

const S = {
  status: null, error: null, name: '', savedExists: false, data: null, tab: 'overview', model: KM, names: {},
  p: { fs: 'RFM', thr: 0.9, k: '', gmm_k: '', cap: 3000, dend: 300, saved: false },
  feat: 'Monetary', logX: true, search: '', explorer: '', inclGmm: true, msg: '',
  hmm: null, hmmK: '', hmmDetail: null, hmmDetailErr: null, hmmId: '', hmmError: null, hmmBusy: false,
};
let hooks = [], custIndex = null;
const $ = (s) => document.querySelector(s);

/* ---------------------------------------------------------------- data */
async function api(path, opts) {
  const r = await fetch(path, opts), j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `Request failed (${r.status})`);
  return j;
}

async function boot() {
  for (;;) {
    try {
      const st = await api('/api/status');
      Object.assign(S, { status: st.status, error: st.error, name: st.name || '', savedExists: st.saved_exists });
      if (st.status === 'ready') { await load(); return; }
      if (st.status === 'error' || st.status === 'idle') { render(); if (st.status === 'error' || !st.saved_exists) return; return; }
    } catch (e) { S.status = 'error'; S.error = e.message; render(); return; }
    render();
    await new Promise((r) => setTimeout(r, 1500));
  }
}

async function load() {
  const p = S.p, q = new URLSearchParams({ fs: p.fs, thr: p.thr, cap: p.cap, dend: p.dend });
  if (p.k) q.set('k', p.k);
  if (p.gmm_k) q.set('gmm_k', p.gmm_k);
  if (p.saved) q.set('saved', '1');
  $('#busy').classList.add('on');
  try {
    S.data = await api('/api/results?' + q);
    const m = S.data.meta;
    Object.assign(S.p, { fs: m.feature_set, thr: m.variance_threshold, k: m.k, gmm_k: m.gmm_k, cap: m.hier_cap });
    if (!S.data.models[S.model]) S.model = KM;
    custIndex = null; if (S.hmmDataKey !== S.name) { S.hmm = null; S.hmmDataKey = S.name; } S.error = null; S.status = 'ready'; loadNames();
  } catch (e) { S.error = e.message; if (!S.data) S.status = 'error'; }
  $('#busy').classList.remove('on');
  render();
}

const nameKey = (model) => ['names', S.p.fs, S.p.thr, S.p.k, S.p.gmm_k, S.p.cap, S.p.saved, model].join('|');
function loadNames() {
  S.names = {};
  for (const m of Object.keys(S.data.models)) {
    try { S.names[m] = JSON.parse(localStorage.getItem(nameKey(m)) || '{}'); } catch { S.names[m] = {}; }
  }
}
function setName(model, cluster, value) {
  (S.names[model] ||= {})[cluster] = value.trim() || undefined;
  try { localStorage.setItem(nameKey(model), JSON.stringify(S.names[model])); } catch { /* storage unavailable */ }
}

/* ------------------------------------------------------------- helpers */
const prof = (model) => S.data.models[model].profile.map((r) => ({ ...r, Segment: S.names[model]?.[r.Cluster] || r.Segment }));
const clusterNames = (model) => Object.fromEntries(prof(model).map((r) => [r.Cluster, r.Segment]));
const legend = (model) => `<div class="legend">${prof(model).map((r) => `<span><i style="background:${colorOf(r.Cluster)}"></i>${r.Cluster}: ${esc(r.Segment)}</span>`).join('')}</div>`;
const tile = (cls, body) => `<section class="tile ${cls}">${body}</section>`;
const head = (title, sub = '', eyebrow = '') => `${eyebrow ? `<div class="eyebrow">${eyebrow}</div>` : ''}<h2>${title}</h2>${sub ? `<p class="sub">${sub}</p>` : ''}`;
const stat = (v, l, sm = false) => `<div class="stat"><div class="v ${sm ? 'sm' : ''}">${v}</div><div class="l">${l}</div></div>`;
const note = (t, kind = '') => `<div class="note ${kind}">${t}</div>`;

function table(cols, rows, { cls = '', best = {} } = {}) {
  const th = cols.map((c) => `<th class="${c.num ? 'num' : ''}">${esc(c.h)}</th>`).join('');
  const body = rows.map((r) => `<tr>${cols.map((c) => {
    const v = c.f ? c.f(r) : r[c.k];
    const isBest = best[c.k] && best[c.k](r);
    return `<td class="${c.num ? 'num' : ''} ${isBest ? 'best' : ''}">${c.raw ? v : esc(v)}</td>`;
  }).join('')}</tr>`).join('');
  return `<div class="scroll ${cls}"><table><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function points(model) {
  const c = S.data.customers, m = S.data.models[model];
  const idx = m.indices || c.CustomerID.map((_, i) => i);
  return { idx, labels: m.labels };
}

function scatterTile(id, model, kind) {
  const c = S.data.customers, { idx, labels } = points(model), cn = clusterNames(model);
  hooks.push(() => {
    const el = document.getElementById(id); if (!el) return;
    const tipf = (i) => `Customer ${c.CustomerID[idx[i]]}\nCluster ${labels[i]}: ${cn[labels[i]]}\nRecency ${c.Recency[idx[i]]} d · £${fmt.compact(c.Monetary[idx[i]])} · ${c.Frequency[idx[i]]} orders`;
    const base = { colors: labels.map(colorOf), tip: tipf };
    if (kind === 'pc') drawScatter(el, { ...base, xs: idx.map((i) => c.PC1[i]), ys: idx.map((i) => c.PC2[i]), xLabel: 'PC1', yLabel: 'PC2' });
    else drawScatter(el, { ...base, xs: idx.map((i) => c.Recency[i]), ys: idx.map((i) => c.Monetary[i]), logY: true, xLabel: 'Recency (days)' });
  });
  return `<canvas class="scatter" id="${id}"></canvas>${legend(model)}`;
}

function quant(arr, q) { const a = [...arr].sort((x, y) => x - y), p = (a.length - 1) * q, f = Math.floor(p); return a[f] + (a[Math.min(f + 1, a.length - 1)] - a[f]) * (p - f); }

/* --------------------------------------------------------------- pages */
function pageOverview(d) {
  const r = d.cleaning.report, steps = d.cleaning.steps, maxRows = r['Original rows'];
  const stepRows = steps.map((s) => ({ ...s, bar: s['Rows remaining'] / maxRows }));
  let h = tile('hero s6 r2', `<div class="eyebrow">Unsupervised learning</div><h1>Who are our customers, judged by how they buy?</h1>
    <p>Online-retail transactions become one row per customer, are scaled and compressed with PCA, then grouped by K-Means (primary), Ward hierarchical clustering and a Gaussian mixture. No true labels exist, so we judge groups by separation and by whether they can be explained. Nothing here predicts churn or future purchases.</p>
    <div class="row" style="margin:16px 0 0"><span class="badge accent">${esc(S.p.saved ? 'Saved artifacts' : S.name)}</span><span class="badge">${esc(d.meta.feature_set)} features</span><span class="badge">k = ${d.meta.k}</span></div>`);
  h += tile('s3', stat(fmt.int(r['Original rows']), 'Original rows')) + tile('s3', stat(fmt.int(r['Final rows']), 'Rows after cleaning'));
  h += tile('s3', stat(fmt.int(r['Removed rows']), 'Rows removed')) + tile('s3', stat(fmt.int(r['Customers']), 'Customers'));
  h += tile('s4', stat(fmt.int(r['Distinct invoices']), 'Distinct invoices')) + tile('s4', stat('£' + fmt.compact(r['Total purchase revenue']), 'Purchase revenue (excludes returns)'));
  h += tile('s4', stat(`${String(r['First purchase']).slice(0, 10)} → ${String(r['Last purchase']).slice(0, 10)}`, 'Date range', true));
  h += tile('s7', head('Cleaning report', 'Sequential counts: a row removed in one step is never counted again.') + table([
    { h: 'Step', k: 'Step' }, { h: 'Removed', k: 'Rows removed', num: true, f: (s) => fmt.int(s['Rows removed']) },
    { h: 'Remaining', k: 'Rows remaining', num: true, f: (s) => fmt.int(s['Rows remaining']) },
    { h: '', raw: true, f: (s) => `<div class="compare-bar" style="width:120px"><b style="width:${(s.bar * 100).toFixed(1)}%"></b></div>` }], stepRows));
  h += tile('s5', head('Data-quality notes') + d.cleaning.warnings.map((w, i) => note(esc(w), i === 0 ? 'warn' : '')).join(''));
  if (d.raw_preview) {
    const cols = Object.keys(d.raw_preview[0]);
    h += tile('s12', head('Raw-data preview', 'First 50 rows, before cleaning.') + table(cols.map((c) => ({ h: c, k: c })), d.raw_preview));
  }
  return h;
}

function pageFeatures(d) {
  const c = d.customers, n = c.CustomerID.length;
  const stats = FEATS.map((f) => { const v = c[f]; return { Feature: f, Min: Math.min(...v), Q1: quant(v, .25), Median: quant(v, .5), Mean: v.reduce((a, b) => a + b, 0) / n, Q3: quant(v, .75), Max: Math.max(...v) }; });
  const used = d.meta.features;
  const q = S.search.toLowerCase(), rows = [];
  for (let i = 0; i < n && rows.length < 60; i++) if (!q || String(c.CustomerID[i]).toLowerCase().includes(q)) rows.push(i);
  let h = tile('s5', head('Feature definitions', `Modelling with <b>${used.join(', ')}</b>. CustomerID is kept for joins only and is never a model feature.`)
    + `<ul class="plain">${Object.entries(d.definitions).map(([k, v]) => `<li><b>${k}</b> – ${esc(v)}${used.includes(k) ? '' : ' <span class="badge">not in current set</span>'}</li>`).join('')}</ul>`);
  h += tile('s7', head('Distribution', 'Original customer values.') + `<div class="row"><select data-act="feat">${FEATS.map((f) => `<option ${f === S.feat ? 'selected' : ''}>${f}</option>`).join('')}</select>
    <label class="ctl"><input type="checkbox" data-act="logx" ${S.logX ? 'checked' : ''}> log x-axis</label></div>` + histChart(c[S.feat], { logX: S.logX }));
  h += tile('s12', head('Summary statistics') + table([{ h: 'Feature', k: 'Feature' }, ...['Min', 'Q1', 'Median', 'Mean', 'Q3', 'Max'].map((k) => ({ h: k, k, num: true, f: (r) => fmt.num(r[k], 1) }))], stats));
  h += tile('s12', head('Customer table', `Showing ${rows.length} of ${fmt.int(n)} customers.`) + `<input class="search" data-act="search" placeholder="Search customer ID…" value="${esc(S.search)}">`
    + table([{ h: 'CustomerID', k: 'id' }, ...FEATS.map((f) => ({ h: f, k: f, num: true, f: (r) => fmt.num(r[f], f === 'Recency' || f === 'Frequency' || f === 'UniqueProducts' ? 0 : 2) }))],
      rows.map((i) => ({ id: c.CustomerID[i], ...Object.fromEntries(FEATS.map((f) => [f, c[f][i]])) }))));
  return h;
}

function pagePCA(d) {
  const m = d.meta, e = d.pca.explained, L = d.pca.loadings;
  const lcell = (v) => `<td class="num" style="background:rgba(${v >= 0 ? '245,158,139' : '110,168,254'},${(Math.abs(v) * .5).toFixed(2)})">${v.toFixed(3)}</td>`;
  let h = tile('s12 flat', `<p class="sub" style="margin:0">Pipeline: <b>log1p → StandardScaler → PCA</b>. The smallest number of components reaching ${Math.round(m.variance_threshold * 100)}% of variance (<b>${m.n_components}</b>) is the input to all three clustering models. The PC1–PC2 plot is a 2-D projection for viewing only.</p>`);
  h += tile('s7', head('Explained variance', `Highlighted bars are the ${m.n_components} retained component(s).`) + varianceChart(e, m.variance_threshold, m.n_components));
  h += tile('s5', head('Variance table') + table([{ h: 'Component', k: 'Component' }, { h: 'Variance', k: 'ExplainedVariance', num: true, f: (r) => fmt.pct(r.ExplainedVariance * 100) }, { h: 'Cumulative', k: 'Cumulative', num: true, f: (r) => fmt.pct(r.Cumulative * 100) }], e));
  h += tile('s6', head('Component loadings', 'Weights of the standardized log-features on each component.')
    + `<div class="scroll"><table><thead><tr><th>Feature</th>${d.pca.pcs.map((p) => `<th class="num">${p}</th>`).join('')}</tr></thead><tbody>${d.pca.features.map((f, i) => `<tr><td>${f}</td>${L[i].map(lcell).join('')}</tr>`).join('')}</tbody></table></div>`);
  h += tile('s6', head('PC1–PC2 projection', `K-Means clusters. Models were trained on ${m.n_components} component(s), not only these two.`) + scatterTile('sc-pca', KM, 'pc'));
  return h;
}

function pageClustering(d) {
  const m = d.meta, km = d.sweeps.kmeans, gm = d.sweeps.gmm, model = S.model, P = prof(model), mm = d.models[model];
  const ks = km.map((r) => r.k);
  let h = tile('s4', head('Elbow plot', 'K-Means inertia (lower = tighter).') + lineChart({ xs: ks, series: [{ name: 'Inertia', ys: km.map((r) => r.Inertia), color: PALETTE[0] }], mark: m.k, xLabel: 'clusters (k)' }));
  h += tile('s4', head('Silhouette', `Recommended k = ${m.k_recommended} (highest).`) + lineChart({ xs: ks, series: [{ name: 'Silhouette', ys: km.map((r) => r.Silhouette), color: PALETTE[2] }], f: (v) => v.toFixed(2), mark: m.k, xLabel: 'clusters (k)' }));
  h += tile('s4', head('GMM AIC / BIC', `Lowest BIC → ${m.gmm_k_bic} components.`) + lineChart({ xs: gm.map((r) => r.k), series: [{ name: 'BIC', ys: gm.map((r) => r.BIC), color: PALETTE[4] }, { name: 'AIC', ys: gm.map((r) => r.AIC), color: PALETTE[3] }], mark: m.gmm_k, xLabel: 'components' }));
  h += tile('s12', head('K-Means candidates', 'Four clusters are not forced; change k in the top bar. Check size balance, not only silhouette.')
    + table([{ h: 'k', k: 'k' }, { h: 'Inertia', k: 'Inertia', num: true, f: (r) => fmt.int(r.Inertia) }, { h: 'Silhouette', k: 'Silhouette', num: true, f: (r) => fmt.num(r.Silhouette, 3) },
      { h: 'Davies-Bouldin', k: 'DB', num: true, f: (r) => fmt.num(r['Davies-Bouldin'], 3) }, { h: 'Calinski-Harabasz', k: 'CH', num: true, f: (r) => fmt.int(r['Calinski-Harabasz']) },
      { h: 'Smallest cluster', k: 's', num: true, f: (r) => fmt.pct(r['Smallest cluster %']) }, { h: 'Sizes', k: 'sz', f: (r) => r['Cluster sizes'] }], km));

  const scope = model === KM ? `All ${fmt.int(m.n_customers)} customers (primary model).` : `${esc(m.scope)}.`;
  h += tile('s12', head('Segment profiles', 'Computed from original (not log-scaled) values. Cluster numbers are arbitrary and can change on retraining; edit names freely.')
    + `<div class="row"><select data-act="model">${Object.keys(d.models).map((k) => `<option ${k === model ? 'selected' : ''}>${esc(k)}</option>`).join('')}</select><span class="badge">${scope}</span></div>`
    + table([{ h: 'ID', k: 'Cluster' }, { h: 'Segment name (editable)', raw: true, f: (r) => `<input class="name-in" data-act="name" data-cluster="${r.Cluster}" value="${esc(r.Segment)}">` },
      { h: 'Customers', k: 'Customers', num: true, f: (r) => fmt.int(r.Customers) }, { h: 'Cust %', k: 'cp', num: true, f: (r) => fmt.pct(r['Customer %']) },
      { h: 'Revenue', k: 'rv', num: true, f: (r) => '£' + fmt.int(r['Total revenue']) }, { h: 'Rev %', k: 'rp', num: true, f: (r) => fmt.pct(r['Revenue %']) },
      { h: 'Mean £', k: 'mm', num: true, f: (r) => fmt.int(r['Mean monetary']) }, { h: 'Median £', k: 'md', num: true, f: (r) => fmt.int(r['Median monetary']) },
      { h: 'Med. recency', k: 'rc', num: true, f: (r) => fmt.int(r['Median recency']) }, { h: 'Med. freq.', k: 'fr', num: true, f: (r) => fmt.num(r['Median frequency'], 1) },
      { h: 'Med. AOV', k: 'ao', num: true, f: (r) => fmt.int(r['Median average order value']) }], P)
    + mm.checks.map((c) => note(esc(c), /No obvious/.test(c) ? 'ok' : 'warn')).join('') + note('Any marketing idea drawn from these segments is a proposed action to test, not a proven outcome.'));
  h += tile('s6', head('Cluster sizes') + barChart({ labels: P.map((r) => `C${r.Cluster}`), values: P.map((r) => r.Customers), colors: P.map((r) => colorOf(r.Cluster)) }) + legend(model));
  h += tile('s6', head('Revenue contribution', 'Share of purchase revenue (returns excluded).') + barChart({ labels: P.map((r) => `C${r.Cluster}`), values: P.map((r) => r['Revenue %']), colors: P.map((r) => colorOf(r.Cluster)), f: (v) => v.toFixed(0) + '%' }));
  const hm = mm.heatmap, hr = Object.keys(hm.rows).sort((a, b) => a - b), cn = clusterNames(model);
  h += tile('s6', head('Feature profile', 'Cluster median ÷ overall median. Blue is below typical, orange above.') + heatmap({ rows: hr.map((c) => `${c}: ${cn[c] || ''}`.slice(0, 34)), cols: hm.features, vals: hr.map((c) => hm.features.map((f) => hm.rows[c][f])) }));
  h += tile('s6', head('Spending distribution', 'Monetary by cluster, log scale. Box = Q1–Q3, line = median.') + boxChart(mm.box));
  h += tile('s6', head('Recency vs monetary', 'Original values; hover a point for details.') + scatterTile('sc-rm', model, 'rm'));
  h += tile('s6', head('PC1–PC2 scatter', `2-D projection only; models were trained on ${m.n_components} component(s).`) + scatterTile('sc-pc2', model, 'pc'));
  h += tile('s12', head('Ward dendrogram', `Sample visualization of ${d.dendrogram.n} customers; it assigns no labels.`)
    + `<div class="row"><label class="ctl">Sample size <select data-act="dend">${[100, 200, 300, 450, 600].map((v) => `<option ${v === S.p.dend ? 'selected' : ''}>${v}</option>`).join('')}</select></label></div>` + dendrogramChart(d.dendrogram));
  return h;
}

function pageCompare(d) {
  const t = d.comparison.table, m = d.meta;
  const bestOf = (key, dir) => { const v = t.map((r) => r[key]).filter((x) => x != null); const b = dir > 0 ? Math.max(...v) : Math.min(...v); return (r) => r[key] === b; };
  const sizes = Object.keys(d.models).filter((k) => k !== KM).map((name) => ({
    label: name.replace(' (', '\n(').split('\n')[0] + ' k=' + prof(name).length,
    parts: prof(name).map((r) => ({ value: r.Customers, color: colorOf(r.Cluster), title: `${name} · cluster ${r.Cluster}: ${r.Customers} customers` })),
  }));
  let h = tile('s12 hero', `<div class="eyebrow">Scope</div><h2>${esc(m.scope)}</h2><p class="sub" style="margin:0">${m.sampled ? 'Sample-based: K-Means, Ward and GMM were all refit on the same reproducible sample, in the same PCA space. Full-data K-Means remains the primary output elsewhere.' : 'Every model uses all customers, in the same PCA space.'}</p>`);
  h += tile('s12', head('Internal metrics', 'Green marks the best value per metric. Metrics are not strictly comparable across different cluster counts. No accuracy/F1 exists: there are no ground-truth labels.')
    + table([{ h: 'Model', k: 'Model' }, { h: 'Customers', k: 'Customers evaluated', num: true, f: (r) => fmt.int(r['Customers evaluated']) }, { h: 'Clusters', k: 'Clusters found', num: true },
      { h: 'Silhouette ↑', k: 'Silhouette', num: true, f: (r) => fmt.num(r.Silhouette, 3) }, { h: 'Davies-Bouldin ↓', k: 'Davies-Bouldin', num: true, f: (r) => fmt.num(r['Davies-Bouldin'], 3) },
      { h: 'Calinski-Harabasz ↑', k: 'Calinski-Harabasz', num: true, f: (r) => fmt.int(r['Calinski-Harabasz']) }, { h: 'Note', k: 'Note' }], t,
      { best: { Silhouette: bestOf('Silhouette', 1), 'Davies-Bouldin': bestOf('Davies-Bouldin', -1), 'Calinski-Harabasz': bestOf('Calinski-Harabasz', 1) } }));
  h += tile('s5', head('GMM information criteria', 'Reported separately; other models have no equivalent.') + table([{ h: 'Model', k: 'Model' }, { h: 'k', k: 'k', num: true }, { h: 'AIC', k: 'AIC', num: true, f: (r) => fmt.int(r.AIC) }, { h: 'BIC', k: 'BIC', num: true, f: (r) => fmt.int(r.BIC) }, { h: 'Converged', k: 'Converged', f: (r) => (r.Converged ? 'yes' : 'no') }], d.comparison.gmm_ic));
  h += tile('s7', head('Cluster-size distribution', 'Each bar is one model, split by cluster.') + stackedBars(sizes));
  h += tile('s12', head('Trade-offs') + `<ul class="plain">
    <li><b>K-Means</b> – compact, similarly sized, easy-to-explain groups; the primary model.</li>
    <li><b>Ward hierarchical</b> – also favours compact groups and shows nesting, but is costly on big data, so it runs on the capped set.</li>
    <li><b>GMM</b> – allows elliptical, overlapping clusters and soft membership; its BIC-selected count can differ and its geometric metrics often look lower because it does not optimise compactness.</li>
    <li>Internal metrics only describe geometric separation. Also weigh size balance and whether each segment can be explained clearly.</li></ul>`);
  return h;
}

function pageExplorer(d) {
  const c = d.customers;
  if (!custIndex) custIndex = new Map(c.CustomerID.map((id, i) => [String(id), i]));
  const id = S.explorer, i = custIndex.get(String(id));
  let h = tile('s5', head('Find a customer', 'Type or pick an existing customer ID.') + `<input class="search" list="ids" data-act="explorer" placeholder="e.g. ${esc(c.CustomerID[0])}" value="${esc(id)}">
    <datalist id="ids">${c.CustomerID.map((x) => `<option value="${esc(x)}">`).join('')}</datalist>`
    + (id && i === undefined ? note('No customer with that ID.', 'warn') : ''));
  if (i === undefined) return h + tile('s7 flat', '<p class="sub" style="margin:0">Select a customer to see their features, segment and how they compare with the segment median.</p>');
  const cl = c.KMeansCluster[i], seg = prof(KM).find((r) => r.Cluster === cl);
  const med = { Recency: seg['Median recency'], Frequency: seg['Median frequency'], Monetary: seg['Median monetary'], TotalQuantity: seg['Median quantity'], UniqueProducts: seg['Median unique products'], AverageOrderValue: seg['Median average order value'] };
  h += tile('s7', `<div class="eyebrow">K-Means segment</div><h2 style="font-size:20px"><i style="display:inline-block;width:10px;height:10px;border-radius:3px;background:${colorOf(cl)};margin-right:8px"></i>Cluster ${cl}: ${esc(seg.Segment)}</h2>
    <p class="sub">${fmt.int(seg.Customers)} customers (${fmt.pct(seg['Customer %'])}) · ${fmt.pct(seg['Revenue %'])} of revenue</p>`
    + (c.GMMCluster[i] != null ? note(`GMM cluster ${c.GMMCluster[i]}: certainty of assignment <b>${fmt.pct(c.GMMProb[i] * 100)}</b> within the fitted mixture. This is the model's membership probability, not the chance that a business label is correct.`)
      : note('Not in the comparison sample, so no GMM probability is available.')));
  h += tile('s12', head('Compared with segment median', 'Bar shows customer ÷ segment median (marker = 1×, capped at 3×).')
    + table([{ h: 'Feature', k: 'f' }, { h: 'Customer', k: 'v', num: true, f: (r) => fmt.num(r.v, 2) }, { h: 'Segment median', k: 'm', num: true, f: (r) => fmt.num(r.m, 2) },
      { h: 'Ratio', k: 'ratio', num: true, f: (r) => (r.m ? r.v / r.m : null) == null ? '–' : fmt.num(r.v / r.m, 2) + '×' },
      { h: '', raw: true, f: (r) => { const x = r.m ? Math.min(r.v / r.m, 3) / 3 * 100 : 0; return `<div class="compare-bar" style="width:160px"><b style="width:${x.toFixed(1)}%"></b><span style="position:absolute;left:33.3%;top:-3px;bottom:-3px;width:1px;background:var(--muted)"></span></div>`; } }],
      FEATS.map((f) => ({ f, v: c[f][i], m: med[f] }))));
  return h;
}


/* ------------------------------------------------------------ HMM tab */
async function loadHmm(k) {
  S.hmmBusy = true; S.hmmError = null;
  $('#busyText').textContent = 'Fitting hidden Markov models… (first run takes about 30 s)'; $('#busy').classList.add('on');
  try { S.hmm = await api('/api/hmm?' + new URLSearchParams(k ? { k } : {})); S.hmmK = S.hmm.meta.k; S.hmmDetail = null; S.hmmId = ''; }
  catch (e) { S.hmmError = e.message; }
  S.hmmBusy = false; $('#busy').classList.remove('on'); $('#busyText').textContent = 'Recomputing…'; render();
}
async function loadHmmCustomer(id) {
  S.hmmId = id; S.hmmDetail = null; S.hmmDetailErr = null;
  if (id) { try { S.hmmDetail = await api('/api/hmm_customer?' + new URLSearchParams({ id, k: S.hmmK })); } catch (e) { S.hmmDetailErr = e.message; } }
  render();
}

function pageHMM(d) {
  if (S.hmmError) return tile('s12', head('Sequences are unavailable', esc(S.hmmError)));
  if (!d) return tile('s12 flat', '<p class="sub" style="margin:0">Loading sequence model…</p>');
  const m = d.meta, names = m.state_names, K = names.length, mod = d.model, syms = m.symbols;
  const short = names.map((n) => n.split(' (')[0]);
  const last = m.months[m.months.length - 1];
  let h = tile('s12 hero', `<div class="eyebrow">Unit 4 · Hidden Markov Models</div><h2 style="font-size:20px">How does a customer's engagement move from month to month?</h2>
    <p class="sub" style="margin:8px 0 0;max-width:90ch">Each customer's history is a sequence of months, from their first purchase to ${last}. Every month is one observed symbol: <b>No purchase</b>, <b>Low spend</b> or <b>High spend</b> (split at £${fmt.int(m.threshold)}, the median positive monthly spend). The HMM assumes a hidden engagement state that follows a Markov chain and produces those symbols. This describes behaviour; it does not identify churn.${m.dropped_partial ? ` The final month (${m.dropped_month}) is only partly observed, so it is excluded.` : ''}</p>
    <div class="row" style="margin:14px 0 0"><span class="badge accent">${fmt.int(m.n_sequences)} sequences</span><span class="badge">${fmt.int(m.n_observations)} customer-months</span><span class="badge">${m.months.length} calendar months</span></div>`);
  h += tile('s7', head('Choosing the number of hidden states', `Lowest BIC is ${m.k_bic} states, but BIC keeps improving as extra states split dormancy patterns. ${m.k} states are shown; change below.`)
    + `<div class="row"><label class="ctl">Hidden states <select data-act="hk">${m.ks.map((k) => `<option value="${k}" ${k === m.k ? 'selected' : ''}>${k}${k === m.k_bic ? ' (lowest BIC)' : ''}</option>`).join('')}</select></label></div>`
    + lineChart({ xs: m.ks, series: [{ name: 'BIC', ys: d.sweep.map((r) => r.BIC), color: PALETTE[4] }, { name: 'AIC', ys: d.sweep.map((r) => r.AIC), color: PALETTE[3] }], mark: m.k, xLabel: 'hidden states', H: 230 }));
  h += tile('s5', head('Candidates') + table([{ h: 'k', k: 'k' }, { h: 'Log-lik.', k: 'LogLik', num: true, f: (r) => fmt.int(r.LogLik) }, { h: 'BIC', k: 'BIC', num: true, f: (r) => fmt.int(r.BIC) },
    { h: 'Params', k: 'Parameters', num: true }, { h: 'EM iters', k: 'EM iterations', num: true }, { h: 'Converged', k: 'Converged', f: (r) => (r.Converged ? 'yes' : 'no') }], d.sweep));
  h += tile('s6', head('Transition matrix A', 'Probability of moving from the row state this month to the column state next month. A strong diagonal means sticky states.')
    + probMatrix({ rows: names, cols: short, vals: mod.A }));
  h += tile('s6', head('Emission matrix B', 'What each hidden state tends to produce. Check these rows to confirm the suggested state names.')
    + probMatrix({ rows: names, cols: syms, vals: mod.B })
    + `<p class="sub" style="margin:12px 0 0">Initial state distribution π: ${mod.pi.map((p, i) => `<span class="badge">${short[i]} ${fmt.pct(p * 100, 0)}</span>`).join(' ')}</p>`);
  h += tile('s6', head('Training curve (Baum-Welch / EM)', 'Log-likelihood after each EM iteration. It never decreases; flattening means convergence.')
    + lineChart({ xs: mod.ll_history.map((_, i) => i + 1), series: [{ name: 'Log-likelihood', ys: mod.ll_history, color: PALETTE[2] }], xLabel: 'EM iteration', H: 240 }));
  h += tile('s6', head('Engagement mix by calendar month', 'Viterbi state of each customer already in the sequence that month. Early months hold only newly-joined customers.')
    + stackedColumns({ labels: m.months, series: names.map((n, i) => ({ name: n, color: stateColor(i), values: d.state_mix.map((r) => r[i]) })) })
    + `<div class="legend">${names.map((n, i) => `<span><i style="background:${stateColor(i)}"></i>${esc(n)}</span>`).join('')}</div>`);

  /* single-customer viewer: forward-backward + Viterbi */
  const cust = d.customers, det = S.hmmDetail;
  let viewer = head('Customer path viewer', 'Pick a customer to see the forward-backward posteriors, the Viterbi path, and the scaling factors that keep the recursion numerically stable.')
    + `<input class="search" list="hid" data-act="hmmid" placeholder="Customer ID, e.g. ${esc(cust.id[0])}" value="${esc(S.hmmId)}"><datalist id="hid">${cust.id.map((x) => `<option value="${esc(x)}">`).join('')}</datalist>`;
  if (S.hmmDetailErr) viewer += note(esc(S.hmmDetailErr), 'warn');
  if (det) {
    viewer += `<div class="row"><span class="badge accent">Customer ${esc(det.id)}</span><span class="badge">log P(sequence) = ${det.log_likelihood.toFixed(2)}</span><span class="badge">Viterbi log P = ${det.viterbi_logprob.toFixed(2)}</span></div>`
      + pathTimeline(det, names, syms)
      + `<div class="legend">${names.map((n, i) => `<span><i style="background:${stateColor(i)}"></i>${i}: ${esc(n)}</span>`).join('')}<span>Observed: – none · L low · H high</span></div>`
      + note('Smoothed probabilities use the whole sequence (forward-backward). Viterbi is the single most likely path, so the two can disagree. Scaling factors c<sub>t</sub> = p(x<sub>t</sub> | x<sub>1..t-1</sub>); their logs sum to the log-likelihood.')
      + `<h2 style="margin-top:16px">Predicted activity next month</h2><div style="max-width:420px">${barChart({ labels: syms, values: det.next_obs.map((p) => p * 100), colors: ['#4a4a58', '#4a6bb5', '#9ec1ff'], f: (v) => v.toFixed(0) + '%', H: 200 })}</div>`;
  }
  h += tile('s12', viewer);

  /* backtest */
  const bt = d.backtest, accs = bt.table.map((x) => x.Accuracy), losses = bt.table.map((x) => x.LogLoss);
  h += tile('s7', head('Predicting the next month: honest backtest', `Each customer's last observed month was hidden and predicted from the earlier months (${fmt.int(bt.n_evaluated)} customers with at least 3 months). All models were trained on the shortened sequences only.`)
    + table([{ h: 'Method', k: 'Method' }, { h: 'Accuracy ↑', k: 'Accuracy', num: true, f: (r) => fmt.pct(r.Accuracy * 100) }, { h: 'Log-loss ↓', k: 'LogLoss', num: true, f: (r) => fmt.num(r.LogLoss, 3) }], bt.table,
      { best: { Accuracy: (r) => r.Accuracy === Math.max(...accs), LogLoss: (r) => r.LogLoss === Math.min(...losses) } })
    + note(`${fmt.pct(bt.actual_share[0] * 100)} of held-out months were "No purchase", so always guessing that already scores about that accuracy. Log-loss, which scores the predicted probabilities, is the fairer comparison. This predicts next-month activity, not churn.`));

  /* link to the K-Means segments */
  const cu = S.data.customers; let cross = '';
  if (cu.CustomerID.length === cust.id.length && String(cu.CustomerID[0]) === cust.id[0] && String(cu.CustomerID[cu.CustomerID.length - 1]) === cust.id[cust.id.length - 1]) {
    const P = prof(KM), at = new Map(P.map((r, i) => [r.Cluster, i])), mat = P.map(() => new Array(K).fill(0));
    cu.KMeansCluster.forEach((c, i) => { if (cust.last_state[i] != null) mat[at.get(c)][cust.last_state[i]]++; });
    cross = `<div class="scroll"><table><thead><tr><th>K-Means segment</th>${short.map((n) => `<th class="num">${esc(n)}</th>`).join('')}</tr></thead><tbody>${P.map((r, i) => {
      const tot = mat[i].reduce((a, b) => a + b, 0) || 1;
      return `<tr><td>${r.Cluster}: ${esc(r.Segment)}</td>${mat[i].map((v) => `<td class="num" style="background:rgba(138,180,255,${(v / tot * 0.55).toFixed(2)})">${fmt.pct(v / tot * 100, 0)}</td>`).join('')}</tr>`;
    }).join('')}</tbody></table></div>`;
  }
  h += tile('s5', head('Latest state by K-Means segment', 'Row share of customers whose most recent month decodes to each hidden state. Links the sequence view to the segmentation.') + (cross || '<p class="sub">Unavailable.</p>'));

  h += tile('s12', head('The algorithms behind this tab') + `<ul class="plain">
    <li><b>Markov model:</b> the next state depends only on the current one, via the transition matrix A.</li>
    <li><b>HMM:</b> states are hidden and emit observable symbols through B; the parameters are π, A and B.</li>
    <li><b>Maximum likelihood (Baum-Welch):</b> EM alternates forward-backward (E-step: expected state and transition counts) with re-estimating π, A and B (M-step).</li>
    <li><b>Forward-backward / sum-product:</b> message passing along the chain computes p(sequence) and the state posteriors without enumerating every path.</li>
    <li><b>Scaling factors:</b> each forward step is renormalised so probabilities do not underflow; the log-likelihood is the sum of the logs of the scale factors.</li>
    <li><b>Viterbi:</b> a max-product version of the same recursion that returns the single most probable state path.</li>
    <li><b>Linear dynamical systems</b> (not implemented): the continuous-state cousin of an HMM, where hidden states are real-valued vectors with linear-Gaussian dynamics and inference uses the Kalman filter and smoother.</li></ul>`);
  return h;
}

/* -------------------------------------------------------------- exports */
function toCSV(rows, cols) {
  const q = (v) => { if (v == null) return ''; v = String(v); return /[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v; };
  return [cols.join(','), ...rows.map((r) => cols.map((k) => q(r[k])).join(','))].join('\n');
}
function download(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' })); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
function exportData(kind) {
  const d = S.data, c = d.customers, n = c.CustomerID.length;
  if (kind === 'assignments') {
    const cn = clusterNames(KM), inCmp = new Set(d.models[Object.keys(d.models).find((k) => k !== KM)].indices);
    const cols = ['CustomerID', ...FEATS, 'KMeansCluster', 'KMeansSegment', ...(S.inclGmm ? ['InComparisonSet', 'GMMCluster', 'GMMMaxMembershipProbability'] : [])];
    const rows = Array.from({ length: n }, (_, i) => ({ ...Object.fromEntries(['CustomerID', ...FEATS, 'KMeansCluster'].map((k) => [k, c[k][i]])), KMeansSegment: cn[c.KMeansCluster[i]],
      InComparisonSet: inCmp.has(i), GMMCluster: c.GMMCluster[i], GMMMaxMembershipProbability: c.GMMProb[i] }));
    return ['customer_assignments.csv', toCSV(rows, cols)];
  }
  if (kind === 'profiles') {
    const rows = Object.keys(d.models).flatMap((m) => prof(m).map((r) => ({ Model: m, ...r })));
    return ['segment_profiles.csv', toCSV(rows, ['Model', ...Object.keys(prof(KM)[0])])];
  }
  if (kind === 'evaluation') {
    const t = d.comparison.table.map((r) => ({ Table: 'Internal metrics', ...r })), g = d.comparison.gmm_ic.map((r) => ({ Table: 'GMM AIC/BIC', ...r }));
    return ['model_evaluation.csv', toCSV([...t, ...g], ['Table', 'Model', 'Scope', 'Customers evaluated', 'Clusters found', 'k', 'Silhouette', 'Davies-Bouldin', 'Calinski-Harabasz', 'AIC', 'BIC', 'Converged', 'Note'])];
  }
  const rep = Object.entries(d.cleaning.report).map(([Step, Value]) => ({ Section: 'Summary', Step, Value }));
  return ['cleaning_summary.csv', toCSV([...d.cleaning.steps.map((s) => ({ Section: 'Cleaning steps', ...s })), ...rep], ['Section', 'Step', 'Rows removed', 'Rows remaining', 'Value'])];
}
function pageDownloads(d) {
  const item = (kind, title, desc) => tile('s6', head(title, desc) + `<button class="btn primary" data-dl="${kind}">Download CSV</button>`);
  return item('assignments', 'Customer features & cluster assignments', 'One row per customer with original features, the K-Means cluster and segment name.')
    + item('profiles', 'Segment profiles', 'Profiles for every model, using your edited segment names.')
    + item('evaluation', 'Model evaluation', 'Internal metrics and GMM AIC/BIC from the comparison.')
    + item('cleaning', 'Cleaning summary', 'Sequential cleaning counts and headline dataset numbers.')
    + tile('s12 flat', `<label class="ctl"><input type="checkbox" data-act="gmmcols" ${S.inclGmm ? 'checked' : ''}> Include GMM cluster and max membership probability (blank for customers outside the comparison sample)</label>`);
}

/* ------------------------------------------------------------ chrome */
function renderControls() {
  const el = $('#controls'), d = S.data;
  const upload = `<button class="btn" data-act="upload">${S.name ? 'Replace dataset' : 'Upload dataset'}</button>`;
  if (!d) { el.innerHTML = upload + (S.savedExists ? '<button class="btn" data-act="saved">Show saved</button>' : ''); return; }
  const m = d.meta, dis = S.p.saved ? 'disabled' : '';
  const opt = (vals, sel, rec, lab) => vals.map((v) => `<option value="${v}" ${v === sel ? 'selected' : ''}>${v}${v === rec ? ` (${lab})` : ''}</option>`).join('');
  el.innerHTML = `
    <div class="seg" title="RFM is the simple default; Extended adds quantity, variety and order value">${['RFM', 'Extended'].map((f) => `<button data-act="fs" data-v="${f}" class="${f === m.feature_set ? 'on' : ''}" ${dis}>${f}</button>`).join('')}</div>
    <label class="ctl">PCA variance <input type="range" min="50" max="99" value="${Math.round(m.variance_threshold * 100)}" data-act="thr" ${dis}><span id="thrv">${Math.round(m.variance_threshold * 100)}%</span></label>
    <label class="ctl">k <select data-act="k" ${dis}>${opt(m.ks, m.k, m.k_recommended, 'best silhouette')}</select></label>
    <label class="ctl">GMM <select data-act="gk" ${dis}>${opt(m.ks, m.gmm_k, m.gmm_k_bic, 'lowest BIC')}</select></label>
    <label class="ctl">Cap <input type="number" min="300" max="20000" step="100" value="${m.hier_cap}" data-act="cap" ${dis}></label>
    ${upload}
    <button class="btn" data-act="save" ${dis}>Save artifacts</button>
    ${S.savedExists ? `<button class="btn ${S.p.saved ? 'primary' : ''}" data-act="saved">${S.p.saved ? 'Showing saved' : 'Show saved'}</button>` : ''}
    ${S.msg ? `<span class="badge">${esc(S.msg)}</span>` : ''}`;
}

function render() {
  hooks = [];
  renderControls();
  const nav = $('#nav'), main = $('#main');
  nav.innerHTML = S.data ? TABS.map(([id, l]) => `<button data-tab="${id}" class="${id === S.tab ? 'on' : ''}">${l}</button>`).join('') : '';
  if (!S.data) {
    if (S.status === 'error') main.innerHTML = `<div class="center"><div class="error-box"><b>Something went wrong</b><p class="sub" style="margin-top:8px">${esc(S.error)}</p><p class="sub">Upload an Online Retail .xlsx/.csv with the required columns (see data/README.md).</p></div></div>`;
    else if (S.status === 'idle') main.innerHTML = '<div class="center"><div><h2>No dataset loaded</h2><p class="sub">Upload the UCI Online Retail file, or place it in data/ and restart.</p></div></div>';
    else main.innerHTML = `<div class="center"><div><div class="spinner"></div>${S.status === 'loading' ? `Reading ${esc(S.name)}… the first Excel read can take a few minutes.` : 'Starting…'}</div></div>`;
    return;
  }
  const pages = { overview: pageOverview, features: pageFeatures, pca: pagePCA, clustering: pageClustering, compare: pageCompare, hmm: () => pageHMM(S.hmm), explorer: pageExplorer, downloads: pageDownloads };
  main.innerHTML = (S.error ? `<div class="tile s12" style="margin-bottom:14px;border-color:#f08a8a55">${esc(S.error)}</div>` : '') + `<div class="grid fade">${pages[S.tab](S.data)}</div>`;
  hooks.forEach((fn) => fn());
}

/* ------------------------------------------------------------ events */
function reload(patch) { Object.assign(S.p, patch); S.msg = ''; load(); }

document.addEventListener('click', async (e) => {
  const t = e.target.closest('[data-tab],[data-act],[data-dl]'); if (!t) return;
  if (t.dataset.tab) { S.tab = t.dataset.tab; render(); window.scrollTo(0, 0); if (S.tab === 'hmm' && !S.hmm && !S.hmmBusy) loadHmm(); }
  else if (t.dataset.dl) { const [n, txt] = exportData(t.dataset.dl); download(n, txt); }
  else if (t.dataset.act === 'fs') reload({ fs: t.dataset.v, k: '', gmm_k: '' });
  else if (t.dataset.act === 'upload') $('#file').click();
  else if (t.dataset.act === 'saved') reload({ saved: !S.p.saved });
  else if (t.dataset.act === 'save') {
    const p = S.p, q = new URLSearchParams({ fs: p.fs, thr: p.thr, cap: p.cap, k: p.k, gmm_k: p.gmm_k });
    try { await api('/api/save?' + q, { method: 'POST' }); S.savedExists = true; S.msg = 'Artifacts saved'; } catch (err) { S.msg = err.message; }
    renderControls();
  }
});

document.addEventListener('change', (e) => {
  const t = e.target, a = t.dataset.act; if (!a) return;
  if (a === 'thr') reload({ thr: +t.value / 100, k: '', gmm_k: '' });
  else if (a === 'k') reload({ k: +t.value });
  else if (a === 'gk') reload({ gmm_k: +t.value });
  else if (a === 'cap') reload({ cap: +t.value });
  else if (a === 'dend') reload({ dend: +t.value });
  else if (a === 'feat') { S.feat = t.value; render(); }
  else if (a === 'logx') { S.logX = t.checked; render(); }
  else if (a === 'gmmcols') S.inclGmm = t.checked;
  else if (a === 'model') { S.model = t.value; render(); }
  else if (a === 'name') { setName(S.model, +t.dataset.cluster, t.value); render(); }
  else if (a === 'hk') loadHmm(+t.value);
  else if (a === 'hmmid') loadHmmCustomer(t.value.trim());
  else if (a === 'explorer') { S.explorer = t.value.trim(); render(); }
});

document.addEventListener('input', (e) => {
  const t = e.target;
  if (t.dataset.act === 'thr') $('#thrv').textContent = t.value + '%';
  if (t.dataset.act === 'search') {
    S.search = t.value; const pos = t.selectionStart; render();
    const el = document.querySelector('[data-act=search]'); if (el) { el.focus(); el.setSelectionRange(pos, pos); }
  }
});

$('#file').addEventListener('change', async (e) => {
  const f = e.target.files[0]; if (!f) return;
  S.data = null; S.status = 'loading'; S.name = f.name; S.p.saved = false; S.p.k = S.p.gmm_k = ''; render();
  try {
    await api('/api/upload?name=' + encodeURIComponent(f.name), { method: 'POST', body: f });
    e.target.value = ''; boot();
  } catch (err) { S.status = 'error'; S.error = err.message; render(); }
});

let rz; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => hooks.forEach((fn) => fn()), 150); });

boot();
