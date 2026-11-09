/* Graph Laplacian and graph Fourier transform / spectral convolution demos. */
(function () {
  if (!document.getElementById('lapSvg')) return;
  const t = U.t;
  const VW = 360, VH = 260;

  /* ---------------- small graphs (12 nodes) ---------------- */
  function makeGraph(name) {
    const n = 12, pos = [], E = [];
    if (name === 'cycle') {
      for (let i = 0; i < n; i++) { const a = -Math.PI / 2 + 2 * Math.PI * i / n; pos.push([180 + 105 * Math.cos(a), 130 + 105 * Math.sin(a)]); E.push([i, (i + 1) % n]); }
    } else if (name === 'path') {
      for (let i = 0; i < n; i++) { pos.push([25 + i * 28.2, 130 + (i % 2 ? -22 : 22)]); if (i) E.push([i - 1, i]); }
    } else if (name === 'comm') {
      for (let c = 0; c < 2; c++) for (let k = 0; k < 6; k++) { const a = 2 * Math.PI * k / 6 + (c ? Math.PI : 0); pos.push([(c ? 265 : 95) + 62 * Math.cos(a), 130 + 62 * Math.sin(a)]); }
      for (let c = 0; c < 2; c++) for (let k = 0; k < 6; k++) { E.push([c * 6 + k, c * 6 + (k + 1) % 6]); if (k < 3) E.push([c * 6 + k, c * 6 + k + 3]); }
      E.push([0, 6]); // the bridge
    } else { // grid 3x4
      for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) { pos.push([60 + c * 80, 50 + r * 80]); const i = r * 4 + c; if (c) E.push([i - 1, i]); if (r) E.push([i - 4, i]); }
    }
    return { n, pos, E };
  }
  const adjM = (G) => { const A = Array.from({ length: G.n }, () => new Array(G.n).fill(0)); G.E.forEach(([a, b]) => { A[a][b] = 1; A[b][a] = 1; }); return A; };
  const deg = (A) => A.map((r) => r.reduce((s, v) => s + v, 0));
  function lap(A, kind) {
    const n = A.length, d = deg(A);
    if (kind === 'L') return A.map((r, i) => r.map((v, j) => (i === j ? d[i] : 0) - v));
    if (kind === 'Lsym') return A.map((r, i) => r.map((v, j) => (i === j ? (d[i] > 0 ? 1 : 0) : 0) - (v && d[i] && d[j] ? v / Math.sqrt(d[i] * d[j]) : 0)));
    // L~ = I - D~^-1/2 (A+I) D~^-1/2 : the Laplacian whose filter is GCN's Â
    const dt = d.map((x) => x + 1);
    return A.map((r, i) => r.map((v, j) => (i === j ? 1 : 0) - (v + (i === j ? 1 : 0)) / Math.sqrt(dt[i] * dt[j])));
  }

  /* Jacobi eigen-decomposition of a symmetric matrix; returns ascending values and column eigenvectors. */
  function eig(M) {
    const n = M.length, a = M.map((r) => r.slice()), V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-16) continue;
        const th = (a[q][q] - a[p][p]) / (2 * a[p][q]), tt = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(tt * tt + 1), s = tt * c;
        for (let k = 0; k < n; k++) { const kp = a[k][p], kq = a[k][q]; a[k][p] = c * kp - s * kq; a[k][q] = s * kp + c * kq; }
        for (let k = 0; k < n; k++) { const pk = a[p][k], qk = a[q][k]; a[p][k] = c * pk - s * qk; a[q][k] = s * pk + c * qk; }
        for (let k = 0; k < n; k++) { const kp = V[k][p], kq = V[k][q]; V[k][p] = c * kp - s * kq; V[k][q] = s * kp + c * kq; }
      }
    }
    const idx = [...Array(n).keys()].sort((x, y) => a[x][x] - a[y][y]);
    const vals = idx.map((i) => (Math.abs(a[i][i]) < 1e-10 ? 0 : a[i][i]));
    const vecs = idx.map((i) => { const v = V.map((r) => r[i]); let m = 0; v.forEach((x, k) => { if (Math.abs(x) > Math.abs(v[m]) + 1e-9) m = k; }); return v[m] < 0 ? v.map((x) => -x) : v; });
    return { vals, vecs }; // vecs[k] = k-th eigenvector (as array over nodes)
  }
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const matVec = (M, x) => M.map((r) => dot(r, x));
  const norm = (x) => Math.sqrt(dot(x, x));
  const maxAbs = (x) => Math.max(1e-9, ...x.map(Math.abs));

  function drawGraph(svg, G, x, opt = {}) {
    svg.setAttribute('viewBox', `0 0 ${VW} ${VH}`); svg.innerHTML = '';
    const sc = opt.scale || maxAbs(x), muted = U.css('--muted'), ink = U.css('--ink');
    G.E.forEach(([a, b]) => {
      const d2 = (x[a] - x[b]) ** 2, w = opt.edgeEnergy ? Math.min(1, d2 / (opt.edgeEnergy)) : 0;
      U.el('line', { x1: G.pos[a][0], y1: G.pos[a][1], x2: G.pos[b][0], y2: G.pos[b][1], stroke: opt.edgeEnergy ? U.css('--red') : muted, 'stroke-width': opt.edgeEnergy ? 1 + 9 * w : 1.6, opacity: opt.edgeEnergy ? 0.2 + 0.75 * w : 0.55 }, svg);
      if (opt.signChange && Math.sign(x[a]) !== Math.sign(x[b]) && Math.abs(x[a]) > 1e-9 && Math.abs(x[b]) > 1e-9) {
        U.el('circle', { cx: (G.pos[a][0] + G.pos[b][0]) / 2, cy: (G.pos[a][1] + G.pos[b][1]) / 2, r: 3.5, fill: U.css('--amber') }, svg);
      }
    });
    G.pos.forEach((p, i) => {
      const g = U.el('g', { transform: `translate(${p[0]},${p[1]})`, style: opt.onClick ? 'cursor:pointer' : '' }, svg);
      if (opt.sel === i) U.el('circle', { r: 19, fill: 'none', stroke: U.css('--accent'), 'stroke-width': 3 }, g);
      U.el('circle', { r: 14, fill: U.valColor(x[i] / sc), stroke: ink, 'stroke-width': 0.8 }, g);
      U.el('text', { y: 4, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#1d2230', text: U.fmt(x[i], opt.digits ?? 2) }, g);
      if (opt.onClick) g.addEventListener('click', () => opt.onClick(i));
    });
  }

  function matrixView(el, M, labels) {
    const n = M.length; el.style.gridTemplateColumns = `18px repeat(${n}, 1fr)`; el.innerHTML = '';
    U.el('div', { class: 'h' }, el); for (let j = 0; j < n; j++) U.el('div', { class: 'h', text: labels ? labels[j] : j }, el);
    const mx = Math.max(...M.flat().map(Math.abs)) || 1;
    for (let i = 0; i < n; i++) {
      U.el('div', { class: 'h', text: i }, el);
      for (let j = 0; j < n; j++) {
        const v = M[i][j], d = U.el('div', { text: Math.abs(v) < 1e-9 ? '' : (Math.abs(v - Math.round(v)) < 1e-9 ? String(Math.round(v)) : v.toFixed(2).replace(/^(-?)0\./, '$1.')) }, el);
        d.style.background = Math.abs(v) < 1e-9 ? 'var(--code)' : U.valColor(v / mx * 0.85);
        d.style.color = '#1d2230';
      }
    }
  }

  function spectrumBars(canvas, vals, opt = {}) {
    const w = 420, h = opt.h || 120, ctx = U.setupCanvas(canvas, w, h);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 30, r: 10, t: 10, b: 22 }, lmax = opt.lmax || Math.max(2, ...vals) * 1.02;
    const X = (l) => m.l + l / lmax * (w - m.l - m.r);
    ctx.strokeStyle = U.css('--line'); ctx.fillStyle = U.css('--muted'); ctx.font = '10px system-ui';
    ctx.beginPath(); ctx.moveTo(m.l, h - m.b); ctx.lineTo(w - m.r, h - m.b); ctx.stroke();
    for (let l = 0; l <= lmax + 1e-9; l += lmax > 4 ? 1 : lmax > 1.6 ? 0.5 : 0.25) { ctx.fillText(l.toFixed(1), X(l) - 7, h - 6); }
    ctx.fillText('λ', w - 12, h - 6);
    const H = h - m.t - m.b;
    if (opt.series) { // spectral coefficients: [{vals:[...], col, offset}]
      const mx = Math.max(1e-9, ...opt.series.flatMap((s) => s.vals.map(Math.abs)));
      opt.series.forEach((s) => s.vals.forEach((v, k) => { ctx.fillStyle = s.col; ctx.globalAlpha = s.alpha ?? 0.85; ctx.fillRect(X(vals[k]) + (s.offset || 0) - 3, h - m.b - Math.abs(v) / mx * H, 6, Math.abs(v) / mx * H); }));
      ctx.globalAlpha = 1;
      if (opt.filter) { // g(λ) curve scaled to [0,1]
        ctx.strokeStyle = U.css('--amber'); ctx.lineWidth = 2; ctx.beginPath();
        for (let p = 0; p <= 100; p++) { const l = lmax * p / 100, g = U.clamp(opt.filter(l), -0.2, 1.2); const yy = h - m.b - g * H; p ? ctx.lineTo(X(l), yy) : ctx.moveTo(X(l), yy); }
        ctx.stroke();
        ctx.fillStyle = U.css('--amber'); ctx.fillText('g(λ)', m.l + 4, m.t + 10);
      }
    } else {
      vals.forEach((l, k) => { ctx.fillStyle = Math.abs(l) < 1e-9 ? U.css('--red') : (k === opt.hi ? U.css('--amber') : U.css('--accent')); ctx.beginPath(); ctx.arc(X(l), h - m.b - 14 - (k % 3) * 10, 5, 0, 7); ctx.fill(); });
      ctx.fillStyle = U.css('--muted');
    }
  }

  /* =================== Section: Laplacian =================== */
  const L1 = { gname: 'comm', G: null, A: null, x: null, mat: 'L', sel: 0, edit: false, first: null };
  function lapReset(keepX) {
    if (!keepX) { L1.G = makeGraph(L1.gname); }
    L1.A = adjM(L1.G);
    if (!keepX || !L1.x) presetSignal('fiedler');
  }
  function presetSignal(kind) {
    const n = L1.G.n, R = U.rng(7 + n + L1.G.E.length);
    if (kind === 'const') L1.x = new Array(n).fill(0.5);
    else if (kind === 'random') L1.x = Array.from({ length: n }, () => +(2 * R() - 1).toFixed(2));
    else if (kind === 'alt') { const e = eig(lap(L1.A, 'L')); L1.x = e.vecs[n - 1].map((v) => v / maxAbs(e.vecs[n - 1])); }
    else { const e = eig(lap(L1.A, 'L')); const k = e.vals.findIndex((v) => v > 1e-9); L1.x = e.vecs[k < 0 ? 1 : k].map((v) => v / maxAbs(e.vecs[k < 0 ? 1 : k])); }
  }
  function lapRender() {
    const { G, A, x } = L1, n = G.n, L = lap(A, 'L'), e = eig(L);
    const quad = dot(x, matVec(L, x)), edgeSum = G.E.reduce((s, [a, b]) => s + (x[a] - x[b]) ** 2, 0);
    const maxEdge = Math.max(1e-9, ...G.E.map(([a, b]) => (x[a] - x[b]) ** 2));
    drawGraph(document.getElementById('lapSvg'), G, x, { edgeEnergy: maxEdge, sel: L1.edit ? L1.first : L1.sel, scale: 1, onClick: (i) => {
      if (L1.edit) {
        if (L1.first === null) L1.first = i; else { const a = L1.first; L1.first = null; if (a !== i) { const k = G.E.findIndex(([u, v]) => (u === a && v === i) || (u === i && v === a)); if (k >= 0) G.E.splice(k, 1); else G.E.push([a, i]); L1.A = adjM(G); } }
      } else L1.sel = i;
      lapRender();
    } });
    const M = L1.mat === 'A' ? A : L1.mat === 'D' ? A.map((r, i) => r.map((_, j) => (i === j ? deg(A)[i] : 0))) : lap(A, L1.mat);
    matrixView(document.getElementById('lapMatrix'), M);
    const zeros = e.vals.filter((v) => Math.abs(v) < 1e-9).length, ncomp = U.components(A.map((r) => r.map((v, j) => (v ? j : -1)).filter((j) => j >= 0))).sizes.length;
    const fied = e.vals.find((v) => v > 1e-9);
    document.getElementById('lapStats').innerHTML = `
      <div class="stat"><div class="v">${U.fmt(quad, 3)}</div><div class="k">xᵀLx</div></div>
      <div class="stat"><div class="v">${U.fmt(edgeSum, 3)}</div><div class="k">Σ<sub>(i,j)∈E</sub> (x<sub>i</sub> − x<sub>j</sub>)²</div></div>
      <div class="stat"><div class="v">${zeros} = ${ncomp}</div><div class="k">${t('zero eigenvalues = connected components', 'нулевых собств. значений = компонент связности')}</div></div>
      <div class="stat"><div class="v">${fied === undefined ? '–' : U.fmt(fied, 3)}</div><div class="k">λ₂ (${t('Fiedler value: how well connected', 'число Фидлера: насколько связен граф')})</div></div>`;
    spectrumBars(document.getElementById('lapSpec'), e.vals, { lmax: Math.max(2, ...e.vals) * 1.05 });
    const sl = document.getElementById('lapVal'); sl.value = x[L1.sel]; document.getElementById('lapValV').textContent = `x${L1.sel} = ${U.fmt(x[L1.sel])}`;
    const names = { A: t('adjacency A', 'смежность A'), D: t('degree D', 'степени D'), L: 'L = D − A', Lsym: 'L_sym = I − D^−½ A D^−½' };
    document.getElementById('lapMatTitle').textContent = names[L1.mat];
  }
  document.getElementById('lapGraph').addEventListener('change', (e) => { L1.gname = e.target.value; lapReset(false); lapRender(); });
  document.querySelectorAll('#lapMat button').forEach((b) => b.addEventListener('click', () => { document.querySelectorAll('#lapMat button').forEach((x) => x.classList.toggle('on', x === b)); L1.mat = b.dataset.m; lapRender(); }));
  document.querySelectorAll('#lapPreset button').forEach((b) => b.addEventListener('click', () => { presetSignal(b.dataset.p); lapRender(); }));
  document.getElementById('lapVal').addEventListener('input', (e) => { L1.x[L1.sel] = +e.target.value; lapRender(); });
  document.getElementById('lapEdit').addEventListener('click', (e) => { L1.edit = !L1.edit; L1.first = null; e.currentTarget.classList.toggle('on', L1.edit); lapRender(); });

  /* =================== Section: graph Fourier & spectral convolution =================== */
  const F = { gname: 'cycle', G: null, A: null, e: null, k: 1, filter: 'gcn', p: 2, seed: 1, clean: null, noisy: null };
  function fReset() {
    F.G = makeGraph(F.gname); F.A = adjM(F.G); F.e = eig(lap(F.A, 'Lt'));
    const n = F.G.n, R = U.rng(F.seed * 31 + 7);
    // clean = mixture of the two smoothest non-constant modes; noisy = clean + white noise
    F.clean = F.e.vecs[1].map((v, i) => 2.2 * v + 1.4 * F.e.vecs[2][i] + 0.3 * F.e.vecs[0][i]);
    F.noisy = F.clean.map((v) => v + 0.35 * R.normal());
    document.getElementById('fK').max = n - 1;
  }
  const filters = {
    gcn: { g: (l, p) => Math.pow(1 - l, p), label: () => `(1 − λ)^${F.p}`, pmin: 1, pmax: 12, step: 1 },
    low: { g: (l, p) => (l <= p ? 1 : 0), label: () => `𝟙[λ ≤ ${U.fmt(F.p)}]`, pmin: 0.05, pmax: 1.5, step: 0.05 },
    heat: { g: (l, p) => Math.exp(-p * l), label: () => `e^(−${U.fmt(F.p, 1)}λ)`, pmin: 0.5, pmax: 12, step: 0.5 },
    high: { g: (l, p) => (l >= p ? 1 : 0), label: () => `𝟙[λ ≥ ${U.fmt(F.p)}]`, pmin: 0.05, pmax: 1.5, step: 0.05 },
  };
  function fRender() {
    const { G, e } = F, n = G.n, k = F.k, u = e.vecs[k];
    // eigenvector browser
    drawGraph(document.getElementById('fEigSvg'), G, u, { signChange: true, digits: 2 });
    const crossings = G.E.filter(([a, b]) => u[a] * u[b] < -1e-12).length;
    document.getElementById('fEigInfo').innerHTML = `<b>u<sub>${k}</sub></b>, λ<sub>${k}</sub> = <b>${U.fmt(e.vals[k], 3)}</b> · ${t('edges with a sign change', 'рёбер со сменой знака')}: <b>${crossings}</b> · u<sub>${k}</sub>ᵀL̃u<sub>${k}</sub> = ${U.fmt(dot(u, matVec(lap(F.A, 'Lt'), u)), 3)}`;
    // line plot of u_k in node order
    const c = document.getElementById('fEigLine'), w = 420, h = 110, ctx = U.setupCanvas(c, w, h);
    ctx.clearRect(0, 0, w, h);
    const X = (i) => 20 + i * (w - 40) / (n - 1), Y = (v) => h / 2 - v / 0.65 * (h / 2 - 10);
    ctx.strokeStyle = U.css('--line'); ctx.beginPath(); ctx.moveTo(10, h / 2); ctx.lineTo(w - 10, h / 2); ctx.stroke();
    ctx.strokeStyle = U.css('--accent'); ctx.lineWidth = 2; ctx.beginPath(); u.forEach((v, i) => (i ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)))); ctx.stroke();
    u.forEach((v, i) => { ctx.fillStyle = U.valColor(v / maxAbs(u)); ctx.beginPath(); ctx.arc(X(i), Y(v), 4, 0, 7); ctx.fill(); });
    ctx.fillStyle = U.css('--muted'); ctx.font = '10px system-ui'; ctx.fillText(t('node index →', 'номер вершины →'), w - 90, h - 4);
    const lm = Math.max(...e.vals) * 1.06;
    spectrumBars(document.getElementById('fEigSpec'), e.vals, { hi: k, lmax: lm, h: 70 });
    document.getElementById('fKv').textContent = k;

    // filtering
    const fl = filters[F.filter], g = (l) => fl.g(l, F.p);
    const xh = e.vecs.map((v) => dot(v, F.noisy)); // GFT: x̂ = Uᵀx
    const yh = xh.map((v, i) => g(e.vals[i]) * v); // ŷ = g(Λ) x̂
    const y = F.noisy.map((_, i) => yh.reduce((s, c2, kk) => s + c2 * e.vecs[kk][i], 0)); // y = U ŷ
    const sc = Math.max(maxAbs(F.noisy), maxAbs(F.clean));
    drawGraph(document.getElementById('fIn'), G, F.noisy, { scale: sc, digits: 1 });
    drawGraph(document.getElementById('fOut'), G, y, { scale: sc, digits: 1 });
    spectrumBars(document.getElementById('fSpec'), e.vals, { lmax: lm, h: 150, filter: g, series: [{ vals: xh, col: U.css('--muted'), offset: -3, alpha: 0.6 }, { vals: yh, col: U.css('--accent'), offset: 3 }] });
    const errIn = norm(F.noisy.map((v, i) => v - F.clean[i])), errOut = norm(y.map((v, i) => v - F.clean[i]));
    let extra = '';
    if (F.filter === 'gcn') { // same result in the vertex domain: Â^K x
      const Ahat = lap(F.A, 'Lt').map((r, i) => r.map((v, j) => (i === j ? 1 : 0) - v));
      let z = F.noisy.slice(); for (let r = 0; r < F.p; r++) z = matVec(Ahat, z);
      const diff = Math.max(...z.map((v, i) => Math.abs(v - y[i])));
      extra = `<br>${t('Check in the vertex domain', 'Проверка в вершинной области')}: Â<sup>${F.p}</sup>x ${t('(K rounds of GCN message passing, no eigenvectors) differs from', '(K раундов передачи сообщений GCN, без собственных векторов) отличается от')} U(1−Λ)<sup>${F.p}</sup>Uᵀx ${t('by at most', 'не более чем на')} <b>${diff.toExponential(1)}</b>.`;
    }
    document.getElementById('fOutInfo').innerHTML = `${t('Filter', 'Фильтр')} g(λ) = <b>${fl.label()}</b> · ${t('distance to the clean signal', 'расстояние до чистого сигнала')}: ${t('noisy', 'зашумлённый')} <b>${U.fmt(errIn)}</b> → ${t('filtered', 'отфильтрованный')} <b>${U.fmt(errOut)}</b>${extra}`;
    const pv = document.getElementById('fP'); pv.min = fl.pmin; pv.max = fl.pmax; pv.step = fl.step; pv.value = F.p;
    document.getElementById('fPv').textContent = F.filter === 'gcn' ? `K = ${F.p}` : F.filter === 'heat' ? `τ = ${U.fmt(F.p, 1)}` : `λ* = ${U.fmt(F.p)}`;
  }
  document.getElementById('fGraph').addEventListener('change', (e) => { F.gname = e.target.value; F.k = Math.min(F.k, 11); fReset(); fRender(); });
  document.getElementById('fK').addEventListener('input', (e) => { F.k = +e.target.value; fRender(); });
  document.querySelectorAll('#fFilter button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('#fFilter button').forEach((x) => x.classList.toggle('on', x === b));
    F.filter = b.dataset.f; F.p = { gcn: 2, low: 0.4, heat: 3, high: 0.8 }[F.filter]; fRender();
  }));
  document.getElementById('fP').addEventListener('input', (e) => { F.p = +e.target.value; fRender(); });
  document.getElementById('fNoise').addEventListener('click', () => { F.seed++; const k = F.k; fReset(); F.k = k; fRender(); });

  lapReset(false); lapRender(); fReset(); fRender();
  U.onTheme.push(() => { lapRender(); fRender(); });
  U.onLang.push(() => { lapRender(); fRender(); });
  window.__spec = { eig, lap, adjM, makeGraph }; // exposed for console checks
})();
