/* Shared helpers: seeded RNG, colours, graph utilities, a tiny MLP, metrics. */
(function () {
  const U = {};

  U.rng = function (seed) {
    let s = (seed >>> 0) || 1;
    const f = () => { s ^= s << 13; s >>>= 0; s ^= s >>> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
    f.normal = () => { let u = 0, v = 0; while (u === 0) u = f(); v = f(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    f.int = (n) => Math.floor(f() * n);
    f.shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = f.int(i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; };
    return f;
  };

  U.css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  U.fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? '–' : Number(x).toFixed(d);
  U.clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  U.$ = (sel, root = document) => root.querySelector(sel);
  U.el = (tag, attrs = {}, parent) => {
    const ns = ['svg', 'g', 'circle', 'line', 'path', 'text', 'rect', 'polygon', 'defs', 'marker'].includes(tag);
    const e = ns ? document.createElementNS('http://www.w3.org/2000/svg', tag) : document.createElement(tag);
    for (const k in attrs) { if (k === 'text') e.textContent = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (parent) parent.appendChild(e);
    return e;
  };

  /* Diverging colour for a value in [-1, 1] (blue - neutral - orange). */
  U.valColor = function (v) {
    const t = U.clamp(v, -1, 1);
    const dark = document.documentElement.dataset.themeResolved === 'dark';
    const mid = dark ? [214, 214, 222] : [240, 238, 232]; // light midpoint in both themes keeps dark node labels readable
    const pos = [232, 128, 32], neg = [52, 102, 214];
    const c = t >= 0 ? pos : neg, a = Math.abs(t);
    const m = mid.map((x, i) => Math.round(x + (c[i] - x) * a));
    return `rgb(${m[0]},${m[1]},${m[2]})`;
  };

  /* Canvas with device pixel ratio handling. Returns ctx with logical size w x h. */
  U.setupCanvas = function (canvas, w, h) {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  };

  /* Symmetric k-NN graph from 2-D points (brute force, fine for ~2k points). */
  U.knnGraph = function (pts, k) {
    const n = pts.length, nb = Array.from({ length: n }, () => new Set());
    const kd = new Float64Array(n * 6);
    for (let i = 0; i < n; i++) {
      const best = [];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = pts[i][0] - pts[j][0], dy = pts[i][1] - pts[j][1], d = dx * dx + dy * dy;
        if (best.length < Math.max(k, 6)) { best.push([d, j]); best.sort((a, b) => a[0] - b[0]); }
        else if (d < best[best.length - 1][0]) { best[best.length - 1] = [d, j]; best.sort((a, b) => a[0] - b[0]); }
      }
      for (let t = 0; t < k; t++) { const j = best[t][1]; nb[i].add(j); nb[j].add(i); }
      for (let t = 0; t < 6; t++) kd[i * 6 + t] = Math.sqrt(best[t][0]);
    }
    return { adj: nb.map((s) => Array.from(s)), kdist: kd };
  };

  U.components = function (adj) {
    const n = adj.length, comp = new Int32Array(n).fill(-1), sizes = [];
    let c = 0;
    for (let s = 0; s < n; s++) {
      if (comp[s] >= 0) continue;
      const st = [s]; comp[s] = c; let sz = 0;
      while (st.length) { const u = st.pop(); sz++; for (const v of adj[u]) if (comp[v] < 0) { comp[v] = c; st.push(v); } }
      sizes.push(sz); c++;
    }
    return { comp, sizes };
  };

  U.bfs = function (adj, s) {
    const d = new Array(adj.length).fill(Infinity); d[s] = 0; const q = [s];
    for (let h = 0; h < q.length; h++) { const u = q[h]; for (const v of adj[u]) if (d[v] === Infinity) { d[v] = d[u] + 1; q.push(v); } }
    return d;
  };

  /* One step of GCN propagation: D^-1/2 (A+I) D^-1/2 X  (X is array of Float64Array rows). */
  U.propagate = function (adj, X) {
    const n = adj.length, deg = adj.map((a) => a.length + 1), out = [];
    for (let i = 0; i < n; i++) {
      const r = new Float64Array(X[i].length);
      const add = (j) => { const w = 1 / Math.sqrt(deg[i] * deg[j]); for (let f = 0; f < r.length; f++) r[f] += w * X[j][f]; };
      add(i); for (const j of adj[i]) add(j);
      out.push(r);
    }
    return out;
  };

  U.standardize = function (X, idx) {
    const d = X[0].length, mu = new Float64Array(d), sd = new Float64Array(d);
    for (const i of idx) for (let f = 0; f < d; f++) mu[f] += X[i][f] / idx.length;
    for (const i of idx) for (let f = 0; f < d; f++) sd[f] += (X[i][f] - mu[f]) ** 2 / idx.length;
    for (let f = 0; f < d; f++) sd[f] = Math.sqrt(sd[f]) || 1;
    return X.map((r) => r.map((v, f) => (v - mu[f]) / sd[f]));
  };

  /* Tiny one-hidden-layer MLP, binary output, class-weighted BCE, full-batch Adam. */
  U.trainMLP = function (X, y, train, opt) {
    const { hidden = 16, lr = 0.03, epochs = 200, seed = 1, l2 = 1e-4 } = opt || {};
    const R = U.rng(seed), d = X[0].length, H = hidden;
    const W1 = Array.from({ length: H }, () => Float64Array.from({ length: d }, () => R.normal() * Math.sqrt(2 / d)));
    const b1 = new Float64Array(H), W2 = Float64Array.from({ length: H }, () => R.normal() * Math.sqrt(1 / H));
    let b2 = 0;
    const pos = train.filter((i) => y[i] === 1).length, neg = train.length - pos;
    const wPos = train.length / (2 * Math.max(pos, 1)), wNeg = train.length / (2 * Math.max(neg, 1));
    const params = [W1, b1, W2], m = new Map(), v = new Map();
    let mb2 = 0, vb2 = 0;
    const b1m = 0.9, b2m = 0.999, eps = 1e-8;
    const h = new Float64Array(H);
    for (let ep = 1; ep <= epochs; ep++) {
      const gW1 = W1.map(() => new Float64Array(d)), gb1 = new Float64Array(H), gW2 = new Float64Array(H);
      let gb2 = 0;
      for (const i of train) {
        const x = X[i];
        for (let k = 0; k < H; k++) { let s = b1[k]; for (let f = 0; f < d; f++) s += W1[k][f] * x[f]; h[k] = Math.tanh(s); }
        let z = b2; for (let k = 0; k < H; k++) z += W2[k] * h[k];
        const p = 1 / (1 + Math.exp(-z)), w = y[i] ? wPos : wNeg, g = w * (p - y[i]) / train.length;
        gb2 += g;
        for (let k = 0; k < H; k++) {
          gW2[k] += g * h[k];
          const gh = g * W2[k] * (1 - h[k] * h[k]);
          gb1[k] += gh; for (let f = 0; f < d; f++) gW1[k][f] += gh * x[f];
        }
      }
      const step = (arr, grad, key) => {
        if (!m.has(key)) { m.set(key, new Float64Array(arr.length)); v.set(key, new Float64Array(arr.length)); }
        const M = m.get(key), V = v.get(key);
        for (let t = 0; t < arr.length; t++) {
          const g = grad[t] + l2 * arr[t];
          M[t] = b1m * M[t] + (1 - b1m) * g; V[t] = b2m * V[t] + (1 - b2m) * g * g;
          arr[t] -= lr * (M[t] / (1 - b1m ** ep)) / (Math.sqrt(V[t] / (1 - b2m ** ep)) + eps);
        }
      };
      W1.forEach((row, k) => step(row, gW1[k], 'w1' + k)); step(b1, gb1, 'b1'); step(W2, gW2, 'w2');
      mb2 = b1m * mb2 + (1 - b1m) * gb2; vb2 = b2m * vb2 + (1 - b2m) * gb2 * gb2;
      b2 -= lr * (mb2 / (1 - b1m ** ep)) / (Math.sqrt(vb2 / (1 - b2m ** ep)) + eps);
    }
    void params;
    return (x) => {
      for (let k = 0; k < H; k++) { let s = b1[k]; for (let f = 0; f < d; f++) s += W1[k][f] * x[f]; h[k] = Math.tanh(s); }
      let z = b2; for (let k = 0; k < H; k++) z += W2[k] * h[k];
      return 1 / (1 + Math.exp(-z));
    };
  };

  U.f1 = function (y, yhat, idx) {
    let tp = 0, fp = 0, fn = 0;
    for (const i of idx) { if (yhat[i] && y[i]) tp++; else if (yhat[i] && !y[i]) fp++; else if (!yhat[i] && y[i]) fn++; }
    const p = tp + fp ? tp / (tp + fp) : 0, r = tp + fn ? tp / (tp + fn) : 0;
    return p + r ? 2 * p * r / (p + r) : 0;
  };

  /* Stratified k folds. */
  U.folds = function (y, k, seed) {
    const R = U.rng(seed), pos = [], neg = [];
    y.forEach((v, i) => (v ? pos : neg).push(i));
    R.shuffle(pos); R.shuffle(neg);
    const f = Array.from({ length: k }, () => []);
    pos.forEach((i, t) => f[t % k].push(i)); neg.forEach((i, t) => f[t % k].push(i));
    return f;
  };

  /* Theme toggle + resolved theme marker for canvas code. */
  U.onTheme = [];
  function resolveTheme() {
    const forced = document.documentElement.dataset.theme;
    const dark = forced ? forced === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.dataset.themeResolved = dark ? 'dark' : 'light';
    U.onTheme.forEach((f) => f());
  }
  U.initTheme = function () {
    try { const t = localStorage.getItem('gnn-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* ignore */ }
    resolveTheme();
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', resolveTheme);
    const b = document.getElementById('themeBtn');
    if (b) b.addEventListener('click', () => {
      const cur = document.documentElement.dataset.themeResolved;
      const next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('gnn-theme', next); } catch (e) { /* ignore */ }
      resolveTheme();
    });
  };

  /* Language: html[lang] toggles .en / .ru blocks in CSS; JS strings use U.t(en, ru). */
  U.onLang = [];
  U.lang = () => (document.documentElement.lang === 'ru' ? 'ru' : 'en');
  U.t = (en, ru) => (U.lang() === 'ru' ? ru : en);
  U.setLang = function (l) {
    document.documentElement.lang = l;
    try { localStorage.setItem('gnn-lang', l); } catch (e) { /* ignore */ }
    document.querySelectorAll('button[data-l]').forEach((b) => b.classList.toggle('on', b.dataset.l === l));
    document.querySelectorAll('[data-en][data-ru]').forEach((e) => { e.textContent = e.dataset[l]; });
    document.title = l === 'ru' ? 'Графовые нейронные сети — интерактивно' : 'Graph Neural Networks, Interactively';
    U.onLang.forEach((f) => f());
  };
  U.initLang = function () {
    let l = null;
    try { l = localStorage.getItem('gnn-lang'); } catch (e) { /* ignore */ }
    const q = new URLSearchParams(location.search).get('lang');
    if (q === 'ru' || q === 'en') l = q;
    if (!l) l = (navigator.language || '').toLowerCase().startsWith('ru') ? 'ru' : 'en';
    document.querySelectorAll('button[data-l]').forEach((b) => b.addEventListener('click', () => U.setLang(b.dataset.l)));
    U.setLang(l);
  };

  window.U = U;
})();
