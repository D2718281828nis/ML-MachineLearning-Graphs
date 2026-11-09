/* Lab: the same GNN with and without edges (Δ_G), plus a separate MLP baseline. */
(function () {
  const cv = document.getElementById('labCanvas');
  if (!cv) return;
  const CW = 640, CH = 460;
  const st = { ds: 'sbm', k: 2, K: 2, colour: 'truth', data: null, res: null, busy: false, sweep: null };

  function fromRG(name) {
    const a = window.RG_DATA[name], pts = [], y = [];
    for (let i = 0; i < a.length; i += 3) { pts.push([a[i], a[i + 1]]); y.push(a[i + 2]); }
    return { pts, y };
  }

  function makeSBM(random) {
    const R = U.rng(7), n = 360, y = [], X = [];
    for (let i = 0; i < n; i++) y.push(i < 90 ? 1 : 0);
    // groups: 2 red communities, 5 green communities
    const grp = y.map((v, i) => (v ? (i % 2) : 2 + (i % 5)));
    for (let i = 0; i < n; i++) X.push(Float64Array.from([R.normal() + (y[i] ? 0.45 : -0.15), R.normal() + (y[i] ? 0.3 : 0)]));
    const adj = Array.from({ length: n }, () => new Set());
    let m = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const p = grp[i] === grp[j] ? 0.11 : 0.0025;
      if (R() < p) { adj[i].add(j); adj[j].add(i); m++; }
    }
    if (random) { // keep the edge count, drop all structure
      adj.forEach((s) => s.clear());
      for (let e = 0; e < m; e++) { const a = R.int(n), b = R.int(n); if (a !== b) { adj[a].add(b); adj[b].add(a); } }
    }
    const A = adj.map((s) => Array.from(s));
    // force layout on the *true* community structure for a readable picture
    const L = Array.from({ length: n }, () => [R() * CW, R() * CH]);
    const Atrue = random ? null : A;
    for (let it = 0; it < 120; it++) {
      const F = L.map(() => [0, 0]);
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        let dx = L[i][0] - L[j][0], dy = L[i][1] - L[j][1]; const d2 = dx * dx + dy * dy + 0.01, f = 600 / d2;
        F[i][0] += dx * f; F[i][1] += dy * f; F[j][0] -= dx * f; F[j][1] -= dy * f;
      }
      const E = Atrue || [];
      E.forEach((nb, i) => nb.forEach((j) => { if (j > i) { const dx = L[j][0] - L[i][0], dy = L[j][1] - L[i][1]; F[i][0] += dx * 0.02; F[i][1] += dy * 0.02; F[j][0] -= dx * 0.02; F[j][1] -= dy * 0.02; } }));
      const cool = 1 - it / 130;
      L.forEach((p, i) => { p[0] += U.clamp(F[i][0] + (CW / 2 - p[0]) * 0.01, -10, 10) * cool; p[1] += U.clamp(F[i][1] + (CH / 2 - p[1]) * 0.01, -10, 10) * cool; p[0] = U.clamp(p[0], 12, CW - 12); p[1] = U.clamp(p[1], 12, CH - 12); });
    }
    if (random) { // place randomly-wired graph by the same feature-free layout as structured one would be meaningless: use grid by index
      L.forEach((p, i) => { const R2 = U.rng(i + 3); p[0] = 20 + R2() * (CW - 40); p[1] = 20 + R2() * (CH - 40); });
    }
    return { pts: L, y, X, adj: A, view: L };
  }

  function build() {
    if (st.ds === 'sbm' || st.ds === 'rand') { st.data = makeSBM(st.ds === 'rand'); }
    else {
      const { pts, y } = fromRG(st.ds);
      const g = U.knnGraph(pts, st.k);
      const X = pts.map((_, i) => Float64Array.from([Math.log(g.kdist[i * 6 + 1]), Math.log(g.kdist[i * 6 + 4])]));
      let mx = [Infinity, -Infinity, Infinity, -Infinity];
      pts.forEach(([x, yy]) => { mx = [Math.min(mx[0], x), Math.max(mx[1], x), Math.min(mx[2], yy), Math.max(mx[3], yy)]; });
      const s = Math.min((CW - 20) / (mx[1] - mx[0]), (CH - 20) / (mx[3] - mx[2]));
      const view = pts.map(([x, yy]) => [10 + (x - mx[0]) * s, 10 + (yy - mx[2]) * s]);
      st.data = { pts, y, X, adj: g.adj, view };
    }
    const { adj, y } = st.data;
    let same = 0, tot = 0, rr = 0, rt = 0;
    adj.forEach((nb, i) => nb.forEach((j) => { tot++; if (y[i] === y[j]) same++; if (y[i]) { rt++; if (y[j]) rr++; } }));
    const { comp, sizes } = U.components(adj);
    const p = y.reduce((a, b) => a + b, 0) / y.length, nflag = Math.round(p * y.length);
    const tie = U.rng(99).shuffle(y.map((_, i) => i)), rank = new Int32Array(y.length); tie.forEach((i, t) => { rank[i] = t; });
    const order = y.map((_, i) => i).sort((a, b) => sizes[comp[b]] - sizes[comp[a]] || rank[a] - rank[b]); // random tie-break: node order must not leak the label
    const flag = new Array(y.length).fill(0); order.slice(0, nflag).forEach((i) => { flag[i] = 1; });
    st.stats = { h: same / tot, h1: rt ? rr / rt : NaN, p, ncomp: sizes.length, compF1: U.f1(y, flag, y.map((_, i) => i)), avgdeg: tot / y.length };
    st.res = null; st.sweep = null;
    stats(); draw(); drawSweep();
  }

  function stats() {
    const s = st.stats;
    document.getElementById('labStats').innerHTML = `
      <div class="stat"><div class="v">${y().length}</div><div class="k">${U.t('nodes · red share', 'вершин · доля красных')} p = ${U.fmt(s.p)}</div></div>
      <div class="stat"><div class="v">${U.fmt(s.avgdeg, 1)}</div><div class="k">${U.t('mean degree', 'средняя степень')}</div></div>
      <div class="stat"><div class="v">${U.fmt(s.h1)}</div><div class="k">${U.t("h₁: share of a red node's edges to red (≈ p ⇒ no signal)", 'h₁: доля рёбер красной вершины, ведущих к красным (≈ p ⇒ сигнала нет)')}</div></div>
      <div class="stat"><div class="v">${s.ncomp}</div><div class="k">${U.t('connected components', 'компонент связности')}${st.ds === 'chains' || st.ds === 'circles' ? ` · ${U.t('size-score', 'оценка по размеру')} F1 = ${U.fmt(s.compF1)}` : ''}</div></div>`;
  }
  const y = () => st.data.y;

  function evalModels(K, withMLP) {
    const { X, adj, y: Y } = st.data;
    let Xg = X.map((r) => Float64Array.from(r));
    for (let t = 0; t < K; t++) Xg = U.propagate(adj, Xg);
    const folds = U.folds(Y, 5, 11), out = { g: [], e: [], m: [], pg: [], pe: [], pm: [] };
    const predG = new Array(Y.length).fill(0), predE = new Array(Y.length).fill(0), predM = new Array(Y.length).fill(0);
    folds.forEach((test, f) => {
      const train = folds.filter((_, t) => t !== f).flat();
      const hp = { hidden: 16, lr: 0.03, epochs: 160, seed: 100 + f }; // identical for GNN and GNN with A = ∅
      const zg = U.standardize(Xg, train), ze = U.standardize(X, train);
      const mg = U.trainMLP(zg, Y, train, hp), me = U.trainMLP(ze, Y, train, hp);
      test.forEach((i) => { predG[i] = mg(zg[i]) > 0.5 ? 1 : 0; predE[i] = me(ze[i]) > 0.5 ? 1 : 0; });
      out.g.push(U.f1(Y, predG, test)); out.e.push(U.f1(Y, predE, test));
      if (withMLP) { // separate baseline with its own hyperparameters
        const mm = U.trainMLP(ze, Y, train, { hidden: 32, lr: 0.01, epochs: 260, seed: 900 + f, l2: 1e-3 });
        test.forEach((i) => { predM[i] = mm(ze[i]) > 0.5 ? 1 : 0; });
        out.m.push(U.f1(Y, predM, test));
      }
    });
    out.predG = predG; out.predE = predE; out.predM = predM;
    return out;
  }
  const mean = (a) => a.reduce((x, y2) => x + y2, 0) / a.length;
  const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((x, v) => x + (v - m) ** 2, 0) / Math.max(a.length - 1, 1)); };

  function run() {
    if (st.busy) return; st.busy = true;
    const out = document.getElementById('labResult');
    out.innerHTML = `<p class="muted">${U.t('Training 3 models × 5 folds in your browser…', 'Обучаем 3 модели × 5 фолдов в браузере…')}</p>`;
    setTimeout(() => {
      const r = evalModels(st.K, true); st.res = r; st.busy = false;
      const d = r.g.map((v, i) => v - r.e[i]), dm = mean(d);
      const allPos = d.every((v) => v > 0);
      const verdict = allPos && dm > 0.02 ? `<span class="verdict yes">${U.t('edge contribution confirmed', 'вклад рёбер подтверждён')}</span>`
        : dm < -0.02 ? `<span class="verdict no">${U.t('edges hurt (Δ_G &lt; 0)', 'рёбра мешают (Δ_G &lt; 0)')}</span>` : `<span class="verdict no">${U.t('not confirmed (Δ_G ≈ 0)', 'не подтверждён (Δ_G ≈ 0)')}</span>`;
      out.innerHTML = `
        <table class="t">
          <tr><th>${U.t('model', 'модель')}</th><th>${U.t('what it sees', 'что видит')}</th><th>F1 (${U.t('red', 'красные')}), ${U.t('5 folds', '5 фолдов')}</th></tr>
          <tr><td><b>GNN</b> (K=${st.K})</td><td>Â<sup>K</sup>X → ${U.t('same MLP head', 'та же MLP-голова')}</td><td class="n">${U.fmt(mean(r.g))} ± ${U.fmt(sd(r.g))}</td></tr>
          <tr><td><b>GNN, A = ∅</b></td><td>X → ${U.t('same head, same seeds, folds', 'та же голова, те же фолды и начальные значения генератора')}</td><td class="n">${U.fmt(mean(r.e))} ± ${U.fmt(sd(r.e))}</td></tr>
          <tr><td>MLP</td><td>X, ${U.t('own hyper-parameters', 'свои гиперпараметры')}</td><td class="n">${U.fmt(mean(r.m))} ± ${U.fmt(sd(r.m))}</td></tr>
        </table>
        <p style="margin:10px 0 4px"><b>Δ_G = F1(GNN) − F1(GNN, A=∅) = ${dm >= 0 ? '+' : ''}${U.fmt(dm, 3)}</b> &nbsp;(${U.t('per fold', 'по фолдам')}: ${d.map((v) => (v >= 0 ? '+' : '') + v.toFixed(2)).join(', ')})</p>
        <p style="margin:4px 0">${verdict}</p>
        <p class="small muted">${U.t('The MLP row differs from the A = ∅ row only because the models differ (width, learning rate, epochs, seeds). That gap says nothing about edges, which is why Δ_G uses the A = ∅ control.', 'Строка MLP отличается от строки A = ∅ только потому, что это разные модели (ширина, скорость обучения, эпохи, начальные значения генератора). Эта разница ничего не говорит о рёбрах, поэтому Δ_G считается относительно контроля A = ∅.')}</p>`;
      draw();
    }, 30);
  }

  function sweep() {
    if (st.busy) return; st.busy = true;
    const Ks = [0, 1, 2, 4, 8, 12, 16, 20, 24, 32], pts = [];
    st.sweep = pts;
    const next = (t) => {
      if (t >= Ks.length) { st.busy = false; drawSweep(); return; }
      const r = evalModels(Ks[t], false);
      pts.push([Ks[t], mean(r.g), mean(r.e)]); drawSweep();
      setTimeout(() => next(t + 1), 10);
    };
    setTimeout(() => next(0), 20);
  }

  function draw() {
    const ctx = U.setupCanvas(cv, CW, CH), { view, adj, y: Y } = st.data;
    ctx.clearRect(0, 0, CW, CH);
    ctx.strokeStyle = U.css('--muted'); ctx.globalAlpha = st.ds === 'rand' ? 0.12 : 0.3; ctx.lineWidth = 0.8;
    ctx.beginPath();
    adj.forEach((nb, i) => nb.forEach((j) => { if (j > i) { ctx.moveTo(view[i][0], view[i][1]); ctx.lineTo(view[j][0], view[j][1]); } }));
    ctx.stroke(); ctx.globalAlpha = 1;
    const red = U.css('--red'), green = U.css('--green'), ink = U.css('--ink');
    const pred = st.res ? (st.colour === 'gnn' ? st.res.predG : st.colour === 'empty' ? st.res.predE : st.colour === 'mlp' ? st.res.predM : null) : null;
    const r = Y.length > 1000 ? 2.6 : 4.5;
    view.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7);
      if (pred) {
        const ok = pred[i] === Y[i];
        ctx.fillStyle = Y[i] ? red : green; ctx.globalAlpha = ok ? 0.9 : 0.25; ctx.fill(); ctx.globalAlpha = 1;
        if (!ok) { ctx.strokeStyle = ink; ctx.lineWidth = 1.2; ctx.stroke(); }
      } else { ctx.fillStyle = Y[i] ? red : green; ctx.fill(); }
    });
  }

  function drawSweep() {
    const c = document.getElementById('labSweep'), w = 420, h = 200, ctx = U.setupCanvas(c, w, h);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 34, r: 10, t: 12, b: 26 }, X = (k) => m.l + (k / 32) * (w - m.l - m.r), Y = (v) => h - m.b - v * (h - m.t - m.b);
    ctx.strokeStyle = U.css('--line'); ctx.fillStyle = U.css('--muted'); ctx.font = '11px system-ui'; ctx.lineWidth = 1;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { ctx.beginPath(); ctx.moveTo(m.l, Y(v)); ctx.lineTo(w - m.r, Y(v)); ctx.stroke(); ctx.fillText(v.toFixed(2), 2, Y(v) + 4); });
    [0, 4, 8, 12, 16, 20, 24, 32].forEach((k) => ctx.fillText(String(k), X(k) - 4, h - 8));
    ctx.fillText(U.t('hops K', 'шаги K'), w - 52, h - 8);
    if (!st.sweep || !st.sweep.length) { ctx.fillText(U.t('Press “Sweep K” to plot F1 against the number of hops.', 'Нажмите «Перебрать K», чтобы построить F1 от числа шагов.'), m.l + 8, Y(0.55)); return; }
    const line = (idx, col, dash) => {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.beginPath();
      st.sweep.forEach((p, t) => (t ? ctx.lineTo(X(p[0]), Y(p[idx])) : ctx.moveTo(X(p[0]), Y(p[idx])))); ctx.stroke(); ctx.setLineDash([]);
      st.sweep.forEach((p) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(p[0]), Y(p[idx]), 2.8, 0, 7); ctx.fill(); });
    };
    line(2, U.css('--muted'), [5, 4]); line(1, U.css('--accent'), []);
    if (st.stats.ncomp > 1 && (st.ds === 'chains' || st.ds === 'circles')) {
      ctx.strokeStyle = U.css('--green'); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(m.l, Y(st.stats.compF1)); ctx.lineTo(w - m.r, Y(st.stats.compF1)); ctx.stroke(); ctx.setLineDash([]);
    }
  }

  const desc = () => ({
    sbm: U.t('Homophilous communities: red nodes sit in two red communities, features are weak (heavily overlapping Gaussians). Edges carry the label signal, so averaging over neighbours denoises features.',
      'Гомофильные сообщества: красные вершины образуют два красных сообщества, признаки слабые (сильно перекрывающиеся гауссианы). Рёбра несут сигнал о классе, поэтому усреднение по соседям убирает шум.'),
    rand: U.t('Same nodes, same features, same number of edges, but edges are wired at random. The graph now carries no label information: h₁ ≈ p.',
      'Те же вершины, те же признаки и столько же рёбер, но рёбра проведены случайно. Граф больше не несёт информации о классе: h₁ ≈ p.'),
    chains: U.t('Repository dataset datasets/df_red_green_chains.csv. Every point locally looks the same (two neighbours ~12 px away); only the chain length differs (red ≈ 38, green 9–20). Features are local k-NN distances, so the A = ∅ model cannot see the class.',
      'Датасет репозитория datasets/df_red_green_chains.csv. Локально все точки выглядят одинаково (два соседа на расстоянии ~12 px); различается только длина цепочки (красные ≈ 38, зелёные 9–20). Признаки — локальные расстояния k-NN, поэтому модель с A = ∅ не видит класс.'),
    circles: U.t('Repository dataset datasets/df_red_green_circles.csv: closed loops have no ends, but small green circles bend more sharply, so the 5th-neighbour distance already separates the classes fairly well (the repository reports 0.82 for the best local method). Here the graph adds less, and over a stronger baseline.',
      'Датасет репозитория datasets/df_red_green_circles.csv: у замкнутых окружностей нет концов, но маленькие зелёные окружности изогнуты сильнее, поэтому расстояние до 5-го соседа уже неплохо разделяет классы (в репозитории лучший локальный метод даёт 0,82). Здесь граф добавляет меньше и поверх более сильного базового уровня.'),
  });
  function setDs(ds) {
    st.ds = ds;
    document.querySelectorAll('#labDs button').forEach((b) => b.classList.toggle('on', b.dataset.d === ds));
    document.getElementById('labK').disabled = ds === 'sbm' || ds === 'rand';
    document.getElementById('labDesc').textContent = desc()[ds];
    document.getElementById('labResult').innerHTML = `<p class="muted">${U.t('Press “Train &amp; compare”.', 'Нажмите «Обучить и сравнить».')}</p>`;
    build();
  }
  document.querySelectorAll('#labDs button').forEach((b) => b.addEventListener('click', () => setDs(b.dataset.d)));
  document.getElementById('labK').addEventListener('input', (e) => { st.k = +e.target.value; document.getElementById('labKv').textContent = st.k; build(); document.getElementById('labResult').innerHTML = `<p class="muted">${U.t('Graph rebuilt. Press “Train &amp; compare”.', 'Граф перестроен. Нажмите «Обучить и сравнить».')}</p>`; });
  document.getElementById('labHops').addEventListener('input', (e) => { st.K = +e.target.value; document.getElementById('labHopsV').textContent = st.K; });
  document.getElementById('labRun').addEventListener('click', run);
  document.getElementById('labSweepBtn').addEventListener('click', sweep);
  document.getElementById('labColour').addEventListener('change', (e) => { st.colour = e.target.value; draw(); });
  U.onTheme.push(() => { if (st.data) { draw(); drawSweep(); } });
  U.onLang.push(() => { if (st.data) { stats(); drawSweep(); document.getElementById('labDesc').textContent = desc()[st.ds]; if (st.res) run(); else document.getElementById('labResult').innerHTML = `<p class="muted">${U.t('Press “Train &amp; compare”.', 'Нажмите «Обучить и сравнить».')}</p>`; } });
  setDs('sbm');
})();
