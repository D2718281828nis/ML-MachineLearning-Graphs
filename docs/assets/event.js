/* "When" and "where": graph-free extreme-value alarm vs graph-based source ranking, gated by Δ_G. */
(function () {
  const svg = document.getElementById('evSvg');
  if (!svg) return;
  const W = 460, H = 360, T = 420, BG = 160, DELAY = 15, N = 14;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const st = { seed: 3, graph: "true", gain: 0.9, ev: null, mc: null };

  // true propagation graph (fixed): a 2-D lattice-like sensor layout
  const P = [], TE = [];
  (function () {
    const R = U.rng(42);
    for (let i = 0; i < N; i++) { const r = Math.floor(i / 4.0), c = i % 4 + (r % 2 ? 0.5 : 0); P.push([60 + c * 105 + (R() - 0.5) * 20, 50 + r * 85 + (R() - 0.5) * 20]); }
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) { const d = Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]); if (d < 125) TE.push([i, j]); }
  })();
  const adjOf = (E) => { const a = Array.from({ length: N }, () => []); E.forEach(([u, v]) => { a[u].push(v); a[v].push(u); }); return a; };
  const Atrue = adjOf(TE);
  const wrongEdges = (() => { const R = U.rng(5), E = new Set(), out = []; while (out.length < TE.length) { const a = R.int(N), b = R.int(N); const k = Math.min(a, b) + '-' + Math.max(a, b); if (a !== b && !E.has(k)) { E.add(k); out.push([a, b]); } } return out; })();
  const Awrong = adjOf(wrongEdges), Anone = adjOf([]);
  const observed = () => (st.graph === 'true' ? Atrue : st.graph === 'wrong' ? Awrong : Anone);

  function simulate(seed, gainSpread) {
    const R = U.rng(seed), src = R.int(N), t0 = BG + 60 + R.int(160);
    const dT = U.bfs(Atrue, src);
    const gains = Array.from({ length: N }, () => Math.max(0.15, 1 + gainSpread * (2 * R() - 1)));
    const X = [];
    for (let i = 0; i < N; i++) {
      const row = new Float64Array(T), ti = t0 + dT[i] * DELAY, amp = 6 * Math.exp(-0.1 * dT[i]) * gains[i];
      for (let t = 0; t < T; t++) {
        const noise = R.normal() * (R() < 0.02 ? 2.2 : 1); // occasional heavy-tailed bursts
        row[t] = noise + (t >= ti ? amp * Math.min(1, (t - ti) / 30) : 0);
      }
      X.push(row);
    }
    return { src, t0, X, dT };
  }

  /* Peaks-over-threshold on the node's own background: exponential tail (GPD with ξ = 0). */
  function evtThreshold(row) {
    const bg = Array.from(row.slice(0, BG)).sort((a, b) => a - b), u = bg[Math.floor(0.9 * bg.length)];
    const exc = bg.filter((v) => v > u).map((v) => v - u), sigma = exc.reduce((a, b) => a + b, 0) / Math.max(exc.length, 1);
    const rate = exc.length / bg.length, q = 2e-4;
    return u + sigma * Math.log(rate / q);
  }
  function alarms(X) {
    return X.map((row) => {
      const thr = evtThreshold(row);
      for (let t = BG; t < T - 3; t++) if (row[t] > thr && row[t + 1] > thr && row[t + 2] > thr) return { thr, t };
      return { thr, t: null };
    });
  }

  /* Source ranking: consistency of alarm times with graph distances from a candidate v. Only the edges make it informative. */
  function rankSources(al, adj) {
    const fired = al.map((a, i) => [i, a.t]).filter((p) => p[1] !== null);
    if (!fired.length) return new Array(N).fill(1 / N);
    const res = [];
    for (let v = 0; v < N; v++) {
      const d = U.bfs(adj, v), cap = 6;
      const dd = fired.map(([i]) => Math.min(d[i], cap));
      const tau = fired.reduce((s, [, t], k) => s + t - DELAY * dd[k], 0) / fired.length;
      res.push(fired.reduce((s, [, t], k) => s + (t - tau - DELAY * dd[k]) ** 2, 0));
    }
    const mn = Math.min(...res), temp = DELAY * DELAY * 4;
    const w = res.map((r) => Math.exp(-(r - mn) / temp)), s = w.reduce((a, b) => a + b, 0);
    return w.map((x) => x / s);
  }

  function newEvent() {
    const ev = simulate(st.seed, st.gain), al = alarms(ev.X);
    const fired = al.filter((a) => a.t !== null).map((a) => a.t);
    ev.al = al; ev.that = fired.length ? Math.min(...fired) : null;
    st.ev = ev; render();
  }

  function render() {
    const ev = st.ev, adj = observed(), Pv = rankSources(ev.al, adj), best = Pv.indexOf(Math.max(...Pv));
    svg.innerHTML = '';
    const muted = U.css('--muted'), acc = U.css('--accent'), ink = U.css('--ink'), red = U.css('--red');
    // true propagation edges faint, observed edges solid
    TE.forEach(([u, v]) => U.el('line', { x1: P[u][0], y1: P[u][1], x2: P[v][0], y2: P[v][1], stroke: muted, 'stroke-width': 1, 'stroke-dasharray': '2 4', opacity: 0.5 }, svg));
    adj.forEach((nb, u) => nb.forEach((v) => { if (v > u) U.el('line', { x1: P[u][0], y1: P[u][1], x2: P[v][0], y2: P[v][1], stroke: acc, 'stroke-width': 2, opacity: 0.55 }, svg); }));
    const order = ev.al.map((a, i) => [i, a.t]).filter((p) => p[1] !== null).sort((a, b) => a[1] - b[1]).map((p) => p[0]);
    for (let i = 0; i < N; i++) {
      const g = U.el('g', { transform: `translate(${P[i][0]},${P[i][1]})` }, svg);
      U.el('circle', { r: 10 + 26 * Math.sqrt(Pv[i]), fill: acc, opacity: 0.18 + 0.6 * Pv[i] }, g);
      U.el('circle', { r: 9, fill: U.css('--panel'), stroke: i === ev.src ? red : ink, 'stroke-width': i === ev.src ? 3 : 1.2 }, g);
      U.el('text', { y: 4, 'text-anchor': 'middle', 'font-size': 10, fill: ink, text: String(i) }, g);
      const k = order.indexOf(i);
      if (k >= 0 && k < 3) U.el('text', { x: 12, y: -10, 'font-size': 10, fill: U.css('--amber'), text: `#${k + 1} ${U.t('alarm', 'тревога')}` }, g);
      U.el('text', { y: 24, 'text-anchor': 'middle', 'font-size': 9.5, fill: muted, text: `P=${U.fmt(Pv[i])}` }, g);
    }
    // traces
    drawTraces(ev);
    const whenErr = ev.that === null ? '–' : (ev.that - ev.t0);
    document.getElementById('evOut').innerHTML = `
      <div class="stat-row">
        <div class="stat"><div class="v">${ev.t0}</div><div class="k">${U.t('true onset t₀', 'истинное начало t₀')}</div></div>
        <div class="stat"><div class="v">${ev.that ?? '–'}</div><div class="k">${U.t('alarm t̂ (no graph used) · delay', 'тревога t̂ (граф не используется) · задержка')} ${whenErr}</div></div>
        <div class="stat"><div class="v">${ev.src}</div><div class="k">${U.t('true source s', 'истинный источник s')}</div></div>
        <div class="stat"><div class="v" style="color:${best === ev.src ? U.css('--green') : U.css('--red')}">${best}</div><div class="k">argmax P(ŝ = v), ${st.graph === 'true' ? U.t('correct edges', 'верные рёбра') : st.graph === 'wrong' ? U.t('wrong edges', 'неверные рёбра') : U.t('no edges', 'без рёбер')}</div></div>
      </div>
      <p class="small muted">${U.t('Switch the observed graph: t̂ stays exactly the same, because the alarm is computed per node from its own background. Only P(ŝ = v) changes.', 'Переключите наблюдаемый граф: t̂ не меняется, потому что тревога вычисляется для каждой вершины по её собственному фону. Меняется только P(ŝ = v).')}</p>`;
  }

  function drawTraces(ev) {
    const c = document.getElementById('evTraces'), w = 560, h = 360, ctx = U.setupCanvas(c, w, h);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 26, r: 8, t: 8, b: 20 }, rowH = (h - m.t - m.b) / N, X = (t) => m.l + (t / T) * (w - m.l - m.r);
    ctx.fillStyle = U.css('--code'); ctx.fillRect(X(0), m.t, X(BG) - X(0), h - m.t - m.b);
    ctx.font = '10px system-ui'; ctx.fillStyle = U.css('--muted'); ctx.fillText(U.t('background window (EVT fit)', 'фоновое окно (подгонка EVT)'), X(4), h - 6);
    for (let i = 0; i < N; i++) {
      const y0 = m.t + rowH * (i + 0.72), sc = rowH / 7, row = ev.X[i];
      ctx.fillStyle = U.css('--muted'); ctx.fillText(String(i), 4, y0);
      ctx.strokeStyle = i === ev.src ? U.css('--red') : U.css('--ink'); ctx.globalAlpha = 0.75; ctx.lineWidth = 0.8; ctx.beginPath();
      for (let t = 0; t < T; t++) { const yy = y0 - U.clamp(row[t], -2, 6) * sc; t ? ctx.lineTo(X(t), yy) : ctx.moveTo(X(t), yy); }
      ctx.stroke(); ctx.globalAlpha = 1;
      ctx.strokeStyle = U.css('--amber'); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(BG), y0 - ev.al[i].thr * sc); ctx.lineTo(X(T), y0 - ev.al[i].thr * sc); ctx.stroke(); ctx.setLineDash([]);
      if (ev.al[i].t !== null) { ctx.fillStyle = U.css('--amber'); ctx.beginPath(); ctx.arc(X(ev.al[i].t), y0 - ev.al[i].thr * sc, 3, 0, 7); ctx.fill(); }
    }
    const vline = (t, col, label) => { ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(t), m.t); ctx.lineTo(X(t), h - m.b); ctx.stroke(); ctx.fillStyle = col; ctx.fillText(label, X(t) + 3, m.t + 9); };
    vline(ev.t0, U.css('--red'), 't₀');
    if (ev.that !== null) vline(ev.that, U.css('--accent'), 't̂');
  }

  function monteCarlo() {
    const out = document.getElementById('evMC');
    out.innerHTML = `<p class="muted">${U.t('Running 300 simulated events…', 'Моделируем 300 событий…')}</p>`;
    setTimeout(() => {
      const n = 300, acc = { true: 0, wrong: 0, none: 0 }, delays = [];
      for (let k = 0; k < n; k++) {
        const ev = simulate(1000 + k, st.gain), al = alarms(ev.X), f = al.filter((a) => a.t !== null).map((a) => a.t);
        if (f.length) delays.push(Math.min(...f) - ev.t0);
        for (const g of ['true', 'wrong', 'none']) {
          const Pv = rankSources(al, g === 'true' ? Atrue : g === 'wrong' ? Awrong : Anone);
          if (Pv.indexOf(Math.max(...Pv)) === ev.src) acc[g]++;
        }
      }
      const a = (g) => acc[g] / n, med = delays.sort((x, y) => x - y)[Math.floor(delays.length / 2)];
      const dTrue = a('true') - a('none'), dWrong = a('wrong') - a('none');
      const v = (d) => (d > 0.05 ? `<span class="verdict yes">${U.t('confirmed: “where” may be reported', 'подтверждён: «где» можно выносить как результат')}</span>` : `<span class="verdict no">${U.t('not confirmed: “where” is auxiliary only', 'не подтверждён: «где» — только вспомогательный ответ')}</span>`);
      out.innerHTML = `
        <table class="t">
          <tr><th>${U.t('observed graph', 'наблюдаемый граф')}</th><th>${U.t('source top-1 accuracy', 'точность источника (top-1)')}</th><th>Δ_G ${U.t('vs', 'относительно')} A = ∅</th><th>${U.t('verdict for “where”', 'вывод для «где»')}</th></tr>
          <tr><td>${U.t('correct edges', 'верные рёбра')}</td><td class="n">${U.fmt(a('true'))}</td><td class="n">${dTrue >= 0 ? '+' : ''}${U.fmt(dTrue)}</td><td>${v(dTrue)}</td></tr>
          <tr><td>${U.t('wrong edges (same count)', 'неверные рёбра (столько же)')}</td><td class="n">${U.fmt(a('wrong'))}</td><td class="n">${dWrong >= 0 ? '+' : ''}${U.fmt(dWrong)}</td><td>${v(dWrong)}</td></tr>
          <tr><td>A = ∅ (${U.t('control', 'контроль')})</td><td class="n">${U.fmt(a('none'))}</td><td class="n">0</td><td class="muted">${U.t('reference', 'эталон')}</td></tr>
        </table>
        <p class="small" style="margin-top:8px">${U.t('“When”: median alarm delay', '«Когда»: медианная задержка тревоги')} t̂ − t₀ = <b>${med}</b> ${U.t('samples, alarms in', 'отсчётов, тревога в')} ${delays.length}/${n} ${U.t('events,', 'событиях,')} <b>${U.t('identical for all three graphs', 'одинаково для всех трёх графов')}</b> (${U.t('chance level for “where” is', 'случайный уровень для «где»:')} 1/${N} ≈ ${U.fmt(1 / N)}).</p>`;
    }, 30);
  }

  document.querySelectorAll('#evGraph button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('#evGraph button').forEach((x) => x.classList.remove('on')); b.classList.add('on'); st.graph = b.dataset.g; render();
  }));
  document.getElementById('evNew').addEventListener('click', () => { st.seed = (st.seed * 7919 + 13) % 100000; newEvent(); });
  document.getElementById('evRun').addEventListener('click', monteCarlo);
  document.getElementById('evGain').addEventListener('input', (e) => { st.gain = +e.target.value; document.getElementById('evGainV').textContent = U.fmt(st.gain); newEvent(); });
  U.onTheme.push(() => st.ev && render());
  U.onLang.push(() => { if (st.ev) render(); if (document.querySelector('#evMC table')) monteCarlo(); else document.getElementById('evMC').innerHTML = `<p class="muted small">${U.t('Not run yet.', 'Ещё не запускалось.')}</p>`; });
  newEvent();
})();
