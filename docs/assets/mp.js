/* Message-passing playground: GCN, GATv2 and the same network with edges removed, with a guided tour. */
(function () {
  const W = 640, H = 380, LAYERS = 3, WGT = 1.5;
  const svg = document.getElementById('mpSvg');
  if (!svg) return;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const t = U.t;

  const init = () => ({
    nodes: [
      { id: 'A', x: 150, y: 200, f: 0.3 }, { id: '1', x: 70, y: 110, f: 0.9 }, { id: '2', x: 170, y: 80, f: 0.8 },
      { id: '3', x: 230, y: 260, f: 0.7 }, { id: 'B', x: 490, y: 200, f: 0.3 }, { id: '5', x: 400, y: 270, f: -0.8 },
      { id: '6', x: 560, y: 110, f: -0.9 }, { id: '7', x: 580, y: 290, f: -0.6 }, { id: '8', x: 320, y: 300, f: 0.0 },
      { id: '9', x: 80, y: 300, f: 0.5 },
    ],
    edges: [[0, 1], [0, 2], [0, 3], [1, 2], [0, 9], [3, 8], [8, 5], [4, 5], [4, 6], [4, 7], [6, 7], [3, 9]],
  });
  let G = init();
  const st = { mode: 'gcn', layer: 0, sel: 0, edit: false, editFirst: null, anim: null, mark: null };
  const IA = 0, IB = 4;

  const nbrs = () => {
    const a = G.nodes.map(() => []);
    if (st.mode === 'none') return a;
    for (const [u, v] of G.edges) { a[u].push(v); a[v].push(u); }
    return a;
  };

  /* GATv2 with fixed, illustrative parameters (in a real model they are learned). */
  const GW = [[1, -1], [-1, 1], [1, 1], [-1, -1]], GA = [-2.5, -2.5, 0.8, 0.0];
  const lrelu = (z) => (z > 0 ? z : 0.2 * z);
  const gatScore = (hi, hj) => { let e = 0; for (let k = 0; k < 4; k++) e += GA[k] * lrelu(GW[k][0] * hi + GW[k][1] * hj); return e; };

  function coeffs(h, adj) {
    const n = h.length, C = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      const N = [i, ...adj[i]];
      if (st.mode === 'gat') {
        const e = N.map((j) => gatScore(h[i], h[j])), mx = Math.max(...e);
        const ex = e.map((v) => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0);
        N.forEach((j, k) => { C[i][j] = ex[k] / s; });
      } else {
        const di = adj[i].length + 1;
        N.forEach((j) => { C[i][j] = 1 / Math.sqrt(di * (adj[j].length + 1)); });
      }
    }
    return C;
  }

  function forward() {
    const adj = nbrs(), Hs = [G.nodes.map((n) => n.f)], Cs = [];
    for (let l = 0; l < LAYERS; l++) {
      const h = Hs[l], C = coeffs(h, adj); Cs.push(C);
      Hs.push(h.map((_, i) => { let s = 0; for (let j = 0; j < h.length; j++) s += C[i][j] * h[j]; return Math.tanh(WGT * s); }));
    }
    return { adj, Hs, Cs };
  }

  function draw(progress) {
    const { adj, Hs, Cs } = forward();
    const h = Hs[st.layer];
    svg.innerHTML = '';
    const ink = U.css('--ink'), muted = U.css('--muted'), line = U.css('--line'), acc = U.css('--accent'), amber = U.css('--amber');

    const rf = new Set([st.sel]);
    if (st.mode !== 'none') { const d = U.bfs(adj, st.sel); d.forEach((v, i) => { if (v <= Math.max(st.layer, 0)) rf.add(i); }); }
    const gEdges = U.el('g', {}, svg);
    const removed = st.mode === 'none';
    for (const [u, v] of G.edges) {
      const a = G.nodes[u], b = G.nodes[v];
      const hl = st.mark === 'nbrs' && !removed && ([u, v].includes(IA) || [u, v].includes(IB));
      U.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: hl ? amber : removed ? line : muted, 'stroke-width': hl ? 3 : 1.5, 'stroke-dasharray': removed ? '4 5' : '', opacity: hl ? 0.9 : removed ? 0.7 : 0.45 }, gEdges);
    }
    if (st.layer < LAYERS && !removed && st.mark !== 'nbrs') {
      const C = Cs[st.layer];
      for (const j of adj[st.sel]) {
        const a = G.nodes[j], b = G.nodes[st.sel];
        U.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: acc, 'stroke-width': 1 + 12 * C[st.sel][j], 'stroke-linecap': 'round', opacity: 0.55 }, gEdges);
      }
    }
    if (progress !== undefined && st.layer < LAYERS) {
      const C = Cs[st.layer];
      for (let i = 0; i < G.nodes.length; i++) {
        for (const j of adj[i]) {
          const a = G.nodes[j], b = G.nodes[i];
          U.el('circle', { cx: a.x + (b.x - a.x) * progress, cy: a.y + (b.y - a.y) * progress, r: 2.5 + 7 * C[i][j], fill: U.valColor(h[j]), stroke: ink, 'stroke-width': 0.6, opacity: 0.95 }, gEdges);
        }
        if (removed) { const b = G.nodes[i]; U.el('circle', { cx: b.x, cy: b.y, r: 25 + 8 * Math.sin(Math.PI * progress), fill: 'none', stroke: acc, 'stroke-width': 1.2, opacity: 0.6 * (1 - progress) }, gEdges); }
      }
    }
    G.nodes.forEach((n, i) => {
      const g = U.el('g', { transform: `translate(${n.x},${n.y})`, style: 'cursor:pointer' }, svg);
      const twin = i === IA || i === IB;
      if (rf.has(i) && i !== st.sel) U.el('circle', { r: 27, fill: 'none', stroke: acc, 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, g);
      if (i === st.sel) U.el('circle', { r: 28, fill: 'none', stroke: acc, 'stroke-width': 3 }, g);
      if (st.mark && twin) U.el('circle', { r: 32, fill: 'none', stroke: amber, 'stroke-width': 2.5 }, g);
      if (st.edit && st.editFirst === i) U.el('circle', { r: 28, fill: 'none', stroke: amber, 'stroke-width': 3 }, g);
      U.el('circle', { r: 21, fill: U.valColor(h[i]), stroke: twin ? ink : muted, 'stroke-width': twin ? 2.2 : 1 }, g);
      U.el('text', { y: -1, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: '#1d2230', text: n.id }, g);
      U.el('text', { y: 12, 'text-anchor': 'middle', 'font-size': 10, fill: '#1d2230', text: U.fmt(h[i]) }, g);
      g.addEventListener('click', () => onNode(i));
    });
    const modeName = st.mode === 'gcn' ? 'GCN' : st.mode === 'gat' ? 'GATv2' : t('edges removed (A = 0)', 'рёбра удалены (A = 0)');
    U.el('text', { x: 12, y: 22, fill: muted, 'font-size': 13, text: `${t('layer', 'слой')} ${st.layer} / ${LAYERS} · ${modeName}` }, svg);
    if (progress === undefined) panels(adj, Hs, Cs);
  }

  function panels(adj, Hs, Cs) {
    const i = st.sel, n = G.nodes[i];
    const calc = document.getElementById('mpCalc');
    if (st.layer >= LAYERS) {
      calc.textContent = t(`Node ${n.id}: final representation after ${LAYERS} layers\nh_${n.id}^(${LAYERS}) = ${U.fmt(Hs[LAYERS][i], 3)}\n\nPress "Reset" to go back to layer 0.`,
        `Вершина ${n.id}: итоговое представление после ${LAYERS} слоёв\nh_${n.id}^(${LAYERS}) = ${U.fmt(Hs[LAYERS][i], 3)}\n\nНажмите «Сброс», чтобы вернуться к слою 0.`);
    } else {
      const C = Cs[st.layer], h = Hs[st.layer], N = [i, ...adj[i]];
      const name = st.mode === 'gat' ? 'α' : 'c';
      let s = 0, out = t(`Node ${n.id}, layer ${st.layer} → ${st.layer + 1}\n`, `Вершина ${n.id}, слой ${st.layer} → ${st.layer + 1}\n`);
      if (st.mode === 'none') out += t('N(i) = {i}: no edges, no messages.\n', 'N(i) = {i}: рёбер нет, сообщений нет.\n');
      else out += `N(${n.id}) ∪ {${n.id}} = {${N.map((j) => G.nodes[j].id).join(', ')}}\n`;
      out += st.mode === 'gat' ? 'α_ij = softmax_j( aᵀ LeakyReLU(W[h_i ‖ h_j]) )\n' : st.mode === 'gcn' ? t('c_ij = 1 / √(d̃_i d̃_j),  d̃ = degree + 1\n', 'c_ij = 1 / √(d̃_i d̃_j),  d̃ = степень + 1\n') : t('c_ii = 1 (Ã = I, D̃ = I)\n', 'c_ii = 1 (Ã = I, D̃ = I)\n');
      for (const j of N) {
        s += C[i][j] * h[j];
        out += `  ${name}_${n.id},${G.nodes[j].id} = ${U.fmt(C[i][j], 3)} × h_${G.nodes[j].id} (${U.fmt(h[j])}) = ${U.fmt(C[i][j] * h[j], 3)}\n`;
      }
      out += `Σ = ${U.fmt(s, 3)}\nh_${n.id}^(${st.layer + 1}) = tanh(${WGT} · Σ) = ${U.fmt(Math.tanh(WGT * s), 3)}`;
      calc.textContent = out;
    }
    const M = document.getElementById('mpMatrix'), C = Cs[Math.min(st.layer, LAYERS - 1)], n2 = G.nodes.length;
    M.style.gridTemplateColumns = `18px repeat(${n2}, 1fr)`;
    M.innerHTML = '';
    U.el('div', { class: 'h' }, M);
    G.nodes.forEach((nd) => U.el('div', { class: 'h', text: nd.id }, M));
    for (let r = 0; r < n2; r++) {
      U.el('div', { class: 'h', text: G.nodes[r].id }, M);
      for (let c = 0; c < n2; c++) {
        const v = C[r][c];
        const d = U.el('div', { text: v ? v.toFixed(2).replace(/^0/, '') : '' }, M);
        d.style.background = v ? `color-mix(in srgb, var(--accent) ${Math.round(15 + 70 * v)}%, transparent)` : 'var(--code)';
        if (r === st.sel) d.style.outline = '1px solid var(--accent)';
      }
    }
    const L = Math.min(st.layer, LAYERS - 1);
    document.getElementById('mpMatrixTitle').textContent = st.mode === 'gat'
      ? t(`Attention matrix α (layer ${L}): rows sum to 1, depend on features`, `Матрица внимания α (слой ${L}): строки в сумме дают 1, зависят от признаков`)
      : st.mode === 'gcn' ? t('Normalised adjacency Â = D̃^−½ (A+I) D̃^−½: fixed by the graph', 'Нормированная матрица смежности Â = D̃^−½ (A+I) D̃^−½: задана графом')
        : t('A = 0 ⇒ Ã = I: every node only talks to itself', 'A = 0 ⇒ Ã = I: каждая вершина «разговаривает» только сама с собой');
    const T = document.getElementById('mpTwins');
    let rows = `<tr><th>${t('layer', 'слой')}</th><th>h_A</th><th>h_B</th><th>A vs B</th></tr>`;
    for (let l = 0; l <= LAYERS; l++) {
      const a = Hs[l][IA], b = Hs[l][IB], same = Math.abs(a - b) < 1e-9;
      rows += `<tr><td>${l}</td><td class="n">${U.fmt(a, 3)}</td><td class="n">${U.fmt(b, 3)}</td><td>${same ? `<span class="verdict no">${t('identical', 'одинаковы')}</span>` : `<span class="verdict yes">${t('different', 'различаются')}</span>`}</td></tr>`;
    }
    T.innerHTML = rows;
    const slider = document.getElementById('mpFeat');
    slider.value = n.f; document.getElementById('mpFeatV').textContent = `${n.id}: x = ${U.fmt(n.f)}`;
  }

  function onNode(i) {
    if (st.edit) {
      if (st.editFirst === null) { st.editFirst = i; }
      else {
        const a = st.editFirst; st.editFirst = null;
        if (a !== i) {
          const k = G.edges.findIndex(([u, v]) => (u === a && v === i) || (u === i && v === a));
          if (k >= 0) G.edges.splice(k, 1); else G.edges.push([a, i]);
        }
      }
    } else st.sel = i;
    draw();
  }

  function step(done) {
    if (st.anim || st.layer >= LAYERS) { if (done) done(); return; }
    const t0 = performance.now(), dur = 900;
    const tick = (now) => {
      const p = Math.min(1, (now - t0) / dur);
      draw(p);
      if (p < 1) st.anim = setTimeout(() => tick(performance.now()), 16);
      else { st.anim = null; st.layer++; draw(); if (done) done(); }
    };
    st.anim = setTimeout(() => tick(performance.now()), 16);
  }
  const runTo = (L, done) => { if (st.layer >= L) { if (done) done(); return; } step(() => setTimeout(() => runTo(L, done), 150)); };

  function setMode(m) {
    st.mode = m;
    document.querySelectorAll('#mpMode button').forEach((x) => x.classList.toggle('on', x.dataset.m === m));
  }
  document.querySelectorAll('#mpMode button').forEach((b) => b.addEventListener('click', () => { setMode(b.dataset.m); draw(); }));
  document.getElementById('mpStep').addEventListener('click', () => step());
  document.getElementById('mpReset').addEventListener('click', () => { st.layer = 0; draw(); });
  document.getElementById('mpRun').addEventListener('click', () => { st.layer = 0; draw(); setTimeout(() => runTo(LAYERS), 150); });
  document.getElementById('mpEdit').addEventListener('click', (e) => {
    st.edit = !st.edit; st.editFirst = null; e.currentTarget.classList.toggle('on', st.edit); draw();
  });
  document.getElementById('mpRestore').addEventListener('click', () => { G = init(); st.layer = 0; draw(); });
  document.getElementById('mpFeat').addEventListener('input', (e) => { G.nodes[st.sel].f = +e.target.value; draw(); });

  /* ---------------- guided tour ---------------- */
  const val = () => forward().Hs;
  const tour = [
    {
      set() { G = init(); setMode('gcn'); st.layer = 0; st.sel = IA; st.mark = 'twins'; },
      en: () => 'Each circle is a <b>node</b>; the number inside is its <b>feature</b> h (think of one sensor reading). Colour shows the value: <span style="color:#e88020">orange = positive</span>, <span style="color:#3466d6">blue = negative</span>. Look at the two nodes ringed in yellow, <b>A</b> and <b>B</b>: both have exactly the same value, <b>0.30</b>.',
      ru: () => 'Каждый кружок — это <b>вершина</b>; число внутри — её <b>признак</b> h (например, показание одного датчика). Цвет показывает значение: <span style="color:#e88020">оранжевый — положительное</span>, <span style="color:#3466d6">синий — отрицательное</span>. Посмотрите на две вершины в жёлтых кольцах, <b>A</b> и <b>B</b>: у обеих одно и то же значение, <b>0,30</b>.',
    },
    {
      set() { st.mark = 'nbrs'; },
      en: () => 'But their <b>surroundings differ</b> (yellow edges). A is connected to 1, 2, 3, 9 — all <b>positive</b>. B is connected to 5, 6, 7 — all <b>negative</b>. Question: can a model tell A from B? A model that looks only at the node itself (an <b>MLP</b>) sees 0.30 and 0.30 — it <b>cannot</b>. A and B are a deliberately built test of exactly this.',
      ru: () => 'Но их <b>окружение разное</b> (жёлтые рёбра). A связана с 1, 2, 3, 9 — все <b>положительные</b>. B связана с 5, 6, 7 — все <b>отрицательные</b>. Вопрос: может ли модель отличить A от B? Модель, которая смотрит только на саму вершину (<b>MLP</b>), видит 0,30 и 0,30 — она <b>не может</b>. Пара A и B построена специально, чтобы проверить именно это.',
    },
    {
      set() { st.mark = 'twins'; st.sel = IA; },
      en: () => '<b>Message passing, part 1 — collect.</b> Every node receives the current values of its neighbours, plus its own. The blue strokes into A show how much each message counts (c<sub>ij</sub>). In GCN c<sub>ij</sub> = 1/√(d̃<sub>i</sub>d̃<sub>j</sub>): messages from busy nodes count less. The exact numbers are in “Computation for the selected node”.',
      ru: () => '<b>Передача сообщений, часть 1 — сбор.</b> Каждая вершина получает текущие значения соседей и своё собственное. Синие штрихи, входящие в A, показывают вес каждого сообщения (c<sub>ij</sub>). В GCN c<sub>ij</sub> = 1/√(d̃<sub>i</sub>d̃<sub>j</sub>): сообщения от вершин с большим числом связей весят меньше. Точные числа — в панели «Вычисление для выбранной вершины».',
    },
    {
      set(done) { runTo(1, done); },
      en: () => { const H = val(); return `<b>Part 2 — sum and update.</b> A adds up the weighted messages and passes the sum through W and a nonlinearity: h<sub>A</sub> = tanh(1.5·Σ) = <b>${U.fmt(H[1][IA])}</b>. B does the same with its negative neighbours: h<sub>B</sub> = <b>${U.fmt(H[1][IB])}</b>. All nodes update at the same time — that is <b>one layer</b>. After one layer A and B are already different.`; },
      ru: () => { const H = val(); return `<b>Часть 2 — сумма и обновление.</b> A складывает взвешенные сообщения и пропускает сумму через W и нелинейность: h<sub>A</sub> = tanh(1,5·Σ) = <b>${U.fmt(H[1][IA])}</b>. B делает то же со своими отрицательными соседями: h<sub>B</sub> = <b>${U.fmt(H[1][IB])}</b>. Все вершины обновляются одновременно — это <b>один слой</b>. Уже после одного слоя A и B различаются.`; },
    },
    {
      set(done) { runTo(LAYERS, done); },
      en: () => '<b>More layers = information from farther away.</b> After layer 2 a node has heard from neighbours of neighbours, after layer 3 from three hops away (dashed rings = receptive field of the selected node). The twins table on the right shows A and B stay clearly different: <b>the graph made them distinguishable</b>.',
      ru: () => '<b>Больше слоёв — информация издалека.</b> После 2-го слоя вершина «слышит» соседей соседей, после 3-го — вершины на расстоянии трёх шагов (пунктирные кольца — рецептивное поле выбранной вершины). Таблица «близнецов» справа показывает: A и B остаются явно разными — <b>граф сделал их различимыми</b>.',
    },
    {
      set(done) { setMode('none'); st.layer = 0; st.mark = 'twins'; draw(); setTimeout(() => runTo(LAYERS, done), 300); },
      en: () => { const H = val(); return `<b>Now remove the edges</b> (A = 0). Each node only talks to itself: h' = tanh(1.5·h). A and B start equal and <b>stay equal at every layer</b> (${U.fmt(H[3][IA])} and ${U.fmt(H[3][IB])}) — exactly what an MLP sees. No choice of weights can separate them. This is <b>local indistinguishability</b>.`; },
      ru: () => { const H = val(); return `<b>Теперь удалим рёбра</b> (A = 0). Каждая вершина общается только сама с собой: h' = tanh(1,5·h). A и B начинают одинаковыми и <b>остаются одинаковыми на каждом слое</b> (${U.fmt(H[3][IA])} и ${U.fmt(H[3][IB])}) — ровно то, что видит MLP. Никакие веса не помогут их разделить. Это <b>локальная неразличимость</b>.`; },
    },
    {
      set() { setMode('gat'); st.layer = 0; st.sel = IA; st.mark = null; },
      en: () => '<b>GATv2:</b> the same procedure, but the weights α<sub>ij</sub> are computed from the features of both ends, so the network chooses whom to listen to. Select a neighbour of A, move the slider below the graph, and watch the strokes and the matrix change. In GCN mode they would not. You can now explore freely.',
      ru: () => '<b>GATv2:</b> та же процедура, но веса α<sub>ij</sub> вычисляются по признакам обеих вершин, и сеть сама выбирает, кого слушать. Выберите соседа A, подвиньте ползунок под графом — штрихи и матрица изменятся. В режиме GCN они бы не изменились. Дальше можно исследовать самостоятельно.',
    },
  ];
  let ti = 0;
  function showTour() {
    const s = tour[ti];
    document.getElementById('tourStep').innerHTML = `${t('Step', 'Шаг')} ${ti + 1} / ${tour.length}<span class="dots">${tour.map((_, k) => `<i class="${k === ti ? 'on' : ''}"></i>`).join('')}</span>`;
    document.getElementById('tourText').innerHTML = (U.lang() === 'ru' ? s.ru : s.en)();
    document.getElementById('tourPrev').disabled = ti === 0;
    document.getElementById('tourNext').disabled = ti === tour.length - 1;
  }
  let busy = false;
  function goTour(k) {
    if (busy) return;
    ti = U.clamp(k, 0, tour.length - 1);
    G = init(); setMode('gcn'); st.layer = 0; st.sel = IA; st.mark = 'twins';
    if (ti === 1) tour[1].set();
    const s = tour[ti];
    busy = true;
    const finish = () => { busy = false; draw(); showTour(); };
    if (s.set.length > 0) { draw(); showTour(); s.set(finish); } else { s.set(); finish(); }
  }
  document.getElementById('tourNext').addEventListener('click', () => goTour(ti + 1));
  document.getElementById('tourPrev').addEventListener('click', () => goTour(ti - 1));
  document.getElementById('tourRestart').addEventListener('click', () => goTour(0));

  U.onTheme.push(() => draw());
  U.onLang.push(() => { draw(); showTour(); });
  goTour(0);
})();
