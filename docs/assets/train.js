/* GNN vs MLP: one training iteration, step by step (input → hidden → output → loss → backprop → update). */
(function () {
  const net = document.getElementById('trNet');
  if (!net) return;
  const t = U.t;

  /* ---------- toy task: 16 nodes, two communities, weak and partly misleading features ---------- */
  const N = 16, R = U.rng(13);
  const pos = [], y = [], X = [];
  for (let i = 0; i < N; i++) {
    const red = i < 8, k = i % 8, ang = (k / 8) * 2 * Math.PI;
    pos.push([(red ? 85 : 235) + 55 * Math.cos(ang), 115 + 75 * Math.sin(ang)]);
    y.push(red ? 1 : 0);
    X.push([U.clamp((red ? 0.35 : -0.35) + 0.75 * R.normal(), -1.5, 1.5), U.clamp(0.6 * R.normal(), -1.5, 1.5)]);
  }
  X[3] = [0.1, 0.0]; X[12] = [0.1, 0.0]; // twins: same features, different class
  X[6] = [-0.8, 0.3]; X[9] = [0.8, -0.2]; // misleading features
  const E = [];
  for (let c = 0; c < 2; c++) for (let k = 0; k < 8; k++) { E.push([c * 8 + k, c * 8 + ((k + 1) % 8)]); if (k % 2 === 0) E.push([c * 8 + k, c * 8 + ((k + 3) % 8)]); }
  E.push([1, 13]); // one cross-community edge
  const adj = Array.from({ length: N }, () => []); E.forEach(([a, b]) => { adj[a].push(b); adj[b].push(a); });
  const train = [0, 2, 5, 8, 10, 14], isTrain = new Set(train), test = [...Array(N).keys()].filter((i) => !isTrain.has(i));

  const Ahat = (M) => M.map((_, i) => { // D^-1/2 (A+I) D^-1/2 · M
    const r = new Array(M[0].length).fill(0), di = adj[i].length + 1;
    [i, ...adj[i]].forEach((j) => { const w = 1 / Math.sqrt(di * (adj[j].length + 1)); for (let f = 0; f < r.length; f++) r[f] += w * M[j][f]; });
    return r;
  });
  const I = (M) => M.map((r) => r.slice());

  const HID = 4, LR = 0.6;
  function makeModel(kind) {
    const r = U.rng(5); // identical initial weights for both models
    return {
      kind, agg: kind === 'gnn' ? Ahat : I,
      W1: Array.from({ length: 2 }, () => Array.from({ length: HID }, () => r.normal() * 0.8)), b1: new Array(HID).fill(0),
      w2: Array.from({ length: HID }, () => r.normal() * 0.8), b2: 0, epoch: 0, hist: [], acc: [],
    };
  }
  function fwd(m) {
    const AX = m.agg(X);
    const Z1 = AX.map((x) => m.b1.map((b, k) => b + x[0] * m.W1[0][k] + x[1] * m.W1[1][k]));
    const H = Z1.map((z) => z.map(Math.tanh));
    const AH = m.agg(H);
    const z2 = AH.map((h) => h.reduce((s, v, k) => s + v * m.w2[k], m.b2));
    const p = z2.map((z) => 1 / (1 + Math.exp(-z)));
    const loss = -train.reduce((s, i) => s + (y[i] ? Math.log(p[i] + 1e-9) : Math.log(1 - p[i] + 1e-9)), 0) / train.length;
    const acc = test.filter((i) => (p[i] > 0.5 ? 1 : 0) === y[i]).length / test.length;
    return { AX, H, AH, p, loss, acc };
  }
  function grads(m, f) {
    const dz2 = f.p.map((p, i) => (isTrain.has(i) ? (p - y[i]) / train.length : 0));
    const dw2 = m.w2.map((_, k) => f.AH.reduce((s, h, i) => s + h[k] * dz2[i], 0)), db2 = dz2.reduce((a, b) => a + b, 0);
    const dAH = dz2.map((g) => m.w2.map((w) => g * w));
    const dH = m.agg(dAH); // Â is symmetric, so Âᵀ = Â
    const dZ1 = dH.map((r, i) => r.map((g, k) => g * (1 - f.H[i][k] ** 2)));
    const dW1 = [0, 1].map((a) => m.b1.map((_, k) => f.AX.reduce((s, x, i) => s + x[a] * dZ1[i][k], 0)));
    const db1 = m.b1.map((_, k) => dZ1.reduce((s, r) => s + r[k], 0));
    return { dW1, db1, dw2, db2 };
  }
  function update(m, g) {
    for (let a = 0; a < 2; a++) for (let k = 0; k < HID; k++) m.W1[a][k] -= LR * g.dW1[a][k];
    for (let k = 0; k < HID; k++) { m.b1[k] -= LR * g.db1[k]; m.w2[k] -= LR * g.dw2[k]; }
    m.b2 -= LR * g.db2; m.epoch++;
  }
  function epoch(m) { const f = fwd(m); update(m, grads(m, f)); const f2 = fwd(m); m.hist.push(f2.loss); m.acc.push(f2.acc); }

  const st = { mlp: makeModel('mlp'), gnn: makeModel('gnn'), phase: 0, sel: 3, g: null, playing: false };
  const PH = ['input', 'agg1', 'hidden', 'agg2', 'output', 'loss', 'backward', 'update'];
  const phName = () => [t('1 · input', '1 · вход'), t('2 · aggregate', '2 · агрегация'), t('3 · hidden layer', '3 · скрытый слой'), t('4 · aggregate', '4 · агрегация'), t('5 · output', '5 · выход'), t('6 · loss', '6 · потери'), t('7 · backprop', '7 · обратное распространение'), t('8 · update weights', '8 · обновление весов')];

  /* ---------- drawing ---------- */
  const NW = 700, NH = 330;
  net.setAttribute('viewBox', `0 0 ${NW} ${NH}`);
  const COLS = [40, 160, 290, 420, 560]; // input, agg1, hidden, agg2, output
  const rowY = { mlp: 20, gnn: 180 };
  const nodeY = (base, k, n) => base + 70 + (k - (n - 1) / 2) * 30;

  function drawNet() {
    net.innerHTML = '';
    const ph = PH[st.phase], ink = U.css('--ink'), muted = U.css('--muted'), acc = U.css('--accent'), line = U.css('--line');
    const colOf = { input: 0, agg1: 1, hidden: 2, agg2: 3, output: 4, loss: 4, backward: -1, update: -1 };
    for (const kind of ['mlp', 'gnn']) {
      const m = st[kind], f = fwd(m), base = rowY[kind], i = st.sel;
      U.el('text', { x: 4, y: base + 8, 'font-size': 13, 'font-weight': 700, fill: ink, text: kind === 'mlp' ? 'MLP' : 'GNN (GCN)' }, net);
      // column highlight
      const c = colOf[ph];
      if (c >= 0) U.el('rect', { x: COLS[c] - 34, y: base + 12, width: 68, height: 136, rx: 10, fill: acc, opacity: 0.1 }, net);
      const xin = X[i], aggX = f.AX[i], hid = f.H[i], aggH = f.AH[i], p = f.p[i];
      const P = { in: [0, 1].map((k) => [COLS[0], nodeY(base, k, 2)]), hid: [...Array(HID).keys()].map((k) => [COLS[2], nodeY(base, k, HID)]), out: [[COLS[4], nodeY(base, 0, 1)]] };
      const back = ph === 'backward' || ph === 'update';
      const g = back ? st.g && st.g[kind] : null;
      // W1 lines
      for (let a = 0; a < 2; a++) for (let k = 0; k < HID; k++) {
        const w = m.W1[a][k], gl = g ? Math.min(1, Math.abs(g.dW1[a][k]) * 6) : 0;
        U.el('line', { x1: P.in[a][0] + 12, y1: P.in[a][1], x2: P.hid[k][0] - 12, y2: P.hid[k][1], stroke: back ? U.css('--red') : (w >= 0 ? '#e88020' : '#3466d6'), 'stroke-width': back ? 0.6 + 6 * gl : 0.5 + 2.5 * Math.min(Math.abs(w), 1.5), opacity: back ? 0.25 + 0.7 * gl : 0.55 }, net);
      }
      for (let k = 0; k < HID; k++) {
        const w = m.w2[k], gl = g ? Math.min(1, Math.abs(g.dw2[k]) * 6) : 0;
        U.el('line', { x1: P.hid[k][0] + 12, y1: P.hid[k][1], x2: P.out[0][0] - 14, y2: P.out[0][1], stroke: back ? U.css('--red') : (w >= 0 ? '#e88020' : '#3466d6'), 'stroke-width': back ? 0.6 + 6 * gl : 0.5 + 2.5 * Math.min(Math.abs(w), 1.5), opacity: back ? 0.25 + 0.7 * gl : 0.55 }, net);
      }
      // aggregation boxes
      [[1, aggX], [3, aggH]].forEach(([col, v]) => {
        const x = COLS[col];
        if (kind === 'gnn') {
          U.el('rect', { x: x - 30, y: base + 22, width: 60, height: 116, rx: 8, fill: U.css('--panel'), stroke: acc, 'stroke-width': 1.5 }, net);
          U.el('text', { x, y: base + 40, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: acc, text: 'Â ·' }, net);
          U.el('text', { x, y: base + 55, 'text-anchor': 'middle', 'font-size': 9.5, fill: muted, text: t('mix with', 'смешать с') }, net);
          U.el('text', { x, y: base + 67, 'text-anchor': 'middle', 'font-size': 9.5, fill: muted, text: t('neighbours', 'соседями') }, net);
          v.slice(0, 4).forEach((val, k) => U.el('text', { x, y: base + 86 + k * 13, 'text-anchor': 'middle', 'font-size': 10, fill: ink, class: 'mono', text: U.fmt(val) }, net));
        } else {
          U.el('rect', { x: x - 30, y: base + 50, width: 60, height: 60, rx: 8, fill: 'none', stroke: line, 'stroke-dasharray': '4 4' }, net);
          U.el('text', { x, y: base + 76, 'text-anchor': 'middle', 'font-size': 10, fill: muted, text: t('no edges:', 'рёбер нет:') }, net);
          U.el('text', { x, y: base + 90, 'text-anchor': 'middle', 'font-size': 10, fill: muted, text: t('skip', 'пропуск') }, net);
        }
      });
      const neuron = ([x, yy], v, label, r = 12) => {
        U.el('circle', { cx: x, cy: yy, r, fill: U.valColor(v), stroke: ink, 'stroke-width': 1 }, net);
        U.el('text', { x, y: yy + 3.5, 'text-anchor': 'middle', 'font-size': 9.5, fill: '#1d2230', text: U.fmt(v) }, net);
        if (label) U.el('text', { x: x - 16, y: yy + 4, 'text-anchor': 'end', 'font-size': 10, fill: muted, text: label }, net);
      };
      neuron(P.in[0], xin[0], 'x₁'); neuron(P.in[1], xin[1], 'x₂');
      P.hid.forEach((q, k) => neuron(q, hid[k], k === 0 ? 'h' : ''));
      U.el('circle', { cx: P.out[0][0], cy: P.out[0][1], r: 16, fill: p > 0.5 ? U.css('--red') : U.css('--green'), opacity: 0.25 + 0.75 * Math.abs(p - 0.5) * 2, stroke: ink }, net);
      U.el('text', { x: P.out[0][0], y: P.out[0][1] + 4, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, fill: ink, text: U.fmt(p) }, net);
      U.el('text', { x: P.out[0][0] + 22, y: P.out[0][1] - 4, 'font-size': 10, fill: muted, text: t('p(red)', 'p(красный)') }, net);
      U.el('text', { x: P.out[0][0] + 22, y: P.out[0][1] + 10, 'font-size': 10, fill: muted, text: `${t('loss', 'потери')} ${U.fmt(f.loss, 3)}` }, net);
      U.el('text', { x: P.out[0][0] + 22, y: P.out[0][1] + 24, 'font-size': 10, fill: muted, text: `${t('test acc', 'точность')} ${U.fmt(f.acc)}` }, net);
    }
    // column headers
    [[0, t('input x_i', 'вход x_i')], [1, '(GNN) Â'], [2, t('hidden tanh(·W₁)', 'скрытый tanh(·W₁)')], [3, '(GNN) Â'], [4, t('output σ(·w₂)', 'выход σ(·w₂)')]].forEach(([c, s]) => U.el('text', { x: COLS[c], y: NH - 4, 'text-anchor': 'middle', 'font-size': 10.5, fill: muted, text: s }, net));
  }

  function drawGraphs() {
    for (const kind of ['mlp', 'gnn']) {
      const s = document.getElementById(kind === 'mlp' ? 'trGmlp' : 'trGgnn'), f = fwd(st[kind]);
      s.setAttribute('viewBox', '0 0 320 230'); s.innerHTML = '';
      E.forEach(([a, b]) => U.el('line', { x1: pos[a][0], y1: pos[a][1], x2: pos[b][0], y2: pos[b][1], stroke: U.css('--muted'), 'stroke-width': 1.2, opacity: kind === 'mlp' ? 0.15 : 0.5, 'stroke-dasharray': kind === 'mlp' ? '3 4' : '' }, s));
      for (let i = 0; i < N; i++) {
        const g = U.el('g', { transform: `translate(${pos[i][0]},${pos[i][1]})`, style: 'cursor:pointer' }, s);
        if (i === st.sel) U.el('circle', { r: 17, fill: 'none', stroke: U.css('--accent'), 'stroke-width': 3 }, g);
        const p = f.p[i], pr = p > 0.5 ? 1 : 0, wrong = !isTrain.has(i) && pr !== y[i];
        U.el('circle', { r: 12, fill: p > 0.5 ? U.css('--red') : U.css('--green'), opacity: 0.25 + 0.75 * Math.abs(p - 0.5) * 2 }, g);
        U.el('circle', { r: 12, fill: 'none', stroke: y[i] ? U.css('--red') : U.css('--green'), 'stroke-width': isTrain.has(i) ? 3.5 : 1.5, 'stroke-dasharray': isTrain.has(i) ? '' : '2 2' }, g);
        U.el('text', { y: 4, 'text-anchor': 'middle', 'font-size': 10, 'font-weight': 700, fill: U.css('--ink'), text: wrong ? '✗' : String(i) }, g);
        g.addEventListener('click', () => { st.sel = i; render(); });
      }
      const fl = document.getElementById(kind === 'mlp' ? 'trAccMlp' : 'trAccGnn');
      fl.textContent = `${t('epoch', 'эпоха')} ${st[kind].epoch} · ${t('loss', 'потери')} ${U.fmt(f.loss, 3)} · ${t('test accuracy', 'точность на тесте')} ${U.fmt(f.acc)}`;
    }
  }

  function drawCurve() {
    const c = document.getElementById('trCurve'), w = 420, h = 150, ctx = U.setupCanvas(c, w, h);
    ctx.clearRect(0, 0, w, h);
    const m = { l: 30, r: 8, t: 8, b: 18 }, n = Math.max(20, st.mlp.hist.length);
    const all = [...st.mlp.hist, ...st.gnn.hist, 0.8], mx = Math.max(...all);
    const X2 = (k) => m.l + (k / n) * (w - m.l - m.r), Y2 = (v) => h - m.b - (v / mx) * (h - m.t - m.b);
    ctx.strokeStyle = U.css('--line'); ctx.fillStyle = U.css('--muted'); ctx.font = '10px system-ui';
    [0, mx / 2, mx].forEach((v) => { ctx.beginPath(); ctx.moveTo(m.l, Y2(v)); ctx.lineTo(w - m.r, Y2(v)); ctx.stroke(); ctx.fillText(v.toFixed(2), 2, Y2(v) + 3); });
    ctx.fillText(t('epoch', 'эпоха'), w - 40, h - 4);
    [['mlp', U.css('--muted'), [5, 4]], ['gnn', U.css('--accent'), []]].forEach(([k, col, dash]) => {
      const hs = st[k].hist; if (!hs.length) return;
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.beginPath();
      hs.forEach((v, e) => (e ? ctx.lineTo(X2(e + 1), Y2(v)) : ctx.moveTo(X2(e + 1), Y2(v)))); ctx.stroke(); ctx.setLineDash([]);
    });
  }

  function explain() {
    const ph = PH[st.phase], i = st.sel, fm = fwd(st.mlp), fg = fwd(st.gnn);
    const nb = adj[i].join(', ');
    const twin = (i === 3 || i === 12) ? t(' (Node 3 and node 12 are twins: identical features, different classes.)', ' (Вершины 3 и 12 — «близнецы»: одинаковые признаки, разные классы.)') : '';
    const T = {
      input: t(`<b>Forward pass, input layer.</b> Both models receive the same two features of node ${i}: x = (${U.fmt(X[i][0])}, ${U.fmt(X[i][1])}).${twin} The features are weak: red nodes have slightly larger x₁ on average, with heavy overlap.`,
        `<b>Прямой проход, входной слой.</b> Обе модели получают одни и те же два признака вершины ${i}: x = (${U.fmt(X[i][0])}, ${U.fmt(X[i][1])}).${twin} Признаки слабые: у красных вершин x₁ в среднем чуть больше, но классы сильно перекрываются.`),
      agg1: t(`<b>Aggregation (GNN only).</b> The GNN replaces x<sub>${i}</sub> by a weighted average over node ${i} and its neighbours {${nb}}: Â·X gives (${U.fmt(fg.AX[i][0])}, ${U.fmt(fg.AX[i][1])}). Neighbours in the same community pull the vector towards the community's typical value, which cancels part of the noise. The MLP skips this step.`,
        `<b>Агрегация (только GNN).</b> GNN заменяет x<sub>${i}</sub> взвешенным средним по вершине ${i} и её соседям {${nb}}: Â·X = (${U.fmt(fg.AX[i][0])}, ${U.fmt(fg.AX[i][1])}). Соседи из того же сообщества тянут вектор к типичному значению сообщества и гасят часть шума. MLP этот шаг пропускает.`),
      hidden: t(`<b>Hidden layer.</b> Each of the 4 hidden neurons computes tanh(w·input + b) — orange lines are positive weights, blue are negative, thickness = |w|. The weights W₁ are <b>shared by all nodes</b> and both models started from identical values. MLP hidden: [${fm.H[i].map((v) => U.fmt(v)).join(', ')}]; GNN: [${fg.H[i].map((v) => U.fmt(v)).join(', ')}].`,
        `<b>Скрытый слой.</b> Каждый из 4 скрытых нейронов вычисляет tanh(w·вход + b): оранжевые линии — положительные веса, синие — отрицательные, толщина = |w|. Веса W₁ <b>общие для всех вершин</b>, и обе модели стартовали с одинаковых значений. Скрытый слой MLP: [${fm.H[i].map((v) => U.fmt(v)).join(', ')}]; GNN: [${fg.H[i].map((v) => U.fmt(v)).join(', ')}].`),
      agg2: t('<b>Second aggregation (GNN only).</b> The hidden vectors are mixed with neighbours once more, so after two layers node i has information from up to 2 hops away. This is exactly a 2-layer GCN: Â·tanh(Â X W₁)·w₂.',
        '<b>Вторая агрегация (только GNN).</b> Скрытые векторы ещё раз смешиваются с соседями, и после двух слоёв вершина i знает о вершинах на расстоянии до 2 шагов. Это и есть двухслойная GCN: Â·tanh(Â X W₁)·w₂.'),
      output: t(`<b>Output layer.</b> One neuron with a sigmoid gives the probability that node ${i} is red: MLP p = ${U.fmt(fm.p[i])}, GNN p = ${U.fmt(fg.p[i])} (true class: ${y[i] ? 'red' : 'green'}). Predicted red if p &gt; 0.5.`,
        `<b>Выходной слой.</b> Один нейрон с сигмоидой даёт вероятность, что вершина ${i} красная: MLP p = ${U.fmt(fm.p[i])}, GNN p = ${U.fmt(fg.p[i])} (истинный класс: ${y[i] ? 'красный' : 'зелёный'}). Красный, если p &gt; 0,5.`),
      loss: t(`<b>Loss.</b> Only the 6 labelled nodes (thick rings: ${train.join(', ')}) enter the loss: L = −mean[y log p + (1−y) log(1−p)]. MLP L = ${U.fmt(fm.loss, 3)}, GNN L = ${U.fmt(fg.loss, 3)}. The other 10 nodes are the test set (dashed rings). In the GNN they still help: unlabelled neighbours pass their features along the edges.`,
        `<b>Функция потерь.</b> В потери входят только 6 размеченных вершин (толстые кольца: ${train.join(', ')}): L = −среднее[y log p + (1−y) log(1−p)]. MLP L = ${U.fmt(fm.loss, 3)}, GNN L = ${U.fmt(fg.loss, 3)}. Остальные 10 вершин — тестовые (пунктир). В GNN они всё равно помогают: неразмеченные соседи передают свои признаки по рёбрам.`),
      backward: t('<b>Backpropagation.</b> The gradient of L flows back from the output to every weight (red lines, thickness = |∂L/∂w|). In the GNN it flows back <b>through Â as well</b>: the error of one labelled node reaches the weights through all of its neighbours. ∂L/∂W₁ = (ÂX)ᵀ[(Â(∂L/∂z ⊗ w₂)) ⊙ (1 − H²)].',
        '<b>Обратное распространение.</b> Градиент L идёт от выхода назад к каждому весу (красные линии, толщина = |∂L/∂w|). В GNN он проходит <b>и через Â</b>: ошибка одной размеченной вершины доходит до весов через всех её соседей. ∂L/∂W₁ = (ÂX)ᵀ[(Â(∂L/∂z ⊗ w₂)) ⊙ (1 − H²)].'),
      update: t(`<b>Update.</b> Gradient descent: W ← W − η·∂L/∂W with η = ${LR}. That completes one epoch. Press “Next step” to start the next forward pass, or “Train 100 epochs” to watch both loss curves. Usually the GNN gets more test nodes right, including twin 3 vs 12, which the MLP can never separate.`,
        `<b>Обновление.</b> Градиентный спуск: W ← W − η·∂L/∂W, η = ${LR}. Одна эпоха завершена. Нажмите «Следующий шаг», чтобы начать новый прямой проход, или «Обучить 100 эпох», чтобы увидеть обе кривые потерь. Обычно GNN правильно классифицирует больше тестовых вершин, в том числе близнецов 3 и 12, которых MLP не разделит никогда.`),
    };
    document.getElementById('trText').innerHTML = T[ph];
    document.getElementById('trPhases').innerHTML = phName().map((s, k) => `<span class="${k === st.phase ? 'on' : ''}">${s}</span>`).join('');
  }

  function render() { drawNet(); drawGraphs(); drawCurve(); explain(); }

  function nextStep() {
    const ph = PH[st.phase];
    if (ph === 'loss') st.g = { mlp: grads(st.mlp, fwd(st.mlp)), gnn: grads(st.gnn, fwd(st.gnn)) };
    if (ph === 'update') {
      for (const k of ['mlp', 'gnn']) { update(st[k], st.g[k]); const f = fwd(st[k]); st[k].hist.push(f.loss); st[k].acc.push(f.acc); }
      st.g = null; st.phase = 0;
    } else st.phase++;
    render();
  }
  function trainN(n) {
    if (st.playing) return; st.playing = true; st.phase = 0; let k = 0;
    const go = () => { for (let r = 0; r < 2 && k < n; r++, k++) { epoch(st.mlp); epoch(st.gnn); } render(); if (k < n) setTimeout(go, 16); else st.playing = false; };
    go();
  }
  document.getElementById('trNext').addEventListener('click', nextStep);
  document.getElementById('trEpoch').addEventListener('click', () => { st.phase = 0; epoch(st.mlp); epoch(st.gnn); render(); });
  document.getElementById('trPlay').addEventListener('click', () => trainN(100));
  document.getElementById('trReset').addEventListener('click', () => { st.mlp = makeModel('mlp'); st.gnn = makeModel('gnn'); st.phase = 0; st.g = null; render(); });
  U.onTheme.push(render); U.onLang.push(render);
  render();
})();
