/* Message-passing playground: GCN, GATv2 and the same network with edges removed. */
(function () {
  const W = 640, H = 380, LAYERS = 3, WGT = 1.5;
  const svg = document.getElementById('mpSvg');
  if (!svg) return;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

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
  const st = { mode: 'gcn', layer: 0, sel: 0, edit: false, editFirst: null, anim: null };
  const IA = 0, IB = 4;

  const nbrs = () => {
    const a = G.nodes.map(() => []);
    if (st.mode === 'none') return a;
    for (const [u, v] of G.edges) { a[u].push(v); a[v].push(u); }
    return a;
  };

  /* GATv2 with fixed, illustrative parameters (in a real model they are learned):
     e_ij = a^T LeakyReLU(W [h_i || h_j]); this choice prefers neighbours similar to i and with high value. */
  const GW = [[1, -1], [-1, 1], [1, 1], [-1, -1]], GA = [-2.5, -2.5, 0.8, 0.0];
  const lrelu = (z) => (z > 0 ? z : 0.2 * z);
  const gatScore = (hi, hj) => { let e = 0; for (let k = 0; k < 4; k++) e += GA[k] * lrelu(GW[k][0] * hi + GW[k][1] * hj); return e; };

  /* coefficient matrix c_ij at layer l given features h */
  function coeffs(h, adj) {
    const n = h.length, C = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      const N = [i, ...adj[i]];
      if (st.mode === 'gat') {
        const e = N.map((j) => gatScore(h[i], h[j])), mx = Math.max(...e);
        const ex = e.map((v) => Math.exp(v - mx)), s = ex.reduce((a, b) => a + b, 0);
        N.forEach((j, t) => { C[i][j] = ex[t] / s; });
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

  const NS = 'http://www.w3.org/2000/svg';
  function draw(progress) {
    const { adj, Hs, Cs } = forward();
    const h = Hs[st.layer];
    svg.innerHTML = '';
    const ink = U.css('--ink'), muted = U.css('--muted'), line = U.css('--line'), acc = U.css('--accent');

    // receptive field of selected node at current layer
    const rf = new Set([st.sel]);
    if (st.mode !== 'none') { const d = U.bfs(adj, st.sel); d.forEach((v, i) => { if (v <= Math.max(st.layer, 0)) rf.add(i); }); }
    const gEdges = U.el('g', {}, svg);
    // draw all graph edges (dashed if removed)
    const removed = st.mode === 'none';
    for (const [u, v] of G.edges) {
      const a = G.nodes[u], b = G.nodes[v];
      U.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: removed ? line : muted, 'stroke-width': 1.5, 'stroke-dasharray': removed ? '4 5' : '', opacity: removed ? 0.7 : 0.45 }, gEdges);
    }
    // coefficient-weighted edges into selected node for next layer
    if (st.layer < LAYERS && !removed) {
      const C = Cs[st.layer];
      for (const j of adj[st.sel]) {
        const a = G.nodes[j], b = G.nodes[st.sel];
        U.el('line', { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: acc, 'stroke-width': 1 + 12 * C[st.sel][j], 'stroke-linecap': 'round', opacity: 0.55 }, gEdges);
      }
    }
    // messages in flight
    if (progress !== undefined && st.layer < LAYERS) {
      const C = Cs[st.layer];
      const t = progress;
      for (let i = 0; i < G.nodes.length; i++) {
        for (const j of adj[i]) {
          const a = G.nodes[j], b = G.nodes[i];
          U.el('circle', { cx: a.x + (b.x - a.x) * t, cy: a.y + (b.y - a.y) * t, r: 2.5 + 7 * C[i][j], fill: U.valColor(h[j]), stroke: ink, 'stroke-width': 0.6, opacity: 0.95 }, gEdges);
        }
        if (removed) { const b = G.nodes[i]; U.el('circle', { cx: b.x, cy: b.y, r: 25 + 8 * Math.sin(Math.PI * t), fill: 'none', stroke: acc, 'stroke-width': 1.2, opacity: 0.6 * (1 - t) }, gEdges); }
      }
    }
    // nodes
    G.nodes.forEach((n, i) => {
      const g = U.el('g', { transform: `translate(${n.x},${n.y})`, style: 'cursor:pointer' }, svg);
      if (rf.has(i) && i !== st.sel) U.el('circle', { r: 27, fill: 'none', stroke: acc, 'stroke-width': 1.5, 'stroke-dasharray': '3 3' }, g);
      if (i === st.sel) U.el('circle', { r: 28, fill: 'none', stroke: acc, 'stroke-width': 3 }, g);
      if (st.edit && st.editFirst === i) U.el('circle', { r: 28, fill: 'none', stroke: U.css('--amber'), 'stroke-width': 3 }, g);
      U.el('circle', { r: 21, fill: U.valColor(h[i]), stroke: (i === IA || i === IB) ? ink : muted, 'stroke-width': (i === IA || i === IB) ? 2.2 : 1 }, g);
      U.el('text', { y: -1, 'text-anchor': 'middle', 'font-size': 12, 'font-weight': 700, fill: '#1d2230', text: n.id }, g);
      U.el('text', { y: 12, 'text-anchor': 'middle', 'font-size': 10, fill: '#1d2230', text: U.fmt(h[i]) }, g);
      g.addEventListener('click', () => onNode(i));
    });
    U.el('text', { x: 12, y: 22, fill: muted, 'font-size': 13, text: `layer ${st.layer} of ${LAYERS} · ${st.mode === 'gcn' ? 'GCN' : st.mode === 'gat' ? 'GATv2' : 'edges removed (A = 0)'}` }, svg);
    if (progress === undefined) panels(adj, Hs, Cs);
  }

  function panels(adj, Hs, Cs) {
    const i = st.sel, n = G.nodes[i];
    const calc = document.getElementById('mpCalc');
    if (st.layer >= LAYERS) {
      calc.textContent = `Node ${n.id}: final representation after ${LAYERS} layers\nh_${n.id}^(${LAYERS}) = ${U.fmt(Hs[LAYERS][i], 3)}\n\nPress "Reset" to go back to layer 0.`;
    } else {
      const C = Cs[st.layer], h = Hs[st.layer], N = [i, ...adj[i]];
      const name = st.mode === 'gat' ? 'α' : 'c';
      let s = 0, out = `Node ${n.id}, layer ${st.layer} → ${st.layer + 1}\n`;
      if (st.mode === 'none') out += 'N(i) = {i}: no edges, no messages.\n';
      else out += `N(${n.id}) ∪ {${n.id}} = {${N.map((j) => G.nodes[j].id).join(', ')}}\n`;
      out += st.mode === 'gat' ? 'α_ij = softmax_j( aᵀ LeakyReLU(W[h_i ‖ h_j]) )\n' : st.mode === 'gcn' ? 'c_ij = 1 / √(d̃_i d̃_j),  d̃ = degree + 1\n' : 'c_ii = 1 (Ã = I, D̃ = I)\n';
      for (const j of N) {
        s += C[i][j] * h[j];
        out += `  ${name}_${n.id},${G.nodes[j].id} = ${U.fmt(C[i][j], 3)} × h_${G.nodes[j].id} (${U.fmt(h[j])}) = ${U.fmt(C[i][j] * h[j], 3)}\n`;
      }
      out += `Σ = ${U.fmt(s, 3)}\nh_${n.id}^(${st.layer + 1}) = tanh(${WGT} · Σ) = ${U.fmt(Math.tanh(WGT * s), 3)}`;
      calc.textContent = out;
    }
    // matrix
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
    document.getElementById('mpMatrixTitle').textContent = st.mode === 'gat' ? `Attention matrix α (layer ${Math.min(st.layer, LAYERS - 1)}): rows sum to 1, depend on features` : st.mode === 'gcn' ? 'Normalised adjacency Â = D̃^−½ (A+I) D̃^−½: fixed by the graph' : 'A = 0 ⇒ Ã = I: every node only talks to itself';
    // twins
    const T = document.getElementById('mpTwins');
    let rows = '<tr><th>layer</th><th>h_A</th><th>h_B</th><th>A vs B</th></tr>';
    for (let l = 0; l <= LAYERS; l++) {
      const a = Hs[l][IA], b = Hs[l][IB], same = Math.abs(a - b) < 1e-9;
      rows += `<tr><td>${l}</td><td class="n">${U.fmt(a, 3)}</td><td class="n">${U.fmt(b, 3)}</td><td>${same ? '<span class="verdict no">identical</span>' : '<span class="verdict yes">different</span>'}</td></tr>`;
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

  function step() {
    if (st.anim || st.layer >= LAYERS) return;
    const t0 = performance.now(), dur = 900;
    const tick = (now) => {
      const t = Math.min(1, (now - t0) / dur);
      draw(t);
      if (t < 1) st.anim = requestAnimationFrame(tick);
      else { st.anim = null; st.layer++; draw(); }
    };
    st.anim = requestAnimationFrame(tick);
  }

  document.querySelectorAll('#mpMode button').forEach((b) => b.addEventListener('click', () => {
    document.querySelectorAll('#mpMode button').forEach((x) => x.classList.remove('on'));
    b.classList.add('on'); st.mode = b.dataset.m; draw();
  }));
  document.getElementById('mpStep').addEventListener('click', step);
  document.getElementById('mpReset').addEventListener('click', () => { st.layer = 0; draw(); });
  document.getElementById('mpRun').addEventListener('click', () => {
    st.layer = 0; draw();
    const go = () => { if (st.layer < LAYERS) { step(); setTimeout(go, 1100); } };
    setTimeout(go, 150);
  });
  document.getElementById('mpEdit').addEventListener('click', (e) => {
    st.edit = !st.edit; st.editFirst = null; e.target.classList.toggle('on', st.edit);
    e.target.textContent = st.edit ? 'Editing edges: click two nodes' : 'Edit edges'; draw();
  });
  document.getElementById('mpRestore').addEventListener('click', () => { G = init(); st.layer = 0; draw(); });
  document.getElementById('mpFeat').addEventListener('input', (e) => { G.nodes[st.sel].f = +e.target.value; draw(); });
  U.onTheme.push(() => draw());
  draw();
})();
