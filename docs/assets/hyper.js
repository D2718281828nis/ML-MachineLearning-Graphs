/* Hyperbolic graphs: tree in Euclidean vs Poincaré disk, Möbius panning, tangent-space aggregation. */
(function () {
  const cE = document.getElementById('hyEuc'), cP = document.getElementById('hyPoi');
  if (!cE) return;
  const S = 330;
  const st = { b: 3, D: 4, sel: null, tree: null, pos: null, drag: null };

  const cosh = Math.cosh, acosh = Math.acosh;
  const hdist = (x, y) => {
    const d2 = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2, nx = 1 - x[0] ** 2 - x[1] ** 2, ny = 1 - y[0] ** 2 - y[1] ** 2;
    return acosh(1 + 2 * d2 / Math.max(nx * ny, 1e-12));
  };
  // Möbius translation z -> (z + a) / (1 + conj(a) z)
  const mobius = (z, a) => {
    const nr = [z[0] + a[0], z[1] + a[1]];
    const dr = [1 + a[0] * z[0] + a[1] * z[1], a[0] * z[1] - a[1] * z[0]];
    const den = dr[0] ** 2 + dr[1] ** 2;
    return [(nr[0] * dr[0] + nr[1] * dr[1]) / den, (nr[1] * dr[0] - nr[0] * dr[1]) / den];
  };

  function buildTree() {
    const nodes = [{ d: 0, a0: 0, a1: 2 * Math.PI, p: -1 }];
    for (let i = 0; i < nodes.length; i++) {
      const n = nodes[i]; if (n.d >= st.D) continue;
      const w = (n.a1 - n.a0) / st.b;
      for (let k = 0; k < st.b; k++) nodes.push({ d: n.d + 1, a0: n.a0 + k * w, a1: n.a0 + (k + 1) * w, p: i });
    }
    const ell = Math.min(0.9 + Math.log(st.b), 4.2 / st.D); // edge length; capped so the leaves stay visible inside the disk
    nodes.forEach((n) => {
      const th = (n.a0 + n.a1) / 2;
      const rho = Math.tanh(n.d * ell / 2); // hyperbolic radius d·ℓ  →  Poincaré radius tanh(r/2)
      n.poi = [rho * Math.cos(th), rho * Math.sin(th)];
      const re = n.d / st.D * 0.96;
      n.euc = [re * Math.cos(th), re * Math.sin(th)];
    });
    st.tree = nodes; st.pos = nodes.map((n) => n.poi.slice()); st.sel = null;
    stats();
  }

  function treeDist(i, j) {
    const T = st.tree; let a = i, b = j, d = 0;
    while (T[a].d > T[b].d) { a = T[a].p; d++; }
    while (T[b].d > T[a].d) { b = T[b].p; d++; }
    while (a !== b) { a = T[a].p; b = T[b].p; d += 2; }
    return d;
  }

  function stats() {
    const T = st.tree, leaves = T.map((n, i) => i).filter((i) => T[i].d === st.D);
    const sample = leaves.length > 120 ? leaves.filter((_, k) => k % Math.ceil(leaves.length / 120) === 0) : leaves;
    let rE = [], rH = [], minE = Infinity, minH = Infinity;
    for (let x = 0; x < sample.length; x++) for (let y = x + 1; y < sample.length; y++) {
      const i = sample[x], j = sample[y], td = treeDist(i, j);
      const de = Math.hypot(T[i].euc[0] - T[j].euc[0], T[i].euc[1] - T[j].euc[1]), dh = hdist(T[i].poi, T[j].poi);
      rE.push(de / td); rH.push(dh / td); minE = Math.min(minE, de); minH = Math.min(minH, dh);
    }
    const dist = (r) => Math.max(...r) / Math.min(...r);
    const ell = Math.min(0.9 + Math.log(st.b), 4.2 / st.D), edgeE = 0.96 / st.D;
    document.getElementById('hyStats').innerHTML = `
      <div class="stat"><div class="v">${T.length}</div><div class="k">${U.t('nodes', 'вершин')}, ${leaves.length} ${U.t('leaves', 'листьев')} = ${st.b}<sup>${st.D}</sup></div></div>
      <div class="stat"><div class="v">${U.fmt(minE / edgeE)}</div><div class="k">${U.t('closest leaves, Euclidean (in edge lengths)', 'ближайшие листья, евклидово (в длинах ребра)')}</div></div>
      <div class="stat"><div class="v">${U.fmt(minH / ell)}</div><div class="k">${U.t('closest leaves, hyperbolic (in edge lengths)', 'ближайшие листья, гиперболически (в длинах ребра)')}</div></div>
      <div class="stat"><div class="v">${U.fmt(dist(rE), 1)} → ${U.fmt(dist(rH), 1)}</div><div class="k">${U.t('distortion of leaf distances, Euclidean → hyperbolic (1 = perfect)', 'искажение расстояний между листьями, евклидово → гиперболическое (1 = идеально)')}</div></div>`;
    const r0 = U.fmt(st.D * ell, 1), e0 = U.fmt(2 * Math.PI * st.D * ell, 0), h0 = U.fmt(2 * Math.PI * Math.sinh(st.D * ell), 0);
    document.getElementById('hyGrowth').textContent = U.t(
      `Leaves grow as ${st.b}^d. A circle of radius r has length 2πr in the plane but 2π·sinh r in hyperbolic space: at r = ${r0} that is ${e0} vs ${h0}, enough room for ${leaves.length} leaves.`,
      `Число листьев растёт как ${st.b}^d. Окружность радиуса r имеет длину 2πr на плоскости, но 2π·sinh r в гиперболическом пространстве: при r = ${r0} это ${e0} против ${h0} — достаточно места для ${leaves.length} листьев.`);
  }

  function geodesic(ctx, p, q, X) {
    const cross = p[0] * q[1] - p[1] * q[0];
    if (Math.abs(cross) < 1e-6) { ctx.moveTo(...X(p)); ctx.lineTo(...X(q)); return; }
    const n2 = p[0] ** 2 + p[1] ** 2, pi = [p[0] / n2, p[1] / n2];
    // circumcentre of p, q, p* (inversion of p)
    const ax = p[0], ay = p[1], bx = q[0], by = q[1], cx = pi[0], cy = pi[1];
    const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if (Math.abs(d) < 1e-9 || n2 < 1e-9) { ctx.moveTo(...X(p)); ctx.lineTo(...X(q)); return; }
    const ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    const uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;
    const r = Math.hypot(ax - ux, ay - uy);
    let a1 = Math.atan2(ay - uy, ax - ux), a2 = Math.atan2(by - uy, bx - ux);
    let diff = a2 - a1; while (diff > Math.PI) diff -= 2 * Math.PI; while (diff < -Math.PI) diff += 2 * Math.PI;
    const steps = 18;
    for (let k = 0; k <= steps; k++) { const a = a1 + diff * k / steps, pt = [ux + r * Math.cos(a), uy + r * Math.sin(a)]; k ? ctx.lineTo(...X(pt)) : ctx.moveTo(...X(pt)); }
  }

  function drawDisk(canvas, key) {
    const ctx = U.setupCanvas(canvas, S, S), R = S / 2 - 8, X = (p) => [S / 2 + p[0] * R, S / 2 + p[1] * R];
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = U.css('--panel'); ctx.strokeStyle = U.css('--muted'); ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(S / 2, S / 2, R, 0, 7); ctx.fill(); ctx.stroke();
    const T = st.tree, pos = key === 'poi' ? st.pos : T.map((n) => n.euc);
    let dsel = null;
    if (st.sel !== null) dsel = T.map((_, j) => treeDist(st.sel, j));
    ctx.strokeStyle = U.css('--accent'); ctx.globalAlpha = 0.55; ctx.lineWidth = 1;
    ctx.beginPath();
    T.forEach((n, i) => { if (n.p >= 0) { if (key === 'poi') geodesic(ctx, pos[n.p], pos[i], X); else { ctx.moveTo(...X(pos[n.p])); ctx.lineTo(...X(pos[i])); } } });
    ctx.stroke(); ctx.globalAlpha = 1;
    T.forEach((n, i) => {
      const p = X(pos[i]);
      let r = key === 'poi' ? Math.max(1.2, 5.5 * (1 - (pos[i][0] ** 2 + pos[i][1] ** 2))) : Math.max(1.6, 5 - n.d * 0.6);
      ctx.beginPath(); ctx.arc(p[0], p[1], r, 0, 7);
      ctx.fillStyle = i === st.sel ? U.css('--red') : dsel ? U.valColor(1 - dsel[i] / (2 * st.D)) : U.css('--ink');
      ctx.fill();
    });
  }
  const drawAll = () => { drawDisk(cE, 'euc'); drawDisk(cP, 'poi'); };

  const toDisk = (e) => { const r = cP.getBoundingClientRect(), R = r.width / 2 - 8 * r.width / S; return [(e.clientX - r.left - r.width / 2) / R, (e.clientY - r.top - r.height / 2) / R]; };
  cP.addEventListener('pointerdown', (e) => { st.drag = { p: toDisk(e), moved: false }; cP.setPointerCapture(e.pointerId); });
  cP.addEventListener('pointermove', (e) => {
    if (!st.drag) return;
    const q = toDisk(e), p = st.drag.p;
    if (q[0] ** 2 + q[1] ** 2 > 0.96) return;
    const a = [(q[0] - p[0]) * 0.9, (q[1] - p[1]) * 0.9];
    if (Math.hypot(a[0], a[1]) < 1e-4) return;
    st.drag.moved = true;
    st.pos = st.pos.map((z) => mobius(z, a)); st.drag.p = q; drawDisk(cP, 'poi');
  });
  cP.addEventListener('pointerup', (e) => {
    const d = st.drag; st.drag = null;
    if (d && !d.moved) {
      const p = toDisk(e); let best = null, bd = 0.05;
      st.pos.forEach((z, i) => { const dd = Math.hypot(z[0] - p[0], z[1] - p[1]); if (dd < bd) { bd = dd; best = i; } });
      st.sel = best; drawAll();
      const info = document.getElementById('hySel');
      if (best !== null) info.textContent = U.t(`Selected node at depth ${st.tree[best].d}. Colours show tree (graph) distance from it; both disks use the same colouring, but only the Poincaré disk keeps far branches far apart.`, `Выбрана вершина на глубине ${st.tree[best].d}. Цвет показывает расстояние по дереву (графу) от неё; раскраска на обоих дисках одинакова, но только диск Пуанкаре держит далёкие ветви далеко друг от друга.`);
    }
  });
  document.getElementById('hyB').addEventListener('input', (e) => { st.b = +e.target.value; document.getElementById('hyBv').textContent = st.b; buildTree(); drawAll(); });
  document.getElementById('hyD').addEventListener('input', (e) => { st.D = +e.target.value; document.getElementById('hyDv').textContent = st.D; buildTree(); drawAll(); });
  document.getElementById('hyReset').addEventListener('click', () => { st.pos = st.tree.map((n) => n.poi.slice()); drawAll(); });

  /* ---------- tangent-space aggregation ---------- */
  const cT = document.getElementById('hyTan'), cS = document.getElementById('hyTanSpace');
  const pts = [[0.55, 0.62], [0.82, 0.18], [0.72, -0.48], [0.15, 0.4]];
  const curv = { c: 1 };
  const log0 = (x, c) => { const n = Math.hypot(x[0], x[1]) || 1e-9, rc = Math.sqrt(c); const s = Math.atanh(Math.min(rc * n, 1 - 1e-7)) / (rc * n); return [x[0] * s, x[1] * s]; };
  const exp0 = (v, c) => { const n = Math.hypot(v[0], v[1]) || 1e-9, rc = Math.sqrt(c); const s = Math.tanh(rc * n) / (rc * n); return [v[0] * s, v[1] * s]; };
  let dragT = null;
  function drawTan() {
    const c = curv.c, maxR = 1 / Math.sqrt(c);
    const ctx = U.setupCanvas(cT, S, S), R = S / 2 - 8, X = (p) => [S / 2 + p[0] / maxR * R, S / 2 + p[1] / maxR * R];
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = U.css('--panel'); ctx.strokeStyle = U.css('--muted'); ctx.beginPath(); ctx.arc(S / 2, S / 2, R, 0, 7); ctx.fill(); ctx.stroke();
    const L = pts.map((p) => log0(p, c)), m = [L.reduce((s, v) => s + v[0], 0) / L.length, L.reduce((s, v) => s + v[1], 0) / L.length];
    const agg = exp0(m, c), eu = [pts.reduce((s, v) => s + v[0], 0) / pts.length, pts.reduce((s, v) => s + v[1], 0) / pts.length];
    ctx.strokeStyle = U.css('--muted'); ctx.setLineDash([3, 3]); ctx.beginPath(); pts.forEach((p) => { ctx.moveTo(S / 2, S / 2); ctx.lineTo(...X(p)); }); ctx.stroke(); ctx.setLineDash([]);
    pts.forEach((p) => { ctx.fillStyle = U.css('--accent'); ctx.beginPath(); ctx.arc(...X(p), 7, 0, 7); ctx.fill(); });
    ctx.fillStyle = U.css('--muted'); ctx.beginPath(); ctx.arc(...X(eu), 6, 0, 7); ctx.fill();
    ctx.fillStyle = U.css('--red'); ctx.beginPath(); ctx.arc(...X(agg), 7, 0, 7); ctx.fill();
    ctx.fillStyle = U.css('--ink'); ctx.font = '11px system-ui'; ctx.fillText('o', S / 2 - 3, S / 2 + 4);
    // tangent space panel
    const ctx2 = U.setupCanvas(cS, S, S), sc = (S / 2 - 16) / Math.max(3, ...L.map((v) => Math.hypot(v[0], v[1]))), Y = (v) => [S / 2 + v[0] * sc, S / 2 + v[1] * sc];
    ctx2.clearRect(0, 0, S, S);
    ctx2.strokeStyle = U.css('--line'); ctx2.beginPath(); ctx2.moveTo(0, S / 2); ctx2.lineTo(S, S / 2); ctx2.moveTo(S / 2, 0); ctx2.lineTo(S / 2, S); ctx2.stroke();
    ctx2.strokeStyle = U.css('--muted'); ctx2.setLineDash([2, 4]); ctx2.beginPath(); ctx2.arc(S / 2, S / 2, sc / Math.sqrt(c), 0, 7); ctx2.stroke(); ctx2.setLineDash([]);
    L.forEach((v) => { ctx2.strokeStyle = U.css('--accent'); ctx2.lineWidth = 2; ctx2.beginPath(); ctx2.moveTo(S / 2, S / 2); ctx2.lineTo(...Y(v)); ctx2.stroke(); ctx2.fillStyle = U.css('--accent'); ctx2.beginPath(); ctx2.arc(...Y(v), 4, 0, 7); ctx2.fill(); });
    ctx2.fillStyle = U.css('--red'); ctx2.beginPath(); ctx2.arc(...Y(m), 6, 0, 7); ctx2.fill();
    ctx2.fillStyle = U.css('--muted'); ctx2.font = '11px system-ui'; ctx2.fillText(U.t('tangent space T₀ (ordinary ℝ²)', 'касательное пространство T₀ (обычное ℝ²)'), 8, 16);
    const hd = (a, b) => { const z = [a[0] * Math.sqrt(c), a[1] * Math.sqrt(c)], w = [b[0] * Math.sqrt(c), b[1] * Math.sqrt(c)]; return hdist(z, w) / Math.sqrt(c); };
    const sumH = (q) => pts.reduce((s, p) => s + hd(p, q) ** 2, 0);
    document.getElementById('hyTanOut').innerHTML = U.t(`Σ d<sub>ℍ</sub>(x<sub>j</sub>, ·)² to the neighbours: tangent aggregate <b>${U.fmt(sumH(agg))}</b>, plain Euclidean average <b>${U.fmt(sumH(eu))}</b>. Lower = more central in the hyperbolic metric. Neither point is the exact hyperbolic centre (Fréchet mean): the layer averages in the tangent space at the origin, which is cheap and differentiable but distorts more the farther the neighbours are from o.`,
      `Σ d<sub>ℍ</sub>(x<sub>j</sub>, ·)² до соседей: агрегат в касательном пространстве <b>${U.fmt(sumH(agg))}</b>, обычное евклидово среднее <b>${U.fmt(sumH(eu))}</b>. Меньше — значит центральнее в гиперболической метрике. Ни одна из точек не является точным гиперболическим центром (средним Фреше): слой усредняет в касательном пространстве в начале координат — это дёшево и дифференцируемо, но искажение растёт по мере удаления соседей от o.`);
  }
  const toT = (e) => { const r = cT.getBoundingClientRect(), R = r.width / 2 - 8 * r.width / S, maxR = 1 / Math.sqrt(curv.c); return [(e.clientX - r.left - r.width / 2) / R * maxR, (e.clientY - r.top - r.height / 2) / R * maxR]; };
  cT.addEventListener('pointerdown', (e) => { const p = toT(e); let best = null, bd = 0.12 / Math.sqrt(curv.c); pts.forEach((q, i) => { const d = Math.hypot(q[0] - p[0], q[1] - p[1]); if (d < bd) { bd = d; best = i; } }); dragT = best; if (best !== null) cT.setPointerCapture(e.pointerId); });
  cT.addEventListener('pointermove', (e) => { if (dragT === null) return; const p = toT(e), n = Math.hypot(p[0], p[1]), mx = 0.985 / Math.sqrt(curv.c); pts[dragT] = n > mx ? [p[0] / n * mx, p[1] / n * mx] : p; drawTan(); });
  cT.addEventListener('pointerup', () => { dragT = null; });

  U.onTheme.push(() => { if (st.tree) { drawAll(); drawTan(); } });
  U.onLang.push(() => { if (st.tree) { stats(); drawTan(); document.getElementById('hySel').textContent = ''; } });
  buildTree(); drawAll(); drawTan();
})();
