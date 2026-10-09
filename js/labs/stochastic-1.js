/* Labs: Stochastic processes, part 1 — Markov chains, classification, stationarity & mixing,
   random walks, branching processes, hidden Markov models. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;

  /* ---------------- shared helpers ---------------- */

  const SER = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'];
  const vecMat = (v, P) => { const K = P.length, out = new Array(K).fill(0); for (let i = 0; i < K; i++) { const vi = v[i]; if (vi) for (let j = 0; j < K; j++) out[j] += vi * P[i][j]; } return out; };
  const tv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]); return s / 2; };
  const normRows = W => W.map(r => { const s = S.sum(r); return s > 0 ? r.map(x => x / s) : r.map((_, j) => 0); });

  // A stationary distribution: solve pi (P - I) = 0, sum = 1; fall back to Cesaro averaging.
  function stationary(P) {
    const K = P.length;
    try {
      const A = S.transpose(P).map((r, i) => r.map((v, j) => v - (i === j ? 1 : 0)));
      A[K - 1] = new Array(K).fill(1);
      const b = new Array(K).fill(0); b[K - 1] = 1;
      const pi = S.solve(A, b);
      if (pi.every(v => Number.isFinite(v) && v > -1e-9)) return pi.map(v => Math.max(0, v));
    } catch (e) { /* singular: reducible chain */ }
    let mu = new Array(K).fill(1 / K), acc = new Array(K).fill(0);
    for (let n = 0; n < 4000; n++) { for (let j = 0; j < K; j++) acc[j] += mu[j]; mu = vecMat(mu, P); }
    return acc.map(v => v / 4000);
  }

  // Three-state chain from six off-diagonal sliders (p01,p02,p10,p12,p20,p21); rows renormalised if they overflow.
  function threeState(off) {
    const P = [[0, off[0], off[1]], [off[2], 0, off[3]], [off[4], off[5], 0]];
    let clipped = false;
    for (let i = 0; i < 3; i++) {
      const s = P[i].reduce((a, b) => a + b, 0);
      if (s > 1) { clipped = true; for (let j = 0; j < 3; j++) P[i][j] /= s; }
      else P[i][i] = 1 - s;
    }
    return { P, clipped };
  }
  function sixSliders(controls, init, onInput) {
    const names = ['$p_{12}$', '$p_{13}$', '$p_{21}$', '$p_{23}$', '$p_{31}$', '$p_{32}$'];
    return names.map((nm, k) => UI.slider(controls, { label: nm, min: 0, max: 1, step: 0.01, value: init[k], onInput }));
  }

  // Draw a transition diagram. pos: [[x,y]], nodeColor: array of tokens, hl: {from,to} edge to highlight.
  function drawChain(p, P, pos, o = {}) {
    const K = P.length, rpx = o.rpx || 17;
    const u = (p.o.x[1] - p.o.x[0]) / p.iw, rD = rpx * u;
    const cx = S.mean(pos.map(q => q[0])), cy = S.mean(pos.map(q => q[1]));
    const head = (x0, y0, x1, y1, col, w, op) => {
      const L = Math.hypot(x1 - x0, y1 - y0) || 1, ux = (x1 - x0) / L, uy = (y1 - y0) / L;
      const hl = (7 + 1.5 * w) * u, hw = (3.5 + 0.8 * w) * u;
      p.segment(x0, y0, x1 - ux * hl * 0.8, y1 - uy * hl * 0.8, { color: col, width: w, opacity: op });
      p.polygon([[x1, y1], [x1 - ux * hl - uy * hw, y1 - uy * hl + ux * hw], [x1 - ux * hl + uy * hw, y1 - uy * hl - ux * hw]], { color: col, opacity: Math.min(1, op + 0.15) });
    };
    for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
      if (i === j || P[i][j] < 1e-3) continue;
      const [x0, y0] = pos[i], [x1, y1] = pos[j];
      const L = Math.hypot(x1 - x0, y1 - y0), ux = (x1 - x0) / L, uy = (y1 - y0) / L, nx = -uy, ny = ux;
      const off = P[j][i] >= 1e-3 ? 7 * u : 0;
      const isH = o.hl && o.hl.from === i && o.hl.to === j;
      const col = isH ? 'accent' : (o.edgeColor || 'muted');
      const w = 0.8 + 4 * P[i][j];
      head(x0 + ux * rD + nx * off, y0 + uy * rD + ny * off, x1 - ux * (rD + 2 * u) + nx * off, y1 - uy * (rD + 2 * u) + ny * off, col, isH ? w + 1 : w, isH ? 1 : 0.35 + 0.6 * P[i][j]);
      if (o.labels !== false && K <= 5) p.text(x0 + (x1 - x0) * 0.5 + nx * (off + 11 * u), y0 + (y1 - y0) * 0.5 + ny * (off + 11 * u), P[i][j].toFixed(2), { anchor: 'middle', dy: 4, size: 10, color: 'muted' });
    }
    for (let i = 0; i < K; i++) {
      if (P[i][i] < 1e-3) continue;
      const [x, y] = pos[i];
      let ox = x - cx, oy = y - cy; const L = Math.hypot(ox, oy) || 1; ox /= L; oy /= L;
      const rr = 10 * u, c0x = x + ox * (rD + rr * 0.6), c0y = y + oy * (rD + rr * 0.6);
      const isH = o.hl && o.hl.from === i && o.hl.to === i;
      p.line(S.linspace(0, 2 * Math.PI, 40).map(t => [c0x + rr * Math.cos(t), c0y + rr * Math.sin(t)]), { color: isH ? 'accent' : 'muted', width: isH ? 2.5 : 0.8 + 3 * P[i][i], opacity: isH ? 1 : 0.4 + 0.5 * P[i][i] });
      if (o.labels !== false && K <= 5) p.text(c0x + ox * (rr + 12 * u), c0y + oy * (rr + 12 * u), P[i][i].toFixed(2), { anchor: 'middle', dy: 4, size: 10, color: 'muted' });
    }
    for (let i = 0; i < K; i++) {
      const c = (o.nodeColor && o.nodeColor[i]) || 's1';
      p.dot(pos[i][0], pos[i][1], { r: rpx, color: c, opacity: o.active === i ? 0.9 : 0.22, stroke: c, strokeWidth: 2 });
      p.text(pos[i][0], pos[i][1], (o.names && o.names[i]) || String(i + 1), { anchor: 'middle', dy: 4, size: 12, weight: 600, color: o.active === i ? 'surface' : 'ink' });
    }
  }
  const circlePos = (K, r = 1, rot = -Math.PI / 2) => Array.from({ length: K }, (_, i) => [r * Math.cos(rot + 2 * Math.PI * i / K), -r * Math.sin(rot + 2 * Math.PI * i / K)]);
  const fmtVec = v => '(' + v.map(x => S.fmt(x, 3)).join(', ') + ')';

  /* ================= markov-chains ================= */

  const TRI_PRESETS = {
    weather: { label: 'Weather: sun / cloud / rain', off: [0.2, 0.1, 0.3, 0.3, 0.2, 0.4] },
    sticky: { label: 'Sticky (rare switches)', off: [0.03, 0.02, 0.02, 0.03, 0.02, 0.03] },
    lazy: { label: 'Well mixed', off: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3] },
    nearcycle: { label: 'Near-cycle 1→2→3→1', off: [0.9, 0, 0, 0.9, 0.9, 0] },
    cycle: { label: 'Deterministic cycle (period 3)', off: [1, 0, 0, 1, 1, 0] },
    absorb: { label: 'State 3 absorbing', off: [0.3, 0.2, 0.3, 0.2, 0, 0] },
  };

  Atlas.lab('markov-chains-simulator', {
    title: 'Simulate a three-state chain',
    blurb: 'Set the off-diagonal transition probabilities (the diagonal is whatever is left over) and press play. Left: the transition graph, with the walker\'s current state filled and its last move highlighted. Right: the fraction of time spent in each state against the stationary distribution $\\pi$. Try the sticky preset: occupation frequencies converge, but slowly, because the chain lingers.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pre = UI.select(controls, { label: 'Preset', options: Object.keys(TRI_PRESETS).map(k => ({ value: k, label: TRI_PRESETS[k].label })), value: 'weather', onChange: v => { TRI_PRESETS[v].off.forEach((x, k) => sl[k].set(x)); rebuild(); } });
      const sl = sixSliders(controls, TRI_PRESETS.weather.off, () => rebuild());
      const speed = UI.slider(controls, { label: 'Steps per second', min: 1, max: 1000, value: 8, log: true, fmt: v => Math.round(v) });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop);
      UI.button(row, 'Step', () => { step(); draw(); });
      UI.button(row, 'Reset counts', () => { reset(); draw(); });
      stage.classList.add('two');
      const pG = new Plot(stage, { width: 420, height: 270, x: [-1.45, 1.45], y: [-1.3, 1.3], equal: true, grid: false, xticks: 0, yticks: 0, title: 'Transition graph' });
      const pO = new Plot(stage, { width: 420, height: 270, x: [0, 3], y: [0, 1], xticks: 0, ylabel: 'fraction of time', title: 'Occupation frequency vs π' });
      const sN = UI.stat(stats, 'Steps $n$');
      const sPi = UI.stat(stats, 'Stationary $\\pi$');
      const sErr = UI.stat(stats, '$\\max_i |\\hat\\pi_i - \\pi_i|$');
      const nt = UI.note(stats, '');
      const pos = [[0, 1.0], [-0.95, -0.62], [0.95, -0.62]];
      let P, pi, x, counts, n, last, acc = 0;
      function rebuild() { const r = threeState(sl.map(s => s.get())); P = r.P; pi = stationary(P); nt.set(r.clipped ? 'A row\'s off-diagonal entries summed to more than 1, so that row was rescaled (its diagonal is 0).' : 'Rows sum to one: $p_{ii} = 1 - \\sum_{j\\ne i}p_{ij}$.'); draw(); }
      function reset() { x = 0; counts = [0, 0, 0]; n = 0; last = null; }
      function step() { const y = ctx.rng.categorical(P[x]); last = { from: x, to: y }; x = y; counts[x]++; n++; }
      function tick(dt) {
        acc += dt * speed.get();
        let k = 0;
        while (acc >= 1 && k < 4000) { step(); acc -= 1; k++; }
        if (acc > 1) acc = 0;
        if (k) draw();
      }
      function draw() {
        pG.clear();
        drawChain(pG, P, pos, { active: x, hl: last, nodeColor: ['s1', 's2', 's3'], names: ['1', '2', '3'] });
        pO.clear();
        const fr = counts.map(c => (n ? c / n : 0));
        for (let i = 0; i < 3; i++) {
          pO.bars([{ x0: i + 0.2, x1: i + 0.8, y: fr[i] }], { color: SER[i], opacity: 0.6 });
          pO.segment(i + 0.12, pi[i], i + 0.88, pi[i], { color: 'ink', width: 2.5 });
          pO.text(i + 0.5, 0, 'state ' + (i + 1), { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        }
        pO.legend([{ label: 'empirical', color: 's1', fill: true }, { label: 'π', color: 'ink' }]);
        sN.set(String(n));
        sPi.set(fmtVec(pi));
        sErr.set(n ? S.fmt(Math.max(...fr.map((f, i) => Math.abs(f - pi[i])))) : '—', n ? `≈ O(1/√n) = ${S.fmt(1 / Math.sqrt(n), 3)}` : '');
      }
      reset(); rebuild();
    },
  });

  const NSTEP_CHAINS = {
    two: { label: 'Two-state (α, β sliders)' },
    weather: { label: 'Weather, 3 states', P: [[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.4, 0.4]] },
    ehren: { label: 'Ehrenfest, N = 4 (period 2)', P: [0, 1, 2, 3, 4].map(k => [0, 1, 2, 3, 4].map(j => (j === k - 1 ? k / 4 : j === k + 1 ? (4 - k) / 4 : 0))) },
    lazyehr: { label: 'Lazy Ehrenfest, N = 4', P: [0, 1, 2, 3, 4].map(k => [0, 1, 2, 3, 4].map(j => (j === k ? 0.5 : j === k - 1 ? k / 8 : j === k + 1 ? (4 - k) / 8 : 0))) },
    ruin: { label: "Gambler's ruin on {0,…,4} (absorbing ends)", P: [0, 1, 2, 3, 4].map(k => [0, 1, 2, 3, 4].map(j => ((k === 0 || k === 4) ? (j === k ? 1 : 0) : (Math.abs(j - k) === 1 ? 0.5 : 0)))) },
  };

  Atlas.lab('markov-chains-nstep', {
    title: 'Chapman–Kolmogorov: the distribution $\\delta_i P^n$ and the matrix $P^n$',
    blurb: 'Left: the law of $X_k$ for $k = 0,\\dots,50$ starting from a fixed state, one curve per state. Right: the full matrix $P^n$ at the chosen $n$ (row = start, column = destination). When the rows of $P^n$ become identical the chain has forgotten where it started. The Ehrenfest chain never forgets (period 2); the absorbing chain forgets only partially.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ch = UI.select(controls, { label: 'Chain', options: Object.keys(NSTEP_CHAINS).map(k => ({ value: k, label: NSTEP_CHAINS[k].label })), value: 'two', onChange: draw });
      const al = UI.slider(controls, { label: 'Two-state $\\alpha = p_{01}$', min: 0, max: 1, step: 0.01, value: 0.2, onInput: draw });
      const be = UI.slider(controls, { label: 'Two-state $\\beta = p_{10}$', min: 0, max: 1, step: 0.01, value: 0.1, onInput: draw });
      const st = UI.slider(controls, { label: 'Start state $i$', min: 0, max: 4, step: 1, value: 0, onInput: draw });
      const nn = UI.slider(controls, { label: 'Power $n$', min: 0, max: 50, step: 1, value: 3, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 50], y: [0, 1], xlabel: 'k', ylabel: 'P(X_k = j | X_0 = i)', title: 'Law of X_k from start i' });
      const pM = new Plot(stage, { width: 420, height: 270, x: [0, 5], y: [0, 5], grid: false, xticks: 0, yticks: 0, margin: { t: 28, r: 12, b: 26, l: 34 }, title: 'Matrix P^n' });
      const sRow = UI.stat(stats, 'Row $i$ of $P^n$');
      const sDob = UI.stat(stats, 'Max TV between rows of $P^n$');
      const sPi = UI.stat(stats, 'A stationary $\\pi$');
      function getP() {
        if (ch.get() === 'two') { const a = al.get(), b = be.get(); return [[1 - a, a], [b, 1 - b]]; }
        return NSTEP_CHAINS[ch.get()].P;
      }
      function draw() {
        const P = getP(), K = P.length, i0 = Math.min(Math.round(st.get()), K - 1), n = Math.round(nn.get());
        const names = ch.get() === 'weather' ? ['1', '2', '3'] : P.map((_, k) => String(k));
        // law over time
        const traj = []; let mu = P.map((_, k) => (k === i0 ? 1 : 0));
        for (let k = 0; k <= 50; k++) { traj.push(mu); mu = vecMat(mu, P); }
        pL.clear();
        pL.vline(n, { color: 'muted' });
        for (let j = 0; j < K; j++) pL.line(traj.map((m, k) => [k, m[j]]), { color: SER[j], width: 2 });
        pL.legend(names.map((nm, j) => ({ label: 'j = ' + nm, color: SER[j] })));
        // matrix power
        let Pn = S.identity(K);
        for (let k = 0; k < n; k++) Pn = S.matmul(Pn, P);
        pM.setDomain([0, K], [0, K]).clear();
        for (let r = 0; r < K; r++) for (let c = 0; c < K; c++) {
          const v = Pn[r][c], y0 = K - r - 1;
          pM.rect(c, y0, c + 1, y0 + 1, { color: r === i0 ? 'accent' : 's1', opacity: 0.06 + 0.8 * v, stroke: 'grid' });
          pM.text(c + 0.5, y0 + 0.5, v.toFixed(K > 3 ? 2 : 3), { anchor: 'middle', dy: 4, size: K > 3 ? 11 : 13, color: v > 0.55 ? 'surface' : 'ink' });
        }
        for (let k = 0; k < K; k++) { pM.text(-0.08, K - k - 0.5, names[k], { anchor: 'end', dy: 4, size: 11, color: 'muted' }); pM.text(k + 0.5, 0, names[k], { anchor: 'middle', dy: 15, size: 11, color: 'muted' }); }
        let dob = 0;
        for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) dob = Math.max(dob, tv(Pn[a], Pn[b]));
        sRow.set(fmtVec(Pn[i0]), `n = ${n}, start i = ${names[i0]}`);
        sDob.set(S.fmt(dob), dob < 0.01 ? 'rows nearly identical: start forgotten' : 'start still matters');
        sPi.set(fmtVec(stationary(P)), ch.get() === 'ruin' ? 'not unique: any mix of δ₀ and δ₄' : '');
      }
      draw();
    },
  });

  /* ================= chain-classification ================= */

  // Graph analysis: communicating classes, closedness, periods.
  function classify(P) {
    const K = P.length, R = P.map((r, i) => r.map((v, j) => v > 1e-12 || i === j));
    for (let k = 0; k < K; k++) for (let i = 0; i < K; i++) if (R[i][k]) for (let j = 0; j < K; j++) if (R[k][j]) R[i][j] = true;
    const cls = new Array(K).fill(-1), classes = [];
    for (let i = 0; i < K; i++) {
      if (cls[i] >= 0) continue;
      const members = [];
      for (let j = 0; j < K; j++) if (R[i][j] && R[j][i]) { cls[j] = classes.length; members.push(j); }
      classes.push({ members });
    }
    for (const c of classes) {
      const id = cls[c.members[0]];
      c.closed = c.members.every(i => P[i].every((v, j) => v <= 1e-12 || cls[j] === id));
      // period via BFS levels inside the class
      const lev = {}; lev[c.members[0]] = 0; const q = [c.members[0]]; let g = 0;
      while (q.length) { const u = q.shift(); for (let v = 0; v < K; v++) if (P[u][v] > 1e-12 && cls[v] === id && lev[v] == null) { lev[v] = lev[u] + 1; q.push(v); } }
      const gcd = (a, b) => (b ? gcd(b, a % b) : a);
      for (const u of c.members) for (const v of c.members) if (P[u][v] > 1e-12) g = gcd(g, Math.abs(lev[u] + 1 - lev[v]));
      c.period = g; // 0 means no cycle inside the class
    }
    return { cls, classes };
  }

  const CLASS_CHAINS = {
    irred: { label: 'Irreducible, aperiodic (cycles of length 2 and 3)', W: [[0, 1, 0, 0, 0], [0.5, 0, 0.5, 0, 0], [0, 0, 0, 1, 0], [0, 0.4, 0, 0, 0.6], [1, 0, 0, 0, 0]] },
    period3: { label: 'Irreducible, period 3', W: [[0, 0, 0.7, 0.3, 0, 0], [0, 0, 0.2, 0.8, 0, 0], [0, 0, 0, 0, 0.5, 0.5], [0, 0, 0, 0, 0.1, 0.9], [0.6, 0.4, 0, 0, 0, 0], [0.3, 0.7, 0, 0, 0, 0]] },
    bip: { label: 'Bipartite 4-cycle, period 2', W: [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]] },
    red: { label: 'Reducible: transient states + two closed classes', W: [[0, 0.5, 0, 0.5, 0, 0, 0], [0.3, 0, 0.4, 0, 0, 0.3, 0], [0, 0.5, 0.5, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0.5, 0.5, 0, 0], [0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0]] },
    ruin: { label: "Gambler's ruin, 6 states: absorbing ends", W: [0, 1, 2, 3, 4, 5].map(k => [0, 1, 2, 3, 4, 5].map(j => ((k === 0 || k === 5) ? (j === k ? 1 : 0) : (Math.abs(j - k) === 1 ? 0.5 : 0)))) },
    random: { label: 'Random sparse chain (press New)' },
  };

  Atlas.lab('chain-classification-explorer', {
    title: 'Communicating classes, recurrence and period',
    blurb: 'The graph is coloured by communicating class: filled colours are closed (recurrent) classes, grey nodes are transient. Right: the return probabilities $p^{(n)}_{ii}$ for a chosen state. Zeros at non-multiples of the period reveal periodicity; for a transient state the stems die out and their sum $\\sum_n p^{(n)}_{ii} = f_i/(1-f_i)$ is finite.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ch = UI.select(controls, { label: 'Chain', options: Object.keys(CLASS_CHAINS).map(k => ({ value: k, label: CLASS_CHAINS[k].label })), value: 'red', onChange: () => { if (ch.get() === 'random' && !randW) randomChain(); draw(); } });
      const si = UI.slider(controls, { label: 'State $i$ for $p^{(n)}_{ii}$', min: 1, max: 7, step: 1, value: 1, onInput: draw });
      UI.button(controls, 'New random chain', () => { randomChain(); ch.set('random'); draw(); });
      stage.classList.add('two');
      const pG = new Plot(stage, { width: 420, height: 270, x: [-1.45, 1.45], y: [-1.3, 1.3], equal: true, grid: false, xticks: 0, yticks: 0, title: 'Classes (filled = closed/recurrent)' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 31], y: [0, 1], xlabel: 'n', ylabel: 'p_ii^(n)', title: 'Return probabilities' });
      const sCls = UI.note(stats, '');
      const sState = UI.stat(stats, 'State $i$');
      const sSum = UI.stat(stats, '$\\sum_{n\\le 30} p^{(n)}_{ii}$');
      const sF = UI.stat(stats, 'Return probability $f_i$');
      let randW = null;
      function randomChain() {
        const K = 7, r = ctx.rng;
        randW = Array.from({ length: K }, (_, i) => {
          const row = new Array(K).fill(0), deg = 1 + Math.floor(r.u() * 2.2);
          for (let d = 0; d < deg; d++) row[Math.floor(r.u() * K)] += 0.2 + r.u();
          return row;
        });
      }
      function draw() {
        const P = normRows(ch.get() === 'random' ? (randW || (randomChain(), randW)) : CLASS_CHAINS[ch.get()].W);
        const K = P.length, { cls, classes } = classify(P);
        const i = Math.min(Math.round(si.get()), K) - 1;
        let ci = 0; const colOf = classes.map(c => (c.closed ? SER[ci++ % 8] : 'muted'));
        pG.clear();
        drawChain(pG, P, circlePos(K, 1.0), { nodeColor: cls.map(c => colOf[c]), active: i, labels: K <= 5 });
        // return probabilities
        const ret = []; let mu = P.map((_, k) => (k === i ? 1 : 0));
        for (let n = 1; n <= 30; n++) { mu = vecMat(mu, P); ret.push([n, mu[i]]); }
        pR.clear();
        pR.stems(ret, { color: colOf[cls[i]] === 'muted' ? 'muted' : colOf[cls[i]], r: 3 });
        const c = classes[cls[i]];
        const sum = S.sum(ret.map(r => r[1]));
        // exact return probability for transient states via the fundamental matrix on transient states
        let f = 1, tot = Infinity;
        if (!c.closed) {
          const T = cls.map((k, s) => (classes[k].closed ? -1 : s)).filter(s => s >= 0);
          const Q = T.map(a => T.map(b => P[a][b]));
          try { const N = S.inv(Q.map((r, a) => r.map((v, b) => (a === b ? 1 : 0) - v))); const Nii = N[T.indexOf(i)][T.indexOf(i)]; tot = Nii - 1; f = tot / Nii; } catch (e) { f = NaN; }
        }
        sCls.set(classes.map((cc, k) => `<span style="color:var(--${colOf[k] === 'muted' ? 'muted' : colOf[k]})">●</span> {${cc.members.map(m => m + 1).join(', ')}}: ${cc.closed ? 'closed, recurrent' : 'transient'}${cc.period ? ', period ' + cc.period : ', no cycle'}`).join('<br>'));
        sState.set(String(i + 1), `${c.closed ? 'recurrent' : 'transient'}, period ${c.period || '—'}`);
        sSum.set(S.fmt(sum), c.closed ? 'diverges as n → ∞' : `total over all n: ${S.fmt(tot)}`);
        sF.set(c.closed ? '1' : S.fmt(f), c.closed ? 'returns are certain' : 'f/(1−f) = Σ p_ii^(n)');
      }
      draw();
    },
  });

  Atlas.lab('chain-classification-return', {
    title: 'Positive, null, or transient: return times of a reflecting walk',
    blurb: 'A walk on $\\{0,1,2,\\dots\\}$ steps up with probability $p$ and down with probability $1-p$ (staying put at 0). We simulate many excursions from 0, each cut off at a horizon $H$, and plot the survival function $\\P_0(T_0 > t)$ on log–log axes. $p<\\tfrac12$: light tail and finite mean. $p=\\tfrac12$: the $t^{-1/2}$ power law of null recurrence. $p>\\tfrac12$: a plateau at the escape probability $2p-1$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pp = UI.slider(controls, { label: 'Up probability $p$', min: 0.3, max: 0.7, step: 0.01, value: 0.5, onChange: restart });
      const H = UI.slider(controls, { label: 'Horizon $H$', min: 100, max: 20000, value: 5000, log: true, fmt: v => Math.round(v), onChange: restart });
      const M = UI.slider(controls, { label: 'Excursions', min: 200, max: 4000, step: 100, value: 1500, onChange: restart });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Simulate', '❚❚ Pause']);
      UI.button(row, 'Restart', () => { restart(); loop.start(); });
      const pS = new Plot(stage, { width: 640, height: 300, x: [1, 5000], y: [1e-4, 1], logx: true, logy: true, xlabel: 't', ylabel: 'P(T₀ > t)', title: 'Survival function of the return time to 0' });
      const sCls = UI.stat(stats, 'Classification', { accent: 'accent' });
      const sDone = UI.stat(stats, 'Excursions simulated');
      const sMean = UI.stat(stats, 'Mean of $\\min(T_0, H)$');
      const sTh = UI.stat(stats, 'Theory $\\E_0 T_0$');
      const sCens = UI.stat(stats, 'Not returned by $H$');
      let times, done, exact;
      function exactSurvival(p, tmax) { // substochastic iteration of the chain killed at 0, started after the first step
        const out = [[1, p]]; let v = new Float64Array(tmax + 3); v[1] = p;
        for (let t = 2; t <= tmax; t++) {
          const w = new Float64Array(tmax + 3); let mass = 0;
          for (let k = 1; k <= Math.min(t, tmax + 1); k++) { const a = v[k]; if (!a) continue; w[k + 1] += a * p; if (k > 1) w[k - 1] += a * (1 - p); }
          v = w; for (let k = 1; k <= tmax + 1; k++) mass += v[k];
          out.push([t, mass]);
        }
        return out;
      }
      function restart() { times = []; done = 0; exact = exactSurvival(pp.get(), Math.min(1500, Math.round(H.get()))); draw(); }
      function tick() {
        const p = pp.get(), h = Math.round(H.get()), m = Math.round(M.get()), u = ctx.rng.u;
        let budget = 400000;
        while (budget > 0 && done < m) {
          let x = 0, t = 0;
          do { t++; if (x === 0) x = u() < p ? 1 : 0; else x += u() < p ? 1 : -1; } while (x !== 0 && t < h);
          times.push(x === 0 ? t : Infinity); done++; budget -= t;
        }
        draw();
        if (done >= m) return false;
      }
      function draw() {
        const p = pp.get(), h = Math.round(H.get());
        pS.setDomain([1, h], [1e-4, 1]).clear();
        if (exact) pS.line(exact.filter(e => e[0] <= h && e[1] > 1e-5), { color: 'ink', width: 1.5, dash: '5 4', opacity: 0.8 });
        if (p === 0.5 || Math.abs(p - 0.5) < 1e-9) pS.line([[1, 0.8], [h, 0.8 / Math.sqrt(h)]], { color: 'muted', width: 1, dash: '2 3' });
        if (p > 0.5) pS.hline(2 * p - 1, { color: 's3', width: 1.5 });
        if (times.length) {
          const s = times.slice().sort((a, b) => a - b), n = s.length, pts = [];
          let idx = 0;
          for (const t of S.linspace(0, Math.log10(h), 120).map(v => 10 ** v)) { while (idx < n && s[idx] <= t) idx++; const sv = (n - idx) / n; if (sv > 0) pts.push([t, sv]); }
          pS.line(pts, { color: 's1', width: 2.5 });
        }
        pS.legend([{ label: 'simulated', color: 's1' }, { label: 'exact (t ≤ 1500)', color: 'ink', dash: true }].concat(p > 0.5 ? [{ label: 'escape prob. 2p − 1', color: 's3' }] : Math.abs(p - 0.5) < 1e-9 ? [{ label: 'slope −1/2', color: 'muted', dash: true }] : []));
        const q = 1 - p;
        sCls.set(p < 0.5 - 1e-9 ? 'positive recurrent' : p > 0.5 + 1e-9 ? 'transient' : 'null recurrent');
        sDone.set(String(done));
        const fin = times.filter(Number.isFinite);
        sMean.set(times.length ? S.fmt(S.mean(times.map(t => (Number.isFinite(t) ? t : h)))) : '—', 'grows with H unless positive recurrent');
        sTh.set(p < 0.5 - 1e-9 ? S.fmt(q / (q - p)) : '∞', p < 0.5 - 1e-9 ? '= 1/π₀ = q/(q − p)' : '');
        sCens.set(times.length ? S.fmt((times.length - fin.length) / times.length) : '—', p > 0.5 ? `→ 2p − 1 = ${S.fmt(2 * p - 1)} as H → ∞` : '→ 0 as H → ∞');
      }
      restart();
      loop.start();
    },
  });

  /* ================= stationary-distributions ================= */

  function eig3(P) { // eigenvalues of a 3x3 stochastic matrix: 1 and roots of z^2 - (tr-1) z + det
    const tr = P[0][0] + P[1][1] + P[2][2], d = S.det(P), b = tr - 1, disc = b * b - 4 * d;
    if (disc >= 0) { const r = Math.sqrt(disc); return [[1, 0], [(b + r) / 2, 0], [(b - r) / 2, 0]]; }
    return [[1, 0], [b / 2, Math.sqrt(-disc) / 2], [b / 2, -Math.sqrt(-disc) / 2]];
  }

  Atlas.lab('stationary-distributions-convergence', {
    title: 'Convergence in total variation and the eigenvalues of $P$',
    blurb: 'Left: $d(n) = \\max_i\\|P^n(i,\\cdot)-\\pi\\|_{TV}$ on a log scale, with the lower bound $\\tfrac12|\\lambda_2|^n$ (and, for reversible chains, the spectral upper bound). Right: the eigenvalues of $P$ in the complex plane. The slope of $\\log d(n)$ is $\\log|\\lambda_2|$. Move toward the cycle preset and watch two eigenvalues approach the unit circle and convergence stall.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pre = UI.select(controls, { label: 'Preset', options: Object.keys(TRI_PRESETS).map(k => ({ value: k, label: TRI_PRESETS[k].label })), value: 'weather', onChange: v => { TRI_PRESETS[v].off.forEach((x, k) => sl[k].set(x)); draw(); } });
      const sl = sixSliders(controls, TRI_PRESETS.weather.off, () => draw());
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, x: [0, 60], y: [1e-6, 1], logy: true, xlabel: 'n', ylabel: 'd(n)', title: 'Distance to stationarity' });
      const pE = new Plot(stage, { width: 420, height: 270, x: [-1.2, 1.2], y: [-1.2, 1.2], equal: true, xlabel: 'Re λ', ylabel: 'Im λ', title: 'Spectrum of P' });
      const sPi = UI.stat(stats, 'Stationary $\\pi$');
      const sL = UI.stat(stats, '$|\\lambda_2|$', { accent: 'accent' });
      const sT = UI.stat(stats, '$t_{\\text{mix}}(1/4)$');
      const sRev = UI.stat(stats, 'Reversible?');
      function draw() {
        const { P } = threeState(sl.map(s => s.get())), pi = stationary(P);
        const rows = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], d = [];
        let cur = rows.map(r => r.slice()), tmix = null;
        for (let n = 0; n <= 60; n++) {
          const dn = Math.max(...cur.map(r => tv(r, pi)));
          d.push([n, dn]); if (tmix == null && dn <= 0.25) tmix = n;
          cur = cur.map(r => vecMat(r, P));
        }
        const ev = eig3(P), mods = ev.map(([a, b]) => Math.hypot(a, b));
        const l2 = Math.max(mods[1], mods[2]);
        let rev = true;
        for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (Math.abs(pi[i] * P[i][j] - pi[j] * P[j][i]) > 1e-9) rev = false;
        pD.clear();
        pD.hline(0.25, { color: 'muted', width: 1 });
        pD.line(d.filter(p => p[1] > 1e-7), { color: 's1', width: 2.5 });
        pD.line(d.map(([n]) => [n, 0.5 * l2 ** n]).filter(p => p[1] > 1e-7), { color: 's2', width: 1.5, dash: true });
        const pmin = Math.min(...pi);
        if (rev && pmin > 1e-9) pD.line(d.map(([n]) => [n, Math.min(1, 0.5 * Math.sqrt((1 - pmin) / pmin) * l2 ** n)]).filter(p => p[1] > 1e-7), { color: 's3', width: 1.5, dash: '2 3' });
        pD.legend([{ label: 'd(n)', color: 's1' }, { label: '½|λ₂|ⁿ', color: 's2', dash: true }].concat(rev ? [{ label: 'spectral upper bound', color: 's3', dash: true }] : []));
        pE.clear();
        pE.line(S.linspace(0, 2 * Math.PI, 120).map(t => [Math.cos(t), Math.sin(t)]), { color: 'muted', width: 1 });
        pE.segment(-1.2, 0, 1.2, 0, { color: 'grid', width: 1 }); pE.segment(0, -1.2, 0, 1.2, { color: 'grid', width: 1 });
        pE.line(S.linspace(0, 2 * Math.PI, 120).map(t => [l2 * Math.cos(t), l2 * Math.sin(t)]), { color: 's2', width: 1, dash: true });
        ev.forEach(([a, b], k) => pE.dot(a, b, { r: 6, color: k === 0 ? 'ink' : 's2' }));
        sPi.set(fmtVec(pi));
        sL.set(S.fmt(l2), ev[1][1] !== 0 ? `complex pair ${S.fmt(ev[1][0], 2)} ± ${S.fmt(Math.abs(ev[1][1]), 2)}i` : `λ₂, λ₃ = ${S.fmt(ev[1][0], 2)}, ${S.fmt(ev[2][0], 2)}`);
        sT.set(tmix == null ? '> 60' : String(tmix), l2 >= 1 - 1e-9 ? 'no convergence: |λ₂| = 1' : `relaxation time 1/(1−|λ₂|) = ${S.fmt(1 / (1 - l2), 2)}`);
        sRev.set(rev ? 'yes' : 'no', rev ? 'detailed balance holds: real spectrum' : 'net probability current around the cycle');
      }
      draw();
    },
  });

  Atlas.lab('stationary-distributions-coupling', {
    title: 'Mixing on a cycle: exact distance vs. coupling',
    blurb: 'Lazy random walk on $\\mathbb Z_N$: stay with probability $a$, else step $\\pm1$. Two walkers start at the farthest-apart states and move independently until they meet. Right: the exact $d(n)$, the coupling bound $\\P(T>n)$, and the spectral bounds $\\tfrac12\\lambda_*^n \\le d(n) \\le \\tfrac12\\sqrt{N-1}\\,\\lambda_*^n$. Double $N$ and the mixing time roughly quadruples. Set $a=0$ with $N$ even to see periodicity break everything.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const Ns = UI.slider(controls, { label: 'Cycle length $N$', min: 4, max: 40, step: 1, value: 16, onChange: run });
      const as = UI.slider(controls, { label: 'Laziness $a$', min: 0, max: 0.9, step: 0.05, value: 0.5, onChange: run });
      UI.button(controls, 'New coupled pair', () => { samplePair(); draw(); }, { primary: true });
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, x: [0, 100], y: [-0.5, 16], xlabel: 'time', ylabel: 'position', title: 'One coupled pair until it meets' });
      const pD = new Plot(stage, { width: 420, height: 270, x: [0, 100], y: [0, 1], xlabel: 'n', ylabel: 'distance', title: 'd(n) and its bounds' });
      const sMix = UI.stat(stats, 'Exact $t_{\\text{mix}}(1/4)$', { accent: 'accent' });
      const sRel = UI.stat(stats, '$t_{\\text{rel}} = 1/(1-\\lambda_*)$');
      const sRef = UI.stat(stats, '$N^2/(2\\pi^2(1-a))$');
      const sT = UI.stat(stats, 'Mean coupling time');
      let N, a, nmax, d, surv, pair, lamStar, meanT;
      function stepW(x) { const u = ctx.rng.u(); if (u < a) return x; return (u < a + (1 - a) / 2) ? (x + 1) % N : (x + N - 1) % N; }
      function samplePair() {
        let x = 0, y = Math.floor(N / 2); const px = [[0, x]], py = [[0, y]];
        for (let t = 1; t <= nmax && x !== y; t++) { x = stepW(x); y = stepW(y); px.push([t, x]); py.push([t, y]); }
        pair = { px, py, met: x === y ? px.length - 1 : null };
      }
      function run() {
        N = Math.round(Ns.get()); a = as.get();
        nmax = Math.min(4000, Math.ceil(1.6 * N * N / Math.max(0.1, 1 - a)));
        // exact distance (vertex-transitive: same from every start)
        let mu = new Float64Array(N); mu[0] = 1; d = [];
        for (let n = 0; n <= nmax; n++) {
          let s = 0; for (let j = 0; j < N; j++) s += Math.abs(mu[j] - 1 / N); d.push([n, s / 2]);
          const nu = new Float64Array(N);
          for (let j = 0; j < N; j++) { const m = mu[j]; if (!m) continue; nu[j] += a * m; nu[(j + 1) % N] += (1 - a) * m / 2; nu[(j + N - 1) % N] += (1 - a) * m / 2; }
          mu = nu;
        }
        lamStar = 0;
        for (let k = 1; k < N; k++) lamStar = Math.max(lamStar, Math.abs(a + (1 - a) * Math.cos(2 * Math.PI * k / N)));
        // coupling simulation: independent moves until meeting, from the farthest pair
        const M = 600, T = [];
        for (let m = 0; m < M; m++) {
          let x = 0, y = Math.floor(N / 2), t = 0;
          while (x !== y && t < nmax) { x = stepW(x); y = stepW(y); t++; }
          T.push(x === y ? t : Infinity);
        }
        T.sort((p, q) => p - q);
        surv = []; let idx = 0;
        for (let n = 0; n <= nmax; n += Math.max(1, Math.floor(nmax / 300))) { while (idx < M && T[idx] <= n) idx++; surv.push([n, (M - idx) / M]); }
        const fin = T.filter(Number.isFinite);
        meanT = { mean: fin.length ? S.mean(fin) : NaN, cens: M - fin.length };
        samplePair();
        draw();
      }
      function draw() {
        pP.setDomain([0, Math.max(10, pair.px.length - 1)], [-0.5, N - 0.5]).clear();
        pP.points(pair.px, { color: 's1', r: 2 });
        pP.points(pair.py, { color: 's2', r: 2 });
        if (pair.met != null) pP.vline(pair.met, { color: 'accent' });
        pP.legend([{ label: 'walker X (from 0)', color: 's1' }, { label: 'walker Y (from N/2)', color: 's2' }]);
        pD.setDomain([0, nmax], [0, 1]).clear();
        pD.hline(0.25, { color: 'muted', width: 1 });
        pD.line(d, { color: 's1', width: 2.5 });
        pD.line(surv, { color: 's2', width: 2 });
        pD.line(d.map(([n]) => [n, Math.min(1, 0.5 * Math.sqrt(N - 1) * lamStar ** n)]), { color: 's3', width: 1.5, dash: true });
        pD.line(d.map(([n]) => [n, 0.5 * lamStar ** n]), { color: 's4', width: 1.5, dash: '2 3' });
        pD.legend([{ label: 'exact d(n)', color: 's1' }, { label: 'coupling P(T > n)', color: 's2' }, { label: '½√(N−1) λ*ⁿ', color: 's3', dash: true }, { label: '½ λ*ⁿ', color: 's4', dash: true }]);
        const tm = d.find(p => p[1] <= 0.25);
        sMix.set(tm ? String(tm[0]) : `> ${nmax}`);
        sRel.set(lamStar >= 1 - 1e-12 ? '∞' : S.fmt(1 / (1 - lamStar), 3), lamStar >= 1 - 1e-12 ? 'periodic: λ = −1 is an eigenvalue' : `λ* = ${S.fmt(lamStar, 4)}`);
        sRef.set(a < 1 ? S.fmt(N * N / (2 * Math.PI * Math.PI * (1 - a)), 3) : '—', 'small-gap approximation');
        sT.set(Number.isFinite(meanT.mean) ? S.fmt(meanT.mean, 3) : '—', meanT.cens ? `${meanT.cens} of 600 pairs never met by n = ${nmax}` : '600 simulated pairs');
      }
      run();
    },
  });

  /* ================= random-walks ================= */

  // Return probabilities u_k = P(S_{2k} = 0) for simple random walk on Z^d, k = 0..K.
  function returnSeries(d, K) {
    const u1 = new Float64Array(K + 1); u1[0] = 1;
    for (let k = 1; k <= K; k++) u1[k] = u1[k - 1] * (2 * k - 1) / (2 * k);
    if (d === 1) return u1;
    if (d === 2) return u1.map(v => v * v);
    const u = new Float64Array(K + 1); u[0] = 1;
    const KX = Math.min(K, 300);
    for (let k = 1; k <= KX; k++) {
      const terms = [];
      for (let i = 0; i <= k; i++) terms.push(2 * S.lchoose(k, i) + S.lchoose(2 * (k - i), k - i));
      u[k] = Math.exp(S.lchoose(2 * k, k) - 2 * k * Math.LN2 - k * Math.log(9) + S.logsumexp(terms));
    }
    for (let k = KX + 1; k <= K; k++) u[k] = 2 * Math.pow(3 / (4 * Math.PI * k), 1.5); // local CLT
    return u;
  }
  // First-return probabilities from the renewal equation u_k = sum_j f_j u_{k-j}.
  function firstReturn(u) {
    const K = u.length - 1, f = new Float64Array(K + 1);
    for (let k = 1; k <= K; k++) { let s = u[k]; for (let j = 1; j < k; j++) s -= f[j] * u[k - j]; f[k] = s; }
    return f;
  }

  Atlas.lab('random-walks-return', {
    title: 'Pólya: returns to the origin in one, two and three dimensions',
    blurb: 'Simulate many simple random walks and count visits to the origin. Right: the mean cumulative number of returns against the exact expectation $\\sum_{2k\\le t}p^{(2k)}_{00}$. In 1-D it grows like $\\sqrt t$, in 2-D like $\\log t$ (still unbounded: recurrent), in 3-D it levels off near $0.516$ (transient: only about 34% of walkers ever return).',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dim = UI.segmented(controls, { label: 'Dimension $d$', options: ['1', '2', '3'], value: '2', onChange: run });
      const nn = UI.slider(controls, { label: 'Steps $n$', min: 100, max: 10000, value: 2000, log: true, fmt: v => Math.round(v), onChange: run });
      const W = UI.slider(controls, { label: 'Walkers', min: 20, max: 200, step: 10, value: 100, onChange: run });
      UI.button(controls, 'New walks', run, { primary: true });
      stage.classList.add('two');
      const pW = new Plot(stage, { width: 420, height: 270, xlabel: '', ylabel: '', title: 'One walk' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'returns to 0 by time t', title: 'Mean number of returns' });
      const sAny = UI.stat(stats, 'Walkers that returned by $n$', { accent: 'accent' });
      const sAnyT = UI.stat(stats, 'Exact $\\P(T_0\\le n)$');
      const sInf = UI.stat(stats, '$\\P(T_0<\\infty)$');
      const sMean = UI.stat(stats, 'Mean returns by $n$');
      function run() {
        const d = +dim.get(), n = Math.round(nn.get()), w = Math.round(W.get()), u = ctx.rng.u;
        const cum = new Float64Array(n + 1); let anyRet = 0; let path = [];
        const stride = Math.max(1, Math.floor(n / 3000));
        for (let k = 0; k < w; k++) {
          const x = [0, 0, 0]; let first = true;
          for (let t = 1; t <= n; t++) {
            const r = Math.floor(u() * 2 * d), ax = r >> 1;
            x[ax] += (r & 1) ? 1 : -1;
            if (x[0] === 0 && x[1] === 0 && x[2] === 0) { cum[t] += 1; if (first) { anyRet++; first = false; } }
            if (k === 0 && t % stride === 0) path.push(d === 1 ? [t, x[0]] : [x[0], x[1]]);
          }
        }
        for (let t = 1; t <= n; t++) cum[t] += cum[t - 1];
        const K = Math.floor(n / 2), uu = returnSeries(d, K), f = firstReturn(uu);
        const exact = [[0, 0]]; let s = 0;
        for (let k = 1; k <= K; k++) { s += uu[k]; exact.push([2 * k, s]); }
        const pRet = S.sum(Array.from(f).slice(1));
        // left: the path
        pW.o.equal = d !== 1;
        if (d === 1) {
          path.unshift([0, 0]);
          const ext = Math.max(5, ...path.map(p => Math.abs(p[1])));
          pW.setDomain([0, n], [-ext * 1.1, ext * 1.1]).clear();
          pW.hline(0, { color: 'muted', width: 1 });
        } else {
          path.unshift([0, 0]);
          const ext = Math.max(5, ...path.map(p => Math.max(Math.abs(p[0]), Math.abs(p[1]))));
          pW.setDomain([-ext * 1.1, ext * 1.1], [-ext * 1.1, ext * 1.1]).clear();
        }
        pW.line(path, { color: 's1', width: 1.2, opacity: 0.8 });
        pW.dot(0, 0, { r: 5, color: 'accent' });
        if (d === 3) pW.text(pW.o.x[0], pW.o.y[1], '(x, y) projection', { dx: 6, dy: 14, size: 11, color: 'muted' });
        // right: cumulative returns
        const emp = []; const st2 = Math.max(1, Math.floor(n / 400));
        for (let t = 0; t <= n; t += st2) emp.push([t, cum[t] / w]);
        const ymax = Math.max(0.6, emp[emp.length - 1][1], s) * 1.15;
        pR.setDomain([0, n], [0, ymax]).clear();
        pR.line(exact, { color: 'ink', width: 1.5, dash: true });
        pR.line(emp, { color: 's1', width: 2.5 });
        if (d === 3) pR.hline(1 / (1 - 0.3405) - 1, { color: 's3', width: 1 });
        pR.legend([{ label: 'simulated mean', color: 's1' }, { label: 'exact Σ p₀₀^(2k)', color: 'ink', dash: true }].concat(d === 3 ? [{ label: 'limit 0.516', color: 's3' }] : []), { left: true });
        sAny.set(S.fmt(anyRet / w, 3), `${anyRet} of ${w}`);
        sAnyT.set(S.fmt(pRet, 3), 'by the renewal equation');
        sInf.set(d === 3 ? '≈ 0.3405' : '1', d === 3 ? 'transient' : 'recurrent');
        sMean.set(S.fmt(cum[n] / w, 3), `exact ${S.fmt(s, 3)}`);
      }
      run();
    },
  });

  Atlas.lab('random-walks-ruin', {
    title: "Gambler's ruin: hitting $N$ before $0$",
    blurb: 'Fortune starts at $i$ and moves $\\pm1$ with probabilities $p$, $1-p$ until it hits $0$ or $N$. Left: sample games (green: reached $N$, red: ruined). Right: the exact success probability $h_j$ for every starting fortune, with the simulated success rate at $i$. A small edge against you ($p = 0.47$) is devastating when $N$ is large.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pp = UI.slider(controls, { label: 'Win probability $p$', min: 0.3, max: 0.7, step: 0.005, value: 0.47, onInput: draw, onChange: sim });
      const Nn = UI.slider(controls, { label: 'Target $N$', min: 2, max: 60, step: 1, value: 20, onInput: draw, onChange: sim });
      const ii = UI.slider(controls, { label: 'Starting fortune $i$', min: 1, max: 59, step: 1, value: 10, onInput: draw, onChange: sim });
      UI.button(controls, 'Play again', sim, { primary: true });
      stage.classList.add('two');
      const pG = new Plot(stage, { width: 420, height: 270, xlabel: 'bet number', ylabel: 'fortune', title: 'Sample games' });
      const pH = new Plot(stage, { width: 420, height: 270, xlabel: 'starting fortune j', ylabel: 'P(reach N first)', title: 'Success probability h_j' });
      const sH = UI.stat(stats, 'Exact $h_i$', { accent: 'accent' });
      const sHe = UI.stat(stats, 'Simulated');
      const sD = UI.stat(stats, 'Exact $\\E$ duration $D_i$');
      const sDe = UI.stat(stats, 'Simulated mean duration');
      let games = [];
      const params = () => { const N = Math.round(Nn.get()), i = Math.min(Math.round(ii.get()), N - 1); return { p: pp.get(), N, i }; };
      function hExact(p, N, j) { if (Math.abs(p - 0.5) < 1e-12) return j / N; const r = (1 - p) / p; return (1 - r ** j) / (1 - r ** N); }
      function dExact(p, N, j) { if (Math.abs(p - 0.5) < 1e-12) return j * (N - j); return (j - N * hExact(p, N, j)) / ((1 - p) - p); }
      function sim() {
        const { p, N, i } = params(), u = ctx.rng.u;
        games = [];
        for (let g = 0; g < 400; g++) {
          let x = i, t = 0; const keep = g < 25, path = keep ? [[0, i]] : null;
          while (x > 0 && x < N && t < 200000) { x += u() < p ? 1 : -1; t++; if (keep) path.push([t, x]); }
          games.push({ win: x >= N, t, path });
        }
        draw();
      }
      function draw() {
        const { p, N, i } = params();
        const shown = games.filter(g => g.path);
        const tmax = Math.max(10, ...shown.map(g => g.t));
        pG.setDomain([0, tmax], [0, N]).clear();
        for (const g of shown) pG.line(g.path, { color: g.win ? 'good' : 'bad', width: 1.2, opacity: 0.55 });
        pG.hline(i, { color: 'muted', width: 1 });
        const hs = S.range(N + 1).map(j => [j, hExact(p, N, j)]);
        pH.setDomain([0, N], [0, 1]).clear();
        pH.line(S.range(N + 1).map(j => [j, j / N]), { color: 'muted', width: 1, dash: true });
        pH.line(hs, { color: 's1', width: 2.5 });
        if (N <= 40) pH.points(hs, { color: 's1', r: 2.5 });
        const wins = games.filter(g => g.win).length, G = games.length || 1, ph = wins / G, se = Math.sqrt(ph * (1 - ph) / G);
        if (games.length) { pH.segment(i, Math.max(0, ph - 2 * se), i, Math.min(1, ph + 2 * se), { color: 's2', width: 2 }); pH.dot(i, ph, { r: 5, color: 's2' }); }
        pH.legend([{ label: 'exact h_j', color: 's1' }, { label: 'fair game j/N', color: 'muted', dash: true }, { label: 'simulated at i (±2 se)', color: 's2' }], { left: true });
        sH.set(S.fmt(hExact(p, N, i), 4));
        sHe.set(games.length ? S.fmt(ph, 3) : '—', `${wins} of ${games.length} games`);
        sD.set(S.fmt(dExact(p, N, i), 4));
        sDe.set(games.length ? S.fmt(S.mean(games.map(g => g.t)), 4) : '—');
      }
      sim();
    },
  });

  Atlas.lab('random-walks-arcsine', {
    title: 'The arcsine law: lopsided leads in a fair game',
    blurb: 'For many fair $\\pm1$ walks of length $2n$, record either the time of the last visit to $0$ or the fraction of time spent above the axis, both as fractions of $2n$. The histogram piles up near $0$ and $1$, matching the arcsine density $1/(\\pi\\sqrt{x(1-x)})$. Even splits are the least likely outcome.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const which = UI.segmented(controls, { label: 'Statistic', options: [{ value: 'last', label: 'Last zero $L_{2n}/2n$' }, { value: 'pos', label: 'Time positive $\\Pi_{2n}/2n$' }], value: 'pos', onChange: draw });
      const nn = UI.slider(controls, { label: 'Walk length $2n$', min: 20, max: 1000, step: 10, value: 200, onChange: run });
      const M = UI.slider(controls, { label: 'Number of walks', min: 500, max: 3000, step: 100, value: 2000, onChange: run });
      UI.button(controls, 'Resample', run, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 1], y: [0, 4], xlabel: 'fraction of the game', ylabel: 'density', title: '' });
      const sTail = UI.stat(stats, 'Share outside $[0.1, 0.9]$', { accent: 'accent' });
      const sTailT = UI.stat(stats, 'Arcsine limit');
      const sMid = UI.stat(stats, 'Share in $[0.4, 0.6]$');
      const sMidT = UI.stat(stats, 'Arcsine limit');
      let last = [], pos = [];
      function run() {
        const L = Math.round(nn.get()), m = Math.round(M.get()), u = ctx.rng.u;
        last = new Array(m); pos = new Array(m);
        for (let k = 0; k < m; k++) {
          let s = 0, lz = 0, ps = 0;
          for (let j = 1; j <= L; j++) { const prev = s; s += u() < 0.5 ? 1 : -1; if (s === 0) lz = j; if (prev > 0 || s > 0) ps++; }
          last[k] = lz / L; pos[k] = ps / L;
        }
        draw();
      }
      const F = x => (2 / Math.PI) * Math.asin(Math.sqrt(Math.min(1, Math.max(0, x))));
      function draw() {
        const v = which.get() === 'last' ? last : pos;
        const hist = S.histogram(v, { bins: 20, lo: 0, hi: 1 });
        p.clear();
        p.bars(hist, { color: 's1', opacity: 0.5 });
        p.line(S.linspace(0.002, 0.998, 300).map(x => [x, 1 / (Math.PI * Math.sqrt(x * (1 - x)))]), { color: 'ink', width: 2 });
        p.hline(1, { color: 'muted', width: 1 });
        p.legend([{ label: 'simulated', color: 's1', fill: true }, { label: 'arcsine density', color: 'ink' }, { label: 'uniform (naive intuition)', color: 'muted', dash: true }]);
        const out = v.filter(x => x < 0.1 || x > 0.9).length / v.length, mid = v.filter(x => x >= 0.4 && x <= 0.6).length / v.length;
        sTail.set(S.fmt(out, 3));
        sTailT.set(S.fmt(2 * F(0.1), 3), 'uniform would give 0.2');
        sMid.set(S.fmt(mid, 3));
        sMidT.set(S.fmt(F(0.6) - F(0.4), 3), 'uniform would give 0.2');
      }
      run();
    },
  });

  /* ================= branching-processes ================= */

  const OFFSPRING = {
    pois: { label: 'Poisson(m)' },
    geom: { label: 'Geometric, mean m' },
    binary: { label: 'Binary split (0 or 2), m ≤ 2' },
    negbin: { label: 'Negative binomial (m, k): superspreading' },
  };
  function offspringModel(fam, m, k) {
    if (fam === 'binary') m = Math.min(m, 2);
    const a = m / (1 + m);
    const G = {
      pois: s => Math.exp(m * (s - 1)),
      geom: s => (1 - a) / (1 - a * s),
      binary: s => 1 - m / 2 + (m / 2) * s * s,
      negbin: s => Math.pow(1 + (m / k) * (1 - s), -k),
    }[fam];
    const sig2 = { pois: m, geom: m * (1 + m), binary: 2 * m - m * m, negbin: m + m * m / k }[fam];
    const one = r => {
      if (fam === 'pois') return r.poisson(m);
      if (fam === 'geom') return m > 0 ? r.poisson(r.gamma(1, m)) : 0;
      if (fam === 'binary') return r.u() < m / 2 ? 2 : 0;
      return m > 0 ? r.poisson(r.gamma(k, m / k)) : 0;
    };
    const total = (r, Z) => { // sum of Z IID offspring counts, sampled in O(1)
      if (Z === 0 || m === 0) return 0;
      if (Z > 5e7) return Math.round(Z * m);
      if (fam === 'pois') return r.poisson(m * Z);
      if (fam === 'geom') return r.poisson(r.gamma(Z, m));
      if (fam === 'binary') return 2 * r.binomial(Z, m / 2);
      return r.poisson(r.gamma(Z * k, m / k));
    };
    let q = 0;
    if (m <= 1) q = 1;
    else { for (let it = 0; it < 20000; it++) { const nq = G(q); if (Math.abs(nq - q) < 1e-14) { q = nq; break; } q = nq; } }
    return { m, G, sig2, one, total, q };
  }
  function offspringControls(controls, onChange) {
    const fam = UI.select(controls, { label: 'Offspring law', options: Object.keys(OFFSPRING).map(k => ({ value: k, label: OFFSPRING[k].label })), value: 'pois', onChange });
    const m = UI.slider(controls, { label: 'Mean offspring $m$', min: 0, max: 3, step: 0.05, value: 1.5, onChange });
    const k = UI.slider(controls, { label: 'NB dispersion $k$', min: 0.05, max: 20, value: 0.5, log: true, fmt: v => v.toFixed(2), onChange });
    return () => offspringModel(fam.get(), m.get(), k.get());
  }

  Atlas.lab('branching-processes-pgf', {
    title: 'A Galton–Watson tree and its generating function',
    blurb: 'Left: one family tree (each generation capped at 70 drawn individuals; the true count $Z_n$ is printed). Right: the offspring pgf $G(s)$, the diagonal, and the cobweb iteration $s_{n+1} = G(s_n)$ from $s_0=0$, whose values are $\\P(Z_n=0)$. The iteration stalls at the extinction probability $q$, the first crossing of the diagonal. The dashed tangent at $s=1$ has slope $m$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const getModel = offspringControls(controls, () => { newTree(); });
      const gens = UI.slider(controls, { label: 'Generations shown', min: 3, max: 10, step: 1, value: 7, onChange: () => newTree() });
      UI.button(controls, 'New tree', () => newTree(), { primary: true });
      stage.classList.add('two');
      const pT = new Plot(stage, { width: 420, height: 270, x: [0, 1.18], y: [-7.5, 0.5], grid: false, xticks: 0, yticks: 0, title: 'One family tree' });
      const pG = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 1], equal: true, xlabel: 's', ylabel: 'G(s)', title: 'pgf and fixed-point iteration' });
      const sM = UI.stat(stats, 'Mean $m$ / variance $\\sigma^2$');
      const sQ = UI.stat(stats, 'Extinction probability $q$', { accent: 'accent' });
      const sGn = UI.stat(stats, '$\\P(Z_n = 0)$ at last generation');
      const sFate = UI.stat(stats, 'This tree');
      const CAP = 70;
      function newTree() {
        const M = getModel(), G = Math.round(gens.get()), r = ctx.rng;
        const layers = [[{ par: -1 }]], Z = [1];
        let fate = null;
        for (let g = 0; g < G; g++) {
          const shown = layers[g].length, next = []; let tot = 0;
          if (Z[g] > 2e5) { tot = M.total(r, Z[g]); }
          else for (let j = 0; j < Z[g]; j++) { const c = M.one(r); tot += c; if (j < shown) for (let c2 = 0; c2 < c && next.length < CAP; c2++) next.push({ par: j }); }
          Z.push(tot); layers.push(next);
          if (tot === 0 && fate == null) fate = g + 1;
        }
        drawAll(M, G, layers, Z, fate);
      }
      function drawAll(M, G, layers, Z, fate) {
        pT.setDomain([0, 1.18], [-G - 0.5, 0.5]).clear();
        const X = layers.map(L => L.map((_, k) => (k + 0.5) / Math.max(L.length, 1)));
        for (let g = 1; g < layers.length; g++) layers[g].forEach((nd, k) => pT.segment(X[g - 1][nd.par], -(g - 1), X[g][k], -g, { color: 'muted', width: 0.8, opacity: 0.7 }));
        for (let g = 0; g < layers.length; g++) {
          pT.points(X[g].map(x => [x, -g]), { color: SER[g % 8], r: layers[g].length > 40 ? 2 : 3.2, opacity: 0.95 });
          pT.text(1.17, -g, 'Z' + g + ' = ' + Z[g], { anchor: 'end', dy: 4, size: 10, color: Z[g] ? 'ink' : 'bad' });
        }
        // pgf panel
        pG.clear();
        pG.line([[0, 0], [1, 1]], { color: 'muted', width: 1 });
        pG.line([[1 - 0.35, 1 - 0.35 * M.m], [1, 1]], { color: 's3', width: 1.2, dash: true });
        pG.line(S.linspace(0, 1, 200).map(s => [s, M.G(s)]), { color: 's1', width: 2.5 });
        let s = 0; const web = [[0, 0]];
        for (let it = 0; it < 25; it++) { const g = M.G(s); web.push([s, g], [g, g]); s = g; }
        pG.line(web, { color: 's2', width: 1.2, opacity: 0.85 });
        pG.dot(M.q, M.q, { r: 5, color: 'accent' });
        pG.text(M.q, M.q, 'q', { dx: -12, dy: -6, size: 12, weight: 600, color: 'accent' });
        let gn = 0; for (let g = 0; g < G; g++) gn = M.G(gn);
        sM.set(`${S.fmt(M.m, 2)} / ${S.fmt(M.sig2, 2)}`, M.m < 1 ? 'subcritical' : M.m > 1 ? 'supercritical' : 'critical');
        sQ.set(S.fmt(M.q, 4), M.m <= 1 ? 'm ≤ 1: extinction certain' : 'smallest root of G(s) = s');
        sGn.set(S.fmt(gn, 4), `n = ${G}`);
        sFate.set(fate ? `extinct at generation ${fate}` : `alive, Z${G} = ${Z[G]}`);
      }
      newTree();
    },
  });

  Atlas.lab('branching-processes-generations', {
    title: 'Many populations: extinction curves and the martingale $Z_n/m^n$',
    blurb: 'Simulate 400 independent Galton–Watson processes for 30 generations. Left: trajectories of $\\log_{10}(1+Z_n)$ (grey once extinct) or of the martingale $W_n = Z_n/m^n$, which settles to a random limit. Right: the fraction extinct by generation $n$ against the exact $G_n(0)$, converging to $q$. Compare Poisson and negative-binomial offspring with the same mean.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const getModel = offspringControls(controls, run);
      const view = UI.segmented(controls, { label: 'Left panel', options: [{ value: 'log', label: '$\\log_{10}(1+Z_n)$' }, { value: 'W', label: '$W_n = Z_n/m^n$' }], value: 'log', onChange: draw });
      UI.button(controls, 'Resimulate', run, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 30], y: [0, 6], xlabel: 'generation n', title: 'Trajectories' });
      const pE = new Plot(stage, { width: 420, height: 270, x: [0, 30], y: [0, 1], xlabel: 'generation n', ylabel: 'P(Z_n = 0)', title: 'Extinction by generation n' });
      const sQ = UI.stat(stats, 'Exact $q$', { accent: 'accent' });
      const sE = UI.stat(stats, 'Extinct by $n = 30$');
      const sW = UI.stat(stats, 'Mean of $W_{30}$');
      const sK = UI.stat(stats, 'Survivors at 30: mean $Z_{30}$');
      const NG = 30, NP = 400;
      let M, paths;
      function run() {
        M = getModel(); const r = ctx.rng;
        paths = [];
        for (let k = 0; k < NP; k++) { const z = [1]; for (let g = 0; g < NG; g++) z.push(M.total(r, z[g])); paths.push(z); }
        draw();
      }
      function draw() {
        const isW = view.get() === 'W' && M.m > 0;
        const val = (z, g) => (isW ? z[g] / M.m ** g : Math.log10(1 + z[g]));
        const shown = paths.slice(0, 80);
        let ymax = isW ? Math.max(2, S.quantile(paths.map(z => z[NG] / M.m ** NG), 0.97)) : Math.max(1, ...shown.map(z => Math.log10(1 + z[NG]))) * 1.1;
        pL.setDomain([0, NG], [0, ymax]).clear();
        for (const z of shown) {
          const dead = z[NG] === 0;
          pL.line(z.map((_, g) => [g, val(z, g)]), { color: dead ? 'muted' : 's1', width: 1, opacity: dead ? 0.35 : 0.6 });
        }
        if (!isW && M.m > 0) pL.line(S.range(NG + 1).map(g => [g, Math.log10(1 + M.m ** g)]), { color: 'ink', width: 2, dash: true });
        if (isW) pL.hline(1, { color: 'ink', width: 1.5 });
        pL.legend(isW ? [{ label: 'E W_n = 1', color: 'ink' }] : [{ label: 'log₁₀(1 + mⁿ)', color: 'ink', dash: true }]);
        const emp = S.range(NG + 1).map(g => [g, paths.filter(z => z[g] === 0).length / NP]);
        const ex = []; let s = 0; for (let g = 0; g <= NG; g++) { ex.push([g, s]); s = M.G(s); }
        pE.clear();
        pE.hline(M.q, { color: 'accent', width: 1.2 });
        pE.line(ex, { color: 'ink', width: 2, dash: true });
        pE.line(emp, { color: 's1', width: 2.5 });
        pE.legend([{ label: 'simulated', color: 's1' }, { label: 'exact G_n(0)', color: 'ink', dash: true }, { label: 'q', color: 'accent' }], { left: true });
        const surv = paths.filter(z => z[NG] > 0);
        sQ.set(S.fmt(M.q, 4), Math.abs(M.m - 1) < 1e-9 ? `critical: P(Z_n > 0) ≈ 2/(nσ²) = ${S.fmt(2 / (NG * M.sig2), 3)}` : '');
        sE.set(S.fmt(emp[NG][1], 3), `exact G₃₀(0) = ${S.fmt(ex[NG][1], 3)}`);
        sW.set(M.m > 0 ? S.fmt(S.mean(paths.map(z => z[NG] / M.m ** NG)), 3) : '—', 'martingale: expectation 1 (heavy right tail)');
        sK.set(surv.length ? S.fmt(S.mean(surv.map(z => z[NG])), 3) : '—', `${surv.length} survivors`);
      }
      run();
    },
  });

  /* ================= hidden-markov ================= */

  // Scaled forward–backward. E[t][k] = b_k(x_t). Returns filtered, smoothed, expected transition counts, log-likelihood.
  function forwardBackward(E, pi, A) {
    const T = E.length, K = pi.length, al = new Array(T), be = new Array(T), c = new Float64Array(T);
    for (let t = 0; t < T; t++) {
      const a = new Float64Array(K);
      for (let l = 0; l < K; l++) { let s = 0; if (t === 0) s = pi[l]; else for (let k = 0; k < K; k++) s += al[t - 1][k] * A[k][l]; a[l] = s * E[t][l]; }
      let z = 0; for (let l = 0; l < K; l++) z += a[l];
      c[t] = z; for (let l = 0; l < K; l++) a[l] /= z; al[t] = a;
    }
    be[T - 1] = new Float64Array(K).fill(1);
    for (let t = T - 2; t >= 0; t--) {
      const b = new Float64Array(K);
      for (let k = 0; k < K; k++) { let s = 0; for (let l = 0; l < K; l++) s += A[k][l] * E[t + 1][l] * be[t + 1][l]; b[k] = s / c[t + 1]; }
      be[t] = b;
    }
    const gamma = al.map((a, t) => Array.from(a, (v, k) => v * be[t][k]));
    const xi = Array.from({ length: K }, () => new Array(K).fill(0));
    for (let t = 0; t < T - 1; t++) for (let k = 0; k < K; k++) for (let l = 0; l < K; l++) xi[k][l] += al[t][k] * A[k][l] * E[t + 1][l] * be[t + 1][l] / c[t + 1];
    let ll = 0; for (let t = 0; t < T; t++) ll += Math.log(c[t]);
    return { filt: al, gamma, xi, ll };
  }
  function viterbi(E, pi, A) {
    const T = E.length, K = pi.length, lA = A.map(r => r.map(v => Math.log(v))), psi = [];
    let d = pi.map((p, k) => Math.log(p) + Math.log(E[0][k]));
    for (let t = 1; t < T; t++) {
      const nd = new Array(K), ps = new Array(K);
      for (let l = 0; l < K; l++) { let best = -Infinity, arg = 0; for (let k = 0; k < K; k++) { const v = d[k] + lA[k][l]; if (v > best) { best = v; arg = k; } } nd[l] = best + Math.log(E[t][l]); ps[l] = arg; }
      psi.push(ps); d = nd;
    }
    const z = new Array(T); z[T - 1] = S.argmax(d);
    for (let t = T - 2; t >= 0; t--) z[t] = psi[t][z[t + 1]];
    return z;
  }
  const casinoB = p6 => [[1, 1, 1, 1, 1, 1].map(v => v / 6), [1, 1, 1, 1, 1, 1].map((_, f) => (f === 5 ? p6 : (1 - p6) / 5))];
  function simulateHMM(r, T, pi, A, B) {
    const z = new Array(T), x = new Array(T);
    for (let t = 0; t < T; t++) { z[t] = t ? r.categorical(A[z[t - 1]]) : r.categorical(pi); x[t] = r.categorical(B[z[t]]); }
    return { z, x };
  }
  const runsOf = (arr, val) => { const out = []; let s = null; for (let t = 0; t <= arr.length; t++) { const on = t < arr.length && arr[t] === val; if (on && s == null) s = t; if (!on && s != null) { out.push([s, t]); s = null; } } return out; };

  Atlas.lab('hidden-markov-casino', {
    title: 'The occasionally dishonest casino',
    blurb: 'A casino secretly switches between a fair die and a loaded one. Shaded background: when the loaded die was truly in use. Curves: smoothed posterior $\\P(Z_t = L\\mid x_{1:T})$ and (optionally) the filter $\\P(Z_t = L\\mid x_{1:t})$. Top strip: the Viterbi path. Dots along the bottom mark sixes. Make the loaded die less loaded, or the switches more frequent, and watch decoding degrade.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const aFL = UI.slider(controls, { label: '$P(F\\to L)$', min: 0.01, max: 0.3, step: 0.01, value: 0.05, onChange: gen });
      const aLF = UI.slider(controls, { label: '$P(L\\to F)$', min: 0.01, max: 0.3, step: 0.01, value: 0.1, onChange: gen });
      const p6 = UI.slider(controls, { label: '$P(6\\mid L)$', min: 0.17, max: 0.9, step: 0.01, value: 0.5, onChange: gen });
      const TT = UI.slider(controls, { label: 'Rolls $T$', min: 100, max: 1000, step: 50, value: 300, onChange: gen });
      const showF = UI.toggle(controls, { label: 'Show filter (past only)', value: true, onChange: draw });
      UI.button(controls, 'New rolls', gen, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 300], y: [-0.12, 1.14], xlabel: 'roll t', ylabel: 'P(loaded)', title: '' });
      const sV = UI.stat(stats, 'Viterbi accuracy', { accent: 'accent' });
      const sP = UI.stat(stats, 'Posterior decoding accuracy');
      const sF = UI.stat(stats, 'Filter-only accuracy');
      const sLL = UI.stat(stats, 'Log-likelihood per roll');
      let data, res, vit;
      function gen() {
        const A = [[1 - aFL.get(), aFL.get()], [aLF.get(), 1 - aLF.get()]], B = casinoB(p6.get());
        const s = aFL.get() + aLF.get(), pi = [aLF.get() / s, aFL.get() / s];
        data = simulateHMM(ctx.rng, Math.round(TT.get()), pi, A, B);
        const E = data.x.map(v => [B[0][v], B[1][v]]);
        res = forwardBackward(E, pi, A);
        vit = viterbi(E, pi, A);
        draw();
      }
      function draw() {
        const T = data.x.length;
        p.setDomain([0, T], [-0.12, 1.14]).clear();
        for (const [a, b] of runsOf(data.z, 1)) p.rect(a, 0, b, 1, { color: 'warn', opacity: 0.16 });
        for (const [a, b] of runsOf(vit, 1)) p.rect(a, 1.04, b, 1.11, { color: 's3', opacity: 0.85 });
        p.text(0, 1.075, 'Viterbi', { dx: -4, dy: 4, anchor: 'end', size: 10, color: 'muted' });
        p.hline(0.5, { color: 'muted', width: 1 });
        if (showF.get()) p.line(res.filt.map((a, t) => [t + 0.5, a[1]]), { color: 's2', width: 1.2, opacity: 0.8 });
        p.line(res.gamma.map((g, t) => [t + 0.5, g[1]]), { color: 's1', width: 2.2 });
        p.points(data.x.map((v, t) => (v === 5 ? [t + 0.5, -0.06] : null)).filter(Boolean), { color: 'ink', r: 1.6, opacity: 0.6 });
        p.legend([{ label: 'truly loaded', color: 'warn', fill: true, opacity: 0.25 }, { label: 'smoothed P(L | all rolls)', color: 's1' }].concat(showF.get() ? [{ label: 'filtered P(L | rolls so far)', color: 's2' }] : []).concat([{ label: 'Viterbi: loaded', color: 's3', fill: true }]));
        const acc = f => data.z.filter((z, t) => z === f(t)).length / T;
        sV.set(S.fmt(acc(t => vit[t]), 3));
        sP.set(S.fmt(acc(t => (res.gamma[t][1] > 0.5 ? 1 : 0)), 3));
        sF.set(S.fmt(acc(t => (res.filt[t][1] > 0.5 ? 1 : 0)), 3));
        sLL.set(S.fmt(res.ll / T, 4), `fair die alone: ${S.fmt(-Math.log(6), 4)}`);
      }
      gen();
    },
  });

  Atlas.lab('hidden-markov-baumwelch', {
    title: 'Baum–Welch: EM for the casino from a random start',
    blurb: 'Rolls are simulated from the true casino ($P(F\\to L)=0.05$, $P(L\\to F)=0.1$, $P(6\\mid L)=0.5$). Baum–Welch starts from random parameters; each frame is one EM iteration. Left: the log-likelihood, which can only go up, against the likelihood at the true parameters. Right: estimated emission distributions of the two hidden states (sorted so the six-heavy state is "L") against the truth. Try short sequences and several restarts.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const TT = UI.slider(controls, { label: 'Rolls $T$', min: 200, max: 3000, step: 100, value: 1000, onChange: () => { newData(); restart(); } });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Run EM', '❚❚ Pause']);
      UI.button(row, 'Random restart', () => { restart(); loop.start(); });
      UI.button(row, 'New data', () => { newData(); restart(); loop.start(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 50], y: [-2, -1.6], xlabel: 'iteration', ylabel: 'log-lik per roll', title: 'EM never decreases the likelihood' });
      const pB = new Plot(stage, { width: 420, height: 270, x: [0.4, 6.6], y: [0, 0.7], xlabel: 'face', ylabel: 'probability', title: 'Emission estimates vs truth' });
      const sIt = UI.stat(stats, 'Iteration');
      const sLL = UI.stat(stats, 'Log-lik per roll', { accent: 'accent' });
      const sA = UI.stat(stats, 'Estimated $P(F\\to L)$, $P(L\\to F)$');
      const s6 = UI.stat(stats, 'Estimated $P(6\\mid L)$');
      const TRUE = { pi: [2 / 3, 1 / 3], A: [[0.95, 0.05], [0.1, 0.9]], B: casinoB(0.5) };
      let data, th, hist, trueLL, fb, acc = 0;
      function newData() {
        data = simulateHMM(ctx.rng, Math.round(TT.get()), TRUE.pi, TRUE.A, TRUE.B);
        trueLL = forwardBackward(data.x.map(v => [TRUE.B[0][v], TRUE.B[1][v]]), TRUE.pi, TRUE.A).ll / data.x.length;
      }
      function evaluate() { // E-step at the current parameters; records the log-likelihood
        fb = forwardBackward(data.x.map(v => [th.B[0][v], th.B[1][v]]), th.pi, th.A);
        hist.push([hist.length, fb.ll / data.x.length]);
      }
      function restart() {
        const r = ctx.rng;
        th = { pi: [0.5, 0.5], A: [r.dirichlet([4, 1]), r.dirichlet([1, 4])], B: [r.dirichlet([3, 3, 3, 3, 3, 3]), r.dirichlet([3, 3, 3, 3, 3, 3])] };
        hist = [];
        evaluate();
        draw();
      }
      function mStep() { // closed-form maximisation of Q using the posteriors in fb
        const x = data.x, T = x.length, g = fb.gamma;
        const pi = g[0].map(v => Math.max(v, 1e-9));
        const A = fb.xi.map(r => { const s = r[0] + r[1]; return r.map(v => v / s); });
        const B = [0, 1].map(k => { const num = new Array(6).fill(1e-12); let den = 6e-12; for (let t = 0; t < T; t++) { num[x[t]] += g[t][k]; den += g[t][k]; } return num.map(v => v / den); });
        th = { pi, A, B };
      }
      function tick(dt) {
        acc += dt; if (acc < 0.06) return; acc = 0;
        mStep();
        evaluate();
        draw();
        const n = hist.length;
        if (n > 300 || (n > 3 && Math.abs(hist[n - 1][1] - hist[n - 2][1]) < 1e-9)) return false;
      }
      function draw() {
        const lo = Math.min(...hist.map(h => h[1]), trueLL), hi = Math.max(...hist.map(h => h[1]), trueLL), pad = (hi - lo) * 0.1 + 1e-3;
        pL.setDomain([0, Math.max(20, hist.length)], [lo - pad, hi + pad]).clear();
        pL.hline(trueLL, { color: 'ink', width: 1.5 });
        pL.line(hist, { color: 's1', width: 2.5 });
        pL.points(hist, { color: 's1', r: 2 });
        pL.legend([{ label: 'EM iterates', color: 's1' }, { label: 'at true parameters', color: 'ink', dash: true }], { left: false });
        const L = th.B[0][5] > th.B[1][5] ? 0 : 1, F = 1 - L;
        pB.clear();
        for (let f = 0; f < 6; f++) {
          pB.bars([{ x0: f + 1 - 0.38, x1: f + 1 - 0.02, y: th.B[F][f] }], { color: 's1', opacity: 0.6 });
          pB.bars([{ x0: f + 1 + 0.02, x1: f + 1 + 0.38, y: th.B[L][f] }], { color: 's2', opacity: 0.6 });
          pB.segment(f + 1 - 0.4, TRUE.B[0][f], f + 1, TRUE.B[0][f], { color: 'ink', width: 2 });
          pB.segment(f + 1, TRUE.B[1][f], f + 1 + 0.4, TRUE.B[1][f], { color: 'ink', width: 2, dash: '3 2' });
        }
        pB.legend([{ label: 'state "F" estimate', color: 's1', fill: true }, { label: 'state "L" estimate', color: 's2', fill: true }, { label: 'truth', color: 'ink' }], { left: true });
        sIt.set(String(hist.length - 1));
        sLL.set(S.fmt(hist[hist.length - 1][1], 4), `true parameters: ${S.fmt(trueLL, 4)}`);
        sA.set(`${S.fmt(th.A[F][L], 3)}, ${S.fmt(th.A[L][F], 3)}`, 'truth: 0.050, 0.100');
        s6.set(S.fmt(th.B[L][5], 3), 'truth: 0.500');
      }
      newData(); restart(); loop.start();
    },
  });
})();
