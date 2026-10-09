/* Labs: Supervised learning II — trees, ensembles, boosting, SVMs, kernels, Gaussian processes. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const range = n => Array.from({ length: n }, (_, i) => i);
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  /* =====================================================================
   * Shared: 2-D classification data on roughly [-3, 3]^2, labels 0/1
   * ===================================================================== */
  const DATA2D = {
    blobs: { label: 'Two Gaussian blobs', gen: r => { const c = r.u() < 0.5 ? 0 : 1; return [[r.normal(c ? 0.8 : -0.8, 1), r.normal(c ? 0.6 : -0.6, 1)], c]; } },
    circle: { label: 'Disc inside a square', gen: r => { const x = r.uniform(-3, 3), y = r.uniform(-3, 3); return [[x, y], x * x + y * y < 3.6 ? 1 : 0]; } },
    xor: { label: 'XOR quadrants', gen: r => { const x = r.uniform(-3, 3), y = r.uniform(-3, 3); return [[x, y], x * y > 0 ? 1 : 0]; } },
    diagonal: { label: 'Oblique linear boundary', gen: r => { const x = r.uniform(-3, 3), y = r.uniform(-3, 3); return [[x, y], y > 0.7 * x + 0.2 ? 1 : 0]; } },
    moons: { label: 'Two moons', gen: r => {
      const c = r.u() < 0.5 ? 0 : 1, t = Math.PI * r.u();
      const x = c ? 1 - Math.cos(t) : Math.cos(t), y = c ? 0.5 - Math.sin(t) : Math.sin(t);
      return [[1.8 * (x - 0.5) + r.normal(0, 0.22), 2.4 * (y - 0.25) + r.normal(0, 0.22)], c];
    } },
  };
  const DATA_OPTS = Object.keys(DATA2D).map(k => ({ value: k, label: DATA2D[k].label }));
  function make2D(kind, n, flip, r) {
    const X = [], y = [];
    for (let i = 0; i < n; i++) { const [x, c] = DATA2D[kind].gen(r); X.push(x); y.push(r.u() < flip ? 1 - c : c); }
    return { X, y };
  }
  // Raster colouring for a class-1 probability p.
  function shader() {
    const c0 = Atlas.rgb('s1'), c1 = Atlas.rgb('s2');
    return p => { const c = Atlas.mix(c0, c1, clamp(p, 0, 1)); return [c[0], c[1], c[2], Math.round(35 + 125 * Math.abs(2 * p - 1))]; };
  }
  function drawPoints(p, X, y, rOf) {
    for (let i = 0; i < X.length; i++) p.dot(X[i][0], X[i][1], { color: y[i] === 1 || y[i] === 1.0 ? 's2' : 's1', r: rOf ? rOf(i) : 3.4, strokeWidth: 1 });
  }

  /* =====================================================================
   * Shared: CART (classification with K classes or regression), weighted
   * ===================================================================== */
  const IMP = {
    gini: (c, W) => { let s = 0; for (let k = 0; k < c.length; k++) s += (c[k] / W) ** 2; return 1 - s; },
    entropy: (c, W) => { let s = 0; for (let k = 0; k < c.length; k++) if (c[k] > 0) { const q = c[k] / W; s -= q * Math.log2(q); } return s; },
    misclass: (c, W) => 1 - Math.max(...c) / W,
  };

  // X: array of feature vectors, y: labels 0..K-1 (class) or numbers (reg).
  // o: { task, K, crit, maxDepth, minLeaf, mtry, rng, w, idx }
  function growTree(X, y, o) {
    const p = X[0].length, K = o.K || 2, reg = o.task === 'reg';
    const w = o.w, maxDepth = o.maxDepth ?? 40, minLeaf = o.minLeaf ?? 1, mtry = Math.min(p, o.mtry || p), rng = o.rng;
    const imp = IMP[o.crit || 'gini'];
    const importance = new Float64Array(p);
    const wt = i => (w ? w[i] : 1);
    let count = 0;
    function summarize(idx) {
      if (reg) {
        let W = 0, s = 0, ss = 0;
        for (const i of idx) { const a = wt(i); W += a; s += a * y[i]; ss += a * y[i] * y[i]; }
        return { W, s, ss, value: s / W, risk: Math.max(0, ss - s * s / W) };
      }
      const c = new Array(K).fill(0); let W = 0;
      for (const i of idx) { c[y[i]] += wt(i); W += wt(i); }
      return { W, counts: c, value: c.map(v => v / W), risk: W - Math.max(...c) };
    }
    function bestSplit(idx, st) {
      const parent = reg ? st.risk : st.W * imp(st.counts, st.W);
      let feats = range(p);
      if (mtry < p) { for (let i = 0; i < mtry; i++) { const j = i + Math.floor(rng.u() * (p - i)); const t = feats[i]; feats[i] = feats[j]; feats[j] = t; } feats = feats.slice(0, mtry); }
      let best = null;
      const m = idx.length;
      for (const f of feats) {
        const ord = idx.slice().sort((a, b) => X[a][f] - X[b][f]);
        let WL = 0, sL = 0, ssL = 0;
        const cL = reg ? null : new Array(K).fill(0), cR = reg ? null : new Array(K).fill(0);
        for (let k = 0; k < m - 1; k++) {
          const i = ord[k], a = wt(i);
          WL += a;
          if (reg) { sL += a * y[i]; ssL += a * y[i] * y[i]; } else cL[y[i]] += a;
          if (k + 1 < minLeaf || m - k - 1 < minLeaf) continue;
          const xa = X[i][f], xb = X[ord[k + 1]][f];
          if (!(xb > xa)) continue;
          const WR = st.W - WL;
          if (WL <= 0 || WR <= 0) continue;
          let child;
          if (reg) {
            const sR = st.s - sL, ssR = st.ss - ssL;
            child = Math.max(0, ssL - sL * sL / WL) + Math.max(0, ssR - sR * sR / WR);
          } else {
            for (let c = 0; c < K; c++) cR[c] = st.counts[c] - cL[c];
            child = WL * imp(cL, WL) + WR * imp(cR, WR);
          }
          const gain = parent - child;
          if (!best || gain > best.gain) best = { gain, f, thr: (xa + xb) / 2 };
        }
      }
      if (best) best.parent = parent;
      return best;
    }
    function build(idx, depth) {
      const st = summarize(idx);
      const node = { n: idx.length, W: st.W, value: st.value, risk: st.risk, depth, leaf: true, pruneAt: Infinity, id: count++ };
      if (depth >= maxDepth || idx.length < 2 * minLeaf || st.risk <= 1e-12 * st.W) return node;
      const b = bestSplit(idx, st);
      if (!b || b.gain <= 1e-9 * b.parent) return node;
      importance[b.f] += b.gain;
      const L = [], R = [];
      for (const i of idx) (X[i][b.f] <= b.thr ? L : R).push(i);
      node.leaf = false; node.feat = b.f; node.thr = b.thr; node.gain = b.gain;
      node.left = build(L, depth + 1);
      node.right = build(R, depth + 1);
      return node;
    }
    const root = build(o.idx || range(X.length), 0);
    return { root, importance, size: count };
  }
  // Leaf reached by x, optionally truncating at depth dcap or at pruning stage k.
  function leafOf(t, x, dcap = Infinity, k = 1e9) {
    while (!t.leaf && t.depth < dcap && t.pruneAt > k) t = x[t.feat] <= t.thr ? t.left : t.right;
    return t;
  }
  function countLeaves(t, dcap = Infinity, k = 1e9) {
    if (t.leaf || t.depth >= dcap || t.pruneAt <= k) return 1;
    return countLeaves(t.left, dcap, k) + countLeaves(t.right, dcap, k);
  }
  // Weakest-link (cost-complexity) pruning. Sets node.pruneAt = first stage at which the node is a leaf.
  // Returns [{alpha, leaves, risk}] for stages 0..K (stage K is the root alone).
  function pruneSequence(root) {
    const seq = [];
    let k = 0, alpha = 0;
    const tol = 1e-10 * (root.risk + 1e-300);
    const post = t => {
      if (t.leaf || t.pruneAt <= k) { t._RT = t.risk; t._L = 1; return; }
      post(t.left); post(t.right);
      t._RT = t.left._RT + t.right._RT; t._L = t.left._L + t.right._L;
    };
    const collect = (t, out) => { if (t.leaf || t.pruneAt <= k) return; out.push(t); collect(t.left, out); collect(t.right, out); };
    for (let guard = 0; guard < 100000; guard++) {
      post(root);
      const internals = [];
      collect(root, internals);
      let gmin = Infinity;
      for (const t of internals) { t._g = (t.risk - t._RT) / (t._L - 1); if (t._g < gmin) gmin = t._g; }
      if (internals.length && gmin <= alpha + tol) {
        for (const t of internals) if (t._g <= gmin + tol) t.pruneAt = k;
        continue;
      }
      seq.push({ alpha, leaves: root._L, risk: root._RT });
      if (!internals.length) break;
      alpha = gmin; k++;
    }
    return seq;
  }
  const stageFor = (seq, beta) => { let s = 0; for (let j = 0; j < seq.length; j++) if (seq[j].alpha <= beta * (1 + 1e-12)) s = j; return s; };

  /* =====================================================================
   * Shared: small dense linear algebra
   * ===================================================================== */
  function chol(A, jitter = 0) {
    const n = A.length, L = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j] + (i === j ? jitter : 0);
      const Li = L[i], Lj = L[j];
      for (let k = 0; k < j; k++) s -= Li[k] * Lj[k];
      if (i === j) { if (!(s > 0)) return null; Li[i] = Math.sqrt(s); } else Li[j] = s / Lj[j];
    }
    return L;
  }
  function cholRobust(A, scale = 1) {
    for (const j of [0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]) { const L = chol(A, j * scale); if (L) return { L, jitter: j * scale }; }
    return null;
  }
  function fwd(L, b) { const n = L.length, x = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; const Li = L[i]; for (let k = 0; k < i; k++) s -= Li[k] * x[k]; x[i] = s / Li[i]; } return x; }
  function bwd(L, b) { const n = L.length, x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; }
  const cholSolve = (L, b) => bwd(L, fwd(L, b));
  const dotv = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

  /* =====================================================================
   * Lab: decision-trees-cart
   * ===================================================================== */
  Atlas.lab('decision-trees-cart', {
    title: 'Growing a classification tree, one level at a time',
    blurb: 'The tree is grown greedily to full depth once; the slider truncates it at depth $d$. Shading is the leaf class proportion, lines are the splits. Try the oblique boundary (staircases), XOR with the misclassification criterion (the root split has no gain), and watch test error turn up as $d$ grows.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: DATA_OPTS, value: 'circle', onChange: newData });
      const crit = UI.segmented(controls, { label: 'Split criterion', options: [{ value: 'gini', label: 'Gini' }, { value: 'entropy', label: 'Entropy' }, { value: 'misclass', label: 'Misclass.' }], value: 'gini', onChange: grow });
      const depth = UI.slider(controls, { label: 'Depth $d$', min: 0, max: 12, step: 1, value: 3, onInput: draw });
      const minLeaf = UI.slider(controls, { label: 'Minimum leaf size', min: 1, max: 30, step: 1, value: 1, onChange: grow });
      const nS = UI.slider(controls, { label: 'Training size $n$', min: 50, max: 400, step: 10, value: 200, onChange: newData });
      const flip = UI.slider(controls, { label: 'Label noise (flip prob.)', min: 0, max: 0.3, step: 0.01, value: 0.05, onChange: newData });
      const showSplits = UI.toggle(controls, { label: 'Show split lines', value: true, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [-3, 3], xlabel: 'x₁', ylabel: 'x₂', title: 'Partition at depth d' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 12], y: [0, 0.6], xlabel: 'depth d', ylabel: 'error rate', title: 'Training vs test error' });
      const sLeaves = UI.stat(stats, 'Leaves $|T|$');
      const sTr = UI.stat(stats, 'Training error');
      const sTe = UI.stat(stats, 'Test error', { accent: 'accent' });
      const sRoot = UI.stat(stats, 'Root split');
      let D, T, tree, errTr, errTe;
      function newData() {
        D = make2D(ds.get(), Math.round(nS.get()), flip.get(), ctx.rng);
        T = make2D(ds.get(), 2000, flip.get(), ctx.rng);
        grow();
      }
      function grow() {
        tree = growTree(D.X, D.y, { K: 2, crit: crit.get(), maxDepth: 12, minLeaf: Math.round(minLeaf.get()) });
        const err = (set, d) => { let e = 0; for (let i = 0; i < set.X.length; i++) { const v = leafOf(tree.root, set.X[i], d).value; if ((v[1] > v[0] ? 1 : 0) !== set.y[i]) e++; } return e / set.X.length; };
        errTr = range(13).map(d => err(D, d));
        errTe = range(13).map(d => err(T, d));
        draw();
      }
      function draw() {
        const d = Math.round(depth.get()), sh = shader();
        pL.clear();
        pL.raster((x, y) => sh(leafOf(tree.root, [x, y], d).value[1]), { nx: 140, ny: 90, pixelated: true });
        if (showSplits.get()) {
          const rec = (t, x0, x1, y0, y1) => {
            if (t.leaf || t.depth >= d) return;
            const wd = Math.max(0.8, 2.4 - 0.35 * t.depth);
            if (t.feat === 0) { pL.segment(t.thr, y0, t.thr, y1, { color: 'ink', width: wd, opacity: 0.8 }); rec(t.left, x0, t.thr, y0, y1); rec(t.right, t.thr, x1, y0, y1); }
            else { pL.segment(x0, t.thr, x1, t.thr, { color: 'ink', width: wd, opacity: 0.8 }); rec(t.left, x0, x1, y0, t.thr); rec(t.right, x0, x1, t.thr, y1); }
          };
          rec(tree.root, -50, 50, -50, 50);
        }
        drawPoints(pL, D.X, D.y);
        pR.clear();
        pR.line(errTr.map((e, i) => [i, e]), { color: 's1', width: 2 });
        pR.line(errTe.map((e, i) => [i, e]), { color: 's2', width: 2 });
        pR.points(errTr.map((e, i) => [i, e]), { color: 's1', r: 2.5 });
        pR.points(errTe.map((e, i) => [i, e]), { color: 's2', r: 2.5 });
        pR.vline(d, { color: 'ink' });
        pR.legend([{ label: 'training', color: 's1' }, { label: 'test (n = 2000)', color: 's2' }]);
        sLeaves.set(String(countLeaves(tree.root, d)), `full tree: ${countLeaves(tree.root)} leaves`);
        sTr.set(S.fmt(errTr[d], 3));
        const best = S.argmin(errTe);
        sTe.set(S.fmt(errTe[d], 3), `best depth ${best} (${S.fmt(errTe[best], 3)})`);
        const r = tree.root;
        sRoot.set(r.leaf ? 'none' : `x${r.feat === 0 ? '₁' : '₂'} ≤ ${S.fmt(r.thr, 2)}`, r.leaf ? 'no split reduces impurity' : `impurity decrease ${S.fmt(r.gain / r.W, 3)} per obs.`);
      }
      newData();
    },
  });

  /* =====================================================================
   * Lab: decision-trees-pruning (1-D regression, weakest-link pruning + CV)
   * ===================================================================== */
  Atlas.lab('decision-trees-pruning', {
    title: 'Cost-complexity pruning and cross-validation',
    blurb: 'A regression tree is grown until leaves have 2 points, then pruned by weakest links into a nested sequence $T_0 \\supset T_1 \\supset \\cdots \\supset \\{\\text{root}\\}$. Right: training error falls monotonically with size; 10-fold CV (with ±1 SE band) and fresh test error are U-shaped. Drag the size slider along the sequence.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nS = UI.slider(controls, { label: 'Training size $n$', min: 40, max: 400, step: 10, value: 120, onChange: newData });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.05, max: 1, step: 0.05, value: 0.4, onChange: newData });
      let seq = null;
      const kS = UI.slider(controls, { label: 'Subtree in the sequence', min: 0, max: 1, step: 1, value: 1, onInput: draw, fmt: v => (seq && seq[seq.length - 1 - v] ? seq[seq.length - 1 - v].leaves + ' leaves' : '') });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newData, { primary: true });
      UI.button(row, 'CV minimum', () => { kS.set(seq.length - 1 - kMin); draw(); });
      UI.button(row, '1-SE rule', () => { kS.set(seq.length - 1 - k1se); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-2, 3], xlabel: 'x', ylabel: 'y', title: 'Pruned tree fit' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [1, 100], y: [0, 1], logx: true, xlabel: 'leaves |T| (log scale)', ylabel: 'mean squared error', title: 'Error along the pruning sequence' });
      const sA = UI.stat(stats, 'Penalty $\\alpha_k$');
      const sLv = UI.stat(stats, 'Leaves $|T_k|$');
      const sTr = UI.stat(stats, 'Training MSE');
      const sCV = UI.stat(stats, '10-fold CV MSE', { accent: 'accent' });
      const sTe = UI.stat(stats, 'Test MSE');
      const f = x => Math.sin(6 * x) + (x > 0.6 ? 1.2 : 0);
      let x, y, tree, cv, cvse, te, kMin, k1se;
      const xt = range(1500).map(i => (i + 0.5) / 1500);
      let yt;
      function newData() {
        const n = Math.round(nS.get()), s = sig.get();
        x = ctx.rng.array(n, () => ctx.rng.u());
        y = x.map(v => f(v) + ctx.rng.normal(0, s));
        yt = xt.map(v => f(v) + ctx.rng.normal(0, s));
        const X = x.map(v => [v]);
        tree = growTree(X, y, { task: 'reg', minLeaf: 1, maxDepth: 40 });
        seq = pruneSequence(tree.root);
        const K = seq.length;
        const beta = seq.map((s0, j) => (j + 1 < K ? Math.sqrt(s0.alpha * seq[j + 1].alpha) : Infinity));
        // 10-fold CV of the whole pruning procedure, evaluated at the geometric midpoints beta_j
        const perm = ctx.rng.shuffle(range(n).slice());
        const sq = range(K).map(() => new Float64Array(n));
        for (let fo = 0; fo < 10; fo++) {
          const test = [], train = [];
          perm.forEach((i, r) => (r % 10 === fo ? test : train).push(i));
          const tf = growTree(X, y, { task: 'reg', minLeaf: 1, maxDepth: 40, idx: train });
          const sf = pruneSequence(tf.root);
          for (let j = 0; j < K; j++) {
            const st = stageFor(sf, beta[j]);
            for (const i of test) sq[j][i] = (y[i] - leafOf(tf.root, X[i], Infinity, st).value) ** 2;
          }
        }
        cv = sq.map(a => S.mean(a));
        cvse = sq.map(a => S.sd(a) / Math.sqrt(n));
        te = seq.map((_, k) => { let e = 0; for (let i = 0; i < xt.length; i++) e += (yt[i] - leafOf(tree.root, [xt[i]], Infinity, k).value) ** 2; return e / xt.length; });
        kMin = S.argmin(cv);
        // 1-SE rule: smallest tree (largest k) with CV within one SE of the minimum
        k1se = kMin;
        for (let k = K - 1; k >= 0; k--) if (cv[k] <= cv[kMin] + cvse[kMin]) { k1se = k; break; }
        kS.input.max = K - 1;
        kS.set(K - 1 - kMin);
        draw();
      }
      function draw() {
        const K = seq.length, k = clamp(K - 1 - Math.round(kS.get()), 0, K - 1);
        pL.clear();
        pL.points(x.map((v, i) => [v, y[i]]), { color: 's3', r: 2.6, opacity: 0.7 });
        pL.line(S.linspace(0, 1, 400).map(v => [v, f(v)]), { color: 'ink', width: 1.3, dash: true, opacity: 0.7 });
        const g = S.linspace(0, 1, 801);
        pL.line(g.map(v => [v, leafOf(tree.root, [v], Infinity, k).value]), { color: 's1', width: 2.2 });
        pL.legend([{ label: 'true f', color: 'ink', dash: true }, { label: `T_${k}`, color: 's1' }]);
        const lv = seq.map(s0 => s0.leaves), n = x.length;
        const ymax = Math.max(1.2 * Math.max(...cv.slice(0, K)), 0.1);
        pR.setDomain([0.8, Math.max(10, lv[0] * 1.25)], [0, Math.min(ymax, 3)]).clear();
        pR.area(lv.map((l, j) => [l, cv[j] + cvse[j]]), { color: 's2', opacity: 0.15, base: lv.map((l, j) => [l, cv[j] - cvse[j]]) });
        pR.step(lv.map((l, j) => [l, seq[j].risk / n]).reverse(), { color: 's1', width: 2 });
        pR.line(lv.map((l, j) => [l, cv[j]]), { color: 's2', width: 2 });
        pR.line(lv.map((l, j) => [l, te[j]]), { color: 's4', width: 1.6, dash: true });
        pR.hline(sig.get() ** 2, { color: 'muted', width: 1 });
        pR.vline(lv[k], { color: 'ink', dash: false, width: 1.4 });
        pR.dot(lv[kMin], cv[kMin], { color: 's2', r: 4.5 });
        pR.dot(lv[k1se], cv[k1se], { color: 's2', r: 4.5, opacity: 0.3, stroke: 's2' });
        pR.legend([{ label: 'training', color: 's1' }, { label: '10-fold CV ± 1 SE', color: 's2' }, { label: 'test', color: 's4', dash: true }, { label: 'σ² (floor)', color: 'muted' }]);
        sA.set(S.fmt(seq[k].alpha, 4), `stage k = ${k} of ${K - 1}`);
        sLv.set(String(lv[k]), `CV pick ${lv[kMin]}, 1-SE pick ${lv[k1se]}`);
        sTr.set(S.fmt(seq[k].risk / n, 3));
        sCV.set(S.fmt(cv[k], 3), '± ' + S.fmt(cvse[k], 3));
        sTe.set(S.fmt(te[k], 3), `σ² = ${S.fmt(sig.get() ** 2, 3)}; test-optimal size ${lv[S.argmin(te)]}`);
      }
      newData();
    },
  });

  /* =====================================================================
   * Lab: bagging-random-forests-surfaces
   * ===================================================================== */
  function bootIdx(n, r) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = Math.floor(r.u() * n); return a; }

  Atlas.lab('bagging-random-forests-surfaces', {
    title: 'One tree vs. an ensemble of trees',
    blurb: 'Left: a single fully grown tree. Right: the average vote of $B$ trees grown on bootstrap samples (bagging), optionally choosing the split variable from a random subset of size $m=1$ (random forest). Press <em>New training sample</em> repeatedly: the single tree lurches, the ensemble barely moves.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: DATA_OPTS, value: 'moons', onChange: newData });
      const nS = UI.slider(controls, { label: 'Training size $n$', min: 60, max: 300, step: 10, value: 150, onChange: newData });
      const flip = UI.slider(controls, { label: 'Label noise', min: 0, max: 0.25, step: 0.01, value: 0.08, onChange: newData });
      const kind = UI.segmented(controls, { label: 'Ensemble', options: [{ value: 'bag', label: 'Bagging ($m=p$)' }, { value: 'rf', label: 'Random forest ($m=1$)' }], value: 'bag', onChange: fit });
      const Bs = UI.slider(controls, { label: 'Number of trees $B$', min: 1, max: 200, step: 1, value: 100, onChange: fit });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New training sample', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [-3, 3], xlabel: 'x₁', ylabel: 'x₂', title: 'Single deep tree' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [-3, 3], xlabel: 'x₁', ylabel: 'x₂', title: 'Ensemble average vote' });
      const sT = UI.stat(stats, 'Single tree test error');
      const sE = UI.stat(stats, 'Ensemble test error', { accent: 'accent' });
      const sO = UI.stat(stats, 'OOB error (ensemble)');
      const sD = UI.stat(stats, 'Tree-to-tree disagreement');
      let D, T, single, trees;
      function newData() {
        D = make2D(ds.get(), Math.round(nS.get()), flip.get(), ctx.rng);
        T = make2D(ds.get(), 1500, flip.get(), ctx.rng);
        single = growTree(D.X, D.y, { K: 2, minLeaf: 1 }).root;
        fit();
      }
      function fit() {
        const B = Math.round(Bs.get()), m = kind.get() === 'rf' ? 1 : 2, n = D.X.length;
        trees = [];
        const oobSum = new Float64Array(n), oobCnt = new Float64Array(n);
        for (let b = 0; b < B; b++) {
          const idx = bootIdx(n, ctx.rng), inBag = new Uint8Array(n);
          for (const i of idx) inBag[i] = 1;
          const t = growTree(D.X, D.y, { K: 2, minLeaf: 1, mtry: m, rng: ctx.rng, idx }).root;
          trees.push(t);
          for (let i = 0; i < n; i++) if (!inBag[i]) { oobSum[i] += leafOf(t, D.X[i]).value[1] > 0.5 ? 1 : 0; oobCnt[i]++; }
        }
        let oe = 0, oc = 0;
        for (let i = 0; i < n; i++) if (oobCnt[i]) { oc++; if ((oobSum[i] / oobCnt[i] > 0.5 ? 1 : 0) !== D.y[i]) oe++; }
        const vote = x => { let s = 0; for (const t of trees) s += leafOf(t, x).value[1] > 0.5 ? 1 : 0; return s / trees.length; };
        let e1 = 0, eB = 0, dis = 0;
        for (let i = 0; i < T.X.length; i++) {
          const v1 = leafOf(single, T.X[i]).value[1] > 0.5 ? 1 : 0, pv = vote(T.X[i]);
          if (v1 !== T.y[i]) e1++;
          if ((pv > 0.5 ? 1 : 0) !== T.y[i]) eB++;
          dis += B > 1 ? 2 * pv * (1 - pv) * B / (B - 1) : 0;
        }
        const nt = T.X.length;
        const sh = shader();
        pL.clear();
        pL.raster((x, y) => sh(leafOf(single, [x, y]).value[1] > 0.5 ? 1 : 0), { nx: 110, ny: 72, pixelated: true });
        drawPoints(pL, D.X, D.y);
        pR.clear();
        pR.raster((x, y) => sh(vote([x, y])), { nx: 110, ny: 72 });
        drawPoints(pR, D.X, D.y);
        sT.set(S.fmt(e1 / nt, 3));
        sE.set(S.fmt(eB / nt, 3), `B = ${B}`);
        sO.set(oc ? S.fmt(oe / oc, 3) : '—', `${oc} of ${n} points have OOB votes`);
        sD.set(S.fmt(dis / nt, 3), 'P(two random trees disagree)');
      }
      newData();
    },
  });

  /* =====================================================================
   * Lab: bagging-random-forests-ntrees (Friedman #1 regression)
   * ===================================================================== */
  Atlas.lab('bagging-random-forests-ntrees', {
    title: 'Test error vs. number of trees, OOB error, and importance',
    blurb: 'Regression on $p=10$ features where only $x_1,\\dots,x_5$ matter: $f(x)=10\\sin(\\pi x_1x_2)+20(x_3-\\tfrac12)^2+10x_4+5x_5$. Both ensembles plateau as $B$ grows (no overfitting in $B$); the plateau is lower when $m\\lt p$ decorrelates the trees. <em>Estimate</em> measures $\\rho$ and $\\sigma^2$ by regrowing on fresh training sets. Toggle the correlated copy to see importance split between twins.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mS = UI.slider(controls, { label: 'Random-forest $m$ (features per split)', min: 1, max: 10, step: 1, value: 3, onChange: reset });
      const nS = UI.slider(controls, { label: 'Training size $n$', min: 100, max: 400, step: 50, value: 200, onChange: reset });
      const twin = UI.toggle(controls, { label: '$x_6$ is a noisy copy of $x_4$', value: false, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Grow forests', '❚❚ Pause']);
      UI.button(row, 'Reset', reset);
      UI.button(row, 'Estimate ρ, σ²', estimate);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 300], y: [0, 20], xlabel: 'number of trees B', ylabel: 'test MSE', title: 'Error as trees are added' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 10], y: [0, 1.1], xticks: 0, ylabel: 'relative importance', title: 'Variable importance (random forest)' });
      const sB = UI.stat(stats, 'Bagging test MSE ($m=10$)');
      const sR = UI.stat(stats, 'Forest test MSE', { accent: 'accent' });
      const sO = UI.stat(stats, 'Forest OOB MSE');
      const sRho = UI.stat(stats, '$\\hat\\rho$: bagging / forest');
      const sS2 = UI.stat(stats, '$\\hat\\sigma^2$ single tree: bag / forest');
      const P = 10, f = x => 10 * Math.sin(Math.PI * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4];
      const gen = (r, n) => {
        const X = [], y = [];
        for (let i = 0; i < n; i++) {
          const x = r.array(P, () => r.u());
          if (twin.get()) x[5] = clamp(x[3] + r.normal(0, 0.05), 0, 1);
          X.push(x); y.push(f(x) + r.normal(0, 1));
        }
        return { X, y };
      };
      let D, T, bag, rf, curves;
      function reset() {
        loop.stop();
        D = gen(ctx.rng, Math.round(nS.get()));
        T = gen(ctx.rng, 300);
        const n = D.X.length;
        const mk = () => ({ sum: new Float64Array(T.X.length), B: 0, oobSum: new Float64Array(n), oobCnt: new Float64Array(n), mdi: new Float64Array(P), perm: new Float64Array(P) });
        bag = mk(); rf = mk();
        curves = { bag: [], rf: [], oob: [] };
        draw();
      }
      function addTree(E, m) {
        const n = D.X.length, idx = bootIdx(n, ctx.rng), inBag = new Uint8Array(n);
        for (const i of idx) inBag[i] = 1;
        const t = growTree(D.X, D.y, { task: 'reg', minLeaf: 5, mtry: m, rng: ctx.rng, idx });
        for (let i = 0; i < T.X.length; i++) E.sum[i] += leafOf(t.root, T.X[i]).value;
        const oob = [];
        for (let i = 0; i < n; i++) if (!inBag[i]) { oob.push(i); E.oobSum[i] += leafOf(t.root, D.X[i]).value; E.oobCnt[i]++; }
        for (let j = 0; j < P; j++) E.mdi[j] += t.importance[j];
        if (oob.length > 1) {
          const base = S.mean(oob.map(i => (D.y[i] - leafOf(t.root, D.X[i]).value) ** 2));
          for (let j = 0; j < P; j++) {
            const vals = ctx.rng.shuffle(oob.map(i => D.X[i][j]));
            let e = 0;
            oob.forEach((i, r) => { const x = D.X[i].slice(); x[j] = vals[r]; e += (D.y[i] - leafOf(t.root, x).value) ** 2; });
            E.perm[j] += e / oob.length - base;
          }
        }
        E.B++;
      }
      const mse = E => { let e = 0; for (let i = 0; i < T.X.length; i++) e += (T.y[i] - E.sum[i] / E.B) ** 2; return e / T.X.length; };
      const oobMse = E => { let e = 0, c = 0; for (let i = 0; i < D.X.length; i++) if (E.oobCnt[i]) { e += (D.y[i] - E.oobSum[i] / E.oobCnt[i]) ** 2; c++; } return c ? e / c : NaN; };
      function step() {
        for (let k = 0; k < 2; k++) {
          if (bag.B >= 300) { draw(); return false; }
          addTree(bag, P); addTree(rf, Math.round(mS.get()));
          curves.bag.push([bag.B, mse(bag)]); curves.rf.push([rf.B, mse(rf)]); curves.oob.push([rf.B, oobMse(rf)]);
        }
        draw();
      }
      function estimate() {
        const R = 6, Tn = 8, r = ctx.rng, nt = 100;
        const go = m => {
          const preds = range(R).map(() => range(Tn).map(() => new Float64Array(nt)));
          for (let a = 0; a < R; a++) {
            const Dr = gen(r, D.X.length);
            for (let b = 0; b < Tn; b++) {
              const t = growTree(Dr.X, Dr.y, { task: 'reg', minLeaf: 5, mtry: m, rng: r, idx: bootIdx(Dr.X.length, r) });
              for (let i = 0; i < nt; i++) preds[a][b][i] = leafOf(t.root, T.X[i]).value;
            }
          }
          let sE = 0, sTot = 0;
          for (let i = 0; i < nt; i++) {
            const gm = range(R).map(a => S.mean(range(Tn).map(b => preds[a][b][i])));
            const within = S.mean(range(R).map(a => S.variance(range(Tn).map(b => preds[a][b][i]))));
            const between = S.variance(gm);
            const vE = Math.max(0, between - within / Tn);
            sE += vE; sTot += vE + within;
          }
          return { rho: sE / sTot, s2: sTot / nt };
        };
        const a = go(P), b = go(Math.round(mS.get()));
        sRho.set(`${S.fmt(a.rho, 2)} / ${S.fmt(b.rho, 2)}`, `floors ρσ²: ${S.fmt(a.rho * a.s2, 2)} / ${S.fmt(b.rho * b.s2, 2)}`);
        sS2.set(`${S.fmt(a.s2, 2)} / ${S.fmt(b.s2, 2)}`, `${6} training sets × ${8} trees`);
      }
      function draw() {
        const early = curves.bag.slice(0, 3).concat(curves.rf.slice(0, 3)).map(p => p[1]).filter(Number.isFinite);
        const ymax = early.length ? Math.max(...early, 4) * 1.1 : 20;
        pL.setDomain([0, Math.max(50, bag.B)], [0, ymax]).clear();
        pL.hline(1, { color: 'muted', width: 1 });
        pL.line(curves.bag, { color: 's1', width: 2 });
        pL.line(curves.rf, { color: 's2', width: 2 });
        pL.line(curves.oob.filter(p => p[0] >= 5), { color: 's2', width: 1.5, dash: true });
        pL.legend([{ label: 'bagging (m = 10)', color: 's1' }, { label: `forest (m = ${Math.round(mS.get())})`, color: 's2' }, { label: 'forest OOB', color: 's2', dash: true }, { label: 'noise σ² = 1', color: 'muted' }]);
        pR.clear();
        if (rf.B) {
          const mx = Math.max(...rf.mdi), px = Math.max(...rf.perm, 1e-12);
          const bm = [], bp = [];
          for (let j = 0; j < P; j++) {
            bm.push({ x0: j + 0.1, x1: j + 0.48, y: rf.mdi[j] / mx });
            bp.push({ x0: j + 0.52, x1: j + 0.9, y: Math.max(0, rf.perm[j]) / px });
          }
          pR.bars(bm, { color: 's3', opacity: 0.8 });
          pR.bars(bp, { color: 's5', opacity: 0.8 });
        }
        for (let j = 0; j < P; j++) pR.text(j + 0.5, 0, 'x' + (j + 1), { anchor: 'middle', dy: 16, size: 11, color: j < 5 ? 'ink' : 'muted' });
        pR.legend([{ label: 'impurity decrease (MDI)', color: 's3', fill: true }, { label: 'OOB permutation', color: 's5', fill: true }]);
        sB.set(bag.B ? S.fmt(mse(bag), 2) : '—', `B = ${bag.B}`);
        sR.set(rf.B ? S.fmt(mse(rf), 2) : '—', `B = ${rf.B}`);
        sO.set(rf.B ? S.fmt(oobMse(rf), 2) : '—', 'no test data used');
      }
      reset();
    },
  });

  /* =====================================================================
   * Lab: boosting-adaboost
   * ===================================================================== */
  // Weighted decision stump: h(x) = s if x_f > thr else -s, labels y in {-1,+1}.
  function bestStump(X, y, w, orders) {
    const p = X[0].length, n = y.length;
    let P = 0, N = 0;
    for (let i = 0; i < n; i++) (y[i] > 0 ? (P += w[i]) : (N += w[i]));
    let best = { err: Math.min(P, N), f: 0, thr: -1e9, s: P >= N ? 1 : -1 };
    for (let f = 0; f < p; f++) {
      const ord = orders[f];
      let lp = 0, ln = 0;
      for (let k = 0; k < n - 1; k++) {
        const i = ord[k];
        if (y[i] > 0) lp += w[i]; else ln += w[i];
        const xa = X[i][f], xb = X[ord[k + 1]][f];
        if (!(xb > xa)) continue;
        const ePlus = lp + (N - ln), eMinus = ln + (P - lp);
        if (ePlus < best.err) best = { err: ePlus, f, thr: (xa + xb) / 2, s: 1 };
        if (eMinus < best.err) best = { err: eMinus, f, thr: (xa + xb) / 2, s: -1 };
      }
    }
    return best;
  }
  const stumpAt = (st, x) => (x[st.f] > st.thr ? st.s : -st.s);

  Atlas.lab('boosting-adaboost', {
    title: 'AdaBoost with decision stumps, round by round',
    blurb: 'Point size is the current weight $w_i$; the dashed line is the newest stump. Shading is $\\hat P(y=1\\mid x)=1/(1+e^{-2F(x)})$, the probability implied by the exponential-loss minimiser. The training error sits under the bound $\\prod_m 2\\sqrt{\\varepsilon_m(1-\\varepsilon_m)}$; switch to <em>Margins</em> to watch the margin distribution keep moving right after training error hits zero.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: DATA_OPTS, value: 'circle', onChange: newData });
      const nS = UI.slider(controls, { label: 'Training size $n$', min: 50, max: 300, step: 10, value: 150, onChange: newData });
      const flip = UI.slider(controls, { label: 'Label noise', min: 0, max: 0.2, step: 0.01, value: 0.03, onChange: newData });
      const view = UI.segmented(controls, { label: 'Right panel', options: ['Errors', 'Margins'], value: 'Errors', onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Boost', '❚❚ Pause']);
      UI.button(row, '+1 round', () => { stepRound(); draw(); });
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [-3, 3], xlabel: 'x₁', ylabel: 'x₂', title: 'Ensemble and current weights' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 50], y: [0, 0.7], xlabel: 'round m', ylabel: 'error', title: 'Training error, its bound, test error' });
      const sM = UI.stat(stats, 'Round $m$');
      const sEps = UI.stat(stats, 'Weighted error $\\varepsilon_m$');
      const sAl = UI.stat(stats, '$\\alpha_m=\\frac12\\log\\frac{1-\\varepsilon_m}{\\varepsilon_m}$');
      const sTr = UI.stat(stats, 'Training error');
      const sBd = UI.stat(stats, 'Bound $\\prod 2\\sqrt{\\varepsilon(1-\\varepsilon)}$');
      const sTe = UI.stat(stats, 'Test error', { accent: 'accent' });
      const MAXM = 400, NX = 105, NY = 70;
      let D, T, yy, yt, orders, w, Ftr, Fte, Fgrid, stumps, alphaSum, bound, hist, snaps, done, last;
      function newData() {
        D = make2D(ds.get(), Math.round(nS.get()), flip.get(), ctx.rng);
        T = make2D(ds.get(), 1500, flip.get(), ctx.rng);
        yy = D.y.map(v => (v ? 1 : -1)); yt = T.y.map(v => (v ? 1 : -1));
        orders = [0, 1].map(f => range(D.X.length).sort((a, b) => D.X[a][f] - D.X[b][f]));
        reset();
      }
      function reset() {
        loop.stop();
        const n = D.X.length;
        w = new Float64Array(n).fill(1 / n);
        Ftr = new Float64Array(n); Fte = new Float64Array(T.X.length); Fgrid = new Float64Array(NX * NY);
        stumps = []; alphaSum = 0; bound = 1; hist = []; snaps = []; done = false; last = null;
        draw();
      }
      function stepRound() {
        if (done || stumps.length >= MAXM) { done = true; return; }
        const st = bestStump(D.X, yy, w, orders);
        const eps = clamp(st.err, 1e-10, 1);
        if (eps >= 0.5 - 1e-12) { done = true; return; }
        const a = 0.5 * Math.log((1 - eps) / eps);
        st.alpha = a; st.eps = eps; stumps.push(st); last = st;
        let Z = 0;
        for (let i = 0; i < w.length; i++) { const h = stumpAt(st, D.X[i]); Ftr[i] += a * h; w[i] *= Math.exp(-a * yy[i] * h); Z += w[i]; }
        for (let i = 0; i < w.length; i++) w[i] /= Z;
        for (let i = 0; i < Fte.length; i++) Fte[i] += a * stumpAt(st, T.X[i]);
        for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) Fgrid[j * NX + i] += a * stumpAt(st, [-3 + 6 * (i + 0.5) / NX, 3 - 6 * (j + 0.5) / NY]);
        alphaSum += a; bound *= 2 * Math.sqrt(eps * (1 - eps));
        let etr = 0, ete = 0;
        for (let i = 0; i < Ftr.length; i++) if (yy[i] * Ftr[i] <= 0) etr++;
        for (let i = 0; i < Fte.length; i++) if (yt[i] * Fte[i] <= 0) ete++;
        hist.push({ m: stumps.length, tr: etr / Ftr.length, te: ete / Fte.length, bound, eps, a });
        if ([5, 25, 100].includes(stumps.length)) snaps.push({ m: stumps.length, mg: margins() });
      }
      const margins = () => Array.from(Ftr, (F, i) => yy[i] * F / alphaSum).sort((a, b) => a - b);
      let acc = 0;
      function tick(dt) {
        acc += dt;
        let k = 0;
        while (acc > 0.04 && k < 3) { stepRound(); acc -= 0.04; k++; }
        draw();
        if (done) return false;
      }
      function draw() {
        const sh = shader(), M = stumps.length;
        pL.clear();
        pL.raster((x, y) => {
          const i = clamp(Math.floor((x + 3) / 6 * NX), 0, NX - 1), j = clamp(Math.floor((3 - y) / 6 * NY), 0, NY - 1);
          return M ? sh(1 / (1 + Math.exp(-2 * Fgrid[j * NX + i]))) : [0, 0, 0, 0];
        }, { nx: NX, ny: NY });
        const n = D.X.length;
        drawPoints(pL, D.X, D.y, i => clamp(1.2 + 3.2 * Math.sqrt(n * w[i]), 1.2, 16));
        if (last) {
          if (last.f === 0) pL.vline(last.thr, { color: 'ink', width: 2 }); else pL.hline(last.thr, { color: 'ink', width: 2 });
        }
        pR.clear();
        if (view.get() === 'Errors') {
          pR.o.title = 'Training error, its bound, test error';
          pR.o.xlabel = 'round m'; pR.o.ylabel = 'error';
          pR.setDomain([0, Math.max(50, M)], [0, 0.7]);
          pR.line(hist.map(h => [h.m, h.bound]), { color: 'muted', width: 1.6, dash: true });
          pR.step(hist.map(h => [h.m, h.tr]), { color: 's1', width: 2 });
          pR.line(hist.map(h => [h.m, h.te]), { color: 's2', width: 2 });
          pR.line(hist.map(h => [h.m, h.eps]), { color: 's4', width: 1, opacity: 0.6 });
          pR.legend([{ label: 'bound ∏ 2√(ε(1−ε))', color: 'muted', dash: true }, { label: 'training error', color: 's1' }, { label: 'test error', color: 's2' }, { label: 'ε_m (weighted)', color: 's4' }]);
        } else {
          pR.o.title = 'Cumulative distribution of normalised margins';
          pR.o.xlabel = 'margin y F(x) / Σα'; pR.o.ylabel = 'fraction ≤ margin';
          pR.setDomain([-1, 1], [0, 1]);
          pR.vline(0, { color: 'muted' });
          const cdf = mg => mg.map((v, i) => [v, (i + 1) / mg.length]);
          snaps.forEach((s, k) => pR.step([[-1, 0]].concat(cdf(s.mg)), { color: ['s5', 's4', 's3'][k], width: 1.5, opacity: 0.8 }));
          if (M) pR.step([[-1, 0]].concat(cdf(margins())).concat([[1, 1]]), { color: 's1', width: 2.2 });
          pR.legend(snaps.map((s, k) => ({ label: 'm = ' + s.m, color: ['s5', 's4', 's3'][k] })).concat(M ? [{ label: 'm = ' + M + ' (now)', color: 's1' }] : []), { left: true });
        }
        const h = hist[hist.length - 1];
        sM.set(String(M), done ? (M >= MAXM ? 'cap reached' : 'stopped: ε ≥ 1/2') : '');
        sEps.set(h ? S.fmt(h.eps, 3) : '—');
        sAl.set(h ? S.fmt(h.a, 3) : '—');
        sTr.set(h ? S.fmt(h.tr, 3) : '—');
        sBd.set(h ? S.fmt(h.bound, 3) : '1');
        sTe.set(h ? S.fmt(h.te, 3) : '—', h ? 'best so far ' + S.fmt(Math.min(...hist.map(q => q.te)), 3) : '');
      }
      newData();
    },
  });

  /* =====================================================================
   * Lab: boosting-gradient (1-D regression)
   * ===================================================================== */
  Atlas.lab('boosting-gradient', {
    title: 'Gradient boosting: fitting trees to pseudo-residuals',
    blurb: 'Each round fits a small regression tree $h_m$ (green) to the negative gradient of the loss, then adds $\\nu h_m$ to the fit. Small $\\nu$ needs more rounds but reaches a lower test error; with $\\nu=1$ the fit chases noise quickly. Add outliers and compare squared with absolute loss.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nu = UI.slider(controls, { label: 'Learning rate $\\nu$', min: 0.01, max: 1, value: 0.1, log: true, fmt: v => S.fmt(v, 2), onChange: reset });
      const dep = UI.slider(controls, { label: 'Tree depth $J$', min: 1, max: 4, step: 1, value: 2, onChange: reset });
      const sub = UI.slider(controls, { label: 'Subsample fraction $\\eta$', min: 0.3, max: 1, step: 0.05, value: 1, onChange: reset });
      const loss = UI.segmented(controls, { label: 'Loss', options: [{ value: 'l2', label: 'Squared' }, { value: 'l1', label: 'Absolute' }], value: 'l2', onChange: reset });
      const out = UI.toggle(controls, { label: 'Contaminate 8% of training $y$ with outliers', value: false, onChange: newData });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Boost', '❚❚ Pause']);
      UI.button(row, '+1 round', () => { stepRound(); draw(); });
      UI.button(row, 'Reset', reset);
      UI.button(row, 'New data', newData);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 10], y: [-3, 3], xlabel: 'x', ylabel: 'y', title: 'Fit F_m and newest tree h_m' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 100], y: [0, 1], xlabel: 'round m', ylabel: 'loss', title: 'Training vs test loss' });
      const sM = UI.stat(stats, 'Round $m$');
      const sTr = UI.stat(stats, 'Training loss');
      const sTe = UI.stat(stats, 'Test loss', { accent: 'accent' });
      const sBest = UI.stat(stats, 'Best test loss (round)');
      const sTru = UI.stat(stats, '$\\int (F_m-f)^2$ vs. truth');
      const f = x => Math.sin(1.1 * x) + 0.4 * Math.cos(2.7 * x);
      const G = S.linspace(0, 10, 301), MAXM = 600;
      let x, y, xt, ytt, F, Ft, Fg, lastH, hist, m, done;
      function newData() {
        const n = 120;
        x = ctx.rng.array(n, () => ctx.rng.uniform(0, 10));
        y = x.map(v => f(v) + ctx.rng.normal(0, 0.35));
        if (out.get()) for (let i = 0; i < n; i++) if (ctx.rng.u() < 0.08) y[i] += (ctx.rng.u() < 0.5 ? -1 : 1) * ctx.rng.uniform(2.5, 5);
        xt = ctx.rng.array(600, () => ctx.rng.uniform(0, 10));
        ytt = xt.map(v => f(v) + ctx.rng.normal(0, 0.35));
        reset();
      }
      const L = (a, b) => (loss.get() === 'l2' ? (a - b) ** 2 : Math.abs(a - b));
      function reset() {
        loop.stop();
        const c = loss.get() === 'l2' ? S.mean(y) : S.median(y);
        F = new Float64Array(x.length).fill(c); Ft = new Float64Array(xt.length).fill(c); Fg = new Float64Array(G.length).fill(c);
        lastH = null; m = 0; done = false;
        hist = [record()];
        draw();
      }
      function record() {
        let tr = 0, te = 0, tru = 0;
        for (let i = 0; i < x.length; i++) tr += L(y[i], F[i]);
        for (let i = 0; i < xt.length; i++) te += L(ytt[i], Ft[i]);
        for (let i = 0; i < G.length; i++) tru += (Fg[i] - f(G[i])) ** 2;
        return { m, tr: tr / x.length, te: te / xt.length, tru: tru / G.length };
      }
      function stepRound() {
        if (m >= MAXM) { done = true; return; }
        const n = x.length, l1 = loss.get() === 'l1', v = nu.get();
        const r = range(n).map(i => (l1 ? Math.sign(y[i] - F[i]) : y[i] - F[i]));
        let idx = range(n);
        if (sub.get() < 1) idx = ctx.rng.shuffle(idx.slice()).slice(0, Math.max(10, Math.round(sub.get() * n)));
        const X = x.map(u => [u]);
        const t = growTree(X, r, { task: 'reg', maxDepth: Math.round(dep.get()), minLeaf: 4, idx }).root;
        if (l1) { // optimal leaf constants for absolute loss: medians of current residuals
          const groups = new Map();
          for (const i of idx) { const lf = leafOf(t, X[i]); if (!groups.has(lf)) groups.set(lf, []); groups.get(lf).push(y[i] - F[i]); }
          for (const [lf, res] of groups) lf.value = S.median(res);
        }
        for (let i = 0; i < n; i++) F[i] += v * leafOf(t, [x[i]]).value;
        for (let i = 0; i < xt.length; i++) Ft[i] += v * leafOf(t, [xt[i]]).value;
        for (let i = 0; i < G.length; i++) Fg[i] += v * leafOf(t, [G[i]]).value;
        lastH = t; m++;
        hist.push(record());
      }
      let acc = 0;
      function tick(dt) {
        acc += dt;
        const per = nu.get() < 0.05 ? 4 : 2;
        let k = 0;
        while (acc > 0.03 && k < per) { stepRound(); acc -= 0.03; k++; }
        draw();
        if (done) return false;
      }
      function draw() {
        pL.clear();
        pL.hline(0, { color: 'faint', dash: false, width: 1 });
        pL.points(x.map((v, i) => [v, y[i]]), { color: 's3', r: 2.6, opacity: 0.65 });
        pL.line(G.map(g => [g, f(g)]), { color: 'ink', dash: true, width: 1.3, opacity: 0.7 });
        if (lastH) pL.line(G.map(g => [g, leafOf(lastH, [g]).value]), { color: 's4', width: 1.6, opacity: 0.85 });
        pL.line(Array.from(Fg, (v, i) => [G[i], v]), { color: 's1', width: 2.4 });
        pL.legend([{ label: 'F_m', color: 's1' }, { label: 'h_m (before ×ν)', color: 's4' }, { label: 'true f', color: 'ink', dash: true }]);
        const mx = Math.max(hist[0].tr, hist[0].te) * 1.05;
        pR.setDomain([0, Math.max(100, m)], [0, mx]).clear();
        pR.line(hist.map(h => [h.m, h.tr]), { color: 's1', width: 2 });
        pR.line(hist.map(h => [h.m, h.te]), { color: 's2', width: 2 });
        const bi = S.argmin(hist.map(h => h.te));
        pR.dot(hist[bi].m, hist[bi].te, { color: 's2', r: 4 });
        pR.legend([{ label: 'training', color: 's1' }, { label: 'test (clean)', color: 's2' }]);
        const h = hist[hist.length - 1];
        sM.set(String(m));
        sTr.set(S.fmt(h.tr, 3), loss.get() === 'l2' ? 'mean squared' : 'mean absolute');
        sTe.set(S.fmt(h.te, 3));
        sBest.set(S.fmt(hist[bi].te, 3), 'at m = ' + hist[bi].m);
        sTru.set(S.fmt(h.tru, 3));
      }
      newData();
    },
  });

  /* =====================================================================
   * Shared: SVM dual solver (SMO with maximal-violating-pair selection)
   * ===================================================================== */
  // min_a 1/2 a'Qa - 1'a  s.t. 0 <= a <= C, y'a = 0, Q_ij = y_i y_j K_ij. Returns alpha, b, iterations.
  function smo(K, y, C, { tol = 1e-3, maxIter = 40000 } = {}) {
    const n = y.length, a = new Float64Array(n), G = new Float64Array(n).fill(-1);
    let it = 0;
    for (; it < maxIter; it++) {
      let Gmax = -Infinity, Gmin = Infinity, i = -1, j = -1;
      for (let t = 0; t < n; t++) {
        const v = -y[t] * G[t];
        if ((y[t] > 0 && a[t] < C) || (y[t] < 0 && a[t] > 0)) { if (v > Gmax) { Gmax = v; i = t; } }
        if ((y[t] > 0 && a[t] > 0) || (y[t] < 0 && a[t] < C)) { if (v < Gmin) { Gmin = v; j = t; } }
      }
      if (i < 0 || j < 0 || Gmax - Gmin < tol) break;
      const Ki = K[i], Kj = K[j], Qij = y[i] * y[j] * Ki[j];
      const ai = a[i], aj = a[j];
      if (y[i] !== y[j]) {
        let q = Ki[i] + Kj[j] + 2 * Qij; if (q <= 0) q = 1e-12;
        const delta = (-G[i] - G[j]) / q, diff = a[i] - a[j];
        a[i] += delta; a[j] += delta;
        if (diff > 0) { if (a[j] < 0) { a[j] = 0; a[i] = diff; } } else if (a[i] < 0) { a[i] = 0; a[j] = -diff; }
        if (diff > 0) { if (a[i] > C) { a[i] = C; a[j] = C - diff; } } else if (a[j] > C) { a[j] = C; a[i] = C + diff; }
      } else {
        let q = Ki[i] + Kj[j] - 2 * Qij; if (q <= 0) q = 1e-12;
        const delta = (G[i] - G[j]) / q, sum = a[i] + a[j];
        a[i] -= delta; a[j] += delta;
        if (sum > C) { if (a[i] > C) { a[i] = C; a[j] = sum - C; } } else if (a[j] < 0) { a[j] = 0; a[i] = sum; }
        if (sum > C) { if (a[j] > C) { a[j] = C; a[i] = sum - C; } } else if (a[i] < 0) { a[i] = 0; a[j] = sum; }
      }
      const di = a[i] - ai, dj = a[j] - aj;
      for (let t = 0; t < n; t++) G[t] += y[t] * (y[i] * Ki[t] * di + y[j] * Kj[t] * dj);
    }
    // offset from free SVs (or the midpoint of the feasible interval)
    let ub = Infinity, lb = -Infinity, sFree = 0, nFree = 0;
    for (let t = 0; t < n; t++) {
      const yG = y[t] * G[t];
      if (a[t] >= C) { if (y[t] < 0) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else if (a[t] <= 0) { if (y[t] > 0) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else { nFree++; sFree += yG; }
    }
    const rho = nFree ? sFree / nFree : (ub + lb) / 2;
    let w2 = 0; for (let t = 0; t < n; t++) w2 += a[t] * (G[t] + 1); // a'Qa
    return { alpha: a, b: -rho, iters: it, w2, dual: S.sum(Array.from(a)) - 0.5 * w2, nFree, converged: it < maxIter };
  }
  const SVM_KERNELS = {
    linear: { label: 'Linear', k: (u, v) => u[0] * v[0] + u[1] * v[1] },
    poly: { label: 'Poly ($d=3$)', k: (u, v) => (1 + (u[0] * v[0] + u[1] * v[1]) / 4) ** 3 },
    rbf: { label: 'RBF', k: (u, v, g) => Math.exp(-g * ((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)) },
  };
  function svmFit(X, y, C, kern, gamma) {
    const kf = SVM_KERNELS[kern].k, n = X.length;
    const K = range(n).map(i => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) K[i][j] = K[j][i] = kf(X[i], X[j], gamma);
    const res = smo(K, y, C);
    const sv = range(n).filter(i => res.alpha[i] > 1e-8);
    res.sv = sv;
    res.f = x => { let s = res.b; for (const i of sv) s += res.alpha[i] * y[i] * kf(X[i], x, gamma); return s; };
    return res;
  }
  function drawSVM(p, X, y, fit, C) {
    const sh = shader();
    p.raster((a, b) => sh(S.logistic(1.6 * fit.f([a, b]))), { nx: 110, ny: 72 });
    p.contour((a, b) => fit.f([a, b]), [-1, 1], { color: 'ink', nx: 70, ny: 46, dash: '4 4', width: 1.2, opacity: 0.7 });
    p.contour((a, b) => fit.f([a, b]), [0], { color: 'ink', nx: 70, ny: 46, width: 2.2 });
    drawPoints(p, X, y.map(v => (v > 0 ? 1 : 0)));
    for (const i of fit.sv) p.dot(X[i][0], X[i][1], { r: 7, opacity: 0, stroke: fit.alpha[i] >= C * (1 - 1e-9) ? 'bad' : 'ink', strokeWidth: 1.6 });
  }

  /* =====================================================================
   * Lab: svm-margin (click to add points)
   * ===================================================================== */
  Atlas.lab('svm-margin', {
    title: 'Soft-margin SVM: click to place points',
    blurb: 'Click to add a point of the chosen class (click an existing point to delete it). The solid curve is $f(x)=0$, dashed curves are $f(x)=\\pm1$. Black rings are free support vectors ($0\\lt\\alpha_i\\lt C$, on the margin), red rings are bounded ones ($\\alpha_i=C$, inside the margin or misclassified). Move a point that is not ringed: nothing changes.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const cls = UI.segmented(controls, { label: 'Click adds class', options: [{ value: 1, label: '+1 (orange)' }, { value: -1, label: '−1 (blue)' }], value: 1 });
      const kern = UI.segmented(controls, { label: 'Kernel', options: Object.keys(SVM_KERNELS).map(k => ({ value: k, label: SVM_KERNELS[k].label })), value: 'linear', onChange: fit });
      const Cs = UI.slider(controls, { label: 'Cost $C$', min: 0.01, max: 1000, value: 1, log: true, fmt: v => S.fmt(v, 2), onChange: fit });
      const gS = UI.slider(controls, { label: 'RBF width $\\gamma$', min: 0.05, max: 20, value: 1, log: true, fmt: v => S.fmt(v, 2), onChange: fit });
      const preset = UI.select(controls, { label: 'Preset', options: [{ value: 'sep', label: 'Separable blobs' }, { value: 'ovl', label: 'Overlapping blobs' }, { value: 'circle', label: 'Disc (needs a kernel)' }, { value: 'xor', label: 'XOR (needs a kernel)' }], value: 'sep', onChange: load });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Reload preset', load, { primary: true });
      UI.button(row, 'Clear', () => { X = []; y = []; fit(); });
      const p = new Plot(stage, { width: 640, height: 340, x: [-4.5, 4.5], y: [-3, 3], xlabel: 'x₁', ylabel: 'x₂' });
      const sSV = UI.stat(stats, 'Support vectors');
      const sM = UI.stat(stats, 'Margin $2/\\|w\\|$', { accent: 'accent' });
      const sTr = UI.stat(stats, 'Training error');
      const sD = UI.stat(stats, 'Dual objective $\\sum\\alpha_i-\\frac12\\|w\\|^2$');
      const note = UI.note(stats, '');
      let X = [], y = [];
      function load() {
        const r = ctx.rng, kind = preset.get();
        X = []; y = [];
        if (kind === 'sep' || kind === 'ovl') {
          const s = kind === 'sep' ? 0.55 : 1.0;
          for (let i = 0; i < 40; i++) { const c = i % 2 ? 1 : -1; X.push([r.normal(1.3 * c, s), r.normal(0.8 * c, s)]); y.push(c); }
        } else {
          const d = make2D(kind, 90, 0, r);
          X = d.X; y = d.y.map(v => (v ? 1 : -1));
        }
        fit();
      }
      let res = null;
      function fit() {
        p.clear();
        const has = y.includes(1) && y.includes(-1);
        res = has ? svmFit(X, y, Cs.get(), kern.get(), gS.get()) : null;
        if (res) drawSVM(p, X, y, res, Cs.get()); else drawPoints(p, X, y.map(v => (v > 0 ? 1 : 0)));
        p.legend([{ label: 'free SV', color: 'ink' }, { label: 'bounded SV (α = C)', color: 'bad' }]);
        if (!res) { sSV.set('—'); sM.set('—'); sTr.set('—'); sD.set('—'); note.set('Add at least one point of each class.'); return; }
        const nb = res.sv.filter(i => res.alpha[i] >= Cs.get() * (1 - 1e-9)).length;
        sSV.set(`${res.sv.length} / ${X.length}`, `${res.sv.length - nb} free, ${nb} at bound`);
        sM.set(S.fmt(2 / Math.sqrt(Math.max(res.w2, 1e-300)), 3), kern.get() === 'linear' ? 'in input space' : 'in feature space');
        let e = 0; for (let i = 0; i < X.length; i++) if (y[i] * res.f(X[i]) <= 0) e++;
        sTr.set(S.fmt(e / X.length, 3));
        sD.set(S.fmt(res.dual, 3), `${res.iters} SMO steps${res.converged ? '' : ' (not converged)'}`);
        note.set(kern.get() === 'linear' ? `$w = (${S.fmt(range(X.length).reduce((s, i) => s + res.alpha[i] * y[i] * X[i][0], 0), 2)},\\ ${S.fmt(range(X.length).reduce((s, i) => s + res.alpha[i] * y[i] * X[i][1], 0), 2)})$, $b = ${S.fmt(res.b, 2)}$` : `$f(x)=\\sum_{i\\in SV}\\alpha_i y_i K(x_i,x)+b$ with ${res.sv.length}$ terms`);
      }
      p.onPointer((a, b, ev, type) => {
        if (type !== 'down' || !p.inside(a, b)) return;
        let k = -1, dmin = 0.18 ** 2;
        X.forEach((q, i) => { const d = (q[0] - a) ** 2 + (q[1] - b) ** 2; if (d < dmin) { dmin = d; k = i; } });
        if (k >= 0) { X.splice(k, 1); y.splice(k, 1); }
        else if (X.length < 150) { X.push([a, b]); y.push(+cls.get()); }
        fit();
      });
      load();
    },
  });

  /* =====================================================================
   * Lab: svm-regularization-path
   * ===================================================================== */
  Atlas.lab('svm-regularization-path', {
    title: 'The cost parameter $C$: from wide margins to overfitting',
    blurb: 'Right: as $C$ grows (weaker regularisation, $\\lambda=1/C$), training error and the fraction of support vectors fall, while test error is U-shaped. Leave-one-out error is bounded by the fraction of support vectors, so a model with few SVs cannot be badly overfit. Try RBF with a large $\\gamma$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: DATA_OPTS, value: 'moons', onChange: newData });
      const flip = UI.slider(controls, { label: 'Label noise', min: 0, max: 0.25, step: 0.01, value: 0.1, onChange: newData });
      const kern = UI.segmented(controls, { label: 'Kernel', options: Object.keys(SVM_KERNELS).map(k => ({ value: k, label: SVM_KERNELS[k].label })), value: 'rbf', onChange: path });
      const gS = UI.slider(controls, { label: 'RBF width $\\gamma$', min: 0.05, max: 20, value: 1, log: true, fmt: v => S.fmt(v, 2), onChange: path });
      const Cs = UI.slider(controls, { label: 'Cost $C$', min: 0.01, max: 1000, value: 1, log: true, fmt: v => S.fmt(v, 2), onChange: show });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [-3, 3], xlabel: 'x₁', ylabel: 'x₂', title: 'Fit at the chosen C' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0.01, 1000], y: [0, 1], logx: true, xlabel: 'C (log scale)', ylabel: 'rate', title: 'Path over C' });
      const sTr = UI.stat(stats, 'Training error');
      const sTe = UI.stat(stats, 'Test error', { accent: 'accent' });
      const sSV = UI.stat(stats, 'Fraction of SVs');
      const sBest = UI.stat(stats, 'Best $C$ on test data');
      const CG = S.linspace(-2, 3, 21).map(v => 10 ** v);
      let D, T, curve;
      function newData() {
        const d = make2D(ds.get(), 100, flip.get(), ctx.rng), t = make2D(ds.get(), 800, flip.get(), ctx.rng);
        D = { X: d.X, y: d.y.map(v => (v ? 1 : -1)) }; T = { X: t.X, y: t.y.map(v => (v ? 1 : -1)) };
        path();
      }
      const evalFit = (C) => {
        const r = svmFit(D.X, D.y, C, kern.get(), gS.get());
        let e = 0, et = 0;
        for (let i = 0; i < D.X.length; i++) if (D.y[i] * r.f(D.X[i]) <= 0) e++;
        for (let i = 0; i < T.X.length; i++) if (T.y[i] * r.f(T.X[i]) <= 0) et++;
        return { r, tr: e / D.X.length, te: et / T.X.length, sv: r.sv.length / D.X.length };
      };
      function path() { curve = CG.map(C => { const q = evalFit(C); return { C, tr: q.tr, te: q.te, sv: q.sv }; }); show(); }
      function show() {
        const C = Cs.get(), q = evalFit(C);
        pL.clear();
        drawSVM(pL, D.X, D.y, q.r, C);
        pR.clear();
        pR.line(curve.map(c => [c.C, c.sv]), { color: 's3', width: 2 });
        pR.line(curve.map(c => [c.C, c.tr]), { color: 's1', width: 2 });
        pR.line(curve.map(c => [c.C, c.te]), { color: 's2', width: 2 });
        pR.vline(C, { color: 'ink', dash: false, width: 1.3 });
        pR.legend([{ label: 'fraction SVs (≥ LOO bound)', color: 's3' }, { label: 'training error', color: 's1' }, { label: 'test error', color: 's2' }]);
        sTr.set(S.fmt(q.tr, 3));
        sTe.set(S.fmt(q.te, 3));
        sSV.set(S.fmt(q.sv, 3), `${q.r.sv.length} of ${D.X.length}`);
        const b = curve[S.argmin(curve.map(c => c.te))];
        sBest.set(S.fmt(b.C, 2), 'test error ' + S.fmt(b.te, 3));
      }
      newData();
    },
  });

  /* =====================================================================
   * Shared: 1-D kernels
   * ===================================================================== */
  const K1 = {
    rbf: { label: 'Gaussian (RBF)', k: (a, b, l) => Math.exp(-0.5 * (a - b) ** 2 / (l * l)) },
    lap: { label: 'Laplacian', k: (a, b, l) => Math.exp(-Math.abs(a - b) / l) },
    poly: { label: 'Polynomial', k: (a, b, l, d) => (1 + a * b / 9) ** d },
  };

  /* =====================================================================
   * Lab: kernel-methods-krr
   * ===================================================================== */
  Atlas.lab('kernel-methods-krr', {
    title: 'Kernel ridge regression and the representer theorem',
    blurb: 'The fit is $\\hat f(x)=\\sum_i \\alpha_i K(x,x_i)$ with $\\alpha=(K+\\lambda I)^{-1}y$: one kernel section per data point (toggle them on). Right: the coefficients $\\alpha_i$. Click the left panel to add data. Watch $\\mathrm{df}_\\lambda=\\operatorname{tr}K(K+\\lambda I)^{-1}$ fall from $n$ to $0$ as $\\lambda$ grows.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const kern = UI.select(controls, { label: 'Kernel', options: Object.keys(K1).map(k => ({ value: k, label: K1[k].label })), value: 'rbf', onChange: fit });
      const lS = UI.slider(controls, { label: 'Length-scale $\\ell$ (RBF, Laplacian)', min: 0.05, max: 5, value: 0.6, log: true, fmt: v => S.fmt(v, 2), onInput: fit });
      const dS = UI.slider(controls, { label: 'Degree $d$ (polynomial)', min: 1, max: 15, step: 1, value: 3, onInput: fit });
      const lam = UI.slider(controls, { label: 'Ridge penalty $\\lambda$', min: 1e-5, max: 100, value: 0.1, log: true, fmt: v => S.fmt(v, 2), onInput: fit });
      const secs = UI.toggle(controls, { label: 'Show $\\alpha_i K(\\cdot,x_i)$ sections', value: false, onChange: fit });
      const nS = UI.slider(controls, { label: 'Initial sample size $n$', min: 5, max: 80, step: 1, value: 25, onChange: newData });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [-2.5, 2.5], xlabel: 'x', ylabel: 'y', title: 'Fit (click to add points)' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [-1, 1], xlabel: 'x_i', ylabel: 'α_i', title: 'Representer coefficients' });
      const sDf = UI.stat(stats, '$\\mathrm{df}_\\lambda=\\operatorname{tr}S_\\lambda$', { accent: 'accent' });
      const sTr = UI.stat(stats, 'Training MSE');
      const sTru = UI.stat(stats, 'MSE vs. true $f$');
      const sN = UI.stat(stats, '$\\|\\hat f\\|_{\\mathcal H}^2=\\alpha^\\top K\\alpha$');
      const f = x => Math.sin(1.5 * x) + 0.3 * x;
      const G = S.linspace(-3, 3, 241);
      let x = [], y = [];
      function newData() {
        const n = Math.round(nS.get());
        x = ctx.rng.array(n, () => ctx.rng.uniform(-3, 3));
        y = x.map(v => f(v) + ctx.rng.normal(0, 0.3));
        fit();
      }
      function fit() {
        const n = x.length, kd = K1[kern.get()], l = lS.get(), d = Math.round(dS.get()), lm = lam.get();
        const kf = (a, b) => kd.k(a, b, l, d);
        pL.clear(); pR.clear();
        pL.line(G.map(g => [g, f(g)]), { color: 'ink', dash: true, width: 1.2, opacity: 0.6 });
        pL.points(x.map((v, i) => [v, y[i]]), { color: 's3', r: 3, opacity: 0.85 });
        if (!n) { sDf.set('—'); sTr.set('—'); sTru.set('—'); sN.set('—'); return; }
        const K = range(n).map(i => range(n).map(j => kf(x[i], x[j])));
        const scale = Math.max(...K.map((r, i) => r[i]));
        const A = K.map((r, i) => r.map((v, j) => v + (i === j ? lm : 0)));
        const ch = cholRobust(A, scale);
        if (!ch) { sDf.set('—'); return; }
        const alpha = cholSolve(ch.L, y);
        // tr (K + lam I)^{-1} = ||L^{-1}||_F^2
        let trInv = 0;
        for (let j = 0; j < n; j++) { const e = new Float64Array(n); e[j] = 1; const c = fwd(ch.L, e); for (let i = 0; i < n; i++) trInv += c[i] * c[i]; }
        const df = n - (lm + ch.jitter) * trInv;
        const fh = g => { let s = 0; for (let i = 0; i < n; i++) s += alpha[i] * kf(g, x[i]); return s; };
        if (secs.get()) for (let i = 0; i < n; i++) pL.line(G.map(g => [g, alpha[i] * kf(g, x[i])]), { color: 's5', width: 1, opacity: 0.45 });
        pL.line(G.map(g => [g, fh(g)]), { color: 's1', width: 2.4 });
        pL.legend([{ label: 'f̂', color: 's1' }, { label: 'true f', color: 'ink', dash: true }].concat(secs.get() ? [{ label: 'α_i K(·, x_i)', color: 's5' }] : []));
        const am = Math.max(...Array.from(alpha, Math.abs), 1e-6) * 1.15;
        pR.setDomain([-3, 3], [-am, am]).clear();
        pR.hline(0, { color: 'muted', dash: false, width: 1 });
        pR.stems(x.map((v, i) => [v, alpha[i]]), { color: 's1', r: 2.5, width: 1.5 });
        let tr = 0; for (let i = 0; i < n; i++) tr += (y[i] - fh(x[i])) ** 2;
        let tru = 0; for (const g of G) tru += (fh(g) - f(g)) ** 2;
        let nrm = 0; for (let i = 0; i < n; i++) nrm += alpha[i] * (y[i] - lm * alpha[i]); // alpha' K alpha = alpha'(y - lam alpha)
        sDf.set(S.fmt(df, 2), `of n = ${n}`);
        sTr.set(S.fmt(tr / n, 3));
        sTru.set(S.fmt(tru / G.length, 3));
        sN.set(S.fmt(nrm, 3));
      }
      pL.onPointer((a, b, ev, type) => { if (type === 'down' && pL.inside(a, b) && x.length < 120) { x.push(a); y.push(b); fit(); } });
      newData();
    },
  });

  /* =====================================================================
   * Lab: kernel-methods-gram (positive definiteness)
   * ===================================================================== */
  Atlas.lab('kernel-methods-gram', {
    title: 'Is it a kernel? Eigenvalues of the Gram matrix',
    blurb: 'Gram matrix of a candidate kernel on an evenly spaced grid. A valid (positive-definite) kernel has no negative eigenvalues on any point set. The power exponential $e^{-|d/\\ell|^q}$ is valid exactly for $0\\lt q\\le 2$: push $q$ past 2 and red (negative) eigenvalues appear. The box and sigmoid "kernels" fail too; the triangle passes (it is valid in one dimension). Left: random functions drawn with the negative part discarded.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const kind = UI.select(controls, { label: 'Candidate kernel', options: [
        { value: 'pexp', label: 'Power exponential exp(−|d/ℓ|^q)' },
        { value: 'box', label: 'Box 1{|d| < ℓ}' },
        { value: 'tri', label: 'Triangle max(0, 1 − |d|/ℓ)' },
        { value: 'sig', label: 'Sigmoid tanh(x x′/ℓ − 1)' },
        { value: 'per', label: 'Periodic (period 2)' },
      ], value: 'pexp', onChange: compute });
      const lS = UI.slider(controls, { label: 'Scale $\\ell$', min: 0.2, max: 3, step: 0.05, value: 1, onInput: compute });
      const qS = UI.slider(controls, { label: 'Exponent $q$ (power exponential)', min: 0.5, max: 3, step: 0.05, value: 2, onInput: compute });
      const nS = UI.slider(controls, { label: 'Grid size $n$', min: 10, max: 60, step: 1, value: 40, onChange: compute });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New draws', () => { Z = null; draw(); }, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [-3, 3], xlabel: 'x', ylabel: 'f(x)', title: 'Draws from N(0, K₊)' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 40], y: [-16, 0.5], xlabel: 'index k', ylabel: 'log₁₀ |λ_k| / λ_max', title: 'Gram eigenvalues (red = negative)' });
      const sMin = UI.stat(stats, '$\\lambda_{\\min}/\\lambda_{\\max}$', { accent: 'accent' });
      const sNeg = UI.stat(stats, 'Clearly negative eigenvalues');
      const sV = UI.stat(stats, 'Verdict on this grid');
      let xs, eig, Z = null;
      const kf = (a, b) => {
        const l = lS.get(), d = Math.abs(a - b);
        switch (kind.get()) {
          case 'pexp': return Math.exp(-Math.pow(d / l, qS.get()));
          case 'box': return d < l ? 1 : 0;
          case 'tri': return Math.max(0, 1 - d / l);
          case 'sig': return Math.tanh(a * b / l - 1);
          default: return Math.exp(-2 * Math.sin(Math.PI * d / 2) ** 2 / (l * l));
        }
      };
      function compute() {
        const n = Math.round(nS.get());
        xs = S.linspace(-3, 3, n);
        const K = xs.map(a => xs.map(b => kf(a, b)));
        eig = S.eigSym(K);
        if (!Z || Z.length !== n) Z = null;
        draw();
      }
      function draw() {
        const n = xs.length, { values, vectors } = eig;
        if (!Z) Z = range(3).map(() => ctx.rng.array(n, () => ctx.rng.normal()));
        const lmax = Math.max(...values.map(Math.abs));
        pL.clear();
        Z.forEach((z, s) => {
          const fv = xs.map((_, i) => { let v = 0; for (let k = 0; k < n; k++) if (values[k] > 0) v += vectors[i][k] * Math.sqrt(values[k]) * z[k]; return v; });
          pL.line(xs.map((xv, i) => [xv, fv[i]]), { color: ['s1', 's2', 's3'][s], width: 1.8 });
        });
        pR.setDomain([0, n + 1], [-16, 0.5]).clear();
        const tol = 1e-10 * lmax;
        let neg = 0;
        const sorted = values.slice().sort((a, b) => b - a);
        sorted.forEach((v, k) => {
          const yv = Math.max(-15.5, Math.log10(Math.max(Math.abs(v), 1e-300) / lmax));
          const isNeg = v < -tol;
          if (isNeg) neg++;
          pR.segment(k + 1, -16, k + 1, yv, { color: isNeg ? 'bad' : Math.abs(v) <= tol ? 'muted' : 's1', width: 3, opacity: 0.8 });
        });
        pR.hline(-10, { color: 'muted', width: 1 });
        pR.text(n + 0.5, -10, 'numerical zero', { anchor: 'end', dy: -4, size: 11, color: 'muted' });
        const lmin = sorted[sorted.length - 1];
        sMin.set(S.fmt(lmin / lmax, 3));
        sNeg.set(String(neg), `of ${n}`);
        sV.set(neg ? '<span style="color:var(--bad)">not PSD</span>' : '<span style="color:var(--good)">PSD</span>', neg ? 'no feature map can produce this matrix' : 'consistent with a valid kernel');
      }
      compute();
    },
  });

  /* =====================================================================
   * Shared: Gaussian-process regression in 1-D
   * ===================================================================== */
  const GPK = {
    se: { label: 'Squared exponential', k: (r, l) => Math.exp(-0.5 * r * r / (l * l)) },
    m32: { label: 'Matérn 3/2', k: (r, l) => { const a = Math.sqrt(3) * Math.abs(r) / l; return (1 + a) * Math.exp(-a); } },
    m52: { label: 'Matérn 5/2', k: (r, l) => { const a = Math.sqrt(5) * Math.abs(r) / l; return (1 + a + a * a / 3) * Math.exp(-a); } },
    per: { label: 'Periodic (period 3)', k: (r, l) => Math.exp(-2 * Math.sin(Math.PI * Math.abs(r) / 3) ** 2 / (l * l)) },
  };
  function gpFit(x, y, kern, l, sf, sn) {
    const n = x.length, kf = GPK[kern].k;
    const K = range(n).map(i => range(n).map(j => sf * sf * kf(x[i] - x[j], l) + (i === j ? sn * sn : 0)));
    const ch = cholRobust(K, sf * sf);
    if (!ch) return null;
    const L = ch.L, alpha = cholSolve(L, y);
    let logdet = 0; for (let i = 0; i < n; i++) logdet += Math.log(L[i][i]);
    const fitTerm = -0.5 * dotv(y, alpha), cplx = -logdet;
    const kstar = g => x.map(xi => sf * sf * kf(g - xi, l));
    return {
      L, alpha, fitTerm, cplx, logml: fitTerm + cplx - 0.5 * n * Math.log(2 * Math.PI),
      mean: g => dotv(kstar(g), alpha),
      vari: g => { const v = fwd(L, kstar(g)); return Math.max(0, sf * sf - dotv(v, v)); },
      kstar,
    };
  }

  /* =====================================================================
   * Lab: gaussian-processes-regression
   * ===================================================================== */
  Atlas.lab('gaussian-processes-regression', {
    title: 'GP regression: click to condition on data',
    blurb: 'Click to add an observation (click near one to remove it). The band is the posterior mean $\\pm 2$ SD of $f(x)$; thin curves are posterior draws. The posterior SD does not depend on $y$, only on where the data are. <em>Maximise evidence</em> fits $(\\ell,\\sigma_f,\\sigma_n)$ by the log marginal likelihood. The KRR overlay with $\\lambda=\\sigma_n^2$ lands exactly on the posterior mean.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const kern = UI.select(controls, { label: 'Covariance function', options: Object.keys(GPK).map(k => ({ value: k, label: GPK[k].label })), value: 'se', onChange: draw });
      const lS = UI.slider(controls, { label: 'Length-scale $\\ell$', min: 0.1, max: 5, value: 1, log: true, fmt: v => S.fmt(v, 2), onInput: draw });
      const fS = UI.slider(controls, { label: 'Signal SD $\\sigma_f$', min: 0.1, max: 5, value: 1, log: true, fmt: v => S.fmt(v, 2), onInput: draw });
      const nS = UI.slider(controls, { label: 'Noise SD $\\sigma_n$', min: 0.01, max: 2, value: 0.1, log: true, fmt: v => S.fmt(v, 3), onInput: draw });
      const samp = UI.toggle(controls, { label: 'Show posterior draws', value: true, onChange: draw });
      const krr = UI.toggle(controls, { label: 'Overlay KRR with $\\lambda=\\sigma_n^2$', value: false, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Maximise evidence', optimise, { primary: true });
      UI.button(row, 'New data', newData);
      UI.button(row, 'Clear', () => { x = []; y = []; draw(); });
      const p = new Plot(stage, { width: 640, height: 320, x: [-5, 5], y: [-3, 3], xlabel: 'x', ylabel: 'f(x)' });
      const sML = UI.stat(stats, '$\\log p(y\\mid X,\\theta)$', { accent: 'accent' });
      const sFit = UI.stat(stats, 'Data fit $-\\frac12 y^\\top K_y^{-1}y$');
      const sCx = UI.stat(stats, 'Complexity $-\\frac12\\log|K_y|$');
      const sN = UI.stat(stats, 'Observations $n$');
      const truth = v => Math.sin(1.3 * v) + 0.4 * Math.cos(3 * v);
      const G = S.linspace(-5, 5, 121);
      let x = [], y = [], Z = null;
      function newData() {
        x = ctx.rng.array(8, () => ctx.rng.uniform(-4.5, 4.5));
        y = x.map(v => truth(v) + ctx.rng.normal(0, 0.1));
        Z = null;
        draw();
      }
      function draw() {
        const l = lS.get(), sf = fS.get(), sn = nS.get(), kf = GPK[kern.get()].k;
        const fit = x.length ? gpFit(x, y, kern.get(), l, sf, sn) : null;
        const mean = g => (fit ? fit.mean(g) : 0), vari = g => (fit ? fit.vari(g) : sf * sf);
        const m = G.map(mean), sd = G.map(g => Math.sqrt(vari(g)));
        p.clear();
        p.area(G.map((g, i) => [g, m[i] + 2 * sd[i]]), { color: 's1', opacity: 0.16, base: G.map((g, i) => [g, m[i] - 2 * sd[i]]) });
        if (samp.get()) {
          // posterior covariance on the grid: K** - V'V with V = L^{-1} K(X, grid)
          const ng = G.length;
          const V = fit ? G.map(g => fwd(fit.L, fit.kstar(g))) : null;
          const C = range(ng).map(i => range(ng).map(j => sf * sf * kf(G[i] - G[j], l) - (V ? dotv(V[i], V[j]) : 0)));
          const ch = cholRobust(C, sf * sf);
          if (ch) {
            if (!Z) Z = range(3).map(() => ctx.rng.array(ng, () => ctx.rng.normal()));
            Z.forEach((z, s) => {
              const Lz = range(ng).map(i => { let v = 0; for (let k = 0; k <= i; k++) v += ch.L[i][k] * z[k]; return v; });
              p.line(G.map((g, i) => [g, m[i] + Lz[i]]), { color: ['s3', 's4', 's5'][s], width: 1.1, opacity: 0.75 });
            });
          }
        }
        p.line(G.map((g, i) => [g, m[i]]), { color: 's1', width: 2.4 });
        if (krr.get() && x.length) {
          const n = x.length, K = range(n).map(i => range(n).map(j => sf * sf * kf(x[i] - x[j], l) + (i === j ? sn * sn : 0)));
          let a = null;
          try { a = S.solve(K, y); } catch (e) { a = null; }
          if (a) p.line(G.map(g => [g, x.reduce((s, xi, i) => s + a[i] * sf * sf * kf(g - xi, l), 0)]), { color: 's2', width: 2, dash: '2 5' });
        }
        p.points(x.map((v, i) => [v, y[i]]), { color: 'ink', r: 3.5, opacity: 0.9 });
        p.legend([{ label: 'posterior mean', color: 's1' }, { label: '± 2 SD', color: 's1', fill: true, opacity: 0.2 }].concat(krr.get() ? [{ label: 'KRR, λ = σₙ²', color: 's2', dash: true }] : []));
        sN.set(String(x.length));
        if (fit) {
          sML.set(S.fmt(fit.logml, 2)); sFit.set(S.fmt(fit.fitTerm, 2)); sCx.set(S.fmt(fit.cplx, 2), 'plus −(n/2) log 2π');
        } else { sML.set('—', 'prior only'); sFit.set('—'); sCx.set('—'); }
      }
      function optimise() {
        if (x.length < 2) return;
        const lo = [Math.log(0.1), Math.log(0.1), Math.log(0.01)], hi = [Math.log(5), Math.log(5), Math.log(2)];
        const obj = t => {
          const q = t.map((v, i) => clamp(v, lo[i], hi[i]));
          const pen = t.reduce((s, v, i) => s + (v - q[i]) ** 2, 0) * 100;
          const fit = gpFit(x, y, kern.get(), Math.exp(q[0]), Math.exp(q[1]), Math.exp(q[2]));
          return (fit ? -fit.logml : 1e9) + pen;
        };
        let best = null;
        for (const s0 of [[0, 0, -2], [Math.log(0.3), 0, -3], [Math.log(2), 0, -1]]) {
          const r = S.nelderMead(obj, s0, { step: 0.5, maxIter: 600 });
          if (!best || r.fx < best.fx) best = r;
        }
        const q = best.x.map((v, i) => clamp(v, lo[i], hi[i]));
        lS.set(Math.exp(q[0])); fS.set(Math.exp(q[1])); nS.set(Math.exp(q[2]));
        draw();
      }
      p.onPointer((a, b, ev, type) => {
        if (type !== 'down' || !p.inside(a, b)) return;
        const k = x.findIndex((v, i) => Math.abs(v - a) < 0.12 && Math.abs(y[i] - b) < 0.2);
        if (k >= 0) { x.splice(k, 1); y.splice(k, 1); } else if (x.length < 60) { x.push(a); y.push(b); }
        draw();
      });
      newData();
    },
  });

  /* =====================================================================
   * Lab: gaussian-processes-evidence (marginal-likelihood landscape)
   * ===================================================================== */
  Atlas.lab('gaussian-processes-evidence', {
    title: 'The marginal-likelihood landscape',
    blurb: 'Left: $\\log p(y\\mid X,\\ell,\\sigma_n)$ over a grid (SE kernel, $\\sigma_f$ fixed by the slider); contours are 1, 2, 5, 10 and 20 nats below the best. The data mix a slow and a fast wave, so there are often two local maxima: "smooth curve + lots of noise" and "wiggly curve + little noise". Click the map to see the fit at that point.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nS = UI.slider(controls, { label: 'Observations $n$', min: 8, max: 40, step: 1, value: 20, onChange: newData });
      const noise = UI.slider(controls, { label: 'True noise SD', min: 0.05, max: 0.6, step: 0.01, value: 0.15, onChange: newData });
      const fS = UI.slider(controls, { label: 'Signal SD $\\sigma_f$', min: 0.2, max: 3, value: 1, log: true, fmt: v => S.fmt(v, 2), onChange: landscape });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', newData, { primary: true });
      UI.button(row, 'Go to best', () => { cur = bestPt.slice(); draw(); });
      stage.classList.add('two');
      const LX = [-1, 1.2], LY = [-2.2, 0.4];
      const pM = new Plot(stage, { width: 420, height: 270, x: LX, y: LY, xlabel: 'log₁₀ ℓ', ylabel: 'log₁₀ σₙ', title: 'log marginal likelihood' });
      const pF = new Plot(stage, { width: 420, height: 270, x: [-5, 5], y: [-3, 3], xlabel: 'x', ylabel: 'y', title: 'Posterior at the selected point' });
      const sSel = UI.stat(stats, 'Selected $(\\ell,\\sigma_n)$');
      const sML = UI.stat(stats, 'log evidence there', { accent: 'accent' });
      const sBest = UI.stat(stats, 'Global grid maximum');
      const sLoc = UI.stat(stats, 'Local maxima on grid');
      const truth = v => Math.sin(0.9 * v) + 0.45 * Math.sin(4.2 * v);
      const NX = 44, NY = 36;
      let x, y, grid, gmax, bestPt, cur, locals;
      function newData() {
        const n = Math.round(nS.get());
        x = ctx.rng.array(n, () => ctx.rng.uniform(-5, 5));
        y = x.map(v => truth(v) + ctx.rng.normal(0, noise.get()));
        landscape();
      }
      const gx = i => LX[0] + (LX[1] - LX[0]) * i / (NX - 1), gy = j => LY[0] + (LY[1] - LY[0]) * j / (NY - 1);
      function landscape() {
        const sf = fS.get();
        grid = range(NY).map(j => range(NX).map(i => { const r = gpFit(x, y, 'se', 10 ** gx(i), sf, 10 ** gy(j)); return r ? r.logml : -1e9; }));
        gmax = -Infinity; locals = [];
        for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
          const v = grid[j][i];
          if (v > gmax) { gmax = v; bestPt = [gx(i), gy(j)]; }
          let isMax = true;
          for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
            if (!dj && !di) continue;
            const jj = j + dj, ii = i + di;
            if (jj >= 0 && jj < NY && ii >= 0 && ii < NX && grid[jj][ii] >= v) isMax = false;
          }
          if (isMax && i > 0 && j > 0 && i < NX - 1 && j < NY - 1) locals.push([gx(i), gy(j), v]);
        }
        cur = bestPt.slice();
        draw();
      }
      const interp = (a, b) => {
        const fi = clamp((a - LX[0]) / (LX[1] - LX[0]) * (NX - 1), 0, NX - 1.0001), fj = clamp((b - LY[0]) / (LY[1] - LY[0]) * (NY - 1), 0, NY - 1.0001);
        const i = Math.floor(fi), j = Math.floor(fj), u = fi - i, v = fj - j;
        return (1 - u) * (1 - v) * grid[j][i] + u * (1 - v) * grid[j][i + 1] + (1 - u) * v * grid[j + 1][i] + u * v * grid[j + 1][i + 1];
      };
      function draw() {
        const c0 = Atlas.rgb('surface'), c1 = Atlas.rgb('s1');
        pM.clear();
        pM.raster((a, b) => { const t = clamp((interp(a, b) - (gmax - 40)) / 40, 0, 1); const c = Atlas.mix(c0, c1, t ** 1.5); return [c[0], c[1], c[2], 255]; }, { nx: 110, ny: 72 });
        pM.contour(interp, [gmax - 1, gmax - 2, gmax - 5, gmax - 10, gmax - 20], { color: 'ink', nx: 80, ny: 60, width: 1, opacity: 0.6 });
        for (const q of locals) pM.dot(q[0], q[1], { color: 'warn', r: 4.5 });
        pM.dot(cur[0], cur[1], { color: 'ink', r: 5.5, stroke: 'surface', strokeWidth: 2 });
        const l = 10 ** cur[0], sn = 10 ** cur[1], sf = fS.get();
        const fit = gpFit(x, y, 'se', l, sf, sn);
        const G = S.linspace(-5, 5, 161);
        pF.clear();
        if (fit) {
          const m = G.map(fit.mean), sd = G.map(g => Math.sqrt(fit.vari(g) + sn * sn));
          pF.area(G.map((g, i) => [g, m[i] + 2 * sd[i]]), { color: 's1', opacity: 0.15, base: G.map((g, i) => [g, m[i] - 2 * sd[i]]) });
          pF.line(G.map((g, i) => [g, m[i]]), { color: 's1', width: 2.2 });
        }
        pF.line(G.map(g => [g, truth(g)]), { color: 'ink', dash: true, width: 1.2, opacity: 0.6 });
        pF.points(x.map((v, i) => [v, y[i]]), { color: 'ink', r: 3.2 });
        pF.legend([{ label: 'mean', color: 's1' }, { label: '± 2 SD of y*', color: 's1', fill: true, opacity: 0.2 }, { label: 'truth', color: 'ink', dash: true }]);
        sSel.set(`(${S.fmt(l, 2)}, ${S.fmt(sn, 3)})`);
        sML.set(fit ? S.fmt(fit.logml, 2) : '—', fit ? S.fmt(fit.logml - gmax, 2) + ' vs. best' : '');
        sBest.set(S.fmt(gmax, 2), `ℓ = ${S.fmt(10 ** bestPt[0], 2)}, σₙ = ${S.fmt(10 ** bestPt[1], 3)}`);
        sLoc.set(String(locals.length), locals.length > 1 ? 'multimodal evidence' : 'unimodal on this grid');
      }
      pM.onPointer((a, b, ev, type) => { if ((type === 'down' || type === 'drag') && pM.inside(a, b)) { cur = [a, b]; draw(); } });
      newData();
    },
  });
})();
