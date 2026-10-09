/* Labs: Deep Learning I — perceptron & MLP, universal approximation, backpropagation,
 * optimizers, initialization & normalization, regularization, and theory (NTK, implicit bias).
 * Everything (the MLP, its backward pass, the optimizers) is implemented here in plain JS. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;

  /* ================================================================
   * Shared numerics
   * ================================================================ */
  const sig = z => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
  const softplus = z => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)));
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const pos = v => (v > 0 && Number.isFinite(v) ? v : NaN); // for log axes
  const logspace = (a, b, k) => S.linspace(Math.log10(a), Math.log10(b), k).map(v => 10 ** v);

  // Activations: f(z) and derivative in terms of z (so dropout scaling never corrupts it).
  const ACTS = {
    relu: { label: 'ReLU', f: z => (z > 0 ? z : 0), d: z => (z > 0 ? 1 : 0), gain: 2 },
    tanh: { label: 'tanh', f: Math.tanh, d: z => { const t = Math.tanh(z); return 1 - t * t; }, gain: 1 },
    sigmoid: { label: 'sigmoid', f: sig, d: z => { const s = sig(z); return s * (1 - s); }, gain: 1 },
    linear: { label: 'linear', f: z => z, d: () => 1, gain: 1 },
  };

  // Solve SPD system A x = b (A: Float64Array m*m, row-major) by Cholesky; A is not modified.
  function cholSolve(A, b, m) {
    const L = new Float64Array(m * m);
    for (let i = 0; i < m; i++) for (let j = 0; j <= i; j++) {
      let s = A[i * m + j];
      for (let k = 0; k < j; k++) s -= L[i * m + k] * L[j * m + k];
      L[i * m + j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / L[j * m + j];
    }
    const z = new Float64Array(m), x = new Float64Array(m);
    for (let i = 0; i < m; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * m + k] * z[k]; z[i] = s / L[i * m + i]; }
    for (let i = m - 1; i >= 0; i--) { let s = z[i]; for (let k = i + 1; k < m; k++) s -= L[k * m + i] * x[k]; x[i] = s / L[i * m + i]; }
    return x;
  }

  /* ================================================================
   * A small multilayer perceptron with manual backprop
   *   layer l maps sizes[l] -> sizes[l+1]; W[l] is (out x in) row-major.
   *   Hidden layers use `act`; the output layer is linear (a logit or a regression value).
   * ================================================================ */
  function mlpNew(sizes, act, rng, gain) {
    const net = { sizes: sizes.slice(), act, L: sizes.length - 1, W: [], b: [], mW: [], mb: [], vW: [], vb: [], t: 0 };
    const g = gain ?? ACTS[act].gain;
    for (let l = 0; l < net.L; l++) {
      const nin = sizes[l], nout = sizes[l + 1];
      const W = new Float64Array(nout * nin), sd = Math.sqrt((l === net.L - 1 ? 1 : g) / nin);
      for (let i = 0; i < W.length; i++) W[i] = rng.normal() * sd;
      net.W.push(W); net.b.push(new Float64Array(nout));
      net.mW.push(new Float64Array(W.length)); net.vW.push(new Float64Array(W.length));
      net.mb.push(new Float64Array(nout)); net.vb.push(new Float64Array(nout));
    }
    return net;
  }
  const mlpParams = net => net.W.reduce((s, W, l) => s + W.length + net.b[l].length, 0);
  const mlpNorm = net => Math.sqrt(net.W.reduce((s, W) => s + W.reduce((a, v) => a + v * v, 0), 0));

  // Forward pass on a batch X (n x d, row-major). drop = dropout probability on hidden units (inverted dropout).
  function mlpForward(net, X, n, drop, rng) {
    const A = [X], Z = [null], M = [null], act = ACTS[net.act];
    for (let l = 0; l < net.L; l++) {
      const nin = net.sizes[l], nout = net.sizes[l + 1], W = net.W[l], b = net.b[l], Ain = A[l];
      const z = new Float64Array(n * nout);
      for (let i = 0; i < n; i++) {
        const io = i * nin, oo = i * nout;
        for (let k = 0; k < nout; k++) {
          let s = b[k];
          const wo = k * nin;
          for (let j = 0; j < nin; j++) s += W[wo + j] * Ain[io + j];
          z[oo + k] = s;
        }
      }
      Z.push(z);
      if (l === net.L - 1) { A.push(z); M.push(null); break; }
      const a = new Float64Array(n * nout);
      for (let q = 0; q < a.length; q++) a[q] = act.f(z[q]);
      let mask = null;
      if (drop > 0) {
        mask = new Float64Array(a.length);
        const keep = 1 - drop;
        for (let q = 0; q < a.length; q++) { mask[q] = rng.u() < keep ? 1 / keep : 0; a[q] *= mask[q]; }
      }
      M.push(mask); A.push(a);
    }
    return { A, Z, M, n };
  }

  // Backward pass: dZ = dLoss/d(output pre-activation), n x out. Returns parameter gradients.
  function mlpBackward(net, cache, dZ) {
    const gW = new Array(net.L), gb = new Array(net.L), deltas = new Array(net.L), n = cache.n, act = ACTS[net.act];
    let delta = dZ;
    for (let l = net.L - 1; l >= 0; l--) {
      deltas[l] = delta; // dLoss / dZ of layer l's output
      const nin = net.sizes[l], nout = net.sizes[l + 1], Ain = cache.A[l], W = net.W[l];
      const G = new Float64Array(nout * nin), gB = new Float64Array(nout);
      for (let i = 0; i < n; i++) for (let k = 0; k < nout; k++) {
        const d = delta[i * nout + k];
        if (d === 0) continue;
        gB[k] += d;
        const go = k * nin, io = i * nin;
        for (let j = 0; j < nin; j++) G[go + j] += d * Ain[io + j];
      }
      gW[l] = G; gb[l] = gB;
      if (l > 0) {
        const dA = new Float64Array(n * nin);
        for (let i = 0; i < n; i++) for (let k = 0; k < nout; k++) {
          const d = delta[i * nout + k];
          if (d === 0) continue;
          const wo = k * nin, io = i * nin;
          for (let j = 0; j < nin; j++) dA[io + j] += W[wo + j] * d;
        }
        const z = cache.Z[l], mask = cache.M[l];
        for (let q = 0; q < dA.length; q++) dA[q] *= act.d(z[q]) * (mask ? mask[q] : 1);
        delta = dA;
      }
    }
    return { gW, gb, deltas };
  }

  // One optimizer step. o = {lr, opt: 'sgd'|'momentum'|'adam', wd, decoupled}
  function mlpStep(net, g, o) {
    net.t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, lr = o.lr, wd = o.wd || 0;
    const c1 = 1 - b1 ** net.t, c2 = 1 - b2 ** net.t;
    for (let l = 0; l < net.L; l++) {
      for (const [P, G, m, v, isW] of [[net.W[l], g.gW[l], net.mW[l], net.vW[l], true], [net.b[l], g.gb[l], net.mb[l], net.vb[l], false]]) {
        for (let i = 0; i < P.length; i++) {
          let gi = G[i];
          if (isW && wd && !o.decoupled) gi += wd * P[i];
          if (o.opt === 'sgd') P[i] -= lr * gi;
          else if (o.opt === 'momentum') { m[i] = 0.9 * m[i] + gi; P[i] -= lr * m[i]; }
          else {
            m[i] = b1 * m[i] + (1 - b1) * gi;
            v[i] = b2 * v[i] + (1 - b2) * gi * gi;
            P[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
          }
          if (isW && wd && o.decoupled) P[i] -= lr * wd * P[i];
        }
      }
    }
  }

  // Binary cross-entropy with logits. Returns {loss, acc, dZ} (dZ already divided by n).
  function bce(zOut, y, n, wantGrad) {
    let loss = 0, acc = 0;
    const dZ = wantGrad ? new Float64Array(n) : null;
    for (let i = 0; i < n; i++) {
      const z = zOut[i];
      loss += softplus(z) - y[i] * z;
      if ((z > 0 ? 1 : 0) === y[i]) acc++;
      if (dZ) dZ[i] = (sig(z) - y[i]) / n;
    }
    return { loss: loss / n, acc: acc / n, dZ };
  }

  /* ---------- 2D classification datasets ---------- */
  const DATASETS = { linear: 'Linear', xor: 'XOR', circles: 'Circles', moons: 'Moons', spiral: 'Spirals' };
  function makeData(kind, n, noise, rng, flip = 0) {
    const X = new Float64Array(2 * n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const cls = i % 2;
      let a = 0, b = 0, c = cls;
      if (kind === 'linear') {
        a = rng.uniform(-2, 2); b = rng.uniform(-2, 2);
        c = a + 0.6 * b + 0.3 + noise * rng.normal() > 0 ? 1 : 0;
      } else if (kind === 'xor') {
        const sx = rng.u() < 0.5 ? -1 : 1, sy = rng.u() < 0.5 ? -1 : 1, s = 0.35 + noise;
        a = 1.1 * sx + s * rng.normal(); b = 1.1 * sy + s * rng.normal(); c = sx * sy > 0 ? 1 : 0;
      } else if (kind === 'circles') {
        const r = (cls ? 0.75 : 1.8) + (0.12 + 0.6 * noise) * rng.normal(), t = rng.uniform(0, 2 * Math.PI);
        a = r * Math.cos(t); b = r * Math.sin(t);
      } else if (kind === 'moons') {
        const t = rng.uniform(0, Math.PI), s = 0.1 + 0.6 * noise;
        if (cls) { a = 1 - Math.cos(t); b = 0.5 - Math.sin(t); } else { a = Math.cos(t); b = Math.sin(t); }
        a = 1.3 * (a - 0.5) + s * rng.normal(); b = 1.3 * (b - 0.25) + s * rng.normal();
      } else { // spiral
        const t = rng.u(), r = 0.15 + 2.0 * t, th = 3.2 * Math.PI * t + cls * Math.PI, s = 0.05 + 0.4 * noise;
        a = r * Math.cos(th) + s * rng.normal(); b = r * Math.sin(th) + s * rng.normal();
      }
      if (flip && rng.u() < flip) c = 1 - c;
      X[2 * i] = a; X[2 * i + 1] = b; y[i] = c;
    }
    return { X, y, n };
  }
  const ptsOf = (D, cls) => { const out = []; for (let i = 0; i < D.n; i++) if (D.y[i] === cls) out.push([D.X[2 * i], D.X[2 * i + 1]]); return out; };
  function drawData(p, D, r = 3.2) {
    const all = [];
    for (let i = 0; i < D.n; i++) all.push([D.X[2 * i], D.X[2 * i + 1]]);
    p.points(all, { color: 'ink', r: r + 1.2, opacity: 0.85 });
    p.points(ptsOf(D, 0), { color: 's1', r, opacity: 1 });
    p.points(ptsOf(D, 1), { color: 's2', r, opacity: 1 });
  }
  // Decision-region raster for a network (logit output) or any logit function f(x, y).
  function regionRaster(p, net, nx = 64, ny = 44) {
    const [x0, x1] = p.o.x, [y0, y1] = p.o.y, N = nx * ny;
    const G = new Float64Array(2 * N);
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      G[2 * k] = x0 + (i + 0.5) / nx * (x1 - x0);
      G[2 * k + 1] = y1 - (j + 0.5) / ny * (y1 - y0);
    }
    const out = mlpForward(net, G, N, 0).A[net.L];
    const c0 = Atlas.rgb('s1'), c1 = Atlas.rgb('s2'), bg = Atlas.rgb('surface');
    p.raster((x, y) => {
      const i = clamp(Math.floor((x - x0) / (x1 - x0) * nx), 0, nx - 1), j = clamp(Math.floor((y1 - y) / (y1 - y0) * ny), 0, ny - 1);
      const pr = sig(out[j * nx + i]), t = Math.abs(pr - 0.5) * 2;
      return Atlas.mix(bg, pr > 0.5 ? c1 : c0, 0.12 + 0.5 * t);
    }, { nx: 128, ny: 88 });
  }
  // Draw the line w0 x + w1 y + b = 0 across the plot.
  function drawLine(p, w0, w1, b, opts) {
    const [x0, x1] = p.o.x, [y0, y1] = p.o.y;
    if (Math.abs(w1) >= Math.abs(w0)) { if (w1 === 0) return; p.line([[x0, -(b + w0 * x0) / w1], [x1, -(b + w0 * x1) / w1]], opts); }
    else p.line([[-(b + w1 * y0) / w0, y0], [-(b + w1 * y1) / w0, y1]], opts);
  }

  /* ================================================================
   * Playground: an MLP trained by full-batch gradient descent on 2D data.
   *   cfg.reg = true adds dropout / weight-decay controls and a small noisy training set.
   * ================================================================ */
  function playground(el, ctx, cfg) {
    const reg = !!cfg.reg;
    const { controls, stage, stats } = UI.layout(el);
    const ds = UI.select(controls, { label: 'Dataset', options: Object.keys(DATASETS).map(k => ({ value: k, label: DATASETS[k] })), value: cfg.data, onChange: newData });
    const noise = UI.slider(controls, reg
      ? { label: 'Label-flip rate', min: 0, max: 0.3, step: 0.01, value: 0.12, onChange: newData }
      : { label: 'Input noise', min: 0, max: 0.6, step: 0.05, value: 0.1, onChange: newData });
    const depth = reg ? null : UI.segmented(controls, { label: 'Hidden layers', options: ['0', '1', '2', '3'], value: '1', onChange: reinit });
    const width = UI.slider(controls, { label: 'Width per hidden layer', min: 1, max: reg ? 96 : 32, step: 1, value: reg ? 64 : 8, onChange: reinit });
    const act = UI.segmented(controls, { label: 'Activation', options: ['relu', 'tanh', 'sigmoid', 'linear'].map(v => ({ value: v, label: ACTS[v].label })), value: reg ? 'relu' : 'tanh', onChange: reinit });
    const opt = UI.segmented(controls, { label: 'Optimizer', options: [{ value: 'sgd', label: 'GD' }, { value: 'momentum', label: 'Momentum' }, { value: 'adam', label: 'Adam' }], value: reg ? 'adam' : 'adam', onChange: () => {} });
    const lr = UI.slider(controls, { label: 'Learning rate $\\eta$', min: 0.001, max: 1, log: true, value: reg ? 0.01 : 0.03, fmt: v => S.fmt(v, 2) });
    let drop = null, wd = null, wdMode = null;
    if (reg) {
      drop = UI.slider(controls, { label: 'Dropout rate $p$', min: 0, max: 0.7, step: 0.05, value: 0 });
      wd = UI.segmented(controls, { label: 'Weight decay $\\lambda$', options: ['0', '1e-3', '1e-2', '0.1', '0.5'], value: '0' });
      wdMode = UI.segmented(controls, { label: 'Decay applied as', options: [{ value: 'l2', label: 'L2 in loss' }, { value: 'dec', label: 'Decoupled (AdamW)' }], value: 'dec' });
    }
    const row = UI.h('div', 'btn-row', controls);
    const loop = ctx.loop(step);
    ctx.playButton(row, loop, ['▶ Train', '❚❚ Pause']);
    UI.button(row, 'Re-initialize', reinit);
    UI.button(row, 'New data', newData);
    stage.classList.add('two');
    const pR = new Plot(stage, { width: 420, height: 300, x: [-2.6, 2.6], y: [-2.6, 2.6], equal: true, title: 'Decision regions (training data)' });
    const pL = new Plot(stage, { width: 420, height: 300, x: [0, 100], y: [0, 1], xlabel: 'epoch', ylabel: 'cross-entropy', title: reg ? 'Train vs. test loss' : 'Training loss' });
    const sE = UI.stat(stats, 'Epochs');
    const sL = UI.stat(stats, 'Train loss', { accent: 'accent' });
    const sA = UI.stat(stats, 'Train / test accuracy');
    const sP = UI.stat(stats, 'Parameters');
    const sN = reg ? UI.stat(stats, 'Weight norm $\\|W\\|_F$') : null;

    let D, T, net, hist, epoch, acc = 0;
    const NTR = reg ? 40 : 200, NTE = reg ? 600 : 300;
    function newData() {
      const fl = reg ? noise.get() : 0, nz = reg ? 0.1 : noise.get();
      D = makeData(ds.get(), NTR, nz, ctx.rng, fl);
      T = makeData(ds.get(), NTE, nz, ctx.rng, fl);
      reinit();
    }
    function reinit() {
      const h = reg ? 2 : +depth.get(), w = Math.round(width.get());
      const sizes = [2]; for (let i = 0; i < h; i++) sizes.push(w); sizes.push(1);
      net = mlpNew(sizes, act.get(), ctx.rng);
      hist = []; epoch = 0;
      record();
      draw();
    }
    function record() {
      const tr = bce(mlpForward(net, D.X, D.n, 0).A[net.L], D.y, D.n, false);
      const te = bce(mlpForward(net, T.X, T.n, 0).A[net.L], T.y, T.n, false);
      hist.push({ e: epoch, tr: tr.loss, te: te.loss, atr: tr.acc, ate: te.acc });
    }
    function trainEpoch() {
      const p = reg ? drop.get() : 0;
      const cache = mlpForward(net, D.X, D.n, p, ctx.rng);
      const { dZ } = bce(cache.A[net.L], D.y, D.n, true);
      const g = mlpBackward(net, cache, dZ);
      mlpStep(net, g, { lr: lr.get(), opt: opt.get(), wd: reg ? +wd.get() : 0, decoupled: reg && wdMode.get() === 'dec' });
      epoch++;
    }
    function step(dt) {
      const cost = D.n * mlpParams(net);
      const per = clamp(Math.round(1.5e6 / cost), 1, 25);
      for (let i = 0; i < per; i++) {
        trainEpoch();
        if (epoch % (reg ? 5 : 2) === 0) record();
      }
      acc += dt;
      if (acc > 0.1) { acc = 0; draw(); }
      if (epoch >= 6000) { draw(); return false; }
    }
    function draw() {
      pR.clear();
      regionRaster(pR, net);
      drawData(pR, D);
      const last = hist[hist.length - 1];
      const ymax = Math.max(0.8, ...hist.map(h => Math.min(3, h.tr)), ...(reg ? hist.map(h => Math.min(3, h.te)) : [])) * 1.08;
      pL.setDomain([0, Math.max(100, epoch)], [0, ymax]).clear();
      pL.hline(Math.LN2, { color: 'faint', width: 1 });
      pL.text(Math.max(100, epoch), Math.LN2, 'log 2 (coin flip)', { anchor: 'end', dy: -4, size: 10, color: 'muted' });
      pL.line(hist.map(h => [h.e, h.tr]), { color: 's1', width: 2 });
      if (reg) {
        pL.line(hist.map(h => [h.e, h.te]), { color: 's2', width: 2 });
        let best = hist[0]; for (const h of hist) if (h.te < best.te) best = h;
        pL.dot(best.e, best.te, { color: 's2', r: 4 });
        pL.legend([{ label: 'train', color: 's1' }, { label: 'test', color: 's2' }]);
      }
      sE.set(String(epoch));
      sL.set(S.fmt(last.tr, 3), reg ? 'test ' + S.fmt(last.te, 3) : '');
      sA.set(`${(100 * last.atr).toFixed(0)}% / ${(100 * last.ate).toFixed(0)}%`);
      sP.set(String(mlpParams(net)), net.L === 1 ? 'logistic regression' : `${net.L - 1} hidden layer(s)`);
      if (sN) sN.set(S.fmt(mlpNorm(net), 3));
    }
    newData();
  }

  Atlas.lab('perceptron-mlp-playground', {
    title: 'A neural-network playground with hand-written backprop',
    blurb: 'Full-batch training of a small MLP with sigmoid-output cross-entropy. With 0 hidden layers this is logistic regression and can only draw a line: try XOR or Circles. Add one hidden layer of width 2–4 and watch the regions bend. Linear activations collapse any depth back to a line; a too-large learning rate makes the loss oscillate.',
    mount(el, ctx) { playground(el, ctx, { data: 'xor' }); },
  });

  /* ================================================================
   * Perceptron
   * ================================================================ */
  Atlas.lab('perceptron-mlp-perceptron', {
    title: 'The perceptron and Novikoff\'s mistake bound',
    blurb: 'The perceptron visits points in turn and updates $w \\leftarrow w + y_i\\tilde x_i$ only on a mistake. On separable data the total number of updates never exceeds $(R/\\gamma)^2$; shrink the margin and watch the bound (and the actual count) grow. On overlapping or XOR data it never settles.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const kind = UI.segmented(controls, { label: 'Data', options: [{ value: 'sep', label: 'Separable' }, { value: 'over', label: 'Overlapping' }, { value: 'xor', label: 'XOR' }], value: 'sep', onChange: newData });
      const gam = UI.slider(controls, { label: 'Enforced margin $\\gamma_0$ (separable)', min: 0, max: 0.6, step: 0.02, value: 0.2, onChange: newData });
      const n = UI.slider(controls, { label: 'Points $n$', min: 10, max: 200, step: 10, value: 60, onChange: newData });
      const speed = UI.slider(controls, { label: 'Checks per second', min: 2, max: 3000, log: true, value: 40, fmt: v => v.toFixed(0) });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Run', '❚❚ Pause']);
      UI.button(row, 'Next update', () => { loop.stop(); runUntilUpdate(); draw(); });
      UI.button(row, 'Reset w', () => { resetW(); draw(); });
      UI.button(row, 'New data', newData);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 300, x: [-2.5, 2.5], y: [-2.5, 2.5], equal: true, title: 'Data and current separator' });
      const pM = new Plot(stage, { width: 420, height: 300, x: [0, 5], y: [0, 10], xlabel: 'epochs (passes over data)', ylabel: 'cumulative updates', title: 'Mistakes vs. the Novikoff bound' });
      const sK = UI.stat(stats, 'Updates $k$', { accent: 'accent' });
      const sB = UI.stat(stats, 'Bound $(R/\\gamma)^2$');
      const sEp = UI.stat(stats, 'Epoch');
      const sErr = UI.stat(stats, 'Training errors now');
      const sSt = UI.stat(stats, 'Status');

      let X, Y, N, wTrue, w, k, checks, order, pos_, epochMistakes, converged, hist, lastIdx, R, gamma, budget = 0;
      function newData() {
        N = Math.round(n.get());
        X = []; Y = [];
        const r = ctx.rng, K = kind.get();
        if (K === 'sep') {
          const th = r.uniform(0, 2 * Math.PI), b = r.uniform(-0.5, 0.5);
          wTrue = [Math.cos(th), Math.sin(th), b];
          let guard = 0;
          while (X.length < N && guard++ < 100000) {
            const x = [r.uniform(-2.3, 2.3), r.uniform(-2.3, 2.3)];
            const d = wTrue[0] * x[0] + wTrue[1] * x[1] + b;
            if (Math.abs(d) < gam.get() || Math.abs(d) < 1e-3) continue;
            X.push(x); Y.push(d > 0 ? 1 : -1);
          }
        } else if (K === 'over') {
          wTrue = null;
          for (let i = 0; i < N; i++) { const c = i % 2 ? 1 : -1; X.push([0.6 * c + 0.8 * r.normal(), 0.4 * c + 0.8 * r.normal()]); Y.push(c); }
        } else {
          wTrue = null;
          for (let i = 0; i < N; i++) { const sx = r.u() < 0.5 ? -1 : 1, sy = r.u() < 0.5 ? -1 : 1; X.push([1.1 * sx + 0.4 * r.normal(), 1.1 * sy + 0.4 * r.normal()]); Y.push(sx * sy); }
        }
        N = X.length;
        R = Math.max(...X.map(x => Math.hypot(x[0], x[1], 1)));
        if (wTrue) {
          const nr = Math.hypot(...wTrue);
          gamma = Math.min(...X.map((x, i) => Y[i] * (wTrue[0] * x[0] + wTrue[1] * x[1] + wTrue[2]) / nr));
        } else gamma = NaN;
        resetW();
        draw();
      }
      function resetW() {
        w = [0, 0, 0]; k = 0; checks = 0; pos_ = 0; epochMistakes = 0; converged = false; hist = [[0, 0]]; lastIdx = -1;
        order = ctx.rng.shuffle(Array.from({ length: N }, (_, i) => i));
      }
      // One check; returns true if it caused an update.
      function check() {
        if (converged) return false;
        const i = order[pos_], x = X[i], y = Y[i];
        const s = w[0] * x[0] + w[1] * x[1] + w[2];
        let upd = false;
        if (y * s <= 0) { w[0] += y * x[0]; w[1] += y * x[1]; w[2] += y; k++; epochMistakes++; upd = true; lastIdx = i; hist.push([checks / N, k]); }
        checks++; pos_++;
        if (pos_ === N) {
          if (epochMistakes === 0) converged = true;
          pos_ = 0; epochMistakes = 0; ctx.rng.shuffle(order);
        }
        return upd;
      }
      function runUntilUpdate() { for (let g = 0; g < 50 * N && !converged; g++) if (check()) break; }
      function tick(dt) {
        budget += dt * speed.get();
        let c = 0;
        while (budget >= 1 && c < 20000) { check(); budget--; c++; if (converged) break; }
        draw();
        if (converged || checks / N > 400) return false;
      }
      function draw() {
        pD.clear();
        const pts = s => X.filter((_, i) => Y[i] === s);
        pD.points(X, { color: 'ink', r: 4.3 });
        pD.points(pts(-1), { color: 's1', r: 3.2, opacity: 1 });
        pD.points(pts(1), { color: 's2', r: 3.2, opacity: 1 });
        if (wTrue) {
          drawLine(pD, wTrue[0], wTrue[1], wTrue[2], { color: 'muted', dash: '5 4', width: 1.3 });
          if (gamma > 0) for (const s of [-1, 1]) drawLine(pD, wTrue[0], wTrue[1], wTrue[2] - s * gamma * Math.hypot(wTrue[0], wTrue[1]), { color: 'faint', dash: '2 4', width: 1 });
        }
        if (w[0] || w[1]) {
          drawLine(pD, w[0], w[1], w[2], { color: 'accent', width: 2.5 });
          const nn = Math.hypot(w[0], w[1]), cx = -w[2] * w[0] / (nn * nn), cy = -w[2] * w[1] / (nn * nn);
          if (Math.abs(cx) < 2.4 && Math.abs(cy) < 2.4) pD.arrow(cx, cy, cx + 0.6 * w[0] / nn, cy + 0.6 * w[1] / nn, { color: 'accent' });
        }
        if (lastIdx >= 0) pD.dot(X[lastIdx][0], X[lastIdx][1], { color: 'warn', r: 7, opacity: 0.35 });
        // mistakes vs epochs
        const bound = gamma > 0 ? (R / gamma) ** 2 : Infinity;
        const ep = checks / N;
        const ymax = Math.max(10, k * 1.15, Number.isFinite(bound) && bound < 4 * Math.max(k, 10) ? bound * 1.1 : 0);
        pM.setDomain([0, Math.max(3, ep * 1.05)], [0, ymax]).clear();
        if (Number.isFinite(bound) && bound <= ymax) { pM.hline(bound, { color: 'bad', dash: '6 4' }); pM.text(0.02 * Math.max(3, ep), bound, 'Novikoff bound', { dy: -5, size: 11, color: 'bad' }); }
        pM.step(hist.concat([[ep, k]]), { color: 'accent', width: 2 });
        let errs = 0;
        for (let i = 0; i < N; i++) if (Y[i] * (w[0] * X[i][0] + w[1] * X[i][1] + w[2]) <= 0) errs++;
        sK.set(String(k));
        sB.set(Number.isFinite(bound) ? S.fmt(bound, 4) : '—', wTrue ? `R = ${S.fmt(R, 3)}, γ = ${S.fmt(gamma, 3)}` : 'not separable: no bound');
        sEp.set(ep.toFixed(1));
        sErr.set(`${errs} / ${N}`);
        sSt.set(converged ? '<span style="color:var(--good)">converged</span>' : (ep > 400 ? 'stopped (400 epochs)' : 'running'), converged ? `zero mistakes in a full pass` : '');
      }
      newData();
    },
  });

  /* ================================================================
   * Universal approximation
   * ================================================================ */
  const TARGETS = {
    smooth: { label: 'Smooth', f: x => Math.sin(2 * Math.PI * x) + 0.4 * x * Math.sin(6 * Math.PI * x) },
    kink: { label: 'Kink', f: x => 1.6 * Math.abs(x - 0.37) - 0.4 + 0.25 * Math.sin(3 * x) },
    wiggly: { label: 'Wiggly', f: x => 0.8 * Math.sin(1 / (x + 0.07)) },
    step: { label: 'Step (discontinuous)', f: x => (x > 0.53 ? 0.6 : -0.6) },
  };
  // Piecewise-linear interpolant on N uniform cells written as a ReLU network:
  //   g(x) = f(t0) + sum_k c_k relu(x - t_k),  c_0 = s_0, c_k = s_k - s_{k-1}
  function reluInterp(f, N) {
    const t = S.linspace(0, 1, N + 1), v = t.map(f), h = 1 / N;
    const s = []; for (let k = 0; k < N; k++) s.push((v[k + 1] - v[k]) / h);
    const c = s.map((sk, k) => (k === 0 ? sk : sk - s[k - 1]));
    const g = x => { let y = v[0]; for (let k = 0; k < N; k++) { const r = x - t[k]; if (r > 0) y += c[k] * r; } return y; };
    return { t, v, c, g, h };
  }
  function interpErrors(f, N, grid, fg) {
    const t = S.linspace(0, 1, N + 1), v = t.map(f);
    let sup = 0, l2 = 0;
    for (let i = 0; i < grid.length; i++) {
      const x = grid[i], k = Math.min(N - 1, Math.floor(x * N)), u = x * N - k;
      const e = Math.abs(v[k] + u * (v[k + 1] - v[k]) - fg[i]);
      if (e > sup) sup = e;
      l2 += e * e;
    }
    return { sup, l2: Math.sqrt(l2 / grid.length) };
  }

  Atlas.lab('universal-approximation-bumps', {
    title: 'Building a function out of ReLUs',
    blurb: 'A one-hidden-layer ReLU network with $N$ units, $g(x)=b+\\sum_k c_k\\,(x-t_k)_+$, can represent any continuous piecewise-linear function with knots $t_k$ — in particular the linear interpolant of $f$. Increase $N$ and read the error rate off the log–log plot: slope $-2$ for smooth targets, $-1$ at a kink, and no convergence at all in sup-norm for a jump.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tg = UI.select(controls, { label: 'Target $f$', options: Object.keys(TARGETS).map(k => ({ value: k, label: TARGETS[k].label })), value: 'smooth', onChange: () => { curve = null; draw(); } });
      const N = UI.slider(controls, { label: 'Hidden units $N$', min: 1, max: 128, step: 1, value: 6, onInput: draw });
      const view = UI.segmented(controls, { label: 'Show decomposition', options: [{ value: 'none', label: 'None' }, { value: 'ramp', label: 'ReLU ramps' }, { value: 'hat', label: 'Hat bumps' }], value: 'hat', onChange: draw });
      stage.classList.add('two');
      const pF = new Plot(stage, { width: 420, height: 290, x: [0, 1], y: [-1.6, 1.6], xlabel: 'x', title: 'Target and network output' });
      const pE = new Plot(stage, { width: 420, height: 290, x: [1, 512], y: [1e-5, 10], logx: true, logy: true, xlabel: 'hidden units N', ylabel: 'error', title: 'Approximation error vs. width' });
      const sN = UI.stat(stats, 'Units / parameters');
      const sSup = UI.stat(stats, 'Sup error $\\|f-g\\|_\\infty$', { accent: 'accent' });
      const sL2 = UI.stat(stats, '$L^2$ error');
      const sRate = UI.stat(stats, 'Empirical rate (sup)');
      const grid = S.linspace(0, 1, 2001);
      let curve = null;
      function computeCurve() {
        const f = TARGETS[tg.get()].f, fg = grid.map(f);
        const Ns = [...new Set(logspace(1, 512, 34).map(Math.round))];
        curve = Ns.map(n => Object.assign({ n }, interpErrors(f, n, grid, fg)));
        curve.fg = fg;
      }
      function draw() {
        if (!curve) computeCurve();
        const f = TARGETS[tg.get()].f, n = Math.round(N.get()), net = reluInterp(f, n);
        const xs = S.linspace(0, 1, 600);
        pF.clear();
        const V = view.get();
        if (V === 'ramp' && n <= 40) {
          // cumulative partial sums show how each ramp bends the function
          for (let k = 0; k < n; k++) pF.line(xs.map(x => [x, net.c[k] * Math.max(0, x - net.t[k]) * 0.15]), { color: 's3', width: 1, opacity: 0.55 });
          pF.text(0.02, -1.45, 'ramps c_k(x − t_k)₊, scaled ×0.15', { size: 10, color: 'muted' });
        } else if (V === 'hat' && n <= 64) {
          for (let k = 0; k <= n; k++) {
            const tk = net.t[k], h = net.h, vk = net.v[k];
            const hx = [tk - h, tk, tk + h].map(x => clamp(x, 0, 1));
            pF.line(hx.map(x => [x, vk * Math.max(0, 1 - Math.abs(x - tk) / h)]), { color: 's3', width: 1, opacity: 0.55 });
          }
          pF.text(0.02, -1.45, 'hat bumps f(t_k)·φ_k(x); each = 3 ReLUs', { size: 10, color: 'muted' });
        }
        pF.line(xs.map(x => [x, f(x)]), { color: 'ink', width: 2, opacity: 0.6 });
        pF.line(xs.map(x => [x, net.g(x)]), { color: 'accent', width: 2.2 });
        if (n <= 64) pF.points(net.t.map((t, k) => [t, net.v[k]]), { color: 'accent', r: 2.5 });
        pF.legend([{ label: 'target f', color: 'ink' }, { label: 'ReLU network g', color: 'accent' }]);

        pE.clear();
        pE.line(curve.map(c => [c.n, pos(c.sup)]), { color: 's1', width: 2 });
        pE.line(curve.map(c => [c.n, pos(c.l2)]), { color: 's2', width: 2 });
        // reference slopes anchored at N = 4
        const c4 = curve.find(c => c.n >= 4) || curve[0];
        pE.line([[4, c4.sup * 2], [512, c4.sup * 2 * (4 / 512) ** 2]], { color: 'faint', dash: '4 4', width: 1.2 });
        pE.line([[4, c4.sup * 2], [512, c4.sup * 2 * (4 / 512)]], { color: 'faint', dash: '1 3', width: 1.2 });
        pE.text(512, c4.sup * 2 * (4 / 512) ** 2, 'slope −2', { anchor: 'end', dy: -5, size: 10, color: 'muted' });
        pE.text(512, c4.sup * 2 * (4 / 512), 'slope −1', { anchor: 'end', dy: -5, size: 10, color: 'muted' });
        const cur = interpErrors(f, n, grid, curve.fg);
        pE.vline(n, { color: 'accent', width: 1.2 });
        pE.legend([{ label: 'sup error', color: 's1' }, { label: 'L² error', color: 's2' }], { left: true });
        const a = curve[curve.length - 8], b = curve[curve.length - 1];
        const rate = Math.log(b.sup / a.sup) / Math.log(b.n / a.n);
        sN.set(`${n} / ${3 * n + 1}`);
        sSup.set(S.fmt(cur.sup, 3));
        sL2.set(S.fmt(cur.l2, 3));
        sRate.set(Number.isFinite(rate) ? S.fmt(rate, 2) : '—', 'log-log slope at large N');
      }
      draw();
    },
  });

  Atlas.lab('universal-approximation-sawtooth', {
    title: 'Depth separation: Telgarsky\'s sawtooth',
    blurb: 'The tent map $T(x)=2(x)_+-4(x-\\tfrac12)_+$ costs 2 ReLUs. Composing it $k$ times gives a sawtooth with $2^k$ linear pieces using only $2k$ units. A one-hidden-layer ReLU net with $m$ units has at most $m+1$ pieces in 1D, so it needs exponentially many units; the shallow approximant shown (the interpolant on $m$ uniform cells) stays far away until $m \\gtrsim 2^k$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const K = UI.slider(controls, { label: 'Depth $k$ (compositions)', min: 1, max: 9, step: 1, value: 4, onInput: draw });
      const m = UI.slider(controls, { label: 'Shallow width $m$', min: 1, max: 160, step: 1, value: 10, onInput: draw });
      const sh = UI.toggle(controls, { label: 'Show shallow approximant', value: true, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 1], y: [-0.05, 1.08], xlabel: 'x', title: '' });
      const sPc = UI.stat(stats, 'Pieces of $T^{k}$', { accent: 'accent' });
      const sDeep = UI.stat(stats, 'Deep ReLU units');
      const sSh = UI.stat(stats, 'Shallow units needed (exact)');
      const sErr = UI.stat(stats, 'Shallow $L^1$ error');
      const relu = z => (z > 0 ? z : 0);
      const tent = x => 2 * relu(x) - 4 * relu(x - 0.5);
      const Tk = (x, k) => { for (let i = 0; i < k; i++) x = tent(x); return x; };
      function draw() {
        const k = Math.round(K.get()), mm = Math.round(m.get());
        const xs = S.linspace(0, 1, 4097), ys = xs.map(x => Tk(x, k));
        p.clear();
        p.line(xs.map((x, i) => [x, ys[i]]), { color: 'accent', width: 1.8 });
        // shallow approximant: interpolant of T^k on m uniform cells (m units incl. the affine part)
        const knots = S.linspace(0, 1, mm + 1), kv = knots.map(x => Tk(x, k));
        const approx = x => { const j = Math.min(mm - 1, Math.floor(x * mm)), u = x * mm - j; return kv[j] + u * (kv[j + 1] - kv[j]); };
        let l1 = 0;
        for (let i = 0; i < xs.length; i++) l1 += Math.abs(approx(xs[i]) - ys[i]);
        l1 /= xs.length;
        if (sh.get()) { p.line(xs.map(x => [x, approx(x)]), { color: 's2', width: 2, opacity: 0.85 }); p.points(knots.map((x, j) => [x, kv[j]]), { color: 's2', r: 2.5 }); }
        p.legend([{ label: `deep: T∘…∘T (${k}×)`, color: 'accent' }].concat(sh.get() ? [{ label: `shallow, ${mm} units`, color: 's2' }] : []));
        sPc.set(String(2 ** k));
        sDeep.set(String(2 * k), `${k} layers × 2 units`);
        sSh.set('≥ ' + String(2 ** k - 1), 'm units ⇒ ≤ m+1 pieces');
        sErr.set(S.fmt(l1, 3), 'mean |T^k − approximant| (T^k has mean ½)');
      }
      draw();
    },
  });

  /* ================================================================
   * Backpropagation
   * ================================================================ */
  Atlas.lab('backpropagation-graph', {
    title: 'Numbers forward, gradients backward',
    blurb: 'A 2–2–1 network $\\hat y = v^\\top \\phi(Wx+b)+c$ with loss $L=\\tfrac12(\\hat y-y)^2$, drawn as a computational graph. Step through the forward pass (values, black) and then the backward pass (adjoints $\\bar u = \\partial L/\\partial u$, coloured): each node only multiplies the adjoint it receives by its local derivative. The table checks every backprop gradient against a central finite difference.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const x1 = UI.slider(controls, { label: 'Input $x_1$', min: -2, max: 2, step: 0.05, value: 1, onInput: draw });
      const x2 = UI.slider(controls, { label: 'Input $x_2$', min: -2, max: 2, step: 0.05, value: -0.5, onInput: draw });
      const yT = UI.slider(controls, { label: 'Target $y$', min: -2, max: 2, step: 0.05, value: 1, onInput: draw });
      const act = UI.segmented(controls, { label: 'Activation $\\phi$', options: ['tanh', 'sigmoid', 'relu'].map(v => ({ value: v, label: ACTS[v].label })), value: 'tanh', onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      let stageN = 8, acc = 0;
      const loop = ctx.loop(dt => { acc += dt; if (acc > 0.8) { acc = 0; stageN++; draw(); } if (stageN >= 8) return false; });
      UI.button(row, '▶ Animate pass', () => { stageN = 0; acc = 0; draw(); loop.start(); }, { primary: true });
      UI.button(row, 'Step', () => { loop.stop(); stageN = (stageN + 1) % 9; draw(); });
      UI.button(row, 'Random weights', () => { randomize(); draw(); });
      const p = new Plot(stage, { width: 640, height: 330, x: [0, 10], y: [0.2, 6.4], grid: false, xticks: 0, yticks: 0, margin: { t: 26, r: 8, b: 8, l: 8 }, title: '' });
      const sL = UI.stat(stats, 'Loss $L$', { accent: 'accent' });
      const sMax = UI.stat(stats, 'Max relative error (backprop vs. FD)');
      const tbl = UI.note(stats, '');
      const P = { W: [0.8, -0.6, 0.4, 0.9], b: [0.1, -0.2], v: [1.1, -0.7], c: 0.05 };
      function randomize() {
        const r = ctx.rng;
        P.W = P.W.map(() => r.normal(0, 0.9)); P.b = P.b.map(() => r.normal(0, 0.3)); P.v = P.v.map(() => r.normal(0, 1)); P.c = r.normal(0, 0.2);
      }
      const flat = Q => [...Q.W, ...Q.b, ...Q.v, Q.c];
      const unflat = a => ({ W: a.slice(0, 4), b: a.slice(4, 6), v: a.slice(6, 8), c: a[8] });
      const NAMES = ['W₁₁', 'W₁₂', 'W₂₁', 'W₂₂', 'b₁', 'b₂', 'v₁', 'v₂', 'c'];
      function forward(Q, x, y, A) {
        const z = [Q.W[0] * x[0] + Q.W[1] * x[1] + Q.b[0], Q.W[2] * x[0] + Q.W[3] * x[1] + Q.b[1]];
        const h = z.map(A.f), yh = Q.v[0] * h[0] + Q.v[1] * h[1] + Q.c;
        return { z, h, yh, L: 0.5 * (yh - y) ** 2 };
      }
      function backward(Q, x, y, A, F) {
        const dyh = F.yh - y;
        const dv = [dyh * F.h[0], dyh * F.h[1]], dc = dyh;
        const dh = [dyh * Q.v[0], dyh * Q.v[1]];
        const dz = [dh[0] * A.d(F.z[0]), dh[1] * A.d(F.z[1])];
        const dW = [dz[0] * x[0], dz[0] * x[1], dz[1] * x[0], dz[1] * x[1]], db = dz.slice();
        const dx = [Q.W[0] * dz[0] + Q.W[2] * dz[1], Q.W[1] * dz[0] + Q.W[3] * dz[1]];
        return { dyh, dv, dc, dh, dz, dW, db, dx };
      }
      const N = {
        x1: [0.9, 4.7], x2: [0.9, 1.5], z1: [3.4, 4.7], z2: [3.4, 1.5], h1: [5.6, 4.7], h2: [5.6, 1.5], yh: [7.6, 3.1], L: [9.3, 3.1],
      };
      const f3 = v => (Math.abs(v) < 5e-4 ? '0' : v.toFixed(3).replace('-', '−'));
      function node(key, label, val, grad, showV, showG) {
        const [x, y] = N[key];
        p.dot(x, y, { r: 17, color: 'surface', stroke: showG ? 's2' : 'ink', strokeWidth: showG ? 2.2 : 1.4 });
        p.text(x, y, label, { anchor: 'middle', dy: 4, size: 13, weight: 600 });
        if (showV) p.text(x, y + 0.6, f3(val), { anchor: 'middle', size: 11.5 });
        if (showG) p.text(x, y - 0.75, '∂ ' + f3(grad), { anchor: 'middle', size: 11.5, color: 's2', weight: 600 });
      }
      function edge(a, b, label, grad, showG, t = 0.3) {
        const [ax, ay] = N[a], [bx, by] = N[b];
        const dx = bx - ax, dy = by - ay, len = Math.hypot(dx / 10 * 624, dy / 6.2 * 296);
        const sh = 19 / len;
        p.arrow(ax + dx * sh, ay + dy * sh, bx - dx * sh, by - dy * sh, { color: 'muted', width: 1.2 });
        if (label != null) {
          const lx = ax + dx * t, ly = ay + dy * t;
          p.text(lx, ly, label, { anchor: 'middle', dy: -5, size: 10.5, color: 'muted' });
          if (showG) p.text(lx, ly, '∂ ' + f3(grad), { anchor: 'middle', dy: 10, size: 10.5, color: 's2' });
        }
      }
      function draw() {
        const A = ACTS[act.get()], x = [x1.get(), x2.get()], y = yT.get();
        const F = forward(P, x, y, A), B = backward(P, x, y, A, F);
        const s = stageN;
        const fv = s >= 0, fz = s >= 1, fh = s >= 2, fy = s >= 3, fL = s >= 4;
        const gy = s >= 5, gh = s >= 6, gz = s >= 7, gx = s >= 8;
        p.clear();
        const titles = ['inputs', 'forward: z = Wx + b', 'forward: h = φ(z)', 'forward: ŷ = vᵀh + c', 'forward: L = ½(ŷ − y)²',
          'backward: ∂L/∂ŷ = ŷ − y', 'backward: ∂L/∂h = (∂L/∂ŷ)·v,  ∂L/∂v = (∂L/∂ŷ)·h', 'backward: ∂L/∂z = ∂L/∂h ⊙ φ′(z)', 'backward: ∂L/∂W = (∂L/∂z) xᵀ,  ∂L/∂x = Wᵀ(∂L/∂z)'];
        p.text(0.1, 6.35, `Stage ${s}/8 — ${titles[s]}`, { size: 12.5, weight: 600, dy: -6 });
        edge('x1', 'z1', 'W₁₁=' + f3(P.W[0]), B.dW[0], gx, 0.32);
        edge('x2', 'z1', 'W₁₂=' + f3(P.W[1]), B.dW[1], gx, 0.22);
        edge('x1', 'z2', 'W₂₁=' + f3(P.W[2]), B.dW[2], gx, 0.22);
        edge('x2', 'z2', 'W₂₂=' + f3(P.W[3]), B.dW[3], gx, 0.32);
        edge('z1', 'h1', 'φ', 0, false, 0.5);
        edge('z2', 'h2', 'φ', 0, false, 0.5);
        edge('h1', 'yh', 'v₁=' + f3(P.v[0]), B.dv[0], gh, 0.45);
        edge('h2', 'yh', 'v₂=' + f3(P.v[1]), B.dv[1], gh, 0.45);
        edge('yh', 'L', null);
        node('x1', 'x₁', x[0], B.dx[0], fv, gx); node('x2', 'x₂', x[1], B.dx[1], fv, gx);
        node('z1', 'z₁', F.z[0], B.dz[0], fz, gz); node('z2', 'z₂', F.z[1], B.dz[1], fz, gz);
        node('h1', 'h₁', F.h[0], B.dh[0], fh, gh); node('h2', 'h₂', F.h[1], B.dh[1], fh, gh);
        node('yh', 'ŷ', F.yh, B.dyh, fy, gy);
        node('L', 'L', F.L, 1, fL, gy);
        p.text(N.z1[0], N.z1[1] + 1.05, `b₁=${f3(P.b[0])}` + (gz ? `  ∂ ${f3(B.db[0])}` : ''), { anchor: 'middle', size: 10, color: gz ? 's2' : 'muted' });
        p.text(N.z2[0], N.z2[1] + 1.05, `b₂=${f3(P.b[1])}` + (gz ? `  ∂ ${f3(B.db[1])}` : ''), { anchor: 'middle', size: 10, color: gz ? 's2' : 'muted' });
        p.text(N.yh[0], N.yh[1] + 1.05, `c=${f3(P.c)}` + (gy ? `  ∂ ${f3(B.dc)}` : ''), { anchor: 'middle', size: 10, color: gy ? 's2' : 'muted' });
        p.text(N.L[0], N.L[1] + 1.15, `y = ${f3(y)}`, { anchor: 'middle', size: 11, color: 'muted' });
        // gradient check
        const th = flat(P), bp = [...B.dW, ...B.db, ...B.dv, B.dc], eps = 1e-5;
        let worst = 0, rows = '';
        th.forEach((v, i) => {
          const a = th.slice(), b = th.slice(); a[i] += eps; b[i] -= eps;
          const fd = (forward(unflat(a), x, y, A).L - forward(unflat(b), x, y, A).L) / (2 * eps);
          const rel = Math.abs(bp[i] - fd) / Math.max(1e-10, Math.abs(bp[i]) + Math.abs(fd));
          worst = Math.max(worst, rel);
          rows += `<tr><td>${NAMES[i]}</td><td>${bp[i].toExponential(4)}</td><td>${fd.toExponential(4)}</td><td>${rel.toExponential(1)}</td></tr>`;
        });
        sL.set(S.fmt(F.L, 4));
        sMax.set(worst.toExponential(1), worst < 1e-6 ? 'agreement to ~10 digits' : 'a ReLU kink near z = 0, or a vanishing gradient');
        tbl.set(`<table><thead><tr><th>param</th><th>backprop</th><th>finite diff.</th><th>rel. err</th></tr></thead><tbody>${rows}</tbody></table>`);
      }
      draw();
    },
  });

  Atlas.lab('backpropagation-vanishing', {
    title: 'Vanishing and exploding gradients',
    blurb: 'A deep fully connected net of width 32 at initialization, weights $\\N(0, g/n_{\\text{in}})$. Backprop multiplies the error signal by $W_\\ell^\\top \\diag(\\phi\'(z_\\ell))$ once per layer, so its size changes geometrically with depth. Sigmoid ($\\phi\' \\le 1/4$) vanishes for any reasonable $g$; ReLU is balanced at $g = 2$ (He); tanh at $g\\approx 1$. Move $g$ off the critical value and watch the slope.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const D = UI.slider(controls, { label: 'Hidden layers $D$', min: 2, max: 60, step: 1, value: 30, onChange: compute });
      const act = UI.segmented(controls, { label: 'Activation', options: ['sigmoid', 'tanh', 'relu'].map(v => ({ value: v, label: ACTS[v].label })), value: 'tanh', onChange: compute });
      const g = UI.slider(controls, { label: 'Init gain $g$ (Var $= g/n_{\\text{in}}$)', min: 0.25, max: 6, log: true, value: 1, fmt: v => S.fmt(v, 3), onChange: compute });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Critical g', () => { g.set(act.get() === 'relu' ? 2 : act.get() === 'tanh' ? 1 : 6); compute(); });
      UI.button(row, 'New weights', compute);
      const p = new Plot(stage, { width: 640, height: 310, x: [0, 31], y: [1e-8, 1e4], logy: true, xlabel: 'layer ℓ (1 = closest to input)', ylabel: 'norm', title: 'Gradient size by layer at initialization' });
      const sR = UI.stat(stats, '$\\|\\partial L/\\partial W_1\\| \\,/\\, \\|\\partial L/\\partial W_D\\|$', { accent: 'accent' });
      const sG = UI.stat(stats, 'Per-layer factor of $\\|\\delta_\\ell\\|$');
      const sA = UI.stat(stats, 'Mean $\\phi\'(z)$, first hidden layer');
      function compute() {
        const d = Math.round(D.get()), w = 32, n = 64;
        const sizes = [w]; for (let i = 0; i < d; i++) sizes.push(w); sizes.push(1);
        const net = mlpNew(sizes, act.get(), ctx.rng, g.get());
        const X = new Float64Array(n * w); for (let i = 0; i < X.length; i++) X[i] = ctx.rng.normal();
        const cache = mlpForward(net, X, n, 0);
        const out = cache.A[net.L], dZ = new Float64Array(n);
        for (let i = 0; i < n; i++) dZ[i] = out[i] / n; // L = (1/2n) sum f^2
        const gr = mlpBackward(net, cache, dZ);
        const gn = gr.gW.map(G => Math.sqrt(G.reduce((s, v) => s + v * v, 0)));
        const dn = gr.deltas.map(Dl => Math.sqrt(Dl.reduce((s, v) => s + v * v, 0)));
        const an = cache.A.slice(1, -1).map(A => Math.sqrt(A.reduce((s, v) => s + v * v, 0) / A.length));
        let dsum = 0; const z1 = cache.Z[1]; for (let q = 0; q < z1.length; q++) dsum += ACTS[act.get()].d(z1[q]);
        // layers 1..d are hidden weight matrices W_0..W_{d-1}; W_d is the output layer
        const L = gn.length;
        const all = gn.concat(dn, an).filter(v => v > 0 && Number.isFinite(v));
        const lo = Math.max(1e-30, Math.min(...all) / 3), hi = Math.max(...all) * 3;
        p.setDomain([0, L + 1], [lo, hi]).clear();
        p.line(gn.map((v, i) => [i + 1, pos(v)]), { color: 's1', width: 2 });
        p.points(gn.map((v, i) => [i + 1, pos(v)]), { color: 's1', r: 3 });
        p.line(dn.map((v, i) => [i + 1, pos(v)]), { color: 's2', width: 2, dash: '5 4' });
        p.line(an.map((v, i) => [i + 1, pos(v)]), { color: 's4', width: 1.8, opacity: 0.8 });
        p.legend([{ label: '‖∂L/∂W_ℓ‖ (gradient)', color: 's1' }, { label: '‖δ_ℓ‖ (backprop signal)', color: 's2', dash: true }, { label: 'RMS activation h_ℓ', color: 's4' }], { left: true });
        const r = gn[0] / gn[L - 2];
        sR.set(Number.isFinite(r) ? r.toExponential(1) : '—', r < 1e-2 ? 'vanishing' : r > 1e2 ? 'exploding' : 'balanced');
        const fac = (dn[0] / dn[L - 2]) ** (1 / Math.max(1, L - 2));
        sG.set(Number.isFinite(fac) ? S.fmt(fac, 3) : '—', 'geometric mean, input-ward');
        sA.set(S.fmt(dsum / z1.length, 3));
      }
      compute();
    },
  });

  /* ================================================================
   * Optimizers
   * ================================================================ */
  function makeSurface(kind, kappa) {
    if (kind === 'quad') {
      const a = 0.5, c = Math.cos(a), s = Math.sin(a), l1 = 1, l2 = 1 / kappa;
      return {
        dom: [[-3, 3], [-2.2, 2.2]], start: [-2.6, 1.6], fstar: 0,
        f: (x, y) => { const u = c * x + s * y, v = -s * x + c * y; return 0.5 * (l2 * u * u + l1 * v * v); },
        g: (x, y) => { const u = c * x + s * y, v = -s * x + c * y, gu = l2 * u, gv = l1 * v; return [c * gu - s * gv, s * gu + c * gv]; },
        levels: logspace(1e-3, 8, 12), defaults: [0.9, 0.05],
      };
    }
    if (kind === 'rosen') {
      return {
        dom: [[-2, 2], [-1, 3]], start: [-1.5, 2.4], fstar: 0,
        f: (x, y) => (1 - x) ** 2 + 100 * (y - x * x) ** 2,
        g: (x, y) => [-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)],
        levels: logspace(0.05, 2000, 13), defaults: [0.0004, 0.03],
      };
    }
    return { // saddle at origin, minima at (0, ±1)
      dom: [[-2, 2], [-1.8, 1.8]], start: [1.8, 1e-4], fstar: -0.25,
      f: (x, y) => 0.5 * x * x - 0.5 * y * y + 0.25 * y ** 4,
      g: (x, y) => [x, -y + y ** 3],
      levels: S.linspace(-0.2, 2, 12), defaults: [0.1, 0.02],
    };
  }
  const OPTS = [['GD', 'gd', 's1'], ['Momentum', 'mom', 's2'], ['Nesterov', 'nag', 's3'], ['RMSProp', 'rms', 's4'], ['Adam', 'adam', 's5']];

  Atlas.lab('sgd-optimizers-race', {
    title: 'Optimizer race on hard surfaces',
    blurb: 'Five optimizers from the same start. The narrow valley of an ill-conditioned quadratic punishes plain GD (rate $1-1/\\kappa$); momentum gets $\\approx 1-1/\\sqrt\\kappa$. Rosenbrock\'s curved valley forces a tiny GD step. At a saddle, GD escapes only exponentially slowly from a tiny initial offset, while the per-coordinate normalization of RMSProp/Adam amplifies it. Add gradient noise to see which methods average it out.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const surf = UI.select(controls, { label: 'Surface', options: [{ value: 'quad', label: 'Ill-conditioned quadratic' }, { value: 'rosen', label: 'Rosenbrock valley' }, { value: 'saddle', label: 'Saddle point' }], value: 'quad', onChange: () => { S0 = null; defaults(); reset(); } });
      const kap = UI.slider(controls, { label: 'Condition number $\\kappa$ (quadratic)', min: 1, max: 300, log: true, value: 50, fmt: v => v.toFixed(0), onChange: () => { S0 = null; reset(); } });
      const etaG = UI.slider(controls, { label: 'Step $\\eta$: GD / momentum / Nesterov', min: 1e-4, max: 2, log: true, value: 0.9, fmt: v => S.fmt(v, 2), onChange: reset });
      const etaA = UI.slider(controls, { label: 'Step $\\eta$: RMSProp / Adam', min: 1e-4, max: 1, log: true, value: 0.05, fmt: v => S.fmt(v, 2), onChange: reset });
      const beta = UI.slider(controls, { label: 'Momentum $\\beta$', min: 0, max: 0.99, step: 0.01, value: 0.9, onChange: reset });
      const noise = UI.slider(controls, { label: 'Gradient noise s.d.', min: 0, max: 2, step: 0.05, value: 0, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Run', '❚❚ Pause']);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pC = new Plot(stage, { width: 420, height: 300, title: 'Trajectories' });
      const pF = new Plot(stage, { width: 420, height: 300, x: [0, 1500], y: [1e-10, 10], logy: true, xlabel: 'iteration', ylabel: 'f − f*', title: 'Suboptimality' });
      const tiles = OPTS.map(([name, , c]) => UI.stat(stats, `<span style="color:var(--${c})">■</span> ${name}`));
      let S0 = null, st, it, acc = 0;
      const TMAX = 1500;
      function surface() { if (!S0) S0 = makeSurface(surf.get(), kap.get()); return S0; }
      function defaults() { const d = surface().defaults; etaG.set(d[0]); etaA.set(d[1]); }
      function reset() {
        loop.stop();
        const sf = surface();
        it = 0;
        st = OPTS.map(() => ({ th: sf.start.slice(), v: [0, 0], m: [0, 0], s: [0, 0], path: [sf.start.slice()], hist: [[0, sf.f(...sf.start) - sf.fstar]], dead: false }));
        draw();
      }
      function stepAll() {
        const sf = surface(), b = beta.get(), eG = etaG.get(), eA = etaA.get(), sd = noise.get();
        it++;
        OPTS.forEach(([, k], j) => {
          const o = st[j];
          if (o.dead) return;
          const nz = () => (sd ? sd * ctx.rng.normal() : 0);
          let g;
          if (k === 'nag') { g = sf.g(o.th[0] + b * o.v[0], o.th[1] + b * o.v[1]); } else g = sf.g(o.th[0], o.th[1]);
          g = [g[0] + nz(), g[1] + nz()];
          for (let i = 0; i < 2; i++) {
            if (k === 'gd') o.th[i] -= eG * g[i];
            else if (k === 'mom' || k === 'nag') { o.v[i] = b * o.v[i] - eG * g[i]; o.th[i] += o.v[i]; }
            else if (k === 'rms') { o.s[i] = 0.9 * o.s[i] + 0.1 * g[i] * g[i]; o.th[i] -= eA * g[i] / (Math.sqrt(o.s[i]) + 1e-8); }
            else {
              o.m[i] = 0.9 * o.m[i] + 0.1 * g[i]; o.s[i] = 0.999 * o.s[i] + 0.001 * g[i] * g[i];
              o.th[i] -= eA * (o.m[i] / (1 - 0.9 ** it)) / (Math.sqrt(o.s[i] / (1 - 0.999 ** it)) + 1e-8);
            }
          }
          const fv = sf.f(o.th[0], o.th[1]) - sf.fstar;
          if (!Number.isFinite(fv) || Math.abs(o.th[0]) > 1e3 || Math.abs(o.th[1]) > 1e3) { o.dead = true; return; }
          o.path.push(o.th.slice());
          o.hist.push([it, fv]);
        });
      }
      function tick(dt) {
        for (let i = 0; i < 6 && it < TMAX; i++) stepAll();
        acc += dt;
        if (acc > 0.06 || it >= TMAX) { acc = 0; draw(); }
        if (it >= TMAX) return false;
      }
      function draw() {
        const sf = surface();
        pC.setDomain(sf.dom[0], sf.dom[1]).clear();
        pC.contour(sf.f, sf.levels, { color: 'faint', width: 1, nx: 70, ny: 50 });
        const opt = surf.get() === 'rosen' ? [[1, 1]] : surf.get() === 'quad' ? [[0, 0]] : [[0, 1], [0, -1]];
        for (const q of opt) pC.dot(q[0], q[1], { color: 'ink', r: 4 });
        pC.dot(sf.start[0], sf.start[1], { color: 'muted', r: 4 });
        OPTS.forEach(([, , c], j) => { pC.line(st[j].path, { color: c, width: 1.8, opacity: 0.9 }); const t = st[j].th; pC.dot(t[0], t[1], { color: c, r: 4 }); });
        const all = st.flatMap(o => o.hist.map(h => h[1])).filter(v => v > 0 && Number.isFinite(v));
        const hi = Math.max(1e-6, ...all) * 3, lo = Math.max(1e-14, Math.min(hi / 1e3, ...all) / 3);
        pF.setDomain([0, TMAX], [lo, hi]).clear();
        OPTS.forEach(([, , c], j) => pF.line(st[j].hist.map(([t, v]) => [t, pos(v)]), { color: c, width: 1.8 }));
        OPTS.forEach(([name], j) => {
          const o = st[j], last = o.hist[o.hist.length - 1][1];
          tiles[j].set(o.dead ? '<span style="color:var(--bad)">diverged</span>' : S.fmt(last, 3), `f − f* after ${it} its`);
        });
      }
      defaults();
      reset();
    },
  });

  Atlas.lab('sgd-optimizers-noise', {
    title: 'SGD noise: the constant-step floor and how decay removes it',
    blurb: 'SGD on least squares with mini-batches of size $B$. With a constant step the iterates converge in expectation only to a noise ball whose size scales like $\\eta\\, s^2 d/(2B)$ (dashed line); a $1/t$ schedule or Polyak averaging of the constant-step iterates drives the error to zero at rate $O(1/t)$. Raise $\\eta$ or shrink $B$ to lift the floor.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const eta = UI.slider(controls, { label: 'Initial step $\\eta_0$', min: 0.003, max: 0.2, log: true, value: 0.05, fmt: v => S.fmt(v, 2), onChange: reset });
      const B = UI.segmented(controls, { label: 'Mini-batch size $B$', options: ['1', '8', '64'], value: '1', onChange: reset });
      const sig_ = UI.slider(controls, { label: 'Noise s.d. $\\sigma$', min: 0.1, max: 2, step: 0.05, value: 1, onChange: newData });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Run SGD', '❚❚ Pause']);
      UI.button(row, 'Reset', reset);
      UI.button(row, 'New data', newData);
      stage.classList.add('two');
      const pC = new Plot(stage, { width: 420, height: 300, equal: true, title: 'Iterates over the empirical-risk contours', xlabel: 'θ₁', ylabel: 'θ₂' });
      const pE = new Plot(stage, { width: 420, height: 300, x: [1, 3e4], y: [1e-6, 10], logx: true, logy: true, xlabel: 'iteration t', ylabel: '‖θ_t − θ̂‖²', title: 'Distance to the minimizer' });
      const SCH = [['Constant η', 's1'], ['η_t ∝ 1/√t', 's3'], ['η_t ∝ 1/t', 's2'], ['Polyak avg. (const.)', 's5']];
      const tiles = SCH.map(([name, c]) => UI.stat(stats, `<span style="color:var(--${c})">■</span> ${name}`));
      const sFloor = UI.stat(stats, 'Predicted floor $\\eta s^2 d/(2B)$');
      const n = 400, d = 2;
      let X, Y, thHat, G, mu, s2, th, avg, paths, hist, t, nextRec, ema, acc = 0;
      const TMAX = 30000;
      function newData() {
        const r = ctx.rng, rho = 0.7, sd = sig_.get();
        X = new Float64Array(2 * n); Y = new Float64Array(n);
        for (let i = 0; i < n; i++) {
          const z1 = r.normal(), z2 = r.normal(), a = z1, b = rho * z1 + Math.sqrt(1 - rho * rho) * z2;
          X[2 * i] = a; X[2 * i + 1] = b; Y[i] = 1.0 * a - 0.5 * b + sd * r.normal();
        }
        let g11 = 0, g12 = 0, g22 = 0, r1 = 0, r2 = 0;
        for (let i = 0; i < n; i++) { const a = X[2 * i], b = X[2 * i + 1]; g11 += a * a; g12 += a * b; g22 += b * b; r1 += a * Y[i]; r2 += b * Y[i]; }
        G = [g11 / n, g12 / n, g22 / n];
        const det = G[0] * G[2] - G[1] * G[1];
        thHat = [(G[2] * r1 / n - G[1] * r2 / n) / det, (G[0] * r2 / n - G[1] * r1 / n) / det];
        const tr = G[0] + G[2], disc = Math.sqrt((G[0] - G[2]) ** 2 + 4 * G[1] * G[1]);
        mu = (tr - disc) / 2;
        s2 = 0; for (let i = 0; i < n; i++) s2 += (Y[i] - X[2 * i] * thHat[0] - X[2 * i + 1] * thHat[1]) ** 2; s2 /= n;
        reset();
      }
      const F = (a, b) => { let s = 0; for (let i = 0; i < n; i++) s += (Y[i] - X[2 * i] * a - X[2 * i + 1] * b) ** 2; return s / (2 * n); };
      function reset() {
        loop.stop();
        const st = [thHat[0] - 2.4, thHat[1] + 1.8];
        th = [0, 1, 2].map(() => st.slice()); avg = st.slice();
        paths = [0, 1, 2, 3].map(() => [st.slice()]);
        hist = [0, 1, 2, 3].map(() => [[1, dist(st)]]);
        t = 0; nextRec = 1; ema = [0, 0, 0, 0].map(() => dist(st));
        draw();
      }
      const dist = q => (q[0] - thHat[0]) ** 2 + (q[1] - thHat[1]) ** 2;
      function steps(k) {
        const e0 = eta.get(), bs = +B.get(), t0 = 1 / (mu * e0);
        for (let s = 0; s < k; s++) {
          t++;
          const etas = [e0, e0 / Math.sqrt(1 + t / 10), 1 / (mu * (t + t0))];
          // one shared mini-batch per step so the schedules see the same noise
          let g = [[0, 0], [0, 0], [0, 0]];
          for (let b = 0; b < bs; b++) {
            const i = (ctx.rng.u() * n) | 0, a = X[2 * i], c = X[2 * i + 1];
            for (let j = 0; j < 3; j++) { const r = a * th[j][0] + c * th[j][1] - Y[i]; g[j][0] += r * a / bs; g[j][1] += r * c / bs; }
          }
          for (let j = 0; j < 3; j++) { th[j][0] -= etas[j] * g[j][0]; th[j][1] -= etas[j] * g[j][1]; }
          avg[0] += (th[0][0] - avg[0]) / t; avg[1] += (th[0][1] - avg[1]) / t;
          if (t % 5 === 0) { for (let j = 0; j < 3; j++) { paths[j].push(th[j].slice()); if (paths[j].length > 600) paths[j].shift(); } paths[3].push(avg.slice()); if (paths[3].length > 600) paths[3].shift(); }
          [th[0], th[1], th[2], avg].forEach((q, j) => { ema[j] += (dist(q) - ema[j]) / Math.min(t, 1000); });
          if (t >= nextRec) { nextRec = Math.ceil(nextRec * 1.04 + 1); [th[0], th[1], th[2], avg].forEach((q, j) => hist[j].push([t, dist(q)])); }
        }
      }
      function tick(dt) {
        steps(Math.min(TMAX - t, 120));
        acc += dt;
        if (acc > 0.08 || t >= TMAX) { acc = 0; draw(); }
        if (t >= TMAX) return false;
      }
      function draw() {
        const c = thHat, R = 3.2;
        pC.setDomain([c[0] - R, c[0] + R], [c[1] - R * 0.75, c[1] + R * 0.75]).clear();
        const f0 = F(c[0], c[1]);
        pC.contour(F, logspace(0.01, 8, 9).map(v => f0 + v), { color: 'faint', width: 1, nx: 40, ny: 30 });
        SCH.forEach(([, col], j) => pC.line(paths[j], { color: col, width: j === 3 ? 2.2 : 1.1, opacity: j === 3 ? 1 : 0.75 }));
        pC.dot(c[0], c[1], { color: 'ink', r: 4.5 });
        const floor = eta.get() * s2 * d / (2 * +B.get());
        const all = hist.flatMap(h => h.map(q => q[1])).filter(v => v > 0);
        pE.setDomain([1, TMAX], [Math.max(1e-9, Math.min(floor / 10, ...all) / 2), Math.max(...all) * 2]).clear();
        pE.hline(floor, { color: 's1', dash: '6 4', width: 1.2 });
        SCH.forEach(([, col], j) => pE.line(hist[j].map(([a, b]) => [a, pos(b)]), { color: col, width: 1.8 }));
        SCH.forEach(([], j) => { const h = hist[j]; tiles[j].set(ema[j].toExponential(1), `running mean of ‖θ − θ̂‖², t = ${t}`); });
        sFloor.set(floor.toExponential(1), `s² = ${S.fmt(s2, 3)}, μ = ${S.fmt(mu, 3)}`);
      }
      newData();
    },
  });

  /* ================================================================
   * Initialization & normalization
   * ================================================================ */
  Atlas.lab('initialization-normalization-signal', {
    title: 'Signal propagation at initialization',
    blurb: 'A width-64 network, weights $\\N(0, g/n_{\\text{in}})$, no training — just a forward pass of 128 random inputs. Left: the RMS of activations by layer. Right: activation histograms at the first, middle and last layer. Try tanh with $g=4$ (saturation), ReLU with $g=1$ (collapse) vs. $g=2$ (He), then switch on BatchNorm or LayerNorm, or residual connections.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const act = UI.segmented(controls, { label: 'Activation', options: ['tanh', 'relu', 'linear'].map(v => ({ value: v, label: ACTS[v].label })), value: 'relu', onChange: compute });
      const g = UI.slider(controls, { label: 'Init gain $g$ (Var $W_{ij} = g/n_{\\text{in}}$)', min: 0.25, max: 8, log: true, value: 1, fmt: v => S.fmt(v, 3), onChange: compute });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'LeCun / Glorot (g = 1)', () => { g.set(1); compute(); }, { small: true });
      UI.button(row, 'He (g = 2)', () => { g.set(2); compute(); }, { small: true });
      const D = UI.slider(controls, { label: 'Depth $D$', min: 2, max: 50, step: 1, value: 30, onChange: compute });
      const norm = UI.segmented(controls, { label: 'Normalization (before $\\phi$)', options: [{ value: 'none', label: 'None' }, { value: 'bn', label: 'BatchNorm' }, { value: 'ln', label: 'LayerNorm' }], value: 'none', onChange: compute });
      const res = UI.toggle(controls, { label: 'Residual: $h_\\ell = h_{\\ell-1} + \\phi(\\cdot)$', value: false, onChange: compute });
      UI.button(controls, 'New weights', compute);
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 290, x: [0, 30], y: [1e-3, 1e3], logy: true, xlabel: 'layer ℓ', ylabel: 'RMS', title: 'Activation scale by layer' });
      const pH = new Plot(stage, { width: 420, height: 290, x: [-3, 3], y: [0, 1], xlabel: 'activation value', ylabel: 'density', title: 'Activation histograms' });
      const sLast = UI.stat(stats, 'RMS of $h_D$', { accent: 'accent' });
      const sCos = UI.stat(stats, 'Mean cosine between inputs at layer $D$');
      const sDead = UI.stat(stats, 'Saturated / dead units at layer $D$');
      const w = 64, n = 128;
      function compute() {
        const A = ACTS[act.get()], d = Math.round(D.get()), sd = Math.sqrt(g.get() / w), r = ctx.rng, mode = norm.get(), R = res.get();
        let h = new Float64Array(n * w);
        for (let i = 0; i < h.length; i++) h[i] = r.normal();
        const rmsH = [], rmsZ = [], layers = [];
        const keep = new Set([1, Math.max(1, Math.round(d / 2)), d]);
        let lastZ = null;
        for (let l = 1; l <= d; l++) {
          const W = new Float64Array(w * w); for (let i = 0; i < W.length; i++) W[i] = r.normal() * sd;
          const z = new Float64Array(n * w);
          for (let i = 0; i < n; i++) for (let k = 0; k < w; k++) { let s = 0; const ko = k * w, io = i * w; for (let j = 0; j < w; j++) s += W[ko + j] * h[io + j]; z[io + k] = s; }
          rmsZ.push(Math.sqrt(z.reduce((s, v) => s + v * v, 0) / z.length));
          if (mode === 'bn') {
            for (let k = 0; k < w; k++) { let m = 0, v = 0; for (let i = 0; i < n; i++) m += z[i * w + k]; m /= n; for (let i = 0; i < n; i++) v += (z[i * w + k] - m) ** 2; v /= n; const is = 1 / Math.sqrt(v + 1e-5); for (let i = 0; i < n; i++) z[i * w + k] = (z[i * w + k] - m) * is; }
          } else if (mode === 'ln') {
            for (let i = 0; i < n; i++) { let m = 0, v = 0; for (let k = 0; k < w; k++) m += z[i * w + k]; m /= w; for (let k = 0; k < w; k++) v += (z[i * w + k] - m) ** 2; v /= w; const is = 1 / Math.sqrt(v + 1e-5); for (let k = 0; k < w; k++) z[i * w + k] = (z[i * w + k] - m) * is; }
          }
          const hn = new Float64Array(n * w);
          for (let q = 0; q < hn.length; q++) hn[q] = (R ? h[q] : 0) + A.f(z[q]);
          h = hn; lastZ = z;
          rmsH.push(Math.sqrt(h.reduce((s, v) => s + v * v, 0) / h.length));
          if (keep.has(l)) layers.push({ l, vals: Array.from(h) });
        }
        // diagnostics at the last layer
        let cs = 0, cnt = 0;
        for (let i = 0; i + 1 < n; i += 2) {
          let ab = 0, aa = 0, bb = 0;
          for (let k = 0; k < w; k++) { const a = h[i * w + k], b = h[(i + 1) * w + k]; ab += a * b; aa += a * a; bb += b * b; }
          if (aa > 0 && bb > 0) { cs += ab / Math.sqrt(aa * bb); cnt++; }
        }
        let bad = 0;
        if (act.get() === 'relu') { for (let k = 0; k < w; k++) { let alive = false; for (let i = 0; i < n; i++) if (lastZ[i * w + k] > 0) { alive = true; break; } if (!alive) bad++; } }
        else if (act.get() === 'tanh') { let sat = 0; for (let q = 0; q < lastZ.length; q++) if (Math.abs(lastZ[q]) > 2.5) sat++; bad = sat / n; }
        // draw scale plot
        const all = rmsH.concat(rmsZ).filter(v => v > 0 && Number.isFinite(v));
        const lo = Math.max(1e-30, Math.min(...all, 0.5) / 3), hi = Math.max(...all, 2) * 3;
        pS.setDomain([0, d + 1], [lo, hi]).clear();
        pS.hline(1, { color: 'faint', width: 1 });
        pS.line(rmsZ.map((v, i) => [i + 1, pos(v)]), { color: 's3', width: 1.6, dash: '5 4' });
        pS.line(rmsH.map((v, i) => [i + 1, pos(v)]), { color: 's1', width: 2.2 });
        pS.points(rmsH.map((v, i) => [i + 1, pos(v)]), { color: 's1', r: 2.5 });
        pS.legend([{ label: 'RMS h_ℓ (post-activation)', color: 's1' }, { label: 'RMS Wh (pre-norm)', color: 's3', dash: true }], { left: true });
        // histograms
        const ref = layers[0].vals.map(Math.abs).sort((a, b) => a - b);
        let X = Math.max(ref[Math.floor(ref.length * 0.995)], 1e-6);
        for (const L of layers) { const s = L.vals.map(Math.abs).sort((a, b) => a - b); X = Math.max(X, s[Math.floor(s.length * 0.995)]); }
        X *= 1.1;
        const lo2 = act.get() === 'relu' && !R ? -0.05 * X : -X;
        const cols = ['s1', 's2', 's5'];
        const hs = layers.map(L => S.histogram(L.vals, { bins: 60, lo: lo2, hi: X }));
        const dmax = Math.max(...hs[0].map(b => b.density)) * 2.5;
        pH.setDomain([lo2, X], [0, dmax]).clear();
        hs.forEach((hb, j) => pH.line(hb.flatMap(b => [[b.x0, b.density], [b.x1, b.density]]), { color: cols[j], width: 1.8, opacity: 0.9 }));
        pH.legend(layers.map((L, j) => ({ label: `layer ${L.l}`, color: cols[j] })));
        sLast.set(S.fmt(rmsH[d - 1], 3), `layer 1: ${S.fmt(rmsH[0], 3)}`);
        sCos.set(cnt ? S.fmt(cs / cnt, 3) : '—', 'inputs start nearly orthogonal (≈ 0)');
        sDead.set(act.get() === 'relu' ? `${bad} / ${w} dead` : act.get() === 'tanh' ? `${S.fmt(100 * bad / w, 2)}% with |z| > 2.5` : '—');
      }
      compute();
    },
  });

  /* ================================================================
   * Regularization
   * ================================================================ */
  Atlas.lab('regularization-deep-dropout', {
    title: 'Overfitting, dropout and weight decay',
    blurb: 'Forty noisy training points (a fraction of labels flipped) and a wide two-hidden-layer ReLU net. With no regularization the training loss heads to 0 while test loss turns up: the net memorizes the flipped labels. Add dropout or weight decay and compare the test curve; compare L2-in-the-loss with decoupled decay under Adam — they are not the same regularizer, and decoupled decay needs a much larger $\\lambda$ to bite. The dot marks the early-stopping point.',
    mount(el, ctx) { playground(el, ctx, { data: 'moons', reg: true }); },
  });

  Atlas.lab('regularization-deep-double-descent', {
    title: 'Double descent with random ReLU features',
    blurb: 'Fit $n$ noisy points with $\\hat f(x)=\\sum_{j=1}^p w_j\\,\\phi_j(x)$, random features $\\phi_j(x)=(a_j(x-c_j))_+/\\sqrt p$, using the minimum-norm least-squares solution (or ridge). Test error rises as $p\\uparrow n$, spikes at the interpolation threshold $p=n$, then falls again. A little ridge removes the spike. Right: the fitted function at the chosen $p$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nS = UI.segmented(controls, { label: 'Training points $n$', options: ['20', '30', '50'], value: '30', onChange: newData });
      const sd = UI.slider(controls, { label: 'Noise s.d. $\\sigma$', min: 0, max: 0.8, step: 0.05, value: 0.3, onChange: newData });
      const lam = UI.segmented(controls, { label: 'Ridge $\\lambda$', options: [{ value: '0', label: '0 (min-norm)' }, { value: '1e-4', label: '1e-4' }, { value: '1e-2', label: '1e-2' }, { value: '1e-1', label: '1e-1' }], value: '0', onChange: computeCurve });
      const pS = UI.slider(controls, { label: 'Features $p$ (right panel)', min: 1, max: 400, step: 1, value: 30, onInput: drawFit });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'At threshold (p = n)', () => { pS.set(+nS.get()); drawFit(); });
      UI.button(row, 'New data & features', newData, { primary: true });
      stage.classList.add('two');
      const pC = new Plot(stage, { width: 420, height: 290, x: [1, 400], y: [1e-3, 10], logx: true, logy: true, xlabel: 'number of features p', ylabel: 'MSE', title: 'Risk curve (median over 12 feature draws)' });
      const pF = new Plot(stage, { width: 420, height: 290, x: [-1, 1], y: [-2.5, 2.5], xlabel: 'x', title: 'Fit at the chosen p' });
      const sT = UI.stat(stats, 'Test MSE at chosen $p$', { accent: 'accent' });
      const sN = UI.stat(stats, '$\\|\\hat w\\|_2$ at chosen $p$');
      const sB = UI.stat(stats, 'Best test MSE (p)');
      const f = x => Math.sin(Math.PI * x) + 0.35 * Math.cos(4 * x);
      const R = 12, PMAX = 400, xt = S.linspace(-1, 1, 200), ft = xt.map(f);
      let xs, ys, feats, curve;
      function newData() {
        const n = +nS.get(), r = ctx.rng;
        xs = r.array(n, i => -1 + (i + 0.5 + r.uniform(-0.3, 0.3)) * 2 / n); ys = xs.map(x => f(x) + sd.get() * r.normal());
        feats = [];
        for (let k = 0; k < R; k++) feats.push({ a: Float64Array.from({ length: PMAX }, () => r.normal()), c: Float64Array.from({ length: PMAX }, () => r.uniform(-1, 1)) });
        computeCurve();
      }
      function phiMat(F, x, p) {
        const m = x.length, M = new Float64Array(m * p), s = 1 / Math.sqrt(p);
        for (let i = 0; i < m; i++) for (let j = 0; j < p; j++) { const u = F.a[j] * (x[i] - F.c[j]); M[i * p + j] = u > 0 ? u * s : 0; }
        return M;
      }
      function fit(F, p) {
        const n = xs.length, Phi = phiMat(F, xs, p), L = +lam.get();
        let w;
        if (p <= n) {
          const A = new Float64Array(p * p), b = new Float64Array(p);
          for (let j = 0; j < p; j++) { for (let k = 0; k <= j; k++) { let s = 0; for (let i = 0; i < n; i++) s += Phi[i * p + j] * Phi[i * p + k]; A[j * p + k] = A[k * p + j] = s; } let s = 0; for (let i = 0; i < n; i++) s += Phi[i * p + j] * ys[i]; b[j] = s; }
          let tr = 0; for (let j = 0; j < p; j++) tr += A[j * p + j];
          for (let j = 0; j < p; j++) A[j * p + j] += L + 1e-12 * tr / p + 1e-300;
          w = cholSolve(A, b, p);
        } else {
          const K = new Float64Array(n * n);
          for (let i = 0; i < n; i++) for (let k = 0; k <= i; k++) { let s = 0; for (let j = 0; j < p; j++) s += Phi[i * p + j] * Phi[k * p + j]; K[i * n + k] = K[k * n + i] = s; }
          let tr = 0; for (let i = 0; i < n; i++) tr += K[i * n + i];
          for (let i = 0; i < n; i++) K[i * n + i] += L + 1e-12 * tr / n + 1e-300;
          const al = cholSolve(K, ys, n);
          w = new Float64Array(p);
          for (let j = 0; j < p; j++) { let s = 0; for (let i = 0; i < n; i++) s += Phi[i * p + j] * al[i]; w[j] = s; }
        }
        const pred = x => { const P = phiMat(F, x, p), out = new Float64Array(x.length); for (let i = 0; i < x.length; i++) { let s = 0; for (let j = 0; j < p; j++) s += P[i * p + j] * w[j]; out[i] = s; } return out; };
        const pt = pred(xt), pr = pred(xs);
        let te = 0, trn = 0;
        for (let i = 0; i < xt.length; i++) te += (pt[i] - ft[i]) ** 2;
        for (let i = 0; i < n; i++) trn += (pr[i] - ys[i]) ** 2;
        return { te: te / xt.length, tr: trn / n, w, pred, norm: Math.sqrt(w.reduce((s, v) => s + v * v, 0)) };
      }
      function computeCurve() {
        const n = xs.length;
        const ps = [...new Set(logspace(1, PMAX, 44).map(Math.round).concat([n - 2, n - 1, n, n + 1, n + 2]))].filter(p => p >= 1 && p <= PMAX).sort((a, b) => a - b);
        curve = ps.map(p => {
          const rs = feats.map(F => fit(F, p));
          return { p, te: S.median(rs.map(r => r.te)), tr: S.median(rs.map(r => r.tr)) };
        });
        drawCurve();
        drawFit();
      }
      function drawCurve() {
        const n = xs.length;
        const vals = curve.flatMap(c => [c.te, c.tr]).filter(v => v > 0 && Number.isFinite(v));
        pC.setDomain([1, PMAX], [Math.max(1e-6, Math.min(...vals) / 2), Math.min(1e6, Math.max(...vals) * 2)]).clear();
        pC.vline(n, { color: 'muted', width: 1.2 });
        pC.text(n, Math.min(1e6, Math.max(...vals) * 2), 'p = n', { dx: 4, dy: 12, size: 11, color: 'muted' });
        if (sd.get() > 0) pC.hline(sd.get() ** 2, { color: 'faint', width: 1 });
        pC.line(curve.map(c => [c.p, pos(c.tr)]), { color: 's3', width: 1.6, dash: '5 4' });
        pC.line(curve.map(c => [c.p, pos(c.te)]), { color: 's1', width: 2.2 });
        pC.points(curve.map(c => [c.p, pos(c.te)]), { color: 's1', r: 2.2 });
        pC.legend([{ label: 'test (excess) MSE', color: 's1' }, { label: 'train MSE', color: 's3', dash: true }]);
        let best = curve[0]; for (const c of curve) if (c.te < best.te) best = c;
        sB.set(S.fmt(best.te, 3), `at p = ${best.p}`);
      }
      let marker = null;
      function drawFit() {
        const p = Math.round(pS.get()), r = fit(feats[0], p);
        const g = S.linspace(-1, 1, 300), pg = r.pred(g);
        pF.clear();
        pF.line(g.map(x => [x, f(x)]), { color: 'ink', width: 1.8, opacity: 0.55 });
        pF.line(g.map((x, i) => [x, clamp(pg[i], -50, 50)]), { color: 'accent', width: 2 });
        pF.points(xs.map((x, i) => [x, ys[i]]), { color: 'ink', r: 4 });
        pF.points(xs.map((x, i) => [x, ys[i]]), { color: 's2', r: 3 });
        pF.legend([{ label: 'truth f', color: 'ink' }, { label: `fit, p = ${p}`, color: 'accent' }]);
        if (marker) marker.remove();
        marker = pC.vline(p, { color: 'accent', width: 1.5, dash: '2 3' });
        sT.set(S.fmt(r.te, 3), `train MSE ${S.fmt(r.tr, 2)} (one feature draw)`);
        sN.set(S.fmt(r.norm, 3));
      }
      newData();
    },
  });

  Atlas.lab('regularization-deep-early-stopping', {
    title: 'Early stopping is (almost) ridge regression',
    blurb: 'Gradient descent from $0$ on least squares shrinks each eigendirection of $G=X^\\top X/n$ by the filter $1-(1-\\eta s)^t$; ridge uses $s/(s+\\lambda)$. With $\\lambda = 1/(\\eta t)$ the two filters nearly coincide. Left: the filters against the eigenvalue $s$ (dots: the two eigenvalues of this design). Right: the GD path and the ridge path through coefficient space.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const T = UI.slider(controls, { label: 'GD iterations $t$', min: 1, max: 10000, log: true, value: 20, fmt: v => v.toFixed(0), onInput: draw });
      const rho = UI.slider(controls, { label: 'Feature correlation $\\rho$', min: 0, max: 0.995, step: 0.005, value: 0.95, onChange: newData });
      UI.button(controls, 'New data', newData);
      stage.classList.add('two');
      const pF = new Plot(stage, { width: 420, height: 290, x: [1e-4, 10], y: [0, 1.05], logx: true, xlabel: 'eigenvalue s of G', ylabel: 'shrinkage factor', title: 'Spectral filters' });
      const pP = new Plot(stage, { width: 420, height: 290, equal: true, xlabel: 'θ₁', ylabel: 'θ₂', title: 'Regularization paths' });
      const sL = UI.stat(stats, 'Matched $\\lambda = 1/(\\eta t)$', { accent: 'accent' });
      const sD = UI.stat(stats, '$\\|\\theta^{GD}_t - \\theta^{ridge}_\\lambda\\| / \\|\\hat\\theta\\|$');
      const sE = UI.stat(stats, 'Eigenvalues of $G$; step $\\eta$');
      const n = 60;
      let G, r, ev, V, eta, thHat, gdPath, ridgePath;
      function newData() {
        const rr = ctx.rng, p = rho.get();
        let g11 = 0, g12 = 0, g22 = 0, r1 = 0, r2 = 0;
        for (let i = 0; i < n; i++) {
          const z1 = rr.normal(), z2 = rr.normal(), a = z1, b = p * z1 + Math.sqrt(1 - p * p) * z2;
          const y = 1.5 * a + 0.5 * b + 0.8 * rr.normal();
          g11 += a * a; g12 += a * b; g22 += b * b; r1 += a * y; r2 += b * y;
        }
        G = [[g11 / n, g12 / n], [g12 / n, g22 / n]]; r = [r1 / n, r2 / n];
        const E = S.eigSym(G); ev = E.values; V = E.vectors;
        eta = 1 / ev[0];
        thHat = S.solve(G, r);
        gdPath = []; let th = [0, 0];
        const tmax = 10000;
        for (let t = 0; t <= tmax; t++) {
          if (t <= 50 || t % 10 === 0) gdPath.push([t, th[0], th[1]]);
          const g0 = G[0][0] * th[0] + G[0][1] * th[1] - r[0], g1 = G[1][0] * th[0] + G[1][1] * th[1] - r[1];
          th = [th[0] - eta * g0, th[1] - eta * g1];
        }
        ridgePath = logspace(1e4, 1e-5, 200).map(l => ridge(l));
        draw();
      }
      const ridge = l => S.solve([[G[0][0] + l, G[0][1]], [G[1][0], G[1][1] + l]], r);
      // exact GD iterate via the spectral formula
      function gdAt(t) {
        const out = [0, 0];
        for (let k = 0; k < 2; k++) {
          const v = [V[0][k], V[1][k]], c = v[0] * r[0] + v[1] * r[1];
          const fct = (1 - (1 - eta * ev[k]) ** t) / ev[k];
          out[0] += fct * c * v[0]; out[1] += fct * c * v[1];
        }
        return out;
      }
      function draw() {
        const t = T.get(), l = 1 / (eta * t);
        const ss = logspace(1e-4, 10, 300);
        pF.clear();
        pF.line(ss.map(s => [s, eta * s < 2 ? 1 - (1 - eta * s) ** t : NaN]), { color: 's1', width: 2.2 });
        pF.line(ss.map(s => [s, s / (s + l)]), { color: 's2', width: 2, dash: '6 4' });
        pF.vline(l, { color: 'faint', width: 1 });
        ev.forEach(e => { pF.dot(e, 1 - (1 - eta * e) ** t, { color: 's1', r: 4.5 }); pF.dot(e, e / (e + l), { color: 's2', r: 3.5 }); });
        pF.legend([{ label: `GD, t = ${Math.round(t)}`, color: 's1' }, { label: 'ridge, λ = 1/(ηt)', color: 's2', dash: true }], { left: true });
        const gd = gdAt(t), rd = ridge(l);
        const pts = [[0, 0], thHat, ...ridgePath];
        const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
        const pad = 0.25;
        pP.setDomain([Math.min(...xs) - pad, Math.max(...xs) + pad], [Math.min(...ys) - pad, Math.max(...ys) + pad]).clear();
        const Fv = (a, b) => 0.5 * (G[0][0] * a * a + 2 * G[0][1] * a * b + G[1][1] * b * b) - r[0] * a - r[1] * b;
        const fmin = Fv(thHat[0], thHat[1]);
        pP.contour(Fv, logspace(0.002, 3, 9).map(v => fmin + v), { color: 'faint', width: 1, nx: 50, ny: 40 });
        pP.line(ridgePath, { color: 's2', width: 2, dash: '6 4' });
        pP.line(gdPath.map(q => [q[1], q[2]]), { color: 's1', width: 2 });
        pP.dot(thHat[0], thHat[1], { color: 'ink', r: 4.5 });
        pP.text(thHat[0], thHat[1], 'OLS', { dx: 6, dy: -6, size: 11 });
        pP.dot(rd[0], rd[1], { color: 's2', r: 5 });
        pP.dot(gd[0], gd[1], { color: 's1', r: 4 });
        pP.legend([{ label: 'GD path', color: 's1' }, { label: 'ridge path', color: 's2', dash: true }], { left: true });
        const nh = Math.hypot(...thHat);
        sL.set(S.fmt(l, 3));
        sD.set(S.fmt(Math.hypot(gd[0] - rd[0], gd[1] - rd[1]) / nh, 3));
        sE.set(`${S.fmt(ev[0], 3)}, ${S.fmt(ev[1], 3)}`, `η = 1/s_max = ${S.fmt(eta, 3)}`);
      }
      newData();
    },
  });

  /* ================================================================
   * Theory: lazy training / NTK, implicit bias
   * ================================================================ */
  Atlas.lab('deep-learning-theory-lazy', {
    title: 'Lazy training: wide networks barely move',
    blurb: 'Two-layer nets $f(x)=m^{-1/2}\\sum_{j} a_j\\,\\phi(w_jx+b_j)$ of five widths, all trained by full-batch GD on the same 8 points with step $1/\\lambda_{\\max}(K_0)$. Left: the trained net (solid) vs. its linearization at initialization trained for the same number of steps (dashed) — i.e. kernel regression with the empirical NTK $K_0$. Right: relative change in weights and in the NTK; both shrink like $m^{-1/2}$. With ReLU, kinks crossing the data make the agreement visibly rougher at a given width.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const WIDTHS = [16, 64, 256, 1024, 4096];
      const sel = UI.segmented(controls, { label: 'Width shown on the left', options: WIDTHS.map(String), value: '16', onChange: draw });
      const act = UI.segmented(controls, { label: 'Activation', options: [{ value: 'relu', label: 'ReLU' }, { value: 'tanh', label: 'tanh' }], value: 'tanh', onChange: init });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Train', '❚❚ Pause']);
      UI.button(row, 'Re-initialize', init);
      stage.classList.add('two');
      const pF = new Plot(stage, { width: 420, height: 290, x: [-1.2, 1.2], y: [-2.5, 2.5], xlabel: 'x', title: 'Network vs. its linearization' });
      const pW = new Plot(stage, { width: 420, height: 290, x: [10, 6000], y: [1e-3, 2], logx: true, logy: true, xlabel: 'width m', ylabel: 'relative change', title: 'How far training moves things' });
      const sT = UI.stat(stats, 'GD steps $t$');
      const sLs = UI.stat(stats, 'Train loss (shown width)', { accent: 'accent' });
      const sGap = UI.stat(stats, '$\\sup_{|x|\\le 1} |f_t(x) - f^{\\text{lin}}_t(x)|$');
      const sW = UI.stat(stats, '$\\|\\theta_t-\\theta_0\\|/\\|\\theta_0\\|$');
      const n = 8, xs = S.linspace(-0.9, 0.9, n), ys = xs.map(x => 0.9 * Math.sin(2.2 * x) + 0.4 * Math.cos(5 * x));
      const grid = S.linspace(-1.2, 1.2, 121);
      let nets, t, acc = 0;
      const TMAX = 3000;
      function kernelAt(N, xa, xb) { // empirical NTK between two point sets for the current parameters
        const A = ACTS[act.get()], m = N.m, K = [];
        for (let i = 0; i < xa.length; i++) {
          K.push(new Float64Array(xb.length));
          for (let j = 0; j < xb.length; j++) {
            let s = 0;
            for (let k = 0; k < m; k++) {
              const ui = N.w[k] * xa[i] + N.b[k], uj = N.w[k] * xb[j] + N.b[k];
              s += A.f(ui) * A.f(uj) + N.a[k] * N.a[k] * A.d(ui) * A.d(uj) * (xa[i] * xb[j] + 1);
            }
            K[i][j] = s / m;
          }
        }
        return K;
      }
      function out(N, x) { const A = ACTS[act.get()]; let s = 0; for (let k = 0; k < N.m; k++) s += N.a[k] * A.f(N.w[k] * x + N.b[k]); return s / Math.sqrt(N.m); }
      function init() {
        loop.stop();
        const r = ctx.rng;
        nets = WIDTHS.map(m => {
          const N = { m, a: new Float64Array(m), w: new Float64Array(m), b: new Float64Array(m) };
          for (let k = 0; k < m; k++) { N.a[k] = r.normal(); N.w[k] = r.normal(); N.b[k] = r.normal(); }
          N.a0 = N.a.slice(); N.w0 = N.w.slice(); N.b0 = N.b.slice();
          N.norm0 = Math.sqrt([N.a, N.w, N.b].reduce((s, P) => s + P.reduce((q, v) => q + v * v, 0), 0));
          N.K0 = kernelAt(N, xs, xs);
          const E = S.eigSym(N.K0.map(row => Array.from(row)));
          N.lam = E.values; N.V = E.vectors;
          N.eta = 1 / N.lam[0];
          N.f0 = xs.map(x => out(N, x));
          N.r0 = N.f0.map((f, i) => f - ys[i]);
          N.f0g = grid.map(x => out(N, x));
          N.Kg = kernelAt(N, grid, xs);
          return N;
        });
        t = 0;
        draw();
      }
      function gdStep(N) {
        const A = ACTS[act.get()], m = N.m, sm = 1 / Math.sqrt(m);
        const res = new Float64Array(n);
        for (let i = 0; i < n; i++) res[i] = out(N, xs[i]) - ys[i];
        for (let k = 0; k < m; k++) {
          let ga = 0, gw = 0, gb = 0;
          for (let i = 0; i < n; i++) {
            const u = N.w[k] * xs[i] + N.b[k], d = A.d(u) * res[i];
            ga += res[i] * A.f(u); gw += d * xs[i]; gb += d;
          }
          const ak = N.a[k];
          N.a[k] -= N.eta * ga * sm; N.w[k] -= N.eta * ak * gw * sm; N.b[k] -= N.eta * ak * gb * sm;
        }
      }
      // Linearized model trained t steps: f_lin(x) = f0(x) - K(x,X) V diag((1-(1-eta*lam)^t)/lam) V^T r0
      function linPred(N, tt) {
        const c = new Float64Array(n);
        for (let k = 0; k < n; k++) {
          const lk = Math.max(N.lam[k], 1e-14);
          let proj = 0; for (let i = 0; i < n; i++) proj += N.V[i][k] * N.r0[i];
          const ff = -Math.expm1(tt * Math.log1p(-Math.min(N.eta * lk, 1))) / lk;
          for (let i = 0; i < n; i++) c[i] += N.V[i][k] * ff * proj;
        }
        return grid.map((x, gi) => { let s = N.f0g[gi]; for (let i = 0; i < n; i++) s -= N.Kg[gi][i] * c[i]; return s; });
      }
      function tick(dt) {
        for (let s = 0; s < 8 && t < TMAX; s++) { for (const N of nets) gdStep(N); t++; }
        acc += dt;
        if (acc > 0.12 || t >= TMAX) { acc = 0; draw(); }
        if (t >= TMAX) return false;
      }
      function rel(N) {
        let s = 0;
        for (let k = 0; k < N.m; k++) s += (N.a[k] - N.a0[k]) ** 2 + (N.w[k] - N.w0[k]) ** 2 + (N.b[k] - N.b0[k]) ** 2;
        return Math.sqrt(s) / N.norm0;
      }
      function kerRel(N) {
        const K = kernelAt(N, xs, xs);
        let a = 0, b = 0;
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { a += (K[i][j] - N.K0[i][j]) ** 2; b += N.K0[i][j] ** 2; }
        return Math.sqrt(a / b);
      }
      function draw() {
        const N = nets[WIDTHS.indexOf(+sel.get())];
        const fg = grid.map(x => out(N, x)), fl = linPred(N, t);
        pF.clear();
        pF.line(grid.map((x, i) => [x, N.f0g[i]]), { color: 'faint', width: 1.4 });
        pF.line(grid.map((x, i) => [x, fl[i]]), { color: 's2', width: 2.4, dash: '6 4' });
        pF.line(grid.map((x, i) => [x, fg[i]]), { color: 's1', width: 2 });
        pF.points(xs.map((x, i) => [x, ys[i]]), { color: 'ink', r: 4 });
        pF.legend([{ label: 'trained net f_t', color: 's1' }, { label: 'linearized f_t^lin (NTK)', color: 's2', dash: true }, { label: 'initial f_0', color: 'faint' }], { left: true });
        let gap = 0; for (let i = 0; i < grid.length; i++) if (Math.abs(grid[i]) <= 1) gap = Math.max(gap, Math.abs(fg[i] - fl[i]));
        const rw = nets.map(rel), rk = nets.map(kerRel);
        const vals = rw.concat(rk).filter(v => v > 0);
        const lo = vals.length ? Math.min(...vals) / 3 : 1e-4, hi = vals.length ? Math.max(...vals) * 3 : 1;
        pW.setDomain([10, 6000], [Math.min(lo, 1e-3), Math.max(hi, 1e-2)]).clear();
        if (t > 0) {
          pW.line(WIDTHS.map((m, i) => [m, pos(rw[i])]), { color: 's1', width: 2 });
          pW.points(WIDTHS.map((m, i) => [m, pos(rw[i])]), { color: 's1', r: 3.5 });
          pW.line(WIDTHS.map((m, i) => [m, pos(rk[i])]), { color: 's3', width: 2 });
          pW.points(WIDTHS.map((m, i) => [m, pos(rk[i])]), { color: 's3', r: 3.5 });
          const a0 = rw[0] * 1.6;
          pW.line([[16, a0], [4096, a0 * Math.sqrt(16 / 4096)]], { color: 'muted', dash: '4 4', width: 1.2 });
          pW.text(4096, a0 * Math.sqrt(16 / 4096), 'slope −½', { anchor: 'end', dy: -6, size: 10, color: 'muted' });
        }
        pW.legend([{ label: '‖θ_t − θ_0‖ / ‖θ_0‖', color: 's1' }, { label: '‖K_t − K_0‖_F / ‖K_0‖_F', color: 's3' }]);
        let loss = 0; for (let i = 0; i < n; i++) loss += 0.5 * (out(N, xs[i]) - ys[i]) ** 2;
        sT.set(String(t));
        sLs.set(S.fmt(loss, 3), `width ${N.m}`);
        sGap.set(S.fmt(gap, 3));
        sW.set(S.fmt(rel(N), 3), `kernel moved ${S.fmt(kerRel(N), 2)}`);
      }
      init();
    },
  });

  Atlas.lab('deep-learning-theory-implicit-bias', {
    title: 'Gradient descent finds the max-margin direction',
    blurb: 'Logistic (or exponential) loss on linearly separable data has no minimizer: $\\|w_t\\|\\to\\infty$. Yet the direction $w_t/\\|w_t\\|$ converges to the hard-margin SVM direction (Soudry et al., 2018) — very slowly, like $1/\\log t$. Time runs on a log scale here; watch the angle shrink and the normalized margin creep up to the maximum $\\gamma^\\star$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const loss = UI.segmented(controls, { label: 'Loss', options: [{ value: 'log', label: 'Logistic' }, { value: 'exp', label: 'Exponential' }], value: 'log', onChange: reset });
      const eta = UI.slider(controls, { label: 'Step size $\\eta$', min: 0.05, max: 4, log: true, value: 1, fmt: v => S.fmt(v, 2), onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(tick);
      ctx.playButton(row, loop, ['▶ Run GD', '❚❚ Pause']);
      UI.button(row, 'Reset', reset);
      UI.button(row, 'New data', newData);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 300, x: [-2.4, 2.4], y: [-2.4, 2.4], equal: true, title: 'Data, SVM direction (dashed), GD direction' });
      const pA = new Plot(stage, { width: 420, height: 300, x: [1, 2e6], y: [0, 30], logx: true, xlabel: 'iteration t', ylabel: 'angle to SVM direction (deg)', title: 'Directional convergence' });
      const sAng = UI.stat(stats, 'Angle to $u^\\star$', { accent: 'accent' });
      const sMg = UI.stat(stats, 'Normalized margin vs. $\\gamma^\\star$');
      const sNm = UI.stat(stats, '$\\|w_t\\|$ and $\\|w_t\\|/\\log t$');
      const sT = UI.stat(stats, 'Iterations $t$');
      const n = 30;
      let X, Y, uS, gS, w, t, hist;
      const TMAX = 2e6;
      function newData() {
        const r = ctx.rng, th0 = r.uniform(0, 2 * Math.PI), u0 = [Math.cos(th0), Math.sin(th0)];
        X = []; Y = [];
        let guard = 0;
        while (X.length < n && guard++ < 1e5) {
          const a = r.uniform(-2.2, 2.2), b = r.uniform(-2.2, 2.2);
          if (a * a + b * b > 4.8) continue;
          const m = a * u0[0] + b * u0[1];
          if (Math.abs(m) < 0.12) continue;
          X.push([a, b]); Y.push(m > 0 ? 1 : -1);
        }
        // hard-margin SVM through the origin: maximize min_i y_i <u(θ), x_i> over the angle
        const marg = th => Math.min(...X.map((x, i) => Y[i] * (Math.cos(th) * x[0] + Math.sin(th) * x[1])));
        let best = 0, bv = -Infinity;
        for (let k = 0; k < 7200; k++) { const th = 2 * Math.PI * k / 7200, v = marg(th); if (v > bv) { bv = v; best = th; } }
        let a = best - 2 * Math.PI / 7200, b = best + 2 * Math.PI / 7200;
        for (let k = 0; k < 60; k++) { const m1 = a + (b - a) / 3, m2 = b - (b - a) / 3; if (marg(m1) < marg(m2)) a = m1; else b = m2; }
        const th = (a + b) / 2;
        uS = [Math.cos(th), Math.sin(th)]; gS = marg(th);
        reset();
      }
      function reset() { loop.stop(); w = [0, 0]; t = 0; hist = []; draw(); }
      function gdSteps(k) {
        const e = eta.get(), isLog = loss.get() === 'log';
        for (let s = 0; s < k; s++) {
          let g0 = 0, g1 = 0;
          for (let i = 0; i < n; i++) {
            const m = Y[i] * (w[0] * X[i][0] + w[1] * X[i][1]);
            const c = isLog ? sig(-m) : Math.exp(-m);
            g0 -= Y[i] * X[i][0] * c; g1 -= Y[i] * X[i][1] * c;
          }
          w[0] -= e * g0 / n; w[1] -= e * g1 / n;
          t++;
        }
      }
      const angle = () => { const nw = Math.hypot(w[0], w[1]); if (!nw) return NaN; return Math.acos(clamp((w[0] * uS[0] + w[1] * uS[1]) / nw, -1, 1)) * 180 / Math.PI; };
      function tick() {
        const k = clamp(Math.round(t * 0.05), 1, 6000);
        gdSteps(Math.min(k, TMAX - t));
        hist.push([t, angle()]);
        draw();
        if (t >= TMAX) return false;
      }
      function draw() {
        pD.clear();
        drawLine(pD, uS[0], uS[1], 0, { color: 'muted', dash: '6 4', width: 1.5 });
        for (const s of [-1, 1]) drawLine(pD, uS[0], uS[1], -s * gS, { color: 'faint', dash: '2 4', width: 1 });
        X.forEach((x, i) => { if (Y[i] * (x[0] * uS[0] + x[1] * uS[1]) < gS + 1e-6) pD.dot(x[0], x[1], { color: 'warn', r: 7.5, opacity: 0.35 }); });
        pD.points(X, { color: 'ink', r: 4.3 });
        pD.points(X.filter((_, i) => Y[i] < 0), { color: 's1', r: 3.2, opacity: 1 });
        pD.points(X.filter((_, i) => Y[i] > 0), { color: 's2', r: 3.2, opacity: 1 });
        const nw = Math.hypot(w[0], w[1]);
        if (nw > 0) { drawLine(pD, w[0], w[1], 0, { color: 'accent', width: 2.2 }); pD.arrow(0, 0, 0.9 * w[0] / nw, 0.9 * w[1] / nw, { color: 'accent' }); }
        const ymax = Math.max(5, ...hist.map(h => h[1]).filter(Number.isFinite)) * 1.1;
        pA.setDomain([1, TMAX], [0, ymax]).clear();
        pA.line(hist.filter(h => h[0] >= 1), { color: 'accent', width: 2 });
        const ang = angle();
        const mg = nw ? Math.min(...X.map((x, i) => Y[i] * (w[0] * x[0] + w[1] * x[1]))) / nw : NaN;
        sAng.set(Number.isFinite(ang) ? ang.toFixed(2) + '°' : '—');
        sMg.set(Number.isFinite(mg) ? `${S.fmt(mg, 3)} / ${S.fmt(gS, 3)}` : '—');
        sNm.set(S.fmt(nw, 3), t > 1 ? `÷ log t = ${S.fmt(nw / Math.log(t), 3)}` : '');
        sT.set(t >= 1e5 ? t.toExponential(2) : String(t));
      }
      newData();
    },
  });
})();
