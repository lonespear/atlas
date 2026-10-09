/* Labs: Supervised learning I (framework, bias–variance, Bayes classifier, LDA/QDA, naive Bayes,
   k-NN, VC theory, classifier evaluation). */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;

  /* ---------------- shared helpers ---------------- */
  const Phi = x => S.cdf.normal(x);
  const sigm = x => 1 / (1 + Math.exp(-x));
  const clampP = p => Math.min(1 - 1e-12, Math.max(1e-12, p));
  const pct = (x, d = 1) => (Number.isFinite(x) ? (100 * x).toFixed(d) + '%' : '—');
  const deg2rad = d => d * Math.PI / 180;

  // Legendre polynomials P_0..P_d at t in [-1, 1] (well-conditioned polynomial basis).
  function legendre(t, d) {
    const r = [1];
    if (d >= 1) r.push(t);
    for (let k = 2; k <= d; k++) r.push(((2 * k - 1) * t * r[k - 1] - (k - 1) * r[k - 2]) / k);
    return r;
  }
  // Least-squares (optionally ridge) polynomial fit on x in [0,1]; returns x -> fitted value.
  function polyFit(xs, ys, d, lam) {
    const X = xs.map(x => legendre(2 * x - 1, d));
    let b;
    try { b = S.ols(X, ys, { lambda: lam, intercept: true }); }
    catch (e) { b = S.ols(X, ys, { lambda: lam + 1e-4, intercept: true }); }
    return x => S.dot(legendre(2 * x - 1, d), b);
  }
  const TRUTHS = {
    sine: { label: 'Sine wave', f: x => Math.sin(2 * Math.PI * x) },
    quad: { label: 'Gentle parabola', f: x => 1.5 * (2 * x - 1) ** 2 - 0.5 },
    wiggly: { label: 'Wiggly, decaying', f: x => Math.sin(5 * Math.PI * x) * (1 - 0.7 * x) },
  };

  // Region colours: blend from the surface colour toward the predicted class colour.
  function classCols() { return { c0: Atlas.rgb('s1'), c1: Atlas.rgb('s2'), bg: Atlas.rgb('surface') }; }
  function shadeP(C, p) {
    const a = 0.1 + 0.42 * Math.min(1, Math.abs(2 * p - 1));
    const m = Atlas.mix(C.bg, p > 0.5 ? C.c1 : C.c0, a);
    return [m[0], m[1], m[2], 255];
  }

  // 2-D Gaussian with covariance [[a,b],[b,d]].
  function gauss2(mu, c) {
    const a = c[0][0], b = c[0][1], d = c[1][1], det = Math.max(a * d - b * b, 1e-12);
    const i0 = d / det, i1 = -b / det, i2 = a / det, lc = -Math.log(2 * Math.PI) - 0.5 * Math.log(det);
    const L = S.cholesky(c);
    return {
      mu, c,
      logpdf(x, y) { const u = x - mu[0], v = y - mu[1]; return lc - 0.5 * (i0 * u * u + 2 * i1 * u * v + i2 * v * v); },
      sample(r) { const z1 = r.normal(), z2 = r.normal(); return [mu[0] + L[0][0] * z1, mu[1] + L[1][0] * z1 + L[1][1] * z2]; },
    };
  }
  // Mean and scatter matrix of 2-D points.
  function meanScatter(P) {
    const n = P.length, mx = S.mean(P.map(p => p[0])), my = S.mean(P.map(p => p[1]));
    let a = 0, b = 0, d = 0;
    for (const p of P) { const u = p[0] - mx, v = p[1] - my; a += u * u; b += u * v; d += v * v; }
    return { n, mu: [mx, my], W: [[a, b], [b, d]] };
  }
  const scaleM = (M, s) => M.map(r => r.map(v => v * s));
  const addM = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
  // R(θ) diag(s1², s2²) R(θ)ᵀ
  function rotCov(deg, s1, s2) {
    const t = deg2rad(deg), c = Math.cos(t), s = Math.sin(t);
    const off = c * s * (s1 * s1 - s2 * s2);
    return [[c * c * s1 * s1 + s * s * s2 * s2, off], [off, s * s * s1 * s1 + c * c * s2 * s2]];
  }
  // Fit LDA (pooled covariance) or QDA (separate covariances); returns log-odds function (class 1 vs 0).
  function fitDA(X0, X1, quadratic) {
    const A = meanScatter(X0), B = meanScatter(X1), n = A.n + B.n;
    const ridge = [[1e-6, 0], [0, 1e-6]];
    let g0, g1;
    if (quadratic) {
      g0 = gauss2(A.mu, addM(scaleM(A.W, 1 / Math.max(1, A.n - 1)), ridge));
      g1 = gauss2(B.mu, addM(scaleM(B.W, 1 / Math.max(1, B.n - 1)), ridge));
    } else {
      const Sp = addM(scaleM(addM(A.W, B.W), 1 / Math.max(1, n - 2)), ridge);
      g0 = gauss2(A.mu, Sp); g1 = gauss2(B.mu, Sp);
    }
    const lp = Math.log(B.n / A.n);
    return (x, y) => lp + g1.logpdf(x, y) - g0.logpdf(x, y);
  }
  // Mean log loss and error of probabilities p against labels y.
  function scoreProbs(p, y) {
    let ll = 0, err = 0, br = 0;
    for (let i = 0; i < p.length; i++) {
      const q = clampP(p[i]);
      ll -= y[i] ? Math.log(q) : Math.log(1 - q);
      br += (p[i] - y[i]) ** 2;
      if ((p[i] > 0.5 ? 1 : 0) !== y[i]) err++;
    }
    return { ll: ll / p.length, err: err / p.length, brier: br / p.length };
  }
  // k-th smallest (0-based) of a Float64Array prefix of length n (destroys order).
  function selectK(a, n, k) {
    let lo = 0, hi = n - 1;
    while (lo < hi) {
      const piv = a[(lo + hi) >> 1];
      let i = lo, j = hi;
      while (i <= j) {
        while (a[i] < piv) i++;
        while (a[j] > piv) j--;
        if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
      }
      if (k <= j) hi = j; else if (k >= i) lo = i; else return a[k];
    }
    return a[k];
  }

  /* =================================================================== */
  /* learning-framework                                                    */
  /* =================================================================== */
  Atlas.lab('learning-framework-complexity', {
    title: 'Training error vs. test error as complexity grows',
    blurb: 'Polynomials of degree $0$–$12$ are fitted by least squares to many independent training sets from the same world. Average training error falls monotonically with degree; average test error is U-shaped and never drops below the Bayes risk $\\sigma^2$. Shrink $n$ or raise the noise and the best degree moves down.',
    mount(el, ctx) {
      const DMAX = 12;
      const { controls, stage, stats } = UI.layout(el);
      const truth = UI.select(controls, { label: 'True regression function', options: Object.keys(TRUTHS).map(k => ({ value: k, label: TRUTHS[k].label })), value: 'sine', onChange: reset });
      const n = UI.slider(controls, { label: 'Training size $n$', min: 10, max: 200, step: 1, value: 25, onChange: reset });
      const sig = UI.slider(controls, { label: 'Noise sd $\\sigma$', min: 0.05, max: 1, step: 0.05, value: 0.3, onChange: reset });
      const deg = UI.slider(controls, { label: 'Degree shown on the left', min: 0, max: DMAX, step: 1, value: 3, onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ More training sets', '❚❚ Pause']);
      UI.button(row, 'New data', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-2, 2], xlabel: 'x', ylabel: 'y', title: 'Latest training set and its fit' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, DMAX], y: [0.01, 10], logy: true, xlabel: 'polynomial degree', ylabel: 'mean squared error', title: 'Risk averaged over training sets' });
      const sTr = UI.stat(stats, 'Avg training error', { accent: 'accent' });
      const sTe = UI.stat(stats, 'Avg test error');
      const sB = UI.stat(stats, 'Bayes risk $\\sigma^2$');
      const sBest = UI.stat(stats, 'Best degree');
      const sM = UI.stat(stats, 'Training sets');
      const G = S.linspace(0, 1, 121);
      let sets = [], last = null;

      function oneSet() {
        const f = TRUTHS[truth.get()].f, nn = Math.round(n.get()), s = sig.get();
        const xs = ctx.rng.array(nn, () => ctx.rng.u());
        const ys = xs.map(x => f(x) + s * ctx.rng.normal());
        const fG = G.map(f), tr = [], te = [], fits = [];
        for (let d = 0; d <= DMAX; d++) {
          const fh = polyFit(xs, ys, d, 1e-9 * nn);
          let a = 0; for (let i = 0; i < nn; i++) a += (ys[i] - fh(xs[i])) ** 2;
          let b = 0; for (let g = 0; g < G.length; g++) b += (fh(G[g]) - fG[g]) ** 2;
          tr.push(a / nn); te.push(s * s + b / G.length); fits.push(fh);
        }
        sets.push({ tr, te });
        last = { xs, ys, fits };
      }
      function reset() { loop.stop(); sets = []; for (let i = 0; i < 20; i++) oneSet(); draw(); }
      function step() {
        for (let i = 0; i < 6; i++) oneSet();
        draw();
        if (sets.length >= 400) return false;
      }
      function draw() {
        const f = TRUTHS[truth.get()].f, s = sig.get(), d = Math.round(deg.get());
        // left
        const fG = G.map(f), lo = Math.min(...fG) - 3 * s - 0.2, hi = Math.max(...fG) + 3 * s + 0.2;
        pL.setDomain([0, 1], [lo, hi]).clear();
        pL.points(last.xs.map((x, i) => [x, last.ys[i]]), { color: 's1', r: 3, opacity: 0.75 });
        pL.line(G.map((x, i) => [x, fG[i]]), { color: 'ink', width: 1.6, opacity: 0.7 });
        pL.line(S.linspace(0, 1, 300).map(x => [x, last.fits[d](x)]), { color: 's2', width: 2.4 });
        pL.legend([{ label: 'true f', color: 'ink' }, { label: `degree ${d} fit`, color: 's2' }]);
        // right
        const M = sets.length;
        const mTr = [], mTe = [];
        for (let k = 0; k <= DMAX; k++) { mTr.push(S.mean(sets.map(z => z.tr[k]))); mTe.push(S.mean(sets.map(z => z.te[k]))); }
        const ymin = Math.max(1e-4, Math.min(...mTr, s * s) * 0.6);
        const ymax = Math.min(Math.max(...mTe) * 1.5, Math.max(1, 200 * s * s));
        pR.setDomain([0, DMAX], [10 ** Math.floor(Math.log10(ymin)), 10 ** Math.ceil(Math.log10(ymax))]).clear();
        for (const z of sets.slice(0, 15)) pR.line(z.te.map((v, k) => [k, v]), { color: 's2', width: 1, opacity: 0.15 });
        pR.hline(s * s, { color: 'ink', dash: '4 4', width: 1.2 });
        pR.line(mTr.map((v, k) => [k, v]), { color: 's1', width: 2.5 });
        pR.line(mTe.map((v, k) => [k, v]), { color: 's2', width: 2.5 });
        pR.vline(d, { color: 'accent', width: 1.2 });
        pR.legend([{ label: 'training error', color: 's1' }, { label: 'test error', color: 's2' }, { label: 'Bayes risk σ²', color: 'ink', dash: true }], { left: true });
        const best = S.argmin(mTe);
        sTr.set(S.fmt(mTr[d]), `degree ${d}`);
        sTe.set(S.fmt(mTe[d]), `optimism ${S.fmt(mTe[d] - mTr[d])}`);
        sB.set(S.fmt(s * s));
        sBest.set(String(best), `test error ${S.fmt(mTe[best])}`);
        sM.set(String(M));
      }
      reset();
    },
  });

  /* =================================================================== */
  /* bias-variance                                                          */
  /* =================================================================== */
  const BV = {
    poly: { label: 'Polynomial', vals: S.range(13), name: v => `degree ${v}`, short: v => String(v), axis: 'degree  (more complex →)' },
    knn: { label: 'k-NN', vals: [60, 40, 30, 20, 15, 10, 7, 5, 3, 2, 1], name: v => `k = ${v}`, short: v => String(v), axis: 'k  (more complex →)' },
    ridge: { label: 'Ridge, degree 10', vals: [1e2, 1e1, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6], name: v => `λ = 1e${Math.round(Math.log10(v))}`, short: v => `1e${Math.round(Math.log10(v))}`, axis: 'λ  (more complex →)' },
  };
  Atlas.lab('bias-variance-decomposition', {
    title: 'Bias², variance and noise from repeated fits',
    blurb: 'Each run draws a fresh training set and refits. Left: the fan of fits at the chosen complexity (spread = variance), their average (gap to the truth = bias) and the truth. Right: integrated bias², variance, noise and their sum against complexity. Compare polynomial degree, $k$ in $k$-NN, and the ridge penalty $\\lambda$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const meth = UI.segmented(controls, { label: 'Procedure', options: Object.keys(BV).map(k => ({ value: k, label: BV[k].label })), value: 'poly', onChange: () => { cx.input.max = String(BV[meth.get()].vals.length - 1); cx.set(Math.min(cx.get(), BV[meth.get()].vals.length - 1)); reset(); } });
      const truth = UI.select(controls, { label: 'True function', options: Object.keys(TRUTHS).map(k => ({ value: k, label: TRUTHS[k].label })), value: 'sine', onChange: reset });
      const n = UI.slider(controls, { label: 'Training size $n$', min: 15, max: 200, step: 1, value: 40, onChange: reset });
      const sig = UI.slider(controls, { label: 'Noise sd $\\sigma$', min: 0.05, max: 1, step: 0.05, value: 0.4, onChange: reset });
      const cx = UI.slider(controls, { label: 'Complexity shown on the left', min: 0, max: 12, step: 1, value: 3, fmt: v => BV[meth.get()].name(BV[meth.get()].vals[Math.min(Math.round(v), BV[meth.get()].vals.length - 1)]), onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ More training sets', '❚❚ Pause']);
      UI.button(row, 'Restart', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-2, 2], xlabel: 'x', ylabel: 'f̂(x)', title: 'Fits from many training sets' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 12], y: [0, 1], xlabel: 'complexity', ylabel: 'integrated error', title: 'Decomposition vs. complexity' });
      const sBi = UI.stat(stats, 'Bias²', { accent: 'accent' });
      const sVa = UI.stat(stats, 'Variance');
      const sNo = UI.stat(stats, 'Noise $\\sigma^2$');
      const sTo = UI.stat(stats, 'Expected test error');
      const sBest = UI.stat(stats, 'Best complexity');
      const sM = UI.stat(stats, 'Training sets');
      const G = S.linspace(0, 1, 81);
      let sum, sq, M, recent;

      function fitAll(xs, ys) {
        const m = BV[meth.get()], nn = xs.length, out = [];
        if (meth.get() === 'knn') {
          const idx = S.range(nn), dist = new Float64Array(nn);
          const res = m.vals.map(() => new Float64Array(G.length));
          for (let g = 0; g < G.length; g++) {
            for (let i = 0; i < nn; i++) dist[i] = Math.abs(xs[i] - G[g]);
            idx.sort((a, b) => dist[a] - dist[b]);
            let acc = 0, j = 0;
            const ks = m.vals.map(k => Math.min(k, nn));
            const order = ks.map((k, c) => [k, c]).sort((a, b) => a[0] - b[0]);
            for (const [k, c] of order) { while (j < k) acc += ys[idx[j++]]; res[c][g] = acc / k; }
          }
          return res;
        }
        for (const v of m.vals) {
          const fh = meth.get() === 'poly' ? polyFit(xs, ys, v, 1e-9 * nn) : polyFit(xs, ys, 10, v * nn);
          out.push(Float64Array.from(G, fh));
        }
        return out;
      }
      function oneSet() {
        const f = TRUTHS[truth.get()].f, nn = Math.round(n.get()), s = sig.get();
        const xs = ctx.rng.array(nn, () => ctx.rng.u());
        const ys = xs.map(x => f(x) + s * ctx.rng.normal());
        const fits = fitAll(xs, ys);
        fits.forEach((F, c) => { for (let g = 0; g < G.length; g++) { sum[c][g] += F[g]; sq[c][g] += F[g] * F[g]; } });
        M++;
        recent.push(fits);
        if (recent.length > 25) recent.shift();
      }
      function reset() {
        loop.stop();
        const C = BV[meth.get()].vals.length;
        sum = S.range(C).map(() => new Float64Array(G.length));
        sq = S.range(C).map(() => new Float64Array(G.length));
        M = 0; recent = [];
        for (let i = 0; i < 30; i++) oneSet();
        draw();
      }
      function step() { for (let i = 0; i < 4; i++) oneSet(); draw(); if (M >= 500) return false; }
      function decomp() {
        const f = TRUTHS[truth.get()].f, fG = G.map(f), C = sum.length;
        const b2 = [], va = [];
        for (let c = 0; c < C; c++) {
          let bb = 0, vv = 0;
          for (let g = 0; g < G.length; g++) { const mu = sum[c][g] / M; bb += (mu - fG[g]) ** 2; vv += Math.max(0, sq[c][g] / M - mu * mu); }
          b2.push(bb / G.length); va.push(vv / G.length);
        }
        return { b2, va, fG };
      }
      function draw() {
        const m = BV[meth.get()], C = m.vals.length, ci = Math.min(Math.round(cx.get()), C - 1), s = sig.get();
        const { b2, va, fG } = decomp();
        const tot = b2.map((b, c) => b + va[c] + s * s);
        // left: fan
        const lo = Math.min(...fG) - 1.2 - s, hi = Math.max(...fG) + 1.2 + s;
        pL.setDomain([0, 1], [lo, hi]).clear();
        for (const F of recent) pL.line(G.map((x, g) => [x, F[ci][g]]), { color: 's1', width: 1, opacity: 0.28 });
        pL.line(G.map((x, g) => [x, sum[ci][g] / M]), { color: 's2', width: 3 });
        pL.line(G.map((x, g) => [x, fG[g]]), { color: 'ink', width: 1.8, dash: '5 4' });
        pL.legend([{ label: 'individual fits', color: 's1' }, { label: 'average fit f̄', color: 's2' }, { label: 'truth f', color: 'ink', dash: true }]);
        // right: decomposition curves
        const ymax = Math.min(Math.max(...tot), 2.5 * (s * s + b2[0] + 0.05)) * 1.1;
        pR.o.xfmt = v => (Math.abs(v - Math.round(v)) < 1e-9 && v >= 0 && v < C ? m.short(m.vals[Math.round(v)]) : '');
        pR.o.xlabel = m.axis;
        pR.setDomain([-0.3, C - 0.7], [0, ymax]).clear();
        pR.hline(s * s, { color: 'muted', dash: '4 4', width: 1.4 });
        pR.line(b2.map((v, c) => [c, v]), { color: 's1', width: 2.4 });
        pR.line(va.map((v, c) => [c, v]), { color: 's2', width: 2.4 });
        pR.line(tot.map((v, c) => [c, v]), { color: 'ink', width: 2.6 });
        pR.vline(ci, { color: 'accent', width: 1.2 });
        pR.legend([{ label: 'bias²', color: 's1' }, { label: 'variance', color: 's2' }, { label: 'noise σ²', color: 'muted', dash: true }, { label: 'total', color: 'ink' }]);
        const best = S.argmin(tot);
        sBi.set(S.fmt(b2[ci]), m.name(m.vals[ci]));
        sVa.set(S.fmt(va[ci]));
        sNo.set(S.fmt(s * s));
        sTo.set(S.fmt(tot[ci]));
        sBest.set(m.name(m.vals[best]), `test error ${S.fmt(tot[best])}`);
        sM.set(String(M));
      }
      reset();
    },
  });

  /* =================================================================== */
  /* bayes-classifier                                                       */
  /* =================================================================== */
  Atlas.lab('bayes-classifier-gaussians', {
    title: 'The Bayes rule for two Gaussian classes',
    blurb: 'Class 0 is $\\N(0,1)$, class 1 is $\\N(\\mu, s^2)$. Left: cost-weighted densities $c_{FP}\\pi_0 f_0$ and $c_{FN}\\pi_1 f_1$ — the Bayes rule picks the higher curve, and the grey area under the lower envelope is the Bayes risk. Right: the posterior $\\eta(x)$ and the cost threshold. Compare a single-threshold rule with the Bayes rule, especially when $s \\ne 1$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mu = UI.slider(controls, { label: 'Class-1 mean $\\mu$', min: 0, max: 4, step: 0.05, value: 2, onInput: draw });
      const sd = UI.slider(controls, { label: 'Class-1 sd $s$ (class 0 has sd 1)', min: 0.4, max: 2.5, step: 0.05, value: 1, onInput: draw });
      const pi1 = UI.slider(controls, { label: 'Prior $\\pi_1 = \\P(Y=1)$', min: 0.05, max: 0.95, step: 0.01, value: 0.5, onInput: draw });
      const cr = UI.slider(controls, { label: 'Cost ratio $c_{FN}/c_{FP}$', min: 0.1, max: 10, log: true, value: 1, fmt: v => v.toFixed(2), onInput: draw });
      const th = UI.slider(controls, { label: 'Your rule: predict 1 if $x > t$', min: -3, max: 6, step: 0.05, value: 1.5, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-4, 7], y: [0, 0.5], xlabel: 'x', ylabel: 'weighted density', title: 'Cost-weighted class densities' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-4, 7], y: [0, 1], xlabel: 'x', ylabel: 'η(x) = P(Y=1 | x)', title: 'Posterior and decision threshold' });
      const sB = UI.stat(stats, 'Bayes risk $R^*$', { accent: 'accent' });
      const sT = UI.stat(stats, 'Risk of your threshold');
      const sX = UI.stat(stats, 'Excess risk');
      const sBT = UI.stat(stats, 'Best single threshold');
      const note = UI.note(stats, '');
      const W = S.linspace(-15, 20, 5001), dW = W[1] - W[0];
      function draw() {
        const m = mu.get(), s = sd.get(), p1 = pi1.get(), p0 = 1 - p1, r = cr.get();
        const cFP = 2 / (1 + r), cFN = 2 * r / (1 + r);   // normalized so cFP + cFN = 2 (equal costs → error rate)
        const w0 = x => cFP * p0 * S.pdf.normal(x), w1 = x => cFN * p1 * S.pdf.normal(x, m, s);
        let bayes = 0; for (const x of W) bayes += Math.min(w0(x), w1(x)) * dW;
        const riskT = t => cFP * p0 * (1 - Phi(t)) + cFN * p1 * Phi((t - m) / s);
        const t = th.get(), rt = riskT(t);
        let bt = -15, br = Infinity; for (let i = 0; i < W.length; i += 2) { const v = riskT(W[i]); if (v < br) { br = v; bt = W[i]; } }
        // Bayes regions (where class 1 wins)
        const ints = []; let open = null;
        for (const x of W) { const one = w1(x) > w0(x); if (one && open === null) open = x; if (!one && open !== null) { ints.push([open, x]); open = null; } }
        if (open !== null) ints.push([open, Infinity]);
        const fmtE = v => (v === Infinity ? '∞' : v <= -14.99 ? '−∞' : S.fmt(v, 2));
        const desc = ints.length ? ints.map(([a, b]) => `(${fmtE(a)}, ${fmtE(b)})`).join(' ∪ ') : 'nowhere (always predict 0)';
        // left
        const xs = S.linspace(-4, 7, 400);
        const ymax = Math.max(...xs.map(w0), ...xs.map(w1)) * 1.15;
        pL.setDomain([-4, 7], [0, ymax]).clear();
        for (const [a, b] of ints) pL.rect(Math.max(a, -4), 0, Math.min(b, 7), ymax * 0.04, { color: 's2', opacity: 0.6 });
        pL.area(xs.map(x => [x, Math.min(w0(x), w1(x))]), { color: 'ink', opacity: 0.18 });
        pL.line(xs.map(x => [x, w0(x)]), { color: 's1', width: 2.2 });
        pL.line(xs.map(x => [x, w1(x)]), { color: 's2', width: 2.2 });
        pL.vline(t, { color: 'accent', width: 1.8 });
        pL.legend([{ label: 'c_FP π₀ f₀', color: 's1' }, { label: 'c_FN π₁ f₁', color: 's2' }, { label: 'Bayes risk', color: 'ink', fill: true, opacity: 0.18 }, { label: 'your t', color: 'accent' }]);
        // right
        const eta = x => { const a = p1 * S.pdf.normal(x, m, s), b = p0 * S.pdf.normal(x); return a + b > 0 ? a / (a + b) : NaN; };
        const thr = cFP / (cFP + cFN);
        pR.setDomain([-4, 7], [0, 1]).clear();
        for (const [a, b] of ints) pR.rect(Math.max(a, -4), 0, Math.min(b, 7), 1, { color: 's2', opacity: 0.08 });
        pR.hline(thr, { color: 'muted', width: 1.4 });
        pR.line(xs.map(x => [x, eta(x)]), { color: 'ink', width: 2.2 });
        pR.vline(t, { color: 'accent', width: 1.8 });
        pR.text(6.9, thr, `threshold ${thr.toFixed(3)}`, { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        sB.set(S.fmt(bayes, 4));
        sT.set(S.fmt(rt, 4), `t = ${t.toFixed(2)}`);
        sX.set(S.fmt(Math.max(0, rt - bayes), 4));
        sBT.set(S.fmt(br, 4), `at t = ${bt.toFixed(2)}`);
        note.set(`Bayes rule predicts 1 on ${desc}.` + (Math.abs(s - 1) < 1e-9 ? ` With $s=1$: $\\Delta = ${m.toFixed(2)}$ and the closed form gives $R^* = ${S.fmt(bayesClosed(m, p0, p1, cFP, cFN), 4)}$.` : ' With $s \\ne 1$ the log-likelihood ratio is quadratic, so the region can be bounded on both sides.'));
      }
      // Equal-variance closed form, with costs folded into the prior odds: R* = cFN π1 Φ(c/Δ − Δ/2) + cFP π0 Φ(−c/Δ − Δ/2).
      function bayesClosed(D, p0, p1, cFP, cFN) {
        const c = Math.log((cFP * p0) / (cFN * p1));
        if (D < 1e-9) return Math.min(cFP * p0, cFN * p1);
        return cFN * p1 * Phi(c / D - D / 2) + cFP * p0 * Phi(-c / D - D / 2);
      }
      draw();
    },
  });

  const ETAS = {
    steep: { label: 'Crosses ½ steeply (α = 1)', f: x => 0.5 + 0.45 * Math.sin(2 * Math.PI * (x - 0.13)) },
    flat: { label: 'Crosses ½ flatly (α = 1/3)', f: x => { const s = Math.sin(2 * Math.PI * (x - 0.13)); return 0.5 + 0.45 * Math.sign(s) * Math.abs(s) ** 3; } },
    gap: { label: 'Jumps over ½ (α = ∞)', f: x => { const s = Math.sin(2 * Math.PI * (x - 0.13)); return 0.5 + Math.sign(s) * (0.2 + 0.25 * Math.abs(s)); } },
  };
  Atlas.lab('bayes-classifier-plugin', {
    title: 'Plug-in classification: excess risk vs. the $2\\,\\E|\\hat\\eta-\\eta|$ bound',
    blurb: 'Estimate $\\eta$ on $[0,1]$ by a histogram (bin averages of the labels) and threshold at $1/2$. Left: one sample, $\\eta$, $\\hat\\eta$, and (red) where the plug-in rule disagrees with the Bayes rule. Right: across many samples, the true excess risk against the $L^1$ and $L^2$ bounds as $n$ grows. The sharper $\\eta$ crosses $1/2$, the faster classification beats regression.',
    mount(el, ctx) {
      const NG = [25, 50, 100, 200, 400, 800, 1600, 3200, 6400];
      const { controls, stage, stats } = UI.layout(el);
      const eta = UI.select(controls, { label: 'Regression function $\\eta$', options: Object.keys(ETAS).map(k => ({ value: k, label: ETAS[k].label })), value: 'steep', onChange: reset });
      const rule = UI.segmented(controls, { label: 'Number of bins $m$', options: [{ value: 'auto', label: '$m \\approx 2n^{1/3}$' }, { value: 'fixed', label: 'fixed' }], value: 'auto', onChange: reset });
      const mFix = UI.slider(controls, { label: 'Fixed $m$', min: 2, max: 60, step: 1, value: 10, onChange: reset });
      const nShow = UI.slider(controls, { label: 'Sample size on the left', min: 25, max: 3200, log: true, value: 200, fmt: v => String(Math.round(v)), onChange: newLeft });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Repeat samples', '❚❚ Pause']);
      UI.button(row, 'New sample', newLeft);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-0.06, 1.06], xlabel: 'x', ylabel: 'η', title: 'One sample' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [20, 8000], y: [1e-5, 1], logx: true, logy: true, xlabel: 'n', ylabel: 'average over samples', title: 'Excess risk and its bounds' });
      const sEx = UI.stat(stats, 'Excess risk (this sample)', { accent: 'accent' });
      const sL1 = UI.stat(stats, '$2\\,\\E|\\hat\\eta-\\eta|$');
      const sL2 = UI.stat(stats, '$2\\sqrt{\\E(\\hat\\eta-\\eta)^2}$');
      const sR = UI.stat(stats, 'Repetitions');
      const FG = S.range(2000).map(i => (i + 0.5) / 2000);
      let acc, reps, left;
      const bins = nn => (rule.get() === 'auto' ? Math.max(1, Math.round(2 * Math.cbrt(nn))) : Math.round(mFix.get()));
      function sampleFit(nn) {
        const f = ETAS[eta.get()].f, m = bins(nn), cnt = new Float64Array(m), ones = new Float64Array(m), xs = [], ys = [];
        for (let i = 0; i < nn; i++) {
          const x = ctx.rng.u(), y = ctx.rng.u() < f(x) ? 1 : 0;
          const b = Math.min(m - 1, Math.floor(x * m)); cnt[b]++; ones[b] += y;
          if (nn <= 3200) { xs.push(x); ys.push(y); }
        }
        const eh = Array.from(cnt, (c, b) => (c ? ones[b] / c : 0.5));
        let ex = 0, l1 = 0, l2 = 0;
        for (const x of FG) {
          const e = f(x), h = eh[Math.min(m - 1, Math.floor(x * m))];
          if ((h > 0.5) !== (e > 0.5)) ex += Math.abs(2 * e - 1);
          l1 += Math.abs(h - e); l2 += (h - e) ** 2;
        }
        return { m, eh, xs, ys, ex: ex / FG.length, l1: 2 * l1 / FG.length, l2: 2 * Math.sqrt(l2 / FG.length) };
      }
      function reset() {
        loop.stop();
        acc = NG.map(() => ({ ex: 0, l1: 0, l2: 0 })); reps = 0;
        for (let i = 0; i < 5; i++) oneRep();
        newLeft();
      }
      function oneRep() { NG.forEach((nn, j) => { const r = sampleFit(nn); acc[j].ex += r.ex; acc[j].l1 += r.l1; acc[j].l2 += r.l2; }); reps++; }
      function newLeft() { left = sampleFit(Math.round(nShow.get())); draw(); }
      function step() { oneRep(); draw(); if (reps >= 300) return false; }
      function draw() {
        const f = ETAS[eta.get()].f, m = left.m;
        pL.clear();
        const xs = S.linspace(0, 1, 400);
        let inD = null;
        for (const x of xs) {
          const dis = (left.eh[Math.min(m - 1, Math.floor(x * m))] > 0.5) !== (f(x) > 0.5);
          if (dis && inD === null) inD = x;
          if (!dis && inD !== null) { pL.rect(inD, -0.06, x, 1.06, { color: 'bad', opacity: 0.14 }); inD = null; }
        }
        if (inD !== null) pL.rect(inD, -0.06, 1, 1.06, { color: 'bad', opacity: 0.14 });
        const shown = Math.min(left.xs.length, 800);
        pL.points(left.xs.slice(0, shown).map((x, i) => [x, left.ys[i] ? 1.03 - 0.03 * ctx.rng.u() : -0.03 + 0.03 * ctx.rng.u()]), { color: 'muted', r: 1.6, opacity: 0.5 });
        pL.hline(0.5, { color: 'muted', width: 1 });
        pL.line(xs.map(x => [x, f(x)]), { color: 'ink', width: 2 });
        const st = []; for (let b = 0; b < m; b++) st.push([b / m, left.eh[b]]); st.push([1, left.eh[m - 1]]);
        pL.step(st, { color: 's2', width: 2.2 });
        pL.legend([{ label: 'η', color: 'ink' }, { label: `η̂ (${m} bins)`, color: 's2' }, { label: 'ĥ ≠ h*', color: 'bad', fill: true, opacity: 0.25 }]);
        pR.clear();
        const pos = v => (v > 0 ? v : NaN);
        const line = (k, c, dash) => pR.line(NG.map((nn, j) => [nn, pos(acc[j][k] / reps)]), { color: c, width: 2.3, dash });
        line('l2', 's3', '5 4'); line('l1', 's1'); line('ex', 's2');
        for (let j = 0; j < NG.length; j++) { const v = pos(acc[j].ex / reps); if (Number.isFinite(v)) pR.dot(NG[j], v, { color: 's2', r: 3.5 }); }
        pR.legend([{ label: 'excess risk', color: 's2' }, { label: '2E|η̂−η|', color: 's1' }, { label: '2√E(η̂−η)²', color: 's3', dash: true }], { left: true });
        sEx.set(S.fmt(left.ex, 4), `n = ${Math.round(nShow.get())}, m = ${m}`);
        sL1.set(S.fmt(left.l1, 4));
        sL2.set(S.fmt(left.l2, 4));
        sR.set(String(reps));
      }
      reset();
    },
  });

  /* =================================================================== */
  /* lda-qda                                                                */
  /* =================================================================== */
  Atlas.lab('lda-qda-regions', {
    title: 'LDA vs. QDA decision regions',
    blurb: 'Two Gaussian classes in the plane. Shading shows the fitted classifier’s posterior; lines show the fitted LDA boundary, the fitted QDA boundary, and the true Bayes boundary (dashed). Increase the covariance mismatch to make LDA biased; shrink $n$ to make QDA noisy. Test errors use 3000 fresh points.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el, { side: true });
      const mis = UI.slider(controls, { label: 'Covariance mismatch', min: 0, max: 1, step: 0.05, value: 0.6, onChange: newWorld });
      const sep = UI.slider(controls, { label: 'Mean separation', min: 0.5, max: 4, step: 0.1, value: 2, onChange: newWorld });
      const n = UI.slider(controls, { label: 'Training points per class', min: 10, max: 300, step: 5, value: 40, onChange: newSample });
      const shade = UI.segmented(controls, { label: 'Shade regions of', options: ['LDA', 'QDA'], value: 'QDA', onChange: draw });
      const showB = UI.toggle(controls, { label: 'Show Bayes boundary', value: true, onChange: draw });
      UI.button(controls, 'New training sample', newSample, { primary: true });
      const p = new Plot(stage, { width: 640, height: 360, x: [-4.5, 4.5], y: [-3.5, 3.5], equal: true, title: 'Fitted decision regions' });
      const sL = UI.stat(stats, 'LDA test error', { accent: 'accent' });
      const sQ = UI.stat(stats, 'QDA test error');
      const sB = UI.stat(stats, 'Bayes rule test error');
      const sTr = UI.stat(stats, 'Training error (LDA / QDA)');
      let G0, G1, X0, X1, T0, T1, lda, qda, bayes;
      const S0 = rotCov(30, 1.0, 0.45), SA = rotCov(-60, 0.35, 1.6);
      function newWorld() {
        const t = mis.get(), d = sep.get();
        G0 = gauss2([-d / 2, 0], S0);
        G1 = gauss2([d / 2, 0], addM(scaleM(S0, 1 - t), scaleM(SA, t)));
        bayes = (x, y) => G1.logpdf(x, y) - G0.logpdf(x, y);
        const r = ctx.rng;
        T0 = r.array(1500, () => G0.sample(r)); T1 = r.array(1500, () => G1.sample(r));
        newSample();
      }
      function newSample() {
        const nn = Math.round(n.get());
        X0 = ctx.rng.array(nn, () => G0.sample(ctx.rng)); X1 = ctx.rng.array(nn, () => G1.sample(ctx.rng));
        lda = fitDA(X0, X1, false); qda = fitDA(X0, X1, true);
        draw();
      }
      const err = (f, A, B) => (A.filter(q => f(q[0], q[1]) > 0).length + B.filter(q => f(q[0], q[1]) <= 0).length) / (A.length + B.length);
      function draw() {
        const C = classCols(), fs = shade.get() === 'LDA' ? lda : qda;
        p.clear();
        p.raster((x, y) => shadeP(C, sigm(fs(x, y))), { nx: 120, ny: 70 });
        p.ellipse(G0.mu[0], G0.mu[1], G0.c, 2, { color: 's1', width: 1, dash: '3 3', opacity: 0.7 });
        p.ellipse(G1.mu[0], G1.mu[1], G1.c, 2, { color: 's2', width: 1, dash: '3 3', opacity: 0.7 });
        p.points(X0, { color: 's1', r: 3, opacity: 0.85 });
        p.points(X1, { color: 's2', r: 3, opacity: 0.85 });
        p.contour(lda, [0], { color: 's3', width: 2.4, nx: 90, ny: 70 });
        p.contour(qda, [0], { color: 's4', width: 2.4, nx: 90, ny: 70 });
        if (showB.get()) p.contour(bayes, [0], { color: 'ink', width: 1.8, dash: '6 4', nx: 90, ny: 70 });
        p.legend([{ label: 'class 0', color: 's1', fill: true }, { label: 'class 1', color: 's2', fill: true }, { label: 'LDA boundary', color: 's3' }, { label: 'QDA boundary', color: 's4' }, { label: 'Bayes boundary', color: 'ink', dash: true }]);
        const eL = err(lda, T0, T1), eQ = err(qda, T0, T1), eB = err(bayes, T0, T1);
        sL.set(pct(eL), `excess ${pct(eL - eB)}`);
        sQ.set(pct(eQ), `excess ${pct(eQ - eB)}`);
        sB.set(pct(eB), 'estimate of R*');
        sTr.set(`${pct(err(lda, X0, X1))} / ${pct(err(qda, X0, X1))}`, 'LDA uses 3 boundary parameters, QDA 6');
      }
      newWorld();
    },
  });

  Atlas.lab('lda-qda-fisher', {
    title: 'Fisher’s criterion: choosing a projection',
    blurb: 'Two classes share the covariance $\\Sigma = \\begin{pmatrix}1&\\rho\\\\ \\rho&1\\end{pmatrix}$ and differ in mean along the horizontal axis. Rotate the projection direction $a$ and watch the projected classes (right). Fisher’s criterion $J(a) = (a^\\top\\delta)^2/a^\\top\\Sigma a$ is maximized by $\\Sigma^{-1}\\delta$, and the best error on the projection is $\\Phi(-\\sqrt{J}/2)$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rho = UI.slider(controls, { label: 'Within-class correlation $\\rho$', min: -0.95, max: 0.95, step: 0.05, value: 0.8, onChange: newSample });
      const dd = UI.slider(controls, { label: 'Mean difference $\\|\\delta\\|$', min: 0.3, max: 3, step: 0.1, value: 1.2, onChange: newSample });
      const ang = UI.slider(controls, { label: 'Projection angle (degrees)', min: 0, max: 179, step: 1, value: 0, onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Snap to Fisher', () => { ang.set(fisherAngle()); draw(); }, { primary: true });
      UI.button(row, 'New sample', newSample);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [-3, 3], equal: true, title: 'Data and directions' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [0, 1], xlabel: 'projection aᵀx', ylabel: 'density', title: 'Projected classes' });
      const sJ = UI.stat(stats, '$J(a)$ for your direction', { accent: 'accent' });
      const sE = UI.stat(stats, 'Best error on your projection');
      const sF = UI.stat(stats, 'Fisher: max $J = \\Delta^2$');
      const sM = UI.stat(stats, 'Mean-difference direction');
      let X0, X1, Sig, d;
      const unit = deg => [Math.cos(deg2rad(deg)), Math.sin(deg2rad(deg))];
      const quad = (a, M) => a[0] * a[0] * M[0][0] + 2 * a[0] * a[1] * M[0][1] + a[1] * a[1] * M[1][1];
      const Jof = a => (a[0] * d) ** 2 / quad(a, Sig);
      function fisherAngle() { const r = rho.get(); let t = Math.atan2(-r, 1) * 180 / Math.PI; if (t < 0) t += 180; return Math.round(t); }
      function newSample() {
        const r = rho.get(); d = dd.get();
        Sig = [[1, r], [r, 1]];
        const g0 = gauss2([-d / 2, 0], Sig), g1 = gauss2([d / 2, 0], Sig);
        X0 = ctx.rng.array(150, () => g0.sample(ctx.rng)); X1 = ctx.rng.array(150, () => g1.sample(ctx.rng));
        draw();
      }
      function draw() {
        const a = unit(ang.get()), r = rho.get();
        pL.clear();
        pL.ellipse(-d / 2, 0, Sig, 2, { color: 's1', width: 1.2, opacity: 0.8 });
        pL.ellipse(d / 2, 0, Sig, 2, { color: 's2', width: 1.2, opacity: 0.8 });
        pL.points(X0, { color: 's1', r: 2.4, opacity: 0.7 });
        pL.points(X1, { color: 's2', r: 2.4, opacity: 0.7 });
        const fa = unit(fisherAngle());
        // sample Fisher direction from pooled within-class scatter
        const A = meanScatter(X0), B = meanScatter(X1), Wm = addM(A.W, B.W);
        const sf = S.solve(Wm, [B.mu[0] - A.mu[0], B.mu[1] - A.mu[1]]), sn = Math.hypot(sf[0], sf[1]);
        const L = 6;
        pL.segment(-L, 0, L, 0, { color: 'muted', width: 1.2, dash: '2 4' });
        pL.segment(-L * fa[0], -L * fa[1], L * fa[0], L * fa[1], { color: 's3', width: 1.8, dash: '6 4' });
        pL.segment(-L * sf[0] / sn, -L * sf[1] / sn, L * sf[0] / sn, L * sf[1] / sn, { color: 's3', width: 1, opacity: 0.6 });
        pL.segment(-L * a[0], -L * a[1], L * a[0], L * a[1], { color: 'accent', width: 2.4 });
        pL.legend([{ label: 'your a', color: 'accent' }, { label: 'Fisher Σ⁻¹δ', color: 's3', dash: true }, { label: 'sample Fisher', color: 's3', width: 1 }, { label: 'mean difference', color: 'muted', dash: true }]);
        // right: projections
        const z0 = X0.map(q => a[0] * q[0] + a[1] * q[1]), z1 = X1.map(q => a[0] * q[0] + a[1] * q[1]);
        const sdp = Math.sqrt(quad(a, Sig)), m0 = -a[0] * d / 2, m1 = a[0] * d / 2;
        const lo = Math.min(m0, m1) - 3.5 * sdp, hi = Math.max(m0, m1) + 3.5 * sdp;
        const h0 = S.histogram(z0, { bins: 28, lo, hi }), h1 = S.histogram(z1, { bins: 28, lo, hi });
        const ymax = Math.max(...h0.map(b => b.density), ...h1.map(b => b.density), S.pdf.normal(0, 0, sdp)) * 1.15;
        pR.setDomain([lo, hi], [0, ymax]).clear();
        pR.bars(h0, { color: 's1', opacity: 0.35 });
        pR.bars(h1, { color: 's2', opacity: 0.35 });
        const gz = S.linspace(lo, hi, 200);
        pR.line(gz.map(z => [z, S.pdf.normal(z, m0, sdp)]), { color: 's1', width: 2 });
        pR.line(gz.map(z => [z, S.pdf.normal(z, m1, sdp)]), { color: 's2', width: 2 });
        pR.vline(0, { color: 'ink', width: 1.2 });
        const J = Jof(a), Jmax = d * d / (1 - r * r), Jm = Jof([1, 0]);
        sJ.set(S.fmt(J), `angle ${Math.round(ang.get())}°`);
        sE.set(pct(Phi(-Math.sqrt(J) / 2)), 'midpoint threshold');
        sF.set(S.fmt(Jmax), `error ${pct(Phi(-Math.sqrt(Jmax) / 2))} at ${fisherAngle()}°`);
        sM.set(`J = ${S.fmt(Jm)}`, `error ${pct(Phi(-Math.sqrt(Jm) / 2))}`);
      }
      newSample();
    },
  });

  /* =================================================================== */
  /* naive-bayes                                                            */
  /* =================================================================== */
  Atlas.lab('naive-bayes-correlation', {
    title: 'Naive Bayes with correlated features',
    blurb: 'Two Gaussian classes share $\\Sigma$ with correlation $\\rho$. Gaussian naive Bayes ignores $\\rho$; LDA estimates it. With the mean difference along the diagonal (45°), the naive boundary stays close to the Bayes boundary as $\\rho \\to 1$ while its probabilities become overconfident (right panel, far from the diagonal) and its log loss climbs. Rotate the mean difference to make the boundary itself wrong.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rho = UI.slider(controls, { label: 'Correlation $\\rho$', min: 0, max: 0.98, step: 0.02, value: 0.8, onChange: newWorld });
      const phi = UI.slider(controls, { label: 'Mean-difference angle (degrees)', min: 0, max: 90, step: 5, value: 45, onChange: newWorld });
      const sep = UI.slider(controls, { label: 'Separation $\\|\\mu_1-\\mu_0\\|$', min: 0.5, max: 3, step: 0.1, value: 1.5, onChange: newWorld });
      const n = UI.slider(controls, { label: 'Training points per class', min: 20, max: 500, step: 10, value: 100, onChange: newSample });
      UI.button(controls, 'New training sample', newSample, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [-3, 3], equal: true, title: 'Naive Bayes regions' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 1], xlabel: 'true P(Y=1 | x)', ylabel: 'predicted', title: 'Predicted vs. true posterior (test points)' });
      const sE = UI.stat(stats, 'Test error: NB / LDA / Bayes', { accent: 'accent' });
      const sLL = UI.stat(stats, 'Log loss: NB / LDA / true');
      const sBr = UI.stat(stats, 'Brier: NB / LDA / true');
      let G0, G1, truth, X0, X1, T, Ty, nb, lda;
      function newWorld() {
        const r = rho.get(), a = deg2rad(phi.get()), s = sep.get();
        const Sig = [[1, r], [r, 1]], u = [Math.cos(a) * s / 2, Math.sin(a) * s / 2];
        G0 = gauss2([-u[0], -u[1]], Sig); G1 = gauss2(u, Sig);
        truth = (x, y) => G1.logpdf(x, y) - G0.logpdf(x, y);
        const rr = ctx.rng;
        T = []; Ty = [];
        for (let i = 0; i < 1500; i++) { T.push(G0.sample(rr)); Ty.push(0); T.push(G1.sample(rr)); Ty.push(1); }
        newSample();
      }
      function fitNB(A, B) {
        const stats1 = P => [0, 1].map(j => { const v = P.map(q => q[j]); return [S.mean(v), Math.max(1e-6, S.variance(v))]; });
        const a = stats1(A), b = stats1(B), lp = Math.log(B.length / A.length);
        const ln = (x, m, v) => -0.5 * Math.log(2 * Math.PI * v) - (x - m) ** 2 / (2 * v);
        return (x, y) => lp + ln(x, b[0][0], b[0][1]) + ln(y, b[1][0], b[1][1]) - ln(x, a[0][0], a[0][1]) - ln(y, a[1][0], a[1][1]);
      }
      function newSample() {
        const nn = Math.round(n.get());
        X0 = ctx.rng.array(nn, () => G0.sample(ctx.rng)); X1 = ctx.rng.array(nn, () => G1.sample(ctx.rng));
        nb = fitNB(X0, X1); lda = fitDA(X0, X1, false);
        draw();
      }
      function draw() {
        const C = classCols();
        pL.clear();
        pL.raster((x, y) => shadeP(C, sigm(nb(x, y))), { nx: 100, ny: 64 });
        pL.points(X0.slice(0, 150), { color: 's1', r: 2.4, opacity: 0.8 });
        pL.points(X1.slice(0, 150), { color: 's2', r: 2.4, opacity: 0.8 });
        pL.contour(nb, [0], { color: 's3', width: 2.4 });
        pL.contour(lda, [0], { color: 's4', width: 2 });
        pL.contour(truth, [0], { color: 'ink', width: 1.6, dash: '6 4' });
        pL.legend([{ label: 'naive Bayes', color: 's3' }, { label: 'LDA', color: 's4' }, { label: 'Bayes', color: 'ink', dash: true }]);
        const pt = T.map(q => sigm(truth(q[0], q[1]))), pn = T.map(q => sigm(nb(q[0], q[1]))), pl = T.map(q => sigm(lda(q[0], q[1])));
        pR.clear();
        pR.segment(0, 0, 1, 1, { color: 'ink', width: 1.2, dash: '4 4' });
        pR.points(pt.slice(0, 700).map((v, i) => [v, pl[i]]), { color: 's4', r: 1.8, opacity: 0.45 });
        pR.points(pt.slice(0, 700).map((v, i) => [v, pn[i]]), { color: 's3', r: 2, opacity: 0.6 });
        pR.legend([{ label: 'naive Bayes', color: 's3' }, { label: 'LDA', color: 's4' }], { left: true });
        const a = scoreProbs(pn, Ty), b = scoreProbs(pl, Ty), c = scoreProbs(pt, Ty);
        sE.set(`${pct(a.err)} / ${pct(b.err)} / ${pct(c.err)}`);
        sLL.set(`${a.ll.toFixed(3)} / ${b.ll.toFixed(3)} / ${c.ll.toFixed(3)}`, 'lower is better');
        sBr.set(`${a.brier.toFixed(3)} / ${b.brier.toFixed(3)} / ${c.brier.toFixed(3)}`);
      }
      newWorld();
    },
  });

  Atlas.lab('naive-bayes-smoothing', {
    title: 'Laplace smoothing: error rate vs. log loss',
    blurb: 'A Bernoulli naive Bayes model with 60 binary features (the true model is itself naive Bayes, so only estimation matters). Curves show test error (left) and test log loss (right) as the smoothing constant $\\alpha$ varies, for the current training sample; dashed lines are the unsmoothed MLE. Small $n$ makes zero counts common: unsmoothed vetoes turn many decisions into coin flips, and the log loss becomes infinite.',
    mount(el, ctx) {
      const P = 60, AL = S.range(25).map(i => 10 ** (-3 + 4 * i / 24));
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Training examples per class', min: 5, max: 300, step: 1, value: 15, onChange: newSample });
      const al = UI.slider(controls, { label: 'Smoothing $\\alpha$', min: 0.001, max: 10, log: true, value: 1, fmt: v => S.fmt(v, 3), onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New training sample', newSample, { primary: true });
      UI.button(row, 'New world', newWorld);
      stage.classList.add('two');
      const pE = new Plot(stage, { width: 420, height: 270, x: [1e-3, 10], y: [0, 0.5], logx: true, xlabel: 'α', ylabel: 'test error', title: 'Classification error' });
      const pL = new Plot(stage, { width: 420, height: 270, x: [1e-3, 10], y: [0, 2], logx: true, xlabel: 'α', ylabel: 'test log loss', title: 'Log loss' });
      const sE = UI.stat(stats, 'Error at your α', { accent: 'accent' });
      const sLL = UI.stat(stats, 'Log loss at your α');
      const sM = UI.stat(stats, 'Unsmoothed MLE (α = 0)');
      const sB = UI.stat(stats, 'True model');
      let t0, t1, TX, Ty, c0, c1, nn, curve, mle, best;
      function sampleX(r, th) { const x = new Uint8Array(P); for (let j = 0; j < P; j++) x[j] = r.u() < th[j] ? 1 : 0; return x; }
      function newWorld() {
        const r = ctx.rng;
        t0 = r.array(P, () => Math.min(0.9, r.beta(0.6, 3)));
        t1 = t0.map(t => Math.min(0.95, Math.max(0.003, t * Math.exp(0.5 * r.normal()))));
        TX = []; Ty = [];
        for (let i = 0; i < 1500; i++) { TX.push(sampleX(r, t0)); Ty.push(0); TX.push(sampleX(r, t1)); Ty.push(1); }
        const lr = TX.map(x => { let s = 0; for (let j = 0; j < P; j++) s += x[j] ? Math.log(t1[j] / t0[j]) : Math.log((1 - t1[j]) / (1 - t0[j])); return s; });
        best = scoreProbs(lr.map(sigm), Ty);
        newSample();
      }
      function newSample() {
        nn = Math.round(n.get());
        c0 = new Float64Array(P); c1 = new Float64Array(P);
        for (let i = 0; i < nn; i++) { const a = sampleX(ctx.rng, t0), b = sampleX(ctx.rng, t1); for (let j = 0; j < P; j++) { c0[j] += a[j]; c1[j] += b[j]; } }
        curve = AL.map(evalAlpha);
        mle = evalAlpha(0);
        draw();
      }
      // Evaluate smoothed NB on the test set; alpha = 0 is the MLE with -Infinity log-probabilities allowed.
      function evalAlpha(a) {
        const w1 = [], w0 = [];
        for (let j = 0; j < P; j++) {
          const p1 = (c1[j] + a) / (nn + 2 * a), p0 = (c0[j] + a) / (nn + 2 * a);
          w1.push([Math.log(p1), Math.log(1 - p1)]); w0.push([Math.log(p0), Math.log(1 - p0)]);
        }
        let err = 0, ll = 0, veto = 0;
        for (let i = 0; i < TX.length; i++) {
          let s1 = 0, s0 = 0; const x = TX[i];
          for (let j = 0; j < P; j++) { s1 += x[j] ? w1[j][0] : w1[j][1]; s0 += x[j] ? w0[j][0] : w0[j][1]; }
          let p;
          if (s1 === -Infinity && s0 === -Infinity) { p = 0.5; veto++; }
          else if (s1 === -Infinity) { p = 0; veto++; }
          else if (s0 === -Infinity) { p = 1; veto++; }
          else p = sigm(s1 - s0);
          if ((p > 0.5 ? 1 : 0) !== Ty[i]) err++;
          const q = Ty[i] ? p : 1 - p;
          ll += q > 0 ? -Math.log(q) : Infinity;
        }
        return { err: err / TX.length, ll: ll / TX.length, veto: veto / TX.length };
      }
      function draw() {
        const a = al.get(), cur = evalAlpha(a);
        const emax = Math.max(0.5, ...curve.map(c => c.err), mle.err) * 1.05;
        pE.setDomain([1e-3, 10], [0, Math.min(1, emax)]).clear();
        pE.hline(best.err, { color: 'ink', dash: '2 3', width: 1.2 });
        pE.hline(mle.err, { color: 'bad', dash: '6 4', width: 1.4 });
        pE.line(AL.map((x, i) => [x, curve[i].err]), { color: 's1', width: 2.4 });
        pE.vline(a, { color: 'accent', width: 1.6 });
        pE.legend([{ label: 'smoothed NB', color: 's1' }, { label: 'MLE (α = 0)', color: 'bad', dash: true }, { label: 'true model', color: 'ink', dash: true }], { left: true });
        const fin = curve.map(c => c.ll).filter(Number.isFinite);
        const lmax = Math.min(Math.max(...fin, best.ll) * 1.15, 5);
        pL.setDomain([1e-3, 10], [0, lmax]).clear();
        pL.hline(best.ll, { color: 'ink', dash: '2 3', width: 1.2 });
        if (Number.isFinite(mle.ll)) pL.hline(mle.ll, { color: 'bad', dash: '6 4', width: 1.4 });
        else pL.text(10, lmax * 0.93, 'MLE log loss = ∞', { anchor: 'end', dx: -6, color: 'bad', size: 12, weight: 600 });
        pL.line(AL.map((x, i) => [x, curve[i].ll]), { color: 's2', width: 2.4 });
        pL.vline(a, { color: 'accent', width: 1.6 });
        sE.set(pct(cur.err), `α = ${S.fmt(a, 3)}`);
        sLL.set(S.fmt(cur.ll));
        sM.set(`${pct(mle.err)} / ${Number.isFinite(mle.ll) ? S.fmt(mle.ll) : '∞'}`, `error / log loss; ${pct(mle.veto)} of test points hit a zero`);
        sB.set(`${pct(best.err)} / ${S.fmt(best.ll)}`, 'Bayes error / log loss');
      }
      newWorld();
    },
  });

  /* =================================================================== */
  /* knn                                                                    */
  /* =================================================================== */
  // Shared 2-D world: each class is an equal mixture of 5 Gaussian blobs.
  const KW = (() => {
    const r = S.rng(7), s = 0.5;
    const C0 = r.array(5, () => [r.normal(-0.8, 0.9), r.normal(0.4, 0.9)]);
    const C1 = r.array(5, () => [r.normal(0.8, 0.9), r.normal(-0.4, 0.9)]);
    const k2 = 1 / (2 * s * s);
    const mix = (C, x, y) => { let t = 0; for (const c of C) t += Math.exp(-((x - c[0]) ** 2 + (y - c[1]) ** 2) * k2); return t; };
    const eta = (x, y) => { const a = mix(C1, x, y), b = mix(C0, x, y); return a + b > 0 ? a / (a + b) : 0.5; };
    const sample = (rr, k) => { const c = rr.choice(k ? C1 : C0); return [c[0] + s * rr.normal(), c[1] + s * rr.normal()]; };
    let bayes = null;
    const bayesErr = () => {
      if (bayes == null) { const rr = S.rng(99); let e = 0; const N = 20000; for (let i = 0; i < N; i++) { const q = sample(rr, i & 1); const h = eta(q[0], q[1]); e += Math.min(h, 1 - h); } bayes = e / N; }
      return bayes;
    };
    return { eta, sample, bayesErr };
  })();

  Atlas.lab('knn-regions', {
    title: 'k-NN decision regions',
    blurb: 'Each class is a mixture of five Gaussian blobs, so the Bayes boundary (dashed) is known. Shading is the $k$-NN vote fraction. Click on the plot to add a training point of the selected class — with $k=1$ a single point flips a whole Voronoi cell. Training error counts each point as its own neighbour.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el, { side: true });
      const n = UI.slider(controls, { label: 'Training points per class', min: 10, max: 200, step: 5, value: 60, onChange: newSample });
      const k = UI.slider(controls, { label: '$k$ (neighbours)', min: 1, max: 99, step: 2, value: 1, onInput: draw });
      const add = UI.segmented(controls, { label: 'Click adds a point of', options: [{ value: 0, label: 'class 0' }, { value: 1, label: 'class 1' }], value: 1 });
      const showB = UI.toggle(controls, { label: 'Show Bayes boundary', value: true, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newSample, { primary: true });
      UI.button(row, 'Remove clicked points', () => { extra = []; rebuild(); });
      const p = new Plot(stage, { width: 640, height: 360, x: [-4, 4], y: [-3, 3], equal: true, title: 'k-NN vote fraction' });
      const sTe = UI.stat(stats, 'Test error', { accent: 'accent' });
      const sTr = UI.stat(stats, 'Training error');
      const sB = UI.stat(stats, 'Bayes error $R^*$');
      const sCH = UI.stat(stats, 'Cover–Hart bound $2R^*(1-R^*)$');
      let base = [], extra = [], pts = [], buf, tmp;
      const r0 = ctx.rng;
      const test = []; for (let i = 0; i < 1000; i++) { test.push([...KW.sample(r0, 0), 0]); test.push([...KW.sample(r0, 1), 1]); }
      function newSample() {
        const nn = Math.round(n.get());
        base = [];
        for (let i = 0; i < nn; i++) { base.push([...KW.sample(ctx.rng, 0), 0]); base.push([...KW.sample(ctx.rng, 1), 1]); }
        rebuild();
      }
      function rebuild() { pts = base.concat(extra); buf = new Float64Array(pts.length); tmp = new Float64Array(pts.length); draw(); }
      function frac(x, y, kk) {
        const N = pts.length; kk = Math.min(kk, N);
        for (let i = 0; i < N; i++) { const dx = pts[i][0] - x, dy = pts[i][1] - y; buf[i] = dx * dx + dy * dy; tmp[i] = buf[i]; }
        const kth = selectK(tmp, N, kk - 1);
        let c = 0, cnt = 0;
        for (let i = 0; i < N; i++) if (buf[i] < kth) { cnt++; c += pts[i][2]; }
        for (let i = 0; i < N && cnt < kk; i++) if (buf[i] === kth) { cnt++; c += pts[i][2]; }
        return c / kk;
      }
      function errOn(set, kk) { let e = 0; for (const q of set) { const f = frac(q[0], q[1], kk); const yh = f > 0.5 ? 1 : f < 0.5 ? 0 : 0.5; e += yh === 0.5 ? 0.5 : (yh !== q[2] ? 1 : 0); } return e / set.length; }
      function draw() {
        const kk = Math.round(k.get()), C = classCols();
        p.clear();
        p.raster((x, y) => shadeP(C, frac(x, y, kk)), { nx: 110, ny: 64 });
        if (showB.get()) p.contour((x, y) => KW.eta(x, y) - 0.5, [0], { color: 'ink', width: 1.8, dash: '6 4', nx: 90, ny: 70 });
        p.points(pts.filter(q => !q[2]).map(q => [q[0], q[1]]), { color: 's1', r: 3, opacity: 0.95 });
        p.points(pts.filter(q => q[2]).map(q => [q[0], q[1]]), { color: 's2', r: 3, opacity: 0.95 });
        for (const q of extra) p.dot(q[0], q[1], { color: q[2] ? 's2' : 's1', r: 5.5, stroke: 'ink', strokeWidth: 1.8 });
        const R = KW.bayesErr();
        sTe.set(pct(errOn(test, kk)), `k = ${kk}, n = ${pts.length}`);
        sTr.set(pct(errOn(pts, kk)), 'self counted as a neighbour');
        sB.set(pct(R));
        sCH.set(pct(2 * R * (1 - R)), 'asymptotic 1-NN error is below this');
      }
      p.onPointer((x, y, e, type) => {
        if (type !== 'down' || !p.inside(x, y)) return;
        extra.push([x, y, +add.get()]);
        rebuild();
      });
      newSample();
    },
  });

  Atlas.lab('knn-error-curve', {
    title: 'Error vs. $k$, and the curse of dimensionality',
    blurb: 'Training and test error of $k$-NN as a function of $k$ (log scale), averaged over repeated training sets from the five-blob world. Add pure-noise dimensions: the signal is unchanged (the Bayes error stays put), but neighbours stop being informative and the whole test curve rises.',
    mount(el, ctx) {
      const KS = [1, 3, 5, 7, 9, 13, 17, 21, 31, 41, 51, 71, 91, 121, 151];
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Training points per class', min: 25, max: 250, step: 5, value: 100, onChange: reset });
      const D = UI.slider(controls, { label: 'Added noise dimensions', min: 0, max: 20, step: 1, value: 0, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Average more runs', '❚❚ Pause']);
      UI.button(row, 'Restart', reset);
      const p = new Plot(stage, { width: 640, height: 300, x: [1, 200], y: [0, 0.5], logx: true, xlabel: 'k (neighbours)', ylabel: 'error rate', title: 'k-NN error, averaged over training sets' });
      const sBest = UI.stat(stats, 'Best $k$ (test)', { accent: 'accent' });
      const s1 = UI.stat(stats, '1-NN test error');
      const sB = UI.stat(stats, 'Bayes error $R^*$');
      const sCH = UI.stat(stats, '$2R^*(1-R^*)$');
      const sR = UI.stat(stats, 'Runs');
      let te, tr, runs;
      function draws(r, k, d) { const q = KW.sample(r, k); for (let j = 0; j < d; j++) q.push(r.normal()); return q; }
      function oneRun() {
        const nn = Math.round(n.get()), d = Math.round(D.get()), N = 2 * nn;
        const X = [], Y = [];
        for (let i = 0; i < nn; i++) { X.push(draws(ctx.rng, 0, d)); Y.push(0); X.push(draws(ctx.rng, 1, d)); Y.push(1); }
        const TX = [], TY = [];
        for (let i = 0; i < 300; i++) { TX.push(draws(ctx.rng, 0, d)); TY.push(0); TX.push(draws(ctx.rng, 1, d)); TY.push(1); }
        const ks = KS.filter(kk => kk <= N);
        const errFor = (Q, QY) => {
          const e = new Float64Array(ks.length), dist = new Float64Array(N), idx = new Int32Array(N);
          for (let q = 0; q < Q.length; q++) {
            const a = Q[q];
            for (let i = 0; i < N; i++) { let s = 0; const b = X[i]; for (let j = 0; j < a.length; j++) { const t = a[j] - b[j]; s += t * t; } dist[i] = s; idx[i] = i; }
            idx.sort((u, v) => dist[u] - dist[v]);
            let c = 0, m = 0;
            ks.forEach((kk, ki) => { while (m < kk) c += Y[idx[m++]]; const f = c / kk; e[ki] += f === 0.5 ? 0.5 : ((f > 0.5 ? 1 : 0) !== QY[q] ? 1 : 0); });
          }
          return Array.from(e, v => v / Q.length);
        };
        const eT = errFor(TX, TY), eR = errFor(X, Y);
        ks.forEach((kk, ki) => { te[ki] += eT[ki]; tr[ki] += eR[ki]; });
        runs++;
      }
      function reset() { loop.stop(); te = new Float64Array(KS.length); tr = new Float64Array(KS.length); runs = 0; oneRun(); draw(); }
      function step() { oneRun(); draw(); if (runs >= 40) return false; }
      function draw() {
        const N = 2 * Math.round(n.get()), ks = KS.filter(kk => kk <= N), R = KW.bayesErr();
        const mt = ks.map((kk, i) => te[i] / runs), mr = ks.map((kk, i) => tr[i] / runs);
        p.setDomain([1, 200], [0, Math.max(0.5, ...mt) * 1.05]).clear();
        p.hline(R, { color: 'ink', dash: '5 4', width: 1.5 });
        p.hline(2 * R * (1 - R), { color: 'muted', dash: '2 4', width: 1.5 });
        p.line(ks.map((kk, i) => [kk, mr[i]]), { color: 's1', width: 2.4 });
        p.line(ks.map((kk, i) => [kk, mt[i]]), { color: 's2', width: 2.4 });
        p.points(ks.map((kk, i) => [kk, mt[i]]), { color: 's2', r: 3 });
        const b = S.argmin(mt);
        p.dot(ks[b], mt[b], { color: 'accent', r: 6 });
        p.legend([{ label: 'test error', color: 's2' }, { label: 'training error', color: 's1' }, { label: 'Bayes error', color: 'ink', dash: true }, { label: 'Cover–Hart 2R*(1−R*)', color: 'muted', dash: true }]);
        sBest.set(String(ks[b]), `test error ${pct(mt[b])}`);
        s1.set(pct(mt[0]));
        sB.set(pct(R));
        sCH.set(pct(2 * R * (1 - R)));
        sR.set(String(runs), `${Math.round(D.get())} noise dims`);
      }
      reset();
    },
  });

  /* =================================================================== */
  /* vc-theory                                                              */
  /* =================================================================== */
  const VCC = {
    thr: { label: 'Thresholds (1-D)', d: 1 },
    int: { label: 'Intervals (1-D)', d: 2 },
    half: { label: 'Half-planes', d: 3 },
    rect: { label: 'Rectangles', d: 4 },
  };
  // Returns a witness hypothesis if the labeling is realizable, else null.
  function realize(cls, P, lab) {
    const pos = P.filter((_, i) => lab[i]), neg = P.filter((_, i) => !lab[i]);
    if (cls === 'thr') {
      if (!pos.length) return { t: 2 };
      const mp = Math.min(...pos.map(q => q[0])), mn = neg.length ? Math.max(...neg.map(q => q[0])) : -1;
      return mn < mp ? { t: neg.length ? (mn + mp) / 2 : mp - 0.03 } : null;
    }
    if (cls === 'int' || cls === 'rect') {
      if (!pos.length) return { empty: true };
      const x0 = Math.min(...pos.map(q => q[0])), x1 = Math.max(...pos.map(q => q[0]));
      const y0 = Math.min(...pos.map(q => q[1])), y1 = Math.max(...pos.map(q => q[1]));
      const bad = cls === 'int' ? neg.some(q => q[0] >= x0 && q[0] <= x1) : neg.some(q => q[0] >= x0 && q[0] <= x1 && q[1] >= y0 && q[1] <= y1);
      return bad ? null : { x0, x1, y0, y1 };
    }
    // half-planes: search normal directions (those orthogonal to pair differences, perturbed, plus a uniform grid)
    if (!pos.length || !neg.length) return { all: !!pos.length };
    const angs = [];
    for (let i = 0; i < P.length; i++) for (let j = i + 1; j < P.length; j++) {
      const a = Math.atan2(P[j][1] - P[i][1], P[j][0] - P[i][0]) + Math.PI / 2;
      for (const s of [0, Math.PI]) for (const e of [-1e-6, 1e-6]) angs.push(a + s + e);
    }
    for (let i = 0; i < 72; i++) angs.push(2 * Math.PI * i / 72);
    for (const a of angs) {
      const w = [Math.cos(a), Math.sin(a)], pr = q => w[0] * q[0] + w[1] * q[1];
      const lo = Math.min(...pos.map(pr)), hi = Math.max(...neg.map(pr));
      if (hi < lo - 1e-12) return { w, b: (lo + hi) / 2 };
    }
    return null;
  }
  // Clip the unit square to {w·x >= b} (Sutherland–Hodgman against one half-plane).
  function clipBox(w, b, box) {
    const out = [], f = q => w[0] * q[0] + w[1] * q[1] - b;
    for (let i = 0; i < box.length; i++) {
      const A = box[i], B = box[(i + 1) % box.length], fa = f(A), fb = f(B);
      if (fa >= 0) out.push(A);
      if ((fa >= 0) !== (fb >= 0)) { const t = fa / (fa - fb); out.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]); }
    }
    return out;
  }
  const phiD = (d, m) => { let s = 0; for (let i = 0; i <= Math.min(d, m); i++) s += Math.exp(S.lchoose(m, i)); return Math.round(s); };

  Atlas.lab('vc-theory-shatter', {
    title: 'The shattering game',
    blurb: 'Click to place up to 7 points (1-D classes use only the horizontal coordinate). The lab tests all $2^m$ labelings against the chosen class and draws a witness hypothesis when one exists. A set is shattered when every labeling is realizable. Find $d$ points that are shattered — then try $d+1$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el, { side: true });
      const cls = UI.segmented(controls, { label: 'Hypothesis class', options: Object.keys(VCC).map(k => ({ value: k, label: VCC[k].label })), value: 'half', onChange: () => { lab = 0; update(); } });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Next labeling', () => { lab = (lab + 1) % (1 << P.length || 1); update(); }, { primary: true });
      UI.button(row, 'Show a failure', () => { const f = fails(); if (f.length) { lab = f.find(l => l > lab) ?? f[0]; } update(); });
      const row2 = UI.h('div', 'btn-row', controls);
      UI.button(row2, 'Random d points', () => randomPts(VCC[cls.get()].d));
      UI.button(row2, 'Random d+1 points', () => randomPts(VCC[cls.get()].d + 1));
      UI.button(row2, 'Clear', () => { P = []; lab = 0; update(); });
      const p = new Plot(stage, { width: 640, height: 360, x: [0, 1], y: [0, 1], equal: true, xticks: 0, yticks: 0, grid: false, title: 'Points and the current labeling' });
      const sM = UI.stat(stats, 'Points $m$', { accent: 'accent' });
      const sR = UI.stat(stats, 'Realizable labelings');
      const sS = UI.stat(stats, 'Shattered?');
      const sD = UI.stat(stats, 'VC dimension of class');
      const sSa = UI.stat(stats, 'Sauer–Shelah bound $\\Phi_d(m)$');
      let P = [[0.25, 0.3], [0.7, 0.25], [0.5, 0.75]], lab = 5;
      function randomPts(m) {
        P = [];
        while (P.length < Math.min(m, 7)) { const q = [0.1 + 0.8 * ctx.rng.u(), 0.12 + 0.76 * ctx.rng.u()]; if (P.every(r => Math.hypot(r[0] - q[0], r[1] - q[1]) > 0.08)) P.push(q); }
        lab = 0; update();
      }
      const labels = l => P.map((_, i) => (l >> i) & 1);
      function fails() { const out = []; for (let l = 0; l < (1 << P.length); l++) if (!realize(cls.get(), P, labels(l))) out.push(l); return out; }
      function update() {
        const c = cls.get(), m = P.length, L = labels(lab), w = m ? realize(c, P, L) : { empty: true };
        p.clear();
        p.rect(0, 0, 1, 1, { color: 'faint', opacity: 0.05 });
        if (w && m) {
          if (c === 'thr') { p.rect(Math.max(0, w.t), 0, 1, 1, { color: 's2', opacity: 0.14 }); if (w.t > 0 && w.t < 1) p.vline(w.t, { color: 's2', dash: false, width: 2 }); }
          else if (c === 'int' && !w.empty) p.rect(w.x0 - 0.02, 0, w.x1 + 0.02, 1, { color: 's2', opacity: 0.16, stroke: 's2' });
          else if (c === 'rect' && !w.empty) p.rect(w.x0 - 0.025, w.y0 - 0.025, w.x1 + 0.025, w.y1 + 0.025, { color: 's2', opacity: 0.16, stroke: 's2' });
          else if (c === 'half' && w.w) { const poly = clipBox(w.w, w.b, [[0, 0], [1, 0], [1, 1], [0, 1]]); if (poly.length > 2) p.polygon(poly, { color: 's2', opacity: 0.14, stroke: 's2', width: 2 }); }
          else if (c === 'half' && w.all) p.rect(0, 0, 1, 1, { color: 's2', opacity: 0.14 });
        }
        if (c === 'thr' || c === 'int') for (const q of P) p.segment(q[0], 0, q[0], q[1], { color: 'faint', width: 1, dash: '2 3' });
        P.forEach((q, i) => p.dot(q[0], q[1], { color: L[i] ? 's2' : 's1', r: 8, stroke: 'ink', strokeWidth: 1.5 }));
        P.forEach((q, i) => p.text(q[0], q[1], String(i + 1), { anchor: 'middle', dy: 4, size: 10, color: 'surface', weight: 700 }));
        p.text(0.02, 0.97, m ? `labeling ${lab + 1} of ${1 << m}: ${w ? 'realizable' : 'NOT realizable'}` : 'click to add points', { size: 13, weight: 600, color: w || !m ? 'ink' : 'bad' });
        const nf = m ? fails().length : 0, d = VCC[c].d;
        sM.set(String(m));
        sR.set(`${(1 << m) - nf} / ${1 << m}`);
        sS.set(m ? (nf === 0 ? '<span style="color:var(--good)">yes</span>' : '<span style="color:var(--bad)">no</span>') : '—');
        sD.set(String(d));
        sSa.set(String(phiD(d, m)), `vs. 2^m = ${1 << m}`);
      }
      p.onPointer((x, y, e, type) => {
        if (type !== 'down' || !p.inside(x, y) || P.length >= 7) return;
        P.push([x, y]); lab = 0; update();
      });
      update();
    },
  });

  Atlas.lab('vc-theory-bound', {
    title: 'Generalization bounds vs. the actual uniform deviation',
    blurb: 'Class: thresholds $h_t(x) = \\ind\\{x > t\\}$ on $X\\sim U(0,1)$, labels $\\ind\\{X > 1/2\\}$ flipped with probability $p$. Here $\\Pi(n) = n+1$ and $\\sup_t|\\hat R_n(t) - R(t)|$ can be computed exactly. The plot compares its simulated $(1-\\delta)$ quantile with the VC bound and the Rademacher–Massart bound, and shows the excess risk of ERM.',
    mount(el, ctx) {
      const NG = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
      const { controls, stage, stats } = UI.layout(el);
      const dl = UI.slider(controls, { label: 'Confidence $\\delta$', min: 0.01, max: 0.2, step: 0.01, value: 0.05, onInput: draw });
      const noise = UI.slider(controls, { label: 'Label noise $p$', min: 0, max: 0.45, step: 0.05, value: 0.2, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Simulate more', '❚❚ Pause']);
      UI.button(row, 'Restart', reset);
      const p = new Plot(stage, { width: 640, height: 320, x: [15, 12000], y: [1e-4, 10], logx: true, logy: true, xlabel: 'n', ylabel: 'deviation', title: 'Bounds on sup |R̂ − R| (log–log)' });
      const sQ = UI.stat(stats, 'Observed quantile, $n=1000$', { accent: 'accent' });
      const sV = UI.stat(stats, 'VC bound, $n=1000$');
      const sRa = UI.stat(stats, 'Rademacher bound, $n=1000$');
      const sE = UI.stat(stats, 'ERM excess risk, $n=1000$');
      const sN = UI.stat(stats, 'Simulations per $n$');
      let dev, exc, reps;
      function one(nn) {
        const pn = noise.get(), xs = new Float64Array(nn), ys = new Uint8Array(nn);
        for (let i = 0; i < nn; i++) xs[i] = ctx.rng.u();
        xs.sort();
        for (let i = 0; i < nn; i++) { const y = xs[i] > 0.5 ? 1 : 0; ys[i] = ctx.rng.u() < pn ? 1 - y : y; }
        const Rt = t => pn + (1 - 2 * pn) * Math.abs(t - 0.5);
        let zerosTot = 0; for (let i = 0; i < nn; i++) zerosTot += 1 - ys[i];
        // t in [x_(i), x_(i+1)): first i points predicted 0
        let ones = 0, zeros = 0, sup = 0, bestR = Infinity, bestT = 0;
        for (let i = 0; i <= nn; i++) {
          if (i > 0) { if (ys[i - 1]) ones++; else zeros++; }
          const Rh = (ones + zerosTot - zeros) / nn;
          const a = i === 0 ? 0 : xs[i - 1], b = i === nn ? 1 : xs[i];
          let m = Math.max(Math.abs(Rh - Rt(a)), Math.abs(Rh - Rt(b)));
          if (a < 0.5 && b > 0.5) m = Math.max(m, Math.abs(Rh - Rt(0.5)));
          if (m > sup) sup = m;
          if (Rh < bestR) { bestR = Rh; bestT = (a + b) / 2; }
        }
        return { sup, ex: Rt(bestT) - Rt(0.5) };
      }
      function reset() { loop.stop(); dev = NG.map(() => []); exc = NG.map(() => 0); reps = 0; for (let i = 0; i < 20; i++) oneRep(); draw(); }
      function oneRep() { NG.forEach((nn, j) => { const r = one(nn); dev[j].push(r.sup); exc[j] += r.ex; }); reps++; }
      function step() { for (let i = 0; i < 4; i++) oneRep(); draw(); if (reps >= 400) return false; }
      const vcB = (nn, d) => Math.sqrt(32 / nn * (Math.log(8 / d) + Math.log(nn + 1)));
      const raB = (nn, d) => 2 * Math.sqrt(2 * Math.log(nn + 1) / nn) + Math.sqrt(Math.log(2 / d) / (2 * nn));
      function draw() {
        const d = dl.get(), xs = NG;
        const q = dev.map(v => S.quantile(v, 1 - d)), mexc = exc.map(v => v / reps);
        p.clear();
        p.hline(1, { color: 'faint', width: 1 });
        p.line(xs.map(nn => [nn, vcB(nn, d)]), { color: 's1', width: 2.4 });
        p.line(xs.map(nn => [nn, raB(nn, d)]), { color: 's3', width: 2.4 });
        p.line(xs.map((nn, j) => [nn, q[j]]), { color: 's2', width: 2.4 });
        p.points(xs.map((nn, j) => [nn, q[j]]), { color: 's2', r: 3.5 });
        p.line(xs.map((nn, j) => [nn, mexc[j] > 0 ? mexc[j] : NaN]), { color: 's4', width: 2, dash: '5 4' });
        p.line(xs.map(nn => [nn, 1 / Math.sqrt(nn)]), { color: 'muted', width: 1, dash: '2 4' });
        p.text(xs[0], 1, 'trivial bound 1', { dy: -5, size: 11, color: 'muted' });
        p.legend([{ label: 'VC bound', color: 's1' }, { label: 'Rademacher + Massart', color: 's3' }, { label: `observed ${Math.round(100 * (1 - d))}% quantile`, color: 's2' }, { label: 'ERM excess risk (mean)', color: 's4', dash: true }, { label: '1/√n', color: 'muted', dash: true }]);
        const j = NG.indexOf(1000);
        sQ.set(S.fmt(q[j]));
        sV.set(S.fmt(vcB(1000, d)), `${(vcB(1000, d) / q[j]).toFixed(0)}× observed`);
        sRa.set(S.fmt(raB(1000, d)), `${(raB(1000, d) / q[j]).toFixed(1)}× observed`);
        sE.set(S.fmt(mexc[j], 4));
        sN.set(String(reps));
      }
      reset();
    },
  });

  /* =================================================================== */
  /* classifier-evaluation                                                  */
  /* =================================================================== */
  Atlas.lab('classifier-evaluation-roc', {
    title: 'ROC, precision–recall and cost-optimal thresholds',
    blurb: 'Scores are $\\N(0,1)$ for negatives and $\\N(\\mu, s^2)$ for positives. Left: the two score densities weighted by prevalence, with your threshold. Right: the ROC (or PR) curve, your operating point (orange), the cost-optimal point and its iso-cost line. AUC $=\\Phi(\\mu/\\sqrt{1+s^2})$ does not depend on prevalence or costs; the optimal threshold and the PR curve do.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const view = UI.segmented(controls, { label: 'Curve', options: ['ROC', 'Precision–recall'], value: 'ROC', onChange: draw });
      const mu = UI.slider(controls, { label: 'Positive-score mean $\\mu$', min: 0, max: 4, step: 0.05, value: 1.5, onInput: draw });
      const sd = UI.slider(controls, { label: 'Positive-score sd $s$', min: 0.5, max: 2, step: 0.05, value: 1, onInput: draw });
      const pi = UI.slider(controls, { label: 'Prevalence $\\pi$', min: 0.01, max: 0.9, step: 0.01, value: 0.2, onInput: () => { sample(); draw(); } });
      const cr = UI.slider(controls, { label: 'Cost ratio $c_{FN}/c_{FP}$', min: 0.1, max: 20, log: true, value: 3, fmt: v => v.toFixed(2), onInput: draw });
      const th = UI.slider(controls, { label: 'Threshold $t$', min: -3, max: 6, step: 0.05, value: 1, onInput: draw });
      UI.button(controls, 'New test sample (n = 400)', () => { sample(); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-4, 7], y: [0, 0.5], xlabel: 'score', ylabel: 'prevalence × density', title: 'Score distributions' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 1], xlabel: 'FPR', ylabel: 'TPR', title: 'ROC curve' });
      const sRates = UI.stat(stats, 'TPR / FPR at $t$', { accent: 'accent' });
      const sPrec = UI.stat(stats, 'Precision / accuracy');
      const sCost = UI.stat(stats, 'Expected cost at $t$');
      const sOpt = UI.stat(stats, 'Cost-optimal threshold');
      const sAUC = UI.stat(stats, 'AUC (exact / sample)');
      let smp;
      function sample() {
        const r = ctx.rng, P = pi.get(), out = [];
        for (let i = 0; i < 400; i++) { const y = r.u() < P ? 1 : 0; out.push([y ? r.normal(mu.get(), sd.get()) : r.normal(), y]); }
        smp = out;
      }
      function empAUC(D) {
        const s = D.slice().sort((a, b) => a[0] - b[0]);
        let sumPos = 0, np = 0, i = 0;
        while (i < s.length) { let j = i; while (j < s.length && s[j][0] === s[i][0]) j++; const avg = (i + j + 1) / 2; for (let k = i; k < j; k++) if (s[k][1]) { sumPos += avg; np++; } i = j; }
        const nn = s.length - np;
        return np && nn ? (sumPos - np * (np + 1) / 2) / (np * nn) : NaN;
      }
      function draw() {
        const m = mu.get(), s = sd.get(), P = pi.get(), r = cr.get(), t = th.get();
        // regenerate the sample's positive scores consistently with current mu, s (keep labels)
        const D = smp.map(([v, y], i) => [y ? m + s * zs[i % zs.length] : v, y]);
        const cFP = 1, cFN = r;
        const TPR = u => 1 - Phi((u - m) / s), FPR = u => 1 - Phi(u);
        const cost = u => cFP * (1 - P) * FPR(u) + cFN * P * (1 - TPR(u));
        const grid = S.linspace(-6, m + 6 * s, 2500);
        let bt = grid[0], bc = Infinity; for (const u of grid) { const c = cost(u); if (c < bc) { bc = c; bt = u; } }
        // left
        const xs = S.linspace(-4, 7, 400), f0 = x => (1 - P) * S.pdf.normal(x), f1 = x => P * S.pdf.normal(x, m, s);
        const ymax = Math.max(...xs.map(f0), ...xs.map(f1)) * 1.15;
        pL.setDomain([-4, 7], [0, ymax]).clear();
        const right = xs.filter(x => x >= t);
        if (right.length > 1) {
          pL.area(right.map(x => [x, f0(x)]), { color: 's1', opacity: 0.3 });
          pL.area(right.map(x => [x, f1(x)]), { color: 's2', opacity: 0.3 });
        }
        pL.line(xs.map(x => [x, f0(x)]), { color: 's1', width: 2.2 });
        pL.line(xs.map(x => [x, f1(x)]), { color: 's2', width: 2.2 });
        pL.vline(t, { color: 'accent', width: 2 });
        pL.vline(bt, { color: 'good', width: 1.4 });
        pL.legend([{ label: 'negatives (1−π)g₀', color: 's1' }, { label: 'positives πg₁', color: 's2' }, { label: 'your t', color: 'accent' }, { label: 'optimal t', color: 'good', dash: true }]);
        // right
        const tg = S.linspace(m + 7 * s, -7, 600);
        const prec = u => { const a = P * TPR(u), b = (1 - P) * FPR(u); return a + b > 1e-14 ? a / (a + b) : NaN; };
        const isROC = view.get() === 'ROC';
        pR.o.xlabel = isROC ? 'FPR' : 'recall (TPR)'; pR.o.ylabel = isROC ? 'TPR' : 'precision';
        pR.setDomain([0, 1], [0, 1.02]).clear();
        // empirical curve from the sample
        const Ds = D.slice().sort((a, b) => b[0] - a[0]), npos = Ds.filter(q => q[1]).length, nneg = Ds.length - npos;
        if (npos && nneg) {
          let tp = 0, fp = 0; const emp = [isROC ? [0, 0] : [0, 1]];
          for (const [, y] of Ds) { if (y) tp++; else fp++; emp.push(isROC ? [fp / nneg, tp / npos] : [tp / npos, tp / (tp + fp)]); }
          pR.line(emp, { color: 'muted', width: 1.2, opacity: 0.8 });
        }
        if (isROC) {
          pR.segment(0, 0, 1, 1, { color: 'faint', width: 1, dash: '3 3' });
          pR.line(tg.map(u => [FPR(u), TPR(u)]), { color: 's3', width: 2.6 });
          const slope = cFP * (1 - P) / (cFN * P), F = FPR(bt), T = TPR(bt);
          pR.segment(F - 2, T - 2 * slope, F + 2, T + 2 * slope, { color: 'good', width: 1.4, dash: '6 4' });
          pR.dot(F, T, { color: 'good', r: 5.5 });
          pR.dot(FPR(t), TPR(t), { color: 'accent', r: 6 });
        } else {
          pR.hline(P, { color: 'faint', width: 1, dash: '3 3' });
          pR.line(tg.map(u => [TPR(u), prec(u)]), { color: 's3', width: 2.6 });
          pR.dot(TPR(bt), prec(bt), { color: 'good', r: 5.5 });
          pR.dot(TPR(t), prec(t), { color: 'accent', r: 6 });
          pR.text(0.98, P, `baseline π = ${P.toFixed(2)}`, { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        }
        pR.legend([{ label: 'exact curve', color: 's3' }, { label: 'sample (n = 400)', color: 'muted', width: 1.2 }].concat(isROC ? [{ label: 'iso-cost line', color: 'good', dash: true }] : []));
        const acc = P * TPR(t) + (1 - P) * (1 - FPR(t));
        sRates.set(`${TPR(t).toFixed(3)} / ${FPR(t).toFixed(3)}`, `t = ${t.toFixed(2)}`);
        sPrec.set(`${S.fmt(prec(t))} / ${acc.toFixed(3)}`);
        sCost.set(S.fmt(cost(t), 4), `optimum ${S.fmt(bc, 4)}`);
        sOpt.set(bt.toFixed(2), `iso-cost slope ${S.fmt(cFP * (1 - P) / (cFN * P))}`);
        sAUC.set(`${Phi(m / Math.sqrt(1 + s * s)).toFixed(3)} / ${S.fmt(empAUC(D))}`);
      }
      const zs = ctx.rng.array(400, () => ctx.rng.normal());
      sample();
      draw();
    },
  });

  Atlas.lab('classifier-evaluation-calibration', {
    title: 'Calibration, AUC and proper scores',
    blurb: 'True posterior $\\eta(x) = \\sigma(a x)$ with $X\\sim\\N(0,1)$. The forecast is a distortion $\\hat p = \\sigma(a x / T + b)$: temperature $T > 1$ makes it underconfident, $T < 1$ overconfident, and $b$ biases it. Left: reliability diagram on a sample. Right: expected Brier score and log loss relative to the truth as functions of $T$ — both minimized at $T=1, b=0$ (propriety). AUC never moves.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const a = UI.slider(controls, { label: 'Signal strength $a$', min: 0.5, max: 5, step: 0.1, value: 2.5, onChange: newSample });
      const T = UI.slider(controls, { label: 'Temperature $T$', min: 0.25, max: 4, log: true, value: 2, fmt: v => v.toFixed(2), onInput: draw });
      const b = UI.slider(controls, { label: 'Bias $b$ (logit shift)', min: -2, max: 2, step: 0.05, value: 0, onInput: draw });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 200, max: 5000, step: 100, value: 2000, onChange: newSample });
      const nb = UI.slider(controls, { label: 'Bins', min: 5, max: 20, step: 1, value: 10, onInput: draw });
      UI.button(controls, 'New sample', newSample, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 1], xlabel: 'forecast p̂', ylabel: 'observed frequency', title: 'Reliability diagram' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0.25, 4], y: [0.9, 2], logx: true, xlabel: 'temperature T', ylabel: 'score ÷ score of truth', title: 'Expected scores vs. temperature' });
      const sAUC = UI.stat(stats, 'AUC', { accent: 'accent' });
      const sBr = UI.stat(stats, 'Brier score');
      const sDec = UI.stat(stats, 'Calibration + refinement');
      const sLL = UI.stat(stats, 'Log loss');
      const sECE = UI.stat(stats, 'ECE (binned)');
      let X, Y, Z;
      function newSample() {
        const nn = Math.round(n.get()), aa = a.get();
        X = ctx.rng.array(nn, () => ctx.rng.normal());
        Y = X.map(x => (ctx.rng.u() < sigm(aa * x) ? 1 : 0));
        const r = S.rng(3); Z = r.array(4000, () => r.normal());   // fixed draws for expected-score curves
        draw();
      }
      // Expected Brier / log loss over X (using η, not Y, for low noise)
      function expected(Tv, bv) {
        const aa = a.get(); let br = 0, ll = 0;
        for (const x of Z) { const e = sigm(aa * x), p = clampP(sigm(aa * x / Tv + bv)); br += (p - e) ** 2 + e * (1 - e); ll -= e * Math.log(p) + (1 - e) * Math.log(1 - p); }
        return { br: br / Z.length, ll: ll / Z.length };
      }
      function auc(scores, labels) {
        const idx = S.range(scores.length).sort((i, j) => scores[i] - scores[j]);
        let sumPos = 0, np = 0;
        idx.forEach((i, r) => { if (labels[i]) { sumPos += r + 1; np++; } });
        const nn = scores.length - np;
        return (sumPos - np * (np + 1) / 2) / (np * nn);
      }
      function draw() {
        const aa = a.get(), Tv = T.get(), bv = b.get(), B = Math.round(nb.get());
        const ph = X.map(x => sigm(aa * x / Tv + bv)), eta = X.map(x => sigm(aa * x));
        // reliability
        const cnt = new Float64Array(B), sp = new Float64Array(B), sy = new Float64Array(B);
        ph.forEach((p, i) => { const k = Math.min(B - 1, Math.floor(p * B)); cnt[k]++; sp[k] += p; sy[k] += Y[i]; });
        pL.clear();
        const hmax = Math.max(...cnt);
        for (let k = 0; k < B; k++) pL.rect(k / B + 0.004, 0, (k + 1) / B - 0.004, 0.18 * cnt[k] / hmax, { color: 'faint', opacity: 0.5 });
        pL.segment(0, 0, 1, 1, { color: 'ink', dash: '4 4', width: 1.2 });
        const cp = S.linspace(0.001, 0.999, 200).map(p => [p, sigm(Tv * (Math.log(p / (1 - p)) - bv))]);
        pL.line(cp, { color: 's3', width: 2 });
        let ece = 0;
        for (let k = 0; k < B; k++) if (cnt[k]) {
          const mp = sp[k] / cnt[k], my = sy[k] / cnt[k];
          ece += cnt[k] / X.length * Math.abs(my - mp);
          pL.dot(mp, my, { color: 's2', r: 2.5 + 4 * Math.sqrt(cnt[k] / hmax) });
        }
        pL.legend([{ label: 'binned frequency', color: 's2', fill: true }, { label: 'true calibration curve', color: 's3' }, { label: 'forecast histogram', color: 'faint', fill: true }], { left: true });
        // expected scores vs T
        const ref = expected(1, 0), Ts = S.linspace(Math.log10(0.25), Math.log10(4), 50).map(v => 10 ** v);
        const ex = Ts.map(tt => expected(tt, bv));
        const ys = ex.flatMap(e => [e.br / ref.br, e.ll / ref.ll]);
        pR.setDomain([0.25, 4], [0.95, Math.min(3, Math.max(...ys)) * 1.03]).clear();
        pR.hline(1, { color: 'ink', dash: '4 4', width: 1.2 });
        pR.line(Ts.map((tt, i) => [tt, ex[i].br / ref.br]), { color: 's1', width: 2.4 });
        pR.line(Ts.map((tt, i) => [tt, ex[i].ll / ref.ll]), { color: 's2', width: 2.4 });
        pR.vline(Tv, { color: 'accent', width: 1.6 });
        pR.legend([{ label: 'Brier', color: 's1' }, { label: 'log loss', color: 's2' }]);
        // stats on the sample
        const sc = scoreProbs(ph, Y);
        let cal = 0, ref2 = 0; ph.forEach((p, i) => { cal += (p - eta[i]) ** 2; ref2 += eta[i] * (1 - eta[i]); });
        sAUC.set(auc(ph, Y).toFixed(4), 'unchanged by T > 0 and b');
        sBr.set(sc.brier.toFixed(4));
        sDec.set(`${(cal / X.length).toFixed(4)} + ${(ref2 / X.length).toFixed(4)}`, 'E(p̂−η)² + Eη(1−η)');
        sLL.set(sc.ll.toFixed(4));
        sECE.set(ece.toFixed(4));
      }
      newSample();
    },
  });
})();
