/* Labs: Statistical Inference I (models, estimation, likelihood, sufficiency, exponential families,
 * MLE, Fisher information, Rao–Blackwell). Content: content/inference-1.js */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const R = String.raw;

  /* ---------------- shared numerical helpers ---------------- */
  function digamma(x) {
    let r = 0;
    while (x < 6) { r -= 1 / x; x += 1; }
    const f = 1 / (x * x);
    return r + Math.log(x) - 0.5 / x - f * (1 / 12 - f * (1 / 120 - f * (1 / 252 - f * (1 / 240 - f / 132))));
  }
  function trigamma(x) {
    let r = 0;
    while (x < 6) { r += 1 / (x * x); x += 1; }
    const f = 1 / (x * x);
    return r + 1 / x + f / 2 + (1 / x) * f * (1 / 6 - f * (1 / 30 - f * (1 / 42 - f / 30)));
  }
  // Gamma shape MLE (scale/rate profiled out): solve log a - digamma(a) = log(mean) - mean(log x).
  function gammaShapeMLE(xs) {
    let m = 0, ml = 0;
    for (const v of xs) { m += v; ml += Math.log(Math.max(v, 1e-300)); }
    m /= xs.length; ml /= xs.length;
    const s = Math.log(m) - ml;
    if (!(s > 1e-12)) return Infinity;
    let a = (3 - s + Math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s);
    for (let i = 0; i < 12; i++) {
      const f = Math.log(a) - digamma(a) - s, d = 1 / a - trigamma(a);
      let an = a - f / d;
      if (!(an > 0)) an = a / 2;
      if (Math.abs(an - a) < 1e-10 * a) { a = an; break; }
      a = an;
    }
    return a;
  }
  const sortNum = a => Float64Array.from(a).sort();
  const medianSorted = s => (s.length % 2 ? s[(s.length - 1) / 2] : 0.5 * (s[s.length / 2 - 1] + s[s.length / 2]));
  function trimmedSorted(s, p) { const k = Math.floor(p * s.length); let t = 0; for (let i = k; i < s.length - k; i++) t += s[i]; return t / (s.length - 2 * k); }
  // Location MLE for scale mixtures of normals by the monotone IRLS / MM iteration, started at the median.
  const LOCW = {
    logistic: u => (Math.abs(u) < 1e-8 ? 0.5 : Math.tanh(u / 2) / u),
    t3: u => 4 / (3 + u * u),
    cauchy: u => 2 / (1 + u * u),
  };
  function locMLE(x, fam, start, iters = 20) {
    const w = LOCW[fam];
    let th = start;
    for (let it = 0; it < iters; it++) {
      let sw = 0, swx = 0;
      for (let i = 0; i < x.length; i++) { const wi = w(x[i] - th); sw += wi; swx += wi * x[i]; }
      const nt = swx / sw;
      if (Math.abs(nt - th) < 1e-9) { th = nt; break; }
      th = nt;
    }
    return th;
  }
  const Phi = x => S.cdf.normal(x), phi = x => S.pdf.normal(x);
  const curve = (lo, hi, f, m = 240) => S.linspace(lo, hi, m).map(v => [v, f(v)]);
  const tv = (p, q) => { let s = 0; for (let i = 0; i < Math.max(p.length, q.length); i++) s += Math.abs((p[i] || 0) - (q[i] || 0)); return s / 2; };
  const fmt = (x, d = 3) => S.fmt(x, d);
  const skew = a => { const m = S.mean(a), s = S.sd(a, 0); let t = 0; for (const v of a) t += ((v - m) / s) ** 3; return t / a.length; };
  const show = (elem, on) => { elem.style.display = on ? '' : 'none'; };

  /* ================================================================
   * statistical-models
   * ================================================================ */
  Atlas.lab('statistical-models-identifiability', {
    title: 'Label switching and weak identification in a mixture',
    blurb: R`Data from $p\,\N(\mu_1,1)+(1-p)\,\N(\mu_2,1)$ with $p$ known. Left: the log-likelihood over $(\mu_1,\mu_2)$ (brighter is higher; the curve encloses an approximate 95% region, $\ell \ge \ell_{\max}-3$). At $p=0.5$ the surface is exactly symmetric about the diagonal, so the data cannot say which component is which. Slide $\mu_1$ toward $\mu_2$ and the two peaks fuse into a long ridge: the parameters become weakly identified and the worst-direction standard error explodes.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const m1 = UI.slider(controls, { label: R`True $\mu_1$`, min: -3, max: 3, step: 0.1, value: -1.5, onChange: newData });
      const m2 = UI.slider(controls, { label: R`True $\mu_2$`, min: -3, max: 3, step: 0.1, value: 1.5, onChange: newData });
      const pp = UI.slider(controls, { label: R`Mixing weight $p$`, min: 0.1, max: 0.9, step: 0.05, value: 0.5, onChange: newData });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 20, max: 500, step: 10, value: 150, onChange: newData });
      UI.button(controls, 'New sample', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-4, 4], y: [-4, 4], xlabel: 'μ₁', ylabel: 'μ₂', title: 'Log-likelihood surface ℓ(μ₁, μ₂)' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [-6, 6], y: [0, 0.5], xlabel: 'x', ylabel: 'density', title: 'Sample and true mixture density' });
      const sGap = UI.stat(stats, R`$\ell(\mu_1,\mu_2)-\ell(\mu_2,\mu_1)$`, { accent: 'accent' });
      const sMle = UI.stat(stats, R`MLE $(\hat\mu_1,\hat\mu_2)$`);
      const sSe = UI.stat(stats, 'Worst-direction SE');
      const sBest = UI.stat(stats, 'Best-direction SE');
      let x = [];
      function newData() {
        const p = pp.get(), a = m1.get(), b = m2.get(), r = ctx.rng;
        x = r.array(Math.round(nS.get()), () => (r.u() < p ? r.normal(a, 1) : r.normal(b, 1)));
        draw();
      }
      function ll(a, b) {
        const p = pp.get();
        let s = 0;
        for (let i = 0; i < x.length; i++) { const da = x[i] - a, db = x[i] - b; s += Math.log(p * Math.exp(-0.5 * da * da) + (1 - p) * Math.exp(-0.5 * db * db) + 1e-300); }
        return s - 0.5 * x.length * Math.log(2 * Math.PI);
      }
      function draw() {
        const NX = 64, G = new Float64Array(NX * NX);
        let best = -Infinity, bi = 0;
        for (let j = 0; j < NX; j++) for (let i = 0; i < NX; i++) {
          const v = ll(-4 + 8 * (i + 0.5) / NX, 4 - 8 * (j + 0.5) / NX);
          G[j * NX + i] = v;
          if (v > best) { best = v; bi = j * NX + i; }
        }
        const start = [-4 + 8 * ((bi % NX) + 0.5) / NX, 4 - 8 * (Math.floor(bi / NX) + 0.5) / NX];
        const opt = S.nelderMead(v => -ll(v[0], v[1]), start, { step: 0.1, maxIter: 400 });
        const mle = opt.x, lmax = -opt.fx;
        const bg = Atlas.rgb('bg'), c1 = Atlas.rgb('s1'), c2 = Atlas.rgb('s4');
        pL.clear();
        pL.raster((u, v) => {
          const i = Math.max(0, Math.min(NX - 1, Math.floor((u + 4) / 8 * NX))), j = Math.max(0, Math.min(NX - 1, Math.floor((4 - v) / 8 * NX)));
          const t = Math.max(0, 1 + (G[j * NX + i] - lmax) / 30);
          return t > 0.9 ? Atlas.mix(c1, c2, (t - 0.9) * 10) : Atlas.mix(bg, c1, t * t / 0.81);
        }, { nx: NX, ny: NX });
        pL.segment(-4, -4, 4, 4, { color: 'faint', width: 1, dash: true });
        pL.contour((u, v) => ll(u, v), [lmax - 3], { color: 'ink', nx: 44, ny: 44, width: 1.4 });
        pL.dot(m1.get(), m2.get(), { color: 'ink', r: 5 });
        pL.dot(m2.get(), m1.get(), { color: 'surface', stroke: 'ink', r: 5 });
        pL.dot(mle[0], mle[1], { color: 'accent', r: 4 });
        pL.text(m1.get(), m2.get(), 'truth', { dx: 8, dy: -6, size: 11 });
        pL.text(m2.get(), m1.get(), 'labels swapped', { dx: 8, dy: 12, size: 11, color: 'muted' });
        // right: data + density
        const p = pp.get(), a = m1.get(), b = m2.get();
        const h = S.histogram(x, { bins: 36, lo: -6, hi: 6 });
        const dens = v => p * phi(v - a) + (1 - p) * phi(v - b);
        const ymax = Math.max(0.45, ...h.map(q => q.density)) * 1.1;
        pR.setDomain([-6, 6], [0, ymax]).clear();
        pR.bars(h, { color: 's1', opacity: 0.35 });
        pR.line(curve(-6, 6, dens), { color: 'ink', width: 2 });
        pR.line(curve(-6, 6, v => p * phi(v - a)), { color: 's2', width: 1.3, dash: true });
        pR.line(curve(-6, 6, v => (1 - p) * phi(v - b)), { color: 's3', width: 1.3, dash: true });
        pR.legend([{ label: 'mixture', color: 'ink' }, { label: 'component 1', color: 's2', dash: true }, { label: 'component 2', color: 's3', dash: true }]);
        // observed information at the MLE by finite differences
        const hh = 1e-3, f = (u, v) => ll(u, v);
        const f0 = f(mle[0], mle[1]);
        const h11 = (f(mle[0] + hh, mle[1]) - 2 * f0 + f(mle[0] - hh, mle[1])) / (hh * hh);
        const h22 = (f(mle[0], mle[1] + hh) - 2 * f0 + f(mle[0], mle[1] - hh)) / (hh * hh);
        const h12 = (f(mle[0] + hh, mle[1] + hh) - f(mle[0] + hh, mle[1] - hh) - f(mle[0] - hh, mle[1] + hh) + f(mle[0] - hh, mle[1] - hh)) / (4 * hh * hh);
        const ev = S.eigSym([[-h11, -h12], [-h12, -h22]]).values;
        const lo = Math.min(...ev), hi = Math.max(...ev);
        sGap.set(fmt(ll(a, b) - ll(b, a), 2), Math.abs(p - 0.5) < 1e-9 ? 'exactly 0 when p = 0.5' : 'swap is distinguishable');
        sMle.set(`(${fmt(mle[0], 2)}, ${fmt(mle[1], 2)})`);
        sSe.set(lo > 0 ? fmt(1 / Math.sqrt(lo), 3) : '∞', 'from smallest eigenvalue of J');
        sBest.set(hi > 0 ? fmt(1 / Math.sqrt(hi), 3) : '∞', 'from largest eigenvalue of J');
      }
      newData();
    },
  });

  const TRUTHS = {
    gamma: { label: 'Gamma(shape 2)', sample: r => r.gamma(2, 1), pdf: x => S.pdf.gamma(x, 2, 1), range: [-1, 9], int: [1e-7, 45] },
    t3: { label: 'Student t, 3 df', sample: r => r.t(3), pdf: x => S.pdf.t(x, 3), range: [-7, 7], int: [-150, 150] },
    bimodal: { label: 'Bimodal mixture', sample: r => (r.u() < 0.5 ? r.normal(-2, 0.8) : r.normal(2, 0.8)), pdf: x => 0.5 * S.pdf.normal(x, -2, 0.8) + 0.5 * S.pdf.normal(x, 2, 0.8), range: [-5.5, 5.5], int: [-12, 12] },
    lognormal: { label: 'Lognormal(0, 0.6)', sample: r => r.lognormal(0, 0.6), pdf: x => S.pdf.lognormal(x, 0, 0.6), range: [-0.5, 5], int: [1e-7, 40] },
    normal: { label: 'Normal(1, 1.5²) — model correct', sample: r => r.normal(1, 1.5), pdf: x => S.pdf.normal(x, 1, 1.5), range: [-4, 6], int: [-12, 14] },
  };
  const FITS = {
    Normal: {
      fit(x) { const m = S.mean(x); return [m, Math.sqrt(S.variance(x, 0))]; },
      pdf: (v, th) => S.pdf.normal(v, th[0], th[1]),
      // model-based vs sandwich SE for the scale parameter (sigma^2 for the normal)
      scaleSE(x, th) { const n = x.length, s2 = th[1] ** 2; let m4 = 0; for (const v of x) m4 += (v - th[0]) ** 4; m4 /= n; return [s2 * Math.sqrt(2 / n), Math.sqrt(Math.max(m4 - s2 * s2, 0) / n)]; },
      names: [R`\mu`, R`\sigma`], scaleName: R`\sigma^2`,
    },
    Laplace: {
      fit(x) { const s = sortNum(x), m = medianSorted(s); let b = 0; for (const v of x) b += Math.abs(v - m); return [m, b / x.length]; },
      pdf: (v, th) => S.pdf.laplace(v, th[0], th[1]),
      scaleSE(x, th) { const n = x.length, d = x.map(v => Math.abs(v - th[0])); return [th[1] / Math.sqrt(n), S.sd(d, 0) / Math.sqrt(n)]; },
      names: [R`m`, R`b`], scaleName: R`b`,
    },
  };
  const pseudoCache = {};
  function pseudoTrue(tk, fk) {
    const key = tk + ':' + fk;
    if (!pseudoCache[key]) { const r = S.rng(7); pseudoCache[key] = FITS[fk].fit(r.array(200000, () => TRUTHS[tk].sample(r))); }
    return pseudoCache[key];
  }
  function klNumeric(p, q, lo, hi, m = 6000) {
    const h = (hi - lo) / m; let s = 0;
    for (let i = 0; i <= m; i++) {
      const x = lo + i * h, a = p(x);
      if (a > 1e-300) { const b = Math.max(q(x), 1e-300); s += (i === 0 || i === m ? 0.5 : 1) * a * Math.log(a / b); }
    }
    return s * h;
  }

  Atlas.lab('statistical-models-misspec', {
    title: 'A wrong model converges to the KL projection',
    blurb: R`Fit a normal or Laplace model by maximum likelihood to data from a different distribution. As $n$ grows (press Grow), the fitted density (blue) settles on the pseudo-true member $f_{\theta^*}$ (dashed) — the member closest to the truth in KL divergence — not on the truth (black). The last tiles compare the model-based standard error of the scale estimate with the sandwich (model-free) one: they agree only when the model is right.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const truth = UI.select(controls, { label: 'True distribution', options: Object.keys(TRUTHS).map(k => ({ value: k, label: TRUTHS[k].label })), value: 'bimodal', onChange: newSample });
      const model = UI.segmented(controls, { label: 'Fitted model', options: ['Normal', 'Laplace'], value: 'Normal', onChange: draw });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 10, max: 20000, value: 30, log: true, fmt: v => String(Math.round(v)), onChange: newSample });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newSample);
      const loop = ctx.loop(grow);
      ctx.playButton(row, loop, ['▶ Grow n', '❚❚ Pause']);
      const p = new Plot(stage, { width: 640, height: 300, xlabel: 'x', ylabel: 'density' });
      const sFit = UI.stat(stats, R`MLE $\hat\theta$`);
      const sStar = UI.stat(stats, R`Pseudo-true $\theta^*$`, { accent: 'accent' });
      const sKL = UI.stat(stats, R`$\KL(p\,\|\,f_{\theta^*})$`);
      const sSEm = UI.stat(stats, 'Scale SE: model-based');
      const sSEs = UI.stat(stats, 'Scale SE: sandwich');
      let x = [];
      function newSample() { loop.stop(); const T = TRUTHS[truth.get()]; x = ctx.rng.array(Math.round(nS.get()), () => T.sample(ctx.rng)); draw(); }
      function grow() {
        const T = TRUTHS[truth.get()];
        const add = Math.ceil(x.length * 0.04) + 1;
        for (let i = 0; i < add && x.length < 20000; i++) x.push(T.sample(ctx.rng));
        nS.set(x.length);
        draw();
        if (x.length >= 20000) return false;
      }
      function draw() {
        const T = TRUTHS[truth.get()], F = FITS[model.get()];
        const th = F.fit(x), ts = pseudoTrue(truth.get(), model.get());
        const [lo, hi] = T.range;
        const h = S.histogram(x, { bins: Math.min(60, Math.max(12, Math.round(2 * Math.sqrt(x.length)))), lo, hi });
        const grid = S.linspace(lo, hi, 300);
        const ymax = Math.max(...grid.map(T.pdf), ...grid.map(v => F.pdf(v, ts)), ...h.map(b => b.density)) * 1.12;
        p.setDomain([lo, hi], [0, Math.min(ymax, 2.5)]).clear();
        p.bars(h, { color: 'faint', opacity: 0.5 });
        p.line(grid.map(v => [v, T.pdf(v)]), { color: 'ink', width: 2 });
        p.line(grid.map(v => [v, F.pdf(v, ts)]), { color: 's2', width: 2, dash: true });
        p.line(grid.map(v => [v, F.pdf(v, th)]), { color: 's1', width: 2 });
        p.legend([{ label: 'truth p', color: 'ink' }, { label: 'fitted f(θ̂)', color: 's1' }, { label: 'KL projection f(θ*)', color: 's2', dash: true }, { label: 'sample', color: 'faint', fill: true }]);
        const nm = F.names;
        sFit.set(`(${fmt(th[0], 2)}, ${fmt(th[1], 2)})`, `$${nm[0]}, ${nm[1]}$, n = ${x.length}`);
        sStar.set(`(${fmt(ts[0], 2)}, ${fmt(ts[1], 2)})`);
        sKL.set(fmt(klNumeric(T.pdf, v => F.pdf(v, ts), T.int[0], T.int[1]), 4), 'best achievable in the model');
        const [m, sw] = F.scaleSE(x, th);
        sSEm.set(fmt(m, 4), `assumes the model, for $${F.scaleName}$`);
        sSEs.set(fmt(sw, 4), sw > 1.15 * m ? 'model understates' : sw < 0.87 * m ? 'model overstates' : 'close');
        ctx.math(stats);
      }
      newSample();
    },
  });

  /* ================================================================
   * point-estimation
   * ================================================================ */
  // Exact moments of the pretest / thresholding estimator T = Xbar * 1{|Xbar| >= c}, Xbar ~ N(theta, 1/n).
  function threshMoments(theta, n, c) {
    const rn = Math.sqrt(n), a = rn * (-c - theta), b = rn * (c - theta);
    const pin = Phi(b) - Phi(a);
    const nmse = n * theta * theta * pin + 1 - pin + b * phi(b) - a * phi(a);
    const bias = -(theta * pin + (phi(a) - phi(b)) / rn);
    return { mse: nmse / n, bias, var: nmse / n - bias * bias };
  }
  const RISK = {
    mean: {
      xlabel: 'θ (true mean, σ = 1)', range: () => [-1.5, 1.5],
      ests: (n, c) => [
        { label: 'X̄', color: 's1', f: th => ({ bias: 0, var: 1 / n }) },
        { label: `${c.toFixed(2)}·X̄ (shrink)`, color: 's2', f: th => ({ bias: (c - 1) * th, var: c * c / n }) },
        { label: 'pretest: 0 unless |X̄| > 1.96/√n', color: 's3', f: th => threshMoments(th, n, 1.96 / Math.sqrt(n)) },
      ],
    },
    binom: {
      xlabel: 'p (true success probability)', range: () => [0, 1],
      ests: n => {
        const mk = (a, b) => p => ({ bias: (a - b * p) / (n + b), var: n * p * (1 - p) / (n + b) ** 2 });
        return [
          { label: 'X/n (MLE, unbiased)', color: 's1', f: mk(0, 0) },
          { label: '(X+1)/(n+2) (Laplace)', color: 's2', f: mk(1, 2) },
          { label: '(X+√n/2)/(n+√n) (minimax)', color: 's3', f: mk(Math.sqrt(n) / 2, Math.sqrt(n)) },
        ];
      },
    },
    var: {
      xlabel: 'σ² (true variance)', range: () => [0, 2],
      ests: n => [n - 1, n, n + 1].map((d, i) => ({
        label: ['SS/(n−1) (unbiased)', 'SS/n (MLE)', 'SS/(n+1) (min MSE)'][i], color: ['s1', 's2', 's3'][i],
        f: s2 => ({ bias: ((n - 1) / d - 1) * s2, var: 2 * (n - 1) * s2 * s2 / (d * d) }),
      })),
    },
  };

  Atlas.lab('point-estimation-risk', {
    title: 'Risk curves: bias² + variance across the parameter space',
    blurb: R`Each curve is an estimator's exact MSE as a function of the true parameter. Click or drag on the plot to pick a parameter value; the tiles split each MSE into bias² and variance there. Notice how the curves cross (no estimator wins everywhere), how shrinkage wins near its target, and how the unbiased variance estimator loses to both biased ones at every $\sigma^2$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sc = UI.segmented(controls, { label: 'Problem', options: [{ value: 'mean', label: 'Normal mean' }, { value: 'binom', label: 'Binomial p' }, { value: 'var', label: 'Normal variance' }], value: 'mean', onChange: () => { pick = null; draw(); } });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 2, max: 100, step: 1, value: 10, onInput: draw });
      const cS = UI.slider(controls, { label: R`Shrinkage factor $c$ (normal mean)`, min: 0, max: 1, step: 0.01, value: 0.8, onInput: draw });
      const p = new Plot(stage, { width: 640, height: 320, xlabel: 'θ', ylabel: 'MSE' });
      const tiles = [0, 1, 2].map(i => UI.stat(stats, ''));
      const note = UI.note(stats, '');
      let pick = null;
      p.onPointer((u, v, e, type) => { if (type === 'down' || type === 'drag') { pick = u; draw(); } });
      function draw() {
        const key = sc.get(), P = RISK[key], n = Math.round(nS.get()), c = cS.get();
        const ests = P.ests(n, c), [lo, hi] = P.range();
        const g = S.linspace(lo, hi, 301);
        const curves = ests.map(E => g.map(t => { const m = E.f(t); return [t, m.bias * m.bias + m.var]; }));
        let ymax = 0;
        for (const cu of curves) for (const q of cu) if (Number.isFinite(q[1])) ymax = Math.max(ymax, q[1]);
        if (key === 'mean') ymax = Math.min(ymax, 4 / n);
        p.o.xlabel = P.xlabel;
        p.setDomain([lo, hi], [0, ymax * 1.08]).clear();
        if (pick == null || pick < lo || pick > hi) pick = key === 'binom' ? 0.5 : key === 'var' ? 1 : 0.3;
        p.vline(pick, { color: 'muted' });
        ests.forEach((E, i) => p.line(curves[i], { color: E.color, width: 2.2 }));
        p.legend(ests.map(E => ({ label: E.label, color: E.color })), { left: key !== 'var' ? false : true });
        ests.forEach((E, i) => {
          const m = E.f(pick), mse = m.bias * m.bias + m.var;
          tiles[i].el.querySelector('.stat-l').textContent = E.label;
          tiles[i].set(fmt(mse, 4), `bias² ${fmt(m.bias * m.bias, 4)} + var ${fmt(m.var, 4)}`);
        });
        const best = S.argmin(ests.map(E => { const m = E.f(pick); return m.bias * m.bias + m.var; }));
        note.set(`At the selected value (${fmt(pick, 2)}) the smallest MSE belongs to <b>${ests[best].label}</b>. ` +
          (key === 'mean' ? `Shrinkage beats $\\bar X$ exactly when $\\theta^2 \\lt \\frac{1+c}{1-c}\\cdot\\frac1n$.` :
            key === 'binom' ? `The minimax estimator has constant risk $1/(4(1+\\sqrt n)^2) = ${fmt(1 / (4 * (1 + Math.sqrt(n)) ** 2), 5)}$.` :
              `All three risks are proportional to $\\sigma^4$, so their ranking is the same at every $\\sigma^2$.`));
      }
      draw();
    },
  });

  Atlas.lab('point-estimation-rates', {
    title: 'Root-n versus n: two consistent estimators of a uniform endpoint',
    blurb: R`Samples from $\text{Unif}(0,1)$. Left: sampling distributions of the moment estimator $2\bar X$ and of $X_{(n)}$ and $\frac{n+1}{n}X_{(n)}$ at the chosen $n$. Right: their standard deviations against $n$ on log–log axes, with reference slopes $-1/2$ and $-1$. Both are consistent; the maximum is an order of magnitude faster and its sampling distribution is not normal.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 5, max: 1000, value: 20, log: true, fmt: v => String(Math.round(v)), onChange: draw });
      UI.button(controls, 'Resimulate', draw, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, xlabel: 'estimate of θ = 1', ylabel: 'density', title: 'Sampling distributions' });
      const pR = new Plot(stage, { width: 420, height: 280, logx: true, logy: true, x: [4, 1200], y: [1e-4, 0.5], xlabel: 'n', ylabel: 'sd of estimator', title: 'Rates of convergence' });
      const s1 = UI.stat(stats, R`$\sqrt n\cdot\text{sd}(2\bar X)$`, { accent: 'accent' });
      const s2 = UI.stat(stats, R`$n\cdot\text{sd}(X_{(n)})$`);
      const s3 = UI.stat(stats, R`Bias of $X_{(n)}$`);
      const s4 = UI.stat(stats, R`MSE ratio $2\bar X : \tfrac{n+1}{n}X_{(n)}$`);
      const NS = [5, 10, 20, 50, 100, 200, 500, 1000];
      let sds = null;
      function sim(n, reps, r) {
        const a = new Float64Array(reps), b = new Float64Array(reps);
        for (let k = 0; k < reps; k++) { let s = 0, m = 0; for (let i = 0; i < n; i++) { const u = r.u(); s += u; if (u > m) m = u; } a[k] = 2 * s / n; b[k] = m; }
        return [a, b];
      }
      function precompute() {
        const r = S.rng(11);
        sds = NS.map(n => { const [a, b] = sim(n, 300, r); const c = b.map(v => v * (n + 1) / n); return [S.sd(a), S.sd(b), S.sd(c)]; });
      }
      function draw() {
        if (!sds) precompute();
        const n = Math.round(nS.get()), reps = 3000;
        const [a, b] = sim(n, reps, ctx.rng), c = b.map(v => v * (n + 1) / n);
        const sa = S.sd(a), lo = Math.max(0, 1 - 4 * sa), hi = Math.max(1 + 4 * sa, 1 + 2 / n);
        const bins = 60;
        const ha = S.histogram(a, { bins, lo, hi }), hb = S.histogram(b, { bins, lo, hi }), hc = S.histogram(c, { bins, lo, hi });
        const ymax = Math.max(...ha.map(q => q.density), ...hb.map(q => q.density), ...hc.map(q => q.density)) * 1.08;
        pL.setDomain([lo, hi], [0, ymax]).clear();
        pL.bars(ha, { color: 's1', opacity: 0.45 });
        pL.bars(hb, { color: 's2', opacity: 0.45 });
        pL.bars(hc, { color: 's3', opacity: 0.45 });
        pL.vline(1, { color: 'ink', dash: false, width: 1.5 });
        pL.legend([{ label: '2X̄', color: 's1', fill: true }, { label: 'X(n)', color: 's2', fill: true }, { label: '(n+1)/n·X(n)', color: 's3', fill: true }], { left: true });
        pR.clear();
        ['s1', 's2', 's3'].forEach((col, j) => {
          const pts = NS.map((m, i) => [m, sds[i][j]]);
          pR.line(pts, { color: col, width: 2 });
          pR.points(pts, { color: col, r: 3 });
        });
        pR.line([[5, sds[0][0] * 1.4], [1000, sds[0][0] * 1.4 * Math.sqrt(5 / 1000)]], { color: 'muted', dash: true, width: 1.2 });
        pR.line([[5, sds[0][1] * 0.6], [1000, sds[0][1] * 0.6 * 5 / 1000]], { color: 'muted', dash: true, width: 1.2 });
        pR.text(1000, sds[0][0] * 1.4 * Math.sqrt(5 / 1000), 'slope −1/2', { anchor: 'end', dy: -6, size: 11, color: 'muted' });
        pR.text(1000, sds[0][1] * 0.6 * 5 / 1000, 'slope −1', { anchor: 'end', dy: 14, size: 11, color: 'muted' });
        pR.vline(n, { color: 'accent', width: 1.2 });
        const mse = (arr) => { let s = 0; for (const v of arr) s += (v - 1) ** 2; return s / arr.length; };
        s1.set(fmt(Math.sqrt(n) * sa), R`limit $1/\sqrt3 \approx 0.577$`);
        s2.set(fmt(n * S.sd(b)), 'limit 1 (Exp(1) spread)');
        s3.set(fmt(S.mean(b) - 1, 4), `exact: −1/(n+1) = ${fmt(-1 / (n + 1), 4)}`);
        s4.set(fmt(mse(a) / mse(c), 2), `theory (n+2)/3 = ${fmt((n + 2) / 3, 2)}`);
        ctx.math(stats);
      }
      draw();
    },
  });

  /* ================================================================
   * method-of-moments
   * ================================================================ */
  const areGamma = a => 1 / (2 * (a + 1) * (a * trigamma(a) - 1));

  Atlas.lab('method-of-moments-gamma', {
    title: 'Moments versus maximum likelihood for a Gamma shape',
    blurb: R`Repeatedly draw $n$ observations from Gamma$(\alpha, 1)$ and estimate $\alpha$ (scale unknown) by the method of moments, $\tilde\alpha = \bar X^2/\hat\sigma^2$, and by maximum likelihood. Left: the two sampling distributions. Right: the theoretical asymptotic relative efficiency $\text{ARE}(\alpha)$ with the simulated variance ratio as a dot. Make $\alpha$ small to watch the moment estimator waste most of the sample.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const aS = UI.slider(controls, { label: R`True shape $\alpha$`, min: 0.2, max: 20, value: 1, log: true, fmt: v => v.toFixed(2), onChange: draw });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 10, max: 500, step: 10, value: 50, onChange: draw });
      UI.button(controls, 'Resimulate', draw, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, xlabel: 'estimate of α', ylabel: 'density', title: 'Sampling distributions' });
      const pR = new Plot(stage, { width: 420, height: 280, logx: true, x: [0.1, 50], y: [0, 1.05], xlabel: 'α', ylabel: 'Var(MLE) / Var(MoM)', title: 'Efficiency of the method of moments' });
      const sM = UI.stat(stats, R`MoM: mean ± sd`);
      const sL = UI.stat(stats, R`MLE: mean ± sd`);
      const sE = UI.stat(stats, 'Simulated variance ratio', { accent: 'accent' });
      const sT = UI.stat(stats, 'Asymptotic ARE');
      const aGrid = S.linspace(Math.log(0.1), Math.log(50), 200).map(Math.exp);
      function draw() {
        const a = aS.get(), n = Math.round(nS.get()), reps = 1200, r = ctx.rng;
        const mom = new Float64Array(reps), mle = new Float64Array(reps), x = new Float64Array(n);
        for (let k = 0; k < reps; k++) {
          let s = 0;
          for (let i = 0; i < n; i++) { x[i] = Math.max(r.gamma(a, 1), 1e-300); s += x[i]; }
          const m = s / n; let v = 0;
          for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
          mom[k] = m * m / (v / n);
          mle[k] = gammaShapeMLE(x);
        }
        const all = sortNum([...mom, ...mle].filter(Number.isFinite));
        const hi = S.quantile(all, 0.99, true), lo = Math.max(0, S.quantile(all, 0.003, true));
        const hm = S.histogram(mom, { bins: 50, lo, hi }), hl = S.histogram(mle, { bins: 50, lo, hi });
        const ymax = Math.max(...hm.map(q => q.density), ...hl.map(q => q.density)) * 1.1;
        pL.setDomain([lo, hi], [0, ymax]).clear();
        pL.bars(hm, { color: 's2', opacity: 0.45 });
        pL.bars(hl, { color: 's1', opacity: 0.45 });
        pL.vline(a, { color: 'ink', dash: false, width: 1.5 });
        pL.legend([{ label: 'method of moments', color: 's2', fill: true }, { label: 'maximum likelihood', color: 's1', fill: true }]);
        const fin = arr => Array.from(arr).filter(Number.isFinite);
        const vm = S.variance(fin(mom)), vl = S.variance(fin(mle)), ratio = vl / vm;
        pR.clear();
        pR.line(aGrid.map(t => [t, areGamma(t)]), { color: 's1', width: 2.2 });
        pR.hline(1, { color: 'faint', width: 1 });
        pR.vline(a, { color: 'muted', width: 1 });
        pR.dot(a, Math.min(1.04, ratio), { color: 'accent', r: 5 });
        pR.text(a, Math.min(1.04, ratio), `simulated, n = ${n}`, { dx: 8, dy: -6, size: 11 });
        sM.set(`${fmt(S.mean(fin(mom)), 2)} ± ${fmt(Math.sqrt(vm), 2)}`, `asymptotic sd ${fmt(Math.sqrt(2 * a * (a + 1) / n), 2)}`);
        sL.set(`${fmt(S.mean(fin(mle)), 2)} ± ${fmt(Math.sqrt(vl), 2)}`, `asymptotic sd ${fmt(Math.sqrt(a / (n * (a * trigamma(a) - 1))), 2)}`);
        sE.set(fmt(ratio, 3), 'Var(MLE)/Var(MoM)');
        sT.set(fmt(areGamma(a), 3), R`$1/[2(\alpha+1)(\alpha\psi'(\alpha)-1)]$`);
        ctx.math(stats);
      }
      draw();
    },
  });

  /* ================================================================
   * likelihood
   * ================================================================ */
  const LMODELS = {
    bern: { label: 'Bernoulli(p), true p = 0.3', truth: 0.3, range: [0.001, 0.999], xl: 'p', sample: r => (r.u() < 0.3 ? 1 : 0), logf: (x, t) => (x ? Math.log(t) : Math.log(1 - t)) },
    pois: { label: 'Poisson(λ), true λ = 3', truth: 3, range: [0.02, 9], xl: 'λ', sample: r => r.poisson(3), logf: (x, t) => x * Math.log(t) - t },
    norm: { label: 'Normal(μ, 1), true μ = 1', truth: 1, range: [-3, 5], xl: 'μ', sample: r => r.normal(1, 1), logf: (x, t) => -0.5 * (x - t) * (x - t) },
    cauchy: { label: 'Cauchy(θ, 1), true θ = 0', truth: 0, range: [-8, 8], xl: 'θ', sample: r => r.cauchy(), logf: (x, t) => -Math.log(1 + (x - t) * (x - t)) },
  };
  const WILKS95 = Math.exp(-3.841458820694124 / 2);

  Atlas.lab('likelihood-accumulate', {
    title: 'Watching a likelihood take shape',
    blurb: R`Observations arrive one at a time; the curve is the relative likelihood $R(\theta)=L(\theta)/L(\hat\theta)$ (or its log). The dashed curve is the quadratic approximation with curvature equal to the observed information $J(\hat\theta)$, and the horizontal line at $e^{-1.92}\approx0.147$ marks the 95% likelihood interval. Try the Cauchy model with few observations — the likelihood can have several peaks — and Bernoulli at small $n$, where the log-likelihood is visibly non-quadratic.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mod = UI.select(controls, { label: 'Model', options: Object.keys(LMODELS).map(k => ({ value: k, label: LMODELS[k].label })), value: 'cauchy', onChange: reset });
      const view = UI.segmented(controls, { label: 'Scale', options: ['Relative likelihood', 'Log-likelihood'], value: 'Relative likelihood', onChange: draw });
      const zoom = UI.toggle(controls, { label: 'Auto-zoom around the MLE', value: true, onChange: draw });
      const speed = UI.slider(controls, { label: 'Observations per second', min: 1, max: 60, step: 1, value: 4 });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Stream data', '❚❚ Pause']);
      UI.button(row, '+1', () => { add(1); draw(); });
      UI.button(row, '+25', () => { add(25); draw(); });
      UI.button(row, 'Reset', reset);
      const p = new Plot(stage, { width: 640, height: 300, xlabel: 'θ', ylabel: 'R(θ)' });
      const sN = UI.stat(stats, 'Observations $n$');
      const sM = UI.stat(stats, R`MLE $\hat\theta$`, { accent: 'accent' });
      const sJ = UI.stat(stats, R`Observed info $J(\hat\theta)$`);
      const sSE = UI.stat(stats, R`$1/\sqrt{J}$`);
      const sLI = UI.stat(stats, '95% likelihood interval');
      const sPk = UI.stat(stats, 'Local maxima');
      let x = [], acc = 0;
      const M = () => LMODELS[mod.get()];
      const ell = t => { const f = M().logf; let s = 0; for (let i = 0; i < x.length; i++) s += f(x[i], t); return s; };
      function add(k) { for (let i = 0; i < k && x.length < 1000; i++) x.push(M().sample(ctx.rng)); }
      function reset() { loop.stop(); x = []; add(2); draw(); }
      function step(dt) { acc += dt * speed.get(); const k = Math.floor(acc); if (k > 0) { acc -= k; add(k); draw(); } if (x.length >= 1000) return false; }
      function draw() {
        const Mo = M(), [r0, r1] = Mo.range;
        // global search on a grid, then golden refinement
        const NG = 500, gg = S.linspace(r0, r1, NG), gl = gg.map(ell);
        let bi = S.argmax(gl);
        const a = gg[Math.max(0, bi - 1)], b = gg[Math.min(NG - 1, bi + 1)];
        const th = S.golden(t => -ell(t), a, b, 1e-10), lmax = ell(th);
        const onEdge = bi === 0 || bi === NG - 1;
        let peaks = 0;
        for (let i = 1; i < NG - 1; i++) if (gl[i] > gl[i - 1] && gl[i] >= gl[i + 1]) peaks++;
        const hh = 1e-4 * (r1 - r0);
        const J = onEdge ? NaN : -(ell(th + hh) - 2 * lmax + ell(th - hh)) / (hh * hh);
        let lo = r0, hi = r1;
        if (zoom.get() && J > 0) { lo = Math.max(r0, th - 6 / Math.sqrt(J)); hi = Math.min(r1, th + 6 / Math.sqrt(J)); }
        const g = S.linspace(lo, hi, 360), lv = g.map(t => ell(t) - lmax);
        const isLog = view.get() === 'Log-likelihood';
        const floor = -12;
        p.o.xlabel = Mo.xl; p.o.ylabel = isLog ? 'ℓ(θ) − ℓ(θ̂)' : 'R(θ)';
        p.setDomain([lo, hi], isLog ? [floor, 0.6] : [0, 1.08]).clear();
        // likelihood interval (from the fine grid)
        let segs = 0, first = null, last = null, inside = false;
        lv.forEach((v, i) => { const ok = v >= Math.log(WILKS95); if (ok && !inside) segs++; if (ok) { if (first == null) first = g[i]; last = g[i]; } inside = ok; });
        if (first != null) p.band(first, last, { color: 's1', opacity: 0.08 });
        p.hline(isLog ? Math.log(WILKS95) : WILKS95, { color: 'muted', width: 1 });
        if (J > 0) p.line(g.map(t => { const q = -0.5 * J * (t - th) ** 2; return [t, isLog ? Math.max(q, floor - 1) : Math.exp(q)]; }), { color: 's2', width: 1.6, dash: true });
        p.line(g.map((t, i) => [t, isLog ? Math.max(lv[i], floor - 1) : Math.exp(lv[i])]), { color: 's1', width: 2.4 });
        p.vline(Mo.truth, { color: 'ink', dash: false, width: 1.2 });
        p.dot(th, isLog ? 0 : 1, { color: 'accent', r: 4.5 });
        p.text(Mo.truth, isLog ? 0.45 : 1.04, 'truth', { dx: 4, size: 11, color: 'muted' });
        // rug of data within view
        for (const v of x) if (v >= lo && v <= hi) p.segment(v, isLog ? floor : 0, v, isLog ? floor + 0.5 : 0.04, { color: 's3', width: 1.2, opacity: 0.7 });
        p.legend([{ label: 'likelihood', color: 's1' }, { label: 'quadratic approx.', color: 's2', dash: true }, { label: 'data (rug)', color: 's3' }], { left: true });
        sN.set(String(x.length));
        sM.set(fmt(th, 3), onEdge ? 'at the edge of Θ: no interior maximum' : '');
        sJ.set(J > 0 ? fmt(J, 2) : '—');
        sSE.set(J > 0 ? fmt(1 / Math.sqrt(J), 3) : '—', J > 0 ? `Wald: [${fmt(th - 1.96 / Math.sqrt(J), 2)}, ${fmt(th + 1.96 / Math.sqrt(J), 2)}]` : '');
        sLI.set(first != null ? `[${fmt(first, 2)}, ${fmt(last, 2)}]` : '—', segs > 1 ? `${segs} disjoint pieces` : (zoom.get() ? 'within the zoomed view' : ''));
        sPk.set(String(peaks), peaks > 1 ? 'multimodal likelihood' : '');
      }
      reset();
    },
  });

  Atlas.lab('likelihood-principle', {
    title: 'Same likelihood, different p-values',
    blurb: R`Design A fixes $n$ trials and counts $r$ successes (binomial); design B samples until the $r$-th success, which arrives on trial $n$ (negative binomial). The likelihoods (top left) are identical, so the likelihood principle says the evidence is the same. The one-sided p-values for $H_0: p=p_0$ vs $p \lt p_0$ (top right) differ because they sum over different unobserved outcomes (bottom).`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rS = UI.slider(controls, { label: R`Successes $r$`, min: 1, max: 12, step: 1, value: 3, onInput: draw });
      const nS = UI.slider(controls, { label: R`Trials $n$`, min: 2, max: 40, step: 1, value: 12, onInput: draw });
      const p0S = UI.slider(controls, { label: R`Null value $p_0$`, min: 0.05, max: 0.95, step: 0.01, value: 0.5, onInput: draw });
      stage.classList.add('two');
      const pLik = new Plot(stage, { width: 420, height: 240, x: [0, 1], y: [0, 1.08], xlabel: 'p', ylabel: 'relative likelihood', title: 'Likelihoods: A (solid) and B (dashed)' });
      const pPv = new Plot(stage, { width: 420, height: 240, x: [0, 1], y: [0, 1], xlabel: 'p₀', ylabel: 'one-sided p-value', title: 'p-value as a function of p₀' });
      const pA = new Plot(stage, { width: 420, height: 220, xlabel: 'successes X in n trials', ylabel: 'P(X = x)', title: 'A: binomial null distribution' });
      const pB = new Plot(stage, { width: 420, height: 220, xlabel: 'trials N to the r-th success', ylabel: 'P(N = k)', title: 'B: negative binomial null distribution' });
      const sA = UI.stat(stats, R`p-value A: $\P(X\le r)$`, { accent: 'accent' });
      const sB = UI.stat(stats, R`p-value B: $\P(N\ge n)$`, { accent: 'accent' });
      const sM = UI.stat(stats, R`MLE (both) $r/n$`);
      const note = UI.note(stats, '');
      function draw() {
        const r = Math.round(rS.get()), n = Math.max(Math.round(nS.get()), r + 1), p0 = p0S.get();
        const pvA = q => S.cdf.binom(r, n, q), pvB = q => S.cdf.binom(r - 1, n - 1, q);
        const ph = r / n, lmax = r * Math.log(ph) + (n - r) * Math.log(1 - ph);
        const lik = q => Math.exp(r * Math.log(q) + (n - r) * Math.log(1 - q) - lmax);
        pLik.clear();
        pLik.line(curve(0.0005, 0.9995, lik, 300), { color: 's1', width: 3 });
        pLik.line(curve(0.0005, 0.9995, lik, 300), { color: 's2', width: 2, dash: true });
        pLik.vline(p0, { color: 'muted', width: 1 });
        pPv.clear();
        pPv.hline(0.05, { color: 'muted', width: 1 });
        pPv.line(curve(0.01, 0.99, pvA, 200), { color: 's1', width: 2.2 });
        pPv.line(curve(0.01, 0.99, pvB, 200), { color: 's2', width: 2.2 });
        pPv.vline(p0, { color: 'muted', width: 1 });
        pPv.dot(p0, pvA(p0), { color: 's1', r: 4 }); pPv.dot(p0, pvB(p0), { color: 's2', r: 4 });
        pPv.legend([{ label: 'A (binomial)', color: 's1' }, { label: 'B (neg. binomial)', color: 's2' }], { left: true });
        const pmfA = S.range(n + 1).map(k => [k, S.pmf.binom(k, n, p0)]);
        const yA = Math.max(...pmfA.map(q => q[1])) * 1.15;
        pA.setDomain([-0.7, n + 0.7], [0, yA]).clear();
        pA.stems(pmfA.filter(q => q[0] > r), { color: 'faint' });
        pA.stems(pmfA.filter(q => q[0] <= r), { color: 's1' });
        const pmfNB = k => Math.exp(S.lchoose(k - 1, r - 1) + r * Math.log(p0) + (k - r) * Math.log(1 - p0));
        let kmax = n + 3; while (kmax < 400 && S.cdf.binom(r - 1, kmax, p0) > 0.002) kmax++;
        const ptsB = []; for (let k = r; k <= kmax; k++) ptsB.push([k, pmfNB(k)]);
        const yB = Math.max(...ptsB.map(q => q[1])) * 1.15;
        pB.setDomain([r - 0.7, kmax + 0.7], [0, yB]).clear();
        pB.stems(ptsB.filter(q => q[0] < n), { color: 'faint', r: 2.5 });
        pB.stems(ptsB.filter(q => q[0] >= n), { color: 's2', r: 2.5 });
        const a = pvA(p0), b = pvB(p0);
        sA.set(fmt(a, 4), a < 0.05 ? 'reject at 5%' : 'do not reject');
        sB.set(fmt(b, 4), b < 0.05 ? 'reject at 5%' : 'do not reject');
        sM.set(fmt(ph, 3), Math.round(nS.get()) <= r ? `n raised to ${n} (need n > r)` : 'identical in A and B');
        note.set((a < 0.05) !== (b < 0.05) ? '<b>The two designs reach opposite conclusions at the 5% level</b> from proportional likelihoods.' : 'Both designs agree at the 5% level here; the p-values still differ. Try $r=3$, $n=12$, $p_0=0.5$.');
      }
      draw();
    },
  });

  /* ================================================================
   * sufficiency
   * ================================================================ */
  Atlas.lab('sufficiency-conditional', {
    title: 'Conditioning on a sufficient statistic removes the parameter',
    blurb: R`Poisson samples of size $n$. Left: the exact conditional distribution of $X_1$ given the sufficient statistic $\sum X_i = t$, computed under two different values $\lambda_1,\lambda_2$ — the two are identical (Binomial$(t,1/n)$), whatever the $\lambda$'s. Right: the conditional distribution of $X_1$ given the maximum $=m$, which is not sufficient: the two laws differ, so the "leftover" data still carry information about $\lambda$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 2, max: 8, step: 1, value: 4, onInput: draw });
      const l1 = UI.slider(controls, { label: R`$\lambda_1$`, min: 0.2, max: 8, step: 0.1, value: 1, onInput: draw });
      const l2 = UI.slider(controls, { label: R`$\lambda_2$`, min: 0.2, max: 8, step: 0.1, value: 5, onInput: draw });
      const tS = UI.slider(controls, { label: R`Condition on $\sum X_i = t$`, min: 0, max: 30, step: 1, value: 8, onInput: draw });
      const mS = UI.slider(controls, { label: R`Condition on $\max X_i = m$`, min: 0, max: 12, step: 1, value: 4, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'k', ylabel: 'P(X₁ = k | T = t)', title: 'Given the sum (sufficient)' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'k', ylabel: 'P(X₁ = k | max = m)', title: 'Given the maximum (not sufficient)' });
      const sT = UI.stat(stats, 'TV distance, given the sum', { accent: 'good' });
      const sM = UI.stat(stats, 'TV distance, given the max', { accent: 'bad' });
      const sE = UI.stat(stats, R`$\E[X_1\mid\max=m]$ under $\lambda_1$ vs $\lambda_2$`);
      function condMax(n, m, lam) {
        const F = k => (k < 0 ? 0 : S.cdf.poisson(k, lam));
        const out = [];
        for (let k = 0; k <= m; k++) out.push(S.pmf.poisson(k, lam) * (k < m ? F(m) ** (n - 1) - F(m - 1) ** (n - 1) : F(m) ** (n - 1)));
        const z = S.sum(out);
        return out.map(v => v / z);
      }
      function draw() {
        const n = Math.round(nS.get()), t = Math.round(tS.get()), m = Math.round(mS.get());
        const cs = S.range(t + 1).map(k => S.pmf.binom(k, t, 1 / n)); // identical under both lambdas
        const c1 = S.range(t + 1).map(k => S.pmf.poisson(k, l1.get()) * S.pmf.poisson(t - k, (n - 1) * l1.get()) / S.pmf.poisson(t, n * l1.get()));
        const c2 = S.range(t + 1).map(k => S.pmf.poisson(k, l2.get()) * S.pmf.poisson(t - k, (n - 1) * l2.get()) / S.pmf.poisson(t, n * l2.get()));
        const ymaxL = Math.max(...c1, ...c2, ...cs) * 1.15 || 1;
        pL.setDomain([-0.8, t + 0.8], [0, ymaxL]).clear();
        pL.stems(c1.map((v, k) => [k - 0.13, v]), { color: 's1' });
        pL.stems(c2.map((v, k) => [k + 0.13, v]), { color: 's2' });
        pL.legend([{ label: `λ₁ = ${l1.get().toFixed(1)}`, color: 's1' }, { label: `λ₂ = ${l2.get().toFixed(1)}`, color: 's2' }]);
        const d1 = condMax(n, m, l1.get()), d2 = condMax(n, m, l2.get());
        const ymaxR = Math.max(...d1, ...d2) * 1.15 || 1;
        pR.setDomain([-0.8, m + 0.8], [0, ymaxR]).clear();
        pR.stems(d1.map((v, k) => [k - 0.13, v]), { color: 's1' });
        pR.stems(d2.map((v, k) => [k + 0.13, v]), { color: 's2' });
        pR.legend([{ label: `λ₁ = ${l1.get().toFixed(1)}`, color: 's1' }, { label: `λ₂ = ${l2.get().toFixed(1)}`, color: 's2' }]);
        sT.set(fmt(tv(c1, c2), 4), `both equal Binomial(${t}, 1/${n})`);
        sM.set(fmt(tv(d1, d2), 4), 'depends on λ');
        const e = d => d.reduce((s, v, k) => s + k * v, 0);
        sE.set(`${fmt(e(d1), 2)} vs ${fmt(e(d2), 2)}`);
      }
      draw();
    },
  });

  const SUFF = {
    norm: {
      label: 'Normal(μ, 1)', n0: 8, grid: [-1.5, 2.5], xl: 'μ', truth: 0.5,
      sample: (r, n) => r.array(n, () => r.normal(0.5, 1)),
      T: { name: 'mean', f: x => S.mean(x) }, U: { name: 'median', f: x => S.median(x) },
      matchT(r, a) { const b = this.sample(r, a.length), d = S.mean(a) - S.mean(b); return b.map(v => v + d); },
      matchU(r, a) { const b = this.sample(r, a.length), d = S.median(a) - S.median(b); return b.map(v => v + d); },
      logL: (x, t) => { let s = 0; for (const v of x) s -= 0.5 * (v - t) ** 2; return s; },
    },
    pois: {
      label: 'Poisson(λ)', n0: 8, grid: [0.05, 6], xl: 'λ', truth: 2,
      sample: (r, n) => r.array(n, () => r.poisson(2)),
      T: { name: 'sum', f: x => S.sum(x) }, U: { name: 'max', f: x => Math.max(...x) },
      matchT(r, a) { const t = S.sum(a), b = new Array(a.length).fill(0); for (let i = 0; i < t; i++) b[(r.u() * a.length) | 0]++; return b; },
      matchU(r, a) {
        const m = Math.max(...a), b = [m];
        while (b.length < a.length) { let v; let g = 0; do { v = r.poisson(2); } while (v > m && ++g < 1000); b.push(Math.min(v, m)); }
        return r.shuffle(b);
      },
      logL: (x, t) => { let s = 0; for (const v of x) s += v * Math.log(t) - t; return s; },
    },
    unif: {
      label: 'Uniform(0, θ)', n0: 8, grid: [0.3, 3.5], xl: 'θ', truth: 2,
      sample: (r, n) => r.array(n, () => r.uniform(0, 2)),
      T: { name: 'max', f: x => Math.max(...x) }, U: { name: 'mean', f: x => S.mean(x) },
      matchT(r, a) { const m = Math.max(...a), b = [m]; while (b.length < a.length) b.push(r.uniform(0, m)); return r.shuffle(b); },
      matchU(r, a) { const b = this.sample(r, a.length), k = S.mean(a) / S.mean(b); return b.map(v => v * k); },
      logL: (x, t) => (Math.max(...x) <= t ? -x.length * Math.log(t) : -Infinity),
    },
  };

  Atlas.lab('sufficiency-same-likelihood', {
    title: 'Different data, same sufficient statistic, same likelihood',
    blurb: R`Data set A (top row) is drawn from the model. Data set B (bottom row) is a different random sample constructed to share one statistic with A. When the shared statistic is sufficient, the two likelihood functions coincide exactly — B is A with the irrelevant noise redrawn. When it is not sufficient, the likelihoods differ. Press "New B" repeatedly.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sc = UI.segmented(controls, { label: 'Model', options: [{ value: 'norm', label: 'Normal' }, { value: 'pois', label: 'Poisson' }, { value: 'unif', label: 'Uniform' }], value: 'norm', onChange: newA });
      const share = UI.segmented(controls, { label: 'B shares with A', options: [{ value: 'T', label: 'sufficient statistic' }, { value: 'U', label: 'non-sufficient statistic' }], value: 'T', onChange: newB });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 3, max: 30, step: 1, value: 8, onChange: newA });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New B', newB, { primary: true });
      UI.button(row, 'New A', newA);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, y: [-0.6, 1.9], yticks: 0, xlabel: 'x', title: 'Data: A (top) and B (bottom)' });
      const pL = new Plot(stage, { width: 420, height: 270, y: [0, 1.08], xlabel: 'θ', ylabel: 'relative likelihood', title: 'Likelihood functions' });
      const sTA = UI.stat(stats, 'Shared statistic (A, B)');
      const sOther = UI.stat(stats, 'Other statistic (A, B)');
      const sDiff = UI.stat(stats, R`$\max_\theta |R_A(\theta)-R_B(\theta)|$`, { accent: 'accent' });
      const sMle = UI.stat(stats, 'MLE (A, B)');
      let A = [], B = [];
      function newA() { const M = SUFF[sc.get()]; A = M.sample(ctx.rng, Math.round(nS.get())); newB(); }
      function newB() { const M = SUFF[sc.get()]; B = share.get() === 'T' ? M.matchT(ctx.rng, A) : M.matchU(ctx.rng, A); draw(); }
      function rel(M, x, g) { const l = g.map(t => M.logL(x, t)), m = Math.max(...l); return l.map(v => (Number.isFinite(v) ? Math.exp(v - m) : 0)); }
      function draw() {
        const M = SUFF[sc.get()];
        const all = A.concat(B), lo = Math.min(...all), hi = Math.max(...all);
        const pad = (hi - lo) * 0.08 + 0.3;
        pD.setDomain([Math.min(lo, 0) - pad * (sc.get() === 'norm' ? 1 : 0.3), hi + pad], [-0.6, 1.9]).clear();
        const strip = (x, y0, col) => { const cnt = {}; for (const v of x) { const k = v.toFixed(6); cnt[k] = (cnt[k] || 0) + 1; pD.dot(v, y0 + 0.13 * (cnt[k] - 1), { color: col, r: 4.5 }); } };
        strip(A, 1.05, 's1'); strip(B, 0, 's2');
        pD.text(pD.o.x[0], 1.05, 'A', { dx: 6, dy: -10, weight: 600 });
        pD.text(pD.o.x[0], 0, 'B', { dx: 6, dy: -10, weight: 600 });
        const g = S.linspace(M.grid[0], M.grid[1], 400);
        const ra = rel(M, A, g), rb = rel(M, B, g);
        pL.o.xlabel = M.xl;
        pL.setDomain(M.grid, [0, 1.08]).clear();
        pL.line(g.map((t, i) => [t, ra[i]]), { color: 's1', width: 3.5, opacity: 0.6 });
        pL.line(g.map((t, i) => [t, rb[i]]), { color: 's2', width: 2, dash: true });
        pL.vline(M.truth, { color: 'muted', width: 1 });
        pL.legend([{ label: 'likelihood from A', color: 's1' }, { label: 'likelihood from B', color: 's2', dash: true }]);
        const shared = share.get() === 'T' ? M.T : M.U, other = share.get() === 'T' ? M.U : M.T;
        sTA.set(`${fmt(shared.f(A), 3)}, ${fmt(shared.f(B), 3)}`, shared.name + (share.get() === 'T' ? ' (sufficient)' : ' (not sufficient)'));
        sOther.set(`${fmt(other.f(A), 3)}, ${fmt(other.f(B), 3)}`, other.name);
        let d = 0; for (let i = 0; i < g.length; i++) d = Math.max(d, Math.abs(ra[i] - rb[i]));
        sDiff.set(fmt(d, 4), d < 1e-6 ? 'identical likelihoods' : 'likelihoods differ');
        sMle.set(`${fmt(g[S.argmax(ra)], 2)}, ${fmt(g[S.argmax(rb)], 2)}`, 'grid maximizers');
      }
      newA();
    },
  });

  /* ================================================================
   * exponential-families
   * ================================================================ */
  const softplus = e => (e > 0 ? e + Math.log1p(Math.exp(-e)) : Math.log1p(Math.exp(e)));
  const EFAM = {
    bern: { label: 'Bernoulli', eta: [-5, 5], A: softplus, A1: e => 1 / (1 + Math.exp(-e)), A2: e => { const q = 1 / (1 + Math.exp(-e)); return q * (1 - q); },
      sample: (r, e) => (r.u() < 1 / (1 + Math.exp(-e)) ? 1 : 0), inv: t => Math.log(t / (1 - t)), interior: t => t > 0 && t < 1, discrete: true, support: () => [0, 1], pmf: (k, e) => { const q = 1 / (1 + Math.exp(-e)); return k ? q : 1 - q; }, xr: () => [-0.6, 1.6] },
    pois: { label: 'Poisson', eta: [-2, 2.5], A: Math.exp, A1: Math.exp, A2: Math.exp,
      sample: (r, e) => r.poisson(Math.exp(e)), inv: Math.log, interior: t => t > 0, discrete: true, support: e => S.range(Math.ceil(Math.exp(e) + 5 * Math.sqrt(Math.exp(e)) + 3)), pmf: (k, e) => S.pmf.poisson(k, Math.exp(e)), xr: e => [-0.6, Math.exp(e) + 5 * Math.sqrt(Math.exp(e)) + 3] },
    expo: { label: 'Exponential', eta: [-4, -0.25], A: e => -Math.log(-e), A1: e => -1 / e, A2: e => 1 / (e * e),
      sample: (r, e) => r.exp(-e), inv: t => -1 / t, interior: t => t > 0, discrete: false, pdf: (x, e) => S.pdf.exp(x, -e), xr: e => [0, 6 / -e] },
    norm: { label: 'Normal (σ = 1)', eta: [-3, 3], A: e => e * e / 2, A1: e => e, A2: () => 1,
      sample: (r, e) => r.normal(e, 1), inv: t => t, interior: () => true, discrete: false, pdf: (x, e) => S.pdf.normal(x, e, 1), xr: e => [e - 4, e + 4] },
  };

  Atlas.lab('exponential-families-cumulant', {
    title: 'The cumulant function: slope = mean, curvature = variance',
    blurb: R`Left: the cumulant function $A(\eta)$ of a one-parameter family. At the chosen $\eta$, the tangent's slope is $\E_\eta T = A'(\eta)$ and the osculating parabola's curvature is $\Var_\eta T = A''(\eta)$. Turn on "fit a sample" to see maximum likelihood as moment matching: $\hat\eta$ is the point where the slope of $A$ equals $\bar T$. With Bernoulli or Poisson and small $n$, $\bar T$ sometimes lands on the boundary of the support, and no slope matches — the MLE does not exist.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fam = UI.segmented(controls, { label: 'Family', options: Object.keys(EFAM).map(k => ({ value: k, label: EFAM[k].label })), value: 'pois', onChange: () => { resample(); } });
      const F = () => EFAM[fam.get()];
      const etaOf = u => { const [a, b] = F().eta; return a + (b - a) * u; };
      const uS = UI.slider(controls, { label: R`Natural parameter $\eta$`, min: 0, max: 1, step: 0.005, value: 0.45, fmt: u => etaOf(u).toFixed(2), onInput: draw, onChange: resample });
      const fit = UI.toggle(controls, { label: 'Fit a sample drawn at this η', value: true, onChange: draw });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 1, max: 50, step: 1, value: 5, onChange: resample });
      UI.button(controls, 'New sample', resample, { primary: true });
      stage.classList.add('two');
      const pA = new Plot(stage, { width: 420, height: 280, xlabel: 'η', ylabel: 'A(η)', title: 'Cumulant function' });
      const pD = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'probability / density', title: 'Distribution at η' });
      const sEta = UI.stat(stats, R`$A(\eta)$`);
      const sMu = UI.stat(stats, R`$A'(\eta) = \E T$`, { accent: 'accent' });
      const sV = UI.stat(stats, R`$A''(\eta) = \Var T$`);
      const sTb = UI.stat(stats, R`Sample mean $\bar T$`);
      const sHat = UI.stat(stats, R`MLE $\hat\eta$`);
      let x = [];
      function resample() { const e = etaOf(uS.get()); x = ctx.rng.array(Math.round(nS.get()), () => F().sample(ctx.rng, e)); draw(); }
      function draw() {
        const Fm = F(), e = etaOf(uS.get()), [a, b] = Fm.eta;
        const g = S.linspace(a, b, 300), Av = g.map(Fm.A);
        const ylo = Math.min(...Av), yhi = Math.max(...Av), pad = (yhi - ylo) * 0.08;
        pA.setDomain([a, b], [ylo - pad, yhi + pad]).clear();
        pA.line(g.map((t, i) => [t, Av[i]]), { color: 'ink', width: 2.2 });
        const A0 = Fm.A(e), A1 = Fm.A1(e), A2 = Fm.A2(e);
        pA.line(g.map(t => [t, A0 + A1 * (t - e)]), { color: 's1', width: 1.8 });
        pA.line(g.map(t => [t, A0 + A1 * (t - e) + 0.5 * A2 * (t - e) ** 2]), { color: 's2', width: 1.5, dash: true });
        pA.dot(e, A0, { color: 's1', r: 5 });
        const tb = x.length ? S.mean(x) : NaN;
        let etaHat = NaN;
        if (fit.get() && x.length && Fm.interior(tb)) {
          etaHat = Fm.inv(tb);
          if (etaHat >= a && etaHat <= b) {
            const Ah = Fm.A(etaHat);
            pA.line(g.map(t => [t, Ah + tb * (t - etaHat)]), { color: 's3', width: 1.6 });
            pA.dot(etaHat, Ah, { color: 's3', r: 5 });
          }
        }
        pA.legend([{ label: 'A(η)', color: 'ink' }, { label: 'tangent: slope E T', color: 's1' }, { label: 'curvature Var T', color: 's2', dash: true }].concat(fit.get() ? [{ label: 'slope T̄ (MLE)', color: 's3' }] : []), { left: true });
        // distribution panel
        const [xl, xh] = Fm.xr(e);
        if (Fm.discrete) {
          const sup = Fm.support(e), pts = sup.map(k => [k, Fm.pmf(k, e)]);
          const ymax = Math.max(...pts.map(q => q[1])) * 1.2;
          pD.setDomain([xl, xh], [0, ymax]).clear();
          pD.stems(pts, { color: 's1' });
          if (fit.get() && x.length) { const cnt = {}; for (const v of x) cnt[v] = (cnt[v] || 0) + 1; for (const k in cnt) pD.dot(+k + 0.18, cnt[k] / x.length, { color: 's3', r: 4 }); }
        } else {
          const gx = S.linspace(xl, xh, 300), pv = gx.map(v => Fm.pdf(v, e));
          const ymax = Math.max(...pv) * 1.2;
          pD.setDomain([xl, xh], [0, ymax]).clear();
          pD.area(gx.map((v, i) => [v, pv[i]]), { color: 's1', opacity: 0.2 });
          pD.line(gx.map((v, i) => [v, pv[i]]), { color: 's1', width: 2 });
          if (fit.get()) for (const v of x) pD.segment(v, 0, v, ymax * 0.06, { color: 's3', width: 1.5 });
        }
        const sd = Math.sqrt(A2);
        pD.band(A1 - sd, A1 + sd, { color: 's2', opacity: 0.1 });
        pD.vline(A1, { color: 's1', dash: false, width: 1.5 });
        if (fit.get() && x.length) pD.vline(tb, { color: 's3', width: 1.5 });
        sEta.set(fmt(A0, 3), `η = ${fmt(e, 2)}`);
        sMu.set(fmt(A1, 3));
        sV.set(fmt(A2, 3), `sd ${fmt(sd, 3)}`);
        sTb.set(fit.get() && x.length ? fmt(tb, 3) : '—', fit.get() ? `n = ${x.length}` : '');
        sHat.set(fit.get() ? (Number.isFinite(etaHat) ? fmt(etaHat, 3) : 'does not exist') : '—', fit.get() && !Number.isFinite(etaHat) ? 'T̄ is on the boundary of the convex support' : '');
      }
      resample();
    },
  });

  const TILT = {
    bimodal: { label: 'Bimodal mixture', xr: [-5, 5], comps: [[0.5, -1.5, 0.6], [0.5, 1.5, 0.6]] },
    normal: { label: 'Standard normal', xr: [-5, 5] },
    uniform: { label: 'Uniform(0, 1)', xr: [-0.05, 1.05] },
    expo: { label: 'Exponential(1)', xr: [0, 8] },
  };
  function tiltA(key, e) {
    if (key === 'bimodal') return Math.log(S.sum(TILT.bimodal.comps.map(([w, m, s]) => w * Math.exp(e * m + 0.5 * e * e * s * s))));
    if (key === 'normal') return e * e / 2;
    if (key === 'uniform') return Math.abs(e) < 1e-8 ? e / 2 : Math.log(Math.expm1(e) / e);
    return e < 1 ? -Math.log(1 - e) : Infinity;
  }
  function tiltBase(key, x) {
    if (key === 'bimodal') return S.sum(TILT.bimodal.comps.map(([w, m, s]) => w * S.pdf.normal(x, m, s)));
    if (key === 'normal') return S.pdf.normal(x);
    if (key === 'uniform') return x >= 0 && x <= 1 ? 1 : 0;
    return x >= 0 ? Math.exp(-x) : 0;
  }

  Atlas.lab('exponential-families-tilting', {
    title: 'Exponential tilting of a base measure',
    blurb: R`Any base density $h$ generates an exponential family $f_\eta(x) = h(x)e^{\eta x - A(\eta)}$. Left: the base (grey) and the tilted density (blue). Right: the mean $A'(\eta)$ as a function of $\eta$, strictly increasing because $A''(\eta)=\Var_\eta X>0$. The tilted normal stays normal; the tilted uniform becomes a truncated exponential; the exponential base can only be tilted for $\eta \lt 1$ — beyond that the integral diverges and $\eta$ leaves the natural parameter space.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const base = UI.select(controls, { label: R`Base density $h$`, options: Object.keys(TILT).map(k => ({ value: k, label: TILT[k].label })), value: 'bimodal', onChange: draw });
      const eS = UI.slider(controls, { label: R`Tilt $\eta$`, min: -3, max: 3, step: 0.01, value: 0.6, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'density', title: 'Base h and tilted f_η' });
      const pR = new Plot(stage, { width: 420, height: 280, x: [-3, 3], xlabel: 'η', ylabel: 'mean A′(η)', title: 'Mean map η ↦ E_η X' });
      const sA = UI.stat(stats, R`$A(\eta)$`);
      const sM = UI.stat(stats, R`Mean $A'(\eta)$`, { accent: 'accent' });
      const sV = UI.stat(stats, R`Variance $A''(\eta)$`);
      const sK = UI.stat(stats, R`$\KL(f_\eta\,\|\,h) = \eta A'(\eta)-A(\eta)$`);
      function draw() {
        const key = base.get(), e = eS.get(), [x0, x1] = TILT[key].xr;
        const A = t => tiltA(key, t), h = 1e-4;
        const A0 = A(e), d1 = (A(e + h) - A(e - h)) / (2 * h), d2 = (A(e + h) - 2 * A0 + A(e - h)) / (h * h);
        const g = S.linspace(x0, x1, 400), hb = g.map(v => tiltBase(key, v));
        const ok = Number.isFinite(A0) && Number.isFinite(A(e + h));
        const fv = ok ? g.map((v, i) => hb[i] * Math.exp(e * v - A0)) : g.map(() => 0);
        const ymax = Math.max(...hb, ...fv) * 1.12;
        pL.setDomain([x0, x1], [0, Math.min(ymax, 4)]).clear();
        pL.area(g.map((v, i) => [v, hb[i]]), { color: 'faint', opacity: 0.45 });
        if (ok) { pL.area(g.map((v, i) => [v, fv[i]]), { color: 's1', opacity: 0.18 }); pL.line(g.map((v, i) => [v, fv[i]]), { color: 's1', width: 2.2 }); pL.vline(d1, { color: 's1', width: 1.4 }); }
        else pL.text((x0 + x1) / 2, ymax * 0.6, 'η is outside the natural parameter space', { anchor: 'middle', color: 'bad', weight: 600 });
        pL.legend([{ label: 'base h', color: 'faint', fill: true }, { label: 'tilted f_η', color: 's1' }]);
        const ge = S.linspace(-3, 3, 301).filter(t => Number.isFinite(A(t + h)));
        const mv = ge.map(t => [t, (A(t + h) - A(t - h)) / (2 * h)]);
        const ys = mv.map(q => q[1]).filter(Number.isFinite);
        const ylo = Math.min(...ys), yhi = Math.min(Math.max(...ys), key === 'expo' ? 12 : Infinity);
        pR.setDomain([-3, 3], [ylo - 0.05 * (yhi - ylo), yhi + 0.05 * (yhi - ylo)]).clear();
        if (key === 'expo') { pR.band(1, 3, { color: 'bad', opacity: 0.08 }); pR.text(2, yhi, 'outside 𝒩', { anchor: 'middle', dy: 14, size: 11, color: 'bad' }); }
        pR.line(mv, { color: 's1', width: 2.2 });
        if (ok) pR.dot(e, d1, { color: 'accent', r: 5 });
        sA.set(ok ? fmt(A0, 3) : '∞');
        sM.set(ok ? fmt(d1, 3) : '—');
        sV.set(ok ? fmt(d2, 3) : '—');
        sK.set(ok ? fmt(e * d1 - A0, 4) : '—', 'smallest KL from h among laws with this mean');
      }
      draw();
    },
  });

  /* ================================================================
   * mle
   * ================================================================ */
  const AMODELS = {
    bern5: { label: 'Bernoulli, p = 0.5', th: 0.5, est: (r, n) => r.binomial(n, 0.5) / n, I: t => 1 / (t * (1 - t)) },
    bern03: { label: 'Bernoulli, p = 0.03 (near boundary)', th: 0.03, est: (r, n) => r.binomial(n, 0.03) / n, I: t => 1 / (t * (1 - t)) },
    pois: { label: 'Poisson, λ = 4', th: 4, est: (r, n) => r.poisson(4 * n) / n, I: t => 1 / t },
    exp: { label: 'Exponential rate, λ = 1', th: 1, est: (r, n) => n / r.gamma(n, 1), I: t => 1 / (t * t) },
    cauchy: { label: 'Cauchy location, θ = 0', th: 0, cauchy: true, I: () => 0.5 },
    unif: { label: 'Uniform(0, θ), θ = 1 (non-regular)', th: 1, est: (r, n) => r.u() ** (1 / n), I: null },
  };

  Atlas.lab('mle-asymptotics', {
    title: 'Is the MLE really normal with variance 1/(nI)?',
    blurb: R`Simulate many data sets, compute the MLE in each, and standardize: $Z = \sqrt{nI(\theta_0)}\,(\hat\theta-\theta_0)$. Left: the histogram of $Z$ against $\N(0,1)$; right: a normal QQ plot. The last tile is the actual coverage of the nominal-95% Wald interval $\hat\theta\pm1.96/\sqrt{nI(\hat\theta)}$. Break it: a Bernoulli near the boundary at small $n$, or the Uniform, whose support moves with $\theta$ (there $Z$ is standardized by its simulated sd, and the limit is not normal at all).`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mod = UI.select(controls, { label: 'Model', options: Object.keys(AMODELS).map(k => ({ value: k, label: AMODELS[k].label })), value: 'exp', onChange: draw });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 5, max: 1000, value: 10, log: true, fmt: v => String(Math.round(v)), onChange: draw });
      UI.button(controls, 'Resimulate', draw, { primary: true });
      stage.classList.add('two');
      const pH = new Plot(stage, { width: 420, height: 280, x: [-4.5, 4.5], xlabel: 'standardized MLE Z', ylabel: 'density', title: 'Sampling distribution vs N(0, 1)' });
      const pQ = new Plot(stage, { width: 420, height: 280, x: [-3.6, 3.6], y: [-4.5, 4.5], xlabel: 'normal quantile', ylabel: 'sample quantile of Z', title: 'Normal QQ plot' });
      const sMean = UI.stat(stats, R`Mean of $Z$`);
      const sSd = UI.stat(stats, R`SD of $Z$`);
      const sSk = UI.stat(stats, 'Skewness of Z');
      const sCov = UI.stat(stats, 'Wald 95% coverage', { accent: 'accent' });
      const sB = UI.stat(stats, 'MLE on the boundary');
      function draw() {
        const M = AMODELS[mod.get()], n = Math.round(nS.get()), r = ctx.rng, th = M.th;
        const reps = M.cauchy ? Math.max(150, Math.min(1500, Math.round(2.5e6 / (n * 12)))) : 3000;
        const est = new Float64Array(reps);
        if (M.cauchy) {
          const x = new Float64Array(n);
          for (let k = 0; k < reps; k++) { for (let i = 0; i < n; i++) x[i] = r.cauchy(); est[k] = locMLE(x, 'cauchy', medianSorted(sortNum(x)), 12); }
        } else for (let k = 0; k < reps; k++) est[k] = M.est(r, n);
        let Z, cover = 0, edge = 0;
        if (M.I) {
          Z = est.map(t => Math.sqrt(n * M.I(th)) * (t - th));
          for (const t of est) {
            const se = 1 / Math.sqrt(n * M.I(t));
            if (Number.isFinite(se) && Math.abs(t - th) <= 1.96 * se) cover++;
            if ((mod.get().startsWith('bern') && (t === 0 || t === 1))) edge++;
          }
        } else {
          const sd = S.sd(est), m = S.mean(est);
          Z = est.map(t => (t - m) / sd);
        }
        const h = S.histogram(Z, { bins: 45, lo: -4.5, hi: 4.5 });
        const ymax = Math.max(0.45, ...h.map(b => b.density)) * 1.1;
        pH.setDomain([-4.5, 4.5], [0, Math.min(ymax, 3)]).clear();
        pH.bars(h, { color: 's1', opacity: 0.5 });
        pH.line(curve(-4.5, 4.5, phi), { color: 'ink', width: 2 });
        const zs = sortNum(Z), step = Math.max(1, Math.floor(zs.length / 400)), qq = [];
        for (let i = 0; i < zs.length; i += step) qq.push([S.qnorm((i + 0.5) / zs.length), Math.max(-4.4, Math.min(4.4, zs[i]))]);
        pQ.clear();
        pQ.segment(-3.6, -3.6, 3.6, 3.6, { color: 'muted', width: 1.2, dash: true });
        pQ.points(qq, { color: 's1', r: 2.2 });
        sMean.set(fmt(S.mean(Z), 3), 'ideal 0');
        sSd.set(fmt(S.sd(Z), 3), M.I ? 'ideal 1' : 'standardized by simulation');
        sSk.set(fmt(skew(Array.from(Z)), 3), M.I ? 'ideal 0' : 'limit: −2 (reflected exponential)');
        sCov.set(M.I ? (100 * cover / reps).toFixed(1) + '%' : 'n/a', M.I ? `nominal 95%, ${reps} data sets` : 'no Fisher information');
        sB.set(mod.get().startsWith('bern') ? (100 * edge / reps).toFixed(1) + '%' : '—', mod.get().startsWith('bern') ? 'p̂ = 0 or 1: SE estimate is 0' : '');
      }
      draw();
    },
  });

  Atlas.lab('mle-failures', {
    title: 'Three ways maximum likelihood breaks',
    blurb: R`<b>Neyman–Scott:</b> strata of size $m$, each with its own mean; the MLE of $\sigma^2$ (blue) converges to $\sigma^2(m-1)/m$, not $\sigma^2=1$, however many strata you add. <b>Mixture spike:</b> center one normal component on a data point and shrink its sd: the log-likelihood climbs without bound (slowly — note the log scale), while rounding the data to a finite precision restores a bounded likelihood. <b>Boundary:</b> with $\mu\ge0$ and true $\mu_0=0$, half the sampling distribution of $\hat\mu=\max(\bar X,0)$ is an atom at zero.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sc = UI.segmented(controls, { label: 'Failure', options: [{ value: 'ns', label: 'Neyman–Scott' }, { value: 'mix', label: 'Mixture spike' }, { value: 'bd', label: 'Boundary' }], value: 'ns', onChange: () => { layoutCtl(); draw(); } });
      const gNS = UI.h('div', '', controls), gMix = UI.h('div', '', controls), gBd = UI.h('div', '', controls);
      const kS = UI.slider(gNS, { label: R`Number of strata $K$`, min: 10, max: 5000, value: 1000, log: true, fmt: v => String(Math.round(v)), onChange: draw });
      const mS = UI.slider(gNS, { label: R`Observations per stratum $m$`, min: 2, max: 10, step: 1, value: 2, onChange: draw });
      const jS = UI.slider(gMix, { label: R`Centre $\mu_1$ on data point #`, min: 1, max: 60, step: 1, value: 45, onInput: draw });
      const rd = UI.toggle(gMix, { label: 'Data rounded to 0.1 (interval likelihood)', value: false, onChange: draw });
      const m0S = UI.slider(gBd, { label: R`True $\mu_0$`, min: 0, max: 0.5, step: 0.01, value: 0, onChange: draw });
      const nS = UI.slider(gBd, { label: R`Sample size $n$`, min: 10, max: 500, step: 10, value: 50, onChange: draw });
      UI.button(controls, 'Resample', () => { mixData = null; draw(); }, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, xlabel: 'x', ylabel: 'y' });
      const s1 = UI.stat(stats, ''), s2 = UI.stat(stats, ''), s3 = UI.stat(stats, '');
      const label = (s, t) => { s.el.querySelector('.stat-l').innerHTML = t; ctx.math(s.el); };
      let mixData = null;
      function layoutCtl() { show(gNS, sc.get() === 'ns'); show(gMix, sc.get() === 'mix'); show(gBd, sc.get() === 'bd'); }
      function drawNS() {
        const K = Math.round(kS.get()), m = Math.round(mS.get()), r = ctx.rng;
        let ss = 0; const pts = [], cor = [];
        const every = Math.max(1, Math.floor(K / 600));
        for (let k = 1; k <= K; k++) {
          const mu = r.normal(0, 5); let s = 0, s2 = 0;
          for (let i = 0; i < m; i++) { const v = r.normal(mu, 1); s += v; s2 += v * v; }
          ss += s2 - s * s / m;
          if (k % every === 0 || k === K) { const e = ss / (k * m); pts.push([Math.log10(k), e]); cor.push([Math.log10(k), e * m / (m - 1)]); }
        }
        p.o.xlabel = 'log₁₀ (number of strata K)'; p.o.ylabel = 'estimate of σ²';
        p.setDomain([0, Math.log10(K)], [0, 2]).clear();
        p.hline(1, { color: 'ink', dash: false, width: 1.5 });
        p.hline((m - 1) / m, { color: 's1', width: 1.2 });
        p.line(cor, { color: 's2', width: 2 });
        p.line(pts, { color: 's1', width: 2.2 });
        p.text(Math.log10(K), (m - 1) / m, 'MLE limit (m−1)/m', { anchor: 'end', dy: 16, size: 11, color: 's1' });
        p.text(Math.log10(K), 1, 'true σ² = 1', { anchor: 'end', dy: -6, size: 11 });
        p.legend([{ label: 'MLE σ̂²', color: 's1' }, { label: 'corrected m/(m−1)·σ̂²', color: 's2' }], { left: true });
        label(s1, R`MLE $\hat\sigma^2$`); s1.set(fmt(pts[pts.length - 1][1], 4), `K = ${K} strata, n = ${K * m}`);
        label(s2, R`Corrected $\tfrac{m}{m-1}\hat\sigma^2$`); s2.set(fmt(cor[cor.length - 1][1], 4), 'consistent');
        label(s3, 'Limit of the MLE'); s3.set(fmt((m - 1) / m, 4), 'parameters grow with n');
      }
      function drawMix() {
        if (!mixData) { const r = ctx.rng; mixData = sortNum(r.array(60, () => (r.u() < 0.5 ? r.normal(0, 1) : r.normal(3, 1)))); }
        const x = mixData, mu1 = x[Math.round(jS.get()) - 1], del = 0.1;
        const dens = (s1v, m1) => { let l = 0; for (const v of x) l += Math.log(0.5 * phi((v - m1) / s1v) / s1v + 0.5 * phi(v - 3)); return l; };
        const rnd = (s1v, m1) => {
          let l = 0;
          for (const v of x) {
            const c = Math.round(v / del) * del;
            const a = 0.5 * (Phi((c + del / 2 - m1) / s1v) - Phi((c - del / 2 - m1) / s1v)) + 0.5 * (Phi(c + del / 2 - 3) - Phi(c - del / 2 - 3));
            l += Math.log(Math.max(a, 1e-300)) - Math.log(del);
          }
          return l;
        };
        const f = rd.get() ? rnd : dens;
        const g = S.linspace(-150, 1, 500);
        const cur = g.map(t => [t, f(10 ** t, mu1)]), ref = g.map(t => [t, f(10 ** t, 0)]);
        const loc = g.filter(t => t > -1).map(t => f(10 ** t, mu1));
        const lmLoc = Math.max(...loc);
        const ys = cur.concat(ref).map(q => q[1]).filter(Number.isFinite);
        const ylo = Math.max(Math.min(...ys), lmLoc - 60), yhi = Math.max(...ys);
        p.o.xlabel = 'log₁₀ σ₁ (sd of component 1)'; p.o.ylabel = 'log-likelihood';
        p.setDomain([-150, 1], [ylo, yhi + 0.06 * (yhi - ylo)]).clear();
        p.hline(lmLoc, { color: 'muted', width: 1 });
        p.line(ref, { color: 's3', width: 1.6, dash: true });
        p.line(cur, { color: rd.get() ? 's2' : 's1', width: 2.4 });
        p.legend([{ label: `μ₁ = x(${Math.round(jS.get())}) = ${mu1.toFixed(2)}`, color: rd.get() ? 's2' : 's1' }, { label: 'μ₁ = 0 (true)', color: 's3', dash: true }], { left: true });
        const cross = cur.filter(q => q[0] < -1 && q[1] > lmLoc).pop();
        label(s1, 'Best "sensible" fit'); s1.set(fmt(lmLoc, 2), 'local max with σ₁ > 0.1');
        label(s2, R`$\ell$ at $\sigma_1 = 10^{-150}$`); s2.set(fmt(cur[0][1], 2), rd.get() ? 'bounded: rounding caps each term' : '→ ∞ as σ₁ → 0');
        label(s3, 'Spike beats the sensible fit when'); s3.set(cross ? `σ₁ < 10^${Math.round(cross[0])}` : 'never (on this range)', rd.get() ? '' : 'log-likelihood grows like −log σ₁');
      }
      function drawBd() {
        const m0 = m0S.get(), n = Math.round(nS.get()), r = ctx.rng, reps = 6000, rn = Math.sqrt(n);
        const Z = new Float64Array(reps); let zero = 0, rej = 0, cov = 0;
        for (let k = 0; k < reps; k++) {
          const xb = r.normal(m0, 1 / rn), mh = Math.max(xb, 0);
          Z[k] = rn * mh;
          if (mh === 0) zero++;
          if (n * mh * mh > 3.841458820694124) rej++;
          if (Math.abs(mh - m0) <= 1.96 / rn) cov++;
        }
        const pos = Array.from(Z).filter(v => v > 0);
        const h = S.histogram(pos, { bins: 40, lo: 0, hi: 5 });
        for (const b of h) b.density = b.count / (reps * (b.x1 - b.x0));
        const ymax = Math.max(0.45, zero / reps, ...h.map(b => b.density)) * 1.12;
        p.o.xlabel = '√n · μ̂'; p.o.ylabel = 'density (atom: probability)';
        p.setDomain([-0.6, 5], [0, ymax]).clear();
        p.bars(h, { color: 's1', opacity: 0.5 });
        p.line(curve(-0.6, 5, v => phi(v - rn * m0)), { color: 'ink', width: 1.6, dash: true });
        p.stems([[0, zero / reps]], { color: 'bad', width: 4, r: 5 });
        p.text(0, zero / reps, `atom at 0: ${(100 * zero / reps).toFixed(1)}%`, { dx: 10, dy: 4, size: 12, color: 'bad', weight: 600 });
        p.legend([{ label: 'positive part of √n μ̂', color: 's1', fill: true }, { label: 'unconstrained √n X̄', color: 'ink', dash: true }]);
        label(s1, R`$\P(\hat\mu = 0)$`); s1.set((100 * zero / reps).toFixed(1) + '%', m0 === 0 ? 'theory 50%' : '');
        label(s2, R`LRT of $\mu=0$ with $\chi^2_1$ cutoff`); s2.set((100 * rej / reps).toFixed(1) + '%', m0 === 0 ? 'nominal 5%; true limit ½χ²₀ + ½χ²₁ gives 2.5%' : 'rejection rate (power)');
        label(s3, R`Wald 95% coverage of $\mu_0$`); s3.set((100 * cov / reps).toFixed(1) + '%', 'interval max(X̄,0) ± 1.96/√n');
      }
      function draw() { const k = sc.get(); if (k === 'ns') drawNS(); else if (k === 'mix') drawMix(); else drawBd(); }
      layoutCtl();
      draw();
    },
  });

  /* ================================================================
   * fisher-information
   * ================================================================ */
  const FMODELS = {
    bern: { label: 'Bernoulli, p = 0.3', th: 0.3, lim: [0.005, 0.995], xl: 'p', sample: r => (r.u() < 0.3 ? 1 : 0), s: (x, t) => x / t - (1 - x) / (1 - t), h: (x, t) => -(x / (t * t) + (1 - x) / ((1 - t) * (1 - t))), l: (x, t) => (x ? Math.log(t) : Math.log(1 - t)), I: t => 1 / (t * (1 - t)) },
    pois: { label: 'Poisson, λ = 3', th: 3, lim: [0.05, 50], xl: 'λ', sample: r => r.poisson(3), s: (x, t) => x / t - 1, h: (x, t) => -x / (t * t), l: (x, t) => x * Math.log(t) - t, I: t => 1 / t },
    norm: { label: 'Normal mean (σ = 1)', th: 0, lim: [-50, 50], xl: 'μ', sample: r => r.normal(), s: (x, t) => x - t, h: () => -1, l: (x, t) => -0.5 * (x - t) ** 2, I: () => 1 },
    expo: { label: 'Exponential rate, λ = 1', th: 1, lim: [0.02, 50], xl: 'λ', sample: r => r.exp(1), s: (x, t) => 1 / t - x, h: (x, t) => -1 / (t * t), l: (x, t) => Math.log(t) - t * x, I: t => 1 / (t * t) },
    cauchy: { label: 'Cauchy location', th: 0, lim: [-50, 50], xl: 'θ', sample: r => r.cauchy(), s: (x, t) => { const u = x - t; return 2 * u / (1 + u * u); }, h: (x, t) => { const u = x - t, q = 1 + u * u; return 2 * (u * u - 1) / (q * q); }, l: (x, t) => -Math.log(1 + (x - t) ** 2), I: () => 0.5 },
  };

  Atlas.lab('fisher-information-score', {
    title: 'Information is the variance of the score and the expected curvature',
    blurb: R`Each thin curve is one data set's log-likelihood, shifted to equal 0 at the true $\theta_0$; the thick curve is the expected log-likelihood $-\tfrac12 nI(\theta_0)(\theta-\theta_0)^2$. Right: the scaled score $\ell_n'(\theta_0)/\sqrt n$ across many data sets, against $\N(0, I(\theta_0))$. The tiles check both identities: $\Var(\text{score})/n$ and $\E[J_n]/n$ both estimate $I(\theta_0)$. For the Cauchy, the observed information varies wildly from sample to sample.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mod = UI.select(controls, { label: 'Model', options: Object.keys(FMODELS).map(k => ({ value: k, label: FMODELS[k].label })), value: 'pois', onChange: draw });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 5, max: 200, step: 5, value: 20, onChange: draw });
      UI.button(controls, 'Resimulate', draw, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, xlabel: 'θ', ylabel: 'ℓ(θ) − ℓ(θ₀)', title: 'Log-likelihoods of 25 data sets' });
      const pR = new Plot(stage, { width: 420, height: 280, xlabel: 'score / √n', ylabel: 'density', title: 'Distribution of the score at θ₀' });
      const sI = UI.stat(stats, R`True $I(\theta_0)$`, { accent: 'accent' });
      const sV = UI.stat(stats, R`$\Var(\ell_n'(\theta_0))/n$`);
      const sJ = UI.stat(stats, R`$\E[J_n(\theta_0)]/n$`);
      const sJsd = UI.stat(stats, R`sd of $J_n(\theta_0)/n$`);
      const sM = UI.stat(stats, R`Mean score / $\sqrt n$`);
      function draw() {
        const M = FMODELS[mod.get()], n = Math.round(nS.get()), r = ctx.rng, th = M.th, I = M.I(th), reps = 1500;
        const sc = new Float64Array(reps), J = new Float64Array(reps), keep = [];
        const x = new Float64Array(n);
        for (let k = 0; k < reps; k++) {
          let s = 0, h = 0;
          for (let i = 0; i < n; i++) { x[i] = M.sample(r); s += M.s(x[i], th); h -= M.h(x[i], th); }
          sc[k] = s; J[k] = h;
          if (k < 25) keep.push(Float64Array.from(x));
        }
        const w = 3.2 / Math.sqrt(n * I), lo = Math.max(M.lim[0], th - w), hi = Math.min(M.lim[1], th + w);
        const g = S.linspace(lo, hi, 90);
        const curves = keep.map(xx => { let l0 = 0; for (const v of xx) l0 += M.l(v, th); return g.map(t => { let l = 0; for (const v of xx) l += M.l(v, t); return [t, l - l0]; }); });
        const ymin = -0.5 * n * I * w * w * 1.6, ymax = 0.5 * n * I * w * w * 0.9;
        pL.o.xlabel = M.xl;
        pL.setDomain([lo, hi], [ymin, ymax]).clear();
        for (const c of curves) pL.line(c.map(q => [q[0], Math.max(q[1], ymin * 1.5)]), { color: 's1', width: 1, opacity: 0.45 });
        pL.line(g.map(t => [t, -0.5 * n * I * (t - th) ** 2]), { color: 'ink', width: 3 });
        pL.vline(th, { color: 'muted', width: 1 });
        const z = sc.map(v => v / Math.sqrt(n)), sdI = Math.sqrt(I);
        const h = S.histogram(z, { bins: 40, lo: -4.5 * sdI, hi: 4.5 * sdI });
        const yR = Math.max(...h.map(b => b.density), S.pdf.normal(0, 0, sdI)) * 1.1;
        pR.setDomain([-4.5 * sdI, 4.5 * sdI], [0, yR]).clear();
        pR.bars(h, { color: 's2', opacity: 0.5 });
        pR.line(curve(-4.5 * sdI, 4.5 * sdI, v => S.pdf.normal(v, 0, sdI)), { color: 'ink', width: 2 });
        const Jn = J.map(v => v / n);
        sI.set(fmt(I, 4));
        sV.set(fmt(S.variance(Array.from(sc)) / n, 4), 'variance of the score');
        sJ.set(fmt(S.mean(Array.from(Jn)), 4), 'expected curvature');
        sJsd.set(fmt(S.sd(Array.from(Jn)), 4), M === FMODELS.norm ? 'constant: no randomness' : 'observed information is random');
        sM.set(fmt(S.mean(Array.from(z)), 4), 'ideal 0');
      }
      draw();
    },
  });

  // f(0) for the t3 density, for the median's asymptotic variance 1/(4 f(0)^2)
  const T3F0 = Math.exp(S.lgamma(2) - S.lgamma(1.5) - 0.5 * Math.log(3 * Math.PI));
  const LOCFAM = {
    normal: { label: 'Normal', sample: r => r.normal(), I: 1, th: { mean: 1, median: Math.PI / 2 }, mle: 'mean' },
    laplace: { label: 'Laplace', sample: r => r.laplace(0, 1), I: 1, th: { mean: 2, median: 1 }, mle: 'median' },
    logistic: { label: 'Logistic', sample: r => r.logistic(0, 1), I: 1 / 3, th: { mean: Math.PI * Math.PI / 3, median: 4 }, mle: 'logistic' },
    t3: { label: 't, 3 df', sample: r => r.t(3), I: 2 / 3, th: { mean: 3, median: 1 / (4 * T3F0 * T3F0) }, mle: 't3' },
    cauchy: { label: 'Cauchy', sample: r => r.cauchy(), I: 0.5, th: { mean: Infinity, median: Math.PI * Math.PI / 4 }, mle: 'cauchy' },
  };

  Atlas.lab('fisher-information-crlb', {
    title: 'Location estimators against the Cramér–Rao bound',
    blurb: R`For a location family with known shape, every unbiased estimator of the centre has $n\Var \ge 1/I$ (dashed line); the estimators here are unbiased by symmetry (except the Cauchy sample mean, which has no mean at all). Bars are simulated $n\Var$; black ticks mark asymptotic values. Efficiency depends on the model: the mean is optimal for the normal and useless for the Cauchy, the median is asymptotically optimal for the Laplace, and the MLE approaches the bound in every family as $n$ grows — slowly for the Laplace, whose MLE is the median.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fam = UI.segmented(controls, { label: 'Error distribution', options: Object.keys(LOCFAM).map(k => ({ value: k, label: LOCFAM[k].label })), value: 'logistic', onChange: draw });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 10, max: 200, step: 10, value: 40, onChange: draw });
      UI.button(controls, 'Resimulate', draw, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 4], xticks: 0, ylabel: 'n · Var(estimator)' });
      const tiles = ['Mean', 'Median', '10% trimmed mean', 'MLE'].map(t => UI.stat(stats, `Efficiency: ${t}`));
      const COLS = ['s1', 's2', 's3', 's5'];
      function draw() {
        const F = LOCFAM[fam.get()], n = Math.round(nS.get()), r = ctx.rng, reps = 1000;
        const est = [new Float64Array(reps), new Float64Array(reps), new Float64Array(reps), new Float64Array(reps)];
        const x = new Float64Array(n);
        for (let k = 0; k < reps; k++) {
          for (let i = 0; i < n; i++) x[i] = F.sample(r);
          const s = Float64Array.from(x).sort(), med = medianSorted(s);
          let m = 0; for (let i = 0; i < n; i++) m += x[i]; m /= n;
          est[0][k] = m; est[1][k] = med; est[2][k] = trimmedSorted(s, 0.1);
          est[3][k] = F.mle === 'mean' ? m : F.mle === 'median' ? med : locMLE(x, F.mle, med, 25);
        }
        const nv = est.map(e => { let s = 0; for (const v of e) s += v * v; return n * s / reps; }); // true centre is 0
        const bound = 1 / F.I, cap = 3.2 * bound;
        p.setDomain([0, 4], [0, cap * 1.12]).clear();
        p.hline(bound, { color: 'ink', width: 1.6 });
        p.text(3.98, bound, 'Cramér–Rao bound 1/I', { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        const theory = [F.th.mean, F.th.median, NaN, bound];
        ['Mean', 'Median', '10% trimmed', 'MLE'].forEach((name, k) => {
          const v = nv[k], shown = Math.min(v, cap);
          p.bars([{ x0: k + 0.18, x1: k + 0.82, y: shown }], { color: COLS[k], opacity: 0.7 });
          p.text(k + 0.5, shown, v > cap ? `${fmt(v, 1)} (off scale)` : fmt(v, 2), { anchor: 'middle', dy: -6, size: 12, weight: 600 });
          if (Number.isFinite(theory[k]) && theory[k] <= cap) p.segment(k + 0.12, theory[k], k + 0.88, theory[k], { color: 'ink', width: 2.5 });
          p.text(k + 0.5, 0, name, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
          tiles[k].set(fmt(Math.min(1, bound / v) * 100, 1) + '%', `asymptotic: ${Number.isFinite(theory[k]) ? (Math.min(1, bound / theory[k]) * 100).toFixed(1) + '%' : (k === 0 ? '0%' : '—')}`);
        });
      }
      draw();
    },
  });

  Atlas.lab('fisher-information-hodges', {
    title: "Hodges' superefficient estimator: beating the bound at a price",
    blurb: R`For $X_i\iid\N(\theta,1)$, Hodges' estimator reports $\bar X$ unless $|\bar X| \lt n^{-1/4}$, in which case it reports 0. Left: exact scaled risk $n\cdot\MSE_\theta$ for $\bar X$ (flat at the Cramér–Rao value 1) and for Hodges. At $\theta=0$ it beats the bound; just beyond $n^{-1/4}$ its risk explodes. Right: the worst-case scaled risk grows like $\sqrt n$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 10, max: 1e6, value: 1000, log: true, fmt: v => Math.round(v).toLocaleString('en-US'), onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, xlabel: 'θ', ylabel: 'n · MSE', title: 'Scaled risk at this n' });
      const pR = new Plot(stage, { width: 420, height: 280, logx: true, logy: true, x: [10, 1e6], y: [0.5, 1000], xlabel: 'n', ylabel: 'max over θ of n · MSE', title: 'Worst-case scaled risk' });
      const s0 = UI.stat(stats, R`$n\MSE$ at $\theta=0$`, { accent: 'good' });
      const sMax = UI.stat(stats, R`$\max_\theta n\MSE$`, { accent: 'bad' });
      const sAt = UI.stat(stats, 'Attained at θ ≈');
      const sC = UI.stat(stats, R`Threshold $n^{-1/4}$`);
      const risk = (t, n) => n * threshMoments(t, n, n ** -0.25).mse;
      const supRisk = n => { const c = n ** -0.25; let m = 0, at = 0; for (const t of S.linspace(0, 3 * c, 400)) { const v = risk(t, n); if (v > m) { m = v; at = t; } } return [m, at]; };
      const sups = S.linspace(1, 6, 26).map(e => [10 ** e, supRisk(10 ** e)[0]]);
      function draw() {
        const n = nS.get(), c = n ** -0.25, g = S.linspace(-3 * c, 3 * c, 500);
        const rv = g.map(t => [t, risk(t, n)]);
        const [m, at] = supRisk(n);
        pL.setDomain([-3 * c, 3 * c], [0, Math.max(1.5, m * 1.1)]).clear();
        pL.band(-c, c, { color: 'faint', opacity: 0.12 });
        pL.hline(1, { color: 's1', dash: false, width: 2.2 });
        pL.line(rv, { color: 's2', width: 2.4 });
        pL.legend([{ label: 'X̄ (= Cramér–Rao value)', color: 's1' }, { label: 'Hodges', color: 's2' }]);
        pR.clear();
        pR.line(sups, { color: 's2', width: 2.2 });
        pR.line([[10, 1], [1e6, 1]], { color: 's1', width: 2 });
        pR.line([[10, sups[0][1]], [1e6, sups[0][1] * Math.sqrt(1e5)]], { color: 'muted', width: 1, dash: true });
        pR.text(1e6, sups[0][1] * Math.sqrt(1e5), 'slope ½', { anchor: 'end', dy: 14, size: 11, color: 'muted' });
        pR.dot(n, m, { color: 'accent', r: 5 });
        s0.set(fmt(risk(0, n), 4), 'below the bound 1');
        sMax.set(fmt(m, 2));
        sAt.set(fmt(at, 4), `= ${fmt(at / c, 2)} × n^(−1/4)`);
        sC.set(fmt(c, 4));
      }
      draw();
    },
  });

  /* ================================================================
   * rao-blackwell
   * ================================================================ */
  const RBS = {
    pois: {
      label: 'Poisson: estimate e^(−λ)', par: u => 0.1 + 3.9 * u, parName: 'λ', target: l => Math.exp(-l),
      names: ['crude 1{X₁ = 0}', 'Rao–Blackwell (1−1/n)^T', 'MLE e^(−X̄)'],
      sim(r, n, l) { const x1 = r.poisson(l), T = x1 + (n > 1 ? r.poisson((n - 1) * l) : 0); return [x1 === 0 ? 1 : 0, (1 - 1 / n) ** T, Math.exp(-T / n)]; },
      theory: (n, l) => ({ crude: Math.exp(-l) * (1 - Math.exp(-l)), rb: Math.exp(-2 * l) * (Math.exp(l / n) - 1), crlb: l * Math.exp(-2 * l) / n }),
    },
    unif: {
      label: 'Uniform(0, θ): estimate θ', par: u => 0.5 + 2.5 * u, parName: 'θ', target: t => t,
      names: ['crude 2X₁', 'Rao–Blackwell (n+1)/n·X₍ₙ₎', 'moment 2X̄'],
      sim(r, n, t) { let s = 0, m = 0, x1 = 0; for (let i = 0; i < n; i++) { const v = t * r.u(); if (i === 0) x1 = v; s += v; if (v > m) m = v; } return [2 * x1, (n + 1) / n * m, 2 * s / n]; },
      theory: (n, t) => ({ crude: t * t / 3, rb: t * t / (n * (n + 2)), crlb: NaN }),
    },
    bern: {
      label: 'Bernoulli: estimate p²', par: u => 0.05 + 0.9 * u, parName: 'p', target: p => p * p,
      names: ['crude X₁X₂', 'Rao–Blackwell T(T−1)/(n(n−1))', 'plug-in (T/n)²'],
      sim(r, n, p) { const a = r.u() < p ? 1 : 0, b = r.u() < p ? 1 : 0, T = a + b + (n > 2 ? r.binomial(n - 2, p) : 0); return [a * b, T * (T - 1) / (n * (n - 1)), (T / n) ** 2]; },
      theory(n, p) { let e2 = 0; for (let t = 0; t <= n; t++) e2 += S.pmf.binom(t, n, p) * (t * (t - 1) / (n * (n - 1))) ** 2; return { crude: p * p * (1 - p * p), rb: e2 - p ** 4, crlb: 4 * p ** 3 * (1 - p) / n }; },
    },
  };

  Atlas.lab('rao-blackwell-variance', {
    title: 'Rao–Blackwellization shrinks the sampling distribution',
    blurb: R`A crude unbiased estimator that uses one or two observations (red), its Rao–Blackwellization $\E[W\mid T]$ given the sufficient statistic (blue), and a third common estimator (green), over thousands of simulated data sets. All three centre near the target (black line); conditioning collapses the spread. Tiles compare simulated and exact variances with the Cramér–Rao bound, which the Rao–Blackwellized estimator (here the UMVUE) need not attain.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sc = UI.segmented(controls, { label: 'Problem', options: Object.keys(RBS).map(k => ({ value: k, label: RBS[k].label })), value: 'pois', onChange: draw });
      const uS = UI.slider(controls, { label: 'Parameter', min: 0, max: 1, step: 0.01, value: 0.25, fmt: u => `${RBS[sc.get()].parName} = ${RBS[sc.get()].par(u).toFixed(2)}`, onChange: draw });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 2, max: 50, step: 1, value: 10, onChange: draw });
      UI.button(controls, 'Resimulate', draw, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, xlabel: 'estimate', ylabel: 'density' });
      const sC = UI.stat(stats, R`Var crude $W$`, { accent: 'bad' });
      const sR = UI.stat(stats, R`Var $\E[W\mid T]$`, { accent: 'accent' });
      const sO = UI.stat(stats, 'MSE of third estimator');
      const sB = UI.stat(stats, 'Cramér–Rao bound');
      const COLS = ['bad', 's1', 's3'];
      function draw() {
        const P = RBS[sc.get()], n = Math.round(nS.get()), th = P.par(uS.get()), reps = 5000, r = ctx.rng;
        const est = [new Float64Array(reps), new Float64Array(reps), new Float64Array(reps)];
        for (let k = 0; k < reps; k++) { const v = P.sim(r, n, th); est[0][k] = v[0]; est[1][k] = v[1]; est[2][k] = v[2]; }
        const tgt = P.target(th);
        const all = sortNum([...est[0], ...est[1], ...est[2]]);
        let lo = S.quantile(all, 0.002, true), hi = S.quantile(all, 0.998, true);
        const pad = (hi - lo) * 0.06 || 0.05; lo -= pad; hi += pad;
        const hs = est.map(e => S.histogram(e, { bins: 70, lo, hi }));
        const ymax = Math.max(...hs[1].map(b => b.density), ...hs[2].map(b => b.density)) * 1.15;
        p.setDomain([lo, hi], [0, ymax]).clear();
        hs.forEach((h, i) => p.bars(h, { color: COLS[i], opacity: i === 0 ? 0.35 : 0.5 }));
        p.vline(tgt, { color: 'ink', dash: false, width: 1.8 });
        p.legend(P.names.map((nm, i) => ({ label: nm, color: COLS[i], fill: true })));
        const T = P.theory(n, th), v = e => S.variance(Array.from(e), 0);
        let mse = 0; for (const x of est[2]) mse += (x - tgt) ** 2; mse /= reps;
        sC.set(fmt(v(est[0]), 5), `exact ${fmt(T.crude, 5)}`);
        sR.set(fmt(v(est[1]), 5), `exact ${fmt(T.rb, 5)}`);
        sO.set(fmt(mse, 5), P.names[2]);
        sB.set(Number.isFinite(T.crlb) ? fmt(T.crlb, 5) : 'n/a', Number.isFinite(T.crlb) ? `RB / bound = ${fmt(T.rb / T.crlb, 3)}` : 'support depends on θ');
        if (sc.get() === 'pois' && v(est[0]) > 0) p.text(hi, ymax * 0.93, 'crude estimator is 0/1: tall spikes off-scale', { anchor: 'end', dx: -8, size: 11, color: 'muted' });
      }
      draw();
    },
  });

  const CONDS = {
    unif: { label: 'Uniform(0,1): W = 2X₁, T = X₍ₙ₎', W: 'W = 2X₁', T: 'T = max', sim(r, n) { let m = 0, x1 = 0; for (let i = 0; i < n; i++) { const v = r.u(); if (!i) x1 = v; if (v > m) m = v; } return [m, 2 * x1]; }, phi: (t, n) => (n + 1) / n * t, xr: () => [0, 1.02], yr: [-0.05, 2.05], jit: [0, 0] },
    pois: { label: 'Poisson(1): W = 1{X₁ = 0}, T = ΣXᵢ', W: 'W = 1{X₁=0}', T: 'T = ΣXᵢ', sim(r, n) { const x1 = r.poisson(1); return [x1 + r.poisson(n - 1), x1 === 0 ? 1 : 0]; }, phi: (t, n) => (1 - 1 / n) ** t, xr: n => [-0.8, n + 4 * Math.sqrt(n) + 1], yr: [-0.15, 1.15], jit: [0.25, 0.05] },
    norm: { label: 'Normal(0,1): W = X₁, T = X̄', W: 'W = X₁', T: 'T = X̄', sim(r, n) { const x1 = r.normal(); let s = x1; for (let i = 1; i < n; i++) s += r.normal(); return [s / n, x1]; }, phi: t => t, xr: n => [-3.5 / Math.sqrt(n), 3.5 / Math.sqrt(n)], yr: [-3.2, 3.2], jit: [0, 0] },
  };

  Atlas.lab('rao-blackwell-conditioning', {
    title: 'Conditioning as projection onto the sufficient statistic',
    blurb: R`Each dot is one simulated data set, placed at (sufficient statistic $T$, crude unbiased estimator $W$). The curve is $\E[W\mid T]$. Press "Project": every dot slides vertically onto the curve. The vertical scatter that disappears is $\E[\Var(W\mid T)]$, exactly the variance that Rao–Blackwellization removes — while the mean, and hence unbiasedness, is unchanged.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sc = UI.segmented(controls, { label: 'Example', options: Object.keys(CONDS).map(k => ({ value: k, label: k === 'unif' ? 'Uniform' : k === 'pois' ? 'Poisson' : 'Normal' })), value: 'unif', onChange: resample });
      const nS = UI.slider(controls, { label: R`Sample size $n$`, min: 2, max: 20, step: 1, value: 5, onChange: resample });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(stepAnim);
      ctx.playButton(row, loop, ['▶ Project', '❚❚ Pause']);
      UI.button(row, 'Resample', resample);
      const p = new Plot(stage, { width: 640, height: 320, xlabel: 'T', ylabel: 'W' });
      const sW = UI.stat(stats, R`$\Var(W)$`, { accent: 'bad' });
      const sP = UI.stat(stats, R`$\Var(\E[W\mid T])$`, { accent: 'accent' });
      const sE = UI.stat(stats, R`$\E[\Var(W\mid T)]$`);
      const sM = UI.stat(stats, 'Means: W vs E[W | T]');
      let pts = [], jit = [], tAnim = 0;
      function resample() {
        loop.stop(); tAnim = 0;
        const C = CONDS[sc.get()], n = Math.round(nS.get()), r = ctx.rng;
        pts = r.array(500, () => C.sim(r, n));
        jit = pts.map(() => [r.uniform(-1, 1) * C.jit[0], r.uniform(-1, 1) * C.jit[1]]);
        draw();
      }
      function stepAnim(dt) { tAnim = Math.min(1, tAnim + dt / 1.4); draw(); if (tAnim >= 1) { tAnim = 0; return false; } }
      function draw() {
        const C = CONDS[sc.get()], n = Math.round(nS.get());
        const [x0, x1] = C.xr(n);
        p.o.xlabel = C.T; p.o.ylabel = C.W;
        p.setDomain([x0, x1], C.yr).clear();
        const s = tAnim > 0 ? tAnim * tAnim * (3 - 2 * tAnim) : 0;
        const show = pts.map(([t, w], i) => { const target = C.phi(t, n); const wy = w + jit[i][1]; return [t + jit[i][0], wy + s * (target - wy)]; });
        p.points(show, { color: 's1', r: 2.6, opacity: 0.55 });
        p.line(curve(Math.max(x0, sc.get() === 'pois' ? 0 : x0), x1, t => C.phi(sc.get() === 'pois' ? Math.max(0, t) : t, n), 200), { color: 'ink', width: 2.4 });
        const W = pts.map(q => q[1]), Ph = pts.map(q => C.phi(q[0], n));
        const vW = S.variance(W, 0), vP = S.variance(Ph, 0);
        let ev = 0; for (let i = 0; i < pts.length; i++) ev += (W[i] - Ph[i]) ** 2; ev /= pts.length;
        sW.set(fmt(vW, 4));
        sP.set(fmt(vP, 4), `${fmt(100 * (1 - vP / vW), 1)}% of the variance removed`);
        sE.set(fmt(ev, 4), `sum with previous: ${fmt(vP + ev, 4)}`);
        sM.set(`${fmt(S.mean(W), 3)} vs ${fmt(S.mean(Ph), 3)}`, 'both unbiased');
      }
      resample();
    },
  });
})();
