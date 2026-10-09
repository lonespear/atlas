/* Labs: Nonparametric Estimation. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  /* Shared helpers                                                       */
  /* ------------------------------------------------------------------ */
  const SQ2PI = Math.sqrt(2 * Math.PI);
  const phi = x => Math.exp(-0.5 * x * x) / SQ2PI;
  const Phi = x => S.cdf.normal(x);
  const sortA = a => a.slice().sort((x, y) => x - y);
  const logspace = (a, b, k) => S.linspace(Math.log10(a), Math.log10(b), k).map(v => 10 ** v);
  function trimmedMean(a, p) { const s = sortA(a), k = Math.floor(p * s.length); return S.mean(s.slice(k, s.length - k)); }
  function mad(a) { const m = S.median(a); return S.median(a.map(v => Math.abs(v - m))); }
  // Location M-estimator by IRLS from the median with MAD/0.6745 scale.
  function mEstimate(a, kind, c) {
    let th = S.median(a);
    const s = mad(a) / 0.6745;
    if (!(s > 0)) return th;
    for (let it = 0; it < 100; it++) {
      let sw = 0, swx = 0;
      for (const x of a) {
        const u = (x - th) / s, au = Math.abs(u);
        let w;
        if (kind === 'huber') w = au <= c ? 1 : c / au;
        else w = au < c ? (1 - (u / c) ** 2) ** 2 : 0;
        sw += w; swx += w * x;
      }
      if (!(sw > 0)) return th;
      const nt = swx / sw;
      if (Math.abs(nt - th) < 1e-10 * s) { th = nt; break; }
      th = nt;
    }
    return th;
  }
  // One-sample Hodges–Lehmann: median of Walsh averages.
  function hodgesLehmann(a) {
    const w = [];
    for (let i = 0; i < a.length; i++) for (let j = i; j < a.length; j++) w.push((a[i] + a[j]) / 2);
    return S.median(w);
  }
  // Trapezoid integral of samples ys on an equally spaced grid with step dx.
  function trapz(ys, dx) { let s = 0; for (let i = 0; i < ys.length; i++) s += (i === 0 || i === ys.length - 1 ? 0.5 : 1) * ys[i]; return s * dx; }
  // Normal mixture helper.
  function mixture(comps) {
    return {
      pdf: x => comps.reduce((s, [w, m, sd]) => s + w * phi((x - m) / sd) / sd, 0),
      cdf: x => comps.reduce((s, [w, m, sd]) => s + w * Phi((x - m) / sd), 0),
      sample: r => { let u = r.u(); for (const [w, m, sd] of comps) { if (u < w) return r.normal(m, sd); u -= w; } const c = comps[comps.length - 1]; return r.normal(c[1], c[2]); },
    };
  }
  const BIMODAL = mixture([[0.5, -1.4, 0.6], [0.5, 1.4, 0.6]]);
  const CLAW = mixture([[0.5, 0, 1], [0.1, -1, 0.1], [0.1, -0.5, 0.1], [0.1, 0, 0.1], [0.1, 0.5, 0.1], [0.1, 1, 0.1]]);

  // Densities used by the density-estimation labs.
  const DENS = {
    normal: { label: 'Normal', pdf: x => phi(x), cdf: Phi, sample: r => r.normal(), range: [-4, 4], wide: [-7, 7] },
    bimodal: { label: 'Bimodal', pdf: BIMODAL.pdf, cdf: BIMODAL.cdf, sample: BIMODAL.sample, range: [-4, 4], wide: [-7, 7] },
    claw: { label: 'Claw', pdf: CLAW.pdf, cdf: CLAW.cdf, sample: CLAW.sample, range: [-3, 3], wide: [-7, 7] },
    exp: { label: 'Exponential (boundary)', pdf: x => S.pdf.exp(x), cdf: x => S.cdf.exp(x), sample: r => r.exp(1), range: [-0.5, 5], wide: [0, 14] },
  };

  // Populations for the ECDF labs (cdfL = left limit; discrete ones list support).
  const POPS = {
    normal: { label: 'Normal(0,1)', sample: r => r.normal(), cdf: Phi, range: [-3.5, 3.5] },
    exp: { label: 'Exponential(1)', sample: r => r.exp(1), cdf: x => S.cdf.exp(x), range: [-0.3, 5] },
    uniform: { label: 'Uniform(0,1)', sample: r => r.u(), cdf: x => S.cdf.uniform(x), range: [-0.1, 1.1] },
    cauchy: { label: 'Cauchy(0,1)', sample: r => r.cauchy(), cdf: x => S.cdf.cauchy(x), range: [-10, 10] },
    bimodal: { label: 'Bimodal mixture', sample: BIMODAL.sample, cdf: BIMODAL.cdf, range: [-3.5, 3.5] },
    poisson: { label: 'Poisson(3) (discrete)', sample: r => r.poisson(3), cdf: x => (x < 0 ? 0 : S.cdf.poisson(Math.floor(x + 1e-9), 3)), range: [-0.5, 10], discrete: true },
  };
  // sup_x |Fn - F| and the location where it is attained.
  function supDist(xs, F, discrete) {
    const n = xs.length;
    let D = 0, at = xs[0], fhat = 0, ftrue = 0;
    if (discrete) {
      const Fn = S.ecdf(xs), top = Math.max(xs[n - 1], 20);
      for (let k = 0; k <= top; k++) { const d = Math.abs(Fn(k) - F(k)); if (d > D) { D = d; at = k; fhat = Fn(k); ftrue = F(k); } }
      return { D, at, fhat, ftrue };
    }
    for (let i = 0; i < n; i++) {
      const f = F(xs[i]);
      const d1 = (i + 1) / n - f, d2 = f - i / n;
      if (d1 > D) { D = d1; at = xs[i]; fhat = (i + 1) / n; ftrue = f; }
      if (d2 > D) { D = d2; at = xs[i]; fhat = i / n; ftrue = f; }
    }
    return { D, at, fhat, ftrue };
  }
  // Kolmogorov distribution of sup|Brownian bridge|.
  function kolmCdf(t) { if (t <= 0.18) return 0; let s = 0; for (let k = 1; k <= 200; k++) { const term = Math.exp(-2 * k * k * t * t); s += k % 2 ? term : -term; if (term < 1e-15) break; } return 1 - 2 * s; }
  function kolmPdf(t) { if (t <= 0.18) return 0; let s = 0; for (let k = 1; k <= 200; k++) { const term = k * k * Math.exp(-2 * k * k * t * t); s += k % 2 ? term : -term; if (term < 1e-15) break; } return 8 * t * s; }
  // Expand (x_i, y_i) into a right-continuous step polyline from lo to hi starting at y0.
  function stepLine(xs, ys, lo, hi, y0) {
    const out = [[lo, y0]];
    let prev = y0;
    for (let i = 0; i < xs.length; i++) { out.push([xs[i], prev]); out.push([xs[i], ys[i]]); prev = ys[i]; }
    out.push([hi, prev]);
    return out;
  }

  /* ------------------------------------------------------------------ */
  /* empirical-cdf                                                        */
  /* ------------------------------------------------------------------ */
  Atlas.lab('empirical-cdf-dkw', {
    title: 'The ECDF, its DKW band, and the sup-distance',
    blurb: R`One sample's $\hat F_n$ (step), the true $F$ (curve) and the DKW band $\hat F_n\pm\epsilon_n$. The red segment marks where $|\hat F_n - F|$ is largest. Grow $n$ and the band shrinks like $1/\sqrt n$; press Repeat to check that the band contains all of $F$ at least $1-\alpha$ of the time, for every population, including the Cauchy and the discrete Poisson.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population $F$', options: Object.keys(POPS).map(k => ({ value: k, label: POPS[k].label })), value: 'normal', onChange: () => { reset(); newSample(); } });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 2000, value: 50, log: true, fmt: v => String(Math.round(v)), onChange: () => { reset(); newSample(); } });
      const al = UI.segmented(controls, { label: 'Band level $1-\\alpha$', options: [{ value: '0.1', label: '90%' }, { value: '0.05', label: '95%' }, { value: '0.01', label: '99%' }], value: '0.05', onChange: () => { reset(); draw(); } });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newSample, { primary: true });
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Repeat 1000×', '❚❚ Pause']);
      const p = new Plot(stage, { width: 640, height: 320, xlabel: 'x', ylabel: 'F(x)', y: [-0.02, 1.02] });
      const sD = UI.stat(stats, '$D_n=\\sup|\\hat F_n-F|$', { accent: 'accent' });
      const sR = UI.stat(stats, '$\\sqrt n\\,D_n$');
      const sE = UI.stat(stats, 'DKW half-width $\\epsilon_n$');
      const sC = UI.stat(stats, 'This band contains $F$?');
      const sCov = UI.stat(stats, 'Coverage over repeats');
      let xs = [], reps = 0, hits = 0;
      const N = () => Math.round(n.get());
      const eps = () => Math.sqrt(Math.log(2 / +al.get()) / (2 * N()));
      function reset() { reps = 0; hits = 0; loop.stop(); }
      function newSample() { const P = POPS[pop.get()]; xs = sortA(ctx.rng.array(N(), () => P.sample(ctx.rng))); draw(); }
      function step() {
        const P = POPS[pop.get()], e = eps(), per = Math.max(1, Math.round(4000 / N()));
        for (let k = 0; k < per && reps < 1000; k++) {
          const ys = sortA(ctx.rng.array(N(), () => P.sample(ctx.rng)));
          if (supDist(ys, P.cdf, P.discrete).D <= e) hits++;
          reps++;
        }
        sCov.set(reps ? (100 * hits / reps).toFixed(1) + '%' : '—', `${reps} samples; nominal ≥ ${(100 * (1 - +al.get())).toFixed(0)}%`);
        if (reps >= 1000) return false;
      }
      function draw() {
        const P = POPS[pop.get()], [lo, hi] = P.range, e = eps(), nn = xs.length;
        p.setDomain([lo, hi], [-0.02, 1.02]).clear();
        const ys = xs.map((_, i) => (i + 1) / nn);
        const up = stepLine(xs, ys.map(v => Math.min(1, v + e)), lo, hi, Math.min(1, e));
        const dn = stepLine(xs, ys.map(v => Math.max(0, v - e)), lo, hi, 0);
        p.area(up, { base: dn, color: 's2', opacity: 0.18 });
        if (P.discrete) {
          const ks = S.range(Math.ceil(hi) + 1);
          p.line(stepLine(ks, ks.map(k => P.cdf(k)), lo, hi, 0), { color: 'ink', width: 1.6 });
        } else p.line(S.linspace(lo, hi, 400).map(x => [x, P.cdf(x)]), { color: 'ink', width: 1.6 });
        p.line(stepLine(xs, ys, lo, hi, 0), { color: 's1', width: 2 });
        const sd = supDist(xs, P.cdf, P.discrete);
        p.segment(sd.at, sd.fhat, sd.at, sd.ftrue, { color: 'bad', width: 3 });
        const off = xs.filter(x => x < lo || x > hi).length;
        if (off) p.text(hi, 0.06, `${off} point(s) off-axis`, { anchor: 'end', dx: -6, size: 11, color: 'muted' });
        p.legend([{ label: 'ECDF', color: 's1' }, { label: 'true F', color: 'ink' }, { label: 'DKW band', color: 's2', fill: true, opacity: 0.3 }, { label: 'max gap', color: 'bad' }], { left: true });
        sD.set(S.fmt(sd.D), `at x = ${S.fmt(sd.at, 2)}`);
        sR.set(S.fmt(Math.sqrt(nn) * sd.D), 'Kolmogorov 95% point: 1.358');
        sE.set(S.fmt(e), `= √(log(2/α)/(2n))`);
        sC.set(sd.D <= e ? '<span style="color:var(--good)">yes</span>' : '<span style="color:var(--bad)">no</span>');
        if (!reps) sCov.set('—', 'press Repeat');
      }
      newSample();
    },
  });

  Atlas.lab('empirical-cdf-ks', {
    title: 'The null distribution of the KS statistic',
    blurb: R`Simulate $\sqrt n D_n$ under $H_0$ and compare it with Kolmogorov's limit (curve). For continuous populations the histogram is the same whatever $F$ is — that is distribution-freeness. Tick "estimate parameters" and the statistic shrinks (the test becomes very conservative), and for the discrete Poisson the limit no longer applies.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population under $H_0$', options: ['uniform', 'normal', 'exp', 'cauchy', 'poisson'].map(k => ({ value: k, label: POPS[k].label })), value: 'normal', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 500, value: 50, log: true, fmt: v => String(Math.round(v)), onChange: reset });
      const est = UI.toggle(controls, { label: 'Estimate parameters from the data (Lilliefors)', value: false, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Simulate', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 2.6], y: [0, 2], xlabel: '√n · Dₙ', ylabel: 'density' });
      const sN = UI.stat(stats, 'Simulated statistics');
      const sRej = UI.stat(stats, 'Rejection rate at 1.358', { accent: 'accent' });
      const sM = UI.stat(stats, 'Mean of $\\sqrt n D_n$');
      const note = UI.note(stats, '');
      let vals = [];
      function fitted(P, key, xs) {
        if (!est.get()) return P.cdf;
        const m = S.mean(xs);
        if (key === 'normal') { const s = S.sd(xs); return x => Phi((x - m) / s); }
        if (key === 'exp') return x => S.cdf.exp(x, 1 / m);
        if (key === 'uniform') { const a = xs[0], b = xs[xs.length - 1]; return x => S.clamp((x - a) / (b - a), 0, 1); }
        if (key === 'cauchy') { const med = S.median(xs), sc = (S.quantile(xs, 0.75, true) - S.quantile(xs, 0.25, true)) / 2; return x => S.cdf.cauchy(x, med, sc); }
        return x => (x < 0 ? 0 : S.cdf.poisson(Math.floor(x + 1e-9), Math.max(m, 1e-3)));
      }
      function reset() { vals = []; loop.stop(); draw(); }
      function step() {
        const key = pop.get(), P = POPS[key], nn = Math.round(n.get());
        const per = Math.max(2, Math.round(6000 / nn));
        for (let k = 0; k < per && vals.length < 5000; k++) {
          const xs = sortA(ctx.rng.array(nn, () => P.sample(ctx.rng)));
          vals.push(Math.sqrt(nn) * supDist(xs, fitted(P, key, xs), P.discrete).D);
        }
        draw();
        if (vals.length >= 5000) return false;
      }
      function draw() {
        p.clear();
        const grid = S.linspace(0.01, 2.6, 300);
        if (vals.length) {
          const h = S.histogram(vals, { bins: 52, lo: 0, hi: 2.6 });
          p.bars(h, { color: 's1', opacity: 0.45 });
        }
        p.band(1.358, 2.6, { color: 'bad', opacity: 0.07 });
        p.line(grid.map(t => [t, kolmPdf(t)]), { color: 'ink', width: 2 });
        p.vline(1.358, { color: 'bad' });
        p.text(1.37, 1.85, 'reject at 5%', { color: 'bad', size: 11 });
        p.legend([{ label: 'simulated √n Dₙ', color: 's1', fill: true }, { label: 'Kolmogorov limit', color: 'ink' }]);
        const rej = vals.filter(v => v > 1.358).length;
        sN.set(String(vals.length));
        sRej.set(vals.length ? (100 * rej / vals.length).toFixed(1) + '%' : '—', 'nominal 5%');
        sM.set(vals.length ? S.fmt(S.mean(vals)) : '—', 'limit: √(π/2)·ln 2 ≈ 0.869');
        note.set(est.get() ? 'Parameters re-estimated in every sample: the fitted CDF hugs $\\hat F_n$, so $D_n$ is too small for Kolmogorov\'s table.' : (pop.get() === 'poisson' ? 'Discrete $F_0$: the statistic is stochastically smaller than the continuous-case limit, so the test is conservative.' : 'Fully specified continuous $F_0$: exact distribution-freeness.'));
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  /* statistical-functionals                                              */
  /* ------------------------------------------------------------------ */
  const qTrim = p => S.q.normal(1 - p);
  const FUNCS = {
    mean: { label: 'Mean', f: S.mean, L: x => x },
    median: { label: 'Median', f: S.median, L: x => (x === 0 ? 0 : Math.sign(x)) * SQ2PI / 2 },
    trim10: { label: '10% trimmed', f: a => trimmedMean(a, 0.1), L: x => S.clamp(x, -qTrim(0.1), qTrim(0.1)) / 0.8 },
    trim25: { label: '25% trimmed', f: a => trimmedMean(a, 0.25), L: x => S.clamp(x, -qTrim(0.25), qTrim(0.25)) / 0.5 },
    variance: { label: 'Variance', f: a => S.variance(a, 0), L: x => x * x - 1 },
    sd: { label: 'SD', f: a => S.sd(a, 0), L: x => (x * x - 1) / 2 },
  };
  Atlas.lab('statistical-functionals-influence', {
    title: 'Sensitivity curves: dragging one observation',
    blurb: R`A fixed sample of $n-1$ standard normal points (ticks) plus one extra point at $x$. The solid curve is the sensitivity curve $SC_n(x) = n\{T(\text{sample}+x) - T(\text{sample})\}$; the dashed curve is the influence function $L(x)$ at $\N(0,1)$. Drag or click on the plot to move the extra point. Bounded curves mean one observation can do only bounded damage; increase $n$ to see $SC_n\to L$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const stat = UI.segmented(controls, { label: 'Statistic $T$', options: Object.keys(FUNCS).map(k => ({ value: k, label: FUNCS[k].label })), value: 'median', onChange: draw });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 400, step: 1, value: 30, onChange: newSample });
      const x0s = UI.slider(controls, { label: 'Extra point $x$', min: -6, max: 6, step: 0.05, value: 2.5, onInput: draw });
      UI.button(controls, 'New base sample', newSample, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, x: [-6, 6], y: [-4, 4], xlabel: 'location of the extra observation x', ylabel: 'influence' });
      const sT0 = UI.stat(stats, '$T$ without $x$');
      const sT1 = UI.stat(stats, '$T$ with $x$');
      const sSC = UI.stat(stats, '$SC_n(x)$', { accent: 'accent' });
      const sL = UI.stat(stats, '$L(x)$ at $\\N(0,1)$');
      const sG = UI.stat(stats, 'Gross-error sensitivity $\\sup|L|$');
      let base = [];
      p.onPointer((x, y, e, type) => { if (type === 'down' || type === 'drag') { x0s.set(S.clamp(Math.round(x * 20) / 20, -6, 6)); draw(); } });
      function newSample() { base = ctx.rng.array(Math.round(n.get()) - 1, () => ctx.rng.normal()); draw(); }
      function draw() {
        const F = FUNCS[stat.get()], nn = base.length + 1, T0 = F.f(base);
        const grid = S.linspace(-6, 6, 241);
        const sc = grid.map(x => [x, nn * (F.f(base.concat([x])) - T0)]);
        const L = grid.map(x => [x, F.L(x)]);
        const allY = sc.concat(L).map(q => q[1]);
        let y0 = Math.max(-12, Math.min(-1.5, ...allY) * 1.1), y1 = Math.min(36, Math.max(1.5, ...allY) * 1.1);
        p.setDomain([-6, 6], [y0, y1]).clear();
        p.hline(0, { color: 'faint', dash: false, width: 1 });
        for (const b of base) if (b > -6 && b < 6) p.segment(b, y0, b, y0 + (y1 - y0) * 0.04, { color: 'muted', width: 1, opacity: 0.6 });
        p.line(L, { color: 'ink', width: 1.6, dash: true });
        p.line(sc, { color: 's1', width: 2.2 });
        const x0 = x0s.get(), T1 = F.f(base.concat([x0])), SC = nn * (T1 - T0);
        p.vline(x0, { color: 's2', width: 1 });
        p.dot(x0, SC, { color: 's2', r: 6 });
        p.legend([{ label: 'sensitivity curve SCₙ', color: 's1' }, { label: 'influence function L', color: 'ink', dash: true }], { left: true });
        const g = Math.max(...grid.map(x => Math.abs(F.L(x))));
        sT0.set(S.fmt(T0)); sT1.set(S.fmt(T1)); sSC.set(S.fmt(SC)); sL.set(S.fmt(F.L(x0)));
        sG.set(['mean', 'variance', 'sd'].includes(stat.get()) ? '∞' : S.fmt(g), ['mean', 'variance', 'sd'].includes(stat.get()) ? 'unbounded: not B-robust' : 'bounded: B-robust');
      }
      newSample();
    },
  });

  /* ------------------------------------------------------------------ */
  /* histograms                                                           */
  /* ------------------------------------------------------------------ */
  const roughCache = {};
  // ∫f² and ∫(f')² on the density's wide support (numerical, cached).
  function roughness(key) {
    if (roughCache[key]) return roughCache[key];
    const D = DENS[key], [a, b] = D.wide, m = 20000, dx = (b - a) / m;
    const xs = S.linspace(a, b, m + 1), f = xs.map(D.pdf);
    const fp = xs.map((x, i) => (i === 0 || i === m ? 0 : (f[i + 1] - f[i - 1]) / (2 * dx)));
    const fpp = xs.map((x, i) => (i === 0 || i === m ? 0 : (f[i + 1] - 2 * f[i] + f[i - 1]) / (dx * dx)));
    roughCache[key] = { f2: trapz(f.map(v => v * v), dx), fp2: trapz(fp.map(v => v * v), dx), fpp2: trapz(fpp.map(v => v * v), dx) };
    return roughCache[key];
  }
  // Exact MISE pieces for a histogram anchored at the left of the wide support.
  function histRisk(key, h, n) {
    const D = DENS[key], [a, b] = D.wide, m = Math.ceil((b - a) / h);
    let sp2 = 0, prev = D.cdf(a);
    for (let j = 1; j <= m; j++) { const c = D.cdf(a + j * h); sp2 += (c - prev) ** 2; prev = c; }
    const r = roughness(key);
    return { isb: Math.max(r.f2 - sp2 / h, 1e-12), iv: (1 - sp2) / (n * h) };
  }
  Atlas.lab('histograms-risk', {
    title: 'Histogram risk: integrated squared bias versus variance',
    blurb: R`Exact MISE of the histogram as a function of the bin width $h$ (log–log), computed from the true bin probabilities: $\int b^2 = \int f^2 - \sum p_j^2/h$ and $\int v = (1-\sum p_j^2)/(nh)$. Dashed lines are the asymptotic approximations $\frac{h^2}{12}\int f'^2$ and $1/(nh)$. Increase $n$ and watch the optimum slide left like $n^{-1/3}$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dens = UI.select(controls, { label: 'Density $f$', options: Object.keys(DENS).map(k => ({ value: k, label: DENS[k].label })), value: 'normal', onChange: draw });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 50, max: 100000, value: 1000, log: true, fmt: v => String(Math.round(v)), onInput: draw });
      const p = new Plot(stage, { width: 640, height: 320, x: [0.01, 3], y: [1e-5, 1], logx: true, logy: true, xlabel: 'bin width h (log scale)', ylabel: 'risk (log scale)' });
      const sH = UI.stat(stats, 'Exact optimal $h$', { accent: 'accent' });
      const sHa = UI.stat(stats, 'Formula $h^* = (6/(n\\int f\'^2))^{1/3}$');
      const sSc = UI.stat(stats, 'Scott rule $3.49\\,\\sigma n^{-1/3}$');
      const sR = UI.stat(stats, 'Minimal MISE');
      function draw() {
        const key = dens.get(), nn = Math.round(n.get()), r = roughness(key);
        const hs = logspace(0.01, 3, 90);
        const rows = hs.map(h => { const q = histRisk(key, h, nn); return { h, isb: q.isb, iv: q.iv, mise: q.isb + q.iv }; });
        const k = S.argmin(rows.map(q => q.mise));
        const ha = (6 / (nn * r.fp2)) ** (1 / 3);
        const sigma = key === 'normal' ? 1 : key === 'bimodal' ? Math.sqrt(1.4 * 1.4 + 0.36) : key === 'claw' ? Math.sqrt(0.5 + 0.5 * (0.01 + 0.5)) : 1;
        const lo = Math.max(1e-7, Math.min(...rows.map(q => q.mise)) / 30), hi = Math.max(...rows.map(q => q.mise)) * 3;
        p.setDomain([0.01, 3], [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi))]).clear();
        p.line(rows.map(q => [q.h, q.isb]), { color: 's1', width: 2 });
        p.line(rows.map(q => [q.h, q.iv]), { color: 's2', width: 2 });
        p.line(rows.map(q => [q.h, q.mise]), { color: 'ink', width: 2.6 });
        p.line(hs.map(h => [h, h * h / 12 * r.fp2]), { color: 's1', width: 1.2, dash: true });
        p.line(hs.map(h => [h, 1 / (nn * h)]), { color: 's2', width: 1.2, dash: true });
        p.vline(rows[k].h, { color: 'ink', width: 1.2 });
        p.vline(ha, { color: 'accent', width: 1.2 });
        p.dot(rows[k].h, rows[k].mise, { color: 'ink', r: 5 });
        p.legend([{ label: 'MISE (exact)', color: 'ink' }, { label: 'integrated sq. bias', color: 's1' }, { label: 'integrated variance', color: 's2' }, { label: 'asymptotic terms', color: 'muted', dash: true }], { left: true });
        sH.set(S.fmt(rows[k].h), key === 'exp' ? 'bins aligned with the jump at 0' : '');
        sHa.set(S.fmt(ha), `∫f'² = ${S.fmt(r.fp2)}`);
        sSc.set(S.fmt(3.49 * sigma * nn ** (-1 / 3)), key === 'normal' ? 'exact for the normal' : 'normal-reference: biased here');
        sR.set(rows[k].mise.toPrecision(3), `≈ ${((0.75) ** (2 / 3) * r.fp2 ** (1 / 3) * nn ** (-2 / 3)).toPrecision(3)} by the asymptotic formula`);
      }
      draw();
    },
  });

  Atlas.lab('histograms-binwidth', {
    title: 'Bin width, bin origin, and cross-validation',
    blurb: R`Left: a histogram of one sample with the true density. Slide $h$ to trade bias for variance, and slide the origin to see modes appear and vanish at a fixed $h$. Right: the leave-one-out CV score $\hat J(h)$ (plus $\int f^2$, so it estimates the ISE) next to the actual ISE of this sample — CV tracks the truth, noisily.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dens = UI.select(controls, { label: 'Density $f$', options: Object.keys(DENS).map(k => ({ value: k, label: DENS[k].label })), value: 'bimodal', onChange: newSample });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 30, max: 3000, value: 300, log: true, fmt: v => String(Math.round(v)), onChange: newSample });
      const hS = UI.slider(controls, { label: 'Bin width $h$', min: 0.03, max: 2, value: 0.4, log: true, fmt: v => v.toFixed(3), onInput: draw });
      const off = UI.slider(controls, { label: 'Origin shift (fraction of $h$)', min: 0, max: 0.95, step: 0.05, value: 0, onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newSample, { primary: true });
      UI.button(row, 'Use CV h', () => { if (hcv) { hS.set(hcv); draw(); } });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'density', title: 'Histogram vs truth' });
      const pR = new Plot(stage, { width: 420, height: 270, logx: true, x: [0.03, 2], xlabel: 'h (log scale)', title: 'CV estimate of ISE vs actual ISE' });
      const sB = UI.stat(stats, 'Bins in range');
      const sM = UI.stat(stats, 'Modes in histogram', { accent: 'accent' });
      const sI = UI.stat(stats, 'ISE at this $h$');
      const sCV = UI.stat(stats, 'CV choice $\\hat h$');
      const sSc = UI.stat(stats, 'Freedman–Diaconis $2\\,\\mathrm{IQR}\\,n^{-1/3}$');
      let xs = [], hcv = null;
      function newSample() { const D = DENS[dens.get()]; xs = sortA(ctx.rng.array(Math.round(n.get()), () => D.sample(ctx.rng))); draw(); }
      function counts(h, x0) { const m = new Map(); for (const x of xs) { const j = Math.floor((x - x0) / h); m.set(j, (m.get(j) || 0) + 1); } return m; }
      function ise(h, x0, D) {
        const m = counts(h, x0), nn = xs.length, [a, b] = D.wide, g = 3000, dx = (b - a) / g;
        const ys = S.linspace(a, b, g + 1).map(x => { const c = m.get(Math.floor((x - x0) / h)) || 0; return (c / (nn * h) - D.pdf(x)) ** 2; });
        return trapz(ys, dx);
      }
      function cvScore(h, x0) {
        const m = counts(h, x0), nn = xs.length;
        let sp2 = 0; for (const c of m.values()) sp2 += (c / nn) ** 2;
        return 2 / ((nn - 1) * h) - (nn + 1) / ((nn - 1) * h) * sp2;
      }
      function draw() {
        const D = DENS[dens.get()], h = hS.get(), [lo, hi] = D.range, nn = xs.length;
        const x0 = lo + off.get() * h;
        // Left panel
        const m = counts(h, x0);
        const j0 = Math.floor((lo - x0) / h), j1 = Math.floor((hi - x0) / h);
        const bins = [];
        for (let j = j0; j <= j1; j++) bins.push({ x0: x0 + j * h, x1: x0 + (j + 1) * h, y: (m.get(j) || 0) / (nn * h) });
        const grid = S.linspace(lo, hi, 300), fv = grid.map(D.pdf);
        const ymax = Math.max(...fv, ...bins.map(b => b.y)) * 1.1;
        pL.setDomain([lo, hi], [0, ymax]).clear();
        pL.bars(bins, { color: 's1', opacity: 0.45, gap: 0.5 });
        pL.line(grid.map((x, i) => [x, fv[i]]), { color: 'ink', width: 1.8 });
        // modes: strict local maxima after collapsing plateaus
        const ys = bins.map(b => b.y).filter((v, i, a) => i === 0 || v !== a[i - 1]);
        let modes = 0; for (let i = 0; i < ys.length; i++) if ((i === 0 || ys[i] > ys[i - 1]) && (i === ys.length - 1 || ys[i] > ys[i + 1]) && ys[i] > 0) modes++;
        // Right panel: CV and ISE curves
        const hs = logspace(0.03, 2, 48), f2 = roughness(dens.get()).f2;
        const cv = hs.map(hh => cvScore(hh, lo + off.get() * hh) + f2), is = hs.map(hh => ise(hh, lo + off.get() * hh, D));
        const k = S.argmin(cv); hcv = hs[k];
        const all = cv.concat(is).filter(Number.isFinite);
        const ylo = Math.min(0, ...all), yhi = Math.min(Math.max(...all), 4 * Math.max(Math.min(...is), 0.005) + Math.min(...is) * 6);
        pR.setDomain([0.03, 2], [ylo, Math.max(yhi, 0.01)]).clear();
        pR.line(hs.map((hh, i) => [hh, is[i]]), { color: 'ink', width: 2 });
        pR.line(hs.map((hh, i) => [hh, cv[i]]), { color: 's2', width: 2 });
        pR.vline(h, { color: 's1', width: 1.5, dash: false });
        pR.vline(hcv, { color: 's2', width: 1.2 });
        pR.legend([{ label: 'actual ISE', color: 'ink' }, { label: 'CV + ∫f²', color: 's2' }, { label: 'current h', color: 's1' }]);
        const iqr = S.quantile(xs, 0.75, true) - S.quantile(xs, 0.25, true);
        sB.set(String(bins.length));
        sM.set(String(modes), dens.get() === 'bimodal' ? 'truth: 2' : dens.get() === 'claw' ? 'truth: 5 claws on a hump' : 'truth: 1');
        sI.set(S.fmt(ise(h, x0, D)));
        sCV.set(S.fmt(hcv));
        sSc.set(S.fmt(2 * iqr * nn ** (-1 / 3)));
      }
      newSample();
    },
  });

  /* ------------------------------------------------------------------ */
  /* kernel-density                                                       */
  /* ------------------------------------------------------------------ */
  // Kernels rescaled to unit variance so that h is comparable across kernels.
  const KERNELS = {
    gauss: { label: 'Gaussian', K: u => phi(u), supp: 6 },
    epa: { label: 'Epanechnikov', K: u => { const c = Math.sqrt(5), v = u / c; return Math.abs(v) <= 1 ? 0.75 * (1 - v * v) / c : 0; }, supp: Math.sqrt(5) },
    tri: { label: 'Triangular', K: u => { const c = Math.sqrt(6), v = Math.abs(u / c); return v <= 1 ? (1 - v) / c : 0; }, supp: Math.sqrt(6) },
    unif: { label: 'Uniform', K: u => { const c = Math.sqrt(3); return Math.abs(u) <= c ? 0.5 / c : 0; }, supp: Math.sqrt(3) },
  };
  // K*K on a lookup table (exact for the Gaussian).
  const convCache = {};
  function convKK(key) {
    if (convCache[key]) return convCache[key];
    const k = KERNELS[key];
    if (key === 'gauss') return (convCache[key] = u => phi(u / Math.SQRT2) / Math.SQRT2);
    const L = 2 * k.supp, m = 800, us = S.linspace(-L, L, m + 1), tg = S.linspace(-k.supp, k.supp, 401), dt = tg[1] - tg[0];
    const tab = us.map(u => trapz(tg.map(t => k.K(t) * k.K(u - t)), dt));
    const f = u => { if (Math.abs(u) >= L) return 0; const pos = (u + L) / (2 * L) * m, i = Math.floor(pos), w = pos - i; return tab[i] * (1 - w) + tab[Math.min(i + 1, m)] * w; };
    return (convCache[key] = f);
  }
  function kdeAt(xs, h, K, x, supp) {
    let s = 0;
    for (const xi of xs) { const u = (x - xi) / h; if (u > -supp && u < supp) s += K(u); }
    return s / (xs.length * h);
  }
  Atlas.lab('kernel-density-bandwidth', {
    title: 'Bandwidth, kernel shape, and LSCV',
    blurb: R`Left: the KDE (a sum of $n$ bumps, shown individually for small $n$) against the true density. Right: the LSCV criterion plus $\int f^2$, an unbiased estimate of MISE, alongside this sample's actual ISE. Switch kernels at fixed $h$ (all are scaled to unit variance) and the estimate barely changes; change $h$ by a factor of 3 and it changes completely. Try the claw: Silverman's rule smooths the claws away.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dens = UI.select(controls, { label: 'Density $f$', options: Object.keys(DENS).map(k => ({ value: k, label: DENS[k].label })), value: 'bimodal', onChange: newSample });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 20, max: 400, value: 100, log: true, fmt: v => String(Math.round(v)), onChange: newSample });
      const hS = UI.slider(controls, { label: 'Bandwidth $h$', min: 0.02, max: 2, value: 0.3, log: true, fmt: v => v.toFixed(3), onInput: draw });
      const ker = UI.segmented(controls, { label: 'Kernel', options: Object.keys(KERNELS).map(k => ({ value: k, label: KERNELS[k].label })), value: 'gauss', onChange: () => { curves(); draw(); } });
      const bumps = UI.toggle(controls, { label: 'Show individual kernels', value: true, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newSample, { primary: true });
      UI.button(row, 'Use LSCV h', () => { hS.set(hcv); draw(); });
      UI.button(row, 'Use Silverman h', () => { hS.set(S.clamp(S.silverman(xs), 0.02, 2)); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'density', title: 'Kernel density estimate' });
      const pR = new Plot(stage, { width: 420, height: 270, logx: true, x: [0.02, 2], xlabel: 'h (log scale)', title: 'LSCV + ∫f² vs actual ISE' });
      const sH = UI.stat(stats, 'Current $h$');
      const sCV = UI.stat(stats, 'LSCV $\\hat h$', { accent: 'accent' });
      const sSil = UI.stat(stats, 'Silverman $h$');
      const sOr = UI.stat(stats, 'Oracle $h$ (min ISE)');
      const sI = UI.stat(stats, 'ISE at current $h$');
      let xs = [], hs = [], cv = [], is = [], hcv = 0.3;
      function newSample() { const D = DENS[dens.get()]; xs = ctx.rng.array(Math.round(n.get()), () => D.sample(ctx.rng)); curves(); draw(); }
      function iseOf(h, k, D, g) {
        const K = KERNELS[k], [a, b] = D.wide, xsG = S.linspace(a, b, g + 1), dx = (b - a) / g;
        return trapz(xsG.map(x => (kdeAt(xs, h, K.K, x, K.supp) - D.pdf(x)) ** 2), dx);
      }
      function curves() {
        const k = ker.get(), K = KERNELS[k], KK = convKK(k), nn = xs.length, D = DENS[dens.get()], f2 = roughness(dens.get()).f2;
        hs = logspace(0.02, 2, 36);
        cv = hs.map(h => {
          let a = 0, b = 0;
          for (let i = 0; i < nn; i++) for (let j = i + 1; j < nn; j++) { const u = (xs[i] - xs[j]) / h; a += 2 * KK(u); b += 2 * K.K(u); }
          a += nn * KK(0);
          return a / (nn * nn * h) - 2 * b / (nn * (nn - 1) * h) + f2;
        });
        is = hs.map(h => iseOf(h, k, D, 500));
        hcv = hs[S.argmin(cv)];
      }
      function draw() {
        const D = DENS[dens.get()], k = ker.get(), K = KERNELS[k], h = hS.get(), [lo, hi] = D.range, nn = xs.length;
        const grid = S.linspace(lo, hi, 320), fh = grid.map(x => kdeAt(xs, h, K.K, x, K.supp)), fv = grid.map(D.pdf);
        const ymax = Math.max(...fv, ...fh) * 1.12;
        pL.setDomain([lo, hi], [0, ymax]).clear();
        if (bumps.get() && nn <= 150) {
          for (const xi of xs) {
            const g2 = S.linspace(Math.max(lo, xi - K.supp * h), Math.min(hi, xi + K.supp * h), 40);
            pL.line(g2.map(x => [x, K.K((x - xi) / h) / (nn * h)]), { color: 's3', width: 1, opacity: 0.55 });
          }
        }
        pL.line(grid.map((x, i) => [x, fv[i]]), { color: 'ink', width: 1.6, dash: true });
        pL.line(grid.map((x, i) => [x, fh[i]]), { color: 's1', width: 2.4 });
        for (const xi of xs) if (xi > lo && xi < hi) pL.segment(xi, 0, xi, ymax * 0.035, { color: 'muted', width: 1, opacity: 0.7 });
        pL.legend([{ label: 'KDE', color: 's1' }, { label: 'true f', color: 'ink', dash: true }].concat(bumps.get() && nn <= 150 ? [{ label: 'K_h(x − Xᵢ)/n', color: 's3' }] : []));
        const all = cv.concat(is).filter(Number.isFinite), mIs = Math.min(...is);
        const ylo = Math.min(...all), yhi = Math.min(Math.max(...all), mIs * 8 + 0.01);
        pR.setDomain([0.02, 2], [Math.min(0, ylo), yhi]).clear();
        pR.line(hs.map((x, i) => [x, is[i]]), { color: 'ink', width: 2 });
        pR.line(hs.map((x, i) => [x, cv[i]]), { color: 's2', width: 2 });
        pR.vline(h, { color: 's1', dash: false, width: 1.5 });
        pR.vline(hcv, { color: 's2', width: 1.2 });
        pR.legend([{ label: 'actual ISE', color: 'ink' }, { label: 'LSCV + ∫f²', color: 's2' }, { label: 'current h', color: 's1' }]);
        sH.set(S.fmt(h));
        sCV.set(S.fmt(hcv));
        sSil.set(S.fmt(S.silverman(xs)));
        sOr.set(S.fmt(hs[S.argmin(is)]));
        sI.set(S.fmt(iseOf(h, k, D, 600)));
      }
      newSample();
    },
  });

  Atlas.lab('kernel-density-curse', {
    title: 'The curse of dimensionality, by simulation',
    blurb: R`Estimate a standard normal density at its mode in $d$ dimensions with a Gaussian product kernel and the normal-reference bandwidth $h = (4/(d+2))^{1/(d+4)}n^{-1/(d+4)}$. Each point is the relative root-MSE over repeated samples. Slopes flatten as $d$ grows (theory: $-2/(d+4)$), so the sample size needed for a fixed accuracy explodes.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const reps = UI.slider(controls, { label: 'Replications per point', min: 10, max: 60, step: 5, value: 30, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run simulation', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 320, logx: true, logy: true, x: [10, 30000], y: [0.01, 2], xlabel: 'sample size n (log scale)', ylabel: 'relative RMSE of f̂(0) (log)' });
      const DS = [1, 2, 3, 5, 8], NS = [10, 30, 100, 300, 1000, 3000, 10000, 30000], COLS = ['s1', 's2', 's3', 's4', 's5'];
      const tiles = DS.map((d, i) => UI.stat(stats, `$d=${d}$: $n$ for 10% rel. RMSE`));
      let res, cell;
      function reset() { res = DS.map(() => NS.map(() => null)); cell = 0; loop.stop(); draw(); }
      function relRMSE(d, n) {
        const h = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4));
        const f0 = (2 * Math.PI) ** (-d / 2), c = 1 / (n * h ** d * (2 * Math.PI) ** (d / 2));
        let se = 0;
        const R = Math.round(reps.get());
        for (let r = 0; r < R; r++) {
          let s = 0;
          for (let i = 0; i < n; i++) { let q = 0; for (let k = 0; k < d; k++) { const z = ctx.rng.normal(); q += z * z; } s += Math.exp(-q / (2 * h * h)); }
          se += (c * s / f0 - 1) ** 2;
        }
        return Math.sqrt(se / R);
      }
      function step() {
        let budget = 0;
        while (cell < DS.length * NS.length && budget < 150000) {
          const i = cell % DS.length, j = Math.floor(cell / DS.length);
          res[i][j] = relRMSE(DS[i], NS[j]);
          budget += NS[j] * DS[i] * reps.get();
          cell++;
        }
        draw();
        if (cell >= DS.length * NS.length) return false;
      }
      function draw() {
        p.clear();
        p.hline(0.1, { color: 'muted' });
        p.text(12, 0.105, '10% relative error', { size: 11, color: 'muted', dy: -4 });
        DS.forEach((d, i) => {
          const pts = NS.map((nn, j) => (res[i][j] == null ? null : [nn, res[i][j]])).filter(Boolean);
          if (pts.length) { p.line(pts, { color: COLS[i], width: 2 }); p.points(pts, { color: COLS[i], r: 3.5 }); }
          // n needed: interpolate in log-log, or extrapolate along theoretical slope
          let need = '—';
          if (pts.length >= 2) {
            const last = pts[pts.length - 1];
            const idx = pts.findIndex(q => q[1] <= 0.1);
            if (idx > 0) { const [a, b] = [pts[idx - 1], pts[idx]], t = (Math.log(0.1) - Math.log(a[1])) / (Math.log(b[1]) - Math.log(a[1])); need = '≈ ' + Math.round(Math.exp(Math.log(a[0]) + t * (Math.log(b[0]) - Math.log(a[0])))).toLocaleString(); }
            else if (idx === 0) need = '≤ 10';
            else if (cell >= DS.length * NS.length) { const sl = -2 / (d + 4); need = '≈ ' + Number((last[0] * (0.1 / last[1]) ** (1 / sl)).toPrecision(2)).toLocaleString() + ' (extrapolated)'; }
          }
          tiles[i].set(need);
        });
        p.legend(DS.map((d, i) => ({ label: `d = ${d}`, color: COLS[i] })));
      }
      reset();
    },
  });

  /* ------------------------------------------------------------------ */
  /* kernel-regression                                                    */
  /* ------------------------------------------------------------------ */
  const RFUN = {
    trend: { label: 'Trend + wiggle', f: x => 2 * x + 0.5 * Math.sin(2 * Math.PI * x) },
    line: { label: 'Straight line', f: x => 3 * x },
    expo: { label: 'Exponential', f: x => Math.exp(3 * x) / 6 },
    sine: { label: 'Sine', f: x => Math.sin(2 * Math.PI * x) },
    bump: { label: 'Wave + sharp bump', f: x => Math.sin(2 * Math.PI * x) + 1.2 * Math.exp(-(((x - 0.72) / 0.04) ** 2)) },
  };
  const DESIGNS = {
    uniform: { label: 'Uniform', draw: r => r.u(), g: () => 1 },
    increasing: { label: 'Increasing density', draw: r => -0.5 + Math.sqrt(0.25 + 2 * r.u()), g: x => 0.5 + x },
    sparse: { label: 'Gap in the middle', draw: r => (r.u() < 0.5 ? 0.4 * r.u() : 0.6 + 0.4 * r.u()), g: x => (x < 0.4 || x > 0.6 ? 1.25 : 0) },
  };
  // Gaussian-kernel weights, computed relative to the largest so that tiny h never underflows.
  function kernW(xs, x, h) {
    let m = Infinity;
    for (const xi of xs) m = Math.min(m, ((x - xi) / h) ** 2);
    return xs.map(xi => Math.exp(-0.5 * (((x - xi) / h) ** 2 - m)));
  }
  function nwW(xs, x, h) { const k = kernW(xs, x, h), s = S.sum(k); return k.map(v => v / s); }
  function llW(xs, x, h) {
    const k = kernW(xs, x, h);
    let s0 = 0, s1 = 0, s2 = 0;
    for (let i = 0; i < xs.length; i++) { const d = xs[i] - x; s0 += k[i]; s1 += k[i] * d; s2 += k[i] * d * d; }
    const den = s0 * s2 - s1 * s1;
    if (!(den > 1e-12 * s0 * s2)) return k.map(v => v / s0);
    return k.map((v, i) => v * (s2 - (xs[i] - x) * s1) / den);
  }
  const dotW = (w, y) => { let s = 0; for (let i = 0; i < w.length; i++) s += w[i] * y[i]; return s; };

  Atlas.lab('kernel-regression-boundary', {
    title: 'Nadaraya–Watson vs local linear at the edges',
    blurb: R`Left: data, the true $r$, the Nadaraya–Watson fit and the local linear fit (same Gaussian kernel and $h$). Right: the weights $\ell_i(x_0)$ each method puts on each observation at the point $x_0$ — click the left plot to move $x_0$. Near an edge NW's weights are one-sided, so it flattens the trend; local linear's weights go negative on the far side, which cancels the slope. Try the straight line with the increasing design to see design bias.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fun = UI.select(controls, { label: 'True regression $r(x)$', options: Object.keys(RFUN).map(k => ({ value: k, label: RFUN[k].label })), value: 'trend', onChange: newData });
      const des = UI.select(controls, { label: 'Design of $x_i$', options: Object.keys(DESIGNS).map(k => ({ value: k, label: DESIGNS[k].label })), value: 'uniform', onChange: newData });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 20, max: 400, value: 100, log: true, fmt: v => String(Math.round(v)), onChange: newData });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0, max: 1, step: 0.02, value: 0.3, onChange: newData });
      const hS = UI.slider(controls, { label: 'Bandwidth $h$', min: 0.01, max: 0.4, value: 0.08, log: true, fmt: v => v.toFixed(3), onInput: draw });
      const x0S = UI.slider(controls, { label: 'Inspect weights at $x_0$', min: 0, max: 1, step: 0.01, value: 0.02, onInput: draw });
      UI.button(controls, 'New data', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', ylabel: 'y', title: 'Fits (click to set x₀)' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'xᵢ', ylabel: 'weight ℓᵢ(x₀)', title: 'Weights at x₀' });
      const sB = UI.stat(stats, 'RMSE near edges (NW / LL)', { accent: 'accent' });
      const sI = UI.stat(stats, 'RMSE interior (NW / LL)');
      const sE = UI.stat(stats, 'Fit at $x_0$: truth / NW / LL');
      const sW = UI.stat(stats, 'Most negative LL weight');
      let xs = [], ys = [];
      pL.onPointer((x, y, e, type) => { if (type === 'down' || type === 'drag') { x0S.set(S.clamp(x, 0, 1)); draw(); } });
      function newData() {
        const D = DESIGNS[des.get()], f = RFUN[fun.get()].f;
        xs = sortA(ctx.rng.array(Math.round(n.get()), () => D.draw(ctx.rng)));
        ys = xs.map(x => f(x) + sig.get() * ctx.rng.normal());
        draw();
      }
      function draw() {
        const f = RFUN[fun.get()].f, h = hS.get(), x0 = x0S.get();
        const grid = S.linspace(0, 1, 201);
        const nw = grid.map(x => dotW(nwW(xs, x, h), ys)), ll = grid.map(x => dotW(llW(xs, x, h), ys)), tr = grid.map(f);
        const yAll = ys.concat(tr);
        const ylo = Math.min(...yAll), yhi = Math.max(...yAll), pad = 0.08 * (yhi - ylo || 1);
        pL.setDomain([0, 1], [ylo - pad, yhi + pad]).clear();
        pL.band(0, Math.min(h, 0.5), { color: 'warn', opacity: 0.06 });
        pL.band(Math.max(1 - h, 0.5), 1, { color: 'warn', opacity: 0.06 });
        pL.points(xs.map((x, i) => [x, ys[i]]), { color: 'muted', r: 2.2, opacity: 0.6 });
        pL.line(grid.map((x, i) => [x, tr[i]]), { color: 'ink', width: 1.6, dash: true });
        pL.line(grid.map((x, i) => [x, nw[i]]), { color: 's1', width: 2.2 });
        pL.line(grid.map((x, i) => [x, ll[i]]), { color: 's2', width: 2.2 });
        pL.vline(x0, { color: 's4', width: 1.2 });
        pL.legend([{ label: 'Nadaraya–Watson', color: 's1' }, { label: 'local linear', color: 's2' }, { label: 'true r', color: 'ink', dash: true }], { left: true });
        const wN = nwW(xs, x0, h), wL = llW(xs, x0, h);
        const wAll = wN.concat(wL), wlo = Math.min(0, ...wAll), whi = Math.max(...wAll);
        pR.setDomain([0, 1], [wlo - 0.06 * (whi - wlo), whi * 1.1]).clear();
        pR.hline(0, { color: 'faint', dash: false, width: 1 });
        pR.line(xs.map((x, i) => [x, wN[i]]), { color: 's1', width: 1.4, opacity: 0.6 });
        pR.line(xs.map((x, i) => [x, wL[i]]), { color: 's2', width: 1.4, opacity: 0.6 });
        pR.points(xs.map((x, i) => [x, wN[i]]), { color: 's1', r: 2.6 });
        pR.points(xs.map((x, i) => [x, wL[i]]), { color: 's2', r: 2.6 });
        pR.vline(x0, { color: 's4', width: 1.2 });
        pR.legend([{ label: 'NW weights', color: 's1' }, { label: 'LL weights', color: 's2' }]);
        const rm = (a, sel) => { let s = 0, c = 0; grid.forEach((x, i) => { if (sel(x)) { s += (a[i] - tr[i]) ** 2; c++; } }); return Math.sqrt(s / c); };
        const edge = x => x <= 0.08 || x >= 0.92, mid = x => x >= 0.2 && x <= 0.8;
        sB.set(`${S.fmt(rm(nw, edge))} / ${S.fmt(rm(ll, edge))}`, 'x ≤ 0.08 or x ≥ 0.92');
        sI.set(`${S.fmt(rm(nw, mid))} / ${S.fmt(rm(ll, mid))}`, '0.2 ≤ x ≤ 0.8');
        sE.set(`${S.fmt(f(x0), 2)} / ${S.fmt(dotW(wN, ys), 2)} / ${S.fmt(dotW(wL, ys), 2)}`);
        sW.set(S.fmt(Math.min(...wL)), `Σℓᵢ = ${S.fmt(S.sum(wL))}, Σℓᵢ(xᵢ−x₀) = ${S.fmt(dotW(wL, xs.map(x => x - x0)))}`);
      }
      newData();
    },
  });

  Atlas.lab('kernel-regression-cv', {
    title: 'Choosing h: leave-one-out CV, GCV and degrees of freedom',
    blurb: R`Left: the fit at the current bandwidth with a pointwise $\pm2\,\widehat{\se}$ band (a band for $\E\hat r$, not for $r$ — watch it miss the peaks). Right: leave-one-out CV (via $1/(1-L_{ii})$), GCV, and the true prediction error $\sigma^2 + \text{ASE}(h)$, which only a simulation can know. The top-axis quantity that matters is the effective degrees of freedom $\tr L$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fun = UI.select(controls, { label: 'True regression $r(x)$', options: Object.keys(RFUN).map(k => ({ value: k, label: RFUN[k].label })), value: 'bump', onChange: newData });
      const meth = UI.segmented(controls, { label: 'Smoother', options: [{ value: 'll', label: 'Local linear' }, { value: 'nw', label: 'Nadaraya–Watson' }], value: 'll', onChange: () => { curves(); draw(); } });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 30, max: 300, value: 120, log: true, fmt: v => String(Math.round(v)), onChange: newData });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.05, max: 1, step: 0.05, value: 0.3, onChange: newData });
      const hS = UI.slider(controls, { label: 'Bandwidth $h$', min: 0.005, max: 0.5, value: 0.05, log: true, fmt: v => v.toFixed(3), onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', newData, { primary: true });
      UI.button(row, 'Use CV h', () => { hS.set(best.cv); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', ylabel: 'y', title: 'Fit with ±2 se band' });
      const pR = new Plot(stage, { width: 420, height: 270, logx: true, x: [0.005, 0.5], xlabel: 'h (log scale)', title: 'Estimated vs true prediction error' });
      const sH = UI.stat(stats, 'Current $h$');
      const sDf = UI.stat(stats, 'Effective df $\\tr L$', { accent: 'accent' });
      const sS = UI.stat(stats, '$\\hat\\sigma$ (true $\\sigma$)');
      const sCV = UI.stat(stats, 'CV / GCV choice of $h$');
      const sA = UI.stat(stats, 'ASE at current $h$');
      let xs = [], ys = [], tr = [], hs = [], cvs = [], gcvs = [], ases = [], dfs = [], best = { cv: 0.05 };
      const W = (x, h) => (meth.get() === 'll' ? llW : nwW)(xs, x, h);
      function newData() {
        const f = RFUN[fun.get()].f;
        xs = sortA(ctx.rng.array(Math.round(n.get()), () => ctx.rng.u()));
        tr = xs.map(f);
        ys = tr.map(v => v + sig.get() * ctx.rng.normal());
        curves(); draw();
      }
      function fitAt(h) {
        const nn = xs.length, fit = new Array(nn), Lii = new Array(nn);
        let df = 0, df2 = 0;
        for (let i = 0; i < nn; i++) { const w = W(xs[i], h); fit[i] = dotW(w, ys); Lii[i] = w[i]; df += w[i]; for (const v of w) df2 += v * v; }
        return { fit, Lii, df, df2 };
      }
      function curves() {
        hs = logspace(0.005, 0.5, 40);
        cvs = []; gcvs = []; ases = []; dfs = [];
        const nn = xs.length;
        for (const h of hs) {
          const q = fitAt(h);
          let cv = 0, rss = 0, ase = 0;
          for (let i = 0; i < nn; i++) { const r = ys[i] - q.fit[i]; rss += r * r; cv += (r / Math.max(1e-9, 1 - q.Lii[i])) ** 2; ase += (q.fit[i] - tr[i]) ** 2; }
          cvs.push(cv / nn); gcvs.push(rss / nn / (1 - q.df / nn) ** 2); ases.push(ase / nn); dfs.push(q.df);
        }
        best = { cv: hs[S.argmin(cvs)], gcv: hs[S.argmin(gcvs)] };
      }
      function draw() {
        const h = hS.get(), f = RFUN[fun.get()].f, nn = xs.length, s2 = sig.get() ** 2;
        const q = fitAt(h);
        let rss = 0, ase = 0; for (let i = 0; i < nn; i++) { rss += (ys[i] - q.fit[i]) ** 2; ase += (q.fit[i] - tr[i]) ** 2; }
        const sh = Math.sqrt(rss / Math.max(1, nn - 2 * q.df + q.df2));
        const grid = S.linspace(0, 1, 160);
        const fw = grid.map(x => { const w = W(x, h); return [dotW(w, ys), Math.sqrt(dotW(w, w))]; });
        const yAll = ys.concat(grid.map(f));
        const ylo = Math.min(...yAll), yhi = Math.max(...yAll), pad = 0.08 * (yhi - ylo);
        pL.setDomain([0, 1], [ylo - pad, yhi + pad]).clear();
        pL.area(grid.map((x, i) => [x, fw[i][0] + 2 * sh * fw[i][1]]), { base: grid.map((x, i) => [x, fw[i][0] - 2 * sh * fw[i][1]]), color: 's1', opacity: 0.18 });
        pL.points(xs.map((x, i) => [x, ys[i]]), { color: 'muted', r: 2.2, opacity: 0.6 });
        pL.line(grid.map(x => [x, f(x)]), { color: 'ink', width: 1.6, dash: true });
        pL.line(grid.map((x, i) => [x, fw[i][0]]), { color: 's1', width: 2.2 });
        pL.legend([{ label: 'fit ± 2 se', color: 's1' }, { label: 'true r', color: 'ink', dash: true }], { left: true });
        const truthCurve = ases.map(a => a + s2);
        const all = cvs.concat(gcvs, truthCurve).filter(Number.isFinite);
        const lo = Math.min(...all), hiY = Math.min(Math.max(...all), lo + 4 * (Math.min(...truthCurve) - lo + s2));
        pR.setDomain([0.005, 0.5], [Math.max(0, lo - 0.05 * (hiY - lo)), hiY]).clear();
        pR.line(hs.map((x, i) => [x, truthCurve[i]]), { color: 'ink', width: 2 });
        pR.line(hs.map((x, i) => [x, cvs[i]]), { color: 's2', width: 2 });
        pR.line(hs.map((x, i) => [x, gcvs[i]]), { color: 's3', width: 1.6, dash: true });
        pR.vline(h, { color: 's1', dash: false, width: 1.5 });
        pR.legend([{ label: 'σ² + ASE (truth)', color: 'ink' }, { label: 'LOO-CV', color: 's2' }, { label: 'GCV', color: 's3', dash: true }]);
        sH.set(S.fmt(h));
        sDf.set(S.fmt(q.df, 1), `tr(LᵀL) = ${S.fmt(q.df2, 1)}`);
        sS.set(`${S.fmt(sh)} (${S.fmt(sig.get(), 2)})`);
        sCV.set(`${S.fmt(best.cv)} / ${S.fmt(best.gcv)}`, `oracle: ${S.fmt(hs[S.argmin(ases)])}`);
        sA.set(S.fmt(ase / nn));
      }
      newData();
    },
  });

  /* ------------------------------------------------------------------ */
  /* splines                                                              */
  /* ------------------------------------------------------------------ */
  // B-spline basis values at x for a clamped knot vector t and degree p (Cox–de Boor).
  function bsplineRow(x, t, p) {
    const m = t.length - 1;
    let N = new Array(m).fill(0);
    if (x >= t[m]) { let i = m - 1; while (i > 0 && t[i] === t[m]) i--; N[i] = 1; }
    else for (let i = 0; i < m; i++) if (t[i] <= x && x < t[i + 1]) { N[i] = 1; break; }
    for (let d = 1; d <= p; d++) {
      const M = new Array(m - d).fill(0);
      for (let i = 0; i < m - d; i++) {
        const a = t[i + d] - t[i], b = t[i + d + 1] - t[i + 1];
        M[i] = (a > 0 ? (x - t[i]) / a * N[i] : 0) + (b > 0 ? (t[i + d + 1] - x) / b * N[i + 1] : 0);
      }
      N = M;
    }
    return N;
  }
  Atlas.lab('splines-basis', {
    title: 'Regression splines in a B-spline basis',
    blurb: R`Least-squares fits in a B-spline basis with $K$ equally spaced interior knots (dotted lines) and degree $M$. Degree 0 is a regressogram, degree 1 a broken line, degree 3 a cubic spline with invisible joins. The right panel shows the basis functions — each nonzero over only $M+1$ knot intervals — optionally scaled by their fitted coefficients, which sum to the fit. Too few knots miss the bump; too many chase the noise.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fun = UI.select(controls, { label: 'True regression $r(x)$', options: Object.keys(RFUN).map(k => ({ value: k, label: RFUN[k].label })), value: 'bump', onChange: newData });
      const deg = UI.segmented(controls, { label: 'Degree $M$', options: [{ value: '0', label: '0' }, { value: '1', label: '1' }, { value: '2', label: '2' }, { value: '3', label: '3' }], value: '3', onChange: draw });
      const K = UI.slider(controls, { label: 'Interior knots $K$', min: 0, max: 30, step: 1, value: 6, onInput: draw });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.05, max: 1, step: 0.05, value: 0.25, onChange: newData });
      const scaled = UI.toggle(controls, { label: 'Scale basis by coefficients', value: false, onChange: draw });
      UI.button(controls, 'New data', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', ylabel: 'y', title: 'Least-squares spline fit' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', title: 'B-spline basis functions' });
      const sP = UI.stat(stats, 'Basis functions $K+M+1$', { accent: 'accent' });
      const sR = UI.stat(stats, 'Training RMSE');
      const sT = UI.stat(stats, 'RMSE vs true $r$');
      const n = 150;
      let xs = [], ys = [];
      function newData() {
        const f = RFUN[fun.get()].f;
        xs = sortA(ctx.rng.array(n, () => ctx.rng.u()));
        ys = xs.map(x => f(x) + sig.get() * ctx.rng.normal());
        draw();
      }
      function draw() {
        const p = +deg.get(), k = Math.round(K.get()), f = RFUN[fun.get()].f;
        const inner = S.range(k).map(i => (i + 1) / (k + 1));
        const t = new Array(p + 1).fill(0).concat(inner, new Array(p + 1).fill(1));
        const B = xs.map(x => bsplineRow(x, t, p));
        let beta;
        try { beta = S.ols(B, ys, { lambda: 1e-8, intercept: false }); } catch (e) { beta = B[0].map(() => 0); }
        const grid = S.linspace(0, 1, 301), G = grid.map(x => bsplineRow(x, t, p));
        const fit = G.map(r => dotW(r, beta));
        const yAll = ys.concat(fit);
        const ylo = Math.min(...yAll), yhi = Math.max(...yAll), pad = 0.08 * (yhi - ylo);
        pL.setDomain([0, 1], [ylo - pad, yhi + pad]).clear();
        for (const z of inner) pL.vline(z, { color: 'faint', width: 1, dash: '2 3' });
        pL.points(xs.map((x, i) => [x, ys[i]]), { color: 'muted', r: 2.2, opacity: 0.6 });
        pL.line(grid.map(x => [x, f(x)]), { color: 'ink', width: 1.5, dash: true });
        pL.line(grid.map((x, i) => [x, fit[i]]), { color: 's1', width: 2.4 });
        const nb = beta.length, cols = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'];
        const curves = S.range(nb).map(j => grid.map((x, i) => [x, G[i][j] * (scaled.get() ? beta[j] : 1)]));
        const ys2 = curves.flat().map(q => q[1]);
        const lo2 = Math.min(0, ...ys2), hi2 = Math.max(...ys2);
        pR.setDomain([0, 1], [lo2 - 0.05 * (hi2 - lo2 || 1), hi2 + 0.08 * (hi2 - lo2 || 1)]).clear();
        for (const z of inner) pR.vline(z, { color: 'faint', width: 1, dash: '2 3' });
        curves.forEach((c, j) => pR.line(c, { color: cols[j % 8], width: 1.6, opacity: 0.85 }));
        if (scaled.get()) pR.line(grid.map((x, i) => [x, fit[i]]), { color: 'ink', width: 1.2, dash: true });
        let rss = 0, te = 0;
        xs.forEach((x, i) => { const v = dotW(B[i], beta); rss += (ys[i] - v) ** 2; });
        grid.forEach((x, i) => { te += (fit[i] - f(x)) ** 2; });
        sP.set(String(nb));
        sR.set(S.fmt(Math.sqrt(rss / n)), `noise σ = ${S.fmt(sig.get(), 2)}`);
        sT.set(S.fmt(Math.sqrt(te / grid.length)));
      }
      newData();
    },
  });

  // Cyclic Jacobi eigen-decomposition with a relative stopping rule.
  function jacobiEig(A) {
    const n = A.length, a = A.map(r => r.slice()), V = S.identity(n);
    let norm = 0; for (const r of a) for (const v of r) norm += v * v;
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
      if (off < 1e-26 * norm) break;
      for (let p = 0; p < n; p++) for (let r = p + 1; r < n; r++) {
        if (Math.abs(a[p][r]) < 1e-300) continue;
        const th = (a[r][r] - a[p][p]) / (2 * a[p][r]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akr = a[k][r]; a[k][p] = c * akp - s * akr; a[k][r] = s * akp + c * akr; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], ark = a[r][k]; a[p][k] = c * apk - s * ark; a[r][k] = s * apk + c * ark; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkr = V[k][r]; V[k][p] = c * vkp - s * vkr; V[k][r] = s * vkp + c * vkr; }
      }
    }
    const idx = S.range(n).sort((i, j) => a[i][i] - a[j][j]); // ascending
    return { values: idx.map(i => a[i][i]), vectors: V.map(r => idx.map(i => r[i])) };
  }
  // Smoothing-spline machinery for sorted distinct knots: penalty matrix K = Q R^{-1} Q^T and M = R^{-1} Q^T.
  function splineSetup(xs) {
    const n = xs.length, h = S.range(n - 1).map(i => xs[i + 1] - xs[i]);
    const Q = S.range(n).map(() => new Array(n - 2).fill(0));
    const Rm = S.range(n - 2).map(() => new Array(n - 2).fill(0));
    for (let c = 0; c < n - 2; c++) {
      Q[c][c] = 1 / h[c]; Q[c + 1][c] = -1 / h[c] - 1 / h[c + 1]; Q[c + 2][c] = 1 / h[c + 1];
      Rm[c][c] = (h[c] + h[c + 1]) / 3;
      if (c + 1 < n - 2) { Rm[c][c + 1] = h[c + 1] / 6; Rm[c + 1][c] = h[c + 1] / 6; }
    }
    const M = S.solve(Rm, S.transpose(Q)); // (n-2) x n
    const Kp = S.matmul(Q, M);
    for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) { const v = (Kp[i][j] + Kp[j][i]) / 2; Kp[i][j] = v; Kp[j][i] = v; }
    let mx = 0; for (const r of Kp) for (const v of r) mx = Math.max(mx, Math.abs(v));
    const E = jacobiEig(Kp.map(r => r.map(v => v / mx)));
    const d = E.values.map(v => (Math.abs(v) < 1e-11 ? 0 : v * mx));
    return { h, M, d, U: E.vectors };
  }
  // Evaluate the natural cubic spline with values g at knots xs, second derivatives gam (interior), on x.
  function ncsEval(xs, g, gam, h, x) {
    const n = xs.length, G = [0].concat(gam, [0]);
    if (x <= xs[0]) { const s = (g[1] - g[0]) / h[0] - h[0] * G[1] / 6; return g[0] + s * (x - xs[0]); }
    if (x >= xs[n - 1]) { const s = (g[n - 1] - g[n - 2]) / h[n - 2] + h[n - 2] * G[n - 2] / 6; return g[n - 1] + s * (x - xs[n - 1]); }
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; }
    const i = lo, hh = h[i], a = x - xs[i], b = xs[i + 1] - x;
    return (a * g[i + 1] + b * g[i]) / hh - a * b / 6 * ((1 + a / hh) * G[i + 1] + (1 + b / hh) * G[i]);
  }
  const kappa = u => { const a = Math.abs(u) / Math.SQRT2; return 0.5 * Math.exp(-a) * Math.sin(a + Math.PI / 4); };

  Atlas.lab('splines-smoothing', {
    title: 'The smoothing spline: λ, degrees of freedom, shrinkage and the equivalent kernel',
    blurb: R`The exact smoothing spline (a natural cubic spline with a knot at every $x_i$), computed through the Demmler–Reinsch eigendecomposition of the penalty matrix. Slide $\log_{10}\lambda$ and watch df move between $n$ (interpolation) and 2 (straight line). Right panel: GCV versus df; the shrinkage factors $1/(1+\lambda d_k)$; or row $x_0$ of the smoother matrix (click the left plot) against Silverman's equivalent kernel.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fun = UI.select(controls, { label: 'True regression $r(x)$', options: Object.keys(RFUN).map(k => ({ value: k, label: RFUN[k].label })), value: 'sine', onChange: newData });
      const des = UI.select(controls, { label: 'Design', options: [{ value: 'uniform', label: 'Uniform' }, { value: 'increasing', label: 'Increasing density' }], value: 'uniform', onChange: newData });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 20, max: 80, step: 1, value: 50, onChange: newData });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.05, max: 1, step: 0.05, value: 0.3, onChange: newData });
      const lam = UI.slider(controls, { label: '$\\log_{10}\\lambda$', min: -9, max: 1, step: 0.05, value: -4, onInput: draw });
      const mode = UI.segmented(controls, { label: 'Right panel', options: [{ value: 'gcv', label: 'GCV' }, { value: 'shrink', label: 'Shrinkage' }, { value: 'kernel', label: 'Equivalent kernel' }], value: 'gcv', onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', newData, { primary: true });
      UI.button(row, 'Use GCV λ', () => { lam.set(lgcv); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', ylabel: 'y', title: 'Smoothing spline fit' });
      const pR = new Plot(stage, { width: 420, height: 270, title: 'GCV against effective df', xlabel: 'df', ylabel: 'GCV' });
      const sDf = UI.stat(stats, '$\\mathrm{df}_\\lambda=\\tr S_\\lambda$', { accent: 'accent' });
      const sG = UI.stat(stats, 'GCV choice');
      const sH = UI.stat(stats, 'Equivalent bandwidth $(\\lambda/n)^{1/4}$');
      const sT = UI.stat(stats, 'RMSE vs true $r$');
      let xs = [], ys = [], setup = null, z = [], lgcv = -4, i0 = 0;
      pL.onPointer((x, y, e, type) => { if ((type === 'down' || type === 'drag') && xs.length) { let b = 0; xs.forEach((v, i) => { if (Math.abs(v - x) < Math.abs(xs[b] - x)) b = i; }); i0 = b; if (mode.get() === 'kernel') draw(); } });
      const gDens = x => (des.get() === 'uniform' ? 1 : 0.5 + x);
      function newData() {
        const f = RFUN[fun.get()].f, nn = Math.round(n.get());
        const D = DESIGNS[des.get()];
        do { xs = sortA(ctx.rng.array(nn, () => D.draw(ctx.rng))); } while (xs.some((v, i) => i && v - xs[i - 1] < 1e-5));
        ys = xs.map(x => f(x) + sig.get() * ctx.rng.normal());
        setup = splineSetup(xs);
        z = S.range(nn).map(k => { let s = 0; for (let i = 0; i < nn; i++) s += setup.U[i][k] * ys[i]; return s; });
        i0 = Math.round(nn / 2);
        // GCV over a grid of log-lambda
        const L = S.linspace(-9, 1, 101);
        const g = L.map(l => gcvAt(10 ** l));
        lgcv = L[S.argmin(g.map(q => q.gcv))];
        draw();
      }
      function gcvAt(lambda) {
        const nn = xs.length; let df = 0, rss = 0;
        setup.d.forEach((dk, k) => { const s = 1 / (1 + lambda * dk); df += s; rss += ((1 - s) * z[k]) ** 2; });
        return { df, gcv: rss / nn / (1 - df / nn) ** 2 };
      }
      function draw() {
        const nn = xs.length, lambda = 10 ** lam.get(), f = RFUN[fun.get()].f;
        const s = setup.d.map(dk => 1 / (1 + lambda * dk));
        const gfit = S.range(nn).map(i => { let v = 0; for (let k = 0; k < nn; k++) v += setup.U[i][k] * s[k] * z[k]; return v; });
        const gam = setup.M.map(r => dotW(r, gfit));
        const grid = S.linspace(0, 1, 300), curve = grid.map(x => ncsEval(xs, gfit, gam, setup.h, x));
        const yAll = ys.concat(grid.map(f));
        const ylo = Math.min(...yAll), yhi = Math.max(...yAll), pad = 0.1 * (yhi - ylo);
        pL.setDomain([0, 1], [ylo - pad, yhi + pad]).clear();
        pL.points(xs.map((x, i) => [x, ys[i]]), { color: 'muted', r: 2.6, opacity: 0.7 });
        pL.line(grid.map(x => [x, f(x)]), { color: 'ink', width: 1.5, dash: true });
        pL.line(grid.map((x, i) => [x, curve[i]]), { color: 's1', width: 2.4 });
        if (mode.get() === 'kernel') pL.vline(xs[i0], { color: 's4', width: 1.2 });
        pL.legend([{ label: 'smoothing spline', color: 's1' }, { label: 'true r', color: 'ink', dash: true }], { left: true });
        const df = S.sum(s), hEq = (lambda / nn) ** 0.25;
        const m = mode.get();
        if (m === 'gcv') {
          const L = S.linspace(-9, 1, 101), q = L.map(l => gcvAt(10 ** l));
          const pts = q.map(o => [o.df, o.gcv]).filter(pp => pp[0] < nn - 0.5);
          const gy = pts.map(pp => pp[1]), glo = Math.min(...gy), ghi = Math.min(Math.max(...gy), glo * 6);
          pR.o.logx = false; pR.o.title = 'GCV against effective df'; pR.o.xlabel = 'df'; pR.o.ylabel = 'GCV';
          pR.setDomain([2, nn], [glo * 0.9, ghi]).clear();
          pR.line(pts, { color: 's2', width: 2 });
          pR.vline(df, { color: 's1', dash: false, width: 1.5 });
          pR.hline(sig.get() ** 2, { color: 'muted' });
          pR.text(nn, sig.get() ** 2, 'σ²', { anchor: 'end', dy: -4, dx: -4, size: 11, color: 'muted' });
        } else if (m === 'shrink') {
          pR.o.logx = false; pR.o.title = 'Shrinkage 1/(1+λdₖ) of the k-th Demmler–Reinsch component'; pR.o.xlabel = 'component k'; pR.o.ylabel = 'factor';
          pR.setDomain([0, nn + 1], [0, 1.05]).clear();
          const zmax = Math.max(...z.slice(2).map(Math.abs)) || 1;
          pR.bars(z.map((v, k) => ({ x0: k + 0.6, x1: k + 1.4, y: Math.min(1, Math.abs(v) / zmax) })), { color: 'faint', opacity: 0.6 });
          pR.stems(s.map((v, k) => [k + 1, v]), { color: 's1', r: 2.5, width: 1.5 });
          pR.legend([{ label: 'shrinkage factor', color: 's1' }, { label: '|coefficient of y| (scaled)', color: 'faint', fill: true }]);
        } else {
          const w = S.range(nn).map(j => { let v = 0; for (let k = 0; k < nn; k++) v += setup.U[i0][k] * s[k] * setup.U[j][k]; return v; });
          const hx = (lambda / (nn * gDens(xs[i0]))) ** 0.25;
          const ek = xs.map(xj => kappa((xs[i0] - xj) / hx) / (nn * gDens(xs[i0]) * hx));
          const all = w.concat(ek), lo = Math.min(0, ...all), hi = Math.max(...all);
          pR.o.logx = false; pR.o.title = 'Row x₀ of S_λ vs equivalent kernel'; pR.o.xlabel = 'xⱼ'; pR.o.ylabel = 'weight';
          pR.setDomain([0, 1], [lo - 0.08 * (hi - lo), hi * 1.12]).clear();
          pR.hline(0, { color: 'faint', dash: false, width: 1 });
          const gg = S.linspace(0, 1, 300);
          pR.line(gg.map(x => [x, kappa((xs[i0] - x) / hx) / (nn * gDens(xs[i0]) * hx)]), { color: 'ink', width: 1.4, dash: true });
          pR.stems(xs.map((x, j) => [x, w[j]]), { color: 's1', r: 2.4, width: 1.2 });
          pR.vline(xs[i0], { color: 's4', width: 1.2 });
          pR.legend([{ label: 'smoother weights', color: 's1' }, { label: 'Silverman kernel', color: 'ink', dash: true }]);
        }
        let te = 0; grid.forEach((x, i) => { te += (curve[i] - f(x)) ** 2; });
        sDf.set(S.fmt(df, 2), `λ = ${lambda.toExponential(1)}`);
        sG.set(`log₁₀λ = ${S.fmt(lgcv, 2)}`, `df = ${S.fmt(gcvAt(10 ** lgcv).df, 1)}`);
        sH.set(S.fmt(hEq), 'kernel scale h(x) for uniform design');
        sT.set(S.fmt(Math.sqrt(te / grid.length)));
      }
      newData();
    },
  });

  /* ------------------------------------------------------------------ */
  /* orthogonal-series                                                    */
  /* ------------------------------------------------------------------ */
  const B1 = (x, a, b) => S.pdf.beta(x, a, b);
  const SDENS = {
    beta: { label: 'Beta(2,5)', pdf: x => B1(x, 2, 5), sample: r => r.beta(2, 5) },
    bimodal: { label: 'Two bumps', pdf: x => 0.5 * B1(x, 6, 18) + 0.5 * B1(x, 18, 6), sample: r => (r.u() < 0.5 ? r.beta(6, 18) : r.beta(18, 6)) },
    step: { label: 'Step (discontinuous)', pdf: x => (x < 0 || x > 1 ? 0 : x < 0.5 ? 1.6 : 0.4), sample: r => (r.u() < 0.8 ? 0.5 * r.u() : 0.5 + 0.5 * r.u()) },
    smooth: { label: 'Smooth wave', pdf: x => (x < 0 || x > 1 ? 0 : 1 + 0.6 * Math.cos(2 * Math.PI * x)), sample: r => { for (;;) { const x = r.u(); if (r.u() * 1.6 <= 1 + 0.6 * Math.cos(2 * Math.PI * x)) return x; } } },
  };
  const cosPhi = (j, x) => (j === 0 ? 1 : Math.SQRT2 * Math.cos(Math.PI * j * x));
  const trueCoefCache = {};
  function trueCoefs(key, J) {
    if (trueCoefCache[key]) return trueCoefCache[key];
    const D = SDENS[key], g = 6000, xs = S.linspace(0, 1, g + 1), fv = xs.map(x => Math.min(D.pdf(x), 1e3));
    const b = S.range(J + 1).map(j => trapz(xs.map((x, i) => fv[i] * cosPhi(j, x)), 1 / g));
    return (trueCoefCache[key] = b);
  }
  Atlas.lab('orthogonal-series-cosine', {
    title: 'Cosine-series density estimation and the choice of J',
    blurb: R`Estimate a density on $[0,1]$ by keeping $J$ cosine terms with $\hat\beta_j = \frac1n\sum\phi_j(X_i)$. Right: the estimated risk $\hat R(J)$ (variance of the kept terms plus the estimated squared size of the dropped ones) against this sample's actual ISE. Note the negative dips and, for the step density, the Gibbs ringing that never goes away.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dens = UI.select(controls, { label: 'Density $f$ on $[0,1]$', options: Object.keys(SDENS).map(k => ({ value: k, label: SDENS[k].label })), value: 'bimodal', onChange: newSample });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 30, max: 5000, value: 300, log: true, fmt: v => String(Math.round(v)), onChange: newSample });
      const J = UI.slider(controls, { label: 'Number of terms $J$', min: 1, max: 40, step: 1, value: 6, onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newSample, { primary: true });
      UI.button(row, 'Use estimated Ĵ', () => { J.set(jhat); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', ylabel: 'density', title: 'Projection estimate' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [1, 40], xlabel: 'J', title: 'Estimated risk vs actual ISE' });
      const sJ = UI.stat(stats, 'Estimated $\\hat J$', { accent: 'accent' });
      const sO = UI.stat(stats, 'Oracle $J$ (min ISE)');
      const sI = UI.stat(stats, 'ISE at current $J$');
      const sNeg = UI.stat(stats, 'Minimum of $\\hat f_J$');
      const JM = 40;
      let xs = [], bh = [], vh = [], jhat = 6, Rh = [], ISE = [];
      function newSample() {
        const D = SDENS[dens.get()], nn = Math.round(n.get());
        xs = ctx.rng.array(nn, () => D.sample(ctx.rng));
        bh = [1]; vh = [0];
        for (let j = 1; j <= JM; j++) { const v = xs.map(x => cosPhi(j, x)); bh.push(S.mean(v)); vh.push(S.variance(v)); }
        const bt = trueCoefs(dens.get(), 400);
        let tail = 0; for (let j = JM + 1; j <= 400; j++) tail += bt[j] ** 2;
        Rh = []; ISE = [];
        for (let k = 1; k <= JM; k++) {
          let r = 0, e = 0;
          for (let j = 1; j <= JM; j++) {
            if (j <= k) { r += vh[j] / nn; e += (bh[j] - bt[j]) ** 2; } else { r += Math.max(0, bh[j] ** 2 - vh[j] / nn); e += bt[j] ** 2; }
          }
          Rh.push(r); ISE.push(e + tail);
        }
        jhat = 1 + S.argmin(Rh);
        draw();
      }
      function draw() {
        const D = SDENS[dens.get()], k = Math.round(J.get());
        const grid = S.linspace(0, 1, 400);
        const est = grid.map(x => { let s = 1; for (let j = 1; j <= k; j++) s += bh[j] * cosPhi(j, x); return s; });
        const tv = grid.map(x => Math.min(D.pdf(x), 8));
        const hist = S.histogram(xs, { bins: 40, lo: 0, hi: 1 });
        const ymax = Math.max(...tv, ...est, ...hist.map(b => b.density)) * 1.08, ymin = Math.min(0, ...est) * 1.1;
        pL.setDomain([0, 1], [ymin, ymax]).clear();
        pL.bars(hist, { color: 'faint', opacity: 0.5 });
        pL.hline(0, { color: 'muted', dash: false, width: 1 });
        pL.line(grid.map((x, i) => [x, tv[i]]), { color: 'ink', width: 1.6, dash: true });
        pL.line(grid.map((x, i) => [x, est[i]]), { color: 's1', width: 2.4 });
        pL.legend([{ label: `estimate, J = ${k}`, color: 's1' }, { label: 'true f', color: 'ink', dash: true }]);
        const shift = ISE[ISE.length - 1] - Rh[Rh.length - 1]; // align: the two differ by a constant
        const all = ISE.concat(Rh.map(v => v + shift));
        const lo = Math.min(...all), hi = Math.min(Math.max(...all), lo * 6 + 0.02);
        pR.setDomain([1, JM], [Math.max(0, lo * 0.8), hi]).clear();
        pR.line(ISE.map((v, i) => [i + 1, v]), { color: 'ink', width: 2 });
        pR.line(Rh.map((v, i) => [i + 1, v + shift]), { color: 's2', width: 2 });
        pR.vline(k, { color: 's1', dash: false, width: 1.5 });
        pR.legend([{ label: 'actual ISE', color: 'ink' }, { label: 'estimated risk R̂ (shifted)', color: 's2' }]);
        sJ.set(String(jhat));
        sO.set(String(1 + S.argmin(ISE)));
        sI.set(S.fmt(ISE[k - 1]));
        const mn = Math.min(...est);
        sNeg.set(S.fmt(mn), mn < 0 ? '<span style="color:var(--bad)">negative: not a density</span>' : '');
      }
      newSample();
    },
  });

  // Donoho–Johnstone test signals on [0,1].
  const DJ_T = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81];
  const SIGNALS = {
    blocks: { label: 'Blocks', f: x => { const hb = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]; let s = 0; DJ_T.forEach((t, j) => { s += hb[j] * (x >= t ? 1 : 0); }); return s; } },
    bumps: { label: 'Bumps', f: x => { const hb = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2], w = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005]; let s = 0; DJ_T.forEach((t, j) => { s += hb[j] * (1 + Math.abs((x - t) / w[j])) ** -4; }); return s; } },
    heavisine: { label: 'HeaviSine', f: x => 4 * Math.sin(4 * Math.PI * x) - Math.sign(x - 0.3) - Math.sign(0.72 - x) },
    doppler: { label: 'Doppler', f: x => Math.sqrt(x * (1 - x)) * Math.sin(2.1 * Math.PI / (x + 0.05)) },
    smooth: { label: 'Smooth sine', f: x => Math.sin(2 * Math.PI * x) + 0.5 * Math.cos(6 * Math.PI * x) },
  };
  function haarFwd(y) {
    const n = y.length, out = new Array(n);
    let a = y.slice(), len = n;
    while (len > 1) {
      const h = len / 2, na = new Array(h);
      for (let i = 0; i < h; i++) { na[i] = (a[2 * i] + a[2 * i + 1]) / Math.SQRT2; out[h + i] = (a[2 * i] - a[2 * i + 1]) / Math.SQRT2; }
      a = na; len = h;
    }
    out[0] = a[0];
    return out;
  }
  function haarInv(c) {
    const n = c.length;
    let a = [c[0]], len = 1;
    while (len < n) {
      const na = new Array(2 * len);
      for (let i = 0; i < len; i++) { const s = a[i], d = c[len + i]; na[2 * i] = (s + d) / Math.SQRT2; na[2 * i + 1] = (s - d) / Math.SQRT2; }
      a = na; len *= 2;
    }
    return a;
  }
  Atlas.lab('orthogonal-series-wavelet', {
    title: 'Haar wavelet thresholding versus the best cosine truncation',
    blurb: R`Noisy samples of a test signal on $n=2^J$ equispaced points. The wavelet estimate thresholds Haar detail coefficients at a multiple of the universal threshold $\hat\sigma\sqrt{2\log n}$ ($\hat\sigma$ from the MAD of the finest level; the three coarsest levels are kept). The comparison is the <em>oracle</em> cosine truncation — the best $J$ chosen with knowledge of the truth. On Blocks and Bumps thresholding wins easily; on the smooth sine the linear method wins.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sigS = UI.select(controls, { label: 'Signal', options: Object.keys(SIGNALS).map(k => ({ value: k, label: SIGNALS[k].label })), value: 'blocks', onChange: newData });
      const nS = UI.segmented(controls, { label: 'Sample size $n$', options: ['256', '512', '1024'], value: '512', onChange: newData });
      const noise = UI.slider(controls, { label: 'Noise SD $\\sigma$ (signal SD = 1)', min: 0.05, max: 1, step: 0.05, value: 0.25, onChange: newData });
      const mult = UI.slider(controls, { label: 'Threshold multiplier', min: 0.1, max: 1.5, step: 0.05, value: 1, onInput: draw });
      const rule = UI.segmented(controls, { label: 'Rule', options: [{ value: 'hard', label: 'Hard' }, { value: 'soft', label: 'Soft' }], value: 'hard', onChange: draw });
      const showCos = UI.toggle(controls, { label: 'Show oracle cosine fit', value: true, onChange: draw });
      UI.button(controls, 'New noise', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', title: 'Reconstruction' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'coefficient index (coarse → fine)', title: 'Haar detail coefficients' });
      const sW = UI.stat(stats, 'MSE wavelet', { accent: 'accent' });
      const sC = UI.stat(stats, 'MSE oracle cosine');
      const sK = UI.stat(stats, 'Coefficients kept');
      const sS = UI.stat(stats, '$\\hat\\sigma$ (MAD) / threshold $\\lambda$');
      let xg = [], fv = [], y = [], c = [], cosFit = [], cosJ = 0, cosMSE = 0, sh = 0;
      function newData() {
        const nn = +nS.get(), F = SIGNALS[sigS.get()].f;
        xg = S.range(nn).map(i => (i + 0.5) / nn);
        const raw = xg.map(F), m = S.mean(raw), sd = S.sd(raw, 0);
        fv = raw.map(v => (v - m) / sd);
        y = fv.map(v => v + noise.get() * ctx.rng.normal());
        c = haarFwd(y);
        sh = S.median(c.slice(nn / 2).map(Math.abs)) / 0.6745;
        // oracle cosine truncation
        const Z = S.range(nn).map(j => { let s = 0; for (let i = 0; i < nn; i++) s += y[i] * cosPhi(j, xg[i]); return s / nn; });
        let cur = new Array(nn).fill(0), bestErr = Infinity, bestFit = null;
        for (let j = 0; j < nn; j++) {
          for (let i = 0; i < nn; i++) cur[i] += Z[j] * cosPhi(j, xg[i]);
          let e = 0; for (let i = 0; i < nn; i++) e += (cur[i] - fv[i]) ** 2;
          if (e < bestErr) { bestErr = e; bestFit = cur.slice(); cosJ = j; }
        }
        cosFit = bestFit; cosMSE = bestErr / nn;
        draw();
      }
      function draw() {
        const nn = y.length, lam = mult.get() * sh * Math.sqrt(2 * Math.log(nn)), soft = rule.get() === 'soft';
        let kept = 0;
        const ct = c.map((v, i) => {
          if (i < 8) return v;
          if (Math.abs(v) <= lam) return 0;
          kept++;
          return soft ? Math.sign(v) * (Math.abs(v) - lam) : v;
        });
        const est = haarInv(ct);
        let mse = 0; for (let i = 0; i < nn; i++) mse += (est[i] - fv[i]) ** 2; mse /= nn;
        const all = y.concat(fv), lo = Math.min(...all), hi = Math.max(...all);
        pL.setDomain([0, 1], [lo, hi]).clear();
        pL.points(xg.map((x, i) => [x, y[i]]), { color: 'muted', r: 1.4, opacity: 0.45 });
        pL.line(xg.map((x, i) => [x, fv[i]]), { color: 'ink', width: 1.2, dash: true });
        if (showCos.get()) pL.line(xg.map((x, i) => [x, cosFit[i]]), { color: 's3', width: 1.6 });
        pL.line(xg.map((x, i) => [x, est[i]]), { color: 's1', width: 2 });
        pL.legend([{ label: 'wavelet estimate', color: 's1' }].concat(showCos.get() ? [{ label: `oracle cosine (J = ${cosJ})`, color: 's3' }] : [], [{ label: 'truth', color: 'ink', dash: true }]), { left: true });
        const cap = Math.max(lam * 6, 1);
        pR.setDomain([0, nn], [-cap, cap]).clear();
        pR.rect(8, -lam, nn, lam, { color: 'bad', opacity: 0.08 });
        for (let L = 8; L < nn; L *= 2) pR.vline(L, { color: 'faint', width: 1 });
        const keptPts = [], killed = [];
        for (let i = 1; i < nn; i++) (i < 8 || Math.abs(c[i]) > lam ? keptPts : killed).push([i, S.clamp(c[i], -cap, cap)]);
        pR.points(killed, { color: 'muted', r: 1.4, opacity: 0.5 });
        pR.points(keptPts, { color: 's1', r: 2.2 });
        pR.legend([{ label: 'kept', color: 's1' }, { label: 'killed (|d| ≤ λ)', color: 'bad', fill: true, opacity: 0.2 }]);
        sW.set(S.fmt(mse), `noisy data: ${S.fmt(noise.get() ** 2)}`);
        sC.set(S.fmt(cosMSE), `best J = ${cosJ}`);
        sK.set(`${kept + 8} of ${nn}`);
        sS.set(`${S.fmt(sh)} / ${S.fmt(lam)}`, `true σ = ${S.fmt(noise.get(), 2)}`);
      }
      newData();
    },
  });
})();
