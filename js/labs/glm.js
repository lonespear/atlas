/* Labs: Generalized Linear Models. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;

  /* ================= shared GLM machinery ================= */
  const Phi = x => S.cdf.normal(x), phi = x => S.pdf.normal(x);
  const EPS = 1e-10;
  const LINKS = {
    identity: { inv: e => e, mueta: () => 1, link: m => m },
    log: { inv: e => Math.exp(Math.min(e, 700)), mueta: e => Math.max(Math.exp(Math.min(e, 700)), 1e-300), link: m => Math.log(m) },
    logit: { inv: e => 1 / (1 + Math.exp(-e)), mueta: e => { const p = 1 / (1 + Math.exp(-e)); return Math.max(p * (1 - p), 1e-300); }, link: m => Math.log(m / (1 - m)) },
    probit: { inv: e => Phi(e), mueta: e => Math.max(phi(e), 1e-300), link: m => S.qnorm(m) },
    cloglog: { inv: e => 1 - Math.exp(-Math.exp(Math.min(e, 700))), mueta: e => Math.max(Math.exp(e - Math.exp(Math.min(e, 700))), 1e-300), link: m => Math.log(-Math.log(1 - m)) },
  };
  const xlogy = (x, y) => (x > 0 ? x * Math.log(y) : 0);
  const FAM = {
    binomial: { V: m => m * (1 - m), clamp: m => S.clamp(m, EPS, 1 - EPS), init: (y, w) => (w * y + 0.5) / (w + 1),
      dev: (y, m, w) => 2 * w * (xlogy(y, y / m) + xlogy(1 - y, (1 - y) / (1 - m))) },
    poisson: { V: m => m, clamp: m => Math.max(m, EPS), init: y => y + 0.1,
      dev: (y, m, w) => 2 * w * (xlogy(y, y / m) - (y - m)) },
    gaussian: { V: () => 1, clamp: m => m, init: y => y, dev: (y, m, w) => w * (y - m) ** 2 },
  };
  const negbin = k => ({ V: m => m + m * m / k, clamp: m => Math.max(m, EPS), init: y => y + 0.1,
    dev: (y, m, w) => 2 * w * (xlogy(y, y / m) - (y + k) * Math.log((y + k) / (m + k))) });

  function crossprod(X, W) { // X'WX
    const p = X[0].length, A = Array.from({ length: p }, () => new Array(p).fill(0));
    for (let i = 0; i < X.length; i++) { const x = X[i], w = W ? W[i] : 1; for (let a = 0; a < p; a++) { const v = w * x[a]; if (v) for (let b = a; b < p; b++) A[a][b] += v * x[b]; } }
    for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) A[a][b] = A[b][a];
    return A;
  }
  function crossvec(X, W, z) { const p = X[0].length, v = new Array(p).fill(0); for (let i = 0; i < X.length; i++) { const w = (W ? W[i] : 1) * z[i]; for (let a = 0; a < p; a++) v[a] += w * X[i][a]; } return v; }
  const safeInv = A => { try { return S.inv(A); } catch (e) { return null; } };

  // General IRLS with step-halving. Returns {beta, mu, eta, dev, iter, converged, info, cov, path}.
  function glmFit(X, y, o = {}) {
    const fam = typeof o.family === 'object' ? o.family : FAM[o.family || 'gaussian'];
    const L = LINKS[o.link || 'identity'];
    const n = y.length, p = X[0].length;
    const w = o.weights || new Array(n).fill(1), off = o.offset || new Array(n).fill(0);
    const maxit = o.maxit ?? 50, tol = o.tol ?? 1e-10, P = o.penalty || null;
    const pen = b => (P ? b.reduce((s, bi, a) => s + bi * b.reduce((t, bj, c) => t + P[a][c] * bj, 0), 0) : 0);
    const devOf = mu => { let d = 0; for (let i = 0; i < n; i++) d += fam.dev(y[i], mu[i], w[i]); return d; };
    let eta = new Array(n), mu = new Array(n), beta = null;
    const evalAt = b => {
      const e = new Array(n), m = new Array(n);
      for (let i = 0; i < n; i++) { let s = off[i]; for (let a = 0; a < p; a++) s += X[i][a] * b[a]; e[i] = s; m[i] = fam.clamp(L.inv(s)); }
      return [e, m];
    };
    if (o.start) { beta = o.start.slice(); [eta, mu] = evalAt(beta); }
    else for (let i = 0; i < n; i++) { mu[i] = fam.clamp(fam.init(y[i], w[i])); eta[i] = L.link(mu[i]); }
    let obj = devOf(mu) + (beta ? pen(beta) : 0), it = 0, converged = false;
    const path = beta ? [beta.slice()] : [];
    for (it = 1; it <= maxit; it++) {
      const W = new Array(n), z = new Array(n);
      for (let i = 0; i < n; i++) { const d = L.mueta(eta[i]); W[i] = w[i] * d * d / Math.max(fam.V(mu[i]), 1e-300); z[i] = eta[i] - off[i] + (y[i] - mu[i]) / d; }
      const A = crossprod(X, W);
      if (P) for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) A[a][b] += P[a][b];
      let nb;
      try { nb = S.solve(A, crossvec(X, W, z)); } catch (e) { break; }
      if (!nb.every(Number.isFinite)) break;
      let step = 1, accepted = false, ne, nm, nobj, bb;
      for (let h = 0; h < 30; h++) {
        bb = beta ? nb.map((v, a) => beta[a] + step * (v - beta[a])) : nb;
        [ne, nm] = evalAt(bb);
        nobj = devOf(nm) + pen(bb);
        if (Number.isFinite(nobj) && (!beta || o.noHalving || nobj <= obj + 1e-9 * (1 + Math.abs(obj)))) { accepted = true; break; }
        step /= 2;
      }
      if (!accepted) break;
      beta = bb; eta = ne; mu = nm; path.push(beta.slice());
      const change = Math.abs(nobj - obj) / (Math.abs(nobj) + 0.1);
      obj = nobj;
      if (change < tol && it > 1) { converged = true; break; }
    }
    if (!beta) beta = new Array(p).fill(0);
    const W = new Array(n);
    for (let i = 0; i < n; i++) { const d = L.mueta(eta[i]); W[i] = w[i] * d * d / Math.max(fam.V(mu[i]), 1e-300); }
    const info = crossprod(X, W);
    const infoP = P ? info.map((r, a) => r.map((v, b) => v + P[a][b])) : info;
    return { beta, eta, mu, W, dev: devOf(mu), iter: it, converged, info, cov: safeInv(infoP), path };
  }
  function pearson(y, mu, V, w) { let s = 0; for (let i = 0; i < y.length; i++) s += (w ? w[i] : 1) * (y[i] - mu[i]) ** 2 / V(mu[i]); return s; }
  const se = (cov, j) => (cov && cov[j][j] > 0 ? Math.sqrt(cov[j][j]) : NaN);

  // Binary logistic log-likelihood for (b0, b1) with scalar x.
  function logitLL(x, y, b0, b1) {
    let l = 0;
    for (let i = 0; i < x.length; i++) { const e = b0 + b1 * x[i]; l += y[i] * e - (e > 0 ? e + Math.log1p(Math.exp(-e)) : Math.log1p(Math.exp(e))); }
    return l;
  }
  // Newton for 2-parameter logistic. Returns {b, cov, iter, path, ok}.
  function logitNewton(x, y, o = {}) {
    let b = o.start ? o.start.slice() : [0, 0];
    const path = [b.slice()], maxit = o.maxit ?? 60, firth = !!o.firth;
    let ok = false, cov = null, it;
    for (it = 1; it <= maxit; it++) {
      let g0 = 0, g1 = 0, h00 = 0, h01 = 0, h11 = 0;
      const pis = new Array(x.length);
      for (let i = 0; i < x.length; i++) { const p = 1 / (1 + Math.exp(-(b[0] + b[1] * x[i]))); pis[i] = p; const w = p * (1 - p); h00 += w; h01 += w * x[i]; h11 += w * x[i] * x[i]; }
      const det = h00 * h11 - h01 * h01;
      if (!(det > 1e-300)) break;
      const i00 = h11 / det, i01 = -h01 / det, i11 = h00 / det;
      for (let i = 0; i < x.length; i++) {
        let r = y[i] - pis[i];
        if (firth) { const w = pis[i] * (1 - pis[i]); const hii = w * (i00 + 2 * i01 * x[i] + i11 * x[i] * x[i]); r += hii * (0.5 - pis[i]); }
        g0 += r; g1 += r * x[i];
      }
      let d0 = i00 * g0 + i01 * g1, d1 = i01 * g0 + i11 * g1;
      const mx = Math.max(Math.abs(d0), Math.abs(d1));
      if (firth && mx > 5) { d0 *= 5 / mx; d1 *= 5 / mx; }
      b = [b[0] + d0, b[1] + d1];
      path.push(b.slice());
      cov = [[i00, i01], [i01, i11]];
      if (mx < 1e-9) { ok = true; break; }
      if (!Number.isFinite(b[0]) || Math.abs(b[1]) > 1e6) break;
    }
    return { b, cov, iter: it, path, ok };
  }
  // 1-D separation check for logistic with one covariate: returns 0 (overlap), +1 (ones to the right), -1 (ones to the left).
  function separation(x, y) {
    let max0 = -Infinity, min0 = Infinity, max1 = -Infinity, min1 = Infinity, n1 = 0;
    for (let i = 0; i < x.length; i++) { if (y[i]) { n1++; max1 = Math.max(max1, x[i]); min1 = Math.min(min1, x[i]); } else { max0 = Math.max(max0, x[i]); min0 = Math.min(min0, x[i]); } }
    if (n1 === 0 || n1 === x.length) return 2;
    if (max0 <= min1) return 1;
    if (max1 <= min0) return -1;
    return 0;
  }
  // Profile log-likelihood of the slope (maximize over intercept by 1-D Newton). Returns [ll, b0].
  function profileLL(x, y, b1, b0start = 0) {
    let b0 = b0start;
    for (let k = 0; k < 60; k++) {
      let g = 0, h = 0;
      for (let i = 0; i < x.length; i++) { const p = 1 / (1 + Math.exp(-(b0 + b1 * x[i]))); g += y[i] - p; h += p * (1 - p); }
      if (!(h > 1e-300)) break;
      let d = g / h; d = S.clamp(d, -5, 5);
      b0 += d;
      if (Math.abs(d) < 1e-10) break;
    }
    return [logitLL(x, y, b0, b1), b0];
  }
  // Randomized-free helpers
  const hist = (vals, lo, hi, bins) => S.histogram(vals.filter(v => Number.isFinite(v) && v >= lo && v <= hi), { bins, lo, hi }).map(b => ({ ...b, density: vals.length ? b.count / (vals.length * (b.x1 - b.x0)) : 0 }));
  const pct = v => (100 * v).toFixed(1) + '%';
  const colorSpan = (ok, txt) => `<span style="color:var(--${ok ? 'good' : 'bad'})">${txt}</span>`;

  /* ================= glm-framework ================= */
  const EDF = {
    normal: { label: 'Normal (σ = 1)', th: [-3, 3], b: t => t * t / 2, b1: t => t, b2: () => 1, a: 1, disc: false,
      dens: (y, mu) => phi(y - mu), yr: mu => [mu - 4, mu + 4] },
    poisson: { label: 'Poisson', th: [-1.5, 2.6], b: t => Math.exp(t), b1: t => Math.exp(t), b2: t => Math.exp(t), a: 1, disc: true,
      dens: (y, mu) => S.pmf.poisson(y, mu), yr: mu => [0, Math.ceil(Math.max(8, mu + 4.5 * Math.sqrt(mu)))] },
    bernoulli: { label: 'Bernoulli', th: [-5, 5], b: t => Math.log1p(Math.exp(t)), b1: t => 1 / (1 + Math.exp(-t)), b2: t => { const p = 1 / (1 + Math.exp(-t)); return p * (1 - p); }, a: 1, disc: true,
      dens: (y, mu) => (y === 1 ? mu : y === 0 ? 1 - mu : 0), yr: () => [0, 1] },
    gamma: { label: 'Gamma (shape ν = 2)', th: [-3, -0.15], b: t => -Math.log(-t), b1: t => -1 / t, b2: t => 1 / (t * t), a: 0.5, disc: false,
      dens: (y, mu) => S.pdf.gamma(y, 2, mu / 2), yr: mu => [0, Math.max(4, 4 * mu)] },
    invgauss: { label: 'Inverse Gaussian (λ = 1)', th: [-2, -0.05], b: t => -Math.sqrt(-2 * t), b1: t => 1 / Math.sqrt(-2 * t), b2: t => (-2 * t) ** -1.5, a: 1, disc: false,
      dens: (y, mu) => (y <= 0 ? 0 : Math.sqrt(1 / (2 * Math.PI * y ** 3)) * Math.exp(-((y - mu) ** 2) / (2 * mu * mu * y))), yr: mu => [0, Math.max(3, 3 * mu)] },
  };
  Atlas.lab('glm-framework-cumulant', {
    title: 'The cumulant function $b(\\theta)$ generates the mean and variance',
    blurb: 'Left: $b(\\theta)$ with its tangent at the chosen $\\theta$ (slope $= \\mu = b\'(\\theta)$) and the osculating parabola (curvature $= b\'\'(\\theta) = V(\\mu)$). Right: the distribution at that $\\theta$, with mean and $\\pm1$ SD. Slide $\\theta$ and watch the variance follow the mean as $V(\\mu)$ dictates.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fam = UI.select(controls, { label: 'Family', options: Object.keys(EDF).map(k => ({ value: k, label: EDF[k].label })), value: 'poisson', onChange: () => { th.set(th.get()); draw(); } });
      const thOf = t => { const F = EDF[fam.get()]; return F.th[0] + t * (F.th[1] - F.th[0]); };
      const th = UI.slider(controls, { label: 'Natural parameter $\\theta$', min: 0, max: 1, step: 0.005, value: 0.55, onInput: draw, fmt: t => S.fmt(thOf(t), 3) });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'θ', ylabel: 'b(θ)', title: 'Cumulant function' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'y', ylabel: 'density / pmf', title: 'Distribution at θ' });
      const sMu = UI.stat(stats, "Mean $\\mu = b'(\\theta)$", { accent: 'accent' });
      const sV = UI.stat(stats, "$V(\\mu) = b''(\\theta)$");
      const sVar = UI.stat(stats, '$\\Var Y = a(\\phi)V(\\mu)$');
      const note = UI.note(stats, '');
      function draw() {
        const F = EDF[fam.get()], t0 = thOf(th.get());
        const mu = F.b1(t0), V = F.b2(t0), vr = F.a * V;
        const grid = S.linspace(F.th[0], F.th[1], 200), bv = grid.map(F.b);
        const [lo, hi] = S.extent(bv), pad = (hi - lo) * 0.1 + 1e-6;
        pL.setDomain(F.th, [lo - pad, hi + pad]).clear();
        pL.line(grid.map((g, i) => [g, bv[i]]), { color: 'ink', width: 2 });
        const b0 = F.b(t0);
        pL.line(grid.map(g => [g, b0 + mu * (g - t0)]), { color: 's1', width: 1.6 });
        pL.line(grid.map(g => [g, b0 + mu * (g - t0) + 0.5 * V * (g - t0) ** 2]), { color: 's2', width: 1.6, dash: true });
        pL.dot(t0, b0, { color: 's1', r: 5 });
        pL.legend([{ label: 'b(θ)', color: 'ink' }, { label: 'tangent, slope μ', color: 's1' }, { label: 'curvature V(μ)', color: 's2', dash: true }], { left: true });
        const [y0, y1] = F.yr(mu), sd = Math.sqrt(vr);
        if (F.disc) {
          const ks = []; for (let k = Math.max(0, Math.floor(y0)); k <= y1; k++) ks.push([k, F.dens(k, mu)]);
          const ym = Math.max(...ks.map(k => k[1])) * 1.15;
          pR.setDomain([y0 - 0.6, y1 + 0.6], [0, ym]).clear();
          pR.band(mu - sd, mu + sd, { color: 's2', opacity: 0.12 });
          pR.stems(ks, { color: 's1' });
        } else {
          const ys = S.linspace(y0, y1, 300), dv = ys.map(v => F.dens(v, mu)).map(v => (Number.isFinite(v) ? v : 0));
          const ym = Math.min(Math.max(...dv) * 1.15, 3);
          pR.setDomain([y0, y1], [0, ym]).clear();
          pR.band(mu - sd, mu + sd, { color: 's2', opacity: 0.12 });
          pR.area(ys.map((v, i) => [v, dv[i]]), { color: 's1', opacity: 0.2 });
          pR.line(ys.map((v, i) => [v, dv[i]]), { color: 's1', width: 2 });
        }
        pR.vline(mu, { color: 'ink', dash: false, width: 1.5 });
        sMu.set(S.fmt(mu, 4));
        sV.set(S.fmt(V, 4));
        sVar.set(S.fmt(vr, 4), `SD ${S.fmt(sd, 3)}`);
        const vtxt = { normal: 'V(\\mu) = 1', poisson: 'V(\\mu) = \\mu', bernoulli: 'V(\\mu) = \\mu(1-\\mu)', gamma: 'V(\\mu) = \\mu^2', invgauss: 'V(\\mu) = \\mu^3' }[fam.get()];
        note.set(`$\\theta = ${S.fmt(t0, 3)}$. Here $${vtxt}$: check that $b''(\\theta)$ equals ${S.fmt(V, 4)} and $\\mu$ gives ${S.fmt({ normal: 1, poisson: mu, bernoulli: mu * (1 - mu), gamma: mu * mu, invgauss: mu ** 3 }[fam.get()], 4)}.`);
      }
      draw();
    },
  });

  const TRUE_LINKS = { logit: [0, 1.5], probit: [0, 0.9], cloglog: [-0.5, 0.8] };
  Atlas.lab('glm-framework-links', {
    title: 'Five links fitted to the same binary data',
    blurb: 'Binary responses are generated from the chosen true link; every link is then fitted by maximum likelihood. Dots on the floor and ceiling are the data, open circles are binned proportions. Compare the deviances: symmetric links are hard to tell apart; identity and log leave $[0,1]$ or bend the wrong way.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const truth = UI.segmented(controls, { label: 'True link', options: ['logit', 'probit', 'cloglog'], value: 'cloglog', onChange: gen });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 50, max: 2000, step: 50, value: 300, onChange: gen });
      UI.button(controls, 'New data', gen, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, x: [-3, 3], y: [-0.15, 1.15], xlabel: 'x', ylabel: 'P(Y = 1 | x)' });
      const fits = ['identity', 'log', 'logit', 'probit', 'cloglog'], cols = ['s5', 's4', 's1', 's2', 's3'];
      const tiles = fits.map(f => UI.stat(stats, `Deviance: ${f}`));
      let x = [], y = [], jit = [];
      function gen() {
        const L = LINKS[truth.get()], [a, b] = TRUE_LINKS[truth.get()], nn = Math.round(n.get());
        x = ctx.rng.array(nn, () => ctx.rng.uniform(-3, 3));
        y = x.map(v => (ctx.rng.u() < L.inv(a + b * v) ? 1 : 0));
        jit = x.map(() => ctx.rng.uniform(-0.035, 0.035));
        draw();
      }
      function draw() {
        const X = x.map(v => [1, v]);
        p.clear();
        p.hline(0, { color: 'faint', dash: false, width: 1 }); p.hline(1, { color: 'faint', dash: false, width: 1 });
        p.points(x.map((v, i) => [v, y[i] ? 1.06 + jit[i] : -0.06 + jit[i]]), { color: 'muted', r: 1.8, opacity: 0.5 });
        const bins = S.histogram(x, { bins: 12, lo: -3, hi: 3 });
        bins.forEach(bn => { const idx = x.map((v, i) => i).filter(i => x[i] >= bn.x0 && x[i] < bn.x1 + (bn.x1 >= 3 ? 1e-9 : 0)); if (idx.length) p.dot((bn.x0 + bn.x1) / 2, S.mean(idx.map(i => y[i])), { color: 'surface', stroke: 'ink', r: 4.5, strokeWidth: 1.5 }); });
        const [a, b] = TRUE_LINKS[truth.get()], Lt = LINKS[truth.get()], grid = S.linspace(-3, 3, 160);
        p.line(grid.map(g => [g, Lt.inv(a + b * g)]), { color: 'ink', width: 2, dash: '6 4' });
        const devs = [];
        fits.forEach((f, k) => {
          const r = glmFit(X, y, { family: 'binomial', link: f, maxit: 100 });
          devs.push(r.dev);
          p.line(grid.map(g => [g, LINKS[f].inv(r.beta[0] + r.beta[1] * g)]), { color: cols[k], width: 2 });
        });
        const best = S.argmin(devs);
        fits.forEach((f, k) => tiles[k].set(S.fmt(devs[k], 5), k === best ? '<b>smallest</b>' : `+${S.fmt(devs[k] - devs[best], 3)}`));
        p.legend(fits.map((f, k) => ({ label: f, color: cols[k] })).concat([{ label: 'truth', color: 'ink' }]), { left: true });
      }
      gen();
    },
  });

  /* ================= logistic-regression ================= */
  Atlas.lab('logistic-regression-sandbox', {
    title: 'Logistic regression sandbox: likelihood surface and separation',
    blurb: 'Click in the upper half of the left panel to add a $y=1$ point, the lower half for $y=0$; click on an existing point to remove it. Right: the log-likelihood over $(\\beta_0,\\beta_1)$, with the likelihood-ratio 95% region (solid) and the Wald ellipse (dashed). Press "Separate" to see the ridge run off to infinity, then turn on Firth.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Overlapping data', () => { randomData(false); draw(); }, { primary: true });
      UI.button(row, 'Separate', () => { randomData(true); draw(); });
      UI.button(row, 'Clear', () => { pts = []; draw(); });
      const firth = UI.toggle(controls, { label: 'Firth (Jeffreys-penalized) estimate', value: false, onChange: draw });
      const showPath = UI.toggle(controls, { label: 'Show Newton iterates', value: true, onChange: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [-0.12, 1.12], xlabel: 'x', ylabel: 'P(Y = 1)', title: 'Data and fitted curve (click to edit)' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'β₀', ylabel: 'β₁', title: 'Log-likelihood ℓ(β₀, β₁)' });
      const sB = UI.stat(stats, 'MLE $(\\hat\\beta_0, \\hat\\beta_1)$', { accent: 'accent' });
      const sSe = UI.stat(stats, '$\\widehat\\se(\\hat\\beta_1)$, Wald $z$');
      const sOR = UI.stat(stats, 'Odds ratio $e^{\\hat\\beta_1}$');
      const sSt = UI.stat(stats, 'Status');
      let pts = [];
      function randomData(sep) {
        pts = [];
        for (let i = 0; i < 24; i++) {
          const xv = ctx.rng.uniform(-3.5, 3.5);
          const yv = sep ? (xv > 0.2 ? 1 : 0) : (ctx.rng.u() < 1 / (1 + Math.exp(-(0.3 + 1.1 * xv))) ? 1 : 0);
          pts.push([xv, yv]);
        }
      }
      pL.onPointer((px, py, e, type) => {
        if (type !== 'down') return;
        const cls = py > 0.5 ? 1 : 0;
        const k = pts.findIndex(q => q[1] === cls && Math.abs(q[0] - px) < 0.12 && Math.abs((cls ? 1 : 0) - py) < 0.15);
        if (k >= 0) pts.splice(k, 1); else if (px > -4 && px < 4) pts.push([px, cls]);
        draw();
      });
      function draw() {
        const x = pts.map(q => q[0]), y = pts.map(q => q[1]);
        pL.clear();
        pL.hline(0, { color: 'faint', dash: false, width: 1 }); pL.hline(1, { color: 'faint', dash: false, width: 1 });
        pL.hline(0.5, { color: 'faint', width: 1 });
        pts.forEach(q => pL.dot(q[0], q[1], { color: q[1] ? 's1' : 's3', r: 5 }));
        pR.clear();
        const sep = x.length ? separation(x, y) : 2;
        if (x.length < 3 || sep === 2) {
          pR.setDomain([-5, 5], [-5, 5]).clear();
          sB.set('—'); sSe.set('—'); sOR.set('—'); sSt.set('Need both classes', 'add points');
          return;
        }
        const fit = logitNewton(x, y, { maxit: sep ? 25 : 60 });
        const fF = firth.get() ? logitNewton(x, y, { firth: true, maxit: 200 }) : null;
        const grid = S.linspace(-4, 4, 200);
        const curve = b => grid.map(g => [g, 1 / (1 + Math.exp(-(b[0] + b[1] * g)))]);
        pL.line(curve(fit.b), { color: 's1', width: 2.2, dash: sep ? true : false });
        if (fF) pL.line(curve(fF.b), { color: 's2', width: 2.2 });
        // Surface
        let xr, yr, lref;
        if (!sep) {
          const s0 = Math.max(se(fit.cov, 0) * 4, 1.2), s1 = Math.max(se(fit.cov, 1) * 4, 1.2);
          xr = [fit.b[0] - s0, fit.b[0] + s0]; yr = [fit.b[1] - s1, fit.b[1] + s1];
          lref = logitLL(x, y, fit.b[0], fit.b[1]);
        } else {
          const ctr = fF ? fF.b : [0, sep * 4];
          xr = [ctr[0] - 8, ctr[0] + 8]; yr = sep > 0 ? [-1, 18] : [-18, 1];
          lref = 0;
        }
        pR.setDomain(xr, yr).clear();
        const c1 = Atlas.rgb('s1'), bg = Atlas.rgb('surface');
        pR.raster((b0, b1) => { const v = logitLL(x, y, b0, b1) - lref; return Atlas.mix(bg, c1, Math.exp(v / 6) * 0.85).concat([255]); }, { nx: 70, ny: 55 });
        const f = (b0, b1) => logitLL(x, y, b0, b1);
        pR.contour(f, [1, 4, 9, 16, 25, 40].map(d => lref - d), { color: 'muted', nx: 50, ny: 40, width: 0.8 });
        pR.contour(f, [lref - 2.996], { color: 'ink', nx: 70, ny: 55, width: 2 });
        if (!sep && fit.cov) pR.ellipse(fit.b[0], fit.b[1], fit.cov, Math.sqrt(5.991), { color: 's4', width: 2, dash: true });
        if (showPath.get()) { pR.line(fit.path, { color: 's1', width: 1.4, opacity: 0.9 }); fit.path.forEach(q => pR.dot(q[0], q[1], { color: 's1', r: 2.5 })); }
        if (!sep) pR.dot(fit.b[0], fit.b[1], { color: 's1', r: 5.5 });
        if (fF) pR.dot(fF.b[0], fF.b[1], { color: 's2', r: 5.5 });
        if (sep) {
          sB.set('does not exist', `iterate ${fit.iter}: (${S.fmt(fit.b[0], 3)}, ${S.fmt(fit.b[1], 3)})`);
          sSe.set(S.fmt(se(fit.cov, 1), 3), `z = ${S.fmt(fit.b[1] / se(fit.cov, 1), 3)} (meaningless)`);
          sOR.set('∞', fF ? `Firth: ${S.fmt(Math.exp(fF.b[1]), 3)}` : 'try Firth');
          sSt.set(colorSpan(false, 'separated'), 'ℓ increases forever along a ray');
        } else {
          sB.set(`(${S.fmt(fit.b[0], 3)}, ${S.fmt(fit.b[1], 3)})`, fF ? `Firth: (${S.fmt(fF.b[0], 3)}, ${S.fmt(fF.b[1], 3)})` : `${fit.iter} Newton steps`);
          sSe.set(S.fmt(se(fit.cov, 1), 3), `z = ${S.fmt(fit.b[1] / se(fit.cov, 1), 3)}`);
          sOR.set(S.fmt(Math.exp(fit.b[1]), 4), `95% Wald: ${S.fmt(Math.exp(fit.b[1] - 1.96 * se(fit.cov, 1)), 3)} – ${S.fmt(Math.exp(fit.b[1] + 1.96 * se(fit.cov, 1)), 3)}`);
          sSt.set(colorSpan(true, 'overlap: MLE exists'), `n = ${x.length}`);
        }
      }
      randomData(false);
      draw();
    },
  });

  Atlas.lab('logistic-regression-casecontrol', {
    title: 'Case-control sampling shifts only the intercept',
    blurb: 'A population of 20,000 has a rare outcome; reference lines are the logistic fit to the whole population. Each experiment samples half the cases and enough controls to give the chosen controls-per-case ratio, then fits a logistic regression. The slope estimates centre on the population log odds ratio; the intercept centres on $\\beta_0 + \\log(s_1/s_0)$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const b0s = UI.slider(controls, { label: 'Population intercept $\\beta_0$', min: -6, max: -1, step: 0.25, value: -4, onChange: popn });
      const b1s = UI.slider(controls, { label: 'Population slope $\\beta_1$', min: 0, max: 2, step: 0.05, value: 0.8, onChange: popn });
      const ratio = UI.slider(controls, { label: 'Controls per case', min: 1, max: 5, step: 1, value: 2, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run studies', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pA = new Plot(stage, { width: 420, height: 270, xlabel: 'slope estimate', ylabel: 'density', title: 'Slope β̂₁ across case-control studies' });
      const pB = new Plot(stage, { width: 420, height: 270, xlabel: 'intercept estimate', ylabel: 'density', title: 'Intercept β̂₀ across studies' });
      const sN = UI.stat(stats, 'Studies');
      const sS = UI.stat(stats, 'Mean $\\hat\\beta_1$', { accent: 'accent' });
      const sI = UI.stat(stats, 'Mean $\\hat\\beta_0$');
      const sP = UI.stat(stats, 'Sample vs population prevalence');
      let cases = [], ctrls = [], s1 = 0.5, s0 = 0, est1 = [], est0 = [], prevS = 0, popB = [0, 0];
      function popn() {
        const N = 20000, r = S.rng(7);
        cases = []; ctrls = [];
        for (let i = 0; i < N; i++) { const xv = r.normal(); if (r.u() < 1 / (1 + Math.exp(-(b0s.get() + b1s.get() * xv)))) cases.push(xv); else ctrls.push(xv); }
        // The finite population's own logistic fit is the estimand the studies target.
        const f = logitNewton(cases.concat(ctrls), cases.map(() => 1).concat(ctrls.map(() => 0)), { start: [b0s.get(), b1s.get()] });
        popB = f.ok ? f.b : [b0s.get(), b1s.get()];
        reset();
      }
      function reset() {
        loop.stop(); est1 = []; est0 = [];
        const n1 = Math.max(5, Math.round(s1 * cases.length));
        s0 = Math.min(1, ratio.get() * n1 / ctrls.length);
        prevS = n1 / (n1 + Math.round(s0 * ctrls.length));
        draw();
      }
      function study() {
        const n1 = Math.max(5, Math.round(s1 * cases.length)), n0 = Math.max(5, Math.round(s0 * ctrls.length));
        const x = [], y = [];
        for (let i = 0; i < n1; i++) { x.push(cases[(ctx.rng.u() * cases.length) | 0]); y.push(1); }
        for (let i = 0; i < n0; i++) { x.push(ctrls[(ctx.rng.u() * ctrls.length) | 0]); y.push(0); }
        const f = logitNewton(x, y, { start: [Math.log(n1 / n0), 0] });
        if (f.ok) { est1.push(f.b[1]); est0.push(f.b[0]); }
      }
      let acc = 0;
      function step(dt) {
        acc += dt;
        const per = Math.max(1, Math.round(30000 / (cases.length * (1 + ratio.get()) * s1 + 1)));
        for (let i = 0; i < per; i++) study();
        if (est1.length >= 1500) { draw(); return false; }
        if (acc > 0.06) { acc = 0; draw(); }
      }
      function draw() {
        const b1 = popB[1], b0 = popB[0], shift = Math.log(s1 / s0);
        const lo1 = b1 - 1, hi1 = b1 + 1;
        const h1 = hist(est1, lo1, hi1, 40);
        pA.setDomain([lo1, hi1], [0, Math.max(1, ...h1.map(b => b.density)) * 1.15]).clear();
        pA.bars(h1, { color: 's1', opacity: 0.55 });
        pA.vline(b1, { color: 'ink', dash: false, width: 2 });
        pA.legend([{ label: 'population fit β₁', color: 'ink' }]);
        const lo0 = Math.min(b0, b0 + shift) - 1.2, hi0 = Math.max(b0, b0 + shift) + 1.2;
        const h0 = hist(est0, lo0, hi0, 50);
        pB.setDomain([lo0, hi0], [0, Math.max(1, ...h0.map(b => b.density)) * 1.15]).clear();
        pB.bars(h0, { color: 's2', opacity: 0.55 });
        pB.vline(b0, { color: 'ink', dash: false, width: 2 });
        pB.vline(b0 + shift, { color: 's2', dash: true, width: 2 });
        pB.legend([{ label: 'population fit β₀', color: 'ink' }, { label: 'β₀ + log(s₁/s₀)', color: 's2', dash: true }]);
        sN.set(String(est1.length), `${cases.length} cases in population`);
        sS.set(est1.length ? S.fmt(S.mean(est1), 4) : '—', `population fit ${S.fmt(b1, 4)}`);
        sI.set(est0.length ? S.fmt(S.mean(est0), 4) : '—', `predicted ${S.fmt(b0 + shift, 4)}`);
        sP.set(`${pct(prevS)} vs ${pct(cases.length / 20000)}`, 'the intercept tracks the sample, not the population');
      }
      popn();
    },
  });

  /* ================= poisson-regression ================= */
  Atlas.lab('poisson-regression-offset', {
    title: 'Exposure, offsets and a biased slope',
    blurb: 'Regions have different exposures $t_i$ and exposure is related to the covariate through $\\gamma$. Left: observed log rates with the offset fit (rate model) and the no-offset fit. Right: slope estimates over repeated data sets. Without the offset the slope estimates $\\beta_1 + \\gamma$, not $\\beta_1$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const gam = UI.slider(controls, { label: 'Exposure–covariate link $\\gamma$', min: -1.5, max: 1.5, step: 0.05, value: 0.8, onChange: reset });
      const b1s = UI.slider(controls, { label: 'True log rate ratio $\\beta_1$', min: -1, max: 1, step: 0.05, value: 0.3, onChange: reset });
      const n = UI.slider(controls, { label: 'Regions $n$', min: 30, max: 400, step: 10, value: 80, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', () => { one(true); draw(); });
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Repeat', '❚❚ Pause']);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-3, 3], xlabel: 'x', ylabel: 'log((y + 0.5)/t)', title: 'One data set: log rate vs x' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'slope estimate', ylabel: 'density', title: 'Sampling distribution of β̂₁' });
      const sO = UI.stat(stats, 'With offset $\\hat\\beta_1$', { accent: 'accent' });
      const sN = UI.stat(stats, 'No offset $\\hat\\beta_1$');
      const sF = UI.stat(stats, 'Coefficient on $\\log t$ (free)');
      const sR = UI.stat(stats, 'Repetitions');
      const B0 = -1;
      let cur = null, eO = [], eN = [];
      function gen() {
        const nn = Math.round(n.get()), x = [], y = [], lt = [];
        for (let i = 0; i < nn; i++) { const xv = ctx.rng.normal(), l = 1.5 + gam.get() * xv + 0.5 * ctx.rng.normal(); x.push(xv); lt.push(l); y.push(ctx.rng.poisson(Math.exp(l + B0 + b1s.get() * xv))); }
        return { x, y, lt };
      }
      function fitAll(d) {
        const X = d.x.map(v => [1, v]);
        const o = glmFit(X, d.y, { family: 'poisson', link: 'log', offset: d.lt });
        const no = glmFit(X, d.y, { family: 'poisson', link: 'log' });
        const fr = glmFit(d.x.map((v, i) => [1, v, d.lt[i]]), d.y, { family: 'poisson', link: 'log' });
        return { o, no, fr };
      }
      function one(show) { const d = gen(), f = fitAll(d); eO.push(f.o.beta[1]); eN.push(f.no.beta[1]); if (show) cur = { d, f }; }
      function reset() { loop.stop(); eO = []; eN = []; one(true); draw(); }
      let acc = 0;
      function step(dt) { acc += dt; for (let i = 0; i < 6; i++) one(false); if (eO.length >= 1000) { draw(); return false; } if (acc > 0.08) { acc = 0; draw(); } }
      function draw() {
        const { d, f } = cur, b1 = b1s.get(), g = gam.get();
        const ly = d.y.map((v, i) => Math.log((v + 0.5) / Math.exp(d.lt[i])));
        const [lo, hi] = S.extent(ly);
        pL.setDomain([-3, 3], [lo - 0.3, hi + 0.3]).clear();
        pL.points(d.x.map((v, i) => [v, ly[i]]), { color: 'muted', r: 2.5, opacity: 0.6 });
        const mlt = S.mean(d.lt), grid = S.linspace(-3, 3, 50);
        pL.line(grid.map(v => [v, B0 + b1 * v]), { color: 'ink', width: 2, dash: '6 4' });
        pL.line(grid.map(v => [v, f.o.beta[0] + f.o.beta[1] * v]), { color: 's1', width: 2.2 });
        pL.line(grid.map(v => [v, f.no.beta[0] + f.no.beta[1] * v - mlt]), { color: 's2', width: 2.2 });
        pL.legend([{ label: 'true log rate', color: 'ink', dash: true }, { label: 'offset fit', color: 's1' }, { label: 'no offset', color: 's2' }], { left: true });
        const all = eO.concat(eN), lo2 = Math.min(b1, b1 + g, ...all) - 0.1, hi2 = Math.max(b1, b1 + g, ...all) + 0.1;
        const hO = hist(eO, lo2, hi2, 50), hN = hist(eN, lo2, hi2, 50);
        pR.setDomain([lo2, hi2], [0, Math.max(1, ...hO.map(b => b.density), ...hN.map(b => b.density)) * 1.15]).clear();
        pR.bars(hO, { color: 's1', opacity: 0.5 }); pR.bars(hN, { color: 's2', opacity: 0.5 });
        pR.vline(b1, { color: 'ink', dash: false, width: 2 }); pR.vline(b1 + g, { color: 's2', width: 1.5 });
        pR.legend([{ label: 'offset', color: 's1', fill: true }, { label: 'no offset', color: 's2', fill: true }, { label: 'true β₁', color: 'ink' }]);
        sO.set(`${S.fmt(f.o.beta[1], 3)} ± ${S.fmt(se(f.o.cov, 1), 2)}`, `truth ${S.fmt(b1, 3)}`);
        sN.set(`${S.fmt(f.no.beta[1], 3)} ± ${S.fmt(se(f.no.cov, 1), 2)}`, `limit β₁ + γ = ${S.fmt(b1 + g, 3)}`);
        sF.set(`${S.fmt(f.fr.beta[2], 3)} ± ${S.fmt(se(f.fr.cov, 2), 2)}`, 'test against 1');
        sR.set(String(eO.length), eO.length > 1 ? `mean: ${S.fmt(S.mean(eO), 3)} vs ${S.fmt(S.mean(eN), 3)}` : '');
      }
      reset();
    },
  });

  Atlas.lab('poisson-regression-ppml', {
    title: 'Log-linear means: Poisson pseudo-ML versus OLS on log y',
    blurb: 'Positive continuous data with $\\E(Y\\mid x) = e^{x}$ (true slope 1) and multiplicative lognormal errors whose log-variance grows with $x$ at rate $h$. OLS on $\\log y$ estimates the slope of $\\E\\log Y$, which is $1 - h/2$; the Poisson pseudo-MLE stays centred on 1.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const h = UI.slider(controls, { label: 'Heteroscedasticity $h$', min: 0, max: 1.2, step: 0.05, value: 0.6, onChange: reset });
      const s0 = UI.slider(controls, { label: 'Baseline log-SD $\\sigma_0$', min: 0.2, max: 1.2, step: 0.05, value: 0.5, onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 50, max: 500, step: 25, value: 200, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', () => { one(true); draw(); });
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Repeat', '❚❚ Pause']);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 2], logy: true, y: [0.01, 100], xlabel: 'x', ylabel: 'y (log scale)', title: 'One data set and fitted means' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'slope estimate', ylabel: 'density', title: 'Slope estimates over repetitions' });
      const sP = UI.stat(stats, 'Mean PPML slope', { accent: 'accent' });
      const sO = UI.stat(stats, 'Mean log-OLS slope');
      const sR = UI.stat(stats, 'Repetitions');
      let cur = null, eP = [], eO = [];
      function gen() {
        const nn = Math.round(n.get()), x = [], y = [];
        for (let i = 0; i < nn; i++) { const xv = ctx.rng.uniform(0, 2), sd = Math.sqrt(s0.get() ** 2 + h.get() * xv); x.push(xv); y.push(Math.exp(xv + sd * ctx.rng.normal() - sd * sd / 2)); }
        return { x, y };
      }
      function one(show) {
        const d = gen(), X = d.x.map(v => [1, v]);
        const pp = glmFit(X, d.y, { family: 'poisson', link: 'log' });
        const ols = S.ols(X, d.y.map(Math.log));
        eP.push(pp.beta[1]); eO.push(ols[1]);
        if (show) cur = { d, pp, ols };
      }
      function reset() { loop.stop(); eP = []; eO = []; one(true); draw(); }
      let acc = 0;
      function step(dt) { acc += dt; for (let i = 0; i < 5; i++) one(false); if (eP.length >= 1000) { draw(); return false; } if (acc > 0.08) { acc = 0; draw(); } }
      function draw() {
        const { d, pp, ols } = cur;
        const ys = d.y.slice().sort((a, b) => a - b);
        pL.setDomain([0, 2], [Math.max(1e-3, ys[0] * 0.8), ys[ys.length - 1] * 1.25]).clear();
        pL.points(d.x.map((v, i) => [v, d.y[i]]), { color: 'muted', r: 2.2, opacity: 0.55 });
        const g = S.linspace(0, 2, 60);
        pL.line(g.map(v => [v, Math.exp(v)]), { color: 'ink', width: 2, dash: '6 4' });
        pL.line(g.map(v => [v, Math.exp(pp.beta[0] + pp.beta[1] * v)]), { color: 's1', width: 2.2 });
        pL.line(g.map(v => [v, Math.exp(ols[0] + ols[1] * v)]), { color: 's2', width: 2.2 });
        pL.legend([{ label: 'true mean', color: 'ink', dash: true }, { label: 'Poisson PML', color: 's1' }, { label: 'exp(OLS on log y)', color: 's2' }], { left: true });
        const lim = 1 - h.get() / 2, all = eP.concat(eO);
        const lo = Math.min(lim, 1, ...all) - 0.1, hi = Math.max(1, ...all) + 0.1;
        const hP = hist(eP, lo, hi, 50), hO = hist(eO, lo, hi, 50);
        pR.setDomain([lo, hi], [0, Math.max(1, ...hP.map(b => b.density), ...hO.map(b => b.density)) * 1.15]).clear();
        pR.bars(hP, { color: 's1', opacity: 0.5 }); pR.bars(hO, { color: 's2', opacity: 0.5 });
        pR.vline(1, { color: 'ink', dash: false, width: 2 }); pR.vline(lim, { color: 's2', width: 1.5 });
        pR.legend([{ label: 'Poisson PML', color: 's1', fill: true }, { label: 'log-OLS', color: 's2', fill: true }]);
        sP.set(S.fmt(S.mean(eP), 4), 'truth 1');
        sO.set(S.fmt(S.mean(eO), 4), `theory 1 − h/2 = ${S.fmt(lim, 3)}`);
        sR.set(String(eP.length));
      }
      reset();
    },
  });

  /* ================= irls ================= */
  const IRLS_FAM = {
    logistic: { link: 'logit', fam: 'binomial', truth: [-0.5, 1.5], dom: [[-4, 3], [-2, 5]], start: [2, -1], ylab: 'P(Y = 1)' },
    probit: { link: 'probit', fam: 'binomial', truth: [-0.3, 0.9], dom: [[-2.5, 2], [-1.5, 3.5]], start: [1.5, -1], ylab: 'P(Y = 1)' },
    poisson: { link: 'log', fam: 'poisson', truth: [0.5, 0.7], dom: [[-2, 3], [-1.5, 2.5]], start: [-1.5, 2], ylab: 'count' },
  };
  Atlas.lab('irls-steps', {
    title: 'IRLS step by step on the likelihood surface',
    blurb: 'Click anywhere on the right panel to choose a starting $\\beta$, then Step or Run. Turn step-halving off to see that even with a concave log-likelihood a raw Newton step from a poor start can overshoot and diverge. Fisher scoring (IRLS), Newton with the observed Hessian, and fixed-step gradient ascent take very different paths. Watch the error column: for canonical links the number of correct digits roughly doubles each step; for the probit, scoring and Newton differ.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fam = UI.segmented(controls, { label: 'Model', options: ['logistic', 'probit', 'poisson'], value: 'logistic', onChange: () => { newData(); } });
      const meth = UI.segmented(controls, { label: 'Method', options: [{ value: 'fisher', label: 'Fisher / IRLS' }, { value: 'newton', label: 'Newton' }, { value: 'grad', label: 'Gradient' }], value: 'fisher', onChange: restart });
      const lr = UI.slider(controls, { label: 'Gradient step size (× 1/n)', min: 0.05, max: 5, step: 0.05, value: 1, log: true, onChange: restart, fmt: v => v.toFixed(2) });
      const row = UI.h('div', 'btn-row', controls);
      const halve = UI.toggle(controls, { label: 'Step-halving (accept a step only if $\ell$ increases)', value: true, onChange: restart });
      UI.button(row, 'Step', () => { doStep(); draw(); }, { primary: true });
      let tacc = 0;
      const loop = ctx.loop(dt => { tacc += dt; if (tacc > 0.35) { tacc = 0; const more = doStep(); draw(); if (!more) return false; } });
      ctx.playButton(row, loop, ['▶ Run', '❚❚ Pause']);
      UI.button(row, 'Restart', restart);
      UI.button(row, 'New data', newData);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-3, 3], xlabel: 'x', ylabel: 'mean', title: 'Current fit' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'β₀', ylabel: 'β₁', title: 'Log-likelihood contours and path (click = start)' });
      const sIt = UI.stat(stats, 'Iteration', { accent: 'accent' });
      const sB = UI.stat(stats, 'Current $\\beta$');
      const sE = UI.stat(stats, 'Error $\\|\\beta^{(t)} - \\hat\\beta\\|$');
      const note = UI.note(stats, '');
      let x = [], y = [], X = [], mle = null, path = [], start = null, lmax = 0;
      const cfg = () => IRLS_FAM[fam.get()];
      function ll(b) {
        const c = cfg(), L = LINKS[c.link]; let s = 0;
        for (let i = 0; i < x.length; i++) {
          const e = b[0] + b[1] * x[i];
          if (c.fam === 'poisson') s += y[i] * e - Math.exp(Math.min(e, 700));
          else { const m = S.clamp(L.inv(e), 1e-300, 1 - 1e-16); s += y[i] ? Math.log(m) : Math.log1p(-m); }
        }
        return s;
      }
      function score(b) {
        const c = cfg(), L = LINKS[c.link], F = FAM[c.fam]; let g0 = 0, g1 = 0;
        for (let i = 0; i < x.length; i++) { const e = b[0] + b[1] * x[i], m = F.clamp(L.inv(e)), d = L.mueta(e), r = (y[i] - m) * d / F.V(m); g0 += r; g1 += r * x[i]; }
        return [g0, g1];
      }
      function fisherInfo(b) {
        const c = cfg(), L = LINKS[c.link], F = FAM[c.fam]; let a = 0, bb = 0, cc = 0;
        for (let i = 0; i < x.length; i++) { const e = b[0] + b[1] * x[i], m = F.clamp(L.inv(e)), d = L.mueta(e), w = d * d / F.V(m); a += w; bb += w * x[i]; cc += w * x[i] * x[i]; }
        return [[a, bb], [bb, cc]];
      }
      function observedInfo(b) {
        const hh = 1e-5, gp0 = score([b[0] + hh, b[1]]), gm0 = score([b[0] - hh, b[1]]), gp1 = score([b[0], b[1] + hh]), gm1 = score([b[0], b[1] - hh]);
        const J00 = -(gp0[0] - gm0[0]) / (2 * hh), J11 = -(gp1[1] - gm1[1]) / (2 * hh), J01 = -((gp1[0] - gm1[0]) + (gp0[1] - gm0[1])) / (4 * hh);
        return [[J00, J01], [J01, J11]];
      }
      function newData() {
        const c = cfg(), L = LINKS[c.link];
        x = ctx.rng.array(80, () => ctx.rng.normal());
        y = x.map(v => { const m = L.inv(c.truth[0] + c.truth[1] * v); return c.fam === 'poisson' ? ctx.rng.poisson(m) : (ctx.rng.u() < m ? 1 : 0); });
        X = x.map(v => [1, v]);
        const f = glmFit(X, y, { family: c.fam, link: c.link, maxit: 100, tol: 1e-14 });
        mle = f.beta; lmax = ll(mle);
        start = c.start.slice();
        restart();
      }
      function restart() { loop.stop(); path = [start.slice()]; draw(); }
      function doStep() {
        const b = path[path.length - 1];
        if (path.length > 60) return false;
        const g = score(b);
        let nb;
        if (meth.get() === 'grad') { const s = lr.get() / x.length; nb = [b[0] + s * g[0], b[1] + s * g[1]]; }
        else {
          const M = meth.get() === 'fisher' ? fisherInfo(b) : observedInfo(b);
          let d; try { d = S.solve(M, g); } catch (e) { return false; }
          nb = [b[0] + d[0], b[1] + d[1]];
          if (halve.get()) { const l0 = ll(b); let k = 0; while (!(ll(nb) >= l0) && k++ < 40) nb = [(nb[0] + b[0]) / 2, (nb[1] + b[1]) / 2]; }
        }
        if (!nb.every(v => Number.isFinite(v) && Math.abs(v) < 1e4)) return false;
        path.push(nb);
        const err = Math.hypot(nb[0] - mle[0], nb[1] - mle[1]);
        return err > 1e-12;
      }
      pR.onPointer((b0, b1, e, type) => { if (type === 'down') { start = [b0, b1]; restart(); } });
      function draw() {
        const c = cfg(), L = LINKS[c.link], b = path[path.length - 1];
        const ymax = c.fam === 'poisson' ? Math.max(...y) + 1 : 1.1;
        pL.setDomain([-3, 3], [c.fam === 'poisson' ? 0 : -0.1, ymax]).clear();
        pL.points(x.map((v, i) => [v, y[i]]), { color: 'muted', r: 2.3, opacity: 0.5 });
        const g = S.linspace(-3, 3, 120);
        pL.line(g.map(v => [v, L.inv(mle[0] + mle[1] * v)]), { color: 'ink', width: 1.5, dash: true });
        pL.line(g.map(v => [v, L.inv(b[0] + b[1] * v)]), { color: 's1', width: 2.4 });
        pL.legend([{ label: 'current', color: 's1' }, { label: 'MLE', color: 'ink', dash: true }], { left: true });
        pR.setDomain(c.dom[0], c.dom[1]).clear();
        pR.contour(ll, [0.5, 2, 5, 10, 20, 40, 80, 160, 320, 640].map(d => lmax - d), { color: 'muted', nx: 60, ny: 50, width: 1 });
        pR.line(path, { color: 's1', width: 1.8 });
        path.forEach((q, i) => pR.dot(q[0], q[1], { color: i ? 's1' : 's4', r: i ? 3 : 5 }));
        pR.dot(mle[0], mle[1], { color: 'ink', r: 4 });
        const t = path.length - 1, err = Math.hypot(b[0] - mle[0], b[1] - mle[1]);
        sIt.set(String(t));
        sB.set(`(${S.fmt(b[0], 5)}, ${S.fmt(b[1], 5)})`, `MLE (${S.fmt(mle[0], 5)}, ${S.fmt(mle[1], 5)})`);
        sE.set(err > 0 ? err.toExponential(2) : '0');
        const rows = path.slice(-8).map((q, k) => { const i = path.length - Math.min(8, path.length) + k, e = Math.hypot(q[0] - mle[0], q[1] - mle[1]); return `<tr><td>${i}</td><td>${S.fmt(q[0], 6)}</td><td>${S.fmt(q[1], 6)}</td><td>${S.fmt(-2 * ll(q), 6)}</td><td>${e > 0 ? Math.log10(e).toFixed(2) : '−∞'}</td></tr>`; }).join('');
        note.set(`<table><thead><tr><th>t</th><th>β₀</th><th>β₁</th><th>−2ℓ</th><th>log₁₀ error</th></tr></thead><tbody>${rows}</tbody></table>`);
      }
      newData();
    },
  });

  /* ================= deviance ================= */
  Atlas.lab('deviance-gof', {
    title: 'When is the deviance $\\chi^2_{n-p}$?',
    blurb: 'Simulate a correctly specified logistic model with $m$ trials per observation, fit it, and record the deviance $D$ and Pearson $X^2$. With $m$ large both follow $\\chi^2_{n-p}$ (curve). Shrink $m$ to 1 and the deviance distribution collapses into a narrow spike set by the fitted probabilities: it no longer measures fit. At intermediate $m$, note that $X^2$ is better calibrated than $D$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const m = UI.segmented(controls, { label: 'Trials per observation $m$', options: ['1', '2', '5', '10', '30'], value: '30', onChange: reset });
      const n = UI.slider(controls, { label: 'Observations $n$', min: 10, max: 60, step: 2, value: 30, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Simulate', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, xlabel: 'D', ylabel: 'density', title: 'Deviance' });
      const pX = new Plot(stage, { width: 420, height: 270, xlabel: 'X²', ylabel: 'density', title: 'Pearson statistic' });
      const sN = UI.stat(stats, 'Simulations');
      const sMD = UI.stat(stats, 'Mean $D$ / mean $X^2$', { accent: 'accent' });
      const sRD = UI.stat(stats, 'Rejection rate at 5%: $D$');
      const sRX = UI.stat(stats, 'Rejection rate at 5%: $X^2$');
      let Ds = [], Xs = [];
      function reset() { loop.stop(); Ds = []; Xs = []; for (let i = 0; i < 40; i++) sim(); draw(); }
      function sim() {
        const nn = Math.round(n.get()), mm = +m.get(), x = S.linspace(-2, 2, nn), X = x.map(v => [1, v]);
        const y = x.map(v => ctx.rng.binomial(mm, 1 / (1 + Math.exp(-(-0.5 + v)))) / mm);
        const f = glmFit(X, y, { family: 'binomial', link: 'logit', weights: x.map(() => mm), maxit: 30 });
        Ds.push(f.dev); Xs.push(pearson(y, f.mu, FAM.binomial.V, x.map(() => mm)));
      }
      let acc = 0;
      function step(dt) { acc += dt; for (let i = 0; i < 25; i++) sim(); if (Ds.length >= 3000) { draw(); return false; } if (acc > 0.08) { acc = 0; draw(); } }
      function panel(p, vals, col) {
        const df = Math.round(n.get()) - 2, hi = Math.max(df + 5 * Math.sqrt(2 * df), S.quantile(vals, 0.995));
        const h = hist(vals, 0, hi, 50), g = S.linspace(0.01, hi, 200), cv = g.map(v => S.pdf.chisq(v, df));
        p.setDomain([0, hi], [0, Math.max(...cv, ...h.map(b => b.density)) * 1.12]).clear();
        p.bars(h, { color: col, opacity: 0.55 });
        p.line(g.map((v, i) => [v, cv[i]]), { color: 'ink', width: 2 });
        p.vline(S.q.chisq(0.95, df), { color: 'bad', width: 1.5 });
        p.legend([{ label: `χ²(${df})`, color: 'ink' }, { label: '95% critical value', color: 'bad', dash: true }]);
      }
      function draw() {
        panel(pD, Ds, 's1'); panel(pX, Xs, 's2');
        const df = Math.round(n.get()) - 2, c = S.q.chisq(0.95, df);
        sN.set(String(Ds.length));
        sMD.set(`${S.fmt(S.mean(Ds), 3)} / ${S.fmt(S.mean(Xs), 3)}`, `df = n − 2 = ${df}`);
        sRD.set(pct(Ds.filter(v => v > c).length / Ds.length), 'nominal 5%');
        sRX.set(pct(Xs.filter(v => v > c).length / Xs.length), 'nominal 5%');
      }
      reset();
    },
  });

  Atlas.lab('deviance-residuals', {
    title: 'Reading residual plots for a Poisson regression',
    blurb: 'A Poisson log-linear model in $x$ is fitted to data from the chosen truth. Right: residuals against the linear predictor with a binned mean (line) and $\\pm2$ guides. Compare Pearson, deviance and randomized quantile residuals, and match each pathology to its signature.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const truth = UI.select(controls, { label: 'True data-generating process', options: [{ value: 'ok', label: 'Correct Poisson model' }, { value: 'od', label: 'Overdispersed (negative binomial, k = 2)' }, { value: 'quad', label: 'Missing quadratic term' }, { value: 'zi', label: 'Zero-inflated (30% structural zeros)' }], value: 'od', onChange: gen });
      const rtype = UI.segmented(controls, { label: 'Residual type', options: [{ value: 'P', label: 'Pearson' }, { value: 'D', label: 'Deviance' }, { value: 'Q', label: 'Rand. quantile' }], value: 'D', onChange: draw });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 50, max: 600, step: 25, value: 200, onChange: gen });
      UI.button(controls, 'New data', gen, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-2, 2], xlabel: 'x', ylabel: 'count', title: 'Data and Poisson fit' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'linear predictor η̂', ylabel: 'residual', title: 'Residuals vs fitted' });
      const sD = UI.stat(stats, 'Deviance $D$ (df)');
      const sX = UI.stat(stats, '$\\hat\\phi = X^2/(n-p)$', { accent: 'accent' });
      const sP = UI.stat(stats, '$P(\\chi^2_{n-p} > D)$');
      const sO = UI.stat(stats, 'Share with $|r| > 2$');
      let x = [], y = [], fit = null, u = [];
      function gen() {
        const nn = Math.round(n.get()), t = truth.get();
        x = ctx.rng.array(nn, () => ctx.rng.uniform(-2, 2));
        y = x.map(v => {
          let mu = Math.exp(1 + 0.6 * v);
          if (t === 'quad') mu = Math.exp(0.6 + 0.6 * v + 0.45 * v * v);
          if (t === 'od') mu = ctx.rng.gamma(2, mu / 2);
          if (t === 'zi' && ctx.rng.u() < 0.3) return 0;
          return ctx.rng.poisson(mu);
        });
        u = x.map(() => ctx.rng.u());
        fit = glmFit(x.map(v => [1, v]), y, { family: 'poisson', link: 'log' });
        draw();
      }
      function resid(i) {
        const m = fit.mu[i], yi = y[i];
        if (rtype.get() === 'P') return (yi - m) / Math.sqrt(m);
        if (rtype.get() === 'D') return Math.sign(yi - m) * Math.sqrt(Math.max(0, FAM.poisson.dev(yi, m, 1)));
        const a = yi > 0 ? S.cdf.poisson(yi - 1, m) : 0, b = S.cdf.poisson(yi, m);
        return S.qnorm(S.clamp(a + u[i] * (b - a), 1e-12, 1 - 1e-12));
      }
      function draw() {
        const g = S.linspace(-2, 2, 100);
        pL.setDomain([-2, 2], [0, Math.max(...y) + 1]).clear();
        pL.points(x.map((v, i) => [v, y[i]]), { color: 'muted', r: 2.3, opacity: 0.55 });
        pL.line(g.map(v => [v, Math.exp(fit.beta[0] + fit.beta[1] * v)]), { color: 's1', width: 2.2 });
        const r = x.map((v, i) => resid(i)), eta = fit.eta;
        const [e0, e1] = S.extent(eta), rmax = Math.max(4, ...r.map(Math.abs)) * 1.05;
        pR.setDomain([e0 - 0.1, e1 + 0.1], [-rmax, rmax]).clear();
        pR.hline(0, { color: 'ink', dash: false, width: 1 }); pR.hline(2, { color: 'muted' }); pR.hline(-2, { color: 'muted' });
        pR.points(eta.map((e, i) => [e, r[i]]), { color: 's2', r: 2.3, opacity: 0.6 });
        const nb = 10, bm = [];
        for (let k = 0; k < nb; k++) { const lo = e0 + (e1 - e0) * k / nb, hi = e0 + (e1 - e0) * (k + 1) / nb; const sel = r.filter((v, i) => eta[i] >= lo && eta[i] <= hi); if (sel.length > 2) bm.push([(lo + hi) / 2, S.mean(sel)]); }
        pR.line(bm, { color: 's1', width: 2.4 });
        const df = y.length - 2, X2 = pearson(y, fit.mu, FAM.poisson.V);
        sD.set(`${S.fmt(fit.dev, 4)} (${df})`);
        sX.set(S.fmt(X2 / df, 3), X2 / df > 1.3 ? colorSpan(false, 'overdispersion or misfit') : 'consistent with 1');
        sP.set(S.fmt(1 - S.cdf.chisq(fit.dev, df), 3), 'counts here are moderate: approximate');
        sO.set(pct(r.filter(v => Math.abs(v) > 2).length / r.length), 'about 5% if the model is right');
      }
      gen();
    },
  });

  /* ================= glm-inference ================= */
  Atlas.lab('glm-inference-profile', {
    title: 'Profile likelihood versus the Wald quadratic',
    blurb: 'Logistic regression of $y$ on one covariate. The solid curve is the profile deviance $2\\{\\ell(\\hat\\beta)-\\ell_p(\\beta_1)\\}$; the dashed parabola is the Wald approximation $(\\beta_1-\\hat\\beta_1)^2/\\widehat\\se^2$. Intervals are where each curve is below 3.84. Increase the true slope with small $n$: the profile becomes skewed and the Wald $z$ can shrink while the LR statistic grows.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 200, step: 2, value: 30, onChange: gen });
      const b1s = UI.slider(controls, { label: 'True slope $\\beta_1$', min: 0, max: 4, step: 0.1, value: 2, onChange: gen });
      UI.button(controls, 'New data', gen, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, xlabel: 'β₁', ylabel: 'deviance difference' });
      const sW = UI.stat(stats, 'Wald $z^2$ for $\\beta_1 = 0$');
      const sL = UI.stat(stats, 'LR statistic for $\\beta_1 = 0$', { accent: 'accent' });
      const sS = UI.stat(stats, 'Score statistic for $\\beta_1 = 0$');
      const sCW = UI.stat(stats, 'Wald 95% CI');
      const sCL = UI.stat(stats, 'Profile 95% CI');
      let x = [], y = [];
      function gen() {
        const nn = Math.round(n.get());
        do { x = ctx.rng.array(nn, () => ctx.rng.normal()); y = x.map(v => (ctx.rng.u() < 1 / (1 + Math.exp(-b1s.get() * v)) ? 1 : 0)); } while (separation(x, y) === 2);
        draw();
      }
      function draw() {
        const sep = separation(x, y), fit = logitNewton(x, y, { maxit: sep ? 30 : 80 });
        const lhat = sep ? 0 : logitLL(x, y, fit.b[0], fit.b[1]), sd = se(fit.cov, 1);
        let lo, hi;
        if (sep) { lo = sep > 0 ? -1 : -20; hi = sep > 0 ? 20 : 1; }
        else { lo = Math.min(-0.5, fit.b[1] - 4.5 * sd); hi = fit.b[1] + Math.max(4.5 * sd, 1); }
        const grid = S.linspace(lo, hi, 220);
        let b0 = 0; const prof = grid.map(b => { const r = profileLL(x, y, b, b0); b0 = r[1]; return 2 * (lhat - r[0]); });
        const ymax = 12;
        p.setDomain([lo, hi], [-1.2, ymax]).clear();
        p.hline(3.841, { color: 'muted' }); p.vline(0, { color: 'faint', dash: false, width: 1 });
        p.text(hi, 3.841, 'χ²₁ 95% = 3.84', { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        p.line(grid.map((b, i) => [b, Math.min(prof[i], 50)]), { color: 's1', width: 2.4 });
        if (!sep) p.line(grid.map(b => [b, ((b - fit.b[1]) / sd) ** 2]), { color: 's2', width: 2, dash: true });
        // profile CI from the grid
        const below = grid.filter((b, i) => prof[i] <= 3.841);
        const pl = below.length ? below[0] : NaN, pu = below.length ? below[below.length - 1] : NaN;
        const plo = sep < 0 || (below.length && pl <= grid[0] + 1e-9) ? -Infinity : pl, phi_ = sep > 0 || (below.length && pu >= grid[grid.length - 1] - 1e-9) ? Infinity : pu;
        p.segment(Number.isFinite(plo) ? plo : lo, -0.4, Number.isFinite(phi_) ? phi_ : hi, -0.4, { color: 's1', width: 5 });
        if (!sep) p.segment(fit.b[1] - 1.96 * sd, -0.9, fit.b[1] + 1.96 * sd, -0.9, { color: 's2', width: 5 });
        p.dot(b1s.get(), -0.65, { color: 'ink', r: 4 });
        p.legend([{ label: 'profile deviance', color: 's1' }, { label: 'Wald quadratic', color: 's2', dash: true }, { label: 'bars: 95% intervals, dot: truth', color: 'ink' }]);
        // statistics at beta1 = 0
        const ybar = S.mean(y), w = ybar * (1 - ybar);
        let u1 = 0, i00 = 0, i01 = 0, i11 = 0;
        for (let i = 0; i < x.length; i++) { u1 += x[i] * (y[i] - ybar); i00 += w; i01 += w * x[i]; i11 += w * x[i] * x[i]; }
        const score = u1 * u1 * i00 / (i00 * i11 - i01 * i01);
        const lr0 = 2 * (lhat - profileLL(x, y, 0, 0)[0]);
        const fmtI = (a, b) => `[${Number.isFinite(a) ? S.fmt(a, 3) : '−∞'}, ${Number.isFinite(b) ? S.fmt(b, 3) : '∞'}]`;
        sW.set(sep ? 'unreliable' : S.fmt((fit.b[1] / sd) ** 2, 4), sep ? 'MLE does not exist' : `β̂₁ = ${S.fmt(fit.b[1], 3)}, se ${S.fmt(sd, 3)}`);
        sL.set(S.fmt(lr0, 4), sep ? 'sup over ℓ (separated)' : '');
        sS.set(S.fmt(score, 4), 'needs only the null fit');
        sCW.set(sep ? 'undefined' : fmtI(fit.b[1] - 1.96 * sd, fit.b[1] + 1.96 * sd));
        sCL.set(fmtI(plo, phi_), sep ? colorSpan(false, 'separation: one endpoint infinite') : '');
      }
      gen();
    },
  });

  Atlas.lab('glm-inference-coverage', {
    title: 'Coverage of Wald and profile-likelihood intervals',
    blurb: 'Repeated logistic regressions with intercept $-1$ and the chosen slope. Each experiment builds a nominal 95% Wald interval and a profile-likelihood interval for $\\beta_1$. With few events the Wald coverage drifts far from 95% (usually above it, because its SE inflates with $|\\hat\\beta_1|$), while the profile interval stays within a point or two. Try $n = 15$ with $\\beta_1 = 0$, then $n = 150$. Separated data sets (MLE undefined) are counted separately.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 150, step: 5, value: 25, onChange: reset });
      const b1s = UI.slider(controls, { label: 'True slope $\\beta_1$', min: 0, max: 3, step: 0.1, value: 1.5, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 280, x: [0, 2], y: [0.7, 1], ylabel: 'empirical coverage', xticks: 0, title: 'Coverage of nominal 95% intervals (non-separated data sets)' });
      const sN = UI.stat(stats, 'Experiments');
      const sS = UI.stat(stats, 'Separated data sets');
      const sW = UI.stat(stats, 'Median width: Wald / profile');
      let hits, count, nsep, wW, wP;
      function reset() { loop.stop(); hits = [0, 0]; count = 0; nsep = 0; wW = []; wP = []; draw(); }
      const C = 3.841;
      function root(f, a, b) { let fa = f(a); for (let k = 0; k < 50; k++) { const m = (a + b) / 2, fm = f(m); if ((fm > 0) === (fa > 0)) { a = m; fa = fm; } else b = m; } return (a + b) / 2; }
      function experiment() {
        const nn = Math.round(n.get()), b1 = b1s.get();
        let x, y, sep;
        do { x = ctx.rng.array(nn, () => ctx.rng.normal()); y = x.map(v => (ctx.rng.u() < 1 / (1 + Math.exp(-(-1 + b1 * v))) ? 1 : 0)); sep = separation(x, y); } while (sep === 2);
        if (sep) { nsep++; return; }
        const f = logitNewton(x, y, { maxit: 80 });
        if (!f.ok) { nsep++; return; }
        const sd = se(f.cov, 1), lhat = logitLL(x, y, f.b[0], f.b[1]);
        const g = b => 2 * (lhat - profileLL(x, y, b, f.b[0])[0]) - C;
        let a = f.b[1] - sd, k = 0; while (g(a) < 0 && k++ < 30) a -= 2 * sd;
        let c = f.b[1] + sd; k = 0; while (g(c) < 0 && k++ < 30) c += 2 * sd;
        const lo = root(g, a, f.b[1]), hi = root(g, f.b[1], c);
        if (f.b[1] - 1.96 * sd <= b1 && b1 <= f.b[1] + 1.96 * sd) hits[0]++;
        if (lo <= b1 && b1 <= hi) hits[1]++;
        wW.push(2 * 1.96 * sd); wP.push(hi - lo);
        count++;
      }
      let acc = 0;
      function step(dt) { acc += dt; for (let i = 0; i < 8; i++) experiment(); if (count >= 2000) { draw(); return false; } if (acc > 0.08) { acc = 0; draw(); } }
      function draw() {
        p.clear();
        p.hline(0.95, { color: 'ink', dash: '4 4', width: 1.2 });
        [['Wald', 's2'], ['Profile likelihood', 's1']].forEach(([lab, c], k) => {
          if (count) {
            const cov = hits[k] / count;
            p.bars([{ x0: k + 0.2, x1: k + 0.8, y: Math.max(0.7, cov) }], { color: c, opacity: 0.75, base: 0.7 });
            p.text(k + 0.5, Math.max(0.7, cov), pct(cov), { anchor: 'middle', dy: -6, size: 12, weight: 600 });
          }
          p.text(k + 0.5, 0.7, lab, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        });
        sN.set(String(count + nsep));
        sS.set(count + nsep ? pct(nsep / (count + nsep)) : '—', 'excluded from the bars');
        sW.set(count ? `${S.fmt(S.median(wW), 3)} / ${S.fmt(S.median(wP), 3)}` : '—');
      }
      reset();
    },
  });

  /* ================= overdispersion ================= */
  Atlas.lab('overdispersion-mixture', {
    title: 'A gamma mixture of Poissons is negative binomial',
    blurb: 'Left: the gamma distribution of the unobserved rate $\\lambda$ (shape $k = 1/\\alpha$, mean $\\mu$). Right: the resulting negative binomial pmf against a Poisson with the same mean, and a sample drawn in two stages (draw $\\lambda$, then $Y\\mid\\lambda$). The variance is $\\mu + \\alpha\\mu^2$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mu = UI.slider(controls, { label: 'Mean $\\mu$', min: 0.5, max: 20, step: 0.5, value: 6, onInput: draw });
      const al = UI.slider(controls, { label: 'Overdispersion $\\alpha = 1/k$', min: 0, max: 3, step: 0.05, value: 0.5, onInput: draw });
      const show = UI.toggle(controls, { label: 'Show a two-stage sample (n = 400)', value: true, onChange: draw });
      UI.button(controls, 'New sample', draw, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'λ', ylabel: 'density', title: 'Mixing distribution of the rate' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'y', ylabel: 'probability', title: 'Marginal distribution of Y' });
      const sV = UI.stat(stats, 'Variance $\\mu + \\alpha\\mu^2$', { accent: 'accent' });
      const sZ = UI.stat(stats, '$P(Y = 0)$: NB vs Poisson');
      const sS = UI.stat(stats, 'Sample mean / variance');
      function draw() {
        const m = mu.get(), a = al.get(), k = a > 0 ? 1 / a : Infinity, V = m + a * m * m;
        const ymax = Math.ceil(m + 4 * Math.sqrt(V)) + 1;
        const lmax = Math.max(1, a > 0 ? S.q.gamma(0.99, k, m / k) * 1.05 : m * 2);
        pL.setDomain([0, lmax], [0, 1]).clear();
        if (a > 0) {
          const g = S.linspace(lmax / 400, lmax, 300), dv = g.map(v => S.pdf.gamma(v, k, m / k)), top = Math.min(Math.max(...dv.filter(Number.isFinite)) * 1.1, 5 / m + 2);
          pL.setDomain([0, lmax], [0, top]).clear();
          pL.area(g.map((v, i) => [v, Math.min(dv[i], top * 2)]), { color: 's3', opacity: 0.25 });
          pL.line(g.map((v, i) => [v, Math.min(dv[i], top * 2)]), { color: 's3', width: 2 });
        } else { pL.vline(m, { color: 's3', dash: false, width: 3 }); pL.text(m, 0.9, 'λ = μ (no heterogeneity)', { dx: 6, size: 11, color: 'muted' }); }
        pL.vline(m, { color: 'ink', width: 1.2 });
        const ks = S.range(ymax + 1);
        const nb = ks.map(y => (a > 0 ? S.pmf.negbinom(y, k, k / (k + m)) : S.pmf.poisson(y, m))), po = ks.map(y => S.pmf.poisson(y, m));
        let samp = null;
        if (show.get()) samp = ctx.rng.array(400, () => ctx.rng.poisson(a > 0 ? ctx.rng.gamma(k, m / k) : m));
        const top = Math.max(...nb, ...po) * 1.2;
        pR.setDomain([-0.8, ymax + 0.8], [0, top]).clear();
        if (samp) { const counts = ks.map(y => samp.filter(v => v === y).length / samp.length); pR.bars(ks.map((y, i) => ({ x0: y - 0.45, x1: y + 0.45, y: counts[i] })), { color: 'faint', opacity: 0.6 }); }
        pR.stems(ks.map((y, i) => [y - 0.15, po[i]]), { color: 's2', r: 2.5, width: 1.5 });
        pR.stems(ks.map((y, i) => [y + 0.15, nb[i]]), { color: 's1', r: 2.5, width: 1.5 });
        pR.legend([{ label: 'negative binomial', color: 's1' }, { label: 'Poisson, same mean', color: 's2' }].concat(samp ? [{ label: 'two-stage sample', color: 'faint', fill: true }] : []));
        sV.set(S.fmt(V, 4), `Poisson would be ${S.fmt(m, 3)}; ratio ${S.fmt(V / m, 3)}`);
        sZ.set(`${S.fmt(nb[0], 3)} vs ${S.fmt(po[0], 3)}`);
        sS.set(samp ? `${S.fmt(S.mean(samp), 3)} / ${S.fmt(S.variance(samp), 3)}` : '—');
      }
      draw();
    },
  });

  Atlas.lab('overdispersion-coverage', {
    title: 'Poisson, quasi-Poisson, sandwich and negative binomial intervals',
    blurb: 'Counts are negative binomial with $\\log\\mu = 1 + 0.5x$ and overdispersion $\\alpha$. Each experiment fits a Poisson GLM (naive, quasi-Poisson and sandwich SEs) and a negative binomial GLM (moment estimate of $\\alpha$), and records whether each 95% Wald interval covers 0.5. Right: average reported SE versus the actual SD of the estimates.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const al = UI.slider(controls, { label: 'Overdispersion $\\alpha$', min: 0, max: 2, step: 0.05, value: 0.6, onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 30, max: 400, step: 10, value: 100, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pC = new Plot(stage, { width: 420, height: 270, x: [0, 4], y: [0.4, 1], ylabel: 'coverage', xticks: 0, title: 'Coverage of nominal 95% intervals' });
      const pS = new Plot(stage, { width: 420, height: 270, x: [0, 4], y: [0, 0.2], ylabel: 'SE', xticks: 0, title: 'Mean reported SE (bar) vs true SD (tick)' });
      const sN = UI.stat(stats, 'Experiments');
      const sA = UI.stat(stats, 'Mean $\\hat\\alpha$ / mean $\\hat\\phi$');
      const M = [['Poisson', 's2'], ['Quasi-P', 's3'], ['Sandwich', 's4'], ['Neg. bin.', 's1']];
      let hits, ses, estP, estNB, aHat, phiHat;
      function reset() { loop.stop(); hits = [0, 0, 0, 0]; ses = [[], [], [], []]; estP = []; estNB = []; aHat = []; phiHat = []; draw(); }
      function experiment() {
        const nn = Math.round(n.get()), a = al.get(), x = ctx.rng.array(nn, () => ctx.rng.normal()), X = x.map(v => [1, v]);
        const y = x.map(v => { const m = Math.exp(1 + 0.5 * v); return ctx.rng.poisson(a > 0 ? ctx.rng.gamma(1 / a, m * a) : m); });
        const fp = glmFit(X, y, { family: 'poisson', link: 'log' });
        if (!fp.cov) return;
        const df = nn - 2, X2 = pearson(y, fp.mu, FAM.poisson.V), ph = X2 / df;
        let b00 = 0, b01 = 0, b11 = 0, num = 0, den = 0;
        for (let i = 0; i < nn; i++) { const r2 = (y[i] - fp.mu[i]) ** 2; b00 += r2; b01 += r2 * x[i]; b11 += r2 * x[i] * x[i]; num += r2 - y[i]; den += fp.mu[i] ** 2; }
        const A = fp.cov, Bm = [[b00, b01], [b01, b11]], sand = S.matmul(S.matmul(A, Bm), A);
        const ah = Math.max(1e-4, num / den);
        const fn = glmFit(X, y, { family: negbin(1 / ah), link: 'log', start: fp.beta });
        const b = fp.beta[1], sP = se(fp.cov, 1), sQ = sP * Math.sqrt(ph), sS = Math.sqrt(sand[1][1]), sN_ = se(fn.cov, 1);
        [[b, sP], [b, sQ], [b, sS], [fn.beta[1], sN_]].forEach(([est, s], k) => { if (Math.abs(est - 0.5) <= 1.96 * s) hits[k]++; ses[k].push(s); });
        estP.push(b); estNB.push(fn.beta[1]); aHat.push(ah); phiHat.push(ph);
      }
      let acc = 0;
      function step(dt) { acc += dt; for (let i = 0; i < 12; i++) experiment(); if (estP.length >= 2000) { draw(); return false; } if (acc > 0.08) { acc = 0; draw(); } }
      function draw() {
        const c = estP.length;
        pC.clear(); pC.hline(0.95, { color: 'ink', dash: '4 4', width: 1.2 });
        const sdP = c > 2 ? S.sd(estP) : NaN, sdN = c > 2 ? S.sd(estNB) : NaN;
        const mse = ses.map(a => (a.length ? S.mean(a) : NaN)), truth = [sdP, sdP, sdP, sdN];
        const top = Math.max(0.05, ...mse.filter(Number.isFinite), ...truth.filter(Number.isFinite)) * 1.25;
        pS.setDomain([0, 4], [0, top]).clear();
        M.forEach(([lab, col], k) => {
          if (c) {
            const cov = hits[k] / c;
            pC.bars([{ x0: k + 0.15, x1: k + 0.85, y: Math.max(0.4, cov) }], { color: col, opacity: 0.75, base: 0.4 });
            pC.text(k + 0.5, Math.max(0.4, cov), pct(cov), { anchor: 'middle', dy: -6, size: 11, weight: 600 });
            pS.bars([{ x0: k + 0.2, x1: k + 0.8, y: mse[k] }], { color: col, opacity: 0.6 });
            if (Number.isFinite(truth[k])) pS.segment(k + 0.1, truth[k], k + 0.9, truth[k], { color: 'ink', width: 3 });
          }
          pC.text(k + 0.5, 0.4, lab, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
          pS.text(k + 0.5, 0, lab, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        });
        sN.set(String(c));
        sA.set(c ? `${S.fmt(S.mean(aHat), 3)} / ${S.fmt(S.mean(phiHat), 3)}` : '—', `true α = ${S.fmt(al.get(), 3)}`);
      }
      reset();
    },
  });

  /* ================= multinomial-ordinal ================= */
  const ORD_F = { logit: { F: t => 1 / (1 + Math.exp(-t)), f: t => S.pdf.logistic(t) }, probit: { F: t => Phi(t), f: t => phi(t) } };
  Atlas.lab('multinomial-ordinal-cutpoints', {
    title: 'Ordinal regression as a sliced latent variable',
    blurb: 'Left: the latent $Y^* = \\beta x + \\varepsilon$ at the chosen $x$, cut at thresholds $\\alpha_1 < \\alpha_2 < \\alpha_3$ into four ordered categories. Right: category probabilities as functions of $x$ (stacked). The cumulative odds ratios per unit of $x$ are identical for every cut under the logit link — proportional odds — but not under the probit.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const link = UI.segmented(controls, { label: 'Latent error', options: [{ value: 'logit', label: 'logistic (logit)' }, { value: 'probit', label: 'normal (probit)' }], value: 'logit', onChange: draw });
      const xs = UI.slider(controls, { label: 'Covariate $x$', min: -3, max: 3, step: 0.05, value: 0.5, onInput: draw });
      const bs = UI.slider(controls, { label: 'Slope $\\beta$', min: -2, max: 2, step: 0.05, value: 1, onInput: draw });
      const a1 = UI.slider(controls, { label: 'Threshold $\\alpha_1$', min: -4, max: 4, step: 0.05, value: -1.5, onInput: draw });
      const a2 = UI.slider(controls, { label: 'Threshold $\\alpha_2$', min: -4, max: 4, step: 0.05, value: 0.3, onInput: draw });
      const a3 = UI.slider(controls, { label: 'Threshold $\\alpha_3$', min: -4, max: 4, step: 0.05, value: 2, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-8, 8], y: [0, 0.45], xlabel: 'latent y*', ylabel: 'density', title: 'Latent variable at x' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-3, 3], y: [0, 1], xlabel: 'x', ylabel: 'cumulative probability', title: 'Category probabilities vs x' });
      const tiles = [1, 2, 3, 4].map(j => UI.stat(stats, `$P(Y = ${j}\\mid x)$`));
      const sOR = UI.stat(stats, 'Cumulative odds ratios, $x \\to x+1$', { accent: 'accent' });
      const cols = ['s1', 's2', 's3', 's4'];
      function draw() {
        const L = ORD_F[link.get()], al = [a1.get(), a2.get(), a3.get()].sort((u, v) => u - v), b = bs.get(), x0 = xs.get();
        const ctr = b * x0, cuts = [-Infinity, ...al, Infinity];
        pL.setDomain([-8, 8], [0, link.get() === 'logit' ? 0.3 : 0.45]).clear();
        for (let j = 0; j < 4; j++) {
          const lo = Math.max(-8, cuts[j]), hi = Math.min(8, cuts[j + 1]);
          if (hi > lo) { const g = S.linspace(lo, hi, 80); pL.area(g.map(v => [v, L.f(v - ctr)]), { color: cols[j], opacity: 0.45 }); }
        }
        pL.line(S.linspace(-8, 8, 300).map(v => [v, L.f(v - ctr)]), { color: 'ink', width: 1.6 });
        al.forEach((a, j) => { pL.vline(a, { color: 'ink', width: 1.2 }); pL.text(a, link.get() === 'logit' ? 0.285 : 0.43, `α${j + 1}`, { anchor: 'middle', size: 11, color: 'muted', dy: 0 }); });
        pL.vline(ctr, { color: 'muted', width: 1 });
        const g = S.linspace(-3, 3, 150), cum = j => g.map(v => (j === 0 ? 0 : j === 4 ? 1 : L.F(al[j - 1] - b * v)));
        pR.clear();
        for (let j = 0; j < 4; j++) { const lo = cum(j), hi = cum(j + 1); pR.area(g.map((v, i) => [v, hi[i]]), { color: cols[j], opacity: 0.45, base: g.map((v, i) => [v, lo[i]]) }); }
        pR.vline(x0, { color: 'ink', dash: false, width: 1.5 });
        pR.legend([1, 2, 3, 4].map(j => ({ label: `Y = ${j}`, color: cols[j - 1], fill: true })));
        const P = [0, 1, 2, 3].map(j => (j === 3 ? 1 : L.F(al[j] - ctr)) - (j === 0 ? 0 : L.F(al[j - 1] - ctr)));
        P.forEach((v, j) => tiles[j].set(S.fmt(v, 3)));
        const odds = t => t / (1 - t);
        const ors = al.map(a => odds(L.F(a - b * (x0 + 1))) / odds(L.F(a - b * x0)));
        sOR.set(ors.map(v => S.fmt(v, 3)).join(', '), link.get() === 'logit' ? `all equal e^(−β) = ${S.fmt(Math.exp(-b), 3)}` : 'differ by cut: odds not proportional');
      }
      draw();
    },
  });

  Atlas.lab('multinomial-ordinal-potest', {
    title: 'Testing proportional odds',
    blurb: 'Three ordered categories generated from cumulative logits $\\alpha_j - \\beta_j x$ with slopes $1 \\mp \\delta/2$. Both the proportional-odds model (common slope) and the non-parallel model are fitted; the LR statistic has 1 df. Dots: observed category shares in bins of $x$; solid: proportional-odds fit; dashed: non-parallel fit. Repeat to estimate the rejection rate.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dl = UI.slider(controls, { label: 'Non-parallelism $\\delta$', min: 0, max: 0.9, step: 0.05, value: 0.3, onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 100, max: 2000, step: 50, value: 400, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', () => { one(true); draw(); });
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Repeat tests', '❚❚ Pause']);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-2, 2], y: [0, 1], xlabel: 'x', ylabel: 'P(Y = j | x)', title: 'Fits and binned proportions' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'LR statistic', ylabel: 'density', title: 'LR statistics over repetitions vs χ²₁' });
      const sLR = UI.stat(stats, 'LR statistic (1 df)', { accent: 'accent' });
      const sB = UI.stat(stats, 'Slopes: common / separate');
      const sR = UI.stat(stats, 'Rejection rate at 5%');
      const F = t => 1 / (1 + Math.exp(-t));
      let cur = null, LRs = [];
      function gen() {
        const nn = Math.round(n.get()), d = dl.get(), x = [], y = [];
        for (let i = 0; i < nn; i++) {
          const xv = ctx.rng.uniform(-2, 2), c1 = F(-1 - (1 - d / 2) * xv), c2 = F(1 - (1 + d / 2) * xv), u = ctx.rng.u();
          x.push(xv); y.push(u < c1 ? 0 : u < c2 ? 1 : 2);
        }
        return { x, y };
      }
      function nll(x, y, a1, a2, b1, b2) {
        let s = 0;
        for (let i = 0; i < x.length; i++) {
          const u1 = a1 - b1 * x[i], u2 = a2 - b2 * x[i];
          if (u2 <= u1) return 1e12;
          const p = y[i] === 0 ? F(u1) : y[i] === 1 ? F(u2) - F(u1) : 1 - F(u2);
          s -= Math.log(Math.max(p, 1e-300));
        }
        return s;
      }
      function fit(d) {
        const po = S.nelderMead(t => nll(d.x, d.y, t[0], t[0] + Math.exp(t[1]), t[2], t[2]), [-1, Math.log(2), 1], { step: 0.3, tol: 1e-13, maxIter: 4000 });
        const pa = [po.x[0], po.x[0] + Math.exp(po.x[1]), po.x[2]];
        const np = S.nelderMead(t => nll(d.x, d.y, t[0], t[1], t[2], t[3]), [pa[0], pa[1], pa[2], pa[2]], { step: 0.2, tol: 1e-13, maxIter: 6000 });
        const lr = Math.max(0, 2 * (po.fx - np.fx));
        return { po: pa, np: np.x, lr };
      }
      function one(show) { const d = gen(), f = fit(d); LRs.push(f.lr); if (show) cur = { d, f }; }
      function reset() { loop.stop(); LRs = []; one(true); draw(); }
      let acc = 0;
      function step(dt) { acc += dt; one(false); if (LRs.length >= 500) { draw(); return false; } if (acc > 0.1) { acc = 0; draw(); } }
      function draw() {
        const { d, f } = cur, cols = ['s1', 's2', 's3'], g = S.linspace(-2, 2, 100);
        pL.clear();
        const probs = (a1, a2, b1, b2, v) => { const c1 = F(a1 - b1 * v), c2 = F(a2 - b2 * v); return [c1, c2 - c1, 1 - c2]; };
        for (let j = 0; j < 3; j++) {
          pL.line(g.map(v => [v, probs(f.po[0], f.po[1], f.po[2], f.po[2], v)[j]]), { color: cols[j], width: 2.2 });
          pL.line(g.map(v => [v, probs(f.np[0], f.np[1], f.np[2], f.np[3], v)[j]]), { color: cols[j], width: 1.6, dash: true });
        }
        for (let k = 0; k < 8; k++) {
          const lo = -2 + k * 0.5, sel = d.x.map((v, i) => i).filter(i => d.x[i] >= lo && d.x[i] < lo + 0.5);
          if (sel.length) for (let j = 0; j < 3; j++) pL.dot(lo + 0.25, sel.filter(i => d.y[i] === j).length / sel.length, { color: cols[j], r: 3.5 });
        }
        pL.legend([0, 1, 2].map(j => ({ label: `Y = ${j + 1}`, color: cols[j] })), { left: true });
        const hi = Math.max(10, ...LRs) * 1.02, h = hist(LRs, 0, hi, 40), gg = S.linspace(0.05, hi, 200);
        pR.setDomain([0, hi], [0, 1.2]).clear();
        pR.bars(h, { color: 's4', opacity: 0.55 });
        pR.line(gg.map(v => [v, S.pdf.chisq(v, 1)]), { color: 'ink', width: 2 });
        pR.vline(3.841, { color: 'bad', width: 1.5 });
        sLR.set(S.fmt(f.lr, 4), `p = ${S.fmt(1 - S.cdf.chisq(f.lr, 1), 3)}`);
        sB.set(`${S.fmt(f.po[2], 3)} / (${S.fmt(f.np[2], 3)}, ${S.fmt(f.np[3], 3)})`, `truth (${S.fmt(1 - dl.get() / 2, 3)}, ${S.fmt(1 + dl.get() / 2, 3)})`);
        sR.set(pct(LRs.filter(v => v > 3.841).length / LRs.length), `${LRs.length} tests; 5% if δ = 0`);
      }
      reset();
    },
  });

  /* ================= glmm-gee ================= */
  Atlas.lab('glmm-gee-attenuation', {
    title: 'Conditional curves, marginal curve',
    blurb: 'Faint curves: the conditional mean $g^{-1}(\\beta_1x + b)$ for clusters at evenly spaced quantiles of $b \\sim \\N(0,\\sigma^2)$. Thick: the typical cluster ($b=0$) and the population-averaged curve obtained by integrating over $b$. Right: the same on the link scale, where conditional curves are parallel lines and the marginal one is flatter (logit, probit) or merely shifted (log).',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const link = UI.segmented(controls, { label: 'Link', options: ['logit', 'probit', 'log'], value: 'logit', onChange: draw });
      const sg = UI.slider(controls, { label: 'Random-intercept SD $\\sigma$', min: 0, max: 4, step: 0.05, value: 2, onInput: draw });
      const b1 = UI.slider(controls, { label: 'Conditional slope $\\beta_1$', min: 0.2, max: 3, step: 0.05, value: 1, onInput: draw });
      const approx = UI.toggle(controls, { label: 'Show the attenuation approximation', value: true, onChange: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [0, 1], xlabel: 'x', ylabel: 'mean', title: 'Mean scale' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [-6, 6], xlabel: 'x', ylabel: 'g(mean)', title: 'Link scale' });
      const sC = UI.stat(stats, 'Conditional slope');
      const sM = UI.stat(stats, 'Marginal slope (fitted)', { accent: 'accent' });
      const sA = UI.stat(stats, 'Formula');
      const qs = S.linspace(-5, 5, 81), qw = qs.map(z => phi(z)), qsum = S.sum(qw);
      function draw() {
        const L = LINKS[link.get()], s = sg.get(), b = b1.get(), g = S.linspace(-4, 4, 121);
        const marg = v => { let t = 0; for (let k = 0; k < qs.length; k++) t += qw[k] * L.inv(b * v + s * qs[k]); return t / qsum; };
        const mv = g.map(marg), isLog = link.get() === 'log';
        const ytop = isLog ? 8 : 1;
        pL.setDomain([-4, 4], [0, ytop]).clear();
        pR.setDomain([-4, 4], isLog ? [-5, 7] : [-6, 6]).clear();
        for (let k = 1; k <= 9; k++) {
          const bk = s * S.qnorm((k - 0.5) / 9);
          pL.line(g.map(v => [v, L.inv(b * v + bk)]), { color: 's3', width: 1, opacity: 0.45 });
          pR.line(g.map(v => [v, b * v + bk]), { color: 's3', width: 1, opacity: 0.45 });
        }
        pL.line(g.map(v => [v, L.inv(b * v)]), { color: 's1', width: 2.6 });
        pL.line(g.map((v, i) => [v, mv[i]]), { color: 's2', width: 2.6 });
        pR.line(g.map(v => [v, b * v]), { color: 's1', width: 2.6 });
        const lm = mv.map(m => L.link(S.clamp(m, 1e-12, isLog ? 1e300 : 1 - 1e-12)));
        pR.line(g.map((v, i) => [v, lm[i]]), { color: 's2', width: 2.6 });
        const c2 = (16 * Math.sqrt(3) / (15 * Math.PI)) ** 2;
        const fslope = { logit: b / Math.sqrt(1 + c2 * s * s), probit: b / Math.sqrt(1 + s * s), log: b }[link.get()];
        const fint = { logit: 0, probit: 0, log: s * s / 2 }[link.get()];
        if (approx.get()) { pL.line(g.map(v => [v, L.inv(fint + fslope * v)]), { color: 'ink', width: 1.5, dash: true }); pR.line(g.map(v => [v, fint + fslope * v]), { color: 'ink', width: 1.5, dash: true }); }
        pL.legend([{ label: 'cluster b = 0', color: 's1' }, { label: 'marginal (integrated)', color: 's2' }].concat(approx.get() ? [{ label: 'formula', color: 'ink', dash: true }] : []), { left: true });
        // least-squares slope of g(marginal) on [-2, 2]
        const idx = g.map((v, i) => i).filter(i => Math.abs(g[i]) <= 2), xs = idx.map(i => g[i]), ys = idx.map(i => lm[i]);
        const ms = S.cov(xs, ys) / S.variance(xs);
        sC.set(S.fmt(b, 3), link.get() === 'log' ? `rate ratio ${S.fmt(Math.exp(b), 3)}` : link.get() === 'logit' ? `odds ratio ${S.fmt(Math.exp(b), 3)}` : '');
        sM.set(S.fmt(ms, 4), `ratio ${S.fmt(ms / b, 3)}`);
        sA.set(S.fmt(fslope, 4), { logit: 'β/√(1 + 0.346σ²) (approx.)', probit: 'β/√(1 + σ²) (exact)', log: 'β (intercept + σ²/2)' }[link.get()]);
      }
      draw();
    },
  });

  Atlas.lab('glmm-gee-sandwich', {
    title: 'Clustered counts: naive, independence-GEE and exchangeable-GEE inference',
    blurb: 'Poisson counts in $K$ clusters of size $m$ with a lognormal random intercept (SD $\\sigma_b$), scaled so the marginal mean is $\\exp(0.5 + 0.5x)$ and the marginal slope is exactly 0.5. Compare coverage of the naive GLM interval with cluster-robust sandwich intervals, and the efficiency of exchangeable over independence working correlation. Even with 100 clusters the uncorrected sandwich undercovers slightly.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const K = UI.slider(controls, { label: 'Clusters $K$', min: 10, max: 100, step: 5, value: 30, onChange: reset });
      const m = UI.slider(controls, { label: 'Cluster size $m$', min: 2, max: 10, step: 1, value: 6, onChange: reset });
      const sb = UI.slider(controls, { label: 'Random-intercept SD $\\sigma_b$', min: 0, max: 1.5, step: 0.05, value: 0.7, onChange: reset });
      const cov = UI.segmented(controls, { label: 'Covariate varies', options: [{ value: 'between', label: 'between clusters' }, { value: 'within', label: 'within clusters' }], value: 'between', onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pC = new Plot(stage, { width: 420, height: 270, x: [0, 3], y: [0.4, 1], ylabel: 'coverage', xticks: 0, title: 'Coverage of nominal 95% intervals' });
      const pS = new Plot(stage, { width: 420, height: 270, x: [0, 3], y: [0, 0.3], ylabel: 'SE', xticks: 0, title: 'Mean reported SE (bar) vs true SD (tick)' });
      const sN = UI.stat(stats, 'Experiments');
      const sA = UI.stat(stats, 'Mean $\\hat\\alpha$ (exchangeable)');
      const sE = UI.stat(stats, 'Efficiency: var(indep)/var(exch)', { accent: 'accent' });
      const M = [['Naive GLM', 's2'], ['GEE indep.', 's4'], ['GEE exch.', 's1']];
      let hits, ses, eI, eX, als;
      function reset() { loop.stop(); hits = [0, 0, 0]; ses = [[], [], []]; eI = []; eX = []; als = []; draw(); }
      function gee(cl, beta0, exch) {
        let beta = beta0.slice(), alpha = 0, H = null;
        for (let it = 0; it < 25; it++) {
          if (exch) {
            let ss = 0, N = 0, cross = 0, pairs = 0;
            for (const c of cl) { let sr = 0, sr2 = 0; for (let j = 0; j < c.x.length; j++) { const mu = Math.exp(beta[0] + beta[1] * c.x[j]), r = (c.y[j] - mu) / Math.sqrt(mu); sr += r; sr2 += r * r; } ss += sr2; N += c.x.length; cross += (sr * sr - sr2) / 2; pairs += c.x.length * (c.x.length - 1) / 2; }
            const ph = ss / Math.max(1, N - 2); alpha = S.clamp(cross / (ph * Math.max(1, pairs - 2)), 0, 0.95);
          }
          H = [[0, 0], [0, 0]]; const U = [0, 0];
          for (const c of cl) {
            const mm = c.x.length, cc = alpha / (1 - alpha + mm * alpha), us = [], rs = [];
            let su0 = 0, su1 = 0, sr = 0;
            for (let j = 0; j < mm; j++) { const mu = Math.exp(beta[0] + beta[1] * c.x[j]), sq = Math.sqrt(mu); us.push([sq, sq * c.x[j]]); rs.push((c.y[j] - mu) / sq); su0 += sq; su1 += sq * c.x[j]; sr += rs[j]; }
            for (let j = 0; j < mm; j++) {
              const u = us[j], rinv = (rs[j] - cc * sr) / (1 - alpha);
              H[0][0] += u[0] * u[0] / (1 - alpha); H[0][1] += u[0] * u[1] / (1 - alpha); H[1][1] += u[1] * u[1] / (1 - alpha);
              U[0] += u[0] * rinv; U[1] += u[1] * rinv;
            }
            H[0][0] -= cc * su0 * su0 / (1 - alpha); H[0][1] -= cc * su0 * su1 / (1 - alpha); H[1][1] -= cc * su1 * su1 / (1 - alpha);
          }
          H[1][0] = H[0][1];
          let d; try { d = S.solve(H, U); } catch (e) { return null; }
          beta = [beta[0] + d[0], beta[1] + d[1]];
          if (Math.abs(d[0]) + Math.abs(d[1]) < 1e-9) break;
        }
        // sandwich at final beta (same weights)
        const meat = [[0, 0], [0, 0]];
        for (const c of cl) {
          const mm = c.x.length, cc = alpha / (1 - alpha + mm * alpha); let sr = 0; const us = [], rs = [];
          for (let j = 0; j < mm; j++) { const mu = Math.exp(beta[0] + beta[1] * c.x[j]), sq = Math.sqrt(mu); us.push([sq, sq * c.x[j]]); rs.push((c.y[j] - mu) / sq); sr += rs[j]; }
          let u0 = 0, u1 = 0;
          for (let j = 0; j < mm; j++) { const rinv = (rs[j] - cc * sr) / (1 - alpha); u0 += us[j][0] * rinv; u1 += us[j][1] * rinv; }
          meat[0][0] += u0 * u0; meat[0][1] += u0 * u1; meat[1][1] += u1 * u1;
        }
        meat[1][0] = meat[0][1];
        const Hi = safeInv(H); if (!Hi) return null;
        const V = S.matmul(S.matmul(Hi, meat), Hi);
        return { beta, naive: Hi, sand: V, alpha };
      }
      function experiment() {
        const KK = Math.round(K.get()), mm = Math.round(m.get()), s = sb.get(), cl = [];
        for (let i = 0; i < KK; i++) {
          const b = s * ctx.rng.normal() - s * s / 2, xc = ctx.rng.normal(), c = { x: [], y: [] };
          for (let j = 0; j < mm; j++) { const xv = cov.get() === 'between' ? xc : ctx.rng.normal(); c.x.push(xv); c.y.push(ctx.rng.poisson(Math.exp(0.5 + 0.5 * xv + b))); }
          cl.push(c);
        }
        const ind = gee(cl, [0.5, 0], false); if (!ind) return;
        const ex = gee(cl, ind.beta, true); if (!ex) return;
        const res = [[ind.beta[1], Math.sqrt(ind.naive[1][1])], [ind.beta[1], Math.sqrt(ind.sand[1][1])], [ex.beta[1], Math.sqrt(ex.sand[1][1])]];
        res.forEach(([e, sd], k) => { if (Math.abs(e - 0.5) <= 1.96 * sd) hits[k]++; ses[k].push(sd); });
        eI.push(ind.beta[1]); eX.push(ex.beta[1]); als.push(ex.alpha);
      }
      let acc = 0;
      function step(dt) { acc += dt; for (let i = 0; i < 6; i++) experiment(); if (eI.length >= 1500) { draw(); return false; } if (acc > 0.08) { acc = 0; draw(); } }
      function draw() {
        const c = eI.length;
        pC.clear(); pC.hline(0.95, { color: 'ink', dash: '4 4', width: 1.2 });
        const sdI = c > 2 ? S.sd(eI) : NaN, sdX = c > 2 ? S.sd(eX) : NaN, truth = [sdI, sdI, sdX];
        const mse = ses.map(a => (a.length ? S.mean(a) : NaN));
        pS.setDomain([0, 3], [0, Math.max(0.05, ...mse.filter(Number.isFinite), ...truth.filter(Number.isFinite)) * 1.25]).clear();
        M.forEach(([lab, col], k) => {
          if (c) {
            const cv = hits[k] / c;
            pC.bars([{ x0: k + 0.15, x1: k + 0.85, y: Math.max(0.4, cv) }], { color: col, opacity: 0.75, base: 0.4 });
            pC.text(k + 0.5, Math.max(0.4, cv), pct(cv), { anchor: 'middle', dy: -6, size: 11, weight: 600 });
            pS.bars([{ x0: k + 0.2, x1: k + 0.8, y: mse[k] }], { color: col, opacity: 0.6 });
            if (Number.isFinite(truth[k])) pS.segment(k + 0.1, truth[k], k + 0.9, truth[k], { color: 'ink', width: 3 });
          }
          pC.text(k + 0.5, 0.4, lab, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
          pS.text(k + 0.5, 0, lab, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        });
        sN.set(String(c));
        sA.set(c ? S.fmt(S.mean(als), 3) : '—');
        sE.set(c > 2 ? S.fmt(S.variance(eI) / S.variance(eX), 3) : '—', '> 1 means exchangeable is more efficient');
      }
      reset();
    },
  });

  /* ================= gam ================= */
  function cardinal(u) { if (u < 0 || u >= 4) return 0; if (u < 1) return u * u * u / 6; if (u < 2) return (-3 * u ** 3 + 12 * u * u - 12 * u + 4) / 6; if (u < 3) return (3 * u ** 3 - 24 * u * u + 60 * u - 44) / 6; return (4 - u) ** 3 / 6; }
  function bbasis(x, nseg = 20, a = 0, b = 1) { const h = (b - a) / nseg, q = nseg + 3, out = new Array(q); for (let j = 0; j < q; j++) out[j] = cardinal((x - (a + (j - 3) * h)) / h); return out; }
  function diffPen(q) { // D2'D2
    const P = Array.from({ length: q }, () => new Array(q).fill(0));
    for (let r = 0; r < q - 2; r++) { const d = [1, -2, 1]; for (let a = 0; a < 3; a++) for (let c = 0; c < 3; c++) P[r + a][r + c] += d[a] * d[c]; }
    return P;
  }
  const addScaled = (A, B, s) => A.map((r, i) => r.map((v, j) => v + s * B[i][j]));

  Atlas.lab('gam-backfitting', {
    title: 'Backfitting two P-spline smoothers',
    blurb: 'Data $y = f_1(x_1) + f_2(x_2) + \\varepsilon$ with correlated covariates. Each sweep smooths the partial residuals $y - \\bar y - \\hat f_2$ on $x_1$, then $y - \\bar y - \\hat f_1$ on $x_2$. Raise the correlation $\\rho$ and watch convergence slow (the change ratio approaches 1): concurvity.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el, {});
      const rho = UI.slider(controls, { label: 'Covariate correlation $\\rho$', min: 0, max: 0.98, step: 0.02, value: 0.7, onChange: gen });
      const lam = UI.slider(controls, { label: 'Smoothing $\\lambda$', min: 0.01, max: 1000, value: 3, log: true, onChange: prep, fmt: v => S.fmt(v, 3) });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 100, max: 400, step: 20, value: 200, onChange: gen });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'One sweep', () => { sweep(); draw(); }, { primary: true });
      let tacc = 0;
      const loop = ctx.loop(dt => { tacc += dt; if (tacc > 0.25) { tacc = 0; const more = sweep(); draw(); if (!more) return false; } });
      ctx.playButton(row, loop, ['▶ Run', '❚❚ Pause']);
      UI.button(row, 'Reset fit', () => { restart(); draw(); });
      UI.button(row, 'New data', gen);
      stage.classList.add('two');
      const p1 = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-2.2, 2.2], xlabel: 'x₁', ylabel: 'partial residual', title: 'Component f₁' });
      const p2 = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-2.2, 2.2], xlabel: 'x₂', ylabel: 'partial residual', title: 'Component f₂' });
      const sT = UI.stat(stats, 'Sweeps', { accent: 'accent' });
      const sD = UI.stat(stats, 'Max change in last sweep');
      const sR = UI.stat(stats, 'Change ratio (≈ contraction rate)');
      const sQ = UI.stat(stats, 'Residual SS / n');
      const f1t = x => Math.sin(2 * Math.PI * x), f2t = x => 4 * (x - 0.5) ** 2 - 1 / 3;
      let x1 = [], x2 = [], y = [], B1 = [], B2 = [], M1 = null, M2 = null, f1 = [], f2 = [], g1 = null, g2 = null, sweeps = 0, lastD = NaN, prevD = NaN, ybar = 0;
      const P = diffPen(23);
      function gen() {
        const nn = Math.round(n.get()), r = rho.get();
        x1 = []; x2 = []; y = [];
        for (let i = 0; i < nn; i++) { const z1 = ctx.rng.normal(), z2 = r * z1 + Math.sqrt(1 - r * r) * ctx.rng.normal(); const a = Phi(z1), b = Phi(z2); x1.push(a); x2.push(b); y.push(f1t(a) + f2t(b) + 0.3 * ctx.rng.normal()); }
        ybar = S.mean(y);
        B1 = x1.map(v => bbasis(v)); B2 = x2.map(v => bbasis(v));
        prep();
      }
      function smootherMat(B) { const A = addScaled(crossprod(B), P, lam.get()); return S.solve(A, S.transpose(B)); } // q x n
      function prep() { M1 = smootherMat(B1); M2 = smootherMat(B2); restart(); draw(); }
      function restart() { loop.stop(); f1 = y.map(() => 0); f2 = y.map(() => 0); g1 = null; g2 = null; sweeps = 0; lastD = NaN; prevD = NaN; }
      function smooth(M, B, r) { const g = M.map(row => row.reduce((s, v, i) => s + v * r[i], 0)); let f = B.map(b => b.reduce((s, v, j) => s + v * g[j], 0)); const mf = S.mean(f); f = f.map(v => v - mf); return { g, f, mf }; }
      function sweep() {
        const o1 = f1, o2 = f2;
        const a = smooth(M1, B1, y.map((v, i) => v - ybar - f2[i])); f1 = a.f; g1 = a;
        const b = smooth(M2, B2, y.map((v, i) => v - ybar - f1[i])); f2 = b.f; g2 = b;
        sweeps++;
        let d = 0; for (let i = 0; i < y.length; i++) d = Math.max(d, Math.abs(f1[i] - o1[i]), Math.abs(f2[i] - o2[i]));
        prevD = lastD; lastD = d;
        return d > 1e-7 && sweeps < 400;
      }
      function curve(gs) { const xs = S.linspace(0, 1, 120); return xs.map(v => { const b = bbasis(Math.min(v, 1 - 1e-9)); return [v, b.reduce((s, w, j) => s + w * gs.g[j], 0) - gs.mf]; }); }
      function draw() {
        const xs = S.linspace(0, 1, 120);
        p1.clear(); p2.clear();
        p1.points(x1.map((v, i) => [v, y[i] - ybar - f2[i]]), { color: 'muted', r: 2, opacity: 0.45 });
        p2.points(x2.map((v, i) => [v, y[i] - ybar - f1[i]]), { color: 'muted', r: 2, opacity: 0.45 });
        p1.line(xs.map(v => [v, f1t(v)]), { color: 'ink', width: 1.6, dash: true });
        p2.line(xs.map(v => [v, f2t(v)]), { color: 'ink', width: 1.6, dash: true });
        if (g1) p1.line(curve(g1), { color: 's1', width: 2.6 });
        if (g2) p2.line(curve(g2), { color: 's2', width: 2.6 });
        p1.legend([{ label: 'f̂₁', color: 's1' }, { label: 'true f₁', color: 'ink', dash: true }]);
        p2.legend([{ label: 'f̂₂', color: 's2' }, { label: 'true f₂', color: 'ink', dash: true }]);
        let rss = 0; for (let i = 0; i < y.length; i++) rss += (y[i] - ybar - f1[i] - f2[i]) ** 2;
        sT.set(String(sweeps), sweeps && lastD <= 1e-7 ? colorSpan(true, 'converged') : '');
        sD.set(Number.isFinite(lastD) ? lastD.toExponential(2) : '—');
        sR.set(Number.isFinite(prevD) && prevD > 0 ? S.fmt(lastD / prevD, 3) : '—');
        sQ.set(S.fmt(rss / y.length, 4), 'noise variance 0.09');
      }
      gen();
    },
  });

  Atlas.lab('gam-smoothing', {
    title: 'Choosing the smoothing parameter',
    blurb: 'A cubic P-spline (20 segments, second-difference penalty) fitted by penalized IRLS. Drag $\\lambda$: small values chase noise (high edf), large values flatten toward a straight line (edf → 2). Right: GCV (Gaussian) or UBRE (binary) across $\\lambda$, with the current choice marked.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const resp = UI.segmented(controls, { label: 'Response', options: [{ value: 'gauss', label: 'Gaussian' }, { value: 'bin', label: 'Binary (logit)' }], value: 'gauss', onChange: gen });
      const lam = UI.slider(controls, { label: 'Smoothing $\\lambda$', min: 0.001, max: 10000, value: 1, log: true, onInput: draw, fmt: v => S.fmt(v, 3) });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 50, max: 500, step: 25, value: 200, onChange: gen });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', gen, { primary: true });
      UI.button(row, 'Go to optimum', () => { if (best) { lam.set(best); draw(); } });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', ylabel: 'y', title: 'Fit' });
      const pR = new Plot(stage, { width: 420, height: 270, logx: true, x: [0.001, 10000], xlabel: 'λ (log scale)', ylabel: 'criterion', title: 'Smoothness selection' });
      const sE = UI.stat(stats, 'Effective df $\\tr A_\\lambda$', { accent: 'accent' });
      const sC = UI.stat(stats, 'Criterion');
      const sB = UI.stat(stats, 'Optimal $\\lambda$ (grid)');
      const P = diffPen(23), ftrue = x => 1.2 * Math.sin(3 * Math.PI * x * x) * (1 - 0.5 * x);
      let x = [], y = [], B = [], crit = [], best = null;
      const lgrid = S.linspace(-3, 4, 36).map(e => 10 ** e);
      function fitAt(l) {
        const bin = resp.get() === 'bin';
        const f = glmFit(B, y, { family: bin ? 'binomial' : 'gaussian', link: bin ? 'logit' : 'identity', penalty: P.map(r => r.map(v => v * l)), maxit: bin ? 40 : 3, tol: 1e-9 });
        const A = f.cov ? S.matmul(f.cov, f.info) : null;
        const edf = A ? A.reduce((s, r, i) => s + r[i], 0) : NaN, nn = y.length;
        const c = bin ? f.dev / nn - 1 + 2 * edf / nn : nn * f.dev / (nn - edf) ** 2;
        return { f, edf, c };
      }
      function gen() {
        const nn = Math.round(n.get()), bin = resp.get() === 'bin';
        x = ctx.rng.array(nn, () => ctx.rng.uniform(0, 1)).sort((a, b) => a - b);
        y = x.map(v => (bin ? (ctx.rng.u() < 1 / (1 + Math.exp(-2 * ftrue(v))) ? 1 : 0) : ftrue(v) + 0.4 * ctx.rng.normal()));
        B = x.map(v => bbasis(Math.min(v, 1 - 1e-9)));
        crit = lgrid.map(l => fitAt(l).c);
        best = lgrid[S.argmin(crit.map(v => (Number.isFinite(v) ? v : Infinity)))];
        draw();
      }
      function draw() {
        const bin = resp.get() === 'bin', l = lam.get(), r = fitAt(l), xs = S.linspace(0, 1, 160);
        const fx = xs.map(v => bbasis(Math.min(v, 1 - 1e-9)).reduce((s, w, j) => s + w * r.f.beta[j], 0));
        if (bin) {
          pL.setDomain([0, 1], [-0.1, 1.1]).clear();
          pL.points(x.map((v, i) => [v, y[i] ? 1.04 : -0.04]), { color: 'muted', r: 1.8, opacity: 0.5 });
          pL.line(xs.map(v => [v, 1 / (1 + Math.exp(-2 * ftrue(v)))]), { color: 'ink', width: 1.6, dash: true });
          pL.line(xs.map((v, i) => [v, 1 / (1 + Math.exp(-fx[i]))]), { color: 's1', width: 2.5 });
        } else {
          pL.setDomain([0, 1], [-2.5, 2.5]).clear();
          pL.points(x.map((v, i) => [v, y[i]]), { color: 'muted', r: 2, opacity: 0.55 });
          pL.line(xs.map(v => [v, ftrue(v)]), { color: 'ink', width: 1.6, dash: true });
          pL.line(xs.map((v, i) => [v, fx[i]]), { color: 's1', width: 2.5 });
        }
        pL.legend([{ label: 'fit', color: 's1' }, { label: 'truth', color: 'ink', dash: true }], { left: true });
        const fin = crit.filter(Number.isFinite), lo = Math.min(...fin), hi = Math.min(Math.max(...fin), lo + 4 * (S.median(fin) - lo + 1e-6));
        pR.setDomain([0.001, 10000], [lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05]).clear();
        pR.line(lgrid.map((v, i) => [v, crit[i]]), { color: 's2', width: 2.2 });
        pR.vline(l, { color: 's1', dash: false, width: 2 });
        if (best) pR.vline(best, { color: 'muted' });
        sE.set(S.fmt(r.edf, 3), 'between 2 and 23');
        sC.set(S.fmt(r.c, 4), bin ? 'UBRE = D/n − 1 + 2·edf/n' : 'GCV = n·RSS/(n − edf)²');
        sB.set(best ? S.fmt(best, 3) : '—');
      }
      gen();
    },
  });
})();
