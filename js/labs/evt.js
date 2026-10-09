/* Labs: Extreme Value Theory. Content in content/evt.js. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const EULER = 0.5772156649015329;
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const fmtN = v => { const e = Math.round(Math.log10(v)); return Math.abs(v - 10 ** e) < 1e-6 * v ? (e <= 3 ? String(10 ** e) : '10^' + e) : String(Math.round(v)); };
  const COLS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'];

  /* ---------------- precise tails and quantiles ---------------- */

  // Standard normal survival, accurate far into the tail (Laplace continued fraction for x >= 3).
  function nsf(x) {
    if (x < 0) return 1 - nsf(-x);
    if (x < 3) return 0.5 * (1 - S.erf(x / Math.SQRT2));
    let t = x;
    for (let k = 60; k >= 1; k--) t = x + k / t;
    return S.pdf.normal(x) / t;
  }
  // Student t survival and upper quantile (x with P(T > x) = s).
  const tSurv = (x, nu) => (x >= 0 ? 0.5 * S.ibeta(nu / (nu + x * x), nu / 2, 0.5) : 1 - 0.5 * S.ibeta(nu / (nu + x * x), nu / 2, 0.5));
  function tQc(s, nu) {
    if (s >= 0.5) return s === 0.5 ? 0 : -tQc(1 - s, nu);
    let lo = 0, hi = 1;
    while (tSurv(hi, nu) > s && hi < 1e300) { lo = hi; hi *= 2; }
    for (let i = 0; i < 80; i++) {
      const mid = 0.5 * (lo + hi);
      if (tSurv(mid, nu) > s) lo = mid; else hi = mid;
      if (hi - lo < 1e-12 * hi) break;
    }
    return 0.5 * (lo + hi);
  }
  // Beta(2,2): survival (1-x)^2 (1+2x); solve in d = 1 - x for precision near the endpoint.
  function beta22Qc(s) {
    let lo = 0, hi = 1;
    for (let i = 0; i < 80; i++) { const d = 0.5 * (lo + hi); if (d * d * (3 - 2 * d) < s) lo = d; else hi = d; }
    return 1 - 0.5 * (lo + hi);
  }
  // "Hill horror" parent: survival e/(x log x) for x >= e (tail index 1, very slowly varying L).
  function horrorQc(s) {
    let y = Math.max(1, 1 - Math.log(s));
    for (let i = 0; i < 50; i++) { const g = y + Math.log(y) - 1 + Math.log(s); y -= g / (1 + 1 / y); if (Math.abs(g) < 1e-13) break; }
    return Math.exp(y);
  }

  const mkT = nu => ({ label: `Student t (ν = ${nu})`, xi: 1 / nu, Qc: s => tQc(s, nu), sf: x => tSurv(x, nu), pdf: x => S.pdf.t(x, nu), sample: r => r.t(nu) });
  // Parent distributions: upper quantile Qc(s) = F^{-1}(1 - s), survival sf, density pdf, limiting shape xi.
  const PAR = {
    normal: { label: 'Normal', xi: 0, Qc: s => -S.qnorm(s), sf: nsf, pdf: x => S.pdf.normal(x), sample: r => r.normal() },
    exp: { label: 'Exponential', xi: 0, Qc: s => -Math.log(s), sf: x => (x < 0 ? 1 : Math.exp(-x)), pdf: x => (x < 0 ? 0 : Math.exp(-x)), sample: r => r.exp(1) },
    lognormal: { label: 'Lognormal', xi: 0, Qc: s => Math.exp(-S.qnorm(s)), sf: x => (x <= 0 ? 1 : nsf(Math.log(x))), pdf: x => S.pdf.lognormal(x), sample: r => r.lognormal(0, 1) },
    t3: mkT(3), t4: mkT(4), t6: mkT(6), t2: mkT(2), t10: mkT(10),
    pareto: { label: 'Pareto (α = 2)', xi: 0.5, Qc: s => s ** -0.5, sf: x => (x < 1 ? 1 : 1 / (x * x)), pdf: x => (x < 1 ? 0 : 2 / (x * x * x)), sample: r => r.pareto(2, 1) },
    cauchy: { label: 'Cauchy', xi: 1, Qc: s => 1 / Math.tan(Math.PI * s), sf: x => (x > 0 ? Math.atan(1 / x) / Math.PI : 0.5 - Math.atan(x) / Math.PI), pdf: x => S.pdf.cauchy(x), sample: r => r.cauchy() },
    uniform: { label: 'Uniform(0,1)', xi: -1, Qc: s => 1 - s, sf: x => (x <= 0 ? 1 : x >= 1 ? 0 : 1 - x), pdf: x => (x >= 0 && x <= 1 ? 1 : 0), sample: r => r.u() },
    beta22: { label: 'Beta(2,2)', xi: -0.5, Qc: beta22Qc, sf: x => (x <= 0 ? 1 : x >= 1 ? 0 : (1 - x) * (1 - x) * (1 + 2 * x)), pdf: x => (x > 0 && x < 1 ? 6 * x * (1 - x) : 0), sample: r => r.beta(2, 2) },
    gpd: { label: 'GPD (ξ = 0.2)', xi: 0.2, Qc: s => (s ** -0.2 - 1) / 0.2, sf: x => (x < 0 ? 1 : (1 + 0.2 * x) ** -5), pdf: x => (x < 0 ? 0 : (1 + 0.2 * x) ** -6), sample: r => r.gpd(1, 0.2) },
    horror: { label: 'Slow: e/(x log x)', xi: 1, Qc: horrorQc, sf: x => (x < Math.E ? 1 : Math.E / (x * Math.log(x))), pdf: x => (x < Math.E ? 0 : Math.E * (Math.log(x) + 1) / (x * Math.log(x)) ** 2), sample: r => horrorQc(1 - r.u()) },
  };
  const opts = keys => keys.map(k => ({ value: k, label: PAR[k].label }));
  const u01 = r => { let u; do { u = r.u(); } while (u <= 0); return u; };
  // Exact draw of the maximum of n IID copies: F^{-1}(U^{1/n}), via the upper-tail probability 1 - U^{1/n}.
  const drawMax = (P, n, r) => P.Qc(-Math.expm1(Math.log(u01(r)) / n));
  // von Mises constants: b_n = F^{-1}(1 - 1/n), a_n = 1/(n f(b_n)); and the local shape r'(b_n), r = sf/pdf.
  function vonMises(P, n) {
    const b = P.Qc(1 / n), a = 1 / (n * P.pdf(b));
    const r = x => P.sf(x) / P.pdf(x), h = 1e-3 * a;
    return { a, b, xin: (r(b + h) - r(b - h)) / (2 * h) };
  }

  /* ---------------- GEV / GPD inference helpers ---------------- */

  const gfun = (xi, L) => (Math.abs(xi) < 1e-8 ? L : -Math.expm1(-xi * L) / xi);
  // T-year return level (upper-tail probability 1/T).
  function retLevel(mu, sg, xi, T) { const L = Math.log(-Math.log1p(-1 / T)); return mu - sg * gfun(xi, L); }

  function gevNll(z, mu, sg, xi) {
    if (!(sg > 0)) return 1e10;
    const m = z.length;
    let s = m * Math.log(sg);
    if (Math.abs(xi) < 1e-8) { for (let i = 0; i < m; i++) { const y = (z[i] - mu) / sg; s += y + Math.exp(-y); } return s; }
    for (let i = 0; i < m; i++) {
      const t = 1 + xi * (z[i] - mu) / sg;
      if (!(t > 0)) return 1e10;
      const lt = Math.log(t);
      s += (1 + 1 / xi) * lt + Math.exp(-lt / xi);
    }
    return Number.isFinite(s) ? s : 1e10;
  }
  // Hosking, Wallis & Wood (1985) L-moment estimator.
  function gevLmom(z) {
    const s = z.slice().sort((a, b) => a - b), n = s.length;
    let b0 = 0, b1 = 0, b2 = 0;
    for (let i = 0; i < n; i++) { b0 += s[i]; b1 += s[i] * i / (n - 1); b2 += s[i] * i * (i - 1) / ((n - 1) * (n - 2)); }
    b0 /= n; b1 /= n; b2 /= n;
    const l1 = b0, l2 = Math.max(2 * b1 - b0, 1e-12), l3 = 6 * b2 - 6 * b1 + b0, t3 = l3 / l2;
    const c = 2 / (3 + t3) - Math.LN2 / Math.log(3);
    const k = clamp(7.8590 * c + 2.9554 * c * c, -0.95, 0.95);
    if (Math.abs(k) < 1e-6) { const sg = l2 / Math.LN2; return { mu: l1 - EULER * sg, sigma: sg, xi: 0 }; }
    const g = S.gamma(1 + k), sg = l2 * k / ((1 - 2 ** -k) * g);
    return { mu: l1 - sg * (1 - g) / k, sigma: sg, xi: -k };
  }
  // Covariance from a numerical Hessian of a negative log-likelihood f at th (null if not usable).
  function hessCov(f, th, h) {
    const k = th.length, f0 = f(th);
    if (!(f0 < 1e9)) return null;
    const ev = (d) => { const q = th.map((v, i) => v + d[i]); return f(q); };
    const H = [];
    for (let i = 0; i < k; i++) {
      H.push(new Array(k).fill(0));
      for (let j = 0; j <= i; j++) {
        let v;
        if (i === j) {
          const d1 = th.map(() => 0); d1[i] = h[i];
          const d2 = th.map(() => 0); d2[i] = -h[i];
          const a = ev(d1), b = ev(d2);
          if (!(a < 1e9 && b < 1e9)) return null;
          v = (a - 2 * f0 + b) / (h[i] * h[i]);
        } else {
          const vals = [[1, 1], [1, -1], [-1, 1], [-1, -1]].map(([si, sj]) => { const d = th.map(() => 0); d[i] = si * h[i]; d[j] = sj * h[j]; return ev(d); });
          if (!vals.every(x => x < 1e9)) return null;
          v = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h[i] * h[j]);
        }
        H[i][j] = v; if (i !== j) H[j][i] = v;
      }
    }
    for (let i = 0; i < k; i++) H[i] = H[i].slice();
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) H[i][j] = H[j][i];
    try {
      const V = S.inv(H);
      for (let i = 0; i < k; i++) if (!(V[i][i] > 0) || !Number.isFinite(V[i][i])) return null;
      return V;
    } catch (e) { return null; }
  }
  // GEV maximum likelihood (start from L-moments). gumbel: fix xi = 0.
  function gevFit(z, { gumbel = false } = {}) {
    const lm = gevLmom(z);
    if (gumbel) {
      const f = p => gevNll(z, p[0], Math.exp(p[1]), 0);
      let r = S.nelderMead(f, [lm.mu, Math.log(lm.sigma)], { step: 0.3, maxIter: 1000, tol: 1e-12 });
      r = S.nelderMead(f, r.x, { step: 0.05, maxIter: 1000, tol: 1e-12 });
      return { mu: r.x[0], sigma: Math.exp(r.x[1]), xi: 0, nll: r.fx, cov: null };
    }
    const f = p => gevNll(z, p[0], Math.exp(p[1]), p[2]);
    let x0 = [lm.mu, Math.log(lm.sigma), lm.xi];
    if (f(x0) >= 1e10) x0 = [lm.mu, Math.log(lm.sigma), 0];
    let r = S.nelderMead(f, x0, { step: 0.3, maxIter: 1500, tol: 1e-12 });
    r = S.nelderMead(f, r.x, { step: 0.05, maxIter: 1500, tol: 1e-12 });
    const th = [r.x[0], Math.exp(r.x[1]), r.x[2]];
    const cov = hessCov(q => gevNll(z, q[0], q[1], q[2]), th, [1e-3 * th[1], 1e-3 * th[1], 1e-3]);
    return { mu: th[0], sigma: th[1], xi: th[2], nll: r.fx, cov };
  }
  // Delta-method variance of the T-year return level.
  function retLevelVar(fit, T) {
    if (!fit.cov) return NaN;
    const { mu, sigma, xi } = fit, L = Math.log(-Math.log1p(-1 / T)), h = 1e-5;
    const g = [1, -gfun(xi, L), (retLevel(mu, sigma, xi + h, T) - retLevel(mu, sigma, xi - h, T)) / (2 * h)];
    let v = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) v += g[i] * fit.cov[i][j] * g[j];
    return v;
  }
  // Profile-likelihood 95% interval for the T-year return level.
  function profileCI(z, fit, T) {
    const L = Math.log(-Math.log1p(-1 / T)), zhat = retLevel(fit.mu, fit.sigma, fit.xi, T);
    const prof = (zp, start) => {
      const f = p => { const sg = Math.exp(p[0]); return gevNll(z, zp + sg * gfun(p[1], L), sg, p[1]); };
      let r = S.nelderMead(f, start, { step: 0.1, maxIter: 600, tol: 1e-11 });
      r = S.nelderMead(f, r.x, { step: 0.02, maxIter: 600, tol: 1e-11 });
      return r;
    };
    const crit = 3.841 / 2, se = Math.sqrt(retLevelVar(fit, T));
    const step0 = Number.isFinite(se) && se > 0 ? se / 4 : 0.05 * (Math.abs(zhat) + fit.sigma);
    const side = dir => {
      let start = [Math.log(fit.sigma), fit.xi], prevZ = zhat, prevD = 0, step = step0;
      for (let i = 0; i < 80; i++) {
        const zp = prevZ + dir * step;
        const r = prof(zp, start);
        const d = r.fx - fit.nll;
        if (d >= crit) return prevZ + (zp - prevZ) * (crit - prevD) / Math.max(1e-12, d - prevD);
        start = r.x; prevZ = zp; prevD = Math.max(0, d);
        if (dir > 0) step *= 1.12;
      }
      return dir > 0 ? Infinity : -Infinity;
    };
    return [side(-1), side(1)];
  }

  function gpdNll(y, sg, xi) {
    if (!(sg > 0)) return 1e10;
    const k = y.length;
    let s = k * Math.log(sg);
    if (Math.abs(xi) < 1e-8) { for (let i = 0; i < k; i++) s += y[i] / sg; return s; }
    for (let i = 0; i < k; i++) { const t = xi * y[i] / sg; if (!(t > -1)) return 1e10; s += (1 + 1 / xi) * Math.log1p(t); }
    return Number.isFinite(s) ? s : 1e10;
  }
  // GPD MLE with expected-information covariance (Var xi = (1+xi)^2/k, Var sigma = 2 sigma^2 (1+xi)/k, Cov = -sigma(1+xi)/k).
  function gpdFit(y, start) {
    const k = y.length, m = S.mean(y), v = k > 1 ? S.variance(y) : m * m;
    let xi0 = clamp(0.5 * (1 - m * m / v), -0.4, 0.8), s0 = Math.max(1e-8, 0.5 * m * (m * m / v + 1));
    if (start) { s0 = start[0]; xi0 = start[1]; }
    if (gpdNll(y, s0, xi0) >= 1e10) { xi0 = 0; s0 = Math.max(m, 1e-8); }
    const f = p => gpdNll(y, Math.exp(p[0]), p[1]);
    let r = S.nelderMead(f, [Math.log(s0), xi0], { step: 0.2, maxIter: 800, tol: 1e-11 });
    r = S.nelderMead(f, r.x, { step: 0.05, maxIter: 800, tol: 1e-11 });
    const sg = Math.exp(r.x[0]), xi = r.x[1], ok = xi > -0.5;
    return { sigma: sg, xi, nll: r.fx, k, vs: ok ? 2 * sg * sg * (1 + xi) / k : NaN, vx: ok ? (1 + xi) * (1 + xi) / k : NaN, cv: ok ? -sg * (1 + xi) / k : NaN };
  }
  const gpdSf = (y, sg, xi) => (y <= 0 ? 1 : Math.abs(xi) < 1e-8 ? Math.exp(-y / sg) : (1 + xi * y / sg > 0 ? (1 + xi * y / sg) ** (-1 / xi) : 0));
  const gpdQc = (s, sg, xi) => (Math.abs(xi) < 1e-8 ? -sg * Math.log(s) : sg * (s ** -xi - 1) / xi);

  // Histogram helper returning density bins over [lo, hi] (values outside are counted in `off`).
  function hist(data, lo, hi, bins) {
    const h = S.histogram(data.filter(v => v >= lo && v <= hi), { bins, lo, hi });
    const w = (hi - lo) / bins;
    for (const b of h) b.density = b.count / (data.length * w);
    h.off = data.filter(v => v < lo || v > hi).length;
    return h;
  }

  /* =========================== evt-intro =========================== */

  Atlas.lab('evt-intro-degenerate', {
    title: 'The law of the maximum: degenerate, then rescued',
    blurb: 'Left: the exact CDF $F^n$ of $M_n$ for $n = 1, 10, 100, \\dots$, which slides toward the right endpoint and never settles. Right: the same curves after the affine normalisation $x\\mapsto a_nx+b_n$ (von Mises constants), collapsing onto the extreme value limit $G_\\xi$ (dashed). Compare how fast the exponential and the normal collapse.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const keys = ['exp', 'normal', 'lognormal', 'pareto', 't3', 'cauchy', 'uniform', 'beta22'];
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(keys), value: 'exp', onChange: draw });
      const nmax = UI.slider(controls, { label: 'Largest $n$ shown', min: 10, max: 1e6, log: true, logStep: 1, value: 1e4, fmt: fmtN, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'P(Mₙ ≤ x)', title: 'Raw: Fⁿ(x)' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-3, 6], y: [0, 1], xlabel: '(x − bₙ)/aₙ', ylabel: 'P', title: 'Normalised: Fⁿ(aₙx + bₙ) and the limit' });
      const sB = UI.stat(stats, '$b_n$ at largest $n$');
      const sA = UI.stat(stats, '$a_n$ at largest $n$');
      const sK = UI.stat(stats, 'Sup distance to $G_\\xi$', { accent: 'accent' });
      const sX = UI.stat(stats, 'Limit shape $\\xi$');
      const Fn = (P, x, n) => { const s = P.sf(x); if (s >= 1) return 0; if (s <= 0) return 1; return Math.exp(n * Math.log1p(-s)); };
      function draw() {
        const key = par.get(), P = PAR[key], N = Math.round(nmax.get());
        const ns = []; for (let n = 1; n <= N * 1.0001; n *= 10) ns.push(Math.round(n));
        const logx = ['pareto', 't3', 'cauchy', 'lognormal'].includes(key);
        const lo = logx ? P.Qc(key === 'lognormal' ? 0.98 : 0.35) : P.Qc(0.98), hi = P.Qc(0.01 / N);
        pL.o.logx = logx;
        pL.setDomain([lo, hi], [0, 1]).clear();
        const grid = logx ? S.linspace(Math.log10(lo), Math.log10(hi), 400).map(v => 10 ** v) : S.linspace(lo, hi, 400);
        ns.forEach((n, i) => pL.line(grid.map(x => [x, Fn(P, x, n)]), { color: COLS[i], width: 2 }));
        pL.legend(ns.map((n, i) => ({ label: 'n = ' + fmtN(n), color: COLS[i] })), { left: true });
        // normalised
        const xs = S.linspace(-3, 6, 300);
        pR.clear();
        ns.forEach((n, i) => {
          if (n < 10) return;
          const { a, b } = vonMises(P, n);
          pR.line(xs.map(x => [x, Fn(P, a * x + b, n)]), { color: COLS[i], width: 2 });
        });
        pR.line(xs.map(x => [x, S.cdf.gev(x, 0, 1, P.xi)]), { color: 'ink', width: 2, dash: '6 4' });
        pR.legend([{ label: 'limit G_ξ', color: 'ink', dash: true }], { left: true });
        const { a, b } = vonMises(P, N);
        let d = 0;
        for (const x of S.linspace(-3, 8, 600)) d = Math.max(d, Math.abs(Fn(P, a * x + b, N) - S.cdf.gev(x, 0, 1, P.xi)));
        sB.set(S.fmt(b, 4)); sA.set(S.fmt(a, 4)); sK.set(S.fmt(d, 4), 'n = ' + fmtN(N)); sX.set(S.fmt(P.xi, 3));
      }
      draw();
    },
  });

  Atlas.lab('evt-intro-tails', {
    title: 'A normal fitted to the bulk, judged in the tail',
    blurb: 'Points: the empirical survival function of a sample. Solid: the true tail $\\bar F$. Dashed: a normal with the sample mean and variance. On log–log axes power-law tails are straight lines. Within the data the curves are close; beyond the data they part by orders of magnitude, and nothing about the first two moments warned you.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(['t3', 't4', 't6', 'lognormal', 'normal']), value: 't4', onChange: newSample });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 100, max: 10000, log: true, value: 1000, fmt: v => String(Math.round(v)), onChange: newSample });
      UI.button(controls, 'New sample', newSample, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, logx: true, logy: true, x: [0.3, 100], y: [1e-7, 1], xlabel: 'x (log scale)', ylabel: 'P(X > x) (log scale)' });
      const sQ = UI.stat(stats, 'True $10^{-4}$ quantile');
      const sN = UI.stat(stats, 'Normal-fit $10^{-4}$ quantile');
      const sR = UI.stat(stats, 'True / nominal prob. at normal-fit quantile', { accent: 'accent' });
      const sQ6 = UI.stat(stats, 'True vs normal $10^{-6}$ quantile');
      let x = [];
      function newSample() { const P = PAR[par.get()]; x = ctx.rng.array(Math.round(n.get()), () => P.sample(ctx.rng)); draw(); }
      function draw() {
        const P = PAR[par.get()], N = x.length, m = S.mean(x), sd = S.sd(x);
        const nQc = s => m - sd * S.qnorm(s);
        const lo = Math.max(0.05, P.Qc(0.3)), hi = Math.max(P.Qc(1e-7), nQc(1e-7)) * 1.3;
        p.setDomain([lo, hi], [1e-7, 1]).clear();
        const srt = x.slice().sort((a, b) => b - a), pts = [];
        for (let i = 0; i < N; i++) if (srt[i] > lo) pts.push([srt[i], (i + 0.5) / N]);
        p.band(lo, srt[0], { color: 'faint', opacity: 0.12 });
        p.points(pts, { color: 's2', r: 2.2, opacity: 0.7 });
        const g = S.linspace(Math.log10(lo), Math.log10(hi), 300).map(v => 10 ** v);
        p.line(g.map(v => [v, P.sf(v)]).filter(q => q[1] > 0), { color: 'ink', width: 2 });
        p.line(g.map(v => [v, nsf((v - m) / sd)]).filter(q => q[1] > 1e-12), { color: 's1', width: 2.2, dash: true });
        p.vline(srt[0], { color: 'muted' });
        p.text(srt[0], 0.5, 'sample max', { dx: 4, size: 11, color: 'muted' });
        p.legend([{ label: 'empirical', color: 's2', fill: true }, { label: 'true tail', color: 'ink' }, { label: 'normal fit', color: 's1', dash: true }]);
        const q4 = P.Qc(1e-4), n4 = nQc(1e-4);
        sQ.set(S.fmt(q4, 2)); sN.set(S.fmt(n4, 2));
        sR.set('×' + S.fmt(P.sf(n4) / 1e-4, 1), 'how much more often the "1-in-10⁴" normal level is exceeded');
        sQ6.set(`${S.fmt(P.Qc(1e-6), 1)} vs ${S.fmt(nQc(1e-6), 1)}`);
      }
      newSample();
    },
  });

  /* ========================= fisher-tippett ========================= */

  Atlas.lab('fisher-tippett-convergence', {
    title: 'Normalised maxima converge to $G_\\xi$ — at very different speeds',
    blurb: 'Histogram: 4000 maxima of $n$ IID draws, drawn exactly as $F^{-1}(U^{1/n})$ and normalised by the von Mises constants $b_n = F^{-1}(1-1/n)$, $a_n = 1/(nf(b_n))$. Solid: the limit $G_\\xi$. Dashed: the penultimate GEV with shape $r\'(b_n)$. For the normal at $n = 10^6$ the histogram looks Gumbel to the eye, yet the fitted $\\hat\\xi$ is still clearly negative, tracking the penultimate shape.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(['normal', 'exp', 'lognormal', 'pareto', 't3', 'cauchy', 'uniform', 'beta22']), value: 'normal', onChange: resample });
      const n = UI.slider(controls, { label: 'Block size $n$', min: 2, max: 1e6, log: true, value: 365, fmt: v => String(Math.round(v)), onChange: resample });
      UI.button(controls, 'Resample', resample, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, x: [-3, 7], y: [0, 0.5], xlabel: '(Mₙ − bₙ)/aₙ', ylabel: 'density' });
      const sX = UI.stat(stats, 'Limit $\\xi$');
      const sP = UI.stat(stats, 'Penultimate $\\xi_n = r\'(b_n)$');
      const sL = UI.stat(stats, 'Fitted $\\hat\\xi$ (L-moments)', { accent: 'accent' });
      const sK = UI.stat(stats, 'KS distance to $G_\\xi$');
      const R = 4000;
      let z = [], vm = null;
      function resample() {
        const P = PAR[par.get()], nn = Math.round(n.get());
        vm = vonMises(P, nn);
        z = new Array(R);
        for (let i = 0; i < R; i++) z[i] = (drawMax(P, nn, ctx.rng) - vm.b) / vm.a;
        draw();
      }
      function draw() {
        const P = PAR[par.get()], xi = P.xi;
        const h = hist(z, -3, 7, 60);
        const gx = S.linspace(-3, 7, 400);
        const lim = gx.map(x => S.pdf.gev(x, 0, 1, xi)), pen = gx.map(x => S.pdf.gev(x, 0, 1, vm.xin));
        const ymax = Math.min(1.6, Math.max(...h.map(b => b.density), ...lim.filter(Number.isFinite), 0.1) * 1.12);
        p.setDomain([-3, 7], [0, ymax]).clear();
        p.bars(h, { color: 's1', opacity: 0.4 });
        p.line(gx.map((x, i) => [x, lim[i]]), { color: 'ink', width: 2.2 });
        p.line(gx.map((x, i) => [x, pen[i]]), { color: 's2', width: 2, dash: true });
        if (h.off) p.text(7, ymax * 0.6, `${h.off} maxima off-axis`, { anchor: 'end', dx: -6, size: 11, color: 'muted' });
        p.legend([{ label: 'simulated maxima', color: 's1', fill: true, opacity: 0.4 }, { label: 'limit G_ξ', color: 'ink' }, { label: 'penultimate GEV', color: 's2', dash: true }]);
        const s = z.slice().sort((a, b) => a - b);
        let D = 0;
        for (let i = 0; i < R; i++) { const G = S.cdf.gev(s[i], 0, 1, xi); D = Math.max(D, Math.abs((i + 1) / R - G), Math.abs(i / R - G)); }
        sX.set(S.fmt(xi, 3)); sP.set(S.fmt(vm.xin, 3)); sL.set(S.fmt(gevLmom(z).xi, 3));
        sK.set(S.fmt(D, 3), D > 0.0215 ? 'above the 5% critical value 0.021' : 'within sampling noise (0.021)');
      }
      resample();
    },
  });

  /* ========================= gev-distribution ========================= */

  Atlas.lab('gev-distribution-explorer', {
    title: 'The GEV family: shape, support and return levels',
    blurb: 'Move $\\xi$ through zero and watch the support switch from bounded above ($\\xi<0$) to unbounded with a power tail ($\\xi>0$). Right: the return level $z_{1/T}$ against $T$ on a log axis — straight for the Gumbel (dashed), convex for $\\xi>0$, concave toward the endpoint for $\\xi<0$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mu = UI.slider(controls, { label: 'Location $\\mu$', min: -2, max: 2, step: 0.1, value: 0, onInput: draw });
      const sg = UI.slider(controls, { label: 'Scale $\\sigma$', min: 0.3, max: 3, step: 0.05, value: 1, onInput: draw });
      const xi = UI.slider(controls, { label: 'Shape $\\xi$', min: -1, max: 1, step: 0.01, value: 0.2, onInput: draw });
      const mode = UI.segmented(controls, { label: 'Left panel', options: ['Density', 'CDF'], value: 'Density', onChange: draw });
      const gum = UI.toggle(controls, { label: 'Overlay Gumbel ($\\xi = 0$)', value: true, onChange: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-5, 12], y: [0, 0.5], xlabel: 'z', ylabel: 'g(z)' });
      const pR = new Plot(stage, { width: 420, height: 270, logx: true, x: [1.01, 1e4], y: [0, 20], xlabel: 'return period T (log)', ylabel: 'return level' });
      const sS = UI.stat(stats, 'Support');
      const sM = UI.stat(stats, 'Mean');
      const sV = UI.stat(stats, 'Std. deviation');
      const sZ = UI.stat(stats, '100-year level $z_{0.01}$', { accent: 'accent' });
      const sK = UI.stat(stats, 'Finite moments');
      function draw() {
        const m = mu.get(), s = sg.get(), x = xi.get(), cdf = mode.get() === 'CDF';
        const gx = S.linspace(-5, 12, 600);
        const f = cdf ? (v, k) => S.cdf.gev(v, m, s, k) : (v, k) => S.pdf.gev(v, m, s, k);
        const ys = gx.map(v => f(v, x)), yg = gx.map(v => f(v, 0));
        const ymax = cdf ? 1.05 : Math.min(3, Math.max(...ys.filter(Number.isFinite), ...(gum.get() ? yg : [0])) * 1.12);
        pL.o.ylabel = cdf ? 'G(z)' : 'g(z)';
        pL.setDomain([-5, 12], [0, ymax]).clear();
        if (Math.abs(x) > 1e-9) {
          const end = m - s / x;
          if (x < 0) pL.band(end, 12, { color: 'faint', opacity: 0.25 }); else pL.band(-5, end, { color: 'faint', opacity: 0.25 });
          pL.vline(end, { color: 'muted' });
        }
        pL.area(gx.map((v, i) => [v, ys[i]]), { color: 's1', opacity: cdf ? 0 : 0.15 });
        pL.line(gx.map((v, i) => [v, ys[i]]), { color: 's1', width: 2.4 });
        if (gum.get()) pL.line(gx.map((v, i) => [v, yg[i]]), { color: 'ink', width: 1.6, dash: true, opacity: 0.7 });
        // return levels
        const Ts = S.linspace(Math.log10(1.01), 4, 200).map(v => 10 ** v);
        const zr = Ts.map(T => retLevel(m, s, x, T)), zg = Ts.map(T => retLevel(m, s, 0, T));
        const all = zr.concat(gum.get() ? zg : []);
        let y0 = Math.min(...all), y1 = Math.max(...all);
        y1 = Math.min(y1, y0 + 60); const pad = 0.06 * (y1 - y0 || 1);
        pR.setDomain([1.01, 1e4], [y0 - pad, y1 + pad]).clear();
        if (x < 0) pR.hline(m - s / x, { color: 'muted' });
        pR.line(Ts.map((T, i) => [T, zr[i]]), { color: 's1', width: 2.4 });
        if (gum.get()) pR.line(Ts.map((T, i) => [T, zg[i]]), { color: 'ink', width: 1.6, dash: true, opacity: 0.7 });
        pR.dot(100, retLevel(m, s, x, 100), { color: 's1', r: 4.5 });
        // stats
        sS.set(Math.abs(x) < 1e-9 ? '(−∞, ∞)' : x > 0 ? `(${S.fmt(m - s / x, 2)}, ∞)` : `(−∞, ${S.fmt(m - s / x, 2)})`);
        const g1 = S.gamma(1 - x), g2 = 2 * x < 1 ? S.gamma(1 - 2 * x) : NaN;
        const mean = Math.abs(x) < 1e-9 ? m + EULER * s : x < 1 ? m + s * (g1 - 1) / x : Infinity;
        const sd = Math.abs(x) < 1e-9 ? s * Math.PI / Math.sqrt(6) : x < 0.5 ? s * Math.sqrt(g2 - g1 * g1) / Math.abs(x) : Infinity;
        sM.set(S.fmt(mean, 3)); sV.set(S.fmt(sd, 3)); sZ.set(S.fmt(retLevel(m, s, x, 100), 3));
        sK.set(x <= 0 ? 'all orders' : `order < ${S.fmt(1 / x, 2)}`);
      }
      draw();
    },
  });

  Atlas.lab('gev-distribution-vonmises', {
    title: 'The local tail shape $r\'(b_n)$ as the block size grows',
    blurb: 'For each parent, $r = \\bar F/f$ is the reciprocal hazard and $r\'(b_n)$, evaluated at the $1-1/n$ quantile, is the shape of the GEV that best describes maxima of $n$ observations. The von Mises condition says it converges to $\\xi$ (dotted lines). Note the $x$-axis: $n$ up to $10^{10}$. The normal and lognormal are still far from $0$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const keys = ['normal', 'lognormal', 'exp', 't3', 'pareto', 'cauchy', 'uniform', 'beta22'];
      const nn = UI.slider(controls, { label: 'Read off at block size $n$', min: 10, max: 1e10, log: true, value: 365, fmt: v => (v < 1e4 ? String(Math.round(v)) : v.toExponential(1)), onInput: draw });
      const p = new Plot(stage, { width: 640, height: 320, logx: true, x: [10, 1e10], y: [-1.15, 1.15], xlabel: 'block size n (log)', ylabel: "r'(bₙ)" });
      const note = UI.note(stats, '');
      const curves = keys.map(k => {
        const P = PAR[k];
        const ns = S.linspace(1, 10, 91).map(v => 10 ** v);
        return ns.map(n => [n, vonMises(P, n).xin]);
      });
      function draw() {
        p.clear();
        keys.forEach((k, i) => {
          p.hline(PAR[k].xi, { color: COLS[i], dash: '2 4', width: 1, opacity: 0.6 });
          p.line(curves[i], { color: COLS[i], width: 2.2 });
        });
        const N = nn.get();
        p.vline(N, { color: 'muted' });
        p.legend(keys.map((k, i) => ({ label: PAR[k].label, color: COLS[i] })));
        const rows = keys.map(k => { const v = vonMises(PAR[k], N).xin; return `<tr><td>${PAR[k].label}</td><td>${S.fmt(v, 3)}</td><td>${S.fmt(PAR[k].xi, 3)}</td></tr>`; }).join('');
        note.set(`<table><thead><tr><th>Parent</th><th>r′(bₙ) at n = ${N < 1e4 ? Math.round(N) : N.toExponential(1)}</th><th>limit ξ</th></tr></thead><tbody>${rows}</tbody></table>`);
      }
      draw();
    },
  });

  /* ========================= block-maxima ========================= */

  const pm = (v, se) => (Number.isFinite(se) ? `${S.fmt(v, 3)} <span style="color:var(--muted)">± ${S.fmt(se, 3)}</span>` : S.fmt(v, 3));
  const seOf = (fit, i) => (fit.cov ? Math.sqrt(fit.cov[i][i]) : NaN);

  Atlas.lab('block-maxima-fit', {
    title: 'Fitting a GEV to simulated block maxima',
    blurb: 'Simulate $m$ blocks of $n$ IID observations, keep each maximum, and fit $\\text{GEV}(\\mu,\\sigma,\\xi)$ by maximum likelihood (solid) and L-moments (dashed). The quantile plot checks the MLE fit where it matters, in the upper tail. Compare $\\hat\\xi$ with the limiting $\\xi$ and with the penultimate shape at this block length; shrink $n$ to see the approximation degrade.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$ of the raw data', options: opts(['normal', 'exp', 'lognormal', 't4', 'uniform', 'beta22']), value: 'normal', onChange: newData });
      const m = UI.slider(controls, { label: 'Number of blocks $m$', min: 10, max: 200, step: 1, value: 50, onChange: newData });
      const nb = UI.slider(controls, { label: 'Block length $n$', min: 5, max: 3650, log: true, value: 365, fmt: v => String(Math.round(v)), onChange: newData });
      UI.button(controls, 'New record', newData, { primary: true });
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, xlabel: 'block maximum', ylabel: 'density', title: 'Maxima and fitted GEV densities' });
      const pQ = new Plot(stage, { width: 420, height: 270, xlabel: 'fitted quantile', ylabel: 'observed maximum', title: 'Quantile plot (MLE)' });
      const sMu = UI.stat(stats, 'MLE $\\hat\\mu$');
      const sSg = UI.stat(stats, 'MLE $\\hat\\sigma$');
      const sXi = UI.stat(stats, 'MLE $\\hat\\xi$', { accent: 'accent' });
      const sXL = UI.stat(stats, 'L-moment $\\hat\\xi$');
      const sPen = UI.stat(stats, 'Penultimate / limit $\\xi$');
      let z = [], fit = null, lm = null;
      function newData() {
        const P = PAR[par.get()], N = Math.round(nb.get());
        z = ctx.rng.array(Math.round(m.get()), () => drawMax(P, N, ctx.rng));
        fit = gevFit(z); lm = gevLmom(z);
        draw();
      }
      function draw() {
        const P = PAR[par.get()], N = Math.round(nb.get());
        const [a0, a1] = S.extent(z), pad = 0.3 * (a1 - a0 || 1), lo = a0 - pad, hi = a1 + pad;
        const h = hist(z, lo, hi, clamp(Math.round(Math.sqrt(z.length) * 1.5), 8, 30));
        const gx = S.linspace(lo, hi, 300);
        const f1 = gx.map(v => S.pdf.gev(v, fit.mu, fit.sigma, fit.xi)), f2 = gx.map(v => S.pdf.gev(v, lm.mu, lm.sigma, lm.xi));
        const ymax = Math.max(...h.map(b => b.density), ...f1.filter(Number.isFinite), ...f2.filter(Number.isFinite)) * 1.12;
        pD.setDomain([lo, hi], [0, Math.min(ymax, 3 * Math.max(...h.map(b => b.density)))]).clear();
        pD.bars(h, { color: 's1', opacity: 0.35 });
        pD.line(gx.map((v, i) => [v, f1[i]]), { color: 's1', width: 2.4 });
        pD.line(gx.map((v, i) => [v, f2[i]]), { color: 's2', width: 2, dash: true });
        for (const v of z) pD.segment(v, 0, v, ymax * 0.04, { color: 's1', width: 1, opacity: 0.7 });
        pD.legend([{ label: 'MLE', color: 's1' }, { label: 'L-moments', color: 's2', dash: true }]);
        const s = z.slice().sort((a, b) => a - b), M = s.length;
        const qq = s.map((v, i) => [S.q.gev((i + 1) / (M + 1), fit.mu, fit.sigma, fit.xi), v]);
        const all = qq.flat().filter(Number.isFinite), q0 = Math.min(...all), q1 = Math.max(...all), qp = 0.05 * (q1 - q0 || 1);
        pQ.setDomain([q0 - qp, q1 + qp], [q0 - qp, q1 + qp]).clear();
        pQ.segment(q0 - qp, q0 - qp, q1 + qp, q1 + qp, { color: 'muted', width: 1.5, dash: true });
        pQ.points(qq, { color: 's1', r: 3, opacity: 0.85 });
        sMu.set(pm(fit.mu, seOf(fit, 0))); sSg.set(pm(fit.sigma, seOf(fit, 1))); sXi.set(pm(fit.xi, seOf(fit, 2)));
        sXL.set(S.fmt(lm.xi, 3));
        sPen.set(`${S.fmt(vonMises(P, N).xin, 3)} / ${S.fmt(P.xi, 2)}`, 'r′(bₙ) at this block length / limit');
      }
      newData();
    },
  });

  Atlas.lab('block-maxima-sampling', {
    title: 'Sampling distribution of $\\hat\\xi$: MLE vs L-moments',
    blurb: 'Repeatedly draw $m$ maxima from $\\text{GEV}(0,1,\\xi)$ and estimate $\\xi$ both ways. Left: the two sampling distributions. Right: a normal quantile plot of the Wald statistic $(\\hat\\xi-\\xi)/\\widehat{\\se}$, which should hug the diagonal if standard asymptotics apply. Try small $m$, and try $\\xi$ below $-1/2$ where Smith (1985) says the usual theory breaks.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const xi = UI.slider(controls, { label: 'True shape $\\xi$', min: -0.8, max: 0.8, step: 0.05, value: 0.1, onChange: reset });
      const m = UI.slider(controls, { label: 'Maxima per sample $m$', min: 10, max: 200, step: 5, value: 30, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pH = new Plot(stage, { width: 420, height: 270, x: [-1.5, 1.5], y: [0, 3], xlabel: 'estimate of ξ', ylabel: 'density', title: 'Sampling distributions' });
      const pQ = new Plot(stage, { width: 420, height: 270, x: [-3.5, 3.5], y: [-5, 5], xlabel: 'N(0,1) quantile', ylabel: 'Wald statistic', title: 'Normal QQ of (ξ̂ − ξ)/se' });
      const sN = UI.stat(stats, 'Experiments');
      const sM = UI.stat(stats, 'MLE bias / RMSE');
      const sL = UI.stat(stats, 'L-mom bias / RMSE');
      const sC = UI.stat(stats, 'Wald 95% coverage', { accent: 'accent' });
      const sF = UI.stat(stats, 'No usable Hessian');
      let eM, eL, W, fails;
      function reset() { eM = []; eL = []; W = []; fails = 0; loop.stop(); draw(); }
      function experiment() {
        const x = xi.get(), mm = Math.round(m.get());
        const z = ctx.rng.array(mm, () => ctx.rng.gev(0, 1, x));
        const f = gevFit(z);
        eM.push(f.xi); eL.push(gevLmom(z).xi);
        const se = seOf(f, 2);
        if (Number.isFinite(se)) W.push((f.xi - x) / se); else fails++;
      }
      let acc = 0;
      function step(dt) {
        acc += dt;
        const per = clamp(Math.round(900 / m.get()), 1, 30);
        for (let i = 0; i < per; i++) experiment();
        if (eM.length >= 2000) { draw(); return false; }
        if (acc > 0.08) { acc = 0; draw(); }
      }
      function draw() {
        const x = xi.get();
        pH.clear();
        pH.vline(x, { color: 'ink', dash: false, width: 1.5 });
        if (eM.length > 5) {
          const hM = hist(eM, -1.5, 1.5, 50), hL = hist(eL, -1.5, 1.5, 50);
          const ym = Math.max(...hM.map(b => b.density), ...hL.map(b => b.density)) * 1.12;
          pH.setDomain([-1.5, 1.5], [0, ym]).clear();
          pH.vline(x, { color: 'ink', dash: false, width: 1.5 });
          pH.bars(hM, { color: 's1', opacity: 0.45 });
          pH.line(hL.map(b => [(b.x0 + b.x1) / 2, b.density]), { color: 's2', width: 2 });
          pH.legend([{ label: 'MLE', color: 's1', fill: true }, { label: 'L-moments', color: 's2' }]);
        }
        pQ.clear();
        pQ.segment(-3.5, -3.5, 3.5, 3.5, { color: 'muted', dash: true, width: 1.5 });
        if (W.length > 5) {
          const w = W.slice().sort((a, b) => a - b), K = w.length;
          const step = Math.max(1, Math.floor(K / 400));
          const pts = [];
          for (let i = 0; i < K; i += step) pts.push([S.qnorm((i + 0.5) / K), clamp(w[i], -4.9, 4.9)]);
          pQ.points(pts, { color: 's1', r: 2.4, opacity: 0.8 });
        }
        const stat = (e) => { if (!e.length) return '—'; const b = S.mean(e) - x; const r = Math.sqrt(S.mean(e.map(v => (v - x) ** 2))); return `${S.fmt(b, 3)} / ${S.fmt(r, 3)}`; };
        sN.set(String(eM.length)); sM.set(stat(eM)); sL.set(stat(eL));
        const cov = W.length ? W.filter(v => Math.abs(v) <= 1.96).length / W.length : NaN;
        sC.set(Number.isFinite(cov) ? (100 * cov).toFixed(1) + '%' : '—', 'nominal 95%');
        sF.set(eM.length ? (100 * fails / eM.length).toFixed(1) + '%' : '—');
      }
      reset();
    },
  });

  /* ========================= return-levels ========================= */

  Atlas.lab('return-levels-plot', {
    title: 'Return-level plot with delta-method and profile-likelihood intervals',
    blurb: 'An $m$-year record of annual maxima from a known $\\text{GEV}(20, 4, \\xi)$ (dashed: true return levels). Solid: fitted return levels with a pointwise 95% delta-method band. Vertical bars at the chosen $T$: delta (thin) and profile-likelihood (thick) 95% intervals. Push $T$ far beyond $m$ and watch the profile interval stretch upward while the delta interval stays symmetric.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const xi = UI.slider(controls, { label: 'True shape $\\xi$', min: -0.3, max: 0.6, step: 0.05, value: 0.1, onChange: newData });
      const m = UI.slider(controls, { label: 'Record length $m$ (years)', min: 15, max: 200, step: 1, value: 40, onChange: newData });
      const T = UI.slider(controls, { label: 'Return period $T$ (years)', min: 2, max: 1000, log: true, value: 100, fmt: v => String(Math.round(v)), onChange: () => { prof = null; draw(); } });
      UI.button(controls, 'New record', newData, { primary: true });
      const p = new Plot(stage, { width: 640, height: 330, logx: true, x: [1.05, 2000], y: [10, 60], xlabel: 'return period T (years, log scale)', ylabel: 'return level' });
      const sT = UI.stat(stats, 'True $z_{1/T}$');
      const sE = UI.stat(stats, 'Estimate $\\hat z_{1/T}$', { accent: 'accent' });
      const sD = UI.stat(stats, 'Delta-method 95% CI');
      const sP = UI.stat(stats, 'Profile 95% CI');
      const sX = UI.stat(stats, '$\\hat\\xi$');
      const MU = 20, SG = 4;
      let z = [], fit = null, prof = null;
      function newData() { z = ctx.rng.array(Math.round(m.get()), () => ctx.rng.gev(MU, SG, xi.get())); fit = gevFit(z); prof = null; draw(); }
      function draw() {
        const x = xi.get(), Tt = T.get();
        if (!prof) prof = profileCI(z, fit, Tt);
        const Ts = S.linspace(Math.log10(1.05), Math.log10(2000), 80).map(v => 10 ** v);
        const tr = Ts.map(t => retLevel(MU, SG, x, t)), fz = Ts.map(t => retLevel(fit.mu, fit.sigma, fit.xi, t));
        const se = Ts.map(t => Math.sqrt(retLevelVar(fit, t)));
        const zt = retLevel(MU, SG, x, Tt), zh = retLevel(fit.mu, fit.sigma, fit.xi, Tt), sh = Math.sqrt(retLevelVar(fit, Tt));
        const s = z.slice().sort((a, b) => a - b), M = s.length;
        const y0 = Math.min(s[0], tr[0]) - 2;
        let y1 = Math.max(tr[tr.length - 1], fz[fz.length - 1], s[M - 1], Number.isFinite(prof[1]) ? prof[1] : -Infinity, Number.isFinite(sh) ? zh + 1.96 * sh : -Infinity);
        y1 = Math.min(y1, tr[tr.length - 1] + 3 * (tr[tr.length - 1] - tr[0])) + 2;
        p.setDomain([1.05, 2000], [y0, y1]).clear();
        if (se.every(Number.isFinite)) p.area(Ts.map((t, i) => [t, fz[i] + 1.96 * se[i]]), { color: 's1', opacity: 0.15, base: Ts.map((t, i) => [t, fz[i] - 1.96 * se[i]]) });
        p.band(M, 2000, { color: 'faint', opacity: 0.12 });
        p.text(M, y1, 'beyond the record', { dx: 4, dy: 14, size: 11, color: 'muted' });
        p.line(Ts.map((t, i) => [t, tr[i]]), { color: 'ink', width: 1.8, dash: true });
        p.line(Ts.map((t, i) => [t, fz[i]]), { color: 's1', width: 2.4 });
        p.points(s.map((v, i) => [(M + 1) / (M + 1 - (i + 1)), v]), { color: 's2', r: 3, opacity: 0.85 });
        if (Number.isFinite(sh)) p.segment(Tt / 1.06, zh - 1.96 * sh, Tt / 1.06, zh + 1.96 * sh, { color: 's1', width: 2 });
        p.segment(Tt * 1.06, prof[0], Tt * 1.06, Number.isFinite(prof[1]) ? prof[1] : y1, { color: 's3', width: 5, opacity: 0.85 });
        p.dot(Tt, zh, { color: 's1', r: 4.5 });
        p.dot(Tt, zt, { color: 'ink', r: 3.5 });
        p.legend([{ label: 'fitted', color: 's1' }, { label: 'truth', color: 'ink', dash: true }, { label: 'annual maxima', color: 's2', fill: true }, { label: 'profile CI', color: 's3', width: 5 }], { left: true });
        const cov = (l, u) => (l <= zt && zt <= u ? '<span style="color:var(--good)">covers truth</span>' : '<span style="color:var(--bad)">misses truth</span>');
        sT.set(S.fmt(zt, 2)); sE.set(S.fmt(zh, 2));
        sD.set(Number.isFinite(sh) ? `[${S.fmt(zh - 1.96 * sh, 1)}, ${S.fmt(zh + 1.96 * sh, 1)}]` : '—', Number.isFinite(sh) ? cov(zh - 1.96 * sh, zh + 1.96 * sh) : '');
        sP.set(`[${S.fmt(prof[0], 1)}, ${Number.isFinite(prof[1]) ? S.fmt(prof[1], 1) : '∞'}]`, cov(prof[0], prof[1]));
        sX.set(pm(fit.xi, seOf(fit, 2)));
      }
      newData();
    },
  });

  Atlas.lab('return-levels-risk', {
    title: 'How often is the estimated $T$-year level exceeded?',
    blurb: 'Each experiment draws an $m$-year record from $\\text{GEV}(0,1,\\xi)$, estimates the $T$-year level, and computes its <em>true</em> annual exceedance probability $\\tilde p$. Left: the distribution of $\\tilde p\\,T$ (1 means "as advertised"), on a log scale. Right: the risk of at least one exceedance during a design life of $L$ years, nominal versus realised. Switch to a Gumbel fit when $\\xi>0$ to see a misspecified model compound the problem.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const T = UI.slider(controls, { label: 'Design return period $T$', min: 10, max: 1000, log: true, value: 100, fmt: v => String(Math.round(v)), onChange: reset });
      const m = UI.slider(controls, { label: 'Record length $m$', min: 10, max: 200, step: 5, value: 30, onChange: reset });
      const L = UI.slider(controls, { label: 'Design life $L$ (years)', min: 10, max: 200, step: 10, value: 50, onChange: draw });
      const xi = UI.slider(controls, { label: 'True shape $\\xi$', min: -0.3, max: 0.5, step: 0.05, value: 0.1, onChange: reset });
      const est = UI.segmented(controls, { label: 'Fitted model', options: ['GEV (MLE)', 'Gumbel (MLE)'], value: 'GEV (MLE)', onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pH = new Plot(stage, { width: 420, height: 270, x: [-2, 2], y: [0, 1], xlabel: 'log₁₀( p̃ · T )', ylabel: 'density', title: 'Realised / nominal exceedance probability' });
      const pB = new Plot(stage, { width: 420, height: 270, x: [0, 2], y: [0, 1], xticks: 0, ylabel: 'P(≥ 1 exceedance in L years)', title: 'Design-life risk' });
      const sN = UI.stat(stats, 'Experiments');
      const sM = UI.stat(stats, 'Mean $\\tilde p\\,T$', { accent: 'accent' });
      const sG = UI.stat(stats, '$P(\\tilde p > 1/T)$');
      const sR = UI.stat(stats, 'Realised vs nominal risk');
      let ps = [];
      function reset() { ps = []; loop.stop(); draw(); }
      function experiment() {
        const x = xi.get(), Tt = T.get();
        const z = ctx.rng.array(Math.round(m.get()), () => ctx.rng.gev(0, 1, x));
        const f = gevFit(z, { gumbel: est.get() !== 'GEV (MLE)' });
        const zh = retLevel(f.mu, f.sigma, f.xi, Tt);
        ps.push(1 - S.cdf.gev(zh, 0, 1, x));
      }
      let acc = 0;
      function step(dt) {
        acc += dt;
        const per = clamp(Math.round(600 / m.get()), 1, 20);
        for (let i = 0; i < per; i++) experiment();
        if (ps.length >= 2000) { draw(); return false; }
        if (acc > 0.08) { acc = 0; draw(); }
      }
      function draw() {
        const Tt = T.get(), LL = L.get();
        const lr = ps.map(p => Math.log10(Math.max(p * Tt, 1e-3)));
        pH.clear();
        if (ps.length > 5) {
          const h = hist(lr, -2, 2, 40), ym = Math.max(...h.map(b => b.density)) * 1.12;
          pH.setDomain([-2, 2], [0, ym]).clear();
          pH.bars(h, { color: 's1', opacity: 0.5 });
        }
        pH.vline(0, { color: 'ink', dash: false, width: 1.5 });
        const nom = 1 - (1 - 1 / Tt) ** LL;
        const real = ps.length ? S.mean(ps.map(p => 1 - (1 - p) ** LL)) : NaN;
        const yb = Math.min(1, Math.max(nom, Number.isFinite(real) ? real : 0) * 1.3 + 0.02);
        pB.setDomain([0, 2], [0, yb]).clear();
        pB.bars([{ x0: 0.2, x1: 0.85, y: nom }], { color: 'ink', opacity: 0.35 });
        if (Number.isFinite(real)) pB.bars([{ x0: 1.15, x1: 1.8, y: real }], { color: 's3', opacity: 0.7 });
        pB.text(0.525, nom, (100 * nom).toFixed(1) + '%', { anchor: 'middle', dy: -6, weight: 600 });
        if (Number.isFinite(real)) pB.text(1.475, real, (100 * real).toFixed(1) + '%', { anchor: 'middle', dy: -6, weight: 600 });
        pB.text(0.525, 0, 'nominal', { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        pB.text(1.475, 0, 'realised (avg)', { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        sN.set(String(ps.length));
        sM.set(ps.length ? S.fmt(S.mean(ps) * Tt, 2) : '—', '1.00 = nominal');
        sG.set(ps.length ? (100 * ps.filter(p => p > 1 / Tt).length / ps.length).toFixed(1) + '%' : '—');
        sR.set(Number.isFinite(real) ? `${(100 * real).toFixed(1)}% vs ${(100 * nom).toFixed(1)}%` : '—');
      }
      reset();
    },
  });

  /* ========================= peaks-over-threshold ========================= */

  Atlas.lab('peaks-over-threshold-fit', {
    title: 'Peaks over threshold: GPD fit and tail extrapolation',
    blurb: 'Choose a threshold $u$ as a sample quantile. Left: the excesses $X-u$ with the fitted GPD density. Right: the tail on a log scale — empirical survival (points), the fitted POT tail $\\hat\\zeta_u(1+\\hat\\xi(x-u)/\\hat\\sigma)^{-1/\\hat\\xi}$ extrapolated beyond the data (solid), and the truth (dashed). Lower the threshold for more data and more bias; raise it for less bias and more noise.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(['t4', 'gpd', 'normal', 'lognormal', 'exp', 'beta22']), value: 't4', onChange: newData });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 500, max: 50000, log: true, value: 5000, fmt: v => String(Math.round(v)), onChange: newData });
      const q = UI.slider(controls, { label: 'Threshold quantile', min: 0.5, max: 0.995, step: 0.005, value: 0.95, onChange: fitDraw });
      UI.button(controls, 'New sample', newData, { primary: true });
      stage.classList.add('two');
      const pE = new Plot(stage, { width: 420, height: 270, xlabel: 'excess y = x − u', ylabel: 'density', title: 'Excesses and fitted GPD' });
      const pT = new Plot(stage, { width: 420, height: 270, logy: true, x: [0, 10], y: [1e-6, 1], xlabel: 'x', ylabel: 'P(X > x)', title: 'Tail fit and extrapolation' });
      const sU = UI.stat(stats, 'Threshold $u$ / exceedances $k$');
      const sS = UI.stat(stats, '$\\hat\\sigma_u$');
      const sX = UI.stat(stats, '$\\hat\\xi$ (limit $\\xi$)', { accent: 'accent' });
      const sQ = UI.stat(stats, '$10^{-5}$ quantile: estimate vs truth');
      let x = [], srt = [], fit = null, u = 0, k = 0;
      function newData() { const P = PAR[par.get()]; x = ctx.rng.array(Math.round(n.get()), () => P.sample(ctx.rng)); srt = x.slice().sort((a, b) => b - a); fitDraw(); }
      function fitDraw() {
        const N = srt.length;
        k = clamp(Math.round((1 - q.get()) * N), 10, N - 1);
        u = srt[k];
        fit = gpdFit(srt.slice(0, k).map(v => v - u));
        draw();
      }
      function draw() {
        const P = PAR[par.get()], N = srt.length, zeta = k / N;
        const y = srt.slice(0, k).map(v => v - u), ymaxX = y[0] * 1.05;
        const h = hist(y, 0, ymaxX, clamp(Math.round(Math.sqrt(k)), 10, 40));
        const gx = S.linspace(0, ymaxX, 300), gd = gx.map(v => S.pdf.gpd(v, fit.sigma, fit.xi));
        const ym = Math.max(...h.map(b => b.density), ...gd.filter(Number.isFinite).slice(1)) * 1.1;
        pE.setDomain([0, ymaxX], [0, ym]).clear();
        pE.bars(h, { color: 's1', opacity: 0.4 });
        pE.line(gx.map((v, i) => [v, gd[i]]), { color: 's1', width: 2.4 });
        // tail plot
        const est = p => u + gpdQc(p / zeta, fit.sigma, fit.xi);
        const xlo = Math.min(u, srt[Math.floor(0.5 * N)]), xhi = Math.max(P.Qc(1e-6), Math.min(est(1e-6), 3 * P.Qc(1e-6)), srt[0]) * 1.05 + 0.1;
        pT.setDomain([xlo, xhi], [1e-6, 1]).clear();
        const pts = [];
        for (let i = 0; i < N; i++) { if (srt[i] < xlo) break; pts.push([srt[i], (i + 0.5) / N]); }
        const stepI = Math.max(1, Math.floor(pts.length / 1500));
        pT.points(pts.filter((_, i) => i % stepI === 0), { color: 's2', r: 2, opacity: 0.6 });
        const gt = S.linspace(xlo, xhi, 300);
        pT.line(gt.map(v => [v, P.sf(v)]).filter(p => p[1] > 0), { color: 'ink', width: 1.8, dash: true });
        pT.line(gt.filter(v => v >= u).map(v => [v, zeta * gpdSf(v - u, fit.sigma, fit.xi)]).filter(p => p[1] > 0), { color: 's1', width: 2.4 });
        pT.vline(u, { color: 'muted' });
        pT.legend([{ label: 'empirical', color: 's2', fill: true }, { label: 'POT fit', color: 's1' }, { label: 'truth', color: 'ink', dash: true }]);
        sU.set(`${S.fmt(u, 2)} / ${k}`);
        sS.set(pm(fit.sigma, Math.sqrt(fit.vs)));
        sX.set(pm(fit.xi, Math.sqrt(fit.vx)), 'limit ξ = ' + S.fmt(P.xi, 2));
        const qe = est(1e-5), qt = P.Qc(1e-5);
        sQ.set(`${S.fmt(qe, 2)} vs ${S.fmt(qt, 2)}`, `relative error ${S.fmt(100 * (qe / qt - 1), 1)}%`);
      }
      newData();
    },
  });

  /* ========================= threshold-selection ========================= */

  Atlas.lab('threshold-selection-plots', {
    title: 'Mean residual life and parameter stability plots',
    blurb: 'Left: the empirical mean excess $\\hat e(u)$ with 95% bands — linear above a good threshold. Right: $\\hat\\xi_u$ or the modified scale $\\hat\\sigma_u-\\hat\\xi_u u$ fitted at a grid of thresholds with 95% intervals — constant above a good threshold. The slider marks a candidate threshold. Compare the exact GPD with the lognormal, whose fitted shape is clearly positive although its limit is $\\xi=0$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(['gpd', 't4', 'lognormal', 'normal', 'exp']), value: 't4', onChange: newData });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 500, max: 10000, log: true, value: 2000, fmt: v => String(Math.round(v)), onChange: newData });
      const q = UI.slider(controls, { label: 'Candidate threshold (quantile)', min: 0.5, max: 0.98, step: 0.005, value: 0.9, onInput: draw });
      const what = UI.segmented(controls, { label: 'Right panel', options: ['Shape ξ', 'Modified scale σ*'], value: 'Shape ξ', onChange: draw });
      UI.button(controls, 'New sample', newData, { primary: true });
      stage.classList.add('two');
      const pM = new Plot(stage, { width: 420, height: 270, xlabel: 'threshold u', ylabel: 'mean excess', title: 'Mean residual life plot' });
      const pS = new Plot(stage, { width: 420, height: 270, xlabel: 'threshold u', ylabel: 'estimate', title: 'Parameter stability' });
      const sU = UI.stat(stats, 'Candidate $u$ / $k$');
      const sX = UI.stat(stats, '$\\hat\\xi_u$', { accent: 'accent' });
      const sS = UI.stat(stats, '$\\hat\\sigma^*_u$');
      const sT = UI.stat(stats, 'Limit $\\xi$ of parent');
      let srt = [], grid = [], mrl = [];
      function fitAt(k, start) {
        const u = srt[k], f = gpdFit(srt.slice(0, k).map(v => v - u), start);
        const ss = f.sigma - f.xi * u, vss = f.vs - 2 * u * f.cv + u * u * f.vx;
        return { u, k, f, ss, sess: Math.sqrt(vss), sex: Math.sqrt(f.vx) };
      }
      function newData() {
        const P = PAR[par.get()];
        srt = ctx.rng.array(Math.round(n.get()), () => P.sample(ctx.rng)).sort((a, b) => b - a);
        const N = srt.length;
        // MRL
        mrl = [];
        const qs = S.linspace(0.3, 1 - 15 / N, 80);
        for (const qq of qs) {
          const k = Math.round((1 - qq) * N); const u = srt[k];
          const ex = srt.slice(0, k).map(v => v - u), e = S.mean(ex), s = S.sd(ex);
          mrl.push([u, e, 1.96 * s / Math.sqrt(k)]);
        }
        // stability grid (warm starts from the previous threshold)
        grid = [];
        let start = null;
        for (const qq of S.linspace(0.5, 0.98, 25)) {
          const k = Math.max(15, Math.round((1 - qq) * N));
          const g = fitAt(k, start);
          start = [g.f.sigma, g.f.xi];
          grid.push(g);
        }
        draw();
      }
      function draw() {
        const P = PAR[par.get()], N = srt.length;
        const kc = Math.max(15, Math.round((1 - q.get()) * N));
        let near = grid.reduce((a, g) => (Math.abs(g.k - kc) < Math.abs(a.k - kc) ? g : a), grid[0]);
        const c = fitAt(kc, [near.f.sigma + near.f.xi * (srt[kc] - near.u), near.f.xi]);
        // MRL
        const ulo = mrl[0][0], uhi = mrl[mrl.length - 1][0];
        const ys = mrl.flatMap(r => [r[1] - r[2], r[1] + r[2]]);
        const y0 = Math.min(...ys), y1 = Math.max(...ys), yp = 0.05 * (y1 - y0 || 1);
        pM.setDomain([ulo, uhi], [y0 - yp, y1 + yp]).clear();
        pM.area(mrl.map(r => [r[0], r[1] + r[2]]), { color: 's1', opacity: 0.15, base: mrl.map(r => [r[0], r[1] - r[2]]) });
        pM.line(mrl.map(r => [r[0], r[1]]), { color: 's1', width: 2.2 });
        pM.vline(c.u, { color: 's3', dash: false, width: 2 });
        // stability
        const shape = what.get() === 'Shape ξ';
        const val = g => (shape ? g.f.xi : g.ss), se = g => (shape ? g.sex : g.sess);
        const lows = grid.map(g => val(g) - 1.96 * (se(g) || 0)), highs = grid.map(g => val(g) + 1.96 * (se(g) || 0));
        let s0 = Math.min(...lows.filter(Number.isFinite)), s1 = Math.max(...highs.filter(Number.isFinite));
        if (shape) { s0 = Math.max(s0, -1); s1 = Math.min(s1, 2); }
        const sp = 0.05 * (s1 - s0 || 1);
        pS.o.ylabel = shape ? 'ξ̂' : 'σ̂ − ξ̂u';
        pS.setDomain([grid[0].u, grid[grid.length - 1].u], [s0 - sp, s1 + sp]).clear();
        if (shape) pS.hline(P.xi, { color: 'ink', dash: '4 4', width: 1.2 });
        grid.forEach((g, i) => { if (Number.isFinite(se(g))) pS.segment(g.u, lows[i], g.u, highs[i], { color: 's2', width: 1.6, opacity: 0.7 }); });
        pS.line(grid.map(g => [g.u, val(g)]), { color: 's2', width: 1.2, opacity: 0.6 });
        pS.points(grid.map(g => [g.u, val(g)]), { color: 's2', r: 3 });
        pS.vline(c.u, { color: 's3', dash: false, width: 2 });
        sU.set(`${S.fmt(c.u, 2)} / ${c.k}`);
        sX.set(pm(c.f.xi, c.sex)); sS.set(pm(c.ss, c.sess)); sT.set(S.fmt(P.xi, 2));
      }
      newData();
    },
  });

  Atlas.lab('threshold-selection-tradeoff', {
    title: 'The bias–variance trade-off in the threshold, by simulation',
    blurb: 'For each simulated sample, fit a GPD using the top fraction $k/n$ of the data and estimate the $0.9999$ quantile (beyond the data for $n\\le 5000$). Accumulated over many samples: squared bias, variance and MSE of the relative error, against $k/n$ (log axes). The exact GPD has no bias, so more data always helps; the Student $t$ and the normal show interior optima, with bias growing quickly once too much of the bulk is used.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(['gpd', 't4', 'normal']), value: 't4', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 500, max: 5000, step: 500, value: 2000, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, logx: true, logy: true, x: [0.005, 0.5], y: [1e-4, 1], xlabel: 'fraction of data used, k/n (log)', ylabel: 'relative error² (log)' });
      const sN = UI.stat(stats, 'Samples');
      const sB = UI.stat(stats, 'MSE-optimal $k/n$', { accent: 'accent' });
      const sR = UI.stat(stats, 'RMSE at optimum');
      const FR = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4], PQ = 1e-4;
      let sum, sq, cnt, starts;
      function reset() { sum = FR.map(() => 0); sq = FR.map(() => 0); cnt = 0; starts = FR.map(() => null); loop.stop(); draw(); }
      function replicate() {
        const P = PAR[par.get()], N = Math.round(n.get()), xt = P.Qc(PQ);
        const s = ctx.rng.array(N, () => P.sample(ctx.rng)).sort((a, b) => b - a);
        FR.forEach((fr, j) => {
          const k = Math.max(10, Math.round(fr * N)), u = s[k];
          const f = gpdFit(s.slice(0, k).map(v => v - u), starts[j]);
          starts[j] = [f.sigma, f.xi];
          const e = (u + gpdQc(PQ / (k / N), f.sigma, f.xi)) / xt - 1;
          sum[j] += e; sq[j] += e * e;
        });
        cnt++;
      }
      function step() {
        replicate();
        if (cnt % 3 === 0) draw();
        if (cnt >= 400) { draw(); return false; }
      }
      function draw() {
        p.clear();
        if (cnt < 3) { sN.set(String(cnt)); sB.set('—'); sR.set('—'); return; }
        const b2 = FR.map((_, j) => (sum[j] / cnt) ** 2), mse = FR.map((_, j) => sq[j] / cnt), va = mse.map((v, j) => Math.max(v - b2[j], 1e-12));
        const all = mse.concat(va, b2).filter(v => v > 1e-9);
        p.setDomain([0.008, 0.5], [Math.max(1e-6, Math.min(...all) / 2), Math.max(...all) * 2]);
        const L = arr => FR.map((fr, j) => [fr, Math.max(arr[j], 1e-9)]);
        p.line(L(b2), { color: 's2', width: 2 }); p.line(L(va), { color: 's3', width: 2 }); p.line(L(mse), { color: 's1', width: 2.8 });
        p.points(L(mse), { color: 's1', r: 3.5 });
        const j = S.argmin(mse);
        p.vline(FR[j], { color: 's1' });
        p.legend([{ label: 'MSE', color: 's1' }, { label: 'bias²', color: 's2' }, { label: 'variance', color: 's3' }], { left: true });
        sN.set(String(cnt)); sB.set(String(FR[j])); sR.set(S.fmt(100 * Math.sqrt(mse[j]), 1) + '%');
      }
      reset();
    },
  });

  /* ========================= tail-index ========================= */

  function hillCurves(desc, kmax) {
    const H = new Array(kmax + 1).fill(NaN), Mo = new Array(kmax + 1).fill(NaN);
    let s1 = 0, s2 = 0;
    const lg = desc.map(v => Math.log(v));
    for (let k = 1; k <= kmax; k++) {
      s1 += lg[k - 1];
      const L = lg[k];
      // sum of (log X_(i) - log X_(k+1))^j for i <= k, from running sums of logs and squared logs
      s2 += lg[k - 1] * lg[k - 1];
      const m1 = s1 / k - L, m2 = s2 / k - 2 * L * s1 / k + L * L;
      H[k] = m1;
      Mo[k] = m2 > 0 ? m1 + 1 - 0.5 / (1 - m1 * m1 / m2) : NaN;
    }
    return { H, Mo };
  }

  Atlas.lab('tail-index-hill', {
    title: 'Hill plots, and the Pickands and moment estimators',
    blurb: 'Estimates of the tail index against the number $k$ of top order statistics used. Hill (solid, with 95% band $\\hat\\gamma(1\\pm1.96/\\sqrt k)$), Pickands and the moment estimator, with the true index dashed. Find the stable region — if there is one. Try the slowly varying "horror" parent, and shift the data to see that Hill is not location invariant.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(['pareto', 't2', 't4', 'cauchy', 'horror', 'normal']), value: 't4', onChange: newData });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 200, max: 20000, log: true, value: 2000, fmt: v => String(Math.round(v)), onChange: newData });
      const shift = UI.slider(controls, { label: 'Shift data by $c$', min: 0, max: 10, step: 0.5, value: 0, onChange: draw });
      const kf = UI.slider(controls, { label: 'Read off at $k/n$', min: 0.002, max: 0.4, log: true, value: 0.05, fmt: v => v.toFixed(3), onInput: draw });
      const logk = UI.toggle(controls, { label: 'Log scale for $k$', value: true, onChange: draw });
      const others = UI.toggle(controls, { label: 'Show Pickands and moment', value: true, onChange: draw });
      UI.button(controls, 'New sample', newData, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, x: [1, 1000], y: [-0.5, 1.5], xlabel: 'k (number of top order statistics)', ylabel: 'tail index estimate' });
      const sH = UI.stat(stats, 'Hill $\\hat\\gamma_k$', { accent: 'accent' });
      const sP = UI.stat(stats, 'Pickands $\\hat\\xi_k$');
      const sM = UI.stat(stats, 'Moment $\\hat\\xi_k$');
      const sT = UI.stat(stats, 'True $\\xi$');
      let raw = [];
      function newData() { const P = PAR[par.get()]; raw = ctx.rng.array(Math.round(n.get()), () => P.sample(ctx.rng)); draw(); }
      function draw() {
        const P = PAR[par.get()], c = shift.get(), N = raw.length;
        const desc = raw.map(v => v + c).sort((a, b) => b - a);
        let npos = 0; while (npos < N && desc[npos] > 0) npos++;
        const kmax = Math.max(2, Math.min(Math.floor(N / 2), npos - 1));
        const { H, Mo } = hillCurves(desc.slice(0, kmax + 1), kmax);
        const Pk = [];
        for (let k = 1; 4 * k <= N; k++) { const v = Math.log((desc[k - 1] - desc[2 * k - 1]) / (desc[2 * k - 1] - desc[4 * k - 1])) / Math.LN2; Pk.push([k, v]); }
        const yTop = Math.max(1.5, 2 * P.xi + 0.5);
        p.o.logx = logk.get();
        p.setDomain([logk.get() ? 2 : 1, kmax], [-0.5, yTop]).clear();
        const ks = []; for (let k = 2; k <= kmax; k++) ks.push(k);
        p.area(ks.map(k => [k, H[k] * (1 + 1.96 / Math.sqrt(k))]), { color: 's1', opacity: 0.15, base: ks.map(k => [k, H[k] * (1 - 1.96 / Math.sqrt(k))]) });
        if (others.get()) {
          p.line(Pk.filter(q => q[0] >= 2 && q[0] <= kmax), { color: 's3', width: 1.2, opacity: 0.8 });
          p.line(ks.map(k => [k, Mo[k]]), { color: 's4', width: 1.5, opacity: 0.9 });
        }
        p.line(ks.map(k => [k, H[k]]), { color: 's1', width: 2.2 });
        p.hline(P.xi, { color: 'ink', dash: '5 4', width: 1.5 });
        const kr = clamp(Math.round(kf.get() * N), 2, kmax);
        p.vline(kr, { color: 'muted' });
        const items = [{ label: 'Hill', color: 's1' }];
        if (others.get()) items.push({ label: 'Pickands', color: 's3' }, { label: 'moment', color: 's4' });
        items.push({ label: 'true ξ', color: 'ink', dash: true });
        p.legend(items);
        sH.set(pm(H[kr], H[kr] / Math.sqrt(kr)), 'k = ' + kr);
        const pk = Pk.find(q => q[0] === kr);
        sP.set(pk ? S.fmt(pk[1], 3) : '— (needs 4k ≤ n)');
        sM.set(S.fmt(Mo[kr], 3));
        sT.set(S.fmt(P.xi, 3));
      }
      newData();
    },
  });

  Atlas.lab('tail-index-mse', {
    title: 'Bias, variance and the optimal $k$ for the Hill estimator',
    blurb: 'Many samples of size $n$; for every $k$ the squared bias, variance and MSE of Hill\'s estimator (log–log axes). Exact Pareto data have no bias, so the optimum is at the largest $k$. For Student $t_\\nu$ the second-order parameter is $\\rho=-2/\\nu$, so the optimal $k$ grows like $n^{4/(\\nu+4)}$: a smaller and smaller fraction as $\\nu$ grows.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(['pareto', 'cauchy', 't2', 't4', 't10']), value: 't4', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 500, max: 5000, step: 500, value: 1000, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, logx: true, logy: true, x: [2, 500], y: [1e-5, 1], xlabel: 'k (log)', ylabel: 'error² (log)' });
      const sN = UI.stat(stats, 'Samples');
      const sK = UI.stat(stats, 'Empirical optimal $k$', { accent: 'accent' });
      const sR = UI.stat(stats, 'RMSE at optimum');
      const sE = UI.stat(stats, 'Theory: $k_{\\text{opt}}\\propto n^{a}$, $a=$');
      let sum, sq, cnt, K;
      function reset() { K = Math.floor(Math.round(n.get()) / 2) - 1; sum = new Float64Array(K + 1); sq = new Float64Array(K + 1); cnt = 0; loop.stop(); draw(); }
      function replicate() {
        const P = PAR[par.get()], N = Math.round(n.get());
        const d = ctx.rng.array(N, () => P.sample(ctx.rng)).sort((a, b) => b - a);
        let s1 = 0;
        for (let k = 1; k <= K; k++) {
          if (!(d[k] > 0)) break;
          s1 += Math.log(d[k - 1]);
          const e = s1 / k - Math.log(d[k]) - P.xi;
          sum[k] += e; sq[k] += e * e;
        }
        cnt++;
      }
      function step() {
        for (let i = 0; i < 8; i++) replicate();
        draw();
        if (cnt >= 800) return false;
      }
      function draw() {
        const key = par.get();
        sE.set(key === 'pareto' ? '1 (no bias)' : key === 'cauchy' ? '4/5' : key === 't2' ? '2/3' : key === 't4' ? '1/2' : '2/7');
        p.clear();
        if (cnt < 2) { sN.set(String(cnt)); sK.set('—'); sR.set('—'); return; }
        const b2 = [], va = [], ms = [];
        for (let k = 2; k <= K; k++) {
          const m = sum[k] / cnt, ms_ = sq[k] / cnt;
          b2.push([k, Math.max(m * m, 1e-9)]); ms.push([k, ms_]); va.push([k, Math.max(ms_ - m * m, 1e-9)]);
        }
        const all = ms.map(q => q[1]).concat(va.map(q => q[1]));
        p.setDomain([2, K], [Math.max(1e-6, Math.min(...all) / 3), Math.max(...all) * 2]);
        p.line(b2, { color: 's2', width: 1.8 }); p.line(va, { color: 's3', width: 1.8 }); p.line(ms, { color: 's1', width: 2.6 });
        const j = S.argmin(ms.map(q => q[1]));
        p.vline(ms[j][0], { color: 's1' });
        p.legend([{ label: 'MSE', color: 's1' }, { label: 'bias²', color: 's2' }, { label: 'variance', color: 's3' }], { left: true });
        sN.set(String(cnt)); sK.set(String(ms[j][0]), `k/n = ${S.fmt(ms[j][0] / Math.round(n.get()), 3)}`); sR.set(S.fmt(Math.sqrt(ms[j][1]), 3));
      }
      reset();
    },
  });

  /* ========================= point-process-evt ========================= */

  // Binomial(n, p) by inversion (p small, mean modest): exact for any n.
  function binomInv(r, n, p) {
    if (p <= 0) return 0;
    if (p >= 1) return n;
    let pk = Math.exp(n * Math.log1p(-p)), cdf = pk, k = 0;
    const u = r.u(), ratio = p / (1 - p);
    while (u > cdf && k < n) { pk *= (n - k) / (k + 1) * ratio; k++; cdf += pk; if (pk < 1e-300 && cdf > 1 - 1e-15) break; }
    return k;
  }

  Atlas.lab('point-process-evt-poisson', {
    title: 'Exceedances form a Poisson process',
    blurb: 'Left: one sample of size $n$ plotted as points $(i/(n+1), (X_i-b_n)/a_n)$, showing only those above level $-1$; the shaded box is $A=(0.25,0.75)\\times(z_0,\\infty)$. Right: over many samples, the distribution of the count $N(A)$ (bars) against the Poisson pmf with mean $\\Lambda(A) = 0.5\\,(1+\\xi z_0)^{-1/\\xi}$ (stems). Small $n$ with a normal parent shows the limit has not yet arrived.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const par = UI.select(controls, { label: 'Parent $F$', options: opts(['exp', 'pareto', 'normal', 'uniform']), value: 'exp', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 1e6, log: true, value: 100, fmt: v => String(Math.round(v)), onChange: reset });
      const z0 = UI.slider(controls, { label: 'Box level $z_0$', min: -0.5, max: 3, step: 0.1, value: 0.5, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Repeat samples', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-1, 6], xlabel: 'time i/(n+1)', ylabel: 'normalised level', title: 'One sample: points above −1' });
      const pC = new Plot(stage, { width: 420, height: 270, x: [-0.5, 8.5], y: [0, 0.5], xlabel: 'N(A)', ylabel: 'probability', title: 'Counts in A vs Poisson(Λ(A))' });
      const sL = UI.stat(stats, '$\\Lambda(A)$ (limit)');
      const sM = UI.stat(stats, 'Mean / variance of $N(A)$', { accent: 'accent' });
      const sZ = UI.stat(stats, '$\\hat P(N=0)$ vs $e^{-\\Lambda}$');
      const sR = UI.stat(stats, 'Samples');
      const ULOW = -1;
      let counts, last;
      function sample() {
        const P = PAR[par.get()], N = Math.round(n.get()), { a, b } = vonMises(P, N);
        const v = a * ULOW + b, p = P.sf(v);
        const K = binomInv(ctx.rng, N, p), pts = [];
        for (let j = 0; j < K; j++) pts.push([ctx.rng.u(), (P.Qc(p * u01(ctx.rng)) - b) / a]);
        return pts;
      }
      function lam() { const xi = PAR[par.get()].xi, z = z0.get(); const t = 1 + xi * z; return t <= 0 ? 0 : 0.5 * (Math.abs(xi) < 1e-9 ? Math.exp(-z) : t ** (-1 / xi)); }
      function reset() { counts = []; last = sample(); loop.stop(); draw(); }
      function step() {
        for (let i = 0; i < 100; i++) { const pts = sample(); counts.push(pts.filter(q => q[0] > 0.25 && q[0] < 0.75 && q[1] > z0.get()).length); last = pts; }
        draw();
        if (counts.length >= 5000) return false;
      }
      function draw() {
        const z = z0.get(), Lm = lam();
        pP.clear();
        pP.rect(0.25, z, 0.75, 6, { color: 's3', opacity: 0.15, stroke: 's3' });
        pP.hline(z, { color: 's3' });
        pP.points(last.map(q => [q[0], Math.min(q[1], 5.9)]), { color: 's1', r: 4, opacity: 0.9 });
        const kmax = 8;
        const emp = new Array(kmax + 1).fill(0);
        for (const c of counts) emp[Math.min(c, kmax)]++;
        const pois = S.range(kmax + 1).map(k => [k, S.pmf.poisson(k, Lm)]);
        const ym = Math.max(0.2, ...pois.map(q => q[1]), ...emp.map(c => (counts.length ? c / counts.length : 0))) * 1.15;
        pC.setDomain([-0.5, kmax + 0.5], [0, ym]).clear();
        if (counts.length) pC.bars(emp.map((c, k) => ({ x0: k - 0.35, x1: k + 0.35, y: c / counts.length })), { color: 's1', opacity: 0.45 });
        pC.stems(pois, { color: 'ink', width: 2 });
        pC.legend([{ label: 'simulated', color: 's1', fill: true }, { label: 'Poisson(Λ(A))', color: 'ink' }]);
        sL.set(S.fmt(Lm, 3));
        if (counts.length > 1) {
          sM.set(`${S.fmt(S.mean(counts), 3)} / ${S.fmt(S.variance(counts), 3)}`, 'Poisson: equal');
          sZ.set(`${S.fmt(emp[0] / counts.length, 3)} vs ${S.fmt(Math.exp(-Lm), 3)}`);
        } else { sM.set('—'); sZ.set('—'); }
        sR.set(String(counts.length));
      }
      reset();
    },
  });

  /* ========================= extremal-index ========================= */

  // Ferro & Segers (2003) intervals estimator from exceedance times.
  function intervalsTheta(times) {
    const N = times.length;
    if (N < 3) return NaN;
    const T = []; for (let i = 1; i < N; i++) T.push(times[i] - times[i - 1]);
    const mx = Math.max(...T);
    let a, b;
    if (mx <= 2) { a = S.sum(T); b = S.sum(T.map(t => t * t)); return Math.min(1, 2 * a * a / ((N - 1) * b)); }
    a = S.sum(T.map(t => t - 1)); b = S.sum(T.map(t => (t - 1) * (t - 2)));
    return Math.min(1, 2 * a * a / ((N - 1) * b));
  }
  // Runs declustering: returns cluster sizes (numbers of exceedances).
  function runsClusters(times, r) {
    const sizes = [];
    let cur = 0, lastT = -Infinity;
    for (const t of times) { if (t - lastT > r) { if (cur) sizes.push(cur); cur = 1; } else cur++; lastT = t; }
    if (cur) sizes.push(cur);
    return sizes;
  }
  const frechet = r => -1 / Math.log(u01(r));

  Atlas.lab('extremal-index-armax', {
    title: 'Clusters of extremes: ARMAX versus Gaussian AR(1)',
    blurb: 'Both series are shown on a common Gumbel scale. Exceedances of the threshold are marked; runs declustering with run length $r$ groups them into clusters. ARMAX has $\\theta = 1-\\alpha$ and geometric cluster sizes; the Gaussian AR(1) has $\\theta=1$ however large $\\phi$ is, though at finite thresholds some clustering remains. Raise the threshold to watch the AR(1) estimates creep toward 1.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const proc = UI.segmented(controls, { label: 'Process', options: ['ARMAX', 'Gaussian AR(1)'], value: 'ARMAX', onChange: newData });
      const dep = UI.slider(controls, { label: 'Dependence $\\alpha$ / $\\phi$', min: 0, max: 0.95, step: 0.05, value: 0.7, onChange: newData });
      const n = UI.slider(controls, { label: 'Series length $n$', min: 2000, max: 50000, log: true, value: 10000, fmt: v => String(Math.round(v)), onChange: newData });
      const q = UI.slider(controls, { label: 'Threshold (quantile)', min: 0.9, max: 0.998, step: 0.001, value: 0.98, onChange: draw });
      const r = UI.slider(controls, { label: 'Run length $r$', min: 1, max: 20, step: 1, value: 3, onChange: draw });
      UI.button(controls, 'New series', newData, { primary: true });
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'Gumbel-scale value', title: 'Series around its largest value' });
      const pC = new Plot(stage, { width: 420, height: 270, x: [0.5, 10.5], y: [0, 1], xlabel: 'cluster size', ylabel: 'proportion', title: 'Cluster sizes (runs declustering)' });
      const sT = UI.stat(stats, 'True $\\theta$ (limit)');
      const sR = UI.stat(stats, 'Runs $\\hat\\theta$', { accent: 'accent' });
      const sI = UI.stat(stats, 'Intervals $\\hat\\theta$');
      const sN = UI.stat(stats, 'Exceedances / clusters');
      let g = [];
      function newData() {
        const N = Math.round(n.get()), a = dep.get();
        g = new Array(N);
        if (proc.get() === 'ARMAX') {
          let x = frechet(ctx.rng);
          for (let t = 0; t < N; t++) { if (t) x = Math.max(a * x, (1 - a) * frechet(ctx.rng)); g[t] = Math.log(x); }
        } else {
          let x = ctx.rng.normal(), c = Math.sqrt(1 - a * a);
          for (let t = 0; t < N; t++) { if (t) x = a * x + c * ctx.rng.normal(); g[t] = Math.max(-10, -Math.log(-Math.log1p(-nsf(x)))); }
        }
        draw();
      }
      function draw() {
        const N = g.length, u = -Math.log(-Math.log(q.get()));
        const times = []; for (let t = 0; t < N; t++) if (g[t] > u) times.push(t);
        const sizes = runsClusters(times, Math.round(r.get()));
        const tmax = S.argmax(g), w0 = clamp(tmax - 150, 0, Math.max(0, N - 300)), w1 = Math.min(N, w0 + 300);
        const win = []; for (let t = w0; t < w1; t++) win.push([t, g[t]]);
        const ylo = Math.min(...win.map(q => q[1])), yhi = Math.max(...win.map(q => q[1]), u + 1);
        pS.setDomain([w0, w1], [ylo - 0.3, yhi + 0.3]).clear();
        pS.line(win, { color: 'muted', width: 1.2 });
        pS.hline(u, { color: 's3', dash: '5 4', width: 1.5 });
        pS.points(win.filter(q => q[1] > u), { color: 's1', r: 3.5 });
        // cluster-size distribution
        const kmax = 10, emp = new Array(kmax + 1).fill(0);
        for (const s of sizes) emp[Math.min(s, kmax)]++;
        const a = dep.get(), armax = proc.get() === 'ARMAX';
        const geo = S.range(kmax).map(j => [j + 1, (1 - a) * a ** j]);
        const ym = Math.max(0.2, ...emp.map(c => c / Math.max(1, sizes.length)), ...(armax ? geo.map(q => q[1]) : [0])) * 1.15;
        pC.setDomain([0.5, kmax + 0.5], [0, ym]).clear();
        pC.bars(emp.slice(1).map((c, j) => ({ x0: j + 1 - 0.35, x1: j + 1 + 0.35, y: c / Math.max(1, sizes.length) })), { color: 's1', opacity: 0.45 });
        if (armax) { pC.stems(geo, { color: 'ink', width: 2 }); pC.legend([{ label: 'runs clusters', color: 's1', fill: true }, { label: 'limit: geometric(1 − α)', color: 'ink' }]); }
        else pC.legend([{ label: 'runs clusters', color: 's1', fill: true }]);
        pC.text(kmax, 0, '≥10', { anchor: 'middle', dy: -4, size: 10, color: 'muted' });
        sT.set(armax ? S.fmt(1 - a, 2) : '1');
        sR.set(times.length ? S.fmt(sizes.length / times.length, 3) : '—', 'clusters / exceedances');
        sI.set(S.fmt(intervalsTheta(times), 3), 'Ferro–Segers');
        sN.set(`${times.length} / ${sizes.length}`);
      }
      newData();
    },
  });

  Atlas.lab('extremal-index-maxima', {
    title: 'The maximum of a dependent series behaves like $F^{n\\theta}$',
    blurb: 'Simulate many ARMAX series of length $n$ (unit Fréchet margins) and plot the empirical CDF of $M_n/n$. Dotted: the IID prediction $F^n$, i.e. $\\exp(-1/x)$. Solid: the dependent prediction $F^{n\\theta}$, $\\exp(-\\theta/x)$ with $\\theta = 1-\\alpha$. The series has $n$ observations but only about $n\\theta$ independent chances to produce a new maximum.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const al = UI.slider(controls, { label: 'ARMAX $\\alpha$', min: 0, max: 0.9, step: 0.05, value: 0.6, onChange: reset });
      const n = UI.slider(controls, { label: 'Block length $n$', min: 20, max: 2000, log: true, value: 365, fmt: v => String(Math.round(v)), onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Simulate', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 5], y: [0, 1], xlabel: 'Mₙ / n', ylabel: 'P(Mₙ/n ≤ x)' });
      const sN = UI.stat(stats, 'Series simulated');
      const sMd = UI.stat(stats, 'Median of $M_n/n$', { accent: 'accent' });
      const sP = UI.stat(stats, 'Predicted: $\\theta/\\log 2$ vs IID $1/\\log 2$');
      const sK = UI.stat(stats, 'KS distance to $F^{n\\theta}$ / $F^n$');
      let M = [];
      function reset() { M = []; loop.stop(); draw(); }
      function series() {
        const a = al.get(), N = Math.round(n.get());
        let x = frechet(ctx.rng), m = x;
        for (let t = 1; t < N; t++) { x = Math.max(a * x, (1 - a) * frechet(ctx.rng)); if (x > m) m = x; }
        return m / N;
      }
      function step() {
        const per = Math.max(1, Math.round(60000 / n.get()));
        for (let i = 0; i < per; i++) M.push(series());
        draw();
        if (M.length >= 3000) return false;
      }
      function draw() {
        const th = 1 - al.get(), N = Math.round(n.get());
        const xs = S.linspace(0.01, 5, 300);
        p.clear();
        p.line(xs.map(x => [x, Math.exp(-1 / x)]), { color: 's2', width: 2, dash: '2 4' });
        p.line(xs.map(x => [x, Math.exp(-(1 + (N - 1) * th) / (N * x))]), { color: 'ink', width: 2 });
        if (M.length) {
          const s = M.slice().sort((a, b) => a - b), K = s.length;
          p.step(s.map((v, i) => [Math.min(v, 5), (i + 1) / K]), { color: 's1', width: 2 });
          let d1 = 0, d2 = 0;
          for (let i = 0; i < K; i++) {
            const G1 = Math.exp(-(1 + (N - 1) * th) / (N * s[i])), G2 = Math.exp(-1 / s[i]);
            d1 = Math.max(d1, Math.abs((i + 1) / K - G1), Math.abs(i / K - G1)); d2 = Math.max(d2, Math.abs((i + 1) / K - G2), Math.abs(i / K - G2));
          }
          sMd.set(S.fmt(s[Math.floor(K / 2)], 3));
          sK.set(`${S.fmt(d1, 3)} / ${S.fmt(d2, 3)}`);
        } else { sMd.set('—'); sK.set('—'); }
        p.legend([{ label: 'simulated', color: 's1' }, { label: 'F^{nθ}', color: 'ink' }, { label: 'F^n (IID)', color: 's2', dash: true }], { left: false });
        sN.set(String(M.length));
        sP.set(`${S.fmt(th / Math.LN2, 3)} vs ${S.fmt(1 / Math.LN2, 3)}`);
      }
      reset();
    },
  });

  /* ========================= multivariate-extremes ========================= */

  // Positive alpha-stable variable with Laplace transform exp(-t^alpha) (Kanter / Chambers–Mallows–Stuck).
  function posStable(r, a) {
    if (a >= 1) return 1;
    const U = Math.PI * u01(r), W = r.exp(1);
    return Math.sin(a * U) / Math.sin(U) ** (1 / a) * (Math.sin((1 - a) * U) / W) ** ((1 - a) / a);
  }
  // Ranks scaled to (0,1).
  function toUnif(x) {
    const n = x.length, idx = S.range(n).sort((i, j) => x[i] - x[j]), u = new Array(n);
    idx.forEach((i, r) => { u[i] = (r + 1) / (n + 1); });
    return u;
  }
  const chiT = (rho, nu) => 2 * tSurv(Math.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1);

  Atlas.lab('multivariate-extremes-chi', {
    title: 'Tail dependence: $\\chi(u)$ and $\\bar\\chi(u)$',
    blurb: 'Left: pairs on standard exponential margins, where joint extremes sit in the upper right. Right: empirical $\\chi(u)$ (solid) and $\\bar\\chi(u)$ (dashed) as $u\\to1$, with their theoretical limits as thin lines. The Gaussian is asymptotically independent ($\\chi=0$, $\\bar\\chi=\\rho$) even at $\\rho=0.9$; the logistic and $t$ copulas are asymptotically dependent ($\\bar\\chi=1$) — the $t$ even at $\\rho=0$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const model = UI.select(controls, { label: 'Dependence model', options: [{ value: 'gauss', label: 'Gaussian copula' }, { value: 'logistic', label: 'Logistic (Gumbel) EV copula' }, { value: 't', label: 'Student t copula (ν = 3)' }, { value: 'indep', label: 'Independence' }], value: 'gauss', onChange: newData });
      const rho = UI.slider(controls, { label: 'Correlation $\\rho$ (Gaussian, t)', min: -0.5, max: 0.99, step: 0.01, value: 0.7, onChange: newData });
      const alpha = UI.slider(controls, { label: 'Logistic $\\alpha$', min: 0.1, max: 1, step: 0.05, value: 0.5, onChange: newData });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 1000, max: 50000, log: true, value: 10000, fmt: v => String(Math.round(v)), onChange: newData });
      UI.button(controls, 'Resample', newData, { primary: true });
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 270, x: [0, 10], y: [0, 10], xlabel: 'X₁ (exponential scale)', ylabel: 'X₂ (exponential scale)', title: 'Sample on exponential margins' });
      const pC = new Plot(stage, { width: 420, height: 270, x: [0.5, 1], y: [-0.4, 1.05], xlabel: 'u', ylabel: 'coefficient', title: 'χ(u) and χ̄(u)' });
      const sC = UI.stat(stats, 'Limit $\\chi$');
      const sCh = UI.stat(stats, '$\\hat\\chi(0.99)$', { accent: 'accent' });
      const sB = UI.stat(stats, 'Limit $\\bar\\chi$');
      const sBh = UI.stat(stats, '$\\hat{\\bar\\chi}(0.99)$');
      let U = [], V = [];
      function newData() {
        const N = Math.round(n.get()), md = model.get(), rr = rho.get(), a = alpha.get();
        const X = new Array(N), Y = new Array(N);
        for (let i = 0; i < N; i++) {
          if (md === 'logistic') { const s = posStable(ctx.rng, a); X[i] = Math.exp(-((ctx.rng.exp(1) / s) ** a)); Y[i] = Math.exp(-((ctx.rng.exp(1) / s) ** a)); }
          else if (md === 'indep') { X[i] = ctx.rng.u(); Y[i] = ctx.rng.u(); }
          else {
            const z1 = ctx.rng.normal(), z2 = rr * z1 + Math.sqrt(1 - rr * rr) * ctx.rng.normal();
            const w = md === 't' ? Math.sqrt(ctx.rng.chisq(3) / 3) : 1;
            X[i] = z1 / w; Y[i] = z2 / w;
          }
        }
        U = toUnif(X); V = toUnif(Y);
        draw();
      }
      function chis(u) {
        let both = 0, bothUp = 0;
        for (let i = 0; i < U.length; i++) { if (U[i] < u && V[i] < u) both++; if (U[i] > u && V[i] > u) bothUp++; }
        const N = U.length, C = both / N, Cb = bothUp / N;
        return [2 - Math.log(C) / Math.log(u), Cb > 0 ? 2 * Math.log(1 - u) / Math.log(Cb) - 1 : NaN];
      }
      function draw() {
        const md = model.get(), rr = rho.get(), a = alpha.get(), N = U.length;
        const pts = [];
        for (let i = 0; i < N; i++) pts.push([-Math.log(1 - U[i]), -Math.log(1 - V[i])]);
        const top = -Math.log(1 / (N + 1)) + 0.3;
        pS.setDomain([0, top], [0, top]).clear();
        pS.points(pts, { color: 's1', r: 1.4, opacity: 0.35 });
        const e99 = -Math.log(0.01);
        pS.rect(e99, e99, top, top, { color: 's3', opacity: 0.12, stroke: 's3' });
        const us = S.linspace(0.5, 1 - 20 / N, 60), cs = us.map(chis);
        pC.clear();
        const limC = md === 'gauss' ? (rr >= 0.999 ? 1 : 0) : md === 'logistic' ? 2 - 2 ** a : md === 't' ? chiT(rr, 3) : 0;
        const limB = md === 'gauss' ? rr : md === 'logistic' ? (a >= 0.999 ? 0 : 1) : md === 't' ? 1 : 0;
        pC.hline(limC, { color: 's1', width: 1, dash: '2 3' });
        pC.hline(limB, { color: 's2', width: 1, dash: '2 3' });
        pC.hline(0, { color: 'muted', width: 0.8, dash: false });
        pC.line(us.map((u, i) => [u, cs[i][0]]), { color: 's1', width: 2.4 });
        pC.line(us.map((u, i) => [u, cs[i][1]]), { color: 's2', width: 2.4, dash: true });
        pC.legend([{ label: 'χ(u)', color: 's1' }, { label: 'χ̄(u)', color: 's2', dash: true }], { left: true });
        const c99 = chis(0.99);
        sC.set(S.fmt(limC, 3)); sCh.set(S.fmt(c99[0], 3)); sB.set(S.fmt(limB, 3)); sBh.set(S.fmt(c99[1], 3));
      }
      newData();
    },
  });

  Atlas.lab('multivariate-extremes-pickands', {
    title: 'The Pickands dependence function from componentwise maxima',
    blurb: 'Estimate $A(w)$ by Pickands\' estimator $\\hat A(w) = m\\big/\\sum_i\\min\\{S_i/(1-w), T_i/w\\}$ from $m$ componentwise maxima transformed by ranks to exponential margins. The triangle and the line $A=1$ bound every valid $A$. Logistic maxima are exactly max-stable. Gaussian block maxima are asymptotically independent ($A\\equiv1$) but approach it slowly: increase the block size and watch.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const model = UI.segmented(controls, { label: 'Maxima from', options: ['Logistic', 'Gaussian blocks'], value: 'Logistic', onChange: newData });
      const alpha = UI.slider(controls, { label: 'Logistic $\\alpha$', min: 0.1, max: 1, step: 0.05, value: 0.5, onChange: newData });
      const rho = UI.slider(controls, { label: 'Gaussian $\\rho$', min: 0, max: 0.99, step: 0.01, value: 0.8, onChange: newData });
      const nb = UI.slider(controls, { label: 'Gaussian block size', min: 1, max: 1000, log: true, value: 50, fmt: v => String(Math.round(v)), onChange: newData });
      const m = UI.slider(controls, { label: 'Number of maxima $m$', min: 25, max: 1000, log: true, value: 200, fmt: v => String(Math.round(v)), onChange: newData });
      UI.button(controls, 'Resample', newData, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 1], y: [0.45, 1.05], xlabel: 'w', ylabel: 'A(w)' });
      const sA = UI.stat(stats, '$\\hat A(1/2)$', { accent: 'accent' });
      const sT = UI.stat(stats, 'Extremal coeff. $2\\hat A(1/2)$');
      const sC = UI.stat(stats, '$\\hat\\chi = 2 - 2\\hat A(1/2)$');
      const sL = UI.stat(stats, 'Truth (limit)');
      let Sx = [], Tx = [];
      function newData() {
        const M = Math.round(m.get()), X = new Array(M), Y = new Array(M);
        if (model.get() === 'Logistic') {
          const a = alpha.get();
          for (let i = 0; i < M; i++) { const s = posStable(ctx.rng, a); X[i] = (ctx.rng.exp(1) / s) ** a; Y[i] = (ctx.rng.exp(1) / s) ** a; X[i] = -X[i]; Y[i] = -Y[i]; }
        } else {
          const rr = rho.get(), c = Math.sqrt(1 - rr * rr), B = Math.round(nb.get());
          for (let i = 0; i < M; i++) {
            let mx = -Infinity, my = -Infinity;
            for (let j = 0; j < B; j++) { const z1 = ctx.rng.normal(), z2 = rr * z1 + c * ctx.rng.normal(); if (z1 > mx) mx = z1; if (z2 > my) my = z2; }
            X[i] = mx; Y[i] = my;
          }
        }
        const U = toUnif(X), V = toUnif(Y);
        Sx = U.map(u => -Math.log(u)); Tx = V.map(v => -Math.log(v));
        draw();
      }
      const Ahat = w => { let s = 0; for (let i = 0; i < Sx.length; i++) s += Math.min(w < 1 ? Sx[i] / (1 - w) : Infinity, w > 0 ? Tx[i] / w : Infinity); return Sx.length / s; };
      function draw() {
        const ws = S.linspace(0, 1, 101), logi = model.get() === 'Logistic', a = alpha.get();
        p.clear();
        p.polygon([[0, 1], [0.5, 0.5], [1, 1]], { color: 'faint', opacity: 0.25 });
        p.line([[0, 1], [0.5, 0.5], [1, 1]], { color: 'muted', width: 1.2 });
        p.hline(1, { color: 'muted', dash: false, width: 1.2 });
        const truth = w => (logi ? (w ** (1 / a) + (1 - w) ** (1 / a)) ** a : 1);
        p.line(ws.map(w => [w, truth(w)]), { color: 'ink', width: 2, dash: true });
        const est = ws.map(w => [w, Ahat(w)]);
        p.line(est, { color: 's1', width: 2.4 });
        p.legend([{ label: 'Pickands estimate', color: 's1' }, { label: logi ? 'true A (logistic)' : 'limit A ≡ 1', color: 'ink', dash: true }, { label: 'bounds', color: 'muted' }]);
        const a5 = Ahat(0.5), t5 = truth(0.5);
        sA.set(S.fmt(a5, 3)); sT.set(S.fmt(2 * a5, 3)); sC.set(S.fmt(2 - 2 * a5, 3));
        sL.set(`A(½) = ${S.fmt(t5, 3)}, χ = ${S.fmt(2 - 2 * t5, 3)}`);
      }
      newData();
    },
  });
})();
