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
  const relabel = (sl, html) => { const l = sl.el.querySelector('.ctl-label'); if (l) { l.innerHTML = html; UI.math(l); } };

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
    blurb: 'Histogram: 4000 maxima of $n$ IID draws, drawn exactly as $F^{-1}(U^{1/n})$ and normalised by the von Mises constants $b_n = F^{-1}(1-1/n)$, $a_n = 1/(nf(b_n))$. Solid: the limit $G_\\xi$. Dashed: the penultimate GEV with shape $r\'(b_n)$. Push $n$ to $10^6$ for the normal and see that the histogram still leans on the penultimate curve.',
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
})();
