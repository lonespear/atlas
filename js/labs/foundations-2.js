/* Labs: Foundations II — inequalities, information, convergence, limit theorems, sampling distributions, order statistics. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const R = String.raw;
  const LOG_TINY = 1e-300;
  const clampP = v => (v > 0 ? Math.min(1, v) : null);           // for log-scale plots: drop zeros
  const ratio = (b, e) => (e > 0 && Number.isFinite(b) ? '×' + S.fmt(b / e, b / e >= 100 ? 0 : 2) + ' exact' : '');

  /* ====================================================================
   * inequalities
   * ================================================================== */
  // Nonnegative distributions with sf (survival), cumulant generating fn K, and log-moments.
  const TAILS = {
    exp: { label: 'Exponential(1)', mean: 1, v: 1, xmax: 16, lmax: 1,
      sf: x => (x <= 0 ? 1 : Math.exp(-x)), K: l => -Math.log(1 - l), lmom: k => S.lgamma(k + 1) },
    gamma: { label: 'Gamma(4, 1)', mean: 4, v: 4, xmax: 26, lmax: 1,
      sf: x => (x <= 0 ? 1 : 1 - S.gammaP(4, x)), K: l => -4 * Math.log(1 - l), lmom: k => S.lgamma(k + 4) - S.lgamma(4) },
    unif: { label: 'Uniform(0, 1)', mean: 0.5, v: 1 / 12, xmax: 1.05, lmax: Infinity, lcap: 2000,
      sf: x => (x <= 0 ? 1 : x >= 1 ? 0 : 1 - x), K: l => (l < 1e-9 ? l / 2 : l + Math.log1p(-Math.exp(-l)) - Math.log(l)), lmom: k => -Math.log(k + 1) },
    half: { label: 'Half-normal |N(0,1)|', mean: Math.sqrt(2 / Math.PI), v: 1 - 2 / Math.PI, xmax: 5.4, lmax: Infinity, lcap: 60,
      sf: x => (x <= 0 ? 1 : 1 - S.gammaP(0.5, x * x / 2)), K: l => Math.log(2) + l * l / 2 + Math.log(S.cdf.normal(l)),
      lmom: k => (k / 2) * Math.log(2) + S.lgamma((k + 1) / 2) - 0.5 * Math.log(Math.PI) },
    lnorm: { label: 'Lognormal(0, 1)', mean: Math.exp(0.5), v: (Math.E - 1) * Math.E, xmax: 160, lmax: 0,
      sf: x => (x <= 0 ? 1 : 1 - S.cdf.normal(Math.log(x))), K: () => Infinity, lmom: k => k * k / 2 },
    pareto: { label: 'Pareto(α = 3, xₘ = 1)', mean: 1.5, v: 0.75, xmax: 120, lmax: 0,
      sf: x => (x <= 1 ? 1 : x ** -3), K: () => Infinity, lmom: k => (k < 3 ? Math.log(3 / (3 - k)) : Infinity) },
  };
  function chernoff(D, x) {
    if (!(D.lmax > 0) || x <= D.mean) return 1;
    const hi = Number.isFinite(D.lmax) ? D.lmax * (1 - 1e-9) : D.lcap;
    const f = l => D.K(l) - l * x;
    const l = S.golden(f, 0, hi, 1e-10);
    return Math.min(1, Math.exp(Math.min(0, f(l))));
  }
  function momentBound(D, x) {
    let best = 0; // log of bound, k = 0 gives log 1 = 0
    for (let k = 1; k <= 120; k++) { const v = D.lmom(k) - k * Math.log(x); if (v < best) best = v; }
    return Math.exp(best);
  }

  Atlas.lab('inequalities-tails', {
    title: 'How tight is each tail bound?',
    blurb: R`The true tail $\P(X\ge x)$ of a nonnegative distribution (black) against Markov, Chebyshev, Cantelli, Chernoff and the best moment bound $\min_k \E X^k/x^k$, on a log scale. Compare light tails (exponential, half-normal) with heavy ones (lognormal, Pareto), where the MGF is infinite and Chernoff collapses to the trivial bound $1$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dist = UI.select(controls, { label: 'Distribution', options: Object.keys(TAILS).map(k => ({ value: k, label: TAILS[k].label })), value: 'exp', onChange: () => { th.set(th.get()); draw(); } });
      const th = UI.slider(controls, { label: 'Threshold $x$ (fraction of axis)', min: 0.02, max: 1, step: 0.01, value: 0.35, onInput: draw,
        fmt: v => 'x = ' + S.fmt(v * TAILS[dist.get()].xmax, 2) });
      const showMom = UI.toggle(controls, { label: 'Best moment bound', value: true, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 1], y: [1e-7, 1.5], logy: true, xlabel: 'x', ylabel: 'P(X ≥ x)  (log scale)' });
      const sE = UI.stat(stats, 'Exact $\\P(X\\ge x)$', { accent: 'accent' });
      const sM = UI.stat(stats, 'Markov $\\E X/x$');
      const sC = UI.stat(stats, 'Chebyshev $\\sigma^2/(x-\\mu)^2$');
      const sCa = UI.stat(stats, 'Cantelli');
      const sCh = UI.stat(stats, 'Chernoff');
      const sMo = UI.stat(stats, 'Moment $\\min_k \\E X^k/x^k$');
      const BOUNDS = [
        ['Markov', 's1', (D, x) => D.mean / x],
        ['Chebyshev', 's2', (D, x) => (x > D.mean ? D.v / (x - D.mean) ** 2 : 1)],
        ['Cantelli', 's3', (D, x) => (x > D.mean ? D.v / (D.v + (x - D.mean) ** 2) : 1)],
        ['Chernoff', 's4', chernoff],
        ['Moment', 's5', momentBound],
      ];
      function draw() {
        const D = TAILS[dist.get()], xmax = D.xmax;
        p.setDomain([0, xmax], [1e-7, 1.5]).clear();
        const xs = S.linspace(xmax * 0.004, xmax, 260);
        p.line(xs.map(x => [x, clampP(D.sf(x))]), { color: 'ink', width: 2.6 });
        const leg = [{ label: 'exact tail', color: 'ink' }];
        for (const [name, c, f] of BOUNDS) {
          if (name === 'Moment' && !showMom.get()) continue;
          p.line(xs.map(x => [x, clampP(Math.min(1, f(D, x)))]), { color: c, width: 1.8, dash: name === 'Moment' ? '2 3' : name === 'Cantelli' ? '6 3' : false });
          leg.push({ label: name, color: c, dash: name === 'Moment' || name === 'Cantelli' });
        }
        const x = th.get() * xmax;
        p.vline(x, { color: 'muted', width: 1 });
        p.vline(D.mean, { color: 'faint', width: 1, dash: '2 3' });
        p.text(D.mean, 1.2, 'μ', { anchor: 'middle', dy: -2, color: 'muted', size: 11 });
        p.legend(leg);
        const e = D.sf(x);
        sE.set(e > 0 ? S.fmt(e) : '0', x <= D.mean ? 'x ≤ μ: two-moment bounds are trivial' : '');
        const vals = BOUNDS.map(([, , f]) => Math.min(1, f(D, x)));
        [sM, sC, sCa, sCh, sMo].forEach((s, i) => s.set(S.fmt(vals[i]), ratio(vals[i], e)));
        if (!(D.lmax > 0)) sCh.set('1 (trivial)', 'MGF is infinite for every λ > 0');
      }
      draw();
    },
  });

  // Bernoulli KL
  const klBern = (a, p) => {
    const t1 = a > 0 ? a * Math.log(a / p) : 0, t2 = a < 1 ? (1 - a) * Math.log((1 - a) / (1 - p)) : 0;
    return t1 + t2;
  };
  function binomTwoSided(n, p, eps) {
    const kl = Math.floor(n * (p - eps) + 1e-9), ku = Math.ceil(n * (p + eps) - 1e-9);
    let s = 0;
    const lp = Math.log(p), lq = Math.log(1 - p), lgn = S.lgamma(n + 1);
    const pm = k => Math.exp(lgn - S.lgamma(k + 1) - S.lgamma(n - k + 1) + k * lp + (n - k) * lq);
    for (let k = 0; k <= Math.min(kl, n); k++) s += pm(k);
    for (let k = Math.max(ku, 0); k <= n; k++) if (k > kl) s += pm(k);
    return Math.min(1, s);
  }

  Atlas.lab('inequalities-concentration', {
    title: 'Concentration of a sample proportion',
    blurb: R`For $X_i\iid\text{Bernoulli}(p)$, the exact probability $\P(|\bar X_n - p|\ge\epsilon)$ (black) against Chebyshev, Hoeffding, Bernstein and the Chernoff–KL bound as $n$ grows. Make $p$ small: Hoeffding ignores the small variance, Bernstein and Chernoff exploit it. The tiles convert each bound into the sample size that guarantees failure probability at most $\delta$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pS = UI.slider(controls, { label: 'Success probability $p$', min: 0.01, max: 0.5, step: 0.01, value: 0.1, onChange: draw });
      const eS = UI.slider(controls, { label: 'Tolerance $\\epsilon$', min: 0.01, max: 0.2, step: 0.005, value: 0.05, onChange: draw });
      const dS = UI.slider(controls, { label: 'Failure probability $\\delta$', min: 0.001, max: 0.2, log: true, value: 0.05, fmt: v => S.fmt(v, 3), onChange: draw });
      const nS = UI.slider(controls, { label: 'Largest $n$ shown', min: 100, max: 4000, step: 100, value: 1500, onChange: draw });
      const showN = UI.toggle(controls, { label: 'Normal approximation (not a bound)', value: true, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 320, x: [1, 1500], y: [1e-6, 2], logy: true, xlabel: 'n', ylabel: 'P(|X̄ − p| ≥ ε)  (log scale)' });
      const tiles = ['Exact (first $n$ below $\\delta$)', 'Chebyshev', 'Hoeffding', 'Bernstein', 'Chernoff–KL'].map((l, i) => UI.stat(stats, l, i === 0 ? { accent: 'accent' } : {}));
      function bounds(n, pp, eps) {
        const v = pp * (1 - pp), b = Math.max(pp, 1 - pp);
        const up = pp + eps <= 1 ? Math.exp(-n * klBern(pp + eps, pp)) : 0;
        const lo = pp - eps >= 0 ? Math.exp(-n * klBern(pp - eps, pp)) : 0;
        return {
          cheb: Math.min(1, v / (n * eps * eps)),
          hoef: Math.min(1, 2 * Math.exp(-2 * n * eps * eps)),
          bern: Math.min(1, 2 * Math.exp(-n * eps * eps / (2 * v + 2 * b * eps / 3))),
          kl: Math.min(1, up + lo),
          norm: 2 * (1 - S.cdf.normal(eps * Math.sqrt(n / v))),
        };
      }
      function draw() {
        const pp = pS.get(), eps = eS.get(), del = dS.get(), N = Math.round(nS.get());
        p.setDomain([1, N], [1e-6, 2]).clear();
        const ns = Array.from(new Set(S.linspace(1, N, 140).map(Math.round)));
        const B = ns.map(n => bounds(n, pp, eps));
        const ex = ns.map(n => binomTwoSided(n, pp, eps));
        p.hline(del, { color: 'faint', width: 1 });
        p.text(N, del, 'δ', { anchor: 'end', dy: -4, dx: -4, color: 'muted', size: 11 });
        const series = [['cheb', 'Chebyshev', 's1'], ['hoef', 'Hoeffding', 's2'], ['bern', 'Bernstein', 's3'], ['kl', 'Chernoff–KL', 's4']];
        for (const [k, , c] of series) p.line(ns.map((n, i) => [n, clampP(B[i][k])]), { color: c, width: 1.8 });
        if (showN.get()) p.line(ns.map((n, i) => [n, clampP(B[i].norm)]), { color: 's5', width: 1.4, dash: true });
        p.line(ns.map((n, i) => [n, clampP(ex[i])]), { color: 'ink', width: 2.2 });
        p.legend([{ label: 'exact (binomial)', color: 'ink' }, ...series.map(([, l, c]) => ({ label: l, color: c })), ...(showN.get() ? [{ label: 'normal approx', color: 's5', dash: true }] : [])]);
        // required sample sizes
        const v = pp * (1 - pp), b = Math.max(pp, 1 - pp);
        const nCheb = Math.ceil(v / (eps * eps * del));
        const nHoef = Math.ceil(Math.log(2 / del) / (2 * eps * eps));
        const nBern = Math.ceil(Math.log(2 / del) * (2 * v + 2 * b * eps / 3) / (eps * eps));
        let lo = 1, hi = 1;
        while (bounds(hi, pp, eps).kl > del && hi < 1e8) hi *= 2;
        while (hi - lo > 1) { const m = (lo + hi) >> 1; if (bounds(m, pp, eps).kl > del) lo = m; else hi = m; }
        const nKL = hi;
        let a = 1, z = Math.max(2, Math.min(nHoef, nKL));
        if (binomTwoSided(z, pp, eps) > del) z = Math.max(z, nHoef);
        while (z - a > 1) { const m = (a + z) >> 1; if (binomTwoSided(m, pp, eps) > del) a = m; else z = m; }
        const nEx = z;
        const rel = m => '×' + S.fmt(m / nEx, 1) + ' the exact n';
        tiles[0].set(String(nEx), 'found by bisection (lattice wiggles)');
        tiles[1].set(String(nCheb), rel(nCheb));
        tiles[2].set(String(nHoef), rel(nHoef));
        tiles[3].set(String(nBern), rel(nBern));
        tiles[4].set(String(nKL), rel(nKL));
      }
      draw();
    },
  });

  /* ====================================================================
   * entropy-kl
   * ================================================================== */
  const trap = (xs, ys) => { let s = 0; for (let i = 1; i < xs.length; i++) s += 0.5 * (ys[i] + ys[i - 1]) * (xs[i] - xs[i - 1]); return s; };
  const lphi = (x, m, s) => -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * Math.log(2 * Math.PI);

  Atlas.lab('entropy-kl-normals', {
    title: 'KL between two normals: the two directions',
    blurb: R`$P = \N(0,1)$ is fixed; move $Q = \N(\mu,\sigma^2)$. Left: the densities. Right: the integrands $p\log(p/q)$ and $q\log(q/p)$, whose areas are the two KL divergences. Shrink $\sigma$ to see $\KL(P\|Q)$ explode while $\KL(Q\|P)$ grows only logarithmically. Pinsker's bound on total variation is shown alongside the exact TV.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mS = UI.slider(controls, { label: 'Mean of $Q$: $\\mu$', min: -3, max: 3, step: 0.05, value: 1, onInput: draw });
      const sS = UI.slider(controls, { label: 'SD of $Q$: $\\sigma$', min: 0.15, max: 4, step: 0.01, value: 0.6, onInput: draw });
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, x: [-6, 6], y: [0, 1], xlabel: 'x', ylabel: 'density', title: 'P (filled) and Q' });
      const pI = new Plot(stage, { width: 420, height: 270, x: [-6, 6], y: [-1, 1], xlabel: 'x', title: 'Integrands of the two KLs' });
      const sF = UI.stat(stats, '$\\KL(P\\|Q)$', { accent: 'accent' });
      const sR = UI.stat(stats, '$\\KL(Q\\|P)$');
      const sTV = UI.stat(stats, 'Total variation');
      const sPin = UI.stat(stats, 'Pinsker $\\sqrt{\\min \\KL/2}$');
      const sH = UI.stat(stats, 'Hellinger$^2$');
      function draw() {
        const mu = mS.get(), sg = sS.get();
        const kf = Math.log(sg) + (1 + mu * mu) / (2 * sg * sg) - 0.5;
        const kr = -Math.log(sg) + (sg * sg + mu * mu) / 2 - 0.5;
        const lo = Math.min(-6, mu - 6 * sg), hi = Math.max(6, mu + 6 * sg);
        const xs = S.linspace(lo, hi, 700);
        const pp = xs.map(x => S.pdf.normal(x)), qq = xs.map(x => S.pdf.normal(x, mu, sg));
        const tv = 0.5 * trap(xs, xs.map((_, i) => Math.abs(pp[i] - qq[i])));
        const hel = 1 - Math.sqrt(2 * sg / (1 + sg * sg)) * Math.exp(-mu * mu / (4 * (1 + sg * sg)));
        const fI = xs.map((x, i) => pp[i] * (lphi(x, 0, 1) - lphi(x, mu, sg)));
        const rI = xs.map((x, i) => qq[i] * (lphi(x, mu, sg) - lphi(x, 0, 1)));
        const ymax = Math.max(0.45, ...qq) * 1.1;
        pD.setDomain([lo, hi], [0, ymax]).clear();
        pD.area(xs.map((x, i) => [x, pp[i]]), { color: 's1', opacity: 0.25 });
        pD.line(xs.map((x, i) => [x, pp[i]]), { color: 's1', width: 2 });
        pD.line(xs.map((x, i) => [x, qq[i]]), { color: 's2', width: 2 });
        pD.legend([{ label: 'P = N(0,1)', color: 's1' }, { label: 'Q', color: 's2' }]);
        const all = fI.concat(rI).filter(Number.isFinite);
        const ihi = Math.min(Math.max(...all), 6), ilo = Math.max(Math.min(...all, -0.05), -1.5);
        pI.setDomain([lo, hi], [ilo * 1.1, ihi * 1.1]).clear();
        pI.hline(0, { color: 'faint', width: 1, dash: false });
        pI.area(xs.map((x, i) => [x, fI[i]]), { color: 's1', opacity: 0.18, base: 0 });
        pI.line(xs.map((x, i) => [x, fI[i]]), { color: 's1', width: 2 });
        pI.area(xs.map((x, i) => [x, rI[i]]), { color: 's2', opacity: 0.18, base: 0 });
        pI.line(xs.map((x, i) => [x, rI[i]]), { color: 's2', width: 2 });
        pI.legend([{ label: 'p log(p/q)', color: 's1' }, { label: 'q log(q/p)', color: 's2' }]);
        sF.set(S.fmt(kf), 'nats');
        sR.set(S.fmt(kr), 'ratio ' + S.fmt(kf / kr, 2));
        sTV.set(S.fmt(tv));
        sPin.set(S.fmt(Math.min(1, Math.sqrt(Math.min(kf, kr) / 2))), 'upper bound on TV');
        sH.set(S.fmt(hel));
      }
      draw();
    },
  });

  Atlas.lab('entropy-kl-fit', {
    title: 'Mass-covering vs. mode-seeking Gaussian fits',
    blurb: R`The target $p$ is a two-component normal mixture. The forward-KL projection (moment matching) and the reverse-KL projection (found numerically from several starts) are drawn over it. Pull the modes apart: forward KL straddles the gap, reverse KL locks onto one mode, and a second local optimum appears. Add your own $q$ to read off both divergences.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dS = UI.slider(controls, { label: 'Separation of modes $d$', min: 0, max: 8, step: 0.1, value: 6, onChange: fit });
      const wS = UI.slider(controls, { label: 'Weight of left mode $w$', min: 0.05, max: 0.95, step: 0.01, value: 0.6, onChange: fit });
      const s2S = UI.slider(controls, { label: 'SD of right mode', min: 0.3, max: 2, step: 0.05, value: 1, onChange: fit });
      UI.divider(controls, 'Your own q');
      const mine = UI.toggle(controls, { label: 'Show my $q$', value: false, onChange: draw });
      const qm = UI.slider(controls, { label: '$\\mu_q$', min: -6, max: 6, step: 0.05, value: 0, onInput: draw });
      const qs = UI.slider(controls, { label: '$\\sigma_q$', min: 0.2, max: 5, step: 0.05, value: 2, onInput: draw });
      const p = new Plot(stage, { width: 640, height: 320, x: [-8, 8], y: [0, 0.5], xlabel: 'x', ylabel: 'density' });
      const sFw = UI.stat(stats, 'Forward: $\\argmin \\KL(p\\|q)$', { accent: 'accent' });
      const sRv = UI.stat(stats, 'Reverse: $\\argmin \\KL(q\\|p)$');
      const sLoc = UI.stat(stats, 'Reverse-KL local optima');
      const sMine = UI.stat(stats, 'My $q$: forward / reverse KL');
      let P, fw, rv;
      const logp = x => {
        const a = Math.log(P.w) + lphi(x, -P.d / 2, 1), b = Math.log(1 - P.w) + lphi(x, P.d / 2, P.s2);
        const m = Math.max(a, b);
        return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
      };
      function kRev(m, s) { // KL(q||p) on q's support
        const xs = S.linspace(m - 8 * s, m + 8 * s, 321);
        return trap(xs, xs.map(x => { const lq = lphi(x, m, s); return Math.exp(lq) * (lq - logp(x)); }));
      }
      function kFwd(m, s) {
        const lo = Math.min(-P.d / 2 - 8, P.d / 2 - 8 * P.s2), hi = Math.max(-P.d / 2 + 8, P.d / 2 + 8 * P.s2);
        const xs = S.linspace(lo, hi, 601);
        return trap(xs, xs.map(x => { const l = logp(x); return Math.exp(l) * (l - lphi(x, m, s)); }));
      }
      function fit() {
        P = { d: dS.get(), w: wS.get(), s2: s2S.get() };
        const m = P.w * (-P.d / 2) + (1 - P.w) * (P.d / 2);
        const v = P.w * (1 + (-P.d / 2 - m) ** 2) + (1 - P.w) * (P.s2 ** 2 + (P.d / 2 - m) ** 2);
        fw = { m, s: Math.sqrt(v), kl: 0 };
        fw.kl = kFwd(fw.m, fw.s);
        const starts = [[-P.d / 2, 0], [P.d / 2, Math.log(P.s2)], [m, Math.log(Math.sqrt(v))]];
        const sols = [];
        const obj = z => kRev(z[0], Math.exp(Math.max(-3, Math.min(2.5, z[1]))));
        for (const st of starts) {
          // Nelder–Mead with restarts: a restart from the end point escapes saddles and premature stalls.
          let r = S.nelderMead(obj, st, { step: 0.4, maxIter: 300, tol: 1e-9 });
          for (let rep = 0; rep < 3; rep++) {
            const r2 = S.nelderMead(obj, [r.x[0] + 0.25, r.x[1]], { step: 0.35, maxIter: 300, tol: 1e-10 });
            const moved = Math.abs(r2.x[0] - r.x[0]) > 0.02 || Math.abs(r2.x[1] - r.x[1]) > 0.02;
            if (r2.fx <= r.fx) r = r2;
            if (!moved) break;
          }
          const sol = { m: r.x[0], s: Math.exp(Math.max(-3, Math.min(2.5, r.x[1]))), kl: r.fx };
          if (!sols.some(o => Math.abs(o.m - sol.m) < 0.15 && Math.abs(o.s - sol.s) < 0.1)) sols.push(sol);
        }
        sols.sort((a, b) => a.kl - b.kl);
        rv = sols;
        draw();
      }
      function draw() {
        const lo = Math.min(-P.d / 2 - 4, fw.m - 3.5 * fw.s), hi = Math.max(P.d / 2 + 4 * P.s2, fw.m + 3.5 * fw.s);
        const xs = S.linspace(lo, hi, 500);
        const pv = xs.map(x => Math.exp(logp(x)));
        const qd = (q) => xs.map(x => [x, S.pdf.normal(x, q.m, q.s)]);
        const ymax = Math.max(...pv, ...rv.map(q => 1 / (q.s * Math.sqrt(2 * Math.PI))), mine.get() ? 1 / (qs.get() * Math.sqrt(2 * Math.PI)) : 0) * 1.12;
        p.setDomain([lo, hi], [0, ymax]).clear();
        p.area(xs.map((x, i) => [x, pv[i]]), { color: 'ink', opacity: 0.12 });
        p.line(xs.map((x, i) => [x, pv[i]]), { color: 'ink', width: 1.5 });
        p.line(qd(fw), { color: 's1', width: 2.4 });
        rv.forEach((q, i) => p.line(qd(q), { color: 's2', width: i === 0 ? 2.4 : 1.6, dash: i === 0 ? false : true }));
        const leg = [{ label: 'target p', color: 'ink', fill: true, opacity: 0.2 }, { label: 'forward KL fit', color: 's1' }, { label: 'reverse KL (global)', color: 's2' }];
        if (rv.length > 1) leg.push({ label: 'reverse KL (local)', color: 's2', dash: true });
        if (mine.get()) {
          const q = { m: qm.get(), s: qs.get() };
          p.line(qd(q), { color: 's3', width: 2 });
          leg.push({ label: 'my q', color: 's3' });
          sMine.set(S.fmt(kFwd(q.m, q.s)) + ' / ' + S.fmt(kRev(q.m, q.s)));
        } else sMine.set('—', 'toggle "Show my q"');
        p.legend(leg);
        sFw.set(`μ=${S.fmt(fw.m, 2)}, σ=${S.fmt(fw.s, 2)}`, `KL(p‖q) = ${S.fmt(fw.kl)}; KL(q‖p) = ${S.fmt(kRev(fw.m, fw.s))}`);
        sRv.set(`μ=${S.fmt(rv[0].m, 2)}, σ=${S.fmt(rv[0].s, 2)}`, `KL(q‖p) = ${S.fmt(rv[0].kl)}; KL(p‖q) = ${S.fmt(kFwd(rv[0].m, rv[0].s))}`);
        sLoc.set(String(rv.length), rv.length > 1 ? rv.slice(1).map(q => `μ=${S.fmt(q.m, 2)} (KL ${S.fmt(q.kl, 3)})`).join('; ') : 'unique from these starts');
      }
      fit();
    },
  });

  /* ====================================================================
   * convergence
   * ================================================================== */
  const SEQS = {
    typewriter: { label: 'Typewriter indicators', usesEps: false, verdict: 'in prob ✓ · L¹ ✓ · a.s. ✗' },
    bern1: { label: 'Independent Bernoulli(1/n)', usesEps: false, verdict: 'in prob ✓ · L¹ ✓ · a.s. ✗' },
    bern2: { label: 'Independent Bernoulli(1/n²)', usesEps: false, verdict: 'in prob ✓ · L¹ ✓ · a.s. ✓' },
    spike: { label: 'n·1{U < 1/n}', usesEps: false, verdict: 'in prob ✓ · a.s. ✓ · L¹ ✗ (E Xₙ = 1)' },
    mean: { label: 'Running mean of ±1 coin flips', usesEps: true, verdict: 'in prob ✓ · L¹ ✓ · a.s. ✓ (SLLN)' },
  };
  Atlas.lab('convergence-paths', {
    title: 'Rows vs. columns: almost sure vs. in probability',
    blurb: R`Top: each row is one sample path $\omega$; a mark at time $n$ means $|X_n(\omega)|>\epsilon$ (log time axis). Bottom: $\P(|X_n|>\epsilon)$ (a column average), $\P(\sup_{m\ge n}|X_m|>\epsilon)$ (rows still active after $n$), and $\E|X_n|$. Convergence in probability needs the first curve to vanish; almost sure convergence needs the second to vanish.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const seq = UI.select(controls, { label: 'Sequence $X_n$', options: Object.keys(SEQS).map(k => ({ value: k, label: SEQS[k].label })), value: 'bern1', onChange: run });
      const eS = UI.slider(controls, { label: 'Tolerance $\\epsilon$ (running mean only)', min: 0.02, max: 0.5, step: 0.01, value: 0.1, onChange: run });
      const NS = UI.slider(controls, { label: 'Horizon $N$', min: 200, max: 20000, log: true, value: 4000, fmt: v => String(Math.round(v)), onChange: run });
      UI.button(controls, 'New paths', run, { primary: true });
      const ROWS = 40, M = 400;
      const pR = new Plot(stage, { width: 640, height: 240, x: [1, 4000], y: [0, ROWS + 1], logx: true, yticks: 0, xlabel: 'n', title: `${ROWS} sample paths: mark where |Xₙ| > ε` });
      const pC = new Plot(stage, { width: 640, height: 240, x: [1, 4000], y: [0, 1.05], logx: true, xlabel: 'n', ylabel: 'probability' });
      const sP = UI.stat(stats, '$\\P(|X_n|>\\epsilon)$ at $n=N/10$', { accent: 'accent' });
      const sA = UI.stat(stats, '$\\P(\\sup_{N\\ge m\\ge n}|X_m|>\\epsilon)$ at $n=N/10$');
      const sE = UI.stat(stats, '$\\E|X_n|$ at $n=N/10$');
      const sV = UI.stat(stats, 'Theory');
      function run() {
        const kind = seq.get(), N = Math.round(NS.get()), eps = SEQS[kind].usesEps ? eS.get() : 0.5, r = ctx.rng;
        const col = new Float64Array(N + 1), absSum = new Float64Array(N + 1), lastHit = new Int32Array(M);
        const marks = [];
        for (let i = 0; i < M; i++) {
          const U = r.u();
          let s = 0, last = 0;
          for (let n = 1; n <= N; n++) {
            let x;
            if (kind === 'typewriter') { const k = Math.floor(Math.log2(n)), j = n - 2 ** k; x = U >= j / 2 ** k && U < (j + 1) / 2 ** k ? 1 : 0; }
            else if (kind === 'bern1') x = r.u() < 1 / n ? 1 : 0;
            else if (kind === 'bern2') x = r.u() < 1 / (n * n) ? 1 : 0;
            else if (kind === 'spike') x = U < 1 / n ? n : 0;
            else { s += r.u() < 0.5 ? 1 : -1; x = s / n; }
            const ax = Math.abs(x);
            absSum[n] += ax;
            if (ax > eps) { col[n]++; last = n; if (i < ROWS) marks.push([n, i + 1]); }
          }
          lastHit[i] = last;
        }
        // P(sup_{m>=n} |X_m| > eps) = fraction of paths whose last hit is >= n
        const cnt = new Float64Array(N + 2);
        for (let i = 0; i < M; i++) cnt[lastHit[i]]++;
        const tail = new Float64Array(N + 2);
        for (let n = N; n >= 1; n--) tail[n] = tail[n + 1] + cnt[n];
        const grid = Array.from(new Set(S.linspace(0, Math.log10(N), 220).map(t => Math.round(10 ** t)))).filter(n => n >= 1 && n <= N);
        pR.setDomain([1, N], [0, ROWS + 1]).clear();
        pR.points(marks.map(([n, i]) => [n, i]), { color: 's1', r: 1.6, opacity: 0.85 });
        pC.setDomain([1, N], [0, 1.05]).clear();
        // smooth the column frequency a little in log-time for readability
        const colS = grid.map(n => { const a = Math.max(1, Math.floor(n / 1.08)), b = Math.min(N, Math.ceil(n * 1.08)); let s = 0; for (let m = a; m <= b; m++) s += col[m]; return s / ((b - a + 1) * M); });
        const absS = grid.map(n => { const a = Math.max(1, Math.floor(n / 1.08)), b = Math.min(N, Math.ceil(n * 1.08)); let s = 0; for (let m = a; m <= b; m++) s += absSum[m]; return s / ((b - a + 1) * M); });
        pC.line(grid.map((n, i) => [n, colS[i]]), { color: 's1', width: 2 });
        pC.line(grid.map(n => [n, tail[n] / M]), { color: 's2', width: 2.2 });
        pC.line(grid.map((n, i) => [n, Math.min(1.04, absS[i])]), { color: 's3', width: 1.6, dash: true });
        if (kind === 'bern1') pC.line(grid.map(n => [n, 1 - (n - 1) / N]), { color: 'muted', width: 1, dash: '2 3' });
        pC.legend([{ label: 'P(|Xₙ| > ε)', color: 's1' }, { label: 'P(sup over m ≥ n)', color: 's2' }, { label: 'E|Xₙ| (capped at 1)', color: 's3', dash: true }]);
        const n0 = Math.max(1, Math.round(N / 10));
        let c = 0, a = 0; for (let m = Math.max(1, Math.floor(n0 / 1.08)); m <= Math.min(N, Math.ceil(n0 * 1.08)); m++) { c += col[m]; a += absSum[m]; }
        const w = Math.min(N, Math.ceil(n0 * 1.08)) - Math.max(1, Math.floor(n0 / 1.08)) + 1;
        sP.set(S.fmt(c / (w * M)), `from ${M} paths`);
        sA.set(S.fmt(tail[n0] / M), kind === 'bern1' ? 'truncated at N; exact value 1 − (n−1)/N → 1' : 'truncated at horizon N');
        sE.set(S.fmt(a / (w * M)));
        sV.set(SEQS[kind].verdict);
      }
      run();
    },
  });

  /* ====================================================================
   * lln
   * ================================================================== */
  const LLN_SRC = {
    bern: { label: 'Bernoulli(1/2)', draw: r => (r.u() < 0.5 ? 1 : 0), mu: 0.5, sd: 0.5, y: [0, 1] },
    exp: { label: 'Exponential(1)', draw: r => r.exp(1), mu: 1, sd: 1, y: [0, 3] },
    pareto: { label: 'Pareto(α), xₘ = 1', pareto: true },
    cauchy: { label: 'Cauchy(0, 1)', draw: r => r.cauchy(), mu: 0, sd: Infinity, y: [-5, 5], centre: 'median 0, no mean' },
    shock: { label: 'Common shock Z + εᵢ', shock: true, mu: 0, sd: Math.SQRT2, y: [-3.5, 3.5] },
  };
  Atlas.lab('lln-paths', {
    title: 'Running means: many paths at once',
    blurb: R`Each curve is the running mean $\bar X_n$ of one infinite sequence, on a log time axis. The shaded band is $\mu\pm\epsilon$; dashed lines are the CLT scale $\mu\pm1.96\sigma/\sqrt n$ and dotted lines the law-of-the-iterated-logarithm envelope. Try the Cauchy (no mean), Pareto with $\alpha$ near or below $1$, and the common-shock model where every path converges — to a different limit.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const src = UI.select(controls, { label: 'Source distribution', options: Object.keys(LLN_SRC).map(k => ({ value: k, label: LLN_SRC[k].label })), value: 'exp', onChange: run });
      const aS = UI.slider(controls, { label: 'Pareto tail index $\\alpha$', min: 0.7, max: 4, step: 0.05, value: 1.5, onChange: run });
      const eS = UI.slider(controls, { label: 'Band half-width $\\epsilon$', min: 0.02, max: 0.5, step: 0.01, value: 0.1, onInput: draw });
      const KS = UI.slider(controls, { label: 'Paths', min: 5, max: 100, step: 5, value: 40, onChange: run });
      const NS = UI.slider(controls, { label: 'Horizon $N$', min: 100, max: 50000, log: true, value: 5000, fmt: v => String(Math.round(v)), onChange: run });
      const env = UI.toggle(controls, { label: 'CLT and LIL envelopes', value: true, onChange: draw });
      UI.button(controls, 'New paths', run, { primary: true });
      const p = new Plot(stage, { width: 640, height: 330, x: [1, 5000], y: [0, 3], logx: true, xlabel: 'n (log scale)', ylabel: 'running mean' });
      const sIn = UI.stat(stats, 'Outside $\\mu\\pm\\epsilon$ at $n=N$', { accent: 'accent' });
      const sEv = UI.stat(stats, 'Left the band during $[N/10, N]$');
      const sSd = UI.stat(stats, 'Spread of $\\bar X_N$ across paths');
      const sMu = UI.stat(stats, 'Target $\\mu$');
      let paths = [], grid = [], info = {};
      function source() {
        const k = src.get(), D = LLN_SRC[k];
        if (D.pareto) {
          const a = aS.get();
          const mu = a > 1 ? a / (a - 1) : Infinity, sd = a > 2 ? Math.sqrt(a / ((a - 1) ** 2 * (a - 2))) : Infinity;
          return { draw: r => r.pareto(a, 1), mu, sd, y: a > 1 ? [0.9, Math.max(3, mu * 2.2)] : [0.9, 60], label: `Pareto(${S.fmt(a, 2)})` };
        }
        return D;
      }
      function run() {
        const D = source(), N = Math.round(NS.get()), K = Math.round(KS.get()), r = ctx.rng;
        grid = Array.from(new Set(S.linspace(0, Math.log10(N), 260).map(t => Math.round(10 ** t)))).filter(n => n >= 1 && n <= N);
        paths = [];
        const lo = Math.max(1, Math.floor(N / 10));
        info = { D, N, finals: [], maxDevLate: [] };
        for (let k = 0; k < K; k++) {
          const Z = D.shock ? r.normal() : 0;
          let s = 0, gi = 0, dev = 0;
          const pts = [];
          for (let n = 1; n <= N; n++) {
            s += D.shock ? Z + r.normal() : D.draw(r);
            const m = s / n;
            if (n >= lo && Number.isFinite(D.mu)) dev = Math.max(dev, Math.abs(m - D.mu));
            if (n === grid[gi]) { pts.push([n, m]); gi++; }
          }
          paths.push(pts);
          info.finals.push(s / N);
          info.maxDevLate.push(dev);
        }
        draw();
      }
      function draw() {
        const { D, N, finals, maxDevLate } = info, eps = eS.get();
        p.setDomain([1, N], D.y).clear();
        if (Number.isFinite(D.mu)) {
          p.polygon([[1, D.mu - eps], [N, D.mu - eps], [N, D.mu + eps], [1, D.mu + eps]], { color: 'accent', opacity: 0.12 });
          p.hline(D.mu, { color: 'ink', width: 1.2, dash: false });
        }
        paths.forEach(pts => p.line(pts, { color: 's1', width: 1, opacity: 0.45 }));
        if (env.get() && Number.isFinite(D.sd) && Number.isFinite(D.mu)) {
          for (const sg of [1, -1]) {
            p.line(grid.map(n => [n, D.mu + sg * 1.96 * D.sd / Math.sqrt(n)]), { color: 's2', width: 1.6, dash: true });
            p.line(grid.filter(n => n >= 3).map(n => [n, D.mu + sg * D.sd * Math.sqrt(2 * Math.log(Math.log(n)) / n)]), { color: 's3', width: 1.6, dash: '2 3' });
          }
          p.legend([{ label: 'paths', color: 's1' }, { label: '±1.96σ/√n', color: 's2', dash: true }, { label: 'LIL envelope', color: 's3', dash: true }]);
        }
        const K = finals.length;
        if (Number.isFinite(D.mu)) {
          const out = finals.filter(m => Math.abs(m - D.mu) > eps).length;
          const ev = maxDevLate.filter(d => d > eps).length;
          sIn.set(`${out} / ${K}`, Number.isFinite(D.sd) ? `Chebyshev bound: ${S.fmt(Math.min(1, D.sd * D.sd / (N * eps * eps)) * K, 1)} paths` : 'no variance: no Chebyshev bound');
          sEv.set(`${ev} / ${K}`, 'a.s. convergence: this → 0 as N grows');
          sMu.set(S.fmt(D.mu), D.shock ? 'but each path tends to its own Z' : '');
        } else {
          sIn.set('—', D.centre || 'mean is infinite');
          sEv.set('—');
          sMu.set(D.centre ? 'undefined' : '∞', D.centre || 'E X = ∞: the averages drift upward');
        }
        const sorted = finals.slice().sort((a, b) => a - b);
        const iqr = S.quantile(sorted, 0.75, true) - S.quantile(sorted, 0.25, true);
        sSd.set('IQR ' + S.fmt(iqr), Number.isFinite(D.sd) ? `σ/√N × 1.35 = ${S.fmt(1.349 * D.sd / Math.sqrt(N))}` : (src.get() === 'cauchy' ? 'Cauchy IQR is 2 for every n' : ''));
      }
      run();
    },
  });

  /* ====================================================================
   * clt
   * ================================================================== */
  const BE_C = 0.4748;
  const lnRho = (() => { // E|X - mu|^3 for Lognormal(0,1) by quadrature in z
    const mu = Math.exp(0.5), zs = S.linspace(-9, 9, 6001);
    return trap(zs, zs.map(z => Math.abs(Math.exp(z) - mu) ** 3 * S.pdf.normal(z)));
  })();
  const CLT_SRC = {
    unif: { label: 'Uniform(0, 1)', draw: r => r.u(), mu: 0.5, sd: Math.sqrt(1 / 12), rho: 1 / 32, gamma: 0 },
    exp: { label: 'Exponential(1)', draw: r => r.exp(1), mu: 1, sd: 1, rho: 12 / Math.E - 2, gamma: 2 },
    bern: { label: 'Bernoulli(0.1)', draw: r => (r.u() < 0.1 ? 1 : 0), mu: 0.1, sd: 0.3, rho: 0.09 * 0.82, gamma: 0.8 / 0.3, lattice: true },
    lnorm: { label: 'Lognormal(0, 1)', draw: r => r.lognormal(0, 1), mu: Math.exp(0.5), sd: Math.sqrt((Math.E - 1) * Math.E), rho: lnRho, gamma: (Math.E + 2) * Math.sqrt(Math.E - 1) },
    p25: { label: 'Pareto(α = 2.5): no 3rd moment', draw: r => r.pareto(2.5, 1), mu: 2.5 / 1.5, sd: Math.sqrt(2.5 / (2.25 * 0.5)), rho: Infinity, gamma: Infinity },
    p15: { label: 'Pareto(α = 1.5): infinite variance', draw: r => r.pareto(1.5, 1), mu: 3, sd: Infinity, rho: Infinity, gamma: Infinity },
    cauchy: { label: 'Cauchy: no mean', draw: r => r.cauchy(), mu: NaN, sd: Infinity, rho: Infinity, gamma: Infinity },
  };
  const N_STEPS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300];

  Atlas.lab('clt-sums', {
    title: 'Standardised sums from skewed and heavy-tailed sources',
    blurb: R`Simulate $4000$ sums of $n$ draws, standardise them by the true $\mu$ and $\sigma$, and compare with $\N(0,1)$: histogram (left) and normal QQ plot (right). The tiles give the Kolmogorov distance to $\Phi$, the Berry–Esseen guarantee and the Edgeworth prediction. Press play to grow $n$. For infinite-variance sources the sums are standardised by their own median and IQR instead — and still refuse to look normal.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const src = UI.select(controls, { label: 'Source distribution', options: Object.keys(CLT_SRC).map(k => ({ value: k, label: CLT_SRC[k].label })), value: 'exp', onChange: run });
      const nS = UI.slider(controls, { label: 'Summands $n$', min: 0, max: N_STEPS.length - 1, step: 1, value: 6, fmt: v => 'n = ' + N_STEPS[Math.round(v)], onChange: run });
      const row = UI.h('div', 'btn-row', controls);
      let acc = 0;
      const loop = ctx.loop(dt => {
        acc += dt;
        if (acc < 0.7) return;
        acc = 0;
        const i = Math.round(nS.get());
        if (i >= N_STEPS.length - 1) return false;
        nS.set(i + 1); run();
      });
      ctx.playButton(row, loop, ['▶ Grow n', '❚❚ Pause']);
      UI.button(row, 'Resimulate', run);
      stage.classList.add('two');
      const pH = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [0, 0.6], xlabel: 'standardised sum', ylabel: 'density', title: 'Histogram vs. N(0,1)' });
      const pQ = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [-4, 4], xlabel: 'normal quantile', ylabel: 'sample quantile', title: 'Normal QQ plot' });
      const sKS = UI.stat(stats, 'Kolmogorov distance $\\sup|F_n-\\Phi|$', { accent: 'accent' });
      const sBE = UI.stat(stats, 'Berry–Esseen bound');
      const sEd = UI.stat(stats, 'Edgeworth: $|\\gamma|\\phi(0)/(6\\sqrt n)$');
      const sSk = UI.stat(stats, 'Skewness of sums');
      const M = 4000;
      function run() {
        const D = CLT_SRC[src.get()], n = N_STEPS[Math.round(nS.get())], r = ctx.rng;
        const z = new Float64Array(M);
        for (let i = 0; i < M; i++) { let s = 0; for (let j = 0; j < n; j++) s += D.draw(r); z[i] = s; }
        let zs = Array.from(z).sort((a, b) => a - b);
        let robust = false;
        if (Number.isFinite(D.sd)) zs = zs.map(s => (s - n * D.mu) / (D.sd * Math.sqrt(n)));
        else { robust = true; const med = S.quantile(zs, 0.5, true), iqr = S.quantile(zs, 0.75, true) - S.quantile(zs, 0.25, true); zs = zs.map(s => (s - med) / (iqr / 1.349)); }
        // KS distance
        let ks = 0;
        for (let i = 0; i < M; i++) { const F = S.cdf.normal(zs[i]); ks = Math.max(ks, Math.abs((i + 1) / M - F), Math.abs(i / M - F)); }
        const m = S.mean(zs), sd = S.sd(zs);
        let sk = 0; for (const v of zs) sk += ((v - m) / sd) ** 3; sk /= M;
        // histogram
        const hist = S.histogram(zs, { bins: D.lattice && n < 40 ? Math.min(60, 8 * Math.ceil(Math.sqrt(n)) + 8) : 48, lo: -4, hi: 4 });
        const ymax = Math.max(0.5, ...hist.map(b => b.density)) * 1.1;
        pH.setDomain([-4, 4], [0, Math.min(ymax, 2.5)]).clear();
        pH.bars(hist, { color: 's1', opacity: 0.45 });
        pH.line(S.linspace(-4, 4, 200).map(x => [x, S.pdf.normal(x)]), { color: 'ink', width: 2 });
        const outside = zs.filter(v => v < -4 || v > 4).length;
        if (outside) pH.text(4, Math.min(ymax, 2.5) * 0.92, `${outside} beyond ±4`, { anchor: 'end', dx: -6, color: 'muted', size: 11 });
        // QQ
        pQ.clear();
        pQ.line([[-4, -4], [4, 4]], { color: 'ink', width: 1.2, dash: true });
        const qq = [];
        for (let k = 0; k < 400; k++) { const pr = (k + 0.5) / 400; qq.push([S.qnorm(pr), zs[Math.min(M - 1, Math.floor(pr * M))]]); }
        pQ.points(qq.filter(([, y]) => y > -4.2 && y < 4.2), { color: 's2', r: 2.4, opacity: 0.8 });
        sKS.set(S.fmt(ks), `Monte Carlo noise floor ≈ ${S.fmt(0.87 / Math.sqrt(M), 3)}` + (robust ? '; robust standardisation' : ''));
        if (Number.isFinite(D.rho)) {
          const be = BE_C * D.rho / (D.sd ** 3 * Math.sqrt(n));
          sBE.set(S.fmt(Math.min(1, be)), `ρ/σ³ = ${S.fmt(D.rho / D.sd ** 3, 2)}`);
          sEd.set(S.fmt(Math.abs(D.gamma) * 0.39894 / (6 * Math.sqrt(n))), D.lattice ? 'lattice: jumps add ≈ 0.4/√(nσ²)' : (D.gamma === 0 ? 'symmetric: error is O(1/n)' : 'leading term of the true error'));
        } else {
          sBE.set('n/a', Number.isFinite(D.sd) ? 'E|X−μ|³ = ∞; CLT holds, no rate' : 'σ = ∞: the CLT itself fails');
          sEd.set('n/a');
        }
        sSk.set(S.fmt(sk, 2), Number.isFinite(D.gamma) ? `theory γ/√n = ${S.fmt(D.gamma / Math.sqrt(n), 2)}` : '');
      }
      run();
    },
  });

  // Regularised lower incomplete gamma with a generous iteration cap (for large shape parameters).
  function gammaPBig(a, x) {
    if (x <= 0) return 0;
    const lead = -x + a * Math.log(x) - S.lgamma(a);
    if (x < a + 1) {
      let ap = a, del = 1 / a, sum = del;
      for (let n = 0; n < 200000; n++) { ap += 1; del *= x / ap; sum += del; if (Math.abs(del) < Math.abs(sum) * 1e-15) break; }
      return sum * Math.exp(lead);
    }
    const TINY = 1e-300;
    let b = x + 1 - a, c = 1 / TINY, d = 1 / b, h = d;
    for (let i = 1; i < 200000; i++) {
      const an = -i * (i - a); b += 2;
      d = an * d + b; if (Math.abs(d) < TINY) d = TINY;
      c = b + an / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d; const del = d * c; h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return 1 - Math.exp(lead) * h;
  }
  const edgeworth = (z, g) => S.cdf.normal(z) - g * (z * z - 1) * S.pdf.normal(z) / 6;

  Atlas.lab('clt-berry-esseen', {
    title: 'The rate of the CLT, computed exactly',
    blurb: R`Log-log plot of the exact Kolmogorov distance between the law of $Z_n$ and $\Phi$ (solid), between $Z_n$ and the one-term Edgeworth approximation (dotted), and the Berry–Esseen bound (dashed). For gamma summands the sum is gamma, for Bernoulli summands it is binomial, so nothing here is simulated. Non-lattice: Edgeworth improves the rate from $n^{-1/2}$ to $n^{-1}$. Lattice: it cannot.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fam = UI.segmented(controls, { label: 'Summands', options: [{ value: 'gamma', label: 'Gamma(k) — continuous' }, { value: 'bern', label: 'Bernoulli(p) — lattice' }], value: 'gamma', onChange: draw });
      const kS = UI.slider(controls, { label: 'Gamma shape $k$ (skewness $2/\\sqrt k$)', min: 0.1, max: 10, log: true, value: 1, fmt: v => S.fmt(v, 2), onChange: draw });
      const pS = UI.slider(controls, { label: 'Bernoulli $p$', min: 0.01, max: 0.5, step: 0.01, value: 0.2, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 330, x: [1, 3000], y: [1e-5, 1], logx: true, logy: true, xlabel: 'n (log scale)', ylabel: 'sup distance (log scale)' });
      const sSl = UI.stat(stats, 'Slope vs. $\\Phi$ (last decade)', { accent: 'accent' });
      const sSe = UI.stat(stats, 'Slope vs. Edgeworth');
      const sRa = UI.stat(stats, 'Berry–Esseen / exact at largest $n$');
      const sG = UI.stat(stats, 'Skewness $\\gamma$, $\\rho/\\sigma^3$');
      const NS = Array.from(new Set(S.linspace(0, Math.log10(3000), 22).map(t => Math.round(10 ** t))));
      const zg = S.linspace(-5, 5, 241);
      function gammaCase(k) {
        const g = 2 / Math.sqrt(k);
        // rho = E|X-k|^3 for Gamma(k,1), exactly via incomplete gamma functions
        const P = (a) => S.gammaP(a, k);
        const lowPart = k ** 3 * P(k) - 3 * k * k * k * P(k + 1) + 3 * k * k * (k + 1) * P(k + 2) - k * (k + 1) * (k + 2) * P(k + 3);
        const rho = 2 * k + 2 * lowPart;
        const out = NS.map(n => {
          const a = n * k, s = Math.sqrt(a);
          let d0 = 0, d1 = 0;
          for (const z of zg) {
            const F = gammaPBig(a, a + z * s);
            d0 = Math.max(d0, Math.abs(F - S.cdf.normal(z)));
            d1 = Math.max(d1, Math.abs(F - edgeworth(z, g / Math.sqrt(n))));
          }
          return { n, d0, d1 };
        });
        return { out, g, rs: rho / k ** 1.5 };
      }
      function bernCase(pp) {
        const v = pp * (1 - pp), g = (1 - 2 * pp) / Math.sqrt(v), rs = (pp * pp + (1 - pp) ** 2) / Math.sqrt(v);
        const out = NS.map(n => {
          const sd = Math.sqrt(n * v), mean = n * pp;
          const k0 = Math.max(0, Math.floor(mean - 40 * sd - 2)), k1 = Math.min(n, Math.ceil(mean + 40 * sd + 2));
          let F = 0, d0 = 0, d1 = 0;
          const lgn = S.lgamma(n + 1), lp = Math.log(pp), lq = Math.log(1 - pp);
          for (let k = k0; k <= k1; k++) {
            const z = (k - mean) / sd, Phi = S.cdf.normal(z), E = edgeworth(z, g / Math.sqrt(n));
            const Fl = F;
            F += Math.exp(lgn - S.lgamma(k + 1) - S.lgamma(n - k + 1) + k * lp + (n - k) * lq);
            d0 = Math.max(d0, Math.abs(Fl - Phi), Math.abs(F - Phi));
            d1 = Math.max(d1, Math.abs(Fl - E), Math.abs(F - E));
          }
          return { n, d0, d1 };
        });
        return { out, g, rs };
      }
      function slope(arr, key) {
        const m = arr.length, a = arr[m - 8] || arr[0], b = arr[m - 1];
        return Math.log(b[key] / a[key]) / Math.log(b.n / a.n);
      }
      function draw() {
        const isG = fam.get() === 'gamma';
        const { out, g, rs } = isG ? gammaCase(kS.get()) : bernCase(pS.get());
        p.clear();
        const pos = v => (v > 1e-7 ? v : null);
        p.line(out.map(o => [o.n, Math.min(1, BE_C * rs / Math.sqrt(o.n))]), { color: 's3', width: 1.8, dash: true });
        p.line(out.map(o => [o.n, pos(o.d0)]), { color: 's1', width: 2.4 });
        p.points(out.map(o => [o.n, pos(o.d0)]), { color: 's1', r: 2.5 });
        p.line(out.map(o => [o.n, pos(o.d1)]), { color: 's2', width: 2, dash: '2 3' });
        p.points(out.map(o => [o.n, pos(o.d1)]), { color: 's2', r: 2.5 });
        p.legend([{ label: 'exact vs. Φ', color: 's1' }, { label: 'exact vs. Edgeworth', color: 's2', dash: true }, { label: 'Berry–Esseen bound', color: 's3', dash: true }], { left: false });
        sSl.set(S.fmt(slope(out, 'd0'), 2), 'theory: −0.5 (or −1 if γ = 0, non-lattice)');
        sSe.set(S.fmt(slope(out, 'd1'), 2), isG ? 'theory: −1 (non-lattice)' : 'theory: −0.5 (lattice jumps)');
        const last = out[out.length - 1];
        sRa.set('×' + S.fmt(BE_C * rs / Math.sqrt(last.n) / last.d0, 1), `n = ${last.n}`);
        sG.set(`${S.fmt(g, 2)}, ${S.fmt(rs, 2)}`);
      }
      draw();
    },
  });

  /* ====================================================================
   * delta-method
   * ================================================================== */
  const DM_SRC = {
    normal: { label: 'Normal(μ, 1)', mu: t => -2 + 5 * t, sd: () => 1, xbar: (r, mu, n) => mu + r.normal() / Math.sqrt(n) },
    exp: { label: 'Exponential(mean μ)', mu: t => 0.1 * 50 ** t, sd: m => m, xbar: (r, mu, n) => r.gamma(n, mu / n) },
    bern: { label: 'Bernoulli(μ)', mu: t => 0.01 + 0.98 * t, sd: m => Math.sqrt(Math.max(0, m * (1 - m))), xbar: (r, mu, n) => r.binomial(n, mu) / n },
    pois: { label: 'Poisson(μ)', mu: t => 0.2 * 100 ** t, sd: m => Math.sqrt(Math.max(0, m)), xbar: (r, mu, n) => r.poisson(n * mu) / n },
  };
  const DM_G = {
    sq: { label: 'x²', g: x => x * x, d1: x => 2 * x, d2: () => 2 },
    inv: { label: '1/x', g: x => 1 / x, d1: x => -1 / (x * x), d2: x => 2 / x ** 3 },
    log: { label: 'log x', g: x => Math.log(x), d1: x => 1 / x, d2: x => -1 / (x * x) },
    sqrt: { label: '√x', g: x => Math.sqrt(x), d1: x => 0.5 / Math.sqrt(x), d2: x => -0.25 / x ** 1.5 },
    logit: { label: 'logit x', g: x => Math.log(x / (1 - x)), d1: x => 1 / (x * (1 - x)), d2: x => (2 * x - 1) / (x * x * (1 - x) ** 2) },
    var: { label: 'x(1−x)', g: x => x * (1 - x), d1: x => 1 - 2 * x, d2: () => -2 },
    exp: { label: 'eˣ', g: x => Math.exp(x), d1: x => Math.exp(x), d2: x => Math.exp(x) },
  };
  // density of a + b Z + c Z^2, Z ~ N(0,1)
  function quadNormalPdf(y, a, b, c) {
    if (Math.abs(c) < 1e-4 * Math.abs(b)) return b !== 0 ? S.pdf.normal(y, a, Math.abs(b)) : 0;
    const dl = b / (2 * c), w = (y - a + b * b / (4 * c)) / c;
    if (w <= 0) return 0;
    const rw = Math.sqrt(w);
    return (S.pdf.normal(rw - dl) + S.pdf.normal(rw + dl)) / (2 * rw * Math.abs(c));
  }

  Atlas.lab('delta-method-lab', {
    title: 'The delta method, first and second order',
    blurb: R`Simulate $g(\bar X_n)$ from $5000$ samples and compare with the first-order normal approximation $\N\big(g(\mu),g'(\mu)^2\sigma^2/n\big)$ (dashed) and the second-order law of $g(\mu)+g'(\mu)\sigma Z/\sqrt n+\tfrac12g''(\mu)\sigma^2Z^2/n$ (solid). Try $x(1-x)$ with a Bernoulli near $\mu=1/2$, $x^2$ with a normal near $\mu=0$, or $1/x$ and $\log x$ near the boundary. The coverage tile shows what this does to delta-method Wald intervals.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const src = UI.select(controls, { label: 'Data', options: Object.keys(DM_SRC).map(k => ({ value: k, label: DM_SRC[k].label })), value: 'bern', onChange: () => { muS.set(muS.get()); run(); } });
      const gS = UI.select(controls, { label: 'Function $g$', options: Object.keys(DM_G).map(k => ({ value: k, label: DM_G[k].label })), value: 'var', onChange: run });
      const muS = UI.slider(controls, { label: 'True mean $\\mu$', min: 0, max: 1, step: 0.005, value: 0.42, fmt: v => 'μ = ' + S.fmt(DM_SRC[src.get()].mu(v), 3), onChange: run });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 1000, log: true, value: 100, fmt: v => String(Math.round(v)), onChange: run });
      UI.button(controls, 'Resimulate', run, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 1], y: [0, 1], xlabel: 'g(X̄)', ylabel: 'density' });
      const sSd = UI.stat(stats, 'SD of $g(\\bar X)$: simulated vs. delta', { accent: 'accent' });
      const sBi = UI.stat(stats, 'Bias: simulated vs. $g\'\'\\sigma^2/(2n)$');
      const sCo = UI.stat(stats, 'Coverage of 95% Wald interval');
      const sRa = UI.stat(stats, 'Curvature ratio $|g\'\'|\\sigma/(\\sqrt n|g\'|)$');
      const M = 5000;
      function run() {
        const D = DM_SRC[src.get()], G = DM_G[gS.get()], n = Math.round(nS.get()), r = ctx.rng;
        const mu = D.mu(muS.get()), sig = D.sd(mu), th = G.g(mu), d1 = G.d1(mu), d2 = G.d2(mu);
        const vals = [];
        let undef = 0, cover = 0;
        for (let i = 0; i < M; i++) {
          const xb = D.xbar(r, mu, n), v = G.g(xb);
          if (!Number.isFinite(v)) { undef++; continue; }
          vals.push(v);
          const se = Math.abs(G.d1(xb)) * D.sd(xb) / Math.sqrt(n);
          if (Number.isFinite(se) && Math.abs(v - th) <= 1.96 * se) cover++;
        }
        p.clear();
        if (!Number.isFinite(th)) { sSd.set('—', 'g(μ) undefined here'); sBi.set('—'); sCo.set('—'); sRa.set('—'); return; }
        const sorted = vals.slice().sort((a, b) => a - b);
        let lo = S.quantile(sorted, 0.005, true), hi = S.quantile(sorted, 0.995, true);
        lo = Math.min(lo, th); hi = Math.max(hi, th);
        const sdD = Math.abs(d1) * sig / Math.sqrt(n);
        if (sdD > 0) { lo = Math.min(lo, th - 3.5 * sdD); hi = Math.max(hi, th + 3.5 * sdD); }
        const pad = (hi - lo) * 0.06 || Math.abs(th) * 0.1 || 0.1; lo -= pad; hi += pad;
        const distinct = new Set(sorted.slice(0, 400)).size;
        const hist = S.histogram(vals, { bins: 60, lo, hi });
        const hmax = Math.max(...hist.map(b => b.density));
        const xs = S.linspace(lo, hi, 400);
        const n1 = sdD > 0 ? xs.map(x => S.pdf.normal(x, th, sdD)) : xs.map(() => null);
        const n2 = xs.map(x => quadNormalPdf(x, th, d1 * sig / Math.sqrt(n), d2 * sig * sig / (2 * n)));
        const ymax = Math.max(hmax, Math.max(...n1.filter(Number.isFinite), 0)) * 1.15 || 1;
        p.setDomain([lo, hi], [0, ymax]).clear();
        p.bars(hist, { color: 's1', opacity: 0.4 });
        p.line(xs.map((x, i) => [x, n2[i] != null ? Math.min(n2[i], ymax * 3) : null]), { color: 's2', width: 2.2 });
        if (sdD > 0) p.line(xs.map((x, i) => [x, n1[i]]), { color: 's3', width: 2, dash: true });
        p.vline(th, { color: 'ink', width: 1.2 });
        p.text(th, ymax, 'g(μ)', { anchor: 'middle', dy: 12, color: 'muted', size: 11 });
        p.legend([{ label: 'simulated g(X̄)', color: 's1', fill: true }, { label: '2nd-order law', color: 's2' }, { label: '1st-order normal', color: 's3', dash: true }]);
        const m = S.mean(vals), sd = S.sd(vals);
        sSd.set(`${S.fmt(sd)} vs ${S.fmt(sdD)}`, undef ? `${undef} of ${M} samples had g(X̄) undefined` : (distinct < 30 ? 'X̄ is discrete here: few distinct values' : ''));
        sBi.set(`${S.fmt(m - th)} vs ${S.fmt(d2 * sig * sig / (2 * n))}`);
        const cov = cover / M;
        sCo.set((100 * cov).toFixed(1) + '%', cov < 0.9 ? '<span style="color:var(--bad)">well below nominal</span>' : 'nominal 95%');
        const ratioV = Math.abs(d2) * sig / (Math.sqrt(n) * Math.abs(d1));
        sRa.set(Number.isFinite(ratioV) ? S.fmt(ratioV, 2) : '∞', ratioV > 0.3 ? 'large: first order is unreliable' : 'small: linearisation is fine');
      }
      run();
    },
  });

  /* ====================================================================
   * sampling-distributions
   * ================================================================== */
  // Populations with mean mu, sd, skewness g, excess kurtosis kap.
  const SD_POP = {
    normal: { label: 'Normal(0, 1)', draw: r => r.normal(), mu: 0, sd: 1, g: 0, kap: 0 },
    unif: { label: 'Uniform(0, 1)', draw: r => r.u(), mu: 0.5, sd: Math.sqrt(1 / 12), g: 0, kap: -1.2 },
    laplace: { label: 'Laplace(0, 1)', draw: r => r.laplace(0, 1), mu: 0, sd: Math.SQRT2, g: 0, kap: 3 },
    t5: { label: 't₅ (heavy tails)', draw: r => r.t(5), mu: 0, sd: Math.sqrt(5 / 3), g: 0, kap: 6 },
    exp: { label: 'Exponential(1)', draw: r => r.exp(1), mu: 1, sd: 1, g: 2, kap: 6 },
    lnorm: { label: 'Lognormal(0, 0.6)', draw: r => r.lognormal(0, 0.6), mu: Math.exp(0.18), sd: Math.sqrt((Math.exp(0.36) - 1) * Math.exp(0.36)),
      g: (Math.exp(0.36) + 2) * Math.sqrt(Math.exp(0.36) - 1), kap: Math.exp(1.44) + 2 * Math.exp(1.08) + 3 * Math.exp(0.72) - 6 },
  };

  Atlas.lab('sampling-distributions-xbar-s2', {
    title: 'Are the sample mean and variance independent?',
    blurb: R`Each dot is one sample of size $n$: its mean $\bar X$ against its variance $S^2$ (left). Right: the distribution of $(n-1)S^2/\sigma^2$ against the $\chi^2_{n-1}$ density that normal theory promises. Under normality the cloud has no trend and the histogram fits. Switch to exponential or $t_5$ data: the cloud tilts or fans out, and the $\chi^2$ test of the true variance rejects far more than $5\%$ — even for large $n$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population', options: Object.keys(SD_POP).map(k => ({ value: k, label: SD_POP[k].label })), value: 'normal', onChange: run });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 3, max: 80, step: 1, value: 10, onChange: run });
      UI.button(controls, 'Resimulate', run, { primary: true });
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 270, xlabel: 'sample mean X̄', ylabel: 'sample variance S²', title: 'Each dot: one sample' });
      const pH = new Plot(stage, { width: 420, height: 270, xlabel: '(n−1)S²/σ²', ylabel: 'density', title: 'Scaled variance vs. χ²(n−1)' });
      const sC = UI.stat(stats, '$\\Corr(\\bar X, S^2)$', { accent: 'accent' });
      const sV = UI.stat(stats, '$\\Var[(n-1)S^2/\\sigma^2]$');
      const sZ = UI.stat(stats, 'Size of 5% $\\chi^2$ test of true $\\sigma^2$');
      const M = 3000;
      function run() {
        const P = SD_POP[pop.get()], n = Math.round(nS.get()), r = ctx.rng, s2 = P.sd * P.sd;
        const xb = new Float64Array(M), vs = new Float64Array(M);
        const x = new Float64Array(n);
        for (let i = 0; i < M; i++) {
          let s = 0; for (let j = 0; j < n; j++) { x[j] = P.draw(r); s += x[j]; }
          const m = s / n; let q = 0; for (let j = 0; j < n; j++) q += (x[j] - m) ** 2;
          xb[i] = m; vs[i] = q / (n - 1);
        }
        const A = Array.from(xb), B = Array.from(vs);
        const corr = S.cor(A, B);
        const sc = B.map(v => (n - 1) * v / s2);
        const lo = S.q.chisq(0.025, n - 1), hi = S.q.chisq(0.975, n - 1);
        const rej = sc.filter(v => v < lo || v > hi).length / M;
        // scatter
        const xs = A.slice().sort((a, b) => a - b), ys = B.slice().sort((a, b) => a - b);
        const x0 = S.quantile(xs, 0.002, true), x1 = S.quantile(xs, 0.998, true), y1 = S.quantile(ys, 0.995, true);
        pS.setDomain([x0, x1], [0, y1 * 1.05]).clear();
        pS.points(A.map((a, i) => [a, B[i]]), { color: 's1', r: 1.7, opacity: 0.45 });
        pS.hline(s2, { color: 'ink', width: 1.2 });
        pS.vline(P.mu, { color: 'ink', width: 1.2 });
        // histogram
        const top = Math.max(S.quantile(sc.slice().sort((a, b) => a - b), 0.995, true), S.q.chisq(0.999, n - 1));
        const hist = S.histogram(sc, { bins: 50, lo: 0, hi: top });
        const grid = S.linspace(top / 400, top, 300), dens = grid.map(v => S.pdf.chisq(v, n - 1));
        const ymax = Math.max(...hist.map(b => b.density), ...dens.filter(Number.isFinite).slice(2)) * 1.12;
        pH.setDomain([0, top], [0, ymax]).clear();
        pH.band(0, lo, { color: 'bad', opacity: 0.07 });
        pH.band(hi, top, { color: 'bad', opacity: 0.07 });
        pH.bars(hist, { color: 's2', opacity: 0.45 });
        pH.line(grid.map((v, i) => [v, dens[i]]), { color: 'ink', width: 2 });
        pH.legend([{ label: 'simulated', color: 's2', fill: true }, { label: 'χ²(n−1)', color: 'ink' }]);
        const varS2 = s2 * s2 * (2 / (n - 1) + P.kap / n);
        const corrT = P.g * P.sd ** 3 / n / Math.sqrt(s2 / n * varS2);
        sC.set(S.fmt(corr, 3), `theory ${S.fmt(corrT, 3)}`);
        sV.set(S.fmt(S.variance(sc), 1), `χ² says ${2 * (n - 1)}; theory ${S.fmt((n - 1) ** 2 * varS2 / (s2 * s2), 1)}`);
        sZ.set((100 * rej).toFixed(1) + '%', rej > 0.08 ? '<span style="color:var(--bad)">anti-conservative</span>' : rej < 0.025 ? 'conservative' : 'close to nominal');
      }
      run();
    },
  });

  Atlas.lab('sampling-distributions-t', {
    title: 'The t statistic: exact for normal data, approximate otherwise',
    blurb: R`Simulate $T=\sqrt n(\bar X-\mu)/S$ from $6000$ samples. Left: its histogram against $t_{n-1}$ and $\N(0,1)$. Right: both tails on a log scale — the lower tail $\P(T<-x)$ and upper tail $\P(T>x)$ separately — so skewness shows up as two different curves. For normal data with small $n$, the $z$ cut-off $1.96$ is too small; for skewed data, even the correct $t$ cut-off rejects unevenly in the two tails.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population', options: Object.keys(SD_POP).map(k => ({ value: k, label: SD_POP[k].label })), value: 'exp', onChange: run });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 2, max: 60, step: 1, value: 8, onChange: run });
      UI.button(controls, 'Resimulate', run, { primary: true });
      stage.classList.add('two');
      const pH = new Plot(stage, { width: 420, height: 270, x: [-6, 6], y: [0, 0.5], xlabel: 'T', ylabel: 'density', title: 'Distribution of T' });
      const pT = new Plot(stage, { width: 420, height: 270, x: [0, 6], y: [1e-4, 0.6], logy: true, xlabel: 'x', ylabel: 'tail probability', title: 'Tails: P(T > x) and P(T < −x)' });
      const sCrit = UI.stat(stats, '$t_{n-1,\\,0.975}$', { accent: 'accent' });
      const sSz = UI.stat(stats, 'Size of 5% $t$-test');
      const sLR = UI.stat(stats, 'Lower / upper rejections');
      const sZ = UI.stat(stats, 'Size if $1.96$ is used');
      const M = 6000;
      function run() {
        const P = SD_POP[pop.get()], n = Math.round(nS.get()), r = ctx.rng;
        const T = new Float64Array(M), x = new Float64Array(n);
        for (let i = 0; i < M; i++) {
          let s = 0; for (let j = 0; j < n; j++) { x[j] = P.draw(r); s += x[j]; }
          const m = s / n; let q = 0; for (let j = 0; j < n; j++) q += (x[j] - m) ** 2;
          const sd = Math.sqrt(q / (n - 1));
          T[i] = sd > 0 ? Math.sqrt(n) * (m - P.mu) / sd : (m > P.mu ? Infinity : -Infinity);
        }
        const df = n - 1, c = S.q.t(0.975, df);
        let lo = 0, hi = 0, z = 0;
        for (const t of T) { if (t < -c) lo++; if (t > c) hi++; if (Math.abs(t) > 1.959964) z++; }
        const hist = S.histogram(Array.from(T), { bins: 60, lo: -6, hi: 6 });
        const xs = S.linspace(-6, 6, 300);
        const ymax = Math.max(0.42, ...hist.map(b => b.density)) * 1.1;
        pH.setDomain([-6, 6], [0, ymax]).clear();
        pH.bars(hist, { color: 's1', opacity: 0.45 });
        pH.line(xs.map(v => [v, S.pdf.t(v, df)]), { color: 'ink', width: 2 });
        pH.line(xs.map(v => [v, S.pdf.normal(v)]), { color: 's3', width: 1.6, dash: true });
        pH.vline(-c, { color: 'muted', width: 1 }); pH.vline(c, { color: 'muted', width: 1 });
        pH.legend([{ label: 'simulated T', color: 's1', fill: true }, { label: `t(${df})`, color: 'ink' }, { label: 'N(0,1)', color: 's3', dash: true }]);
        // tails
        const sorted = Array.from(T).sort((a, b) => a - b);
        const upper = v => { let a = 0, b = M; while (a < b) { const m = (a + b) >> 1; if (sorted[m] <= v) a = m + 1; else b = m; } return (M - a) / M; };
        const lower = v => { let a = 0, b = M; while (a < b) { const m = (a + b) >> 1; if (sorted[m] < -v) a = m + 1; else b = m; } return a / M; };
        const g = S.linspace(0, 6, 121);
        pT.clear();
        pT.line(g.map(v => [v, clampP(1 - S.cdf.t(v, df))]), { color: 'ink', width: 2 });
        pT.line(g.map(v => [v, clampP(1 - S.cdf.normal(v))]), { color: 's3', width: 1.6, dash: true });
        pT.line(g.map(v => [v, upper(v) > 0 ? upper(v) : null]), { color: 's1', width: 2 });
        pT.line(g.map(v => [v, lower(v) > 0 ? lower(v) : null]), { color: 's2', width: 2 });
        pT.vline(c, { color: 'muted', width: 1 });
        pT.legend([{ label: 'P(T > x) simulated', color: 's1' }, { label: 'P(T < −x) simulated', color: 's2' }, { label: `t(${df}) tail`, color: 'ink' }, { label: 'normal tail', color: 's3', dash: true }]);
        sCrit.set(S.fmt(c, 3), 'vs 1.960 for the normal');
        const size = (lo + hi) / M;
        sSz.set((100 * size).toFixed(1) + '%', `Monte Carlo ± ${(100 * 1.96 * Math.sqrt(0.05 * 0.95 / M)).toFixed(1)}%`);
        sLR.set(`${(100 * lo / M).toFixed(1)}% / ${(100 * hi / M).toFixed(1)}%`, 'nominal 2.5% / 2.5%');
        sZ.set((100 * z / M).toFixed(1) + '%');
      }
      run();
    },
  });

  /* ====================================================================
   * order-statistics
   * ================================================================== */
  const OS_POP = {
    unif: { label: 'Uniform(0, 1)', draw: r => r.u(), pdf: x => S.pdf.uniform(x), cdf: x => S.cdf.uniform(x), q: u => u, range: [0, 1] },
    exp: { label: 'Exponential(1)', draw: r => r.exp(1), pdf: x => S.pdf.exp(x), cdf: x => S.cdf.exp(x), q: u => -Math.log(1 - u), range: null },
    normal: { label: 'Normal(0, 1)', draw: r => r.normal(), pdf: x => S.pdf.normal(x), cdf: x => S.cdf.normal(x), q: u => S.qnorm(u), range: null },
    cauchy: { label: 'Cauchy(0, 1)', draw: r => r.cauchy(), pdf: x => S.pdf.cauchy(x), cdf: x => S.cdf.cauchy(x), q: u => S.q.cauchy(u), range: null, heavy: true },
  };
  function osDensity(P, n, k) {
    const c = S.lgamma(n + 1) - S.lgamma(k) - S.lgamma(n - k + 1);
    return x => {
      const F = P.cdf(x), f = P.pdf(x);
      if (!(f > 0) || F <= 0 || F >= 1) return 0;
      return Math.exp(c + (k - 1) * Math.log(F) + (n - k) * Math.log(1 - F) + Math.log(f));
    };
  }
  function osMoments(P, n, k) { // E and SD of X_(k) via u-substitution with a Beta(k, n-k+1) weight
    const G = 6000, la = S.lbeta(k, n - k + 1);
    let m1 = 0, m2 = 0;
    for (let i = 0; i < G; i++) {
      const u = (i + 0.5) / G, w = Math.exp((k - 1) * Math.log(u) + (n - k) * Math.log(1 - u) - la) / G, x = P.q(u);
      m1 += w * x; m2 += w * x * x;
    }
    return { mean: m1, sd: Math.sqrt(Math.max(0, m2 - m1 * m1)) };
  }

  Atlas.lab('order-statistics-density', {
    title: 'The density of the k-th order statistic',
    blurb: R`Simulate $4000$ samples of size $n$, sort each, and histogram the $k$-th smallest value; the black curve is $f_{(k)}(x) = \frac{n!}{(k-1)!(n-k)!}F^{k-1}(1-F)^{n-k}f$. The dashed curve is the large-sample normal approximation $\N\big(\xi_p,\,p(1-p)/(nf(\xi_p)^2)\big)$ with $p=k/(n+1)$. It is good for central ranks and fails at the extremes $k=1$ and $k=n$ — most dramatically for the Cauchy.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population', options: Object.keys(OS_POP).map(k => ({ value: k, label: OS_POP[k].label })), value: 'exp', onChange: run });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 1, max: 100, step: 1, value: 15, onChange: () => { kS.set(kS.get()); run(); } });
      const kOf = v => Math.max(1, Math.min(Math.round(nS.get()), 1 + Math.round(v * (Math.round(nS.get()) - 1))));
      const kS = UI.slider(controls, { label: 'Rank $k$', min: 0, max: 1, step: 0.005, value: 0.5, fmt: v => 'k = ' + kOf(v), onChange: run });
      const showAll = UI.toggle(controls, { label: 'Faintly show all $n$ order-statistic densities', value: true, onChange: run });
      const showN = UI.toggle(controls, { label: 'Normal approximation', value: true, onChange: run });
      UI.button(controls, 'Resimulate', run, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 1], y: [0, 1], xlabel: 'x', ylabel: 'density' });
      const sK = UI.stat(stats, 'Rank', { accent: 'accent' });
      const sM = UI.stat(stats, 'Mean: simulated vs. formula');
      const sS = UI.stat(stats, 'SD: simulated vs. formula');
      const sA = UI.stat(stats, 'Asymptotic SD $\\sqrt{p(1-p)/n}/f(\\xi_p)$');
      const M = 4000;
      function run() {
        const P = OS_POP[pop.get()], n = Math.round(nS.get()), k = kOf(kS.get()), r = ctx.rng;
        const vals = new Float64Array(M), all = [];
        const x = new Float64Array(n);
        for (let i = 0; i < M; i++) {
          for (let j = 0; j < n; j++) x[j] = P.draw(r);
          x.sort();
          vals[i] = x[k - 1];
        }
        const sorted = Array.from(vals).sort((a, b) => a - b);
        let lo, hi;
        if (P.range) { lo = P.range[0]; hi = P.range[1]; }
        else { lo = S.quantile(sorted, 0.003, true); hi = S.quantile(sorted, 0.997, true); const pad = (hi - lo) * 0.12; lo -= pad; hi += pad; if (pop.get() === 'exp') lo = Math.max(lo, 0); }
        const hist = S.histogram(sorted, { bins: 50, lo, hi });
        const fk = osDensity(P, n, k);
        const xs = S.linspace(lo, hi, 400);
        const dk = xs.map(fk);
        const pk = k / (n + 1), xi = P.q(pk), fxi = P.pdf(xi), asd = Math.sqrt(pk * (1 - pk) / n) / fxi;
        const ymax = Math.max(...hist.map(b => b.density), ...dk.filter(Number.isFinite)) * 1.15;
        p.setDomain([lo, hi], [0, ymax]).clear();
        if (showAll.get() && n <= 40) for (let j = 1; j <= n; j++) if (j !== k) { const fj = osDensity(P, n, j); p.line(xs.map(v => [v, fj(v)]), { color: 'faint', width: 1, opacity: 0.8 }); }
        p.line(xs.map(v => [v, P.pdf(v)]), { color: 'muted', width: 1.2, dash: '2 3' });
        p.bars(hist, { color: 's1', opacity: 0.45 });
        p.line(xs.map((v, i) => [v, dk[i]]), { color: 'ink', width: 2.2 });
        if (showN.get() && n > 1) p.line(xs.map(v => [v, S.pdf.normal(v, xi, asd)]), { color: 's3', width: 1.8, dash: true });
        p.legend([{ label: `simulated X(${k})`, color: 's1', fill: true }, { label: 'exact density', color: 'ink' }, ...(showN.get() && n > 1 ? [{ label: 'normal approx.', color: 's3', dash: true }] : []), { label: 'population f', color: 'muted', dash: true }]);
        const mS = S.mean(sorted), sdS = S.sd(sorted);
        sK.set(`k = ${k} of ${n}`, k === 1 ? 'minimum' : k === n ? 'maximum' : (n % 2 === 1 && k === (n + 1) / 2 ? 'median' : `p ≈ ${S.fmt(pk, 2)}`));
        if (P.heavy && (k <= 1 || k >= n)) { sM.set(S.fmt(mS) + ' vs —', 'E X(k) does not exist for k = 1 or n'); sS.set(S.fmt(sdS) + ' vs —'); }
        else if (P.heavy && (k <= 2 || k >= n - 1)) { const mo = osMoments(P, n, k); sM.set(`${S.fmt(mS)} vs ${S.fmt(mo.mean)}`); sS.set(S.fmt(sdS) + ' vs —', 'variance needs 3 ≤ k ≤ n − 2'); }
        else { const mo = osMoments(P, n, k); sM.set(`${S.fmt(mS)} vs ${S.fmt(mo.mean)}`); sS.set(`${S.fmt(sdS)} vs ${S.fmt(mo.sd)}`); }
        sA.set(n > 1 ? S.fmt(asd) : '—', 'with p = k/(n+1)');
      }
      run();
    },
  });

  const EFF_POP = {
    normal: { label: 'Normal(0, 1)', draw: r => r.normal(), f0: 1 / Math.sqrt(2 * Math.PI), v: 1 },
    laplace: { label: 'Laplace(0, 1)', draw: r => r.laplace(0, 1), f0: 0.5, v: 2 },
    t3: { label: 't₃', draw: r => r.t(3), f0: 2 / (Math.PI * Math.sqrt(3)), v: 3 },
    contam: { label: '0.9·N(0,1) + 0.1·N(0,9)', draw: r => (r.u() < 0.1 ? 3 : 1) * r.normal(), f0: (0.9 + 0.1 / 3) / Math.sqrt(2 * Math.PI), v: 1.8 },
    unif: { label: 'Uniform(−½, ½)', draw: r => r.u() - 0.5, f0: 1, v: 1 / 12 },
    cauchy: { label: 'Cauchy(0, 1)', draw: r => r.cauchy(), f0: 1 / Math.PI, v: Infinity },
  };
  const ESTS = [
    ['Mean', 's1', x => S.mean(x)],
    ['Median', 's2', x => S.quantile(x, 0.5, true)],
    ['10% trimmed', 's3', x => { const k = Math.floor(0.1 * x.length); let s = 0; for (let i = k; i < x.length - k; i++) s += x[i]; return s / (x.length - 2 * k); }],
    ['Midrange', 's4', x => (x[0] + x[x.length - 1]) / 2],
  ];

  Atlas.lab('order-statistics-efficiency', {
    title: 'Median vs. mean (and friends): who wins depends on the tails',
    blurb: R`Draw $3000$ samples of size $n$ from a symmetric population centred at $0$ and compute four location estimators built from order statistics. Left: their sampling distributions. Right: $n\times$ variance, with the asymptotic values $\sigma^2$ (mean) and $1/(4f(0)^2)$ (median) marked. The mean wins for the normal, the median for heavy tails, and the midrange — which uses only the two extremes — for the uniform.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population', options: Object.keys(EFF_POP).map(k => ({ value: k, label: EFF_POP[k].label })), value: 'normal', onChange: run });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 201, step: 2, value: 41, onChange: run });
      UI.button(controls, 'Resimulate', run, { primary: true });
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, xlabel: 'estimate', ylabel: 'density', title: 'Sampling distributions' });
      const pB = new Plot(stage, { width: 420, height: 270, x: [0, 4], y: [0, 1], xticks: 0, ylabel: 'n × variance', title: 'n × Var (lower is better)' });
      const sMe = UI.stat(stats, '$n\\Var$ mean: sim / theory', { accent: 'accent' });
      const sMd = UI.stat(stats, '$n\\Var$ median: sim / $1/(4f(0)^2)$');
      const sAre = UI.stat(stats, 'ARE median : mean');
      const sBest = UI.stat(stats, 'Best of the four');
      const M = 3000;
      function run() {
        const P = EFF_POP[pop.get()], n = Math.round(nS.get()), r = ctx.rng;
        const est = ESTS.map(() => new Float64Array(M));
        const x = new Float64Array(n);
        for (let i = 0; i < M; i++) {
          for (let j = 0; j < n; j++) x[j] = P.draw(r);
          x.sort();
          ESTS.forEach(([, , f], e) => { est[e][i] = f(x); });
        }
        const nv = est.map(a => n * S.variance(Array.from(a)));
        const med = Array.from(est[1]).sort((a, b) => a - b);
        const span = Math.max(4 * S.sd(med), 1e-6);
        pD.setDomain([-span, span], [0, 1]).clear();
        let ymax = 0;
        const curves = est.map(a => {
          const arr = Array.from(a).filter(v => Math.abs(v) < 3 * span);
          const bw = S.silverman(arr) || span / 20;
          const kd = S.kde(arr.length > 1500 ? arr.slice(0, 1500) : arr, bw);
          const xs = S.linspace(-span, span, 160);
          const ys = xs.map(v => kd(v) * arr.length / M);
          ymax = Math.max(ymax, ...ys);
          return xs.map((v, i) => [v, ys[i]]);
        });
        pD.setDomain([-span, span], [0, ymax * 1.12]);
        curves.forEach((c, e) => pD.line(c, { color: ESTS[e][1], width: 2 }));
        pD.legend(ESTS.map(([l, c]) => ({ label: l, color: c })));
        const asyMed = 1 / (4 * P.f0 * P.f0);
        const cap = Math.max(...nv.filter((v, e) => e > 0 && Number.isFinite(v)), Number.isFinite(P.v) ? P.v : 0, asyMed) * 1.3;
        pB.setDomain([0, 4], [0, cap]).clear();
        ESTS.forEach(([l, c], e) => {
          const v = nv[e], h = Math.min(v, cap);
          pB.bars([{ x0: e + 0.15, x1: e + 0.85, y: h }], { color: c, opacity: 0.75 });
          pB.text(e + 0.5, h, v > cap ? S.fmt(v, 1) + ' ↑' : S.fmt(v, 2), { anchor: 'middle', dy: -5, size: 11, weight: 600 });
          pB.text(e + 0.5, 0, l, { anchor: 'middle', dy: 14, size: 11, color: 'muted' });
        });
        if (Number.isFinite(P.v)) pB.segment(0.1, P.v, 0.9, P.v, { color: 'ink', width: 2, dash: true });
        pB.segment(1.1, asyMed, 1.9, asyMed, { color: 'ink', width: 2, dash: true });
        sMe.set(`${S.fmt(nv[0], 2)} / ${Number.isFinite(P.v) ? S.fmt(P.v, 2) : '∞'}`, Number.isFinite(P.v) ? '' : 'the mean of Cauchy data is Cauchy: no variance');
        sMd.set(`${S.fmt(nv[1], 2)} / ${S.fmt(asyMed, 2)}`);
        sAre.set(Number.isFinite(P.v) ? S.fmt(P.v / asyMed, 2) : '∞', 'asymptotic σ²·4f(0)²; >1 favours the median');
        const b = S.argmin(nv);
        sBest.set(ESTS[b][0]);
      }
      run();
    },
  });
})();
