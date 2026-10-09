/* Labs: Markov Chain Monte Carlo. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const NS = 'http://www.w3.org/2000/svg';
  const LOG2PI = Math.log(2 * Math.PI);

  /* ================================================================== */
  /* Shared helpers                                                       */
  /* ================================================================== */

  // A <g> drawn beneath the plot's data layer that survives p.clear(): used for static target shading.
  function underlay(p) {
    const g = document.createElementNS(NS, 'g');
    g.setAttribute('clip-path', p.layer.getAttribute('clip-path'));
    p.svg.insertBefore(g, p.layer);
    return g;
  }
  // Shade a 2D log density (log scale, `depth` nats deep) and draw contours at Gaussian-like levels.
  function shade(p, g, lp, { color = 's1', depth = 9, contours = true } = {}) {
    g.innerHTML = '';
    const [x0, x1] = p.o.x, [y0, y1] = p.o.y;
    let mx = -Infinity;
    for (let i = 0; i <= 60; i++) for (let j = 0; j <= 45; j++) {
      const v = lp(x0 + (x1 - x0) * i / 60, y0 + (y1 - y0) * j / 45);
      if (v > mx) mx = v;
    }
    const c = Atlas.rgb(color);
    p.raster((x, y) => { const t = Math.max(0, 1 + (lp(x, y) - mx) / depth); return [c[0], c[1], c[2], Math.round(120 * t * t)]; }, { parent: g, nx: 110, ny: 80 });
    if (contours) p.contour(lp, [0.5, 2, 4.5, 8].map(d => mx - d), { parent: g, color: 'muted', opacity: 0.5, width: 1, nx: 70, ny: 55 });
  }
  const lse = (a, b) => (a > b ? a + Math.log1p(Math.exp(b - a)) : b + Math.log1p(Math.exp(a - b)));
  const fmt = (x, d = 3) => (Number.isFinite(x) ? S.fmt(x, d) : (x === Infinity ? '∞' : '—'));
  const pct = x => (Number.isFinite(x) ? (100 * x).toFixed(1) + '%' : '—');

  // Integrated autocorrelation time by Geyer's initial monotone sequence estimator.
  function iact(x, maxLag = 4000) {
    const n = x.length;
    if (n < 20) return NaN;
    let m = 0;
    for (let i = 0; i < n; i++) m += x[i];
    m /= n;
    const c = new Float64Array(n);
    for (let i = 0; i < n; i++) c[i] = x[i] - m;
    const g = k => { let s = 0; for (let i = 0; i + k < n; i++) s += c[i] * c[i + k]; return s / n; };
    const g0 = g(0);
    if (!(g0 > 1e-300)) return Infinity;
    let tau = -1, prev = Infinity;
    const L = Math.min(n - 2, maxLag);
    for (let k = 0; k + 1 <= L; k += 2) {
      let G = (g(k) + g(k + 1)) / g0;
      if (!(G > 0)) break;
      if (G > prev) G = prev;
      prev = G;
      tau += 2 * G;
    }
    return Math.max(tau, 1e-3);
  }
  function acf(x, maxLag) {
    const n = x.length, m = S.mean(x), out = [];
    let g0 = 0;
    for (let i = 0; i < n; i++) g0 += (x[i] - m) ** 2;
    for (let k = 0; k <= maxLag; k++) {
      let s = 0;
      for (let i = 0; i + k < n; i++) s += (x[i] - m) * (x[i + k] - m);
      out.push(g0 > 0 ? s / g0 : NaN);
    }
    return out;
  }

  // R-hat family (Gelman–Rubin; split; rank-normalized split, Vehtari et al. 2021) and multi-chain ESS.
  function rhatBasic(chains) {
    const M = chains.length, N = Math.min(...chains.map(c => c.length));
    if (M < 2 || N < 4) return NaN;
    const means = chains.map(c => { let s = 0; for (let i = 0; i < N; i++) s += c[i]; return s / N; });
    const vars = chains.map((c, m) => { let s = 0; for (let i = 0; i < N; i++) s += (c[i] - means[m]) ** 2; return s / (N - 1); });
    const B = N * S.variance(means), W = S.mean(vars);
    if (!(W > 0)) return Infinity;
    return Math.sqrt(((N - 1) / N * W + B / N) / W);
  }
  function split(chains) {
    const out = [];
    for (const c of chains) { const h = Math.floor(c.length / 2); out.push(c.slice(0, h), c.slice(c.length - h)); }
    return out;
  }
  function rankNormalize(chains) {
    const all = [];
    chains.forEach((c, m) => c.forEach((v, i) => all.push([v, m, i])));
    all.sort((a, b) => a[0] - b[0]);
    const Stot = all.length, out = chains.map(c => new Array(c.length));
    for (let i = 0; i < Stot;) {
      let j = i;
      while (j + 1 < Stot && all[j + 1][0] === all[i][0]) j++;
      const r = (i + j) / 2 + 1; // average rank (1-based) for ties
      const z = S.qnorm((r - 0.375) / (Stot + 0.25));
      for (let k = i; k <= j; k++) out[all[k][1]][all[k][2]] = z;
      i = j + 1;
    }
    return out;
  }
  function rhatRank(chains) {
    const sp = split(chains);
    const bulk = rhatBasic(rankNormalize(sp));
    const med = S.median([].concat(...sp));
    const fold = rhatBasic(rankNormalize(sp.map(c => c.map(v => Math.abs(v - med)))));
    return Math.max(bulk, fold);
  }
  function essMulti(chains) {
    const M = chains.length, N = Math.min(...chains.map(c => c.length));
    if (M < 1 || N < 8) return NaN;
    const cs = chains.map(c => { const m = S.mean(c.slice(0, N)); return Float64Array.from(c.slice(0, N), v => v - m); });
    const acov = (c, k) => { let s = 0; for (let i = 0; i + k < N; i++) s += c[i] * c[i + k]; return s / N; };
    const means = chains.map(c => S.mean(c.slice(0, N)));
    const vars = cs.map(c => acov(c, 0) * N / (N - 1));
    const W = S.mean(vars), B = M > 1 ? N * S.variance(means) : 0;
    const vp = (N - 1) / N * W + B / N;
    if (!(vp > 0)) return 0;
    const rho = k => 1 - (W - S.mean(cs.map(c => acov(c, k) * N / (N - 1)))) / vp;
    let tau = -1, prev = Infinity;
    for (let k = 0; k + 1 < Math.min(N - 1, 3000); k += 2) {
      let P = rho(k) + rho(k + 1);
      if (!(P > 0)) break;
      if (P > prev) P = prev;
      prev = P;
      tau += 2 * P;
    }
    return (M * N) / Math.max(tau, 1e-3);
  }

  // Shared 2D targets: log density (unnormalized), gradient, display box, moments, a start point.
  const T2 = {
    gauss: (() => {
      const r = 0.9, k = 1 / (1 - r * r);
      return {
        label: 'Correlated Gaussian (ρ = 0.9)', x: [-3.6, 3.6], y: [-3.6, 3.6], center: [0, 0], mean: [0, 0], sd: [1, 1], start: [-2.6, 2.6],
        lp: (x, y) => -0.5 * k * (x * x - 2 * r * x * y + y * y),
        grad: (x, y) => [-k * (x - r * y), -k * (y - r * x)],
      };
    })(),
    banana: {
      label: 'Banana', x: [-5, 5], y: [-3, 8.5], center: [0, 0.125], mean: [0, 0.125], sd: [1.5, Math.sqrt(0.5 * 2.25 * 2.25 + 0.64)], start: [3.5, 5.5],
      lp: (x, y) => { const u = y - 0.5 * x * x + 1; return -x * x / 4.5 - u * u / 1.28; },
      grad: (x, y) => { const u = y - 0.5 * x * x + 1; return [-x / 2.25 + u * x / 0.64, -u / 0.64]; },
    },
    bimodal: {
      label: 'Bimodal (two Gaussians)', x: [-4.5, 4.5], y: [-4.5, 4.5], center: [0, 0], mean: [0, 0], sd: [Math.sqrt(4.36), Math.sqrt(4.36)], start: [-2, -2],
      lp: (x, y) => lse(-((x + 2) ** 2 + (y + 2) ** 2) / 0.72, -((x - 2) ** 2 + (y - 2) ** 2) / 0.72),
      grad: (x, y) => {
        const a = -((x + 2) ** 2 + (y + 2) ** 2) / 0.72, b = -((x - 2) ** 2 + (y - 2) ** 2) / 0.72, l = lse(a, b);
        const r1 = Math.exp(a - l), r2 = Math.exp(b - l);
        return [-(r1 * (x + 2) + r2 * (x - 2)) / 0.36, -(r1 * (y + 2) + r2 * (y - 2)) / 0.36];
      },
    },
    funnel: {
      label: "Neal's funnel (x, v)", x: [-12, 12], y: [-8, 8], center: [0, 0], mean: [0, 0], sd: [Math.exp(2.25), 3], start: [4, 3],
      lp: (x, v) => -v * v / 18 - v / 2 - 0.5 * x * x * Math.exp(-v),
      grad: (x, v) => { const e = Math.exp(-v); return [-x * e, -v / 9 - 0.5 + 0.5 * x * x * e]; },
    },
  };
  const targetOptions = keys => keys.map(k => ({ value: k, label: T2[k].label }));

  /* ================================================================== */
  /* mcmc-foundations                                                    */
  /* ================================================================== */
  const KST = 12;
  const CHAIN_TARGETS = {
    bimodal: { label: 'Bimodal', w: i => Math.exp(-((i - 2) ** 2) / 3) + 0.6 * Math.exp(-((i - 9) ** 2) / 2) + 1e-3 },
    geometric: { label: 'Geometric', w: i => 0.7 ** i },
    wavy: { label: 'Wavy', w: i => 1 + 0.6 * Math.sin(1.3 * i) },
  };
  function buildP(pi, prop) {
    const qR = prop === 'sym' ? 0.5 : 0.8, qL = 1 - qR;
    const P = Array.from({ length: KST }, () => new Array(KST).fill(0));
    for (let i = 0; i < KST; i++) {
      for (const [j, q, qrev] of [[i + 1, qR, qL], [i - 1, qL, qR]]) {
        if (j < 0 || j >= KST) continue; // proposals off the ends are rejected
        let ratio = pi[j] / pi[i];
        if (prop === 'hast') ratio *= qrev / q;
        P[i][j] = q * Math.min(1, ratio);
      }
      let s = 0;
      for (let j = 0; j < KST; j++) if (j !== i) s += P[i][j];
      P[i][i] = 1 - s;
    }
    return P;
  }

  Atlas.lab('mcmc-foundations-chain', {
    title: 'Detailed balance on a 12-state chain',
    blurb: 'A Metropolis chain on $\\{0,\\dots,11\\}$ with nearest-neighbour proposals. Left: target $\\pi$ (stems), visit frequencies (bars), and the kernel\'s true stationary law (dots). Right: exact $\\|P^t(x_0,\\cdot)-\\pi\\|_{\\mathrm{TV}}$. Switch to the biased proposal <em>without</em> the Hastings correction: the chain still converges, to the wrong distribution.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tsel = UI.select(controls, { label: 'Target $\\pi$', options: Object.keys(CHAIN_TARGETS).map(k => ({ value: k, label: CHAIN_TARGETS[k].label })), value: 'bimodal', onChange: rebuild });
      const psel = UI.select(controls, { label: 'Proposal', options: [
        { value: 'sym', label: 'Symmetric ±1 (Metropolis)' },
        { value: 'hast', label: 'Biased 80% right + Hastings ratio' },
        { value: 'nohast', label: 'Biased 80% right, no correction' }], value: 'sym', onChange: rebuild });
      const x0s = UI.slider(controls, { label: 'Start state $x_0$', min: 0, max: KST - 1, step: 1, value: 0, onChange: rebuild });
      const sp = UI.slider(controls, { label: 'Steps per frame', min: 1, max: 500, value: 20, log: true, fmt: v => String(Math.round(v)) });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(() => { for (let k = 0; k < Math.round(sp.get()); k++) step(); draw(); if (nSteps >= 200000) return false; });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', rebuild);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-0.6, KST - 0.4], y: [0, 0.3], xlabel: 'state', ylabel: 'probability', title: 'Target vs. what the chain does' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 300], y: [1e-4, 1], logy: true, xlabel: 't', ylabel: 'TV distance', title: 'Exact distance to target from x₀' });
      const sN = UI.stat(stats, 'Steps simulated');
      const sTVe = UI.stat(stats, 'TV(empirical, $\\pi$)');
      const sTVs = UI.stat(stats, 'TV(stationary of $P$, $\\pi$)');
      const sDB = UI.stat(stats, 'Detailed balance w.r.t. $\\pi$');
      const sGap = UI.stat(stats, 'SLEM $|\\lambda_2|$');
      let pi, P, nu, tv, state, counts, nSteps;
      function rebuild() {
        const T = CHAIN_TARGETS[tsel.get()];
        const w = Array.from({ length: KST }, (_, i) => T.w(i)), Z = S.sum(w);
        pi = w.map(v => v / Z);
        P = buildP(pi, psel.get());
        // Stationary law of P (birth–death chain: ratio of neighbouring masses = ratio of rates).
        const r = [1];
        for (let i = 0; i + 1 < KST; i++) r.push(r[i] * P[i][i + 1] / P[i + 1][i]);
        const Zr = S.sum(r);
        nu = r.map(v => v / Zr);
        // Exact TV curve from x0.
        let mu = new Array(KST).fill(0);
        mu[Math.round(x0s.get())] = 1;
        tv = [];
        for (let t = 0; t <= 300; t++) {
          tv.push([t, Math.max(1e-4, 0.5 * mu.reduce((s, m, i) => s + Math.abs(m - pi[i]), 0))]);
          const nm = new Array(KST).fill(0);
          for (let i = 0; i < KST; i++) if (mu[i]) for (let j = 0; j < KST; j++) nm[j] += mu[i] * P[i][j];
          mu = nm;
        }
        // Spectral gap via symmetrization with the chain's own stationary law.
        const A = P.map((rowP, i) => rowP.map((p, j) => p * Math.sqrt(nu[i] / nu[j])));
        for (let i = 0; i < KST; i++) for (let j = i + 1; j < KST; j++) { const m = (A[i][j] + A[j][i]) / 2; A[i][j] = A[j][i] = m; }
        const ev = S.eigSym(A).values.slice().sort((a, b) => b - a);
        const slem = Math.max(Math.abs(ev[1]), Math.abs(ev[KST - 1]));
        let viol = 0, scale = 0;
        for (let i = 0; i < KST; i++) for (let j = 0; j < KST; j++) { viol = Math.max(viol, Math.abs(pi[i] * P[i][j] - pi[j] * P[j][i])); scale = Math.max(scale, pi[i] * P[i][j]); }
        sDB.set(viol < 1e-12 * Math.max(scale, 1e-300) + 1e-15 ? '<span style="color:var(--good)">holds</span>' : '<span style="color:var(--bad)">fails</span>', `max violation ${S.fmt(viol, 2)}`);
        sGap.set(S.fmt(slem, 4), `relaxation time ≈ ${S.fmt(1 / (1 - slem), 3)} steps`);
        sTVs.set(S.fmt(0.5 * nu.reduce((s, v, i) => s + Math.abs(v - pi[i]), 0), 3));
        state = Math.round(x0s.get());
        counts = new Array(KST).fill(0);
        nSteps = 0;
        loop.stop();
        draw();
      }
      function step() {
        const u = ctx.rng.u(), row = P[state];
        let c = 0, j = 0;
        for (; j < KST - 1; j++) { c += row[j]; if (u < c) break; }
        state = j;
        counts[state]++;
        nSteps++;
      }
      function draw() {
        const ymax = Math.max(...pi, ...nu, ...counts.map(c => (nSteps ? c / nSteps : 0))) * 1.2;
        pL.setDomain(null, [0, ymax]).clear();
        if (nSteps) pL.bars(counts.map((c, i) => ({ x0: i - 0.38, x1: i + 0.38, y: c / nSteps })), { color: 's2', opacity: 0.45 });
        pL.stems(pi.map((p, i) => [i, p]), { color: 'ink', width: 2, r: 3 });
        nu.forEach((v, i) => pL.dot(i, v, { color: 'bad', r: 4 }));
        pL.dot(state, 0, { color: 'accent', r: 6 });
        pL.legend([{ label: 'target π', color: 'ink' }, { label: 'visit frequency', color: 's2', fill: true }, { label: 'stationary law of P', color: 'bad' }]);
        pR.clear();
        pR.line(tv, { color: 's1', width: 2 });
        if (nSteps && nSteps <= 300) pR.vline(nSteps, { color: 'accent' });
        sN.set(String(nSteps));
        sTVe.set(nSteps ? S.fmt(0.5 * counts.reduce((s, c, i) => s + Math.abs(c / nSteps - pi[i]), 0), 3) : '—');
      }
      rebuild();
    },
  });

  Atlas.lab('mcmc-foundations-clt', {
    title: 'The Markov chain CLT: variance inflation in action',
    blurb: 'Run 300 independent random-walk Metropolis chains of length $n$ on $\\N(0,1)$ and histogram $\\sqrt n\\,\\bar X_n$. An IID sample would give $\\N(0,1)$ (dashed); the chain gives $\\N(0,\\tau)$ with $\\tau = 1+2\\sum\\rho_k$. Small proposal scales inflate $\\tau$ and wreck naive confidence intervals; batch means repairs them. Start far away to see burn-in bias.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sc = UI.slider(controls, { label: 'Proposal scale $\\sigma$', min: 0.05, max: 20, value: 0.5, log: true, fmt: v => v.toFixed(2), onChange: run });
      const nn = UI.slider(controls, { label: 'Chain length $n$', min: 100, max: 4000, step: 100, value: 1000, onChange: run });
      const st = UI.segmented(controls, { label: 'Start', options: [{ value: 'stat', label: '$x_0\\sim\\pi$' }, { value: 'far', label: '$x_0 = 10$' }], value: 'stat', onChange: run });
      UI.button(controls, 'Rerun 300 chains', run, { primary: true });
      stage.classList.add('two');
      const pH = new Plot(stage, { width: 420, height: 270, xlabel: '√n · (chain mean)', ylabel: 'density', title: 'Sampling distribution of the ergodic mean' });
      const pT = new Plot(stage, { width: 420, height: 270, x: [0, 500], y: [-3.5, 3.5], xlabel: 'iteration', ylabel: 'x', title: 'One chain (first 500 steps)' });
      const sA = UI.stat(stats, 'Acceptance rate');
      const sI = UI.stat(stats, 'Inflation $n\\Var(\\bar X_n)$', { accent: 'accent' });
      const sB = UI.stat(stats, 'Mean of $\\bar X_n$ (truth 0)');
      const sC = UI.stat(stats, 'Naive 95% CI coverage');
      const sCB = UI.stat(stats, 'Batch-means 95% CI coverage');
      function run() {
        const rng = ctx.rng, s = sc.get(), n = Math.round(nn.get()), R = 300, far = st.get() === 'far';
        const b = Math.max(1, Math.floor(Math.sqrt(n))), a = Math.floor(n / b);
        const means = new Float64Array(R), bm = new Float64Array(a), trace = [];
        let acc = 0, cover = 0, coverBM = 0;
        for (let r = 0; r < R; r++) {
          let x = far ? 10 : rng.normal(), lx = -0.5 * x * x, sum = 0, sum2 = 0;
          bm.fill(0);
          for (let t = 0; t < n; t++) {
            const y = x + s * rng.normal(), ly = -0.5 * y * y;
            if (Math.log(rng.u()) < ly - lx) { x = y; lx = ly; acc++; }
            sum += x; sum2 += x * x;
            const bi = (t / b) | 0;
            if (bi < a) bm[bi] += x;
            if (r === 0 && t < 500) trace.push([t, x]);
          }
          const m = sum / n, v = Math.max(0, (sum2 - n * m * m) / (n - 1));
          means[r] = m;
          if (Math.abs(m) <= 1.96 * Math.sqrt(v / n)) cover++;
          let bmean = 0;
          for (let j = 0; j < a; j++) { bm[j] /= b; bmean += bm[j]; }
          bmean /= a;
          let sb = 0;
          for (let j = 0; j < a; j++) sb += (bm[j] - bmean) ** 2;
          const sig2 = b * sb / (a - 1);
          if (Math.abs(m) <= 1.96 * Math.sqrt(sig2 / n)) coverBM++;
        }
        const z = Array.from(means, m => Math.sqrt(n) * m);
        const infl = n * S.variance(Array.from(means));
        let [lo, hi] = S.extent(z);
        const L = Math.max(4, Math.abs(lo), Math.abs(hi)) * 1.05;
        lo = -L; hi = L;
        const hist = S.histogram(z, { bins: 36, lo, hi });
        const grid = S.linspace(lo, hi, 240);
        const ymax = Math.max(...hist.map(h => h.density), S.pdf.normal(0), S.pdf.normal(0, 0, Math.sqrt(Math.max(infl, 1e-6)))) * 1.1;
        pH.setDomain([lo, hi], [0, Math.min(ymax, 1.5)]).clear();
        pH.bars(hist, { color: 's2', opacity: 0.45 });
        pH.line(grid.map(g => [g, S.pdf.normal(g)]), { color: 'muted', width: 2, dash: true });
        if (infl > 0) pH.line(grid.map(g => [g, S.pdf.normal(g, 0, Math.sqrt(infl))]), { color: 'ink', width: 2 });
        pH.legend([{ label: 'IID theory N(0,1)', color: 'muted', dash: true }, { label: 'N(0, n·Var X̄)', color: 'ink' }, { label: '300 chains', color: 's2', fill: true }]);
        const yr = Math.max(3.5, ...trace.map(p => Math.abs(p[1]))) * 1.05;
        pT.setDomain([0, Math.min(500, n)], [-yr, yr]).clear();
        pT.hline(0, { color: 'faint' });
        pT.line(trace, { color: 's1', width: 1.3 });
        sA.set(pct(acc / (R * n)));
        sI.set(S.fmt(infl, 3), 'estimates τ = 1 + 2Σρ<sub>k</sub>');
        sB.set(S.fmt(S.mean(Array.from(means)), 3));
        sC.set(pct(cover / R), 'using s/√n');
        sCB.set(pct(coverBM / R), `batch size b = ${b}`);
      }
      run();
    },
  });

  /* ================================================================== */
  /* metropolis-hastings                                                 */
  /* ================================================================== */
  Atlas.lab('metropolis-hastings-rw', {
    title: 'Random-walk and independence Metropolis on 2D targets',
    blurb: 'Tune the proposal scale and watch acceptance rate, trace, and effective sample size move in opposite directions. Too small: everything is accepted but the chain crawls. Too large: everything is rejected and the trace goes flat. The independence sampler is superb when its Gaussian covers the target and sticks for long stretches when it is too narrow. Click the left panel to move the chain.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tsel = UI.select(controls, { label: 'Target $\\pi$', options: targetOptions(Object.keys(T2)), value: 'gauss', onChange: () => { setTarget(); reset(); } });
      const kind = UI.segmented(controls, { label: 'Proposal', options: [{ value: 'rw', label: 'Random walk' }, { value: 'ind', label: 'Independence' }], value: 'rw', onChange: reset });
      const sc = UI.slider(controls, { label: 'Proposal scale $\\sigma$', min: 0.05, max: 20, value: 0.6, log: true, fmt: v => v.toFixed(2), onChange: reset });
      const sp = UI.slider(controls, { label: 'Steps per frame', min: 1, max: 200, value: 5, log: true, fmt: v => String(Math.round(v)) });
      const row = UI.h('div', 'btn-row', controls);
      const MAXN = 20000;
      const loop = ctx.loop(() => {
        const k = Math.round(sp.get());
        for (let i = 0; i < k && n < MAXN; i++) mh();
        frame++;
        if (frame % 20 === 0) updateEss();
        draw();
        if (n >= MAXN) { updateEss(); draw(); return false; }
      });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 300, equal: true, xlabel: 'x₁', ylabel: 'x₂', title: 'Draws (dots) and recent path' });
      const bg = underlay(pS);
      const pT = new Plot(stage, { width: 420, height: 300, xlabel: 'iteration', ylabel: 'x₁', title: 'Trace of x₁ (last 1000 iterations)' });
      const sAcc = UI.stat(stats, 'Acceptance rate', { accent: 'accent' });
      const sE1 = UI.stat(stats, 'ESS of $x_1$');
      const sE2 = UI.stat(stats, 'ESS of $x_2$');
      const sM = UI.stat(stats, 'Estimate of $\\E x_1$ ± MCSE');
      const sN = UI.stat(stats, 'Iterations');
      let T, X, lpX, xs, ys, nAcc, n, last, ess, frame = 0;
      pS.onPointer((x, y, e, type) => { if (type === 'down' && pS.inside(x, y)) { X = [x, y]; lpX = T.lp(x, y); last = null; draw(); } });
      function setTarget() { T = T2[tsel.get()]; pS.setDomain(T.x, T.y); shade(pS, bg, T.lp); }
      function reset() { X = T.start.slice(); lpX = T.lp(X[0], X[1]); xs = [X[0]]; ys = [X[1]]; nAcc = 0; n = 0; last = null; ess = [NaN, NaN]; frame = 0; draw(); }
      function mh() {
        const r = ctx.rng, s = sc.get();
        let Y, lpY, logr;
        if (kind.get() === 'rw') {
          Y = [X[0] + s * r.normal(), X[1] + s * r.normal()];
          lpY = T.lp(Y[0], Y[1]);
          logr = lpY - lpX;
        } else {
          const c = T.center, lg = z => -((z[0] - c[0]) ** 2 + (z[1] - c[1]) ** 2) / (2 * s * s);
          Y = [c[0] + s * r.normal(), c[1] + s * r.normal()];
          lpY = T.lp(Y[0], Y[1]);
          logr = lpY - lpX + lg(X) - lg(Y);
        }
        const acc = Math.log(r.u()) < logr;
        last = { from: X, to: Y, acc };
        if (acc) { X = Y; lpX = lpY; nAcc++; }
        n++;
        xs.push(X[0]); ys.push(X[1]);
      }
      function updateEss() { if (xs.length > 50) ess = [xs.length / iact(xs), ys.length / iact(ys)]; }
      function draw() {
        pS.clear();
        const k0 = Math.max(0, xs.length - 3000), pts = [];
        for (let i = k0; i < xs.length; i++) pts.push([xs[i], ys[i]]);
        pS.points(pts, { color: 's2', r: 1.6, opacity: 0.45 });
        const k1 = Math.max(0, xs.length - 40), path = [];
        for (let i = k1; i < xs.length; i++) path.push([xs[i], ys[i]]);
        pS.line(path, { color: 'ink', width: 1.2, opacity: 0.7 });
        if (last) { pS.segment(last.from[0], last.from[1], last.to[0], last.to[1], { color: last.acc ? 'good' : 'bad', width: 1, dash: true }); pS.dot(last.to[0], last.to[1], { color: last.acc ? 'good' : 'bad', r: 3.5 }); }
        pS.dot(X[0], X[1], { color: 'accent', r: 5 });
        const t0 = Math.max(0, xs.length - 1000), tr = [];
        for (let i = t0; i < xs.length; i++) tr.push([i, xs[i]]);
        pT.setDomain([t0, Math.max(t0 + 1000, 10)], T.x).clear();
        pT.hline(T.mean[0], { color: 'muted' });
        pT.line(tr, { color: 's1', width: 1.2 });
        sAcc.set(n ? pct(nAcc / n) : '—', 'RWM optimum in high d: 23.4%');
        sE1.set(Number.isFinite(ess[0]) ? Math.round(ess[0]).toString() : '—', Number.isFinite(ess[0]) ? `${(1000 * ess[0] / xs.length).toFixed(1)} per 1000 draws` : 'updates while running');
        sE2.set(Number.isFinite(ess[1]) ? Math.round(ess[1]).toString() : '—');
        const m = S.mean(xs), mcse = Number.isFinite(ess[0]) && ess[0] > 0 ? S.sd(xs) / Math.sqrt(ess[0]) : NaN;
        sM.set(`${S.fmt(m, 3)} ± ${fmt(mcse, 2)}`, `truth ${T.mean[0]}`);
        sN.set(String(n));
      }
      setTarget();
      reset();
    },
  });

  Atlas.lab('metropolis-hastings-scaling', {
    title: 'Optimal scaling: why 0.234',
    blurb: 'RWM on an IID $\\N(0,I_d)$ target with proposal $\\N(0, (\\ell^2/d) I)$, for 22 values of $\\ell$. Left: expected squared jump distance (a proxy for efficiency) against acceptance rate; the curve is the $d\\to\\infty$ limit $h(\\ell)=2\\ell^2\\Phi(-\\ell/2)$, maximized at acceptance $0.234$. Increase $d$ and watch the simulations collapse onto it; at $d=1$ the optimum acceptance is near $0.44$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dd = UI.slider(controls, { label: 'Dimension $d$', min: 1, max: 100, step: 1, value: 20, onChange: run });
      const NN = UI.slider(controls, { label: 'Iterations per $\\ell$', min: 300, max: 4000, step: 100, value: 1500, onChange: run });
      UI.button(controls, 'Rerun', run, { primary: true });
      stage.classList.add('two');
      const pE = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 1.6], xlabel: 'acceptance rate', ylabel: 'ESJD (sum over coordinates)', title: 'Efficiency vs. acceptance' });
      const pA = new Plot(stage, { width: 420, height: 270, x: [0, 8.2], y: [0, 1], xlabel: 'ℓ (proposal sd = ℓ/√d)', ylabel: 'acceptance rate', title: 'Acceptance vs. scale' });
      const sL = UI.stat(stats, 'Best simulated $\\ell$', { accent: 'accent' });
      const sA = UI.stat(stats, 'Its acceptance rate');
      const sT = UI.stat(stats, 'Limit theory');
      sT.set('ℓ = 2.38, a = 0.234');
      const theory = S.linspace(0.02, 10, 300).map(l => { const a = 2 * S.cdf.normal(-l / 2); return { l, a, h: l * l * a }; });
      function run() {
        const rng = ctx.rng, d = Math.round(dd.get()), N = Math.round(NN.get());
        const ls = S.linspace(Math.log(0.3), Math.log(8), 22).map(Math.exp);
        const x = new Float64Array(d), y = new Float64Array(d);
        const sims = ls.map(l => {
          const s = l / Math.sqrt(d);
          let lx = 0;
          for (let i = 0; i < d; i++) { x[i] = rng.normal(); lx += x[i] * x[i]; }
          lx *= -0.5;
          let acc = 0, jump = 0;
          for (let t = 0; t < N; t++) {
            let ly = 0;
            for (let i = 0; i < d; i++) { y[i] = x[i] + s * rng.normal(); ly += y[i] * y[i]; }
            ly *= -0.5;
            if (Math.log(rng.u()) < ly - lx) {
              let j = 0;
              for (let i = 0; i < d; i++) { j += (y[i] - x[i]) ** 2; x[i] = y[i]; }
              jump += j; lx = ly; acc++;
            }
          }
          return { l, a: acc / N, h: jump / N };
        });
        const ymax = Math.max(1.5, ...sims.map(s => s.h)) * 1.1;
        pE.setDomain(null, [0, ymax]).clear();
        pE.vline(0.234, { color: 'accent' });
        pE.line(theory.map(t => [t.a, t.h]), { color: 'ink', width: 2 });
        pE.points(sims.map(s => [s.a, s.h]), { color: 's2', r: 4, opacity: 0.85 });
        pE.text(0.24, ymax * 0.95, '0.234', { color: 'accent', size: 11, dx: 4 });
        pE.legend([{ label: 'd → ∞ limit', color: 'ink' }, { label: `simulated, d = ${d}`, color: 's2' }]);
        pA.clear();
        pA.hline(0.234, { color: 'accent' });
        pA.vline(2.38, { color: 'accent' });
        pA.line(theory.filter(t => t.l <= 8.2).map(t => [t.l, t.a]), { color: 'ink', width: 2 });
        pA.points(sims.map(s => [s.l, s.a]), { color: 's2', r: 4, opacity: 0.85 });
        const best = sims.reduce((a, b) => (b.h > a.h ? b : a));
        sL.set(S.fmt(best.l, 3));
        sA.set(S.fmt(best.a, 3));
      }
      run();
    },
  });

  /* ================================================================== */
  /* gibbs-sampling                                                      */
  /* ================================================================== */
  Atlas.lab('gibbs-sampling-zigzag', {
    title: 'Gibbs zig-zag on a correlated Gaussian',
    blurb: 'Systematic-scan Gibbs on a bivariate normal moves only along the axes, in steps of the conditional width $\\sqrt{1-\\rho^2}$. The $x$-chain is an AR(1) with coefficient $\\rho^2$ (right: empirical ACF vs. $\\rho^{2k}$). Push $\\rho$ toward 1 and watch the walk freeze; switch to the blocked update to draw $(x,y)$ jointly.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rs = UI.slider(controls, { label: 'Correlation $\\rho$', min: 0, max: 0.999, step: 0.001, value: 0.9, onChange: reset, onInput: () => drawBg() });
      const mode = UI.segmented(controls, { label: 'Update', options: [{ value: 'sys', label: 'Systematic' }, { value: 'rand', label: 'Random scan' }, { value: 'block', label: 'Blocked' }], value: 'sys', onChange: reset });
      const sp = UI.slider(controls, { label: 'Sweeps per frame', min: 1, max: 100, value: 1, log: true, fmt: v => String(Math.round(v)) });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(() => {
        for (let k = 0; k < Math.round(sp.get()); k++) sweep();
        if (++frame % 30 === 0) tau = xs.length > 100 ? iact(xs) : NaN;
        draw();
        if (xs.length >= 30000) return false;
      });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 300, x: [-3.5, 3.5], y: [-3.5, 3.5], equal: true, xlabel: 'x', ylabel: 'y', title: 'Path (last 30 moves) and draws' });
      const pA = new Plot(stage, { width: 420, height: 300, x: [-0.5, 40.5], y: [-0.2, 1.05], xlabel: 'lag k', ylabel: 'autocorrelation', title: 'ACF of the x-chain (per sweep)' });
      const sR = UI.stat(stats, 'Lag-1 ACF (theory $\\rho^2$)', { accent: 'accent' });
      const sT = UI.stat(stats, 'IACT theory $\\frac{1+\\rho^2}{1-\\rho^2}$');
      const sE = UI.stat(stats, 'Estimated IACT');
      const sN = UI.stat(stats, 'Sweeps');
      let x, y, xs, pairs, path, tau, frame = 0;
      function reset() { x = -2.5; y = 2.5 * rs.get(); xs = []; pairs = []; path = [[x, y]]; tau = NaN; frame = 0; draw(); }
      function drawBg() { draw(); }
      function sweep() {
        const r = ctx.rng, rho = rs.get(), c = Math.sqrt(1 - rho * rho), m = mode.get();
        if (m === 'sys') {
          x = rho * y + c * r.normal(); path.push([x, y]);
          y = rho * x + c * r.normal(); path.push([x, y]);
        } else if (m === 'rand') {
          for (let k = 0; k < 2; k++) {
            if (r.u() < 0.5) x = rho * y + c * r.normal(); else y = rho * x + c * r.normal();
            path.push([x, y]);
          }
        } else {
          const z1 = r.normal(), z2 = r.normal();
          x = z1; y = rho * z1 + c * z2; path.push([x, y]);
        }
        if (path.length > 60) path.splice(0, path.length - 60);
        xs.push(x);
        pairs.push([x, y]);
        if (pairs.length > 3000) pairs.splice(0, pairs.length - 3000);
      }
      function draw() {
        const rho = rs.get();
        pS.clear();
        for (const k of [1, 2, 3]) pS.ellipse(0, 0, [[1, rho], [rho, 1]], k, { color: 'muted', width: 1, opacity: 0.6 });
        pS.points(pairs, { color: 's2', r: 1.6, opacity: 0.45 });
        pS.line(path.slice(-30), { color: 'ink', width: 1.3 });
        pS.dot(x, y, { color: 'accent', r: 5 });
        pA.clear();
        pA.hline(0, { color: 'faint', dash: false, width: 1 });
        const m = mode.get();
        if (m === 'sys') pA.line(S.range(41).map(k => [k, rho ** (2 * k)]), { color: 'ink', width: 2 });
        if (m === 'block') pA.line([[0, 1], [0.5, 0], [40, 0]], { color: 'ink', width: 2 });
        const tail = xs.slice(Math.max(0, xs.length - 5000));
        if (tail.length > 60) {
          const a = acf(tail, 40);
          pA.bars(a.map((v, k) => ({ x0: k - 0.35, x1: k + 0.35, y: v })), { color: 's2', opacity: 0.6, base: 0 });
          sR.set(S.fmt(a[1], 3), `theory ${S.fmt(rho * rho, 3)} (systematic)`);
        } else sR.set('—', `theory ${S.fmt(rho * rho, 3)} (systematic)`);
        pA.legend([{ label: 'empirical', color: 's2', fill: true }, { label: m === 'rand' ? 'no simple theory' : 'theory', color: 'ink' }]);
        const th = m === 'block' ? 1 : m === 'sys' ? (1 + rho * rho) / (1 - rho * rho) : NaN;
        sT.set(m === 'rand' ? '—' : S.fmt(th, 4), m === 'block' ? 'blocked draws are IID' : '');
        sE.set(fmt(tau, 3), Number.isFinite(tau) ? `ESS ≈ ${Math.round(xs.length / tau)} of ${xs.length}` : 'updates while running');
        sN.set(String(xs.length));
      }
      reset();
    },
  });

  Atlas.lab('gibbs-sampling-normal', {
    title: 'Gibbs for a normal model with a semi-conjugate prior',
    blurb: 'Data $y_i\\sim\\N(\\mu,\\sigma^2)$ (truth $\\mu=2$, $\\sigma=1.5$); priors $\\mu\\sim\\N(0,\\tau_0^2)$, $\\sigma^2\\sim\\text{Inv-Gamma}(2,2)$. The sampler alternates the normal conditional for $\\mu$ (horizontal moves) and the inverse-gamma conditional for $\\sigma^2$ (vertical moves). Contours show the exact posterior computed on a grid. A tight prior that conflicts with the data creates dependence between $\\mu$ and $\\sigma$, and the zig-zag slows down.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ns = UI.slider(controls, { label: 'Sample size $n$', min: 3, max: 200, step: 1, value: 10, onChange: newData });
      const ts = UI.slider(controls, { label: 'Prior sd $\\tau_0$ of $\\mu$', min: 0.05, max: 100, value: 10, log: true, fmt: v => v.toFixed(2), onChange: setup });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(() => { for (let k = 0; k < 3; k++) iter(); draw(); if (mus.length >= 20000) return false; });
      ctx.playButton(row, loop);
      UI.button(row, 'New data', newData);
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 290, xlabel: 'μ', ylabel: 'σ', title: 'Gibbs path on the exact posterior' });
      const bg = underlay(pS);
      const pH = new Plot(stage, { width: 420, height: 290, xlabel: 'μ', ylabel: 'density', title: 'Marginal posterior of μ' });
      const sMu = UI.stat(stats, 'Posterior mean of $\\mu$: Gibbs', { accent: 'accent' });
      const sMuE = UI.stat(stats, 'Exact (grid)');
      const sSig = UI.stat(stats, 'Posterior mean of $\\sigma$: Gibbs / exact');
      const sR = UI.stat(stats, 'Lag-1 ACF of $\\mu$-chain');
      const sN = UI.stat(stats, 'Iterations');
      const A = 2, Bp = 2, MU0 = 0;
      let y = [], ybar, SS0, mu, s2, mus, sigs, path, grid, marg, exactMu, exactSig;
      function newData() {
        const n = Math.round(ns.get());
        y = ctx.rng.array(n, () => 2 + 1.5 * ctx.rng.normal());
        setup();
      }
      function logPost(m, sg) {
        const n = y.length, t0 = ts.get();
        const SS = SS0 + n * (ybar - m) ** 2;
        return -n * Math.log(sg) - SS / (2 * sg * sg) - (m - MU0) ** 2 / (2 * t0 * t0) - 2 * (A + 1) * Math.log(sg) - Bp / (sg * sg) + Math.log(sg);
      }
      function setup() {
        const n = y.length;
        ybar = S.mean(y);
        SS0 = y.reduce((s, v) => s + (v - ybar) ** 2, 0);
        const sd = Math.sqrt(SS0 / Math.max(1, n - 1)) || 1;
        // Pass 1: coarse box, find the high-posterior region.
        let mlo = Math.min(MU0, ybar) - 4 * sd - 1, mhi = Math.max(MU0, ybar) + 4 * sd + 1, slo = 0.02, shi = 4 * sd + 4;
        let mx = -Infinity;
        const G = 120, cells = [];
        for (let i = 0; i <= G; i++) for (let j = 0; j <= G; j++) {
          const m = mlo + (mhi - mlo) * i / G, sg = slo + (shi - slo) * j / G, v = logPost(m, sg);
          cells.push([m, sg, v]);
          if (v > mx) mx = v;
        }
        const keep = cells.filter(c => c[2] > mx - 10);
        const [a0, a1] = S.extent(keep.map(c => c[0])), [b0, b1] = S.extent(keep.map(c => c[1]));
        const pm = (a1 - a0) * 0.15 + 1e-3, ps = (b1 - b0) * 0.15 + 1e-3;
        mlo = a0 - pm; mhi = a1 + pm; slo = Math.max(1e-3, b0 - ps); shi = b1 + ps;
        pS.setDomain([mlo, mhi], [slo, shi]);
        shade(pS, bg, logPost, { contours: true });
        // Exact marginal of mu and posterior means on a fine grid.
        const GM = 200, GS = 200, ms = S.linspace(mlo, mhi, GM), ss = S.linspace(slo, shi, GS);
        let mx2 = -Infinity;
        const L = ms.map(m => ss.map(sg => { const v = logPost(m, sg); if (v > mx2) mx2 = v; return v; }));
        let Z = 0, Em = 0, Es = 0;
        marg = ms.map((m, i) => { let t = 0; ss.forEach((sg, j) => { const w = Math.exp(L[i][j] - mx2); t += w; Es += w * sg; }); Z += t; Em += t * m; return t; });
        const dm = ms[1] - ms[0];
        marg = marg.map((v, i) => [ms[i], v / (Z * dm)]);
        exactMu = Em / Z; exactSig = Es / Z;
        grid = { mlo, mhi };
        mu = ybar; s2 = sd * sd; mus = []; sigs = []; path = [[mu, Math.sqrt(s2)]];
        draw();
      }
      function iter() {
        const r = ctx.rng, n = y.length, t0 = ts.get();
        const prec = 1 / (t0 * t0) + n / s2, mean = (MU0 / (t0 * t0) + n * ybar / s2) / prec;
        mu = mean + r.normal() / Math.sqrt(prec);
        path.push([mu, Math.sqrt(s2)]);
        const SS = SS0 + n * (ybar - mu) ** 2;
        s2 = 1 / r.gamma(A + n / 2, 1 / (Bp + SS / 2));
        path.push([mu, Math.sqrt(s2)]);
        if (path.length > 80) path.splice(0, path.length - 80);
        mus.push(mu); sigs.push(Math.sqrt(s2));
      }
      function draw() {
        pS.clear();
        const k0 = Math.max(0, mus.length - 3000), pts = [];
        for (let i = k0; i < mus.length; i++) pts.push([mus[i], sigs[i]]);
        pS.points(pts, { color: 's2', r: 1.6, opacity: 0.45 });
        pS.line(path.slice(-24), { color: 'ink', width: 1.2 });
        pS.dot(mu, Math.sqrt(s2), { color: 'accent', r: 5 });
        for (const v of y) if (v >= pS.o.x[0] && v <= pS.o.x[1]) pS.segment(v, pS.o.y[0], v, pS.o.y[0] + (pS.o.y[1] - pS.o.y[0]) * 0.03, { color: 's4', width: 1.2 });
        const ymax = Math.max(...marg.map(p => p[1])) * 1.25;
        pH.setDomain([grid.mlo, grid.mhi], [0, ymax]).clear();
        if (mus.length > 20) pH.bars(S.histogram(mus, { bins: 40, lo: grid.mlo, hi: grid.mhi }), { color: 's2', opacity: 0.45 });
        pH.line(marg, { color: 'ink', width: 2 });
        pH.vline(ybar, { color: 's4' });
        pH.legend([{ label: 'exact marginal', color: 'ink' }, { label: 'Gibbs draws', color: 's2', fill: true }, { label: 'sample mean ȳ', color: 's4' }]);
        sMu.set(mus.length ? S.fmt(S.mean(mus), 4) : '—');
        sMuE.set(S.fmt(exactMu, 4));
        sSig.set(`${mus.length ? S.fmt(S.mean(sigs), 3) : '—'} / ${S.fmt(exactSig, 3)}`);
        sR.set(mus.length > 50 ? S.fmt(acf(mus.slice(-5000), 1)[1], 3) : '—');
        sN.set(String(mus.length));
      }
      newData();
    },
  });

  /* ================================================================== */
  /* mcmc-diagnostics                                                    */
  /* ================================================================== */
  Atlas.lab('mcmc-diagnostics-acf', {
    title: 'Estimating the Monte Carlo error from one run',
    blurb: 'A Gaussian AR(1) chain with $\\rho_k=\\phi^k$, so the true $\\tau=(1+\\phi)/(1-\\phi)$ is known. Left: sample ACF (bars) vs. truth, with the lag where Geyer\'s initial positive sequence stops. Right: the batch-means estimate of $\\tau$ as a function of batch size $b$: too-short batches are biased toward 1 (the IID answer); too-long batches are noisy.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ph = UI.slider(controls, { label: 'AR coefficient $\\phi$', min: -0.9, max: 0.99, step: 0.01, value: 0.9, onChange: gen });
      const ns = UI.slider(controls, { label: 'Chain length $n$', min: 500, max: 50000, value: 5000, log: true, fmt: v => String(Math.round(v)), onChange: gen });
      UI.button(controls, 'New chain', gen, { primary: true });
      stage.classList.add('two');
      const pA = new Plot(stage, { width: 420, height: 270, x: [-0.5, 60.5], y: [-1, 1.05], xlabel: 'lag k', ylabel: 'ρ̂ₖ', title: 'Sample ACF vs. truth' });
      const pB = new Plot(stage, { width: 420, height: 270, xlabel: 'batch size b (log₂ scale)', ylabel: 'τ estimate', title: 'Batch means vs. batch size', xfmt: v => String(Math.round(2 ** v)) });
      const sTrue = UI.stat(stats, 'True $\\tau$', { accent: 'accent' });
      const sG = UI.stat(stats, "Geyer $\\hat\\tau$");
      const sBM = UI.stat(stats, 'Batch means $\\hat\\tau$ ($b=\\sqrt n$)');
      const sESS = UI.stat(stats, 'ESS');
      const sSE = UI.stat(stats, 'Naive SE vs. MCSE');
      function gen() {
        const r = ctx.rng, phi = ph.get(), n = Math.round(ns.get()), c = Math.sqrt(1 - phi * phi);
        const x = new Float64Array(n);
        x[0] = r.normal();
        for (let t = 1; t < n; t++) x[t] = phi * x[t - 1] + c * r.normal();
        const tauT = (1 + phi) / (1 - phi);
        const a = acf(x, 60);
        // Geyer truncation lag (recompute to find stopping point)
        let stop = 60;
        for (let k = 0; k + 1 <= 60; k += 2) if (!(a[k] + a[k + 1] > 0)) { stop = k; break; }
        const tauG = iact(x);
        const g0 = S.variance(Array.from(x), 0);
        const bmTau = b => {
          const A = Math.floor(n / b);
          if (A < 2) return NaN;
          const ms = new Array(A).fill(0);
          for (let j = 0; j < A; j++) { let s = 0; for (let i = 0; i < b; i++) s += x[j * b + i]; ms[j] = s / b; }
          return b * S.variance(ms) / g0;
        };
        const bs = [];
        for (let e = 0; 2 ** e <= n / 4; e++) bs.push(e);
        const est = bs.map(e => [e, bmTau(2 ** e)]);
        const bsq = Math.floor(Math.sqrt(n)), tauBM = bmTau(bsq);
        pA.clear();
        pA.hline(0, { color: 'faint', dash: false, width: 1 });
        pA.band(stop - 0.5, 60.5, { color: 'faint', opacity: 0.15 });
        pA.bars(a.map((v, k) => ({ x0: k - 0.35, x1: k + 0.35, y: v })), { color: 's2', opacity: 0.6 });
        pA.line(S.range(61).map(k => [k, phi ** k]), { color: 'ink', width: 2 });
        pA.vline(stop - 0.5, { color: 'accent' });
        pA.text(stop, 0.9, 'Geyer stops', { color: 'accent', size: 11, dx: 4 });
        pA.legend([{ label: 'sample ACF', color: 's2', fill: true }, { label: 'true φᵏ', color: 'ink' }]);
        const ys = est.map(e => e[1]).filter(Number.isFinite);
        const ymax = Math.max(tauT, ...ys) * 1.15, ymin = Math.min(0, ...ys);
        pB.setDomain([-0.3, bs[bs.length - 1] + 0.3], [ymin, ymax]).clear();
        pB.hline(tauT, { color: 'ink', dash: false, width: 1.5 });
        pB.hline(1, { color: 'muted' });
        pB.vline(Math.log2(bsq), { color: 'accent' });
        pB.line(est, { color: 's1', width: 2 });
        pB.points(est, { color: 's1', r: 3.5 });
        pB.legend([{ label: 'batch-means τ̂', color: 's1' }, { label: 'true τ', color: 'ink' }, { label: 'IID value 1', color: 'muted', dash: true }, { label: 'b = √n', color: 'accent', dash: true }]);
        sTrue.set(S.fmt(tauT, 4));
        sG.set(fmt(tauG, 4));
        sBM.set(fmt(tauBM, 4), `b = ${bsq}`);
        sESS.set(Number.isFinite(tauG) ? String(Math.round(n / tauG)) : '—', `of n = ${n}`);
        const sd = Math.sqrt(g0);
        sSE.set(`${S.fmt(sd / Math.sqrt(n), 2)} vs ${fmt(sd * Math.sqrt(tauG / n), 2)}`);
      }
      gen();
    },
  });

  Atlas.lab('mcmc-diagnostics-rhat', {
    title: 'Four chains, trace plots, and R̂',
    blurb: 'Four random-walk Metropolis chains start from dispersed points. Left: traces of the monitored coordinate. Right: classic $\\hat R$ and rank-normalized split-$\\hat R$ as the run proceeds, against the 1.01 threshold. With a small proposal scale the chains take a long time to agree; on the bimodal target with a small scale they may never agree. Try a scale that lets chains hop between modes.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tsel = UI.select(controls, { label: 'Target', options: targetOptions(Object.keys(T2)), value: 'banana', onChange: reset });
      const sc = UI.slider(controls, { label: 'Proposal scale $\\sigma$', min: 0.05, max: 10, value: 0.25, log: true, fmt: v => v.toFixed(2), onChange: reset });
      const warm = UI.toggle(controls, { label: 'Discard first half as warm-up', value: true, onChange: () => { recompute(true); draw(); } });
      const row = UI.h('div', 'btn-row', controls);
      const MAXN = 4000;
      const loop = ctx.loop(() => {
        for (let k = 0; k < 10 && n < MAXN; k++) step();
        if (n % 50 === 0 || n >= MAXN) recompute(false);
        draw();
        if (n >= MAXN) return false;
      });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pT = new Plot(stage, { width: 420, height: 280, xlabel: 'iteration', ylabel: 'monitored coordinate', title: 'Trace plots' });
      const pR = new Plot(stage, { width: 420, height: 280, xlabel: 'iteration', ylabel: 'R̂', title: 'R̂ over the run' });
      const sN = UI.stat(stats, 'Iterations per chain');
      const sC = UI.stat(stats, 'Classic $\\hat R$');
      const sRk = UI.stat(stats, 'Rank-normalized split $\\hat R$', { accent: 'accent' });
      const sE = UI.stat(stats, 'Bulk ESS (all chains)');
      const sA = UI.stat(stats, 'Acceptance rate');
      const COLORS = ['s1', 's2', 's3', 's5'];
      let T, coord, X, LP, ch, n, nAcc, hist, cur;
      function reset() {
        T = T2[tsel.get()];
        coord = tsel.get() === 'funnel' ? 1 : 0;
        const cx = (T.x[0] + T.x[1]) / 2, cy = (T.y[0] + T.y[1]) / 2, hx = (T.x[1] - T.x[0]) * 0.4, hy = (T.y[1] - T.y[0]) * 0.4;
        X = [[cx - hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy], [cx + hx, cy - hy]];
        LP = X.map(p => T.lp(p[0], p[1]));
        ch = [[], [], [], []];
        n = 0; nAcc = 0; hist = []; cur = { c: NaN, r: NaN, e: NaN };
        pT.setDomain([0, MAXN], coord ? T.y : T.x);
        draw();
      }
      function step() {
        const r = ctx.rng, s = sc.get();
        for (let m = 0; m < 4; m++) {
          const y = [X[m][0] + s * r.normal(), X[m][1] + s * r.normal()], ly = T.lp(y[0], y[1]);
          if (Math.log(r.u()) < ly - LP[m]) { X[m] = y; LP[m] = ly; nAcc++; }
          ch[m].push(X[m][coord]);
        }
        n++;
      }
      function draws() { return warm.get() ? ch.map(c => c.slice(Math.floor(c.length / 2))) : ch; }
      function recompute(rebuild) {
        if (rebuild) {
          hist = [];
          for (let t = 50; t <= n; t += 50) {
            const sub = ch.map(c => c.slice(warm.get() ? Math.floor(t / 2) : 0, t));
            hist.push([t, rhatBasic(sub), rhatRank(sub)]);
          }
        } else if (n >= 50) {
          const d = draws();
          hist.push([n, rhatBasic(d), rhatRank(d)]);
        }
        if (n >= 50) { const d = draws(); cur = { c: rhatBasic(d), r: rhatRank(d), e: essMulti(split(rankNormalize(d))) }; }
      }
      function draw() {
        pT.clear();
        const stride = Math.max(1, Math.floor(n / 500));
        ch.forEach((c, m) => {
          const pts = [];
          for (let i = 0; i < c.length; i += stride) pts.push([i, c[i]]);
          pT.line(pts, { color: COLORS[m], width: 1.1, opacity: 0.85 });
        });
        if (warm.get() && n > 0) pT.band(0, n / 2, { color: 'faint', opacity: 0.18 });
        const ys = hist.flatMap(h => [h[1], h[2]]).filter(Number.isFinite);
        const ymax = Math.min(4, Math.max(1.3, ...ys) * 1.05);
        pR.setDomain([0, MAXN], [0.95, ymax]).clear();
        pR.hline(1.01, { color: 'good' });
        pR.hline(1.1, { color: 'warn' });
        pR.line(hist.map(h => [h[0], Math.min(h[1], ymax)]), { color: 's3', width: 1.8 });
        pR.line(hist.map(h => [h[0], Math.min(h[2], ymax)]), { color: 'accent', width: 2.2 });
        pR.legend([{ label: 'classic R̂', color: 's3' }, { label: 'rank-normalized split R̂', color: 'accent' }, { label: '1.01', color: 'good', dash: true }, { label: '1.1', color: 'warn', dash: true }]);
        sN.set(String(n), coord ? 'monitoring v' : 'monitoring x₁');
        sC.set(fmt(cur.c, 3));
        sRk.set(fmt(cur.r, 3), Number.isFinite(cur.r) ? (cur.r < 1.01 ? '<span style="color:var(--good)">below 1.01</span>' : '<span style="color:var(--bad)">not converged</span>') : '');
        sE.set(Number.isFinite(cur.e) ? String(Math.round(cur.e)) : '—', 'rank-normalized, Geyer truncation');
        sA.set(n ? pct(nAcc / (4 * n)) : '—');
      }
      reset();
    },
  });

  /* ================================================================== */
  /* slice-sampling                                                      */
  /* ================================================================== */
  const SL = {
    bimodal: { label: 'Two separated modes', range: [-6, 6], logZ: 0.5 * LOG2PI, pPos: 0.4,
      lf: x => lse(Math.log(0.6) - 0.5 * ((x + 3) / 0.6) ** 2 - Math.log(0.6), Math.log(0.4) - 0.5 * ((x - 3) / 0.6) ** 2 - Math.log(0.6)) },
    gamma: { label: 'Gamma(3, 1)', range: [-0.5, 12], logZ: Math.log(2), pPos: 1, lf: x => (x > 0 ? 2 * Math.log(x) - x : -Infinity) },
    spike: { label: 'Spike and slab', range: [-7, 7], logZ: 0.5 * LOG2PI, pPos: 0.5,
      lf: x => lse(Math.log(0.5) - 0.5 * (x / 0.1) ** 2 - Math.log(0.1), Math.log(0.5) - 0.5 * (x / 2) ** 2 - Math.log(2)) },
    cauchy: { label: 'Cauchy (heavy tails)', range: [-15, 15], logZ: Math.log(Math.PI), pPos: 0.5, lf: x => -Math.log1p(x * x) },
  };
  // One univariate slice update (Neal 2003: stepping out with budget m, then shrinkage), recording events.
  function sliceUpdate(lf, x0, w, rng, m = 200) {
    const ev = [];
    let evals = 0;
    const f = x => { evals++; return lf(x); };
    const ly = lf(x0) - rng.exp(1);
    let L = x0 - w * rng.u(), R = L + w;
    let J = Math.floor(m * rng.u()), K = m - 1 - J;
    ev.push({ t: 'height', ly, L, R });
    while (J > 0 && f(L) > ly) { L -= w; J--; ev.push({ t: 'step', ly, L, R }); }
    while (K > 0 && f(R) > ly) { R += w; K--; ev.push({ t: 'step', ly, L, R }); }
    const rej = [];
    for (let it = 0; it < 1000; it++) {
      const x1 = L + (R - L) * rng.u();
      if (f(x1) > ly) { ev.push({ t: 'acc', ly, L, R, x: x1, rej: rej.slice() }); return { x: x1, ev, evals }; }
      rej.push(x1);
      ev.push({ t: 'rej', ly, L, R, x: x1, rej: rej.slice() });
      if (x1 < x0) L = x1; else R = x1;
    }
    return { x: x0, ev, evals };
  }

  Atlas.lab('slice-sampling-stepout', {
    title: 'Stepping out and shrinking a slice',
    blurb: 'Animated mode shows one update at a time: draw a height under $f(x_0)$, step out in units of $w$ until both ends leave the slice (green: where $f>u$), then propose uniformly and shrink toward $x_0$ after each rejection (red). Vary $w$ over three orders of magnitude: the sampler stays exact and the cost changes only modestly. On the separated modes, a small $w$ never crosses the gap.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tsel = UI.select(controls, { label: 'Target', options: Object.keys(SL).map(k => ({ value: k, label: SL[k].label })), value: 'bimodal', onChange: reset });
      const ws = UI.slider(controls, { label: 'Initial width $w$', min: 0.05, max: 50, value: 1, log: true, fmt: v => v.toFixed(2), onChange: reset });
      const mode = UI.segmented(controls, { label: 'Mode', options: [{ value: 'anim', label: 'Animated' }, { value: 'fast', label: 'Fast' }], value: 'anim' });
      const row = UI.h('div', 'btn-row', controls);
      let acc = 0;
      const loop = ctx.loop(dt => {
        if (mode.get() === 'fast') { for (let k = 0; k < 100; k++) fullUpdate(); draw(); }
        else { acc += dt; if (acc > 0.28) { acc = 0; advance(); draw(); } }
        if (xs.length >= 50000) return false;
      });
      ctx.playButton(row, loop);
      UI.button(row, 'Next event', () => { advance(); draw(); });
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'density', title: 'The slice at the current height' });
      const pH = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'density', title: 'Samples vs. target' });
      const sN = UI.stat(stats, 'Samples');
      const sEv = UI.stat(stats, 'Evaluations per sample', { accent: 'accent' });
      const sR = UI.stat(stats, 'Lag-1 autocorrelation');
      const sP = UI.stat(stats, '$\\P(x>0)$: estimate vs truth');
      let T, x, xs, evals, cur, ei, Z, grid, ymax;
      function reset() {
        T = SL[tsel.get()];
        x = tsel.get() === 'gamma' ? 2 : tsel.get() === 'bimodal' ? -3 : 0.5;
        xs = []; evals = 0; cur = null; ei = 0;
        Z = Math.exp(T.logZ);
        grid = S.linspace(T.range[0], T.range[1], 500).map(v => [v, Math.exp(T.lf(v)) / Z]);
        ymax = Math.max(...grid.map(g => g[1])) * 1.12;
        draw();
      }
      function fullUpdate() { const r = sliceUpdate(T.lf, x, ws.get(), ctx.rng); x = r.x; xs.push(x); evals += r.evals; cur = r; ei = r.ev.length - 1; }
      function advance() {
        if (!cur || ei >= cur.ev.length - 1) { cur = sliceUpdate(T.lf, x, ws.get(), ctx.rng); cur.x0 = x; ei = 0; evals += cur.evals; return; }
        ei++;
        if (cur.ev[ei].t === 'acc') { x = cur.x; xs.push(x); }
      }
      function draw() {
        pS.setDomain(T.range, [0, ymax]).clear();
        pS.area(grid, { color: 's1', opacity: 0.12 });
        pS.line(grid, { color: 's1', width: 2 });
        if (cur) {
          const e = cur.ev[ei], x0 = cur.x0 ?? (cur.ev.length ? x : x), u = Math.exp(e.ly) / Z;
          const fx0 = Math.exp(T.lf(cur.x0 ?? x)) / Z;
          if (cur.x0 != null) pS.segment(cur.x0, 0, cur.x0, fx0, { color: 'muted', width: 1.2, dash: true });
          // where the slice is (f > u) at this height
          let run = null;
          for (const [gx, gy] of grid) {
            if (gy > u) { if (!run) run = [gx, gx]; else run[1] = gx; }
            else if (run) { pS.segment(run[0], u, run[1], u, { color: 'good', width: 6, opacity: 0.35 }); run = null; }
          }
          if (run) pS.segment(run[0], u, run[1], u, { color: 'good', width: 6, opacity: 0.35 });
          pS.segment(e.L, u, e.R, u, { color: 'ink', width: 1.8 });
          pS.segment(e.L, u - ymax * 0.03, e.L, u + ymax * 0.03, { color: 'ink', width: 1.8 });
          pS.segment(e.R, u - ymax * 0.03, e.R, u + ymax * 0.03, { color: 'ink', width: 1.8 });
          for (const r of e.rej || []) pS.dot(r, u, { color: 'bad', r: 4 });
          if (e.t === 'acc') pS.dot(e.x, u, { color: 'good', r: 5.5 });
          if (cur.x0 != null) pS.dot(cur.x0, u, { color: 'accent', r: 4.5 });
          const lab = { height: 'draw height u ~ U(0, f(x₀))', step: 'stepping out', rej: 'proposal outside slice: shrink', acc: 'accepted' }[e.t];
          pS.text(T.range[0], ymax * 0.93, lab, { dx: 8, size: 12, color: 'muted' });
        }
        pS.dot(x, 0, { color: 'accent', r: 5 });
        pH.setDomain(T.range, [0, ymax]).clear();
        if (xs.length > 10) {
          const inR = xs.filter(v => v >= T.range[0] && v <= T.range[1]);
          const h = S.histogram(inR, { bins: 60, lo: T.range[0], hi: T.range[1] });
          for (const b of h) b.density = b.count / (xs.length * (b.x1 - b.x0));
          pH.bars(h, { color: 's2', opacity: 0.5 });
        }
        pH.line(grid, { color: 'ink', width: 2 });
        sN.set(String(xs.length));
        sEv.set(xs.length ? S.fmt(evals / xs.length, 3) : '—', `w = ${ws.get().toFixed(2)}`);
        sR.set(xs.length > 30 ? S.fmt(acf(xs.slice(-5000), 1)[1], 3) : '—');
        sP.set(xs.length ? `${S.fmt(xs.filter(v => v > 0).length / xs.length, 3)} vs ${T.pPos}` : '—');
      }
      reset();
    },
  });

  /* ================================================================== */
  /* langevin-mala                                                       */
  /* ================================================================== */
  const LG = {
    gauss: { label: 'Gaussian N(0, 1)', range: [-5, 5], lp: x => -0.5 * x * x, g: x => -x },
    quartic: { label: 'Light tails ∝ exp(−x⁴/4)', range: [-3, 3], lp: x => -(x ** 4) / 4, g: x => -(x ** 3) },
    bimodal: { label: 'Bimodal mixture', range: [-5, 5],
      lp: x => lse(-0.5 * ((x + 2) / 0.6) ** 2, -0.5 * ((x - 2) / 0.6) ** 2),
      g: x => { const a = -0.5 * ((x + 2) / 0.6) ** 2, b = -0.5 * ((x - 2) / 0.6) ** 2, l = lse(a, b); return -(Math.exp(a - l) * (x + 2) + Math.exp(b - l) * (x - 2)) / 0.36; } },
    t5: { label: 'Student t₅', range: [-7, 7], lp: x => -3 * Math.log1p(x * x / 5), g: x => -6 * x / (5 + x * x) },
  };
  const lgCache = {};
  function lgMoments(k) {
    if (lgCache[k]) return lgCache[k];
    const T = LG[k], xs = S.linspace(-40, 40, 16001), dx = xs[1] - xs[0];
    let Z = 0, m2 = 0;
    for (const x of xs) { const w = Math.exp(T.lp(x)); Z += w; m2 += w * x * x; }
    return (lgCache[k] = { Z: Z * dx, v: m2 / Z });
  }

  Atlas.lab('langevin-mala-bias', {
    title: 'ULA is biased, MALA is exact',
    blurb: 'Both chains use the same Euler step $x+\\frac h2\\nabla\\log\\pi(x)+\\sqrt h\\,\\xi$. ULA (left) keeps every step; MALA (right) accepts or rejects it. On the Gaussian, ULA\'s variance is exactly $1/(1-h/4)$ and it explodes for $h\\ge4$. On the light-tailed target, a large step makes ULA blow up (it is restarted at 0 and counted). MALA stays correct at every $h$; only its acceptance rate pays.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tsel = UI.select(controls, { label: 'Target', options: Object.keys(LG).map(k => ({ value: k, label: LG[k].label })), value: 'gauss', onChange: reset });
      const hs = UI.slider(controls, { label: 'Step size $h$', min: 0.01, max: 5, value: 0.8, log: true, fmt: v => v.toFixed(3), onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(() => { for (let k = 0; k < 250; k++) step(); draw(); if (ua.length >= 40000) return false; });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pU = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'density', title: 'ULA (no correction)' });
      const pM = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'density', title: 'MALA (Metropolis-adjusted)' });
      const sVt = UI.stat(stats, 'True $\\Var_\\pi(x)$');
      const sVu = UI.stat(stats, 'ULA variance', { accent: 'accent' });
      const sVg = UI.stat(stats, 'ULA theory (Gaussian)');
      const sVm = UI.stat(stats, 'MALA variance');
      const sA = UI.stat(stats, 'MALA acceptance');
      const sB = UI.stat(stats, 'ULA blow-ups');
      let T, mom, xu, xm, ua, ma, nAcc, nM, blow, grid, ymax;
      function reset() {
        T = LG[tsel.get()]; mom = lgMoments(tsel.get());
        xu = 0; xm = 0; ua = []; ma = []; nAcc = 0; nM = 0; blow = 0;
        grid = S.linspace(T.range[0], T.range[1], 400).map(x => [x, Math.exp(T.lp(x)) / mom.Z]);
        ymax = Math.max(...grid.map(g => g[1])) * 1.4;
        draw();
      }
      function step() {
        const r = ctx.rng, h = hs.get(), sh = Math.sqrt(h);
        // ULA
        let y = xu + 0.5 * h * T.g(xu) + sh * r.normal();
        if (!(Math.abs(y) < 1e6)) { blow++; y = 0; }
        xu = y; ua.push(xu);
        // MALA
        const gx = T.g(xm), z = xm + 0.5 * h * gx + sh * r.normal(), gz = T.g(z);
        const lq = (to, from, gf) => -((to - from - 0.5 * h * gf) ** 2) / (2 * h);
        const logr = T.lp(z) - T.lp(xm) + lq(xm, z, gz) - lq(z, xm, gx);
        nM++;
        if (Math.log(r.u()) < logr) { xm = z; nAcc++; }
        ma.push(xm);
      }
      function panel(p, arr) {
        p.setDomain(T.range, [0, ymax]).clear();
        if (arr.length > 20) {
          const h = S.histogram(arr.filter(v => v >= T.range[0] && v <= T.range[1]), { bins: 60, lo: T.range[0], hi: T.range[1] });
          for (const b of h) b.density = b.count / (arr.length * (b.x1 - b.x0));
          p.bars(h, { color: 's2', opacity: 0.5 });
        }
        p.line(grid, { color: 'ink', width: 2 });
        p.legend([{ label: 'target π', color: 'ink' }, { label: 'chain', color: 's2', fill: true }]);
      }
      function draw() {
        panel(pU, ua); panel(pM, ma);
        const h = hs.get();
        sVt.set(S.fmt(mom.v, 4));
        sVu.set(ua.length > 20 ? S.fmt(S.variance(ua), 4) : '—');
        sVg.set(tsel.get() === 'gauss' ? (h < 4 ? S.fmt(1 / (1 - h / 4), 4) : '∞ (unstable)') : '—', tsel.get() === 'gauss' ? '1/(1 − h/4)' : 'Gaussian target only');
        sVm.set(ma.length > 20 ? S.fmt(S.variance(ma), 4) : '—');
        sA.set(nM ? pct(nAcc / nM) : '—', 'optimum in high d: 57.4%');
        sB.set(String(blow));
      }
      reset();
    },
  });

  /* ================================================================== */
  /* hamiltonian-mc                                                      */
  /* ================================================================== */
  Atlas.lab('hamiltonian-mc-integrator', {
    title: 'Leapfrog vs. Euler in phase space',
    blurb: 'One particle in the potential $U(q)$ with $H=U(q)+p^2/2$, started at $(1.5, 0)$, together with a small disc of nearby starting points. Explicit Euler spirals outward and inflates the disc (area $\\times(1+\\varepsilon^2)^n$ for the quadratic); leapfrog stays on a slightly distorted orbit, keeps the energy error bounded, and preserves area exactly. For the quadratic potential, leapfrog becomes unstable at $\\varepsilon=2$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const es = UI.slider(controls, { label: 'Step size $\\varepsilon$', min: 0.05, max: 2.5, step: 0.01, value: 0.4, onInput: draw });
      const ns = UI.slider(controls, { label: 'Number of steps $n$', min: 1, max: 200, step: 1, value: 40, onInput: draw });
      const pot = UI.segmented(controls, { label: 'Potential $U(q)$', options: [{ value: 'quad', label: '$q^2/2$' }, { value: 'quart', label: '$q^4/4$' }], value: 'quad', onChange: draw });
      const eu = UI.toggle(controls, { label: 'Show explicit Euler', value: true, onChange: draw });
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 300, x: [-3, 3], y: [-3, 3], equal: true, xlabel: 'q', ylabel: 'p', title: 'Phase space' });
      const pE = new Plot(stage, { width: 420, height: 300, xlabel: 'step', ylabel: 'H − H₀', title: 'Energy error along the trajectory' });
      const sLH = UI.stat(stats, 'Leapfrog max $|\\Delta H|$', { accent: 'accent' });
      const sEH = UI.stat(stats, 'Euler max $|\\Delta H|$');
      const sLA = UI.stat(stats, 'Leapfrog area ratio');
      const sEA = UI.stat(stats, 'Euler area ratio');
      const sSt = UI.stat(stats, 'Stability (quadratic)');
      const ok = v => Number.isFinite(v) && Math.abs(v) < 1e6;
      function draw() {
        const e = es.get(), n = Math.round(ns.get()), quad = pot.get() === 'quad';
        const dU = quad ? q => q : q => q * q * q, U = quad ? q => q * q / 2 : q => q ** 4 / 4, H = (q, p) => U(q) + p * p / 2;
        const leap = (q, p) => { p -= 0.5 * e * dU(q); q += e * p; p -= 0.5 * e * dU(q); return [q, p]; };
        const euler = (q, p) => [q + e * p, p - e * dU(q)];
        const run = (stepf, q, p) => { const out = [[q, p]]; for (let i = 0; i < n; i++) { [q, p] = stepf(q, p); if (!ok(q) || !ok(p)) break; out.push([q, p]); } return out; };
        const H0 = H(1.5, 0);
        const TL = run(leap, 1.5, 0), TE = run(euler, 1.5, 0);
        const disc = S.linspace(0, 2 * Math.PI, 81).slice(0, 80).map(t => [1.5 + 0.25 * Math.cos(t), 0.25 * Math.sin(t)]);
        const mapN = (stepf, pts) => pts.map(([q, p]) => { for (let i = 0; i < n; i++) { [q, p] = stepf(q, p); if (!ok(q) || !ok(p)) return [NaN, NaN]; } return [q, p]; });
        const area = pts => { if (pts.some(pt => !Number.isFinite(pt[0]))) return Infinity; let s = 0; for (let i = 0; i < pts.length; i++) { const a = pts[i], b = pts[(i + 1) % pts.length]; s += a[0] * b[1] - b[0] * a[1]; } return Math.abs(s) / 2; };
        const A0 = area(disc), DL = mapN(leap, disc), DE = mapN(euler, disc);
        pP.clear();
        pP.contour(H, [H0], { color: 'muted', width: 1.5, nx: 120, ny: 120 });
        pP.polygon(disc, { color: 'muted', opacity: 0.25 });
        if (DL.every(pt => Number.isFinite(pt[0]))) pP.polygon(DL, { color: 's1', opacity: 0.35, stroke: 's1', width: 1 });
        if (eu.get() && DE.every(pt => Number.isFinite(pt[0]))) pP.polygon(DE, { color: 's3', opacity: 0.25, stroke: 's3', width: 1 });
        if (eu.get()) { pP.line(TE, { color: 's3', width: 1.4 }); pP.points(TE, { color: 's3', r: 2 }); }
        pP.line(TL, { color: 's1', width: 1.6 });
        pP.points(TL, { color: 's1', r: 2.4 });
        pP.dot(1.5, 0, { color: 'accent', r: 5 });
        pP.legend([{ label: 'exact level set', color: 'muted' }, { label: 'leapfrog', color: 's1' }].concat(eu.get() ? [{ label: 'Euler', color: 's3' }] : []), { left: true });
        const dL = TL.map(([q, p], i) => [i, H(q, p) - H0]), dE = TE.map(([q, p], i) => [i, H(q, p) - H0]);
        const vals = dL.concat(eu.get() ? dE : []).map(d => d[1]).filter(Number.isFinite);
        const ym = Math.min(10, Math.max(0.05, ...vals.map(Math.abs))) * 1.1;
        pE.setDomain([0, Math.max(n, 1)], [-ym, ym]).clear();
        pE.hline(0, { color: 'faint', dash: false, width: 1 });
        if (eu.get()) pE.line(dE.map(d => [d[0], Math.max(-ym, Math.min(ym, d[1]))]), { color: 's3', width: 1.6 });
        pE.line(dL.map(d => [d[0], Math.max(-ym, Math.min(ym, d[1]))]), { color: 's1', width: 2 });
        const mx = arr => (arr.length < n + 1 ? Infinity : Math.max(...arr.map(d => Math.abs(d[1]))));
        sLH.set(fmt(mx(dL), 3), TL.length < n + 1 ? 'trajectory blew up' : '');
        sEH.set(fmt(mx(dE), 3));
        sLA.set(fmt(area(DL) / A0, 4), 'exactly 1 in theory');
        sEA.set(fmt(area(DE) / A0, 4), quad ? `theory (1+ε²)ⁿ = ${fmt((1 + e * e) ** n, 4)}` : '');
        sSt.set(quad ? (e < 2 ? '<span style="color:var(--good)">ε &lt; 2: stable</span>' : '<span style="color:var(--bad)">ε ≥ 2: unstable</span>') : 'amplitude-dependent', quad ? 'leapfrog needs ε < 2σ' : '');
      }
      draw();
    },
  });

  // One HMC transition with identity mass matrix. Returns the trajectory for drawing.
  function hmcStep(q, lp, grad, eps, L, rng, keepTraj) {
    const d = q.length, p = q.map(() => rng.normal());
    let lq = lp(q);
    const H0 = -lq + 0.5 * p.reduce((s, v) => s + v * v, 0);
    let qq = q.slice(), pp = p.slice(), g = grad(qq);
    const traj = keepTraj ? [qq.slice()] : null, dH = keepTraj ? [0] : null;
    let div = false, H = H0;
    for (let l = 0; l < L; l++) {
      for (let i = 0; i < d; i++) pp[i] += 0.5 * eps * g[i];
      for (let i = 0; i < d; i++) qq[i] += eps * pp[i];
      g = grad(qq);
      for (let i = 0; i < d; i++) pp[i] += 0.5 * eps * g[i];
      H = -lp(qq) + 0.5 * pp.reduce((s, v) => s + v * v, 0);
      if (keepTraj) { traj.push(qq.slice()); dH.push(H - H0); }
      if (!Number.isFinite(H) || H - H0 > 1000 || !g.every(Number.isFinite)) { div = true; break; }
    }
    const a = div ? 0 : Math.min(1, Math.exp(H0 - H));
    const acc = !div && rng.u() < a;
    return { q: acc ? qq : q, acc, a, div, traj, dH };
  }

  Atlas.lab('hamiltonian-mc-trajectory', {
    title: 'HMC trajectories, energy error, and divergences',
    blurb: 'Each iteration draws a momentum and follows $L$ leapfrog steps (dots). Green trajectories were accepted, red rejected. Right: $H-H_0$ along the latest trajectory. Long trajectories carry the chain across the target in one move with high acceptance; on the funnel, the step size that works in the mouth is unstable in the neck, and divergent transitions (red dots at their starting points) pile up exactly where the sampler fails to go.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tsel = UI.select(controls, { label: 'Target $\\pi$', options: targetOptions(Object.keys(T2)), value: 'banana', onChange: () => { setTarget(); reset(); } });
      const es = UI.slider(controls, { label: 'Step size $\\varepsilon$', min: 0.01, max: 2, value: 0.15, log: true, fmt: v => v.toFixed(3), onChange: reset });
      const Ls = UI.slider(controls, { label: 'Leapfrog steps $L$', min: 1, max: 100, step: 1, value: 25, onChange: reset });
      const jit = UI.toggle(controls, { label: 'Jitter $\\varepsilon$ by ±20%', value: false });
      const sp = UI.slider(controls, { label: 'Iterations per second', min: 1, max: 120, value: 6, log: true, fmt: v => String(Math.round(v)) });
      const row = UI.h('div', 'btn-row', controls);
      let accT = 0;
      const loop = ctx.loop(dt => {
        accT += dt;
        const per = 1 / sp.get();
        let k = 0;
        while (accT > per && k < 50) { accT -= per; iter(); k++; }
        if (k) draw();
        if (n >= 20000) return false;
      });
      ctx.playButton(row, loop);
      UI.button(row, 'One iteration', () => { iter(); draw(); });
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 300, equal: true, xlabel: 'q₁', ylabel: 'q₂', title: 'Draws and the latest trajectory' });
      const bg = underlay(pS);
      const pE = new Plot(stage, { width: 420, height: 300, xlabel: 'leapfrog step', ylabel: 'H − H₀', title: 'Energy error, latest trajectory' });
      const sAcc = UI.stat(stats, 'Mean acceptance prob.', { accent: 'accent' });
      const sDiv = UI.stat(stats, 'Divergences');
      const sE = UI.stat(stats, 'ESS of $q_1$');
      const sEg = UI.stat(stats, 'ESS per 1000 gradients');
      const sN = UI.stat(stats, 'Iterations');
      let T, q, xs, ys, n, sumA, nDiv, divPts, last, ess;
      function setTarget() { T = T2[tsel.get()]; pS.setDomain(T.x, T.y); shade(pS, bg, T.lp); }
      function reset() { q = T.start.slice(); xs = [q[0]]; ys = [q[1]]; n = 0; sumA = 0; nDiv = 0; divPts = []; last = null; ess = NaN; draw(); }
      const lp = v => T.lp(v[0], v[1]), gr = v => T.grad(v[0], v[1]);
      function iter() {
        const eps = es.get() * (jit.get() ? 0.8 + 0.4 * ctx.rng.u() : 1);
        const start = q.slice();
        const r = hmcStep(q, lp, gr, eps, Math.round(Ls.get()), ctx.rng, true);
        q = r.q; n++; sumA += r.a;
        if (r.div) { nDiv++; divPts.push(start); }
        last = r;
        xs.push(q[0]); ys.push(q[1]);
        if (n % 25 === 0) ess = xs.length / iact(xs);
      }
      function draw() {
        pS.clear();
        const k0 = Math.max(0, xs.length - 3000), pts = [];
        for (let i = k0; i < xs.length; i++) pts.push([xs[i], ys[i]]);
        pS.points(pts, { color: 's2', r: 1.8, opacity: 0.5 });
        pS.points(divPts.slice(-500), { color: 'bad', r: 2.6, opacity: 0.9 });
        pE.clear();
        if (last) {
          const c = last.acc ? 'good' : 'bad';
          pS.line(last.traj, { color: c, width: 1.6 });
          pS.points(last.traj, { color: c, r: 2.2 });
          if (last.div) { const e = last.traj[last.traj.length - 1]; pS.dot(Math.max(T.x[0], Math.min(T.x[1], e[0])), Math.max(T.y[0], Math.min(T.y[1], e[1])), { color: 'warn', r: 6 }); }
          const fin = last.dH.filter(Number.isFinite);
          const ym = Math.min(20, Math.max(0.2, ...fin.map(Math.abs))) * 1.1;
          pE.setDomain([0, Math.max(1, Math.round(Ls.get()))], [-ym, ym]).clear();
          pE.hline(0, { color: 'faint', dash: false, width: 1 });
          pE.line(last.dH.map((v, i) => [i, Math.max(-ym, Math.min(ym, v))]), { color: c, width: 2 });
          if (last.div) pE.text(last.dH.length - 1, ym * 0.85, 'divergent', { color: 'bad', anchor: 'end', size: 12, weight: 600 });
        }
        pS.dot(q[0], q[1], { color: 'accent', r: 5 });
        sAcc.set(n ? S.fmt(sumA / n, 3) : '—', 'HMC optimum in high d: 0.651');
        sDiv.set(String(nDiv), n ? pct(nDiv / n) + ' of iterations' : '');
        sE.set(Number.isFinite(ess) ? String(Math.round(ess)) : '—');
        sEg.set(Number.isFinite(ess) && n ? S.fmt(1000 * ess / (n * Math.round(Ls.get())), 3) : '—');
        sN.set(String(n));
      }
      setTarget();
      reset();
    },
  });

  /* ================================================================== */
  /* tempering                                                           */
  /* ================================================================== */
  Atlas.lab('tempering-ladder', {
    title: 'Parallel tempering on a bimodal target',
    blurb: 'Chains at temperatures $T_k$ (geometric ladder, $T_1=1$ at the bottom) each run random-walk Metropolis on $\\pi^{1/T_k}$; adjacent chains propose swaps. Hot chains hop between modes, and swaps pass their states down to the cold chain. Right: the cold chain (blue) vs. a plain random-walk chain (orange) at $T=1$. Increase the separation, set one temperature (no tempering), or make the mode widths unequal to see when it fails.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.slider(controls, { label: 'Mode separation $\\pm d$', min: 1, max: 6, step: 0.1, value: 3.5, onChange: reset });
      const Ks = UI.slider(controls, { label: 'Number of temperatures $K$', min: 1, max: 8, step: 1, value: 5, onChange: reset });
      const Tm = UI.slider(controls, { label: 'Hottest temperature $T_K$', min: 1.5, max: 1000, value: 60, log: true, fmt: v => v.toFixed(1), onChange: reset });
      const ms = UI.segmented(controls, { label: 'Modes', options: [{ value: 'eq', label: 'Equal widths' }, { value: 'uneq', label: 'Narrow + wide' }], value: 'eq', onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(() => { for (let k = 0; k < 20; k++) iter(); draw(); if (n >= 60000) return false; });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, xlabel: 'x', ylabel: 'temperature rung', title: 'Tempered targets and current states', yticks: 0 });
      const pT = new Plot(stage, { width: 420, height: 300, xlabel: 'iteration', ylabel: 'x', title: 'Cold chain vs. plain RWM' });
      const sPT = UI.stat(stats, 'Cold chain: $\\P(x>0)$', { accent: 'accent' });
      const sRW = UI.stat(stats, 'Plain RWM: $\\P(x>0)$');
      const sTr = UI.stat(stats, 'Truth $\\P(x>0)$');
      const sSw = UI.stat(stats, 'Mode switches (cold / plain)');
      const note = UI.note(stats, '');
      let d, K, betas, w, s, lp, xs, xr, n, posC, posR, swA, swN, swC, swR, trC, trR, truth;
      function reset() {
        d = ds.get(); K = Math.round(Ks.get());
        const unequal = ms.get() === 'uneq';
        w = unequal ? [0.5, 0.5] : [0.3, 0.7];
        s = unequal ? [0.15, 1.2] : [0.5, 0.5];
        const lw = w.map(Math.log), ls = s.map(Math.log);
        lp = x => lse(lw[0] - ls[0] - 0.5 * ((x + d) / s[0]) ** 2, lw[1] - ls[1] - 0.5 * ((x - d) / s[1]) ** 2);
        betas = S.range(K).map(k => (K === 1 ? 1 : Tm.get() ** (-k / (K - 1))));
        xs = betas.map(() => -d); xr = -d;
        n = 0; posC = 0; posR = 0; swA = new Array(Math.max(1, K - 1)).fill(0); swN = new Array(Math.max(1, K - 1)).fill(0);
        swC = 0; swR = 0; trC = []; trR = [];
        truth = w[1] * S.cdf.normal(d / s[1]) + w[0] * (1 - S.cdf.normal(d / s[0]));
        draw();
      }
      function rw(x, beta, sc) {
        const y = x + sc * ctx.rng.normal();
        return Math.log(ctx.rng.u()) < beta * (lp(y) - lp(x)) ? y : x;
      }
      function iter() {
        const prevC = xs[0] > 0, prevR = xr > 0;
        for (let k = 0; k < K; k++) xs[k] = rw(xs[k], betas[k], 0.5 / Math.sqrt(betas[k]));
        xr = rw(xr, 1, 0.5);
        if (K > 1) {
          const i = Math.floor(ctx.rng.u() * (K - 1)), j = i + 1;
          swN[i]++;
          if (Math.log(ctx.rng.u()) < (betas[i] - betas[j]) * (lp(xs[j]) - lp(xs[i]))) { const t = xs[i]; xs[i] = xs[j]; xs[j] = t; swA[i]++; }
        }
        n++;
        if (xs[0] > 0) posC++;
        if (xr > 0) posR++;
        if ((xs[0] > 0) !== prevC) swC++;
        if ((xr > 0) !== prevR) swR++;
        trC.push(xs[0]); trR.push(xr);
        if (trC.length > 1500) { trC.shift(); trR.shift(); }
      }
      function draw() {
        const lo = -d - 4, hi = d + 4;
        pL.setDomain([lo, hi], [-0.15, K + 0.05]).clear();
        const grid = S.linspace(lo, hi, 240), base = grid.map(lp);
        for (let k = 0; k < K; k++) {
          const v = base.map(b => betas[k] * b), mx = Math.max(...v);
          const pts = grid.map((g, i) => [g, k + 0.82 * Math.exp(v[i] - mx)]);
          pL.area(pts, { color: k === 0 ? 's1' : 's4', opacity: k === 0 ? 0.25 : 0.12, base: k });
          pL.line(pts, { color: k === 0 ? 's1' : 's4', width: 1.5 });
          pL.dot(xs[k], k + 0.82 * Math.exp(betas[k] * lp(xs[k]) - mx), { color: 'accent', r: 5 });
          pL.text(lo, k + 0.62, `T = ${S.fmt(1 / betas[k], 3)}`, { dx: 6, size: 11, color: 'muted' });
        }
        pT.setDomain([Math.max(0, n - 1500), Math.max(n, 1500)], [lo, hi]).clear();
        pT.hline(0, { color: 'faint' });
        const t0 = n - trC.length;
        pT.line(trR.map((v, i) => [t0 + i, v]), { color: 's3', width: 1.1, opacity: 0.85 });
        pT.line(trC.map((v, i) => [t0 + i, v]), { color: 's1', width: 1.3 });
        pT.legend([{ label: 'cold PT chain', color: 's1' }, { label: 'plain RWM, T = 1', color: 's3' }]);
        sPT.set(n ? S.fmt(posC / n, 3) : '—');
        sRW.set(n ? S.fmt(posR / n, 3) : '—');
        sTr.set(S.fmt(truth, 3));
        sSw.set(`${swC} / ${swR}`);
        note.set(K > 1 ? 'Swap acceptance, coldest pair first: ' + swA.map((a, i) => (swN[i] ? S.fmt(a / swN[i], 2) : '—')).join(' · ') : 'One temperature: no swaps, plain RWM.');
      }
      reset();
    },
  });

  /* ================================================================== */
  /* hierarchical-mcmc                                                   */
  /* ================================================================== */
  Atlas.lab('hierarchical-mcmc-neal', {
    title: "Neal's funnel: centered vs. non-centered",
    blurb: 'Left: the sampler works directly in $(x, v)$ with $v\\sim\\N(0,9)$, $x\\mid v\\sim\\N(0,e^v)$. Right: it works in $(z, v)$ with $x=e^{v/2}z$, $z\\sim\\N(0,1)$, an isotropic Gaussian, and the draws are mapped back. Same target, same sampler, same step size. Watch which one reaches the neck ($v\\lt -3$ has probability 0.159) and where HMC\'s divergences appear.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const kind = UI.segmented(controls, { label: 'Sampler', options: [{ value: 'rwm', label: 'RW Metropolis' }, { value: 'hmc', label: 'HMC ($L=10$)' }], value: 'hmc', onChange: reset });
      const sc = UI.slider(controls, { label: 'Step size / proposal scale', min: 0.02, max: 3, value: 0.3, log: true, fmt: v => v.toFixed(3), onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(() => { const k = kind.get() === 'hmc' ? 3 : 25; for (let i = 0; i < k; i++) iter(); draw(); if (C.v.length >= 20000) return false; });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const F = T2.funnel;
      const pC = new Plot(stage, { width: 420, height: 300, x: F.x, y: F.y, xlabel: 'x', ylabel: 'v', title: 'Centered: sample (x, v)' });
      const pN = new Plot(stage, { width: 420, height: 300, x: F.x, y: F.y, xlabel: 'x = exp(v/2)·z', ylabel: 'v', title: 'Non-centered: sample (z, v), map back' });
      shade(pC, underlay(pC), F.lp, { contours: false });
      shade(pN, underlay(pN), F.lp, { contours: false });
      const sCv = UI.stat(stats, 'Centered: $\\P(v\\lt-3)$');
      const sNv = UI.stat(stats, 'Non-centered: $\\P(v\\lt-3)$', { accent: 'accent' });
      const sTr = UI.stat(stats, 'Truth');
      sTr.set(S.fmt(S.cdf.normal(-1), 3), 'and sd(v) = 3');
      const sCs = UI.stat(stats, 'sd($v$): centered / non-centered');
      const sCd = UI.stat(stats, 'Divergences or acceptance');
      const lpC = q => F.lp(q[0], q[1]), grC = q => F.grad(q[0], q[1]);
      const lpN = q => -q[1] * q[1] / 18 - 0.5 * q[0] * q[0], grN = q => [-q[0], -q[1] / 9];
      let C, N;
      function fresh(q) { return { q, v: [], x: [], acc: 0, n: 0, div: 0, divPts: [] }; }
      function reset() { C = fresh([1, 1]); N = fresh([1, 1]); draw(); }
      function rwm(st, lp) {
        const s = sc.get(), y = [st.q[0] + s * ctx.rng.normal(), st.q[1] + s * ctx.rng.normal()];
        if (Math.log(ctx.rng.u()) < lp(y) - lp(st.q)) { st.q = y; st.acc++; }
      }
      function hmc(st, lp, gr, toX) {
        const start = st.q.slice(), r = hmcStep(st.q, lp, gr, sc.get(), 10, ctx.rng, false);
        st.q = r.q;
        if (r.acc) st.acc++;
        if (r.div) { st.div++; st.divPts.push(toX(start)); }
      }
      const idX = q => q, ncX = q => [Math.exp(q[1] / 2) * q[0], q[1]];
      function iter() {
        if (kind.get() === 'rwm') { rwm(C, lpC); rwm(N, lpN); } else { hmc(C, lpC, grC, idX); hmc(N, lpN, grN, ncX); }
        for (const [st, f] of [[C, idX], [N, ncX]]) { const xv = f(st.q); st.x.push(xv[0]); st.v.push(xv[1]); st.n++; }
      }
      function panel(p, st) {
        p.clear();
        const k0 = Math.max(0, st.v.length - 4000), pts = [];
        for (let i = k0; i < st.v.length; i++) pts.push([st.x[i], st.v[i]]);
        p.points(pts, { color: 's2', r: 1.6, opacity: 0.5 });
        p.points(st.divPts.slice(-500), { color: 'bad', r: 2.6 });
        p.hline(-3, { color: 'accent' });
      }
      function draw() {
        panel(pC, C); panel(pN, N);
        const pv = st => (st.v.length ? st.v.filter(v => v < -3).length / st.v.length : NaN);
        sCv.set(fmt(pv(C), 3), C.v.length ? `min v reached ${S.fmt(Math.min(...C.v), 3)}` : '');
        sNv.set(fmt(pv(N), 3), N.v.length ? `min v reached ${S.fmt(Math.min(...N.v), 3)}` : '');
        sCs.set(C.v.length > 2 ? `${S.fmt(S.sd(C.v), 3)} / ${S.fmt(S.sd(N.v), 3)}` : '—');
        if (kind.get() === 'hmc') sCd.set(`${C.div} / ${N.div}`, 'divergences: centered / non-centered');
        else sCd.set(C.n ? `${pct(C.acc / C.n)} / ${pct(N.acc / N.n)}` : '—', 'acceptance: centered / non-centered');
      }
      reset();
    },
  });

  const Y8 = [28, 8, -3, 7, -1, 1, 18, 12], S8 = [15, 10, 16, 11, 9, 11, 10, 18];
  function eightModel(k) {
    const J = 8, sig2 = S8.map(s => (s * k) ** 2);
    const prior = (mu, eta) => { const t2 = Math.exp(2 * eta); return [-mu * mu / 50 - Math.log1p(t2 / 25) + eta, -mu / 25, -(2 * t2 / 25) / (1 + t2 / 25) + 1]; };
    const centered = {
      lp: q => {
        const mu = q[0], eta = q[1], t2 = Math.exp(2 * eta);
        let s = prior(mu, eta)[0] - J * eta;
        for (let j = 0; j < J; j++) s -= (q[2 + j] - mu) ** 2 / (2 * t2) + (Y8[j] - q[2 + j]) ** 2 / (2 * sig2[j]);
        return s;
      },
      grad: q => {
        const mu = q[0], eta = q[1], t2 = Math.exp(2 * eta), pr = prior(mu, eta), g = new Array(10);
        let gmu = pr[1], geta = pr[2] - J;
        for (let j = 0; j < J; j++) {
          const dth = q[2 + j] - mu;
          gmu += dth / t2; geta += dth * dth / t2;
          g[2 + j] = -dth / t2 + (Y8[j] - q[2 + j]) / sig2[j];
        }
        g[0] = gmu; g[1] = geta;
        return g;
      },
      theta1: q => q[2],
    };
    const noncentered = {
      lp: q => {
        const mu = q[0], eta = q[1], tau = Math.exp(eta);
        let s = prior(mu, eta)[0];
        for (let j = 0; j < J; j++) s -= 0.5 * q[2 + j] ** 2 + (Y8[j] - mu - tau * q[2 + j]) ** 2 / (2 * sig2[j]);
        return s;
      },
      grad: q => {
        const mu = q[0], eta = q[1], tau = Math.exp(eta), pr = prior(mu, eta), g = new Array(10);
        let gmu = pr[1], geta = pr[2];
        for (let j = 0; j < J; j++) {
          const r = (Y8[j] - mu - tau * q[2 + j]) / sig2[j];
          gmu += r; geta += r * tau * q[2 + j];
          g[2 + j] = -q[2 + j] + tau * r;
        }
        g[0] = gmu; g[1] = geta;
        return g;
      },
      theta1: q => q[0] + Math.exp(q[1]) * q[2],
    };
    return { centered, noncentered };
  }

  Atlas.lab('hierarchical-mcmc-eight', {
    title: 'Eight schools: centered vs. non-centered HMC',
    blurb: 'HMC (fixed $\\varepsilon$ and $L$) on the eight-schools model with $\\mu\\sim\\N(0,5^2)$, $\\tau\\sim$ half-Cauchy$(0,5)$, run in both parameterizations at once. Each dot is a draw of $(\\theta_1,\\log\\tau)$; red dots mark divergent transitions. With the real standard errors ($k=1$) the data are weak and the centered sampler cannot enter the funnel\'s neck. Shrink the standard errors (small $k$) to make the data strong, and the verdict reverses.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ks = UI.slider(controls, { label: 'Standard-error multiplier $k$', min: 0.05, max: 2, value: 1, log: true, fmt: v => v.toFixed(2), onChange: reset });
      const es = UI.slider(controls, { label: 'Step size $\\varepsilon$', min: 0.02, max: 0.6, value: 0.2, log: true, fmt: v => v.toFixed(3), onChange: reset });
      const Ls = UI.slider(controls, { label: 'Leapfrog steps $L$', min: 3, max: 60, step: 1, value: 15, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(() => { for (let i = 0; i < 4; i++) iter(); draw(); if (C.n >= 6000) return false; });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pC = new Plot(stage, { width: 420, height: 290, x: [-25, 60], y: [-7, 4.5], xlabel: 'θ₁', ylabel: 'log τ', title: 'Centered (μ, τ, θ)' });
      const pN = new Plot(stage, { width: 420, height: 290, x: [-25, 60], y: [-7, 4.5], xlabel: 'θ₁', ylabel: 'log τ', title: 'Non-centered (μ, τ, z), θ = μ + τz' });
      const sCd = UI.stat(stats, 'Centered: divergences');
      const sNd = UI.stat(stats, 'Non-centered: divergences', { accent: 'accent' });
      const sCe = UI.stat(stats, 'ESS of $\\log\\tau$ (C / NC)');
      const sCm = UI.stat(stats, 'Mean of $\\log\\tau$ (C / NC)');
      const sCa = UI.stat(stats, 'Mean acceptance (C / NC)');
      const sN = UI.stat(stats, 'Iterations');
      let M, C, N;
      function fresh(model, q) { return { model, q, t1: [], lt: [], n: 0, div: 0, divPts: [], sa: 0 }; }
      function reset() {
        M = eightModel(ks.get());
        const mu = 5, eta = Math.log(5), th = Y8.map(y => (y + mu) / 2);
        C = fresh(M.centered, [mu, eta, ...th]);
        N = fresh(M.noncentered, [mu, eta, ...th.map(t => (t - mu) / 5)]);
        draw();
      }
      function step(st) {
        const start = st.q.slice(), r = hmcStep(st.q, st.model.lp, st.model.grad, es.get(), Math.round(Ls.get()), ctx.rng, false);
        st.q = r.q; st.n++; st.sa += r.a;
        if (r.div) { st.div++; st.divPts.push([st.model.theta1(start), start[1]]); }
        st.t1.push(st.model.theta1(st.q)); st.lt.push(st.q[1]);
      }
      function iter() { step(C); step(N); }
      function panel(p, st) {
        p.clear();
        const k0 = Math.max(0, st.lt.length - 4000), pts = [];
        for (let i = k0; i < st.lt.length; i++) pts.push([st.t1[i], st.lt[i]]);
        p.points(pts, { color: 's2', r: 1.8, opacity: 0.5 });
        p.points(st.divPts.slice(-600), { color: 'bad', r: 2.8 });
      }
      let essC = NaN, essN = NaN;
      function draw() {
        panel(pC, C); panel(pN, N);
        if (C.n % 100 === 0 && C.n > 100) { essC = C.lt.length / iact(C.lt); essN = N.lt.length / iact(N.lt); }
        if (C.n === 0) { essC = NaN; essN = NaN; }
        sCd.set(String(C.div), C.n ? pct(C.div / C.n) : '');
        sNd.set(String(N.div), N.n ? pct(N.div / N.n) : '');
        sCe.set(`${Number.isFinite(essC) ? Math.round(essC) : '—'} / ${Number.isFinite(essN) ? Math.round(essN) : '—'}`);
        sCm.set(C.n ? `${S.fmt(S.mean(C.lt), 3)} / ${S.fmt(S.mean(N.lt), 3)}` : '—');
        sCa.set(C.n ? `${S.fmt(C.sa / C.n, 2)} / ${S.fmt(N.sa / N.n, 2)}` : '—');
        sN.set(String(C.n));
      }
      reset();
    },
  });

  /* ================================================================== */
  /* variational-inference                                               */
  /* ================================================================== */
  Atlas.lab('variational-inference-meanfield', {
    title: 'Mean-field CAVI on a correlated Gaussian',
    blurb: 'Coordinate ascent updates one factor at a time (the mean path zig-zags exactly like Gibbs, contracting by $\\rho^2$ per sweep). The means converge to the truth, but the fitted factorized Gaussian (filled) uses the conditional variances $1/\\Lambda_{jj}$, so it is too narrow by $\\sqrt{1-\\rho^2}$. The dashed ellipse is the factorized fit that minimizes the forward KL instead: it matches the marginal variances.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rs = UI.slider(controls, { label: 'Correlation $\\rho$', min: -0.95, max: 0.95, step: 0.01, value: 0.85, onChange: reset });
      const s2s = UI.slider(controls, { label: 'Scale $\\sigma_2$', min: 0.3, max: 3, value: 1, log: true, fmt: v => v.toFixed(2), onChange: reset });
      const fw = UI.toggle(controls, { label: 'Show forward-KL fit', value: true, onChange: () => draw() });
      const row = UI.h('div', 'btn-row', controls);
      let acc = 0;
      const loop = ctx.loop(dt => { acc += dt; if (acc > 0.35) { acc = 0; half(); draw(); } if (hist.length > 60) return false; });
      ctx.playButton(row, loop);
      UI.button(row, 'One update', () => { half(); draw(); });
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 300, x: [-4, 4], y: [-4, 4], equal: true, xlabel: 'θ₁', ylabel: 'θ₂', title: 'Posterior (lines) and q (filled)' });
      const pK = new Plot(stage, { width: 420, height: 300, y: [1e-4, 10], logy: true, xlabel: 'coordinate update', ylabel: 'KL(q ‖ p)', title: 'KL to the posterior' });
      const sSd = UI.stat(stats, 'sd of $\\theta_1$: $q$ vs. truth', { accent: 'accent' });
      const sRt = UI.stat(stats, 'Ratio (theory $\\sqrt{1-\\rho^2}$)');
      const sKL = UI.stat(stats, 'KL$(q\\|p)$ now');
      const sFl = UI.stat(stats, 'Floor $-\\frac12\\log(1-\\rho^2)$');
      const MU = [1, 0.5];
      let Sig, Lam, m, turn, path, hist;
      function KL(mm) {
        const D = [1 / Lam[0][0], 1 / Lam[1][1]], d = [MU[0] - mm[0], MU[1] - mm[1]];
        const tr = Lam[0][0] * D[0] + Lam[1][1] * D[1];
        const quad = d[0] * (Lam[0][0] * d[0] + Lam[0][1] * d[1]) + d[1] * (Lam[1][0] * d[0] + Lam[1][1] * d[1]);
        return 0.5 * (tr + quad - 2 + Math.log(S.det(Sig) / (D[0] * D[1])));
      }
      function reset() {
        const r = rs.get(), s2 = s2s.get();
        Sig = [[1, r * s2], [r * s2, s2 * s2]];
        Lam = S.inv(Sig);
        m = [-2.5, 3]; turn = 0; path = [m.slice()]; hist = [[0, KL(m)]];
        draw();
      }
      function half() {
        if (turn === 0) m[0] = MU[0] - (Lam[0][1] / Lam[0][0]) * (m[1] - MU[1]);
        else m[1] = MU[1] - (Lam[1][0] / Lam[1][1]) * (m[0] - MU[0]);
        turn = 1 - turn;
        path.push(m.slice());
        hist.push([hist.length, KL(m)]);
      }
      function draw() {
        const r = rs.get();
        pS.clear();
        for (const k of [1, 2]) pS.ellipse(MU[0], MU[1], Sig, k, { color: 'ink', width: 1.5, opacity: k === 1 ? 0.9 : 0.5 });
        const Q = [[1 / Lam[0][0], 0], [0, 1 / Lam[1][1]]];
        pS.ellipse(m[0], m[1], Q, 2, { color: 's2', fill: true, opacity: 0.18, stroke: 's2' });
        pS.ellipse(m[0], m[1], Q, 1, { color: 's2', fill: true, opacity: 0.3, stroke: 's2' });
        if (fw.get()) pS.ellipse(MU[0], MU[1], [[Sig[0][0], 0], [0, Sig[1][1]]], 2, { color: 's3', width: 1.8, dash: true });
        pS.line(path, { color: 'accent', width: 1.4 });
        pS.points(path, { color: 'accent', r: 2.5 });
        pS.legend([{ label: 'posterior (1, 2 sd)', color: 'ink' }, { label: 'mean-field q', color: 's2', fill: true }].concat(fw.get() ? [{ label: 'forward-KL fit', color: 's3', dash: true }] : []), { left: true });
        const floor = -0.5 * Math.log(1 - r * r);
        pK.setDomain([0, Math.max(10, hist.length)], [1e-4, Math.max(10, hist[0][1] * 1.5)]).clear();
        if (floor > 1e-4) pK.hline(floor, { color: 'muted' });
        pK.line(hist.map(h => [h[0], Math.max(1e-4, h[1])]), { color: 's2', width: 2 });
        pK.points(hist.map(h => [h[0], Math.max(1e-4, h[1])]), { color: 's2', r: 3 });
        const sdq = Math.sqrt(1 / Lam[0][0]);
        sSd.set(`${S.fmt(sdq, 3)} vs ${S.fmt(1, 3)}`);
        sRt.set(S.fmt(sdq, 4), `theory ${S.fmt(Math.sqrt(1 - r * r), 4)}`);
        sKL.set(S.fmt(KL(m), 4), `${Math.floor(hist.length / 2)} sweeps; error contracts by ρ² = ${S.fmt(r * r, 3)} per sweep`);
        sFl.set(S.fmt(floor, 4), 'irreducible: q cannot represent correlation');
      }
      reset();
    },
  });

  const VT = {
    bimodal: { label: 'Bimodal (weights 0.35 / 0.65)', range: [-5, 5],
      lp: x => lse(Math.log(0.35) - 0.5 * ((x + 2) / 0.6) ** 2, Math.log(0.65) - 0.5 * ((x - 2) / 0.6) ** 2),
      g: x => { const a = Math.log(0.35) - 0.5 * ((x + 2) / 0.6) ** 2, b = Math.log(0.65) - 0.5 * ((x - 2) / 0.6) ** 2, l = lse(a, b); return -(Math.exp(a - l) * (x + 2) + Math.exp(b - l) * (x - 2)) / 0.36; } },
    gumbel: { label: 'Skewed (Gumbel)', range: [-4, 7], lp: x => -(x + Math.exp(-x)), g: x => -1 + Math.exp(-x) },
    t3: { label: 'Heavy tails (Student t₃)', range: [-8, 8], lp: x => -2 * Math.log1p(x * x / 3), g: x => -4 * x / (3 + x * x) },
    gauss: { label: 'Gaussian N(1, 1.5²)', range: [-5, 7], lp: x => -0.5 * ((x - 1) / 1.5) ** 2, g: x => -(x - 1) / 2.25 },
  };
  const vtCache = {};
  function vtMoments(k) {
    if (vtCache[k]) return vtCache[k];
    const T = VT[k], xs = S.linspace(-60, 60, 24001), dx = xs[1] - xs[0];
    let Z = 0, m1 = 0, m2 = 0;
    for (const x of xs) { const w = Math.exp(T.lp(x)); if (!Number.isFinite(w)) continue; Z += w; m1 += w * x; m2 += w * x * x; }
    const mean = m1 / Z;
    return (vtCache[k] = { logZ: Math.log(Z * dx), mean, sd: Math.sqrt(m2 / Z - mean * mean) });
  }
  function klQP(T, logZ, m, s) {
    const xs = S.linspace(m - 8 * s, m + 8 * s, 400), dx = xs[1] - xs[0];
    let kl = 0;
    for (const x of xs) {
      const lq = -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * LOG2PI, lp = T.lp(x) - logZ;
      const q = Math.exp(lq);
      if (q > 0) kl += q * (lq - (Number.isFinite(lp) ? lp : -1e6)) * dx;
    }
    return kl;
  }

  Atlas.lab('variational-inference-reparam', {
    title: 'Reparameterization-gradient VI with a Gaussian family',
    blurb: 'Fit $q=\\N(m,s^2)$ by stochastic gradient ascent on the ELBO with $K$ reparameterized samples $\\theta=m+s\\varepsilon$. On the bimodal target the fit locks onto one mode (which one depends on the start): reverse KL is mode-seeking. The dashed curve is the moment-matched Gaussian (the forward-KL optimum); its reverse KL is shown for comparison. Fewer samples $K$ mean noisier gradients and a jittery KL trace.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tsel = UI.select(controls, { label: 'Target $p$', options: Object.keys(VT).map(k => ({ value: k, label: VT[k].label })), value: 'bimodal', onChange: reset });
      const m0s = UI.slider(controls, { label: 'Initial mean $m_0$', min: -4, max: 4, step: 0.1, value: -3, onChange: reset });
      const Ks = UI.slider(controls, { label: 'Samples per gradient $K$', min: 1, max: 100, step: 1, value: 5 });
      const lrs = UI.slider(controls, { label: 'Learning rate', min: 0.002, max: 0.3, value: 0.03, log: true, fmt: v => v.toFixed(3) });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(() => { for (let i = 0; i < 2; i++) iter(); draw(); if (it >= 3000) return false; });
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 280, xlabel: 'θ', ylabel: 'density', title: 'Target and current q' });
      const pK = new Plot(stage, { width: 420, height: 280, xlabel: 'iteration', ylabel: 'KL(q ‖ p)', title: 'Exact KL (line) and K-sample estimates (dots)' });
      const sM = UI.stat(stats, '$m$, $s$', { accent: 'accent' });
      const sK = UI.stat(stats, 'KL$(q\\|p)$');
      const sF = UI.stat(stats, 'KL$(q_{\\text{moment}}\\|p)$');
      const sG = UI.stat(stats, 'Gradient noise sd ($m$)');
      const sI = UI.stat(stats, 'Iterations');
      let T, mom, m, ls, it, hist, est, gsd, grid, ymax, klF;
      function reset() {
        T = VT[tsel.get()]; mom = vtMoments(tsel.get());
        m = m0s.get(); ls = 0; it = 0; gsd = NaN;
        grid = S.linspace(T.range[0], T.range[1], 400);
        ymax = Math.max(...grid.map(x => Math.exp(T.lp(x) - mom.logZ))) * 1.5;
        klF = klQP(T, mom.logZ, mom.mean, mom.sd);
        hist = [[0, klQP(T, mom.logZ, m, Math.exp(ls))]]; est = [];
        draw();
      }
      const clip = v => Math.max(-50, Math.min(50, Number.isFinite(v) ? v : 0));
      function iter() {
        const K = Math.round(Ks.get()), s = Math.exp(ls), lr = lrs.get();
        let gm = 0, gl = 0, gm2 = 0, klh = 0;
        for (let k = 0; k < K; k++) {
          const e = ctx.rng.normal(), x = m + s * e, g = clip(T.g(x));
          gm += g; gm2 += g * g; gl += g * s * e;
          klh += (-0.5 * e * e - ls - 0.5 * LOG2PI) - (T.lp(x) - mom.logZ);
        }
        gm /= K; gl = gl / K + 1;
        gsd = K > 1 ? Math.sqrt(Math.max(0, gm2 / K - gm * gm) / K) : Math.abs(gm);
        m += lr * clip(gm);
        ls = Math.max(-6, Math.min(4, ls + lr * clip(gl)));
        it++;
        hist.push([it, klQP(T, mom.logZ, m, Math.exp(ls))]);
        est.push([it, klh / K]);
      }
      function draw() {
        const s = Math.exp(ls);
        pD.setDomain(T.range, [0, ymax]).clear();
        const dens = grid.map(x => [x, Math.exp(T.lp(x) - mom.logZ)]);
        pD.area(dens, { color: 'ink', opacity: 0.08 });
        pD.line(dens, { color: 'ink', width: 2 });
        pD.line(grid.map(x => [x, S.pdf.normal(x, mom.mean, mom.sd)]), { color: 's3', width: 1.8, dash: true });
        const qd = grid.map(x => [x, S.pdf.normal(x, m, s)]);
        pD.area(qd, { color: 's2', opacity: 0.25 });
        pD.line(qd, { color: 's2', width: 2 });
        pD.legend([{ label: 'target p', color: 'ink' }, { label: 'q (reverse KL)', color: 's2' }, { label: 'moment match (forward KL)', color: 's3', dash: true }]);
        const ex = hist.map(h => h[1]);
        const top = Math.max(0.5, ...ex.slice(0, 50), klF) * 1.2;
        pK.setDomain([0, Math.max(100, it)], [-0.1 * top, top]).clear();
        pK.hline(0, { color: 'faint', dash: false, width: 1 });
        pK.hline(klF, { color: 's3' });
        const stride = Math.max(1, Math.floor(est.length / 600));
        pK.points(est.filter((_, i) => i % stride === 0).map(e => [e[0], Math.max(-0.1 * top, Math.min(top, e[1]))]), { color: 'muted', r: 1.5, opacity: 0.5 });
        pK.line(hist.filter((_, i) => i % stride === 0 || i === hist.length - 1), { color: 's2', width: 2 });
        sM.set(`${S.fmt(m, 3)}, ${S.fmt(s, 3)}`, `target mean ${S.fmt(mom.mean, 3)}, sd ${S.fmt(mom.sd, 3)}`);
        sK.set(S.fmt(hist[hist.length - 1][1], 4));
        sF.set(S.fmt(klF, 4), 'moment-matched Gaussian');
        sG.set(fmt(gsd, 3), `K = ${Math.round(Ks.get())}`);
        sI.set(String(it));
      }
      reset();
    },
  });
})();
