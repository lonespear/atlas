/* Labs: Stochastic Processes II — Poisson processes, CTMCs, queues, martingales, Brownian motion,
   Itô calculus, time series. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const fmt = (x, d = 3) => (Number.isFinite(x) ? S.fmt(x, d) : '—');
  const pct = x => (Number.isFinite(x) ? (100 * x).toFixed(1) + '%' : '—');

  /* ================================================================== */
  /* Poisson process                                                     */
  /* ================================================================== */

  Atlas.lab('poisson-process-timeline', {
    title: 'Arrivals, counts and gaps',
    blurb: 'One realization on $[0,T]$. Bottom left: counts in windows of width $w$ against the Poisson$(\\lambda w)$ pmf; bottom right: gaps against the Exp$(\\lambda)$ density. The two Poisson constructions are indistinguishable; the renewal and clustered processes have the same mean rate but break the dispersion index $\\Var/\\text{mean}=1$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const proc = UI.select(controls, { label: 'Process', value: 'exp', onChange: gen, options: [
        { value: 'exp', label: 'Poisson: IID exponential gaps' },
        { value: 'unif', label: 'Poisson: N(T) then sorted uniforms' },
        { value: 'gamma', label: 'Renewal: Gamma gaps (regular)' },
        { value: 'cox', label: 'Cox: randomly switching rate (clustered)' },
      ] });
      const lam = UI.slider(controls, { label: 'Rate $\\lambda$', min: 0.5, max: 10, step: 0.1, value: 3, onChange: gen });
      const T = UI.slider(controls, { label: 'Horizon $T$', min: 20, max: 500, step: 10, value: 200, onChange: gen });
      const w = UI.slider(controls, { label: 'Window width $w$', min: 0.2, max: 5, step: 0.1, value: 1, onInput: draw });
      const k = UI.slider(controls, { label: 'Gamma shape $k$ (renewal only)', min: 1, max: 10, step: 0.5, value: 4, onChange: gen });
      UI.button(controls, 'New realization', gen, { primary: true });
      const pT = new Plot(stage, { width: 860, height: 150, xlabel: 't', ylabel: 'N(t)', title: 'Counting process (first 30 time units) with arrival ticks' });
      const row = UI.h('div', 'lab-stage two', stage);
      const pC = new Plot(row, { width: 420, height: 250, xlabel: 'count in window', ylabel: 'frequency', title: 'Window counts vs Poisson(λw)' });
      const pG = new Plot(row, { width: 420, height: 250, xlabel: 'gap length', ylabel: 'density', title: 'Inter-arrival gaps vs Exp(λ)' });
      const sN = UI.stat(stats, 'Events $N(T)$');
      const sL = UI.stat(stats, '$\\hat\\lambda = N(T)/T$', { accent: 'accent' });
      const sD = UI.stat(stats, 'Dispersion index $s^2/\\bar x$');
      const sCV = UI.stat(stats, 'Gap CV (Poisson: 1)');
      let times = [];
      function gen() {
        const r = ctx.rng, L = lam.get(), TT = T.get(), p = proc.get();
        times = [];
        if (p === 'exp') { let t = r.exp(L); while (t < TT) { times.push(t); t += r.exp(L); } }
        else if (p === 'unif') { const n = r.poisson(L * TT); for (let i = 0; i < n; i++) times.push(r.uniform(0, TT)); times.sort((a, b) => a - b); }
        else if (p === 'gamma') { const kk = k.get(); let t = r.gamma(kk, 1 / (kk * L)); while (t < TT) { times.push(t); t += r.gamma(kk, 1 / (kk * L)); } }
        else { // Markov-modulated: rate 0.2λ or 1.8λ, regimes of mean length 5
          let t = 0, hi = r.u() < 0.5;
          while (t < TT) {
            const end = Math.min(TT, t + r.exp(1 / 5)), rate = L * (hi ? 1.8 : 0.2);
            let s = t + r.exp(rate);
            while (s < end) { times.push(s); s += r.exp(rate); }
            t = end; hi = !hi;
          }
        }
        draw();
      }
      function draw() {
        const L = lam.get(), TT = T.get(), ww = w.get();
        // timeline
        const shown = Math.min(30, TT), early = times.filter(t => t <= shown);
        const ymax = Math.max(4, early.length + 1);
        pT.setDomain([0, shown], [0, ymax * 1.05]).clear();
        const st = [[0, 0]]; early.forEach((t, i) => st.push([t, i + 1])); st.push([shown, early.length]);
        pT.step(st, { color: 's1', width: 2 });
        pT.line([[0, 0], [shown, L * shown]], { color: 'muted', dash: true, width: 1.2 });
        for (const t of early) pT.segment(t, 0, t, ymax * 0.08, { color: 's2', width: 1.5 });
        // window counts
        const nW = Math.floor(TT / ww), counts = new Array(nW).fill(0);
        for (const t of times) { const j = Math.floor(t / ww); if (j < nW) counts[j]++; }
        const m = S.mean(counts), v = nW > 1 ? S.variance(counts) : NaN;
        const kmax = Math.max(...counts, Math.ceil(L * ww + 4 * Math.sqrt(L * ww)) + 1);
        const freq = new Array(kmax + 1).fill(0); for (const c of counts) freq[c]++;
        const bars = freq.map((f, i) => ({ x0: i - 0.4, x1: i + 0.4, y: f / nW }));
        const pmf = S.range(kmax + 1).map(i => [i, S.pmf.poisson(i, L * ww)]);
        const yC = Math.max(...bars.map(b => b.y), ...pmf.map(p => p[1])) * 1.15;
        pC.setDomain([-0.7, kmax + 0.7], [0, yC]).clear();
        pC.bars(bars, { color: 's2', opacity: 0.45 });
        pC.stems(pmf, { color: 'ink', width: 1.5, r: 3 });
        pC.legend([{ label: 'observed', color: 's2', fill: true }, { label: 'Poisson pmf', color: 'ink' }]);
        // gaps
        const gaps = times.map((t, i) => t - (i ? times[i - 1] : 0));
        const gmax = 5 / L;
        const hg = S.histogram(gaps.filter(g => g <= gmax), { bins: 30, lo: 0, hi: gmax });
        for (const b of hg) b.density = gaps.length ? b.count / (gaps.length * (b.x1 - b.x0)) : 0;
        const yG = Math.max(L, ...hg.map(b => b.density)) * 1.12;
        pG.setDomain([0, gmax], [0, yG]).clear();
        pG.bars(hg, { color: 's3', opacity: 0.45 });
        pG.line(S.linspace(0, gmax, 150).map(x => [x, S.pdf.exp(x, L)]), { color: 'ink', width: 2 });
        pG.legend([{ label: 'gaps', color: 's3', fill: true }, { label: 'Exp(λ) density', color: 'ink' }]);
        // stats
        const n = times.length;
        sN.set(String(n), `expected ${fmt(L * TT, 4)}`);
        const lo = n ? S.q.chisq(0.025, 2 * n) / (2 * TT) : 0, hi = S.q.chisq(0.975, 2 * n + 2) / (2 * TT);
        sL.set(fmt(n / TT), `95% exact CI [${fmt(lo)}, ${fmt(hi)}]`);
        sD.set(fmt(v / m), `${nW} windows; Poisson ⇒ ≈ 1`);
        sCV.set(gaps.length > 2 ? fmt(S.sd(gaps) / S.mean(gaps)) : '—');
      }
      gen();
    },
  });

  const INTENS = {
    sine: { label: 'Daily cycle: a(1 + 0.8 sin(2πt/10))', f: (t, a) => a * (1 + 0.8 * Math.sin(2 * Math.PI * t / 10)), sup: a => 1.8 * a },
    ramp: { label: 'Ageing: a (t/15)²', f: (t, a) => a * (t / 15) ** 2, sup: a => 4 * a },
    burst: { label: 'Burst: a(0.3 + 3 exp(−(t−12)²/2))', f: (t, a) => a * (0.3 + 3 * Math.exp(-((t - 12) ** 2) / 2)), sup: a => 3.3 * a },
  };
  const TH = 30;
  function cumInt(f, a) { const g = S.linspace(0, TH, 601); let s = 0; for (let i = 1; i < g.length; i++) s += 0.5 * (f(g[i - 1], a) + f(g[i], a)) * (g[i] - g[i - 1]); return s; }

  Atlas.lab('poisson-process-thinning', {
    title: 'Simulating a non-homogeneous process by thinning',
    blurb: 'Candidates are a rate-$\\lambda^*$ Poisson process; giving each a uniform height in $[0,\\lambda^*]$ and keeping those under the curve keeps each with probability $\\lambda(t)/\\lambda^*$. Run many replicates: the histogram of kept times converges to $\\lambda(t)$. Set the envelope factor below 1 and the peaks are silently clipped.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const shape = UI.select(controls, { label: 'Intensity $\\lambda(t)$', options: Object.keys(INTENS).map(k => ({ value: k, label: INTENS[k].label })), value: 'sine', onChange: reset });
      const a = UI.slider(controls, { label: 'Scale $a$', min: 0.5, max: 6, step: 0.1, value: 2, onChange: reset });
      const fac = UI.slider(controls, { label: 'Envelope $\\lambda^*/\\sup\\lambda$', min: 0.5, max: 3, step: 0.05, value: 1.2, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run replicates', '❚❚ Pause']);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pA = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'height U·λ*', title: 'One replicate: keep points under λ(t)' });
      const pB = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'kept events per unit time', title: 'Accumulated replicates vs λ(t)' });
      const sR = UI.stat(stats, 'Replicates');
      const sAcc = UI.stat(stats, 'Acceptance rate', { accent: 'accent' });
      const sM = UI.stat(stats, 'Mean count vs $\\Lambda(T)$');
      const sD = UI.stat(stats, 'Var / mean of count');
      const NB = 60;
      let hist, reps, cand, kept, counts, last;
      function replicate() {
        const I = INTENS[shape.get()], aa = a.get(), ls = I.sup(aa) * fac.get(), r = ctx.rng;
        const pts = []; let t = r.exp(ls), nk = 0;
        while (t < TH) { const u = r.uniform(0, ls), keep = u < I.f(t, aa); pts.push([t, u, keep]); if (keep) { nk++; hist[Math.min(NB - 1, Math.floor(t / TH * NB))]++; } t += r.exp(ls); }
        reps++; cand += pts.length; kept += nk; counts.push(nk); last = { pts, ls };
      }
      function reset() { loop.stop(); hist = new Array(NB).fill(0); reps = 0; cand = 0; kept = 0; counts = []; replicate(); draw(); }
      function step() { for (let i = 0; i < 25; i++) replicate(); draw(); if (reps >= 3000) return false; }
      function draw() {
        const I = INTENS[shape.get()], aa = a.get(), sup = I.sup(aa), ls = last.ls;
        const grid = S.linspace(0, TH, 300), curve = grid.map(t => [t, I.f(t, aa)]);
        const ytop = Math.max(sup, ls) * 1.1;
        pA.setDomain([0, TH], [0, ytop]).clear();
        pA.area(curve, { color: 's1', opacity: 0.12 });
        pA.line(curve, { color: 's1', width: 2 });
        pA.hline(ls, { color: 'warn', width: 1.5 });
        pA.points(last.pts.filter(p => !p[2]).map(p => [p[0], p[1]]), { color: 'muted', r: 2.2, opacity: 0.5 });
        pA.points(last.pts.filter(p => p[2]).map(p => [p[0], p[1]]), { color: 's2', r: 2.8, opacity: 0.9 });
        for (const p of last.pts) if (p[2]) pA.segment(p[0], 0, p[0], ytop * 0.04, { color: 's2', width: 1.2 });
        pA.legend([{ label: 'λ(t)', color: 's1' }, { label: 'envelope λ*', color: 'warn' }, { label: 'kept', color: 's2', fill: true, opacity: 0.9 }, { label: 'rejected', color: 'muted', fill: true }]);
        const bw = TH / NB, bars = hist.map((c, i) => ({ x0: i * bw, x1: (i + 1) * bw, y: c / (reps * bw) }));
        pB.setDomain([0, TH], [0, Math.max(sup, ...bars.map(b => b.y)) * 1.15]).clear();
        pB.bars(bars, { color: 's2', opacity: 0.45 });
        pB.line(curve, { color: 's1', width: 2 });
        if (fac.get() < 1) { pB.line(grid.map(t => [t, Math.min(I.f(t, aa), ls)]), { color: 'bad', width: 1.5, dash: true }); pB.legend([{ label: 'target λ(t)', color: 's1' }, { label: 'min(λ, λ*): what you get', color: 'bad', dash: true }]); }
        else pB.legend([{ label: 'target λ(t)', color: 's1' }, { label: 'kept times', color: 's2', fill: true }]);
        const Lam = cumInt(I.f, aa);
        sR.set(String(reps));
        sAcc.set(pct(kept / cand), `theory Λ(T)/(λ*T) = ${pct(Lam / (ls * TH))}${fac.get() < 1 ? ' (if envelope were valid)' : ''}`);
        sM.set(fmt(kept / reps), `Λ(T) = ${fmt(Lam)}`);
        sD.set(counts.length > 2 ? fmt(S.variance(counts) / S.mean(counts)) : '—', 'Poisson ⇒ 1');
      }
      reset();
    },
  });

  /* ================================================================== */
  /* CTMC                                                                */
  /* ================================================================== */

  const BD = {
    mm1: { label: 'Single server: λₙ = λ, μₙ = μ', b: (n, l) => l, d: (n, m) => (n > 0 ? m : 0), cap: Infinity },
    mminf: { label: 'Immigration–death: λₙ = λ, μₙ = nμ', b: (n, l) => l, d: (n, m) => n * m, cap: Infinity },
    fin: { label: 'Finite source N = 20: λₙ = λ(N − n), μₙ = nμ', b: (n, l) => l * Math.max(0, 20 - n), d: (n, m) => n * m, cap: 20 },
    lin: { label: 'Linear + immigration: λₙ = 1 + nλ, μₙ = nμ', b: (n, l) => 1 + n * l, d: (n, m) => n * m, cap: Infinity },
  };
  const NMAX = 3000;
  function bdStationary(M, l, m) {
    const th = [1]; let s = 1;
    for (let n = 1; n <= Math.min(M.cap, NMAX); n++) { const v = th[n - 1] * M.b(n - 1, l) / M.d(n, m); th.push(v); s += v; if (!Number.isFinite(s)) return null; }
    // Divergence check: if mass still substantial at the truncation edge, no stationary law.
    if (M.cap === Infinity && th[NMAX] / s > 1e-6) return null;
    return th.map(v => v / s);
  }

  Atlas.lab('ctmc-gillespie', {
    title: 'Birth–death chains by the Gillespie algorithm',
    blurb: 'Exact simulation: wait an Exp$(\\lambda_n+\\mu_n)$ time, then jump up with probability $\\lambda_n/(\\lambda_n+\\mu_n)$. The time-weighted occupation histogram converges to the detailed-balance solution $\\pi_n\\propto\\prod\\lambda_{k-1}/\\mu_k$. Push $\\lambda\\ge\\mu$ in the single-server or linear models to lose the stationary distribution.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const model = UI.select(controls, { label: 'Model', options: Object.keys(BD).map(k => ({ value: k, label: BD[k].label })), value: 'mminf', onChange: reset });
      const lam = UI.slider(controls, { label: 'Birth parameter $\\lambda$', min: 0.1, max: 4, step: 0.05, value: 3, onChange: reset });
      const mu = UI.slider(controls, { label: 'Death parameter $\\mu$', min: 0.1, max: 4, step: 0.05, value: 0.5, onChange: reset });
      const speed = UI.slider(controls, { label: 'Speed (time units / s)', min: 1, max: 1000, value: 40, log: true, fmt: v => v.toFixed(0) });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'X(t)', title: 'Path (last 60 time units)' });
      const pH = new Plot(stage, { width: 420, height: 270, xlabel: 'state n', ylabel: 'probability', title: 'Fraction of time in n vs π' });
      const sT = UI.stat(stats, 'Simulated time $t$');
      const sJ = UI.stat(stats, 'Jumps');
      const sTV = UI.stat(stats, 'TV distance to $\\pi$', { accent: 'accent' });
      const sM = UI.stat(stats, 'Time-avg $X$ vs $\\E_\\pi X$');
      let x, t, occ, jumps, path, pi, area;
      function reset() {
        loop.stop();
        const M = BD[model.get()];
        pi = bdStationary(M, lam.get(), mu.get());
        x = 0; t = 0; occ = new Map(); jumps = 0; path = [[0, 0]]; area = 0;
        advance(150);
        draw();
      }
      function advance(dT) {
        const M = BD[model.get()], l = lam.get(), m = mu.get(), r = ctx.rng, tEnd = t + dT;
        let guard = 0;
        while (guard++ < 20000) {
          const bR = x >= 2000 ? 0 : M.b(x, l), dR = M.d(x, m), tot = bR + dR;
          const hold = tot > 0 ? r.exp(tot) : Infinity;
          const stay = Math.min(hold, tEnd - t);
          occ.set(x, (occ.get(x) || 0) + stay); area += x * stay;
          if (t + hold > tEnd) { t = tEnd; break; }
          t += hold; x = r.u() * tot < bR ? x + 1 : x - 1; jumps++;
          path.push([t, x]);
        }
        const cut = t - 60; let i = 0; while (i < path.length - 1 && path[i + 1][0] < cut) i++;
        if (i > 0) path = path.slice(i);
      }
      function step(dt) { advance(speed.get() * dt); draw(); }
      function draw() {
        const t0 = Math.max(0, t - 60);
        const pts = path.map(p => [Math.max(p[0], t0), p[1]]); pts.push([t, x]);
        const ymax = Math.max(5, ...pts.map(p => p[1])) * 1.1;
        pP.setDomain([t0, Math.max(t, 60)], [0, ymax]).clear();
        pP.step(pts, { color: 's1', width: 1.6 });
        let mean = NaN;
        if (pi) { mean = pi.reduce((s, p, n) => s + n * p, 0); pP.hline(mean, { color: 'ink', width: 1.2 }); }
        // occupation
        let nTop = 0; for (const k of occ.keys()) nTop = Math.max(nTop, k);
        if (pi) { let c = 0, n = 0; while (n < pi.length && c < 0.999) { c += pi[n]; n++; } nTop = Math.max(nTop, n); }
        nTop = Math.min(nTop, 150);
        const bars = [];
        for (let n = 0; n <= nTop; n++) bars.push({ x0: n - 0.42, x1: n + 0.42, y: (occ.get(n) || 0) / t });
        const yH = Math.max(...bars.map(b => b.y), ...(pi ? pi.slice(0, nTop + 1) : [0])) * 1.15 || 1;
        pH.setDomain([-0.8, nTop + 0.8], [0, yH]).clear();
        pH.bars(bars, { color: 's1', opacity: 0.45 });
        if (pi) { pH.points(pi.slice(0, nTop + 1).map((p, n) => [n, p]), { color: 'ink', r: 2.8 }); pH.legend([{ label: 'time fraction', color: 's1', fill: true }, { label: 'π (detailed balance)', color: 'ink' }]); }
        else pH.text(nTop / 2, yH * 0.85, 'No stationary distribution (Σθₙ = ∞)', { anchor: 'middle', color: 'bad', size: 13, weight: 600 });
        sT.set(fmt(t, 4)); sJ.set(String(jumps));
        if (pi) { let tv = 0; const keys = new Set([...occ.keys(), ...S.range(pi.length)]); for (const k of keys) tv += Math.abs((occ.get(k) || 0) / t - (pi[k] || 0)); sTV.set(fmt(tv / 2)); }
        else sTV.set('—', 'chain is not positive recurrent');
        sM.set(fmt(area / t), pi ? `E_π X = ${fmt(mean)}` : 'drifts off');
      }
      reset();
    },
  });

  // 3x3 helpers
  const mm3 = (A, B) => A.map((r, i) => B[0].map((_, j) => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));
  function expm3(Q, t) { // uniformization
    const g = Math.max(...Q.map((r, i) => -r[i])) || 1e-9, Pt = Q.map((r, i) => r.map((v, j) => (i === j ? 1 : 0) + v / g));
    let term = [[1, 0, 0], [0, 1, 0], [0, 0, 1]], out = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], w = Math.exp(-g * t);
    const K = Math.ceil(g * t + 10 * Math.sqrt(g * t + 1) + 10);
    if (w === 0) { // large gt: scale and square
      const P1 = expm3(Q, t / 64); let P = P1; for (let i = 0; i < 6; i++) P = mm3(P, P); return P;
    }
    for (let k = 0; k <= K; k++) {
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) out[i][j] += w * term[i][j];
      term = mm3(term, Pt); w *= g * t / (k + 1);
    }
    return out;
  }

  Atlas.lab('ctmc-transient', {
    title: 'Transient behaviour: $P(t)=e^{tQ}$ and two stationary laws',
    blurb: 'A three-state chain with clockwise rates $a,b,c$ and a common counter-clockwise rate $r$, started in state 0. Curves are $P_{0j}(t)$ from the matrix exponential; dots are fractions of 400 Gillespie paths. Make the cycle strongly one-directional (small $r$) to get complex eigenvalues of $Q$ and damped oscillations — impossible for a reversible chain. Right: the time-stationary $\\pi$ vs the jump chain\'s $\\nu\\propto\\pi_jq_j$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const a = UI.slider(controls, { label: 'Rate $0\\to1$: $a$', min: 0.1, max: 5, step: 0.05, value: 3, onChange: draw });
      const b = UI.slider(controls, { label: 'Rate $1\\to2$: $b$', min: 0.1, max: 5, step: 0.05, value: 3, onChange: draw });
      const c = UI.slider(controls, { label: 'Rate $2\\to0$: $c$', min: 0.1, max: 5, step: 0.05, value: 0.6, onChange: draw });
      const r = UI.slider(controls, { label: 'Reverse rate $r$ (all)', min: 0, max: 3, step: 0.05, value: 0.05, onChange: draw });
      UI.button(controls, 'Resimulate paths', draw, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'P(X_t = j | X_0 = 0)', title: 'Transition probabilities from state 0' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-0.5, 2.5], y: [0, 1], xlabel: 'state', ylabel: 'probability', title: 'π (time) vs ν (jump chain)', xticks: 3 });
      const sEig = UI.stat(stats, 'Nonzero eigenvalues of $Q$', { accent: 'accent' });
      const sRel = UI.stat(stats, 'Relaxation time $1/|\\mathrm{Re}\\,\\lambda_2|$');
      const sQ = UI.stat(stats, 'Exit rates $q_j$');
      const note = UI.note(stats, '');
      const COLS = ['s1', 's2', 's3'];
      function draw() {
        const A = a.get(), B = b.get(), C = c.get(), Rr = r.get();
        const Q = [[-(A + Rr), A, Rr], [Rr, -(B + Rr), B], [C, Rr, -(C + Rr)]];
        const q = [A + Rr, B + Rr, C + Rr];
        // stationary: solve pi Q = 0 with normalization (replace last eq)
        const M = [[Q[0][0], Q[1][0], Q[2][0]], [Q[0][1], Q[1][1], Q[2][1]], [1, 1, 1]];
        const pi = S.solve(M, [0, 0, 1]);
        const nuRaw = pi.map((p, j) => p * q[j]), ns = S.sum(nuRaw), nu = nuRaw.map(v => v / ns);
        // eigenvalues: lambda^2 - tr lambda + E2 = 0
        const tr = Q[0][0] + Q[1][1] + Q[2][2];
        const E2 = (Q[0][0] * Q[1][1] - Q[0][1] * Q[1][0]) + (Q[0][0] * Q[2][2] - Q[0][2] * Q[2][0]) + (Q[1][1] * Q[2][2] - Q[1][2] * Q[2][1]);
        const disc = tr * tr - 4 * E2;
        let eigTxt, reMax;
        if (disc >= 0) { const l1 = (tr + Math.sqrt(disc)) / 2, l2 = (tr - Math.sqrt(disc)) / 2; eigTxt = `${fmt(l1)}, ${fmt(l2)}`; reMax = l1; }
        else { const re = tr / 2, im = Math.sqrt(-disc) / 2; eigTxt = `${fmt(re)} ± ${fmt(im)}i`; reMax = re; }
        const Tm = Math.min(12, 6 / Math.abs(reMax));
        const grid = S.linspace(0, Tm, 140), dt = grid[1];
        const Pd = expm3(Q, dt);
        let P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
        const curves = [[], [], []];
        for (const g of grid) { for (let j = 0; j < 3; j++) curves[j].push([g, P[0][j]]); P = mm3(P, Pd); }
        // Monte Carlo
        const NP = 400, tg = S.linspace(0, Tm, 25), cnt = tg.map(() => [0, 0, 0]), rng = ctx.rng;
        for (let p = 0; p < NP; p++) {
          let x = 0, t = 0, k = 0;
          for (;;) {
            const hold = rng.exp(q[x]), tn = t + hold;
            while (k < tg.length && tg[k] < tn) { cnt[k][x]++; k++; }
            if (k >= tg.length) break;
            t = tn; const u = rng.u() * q[x];
            if (x === 0) x = u < A ? 1 : 2; else if (x === 1) x = u < B ? 2 : 0; else x = u < C ? 0 : 1;
          }
        }
        pL.setDomain([0, Tm], [0, 1]).clear();
        for (let j = 0; j < 3; j++) {
          pL.hline(pi[j], { color: COLS[j], width: 1, dash: '3 4', opacity: 0.8 });
          pL.line(curves[j], { color: COLS[j], width: 2.2 });
          pL.points(tg.map((g, k) => [g, cnt[k][j] / NP]), { color: COLS[j], r: 2.6, opacity: 0.85 });
        }
        pL.legend([0, 1, 2].map(j => ({ label: `j = ${j}`, color: COLS[j] })));
        pR.clear();
        for (let j = 0; j < 3; j++) {
          pR.bars([{ x0: j - 0.38, x1: j - 0.02, y: pi[j] }], { color: 's1', opacity: 0.7 });
          pR.bars([{ x0: j + 0.02, x1: j + 0.38, y: nu[j] }], { color: 's4', opacity: 0.7 });
          pR.text(j - 0.2, pi[j], fmt(pi[j], 2), { anchor: 'middle', dy: -5, size: 11 });
          pR.text(j + 0.2, nu[j], fmt(nu[j], 2), { anchor: 'middle', dy: -5, size: 11 });
        }
        pR.legend([{ label: 'π: fraction of time', color: 's1', fill: true }, { label: 'ν: fraction of visits', color: 's4', fill: true }]);
        sEig.set(eigTxt, disc < 0 ? 'complex ⇒ oscillation' : 'real ⇒ monotone-type decay');
        sRel.set(fmt(1 / Math.abs(reMax)));
        sQ.set(q.map(v => fmt(v, 2)).join(', '));
        note.set(Rr === 0 ? 'With $r=0$ the chain cycles deterministically in order $0\\to1\\to2\\to0$, so $\\nu$ is uniform while $\\pi_j\\propto 1/q_j$.' : 'Check: $\\nu_j\\propto\\pi_j q_j$. States left quickly are visited often but occupied briefly.');
      }
      draw();
    },
  });

  /* ================================================================== */
  /* Queueing                                                            */
  /* ================================================================== */

  const H2P = 0.5 * (1 + Math.sqrt(3 / 5)); // balanced-means hyperexponential with mean 1, c^2 = 4
  const SERV = {
    exp: { label: 'Exponential ($c_s^2=1$)', draw: r => r.exp(1), m2: 2 },
    det: { label: 'Deterministic ($c_s^2=0$)', draw: () => 1, m2: 1 },
    hyp: { label: 'Hyperexponential ($c_s^2=4$)', draw: r => (r.u() < H2P ? r.exp(2 * H2P) : r.exp(2 * (1 - H2P))), m2: 5 },
  };

  Atlas.lab('queueing-mm1', {
    title: 'A single-server queue near capacity',
    blurb: 'Poisson arrivals at rate $\\lambda$, service mean 1 (so $\\rho=\\lambda$), FCFS. Watch the path and the running averages. Check Little\'s law $\\hat L\\approx\\hat\\lambda\\hat W$ and PASTA (fraction of arrivals finding the system empty $\\approx 1-\\rho$). Push $\\rho$ toward 1, and switch to highly variable service, to see Pollaczek–Khinchine at work — and how slowly the averages settle.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rho = UI.slider(controls, { label: 'Utilization $\\rho=\\lambda/\\mu$', min: 0.1, max: 0.98, step: 0.01, value: 0.8, onChange: reset });
      const serv = UI.segmented(controls, { label: 'Service distribution', options: Object.keys(SERV).map(k => ({ value: k, label: SERV[k].label })), value: 'exp', onChange: reset });
      const speed = UI.slider(controls, { label: 'Speed (service times / s)', min: 5, max: 5000, value: 100, log: true, fmt: v => v.toFixed(0) });
      const view = UI.segmented(controls, { label: 'Right panel', options: [{ value: 'dist', label: 'Queue-length law' }, { value: 'curve', label: '$L$ vs $\\rho$' }], value: 'dist', onChange: () => draw() });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'N(t)', title: 'Number in system (last 100 time units)' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'n', ylabel: 'probability', title: '' });
      const sL = UI.stat(stats, 'Time-avg $\\hat L$ vs theory', { accent: 'accent' });
      const sW = UI.stat(stats, 'Mean sojourn $\\hat W$ vs theory');
      const sLit = UI.stat(stats, "Little: $\\hat\\lambda\\hat W$");
      const sPA = UI.stat(stats, 'Arrivals finding empty');
      const sT = UI.stat(stats, 'Customers served');
      let clock, nextA, lastD, deps, N, area, occ, path, served, sumW, arrivals, emptyArr;
      function reset() {
        loop.stop();
        clock = 0; nextA = ctx.rng.exp(rho.get()); lastD = 0; deps = []; N = 0; area = 0; occ = []; path = [[0, 0]];
        served = 0; sumW = 0; arrivals = 0; emptyArr = 0;
        advance(300); draw();
      }
      // deps: sorted queue of [departureTime, arrivalTime]
      function advance(dT) {
        const lam = rho.get(), S0 = SERV[serv.get()], r = ctx.rng, tEnd = clock + dT;
        let guard = 0, head = 0;
        while (guard++ < 40000) {
          const tD = head < deps.length ? deps[head][0] : Infinity;
          const tn = Math.min(nextA, tD);
          if (tn > tEnd) break;
          const dtt = tn - clock; area += N * dtt; occ[N] = (occ[N] || 0) + dtt; clock = tn;
          if (nextA <= tD) {
            arrivals++; if (N === 0) emptyArr++;
            const start = Math.max(nextA, lastD), d = start + S0.draw(r);
            lastD = d; deps.push([d, nextA]); N++;
            nextA += r.exp(lam);
          } else { served++; sumW += deps[head][0] - deps[head][1]; head++; N--; }
          path.push([clock, N]);
        }
        const dtt = tEnd - clock; if (dtt > 0) { area += N * dtt; occ[N] = (occ[N] || 0) + dtt; clock = tEnd; }
        deps = deps.slice(head);
        const cut = clock - 100; let i = 0; while (i < path.length - 1 && path[i + 1][0] < cut) i++;
        if (i > 0) path = path.slice(i);
      }
      function step(dt) { advance(speed.get() * dt); draw(); if (served > 400000) return false; }
      function draw() {
        const p = rho.get(), S0 = SERV[serv.get()];
        const Lth = p + p * p * S0.m2 / (2 * (1 - p)), Wth = Lth / p;
        const t0 = Math.max(0, clock - 100), pts = path.map(q => [Math.max(q[0], t0), q[1]]); pts.push([clock, N]);
        const ym = Math.max(6, ...pts.map(q => q[1])) * 1.1;
        pP.setDomain([t0, Math.max(clock, 100)], [0, ym]).clear();
        pP.step(pts, { color: 's1', width: 1.5 });
        pP.hline(Lth, { color: 'ink', width: 1.2 });
        pP.text(Math.max(clock, 100), Lth, 'theory L', { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        if (view.get() === 'dist') {
          pR.o.title = 'Time in state n vs (1−ρ)ρⁿ'; pR.o.xlabel = 'n'; pR.o.ylabel = 'probability';
          const nTop = Math.min(80, Math.max(10, Math.ceil(Math.log(0.002) / Math.log(p)), occ.length - 1));
          const bars = S.range(nTop + 1).map(n => ({ x0: n - 0.42, x1: n + 0.42, y: (occ[n] || 0) / clock }));
          const yy = Math.max(1 - p, ...bars.map(b => b.y)) * 1.15;
          pR.setDomain([-0.8, nTop + 0.8], [0, yy]).clear();
          pR.bars(bars, { color: 's1', opacity: 0.45 });
          if (serv.get() === 'exp') { pR.points(S.range(nTop + 1).map(n => [n, (1 - p) * p ** n]), { color: 'ink', r: 2.6 }); pR.legend([{ label: 'simulated', color: 's1', fill: true }, { label: 'geometric (M/M/1)', color: 'ink' }]); }
          else pR.legend([{ label: 'simulated (not geometric)', color: 's1', fill: true }]);
        } else {
          pR.o.title = 'Mean number in system: Pollaczek–Khinchine'; pR.o.xlabel = 'ρ'; pR.o.ylabel = 'L';
          const ys = 50;
          pR.setDomain([0, 1], [0, ys]).clear();
          for (const [k, c] of [['det', 's3'], ['exp', 's1'], ['hyp', 's5']]) pR.line(S.linspace(0, 0.995, 200).map(x => [x, x + x * x * SERV[k].m2 / (2 * (1 - x))]), { color: c, width: serv.get() === k ? 3 : 1.4, opacity: serv.get() === k ? 1 : 0.6 });
          pR.dot(p, Math.min(area / clock, ys), { color: 'bad', r: 5 });
          pR.legend([{ label: 'deterministic', color: 's3' }, { label: 'exponential', color: 's1' }, { label: 'hyperexp. c²=4', color: 's5' }, { label: 'simulated', color: 'bad' }], { left: true });
        }
        pR.drawAxes();
        const Lhat = area / clock, What = served ? sumW / served : NaN, lamHat = arrivals / clock;
        sL.set(fmt(Lhat), `theory ${fmt(Lth)}`);
        sW.set(fmt(What), `theory ${fmt(Wth)}`);
        sLit.set(fmt(lamHat * What), `vs L̂ = ${fmt(Lhat)}`);
        sPA.set(pct(emptyArr / arrivals), `time empty ${pct((occ[0] || 0) / clock)}; 1−ρ = ${pct(1 - p)}`);
        sT.set(String(served), `t = ${fmt(clock, 5)}`);
      }
      reset();
    },
  });

  function erlangB(c, a) { let b = 1; for (let k = 1; k <= c; k++) b = a * b / (k + a * b); return b; }
  function erlangC(c, a) { if (a >= c) return 1; const b = erlangB(c, a); return c * b / (c - a * (1 - b)); }

  Atlas.lab('queueing-erlang', {
    title: 'Staffing with Erlang B and C, and the value of pooling',
    blurb: 'Left: for offered load $a=\\lambda/\\mu$ Erlangs, the waiting probability (Erlang C), blocking probability without a queue (Erlang B), and mean wait as functions of the number of servers $c$. Right: at fixed utilization $\\rho$, one pooled M/M/$c$ queue vs $c$ separate M/M/1 queues.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const a = UI.slider(controls, { label: 'Offered load $a$ (Erlangs)', min: 1, max: 60, step: 0.5, value: 20, onInput: draw });
      const c = UI.slider(controls, { label: 'Servers $c$', min: 1, max: 100, step: 1, value: 24, onInput: draw });
      const rho = UI.slider(controls, { label: 'Pooling panel: utilization $\\rho$', min: 0.5, max: 0.98, step: 0.01, value: 0.9, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'servers c', ylabel: 'probability', title: 'P(wait) [C], P(block) [B], and W_q·μ' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'number of servers pooled', ylabel: 'mean wait W_q (×1/μ)', title: 'Pooled M/M/c vs separate M/M/1', logy: true, x: [1, 50], y: [0.01, 10] });
      const sRho = UI.stat(stats, 'Utilization $\\rho=a/c$');
      const sC = UI.stat(stats, 'Erlang C: $P(\\text{wait})$', { accent: 'accent' });
      const sW = UI.stat(stats, '$W_q$ (in mean service times)');
      const sB = UI.stat(stats, 'Erlang B: $P(\\text{block})$');
      const sBeta = UI.stat(stats, 'Square-root staffing $\\beta=(c-a)/\\sqrt a$');
      function draw() {
        const A = a.get(), cc = c.get();
        const cLo = Math.max(1, Math.floor(A) + 1), cHi = Math.max(cLo + 6, Math.ceil(A + 4 * Math.sqrt(A) + 4));
        pL.setDomain([cLo - 0.5, cHi + 0.5], [0, 1.02]).clear();
        const cs = S.range(cHi - cLo + 1).map(i => cLo + i);
        pL.line(cs.map(k => [k, erlangC(k, A)]), { color: 's1', width: 2.2 });
        pL.points(cs.map(k => [k, erlangC(k, A)]), { color: 's1', r: 2.2 });
        pL.line(cs.map(k => [k, erlangB(k, A)]), { color: 's2', width: 2.2 });
        pL.line(cs.map(k => [k, Math.min(1.02, erlangC(k, A) / (k - A))]), { color: 's4', width: 1.8, dash: true });
        if (cc >= cLo - 0.5 && cc <= cHi + 0.5) pL.vline(cc, { color: 'ink', width: 1.2 });
        pL.legend([{ label: 'Erlang C: P(wait)', color: 's1' }, { label: 'Erlang B: P(block)', color: 's2' }, { label: 'W_q · μ (clipped at 1)', color: 's4', dash: true }]);
        // pooling
        const p = rho.get(), ks = S.range(50).map(i => i + 1);
        const sep = p / (1 - p);
        const pooled = ks.map(k => [k, erlangC(k, k * p) / (k * (1 - p))]);
        pR.setDomain([1, 50], [Math.max(1e-4, Math.min(...pooled.map(q => q[1])) * 0.5), sep * 2]).clear();
        pR.line([[1, sep], [50, sep]], { color: 's2', width: 2.2 });
        pR.line(pooled, { color: 's1', width: 2.2 });
        pR.legend([{ label: 'c separate M/M/1 queues', color: 's2' }, { label: 'one pooled M/M/c', color: 's1' }]);
        if (cc <= A) {
          sRho.set(fmt(A / cc), '<span style="color:var(--bad)">ρ ≥ 1: unstable</span>');
          sC.set('1'); sW.set('∞');
        } else {
          const C = erlangC(cc, A);
          sRho.set(fmt(A / cc)); sC.set(fmt(C)); sW.set(fmt(C / (cc - A)));
        }
        sB.set(fmt(erlangB(cc, A)), 'loss system, any service law');
        sBeta.set(fmt((cc - A) / Math.sqrt(A)));
      }
      draw();
    },
  });

  /* ================================================================== */
  /* Martingales                                                         */
  /* ================================================================== */

  Atlas.lab('martingales-betting', {
    title: 'Betting systems against a fair coin',
    blurb: 'Each session plays even-money bets until the strategy stops or the bankroll $2^m-1$ cannot cover the next stake. Doubling wins $+1$ almost always and occasionally loses everything; flat betting toward a target is gambler\'s ruin. In a fair game the mean net gain stays at 0 — optional stopping with a bounded stopping time. Wald\'s identity predicts the mean gain as $(2p-1)\\times$ mean total amount staked, for any strategy.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const strat = UI.segmented(controls, { label: 'Strategy', options: [{ value: 'dbl', label: 'Doubling' }, { value: 'flat', label: 'Flat bets to a target' }], value: 'dbl', onChange: reset });
      const m = UI.slider(controls, { label: 'Bankroll $2^m-1$: $m$', min: 2, max: 12, step: 1, value: 6, onChange: reset, fmt: v => `${v} (${2 ** v - 1} units)` });
      const p = UI.slider(controls, { label: 'Win probability $p$', min: 0.4, max: 0.6, step: 0.001, value: 0.5, onChange: reset });
      const W = UI.slider(controls, { label: 'Flat: target profit', min: 1, max: 40, step: 1, value: 10, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run sessions', '❚❚ Pause']);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, xlabel: 'round', ylabel: 'net gain', title: 'First 25 sessions' });
      const pH = new Plot(stage, { width: 420, height: 270, xlabel: 'final net gain', ylabel: 'fraction of sessions', title: 'Distribution of the outcome' });
      const sN = UI.stat(stats, 'Sessions');
      const sWin = UI.stat(stats, 'P(session ends in profit)');
      const sMean = UI.stat(stats, 'Mean net gain ± 2 SE', { accent: 'accent' });
      const sWald = UI.stat(stats, 'Wald: $(2p-1)\\times$ mean staked');
      const sTh = UI.stat(stats, 'Theory');
      const MAXR = 4000;
      let outcomes, paths, nS, sumG, sumG2, sumStake, wins;
      function session(keepPath) {
        const r = ctx.rng, pp = p.get(), bank = 2 ** m.get() - 1, isD = strat.get() === 'dbl', tgt = W.get();
        let g = 0, stake = 1, staked = 0, k = 0;
        const path = keepPath ? [[0, 0]] : null;
        while (k < MAXR) {
          if (isD) { if (stake > bank + g) break; } else if (g <= -bank || g >= tgt) break;
          k++; staked += stake;
          if (r.u() < pp) { g += stake; if (isD) { if (path) path.push([k, g]); break; } }
          else { g -= stake; if (isD) stake *= 2; }
          if (path) path.push([k, g]);
        }
        return { g, staked, path };
      }
      function reset() { loop.stop(); outcomes = new Map(); paths = []; nS = 0; sumG = 0; sumG2 = 0; sumStake = 0; wins = 0; for (let i = 0; i < 200; i++) add(); draw(); }
      function add() {
        const s = session(paths.length < 25);
        if (s.path) paths.push(s.path);
        nS++; sumG += s.g; sumG2 += s.g * s.g; sumStake += s.staked; if (s.g > 0) wins++;
        outcomes.set(s.g, (outcomes.get(s.g) || 0) + 1);
      }
      function step() { for (let i = 0; i < 150; i++) add(); draw(); if (nS >= 100000) return false; }
      function draw() {
        const bank = 2 ** m.get() - 1, pp = p.get(), isD = strat.get() === 'dbl', tgt = W.get();
        const xm = Math.max(10, ...paths.map(q => q[q.length - 1][0]));
        pP.setDomain([0, xm], [-bank * 1.05, Math.max(isD ? 2 : tgt * 1.1, 2)]).clear();
        pP.hline(0, { color: 'muted', width: 1 });
        paths.forEach((q, i) => pP.line(q, { color: q[q.length - 1][1] > 0 ? 's1' : 'bad', width: 1.3, opacity: 0.75 }));
        const vals = [...outcomes.keys()].sort((a, b) => a - b);
        const lo = Math.min(-bank, ...vals), hi = Math.max(isD ? 1 : tgt, ...vals);
        const pts = vals.map(v => [v, outcomes.get(v) / nS]);
        pH.setDomain([lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.05], [0, 1.05]).clear();
        pH.stems(pts, { color: 's2', width: 2.5, r: 3.5 });
        pts.filter(q => q[1] > 0.002).slice(0, 12).forEach(q => pH.text(q[0], q[1], (100 * q[1]).toFixed(q[1] < 0.01 ? 2 : 1) + '%', { anchor: q[0] > (lo + hi) / 2 ? 'end' : 'start', dx: q[0] > (lo + hi) / 2 ? -6 : 6, dy: -4, size: 11 }));
        const mean = sumG / nS, se = Math.sqrt(Math.max(0, sumG2 / nS - mean * mean) / nS);
        sN.set(String(nS));
        sWin.set(pct(wins / nS));
        sMean.set(`${fmt(mean)} ± ${fmt(2 * se, 2)}`);
        sWald.set(fmt((2 * pp - 1) * sumStake / nS), `mean staked ${fmt(sumStake / nS)}`);
        if (isD) {
          const ql = (1 - pp) ** m.get();
          sTh.set(fmt((1 - ql) - ql * bank), `P(ruin) = ${(1 - pp) ** m.get() < 1e-3 ? ((1 - pp) ** m.get()).toExponential(2) : fmt(ql)}`);
        } else {
          let P;
          if (Math.abs(pp - 0.5) < 1e-9) P = bank / (bank + tgt);
          else { const rr = (1 - pp) / pp; P = (1 - rr ** bank) / (1 - rr ** (bank + tgt)); }
          sTh.set(fmt(tgt * P - bank * (1 - P)), `P(reach target) = ${fmt(P)} (ignoring the ${MAXR}-round cap)`);
        }
      }
      reset();
    },
  });

  Atlas.lab('martingales-polya', {
    title: "Pólya's urn: a martingale with a random limit",
    blurb: 'Start with $a$ red and $b$ blue; draw a ball, return it with another of its colour. The red fraction is a bounded martingale, so each urn converges — but to a different limit. Left: 30 urns. Right: the fraction after $n$ draws across 2000 urns, against the Beta$(a,b)$ law of the limit. Its mean never moves from $a/(a+b)$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const a = UI.slider(controls, { label: 'Initial red $a$', min: 1, max: 10, step: 1, value: 1, onChange: run });
      const b = UI.slider(controls, { label: 'Initial blue $b$', min: 1, max: 10, step: 1, value: 1, onChange: run });
      const n = UI.slider(controls, { label: 'Draws $n$', min: 10, max: 2000, value: 500, log: true, fmt: v => v.toFixed(0), onChange: run });
      UI.button(controls, 'New urns', run, { primary: true });
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, x: [0, 500], y: [0, 1], xlabel: 'draw', ylabel: 'red fraction', title: 'Thirty urns' });
      const pH = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 3], xlabel: 'red fraction after n draws', ylabel: 'density', title: 'Across 2000 urns vs Beta(a, b)' });
      const sM = UI.stat(stats, 'Mean red fraction', { accent: 'accent' });
      const sV = UI.stat(stats, 'Variance vs Beta limit');
      const sC = UI.stat(stats, 'Avg |change| over last half');
      function run() {
        const A = a.get(), B = b.get(), N = Math.round(n.get()), r = ctx.rng, M = 2000;
        const fin = new Array(M), paths = [];
        let move = 0;
        const stride = Math.max(1, Math.floor(N / 300));
        for (let u = 0; u < M; u++) {
          let red = A, tot = A + B, mid = 0;
          const path = u < 30 ? [[0, red / tot]] : null;
          for (let k = 1; k <= N; k++) {
            if (r.u() * tot < red) red++;
            tot++;
            if (path && (k % stride === 0 || k === N)) path.push([k, red / tot]);
            if (k === Math.floor(N / 2)) mid = red / tot;
          }
          fin[u] = red / tot; move += Math.abs(fin[u] - mid);
          if (path) paths.push(path);
        }
        pP.setDomain([0, N], [0, 1]).clear();
        paths.forEach((q, i) => pP.line(q, { color: `s${(i % 6) + 1}`, width: 1.2, opacity: 0.75 }));
        pP.hline(A / (A + B), { color: 'ink', width: 1.2 });
        const h = S.histogram(fin, { bins: 40, lo: 0, hi: 1 });
        const g = S.linspace(0.001, 0.999, 300).map(x => [x, S.pdf.beta(x, A, B)]);
        const ym = Math.min(8, Math.max(...h.map(q => q.density), ...g.map(q => q[1])) * 1.1);
        pH.setDomain([0, 1], [0, ym]).clear();
        pH.bars(h, { color: 's2', opacity: 0.5 });
        pH.line(g, { color: 'ink', width: 2 });
        pH.legend([{ label: 'simulated', color: 's2', fill: true }, { label: 'Beta(a, b)', color: 'ink' }]);
        sM.set(fmt(S.mean(fin)), `a/(a+b) = ${fmt(A / (A + B))}`);
        sV.set(fmt(S.variance(fin)), `Beta: ${fmt(A * B / ((A + B) ** 2 * (A + B + 1)))}`);
        sC.set(fmt(move / M), 'shrinks as n grows: convergence');
      }
      run();
    },
  });

  /* ================================================================== */
  /* Brownian motion                                                     */
  /* ================================================================== */

  const STEPS = {
    rad: { label: 'Rademacher ±1', d: r => (r.u() < 0.5 ? -1 : 1) },
    unif: { label: 'Uniform(−√3, √3)', d: r => r.uniform(-Math.sqrt(3), Math.sqrt(3)) },
    exp: { label: 'Exp(1) − 1 (skewed)', d: r => r.exp(1) - 1 },
    t3: { label: 'Student t₃ / √3 (heavy-ish)', d: r => r.t(3) / Math.sqrt(3) },
  };

  Atlas.lab('brownian-motion-donsker', {
    title: "Donsker: random walks become Brownian motion",
    blurb: 'Rescaled walks $S_{\\lfloor nt\\rfloor}/\\sqrt n$ with unit-variance steps. Right: the running maximum over $[0,1]$ across 3000 walks against its Brownian limit, the half-normal density $2\\varphi(m)$ from the reflection principle. Small $n$ and skewed steps show the discrete undershoot; large $n$ forgets the step law entirely.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const st = UI.select(controls, { label: 'Step distribution', options: Object.keys(STEPS).map(k => ({ value: k, label: STEPS[k].label })), value: 'rad', onChange: run });
      const n = UI.slider(controls, { label: 'Steps $n$', min: 4, max: 2000, value: 16, log: true, fmt: v => v.toFixed(0), onChange: run });
      UI.button(controls, 'New walks', run, { primary: true });
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-3, 3], xlabel: 't', ylabel: 'W_n(t)', title: 'Ten rescaled walks' });
      const pH = new Plot(stage, { width: 420, height: 270, x: [0, 3.5], y: [0, 1], xlabel: 'max of W_n over [0,1]', ylabel: 'density', title: 'Running maximum vs |N(0,1)|' });
      const sP = UI.stat(stats, '$P(M\\ge1)$', { accent: 'accent' });
      const sE = UI.stat(stats, '$\\E M$');
      const sZ = UI.stat(stats, '$P(M=0)$');
      const sKS = UI.stat(stats, 'KS distance to half-normal');
      function run() {
        const N = Math.round(n.get()), D = STEPS[st.get()].d, r = ctx.rng, M = 3000, sq = Math.sqrt(N);
        const mx = new Array(M), paths = [];
        for (let u = 0; u < M; u++) {
          let s = 0, best = 0;
          const path = u < 10 ? [[0, 0]] : null;
          for (let k = 1; k <= N; k++) { s += D(r); if (s > best) best = s; if (path) path.push([k / N, s / sq]); }
          mx[u] = best / sq; if (path) paths.push(path);
        }
        pP.clear();
        pP.hline(0, { color: 'muted', width: 1 });
        paths.forEach((q, i) => pP.line(q, { color: `s${(i % 6) + 1}`, width: 1.3, opacity: 0.85 }));
        const pos = mx.filter(v => v > 1e-12);
        const h = S.histogram(pos, { bins: 35, lo: 0, hi: 3.5 });
        for (const b of h) b.density = b.count / (M * (b.x1 - b.x0));
        pH.setDomain([0, 3.5], [0, Math.max(0.9, ...h.map(b => b.density)) * 1.1]).clear();
        pH.bars(h, { color: 's2', opacity: 0.5 });
        pH.line(S.linspace(0, 3.5, 200).map(x => [x, 2 * S.pdf.normal(x)]), { color: 'ink', width: 2 });
        pH.legend([{ label: 'walk maxima (M > 0)', color: 's2', fill: true }, { label: '2φ(m), Brownian limit', color: 'ink' }]);
        const srt = mx.slice().sort((x, y) => x - y);
        let ks = 0; srt.forEach((v, i) => { const F = 2 * S.cdf.normal(v) - 1; ks = Math.max(ks, Math.abs((i + 1) / M - F), Math.abs(i / M - F)); });
        sP.set(fmt(mx.filter(v => v >= 1).length / M), `limit 2(1−Φ(1)) = ${fmt(2 * (1 - S.cdf.normal(1)))}`);
        sE.set(fmt(S.mean(mx)), `limit √(2/π) = ${fmt(Math.sqrt(2 / Math.PI))}`);
        sZ.set(fmt((M - pos.length) / M), 'walk never goes above 0; → 0 like n^(−1/2)');
        sKS.set(fmt(ks));
      }
      run();
    },
  });

  Atlas.lab('brownian-motion-qv', {
    title: 'Quadratic variation and $p$-variation of one path',
    blurb: 'One Brownian path on $[0,T]$ sampled at $2^{14}$ points. Sum $|\\Delta B|^p$ over partitions with $2^k$ intervals. For $p=2$ the sum settles at $T$; for $p< 2$ it diverges like $n^{1-p/2}$ (infinite length when $p=1$); for $p>2$ it vanishes. A smooth curve has zero quadratic variation and finite length.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const kind = UI.segmented(controls, { label: 'Path', options: [{ value: 'bm', label: 'Brownian' }, { value: 'drift', label: 'Brownian + drift 3t' }, { value: 'smooth', label: 'Smooth curve' }], value: 'bm', onChange: gen });
      const T = UI.slider(controls, { label: 'Horizon $T$', min: 0.25, max: 4, step: 0.25, value: 1, onChange: gen });
      const pv = UI.slider(controls, { label: 'Exponent $p$ (orange curve)', min: 0.5, max: 3, step: 0.05, value: 1, onInput: draw });
      UI.button(controls, 'New path', gen, { primary: true });
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'B(t)', title: 'The path' });
      const pV = new Plot(stage, { width: 420, height: 270, x: [0, 14], y: [0.01, 100], logy: true, xlabel: 'log₂(number of intervals)', ylabel: 'Σ|ΔB|^p', title: 'Variation sums as the mesh shrinks' });
      const sQ = UI.stat(stats, 'Quadratic variation at $2^{14}$', { accent: 'accent' });
      const sV = UI.stat(stats, 'Total variation ($p=1$) at $2^{14}$');
      const sP = UI.stat(stats, '$p$-variation at $2^{14}$');
      const K = 14, NF = 2 ** K;
      let x = [];
      function gen() {
        const TT = T.get(), r = ctx.rng, dt = TT / NF, k = kind.get();
        x = new Float64Array(NF + 1);
        if (k === 'smooth') for (let i = 0; i <= NF; i++) { const t = i * dt; x[i] = Math.sqrt(TT) * (Math.sin(2 * Math.PI * t / TT) + 0.5 * Math.sin(6 * Math.PI * t / TT)); }
        else { const sd = Math.sqrt(dt), mu = k === 'drift' ? 3 : 0; for (let i = 1; i <= NF; i++) x[i] = x[i - 1] + mu * dt + sd * r.normal(); }
        draw();
      }
      function vsum(p, k) { const step = NF >> k; let s = 0; for (let i = step; i <= NF; i += step) s += Math.abs(x[i] - x[i - step]) ** p; return s; }
      function draw() {
        const TT = T.get(), p = pv.get(), dt = TT / NF;
        const ex = S.extent(x);
        pP.setDomain([0, TT], [ex[0] - 0.1, ex[1] + 0.1]).clear();
        const pts = []; for (let i = 0; i <= NF; i += 4) pts.push([i * dt, x[i]]);
        pP.line(pts, { color: 's1', width: 1.2 });
        const ks = S.range(K + 1);
        const qv = ks.map(k => [k, vsum(2, k)]), tv = ks.map(k => [k, vsum(1, k)]), pp = ks.map(k => [k, vsum(p, k)]);
        const all = [...qv, ...tv, ...pp].map(q => q[1]).filter(v => v > 0);
        const lo = Math.max(1e-6, Math.min(...all) / 2), hi = Math.max(...all) * 2;
        pV.setDomain([0, K], [lo, hi]).clear();
        if (kind.get() !== 'smooth') pV.hline(TT, { color: 'ink', width: 1.2 });
        pV.line(qv, { color: 's1', width: 2.4 }); pV.points(qv, { color: 's1', r: 2.5 });
        pV.line(tv, { color: 's2', width: 2.4 }); pV.points(tv, { color: 's2', r: 2.5 });
        pV.line(pp, { color: 's4', width: 1.8, dash: true });
        pV.legend([{ label: 'p = 2 (quadratic)', color: 's1' }, { label: 'p = 1 (length)', color: 's2' }, { label: `p = ${p.toFixed(2)}`, color: 's4', dash: true }, { label: 'T', color: 'ink' }], { left: true });
        sQ.set(fmt(qv[K][1]), kind.get() === 'smooth' ? 'smooth ⇒ → 0' : `T = ${TT}`);
        sV.set(fmt(tv[K][1]), kind.get() === 'smooth' ? 'finite: the arc length in t' : `≈ √(2nT/π) = ${fmt(Math.sqrt(2 * NF * TT / Math.PI))}`);
        sP.set(fmt(pp[K][1]));
      }
      gen();
    },
  });

  /* ================================================================== */
  /* Itô calculus                                                        */
  /* ================================================================== */

  Atlas.lab('stochastic-calculus-riemann', {
    title: 'Left, right and midpoint sums for $\\int_0^T B\\,dB$',
    blurb: 'Left: on one path, the three Riemann sums as the partition is refined converge to three different limits, $\\tfrac12(B_T^2\\mp T)$ and $\\tfrac12B_T^2$. Right: across 3000 paths, the left (Itô) sums have mean 0 and variance $T^2/2$ (Itô isometry), matching the law of $\\tfrac T2(\\chi^2_1-1)$; the right sums are shifted by $T$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const T = UI.slider(controls, { label: 'Horizon $T$', min: 0.5, max: 4, step: 0.25, value: 2, onChange: () => { gen(); mc(); } });
      const k = UI.slider(controls, { label: 'Right panel: $2^k$ intervals, $k$', min: 1, max: 10, step: 1, value: 3, onChange: mc });
      UI.button(controls, 'New path', () => { gen(); mc(); }, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 14], y: [-2, 2], xlabel: 'log₂(number of intervals)', ylabel: 'Riemann sum', title: 'One path: three limits' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-2, 6], y: [0, 1], xlabel: 'value of the sum', ylabel: 'density', title: 'Many paths: left vs right sums' });
      const sB = UI.stat(stats, '$B_T$ on this path');
      const sI = UI.stat(stats, 'Itô limit $\\tfrac12(B_T^2-T)$', { accent: 'accent' });
      const sML = UI.stat(stats, 'Mean / var of left sums');
      const sMR = UI.stat(stats, 'Mean of right sums');
      const K = 14, NF = 2 ** K;
      let b;
      function gen() {
        const TT = T.get(), sd = Math.sqrt(TT / NF), r = ctx.rng;
        b = new Float64Array(NF + 1); for (let i = 1; i <= NF; i++) b[i] = b[i - 1] + sd * r.normal();
        const L = [], Rr = [], M = [];
        for (let kk = 0; kk <= K; kk++) {
          const step = NF >> kk; let l = 0, rr = 0;
          for (let i = step; i <= NF; i += step) { const d = b[i] - b[i - step]; l += b[i - step] * d; rr += b[i] * d; }
          L.push([kk, l]); Rr.push([kk, rr]); M.push([kk, 0.5 * (l + rr)]);
        }
        const BT = b[NF], all = [...L, ...Rr].map(q => q[1]).concat([0.5 * (BT * BT - TT), 0.5 * (BT * BT + TT)]);
        const ex = S.extent(all), pad = (ex[1] - ex[0]) * 0.1 + 0.1;
        pL.setDomain([0, K], [ex[0] - pad, ex[1] + pad]).clear();
        pL.hline(0.5 * (BT * BT - TT), { color: 's1', width: 1.2 });
        pL.hline(0.5 * (BT * BT + TT), { color: 's2', width: 1.2 });
        pL.hline(0.5 * BT * BT, { color: 's3', width: 1.2 });
        pL.line(L, { color: 's1', width: 2.2 }); pL.points(L, { color: 's1', r: 2.5 });
        pL.line(Rr, { color: 's2', width: 2.2 }); pL.points(Rr, { color: 's2', r: 2.5 });
        pL.line(M, { color: 's3', width: 2.2 }); pL.points(M, { color: 's3', r: 2.5 });
        pL.legend([{ label: 'left (Itô)', color: 's1' }, { label: 'right', color: 's2' }, { label: 'trapezoid (Stratonovich)', color: 's3' }]);
        sB.set(fmt(BT)); sI.set(fmt(0.5 * (BT * BT - TT)), `Stratonovich ½B² = ${fmt(0.5 * BT * BT)}`);
      }
      function mc() {
        const TT = T.get(), n = 2 ** k.get(), sd = Math.sqrt(TT / n), r = ctx.rng, M = 3000;
        const L = new Array(M), Rr = new Array(M);
        for (let u = 0; u < M; u++) {
          let x = 0, l = 0, rr = 0;
          for (let i = 0; i < n; i++) { const d = sd * r.normal(); l += x * d; x += d; rr += x * d; }
          L[u] = l; Rr[u] = rr;
        }
        const lo = -TT / 2 - 0.2, hi = TT * 2.5;
        const hL = S.histogram(L, { bins: 50, lo, hi }), hR = S.histogram(Rr, { bins: 50, lo, hi });
        for (const h of [hL, hR]) for (const q of h) q.density = q.count / (M * (q.x1 - q.x0));
        const dens = S.linspace(lo, hi, 400).map(y => { const c = 2 * y / TT + 1; return [y, c > 0 ? Math.min(50, (2 / TT) * S.pdf.chisq(c, 1)) : 0]; });
        const ym = Math.max(...hL.map(q => q.density), ...hR.map(q => q.density)) * 1.2;
        pR.setDomain([lo, hi], [0, ym]).clear();
        pR.bars(hR, { color: 's2', opacity: 0.35 });
        pR.bars(hL, { color: 's1', opacity: 0.45 });
        pR.line(dens, { color: 'ink', width: 1.8 });
        pR.vline(0, { color: 's1', width: 1.2 }); pR.vline(TT, { color: 's2', width: 1.2 });
        pR.legend([{ label: 'left sums', color: 's1', fill: true }, { label: 'right sums', color: 's2', fill: true }, { label: '(T/2)(χ²₁ − 1)', color: 'ink' }]);
        sML.set(`${fmt(S.mean(L))} / ${fmt(S.variance(L))}`, `isometry: 0 / T²/2 = ${fmt(TT * TT / 2)}`);
        sMR.set(fmt(S.mean(Rr)), `≈ T = ${TT}`);
      }
      gen(); mc();
    },
  });

  const SDE = {
    gbm: { label: 'GBM: $dS=\\mu S\\,dt+\\sigma S\\,dB$', x0: 1, mu: (x, a) => a.mu * x, sig: (x, a) => a.sig * x, dsig: (x, a) => a.sig },
    ou: { label: 'OU: $dX=-\\theta X\\,dt+\\sigma\\,dB$', x0: 2, mu: (x, a) => -a.theta * x, sig: (x, a) => a.sig, dsig: () => 0 },
  };

  Atlas.lab('stochastic-calculus-em', {
    title: 'Euler–Maruyama vs Milstein: strong convergence and stability',
    blurb: 'Both schemes use the same Brownian path as the reference solution (exact for GBM, a $2^{12}$-step integral for OU). Right: strong error $\\E|X_1-\\hat X_1|$ over 200 paths against step size, log–log; slopes $1/2$ and $1$ are the theoretical orders. For OU (additive noise) the schemes coincide. Push $\\theta$ up with coarse steps to watch EM explode once $\\Delta>2/\\theta$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const model = UI.segmented(controls, { label: 'SDE', options: Object.keys(SDE).map(k => ({ value: k, label: SDE[k].label })), value: 'gbm', onChange: full });
      const mu = UI.slider(controls, { label: 'GBM drift $\\mu$', min: -1, max: 2, step: 0.05, value: 0.5, onChange: full });
      const sig = UI.slider(controls, { label: 'Volatility $\\sigma$', min: 0.1, max: 2, step: 0.05, value: 1, onChange: full });
      const theta = UI.slider(controls, { label: 'OU mean reversion $\\theta$', min: 0.5, max: 60, step: 0.5, value: 2, onChange: full });
      const k = UI.slider(controls, { label: 'Steps $2^k$ on $[0,1]$: $k$', min: 1, max: 10, step: 1, value: 4, onInput: path });
      UI.button(controls, 'New path', () => { seedPath(); path(); }, { primary: true });
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 3], xlabel: 't', ylabel: 'X(t)', title: 'One path: exact vs schemes' });
      const pE = new Plot(stage, { width: 420, height: 270, x: [0.0005, 1], y: [1e-4, 10], logx: true, logy: true, xlabel: 'step size Δ', ylabel: 'E|X₁ − X̂₁|', title: 'Strong error (200 paths)' });
      const sD = UI.stat(stats, 'Step $\\Delta$');
      const sErr = UI.stat(stats, 'Endpoint error, this path: EM / Milstein', { accent: 'accent' });
      const sSl = UI.stat(stats, 'Fitted slopes: EM / Milstein');
      const sStab = UI.stat(stats, 'EM stability factor $|1-\\theta\\Delta|$');
      const KF = 12, NF = 2 ** KF;
      const par = () => ({ mu: mu.get(), sig: sig.get(), theta: theta.get() });
      function exactPath(dB, M, a) {
        const x = new Float64Array(NF + 1); x[0] = M.x0; const dt = 1 / NF;
        if (model.get() === 'gbm') { let B = 0; for (let i = 1; i <= NF; i++) { B += dB[i - 1]; x[i] = M.x0 * Math.exp((a.mu - a.sig * a.sig / 2) * i * dt + a.sig * B); } }
        else { const e = Math.exp(-a.theta * dt), e2 = Math.exp(-a.theta * dt / 2); for (let i = 1; i <= NF; i++) x[i] = e * x[i - 1] + a.sig * e2 * dB[i - 1]; }
        return x;
      }
      function scheme(dB, M, a, kk, mil) {
        const n = 2 ** kk, agg = NF / n, h = 1 / n, out = [[0, M.x0]];
        let x = M.x0;
        for (let j = 0; j < n; j++) {
          let d = 0; for (let i = j * agg; i < (j + 1) * agg; i++) d += dB[i];
          const s = M.sig(x, a);
          x = x + M.mu(x, a) * h + s * d + (mil ? 0.5 * s * M.dsig(x, a) * (d * d - h) : 0);
          if (!Number.isFinite(x) || Math.abs(x) > 1e8) x = Math.sign(x || 1) * 1e8;
          out.push([(j + 1) * h, x]);
        }
        return out;
      }
      let dB0;
      function seedPath() { const r = ctx.rng, sd = Math.sqrt(1 / NF); dB0 = new Float64Array(NF); for (let i = 0; i < NF; i++) dB0[i] = sd * r.normal(); }
      function path() {
        const M = SDE[model.get()], a = par(), kk = k.get();
        const ex = exactPath(dB0, M, a), em = scheme(dB0, M, a, kk, false), mi = scheme(dB0, M, a, kk, true);
        const ePts = []; for (let i = 0; i <= NF; i += 8) ePts.push([i / NF, ex[i]]);
        const vals = ePts.map(q => q[1]);
        const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.25 + 0.1;
        pP.setDomain([0, 1], [lo - pad, hi + pad]).clear();
        pP.line(ePts, { color: 'ink', width: 1.6 });
        pP.line(em, { color: 's2', width: 2 }); pP.points(em, { color: 's2', r: 2.5 });
        if (model.get() === 'gbm') { pP.line(mi, { color: 's1', width: 2, dash: true }); pP.points(mi, { color: 's1', r: 2.5 }); }
        pP.legend([{ label: 'reference solution', color: 'ink' }, { label: 'Euler–Maruyama', color: 's2' }].concat(model.get() === 'gbm' ? [{ label: 'Milstein', color: 's1', dash: true }] : []));
        const h = 1 / 2 ** kk;
        sD.set(`1/${2 ** kk}`);
        sErr.set(`${fmt(Math.abs(em[em.length - 1][1] - ex[NF]), 2)} / ${fmt(Math.abs(mi[mi.length - 1][1] - ex[NF]), 2)}`);
        const f = Math.abs(1 - a.theta * h);
        sStab.set(model.get() === 'ou' ? fmt(f) : '—', model.get() === 'ou' ? (f >= 1 ? '<span style="color:var(--bad)">≥ 1: EM unstable</span>' : 'stable (< 1)') : 'applies to OU');
      }
      function errors() {
        const M = SDE[model.get()], a = par(), r = S.rng(7), P = 200, sd = Math.sqrt(1 / NF);
        const ks = S.range(9).map(i => i + 2), eE = ks.map(() => 0), eM = ks.map(() => 0);
        const dB = new Float64Array(NF);
        for (let pth = 0; pth < P; pth++) {
          for (let i = 0; i < NF; i++) dB[i] = sd * r.normal();
          const ex = exactPath(dB, M, a)[NF];
          ks.forEach((kk, j) => {
            eE[j] += Math.min(1e6, Math.abs(scheme(dB, M, a, kk, false)[2 ** kk][1] - ex)) / P;
            eM[j] += Math.min(1e6, Math.abs(scheme(dB, M, a, kk, true)[2 ** kk][1] - ex)) / P;
          });
        }
        const hs = ks.map(kk => 2 ** -kk);
        const ptsE = hs.map((h, j) => [h, eE[j]]), ptsM = hs.map((h, j) => [h, eM[j]]);
        const all = [...eE, ...eM].filter(v => v > 0);
        const lo = Math.max(1e-7, Math.min(...all) / 3), hi = Math.max(...all) * 3;
        pE.setDomain([hs[hs.length - 1] / 1.5, 0.5], [lo, hi]).clear();
        const anchorE = eE[eE.length - 1], anchorM = eM[eM.length - 1], hmin = hs[hs.length - 1];
        pE.line(hs.map(h => [h, anchorE * Math.sqrt(h / hmin)]), { color: 's2', width: 1, dash: true, opacity: 0.6 });
        pE.line(hs.map(h => [h, anchorM * (h / hmin)]), { color: 's1', width: 1, dash: true, opacity: 0.6 });
        pE.line(ptsE, { color: 's2', width: 2.2 }); pE.points(ptsE, { color: 's2', r: 3 });
        pE.line(ptsM, { color: 's1', width: 2.2 }); pE.points(ptsM, { color: 's1', r: 3 });
        pE.legend([{ label: 'EM', color: 's2' }, { label: 'Milstein', color: 's1' }, { label: 'slope ½ / slope 1 guides', color: 'muted', dash: true }], { left: true });
        const slope = e => { const xs = hs.slice(-5).map(Math.log), ys = e.slice(-5).map(v => Math.log(Math.max(v, 1e-12))); const mx = S.mean(xs), my = S.mean(ys); let sxy = 0, sxx = 0; xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; }); return sxy / sxx; };
        sSl.set(`${fmt(slope(eE), 2)} / ${fmt(slope(eM), 2)}`, 'over the five finest steps');
      }
      function full() { path(); errors(); }
      seedPath(); full();
    },
  });

  /* ================================================================== */
  /* Time series                                                         */
  /* ================================================================== */

  function acfOf(x, H) {
    const n = x.length, m = S.mean(x); let g0 = 0;
    for (let t = 0; t < n; t++) g0 += (x[t] - m) ** 2;
    const out = [];
    for (let h = 1; h <= H; h++) { let s = 0; for (let t = 0; t + h < n; t++) s += (x[t + h] - m) * (x[t] - m); out.push(g0 > 0 ? s / g0 : 0); }
    return out;
  }
  function pacfFromAcf(rho) { // Durbin–Levinson; rho[h-1] = rho(h)
    const H = rho.length, out = []; let prev = [];
    for (let h = 1; h <= H; h++) {
      let num = rho[h - 1], den = 1;
      for (let k = 1; k < h; k++) { num -= prev[k - 1] * rho[h - k - 1]; den -= prev[k - 1] * rho[k - 1]; }
      const phh = Math.abs(den) < 1e-12 ? 0 : num / den;
      const cur = []; for (let k = 1; k < h; k++) cur.push(prev[k - 1] - phh * prev[h - k - 1]);
      cur.push(phh); prev = cur; out.push(phh);
    }
    return out;
  }
  function armaAcf(p1, p2, th, H) {
    const J = 600, psi = [1];
    for (let j = 1; j < J; j++) psi.push((j === 1 ? th : 0) + p1 * psi[j - 1] + (j >= 2 ? p2 * psi[j - 2] : 0));
    const g = h => { let s = 0; for (let j = 0; j + h < J; j++) s += psi[j] * psi[j + h]; return s; };
    const g0 = g(0); return S.range(H).map(h => g(h + 1) / g0);
  }
  const causal2 = (p1, p2) => p1 + p2 < 1 && p2 - p1 < 1 && Math.abs(p2) < 1;

  Atlas.lab('time-series-arma', {
    title: 'ARMA simulator: ACF, PACF and the causality triangle',
    blurb: 'Simulate $X_t=\\phi_1X_{t-1}+\\phi_2X_{t-2}+Z_t+\\theta Z_{t-1}$. Bars: sample ACF and PACF with $\\pm1.96/\\sqrt n$ bands; dots: theoretical values. Click inside the triangle to choose $(\\phi_1,\\phi_2)$; the shaded region below the parabola gives complex roots and pseudo-cycles. Leave the triangle and the recursion explodes.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const preset = UI.select(controls, { label: 'Preset', value: 'ar2c', onChange: v => { const P = PRE[v]; p1.set(P[0]); p2.set(P[1]); th.set(P[2]); run(); }, options: [
        { value: 'ar1', label: 'AR(1), φ = 0.8' }, { value: 'ar1n', label: 'AR(1), φ = −0.7' }, { value: 'ar2c', label: 'AR(2), complex roots' },
        { value: 'ma1', label: 'MA(1), θ = 0.8' }, { value: 'arma', label: 'ARMA(1,1)' }, { value: 'wn', label: 'White noise' }, { value: 'ncaus', label: 'Non-causal AR(1), φ = 1.02' },
      ] });
      const PRE = { ar1: [0.8, 0, 0], ar1n: [-0.7, 0, 0], ar2c: [1.3, -0.75, 0], ma1: [0, 0, 0.8], arma: [0.7, 0, 0.5], wn: [0, 0, 0], ncaus: [1.02, 0, 0] };
      const p1 = UI.slider(controls, { label: '$\\phi_1$', min: -2, max: 2, step: 0.01, value: 1.3, onChange: run });
      const p2 = UI.slider(controls, { label: '$\\phi_2$', min: -1, max: 1, step: 0.01, value: -0.75, onChange: run });
      const th = UI.slider(controls, { label: '$\\theta$ (MA)', min: -2, max: 2, step: 0.01, value: 0, onChange: run });
      const n = UI.slider(controls, { label: 'Length $n$', min: 50, max: 2000, step: 50, value: 300, onChange: run });
      UI.button(controls, 'New series', run, { primary: true });
      const top = UI.h('div', 'lab-stage two', stage), bot = UI.h('div', 'lab-stage two', stage);
      const pS = new Plot(top, { width: 420, height: 230, xlabel: 't', ylabel: 'X_t', title: 'Simulated series' });
      const pT = new Plot(top, { width: 420, height: 230, x: [-2.2, 2.2], y: [-1.2, 1.2], xlabel: 'φ₁', ylabel: 'φ₂', title: 'Causality region (click to set φ₁, φ₂)' });
      const pA = new Plot(bot, { width: 420, height: 230, x: [0.3, 25.7], y: [-1, 1], xlabel: 'lag h', ylabel: 'ρ(h)', title: 'ACF' });
      const pQ = new Plot(bot, { width: 420, height: 230, x: [0.3, 25.7], y: [-1, 1], xlabel: 'lag h', ylabel: 'α(h)', title: 'PACF' });
      const sC = UI.stat(stats, 'Causal?', { accent: 'accent' });
      const sI = UI.stat(stats, 'Invertible?');
      const sR = UI.stat(stats, 'AR roots of $\\phi(z)$ (moduli)');
      const sY = UI.stat(stats, 'Yule–Walker AR(2) fit');
      pT.onPointer((x, y, e, type) => { if (type === 'down' || type === 'drag') { p1.set(S.clamp(Math.round(x * 100) / 100, -2, 2)); p2.set(S.clamp(Math.round(y * 100) / 100, -1, 1)); run(); } });
      const H = 25;
      function run() {
        const a1 = p1.get(), a2 = p2.get(), t = th.get(), N = Math.round(n.get()), r = ctx.rng, ok = causal2(a1, a2);
        const burn = ok ? 300 : 0, x = [];
        let x1 = 0, x2 = 0, zPrev = 0;
        for (let i = 0; i < N + burn; i++) {
          const z = r.normal(); let v = a1 * x1 + a2 * x2 + z + t * zPrev;
          if (!Number.isFinite(v) || Math.abs(v) > 1e12) v = Math.sign(v || 1) * 1e12;
          x2 = x1; x1 = v; zPrev = z; if (i >= burn) x.push(v);
        }
        const ex = S.extent(x), pad = (ex[1] - ex[0]) * 0.05 + 1e-9;
        pS.setDomain([0, N], [ex[0] - pad, ex[1] + pad]).clear();
        pS.line(x.map((v, i) => [i, v]), { color: ok ? 's1' : 'bad', width: 1.1 });
        if (!ok) pS.text(N / 2, ex[1], 'outside the triangle: forward recursion explodes', { anchor: 'middle', dy: 14, color: 'bad', size: 12, weight: 600 });
        // triangle
        pT.clear();
        pT.polygon([[-2, -1], [2, -1], [0, 1]], { color: 's3', opacity: 0.12 });
        pT.polygon(S.linspace(-2, 2, 80).map(u => [u, -u * u / 4]).concat([[2, -1], [-2, -1]]), { color: 's5', opacity: 0.15 });
        pT.line([[-2, -1], [2, -1], [0, 1], [-2, -1]], { color: 's3', width: 1.5 });
        pT.line(S.linspace(-2, 2, 80).map(u => [u, -u * u / 4]), { color: 's5', width: 1, dash: true });
        pT.dot(a1, a2, { color: ok ? 's1' : 'bad', r: 6 });
        pT.text(0, 0.35, 'real roots', { anchor: 'middle', size: 11, color: 'muted' });
        pT.text(0, -0.7, 'complex roots', { anchor: 'middle', size: 11, color: 'muted' });
        // ACF / PACF
        const acf = acfOf(x, H), pacf = pacfFromAcf(acf), band = 1.96 / Math.sqrt(N);
        const thA = ok ? armaAcf(a1, a2, t, H) : null, thP = thA ? pacfFromAcf(thA) : null;
        for (const [P, v, tv] of [[pA, acf, thA], [pQ, pacf, thP]]) {
          P.clear();
          P.rect(0.3, -band, 25.7, band, { color: 'accent', opacity: 0.12 });
          P.hline(0, { color: 'muted', width: 1, dash: false });
          P.bars(v.map((y, i) => ({ x0: i + 1 - 0.3, x1: i + 1 + 0.3, y })), { color: 's1', opacity: 0.6 });
          if (tv) P.points(tv.map((y, i) => [i + 1, y]), { color: 'ink', r: 3 });
        }
        pA.legend([{ label: 'sample', color: 's1', fill: true }, { label: 'theory', color: 'ink' }, { label: '±1.96/√n', color: 'accent', fill: true, opacity: 0.2 }]);
        // stats
        sC.set(ok ? 'yes' : '<span style="color:var(--bad)">no</span>', 'triangle: φ₁+φ₂<1, φ₂−φ₁<1, |φ₂|<1');
        sI.set(Math.abs(t) < 1 ? 'yes' : '<span style="color:var(--bad)">no</span>', '|θ| < 1');
        let roots;
        if (Math.abs(a2) < 1e-9) roots = Math.abs(a1) < 1e-9 ? 'none (no AR part)' : fmt(Math.abs(1 / a1));
        else { const d = a1 * a1 + 4 * a2; roots = d >= 0 ? [(-a1 + Math.sqrt(d)) / (2 * a2), (-a1 - Math.sqrt(d)) / (2 * a2)].map(z => fmt(Math.abs(z))).join(', ') : `${fmt(Math.sqrt(-1 / a2))} (complex pair)`; }
        sR.set(roots, 'causal ⇔ all > 1');
        const r1 = acf[0], r2 = acf[1], den = 1 - r1 * r1;
        sY.set(`φ̂₁ = ${fmt(r1 * (1 - r2) / den, 2)}, φ̂₂ = ${fmt((r2 - r1 * r1) / den, 2)}`);
      }
      run();
    },
  });

  Atlas.lab('time-series-unit-root', {
    title: 'Estimating $\\phi$ near a unit root',
    blurb: 'Fit $\\hat\\phi$ by least squares to 2000 simulated AR(1) series of length $n$. Left: $\\hat\\phi$ against the textbook approximation $\\N(\\phi,(1-\\phi^2)/n)$. Right: the usual $t$-statistic against $\\N(0,1)$. As $\\phi\\to1$ the estimator is biased down, the normal approximation collapses, and at $\\phi=1$ the $t$-statistic follows the left-shifted Dickey–Fuller law: nominal 5% tests reject far too often.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const phi = UI.slider(controls, { label: 'True $\\phi$', min: 0, max: 1, step: 0.01, value: 0.9, onChange: run });
      const n = UI.slider(controls, { label: 'Length $n$', min: 25, max: 500, step: 5, value: 100, onChange: run });
      UI.button(controls, 'Resimulate', run, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'φ̂', ylabel: 'density', title: 'Sampling distribution of φ̂' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-5, 4], y: [0, 0.5], xlabel: 't = (φ̂ − φ)/se', ylabel: 'density', title: 't-statistic vs N(0,1)' });
      const sB = UI.stat(stats, 'Mean $\\hat\\phi$ (bias)', { accent: 'accent' });
      const sS = UI.stat(stats, 'SD of $\\hat\\phi$ vs $\\sqrt{(1-\\phi^2)/n}$');
      const sT = UI.stat(stats, 'Size of nominal 5% two-sided $t$-test');
      const sQ = UI.stat(stats, '5% quantile of $t$');
      function run() {
        const ph = phi.get(), N = Math.round(n.get()), r = ctx.rng, M = 2000;
        const est = new Array(M), ts = new Array(M);
        for (let u = 0; u < M; u++) {
          let x = ph < 1 ? r.normal() / Math.sqrt(1 - ph * ph) : 0, sxy = 0, sxx = 0;
          const xs = [x];
          for (let i = 1; i <= N; i++) { const y = ph * x + r.normal(); sxy += x * y; sxx += x * x; x = y; xs.push(x); }
          const f = sxy / sxx;
          let rss = 0; for (let i = 1; i <= N; i++) rss += (xs[i] - f * xs[i - 1]) ** 2;
          est[u] = f; ts[u] = (f - ph) / Math.sqrt(rss / (N - 1) / sxx);
        }
        const sd0 = Math.sqrt(Math.max(1 - ph * ph, 0) / N);
        const lo = Math.min(...est), hi = Math.max(...est, ph + 3 * sd0), pad = (hi - lo) * 0.05;
        const h = S.histogram(est, { bins: 50, lo: lo - pad, hi: hi + pad });
        const nd = sd0 > 1e-6 ? S.linspace(lo - pad, hi + pad, 300).map(v => [v, S.pdf.normal(v, ph, sd0)]) : [];
        const ym = Math.max(...h.map(b => b.density), ...(nd.length ? nd.map(q => q[1]) : [0])) * 1.12;
        pL.setDomain([lo - pad, hi + pad], [0, ym]).clear();
        pL.bars(h, { color: 's1', opacity: 0.5 });
        if (nd.length) pL.line(nd, { color: 'ink', width: 2 });
        pL.vline(ph, { color: 'bad', width: 1.5 });
        pL.legend([{ label: 'simulated φ̂', color: 's1', fill: true }, { label: 'N(φ, (1−φ²)/n)', color: 'ink' }, { label: 'true φ', color: 'bad', dash: true }]);
        const ht = S.histogram(ts.filter(v => v > -5 && v < 4), { bins: 45, lo: -5, hi: 4 });
        for (const b of ht) b.density = b.count / (M * (b.x1 - b.x0));
        pR.setDomain([-5, 4], [0, Math.max(0.45, ...ht.map(b => b.density)) * 1.1]).clear();
        pR.bars(ht, { color: 's2', opacity: 0.5 });
        pR.line(S.linspace(-5, 4, 200).map(v => [v, S.pdf.normal(v)]), { color: 'ink', width: 2 });
        pR.vline(-1.96, { color: 'muted', width: 1 }); pR.vline(1.96, { color: 'muted', width: 1 });
        pR.legend([{ label: 'simulated t', color: 's2', fill: true }, { label: 'N(0,1)', color: 'ink' }], { left: true });
        const st = ts.slice().sort((a, b) => a - b);
        sB.set(fmt(S.mean(est)), `bias ${fmt(S.mean(est) - ph, 2)}`);
        sS.set(fmt(S.sd(est)), `asymptotic ${fmt(sd0)}`);
        sT.set(pct(ts.filter(v => Math.abs(v) > 1.96).length / M), 'should be 5.0%');
        sQ.set(fmt(S.quantile(st, 0.05, true), 2), 'N(0,1): −1.64; Dickey–Fuller (no intercept): ≈ −1.95');
      }
      run();
    },
  });
})();
