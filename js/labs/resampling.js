/* Labs: Bootstrap & Resampling. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;

  // Shared populations. `sample(r)` draws one value; `dens` is the pdf for drawing.
  const POPS = {
    normal: { label: 'Normal(0,1)', sample: r => r.normal(), dens: x => S.pdf.normal(x), range: [-3.5, 3.5], mean: 0, median: 0, sd: 1, max: Infinity },
    exp: { label: 'Exponential(1)', sample: r => r.exp(1), dens: x => S.pdf.exp(x), range: [0, 6], mean: 1, median: Math.LN2, sd: 1, max: Infinity },
    lognormal: { label: 'Lognormal(0,1)', sample: r => r.lognormal(0, 1), dens: x => S.pdf.lognormal(x), range: [0, 8], mean: Math.exp(0.5), median: 1, sd: Math.sqrt((Math.E - 1) * Math.E), max: Infinity },
    uniform: { label: 'Uniform(0,1)', sample: r => r.uniform(), dens: x => S.pdf.uniform(x), range: [-0.1, 1.1], mean: 0.5, median: 0.5, sd: 1 / Math.sqrt(12), max: 1 },
    cauchy: { label: 'Cauchy(0,1)', sample: r => r.cauchy(), dens: x => S.pdf.cauchy(x), range: [-8, 8], mean: NaN, median: 0, sd: NaN, max: Infinity },
  };
  const trimmed = (a, p = 0.1) => { const s = a.slice().sort((x, y) => x - y), k = Math.floor(p * s.length); return S.mean(s.slice(k, s.length - k)); };
  const STATS = {
    mean: { label: 'Mean', f: S.mean },
    median: { label: 'Median', f: S.median },
    sd: { label: 'SD', f: a => S.sd(a) },
    trim: { label: '10% trimmed', f: a => trimmed(a) },
    max: { label: 'Maximum', f: a => Math.max(...a) },
  };
  // True functional T(F); trimmed mean computed once by brute force.
  const trueCache = {};
  function trueTheta(pop, stat) {
    const P = POPS[pop];
    if (stat === 'trim') {
      const k = pop + ':trim';
      if (trueCache[k] == null) { const r = S.rng(1); trueCache[k] = trimmed(r.array(200000, () => P.sample(r))); }
      return trueCache[k];
    }
    return P[stat];
  }
  const resample = (r, x) => { const n = x.length, out = new Array(n); for (let i = 0; i < n; i++) out[i] = x[(r.u() * n) | 0]; return out; };

  Atlas.lab('bootstrap-world', {
    title: 'The real world and the bootstrap world',
    blurb: 'Left: one sample from $F$. Right: the bootstrap distribution of $T_n^*$ (bars) against the true sampling distribution of $T_n$ (curve, by brute-force simulation from $F$). Try the maximum of a Uniform, or the mean of a Cauchy, to watch the bootstrap fail.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population $F$', options: Object.keys(POPS).map(k => ({ value: k, label: POPS[k].label })), value: 'exp', onChange: () => { truth = null; newSample(); } });
      const stat = UI.segmented(controls, { label: 'Statistic $T_n$', options: Object.keys(STATS).map(k => ({ value: k, label: STATS[k].label })), value: 'median', onChange: () => { truth = null; boot(); } });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 200, step: 1, value: 30, onChange: () => { truth = null; newSample(); } });
      const B = UI.slider(controls, { label: 'Resamples $B$', min: 100, max: 5000, step: 100, value: 2000, onChange: () => boot() });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newSample, { primary: true });
      UI.button(row, 'Re-bootstrap', boot);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'density', title: 'Real world: one sample from F' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'T', ylabel: 'density', title: 'Bootstrap world vs. truth' });
      const sT = UI.stat(stats, 'Observed $T_n$', { accent: 'accent' });
      const sTh = UI.stat(stats, 'True $\\theta = T(F)$');
      const sSe = UI.stat(stats, '$\\widehat{\\se}_{\\text{boot}}$');
      const sSeT = UI.stat(stats, 'True $\\se_F(T_n)$');
      const sPct = UI.stat(stats, '95% percentile CI');
      const sPiv = UI.stat(stats, '95% pivotal CI');

      let x = [], truth = null, reps = [];
      function newSample() {
        const P = POPS[pop.get()];
        x = ctx.rng.array(Math.round(n.get()), () => P.sample(ctx.rng));
        boot();
      }
      function computeTruth() {
        const P = POPS[pop.get()], f = STATS[stat.get()].f, nn = Math.round(n.get());
        const r = S.rng(99);
        const vals = r.array(3000, () => f(r.array(nn, () => P.sample(r))));
        truth = { vals, sd: S.sd(vals) };
      }
      function boot() {
        if (!truth) computeTruth();
        const f = STATS[stat.get()].f, BB = Math.round(B.get());
        const T = f(x);
        reps = new Array(BB);
        for (let b = 0; b < BB; b++) reps[b] = f(resample(ctx.rng, x));
        reps.sort((a, b) => a - b);
        draw(T);
      }
      function draw(T) {
        const P = POPS[pop.get()], th = trueTheta(pop.get(), stat.get());
        // Left: sample histogram + population density
        const [lo, hi] = P.range;
        const xs = x.filter(v => v >= lo && v <= hi);
        const hist = S.histogram(xs, { bins: Math.min(30, Math.max(8, Math.round(Math.sqrt(x.length) * 1.5))), lo, hi });
        for (const b of hist) b.density = b.count / (x.length * (b.x1 - b.x0));
        const ymaxL = Math.max(...hist.map(b => b.density), ...S.linspace(lo, hi, 200).map(P.dens).filter(Number.isFinite)) * 1.12;
        pL.setDomain([lo, hi], [0, Math.min(ymaxL, 3)]).clear();
        pL.bars(hist, { color: 's1', opacity: 0.35 });
        pL.line(S.linspace(lo, hi, 300).map(v => [v, P.dens(v)]), { color: 'ink', width: 1.5, opacity: 0.7 });
        for (const v of x) if (v >= lo && v <= hi) pL.segment(v, 0, v, ymaxL * 0.05, { color: 's1', width: 1.2, opacity: 0.8 });
        if (x.length - xs.length) pL.text(hi, ymaxL * 0.92, `${x.length - xs.length} point(s) off-axis`, { anchor: 'end', color: 'muted', size: 11, dx: -6 });
        pL.legend([{ label: 'sample', color: 's1', fill: true, opacity: 0.35 }, { label: 'true F', color: 'ink' }]);

        // Right: bootstrap distribution vs truth
        const tv = truth.vals.slice().sort((a, b) => a - b);
        let a0 = Math.min(S.quantile(tv, 0.005, true), reps[Math.floor(reps.length * 0.005)]);
        let a1 = Math.max(S.quantile(tv, 0.995, true), reps[Math.ceil(reps.length * 0.995) - 1]);
        const pad = (a1 - a0) * 0.08 || 0.1; a0 -= pad; a1 += pad;
        const hb = S.histogram(reps, { bins: 45, lo: a0, hi: a1 });
        const kd = S.kde(tv.filter(v => v >= a0 && v <= a1), S.silverman(tv) * 0.9);
        const grid = S.linspace(a0, a1, 220), kv = grid.map(kd);
        const ymaxR = Math.max(...hb.map(b => b.density), ...kv) * 1.12;
        pR.setDomain([a0, a1], [0, ymaxR]).clear();
        const qlo = S.quantile(reps, 0.025, true), qhi = S.quantile(reps, 0.975, true);
        pR.band(qlo, qhi, { color: 's2', opacity: 0.1 });
        pR.bars(hb, { color: 's2', opacity: 0.5 });
        pR.line(grid.map((g, i) => [g, kv[i]]), { color: 'ink', width: 2 });
        pR.vline(T, { color: 's2', dash: false, width: 2 });
        if (Number.isFinite(th)) pR.vline(th, { color: 'ink', width: 1.5 });
        pR.legend([{ label: 'bootstrap T*', color: 's2', fill: true }, { label: 'true law of T', color: 'ink' }, { label: 'observed T', color: 's2' }]);

        const seB = S.sd(reps, 0);
        sT.set(S.fmt(T));
        sTh.set(Number.isFinite(th) ? S.fmt(th) : (Number.isNaN(th) ? 'undefined' : '∞'));
        sSe.set(S.fmt(seB));
        sSeT.set(S.fmt(truth.sd), 'from 3000 simulated samples');
        const cover = (l, u) => (Number.isFinite(th) ? (l <= th && th <= u ? '<span style="color:var(--good)">covers θ</span>' : '<span style="color:var(--bad)">misses θ</span>') : '');
        sPct.set(`[${S.fmt(qlo, 2)}, ${S.fmt(qhi, 2)}]`, cover(qlo, qhi));
        sPiv.set(`[${S.fmt(2 * T - qhi, 2)}, ${S.fmt(2 * T - qlo, 2)}]`, cover(2 * T - qhi, 2 * T - qlo));
      }
      newSample();
    },
  });

  Atlas.lab('bootstrap-coverage', {
    title: 'Do bootstrap intervals deliver 95%?',
    blurb: 'Repeat the whole experiment many times: draw a fresh sample, bootstrap it, build four nominal-95% intervals, and record whether each covers the true $\\theta$. Small $n$ and skewed $F$ separate the methods; the studentized interval (mean only) stays closest to 95%.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population', options: ['exp', 'lognormal', 'normal', 'uniform'].map(k => ({ value: k, label: POPS[k].label })), value: 'lognormal', onChange: reset });
      const stat = UI.segmented(controls, { label: 'Statistic', options: [{ value: 'mean', label: 'Mean' }, { value: 'median', label: 'Median' }], value: 'mean', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 100, step: 1, value: 15, onChange: reset });
      const B = UI.slider(controls, { label: 'Resamples $B$ per experiment', min: 200, max: 2000, step: 100, value: 400, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pBar = new Plot(stage, { width: 420, height: 270, x: [0, 4], y: [0.7, 1], ylabel: 'empirical coverage', title: 'Coverage by method', xticks: 0 });
      const pInt = new Plot(stage, { width: 420, height: 270, y: [0, 40], yticks: 0, xlabel: 'θ', title: 'Last 40 intervals (pivotal)' });
      const sN = UI.stat(stats, 'Experiments');
      const sW = UI.stat(stats, 'Mean width (percentile)');
      const METHODS = [['Normal', 's1'], ['Percentile', 's2'], ['Pivotal', 's3'], ['Studentized', 's5']];
      let hits, count, widths, recent;
      function reset() { hits = [0, 0, 0, 0]; count = 0; widths = 0; recent = []; loop.stop(); draw(); }
      function experiment() {
        const P = POPS[pop.get()], isMean = stat.get() === 'mean', f = isMean ? S.mean : S.median;
        const th = trueTheta(pop.get(), stat.get()), nn = Math.round(n.get()), BB = Math.round(B.get());
        const x = ctx.rng.array(nn, () => P.sample(ctx.rng));
        const T = f(x), seHat = S.sd(x) / Math.sqrt(nn);
        const reps = new Array(BB), zs = [];
        for (let b = 0; b < BB; b++) {
          const xs = resample(ctx.rng, x);
          reps[b] = f(xs);
          if (isMean) { const se = S.sd(xs) / Math.sqrt(nn); if (se > 0) zs.push((reps[b] - T) / se); }
        }
        reps.sort((a, b) => a - b);
        const seB = S.sd(reps), q = p => S.quantile(reps, p, true);
        const ints = [[T - 1.96 * seB, T + 1.96 * seB], [q(0.025), q(0.975)], [2 * T - q(0.975), 2 * T - q(0.025)]];
        if (isMean && zs.length > 10) { zs.sort((a, b) => a - b); ints.push([T - S.quantile(zs, 0.975, true) * seHat, T - S.quantile(zs, 0.025, true) * seHat]); }
        else ints.push(null);
        ints.forEach((I, k) => { if (I && I[0] <= th && th <= I[1]) hits[k]++; });
        widths += ints[1][1] - ints[1][0];
        count++;
        recent.push({ I: ints[2], ok: ints[2][0] <= th && th <= ints[2][1] });
        if (recent.length > 40) recent.shift();
      }
      let acc = 0;
      function step(dt) {
        acc += dt;
        const per = Math.max(1, Math.round(60000 / (B.get() * n.get())));
        for (let i = 0; i < per; i++) experiment();
        if (count >= 2000) { draw(); return false; }
        if (acc > 0.05) { acc = 0; draw(); }
      }
      function draw() {
        const isMean = stat.get() === 'mean';
        pBar.clear();
        pBar.hline(0.95, { color: 'ink', dash: '4 4', width: 1.2 });
        pBar.text(3.98, 0.952, 'nominal 95%', { anchor: 'end', dy: -4, size: 11, color: 'muted' });
        METHODS.forEach(([name, c], k) => {
          const cov = count ? hits[k] / count : NaN;
          const avail = k < 3 || isMean;
          if (avail && count) {
            pBar.bars([{ x0: k + 0.15, x1: k + 0.85, y: Math.max(0.7, cov) }], { color: c, opacity: 0.75, base: 0.7 });
            pBar.text(k + 0.5, Math.max(0.7, cov), (100 * cov).toFixed(1) + '%', { anchor: 'middle', dy: -6, size: 12, weight: 600 });
          }
          pBar.text(k + 0.5, 0.7, avail ? name : name + ' (mean only)', { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        });
        const th = trueTheta(pop.get(), stat.get());
        if (recent.length) {
          const lo = Math.min(...recent.map(r => r.I[0]), th), hi = Math.max(...recent.map(r => r.I[1]), th), pad = (hi - lo) * 0.05;
          pInt.setDomain([lo - pad, hi + pad], [0, 41]).clear();
          recent.forEach((r, i) => pInt.segment(r.I[0], i + 1, r.I[1], i + 1, { color: r.ok ? 's3' : 'bad', width: 2.5 }));
          pInt.vline(th, { color: 'ink', dash: false, width: 1.5 });
        } else pInt.clear();
        sN.set(String(count));
        sW.set(count ? S.fmt(widths / count) : '—');
      }
      reset();
    },
  });
})();
