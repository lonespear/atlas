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

  /* ===================================================================== jackknife */
  const sortNum = a => Float64Array.from(a).sort();
  // Leave-one-out medians from sorted data s: deleting rank k (0-based).
  function looMedianSorted(s, k) {
    const m = s.length - 1, get = j => (j < k ? s[j] : s[j + 1]);
    return m % 2 ? get((m - 1) / 2) : 0.5 * (get(m / 2 - 1) + get(m / 2));
  }
  // Generic leave-one-out values (O(n^2); fine for n <= ~100).
  function looValues(x, f) {
    const n = x.length, out = new Array(n), y = new Array(n - 1);
    for (let i = 0; i < n; i++) {
      for (let j = 0, k = 0; j < n; j++) if (j !== i) y[k++] = x[j];
      out[i] = f(y);
    }
    return out;
  }
  function jackStats(theta, loo) {
    const n = loo.length, mdot = S.mean(loo);
    let ss = 0; for (const v of loo) ss += (v - mdot) ** 2;
    const bias = (n - 1) * (mdot - theta);
    return { mdot, bias, jack: theta - bias, v: (n - 1) / n * ss, pseudo: loo.map(v => n * theta - (n - 1) * v) };
  }

  Atlas.lab('jackknife-pseudovalues', {
    title: 'Leave-one-out values and pseudo-values',
    blurb: 'Left: each observation $X_i$ against the statistic recomputed without it, $\\hat\\theta_{(i)}$. Right: the pseudo-values $\\tilde\\theta_i = n\\hat\\theta-(n-1)\\hat\\theta_{(i)}$ against $X_i$. For the mean the pseudo-values sit on the diagonal (they are the data); for the plug-in variance the jackknife returns $s^2$ exactly; for the median the leave-one-out values collapse to two levels.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population $F$', options: ['normal', 'exp', 'lognormal', 'uniform'].map(k => ({ value: k, label: POPS[k].label })), value: 'exp', onChange: newSample });
      const JST = {
        mean: { label: 'Mean', f: S.mean },
        pvar: { label: 'Plug-in var', f: a => S.variance(a, 0) },
        median: { label: 'Median', f: S.median },
        trim: { label: '10% trimmed', f: a => trimmed(a) },
        logmean: { label: 'log(mean)', f: a => Math.log(S.mean(a)) },
      };
      const stat = UI.segmented(controls, { label: 'Statistic', options: Object.keys(JST).map(k => ({ value: k, label: JST[k].label })), value: 'median', onChange: draw });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 6, max: 60, step: 1, value: 20, onChange: newSample });
      UI.button(controls, 'New sample', newSample, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'observation X_i', ylabel: 'θ̂(i)', title: 'Leave-one-out values' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'observation X_i', ylabel: 'pseudo-value', title: 'Pseudo-values' });
      const sT = UI.stat(stats, '$\\hat\\theta$', { accent: 'accent' });
      const sB = UI.stat(stats, '$\\widehat{\\Bias}_{\\text{jack}}$');
      const sJ = UI.stat(stats, '$\\hat\\theta_{\\text{jack}}$');
      const sSe = UI.stat(stats, '$\\widehat{\\se}_{\\text{jack}}$');
      const sBo = UI.stat(stats, '$\\widehat{\\se}_{\\text{boot}}$ ($B=1000$)');
      const sD = UI.stat(stats, 'Distinct $\\hat\\theta_{(i)}$');
      const note = UI.note(stats, '');
      let x = [];
      function newSample() { const P = POPS[pop.get()]; x = ctx.rng.array(Math.round(n.get()), () => P.sample(ctx.rng)); draw(); }
      function draw() {
        const f = JST[stat.get()].f, T = f(x), loo = looValues(x, f), J = jackStats(T, loo);
        const [x0, x1] = S.extent(x), padx = (x1 - x0) * 0.06 || 0.1;
        const [l0, l1] = S.extent(loo.concat([T])), padl = (l1 - l0) * 0.12 || 0.05;
        pL.setDomain([x0 - padx, x1 + padx], [l0 - padl, l1 + padl]).clear();
        pL.hline(T, { color: 's2', width: 1.5 });
        pL.hline(J.mdot, { color: 's4', width: 1.5, dash: '2 3' });
        pL.points(x.map((v, i) => [v, loo[i]]), { color: 's1', r: 3.5, opacity: 0.85 });
        pL.legend([{ label: 'θ̂ (all data)', color: 's2', dash: true }, { label: 'mean of θ̂(i)', color: 's4', dash: true }]);
        const ps = J.pseudo, [p0, p1] = S.extent(ps.concat(x)), padp = (p1 - p0) * 0.08 || 0.1;
        pR.setDomain([x0 - padx, x1 + padx], [p0 - padp, p1 + padp]).clear();
        pR.line([[x0 - padx, x0 - padx], [x1 + padx, x1 + padx]], { color: 'faint', width: 1, dash: true });
        pR.hline(J.jack, { color: 's3', width: 1.5 });
        pR.points(x.map((v, i) => [v, ps[i]]), { color: 's5', r: 3.5, opacity: 0.85 });
        pR.legend([{ label: 'pseudo-values', color: 's5' }, { label: 'θ̂_jack = their mean', color: 's3', dash: true }, { label: 'y = x', color: 'faint', dash: true }]);
        let bs = 0, bss = 0; const BB = 1000;
        for (let b = 0; b < BB; b++) { const v = f(resample(ctx.rng, x)); bs += v; bss += v * v; }
        const seB = Math.sqrt(Math.max(0, bss / BB - (bs / BB) ** 2));
        sT.set(S.fmt(T)); sB.set(S.fmt(J.bias)); sJ.set(S.fmt(J.jack)); sSe.set(S.fmt(Math.sqrt(J.v))); sBo.set(S.fmt(seB));
        const distinct = new Set(loo.map(v => v.toPrecision(10))).size;
        sD.set(String(distinct), 'of n = ' + x.length);
        const k = stat.get();
        note.set(k === 'pvar' ? `Check: $s^2 = ${S.fmt(S.variance(x))}$ equals $\\hat\\theta_{\\text{jack}}$ (V-statistic $\\to$ U-statistic).`
          : k === 'mean' ? 'Pseudo-values lie exactly on $y=x$; the bias estimate is zero and $\\widehat{\\se}_{\\text{jack}} = s/\\sqrt n$.'
          : k === 'median' ? 'Only two (or three) distinct leave-one-out values: $v_{\\text{jack}}$ depends on one spacing between order statistics. Resample a few times and watch $\\widehat{\\se}_{\\text{jack}}$ jump around relative to the bootstrap.'
          : k === 'logmean' ? 'Concave $g$: $\\E\\log\\bar X < \\log\\mu$, a bias of order $1/n$ that the jackknife estimates and removes.'
          : 'A smooth statistic: leave-one-out values vary continuously with $X_i$ (flat in the trimmed tails).');
      }
      newSample();
    },
  });

  Atlas.lab('jackknife-median-failure', {
    title: 'Is the jackknife variance consistent?',
    blurb: 'Each experiment draws a fresh sample and records $n\\,v_{\\text{jack}}$ divided by the true asymptotic variance ($\\sigma^2$ for the mean, $1/(4f(\\xi)^2)$ for the median). A consistent estimator piles up at 1 as $n$ grows. For the median the ratio follows the $E^2$ law (curve) at every $n$; switch on the bootstrap to see it concentrate instead.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const TARGET = { normal: { mean: 1, median: Math.PI / 2 }, exp: { mean: 1, median: 1 }, uniform: { mean: 1 / 12, median: 0.25 } };
      const pop = UI.select(controls, { label: 'Population', options: ['normal', 'exp', 'uniform'].map(k => ({ value: k, label: POPS[k].label })), value: 'normal', onChange: reset });
      const stat = UI.segmented(controls, { label: 'Statistic', options: [{ value: 'median', label: 'Median' }, { value: 'mean', label: 'Mean' }], value: 'median', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 1000, step: 10, value: 100, onChange: reset });
      const boot = UI.toggle(controls, { label: 'Also bootstrap ($B=200$)', value: false, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 5], y: [0, 2], xlabel: 'n · v / (true asymptotic variance)', ylabel: 'density', title: 'Distribution of the scaled variance estimate' });
      const sN = UI.stat(stats, 'Experiments');
      const sM = UI.stat(stats, 'Mean ratio (jackknife)', { accent: 'accent' });
      const sW = UI.stat(stats, 'Within ±25% of 1 (jackknife)');
      const sBo = UI.stat(stats, 'Within ±25% of 1 (bootstrap)');
      let rJ, rB;
      function reset() { rJ = []; rB = []; loop.stop(); draw(); }
      function experiment() {
        const P = POPS[pop.get()], nn = Math.round(n.get()), isMed = stat.get() === 'median', tgt = TARGET[pop.get()][stat.get()];
        const x = ctx.rng.array(nn, () => P.sample(ctx.rng));
        let v;
        if (isMed) {
          const s = sortNum(x), loo = new Array(nn);
          for (let k = 0; k < nn; k++) loo[k] = looMedianSorted(s, k);
          v = jackStats(S.quantile(s, 0.5, true), loo).v;
        } else v = S.variance(x) / nn;
        rJ.push(nn * v / tgt);
        if (boot.get()) {
          const f = isMed ? (a => S.quantile(sortNum(a), 0.5, true)) : S.mean, reps = new Array(200);
          for (let b = 0; b < 200; b++) reps[b] = f(resample(ctx.rng, x));
          rB.push(nn * S.variance(reps) / tgt);
        }
      }
      function step() {
        const t0 = Date.now();
        do experiment(); while (Date.now() - t0 < 14 && rJ.length < 4000);
        draw();
        if (rJ.length >= 4000) return false;
      }
      function draw() {
        p.clear();
        const hJ = S.histogram(rJ, { bins: 50, lo: 0, hi: 5 });
        const hB = S.histogram(rB, { bins: 50, lo: 0, hi: 5 });
        const isMed = stat.get() === 'median';
        const curve = S.linspace(0.02, 5, 250).map(y => [y, Math.exp(-Math.sqrt(y)) / (2 * Math.sqrt(y))]);
        const ymax = Math.max(1.2, ...hJ.map(b => b.density), ...hB.map(b => b.density)) * 1.1;
        p.setDomain([0, 5], [0, Math.min(ymax, 8)]);
        p.vline(1, { color: 'ink', width: 1.5 });
        if (rJ.length) p.bars(hJ, { color: 's1', opacity: 0.5 });
        if (rB.length) p.bars(hB, { color: 's3', opacity: 0.45 });
        if (isMed) p.line(curve, { color: 'bad', width: 2 });
        const leg = [{ label: 'jackknife', color: 's1', fill: true }];
        if (boot.get()) leg.push({ label: 'bootstrap', color: 's3', fill: true });
        if (isMed) leg.push({ label: 'E² limit, E ~ Exp(1)', color: 'bad' });
        p.legend(leg);
        const within = a => (a.length ? (100 * a.filter(r => Math.abs(r - 1) <= 0.25).length / a.length).toFixed(1) + '%' : '—');
        sN.set(String(rJ.length));
        sM.set(rJ.length ? S.fmt(S.mean(rJ), 2) : '—', isMed ? 'limit law has mean 2' : 'should approach 1');
        sW.set(within(rJ), isMed ? 'limit: P(0.866 ≤ E ≤ 1.118) ≈ 9.4%' : '');
        sBo.set(boot.get() ? within(rB) : 'off');
      }
      reset();
    },
  });

  /* ===================================================================== bca-intervals */
  const Phi = x => S.cdf.normal(x);
  const qN = p => S.qnorm(p);
  // BCa ingredients from sorted replicates, the observed statistic, and jackknife values.
  function bcaParts(sorted, theta, loo) {
    const B = sorted.length;
    let below = 0, eq = 0;
    for (const v of sorted) { if (v < theta) below++; else if (v === theta) eq++; }
    const p0 = S.clamp((below + 0.5 * eq) / B, 0.5 / B, 1 - 0.5 / B);
    const z0 = qN(p0);
    const md = S.mean(loo);
    let s2 = 0, s3 = 0;
    for (const v of loo) { const u = md - v; s2 += u * u; s3 += u * u * u; }
    const a = s2 > 0 ? s3 / (6 * Math.pow(s2, 1.5)) : 0;
    return { z0, a };
  }
  function bcaLevel(z0, a, alpha) {
    const w = z0 + qN(alpha), den = 1 - a * w;
    if (den <= 0) return alpha < 0.5 ? 0 : 1;
    return Phi(z0 + w / den);
  }
  const qSorted = (s, p) => S.quantile(s, S.clamp(p, 0, 1), true);
  // Statistics with an analytic standard error (for the bootstrap-t).
  const BCA_STATS = {
    mean: { label: 'Mean', f: S.mean, se: a => S.sd(a) / Math.sqrt(a.length), truth: P => P.mean },
    variance: {
      label: 'Variance', f: a => S.variance(a),
      se: a => { const n = a.length, m = S.mean(a); let m2 = 0, m4 = 0; for (const v of a) { const d = (v - m) ** 2; m2 += d; m4 += d * d; } m2 /= n; m4 /= n; return Math.sqrt(Math.max(m4 - m2 * m2, 1e-300) / n); },
      truth: P => P.sd * P.sd,
    },
  };

  Atlas.lab('bca-intervals-anatomy', {
    title: 'Anatomy of a BCa interval',
    blurb: 'Left: bootstrap replicates, the share below $\\hat\\theta$ (which sets $z_0$), and the percentile versus BC$_a$ endpoints. Right: the map from nominal level $\\alpha$ to the percentile level BC$_a$ actually uses. With $z_0 = a = 0$ it is the diagonal; skewed data bend it.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population $F$', options: ['normal', 'exp', 'lognormal', 'uniform'].map(k => ({ value: k, label: POPS[k].label })), value: 'lognormal', onChange: newSample });
      const stat = UI.segmented(controls, { label: 'Statistic', options: [{ value: 'mean', label: 'Mean' }, { value: 'variance', label: 'Variance' }, { value: 'median', label: 'Median' }], value: 'mean', onChange: draw });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 8, max: 150, step: 1, value: 25, onChange: newSample });
      const conf = UI.segmented(controls, { label: 'Confidence', options: ['90%', '95%'], value: '95%', onChange: draw });
      UI.button(controls, 'New sample', newSample, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'θ̂*', ylabel: 'density', title: 'Bootstrap replicates (B = 3000)' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 1], xlabel: 'nominal level α', ylabel: 'percentile level used', title: 'BCa level adjustment' });
      const sZ = UI.stat(stats, '$\\hat z_0$', { accent: 'accent' });
      const sA = UI.stat(stats, '$\\hat a$', { accent: 'accent' });
      const sLv = UI.stat(stats, 'Levels used $(\\alpha_1,\\alpha_2)$');
      const sP = UI.stat(stats, 'Percentile CI');
      const sBC = UI.stat(stats, 'BC$_a$ CI');
      const sTh = UI.stat(stats, 'True $\\theta$');
      let x = [];
      function newSample() { const P = POPS[pop.get()]; x = ctx.rng.array(Math.round(n.get()), () => P.sample(ctx.rng)); draw(); }
      function draw() {
        const P = POPS[pop.get()], k = stat.get();
        const f = k === 'median' ? S.median : BCA_STATS[k].f;
        const th = k === 'median' ? P.median : BCA_STATS[k].truth(P);
        const T = f(x), BB = 3000, reps = new Array(BB);
        for (let b = 0; b < BB; b++) reps[b] = f(resample(ctx.rng, x));
        reps.sort((u, v) => u - v);
        const { z0, a } = bcaParts(reps, T, looValues(x, f));
        const al = conf.get() === '95%' ? 0.025 : 0.05;
        const l1 = bcaLevel(z0, a, al), l2 = bcaLevel(z0, a, 1 - al);
        const pc = [qSorted(reps, al), qSorted(reps, 1 - al)], bc = [qSorted(reps, l1), qSorted(reps, l2)];
        let lo = qSorted(reps, 0.002), hi = qSorted(reps, 0.998);
        lo = Math.min(lo, th); hi = Math.max(hi, th);
        const pad = (hi - lo) * 0.06 || 0.1;
        const hist = S.histogram(reps, { bins: 60, lo: lo - pad, hi: hi + pad });
        pL.setDomain([lo - pad, hi + pad], [0, Math.max(...hist.map(b => b.density)) * 1.25 || 1]).clear();
        pL.bars(hist.map(b => Object.assign({}, b, { color: b.x1 <= T ? 's1' : 's2' })), { opacity: 0.5 });
        pL.band(pc[0], pc[1], { color: 's2', opacity: 0.08 });
        pL.vline(pc[0], { color: 's2', width: 1.5 }); pL.vline(pc[1], { color: 's2', width: 1.5 });
        pL.vline(bc[0], { color: 's5', width: 2.5, dash: false }); pL.vline(bc[1], { color: 's5', width: 2.5, dash: false });
        pL.vline(T, { color: 'ink', width: 1.5, dash: '2 3' });
        pL.vline(th, { color: 'ink', width: 2, dash: false });
        pL.legend([{ label: 'below θ̂ (sets z0)', color: 's1', fill: true }, { label: 'percentile ends', color: 's2', dash: true }, { label: 'BCa ends', color: 's5' }, { label: 'true θ', color: 'ink' }]);
        pR.clear();
        pR.line([[0, 0], [1, 1]], { color: 'faint', width: 1, dash: true });
        const grid = S.linspace(0.002, 0.998, 200);
        pR.line(grid.map(g => [g, bcaLevel(z0, a, g)]), { color: 's5', width: 2.5 });
        pR.dot(al, l1, { color: 's5', r: 5 }); pR.dot(1 - al, l2, { color: 's5', r: 5 });
        pR.segment(al, 0, al, l1, { color: 'muted', width: 1, dash: true }); pR.segment(1 - al, 0, 1 - al, l2, { color: 'muted', width: 1, dash: true });
        pR.legend([{ label: 'BCa', color: 's5' }, { label: 'percentile (identity)', color: 'faint', dash: true }], { left: true });
        const ci = I => `[${S.fmt(I[0], 3)}, ${S.fmt(I[1], 3)}]`;
        const cov = I => (I[0] <= th && th <= I[1] ? '<span style="color:var(--good)">covers θ</span>' : '<span style="color:var(--bad)">misses θ</span>');
        sZ.set(S.fmt(z0, 3), `${(100 * Phi(z0)).toFixed(1)}% of replicates below θ̂`);
        sA.set(S.fmt(a, 4));
        sLv.set(`(${(100 * l1).toFixed(1)}%, ${(100 * l2).toFixed(1)}%)`, `instead of (${100 * al}%, ${100 * (1 - al)}%)`);
        sP.set(ci(pc), cov(pc)); sBC.set(ci(bc), cov(bc)); sTh.set(S.fmt(th));
      }
      newSample();
    },
  });

  Atlas.lab('bca-intervals-coverage', {
    title: 'Where do the misses fall?',
    blurb: 'Repeat: sample, bootstrap, build five nominal-95% intervals, and record whether $\\theta$ fell <em>below</em> the interval (left miss) or <em>above</em> it (right miss). Each should be 2.5%. First-order methods balance their total but not their tails; BC$_a$ and the bootstrap-$t$ fix the tails.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Population', options: ['exp', 'lognormal', 'normal', 'uniform'].map(k => ({ value: k, label: POPS[k].label })), value: 'lognormal', onChange: reset });
      const stat = UI.segmented(controls, { label: 'Statistic', options: Object.keys(BCA_STATS).map(k => ({ value: k, label: BCA_STATS[k].label })), value: 'mean', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 8, max: 80, step: 1, value: 15, onChange: reset });
      const B = UI.slider(controls, { label: 'Resamples $B$', min: 400, max: 3000, step: 100, value: 1000, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const M = [['Percentile', 's2'], ['Basic', 's3'], ['BC', 's4'], ['BCa', 's5'], ['Boot-t', 's1']];
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 5], y: [0, 0.15], ylabel: 'miss rate', title: 'Left misses (θ below interval, light) and right misses (θ above, dark)', xticks: 0 });
      const sN = UI.stat(stats, 'Experiments');
      const sTot = UI.stat(stats, 'Total coverage: Pct / BCa / Boot-t');
      const sW = UI.stat(stats, 'Median width: Pct / BCa / Boot-t');
      let L, Rr, count, widths;
      function reset() { L = M.map(() => 0); Rr = M.map(() => 0); count = 0; widths = M.map(() => []); loop.stop(); draw(); }
      function experiment() {
        const P = POPS[pop.get()], st = BCA_STATS[stat.get()], th = st.truth(P);
        const nn = Math.round(n.get()), BB = Math.round(B.get());
        const x = ctx.rng.array(nn, () => P.sample(ctx.rng));
        const T = st.f(x), se = st.se(x), reps = new Array(BB), zs = [];
        for (let b = 0; b < BB; b++) {
          const xs = resample(ctx.rng, x), t = st.f(xs), s = st.se(xs);
          reps[b] = t;
          if (s > 0) zs.push((t - T) / s);
        }
        reps.sort((u, v) => u - v); zs.sort((u, v) => u - v);
        const { z0, a } = bcaParts(reps, T, looValues(x, st.f));
        const q = pp => qSorted(reps, pp);
        const ints = [
          [q(0.025), q(0.975)],
          [2 * T - q(0.975), 2 * T - q(0.025)],
          [q(bcaLevel(z0, 0, 0.025)), q(bcaLevel(z0, 0, 0.975))],
          [q(bcaLevel(z0, a, 0.025)), q(bcaLevel(z0, a, 0.975))],
          [T - qSorted(zs, 0.975) * se, T - qSorted(zs, 0.025) * se],
        ];
        ints.forEach((I, k) => { if (th < I[0]) L[k]++; else if (th > I[1]) Rr[k]++; widths[k].push(I[1] - I[0]); });
        count++;
      }
      function step() {
        const t0 = Date.now();
        do experiment(); while (Date.now() - t0 < 14 && count < 3000);
        draw();
        if (count >= 3000) return false;
      }
      function draw() {
        p.clear();
        p.hline(0.025, { color: 'ink', dash: '4 4', width: 1.2 });
        p.text(4.98, 0.025, 'target 2.5% per tail', { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        let ymax = 0.15;
        if (count) ymax = Math.max(0.08, ...L.map(v => v / count), ...Rr.map(v => v / count)) * 1.18;
        p.setDomain([0, 5], [0, ymax]);
        M.forEach(([name, c], k) => {
          if (count) {
            const l = L[k] / count, r = Rr[k] / count;
            p.bars([{ x0: k + 0.12, x1: k + 0.49, y: l }], { color: c, opacity: 0.35 });
            p.bars([{ x0: k + 0.51, x1: k + 0.88, y: r }], { color: c, opacity: 0.85 });
            p.text(k + 0.3, l, (100 * l).toFixed(1), { anchor: 'middle', dy: -5, size: 11 });
            p.text(k + 0.7, r, (100 * r).toFixed(1), { anchor: 'middle', dy: -5, size: 11 });
          }
          p.text(k + 0.5, 0, name, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        });
        sN.set(String(count));
        const tot = k => (count ? (100 * (1 - (L[k] + Rr[k]) / count)).toFixed(1) + '%' : '—');
        sTot.set(`${tot(0)} / ${tot(3)} / ${tot(4)}`);
        const mw = k => (widths[k].length ? S.fmt(S.median(widths[k]), 2) : '—');
        sW.set(`${mw(0)} / ${mw(3)} / ${mw(4)}`);
      }
      reset();
    },
  });

  /* ===================================================================== parametric-bootstrap */
  // Draw the two-tailed miss chart used by several coverage labs.
  function drawMissBars(p, names, colors, L, Rr, count, title) {
    p.clear();
    let ymax = 0.15;
    if (count) ymax = Math.max(0.08, ...L.map(v => v / count), ...Rr.map(v => v / count)) * 1.18;
    p.setDomain([0, names.length], [0, ymax]);
    p.hline(0.025, { color: 'ink', dash: '4 4', width: 1.2 });
    p.text(names.length - 0.02, 0.025, 'target 2.5% per tail', { anchor: 'end', dy: -5, size: 11, color: 'muted' });
    names.forEach((name, k) => {
      if (count) {
        const l = L[k] / count, r = Rr[k] / count;
        p.bars([{ x0: k + 0.12, x1: k + 0.49, y: l }], { color: colors[k], opacity: 0.35 });
        p.bars([{ x0: k + 0.51, x1: k + 0.88, y: r }], { color: colors[k], opacity: 0.85 });
        p.text(k + 0.3, l, (100 * l).toFixed(1), { anchor: 'middle', dy: -5, size: 11 });
        p.text(k + 0.7, r, (100 * r).toFixed(1), { anchor: 'middle', dy: -5, size: 11 });
      }
      p.text(k + 0.5, 0, name, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
    });
  }

  Atlas.lab('parametric-bootstrap-exponential', {
    title: 'Exponential rate: parametric vs. nonparametric',
    blurb: 'The model is $X_i\\iid\\text{Exp}(\\lambda)$ with $\\hat\\lambda = 1/\\bar X$; the target is $\\lambda = 1/\\E X$. Light bars: $\\lambda$ fell below the interval; dark bars: above. When the model is right the parametric pivot interval is exact even at $n=3$. Switch the true distribution to a gamma (underdispersed) or lognormal (overdispersed) and watch the parametric methods break while the nonparametric ones keep roughly their (imperfect) calibration.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const TRUE = {
        exp: { label: 'Exponential (model correct)', draw: r => r.exp(1) },
        gamma: { label: 'Gamma(3), mean 1 (underdispersed)', draw: r => r.gamma(3, 1 / 3) },
        lnorm: { label: 'Lognormal, mean 1 (overdispersed)', draw: r => r.lognormal(-0.5, 1) },
      };
      const truth = UI.select(controls, { label: 'True distribution (mean 1, so $\\lambda=1$)', options: Object.keys(TRUE).map(k => ({ value: k, label: TRUE[k].label })), value: 'exp', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 3, max: 60, step: 1, value: 6, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const NAMES = ['Wald (delta)', 'Param pct', 'Param pivot', 'Nonpar pct', 'Nonpar basic'];
      const COLS = ['s1', 's2', 's5', 's3', 's4'];
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 5], y: [0, 0.15], ylabel: 'miss rate', title: 'Nominal 95% intervals for λ: left (light) and right (dark) misses', xticks: 0 });
      const sN = UI.stat(stats, 'Experiments');
      const sC = UI.stat(stats, 'Coverage: Param pivot', { accent: 'accent' });
      const sC2 = UI.stat(stats, 'Coverage: Nonpar pct');
      const sSe = UI.stat(stats, 'Mean SE ratio (param / nonpar)');
      let L, Rr, count, seRatio;
      const BB = 800;
      function reset() { L = NAMES.map(() => 0); Rr = NAMES.map(() => 0); count = 0; seRatio = 0; loop.stop(); draw(); }
      function experiment() {
        const nn = Math.round(n.get()), D = TRUE[truth.get()];
        const x = ctx.rng.array(nn, () => D.draw(ctx.rng));
        const lam = 1 / S.mean(x), th = 1;
        const par = new Array(BB), np = new Array(BB);
        for (let b = 0; b < BB; b++) {
          par[b] = 1 / ctx.rng.gamma(nn, 1 / (nn * lam)); // mean of n Exp(lam) draws ~ Gamma(n, 1/(n lam))
          let s = 0; for (let i = 0; i < nn; i++) s += x[(ctx.rng.u() * nn) | 0];
          np[b] = nn / s;
        }
        par.sort((a, b) => a - b); np.sort((a, b) => a - b);
        const qp = pp => S.quantile(par, pp, true), qn = pp => S.quantile(np, pp, true);
        const ints = [
          [lam - 1.96 * lam / Math.sqrt(nn), lam + 1.96 * lam / Math.sqrt(nn)],
          [qp(0.025), qp(0.975)],
          [lam * lam / qp(0.975), lam * lam / qp(0.025)],
          [qn(0.025), qn(0.975)],
          [2 * lam - qn(0.975), 2 * lam - qn(0.025)],
        ];
        ints.forEach((I, k) => { if (th < I[0]) L[k]++; else if (th > I[1]) Rr[k]++; });
        const sdn = S.sd(np);
        if (sdn > 0 && Number.isFinite(sdn)) seRatio += S.sd(par) / sdn;
        count++;
      }
      function step() {
        const t0 = Date.now();
        do experiment(); while (Date.now() - t0 < 14 && count < 4000);
        draw();
        if (count >= 4000) return false;
      }
      function draw() {
        drawMissBars(p, NAMES, COLS, L, Rr, count);
        const cov = k => (count ? (100 * (1 - (L[k] + Rr[k]) / count)).toFixed(1) + '%' : '—');
        sN.set(String(count)); sC.set(cov(2)); sC2.set(cov(3));
        sSe.set(count ? S.fmt(seRatio / count, 2) : '—', 'param SE uses only the model');
      }
      reset();
    },
  });

  Atlas.lab('parametric-bootstrap-max', {
    title: 'The uniform maximum: parametric bootstrap is exact',
    blurb: 'For $X_i\\iid\\text{Unif}(0,\\theta)$ and $\\hat\\theta = X_{(n)}$, the true law of $n(\\theta - X_{(n)})/\\theta$ (curve) is close to Exp(1). The nonparametric bootstrap puts an atom of mass $1-(1-1/n)^n\\approx 63\\%$ at zero; the parametric bootstrap, simulating $\\text{Unif}(0,\\hat\\theta)$, reproduces the curve exactly because $X_{(n)}/\\theta$ is a pivot.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 100, step: 1, value: 30, onChange: newSample });
      const B = UI.slider(controls, { label: 'Resamples $B$', min: 500, max: 5000, step: 500, value: 2000, onChange: draw });
      UI.button(controls, 'New sample', newSample, { primary: true });
      stage.classList.add('two');
      const pN = new Plot(stage, { width: 420, height: 270, x: [0, 6], y: [0, 1.2], xlabel: 'n(θ̂ − θ̂*)/θ̂', ylabel: 'density', title: 'Nonparametric bootstrap' });
      const pP = new Plot(stage, { width: 420, height: 270, x: [0, 6], y: [0, 1.2], xlabel: 'n(θ̂ − θ̂*)/θ̂', ylabel: 'density', title: 'Parametric bootstrap' });
      const sA = UI.stat(stats, 'Mass at 0 (nonpar)', { accent: 'accent' });
      const sAt = UI.stat(stats, '$1-(1-1/n)^n$');
      const sCovN = UI.stat(stats, '95% CI coverage, nonpar');
      const sCovP = UI.stat(stats, '95% CI coverage, param');
      const note = UI.note(stats, '');
      let x = [];
      const trueDens = (t, nn) => (t < nn ? (1 - t / nn) ** (nn - 1) : 0); // density of n(1 - X_(n))
      function newSample() { x = ctx.rng.array(Math.round(n.get()), () => ctx.rng.uniform()); draw(); }
      function bootD(xx, BB, param, r) {
        const nn = xx.length, th = Math.max(...xx), out = new Array(BB);
        for (let b = 0; b < BB; b++) {
          let m;
          if (param) m = th * Math.pow(r.u(), 1 / nn);
          else { m = -Infinity; for (let i = 0; i < nn; i++) { const v = xx[(r.u() * nn) | 0]; if (v > m) m = v; } }
          out[b] = nn * (th - m) / th;
        }
        return out;
      }
      const covCache = {};
      function coverage(param) {
        const key = param + ':' + Math.round(n.get());
        if (covCache[key] == null) covCache[key] = coverageRun(param);
        return covCache[key];
      }
      function coverageRun(param) {
        const r = S.rng(4242), nn = Math.round(n.get()), reps = 400;
        let hit = 0;
        for (let e = 0; e < reps; e++) {
          const xx = r.array(nn, () => r.u()), th = Math.max(...xx);
          const d = bootD(xx, 300, param, r).sort((a, b) => a - b);
          // pivot n(theta - th)/theta approx by d: theta in [th / (1 - q_lo/n), th / (1 - q_hi/n)]
          const lo = th / (1 - S.quantile(d, 0.025, true) / nn), hi = th / (1 - S.quantile(d, 0.975, true) / nn);
          if (lo <= 1 && 1 <= hi) hit++;
        }
        return hit / reps;
      }
      function draw() {
        const nn = x.length, BB = Math.round(B.get());
        const dN = bootD(x, BB, false, ctx.rng), dP = bootD(x, BB, true, ctx.rng);
        const curve = S.linspace(0, 6, 200).map(t => [t, trueDens(t, nn)]);
        const atom = dN.filter(v => v === 0).length / BB;
        for (const [pl, d, col] of [[pN, dN, 's3'], [pP, dP, 's5']]) {
          const h = S.histogram(d.filter(v => v > 0), { bins: 40, lo: 0, hi: 6 });
          for (const b of h) b.density = b.count / (BB * (b.x1 - b.x0));
          pl.clear();
          pl.bars(h, { color: col, opacity: 0.5 });
          pl.line(curve, { color: 'ink', width: 2 });
          pl.legend([{ label: 'bootstrap (continuous part)', color: col, fill: true }, { label: 'true law', color: 'ink' }]);
        }
        if (atom > 0) {
          pN.rect(0, 0, 0.12, 1.2, { color: 'bad', opacity: 0.8 });
          pN.text(0.18, 1.08, `atom at 0: ${(100 * atom).toFixed(1)}% of replicates`, { color: 'bad', size: 12, weight: 600 });
        }
        sA.set((100 * atom).toFixed(1) + '%');
        sAt.set((100 * (1 - (1 - 1 / nn) ** nn)).toFixed(1) + '%');
        sCovN.set((100 * coverage(false)).toFixed(1) + '%', '400 experiments, B = 300');
        sCovP.set((100 * coverage(true)).toFixed(1) + '%', '400 experiments, B = 300');
        note.set('Interval: invert the pivot $n(\\theta - X_{(n)})/\\theta$ using bootstrap quantiles of $n(\\hat\\theta-\\hat\\theta^*)/\\hat\\theta$. The nonparametric lower quantile is 0 (the atom), so its interval is anchored at $X_{(n)}$ and is too short.');
      }
      newSample();
    },
  });

  /* ===================================================================== bootstrap-regression */
  function slr(x, y) { // simple linear regression pieces
    const n = x.length, mx = S.mean(x), my = S.mean(y);
    let sxx = 0, sxy = 0;
    for (let i = 0; i < n; i++) { const d = x[i] - mx; sxx += d * d; sxy += d * (y[i] - my); }
    const b1 = sxy / sxx, b0 = my - b1 * mx;
    return { n, mx, sxx, b0, b1 };
  }

  Atlas.lab('bootstrap-regression-hetero', {
    title: 'Heteroscedastic regression: which bootstrap SE is right?',
    blurb: 'Data follow $y = 1 + 2x + \\sigma(x)\\varepsilon$ with $\\sigma(x)^2 = 0.25(1 + \\gamma x^2)$. Each experiment draws a fresh data set, computes five standard errors for the slope, and records whether $\\hat\\beta_1 \\pm 1.96\\,\\widehat{\\se}$ covers 2. At $\\gamma = 0$ everything works; increase $\\gamma$ and the classical and residual-bootstrap intervals fall away from 95% while the sandwich, pairs and wild bootstraps hold. A normal or skewed design (more high-leverage points) makes it worse.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const DES = {
        unif: { label: 'Uniform(−1.7, 1.7)', draw: r => r.uniform(-1.7, 1.7) },
        normal: { label: 'Normal(0, 1)', draw: r => r.normal() },
        skew: { label: 'Skewed: Exp(1) − 1', draw: r => r.exp(1) - 1 },
      };
      const gam = UI.slider(controls, { label: 'Heteroscedasticity $\\gamma$', min: 0, max: 8, step: 0.25, value: 4, onChange: reset });
      const des = UI.select(controls, { label: 'Design distribution of $x$', options: Object.keys(DES).map(k => ({ value: k, label: DES[k].label })), value: 'normal', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 15, max: 200, step: 5, value: 40, onChange: reset });
      const B = UI.slider(controls, { label: 'Resamples $B$', min: 100, max: 1000, step: 50, value: 300, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'y', title: 'Latest data set and OLS fit' });
      const pC = new Plot(stage, { width: 420, height: 270, x: [0, 5], y: [0.6, 1], ylabel: 'coverage', title: 'Coverage of slope ± 1.96 SE', xticks: 0 });
      const M = [['Classical', 's1'], ['HC0', 's6'], ['Residual', 's2'], ['Pairs', 's3'], ['Wild', 's5']];
      const sN = UI.stat(stats, 'Experiments');
      const sSD = UI.stat(stats, 'Actual SD of $\\hat\\beta_1$');
      const sR = UI.stat(stats, 'Mean SE / actual SD', { accent: 'accent' });
      const sK = UI.stat(stats, 'Theory: classical SE too small by');
      let hits, ses, b1s, count, last;
      const sig = (xx, g) => 0.5 * Math.sqrt(1 + g * xx * xx);
      let theo = 1;
      function theory() {
        const g = gam.get();
        // Theory: variance ratio sum d^2 s_i^2 / (mean s^2 * sum d^2) estimated by Monte Carlo on the design
        const r = S.rng(11), D = DES[des.get()], xx = r.array(20000, () => D.draw(r)), mx = S.mean(xx);
        let a = 0, b = 0, c = 0;
        for (const v of xx) { const dd = (v - mx) ** 2, s = sig(v, g) ** 2; a += dd * s; b += dd; c += s; }
        theo = Math.sqrt((a / xx.length) / ((c / xx.length) * (b / xx.length)));
      }
      function reset() { theory(); hits = M.map(() => 0); ses = M.map(() => 0); b1s = []; count = 0; last = null; loop.stop(); experiment(); draw(); }
      function experiment() {
        const r = ctx.rng, nn = Math.round(n.get()), g = gam.get(), BB = Math.round(B.get()), D = DES[des.get()];
        const x = r.array(nn, () => D.draw(r)), y = x.map(v => 1 + 2 * v + sig(v, g) * r.normal());
        const F = slr(x, y), d = x.map(v => v - F.mx);
        const e = y.map((v, i) => v - F.b0 - F.b1 * x[i]);
        const s2 = S.sum(e.map(v => v * v)) / (nn - 2);
        const se = [Math.sqrt(s2 / F.sxx), Math.sqrt(S.sum(d.map((di, i) => di * di * e[i] * e[i]))) / F.sxx];
        // residual bootstrap (rescaled residuals)
        const sc = Math.sqrt(nn / (nn - 2)), er = e.map(v => v * sc);
        const et = e.map((v, i) => v / Math.sqrt(1 - 1 / nn - d[i] * d[i] / F.sxx)); // HC2-type for wild
        let a1 = 0, a2 = 0, w1 = 0, w2 = 0, p1 = 0, p2 = 0, pk = 0;
        const xs = new Array(nn), ys = new Array(nn);
        for (let b = 0; b < BB; b++) {
          let sr = 0, sw = 0;
          for (let i = 0; i < nn; i++) {
            sr += d[i] * er[(r.u() * nn) | 0];
            sw += d[i] * et[i] * (r.u() < 0.5 ? -1 : 1);
          }
          const br = sr / F.sxx, bw = sw / F.sxx;
          a1 += br; a2 += br * br; w1 += bw; w2 += bw * bw;
          for (let i = 0; i < nn; i++) { const j = (r.u() * nn) | 0; xs[i] = x[j]; ys[i] = y[j]; }
          const Fp = slr(xs, ys);
          if (Number.isFinite(Fp.b1) && Fp.sxx > 1e-12) { p1 += Fp.b1; p2 += Fp.b1 * Fp.b1; pk++; }
        }
        const vsd = (s1, s2_, k) => Math.sqrt(Math.max(0, (s2_ - s1 * s1 / k) / (k - 1)));
        se.push(vsd(a1, a2, BB), vsd(p1, p2, pk), vsd(w1, w2, BB));
        se.forEach((s, k) => { if (Math.abs(F.b1 - 2) <= 1.96 * s) hits[k]++; ses[k] += s; });
        b1s.push(F.b1);
        count++;
        last = { x, y, F };
      }
      function step() {
        const t0 = Date.now();
        do experiment(); while (Date.now() - t0 < 14 && count < 3000);
        draw();
        if (count >= 3000) return false;
      }
      function draw() {
        const g = gam.get();
        if (last) {
          const { x, y, F } = last, [x0, x1] = S.extent(x), [y0, y1] = S.extent(y);
          const pad = (x1 - x0) * 0.05;
          pD.setDomain([x0 - pad, x1 + pad], [y0 - 0.5, y1 + 0.5]).clear();
          const grid = S.linspace(x0 - pad, x1 + pad, 80);
          pD.area(grid.map(v => [v, 1 + 2 * v + 2 * sig(v, g)]), { color: 's4', opacity: 0.12, base: grid.map(v => [v, 1 + 2 * v - 2 * sig(v, g)]) });
          pD.points(x.map((v, i) => [v, y[i]]), { color: 's1', r: 3, opacity: 0.75 });
          pD.line(grid.map(v => [v, F.b0 + F.b1 * v]), { color: 's2', width: 2 });
          pD.line(grid.map(v => [v, 1 + 2 * v]), { color: 'ink', width: 1.2, dash: true });
          pD.legend([{ label: 'OLS fit', color: 's2' }, { label: 'truth ± 2σ(x)', color: 's4', fill: true, opacity: 0.25 }], { left: true });
        }
        pC.clear();
        pC.hline(0.95, { color: 'ink', dash: '4 4', width: 1.2 });
        const sdTrue = b1s.length > 2 ? S.sd(b1s) : NaN;
        M.forEach(([name, c], k) => {
          const cov = count ? hits[k] / count : NaN;
          if (count) {
            pC.bars([{ x0: k + 0.15, x1: k + 0.85, y: Math.max(0.6, cov) }], { color: c, opacity: 0.75, base: 0.6 });
            pC.text(k + 0.5, Math.max(0.6, cov), (100 * cov).toFixed(1) + '%', { anchor: 'middle', dy: -6, size: 11, weight: 600 });
          }
          pC.text(k + 0.5, 0.6, name, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        });
        sN.set(String(count));
        sSD.set(S.fmt(sdTrue), 'across experiments');
        sR.set(count > 2 ? M.map(([nm], k) => `${nm[0]}${nm[1]}: ${S.fmt(ses[k] / count / sdTrue, 2)}`).join(' · ') : '—');
        sK.set('×' + S.fmt(theo, 2), 'asymptotic true SE / classical SE');
      }
      reset();
    },
  });

  /* ===================================================================== block-bootstrap */
  function ar1(r, n, phi) {
    const x = new Array(n);
    let v = r.normal() / Math.sqrt(1 - phi * phi);
    for (let t = 0; t < n; t++) { x[t] = v; v = phi * v + r.normal(); }
    return x;
  }
  // Exact n * Var(Xbar) for a stationary AR(1) with unit innovation variance.
  function ar1nVar(n, phi) {
    const g0 = 1 / (1 - phi * phi);
    let s = 1, p = 1;
    for (let h = 1; h < n; h++) { p *= phi; s += 2 * (1 - h / n) * p; if (Math.abs(p) < 1e-14) break; }
    return g0 * s;
  }
  // Exact circular-block-bootstrap n * Var*(Xbar*) for block length l, using prefix sums.
  function cbbNVar(x, l, pre) {
    const n = x.length, m = S.mean(x);
    if (!pre) { pre = new Float64Array(n + l + 1); for (let t = 0; t < n + l; t++) pre[t + 1] = pre[t] + x[t % n]; }
    let ss = 0;
    for (let i = 0; i < n; i++) { const M = (pre[i + l] - pre[i]) / l; ss += (M - m) * (M - m); }
    return l / n * ss;
  }
  function prefixCirc(x, L) {
    const n = x.length, pre = new Float64Array(n + L + 1);
    for (let t = 0; t < n + L; t++) pre[t + 1] = pre[t] + x[t % n];
    return pre;
  }
  const ellOpt = (phi, n) => Math.pow(6 * phi * phi / (1 - phi * phi) ** 2, 1 / 3) * Math.pow(n, 1 / 3);

  Atlas.lab('block-bootstrap-ar1', {
    title: 'Block bootstrap of an AR(1) mean',
    blurb: 'Left: one circular-block resample of the series (blocks alternate colour; dotted lines mark the seams). Right: the exact circular block bootstrap SE of $\\bar X$ as a function of block length $\\ell$, against the true SE and the IID bootstrap ($\\ell = 1$). The curve climbs from the IID value toward the truth, then wanders as blocks become few. Optionally overlay Monte Carlo stationary-bootstrap estimates.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const phi = UI.slider(controls, { label: 'Autocorrelation $\\phi$', min: -0.5, max: 0.95, step: 0.05, value: 0.7, onChange: newSeries });
      const n = UI.slider(controls, { label: 'Series length $n$', min: 60, max: 1000, step: 20, value: 300, onChange: newSeries });
      const ell = UI.slider(controls, { label: 'Block length $\\ell$', min: 1, max: 60, step: 1, value: 8, onInput: draw });
      const sb = UI.toggle(controls, { label: 'Overlay stationary bootstrap ($B=300$)', value: false, onChange: draw });
      UI.button(controls, 'New series', newSeries, { primary: true });
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 270, xlabel: 't', ylabel: 'X*_t', title: 'One block-bootstrap resample' });
      const pE = new Plot(stage, { width: 420, height: 270, xlabel: 'block length ℓ', ylabel: 'SE of mean', title: 'Bootstrap SE vs. block length' });
      const sT = UI.stat(stats, 'True $\\se(\\bar X)$', { accent: 'accent' });
      const sI = UI.stat(stats, 'IID bootstrap SE ($\\ell = 1$)');
      const sB = UI.stat(stats, 'Block bootstrap SE at $\\ell$');
      const sO = UI.stat(stats, 'Theory $\\ell_{\\text{opt}}$');
      const sR = UI.stat(stats, '$\\sigma^2_\\infty/\\gamma(0) = (1+\\phi)/(1-\\phi)$');
      let x = [], pre = null, curve = [], sbPts = null;
      function newSeries() { x = ar1(ctx.rng, Math.round(n.get()), phi.get()); pre = prefixCirc(x, 61); curve = []; for (let l = 1; l <= 60; l++) curve.push([l, Math.sqrt(cbbNVar(x, l, pre) / x.length)]); sbPts = null; draw(); }
      function stationaryBoot(l) {
        const nn = x.length, p = 1 / l, BB = 300, r = ctx.rng;
        let s1 = 0, s2 = 0;
        for (let b = 0; b < BB; b++) {
          let sum = 0, t = 0, pos = (r.u() * nn) | 0;
          while (t < nn) { sum += x[pos]; t++; if (r.u() < p) pos = (r.u() * nn) | 0; else pos = (pos + 1) % nn; }
          const m = sum / nn; s1 += m; s2 += m * m;
        }
        return Math.sqrt(Math.max(0, (s2 - s1 * s1 / BB) / (BB - 1)));
      }
      function draw() {
        const nn = x.length, l = Math.round(ell.get()), ph = phi.get();
        // one resample
        const r = ctx.rng, k = Math.ceil(nn / l), xs = [], seams = [];
        for (let j = 0; j < k && xs.length < nn; j++) {
          const st = (r.u() * nn) | 0;
          for (let q = 0; q < l && xs.length < nn; q++) xs.push([xs.length, x[(st + q) % nn], j % 2]);
          seams.push(xs.length);
        }
        const [y0, y1] = S.extent(x), pad = (y1 - y0) * 0.08;
        pS.setDomain([0, nn], [y0 - pad, y1 + pad]).clear();
        if (l === 1) pS.line(xs.map(p => [p[0], p[1]]), { color: 's1', width: 1 });
        else {
          let seg = [], par = xs[0][2];
          for (const p of xs) {
            if (p[2] !== par) { pS.line(seg, { color: par ? 's2' : 's1', width: 1.3 }); seg = []; par = p[2]; }
            seg.push([p[0], p[1]]);
          }
          pS.line(seg, { color: par ? 's2' : 's1', width: 1.3 });
          if (nn / l <= 60) for (const s of seams) pS.vline(s - 0.5, { color: 'faint', width: 0.8, dash: '1 3' });
        }
        pS.hline(S.mean(x), { color: 'ink', width: 1 });
        // SE curve
        const trueSE = Math.sqrt(ar1nVar(nn, ph) / nn), iid = curve[0][1];
        const ymax = Math.max(trueSE, ...curve.map(c => c[1])) * 1.25;
        pE.setDomain([1, 60], [0, ymax]).clear();
        pE.hline(trueSE, { color: 'ink', width: 1.8, dash: false });
        pE.hline(iid, { color: 'bad', width: 1.2 });
        pE.line(curve, { color: 's2', width: 2.2 });
        const lo = ellOpt(ph, nn);
        if (ph > 0 && lo >= 1 && lo <= 60) pE.vline(lo, { color: 's5', width: 1.5 });
        pE.dot(l, curve[l - 1][1], { color: 's2', r: 5 });
        if (sb.get()) {
          if (!sbPts) sbPts = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48].map(m => [m, stationaryBoot(m)]);
          pE.points(sbPts, { color: 's4', r: 4 });
        }
        const leg = [{ label: 'circular block (exact)', color: 's2' }, { label: 'true SE', color: 'ink' }, { label: 'IID bootstrap', color: 'bad', dash: true }];
        if (ph > 0) leg.push({ label: 'theory ℓ_opt', color: 's5', dash: true });
        if (sb.get()) leg.push({ label: 'stationary bootstrap', color: 's4' });
        pE.legend(leg);
        sT.set(S.fmt(trueSE, 4)); sI.set(S.fmt(iid, 4), `ratio to truth ${S.fmt(iid / trueSE, 2)}`);
        sB.set(S.fmt(curve[l - 1][1], 4), `ratio to truth ${S.fmt(curve[l - 1][1] / trueSE, 2)}`);
        sO.set(ph > 0 ? S.fmt(lo, 1) : '—', ph > 0 ? '$(6\\phi^2/(1-\\phi^2)^2)^{1/3}n^{1/3}$' : 'needs φ > 0');
        sR.set(S.fmt((1 + ph) / (1 - ph), 2));
      }
      newSeries();
    },
  });

  Atlas.lab('block-bootstrap-tradeoff', {
    title: 'The block-length tradeoff, measured',
    blurb: 'Many independent AR(1) series; for each, the exact circular block bootstrap estimate $\\hat\\sigma^2(\\ell)$ of $n\\Var(\\bar X)$ for every $\\ell$. Plotted (relative to the truth): squared bias, variance, and their sum. Bias falls like $1/\\ell^2$, variance rises like $\\ell/n$; the dashed curve is the asymptotic MSE $\\big(\\tfrac{2\\phi}{1-\\phi^2}\\big)^2\\ell^{-2} + \\tfrac43\\ell/n$. Increase $n$ and watch the optimum move right like $n^{1/3}$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const phi = UI.slider(controls, { label: 'Autocorrelation $\\phi$', min: 0.1, max: 0.9, step: 0.05, value: 0.6, onChange: reset });
      const n = UI.slider(controls, { label: 'Series length $n$', min: 100, max: 4000, step: 100, value: 400, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Simulate series', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, x: [1, 60], y: [0, 0.5], xlabel: 'block length ℓ', ylabel: 'relative error²', title: 'Bias², variance and MSE of the block variance estimate' });
      const sN = UI.stat(stats, 'Series simulated');
      const sE = UI.stat(stats, 'Empirical best $\\ell$', { accent: 'accent' });
      const sO = UI.stat(stats, 'Theory $\\ell_{\\text{opt}}$');
      const sM = UI.stat(stats, 'Relative RMSE at best $\\ell$');
      const sI = UI.stat(stats, 'Relative bias at $\\ell=1$ (IID)');
      let L = 60, s1, s2, cnt, target;
      function reset() {
        L = Math.min(60, Math.floor(n.get() / 4));
        s1 = new Float64Array(L + 1); s2 = new Float64Array(L + 1); cnt = 0;
        target = ar1nVar(Math.round(n.get()), phi.get());
        loop.stop(); draw();
      }
      function one() {
        const nn = Math.round(n.get()), x = ar1(ctx.rng, nn, phi.get()), pre = prefixCirc(x, L + 1);
        for (let l = 1; l <= L; l++) { const v = cbbNVar(x, l, pre) / target; s1[l] += v; s2[l] += v * v; }
        cnt++;
      }
      function step() {
        const t0 = Date.now();
        do one(); while (Date.now() - t0 < 14 && cnt < 3000);
        draw();
        if (cnt >= 3000) return false;
      }
      function draw() {
        const ph = phi.get(), nn = Math.round(n.get());
        const c = ph * 2 / (1 - ph * ph);
        const theory = [];
        for (let l = 1; l <= L; l += 0.25) theory.push([l, (c / l) ** 2 + 4 / 3 * l / nn]);
        let b2 = [], vv = [], mse = [];
        if (cnt > 1) for (let l = 1; l <= L; l++) {
          const m = s1[l] / cnt, v = s2[l] / cnt - m * m;
          b2.push([l, (m - 1) ** 2]); vv.push([l, v]); mse.push([l, (m - 1) ** 2 + v]);
        }
        const best = mse.length ? mse.reduce((a, q) => (q[1] < a[1] ? q : a)) : null;
        const ymax = best ? Math.min(1, Math.max(0.05, best[1] * 6)) : 0.5;
        p.setDomain([1, L], [0, ymax]).clear();
        p.line(theory, { color: 'muted', width: 1.5, dash: true });
        if (mse.length) {
          p.line(b2, { color: 's1', width: 2 });
          p.line(vv, { color: 's3', width: 2 });
          p.line(mse, { color: 's5', width: 3 });
          p.dot(best[0], best[1], { color: 's5', r: 5 });
        }
        const lo = ellOpt(ph, nn);
        if (lo <= L) p.vline(lo, { color: 's5', width: 1.2 });
        p.legend([{ label: 'bias²', color: 's1' }, { label: 'variance', color: 's3' }, { label: 'MSE', color: 's5' }, { label: 'asymptotic MSE', color: 'muted', dash: true }]);
        sN.set(String(cnt));
        sE.set(best ? String(best[0]) : '—');
        sO.set(S.fmt(lo, 1));
        sM.set(best ? S.fmt(Math.sqrt(best[1]), 3) : '—');
        sI.set(cnt > 1 ? S.fmt(s1[1] / cnt - 1, 3) : '—', `theory ${S.fmt((1 - ph) / (1 + ph) - 1, 3)}`);
      }
      reset();
    },
  });

  /* ===================================================================== subsampling */
  // Max of a random subset of size m (without replacement) via partial Fisher–Yates on an index scratch array.
  function subMax(x, m, r, idx) {
    const n = x.length;
    let best = -Infinity;
    for (let i = 0; i < m; i++) {
      const j = i + ((r.u() * (n - i)) | 0);
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      if (x[idx[i]] > best) best = x[idx[i]];
    }
    return best;
  }
  function subMean(x, m, r, idx) {
    const n = x.length;
    let s = 0;
    for (let i = 0; i < m; i++) {
      const j = i + ((r.u() * (n - i)) | 0);
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
      s += x[idx[i]];
    }
    return s / m;
  }
  function bootMax(x, m, r) { const n = x.length; let best = -Infinity; for (let i = 0; i < m; i++) { const v = x[(r.u() * n) | 0]; if (v > best) best = v; } return best; }
  function bootMean(x, m, r) { const n = x.length; let s = 0; for (let i = 0; i < m; i++) s += x[(r.u() * n) | 0]; return s / m; }
  const ksExp = sorted => { // KS distance between a sorted sample (>= 0) and Exp(1)
    const n = sorted.length; let d = 0;
    for (let i = 0; i < n; i++) { const F = 1 - Math.exp(-sorted[i]); d = Math.max(d, Math.abs(F - i / n), Math.abs((i + 1) / n - F)); }
    return d;
  };

  Atlas.lab('subsampling-uniform-max', {
    title: 'Fixing the uniform maximum',
    blurb: 'Sample $n$ points from Unif(0,1); target law: $n(\\theta - X_{(n)})/\\theta\\approx\\text{Exp}(1)$ (curve). Left: the ordinary bootstrap, with its atom at zero. Right: subsampling ($m$ without replacement) and the $m$-out-of-$n$ bootstrap, both scaled by $m$. Push $m$ toward $n$ to watch the atom return; push it too small and the approximation gets coarse.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 50, max: 2000, step: 50, value: 500, onChange: newSample });
      const mf = UI.slider(controls, { label: 'Subsample size $m = n^{\\gamma}$, $\\gamma$', min: 0.3, max: 1, step: 0.02, value: 0.6, onChange: draw });
      UI.button(controls, 'New sample', newSample, { primary: true });
      stage.classList.add('two');
      const pB = new Plot(stage, { width: 420, height: 270, x: [0, 5], y: [0, 1.1], xlabel: 'n(θ̂ − θ̂*)/θ̂', ylabel: 'density', title: 'n-out-of-n bootstrap' });
      const pS = new Plot(stage, { width: 420, height: 270, x: [0, 5], y: [0, 1.1], xlabel: 'm(θ̂ − θ̂*_m)/θ̂', ylabel: 'density', title: 'Subsampling and m-out-of-n' });
      const sM = UI.stat(stats, '$m$');
      const sA = UI.stat(stats, 'Atom at 0: bootstrap / m-of-n / subsample', { accent: 'accent' });
      const sK = UI.stat(stats, 'KS distance to Exp(1): bootstrap / m-of-n / subsample');
      const sT = UI.stat(stats, 'TV bound $m(m-1)/2n$');
      let x = [];
      const BB = 3000;
      function newSample() { x = ctx.rng.array(Math.round(n.get()), () => ctx.rng.uniform()); draw(); }
      function dist(kind, m) {
        const th = Math.max(...x), r = ctx.rng, idx = Int32Array.from(S.range(x.length)), out = new Array(BB);
        for (let b = 0; b < BB; b++) {
          const v = kind === 'sub' ? subMax(x, m, r, idx) : bootMax(x, m, r);
          out[b] = m * (th - v) / th;
        }
        return out.sort((a, b) => a - b);
      }
      function hist(pl, d, col, opac) {
        const pos = d.filter(v => v > 0);
        const h = S.histogram(pos, { bins: 40, lo: 0, hi: 5 });
        for (const b of h) b.density = b.count / (d.length * (b.x1 - b.x0));
        pl.bars(h, { color: col, opacity: opac });
      }
      function draw() {
        const nn = x.length, m = Math.max(2, Math.min(nn, Math.round(nn ** mf.get())));
        const dB = dist('boot', nn), dM = dist('boot', m), dS = m < nn ? dist('sub', m) : dB.map(() => 0);
        const curve = S.linspace(0, 5, 200).map(t => [t, Math.exp(-t)]);
        const atom = d => d.filter(v => v === 0).length / d.length;
        pB.clear(); hist(pB, dB, 's3', 0.5); pB.line(curve, { color: 'ink', width: 2 });
        const aB = atom(dB);
        pB.rect(0, 0, 0.1, 1.1, { color: 'bad', opacity: 0.8 });
        pB.text(0.18, 1.0, `atom at 0: ${(100 * aB).toFixed(1)}%`, { color: 'bad', size: 12, weight: 600 });
        pB.legend([{ label: 'bootstrap', color: 's3', fill: true }, { label: 'Exp(1)', color: 'ink' }]);
        pS.clear(); hist(pS, dS, 's5', 0.45); hist(pS, dM, 's2', 0.3); pS.line(curve, { color: 'ink', width: 2 });
        pS.legend([{ label: 'subsampling', color: 's5', fill: true }, { label: 'm-out-of-n', color: 's2', fill: true }, { label: 'Exp(1)', color: 'ink' }]);
        sM.set(String(m), `m/n = ${S.fmt(m / nn, 3)}`);
        sA.set(`${(100 * aB).toFixed(1)}% / ${(100 * atom(dM)).toFixed(1)}% / ${m < nn ? (100 * atom(dS)).toFixed(1) + '%' : '—'}`, `theory m-of-n: ${(100 * (1 - (1 - 1 / nn) ** m)).toFixed(1)}%, subsample: ${(100 * m / nn).toFixed(1)}%`);
        sK.set(`${S.fmt(ksExp(dB), 3)} / ${S.fmt(ksExp(dM), 3)} / ${m < nn ? S.fmt(ksExp(dS), 3) : '—'}`);
        sT.set(S.fmt(Math.min(1, m * (m - 1) / (2 * nn)), 3), m * (m - 1) >= 2 * nn ? 'bound vacuous: m² is not small relative to n' : 'm-of-n vs subsampling');
      }
      newSample();
    },
  });

  Atlas.lab('subsampling-choice-of-m', {
    title: 'Coverage as a function of m',
    blurb: 'Each experiment builds nominal-95% intervals $[\\hat\\theta_n - c_{0.975}/\\tau_n,\\ \\hat\\theta_n - c_{0.025}/\\tau_n]$ from subsampling and from the $m$-out-of-$n$ bootstrap, for a grid of $m$ (the right-most point $m=n$ with replacement is the ordinary bootstrap). Uniform max: rate $n$. Boundary: $\\hat\\theta = \\max(\\bar X, 0)$ with true $\\theta = 0$, rate $\\sqrt n$. Smooth: the mean, where every method works.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const PROB = {
        max: { label: 'Uniform max (rate n)', draw: r => r.uniform(), theta: 1, stat: x => Math.max(...x), tau: k => k, sub: subMax, boot: bootMax },
        bnd: { label: 'Boundary: max(mean, 0) at θ = 0 (rate √n)', draw: r => r.normal(), theta: 0, stat: x => Math.max(S.mean(x), 0), tau: k => Math.sqrt(k), sub: (x, m, r, idx) => Math.max(subMean(x, m, r, idx), 0), boot: (x, m, r) => Math.max(bootMean(x, m, r), 0) },
        mean: { label: 'Smooth: mean of Exp(1) (rate √n)', draw: r => r.exp(1), theta: 1, stat: S.mean, tau: k => Math.sqrt(k), sub: subMean, boot: bootMean },
      };
      const prob = UI.select(controls, { label: 'Problem', options: Object.keys(PROB).map(k => ({ value: k, label: PROB[k].label })), value: 'max', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 64, max: 512, step: 64, value: 256, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, x: [2, 9], y: [0, 1], xlabel: 'm (log scale)', ylabel: 'coverage', title: 'Empirical coverage of nominal 95% intervals', xfmt: v => String(Math.round(2 ** v)) });
      const sN = UI.stat(stats, 'Experiments');
      const sBS = UI.stat(stats, 'Ordinary bootstrap ($m=n$)', { accent: 'accent' });
      const sBest = UI.stat(stats, 'Best subsampling $m$ (coverage)');
      let ms, hS, hB, count;
      const BB = 200;
      function reset() {
        const nn = Math.round(n.get());
        ms = []; for (let m = 4; m < nn; m *= 2) ms.push(m); ms.push(nn);
        hS = ms.map(() => 0); hB = ms.map(() => 0); count = 0; loop.stop(); draw();
      }
      function experiment() {
        const P = PROB[prob.get()], nn = Math.round(n.get()), r = ctx.rng;
        const x = r.array(nn, () => P.draw(r)), T = P.stat(x), tn = P.tau(nn);
        const idx = Int32Array.from(S.range(nn)), vs = new Array(BB), vb = new Array(BB);
        ms.forEach((m, k) => {
          const tm = P.tau(m);
          for (let b = 0; b < BB; b++) vb[b] = tm * (P.boot(x, m, r) - T);
          vb.sort((a, c) => a - c);
          const lo = T - S.quantile(vb, 0.975, true) / tn, hi = T - S.quantile(vb, 0.025, true) / tn;
          if (lo <= P.theta && P.theta <= hi) hB[k]++;
          if (m < nn) {
            for (let b = 0; b < BB; b++) vs[b] = tm * (P.sub(x, m, r, idx) - T);
            vs.sort((a, c) => a - c);
            const l2 = T - S.quantile(vs, 0.975, true) / tn, h2 = T - S.quantile(vs, 0.025, true) / tn;
            if (l2 <= P.theta && P.theta <= h2) hS[k]++;
          }
        });
        count++;
      }
      function step() {
        const t0 = Date.now();
        do experiment(); while (Date.now() - t0 < 14 && count < 2000);
        draw();
        if (count >= 2000) return false;
      }
      function draw() {
        const nn = Math.round(n.get());
        p.setDomain([2, Math.log2(nn) + 0.2], [0, 1.02]).clear();
        p.hline(0.95, { color: 'ink', dash: '4 4', width: 1.2 });
        if (count) {
          const lx = ms.map(m => Math.log2(m));
          const cs = ms.map((m, k) => (m < nn ? [lx[k], hS[k] / count] : null)).filter(Boolean);
          const cb = ms.map((m, k) => [lx[k], hB[k] / count]);
          p.line(cs, { color: 's5', width: 2.5 }); for (const q of cs) p.dot(q[0], q[1], { color: 's5', r: 4 });
          p.line(cb, { color: 's2', width: 2.5 }); for (const q of cb) p.dot(q[0], q[1], { color: 's2', r: 4 });
          p.dot(lx[lx.length - 1], cb[cb.length - 1][1], { color: 'bad', r: 6 });
          const bestK = cs.reduce((a, q, i) => (Math.abs(q[1] - 0.95) < Math.abs(cs[a][1] - 0.95) ? i : a), 0);
          sBest.set(`m = ${ms[bestK]} (${(100 * cs[bestK][1]).toFixed(1)}%)`);
          sBS.set((100 * cb[cb.length - 1][1]).toFixed(1) + '%');
        } else { sBest.set('—'); sBS.set('—'); }
        p.legend([{ label: 'subsampling', color: 's5' }, { label: 'm-out-of-n bootstrap', color: 's2' }, { label: 'ordinary bootstrap (m = n)', color: 'bad' }], { left: true });
        sN.set(String(count));
      }
      reset();
    },
  });

  /* ===================================================================== permutation-tests */
  function welchT(z, m) { // z pooled, first m are group X
    const N = z.length, n = N - m;
    let sx = 0, sy = 0;
    for (let i = 0; i < m; i++) sx += z[i];
    for (let i = m; i < N; i++) sy += z[i];
    const mx = sx / m, my = sy / n;
    let vx = 0, vy = 0;
    for (let i = 0; i < m; i++) vx += (z[i] - mx) ** 2;
    for (let i = m; i < N; i++) vy += (z[i] - my) ** 2;
    vx /= (m - 1); vy /= (n - 1);
    const se2 = vx / m + vy / n;
    const df = se2 * se2 / ((vx / m) ** 2 / (m - 1) + (vy / n) ** 2 / (n - 1));
    return { d: mx - my, t: (mx - my) / Math.sqrt(se2), df, vx, vy };
  }
  function meanDiff(z, m) { let a = 0, b = 0; for (let i = 0; i < m; i++) a += z[i]; for (let i = m; i < z.length; i++) b += z[i]; return a / m - b / (z.length - m); }
  const PERM_POPS = {
    normal: { label: 'Normal', draw: r => r.normal() },
    exp: { label: 'Exponential (skewed)', draw: r => r.exp(1) - 1 },
    t3: { label: 't₃ (heavy tails)', draw: r => r.t(3) },
    cauchy: { label: 'Cauchy', draw: r => r.cauchy() },
  };

  Atlas.lab('permutation-tests-null', {
    title: 'Building a permutation null distribution',
    blurb: 'Two groups (blue, orange) differ by a shift $\\delta$. Press run: each frame relabels the pooled data at random (left, current permutation) and adds the permuted statistic to the histogram (right). The Monte Carlo $p$-value $(1+\\#)/(1+B)$ settles as $B$ grows. Compare with Welch\'s $t$-test, and try Cauchy data, where the $t$-test\'s assumptions fail but the permutation test stays exact.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Distribution', options: Object.keys(PERM_POPS).map(k => ({ value: k, label: PERM_POPS[k].label })), value: 'normal', onChange: newData });
      const TS = {
        mean: { label: 'Mean diff', f: (z, m) => meanDiff(z, m) },
        median: { label: 'Median diff', f: (z, m) => S.median(z.slice(0, m)) - S.median(z.slice(m)) },
        welch: { label: 'Welch t', f: (z, m) => welchT(z, m).t },
        rank: {
          label: 'Rank sum', f: (z, m, ranks) => { let s = 0; for (let i = 0; i < m; i++) s += ranks[i]; return s - m * (z.length + 1) / 2; },
        },
      };
      const stat = UI.segmented(controls, { label: 'Statistic (two-sided, $|T|$)', options: Object.keys(TS).map(k => ({ value: k, label: TS[k].label })), value: 'mean', onChange: restart });
      const m = UI.slider(controls, { label: 'Group sizes $m = n$', min: 3, max: 40, step: 1, value: 12, onChange: newData });
      const delta = UI.slider(controls, { label: 'Shift $\\delta$', min: 0, max: 2, step: 0.05, value: 1.2, onChange: newData });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Permute', '❚❚ Pause']);
      UI.button(row, 'New data', newData);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, y: [-0.8, 1.8], yticks: 0, xlabel: 'value', title: 'Pooled data, current labels' });
      const pH = new Plot(stage, { width: 420, height: 270, xlabel: 'permuted |T|', ylabel: 'density', title: 'Permutation distribution' });
      const sB = UI.stat(stats, 'Permutations $B$');
      const sP = UI.stat(stats, '$\\hat p = (1+\\#)/(1+B)$', { accent: 'accent' });
      const sW = UI.stat(stats, 'Welch $t$-test $p$');
      const sE = UI.stat(stats, 'Distinct labellings $\\binom{2m}{m}$');
      let z = [], ranks = [], obs = 0, perms = [], ge = 0, cur = null, acc = 0;
      function newData() {
        const P = PERM_POPS[pop.get()], mm = Math.round(m.get()), d = delta.get(), r = ctx.rng;
        z = r.array(mm, () => P.draw(r) + d).concat(r.array(mm, () => P.draw(r)));
        const ord = S.range(z.length).sort((a, b) => z[a] - z[b]);
        ranks = new Array(z.length); ord.forEach((i, k) => { ranks[i] = k + 1; });
        restart();
      }
      function restart() {
        const mm = z.length / 2;
        obs = Math.abs(TS[stat.get()].f(z, mm, ranks));
        perms = []; ge = 0; cur = null; acc = 0; loop.stop(); draw();
      }
      function permuteOnce() {
        const N = z.length, mm = N / 2, idx = ctx.rng.shuffle(S.range(N));
        const zp = idx.map(i => z[i]), rp = idx.map(i => ranks[i]);
        const t = Math.abs(TS[stat.get()].f(zp, mm, rp));
        perms.push(t); if (t >= obs - 1e-12) ge++;
        cur = idx;
      }
      function step(dt) {
        acc += dt;
        const per = perms.length < 30 ? (acc > 0.12 ? 1 : 0) : Math.min(400, Math.ceil(perms.length / 15));
        if (per) acc = 0;
        for (let i = 0; i < per && perms.length < 9999; i++) permuteOnce();
        draw();
        if (perms.length >= 9999) return false;
      }
      function draw() {
        const N = z.length, mm = N / 2;
        const [lo, hi] = S.extent(z), pad = (hi - lo) * 0.05 || 1;
        const clip = v => S.clamp(v, lo - pad, hi + pad);
        pD.setDomain([lo - pad, hi + pad], [-0.8, 1.8]).clear();
        pD.text(lo - pad, 1.55, 'observed labels', { dx: 4, size: 11, color: 'muted' });
        pD.text(lo - pad, 0.55, cur ? 'current permutation' : 'press ▶ to permute', { dx: 4, size: 11, color: 'muted' });
        for (let i = 0; i < N; i++) pD.dot(clip(z[i]), i < mm ? 1.3 : 1.1, { color: i < mm ? 's1' : 's2', r: 4, opacity: 0.85 });
        if (cur) for (let k = 0; k < N; k++) pD.dot(clip(z[cur[k]]), k < mm ? 0.3 : 0.1, { color: k < mm ? 's1' : 's2', r: 4, opacity: 0.85 });
        pD.hline(0.8, { color: 'faint', width: 1, dash: false });
        if (perms.length) {
          const hiH = Math.max(obs, ...perms.slice(-2000), S.quantile(perms, 0.999)) * 1.08 || 1;
          const h = S.histogram(perms, { bins: 40, lo: 0, hi: hiH });
          pH.setDomain([0, hiH], [0, Math.max(...h.map(b => b.density)) * 1.15 || 1]).clear();
          pH.bars(h.map(b => Object.assign({}, b, { color: b.x0 >= obs ? 'bad' : 's4' })), { opacity: 0.6 });
        } else pH.setDomain([0, Math.max(obs, 1e-6) * 1.5], [0, 1]).clear();
        pH.vline(obs, { color: 'ink', width: 2, dash: false });
        pH.text(obs, pH.o.y[1] * 0.92, ' observed', { size: 11 });
        const B = perms.length, ph = (1 + ge) / (1 + B);
        sB.set(String(B));
        sP.set(B ? S.fmt(ph, 4) : '—', B ? `MC s.e. ≈ ${S.fmt(Math.sqrt(ph * (1 - ph) / B), 3)}` : '');
        const W = welchT(z, mm), pw = 2 * (1 - S.cdf.t(Math.abs(W.t), W.df));
        sW.set(S.fmt(pw, 4), `t = ${S.fmt(W.t, 2)}, df = ${S.fmt(W.df, 1)}`);
        sE.set(Math.exp(S.lchoose(N, mm)).toExponential(2).replace('e+', '×10^'));
      }
      newData();
    },
  });

  Atlas.lab('permutation-tests-behrens-fisher', {
    title: 'Permutation tests when variances differ',
    blurb: 'Both groups have the <em>same mean</em> (so $H_0:\\mu_X=\\mu_Y$ is true) but different spreads. Bars show the rejection rate at $\\alpha = 0.05$. Unbalanced sizes with the larger variance in the smaller group make the raw-difference permutation test (and the pooled $t$-test) over-reject; permuting the studentized Welch statistic restores the 5% level, as theory predicts. Equal sizes or equal variances remove the problem.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.select(controls, { label: 'Error distribution', options: ['normal', 'exp', 't3'].map(k => ({ value: k, label: PERM_POPS[k].label })), value: 'normal', onChange: reset });
      const m = UI.slider(controls, { label: 'Size of group X, $m$', min: 5, max: 60, step: 1, value: 10, onChange: reset });
      const n = UI.slider(controls, { label: 'Size of group Y, $n$', min: 5, max: 60, step: 1, value: 40, onChange: reset });
      const ratio = UI.slider(controls, { label: 'SD ratio $\\sigma_X/\\sigma_Y$', min: 0.25, max: 4, step: 0.05, value: 3, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const M = [['Perm: mean diff', 's1'], ['Perm: Welch t', 's5'], ['Pooled t-test', 's2'], ['Welch t-test', 's3']];
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 4], y: [0, 0.2], ylabel: 'rejection rate', title: 'Type I error at nominal 5% (B = 199 permutations per test)', xticks: 0 });
      const sN = UI.stat(stats, 'Experiments');
      const sV = UI.stat(stats, 'Theory: perm var / true var of $\\bar X - \\bar Y$', { accent: 'accent' });
      const sR = UI.stat(stats, 'Rates: perm-diff / perm-Welch');
      let rej, count;
      const BB = 199;
      function reset() { rej = M.map(() => 0); count = 0; loop.stop(); draw(); }
      function experiment() {
        const P = PERM_POPS[pop.get()], mm = Math.round(m.get()), nn = Math.round(n.get()), sx = ratio.get(), r = ctx.rng;
        const z = r.array(mm, () => sx * P.draw(r)).concat(r.array(nn, () => P.draw(r)));
        const N = mm + nn;
        const W = welchT(z, mm), d0 = Math.abs(W.d), t0 = Math.abs(W.t);
        let gd = 0, gt = 0;
        const zp = z.slice();
        for (let b = 0; b < BB; b++) {
          r.shuffle(zp);
          const w = welchT(zp, mm);
          if (Math.abs(w.d) >= d0) gd++;
          if (Math.abs(w.t) >= t0) gt++;
        }
        if ((1 + gd) / (1 + BB) <= 0.05) rej[0]++;
        if ((1 + gt) / (1 + BB) <= 0.05) rej[1]++;
        const sp2 = ((mm - 1) * W.vx + (nn - 1) * W.vy) / (N - 2);
        const tp = W.d / Math.sqrt(sp2 * (1 / mm + 1 / nn));
        if (2 * (1 - S.cdf.t(Math.abs(tp), N - 2)) <= 0.05) rej[2]++;
        if (2 * (1 - S.cdf.t(t0, W.df)) <= 0.05) rej[3]++;
        count++;
      }
      function step() {
        const t0 = Date.now();
        do experiment(); while (Date.now() - t0 < 14 && count < 5000);
        draw();
        if (count >= 5000) return false;
      }
      function draw() {
        const mm = Math.round(m.get()), nn = Math.round(n.get()), N = mm + nn, lam = mm / N, sx2 = ratio.get() ** 2;
        let ymax = 0.2;
        if (count) ymax = Math.max(0.12, ...rej.map(v => v / count)) * 1.2;
        p.setDomain([0, 4], [0, ymax]).clear();
        if (count) { const se = Math.sqrt(0.05 * 0.95 / count); p.rect(0, 0.05 - 2 * se, 4, 0.05 + 2 * se, { color: 'muted', opacity: 0.12 }); }
        p.hline(0.05, { color: 'ink', dash: '4 4', width: 1.2 });
        M.forEach(([name, c], k) => {
          if (count) {
            const v = rej[k] / count;
            p.bars([{ x0: k + 0.15, x1: k + 0.85, y: v }], { color: c, opacity: 0.75 });
            p.text(k + 0.5, v, (100 * v).toFixed(1) + '%', { anchor: 'middle', dy: -6, size: 12, weight: 600 });
          }
          p.text(k + 0.5, 0, name, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        });
        // variance ratio from the Romano formula (finite-sample version)
        const pooled = lam * sx2 + (1 - lam) * 1, permV = pooled * (1 / mm + 1 / nn), trueV = sx2 / mm + 1 / nn;
        sN.set(String(count));
        sV.set(S.fmt(permV / trueV, 3), permV / trueV < 1 ? 'too narrow → over-rejects' : permV / trueV > 1 ? 'too wide → conservative' : 'matches');
        sR.set(count ? `${(100 * rej[0] / count).toFixed(1)}% / ${(100 * rej[1] / count).toFixed(1)}%` : '—');
      }
      reset();
    },
  });

  /* ===================================================================== cross-validation */
  // Legendre basis row [P_0(x), ..., P_d(x)] on [-1, 1].
  function legendreRow(x, d) {
    const row = [1];
    if (d >= 1) row.push(x);
    for (let k = 2; k <= d; k++) row.push(((2 * k - 1) * x * row[k - 1] - (k - 1) * row[k - 2]) / k);
    return row;
  }
  // Least-squares fit on rows idx; returns coefficient vector or null if singular.
  function polyFit(X, y, idx) {
    const p = X[0].length, A = Array.from({ length: p }, () => new Array(p).fill(0)), b = new Array(p).fill(0);
    for (const i of idx) { const r = X[i]; for (let a = 0; a < p; a++) { b[a] += r[a] * y[i]; for (let c = 0; c <= a; c++) A[a][c] += r[a] * r[c]; } }
    for (let a = 0; a < p; a++) for (let c = a + 1; c < p; c++) A[a][c] = A[c][a];
    try { return { beta: S.solve(A, b), A }; } catch (e) { return null; }
  }
  const dotv = (r, b) => { let s = 0; for (let k = 0; k < r.length; k++) s += r[k] * b[k]; return s; };
  const cvTruth = x => Math.sin(2.5 * x) + 0.3 * x * x;

  Atlas.lab('cross-validation-curves', {
    title: 'Cross-validation curves for polynomial degree',
    blurb: 'Fit polynomials of degree 0–12 to noisy data. Right: training error, $K$-fold CV (thick, with ±1 SE bars and thin per-fold curves), and the true test error of each fitted model. The filled dot is the CV minimum, the ring the one-standard-error choice. Reshuffle the folds to see how much the CV curve — and the chosen degree — depends on an arbitrary partition. LOOCV uses the hat-matrix shortcut; the panel checks it against brute-force refitting.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 20, max: 200, step: 5, value: 40, onChange: newData });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.05, max: 1.2, step: 0.05, value: 0.4, onChange: newData });
      const K = UI.segmented(controls, { label: 'Folds $K$', options: ['2', '5', '10', 'LOO'], value: '10', onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', newData, { primary: true });
      UI.button(row, 'Reshuffle folds', () => { foldSeed++; draw(); });
      stage.classList.add('two');
      const pF = new Plot(stage, { width: 420, height: 270, x: [-1, 1], xlabel: 'x', ylabel: 'y', title: 'Data and selected fits' });
      const pE = new Plot(stage, { width: 420, height: 270, x: [-0.5, 12.5], xlabel: 'polynomial degree', ylabel: 'mean squared error', title: 'Error curves' });
      const sMin = UI.stat(stats, 'CV-minimizing degree', { accent: 'accent' });
      const s1se = UI.stat(stats, 'One-SE-rule degree');
      const sBest = UI.stat(stats, 'Truly best degree (this data set)');
      const sChk = UI.stat(stats, 'LOO shortcut vs. brute force');
      let x = [], y = [], foldSeed = 1;
      const grid = S.linspace(-1, 1, 300);
      function newData() {
        const r = ctx.rng, nn = Math.round(n.get()), s = sig.get();
        x = r.array(nn, () => r.uniform(-1, 1)); y = x.map(v => cvTruth(v) + s * r.normal());
        draw();
      }
      function draw() {
        const nn = x.length, s = sig.get(), kv = K.get(), KK = kv === 'LOO' ? nn : +kv;
        const ntr = nn - Math.ceil(nn / KK), Dmax = Math.max(0, Math.min(12, ntr - 2));
        const all = S.range(nn);
        const Xd = x.map(v => legendreRow(v, 12)), Gd = grid.map(v => legendreRow(v, 12));
        const cut = (rows, d) => rows.map(r => r.slice(0, d + 1));
        // folds
        const fr = S.rng(foldSeed * 7919 + nn), perm = fr.shuffle(S.range(nn)), fold = new Array(nn);
        perm.forEach((i, k) => { fold[i] = k % KK; });
        const train = [], cv = [], se = [], test = [], foldErr = [];
        let chk = null;
        for (let d = 0; d <= Dmax; d++) {
          const X = cut(Xd, d), G = cut(Gd, d);
          const full = polyFit(X, y, all);
          if (!full) { train.push(NaN); cv.push(NaN); se.push(NaN); test.push(NaN); foldErr.push([]); continue; }
          const yh = X.map(r => dotv(r, full.beta));
          train.push(S.mean(yh.map((v, i) => (y[i] - v) ** 2)));
          test.push(s * s + S.mean(G.map((r, j) => (dotv(r, full.beta) - cvTruth(grid[j])) ** 2)));
          if (KK === nn) {
            const Ai = S.inv(full.A);
            const loo = X.map((r, i) => { const h = dotv(r, S.matvec(Ai, r)); return ((y[i] - yh[i]) / (1 - h)) ** 2; });
            cv.push(S.mean(loo)); se.push(S.sd(loo) / Math.sqrt(nn)); foldErr.push([]);
          } else {
            const errs = [];
            let tot = 0;
            for (let k = 0; k < KK; k++) {
              const tr = all.filter(i => fold[i] !== k), te = all.filter(i => fold[i] === k);
              const f = polyFit(X, y, tr);
              let e = 0;
              for (const i of te) { const v = f ? dotv(X[i], f.beta) : NaN; e += (y[i] - v) ** 2; }
              tot += e; errs.push(e / te.length);
            }
            cv.push(tot / nn); se.push(S.sd(errs) / Math.sqrt(KK)); foldErr.push(errs);
          }
        }
        const valid = cv.map((v, d) => [d, v]).filter(q => Number.isFinite(q[1]));
        const dMin = valid.reduce((a, q) => (q[1] < a[1] ? q : a))[0];
        const thr = cv[dMin] + se[dMin];
        const d1 = valid.find(q => q[1] <= thr)[0];
        const tv = test.map((v, d) => [d, v]).filter(q => Number.isFinite(q[1]));
        const dBest = tv.reduce((a, q) => (q[1] < a[1] ? q : a))[0];
        // brute-force LOO check at the CV-chosen degree
        if (KK === nn) {
          const X = cut(Xd, dMin), full = polyFit(X, y, all), Ai = S.inv(full.A);
          let mx = 0;
          for (let i = 0; i < nn; i++) {
            const f = polyFit(X, y, all.filter(j => j !== i));
            if (!f) continue;
            const brute = y[i] - dotv(X[i], f.beta);
            const h = dotv(X[i], S.matvec(Ai, X[i])), sc = (y[i] - dotv(X[i], full.beta)) / (1 - h);
            mx = Math.max(mx, Math.abs(brute - sc));
          }
          chk = mx;
        }
        // left panel
        const [y0, y1] = S.extent(y.concat(grid.map(cvTruth)));
        pF.setDomain([-1, 1], [y0 - 0.3, y1 + 0.3]).clear();
        pF.points(x.map((v, i) => [v, y[i]]), { color: 'muted', r: 3, opacity: 0.8 });
        pF.line(grid.map(g => [g, cvTruth(g)]), { color: 'ink', width: 1.5, dash: true });
        const fitLine = (d, col, dash) => { const f = polyFit(cut(Xd, d), y, all); if (f) pF.line(cut(Gd, d).map((r, j) => [grid[j], dotv(r, f.beta)]), { color: col, width: 2.2, dash }); };
        fitLine(dMin, 's5', false);
        if (d1 !== dMin) fitLine(d1, 's2', true);
        pF.legend([{ label: 'truth', color: 'ink', dash: true }, { label: `CV min (deg ${dMin})`, color: 's5' }, { label: `1-SE (deg ${d1})`, color: 's2', dash: true }], { left: true });
        // right panel
        const finite = cv.concat(test, train).filter(Number.isFinite);
        const ymax = Math.min(Math.max(...finite), Math.max(4 * Math.min(...test.filter(Number.isFinite)), 2 * cv[0] || 1));
        pE.setDomain([-0.5, 12.5], [0, ymax * 1.05]).clear();
        for (let k = 0; k < (foldErr[0] || []).length; k++) pE.line(foldErr.map((e, d) => [d, e[k]]), { color: 's5', width: 0.8, opacity: 0.25 });
        pE.line(train.map((v, d) => [d, v]), { color: 's1', width: 2 });
        pE.line(test.map((v, d) => [d, v]), { color: 'ink', width: 2, dash: true });
        pE.line(cv.map((v, d) => [d, v]), { color: 's5', width: 3 });
        cv.forEach((v, d) => { if (Number.isFinite(v)) pE.segment(d, v - se[d], d, v + se[d], { color: 's5', width: 1.2 }); });
        pE.hline(thr, { color: 's2', width: 1, dash: '3 3' });
        pE.dot(dMin, cv[dMin], { color: 's5', r: 6 });
        pE.dot(d1, cv[d1], { color: 'surface', stroke: 's2', r: 6, strokeWidth: 2.5 });
        pE.hline(s * s, { color: 'faint', width: 1 });
        pE.legend([{ label: 'training error', color: 's1' }, { label: `CV (${kv === 'LOO' ? 'LOO' : KK + '-fold'})`, color: 's5' }, { label: 'true test error', color: 'ink', dash: true }, { label: 'σ² (irreducible)', color: 'faint', dash: true }]);
        sMin.set(String(dMin)); s1se.set(String(d1)); sBest.set(String(dBest), `test MSE ${S.fmt(test[dBest], 3)} vs. ${S.fmt(test[dMin], 3)} at CV choice`);
        sChk.set(chk == null ? 'select LOO' : `max diff ${chk.toExponential(1)}`, chk == null ? '' : `at degree ${dMin}`);
      }
      newData();
    },
  });

  Atlas.lab('cross-validation-leakage', {
    title: 'Leakage: feature selection outside the folds',
    blurb: 'Labels are random coin flips and all $p$ features are noise (unless you add signal). The <em>wrong</em> pipeline selects the $k$ features most associated with the label using all $n$ samples, then runs 5-fold CV of a nearest-centroid classifier. The <em>right</em> pipeline repeats the selection inside each training fold. Each experiment adds one point to each histogram. Pure noise should give 50% accuracy.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const p = UI.slider(controls, { label: 'Number of features $p$', min: 50, max: 2000, step: 50, value: 1000, onChange: reset });
      const k = UI.slider(controls, { label: 'Features selected $k$', min: 1, max: 50, step: 1, value: 20, onChange: reset });
      const nS = UI.slider(controls, { label: 'Samples $n$', min: 20, max: 100, step: 10, value: 40, onChange: reset });
      const sigl = UI.slider(controls, { label: 'True signal (shift in 5 features)', min: 0, max: 1.5, step: 0.1, value: 0, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const pl = new Plot(stage, { width: 640, height: 300, x: [0, 1], y: [0, 10], xlabel: 'cross-validated accuracy', ylabel: 'density', title: 'CV accuracy over repeated experiments' });
      const sN = UI.stat(stats, 'Experiments');
      const sW = UI.stat(stats, 'Mean CV accuracy: selection outside folds', { accent: 'accent' });
      const sR = UI.stat(stats, 'Mean CV accuracy: selection inside folds');
      const sT = UI.stat(stats, 'True accuracy of the right pipeline');
      let accW, accR, accT;
      function reset() { accW = []; accR = []; accT = []; loop.stop(); draw(); }
      function topK(Xc, lab, idx, kk) { // features with the largest |mean difference| over rows idx
        const P = Xc.length, sc = new Float64Array(P);
        let n1 = 0; for (const i of idx) if (lab[i]) n1++;
        const n0 = idx.length - n1;
        for (let j = 0; j < P; j++) { let a = 0, b = 0; const col = Xc[j]; for (const i of idx) { if (lab[i]) a += col[i]; else b += col[i]; } sc[j] = Math.abs(a / Math.max(n1, 1) - b / Math.max(n0, 1)); }
        return S.range(P).sort((u, v) => sc[v] - sc[u]).slice(0, kk);
      }
      function centroidAcc(Xc, lab, feats, tr, te) {
        const c1 = feats.map(() => 0), c0 = feats.map(() => 0);
        let n1 = 0, n0 = 0;
        for (const i of tr) { if (lab[i]) n1++; else n0++; feats.forEach((j, q) => { if (lab[i]) c1[q] += Xc[j][i]; else c0[q] += Xc[j][i]; }); }
        for (let q = 0; q < feats.length; q++) { c1[q] /= Math.max(n1, 1); c0[q] /= Math.max(n0, 1); }
        let ok = 0;
        for (const i of te) {
          let d1 = 0, d0 = 0;
          feats.forEach((j, q) => { d1 += (Xc[j][i] - c1[q]) ** 2; d0 += (Xc[j][i] - c0[q]) ** 2; });
          if ((d1 < d0 ? 1 : 0) === lab[i]) ok++;
        }
        return ok;
      }
      function experiment() {
        const r = ctx.rng, n = Math.round(nS.get()), P = Math.round(p.get()), kk = Math.min(Math.round(k.get()), P), s = sigl.get();
        const lab = r.shuffle(S.range(n).map(i => (i < n / 2 ? 1 : 0)));
        const Xc = Array.from({ length: P }, (_, j) => Float64Array.from({ length: n }, (_, i) => r.normal() + (j < 5 && lab[i] ? s : 0)));
        const all = S.range(n), fold = r.shuffle(S.range(n).map(i => i % 5));
        const leakFeats = topK(Xc, lab, all, kk);
        let okW = 0, okR = 0;
        for (let f = 0; f < 5; f++) {
          const tr = all.filter(i => fold[i] !== f), te = all.filter(i => fold[i] === f);
          okW += centroidAcc(Xc, lab, leakFeats, tr, te);
          okR += centroidAcc(Xc, lab, topK(Xc, lab, tr, kk), tr, te);
        }
        accW.push(okW / n); accR.push(okR / n);
        // true accuracy of the right pipeline trained on all n: score on fresh data (exactly 0.5 when there is no signal)
        if (s > 0) {
          const feats = topK(Xc, lab, all, kk), nt = 200;
          const labT = S.range(nt).map(i => (i % 2)), XT = Array.from({ length: P }, () => null);
          for (const j of feats) XT[j] = Float64Array.from({ length: nt }, (_, i) => r.normal() + (j < 5 && labT[i] ? s : 0));
          const XX = Xc.map((col, j) => (XT[j] ? Float64Array.from([...col, ...XT[j]]) : col));
          const labAll = lab.concat(labT);
          accT.push(centroidAcc(XX, labAll, feats, all, S.range(nt).map(i => n + i)) / nt);
        } else accT.push(0.5);
      }
      function step() {
        const t0 = Date.now();
        do experiment(); while (Date.now() - t0 < 14 && accW.length < 1500);
        draw();
        if (accW.length >= 1500) return false;
      }
      function draw() {
        const hW = S.histogram(accW, { bins: 40, lo: 0, hi: 1 }), hR = S.histogram(accR, { bins: 40, lo: 0, hi: 1 });
        const ymax = Math.max(5, ...hW.map(b => b.density), ...hR.map(b => b.density)) * 1.12;
        pl.setDomain([0, 1], [0, ymax]).clear();
        pl.vline(0.5, { color: 'ink', width: 1.5 });
        if (accW.length) { pl.bars(hR, { color: 's3', opacity: 0.55 }); pl.bars(hW, { color: 'bad', opacity: 0.55 }); }
        pl.legend([{ label: 'selection outside folds (wrong)', color: 'bad', fill: true }, { label: 'selection inside folds (right)', color: 's3', fill: true }], { left: true });
        sN.set(String(accW.length));
        sW.set(accW.length ? (100 * S.mean(accW)).toFixed(1) + '%' : '—');
        sR.set(accR.length ? (100 * S.mean(accR)).toFixed(1) + '%' : '—');
        sT.set(accT.length ? (100 * S.mean(accT)).toFixed(1) + '%' : '—', sigl.get() > 0 ? 'fresh test data, 200 points' : 'no signal: exactly 50%');
      }
      reset();
    },
  });
})();
