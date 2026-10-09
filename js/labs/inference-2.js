/* Labs: Statistical inference, part 2 (Bayes, testing, intervals, multiplicity, decisions, causality). */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const Z975 = 1.959963984540054;
  const pct = (x, d = 1) => (Number.isFinite(x) ? (100 * x).toFixed(d) + '%' : '—');
  const ivl = (I, d = 3) => `[${S.fmt(I[0], d)}, ${S.fmt(I[1], d)}]`;
  const cov = ok => (ok ? '<span style="color:var(--good)">covers truth</span>' : '<span style="color:var(--bad)">misses truth</span>');

  // Beta quantile with safe handling of the boundaries.
  const qbeta = (p, a, b) => (p <= 0 ? 0 : p >= 1 ? 1 : S.q.beta(p, a, b));
  // Shortest (HPD) interval of a unimodal Beta: minimise q(t+1-alpha) - q(t) over t in [0, alpha].
  function betaHPD(a, b, alpha) {
    if (a <= 1 && b > 1) return [0, qbeta(1 - alpha, a, b)];
    if (b <= 1 && a > 1) return [qbeta(alpha, a, b), 1];
    if (a <= 1 && b <= 1) return [qbeta(alpha / 2, a, b), qbeta(1 - alpha / 2, a, b)]; // U-shaped: HPD is not an interval
    let lo = 0, hi = alpha;
    const L = t => qbeta(t + 1 - alpha, a, b) - qbeta(t, a, b);
    for (let i = 0; i < 40; i++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (L(m1) < L(m2)) hi = m2; else lo = m1; }
    const t = (lo + hi) / 2;
    return [qbeta(t, a, b), qbeta(t + 1 - alpha, a, b)];
  }
  const waldBin = (k, n, z = Z975) => { const p = k / n, h = z * Math.sqrt(p * (1 - p) / n); return [Math.max(0, p - h), Math.min(1, p + h)]; };

  /* ------------------------------------------------------------------ */
  Atlas.lab('bayesian-inference-beta-binomial', {
    title: 'Beta–Binomial updating, one coin flip at a time',
    blurb: 'Flip a coin with true success probability $p$ and watch the Beta prior (dashed) become the posterior (filled). The faint curve is the normalised likelihood. Make the prior strong and wrong (e.g. $a=b=30$ with $p=0.85$) and count how many flips it takes the data to overrule it; compare the credible interval with the Wald confidence interval at small $n$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const a = UI.slider(controls, { label: 'Prior $a$', min: 0.5, max: 30, step: 0.5, value: 2, onInput: draw });
      const b = UI.slider(controls, { label: 'Prior $b$', min: 0.5, max: 30, step: 0.5, value: 2, onInput: draw });
      const pTrue = UI.slider(controls, { label: 'True $p$', min: 0.01, max: 0.99, step: 0.01, value: 0.3, onChange: reset });
      const kind = UI.segmented(controls, { label: 'Credible interval', options: ['Equal-tailed', 'HPD'], value: 'Equal-tailed', onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, '+1 flip', () => flip(1), { primary: true });
      UI.button(row, '+10', () => flip(10));
      UI.button(row, '+100', () => flip(100));
      const row2 = UI.h('div', 'btn-row', controls);
      let acc = 0;
      const loop = ctx.loop(dt => { acc += dt; if (acc > 0.12) { acc = 0; flip(1); } if (n >= 1000) return false; });
      ctx.playButton(row2, loop, ['▶ Auto-flip', '❚❚ Pause']);
      UI.button(row2, 'Reset', reset);
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 1], y: [0, 5], xlabel: 'p', ylabel: 'density' });
      const sN = UI.stat(stats, 'Data', { accent: 'accent' });
      const sPm = UI.stat(stats, 'Posterior mean');
      const sW = UI.stat(stats, 'Weight on MLE $w$');
      const sCr = UI.stat(stats, '95% credible');
      const sWa = UI.stat(stats, '95% Wald CI');
      let n = 0, k = 0;
      function reset() { n = 0; k = 0; loop.stop(); draw(); }
      function flip(m) { for (let i = 0; i < m; i++) { n++; if (ctx.rng.u() < pTrue.get()) k++; } draw(); }
      function draw() {
        const A = a.get(), B = b.get(), pa = A + k, pb = B + n - k, pt = pTrue.get();
        const xs = S.linspace(0.0015, 0.9985, 400);
        const post = xs.map(x => S.pdf.beta(x, pa, pb));
        const prior = xs.map(x => S.pdf.beta(x, A, B));
        const lik = xs.map(x => S.pdf.beta(x, k + 1, n - k + 1));
        const finite = arr => arr.filter(Number.isFinite);
        const ymax = Math.min(60, Math.max(3, ...finite(post).slice(5, -5)) * 1.15);
        p.setDomain([0, 1], [0, ymax]).clear();
        const cl = arr => xs.map((x, i) => [x, Math.min(arr[i], ymax * 1.05)]);
        const CI = kind.get() === 'HPD' ? betaHPD(pa, pb, 0.05) : [qbeta(0.025, pa, pb), qbeta(0.975, pa, pb)];
        p.band(CI[0], CI[1], { color: 's1', opacity: 0.1 });
        p.area(cl(post), { color: 's1', opacity: 0.3 });
        p.line(cl(post), { color: 's1', width: 2.2 });
        p.line(cl(prior), { color: 's2', width: 1.8, dash: true });
        if (n > 0) p.line(cl(lik), { color: 'muted', width: 1.2, opacity: 0.7 });
        p.vline(pt, { color: 'ink', dash: false, width: 1.5 });
        if (n > 0) {
          const W = waldBin(k, n);
          p.segment(W[0], ymax * 0.06, W[1], ymax * 0.06, { color: 's3', width: 4 });
          p.text(W[1], ymax * 0.06, 'Wald', { dx: 6, dy: 4, size: 11, color: 's3' });
        }
        p.segment(CI[0], ymax * 0.12, CI[1], ymax * 0.12, { color: 's1', width: 4 });
        p.text(CI[1], ymax * 0.12, 'credible', { dx: 6, dy: 4, size: 11, color: 's1' });
        p.legend([{ label: 'prior', color: 's2', dash: true }, { label: 'likelihood (normalised)', color: 'muted' }, { label: 'posterior', color: 's1', fill: true }, { label: 'true p', color: 'ink' }]);
        sN.set(`${k} / ${n}`, 'successes / flips');
        sPm.set(S.fmt(pa / (pa + pb)), n ? `MLE ${S.fmt(k / n)} · prior mean ${S.fmt(A / (A + B))}` : `prior mean ${S.fmt(A / (A + B))}`);
        sW.set(S.fmt(n / (n + A + B)), `prior worth ${S.fmt(A + B, 1)} flips`);
        sCr.set(ivl(CI), cov(CI[0] <= pt && pt <= CI[1]));
        if (n > 0) { const W = waldBin(k, n); sWa.set(ivl(W), (W[1] - W[0] < 1e-12 ? 'degenerate! ' : '') + cov(W[0] <= pt && pt <= W[1])); }
        else sWa.set('—', 'no data yet');
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  Atlas.lab('bayesian-inference-coverage', {
    title: 'Frequentist coverage of a 95% credible interval',
    blurb: 'For each true $p$, the probability (over $X\\sim\\text{Bin}(n,p)$) that the equal-tailed 95% credible interval covers $p$ — computed exactly by summing over all $n+1$ outcomes. Bernstein–von Mises predicts 95% as $n$ grows, whatever the prior. Watch an informative prior break coverage where it is wrong, and compare with the Wald interval near 0 and 1.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 300, step: 1, value: 20, onChange: draw });
      const pre = UI.segmented(controls, { label: 'Prior', options: [{ value: 'j', label: 'Jeffreys ½,½' }, { value: 'u', label: 'Uniform 1,1' }, { value: 'c', label: 'Custom' }], value: 'j', onChange: draw });
      const a = UI.slider(controls, { label: 'Custom $a$', min: 0.5, max: 40, step: 0.5, value: 10, onChange: () => { pre.set('c'); draw(); } });
      const b = UI.slider(controls, { label: 'Custom $b$', min: 0.5, max: 40, step: 0.5, value: 10, onChange: () => { pre.set('c'); draw(); } });
      const showW = UI.toggle(controls, { label: 'Show Wald interval', value: true, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 1], y: [0, 1], xlabel: 'true p', ylabel: 'coverage' });
      const sAvg = UI.stat(stats, 'Average coverage (credible)', { accent: 'accent' });
      const sMin = UI.stat(stats, 'Minimum coverage (credible)');
      const sWa = UI.stat(stats, 'Average coverage (Wald)');
      function draw() {
        const N = Math.round(n.get());
        const [A, B] = pre.get() === 'j' ? [0.5, 0.5] : pre.get() === 'u' ? [1, 1] : [a.get(), b.get()];
        const cr = [], wa = [];
        for (let k = 0; k <= N; k++) { cr.push([qbeta(0.025, A + k, B + N - k), qbeta(0.975, A + k, B + N - k)]); wa.push(waldBin(k, N)); }
        const grid = S.linspace(0.0025, 0.9975, 200);
        const cC = [], cW = [];
        for (const pp of grid) {
          let sc = 0, sw = 0;
          for (let k = 0; k <= N; k++) {
            const w = S.pmf.binom(k, N, pp);
            if (cr[k][0] <= pp && pp <= cr[k][1]) sc += w;
            if (wa[k][0] <= pp && pp <= wa[k][1]) sw += w;
          }
          cC.push([pp, sc]); cW.push([pp, sw]);
        }
        const ylo = Math.max(0, Math.min(0.8, ...cC.map(d => d[1]), ...(showW.get() ? cW.map(d => d[1]) : [1])) - 0.03);
        p.setDomain([0, 1], [ylo, 1]).clear();
        p.hline(0.95, { color: 'ink', width: 1.2 });
        if (showW.get()) p.line(cW, { color: 's3', width: 1.4, opacity: 0.85 });
        p.line(cC, { color: 's1', width: 2 });
        p.text(0.99, 0.95, 'nominal 95%', { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        p.legend([{ label: `credible, Beta(${S.fmt(A, 1)}, ${S.fmt(B, 1)}) prior`, color: 's1' }].concat(showW.get() ? [{ label: 'Wald', color: 's3' }] : []), { left: true });
        sAvg.set(pct(S.mean(cC.map(d => d[1]))), 'averaged over p uniform');
        const im = S.argmin(cC.map(d => d[1]));
        sMin.set(pct(cC[im][1]), `at p = ${S.fmt(cC[im][0], 3)}`);
        sWa.set(pct(S.mean(cW.map(d => d[1]))), `min ${pct(Math.min(...cW.map(d => d[1])))}`);
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  Atlas.lab('hypothesis-testing-power', {
    title: 'Size, power and the standardised effect',
    blurb: 'Left: the null and alternative sampling distributions of the $z$ statistic; red area is the type I error $\\alpha$, green area is the power. Right: the whole power function, for $n/2$, $n$ and $2n$. Power depends on $n$, $\\sigma$ and the effect only through $\\delta=\\sqrt n\\,\\Delta/\\sigma$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 2, max: 400, step: 1, value: 25, onInput: draw });
      const eff = UI.slider(controls, { label: 'True effect $\\Delta/\\sigma$', min: -0.6, max: 1.2, step: 0.01, value: 0.4, onInput: draw });
      const al = UI.slider(controls, { label: 'Level $\\alpha$', min: 0.001, max: 0.2, step: 0.001, value: 0.05, onInput: draw });
      const side = UI.segmented(controls, { label: 'Alternative', options: [{ value: 'one', label: 'One-sided $\\mu>\\mu_0$' }, { value: 'two', label: 'Two-sided' }], value: 'two', onChange: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-4, 8], y: [0, 0.45], xlabel: 'z statistic', ylabel: 'density', title: 'Sampling distributions of Z' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-0.6, 1.2], y: [0, 1], xlabel: 'Δ/σ', ylabel: 'power β', title: 'Power function' });
      const sD = UI.stat(stats, 'Standardised effect $\\delta$');
      const sP = UI.stat(stats, 'Power at this effect', { accent: 'accent' });
      const sN80 = UI.stat(stats, '$n$ for 80% power');
      const power = (d, alpha, two) => (two ? 1 - S.cdf.normal(S.qnorm(1 - alpha / 2) - d) + S.cdf.normal(-S.qnorm(1 - alpha / 2) - d) : 1 - S.cdf.normal(S.qnorm(1 - alpha) - d));
      function draw() {
        const N = Math.round(n.get()), E = eff.get(), alpha = al.get(), two = side.get() === 'two';
        const d = Math.sqrt(N) * E;
        const lo = Math.min(-4, d - 4), hi = Math.max(4, d + 4);
        pL.setDomain([lo, hi], [0, 0.45]).clear();
        const xs = S.linspace(lo, hi, 400);
        const c = two ? S.qnorm(1 - alpha / 2) : S.qnorm(1 - alpha);
        const inR = x => (two ? Math.abs(x) > c : x > c);
        const seg = f => { const out = []; let cur = null; for (const x of xs) { if (inR(x)) { (cur = cur || []).push([x, f(x)]); } else if (cur) { out.push(cur); cur = null; } } if (cur) out.push(cur); return out; };
        for (const s of seg(x => S.pdf.normal(x, d))) pL.area(s, { color: 'good', opacity: 0.35 });
        for (const s of seg(x => S.pdf.normal(x))) pL.area(s, { color: 'bad', opacity: 0.55 });
        pL.line(xs.map(x => [x, S.pdf.normal(x)]), { color: 'ink', width: 1.8 });
        pL.line(xs.map(x => [x, S.pdf.normal(x, d)]), { color: 's1', width: 2 });
        pL.vline(c, { color: 'muted' }); if (two) pL.vline(-c, { color: 'muted' });
        pL.legend([{ label: 'under H₀', color: 'ink' }, { label: 'under H₁', color: 's1' }, { label: 'α', color: 'bad', fill: true }, { label: 'power', color: 'good', fill: true }]);
        pR.clear();
        const g = S.linspace(-0.6, 1.2, 241);
        [[Math.max(1, N / 2), 's2', 1.3, true], [2 * N, 's3', 1.3, true], [N, 's1', 2.4, false]].forEach(([m, col, w, dash]) =>
          pR.line(g.map(e => [e, power(Math.sqrt(m) * e, alpha, two)]), { color: col, width: w, dash }));
        pR.hline(alpha, { color: 'bad', width: 1 });
        pR.vline(E, { color: 'muted' });
        pR.dot(E, power(d, alpha, two), { color: 's1', r: 5 });
        pR.legend([{ label: 'n/2', color: 's2', dash: true }, { label: 'n', color: 's1' }, { label: '2n', color: 's3', dash: true }], { left: true });
        const pw = power(d, alpha, two);
        sD.set(S.fmt(d, 2), `√${N} × ${S.fmt(E, 2)}`);
        sP.set(pct(pw), E === 0 ? 'equals α at the null' : (two || E > 0 ? '' : 'below α: wrong direction'));
        const za = two ? S.qnorm(1 - alpha / 2) : S.qnorm(1 - alpha);
        sN80.set(Math.abs(E) < 1e-9 ? '∞' : String(Math.ceil(((za + S.qnorm(0.8)) / Math.abs(E)) ** 2)), two ? 'ignoring the far tail' : (E < 0 ? 'impossible one-sided' : ''));
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  Atlas.lab('hypothesis-testing-pvalues', {
    title: 'The distribution of p-values',
    blurb: 'Run thousands of experiments and histogram their p-values. Under $H_0$ (effect 0) a continuous test gives a flat histogram; the exact binomial test is conservative (super-uniform); the $t$ test on skewed data at small $n$ is not uniform at all. Raise the effect and the mass piles up at 0 — the fraction below $\\alpha$ is the power.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const test = UI.select(controls, { label: 'Experiment', options: [
        { value: 't', label: 't test, normal data' },
        { value: 'b', label: 'Exact binomial test (discrete)' },
        { value: 'ln', label: 't test, lognormal data (skewed)' },
      ], value: 't', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 4, max: 100, step: 1, value: 15, onChange: reset });
      const eff = UI.slider(controls, { label: 'Effect (in SDs)', min: 0, max: 1.2, step: 0.01, value: 0, onChange: reset });
      const al = UI.slider(controls, { label: 'Level $\\alpha$', min: 0.01, max: 0.2, step: 0.01, value: 0.05, onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run experiments', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 1], y: [0, 3], xlabel: 'p-value', ylabel: 'density' });
      const sN = UI.stat(stats, 'Experiments');
      const sR = UI.stat(stats, 'Fraction $p\\le\\alpha$', { accent: 'accent' });
      const sQ = UI.stat(stats, 'Median p-value');
      let ps = [], binCdf = null;
      const LNM = Math.exp(0.5), LNSD = Math.sqrt((Math.E - 1) * Math.E);
      function reset() {
        ps = []; loop.stop();
        const N = Math.round(n.get());
        binCdf = []; let s = 0; for (let k = 0; k <= N; k++) { s += S.pmf.binom(k, N, 0.5); binCdf.push(s); }
        for (let i = 0; i < 300; i++) ps.push(experiment());
        draw();
      }
      function experiment() {
        const N = Math.round(n.get()), E = eff.get(), r = ctx.rng, kind = test.get();
        if (kind === 'b') {
          const pp = Math.min(0.99, 0.5 + 0.5 * E);
          let k = 0; for (let i = 0; i < N; i++) if (r.u() < pp) k++;
          const lower = binCdf[k], upper = 1 - (k > 0 ? binCdf[k - 1] : 0);
          return Math.min(1, 2 * Math.min(lower, upper));
        }
        let sx = 0, sxx = 0;
        for (let i = 0; i < N; i++) {
          const x = kind === 't' ? r.normal() + E : (r.lognormal(0, 1) - LNM) / LNSD + E;
          sx += x; sxx += x * x;
        }
        const m = sx / N, v = (sxx - N * m * m) / (N - 1), t = m / Math.sqrt(v / N);
        return 2 * (1 - S.cdf.t(Math.abs(t), N - 1));
      }
      function step() {
        for (let i = 0; i < 60; i++) ps.push(experiment());
        draw();
        if (ps.length >= 20000) return false;
      }
      function draw() {
        const h = S.histogram(ps, { bins: 20, lo: 0, hi: 1 });
        const ymax = Math.max(2, ...h.map(b => b.density)) * 1.1;
        p.setDomain([0, 1], [0, ymax]).clear();
        const alpha = al.get();
        p.band(0, alpha, { color: 'bad', opacity: 0.08 });
        p.bars(h, { color: 's1', opacity: 0.6 });
        p.hline(1, { color: 'ink', width: 1.4 });
        p.vline(alpha, { color: 'bad' });
        p.text(0.99, 1, 'Uniform(0,1)', { anchor: 'end', dy: -6, size: 11, color: 'muted' });
        const frac = ps.filter(x => x <= alpha + 1e-12).length / ps.length;
        sN.set(String(ps.length));
        sR.set(pct(frac), eff.get() === 0 ? `empirical size; nominal ${pct(alpha, 0)}` : 'empirical power');
        sQ.set(S.fmt(S.median(ps)));
      }
      reset();
    },
  });

  /* ------------------------------------------------------------------ */
  // Two simple hypotheses on one observation. Each scenario: f0/F0 is N(0,1); f1/F1 depends on the parameter.
  const NP = {
    shift: { label: 'Mean shift: N(0,1) vs N(θ,1)', par: { min: -3, max: 3, value: 1.5, label: 'Alternative mean θ' },
      f1: (x, t) => S.pdf.normal(x, t), F1: (x, t) => S.cdf.normal(x, t) },
    scale: { label: 'Scale: N(0,1) vs N(0,θ²)', par: { min: 0.3, max: 4, value: 2, label: 'Alternative SD θ' },
      f1: (x, t) => S.pdf.normal(x, 0, t), F1: (x, t) => S.cdf.normal(x, 0, t) },
    cauchy: { label: 'Heavy tail: N(0,1) vs Cauchy(θ,1)', par: { min: -3, max: 4, value: 2, label: 'Cauchy location θ' },
      f1: (x, t) => S.pdf.cauchy(x, t), F1: (x, t) => S.cdf.cauchy(x, t) },
    mix: { label: 'Mixture: N(0,1) vs ½N(−θ,1)+½N(θ,1)', par: { min: 0, max: 4, value: 2, label: 'Component mean θ' },
      f1: (x, t) => 0.5 * (S.pdf.normal(x, -t) + S.pdf.normal(x, t)), F1: (x, t) => 0.5 * (S.cdf.normal(x, -t) + S.cdf.normal(x, t)) },
  };
  Atlas.lab('neyman-pearson-region', {
    title: 'The most powerful rejection region',
    blurb: 'One observation $x$; $H_0: \\N(0,1)$ against a simple alternative. The Neyman–Pearson region (shaded) collects the $x$ with the largest likelihood ratio $f_1/f_0$ until its null probability reaches $\\alpha$. Right: its ROC curve versus the naive test "reject when $x$ is large". They agree only when the likelihood ratio is monotone in $x$ — try the scale, Cauchy and mixture alternatives.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sc = UI.select(controls, { label: 'Alternative $f_1$', options: Object.keys(NP).map(k => ({ value: k, label: NP[k].label })), value: 'shift', onChange: () => { const P = NP[sc.get()].par; th.input.min = P.min; th.input.max = P.max; th.set(P.value); draw(); } });
      const th = UI.slider(controls, { label: 'Alternative parameter $\\theta$', min: -3, max: 3, step: 0.05, value: 1.5, onInput: draw });
      const al = UI.slider(controls, { label: 'Size $\\alpha$', min: 0.005, max: 0.5, step: 0.005, value: 0.05, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-6, 6], y: [0, 0.5], xlabel: 'x', ylabel: 'density', title: 'Densities and the MP region' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 1], xlabel: 'size (type I error)', ylabel: 'power', title: 'ROC: LR test vs naive test' });
      const sK = UI.stat(stats, 'Threshold $k$ on $f_1/f_0$');
      const sP = UI.stat(stats, 'Power of MP test', { accent: 'accent' });
      const sNv = UI.stat(stats, 'Power of naive test');
      const L = 12, M = 2400;
      const edges = S.linspace(-L, L, M + 1);
      function draw() {
        const P = NP[sc.get()], t = Math.min(P.par.max, Math.max(P.par.min, th.get())), alpha = al.get();
        // Cells with exact masses (including the two infinite tails).
        // Interior cells: masses from densities (dx = 0.01); tails beyond ±L: exact f1 mass, f0 mass is negligible.
        const dx = (2 * L) / M, cells = [];
        const lt = P.F1(-L, t), rt = 1 - P.F1(L, t);
        cells.push({ a: -Infinity, b: -L, m0: 0, m1: lt, r: lt > 0 ? Infinity : 0 });
        for (let i = 0; i < M; i++) {
          const x = edges[i] + dx / 2, m0 = S.pdf.normal(x) * dx, m1 = P.f1(x, t) * dx;
          cells.push({ a: edges[i], b: edges[i + 1], m0, m1, r: m0 > 0 ? m1 / m0 : (m1 > 0 ? Infinity : 0) });
        }
        cells.push({ a: L, b: Infinity, m0: 0, m1: rt, r: rt > 0 ? Infinity : 0 });
        const order = cells.slice().sort((u, v) => v.r - u.r);
        let s0 = 0, s1 = 0, k = NaN, power = NaN;
        for (const c of order) {
          if (!Number.isNaN(power)) break;
          if (s0 + c.m0 <= alpha) { c.sel = 1; s0 += c.m0; s1 += c.m1; }
          else { const g = (alpha - s0) / c.m0; c.sel = g; power = s1 + g * c.m1; k = c.r; }
        }
        if (Number.isNaN(power)) power = s1;
        const roc = [[0, 0]];
        { let a0 = 0, a1 = 0, i = 0; for (const c of order) { a0 += c.m0; a1 += c.m1; if (++i % 6 === 0) roc.push([Math.min(1, a0), Math.min(1, a1)]); } roc.push([1, 1]); }
        // Naive test: reject for x > c.
        const naive = S.linspace(-L, L, 400).map(c => [1 - S.cdf.normal(c), 1 - P.F1(c, t)]).reverse();
        const cN = S.qnorm(1 - alpha), pN = 1 - P.F1(cN, t);
        // Left panel.
        const xr = sc.get() === 'scale' ? Math.max(4, 3.2 * t) : 6;
        const xs = S.linspace(-xr, xr, 500);
        const ymax = Math.max(0.42, ...xs.map(x => P.f1(x, t))) * 1.1;
        pL.setDomain([-xr, xr], [0, ymax]).clear();
        let run = null;
        const flush = end => { if (run) { const u = Math.max(-xr, run.a), v = Math.min(xr, end); if (v > u) pL.band(u, v, { color: 'accent', opacity: 0.18 * run.w + 0.04 }); run = null; } };
        for (const c of cells) {
          if (c.sel) { if (run && Math.abs(run.w - c.sel) < 1e-9) continue; flush(c.a); run = { a: c.a, w: c.sel }; }
          else flush(c.a);
        }
        flush(L);
        pL.line(xs.map(x => [x, S.pdf.normal(x)]), { color: 'ink', width: 2 });
        pL.line(xs.map(x => [x, P.f1(x, t)]), { color: 's1', width: 2 });
        pL.vline(cN, { color: 's4', width: 1.2 });
        pL.legend([{ label: 'f₀ = N(0,1)', color: 'ink' }, { label: 'f₁', color: 's1' }, { label: 'MP region', color: 'accent', fill: true, opacity: 0.25 }, { label: 'naive cut-off', color: 's4', dash: true }]);
        // Right panel.
        pR.clear();
        pR.line([[0, 0], [1, 1]], { color: 'faint', width: 1, dash: true });
        pR.line(naive, { color: 's4', width: 1.6 });
        pR.line(roc, { color: 's1', width: 2.2 });
        pR.dot(alpha, power, { color: 's1', r: 5 });
        pR.dot(alpha, pN, { color: 's4', r: 4 });
        pR.legend([{ label: 'likelihood-ratio test', color: 's1' }, { label: 'reject for large x', color: 's4' }], { left: false });
        sK.set(S.fmt(k, 3), `log k = ${S.fmt(Math.log(k), 2)}`);
        sP.set(pct(power, 1));
        sNv.set(pct(pN, 1), pN < power - 0.005 ? 'strictly worse' : 'same — LR is monotone in x');
      }
      draw();
    },
  });
  /* ------------------------------------------------------------------ */
  // Binomial / Poisson log-likelihood machinery for the trinity.
  const xlogy = (x, y) => (x === 0 ? 0 : x * Math.log(y));
  const LIK = {
    binom: {
      ll: (t, n, s) => xlogy(s, t) + xlogy(n - s, 1 - t),
      mle: (n, s) => s / n,
      info: (t, n) => n / (t * (1 - t)),          // total Fisher information
      score: (t, n, s) => s / t - (n - s) / (1 - t),
      range: () => [0, 1],
      lo: 0, hi: 1,
    },
    pois: {
      ll: (t, n, s) => xlogy(s, t) - n * t,
      mle: (n, s) => s / n,
      info: (t, n) => n / t,
      score: (t, n, s) => s / t - n,
      range: (n, s, t0) => [0, Math.max(1, 2.2 * s / n, 1.5 * t0)],
      lo: 0, hi: Infinity,
    },
  };
  // LR interval {t : 2[ll(mle) - ll(t)] <= c} by bisection on each side of the MLE.
  function lrInterval(M, n, s, c, hiCap) {
    const th = M.mle(n, s), top = M.ll(th, n, s), g = t => 2 * (top - M.ll(t, n, s)) - c;
    const solve = (a, b) => { for (let i = 0; i < 80; i++) { const m = (a + b) / 2; if ((g(a) > 0) === (g(m) > 0)) a = m; else b = m; } return (a + b) / 2; };
    const eps = 1e-12;
    const L = th <= eps ? 0 : (g(eps) <= 0 ? 0 : solve(eps, th));
    let hi = Math.min(hiCap, th <= 0 ? 1 : 2 * th + 10);
    if (M === LIK.binom) hi = 1 - eps;
    else while (g(hi) < 0) hi *= 2;
    const U = th >= 1 - eps && M === LIK.binom ? 1 : (g(hi) <= 0 ? hi : solve(th, hi));
    return [L, U];
  }
  const wilson = (s, n, z = Z975) => {
    const p = s / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z / d * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
    return [Math.max(0, c - h), Math.min(1, c + h)];
  };

  Atlas.lab('likelihood-ratio-tests-geometry', {
    title: 'Wald, score and likelihood ratio on one log-likelihood',
    blurb: 'The curve is $\\ell(\\theta)-\\ell(\\hat\\theta)$. Half the LR statistic is the drop of the curve at $\\theta_0$ (purple); half the Wald statistic is the drop of the quadratic fitted at $\\hat\\theta$ (dashed); half the score statistic is the rise of the quadratic fitted at $\\theta_0$ (dotted). With few counts and $\\hat\\theta$ near the boundary the three disagree; with many they coincide.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mod = UI.segmented(controls, { label: 'Model', options: [{ value: 'binom', label: 'Binomial' }, { value: 'pois', label: 'Poisson' }], value: 'binom', onChange: () => { est.set(est.get()); t0.set(t0.get()); draw(); } });
      const sc = v => (mod.get() === 'pois' ? 4 * v : v).toFixed(2);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 400, step: 1, value: 20, onInput: draw });
      const est = UI.slider(controls, { label: 'Observed $\\hat\\theta$ (rounded to data)', min: 0, max: 1, step: 0.01, value: 0.15, fmt: sc, onInput: draw });
      const t0 = UI.slider(controls, { label: 'Null value $\\theta_0$', min: 0.02, max: 0.98, step: 0.01, value: 0.4, fmt: sc, onInput: draw });
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 1], y: [-5, 0.5], xlabel: 'θ', ylabel: 'ℓ(θ) − ℓ(θ̂)' });
      const sL = UI.stat(stats, 'LR $\\Lambda$', { accent: 'accent' });
      const sW = UI.stat(stats, 'Wald $W$');
      const sS = UI.stat(stats, 'Score $S$');
      const sI = UI.note(stats, '');
      const chiP = x => (Number.isFinite(x) ? 1 - S.cdf.chisq(Math.max(x, 0), 1) : 0);
      function draw() {
        const M = LIK[mod.get()], N = Math.round(n.get());
        // Binomial: θ̂ ∈ [0,1] maps to k = round(θ̂ n). Poisson: θ̂ ∈ [0,1] maps to mean in [0, 4].
        const isB = mod.get() === 'binom';
        const s = isB ? Math.round(est.get() * N) : Math.round(est.get() * 4 * N);
        const th = M.mle(N, s), th0 = isB ? t0.get() : t0.get() * 4;
        const [x0, x1] = isB ? [0, 1] : M.range(N, s, th0);
        const top = M.ll(th, N, s);
        p.setDomain([x0, x1], [-5, 0.6]).clear();
        const xs = S.linspace(x0 + 1e-4, x1 - (isB ? 1e-4 : 0), 500);
        p.hline(-S.q.chisq(0.95, 1) / 2, { color: 'faint', width: 1 });
        p.text(x1, -S.q.chisq(0.95, 1) / 2, '−χ²₁,₀.₉₅ / 2', { anchor: 'end', dy: -5, size: 10, color: 'muted' });
        p.line(xs.map(t => [t, Math.max(-6, M.ll(t, N, s) - top)]), { color: 's1', width: 2.4 });
        const Ih = th > 0 && (!isB || th < 1) ? M.info(th, N) : Infinity;
        if (Number.isFinite(Ih)) p.line(xs.map(t => [t, Math.max(-6, -0.5 * Ih * (t - th) ** 2)]), { color: 's2', width: 1.6, dash: true });
        const I0 = M.info(th0, N), U0 = M.score(th0, N, s), l0 = M.ll(th0, N, s) - top;
        p.line(xs.map(t => [t, Math.max(-6, l0 + U0 * (t - th0) - 0.5 * I0 * (t - th0) ** 2)]), { color: 's3', width: 1.6, dash: '2 3' });
        const LR = -2 * l0, W = Number.isFinite(Ih) ? Ih * (th - th0) ** 2 : Infinity, Sc = U0 * U0 / I0;
        // Geometric readouts at θ0 / at the score-quadratic's peak.
        p.segment(th0, 0, th0, l0, { color: 's5', width: 3 });
        if (Number.isFinite(W)) p.segment(th0 + (x1 - x0) * 0.006, 0, th0 + (x1 - x0) * 0.006, -W / 2, { color: 's2', width: 2 });
        const tpk = th0 + U0 / I0;
        p.segment(tpk, l0, tpk, l0 + Sc / 2, { color: 's3', width: 2 });
        p.dot(th, 0, { color: 's1', r: 4 });
        p.dot(th0, l0, { color: 's5', r: 4 });
        // Intervals.
        const c = S.q.chisq(0.95, 1), z = Math.sqrt(c);
        const iLR = lrInterval(M, N, s, c, x1 * 4 + 10);
        const iW = Number.isFinite(Ih) ? [Math.max(0, th - z / Math.sqrt(Ih)), isB ? Math.min(1, th + z / Math.sqrt(Ih)) : th + z / Math.sqrt(Ih)] : [th, th];
        const iS = isB ? wilson(s, N, z) : (() => { const a = th + z * z / (2 * N), h = z * Math.sqrt(th / N + z * z / (4 * N * N)); return [Math.max(0, a - h), a + h]; })();
        [[iLR, 's5', -4.2], [iW, 's2', -4.5], [iS, 's3', -4.8]].forEach(([I, col, y]) => p.segment(I[0], y, Math.min(I[1], x1), y, { color: col, width: 4 }));
        p.legend([{ label: 'log-likelihood', color: 's1' }, { label: 'Wald quadratic at θ̂', color: 's2', dash: true }, { label: 'score quadratic at θ₀', color: 's3', dash: true }, { label: 'LR drop / LR interval', color: 's5' }], { left: !isB || th > 0.5 });
        sL.set(S.fmt(LR, 2), `p = ${S.fmt(chiP(LR), 4)}`);
        sW.set(S.fmt(W, 2), `p = ${S.fmt(chiP(W), 4)}`);
        sS.set(S.fmt(Sc, 2), `p = ${S.fmt(chiP(Sc), 4)}`);
        sI.set(`Data: ${isB ? `${s} successes in ${N}` : `total count ${s} over ${N} units`}, $\\hat\\theta = ${S.fmt(th, 3)}$. 95% intervals (bars at bottom) — LR ${ivl(iLR)}, Wald ${ivl(iW)}, score ${ivl(iS)}.`);
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  Atlas.lab('likelihood-ratio-tests-coverage', {
    title: 'Exact coverage of four binomial intervals',
    blurb: 'Exact coverage $C(p)=\\sum_x \\binom nx p^x(1-p)^{n-x}\\,\\ind\\{p\\in I(x)\\}$ of nominal-95% intervals obtained by inverting the Wald, score (Wilson) and LR tests, plus the exact Clopper–Pearson interval. Note the jagged oscillation (lattice effects), the collapse of Wald near 0 and 1, and the price Clopper–Pearson pays for never dropping below 95%.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 200, step: 1, value: 30, onChange: draw });
      const show = {};
      const METH = [['wald', 'Wald', 's2'], ['wilson', 'Score (Wilson)', 's3'], ['lr', 'Likelihood ratio', 's5'], ['cp', 'Clopper–Pearson', 's1']];
      for (const [k, lab] of METH) show[k] = UI.toggle(controls, { label: lab, value: true, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 1], y: [0.6, 1], xlabel: 'true p', ylabel: 'coverage' });
      const tiles = {};
      for (const [k, lab] of METH) tiles[k] = UI.stat(stats, lab);
      function draw() {
        const N = Math.round(n.get()), c = S.q.chisq(0.95, 1);
        const I = { wald: [], wilson: [], lr: [], cp: [] };
        for (let x = 0; x <= N; x++) {
          I.wald.push(waldBin(x, N));
          I.wilson.push(wilson(x, N));
          I.lr.push(lrInterval(LIK.binom, N, x, c, 1));
          I.cp.push([x === 0 ? 0 : qbeta(0.025, x, N - x + 1), x === N ? 1 : qbeta(0.975, x + 1, N - x)]);
        }
        const grid = S.linspace(0.001, 0.999, 333);
        const curves = { wald: [], wilson: [], lr: [], cp: [] };
        for (const pp of grid) {
          const acc = { wald: 0, wilson: 0, lr: 0, cp: 0 };
          for (let x = 0; x <= N; x++) {
            const w = S.pmf.binom(x, N, pp);
            for (const k in acc) if (I[k][x][0] <= pp && pp <= I[k][x][1]) acc[k] += w;
          }
          for (const k in acc) curves[k].push([pp, acc[k]]);
        }
        let ylo = 0.8;
        for (const [k] of METH) if (show[k].get()) ylo = Math.min(ylo, ...curves[k].map(d => d[1]));
        p.setDomain([0, 1], [Math.max(0, ylo - 0.02), 1.005]).clear();
        p.hline(0.95, { color: 'ink', width: 1.2 });
        for (const [k, , col] of METH) if (show[k].get()) p.line(curves[k], { color: col, width: k === 'wald' ? 1.3 : 1.7, opacity: 0.9 });
        p.legend(METH.filter(([k]) => show[k].get()).map(([, lab, col]) => ({ label: lab, color: col })), { left: false });
        for (const [k] of METH) {
          const v = curves[k].map(d => d[1]);
          const mid = curves[k].filter(d => d[0] >= 0.1 && d[0] <= 0.9).map(d => d[1]);
          tiles[k].set(pct(S.mean(v)), `min ${pct(Math.min(...v))} overall · ${pct(Math.min(...mid))} on [0.1, 0.9]`);
        }
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  const CIPOPS = {
    normal: { label: 'Normal(0, 1)', draw: r => r.normal(), mu: 0, sd: 1 },
    exp: { label: 'Exponential(1) — skewed', draw: r => r.exp(1), mu: 1, sd: 1 },
    lognormal: { label: 'Lognormal(0, 1) — very skewed', draw: r => r.lognormal(0, 1), mu: Math.exp(0.5), sd: Math.sqrt((Math.E - 1) * Math.E) },
  };
  Atlas.lab('confidence-intervals-dance', {
    title: 'The dance of confidence intervals',
    blurb: 'Each horizontal bar is an interval for the mean $\\mu$ from a fresh sample; red bars miss. The procedure, not any single interval, carries the guarantee. Try the naive interval $\\bar X\\pm z\\,S/\\sqrt n$ at $n=4$, or the $t$ interval on lognormal data, and watch the running coverage fall short of nominal.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el, { side: true });
      const pop = UI.select(controls, { label: 'Population', options: Object.keys(CIPOPS).map(k => ({ value: k, label: CIPOPS[k].label })), value: 'normal', onChange: reset });
      const meth = UI.segmented(controls, { label: 'Interval', options: [{ value: 'z', label: 'z, σ known' }, { value: 't', label: 't, σ unknown' }, { value: 'zs', label: 'z with S (naive)' }], value: 't', onChange: reset });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 2, max: 100, step: 1, value: 5, onChange: reset });
      const lev = UI.slider(controls, { label: 'Confidence level', min: 0.5, max: 0.99, step: 0.01, value: 0.95, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      let acc = 0;
      const loop = ctx.loop(dt => { acc += dt; if (acc > 0.05) { acc = 0; push(); draw(); } });
      ctx.playButton(row, loop, ['▶ Dance', '❚❚ Pause']);
      UI.button(row, '+100', () => { for (let i = 0; i < 100; i++) push(); draw(); });
      UI.button(row, 'Reset', reset);
      const p = new Plot(stage, { width: 640, height: 380, x: [-2, 2], y: [0, 101], xlabel: 'parameter value', yticks: 0 });
      const sShown = UI.stat(stats, 'Covering, last 100', { accent: 'accent' });
      const sAll = UI.stat(stats, 'Running coverage');
      const sW = UI.stat(stats, 'Mean width');
      let ivs = [], total = 0, hits = 0, wsum = 0;
      function push() {
        const P = CIPOPS[pop.get()], N = Math.round(n.get()), L = lev.get(), r = ctx.rng;
        const x = r.array(N, () => P.draw(r)), m = S.mean(x), sd = S.sd(x);
        let h;
        if (meth.get() === 'z') h = S.qnorm(1 - (1 - L) / 2) * P.sd / Math.sqrt(N);
        else if (meth.get() === 't') h = S.q.t(1 - (1 - L) / 2, N - 1) * sd / Math.sqrt(N);
        else h = S.qnorm(1 - (1 - L) / 2) * sd / Math.sqrt(N);
        const I = [m - h, m + h], ok = I[0] <= P.mu && P.mu <= I[1];
        ivs.push({ I, ok }); if (ivs.length > 100) ivs.shift();
        total++; if (ok) hits++; wsum += 2 * h;
      }
      function reset() { ivs = []; total = hits = wsum = 0; loop.stop(); for (let i = 0; i < 100; i++) push(); draw(); }
      function draw() {
        const P = CIPOPS[pop.get()];
        const span = 4.5 * P.sd / Math.sqrt(Math.max(2, n.get())) + 0.3 * P.sd;
        p.setDomain([P.mu - span, P.mu + span], [0, 101]).clear();
        ivs.forEach((v, i) => {
          p.segment(v.I[0], i + 1, v.I[1], i + 1, { color: v.ok ? 's1' : 'bad', width: v.ok ? 1.8 : 2.6, opacity: v.ok ? 0.75 : 1 });
          p.dot((v.I[0] + v.I[1]) / 2, i + 1, { color: v.ok ? 's1' : 'bad', r: 1.8, strokeWidth: 0 });
        });
        p.vline(P.mu, { color: 'ink', dash: false, width: 1.6 });
        p.text(P.mu, 100, ' true μ', { anchor: 'start', dy: -2, size: 11, color: 'muted' });
        const shown = ivs.filter(v => v.ok).length;
        sShown.set(`${shown} / ${ivs.length}`, `nominal ${pct(lev.get(), 0)}`);
        sAll.set(pct(hits / total), `${total} intervals`);
        sW.set(S.fmt(wsum / total));
      }
      reset();
    },
  });

  /* ------------------------------------------------------------------ */
  Atlas.lab('confidence-intervals-shortest', {
    title: 'Equal tails are not shortest: the variance interval',
    blurb: 'The pivot $(n-1)S^2/\\sigma^2\\sim\\chi^2_{n-1}$. Slide how the error $\\alpha$ is split between the two tails and watch the interval length for $\\sigma^2$ (proportional to $1/a-1/b$) and for $\\sigma$ (proportional to $1/\\sqrt a-1/\\sqrt b$). The two optima differ: "shortest" depends on the parameterisation, while equal tails is optimal for neither.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 3, max: 60, step: 1, value: 8, onChange: draw });
      const al = UI.slider(controls, { label: 'Level $\\alpha$', min: 0.01, max: 0.2, step: 0.01, value: 0.05, onChange: draw });
      const lam = UI.slider(controls, { label: 'Share of $\\alpha$ in lower tail', min: 0.01, max: 0.99, step: 0.01, value: 0.5, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 20], y: [0, 0.2], xlabel: 'pivot value', ylabel: 'density', title: 'χ² pivot and the chosen quantiles' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0.6, 1.6], xlabel: 'share of α in lower tail', ylabel: 'length / equal-tailed', title: 'Interval length vs tail split' });
      const sA = UI.stat(stats, 'Quantiles $a,\\,b$');
      const sV = UI.stat(stats, 'Length for $\\sigma^2$ vs equal tails', { accent: 'accent' });
      const sS = UI.stat(stats, 'Length for $\\sigma$ vs equal tails');
      const sO = UI.stat(stats, 'Optimal lower share');
      let cache = null;
      function curves() {
        const k = Math.round(n.get()) - 1, alpha = al.get();
        const key = k + ':' + alpha;
        if (cache && cache.key === key) return cache;
        const grid = S.linspace(0.01, 0.99, 99);
        const L2 = l => { const a = S.q.chisq(l * alpha, k), b = S.q.chisq(1 - (1 - l) * alpha, k); return [a, b]; };
        const ab = grid.map(L2);
        const [ae, be] = L2(0.5);
        const v0 = 1 / ae - 1 / be, s0 = 1 / Math.sqrt(ae) - 1 / Math.sqrt(be);
        const vr = ab.map(([a, b], i) => [grid[i], (1 / a - 1 / b) / v0]);
        const sr = ab.map(([a, b], i) => [grid[i], (1 / Math.sqrt(a) - 1 / Math.sqrt(b)) / s0]);
        cache = { key, k, alpha, L2, vr, sr, v0, s0 };
        return cache;
      }
      function draw() {
        const C = curves(), l = lam.get(), [a, b] = C.L2(l), k = C.k;
        const hi = S.q.chisq(0.999, k) * 1.05;
        const xs = S.linspace(0.001, hi, 300), ys = xs.map(x => S.pdf.chisq(x, k));
        const ymax = Math.min(Math.max(...ys.filter(Number.isFinite)) * 1.15, 1.2);
        pL.setDomain([0, hi], [0, ymax]).clear();
        pL.area(xs.filter(x => x <= a).map(x => [x, Math.min(ymax, S.pdf.chisq(x, k))]), { color: 'bad', opacity: 0.45 });
        pL.area(xs.filter(x => x >= b).map(x => [x, S.pdf.chisq(x, k)]), { color: 'bad', opacity: 0.45 });
        pL.line(xs.map((x, i) => [x, Math.min(ymax, ys[i])]), { color: 'ink', width: 2 });
        pL.vline(a, { color: 's1' }); pL.vline(b, { color: 's1' });
        const ymaxR = Math.min(3, Math.max(1.4, ...C.vr.map(d => d[1]).filter(v => v < 3)) * 1.02);
        pR.setDomain([0, 1], [Math.min(...C.vr.map(d => d[1]), ...C.sr.map(d => d[1])) * 0.97, ymaxR]).clear();
        pR.hline(1, { color: 'faint', width: 1 });
        pR.line(C.vr, { color: 's1', width: 2 });
        pR.line(C.sr, { color: 's3', width: 2 });
        const iv = S.argmin(C.vr.map(d => d[1])), is = S.argmin(C.sr.map(d => d[1]));
        pR.dot(C.vr[iv][0], C.vr[iv][1], { color: 's1', r: 4 });
        pR.dot(C.sr[is][0], C.sr[is][1], { color: 's3', r: 4 });
        pR.vline(l, { color: 'muted' });
        pR.legend([{ label: 'interval for σ²', color: 's1' }, { label: 'interval for σ', color: 's3' }]);
        const rv = (1 / a - 1 / b) / C.v0, rs = (1 / Math.sqrt(a) - 1 / Math.sqrt(b)) / C.s0;
        sA.set(`${S.fmt(a, 2)}, ${S.fmt(b, 2)}`, `tails ${S.fmt(l * C.alpha, 3)} / ${S.fmt((1 - l) * C.alpha, 3)}`);
        sV.set(pct(rv), `a²f(a) − b²f(b) = ${S.fmt(a * a * S.pdf.chisq(a, k) - b * b * S.pdf.chisq(b, k), 3)}`);
        sS.set(pct(rs));
        sO.set(`σ²: ${S.fmt(C.vr[iv][0], 2)} · σ: ${S.fmt(C.sr[is][0], 2)}`, `saves ${pct(1 - C.vr[iv][1], 0)} / ${pct(1 - C.sr[is][1], 0)}`);
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  // Rejection sets for the standard procedures. p: array of p-values. Returns boolean arrays.
  function mtProcedures(p, q) {
    const m = p.length, ord = S.range(m).sort((i, j) => p[i] - p[j]);
    const none = p.map(v => v <= q), bonf = p.map(v => v <= q / m), holm = new Array(m).fill(false), bh = new Array(m).fill(false);
    for (let j = 0; j < m; j++) { if (p[ord[j]] <= q / (m - j)) holm[ord[j]] = true; else break; }
    let R = 0;
    for (let k = m; k >= 1; k--) if (p[ord[k - 1]] <= q * k / m) { R = k; break; }
    for (let j = 0; j < R; j++) bh[ord[j]] = true;
    return { none, bonf, holm, bh, ord, R };
  }
  function mtDraw(r, m, pi1, mu, rho) {
    const m1 = Math.round(pi1 * m), alt = new Array(m).fill(false);
    for (let i = 0; i < m1; i++) alt[i] = true;
    const W = r.normal(), a = Math.sqrt(Math.max(0, rho)), b = Math.sqrt(1 - Math.max(0, rho));
    const z = alt.map(t => a * W + b * r.normal() + (t ? mu : 0));
    return { alt, z, p: z.map(v => 1 - S.cdf.normal(v)) };
  }
  const mtCount = (rej, alt) => { let R = 0, V = 0, Sx = 0; rej.forEach((x, i) => { if (x) { R++; if (alt[i]) Sx++; else V++; } }); return { R, V, S: Sx }; };

  Atlas.lab('multiple-testing-bh', {
    title: 'Benjamini–Hochberg vs Bonferroni on one data set',
    blurb: 'Sorted p-values against rank (zoomed to the smallest ones); orange points are true nulls, blue are real effects. BH rejects everything up to the last point under the line of slope $q/m$; Bonferroni only what is under the flat line $q/m$; Holm under the slowly rising curve. Increase $m$ with a fixed fraction of signals and watch Bonferroni’s discoveries vanish while BH’s scale up.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const m = UI.slider(controls, { label: 'Number of tests $m$', min: 10, max: 5000, step: 1, value: 200, log: true, fmt: v => String(Math.round(v)), onChange: regen });
      const pi1 = UI.slider(controls, { label: 'Fraction of real effects $\\pi_1$', min: 0, max: 0.5, step: 0.01, value: 0.1, onChange: regen });
      const mu = UI.slider(controls, { label: 'Effect size (z units)', min: 0, max: 5, step: 0.1, value: 3, onChange: regen });
      const q = UI.slider(controls, { label: 'Level $q$ (and $\\alpha$)', min: 0.01, max: 0.3, step: 0.01, value: 0.1, onInput: draw });
      UI.button(controls, 'New data', regen, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 50], y: [0, 0.05], xlabel: 'rank k', ylabel: 'sorted p-value' });
      const T = {
        none: UI.stat(stats, 'Unadjusted $p\\le q$'), bonf: UI.stat(stats, 'Bonferroni'),
        holm: UI.stat(stats, 'Holm'), bh: UI.stat(stats, 'Benjamini–Hochberg', { accent: 'accent' }),
      };
      let D = null;
      function regen() { D = mtDraw(ctx.rng, Math.round(m.get()), pi1.get(), mu.get(), 0); draw(); }
      function draw() {
        const M = D.p.length, Q = q.get(), res = mtProcedures(D.p, Q);
        const K = Math.min(M, Math.max(30, Math.ceil(res.R * 1.6) + 10));
        const sorted = res.ord.slice(0, K);
        const ymax = Math.max(Q * K / M * 1.25, D.p[sorted[K - 1]] * 1.05, 1e-6);
        p.setDomain([0, K + 1], [0, ymax]).clear();
        if (res.R) p.band(0.5, res.R + 0.5, { color: 'good', opacity: 0.08 });
        p.line([[0, 0], [K + 1, Q * (K + 1) / M]], { color: 's1', width: 2 });
        p.hline(Q / M, { color: 's4', width: 1.6, dash: true });
        p.line(S.range(K).map(j => [j + 1, Q / (M - j)]), { color: 's6', width: 1.4, dash: '2 3' });
        sorted.forEach((i, j) => p.dot(j + 1, D.p[i], { color: D.alt[i] ? 's1' : 's2', r: K > 120 ? 2.2 : 3.5 }));
        p.legend([{ label: 'BH line qk/m', color: 's1' }, { label: 'Holm q/(m−k+1)', color: 's6', dash: true }, { label: 'Bonferroni q/m', color: 's4', dash: true }, { label: 'real effect', color: 's1', fill: true }, { label: 'true null', color: 's2', fill: true }], { left: true });
        for (const k of ['none', 'bonf', 'holm', 'bh']) {
          const c = mtCount(res[k], D.alt);
          T[k].set(`${c.R} rejected`, `${c.V} false · FDP ${c.R ? S.fmt(c.V / c.R, 2) : '0'} · found ${c.S}/${D.alt.filter(Boolean).length}`);
        }
      }
      regen();
    },
  });

  Atlas.lab('multiple-testing-rates', {
    title: 'FWER, FDR and power over many replications',
    blurb: 'Each replication draws $m$ one-sided $z$-tests and applies four procedures; bars show the running FWER $\\P(V\\ge1)$, FDR $\\E[V/\\max(R,1)]$ and average power. BH’s FDR sits near $\\pi_0 q$; Bonferroni and Holm keep FWER below $q$ at a steep cost in power. Add equicorrelation $\\rho$: FDR stays controlled, but the spread of the per-study FDP (tile) balloons.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const m = UI.segmented(controls, { label: 'Number of tests $m$', options: ['10', '100', '1000'], value: '100', onChange: reset });
      const pi1 = UI.slider(controls, { label: 'Fraction of real effects $\\pi_1$', min: 0, max: 0.8, step: 0.05, value: 0.2, onChange: reset });
      const mu = UI.slider(controls, { label: 'Effect size (z units)', min: 0, max: 5, step: 0.1, value: 2.5, onChange: reset });
      const q = UI.slider(controls, { label: 'Level $q$', min: 0.01, max: 0.3, step: 0.01, value: 0.1, onChange: reset });
      const rho = UI.slider(controls, { label: 'Equicorrelation $\\rho$', min: 0, max: 0.9, step: 0.05, value: 0, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 15], y: [0, 1], ylabel: 'rate', xticks: 0 });
      const sN = UI.stat(stats, 'Replications');
      const sSd = UI.stat(stats, 'SD of BH’s FDP across studies', { accent: 'accent' });
      const sP = UI.stat(stats, '$\\P(\\text{FDP}>2q)$ for BH');
      const KEYS = [['none', 'Unadjusted', 's2'], ['bonf', 'Bonferroni', 's4'], ['holm', 'Holm', 's6'], ['bh', 'BH', 's1']];
      let A;
      function reset() { A = { n: 0 }; for (const [k] of KEYS) A[k] = { fw: 0, fdp: 0, pw: 0 }; A.fdps = []; loop.stop(); for (let i = 0; i < 40; i++) rep(); draw(); }
      function rep() {
        const M = +m.get(), D = mtDraw(ctx.rng, M, pi1.get(), mu.get(), rho.get()), res = mtProcedures(D.p, q.get());
        const m1 = D.alt.filter(Boolean).length;
        for (const [k] of KEYS) {
          const c = mtCount(res[k], D.alt);
          A[k].fw += c.V >= 1 ? 1 : 0; A[k].fdp += c.R ? c.V / c.R : 0; A[k].pw += m1 ? c.S / m1 : 0;
          if (k === 'bh') A.fdps.push(c.R ? c.V / c.R : 0);
        }
        A.n++;
      }
      function step() {
        const per = Math.max(1, Math.round(4000 / +m.get()));
        for (let i = 0; i < per; i++) rep();
        draw();
        if (A.n >= 20000) return false;
      }
      function draw() {
        p.clear();
        p.hline(q.get(), { color: 'bad', width: 1.2 });
        p.text(14.9, q.get(), 'q', { anchor: 'end', dy: -4, size: 11, color: 'bad' });
        ['FWER', 'FDR', 'Average power'].forEach((g, gi) => {
          KEYS.forEach(([k, , col], j) => {
            const v = gi === 0 ? A[k].fw / A.n : gi === 1 ? A[k].fdp / A.n : A[k].pw / A.n;
            const x0 = gi * 5 + 0.5 + j;
            p.bars([{ x0, x1: x0 + 0.9, y: v }], { key: 'y', color: col, opacity: 0.8, gap: 0 });
            p.text(x0 + 0.45, v, v < 0.995 ? S.fmt(v, 2) : '1', { anchor: 'middle', dy: -4, size: 10 });
          });
          p.text(gi * 5 + 2.5, 0, g, { anchor: 'middle', dy: 16, size: 12, weight: 600 });
        });
        p.legend(KEYS.map(([, lab, col]) => ({ label: lab, color: col, fill: true })), { left: true });
        sN.set(String(A.n));
        const sd = A.fdps.length > 1 ? S.sd(A.fdps) : NaN;
        sSd.set(S.fmt(sd, 3), `mean ${S.fmt(S.mean(A.fdps), 3)}; target ≤ π₀q = ${S.fmt((1 - pi1.get()) * q.get(), 3)}`);
        sP.set(pct(A.fdps.filter(v => v > 2 * q.get()).length / A.fdps.length));
      }
      reset();
    },
  });
  /* ------------------------------------------------------------------ */
  Atlas.lab('decision-theory-risk', {
    title: 'Risk functions of binomial estimators',
    blurb: 'Squared-error risk $R(\\theta,\\delta)$ of the MLE $X/n$, a Bayes rule $(X+a)/(n+a+b)$ for a Beta$(a,b)$ prior, and the minimax rule (the Bayes rule with $a=b=\\sqrt n/2$, whose risk is flat). No curve lies below all others: pick a criterion (maximum risk, or average risk under a uniform prior) and a different rule wins.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 1, max: 200, step: 1, value: 16, onInput: draw });
      const a = UI.slider(controls, { label: 'Bayes prior $a$', min: 0.5, max: 20, step: 0.5, value: 4, onInput: draw });
      const b = UI.slider(controls, { label: 'Bayes prior $b$', min: 0.5, max: 20, step: 0.5, value: 4, onInput: draw });
      const half = UI.toggle(controls, { label: 'Show the constant rule $\\delta\\equiv\\tfrac12$', value: false, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 1], y: [0, 0.02], xlabel: 'θ', ylabel: 'risk (MSE)' });
      const sM = UI.stat(stats, 'Max risk: MLE');
      const sB = UI.stat(stats, 'Max risk: Bayes$(a,b)$');
      const sX = UI.stat(stats, 'Max risk: minimax', { accent: 'accent' });
      const sU = UI.stat(stats, 'Uniform-prior Bayes risk');
      const risk = (t, N, A, B) => (N * t * (1 - t) + (A - (A + B) * t) ** 2) / (N + A + B) ** 2;
      const urisk = (N, A, B) => (N / 6 + A * A - A * (A + B) + (A + B) ** 2 / 3) / (N + A + B) ** 2;
      function draw() {
        const N = Math.round(n.get()), A = a.get(), B = b.get(), c = Math.sqrt(N) / 2;
        const ts = S.linspace(0, 1, 301);
        const mle = ts.map(t => [t, risk(t, N, 0, 0)]), bay = ts.map(t => [t, risk(t, N, A, B)]), mm = ts.map(t => [t, risk(t, N, c, c)]);
        const ymax = Math.max(1 / (4 * N), ...bay.map(d => d[1]).slice(0, 301)) * 1.15;
        p.setDomain([0, 1], [0, ymax]).clear();
        // Where minimax beats the MLE.
        const r0 = 1 / (4 * (1 + Math.sqrt(N)) ** 2), disc = Math.sqrt(Math.max(0, 1 - 4 * N * r0));
        p.band((1 - disc) / 2, (1 + disc) / 2, { color: 's3', opacity: 0.07 });
        if (half.get()) p.line(ts.map(t => [t, (t - 0.5) ** 2]), { color: 'muted', width: 1.4, dash: '2 3' });
        p.line(mle, { color: 's2', width: 2 });
        p.line(bay, { color: 's1', width: 2 });
        p.line(mm, { color: 's3', width: 2.4 });
        p.legend([{ label: 'MLE X/n', color: 's2' }, { label: `Bayes, Beta(${S.fmt(A, 1)}, ${S.fmt(B, 1)})`, color: 's1' }, { label: 'minimax (flat)', color: 's3' }].concat(half.get() ? [{ label: 'δ ≡ ½', color: 'muted', dash: true }] : []));
        const mx = arr => Math.max(...arr.map(d => d[1]));
        sM.set(S.fmt(mx(mle), 5), `= 1/(4n) at θ = ½`);
        sB.set(S.fmt(mx(bay), 5));
        sX.set(S.fmt(r0, 5), `beats MLE on [${S.fmt((1 - disc) / 2, 2)}, ${S.fmt((1 + disc) / 2, 2)}]`);
        const u = { MLE: urisk(N, 0, 0), Bayes: urisk(N, A, B), minimax: r0, 'Beta(1,1)': urisk(N, 1, 1) };
        const best = Object.keys(u).reduce((x, y) => (u[x] <= u[y] ? x : y));
        sU.set(`${best}`, Object.entries(u).map(([k, v]) => `${k} ${S.fmt(v, 5)}`).join(' · ') + ' (the Beta(1,1) rule is optimal here)');
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  // E[1/||X||^2] for X ~ N_d(theta, I), ||theta||^2 = lam: Poisson mixture of 1/(d+2k-2).
  function invNoncentral(d, lam) {
    const h = lam / 2;
    let s = 0;
    const K = Math.ceil(h + 12 * Math.sqrt(h + 1) + 20);
    for (let k = 0; k <= K; k++) s += Math.exp(-h + (k ? k * Math.log(h) : 0) - S.lgamma(k + 1)) / (d + 2 * k - 2);
    return s;
  }
  Atlas.lab('decision-theory-james-stein', {
    title: 'Stein’s paradox: shrinkage beats the MLE in d ≥ 3',
    blurb: 'Left: total risk $\\E\\|\\delta(X)-\\theta\\|^2$ as a function of $\\|\\theta\\|$ for the MLE (constant $d$), James–Stein (exact, via the noncentral $\\chi^2$ series) and positive-part James–Stein (Monte Carlo). Right: one draw — true means, observations, and their James–Stein shrinkage toward 0. The gain is enormous near the origin and never negative.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const d = UI.slider(controls, { label: 'Dimension $d$', min: 3, max: 50, step: 1, value: 10, onChange: () => { rc = null; draw(); } });
      const nt = UI.slider(controls, { label: 'Signal size $\\|\\theta\\|$', min: 0, max: 12, step: 0.1, value: 2, onInput: () => draw() });
      UI.button(controls, 'New draw', () => { seedDraw(); draw(); }, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 12], y: [0, 10], xlabel: '‖θ‖', ylabel: 'total risk', title: 'Risk functions' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 11], y: [-4, 4], xlabel: 'coordinate i', ylabel: 'value', title: 'One draw' });
      const sR = UI.stat(stats, 'JS risk / MLE risk at this $\\|\\theta\\|$', { accent: 'accent' });
      const sF = UI.stat(stats, 'Shrinkage factor this draw');
      const sL = UI.stat(stats, 'Loss this draw: MLE / JS / JS$^+$');
      let rc = null, dir = null, noise = null;
      function seedDraw() {
        const D = Math.round(d.get());
        dir = ctx.rng.array(D, () => ctx.rng.normal());
        const nr = Math.sqrt(S.sum(dir.map(v => v * v))) || 1;
        dir = dir.map(v => v / nr);
        noise = ctx.rng.array(D, () => ctx.rng.normal());
      }
      function riskCurves() {
        const D = Math.round(d.get()), grid = S.linspace(0, 12, 49), r = S.rng(7);
        const js = grid.map(t => [t, D - (D - 2) ** 2 * invNoncentral(D, t * t)]);
        const reps = Math.max(150, Math.round(6000 / D));
        const jsp = grid.map(t => {
          let acc = 0;
          for (let k = 0; k < reps; k++) {
            let nx = 0; const x = new Array(D);
            for (let i = 0; i < D; i++) { x[i] = (i === 0 ? t : 0) + r.normal(); nx += x[i] * x[i]; }
            const f = Math.max(0, 1 - (D - 2) / nx);
            for (let i = 0; i < D; i++) acc += (f * x[i] - (i === 0 ? t : 0)) ** 2;
          }
          return [t, acc / reps];
        });
        return { D, js, jsp };
      }
      function draw() {
        if (!rc) { rc = riskCurves(); seedDraw(); }
        const D = rc.D, T = nt.get();
        pL.setDomain([0, 12], [0, D * 1.12]).clear();
        pL.hline(D, { color: 's2', dash: false, width: 2 });
        pL.line(rc.jsp, { color: 's3', width: 1.8, dash: true });
        pL.line(rc.js, { color: 's1', width: 2.4 });
        pL.hline(2, { color: 'faint', width: 1 });
        pL.vline(T, { color: 'muted' });
        const jr = D - (D - 2) ** 2 * invNoncentral(D, T * T);
        pL.dot(T, jr, { color: 's1', r: 5 });
        pL.legend([{ label: 'MLE (= d)', color: 's2' }, { label: 'James–Stein', color: 's1' }, { label: 'positive part', color: 's3', dash: true }], { left: false });
        // One draw.
        const th = dir.map(v => v * T), x = th.map((v, i) => v + noise[i]);
        const nx = S.sum(x.map(v => v * v)), f = 1 - (D - 2) / nx, fp = Math.max(0, f);
        const js = x.map(v => f * v), jsp = x.map(v => fp * v);
        const loss = est => S.sum(est.map((v, i) => (v - th[i]) ** 2));
        const lim = Math.max(3, ...x.map(Math.abs), ...th.map(Math.abs)) * 1.1;
        pR.setDomain([0, D + 1], [-lim, lim]).clear();
        pR.hline(0, { color: 'faint', dash: false, width: 1 });
        for (let i = 0; i < D; i++) {
          pR.segment(i + 1, x[i], i + 1, jsp[i], { color: 's1', width: 1.4, opacity: 0.6 });
          pR.dot(i + 1, x[i], { color: 's2', r: D > 30 ? 2.5 : 3.5 });
          pR.dot(i + 1, jsp[i], { color: 's1', r: D > 30 ? 2.5 : 3.5 });
          pR.segment(i + 0.65, th[i], i + 1.35, th[i], { color: 'ink', width: 2 });
        }
        pR.legend([{ label: 'true θᵢ', color: 'ink' }, { label: 'Xᵢ (MLE)', color: 's2', fill: true }, { label: 'JS⁺ estimate', color: 's1', fill: true }]);
        sR.set(pct(jr / D), `risk ${S.fmt(jr, 2)} vs ${D}`);
        sF.set(S.fmt(f, 3), f < 0 ? 'negative: plain JS flips the sign; JS⁺ sets it to 0' : `‖X‖² = ${S.fmt(nx, 2)}`);
        sL.set(`${S.fmt(loss(x), 2)} / ${S.fmt(loss(js), 2)} / ${S.fmt(loss(jsp), 2)}`);
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  Atlas.lab('causal-inference-simpson', {
    title: 'Simpson’s paradox explorer',
    blurb: 'Severity $Z$ affects both who is treated and who recovers. Bars are recovery rates by treatment within each stratum and overall. Make severe patients much more likely to be treated and the crude comparison reverses the true (stratum-constant) effect; standardising over $Z$ recovers it. Switch to a finite sample to see the noise around these population numbers.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pz = UI.slider(controls, { label: 'Share of severe cases $\\P(Z=1)$', min: 0.1, max: 0.9, step: 0.05, value: 0.5, onInput: draw });
      const t0 = UI.slider(controls, { label: '$\\P(\\text{treated}\\mid\\text{mild})$', min: 0.05, max: 0.95, step: 0.05, value: 0.2, onInput: draw });
      const t1 = UI.slider(controls, { label: '$\\P(\\text{treated}\\mid\\text{severe})$', min: 0.05, max: 0.95, step: 0.05, value: 0.8, onInput: draw });
      const tau = UI.slider(controls, { label: 'True treatment effect $\\tau$', min: -0.25, max: 0.25, step: 0.01, value: 0.08, onInput: draw });
      const gap = UI.slider(controls, { label: 'Severity penalty on recovery', min: 0, max: 0.45, step: 0.01, value: 0.35, onInput: draw });
      const mode = UI.segmented(controls, { label: 'Data', options: [{ value: 'pop', label: 'Population' }, { value: 'smp', label: 'Sample' }], value: 'pop', onChange: draw });
      const n = UI.slider(controls, { label: 'Sample size (sample mode)', min: 50, max: 5000, step: 50, value: 500, onChange: draw });
      UI.button(controls, 'Redraw sample', () => { mode.set('smp'); draw(); });
      const p = new Plot(stage, { width: 640, height: 300, x: [0, 9], y: [0, 1], ylabel: 'recovery rate', xticks: 0 });
      const sC = UI.stat(stats, 'Crude difference');
      const sS = UI.stat(stats, 'Within mild / severe');
      const sA = UI.stat(stats, 'Standardised (adjusted)', { accent: 'accent' });
      const sT = UI.stat(stats, 'True ATE');
      const cl = v => Math.min(0.99, Math.max(0.01, v));
      function draw() {
        const PZ = pz.get(), T0 = t0.get(), T1 = t1.get(), TAU = tau.get(), G = gap.get();
        const rate = (a, z) => cl(0.7 - G * z + TAU * a);
        // cell[z][a] = {n, y}  (population: expected proportions; sample: counts)
        const cell = [[{}, {}], [{}, {}]];
        for (const z of [0, 1]) for (const a of [0, 1]) {
          const pa = z ? T1 : T0, w = (z ? PZ : 1 - PZ) * (a ? pa : 1 - pa);
          cell[z][a] = { n: w, y: w * rate(a, z) };
        }
        if (mode.get() === 'smp') {
          for (const z of [0, 1]) for (const a of [0, 1]) cell[z][a] = { n: 0, y: 0 };
          const N = Math.round(n.get()), r = ctx.rng;
          for (let i = 0; i < N; i++) {
            const z = r.u() < PZ ? 1 : 0, a = r.u() < (z ? T1 : T0) ? 1 : 0, y = r.u() < rate(a, z) ? 1 : 0;
            cell[z][a].n++; cell[z][a].y += y;
          }
        }
        const R = (z, a) => (cell[z][a].n ? cell[z][a].y / cell[z][a].n : NaN);
        const tot = a => { const nn = cell[0][a].n + cell[1][a].n; return nn ? (cell[0][a].y + cell[1][a].y) / nn : NaN; };
        const nz = z => cell[z][0].n + cell[z][1].n, Ntot = nz(0) + nz(1);
        const wz = [nz(0) / Ntot, nz(1) / Ntot];
        const crude = tot(1) - tot(0), d0 = R(0, 1) - R(0, 0), d1 = R(1, 1) - R(1, 0);
        const adj = wz[0] * d0 + wz[1] * d1;
        const ate = (1 - PZ) * (rate(1, 0) - rate(0, 0)) + PZ * (rate(1, 1) - rate(0, 1));
        p.clear();
        const groups = [['Mild', R(0, 1), R(0, 0), nz(0)], ['Severe', R(1, 1), R(1, 0), nz(1)], ['All (crude)', tot(1), tot(0), Ntot]];
        groups.forEach(([lab, r1, r0], g) => {
          const x = g * 3 + 0.4;
          p.bars([{ x0: x, x1: x + 1, y: r1 || 0 }], { key: 'y', color: 's1', opacity: 0.8, gap: 0 });
          p.bars([{ x0: x + 1.1, x1: x + 2.1, y: r0 || 0 }], { key: 'y', color: 's2', opacity: 0.8, gap: 0 });
          p.text(x + 0.5, r1 || 0, pct(r1, 0), { anchor: 'middle', dy: -4, size: 11 });
          p.text(x + 1.6, r0 || 0, pct(r0, 0), { anchor: 'middle', dy: -4, size: 11 });
          p.text(x + 1.05, 0, lab, { anchor: 'middle', dy: 16, size: 12, weight: 600 });
        });
        p.legend([{ label: 'treated', color: 's1', fill: true }, { label: 'untreated', color: 's2', fill: true }]);
        const sg = v => (Number.isFinite(v) ? (v >= 0 ? '+' : '') + S.fmt(100 * v, 1) + ' pts' : '—');
        const rev = Number.isFinite(crude) && Number.isFinite(adj) && Math.sign(crude) !== Math.sign(adj) && Math.abs(crude) > 0.005;
        sC.set(sg(crude), rev ? '<span style="color:var(--bad)">sign reversed!</span>' : `${pct(cell[1][1].n / (cell[0][1].n + cell[1][1].n), 0)} of treated are severe`);
        sS.set(`${sg(d0)} / ${sg(d1)}`);
        sA.set(sg(adj), 'Σ_z P(z) [rate(1,z) − rate(0,z)]');
        sT.set(sg(ate));
      }
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  const DAGS = {
    conf: { label: 'Confounder  A ← Z → Y', node: 'Z', edges: [['Z', 'A', 'γ'], ['Z', 'Y', 'γ'], ['A', 'Y', 'β']], total: (b, g) => b, adjustRight: true },
    med: { label: 'Mediator  A → M → Y', node: 'M', edges: [['A', 'M', 'γ'], ['M', 'Y', 'γ'], ['A', 'Y', 'β']], total: (b, g) => b + g * g, adjustRight: false },
    coll: { label: 'Collider  A → C ← Y', node: 'C', edges: [['A', 'C', 'γ'], ['Y', 'C', 'γ'], ['A', 'Y', 'β']], total: (b, g) => b, adjustRight: false },
  };
  function olsFit(cols, y) {
    const n = y.length, X = y.map((_, i) => [1, ...cols.map(c => c[i])]);
    const Xt = S.transpose(X), XtX = S.matmul(Xt, X), beta = S.solve(XtX, S.matvec(Xt, y));
    let rss = 0; for (let i = 0; i < n; i++) { const f = S.dot(X[i], beta); rss += (y[i] - f) ** 2; }
    const s2 = rss / (n - X[0].length), V = S.inv(XtX);
    return { beta, se: beta.map((_, j) => Math.sqrt(s2 * V[j][j])) };
  }
  Atlas.lab('causal-inference-dag', {
    title: 'To adjust or not to adjust: three DAGs',
    blurb: 'Linear Gaussian structural models with direct effect $\\beta$ of $A$ on $Y$ and path strength $\\gamma$ through a third variable. Compare the regression coefficient on $A$ with and without adjusting for the third variable against the true total causal effect. Adjustment is right for a confounder, wrong for a mediator (it returns the direct effect only), and harmful for a collider.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dag = UI.select(controls, { label: 'Causal structure', options: Object.keys(DAGS).map(k => ({ value: k, label: DAGS[k].label })), value: 'conf', onChange: resample });
      const beta = UI.slider(controls, { label: 'Direct effect $\\beta$ of $A$ on $Y$', min: -1, max: 1, step: 0.05, value: 0.5, onChange: resample });
      const gam = UI.slider(controls, { label: 'Path strength $\\gamma$', min: 0, max: 1.5, step: 0.05, value: 1, onChange: resample });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 50, max: 3000, step: 50, value: 500, onChange: resample });
      UI.button(controls, 'Resample', resample, { primary: true });
      stage.classList.add('two');
      const pG = new Plot(stage, { width: 420, height: 270, x: [0, 10], y: [0, 6], xticks: 0, yticks: 0, grid: false, title: 'Structural model' });
      const pS = new Plot(stage, { width: 420, height: 270, x: [-4, 4], y: [-4, 4], xlabel: 'A', ylabel: 'Y', title: 'Data and fitted slopes' });
      const sT = UI.stat(stats, 'True total effect of $A$');
      const sU = UI.stat(stats, 'Unadjusted slope');
      const sA = UI.stat(stats, 'Adjusted for third variable');
      const sV = UI.note(stats, '');
      let D = null;
      function resample() {
        const k = dag.get(), b = beta.get(), g = gam.get(), N = Math.round(n.get()), r = ctx.rng;
        const A = [], Y = [], W = [];
        for (let i = 0; i < N; i++) {
          let a, y, w;
          if (k === 'conf') { w = r.normal(); a = g * w + r.normal(); y = b * a + g * w + r.normal(); }
          else if (k === 'med') { a = r.normal(); w = g * a + r.normal(); y = b * a + g * w + r.normal(); }
          else { a = r.normal(); y = b * a + r.normal(); w = g * a + g * y + r.normal(); }
          A.push(a); Y.push(y); W.push(w);
        }
        D = { k, b, g, A, Y, W, un: olsFit([A], Y), ad: olsFit([A, W], Y) };
        draw();
      }
      function draw() {
        const G = DAGS[D.k];
        // DAG diagram.
        pG.clear();
        const pos = { A: [2, 1.4], Y: [8, 1.4] }; pos[G.node] = [5, 4.6];
        for (const [u, v, lab] of G.edges) {
          const [x0, y0] = pos[u], [x1, y1] = pos[v], L = Math.hypot(x1 - x0, y1 - y0), sx = (x1 - x0) / L, sy = (y1 - y0) / L, rr = 0.75;
          pG.arrow(x0 + sx * rr, y0 + sy * rr, x1 - sx * rr, y1 - sy * rr, { color: lab === 'β' ? 's1' : 's3', width: 2 });
          pG.text((x0 + x1) / 2, (y0 + y1) / 2, `${lab} = ${S.fmt(lab === 'β' ? D.b : D.g, 2)}`, { anchor: 'middle', dy: lab === 'β' ? 18 : -6, size: 12, color: lab === 'β' ? 's1' : 's3' });
        }
        for (const [k, [x, y]] of Object.entries(pos)) {
          pG.dot(x, y, { color: k === G.node ? 's4' : 'surface', r: 17, stroke: 'ink', strokeWidth: 1.5 });
          pG.text(x, y, k, { anchor: 'middle', dy: 5, size: 15, weight: 700 });
        }
        pG.text(5, 0.2, G.adjustRight ? 'Z opens a back-door path: adjust for it' : (D.k === 'med' ? 'M lies on the causal path: do not adjust (total effect)' : 'C is a collider: adjusting opens A → C ← Y'), { anchor: 'middle', size: 11, color: 'muted' });
        // Scatter.
        const lim = Math.max(3, S.quantile(D.A.map(Math.abs), 0.99), S.quantile(D.Y.map(Math.abs), 0.99)) * 1.05;
        pS.setDomain([-lim, lim], [-lim, lim]).clear();
        const m = Math.min(D.A.length, 800);
        pS.points(D.A.slice(0, m).map((a, i) => [a, D.Y[i]]), { color: 'muted', r: 1.8, opacity: 0.35 });
        const mA = S.mean(D.A), mY = S.mean(D.Y), tot = G.total(D.b, D.g);
        const ln = (s, col, dash, w) => pS.line([[-lim, mY + s * (-lim - mA)], [lim, mY + s * (lim - mA)]], { color: col, width: w, dash });
        ln(tot, 'ink', false, 2);
        ln(D.un.beta[1], 's2', true, 2);
        ln(D.ad.beta[1], 's4', true, 2);
        pS.legend([{ label: 'true total effect', color: 'ink' }, { label: 'unadjusted', color: 's2', dash: true }, { label: `adjusted for ${G.node}`, color: 's4', dash: true }], { left: true });
        const fmtE = f => `${S.fmt(f.beta[1], 3)} ± ${S.fmt(1.96 * f.se[1], 3)}`;
        const ok = f => Math.abs(f.beta[1] - tot) <= 1.96 * f.se[1] ? '<span style="color:var(--good)">CI covers truth</span>' : '<span style="color:var(--bad)">biased</span>';
        sT.set(S.fmt(tot, 3), D.k === 'med' ? `β + γ² (direct ${S.fmt(D.b, 2)} + indirect ${S.fmt(D.g * D.g, 2)})` : 'β');
        sU.set(fmtE(D.un), ok(D.un));
        sA.set(fmtE(D.ad), ok(D.ad));
        const theory = D.k === 'conf' ? `Unadjusted converges to $\\beta + \\gamma^2/(\\gamma^2+1)$ = ${S.fmt(D.b + D.g * D.g / (D.g * D.g + 1), 3)}.`
          : D.k === 'med' ? `Adjusted converges to the direct effect $\\beta$ = ${S.fmt(D.b, 3)}, not the total effect.`
            : 'Adjusting for the collider induces a spurious dependence between $A$ and $Y$.';
        sV.set(theory);
      }
      resample();
    },
  });
})();
