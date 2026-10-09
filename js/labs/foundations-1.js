/* Labs: Probability Foundations I. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const pct = (x, d = 1) => (Number.isFinite(x) ? (100 * x).toFixed(d) + '%' : '—');
  const TAU = 2 * Math.PI;

  /* ================= probability-spaces ================= */

  Atlas.lab('probability-spaces-bertrand', {
    title: "Bertrand's paradox: three meanings of \"a random chord\"",
    blurb: 'Each method draws chords of the unit circle "uniformly at random", yet the long-run fraction longer than the inscribed triangle\'s side ($\\sqrt3$) converges to $1/3$, $1/2$ or $1/4$. Watch the midpoints: the three methods put mass in different places.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const method = UI.segmented(controls, { label: 'Sampling procedure', options: [{ value: 'end', label: 'Random endpoints' }, { value: 'rad', label: 'Random radius' }, { value: 'mid', label: 'Random midpoint' }], value: 'end', onChange: reset });
      const showMid = UI.toggle(controls, { label: 'Show midpoints instead of chords', value: false, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Draw chords', '❚❚ Pause']);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pC = new Plot(stage, { width: 420, height: 300, x: [-1.15, 1.15], y: [-1.15, 1.15], equal: true, xticks: 0, yticks: 0, grid: false, title: 'Last 150 chords (long = coloured)' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [1, 4], y: [0, 1], logx: true, xlabel: 'chords drawn', ylabel: 'fraction long', title: 'Running proportion' });
      const sN = UI.stat(stats, 'Chords drawn');
      const sP = UI.stat(stats, 'Fraction longer than $\\sqrt3$', { accent: 'accent' });
      const sT = UI.stat(stats, 'Theory for this procedure');
      const THEORY = { end: 1 / 3, rad: 1 / 2, mid: 1 / 4 };
      let chords, count, longs, hist;
      function chord() {
        const r = ctx.rng, m = method.get();
        let d, phi;
        if (m === 'end') {
          const a = r.uniform(0, TAU), b = r.uniform(0, TAU);
          const p1 = [Math.cos(a), Math.sin(a)], p2 = [Math.cos(b), Math.sin(b)];
          const L = Math.hypot(p1[0] - p2[0], p1[1] - p2[1]);
          return { p1, p2, mid: [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2], long: L > Math.sqrt(3) };
        }
        phi = r.uniform(0, TAU);
        d = m === 'rad' ? r.uniform() : Math.sqrt(r.uniform());
        const mx = d * Math.cos(phi), my = d * Math.sin(phi), h = Math.sqrt(1 - d * d);
        const ux = -Math.sin(phi), uy = Math.cos(phi);
        return { p1: [mx - h * ux, my - h * uy], p2: [mx + h * ux, my + h * uy], mid: [mx, my], long: d < 0.5 };
      }
      function add(k) {
        for (let i = 0; i < k; i++) {
          const c = chord();
          chords.push(c); if (chords.length > 150) chords.shift();
          count++; if (c.long) longs++;
          if (count < 50 || count % Math.ceil(count / 200) === 0) hist.push([count, longs / count]);
        }
      }
      function reset() { chords = []; count = 0; longs = 0; hist = []; loop.stop(); add(60); draw(); }
      function step() { add(25); if (count >= 20000) { draw(); return false; } draw(); }
      function draw() {
        pC.clear();
        const circ = S.linspace(0, TAU, 120).map(t => [Math.cos(t), Math.sin(t)]);
        pC.line(circ, { color: 'ink', width: 1.5 });
        pC.polygon([[0, 1], [-Math.sqrt(3) / 2, -0.5], [Math.sqrt(3) / 2, -0.5]], { color: 'muted', opacity: 0.06, stroke: 'muted', width: 1 });
        pC.line(S.linspace(0, TAU, 80).map(t => [0.5 * Math.cos(t), 0.5 * Math.sin(t)]), { color: 'muted', width: 1, dash: true });
        if (showMid.get()) {
          pC.points(chords.filter(c => c.long).map(c => c.mid), { color: 's1', r: 2.2 });
          pC.points(chords.filter(c => !c.long).map(c => c.mid), { color: 'muted', r: 2.2 });
        } else {
          for (const c of chords) pC.segment(c.p1[0], c.p1[1], c.p2[0], c.p2[1], { color: c.long ? 's1' : 'muted', width: 1, opacity: c.long ? 0.6 : 0.3 });
        }
        const hi = Math.max(10, count);
        pR.setDomain([1, hi], [0, 1]).clear();
        for (const [k, v] of Object.entries(THEORY)) {
          pR.hline(v, { color: k === method.get() ? 'ink' : 'faint', width: 1.2 });
          pR.text(hi, v, k === 'end' ? '1/3' : k === 'rad' ? '1/2' : '1/4', { anchor: 'end', dy: -4, dx: -4, size: 11, color: 'muted' });
        }
        if (hist.length) pR.line(hist, { color: 's1', width: 2 });
        sN.set(String(count));
        sP.set(count ? S.fmt(longs / count) : '—');
        sT.set(S.fmt(THEORY[method.get()]), 'the triangle side is $\\sqrt3$; dashed circle has radius 1/2');
      }
      reset();
    },
  });

  Atlas.lab('probability-spaces-matching', {
    title: 'The matching problem: inclusion–exclusion and Bonferroni',
    blurb: 'Shuffle $n$ cards and count fixed points. Left: the Bonferroni partial sums $S_1 - S_2 + \\cdots \\pm S_r$ alternate above and below the exact $\\P(\\text{at least one match})$. Right: the number of matches is nearly Poisson(1) for every $n \\ge 5$ or so.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Number of cards $n$', min: 2, max: 30, step: 1, value: 6, onInput: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Shuffle', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pB = new Plot(stage, { width: 420, height: 270, x: [0.5, 8.5], y: [0, 1.05], xlabel: 'terms kept r', ylabel: 'partial sum', title: 'Bonferroni partial sums' });
      const pD = new Plot(stage, { width: 420, height: 270, x: [-0.6, 7.6], y: [0, 0.55], xlabel: 'number of matches', ylabel: 'probability', title: 'Distribution of matches' });
      const sEx = UI.stat(stats, 'Exact $\\P(\\ge 1\\text{ match})$', { accent: 'accent' });
      const sSim = UI.stat(stats, 'Simulated');
      const sLim = UI.stat(stats, 'Limit $1 - e^{-1}$');
      const sN = UI.stat(stats, 'Shuffles');
      let counts, total;
      const fact = k => Math.exp(S.lgamma(k + 1));
      function exactPmf(nn, k) { let s = 0; for (let j = 0; j <= nn - k; j++) s += (j % 2 ? -1 : 1) / fact(j); return s / fact(k); }
      function reset() { counts = new Array(31).fill(0); total = 0; loop.stop(); draw(); }
      function step() {
        const nn = Math.round(n.get()), perm = S.range(nn);
        for (let rep = 0; rep < 300; rep++) {
          ctx.rng.shuffle(perm);
          let m = 0; for (let i = 0; i < nn; i++) if (perm[i] === i) m++;
          counts[m]++; total++;
        }
        draw();
        if (total >= 30000) return false;
      }
      function draw() {
        const nn = Math.round(n.get()), R = Math.min(nn, 8);
        let exact = 0; for (let k = 1; k <= nn; k++) exact += (k % 2 ? 1 : -1) / fact(k);
        pB.clear();
        pB.hline(exact, { color: 'ink', dash: false, width: 1.5 });
        pB.hline(1 - Math.exp(-1), { color: 'muted', width: 1 });
        const ps = []; let s = 0;
        for (let r = 1; r <= R; r++) { s += (r % 2 ? 1 : -1) / fact(r); ps.push([r, s]); }
        pB.line(ps, { color: 'faint', width: 1.2 });
        ps.forEach(([r, v]) => pB.dot(r, v, { color: r % 2 ? 's1' : 's2', r: 5 }));
        pB.legend([{ label: 'odd r: upper bound', color: 's1' }, { label: 'even r: lower bound', color: 's2' }, { label: 'exact', color: 'ink' }]);
        pD.clear();
        if (total) pD.bars(S.range(8).map(k => ({ x0: k - 0.4, x1: k + 0.4, y: counts[k] / total })), { color: 's2', opacity: 0.35 });
        pD.stems(S.range(Math.min(nn, 7) + 1).map(k => [k - 0.12, exactPmf(nn, k)]), { color: 's1', r: 3 });
        pD.stems(S.range(8).map(k => [k + 0.12, S.pmf.poisson(k, 1)]), { color: 's3', r: 3 });
        pD.legend([{ label: 'exact', color: 's1' }, { label: 'Poisson(1)', color: 's3' }, { label: 'simulated', color: 's2', fill: true }]);
        sEx.set(S.fmt(exact, 4), `${nn} terms of the alternating series`);
        sSim.set(total ? S.fmt(1 - counts[0] / total, 4) : '—');
        sLim.set(S.fmt(1 - Math.exp(-1), 4));
        sN.set(String(total));
      }
      reset();
    },
  });

  /* ================= conditional-probability ================= */

  Atlas.lab('conditional-probability-base-rate', {
    title: 'Base rates: who actually tests positive?',
    blurb: 'Ten thousand people, each a dot. Red: sick and positive; amber: healthy but positive. Lower the prevalence and watch false positives swamp true ones, even with an accurate test. Requiring repeated (conditionally independent) positives multiplies the likelihood ratio.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const prev = UI.slider(controls, { label: 'Prevalence $\\P(D)$', min: 0.0005, max: 0.5, value: 0.01, log: true, fmt: v => pct(v, v < 0.01 ? 2 : 1), onInput: draw });
      const sens = UI.slider(controls, { label: 'Sensitivity $\\P(+\\mid D)$', min: 0.5, max: 0.999, step: 0.001, value: 0.95, onInput: draw });
      const spec = UI.slider(controls, { label: 'Specificity $\\P(-\\mid D^c)$', min: 0.5, max: 0.999, step: 0.001, value: 0.9, onInput: draw });
      const tests = UI.segmented(controls, { label: 'Positive on how many independent tests?', options: ['1', '2', '3'], value: '1', onChange: draw });
      stage.classList.add('two');
      const NX = 125, NY = 80, NP = NX * NY;
      const pG = new Plot(stage, { width: 420, height: 290, x: [-0.5, NX - 0.5], y: [-0.5, NY - 0.5], xticks: 0, yticks: 0, grid: false, title: '10,000 people' });
      const pC = new Plot(stage, { width: 420, height: 290, x: [0.0005, 0.5], y: [0, 1], logx: true, xlabel: 'prevalence (log scale)', ylabel: 'P(D | positive)', title: 'Posterior vs prevalence' });
      const sPPV = UI.stat(stats, '$\\P(D\\mid\\text{positive})$', { accent: 'accent' });
      const sLR = UI.stat(stats, 'Likelihood ratio');
      const sCounts = UI.stat(stats, 'True / false positives');
      const sNPV = UI.stat(stats, '$\\P(D\\mid -)$, one test');
      const post = (pi, se, sp, k) => { const a = pi * se ** k, b = (1 - pi) * (1 - sp) ** k; return a / (a + b); };
      // Fixed random placement of people so the picture is stable while sliding.
      const order = ctx.rng.shuffle(S.range(NP));
      function draw() {
        const pi = prev.get(), se = sens.get(), sp = spec.get(), k = +tests.get();
        const nD = Math.round(NP * pi);
        const tp = Math.round(nD * se ** k), fp = Math.round((NP - nD) * (1 - sp) ** k);
        const groups = { tp: [], fn: [], fp: [], tn: [] };
        for (let i = 0; i < NP; i++) {
          const id = order[i], pt = [id % NX, Math.floor(id / NX)];
          if (i < tp) groups.tp.push(pt); else if (i < nD) groups.fn.push(pt);
          else if (i < nD + fp) groups.fp.push(pt); else groups.tn.push(pt);
        }
        pG.clear();
        pG.points(groups.tn, { color: 'faint', r: 1.1, opacity: 0.5 });
        pG.points(groups.fn, { color: 's2', r: 1.6, opacity: 0.9 });
        pG.points(groups.fp, { color: 'warn', r: 1.6, opacity: 0.95 });
        pG.points(groups.tp, { color: 'bad', r: 1.9, opacity: 1 });
        pG.legend([{ label: 'sick, positive', color: 'bad', fill: true }, { label: 'healthy, positive', color: 'warn', fill: true }, { label: 'sick, negative', color: 's2', fill: true }]);
        pC.clear();
        const xs = S.linspace(Math.log10(0.0005), Math.log10(0.5), 150).map(v => 10 ** v);
        ['1', '2', '3'].forEach((kk, j) => pC.line(xs.map(x => [x, post(x, se, sp, +kk)]), { color: ['s1', 's3', 's5'][j], width: kk === tests.get() ? 2.5 : 1.2, opacity: kk === tests.get() ? 1 : 0.5 }));
        const pp = post(pi, se, sp, k);
        pC.vline(pi, { color: 'muted' });
        pC.dot(pi, pp, { color: 'accent', r: 5 });
        pC.legend([{ label: '1 test', color: 's1' }, { label: '2 tests', color: 's3' }, { label: '3 tests', color: 's5' }], { left: true });
        const LR = (se / (1 - sp)) ** k;
        sPPV.set(pct(pp), 'prior ' + pct(pi, 2));
        sLR.set(S.fmt(LR, 1), 'posterior odds = LR × prior odds');
        sCounts.set(`${tp} / ${fp}`, 'of 10,000 (expected counts)');
        sNPV.set(pct(pi * (1 - se) / (pi * (1 - se) + (1 - pi) * sp), 3));
      }
      draw();
    },
  });

  Atlas.lab('conditional-probability-monty-hall', {
    title: 'Monty Hall: the host\'s protocol is the evidence',
    blurb: 'You always pick door 1; the host opens $k$ other doors. A host who <em>knows</em> where the car is and always avoids it makes switching pay. A host who opens doors <em>blindly</em> (games where he reveals the car are discarded) makes every closed door equally likely, even though you see exactly the same thing.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nD = UI.slider(controls, { label: 'Doors $n$', min: 3, max: 10, step: 1, value: 3, onInput: reset });
      const kO = UI.slider(controls, { label: 'Doors opened $k$ (at most $n-2$)', min: 1, max: 8, step: 1, value: 1, onInput: reset });
      const host = UI.segmented(controls, { label: 'Host', options: [{ value: 'knows', label: 'Knows (avoids car)' }, { value: 'blind', label: 'Ignorant (opens at random)' }], value: 'knows', onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Play games', '❚❚ Pause']);
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pB = new Plot(stage, { width: 420, height: 270, x: [0, 2], y: [0, 1], xticks: 0, ylabel: 'win rate', title: 'Win rate by strategy' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [1, 1000], y: [0, 1], logx: true, xlabel: 'games kept', ylabel: 'running win rate', title: 'Convergence' });
      const sK = UI.stat(stats, 'Games kept');
      const sD = UI.stat(stats, 'Discarded (car revealed)');
      const sStick = UI.stat(stats, 'Stick wins');
      const sSwitch = UI.stat(stats, 'Switch wins', { accent: 'accent' });
      let kept, disc, stickW, switchW, hist;
      const kk = () => Math.min(Math.round(kO.get()), Math.round(nD.get()) - 2);
      function theory() {
        const n = Math.round(nD.get()), k = kk();
        return host.get() === 'knows' ? [1 / n, (n - 1) / (n * (n - 1 - k))] : [1 / (n - k), 1 / (n - k)];
      }
      function reset() { kept = 0; disc = 0; stickW = 0; switchW = 0; hist = []; loop.stop(); draw(); }
      function game() {
        const n = Math.round(nD.get()), k = kk(), r = ctx.rng;
        const car = r.int(0, n - 1);
        const others = S.range(n - 1).map(i => i + 1);
        let opened;
        if (host.get() === 'knows') opened = r.sample(others.filter(d => d !== car), k, false);
        else { opened = r.sample(others, k, false); if (opened.includes(car)) { disc++; return; } }
        const closed = others.filter(d => !opened.includes(d));
        const sw = r.choice(closed);
        kept++;
        if (car === 0) stickW++;
        if (sw === car) switchW++;
        if (kept < 40 || kept % Math.ceil(kept / 150) === 0) hist.push([kept, stickW / kept, switchW / kept]);
      }
      function step() { for (let i = 0; i < 60; i++) game(); draw(); if (kept >= 20000) return false; }
      function draw() {
        const [tS, tW] = theory();
        pB.clear();
        [[stickW, tS, 'Stick', 's2'], [switchW, tW, 'Switch (to a random closed door)', 's1']].forEach(([w, t, lab, c], i) => {
          const v = kept ? w / kept : 0;
          if (kept) pB.bars([{ x0: i + 0.2, x1: i + 0.8, y: v }], { color: c, opacity: 0.7 });
          pB.segment(i + 0.12, t, i + 0.88, t, { color: 'ink', width: 2, dash: '4 3' });
          if (kept) pB.text(i + 0.5, v, pct(v), { anchor: 'middle', dy: -6, size: 12, weight: 600 });
          pB.text(i + 0.5, 0, lab, { anchor: 'middle', dy: 14, size: 11, color: 'muted' });
        });
        pB.legend([{ label: 'theory', color: 'ink', dash: true }]);
        const hi = Math.max(1000, kept);
        pR.setDomain([1, hi], [0, 1]).clear();
        pR.hline(tS, { color: 's2', width: 1 }); pR.hline(tW, { color: 's1', width: 1 });
        if (hist.length) { pR.line(hist.map(h => [h[0], h[1]]), { color: 's2', width: 2 }); pR.line(hist.map(h => [h[0], h[2]]), { color: 's1', width: 2 }); }
        sK.set(String(kept));
        sD.set(host.get() === 'knows' ? '0' : String(disc), host.get() === 'knows' ? 'never happens' : pct(disc / Math.max(1, kept + disc)) + ' of games');
        sStick.set(kept ? pct(stickW / kept) : '—', 'theory ' + pct(tS));
        sSwitch.set(kept ? pct(switchW / kept) : '—', 'theory ' + pct(tW));
      }
      reset();
    },
  });

  /* ================= random-variables ================= */

  const CONT = {
    normal: { label: 'Normal(0,1)', cdf: x => S.cdf.normal(x), pdf: x => S.pdf.normal(x), q: u => S.q.normal(u), range: [-3.5, 3.5] },
    exp: { label: 'Exponential(1)', cdf: x => S.cdf.exp(x), pdf: x => S.pdf.exp(x), q: u => S.q.exp(u), range: [-0.5, 5] },
    unif: { label: 'Uniform(0,2)', cdf: x => S.cdf.uniform(x, 0, 2), pdf: x => S.pdf.uniform(x, 0, 2), q: u => 2 * u, range: [-0.5, 2.5] },
  };

  Atlas.lab('random-variables-cdf', {
    title: 'A mixed distribution: jumps, flats and the quantile function',
    blurb: 'With probability $w$ the variable equals $a$ exactly; otherwise it follows a continuous law. The CDF jumps by $w$ at $a$. Arrows show the inverse transform $X = F^{-1}(U)$: every $U$ that lands in the jump maps to the atom.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fam = UI.select(controls, { label: 'Continuous part $G$', options: Object.keys(CONT).map(k => ({ value: k, label: CONT[k].label })), value: 'exp', onChange: () => { newDraw(); } });
      const w = UI.slider(controls, { label: 'Atom weight $w = \\P(X=a)$', min: 0, max: 0.6, step: 0.01, value: 0.3, onInput: draw });
      const a = UI.slider(controls, { label: 'Atom location $a$', min: -2, max: 3, step: 0.05, value: 0, onInput: draw });
      const n = UI.slider(controls, { label: 'Draws $n$', min: 100, max: 5000, step: 100, value: 2000, onChange: newDraw });
      UI.button(controls, 'New uniforms', newDraw, { primary: true });
      stage.classList.add('two');
      const pF = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'F(x)', title: 'CDF and inverse transform' });
      const pH = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'density', title: 'Draws of F⁻¹(U)' });
      const sW = UI.stat(stats, 'Jump $F(a) - F(a^-)$');
      const sE = UI.stat(stats, 'Fraction of draws equal to $a$', { accent: 'accent' });
      const sM = UI.stat(stats, 'Median $F^{-1}(1/2)$');
      const sB = UI.stat(stats, '$\\P(X \\lt a)$ vs $\\P(X \\le a)$');
      let U = [];
      function newDraw() { U = ctx.rng.array(Math.round(n.get()), () => ctx.rng.u()); draw(); }
      function draw() {
        const G = CONT[fam.get()], ww = w.get(), aa = a.get();
        const F = x => (1 - ww) * G.cdf(x) + (x >= aa ? ww : 0);
        const lo0 = (1 - ww) * G.cdf(aa);
        const Finv = u => (u <= lo0 ? G.q(Math.min(u / (1 - ww), 1 - 1e-12)) : u <= lo0 + ww ? aa : G.q(Math.min((u - ww) / (1 - ww), 1 - 1e-12)));
        const lo = Math.min(G.range[0], aa - 0.5), hi = Math.max(G.range[1], aa + 0.5);
        pF.setDomain([lo, hi], [0, 1.02]).clear();
        const left = S.linspace(lo, aa, 150).map(x => [x, (1 - ww) * G.cdf(x)]);
        const right = S.linspace(aa, hi, 150).map(x => [x, F(x)]);
        pF.rect(lo, lo0, hi, lo0 + ww, { color: 's2', opacity: 0.08 });
        // inverse-transform arrows for the first 22 uniforms
        U.slice(0, 22).forEach(u => {
          const x = Finv(u), inJump = ww > 0 && u > lo0 && u <= lo0 + ww;
          pF.segment(lo, u, x, u, { color: inJump ? 's2' : 's1', width: 1, opacity: 0.45 });
          pF.segment(x, u, x, 0, { color: inJump ? 's2' : 's1', width: 1, opacity: 0.25, dash: '2 3' });
        });
        pF.line(left, { color: 'ink', width: 2.2 });
        pF.line(right, { color: 'ink', width: 2.2 });
        if (ww > 0) { pF.dot(aa, lo0, { color: 'surface', stroke: 'ink', r: 4 }); pF.dot(aa, lo0 + ww, { color: 'ink', r: 4 }); }
        const xs = U.map(Finv);
        const atomCount = xs.filter(x => x === aa).length;
        const cont = xs.filter(x => x !== aa);
        const hist = S.histogram(cont, { bins: 40, lo, hi });
        for (const b of hist) b.density = b.count / (xs.length * (b.x1 - b.x0));
        const dens = S.linspace(lo, hi, 300).map(x => [x, (1 - ww) * G.pdf(x)]);
        const ymax = Math.max(0.3, ...hist.map(b => b.density), ...dens.map(d => d[1]).filter(Number.isFinite)) * 1.2;
        pH.setDomain([lo, hi], [0, ymax]).clear();
        pH.bars(hist, { color: 's1', opacity: 0.4 });
        pH.line(dens, { color: 'ink', width: 1.8 });
        if (ww > 0) {
          pH.segment(aa, 0, aa, ymax * 0.88, { color: 's2', width: 3 });
          pH.dot(aa, ymax * 0.88, { color: 's2', r: 5 });
          pH.text(aa, ymax * 0.88, `atom: ${pct(atomCount / xs.length)}`, { dx: 8, dy: 4, size: 11, color: 's2' });
        }
        pH.legend([{ label: '(1−w)·g(x)', color: 'ink' }, { label: 'draws (continuous part)', color: 's1', fill: true }]);
        sW.set(S.fmt(ww, 2));
        sE.set(pct(atomCount / Math.max(1, xs.length)));
        sM.set(S.fmt(Finv(0.5)), lo0 < 0.5 && lo0 + ww >= 0.5 ? 'the median is the atom' : '');
        sB.set(`${S.fmt(lo0)} vs ${S.fmt(lo0 + ww)}`);
      }
      newDraw();
    },
  });

  const PIT_TRUTH = {
    normal: { label: 'Normal(0,1)', draw: r => r.normal(), cdf: x => S.cdf.normal(x) },
    t3: { label: 'Student t, 3 df', draw: r => r.t(3), cdf: x => S.cdf.t(x, 3) },
    exp: { label: 'Exponential(1)', draw: r => r.exp(1), cdf: x => S.cdf.exp(x) },
    pois: { label: 'Poisson(4) (discrete)', draw: r => r.poisson(4), cdf: x => (x < 0 ? 0 : S.cdf.poisson(Math.floor(x), 4)), discrete: true },
  };

  Atlas.lab('random-variables-pit', {
    title: 'The probability integral transform as a model check',
    blurb: 'Compute $U_i = G(X_i)$ under a hypothesized CDF $G$. If $G$ is the true continuous $F$, the $U_i$ are uniform. A model that is too narrow gives a U-shape, too wide a hump, biased a slope. With discrete data even the true $F$ fails — until you randomize.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const truth = UI.select(controls, { label: 'True distribution $F$', options: Object.keys(PIT_TRUTH).map(k => ({ value: k, label: PIT_TRUTH[k].label })), value: 'normal', onChange: newSample });
      const model = UI.segmented(controls, { label: 'Hypothesized $G$', options: [{ value: 'true', label: 'The true F' }, { value: 'norm', label: 'Normal(μ, σ)' }], value: 'norm', onChange: draw });
      const mu = UI.slider(controls, { label: 'Model mean $\\mu$', min: -2, max: 2, step: 0.05, value: 0, onInput: draw });
      const sg = UI.slider(controls, { label: 'Model sd $\\sigma$', min: 0.3, max: 3, step: 0.05, value: 1, onInput: draw });
      const rand = UI.toggle(controls, { label: 'Randomized PIT (discrete data, true F)', value: false, onChange: draw });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 50, max: 5000, step: 50, value: 1000, onChange: newSample });
      UI.button(controls, 'New sample', newSample, { primary: true });
      stage.classList.add('two');
      const pH = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 3], xlabel: 'U = G(X)', ylabel: 'density', title: 'PIT histogram' });
      const pE = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [0, 1], xlabel: 'u', ylabel: 'ECDF of U', title: 'ECDF of U vs uniform' });
      const sD = UI.stat(stats, 'KS distance $D_n$', { accent: 'accent' });
      const sC = UI.stat(stats, '5% critical value');
      const sV = UI.stat(stats, 'Verdict');
      let X = [], V = [];
      function newSample() {
        const T = PIT_TRUTH[truth.get()], nn = Math.round(n.get());
        X = ctx.rng.array(nn, () => T.draw(ctx.rng));
        V = ctx.rng.array(nn, () => ctx.rng.u());
        draw();
      }
      function draw() {
        const T = PIT_TRUTH[truth.get()];
        let U;
        if (model.get() === 'true') {
          if (T.discrete && rand.get()) U = X.map((x, i) => { const Fm = T.cdf(x - 1), Fx = T.cdf(x); return Fm + V[i] * (Fx - Fm); });
          else U = X.map(T.cdf);
        } else U = X.map(x => S.cdf.normal(x, mu.get(), sg.get()));
        const nn = U.length, s = U.slice().sort((a, b) => a - b);
        let D = 0;
        for (let i = 0; i < nn; i++) D = Math.max(D, Math.abs((i + 1) / nn - s[i]), Math.abs(s[i] - i / nn));
        const crit = 1.358 / Math.sqrt(nn);
        const h = S.histogram(U, { bins: 20, lo: 0, hi: 1 });
        const ymax = Math.max(2, ...h.map(b => b.density)) * 1.1;
        pH.setDomain([0, 1], [0, ymax]).clear();
        pH.bars(h, { color: D > crit ? 'bad' : 's1', opacity: 0.45 });
        pH.hline(1, { color: 'ink', dash: false, width: 1.5 });
        pE.clear();
        pE.line([[0, 0], [1, 1]], { color: 'ink', width: 1.2, dash: true });
        pE.line([[0, crit], [1 - crit, 1]], { color: 'faint', width: 1 });
        pE.line([[crit, 0], [1, 1 - crit]], { color: 'faint', width: 1 });
        const stepPts = [[0, 0]];
        const stride = Math.max(1, Math.floor(nn / 400));
        for (let i = 0; i < nn; i += stride) stepPts.push([s[i], (i + 1) / nn]);
        stepPts.push([1, 1]);
        pE.step(stepPts, { color: D > crit ? 'bad' : 's1', width: 2 });
        sD.set(S.fmt(D, 4));
        sC.set(S.fmt(crit, 4), 'asymptotic, $1.358/\\sqrt n$');
        let verdict = D > crit ? '<span style="color:var(--bad)">reject uniformity</span>' : '<span style="color:var(--good)">consistent with uniform</span>';
        sV.set(verdict, T.discrete && model.get() === 'true' && !rand.get() ? 'F(X) is discrete here: only countably many values' : '');
      }
      newSample();
    },
  });

  /* ================= distribution-families ================= */

  const lgam = S.lgamma;
  const FAM = {
    binom: { label: 'Binomial(n, p)', disc: true, params: [{ k: 'n', label: 'Trials $n$', min: 1, max: 100, step: 1, value: 20 }, { k: 'p', label: 'Success prob. $p$', min: 0.01, max: 0.99, step: 0.01, value: 0.3 }],
      f: (x, P) => S.pmf.binom(x, Math.round(P.n), P.p), mean: P => Math.round(P.n) * P.p, var: P => Math.round(P.n) * P.p * (1 - P.p), range: P => [0, Math.round(P.n)] },
    pois: { label: 'Poisson(λ)', disc: true, params: [{ k: 'l', label: 'Rate $\\lambda$', min: 0.1, max: 50, step: 0.1, value: 4 }],
      f: (x, P) => S.pmf.poisson(x, P.l), mean: P => P.l, var: P => P.l, range: P => [0, Math.ceil(P.l + 6 * Math.sqrt(P.l) + 5)] },
    geom: { label: 'Geometric(p), trials', disc: true, params: [{ k: 'p', label: 'Success prob. $p$', min: 0.02, max: 0.95, step: 0.01, value: 0.2 }],
      f: (x, P) => S.pmf.geom(x, P.p), mean: P => 1 / P.p, var: P => (1 - P.p) / P.p ** 2, range: P => [1, Math.ceil(Math.log(0.001) / Math.log(1 - P.p)) + 1] },
    negbin: { label: 'Negative binomial(r, p), failures', disc: true, params: [{ k: 'r', label: 'Successes $r$', min: 0.5, max: 30, step: 0.5, value: 3 }, { k: 'p', label: 'Success prob. $p$', min: 0.05, max: 0.95, step: 0.01, value: 0.4 }],
      f: (x, P) => S.pmf.negbinom(x, P.r, P.p), mean: P => P.r * (1 - P.p) / P.p, var: P => P.r * (1 - P.p) / P.p ** 2,
      range: P => [0, Math.ceil(P.r * (1 - P.p) / P.p + 6 * Math.sqrt(P.r * (1 - P.p)) / P.p + 3)] },
    hyper: { label: 'Hypergeometric(N, K, n)', disc: true, params: [{ k: 'N', label: 'Population $N$', min: 10, max: 200, step: 1, value: 50 }, { k: 'K', label: 'Successes $K$ (≤ N)', min: 1, max: 200, step: 1, value: 20 }, { k: 'n', label: 'Draws $n$ (≤ N)', min: 1, max: 200, step: 1, value: 10 }],
      fix: P => { P.N = Math.round(P.N); P.K = Math.min(Math.round(P.K), P.N); P.n = Math.min(Math.round(P.n), P.N); return P; },
      f: (x, P) => (x < Math.max(0, P.n - P.N + P.K) || x > Math.min(P.n, P.K) ? 0 : Math.exp(S.lchoose(P.K, x) + S.lchoose(P.N - P.K, P.n - x) - S.lchoose(P.N, P.n))),
      mean: P => P.n * P.K / P.N, var: P => (P.N > 1 ? P.n * (P.K / P.N) * (1 - P.K / P.N) * (P.N - P.n) / (P.N - 1) : 0), range: P => [0, P.n] },
    normal: { label: 'Normal(μ, σ²)', params: [{ k: 'm', label: 'Mean $\\mu$', min: -3, max: 3, step: 0.1, value: 0 }, { k: 's', label: 'SD $\\sigma$', min: 0.2, max: 3, step: 0.05, value: 1 }],
      f: (x, P) => S.pdf.normal(x, P.m, P.s), F: (x, P) => S.cdf.normal(x, P.m, P.s), mean: P => P.m, var: P => P.s ** 2, range: P => [P.m - 4 * P.s, P.m + 4 * P.s] },
    exp: { label: 'Exponential (rate λ)', params: [{ k: 'l', label: 'Rate $\\lambda$', min: 0.1, max: 5, step: 0.05, value: 1 }],
      f: (x, P) => S.pdf.exp(x, P.l), F: (x, P) => S.cdf.exp(x, P.l), mean: P => 1 / P.l, var: P => 1 / P.l ** 2, range: P => [0, -Math.log(0.002) / P.l] },
    gamma: { label: 'Gamma(α, β), scale β', params: [{ k: 'a', label: 'Shape $\\alpha$', min: 0.2, max: 20, step: 0.1, value: 2 }, { k: 'b', label: 'Scale $\\beta$', min: 0.1, max: 5, step: 0.05, value: 1 }],
      f: (x, P) => S.pdf.gamma(x, P.a, P.b), F: (x, P) => S.cdf.gamma(x, P.a, P.b), mean: P => P.a * P.b, var: P => P.a * P.b ** 2, range: P => [0, S.q.gamma(0.998, P.a, P.b)] },
    beta: { label: 'Beta(a, b)', params: [{ k: 'a', label: '$a$', min: 0.2, max: 10, step: 0.1, value: 2 }, { k: 'b', label: '$b$', min: 0.2, max: 10, step: 0.1, value: 5 }],
      f: (x, P) => S.pdf.beta(x, P.a, P.b), F: (x, P) => S.cdf.beta(x, P.a, P.b), mean: P => P.a / (P.a + P.b), var: P => P.a * P.b / ((P.a + P.b) ** 2 * (P.a + P.b + 1)), range: () => [0, 1] },
    chisq: { label: 'Chi-square(k)', params: [{ k: 'k', label: 'Degrees of freedom $k$', min: 1, max: 50, step: 1, value: 4 }],
      f: (x, P) => S.pdf.chisq(x, P.k), F: (x, P) => S.cdf.chisq(x, P.k), mean: P => P.k, var: P => 2 * P.k, range: P => [0, S.q.chisq(0.998, P.k)] },
    t: { label: 'Student t(ν)', params: [{ k: 'v', label: 'Degrees of freedom $\\nu$', min: 0.5, max: 50, step: 0.5, value: 3 }],
      f: (x, P) => S.pdf.t(x, P.v), F: (x, P) => S.cdf.t(x, P.v), mean: P => (P.v > 1 ? 0 : NaN), var: P => (P.v > 2 ? P.v / (P.v - 2) : P.v > 1 ? Infinity : NaN), range: () => [-6, 6] },
    cauchy: { label: 'Cauchy(0, s)', params: [{ k: 's', label: 'Scale $s$', min: 0.2, max: 3, step: 0.05, value: 1 }],
      f: (x, P) => S.pdf.cauchy(x, 0, P.s), F: (x, P) => S.cdf.cauchy(x, 0, P.s), mean: () => NaN, var: () => NaN, range: P => [-10 * P.s, 10 * P.s] },
    lognormal: { label: 'Lognormal(μ, σ²)', params: [{ k: 'm', label: 'Log-mean $\\mu$', min: -1, max: 1, step: 0.05, value: 0 }, { k: 's', label: 'Log-SD $\\sigma$', min: 0.1, max: 1.5, step: 0.05, value: 0.5 }],
      f: (x, P) => S.pdf.lognormal(x, P.m, P.s), F: (x, P) => (x <= 0 ? 0 : S.cdf.normal(Math.log(x), P.m, P.s)), mean: P => Math.exp(P.m + P.s ** 2 / 2),
      var: P => (Math.exp(P.s ** 2) - 1) * Math.exp(2 * P.m + P.s ** 2), range: P => [0, Math.exp(P.m + 2.6 * P.s)] },
    weibull: { label: 'Weibull(k, λ)', params: [{ k: 'k', label: 'Shape $k$', min: 0.3, max: 5, step: 0.05, value: 1.5 }, { k: 'l', label: 'Scale $\\lambda$', min: 0.2, max: 3, step: 0.05, value: 1 }],
      f: (x, P) => S.pdf.weibull(x, P.k, P.l), F: (x, P) => S.cdf.weibull(x, P.k, P.l), mean: P => P.l * S.gamma(1 + 1 / P.k),
      var: P => P.l ** 2 * (S.gamma(1 + 2 / P.k) - S.gamma(1 + 1 / P.k) ** 2), range: P => [0, P.l * (-Math.log(0.002)) ** (1 / P.k)] },
    pareto: { label: 'Pareto(α), x_m = 1', params: [{ k: 'a', label: 'Tail index $\\alpha$', min: 0.5, max: 6, step: 0.05, value: 2.5 }],
      f: (x, P) => S.pdf.pareto(x, P.a, 1), F: (x, P) => S.cdf.pareto(x, P.a, 1), mean: P => (P.a > 1 ? P.a / (P.a - 1) : Infinity),
      var: P => (P.a > 2 ? P.a / ((P.a - 1) ** 2 * (P.a - 2)) : P.a > 1 ? Infinity : NaN), range: P => [0, Math.min(0.01 ** (-1 / P.a), 30)] },
  };
  const fmtMom = v => (Number.isNaN(v) ? 'undefined' : S.fmt(v));

  Atlas.lab('distribution-families-explorer', {
    title: 'Distribution explorer: pmf/pdf and CDF',
    blurb: 'Pick a family and move its parameters. The dashed curve is a normal with the same mean and variance: compare how much probability lies within $\\mu \\pm 2\\sigma$ (95.4% for a normal) — heavy and skewed families differ exactly where it matters.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fam = UI.select(controls, { label: 'Family', options: Object.keys(FAM).map(k => ({ value: k, label: FAM[k].label })), value: 'gamma', onChange: build });
      const box = UI.h('div', '', controls);
      const ov = UI.toggle(controls, { label: 'Overlay matched normal', value: true, onChange: draw });
      stage.classList.add('two');
      const pP = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'p(x) or f(x)', title: 'Mass / density' });
      const pF = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'F(x)', title: 'CDF' });
      const sM = UI.stat(stats, 'Mean $\\mu$');
      const sS = UI.stat(stats, 'SD $\\sigma$');
      const sMed = UI.stat(stats, 'Median');
      const s2 = UI.stat(stats, '$\\P(|X-\\mu| \\le 2\\sigma)$', { accent: 'accent' });
      let sliders = [];
      function build() {
        box.innerHTML = '';
        sliders = FAM[fam.get()].params.map(q => [q.k, UI.slider(box, Object.assign({}, q, { onInput: draw }))]);
        draw();
      }
      function params() { const P = {}; for (const [k, s] of sliders) P[k] = s.get(); const D = FAM[fam.get()]; return D.fix ? D.fix(P) : P; }
      function draw() {
        const D = FAM[fam.get()], P = params();
        const [lo, hi] = D.range(P), mu = D.mean(P), v = D.var(P), sd = Math.sqrt(v);
        const finite = Number.isFinite(mu) && Number.isFinite(sd) && sd > 0;
        let within = NaN, med;
        if (D.disc) {
          const ks = S.range(hi - lo + 1).map(i => lo + i), pm = ks.map(k => D.f(k, P));
          const cum = S.cumsum(pm);
          const ymax = Math.max(...pm) * 1.18 || 1;
          pP.setDomain([lo - 0.8, hi + 0.8], [0, ymax]).clear();
          pP.stems(ks.map((k, i) => [k, pm[i]]), { color: 's1', r: ks.length > 60 ? 1.8 : 3, width: ks.length > 60 ? 1.2 : 2 });
          pF.setDomain([lo - 0.8, hi + 0.8], [0, 1.03]).clear();
          pF.step([[lo - 0.8, 0]].concat(ks.map((k, i) => [k, cum[i]])).concat([[hi + 0.8, cum[cum.length - 1]]]), { color: 's1', width: 2 });
          med = ks[cum.findIndex(c => c >= 0.5)];
          if (finite) within = ks.reduce((s, k, i) => s + (Math.abs(k - mu) <= 2 * sd + 1e-12 ? pm[i] : 0), 0);
          if (ov.get() && finite) {
            const xs = S.linspace(lo - 0.8, hi + 0.8, 300);
            pP.line(xs.map(x => [x, S.pdf.normal(x, mu, sd)]), { color: 'ink', dash: true, width: 1.5 });
            pF.line(xs.map(x => [x, S.cdf.normal(x, mu, sd)]), { color: 'ink', dash: true, width: 1.5 });
          }
        } else {
          const xs = S.linspace(lo, hi, 400), ys = xs.map(x => D.f(x, P));
          const fin = ys.filter(y => Number.isFinite(y)).sort((a, b) => a - b);
          const ymax = Math.min(fin[fin.length - 1], 4 * S.quantile(fin, 0.9, true) + 1e-9) * 1.15;
          pP.setDomain([lo, hi], [0, ymax]).clear();
          pP.area(xs.map((x, i) => [x, Math.min(ys[i], ymax * 2)]), { color: 's1', opacity: 0.15 });
          pP.line(xs.map((x, i) => [x, ys[i]]), { color: 's1', width: 2 });
          pF.setDomain([lo, hi], [0, 1.03]).clear();
          pF.line(xs.map(x => [x, D.F(x, P)]), { color: 's1', width: 2 });
          med = S.invert(x => D.F(x, P), 0.5, lo - 10 * (hi - lo), hi);
          if (finite) within = D.F(mu + 2 * sd, P) - D.F(mu - 2 * sd, P);
          if (ov.get() && finite) {
            pP.line(xs.map(x => [x, S.pdf.normal(x, mu, sd)]), { color: 'ink', dash: true, width: 1.5 });
            pF.line(xs.map(x => [x, S.cdf.normal(x, mu, sd)]), { color: 'ink', dash: true, width: 1.5 });
          }
        }
        if (Number.isFinite(mu)) { pP.vline(mu, { color: 's2' }); if (finite) pP.band(mu - 2 * sd, mu + 2 * sd, { color: 's2', opacity: 0.06 }); }
        pF.hline(0.5, { color: 'faint', width: 1 });
        pP.legend([{ label: D.disc ? 'pmf' : 'pdf', color: 's1' }].concat(ov.get() && finite ? [{ label: 'normal, same μ and σ', color: 'ink', dash: true }] : []));
        sM.set(fmtMom(mu));
        sS.set(Number.isNaN(v) ? 'undefined' : S.fmt(sd));
        sMed.set(S.fmt(med));
        s2.set(finite ? pct(within) : '—', finite ? 'normal: 95.4%' : 'needs finite variance');
      }
      build();
    },
  });

  // Distances between a family and its limit, as a function of the size parameter.
  function kolmDiscrete(pmfAt, lo, hi, m, s) {
    let F = 0, D = 0;
    for (let k = lo; k <= hi; k++) {
      const Phi = S.cdf.normal((k - m) / s), prev = F;
      F += pmfAt(k);
      D = Math.max(D, Math.abs(F - Phi), Math.abs(prev - Phi));
    }
    return D;
  }
  const ZG = S.linspace(-6, 6, 181);
  const REL = {
    binpois: { label: 'Binomial(n, λ/n) → Poisson(λ)', metric: 'total variation', slope: -1, nmin: lam => Math.ceil(lam) + 1,
      dist(n, lam) {
        const p = lam / n, K = Math.ceil(lam + 10 * Math.sqrt(lam) + 10);
        let s = 0, sb = 0, sp = 0;
        for (let k = 0; k <= K; k++) { const b = k <= n ? S.pmf.binom(k, n, p) : 0, q = S.pmf.poisson(k, lam); s += Math.abs(b - q); sb += b; sp += q; }
        return 0.5 * (s + Math.max(0, 1 - sb) + Math.max(0, 1 - sp));
      },
      bound: (n, lam) => ['Le Cam bound $\\lambda^2/n$', lam * lam / n] },
    binnorm: { label: 'Binomial(n, p) → Normal', metric: 'Kolmogorov', slope: -0.5, nmin: () => 2,
      dist(n, _l, p) { const m = n * p, s = Math.sqrt(n * p * (1 - p)); return kolmDiscrete(k => S.pmf.binom(k, n, p), Math.max(0, Math.floor(m - 12 * s)), Math.min(n, Math.ceil(m + 12 * s)), m, s); },
      bound: (n, _l, p) => ['Berry–Esseen bound', 0.4748 * (p * p + (1 - p) * (1 - p)) / Math.sqrt(n * p * (1 - p))] },
    poisnorm: { label: 'Poisson(λ) → Normal, λ = n', metric: 'Kolmogorov', slope: -0.5, nmin: () => 1,
      dist(n) { const s = Math.sqrt(n); return kolmDiscrete(k => S.pmf.poisson(k, n), Math.max(0, Math.floor(n - 12 * s)), Math.ceil(n + 12 * s + 10), n, s); } },
    tnorm: { label: 'Student t(ν) → Normal, ν = n', metric: 'Kolmogorov', slope: -1, nmin: () => 1,
      dist(n) { let D = 0; for (const x of ZG) D = Math.max(D, Math.abs(S.cdf.t(x, n) - S.cdf.normal(x))); return D; } },
    gammanorm: { label: 'Standardized Gamma(α, 1) → Normal, α = n', metric: 'Kolmogorov', slope: -0.5, nmin: () => 1,
      dist(n) { let D = 0; const s = Math.sqrt(n); for (const z of ZG) { const x = n + z * s; D = Math.max(D, Math.abs((x <= 0 ? 0 : S.gammaP(n, x)) - S.cdf.normal(z))); } return D; } },
  };

  Atlas.lab('distribution-families-limits', {
    title: 'Limit theorems between families, with rates',
    blurb: 'Left: a family next to its limit at the current size. Right: the distance on log–log axes as the size grows. The Poisson approximation improves like $1/n$; normal approximations to skewed discrete laws like $1/\\sqrt n$ (Berry–Esseen); $t_\\nu \\to$ normal like $1/\\nu$ because the $t$ is symmetric.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rel = UI.select(controls, { label: 'Relationship', options: Object.keys(REL).map(k => ({ value: k, label: REL[k].label })), value: 'binpois', onChange: () => { curve = null; draw(); } });
      const n = UI.slider(controls, { label: 'Size $n$ (log scale)', min: 1, max: 2000, value: 20, log: true, fmt: v => String(Math.round(v)), onInput: draw });
      const lam = UI.slider(controls, { label: '$\\lambda$ (binomial → Poisson)', min: 0.5, max: 10, step: 0.1, value: 3, onInput: () => { curve = null; draw(); } });
      const pp = UI.slider(controls, { label: '$p$ (binomial → normal)', min: 0.02, max: 0.5, step: 0.01, value: 0.1, onInput: () => { curve = null; draw(); } });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'probability / density', title: 'Family vs limit' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [1, 2000], y: [1e-4, 1], logx: true, logy: true, xlabel: 'n', ylabel: 'distance', title: 'Distance vs n (log–log)' });
      const sD = UI.stat(stats, 'Distance at $n$', { accent: 'accent' });
      const sB = UI.stat(stats, 'Bound');
      const sR = UI.stat(stats, 'Scaled error');
      let curve = null;
      function nNow() { const R = REL[rel.get()]; return Math.max(R.nmin(lam.get()), Math.round(n.get())); }
      function computeCurve() {
        const R = REL[rel.get()], n0 = R.nmin(lam.get());
        const grid = [...new Set(S.linspace(Math.log10(n0), Math.log10(2000), 22).map(v => Math.round(10 ** v)))];
        curve = grid.map(m => [m, R.dist(m, lam.get(), pp.get())]).filter(d => d[1] > 0);
      }
      function draw() {
        const key = rel.get(), R = REL[key], m = nNow(), L = lam.get(), p = pp.get();
        if (!curve) computeCurve();
        // left panel
        pL.clear();
        if (key === 'binpois') {
          const K = Math.ceil(L + 4 * Math.sqrt(L) + 3);
          pL.setDomain([-0.7, K + 0.7], [0, Math.max(S.pmf.poisson(Math.floor(L), L), S.pmf.binom(Math.floor(L), m, L / m)) * 1.25]).clear();
          pL.stems(S.range(K + 1).map(k => [k - 0.15, S.pmf.binom(k, m, L / m)]), { color: 's1', r: 3 });
          pL.stems(S.range(K + 1).map(k => [k + 0.15, S.pmf.poisson(k, L)]), { color: 's3', r: 3 });
          pL.legend([{ label: `Binomial(${m}, ${S.fmt(L / m, 3)})`, color: 's1' }, { label: `Poisson(${S.fmt(L, 1)})`, color: 's3' }]);
        } else if (key === 'binnorm' || key === 'poisnorm') {
          const mean = key === 'binnorm' ? m * p : m, sd = key === 'binnorm' ? Math.sqrt(m * p * (1 - p)) : Math.sqrt(m);
          const lo = Math.max(0, Math.floor(mean - 4.5 * sd) - 1), hi = Math.ceil(mean + 4.5 * sd) + 1;
          const f = k => (key === 'binnorm' ? S.pmf.binom(k, m, p) : S.pmf.poisson(k, m));
          const ks = S.range(hi - lo + 1).map(i => lo + i).filter(k => key !== 'binnorm' || k <= m);
          const ymax = Math.max(...ks.map(f), S.pdf.normal(mean, mean, sd)) * 1.2;
          pL.setDomain([lo - 0.5, hi + 0.5], [0, ymax]).clear();
          pL.stems(ks.map(k => [k, f(k)]), { color: 's1', r: ks.length > 60 ? 1.5 : 3, width: ks.length > 60 ? 1 : 2 });
          pL.line(S.linspace(lo - 0.5, hi + 0.5, 300).map(x => [x, S.pdf.normal(x, mean, sd)]), { color: 'ink', dash: true, width: 1.5 });
          pL.legend([{ label: key === 'binnorm' ? 'binomial pmf' : 'Poisson pmf', color: 's1' }, { label: 'normal density', color: 'ink', dash: true }]);
        } else {
          const xs = S.linspace(-5, 5, 300);
          const fz = key === 'tnorm' ? (z => S.pdf.t(z, m)) : (z => { const x = m + z * Math.sqrt(m); return x <= 0 ? 0 : Math.sqrt(m) * S.pdf.gamma(x, m, 1); });
          const ys = xs.map(fz), ymax = Math.min(1.2, Math.max(0.45, ...ys.filter(Number.isFinite)) * 1.1);
          pL.setDomain([-5, 5], [0, ymax]).clear();
          pL.line(xs.map((x, i) => [x, ys[i]]), { color: 's1', width: 2 });
          pL.line(xs.map(x => [x, S.pdf.normal(x)]), { color: 'ink', dash: true, width: 1.5 });
          pL.legend([{ label: key === 'tnorm' ? `t, ν = ${m}` : `standardized Gamma(${m})`, color: 's1' }, { label: 'N(0,1)', color: 'ink', dash: true }]);
        }
        // right panel
        const dNow = R.dist(m, L, p);
        const vals = curve.map(c => c[1]).concat([dNow]).filter(v => v > 0);
        const ylo = 10 ** Math.floor(Math.log10(Math.min(...vals))), yhi = 10 ** Math.ceil(Math.log10(Math.max(...vals)));
        pR.setDomain([1, 2000], [ylo, yhi]).clear();
        pR.line(curve, { color: 's1', width: 2 });
        curve.forEach(c => pR.dot(c[0], c[1], { color: 's1', r: 2.5 }));
        const last = curve[curve.length - 1];
        const ref = S.linspace(Math.log10(curve[0][0]), Math.log10(2000), 30).map(v => { const x = 10 ** v; return [x, last[1] * (x / last[0]) ** R.slope]; });
        pR.line(ref, { color: 'muted', dash: true, width: 1.2 });
        pR.text(2000, last[1], `slope ${R.slope === -1 ? '−1' : '−1/2'}`, { anchor: 'end', dy: -10, size: 11, color: 'muted' });
        if (dNow > 0) pR.dot(m, dNow, { color: 'accent', r: 6 });
        sD.set(S.fmt(dNow, 4), R.metric + ' distance');
        if (R.bound) { const [lab, b] = R.bound(m, L, p); sB.set(S.fmt(b, 4), lab); } else sB.set('—', '');
        sR.set(S.fmt(dNow * m ** (-R.slope), 3), R.slope === -1 ? 'n × distance (levels off)' : '√n × distance (levels off)');
      }
      draw();
    },
  });

  /* ================= transformations ================= */

  const XDIST = {
    normal: { label: 'Normal(0,1)', pdf: x => S.pdf.normal(x), draw: r => r.normal(), range: [-3.5, 3.5] },
    unif: { label: 'Uniform(0,1)', pdf: x => (x > 0 && x < 1 ? 1 : 0), draw: r => r.u(), range: [-0.1, 1.1] },
    unifpm: { label: 'Uniform(−1,1)', pdf: x => (x > -1 && x < 1 ? 0.5 : 0), draw: r => r.uniform(-1, 1), range: [-1.2, 1.2] },
    exp: { label: 'Exponential(1)', pdf: x => S.pdf.exp(x), draw: r => r.exp(1), range: [-0.2, 5] },
  };
  // Each transform lists its inverse branches at y: [x_j(y), |dx_j/dy|].
  const TRANS = {
    sq: { label: 'Y = X²  (two branches)', g: x => x * x, br: y => (y > 0 ? [[Math.sqrt(y), 0.5 / Math.sqrt(y)], [-Math.sqrt(y), 0.5 / Math.sqrt(y)]] : []) },
    exp: { label: 'Y = exp(X)', g: Math.exp, br: y => (y > 0 ? [[Math.log(y), 1 / y]] : []) },
    log: { label: 'Y = log X  (needs X > 0)', g: x => (x > 0 ? Math.log(x) : NaN), br: y => [[Math.exp(y), Math.exp(y)]] },
    inv: { label: 'Y = 1/X', g: x => 1 / x, br: y => (y !== 0 ? [[1 / y, 1 / (y * y)]] : []) },
    cube: { label: 'Y = X³', g: x => x ** 3, br: y => { const c = Math.cbrt(y); return y !== 0 ? [[c, 1 / (3 * c * c)]] : []; } },
    sig: { label: 'Y = 1/(1 + e^(−X))', g: x => S.logistic(x), br: y => (y > 0 && y < 1 ? [[S.logit(y), 1 / (y * (1 - y))]] : []) },
    tan: { label: 'Y = tan(π(X − ½))', g: x => Math.tan(Math.PI * (x - 0.5)), br: y => { const b = 0.5 + Math.atan(y) / Math.PI, d = 1 / (Math.PI * (1 + y * y)), out = []; for (let j = -8; j <= 8; j++) out.push([b + j, d]); return out; } },
  };

  Atlas.lab('transformations-jacobian', {
    title: 'Change of variables: where the Jacobian comes from',
    blurb: 'Samples of $X$ are pushed through $g$. Left: the map, with a small interval $[x, x+\\Delta]$ and its image — the stretch factor $|g\'(x)|$. Right: the histogram of $Y$ against $f_Y(y) = \\sum_j f_X(x_j)\\,|dx_j/dy|$. Toggle the naive formula without the Jacobian to see it fail.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const xd = UI.select(controls, { label: 'Distribution of $X$', options: Object.keys(XDIST).map(k => ({ value: k, label: XDIST[k].label })), value: 'normal', onChange: resample });
      const tr = UI.select(controls, { label: 'Transformation $g$', options: Object.keys(TRANS).map(k => ({ value: k, label: TRANS[k].label })), value: 'sq', onChange: draw });
      const pos = UI.slider(controls, { label: 'Probe position $x$', min: 0.02, max: 0.95, step: 0.005, value: 0.6, fmt: v => S.fmt(lerpX(v), 2), onInput: draw });
      const naive = UI.toggle(controls, { label: 'Show naive $f_X(g^{-1}(y))$ (no Jacobian)', value: false, onChange: draw });
      UI.button(controls, 'Resample', resample, { primary: true });
      stage.classList.add('two');
      const pG = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'y = g(x)', title: 'The map g' });
      const pY = new Plot(stage, { width: 420, height: 280, xlabel: 'y', ylabel: 'density', title: 'Distribution of Y = g(X)' });
      const sSt = UI.stat(stats, 'Stretch $|\\Delta y / \\Delta x|$ at probe');
      const sI = UI.stat(stats, '$\\int f_Y$ over window', { accent: 'accent' });
      const sF = UI.stat(stats, 'Sample fraction in window');
      const sN = UI.stat(stats, '$\\int$ naive over window');
      let X = [];
      function lerpX(v) { const r = XDIST[xd.get ? xd.get() : 'normal'].range; return r[0] + v * (r[1] - r[0]); }
      function resample() { const D = XDIST[xd.get()]; X = ctx.rng.array(20000, () => D.draw(ctx.rng)); draw(); }
      function draw() {
        const D = XDIST[xd.get()], T = TRANS[tr.get()];
        const Y = X.map(T.g).filter(Number.isFinite).sort((a, b) => a - b);
        const dropped = X.length - Y.length;
        let ylo = S.quantile(Y, 0.01, true), yhi = S.quantile(Y, 0.99, true);
        const pad = (yhi - ylo) * 0.06 || 0.5; ylo -= pad; yhi += pad;
        if (Y[0] >= 0 && ylo < 0) ylo = -pad * 0.5;
        const fY = y => T.br(y).reduce((s, [x, d]) => s + D.pdf(x) * d, 0);
        const fN = y => T.br(y).reduce((s, [x]) => s + D.pdf(x), 0);
        // left: the map
        const [x0r, x1r] = D.range;
        pG.setDomain([x0r, x1r], [ylo, yhi]).clear();
        const gx = S.linspace(x0r, x1r, 500), pts = [];
        let prev = null;
        for (const x of gx) { const y = T.g(x); if (prev != null && Number.isFinite(y) && Number.isFinite(prev) && Math.abs(y - prev) > 5 * (yhi - ylo)) pts.push(null); pts.push([x, y]); prev = y; }
        const xp = x0r + pos.get() * (x1r - x0r), dx = (x1r - x0r) / 40;
        const ya = T.g(xp), yb = T.g(xp + dx);
        pG.band(xp, xp + dx, { color: 's2', opacity: 0.18 });
        if (Number.isFinite(ya) && Number.isFinite(yb)) pG.rect(x0r, Math.min(ya, yb), x1r, Math.max(ya, yb), { color: 's2', opacity: 0.18 });
        pG.line(pts, { color: 'ink', width: 2 });
        const dens = S.linspace(x0r, x1r, 200).map(x => [x, ylo + (yhi - ylo) * 0.25 * D.pdf(x) / Math.max(...S.linspace(x0r, x1r, 60).map(D.pdf))]);
        pG.area(dens, { color: 's1', opacity: 0.15, base: ylo });
        pG.text(x0r, ylo, 'f_X (scaled)', { dx: 6, dy: -6, size: 11, color: 's1' });
        // right: histogram and densities
        const h = S.histogram(Y, { bins: 60, lo: ylo, hi: yhi });
        for (const b of h) b.density = b.count / (X.length * (b.x1 - b.x0));
        const ys = S.linspace(ylo, yhi, 500), fv = ys.map(fY);
        const hmax = Math.max(...h.map(b => b.density));
        const ymax = Math.max(hmax, 0.05) * 1.3;
        pY.setDomain([ylo, yhi], [0, ymax]).clear();
        pY.bars(h, { color: 's1', opacity: 0.35 });
        pY.line(ys.map((y, i) => [y, fv[i]]), { color: 'ink', width: 2 });
        if (naive.get()) pY.line(ys.map(y => [y, fN(y)]), { color: 'bad', width: 2, dash: true });
        if (Number.isFinite(ya) && Number.isFinite(yb)) pY.band(Math.min(ya, yb), Math.max(ya, yb), { color: 's2', opacity: 0.18 });
        pY.legend([{ label: 'f_Y with Jacobian', color: 'ink' }].concat(naive.get() ? [{ label: 'naive (wrong)', color: 'bad', dash: true }] : []).concat([{ label: 'samples', color: 's1', fill: true }]));
        // numerics (midpoint rule on 2000 cells)
        const M = 2000, w = (yhi - ylo) / M;
        let I = 0, IN = 0;
        for (let i = 0; i < M; i++) { const y = ylo + (i + 0.5) * w, a = fY(y), b = fN(y); if (Number.isFinite(a)) I += a * w; if (Number.isFinite(b)) IN += b * w; }
        const inWin = Y.filter(y => y >= ylo && y <= yhi).length / X.length;
        const st = Number.isFinite(ya) && Number.isFinite(yb) ? Math.abs(yb - ya) / dx : NaN;
        sSt.set(S.fmt(st), 'density at the image is divided by this');
        sI.set(S.fmt(I, 4));
        sF.set(S.fmt(inWin, 4), dropped ? `${dropped} draws had g undefined` : '');
        sN.set(S.fmt(IN, 4), naive.get() ? 'should match the left tile if it were a density' : 'toggle to plot it');
      }
      resample();
    },
  });

  const F2 = {
    U: { label: 'Uniform(0,1)', pdf: x => (x >= 0 && x <= 1 ? 1 : 0), cdf: x => S.cdf.uniform(x), draw: r => r.u(), sup: [0, 1] },
    E: { label: 'Exponential(1)', pdf: x => S.pdf.exp(x), cdf: x => S.cdf.exp(x), draw: r => r.exp(1), sup: [0, 30] },
    N: { label: 'Normal(0,1)', pdf: x => S.pdf.normal(x), cdf: x => S.cdf.normal(x), draw: r => r.normal(), sup: [-8, 8] },
  };
  const mid = (fn, [a, b], m = 800) => { const h = (b - a) / m; let s = 0; for (let i = 0; i < m; i++) s += fn(a + (i + 0.5) * h); return s * h; };
  const OPS = {
    sum: { label: 'X + Y', z: (x, y) => x + y, f: (X, Y) => z => mid(x => X.pdf(x) * Y.pdf(z - x), X.sup), formula: '$\\int f_X(x) f_Y(z-x)\\,dx$' },
    diff: { label: 'X − Y', z: (x, y) => x - y, f: (X, Y) => z => mid(x => X.pdf(x) * Y.pdf(x - z), X.sup), formula: '$\\int f_X(x) f_Y(x-z)\\,dx$' },
    prod: { label: 'X · Y', z: (x, y) => x * y, f: (X, Y) => z => mid(x => X.pdf(x) * Y.pdf(z / x) / Math.abs(x), X.sup), formula: '$\\int f_X(x) f_Y(z/x)\\,|x|^{-1}dx$' },
    ratio: { label: 'X / Y', z: (x, y) => x / y, f: (X, Y) => z => mid(y => Math.abs(y) * X.pdf(z * y) * Y.pdf(y), Y.sup), formula: '$\\int |y|\\, f_X(zy) f_Y(y)\\,dy$' },
    max: { label: 'max(X, Y)', z: Math.max, f: (X, Y) => z => X.pdf(z) * Y.cdf(z) + X.cdf(z) * Y.pdf(z), formula: '$f_X F_Y + F_X f_Y$' },
    min: { label: 'min(X, Y)', z: Math.min, f: (X, Y) => z => X.pdf(z) * (1 - Y.cdf(z)) + (1 - X.cdf(z)) * Y.pdf(z), formula: '$f_X(1-F_Y) + (1-F_X) f_Y$' },
  };
  const NAMED = { 'sum:UU': 'triangular on [0, 2]', 'sum:EE': 'Gamma(2, 1)', 'sum:NN': 'N(0, 2)', 'diff:NN': 'N(0, 2)', 'diff:EE': 'Laplace(0, 1)', 'diff:UU': 'triangular on [−1, 1]',
    'ratio:NN': 'standard Cauchy — no mean!', 'max:UU': 'Beta(2, 1)', 'min:UU': 'Beta(1, 2)', 'min:EE': 'Exponential(2)', 'prod:UU': 'density −log z on (0, 1)', 'ratio:EE': 'F-like: density 1/(1+z)², no mean' };

  Atlas.lab('transformations-algebra', {
    title: 'Sums, products, ratios and extremes of independent variables',
    blurb: 'Pick two independent families and an operation. The curve is the exact density from the change-of-variables integral, evaluated numerically; the bars are 20,000 simulated values. Try Normal / Normal, or Exponential − Exponential.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fx = UI.select(controls, { label: 'Distribution of $X$', options: Object.keys(F2).map(k => ({ value: k, label: F2[k].label })), value: 'N', onChange: resample });
      const fy = UI.select(controls, { label: 'Distribution of $Y$ (independent)', options: Object.keys(F2).map(k => ({ value: k, label: F2[k].label })), value: 'N', onChange: resample });
      const op = UI.segmented(controls, { label: 'Combination $Z$', options: Object.keys(OPS).map(k => ({ value: k, label: OPS[k].label })), value: 'ratio', onChange: draw });
      UI.button(controls, 'Resample', resample, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, xlabel: 'z', ylabel: 'density' });
      const sF = UI.note(stats, '');
      const sI = UI.stat(stats, '$\\int f_Z$ over window', { accent: 'accent' });
      const sW = UI.stat(stats, 'Sample fraction in window');
      const sM = UI.stat(stats, 'Sample mean of $Z$');
      let xs = [], ys = [];
      function resample() { const X = F2[fx.get()], Y = F2[fy.get()]; xs = ctx.rng.array(20000, () => X.draw(ctx.rng)); ys = ctx.rng.array(20000, () => Y.draw(ctx.rng)); draw(); }
      function draw() {
        const X = F2[fx.get()], Y = F2[fy.get()], O = OPS[op.get()];
        const Z = xs.map((x, i) => O.z(x, ys[i])).filter(Number.isFinite);
        const sz = Z.slice().sort((a, b) => a - b);
        let lo = S.quantile(sz, 0.005, true), hi = S.quantile(sz, 0.995, true);
        const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
        const f = O.f(X, Y);
        const grid = S.linspace(lo, hi, 220), fv = grid.map(f);
        const h = S.histogram(Z, { bins: 70, lo, hi });
        const ymax = Math.max(...h.map(b => b.density), ...fv.filter(Number.isFinite).sort((a, b) => a - b).slice(0, -3)) * 1.15;
        p.setDomain([lo, hi], [0, ymax]).clear();
        p.bars(h, { color: 's2', opacity: 0.4 });
        p.line(grid.map((z, i) => [z, fv[i]]), { color: 'ink', width: 2 });
        p.legend([{ label: 'exact density (numerical integral)', color: 'ink' }, { label: 'simulated', color: 's2', fill: true }]);
        const w = grid[1] - grid[0];
        let I = 0; for (let i = 0; i < grid.length - 1; i++) { const a = fv[i], b = fv[i + 1]; if (Number.isFinite(a) && Number.isFinite(b)) I += 0.5 * (a + b) * w; }
        const name = NAMED[`${op.get()}:${fx.get()}${fy.get()}`];
        sF.set(`Formula: ${O.formula}${name ? `<br>Known result: <strong>${name}</strong>` : ''}`);
        sI.set(S.fmt(I, 3));
        sW.set(S.fmt(Z.filter(z => z >= lo && z <= hi).length / Z.length, 3));
        sM.set(S.fmt(S.mean(Z)), op.get() === 'ratio' ? 'resample a few times: is it stable?' : '');
      }
      resample();
    },
  });

  /* ================= expectation ================= */

  const HT = {
    pareto: { label: 'Pareto(α), x_m = 1', draw: (r, a) => r.pareto(a, 1), mean: a => (a > 1 ? a / (a - 1) : Infinity), var: a => (a > 2 ? a / ((a - 1) ** 2 * (a - 2)) : Infinity) },
    t: { label: 'Student t, ν = α (α = 1: Cauchy)', draw: (r, a) => r.t(a), mean: a => (a > 1 ? 0 : NaN), var: a => (a > 2 ? a / (a - 2) : a > 1 ? Infinity : NaN) },
    normal: { label: 'Normal(0,1) (α ignored)', draw: r => r.normal(), mean: () => 0, var: () => 1 },
    lognormal: { label: 'Lognormal(0, 1.5²) (α ignored)', draw: r => r.lognormal(0, 1.5), mean: () => Math.exp(1.125), var: () => (Math.exp(2.25) - 1) * Math.exp(2.25) },
  };

  Atlas.lab('expectation-heavy-tails', {
    title: 'When the mean does not exist: running averages under heavy tails',
    blurb: 'Six independent running means $\\bar X_n$ (left) and the share of $\\sum|X_i|$ contributed by the single largest observation (right). Lower the tail index $\\alpha$ below 2, then below 1: convergence slows, then stops, and one observation comes to dominate the sum.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dist = UI.select(controls, { label: 'Distribution', options: Object.keys(HT).map(k => ({ value: k, label: HT[k].label })), value: 'pareto', onChange: reset });
      const al = UI.slider(controls, { label: 'Tail index $\\alpha$', min: 0.6, max: 4, step: 0.05, value: 1.5, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Continue to n = 20,000', '❚❚ Pause']);
      UI.button(row, 'New paths', reset);
      stage.classList.add('two');
      const NMAX = 20000, KP = 6, COLS = ['s1', 's2', 's3', 's4', 's5', 's6'];
      const pM = new Plot(stage, { width: 420, height: 280, x: [1, NMAX], y: [0, 1], logx: true, xlabel: 'n', ylabel: 'running mean', title: 'Running means' });
      const pS = new Plot(stage, { width: 420, height: 280, x: [1, NMAX], y: [0, 1], logx: true, xlabel: 'n', ylabel: 'max|X| / Σ|X|', title: 'Share of the largest observation' });
      const sMu = UI.stat(stats, 'Population mean');
      const sV = UI.stat(stats, 'Population variance');
      const sSp = UI.stat(stats, 'Spread of the 6 current means', { accent: 'accent' });
      const sSh = UI.stat(stats, 'Median largest-share');
      const checks = new Set(S.linspace(0, Math.log10(NMAX), 220).map(v => Math.round(10 ** v)));
      let P;
      function reset() {
        P = S.range(KP).map(() => ({ n: 0, sum: 0, sabs: 0, mx: 0, rec: [] }));
        loop.stop(); advance(1500); draw();
      }
      function advance(k) {
        const D = HT[dist.get()], a = al.get();
        for (const q of P) {
          const stop = Math.min(NMAX, q.n + k);
          while (q.n < stop) {
            const x = D.draw(ctx.rng, a);
            q.n++; q.sum += x; q.sabs += Math.abs(x); if (Math.abs(x) > q.mx) q.mx = Math.abs(x);
            if (checks.has(q.n)) q.rec.push([q.n, q.sum / q.n, q.mx / q.sabs]);
          }
        }
      }
      function step() { advance(300); draw(); if (P[0].n >= NMAX) return false; }
      function draw() {
        const D = HT[dist.get()], a = al.get(), mu = D.mean(a), v = D.var(a);
        const vals = [];
        for (const q of P) for (const r of q.rec) if (r[0] >= 30) vals.push(r[1]);
        if (Number.isFinite(mu)) vals.push(mu);
        let [lo, hi] = vals.length ? S.extent(vals) : [-1, 1];
        if (!(hi > lo)) { lo -= 1; hi += 1; }
        const pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
        pM.setDomain([1, NMAX], [lo, hi]).clear();
        if (Number.isFinite(mu)) pM.hline(mu, { color: 'ink', width: 1.5 });
        P.forEach((q, i) => pM.line(q.rec.map(r => [r[0], r[1]]), { color: COLS[i], width: 1.6 }));
        pS.clear();
        P.forEach((q, i) => pS.line(q.rec.map(r => [r[0], r[2]]), { color: COLS[i], width: 1.4, opacity: 0.85 }));
        const cur = P.map(q => q.sum / q.n);
        sMu.set(Number.isNaN(mu) ? 'undefined' : S.fmt(mu), Number.isFinite(mu) ? 'black line' : 'no finite mean');
        sV.set(Number.isNaN(v) ? 'undefined' : S.fmt(v));
        sSp.set(S.fmt(S.sd(cur)), `at n = ${P[0].n}`);
        sSh.set(S.fmt(S.median(P.map(q => q.mx / q.sabs)), 3), 'tends to 0 iff the mean is finite');
      }
      reset();
    },
  });

  const PHI = {
    sq: { label: 'φ(x) = x²', f: x => x * x, d2: () => 2 },
    exp: { label: 'φ(x) = eˣ', f: Math.exp, d2: Math.exp },
    nlog: { label: 'φ(x) = −log x', f: x => -Math.log(x), d2: x => 1 / (x * x), pos: true },
    sqrt: { label: 'φ(x) = √x (concave)', f: Math.sqrt, d2: x => -0.25 * x ** -1.5, pos: true },
    abs: { label: 'φ(x) = |x − 1| (kink)', f: x => Math.abs(x - 1), d2: x => (Math.abs(x - 1) < 1e-9 ? NaN : 0) },
  };

  Atlas.lab('expectation-jensen', {
    title: "Jensen's inequality: curvature times spread",
    blurb: 'Points $(X_i, \\varphi(X_i))$ lie on the curve; their centre of mass $(\\E X, \\E\\varphi(X))$ lies above a convex curve and below a concave one. The gap is approximately $\\tfrac12\\varphi\'\'(\\mu)\\Var X$ — except at a kink, or when $\\E\\varphi(X)$ is infinite.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ph = UI.select(controls, { label: 'Function $\\varphi$', options: Object.keys(PHI).map(k => ({ value: k, label: PHI[k].label })), value: 'exp', onChange: draw });
      const xd = UI.segmented(controls, { label: 'Distribution of $X$', options: [{ value: 'two', label: 'Two-point μ ± s' }, { value: 'unif', label: 'Uniform(μ−s, μ+s)' }, { value: 'sexp', label: 'μ − s + s·Exp(1)' }], value: 'two', onChange: draw });
      const mu = UI.slider(controls, { label: 'Mean $\\mu$', min: 0.2, max: 3, step: 0.01, value: 1, onInput: draw });
      const sd = UI.slider(controls, { label: 'Spread $s$ (= SD for two-point and shifted exp.)', min: 0, max: 2, step: 0.01, value: 0.6, onInput: draw });
      const tang = UI.toggle(controls, { label: 'Show supporting line at $\\mu$', value: true, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 320, xlabel: 'x', ylabel: 'φ(x)' });
      const sA = UI.stat(stats, '$\\varphi(\\E X)$');
      const sB = UI.stat(stats, '$\\E\\varphi(X)$', { accent: 'accent' });
      const sG = UI.stat(stats, 'Jensen gap');
      const sQ = UI.stat(stats, '$\\tfrac12\\varphi\'\'(\\mu)\\Var X$');
      const base = ctx.rng.array(80, () => [ctx.rng.u(), ctx.rng.exp(1)]);
      function draw() {
        const F = PHI[ph.get()], m = mu.get(), kind = xd.get();
        let s = sd.get(), clamped = false;
        if (F.pos && s > m - 0.02) { s = Math.max(0, m - 0.02); clamped = true; }
        const lo = m - s, hi = kind === 'sexp' ? m + 3 * s : m + s;
        let Ef, varX;
        if (kind === 'two') { Ef = 0.5 * (F.f(lo) + F.f(m + s)); varX = s * s; }
        else if (kind === 'unif') { Ef = s > 0 ? mid(F.f, [lo, hi], 400) / (hi - lo) : F.f(m); varX = s * s / 3; }
        else {
          varX = s * s;
          if (ph.get() === 'exp') Ef = s >= 1 ? Infinity : Math.exp(m - s) / (1 - s);
          else Ef = s > 0 ? mid(t => F.f(m - s + s * t) * Math.exp(-t), [0, 50], 3000) : F.f(m);
        }
        const xl = F.pos ? Math.max(0.01, lo - 0.4) : lo - 0.6, xh = hi + 0.6;
        const xs = S.linspace(xl, xh, 300), fs = xs.map(F.f);
        const fin = fs.concat(Number.isFinite(Ef) ? [Ef] : []);
        let ylo = Math.min(...fin), yhi = Math.max(...fin);
        if (F.pos && ph.get() === 'nlog') yhi = Math.min(yhi, Math.max(F.f(lo), Ef) + 1);
        const pad = (yhi - ylo) * 0.12 || 0.5; ylo -= pad; yhi += pad;
        p.setDomain([xl, xh], [ylo, yhi]).clear();
        p.line(xs.map((x, i) => [x, fs[i]]), { color: 'ink', width: 2.2 });
        const pts = kind === 'two' ? [[lo, F.f(lo)], [m + s, F.f(m + s)]] : base.map(([u, e]) => { const x = kind === 'unif' ? lo + u * (hi - lo) : m - s + s * e; return [x, F.f(x)]; });
        if (kind === 'two') p.segment(lo, F.f(lo), m + s, F.f(m + s), { color: 's1', width: 1.5, dash: true });
        p.points(pts, { color: 's1', r: kind === 'two' ? 5 : 3, opacity: 0.55 });
        if (tang.get()) {
          const h = 1e-4, d1 = (F.f(m + h) - F.f(m - h)) / (2 * h);
          if (Number.isFinite(d1)) p.line([[xl, F.f(m) + d1 * (xl - m)], [xh, F.f(m) + d1 * (xh - m)]], { color: 'muted', width: 1.2, dash: '6 4' });
        }
        p.vline(m, { color: 'faint' });
        const top = Number.isFinite(Ef) ? Ef : yhi;
        p.segment(m, F.f(m), m, top, { color: 'accent', width: 3 });
        p.dot(m, F.f(m), { color: 's2', r: 6 });
        if (Number.isFinite(Ef)) p.dot(m, Ef, { color: 's1', r: 6 });
        else p.text(m, yhi, '  E φ(X) = ∞', { dy: 14, size: 12, color: 'bad', weight: 600 });
        p.legend([{ label: 'φ', color: 'ink' }, { label: '(EX, Eφ(X))', color: 's1', fill: true }, { label: '(EX, φ(EX))', color: 's2', fill: true }], { left: true });
        const gap = Ef - F.f(m), approx = 0.5 * F.d2(m) * varX;
        sA.set(S.fmt(F.f(m)));
        sB.set(S.fmt(Ef), clamped ? 's reduced to keep X > 0' : '');
        sG.set(S.fmt(gap), gap >= -1e-12 ? 'φ(EX) ≤ Eφ(X)' : 'concave: inequality reversed');
        sQ.set(Number.isNaN(approx) ? 'undefined (kink at μ)' : S.fmt(approx), 'second-order (delta-method) approximation');
      }
      draw();
    },
  });

  /* ================= mgf ================= */

  const MG = {
    normal: { label: 'Normal(0,1)', M: t => Math.exp(t * t / 2), edge: Infinity, mean: 0, m2: 1, draw: r => r.normal() },
    exp: { label: 'Exponential(1)', M: t => (t < 1 ? 1 / (1 - t) : Infinity), edge: 1, mean: 1, m2: 2, draw: r => r.exp(1) },
    gamma: { label: 'Gamma(3, 1)', M: t => (t < 1 ? (1 - t) ** -3 : Infinity), edge: 1, mean: 3, m2: 12, draw: r => r.gamma(3) },
    pois: { label: 'Poisson(3)', M: t => Math.exp(3 * (Math.exp(t) - 1)), edge: Infinity, mean: 3, m2: 12, draw: r => r.poisson(3) },
    bern: { label: 'Bernoulli(0.3)', M: t => 0.7 + 0.3 * Math.exp(t), edge: Infinity, mean: 0.3, m2: 0.3, draw: r => r.bernoulli(0.3) },
    unif: { label: 'Uniform(0,1)', M: t => (Math.abs(t) < 1e-9 ? 1 : (Math.exp(t) - 1) / t), edge: Infinity, mean: 0.5, m2: 1 / 3, draw: r => r.u() },
    lognorm: { label: 'Lognormal(0,1): no MGF', M: t => (t > 0 ? Infinity : t === 0 ? 1 : mid(z => Math.exp(t * Math.exp(z)) * S.pdf.normal(z), [-8, 8], 400)), edge: 0, mean: Math.exp(0.5), m2: Math.exp(2), draw: r => r.lognormal(0, 1) },
  };

  Atlas.lab('mgf-explorer', {
    title: 'MGF and cumulant generating function, true and empirical',
    blurb: 'Left: $M(t)$ with its tangent $1+\\mu t$ and osculating parabola $1+\\mu t+\\E X^2 t^2/2$ at 0; right: $K(t)=\\log M(t)$ and $\\mu t + \\sigma^2 t^2/2$. The thin curve is the empirical MGF $\\frac1n\\sum e^{tX_i}$. Move $t_0$ outward: the empirical MGF becomes the property of the largest few observations.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dist = UI.select(controls, { label: 'Distribution', options: Object.keys(MG).map(k => ({ value: k, label: MG[k].label })), value: 'exp', onChange: resample });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 5000, step: 10, value: 200, onChange: resample });
      const t0 = UI.slider(controls, { label: 'Probe $t_0$', min: -2, max: 2, step: 0.01, value: 0.4, onInput: draw });
      UI.button(controls, 'Resample', resample, { primary: true });
      stage.classList.add('two');
      const pM = new Plot(stage, { width: 420, height: 280, x: [-2, 2], y: [0, 8], xlabel: 't', ylabel: 'M(t)', title: 'Moment generating function' });
      const pK = new Plot(stage, { width: 420, height: 280, x: [-2, 2], y: [-3, 6], xlabel: 't', ylabel: 'K(t) = log M(t)', title: 'Cumulant generating function' });
      const sMo = UI.stat(stats, '$\\mu = M\'(0)$, $\\sigma^2 = K\'\'(0)$');
      const sTr = UI.stat(stats, '$M(t_0)$ true');
      const sEm = UI.stat(stats, '$\\hat M(t_0)$ empirical', { accent: 'accent' });
      const sSh = UI.stat(stats, 'Share of largest term in $\\hat M(t_0)$');
      let X = [];
      function resample() { const D = MG[dist.get()]; X = ctx.rng.array(Math.round(n.get()), () => D.draw(ctx.rng)); draw(); }
      const empM = t => { let s = 0; for (const x of X) s += Math.exp(t * x); return s / X.length; };
      function draw() {
        const D = MG[dist.get()], T0 = t0.get();
        const ts = S.linspace(-2, 2, 161);
        const Mv = ts.map(D.M), Me = ts.map(empM);
        const varX = D.m2 - D.mean ** 2;
        pM.clear();
        if (D.edge < 2) { pM.band(D.edge, 2, { color: 'bad', opacity: 0.08 }); pM.text(Math.max(D.edge, -2) + 0.05, 7.5, 'M(t) = ∞', { size: 11, color: 'bad' }); }
        pM.line(ts.map(t => [t, 1 + D.mean * t]), { color: 'faint', width: 1.2 });
        pM.line(ts.map(t => [t, 1 + D.mean * t + D.m2 * t * t / 2]), { color: 's3', dash: true, width: 1.5 });
        pM.line(ts.map((t, i) => [t, Number.isFinite(Mv[i]) ? Mv[i] : NaN]), { color: 'ink', width: 2.2 });
        pM.line(ts.map((t, i) => [t, Me[i]]), { color: 's1', width: 1.4 });
        pM.vline(T0, { color: 'accent' });
        pM.legend([{ label: 'M(t)', color: 'ink' }, { label: 'empirical', color: 's1' }, { label: '2nd-order Taylor', color: 's3', dash: true }], { left: true });
        pK.clear();
        if (D.edge < 2) pK.band(D.edge, 2, { color: 'bad', opacity: 0.08 });
        pK.line(ts.map(t => [t, D.mean * t + varX * t * t / 2]), { color: 's3', dash: true, width: 1.5 });
        pK.line(ts.map((t, i) => [t, Number.isFinite(Mv[i]) ? Math.log(Mv[i]) : NaN]), { color: 'ink', width: 2.2 });
        pK.line(ts.map((t, i) => [t, Math.log(Me[i])]), { color: 's1', width: 1.4 });
        pK.vline(T0, { color: 'accent' });
        pK.legend([{ label: 'K(t)', color: 'ink' }, { label: 'μt + σ²t²/2', color: 's3', dash: true }], { left: true });
        const lw = X.map(x => T0 * x), lse = S.logsumexp(lw), share = Math.exp(Math.max(...lw) - lse);
        sMo.set(`${S.fmt(D.mean)}, ${S.fmt(varX)}`);
        const Mt = D.M(T0);
        sTr.set(S.fmt(Mt), Number.isFinite(Mt) ? '' : 'beyond the domain: the expectation is infinite');
        sEm.set(S.fmt(empM(T0)), 'always finite for a finite sample');
        sSh.set(pct(share), 'one observation out of ' + X.length);
      }
      resample();
    },
  });

  const CFD = {
    unif: { label: 'Uniform, standardized', phi: t => { const a = Math.sqrt(3) * t; return [Math.abs(a) < 1e-12 ? 1 : Math.sin(a) / a, 0]; }, draw: r => r.uniform(-Math.sqrt(3), Math.sqrt(3)) },
    exp: { label: 'Exponential(1) − 1 (skewed)', phi: t => { const d = 1 + t * t; return [(Math.cos(t) + t * Math.sin(t)) / d, (t * Math.cos(t) - Math.sin(t)) / d]; }, draw: r => r.exp(1) - 1 },
    bern: { label: 'Bernoulli(0.1), standardized', phi: t => {
      const p = 0.1, s = Math.sqrt(p * (1 - p)), a = 1 - p + p * Math.cos(t / s), b = p * Math.sin(t / s), c = Math.cos(t * p / s), sn = Math.sin(t * p / s);
      return [a * c + b * sn, b * c - a * sn]; }, draw: r => (r.bernoulli(0.1) - 0.1) / Math.sqrt(0.09) },
    rad: { label: 'Rademacher ±1 (lattice)', phi: t => [Math.cos(t), 0], draw: r => (r.u() < 0.5 ? -1 : 1) },
    cauchy: { label: 'Cauchy (no mean, no variance)', phi: t => [Math.exp(-Math.abs(t)), 0], draw: r => r.cauchy() },
  };
  function cpow([re, im], n) { const r = Math.hypot(re, im), th = Math.atan2(im, re), R = r ** n; return [R * Math.cos(n * th), R * Math.sin(n * th)]; }

  Atlas.lab('mgf-cf-clt', {
    title: 'The CLT through characteristic functions',
    blurb: 'The CF of $S_n/c_n$ is $\\varphi(t/c_n)^n$. With $c_n=\\sqrt n$ and finite variance it converges to $e^{-t^2/2}$ (dashed). With $c_n = n$ it converges to 1: the law of large numbers. The Cauchy breaks both: $S_n/n$ is exactly Cauchy for every $n$, and under $\\sqrt n$ scaling the limit is discontinuous at 0.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dist = UI.select(controls, { label: 'Summand distribution', options: Object.keys(CFD).map(k => ({ value: k, label: CFD[k].label })), value: 'exp', onChange: () => { simulate(); } });
      const n = UI.slider(controls, { label: 'Number of summands $n$', min: 1, max: 200, value: 1, log: true, fmt: v => String(Math.round(v)), onInput: drawCF, onChange: simulate });
      const norm = UI.segmented(controls, { label: 'Scaling $c_n$', options: [{ value: 'sqrt', label: '√n (CLT)' }, { value: 'n', label: 'n (LLN)' }], value: 'sqrt', onChange: simulate });
      stage.classList.add('two');
      const pC = new Plot(stage, { width: 420, height: 280, x: [-8, 8], y: [-1.05, 1.05], xlabel: 't', ylabel: 'φ(t)', title: 'CF of the scaled sum' });
      const pH = new Plot(stage, { width: 420, height: 280, x: [-5, 5], y: [0, 0.6], xlabel: 'S_n / c_n', ylabel: 'density', title: '3000 simulated scaled sums' });
      const sSup = UI.stat(stats, '$\\sup_t |\\varphi_n(t) - \\text{limit}|$', { accent: 'accent' });
      const sIn = UI.stat(stats, 'Fraction of sums in $[-5, 5]$');
      const sL = UI.stat(stats, 'Limit');
      function limit(t) {
        const cau = dist.get() === 'cauchy', sq = norm.get() === 'sqrt';
        if (cau) return sq ? (t === 0 ? 1 : 0) : Math.exp(-Math.abs(t));
        return sq ? Math.exp(-t * t / 2) : 1;
      }
      function drawCF() {
        const D = CFD[dist.get()], m = Math.round(n.get()), c = norm.get() === 'sqrt' ? Math.sqrt(m) : m;
        const ts = S.linspace(-8, 8, 481);
        const ph = ts.map(t => cpow(D.phi(t / c), m));
        pC.clear();
        pC.hline(0, { color: 'faint', dash: false, width: 1 });
        pC.line(ts.map(t => [t, limit(t)]), { color: 'ink', dash: true, width: 1.6 });
        pC.line(ts.map((t, i) => [t, ph[i][0]]), { color: 's1', width: 2 });
        pC.line(ts.map((t, i) => [t, ph[i][1]]), { color: 's2', width: 1.6 });
        pC.legend([{ label: 'Re φ_n', color: 's1' }, { label: 'Im φ_n', color: 's2' }, { label: 'limit', color: 'ink', dash: true }]);
        let sup = 0;
        ts.forEach((t, i) => { if (t !== 0) sup = Math.max(sup, Math.hypot(ph[i][0] - limit(t), ph[i][1])); });
        sSup.set(S.fmt(sup, 4), `n = ${m}`);
        const cau = dist.get() === 'cauchy', sq = norm.get() === 'sqrt';
        sL.set(cau ? (sq ? 'none (not continuous at 0)' : 'Cauchy, for every n') : (sq ? 'N(0, 1)' : 'point mass at 0'));
      }
      function simulate() {
        const D = CFD[dist.get()], m = Math.round(n.get()), c = norm.get() === 'sqrt' ? Math.sqrt(m) : m;
        const Z = new Array(3000);
        for (let r = 0; r < 3000; r++) { let s = 0; for (let i = 0; i < m; i++) s += D.draw(ctx.rng); Z[r] = s / c; }
        const h = S.histogram(Z, { bins: 60, lo: -5, hi: 5 });
        const cau = dist.get() === 'cauchy', sq = norm.get() === 'sqrt';
        const ref = x => (cau ? (sq ? NaN : S.pdf.cauchy(x)) : (sq ? S.pdf.normal(x) : NaN));
        const ymax = Math.max(0.45, ...h.map(b => b.density)) * 1.1;
        pH.setDomain([-5, 5], [0, ymax]).clear();
        pH.bars(h, { color: 's1', opacity: 0.45 });
        pH.line(S.linspace(-5, 5, 300).map(x => [x, ref(x)]), { color: 'ink', dash: true, width: 1.6 });
        sIn.set(S.fmt(Z.filter(z => Math.abs(z) <= 5).length / Z.length, 3));
        drawCF();
      }
      simulate();
    },
  });

  /* ================= multivariate ================= */

  Atlas.lab('multivariate-bivariate-normal', {
    title: 'The bivariate normal: slices, regression lines and the principal axis',
    blurb: 'Drag on the scatter (or use the slider) to choose $x_0$. Right: the conditional law of $Y$ given $X = x_0$ — normal, centred on the regression line, with variance $\\sigma_Y^2(1-\\rho^2)$ whatever $x_0$ is. Note that the regression line is flatter than the ellipse\'s major axis: regression to the mean.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rho = UI.slider(controls, { label: 'Correlation $\\rho$', min: -0.95, max: 0.95, step: 0.01, value: 0.7, onInput: draw });
      const sx = UI.slider(controls, { label: '$\\sigma_X$', min: 0.5, max: 2, step: 0.05, value: 1, onInput: draw });
      const sy = UI.slider(controls, { label: '$\\sigma_Y$', min: 0.5, max: 2, step: 0.05, value: 1, onInput: draw });
      const x0 = UI.slider(controls, { label: 'Conditioning value $x_0$', min: -3, max: 3, step: 0.05, value: 1.5, onInput: draw });
      const tXY = UI.toggle(controls, { label: 'Show regression of $X$ on $Y$', value: false, onChange: draw });
      const tPA = UI.toggle(controls, { label: 'Show principal axis', value: true, onChange: draw });
      UI.button(controls, 'Resample', resample, { primary: true });
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 300, x: [-5, 5], y: [-5, 5], equal: true, xlabel: 'x', ylabel: 'y', title: 'Joint distribution' });
      const pC = new Plot(stage, { width: 420, height: 300, x: [-6, 6], y: [0, 0.8], xlabel: 'y', ylabel: 'density', title: 'Y given X ≈ x₀' });
      const sE = UI.stat(stats, '$\\E[Y\\mid X=x_0]$', { accent: 'accent' });
      const sSD = UI.stat(stats, '$\\text{SD}(Y\\mid X=x_0)$');
      const sR2 = UI.stat(stats, 'Variance explained $\\rho^2$');
      const sB = UI.stat(stats, 'Points in slice');
      let Z = [];
      const H = 0.2;
      function resample() { Z = ctx.rng.array(4000, () => [ctx.rng.normal(), ctx.rng.normal()]); draw(); }
      pS.onPointer((x, y, e, type) => { if (type === 'down' || type === 'drag') { x0.set(S.clamp(x, -3, 3)); draw(); } });
      function draw() {
        const r = rho.get(), a = sx.get(), b = sy.get(), x0v = x0.get(), q = Math.sqrt(1 - r * r);
        const P = Z.map(([z1, z2]) => [a * z1, b * (r * z1 + q * z2)]);
        const [X0, X1] = pS.o.x;
        const cov = [[a * a, r * a * b], [r * a * b, b * b]];
        pS.clear();
        pS.band(x0v - H, x0v + H, { color: 's2', opacity: 0.12 });
        pS.points(P.slice(0, 700), { color: 'muted', r: 1.8, opacity: 0.45 });
        pS.ellipse(0, 0, cov, 1, { color: 's4', width: 1.4 });
        pS.ellipse(0, 0, cov, 2, { color: 's4', width: 1.4, dash: true });
        const slope = r * b / a;
        pS.line([[X0, slope * X0], [X1, slope * X1]], { color: 's1', width: 2.4 });
        if (tXY.get() && Math.abs(r) > 0.02) { const s2 = b / (r * a); pS.line([[X0, s2 * X0], [X1, s2 * X1]], { color: 's3', width: 2 }); }
        if (tPA.get()) { const ev = S.eigSym(cov).vectors, vx = ev[0][0], vy = ev[1][0]; pS.line([[-10 * vx, -10 * vy], [10 * vx, 10 * vy]], { color: 'ink', width: 1.2, dash: '6 4' }); }
        pS.dot(x0v, slope * x0v, { color: 's1', r: 6 });
        pS.legend([{ label: 'E[Y | X = x]', color: 's1' }].concat(tXY.get() ? [{ label: 'E[X | Y = y]', color: 's3' }] : []).concat(tPA.get() ? [{ label: 'major axis', color: 'ink', dash: true }] : []), { left: true });
        const m = slope * x0v, sd = b * q;
        const slice = P.filter(p => Math.abs(p[0] - x0v) < H).map(p => p[1]);
        const ys = S.linspace(-6, 6, 300);
        const h = S.histogram(slice, { bins: 36, lo: -6, hi: 6 });
        for (const bb of h) bb.density = slice.length ? bb.count / (slice.length * (bb.x1 - bb.x0)) : 0;
        const ymax = Math.max(S.pdf.normal(0, 0, sd), ...h.map(bb => bb.density)) * 1.15;
        pC.setDomain([-6, 6], [0, ymax]).clear();
        pC.bars(h, { color: 's2', opacity: 0.35 });
        pC.line(ys.map(y => [y, S.pdf.normal(y, 0, b)]), { color: 'muted', dash: true, width: 1.5 });
        pC.area(ys.map(y => [y, S.pdf.normal(y, m, sd)]), { color: 's1', opacity: 0.15 });
        pC.line(ys.map(y => [y, S.pdf.normal(y, m, sd)]), { color: 's1', width: 2.2 });
        pC.vline(m, { color: 's1' });
        pC.legend([{ label: 'f(y | x₀)', color: 's1' }, { label: 'marginal f(y)', color: 'muted', dash: true }, { label: 'points in slice', color: 's2', fill: true }]);
        sE.set(S.fmt(m), `= ρ σ_Y/σ_X · x₀`);
        sSD.set(S.fmt(sd), `marginal SD ${S.fmt(b, 2)}`);
        sR2.set(pct(r * r));
        sB.set(String(slice.length), `|x − x₀| < ${H}`);
      }
      resample();
    },
  });

  Atlas.lab('multivariate-not-jointly-normal', {
    title: 'Normal marginals, zero correlation, total dependence',
    blurb: 'Let $X\\sim\\N(0,1)$ and $Y = X$ if $|X|\\le c$, else $Y=-X$. Both are standard normal for every $c$. Tune $c$ until $\\rho = 0$ (about 1.54): the variables are uncorrelated yet $|Y|=|X|$, and $X+Y$ has an atom at 0, so the pair is not bivariate normal.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const c = UI.slider(controls, { label: 'Cutoff $c$', min: 0, max: 3.5, step: 0.01, value: 1.0, onInput: draw });
      UI.button(controls, 'Set c so that ρ = 0', () => { c.set(S.invert(v => rhoC(v) + 1, 1, 0, 3.5)); draw(); });
      UI.button(controls, 'Resample', resample, { primary: true });
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 280, x: [-4, 4], y: [-4, 4], equal: true, xlabel: 'x', ylabel: 'y', title: '(X, Y)' });
      const pH = new Plot(stage, { width: 420, height: 280, x: [-6, 6], y: [0, 0.5], xlabel: 'x + y', ylabel: 'density', title: 'X + Y' });
      const sR = UI.stat(stats, 'Correlation $\\rho(c)$', { accent: 'accent' });
      const sRs = UI.stat(stats, 'Sample correlation');
      const sA = UI.stat(stats, '$\\P(X+Y=0)$');
      const sY = UI.stat(stats, 'Sample SD of $Y$');
      let X = [];
      function rhoC(cc) { return 2 * ((2 * S.cdf.normal(cc) - 1) - 2 * cc * S.pdf.normal(cc)) - 1; }
      function resample() { X = ctx.rng.array(3000, () => ctx.rng.normal()); draw(); }
      function draw() {
        const cc = c.get(), Y = X.map(x => (Math.abs(x) <= cc ? x : -x)), r = rhoC(cc);
        pS.clear();
        pS.band(-cc, cc, { color: 's1', opacity: 0.06 });
        pS.points(X.slice(0, 1500).map((x, i) => [x, Y[i]]), { color: 's1', r: 1.8, opacity: 0.5 });
        const Ssum = X.map((x, i) => x + Y[i]), nz = Ssum.filter(s => Math.abs(s) > 1e-12);
        const h = S.histogram(nz, { bins: 60, lo: -6, hi: 6 });
        for (const b of h) b.density = b.count / (X.length * (b.x1 - b.x0));
        const v = 2 + 2 * r, atom = 2 * (1 - S.cdf.normal(cc));
        const ymax = Math.max(0.3, ...h.map(b => b.density), v > 0.05 ? S.pdf.normal(0, 0, Math.sqrt(v)) : 0) * 1.2;
        pH.setDomain([-6, 6], [0, ymax]).clear();
        pH.bars(h, { color: 's1', opacity: 0.4 });
        if (v > 0.05) pH.line(S.linspace(-6, 6, 300).map(x => [x, S.pdf.normal(x, 0, Math.sqrt(v))]), { color: 'bad', dash: true, width: 1.8 });
        if (atom > 0.002) { pH.segment(0, 0, 0, ymax * 0.85, { color: 's2', width: 3 }); pH.dot(0, ymax * 0.85, { color: 's2', r: 5 }); pH.text(0, ymax * 0.85, `atom ${pct(atom)}`, { dx: 8, dy: 4, size: 11, color: 's2' }); }
        pH.legend([{ label: 'N(0, 2+2ρ): if jointly normal', color: 'bad', dash: true }, { label: 'continuous part', color: 's1', fill: true }], { left: true });
        sR.set(S.fmt(r, 3));
        sRs.set(S.fmt(S.cor(X, Y), 3));
        sA.set(pct(atom), 'a bivariate normal would give 0');
        sY.set(S.fmt(S.sd(Y), 3), 'Y is N(0,1) for every c');
      }
      resample();
    },
  });

  /* ================= conditional-expectation ================= */

  const MSH = {
    lin: { label: 'Linear: 0.8x', m: x => 0.8 * x },
    quad: { label: 'Quadratic: x² − 4/3', m: x => x * x - 4 / 3 },
    sine: { label: 'Sine: 1.5 sin(1.5x)', m: x => 1.5 * Math.sin(1.5 * x) },
    step: { label: 'Step: sign(x)', m: x => (x < 0 ? -1 : 1) },
  };

  Atlas.lab('conditional-expectation-projection', {
    title: 'E[Y | X] is the best predictor',
    blurb: '$X\\sim$ Unif$(-2,2)$ and $Y = m(X) + $ noise. Compare the best constant, the best linear predictor and the true $\\E[Y\\mid X]$ (exact test MSEs, right), with a data-driven estimate: binned averages. Few bins: bias; many bins: variance — training error keeps falling while test error turns back up.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const shape = UI.select(controls, { label: 'Regression function $m$', options: Object.keys(MSH).map(k => ({ value: k, label: MSH[k].label })), value: 'sine', onChange: draw });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.1, max: 2, step: 0.05, value: 0.6, onInput: draw });
      const het = UI.toggle(controls, { label: 'Heteroscedastic noise ($\\propto 0.25 + 0.75|x|$)', value: false, onChange: draw });
      const n = UI.slider(controls, { label: 'Training size $n$', min: 50, max: 3000, step: 50, value: 300, onChange: resample });
      const bins = UI.slider(controls, { label: 'Bins for the estimate', min: 1, max: 80, step: 1, value: 8, onInput: draw });
      UI.button(controls, 'New training sample', resample, { primary: true });
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 280, x: [-2, 2], y: [-4, 4], xlabel: 'x', ylabel: 'y', title: 'Data and predictors' });
      const pB = new Plot(stage, { width: 420, height: 280, x: [0, 4], y: [0, 1], xticks: 0, ylabel: 'test MSE', title: 'Mean squared prediction error' });
      const sC = UI.stat(stats, 'MSE constant $= \\Var Y$');
      const sL = UI.stat(stats, 'MSE best linear');
      const sBn = UI.stat(stats, 'MSE binned (test / train)', { accent: 'accent' });
      const sM = UI.stat(stats, 'MSE of $\\E[Y\\mid X]$ $= \\E\\Var(Y\\mid X)$');
      let xs = [], es = [];
      const G = S.linspace(-2, 2, 2001).map((x, i, a) => (i < a.length - 1 ? (x + a[i + 1]) / 2 : null)).filter(v => v != null);
      const sdAt = x => sig.get() * (het.get() ? 0.25 + 0.75 * Math.abs(x) : 1);
      function resample() { xs = ctx.rng.array(Math.round(n.get()), () => ctx.rng.uniform(-2, 2)); es = ctx.rng.array(xs.length, () => ctx.rng.normal()); draw(); }
      function draw() {
        const m = MSH[shape.get()].m, B = Math.round(bins.get());
        const ys = xs.map((x, i) => m(x) + sdAt(x) * es[i]);
        // population quantities (X uniform on [-2, 2]: average over the midpoint grid G)
        const mg = G.map(m), Em = S.mean(mg), Es2 = S.mean(G.map(x => sdAt(x) ** 2));
        const varM = S.mean(mg.map(v => (v - Em) ** 2)), beta = S.mean(G.map((x, i) => x * mg[i])) / (4 / 3);
        const mseC = varM + Es2, mseL = S.mean(G.map((x, i) => (mg[i] - Em - beta * x) ** 2)) + Es2, mseM = Es2;
        // binned estimate
        const sums = new Array(B).fill(0), cnt = new Array(B).fill(0), ybar = S.mean(ys);
        const bi = x => Math.min(B - 1, Math.floor((x + 2) / 4 * B));
        xs.forEach((x, i) => { const k = bi(x); sums[k] += ys[i]; cnt[k]++; });
        const gh = sums.map((s, k) => (cnt[k] ? s / cnt[k] : ybar));
        const mseB = S.mean(G.map((x, i) => (mg[i] - gh[bi(x)]) ** 2)) + Es2;
        const trainB = S.mean(xs.map((x, i) => (ys[i] - gh[bi(x)]) ** 2));
        // left panel
        const ext = S.extent(ys.concat(mg));
        pS.setDomain([-2, 2], [Math.max(-8, ext[0] - 0.3), Math.min(8, ext[1] + 0.3)]).clear();
        pS.points(xs.slice(0, 1500).map((x, i) => [x, ys[i]]), { color: 'muted', r: 1.8, opacity: 0.45 });
        pS.hline(Em, { color: 's4', width: 1.5 });
        pS.line([[-2, Em - 2 * beta], [2, Em + 2 * beta]], { color: 's3', width: 2, dash: true });
        const stepPts = [];
        for (let k = 0; k < B; k++) { const a = -2 + 4 * k / B, b = -2 + 4 * (k + 1) / B; stepPts.push([a, gh[k]], [b, gh[k]]); if (k < B - 1) stepPts.push(null); }
        pS.line(stepPts, { color: 's2', width: 2.4 });
        pS.line(G.filter((_, i) => i % 10 === 0).map(x => [x, m(x)]), { color: 'ink', width: 2 });
        pS.legend([{ label: 'E[Y | X]', color: 'ink' }, { label: 'binned means', color: 's2' }, { label: 'best linear', color: 's3', dash: true }, { label: 'best constant', color: 's4' }], { left: true });
        // right panel
        const vals = [[mseC, 'constant', 's4'], [mseL, 'linear', 's3'], [mseB, 'binned', 's2'], [mseM, 'E[Y|X]', 'ink']];
        const ymax = Math.max(...vals.map(v => v[0]), trainB) * 1.2;
        pB.setDomain([0, 4], [0, ymax]).clear();
        vals.forEach(([v, lab, c], k) => {
          pB.bars([{ x0: k + 0.15, x1: k + 0.85, y: v }], { color: c, opacity: 0.6 });
          pB.text(k + 0.5, v, S.fmt(v, 3), { anchor: 'middle', dy: -6, size: 11, weight: 600 });
          pB.text(k + 0.5, 0, lab, { anchor: 'middle', dy: 14, size: 11, color: 'muted' });
        });
        pB.segment(2.1, trainB, 2.9, trainB, { color: 'bad', width: 2.5 });
        pB.text(2.9, trainB, 'train', { dx: 4, dy: 4, size: 10, color: 'bad' });
        sC.set(S.fmt(mseC, 3));
        sL.set(S.fmt(mseL, 3), `slope ${S.fmt(beta, 3)}`);
        sBn.set(`${S.fmt(mseB, 3)} / ${S.fmt(trainB, 3)}`, `${B} bins`);
        sM.set(S.fmt(mseM, 3), 'the floor no predictor can beat');
      }
      resample();
    },
  });

  Atlas.lab('conditional-expectation-total-variance', {
    title: 'The law of total variance, within and between',
    blurb: 'A group $G$ is drawn, then $Y\\mid G=g \\sim \\N(\\mu_g, s_g^2)$. Spreading the group means grows the between part $\\Var\\E[Y\\mid G]$; noisier groups grow the within part $\\E\\Var(Y\\mid G)$. The sample decomposition (one-way ANOVA) adds up exactly, too.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const K = UI.slider(controls, { label: 'Groups $K$', min: 2, max: 6, step: 1, value: 4, onInput: draw });
      const d = UI.slider(controls, { label: 'Spacing of group means $d$', min: 0, max: 4, step: 0.05, value: 1.5, onInput: draw });
      const s = UI.slider(controls, { label: 'Within-group SD $s$', min: 0.2, max: 3, step: 0.05, value: 1, onInput: draw });
      const uneqS = UI.toggle(controls, { label: 'Unequal within-group SDs', value: false, onChange: draw });
      const uneqP = UI.toggle(controls, { label: 'Unequal group sizes', value: false, onChange: draw });
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 100, max: 3000, step: 100, value: 600, onChange: resample });
      UI.button(controls, 'Resample', resample, { primary: true });
      stage.classList.add('two');
      const pS = new Plot(stage, { width: 420, height: 280, xlabel: 'group', ylabel: 'y', title: 'Observations by group', xticks: 0 });
      const pB = new Plot(stage, { width: 420, height: 280, x: [0, 2], y: [0, 1], xticks: 0, ylabel: 'variance', title: 'Var Y = E Var(Y|G) + Var E[Y|G]' });
      const sT = UI.stat(stats, '$\\Var Y$');
      const sW = UI.stat(stats, '$\\E\\Var(Y\\mid G)$ (within)');
      const sBt = UI.stat(stats, '$\\Var\\E[Y\\mid G]$ (between)');
      const sE = UI.stat(stats, 'Explained $\\eta^2$', { accent: 'accent' });
      const COLS = ['s1', 's2', 's3', 's4', 's5', 's6'];
      let U = [], E = [], J = [];
      function resample() { const m = Math.round(n.get()); U = ctx.rng.array(m, () => ctx.rng.u()); E = ctx.rng.array(m, () => ctx.rng.normal()); J = ctx.rng.array(m, () => ctx.rng.u() - 0.5); draw(); }
      function draw() {
        const k = Math.round(K.get()), dd = d.get(), ss = s.get();
        const w = S.range(k).map(g => (uneqP.get() ? g + 1 : 1)), ws = S.sum(w), pr = w.map(v => v / ws);
        const mu = S.range(k).map(g => dd * (g - (k - 1) / 2)), sg = S.range(k).map(g => (uneqS.get() ? ss * (0.4 + 1.2 * g / (k - 1)) : ss));
        const cp = S.cumsum(pr);
        const grp = U.map(u => { let g = cp.findIndex(c => u <= c); return g < 0 ? k - 1 : g; });
        const y = grp.map((g, i) => mu[g] + sg[g] * E[i]);
        const mbar = S.sum(pr.map((p, g) => p * mu[g]));
        const within = S.sum(pr.map((p, g) => p * sg[g] ** 2)), between = S.sum(pr.map((p, g) => p * (mu[g] - mbar) ** 2));
        // sample decomposition
        const N = y.length, yb = S.mean(y), gs = new Array(k).fill(0), gc = new Array(k).fill(0);
        grp.forEach((g, i) => { gs[g] += y[i]; gc[g]++; });
        const gm = gs.map((v, g) => (gc[g] ? v / gc[g] : 0));
        let ssw = 0, ssb = 0;
        grp.forEach((g, i) => { ssw += (y[i] - gm[g]) ** 2; });
        gc.forEach((c, g) => { ssb += c * (gm[g] - yb) ** 2; });
        ssw /= N; ssb /= N;
        const ext = S.extent(y);
        pS.setDomain([-0.6, k - 0.4], [ext[0] - 0.5, ext[1] + 0.5]).clear();
        for (let g = 0; g < k; g++) {
          const pts = []; grp.forEach((gg, i) => { if (gg === g && pts.length < 400) pts.push([g + 0.6 * J[i], y[i]]); });
          pS.points(pts, { color: COLS[g], r: 1.8, opacity: 0.5 });
          if (gc[g]) pS.segment(g - 0.35, gm[g], g + 0.35, gm[g], { color: COLS[g], width: 3 });
          pS.text(g, ext[0] - 0.5, `p = ${S.fmt(pr[g], 2)}`, { anchor: 'middle', dy: 14, size: 10, color: 'muted' });
        }
        pS.hline(yb, { color: 'ink', width: 1.2 });
        const tot = Math.max(within + between, ssw + ssb);
        pB.setDomain([0, 2], [0, tot * 1.2]).clear();
        [[within, between, 'theory'], [ssw, ssb, 'sample']].forEach(([a, b, lab], i) => {
          pB.bars([{ x0: i + 0.2, x1: i + 0.8, y: a }], { color: 's1', opacity: 0.6 });
          pB.bars([{ x0: i + 0.2, x1: i + 0.8, y: a + b }], { color: 's2', opacity: 0.6, base: a });
          pB.text(i + 0.5, a + b, S.fmt(a + b, 3), { anchor: 'middle', dy: -6, size: 12, weight: 600 });
          pB.text(i + 0.5, 0, lab, { anchor: 'middle', dy: 14, size: 11, color: 'muted' });
        });
        pB.legend([{ label: 'within: E Var(Y|G)', color: 's1', fill: true }, { label: 'between: Var E[Y|G]', color: 's2', fill: true }], { left: true });
        sT.set(S.fmt(within + between, 3), `sample ${S.fmt(S.variance(y, 0), 3)}`);
        sW.set(S.fmt(within, 3), `sample ${S.fmt(ssw, 3)}`);
        sBt.set(S.fmt(between, 3), `sample ${S.fmt(ssb, 3)}`);
        sE.set(pct(between / (within + between)), `sample ${pct(ssb / (ssw + ssb))}`);
      }
      resample();
    },
  });
})();
