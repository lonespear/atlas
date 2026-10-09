/* Labs: Simulation Theory. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const fmt = (x, d = 3) => S.fmt(x, d);
  const pct = (x, d = 1) => (100 * x).toFixed(d) + '%';

  /* ================= pseudo-random ================= */

  // Shortest dual vector of the 2-D LCG lattice: min ||h|| over h1 + a h2 = 0 (mod m), h != 0.
  function spectral2(a, m) {
    let best = { len: m, h1: m, h2: 0 };
    const lim = Math.min(m, Math.ceil(1.2 * Math.sqrt(m)) + 2);
    for (let h2 = 1; h2 <= lim; h2++) {
      let h1 = (-a * h2) % m; if (h1 < 0) h1 += m;
      if (h1 > m / 2) h1 -= m;
      const len = Math.hypot(h1, h2);
      if (len < best.len) best = { len, h1, h2 };
    }
    return best;
  }
  function lcgPeriod(a, c, m, x0) {
    const seen = new Int32Array(m).fill(-1);
    let x = x0, n = 0;
    while (seen[x] < 0) { seen[x] = n++; x = (a * x + c) % m; }
    return n - seen[x];
  }
  function primeFactors(m) { const f = []; for (let p = 2; p * p <= m; p++) if (m % p === 0) { f.push(p); while (m % p === 0) m /= p; } if (m > 1) f.push(m); return f; }
  const gcd = (x, y) => (y ? gcd(y, x % y) : x);

  Atlas.lab('pseudo-random-lcg-lattice', {
    title: 'An LCG\'s pairs live on a lattice',
    blurb: 'All pairs $(u_n, u_{n+1})$ over one period of $x_{n+1} = (a x_n + c) \\bmod m$. Hull–Dobell decides the period; the spectral test $\\nu_2$ decides how far apart the covering lines are. Try $a = 1$, or $a \\equiv 1 \\pmod 4$ just above $m/2$, then press "Best multiplier".',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el, { side: true });
      const mSel = UI.select(controls, { label: 'Modulus $m$', options: [256, 1024, 4096].map(v => ({ value: String(v), label: String(v) + ' = 2^' + Math.log2(v) })), value: '1024', onChange: draw });
      const aS = UI.slider(controls, { label: 'Multiplier $a$', min: 1, max: 4095, step: 1, value: 517, onInput: draw });
      const cS = UI.slider(controls, { label: 'Increment $c$', min: 0, max: 255, step: 1, value: 1, onInput: draw });
      const lines = UI.toggle(controls, { label: 'Show covering lines', value: true, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Best multiplier', () => searchA(true), { primary: true });
      UI.button(row, 'Worst multiplier', () => searchA(false));
      const p = new Plot(stage, { width: 520, height: 520, x: [0, 1], y: [0, 1], xlabel: 'uₙ', ylabel: 'uₙ₊₁', equal: true });
      const sP = UI.stat(stats, 'Period (seed 1)', { accent: 'accent' });
      const sHD = UI.stat(stats, 'Hull–Dobell conditions');
      const sNu = UI.stat(stats, 'Spectral test $\\nu_2$');
      const sGap = UI.stat(stats, 'Max line spacing $1/\\nu_2$');
      function params() {
        const m = +mSel.get();
        return { m, a: Math.max(1, Math.round(aS.get()) % m), c: Math.round(cS.get()) % m };
      }
      function searchA(best) {
        const m = +mSel.get();
        let ba = 1, bv = best ? -1 : Infinity;
        for (let a = 5; a < m; a += 4) { const v = spectral2(a, m).len; if (best ? v > bv : v < bv) { bv = v; ba = a; } }
        aS.set(ba); if (Math.round(cS.get()) % 2 === 0) cS.set(1);
        draw();
      }
      function draw() {
        const { m, a, c } = params();
        const x0 = 1 % m, per = lcgPeriod(a, c, m, x0);
        const pts = [];
        let x = x0;
        for (let i = 0; i < per; i++) { const y = (a * x + c) % m; pts.push([x / m, y / m]); x = y; }
        p.clear();
        const sp = spectral2(a, m);
        if (lines.get() && sp.len < m) {
          const { h1, h2 } = sp, b = ((h1 * pts[0][0] + h2 * pts[0][1]) % 1 + 1) % 1;
          const lo = Math.min(0, h1) + Math.min(0, h2), hi = Math.max(0, h1) + Math.max(0, h2);
          let nl = 0;
          for (let k = Math.floor(lo - b) - 1; k <= Math.ceil(hi - b) + 1 && nl < 400; k++, nl++) {
            const v = b + k;
            if (h2 !== 0) p.segment(-0.1, (v - h1 * -0.1) / h2, 1.1, (v - h1 * 1.1) / h2, { color: 's2', width: 1, opacity: 0.45 });
            else p.segment(v / h1, -0.1, v / h1, 1.1, { color: 's2', width: 1, opacity: 0.45 });
          }
        }
        p.points(pts, { color: 's1', r: m > 2000 ? 1.4 : 2, opacity: 0.85 });
        const pf = primeFactors(m);
        const c1 = gcd(c, m) === 1, c2 = pf.every(q => (a - 1) % q === 0), c3 = m % 4 !== 0 || (a - 1) % 4 === 0;
        const mark = ok => (ok ? '<span style="color:var(--good)">✓</span>' : '<span style="color:var(--bad)">✗</span>');
        sP.set(String(per), per === m ? 'full period' : `${pct(per / m)} of m`);
        sHD.set(`${mark(c1)} ${mark(c2)} ${mark(c3)}`, 'gcd(c,m)=1 · p|m ⇒ p|a−1 · 4|m ⇒ 4|a−1');
        sNu.set(fmt(sp.len, 3), `h = (${sp.h1}, ${sp.h2}); bound ≈ ${fmt(Math.sqrt(2 / Math.sqrt(3) * m), 3)}`);
        sGap.set(fmt(1 / sp.len, 3), `≈ ${Math.abs(sp.h1) + Math.abs(sp.h2)} lines cover the square`);
      }
      draw();
    },
  });

  // Three generators producing u in [0,1).
  function makeGen(kind, rng) {
    if (kind === 'randu') { let x = 1; return () => { x = (65539 * x) % 2147483648; return x / 2147483648; }; }
    if (kind === 'minstd') { let x = 1; return () => { x = (16807 * x) % 2147483647; return x / 2147483647; }; }
    return () => rng.u();
  }

  Atlas.lab('pseudo-random-randu', {
    title: 'Rotating RANDU until its 15 planes appear',
    blurb: 'Consecutive triples $(u_n,u_{n+1},u_{n+2})$ projected onto a plane. At angle $0^\\circ$ the viewing axis is the normal $(9,-6,1)$ and RANDU collapses onto 15 stripes; MINSTD and a modern generator (mulberry32) show no structure at this scale. Right: the distribution of $9u_n - 6u_{n+1} + u_{n+2}$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const gen = UI.segmented(controls, { label: 'Generator', options: [{ value: 'randu', label: 'RANDU' }, { value: 'minstd', label: 'MINSTD' }, { value: 'good', label: 'mulberry32' }], value: 'randu', onChange: regen });
      const nS = UI.slider(controls, { label: 'Triples', min: 1000, max: 20000, step: 1000, value: 6000, onChange: regen });
      const ang = UI.slider(controls, { label: 'View angle (degrees)', min: -90, max: 90, step: 1, value: 35, onInput: draw, fmt: v => v + '°' });
      UI.button(controls, 'Snap to 0°', () => { ang.set(0); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-0.95, 0.95], y: [-0.95, 0.95], xlabel: 'axis 1', ylabel: 'axis 2', equal: true, title: 'Projection of triples' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [-6, 10], y: [0, 1], xlabel: '9uₙ − 6uₙ₊₁ + uₙ₊₂', ylabel: 'density', title: 'A linear combination of a triple' });
      const sI = UI.stat(stats, 'Integer-valued fraction', { accent: 'accent' });
      const sD = UI.stat(stats, 'Distinct values');
      const nh = Math.sqrt(118), H = [9 / nh, -6 / nh, 1 / nh];
      const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
      const norm = v => { const l = Math.hypot(...v); return v.map(x => x / l); };
      const P1 = norm(cross(H, [0, 0, 1])), P2 = norm(cross(H, P1));
      let T = [], comb = [];
      function regen() {
        const g = makeGen(gen.get(), ctx.rng), n = Math.round(nS.get());
        let a = g(), b = g();
        T = new Array(n); comb = new Array(n);
        for (let i = 0; i < n; i++) { const c = g(); T[i] = [a - 0.5, b - 0.5, c - 0.5]; comb[i] = 9 * a - 6 * b + c; a = b; b = c; }
        draw();
      }
      function draw() {
        const th = ang.get() * Math.PI / 180, ct = Math.cos(th), st = Math.sin(th);
        const A1 = [0, 1, 2].map(k => ct * H[k] + st * P1[k]);
        const pts = T.map(t => [t[0] * A1[0] + t[1] * A1[1] + t[2] * A1[2], t[0] * P2[0] + t[1] * P2[1] + t[2] * P2[2]]);
        pL.clear();
        pL.points(pts, { color: 's1', r: 1.1, opacity: 0.55 });
        const hb = S.histogram(comb, { bins: 320, lo: -6, hi: 10 });
        const ymax = Math.min(4, Math.max(...hb.map(b => b.density)) * 1.1);
        pR.setDomain([-6, 10], [0, Math.max(0.2, ymax)]).clear();
        pR.bars(hb, { color: 's2', opacity: 0.7, gap: 0 });
        let integ = 0; const vals = new Set();
        for (const v of comb) { const r = Math.round(v); if (Math.abs(v - r) < 1e-6) { integ++; vals.add(r); } }
        sI.set(pct(integ / comb.length), integ === comb.length ? 'every triple on a plane 9x−6y+z = k' : 'no plane structure in this direction');
        sD.set(integ === comb.length ? String(vals.size) + ' planes' : '—', integ === comb.length ? `k ∈ {${Math.min(...vals)},…,${Math.max(...vals)}}` : '');
      }
      regen();
    },
  });

  /* ================= inverse-transform ================= */

  const INV = {
    exp: { label: 'Exponential(1)', F: x => (x < 0 ? 0 : 1 - Math.exp(-x)), Q: u => -Math.log(1 - u), range: [-0.3, 5] },
    normal: { label: 'Normal(0,1)', F: x => S.cdf.normal(x), Q: u => S.q.normal(u), range: [-3.5, 3.5] },
    poisson: { label: 'Poisson(3)', F: x => (x < 0 ? 0 : S.cdf.poisson(Math.floor(x), 3)), Q: u => { let k = 0, P = S.pmf.poisson(0, 3); while (P < u && k < 60) { k++; P += S.pmf.poisson(k, 3); } return k; }, range: [-0.5, 10], discrete: true },
    mixed: { label: 'Atom at 0 (30%) + Exp(1)', F: x => (x < 0 ? 0 : 0.3 + 0.7 * (1 - Math.exp(-x))), Q: u => (u <= 0.3 ? 0 : -Math.log((1 - u) / 0.7)), range: [-0.5, 5], jumps: [0] },
    gap: { label: 'Uniform on [0,1] ∪ [2,3]', F: x => (x < 0 ? 0 : x <= 1 ? x / 2 : x <= 2 ? 0.5 : x <= 3 ? 0.5 + (x - 2) / 2 : 1), Q: u => (u <= 0.5 ? 2 * u : 2 + 2 * (u - 0.5)), range: [-0.3, 3.3] },
  };
  function cdfPath(D) {
    const [lo, hi] = D.range;
    if (D.discrete) { const pts = [[lo, 0]]; for (let k = 0; k <= hi; k++) pts.push([k, D.F(k)]); pts.push([hi, D.F(hi)]); return { step: pts }; }
    const xs = S.linspace(lo, hi, 500);
    if (D.jumps) for (const j of D.jumps) { xs.push(j - 1e-9); xs.push(j); }
    xs.sort((a, b) => a - b);
    return { line: xs.map(x => [x, D.F(x)]) };
  }

  Atlas.lab('inverse-transform-mapping', {
    title: 'Uniforms through the generalized inverse',
    blurb: 'Each uniform $U$ on the vertical axis is carried across to the CDF and down to $X = F^{-}(U)$. Right: the empirical CDF of the draws converging to $F$. Try the atom (a whole interval of $U$ maps to one point) and the gap (no $U$ maps inside the flat stretch).',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dist = UI.select(controls, { label: 'Distribution $F$', options: Object.keys(INV).map(k => ({ value: k, label: INV[k].label })), value: 'mixed', onChange: clear });
      const speed = UI.slider(controls, { label: 'Draws per second', min: 2, max: 400, step: 1, value: 12, log: true, fmt: v => Math.round(v) });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Draw', '❚❚ Pause']);
      UI.button(row, '+1000', () => { for (let i = 0; i < 1000; i++) add(); draw(); });
      UI.button(row, 'Clear', clear);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, y: [0, 1], xlabel: 'x', ylabel: 'u = F(x)', title: 'X = F⁻(U)' });
      const pR = new Plot(stage, { width: 420, height: 280, y: [0, 1], xlabel: 'x', ylabel: 'F(x)', title: 'Empirical CDF of draws vs F' });
      const sN = UI.stat(stats, 'Draws $n$');
      const sKS = UI.stat(stats, 'KS distance $\\sup_x|F_n - F|$', { accent: 'accent' });
      const sRt = UI.stat(stats, '$\\sqrt n \\times$ KS');
      let xs = [], recent = [], acc = 0;
      function clear() { xs = []; recent = []; draw(); }
      function add() { const D = INV[dist.get()], u = ctx.rng.u(), x = D.Q(u); xs.push(x); recent.push([u, x]); if (recent.length > 14) recent.shift(); }
      function step(dt) { acc += dt * speed.get(); let k = 0; while (acc >= 1 && k < 50) { add(); acc -= 1; k++; } if (k) draw(); }
      function draw() {
        const D = INV[dist.get()], [lo, hi] = D.range, path = cdfPath(D);
        pL.setDomain([lo, hi], [0, 1]).clear(); pR.setDomain([lo, hi], [0, 1]).clear();
        if (path.step) { pL.step(path.step, { color: 'ink', width: 2 }); pR.step(path.step, { color: 'ink', width: 2 }); }
        else { pL.line(path.line, { color: 'ink', width: 2 }); pR.line(path.line, { color: 'ink', width: 2 }); }
        recent.forEach(([u, x], i) => {
          const op = 0.25 + 0.75 * (i + 1) / recent.length;
          pL.segment(lo, u, x, u, { color: 's1', width: 1.4, opacity: op });
          pL.arrow(x, u, x, 0, { color: 's1', width: 1.4, opacity: op });
          pL.dot(lo, u, { color: 's1', r: 3, opacity: op });
        });
        if (xs.length) {
          const s = xs.slice().sort((a, b) => a - b), n = s.length, pts = [[lo, 0]];
          let D_ = 0;
          for (let i = 0; i < n; i++) {
            pts.push([s[i], (i + 1) / n]);
            const F = D.F(s[i]), Fm = D.F(s[i] - 1e-9);
            D_ = Math.max(D_, Math.abs((i + 1) / n - F), Math.abs(i / n - Fm));
          }
          pts.push([hi, 1]);
          pR.step(pts, { color: 's1', width: 1.6 });
          sKS.set(fmt(D_, 3)); sRt.set(fmt(Math.sqrt(n) * D_, 3), 'stays O(1): KS error ∝ n^{-1/2}');
        } else { sKS.set('—'); sRt.set('—'); }
        pR.legend([{ label: 'F', color: 'ink' }, { label: 'empirical', color: 's1' }], { left: true });
        sN.set(String(xs.length));
      }
      draw();
    },
  });

  // Kolmogorov asymptotic p-value.
  function ksP(D, n) {
    const l = (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * D;
    if (l < 0.2) return 1;
    let s = 0; for (let k = 1; k < 100; k++) s += 2 * (k % 2 ? 1 : -1) * Math.exp(-2 * k * k * l * l);
    return Math.max(0, Math.min(1, s));
  }
  const nbCdf = (x, r, p) => { let s = 0; for (let k = 0; k <= x; k++) s += S.pmf.negbinom(k, r, p); return Math.min(1, s); };
  const PIT = {
    normal: { label: 'Normal(0,1)', draw: r => r.normal(), models: { correct: x => S.cdf.normal(x), shift: x => S.cdf.normal(x, 0.5, 1), wide: x => S.cdf.normal(x, 0, 1.5), narrow: x => S.cdf.normal(x, 0, 0.7) } },
    exp: { label: 'Exponential(1)', draw: r => r.exp(1), models: { correct: x => S.cdf.exp(x), shift: x => S.cdf.exp(x, 1 / 1.4), wide: x => S.cdf.gamma(x, 0.6, 1 / 0.6), narrow: x => S.cdf.gamma(x, 2, 0.5) } },
    poisson: { label: 'Poisson(4)', discrete: true, draw: r => r.poisson(4), models: { correct: x => (x < 0 ? 0 : S.cdf.poisson(x, 4)), shift: x => (x < 0 ? 0 : S.cdf.poisson(x, 5)), wide: x => (x < 0 ? 0 : nbCdf(x, 4, 0.5)), narrow: x => (x < 0 ? 0 : S.cdf.binom(x, 8, 0.5)) } },
  };

  Atlas.lab('inverse-transform-pit', {
    title: 'The PIT as a model check',
    blurb: 'Draw data from the true law and transform by a model CDF $F_0$. Under the correct continuous model $F_0(X)$ is uniform; a wrong mean tilts the histogram, a wrong spread bends it into a U or a hump. For a discrete law $F(X)$ is never uniform; switch on the randomized PIT to fix it.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const truth = UI.select(controls, { label: 'True law of $X$', options: Object.keys(PIT).map(k => ({ value: k, label: PIT[k].label })), value: 'normal', onChange: regen });
      const model = UI.segmented(controls, { label: 'Model $F_0$', options: [{ value: 'correct', label: 'Correct' }, { value: 'shift', label: 'Mean too high' }, { value: 'wide', label: 'Too dispersed' }, { value: 'narrow', label: 'Too concentrated' }], value: 'correct', onChange: draw });
      const rnd = UI.toggle(controls, { label: 'Randomized PIT (discrete laws)', value: false, onChange: draw });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 100, max: 5000, step: 100, value: 1000, onChange: regen });
      UI.button(controls, 'New sample', regen, { primary: true });
      const p = new Plot(stage, { width: 640, height: 280, x: [0, 1], y: [0, 2.5], xlabel: 'F₀(X)', ylabel: 'density' });
      const sD = UI.stat(stats, 'KS distance to Unif(0,1)', { accent: 'accent' });
      const sP = UI.stat(stats, 'KS $p$-value (approx.)');
      const note = UI.note(stats, '');
      let x = [], v = [];
      function regen() { const T = PIT[truth.get()], n = Math.round(nS.get()); x = ctx.rng.array(n, () => T.draw(ctx.rng)); v = ctx.rng.array(n, () => ctx.rng.u()); draw(); }
      function draw() {
        const T = PIT[truth.get()], F0 = T.models[model.get()];
        const u = x.map((xi, i) => {
          const F = F0(xi);
          if (T.discrete && rnd.get()) { const Fm = F0(xi - 1); return Fm + v[i] * (F - Fm); }
          return F;
        });
        const hb = S.histogram(u, { bins: 25, lo: 0, hi: 1 });
        const ymax = Math.max(2, Math.max(...hb.map(b => b.density)) * 1.1);
        p.setDomain([0, 1], [0, ymax]).clear();
        p.bars(hb, { color: 's1', opacity: 0.6 });
        p.hline(1, { color: 'ink', dash: '4 4', width: 1.5 });
        const s = u.slice().sort((a, b) => a - b), n = s.length;
        let D = 0; for (let i = 0; i < n; i++) D = Math.max(D, (i + 1) / n - s[i], s[i] - i / n);
        sD.set(fmt(D, 3));
        const pv = ksP(D, n);
        sP.set(pv < 1e-4 ? '< 0.0001' : fmt(pv, 3), pv < 0.05 ? '<span style="color:var(--bad)">reject uniformity</span>' : '<span style="color:var(--good)">consistent with uniform</span>');
        note.set(T.discrete && !rnd.get() ? 'Discrete $X$: $F_0(X)$ takes only a few values, so the PIT is lumpy even under the correct model.' : model.get() === 'correct' ? 'Correct model: the PIT values are (exactly) uniform.' : 'Misspecified model: the shape of the PIT histogram diagnoses how.');
      }
      regen();
    },
  });

  /* ================= rejection-sampling ================= */

  const mixPdf = x => 0.6 * S.pdf.normal(x, -1.2, 0.5) + 0.4 * S.pdf.normal(x, 1.5, 0.8);
  const RJ_T = {
    normal: { label: 'Normal(0,1)', f: x => S.pdf.normal(x), range: [-5, 5] },
    mix: { label: 'Bimodal mixture', f: mixPdf, range: [-5, 5] },
    beta: { label: 'Beta(2.5, 4)', f: x => S.pdf.beta(x, 2.5, 4), range: [-0.05, 1.05], bounded: [0, 1] },
  };
  const RJ_G = {
    cauchy: { label: 'Cauchy', g: (x, s) => S.pdf.cauchy(x, 0, s), draw: (r, s) => r.cauchy(0, s) },
    laplace: { label: 'Laplace', g: (x, s) => S.pdf.laplace(x, 0, s), draw: (r, s) => r.laplace(0, s) },
    normal: { label: 'Normal', g: (x, s) => S.pdf.normal(x, 0, s), draw: (r, s) => r.normal(0, s) },
    unif: { label: 'Uniform(0,1)', g: x => (x >= 0 && x <= 1 ? 1 : 0), draw: r => r.u() },
  };

  Atlas.lab('rejection-sampling-darts', {
    title: 'Throwing darts under $M g$',
    blurb: 'Darts land uniformly under $Mg$; those under $f$ are kept. The acceptance rate converges to $1/M$. Set $M/M^\\star < 1$ and the envelope no longer covers $f$: the output silently follows $\\min(f, Mg)$ (dashed). A Normal proposal narrower than the target has $M^\\star = \\infty$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tgt = UI.select(controls, { label: 'Target $f$', options: Object.keys(RJ_T).map(k => ({ value: k, label: RJ_T[k].label })), value: 'normal', onChange: reset });
      const prop = UI.segmented(controls, { label: 'Proposal $g$', options: Object.keys(RJ_G).map(k => ({ value: k, label: RJ_G[k].label })), value: 'cauchy', onChange: reset });
      const sc = UI.slider(controls, { label: 'Proposal scale $s$', min: 0.3, max: 3, step: 0.05, value: 1, onChange: reset });
      const mr = UI.slider(controls, { label: 'Envelope $M / M^\\star$', min: 0.5, max: 4, step: 0.05, value: 1, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Throw', '❚❚ Pause']);
      UI.button(row, '+2000', () => { throwN(2000); draw(); });
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'height', title: 'Darts under M·g(x)' });
      const pR = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'density', title: 'Accepted draws vs f' });
      const sM = UI.stat(stats, '$M^\\star = \\sup f/g$');
      const sAcc = UI.stat(stats, 'Acceptance: empirical vs $1/M$', { accent: 'accent' });
      const sN = UI.stat(stats, 'Accepted / thrown');
      const sW = UI.stat(stats, 'Envelope valid?');
      let darts = [], acc = [], tries = 0, Mstar = 1, M = 1, gfun, gdraw;
      function setup() {
        const T = RJ_T[tgt.get()], G = RJ_G[prop.get()], s = sc.get();
        gfun = x => G.g(x, s); gdraw = r => G.draw(r, s);
        let mx = 0, uncovered = false;
        const [a, b] = T.bounded || [-12, 12];
        for (const x of S.linspace(a + 1e-6, b - 1e-6, 4001)) {
          const gx = gfun(x), fx = T.f(x);
          if (gx > 0) mx = Math.max(mx, fx / gx); else if (fx > 1e-12) uncovered = true;
        }
        // Support not covered, or ratio still growing at the edge of the grid: no finite envelope.
        const rat = x => { const gx = gfun(x); return gx > 0 ? T.f(x) / gx : 0; };
        const growing = !T.bounded && (rat(12) > 1.01 * rat(10) || rat(-12) > 1.01 * rat(-10));
        Mstar = uncovered || growing ? Infinity : mx;
        M = mr.get() * Math.max(mx, 1e-9);
      }
      function reset() { darts = []; acc = []; tries = 0; setup(); draw(); }
      function throwN(k) {
        const T = RJ_T[tgt.get()];
        for (let i = 0; i < k; i++) {
          const x = gdraw(ctx.rng), y = ctx.rng.u() * M * gfun(x), ok = y <= T.f(x);
          tries++; if (ok) acc.push(x);
          if (darts.length < 2500 || i >= k - 60) { darts.push([x, y, ok]); if (darts.length > 2500) darts.shift(); }
        }
      }
      function step() { throwN(25); draw(); if (tries > 200000) return false; }
      function draw() {
        const T = RJ_T[tgt.get()], [lo, hi] = T.range, xs = S.linspace(lo, hi, 400);
        const fv = xs.map(T.f), mg = xs.map(x => M * gfun(x));
        const ymax = Math.max(...fv) * 1.6;
        pL.setDomain([lo, hi], [0, ymax]).clear();
        pL.area(xs.map((x, i) => [x, mg[i]]), { color: 's2', opacity: 0.08 });
        pL.line(xs.map((x, i) => [x, mg[i]]), { color: 's2', width: 2 });
        if (M < Mstar) pL.area(xs.map((x, i) => [x, Math.max(fv[i], mg[i])]), { color: 'bad', opacity: 0.25, base: xs.map((x, i) => [x, mg[i]]) });
        const okp = [], badp = [];
        for (const [x, y, ok] of darts) (ok ? okp : badp).push([x, y]);
        pL.points(badp, { color: 'muted', r: 1.4, opacity: 0.35 });
        pL.points(okp, { color: 's1', r: 1.6, opacity: 0.7 });
        pL.line(xs.map((x, i) => [x, fv[i]]), { color: 'ink', width: 2 });
        pL.legend([{ label: 'M·g', color: 's2' }, { label: 'f', color: 'ink' }]);
        // right
        const hb = S.histogram(acc, { bins: 40, lo, hi });
        const clip = xs.map((x, i) => Math.min(fv[i], mg[i])), dx = (hi - lo) / 399;
        const zc = S.sum(clip) * dx;
        const ymR = Math.max(...fv, ...hb.map(b => b.density)) * 1.15;
        pR.setDomain([lo, hi], [0, ymR]).clear();
        pR.bars(hb, { color: 's1', opacity: 0.5 });
        pR.line(xs.map((x, i) => [x, fv[i]]), { color: 'ink', width: 2 });
        if (M < Mstar) pR.line(xs.map((x, i) => [x, clip[i] / zc]), { color: 'bad', width: 2, dash: true });
        pR.legend(M < Mstar ? [{ label: 'f', color: 'ink' }, { label: 'min(f, Mg) normalized', color: 'bad', dash: true }] : [{ label: 'f', color: 'ink' }, { label: 'accepted', color: 's1', fill: true }]);
        sM.set(Number.isFinite(Mstar) ? fmt(Mstar, 3) : '∞', Number.isFinite(Mstar) ? `using M = ${fmt(M, 3)}` : 'f/g unbounded: proposal tails too light');
        sAcc.set(tries ? pct(acc.length / tries) : '—', `1/M = ${pct(1 / M)}`);
        sN.set(`${acc.length} / ${tries}`);
        sW.set(M >= Mstar ? '<span style="color:var(--good)">yes: f ≤ Mg</span>' : '<span style="color:var(--bad)">no: output is biased</span>');
      }
      reset();
    },
  });

  /* ---------- adaptive rejection sampling ---------- */
  const ARS_T = {
    normal: { label: 'Normal(0,1)', h: x => -x * x / 2, dh: x => -x, lo: -Infinity, hi: Infinity, init: [-1.6, 2.1], range: [-4, 4], Z: Math.sqrt(2 * Math.PI) },
    gamma: { label: 'Gamma(3,1)', h: x => 2 * Math.log(x) - x, dh: x => 2 / x - 1, lo: 0, hi: Infinity, init: [0.6, 5.5], range: [0, 12], Z: 2 },
    logistic: { label: 'Logistic', h: x => -x - 2 * Math.log1p(Math.exp(-x)), dh: x => -1 + 2 / (1 + Math.exp(x)), lo: -Infinity, hi: Infinity, init: [-2.8, 2.2], range: [-7, 7], Z: 1 },
  };
  function buildHull(T, X) {
    const k = X.length, H = X.map(T.h), D = X.map(T.dh), z = [T.lo];
    for (let j = 0; j < k - 1; j++) {
      const den = D[j] - D[j + 1];
      z.push(Math.abs(den) < 1e-12 ? (X[j] + X[j + 1]) / 2 : (H[j + 1] - H[j] - X[j + 1] * D[j + 1] + X[j] * D[j]) / den);
    }
    z.push(T.hi);
    const mass = [];
    for (let j = 0; j < k; j++) {
      const a = z[j], b = z[j + 1], s = D[j], ua = H[j] + s * (a - X[j]), ub = H[j] + s * (b - X[j]);
      let m;
      if (Math.abs(s) < 1e-10) m = Math.exp(H[j]) * (b - a);
      else m = ((Number.isFinite(b) ? Math.exp(ub) : 0) - (Number.isFinite(a) ? Math.exp(ua) : 0)) / s;
      mass.push(Math.max(0, m));
    }
    const total = S.sum(mass);
    const u = x => { let j = 0; while (j < k - 1 && x > z[j + 1]) j++; return H[j] + D[j] * (x - X[j]); };
    const l = x => { if (x < X[0] || x > X[k - 1]) return -Infinity; let j = 0; while (j < k - 2 && x > X[j + 1]) j++; const t = (x - X[j]) / (X[j + 1] - X[j]); return H[j] + t * (H[j + 1] - H[j]); };
    function sample(r) {
      let v = r.u() * total, j = 0;
      while (j < k - 1 && v > mass[j]) { v -= mass[j]; j++; }
      const a = z[j], b = z[j + 1], s = D[j], w = r.u();
      if (Math.abs(s) < 1e-10) return a + w * (b - a);
      if (Number.isFinite(a)) return a + Math.log1p(w * Math.expm1(s * (b - a))) / s;
      return b + Math.log(w) / s;
    }
    return { u, l, sample, total, z };
  }

  Atlas.lab('rejection-sampling-ars', {
    title: 'Adaptive rejection sampling builds its own envelope',
    blurb: 'Left, on the log scale: $\\log f$ (black), the tangent upper hull (envelope) and the chord lower hull (squeeze). Every time the squeeze fails, $\\log f$ is evaluated and the point becomes a new abscissa; watch the acceptance rate climb toward 1 and evaluations of $f$ become rare.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tgt = UI.segmented(controls, { label: 'Log-concave target', options: Object.keys(ARS_T).map(k => ({ value: k, label: ARS_T[k].label })), value: 'gamma', onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Sample', '❚❚ Pause']);
      UI.button(row, 'Draw 1', () => { drawOne(); draw(); });
      UI.button(row, '+500', () => { for (let i = 0; i < 500; i++) drawOne(); draw(); });
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'log density', title: 'Hulls on the log scale' });
      const pR = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'unnormalized density', title: 'Envelope, squeeze and draws' });
      const sK = UI.stat(stats, 'Abscissae $k$');
      const sA = UI.stat(stats, 'Acceptance prob. of envelope', { accent: 'accent' });
      const sQ = UI.stat(stats, 'Squeeze accepts (current)');
      const sE = UI.stat(stats, 'Evaluations of $f$ per draw');
      let X, hull, out, evals, last;
      function reset() { const T = ARS_T[tgt.get()]; X = T.init.slice(); hull = buildHull(T, X); out = []; evals = 0; last = null; draw(); }
      function drawOne() {
        const T = ARS_T[tgt.get()];
        for (let tries = 0; tries < 1000; tries++) {
          const x = hull.sample(ctx.rng), w = ctx.rng.u(), ux = hull.u(x);
          if (!(x > T.lo && x < T.hi) || !Number.isFinite(x)) continue;
          if (Math.log(w) <= hull.l(x) - ux) { out.push(x); last = { x, kind: 'squeeze' }; return; }
          evals++;
          const ok = Math.log(w) <= T.h(x) - ux;
          if (X.length < 40) { X.push(x); X.sort((a, b) => a - b); hull = buildHull(T, X); }
          if (ok) { out.push(x); last = { x, kind: 'evaluated' }; return; }
          last = { x, kind: 'rejected' };
        }
      }
      function step() { drawOne(); draw(); if (out.length > 5000) return false; }
      function draw() {
        const T = ARS_T[tgt.get()], [lo, hi] = T.range, xs = S.linspace(lo + 1e-4, hi, 500);
        const hv = xs.map(T.h), uv = xs.map(hull.u), lv = xs.map(hull.l);
        const hmax = Math.max(...hv);
        pL.setDomain([lo, hi], [hmax - 7, hmax + 1.2]).clear();
        pL.line(xs.map((x, i) => [x, uv[i]]), { color: 's2', width: 2 });
        pL.line(xs.map((x, i) => [x, lv[i]]), { color: 's3', width: 2 });
        pL.line(xs.map((x, i) => [x, hv[i]]), { color: 'ink', width: 1.6 });
        for (const x of X) pL.dot(x, T.h(x), { color: 's2', r: 3.5 });
        if (last) pL.vline(last.x, { color: last.kind === 'rejected' ? 'bad' : 's1', width: 1.2 });
        pL.legend([{ label: 'upper hull u', color: 's2' }, { label: 'lower hull l', color: 's3' }, { label: 'log f', color: 'ink' }]);
        const fe = hv.map(Math.exp), ue = uv.map(Math.exp), le = lv.map(Math.exp);
        const ym = Math.max(...fe) * 1.6;
        pR.setDomain([lo, hi], [0, ym]).clear();
        if (out.length) { const hb = S.histogram(out, { bins: 40, lo, hi }); for (const b of hb) b.y = b.density * T.Z; pR.bars(hb, { color: 's1', opacity: 0.4 }); }
        pR.line(xs.map((x, i) => [x, ue[i]]), { color: 's2', width: 2 });
        pR.area(xs.map((x, i) => [x, le[i]]), { color: 's3', opacity: 0.15 });
        pR.line(xs.map((x, i) => [x, fe[i]]), { color: 'ink', width: 1.6 });
        const dx = (hi - lo) / 499, squeezeMass = S.sum(le) * dx;
        sK.set(String(X.length));
        sA.set(pct(T.Z / hull.total), `∫f / ∫exp(u) with ${out.length} draws so far`);
        sQ.set(pct(Math.min(1, squeezeMass / hull.total)), 'of proposals accepted without evaluating f');
        sE.set(out.length ? fmt(evals / out.length, 3) : '—', last ? 'last proposal: ' + last.kind : '');
      }
      reset();
    },
  });

  /* ================= transformation-methods ================= */

  const BANDS = ['s1', 's2', 's3', 's4', 's5'];
  Atlas.lab('transformation-methods-box-muller', {
    title: 'Box–Muller: from the unit square to the plane',
    blurb: 'Left: the uniform inputs, coloured in five bands of $U_1$ (radius) or $U_2$ (angle). Right: their images. Bands of $U_1$ become rings, bands of $U_2$ become wedges. Feed the map a poor LCG and its lattice of pairs turns into spirals: the Neave effect.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const meth = UI.segmented(controls, { label: 'Method', options: [{ value: 'bm', label: 'Box–Muller' }, { value: 'polar', label: 'Polar (Marsaglia)' }], value: 'bm', onChange: regen });
      const src = UI.segmented(controls, { label: 'Uniform source', options: [{ value: 'good', label: 'Good PRNG' }, { value: 'lcg', label: 'Poor LCG (m = 4096, a = 65)' }], value: 'good', onChange: regen });
      const by = UI.segmented(controls, { label: 'Colour by', options: [{ value: 'r', label: 'U₁ (radius)' }, { value: 't', label: 'U₂ (angle)' }], value: 'r', onChange: draw });
      const nS = UI.slider(controls, { label: 'Pairs', min: 500, max: 4000, step: 100, value: 2000, onChange: regen });
      UI.button(controls, 'Resample', regen, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [0, 1], y: [0, 1], xlabel: 'U₁', ylabel: 'U₂', equal: true, title: 'Inputs' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [-4, 4], y: [-4, 4], xlabel: 'Z₁', ylabel: 'Z₂', equal: true, title: 'Outputs' });
      const sM = UI.stat(stats, 'Mean, var of $Z_1$');
      const sC = UI.stat(stats, '$\\Corr(Z_1, Z_2)$');
      const sT = UI.stat(stats, '$P(|Z_1| > 2)$: observed vs 4.55%');
      const sA = UI.stat(stats, 'Polar acceptance (vs $\\pi/4$)');
      let pairs = [], tries = 0;
      function regen() {
        const n = Math.round(nS.get());
        let x = 1;
        const u = src.get() === 'lcg' ? () => { x = (65 * x + 1) % 4096; return (x + 0.5) / 4096; } : () => ctx.rng.u();
        pairs = []; tries = 0;
        let nok = 0;
        while (nok < n && tries < 20 * n) {
          if (meth.get() === 'bm') {
            const u1 = u(), u2 = u(); tries++;
            const r = Math.sqrt(-2 * Math.log(u1)), t = 2 * Math.PI * u2;
            pairs.push({ in: [u1, u2], z: [r * Math.cos(t), r * Math.sin(t)], cr: u1, ct: u2, ok: true }); nok++;
          } else {
            const v1 = 2 * u() - 1, v2 = 2 * u() - 1, s = v1 * v1 + v2 * v2; tries++;
            if (s >= 1 || s === 0) { pairs.push({ in: [v1, v2], ok: false }); continue; }
            const f = Math.sqrt(-2 * Math.log(s) / s), ang = (Math.atan2(v2, v1) / (2 * Math.PI) + 1) % 1;
            pairs.push({ in: [v1, v2], z: [v1 * f, v2 * f], cr: s, ct: ang, ok: true }); nok++;
          }
        }
        draw();
      }
      function draw() {
        const polar = meth.get() === 'polar';
        pL.setDomain(polar ? [-1, 1] : [0, 1], polar ? [-1, 1] : [0, 1]).clear();
        if (polar) pL.ellipse(0, 0, [[1, 0], [0, 1]], 1, { color: 'ink', width: 1.2 });
        pR.clear();
        const groupsL = BANDS.map(() => []), groupsR = BANDS.map(() => []), rej = [];
        const ok = pairs.filter(p => p.ok);
        for (const p of pairs) {
          if (!p.ok) { rej.push(p.in); continue; }
          const c = Math.min(4, Math.floor(5 * (by.get() === 'r' ? p.cr : p.ct)));
          groupsL[c].push(p.in); groupsR[c].push(p.z);
        }
        pL.points(rej, { color: 'muted', r: 1.3, opacity: 0.35 });
        BANDS.forEach((c, i) => { pL.points(groupsL[i], { color: c, r: 1.5, opacity: 0.75 }); pR.points(groupsR[i], { color: c, r: 1.5, opacity: 0.75 }); });
        const z1 = ok.map(p => p.z[0]), z2 = ok.map(p => p.z[1]);
        sM.set(`${fmt(S.mean(z1), 2)}, ${fmt(S.variance(z1), 3)}`, 'target 0, 1');
        sC.set(fmt(S.cor(z1, z2), 3), 'target 0');
        const tail = z1.filter(z => Math.abs(z) > 2).length / z1.length;
        sT.set(pct(tail, 2), `MCSE ≈ ${pct(Math.sqrt(0.0455 * 0.9545 / z1.length), 2)}`);
        sA.set(polar ? pct(ok.length / tries) : '—', polar ? 'π/4 = 78.5%' : 'Box–Muller never rejects');
      }
      regen();
    },
  });

  const ROU = {
    normal: { label: 'Normal', h: x => Math.exp(-x * x / 2), a: 1, bm: -Math.sqrt(2 / Math.E), bp: Math.sqrt(2 / Math.E), Z: Math.sqrt(2 * Math.PI), range: [-4, 4], xr: [-6, 6] },
    cauchy: { label: 'Cauchy', h: x => 1 / (1 + x * x), a: 1, bm: -1, bp: 1, Z: Math.PI, range: [-8, 8], xr: [-400, 400] },
    exp: { label: 'Exponential', h: x => (x < 0 ? 0 : Math.exp(-x)), a: 1, bm: 0, bp: 2 / Math.E, Z: 1, range: [0, 6], xr: [0, 40] },
    gamma: { label: 'Gamma(3)', h: x => (x < 0 ? 0 : x * x * Math.exp(-x)), a: 2 / Math.E, bm: 0, bp: 16 / Math.E ** 2, Z: 2, range: [0, 12], xr: [0, 60] },
  };
  Atlas.lab('transformation-methods-rou', {
    title: 'Ratio-of-uniforms: a region whose slopes are the sample',
    blurb: 'Points are thrown uniformly in the bounding rectangle; those inside $C_h = \\{0 < u \\le \\sqrt{h(v/u)}\\}$ are kept and return $X = V/U$, the slope of the ray through the point. For the Cauchy, $C_h$ is exactly a half-disc.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tgt = UI.segmented(controls, { label: 'Target $h$', options: Object.keys(ROU).map(k => ({ value: k, label: ROU[k].label })), value: 'normal', onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Throw', '❚❚ Pause']);
      UI.button(row, '+3000', () => { throwN(3000); draw(); });
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, xlabel: 'u', ylabel: 'v', equal: true, title: 'The region C_h in its rectangle' });
      const pR = new Plot(stage, { width: 420, height: 300, xlabel: 'x = v/u', ylabel: 'density', title: 'Accepted ratios vs target' });
      const sA = UI.stat(stats, 'Acceptance: empirical', { accent: 'accent' });
      const sT = UI.stat(stats, 'Theory $\\tfrac12\\int h / [a(b_+ - b_-)]$');
      const sN = UI.stat(stats, 'Accepted / thrown');
      let pts = [], xs = [], tries = 0;
      function reset() { pts = []; xs = []; tries = 0; draw(); }
      function throwN(k) {
        const T = ROU[tgt.get()];
        for (let i = 0; i < k; i++) {
          const u = T.a * ctx.rng.u(), v = T.bm + (T.bp - T.bm) * ctx.rng.u(); tries++;
          const ok = u > 0 && u * u <= T.h(v / u);
          if (ok) xs.push(v / u);
          if (pts.length < 3000 || i >= k - 50) { pts.push([u, v, ok]); if (pts.length > 3000) pts.shift(); }
        }
      }
      function step() { throwN(40); draw(); if (tries > 300000) return false; }
      function draw() {
        const T = ROU[tgt.get()];
        pL.setDomain([-0.05 * T.a, 1.05 * T.a], [T.bm - 0.05 * (T.bp - T.bm), T.bp + 0.05 * (T.bp - T.bm)]).clear();
        pL.rect(0, T.bm, T.a, T.bp, { color: 'faint', opacity: 0.25, stroke: 'muted' });
        const g = [];
        const tt = S.linspace(-1, 1, 801);
        for (const t of tt) { const x = T.xr[0] + (T.xr[1] - T.xr[0]) * (0.5 + 0.5 * Math.sign(t) * t * t); const r = Math.sqrt(T.h(x)); g.push([r, x * r]); }
        g.sort((p, q) => Math.atan2(p[1], p[0]) - Math.atan2(q[1], q[0]));
        pL.polygon([[0, 0], ...g], { color: 's2', opacity: 0.18, stroke: 's2', width: 1.5 });
        const okp = [], bad = [];
        for (const [u, v, ok] of pts) (ok ? okp : bad).push([u, v]);
        pL.points(bad, { color: 'muted', r: 1.3, opacity: 0.35 });
        pL.points(okp, { color: 's1', r: 1.5, opacity: 0.7 });
        const [lo, hi] = T.range, gx = S.linspace(lo, hi, 300), dens = gx.map(x => T.h(x) / T.Z);
        const hb = S.histogram(xs, { bins: 45, lo, hi });
        pR.setDomain([lo, hi], [0, Math.max(...dens, ...hb.map(b => b.density)) * 1.15]).clear();
        pR.bars(hb, { color: 's1', opacity: 0.5 });
        pR.line(gx.map((x, i) => [x, dens[i]]), { color: 'ink', width: 2 });
        const th = 0.5 * T.Z / (T.a * (T.bp - T.bm));
        sA.set(tries ? pct(xs.length / tries) : '—');
        sT.set(pct(th));
        sN.set(`${xs.length} / ${tries}`, xs.length ? `${pct(xs.filter(x => x < lo || x > hi).length / xs.length)} of draws off-axis` : '');
      }
      reset();
    },
  });

  /* ================= monte-carlo-integration ================= */

  const MCI = {
    hit: { label: 'π, hit-or-miss', dim: 2, h: r => { const a = r.u(), b = r.u(); return a * a + b * b <= 1 ? 4 : 0; }, I: Math.PI, sigma2: Math.PI * (4 - Math.PI) },
    crude: { label: 'π, crude 4√(1−u²)', dim: 1, h: r => { const u = r.u(); return 4 * Math.sqrt(1 - u * u); }, I: Math.PI, sigma2: 32 / 3 - Math.PI ** 2 },
    exp: { label: '∫ eᵘ du = e − 1', dim: 1, h: r => Math.exp(r.u()), I: Math.E - 1, sigma2: (Math.E ** 2 - 1) / 2 - (Math.E - 1) ** 2 },
    inf: { label: '∫ u^(−1/2) du = 2 (infinite variance)', dim: 1, h: r => 1 / Math.sqrt(1 - r.u()), I: 2, sigma2: Infinity },
  };
  Atlas.lab('monte-carlo-integration-convergence', {
    title: 'The $n^{-1/2}$ law, and when it breaks',
    blurb: 'Left: one running estimate with its $\\pm 2\\,\\widehat{\\se}$ band. Right: absolute errors of 20 independent runs (faint) and their RMS (bold) on log–log axes, against $\\sigma/\\sqrt n$. Compare hit-or-miss with crude for $\\pi$, then try the infinite-variance integrand: the error bars lie and the slope is wrong.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ig = UI.select(controls, { label: 'Integral', options: Object.keys(MCI).map(k => ({ value: k, label: MCI[k].label })), value: 'hit', onChange: run });
      const NS = UI.slider(controls, { label: 'Draws per run $n$', min: 1000, max: 100000, value: 20000, log: true, fmt: v => Math.round(v).toLocaleString(), onChange: run });
      UI.button(controls, 'New runs', run, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, logx: true, x: [1, 1e4], y: [0, 1], xlabel: 'n', ylabel: 'estimate', title: 'Running estimate ± 2 se' });
      const pR = new Plot(stage, { width: 420, height: 280, logx: true, logy: true, x: [1, 1e4], y: [1e-5, 10], xlabel: 'n', ylabel: '|error|', title: 'Error of 20 runs (log–log)' });
      const sE = UI.stat(stats, 'Estimate ± se (run 1)', { accent: 'accent' });
      const sS = UI.stat(stats, '$\\sigma^2 = \\Var h$');
      const sC = UI.stat(stats, 'Runs whose 95% CI covers');
      const sSl = UI.stat(stats, 'Fitted log–log slope of RMS');
      function run() {
        const M = MCI[ig.get()], N = Math.round(NS.get()), K = 20;
        const cps = []; for (let e = 0; e <= Math.log10(N) + 1e-9; e += 0.05) { const c = Math.round(10 ** e); if (!cps.length || c > cps[cps.length - 1]) cps.push(c); }
        if (cps[cps.length - 1] !== N) cps.push(N);
        const errs = [], path = [];
        let cover = 0, last;
        for (let k = 0; k < K; k++) {
          let s = 0, s2 = 0, j = 0; const e = [];
          for (let i = 1; i <= N; i++) {
            const v = M.h(ctx.rng); s += v; s2 += v * v;
            if (i === cps[j]) {
              const m = s / i, sd = i > 1 ? Math.sqrt(Math.max(0, (s2 - i * m * m) / (i - 1))) : 0;
              e.push(Math.abs(m - M.I));
              if (k === 0) path.push([i, m, sd / Math.sqrt(i)]);
              j++;
            }
          }
          errs.push(e);
          const m = s / N, se = Math.sqrt(Math.max(0, (s2 - N * m * m) / (N - 1)) / N);
          if (Math.abs(m - M.I) <= 1.96 * se) cover++;
          if (k === 0) last = { m, se };
        }
        const rms = cps.map((_, j) => Math.sqrt(S.mean(errs.map(e => e[j] * e[j]))));
        // left panel
        const mid = path.filter(p => p[0] >= 30);
        const lo = Math.min(...mid.map(p => p[1] - 2 * p[2]), M.I), hi = Math.max(...mid.map(p => p[1] + 2 * p[2]), M.I), pad = (hi - lo) * 0.1 || 0.1;
        pL.setDomain([1, N], [lo - pad, hi + pad]).clear();
        pL.area(path.map(p => [p[0], p[1] + 2 * p[2]]), { color: 's1', opacity: 0.15, base: path.map(p => [p[0], p[1] - 2 * p[2]]) });
        pL.line(path.map(p => [p[0], p[1]]), { color: 's1', width: 1.8 });
        pL.hline(M.I, { color: 'ink', width: 1.4 });
        // right panel
        const floor = 1e-6;
        pR.setDomain([1, N], [1e-5, 10]).clear();
        for (const e of errs) pR.line(cps.map((c, j) => [c, Math.max(floor, e[j])]), { color: 's2', width: 1, opacity: 0.25 });
        pR.line(cps.map((c, j) => [c, Math.max(floor, rms[j])]), { color: 's2', width: 2.6 });
        if (Number.isFinite(M.sigma2)) pR.line(cps.map(c => [c, Math.sqrt(M.sigma2 / c)]), { color: 'ink', width: 1.5, dash: true });
        pR.legend([{ label: 'RMS error', color: 's2' }, { label: 'σ/√n', color: 'ink', dash: true }], { left: true });
        // slope over the second half of checkpoints (log scale)
        const idx = cps.map((c, j) => j).filter(j => cps[j] >= Math.sqrt(N) && rms[j] > 0);
        const lx = idx.map(j => Math.log10(cps[j])), ly = idx.map(j => Math.log10(rms[j]));
        const mx = S.mean(lx), my = S.mean(ly);
        const slope = S.sum(lx.map((x, i) => (x - mx) * (ly[i] - my))) / S.sum(lx.map(x => (x - mx) ** 2));
        sE.set(`${fmt(last.m, 4)} ± ${fmt(last.se, 2)}`, `true ${fmt(M.I, 5)}`);
        sS.set(Number.isFinite(M.sigma2) ? fmt(M.sigma2, 3) : '∞', Number.isFinite(M.sigma2) ? '' : 'CLT does not apply');
        sC.set(`${cover} / ${K}`, 'nominal 95%');
        sSl.set(fmt(slope, 2), 'theory −0.5 when σ² < ∞');
      }
      run();
    },
  });

  Atlas.lab('monte-carlo-integration-dimension', {
    title: 'Monte Carlo vs the midpoint rule as dimension grows',
    blurb: 'Integrate $\\prod_{j=1}^d \\tfrac{\\pi}{2}\\sin(\\pi x_j)$ (exact value 1) over $[0,1]^d$. The product midpoint rule with $m$ nodes per axis uses $N = m^d$ points and has error $\\propto N^{-2/d}$; Monte Carlo has RMSE $\\sigma/\\sqrt N$ with $\\sigma^2 = (\\pi^2/8)^d - 1$. Raise $d$ and watch the crossover move left while $\\sigma$ grows.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dS = UI.slider(controls, { label: 'Dimension $d$', min: 1, max: 10, step: 1, value: 3, onChange: run });
      const NS = UI.slider(controls, { label: 'Largest budget $N$', min: 1000, max: 200000, value: 100000, log: true, fmt: v => Math.round(v).toLocaleString(), onChange: run });
      UI.button(controls, 'New MC run', run, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, logx: true, logy: true, x: [1, 1e5], y: [1e-9, 10], xlabel: 'N (function evaluations)', ylabel: '|error|' });
      const sMid = UI.stat(stats, 'Midpoint slope');
      const sMC = UI.stat(stats, 'MC slope');
      const sSig = UI.stat(stats, '$\\sigma^2$ of the integrand');
      const sX = UI.stat(stats, 'Crossover $N$', { accent: 'accent' });
      function run() {
        const d = Math.round(dS.get()), N = Math.round(NS.get());
        const sig2 = (Math.PI ** 2 / 8) ** d - 1;
        const mid = [];
        for (let m = 1; m ** d <= N * 1.0001; m++) { const q = (Math.PI / (2 * m)) / Math.sin(Math.PI / (2 * m)); mid.push([m ** d, Math.abs(q ** d - 1)]); if (d === 1 && m > 3000) break; }
        // MC realized error at checkpoints
        const cps = []; for (let e = 1; e <= Math.log10(N) + 1e-9; e += 0.1) cps.push(Math.round(10 ** e));
        const mc = []; let s = 0, j = 0;
        for (let i = 1; i <= N && j < cps.length; i++) {
          let v = 1; for (let k = 0; k < d; k++) v *= (Math.PI / 2) * Math.sin(Math.PI * ctx.rng.u());
          s += v;
          if (i === cps[j]) { mc.push([i, Math.max(1e-12, Math.abs(s / i - 1))]); j++; }
        }
        const ylo = 1e-9, yhi = 10;
        p.setDomain([1, N], [ylo, yhi]).clear();
        p.line([[1, Math.sqrt(sig2)], [N, Math.sqrt(sig2 / N)]], { color: 's2', width: 2.2 });
        p.points(mc, { color: 's2', r: 2.5, opacity: 0.7 });
        p.line(mid.map(q => [q[0], Math.max(ylo, q[1])]), { color: 's1', width: 2.2 });
        p.points(mid.map(q => [q[0], Math.max(ylo, q[1])]), { color: 's1', r: 3 });
        p.legend([{ label: 'midpoint rule (m^d points)', color: 's1' }, { label: 'MC RMSE σ/√N', color: 's2' }, { label: 'MC realized |error|', color: 's2', fill: true }]);
        // crossover: midpoint error ~ C N^{-2/d}; MC ~ sigma N^{-1/2}
        let cross = null;
        for (const [n, e] of mid) if (n > 1 && e > Math.sqrt(sig2 / n)) cross = n;
        const last = mid[mid.length - 1];
        sMid.set(fmt(-2 / d, 3), 'theory −2/d');
        sMC.set('−0.5', 'any d');
        sSig.set(fmt(sig2, 3), d > 1 ? `grows like (π²/8)^d = ${fmt((Math.PI ** 2 / 8) ** d, 3)}` : '');
        if (cross === null) sX.set('none', 'midpoint wins over the whole range');
        else if (cross === last[0]) sX.set('> ' + Math.round(last[0]).toLocaleString(), 'MC better at every budget shown');
        else sX.set('≈ ' + Math.round(cross).toLocaleString(), 'midpoint wins only beyond this N');
      }
      run();
    },
  });

  /* ================= importance-sampling ================= */

  // Accurate normal upper tail via Simpson integration (avoids cancellation in 1 − Φ).
  function normTail(c) {
    if (c < 0) return 1 - normTail(-c);
    const b = c + 20, n = 4000, h = (b - c) / n;
    let s = S.pdf.normal(c) + S.pdf.normal(b);
    for (let i = 1; i < n; i++) s += (i % 2 ? 4 : 2) * S.pdf.normal(c + i * h);
    return s * h / 3;
  }
  Atlas.lab('importance-sampling-tail', {
    title: 'A rare tail probability by importance sampling',
    blurb: 'Estimate $p = P(Z > c)$ for $Z \\sim N(0,1)$. The dashed curve is the zero-variance proposal $g^\\star \\propto \\ind\\{x > c\\}\\phi(x)$. Slide the proposal toward it: the relative error collapses. With shift $0$ this is crude Monte Carlo, which usually sees no hits at all.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const fam = UI.segmented(controls, { label: 'Proposal family', options: [{ value: 'shift', label: 'Mean shift N(θ, 1)' }, { value: 'expo', label: 'c + Exp(rate θ)' }], value: 'shift', onChange: run });
      const cS = UI.slider(controls, { label: 'Threshold $c$', min: 1, max: 6, step: 0.1, value: 4, onChange: run });
      const tS = UI.slider(controls, { label: 'Proposal parameter $\\theta$', min: 0, max: 8, step: 0.05, value: 2, onInput: run });
      const nS = UI.slider(controls, { label: 'Draws $n$', min: 100, max: 100000, value: 5000, log: true, fmt: v => Math.round(v).toLocaleString(), onChange: run });
      UI.button(controls, 'Resample', run, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, xlabel: 'x', ylabel: 'density', title: 'Target, optimal and chosen proposals' });
      const pR = new Plot(stage, { width: 420, height: 280, xlabel: 'log₁₀( h·w / p )', ylabel: 'density', title: 'Contributions of the hits' });
      const sE = UI.stat(stats, 'Estimate $\\hat p$ ± se', { accent: 'accent' });
      const sP = UI.stat(stats, 'True $p$');
      const sR = UI.stat(stats, 'Relative error: IS vs crude');
      const sESS = UI.stat(stats, 'Hits / ESS of hits');
      function run() {
        const c = cS.get(), th = Math.max(fam.get() === 'expo' ? 0.05 : 0, tS.get()), n = Math.round(nS.get()), expo = fam.get() === 'expo';
        const p = normTail(c);
        const g = x => (expo ? (x >= c ? th * Math.exp(-th * (x - c)) : 0) : S.pdf.normal(x, th, 1));
        const draw = () => (expo ? c + ctx.rng.exp(th) : ctx.rng.normal(th, 1));
        // theoretical second moment E_g[h^2 w^2] = ∫_c^∞ φ²/g, by Simpson on [c, c+12]
        let m2 = 0; { const b = c + 12, K = 3000, hh = (b - c) / K; for (let i = 0; i <= K; i++) { const x = c + i * hh, wgt = i === 0 || i === K ? 1 : i % 2 ? 4 : 2; const gx = g(x); m2 += wgt * (gx > 0 ? S.pdf.normal(x) ** 2 / gx : Infinity); } m2 *= hh / 3; }
        const relTh = Math.sqrt(Math.max(0, m2 - p * p) / n) / p;
        let s = 0, s2 = 0; const contrib = [];
        for (let i = 0; i < n; i++) {
          const x = draw();
          if (x > c) { const w = S.pdf.normal(x) / g(x); s += w; s2 += w * w; contrib.push(w); }
        }
        const est = s / n, se = Math.sqrt(Math.max(0, s2 / n - est * est) / (n - 1));
        // left plot
        const lo = -3, hi = Math.max(c + 3, th + 3), xs = S.linspace(lo, hi, 500);
        const gstar = xs.map(x => (x > c ? S.pdf.normal(x) / p : 0));
        const gv = xs.map(g);
        const ym = Math.min(Math.max(...gstar, ...gv, 0.4) * 1.1, 8);
        pL.setDomain([lo, hi], [0, ym]).clear();
        pL.line(xs.map((x, i) => [x, S.pdf.normal(x)]), { color: 'muted', width: 1.5 });
        pL.line(xs.map((x, i) => [x, gstar[i]]), { color: 'ink', width: 2, dash: true });
        pL.area(xs.map((x, i) => [x, gv[i]]), { color: 's1', opacity: 0.2 });
        pL.line(xs.map((x, i) => [x, gv[i]]), { color: 's1', width: 2 });
        pL.vline(c, { color: 'bad', width: 1.2 });
        pL.legend([{ label: 'f = φ', color: 'muted' }, { label: 'g* ∝ 1{x>c}φ', color: 'ink', dash: true }, { label: 'proposal g', color: 's1' }]);
        // right plot
        pR.setDomain([-3, 2], [0, 1]).clear();
        if (contrib.length) {
          const lv = contrib.map(w => Math.log10(w / p));
          const hb = S.histogram(lv, { bins: 50, lo: -3, hi: 2 });
          pR.setDomain([-3, 2], [0, Math.max(...hb.map(b => b.density)) * 1.15 || 1]);
          pR.bars(hb, { color: 's2', opacity: 0.6 });
        } else pR.text(-0.5, 0.5, 'no draws beyond c', { anchor: 'middle', color: 'muted' });
        pR.vline(0, { color: 'ink', width: 1.3 });
        const ess = s2 > 0 ? (s * s) / s2 : 0;
        sE.set(contrib.length ? `${est.toExponential(3)} ± ${se.toExponential(1)}` : '0 (no hits)');
        sP.set(p.toExponential(3));
        const crude = Math.sqrt((1 - p) / (n * p));
        sR.set(`${Number.isFinite(relTh) ? fmt(relTh, 3) : '∞'} vs ${fmt(crude, 3)}`, Number.isFinite(relTh) && relTh > 0 ? `efficiency gain ≈ ${fmt((crude / relTh) ** 2, 3)}×` : '');
        sESS.set(`${contrib.length} / ${fmt(ess, 4)}`, 'all draws with equal h·w would give ESS = hits');
      }
      run();
    },
  });

  Atlas.lab('importance-sampling-weights', {
    title: 'Proposal tails: when the weights have infinite variance',
    blurb: 'Estimate $E_f[X^2] = 1$ for $f = N(0,1)$ with proposal $g = N(0, \\sigma^2)$. For $\\sigma^2 > 1/2$ the running estimates settle inside the $\\pm 2\\sigma_g/\\sqrt n$ funnel. Push $\\sigma$ below $0.707$: the estimator is still unbiased, but the variance is infinite and the paths jump whenever a draw lands in $g$\'s thin tail.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const sg = UI.slider(controls, { label: 'Proposal sd $\\sigma$', min: 0.4, max: 3, step: 0.01, value: 0.6, onChange: run });
      const nS = UI.slider(controls, { label: 'Draws per path $n$', min: 1000, max: 100000, value: 20000, log: true, fmt: v => Math.round(v).toLocaleString(), onChange: run });
      UI.button(controls, 'New paths', run, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, logx: true, x: [10, 1e4], y: [0, 2], xlabel: 'n', ylabel: 'estimate of E[X²]', title: '8 running IS estimates' });
      const pR = new Plot(stage, { width: 420, height: 280, xlabel: 'log₁₀ w', ylabel: 'density', title: 'Weights of path 1' });
      const sV = UI.stat(stats, '$\\Var_g(h w)$ (theory)', { accent: 'accent' });
      const sEw = UI.stat(stats, '$E_g w^2$ and ESS/$n$ (theory)');
      const sESS = UI.stat(stats, 'ESS/$n$ observed (path 1)');
      const sMax = UI.stat(stats, 'Largest single share of $\\sum w$');
      function run() {
        const s = sg.get(), n = Math.round(nS.get()), K = 8;
        const a = 1 - 1 / (2 * s * s);
        const finite = a > 0;
        const m2 = finite ? (3 * s / (4 * Math.SQRT2)) * a ** -2.5 : Infinity, varhw = m2 - 1;
        const Ew2 = finite ? s * s / Math.sqrt(2 * s * s - 1) : Infinity;
        const cps = []; for (let e = 1; e <= Math.log10(n) + 1e-9; e += 0.04) { const c = Math.round(10 ** e); if (!cps.length || c > cps[cps.length - 1]) cps.push(c); }
        const paths = []; let w1 = [];
        for (let k = 0; k < K; k++) {
          let sum = 0, j = 0; const path = [];
          for (let i = 1; i <= n; i++) {
            const x = ctx.rng.normal(0, s), w = s * Math.exp(-x * x / 2 + x * x / (2 * s * s));
            sum += x * x * w;
            if (k === 0) w1.push(w);
            if (i === cps[j]) { path.push([i, sum / i]); j++; }
          }
          paths.push(path);
        }
        const allY = paths.flat().filter(p => p[0] >= 30).map(p => p[1]);
        const lo = Math.max(-0.2, Math.min(...allY, 0.5)), hi = Math.min(Math.max(...allY, 1.5), 6);
        pL.setDomain([10, n], [lo, hi]).clear();
        if (finite) {
          const band = cps.map(c => [c, 1 + 2 * Math.sqrt(varhw / c)]), bandL = cps.map(c => [c, 1 - 2 * Math.sqrt(varhw / c)]);
          pL.area(band, { color: 'ink', opacity: 0.08, base: bandL });
        }
        paths.forEach((pth, k) => pL.line(pth, { color: k === 0 ? 's1' : 's2', width: k === 0 ? 2 : 1.1, opacity: k === 0 ? 1 : 0.55 }));
        pL.hline(1, { color: 'ink', width: 1.3 });
        const lw = w1.map(w => Math.log10(w));
        const [e0, e1] = S.extent(lw);
        const hb = S.histogram(lw, { bins: 50, lo: Math.min(e0, -1), hi: Math.max(e1, 0.5) });
        pR.setDomain([Math.min(e0, -1), Math.max(e1, 0.5)], [0, Math.max(...hb.map(b => b.density)) * 1.15]).clear();
        pR.bars(hb, { color: 's1', opacity: 0.6 });
        const sw = S.sum(w1), sw2 = S.sum(w1.map(w => w * w)), mx = Math.max(...w1);
        sV.set(finite ? fmt(varhw, 3) : '∞', finite ? 'compare 2 for exact sampling (Var X²)' : 'σ² ≤ 1/2: no CLT, no valid error bar');
        sEw.set(finite ? `${fmt(Ew2, 3)} / ${pct(1 / Ew2)}` : '∞ / 0%');
        sESS.set(pct(sw * sw / sw2 / n));
        sMax.set(pct(mx / sw, 2), `largest log₁₀ w = ${fmt(Math.log10(mx), 3)}`);
      }
      run();
    },
  });

  /* ================= variance-reduction ================= */

  // Integrands on (0,1); true values by a fine midpoint rule.
  const VR = {
    exp: { label: 'eᵘ (monotone, nearly linear)', h: u => Math.exp(u) },
    call: { label: 'max(0, e^{0.3 + 0.5 Φ⁻¹(u)} − 1.4) (option-like)', h: u => Math.max(0, Math.exp(0.3 + 0.5 * S.qnorm(Math.min(1 - 1e-12, Math.max(1e-12, u)))) - 1.4) },
    sym: { label: '(u − ½)² (symmetric)', h: u => (u - 0.5) ** 2 },
    step: { label: '1{u > 0.8} (step)', h: u => (u > 0.8 ? 1 : 0) },
  };
  const vrTruth = {};
  function vrTrue(k) { if (vrTruth[k] == null) { let s = 0; const M = 200000; for (let i = 0; i < M; i++) s += VR[k].h((i + 0.5) / M); vrTruth[k] = s / M; } return vrTruth[k]; }
  const VR_M = [['Crude', 's1'], ['Antithetic', 's2'], ['Control (C = U)', 's3'], ['Stratified (10 strata)', 's5']];

  Atlas.lab('variance-reduction-compare', {
    title: 'Four estimators, one budget',
    blurb: 'Each dot is one estimate of $\\int_0^1 h(u)\\,du$ using exactly $n$ evaluations of $h$; each row is a method, repeated many times. The spread of a row is that method\'s Monte Carlo error. Monotone $h$ favours antithetics; nearly linear $h$ favours the control $U$; the symmetric integrand punishes antithetics.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ig = UI.select(controls, { label: 'Integrand $h$', options: Object.keys(VR).map(k => ({ value: k, label: VR[k].label })), value: 'exp', onChange: run });
      const nS = UI.slider(controls, { label: 'Evaluations per estimate $n$', min: 20, max: 1000, step: 20, value: 100, onChange: run });
      const RS = UI.slider(controls, { label: 'Repetitions', min: 100, max: 1000, step: 50, value: 300, onChange: run });
      UI.button(controls, 'Rerun', run, { primary: true });
      const p = new Plot(stage, { width: 640, height: 300, y: [-0.5, 3.5], yticks: 0, xlabel: 'estimate' });
      const tiles = VR_M.map(([name], i) => UI.stat(stats, name, i === 0 ? {} : {}));
      function run() {
        const key = ig.get(), h = VR[key].h, I = vrTrue(key), n = Math.round(nS.get()), R = Math.round(RS.get());
        const est = VR_M.map(() => []);
        for (let r = 0; r < R; r++) {
          // crude
          let s = 0; for (let i = 0; i < n; i++) s += h(ctx.rng.u()); est[0].push(s / n);
          // antithetic: n/2 pairs
          s = 0; for (let i = 0; i < n / 2; i++) { const u = ctx.rng.u(); s += h(u) + h(1 - u); } est[1].push(s / n);
          // control variate with estimated beta
          const ys = new Array(n), cs = new Array(n);
          for (let i = 0; i < n; i++) { cs[i] = ctx.rng.u(); ys[i] = h(cs[i]); }
          const my = S.mean(ys), mc = S.mean(cs);
          let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (cs[i] - mc) * (ys[i] - my); sxx += (cs[i] - mc) ** 2; }
          est[2].push(my - (sxx > 0 ? sxy / sxx : 0) * (mc - 0.5));
          // stratified: 10 equal strata, n/10 each
          const K = 10, nk = n / K; s = 0;
          for (let k = 0; k < K; k++) { let sk = 0; for (let i = 0; i < nk; i++) sk += h((k + ctx.rng.u()) / K); s += sk / nk / K; }
          est[3].push(s);
        }
        const vars = est.map(e => S.variance(e));
        const all = est.flat(), lo = S.quantile(all, 0.002), hi = S.quantile(all, 0.998), pad = (hi - lo) * 0.08 || 0.01;
        p.setDomain([Math.min(lo, I) - pad, Math.max(hi, I) + pad], [-0.6, 3.6]).clear();
        VR_M.forEach(([name, c], m) => {
          const y = 3 - m;
          p.points(est[m].map(v => [v, y + (ctx.rng.u() - 0.5) * 0.5]), { color: c, r: 1.8, opacity: 0.5 });
          const sd = Math.sqrt(vars[m]), mu = S.mean(est[m]);
          p.segment(mu - 2 * sd, y + 0.34, mu + 2 * sd, y + 0.34, { color: c, width: 3 });
          p.text(Math.min(lo, I) - pad, y + 0.3, name, { dx: 4, size: 11, color: 'muted' });
        });
        p.vline(I, { color: 'ink', width: 1.5, dash: false });
        VR_M.forEach(([name], m) => {
          const f = vars[0] / vars[m];
          tiles[m].set(m === 0 ? `sd ${fmt(Math.sqrt(vars[0]), 3)}` : `${fmt(f, 3)}×`, m === 0 ? `true value ${fmt(I, 5)}` : (f >= 1 ? 'variance reduction vs crude' : '<span style="color:var(--bad)">worse than crude</span>'));
        });
      }
      run();
    },
  });

  const NEY = {
    spike: { label: 'Smooth + sharp spike near 0.85', h: u => 0.3 * Math.sin(2 * Math.PI * u) + 4 * Math.exp(-(((u - 0.85) / 0.03) ** 2)) },
    exp: { label: 'e^{4u} (variation grows to the right)', h: u => Math.exp(4 * u) },
    step: { label: 'Step at 0.37', h: u => (u > 0.37 ? 1 : 0) },
  };
  Atlas.lab('variance-reduction-neyman', {
    title: 'Proportional vs Neyman allocation',
    blurb: 'Split $[0,1]$ into $K$ equal strata. Proportional allocation puts $n/K$ draws in each; Neyman allocation puts $n_k \\propto p_k\\sigma_k$, concentrating draws where $h$ varies. Variances below are exact (per-stratum moments by quadrature) for the rounded integer allocations.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ig = UI.select(controls, { label: 'Integrand', options: Object.keys(NEY).map(k => ({ value: k, label: NEY[k].label })), value: 'spike', onChange: draw });
      const KS = UI.slider(controls, { label: 'Strata $K$', min: 2, max: 25, step: 1, value: 10, onInput: draw });
      const nS = UI.slider(controls, { label: 'Total draws $n$', min: 50, max: 2000, step: 50, value: 200, onInput: draw });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, x: [0, 1], xlabel: 'u', ylabel: 'h(u)', title: 'Integrand and strata' });
      const pR = new Plot(stage, { width: 420, height: 280, x: [0, 1], xlabel: 'stratum', ylabel: 'draws n_k', title: 'Allocations' });
      const sC = UI.stat(stats, 'Crude: $\\sigma^2/n$');
      const sP = UI.stat(stats, 'Proportional');
      const sN = UI.stat(stats, 'Neyman', { accent: 'accent' });
      const sG = UI.stat(stats, 'Gains vs crude');
      function draw() {
        const h = NEY[ig.get()].h, K = Math.round(KS.get()), n = Math.round(nS.get());
        const M = 400, mu = [], sd = [];
        let tot = 0, tot2 = 0;
        for (let k = 0; k < K; k++) {
          let s = 0, s2 = 0;
          for (let i = 0; i < M; i++) { const v = h((k + (i + 0.5) / M) / K); s += v; s2 += v * v; }
          mu.push(s / M); sd.push(Math.sqrt(Math.max(0, s2 / M - (s / M) ** 2))); tot += s / M / K; tot2 += s2 / M / K;
        }
        const sig2 = tot2 - tot * tot, pk = 1 / K;
        const prop = Array(K).fill(n / K).map(Math.round).map(v => Math.max(1, v));
        const sumPS = S.sum(sd) * pk;
        const ney = sd.map(s => Math.max(1, Math.round(n * (sumPS > 0 ? pk * s / sumPS : pk))));
        const V = alloc => S.sum(alloc.map((nk, k) => pk * pk * sd[k] * sd[k] / nk));
        const vC = sig2 / n, vP = V(prop), vN = V(ney);
        const xs = S.linspace(0, 1, 600), hv = xs.map(h), [a, b] = S.extent(hv), pad = (b - a) * 0.08 || 0.1;
        pL.setDomain([0, 1], [a - pad, b + pad]).clear();
        for (let k = 1; k < K; k++) pL.vline(k / K, { color: 'faint', width: 1 });
        pL.line(xs.map((x, i) => [x, hv[i]]), { color: 'ink', width: 2 });
        mu.forEach((m, k) => pL.segment(k / K, m, (k + 1) / K, m, { color: 's2', width: 2 }));
        const ym = Math.max(...ney, ...prop) * 1.15;
        pR.setDomain([0, 1], [0, ym]).clear();
        pR.bars(prop.map((v, k) => ({ x0: k / K + 0.05 / K, x1: k / K + 0.5 / K, y: v })), { color: 's3', opacity: 0.6 });
        pR.bars(ney.map((v, k) => ({ x0: k / K + 0.5 / K, x1: k / K + 0.95 / K, y: v })), { color: 's1', opacity: 0.75 });
        pR.legend([{ label: 'proportional', color: 's3', fill: true }, { label: 'Neyman', color: 's1', fill: true }], { left: true });
        sC.set(vC.toExponential(2));
        sP.set(vP.toExponential(2), 'Σ p_k σ_k² / n');
        sN.set(vN.toExponential(2), '(Σ p_k σ_k)² / n');
        sG.set(`${fmt(vC / vP, 3)}× / ${fmt(vC / vN, 3)}×`, 'proportional / Neyman');
      }
      draw();
    },
  });

  /* ================= quasi-monte-carlo ================= */

  const PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
  function radInv(i, b) { let f = 1 / b, r = 0; while (i > 0) { r += f * (i % b); i = Math.floor(i / b); f /= b; } return r; }
  // 2-D Sobol: dim 1 = van der Corput base 2; dim 2 from primitive polynomial x + 1 (m = 1, 3, 5, 15, 17, 51, ...).
  const SOB_M = (() => { const m = [0, 1]; for (let k = 2; k <= 30; k++) m.push((2 * m[k - 1]) ^ m[k - 1]); return m; })();
  function sobol2(i) {
    let a = 0, b = 0;
    for (let k = 1; i > 0 && k <= 30; k++, i = Math.floor(i / 2)) if (i & 1) { a ^= 1 << (30 - k); b ^= SOB_M[k] << (30 - k); }
    return [a / 1073741824, b / 1073741824];
  }
  // Warnock's formula for the L2-star discrepancy.
  function l2star(P) {
    const n = P.length, d = P[0] ? P[0].length : 1;
    let s1 = 0; for (const x of P) { let pr = 1; for (const v of x) pr *= 1 - v * v; s1 += pr; }
    let s2 = 0; for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { let pr = 1; for (let k = 0; k < d; k++) pr *= 1 - Math.max(P[i][k], P[j][k]); s2 += pr; }
    return Math.sqrt(Math.max(0, 3 ** -d - 2 ** (1 - d) / n * s1 + s2 / (n * n)));
  }

  Atlas.lab('quasi-monte-carlo-points', {
    title: 'Random points vs low-discrepancy points',
    blurb: 'Watch points arrive one at a time. Random points clump and leave holes; Halton and Sobol fill the largest gaps first. With $n = 2^m$ Sobol points, every dyadic box of area $1/n$ (shown) holds exactly one point. Halton with bases 29 and 31 shows the correlation of large prime bases.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const kind = UI.select(controls, { label: 'Point set', options: [{ value: 'rand', label: 'Pseudo-random' }, { value: 'h23', label: 'Halton (2, 3)' }, { value: 'sob', label: 'Sobol (2-D)' }, { value: 'h2931', label: 'Halton (29, 31)' }], value: 'sob', onChange: reset });
      const mS = UI.slider(controls, { label: 'Target $n = 2^m$', min: 4, max: 10, step: 1, value: 7, fmt: v => `2^${v} = ${2 ** v}`, onChange: reset });
      const shift = UI.toggle(controls, { label: 'Random shift (Cranley–Patterson)', value: false, onChange: () => { delta = [ctx.rng.u(), ctx.rng.u()]; draw(); } });
      const boxes = UI.toggle(controls, { label: 'Show dyadic boxes of area 1/n', value: true, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Add points', '❚❚ Pause']);
      UI.button(row, 'All at once', () => { cnt = 2 ** Math.round(mS.get()); draw(); });
      UI.button(row, 'Reset', reset);
      const p = new Plot(stage, { width: 520, height: 520, x: [0, 1], y: [0, 1], xlabel: 'x₁', ylabel: 'x₂', equal: true });
      const sN = UI.stat(stats, 'Points shown');
      const sD = UI.stat(stats, '$L_2$-star discrepancy', { accent: 'accent' });
      const sR = UI.stat(stats, 'Expected for random points');
      const sB = UI.stat(stats, 'Boxes with exactly one point');
      let pts = [], cnt = 0, acc = 0, delta = [0, 0];
      function gen() {
        const n = 2 ** Math.round(mS.get()), k = kind.get();
        pts = [];
        for (let i = 0; i < n; i++) {
          if (k === 'rand') pts.push([ctx.rng.u(), ctx.rng.u()]);
          else if (k === 'sob') pts.push(sobol2(i));
          else if (k === 'h23') pts.push([radInv(i + 1, 2), radInv(i + 1, 3)]);
          else pts.push([radInv(i + 1, 29), radInv(i + 1, 31)]);
        }
      }
      function reset() { gen(); cnt = Math.min(pts.length, 2 ** Math.round(mS.get())); loop.stop(); draw(); }
      function step(dt) { acc += dt * Math.max(8, pts.length / 6); let k = 0; while (acc >= 1 && cnt < pts.length) { cnt++; acc -= 1; k++; } if (k) draw(); if (cnt >= pts.length) return false; }
      function draw() {
        const P = pts.slice(0, cnt).map(([a, b]) => (shift.get() ? [(a + delta[0]) % 1, (b + delta[1]) % 1] : [a, b]));
        p.clear();
        const m = Math.round(mS.get()), n = 2 ** m, a = Math.floor(m / 2), b = m - a;
        if (boxes.get()) {
          for (let i = 1; i < 2 ** a; i++) p.vline(i / 2 ** a, { color: 'faint', width: 1, dash: false });
          for (let j = 1; j < 2 ** b; j++) p.hline(j / 2 ** b, { color: 'faint', width: 1, dash: false });
        }
        p.points(P.slice(0, Math.max(0, P.length - 1)), { color: 's1', r: n > 300 ? 2.2 : 3.2, opacity: 0.85 });
        if (P.length) p.dot(P[P.length - 1][0], P[P.length - 1][1], { color: 's4', r: 5 });
        sN.set(`${P.length} / ${n}`);
        if (P.length > 1) {
          const D = l2star(P), Er = Math.sqrt((1 / 4 - 1 / 9) / P.length);
          sD.set(D.toExponential(2));
          sR.set(Er.toExponential(2), `ratio ${fmt(Er / D, 3)}×`);
        } else { sD.set('—'); sR.set('—'); }
        const counts = new Map();
        for (const [x, y] of P) { const key = Math.floor(x * 2 ** a) + ',' + Math.floor(y * 2 ** b); counts.set(key, (counts.get(key) || 0) + 1); }
        let one = 0; for (const v of counts.values()) if (v === 1) one++;
        sB.set(`${one} / ${n}`, P.length === n ? (one === n ? 'a perfect (0, m, 2)-net' : 'not a net at this resolution') : 'fill to n to check');
      }
      reset();
    },
  });

  const QMC_F = {
    smooth: { label: '∏ (π/2) sin(π xⱼ)  (smooth)', f: (x, d) => { let v = 1; for (let j = 0; j < d; j++) v *= (Math.PI / 2) * Math.sin(Math.PI * x[j]); return v; }, I: () => 1 },
    weighted: { label: '∏ (1 + (2xⱼ − 1)/j²)  (low effective dimension)', f: (x, d) => { let v = 1; for (let j = 0; j < d; j++) v *= 1 + (2 * x[j] - 1) / ((j + 1) * (j + 1)); return v; }, I: () => 1 },
    disc: { label: '1{Σ xⱼ ≤ d/2}  (discontinuous)', f: (x, d) => { let s = 0; for (let j = 0; j < d; j++) s += x[j]; return s <= d / 2 ? 1 : 0; }, I: () => 0.5 },
  };
  Atlas.lab('quasi-monte-carlo-error', {
    title: 'Integration error: MC, Halton, randomized Halton',
    blurb: 'RMS error over 10 replications (independent MC runs, or 10 random shifts of the same Halton points) versus $n$, on log–log axes, with the deterministic Halton error for reference. Smooth, low-effective-dimension integrands approach slope $-1$; the discontinuous one, and high $d$, pull QMC back toward $-1/2$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ig = UI.select(controls, { label: 'Integrand', options: Object.keys(QMC_F).map(k => ({ value: k, label: QMC_F[k].label })), value: 'smooth', onChange: run });
      const dS = UI.slider(controls, { label: 'Dimension $d$', min: 1, max: 10, step: 1, value: 2, onChange: run });
      const mS = UI.slider(controls, { label: 'Largest $n = 2^m$', min: 8, max: 14, step: 1, value: 12, fmt: v => `2^${v} = ${2 ** v}`, onChange: run });
      UI.button(controls, 'New randomization', run, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, logx: true, logy: true, x: [1, 1e4], y: [1e-6, 1], xlabel: 'n', ylabel: 'RMS error' });
      const sMC = UI.stat(stats, 'MC slope');
      const sR = UI.stat(stats, 'RQMC slope', { accent: 'accent' });
      const sH = UI.stat(stats, 'Halton slope (deterministic)');
      const sG = UI.stat(stats, 'RQMC gain at largest $n$');
      function run() {
        const F = QMC_F[ig.get()], d = Math.round(dS.get()), N = 2 ** Math.round(mS.get()), I = F.I(d), R = 10;
        const H = []; for (let i = 0; i < N; i++) { const x = new Array(d); for (let j = 0; j < d; j++) x[j] = radInv(i + 1, PRIMES[j]); H.push(x); }
        const cps = []; for (let k = 3; 2 ** k <= N; k++) cps.push(2 ** k);
        const se = (arr) => Math.sqrt(S.mean(arr.map(v => (v - I) ** 2)));
        const mc = cps.map(() => []), rq = cps.map(() => []), det = [];
        const x = new Array(d);
        for (let r = 0; r < R; r++) {
          let s = 0, j = 0;
          for (let i = 1; i <= N; i++) { for (let k = 0; k < d; k++) x[k] = ctx.rng.u(); s += F.f(x, d); if (i === cps[j]) { mc[j].push(s / i); j++; } }
          const sh = Array.from({ length: d }, () => ctx.rng.u());
          s = 0; j = 0;
          for (let i = 1; i <= N; i++) { const h = H[i - 1]; for (let k = 0; k < d; k++) { const v = h[k] + sh[k]; x[k] = v >= 1 ? v - 1 : v; } s += F.f(x, d); if (i === cps[j]) { rq[j].push(s / i); j++; } }
        }
        { let s = 0, j = 0; for (let i = 1; i <= N; i++) { s += F.f(H[i - 1], d); if (i === cps[j]) { det.push(Math.abs(s / i - I)); j++; } } }
        const eMC = mc.map(se), eRQ = rq.map(se);
        const floor = 1e-7;
        const all = [...eMC, ...eRQ, ...det].filter(v => v > 0);
        p.setDomain([4, N * 1.5], [Math.max(1e-7, Math.min(...all) / 3), Math.max(...all) * 3]).clear();
        p.line(cps.map((c, j) => [c, Math.max(floor, eMC[j])]), { color: 's2', width: 2.2 });
        p.points(cps.map((c, j) => [c, Math.max(floor, eMC[j])]), { color: 's2', r: 3 });
        p.line(cps.map((c, j) => [c, Math.max(floor, eRQ[j])]), { color: 's1', width: 2.2 });
        p.points(cps.map((c, j) => [c, Math.max(floor, eRQ[j])]), { color: 's1', r: 3 });
        p.line(cps.map((c, j) => [c, Math.max(floor, det[j])]), { color: 'muted', width: 1.4, dash: true });
        const c0 = cps[0], e0 = eMC[0];
        p.line([[c0, e0], [N, e0 * Math.sqrt(c0 / N)]], { color: 'faint', width: 1, dash: '2 4' });
        p.line([[c0, e0], [N, e0 * c0 / N]], { color: 'faint', width: 1, dash: '2 4' });
        p.legend([{ label: 'Monte Carlo', color: 's2' }, { label: 'randomly shifted Halton', color: 's1' }, { label: 'Halton (deterministic)', color: 'muted', dash: true }, { label: 'slopes −1/2 and −1', color: 'faint', dash: true }], { left: true });
        const slope = (e) => { const idx = cps.map((_, j) => j).filter(j => j >= cps.length / 2 && e[j] > 0); const lx = idx.map(j => Math.log(cps[j])), ly = idx.map(j => Math.log(e[j])); const mx = S.mean(lx), my = S.mean(ly); return S.sum(lx.map((v, i) => (v - mx) * (ly[i] - my))) / S.sum(lx.map(v => (v - mx) ** 2)); };
        sMC.set(fmt(slope(eMC), 2), 'theory −0.5');
        sR.set(fmt(slope(eRQ), 2));
        sH.set(fmt(slope(det), 2), 'noisy: one deterministic sequence');
        sG.set(fmt((eMC[eMC.length - 1] / eRQ[eRQ.length - 1]) ** 2, 3) + '×', 'variance ratio MC / RQMC');
      }
      run();
    },
  });

  /* ================= particle-filters ================= */

  function systematic(W, N, r) {
    const idx = new Array(N), u0 = r.u() / N; let c = W[0], j = 0;
    for (let i = 0; i < N; i++) { const u = u0 + i / N; while (u > c && j < W.length - 1) { j++; c += W[j]; } idx[i] = j; }
    return idx;
  }
  function multinomial(W, N, r) {
    const cum = S.cumsum(W), idx = new Array(N);
    for (let i = 0; i < N; i++) { const u = r.u() * cum[cum.length - 1]; let lo = 0, hi = cum.length - 1; while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] < u) lo = m + 1; else hi = m; } idx[i] = lo; }
    return idx;
  }
  const kitF = (x, t) => 0.5 * x + 25 * x / (1 + x * x) + 8 * Math.cos(1.2 * t);
  const LOG2PI = Math.log(2 * Math.PI);

  Atlas.lab('particle-filters-tracking', {
    title: 'A bootstrap particle filter on a bimodal tracking problem',
    blurb: 'The benchmark $X_t = X_{t-1}/2 + 25X_{t-1}/(1+X_{t-1}^2) + 8\\cos(1.2t) + V_t$, $Y_t = X_t^2/20 + W_t$. The cloud shows the filter (resampled for display); the observation cannot tell $x$ from $-x$, so the cloud often splits. Set the ESS threshold to 0 to run plain SIS and watch the ESS collapse; set it to 1 to resample at every step.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const NS = UI.slider(controls, { label: 'Particles $N$', min: 20, max: 2000, value: 500, log: true, fmt: v => Math.round(v), onChange: filter });
      const tau = UI.slider(controls, { label: 'Resample when ESS/N <', min: 0, max: 1, step: 0.05, value: 0.5, onChange: filter });
      const sch = UI.segmented(controls, { label: 'Scheme', options: [{ value: 'sys', label: 'Systematic' }, { value: 'mult', label: 'Multinomial' }], value: 'sys', onChange: filter });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Animate', '❚❚ Pause']);
      UI.button(row, 'New data', () => { simulate(); filter(); });
      UI.button(row, 'Rerun filter', filter);
      const T = 60;
      const pT = new Plot(stage, { width: 640, height: 280, x: [0, T], y: [-25, 25], xlabel: 't', ylabel: 'state x', title: 'True state, particle cloud and filtered mean' });
      const pE = new Plot(stage, { width: 640, height: 170, x: [0, T], y: [0, 1], xlabel: 't', ylabel: 'ESS / N', title: 'Effective sample size before resampling' });
      const sR = UI.stat(stats, 'RMSE of filtered mean', { accent: 'accent' });
      const sRs = UI.stat(stats, 'Resampling steps');
      const sMin = UI.stat(stats, 'Minimum ESS');
      const sL = UI.stat(stats, '$\\log \\hat p(y_{1:T})$');
      let xs = [], ys = [], clouds = [], means = [], ess = [], resampled = [], ll = 0, shown = T;
      function simulate() {
        xs = [0]; ys = [null];
        let x = ctx.rng.normal(0, Math.sqrt(5));
        xs[0] = x;
        for (let t = 1; t <= T; t++) { x = kitF(x, t) + ctx.rng.normal(0, Math.sqrt(10)); xs.push(x); ys.push(x * x / 20 + ctx.rng.normal()); }
      }
      function filter() {
        const N = Math.round(NS.get()), th = tau.get(), rs = sch.get() === 'sys' ? systematic : multinomial;
        let P = ctx.rng.array(N, () => ctx.rng.normal(0, Math.sqrt(5)));
        let W = new Array(N).fill(1 / N);
        clouds = [P.slice(0, 120)]; means = [S.mean(P)]; ess = [1]; resampled = [false]; ll = 0;
        for (let t = 1; t <= T; t++) {
          const e = 1 / S.sum(W.map(w => w * w)) / N;
          let res = false;
          if (e < th || th >= 1) { const idx = rs(W, N, ctx.rng); P = idx.map(i => P[i]); W = new Array(N).fill(1 / N); res = true; }
          const Pn = new Array(N), lw = new Array(N);
          for (let i = 0; i < N; i++) { Pn[i] = kitF(P[i], t) + ctx.rng.normal(0, Math.sqrt(10)); const r = ys[t] - Pn[i] * Pn[i] / 20; lw[i] = Math.log(W[i]) - 0.5 * r * r - 0.5 * LOG2PI; }
          const m = Math.max(...lw); let sw = 0; const w = lw.map(v => { const z = Math.exp(v - m); sw += z; return z; });
          ll += m + Math.log(sw);
          W = w.map(v => v / sw); P = Pn;
          ess.push(e); resampled.push(res);
          let mu = 0; for (let i = 0; i < N; i++) mu += W[i] * P[i]; means.push(mu);
          const di = systematic(W, Math.min(120, N), ctx.rng); clouds.push(di.map(i => P[i]));
        }
        shown = T; draw();
      }
      let acc = 0;
      function step(dt) { acc += dt * 8; if (acc >= 1) { acc = 0; shown = shown >= T ? 1 : shown + 1; draw(); } }
      function draw() {
        const lim = Math.max(25, ...xs.map(Math.abs)) * 1.05;
        pT.setDomain([0, T], [-lim, lim]).clear();
        for (let t = 0; t <= shown; t++) pT.points(clouds[t].map(v => [t, v]), { color: 's1', r: 1.6, opacity: 0.18 });
        pT.line(xs.slice(0, shown + 1).map((v, t) => [t, v]), { color: 'ink', width: 2 });
        pT.line(means.slice(0, shown + 1).map((v, t) => [t, v]), { color: 's2', width: 2, dash: true });
        pT.legend([{ label: 'true x_t', color: 'ink' }, { label: 'filtered mean', color: 's2', dash: true }, { label: 'particles', color: 's1', fill: true, opacity: 0.3 }], { left: true });
        pE.clear();
        pE.hline(tau.get(), { color: 'bad', width: 1.2 });
        pE.line(ess.slice(1, shown + 1).map((v, t) => [t + 1, v]), { color: 's3', width: 2 });
        for (let t = 1; t <= shown; t++) if (resampled[t]) pE.dot(t, ess[t], { color: 's4', r: 3 });
        const err = xs.slice(1).map((v, t) => (v - means[t + 1]) ** 2);
        sR.set(fmt(Math.sqrt(S.mean(err)), 3), 'vs truth over t = 1…60');
        sRs.set(String(resampled.filter(Boolean).length), 'dots in the ESS panel');
        sMin.set(fmt(Math.min(...ess.slice(1)) * Math.round(NS.get()), 3), 'particles');
        sL.set(fmt(ll, 4), 'unbiased on the exp scale');
      }
      simulate(); filter();
    },
  });

  Atlas.lab('particle-filters-likelihood', {
    title: 'The particle likelihood is unbiased; its log is not',
    blurb: 'Linear-Gaussian model $X_t = 0.9X_{t-1} + V_t$, $Y_t = X_t + W_t$, $T = 50$, where the Kalman filter gives the exact $\\log p(y_{1:T})$. Each run of a bootstrap filter gives $\\log\\hat p$; the histogram shows $\\log\\hat p - \\log p$ over 60 runs. Its mean sits near $-\\tfrac12\\Var(\\log\\hat p)$. More informative observations (small $r$) need more particles.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const NS = UI.slider(controls, { label: 'Particles $N$', min: 10, max: 1000, value: 100, log: true, fmt: v => Math.round(v), onChange: run });
      const rS = UI.slider(controls, { label: 'Observation variance $r$', min: 0.05, max: 4, step: 0.05, value: 1, onChange: () => { simulate(); run(); } });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Rerun filters', run, { primary: true });
      UI.button(row, 'New data', () => { simulate(); run(); });
      const p = new Plot(stage, { width: 640, height: 280, xlabel: 'log p̂ − log p', ylabel: 'density' });
      const sV = UI.stat(stats, '$\\Var(\\log\\hat p)$', { accent: 'accent' });
      const sM = UI.stat(stats, 'Mean of $\\log\\hat p - \\log p$');
      const sR = UI.stat(stats, 'Mean of $\\hat p / p$');
      const sN = UI.stat(stats, '$N$ for $\\Var(\\log \\hat p) \\approx 1$');
      const T = 50, phi = 0.9, q = 1;
      let ys = [];
      function simulate() {
        const r = rS.get(); let x = ctx.rng.normal(0, Math.sqrt(q / (1 - phi * phi))); ys = [];
        for (let t = 0; t < T; t++) { if (t > 0) x = phi * x + ctx.rng.normal(0, Math.sqrt(q)); ys.push(x + ctx.rng.normal(0, Math.sqrt(r))); }
      }
      function kalman() {
        const r = rS.get(); let m = 0, P = q / (1 - phi * phi), ll = 0;
        for (let t = 0; t < T; t++) {
          if (t > 0) { m = phi * m; P = phi * phi * P + q; }
          const Sv = P + r, e = ys[t] - m; ll += -0.5 * (LOG2PI + Math.log(Sv) + e * e / Sv);
          const K = P / Sv; m += K * e; P = (1 - K) * P;
        }
        return ll;
      }
      function pf(N) {
        const r = rS.get(); let X = ctx.rng.array(N, () => ctx.rng.normal(0, Math.sqrt(q / (1 - phi * phi)))), ll = 0;
        const lw = new Array(N);
        for (let t = 0; t < T; t++) {
          if (t > 0) for (let i = 0; i < N; i++) X[i] = phi * X[i] + ctx.rng.normal(0, Math.sqrt(q));
          let m = -Infinity;
          for (let i = 0; i < N; i++) { const e = ys[t] - X[i]; lw[i] = -0.5 * (LOG2PI + Math.log(r) + e * e / r); if (lw[i] > m) m = lw[i]; }
          let sw = 0; const W = new Array(N); for (let i = 0; i < N; i++) { W[i] = Math.exp(lw[i] - m); sw += W[i]; }
          ll += m + Math.log(sw / N);
          for (let i = 0; i < N; i++) W[i] /= sw;
          const idx = systematic(W, N, ctx.rng); X = idx.map(i => X[i]);
        }
        return ll;
      }
      function run() {
        const N = Math.round(NS.get()), R = 60, exact = kalman();
        const d = []; for (let k = 0; k < R; k++) d.push(pf(N) - exact);
        const v = S.variance(d), mu = S.mean(d);
        const [a, b] = S.extent(d), lo = Math.min(a, -0.5) - 0.3, hi = Math.max(b, 0.5) + 0.3;
        const hb = S.histogram(d, { bins: 24, lo, hi });
        p.setDomain([lo, hi], [0, Math.max(...hb.map(x => x.density)) * 1.2]).clear();
        p.bars(hb, { color: 's1', opacity: 0.55 });
        p.vline(0, { color: 'ink', width: 1.6, dash: false });
        p.vline(mu, { color: 's2', width: 2 });
        p.vline(-v / 2, { color: 's3', width: 2 });
        p.legend([{ label: 'exact (Kalman)', color: 'ink' }, { label: 'mean of runs', color: 's2', dash: true }, { label: '−Var/2', color: 's3', dash: true }]);
        const ratio = S.mean(d.map(Math.exp));
        sV.set(fmt(v, 3));
        sM.set(fmt(mu, 3), `−Var/2 = ${fmt(-v / 2, 3)}`);
        sR.set(fmt(ratio, 3), 'should be ≈ 1 (noisy: heavy right tail)');
        sN.set(String(Math.max(1, Math.round(N * v))), 'Var(log p̂) ∝ 1/N');
      }
      simulate(); run();
    },
  });

  /* ================= simulation-studies ================= */

  const zq = 1.959963984540054;
  const CI = {
    wald: { label: 'Wald', c: 's1', f: (k, n) => { const p = k / n, h = zq * Math.sqrt(p * (1 - p) / n); return [p - h, p + h]; } },
    wilson: { label: 'Wilson', c: 's2', f: (k, n) => { const p = k / n, z2 = zq * zq, den = 1 + z2 / n, c = (p + z2 / (2 * n)) / den, h = zq * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / den; return [c - h, c + h]; } },
    ac: { label: 'Agresti–Coull', c: 's3', f: (k, n) => { const nt = n + zq * zq, p = (k + zq * zq / 2) / nt, h = zq * Math.sqrt(p * (1 - p) / nt); return [p - h, p + h]; } },
  };
  function exactCov(method, n, p) { let s = 0; for (let k = 0; k <= n; k++) { const [L, U] = CI[method].f(k, n); if (L <= p && p <= U) s += S.pmf.binom(k, n, p); } return s; }

  Atlas.lab('simulation-studies-coverage', {
    title: 'Simulated coverage, its Monte Carlo error, and the exact answer',
    blurb: 'Curves: exact coverage of three nominal-95% intervals for a binomial proportion, as a function of $p$. Dots with $\\pm 2$ MCSE bars: what a simulation study with $R$ replications reports at 15 values of $p$. About 1 bar in 20 should miss the curve; with small $R$, the Wald interval\'s erratic undercoverage is easy to misread as noise.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nSel = UI.segmented(controls, { label: 'Sample size $n$', options: ['10', '20', '50', '100'], value: '20', onChange: run });
      const RS = UI.slider(controls, { label: 'Replications $R$', min: 100, max: 10000, value: 500, log: true, fmt: v => Math.round(v), onChange: run });
      const meth = UI.segmented(controls, { label: 'Simulate', options: Object.keys(CI).map(k => ({ value: k, label: CI[k].label })), value: 'wald', onChange: run });
      UI.button(controls, 'Rerun simulation', run, { primary: true });
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 1], y: [0.7, 1], xlabel: 'true p', ylabel: 'coverage' });
      const sM = UI.stat(stats, 'Typical MCSE', { accent: 'accent' });
      const sMiss = UI.stat(stats, 'Bars missing the exact curve');
      const sMin = UI.stat(stats, 'Worst exact coverage (this method)');
      const sAvg = UI.stat(stats, 'Average exact coverage over $p$');
      function run() {
        const n = +nSel.get(), R = Math.round(RS.get()), m = meth.get();
        const grid = S.linspace(0.005, 0.995, 397);
        p.clear();
        p.hline(0.95, { color: 'ink', width: 1.2 });
        const curves = {};
        for (const k of Object.keys(CI)) {
          curves[k] = grid.map(q => exactCov(k, n, q));
          p.line(grid.map((q, i) => [q, Math.max(0.7, curves[k][i])]), { color: CI[k].c, width: k === m ? 2 : 1.1, opacity: k === m ? 1 : 0.45 });
        }
        const ps = S.linspace(0.05, 0.95, 15);
        let miss = 0, mcses = [];
        for (const q of ps) {
          let hit = 0;
          for (let r = 0; r < R; r++) { const k = ctx.rng.binomial(n, q), [L, U] = CI[m].f(k, n); if (L <= q && q <= U) hit++; }
          const c = hit / R, se = Math.sqrt(Math.max(c * (1 - c), 1 / R) / R), ex = exactCov(m, n, q);
          mcses.push(se);
          const ok = Math.abs(c - ex) <= 2 * se; if (!ok) miss++;
          p.segment(q, Math.max(0.7, c - 2 * se), q, Math.min(1, c + 2 * se), { color: ok ? 'ink' : 'bad', width: 2 });
          p.dot(q, Math.max(0.7, c), { color: ok ? CI[m].c : 'bad', r: 4 });
        }
        p.legend(Object.keys(CI).map(k => ({ label: CI[k].label, color: CI[k].c })), { left: true });
        const cv = curves[m];
        sM.set(pct(S.mean(mcses), 2), `R = ${R}`);
        sMiss.set(`${miss} / 15`, 'expected ≈ 0.7 by chance');
        sMin.set(pct(Math.min(...cv)), `at p = ${fmt(grid[S.argmin(cv)], 3)}`);
        sAvg.set(pct(S.mean(cv)));
      }
      run();
    },
  });

  const POPS_SS = {
    normal: { label: 'Normal', draw: r => r.normal(), mean: 0 },
    exp: { label: 'Exponential', draw: r => r.exp(1), mean: 1 },
    lognormal: { label: 'Lognormal(0, 1)', draw: r => r.lognormal(0, 1), mean: Math.exp(0.5) },
  };
  Atlas.lab('simulation-studies-mcse', {
    title: 'How many replications? Watching a coverage estimate settle',
    blurb: 'Replications of the $t$-interval for a mean accumulate one batch at a time. The band is $\\pm 2$ MCSE around the running estimate; the question is when it stops straddling 95%. Skewed populations with small $n$ undercover, but it takes thousands of replications to say so confidently.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pop = UI.segmented(controls, { label: 'Population', options: Object.keys(POPS_SS).map(k => ({ value: k, label: POPS_SS[k].label })), value: 'lognormal', onChange: reset });
      const nS = UI.slider(controls, { label: 'Sample size $n$ per dataset', min: 5, max: 100, step: 1, value: 15, onChange: reset });
      const tgt = UI.slider(controls, { label: 'Target MCSE (percentage points)', min: 0.1, max: 2, step: 0.1, value: 0.5, onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run replications', '❚❚ Pause']);
      UI.button(row, '+2000', () => { batch(2000); draw(); });
      UI.button(row, 'Reset', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, logx: true, x: [10, 20000], y: [0.8, 1], xlabel: 'replications R', ylabel: 'coverage', title: 'Running coverage ± 2 MCSE' });
      const pR = new Plot(stage, { width: 420, height: 280, logx: true, logy: true, x: [0.001, 0.02], y: [100, 1e6], xlabel: 'target MCSE', ylabel: 'R needed', title: 'R = c(1 − c) / MCSE²' });
      const sC = UI.stat(stats, 'Coverage ± MCSE', { accent: 'accent' });
      const sZ = UI.stat(stats, 'Distance from 95% in MCSEs');
      const sNeed = UI.stat(stats, '$R$ needed for target MCSE');
      let hits = 0, R = 0, path = [];
      function reset() { hits = 0; R = 0; path = []; loop.stop(); draw(); }
      function batch(k) {
        const P = POPS_SS[pop.get()], n = Math.round(nS.get()), tq = S.q.t(0.975, n - 1);
        for (let i = 0; i < k; i++) {
          let s = 0, s2 = 0; for (let j = 0; j < n; j++) { const v = P.draw(ctx.rng); s += v; s2 += v * v; }
          const m = s / n, sd = Math.sqrt(Math.max(0, (s2 - n * m * m) / (n - 1))), h = tq * sd / Math.sqrt(n);
          if (Math.abs(m - P.mean) <= h) hits++;
          R++;
          if (R >= 10 && (path.length === 0 || R >= path[path.length - 1][0] * 1.04)) path.push([R, hits / R]);
        }
      }
      function step() { batch(Math.max(10, Math.round(R * 0.03))); draw(); if (R >= 20000) return false; }
      function draw() {
        pL.clear();
        pL.hline(0.95, { color: 'ink', width: 1.3 });
        if (path.length > 1) {
          const up = path.map(([r, c]) => [r, Math.min(1, c + 2 * Math.sqrt(Math.max(c * (1 - c), 1e-4) / r))]);
          const dn = path.map(([r, c]) => [r, Math.max(0.8, c - 2 * Math.sqrt(Math.max(c * (1 - c), 1e-4) / r))]);
          pL.area(up, { color: 's1', opacity: 0.15, base: dn });
          pL.line(path.map(([r, c]) => [r, Math.max(0.8, c)]), { color: 's1', width: 2 });
        }
        const c = R ? hits / R : 0.95, se = R ? Math.sqrt(c * (1 - c) / R) : NaN, t = tgt.get() / 100;
        const xs = S.linspace(Math.log10(0.001), Math.log10(0.02), 60).map(v => 10 ** v);
        pR.clear();
        pR.line(xs.map(m => [m, Math.max(100, c * (1 - c) / (m * m))]), { color: 's2', width: 2 });
        pR.dot(t, Math.max(100, c * (1 - c) / (t * t)), { color: 's4', r: 5 });
        if (R && se > 0.001 && se < 0.02) pR.dot(se, Math.max(100, R), { color: 's1', r: 5 });
        pR.legend([{ label: 'required R', color: 's2' }, { label: 'target', color: 's4', fill: true }, { label: 'current run', color: 's1', fill: true }]);
        sC.set(R ? `${pct(c, 2)} ± ${pct(se, 2)}` : '—', `R = ${R}`);
        sZ.set(R && se > 0 ? fmt((c - 0.95) / se, 3) : '—', R && se > 0 && Math.abs(c - 0.95) > 2 * se ? '<span style="color:var(--bad)">departure from 95% detected</span>' : 'consistent with 95% so far');
        sNeed.set(String(Math.ceil(c * (1 - c) / (t * t))), `for MCSE = ${fmt(100 * t, 2)} points`);
      }
      reset();
    },
  });
})();
