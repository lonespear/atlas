/* Labs: Regression & Linear Models. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const R = String.raw;

  /* ---------------- shared helpers ---------------- */

  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const sq = a => dot(a, a);

  // Least squares with full diagnostics. X: rows (n x p). Throws if X'X singular.
  function fitLS(X, y) {
    const n = y.length, p = X[0].length;
    const Xt = S.transpose(X), Ai = S.inv(S.matmul(Xt, X));
    const beta = S.matvec(Ai, S.matvec(Xt, y));
    const fit = X.map(r => dot(r, beta)), res = y.map((v, i) => v - fit[i]);
    const h = X.map(r => dot(r, S.matvec(Ai, r)));
    const rss = sq(res), s2 = rss / Math.max(1, n - p);
    return { beta, fit, res, h, Ai, rss, s2, n, p };
  }

  // Simple regression from arrays.
  function slr(x, y) {
    const n = x.length, mx = S.mean(x), my = S.mean(y);
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < n; i++) { const dx = x[i] - mx, dy = y[i] - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
    const b1 = sxx > 0 ? sxy / sxx : NaN, b0 = my - b1 * mx;
    const rss = Math.max(0, syy - b1 * sxy);
    return { b0, b1, sxx, sxy, syy, mx, my, rss, n, s2: rss / Math.max(1, n - 2), r2: syy > 0 ? 1 - rss / syy : NaN };
  }

  // Orthonormal columns by twice-iterated modified Gram–Schmidt. cols: array of length-n vectors.
  function mgs(cols) {
    const Q = [];
    for (const c of cols) {
      const v = c.slice();
      for (let pass = 0; pass < 2; pass++) for (const q of Q) { const d = dot(q, v); for (let i = 0; i < v.length; i++) v[i] -= d * q[i]; }
      const nr = Math.sqrt(sq(v));
      Q.push(v.map(t => t / (nr || 1)));
    }
    return Q;
  }

  // Legendre polynomials P_0..P_D at t in [-1,1].
  function legendre(t, D) {
    const P = [1, t];
    for (let k = 1; k < D; k++) P.push(((2 * k + 1) * t * P[k] - k * P[k - 1]) / (k + 1));
    return P.slice(0, D + 1);
  }

  const normQ = (i, n) => S.qnorm((i + 1 - 0.375) / (n + 0.25)); // Blom plotting positions
  const pct = v => (100 * v).toFixed(1) + '%';
  const good = s => `<span style="color:var(--good)">${s}</span>`;
  const bad = s => `<span style="color:var(--bad)">${s}</span>`;

  /* ================================================================ */
  /* simple-linear-regression                                         */
  /* ================================================================ */

  Atlas.lab('simple-linear-regression-drag', {
    title: 'Least squares by hand',
    blurb: R`Drag points, or click empty space to add one. Left: the least-squares line, with residuals drawn as literal squares whose total area is the RSS. Right: RSS as a function of $(b_0,b_1)$ — an elliptical bowl whose bottom is the least-squares estimate. Switch on the manual line and try to beat it; drag a point to the far right and watch one observation tilt the whole line.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const mode = UI.segmented(controls, { label: 'Click on the plot to', options: [{ value: 'add', label: 'Add / drag' }, { value: 'del', label: 'Remove' }], value: 'add' });
      const sqT = UI.toggle(controls, { label: 'Draw squared residuals', value: true, onChange: draw });
      const man = UI.toggle(controls, { label: 'Show my manual line', value: false, onChange: draw });
      const b0s = UI.slider(controls, { label: 'Manual intercept $b_0$', min: -4, max: 8, step: 0.05, value: 1, onInput: () => { if (!man.get()) man.set(true); draw(); } });
      const b1s = UI.slider(controls, { label: 'Manual slope $b_1$', min: -1.5, max: 2.5, step: 0.01, value: 0.4, onInput: () => { if (!man.get()) man.set(true); draw(); } });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New data', newData, { primary: true });
      UI.button(row, 'Add outlier', () => { pts.push([9.4, 1.2]); draw(); });
      UI.button(row, 'Snap manual to OLS', () => { const f = fit(); if (f) { b0s.set(f.b0); b1s.set(f.b1); } draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 10], y: [0, 10], xlabel: 'x', ylabel: 'y', title: 'Data, fit, residual squares' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'intercept b₀', ylabel: 'slope b₁', title: 'RSS(b₀, b₁) contours' });
      const sB = UI.stat(stats, 'OLS $(\\hat\\beta_0,\\hat\\beta_1)$', { accent: 'accent' });
      const sRss = UI.stat(stats, 'Minimum RSS');
      const sMan = UI.stat(stats, 'RSS of manual line');
      const sR2 = UI.stat(stats, '$R^2$');
      const sN = UI.stat(stats, '$n$');

      let pts = [], drag = -1;
      function newData() {
        pts = ctx.rng.array(8, () => { const x = ctx.rng.uniform(0.8, 9.2); return [x, S.clamp(1.5 + 0.6 * x + ctx.rng.normal(0, 0.9), 0.3, 9.7)]; });
        draw();
      }
      function fit() {
        const n = pts.length;
        let sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
        for (const [x, y] of pts) { sx += x; sy += y; sxx += x * x; sxy += x * y; syy += y * y; }
        const Sxx = sxx - sx * sx / n, Sxy = sxy - sx * sy / n, Syy = syy - sy * sy / n;
        if (Sxx < 1e-9) return null;
        const b1 = Sxy / Sxx, b0 = sy / n - b1 * sx / n;
        const rssAt = (a, b) => syy - 2 * a * sy - 2 * b * sxy + n * a * a + 2 * a * b * sx + b * b * sxx;
        const det = n * sxx - sx * sx;
        return { n, b0, b1, rss: Math.max(0, Syy - b1 * Sxy), Syy, rssAt, Ai: [[sxx / det, -sx / det], [-sx / det, n / det]] };
      }
      pL.onPointer((x, y, evt, type) => {
        if (type === 'down') {
          let best = -1, bd = Infinity;
          pts.forEach((q, i) => { const d = ((q[0] - x) * pL.iw / 10) ** 2 + ((q[1] - y) * pL.ih / 10) ** 2; if (d < bd) { bd = d; best = i; } });
          if (bd < 144) {
            if (mode.get() === 'del') { if (pts.length > 3) pts.splice(best, 1); draw(); }
            else drag = best;
          } else if (mode.get() === 'add' && pL.inside(x, y)) { pts.push([x, y]); drag = pts.length - 1; draw(); }
        } else if (type === 'drag' && drag >= 0) {
          pts[drag] = [S.clamp(x, 0.05, 9.95), S.clamp(y, 0.05, 9.95)];
          draw();
        } else if (type === 'up' || type === 'leave') drag = -1;
      });
      function draw() {
        const f = fit();
        pL.clear();
        if (f) {
          const sx = pL.ih / pL.iw; // converts a y-length into the x-length that looks equal on screen
          for (const [x, y] of pts) {
            const yh = f.b0 + f.b1 * x, e = y - yh;
            if (sqT.get() && Math.abs(e) > 1e-6) {
              const w = Math.abs(e) * sx, x1 = x < 5 ? x + w : x - w;
              pL.rect(x, yh, x1, y, { color: e > 0 ? 's2' : 's4', opacity: 0.18 });
            }
            pL.segment(x, y, x, yh, { color: e > 0 ? 's2' : 's4', width: 1.5, opacity: 0.85 });
          }
          pL.line([[0, f.b0], [10, f.b0 + 10 * f.b1]], { color: 's1', width: 2.5 });
          pL.dot(S.mean(pts.map(q => q[0])), S.mean(pts.map(q => q[1])), { color: 's1', r: 3.5 });
        }
        if (man.get()) pL.line([[0, b0s.get()], [10, b0s.get() + 10 * b1s.get()]], { color: 's3', width: 2, dash: true });
        for (const [x, y] of pts) pL.dot(x, y, { color: 'ink', r: 4.5 });
        pL.legend([{ label: 'least squares', color: 's1' }].concat(man.get() ? [{ label: 'manual', color: 's3', dash: true }] : []), { left: true });

        pR.clear();
        if (!f) { sB.set('—', 'all x equal: slope not identified'); return; }
        const c = Math.max(f.rss, 2) * 3;
        let w0 = Math.sqrt(c * f.Ai[0][0]), w1 = Math.sqrt(c * f.Ai[1][1]);
        let lo0 = f.b0 - w0, hi0 = f.b0 + w0, lo1 = f.b1 - w1, hi1 = f.b1 + w1;
        if (man.get()) { lo0 = Math.min(lo0, b0s.get() - 0.1 * w0); hi0 = Math.max(hi0, b0s.get() + 0.1 * w0); lo1 = Math.min(lo1, b1s.get() - 0.1 * w1); hi1 = Math.max(hi1, b1s.get() + 0.1 * w1); }
        pR.setDomain([lo0, hi0], [lo1, hi1]).clear();
        const levels = [0.03, 0.1, 0.25, 0.5, 0.8, 1.2].map(t => f.rss + c * t);
        pR.contour((a, b) => f.rssAt(a, b), levels, { color: 's1', nx: 70, ny: 50, opacity: 0.7 });
        pR.dot(f.b0, f.b1, { color: 's1', r: 5 });
        pR.text(f.b0, f.b1, 'OLS', { dx: 8, dy: -6, size: 11, color: 's1', weight: 600 });
        if (man.get()) {
          pR.dot(b0s.get(), b1s.get(), { color: 's3', r: 5 });
          pR.contour((a, b) => f.rssAt(a, b), [f.rssAt(b0s.get(), b1s.get())], { color: 's3', nx: 70, ny: 50, dash: '4 3' });
        }
        sB.set(`(${S.fmt(f.b0, 2)}, ${S.fmt(f.b1, 3)})`);
        sRss.set(S.fmt(f.rss, 2), 'sum of the square areas');
        if (man.get()) {
          const rm = f.rssAt(b0s.get(), b1s.get());
          sMan.set(S.fmt(rm, 2), f.rss > 0 ? `+${S.fmt(100 * (rm - f.rss) / f.rss, 1)}% above the minimum` : '');
        } else sMan.set('—', 'turn on the manual line');
        sR2.set(f.Syy > 0 ? S.fmt(1 - f.rss / f.Syy, 3) : '—');
        sN.set(String(f.n));
      }
      newData();
    },
  });

  Atlas.lab('simple-linear-regression-sampling', {
    title: 'Design spread controls precision',
    blurb: R`Repeatedly draw $Y_i=2+0.5x_i+\varepsilon_i$ at fixed design points and refit. The histogram of $\hat\beta_1$ matches $\N(\beta_1,\sigma^2/S_{xx})$. Narrow the spread of the $x$'s and the slope estimates fan out — far faster than reducing $n$ does. "Two ends" is the variance-minimizing design.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 4, max: 100, step: 1, value: 20, onChange: reset });
      const w = UI.slider(controls, { label: 'Half-width of $x$ range', min: 0.3, max: 5, step: 0.1, value: 2.5, onChange: reset });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.2, max: 3, step: 0.1, value: 1, onChange: reset });
      const des = UI.segmented(controls, { label: 'Design', options: [{ value: 'even', label: 'Evenly spread' }, { value: 'ends', label: 'Two ends' }], value: 'even', onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Draw samples', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 10], y: [-2, 12], xlabel: 'x', ylabel: 'y', title: 'Latest sample; last 25 fitted lines' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'β̂₁', ylabel: 'density', title: 'Sampling distribution of the slope' });
      const sSxx = UI.stat(stats, '$S_{xx}$');
      const sTh = UI.stat(stats, 'Theory $\\sigma/\\sqrt{S_{xx}}$', { accent: 'accent' });
      const sEm = UI.stat(stats, 'Empirical SD of $\\hat\\beta_1$');
      const sCov = UI.stat(stats, '95% $t$-interval coverage');
      const sCnt = UI.stat(stats, 'Samples');
      const B0 = 2, B1 = 0.5;
      let xs, Sxx, reps, lines, last, cover;
      function design() {
        const nn = Math.round(n.get()), ww = w.get();
        if (des.get() === 'ends') return S.range(nn).map(i => (i < nn / 2 ? 5 - ww : 5 + ww));
        return S.linspace(5 - ww, 5 + ww, nn);
      }
      let tcrit = 2;
      function reset() {
        loop.stop();
        xs = design(); const m = S.mean(xs); Sxx = xs.reduce((s, v) => s + (v - m) ** 2, 0);
        tcrit = S.q.t(0.975, Math.max(1, xs.length - 2));
        reps = []; lines = []; cover = 0; last = null;
        for (let i = 0; i < 200; i++) one();
        draw();
      }
      function one() {
        const y = xs.map(x => B0 + B1 * x + ctx.rng.normal(0, sig.get()));
        const f = slr(xs, y);
        reps.push(f.b1);
        if (Math.abs(f.b1 - B1) <= tcrit * Math.sqrt(f.s2 / f.sxx)) cover++;
        lines.push([f.b0, f.b1]); if (lines.length > 25) lines.shift();
        last = y;
      }
      function step() {
        for (let i = 0; i < 40; i++) one();
        draw();
        if (reps.length >= 5000) return false;
      }
      function draw() {
        pL.clear();
        for (const [a, b] of lines) pL.line([[0, a], [10, a + 10 * b]], { color: 's1', width: 1, opacity: 0.3 });
        pL.line([[0, B0], [10, B0 + 10 * B1]], { color: 'ink', width: 2 });
        if (last) pL.points(xs.map((x, i) => [x, last[i]]), { color: 's2', r: 3 });
        pL.legend([{ label: 'true line', color: 'ink' }, { label: 'fitted lines', color: 's1' }], { left: true });
        const sd = sig.get() / Math.sqrt(Sxx);
        const lo = B1 - 4.5 * sd, hi = B1 + 4.5 * sd;
        const hb = S.histogram(reps, { bins: 40, lo, hi });
        const grid = S.linspace(lo, hi, 200), dens = grid.map(v => S.pdf.normal(v, B1, sd));
        pR.setDomain([lo, hi], [0, Math.max(...dens, ...hb.map(b => b.density)) * 1.12]).clear();
        pR.bars(hb, { color: 's1', opacity: 0.45 });
        pR.line(grid.map((g, i) => [g, dens[i]]), { color: 'ink', width: 2 });
        pR.vline(B1, { color: 'ink' });
        pR.legend([{ label: 'simulated β̂₁', color: 's1', fill: true }, { label: 'N(β₁, σ²/Sxx)', color: 'ink' }]);
        sSxx.set(S.fmt(Sxx, 1));
        sTh.set(S.fmt(sd, 4));
        sEm.set(S.fmt(S.sd(reps), 4));
        sCov.set(pct(cover / reps.length));
        sCnt.set(String(reps.length));
      }
      reset();
    },
  });

  /* ================================================================ */
  /* ols-geometry                                                     */
  /* ================================================================ */

  Atlas.lab('ols-geometry-projection', {
    title: 'Least squares is orthogonal projection',
    blurb: R`Here $n=3$: the response $y\in\R^3$ (orange) and two predictor columns $x_1,x_2$ (blue) spanning a plane, the column space $\mathcal C(X)$. The fit $\hat y$ (green) is the foot of the perpendicular from $y$; the residual $e$ (red, dashed) is orthogonal to the plane. Rotate the view to check the right angle, then shrink the angle between $x_1$ and $x_2$: $\hat y$ does not move, but the coefficients needed to reach it explode.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const y1 = UI.slider(controls, { label: '$y_1$', min: -2, max: 3, step: 0.05, value: 1.6, onInput: draw });
      const y2 = UI.slider(controls, { label: '$y_2$', min: -2, max: 3, step: 0.05, value: 1.2, onInput: draw });
      const y3 = UI.slider(controls, { label: '$y_3$', min: -2, max: 3, step: 0.05, value: 2.4, onInput: draw });
      const ang = UI.slider(controls, { label: 'Angle between $x_1$ and $x_2$ (degrees)', min: 2, max: 90, step: 1, value: 60, onInput: draw });
      const az = UI.slider(controls, { label: 'View azimuth', min: -180, max: 180, step: 1, value: -35, onInput: draw });
      const elv = UI.slider(controls, { label: 'View elevation', min: -10, max: 80, step: 1, value: 22, onInput: draw });
      const p = new Plot(stage, { width: 640, height: 360, x: [-4, 4], y: [-3, 3.4], equal: true, grid: false, xticks: 0, yticks: 0 });
      const sB = UI.stat(stats, '$\\hat\\beta=(\\hat\\beta_1,\\hat\\beta_2)$', { accent: 'accent' });
      const sO = UI.stat(stats, '$x_1^\\top e,\\ x_2^\\top e$');
      const sP = UI.stat(stats, '$\\|y\\|^2=\\|\\hat y\\|^2+\\|e\\|^2$');
      const sV = UI.stat(stats, '$\\Var\\hat\\beta_1/\\sigma^2$');
      const norm = v => { const s = Math.sqrt(sq(v)); return v.map(t => t / s); };
      const a = norm([1, 0.15, 0.35]);
      let c = [-0.2, 1, 0.1]; const ca = dot(c, a); c = norm(c.map((t, i) => t - ca * a[i]));
      const add = (u, v, k = 1) => u.map((t, i) => t + k * v[i]);
      const mul = (u, k) => u.map(t => t * k);
      function proj(v) {
        const A = az.get() * Math.PI / 180, E = elv.get() * Math.PI / 180;
        const X = Math.cos(A) * v[0] - Math.sin(A) * v[1], D = Math.sin(A) * v[0] + Math.cos(A) * v[1];
        return [X, v[2] * Math.cos(E) + D * Math.sin(E)];
      }
      const seg = (u, v, o) => { const P = proj(u), Q = proj(v); p.segment(P[0], P[1], Q[0], Q[1], o); };
      const arr = (u, v, o) => { const P = proj(u), Q = proj(v); p.arrow(P[0], P[1], Q[0], Q[1], o); };
      const lab = (v, s, o) => { const P = proj(v); p.text(P[0], P[1], s, Object.assign({ dx: 6, dy: -6, size: 13, weight: 600 }, o)); };
      function draw() {
        const phi = ang.get() * Math.PI / 180;
        const x1 = mul(a, 2), x2 = mul(add(mul(a, Math.cos(phi)), c, Math.sin(phi)), 2);
        const y = [y1.get(), y2.get(), y3.get()];
        const X = [0, 1, 2].map(i => [x1[i], x2[i]]);
        const f = fitLS(X, y);
        const yh = add(mul(x1, f.beta[0]), x2, f.beta[1]), e = add(y, yh, -1);
        p.clear();
        const O = [0, 0, 0], L = 3.2;
        const corners = [add(mul(a, L), c, L), add(mul(a, L), c, -L), add(mul(a, -L), c, -L), add(mul(a, -L), c, L)];
        p.polygon(corners.map(proj), { color: 's1', opacity: 0.08, stroke: 's1', width: 0.8 });
        for (let k = -3; k <= 3; k++) { seg(add(mul(a, k), c, -L), add(mul(a, k), c, L), { color: 's1', width: 0.5, opacity: 0.25 }); seg(add(mul(c, k), a, -L), add(mul(c, k), a, L), { color: 's1', width: 0.5, opacity: 0.25 }); }
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((u, i) => { arr(O, mul(u, 1.2), { color: 'faint', width: 1 }); lab(mul(u, 1.25), ['e₁', 'e₂', 'e₃'][i], { color: 'muted', size: 11, weight: 400 }); });
        // parallelogram: yhat = b1 x1 + b2 x2
        const c1 = mul(x1, f.beta[0]), c2 = mul(x2, f.beta[1]);
        seg(O, c1, { color: 's1', width: 1, dash: '3 3', opacity: 0.7 });
        seg(c1, yh, { color: 's1', width: 1, dash: '3 3', opacity: 0.7 });
        seg(O, c2, { color: 's1', width: 1, dash: '3 3', opacity: 0.7 });
        seg(c2, yh, { color: 's1', width: 1, dash: '3 3', opacity: 0.7 });
        arr(O, x1, { color: 's1', width: 2.2 }); lab(x1, 'x₁', { color: 's1' });
        arr(O, x2, { color: 's1', width: 2.2 }); lab(x2, 'x₂', { color: 's1' });
        arr(O, yh, { color: 's3', width: 2.4 }); lab(yh, 'ŷ = Hy', { color: 's3', dy: 16 });
        seg(yh, y, { color: 'bad', width: 2, dash: '5 4' });
        // right-angle marker
        const en = Math.sqrt(sq(e));
        if (en > 0.15) {
          const u = mul(e, 0.22 / en), wv = norm(yh.some(t => Math.abs(t) > 1e-6) ? yh : a), w2 = mul(wv, 0.22);
          const P1 = add(yh, u), P2 = add(add(yh, u), w2), P3 = add(yh, w2);
          seg(P1, P2, { color: 'bad', width: 1.2 }); seg(P2, P3, { color: 'bad', width: 1.2 });
        }
        arr(O, y, { color: 's2', width: 2.4 }); lab(y, 'y', { color: 's2' });
        lab(add(yh, e, 0.5), 'e', { color: 'bad' });
        p.text(-3.9, 3.1, 'shaded plane = column space C(X)', { size: 11, color: 'muted' });
        sB.set(`(${S.fmt(f.beta[0], 2)}, ${S.fmt(f.beta[1], 2)})`);
        sO.set(`${S.fmt(Math.abs(dot(x1, e)) < 5e-13 ? 0 : dot(x1, e), 3)}, ${S.fmt(Math.abs(dot(x2, e)) < 5e-13 ? 0 : dot(x2, e), 3)}`, 'normal equations: residual ⟂ columns');
        sP.set(`${S.fmt(sq(y), 2)} = ${S.fmt(sq(yh), 2)} + ${S.fmt(sq(e), 2)}`);
        sV.set(S.fmt(f.Ai[0][0], 3), `grows like 1/sin²(angle) as columns align`);
      }
      draw();
    },
  });

  Atlas.lab('ols-geometry-fwl', {
    title: 'Frisch–Waugh–Lovell and omitted-variable bias',
    blurb: R`Data from $y=\beta_1x_1+\beta_2x_2+\varepsilon$ with $\Corr(x_1,x_2)=\rho$. Left: the simple regression of $y$ on $x_1$ alone, whose slope absorbs part of $x_2$'s effect. Right: the added-variable plot — both $y$ and $x_1$ residualized on $x_2$ — whose slope is <em>exactly</em> the multiple-regression coefficient $\hat\beta_1$. Set $\beta_2=0$ or $\rho=0$ to make the bias vanish.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rho = UI.slider(controls, { label: 'Correlation $\\rho$ of $x_1,x_2$', min: -0.95, max: 0.95, step: 0.05, value: 0.7, onInput: draw });
      const b1 = UI.slider(controls, { label: 'True $\\beta_1$', min: -1, max: 2, step: 0.1, value: 1, onInput: draw });
      const b2 = UI.slider(controls, { label: 'True $\\beta_2$ (omitted in the left panel)', min: -2, max: 2, step: 0.1, value: 1.5, onInput: draw });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 20, max: 400, step: 10, value: 100, onChange: gen });
      UI.button(controls, 'New sample', gen, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'x₁', ylabel: 'y', title: 'y on x₁ alone (x₂ omitted)' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'x₁ residualized on x₂', ylabel: 'y residualized on x₂', title: 'Added-variable plot' });
      const sS = UI.stat(stats, 'Simple slope $\\tilde\\beta_1$');
      const sM = UI.stat(stats, 'Multiple $\\hat\\beta_1$ = AVP slope', { accent: 'accent' });
      const sO = UI.stat(stats, '$\\hat\\beta_1+\\hat\\beta_2\\hat\\delta$');
      const sV = UI.stat(stats, 'VIF $=1/(1-r_{12}^2)$');
      let z1, z2, eps;
      function gen() { const n = Math.round(nS.get()); z1 = ctx.rng.array(n, () => ctx.rng.normal()); z2 = ctx.rng.array(n, () => ctx.rng.normal()); eps = ctx.rng.array(n, () => ctx.rng.normal()); draw(); }
      const resid = (u, v) => { const f = slr(v, u); return u.map((t, i) => t - f.b0 - f.b1 * v[i]); };
      function draw() {
        const r = rho.get(), n = z1.length;
        const x1 = z1.slice(), x2 = z1.map((t, i) => r * t + Math.sqrt(1 - r * r) * z2[i]);
        const y = x1.map((t, i) => b1.get() * t + b2.get() * x2[i] + eps[i]);
        const simple = slr(x1, y);
        const full = fitLS(S.range(n).map(i => [1, x1[i], x2[i]]), y);
        const rx = resid(x1, x2), ry = resid(y, x2);
        const avp = slr(rx, ry);
        const delta = slr(x1, x2).b1;
        // left
        const ex = S.extent(x1), ey = S.extent(y), px = (ex[1] - ex[0]) * 0.05, py = (ey[1] - ey[0]) * 0.06;
        pL.setDomain([ex[0] - px, ex[1] + px], [ey[0] - py, ey[1] + py]).clear();
        pL.points(x1.map((t, i) => [t, y[i]]), { color: 'muted', r: 2.5, opacity: 0.6 });
        pL.line([[ex[0] - px, simple.b0 + simple.b1 * (ex[0] - px)], [ex[1] + px, simple.b0 + simple.b1 * (ex[1] + px)]], { color: 's2', width: 2.5 });
        const mx = S.mean(x1), my = S.mean(y);
        pL.line([[ex[0] - px, my + full.beta[1] * (ex[0] - px - mx)], [ex[1] + px, my + full.beta[1] * (ex[1] + px - mx)]], { color: 's1', width: 2, dash: true });
        pL.legend([{ label: `simple slope ${S.fmt(simple.b1, 2)}`, color: 's2' }, { label: `multiple β̂₁ ${S.fmt(full.beta[1], 2)}`, color: 's1', dash: true }], { left: true });
        // right
        const erx = S.extent(rx), ery = S.extent(ry), qx = (erx[1] - erx[0]) * 0.05 || 0.1, qy = (ery[1] - ery[0]) * 0.06 || 0.1;
        pR.setDomain([erx[0] - qx, erx[1] + qx], [ery[0] - qy, ery[1] + qy]).clear();
        pR.points(rx.map((t, i) => [t, ry[i]]), { color: 's1', r: 2.5, opacity: 0.6 });
        pR.line([[erx[0] - qx, avp.b0 + avp.b1 * (erx[0] - qx)], [erx[1] + qx, avp.b0 + avp.b1 * (erx[1] + qx)]], { color: 's1', width: 2.5 });
        pR.line([[erx[0] - qx, b1.get() * (erx[0] - qx)], [erx[1] + qx, b1.get() * (erx[1] + qx)]], { color: 'ink', width: 1.2, dash: '2 4' });
        pR.legend([{ label: `AVP slope ${S.fmt(avp.b1, 3)}`, color: 's1' }, { label: 'true β₁', color: 'ink', dash: true }], { left: true });
        sS.set(S.fmt(simple.b1, 3), `bias ≈ β₂·δ = ${S.fmt(b2.get() * r, 2)} in population`);
        sM.set(S.fmt(full.beta[1], 3), `AVP slope ${S.fmt(avp.b1, 3)}`);
        sO.set(S.fmt(full.beta[1] + full.beta[2] * delta, 3), `= simple slope exactly (δ̂ = ${S.fmt(delta, 2)})`);
        const r12 = S.cor(x1, x2);
        sV.set(S.fmt(1 / (1 - r12 * r12), 2), `r₁₂ = ${S.fmt(r12, 2)}`);
      }
      gen();
    },
  });

  /* ================================================================ */
  /* gauss-markov                                                     */
  /* ================================================================ */

  Atlas.lab('gauss-markov-race', {
    title: 'Racing linear unbiased estimators',
    blurb: R`Five slope estimators at design $x_i=1,\dots,n$, evaluated by simulation. Under homoscedastic errors (Gauss–Markov holds) OLS has the smallest variance among the linear ones, as the theorem promises. Switch to heteroscedastic errors and correctly weighted least squares wins; switch to heavy-tailed (contaminated) errors and the nonlinear Theil–Sen estimator wins. Neither contradicts the theorem — each breaks one premise.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const errs = UI.segmented(controls, { label: 'Errors', options: [{ value: 'norm', label: 'Normal, equal variance' }, { value: 'het', label: 'SD ∝ x' }, { value: 'cont', label: 'Heavy-tailed' }], value: 'norm', onChange: reset });
      const nS = UI.slider(controls, { label: 'Sample size $n$ (even)', min: 6, max: 60, step: 2, value: 20, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Simulate', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 270, xlabel: 'slope estimate', ylabel: 'density', title: 'Sampling distributions' });
      const pB = new Plot(stage, { width: 420, height: 270, x: [0, 5], y: [0, 4], ylabel: 'variance ÷ OLS variance', title: 'Relative variance (lower is better)', xticks: 0 });
      const sN = UI.stat(stats, 'Replications');
      const sBest = UI.stat(stats, 'Lowest variance', { accent: 'accent' });
      const sOls = UI.stat(stats, 'OLS variance');
      const EST = [
        { name: 'OLS', color: 's1', linear: true },
        { name: 'Endpoints', color: 's5', linear: true },
        { name: 'Grouping', color: 's6', linear: true },
        { name: 'WLS 1/x²', color: 's3', linear: true },
        { name: 'Theil–Sen', color: 's2', linear: false },
      ];
      let vals, xs, theory;
      function reset() {
        loop.stop();
        const n = Math.round(nS.get());
        xs = S.range(n).map(i => i + 1);
        const m = S.mean(xs), Sxx = xs.reduce((s, v) => s + (v - m) ** 2, 0), h = n / 2;
        const dxg = S.mean(xs.slice(h)) - S.mean(xs.slice(0, h));
        theory = [1, (2 / (n - 1) ** 2) * Sxx, (4 / n) / dxg ** 2 * Sxx];
        vals = EST.map(() => []);
        for (let i = 0; i < 150; i++) one();
        draw();
      }
      const sigAt = x => (errs.get() === 'het' ? x / S.mean(xs) : 1);
      function err(x) {
        if (errs.get() === 'cont') return ctx.rng.u() < 0.1 ? ctx.rng.normal(0, 6) : ctx.rng.normal(0, 1);
        return ctx.rng.normal(0, sigAt(x));
      }
      function one() {
        const n = xs.length, y = xs.map(x => 1 + 1 * x + err(x));
        vals[0].push(slr(xs, y).b1);
        vals[1].push((y[n - 1] - y[0]) / (xs[n - 1] - xs[0]));
        const h = n / 2;
        vals[2].push((S.mean(y.slice(h)) - S.mean(y.slice(0, h))) / (S.mean(xs.slice(h)) - S.mean(xs.slice(0, h))));
        // WLS with weights 1/x^2
        let sw = 0, swx = 0, swy = 0, swxx = 0, swxy = 0;
        for (let i = 0; i < n; i++) { const wt = 1 / (xs[i] * xs[i]); sw += wt; swx += wt * xs[i]; swy += wt * y[i]; swxx += wt * xs[i] * xs[i]; swxy += wt * xs[i] * y[i]; }
        vals[3].push((sw * swxy - swx * swy) / (sw * swxx - swx * swx));
        const sl = [];
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) sl.push((y[j] - y[i]) / (xs[j] - xs[i]));
        vals[4].push(S.median(sl));
      }
      function step() {
        const per = Math.max(3, Math.round(3000 / (xs.length * xs.length)));
        for (let i = 0; i < per; i++) one();
        draw();
        if (vals[0].length >= 4000) return false;
      }
      function draw() {
        const vars = vals.map(v => S.variance(v));
        const sdO = Math.sqrt(vars[0]);
        const sdMax = Math.sqrt(Math.max(...vars));
        const lo = 1 - 3.2 * Math.min(sdMax, 4 * sdO), hi = 1 + 3.2 * Math.min(sdMax, 4 * sdO);
        const curves = vals.map(v => S.histogram(v, { bins: 36, lo, hi }).map(b => [(b.x0 + b.x1) / 2, b.density]));
        const ymax = Math.max(...curves.flat().map(q => q[1])) * 1.1 || 1;
        pD.setDomain([lo, hi], [0, ymax]).clear();
        curves.forEach((cv, k) => pD.line(cv, { color: EST[k].color, width: k === 0 ? 2.6 : 1.8, opacity: 0.9 }));
        pD.vline(1, { color: 'ink' });
        pD.legend(EST.map(e => ({ label: e.name, color: e.color })), { left: true });
        const ratios = vars.map(v => v / vars[0]);
        const top = Math.max(2, Math.min(12, Math.max(...ratios) * 1.18));
        pB.setDomain([0, 5], [0, top]).clear();
        pB.hline(1, { color: 'ink', dash: '4 4', width: 1.2 });
        ratios.forEach((r, k) => {
          pB.bars([{ x0: k + 0.15, x1: k + 0.85, y: Math.min(r, top) }], { color: EST[k].color, opacity: 0.75 });
          pB.text(k + 0.5, Math.min(r, top), S.fmt(r, 2), { anchor: 'middle', dy: -5, size: 11, weight: 600 });
          pB.text(k + 0.5, 0, EST[k].name, { anchor: 'middle', dy: 15, size: 10, color: 'muted' });
          if (errs.get() !== 'het' && k < 3 && errs.get() === 'norm') pB.segment(k + 0.1, theory[k], k + 0.9, theory[k], { color: 'ink', width: 2 });
        });
        if (errs.get() === 'norm') pB.text(4.95, top * 0.95, '— theory (Gauss–Markov setting)', { anchor: 'end', size: 10, color: 'muted' });
        sN.set(String(vals[0].length));
        const kb = S.argmin(vars);
        sBest.set(EST[kb].name, EST[kb].linear ? 'a linear unbiased estimator' : 'nonlinear: outside the theorem');
        sOls.set(S.fmt(vars[0], 5));
      }
      reset();
    },
  });

  /* ================================================================ */
  /* regression-inference                                             */
  /* ================================================================ */

  Atlas.lab('regression-inference-bands', {
    title: 'Confidence band versus prediction band',
    blurb: R`The inner band is for the regression line $\E(Y\mid x)$; the outer band is for a new observation $Y_0$. Increase $n$: the inner band collapses toward the line, the outer one does not. Toggle the simultaneous (Working–Hotelling) band, then switch to skewed errors and run the coverage check: the confidence interval still covers about 95%, but the prediction interval misses unevenly in the two tails.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 200, step: 1, value: 15, onChange: gen });
      const sig = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.2, max: 3, step: 0.1, value: 1, onChange: gen });
      const lev = UI.select(controls, { label: 'Level', options: [{ value: '0.8', label: '80%' }, { value: '0.9', label: '90%' }, { value: '0.95', label: '95%' }, { value: '0.99', label: '99%' }], value: '0.95', onChange: draw });
      const dist = UI.segmented(controls, { label: 'Error distribution', options: [{ value: 'norm', label: 'Normal' }, { value: 'skew', label: 'Skewed (exponential)' }], value: 'norm', onChange: gen });
      const wh = UI.toggle(controls, { label: 'Simultaneous (Working–Hotelling) band', value: false, onChange: draw });
      const x0s = UI.slider(controls, { label: 'Evaluate at $x_0$', min: 0, max: 12, step: 0.1, value: 9, onInput: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', gen, { primary: true });
      UI.button(row, 'Coverage check (2000 samples)', coverage);
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 12], y: [-2, 12], xlabel: 'x', ylabel: 'y' });
      const sCI = UI.stat(stats, 'CI width at $x_0$', { accent: 'accent' });
      const sPI = UI.stat(stats, 'PI width at $x_0$');
      const sCovC = UI.stat(stats, 'CI coverage of $\\E(Y_0)$');
      const sCovP = UI.stat(stats, 'PI coverage of $Y_0$');
      const B0 = 2, B1 = 0.5;
      let x = [], y = [];
      const errDraw = () => (dist.get() === 'skew' ? sig.get() * (ctx.rng.exp(1) - 1) : ctx.rng.normal(0, sig.get()));
      function gen() {
        const n = Math.round(nS.get());
        x = ctx.rng.array(n, () => ctx.rng.uniform(0, 10));
        y = x.map(v => B0 + B1 * v + errDraw());
        sCovC.set('—', 'press coverage check'); sCovP.set('—');
        draw();
      }
      function crit(n) {
        const a = 1 - (+lev.get());
        return { t: S.q.t(1 - a / 2, n - 2), w: Math.sqrt(2 * S.q.f(1 - a, 2, n - 2)) };
      }
      function draw() {
        const f = slr(x, y), n = x.length, s = Math.sqrt(f.s2), c = crit(n);
        const m = wh.get() ? c.w : c.t;
        const g = S.linspace(0, 12, 121);
        const seM = v => s * Math.sqrt(1 / n + (v - f.mx) ** 2 / f.sxx);
        const seP = v => s * Math.sqrt(1 + 1 / n + (v - f.mx) ** 2 / f.sxx);
        const yh = v => f.b0 + f.b1 * v;
        const ylo = Math.min(-2, ...y) - 0.5, yhi = Math.max(10, ...y) + 0.5;
        p.setDomain([0, 12], [ylo, yhi]).clear();
        p.band(10, 12, { color: 'faint', opacity: 0.25 });
        p.text(11, yhi, 'no data here', { anchor: 'middle', dy: 14, size: 11, color: 'muted' });
        p.area(g.map(v => [v, yh(v) + c.t * seP(v)]), { base: g.map(v => [v, yh(v) - c.t * seP(v)]), color: 's2', opacity: 0.12 });
        p.area(g.map(v => [v, yh(v) + m * seM(v)]), { base: g.map(v => [v, yh(v) - m * seM(v)]), color: 's1', opacity: 0.28 });
        p.line([[0, B0], [12, B0 + 12 * B1]], { color: 'ink', width: 1.5, dash: '2 4' });
        p.line(g.map(v => [v, yh(v)]), { color: 's1', width: 2.5 });
        p.points(x.map((v, i) => [v, y[i]]), { color: 'ink', r: 3, opacity: 0.75 });
        const x0 = x0s.get();
        p.vline(x0, { color: 's3', width: 1.2 });
        p.legend([{ label: wh.get() ? 'simultaneous band for the line' : 'pointwise CI for the line', color: 's1', fill: true, opacity: 0.4 }, { label: 'prediction band for new Y', color: 's2', fill: true, opacity: 0.2 }, { label: 'true line', color: 'ink', dash: true }], { left: true });
        sCI.set(S.fmt(2 * m * seM(x0), 2), wh.get() ? `multiplier √(2F) = ${S.fmt(c.w, 2)}` : `multiplier t = ${S.fmt(c.t, 2)}`);
        sPI.set(S.fmt(2 * c.t * seP(x0), 2), `limit as n→∞: ${S.fmt(2 * S.qnorm(1 - (1 - (+lev.get())) / 2) * sig.get(), 2)}`);
      }
      function coverage() {
        const n = x.length, x0 = x0s.get(), mu0 = B0 + B1 * x0, c = crit(n);
        let cc = 0, pc = 0, lowMiss = 0, highMiss = 0;
        const NR = 2000;
        for (let r = 0; r < NR; r++) {
          const yy = x.map(v => B0 + B1 * v + errDraw());
          const f = slr(x, yy), s = Math.sqrt(f.s2), yh = f.b0 + f.b1 * x0;
          const hm = c.t * s * Math.sqrt(1 / n + (x0 - f.mx) ** 2 / f.sxx), hp = c.t * s * Math.sqrt(1 + 1 / n + (x0 - f.mx) ** 2 / f.sxx);
          if (Math.abs(yh - mu0) <= hm) cc++;
          const y0 = mu0 + errDraw();
          if (y0 < yh - hp) lowMiss++; else if (y0 > yh + hp) highMiss++; else pc++;
        }
        sCovC.set(pct(cc / NR), `nominal ${pct(+lev.get())} (pointwise t)`);
        sCovP.set(pct(pc / NR), `misses: ${pct(lowMiss / NR)} below, ${pct(highMiss / NR)} above`);
      }
      gen();
    },
  });

  Atlas.lab('regression-inference-ellipse', {
    title: 'Joint F versus individual t',
    blurb: R`Two correlated predictors, both with true coefficient $b$. Left: one sample's 95% joint confidence ellipse for $(\beta_1,\beta_2)$ and the rectangle of individual 95% $t$-intervals. Right: many simulated $(\hat\beta_1,\hat\beta_2)$, coloured by verdict. Push $\rho$ toward $0.98$: the ellipse becomes a cigar, and frequently the $F$-test rejects $\beta_1=\beta_2=0$ while neither $t$-test rejects its own coefficient.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rho = UI.slider(controls, { label: 'Predictor correlation $\\rho$', min: 0, max: 0.99, step: 0.01, value: 0.95, onChange: reset });
      const bS = UI.slider(controls, { label: 'True $\\beta_1=\\beta_2=b$', min: 0, max: 1, step: 0.05, value: 0.3, onChange: reset });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 200, step: 5, value: 30, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Simulate', '❚❚ Pause']);
      UI.button(row, 'New sample', () => { cur = sample(); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, equal: true, xlabel: 'β₁', ylabel: 'β₂', title: 'One sample: ellipse vs rectangle' });
      const pR = new Plot(stage, { width: 420, height: 300, equal: true, xlabel: 'β̂₁', ylabel: 'β̂₂', title: 'Repeated samples' });
      const sF = UI.stat(stats, '$F$ rejects');
      const sT = UI.stat(stats, 'Some $t$ rejects');
      const sFo = UI.stat(stats, '$F$ rejects, no $t$ does', { accent: 'accent' });
      const sN = UI.stat(stats, 'Samples');
      let X, cur, sims, cnt;
      function design() {
        const n = Math.round(nS.get()), r = rho.get();
        X = S.range(n).map(() => { const z1 = ctx.rng.normal(), z2 = ctx.rng.normal(); return [1, z1, r * z1 + Math.sqrt(1 - r * r) * z2]; });
      }
      let tc, Fc;
      function sample() {
        const b = bS.get(), n = X.length;
        const y = X.map(r => b * r[1] + b * r[2] + ctx.rng.normal());
        const f = fitLS(X, y);
        const C = [[f.Ai[1][1], f.Ai[1][2]], [f.Ai[2][1], f.Ai[2][2]]], Ci = S.inv(C);
        const d = [f.beta[1], f.beta[2]];
        const F = (d[0] * (Ci[0][0] * d[0] + Ci[0][1] * d[1]) + d[1] * (Ci[1][0] * d[0] + Ci[1][1] * d[1])) / (2 * f.s2);
        const t1 = d[0] / Math.sqrt(f.s2 * C[0][0]), t2 = d[1] / Math.sqrt(f.s2 * C[1][1]);
        return { d, C, s2: f.s2, F, t1, t2, rejF: F > Fc, rejT: Math.abs(t1) > tc || Math.abs(t2) > tc, n };
      }
      function reset() {
        loop.stop();
        design();
        const n = X.length; tc = S.q.t(0.975, n - 3); Fc = S.q.f(0.95, 2, n - 3);
        sims = []; cnt = { F: 0, T: 0, Fo: 0 };
        cur = sample();
        for (let i = 0; i < 100; i++) add();
        draw();
      }
      function add() { const s = sample(); sims.push(s); if (s.rejF) cnt.F++; if (s.rejT) cnt.T++; if (s.rejF && !s.rejT) cnt.Fo++; }
      function step() { for (let i = 0; i < 25; i++) add(); draw(); if (sims.length >= 3000) return false; }
      function domain() {
        const b = bS.get(), sd = Math.sqrt(cur.s2 * Math.max(cur.C[0][0], cur.C[1][1])) || 0.5;
        const h = Math.max(3.2 * sd, b + 0.3);
        return [[b - h, b + h], [b - h, b + h]];
      }
      function draw() {
        const [dx, dy] = domain(), b = bS.get();
        pL.setDomain(dx, dy).clear();
        pR.setDomain(dx, dy).clear();
        for (const P of [pL, pR]) { P.hline(0, { color: 'faint', dash: false, width: 1 }); P.vline(0, { color: 'faint', dash: false, width: 1 }); }
        const k = Math.sqrt(2 * Fc), cov = cur.C.map(r => r.map(v => v * cur.s2));
        pL.ellipse(cur.d[0], cur.d[1], cov, k, { color: 's1', fill: true, opacity: 0.2, stroke: 's1' });
        const h1 = tc * Math.sqrt(cov[0][0]), h2 = tc * Math.sqrt(cov[1][1]);
        pL.rect(cur.d[0] - h1, cur.d[1] - h2, cur.d[0] + h1, cur.d[1] + h2, { color: 's2', opacity: 0.06, stroke: 's2' });
        pL.dot(0, 0, { color: 'ink', r: 4 });
        pL.text(0, 0, 'origin', { dx: 6, dy: 14, size: 11, color: 'muted' });
        pL.dot(b, b, { color: 'ink', r: 3, opacity: 0.6 });
        pL.dot(cur.d[0], cur.d[1], { color: 's1', r: 4.5 });
        pL.legend([{ label: '95% joint ellipse (F)', color: 's1', fill: true }, { label: 'individual t-intervals', color: 's2' }]);
        const groups = { fo: [], t: [], none: [] };
        for (const s of sims) (s.rejF && !s.rejT ? groups.fo : s.rejT ? groups.t : groups.none).push(s.d);
        pR.points(groups.none, { color: 'muted', r: 1.6, opacity: 0.4 });
        pR.points(groups.t, { color: 's1', r: 1.6, opacity: 0.45 });
        pR.points(groups.fo, { color: 's4', r: 2, opacity: 0.8 });
        pR.dot(b, b, { color: 'ink', r: 4 });
        pR.legend([{ label: 'F only', color: 's4' }, { label: 'some t rejects', color: 's1' }, { label: 'nothing rejects', color: 'muted' }]);
        const N = sims.length;
        sF.set(pct(cnt.F / N), `this sample: F = ${S.fmt(cur.F, 2)} (crit ${S.fmt(Fc, 2)})`);
        sT.set(pct(cnt.T / N), `this sample: |t| = ${S.fmt(Math.abs(cur.t1), 2)}, ${S.fmt(Math.abs(cur.t2), 2)}`);
        sFo.set(pct(cnt.Fo / N));
        sN.set(String(N));
      }
      reset();
    },
  });

  /* ================================================================ */
  /* anova                                                            */
  /* ================================================================ */

  // F density (not in Atlas.S).
  function fpdf(x, d1, d2) {
    if (x <= 0) return 0;
    return Math.exp(0.5 * d1 * Math.log(d1 / d2) + (0.5 * d1 - 1) * Math.log(x) - 0.5 * (d1 + d2) * Math.log(1 + d1 * x / d2) - S.lbeta(d1 / 2, d2 / 2));
  }

  Atlas.lab('anova-decomposition', {
    title: 'Between-group versus within-group variation',
    blurb: R`Each column is a group; short bars are group means, the horizontal line is the grand mean. The $F$ statistic compares the spread of the group means (scaled by $n$) with the spread inside groups. Move the true effect spread to zero to see $F$ fluctuate around 1 under $H_0$; increase $n$ per group and watch power climb, since the noncentrality $\lambda=\sum_in_i(\mu_i-\bar\mu)^2/\sigma^2$ grows linearly in $n$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const kS = UI.slider(controls, { label: 'Number of groups $k$', min: 2, max: 8, step: 1, value: 4, onChange: gen });
      const nS = UI.slider(controls, { label: 'Observations per group $n$', min: 2, max: 50, step: 1, value: 8, onChange: gen });
      const dS = UI.slider(controls, { label: 'True SD of group means $\\delta$', min: 0, max: 2, step: 0.05, value: 0.5, onInput: draw });
      const sS = UI.slider(controls, { label: 'Within-group SD $\\sigma$', min: 0.2, max: 3, step: 0.1, value: 1, onInput: draw });
      UI.button(controls, 'New sample', gen, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'group', ylabel: 'y', title: 'Data, group means, grand mean', xticks: 0 });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'F', ylabel: 'density', title: 'Null distribution of F' });
      const sSS = UI.stat(stats, 'SST = SSB + SSW');
      const sF = UI.stat(stats, '$F=\\text{MSB}/\\text{MSW}$', { accent: 'accent' });
      const sP = UI.stat(stats, '$p$-value');
      const sL = UI.stat(stats, 'Noncentrality $\\lambda$');
      const sPow = UI.stat(stats, 'Power at 5%');
      let eps, jit;
      function gen() {
        const k = Math.round(kS.get()), n = Math.round(nS.get());
        eps = S.range(k).map(() => ctx.rng.array(n, () => ctx.rng.normal()));
        jit = S.range(k).map(() => ctx.rng.array(n, () => ctx.rng.uniform(-0.22, 0.22)));
        draw();
      }
      function draw() {
        const k = eps.length, n = eps[0].length, N = k * n, d = dS.get(), sg = sS.get();
        const pat = S.linspace(-1, 1, k), psd = Math.sqrt(S.mean(pat.map(v => v * v))) || 1;
        const mu = pat.map(v => 10 + d * v / psd);
        const y = eps.map((row, i) => row.map(e => mu[i] + sg * e));
        const gm = y.map(r => S.mean(r)), grand = S.mean(gm);
        let ssb = 0, ssw = 0;
        y.forEach((r, i) => { ssb += n * (gm[i] - grand) ** 2; for (const v of r) ssw += (v - gm[i]) ** 2; });
        const d1 = k - 1, d2 = N - k, F = (ssb / d1) / (ssw / d2), pv = 1 - S.cdf.f(F, d1, d2);
        const lam = n * k * d * d / (sg * sg);
        // left
        const all = y.flat(), e = S.extent(all), pad = (e[1] - e[0]) * 0.08 + 0.1;
        pL.setDomain([0.4, k + 0.6], [e[0] - pad, e[1] + pad]).clear();
        pL.hline(grand, { color: 'ink', dash: false, width: 1.5 });
        y.forEach((r, i) => {
          const xc = i + 1;
          r.forEach((v, j) => pL.segment(xc + jit[i][j], v, xc + jit[i][j], gm[i], { color: 's4', width: 0.8, opacity: 0.35 }));
          pL.segment(xc, grand, xc, gm[i], { color: 's2', width: 4, opacity: 0.6 });
          pL.points(r.map((v, j) => [xc + jit[i][j], v]), { color: 'ink', r: 2.2, opacity: 0.6 });
          pL.segment(xc - 0.35, gm[i], xc + 0.35, gm[i], { color: 's1', width: 3 });
          pL.segment(xc - 0.35, mu[i], xc + 0.35, mu[i], { color: 'muted', width: 1, dash: '2 3' });
          pL.text(xc, e[0] - pad, String(xc), { anchor: 'middle', dy: 14, size: 11, color: 'muted' });
        });
        pL.legend([{ label: 'between (group mean − grand)', color: 's2', width: 4 }, { label: 'within (point − group mean)', color: 's4' }, { label: 'true mean', color: 'muted', dash: true }], { left: true });
        // right
        const Fc = S.q.f(0.95, d1, d2);
        const hi = Math.max(5, Fc * 1.8, Math.min(F * 1.15, 60));
        const g = S.linspace(hi / 400, hi, 300), dv = g.map(v => fpdf(v, d1, d2));
        const ym = Math.min(Math.max(...dv) * 1.1, 2.5);
        pR.setDomain([0, hi], [0, ym]).clear();
        const tail = g.filter(v => v >= Fc);
        if (tail.length > 1) pR.area(tail.map(v => [v, fpdf(v, d1, d2)]), { color: 'bad', opacity: 0.25 });
        pR.line(g.map((v, i) => [v, dv[i]]), { color: 'ink', width: 2 });
        pR.vline(Math.min(F, hi), { color: 's1', dash: false, width: 2.5 });
        pR.text(Math.min(F, hi), ym, F > hi ? `F = ${S.fmt(F, 1)} →` : 'observed F', { anchor: F > hi * 0.7 ? 'end' : 'start', dx: F > hi * 0.7 ? -6 : 6, dy: 14, size: 11, color: 's1' });
        pR.legend([{ label: `F(${d1}, ${d2}) density`, color: 'ink' }, { label: '5% rejection region', color: 'bad', fill: true, opacity: 0.3 }]);
        // power by simulating the noncentral F
        let rej = 0; const r = S.rng(7), M = 3000;
        for (let m = 0; m < M; m++) {
          const z = r.normal() + Math.sqrt(lam), num = (z * z + (d1 > 1 ? r.chisq(d1 - 1) : 0)) / d1, den = r.chisq(d2) / d2;
          if (num / den > Fc) rej++;
        }
        sSS.set(`${S.fmt(ssb + ssw, 1)} = ${S.fmt(ssb, 1)} + ${S.fmt(ssw, 1)}`, `df: ${N - 1} = ${d1} + ${d2}`);
        sF.set(S.fmt(F, 3), `critical value ${S.fmt(Fc, 2)}`);
        sP.set(pv < 1e-4 ? pv.toExponential(1) : S.fmt(pv, 4), pv < 0.05 ? bad('reject H₀ at 5%') : 'do not reject');
        sL.set(S.fmt(lam, 2), 'E[MSB]/σ² = 1 + λ/(k−1)');
        sPow.set(pct(rej / M), 'P(noncentral F > critical)');
      }
      gen();
    },
  });

  /* ================================================================ */
  /* regression-diagnostics                                           */
  /* ================================================================ */

  Atlas.lab('regression-diagnostics-influence', {
    title: "Leverage, outlyingness and Cook's distance",
    blurb: R`Drag the orange point (or click anywhere to move it). Left: the fit with it (solid) and without it (dashed). Right: every point's leverage $h_{ii}$ against its standardized residual $r_i$, with Cook's distance contours $D=0.5$ and $D=1$. Influence needs <em>both</em>: an outlier near the centre barely moves the line, a far-out point on the trend barely moves it, but a far-out point off the trend rotates it.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pre = UI.segmented(controls, { label: 'Place the orange point', options: [{ value: 'c', label: 'Outlier at centre' }, { value: 'on', label: 'Far out, on trend' }, { value: 'off', label: 'Far out, off trend' }], value: 'off', onChange: v => { sp = preset(v); draw(); } });
      UI.button(controls, 'New background data', gen, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 10], y: [-1, 12], xlabel: 'x', ylabel: 'y', title: 'Fit with and without the orange point' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 1], y: [-5, 5], xlabel: 'leverage hᵢᵢ', ylabel: 'standardized residual rᵢ', title: 'Residuals vs leverage' });
      const sH = UI.stat(stats, 'Leverage $h_{ii}$');
      const sR = UI.stat(stats, 'Studentized $t_i$');
      const sD = UI.stat(stats, "Cook's $D_i$", { accent: 'accent' });
      const sS = UI.stat(stats, 'Slope with / without');
      let base = [], sp;
      const preset = v => (v === 'c' ? [3.5, 1 + 0.8 * 3.5 + 3] : v === 'on' ? [9.5, 1 + 0.8 * 9.5] : [9.5, 2.5]);
      function gen() { base = ctx.rng.array(14, () => { const x = ctx.rng.uniform(1, 6); return [x, 1 + 0.8 * x + ctx.rng.normal(0, 0.5)]; }); draw(); }
      pL.onPointer((x, y, evt, type) => { if ((type === 'down' || type === 'drag') && pL.inside(x, y)) { sp = [x, y]; draw(); } });
      function draw() {
        const pts = base.concat([sp]), n = pts.length, p = 2;
        const X = pts.map(q => [1, q[0]]), y = pts.map(q => q[1]);
        const f = fitLS(X, y), g = slr(base.map(q => q[0]), base.map(q => q[1]));
        const s = Math.sqrt(f.s2);
        const r = f.res.map((e, i) => e / (s * Math.sqrt(Math.max(1e-12, 1 - f.h[i]))));
        const t = r.map(v => v * Math.sqrt((n - p - 1) / Math.max(1e-9, n - p - v * v)));
        const D = r.map((v, i) => v * v * f.h[i] / (p * Math.max(1e-12, 1 - f.h[i])));
        pL.clear();
        pL.line([[0, f.beta[0]], [10, f.beta[0] + 10 * f.beta[1]]], { color: 's1', width: 2.5 });
        pL.line([[0, g.b0], [10, g.b0 + 10 * g.b1]], { color: 'muted', width: 2, dash: true });
        pL.points(base, { color: 'ink', r: 3.2, opacity: 0.75 });
        pL.dot(sp[0], sp[1], { color: 's2', r: 6 });
        pL.legend([{ label: 'all points', color: 's1' }, { label: 'without orange', color: 'muted', dash: true }], { left: true });
        pR.clear();
        const hs = S.linspace(0.02, 0.995, 120);
        for (const [Dv, op] of [[0.5, 0.5], [1, 0.9]]) {
          const curve = hs.map(h => [h, Math.sqrt(Dv * p * (1 - h) / h)]);
          pR.line(curve, { color: 'bad', width: 1.3, dash: '4 3', opacity: op });
          pR.line(curve.map(([h, v]) => [h, -v]), { color: 'bad', width: 1.3, dash: '4 3', opacity: op });
          pR.text(0.97, Math.sqrt(Dv * p * 0.03 / 0.97) + 0.35, `D = ${Dv}`, { anchor: 'end', size: 10, color: 'bad' });
        }
        pR.vline(2 * p / n, { color: 'muted', width: 1 });
        pR.text(2 * p / n, 4.6, '2p/n', { dx: 4, size: 10, color: 'muted' });
        pR.hline(0, { color: 'faint', dash: false, width: 1 });
        pR.points(f.h.slice(0, n - 1).map((h, i) => [h, S.clamp(r[i], -4.9, 4.9)]), { color: 'ink', r: 3, opacity: 0.7 });
        pR.dot(f.h[n - 1], S.clamp(r[n - 1], -4.9, 4.9), { color: 's2', r: 6 });
        const k = n - 1;
        sH.set(S.fmt(f.h[k], 3), f.h[k] > 2 * p / n ? bad('above 2p/n = ' + S.fmt(2 * p / n, 2)) : `average p/n = ${S.fmt(p / n, 2)}`);
        sR.set(S.fmt(t[k], 2), `r = ${S.fmt(r[k], 2)}; reference t(${n - p - 1})`);
        sD.set(S.fmt(D[k], 3), D[k] > 1 ? bad('highly influential') : D[k] > 4 / n ? 'above 4/n' : 'not influential');
        sS.set(`${S.fmt(f.beta[1], 3)} / ${S.fmt(g.b1, 3)}`);
      }
      sp = preset('off');
      gen();
    },
  });

  Atlas.lab('regression-diagnostics-residuals', {
    title: 'A residual-plot gallery',
    blurb: R`A straight line is fitted to data from five different truths. Left: residuals against fitted values with a smoother; right: a normal QQ plot of studentized residuals. Learn each signature: the arch of curvature, the funnel of heteroscedasticity, the S-shaped QQ plot of heavy tails, the bent QQ plot of skewness. Then press "New sample" on the correct model a few times to calibrate how much pattern pure noise produces.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const truth = UI.select(controls, { label: 'True model', options: [
        { value: 'ok', label: 'Correct: linear, normal, constant variance' },
        { value: 'curve', label: 'Curvature in the mean' },
        { value: 'het', label: 'Variance grows with x' },
        { value: 'tails', label: 'Heavy-tailed errors (t₃)' },
        { value: 'skew', label: 'Right-skewed errors' },
      ], value: 'curve', onChange: gen });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 20, max: 400, step: 10, value: 100, onChange: gen });
      UI.button(controls, 'New sample', gen, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'fitted value', ylabel: 'residual', title: 'Residuals vs fitted' });
      const pR = new Plot(stage, { width: 420, height: 270, xlabel: 'normal quantile', ylabel: 'studentized residual', title: 'Normal QQ plot' });
      const sBP = UI.stat(stats, 'Breusch–Pagan $nR^2$');
      const sMax = UI.stat(stats, 'Largest $|t_i|$');
      const sSk = UI.stat(stats, 'Residual skewness');
      function gen() {
        const n = Math.round(nS.get()), tr = truth.get();
        const x = ctx.rng.array(n, () => ctx.rng.uniform(0, 10));
        const y = x.map(v => {
          let m = 1 + 0.5 * v, e;
          if (tr === 'curve') m += 0.12 * (v - 5) ** 2;
          if (tr === 'het') e = ctx.rng.normal(0, 0.1 + 0.2 * v);
          else if (tr === 'tails') e = 0.6 * ctx.rng.t(3);
          else if (tr === 'skew') e = 0.8 * (ctx.rng.exp(1) - 1);
          else e = ctx.rng.normal(0, 0.8);
          return m + e;
        });
        const f = fitLS(x.map(v => [1, v]), y), p = 2, s = Math.sqrt(f.s2);
        const r = f.res.map((e, i) => e / (s * Math.sqrt(1 - f.h[i])));
        const t = r.map(v => v * Math.sqrt((n - p - 1) / Math.max(1e-9, n - p - v * v)));
        // left: residuals vs fitted with Gaussian kernel smoother
        const ef = S.extent(f.fit), er = S.extent(f.res), padx = (ef[1] - ef[0]) * 0.05, pady = (er[1] - er[0]) * 0.08;
        pL.setDomain([ef[0] - padx, ef[1] + padx], [er[0] - pady, er[1] + pady]).clear();
        pL.hline(0, { color: 'ink', width: 1 });
        pL.points(f.fit.map((v, i) => [v, f.res[i]]), { color: 's1', r: 2.5, opacity: 0.6 });
        const bw = 0.12 * (ef[1] - ef[0]), grid = S.linspace(ef[0], ef[1], 60);
        pL.line(grid.map(g0 => { let sw = 0, swy = 0; for (let i = 0; i < n; i++) { const w = Math.exp(-0.5 * ((g0 - f.fit[i]) / bw) ** 2); sw += w; swy += w * f.res[i]; } return [g0, swy / sw]; }), { color: 's2', width: 2.5 });
        // right: QQ
        const ts = t.slice().sort((a, b) => a - b), qs = ts.map((_, i) => normQ(i, n));
        const lim = Math.max(3, Math.abs(ts[0]), Math.abs(ts[n - 1])) * 1.05, ql = Math.abs(qs[0]) * 1.1;
        pR.setDomain([-ql, ql], [-lim, lim]).clear();
        pR.line([[-ql, -ql], [ql, ql]], { color: 'ink', width: 1.2, dash: true });
        pR.points(qs.map((q, i) => [q, ts[i]]), { color: 's1', r: 2.5, opacity: 0.7 });
        // Breusch–Pagan (Koenker): regress e^2 on x
        const e2 = f.res.map(v => v * v), g = slr(x, e2), LM = n * g.r2, pBP = 1 - S.cdf.chisq(LM, 1);
        sBP.set(S.fmt(LM, 2), `p = ${S.fmt(pBP, 4)} vs χ²₁` + (pBP < 0.05 ? ' — ' + bad('heteroscedastic') : ''));
        const tmax = Math.max(...t.map(Math.abs)), pB = Math.min(1, n * 2 * (1 - S.cdf.t(tmax, n - p - 1)));
        sMax.set(S.fmt(tmax, 2), `Bonferroni p = ${S.fmt(pB, 3)}`);
        const m3 = S.mean(f.res.map(v => v ** 3)), m2 = S.mean(f.res.map(v => v * v));
        sSk.set(S.fmt(m3 / m2 ** 1.5, 2), 'about 0 for symmetric errors');
      }
      gen();
    },
  });

  /* ================================================================ */
  /* weighted-gls                                                     */
  /* ================================================================ */

  Atlas.lab('weighted-gls-sandwich', {
    title: 'Naive, sandwich, and weighted standard errors',
    blurb: R`Errors have SD proportional to $x^\gamma$. Each replication fits OLS and WLS (with the correct weights) at a fixed design and records whether four nominal-95% intervals for the slope cover the truth. At $\gamma=0$ all are fine. As $\gamma$ grows the naive OLS interval under-covers, the sandwich intervals recover, HC0 lagging HC3 in small samples — especially with a skewed design that creates high-leverage points — and WLS stays exact while also being narrower.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const gS = UI.slider(controls, { label: 'Heteroscedasticity $\\gamma$ (SD ∝ $x^\\gamma$)', min: 0, max: 2, step: 0.1, value: 1.2, onChange: reset });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 400, step: 5, value: 25, onChange: reset });
      const des = UI.segmented(controls, { label: 'Design', options: [{ value: 'u', label: 'Uniform x' }, { value: 'sk', label: 'Skewed x (high leverage)' }], value: 'sk', onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Simulate', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, xlabel: 'x', ylabel: 'y', title: 'One replication' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 4], y: [0.5, 1], ylabel: 'coverage of 95% interval', title: 'Coverage of the slope interval', xticks: 0 });
      const sN = UI.stat(stats, 'Replications');
      const sSE = UI.stat(stats, 'Mean SE ÷ true SD (OLS)');
      const sEff = UI.stat(stats, 'Var(WLS) ÷ Var(OLS)', { accent: 'accent' });
      const M = [['Naive OLS', 's5'], ['HC0', 's4'], ['HC3', 's1'], ['WLS', 's3']];
      let x, sd, hits, cnt, b1s, bws, seN, se3, last, tc;
      function reset() {
        loop.stop();
        const n = Math.round(nS.get());
        x = des.get() === 'u' ? ctx.rng.array(n, () => ctx.rng.uniform(1, 10)) : ctx.rng.array(n, () => 1 + Math.min(40, 1.5 * ctx.rng.lognormal(0, 0.9)));
        const mx = S.mean(x);
        sd = x.map(v => (v / mx) ** gS.get());
        tc = S.q.t(0.975, n - 2);
        hits = [0, 0, 0, 0]; cnt = 0; b1s = []; bws = []; seN = 0; se3 = 0;
        for (let i = 0; i < 100; i++) one();
        draw();
      }
      function one() {
        const n = x.length, y = x.map((v, i) => 1 + 0.5 * v + sd[i] * ctx.rng.normal());
        const f = slr(x, y), c = x.map(v => (v - f.mx) / f.sxx);
        const e = x.map((v, i) => y[i] - f.b0 - f.b1 * v), h = x.map(v => 1 / n + (v - f.mx) ** 2 / f.sxx);
        let v0 = 0, v3 = 0;
        for (let i = 0; i < n; i++) { v0 += c[i] * c[i] * e[i] * e[i]; v3 += c[i] * c[i] * e[i] * e[i] / (1 - h[i]) ** 2; }
        const vn = f.s2 / f.sxx;
        // WLS with w = 1/sd^2
        let sw = 0, swx = 0, swy = 0;
        for (let i = 0; i < n; i++) { const w = 1 / (sd[i] * sd[i]); sw += w; swx += w * x[i]; swy += w * y[i]; }
        const xw = swx / sw, yw = swy / sw; let sxxw = 0, sxyw = 0;
        for (let i = 0; i < n; i++) { const w = 1 / (sd[i] * sd[i]); sxxw += w * (x[i] - xw) ** 2; sxyw += w * (x[i] - xw) * (y[i] - yw); }
        const bw = sxyw / sxxw, aw = yw - bw * xw;
        let rw = 0; for (let i = 0; i < n; i++) rw += (y[i] - aw - bw * x[i]) ** 2 / (sd[i] * sd[i]);
        const vw = rw / (n - 2) / sxxw;
        const ses = [Math.sqrt(vn), Math.sqrt(v0), Math.sqrt(v3)];
        ses.forEach((se, k) => { if (Math.abs(f.b1 - 0.5) <= tc * se) hits[k]++; });
        if (Math.abs(bw - 0.5) <= tc * Math.sqrt(vw)) hits[3]++;
        cnt++; b1s.push(f.b1); bws.push(bw); seN += ses[0]; se3 += ses[2];
        last = { y, f, aw, bw };
      }
      function step() { const per = Math.max(2, Math.round(8000 / x.length)); for (let i = 0; i < per; i++) one(); draw(); if (cnt >= 4000) return false; }
      function draw() {
        const ex = S.extent(x), ey = S.extent(last.y), px = (ex[1] - ex[0]) * 0.05, py = (ey[1] - ey[0]) * 0.08;
        pL.setDomain([ex[0] - px, ex[1] + px], [ey[0] - py, ey[1] + py]).clear();
        pL.points(x.map((v, i) => [v, last.y[i]]), { color: 'ink', r: 2.8, opacity: 0.7 });
        const L = [ex[0] - px, ex[1] + px];
        pL.line(L.map(v => [v, last.f.b0 + last.f.b1 * v]), { color: 's1', width: 2.2 });
        pL.line(L.map(v => [v, last.aw + last.bw * v]), { color: 's3', width: 2.2, dash: true });
        pL.line(L.map(v => [v, 1 + 0.5 * v]), { color: 'muted', width: 1, dash: '2 4' });
        pL.legend([{ label: 'OLS', color: 's1' }, { label: 'WLS', color: 's3', dash: true }], { left: true });
        pR.clear();
        pR.hline(0.95, { color: 'ink', dash: '4 4', width: 1.2 });
        M.forEach(([name, col], k) => {
          const cv = hits[k] / cnt;
          pR.bars([{ x0: k + 0.15, x1: k + 0.85, y: Math.max(0.5, cv) }], { color: col, opacity: 0.75, base: 0.5 });
          pR.text(k + 0.5, Math.max(0.5, cv), pct(cv), { anchor: 'middle', dy: -6, size: 12, weight: 600 });
          pR.text(k + 0.5, 0.5, name, { anchor: 'middle', dy: 16, size: 11, color: 'muted' });
        });
        sN.set(String(cnt));
        const trueSd = S.sd(b1s);
        sSE.set(`${S.fmt(seN / cnt / trueSd, 2)} naive, ${S.fmt(se3 / cnt / trueSd, 2)} HC3`, 'below 1 means overconfident');
        sEff.set(S.fmt(S.variance(bws) / S.variance(b1s), 3), 'below 1: WLS is more efficient');
      }
      reset();
    },
  });

  /* ================================================================ */
  /* model-selection                                                  */
  /* ================================================================ */

  const TRUTHS = {
    sin: { label: 'Smooth wave (not a polynomial)', f: x => Math.sin(2 * Math.PI * x), deg: null },
    cubic: { label: 'Cubic polynomial', f: x => { const t = 2 * x - 1; return 2.5 * t ** 3 - t; }, deg: 3 },
    step: { label: 'Step (non-smooth)', f: x => (x > 0.5 ? 0.6 : -0.6), deg: null },
  };
  const DMAX = 10;

  // All nested polynomial fits of degree 0..DMAX in one QR. Returns per-degree criteria.
  function polyCriteria(x, y, mu, sigma) {
    const n = x.length;
    const B = x.map(v => legendre(2 * v - 1, DMAX));
    const cols = S.range(DMAX + 1).map(k => B.map(r => r[k]));
    const Q = mgs(cols);
    const yy = sq(y), mm = mu ? sq(mu) : 0;
    const out = [];
    const h = new Array(n).fill(0), fit = new Array(n).fill(0);
    let proj = 0, projMu = 0;
    for (let k = 0; k <= DMAX; k++) {
      const q = Q[k], c = dot(q, y);
      proj += c * c;
      for (let i = 0; i < n; i++) { h[i] += q[i] * q[i]; fit[i] += c * q[i]; }
      const p = k + 1, rss = Math.max(1e-300, yy - proj);
      let loo = 0; for (let i = 0; i < n; i++) loo += ((y[i] - fit[i]) / Math.max(1e-9, 1 - h[i])) ** 2;
      let risk = null;
      if (mu) { const cm = dot(q, mu); projMu += cm * cm; risk = sigma * sigma + Math.max(0, mm - projMu) / n + p * sigma * sigma / n; }
      out.push({ deg: k, p, rss, loo: loo / n, risk });
    }
    const s2 = out[DMAX].rss / Math.max(1, n - DMAX - 1);
    for (const o of out) {
      o.train = o.rss / n;
      o.cp = (o.rss + 2 * o.p * s2) / n;
      o.aic = (o.rss / n) * Math.exp(2 * o.p / n);
      o.bic = (o.rss / n) * Math.exp(o.p * Math.log(n) / n);
    }
    return out;
  }
  const argminBy = (arr, key) => arr.reduce((b, o) => (o[key] < b[key] ? o : b), arr[0]).deg;

  Atlas.lab('model-selection-curves', {
    title: 'Training error, Cp, AIC, BIC, LOOCV and the true risk',
    blurb: R`Polynomials of degree 0–10 are fitted to one sample. Right: every criterion on a common mean-squared-error scale (AIC and BIC shown as $e^{\text{IC}/n}$, a monotone transform), against the exact in-sample prediction risk, which is computable here because we know the truth. Training error falls forever; the honest criteria turn up. Shrink $n$ to see LOOCV and $C_p$ become erratic; switch to the cubic truth to see BIC lock onto degree 3.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tr = UI.select(controls, { label: 'True mean function', options: Object.keys(TRUTHS).map(k => ({ value: k, label: TRUTHS[k].label })), value: 'sin', onChange: gen });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 15, max: 200, step: 1, value: 30, onChange: gen });
      const sS = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.05, max: 1, step: 0.05, value: 0.3, onChange: gen });
      const dS = UI.slider(controls, { label: 'Show fit of degree', min: 0, max: DMAX, step: 1, value: 3, onInput: draw });
      UI.button(controls, 'New sample', gen, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [0, 1], xlabel: 'x', ylabel: 'y', title: 'Data, truth, and a fitted polynomial' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [-0.3, DMAX + 0.3], y: [0.01, 1], logy: true, xlabel: 'polynomial degree', ylabel: 'MSE scale (log)', title: 'Criteria vs degree' });
      const sCp = UI.stat(stats, '$C_p$ / AIC pick');
      const sBic = UI.stat(stats, 'BIC pick');
      const sCv = UI.stat(stats, 'LOOCV pick');
      const sOr = UI.stat(stats, 'Oracle (min true risk)', { accent: 'accent' });
      let x, y, crit;
      function gen() {
        const n = Math.round(nS.get()), T = TRUTHS[tr.get()];
        x = ctx.rng.array(n, () => ctx.rng.u()).sort((a, b) => a - b);
        const mu = x.map(T.f);
        y = mu.map(m => m + ctx.rng.normal(0, sS.get()));
        crit = polyCriteria(x, y, mu, sS.get());
        draw();
      }
      const SER = [['train', 'training MSE', 'muted'], ['cp', 'Cp / n', 's1'], ['aic', 'exp(AIC/n)', 's5'], ['bic', 'exp(BIC/n)', 's2'], ['loo', 'LOOCV', 's3'], ['risk', 'true risk', 'ink']];
      function draw() {
        const T = TRUTHS[tr.get()], d = Math.round(dS.get());
        const grid = S.linspace(0, 1, 200);
        const ey = S.extent(y), pad = (ey[1] - ey[0]) * 0.15;
        pL.setDomain([0, 1], [ey[0] - pad, ey[1] + pad]).clear();
        pL.line(grid.map(v => [v, T.f(v)]), { color: 'ink', width: 1.5, dash: '3 3' });
        try {
          const f = fitLS(x.map(v => legendre(2 * v - 1, d)), y);
          pL.line(grid.map(v => [v, dot(legendre(2 * v - 1, d), f.beta)]), { color: 's1', width: 2.5 });
        } catch (e) { /* singular: skip */ }
        pL.points(x.map((v, i) => [v, y[i]]), { color: 'ink', r: 2.6, opacity: 0.7 });
        pL.legend([{ label: 'truth', color: 'ink', dash: true }, { label: `degree ${d} fit`, color: 's1' }], { left: true });
        const vals = crit.flatMap(o => SER.map(s => o[s[0]])).filter(v => v > 0 && Number.isFinite(v));
        const riskMin = Math.min(...crit.map(o => o.risk));
        const lo = Math.max(Math.min(...vals) * 0.7, riskMin * 0.05), hi = Math.min(Math.max(...vals) * 1.3, riskMin * 60);
        pR.setDomain([-0.3, DMAX + 0.3], [lo, hi]).clear();
        pR.vline(d, { color: 'faint', width: 1 });
        for (const [key, , col] of SER) {
          pR.line(crit.map(o => [o.deg, S.clamp(o[key], lo, hi)]), { color: col, width: key === 'risk' ? 2.8 : 1.8, dash: key === 'aic' ? '5 3' : false });
          if (key !== 'train') { const m = argminBy(crit, key); pR.dot(m, S.clamp(crit[m][key], lo, hi), { color: col, r: 4.5 }); }
        }
        pR.legend(SER.map(([, label, col]) => ({ label, color: col, dash: label === 'exp(AIC/n)' })));
        sCp.set(`${argminBy(crit, 'cp')} / ${argminBy(crit, 'aic')}`);
        sBic.set(String(argminBy(crit, 'bic')));
        sCv.set(String(argminBy(crit, 'loo')));
        const o = argminBy(crit, 'risk');
        sOr.set(String(o), T.deg != null ? `true degree ${T.deg}` : 'truth is not a polynomial');
      }
      gen();
    },
  });

  Atlas.lab('model-selection-frequency', {
    title: 'Which degree does each criterion choose?',
    blurb: R`Repeat the experiment many times and tally the selected degree. With a cubic truth (in the candidate set), BIC's choices concentrate on degree 3 as $n$ grows — consistency — while AIC keeps picking too-large models a fixed fraction of the time. With a non-polynomial truth there is no correct degree; then compare the excess prediction risk of each choice relative to the best candidate, where AIC and LOOCV tend to do at least as well as BIC.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tr = UI.segmented(controls, { label: 'Truth', options: [{ value: 'cubic', label: 'Cubic (in set)' }, { value: 'sin', label: 'Wave (not in set)' }], value: 'cubic', onChange: reset });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 20, max: 2000, value: 100, log: true, fmt: v => String(Math.round(v)), onChange: reset });
      const sS = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.1, max: 1, step: 0.05, value: 0.5, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Run', '❚❚ Pause']);
      UI.button(row, 'Clear', reset);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [-0.5, DMAX + 0.5], y: [0, 1], xlabel: 'selected degree', ylabel: 'frequency', title: 'Selected degree' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [0, 3], y: [0, 0.5], ylabel: 'mean excess risk (relative)', title: 'Risk of chosen model vs oracle', xticks: 0 });
      const sN = UI.stat(stats, 'Replications');
      const sA = UI.stat(stats, 'AIC: P(correct)');
      const sB = UI.stat(stats, 'BIC: P(correct)', { accent: 'accent' });
      const sC = UI.stat(stats, 'LOOCV: P(correct)');
      const CR = [['aic', 'AIC', 's5'], ['bic', 'BIC', 's2'], ['loo', 'LOOCV', 's3']];
      let tally, excess, cnt;
      function reset() { loop.stop(); tally = CR.map(() => new Array(DMAX + 1).fill(0)); excess = CR.map(() => 0); cnt = 0; for (let i = 0; i < 20; i++) one(); draw(); }
      function one() {
        const n = Math.round(nS.get()), T = TRUTHS[tr.get()];
        const x = ctx.rng.array(n, () => ctx.rng.u()), mu = x.map(T.f), y = mu.map(m => m + ctx.rng.normal(0, sS.get()));
        const c = polyCriteria(x, y, mu, sS.get());
        const best = Math.min(...c.map(o => o.risk));
        CR.forEach(([key], k) => { const d = argminBy(c, key); tally[k][d]++; excess[k] += c[d].risk / best - 1; });
        cnt++;
      }
      function step() { const per = Math.max(1, Math.round(30000 / nS.get())); for (let i = 0; i < per; i++) one(); draw(); if (cnt >= 2000) return false; }
      function draw() {
        pL.clear();
        const w = 0.27;
        CR.forEach(([, , col], k) => pL.bars(tally[k].map((t, d) => ({ x0: d - 0.4 + k * w, x1: d - 0.4 + (k + 1) * w, y: t / cnt })), { color: col, opacity: 0.8, gap: 0.5 }));
        if (tr.get() === 'cubic') pL.vline(3, { color: 'ink', width: 1 });
        pL.legend(CR.map(([, label, col]) => ({ label, color: col, fill: true })));
        const ex = excess.map(e => e / cnt), top = Math.max(0.05, Math.max(...ex) * 1.25);
        pR.setDomain([0, 3], [0, top]).clear();
        CR.forEach(([, label, col], k) => {
          pR.bars([{ x0: k + 0.2, x1: k + 0.8, y: ex[k] }], { color: col, opacity: 0.8 });
          pR.text(k + 0.5, ex[k], pct(ex[k]), { anchor: 'middle', dy: -5, size: 11, weight: 600 });
          pR.text(k + 0.5, 0, label, { anchor: 'middle', dy: 15, size: 11, color: 'muted' });
        });
        sN.set(String(cnt));
        const corr = k => (tr.get() === 'cubic' ? pct(tally[k][3] / cnt) : 'n/a');
        sA.set(corr(0)); sB.set(corr(1)); sC.set(corr(2));
      }
      reset();
    },
  });

  /* ================================================================ */
  /* ridge-regression                                                 */
  /* ================================================================ */

  // Equicorrelated design via one shared factor; columns centred.
  function equiDesign(rng, n, p, rho) {
    const X = S.range(n).map(() => { const f = rng.normal(); return S.range(p).map(() => Math.sqrt(rho) * f + Math.sqrt(1 - rho) * rng.normal()); });
    for (let j = 0; j < p; j++) { const m = S.mean(X.map(r => r[j])); for (const r of X) r[j] -= m; }
    return X;
  }
  // (b - beta)' Sigma (b - beta) for Sigma = (1 - rho) I + rho 11'.
  const equiQuad = (d, rho) => (1 - rho) * sq(d) + rho * S.sum(d) ** 2;

  Atlas.lab('ridge-regression-path', {
    title: 'Ridge coefficient paths and the choice of λ',
    blurb: R`Eight equicorrelated predictors, true $\beta=(3,1.5,0,0,2,0,0,-1)$. Left: the ridge path $\hat\beta_\lambda$ against $\lambda$ (log scale) — all coefficients shrink smoothly toward zero, the correlated ones toward each other, and none is ever exactly zero. Right: GCV and the true expected prediction error. With high correlation and few observations, the best $\lambda$ beats least squares (the dashed level) by a wide margin; let $n < p$ and least squares does not even exist.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rS = UI.slider(controls, { label: 'Predictor correlation $\\rho$', min: 0, max: 0.99, step: 0.01, value: 0.9, onChange: gen });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 6, max: 200, step: 1, value: 25, onChange: gen });
      const sS = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.5, max: 6, step: 0.1, value: 3, onChange: gen });
      const lS = UI.slider(controls, { label: 'Marked $\\lambda$', min: 1e-3, max: 1e4, value: 5, log: true, fmt: v => S.fmt(v, 3), onInput: draw });
      UI.button(controls, 'New sample', gen, { primary: true });
      stage.classList.add('two');
      const lam = S.linspace(-3, 4, 90).map(t => 10 ** t);
      const pL = new Plot(stage, { width: 420, height: 270, x: [1e-3, 1e4], logx: true, xlabel: 'λ (log scale)', ylabel: 'coefficient', title: 'Ridge path' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [1e-3, 1e4], y: [0.1, 10], logx: true, logy: true, xlabel: 'λ (log scale)', ylabel: 'error (log)', title: 'Choosing λ' });
      const sDf = UI.stat(stats, '$\\text{df}(\\lambda)$ at marked λ');
      const sE = UI.stat(stats, 'Test error: marked λ vs OLS');
      const sG = UI.stat(stats, 'GCV choice', { accent: 'accent' });
      const sK = UI.stat(stats, 'Condition number $d_1^2/d_p^2$');
      const BETA = [3, 1.5, 0, 0, 2, 0, 0, -1], P = 8;
      let path, d2, gcv, test, ols;
      function gen() {
        const n = Math.round(nS.get()), rho = rS.get();
        const X = equiDesign(ctx.rng, n, P, rho);
        let y = X.map(r => dot(r, BETA) + ctx.rng.normal(0, sS.get()));
        const my = S.mean(y); y = y.map(v => v - my);
        const XtX = S.matmul(S.transpose(X), X), Xty = S.matvec(S.transpose(X), y);
        const { values, vectors: V } = S.eigSym(XtX);
        d2 = values.map(v => Math.max(v, 0));
        const z = S.range(P).map(j => S.range(P).reduce((s, i) => s + V[i][j] * Xty[i], 0)); // V'X'y
        const solve = L => { const c = z.map((zj, j) => zj / (d2[j] + L)); return S.range(P).map(i => S.range(P).reduce((s, j) => s + V[i][j] * c[j], 0)); };
        const evalAt = L => {
          const b = solve(L), rss = sq(X.map((r, i) => y[i] - dot(r, b))), df = S.sum(d2.map(v => v / (v + L)));
          return { b, df, gcv: (rss / n) / (1 - df / n) ** 2, test: sS.get() ** 2 + equiQuad(b.map((v, i) => v - BETA[i]), rho) };
        };
        path = lam.map(evalAt);
        gcv = path.map(o => (o.df < n - 0.5 ? o.gcv : NaN));
        test = path.map(o => o.test);
        ols = d2[P - 1] > 1e-8 * d2[0] && n > P ? evalAt(0) : null;
        draw.evalAt = evalAt;
        draw();
      }
      function draw() {
        const L = lS.get(), cur = draw.evalAt(L);
        const all = path.flatMap(o => o.b).concat(BETA);
        let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.08; lo = Math.max(lo - pad, -8); hi = Math.min(hi + pad, 10);
        pL.setDomain([1e-3, 1e4], [lo, hi]).clear();
        pL.hline(0, { color: 'faint', dash: false, width: 1 });
        for (let j = 0; j < P; j++) {
          const col = 's' + (j + 1);
          pL.line(path.map((o, k) => [lam[k], S.clamp(o.b[j], lo, hi)]), { color: col, width: BETA[j] ? 2.2 : 1.3, opacity: BETA[j] ? 1 : 0.7 });
          pL.dot(1.2e-3, S.clamp(BETA[j], lo, hi), { color: col, r: 3, opacity: BETA[j] ? 1 : 0.5 });
        }
        pL.vline(L, { color: 'ink', width: 1.2 });
        pL.text(1.4e-3, hi, 'dots: true β', { dy: 12, size: 10, color: 'muted' });
        const vals = test.concat(gcv.filter(Number.isFinite)).concat(ols ? [ols.test] : []);
        const ylo = Math.min(...vals) * 0.8, yhi = Math.min(Math.max(...vals) * 1.2, Math.min(...vals) * 50);
        pR.setDomain([1e-3, 1e4], [ylo, yhi]).clear();
        pR.line(lam.map((l, k) => [l, S.clamp(test[k], ylo, yhi)]), { color: 'ink', width: 2.5 });
        pR.line(lam.map((l, k) => [l, Number.isFinite(gcv[k]) ? S.clamp(gcv[k], ylo, yhi) : NaN]), { color: 's3', width: 2 });
        if (ols) pR.hline(S.clamp(ols.test, ylo, yhi), { color: 's1', width: 1.3 });
        pR.hline(sS.get() ** 2, { color: 'faint', width: 1 });
        const kt = S.argmin(test), kg = S.argmin(gcv.map(v => (Number.isFinite(v) ? v : Infinity)));
        pR.dot(lam[kt], test[kt], { color: 'ink', r: 4.5 });
        if (Number.isFinite(gcv[kg])) pR.dot(lam[kg], S.clamp(gcv[kg], ylo, yhi), { color: 's3', r: 4.5 });
        pR.vline(L, { color: 'ink', width: 1.2 });
        pR.legend([{ label: 'true prediction error', color: 'ink' }, { label: 'GCV', color: 's3' }].concat(ols ? [{ label: 'least squares', color: 's1', dash: true }] : []));
        sDf.set(S.fmt(cur.df, 2), `of ${Math.min(P, Math.round(nS.get()))} possible`);
        sE.set(`${S.fmt(cur.test, 2)} vs ${ols ? S.fmt(ols.test, 2) : 'undefined'}`, ols ? '' : 'p ≥ n: XᵀX is singular');
        sG.set(Number.isFinite(gcv[kg]) ? 'λ = ' + S.fmt(lam[kg], 3) : '—', Number.isFinite(gcv[kg]) ? `df ${S.fmt(path[kg].df, 2)}; true-best λ = ${S.fmt(lam[kt], 3)}` : '');
        sK.set(d2[P - 1] > 1e-10 ? S.fmt(d2[0] / d2[P - 1], 1) : '∞');
      }
      gen();
    },
  });

  Atlas.lab('ridge-regression-bias-variance', {
    title: 'Bias–variance along principal directions (exact)',
    blurb: R`The exact MSE of ridge for $p=10$ equicorrelated predictors, decomposed into squared bias and variance as functions of $\lambda$. The design has one high-variance direction ($\mathbf 1/\sqrt{p}$) and nine low-variance ones. When the signal lies along the high-variance direction, heavy shrinkage of the noisy directions is nearly free; when it lies along a low-variance direction, shrinking it costs bias and the gain over least squares is much smaller. The vertical line marks the Hoerl–Kennard bound $\sigma^2/\max_j\theta_j^2$, below which the MSE is guaranteed to decrease.`,
    mount(el, ctx) {
      const { controls, stats, stage } = UI.layout(el);
      const sig = UI.segmented(controls, { label: 'Signal direction', options: [{ value: 'top', label: 'High-variance PC' }, { value: 'spread', label: 'Spread evenly' }, { value: 'bottom', label: 'Low-variance PC' }], value: 'spread', onChange: draw });
      const rS = UI.slider(controls, { label: 'Predictor correlation $\\rho$', min: 0, max: 0.99, step: 0.01, value: 0.8, onInput: draw });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 12, max: 500, step: 1, value: 40, onInput: draw });
      const sS = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.2, max: 5, step: 0.1, value: 2, onInput: draw });
      const bS = UI.slider(controls, { label: 'Signal size $\\|\\beta\\|$', min: 0.1, max: 6, step: 0.1, value: 2, onInput: draw });
      const p = new Plot(stage, { width: 640, height: 320, x: [1e-3, 1e5], y: [0.01, 1], logx: true, logy: true, xlabel: 'λ (log scale)', ylabel: 'E‖β̂_λ − β‖² (log)' });
      const sO = UI.stat(stats, 'OLS MSE $\\sigma^2\\sum_j d_j^{-2}$');
      const sM = UI.stat(stats, 'Best ridge MSE', { accent: 'accent' });
      const sL = UI.stat(stats, 'Optimal $\\lambda^\\star$');
      const sH = UI.stat(stats, 'Hoerl–Kennard bound');
      const PP = 10, lam = S.linspace(-3, 5, 240).map(t => 10 ** t);
      function draw() {
        const rho = rS.get(), n = Math.round(nS.get()), s2 = sS.get() ** 2, B = bS.get();
        const d2 = [n * (1 + (PP - 1) * rho)].concat(new Array(PP - 1).fill(n * (1 - rho) + 1e-9));
        const th = sig.get() === 'top' ? [B].concat(new Array(PP - 1).fill(0)) : sig.get() === 'bottom' ? new Array(PP - 1).fill(0).concat([B]) : new Array(PP).fill(B / Math.sqrt(PP));
        const parts = lam.map(L => {
          let b2 = 0, v = 0;
          for (let j = 0; j < PP; j++) { b2 += (L * th[j] / (d2[j] + L)) ** 2; v += s2 * d2[j] / (d2[j] + L) ** 2; }
          return { b2, v, m: b2 + v };
        });
        const olsM = s2 * S.sum(d2.map(v => 1 / v));
        const k = S.argmin(parts.map(o => o.m));
        const hk = s2 / Math.max(...th.map(t => t * t));
        const ys = parts.flatMap(o => [o.b2, o.v, o.m]).filter(v => v > 0);
        const hi = Math.max(olsM, ...parts.map(o => o.m)) * 2, lo = Math.max(Math.min(...ys), hi * 1e-5);
        p.setDomain([1e-3, 1e5], [lo, hi]).clear();
        p.band(1e-3, Math.min(hk, 1e5), { color: 'good', opacity: 0.07 });
        p.line(lam.map((L, i) => [L, Math.max(lo, parts[i].b2)]), { color: 's2', width: 2 });
        p.line(lam.map((L, i) => [L, Math.max(lo, parts[i].v)]), { color: 's3', width: 2 });
        p.line(lam.map((L, i) => [L, parts[i].m]), { color: 's1', width: 3 });
        p.hline(olsM, { color: 'ink', width: 1.2 });
        p.vline(Math.min(hk, 1e5), { color: 'good', width: 1.5 });
        p.dot(lam[k], parts[k].m, { color: 's1', r: 5 });
        d2.slice(0, 2).forEach((v, i) => { p.vline(v, { color: 'faint', width: 1, dash: '1 3' }); p.text(v, lo, i ? 'd²(low PCs)' : 'd²(top PC)', { dx: 3, dy: -6, size: 10, color: 'muted' }); });
        p.legend([{ label: 'MSE', color: 's1', width: 3 }, { label: 'squared bias', color: 's2' }, { label: 'variance', color: 's3' }, { label: 'least squares', color: 'ink', dash: true }, { label: 'guaranteed-improvement zone', color: 'good', fill: true, opacity: 0.2 }]);
        sO.set(S.fmt(olsM, 3));
        sM.set(S.fmt(parts[k].m, 3), `${S.fmt(100 * (1 - parts[k].m / olsM), 1)}% below OLS`);
        sL.set(S.fmt(lam[k], 3), `df = ${S.fmt(S.sum(d2.map(v => v / (v + lam[k]))), 2)}`);
        sH.set(S.fmt(hk, 3), 'σ² / max θⱼ²');
      }
      draw();
    },
  });

  /* ================================================================ */
  /* lasso                                                            */
  /* ================================================================ */

  Atlas.lab('lasso-geometry', {
    title: 'Diamonds have corners',
    blurb: R`The RSS contours are ellipses centred at the least-squares estimate $\hat\beta$ (drag it with the sliders or click the plot). The constrained solution sits where the smallest contour touches the constraint set: the $\ell_1$ diamond (lasso) or the $\ell_2$ disc (ridge). The diamond's corners lie on the axes, so the lasso solution frequently has an exact zero; the disc's solution essentially never does. The last tile estimates how often each yields a sparse solution when $\hat\beta$ falls at random in the square.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const b1 = UI.slider(controls, { label: 'OLS $\\hat\\beta_1$', min: -3, max: 3, step: 0.05, value: 2.2, onInput: draw });
      const b2 = UI.slider(controls, { label: 'OLS $\\hat\\beta_2$', min: -3, max: 3, step: 0.05, value: 0.9, onInput: draw });
      const rS = UI.slider(controls, { label: 'Predictor correlation $\\rho$ (ellipse tilt)', min: -0.9, max: 0.9, step: 0.05, value: 0.5, onInput: draw, onChange: freq });
      const tS = UI.slider(controls, { label: 'Budget $t$', min: 0.2, max: 3, step: 0.05, value: 1.2, onInput: draw, onChange: freq });
      const rid = UI.toggle(controls, { label: 'Show ridge disc', value: true, onChange: draw });
      const p = new Plot(stage, { width: 640, height: 340, x: [-3.4, 3.4], y: [-3.4, 3.4], equal: true, xlabel: 'β₁', ylabel: 'β₂' });
      const sL = UI.stat(stats, 'Lasso solution', { accent: 'accent' });
      const sR = UI.stat(stats, 'Ridge-constraint solution');
      const sF = UI.stat(stats, 'Sparse solutions (random $\\hat\\beta$)');
      p.onPointer((x, y, e, type) => { if (type === 'down' || type === 'drag') { b1.set(S.clamp(x, -3, 3)); b2.set(S.clamp(y, -3, 3)); draw(); } });
      const Qf = (b, c, rho) => { const u = b[0] - c[0], v = b[1] - c[1]; return u * u + 2 * rho * u * v + v * v; };
      function lassoSol(c, rho, t) {
        if (Math.abs(c[0]) + Math.abs(c[1]) <= t) return c.slice();
        const V = [[t, 0], [0, t], [-t, 0], [0, -t]];
        let best = null, bq = Infinity;
        for (let k = 0; k < 4; k++) {
          const P = V[k], D = [V[(k + 1) % 4][0] - P[0], V[(k + 1) % 4][1] - P[1]];
          const u = [P[0] - c[0], P[1] - c[1]];
          const AD = [D[0] + rho * D[1], rho * D[0] + D[1]];
          const s = S.clamp(-(u[0] * AD[0] + u[1] * AD[1]) / (D[0] * AD[0] + D[1] * AD[1]), 0, 1);
          const b = [P[0] + s * D[0], P[1] + s * D[1]], q = Qf(b, c, rho);
          if (q < bq) { bq = q; best = b; }
        }
        return best;
      }
      function ridgeSol(c, rho, t) {
        if (Math.hypot(c[0], c[1]) <= t) return c.slice();
        const g = a => Qf([t * Math.cos(a), t * Math.sin(a)], c, rho);
        let ba = 0, bv = Infinity;
        for (let i = 0; i < 360; i++) { const a = i * Math.PI / 180, v = g(a); if (v < bv) { bv = v; ba = a; } }
        const a = S.golden(g, ba - Math.PI / 180, ba + Math.PI / 180, 1e-10);
        return [t * Math.cos(a), t * Math.sin(a)];
      }
      const isZero = v => Math.abs(v) < 1e-9;
      let freqTxt = '';
      function freq() {
        const r = S.rng(3), rho = rS.get(), t = tS.get();
        let nl = 0, nr = 0, out = 0;
        for (let i = 0; i < 1500; i++) {
          const c = [r.uniform(-3, 3), r.uniform(-3, 3)];
          if (Math.abs(c[0]) + Math.abs(c[1]) <= t) continue;
          out++;
          const l = lassoSol(c, rho, t); if (isZero(l[0]) || isZero(l[1])) nl++;
          const g = ridgeSol(c, rho, t); if (Math.abs(g[0]) < 1e-6 || Math.abs(g[1]) < 1e-6) nr++;
        }
        freqTxt = `lasso ${pct(nl / out)}, ridge ${pct(nr / out)}`;
        sF.set(freqTxt, 'among β̂ outside the constraint');
      }
      function draw() {
        const c = [b1.get(), b2.get()], rho = rS.get(), t = tS.get();
        const L = lassoSol(c, rho, t), G = ridgeSol(c, rho, t);
        const qL = Qf(L, c, rho), qG = Qf(G, c, rho);
        p.clear();
        p.hline(0, { color: 'faint', dash: false, width: 1 }); p.vline(0, { color: 'faint', dash: false, width: 1 });
        const f = (x, y) => Qf([x, y], c, rho);
        p.contour(f, [0.05, 0.3, 0.8, 1.6, 3, 5, 8].map(v => v * Math.max(0.3, qL)), { color: 'muted', width: 0.8, opacity: 0.6, nx: 90, ny: 70 });
        p.polygon([[t, 0], [0, t], [-t, 0], [0, -t]], { color: 's1', opacity: 0.15, stroke: 's1', width: 1.5 });
        if (rid.get()) p.line(S.linspace(0, 2 * Math.PI, 120).map(a => [t * Math.cos(a), t * Math.sin(a)]), { color: 's2', width: 1.5 });
        if (qL > 1e-12) p.contour(f, [qL], { color: 's1', width: 2, nx: 120, ny: 90 });
        if (rid.get() && qG > 1e-12) p.contour(f, [qG], { color: 's2', width: 1.5, dash: '4 3', nx: 120, ny: 90 });
        p.dot(c[0], c[1], { color: 'ink', r: 5 });
        p.text(c[0], c[1], 'β̂ (OLS)', { dx: 8, dy: -6, size: 11 });
        if (rid.get()) p.dot(G[0], G[1], { color: 's2', r: 5 });
        p.dot(L[0], L[1], { color: 's1', r: 6 });
        p.legend([{ label: 'lasso: |β₁|+|β₂| ≤ t', color: 's1' }].concat(rid.get() ? [{ label: 'ridge: β₁²+β₂² ≤ t²', color: 's2' }] : []), { left: true });
        const zl = isZero(L[0]) ? 'β₁ = 0 exactly' : isZero(L[1]) ? 'β₂ = 0 exactly' : 'both nonzero';
        sL.set(`(${S.fmt(L[0], 3)}, ${S.fmt(L[1], 3)})`, zl);
        sR.set(`(${S.fmt(G[0], 3)}, ${S.fmt(G[1], 3)})`, 'never exactly zero unless β̂ is on an axis');
        if (!freqTxt) freq();
      }
      draw();
    },
  });

  // Elastic-net coordinate descent (glmnet scaling) along a decreasing lambda grid with warm starts.
  // X: rows, columns standardized so (1/n) sum x^2 = 1; y centred. Returns array of coefficient vectors.
  function enetPath(X, y, lams, alpha) {
    const n = X.length, p = X[0].length;
    const cols = S.range(p).map(j => X.map(r => r[j]));
    const cn = cols.map(c => sq(c) / n);
    const b = new Array(p).fill(0), r = y.slice(), out = [];
    for (const L of lams) {
      for (let sweep = 0; sweep < 300; sweep++) {
        let maxd = 0;
        for (let j = 0; j < p; j++) {
          const c = cols[j], old = b[j];
          let z = 0; for (let i = 0; i < n; i++) z += c[i] * r[i];
          z = z / n + cn[j] * old;
          const nb = Math.sign(z) * Math.max(Math.abs(z) - L * alpha, 0) / (cn[j] + L * (1 - alpha));
          if (nb !== old) { const d = nb - old; for (let i = 0; i < n; i++) r[i] -= d * c[i]; b[j] = nb; maxd = Math.max(maxd, Math.abs(d)); }
        }
        if (maxd < 1e-7) break;
      }
      out.push(b.slice());
    }
    return out;
  }

  Atlas.lab('lasso-path', {
    title: 'Lasso and elastic-net paths with cross-validation',
    blurb: R`Ten predictors with a common correlation $\rho$; the first $s$ carry signal (coloured), the rest are noise (grey). Left: the coefficient path from coordinate descent as $\lambda$ decreases from $\lambda_{\max}$ (right) to near zero (left) — piecewise linear for the lasso, variables entering one by one. Right: 5-fold cross-validation error with one-standard-error band, and the true prediction error. Lower $\alpha$ toward ridge to watch correlated variables enter together.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const aS = UI.slider(controls, { label: 'Elastic-net mixing $\\alpha$ (1 = lasso)', min: 0.05, max: 1, step: 0.05, value: 1, onChange: gen });
      const kS = UI.slider(controls, { label: 'True nonzeros $s$', min: 1, max: 10, step: 1, value: 3, onChange: gen });
      const rS = UI.slider(controls, { label: 'Predictor correlation $\\rho$', min: 0, max: 0.95, step: 0.05, value: 0.5, onChange: gen });
      const sS = UI.slider(controls, { label: 'Noise SD $\\sigma$', min: 0.5, max: 4, step: 0.1, value: 1.5, onChange: gen });
      UI.button(controls, 'New sample', gen, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 270, x: [1e-3, 1], logx: true, xlabel: 'λ (log scale)', ylabel: 'coefficient', title: 'Coefficient path' });
      const pR = new Plot(stage, { width: 420, height: 270, x: [1e-3, 1], logx: true, xlabel: 'λ (log scale)', ylabel: 'mean squared error', title: '5-fold CV vs truth' });
      const sLam = UI.stat(stats, '$\\lambda_{\\min}$ (CV)');
      const sSel = UI.stat(stats, 'Selected at $\\lambda_{\\min}$', { accent: 'accent' });
      const sSel1 = UI.stat(stats, 'Selected at $\\lambda_{1\\text{se}}$');
      const sErr = UI.stat(stats, 'True error: CV choice vs smallest λ');
      const P = 10, N = 60, BV = [3, -2, 1.5, 1, -1, 0.8, -0.8, 0.6, -0.6, 0.5];
      let st;
      function gen() {
        const rho = rS.get(), s = Math.round(kS.get()), alpha = aS.get(), sig = sS.get();
        const beta = S.range(P).map(j => (j < s ? BV[j] : 0));
        const draw1 = m => S.range(m).map(() => { const f = ctx.rng.normal(); return S.range(P).map(() => Math.sqrt(rho) * f + Math.sqrt(1 - rho) * ctx.rng.normal()); });
        const Xr = draw1(N), yr = Xr.map(r => dot(r, beta) + ctx.rng.normal(0, sig));
        const mu = S.range(P).map(j => S.mean(Xr.map(r => r[j]))), sd = S.range(P).map(j => Math.sqrt(S.mean(Xr.map(r => (r[j] - mu[j]) ** 2))));
        const std = r => r.map((v, j) => (v - mu[j]) / sd[j]);
        const X = Xr.map(std), my = S.mean(yr), y = yr.map(v => v - my);
        const lmax = Math.max(...S.range(P).map(j => Math.abs(dot(X.map(r => r[j]), y)) / N)) / alpha;
        const lams = S.linspace(0, -3, 50).map(t => lmax * 10 ** t);
        const path = enetPath(X, y, lams, alpha);
        // 5-fold CV (standardization from the full sample, a simplification)
        const idx = ctx.rng.shuffle(S.range(N)), K = 5, errs = lams.map(() => []);
        for (let k = 0; k < K; k++) {
          const te = new Set(idx.filter((_, i) => i % K === k));
          const tr = S.range(N).filter(i => !te.has(i));
          const ytrm = S.mean(tr.map(i => y[i]));
          const pth = enetPath(tr.map(i => X[i]), tr.map(i => y[i] - ytrm), lams, alpha);
          pth.forEach((b, l) => errs[l].push(S.mean([...te].map(i => (y[i] - ytrm - dot(X[i], b)) ** 2))));
        }
        const cv = errs.map(e => S.mean(e)), cvse = errs.map(e => S.sd(e) / Math.sqrt(K));
        // true prediction error on a large test sample from the same population
        const Xt = draw1(600), mt = Xt.map(r => dot(r, beta)), Xts = Xt.map(std);
        const truth = path.map(b => sig * sig + S.mean(Xts.map((r, i) => (mt[i] - my - dot(r, b)) ** 2)));
        const kmin = S.argmin(cv);
        let k1 = kmin; for (let l = 0; l <= kmin; l++) if (cv[l] <= cv[kmin] + cvse[kmin]) { k1 = l; break; }
        st = { beta, lams, path, cv, cvse, truth, kmin, k1, sd, s };
        draw();
      }
      function draw() {
        const { beta, lams, path, cv, cvse, truth, kmin, k1, s } = st;
        const lo = lams[lams.length - 1], hi = lams[0];
        const all = path.flat(), ex = S.extent(all.concat([0])), pad = (ex[1] - ex[0]) * 0.08 + 0.05;
        pL.setDomain([lo, hi], [ex[0] - pad, ex[1] + pad]).clear();
        pL.hline(0, { color: 'faint', dash: false, width: 1 });
        for (let j = P - 1; j >= 0; j--) pL.line(path.map((b, l) => [lams[l], b[j]]), { color: j < s ? 's' + (1 + (j % 8)) : 'muted', width: j < s ? 2.2 : 1.2, opacity: j < s ? 1 : 0.7 });
        pL.vline(lams[kmin], { color: 's3', width: 1.3 });
        pL.vline(lams[k1], { color: 's3', width: 1, dash: '1 3' });
        pL.text(hi, ex[1] + pad, 'λmax →', { anchor: 'end', dy: 12, dx: -4, size: 10, color: 'muted' });
        const ys = cv.map((v, l) => v + cvse[l]).concat(truth);
        const ylo = Math.min(...cv.map((v, l) => v - cvse[l]), ...truth) * 0.9, yhi = Math.max(...ys) * 1.05;
        pR.setDomain([lo, hi], [Math.max(0, ylo), yhi]).clear();
        pR.area(lams.map((l, k) => [l, cv[k] + cvse[k]]), { base: lams.map((l, k) => [l, cv[k] - cvse[k]]), color: 's3', opacity: 0.15 });
        pR.line(lams.map((l, k) => [l, cv[k]]), { color: 's3', width: 2.2 });
        pR.line(lams.map((l, k) => [l, truth[k]]), { color: 'ink', width: 2 });
        pR.dot(lams[kmin], cv[kmin], { color: 's3', r: 4.5 });
        pR.vline(lams[k1], { color: 's3', width: 1, dash: '1 3' });
        pR.legend([{ label: 'CV ± 1 SE', color: 's3' }, { label: 'true prediction error', color: 'ink' }]);
        const nz = b => b.map((v, j) => [v, j]).filter(([v]) => Math.abs(v) > 1e-10);
        const desc = b => { const z = nz(b); const tp = z.filter(([, j]) => beta[j] !== 0).length; return [`${z.length} variables`, `${tp} true, ${z.length - tp} false positives`]; };
        sLam.set(S.fmt(lams[kmin], 3), `λmax = ${S.fmt(lams[0], 3)}`);
        const d0 = desc(path[kmin]), d1 = desc(path[k1]);
        sSel.set(d0[0], d0[1]);
        sSel1.set(d1[0], d1[1]);
        sErr.set(`${S.fmt(truth[kmin], 2)} vs ${S.fmt(truth[truth.length - 1], 2)}`, `noise floor σ² = ${S.fmt(sS.get() ** 2, 2)}`);
      }
      gen();
    },
  });

  /* ================================================================ */
  /* mixed-models                                                     */
  /* ================================================================ */

  // REML for the one-way random-intercept model from group summaries {n, ybar, W}.
  function remlOneWay(g) {
    const negll = th => {
      const s2 = Math.exp(th[0]), t2 = Math.exp(th[1]);
      let ldet = 0, sw = 0, swy = 0, W = 0;
      for (const q of g) { const v = s2 + q.n * t2; ldet += (q.n - 1) * Math.log(s2) + Math.log(v); sw += q.n / v; swy += q.n * q.ybar / v; W += q.W; }
      const mu = swy / sw;
      let quad = W / s2;
      for (const q of g) quad += q.n * (q.ybar - mu) ** 2 / (s2 + q.n * t2);
      return 0.5 * (ldet + Math.log(sw) + quad);
    };
    const allY = S.sum(g.map(q => q.n * q.ybar)) / S.sum(g.map(q => q.n));
    const W = S.sum(g.map(q => q.W)), dfw = S.sum(g.map(q => q.n - 1));
    const s20 = dfw > 0 ? Math.max(W / dfw, 1e-4) : 1, t20 = Math.max(S.variance(g.map(q => q.ybar)) - s20 / S.mean(g.map(q => q.n)), 0.05 * s20);
    const best = S.nelderMead(th => negll([th[0], Math.max(th[1], -20)]), [Math.log(s20), Math.log(t20)], { step: 0.6, maxIter: 600 });
    return { s2: Math.exp(best.x[0]), t2: Math.exp(Math.max(best.x[1], -20)), grand: allY };
  }

  Atlas.lab('mixed-models-pooling', {
    title: 'Partial pooling and BLUP shrinkage',
    blurb: R`Twelve groups of very different sizes (sorted, smallest on the left). Open circles are raw group means (no pooling), filled dots the BLUPs (partial pooling), crosses the true group means. Each arrow shows how far the random-effects model shrinks a group toward the overall mean: far for small groups, hardly at all for large ones. Variance components are estimated by REML. Compare the mean squared errors, then run many replications: partial pooling wins on average, most clearly when $\tau$ is small relative to $\sigma$.`,
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const tS = UI.slider(controls, { label: 'Between-group SD $\\tau$', min: 0.05, max: 3, step: 0.05, value: 0.8, onChange: gen });
      const sS = UI.slider(controls, { label: 'Within-group SD $\\sigma$', min: 0.2, max: 5, step: 0.1, value: 2, onChange: gen });
      const mS = UI.slider(controls, { label: 'Group-size multiplier', min: 0.3, max: 4, step: 0.1, value: 1, onChange: gen });
      const est = UI.toggle(controls, { label: 'Estimate $\\tau^2,\\sigma^2$ by REML (off: use the truth)', value: true, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', gen, { primary: true });
      UI.button(row, 'Run 300 replications', replicate);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 290, xlabel: 'group (sorted by size)', ylabel: 'group mean', title: 'No pooling → partial pooling', xticks: 0 });
      const pR = new Plot(stage, { width: 420, height: 290, logx: true, x: [0.8, 300], y: [0, 1], xlabel: 'group size nᵢ (log)', ylabel: 'weight on own mean', title: 'Shrinkage weight nτ² / (nτ² + σ²)' });
      const sNo = UI.stat(stats, 'MSE, no pooling');
      const sCo = UI.stat(stats, 'MSE, complete pooling');
      const sPa = UI.stat(stats, 'MSE, partial pooling (BLUP)', { accent: 'accent' });
      const sVc = UI.stat(stats, 'REML $\\hat\\tau,\\ \\hat\\sigma$');
      const sRep = UI.stat(stats, 'Average over replications');
      const BASE = [1, 2, 2, 3, 4, 5, 7, 10, 14, 20, 30, 50], MU = 10;
      let grp;
      function simulate(rng) {
        const sizes = BASE.map(b => Math.max(1, Math.round(b * mS.get())));
        return sizes.map(n => {
          const mu = MU + rng.normal(0, tS.get()), y = rng.array(n, () => mu + rng.normal(0, sS.get()));
          const ybar = S.mean(y);
          return { n, mu, y, ybar, W: y.reduce((s, v) => s + (v - ybar) ** 2, 0) };
        });
      }
      function analyse(g, useEst) {
        let s2 = sS.get() ** 2, t2 = tS.get() ** 2, vc = null;
        if (useEst) { vc = remlOneWay(g); s2 = vc.s2; t2 = vc.t2; }
        const w = g.map(q => 1 / (t2 + s2 / q.n)), muHat = S.sum(g.map((q, i) => w[i] * q.ybar)) / S.sum(w);
        const shrink = g.map(q => (q.n * t2) / (q.n * t2 + s2));
        const blup = g.map((q, i) => muHat + shrink[i] * (q.ybar - muHat));
        const pooled = S.sum(g.map(q => q.n * q.ybar)) / S.sum(g.map(q => q.n));
        const mse = f => S.mean(g.map((q, i) => (f(q, i) - q.mu) ** 2));
        return { s2, t2, muHat, shrink, blup, pooled, vc, no: mse(q => q.ybar), co: mse(() => pooled), pa: mse((q, i) => blup[i]) };
      }
      function gen() { grp = simulate(ctx.rng); sRep.set('—', 'press “Run 300 replications”'); draw(); }
      function draw() {
        const A = analyse(grp, est.get()), G = grp.length;
        const vals = grp.flatMap(q => [q.ybar, q.mu]), e = S.extent(vals), pad = (e[1] - e[0]) * 0.1 + 0.2;
        pL.setDomain([0.4, G + 0.6], [e[0] - pad, e[1] + pad]).clear();
        pL.hline(A.muHat, { color: 's2', width: 1.3 });
        grp.forEach((q, i) => {
          const x = i + 1;
          if (Math.abs(A.blup[i] - q.ybar) > 1e-3) pL.arrow(x, q.ybar, x, A.blup[i], { color: 's1', width: 1.3 });
          pL.dot(x, q.ybar, { color: 'surface', stroke: 'ink', r: 4.5, strokeWidth: 1.5 });
          pL.dot(x, A.blup[i], { color: 's1', r: 4.5 });
          const d = 0.12, yr = (e[1] - e[0] + 2 * pad) * 0.018;
          pL.segment(x - d, q.mu - yr, x + d, q.mu + yr, { color: 's3', width: 2 }); pL.segment(x - d, q.mu + yr, x + d, q.mu - yr, { color: 's3', width: 2 });
          pL.text(x, e[0] - pad, String(q.n), { anchor: 'middle', dy: 14, size: 10, color: 'muted' });
        });
        pL.legend([{ label: 'raw mean', color: 'ink' }, { label: 'BLUP', color: 's1' }, { label: 'true mean', color: 's3' }, { label: 'estimated μ', color: 's2', dash: true }]);
        const ng = S.linspace(Math.log10(0.8), Math.log10(300), 120).map(t => 10 ** t);
        pR.clear();
        pR.line(ng.map(n => [n, n * tS.get() ** 2 / (n * tS.get() ** 2 + sS.get() ** 2)]), { color: 'ink', width: 1.5, dash: '4 3' });
        if (est.get()) pR.line(ng.map(n => [n, n * A.t2 / (n * A.t2 + A.s2)]), { color: 's1', width: 2.2 });
        pR.points(grp.map((q, i) => [q.n, A.shrink[i]]), { color: 's1', r: 4 });
        pR.legend([{ label: 'with true τ², σ²', color: 'ink', dash: true }].concat(est.get() ? [{ label: 'with REML estimates', color: 's1' }] : []), { left: true, x: undefined });
        const best = Math.min(A.no, A.co, A.pa);
        const mark = v => (v === best ? good(S.fmt(v, 3)) : S.fmt(v, 3));
        sNo.set(mark(A.no)); sCo.set(mark(A.co)); sPa.set(mark(A.pa));
        sVc.set(A.vc ? `${S.fmt(Math.sqrt(A.vc.t2), 2)}, ${S.fmt(Math.sqrt(A.vc.s2), 2)}` : 'using truth', `true ${S.fmt(tS.get(), 2)}, ${S.fmt(sS.get(), 2)}`);
      }
      function replicate() {
        const r = ctx.newRng(); let no = 0, co = 0, pa = 0; const M = 300;
        for (let m = 0; m < M; m++) { const A = analyse(simulate(r), est.get()); no += A.no; co += A.co; pa += A.pa; }
        sRep.set(`${S.fmt(no / M, 3)} / ${S.fmt(co / M, 3)} / ${S.fmt(pa / M, 3)}`, 'no / complete / partial pooling');
      }
      gen();
    },
  });
})();
