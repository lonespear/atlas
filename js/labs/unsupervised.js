/* Labs: Unsupervised Learning (clustering, mixtures & EM, dimension reduction, graphical models). */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const CL = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'];

  /* ------------------------------------------------------------------ helpers */

  const sq2 = (a, b) => { const dx = a[0] - b[0], dy = a[1] - b[1]; return dx * dx + dy * dy; };
  const rgbStr = c => `rgb(${c[0]},${c[1]},${c[2]})`;
  const palette = () => CL.map(c => Atlas.rgb(c));
  const hide = (ctl, hidden) => { ctl.el.style.display = hidden ? 'none' : ''; };

  // Symmetric eigendecomposition: Householder tridiagonalisation + implicit QL (after JAMA's tred2/tql2).
  // Returns eigenvalues ascending and vec(j) = j-th eigenvector (unit norm).
  function eigh(A) {
    const n = A.length;
    const V = A.map(r => Float64Array.from(r));
    const d = new Float64Array(n), e = new Float64Array(n);
    for (let j = 0; j < n; j++) d[j] = V[n - 1][j];
    for (let i = n - 1; i > 0; i--) {
      let scale = 0, h = 0;
      for (let k = 0; k < i; k++) scale += Math.abs(d[k]);
      if (scale === 0) {
        e[i] = d[i - 1];
        for (let j = 0; j < i; j++) { d[j] = V[i - 1][j]; V[i][j] = 0; V[j][i] = 0; }
      } else {
        for (let k = 0; k < i; k++) { d[k] /= scale; h += d[k] * d[k]; }
        let f = d[i - 1], g = Math.sqrt(h);
        if (f > 0) g = -g;
        e[i] = scale * g; h -= f * g; d[i - 1] = f - g;
        for (let j = 0; j < i; j++) e[j] = 0;
        for (let j = 0; j < i; j++) {
          f = d[j]; V[j][i] = f; g = e[j] + V[j][j] * f;
          for (let k = j + 1; k <= i - 1; k++) { g += V[k][j] * d[k]; e[k] += V[k][j] * f; }
          e[j] = g;
        }
        f = 0;
        for (let j = 0; j < i; j++) { e[j] /= h; f += e[j] * d[j]; }
        const hh = f / (h + h);
        for (let j = 0; j < i; j++) e[j] -= hh * d[j];
        for (let j = 0; j < i; j++) {
          f = d[j]; g = e[j];
          for (let k = j; k <= i - 1; k++) V[k][j] -= f * e[k] + g * d[k];
          d[j] = V[i - 1][j]; V[i][j] = 0;
        }
      }
      d[i] = h;
    }
    for (let i = 0; i < n - 1; i++) {
      V[n - 1][i] = V[i][i]; V[i][i] = 1;
      const h = d[i + 1];
      if (h !== 0) {
        for (let k = 0; k <= i; k++) d[k] = V[k][i + 1] / h;
        for (let j = 0; j <= i; j++) {
          let g = 0;
          for (let k = 0; k <= i; k++) g += V[k][i + 1] * V[k][j];
          for (let k = 0; k <= i; k++) V[k][j] -= g * d[k];
        }
      }
      for (let k = 0; k <= i; k++) V[k][i + 1] = 0;
    }
    for (let j = 0; j < n; j++) { d[j] = V[n - 1][j]; V[n - 1][j] = 0; }
    V[n - 1][n - 1] = 1;
    e[0] = 0;
    for (let i = 1; i < n; i++) e[i - 1] = e[i];
    e[n - 1] = 0;
    let f = 0, tst1 = 0;
    const eps = 2 ** -52;
    for (let l = 0; l < n; l++) {
      tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
      let m = l;
      while (m < n - 1 && Math.abs(e[m]) > eps * tst1) m++;
      if (m > l) {
        let iter = 0;
        do {
          iter++;
          let g = d[l], p = (d[l + 1] - g) / (2 * e[l]), r = Math.hypot(p, 1);
          if (p < 0) r = -r;
          d[l] = e[l] / (p + r); d[l + 1] = e[l] * (p + r);
          const dl1 = d[l + 1];
          let h = g - d[l];
          for (let i = l + 2; i < n; i++) d[i] -= h;
          f += h;
          p = d[m];
          let c = 1, c2 = 1, c3 = 1, s = 0, s2 = 0;
          const el1 = e[l + 1];
          for (let i = m - 1; i >= l; i--) {
            c3 = c2; c2 = c; s2 = s;
            g = c * e[i]; h = c * p; r = Math.hypot(p, e[i]);
            e[i + 1] = s * r; s = e[i] / r; c = p / r;
            p = c * d[i] - s * g;
            d[i + 1] = h + s * (c * g + s * d[i]);
            for (let k = 0; k < n; k++) { const vk = V[k]; h = vk[i + 1]; vk[i + 1] = s * vk[i] + c * h; vk[i] = c * vk[i] - s * h; }
          }
          p = -s * s2 * c3 * el1 * e[l] / dl1;
          e[l] = s * p; d[l] = c * p;
        } while (Math.abs(e[l]) > eps * tst1 && iter < 80);
      }
      d[l] += f; e[l] = 0;
    }
    const idx = Array.from(d.keys()).sort((a, b) => d[a] - d[b]);
    return { values: idx.map(i => d[i]), vec: j => V.map(r => r[idx[j]]) };
  }

  // Adjusted Rand index between two label vectors (any integer labels; -1 treated as its own label).
  function ari(a, b) {
    const n = a.length, key = new Map(), ra = new Map(), rb = new Map();
    for (let i = 0; i < n; i++) {
      const k = a[i] + ',' + b[i];
      key.set(k, (key.get(k) || 0) + 1); ra.set(a[i], (ra.get(a[i]) || 0) + 1); rb.set(b[i], (rb.get(b[i]) || 0) + 1);
    }
    const c2 = x => x * (x - 1) / 2;
    let sij = 0, sa = 0, sb = 0;
    for (const v of key.values()) sij += c2(v);
    for (const v of ra.values()) sa += c2(v);
    for (const v of rb.values()) sb += c2(v);
    const exp = sa * sb / c2(n), mx = (sa + sb) / 2;
    return mx === exp ? 1 : (sij - exp) / (mx - exp);
  }

  /* ---------- 2-D datasets: each returns { X: [[x,y]], y: labels } ---------- */
  function gauss2(r, n, mu, sx, sy, rot, label, X, Y) {
    const c = Math.cos(rot), s = Math.sin(rot);
    for (let i = 0; i < n; i++) {
      const a = r.normal() * sx, b = r.normal() * sy;
      X.push([mu[0] + c * a - s * b, mu[1] + s * a + c * b]); Y.push(label);
    }
  }
  const DATA2 = {
    blobs4: { label: 'Four round blobs', gen(r) { const X = [], Y = []; [[-1.8, 1.3], [1.8, 1.3], [-1.8, -1.3], [1.8, -1.3]].forEach((m, k) => gauss2(r, 70, m, 0.55, 0.55, 0, k, X, Y)); return { X, Y }; } },
    unequal: { label: 'Unequal sizes', gen(r) { const X = [], Y = []; gauss2(r, 270, [-1, 0], 1, 1, 0, 0, X, Y); gauss2(r, 30, [2, 0], 0.32, 0.32, 0, 1, X, Y); return { X, Y }; } },
    aniso: { label: 'Elongated bars', gen(r) { const X = [], Y = []; [-1.3, 0, 1.3].forEach((m, k) => gauss2(r, 90, [0, m], 1.5, 0.17, 0, k, X, Y)); return { X, Y }; } },
    ellipses: { label: 'Rotated ellipses', gen(r) { const X = [], Y = []; gauss2(r, 120, [-1.6, 0.8], 1.1, 0.3, 0.6, 0, X, Y); gauss2(r, 100, [1.4, 1.1], 0.6, 0.45, -0.4, 1, X, Y); gauss2(r, 90, [0.4, -1.5], 1.3, 0.25, -0.15, 2, X, Y); return { X, Y }; } },
    overlap: { label: 'Overlapping', gen(r) { const X = [], Y = []; gauss2(r, 180, [-0.5, 0], 1.3, 0.6, 0.3, 0, X, Y); gauss2(r, 120, [0.9, 0.4], 0.45, 0.45, 0, 1, X, Y); return { X, Y }; } },
    moons: { label: 'Two moons', gen(r, noise = 0.12) {
      const X = [], Y = [];
      for (let i = 0; i < 260; i++) {
        const k = i % 2, t = Math.PI * r.u();
        const x = k ? 1 - Math.cos(t) : Math.cos(t), y = k ? 0.5 - Math.sin(t) : Math.sin(t);
        X.push([1.6 * (x - 0.5) + r.normal(0, noise), 1.6 * (y - 0.25) + r.normal(0, noise)]); Y.push(k);
      }
      return { X, Y };
    } },
    rings: { label: 'Concentric rings', gen(r, noise = 0.12) {
      const X = [], Y = [];
      for (let i = 0; i < 260; i++) {
        const k = i < 100 ? 0 : 1, t = 2 * Math.PI * r.u(), R = k ? 2.3 : 0.9;
        X.push([R * Math.cos(t) + r.normal(0, noise), R * Math.sin(t) + r.normal(0, noise)]); Y.push(k);
      }
      return { X, Y };
    } },
  };
  const dataOpts = keys => keys.map(k => ({ value: k, label: DATA2[k].label }));

  /* ---------- k-means helpers ---------- */
  function assign(X, C, lab) {
    let J = 0, changed = 0;
    for (let i = 0; i < X.length; i++) {
      let best = 0, bd = Infinity;
      for (let k = 0; k < C.length; k++) { const d = sq2(X[i], C[k]); if (d < bd - 1e-12) { bd = d; best = k; } }
      if (lab[i] !== best) changed++;
      lab[i] = best; J += bd;
    }
    return { J, changed };
  }
  function updateMeans(X, lab, C) {
    const K = C.length, s = C.map(() => [0, 0, 0]);
    for (let i = 0; i < X.length; i++) { const a = s[lab[i]]; a[0] += X[i][0]; a[1] += X[i][1]; a[2]++; }
    return s.map((a, k) => (a[2] ? [a[0] / a[2], a[1] / a[2]] : C[k].slice()));
  }
  const objective = (X, lab, C) => X.reduce((t, x, i) => t + sq2(x, C[lab[i]]), 0);
  function seedRandom(r, X, K) { return r.sample(X.map((_, i) => i), K, false).map(i => X[i].slice()); }
  function seedPP(r, X, K) {
    const C = [X[Math.floor(r.u() * X.length)].slice()];
    const D = X.map(x => sq2(x, C[0]));
    while (C.length < K) {
      const tot = S.sum(D);
      let u = r.u() * tot, j = 0;
      while (j < X.length - 1 && (u -= D[j]) > 0) j++;
      C.push(X[j].slice());
      for (let i = 0; i < X.length; i++) D[i] = Math.min(D[i], sq2(X[i], X[j]));
    }
    return C;
  }
  function lloyd(X, C, maxIt = 200) {
    const lab = new Int32Array(X.length).fill(-1);
    let it = 0;
    for (; it < maxIt; it++) {
      const { changed } = assign(X, C, lab);
      if (!changed) break;
      C = updateMeans(X, lab, C);
    }
    return { C, lab, J: objective(X, lab, C), it };
  }
  // Voronoi raster for centres C.
  function voronoi(p, C, alpha = 0.16) {
    const bg = Atlas.rgb('surface'), pal = palette().map(c => Atlas.mix(bg, c, alpha));
    p.raster((x, y) => {
      let best = 0, bd = Infinity;
      for (let k = 0; k < C.length; k++) { const d = (x - C[k][0]) ** 2 + (y - C[k][1]) ** 2; if (d < bd) { bd = d; best = k; } }
      return pal[best % 8];
    }, { nx: 130, ny: 95 });
  }
  function drawClusters(p, X, lab, opts = {}) {
    const K = Math.max(0, ...lab) + 1;
    for (let k = 0; k < K; k++) p.points(X.filter((_, i) => lab[i] === k), { color: CL[k % 8], r: opts.r || 2.6, opacity: opts.opacity || 0.85 });
    const noise = X.filter((_, i) => lab[i] < 0);
    if (noise.length) p.points(noise, { color: 'faint', r: opts.r || 2.6, opacity: 0.8 });
  }
  function drawCentres(p, C) { C.forEach((c, k) => p.dot(c[0], c[1], { color: CL[k % 8], r: 7, stroke: 'ink', strokeWidth: 2 })); }

  /* ---------- 2-D Gaussian mixture helpers ---------- */
  function logN2(x, m, c) {
    const a = c[0][0], b = c[0][1], d = c[1][1], det = a * d - b * b;
    const dx = x[0] - m[0], dy = x[1] - m[1];
    return -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * (d * dx * dx - 2 * b * dx * dy + a * dy * dy) / det;
  }
  function gmmE(X, g) {
    const K = g.pi.length, R = [];
    let ll = 0;
    const lp = new Array(K);
    for (const x of X) {
      let m = -Infinity;
      for (let k = 0; k < K; k++) { lp[k] = Math.log(g.pi[k]) + logN2(x, g.mu[k], g.cov[k]); if (lp[k] > m) m = lp[k]; }
      let s = 0;
      for (let k = 0; k < K; k++) s += Math.exp(lp[k] - m);
      const lse = m + Math.log(s);
      ll += lse;
      R.push(lp.map(v => Math.exp(v - lse)));
    }
    return { R, ll };
  }
  function gmmM(X, R, type, floor) {
    const K = R[0].length, n = X.length;
    const Nk = new Array(K).fill(0), mu = [], cov = [];
    for (let k = 0; k < K; k++) {
      let s0 = 0, s1 = 0;
      for (let i = 0; i < n; i++) { Nk[k] += R[i][k]; s0 += R[i][k] * X[i][0]; s1 += R[i][k] * X[i][1]; }
      const N = Math.max(Nk[k], 1e-12);
      mu.push([s0 / N, s1 / N]);
      let a = 0, b = 0, d = 0;
      for (let i = 0; i < n; i++) { const dx = X[i][0] - mu[k][0], dy = X[i][1] - mu[k][1], r = R[i][k]; a += r * dx * dx; b += r * dx * dy; d += r * dy * dy; }
      cov.push([[a / N, b / N], [b / N, d / N]]);
    }
    if (type === 'tied') {
      const T = [[0, 0], [0, 0]];
      for (let k = 0; k < K; k++) for (let u = 0; u < 2; u++) for (let v = 0; v < 2; v++) T[u][v] += Nk[k] * cov[k][u][v] / n;
      for (let k = 0; k < K; k++) cov[k] = T.map(r => r.slice());
    } else if (type === 'diag') {
      for (const c of cov) c[0][1] = c[1][0] = 0;
    } else if (type === 'sph') {
      for (const c of cov) { const v = (c[0][0] + c[1][1]) / 2; c[0][0] = c[1][1] = v; c[0][1] = c[1][0] = 0; }
    }
    const fl = Math.max(floor, 1e-12);
    for (const c of cov) { c[0][0] += fl; c[1][1] += fl; }
    return { pi: Nk.map(v => Math.max(v, 1e-12) / n), mu, cov };
  }
  const nParams2 = (K, type) => (K - 1) + 2 * K + ({ full: 3 * K, diag: 2 * K, sph: K, tied: 3 }[type]);
  function gmmInit(r, X, K) {
    const mu = seedPP(r, X, K), vx = S.variance(X.map(x => x[0])), vy = S.variance(X.map(x => x[1]));
    const s = (vx + vy) / 2 / Math.max(1, K);
    return { pi: new Array(K).fill(1 / K), mu, cov: mu.map(() => [[s, 0], [0, s]]) };
  }
  function gmmFit(r, X, K, { type = 'full', floor = 1e-6, iters = 60, restarts = 3 } = {}) {
    let best = null;
    for (let rep = 0; rep < restarts; rep++) {
      let g = gmmInit(r, X, K), prev = -Infinity, ll = -Infinity;
      for (let t = 0; t < iters; t++) {
        const E = gmmE(X, g); ll = E.ll;
        if (ll - prev < 1e-6 * Math.abs(ll)) break;
        prev = ll;
        g = gmmM(X, E.R, type, floor);
      }
      ll = gmmE(X, g).ll;
      if (!best || ll > best.ll) best = { g, ll };
    }
    return best;
  }
  const minEig2 = c => { const a = c[0][0], b = c[0][1], d = c[1][1]; return (a + d) / 2 - Math.sqrt(((a - d) / 2) ** 2 + b * b); };

  /* ================================================================== k-means */

  Atlas.lab('kmeans-lloyd', {
    title: "Lloyd's algorithm, one half-step at a time",
    blurb: 'Each Step is half an iteration: <em>assign</em> points to the nearest centre (the shaded Voronoi cells), then <em>update</em> each centre to its cluster mean. The objective $J$ on the right can only go down. Try “Unequal sizes”, “Elongated bars”, “Two moons” or “Concentric rings” to see where the objective itself is the problem, and “New start” to land in different local optima.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: dataOpts(['blobs4', 'unequal', 'aniso', 'moons', 'rings']), value: 'blobs4', onChange: newData });
      const Ks = UI.slider(controls, { label: 'Clusters $K$', min: 2, max: 8, step: 1, value: 4, onChange: reinit });
      const init = UI.segmented(controls, { label: 'Initialisation', options: [{ value: 'rand', label: 'Random points' }, { value: 'pp', label: 'k-means++' }], value: 'rand', onChange: reinit });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Step', () => { step(); draw(); }, { primary: true });
      let acc = 0;
      const loop = ctx.loop(dt => { acc += dt; if (acc > 0.3) { acc = 0; const more = step(); draw(); if (!more) return false; } });
      ctx.playButton(row, loop);
      const row2 = UI.h('div', 'btn-row', controls);
      UI.button(row2, 'New start', reinit);
      UI.button(row2, 'New data', newData);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-4, 4], y: [-3, 3], equal: true, title: 'Data, centres and Voronoi cells' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0, 10], y: [0, 1], xlabel: 'half-step', ylabel: 'objective J', title: 'Within-cluster sum of squares' });
      const sIt = UI.stat(stats, 'Lloyd iterations');
      const sJ = UI.stat(stats, 'Objective $J$', { accent: 'accent' });
      const sCh = UI.stat(stats, 'Points reassigned (last assign)');
      const sStatus = UI.stat(stats, 'Status');

      let X, Y, C, lab, phase, hist, it, done, lastChanged, trail;
      function newData() { ({ X, Y } = DATA2[ds.get()].gen(ctx.rng)); reinit(); }
      function reinit() {
        loop.stop();
        const K = Math.round(Ks.get());
        C = init.get() === 'pp' ? seedPP(ctx.rng, X, K) : seedRandom(ctx.rng, X, K);
        lab = new Int32Array(X.length).fill(-1);
        phase = 'assign'; hist = []; it = 0; done = false; lastChanged = null;
        trail = C.map(c => [c.slice()]);
        draw();
      }
      function step() {
        if (done) return false;
        if (phase === 'assign') {
          const { J, changed } = assign(X, C, lab);
          lastChanged = changed;
          hist.push({ J, kind: 'assign' });
          if (changed === 0) { done = true; return false; }
          phase = 'update';
        } else {
          C = updateMeans(X, lab, C);
          C.forEach((c, k) => trail[k].push(c.slice()));
          hist.push({ J: objective(X, lab, C), kind: 'update' });
          it++;
          phase = 'assign';
        }
        return true;
      }
      function draw() {
        pL.clear();
        voronoi(pL, C);
        if (lab[0] < 0) pL.points(X, { color: 'muted', r: 2.6, opacity: 0.7 });
        else drawClusters(pL, X, Array.from(lab));
        trail.forEach((t, k) => { if (t.length > 1) pL.line(t, { color: 'ink', width: 1.2, opacity: 0.55, dash: '3 3' }); });
        drawCentres(pL, C);
        const Js = hist.map(h => h.J);
        const ymax = Js.length ? Math.max(...Js) * 1.05 : 1, ymin = Js.length ? Math.min(...Js) * 0.95 : 0;
        pR.setDomain([0, Math.max(6, hist.length + 1)], [Math.max(0, ymin - (ymax - ymin) * 0.1), ymax + 1e-9]).clear();
        if (hist.length) {
          pR.line(hist.map((h, i) => [i + 1, h.J]), { color: 'muted', width: 1.5 });
          pR.points(hist.filter(h => h.kind === 'assign').map(h => [hist.indexOf(h) + 1, h.J]), { color: 's1', r: 4 });
          pR.points(hist.filter(h => h.kind === 'update').map(h => [hist.indexOf(h) + 1, h.J]), { color: 's2', r: 4 });
        }
        pR.legend([{ label: 'after assign', color: 's1' }, { label: 'after update', color: 's2' }]);
        sIt.set(String(it));
        sJ.set(hist.length ? S.fmt(hist[hist.length - 1].J, 4) : '—');
        sCh.set(lastChanged == null ? '—' : String(lastChanged));
        sStatus.set(done ? '<span style="color:var(--good)">converged</span>' : (phase === 'assign' ? 'next: assign' : 'next: update'), done ? 'a fixed point — not necessarily the global optimum' : '');
      }
      newData();
    },
  });

  Atlas.lab('kmeans-init', {
    title: 'Random seeding versus $k$-means++',
    blurb: 'Many restarts of Lloyd’s algorithm on $K$ well-separated blobs, from uniformly random seeds and from $D^2$ ($k$-means++) seeds. The histogram shows final $W$ relative to the best solution ever found. Random seeding frequently puts two centres in one blob and none in another, a trap Lloyd cannot escape; increase $K$ to watch it get worse.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const Ks = UI.slider(controls, { label: 'Number of blobs $=K$', min: 3, max: 12, step: 1, value: 9, onChange: newData });
      const sep = UI.slider(controls, { label: 'Separation (blob SDs)', min: 3, max: 12, step: 0.5, value: 7, onChange: newData });
      const Rn = UI.slider(controls, { label: 'Restarts per method', min: 50, max: 500, step: 50, value: 200, onChange: resetRuns });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(stepRuns);
      ctx.playButton(row, loop, ['▶ Run restarts', '❚❚ Pause']);
      UI.button(row, 'New data', newData);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-1, 1], y: [-1, 1], equal: true, title: 'Latest random-seed solution' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [1, 3], y: [0, 1], xlabel: 'final W / best W', ylabel: 'fraction of runs', title: 'Final objective over restarts' });
      const sN = UI.stat(stats, 'Restarts done');
      const sOptR = UI.stat(stats, 'Reach best: random');
      const sOptP = UI.stat(stats, 'Reach best: $k$-means++', { accent: 'accent' });
      const sSeed = UI.stat(stats, '$\\E W_{\\text{seed}}/W_{\\text{best}}$ (++)');

      let X, runs, lastRand;
      function newData() {
        const K = Math.round(Ks.get()), g = Math.ceil(Math.sqrt(K)), sp = sep.get() * 0.2;
        X = [];
        for (let k = 0; k < K; k++) {
          const cx = (k % g) * sp, cy = Math.floor(k / g) * sp;
          for (let i = 0; i < 30; i++) X.push([cx + ctx.rng.normal(0, 0.2), cy + ctx.rng.normal(0, 0.2)]);
        }
        resetRuns();
      }
      function resetRuns() {
        loop.stop();
        runs = { rand: [], pp: [], seedPP: [], seedR: [] }; lastRand = null;
        for (let i = 0; i < 15; i++) oneEach();
        draw();
      }
      function oneEach() {
        const K = Math.round(Ks.get());
        const cr = seedRandom(ctx.rng, X, K), cp = seedPP(ctx.rng, X, K);
        runs.seedR.push(objective(X, assignLab(cr), cr));
        runs.seedPP.push(objective(X, assignLab(cp), cp));
        const a = lloyd(X, cr), b = lloyd(X, cp);
        runs.rand.push(a.J); runs.pp.push(b.J); lastRand = a;
      }
      function assignLab(C) { const lab = new Int32Array(X.length).fill(-1); assign(X, C, lab); return lab; }
      function stepRuns() {
        const R = Math.round(Rn.get());
        for (let i = 0; i < 6 && runs.rand.length < R; i++) oneEach();
        draw();
        if (runs.rand.length >= R) return false;
      }
      function draw() {
        const best = Math.min(...runs.rand, ...runs.pp);
        const ext = S.extent(X.flat());
        pL.setDomain([ext[0] - 0.4, ext[1] + 0.4], [ext[0] - 0.4, ext[1] + 0.4]).clear();
        if (lastRand) { voronoi(pL, lastRand.C); drawClusters(pL, X, Array.from(lastRand.lab), { r: 2.2 }); drawCentres(pL, lastRand.C); }
        const rr = runs.rand.map(v => v / best), rp = runs.pp.map(v => v / best);
        const hi = Math.max(1.2, S.quantile(rr.concat(rp).sort((a, b) => a - b), 0.98, true) * 1.05);
        const bins = 30, h1 = S.histogram(rr, { bins, lo: 1 - (hi - 1) / bins, hi }), h2 = S.histogram(rp, { bins, lo: 1 - (hi - 1) / bins, hi });
        for (const b of h1) b.y = b.count / rr.length;
        for (const b of h2) b.y = b.count / rp.length;
        const ym = Math.max(0.05, ...h1.map(b => b.y), ...h2.map(b => b.y)) * 1.1;
        pR.setDomain([1 - (hi - 1) / bins, hi], [0, ym]).clear();
        pR.bars(h1, { color: 's4', opacity: 0.5 });
        pR.bars(h2, { color: 's3', opacity: 0.5 });
        pR.legend([{ label: 'random seeds', color: 's4', fill: true }, { label: 'k-means++ seeds', color: 's3', fill: true }]);
        const K = Math.round(Ks.get());
        const frac = a => a.filter(v => v < 1.01).length / a.length;
        sN.set(String(runs.rand.length));
        sOptR.set((100 * frac(rr)).toFixed(0) + '%', `mean ratio ${S.fmt(S.mean(rr), 3)}`);
        sOptP.set((100 * frac(rp)).toFixed(0) + '%', `mean ratio ${S.fmt(S.mean(rp), 3)}`);
        sSeed.set(S.fmt(S.mean(runs.seedPP) / best, 3), `bound 8(ln K + 2) = ${S.fmt(8 * (Math.log(K) + 2), 3)}; random seeds: ${S.fmt(S.mean(runs.seedR) / best, 3)}`);
      }
      newData();
    },
  });

  /* ================================================================== Gaussian mixtures */

  Atlas.lab('gaussian-mixtures-em', {
    title: 'EM for a Gaussian mixture, step by step',
    blurb: 'Points are coloured by their responsibilities (a blend of component colours); ellipses are the 1- and 2-SD contours. The log-likelihood never decreases (exactly so when $\\epsilon = 0$). After EM has converged, press <em>Add a spike component</em>: an extra component is parked on one data point with a tiny covariance. Keep running EM. With a small floor $\\epsilon$ the spike stays put and the likelihood jumps by an amount that grows like $\\log(1/\\epsilon)$ as you lower the floor — the likelihood is unbounded and only the floor limits it; with a large floor the spike is absorbed or wanders off.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: dataOpts(['ellipses', 'unequal', 'overlap', 'aniso', 'blobs4']), value: 'ellipses', onChange: newData });
      const Ks = UI.slider(controls, { label: 'Components $K$', min: 1, max: 6, step: 1, value: 3, onChange: reinit });
      const type = UI.segmented(controls, { label: 'Covariance', options: [{ value: 'full', label: 'Full' }, { value: 'diag', label: 'Diagonal' }, { value: 'sph', label: 'Spherical' }, { value: 'tied', label: 'Tied' }], value: 'full', onChange: reinit });
      const floor = UI.slider(controls, { label: 'Covariance floor $\\epsilon$ ($\\Sigma_k + \\epsilon I$)', min: 1e-9, max: 0.1, log: true, value: 1e-8, fmt: v => v.toExponential(0) });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'EM step', () => { step(); draw(); }, { primary: true });
      let acc = 0;
      const loop = ctx.loop(dt => { acc += dt; if (acc > 0.12) { acc = 0; step(); draw(); if (hist.length > 400) return false; } });
      ctx.playButton(row, loop);
      const row2 = UI.h('div', 'btn-row', controls);
      UI.button(row2, 'Restart', reinit);
      UI.button(row2, 'New data', newData);
      UI.button(row2, 'Add a spike component', collapse);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-4, 4], y: [-3, 3], equal: true, title: 'Responsibilities and component ellipses' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0, 10], y: [0, 1], xlabel: 'EM iteration', ylabel: 'log-likelihood', title: 'Observed-data log-likelihood' });
      const sIt = UI.stat(stats, 'Iteration');
      const sLL = UI.stat(stats, 'Log-likelihood $\\ell(\\theta)$', { accent: 'accent' });
      const sBIC = UI.stat(stats, 'BIC $= -2\\ell + d\\log n$');
      const sMin = UI.stat(stats, 'Smallest covariance eigenvalue');

      let X, g, R, hist;
      function newData() { ({ X } = DATA2[ds.get()].gen(ctx.rng)); reinit(); }
      function reinit() {
        g = gmmInit(ctx.rng, X, Math.round(Ks.get()));
        const E = gmmE(X, g); R = E.R; hist = [E.ll];
        draw();
      }
      function step() {
        g = gmmM(X, R, type.get(), floor.get());
        const E = gmmE(X, g); R = E.R; hist.push(E.ll);
      }
      function collapse() {
        if (g.pi.length >= 8) return;
        const i = Math.floor(ctx.rng.u() * X.length);
        g.mu.push(X[i].slice()); g.cov.push([[1e-4, 0], [0, 1e-4]]); g.pi.push(1 / X.length);
        const s = S.sum(g.pi); g.pi = g.pi.map(v => v / s);
        const E = gmmE(X, g); R = E.R; hist.push(E.ll);
        draw();
      }
      function draw() {
        const pal = palette(), K = g.pi.length;
        pL.clear();
        for (let i = 0; i < X.length; i++) {
          const c = [0, 0, 0];
          for (let k = 0; k < K; k++) for (let j = 0; j < 3; j++) c[j] += R[i][k] * pal[k % 8][j];
          pL.dot(X[i][0], X[i][1], { color: rgbStr(c.map(Math.round)), r: 3, strokeWidth: 0.5 });
        }
        for (let k = 0; k < K; k++) {
          pL.ellipse(g.mu[k][0], g.mu[k][1], g.cov[k], 1, { color: CL[k % 8], width: 2 });
          pL.ellipse(g.mu[k][0], g.mu[k][1], g.cov[k], 2, { color: CL[k % 8], width: 1.2, dash: true });
          pL.dot(g.mu[k][0], g.mu[k][1], { color: CL[k % 8], r: 5, stroke: 'ink' });
        }
        const lo = Math.min(...hist), hi = Math.max(...hist), pad = Math.max(1, (hi - lo) * 0.08);
        pR.setDomain([0, Math.max(10, hist.length)], [lo - pad, hi + pad]).clear();
        pR.line(hist.map((v, t) => [t, v]), { color: 's1', width: 2 });
        pR.points(hist.map((v, t) => [t, v]), { color: 's1', r: 2 });
        const ll = hist[hist.length - 1], n = X.length;
        sIt.set(String(hist.length - 1));
        sLL.set(S.fmt(ll, 5), hist.length > 1 ? `change ${S.fmt(ll - hist[hist.length - 2], 3)}` : '');
        sBIC.set(S.fmt(-2 * ll + nParams2(K, type.get()) * Math.log(n), 5), `d = ${nParams2(K, type.get())} parameters`);
        const me = Math.min(...g.cov.map(minEig2));
        sMin.set(me.toExponential(2), me < 1e-3 ? '<span style="color:var(--bad)">a component is collapsing</span>' : '');
      }
      newData();
    },
  });

  Atlas.lab('gaussian-mixtures-bic', {
    title: 'Choosing $K$ by BIC',
    blurb: 'Data are drawn from a mixture of a known number of groups; full-covariance GMMs with $K=1,\\dots,8$ are fitted (best of 3 EM runs each) and scored by BIC and AIC. With Gaussian groups BIC usually recovers the truth. Make the groups skewed or curved and BIC starts spending several Gaussians on each group: it selects a good density, not a count of “clusters”.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const K0 = UI.slider(controls, { label: 'True number of groups', min: 1, max: 5, step: 1, value: 3, onChange: newData });
      const shape = UI.segmented(controls, { label: 'Group shape', options: [{ value: 'gauss', label: 'Gaussian' }, { value: 'skew', label: 'Skewed' }, { value: 'curve', label: 'Curved' }], value: 'gauss', onChange: newData });
      const sep = UI.slider(controls, { label: 'Separation', min: 1, max: 5, step: 0.25, value: 3.5, onChange: newData });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 60, max: 600, step: 20, value: 300, onChange: newData });
      UI.button(controls, 'New data', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-4, 4], y: [-3, 3], equal: true, title: 'Data and the BIC-selected mixture' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0.5, 8.5], y: [0, 1], xlabel: 'K', ylabel: 'criterion', title: 'BIC and AIC (lower is better)' });
      const sB = UI.stat(stats, '$K$ chosen by BIC', { accent: 'accent' });
      const sA = UI.stat(stats, '$K$ chosen by AIC');
      const sT = UI.stat(stats, 'True groups');
      let X, fits;
      function newData() {
        const k0 = Math.round(K0.get()), n = Math.round(nS.get()), s = sep.get() * 0.45;
        X = [];
        const centres = Array.from({ length: k0 }, (_, k) => (k0 === 1 ? [0, 0] : [s * Math.cos(2 * Math.PI * k / k0 + 0.3), s * Math.sin(2 * Math.PI * k / k0 + 0.3)]));
        for (let i = 0; i < n; i++) {
          const k = i % k0, c = centres[k], z1 = ctx.rng.normal(), z2 = ctx.rng.normal();
          let dx, dy;
          if (shape.get() === 'gauss') { dx = 0.45 * z1; dy = 0.3 * z2; }
          else if (shape.get() === 'skew') { dx = 0.45 * (Math.exp(0.75 * z1) - Math.exp(0.28125)) / 1.1; dy = 0.3 * z2; }
          else { dx = 0.6 * z1; dy = 0.22 * (z1 * z1 - 1) + 0.12 * z2; }
          const a = 2 * Math.PI * k / Math.max(k0, 1);
          X.push([c[0] + Math.cos(a) * dx - Math.sin(a) * dy, c[1] + Math.sin(a) * dx + Math.cos(a) * dy]);
        }
        fits = [];
        for (let K = 1; K <= 8; K++) {
          const f = gmmFit(ctx.rng, X, K, { type: 'full', floor: 1e-4, iters: 60, restarts: 3 });
          const d = nParams2(K, 'full');
          fits.push({ K, g: f.g, ll: f.ll, bic: -2 * f.ll + d * Math.log(n), aic: -2 * f.ll + 2 * d });
        }
        draw();
      }
      function draw() {
        const kb = fits.reduce((a, b) => (b.bic < a.bic ? b : a)), ka = fits.reduce((a, b) => (b.aic < a.aic ? b : a));
        const ex = S.extent(X.map(x => x[0])), ey = S.extent(X.map(x => x[1]));
        pL.setDomain([ex[0] - 0.5, ex[1] + 0.5], [ey[0] - 0.5, ey[1] + 0.5]).clear();
        const lab = X.map(x => S.argmax(kb.g.pi.map((p, k) => Math.log(p) + logN2(x, kb.g.mu[k], kb.g.cov[k]))));
        drawClusters(pL, X, lab, { r: 2.4, opacity: 0.7 });
        kb.g.mu.forEach((m, k) => pL.ellipse(m[0], m[1], kb.g.cov[k], 2, { color: CL[k % 8], width: 2 }));
        const vals = fits.flatMap(f => [f.bic, f.aic]), lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.08 || 1;
        pR.setDomain([0.5, 8.5], [lo - pad, hi + pad]).clear();
        pR.line(fits.map(f => [f.K, f.bic]), { color: 's1', width: 2 });
        pR.points(fits.map(f => [f.K, f.bic]), { color: 's1', r: 3.5 });
        pR.line(fits.map(f => [f.K, f.aic]), { color: 's2', width: 2, dash: true });
        pR.points(fits.map(f => [f.K, f.aic]), { color: 's2', r: 3 });
        pR.dot(kb.K, kb.bic, { color: 's1', r: 7, stroke: 'ink' });
        pR.vline(Math.round(K0.get()), { color: 'ink', width: 1.2 });
        pR.legend([{ label: 'BIC', color: 's1' }, { label: 'AIC', color: 's2', dash: true }, { label: 'true groups', color: 'ink' }]);
        sB.set(String(kb.K)); sA.set(String(ka.K)); sT.set(String(Math.round(K0.get())));
      }
      newData();
    },
  });

  /* ================================================================== EM: the lower bound */

  Atlas.lab('em-algorithm-bound', {
    title: 'The lower bound that touches the likelihood',
    blurb: 'Black: the observed-data log-likelihood $\\ell(\\theta)$. Orange: the bound $\\mathcal L(q^{(t+1)},\\theta)$ built in the E-step at $\\theta^{(t)}$ — it lies below $\\ell$ and touches it there. The M-step jumps to the bound’s maximum. Right: the error $|\\theta^{(t)}-\\hat\\theta|$ on a log scale falls along a straight line whose slope is the fraction of missing information. Push the components together, or censor more, and watch EM crawl.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const prob = UI.segmented(controls, { label: 'Problem', options: [{ value: 'mix', label: 'Mixture: unknown mean' }, { value: 'cens', label: 'Censored exponential' }], value: 'mix', onChange: () => { syncCtl(); newData(); } });
      const nS = UI.slider(controls, { label: 'Sample size $n$', min: 10, max: 400, step: 10, value: 80, onChange: newData });
      const mu0 = UI.slider(controls, { label: 'True mean $\\mu$ of component 2', min: 0.25, max: 4, step: 0.25, value: 2, onChange: newData });
      const cf = UI.slider(controls, { label: 'Censoring probability', min: 0.05, max: 0.95, step: 0.05, value: 0.6, onChange: newData });
      const st = UI.slider(controls, { label: 'Start $\\theta^{(0)}$', min: -3, max: 6, step: 0.1, value: -1.5, onChange: reset });
      const stL = UI.slider(controls, { label: 'Start $\\lambda^{(0)}$', min: 0.1, max: 4, step: 0.05, value: 3, onChange: reset });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'EM step', () => { step(); draw(); }, { primary: true });
      let acc = 0;
      const loop = ctx.loop(dt => { acc += dt; if (acc > 0.5) { acc = 0; step(); draw(); if (path.length > 60) return false; } });
      ctx.playButton(row, loop);
      const row2 = UI.h('div', 'btn-row', controls);
      UI.button(row2, 'Reset', reset);
      UI.button(row2, 'New data', newData);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-3, 6], y: [0, 1], xlabel: 'θ', ylabel: 'log-likelihood', title: 'ℓ(θ) and the EM minorant' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0, 20], y: [-12, 1], xlabel: 'iteration t', ylabel: 'log10 |θ(t) − θ̂|', title: 'Linear convergence' });
      const sTh = UI.stat(stats, 'Current $\\theta^{(t)}$', { accent: 'accent' });
      const sLL = UI.stat(stats, '$\\ell(\\theta^{(t)})$');
      const sRate = UI.stat(stats, 'Observed error ratio');
      const sTh2 = UI.stat(stats, 'Missing-information fraction $I_c^{-1}I_m$');

      let x, cens, path, bounds, thetaHat;
      function syncCtl() { const m = prob.get() === 'mix'; hide(mu0, !m); hide(st, !m); hide(cf, m); hide(stL, m); }
      const phi = z => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
      // Problem definitions.
      const P = {
        mix: {
          ll: th => S.sum(x.map(v => Math.log(0.5 * phi(v) + 0.5 * phi(v - th)))),
          resp: th => x.map(v => { const a = phi(v - th), b = phi(v); return a / (a + b); }),
          bound(th0) { // returns a function of theta
            const r = this.resp(th0);
            const H = S.sum(r.map(q => (q > 0 && q < 1 ? -q * Math.log(q) - (1 - q) * Math.log(1 - q) : 0)));
            const c0 = S.sum(x.map((v, i) => (1 - r[i]) * Math.log(0.5 * phi(v)) + r[i] * Math.log(0.5)));
            return th => c0 + S.sum(x.map((v, i) => r[i] * Math.log(phi(v - th)))) + H;
          },
          em(th) { const r = this.resp(th); return S.sum(r.map((q, i) => q * x[i])) / S.sum(r); },
          rate(th) { const r = this.resp(th); return S.sum(r.map((q, i) => q * (1 - q) * (x[i] - th) ** 2)) / S.sum(r); },
          dom: () => [-3, 6],
        },
        cens: {
          stats() { const d = x.filter((_, i) => !cens[i]).length; return { d, m: x.length - d, T: S.sum(x), n: x.length }; },
          ll(l) { const { d, T } = this.stats(); return d * Math.log(l) - l * T; },
          bound(l0) { const { n, m, T } = this.stats(); return l => n * Math.log(l) - l * (T + m / l0) + m * (1 - Math.log(l0)); },
          em(l) { const { n, m, T } = this.stats(); return n / (T + m / l); },
          rate() { const { n, m } = this.stats(); return m / n; },
          dom() { const { d, T } = this.stats(); const lh = Math.max(d, 1) / T; return [0.02, Math.max(3 * lh, stL.get() * 1.15)]; },
        },
      };
      function newData() {
        const n = Math.round(nS.get());
        if (prob.get() === 'mix') {
          x = ctx.rng.array(n, () => (ctx.rng.u() < 0.5 ? ctx.rng.normal() : ctx.rng.normal(mu0.get(), 1)));
          cens = null;
        } else {
          const c = -Math.log(cf.get()); // P(Y > c) = cf for Exp(1)
          const y = ctx.rng.array(n, () => ctx.rng.exp(1));
          cens = y.map(v => v > c); x = y.map(v => Math.min(v, c));
          if (cens.every(Boolean)) { cens[0] = false; x[0] = c * 0.5; }
        }
        reset();
      }
      function reset() {
        loop.stop();
        const pr = P[prob.get()];
        const th0 = prob.get() === 'mix' ? st.get() : stL.get();
        path = [th0]; bounds = [];
        // Limit of this EM run: iterate, then polish (mixture) with Newton steps.
        let th = th0;
        for (let i = 0; i < 4000; i++) { const t2 = pr.em(th); if (Math.abs(t2 - th) < 1e-15) { th = t2; break; } th = t2; }
        if (prob.get() === 'mix') for (let k = 0; k < 30; k++) {
          const r = pr.resp(th);
          const g = S.sum(r.map((q, i) => q * (x[i] - th))), h = S.sum(r.map((q, i) => -q + q * (1 - q) * (x[i] - th) ** 2));
          if (!(h < 0)) break;
          const nt = th - g / h;
          if (!Number.isFinite(nt) || Math.abs(nt - th) > 0.5) break;
          th = nt;
        }
        thetaHat = th;
        draw();
      }
      function step() {
        const pr = P[prob.get()], th = path[path.length - 1];
        bounds.push({ at: th, f: pr.bound(th) });
        path.push(pr.em(th));
      }
      function draw() {
        const pr = P[prob.get()], [a, b] = pr.dom();
        const grid = S.linspace(a, b, 260), ll = grid.map(t => pr.ll(t));
        const llHat = pr.ll(thetaHat), llStart = pr.ll(path[0]);
        const span = Math.max(2, (llHat - llStart) * 1.4);
        pL.setDomain([a, b], [llHat - span, llHat + 0.12 * span]).clear();
        pL.line(grid.map((t, i) => [t, ll[i]]), { color: 'ink', width: 2.2 });
        bounds.slice(0, -1).slice(-4).forEach(B => pL.line(grid.map(t => [t, B.f(t)]), { color: 's2', width: 1.2, opacity: 0.25 }));
        const cur = path[path.length - 1];
        if (bounds.length) {
          const B = bounds[bounds.length - 1], prev = path[path.length - 2];
          pL.line(grid.map(t => [t, B.f(t)]), { color: 's2', width: 2.2 });
          pL.vline(prev, { color: 's2', width: 1 });
          pL.dot(prev, pr.ll(prev), { color: 's2', r: 5 });
          pL.dot(cur, B.f(cur), { color: 's2', r: 4 });
          if (Math.abs(cur - prev) > (b - a) * 0.01) pL.arrow(prev, pr.ll(prev), cur, B.f(cur), { color: 's2', width: 1.4 });
        }
        pL.vline(thetaHat, { color: 'muted', width: 1 });
        pL.dot(cur, pr.ll(cur), { color: 'accent', r: 6, stroke: 'ink' });
        pL.legend([{ label: 'ℓ(θ)', color: 'ink' }, { label: 'bound from last E-step', color: 's2' }, { label: 'θ̂ (limit of this run)', color: 'muted', dash: true }]);
        const errs = path.map((t, i) => [i, Math.log10(Math.max(Math.abs(t - thetaHat), 1e-16))]);
        const rate = pr.rate(thetaHat);
        pR.setDomain([0, Math.max(15, path.length + 2)], [-16, Math.max(1, errs[0][1] + 0.5)]).clear();
        if (rate > 0 && rate < 1) pR.line([[0, errs[0][1]], [80, errs[0][1] + 80 * Math.log10(rate)]], { color: 'muted', dash: true, width: 1.5 });
        pR.line(errs, { color: 's1', width: 2 });
        pR.points(errs, { color: 's1', r: 3 });
        pR.legend([{ label: 'EM error', color: 's1' }, { label: 'slope = log10(missing fraction)', color: 'muted', dash: true }]);
        sTh.set(S.fmt(cur, 6), `θ̂ = ${S.fmt(thetaHat, 6)}`);
        sLL.set(S.fmt(pr.ll(cur), 6));
        const k = path.length;
        const ratio = k >= 3 ? (path[k - 1] - thetaHat) / (path[k - 2] - thetaHat) : NaN;
        sRate.set(Number.isFinite(ratio) && Math.abs(path[k - 2] - thetaHat) > 1e-12 ? S.fmt(ratio, 4) : '—', '(θ(t+1) − θ̂)/(θ(t) − θ̂)');
        sTh2.set(S.fmt(rate, 4), prob.get() === 'cens' ? 'exactly the censored fraction m/n' : 'Σ r(1−r)(x−θ̂)² / Σ r');
      }
      syncCtl();
      newData();
    },
  });

  /* ================================================================== Hierarchical clustering */

  // Agglomerative clustering with Lance–Williams updates. Returns merges [{a, b, h}] with node ids
  // 0..n-1 for leaves and n+t for the t-th merge.
  function agglomerate(X, linkage) {
    const n = X.length, D = new Float64Array(n * n), size = new Array(n).fill(1), node = X.map((_, i) => i), alive = new Array(n).fill(true);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const d = Math.sqrt(sq2(X[i], X[j])); D[i * n + j] = linkage === 'ward' ? d * d : d; }
    const merges = [];
    for (let t = 0; t < n - 1; t++) {
      let bi = -1, bj = -1, bd = Infinity;
      for (let i = 0; i < n; i++) if (alive[i]) for (let j = i + 1; j < n; j++) if (alive[j] && D[i * n + j] < bd) { bd = D[i * n + j]; bi = i; bj = j; }
      merges.push({ a: node[bi], b: node[bj], h: linkage === 'ward' ? Math.sqrt(Math.max(bd, 0)) : bd });
      const ni = size[bi], nj = size[bj];
      for (let k = 0; k < n; k++) {
        if (!alive[k] || k === bi || k === bj) continue;
        const dik = D[bi * n + k], djk = D[bj * n + k], nk = size[k];
        let v;
        if (linkage === 'single') v = Math.min(dik, djk);
        else if (linkage === 'complete') v = Math.max(dik, djk);
        else if (linkage === 'average') v = (ni * dik + nj * djk) / (ni + nj);
        else v = ((ni + nk) * dik + (nj + nk) * djk - nk * bd) / (ni + nj + nk);
        D[bi * n + k] = D[k * n + bi] = v;
      }
      alive[bj] = false; size[bi] = ni + nj; node[bi] = n + t;
    }
    return merges;
  }
  function cutTree(n, merges, K) {
    const parent = Array.from({ length: 2 * n - 1 }, (_, i) => i);
    const find = i => (parent[i] === i ? i : (parent[i] = find(parent[i])));
    for (let t = 0; t < n - K; t++) { parent[find(merges[t].a)] = n + t; parent[find(merges[t].b)] = n + t; }
    const roots = new Map(), lab = [];
    for (let i = 0; i < n; i++) { const r = find(i); if (!roots.has(r)) roots.set(r, roots.size); lab.push(roots.get(r)); }
    return lab;
  }
  function leavesOf(n, merges) {
    const L = [];
    for (let i = 0; i < n; i++) L.push([i]);
    for (const m of merges) L.push(L[m.a].concat(L[m.b]));
    return L;
  }
  function primMST(X) {
    const n = X.length, inT = new Array(n).fill(false), best = new Array(n).fill(Infinity), from = new Array(n).fill(-1), edges = [];
    best[0] = 0;
    for (let it = 0; it < n; it++) {
      let u = -1;
      for (let i = 0; i < n; i++) if (!inT[i] && (u < 0 || best[i] < best[u])) u = i;
      inT[u] = true;
      if (from[u] >= 0) edges.push([from[u], u, Math.sqrt(best[u])]);
      for (let v = 0; v < n; v++) if (!inT[v]) { const d = sq2(X[u], X[v]); if (d < best[v]) { best[v] = d; from[v] = u; } }
    }
    return edges;
  }
  const HDATA = {
    bridge: { label: 'Two blobs + a bridge', gen(r) {
      const X = [];
      for (const c of [-2, 2]) for (let i = 0; i < 24; i++) X.push([c + r.normal(0, 0.38), r.normal(0, 0.38)]);
      for (let i = 0; i < 9; i++) X.push([-1.25 + 2.5 * i / 8 + r.normal(0, 0.04), r.normal(0, 0.06)]);
      return X;
    } },
    three: { label: 'Three blobs, unequal', gen(r) {
      const X = [];
      [[[-2, 1], 0.5, 26], [[1.8, 1.2], 0.3, 14], [[0, -1.4], 0.65, 26]].forEach(([c, s, m]) => { for (let i = 0; i < m; i++) X.push([c[0] + r.normal(0, s), c[1] + r.normal(0, s)]); });
      return X;
    } },
    bars: { label: 'Two long bars', gen(r) {
      const X = [];
      for (const y of [-0.6, 0.6]) for (let i = 0; i < 30; i++) X.push([-3 + 6 * i / 29 + r.normal(0, 0.05), y + r.normal(0, 0.12)]);
      return X;
    } },
    noise: { label: 'Uniform noise (no clusters)', gen(r) { return r.array(55, () => [r.uniform(-3, 3), r.uniform(-2, 2)]); } },
  };

  Atlas.lab('hierarchical-clustering-dendrogram', {
    title: 'Dendrograms under four linkages',
    blurb: 'Left: the data coloured by the $K$-cluster cut; click to add points. Right: the dendrogram, with the cut drawn as a dashed line. On “Two blobs + a bridge”, single linkage chains straight across the bridge, while complete, average and Ward keep the blobs apart. On “Two long bars” the roles reverse. Toggle the minimum spanning tree to see that single linkage merges exactly along its edges.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: Object.keys(HDATA).map(k => ({ value: k, label: HDATA[k].label })), value: 'bridge', onChange: newData });
      const link = UI.segmented(controls, { label: 'Linkage', options: [{ value: 'single', label: 'Single' }, { value: 'complete', label: 'Complete' }, { value: 'average', label: 'Average' }, { value: 'ward', label: 'Ward' }], value: 'single', onChange: compute });
      const Ks = UI.slider(controls, { label: 'Cut into $K$ clusters', min: 1, max: 8, step: 1, value: 2, onInput: draw });
      const mst = UI.toggle(controls, { label: 'Show minimum spanning tree', value: false, onChange: draw });
      UI.button(controls, 'New data', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-3.6, 3.6], y: [-2.6, 2.6], equal: true, title: 'Data (click to add a point)' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0, 1], y: [0, 1], xticks: 0, ylabel: 'merge height', title: 'Dendrogram' });
      const sH = UI.stat(stats, 'Cut height');
      const sSz = UI.stat(stats, 'Cluster sizes');
      const sCo = UI.stat(stats, 'Cophenetic correlation', { accent: 'accent' });
      const sMono = UI.stat(stats, 'Monotone merge heights?');
      let X, merges, coph, mstEdges;
      pL.onPointer((x, y, e, type) => { if (type === 'down' && pL.inside(x, y) && X.length < 90) { X.push([x, y]); compute(); } });
      function newData() { X = HDATA[ds.get()].gen(ctx.rng); compute(); }
      function compute() {
        merges = agglomerate(X, link.get());
        mstEdges = primMST(X);
        const n = X.length, L = leavesOf(n, merges), u = new Float64Array(n * n);
        merges.forEach((m, t) => { for (const i of L[m.a]) for (const j of L[m.b]) u[i * n + j] = u[j * n + i] = m.h; });
        const dv = [], uv = [];
        for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { dv.push(Math.sqrt(sq2(X[i], X[j]))); uv.push(u[i * n + j]); }
        coph = S.cor(dv, uv);
        draw();
      }
      function draw() {
        const n = X.length, K = Math.min(Math.round(Ks.get()), n), lab = cutTree(n, merges, K);
        pL.clear();
        if (mst.get()) for (const [a, b] of mstEdges) pL.segment(X[a][0], X[a][1], X[b][0], X[b][1], { color: 'muted', width: 1.2, opacity: 0.8 });
        drawClusters(pL, X, lab, { r: 3.6, opacity: 0.9 });
        // Dendrogram layout.
        const order = [], xs = new Array(2 * n - 1), ys = new Array(2 * n - 1).fill(0);
        (function walk(v) { if (v < n) { xs[v] = order.length; order.push(v); return; } const m = merges[v - n]; walk(m.a); walk(m.b); xs[v] = (xs[m.a] + xs[m.b]) / 2; ys[v] = m.h; })(2 * n - 2);
        const H = merges[n - 2].h, cutT = n - K;
        pR.setDomain([-1, n], [0, H * 1.06]).clear();
        // cluster of each node below the cut
        const nodeLab = new Array(2 * n - 1).fill(-1);
        for (let i = 0; i < n; i++) nodeLab[i] = lab[i];
        merges.forEach((m, t) => { nodeLab[n + t] = t < cutT ? nodeLab[m.a] : -1; });
        merges.forEach((m, t) => {
          const v = n + t, c = t < cutT ? CL[nodeLab[v] % 8] : 'muted', w = t < cutT ? 1.6 : 1.3;
          const ca = nodeLab[m.a] >= 0 && t >= cutT ? CL[nodeLab[m.a] % 8] : c, cb = nodeLab[m.b] >= 0 && t >= cutT ? CL[nodeLab[m.b] % 8] : c;
          pR.segment(xs[m.a], ys[m.a], xs[m.a], m.h, { color: ca, width: w });
          pR.segment(xs[m.b], ys[m.b], xs[m.b], m.h, { color: cb, width: w });
          pR.segment(xs[m.a], m.h, xs[m.b], m.h, { color: c, width: w });
        });
        const hc = K <= 1 ? H * 1.03 : (merges[cutT - 1] ? (merges[cutT - 1].h + merges[cutT].h) / 2 : merges[cutT].h / 2);
        pR.hline(hc, { color: 'ink', width: 1.4 });
        const sizes = Array.from({ length: K }, (_, k) => lab.filter(l => l === k).length).sort((a, b) => b - a);
        sH.set(S.fmt(hc, 3), `next merge at ${S.fmt(K > 1 ? merges[cutT].h : NaN, 3)}`);
        sSz.set(sizes.join(' · '));
        sCo.set(S.fmt(coph, 3), 'corr(d(i,j), u(i,j)) over all pairs');
        let mono = true;
        for (let t = 1; t < merges.length; t++) if (merges[t].h < merges[t - 1].h - 1e-12) mono = false;
        sMono.set(mono ? 'yes (ultrametric)' : '<span style="color:var(--bad)">no — inversion</span>');
      }
      newData();
    },
  });

  /* ================================================================== Spectral & density clustering */

  function kmeansBest(r, X, K, starts = 8) {
    let best = null;
    for (let s = 0; s < starts; s++) {
      const res = lloydND(X, seedPPND(r, X, K));
      if (!best || res.J < best.J) best = res;
    }
    return best;
  }
  const sqN = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return s; };
  function seedPPND(r, X, K) {
    const C = [X[Math.floor(r.u() * X.length)].slice()], D = X.map(x => sqN(x, C[0]));
    while (C.length < K) {
      let u = r.u() * S.sum(D), j = 0;
      while (j < X.length - 1 && (u -= D[j]) > 0) j++;
      C.push(X[j].slice());
      for (let i = 0; i < X.length; i++) D[i] = Math.min(D[i], sqN(X[i], X[j]));
    }
    return C;
  }
  function lloydND(X, C) {
    const n = X.length, p = X[0].length, lab = new Int32Array(n).fill(-1);
    let J = 0;
    for (let it = 0; it < 100; it++) {
      let changed = 0; J = 0;
      for (let i = 0; i < n; i++) {
        let b = 0, bd = Infinity;
        for (let k = 0; k < C.length; k++) { const d = sqN(X[i], C[k]); if (d < bd) { bd = d; b = k; } }
        if (lab[i] !== b) { changed++; lab[i] = b; }
        J += bd;
      }
      if (!changed) break;
      const s = C.map(() => new Array(p + 1).fill(0));
      for (let i = 0; i < n; i++) { const a = s[lab[i]]; for (let j = 0; j < p; j++) a[j] += X[i][j]; a[p]++; }
      C = s.map((a, k) => (a[p] ? a.slice(0, p).map(v => v / a[p]) : C[k]));
    }
    return { C, lab: Array.from(lab), J };
  }

  Atlas.lab('spectral-density-clustering-spectral', {
    title: 'Spectral clustering versus k-means',
    blurb: 'Build a similarity graph, take the bottom $K$ eigenvectors of $L_{\\text{sym}}$, normalise rows, run k-means on them. Right: the smallest eigenvalues of $L_{\\text{sym}}$ — a big gap after the $K$-th means $K$ well-separated groups. Shrink $\\sigma$ until the graph falls apart (many zero eigenvalues), or grow it until the gap closes and spectral clustering degrades toward a straight cut.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: dataOpts(['moons', 'rings', 'blobs4', 'aniso']), value: 'rings', onChange: newData });
      const Ks = UI.slider(controls, { label: 'Clusters $K$', min: 2, max: 5, step: 1, value: 2, onChange: compute });
      const graph = UI.segmented(controls, { label: 'Graph', options: [{ value: 'gauss', label: 'Gaussian, all pairs' }, { value: 'knn', label: '$k$-NN ($k=10$)' }], value: 'gauss', onChange: () => { hide(sig, graph.get() !== 'gauss'); compute(); } });
      const sig = UI.slider(controls, { label: 'Kernel scale $\\sigma$', min: 0.03, max: 3, log: true, value: 0.25, fmt: v => v.toFixed(3), onChange: compute });
      const view = UI.segmented(controls, { label: 'Show', options: [{ value: 'spec', label: 'Spectral' }, { value: 'km', label: 'k-means' }, { value: 'emb', label: 'Embedding' }], value: 'spec', onChange: draw });
      UI.button(controls, 'New data', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-4, 4], y: [-3, 3], equal: true, title: 'Clustering' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0.5, 10.5], y: [0, 1], xlabel: 'index j', ylabel: 'eigenvalue of L_sym', title: 'Bottom of the spectrum' });
      const sGap = UI.stat(stats, 'Eigengap $\\lambda_{K+1}-\\lambda_K$', { accent: 'accent' });
      const sZero = UI.stat(stats, 'Eigenvalues $\\lt 10^{-8}$', {});
      const sAS = UI.stat(stats, 'ARI spectral vs truth');
      const sAK = UI.stat(stats, 'ARI k-means vs truth');
      let X, Y, lam, emb, labS, labK;
      function newData() { ({ X, Y } = DATA2[ds.get()].gen(ctx.rng)); hide(sig, graph.get() !== 'gauss'); compute(); }
      function compute() {
        const n = X.length, K = Math.round(Ks.get());
        const W = Array.from({ length: n }, () => new Float64Array(n));
        if (graph.get() === 'gauss') {
          const s2 = 2 * sig.get() ** 2;
          for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) W[i][j] = W[j][i] = Math.exp(-sq2(X[i], X[j]) / s2);
        } else {
          for (let i = 0; i < n; i++) {
            const d = X.map((x, j) => [sq2(X[i], x), j]).sort((a, b) => a[0] - b[0]);
            for (let t = 1; t <= 10; t++) { const j = d[t][1]; W[i][j] = W[j][i] = 1; }
          }
        }
        const deg = W.map(r => S.sum(r)), dis = deg.map(d => (d > 1e-300 ? 1 / Math.sqrt(d) : 0));
        const M = W.map((r, i) => Array.from(r, (w, j) => w * dis[i] * dis[j]));
        const E = eigh(M); // ascending mu; L_sym eigenvalues 1 - mu
        lam = E.values.slice().reverse().map(m => 1 - m);
        const V = [];
        for (let k = 0; k < K; k++) V.push(E.vec(n - 1 - k));
        emb = X.map((_, i) => { const row = V.map(v => v[i]), nr = Math.hypot(...row) || 1; return row.map(v => v / nr); });
        labS = kmeansBest(ctx.rng, emb, K).lab;
        labK = kmeansBest(ctx.rng, X, K).lab;
        draw();
      }
      function draw() {
        const K = Math.round(Ks.get()), v = view.get();
        pL.clear();
        if (v === 'emb') {
          pL.setDomain([-1.3, 1.3], [-1.3, 1.3]);
          const pts = emb.map(r => [r[0], r[1] ?? 0]);
          drawClusters(pL, pts, labS, { r: 3, opacity: 0.6 });
        } else {
          pL.setDomain([-4, 4], [-3, 3]);
          drawClusters(pL, X, v === 'km' ? labK : labS, { r: 2.8 });
        }
        const m = Math.min(10, lam.length), top = Math.max(0.05, lam[m - 1] * 1.1);
        pR.setDomain([0.5, m + 0.5], [0, top]).clear();
        pR.band(0.5, K + 0.5, { color: 'accent', opacity: 0.08 });
        pR.stems(lam.slice(0, m).map((l, j) => [j + 1, Math.max(l, 0)]), { color: 's1' });
        sGap.set(S.fmt(lam[K] - lam[K - 1], 3), `λ_K = ${S.fmt(lam[K - 1], 3)}, λ_{K+1} = ${S.fmt(lam[K], 3)}`);
        sZero.set(String(lam.filter(l => l < 1e-8).length), 'one per connected component');
        sAS.set(S.fmt(ari(labS, Y), 3));
        sAK.set(S.fmt(ari(labK, Y), 3));
        pL.o.title = v === 'emb' ? 'Rows of the normalised eigenvector matrix' : v === 'km' ? 'k-means on the raw data' : 'Spectral clustering';
        pL.drawAxes();
      }
      newData();
    },
  });

  const DDATA = {
    moonsNoise: { label: 'Moons + background noise', gen(r) {
      const { X, Y } = DATA2.moons.gen(r, 0.1);
      for (let i = 0; i < 40; i++) { X.push([r.uniform(-3.2, 3.2), r.uniform(-2.2, 2.2)]); Y.push(-1); }
      return { X, Y };
    } },
    varied: { label: 'Three densities', gen(r) {
      const X = [], Y = [];
      gauss2(r, 80, [-0.9, 1.1], 0.15, 0.15, 0, 0, X, Y); gauss2(r, 80, [-0.1, 1.1], 0.15, 0.15, 0, 1, X, Y); gauss2(r, 70, [0.6, -1.1], 0.85, 0.55, 0, 2, X, Y);
      return { X, Y };
    } },
    ringsNoise: { label: 'Rings + background noise', gen(r) {
      const { X, Y } = DATA2.rings.gen(r, 0.1);
      for (let i = 0; i < 40; i++) { X.push([r.uniform(-3.2, 3.2), r.uniform(-2.8, 2.8)]); Y.push(-1); }
      return { X, Y };
    } },
  };
  function dbscan(X, eps, minPts) {
    const n = X.length, e2 = eps * eps, nb = [];
    for (let i = 0; i < n; i++) { const a = []; for (let j = 0; j < n; j++) if (sq2(X[i], X[j]) <= e2) a.push(j); nb.push(a); }
    const core = nb.map(a => a.length >= minPts), lab = new Array(n).fill(-1);
    let c = 0;
    for (let i = 0; i < n; i++) {
      if (!core[i] || lab[i] >= 0) continue;
      const q = [i]; lab[i] = c;
      while (q.length) { const u = q.pop(); for (const v of nb[u]) if (lab[v] < 0) { lab[v] = c; if (core[v]) q.push(v); } }
      c++;
    }
    return { lab, core, nClusters: c };
  }

  Atlas.lab('spectral-density-clustering-dbscan', {
    title: 'DBSCAN: core, border and noise',
    blurb: 'Large dots are core points (at least minPts points within $\\varepsilon$), small dots border points, grey dots noise. Hover to see the $\\varepsilon$-ball. Right: each point’s distance to its minPts-th neighbour, sorted — points above the line are not core. On “Three densities” no single $\\varepsilon$ works: small enough to keep the tight cluster clean and the wide cluster dissolves into noise.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data', options: Object.keys(DDATA).map(k => ({ value: k, label: DDATA[k].label })), value: 'moonsNoise', onChange: newData });
      const eps = UI.slider(controls, { label: 'Radius $\\varepsilon$', min: 0.05, max: 1.2, step: 0.01, value: 0.3, onInput: compute });
      const mp = UI.slider(controls, { label: 'minPts $m$', min: 2, max: 20, step: 1, value: 5, onInput: compute });
      UI.button(controls, 'New data', newData, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-3.4, 3.4], y: [-2.8, 2.8], equal: true, title: 'DBSCAN clusters' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0, 1], y: [0, 1], xlabel: 'points, sorted', ylabel: 'distance to m-th neighbour', title: 'k-distance plot' });
      const sC = UI.stat(stats, 'Clusters found', { accent: 'accent' });
      const sP = UI.stat(stats, 'Core / border / noise');
      const sA = UI.stat(stats, 'True groups recovered');
      let X, Y, res, ball = null;
      pL.onPointer((x, y, e, type) => {
        if (ball) { ball.remove(); ball = null; }
        if ((type === 'move' || type === 'drag') && pL.inside(x, y)) { const r2 = eps.get() ** 2; ball = pL.ellipse(x, y, [[r2, 0], [0, r2]], 1, { color: 'ink', width: 1.2, dash: '3 3' }); }
      });
      function newData() { ({ X, Y } = DDATA[ds.get()].gen(ctx.rng)); compute(); }
      function compute() {
        const m = Math.round(mp.get());
        res = dbscan(X, eps.get(), m);
        pL.clear(); ball = null;
        const K = res.nClusters;
        for (let k = 0; k < K; k++) {
          pL.points(X.filter((_, i) => res.lab[i] === k && res.core[i]), { color: CL[k % 8], r: 3.4, opacity: 0.9 });
          pL.points(X.filter((_, i) => res.lab[i] === k && !res.core[i]), { color: CL[k % 8], r: 1.8, opacity: 0.8 });
        }
        pL.points(X.filter((_, i) => res.lab[i] < 0), { color: 'faint', r: 2, opacity: 0.9 });
        // k-distance: distance to the m-th nearest point counting the point itself.
        const kd = X.map(x => { const d = X.map(z => sq2(x, z)).sort((a, b) => a - b); return Math.sqrt(d[Math.min(m - 1, d.length - 1)]); }).sort((a, b) => b - a);
        const top = Math.max(eps.get() * 1.3, S.quantile(kd.slice().reverse(), 0.98, true) * 1.1);
        pR.setDomain([0, kd.length], [0, top]).clear();
        pR.area(kd.map((v, i) => [i, Math.min(v, top)]), { color: 's1', opacity: 0.15 });
        pR.line(kd.map((v, i) => [i, v]), { color: 's1', width: 2 });
        pR.hline(eps.get(), { color: 'ink', width: 1.4 });
        pR.text(kd.length * 0.98, eps.get(), 'ε', { anchor: 'end', dy: -5, size: 13 });
        const nCore = res.core.filter(Boolean).length, nNoise = res.lab.filter(l => l < 0).length;
        sC.set(String(K));
        sP.set(`${nCore} / ${X.length - nCore - nNoise} / ${nNoise}`);
        // A true group counts as recovered if one DBSCAN cluster holds >= 80% of it and is >= 80% made of it.
        const groups = [...new Set(Y.filter(v => v >= 0))];
        let rec = 0;
        for (const gq of groups) {
          const idx = Y.map((v, i) => (v === gq ? i : -1)).filter(i => i >= 0);
          for (let k = 0; k < K; k++) {
            const inK = res.lab.filter(l => l === k).length, both = idx.filter(i => res.lab[i] === k).length;
            if (both >= 0.8 * idx.length && both >= 0.8 * inK) { rec++; break; }
          }
        }
        sA.set(`${rec} of ${groups.length}`, `ARI (noise as a class) ${S.fmt(ari(res.lab, Y), 3)}`);
      }
      newData();
    },
  });

  /* ================================================================== PCA */

  Atlas.lab('pca-projection', {
    title: 'Maximum variance = minimum reconstruction error',
    blurb: 'Rotate a candidate axis through the cloud. Every point is projected onto it (thin segments are the residuals). Right: variance captured along the axis and mean squared residual, as functions of the angle — they always sum to the total variance $\\tr S$, so the maximiser of one is the minimiser of the other: the first principal direction. Click on the left plot to add an outlier and watch PC1 swing toward it.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const rho = UI.slider(controls, { label: 'Correlation $\\rho$', min: -0.95, max: 0.95, step: 0.05, value: 0.6, onChange: newData });
      const s2 = UI.slider(controls, { label: 'SD of $x_2$ (SD of $x_1$ = 1.4)', min: 0.3, max: 1.4, step: 0.05, value: 0.9, onChange: newData });
      const ang = UI.slider(controls, { label: 'Candidate axis angle (degrees)', min: 0, max: 180, step: 1, value: 100, onInput: () => { snap.set(false); draw(); } });
      const snap = UI.toggle(controls, { label: 'Snap to PC1', value: false, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'New sample', newData, { primary: true });
      UI.button(row, 'Remove added points', () => { X = X.slice(0, base); draw(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [-4.5, 4.5], y: [-3.4, 3.4], equal: true, title: 'Projections onto a candidate axis' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0, 180], y: [0, 1], xlabel: 'axis angle (degrees)', ylabel: 'per-point average', title: 'Captured variance vs. residual' });
      const sV = UI.stat(stats, 'Variance captured $a^\\top Sa$', { accent: 'accent' });
      const sR = UI.stat(stats, 'Mean squared residual');
      const sT = UI.stat(stats, 'Sum $= \\tr S$');
      const sPC = UI.stat(stats, 'PC1: angle, share of variance');
      let X, base;
      pL.onPointer((x, y, e, type) => { if (type === 'down' && pL.inside(x, y)) { X.push([x, y]); draw(); } });
      function newData() {
        const a = 1.4, b = s2.get(), r = rho.get(), Sig = [[a * a, r * a * b], [r * a * b, b * b]];
        X = ctx.rng.array(120, () => ctx.rng.mvnormal([0, 0], Sig));
        base = X.length;
        draw();
      }
      function draw() {
        const n = X.length, m = [S.mean(X.map(x => x[0])), S.mean(X.map(x => x[1]))];
        let sxx = 0, sxy = 0, syy = 0;
        for (const x of X) { const dx = x[0] - m[0], dy = x[1] - m[1]; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
        const Sm = [[sxx / n, sxy / n], [sxy / n, syy / n]], E = S.eigSym(Sm), tr = Sm[0][0] + Sm[1][1];
        let pcAng = Math.atan2(E.vectors[1][0], E.vectors[0][0]) * 180 / Math.PI;
        if (pcAng < 0) pcAng += 180; if (pcAng >= 180) pcAng -= 180;
        if (snap.get()) ang.set(Math.round(pcAng * 10) / 10);
        const th = (snap.get() ? pcAng : ang.get()) * Math.PI / 180, u = [Math.cos(th), Math.sin(th)];
        const capt = t => { const c = Math.cos(t), s = Math.sin(t); return c * c * Sm[0][0] + 2 * c * s * Sm[0][1] + s * s * Sm[1][1]; };
        pL.clear();
        const L = 7;
        pL.segment(m[0] - L * u[0], m[1] - L * u[1], m[0] + L * u[0], m[1] + L * u[1], { color: 's2', width: 2 });
        for (const x of X) {
          const t = (x[0] - m[0]) * u[0] + (x[1] - m[1]) * u[1], px = m[0] + t * u[0], py = m[1] + t * u[1];
          pL.segment(x[0], x[1], px, py, { color: 's4', width: 1, opacity: 0.45 });
        }
        pL.points(X.slice(0, base), { color: 's1', r: 2.8, opacity: 0.85 });
        if (X.length > base) X.slice(base).forEach(x => pL.dot(x[0], x[1], { color: 'bad', r: 5 }));
        pL.points(X.map(x => { const t = (x[0] - m[0]) * u[0] + (x[1] - m[1]) * u[1]; return [m[0] + t * u[0], m[1] + t * u[1]]; }), { color: 's2', r: 1.8, opacity: 0.9 });
        for (let k = 0; k < 2; k++) { const l = 2 * Math.sqrt(Math.max(E.values[k], 0)); pL.arrow(m[0], m[1], m[0] + l * E.vectors[0][k], m[1] + l * E.vectors[1][k], { color: 'ink', width: 1.8 }); }
        pL.text(m[0] + 2 * Math.sqrt(E.values[0]) * E.vectors[0][0], m[1] + 2 * Math.sqrt(E.values[0]) * E.vectors[1][0], 'PC1', { dx: 6, dy: -4, weight: 600 });
        const grid = S.linspace(0, 180, 181);
        pR.setDomain([0, 180], [0, tr * 1.08]).clear();
        pR.hline(tr, { color: 'ink', width: 1.2 });
        pR.line(grid.map(g => [g, capt(g * Math.PI / 180)]), { color: 's2', width: 2.2 });
        pR.line(grid.map(g => [g, tr - capt(g * Math.PI / 180)]), { color: 's4', width: 2.2 });
        pR.vline(pcAng, { color: 'muted', width: 1 });
        const deg = th * 180 / Math.PI;
        pR.dot(deg, capt(th), { color: 's2', r: 5 });
        pR.dot(deg, tr - capt(th), { color: 's4', r: 5 });
        pR.legend([{ label: 'variance captured', color: 's2' }, { label: 'mean sq. residual', color: 's4' }, { label: 'total tr S', color: 'ink', dash: true }]);
        sV.set(S.fmt(capt(th), 4));
        sR.set(S.fmt(tr - capt(th), 4));
        sT.set(S.fmt(tr, 4));
        sPC.set(`${S.fmt(pcAng, 3)}°`, `λ₁/(λ₁+λ₂) = ${S.fmt(E.values[0] / tr, 3)}`);
      }
      newData();
    },
  });

  Atlas.lab('pca-spiked', {
    title: 'Noise eigenvalues and the BBP transition',
    blurb: 'Data with covariance $I + \\ell\\,vv^\\top$: white noise plus one spike of strength $\\ell$. Left: all $p$ sample eigenvalues against the Marchenko–Pastur density for pure noise. Even with $\\ell = 0$ the eigenvalues spread far from 1. Right: how well the top sample eigenvector aligns with $v$. Below $\\ell = \\sqrt{\\gamma}$ the spike is invisible; above it the top eigenvalue pops out of the bulk (overshooting $1+\\ell$) and the alignment rises.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const pS = UI.slider(controls, { label: 'Dimension $p$', min: 20, max: 150, step: 10, value: 100, onChange: sample });
      const gS = UI.slider(controls, { label: 'Aspect ratio $\\gamma = p/n$', min: 0.1, max: 1, step: 0.05, value: 0.25, onChange: sample });
      const lS = UI.slider(controls, { label: 'Spike strength $\\ell$', min: 0, max: 4, step: 0.1, value: 1.0, onChange: sample });
      UI.button(controls, 'Resample', sample, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [0, 4], y: [0, 1], xlabel: 'eigenvalue', ylabel: 'density', title: 'Sample eigenvalues vs. Marchenko–Pastur' });
      const pR = new Plot(stage, { width: 420, height: 300, x: [0, 4], y: [0, 1], xlabel: 'spike strength ℓ', ylabel: '⟨v̂, v⟩²', title: 'Eigenvector alignment' });
      const sTop = UI.stat(stats, 'Top sample eigenvalue', { accent: 'accent' });
      const sEdge = UI.stat(stats, 'Bulk edge $(1+\\sqrt\\gamma)^2$');
      const sOv = UI.stat(stats, 'Alignment $\\langle\\hat v,v\\rangle^2$');
      const sThr = UI.stat(stats, 'BBP threshold $\\sqrt\\gamma$');
      function sample() {
        const p = Math.round(pS.get()), g = gS.get(), n = Math.round(p / g), l = lS.get(), sl = Math.sqrt(l);
        const Sm = Array.from({ length: p }, () => new Float64Array(p));
        const x = new Float64Array(p);
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < p; j++) x[j] = ctx.rng.normal();
          x[0] += sl * ctx.rng.normal();
          for (let a = 0; a < p; a++) { const xa = x[a], row = Sm[a]; for (let b = a; b < p; b++) row[b] += xa * x[b]; }
        }
        for (let a = 0; a < p; a++) for (let b = a; b < p; b++) { Sm[a][b] /= n; Sm[b][a] = Sm[a][b]; }
        const E = eigh(Sm), ev = E.values, top = ev[p - 1], vh = E.vec(p - 1), ov = vh[0] * vh[0];
        const lo = (1 - Math.sqrt(g)) ** 2, hi = (1 + Math.sqrt(g)) ** 2;
        const spikeTh = l > Math.sqrt(g) ? (1 + l) * (1 + g / l) : hi;
        const xmax = Math.max(hi, top, 1 + l, spikeTh) * 1.08;
        const mp = S.linspace(lo, hi, 200).map(t => [t, Math.sqrt(Math.max((hi - t) * (t - lo), 0)) / (2 * Math.PI * g * t)]);
        const hist = S.histogram(ev, { bins: 40, lo: 0, hi: xmax });
        const ym = Math.max(...mp.map(q => q[1]), ...hist.map(b => b.density)) * 1.1;
        pL.setDomain([0, xmax], [0, ym]).clear();
        pL.bars(hist, { color: 's1', opacity: 0.45 });
        pL.line(mp, { color: 'ink', width: 2 });
        pL.vline(hi, { color: 'muted', width: 1.2 });
        pL.vline(1 + l, { color: 's3', width: 1.5 });
        if (l > Math.sqrt(g)) pL.vline(spikeTh, { color: 's2', width: 1.5 });
        pL.dot(top, ym * 0.04, { color: 's2', r: 5, stroke: 'ink' });
        pL.legend([{ label: 'sample eigenvalues', color: 's1', fill: true }, { label: 'MP density (pure noise)', color: 'ink' }, { label: 'true spike 1+ℓ', color: 's3', dash: true }, { label: 'predicted top (BBP)', color: 's2', dash: true }]);
        const theo = t => (t > Math.sqrt(g) ? (1 - g / (t * t)) / (1 + g / t) : 0);
        pR.setDomain([0, 4], [0, 1.02]).clear();
        pR.vline(Math.sqrt(g), { color: 'muted', width: 1.2 });
        pR.line(S.linspace(0, 4, 400).map(t => [t, theo(t)]), { color: 'ink', width: 2 });
        pR.dot(l, ov, { color: 's2', r: 6, stroke: 'ink' });
        pR.legend([{ label: 'asymptotic theory', color: 'ink' }, { label: 'threshold √γ', color: 'muted', dash: true }], { left: true });
        sTop.set(S.fmt(top, 4), `theory ${S.fmt(spikeTh, 4)}; true spike 1+ℓ = ${S.fmt(1 + l, 3)}`);
        sEdge.set(S.fmt(hi, 4), `n = ${n}, p = ${p}`);
        sOv.set(S.fmt(ov, 3), `theory ${S.fmt(theo(l), 3)}`);
        sThr.set(S.fmt(Math.sqrt(g), 3), l > Math.sqrt(g) ? 'spike above threshold' : 'spike below threshold');
      }
      sample();
    },
  });

  /* @@LABS@@ */
})();
