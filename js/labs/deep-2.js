/* Labs: Deep Learning II — CNNs, RNNs, attention, autoencoders/VAEs, GANs, diffusion. */
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const sigm = x => 1 / (1 + Math.exp(-x));

  /* ================================================================== */
  /*  Shared image helpers (CNN labs)                                    */
  /* ================================================================== */
  const IW = 28, IH = 28;
  function segDist(px, py, ax, ay, bx, by) {
    const vx = bx - ax, vy = by - ay, wx = px - ax, wy = py - ay;
    const t = clamp((wx * vx + wy * vy) / (vx * vx + vy * vy), 0, 1);
    return Math.hypot(px - (ax + t * vx), py - (ay + t * vy));
  }
  const inkOf = (d, r) => clamp(r + 0.5 - d, 0, 1);
  const IMAGES = {
    seven: { label: 'Digit "7"', f: (x, y) => inkOf(Math.min(segDist(x, y, 6, 6, 21, 6), segDist(x, y, 21, 6, 11, 23)), 1.5) },
    ring: { label: 'Ring', f: (x, y) => inkOf(Math.abs(Math.hypot(x - 14, y - 14) - 8), 1.4) },
    square: { label: 'Square', f: (x, y) => { const dx = Math.abs(x - 14), dy = Math.abs(y - 14); return inkOf(Math.abs(Math.max(dx, dy) - 7), 1.2); } },
    plus: { label: 'Plus', f: (x, y) => inkOf(Math.min(segDist(x, y, 14, 5, 14, 23), segDist(x, y, 5, 14, 23, 14)), 1.4) },
    checker: { label: 'Checkerboard', f: (x, y) => ((Math.floor(x / 4) + Math.floor(y / 4)) % 2 ? 1 : 0) },
    stripes: { label: 'Fine stripes', f: (x) => (Math.floor(x) % 2 ? 1 : 0) },
  };
  function makeImage(key) {
    const a = new Float64Array(IW * IH);
    for (let r = 0; r < IH; r++) for (let c = 0; c < IW; c++) a[r * IW + c] = IMAGES[key].f(c + 0.5, r + 0.5);
    return a;
  }
  const KERNELS = {
    identity: { label: 'Identity', k: [0, 0, 0, 0, 1, 0, 0, 0, 0] },
    box: { label: 'Box blur', k: Array(9).fill(1 / 9) },
    gauss: { label: 'Gaussian blur', k: [1, 2, 1, 2, 4, 2, 1, 2, 1].map(v => v / 16) },
    sobelx: { label: 'Sobel x (vertical edges)', k: [-1, 0, 1, -2, 0, 2, -1, 0, 1] },
    sobely: { label: 'Sobel y (horizontal edges)', k: [-1, -2, -1, 0, 0, 0, 1, 2, 1] },
    laplace: { label: 'Laplacian', k: [0, 1, 0, 1, -4, 1, 0, 1, 0] },
    sharpen: { label: 'Sharpen', k: [0, -1, 0, -1, 5, -1, 0, -1, 0] },
    custom: { label: 'Custom (sliders)', k: null },
  };
  // 2-D cross-correlation with a 3x3 kernel. mode: 'zero' | 'circular'.
  function conv2d(img, w, h, K, { stride = 1, pad = 1, dil = 1, mode = 'zero' } = {}) {
    const keff = dil * 2 + 1;
    const ow = Math.floor((w + 2 * pad - keff) / stride) + 1, oh = Math.floor((h + 2 * pad - keff) / stride) + 1;
    if (ow < 1 || oh < 1) return { data: new Float64Array(0), w: 0, h: 0 };
    const out = new Float64Array(ow * oh);
    for (let oy = 0; oy < oh; oy++) for (let ox = 0; ox < ow; ox++) {
      let s = 0;
      for (let u = 0; u < 3; u++) for (let v = 0; v < 3; v++) {
        let r = oy * stride + u * dil - pad, c = ox * stride + v * dil - pad;
        if (mode === 'circular') { r = ((r % h) + h) % h; c = ((c % w) + w) % w; }
        else if (r < 0 || r >= h || c < 0 || c >= w) continue;
        s += K[u * 3 + v] * img[r * w + c];
      }
      out[oy * ow + ox] = s;
    }
    return { data: out, w: ow, h: oh };
  }
  function maxPool2(m) {
    const ow = Math.floor(m.w / 2), oh = Math.floor(m.h / 2), out = new Float64Array(ow * oh);
    for (let r = 0; r < oh; r++) for (let c = 0; c < ow; c++) {
      const i = 2 * r * m.w + 2 * c;
      out[r * ow + c] = Math.max(m.data[i], m.data[i + 1], m.data[i + m.w], m.data[i + m.w + 1]);
    }
    return { data: out, w: ow, h: oh };
  }
  function subsample2(m) {
    const ow = Math.ceil(m.w / 2), oh = Math.ceil(m.h / 2), out = new Float64Array(ow * oh);
    for (let r = 0; r < oh; r++) for (let c = 0; c < ow; c++) out[r * ow + c] = m.data[2 * r * m.w + 2 * c];
    return { data: out, w: ow, h: oh };
  }
  function shiftImg(img, w, h, dx, dy, circular) {
    const out = new Float64Array(w * h);
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      let sr = r - dy, sc = c - dx;
      if (circular) { sr = ((sr % h) + h) % h; sc = ((sc % w) + w) % w; }
      else if (sr < 0 || sr >= h || sc < 0 || sc >= w) continue;
      out[r * w + c] = img[sr * w + sc];
    }
    return out;
  }
  const absMax = a => { let m = 0; for (const v of a) m = Math.max(m, Math.abs(v)); return m; };
  // Draw a w x h grid (row 0 at the top) into an equal-aspect plot; pad shows a frame of zeros.
  function drawGrid(p, arr, w, h, color, pad = 0) {
    p.setDomain([-pad, w + pad], [-pad, h + pad]).clear();
    const [x0, x1] = p.o.x, [y0, y1] = p.o.y;
    const nx = Math.min(480, Math.ceil((x1 - x0) * 5)), ny = Math.min(480, Math.ceil((y1 - y0) * 5));
    const padCol = color(0).slice(0, 3).concat([110]);
    p.raster((x, y) => {
      if (x < -pad || x >= w + pad || y < -pad || y >= h + pad) return [0, 0, 0, 0];
      if (x < 0 || x >= w || y < 0 || y >= h) return padCol;
      const c = Math.min(w - 1, Math.floor(x)), r = Math.min(h - 1, Math.floor(h - y));
      return color(arr[r * w + c]);
    }, { nx, ny, pixelated: true });
    p.rect(0, 0, w, h, { color: 'faint', opacity: 0, stroke: 'muted' });
  }
  function grayMap() {
    const bg = Atlas.rgb('surface'), fg = Atlas.rgb('ink');
    return v => Atlas.mix(bg, fg, clamp(v, 0, 1)).concat([255]);
  }
  function divMap(scale) {
    const bg = Atlas.rgb('surface'), pos = Atlas.rgb('s1'), neg = Atlas.rgb('s4');
    const sc = scale > 1e-12 ? scale : 1;
    return v => { const t = clamp(v / sc, -1, 1); return (t >= 0 ? Atlas.mix(bg, pos, t) : Atlas.mix(bg, neg, -t)).concat([255]); };
  }
  function kernelTable(K) {
    const cells = K.map(v => `<td style="padding:2px 6px;text-align:right;font-variant-numeric:tabular-nums">${S.fmt(v, 2)}</td>`);
    return `<table style="margin:0;font-size:12px"><tbody><tr>${cells.slice(0, 3).join('')}</tr><tr>${cells.slice(3, 6).join('')}</tr><tr>${cells.slice(6).join('')}</tr></tbody></table>`;
  }

  /* ------------------------------------------------------------------ */
  Atlas.lab('cnn-conv-explorer', {
    title: 'Convolution explorer: kernels, stride, padding, dilation',
    blurb: 'Left: the input with its zero padding (faint frame). Right: the feature map. Hover over the feature map to see the receptive field of that unit outlined on the input. Change stride, padding and dilation and check the output size against $\\lfloor (n+2p-d(k-1)-1)/s\\rfloor+1$. Note how the same 10 parameters are applied at every position.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const imgSel = UI.select(controls, { label: 'Input image', options: Object.keys(IMAGES).map(k => ({ value: k, label: IMAGES[k].label })), value: 'seven', onChange: () => { img = makeImage(imgSel.get()); draw(); } });
      const kSel = UI.select(controls, { label: 'Kernel', options: Object.keys(KERNELS).map(k => ({ value: k, label: KERNELS[k].label })), value: 'sobelx', onChange: draw });
      const st = UI.slider(controls, { label: 'Stride $s$', min: 1, max: 3, step: 1, value: 1, onInput: draw });
      const pd = UI.slider(controls, { label: 'Padding $p$', min: 0, max: 3, step: 1, value: 1, onInput: draw });
      const dl = UI.slider(controls, { label: 'Dilation $d$', min: 1, max: 3, step: 1, value: 1, onInput: draw });
      const relu = UI.toggle(controls, { label: 'ReLU after conv', value: false, onChange: draw });
      const pool = UI.toggle(controls, { label: '2×2 max-pool after', value: false, onChange: draw });
      UI.divider(controls, 'Custom kernel');
      const cC = UI.slider(controls, { label: 'centre', min: -4, max: 8, step: 0.5, value: 4, onInput: () => { kSel.set('custom'); draw(); } });
      const cE = UI.slider(controls, { label: 'edge neighbours', min: -2, max: 2, step: 0.25, value: -1, onInput: () => { kSel.set('custom'); draw(); } });
      const cK = UI.slider(controls, { label: 'corners', min: -2, max: 2, step: 0.25, value: 0, onInput: () => { kSel.set('custom'); draw(); } });
      const cA = UI.slider(controls, { label: 'left/right tilt', min: -2, max: 2, step: 0.25, value: 0, onInput: () => { kSel.set('custom'); draw(); } });
      stage.classList.add('two');
      const pIn = new Plot(stage, { width: 420, height: 300, equal: true, xticks: 0, yticks: 0, grid: false, title: 'Input (28×28) + padding', margin: { t: 28, r: 10, b: 12, l: 10 } });
      const pOut = new Plot(stage, { width: 420, height: 300, equal: true, xticks: 0, yticks: 0, grid: false, title: 'Feature map', margin: { t: 28, r: 10, b: 12, l: 10 } });
      const sSize = UI.stat(stats, 'Output size', { accent: 'accent' });
      const sPar = UI.stat(stats, 'Parameters: conv vs dense');
      const sRF = UI.stat(stats, 'Receptive field of one output');
      const sRange = UI.stat(stats, 'Activation range');
      const kNote = UI.note(stats, '');
      let img = makeImage('seven'), hover = null, cur = null;

      function kernel() {
        const k = kSel.get();
        if (k !== 'custom') return KERNELS[k].k;
        const c = cC.get(), e = cE.get(), q = cK.get(), a = cA.get();
        return [q - a, e, q + a, e - a, c, e + a, q - a, e, q + a];
      }
      function draw() {
        const K = kernel(), s = st.get(), p = pd.get(), d = dl.get();
        let m = conv2d(img, IW, IH, K, { stride: s, pad: p, dil: d });
        if (relu.get()) m = { data: m.data.map(v => Math.max(0, v)), w: m.w, h: m.h };
        const conv = m;
        if (pool.get() && m.w >= 2 && m.h >= 2) m = maxPool2(m);
        cur = { m, conv, s, p, d };
        drawGrid(pIn, img, IW, IH, grayMap(), p);
        if (hover && m.w) {
          const pool2 = pool.get() ? 2 : 1, keff = d * 2 + 1;
          const c0 = hover[0] * pool2 * s - p, r0 = hover[1] * pool2 * s - p;
          const span = (pool2 - 1) * s + keff;
          pIn.rect(c0, IH - r0 - span, c0 + span, IH - r0, { color: 'accent', opacity: 0.12, stroke: 'accent' });
        }
        if (m.w) {
          const sc = absMax(m.data);
          drawGrid(pOut, m.data, m.w, m.h, divMap(sc));
          if (hover) pOut.rect(hover[0], m.h - hover[1] - 1, hover[0] + 1, m.h - hover[1], { color: 'accent', opacity: 0.25, stroke: 'accent' });
        } else pOut.clear();
        const keff = d * 2 + 1, n1 = Math.floor((IW + 2 * p - keff) / s) + 1;
        sSize.set(m.w ? `${m.w}×${m.h}` : 'empty', `conv: ⌊(28+${2 * p}−${d * 2}−1)/${s}⌋+1 = ${n1}${pool.get() ? `; pool → ${Math.floor(n1 / 2)}` : ''}`);
        sPar.set(`10 vs ${(IW * IH * (m.w * m.h || 0) + (m.w * m.h || 0)).toLocaleString()}`, '9 weights + 1 bias, shared across all positions');
        const rf = pool.get() ? keff + s : keff;
        sRF.set(`${rf}×${rf}`, pool.get() ? 'r = k_eff + (2−1)·s after pooling' : 'k_eff = d(k−1)+1');
        if (m.w) { const [lo, hi] = S.extent(m.data); sRange.set(`[${S.fmt(lo, 2)}, ${S.fmt(hi, 2)}]`, 'blue > 0, red < 0'); }
        kNote.set('<b>Kernel</b> (cross-correlation weights)' + kernelTable(K));
      }
      pOut.onPointer((x, y, e, type) => {
        if (!cur || !cur.m.w) return;
        if (type === 'leave') { hover = null; draw(); return; }
        const c = Math.floor(x), r = Math.floor(cur.m.h - y);
        const nh = (c >= 0 && c < cur.m.w && r >= 0 && r < cur.m.h) ? [c, r] : null;
        if (String(nh) !== String(hover)) { hover = nh; draw(); }
      });
      draw();
    },
  });

  /* ------------------------------------------------------------------ */
  Atlas.lab('cnn-equivariance', {
    title: 'Translation equivariance — and how subsampling breaks it',
    blurb: 'We compute $F(T_\\tau x)$ and compare it with $T_{\\tau\'}F(x)$, where $F$ = conv → ReLU → (downsample) and $\\tau\'$ is the shift translated to the output grid. With stride 1 and circular boundaries the error is exactly zero. Switch on a 2× downsampling and try odd shifts; then compare max-pooling with blur-then-subsample. The right panel shows where the two disagree.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const imgSel = UI.select(controls, { label: 'Input image', options: Object.keys(IMAGES).map(k => ({ value: k, label: IMAGES[k].label })), value: 'seven', onChange: draw });
      const kSel = UI.select(controls, { label: 'Kernel', options: ['sobelx', 'sobely', 'laplace', 'gauss', 'sharpen'].map(k => ({ value: k, label: KERNELS[k].label })), value: 'sobelx', onChange: draw });
      const dx = UI.slider(controls, { label: 'Shift $\\tau_x$ (pixels)', min: -6, max: 6, step: 1, value: 1, onInput: draw });
      const dy = UI.slider(controls, { label: 'Shift $\\tau_y$ (pixels)', min: -6, max: 6, step: 1, value: 0, onInput: draw });
      const bnd = UI.segmented(controls, { label: 'Boundary', options: [{ value: 'circular', label: 'Circular (torus)' }, { value: 'zero', label: 'Zero padding' }], value: 'circular', onChange: draw });
      const down = UI.select(controls, { label: 'Downsampling', options: [{ value: 'none', label: 'None (stride 1)' }, { value: 'stride', label: 'Stride-2 convolution' }, { value: 'max', label: '2×2 max-pool' }, { value: 'blur', label: 'Blur, then subsample 2' }], value: 'none', onChange: draw });
      stage.classList.add('two');
      const pA = new Plot(stage, { width: 420, height: 300, equal: true, xticks: 0, yticks: 0, grid: false, title: 'F(shifted input)', margin: { t: 28, r: 10, b: 12, l: 10 } });
      const pD = new Plot(stage, { width: 420, height: 300, equal: true, xticks: 0, yticks: 0, grid: false, title: '|F(T x) − T′ F(x)|', margin: { t: 28, r: 10, b: 12, l: 10 } });
      const sEq = UI.stat(stats, 'Equivariance error', { accent: 'accent' });
      const sGap = UI.stat(stats, 'Global-average-pool change');
      const sGmx = UI.stat(stats, 'Global-max-pool change');
      const sNote = UI.note(stats, '');
      const BLUR = [1, 2, 1, 2, 4, 2, 1, 2, 1].map(v => v / 16);

      function F(x, circ) {
        const K = KERNELS[kSel.get()].k, mode = circ ? 'circular' : 'zero', dn = down.get();
        let m = conv2d(x, IW, IH, K, { stride: dn === 'stride' ? 2 : 1, pad: 1, mode });
        m = { data: m.data.map(v => Math.max(0, v)), w: m.w, h: m.h };
        if (dn === 'max') m = maxPool2(m);
        if (dn === 'blur') m = subsample2(conv2d(m.data, m.w, m.h, BLUR, { pad: 1, mode }));
        return m;
      }
      function draw() {
        const circ = bnd.get() === 'circular', tx = dx.get(), ty = dy.get(), dn = down.get();
        const x = makeImage(imgSel.get());
        const A = F(shiftImg(x, IW, IH, tx, ty, circ), circ);
        const B0 = F(x, circ);
        const f = dn === 'none' ? 1 : 2;
        const ox = Math.round(tx / f), oy = Math.round(ty / f);
        const B = shiftImg(B0.data, B0.w, B0.h, ox, oy, circ);
        const D = A.data.map((v, i) => Math.abs(v - B[i]));
        let num = 0, den = 0;
        for (let i = 0; i < D.length; i++) { num += D[i] * D[i]; den += B[i] * B[i]; }
        const sc = Math.max(absMax(A.data), 1e-9);
        drawGrid(pA, A.data, A.w, A.h, divMap(sc));
        drawGrid(pD, D, A.w, A.h, divMap(sc));
        const rel = Math.sqrt(num / Math.max(den, 1e-12));
        sEq.set((100 * rel).toFixed(rel < 0.001 ? 3 : 1) + '%', `relative L2, output shift (${ox}, ${oy})`);
        const gA = S.mean(Array.from(A.data)), gB = S.mean(Array.from(B0.data));
        const mA = absMax(A.data), mB = absMax(B0.data);
        sGap.set((100 * Math.abs(gA - gB) / Math.max(Math.abs(gB), 1e-12)).toFixed(2) + '%', 'invariance of the mean feature');
        sGmx.set((100 * Math.abs(mA - mB) / Math.max(mB, 1e-12)).toFixed(2) + '%', 'invariance of the max feature');
        const exact = dn === 'none' || (tx % 2 === 0 && ty % 2 === 0);
        sNote.set(exact ? (circ ? 'Shift is compatible with the output grid and the domain is a torus: equivariance should be exact (error 0).' : 'Compatible shift, but zero padding: content leaving the frame and the zero border make the maps differ near the edges only.')
          : 'Odd shift with 2× subsampling: no integer output shift matches, so the map changes (aliasing). Blur-then-subsample usually reduces the global-pool changes compared with max-pooling or striding.');
      }
      draw();
    },
  });

  /* ================================================================== */
  /*  RNN gradient lab                                                   */
  /* ================================================================== */
  function orthogonal(r, n) {
    const Q = [];
    for (let i = 0; i < n; i++) {
      let v = r.array(n, () => r.normal());
      for (const q of Q) { const d = S.dot(v, q); v = v.map((x, j) => x - d * q[j]); }
      const nv = Math.hypot(...v); Q.push(v.map(x => x / nv));
    }
    return Q; // rows orthonormal => orthogonal matrix
  }
  const matT_vec = (W, v) => { const n = W.length, out = new Array(n).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[j] += W[i][j] * v[i]; return out; };
  const vnorm = v => Math.hypot(...v);

  Atlas.lab('rnn-lstm-gradients', {
    title: 'Vanishing gradients: Jacobian products vs. the additive cell path',
    blurb: 'A 32-unit RNN is run for $T$ steps on random inputs; we backpropagate a unit vector from $\\mathbf h_T$ and plot $\\log_{10}\\lVert\\partial\\mathbf h_T/\\partial\\mathbf h_{T-k}\\rVert$ against the lag $k$ (averaged over 8 directions). The orange curve is the LSTM direct cell path $\\prod_j \\mathbf f_j$. Push the scale of $W$ above 1 with a linear RNN to see explosion; raise the input scale to see $\\tanh$ saturation accelerate vanishing; raise the forget bias to stretch the LSTM memory.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const n = 32;
      const rho = UI.slider(controls, { label: 'Scale of $W$ (spectral radius $\\approx\\rho$)', min: 0.5, max: 1.6, step: 0.01, value: 0.95, onInput: run });
      const wtype = UI.segmented(controls, { label: '$W$ type', options: [{ value: 'orth', label: 'ρ × orthogonal' }, { value: 'gauss', label: 'ρ × Gaussian/√n' }], value: 'orth', onChange: run });
      const nl = UI.segmented(controls, { label: 'Nonlinearity $\\phi$', options: ['tanh', 'linear'], value: 'tanh', onChange: run });
      const insc = UI.slider(controls, { label: 'Input scale', min: 0, max: 3, step: 0.05, value: 0.5, onInput: run });
      const bf = UI.slider(controls, { label: 'LSTM forget bias $b_f$', min: -2, max: 6, step: 0.1, value: 3, onInput: run });
      const T = UI.slider(controls, { label: 'Sequence length $T$', min: 20, max: 200, step: 5, value: 100, onChange: run });
      UI.button(controls, 'New random weights', () => { base = null; run(); });
      const p = new Plot(stage, { width: 640, height: 320, x: [0, 100], y: [-12, 4], xlabel: 'lag k (steps back from T)', ylabel: 'log10 gradient norm', title: '' });
      const sSig = UI.stat(stats, '$\\sigma_{\\max}(W)$ / $\\rho(W)$');
      const sV = UI.stat(stats, 'Vanilla: per-step factor', { accent: 'accent' });
      const sL = UI.stat(stats, 'LSTM path: per-step factor');
      const sSat = UI.stat(stats, 'Mean $\\phi\'(a)$ along the run');
      let base = null;
      function build() {
        const r = ctx.rng;
        base = { O: orthogonal(r, n), G: r.array(n, () => r.array(n, () => r.normal() / Math.sqrt(n))), u: r.array(n, () => r.normal()),
          xs: r.array(200, () => r.normal()), vs: r.array(8, () => { const v = r.array(n, () => r.normal()); const m = vnorm(v); return v.map(x => x / m); }),
          gn: r.array(200, () => r.array(n, () => r.normal())) };
      }
      function run() {
        if (!base) build();
        const TT = Math.round(T.get()), sc = rho.get(), tanh = nl.get() === 'tanh';
        const W0 = wtype.get() === 'orth' ? base.O : base.G;
        const W = W0.map(row => row.map(v => v * sc));
        // forward
        let h = new Array(n).fill(0);
        const dphi = [];
        let satSum = 0;
        for (let t = 0; t < TT; t++) {
          const a = S.matvec(W, h).map((v, i) => v + insc.get() * base.u[i] * base.xs[t]);
          h = tanh ? a.map(Math.tanh) : a.map(v => clamp(v, -1e150, 1e150));
          const dp = tanh ? h.map(v => 1 - v * v) : new Array(n).fill(1);
          dphi.push(dp); satSum += S.mean(dp);
        }
        // backward (vanilla)
        const van = new Array(TT).fill(0);
        for (const v0 of base.vs) {
          let d = v0.slice(); let logScale = 0;
          for (let k = 0; k < TT; k++) {
            van[k] += (logScale + Math.log10(Math.max(vnorm(d), 1e-300))) / base.vs.length;
            const dp = dphi[TT - 1 - k];
            d = matT_vec(W, d.map((x, i) => x * dp[i]));
            const m = vnorm(d); if (m > 0 && Number.isFinite(m)) { logScale += Math.log10(m); d = d.map(x => x / m); } else { logScale = -Infinity; }
          }
        }
        // LSTM direct path: prod of forget gates
        const lst = new Array(TT).fill(0); let lf = 0;
        for (const v0 of base.vs) {
          let d = v0.slice();
          for (let k = 0; k < TT; k++) {
            lst[k] += Math.log10(Math.max(vnorm(d), 1e-300)) / base.vs.length;
            const g = base.gn[(TT - 1 - k) % 200];
            d = d.map((x, i) => x * sigm(bf.get() + 0.5 * g[i]));
          }
        }
        for (let k = 0; k < 200; k++) for (let i = 0; i < n; i++) lf += Math.log(sigm(bf.get() + 0.5 * base.gn[k][i])) / (200 * n);
        // spectral quantities
        let v = base.vs[0].slice();
        for (let it = 0; it < 60; it++) { v = matT_vec(W, S.matvec(W, v)); const m = vnorm(v); v = v.map(x => x / m); }
        const smax = vnorm(S.matvec(W, v));
        let w = base.vs[1].slice(), lr = 0; const M = 120;
        for (let it = 0; it < M; it++) { w = S.matvec(W, w); const m = vnorm(w); lr += Math.log(m); w = w.map(x => x / m); }
        const rhoEst = Math.exp(lr / M);
        // plot
        const ys = van.concat(lst).filter(Number.isFinite);
        const ylo = Math.max(-16, Math.min(-1, ...ys) - 0.5), yhi = Math.min(16, Math.max(1, ...ys) + 0.5);
        p.setDomain([0, TT], [ylo, yhi]).clear();
        p.hline(0, { color: 'faint', dash: false, width: 1 });
        p.line(S.range(TT).map(k => [k, k * Math.log10(smax)]), { color: 's1', dash: '5 4', width: 1.3, opacity: 0.7 });
        p.line(van.map((y, k) => [k, y]), { color: 's1', width: 2.4 });
        p.line(lst.map((y, k) => [k, y]), { color: 's2', width: 2.4 });
        p.legend([{ label: 'vanilla RNN: ∏ diag(φ′)W', color: 's1' }, { label: 'bound (γ σmax)^k, γ = 1', color: 's1', dash: true }, { label: 'LSTM cell path: ∏ diag(f)', color: 's2' }], { left: true });
        sSig.set(`${S.fmt(smax, 3)} / ${S.fmt(rhoEst, 3)}`, wtype.get() === 'gauss' ? 'non-normal W: σmax ≈ 2ρ, so the bound is loose' : 'orthogonal: σmax = ρ');
        const half = Math.floor(TT / 2), fv = Number.isFinite(van[TT - 1]) ? 10 ** ((van[TT - 1] - van[half]) / (TT - 1 - half)) : 0;
        sV.set(S.fmt(fv, 3), fv < 1 ? `10⁻⁶ after ≈ ${Number.isFinite(Math.log(fv)) ? Math.round(-6 * Math.LN10 / Math.log(fv)) : '—'} steps` : 'gradient grows: exploding');
        const ff = Math.exp(lf);
        sL.set(S.fmt(ff, 3), `half-life ≈ ${S.fmt(Math.LN2 / Math.max(1e-9, -lf), 3)} steps`);
        sSat.set(S.fmt(satSum / TT, 3), tanh ? 'below 1 = saturation shrinks the Jacobian' : 'linear: φ′ ≡ 1');
      }
      run();
    },
  });

  /* ================================================================== */
  /*  Attention labs                                                     */
  /* ================================================================== */
  function softmax(s) { const m = Math.max(...s); const e = s.map(v => Math.exp(v - m)); const z = S.sum(e); return e.map(v => v / z); }

  Atlas.lab('attention-transformers-kernel', {
    title: 'Attention as a Nadaraya–Watson smoother',
    blurb: 'Keys are positions $x_j$, values are noisy $y_j$. Each query $x_q$ returns $\\sum_j \\mathrm{softmax}_j(s(x_q,x_j))\\,y_j$. With a Gaussian score or a dot product of unit-circle embeddings, the result tracks a Gaussian kernel smoother with bandwidth $\\tau$ (dashed). With a raw scalar dot product, attention is not local at all. Click the left plot to move the query; the right panel is the full self-attention matrix among the data points.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const score = UI.segmented(controls, { label: 'Score $s(q,k)$', options: [{ value: 'gauss', label: 'Gaussian' }, { value: 'circle', label: 'Unit-circle dot' }, { value: 'raw', label: 'Raw dot' }], value: 'circle', onChange: draw });
      const tau = UI.slider(controls, { label: 'Bandwidth / temperature $\\tau$', min: 0.005, max: 1, value: 0.06, log: true, fmt: v => v.toFixed(3), onInput: draw });
      const nS = UI.slider(controls, { label: 'Number of tokens $n$', min: 8, max: 60, step: 1, value: 30, onChange: newData });
      const causal = UI.toggle(controls, { label: 'Causal mask (attend only to $x_j \\le x_q$)', value: false, onChange: draw });
      UI.button(controls, 'New data', newData);
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 300, x: [0, 1], y: [-2.2, 1.8], xlabel: 'position x', ylabel: 'value', title: 'Smoother output and one query\'s weights' });
      const pR = new Plot(stage, { width: 420, height: 300, equal: true, xticks: 0, yticks: 0, grid: false, title: 'Self-attention matrix (row = query)', margin: { t: 28, r: 10, b: 12, l: 10 } });
      const sEff = UI.stat(stats, 'Effective # attended $1/\\sum p_j^2$', { accent: 'accent' });
      const sMax = UI.stat(stats, 'Largest weight');
      const sEnt = UI.stat(stats, 'Entropy / log n');
      const sMap = UI.stat(stats, 'Score in use');
      let xs = [], ys = [], xq = 0.37;
      const om = Math.PI;
      function newData() {
        const n = Math.round(nS.get());
        xs = ctx.rng.array(n, () => ctx.rng.uniform()).sort((a, b) => a - b);
        ys = xs.map(x => Math.sin(2 * Math.PI * x) * 1.1 + 0.3 * ctx.rng.normal());
        draw();
      }
      function scoreFn() {
        const t = tau.get(), k = score.get();
        if (k === 'gauss') return (q, x) => -((q - x) ** 2) / (2 * t * t);
        if (k === 'circle') { const beta = 1 / (om * om * t * t); return (q, x) => beta * Math.cos(om * (q - x)); }
        return (q, x) => (2 * q - 1) * (2 * x - 1) / t;
      }
      function weights(q, sf) {
        const s = xs.map(x => (causal.get() && x > q + 1e-12) ? -Infinity : sf(q, x));
        if (s.every(v => v === -Infinity)) return xs.map(() => 0);
        return softmax(s);
      }
      function draw() {
        const sf = scoreFn(), t = tau.get();
        pL.clear();
        const grid = S.linspace(0, 1, 220);
        const att = grid.map(q => { const w = weights(q, sf); return S.sum(w) > 0 ? [q, S.dot(w, ys)] : null; });
        const nw = grid.map(q => { const w = weights(q, (a, b) => -((a - b) ** 2) / (2 * t * t)); return S.sum(w) > 0 ? [q, S.dot(w, ys)] : null; });
        const wq = weights(xq, sf), mx = Math.max(...wq, 1e-12);
        for (let j = 0; j < xs.length; j++) if (wq[j] > 1e-4) pL.segment(xs[j], -2.2, xs[j], -2.2 + 0.9 * wq[j] / mx, { color: 's3', width: 3, opacity: 0.8 });
        pL.vline(xq, { color: 's3', width: 1.5 });
        pL.line(grid.map(x => [x, Math.sin(2 * Math.PI * x) * 1.1]), { color: 'faint', width: 1.2 });
        pL.points(xs.map((x, j) => [x, ys[j]]), { color: 'ink', r: 2.6, opacity: 0.75 });
        if (score.get() !== 'raw') pL.line(nw, { color: 's2', width: 2, dash: '5 4' });
        pL.line(att, { color: 's1', width: 2.4 });
        pL.legend([{ label: 'attention output', color: 's1' }, { label: 'Gaussian NW, h = τ', color: 's2', dash: true }, { label: 'weights of query', color: 's3' }]);
        // heat map of self-attention among data points
        const n = xs.length, A = xs.map(q => weights(q, sf));
        const bg = Atlas.rgb('surface'), fg = Atlas.rgb('s1'), mm = Math.max(...A.map(r => Math.max(...r)));
        pR.setDomain([0, n], [0, n]).clear();
        pR.raster((x, y) => {
          if (x < 0 || x >= n || y < 0 || y >= n) return [0, 0, 0, 0];
          const i = Math.min(n - 1, Math.floor(n - y)), j = Math.min(n - 1, Math.floor(x));
          return Atlas.mix(bg, fg, Math.sqrt(A[i][j] / mm)).concat([255]);
        }, { nx: Math.min(400, n * 6), ny: Math.min(400, n * 6), pixelated: true });
        pR.rect(0, 0, n, n, { color: 'faint', opacity: 0, stroke: 'muted' });
        const sq = S.sum(wq.map(v => v * v)), H = -S.sum(wq.map(v => (v > 0 ? v * Math.log(v) : 0)));
        sEff.set(sq > 0 ? S.fmt(1 / sq, 3) : '—', `of n = ${n} tokens`);
        sMax.set(S.fmt(Math.max(...wq), 3));
        sEnt.set(S.fmt(H / Math.log(n), 3), '1 = uniform, 0 = one-hot');
        const k = score.get();
        sMap.set(k === 'gauss' ? '−(q−k)²/2τ²' : k === 'circle' ? `β cos(π(q−k)), β = ${S.fmt(1 / (om * om * t * t), 3)}` : '(2q−1)(2k−1)/τ', k === 'circle' ? '≈ Gaussian, h = 1/(π√β) = τ' : k === 'raw' ? 'monotone in k: not a local kernel' : 'exact NW kernel');
      }
      pL.onPointer((x, y, e, type) => { if ((type === 'down' || type === 'drag') && x >= 0 && x <= 1) { xq = x; draw(); } });
      newData();
    },
  });

  Atlas.lab('attention-transformers-scaling', {
    title: 'Why $1/\\sqrt{d_k}$: logit variance and softmax saturation',
    blurb: 'Queries and keys have iid $\\N(0,1)$ coordinates, so $\\mathbf q^\\top\\mathbf k$ has variance $d_k$. (We simulate this exactly and cheaply: given $\\mathbf q$, the $n$ logits are iid $\\N(0,\\lVert\\mathbf q\\rVert^2)$ with $\\lVert\\mathbf q\\rVert^2\\sim\\chi^2_{d}$.) Left: the sorted attention weights of one query, unscaled vs scaled. Right: how entropy and the size of the softmax Jacobian $\\diag(\\mathbf p)-\\mathbf p\\mathbf p^\\top$ depend on $d_k$.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const dS = UI.slider(controls, { label: 'Key dimension $d_k$', min: 1, max: 2048, value: 256, log: true, fmt: v => String(Math.round(v)), onInput: drawLeft });
      const nS = UI.slider(controls, { label: 'Number of keys $n$', min: 4, max: 512, value: 64, log: true, fmt: v => String(Math.round(v)), onChange: () => { curve = null; draw(); } });
      UI.button(controls, 'New query', drawLeft, { primary: true });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, x: [0, 20], y: [0, 1], xlabel: 'key rank (sorted by weight)', ylabel: 'attention weight', title: 'One query: top-20 weights' });
      const pR = new Plot(stage, { width: 420, height: 280, x: [1, 2048], y: [0, 1], logx: true, xlabel: 'd_k', ylabel: 'average over 150 queries', title: 'Saturation vs dimension' });
      const sSd = UI.stat(stats, 'Logit sd: unscaled / scaled', { accent: 'accent' });
      const sMx = UI.stat(stats, 'Max weight: unscaled / scaled');
      const sJ = UI.stat(stats, '$\\lVert\\diag(\\mathbf p)-\\mathbf p\\mathbf p^\\top\\rVert_F$: unscaled / scaled');
      let curve = null;
      const jac = p => { let s2 = 0, s3 = 0; for (const v of p) { s2 += v * v; s3 += v * v * v; } return Math.sqrt(Math.max(0, s2 - 2 * s3 + s2 * s2)); };
      const ent = p => -S.sum(p.map(v => (v > 0 ? v * Math.log(v) : 0)));
      function trial(r, d, n) {
        const R = Math.sqrt(r.chisq(d)), z = r.array(n, () => r.normal());
        return { raw: softmax(z.map(v => R * v)), sc: softmax(z.map(v => R * v / Math.sqrt(d))), sd: R };
      }
      function computeCurve() {
        const n = Math.round(nS.get()), r = S.rng(31), ds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
        curve = ds.map(d => {
          let eR = 0, eS = 0, jR = 0, jS = 0; const M = 150;
          for (let m = 0; m < M; m++) { const t = trial(r, d, n); eR += ent(t.raw) / Math.log(n); eS += ent(t.sc) / Math.log(n); jR += jac(t.raw); jS += jac(t.sc); }
          return { d, eR: eR / M, eS: eS / M, jR: jR / M, jS: jS / M };
        });
      }
      function draw() { drawRight(); drawLeft(); }
      function drawRight() {
        if (!curve) computeCurve();
        pR.clear();
        const jm = Math.max(...curve.map(c => Math.max(c.jR, c.jS))) * 1.1;
        pR.line(curve.map(c => [c.d, c.eR]), { color: 's1', width: 2.2 });
        pR.line(curve.map(c => [c.d, c.eS]), { color: 's2', width: 2.2 });
        pR.line(curve.map(c => [c.d, c.jR / jm]), { color: 's1', width: 1.6, dash: '5 4' });
        pR.line(curve.map(c => [c.d, c.jS / jm]), { color: 's2', width: 1.6, dash: '5 4' });
        pR.vline(dS.get(), { color: 'muted' });
        pR.legend([{ label: 'entropy/log n, unscaled', color: 's1' }, { label: 'entropy/log n, scaled', color: 's2' }, { label: 'Jacobian norm (rel.), unscaled', color: 's1', dash: true }, { label: 'Jacobian norm (rel.), scaled', color: 's2', dash: true }], { left: true });
      }
      function drawLeft() {
        const d = Math.round(dS.get()), n = Math.round(nS.get());
        const t = trial(ctx.rng, d, n);
        const a = t.raw.slice().sort((x, y) => y - x).slice(0, 20), b = t.sc.slice().sort((x, y) => y - x).slice(0, 20);
        pL.setDomain([0, 20], [0, 1]).clear();
        pL.bars(a.map((v, i) => ({ x0: i + 0.05, x1: i + 0.48, y: v })), { color: 's1', opacity: 0.8 });
        pL.bars(b.map((v, i) => ({ x0: i + 0.52, x1: i + 0.95, y: v })), { color: 's2', opacity: 0.8 });
        pL.hline(1 / n, { color: 'muted' });
        pL.text(19.8, 1 / n, 'uniform 1/n', { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        pL.legend([{ label: 'unscaled q·k', color: 's1', fill: true }, { label: 'scaled q·k/√d', color: 's2', fill: true }]);
        sSd.set(`${S.fmt(t.sd, 3)} / ${S.fmt(t.sd / Math.sqrt(d), 3)}`, `theory: √d = ${S.fmt(Math.sqrt(d), 3)} / 1`);
        sMx.set(`${S.fmt(a[0], 3)} / ${S.fmt(b[0], 3)}`);
        sJ.set(`${S.fmt(jac(t.raw), 3)} / ${S.fmt(jac(t.sc), 3)}`, 'near 0 = no gradient through the softmax');
        if (curve) { drawRight(); }
      }
      draw();
    },
  });

  /* ================================================================== */
  /*  Autoencoder / VAE labs                                             */
  /* ================================================================== */
  Atlas.lab('autoencoders-vae-pca', {
    title: 'A linear autoencoder rediscovers PCA',
    blurb: 'A $2\\to1\\to2$ linear autoencoder $\\hat{\\mathbf x} = AB\\mathbf x$ is trained by gradient descent on squared error. Watch the decoder direction (span of $A$) rotate onto the first principal axis and the loss fall to $\\lambda_2$, the discarded eigenvalue. Untied, the encoder direction $B^\\top$ need not equal $A$ until convergence (the reconstruction is an oblique projection along the way). On the curved data the best line is still PC1 — a linear bottleneck cannot follow the arc.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const shape = UI.select(controls, { label: 'Data', options: [{ value: 'elong', label: 'Elongated Gaussian' }, { value: 'iso', label: 'Nearly isotropic' }, { value: 'arc', label: 'Curved arc' }], value: 'elong', onChange: reset });
      const lr = UI.slider(controls, { label: 'Learning rate $\\eta$', min: 0.002, max: 0.3, value: 0.03, log: true, fmt: v => v.toFixed(3) });
      const tied = UI.toggle(controls, { label: 'Tie weights ($B = A^\\top$)', value: false, onChange: () => { if (tied.get()) { B = A.slice(); } draw(); } });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Train', '❚❚ Pause']);
      UI.button(row, 'Re-initialize', reset);
      stage.classList.add('two');
      const pD = new Plot(stage, { width: 420, height: 300, x: [-4, 4], y: [-3, 3], equal: true, xlabel: 'x₁', ylabel: 'x₂', title: 'Data, reconstructions, directions' });
      const pL = new Plot(stage, { width: 420, height: 300, x: [0, 400], y: [0, 4], xlabel: 'iteration', ylabel: 'mean squared error', title: 'Training loss' });
      const sLoss = UI.stat(stats, 'Loss / optimum $\\lambda_2$', { accent: 'accent' });
      const sAng = UI.stat(stats, 'Angle(span A, PC1)');
      const sEnc = UI.stat(stats, 'Angle(Bᵀ, A)');
      const sProj = UI.stat(stats, '$\\lVert AB - \\mathbf u_1\\mathbf u_1^\\top\\rVert_F$');
      let X = [], A = [0, 0], B = [0, 0], hist = [], it = 0, eig = null;
      function genData() {
        const r = S.rng(11 + ['elong', 'iso', 'arc'].indexOf(shape.get())), n = 160;
        let pts;
        if (shape.get() === 'arc') pts = r.array(n, () => { const t = r.uniform(-1.3, 1.3); return [2.6 * Math.sin(t) + 0.15 * r.normal(), 2.4 * Math.cos(t) - 1.4 + 0.15 * r.normal()]; });
        else {
          const [s1, s2] = shape.get() === 'elong' ? [1.8, 0.45] : [1.15, 1.0], th = 0.55;
          pts = r.array(n, () => { const a = s1 * r.normal(), b = s2 * r.normal(); return [a * Math.cos(th) - b * Math.sin(th), a * Math.sin(th) + b * Math.cos(th)]; });
        }
        const mx = S.mean(pts.map(p => p[0])), my = S.mean(pts.map(p => p[1]));
        X = pts.map(p => [p[0] - mx, p[1] - my]);
        const C = [[0, 0], [0, 0]];
        for (const [a, b] of X) { C[0][0] += a * a / n; C[0][1] += a * b / n; C[1][1] += b * b / n; }
        C[1][0] = C[0][1];
        eig = S.eigSym(C);
      }
      function loss() { let s = 0; for (const x of X) { const c = B[0] * x[0] + B[1] * x[1]; s += (x[0] - A[0] * c) ** 2 + (x[1] - A[1] * c) ** 2; } return s / X.length; }
      function reset() {
        loop.stop(); genData();
        const r = ctx.rng; A = [r.normal(0, 0.3), r.normal(0, 0.3)]; B = tied.get() ? A.slice() : [r.normal(0, 0.3), r.normal(0, 0.3)];
        // start pointing away from PC1 so there is something to watch
        const u = [eig.vectors[0][1], eig.vectors[1][1]];
        A = [0.5 * u[0] + A[0] * 0.3, 0.5 * u[1] + A[1] * 0.3]; if (tied.get()) B = A.slice();
        hist = [loss()]; it = 0; draw();
      }
      function gdStep() {
        const n = X.length, eta = lr.get();
        let gA0 = 0, gA1 = 0, gB0 = 0, gB1 = 0;
        for (const x of X) {
          const c = B[0] * x[0] + B[1] * x[1], r0 = x[0] - A[0] * c, r1 = x[1] - A[1] * c, ar = A[0] * r0 + A[1] * r1;
          gA0 += -2 * r0 * c / n; gA1 += -2 * r1 * c / n; gB0 += -2 * ar * x[0] / n; gB1 += -2 * ar * x[1] / n;
        }
        if (tied.get()) { A = [A[0] - eta * (gA0 + gB0), A[1] - eta * (gA1 + gB1)]; B = A.slice(); }
        else { A = [A[0] - eta * gA0, A[1] - eta * gA1]; B = [B[0] - eta * gB0, B[1] - eta * gB1]; }
        if (!A.every(Number.isFinite) || !B.every(Number.isFinite) || Math.hypot(...A) > 1e6) { A = [0.1, 0.1]; B = [0.1, 0.1]; }
        it++; hist.push(loss());
      }
      function step() { for (let k = 0; k < 4; k++) gdStep(); draw(); if (it >= 4000) return false; }
      const ang = (u, v) => { const c = Math.abs(u[0] * v[0] + u[1] * v[1]) / (Math.hypot(...u) * Math.hypot(...v) || 1); return Math.acos(clamp(c, 0, 1)) * 180 / Math.PI; };
      function draw() {
        const u = [eig.vectors[0][0], eig.vectors[1][0]], l2 = eig.values[1];
        pD.clear();
        pD.line([[-6 * u[0], -6 * u[1]], [6 * u[0], 6 * u[1]]], { color: 'ink', width: 1.4, dash: '6 4', opacity: 0.7 });
        const an = Math.hypot(...A) || 1, ad = [A[0] / an, A[1] / an];
        pD.line([[-6 * ad[0], -6 * ad[1]], [6 * ad[0], 6 * ad[1]]], { color: 's2', width: 2.2 });
        const R = X.map(x => { const c = B[0] * x[0] + B[1] * x[1]; return [A[0] * c, A[1] * c]; });
        X.forEach((x, i) => { if (i % 2 === 0) pD.segment(x[0], x[1], R[i][0], R[i][1], { color: 'faint', width: 1 }); });
        pD.points(X, { color: 's1', r: 2.6, opacity: 0.7 });
        pD.points(R, { color: 's2', r: 2.2, opacity: 0.85 });
        const bn = Math.hypot(...B) || 1;
        pD.arrow(0, 0, 2.2 * B[0] / bn, 2.2 * B[1] / bn, { color: 's3', width: 2 });
        pD.legend([{ label: 'PC1', color: 'ink', dash: true }, { label: 'decoder span(A)', color: 's2' }, { label: 'encoder direction Bᵀ', color: 's3' }, { label: 'x̂ = ABx', color: 's2', fill: true }], { left: true });
        const N = Math.max(400, hist.length);
        const ymax = Math.max(hist[0], l2 * 1.5, 0.1) * 1.05;
        pL.setDomain([0, N], [0, ymax]).clear();
        pL.hline(l2, { color: 'good', dash: '5 4' });
        pL.text(N * 0.98, l2, 'λ₂ (optimal)', { anchor: 'end', dy: -5, size: 11, color: 'good' });
        pL.line(hist.map((v, i) => [i, v]), { color: 's1', width: 2 });
        const L = hist[hist.length - 1];
        sLoss.set(`${S.fmt(L, 3)} / ${S.fmt(l2, 3)}`, `iteration ${it}`);
        sAng.set(ang(A, u).toFixed(2) + '°');
        sEnc.set(ang(A, B).toFixed(2) + '°', tied.get() ? 'tied: 0 by construction' : 'oblique projection until this → 0');
        const M = [[A[0] * B[0], A[0] * B[1]], [A[1] * B[0], A[1] * B[1]]], P = [[u[0] * u[0], u[0] * u[1]], [u[1] * u[0], u[1] * u[1]]];
        sProj.set(S.fmt(Math.sqrt((M[0][0] - P[0][0]) ** 2 + (M[0][1] - P[0][1]) ** 2 + (M[1][0] - P[1][0]) ** 2 + (M[1][1] - P[1][1]) ** 2), 3), 'AB → projection onto PC1');
      }
      reset();
    },
  });

  Atlas.lab('autoencoders-vae-elbo', {
    title: 'The ELBO and two gradient estimators, on an exactly solvable model',
    blurb: 'Model: $z\\sim\\N(0,1)$, $x\\mid z\\sim\\N(az,\\sigma^2)$, one observed $x$. Variational $q(z)=\\N(m,s^2)$. Left: prior, exact posterior and $q$; the gap $\\log p(x)-\\mathrm{ELBO}$ equals $\\KL(q\\Vert p(z\\mid x))$. Right: 1500 replicates of the reparameterization and score-function estimates of $\\partial\\,\\mathrm{ELBO}/\\partial m$, each averaging $S$ samples. Shrink $s$ or move $m$ away from 0 and watch the score-function spread explode.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const xS = UI.slider(controls, { label: 'Observed $x$', min: -4, max: 4, step: 0.05, value: 2, onInput: draw });
      const aS = UI.slider(controls, { label: 'Decoder weight $a$', min: 0, max: 3, step: 0.05, value: 1.5, onInput: draw });
      const sgS = UI.slider(controls, { label: 'Decoder noise $\\sigma$', min: 0.2, max: 2, step: 0.05, value: 0.7, onInput: draw });
      UI.divider(controls, 'Variational q');
      const mS = UI.slider(controls, { label: 'Mean $m$', min: -3, max: 3, step: 0.02, value: 0.3, onInput: draw });
      const sS = UI.slider(controls, { label: 'Std. dev. $s$', min: 0.03, max: 2, value: 0.6, log: true, fmt: v => v.toFixed(3), onInput: draw });
      const nS = UI.slider(controls, { label: 'MC samples per estimate $S$', min: 1, max: 64, step: 1, value: 1, onChange: draw });
      const base = UI.toggle(controls, { label: 'Baseline $F(m)$ for score function', value: false, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      UI.button(row, 'Set q = posterior', () => { const { pm, ps } = post(); mS.set(pm); sS.set(clamp(ps, 0.03, 2)); draw(); }, { primary: true });
      UI.button(row, 'Resample', draw);
      stage.classList.add('two');
      const pZ = new Plot(stage, { width: 420, height: 280, x: [-4, 4], y: [0, 1], xlabel: 'z', ylabel: 'density', title: 'Prior, exact posterior and q' });
      const pG = new Plot(stage, { width: 420, height: 280, x: [-5, 5], y: [0, 1], xlabel: '∂ELBO/∂m estimate', ylabel: 'density', title: 'Gradient estimators (1500 replicates)' });
      const sLp = UI.stat(stats, '$\\log p(x)$');
      const sEl = UI.stat(stats, 'ELBO = recon − KL(q‖prior)', { accent: 'accent' });
      const sGap = UI.stat(stats, 'Gap = KL(q‖posterior)');
      const sRep = UI.stat(stats, 'sd: reparameterization');
      const sSf = UI.stat(stats, 'sd: score function');
      function post() { const a = aS.get(), sg = sgS.get(), x = xS.get(), D = a * a + sg * sg; return { pm: a * x / D, ps: sg / Math.sqrt(D), D }; }
      function draw() {
        const a = aS.get(), sg = sgS.get(), x = xS.get(), m = mS.get(), s = sS.get(), SS = Math.round(nS.get());
        const { pm, ps, D } = post();
        const logpx = -0.5 * Math.log(2 * Math.PI * D) - x * x / (2 * D);
        const recon = -0.5 * Math.log(2 * Math.PI * sg * sg) - ((x - a * m) ** 2 + a * a * s * s) / (2 * sg * sg);
        const klp = 0.5 * (m * m + s * s - 1 - Math.log(s * s));
        const elbo = recon - klp;
        const gap = Math.log(ps / s) + (s * s + (m - pm) ** 2) / (2 * ps * ps) - 0.5;
        const gTrue = a * (x - a * m) / (sg * sg) - m;
        // left
        const zs = S.linspace(-4, 4, 300);
        const qd = zs.map(z => S.pdf.normal(z, m, s)), pd = zs.map(z => S.pdf.normal(z, pm, ps));
        const ymax = Math.min(8, Math.max(...qd, ...pd, 0.45) * 1.1);
        pZ.setDomain([-4, 4], [0, ymax]).clear();
        pZ.line(zs.map(z => [z, S.pdf.normal(z)]), { color: 'muted', width: 1.6, dash: '5 4' });
        pZ.area(zs.map((z, i) => [z, qd[i]]), { color: 's1', opacity: 0.2 });
        pZ.line(zs.map((z, i) => [z, qd[i]]), { color: 's1', width: 2 });
        pZ.line(zs.map((z, i) => [z, pd[i]]), { color: 'ink', width: 2 });
        pZ.legend([{ label: 'prior p(z)', color: 'muted', dash: true }, { label: 'posterior p(z|x)', color: 'ink' }, { label: 'q(z)', color: 's1' }]);
        // right: estimator replicates
        const F = z => -0.5 * Math.log(2 * Math.PI * sg * sg) - (x - a * z) ** 2 / (2 * sg * sg);
        const Fp = z => a * (x - a * z) / (sg * sg);
        const b0 = base.get() ? F(m) : 0;
        const R = 1500, rep = new Array(R), sf = new Array(R);
        for (let k = 0; k < R; k++) {
          let gr = 0, gs = 0;
          for (let j = 0; j < SS; j++) { const e = ctx.rng.normal(), z = m + s * e; gr += Fp(z); gs += (F(z) - b0) * e / s; }
          rep[k] = gr / SS - m; sf[k] = gs / SS - m;
        }
        const all = rep.concat(sf).sort((u, v) => u - v);
        let lo = Math.min(S.quantile(all, 0.01, true), gTrue - 0.5), hi = Math.max(S.quantile(all, 0.99, true), gTrue + 0.5);
        const pad = (hi - lo) * 0.05; lo -= pad; hi += pad;
        const hR = S.histogram(rep.filter(v => v >= lo && v <= hi), { bins: 60, lo, hi }), hS = S.histogram(sf.filter(v => v >= lo && v <= hi), { bins: 60, lo, hi });
        for (const h of [hR, hS]) for (const b of h) b.density = b.count / (R * (b.x1 - b.x0));
        const ym = Math.max(...hR.map(b => b.density), ...hS.map(b => b.density)) * 1.1;
        pG.setDomain([lo, hi], [0, ym || 1]).clear();
        pG.bars(hS, { color: 's4', opacity: 0.45 });
        pG.bars(hR, { color: 's2', opacity: 0.6 });
        pG.vline(gTrue, { color: 'ink', dash: false, width: 2 });
        pG.legend([{ label: 'reparameterization', color: 's2', fill: true }, { label: 'score function', color: 's4', fill: true }, { label: 'true gradient', color: 'ink' }]);
        sLp.set(S.fmt(logpx, 4));
        sEl.set(S.fmt(elbo, 4), `recon ${S.fmt(recon, 3)}, KL ${S.fmt(klp, 3)}`);
        sGap.set(S.fmt(gap, 4), `check: log p − ELBO = ${S.fmt(logpx - elbo, 4)}`);
        const sdR = S.sd(rep), sdS = S.sd(sf);
        sRep.set(S.fmt(sdR, 3), `theory ${S.fmt(a * a * s / (sg * sg) / Math.sqrt(SS), 3)}`);
        sSf.set(S.fmt(sdS, 3), `${sdR > 0 ? S.fmt(sdS / sdR, 3) + '× larger' : 'reparam. has zero variance'}`);
      }
      draw();
    },
  });

  /* ================================================================== */
  /*  GAN labs                                                           */
  /* ================================================================== */
  // Tiny 1 -> H -> 1 tanh MLP with flat parameters [w1(H), b1(H), w2(H), b2].
  function mlpInit(r, H, s1, s2) { const P = new Float64Array(3 * H + 1); for (let j = 0; j < H; j++) { P[j] = r.normal(0, s1); P[H + j] = r.normal(0, s1); P[2 * H + j] = r.normal(0, s2); } return P; }
  function mlpF(P, H, x, hb) { let o = P[3 * H]; for (let j = 0; j < H; j++) { const h = Math.tanh(P[j] * x + P[H + j]); hb[j] = h; o += P[2 * H + j] * h; } return o; }
  function mlpB(P, H, x, hb, g, G) { // accumulate dL/dP given dL/dout = g; return dL/dx
    G[3 * H] += g; let dx = 0;
    for (let j = 0; j < H; j++) { const h = hb[j]; G[2 * H + j] += g * h; const dp = g * P[2 * H + j] * (1 - h * h); G[j] += dp * x; G[H + j] += dp; dx += dp * P[j]; }
    return dx;
  }
  function mlpDx(P, H, hb) { let dx = 0; for (let j = 0; j < H; j++) dx += P[2 * H + j] * (1 - hb[j] * hb[j]) * P[j]; return dx; }
  function adam(P) { return { P, m: new Float64Array(P.length), v: new Float64Array(P.length), t: 0 }; }
  function adamStep(o, G, lr, b1 = 0.5, b2 = 0.999) {
    o.t++; const c1 = 1 - b1 ** o.t, c2 = 1 - b2 ** o.t;
    for (let i = 0; i < G.length; i++) { o.m[i] = b1 * o.m[i] + (1 - b1) * G[i]; o.v[i] = b2 * o.v[i] + (1 - b2) * G[i] * G[i]; o.P[i] -= lr * (o.m[i] / c1) / (Math.sqrt(o.v[i] / c2) + 1e-8); }
  }
  const GAN_DATA = {
    two: { label: '2 modes', mu: [-2, 2], w: [0.5, 0.5] },
    three: { label: '3 modes', mu: [-3, 0, 3], w: [1 / 3, 1 / 3, 1 / 3] },
    four: { label: '4 modes', mu: [-3, -1, 1, 3], w: [0.25, 0.25, 0.25, 0.25] },
    uneq: { label: '3 unequal modes', mu: [-2.5, 0, 2.5], w: [0.6, 0.3, 0.1] },
  };
  const GSD = 0.25;

  Atlas.lab('gans-training', {
    title: 'Training a 1-D GAN: coverage, collapse and the optimal discriminator',
    blurb: 'Generator and discriminator are small tanh networks trained by alternating Adam steps with hand-written backpropagation. Left: data density (line) vs a histogram of generator samples. Right: the current $D(x)$ against $D^*=p_{\\text{data}}/(p_{\\text{data}}+p_g)$. Several modes and the saturating loss often end in collapse onto one or two modes; compare the non-saturating and WGAN (weight-clipped critic) losses, more critic steps, and different seeds.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const data = UI.select(controls, { label: 'Data', options: Object.keys(GAN_DATA).map(k => ({ value: k, label: GAN_DATA[k].label })), value: 'three', onChange: reset });
      const lossS = UI.select(controls, { label: 'Generator / critic loss', options: [{ value: 'ns', label: 'Non-saturating' }, { value: 'mm', label: 'Minimax (saturating)' }, { value: 'w', label: 'WGAN, clipped critic' }], value: 'ns', onChange: reset });
      const kS = UI.slider(controls, { label: 'Critic steps per G step $k$', min: 1, max: 5, step: 1, value: 1 });
      const lrS = UI.slider(controls, { label: 'Learning rate', min: 0.0003, max: 0.01, value: 0.002, log: true, fmt: v => v.toFixed(4) });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(step);
      ctx.playButton(row, loop, ['▶ Train', '❚❚ Pause']);
      UI.button(row, 'Reset (new seed)', () => { seed++; reset(); });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, x: [-5, 5], y: [0, 1], xlabel: 'x', ylabel: 'density', title: 'Data vs generator' });
      const pR = new Plot(stage, { width: 420, height: 280, x: [-5, 5], y: [0, 1], xlabel: 'x', ylabel: 'D(x)', title: 'Discriminator vs D*' });
      const sIt = UI.stat(stats, 'Generator steps');
      const sMode = UI.stat(stats, 'Modes covered', { accent: 'accent' });
      const sJS = UI.stat(stats, 'JS(data, gen) (binned)');
      const sW = UI.stat(stats, '$W_1$(data, gen)');
      const H = 16, B = 64, NB = 100, LO = -5, HI = 5;
      let seed = 1, G, D, it, ref, view;
      const hG = new Float64Array(H), hD = new Float64Array(H);
      const dataDens = (x, cfg) => cfg.w.reduce((s, w, i) => s + w * S.pdf.normal(x, cfg.mu[i], GSD), 0);
      const sampleData = (r, cfg) => cfg.mu[r.categorical(cfg.w)] + GSD * r.normal();
      function reset() {
        loop.stop();
        const r = S.rng(1000 + seed);
        G = adam(mlpInit(r, H, 1.0, 0.3)); D = adam(mlpInit(r, H, 1.0, 0.5));
        it = 0;
        const cfg = GAN_DATA[data.get()];
        const rr = S.rng(5); ref = rr.array(1500, () => sampleData(rr, cfg)).sort((a, b) => a - b);
        view = null; draw();
      }
      function trainIter() {
        const cfg = GAN_DATA[data.get()], r = ctx.rng, lr = lrS.get(), wg = lossS.get() === 'w';
        const GD = new Float64Array(D.P.length), GG = new Float64Array(G.P.length);
        for (let k = 0; k < kS.get(); k++) {
          GD.fill(0);
          for (let b = 0; b < B; b++) {
            const xr = sampleData(r, cfg), ar = mlpF(D.P, H, xr, hD);
            mlpB(D.P, H, xr, hD, (wg ? -1 : -(1 - sigm(ar))) / B, GD);
            const xf = mlpF(G.P, H, r.normal(), hG), af = mlpF(D.P, H, xf, hD);
            mlpB(D.P, H, xf, hD, (wg ? 1 : sigm(af)) / B, GD);
          }
          adamStep(D, GD, lr);
          if (wg) for (let i = 0; i < D.P.length; i++) D.P[i] = clamp(D.P[i], -0.5, 0.5);
        }
        for (let b = 0; b < B; b++) {
          const z = r.normal(), x = mlpF(G.P, H, z, hG), a = mlpF(D.P, H, x, hD);
          const L = lossS.get();
          const ga = L === 'w' ? -1 : L === 'mm' ? -sigm(a) : -(1 - sigm(a)); // dLoss_G / d a  (G minimizes)
          mlpB(G.P, H, z, hG, ga * mlpDx(D.P, H, hD) / B, GG);
        }
        adamStep(G, GG, lr);
        it++;
      }
      function step() {
        for (let i = 0; i < 6; i++) trainIter();
        draw();
        if (it >= 6000) return false;
      }
      function draw() {
        const cfg = GAN_DATA[data.get()], r = S.rng(77 + it);
        const gs = r.array(2000, () => mlpF(G.P, H, r.normal(), hG));
        const hist = S.histogram(gs.filter(v => v >= LO && v < HI), { bins: NB, lo: LO, hi: HI });
        for (const b of hist) b.density = b.count / (gs.length * (b.x1 - b.x0));
        const grid = S.linspace(LO, HI, 300), dd = grid.map(x => dataDens(x, cfg));
        const ym = Math.min(3.5, Math.max(...dd, ...hist.map(b => b.density)) * 1.1);
        pL.setDomain([LO, HI], [0, ym]).clear();
        pL.bars(hist, { color: 's2', opacity: 0.55 });
        pL.line(grid.map((x, i) => [x, dd[i]]), { color: 'ink', width: 2 });
        pL.legend([{ label: 'p_data', color: 'ink' }, { label: 'generator samples', color: 's2', fill: true }]);
        // right: D and D*
        pR.setDomain([LO, HI], [0, 1]).clear();
        const wg = lossS.get() === 'w';
        const dv = grid.map(x => mlpF(D.P, H, x, hD));
        if (wg) {
          const [lo, hi] = S.extent(dv), sp = hi - lo || 1;
          pR.line(grid.map((x, i) => [x, (dv[i] - lo) / sp]), { color: 's1', width: 2.2 });
        } else pR.line(grid.map((x, i) => [x, sigm(dv[i])]), { color: 's1', width: 2.2 });
        const dstar = hist.map(b => { const xc = (b.x0 + b.x1) / 2, pd = dataDens(xc, cfg), pg = b.density; return (pd + pg) > 1e-3 ? [xc, pd / (pd + pg)] : null; });
        pR.line(dstar, { color: 's3', width: 1.6, dash: '4 3' });
        pR.hline(0.5, { color: 'muted' });
        pR.legend([{ label: wg ? 'critic f(x), rescaled to [0,1]' : 'D(x)', color: 's1' }, { label: 'D* = p_d/(p_d+p_g)', color: 's3', dash: true }]);
        // stats
        let covered = 0;
        cfg.mu.forEach((mu, i) => { const mass = gs.filter(v => Math.abs(v - mu) < 3 * GSD).length / gs.length; if (mass >= 0.3 * cfg.w[i]) covered++; });
        let js = 0;
        for (const b of hist) {
          const w = b.x1 - b.x0, P = dataDens((b.x0 + b.x1) / 2, cfg) * w, Q = b.density * w, M = (P + Q) / 2;
          if (P > 0 && M > 0) js += 0.5 * P * Math.log(P / M);
          if (Q > 0 && M > 0) js += 0.5 * Q * Math.log(Q / M);
        }
        const gsS = gs.slice(0, ref.length).sort((a, b) => a - b);
        let w1 = 0; for (let i = 0; i < ref.length; i++) w1 += Math.abs(ref[i] - gsS[i]) / ref.length;
        sIt.set(String(it));
        sMode.set(`${covered} / ${cfg.mu.length}`, 'mode counted if it gets ≥ 30% of its share');
        sJS.set(S.fmt(js, 3), 'max log 2 ≈ 0.693');
        sW.set(S.fmt(w1, 3), 'from sorted samples (quantile coupling)');
      }
      reset();
    },
  });

  Atlas.lab('gans-divergences', {
    title: 'Jensen–Shannon saturates; Wasserstein does not',
    blurb: 'The data are fixed at 0 and the "generator" is the same shape shifted by $\\theta$. Left: both densities and the optimal discriminator $D^*$. Right: $\\mathrm{JS}$ and $W_1=|\\theta|$ as functions of the shift, with the current $\\theta$ marked. Make the distributions narrow: JS sits flat at $\\log 2$ (zero gradient) almost everywhere, while $W_1$ always points home. With uniform shapes and disjoint supports the plateau is exact.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const th = UI.slider(controls, { label: 'Generator shift $\\theta$', min: -4, max: 4, step: 0.01, value: 1.5, onInput: drawL });
      const wd = UI.slider(controls, { label: 'Width (sd) $s$', min: 0.05, max: 1.5, value: 0.4, log: true, fmt: v => v.toFixed(3), onInput: () => { curve = null; draw(); } });
      const fam = UI.segmented(controls, { label: 'Shape', options: [{ value: 'gauss', label: 'Gaussian' }, { value: 'unif', label: 'Uniform' }], value: 'gauss', onChange: () => { curve = null; draw(); } });
      stage.classList.add('two');
      const pL = new Plot(stage, { width: 420, height: 280, x: [-5, 5], y: [0, 1], xlabel: 'x', ylabel: 'density', title: 'p_data, p_g and D* (scaled)' });
      const pR = new Plot(stage, { width: 420, height: 280, x: [-4, 4], y: [0, 4.2], xlabel: 'shift θ', ylabel: 'divergence', title: 'JS vs W₁ as the generator moves' });
      const sJS = UI.stat(stats, '$\\mathrm{JS}(p_{\\text{data}}\\Vert p_g)$', { accent: 'accent' });
      const sV = UI.stat(stats, '$V(D^*,G) = 2\\,\\mathrm{JS}-\\log 4$');
      const sG = UI.stat(stats, '$d\\,\\mathrm{JS}/d\\theta$');
      const sW = UI.stat(stats, '$W_1$ and its slope');
      let curve = null;
      const dens = (x, mu, s) => (fam.get() === 'gauss' ? S.pdf.normal(x, mu, s) : (Math.abs(x - mu) <= s * Math.sqrt(3) ? 1 / (2 * s * Math.sqrt(3)) : 0));
      function JS(t) {
        const s = wd.get(), lo = Math.min(0, t) - 7 * s, hi = Math.max(0, t) + 7 * s, N = 3000, h = (hi - lo) / N;
        let js = 0;
        for (let i = 0; i < N; i++) {
          const x = lo + (i + 0.5) * h, p = dens(x, 0, s), q = dens(x, t, s), m = (p + q) / 2;
          if (p > 0) js += 0.5 * p * Math.log(p / m) * h;
          if (q > 0) js += 0.5 * q * Math.log(q / m) * h;
        }
        return clamp(js, 0, Math.LN2);
      }
      function draw() { if (!curve) curve = S.linspace(-4, 4, 161).map(t => [t, JS(t)]); drawL(); }
      function drawL() {
        if (!curve) curve = S.linspace(-4, 4, 161).map(t => [t, JS(t)]);
        const t = th.get(), s = wd.get();
        const xs = S.linspace(-5, 5, 800), pd = xs.map(x => dens(x, 0, s)), pg = xs.map(x => dens(x, t, s));
        const ym = Math.min(9, Math.max(...pd, ...pg)) * 1.12;
        pL.setDomain([-5, 5], [0, ym]).clear();
        pL.area(xs.map((x, i) => [x, pd[i]]), { color: 'ink', opacity: 0.12 });
        pL.line(xs.map((x, i) => [x, pd[i]]), { color: 'ink', width: 2 });
        pL.area(xs.map((x, i) => [x, pg[i]]), { color: 's2', opacity: 0.18 });
        pL.line(xs.map((x, i) => [x, pg[i]]), { color: 's2', width: 2 });
        pL.line(xs.map((x, i) => (pd[i] + pg[i] > 1e-6 ? [x, ym * 0.9 * pd[i] / (pd[i] + pg[i])] : null)), { color: 's3', width: 1.6, dash: '4 3' });
        pL.legend([{ label: 'p_data', color: 'ink' }, { label: 'p_g', color: 's2' }, { label: 'D* (×0.9 of panel height)', color: 's3', dash: true }]);
        pR.clear();
        pR.line(S.linspace(-4, 4, 3).map(v => [v, Math.abs(v)]), { color: 's1', width: 2.2 });
        pR.line(curve, { color: 's4', width: 2.4 });
        pR.hline(Math.LN2, { color: 'muted' });
        pR.text(3.95, Math.LN2, 'log 2', { anchor: 'end', dy: -5, size: 11, color: 'muted' });
        const j = JS(t);
        pR.dot(t, j, { color: 's4', r: 5 });
        pR.dot(t, Math.abs(t), { color: 's1', r: 5 });
        pR.legend([{ label: 'W₁ = |θ|', color: 's1' }, { label: 'JS', color: 's4' }], { left: true });
        const e = 0.01, dj = (JS(t + e) - JS(t - e)) / (2 * e);
        sJS.set(S.fmt(j, 4), `${(100 * j / Math.LN2).toFixed(1)}% of log 2`);
        sV.set(S.fmt(2 * j - Math.log(4), 4), '0 = perfect discriminator');
        sG.set(S.fmt(dj, 3), Math.abs(dj) < 1e-3 ? 'flat: no signal for the generator' : 'gradient available');
        sW.set(`${S.fmt(Math.abs(t), 3)}, slope ${t === 0 ? '0' : (t > 0 ? '+1' : '−1')}`);
      }
      draw();
    },
  });

  /* ================================================================== */
  /*  Diffusion labs (exact score of a noised Gaussian mixture)          */
  /* ================================================================== */
  const B0 = 0.1, B1 = 20, TMIN = 1e-3;
  const betaT = t => B0 + (B1 - B0) * t;
  const abarT = t => Math.exp(-(B0 * t + 0.5 * (B1 - B0) * t * t));
  function mixtures() {
    const ring = S.range(8).map(k => ({ mu: [2 * Math.cos(2 * Math.PI * k / 8), 2 * Math.sin(2 * Math.PI * k / 8)], s: 0.15, w: 1 / 8, g: k }));
    const blobs = [{ mu: [-1.5, -1], s: 0.35, w: 0.6, g: 0 }, { mu: [1.6, -0.6], s: 0.25, w: 0.3, g: 1 }, { mu: [0.1, 1.8], s: 0.2, w: 0.1, g: 2 }];
    const moons = [];
    for (let k = 0; k < 12; k++) { const a = Math.PI * k / 11; moons.push({ mu: [1.6 * Math.cos(a) - 0.8, 1.6 * Math.sin(a) - 0.3], s: 0.12, w: 1 / 24, g: 0 }); moons.push({ mu: [0.8 - 1.6 * Math.cos(a), 0.3 - 1.6 * Math.sin(a)], s: 0.12, w: 1 / 24, g: 1 }); }
    return {
      ring: { label: 'Ring of 8', comps: ring, groups: 8, gw: Array(8).fill(1 / 8) },
      blobs: { label: 'Unequal blobs (0.6 / 0.3 / 0.1)', comps: blobs, groups: 3, gw: [0.6, 0.3, 0.1] },
      moons: { label: 'Two moons', comps: moons, groups: 2, gw: [0.5, 0.5] },
    };
  }
  const MIX = mixtures();
  // Score of p_t and posterior mean E[x0 | x_t] for the noised mixture.
  function scoreAt(mix, x, y, t, out) {
    const ab = abarT(t), a = Math.sqrt(ab), C = mix.comps;
    let mx = -Infinity;
    const lw = new Float64Array(C.length), V = new Float64Array(C.length);
    for (let k = 0; k < C.length; k++) {
      const v = ab * C[k].s * C[k].s + 1 - ab, dx = x - a * C[k].mu[0], dy = y - a * C[k].mu[1];
      V[k] = v; lw[k] = Math.log(C[k].w) - Math.log(2 * Math.PI * v) - (dx * dx + dy * dy) / (2 * v);
      if (lw[k] > mx) mx = lw[k];
    }
    let Z = 0; for (let k = 0; k < C.length; k++) { lw[k] = Math.exp(lw[k] - mx); Z += lw[k]; }
    let sx = 0, sy = 0, ex = 0, ey = 0;
    for (let k = 0; k < C.length; k++) {
      const r = lw[k] / Z, v = V[k], c = C[k];
      sx += r * (a * c.mu[0] - x) / v; sy += r * (a * c.mu[1] - y) / v;
      const gain = a * c.s * c.s / v;
      ex += r * (c.mu[0] + gain * (x - a * c.mu[0])); ey += r * (c.mu[1] + gain * (y - a * c.mu[1]));
    }
    if (out) { out.logp = mx + Math.log(Z); out.ex = ex; out.ey = ey; }
    return [sx, sy];
  }
  function densRaster(p, mix, t) {
    const bg = Atlas.rgb('surface'), fg = Atlas.rgb('s5'), o = {};
    const nx = 96, ny = 72, vals = new Float64Array(nx * ny);
    // first pass for max (sample the same grid the raster uses)
    let mx = -Infinity, k = 0;
    const xs = [], ys = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const x = p.ix(p.m.l + (i + 0.5) / nx * p.iw), y = p.iy(p.m.t + (j + 0.5) / ny * p.ih);
      scoreAt(mix, x, y, t, o); vals[k] = o.logp; if (o.logp > mx) mx = o.logp; k++;
    }
    let idx = 0;
    p.raster(() => { const v = Math.exp(vals[idx++] - mx); return Atlas.mix(bg, fg, Math.pow(v, 0.55) * 0.8).concat([255]); }, { nx, ny });
  }
  function sampleMix(r, mix) { const c = mix.comps[r.categorical(mix.comps.map(c => c.w))]; return [c.mu[0] + c.s * r.normal(), c.mu[1] + c.s * r.normal()]; }

  Atlas.lab('diffusion-models-forward', {
    title: 'The forward process, the score field and Tweedie denoising',
    blurb: 'Data from a 2-D Gaussian mixture are noised by the VP-SDE: $\\mathbf x_t=\\sqrt{\\bar\\alpha_t}\\mathbf x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\boldsymbol\\epsilon$ (each particle keeps its own $\\boldsymbol\\epsilon$). Shading is the exact marginal $p_t$; arrows are its score $\\nabla\\log p_t$. Click anywhere to place a noisy observation $\\mathbf x_t$: the arrow shows the Tweedie/posterior-mean denoised point $\\E[\\mathbf x_0\\mid\\mathbf x_t]$. At large $t$ it collapses to the data mean; at small $t$ it snaps to the nearest mode.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data distribution', options: Object.keys(MIX).map(k => ({ value: k, label: MIX[k].label })), value: 'ring', onChange: resample });
      const tS = UI.slider(controls, { label: 'Diffusion time $t$', min: 0, max: 1, step: 0.005, value: 0.15, onInput: draw });
      const arrows = UI.toggle(controls, { label: 'Show score field', value: true, onChange: draw });
      const shade = UI.toggle(controls, { label: 'Shade density $p_t$', value: true, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      let dir = 1;
      const loop = ctx.loop(dt => { let t = tS.get() + dir * dt * 0.25; if (t >= 1) { t = 1; dir = -1; } if (t <= 0) { t = 0; dir = 1; } tS.set(t); draw(); });
      ctx.playButton(row, loop, ['▶ Animate t', '❚❚ Pause']);
      UI.button(row, 'Resample', resample);
      const p = new Plot(stage, { width: 640, height: 400, x: [-3.6, 3.6], y: [-3.2, 3.2], equal: true, xlabel: 'x₁', ylabel: 'x₂', title: '' });
      const sA = UI.stat(stats, '$\\bar\\alpha(t)$', { accent: 'accent' });
      const sSNR = UI.stat(stats, 'SNR $=\\bar\\alpha/(1-\\bar\\alpha)$');
      const sQ = UI.stat(stats, 'Query $\\mathbf x_t$ → $\\E[\\mathbf x_0\\mid\\mathbf x_t]$');
      const sS = UI.stat(stats, '$\\lVert\\nabla\\log p_t(\\mathbf x_t)\\rVert$');
      let x0 = [], eps = [], q = [1.2, 0.4];
      function resample() { const mix = MIX[ds.get()]; x0 = ctx.rng.array(400, () => sampleMix(ctx.rng, mix)); eps = ctx.rng.array(400, () => [ctx.rng.normal(), ctx.rng.normal()]); draw(); }
      function draw() {
        const mix = MIX[ds.get()], t = Math.max(tS.get(), 1e-4), ab = abarT(t), a = Math.sqrt(ab), sg = Math.sqrt(1 - ab);
        p.clear();
        if (shade.get()) densRaster(p, mix, t);
        p.points(x0.map((x, i) => [a * x[0] + sg * eps[i][0], a * x[1] + sg * eps[i][1]]), { color: 's1', r: 2.2, opacity: 0.6 });
        if (arrows.get()) {
          const gx = S.linspace(-3.3, 3.3, 15), gy = S.linspace(-2.9, 2.9, 12);
          for (const x of gx) for (const y of gy) {
            const [u, v] = scoreAt(mix, x, y, t), m = Math.hypot(u, v);
            if (m < 1e-9) continue;
            const L = 0.36 * Math.min(1, m / 3) + 0.06;
            p.arrow(x, y, x + L * u / m, y + L * v / m, { color: 'muted', width: 1.1, opacity: 0.8 });
          }
        }
        const o = {}; const sc = scoreAt(mix, q[0], q[1], t, o);
        p.dot(q[0], q[1], { color: 's3', r: 6 });
        p.arrow(q[0], q[1], o.ex, o.ey, { color: 's3', width: 2.2 });
        p.dot(o.ex, o.ey, { color: 's3', r: 4, opacity: 0.9 });
        sA.set(S.fmt(ab, 4), `t = ${t.toFixed(3)}, noise sd √(1−ᾱ) = ${S.fmt(sg, 3)}`);
        sSNR.set(ab < 1 ? S.fmt(ab / (1 - ab), 3) : '∞');
        sQ.set(`(${S.fmt(o.ex, 2)}, ${S.fmt(o.ey, 2)})`, `from (${S.fmt(q[0], 2)}, ${S.fmt(q[1], 2)}); equals (x + (1−ᾱ)∇log p)/√ᾱ`);
        sS.set(S.fmt(Math.hypot(...sc), 3));
      }
      p.onPointer((x, y, e, type) => { if (type === 'down' || type === 'drag') { q = [x, y]; draw(); } });
      resample();
    },
  });

  Atlas.lab('diffusion-models-reverse', {
    title: 'Sampling backwards: reverse SDE vs probability-flow ODE',
    blurb: 'Particles start from $\\N(0,I)$ at $t=1$ and are integrated back to $t=10^{-3}$ using the <em>exact</em> score of the noised mixture, so all error is discretization error. The reverse SDE (Euler–Maruyama) injects noise and includes a Langevin-like correction; the probability-flow ODE (Euler) is deterministic — follow the highlighted trajectories. Cut the number of steps to 5–20 and compare the samplers; the sliced $W_1$ is shown next to the noise floor between two exact samples.',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);
      const ds = UI.select(controls, { label: 'Data distribution', options: Object.keys(MIX).map(k => ({ value: k, label: MIX[k].label })), value: 'blobs', onChange: restart });
      const samp = UI.segmented(controls, { label: 'Sampler', options: [{ value: 'sde', label: 'Reverse SDE' }, { value: 'ode', label: 'Prob.-flow ODE' }], value: 'sde', onChange: restart });
      const nSteps = UI.slider(controls, { label: 'Number of steps $N$', min: 2, max: 1000, value: 100, log: true, fmt: v => String(Math.round(v)), onChange: restart });
      const traj = UI.toggle(controls, { label: 'Show trajectories', value: true, onChange: draw });
      const row = UI.h('div', 'btn-row', controls);
      const loop = ctx.loop(stepFrame);
      ctx.playButton(row, loop, ['▶ Animate', '❚❚ Pause']);
      UI.button(row, 'New noise', () => { seedN++; restart(); });
      const p = new Plot(stage, { width: 640, height: 400, x: [-3.6, 3.6], y: [-3.2, 3.2], equal: true, xlabel: 'x₁', ylabel: 'x₂', title: '' });
      const sT = UI.stat(stats, 'Time $t$ / step', { accent: 'accent' });
      const sW = UI.stat(stats, 'Sliced $W_1$ to data');
      const sM = UI.stat(stats, 'Mode masses (truth)');
      const NP = 600, NT = 14;
      let seedN = 1, X, k, paths, refA, refB, floor = null, animating = false;
      function initState() {
        const r = S.rng(400 + seedN);
        X = r.array(NP, () => [r.normal(), r.normal()]);
        paths = X.slice(0, NT).map(x => [x.slice()]);
        k = 0;
      }
      function oneStep(r) {
        const mix = MIX[ds.get()], N = Math.round(nSteps.get()), dt = (1 - TMIN) / N, t = 1 - k * dt, b = betaT(t), sde = samp.get() === 'sde';
        for (let i = 0; i < NP; i++) {
          const x = X[i], [sx, sy] = scoreAt(mix, x[0], x[1], t);
          if (sde) { const g = Math.sqrt(b * dt); X[i] = [x[0] + (0.5 * b * x[0] + b * sx) * dt + g * r.normal(), x[1] + (0.5 * b * x[1] + b * sy) * dt + g * r.normal()]; }
          else X[i] = [x[0] + 0.5 * b * (x[0] + sx) * dt, x[1] + 0.5 * b * (x[1] + sy) * dt];
          if (i < NT) paths[i].push(X[i].slice());
        }
        k++;
      }
      let noiseR;
      function restart() {
        loop.stop(); animating = false; initState(); noiseR = S.rng(900 + seedN);
        const N = Math.round(nSteps.get());
        while (k < N) oneStep(noiseR);
        draw();
      }
      function stepFrame() {
        if (!animating) { initState(); noiseR = S.rng(900 + seedN); animating = true; }
        const N = Math.round(nSteps.get()), per = Math.max(1, Math.ceil(N / 120));
        for (let j = 0; j < per && k < N; j++) oneStep(noiseR);
        draw();
        if (k >= N) { animating = false; return false; }
      }
      function slicedW1(A, Bs) {
        let tot = 0; const M = 24;
        for (let d = 0; d < M; d++) {
          const th = Math.PI * d / M, c = Math.cos(th), s = Math.sin(th);
          const pa = A.map(x => x[0] * c + x[1] * s).sort((u, v) => u - v), pb = Bs.map(x => x[0] * c + x[1] * s).sort((u, v) => u - v);
          let w = 0; for (let i = 0; i < pa.length; i++) w += Math.abs(pa[i] - pb[i]); tot += w / pa.length;
        }
        return tot / M;
      }
      function draw() {
        const mix = MIX[ds.get()], N = Math.round(nSteps.get()), dt = (1 - TMIN) / N, t = Math.max(TMIN, 1 - k * dt);
        p.clear();
        densRaster(p, mix, t);
        if (traj.get()) paths.forEach(pt => p.line(pt, { color: 's2', width: 1.4, opacity: 0.85 }));
        p.points(X, { color: 's1', r: 2.1, opacity: 0.7 });
        if (traj.get()) paths.forEach(pt => p.dot(pt[pt.length - 1][0], pt[pt.length - 1][1], { color: 's2', r: 3.5 }));
        sT.set(`${t.toFixed(3)} / ${k} of ${N}`, 'shading = exact p_t at the current time');
        if (k >= N) {
          if (!floor || floor.key !== ds.get()) {
            const r = S.rng(3); refA = r.array(NP, () => sampleMix(r, mix)); refB = r.array(NP, () => sampleMix(r, mix));
            floor = { key: ds.get(), v: slicedW1(refA, refB) };
          }
          sW.set(S.fmt(slicedW1(X, refA), 3), `noise floor (two exact samples): ${S.fmt(floor.v, 3)}`);
          const cnt = new Array(mix.groups).fill(0);
          for (const x of X) { let best = 0, bd = Infinity; mix.comps.forEach((c, i) => { const d = (x[0] - c.mu[0]) ** 2 + (x[1] - c.mu[1]) ** 2; if (d < bd) { bd = d; best = i; } }); cnt[mix.comps[best].g]++; }
          if (mix.groups <= 3) sM.set(cnt.map(c => (c / NP).toFixed(2)).join(' / '), `truth ${mix.gw.map(w => w.toFixed(2)).join(' / ')}`);
          else sM.set(`${Math.min(...cnt) / NP < 0.01 ? 'some empty' : 'all present'}`, `per-mode range ${(Math.min(...cnt) / NP).toFixed(3)}–${(Math.max(...cnt) / NP).toFixed(3)} (truth 0.125)`);
        } else { sW.set('…', 'shown when sampling ends'); sM.set('…'); }
      }
      restart();
    },
  });
})();
