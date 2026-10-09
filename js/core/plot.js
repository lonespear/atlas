/* Atlas.Plot — a small SVG plotting surface that every lab draws on.
 *
 *   const p = new Atlas.Plot(el, { x: [-4, 4], y: [0, 0.5], xlabel: 'x', ylabel: 'density' });
 *   p.line(xs.map(x => [x, f(x)]), { color: 's1', width: 2 });
 *   p.bars(Atlas.S.histogram(data, { bins: 40 }), { color: 's2' });
 *   p.clear();   // wipes data marks, keeps axes; call before each redraw
 *
 * Colours are theme tokens ('s1'..'s8', 'ink', 'muted', 'faint', 'accent',
 * 'good', 'bad', 'warn') or any CSS colour. Tokens resolve through CSS
 * variables, so plots follow light/dark theme switches with no redraw.
 */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const TOKENS = new Set(['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 'ink', 'muted', 'faint', 'accent', 'good', 'bad', 'warn', 'grid', 'bg', 'surface']);
  const col = c => (c == null ? 'var(--s1)' : TOKENS.has(c) ? `var(--${c})` : c);

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // "Nice" tick values for an interval (Heckbert).
  function ticks(lo, hi, n = 6) {
    if (!(hi > lo)) return [lo];
    const span = hi - lo, raw = span / n, mag = 10 ** Math.floor(Math.log10(raw));
    const r = raw / mag, step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  function tickFmt(v, step) {
    const d = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
    if (Math.abs(v) >= 1e5 || (Math.abs(v) < 1e-3 && v !== 0)) return v.toExponential(1);
    return v.toFixed(Math.min(d, 6)).replace('-', '−');
  }

  class Plot {
    constructor(container, opts = {}) {
      this.o = Object.assign({ width: 640, height: 300, x: [0, 1], y: [0, 1], xlabel: '', ylabel: '',
        margin: null, grid: true, xticks: 6, yticks: 5, logx: false, logy: false, equal: false, title: '' }, opts);
      const m = this.o.margin || { t: this.o.title ? 28 : 14, r: 16, b: this.o.xlabel ? 44 : 30, l: this.o.ylabel ? 58 : 46 };
      this.m = m;
      this.svg = el('svg', { viewBox: `0 0 ${this.o.width} ${this.o.height}`, class: 'plot', role: 'img', preserveAspectRatio: 'xMidYMid meet' });
      if (this.o.ariaLabel) this.svg.setAttribute('aria-label', this.o.ariaLabel);
      container.appendChild(this.svg);
      this.axesG = el('g', { class: 'axes' }, this.svg);
      const clipId = 'clip' + Math.random().toString(36).slice(2, 9);
      const defs = el('defs', {}, this.svg);
      this.clipRect = el('rect', {}, el('clipPath', { id: clipId }, defs));
      this.defs = defs;
      this.layer = el('g', { 'clip-path': `url(#${clipId})`, class: 'marks' }, this.svg);
      this.overlay = el('g', { class: 'overlay' }, this.svg); // unclipped (labels, legends)
      this.setDomain(this.o.x, this.o.y);
    }

    get iw() { return this.o.width - this.m.l - this.m.r; }
    get ih() { return this.o.height - this.m.t - this.m.b; }

    setDomain(x, y) {
      if (x) this.o.x = x.slice();
      if (y) this.o.y = y.slice();
      if (this.o.equal) { // equal aspect: widen whichever axis is short
        const [x0, x1] = this.o.x, [y0, y1] = this.o.y;
        const ux = (x1 - x0) / this.iw, uy = (y1 - y0) / this.ih;
        if (ux > uy) { const c = (y0 + y1) / 2, h = ux * this.ih / 2; this.o.y = [c - h, c + h]; }
        else { const c = (x0 + x1) / 2, h = uy * this.iw / 2; this.o.x = [c - h, c + h]; }
      }
      const tf = (lg) => (lg ? Math.log10 : v => v);
      const fx = tf(this.o.logx), fy = tf(this.o.logy);
      const [x0, x1] = this.o.x.map(fx), [y0, y1] = this.o.y.map(fy);
      this.sx = v => this.m.l + (fx(v) - x0) / (x1 - x0) * this.iw;
      this.sy = v => this.m.t + this.ih - (fy(v) - y0) / (y1 - y0) * this.ih;
      this.ix = px => { const t = x0 + (px - this.m.l) / this.iw * (x1 - x0); return this.o.logx ? 10 ** t : t; };
      this.iy = py => { const t = y0 + (this.m.t + this.ih - py) / this.ih * (y1 - y0); return this.o.logy ? 10 ** t : t; };
      this.clipRect.setAttribute('x', this.m.l); this.clipRect.setAttribute('y', this.m.t);
      this.clipRect.setAttribute('width', this.iw); this.clipRect.setAttribute('height', this.ih);
      this.drawAxes();
      return this;
    }

    drawAxes() {
      const g = this.axesG, o = this.o, m = this.m;
      g.innerHTML = '';
      const logTicks = (lo, hi) => { const out = []; for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) out.push(10 ** e); return out; };
      const xt = o.logx ? logTicks(...o.x) : ticks(o.x[0], o.x[1], o.xticks);
      const yt = o.logy ? logTicks(...o.y) : ticks(o.y[0], o.y[1], o.yticks);
      const xs = xt.length > 1 ? xt[1] - xt[0] : 1, ys = yt.length > 1 ? yt[1] - yt[0] : 1;
      const bottom = m.t + this.ih, right = m.l + this.iw;
      el('rect', { x: m.l, y: m.t, width: this.iw, height: this.ih, class: 'plot-bg' }, g);
      if (o.grid) {
        for (const v of xt) el('line', { x1: this.sx(v), x2: this.sx(v), y1: m.t, y2: bottom, class: 'grid' }, g);
        for (const v of yt) el('line', { x1: m.l, x2: right, y1: this.sy(v), y2: this.sy(v), class: 'grid' }, g);
      }
      el('line', { x1: m.l, x2: right, y1: bottom, y2: bottom, class: 'axis' }, g);
      el('line', { x1: m.l, x2: m.l, y1: m.t, y2: bottom, class: 'axis' }, g);
      if (o.xticks) for (const v of xt) {
        const t = el('text', { x: this.sx(v), y: bottom + 16, class: 'tick', 'text-anchor': 'middle' }, g);
        t.textContent = o.logx ? '10^' + Math.round(Math.log10(v)) : (o.xfmt ? o.xfmt(v) : tickFmt(v, xs));
      }
      if (o.yticks) for (const v of yt) {
        const t = el('text', { x: m.l - 7, y: this.sy(v) + 4, class: 'tick', 'text-anchor': 'end' }, g);
        t.textContent = o.logy ? '10^' + Math.round(Math.log10(v)) : (o.yfmt ? o.yfmt(v) : tickFmt(v, ys));
      }
      if (o.xlabel) { const t = el('text', { x: m.l + this.iw / 2, y: o.height - 8, class: 'label', 'text-anchor': 'middle' }, g); t.textContent = o.xlabel; }
      if (o.ylabel) {
        const t = el('text', { x: 14, y: m.t + this.ih / 2, class: 'label', 'text-anchor': 'middle', transform: `rotate(-90 14 ${m.t + this.ih / 2})` }, g);
        t.textContent = o.ylabel;
      }
      if (o.title) { const t = el('text', { x: m.l, y: 17, class: 'ptitle' }, g); t.textContent = o.title; }
    }

    clear() { this.layer.innerHTML = ''; this.overlay.innerHTML = ''; return this; }

    _style(opts, fill) {
      const c = col(opts.color);
      const s = [];
      if (fill) s.push(`fill:${c}`, `fill-opacity:${opts.opacity ?? 0.25}`, opts.stroke ? `stroke:${col(opts.stroke)}` : 'stroke:none');
      else s.push('fill:none', `stroke:${c}`, `stroke-width:${opts.width ?? 2}`, `stroke-opacity:${opts.opacity ?? 1}`);
      if (opts.dash) s.push(`stroke-dasharray:${opts.dash === true ? '5 4' : opts.dash}`);
      return s.join(';');
    }
    _path(pts) {
      let d = '', pen = false;
      for (const p of pts) {
        if (p == null || !Number.isFinite(p[1]) || !Number.isFinite(p[0])) { pen = false; continue; }
        const X = this.sx(p[0]), Y = Math.max(-1e5, Math.min(1e5, this.sy(p[1])));
        d += (pen ? 'L' : 'M') + X.toFixed(2) + ' ' + Y.toFixed(2);
        pen = true;
      }
      return d;
    }

    // pts: [[x,y],...]; non-finite y breaks the line.
    line(pts, opts = {}) {
      return el('path', { d: this._path(pts), style: this._style(opts), 'stroke-linejoin': 'round', 'stroke-linecap': 'round', class: opts.class }, opts.parent || this.layer);
    }
    // Fill between pts and a baseline (number) or a second curve (array of [x,y] aligned with pts).
    area(pts, opts = {}) {
      const base = opts.base ?? 0;
      const lower = Array.isArray(base) ? base.slice().reverse() : [[pts[pts.length - 1][0], base], [pts[0][0], base]];
      const d = this._path(pts) + this._path(lower).replace(/^M/, 'L') + 'Z';
      return el('path', { d, style: this._style(opts, true) }, opts.parent || this.layer);
    }
    // Many points drawn as one path (fast for thousands of points).
    points(pts, opts = {}) {
      const r = opts.r ?? 3;
      if (opts.each) return pts.map(p => this.dot(p[0], p[1], opts));
      let d = '';
      for (const p of pts) if (Number.isFinite(p[0]) && Number.isFinite(p[1])) d += `M${this.sx(p[0]).toFixed(1)} ${this.sy(p[1]).toFixed(1)}h0`;
      return el('path', { d, style: `stroke:${col(opts.color)};stroke-width:${2 * r};stroke-linecap:round;stroke-opacity:${opts.opacity ?? 0.8};fill:none` }, opts.parent || this.layer);
    }
    dot(x, y, opts = {}) {
      return el('circle', { cx: this.sx(x), cy: this.sy(y), r: opts.r ?? 4,
        style: `fill:${col(opts.color)};fill-opacity:${opts.opacity ?? 1};stroke:${opts.stroke ? col(opts.stroke) : 'var(--surface)'};stroke-width:${opts.strokeWidth ?? 1.5}`, class: opts.class }, opts.parent || this.layer);
    }
    // bins: [{x0, x1, y}] or histogram output (uses .density unless opts.key given).
    bars(bins, opts = {}) {
      const key = opts.key || 'density';
      const g = el('g', {}, opts.parent || this.layer);
      const y0 = this.sy(opts.base ?? 0);
      for (const b of bins) {
        const v = b.y ?? b[key];
        if (!Number.isFinite(v)) continue;
        const X = this.sx(b.x0), W = Math.max(0.5, this.sx(b.x1) - X - (opts.gap ?? 1));
        const Y = this.sy(v);
        el('rect', { x: X, y: Math.min(Y, y0), width: W, height: Math.abs(y0 - Y),
          style: `fill:${col(b.color || opts.color)};fill-opacity:${opts.opacity ?? 0.55}`, rx: opts.rx ?? 1 }, g);
      }
      return g;
    }
    // Lollipop / stem plot for pmfs: pts [[k, p]].
    stems(pts, opts = {}) {
      const g = el('g', {}, opts.parent || this.layer);
      for (const [x, y] of pts) {
        el('line', { x1: this.sx(x), x2: this.sx(x), y1: this.sy(0), y2: this.sy(y), style: `stroke:${col(opts.color)};stroke-width:${opts.width ?? 2}` }, g);
        el('circle', { cx: this.sx(x), cy: this.sy(y), r: opts.r ?? 3.5, style: `fill:${col(opts.color)}` }, g);
      }
      return g;
    }
    // Step function from sorted xs: F jumps at each x. pts: [[x, F(x)]].
    step(pts, opts = {}) {
      const out = [];
      for (let i = 0; i < pts.length; i++) {
        out.push(pts[i]);
        if (i + 1 < pts.length) out.push([pts[i + 1][0], pts[i][1]]);
      }
      return this.line(out, opts);
    }
    segment(x0, y0, x1, y1, opts = {}) {
      return el('line', { x1: this.sx(x0), y1: this.sy(y0), x2: this.sx(x1), y2: this.sy(y1), style: this._style(opts), 'stroke-linecap': 'round' }, opts.parent || this.layer);
    }
    vline(x, opts = {}) { return this.segment(x, this.o.y[0], x, this.o.y[1], Object.assign({ width: 1.5, dash: true, color: 'muted' }, opts)); }
    hline(y, opts = {}) { return this.segment(this.o.x[0], y, this.o.x[1], y, Object.assign({ width: 1.5, dash: true, color: 'muted' }, opts)); }
    rect(x0, y0, x1, y1, opts = {}) {
      return el('rect', { x: Math.min(this.sx(x0), this.sx(x1)), y: Math.min(this.sy(y0), this.sy(y1)),
        width: Math.abs(this.sx(x1) - this.sx(x0)), height: Math.abs(this.sy(y1) - this.sy(y0)),
        style: this._style(Object.assign({ opacity: 0.15 }, opts), true) + (opts.stroke ? `;stroke:${col(opts.stroke)};stroke-width:1` : ''), rx: opts.rx }, opts.parent || this.layer);
    }
    // Shaded vertical band between x0 and x1.
    band(x0, x1, opts = {}) { return this.rect(x0, this.o.y[0], x1, this.o.y[1], Object.assign({ color: 'accent', opacity: 0.1 }, opts)); }
    text(x, y, str, opts = {}) {
      const t = el('text', { x: this.sx(x) + (opts.dx || 0), y: this.sy(y) + (opts.dy || 0), class: 'annot', 'text-anchor': opts.anchor || 'start',
        style: `fill:${col(opts.color || 'ink')};font-size:${opts.size || 12}px;${opts.weight ? 'font-weight:' + opts.weight : ''}` }, opts.parent || this.overlay);
      t.textContent = str;
      return t;
    }
    arrow(x0, y0, x1, y1, opts = {}) {
      const id = 'ah' + Math.random().toString(36).slice(2, 8);
      const mk = el('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, this.defs);
      el('path', { d: 'M0 0L10 5L0 10z', style: `fill:${col(opts.color || 'ink')}` }, mk);
      const l = this.segment(x0, y0, x1, y1, Object.assign({ color: 'ink', width: 1.6 }, opts));
      l.setAttribute('marker-end', `url(#${id})`);
      return l;
    }
    polygon(pts, opts = {}) { return el('path', { d: this._path(pts) + 'Z', style: this._style(opts, true) + (opts.stroke ? `;stroke:${col(opts.stroke)};stroke-width:${opts.width ?? 1.5}` : '') }, opts.parent || this.layer); }
    // Ellipse for a 2x2 covariance at a given Mahalanobis radius k.
    ellipse(mx, my, cov, k = 2, opts = {}) {
      const { values, vectors } = window.Atlas.S.eigSym(cov);
      const pts = window.Atlas.S.linspace(0, 2 * Math.PI, 90).map(t => {
        const a = k * Math.sqrt(Math.max(values[0], 0)) * Math.cos(t), b = k * Math.sqrt(Math.max(values[1], 0)) * Math.sin(t);
        return [mx + vectors[0][0] * a + vectors[0][1] * b, my + vectors[1][0] * a + vectors[1][1] * b];
      });
      return opts.fill ? this.polygon(pts, opts) : this.line(pts, opts);
    }
    // Raster heat map: f(x, y) -> [r, g, b, a] (0-255). Drawn at nx x ny resolution and scaled smoothly.
    raster(f, opts = {}) {
      const nx = opts.nx || 120, ny = opts.ny || 80;
      const c = document.createElement('canvas');
      c.width = nx; c.height = ny;
      const ctx = c.getContext('2d'), img = ctx.createImageData(nx, ny);
      for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const x = this.ix(this.m.l + (i + 0.5) / nx * this.iw), y = this.iy(this.m.t + (j + 0.5) / ny * this.ih);
        const rgba = f(x, y), k = 4 * (j * nx + i);
        img.data[k] = rgba[0]; img.data[k + 1] = rgba[1]; img.data[k + 2] = rgba[2]; img.data[k + 3] = rgba[3] ?? 255;
      }
      ctx.putImageData(img, 0, 0);
      const im = el('image', { x: this.m.l, y: this.m.t, width: this.iw, height: this.ih, preserveAspectRatio: 'none', href: c.toDataURL(), style: opts.pixelated ? 'image-rendering:pixelated' : '' }, opts.parent || this.layer);
      if (opts.opacity != null) im.setAttribute('opacity', opts.opacity);
      return im;
    }
    // Contour lines of f at the given levels (marching squares). Returns a <g>.
    contour(f, levels, opts = {}) {
      const nx = opts.nx || 80, ny = opts.ny || 60;
      const [x0, x1] = this.o.x, [y0, y1] = this.o.y;
      const gx = i => x0 + (x1 - x0) * i / nx, gy = j => y0 + (y1 - y0) * j / ny;
      const F = [];
      for (let j = 0; j <= ny; j++) { F.push([]); for (let i = 0; i <= nx; i++) F[j].push(f(gx(i), gy(j))); }
      const g = el('g', {}, opts.parent || this.layer);
      levels.forEach((lv, li) => {
        let d = '';
        const P = (x, y) => `${this.sx(x).toFixed(1)} ${this.sy(y).toFixed(1)}`;
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const v = [F[j][i], F[j][i + 1], F[j + 1][i + 1], F[j + 1][i]];
          const xs = [gx(i), gx(i + 1), gx(i + 1), gx(i)], ys = [gy(j), gy(j), gy(j + 1), gy(j + 1)];
          const cross = [];
          for (let e = 0; e < 4; e++) {
            const a = e, b = (e + 1) % 4;
            if ((v[a] < lv) !== (v[b] < lv)) {
              const t = (lv - v[a]) / (v[b] - v[a]);
              cross.push([xs[a] + t * (xs[b] - xs[a]), ys[a] + t * (ys[b] - ys[a])]);
            }
          }
          if (cross.length >= 2) d += `M${P(...cross[0])}L${P(...cross[1])}`;
          if (cross.length === 4) d += `M${P(...cross[2])}L${P(...cross[3])}`;
        }
        const c = Array.isArray(opts.color) ? opts.color[li % opts.color.length] : opts.color;
        el('path', { d, style: this._style(Object.assign({}, opts, { color: c, width: opts.width ?? 1.2 })) }, g);
      });
      return g;
    }
    legend(items, opts = {}) {
      const g = el('g', { class: 'legend' }, this.overlay);
      const x = opts.x ?? (opts.left ? this.m.l + 10 : this.m.l + this.iw - 10), anchorRight = !opts.left;
      let y = this.m.t + 14;
      for (const it of items) {
        const t = el('text', { x: anchorRight ? x - 22 : x + 22, y: y + 4, 'text-anchor': anchorRight ? 'end' : 'start', class: 'legend-t' }, g);
        t.textContent = it.label;
        const lx = anchorRight ? x - 16 : x;
        if (it.fill) el('rect', { x: lx, y: y - 5, width: 14, height: 10, rx: 2, style: `fill:${col(it.color)};fill-opacity:${it.opacity ?? 0.55}` }, g);
        else el('line', { x1: lx, x2: lx + 16, y1: y, y2: y, style: `stroke:${col(it.color)};stroke-width:${it.width ?? 2.5};${it.dash ? 'stroke-dasharray:5 4' : ''}` }, g);
        y += 18;
      }
      return g;
    }

    // Pointer events in data coordinates: handler(x, y, event, type) for 'down' | 'move' | 'up' | 'leave'.
    onPointer(handler) {
      const toData = (e) => {
        const r = this.svg.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width * this.o.width, py = (e.clientY - r.top) / r.height * this.o.height;
        return [this.ix(px), this.iy(py)];
      };
      this.svg.style.touchAction = 'none';
      this.svg.style.cursor = 'crosshair';
      let down = false;
      this.svg.addEventListener('pointerdown', e => { down = true; this.svg.setPointerCapture(e.pointerId); handler(...toData(e), e, 'down'); });
      this.svg.addEventListener('pointermove', e => handler(...toData(e), e, down ? 'drag' : 'move'));
      this.svg.addEventListener('pointerup', e => { down = false; handler(...toData(e), e, 'up'); });
      this.svg.addEventListener('pointerleave', e => handler(...toData(e), e, 'leave'));
      return this;
    }
    // Is a data-space point inside the plotting area?
    inside(x, y) { return x >= Math.min(...this.o.x) && x <= Math.max(...this.o.x) && y >= Math.min(...this.o.y) && y <= Math.max(...this.o.y); }
  }

  Plot.ticks = ticks;

  // Resolve a theme token to an [r,g,b] triple (for raster colouring).
  function rgb(token) {
    const v = getComputedStyle(document.documentElement).getPropertyValue('--' + token).trim() || token;
    const c = document.createElement('canvas').getContext('2d');
    c.fillStyle = v;
    const h = c.fillStyle; // normalised to #rrggbb
    if (h[0] === '#') return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
    const m = h.match(/[\d.]+/g); return m ? m.slice(0, 3).map(Number) : [128, 128, 128];
  }
  // Linear blend of rgb triples.
  const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

  window.Atlas.Plot = Plot;
  window.Atlas.rgb = rgb;
  window.Atlas.mix = mix;
})();
