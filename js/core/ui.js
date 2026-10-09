/* Atlas.UI — controls and layout used inside labs.
 *
 *   mount(el, ctx) {
 *     const { controls, stage, stats } = Atlas.UI.layout(el);
 *     const n = Atlas.UI.slider(controls, { label: 'Sample size n', min: 5, max: 500, step: 5, value: 50, onInput: draw });
 *     const mode = Atlas.UI.segmented(controls, { label: 'Kernel', options: ['Gaussian', 'Epanechnikov'], value: 'Gaussian', onChange: draw });
 *     Atlas.UI.button(controls, 'Resample', draw, { primary: true });
 *     const m = Atlas.UI.stat(stats, 'Sample mean');
 *     const plot = new Atlas.Plot(stage, {...});
 *     function draw() { ...; m.set(Atlas.S.fmt(xbar)); }
 *     draw();
 *   }
 *
 * ctx (passed to mount) provides:
 *   ctx.rng            a seeded Atlas.S.rng — reset re-seeds it
 *   ctx.newRng()       a fresh generator with a new random seed
 *   ctx.loop(step)     animation loop; step(dtSeconds) returns false to stop.
 *                      Returns {start, stop, toggle, running}. Auto-stopped on unmount
 *                      and paused while the lab is scrolled off-screen.
 *   ctx.onCleanup(fn)  run fn when the lab unmounts (route change / reset)
 *   ctx.math(el)       typeset KaTeX inside el
 */
(function () {
  'use strict';
  const h = (tag, cls, parent, html) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    if (parent) parent.appendChild(e);
    return e;
  };

  function math(el) {
    if (window.renderMathInElement) {
      window.renderMathInElement(el, {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '\\[', right: '\\]', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
        ],
        macros: window.Atlas.macros,
        throwOnError: false,
        trust: false,
      });
    }
    return el;
  }

  // Inline KaTeX string -> HTML (for readouts).
  function tex(src, display = false) {
    if (!window.katex) return src;
    return window.katex.renderToString(src, { displayMode: display, throwOnError: false, macros: window.Atlas.macros });
  }

  function layout(el, { side = false } = {}) {
    el.classList.add(side ? 'lab-side' : 'lab-stack');
    const controls = h('div', 'lab-controls', el);
    const main = h('div', 'lab-main', el);
    const stage = h('div', 'lab-stage', main);
    const stats = h('div', 'lab-stats', main);
    return { controls, stage, stats };
  }

  function slider(parent, o) {
    const fmt = o.fmt || (v => {
      const st = o.step ?? 1;
      const d = st >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(st)));
      return (+v).toFixed(d);
    });
    const wrap = h('label', 'ctl ctl-slider', parent);
    const top = h('div', 'ctl-top', wrap);
    const lab = h('span', 'ctl-label', top, o.label);
    const val = h('span', 'ctl-val', top);
    const inp = h('input', null, wrap);
    inp.type = 'range';
    inp.min = o.min; inp.max = o.max; inp.step = o.step ?? 1; inp.value = o.value;
    const toValue = () => (o.log ? 10 ** +inp.value : +inp.value);
    if (o.log) { inp.min = Math.log10(o.min); inp.max = Math.log10(o.max); inp.step = o.logStep ?? 0.01; inp.value = Math.log10(o.value); }
    const paint = () => {
      val.textContent = fmt(toValue());
      const t = (inp.value - inp.min) / (inp.max - inp.min);
      inp.style.setProperty('--fill', (100 * t).toFixed(1) + '%');
    };
    paint();
    math(lab);
    inp.addEventListener('input', () => { paint(); o.onInput && o.onInput(toValue()); });
    inp.addEventListener('change', () => { o.onChange && o.onChange(toValue()); });
    return {
      el: wrap, input: inp,
      get: toValue,
      set(v, fire = false) { inp.value = o.log ? Math.log10(v) : v; paint(); if (fire && o.onInput) o.onInput(toValue()); },
      get value() { return toValue(); },
    };
  }

  // options: array of strings or {value, label}
  function segmented(parent, o) {
    const opts = o.options.map(x => (typeof x === 'object' ? x : { value: x, label: x }));
    const wrap = h('div', 'ctl ctl-seg', parent);
    if (o.label) math(h('span', 'ctl-label', wrap, o.label));
    const row = h('div', 'seg', wrap);
    let cur = o.value ?? opts[0].value;
    const btns = opts.map(op => {
      const b = h('button', 'seg-b', row, op.label);
      b.type = 'button';
      b.addEventListener('click', () => { set(op.value); o.onChange && o.onChange(cur); });
      return [op.value, b];
    });
    math(row);
    function set(v) { cur = v; for (const [val, b] of btns) b.classList.toggle('on', val === v); }
    set(cur);
    return { el: wrap, get: () => cur, set, get value() { return cur; } };
  }

  function select(parent, o) {
    const opts = o.options.map(x => (typeof x === 'object' ? x : { value: x, label: x }));
    const wrap = h('label', 'ctl ctl-select', parent);
    if (o.label) math(h('span', 'ctl-label', wrap, o.label));
    const s = h('select', null, wrap);
    for (const op of opts) { const e = h('option', null, s, op.label); e.value = op.value; }
    s.value = o.value ?? opts[0].value;
    s.addEventListener('change', () => o.onChange && o.onChange(s.value));
    return { el: wrap, get: () => s.value, set: v => { s.value = v; }, get value() { return s.value; } };
  }

  function toggle(parent, o) {
    const wrap = h('label', 'ctl ctl-toggle', parent);
    const inp = h('input', null, wrap);
    inp.type = 'checkbox'; inp.checked = !!o.value;
    h('span', 'switch', wrap);
    math(h('span', 'ctl-label', wrap, o.label));
    inp.addEventListener('change', () => o.onChange && o.onChange(inp.checked));
    return { el: wrap, get: () => inp.checked, set: v => { inp.checked = v; }, get value() { return inp.checked; } };
  }

  function button(parent, label, onClick, o = {}) {
    const b = h('button', 'btn' + (o.primary ? ' btn-primary' : '') + (o.small ? ' btn-sm' : ''), parent, label);
    b.type = 'button';
    b.addEventListener('click', onClick);
    return b;
  }

  // A stat tile: label over a big value. set(v, sub?) updates it.
  function stat(parent, label, o = {}) {
    const wrap = h('div', 'stat' + (o.accent ? ' stat-' + o.accent : ''), parent);
    math(h('div', 'stat-l', wrap, label));
    const v = h('div', 'stat-v', wrap, o.value ?? '—');
    const sub = h('div', 'stat-s', wrap, '');
    return { el: wrap, set(val, s) { v.innerHTML = val; if (s != null) sub.innerHTML = s; } };
  }

  // Free-form note / readout area (KaTeX-rendered on set).
  function note(parent, html = '') {
    const e = h('div', 'lab-note', parent, html);
    math(e);
    return { el: e, set(x) { e.innerHTML = x; math(e); } };
  }

  // A labelled row break in the controls column.
  function divider(parent, label) { return h('div', 'ctl-div', parent, label || ''); }

  // Builds the per-mount context object.
  function makeContext(el, seed) {
    const cleanups = [];
    const loops = [];
    let visible = true;
    const io = 'IntersectionObserver' in window ? new IntersectionObserver(es => { for (const e of es) visible = e.isIntersecting; }) : null;
    if (io) io.observe(el);
    cleanups.push(() => io && io.disconnect());
    const ctx = {
      el,
      seed,
      rng: window.Atlas.S.rng(seed),
      newRng: () => window.Atlas.S.rng(),
      math,
      tex,
      onCleanup: fn => cleanups.push(fn),
      loop(step) {
        let raf = null, last = 0;
        const L = {
          running: false,
          start() {
            if (L.running) return L;
            L.running = true; last = performance.now();
            const tick = now => {
              if (!L.running) return;
              const dt = Math.min(0.1, (now - last) / 1000); last = now;
              if (visible && step(dt) === false) { L.stop(); return; }
              raf = requestAnimationFrame(tick);
            };
            raf = requestAnimationFrame(tick);
            L.onchange && L.onchange(true);
            return L;
          },
          stop() { L.running = false; if (raf) cancelAnimationFrame(raf); raf = null; L.onchange && L.onchange(false); return L; },
          toggle() { return L.running ? L.stop() : L.start(); },
        };
        loops.push(L);
        return L;
      },
      // Play/pause button wired to a loop.
      playButton(parent, loop, labels = ['▶ Run', '❚❚ Pause']) {
        const b = button(parent, labels[0], () => loop.toggle(), { primary: true });
        loop.onchange = on => { b.textContent = on ? labels[1] : labels[0]; };
        return b;
      },
      destroy() { for (const L of loops) L.stop(); for (const f of cleanups.splice(0)) { try { f(); } catch (e) { console.error(e); } } },
    };
    return ctx;
  }

  window.Atlas.UI = { h, math, tex, layout, slider, segmented, select, toggle, button, stat, note, divider, makeContext };

  // KaTeX macros available everywhere (content and labs).
  window.Atlas.macros = {
    '\\E': '\\mathbb{E}',
    '\\P': '\\mathbb{P}',
    '\\R': '\\mathbb{R}',
    '\\N': '\\mathcal{N}',
    '\\Var': '\\operatorname{Var}',
    '\\Cov': '\\operatorname{Cov}',
    '\\Corr': '\\operatorname{Corr}',
    '\\Bias': '\\operatorname{Bias}',
    '\\MSE': '\\operatorname{MSE}',
    '\\se': '\\operatorname{se}',
    '\\argmax': '\\operatorname*{arg\\,max}',
    '\\argmin': '\\operatorname*{arg\\,min}',
    '\\iid': '\\overset{\\text{iid}}{\\sim}',
    '\\indep': '\\perp\\!\\!\\!\\perp',
    '\\ind': '\\mathbb{1}',
    '\\tr': '\\operatorname{tr}',
    '\\diag': '\\operatorname{diag}',
    '\\sgn': '\\operatorname{sgn}',
    '\\KL': '\\operatorname{KL}',
    '\\dto': '\\xrightarrow{d}',
    '\\pto': '\\xrightarrow{p}',
    '\\asto': '\\xrightarrow{\\text{a.s.}}',
    '\\bx': '\\mathbf{x}',
    '\\by': '\\mathbf{y}',
    '\\bX': '\\mathbf{X}',
    '\\bY': '\\mathbf{Y}',
    '\\bbeta': '\\boldsymbol{\\beta}',
    '\\btheta': '\\boldsymbol{\\theta}',
    '\\bmu': '\\boldsymbol{\\mu}',
    '\\bSigma': '\\boldsymbol{\\Sigma}',
    '\\hth': '\\hat{\\theta}',
    '\\d': '\\,\\mathrm{d}',
  };
})();
