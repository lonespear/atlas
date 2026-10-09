# Authoring guide

The Atlas is a static site (no build step). Content and labs are plain JS files that register into
`window.Atlas`. **Reference implementation: `content/resampling.js` (topic `bootstrap`) and
`js/labs/resampling.js`.** Read both before writing anything. Match their depth and style.

Run `node tools/check.js --only <file-prefix>` (e.g. `--only evt`, `--only deep-1`) until it reports
0 errors. Preview with `python -m http.server 8840` from the repo root → http://127.0.0.1:8840/#/t/<id>.

## Files

- `content/<file>.js` — topics. `js/labs/<file>.js` — labs. `<file>` is the `file` field in
  `tools/catalog.json` (e.g. `foundations-1`). Both are already listed in `index.html`.
- **Never edit** `js/core/*`, `js/app.js`, `css/atlas.css`, `index.html`, `content/domains.js`,
  `tools/*`, or another author's files. If you need a numerical helper that `Atlas.S` lacks,
  write it inside your own lab file's IIFE.

## Topic schema

```js
(function () {
  const R = String.raw;          // ALWAYS: content strings are String.raw template literals
  Atlas.topic({
    id: 'mle',                   // exactly the catalog id
    domain: 'inference',
    order: 70,                   // exactly the catalog order
    title: 'Maximum Likelihood Estimation',
    subtitle: 'One line, sentence case, no trailing period',
    level: 2,                    // 1 = introductory, 2 = core graduate, 3 = advanced
    sources: { aos: '9.3–9.8', cb: '§7.2.2, §10.1', other: 'Optional further reading' },
    prereqs: ['likelihood'],     // catalog ids, the topics you truly need first (keep it to 1–3)
    related: ['fisher-information', 'em-algorithm'],
    tags: ['estimation', 'asymptotics'],
    summary: R`One paragraph lede (3–5 sentences): what it is, why it matters, the key idea.`,
    survey: {                    // one row in the Method Survey table (short phrases, math allowed)
      task: '', assumptions: '', output: '', strengths: '', weaknesses: '', complexity: '',
    },
    sections: [
      { title: 'Intuition', html: R`...`, lab: 'mle-likelihood-surface' },  // `lab` appends a lab after the html
      ...
    ],
    quiz: [ { q: R`...`, a: R`...` }, ... ],   // 3–5 questions; answers explain, not just state
  });
})();
```

### Content rules

1. **String.raw everywhere** (`R\`...\``), so `\frac` etc. survive. Never write `${` anywhere in content
   (it interpolates). If TeX needs `${`, put a space: `$ {n \choose k}$` or use `\binom`.
2. **Math**: `$...$` inline, `$$...$$` display. KaTeX, not full LaTeX — no `\begin{align}` (use
   `\begin{aligned}` inside `$$`), no `\label`, no `\newcommand`. Literal dollar sign: `\$` won't work
   in prose; write "USD" or "dollars". Macros available: `\E \P \R \N \Var \Cov \Corr \Bias \MSE \se
   \argmax \argmin \iid \indep \ind \tr \diag \sgn \KL \dto \pto \asto \bx \by \bX \bY \bbeta \btheta
   \bmu \bSigma \hth \d` (`\d x` gives a thin space + upright d).
3. **Blocks** (plain HTML inside `html`):
   - `<div class="def" data-name="Sufficient statistic">…</div>` — definition
   - `<div class="thm" data-kind="Theorem|Lemma|Proposition|Corollary" data-name="Cramér–Rao">…</div>`
   - `<details class="proof"><summary>Proof</summary>…</details>` directly after a `.thm` (attaches to it).
     Use "Proof sketch" in the summary when you skip steps. Proofs should be real proofs.
   - `<div class="note">…</div>` — intuition box (`data-name="…"` overrides the "Intuition" label)
   - `<div class="ex" data-name="…">…</div>` — worked example
   - `<div class="warn" data-name="…">…</div>` — pitfall
   - `<div class="algo" data-name="…"><ol>…</ol></div>` — algorithm
   - `<div class="cols"><div><h4>…</h4>…</div>…</div>` — side-by-side comparison cards
   - tables: plain `<table><thead>…</thead><tbody>…</tbody></table>`
   - Inline lab anywhere: `<div data-lab="lab-id"></div>`
   - Cross-links: `<a href="#/t/<catalog-id>">text</a>`
4. **Depth.** Each topic is a mini-chapter: typically 5–7 sections, ~1,500–3,000 words. A good arc:
   motivation/problem → intuition → formal setup & definitions → main theorems **with proofs or proof
   sketches** → worked example(s) → interactive lab(s) → pitfalls / when it fails → connections /
   extensions. State regularity conditions. Give the *why*, not only the *what*. Graduate level,
   written for someone who wants to understand the theory deeply.
5. **Originality.** Write in your own words. Cite chapter/section numbers from All of Statistics (AoS) and
   Casella & Berger (C&B) as given in the catalog, but never reproduce their text, proofs verbatim,
   examples verbatim, or exercises. Standard results and standard proofs are fine in your own
   exposition. Attribute named results (e.g. "Bickel & Freedman, 1981").
6. **Correctness above all.** Every formula, constant, and condition must be right. If unsure of a
   detail, state the result in the form you are sure of.
7. Prefer `prereqs` that are genuinely necessary; `related` for lateral links. Both must be catalog ids.
8. No emojis. Typographic dashes (–, —) and real minus signs are welcome.

## Lab API

```js
(function () {
  const S = Atlas.S, UI = Atlas.UI, Plot = Atlas.Plot;
  Atlas.lab('mle-likelihood-surface', {          // id MUST start with the topic id
    title: 'Short title (math ok)',
    blurb: 'One or two sentences: what to look at, what to try. Math ok (normal JS string: escape backslashes as \\, or use String.raw).',
    mount(el, ctx) {
      const { controls, stage, stats } = UI.layout(el);       // or UI.layout(el, { side: true })
      const n = UI.slider(controls, { label: 'Sample size $n$', min: 5, max: 500, step: 5, value: 50, onInput: draw });
      // onInput fires continuously while dragging; use onChange for expensive recomputes.
      // Also: { log: true } for log-scale sliders; fmt: v => string for display.
      const fam = UI.segmented(controls, { label: 'Family', options: ['Normal', 'Exponential'], value: 'Normal', onChange: draw });
      const sel = UI.select(controls, { label: 'Data', options: [{ value: 'a', label: 'A' }], onChange: draw });
      const tg = UI.toggle(controls, { label: 'Show truth', value: true, onChange: draw });
      UI.button(controls, 'Resample', resample, { primary: true });
      const loop = ctx.loop(dt => { /* advance; return false to stop */ });
      ctx.playButton(controls, loop);                         // ▶/❚❚ button bound to the loop
      const st = UI.stat(stats, 'Estimate $\\hat\\theta$');   // st.set(html, subtitleHtml?)
      const note = UI.note(stats, '');                        // note.set(html) – KaTeX rendered

      const p = new Plot(stage, { width: 640, height: 300, x: [-4, 4], y: [0, 0.5], xlabel: 'x', ylabel: 'density', title: '' });
      // stage.classList.add('two') + two Plots of width ~420 for side-by-side panels.
      function draw() {
        p.setDomain([lo, hi], [0, ymax]).clear();             // clear() before each redraw
        p.line(pts, { color: 's1', width: 2, dash: true, opacity: 1 });
        p.area(pts, { color: 's1', opacity: 0.2, base: 0 });
        p.bars(S.histogram(data, { bins: 40 }), { color: 's2' });
        p.points(pts, { color: 's3', r: 3, opacity: 0.7 });   // fast, thousands ok
        p.dot(x, y, { color: 's4', r: 5 });                   // individual circle
        p.stems([[k, pk], ...], { color: 's1' });             // pmf lollipops
        p.step(pts, {...}); p.segment(x0, y0, x1, y1, {...}); p.vline(x, {...}); p.hline(y, {...});
        p.band(x0, x1, {...}); p.rect(x0, y0, x1, y1, {...}); p.polygon(pts, {...}); p.arrow(x0, y0, x1, y1, {...});
        p.ellipse(mx, my, cov2x2, k, {...}); p.contour(f, levels, { color, nx, ny });
        p.raster((x, y) => [r, g, b, a], { nx: 120, ny: 80 });   // heat maps / decision regions
        p.text(x, y, 'label', { anchor: 'middle', dx, dy, size, color, weight });
        p.legend([{ label: 'MLE', color: 's1' }, { label: 'data', color: 's2', fill: true }]);
      }
      p.onPointer((x, y, evt, type) => { /* type: 'down' | 'drag' | 'move' | 'up' | 'leave' — click to add points, drag, etc. */ });
      draw();
    },
  });
})();
```

- Colours: tokens `'s1'…'s8'` (series), `'ink'`, `'muted'`, `'faint'`, `'accent'`, `'good'`, `'bad'`,
  `'warn'`. They follow light/dark theme automatically. For rasters use `Atlas.rgb('s1')` → `[r,g,b]`
  and `Atlas.mix(a, b, t)`; re-read colours on each draw.
- Randomness: **always** `ctx.rng` (seeded; Reset re-seeds). `ctx.rng.normal(mu, sd)`, `.uniform(a,b)`,
  `.exp(rate)`, `.gamma(shape, scale)`, `.beta(a,b)`, `.poisson(l)`, `.binomial(n,p)`, `.t(df)`,
  `.chisq(k)`, `.cauchy()`, `.laplace()`, `.lognormal()`, `.weibull(k,l)`, `.pareto(a,xm)`,
  `.gumbel()`, `.gev(mu,s,xi)`, `.gpd(s,xi)`, `.categorical(p)`, `.choice(arr)`, `.sample(arr,n,replace)`,
  `.shuffle(arr)`, `.mvnormal(mu, Sigma)`, `.dirichlet(alpha)`, `.array(n, i => …)`, `.u()` (raw uniform).
- `Atlas.S`: `pdf.{normal,uniform,exp,gamma,beta,t,chisq,cauchy,laplace,logistic,lognormal,weibull,pareto,gev,gpd,mvnormal2}`,
  `pmf.{binom,poisson,geom,negbinom,bernoulli}`, `cdf.{…, f, binom, poisson}`, `q.{normal,exp,t,chisq,gamma,beta,f,cauchy,logistic,weibull,pareto,gev,gpd}`,
  `lgamma gamma lbeta lchoose gammaP ibeta erf qnorm invert`, `sum mean variance sd quantile median extent cov cor cumsum linspace range clamp logistic logit logsumexp argmax argmin histogram ecdf kde silverman`,
  `transpose matmul matvec dot identity solve inv det cholesky eigSym ols`, `nelderMead golden grad`, `fmt(x, digits)`.
- Performance: initial mount < ~300 ms; animations must do bounded work per frame. Use `ctx.loop`
  (auto-paused off-screen and stopped on unmount) — never raw `setInterval`. Register any other
  listeners/timers with `ctx.onCleanup`.
- Every lab should teach something specific that prose cannot: let the reader *break* the method
  (violate an assumption, push a parameter to its limit) and see the consequence. Show numbers
  (stat tiles) alongside pictures. 2–3 labs per topic where they add value; at least 1 per topic.
- Plot sizes: single panel 640×300 (or ×340); two panels 420×270 each with `stage.classList.add('two')`.
