/* Atlas application: routing, views, search, progress. Loaded last. */
(function () {
  'use strict';
  const A = window.Atlas;
  const { h, math } = A.UI;
  const $ = (s, r = document) => r.querySelector(s);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const domainNum = d => d.color.match(/--d(\d+)/)?.[1] || '1';

  /* ---------------- persistence ---------------- */
  const store = {
    get(k, d) { try { const v = localStorage.getItem('atlas:' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem('atlas:' + k, JSON.stringify(v)); } catch (e) { /* storage blocked */ } },
  };
  let done = new Set(store.get('done', []));
  const isDone = id => done.has(id);
  function setDone(id, v) { v ? done.add(id) : done.delete(id); store.set('done', [...done]); renderSidebar(); }

  /* ---------------- theme ---------------- */
  function applyTheme(t) {
    if (t === 'system') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', t);
    store.set('theme', t);
    const b = $('#theme-btn');
    if (b) b.innerHTML = t === 'dark' ? '☾' : t === 'light' ? '☀' : '◐';
    if (b) b.title = 'Theme: ' + t;
  }
  function cycleTheme() {
    const order = ['system', 'light', 'dark'];
    applyTheme(order[(order.indexOf(store.get('theme', 'system')) + 1) % 3]);
    document.dispatchEvent(new CustomEvent('atlas:theme'));
  }

  /* ---------------- sidebar ---------------- */
  function renderSidebar() {
    const nav = $('#nav-domains');
    const route = parseRoute();
    const activeTopic = route.view === 'topic' ? route.id : null;
    const activeDomain = route.view === 'domain' ? route.id : activeTopic && A.topicMap[activeTopic] ? A.topicMap[activeTopic].domain : null;
    const openState = store.get('navOpen', {});
    nav.innerHTML = '';
    for (const d of A.domains) {
      const det = h('details', 'nav-domain', nav);
      det.style.setProperty('--dc', d.color);
      det.open = d.id === activeDomain || !!openState[d.id];
      const nd = d.topics.filter(t => isDone(t.id)).length;
      const sum = h('summary', null, det,
        `<span class="chev">▶</span><span class="sym">${esc(d.symbol)}</span><span>${esc(d.title)}</span><span class="cnt">${nd ? nd + '/' : ''}${d.topics.length}</span>`);
      sum.addEventListener('dblclick', () => { location.hash = '#/d/' + d.id; });
      det.addEventListener('toggle', () => { const s = store.get('navOpen', {}); s[d.id] = det.open; store.set('navOpen', s); });
      const list = h('div', 'nav-topics', det);
      h('a', 'nav-topic' + (route.view === 'domain' && route.id === d.id ? ' active' : ''), list, '<em>Overview</em>').href = '#/d/' + d.id;
      for (const t of d.topics) {
        const a = h('a', 'nav-topic' + (t.id === activeTopic ? ' active' : ''), list, `<span>${esc(t.title)}</span>${isDone(t.id) ? '<span class="done">✓</span>' : ''}`);
        a.href = '#/t/' + t.id;
      }
    }
    for (const a of document.querySelectorAll('.nav-main .nav-link')) a.classList.toggle('active', a.dataset.view === route.view);
    const act = nav.querySelector('.nav-topic.active');
    if (act) { const r = act.getBoundingClientRect(), s = $('#sidebar').getBoundingClientRect(); if (r.top < s.top || r.bottom > s.bottom) act.scrollIntoView({ block: 'center' }); }
  }

  /* ---------------- routing ---------------- */
  function parseRoute() {
    const p = location.hash.replace(/^#\/?/, '').split('/');
    if (p[0] === 't' && p[1]) return { view: 'topic', id: decodeURIComponent(p[1]), anchor: p[2] };
    if (p[0] === 'd' && p[1]) return { view: 'domain', id: decodeURIComponent(p[1]) };
    if (p[0] === 'map') return { view: 'map' };
    if (p[0] === 'survey') return { view: 'survey' };
    if (p[0] === 'progress') return { view: 'progress' };
    return { view: 'home' };
  }

  let mounted = []; // live lab contexts
  let viewCleanup = [];
  function teardown() {
    for (const c of mounted) c.destroy();
    mounted = [];
    for (const f of viewCleanup.splice(0)) f();
  }

  function route() {
    teardown();
    const r = parseRoute();
    const view = $('#view');
    view.innerHTML = '';
    view.className = 'view';
    void view.offsetWidth; // restart fade-in animation
    document.body.classList.remove('nav-open');
    const crumbs = $('#crumbs');
    crumbs.innerHTML = '';
    if (r.view === 'topic') renderTopic(view, r.id, crumbs, r.anchor);
    else if (r.view === 'domain') renderDomain(view, r.id, crumbs);
    else if (r.view === 'map') renderMap(view, crumbs);
    else if (r.view === 'survey') renderSurvey(view, crumbs);
    else if (r.view === 'progress') renderProgress(view, crumbs);
    else renderHome(view, crumbs);
    renderSidebar();
    if (!(r.view === 'topic' && r.anchor)) window.scrollTo(0, 0);
  }

  function crumb(c, items) {
    c.innerHTML = items.map((it, i) => (i ? '<span class="sep">/</span>' : '') + (it.href ? `<a href="${it.href}">${esc(it.label)}</a>` : `<span>${esc(it.label)}</span>`)).join('');
  }

  const levelHtml = (n) => `<span class="level" title="Level ${n} of 3">${[1, 2, 3].map(i => `<i class="${i <= n ? 'on' : ''}"></i>`).join('')}</span>`;
  const labCount = t => (t.sections.map(s => s.html || '').join('').match(/data-lab=/g) || []).length;
  const allLabIds = () => A.topics.reduce((n, t) => n + labCount(t), 0);

  /* ---------------- home ---------------- */
  function renderHome(view, crumbs) {
    crumb(crumbs, [{ label: 'Atlas' }]);
    document.title = 'Atlas · Statistical Theory & Learning';
    const nTopics = A.topics.length, nDone = A.topics.filter(t => isDone(t.id)).length;
    const hero = h('section', 'hero', view);
    const cv = h('canvas', null, hero);
    h('div', 'hero-body', hero, `
      <div class="hero-kicker">A field guide to statistical theory &amp; machine learning</div>
      <h1>From <em>Ω</em> to the transformer.</h1>
      <p>A survey of probability, inference, models, and learning: each method derived from first principles, proved where it matters, and opened up in an interactive lab you can break. Built on Wasserman's <i>All of Statistics</i> and Casella &amp; Berger's <i>Statistical Inference</i>.</p>
      <div class="hero-stats">
        <div><div class="n">${A.domains.length}</div><div class="l">fields</div></div>
        <div><div class="n">${nTopics}</div><div class="l">topics</div></div>
        <div><div class="n">${allLabIds()}</div><div class="l">interactive labs</div></div>
        <div><div class="n">${nDone}</div><div class="l">marked understood</div></div>
      </div>`);
    heroAnimation(cv);

    const sh = h('div', 'section-h', view, '<h2>The fields</h2><span class="aside">Ordered roughly by dependency, from foundations upward.</span>');
    const grid = h('div', 'domain-grid', view);
    for (const d of A.domains) {
      const a = h('a', 'domain-card', grid);
      a.href = '#/d/' + d.id;
      a.style.setProperty('--dc', d.color);
      const nd = d.topics.filter(t => isDone(t.id)).length;
      a.innerHTML = `
        <div class="top"><div class="sym">${esc(d.symbol)}</div><div><h3>${esc(d.title)}</h3><div class="meta">${d.topics.length} topics · AoS ${esc(d.aos)} · C&amp;B ${esc(d.cb)}</div></div></div>
        <p>${esc(d.blurb)}</p>
        <div class="progress"><i style="width:${d.topics.length ? (100 * nd / d.topics.length).toFixed(0) : 0}%"></i></div>`;
    }
    void sh;

    h('div', 'section-h', view, '<h2>The two anchor texts</h2>');
    h('div', 'texts', view, `
      <div class="text-card"><h4>All of Statistics</h4><div class="by">Larry Wasserman · Springer, 2004 · cited as <b>AoS</b></div>
        <p>A fast, rigorous tour that puts classical inference and modern methods — nonparametrics, bootstrap, classification, simulation — on one footing. The Atlas follows its breadth.</p></div>
      <div class="text-card"><h4>Statistical Inference</h4><div class="by">George Casella &amp; Roger L. Berger · 2nd ed., Duxbury, 2002 · cited as <b>C&amp;B</b></div>
        <p>The canonical graduate treatment of distribution theory, sufficiency, point estimation, testing, and interval estimation. The Atlas follows its depth: proofs, regularity conditions, and optimality.</p></div>
      <div class="text-card"><h4>How to read a topic</h4><div class="by">Every page has the same anatomy</div>
        <p>A one-paragraph lede, the intuition, the formal setup, the key theorems with collapsible proofs, an interactive lab, worked examples and pitfalls, then self-check questions and links to what comes before and after.</p></div>`);
  }

  // Hero: Galton-style random walks settling into a bell curve.
  function heroAnimation(cv) {
    const ctx = cv.getContext('2d');
    const rng = A.S.rng(7);
    let W = 0, H = 0, dpr = 1;
    const resize = () => { dpr = window.devicePixelRatio || 1; W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); };
    resize();
    const bins = new Array(61).fill(0);
    const walkers = [];
    const css = getComputedStyle(document.documentElement);
    let colors = [];
    const readColors = () => { colors = ['--d1', '--d2', '--d12', '--d7', '--d3'].map(v => css.getPropertyValue(v).trim()); };
    readColors();
    const steps = 60;
    let raf, t = 0;
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    function spawn() {
      const pts = [0];
      for (let i = 0; i < steps; i++) pts.push(pts[pts.length - 1] + (rng.u() < 0.5 ? -1 : 1));
      walkers.push({ pts, k: 0, c: colors[walkers.length % colors.length] });
    }
    function frame() {
      t++;
      if (t % 3 === 0 && walkers.length < 400) spawn();
      ctx.clearRect(0, 0, W, H);
      const x0 = W * 0.42, xw = W * 0.56, ymid = H * 0.5, dy = H / 70;
      // histogram of finished walks
      const maxb = Math.max(8, ...bins);
      ctx.globalAlpha = 0.18;
      for (let i = 0; i < bins.length; i++) {
        const y = ymid + (i - 30) * dy;
        const w = bins[i] / maxb * W * 0.16;
        ctx.fillStyle = colors[0];
        ctx.fillRect(x0 + xw - w, y - dy * 0.42, w, dy * 0.84);
      }
      // normal envelope
      ctx.globalAlpha = 0.5; ctx.strokeStyle = colors[1]; ctx.lineWidth = 1.5; ctx.beginPath();
      const tot = bins.reduce((s, b) => s + b, 0);
      if (tot > 20) {
        for (let i = -30; i <= 30; i += 0.25) {
          const dens = Math.exp(-i * i / (2 * steps)) / Math.sqrt(2 * Math.PI * steps) * tot * 2;
          const x = x0 + xw - dens / maxb * W * 0.16, y = ymid + i * dy;
          i === -30 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }
      // walkers
      for (const w of walkers) {
        if (w.k < steps) w.k += 0.5;
        const n = Math.floor(w.k);
        ctx.globalAlpha = w.k >= steps ? 0.05 : 0.35;
        ctx.strokeStyle = w.c; ctx.lineWidth = 1; ctx.beginPath();
        for (let i = 0; i <= n; i++) { const x = x0 + i / steps * xw, y = ymid + w.pts[i] * dy; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.stroke();
        if (w.k >= steps && !w.counted) { w.counted = true; bins[Math.max(0, Math.min(60, (w.pts[steps] / 2 + 15) * 2 | 0))]++; }
      }
      ctx.globalAlpha = 1;
      // fade-out left edge so text stays legible
      const g = ctx.createLinearGradient(0, 0, W * 0.62, 0);
      const bg = css.getPropertyValue('--surface').trim();
      g.addColorStop(0, bg); g.addColorStop(0.55, bg); g.addColorStop(1, 'transparent');
      ctx.fillStyle = g; ctx.fillRect(0, 0, W * 0.62, H);
      if (walkers.length > 380) { walkers.splice(0, 1); }
      raf = requestAnimationFrame(frame);
    }
    if (reduced) { for (let i = 0; i < 300; i++) spawn(); walkers.forEach(w => { w.k = steps; }); frame(); cancelAnimationFrame(raf); }
    else raf = requestAnimationFrame(frame);
    const onResize = () => resize();
    const onTheme = () => setTimeout(readColors, 30);
    window.addEventListener('resize', onResize);
    document.addEventListener('atlas:theme', onTheme);
    viewCleanup.push(() => { cancelAnimationFrame(raf); window.removeEventListener('resize', onResize); document.removeEventListener('atlas:theme', onTheme); });
  }

  /* ---------------- domain ---------------- */
  function renderDomain(view, id, crumbs) {
    const d = A.domainMap[id];
    if (!d) return notFound(view);
    document.title = d.title + ' · Atlas';
    view.style.setProperty('--dc', d.color);
    crumb(crumbs, [{ label: 'Atlas', href: '#/' }, { label: d.title }]);
    const nd = d.topics.filter(t => isDone(t.id)).length;
    h('header', 'dhead', view, `
      <div class="sym">${esc(d.symbol)}</div>
      <h1>${esc(d.title)}</h1>
      <p>${esc(d.blurb)}</p>
      <div class="chips">
        <span class="chip dc"><b>${d.topics.length}</b> topics</span>
        <span class="chip">All of Statistics · ${esc(d.aos)}</span>
        <span class="chip">Casella &amp; Berger · ${esc(d.cb)}</span>
        <span class="chip">${nd} of ${d.topics.length} understood</span>
      </div>`);
    if (d.intro) math(h('div', 'prose', view, d.intro));
    const list = h('div', 'topic-list', view);
    d.topics.forEach((t, i) => {
      const a = h('a', 'topic-row' + (isDone(t.id) ? ' done' : ''), list);
      a.href = '#/t/' + t.id;
      const nl = labCount(t);
      a.innerHTML = `
        <div class="num">${isDone(t.id) ? '✓' : String(i + 1).padStart(2, '0')}</div>
        <div><h3>${esc(t.title)}</h3><p>${esc(t.subtitle || '')}</p></div>
        <div class="side">${levelHtml(t.level)}${nl ? `<span class="badge-lab">${nl} lab${nl > 1 ? 's' : ''}</span>` : ''}</div>`;
    });
    if (!d.topics.length) h('div', 'empty-state', list, '<h2>Under construction</h2><p>Topics for this field have not been written yet.</p>');
    math(list);
  }

  /* ---------------- topic ---------------- */
  function renderTopic(view, id, crumbs, anchor) {
    const t = A.topicMap[id];
    if (!t) return notFound(view);
    const d = A.domainMap[t.domain];
    document.title = t.title + ' · Atlas';
    view.style.setProperty('--dc', d.color);
    crumb(crumbs, [{ label: 'Atlas', href: '#/' }, { label: d.title, href: '#/d/' + d.id }, { label: t.title }]);

    const wrap = h('div', 'topic-wrap', view);
    const art = h('article', 'article', wrap);
    const toc = h('nav', 'toc', wrap);

    const src = [];
    if (t.sources && t.sources.aos) src.push(`<span class="chip" title="Wasserman, All of Statistics">AoS ${esc(t.sources.aos)}</span>`);
    if (t.sources && t.sources.cb) src.push(`<span class="chip" title="Casella & Berger, Statistical Inference">C&amp;B ${esc(t.sources.cb)}</span>`);
    if (t.sources && t.sources.other) src.push(`<span class="chip">${esc(t.sources.other)}</span>`);
    const head = h('header', 'thead', art, `
      <div class="kicker"><span>${esc(d.symbol)}</span><a href="#/d/${d.id}" style="color:inherit">${esc(d.title)}</a></div>
      <h1>${t.title}</h1>
      ${t.subtitle ? `<p class="subtitle">${t.subtitle}</p>` : ''}
      <div class="chips">${levelHtml(t.level)} ${src.join(' ')} ${t.tags.map(x => `<span class="chip">${esc(x)}</span>`).join(' ')}</div>`);
    math(head);

    if (t.summary) math(h('div', 'lede', art, t.summary));

    const path = A.pathTo(t.id).filter(x => x.id !== t.id);
    if (path.length) {
      h('div', 'pathbox', art, `<b>Path to here</b> — the prerequisite chain in study order:
        <div class="steps">${path.map((p, i) => `${i ? '<span class="arrow">→</span>' : ''}<a class="chip${isDone(p.id) ? '' : ''}" href="#/t/${p.id}" style="--dc:${A.domainMap[p.domain].color}">${isDone(p.id) ? '✓ ' : ''}${esc(p.title)}</a>`).join('')}</div>`);
    }

    const prose = h('div', 'prose', art);
    const tocItems = [];
    t.sections.forEach((s, i) => {
      const sid = 's' + (i + 1);
      const h2 = h('h2', null, prose, `<span class="secnum">${String(i + 1).padStart(2, '0')}</span><span>${s.title}</span>`);
      h2.id = sid;
      tocItems.push({ id: sid, title: s.title });
      const body = h('div', 'sec', prose, s.html || '');
      if (s.lab) h('div', 'lab', body).dataset.lab = s.lab;
    });
    // Proofs that directly follow a theorem attach visually to it.
    for (const p of prose.querySelectorAll('details.proof')) {
      const prev = p.previousElementSibling;
      if (prev && prev.classList.contains('thm')) p.classList.add('after');
    }
    math(prose);
    mountLabs(prose);

    if (t.quiz.length) {
      const qi = t.sections.length + 1;
      const h2 = h('h2', null, prose, `<span class="secnum">${String(qi).padStart(2, '0')}</span><span>Check yourself</span>`);
      h2.id = 'quiz';
      tocItems.push({ id: 'quiz', title: 'Check yourself' });
      const qz = h('div', 'quiz', prose);
      t.quiz.forEach((q, i) => {
        const c = h('div', 'qcard', qz, `<div class="q"><b>Q${i + 1}</b>${q.q}</div><details><summary>Reveal answer</summary><div class="a">${q.a}</div></details>`);
        math(c);
      });
    }

    const conn = [];
    for (const p of t.prereqs) if (A.topicMap[p]) conn.push(['Builds on', A.topicMap[p]]);
    for (const p of t.unlocks) if (A.topicMap[p]) conn.push(['Leads to', A.topicMap[p]]);
    for (const p of t.related) if (A.topicMap[p]) conn.push(['Related', A.topicMap[p]]);
    if (conn.length) {
      const h2 = h('h2', null, prose, `<span class="secnum">→</span><span>Connections</span>`);
      h2.id = 'connections';
      tocItems.push({ id: 'connections', title: 'Connections' });
      h('div', 'connect', prose, conn.map(([k, x]) => `<a href="#/t/${x.id}" style="--dc:${A.domainMap[x.domain].color}"><div class="k">${k} · ${esc(A.domainMap[x.domain].title)}</div><div class="t">${x.title}</div></a>`).join(''));
      math(prose.lastElementChild);
    }

    // footer: mark understood + prev/next within domain
    const foot = h('div', 'tfoot', art);
    const mb = h('button', 'btn mark-btn', foot);
    const paintMark = () => { mb.classList.toggle('done', isDone(t.id)); mb.innerHTML = isDone(t.id) ? '✓ Marked as understood' : '○ Mark as understood'; };
    paintMark();
    mb.addEventListener('click', () => { setDone(t.id, !isDone(t.id)); paintMark(); });
    const idx = d.topics.indexOf(t);
    const pn = h('div', 'pn', foot);
    if (idx > 0) h('a', null, pn, `<small>← Previous</small>${d.topics[idx - 1].title}`).href = '#/t/' + d.topics[idx - 1].id;
    if (idx < d.topics.length - 1) h('a', null, pn, `<small>Next →</small>${d.topics[idx + 1].title}`).href = '#/t/' + d.topics[idx + 1].id;
    math(pn);

    // TOC with scroll-spy
    toc.innerHTML = `<h5>On this page</h5>` + tocItems.map(x => `<a href="#/t/${t.id}/${x.id}" data-id="${x.id}">${x.title}</a>`).join('') +
      `<div class="toc-meta">
        ${t.sources && t.sources.aos ? `<div><b>All of Statistics</b>${esc(t.sources.aos)}</div>` : ''}
        ${t.sources && t.sources.cb ? `<div><b>Casella &amp; Berger</b>${esc(t.sources.cb)}</div>` : ''}
        ${t.sources && t.sources.other ? `<div><b>Further</b>${esc(t.sources.other)}</div>` : ''}
      </div>`;
    math(toc);
    toc.addEventListener('click', e => {
      const a = e.target.closest('a[data-id]');
      if (!a) return;
      e.preventDefault();
      const target = document.getElementById(a.dataset.id);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      history.replaceState(null, '', a.getAttribute('href'));
    });
    const links = [...toc.querySelectorAll('a[data-id]')];
    const heads = tocItems.map(x => document.getElementById(x.id));
    const spy = () => {
      let cur = heads[0] && heads[0].id;
      for (const hd of heads) if (hd && hd.getBoundingClientRect().top < 140) cur = hd.id;
      for (const a of links) a.classList.toggle('on', a.dataset.id === cur);
    };
    window.addEventListener('scroll', spy, { passive: true });
    viewCleanup.push(() => window.removeEventListener('scroll', spy));
    spy();
    if (anchor) setTimeout(() => { const el = document.getElementById(anchor); if (el) el.scrollIntoView({ block: 'start' }); }, 50);
  }

  function mountLabs(root) {
    for (const holder of root.querySelectorAll('[data-lab]')) mountLab(holder, holder.dataset.lab);
  }

  function mountLab(holder, labId) {
    const def = A.labs[labId];
    holder.classList.add('lab');
    holder.innerHTML = '';
    if (!def) { h('div', 'lab-body lab-error', holder, `Lab "${esc(labId)}" is not registered.`); return; }
    const head = h('div', 'lab-head', holder, `<span class="tag">Lab</span><h4>${def.title || labId}</h4>`);
    math(head);
    let seed = (Math.random() * 2 ** 31) >>> 0;
    const reset = A.UI.button(head, '↺ Reset', () => run(true), { small: true });
    reset.classList.add('btn-ghost');
    reset.title = 'Re-run with a fresh random seed';
    if (def.blurb) math(h('div', 'lab-blurb', holder, def.blurb));
    const body = h('div', 'lab-body', holder);
    let ctx = null;
    function run(fresh) {
      if (ctx) { ctx.destroy(); mounted = mounted.filter(c => c !== ctx); }
      if (fresh) seed = (Math.random() * 2 ** 31) >>> 0;
      body.innerHTML = '';
      body.className = 'lab-body';
      ctx = A.UI.makeContext(body, seed);
      mounted.push(ctx);
      try { def.mount(body, ctx); }
      catch (e) { console.error(e); h('div', 'lab-error', body, `Lab error: ${esc(e.message)}`); }
    }
    // Defer mounting until the lab is near the viewport (keeps long pages fast).
    if ('IntersectionObserver' in window) {
      const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); run(false); } }, { rootMargin: '400px' });
      io.observe(holder);
      viewCleanup.push(() => io.disconnect());
    } else run(false);
  }

  /* ---------------- knowledge map ---------------- */
  function renderMap(view, crumbs) {
    crumb(crumbs, [{ label: 'Atlas', href: '#/' }, { label: 'Knowledge map' }]);
    document.title = 'Knowledge map · Atlas';
    h('div', 'section-h', view, `<h2>Knowledge map</h2><span class="aside">${A.topics.length} topics · arrows point from prerequisite to what it unlocks. Hover to trace a neighbourhood, click to open.</span>`);
    const wrap = h('div', 'map-wrap', view);
    if (!window.d3) { wrap.innerHTML = '<div class="empty-state">d3 failed to load.</div>'; return; }
    const d3 = window.d3;
    const W = wrap.clientWidth, H = wrap.clientHeight;
    const svg = d3.select(wrap).append('svg').attr('viewBox', [-W / 2, -H / 2, W, H]);
    const defs = svg.append('defs');
    defs.append('marker').attr('id', 'map-arrow').attr('viewBox', '0 -4 8 8').attr('refX', 15).attr('refY', 0).attr('markerWidth', 6).attr('markerHeight', 6).attr('orient', 'auto')
      .append('path').attr('d', 'M0,-4L8,0L0,4').attr('fill', 'var(--faint)');
    const g = svg.append('g');
    svg.call(d3.zoom().scaleExtent([0.25, 4]).on('zoom', e => g.attr('transform', e.transform)));

    const nD = A.domains.length;
    const R = Math.min(W, H) * 0.36;
    const center = {};
    A.domains.forEach((d, i) => { const a = -Math.PI / 2 + 2 * Math.PI * i / nD; center[d.id] = [R * Math.cos(a), R * Math.sin(a)]; });
    const nodes = A.topics.map(t => ({ id: t.id, t, d: t.domain, x: center[t.domain][0] + (Math.random() - 0.5) * 40, y: center[t.domain][1] + (Math.random() - 0.5) * 40 }));
    const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
    const links = [];
    for (const t of A.topics) for (const p of t.prereqs) if (byId[p]) links.push({ source: byId[p], target: byId[t.id] });
    const deg = {};
    for (const l of links) { deg[l.source.id] = (deg[l.source.id] || 0) + 1; deg[l.target.id] = (deg[l.target.id] || 0) + 1; }
    const rad = n => 5 + Math.min(7, Math.sqrt(deg[n.id] || 0) * 1.8);

    const hullG = g.append('g');
    const linkSel = g.append('g').selectAll('path').data(links).join('path').attr('class', 'map-link').attr('marker-end', 'url(#map-arrow)');
    const nodeSel = g.append('g').selectAll('g').data(nodes).join('g').attr('class', 'map-node')
      .on('click', (e, n) => { location.hash = '#/t/' + n.id; });
    nodeSel.append('circle').attr('r', rad).attr('fill', n => A.domainMap[n.d].color);
    nodeSel.append('text').attr('dy', n => rad(n) + 12).attr('text-anchor', 'middle').text(n => n.t.title.length > 30 ? n.t.title.slice(0, 28) + '…' : n.t.title)
      .each(function () { this.textContent = this.textContent.replace(/<[^>]+>/g, ''); });
    const dLabels = g.append('g').selectAll('text').data(A.domains).join('text').attr('class', 'map-dlabel')
      .attr('fill', d => d.color).text(d => d.title);

    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).distance(l => (l.source.d === l.target.d ? 40 : 120)).strength(l => (l.source.d === l.target.d ? 0.5 : 0.04)))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('collide', d3.forceCollide(n => rad(n) + 8))
      .force('x', d3.forceX(n => center[n.d][0]).strength(0.12))
      .force('y', d3.forceY(n => center[n.d][1]).strength(0.12));

    const tip = h('div', 'map-tip', wrap);
    const hulls = () => {
      const data = A.domains.map(d => ({ d, pts: nodes.filter(n => n.d === d.id).map(n => [n.x, n.y]) })).filter(x => x.pts.length);
      hullG.selectAll('path').data(data).join('path').attr('class', 'map-hull')
        .attr('fill', x => x.d.color).attr('stroke', x => x.d.color)
        .attr('d', x => {
          const pad = 26;
          const pts = x.pts.flatMap(([px, py]) => [[px - pad, py], [px + pad, py], [px, py - pad], [px, py + pad]]);
          const hull = d3.polygonHull(pts);
          return hull ? 'M' + hull.join('L') + 'Z' : '';
        });
      dLabels.attr('x', d => { const ps = nodes.filter(n => n.d === d.id); return ps.length ? d3.mean(ps, n => n.x) : center[d.id][0]; })
        .attr('y', d => { const ps = nodes.filter(n => n.d === d.id); return ps.length ? d3.min(ps, n => n.y) - 34 : center[d.id][1]; });
    };
    sim.on('tick', () => {
      linkSel.attr('d', l => `M${l.source.x},${l.source.y}L${l.target.x},${l.target.y}`);
      nodeSel.attr('transform', n => `translate(${n.x},${n.y})`);
      hulls();
    });
    nodeSel.call(d3.drag()
      .on('start', (e, n) => { if (!e.active) sim.alphaTarget(0.2).restart(); n.fx = n.x; n.fy = n.y; })
      .on('drag', (e, n) => { n.fx = e.x; n.fy = e.y; })
      .on('end', (e, n) => { if (!e.active) sim.alphaTarget(0); n.fx = null; n.fy = null; }));

    const neighbours = n => {
      const s = new Set([n.id]);
      for (const l of links) { if (l.source.id === n.id) s.add(l.target.id); if (l.target.id === n.id) s.add(l.source.id); }
      return s;
    };
    nodeSel.on('mouseenter', (e, n) => {
      const nb = neighbours(n);
      nodeSel.classed('dim', x => !nb.has(x.id));
      linkSel.classed('dim', l => l.source.id !== n.id && l.target.id !== n.id).classed('hl', l => l.source.id === n.id || l.target.id === n.id);
      tip.style.display = 'block';
      tip.innerHTML = `<b>${n.t.title}</b>${esc(A.domainMap[n.d].title)}${n.t.subtitle ? ' — ' + n.t.subtitle : ''}`;
      math(tip);
    }).on('mousemove', e => {
      const r = wrap.getBoundingClientRect();
      tip.style.left = Math.min(r.width - 290, e.clientX - r.left + 14) + 'px';
      tip.style.top = (e.clientY - r.top + 14) + 'px';
    }).on('mouseleave', () => {
      nodeSel.classed('dim', false); linkSel.classed('dim', false).classed('hl', false); tip.style.display = 'none';
    });

    // domain filter legend
    const hidden = new Set();
    const leg = h('div', 'map-legend', wrap);
    for (const d of A.domains) {
      const b = h('button', null, leg, `<i></i>${esc(d.title)}`);
      b.style.setProperty('--dc', d.color);
      b.addEventListener('click', () => {
        hidden.has(d.id) ? hidden.delete(d.id) : hidden.add(d.id);
        b.classList.toggle('off', hidden.has(d.id));
        nodeSel.style('display', n => (hidden.has(n.d) ? 'none' : null));
        linkSel.style('display', l => (hidden.has(l.source.d) || hidden.has(l.target.d) ? 'none' : null));
        hullG.selectAll('path').style('display', x => (hidden.has(x.d.id) ? 'none' : null));
        dLabels.style('display', x => (hidden.has(x.id) ? 'none' : null));
      });
    }
    h('div', 'map-hint', wrap, 'scroll to zoom · drag to pan · drag nodes');
    viewCleanup.push(() => sim.stop());
  }

  /* ---------------- survey ---------------- */
  function renderSurvey(view, crumbs) {
    crumb(crumbs, [{ label: 'Atlas', href: '#/' }, { label: 'Method survey' }]);
    document.title = 'Method survey · Atlas';
    h('div', 'section-h', view, '<h2>Method survey</h2><span class="aside">Every method side by side: what it assumes, what it gives you, and where it breaks.</span>');
    const rows = A.topics.filter(t => t.survey);
    const tools = h('div', 'survey-tools', view);
    const q = h('input', null, tools);
    q.type = 'search'; q.placeholder = 'Filter methods, assumptions, use cases…';
    const dsel = A.UI.select(tools, { options: [{ value: '', label: 'All fields' }].concat(A.domains.filter(d => rows.some(r => r.domain === d.id)).map(d => ({ value: d.id, label: d.title }))), onChange: draw });
    const wrapT = h('div', 'survey-table-wrap', view);
    const cols = [
      ['Method', null],
      ['Task', 'task'],
      ['Assumptions', 'assumptions'],
      ['Output / estimand', 'output'],
      ['Strengths', 'strengths'],
      ['Weaknesses', 'weaknesses'],
      ['Cost', 'complexity'],
    ];
    function draw() {
      const s = q.value.trim().toLowerCase(), dv = dsel.get();
      const filt = rows.filter(t => (!dv || t.domain === dv) && (!s || (t.title + ' ' + Object.values(t.survey).join(' ') + ' ' + t.tags.join(' ')).toLowerCase().includes(s)));
      wrapT.innerHTML = `<table class="survey"><thead><tr>${cols.map(c => `<th>${c[0]}</th>`).join('')}</tr></thead><tbody>${
        filt.map(t => `<tr style="--dc:${A.domainMap[t.domain].color}"><td class="m"><a href="#/t/${t.id}">${t.title}</a><br><span class="chip dc">${esc(A.domainMap[t.domain].title)}</span></td>${
          cols.slice(1).map(c => `<td>${t.survey[c[1]] || '—'}</td>`).join('')}</tr>`).join('')
      }</tbody></table>`;
      if (!filt.length) wrapT.innerHTML = '<div class="empty-state">No methods match.</div>';
      math(wrapT);
    }
    q.addEventListener('input', draw);
    draw();
  }

  /* ---------------- progress ---------------- */
  function renderProgress(view, crumbs) {
    crumb(crumbs, [{ label: 'Atlas', href: '#/' }, { label: 'Progress' }]);
    document.title = 'Progress · Atlas';
    const n = A.topics.filter(t => isDone(t.id)).length;
    h('div', 'section-h', view, `<h2>Your progress</h2><span class="aside">${n} of ${A.topics.length} topics marked understood. Stored in this browser only.</span>`);
    const grid = h('div', 'prog-grid', view);
    for (const d of A.domains) {
      const nd = d.topics.filter(t => isDone(t.id)).length;
      const c = h('div', 'prog-card', grid, `<h4>${esc(d.title)}<span>${nd}/${d.topics.length}</span></h4>
        <div class="progress"><i style="width:${d.topics.length ? 100 * nd / d.topics.length : 0}%"></i></div>
        <ul>${d.topics.map(t => `<li class="${isDone(t.id) ? 'done' : ''}"><a href="#/t/${t.id}">${t.title}</a></li>`).join('')}</ul>`);
      c.style.setProperty('--dc', d.color);
      math(c);
    }
    const tools = h('div', 'btn-row', view);
    tools.style.marginTop = '24px';
    A.UI.button(tools, 'Reset all progress', () => { done = new Set(); store.set('done', []); route(); });
  }

  function notFound(view) {
    h('div', 'empty-state', view, '<h2>Not in the Atlas (yet)</h2><p>That page does not exist. <a href="#/">Return home</a>.</p>');
  }

  /* ---------------- command palette search ---------------- */
  const index = [];
  function buildIndex() {
    const strip = s => (s || '').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
    for (const d of A.domains) index.push({ kind: 'domain', href: '#/d/' + d.id, title: d.title, sub: d.blurb, sym: d.symbol, color: d.color, hay: (d.title + ' ' + d.blurb).toLowerCase(), body: '' });
    for (const t of A.topics) {
      const d = A.domainMap[t.domain];
      if (!d) continue;
      index.push({ kind: 'topic', href: '#/t/' + t.id, title: strip(t.title), sub: d.title + ' · ' + strip(t.subtitle), sym: d.symbol, color: d.color,
        hay: (strip(t.title) + ' ' + t.tags.join(' ') + ' ' + strip(t.subtitle)).toLowerCase(),
        body: (strip(t.summary) + ' ' + t.sections.map(s => strip(s.title) + ' ' + strip(s.html)).join(' ')).toLowerCase() });
    }
  }
  function search(qs) {
    const terms = qs.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return index.filter(x => x.kind === 'domain');
    const scored = [];
    for (const it of index) {
      let s = 0, ok = true;
      for (const term of terms) {
        const inT = it.title.toLowerCase().indexOf(term);
        const inH = it.hay.indexOf(term);
        const inB = it.body.indexOf(term);
        if (inT === 0) s += 12; else if (inT > 0) s += 8; else if (inH >= 0) s += 4; else if (inB >= 0) s += 1; else { ok = false; break; }
      }
      if (ok) scored.push([s + (it.kind === 'domain' ? 1 : 0), it]);
    }
    return scored.sort((a, b) => b[0] - a[0]).slice(0, 30).map(x => x[1]);
  }
  function setupPalette() {
    const back = $('#palette');
    const input = $('#palette input');
    const res = $('#palette .results');
    let items = [], sel = 0;
    const hl = (s, q) => { let o = esc(s); for (const term of q.toLowerCase().split(/\s+/).filter(x => x.length > 1)) o = o.replace(new RegExp('(' + term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>'); return o; };
    function draw() {
      items = search(input.value);
      sel = Math.min(sel, Math.max(0, items.length - 1));
      res.innerHTML = items.length ? items.map((it, i) => `<div class="r${i === sel ? ' sel' : ''}" data-i="${i}" style="--dc:${it.color}"><div class="sym">${esc(it.sym)}</div><div><div class="t">${hl(it.title, input.value)}</div><div class="s">${esc(it.sub)}</div></div></div>`).join('')
        : '<div class="empty">Nothing found.</div>';
      const s = res.querySelector('.sel');
      if (s) s.scrollIntoView({ block: 'nearest' });
    }
    function open() { back.classList.add('open'); input.value = ''; sel = 0; draw(); setTimeout(() => input.focus(), 10); }
    function close() { back.classList.remove('open'); }
    function go(i) { const it = items[i]; if (it) { close(); location.hash = it.href; } }
    input.addEventListener('input', () => { sel = 0; draw(); });
    input.addEventListener('keydown', e => {
      if (e.key === 'ArrowDown') { sel = Math.min(items.length - 1, sel + 1); draw(); e.preventDefault(); }
      else if (e.key === 'ArrowUp') { sel = Math.max(0, sel - 1); draw(); e.preventDefault(); }
      else if (e.key === 'Enter') go(sel);
      else if (e.key === 'Escape') close();
    });
    res.addEventListener('click', e => { const r = e.target.closest('.r'); if (r) go(+r.dataset.i); });
    back.addEventListener('click', e => { if (e.target === back) close(); });
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); open(); }
      else if (e.key === '/' && !/input|textarea|select/i.test(document.activeElement.tagName)) { e.preventDefault(); open(); }
    });
    $('#search-btn').addEventListener('click', open);
  }

  /* ---------------- boot ---------------- */
  function boot() {
    A.link();
    applyTheme(store.get('theme', 'system'));
    $('#theme-btn').addEventListener('click', cycleTheme);
    $('#menu-btn').addEventListener('click', () => document.body.classList.toggle('nav-open'));
    document.addEventListener('click', e => { if (document.body.classList.contains('nav-open') && !e.target.closest('#sidebar') && !e.target.closest('#menu-btn')) document.body.classList.remove('nav-open'); });
    window.addEventListener('scroll', () => $('#topbar').classList.toggle('scrolled', window.scrollY > 8), { passive: true });
    buildIndex();
    setupPalette();
    window.addEventListener('hashchange', () => {
      const r = parseRoute();
      // Same topic, only the section anchor changed: scroll, do not re-render.
      if (r.view === 'topic' && r.anchor && A.topicMap[r.id] && document.title.startsWith(A.topicMap[r.id].title)) {
        const el = document.getElementById(r.anchor);
        if (el) { el.scrollIntoView({ behavior: 'smooth' }); return; }
      }
      route();
    });
    route();
    if (A.problems.length) console.warn(`[Atlas] ${A.problems.length} content problem(s) — see above.`);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
