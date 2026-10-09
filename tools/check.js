#!/usr/bin/env node
/* Atlas content checker.
 *
 *   node tools/check.js                 check everything
 *   node tools/check.js --only evt      only report on files whose name starts with "evt"
 *   node tools/check.js --no-labs       skip the (slower) lab smoke tests
 *
 * Loads every script listed in index.html into jsdom, then checks:
 *   - schema: required fields, domain exists, level 1–3, ids are kebab-case
 *   - prereqs/related point at topics in tools/catalog.json (the planned universe)
 *   - prerequisite graph is acyclic
 *   - every data-lab / section.lab is registered
 *   - all math parses in KaTeX (title, subtitle, summary, sections, quiz, survey, lab blurbs)
 *   - stray "$" left over after rendering (unbalanced delimiters)
 *   - div / details / table / ul / ol tags are balanced
 *   - labs mount without throwing, survive poking every control, and run their loops briefly
 * Exit code 1 if any ERROR was found.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const { JSDOM, VirtualConsole } = require('jsdom');

const ROOT = path.resolve(__dirname, '..');
const args = process.argv.slice(2);
const only = args.includes('--only') ? args[args.indexOf('--only') + 1] : null;
const noLabs = args.includes('--no-labs');
const catalog = JSON.parse(fs.readFileSync(path.join(ROOT, 'tools', 'catalog.json'), 'utf8'));
const catalogIds = new Set(Object.values(catalog).flat().map(t => t.id));

const errors = [];
const warns = [];
const err = (where, msg) => errors.push(`ERROR  ${where}: ${msg}`);
const warn = (where, msg) => warns.push(`warn   ${where}: ${msg}`);

const html = fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8');
const srcs = [...html.matchAll(/<script src="([^"]+)"/g)].map(m => m[1]).filter(s => !/app\.js$/.test(s));

const vc = new VirtualConsole();
const runtimeErrors = [];
vc.on('jsdomError', e => runtimeErrors.push(e.message + (e.detail ? ' ' + (e.detail.stack || e.detail) : '')));
vc.on('error', (...a) => { if (!/KaTeX auto-render/.test(String(a[0]))) runtimeErrors.push(a.join(' ')); });
const dom = new JSDOM('<!doctype html><html><body></body></html>', { runScripts: 'outside-only', pretendToBeVisual: true, virtualConsole: vc });
const { window } = dom;

// Canvas stub (jsdom has no canvas backend).
window.HTMLCanvasElement.prototype.getContext = function () {
  const store = {};
  return new Proxy({}, {
    get(_, k) {
      if (k === 'createImageData' || k === 'getImageData') return (w, h) => ({ width: w, height: h, data: new Uint8ClampedArray(w * h * 4) });
      if (k === 'createLinearGradient' || k === 'createRadialGradient' || k === 'createPattern') return () => ({ addColorStop() {} });
      if (k === 'measureText') return s => ({ width: String(s).length * 6 });
      if (k in store) return store[k];
      return () => {};
    },
    set(_, k, v) { store[k] = v; return true; },
  });
};
window.HTMLCanvasElement.prototype.toDataURL = () => 'data:,';

const fileOf = {}; // registry key -> file
let currentFile = null;
for (const s of srcs) {
  const p = path.join(ROOT, s);
  if (!fs.existsSync(p)) { if (/content\/|labs\//.test(s)) warn(s, 'script listed in index.html does not exist yet'); else err(s, 'missing core script'); continue; }
  currentFile = s;
  const before = { t: new Set(window.Atlas ? window.Atlas.topics.map(t => t.id) : []), l: new Set(window.Atlas ? Object.keys(window.Atlas.labs) : []) };
  try { window.eval(fs.readFileSync(p, 'utf8') + `\n//# sourceURL=${s}`); }
  catch (e) { err(s, 'failed to load: ' + e.message); continue; }
  if (window.Atlas) {
    for (const t of window.Atlas.topics) if (!before.t.has(t.id)) fileOf['t:' + t.id] = s;
    for (const k of Object.keys(window.Atlas.labs)) if (!before.l.has(k)) fileOf['l:' + k] = s;
  }
}
const A = window.Atlas;
for (const p of A.problems) err('registry', p);
const inScope = file => !only || path.basename(file || '').startsWith(only);

/* ---------- schema + links ---------- */
const domainIds = new Set(A.domains.map(d => d.id));
for (const t of A.topics) {
  const f = fileOf['t:' + t.id];
  if (!inScope(f)) continue;
  const W = `${f} [${t.id}]`;
  if (!/^[a-z0-9]+(-[a-z0-9]+)*$/.test(t.id)) err(W, 'id must be kebab-case');
  if (!domainIds.has(t.domain)) err(W, `unknown domain "${t.domain}"`);
  for (const k of ['title', 'subtitle', 'summary']) if (!t[k]) err(W, `missing ${k}`);
  if (!t.sources || (!t.sources.aos && !t.sources.cb && !t.sources.other)) warn(W, 'no sources');
  if (![1, 2, 3].includes(t.level)) err(W, 'level must be 1, 2 or 3');
  if (t.sections.length < 3) warn(W, `only ${t.sections.length} sections`);
  if (t.quiz.length < 2) warn(W, `only ${t.quiz.length} quiz questions`);
  if (!t.survey) warn(W, 'no survey entry');
  if (!catalogIds.has(t.id)) warn(W, 'id not in tools/catalog.json (fine if intentional; others cannot link to it)');
  for (const p of t.prereqs) if (!catalogIds.has(p) && !A.topicMap[p]) err(W, `prereq "${p}" is not a catalog id`);
  for (const p of t.related) if (!catalogIds.has(p) && !A.topicMap[p]) err(W, `related "${p}" is not a catalog id`);
  const allHtml = t.sections.map(s => s.html || '').join('\n');
  for (const m of allHtml.matchAll(/href="#\/t\/([^"/]+)/g)) if (!catalogIds.has(m[1]) && !A.topicMap[m[1]]) err(W, `link to unknown topic "${m[1]}"`);
  const labRefs = [...allHtml.matchAll(/data-lab="([^"]+)"/g)].map(m => m[1]).concat(t.sections.filter(s => s.lab).map(s => s.lab));
  for (const l of labRefs) if (!A.labs[l]) err(W, `lab "${l}" is not registered`);
  // tag balance
  for (const tag of ['div', 'details', 'table', 'ul', 'ol', 'summary']) {
    t.sections.forEach((s, i) => {
      const h = s.html || '';
      const o = (h.match(new RegExp(`<${tag}[\\s>]`, 'g')) || []).length, c = (h.match(new RegExp(`</${tag}>`, 'g')) || []).length;
      if (o !== c) err(W, `section ${i + 1} "${s.title}": <${tag}> opened ${o}, closed ${c}`);
    });
  }
}

// cycles in prereq graph (over loaded topics)
{
  const state = {};
  const stack = [];
  const visit = id => {
    if (state[id] === 2) return;
    if (state[id] === 1) { err('prereqs', 'cycle: ' + stack.slice(stack.indexOf(id)).concat(id).join(' → ')); return; }
    state[id] = 1; stack.push(id);
    for (const p of (A.topicMap[id] || { prereqs: [] }).prereqs) if (A.topicMap[p]) visit(p);
    stack.pop(); state[id] = 2;
  };
  for (const t of A.topics) visit(t.id);
}

/* ---------- math ---------- */
const doc = window.document;
function checkMath(where, htmlStr) {
  if (!htmlStr) return;
  // "<" followed by a letter/"!"/"/" inside math is parsed by the browser as an HTML tag and eats the formula.
  for (const m of htmlStr.matchAll(/\$\$([\s\S]+?)\$\$|\$([^$]+?)\$/g)) {
    const tex = m[1] || m[2];
    const bad = tex.match(/<[A-Za-z!\/]/);
    if (bad) err(where, `"<" directly followed by "${bad[0][1]}" inside math (browser reads it as an HTML tag) — write "< ${bad[0][1]}" or \\lt: ` + JSON.stringify(tex.slice(0, 80)));
  }
  const div = doc.createElement('div');
  div.innerHTML = htmlStr;
  window.Atlas.UI.math(div);
  for (const e of div.querySelectorAll('.katex-error')) err(where, 'KaTeX: ' + (e.getAttribute('title') || e.textContent).slice(0, 220));
  // stray dollars in text nodes outside rendered math
  const walker = doc.createTreeWalker(div, window.NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    if (n.parentElement.closest('.katex, code, pre')) continue;
    if (/\$/.test(n.textContent)) { warn(where, 'stray "$" (unbalanced math?) near: ' + JSON.stringify(n.textContent.trim().slice(0, 80))); break; }
    if (/\\\(|\\\[|\\frac|\\mathbb|\\sum|\\int/.test(n.textContent)) { warn(where, 'raw TeX outside math near: ' + JSON.stringify(n.textContent.trim().slice(0, 80))); break; }
  }
}
for (const t of A.topics) {
  const f = fileOf['t:' + t.id];
  if (!inScope(f)) continue;
  const W = `${f} [${t.id}]`;
  checkMath(W + ' title', t.title);
  checkMath(W + ' subtitle', t.subtitle);
  checkMath(W + ' summary', t.summary);
  t.sections.forEach((s, i) => { checkMath(`${W} §${i + 1} title`, s.title); checkMath(`${W} §${i + 1} "${s.title}"`, s.html); });
  t.quiz.forEach((q, i) => { checkMath(`${W} quiz ${i + 1} q`, q.q); checkMath(`${W} quiz ${i + 1} a`, q.a); });
  if (t.survey) for (const [k, v] of Object.entries(t.survey)) checkMath(`${W} survey.${k}`, v);
}
for (const [id, def] of Object.entries(A.labs)) {
  if (!inScope(fileOf['l:' + id])) continue;
  checkMath(`${fileOf['l:' + id]} lab ${id} title`, def.title);
  checkMath(`${fileOf['l:' + id]} lab ${id} blurb`, def.blurb);
  if (typeof def.mount !== 'function') err(`${fileOf['l:' + id]} lab ${id}`, 'mount is not a function');
}

/* ---------- lab smoke tests ---------- */
async function smoke() {
  const ids = Object.keys(A.labs).filter(id => inScope(fileOf['l:' + id]));
  for (const id of ids) {
    const W = `${fileOf['l:' + id]} lab ${id}`;
    const el = doc.createElement('div');
    doc.body.appendChild(el);
    runtimeErrors.length = 0;
    const t0 = Date.now();
    let ctx;
    try {
      ctx = A.UI.makeContext(el, 12345);
      A.labs[id].mount(el, ctx);
    } catch (e) { err(W, 'mount threw: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' | ')); el.remove(); continue; }
    const tMount = Date.now() - t0;
    try {
      for (const inp of el.querySelectorAll('input[type=range]')) {
        for (const v of [inp.max, inp.min, (+inp.min + +inp.max) / 2]) {
          inp.value = v;
          inp.dispatchEvent(new window.Event('input', { bubbles: true }));
          inp.dispatchEvent(new window.Event('change', { bubbles: true }));
        }
      }
      for (const s of el.querySelectorAll('select')) for (const o of s.options) { s.value = o.value; s.dispatchEvent(new window.Event('change', { bubbles: true })); }
      for (const b of el.querySelectorAll('.seg-b')) b.click();
      for (const c of el.querySelectorAll('input[type=checkbox]')) { c.checked = !c.checked; c.dispatchEvent(new window.Event('change', { bubbles: true })); }
      for (const b of el.querySelectorAll('button.btn')) b.click();
    } catch (e) { err(W, 'control interaction threw: ' + (e.stack || e.message).split('\n').slice(0, 3).join(' | ')); }
    await new Promise(r => setTimeout(r, 400)); // let any started loops tick
    try { ctx.destroy(); } catch (e) { err(W, 'destroy threw: ' + e.message); }
    const total = Date.now() - t0;
    for (const m of new Set(runtimeErrors)) err(W, 'runtime: ' + m.split('\n').slice(0, 3).join(' | '));
    if (!el.querySelector('svg, canvas')) warn(W, 'rendered no svg/canvas');
    if (tMount > 1500) warn(W, `slow mount: ${tMount} ms (keep initial render < ~300 ms in a real browser)`);
    if (total > 8000) warn(W, `slow interaction sweep: ${total} ms`);
    el.remove();
  }
  return ids.length;
}

(async () => {
  const nLabs = noLabs ? 0 : await smoke();
  const nT = A.topics.filter(t => inScope(fileOf['t:' + t.id])).length;
  for (const w of warns) console.log(w);
  for (const e of errors) console.log(e);
  console.log(`\n${nT} topics, ${Object.keys(A.labs).filter(id => inScope(fileOf['l:' + id])).length} labs checked${noLabs ? ' (labs not smoke-tested)' : ` (${nLabs} smoke-tested)`}. ${errors.length} error(s), ${warns.length} warning(s).`);
  process.exit(errors.length ? 1 : 0);
})();
