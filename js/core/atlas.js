/* Atlas registry — every other script registers into this.
 *
 *   Atlas.domain({...})   a top-level field (content/domains.js)
 *   Atlas.topic({...})    one article (content/<domain>.js)
 *   Atlas.lab(id, {...})  one interactive widget (js/labs/<domain>.js)
 *
 * See AUTHORING.md for the full schema. Nothing here touches the DOM.
 */
window.Atlas = (function () {
  'use strict';
  const domains = [];
  const domainMap = {};
  const topics = [];
  const topicMap = {};
  const labs = {};
  const problems = [];

  function domain(d) {
    if (domainMap[d.id]) problems.push(`duplicate domain "${d.id}"`);
    d.topics = [];
    domains.push(d);
    domainMap[d.id] = d;
  }

  function topic(t) {
    if (topicMap[t.id]) problems.push(`duplicate topic "${t.id}"`);
    t.prereqs = t.prereqs || [];
    t.related = t.related || [];
    t.tags = t.tags || [];
    t.sections = t.sections || [];
    t.quiz = t.quiz || [];
    t.level = t.level || 2;
    t.order = t.order == null ? 999 : t.order;
    topics.push(t);
    topicMap[t.id] = t;
  }

  function lab(id, def) {
    if (labs[id]) problems.push(`duplicate lab "${id}"`);
    def.id = id;
    labs[id] = def;
  }

  // Called once by app.js after every content script has loaded.
  function link() {
    for (const t of topics) {
      const d = domainMap[t.domain];
      if (!d) { problems.push(`topic "${t.id}" has unknown domain "${t.domain}"`); continue; }
      d.topics.push(t);
      for (const p of t.prereqs) if (!topicMap[p]) problems.push(`topic "${t.id}" prereq "${p}" not found`);
      for (const r of t.related) if (!topicMap[r]) problems.push(`topic "${t.id}" related "${r}" not found`);
    }
    for (const d of domains) d.topics.sort((a, b) => a.order - b.order || a.title.localeCompare(b.title));
    // Reverse edges: what builds on this topic.
    for (const t of topics) t.unlocks = [];
    for (const t of topics) for (const p of t.prereqs) if (topicMap[p]) topicMap[p].unlocks.push(t.id);
    if (problems.length) console.warn('[Atlas] content problems:\n  ' + problems.join('\n  '));
  }

  // Prerequisite closure of a topic in a valid study order (deepest first).
  function pathTo(id) {
    const seen = new Set();
    const out = [];
    (function visit(x) {
      if (seen.has(x) || !topicMap[x]) return;
      seen.add(x);
      for (const p of topicMap[x].prereqs) visit(p);
      out.push(topicMap[x]);
    })(id);
    return out;
  }

  return { domains, domainMap, topics, topicMap, labs, problems, domain, topic, lab, link, pathTo };
})();
