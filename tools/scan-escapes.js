#!/usr/bin/env node
/* Finds TeX commands written with a single backslash inside ordinary ('...' / "...") JS strings
 * in lab files. There, "\rho" becomes CR + "ho", "\theta" becomes TAB + "heta", "\nu" becomes
 * newline + "u", and unknown escapes like "\mu" silently lose the backslash — all of which break
 * the math. Template literals (String.raw or `...`) are skipped.
 *
 *   node tools/scan-escapes.js
 */
'use strict';
const fs = require('fs');
const path = require('path');
const dir = path.join(__dirname, '..', 'js', 'labs');
let total = 0;
for (const f of fs.readdirSync(dir)) {
  const src = fs.readFileSync(path.join(dir, f), 'utf8');
  // Tokenise just enough to find ordinary string literals, skipping comments and template literals.
  let i = 0, line = 1;
  while (i < src.length) {
    const c = src[i];
    if (c === '\n') { line++; i++; continue; }
    if (c === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') i++; continue; }
    if (c === '/' && src[i + 1] === '*') { const j = src.indexOf('*/', i + 2); for (let k = i; k < j; k++) if (src[k] === '\n') line++; i = j + 2; continue; }
    if (c === '`') { // skip template literal (no nesting of ${} with backticks assumed)
      i++;
      while (i < src.length && src[i] !== '`') { if (src[i] === '\\') i++; if (src[i] === '\n') line++; i++; }
      i++; continue;
    }
    if (c === "'" || c === '"') {
      const start = i, startLine = line;
      i++;
      let raw = '';
      while (i < src.length && src[i] !== c && src[i] !== '\n') { if (src[i] === '\\') { raw += src[i] + src[i + 1]; i += 2; continue; } raw += src[i]; i++; }
      i++;
      // A single backslash (not "\\") followed by two or more letters looks like a TeX command.
      const m = raw.match(/(^|[^\\])\\([A-Za-z]{2,})/);
      if (m && !/^(u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2})/.test(m[2])) {
        total++;
        console.log(`${f}:${startLine}  \\${m[2]}   ${src.slice(start, Math.min(i, start + 100))}`);
      }
      continue;
    }
    i++;
  }
}
console.log(`\n${total} suspicious string(s).`);
process.exit(total ? 1 : 0);
