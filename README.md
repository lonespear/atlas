# Atlas — Statistical Theory & Learning

An interactive field guide to probability, statistical inference, and machine learning, anchored in
Wasserman's *All of Statistics* and Casella & Berger's *Statistical Inference*. Each topic is derived
from first principles, with proofs and an interactive lab you can break.

## Run

No build step. Libraries are vendored in `vendor/`, so it works offline.

```
python -m http.server 8840      # then open http://127.0.0.1:8840
```

(Opening `index.html` directly also works in most browsers.)

## Layout

| Path | What |
|---|---|
| `content/*.js` | Topics (one file per field or half-field) |
| `js/labs/*.js` | Interactive labs |
| `js/core/` | Registry, numerics (`Atlas.S`), SVG plotting (`Atlas.Plot`), controls (`Atlas.UI`) |
| `js/app.js` | Router, views, knowledge map, method survey, search |
| `tools/catalog.json` | The planned topic universe (ids, order, chapter refs) |
| `tools/scan-escapes.js` | Finds single-backslash TeX in lab string literals (`'ho'` → carriage return) |
| `tools/check.js` | Validator: schema, links, KaTeX, tag balance, lab smoke tests (`npm install` first) |

See `AUTHORING.md` to add or edit topics.
