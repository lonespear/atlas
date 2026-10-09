/* Atlas.S — the numerical toolkit every lab shares.
 *
 * Random generation is always through a seeded generator, so a lab can be
 * reset to reproduce exactly the same draws:
 *
 *   const r = Atlas.S.rng(42);
 *   r.normal(0, 1); r.gamma(2, 1); r.poisson(3.5); ...
 *
 * Densities / CDFs / quantiles live under S.pdf, S.cdf, S.q (by family name).
 * Matrices are plain arrays of row arrays.
 */
(function () {
  'use strict';
  const S = {};
  const LN_SQRT_2PI = 0.5 * Math.log(2 * Math.PI);

  /* ---------- special functions ---------- */

  const LANCZOS = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];

  function lgamma(x) {
    if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
    x -= 1;
    let a = LANCZOS[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += LANCZOS[i] / (x + i);
    return LN_SQRT_2PI + (x + 0.5) * Math.log(t) - t + Math.log(a);
  }
  const gammaFn = x => (x < 0.5 ? Math.PI / (Math.sin(Math.PI * x) * gammaFn(1 - x)) : Math.exp(lgamma(x)));
  const lbeta = (a, b) => lgamma(a) + lgamma(b) - lgamma(a + b);
  const lchoose = (n, k) => lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);

  // Regularised lower incomplete gamma P(a, x).
  function gammaP(a, x) {
    if (x <= 0) return 0;
    const lead = -x + a * Math.log(x) - lgamma(a);
    if (x < a + 1) {
      let ap = a, del = 1 / a, sum = del;
      for (let n = 0; n < 1000; n++) {
        ap += 1; del *= x / ap; sum += del;
        if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
      }
      return sum * Math.exp(lead);
    }
    const TINY = 1e-300;
    let b = x + 1 - a, c = 1 / TINY, d = 1 / b, h = d;
    for (let i = 1; i < 1000; i++) {
      const an = -i * (i - a);
      b += 2;
      d = an * d + b; if (Math.abs(d) < TINY) d = TINY;
      c = b + an / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return 1 - Math.exp(lead) * h;
  }

  function betacf(a, b, x) {
    const TINY = 1e-300, qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < TINY) d = TINY;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= 1000; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < TINY) d = TINY;
      c = 1 + aa / c; if (Math.abs(c) < TINY) c = TINY;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 1e-15) break;
    }
    return h;
  }

  // Regularised incomplete beta I_x(a, b).
  function ibeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(-lbeta(a, b) + a * Math.log(x) + b * Math.log(1 - x));
    return x < (a + 1) / (a + b + 2) ? bt * betacf(a, b, x) / a : 1 - bt * betacf(b, a, 1 - x) / b;
  }

  const erf = x => (x >= 0 ? gammaP(0.5, x * x) : -gammaP(0.5, x * x));

  // Acklam's inverse normal CDF with one Halley refinement step (|err| < 1e-12).
  function qnorm(p) {
    if (p <= 0) return -Infinity;
    if (p >= 1) return Infinity;
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const plow = 0.02425;
    let x;
    if (p < plow) {
      const q = Math.sqrt(-2 * Math.log(p));
      x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p <= 1 - plow) {
      const q = p - 0.5, r = q * q;
      x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    } else {
      const q = Math.sqrt(-2 * Math.log(1 - p));
      x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    const e = 0.5 * (1 + erf(x / Math.SQRT2)) - p;
    const u = e * Math.sqrt(2 * Math.PI) * Math.exp(x * x / 2);
    return x - u / (1 + x * u / 2);
  }

  // Invert a monotone CDF by bisection on [lo, hi] (expands hi if needed).
  function invert(cdf, p, lo, hi) {
    let guard = 0;
    while (cdf(hi) < p && guard++ < 200) hi = hi * 2 + 1;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (cdf(mid) < p) lo = mid; else hi = mid;
      if (hi - lo < 1e-12 * (1 + Math.abs(mid))) break;
    }
    return (lo + hi) / 2;
  }

  Object.assign(S, { lgamma, gamma: gammaFn, lbeta, lchoose, gammaP, ibeta, erf, qnorm, invert });

  /* ---------- densities, CDFs, quantiles ---------- */

  const pdf = {
    normal: (x, mu = 0, sd = 1) => Math.exp(-0.5 * ((x - mu) / sd) ** 2 - LN_SQRT_2PI) / sd,
    uniform: (x, a = 0, b = 1) => (x >= a && x <= b ? 1 / (b - a) : 0),
    exp: (x, rate = 1) => (x < 0 ? 0 : rate * Math.exp(-rate * x)),
    gamma: (x, shape, scale = 1) => (x < 0 ? 0 : x === 0 ? (shape === 1 ? 1 / scale : shape < 1 ? Infinity : 0)
      : Math.exp((shape - 1) * Math.log(x) - x / scale - lgamma(shape) - shape * Math.log(scale))),
    beta: (x, a, b) => (x < 0 || x > 1 ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lbeta(a, b))),
    t: (x, df) => Math.exp(lgamma((df + 1) / 2) - lgamma(df / 2) - 0.5 * Math.log(df * Math.PI) - (df + 1) / 2 * Math.log(1 + x * x / df)),
    chisq: (x, k) => pdf.gamma(x, k / 2, 2),
    cauchy: (x, loc = 0, scale = 1) => 1 / (Math.PI * scale * (1 + ((x - loc) / scale) ** 2)),
    laplace: (x, mu = 0, b = 1) => Math.exp(-Math.abs(x - mu) / b) / (2 * b),
    logistic: (x, mu = 0, s = 1) => { const e = Math.exp(-(x - mu) / s); return e / (s * (1 + e) ** 2); },
    lognormal: (x, mu = 0, sd = 1) => (x <= 0 ? 0 : pdf.normal(Math.log(x), mu, sd) / x),
    weibull: (x, k, lam = 1) => (x < 0 ? 0 : (k / lam) * (x / lam) ** (k - 1) * Math.exp(-((x / lam) ** k))),
    pareto: (x, alpha, xm = 1) => (x < xm ? 0 : alpha * xm ** alpha / x ** (alpha + 1)),
    // Generalised extreme value, shape xi (xi -> 0 is Gumbel).
    gev: (x, mu = 0, sigma = 1, xi = 0) => {
      const z = (x - mu) / sigma;
      if (Math.abs(xi) < 1e-9) return Math.exp(-z - Math.exp(-z)) / sigma;
      const s = 1 + xi * z;
      if (s <= 0) return 0;
      const tt = s ** (-1 / xi);
      return tt ** (xi + 1) * Math.exp(-tt) / sigma;
    },
    // Generalised Pareto (threshold 0), shape xi.
    gpd: (y, sigma = 1, xi = 0) => {
      if (y < 0) return 0;
      if (Math.abs(xi) < 1e-9) return Math.exp(-y / sigma) / sigma;
      const s = 1 + xi * y / sigma;
      return s <= 0 ? 0 : s ** (-1 / xi - 1) / sigma;
    },
    mvnormal2: (x, y, mx, my, sx, sy, rho) => {
      const zx = (x - mx) / sx, zy = (y - my) / sy, om = 1 - rho * rho;
      return Math.exp(-(zx * zx - 2 * rho * zx * zy + zy * zy) / (2 * om)) / (2 * Math.PI * sx * sy * Math.sqrt(om));
    },
  };

  const pmf = {
    binom: (k, n, p) => (k < 0 || k > n || k !== Math.floor(k) ? 0 : Math.exp(lchoose(n, k) + k * Math.log(p) + (n - k) * Math.log(1 - p))),
    poisson: (k, lam) => (k < 0 || k !== Math.floor(k) ? 0 : Math.exp(k * Math.log(lam) - lam - lgamma(k + 1))),
    geom: (k, p) => (k < 1 ? 0 : p * (1 - p) ** (k - 1)), // trials until first success
    negbinom: (k, r, p) => (k < 0 ? 0 : Math.exp(lgamma(k + r) - lgamma(k + 1) - lgamma(r) + r * Math.log(p) + k * Math.log(1 - p))),
    bernoulli: (k, p) => (k === 1 ? p : k === 0 ? 1 - p : 0),
  };

  const cdf = {
    normal: (x, mu = 0, sd = 1) => 0.5 * (1 + erf((x - mu) / (sd * Math.SQRT2))),
    uniform: (x, a = 0, b = 1) => Math.min(1, Math.max(0, (x - a) / (b - a))),
    exp: (x, rate = 1) => (x < 0 ? 0 : 1 - Math.exp(-rate * x)),
    gamma: (x, shape, scale = 1) => gammaP(shape, x / scale),
    beta: (x, a, b) => ibeta(x, a, b),
    chisq: (x, k) => gammaP(k / 2, x / 2),
    t: (x, df) => { const p = 0.5 * ibeta(df / (df + x * x), df / 2, 0.5); return x > 0 ? 1 - p : p; },
    f: (x, d1, d2) => (x <= 0 ? 0 : ibeta(d1 * x / (d1 * x + d2), d1 / 2, d2 / 2)),
    cauchy: (x, loc = 0, scale = 1) => 0.5 + Math.atan((x - loc) / scale) / Math.PI,
    logistic: (x, mu = 0, s = 1) => 1 / (1 + Math.exp(-(x - mu) / s)),
    weibull: (x, k, lam = 1) => (x < 0 ? 0 : 1 - Math.exp(-((x / lam) ** k))),
    pareto: (x, alpha, xm = 1) => (x < xm ? 0 : 1 - (xm / x) ** alpha),
    gev: (x, mu = 0, sigma = 1, xi = 0) => {
      const z = (x - mu) / sigma;
      if (Math.abs(xi) < 1e-9) return Math.exp(-Math.exp(-z));
      const s = 1 + xi * z;
      if (s <= 0) return xi > 0 ? 0 : 1;
      return Math.exp(-(s ** (-1 / xi)));
    },
    gpd: (y, sigma = 1, xi = 0) => {
      if (y <= 0) return 0;
      if (Math.abs(xi) < 1e-9) return 1 - Math.exp(-y / sigma);
      const s = 1 + xi * y / sigma;
      return s <= 0 ? 1 : 1 - s ** (-1 / xi);
    },
    binom: (k, n, p) => { let s = 0; for (let i = 0; i <= Math.floor(k); i++) s += pmf.binom(i, n, p); return Math.min(1, s); },
    poisson: (k, lam) => (k < 0 ? 0 : 1 - gammaP(Math.floor(k) + 1, lam)),
  };

  const q = {
    normal: (p, mu = 0, sd = 1) => mu + sd * qnorm(p),
    exp: (p, rate = 1) => -Math.log(1 - p) / rate,
    t: (p, df) => (p === 0.5 ? 0 : p < 0.5 ? -q.t(1 - p, df) : invert(x => cdf.t(x, df), p, 0, 10)),
    chisq: (p, k) => invert(x => cdf.chisq(x, k), p, 0, k + 10),
    gamma: (p, shape, scale = 1) => scale * invert(x => gammaP(shape, x), p, 0, shape + 10),
    beta: (p, a, b) => invert(x => ibeta(x, a, b), p, 0, 1),
    f: (p, d1, d2) => invert(x => cdf.f(x, d1, d2), p, 0, 10),
    cauchy: (p, loc = 0, scale = 1) => loc + scale * Math.tan(Math.PI * (p - 0.5)),
    logistic: (p, mu = 0, s = 1) => mu + s * Math.log(p / (1 - p)),
    weibull: (p, k, lam = 1) => lam * (-Math.log(1 - p)) ** (1 / k),
    pareto: (p, alpha, xm = 1) => xm * (1 - p) ** (-1 / alpha),
    gev: (p, mu = 0, sigma = 1, xi = 0) => (Math.abs(xi) < 1e-9 ? mu - sigma * Math.log(-Math.log(p))
      : mu + sigma * ((-Math.log(p)) ** (-xi) - 1) / xi),
    gpd: (p, sigma = 1, xi = 0) => (Math.abs(xi) < 1e-9 ? -sigma * Math.log(1 - p) : sigma * ((1 - p) ** (-xi) - 1) / xi),
  };

  Object.assign(S, { pdf, pmf, cdf, q });

  /* ---------- seeded random generation ---------- */

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function rng(seed) {
    if (seed == null) seed = (Math.random() * 2 ** 32) >>> 0;
    const u = mulberry32(seed);
    let spare = null;
    const R = {
      seed,
      uniform: (a = 0, b = 1) => a + (b - a) * u(),
      int: (lo, hi) => lo + Math.floor(u() * (hi - lo + 1)), // inclusive
      normal(mu = 0, sd = 1) {
        if (spare !== null) { const z = spare; spare = null; return mu + sd * z; }
        let x, y, s;
        do { x = 2 * u() - 1; y = 2 * u() - 1; s = x * x + y * y; } while (s >= 1 || s === 0);
        const f = Math.sqrt(-2 * Math.log(s) / s);
        spare = y * f;
        return mu + sd * x * f;
      },
      exp: (rate = 1) => -Math.log(1 - u()) / rate,
      gamma(shape, scale = 1) { // Marsaglia–Tsang
        if (shape < 1) return R.gamma(shape + 1, scale) * u() ** (1 / shape);
        const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
        for (;;) {
          let x, v;
          do { x = R.normal(); v = 1 + c * x; } while (v <= 0);
          v = v * v * v;
          const uu = u();
          if (uu < 1 - 0.0331 * x ** 4 || Math.log(uu) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v * scale;
        }
      },
      beta(a, b) { const x = R.gamma(a), y = R.gamma(b); return x / (x + y); },
      chisq: k => 2 * R.gamma(k / 2),
      t: df => R.normal() / Math.sqrt(R.chisq(df) / df),
      cauchy: (loc = 0, scale = 1) => loc + scale * Math.tan(Math.PI * (u() - 0.5)),
      laplace: (mu = 0, b = 1) => { const v = u() - 0.5; return mu - b * Math.sign(v) * Math.log(1 - 2 * Math.abs(v)); },
      logistic: (mu = 0, s = 1) => { const v = u(); return mu + s * Math.log(v / (1 - v)); },
      lognormal: (mu = 0, sd = 1) => Math.exp(R.normal(mu, sd)),
      weibull: (k, lam = 1) => lam * (-Math.log(1 - u())) ** (1 / k),
      pareto: (alpha, xm = 1) => xm * (1 - u()) ** (-1 / alpha),
      gumbel: (mu = 0, beta = 1) => mu - beta * Math.log(-Math.log(u())),
      gev: (mu = 0, sigma = 1, xi = 0) => q.gev(u(), mu, sigma, xi),
      gpd: (sigma = 1, xi = 0) => q.gpd(u(), sigma, xi),
      bernoulli: p => (u() < p ? 1 : 0),
      binomial(n, p) {
        if (n <= 2000) { let k = 0; for (let i = 0; i < n; i++) if (u() < p) k++; return k; }
        return Math.max(0, Math.min(n, Math.round(R.normal(n * p, Math.sqrt(n * p * (1 - p))))));
      },
      poisson(lam) {
        if (lam < 30) { // Knuth
          const L = Math.exp(-lam);
          let k = 0, p = 1;
          do { k++; p *= u(); } while (p > L);
          return k - 1;
        }
        // Hörmann PTRS transformed rejection
        const slam = Math.sqrt(lam), loglam = Math.log(lam);
        const b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
        const invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
        for (;;) {
          const U = u() - 0.5, V = u(), us = 0.5 - Math.abs(U);
          const k = Math.floor((2 * a / us + b) * U + lam + 0.43);
          if (us >= 0.07 && V <= vr) return k;
          if (k < 0 || (us < 0.013 && V > us)) continue;
          if (Math.log(V) + Math.log(invalpha) - Math.log(a / (us * us) + b) <= -lam + k * loglam - lgamma(k + 1)) return k;
        }
      },
      geom: p => Math.ceil(Math.log(1 - u()) / Math.log(1 - p)), // trials until first success
      categorical(probs) {
        let v = u() * probs.reduce((s, x) => s + x, 0);
        for (let i = 0; i < probs.length; i++) { v -= probs[i]; if (v <= 0) return i; }
        return probs.length - 1;
      },
      choice: arr => arr[Math.floor(u() * arr.length)],
      sample(arr, n, replace = true) {
        if (replace) return Array.from({ length: n }, () => arr[Math.floor(u() * arr.length)]);
        return R.shuffle(arr.slice()).slice(0, n);
      },
      shuffle(arr) {
        for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(u() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
        return arr;
      },
      // Multivariate normal via Cholesky; Sigma is a k x k array.
      mvnormal(mu, Sigma) {
        const L = cholesky(Sigma);
        const z = mu.map(() => R.normal());
        return mu.map((m, i) => m + L[i].reduce((s, l, j) => s + l * z[j], 0));
      },
      dirichlet(alpha) { const g = alpha.map(a => R.gamma(a)); const s = sum(g); return g.map(x => x / s); },
      array: (n, f) => Array.from({ length: n }, (_, i) => f(i)),
    };
    R.u = u;
    return R;
  }
  S.rng = rng;

  /* ---------- descriptive summaries ---------- */

  function sum(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s; }
  const mean = a => sum(a) / a.length;
  function variance(a, ddof = 1) {
    const m = mean(a);
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
    return s / (a.length - ddof);
  }
  const sd = (a, ddof = 1) => Math.sqrt(variance(a, ddof));
  function quantile(a, p, sorted = false) { // type 7 (R default)
    const s = sorted ? a : a.slice().sort((x, y) => x - y);
    const h = (s.length - 1) * p, lo = Math.floor(h), hi = Math.ceil(h);
    return s[lo] + (h - lo) * (s[hi] - s[lo]);
  }
  const median = a => quantile(a, 0.5);
  function extent(a) { let lo = Infinity, hi = -Infinity; for (const x of a) { if (x < lo) lo = x; if (x > hi) hi = x; } return [lo, hi]; }
  function cov(a, b) { const ma = mean(a), mb = mean(b); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - ma) * (b[i] - mb); return s / (a.length - 1); }
  const cor = (a, b) => cov(a, b) / (sd(a) * sd(b));
  function cumsum(a) { const out = new Array(a.length); let s = 0; for (let i = 0; i < a.length; i++) { s += a[i]; out[i] = s; } return out; }
  const linspace = (a, b, n) => Array.from({ length: n }, (_, i) => a + (b - a) * i / (n - 1));
  const range = (n) => Array.from({ length: n }, (_, i) => i);
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  const logistic = x => 1 / (1 + Math.exp(-x));
  const logit = p => Math.log(p / (1 - p));
  function logsumexp(a) { const m = Math.max(...a); return m + Math.log(sum(a.map(x => Math.exp(x - m)))); }
  function argmax(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; }
  function argmin(a) { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] < a[k]) k = i; return k; }

  // Histogram with equal-width bins. Returns [{x0, x1, count, density}].
  function histogram(data, { bins = 30, lo, hi } = {}) {
    if (lo == null || hi == null) { const e = extent(data); lo = lo ?? e[0]; hi = hi ?? e[1]; }
    if (hi === lo) hi = lo + 1;
    const w = (hi - lo) / bins;
    const out = Array.from({ length: bins }, (_, i) => ({ x0: lo + i * w, x1: lo + (i + 1) * w, count: 0, density: 0 }));
    let n = 0;
    for (const x of data) {
      if (x < lo || x > hi) continue;
      out[Math.min(bins - 1, Math.floor((x - lo) / w))].count++;
      n++;
    }
    for (const b of out) b.density = n ? b.count / (data.length * w) : 0;
    return out;
  }

  // Empirical CDF as a step function x -> F_n(x).
  function ecdf(data) {
    const s = data.slice().sort((a, b) => a - b), n = s.length;
    const F = x => { let lo = 0, hi = n; while (lo < hi) { const m = (lo + hi) >> 1; if (s[m] <= x) lo = m + 1; else hi = m; } return lo / n; };
    F.sorted = s;
    return F;
  }

  const silverman = data => 0.9 * Math.min(sd(data), (quantile(data, 0.75) - quantile(data, 0.25)) / 1.34) * data.length ** -0.2;

  // Gaussian kernel density estimate; returns x -> f_hat(x).
  function kde(data, bw) {
    bw = bw || silverman(data);
    const n = data.length;
    return x => { let s = 0; for (let i = 0; i < n; i++) s += Math.exp(-0.5 * ((x - data[i]) / bw) ** 2); return s / (n * bw * Math.sqrt(2 * Math.PI)); };
  }

  Object.assign(S, { sum, mean, variance, sd, quantile, median, extent, cov, cor, cumsum, linspace, range, clamp, logistic, logit, logsumexp, argmax, argmin, histogram, ecdf, kde, silverman });

  /* ---------- linear algebra (small, dense) ---------- */

  const transpose = A => A[0].map((_, j) => A.map(r => r[j]));
  function matmul(A, B) {
    const n = A.length, m = B[0].length, k = B.length;
    const C = Array.from({ length: n }, () => new Array(m).fill(0));
    for (let i = 0; i < n; i++) for (let l = 0; l < k; l++) { const a = A[i][l]; if (a) for (let j = 0; j < m; j++) C[i][j] += a * B[l][j]; }
    return C;
  }
  const matvec = (A, v) => A.map(r => r.reduce((s, a, j) => s + a * v[j], 0));
  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  const identity = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  // Solve A x = b (b may be a vector or a matrix) by Gaussian elimination with partial pivoting.
  function solve(A, b) {
    const n = A.length, vec = !Array.isArray(b[0]);
    const M = A.map((r, i) => r.concat(vec ? [b[i]] : b[i]));
    const w = M[0].length;
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      const piv = M[c][c];
      if (Math.abs(piv) < 1e-14) throw new Error('singular matrix');
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / piv;
        if (f) for (let j = c; j < w; j++) M[r][j] -= f * M[c][j];
      }
    }
    const X = M.map((r, i) => r.slice(n).map(v => v / M[i][i]));
    return vec ? X.map(r => r[0]) : X;
  }
  const inv = A => solve(A, identity(A.length));
  function det(A) {
    const n = A.length, M = A.map(r => r.slice());
    let d = 1;
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (M[p][c] === 0) return 0;
      if (p !== c) { [M[c], M[p]] = [M[p], M[c]]; d = -d; }
      d *= M[c][c];
      for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; for (let j = c; j < n; j++) M[r][j] -= f * M[c][j]; }
    }
    return d;
  }
  function cholesky(A) {
    const n = A.length, L = Array.from({ length: n }, () => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(s, 0)) : s / L[j][j];
    }
    return L;
  }
  // Symmetric eigendecomposition (Jacobi). Returns {values, vectors} with vectors as columns, sorted descending.
  function eigSym(A) {
    const n = A.length, a = A.map(r => r.slice()), V = identity(n);
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
      if (off < 1e-20) break;
      for (let p = 0; p < n; p++) for (let r = p + 1; r < n; r++) {
        if (Math.abs(a[p][r]) < 1e-15) continue;
        const th = (a[r][r] - a[p][p]) / (2 * a[p][r]);
        const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) { const akp = a[k][p], akr = a[k][r]; a[k][p] = c * akp - s * akr; a[k][r] = s * akp + c * akr; }
        for (let k = 0; k < n; k++) { const apk = a[p][k], ark = a[r][k]; a[p][k] = c * apk - s * ark; a[r][k] = s * apk + c * ark; }
        for (let k = 0; k < n; k++) { const vkp = V[k][p], vkr = V[k][r]; V[k][p] = c * vkp - s * vkr; V[k][r] = s * vkp + c * vkr; }
      }
    }
    const idx = range(n).sort((i, j) => a[j][j] - a[i][i]);
    return { values: idx.map(i => a[i][i]), vectors: V.map(r => idx.map(i => r[i])) };
  }
  // Least squares: beta = (X'X)^{-1} X'y, with optional ridge penalty lambda (not applied to column 0 if intercept=true).
  function ols(X, y, { lambda = 0, intercept = true } = {}) {
    const Xt = transpose(X), XtX = matmul(Xt, X);
    if (lambda) for (let i = intercept ? 1 : 0; i < XtX.length; i++) XtX[i][i] += lambda;
    return solve(XtX, matvec(Xt, y));
  }

  Object.assign(S, { transpose, matmul, matvec, dot, identity, solve, inv, det, cholesky, eigSym, ols });

  /* ---------- optimisation ---------- */

  // Nelder–Mead minimiser. f: number[] -> number. Returns {x, fx, iters}.
  function nelderMead(f, x0, { step = 0.5, maxIter = 2000, tol = 1e-10 } = {}) {
    const n = x0.length;
    let simplex = [x0.slice()];
    for (let i = 0; i < n; i++) { const x = x0.slice(); x[i] += step * (Math.abs(x[i]) > 1e-8 ? Math.abs(x[i]) : 1); simplex.push(x); }
    let fs = simplex.map(f);
    let it = 0;
    for (; it < maxIter; it++) {
      const ord = range(n + 1).sort((i, j) => fs[i] - fs[j]);
      simplex = ord.map(i => simplex[i]); fs = ord.map(i => fs[i]);
      if (Math.abs(fs[n] - fs[0]) < tol * (1 + Math.abs(fs[0]))) break;
      const c = range(n).map(j => mean(simplex.slice(0, n).map(x => x[j])));
      const lin = (a, t) => c.map((cj, j) => cj + t * (a[j] - cj));
      const xr = lin(simplex[n], -1), fr = f(xr);
      if (fr < fs[0]) {
        const xe = lin(simplex[n], -2), fe = f(xe);
        if (fe < fr) { simplex[n] = xe; fs[n] = fe; } else { simplex[n] = xr; fs[n] = fr; }
      } else if (fr < fs[n - 1]) { simplex[n] = xr; fs[n] = fr; }
      else {
        const xc = lin(simplex[n], 0.5), fc = f(xc);
        if (fc < fs[n]) { simplex[n] = xc; fs[n] = fc; }
        else for (let i = 1; i <= n; i++) { simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j])); fs[i] = f(simplex[i]); }
      }
    }
    return { x: simplex[0], fx: fs[0], iters: it };
  }
  // Golden-section minimiser on [a, b].
  function golden(f, a, b, tol = 1e-9) {
    const g = (Math.sqrt(5) - 1) / 2;
    let c = b - g * (b - a), d = a + g * (b - a);
    while (Math.abs(b - a) > tol * (1 + Math.abs(a) + Math.abs(b))) {
      if (f(c) < f(d)) b = d; else a = c;
      c = b - g * (b - a); d = a + g * (b - a);
    }
    return (a + b) / 2;
  }
  // Central-difference gradient.
  const grad = (f, x, h = 1e-5) => x.map((_, i) => { const a = x.slice(), b = x.slice(); a[i] += h; b[i] -= h; return (f(a) - f(b)) / (2 * h); });

  Object.assign(S, { nelderMead, golden, grad });

  /* ---------- formatting ---------- */

  S.fmt = (x, d = 3) => {
    if (x == null || Number.isNaN(x)) return '—';
    if (!Number.isFinite(x)) return x > 0 ? '∞' : '−∞';
    const ax = Math.abs(x);
    if (ax !== 0 && (ax < 1e-3 || ax >= 1e5)) return x.toExponential(Math.max(1, d - 1)).replace('-', '−');
    return x.toFixed(d).replace('-', '−');
  };

  window.Atlas.S = S;
})();
