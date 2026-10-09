/* The twelve fields of the Atlas, in reading order.
 * `aos` / `cb` give the chapters of the two anchor texts the field draws on:
 *   AoS = Wasserman, All of Statistics (2004)
 *   C&B = Casella & Berger, Statistical Inference, 2nd ed. (2002)
 */
(function () {
  const D = Atlas.domain;

  D({ id: 'foundations', order: 1, symbol: 'Ω', color: 'var(--d1)', title: 'Probability Foundations',
    blurb: 'Sample spaces, random variables, expectation, the inequalities that bound them, and the limit theorems that make statistics possible.',
    aos: 'Ch. 1–5', cb: 'Ch. 1–5' });

  D({ id: 'inference', order: 2, symbol: 'θ̂', color: 'var(--d2)', title: 'Statistical Inference',
    blurb: 'Estimation, sufficiency, likelihood, testing, intervals, Bayesian inference, and decision theory: the logic of learning from data.',
    aos: 'Ch. 6, 9–12', cb: 'Ch. 6–10' });

  D({ id: 'regression', order: 3, symbol: 'β', color: 'var(--d3)', title: 'Regression & Linear Models',
    blurb: 'Least squares as projection, Gauss–Markov, inference for coefficients, diagnostics, model selection, and shrinkage.',
    aos: 'Ch. 13', cb: 'Ch. 11–12' });

  D({ id: 'glm', order: 4, symbol: 'η', color: 'var(--d4)', title: 'Generalized Linear Models',
    blurb: 'Exponential families, link functions, IRLS, deviance, logistic and Poisson regression, overdispersion, and their extensions.',
    aos: 'Ch. 13.7', cb: 'Ch. 3.4, 12.3' });

  D({ id: 'nonparametric', order: 5, symbol: 'f̂', color: 'var(--d5)', title: 'Nonparametric Estimation',
    blurb: 'Estimating distributions and functions without a parametric form: empirical CDFs, density estimation, smoothing, and splines.',
    aos: 'Ch. 7, 19–21', cb: '§5.4, §10.1' });

  D({ id: 'stochastic', order: 6, symbol: 'Pᵢⱼ', color: 'var(--d6)', title: 'Stochastic Processes',
    blurb: 'Markov chains, Poisson processes, random walks, Brownian motion, martingales, and queues — randomness that unfolds in time.',
    aos: 'Ch. 23', cb: '—' });

  D({ id: 'simulation', order: 7, symbol: '∫', color: 'var(--d7)', title: 'Simulation Theory',
    blurb: 'Generating randomness, inversion and rejection, Monte Carlo integration, importance sampling, and variance reduction.',
    aos: 'Ch. 24.1–24.3', cb: '§5.6' });

  D({ id: 'mcmc', order: 8, symbol: 'π', color: 'var(--d8)', title: 'Markov Chain Monte Carlo',
    blurb: 'Building Markov chains whose stationary law is the posterior: Metropolis–Hastings, Gibbs, Hamiltonian dynamics, and diagnosing convergence.',
    aos: 'Ch. 24.4–24.6', cb: '§5.6.3' });

  D({ id: 'resampling', order: 9, symbol: 'F̂ₙ', color: 'var(--d9)', title: 'Bootstrap & Resampling',
    blurb: 'Plug-in estimation, the bootstrap and its intervals, the jackknife, permutation tests, and cross-validation.',
    aos: 'Ch. 8, 10.5', cb: '§10.1.4' });

  D({ id: 'evt', order: 10, symbol: 'ξ', color: 'var(--d10)', title: 'Extreme Value Theory',
    blurb: 'The limit laws of maxima, block maxima and the GEV, threshold exceedances and the GPD, tail indices, and return levels.',
    aos: '—', cb: '§5.4 (order statistics)' });

  D({ id: 'supervised', order: 11, symbol: 'ŷ', color: 'var(--d11)', title: 'Supervised Learning',
    blurb: 'Risk and the bias–variance trade-off, linear classifiers, trees and ensembles, kernels and SVMs, and the theory of generalization.',
    aos: 'Ch. 22', cb: '—' });

  D({ id: 'unsupervised', order: 12, symbol: '𝒞', color: 'var(--d12)', title: 'Unsupervised Learning',
    blurb: 'Clustering, mixture models and EM, dimensionality reduction, manifold learning, and graphical models.',
    aos: 'Ch. 17–18, 20', cb: '—' });

  D({ id: 'deep', order: 13, symbol: '∇', color: 'var(--d13)', title: 'Deep Learning',
    blurb: 'Neural networks from first principles: backpropagation, optimization, regularization, convolution, attention, and deep generative models.',
    aos: '—', cb: '—' });
})();
