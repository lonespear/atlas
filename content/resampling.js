/* Bootstrap & Resampling. Reference implementation of the topic schema — see AUTHORING.md. */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'bootstrap',
    domain: 'resampling',
    order: 20,
    title: 'The Nonparametric Bootstrap',
    subtitle: 'Estimating a sampling distribution by sampling from the sample',
    level: 2,
    sources: { aos: 'Ch. 8', cb: '§10.1.4', other: 'Efron & Tibshirani (1993); Hall (1992)' },
    prereqs: ['empirical-cdf'],
    related: ['jackknife', 'delta-method', 'confidence-intervals'],
    tags: ['plug-in', 'standard error', 'confidence interval', 'Monte Carlo'],
    summary: R`The statistic you care about — a median, a correlation, a ratio of eigenvalues — has a sampling distribution that depends on the unknown $F$. The bootstrap estimates that distribution by <em>plugging in</em> the empirical distribution $\hat F_n$ for $F$ and simulating. It is two approximations stacked: a statistical one ($\hat F_n \approx F$, which costs you $O(n^{-1/2})$ and cannot be bought down) and a Monte Carlo one ($B$ resamples, which you can make as small as you like).`,
    survey: {
      task: 'Standard errors, bias, and confidence intervals for (almost) any statistic',
      assumptions: R`IID data; statistic a smooth functional of $F$ (Hadamard differentiable); finite variance`,
      output: R`Bootstrap distribution of $T_n^*$; $\widehat{\se}_{\text{boot}}$; percentile / pivotal / studentized intervals`,
      strengths: 'No derivations: one recipe for any statistic. Studentized intervals are second-order accurate.',
      weaknesses: 'Fails for extremes (max), boundary parameters, heavy tails, and dependent data without modification.',
      complexity: R`$O(B \cdot \text{cost}(T_n))$, typically $B \approx 10^3$–$10^4$`,
    },
    sections: [
      {
        title: 'The problem the bootstrap solves',
        html: R`
<p>Let $X_1,\dots,X_n \iid F$ and let $T_n = g(X_1,\dots,X_n)$ be any statistic. To say how precise $T_n$ is we want its <strong>sampling distribution</strong>, or at least its standard error
$$\se_F(T_n) = \sqrt{\Var_F(T_n)}.$$
For the sample mean this is easy: $\se(\bar X_n) = \sigma/\sqrt n$, and we plug in $s$ for $\sigma$. For the sample median, $\se$ involves the density $f$ at the true median (see <a href="#/t/delta-method">the delta method</a>), which is awkward to estimate. For the correlation coefficient, a trimmed mean, or the ratio of the top two eigenvalues of a covariance matrix, there is no usable closed form.</p>
<p>The obstacle is always the same: $\Var_F(T_n)$ is a functional of the <em>unknown</em> $F$.</p>
<div class="note">
<p>Write the variance as an explicit function of the distribution: $v(F) = \Var_F(T_n)$. We cannot evaluate $v(F)$, but we have an excellent estimate of $F$ itself, the empirical CDF $\hat F_n$, which converges to $F$ uniformly (Glivenko–Cantelli). So estimate $v(F)$ by $v(\hat F_n)$. That is the whole idea. The rest is how to compute $v(\hat F_n)$ and when the substitution is justified.</p>
</div>`,
      },
      {
        title: 'Two approximations, stacked',
        html: R`
<p>The bootstrap estimate of variance involves two distinct approximations:</p>
$$\underbrace{\Var_F(T_n)}_{\text{target}} \;\overset{\text{statistical}}{\approx}\; \underbrace{\Var_{\hat F_n}(T_n)}_{\text{ideal bootstrap}} \;\overset{\text{Monte Carlo}}{\approx}\; \underbrace{v_{\text{boot}} = \frac1B\sum_{b=1}^B\Big(T^*_{n,b} - \bar T^*\Big)^2}_{\text{what we compute}}.$$
<p>The first error depends on $n$ and is the price of not knowing $F$; nothing computational can remove it. The second depends only on $B$ and goes to zero as $B\to\infty$ by the law of large numbers. In practice $B = 1000$ makes the Monte Carlo part negligible for standard errors, and $B \gtrsim 2000$ is advisable for percentile endpoints, which live in the tails.</p>
<div class="def" data-name="Bootstrap sample">
<p>A <em>bootstrap sample</em> $X_1^*,\dots,X_n^*$ is an IID draw of size $n$ from $\hat F_n$. Because $\hat F_n$ puts mass $1/n$ on each observation, this is equivalent to <strong>sampling $n$ points from the data with replacement</strong>. The bootstrap replicate of the statistic is $T_n^* = g(X_1^*,\dots,X_n^*)$.</p>
</div>
<div class="algo" data-name="Bootstrap standard error">
<ol>
<li>For $b = 1,\dots,B$: draw $X^*_{1},\dots,X^*_{n}$ with replacement from $\{X_1,\dots,X_n\}$ and compute $T^*_{n,b}$.</li>
<li>Return $\widehat{\se}_{\text{boot}} = \sqrt{\tfrac{1}{B}\sum_b (T^*_{n,b}-\bar T^*)^2}$.</li>
</ol>
</div>
<p>The same replicates also give a bias estimate, $\widehat{\Bias} = \bar T^* - T_n$, mirroring $\Bias = \E_F T_n - \theta$ with $\hat F_n$ in place of $F$ and $T_n = T(\hat F_n)$ in place of $\theta = T(F)$.</p>
<div class="ex" data-name="How many distinct points?">
<p>A bootstrap sample omits each observation with probability $(1-1/n)^n \to e^{-1} \approx 0.368$. So a typical resample contains only about $63.2\%$ of the distinct data points, some repeated. This is the source of the "0.632" estimators in prediction-error estimation, and it explains why the bootstrap struggles with statistics like the maximum: $X_{(n)}^* = X_{(n)}$ with probability $1-(1-1/n)^n \to 0.632$, so the bootstrap distribution of the max has a huge atom that the true sampling distribution does not.</p>
</div>`,
        lab: 'bootstrap-world',
      },
      {
        title: 'Bootstrap confidence intervals',
        html: R`
<p>Once we have $T^*_{n,1},\dots,T^*_{n,B}$, there are several ways to turn them into an interval for $\theta = T(F)$. Write $\theta^*_\alpha$ for the $\alpha$ sample quantile of the replicates.</p>
<div class="cols">
<div><h4>Normal</h4>$$T_n \pm z_{\alpha/2}\,\widehat{\se}_{\text{boot}}$$<p>Simple; inherits the normal approximation's errors when $T_n$ is skewed.</p></div>
<div><h4>Percentile</h4>$$\big(\theta^*_{\alpha/2},\; \theta^*_{1-\alpha/2}\big)$$<p>Transformation-respecting: if it is right for some monotone $\phi(\theta)$, it is right for $\theta$.</p></div>
<div><h4>Pivotal (basic)</h4>$$\big(2T_n - \theta^*_{1-\alpha/2},\; 2T_n - \theta^*_{\alpha/2}\big)$$<p>Approximates the law of the pivot $T_n - \theta$ by that of $T_n^* - T_n$.</p></div>
</div>
<p>The pivotal interval deserves a derivation, because it is the one whose logic is cleanest and it shows why the quantiles appear "reflected".</p>
<div class="thm" data-kind="Proposition" data-name="Pivotal bootstrap interval">
<p>Let $R_n = T_n - \theta$ with CDF $H(r) = \P_F(R_n \le r)$. If $H$ were known, $C_n = (T_n - H^{-1}(1-\alpha/2),\; T_n - H^{-1}(\alpha/2))$ would have coverage exactly $1-\alpha$. Estimating $H$ by the bootstrap distribution of $R^*_n = T^*_n - T_n$ gives the pivotal interval above.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\P(T_n - H^{-1}(1-\alpha/2) \le \theta \le T_n - H^{-1}(\alpha/2)) = \P(H^{-1}(\alpha/2) \le R_n \le H^{-1}(1-\alpha/2)) = (1-\alpha/2) - \alpha/2 = 1-\alpha.$ The bootstrap estimate is $\hat H(r) = \frac1B\sum_b \ind(T^*_{n,b} - T_n \le r)$, whose $\beta$ quantile is $\theta^*_\beta - T_n$. Substituting, the lower endpoint is $T_n - (\theta^*_{1-\alpha/2} - T_n) = 2T_n - \theta^*_{1-\alpha/2}$, and likewise for the upper.</p>
</details>
<p>Better still is the <strong>studentized</strong> (bootstrap-$t$) interval: bootstrap the approximately pivotal quantity $Z_n = (T_n-\theta)/\widehat{\se}$, computing a standard error inside every resample (often by a nested bootstrap), then invert. It is the interval with the best theoretical accuracy, discussed next. The <strong>BC$_a$</strong> interval achieves the same accuracy without nesting by adjusting percentile levels for bias and skewness.</p>`,
        lab: 'bootstrap-coverage',
      },
      {
        title: 'Why it works: consistency and second-order accuracy',
        html: R`
<p>"Works" has a precise meaning: the bootstrap distribution, which is a <em>random</em> distribution (it depends on the data), should converge to the true sampling distribution.</p>
<div class="thm" data-name="Bootstrap consistency for the mean (Bickel & Freedman, 1981)">
<p>If $\E_F X^2 < \infty$, then with $\mu = \E X$ and $\bar X_n^*$ the mean of a bootstrap sample,
$$\sup_x \Big| \P_{\hat F_n}\!\big(\sqrt n(\bar X^*_n - \bar X_n) \le x\big) - \P_F\!\big(\sqrt n(\bar X_n - \mu)\le x\big)\Big| \xrightarrow{\text{a.s.}} 0.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Conditional on the data, $\bar X^*_n$ is a mean of IID draws from $\hat F_n$, which has mean $\bar X_n$ and variance $\hat\sigma^2_n = \frac1n\sum(X_i-\bar X_n)^2 \to \sigma^2$ a.s. by the strong law. Apply the Lindeberg CLT to the triangular array of bootstrap draws: the Lindeberg condition holds a.s. because $\frac1n\sum X_i^2\ind(|X_i|>\epsilon\sqrt n) \to 0$ a.s. when $\E X^2<\infty$. So $\sqrt n(\bar X^*_n-\bar X_n) \dto \N(0,\sigma^2)$ conditionally, a.s.; the true sampling distribution has the same limit by the ordinary CLT; and since the limit is continuous, Pólya's theorem upgrades pointwise convergence of CDFs to uniform.</p>
</details>
<p>More generally the bootstrap is consistent for any statistic $T(\hat F_n)$ where $T$ is <a href="#/t/delta-method">Hadamard differentiable</a> — the functional delta method applies equally in the real world ($\hat F_n$ around $F$) and the bootstrap world ($\hat F^*_n$ around $\hat F_n$). Consistency alone puts the bootstrap on a par with the normal approximation. Its real advantage shows up in the next term of the expansion.</p>
<div class="thm" data-kind="Theorem" data-name="Second-order accuracy (Hall, 1992)">
<p>For smooth functions of means, the CDF of the studentized statistic has an Edgeworth expansion $\P(Z_n\le x) = \Phi(x) + n^{-1/2}p(x)\phi(x) + O(n^{-1})$ where $p$ depends on the skewness of $F$. The bootstrap-$t$ distribution has the same expansion with $F$ replaced by $\hat F_n$, so $p$ is estimated with error $O_p(n^{-1/2})$, and
$$\P_{\hat F_n}(Z^*_n \le x) - \P_F(Z_n\le x) = O_p(n^{-1}),$$
versus $O(n^{-1/2})$ for the normal approximation. Coverage error of two-sided studentized intervals is $O(n^{-2})$ vs. $O(n^{-1})$.</p>
</div>
<p>The intuition: the normal approximation ignores skewness entirely, while the bootstrap estimates it from the data. Percentile and basic intervals are only first-order accurate because $T_n - \theta$ is not pivotal (its spread depends on $F$); studentizing removes that dependence.</p>`,
      },
      {
        title: 'When the bootstrap fails',
        html: R`
<p>Consistency rests on smoothness of $T$ and on the resampling mimicking how the data were generated. Break either and the bootstrap can be confidently wrong.</p>
<div class="warn" data-name="Extremes and boundaries">
<p>For $X_i \sim \text{Unif}(0,\theta)$ and $T_n = X_{(n)}$, $n(\theta - X_{(n)})/\theta \dto \text{Exp}(1)$, a continuous law. But $\P^*(X^*_{(n)} = X_{(n)}) \to 1-e^{-1}$, so the bootstrap law has an atom at 0 and never converges to it. The maximum is not a smooth functional of $F$. Fixes: the $m$-out-of-$n$ bootstrap ($m\to\infty$, $m/n\to0$), or the parametric bootstrap. Try "Uniform" with "maximum" in the first lab.</p>
</div>
<div class="warn" data-name="Heavy tails">
<p>If $\E X^2 = \infty$ (e.g. a Pareto with $\alpha<2$), the bootstrap of the mean is inconsistent: the bootstrap distribution converges to a <em>random</em> limit (Athreya, 1987). For the Cauchy, the sample mean has no limiting variance at all. Robust statistics (median, trimmed mean) restore validity.</p>
</div>
<div class="warn" data-name="Dependence">
<p>Resampling individual points destroys serial correlation, so for time series the IID bootstrap underestimates the variance of $\bar X_n$ (positive autocorrelation inflates it). Use the <em>moving block</em> or <em>stationary</em> bootstrap, which resample contiguous blocks of length $\ell \to \infty$, $\ell/n \to 0$.</p>
</div>
<p>A useful rule: the bootstrap works when the statistic is asymptotically normal for reasons a delta-method argument could explain. When the limit law is non-normal, check the literature before trusting it.</p>`,
      },
      {
        title: 'Variants worth knowing',
        html: R`
<table>
<thead><tr><th>Variant</th><th>Resample from</th><th>Use when</th></tr></thead>
<tbody>
<tr><td>Parametric</td><td>$F_{\hat\theta}$ (fitted model)</td><td>Model is trusted; small $n$; extremes</td></tr>
<tr><td>Residual (regression)</td><td>$\hat y_i + e^*_i$, $e^*$ drawn from residuals</td><td>Fixed design, homoscedastic errors</td></tr>
<tr><td>Pairs (regression)</td><td>$(x_i, y_i)$ pairs</td><td>Random design or heteroscedasticity</td></tr>
<tr><td>Wild</td><td>$\hat y_i + e_i v_i$, $\E v_i=0$, $\E v_i^2=1$</td><td>Heteroscedastic errors, fixed design</td></tr>
<tr><td>Block / stationary</td><td>Contiguous blocks</td><td>Time series, spatial data</td></tr>
<tr><td>Bayesian</td><td>Dirichlet$(1,\dots,1)$ weights on data</td><td>Posterior under a nonparametric model</td></tr>
</tbody>
</table>
<p>Each is the plug-in principle applied to a different estimate of the data-generating mechanism. Choosing a variant is choosing what you believe about how the data arose.</p>`,
      },
    ],
    quiz: [
      { q: R`You compute $\widehat{\se}_{\text{boot}}$ with $B=50$ and again with $B=50{,}000$. Which source of error did the larger $B$ reduce, and which did it leave untouched?`,
        a: R`It reduced the Monte Carlo error of approximating $\Var_{\hat F_n}(T_n)$. It cannot reduce the statistical error $\Var_{\hat F_n}(T_n) - \Var_F(T_n)$, which depends on how well $\hat F_n$ approximates $F$ and therefore only on $n$.` },
      { q: R`Why is the pivotal interval $(2T_n - \theta^*_{1-\alpha/2},\, 2T_n - \theta^*_{\alpha/2})$ "reflected" relative to the percentile interval?`,
        a: R`It estimates the distribution of the pivot $T_n - \theta$ by that of $T_n^* - T_n$. If the bootstrap replicates are skewed right of $T_n$, the pivot suggests $T_n$ tends to <em>overshoot</em> $\theta$, so the interval for $\theta$ extends further to the left. Inverting $\theta = T_n - (T_n-\theta)$ flips the quantiles.` },
      { q: R`What is the probability that a particular observation $X_i$ appears in a given bootstrap sample, as $n\to\infty$?`,
        a: R`$1 - (1-1/n)^n \to 1 - e^{-1} \approx 0.632$.` },
      { q: R`Give two statistics for which the naive IID bootstrap is inconsistent, and a remedy for each.`,
        a: R`(1) The sample maximum under $\text{Unif}(0,\theta)$ — use the $m$-out-of-$n$ or parametric bootstrap. (2) The sample mean when $\E X^2=\infty$ — use a robust statistic or $m$-out-of-$n$. Also: the mean of a dependent series — use a block bootstrap.` },
    ],
  });
})();
