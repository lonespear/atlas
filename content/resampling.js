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
<p>Better still is the <strong>studentized</strong> (bootstrap-$t$) interval: bootstrap the approximately pivotal quantity $Z_n = (T_n-\theta)/\widehat{\se}$, computing a standard error inside every resample (often by a nested bootstrap), then invert. It is the interval with the best theoretical accuracy, discussed next. The <strong>BC$_a$</strong> interval achieves the same accuracy without nesting by adjusting percentile levels for bias and skewness; both are developed in <a href="#/t/bca-intervals">Better Bootstrap Intervals</a>.</p>`,
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
versus $O(n^{-1/2})$ for the normal approximation. One-sided studentized bounds have coverage error $O(n^{-1})$ versus $O(n^{-1/2})$ for normal, percentile and basic bounds; <em>symmetric</em> two-sided studentized intervals reach $O(n^{-2})$ (Hall, 1988). See <a href="#/t/bca-intervals">studentized and BC$_a$ intervals</a>.</p>
</div>
<p>The intuition: the normal approximation ignores skewness entirely, while the bootstrap estimates it from the data. Percentile and basic intervals are only first-order accurate because $T_n - \theta$ is not pivotal (its spread depends on $F$); studentizing removes that dependence.</p>`,
      },
      {
        title: 'When the bootstrap fails',
        html: R`
<p>Consistency rests on smoothness of $T$ and on the resampling mimicking how the data were generated. Break either and the bootstrap can be confidently wrong.</p>
<div class="warn" data-name="Extremes and boundaries">
<p>For $X_i \sim \text{Unif}(0,\theta)$ and $T_n = X_{(n)}$, $n(\theta - X_{(n)})/\theta \dto \text{Exp}(1)$, a continuous law. But $\P^*(X^*_{(n)} = X_{(n)}) \to 1-e^{-1}$, so the bootstrap law has an atom at 0 and never converges to it. The maximum is not a smooth functional of $F$. Fixes: the <a href="#/t/subsampling">$m$-out-of-$n$ bootstrap</a> ($m\to\infty$, $m/n\to0$), or the parametric bootstrap. Try "Uniform" with "maximum" in the first lab.</p>
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
<tr><td><a href="#/t/parametric-bootstrap">Parametric</a></td><td>$F_{\hat\theta}$ (fitted model)</td><td>Model is trusted; small $n$; extremes</td></tr>
<tr><td><a href="#/t/bootstrap-regression">Residual (regression)</a></td><td>$\hat y_i + e^*_i$, $e^*$ drawn from residuals</td><td>Fixed design, homoscedastic errors</td></tr>
<tr><td><a href="#/t/bootstrap-regression">Pairs (regression)</a></td><td>$(x_i, y_i)$ pairs</td><td>Random design or heteroscedasticity</td></tr>
<tr><td><a href="#/t/bootstrap-regression">Wild</a></td><td>$\hat y_i + e_i v_i$, $\E v_i=0$, $\E v_i^2=1$</td><td>Heteroscedastic errors, fixed design</td></tr>
<tr><td><a href="#/t/block-bootstrap">Block / stationary</a></td><td>Contiguous blocks</td><td>Time series, spatial data</td></tr>
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

/* ---------------------------------------------------------------- jackknife */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'jackknife',
    domain: 'resampling',
    order: 10,
    title: 'The Jackknife',
    subtitle: 'Leave-one-out recomputation for bias and variance, and why it breaks for the median',
    level: 2,
    sources: { aos: '§8.5', other: 'Quenouille (1949, 1956); Tukey (1958); Miller (1974); Efron & Stein (1981); Shao & Tu (1995, Ch. 2)' },
    prereqs: ['empirical-cdf', 'delta-method'],
    related: ['bootstrap', 'statistical-functionals', 'bca-intervals', 'robust-statistics', 'cross-validation'],
    tags: ['leave-one-out', 'bias correction', 'standard error', 'influence function', 'pseudo-values'],
    summary: R`The jackknife recomputes a statistic $n$ times, each time with one observation deleted, and reads bias and variance off the spread of those leave-one-out values. It predates the bootstrap by two decades and is, in a precise sense, a linear approximation to it: the leave-one-out differences are finite-difference estimates of the influence function, so the jackknife variance is a numerical version of the nonparametric delta method. That explains both its strength (it is deterministic, cheap, and exactly right for means and $U$-statistics) and its weakness (it fails when the statistic is not smooth, the median being the classic example).`,
    survey: {
      task: 'Bias correction and standard errors for smooth statistics; acceleration constant for BCa',
      assumptions: R`IID data; statistic a smooth (differentiable) functional of $F$; bias expandable in powers of $1/n$`,
      output: R`$\hat\theta_{\text{jack}}$, $\widehat{\Bias}_{\text{jack}}$, $\widehat{\se}_{\text{jack}}$, pseudo-values, empirical influence values`,
      strengths: R`Deterministic, exactly $n$ recomputations, removes $O(1/n)$ bias; exact for means and $U$-statistics`,
      weaknesses: R`Inconsistent variance for non-smooth statistics (median, quantiles); no distributional information for CIs`,
      complexity: R`$n$ evaluations of $T$ (or $\binom{n}{d}$ for delete-$d$)`,
    },
    sections: [
      {
        title: 'Leave-one-out values',
        html: R`
<p>Let $X_1,\dots,X_n \iid F$ and $\hat\theta = T_n(X_1,\dots,X_n)$ estimate $\theta = T(F)$. The jackknife is built from the $n$ statistics computed on the data with one point removed:</p>
$$\hat\theta_{(i)} = T_{n-1}(X_1,\dots,X_{i-1},X_{i+1},\dots,X_n), \qquad \hat\theta_{(\cdot)} = \frac1n\sum_{i=1}^n \hat\theta_{(i)}.$$
<p>Quenouille (1949) introduced them to reduce bias; Tukey (1958) noticed that they also measure variability and named the method after a rough-and-ready pocket tool. The logic is the same as for the <a href="#/t/bootstrap">bootstrap</a> — perturb the empirical distribution, watch the statistic move — but the perturbations are $n$ fixed, tiny, deterministic ones rather than random resamples.</p>
<div class="def" data-name="Jackknife estimates">
<p>With $\hat\theta_{(i)}$ and $\hat\theta_{(\cdot)}$ as above,</p>
$$\widehat{\Bias}_{\text{jack}} = (n-1)\big(\hat\theta_{(\cdot)} - \hat\theta\big), \qquad \hat\theta_{\text{jack}} = \hat\theta - \widehat{\Bias}_{\text{jack}} = n\hat\theta - (n-1)\hat\theta_{(\cdot)},$$
$$v_{\text{jack}} = \frac{n-1}{n}\sum_{i=1}^n \big(\hat\theta_{(i)} - \hat\theta_{(\cdot)}\big)^2, \qquad \widehat{\se}_{\text{jack}} = \sqrt{v_{\text{jack}}}.$$
</div>
<div class="note">
<p>The factors $(n-1)$ look odd at first. They are there because deleting one point is a <em>much smaller</em> perturbation than drawing a new sample: $\hat\theta_{(i)}$ and $\hat\theta$ share $n-1$ observations, so the leave-one-out values are packed roughly $n-1$ times more tightly than independent replicates would be. For the mean, $\hat\theta_{(i)} - \hat\theta = (\bar X - X_i)/(n-1)$ exactly, so the spread of the $\hat\theta_{(i)}$ is the spread of the data shrunk by $1/(n-1)$. The inflation factors undo that shrinkage.</p>
</div>`,
      },
      {
        title: 'Bias reduction: removing the $1/n$ term',
        html: R`
<p>Many estimators have a bias that expands in powers of $1/n$: the plug-in variance, the ratio $\bar Y/\bar X$, $g(\bar X)$ for smooth nonlinear $g$, the MLE in regular models. The jackknife exploits the fact that the leading term has a known dependence on sample size.</p>
<div class="thm" data-kind="Theorem" data-name="Jackknife removes the first-order bias (Quenouille)">
<p>Suppose that for every sample size $m$, $\E\,\hat\theta_m = \theta + \dfrac{a}{m} + \dfrac{b}{m^2} + O(m^{-3})$ with constants $a, b$ not depending on $m$. Then
$$\E\,\hat\theta_{\text{jack}} = \theta - \frac{b}{n(n-1)} + O(n^{-2}) = \theta + O(n^{-2}).$$
In particular, if the bias is exactly $a/n$, the jackknife estimator is exactly unbiased.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Each $\hat\theta_{(i)}$ is the same statistic computed on an IID sample of size $n-1$, so $\E\,\hat\theta_{(i)} = \theta + a/(n-1) + b/(n-1)^2 + O(n^{-3})$, and the same holds for their average $\hat\theta_{(\cdot)}$. Hence
$$\E\,\hat\theta_{\text{jack}} = n\Big(\theta + \frac an + \frac b{n^2}\Big) - (n-1)\Big(\theta + \frac a{n-1} + \frac b{(n-1)^2}\Big) + O(n^{-2}) = \theta + \frac bn - \frac b{n-1} + O(n^{-2}).$$
The $\theta$ terms combine to $\theta$, the $a$ terms cancel exactly, and $b/n - b/(n-1) = -b/(n(n-1))$. (The remainders $n\cdot O(n^{-3})$ and $(n-1)\cdot O(n^{-3})$ are $O(n^{-2})$.)</p>
</details>
<p>The argument is Richardson extrapolation: two estimators at sample sizes $n$ and $n-1$ whose biases have known ratio $(n-1)/n$ in their leading term are combined so that term cancels. Nothing about the shape of $T$ was used beyond the bias expansion.</p>
<div class="thm" data-kind="Proposition" data-name="Jackknife turns V-statistics into U-statistics">
<p>Let $h$ be symmetric with $h(x,x) = 0$ and $\hat\theta = n^{-2}\sum_{i,j} h(X_i,X_j)$ (a $V$-statistic). Then $\hat\theta_{\text{jack}} = U_n := \frac{1}{n(n-1)}\sum_{i\ne j} h(X_i,X_j)$, the unbiased $U$-statistic. Taking $h(x,y) = (x-y)^2/2$, the jackknife of the plug-in variance $\frac1n\sum(X_i-\bar X)^2$ is exactly $s^2 = \frac{1}{n-1}\sum(X_i-\bar X)^2$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Because the diagonal vanishes, $\hat\theta = \frac{n-1}{n}U_n$. Likewise $\hat\theta_{(i)} = \frac{n-2}{n-1}U_{(i)}$, where $U_{(i)}$ averages $h$ over the $(n-1)(n-2)$ ordered pairs not involving $i$. Every ordered pair $(j,k)$, $j\ne k$, avoids exactly $n-2$ of the indices $i$, so $\sum_i U_{(i)} = \frac{n-2}{(n-1)(n-2)}\sum_{j\ne k}h(X_j,X_k) = nU_n$, i.e. $\frac1n\sum_iU_{(i)} = U_n$. Therefore $\hat\theta_{(\cdot)} = \frac{n-2}{n-1}U_n$ and
$$\hat\theta_{\text{jack}} = n\cdot\tfrac{n-1}{n}U_n - (n-1)\cdot\tfrac{n-2}{n-1}U_n = (n-1)U_n - (n-2)U_n = U_n.$$
For the variance, $\frac{1}{2n^2}\sum_{i,j}(X_i-X_j)^2 = \frac1n\sum(X_i-\bar X)^2$, and the corresponding $U$-statistic is $s^2$.</p>
</details>
<div class="warn" data-name="Bias reduction is not free">
<p>Removing bias usually increases variance, and for estimators whose bias is already negligible relative to their standard error ($O(1/n)$ vs. $O(1/\sqrt n)$), the correction rarely improves mean squared error. Use $\widehat{\Bias}_{\text{jack}}$ primarily as a <em>diagnostic</em>: if it is a sizeable fraction of $\widehat{\se}$, the estimator or its parametrization deserves attention. And if the bias does not expand in powers of $1/n$ (e.g. it is of order $n^{-1/2}$ or comes from a boundary), the theorem says nothing.</p>
</div>`,
      },
      {
        title: 'Pseudo-values and the variance estimator',
        html: R`
<p>Tukey's insight was to rewrite the bias-corrected estimator as an average.</p>
<div class="def" data-name="Pseudo-values">
<p>The $i$-th pseudo-value is $\tilde\theta_i = n\hat\theta - (n-1)\hat\theta_{(i)}$. Then $\hat\theta_{\text{jack}} = \frac1n\sum_i\tilde\theta_i$ and
$$v_{\text{jack}} = \frac{1}{n(n-1)}\sum_{i=1}^n\big(\tilde\theta_i - \bar{\tilde\theta}\big)^2,$$
which is precisely the usual estimate $s^2/n$ of the variance of a mean, applied to the pseudo-values.</p>
</div>
<p>(The two expressions for $v_{\text{jack}}$ agree because $\tilde\theta_i - \bar{\tilde\theta} = -(n-1)(\hat\theta_{(i)} - \hat\theta_{(\cdot)})$, so $\frac{1}{n(n-1)}(n-1)^2\sum(\hat\theta_{(i)}-\hat\theta_{(\cdot)})^2 = \frac{n-1}{n}\sum(\hat\theta_{(i)}-\hat\theta_{(\cdot)})^2$.)</p>
<p>Tukey's heuristic: treat the pseudo-values as if they were $n$ approximately IID observations whose mean is $\hat\theta_{\text{jack}}$. Then $\hat\theta_{\text{jack}} \pm t_{n-1,\alpha/2}\,\widehat{\se}_{\text{jack}}$ is an approximate interval. The heuristic is exact for the mean:</p>
<div class="ex" data-name="The mean: pseudo-values are the data">
<p>For $\hat\theta = \bar X$, $\hat\theta_{(i)} = (n\bar X - X_i)/(n-1)$, so $\tilde\theta_i = n\bar X - (n\bar X - X_i) = X_i$. Thus $\hat\theta_{\text{jack}} = \bar X$, $\widehat{\Bias}_{\text{jack}} = 0$, and $v_{\text{jack}} = s^2/n$ — the textbook answer, with the unbiased divisor $n-1$. Compare the bootstrap, whose ideal variance for the mean is $\hat\sigma^2/n = \frac{n-1}{n}\cdot s^2/n$: the jackknife is the one that is exactly unbiased here.</p>
</div>
<div class="ex" data-name="A ratio estimator">
<p>With pairs $(X_i,Y_i)$ and $\hat\theta = \bar Y/\bar X$, write $\theta = \mu_Y/\mu_X$. A second-order Taylor expansion gives $\E\hat\theta = \theta + \frac1n\cdot\frac{\theta\sigma_X^2 - \sigma_{XY}}{\mu_X^2} + O(n^{-2})$, a bias with exactly the $a/n$ structure the theorem needs, so $\hat\theta_{\text{jack}}$ has bias $O(n^{-2})$. Each $\hat\theta_{(i)} = (n\bar Y - Y_i)/(n\bar X - X_i)$ costs $O(1)$ after computing the totals, so the whole jackknife costs $O(n)$.</p>
</div>
<p>In the lab, compare the leave-one-out values for the mean (a smooth cloud, pseudo-values equal to the data), the plug-in variance (bias correction recovers $s^2$ exactly), and the median (two distinct values only).</p>`,
        lab: 'jackknife-pseudovalues',
      },
      {
        title: 'The jackknife as a linearization: influence functions',
        html: R`
<p>Why should $v_{\text{jack}}$ estimate a variance at all? Write $\hat\theta = T(\hat F_n)$ for a <a href="#/t/statistical-functionals">statistical functional</a> $T$. Deleting $X_i$ changes the empirical distribution to
$$\hat F_{(i)} = \frac{n\hat F_n - \delta_{X_i}}{n-1} = (1-\varepsilon)\hat F_n + \varepsilon\,\delta_{X_i}, \qquad \varepsilon = -\frac{1}{n-1}.$$
This is a contamination of $\hat F_n$ in the direction of the point mass at $X_i$ with a small <em>negative</em> weight. Recall the influence function $L(x;F) = \lim_{\varepsilon\to0}\big(T((1-\varepsilon)F + \varepsilon\delta_x) - T(F)\big)/\varepsilon$. Hence</p>
$$\tilde\theta_i - \hat\theta = (n-1)\big(\hat\theta - \hat\theta_{(i)}\big) = \frac{T((1-\varepsilon)\hat F_n + \varepsilon\delta_{X_i}) - T(\hat F_n)}{\varepsilon} \approx L(X_i;\hat F_n).$$
<p>The centered pseudo-values are finite-difference approximations, with step $-1/(n-1)$, to the <strong>empirical influence values</strong>. Since the delta-method variance of a smooth functional is $\Var_F L(X;F)/n$, and $v_{\text{jack}}\approx\frac{1}{n^2}\sum_i L(X_i;\hat F_n)^2$, the jackknife is the nonparametric delta method with numerical differentiation. Replacing the finite difference by the exact derivative gives the <em>infinitesimal jackknife</em> (Jaeckel, 1972), which coincides with the nonparametric delta method and with the variance of the linear approximation to the bootstrap.</p>
<div class="thm" data-kind="Theorem" data-name="Consistency for smooth functions of means (Miller, 1974; Shao & Tu, 1995)">
<p>Let $X_i\in\R^d$ be IID with mean $\mu$ and finite covariance $\Sigma$, and $\hat\theta = g(\bar X)$ with $g$ continuously differentiable in a neighbourhood of $\mu$ and $\nabla g(\mu)\ne0$. Then
$$n\,v_{\text{jack}} \asto \nabla g(\mu)^\top\Sigma\,\nabla g(\mu),$$
the asymptotic variance of $\sqrt n(\hat\theta-\theta)$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$\bar X_{(i)} - \bar X = (\bar X - X_i)/(n-1)$. By the mean value theorem, $\hat\theta_{(i)} - \hat\theta = \nabla g(\xi_i)^\top(\bar X_{(i)}-\bar X)$ with $\xi_i$ on the segment between $\bar X_{(i)}$ and $\bar X$. Since $\E\|X\|^2<\infty$ implies $\max_i\|X_i\|/\sqrt n\to0$ a.s., we get $\max_i\|\bar X_{(i)}-\bar X\|\to0$ and so $\max_i\|\nabla g(\xi_i)-\nabla g(\mu)\|\to0$ a.s. by the SLLN and continuity of $\nabla g$. Therefore
$$n v_{\text{jack}} = (n-1)\sum_i\big(\hat\theta_{(i)}-\hat\theta_{(\cdot)}\big)^2 = \frac{1}{n-1}\sum_i\Big(\nabla g(\mu)^\top(X_i-\bar X)\Big)^2 + o(1) \asto \nabla g(\mu)^\top\Sigma\nabla g(\mu),$$
where the remainder is bounded by a multiple of $\max_i\|\nabla g(\xi_i)-\nabla g(\mu)\|\cdot\frac{1}{n-1}\sum_i\|X_i-\bar X\|^2 \to 0$, and the last step is the SLLN for the sample covariance.</p>
</details>
<p>More generally, the jackknife variance is consistent for functionals that are suitably <em>continuously</em> (Fréchet) differentiable. Hadamard differentiability, which suffices for the bootstrap, is not enough: the jackknife needs the derivative to be stable under the $O(1/n)$ point-deletion perturbations it actually applies, and that is precisely what fails next.</p>`,
      },
      {
        title: 'Where the jackknife fails: the median',
        html: R`
<p>Take $n = 2m$ and $\hat\theta$ the sample median $\frac12(X_{(m)} + X_{(m+1)})$. Delete an observation from the lower half (one of $X_{(1)},\dots,X_{(m)}$): the remaining $2m-1$ points have median $X_{(m+1)}$. Delete one from the upper half: the median becomes $X_{(m)}$. So the $n$ leave-one-out values take <em>only two distinct values</em>, each $m$ times, and</p>
$$\hat\theta_{(\cdot)} = \tfrac12(X_{(m)}+X_{(m+1)}),\qquad v_{\text{jack}} = \frac{n-1}{n}\cdot n\cdot\Big(\frac{X_{(m+1)}-X_{(m)}}{2}\Big)^2 = \frac{n-1}{4}\big(X_{(m+1)}-X_{(m)}\big)^2.$$
<p>All the information is in a single spacing between two adjacent order statistics. That spacing is noisy: it does not average over anything.</p>
<div class="thm" data-kind="Theorem" data-name="Inconsistency of the jackknife variance for the median (Efron, 1982)">
<p>If $F$ has a density $f$ that is continuous and positive at the median $\xi$, then the true asymptotic variance is $n\Var(\hat\theta)\to 1/(4f(\xi)^2)$, but
$$\frac{n\,v_{\text{jack}}}{1/(4f(\xi)^2)} \dto E^2, \qquad E\sim\text{Exp}(1),$$
equivalently $(\chi^2_2/2)^2$. The limit is random: $n v_{\text{jack}}$ does not converge to the target, and its limiting mean is $\E E^2 = 2$, so on average the jackknife <em>doubles</em> the variance.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>By the probability integral transform, $X_{(k)} = F^{-1}(U_{(k)})$ with uniform order statistics. Uniform spacings satisfy $n(U_{(m+1)}-U_{(m)})\dto\text{Exp}(1)$ (a spacing of $n$ uniforms has the Beta$(1,n)$ distribution). Since $U_{(m)},U_{(m+1)}\to\frac12$ and $F^{-1}$ has derivative $1/f(\xi)$ at $\frac12$, $n f(\xi)\,(X_{(m+1)}-X_{(m)})\dto E$. Then
$$n v_{\text{jack}} = \frac{n(n-1)}{4}(X_{(m+1)}-X_{(m)})^2 = \frac{n-1}{4n}\cdot\frac{\big(nf(\xi)(X_{(m+1)}-X_{(m)})\big)^2}{f(\xi)^2} \dto \frac{E^2}{4f(\xi)^2}.$$</p>
</details>
<p>The diagnosis matches the influence-function view: the median's influence function $\sgn(x-\xi)/(2f(\xi))$ involves the density, and a single-point deletion moves the median by one spacing, which is a terrible estimate of $1/f(\xi)$. The <a href="#/t/bootstrap">bootstrap</a>, by contrast, is consistent for the median, because its resamples perturb $\hat F_n$ by $O(n^{-1/2})$ — enough to move the median across many spacings and average them.</p>
<div class="note" data-name="The delete-d fix">
<p>Deleting $d$ points at a time, with $v_{\text{jack},d} = \frac{n-d}{d\binom nd}\sum_{|S|=d}(\hat\theta_{(S)} - \bar\theta_{(\cdot)})^2$, makes the perturbation larger. Shao & Wu (1989) show the delete-$d$ jackknife variance is consistent for sample quantiles provided $\sqrt n/d\to0$ and $n-d\to\infty$. In practice one samples subsets at random, which brings the method close to <a href="#/t/subsampling">subsampling</a>.</p>
</div>
<p>The second lab repeats the experiment many times and plots the distribution of $n v_{\text{jack}}$ divided by the true asymptotic variance. For the mean it piles up at 1; for the median it follows the heavy-tailed $E^2$ law no matter how large $n$ gets.</p>`,
        lab: 'jackknife-median-failure',
      },
      {
        title: 'Jackknife versus bootstrap, and modern uses',
        html: R`
<table>
<thead><tr><th></th><th>Jackknife</th><th>Bootstrap</th></tr></thead>
<tbody>
<tr><td>Perturbation of $\hat F_n$</td><td>$n$ deterministic, size $1/n$</td><td>$B$ random, size $\asymp n^{-1/2}$</td></tr>
<tr><td>Cost</td><td>$n$ evaluations</td><td>$B$ evaluations ($10^3$–$10^4$)</td></tr>
<tr><td>Variance of smooth statistics</td><td>Consistent; essentially the delta method</td><td>Consistent</td></tr>
<tr><td>Non-smooth (median, quantiles)</td><td>Inconsistent (delete-1)</td><td>Consistent</td></tr>
<tr><td>Sampling distribution / CIs</td><td>Only via a normal or $t$ approximation</td><td>Yes (percentile, BC$_a$, bootstrap-$t$)</td></tr>
<tr><td>Monte Carlo noise</td><td>None</td><td>$O(B^{-1/2})$</td></tr>
</tbody>
</table>
<p>Efron (1982) showed that the jackknife variance is, up to a factor $n/(n-1)$, the bootstrap variance of the best approximation to the statistic that is <em>linear</em> in the resampling weights. When $T$ is nearly linear the two agree; when it is not, the bootstrap captures curvature the jackknife cannot.</p>
<div class="thm" data-kind="Theorem" data-name="Efron–Stein inequality (Efron & Stein, 1981)">
<p>Let $S$ be any symmetric function of $n-1$ arguments with $\E S^2<\infty$, let $X_1,\dots,X_n$ be IID, and let $S_{(i)}$ be $S$ evaluated on the sample with $X_i$ removed, $S_{(\cdot)} = \frac1n\sum_iS_{(i)}$. Then
$$\Var(S(X_1,\dots,X_{n-1})) \le \E\sum_{i=1}^n\big(S_{(i)} - S_{(\cdot)}\big)^2.$$</p>
</div>
<p>In jackknife notation the right-hand side is $\frac{n}{n-1}\E\,v_{\text{jack}}$, so the jackknife variance, on average, can only <em>over</em>state the variance of the statistic at sample size $n-1$. The $E^2$ limit for the median (mean 2) is a concrete instance of that conservatism.</p>
<p>The jackknife survives in several places:</p>
<ul>
<li><strong>Acceleration in BC$_a$.</strong> The skewness of the leave-one-out values estimates the acceleration constant $a$; see <a href="#/t/bca-intervals">BC$_a$ intervals</a>.</li>
<li><strong>Jackknife-after-bootstrap</strong> (Efron, 1992): assess the variability of a bootstrap quantity by grouping the existing resamples according to which observation they omit — no second level of resampling needed.</li>
<li><strong>Cross-validation</strong> is a leave-one-out computation of a different target (prediction error); the hat-matrix shortcut in <a href="#/t/cross-validation">cross-validation</a> is the same algebra as the jackknife of a least-squares fit.</li>
<li><strong>Jackknife+ prediction intervals</strong> (Barber, Candès, Ramdas & Tibshirani, 2021) use leave-one-out residuals and predictions to give distribution-free predictive coverage of at least $1-2\alpha$.</li>
<li><strong>Delete-a-group jackknife</strong> for complex surveys and clustered data: delete one cluster (or replicate group) at a time, which respects the dependence structure.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Show that for $\hat\theta = \bar X$ the pseudo-values are the observations themselves, and conclude what $\widehat{\se}_{\text{jack}}$ is.`,
        a: R`$\hat\theta_{(i)} = (n\bar X - X_i)/(n-1)$, so $\tilde\theta_i = n\bar X - (n-1)\hat\theta_{(i)} = X_i$. The jackknife variance is the sample variance of the pseudo-values divided by $n$, i.e. $s^2/n$ with divisor $n-1$, so $\widehat{\se}_{\text{jack}} = s/\sqrt n$, the standard textbook value.` },
      { q: R`An estimator has $\E\hat\theta_n = \theta + 3/n$ exactly. What is the bias of $\hat\theta_{\text{jack}}$? What if instead $\E\hat\theta_n = \theta + 3/\sqrt n$?`,
        a: R`In the first case it is exactly unbiased: $n(\theta + 3/n) - (n-1)(\theta + 3/(n-1)) = \theta$. In the second, $\E\hat\theta_{\text{jack}} - \theta = 3\big(\sqrt n - (n-1)/\sqrt{n-1}\big) = 3(\sqrt n - \sqrt{n-1}) \approx 3/(2\sqrt n)$: the correction only halves the bias, because the jackknife is calibrated for a $1/n$ expansion, not $n^{-1/2}$.` },
      { q: R`For $n = 2m$, why do the leave-one-out medians take only two values, and what does that imply for $v_{\text{jack}}$?`,
        a: R`Deleting any point below the median leaves $X_{(m+1)}$ as the middle of $2m-1$ points; deleting any point above leaves $X_{(m)}$. So $v_{\text{jack}} = \frac{n-1}{4}(X_{(m+1)} - X_{(m)})^2$ depends on a single spacing, which after scaling by $nf(\xi)$ is asymptotically Exp(1): the variance estimate has a non-degenerate random limit $E^2/(4f^2)$ with mean twice the truth, so it is inconsistent.` },
      { q: R`In what sense is the jackknife a numerical delta method?`,
        a: R`Deleting $X_i$ moves $\hat F_n$ to $(1-\varepsilon)\hat F_n + \varepsilon\delta_{X_i}$ with $\varepsilon = -1/(n-1)$, so $(n-1)(\hat\theta - \hat\theta_{(i)})$ is a finite-difference approximation of the empirical influence value $L(X_i;\hat F_n)$. The jackknife variance is then $\approx n^{-2}\sum L(X_i;\hat F_n)^2$, which is the plug-in estimate of the delta-method variance $\Var L(X;F)/n$.` },
      { q: R`Why does the bootstrap succeed for the median where the delete-1 jackknife fails?`,
        a: R`A bootstrap resample perturbs $\hat F_n$ by order $n^{-1/2}$ in sup norm, moving the median across roughly $\sqrt n$ spacings, so the spread of $\hat\theta^*$ averages many spacings and estimates $1/f(\xi)$ consistently. The jackknife's $1/n$ perturbations move it by one spacing only. Deleting $d\gg\sqrt n$ points restores consistency.` },
    ],
  });
})();

/* ---------------------------------------------------------------- bca-intervals */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'bca-intervals',
    domain: 'resampling',
    order: 30,
    title: 'Better Bootstrap Intervals: Studentized & BCa',
    subtitle: 'Second-order accurate confidence limits by studentizing or by correcting percentiles for bias and skewness',
    level: 3,
    sources: { aos: '§8.3', other: 'Efron (1987); Hall (1988, 1992); DiCiccio & Efron (1992, 1996); Efron & Tibshirani (1993, Ch. 12–14, 22)' },
    prereqs: ['bootstrap', 'jackknife'],
    related: ['confidence-intervals', 'delta-method', 'parametric-bootstrap', 'transformations'],
    tags: ['confidence interval', 'Edgeworth expansion', 'coverage error', 'acceleration', 'bias correction'],
    summary: R`Percentile and basic bootstrap intervals get the center right but the tails wrong: their one-sided coverage errors are of order $n^{-1/2}$, the same as the normal approximation. Two fixes reach order $n^{-1}$. The bootstrap-$t$ resamples a studentized pivot; BC$_a$ keeps the percentile method's transformation invariance but shifts the percentile levels using a bias-correction $z_0$ (read off the bootstrap histogram) and an acceleration $a$ (read off a jackknife). Both are justified by Edgeworth expansions, and BC$_a$ has an elegant exact derivation under a normalizing-transformation model in which the transformation never has to be known.`,
    survey: {
      task: 'Accurate one- and two-sided bootstrap confidence limits',
      assumptions: R`IID; $\hat\theta$ a smooth function of means (Edgeworth expansion exists); for bootstrap-$t$ a stable $\hat\sigma$`,
      output: R`Endpoints $\hat\theta[\alpha]$ with $\P(\theta\le\hat\theta[\alpha]) = \alpha + O(n^{-1})$`,
      strengths: 'Second-order accurate; BCa is transformation-respecting and range-preserving; bootstrap-t is conceptually simple',
      weaknesses: R`Bootstrap-$t$ is erratic with unstable $\hat\sigma$ and not invariant; BC$_a$ needs large $B$ and breaks down when $|z_0|$ or $|a|$ is large`,
      complexity: R`BC$_a$: $B + n$ evaluations; bootstrap-$t$: $B$ (analytic $\hat\sigma^*$) or $B\times B_2$ (nested)`,
    },
    sections: [
      {
        title: 'Measuring interval accuracy',
        html: R`
<p>Write $\hat\theta[\alpha]$ for an approximate <em>$\alpha$-level endpoint</em>: a statistic intended to satisfy $\P(\theta \le \hat\theta[\alpha]) = \alpha$. A two-sided equal-tailed $1-2\alpha$ interval is $(\hat\theta[\alpha], \hat\theta[1-\alpha])$. The quality of a method is the order of its coverage error.</p>
<div class="def" data-name="First- and second-order accuracy">
<p>An endpoint is <strong>first-order accurate</strong> if $\P(\theta\le\hat\theta[\alpha]) = \alpha + O(n^{-1/2})$ and <strong>second-order accurate</strong> if the error is $O(n^{-1})$. Equivalently (for smooth problems), if $\hat\theta_{\text{exact}}[\alpha]$ is the ideal endpoint, first-order means $\hat\theta[\alpha] - \hat\theta_{\text{exact}}[\alpha] = O_p(n^{-1})$ and second-order means $O_p(n^{-3/2})$.</p>
</div>
<p>The source of the $n^{-1/2}$ error is skewness. For a standardized smooth statistic the <strong>Edgeworth expansion</strong> reads
$$\P\Big(\frac{\sqrt n(\hat\theta-\theta)}{\sigma}\le x\Big) = \Phi(x) + n^{-1/2}p_1(x)\phi(x) + n^{-1}p_2(x)\phi(x) + \cdots,$$
where $p_1$ is an <em>even</em> quadratic polynomial whose coefficients involve the skewness of $F$ (for the mean, $p_1(x) = -\frac{\gamma}{6}(x^2-1)$ with $\gamma = \E(X-\mu)^3/\sigma^3$), and $p_2$ is odd. The normal approximation drops the $p_1$ term, so its one-sided error is $O(n^{-1/2})$.</p>
<div class="note" data-name="Why two-sided intervals hide the problem">
<p>For an equal-tailed two-sided interval the coverage is $\P(\text{below upper}) - \P(\text{below lower})$. Because $p_1$ is even, its contributions at $x$ and $-x$ are identical and cancel in the difference. So normal, percentile, and basic two-sided intervals all have coverage error $O(n^{-1})$ — but they achieve it by being wrong in <em>both</em> tails in compensating directions: too many misses on one side, too few on the other. If you care where the misses fall (one-sided tests, risk bounds, asymmetric losses), only second-order methods help. The lab below records left and right misses separately for exactly this reason.</p>
</div>
<table>
<thead><tr><th>Method</th><th>One-sided coverage error</th><th>Two-sided (equal-tailed)</th><th>Transformation-respecting</th></tr></thead>
<tbody>
<tr><td>Normal $\hat\theta\pm z\,\widehat{\se}$</td><td>$O(n^{-1/2})$</td><td>$O(n^{-1})$</td><td>No</td></tr>
<tr><td>Percentile</td><td>$O(n^{-1/2})$</td><td>$O(n^{-1})$</td><td>Yes</td></tr>
<tr><td>Basic (pivotal)</td><td>$O(n^{-1/2})$</td><td>$O(n^{-1})$</td><td>No</td></tr>
<tr><td>Bootstrap-$t$</td><td>$O(n^{-1})$</td><td>$O(n^{-1})$; symmetric version $O(n^{-2})$</td><td>No</td></tr>
<tr><td>BC$_a$</td><td>$O(n^{-1})$</td><td>$O(n^{-1})$</td><td>Yes</td></tr>
</tbody>
</table>`,
      },
      {
        title: 'The bootstrap-t',
        html: R`
<p>The idea is to bootstrap a quantity whose distribution depends on $F$ as little as possible. With $\hat\sigma$ a consistent standard error estimate, the studentized statistic $Z = (\hat\theta - \theta)/\hat\sigma$ is <em>asymptotically pivotal</em>: its limit $\N(0,1)$ does not depend on $F$.</p>
<div class="algo" data-name="Bootstrap-t interval">
<ol>
<li>Compute $\hat\theta$ and $\hat\sigma$ from the data.</li>
<li>For $b = 1,\dots,B$: resample, compute $\hat\theta^*_b$ and its own standard error $\hat\sigma^*_b$ <em>from the resample</em>, and set $Z^*_b = (\hat\theta^*_b - \hat\theta)/\hat\sigma^*_b$.</li>
<li>Let $\hat t_\beta$ be the $\beta$-quantile of the $Z^*_b$. Return $\big(\hat\theta - \hat t_{1-\alpha}\hat\sigma,\ \hat\theta - \hat t_{\alpha}\hat\sigma\big)$.</li>
</ol>
</div>
<p>Note the reflection, as in the basic interval: a long right tail in $Z^*$ (overshoots) pushes the <em>lower</em> endpoint down. When no formula for $\hat\sigma^*$ exists one uses a nested bootstrap ($B_2\approx 25$–$50$ inner resamples per outer resample) or a jackknife inside each resample.</p>
<div class="thm" data-kind="Theorem" data-name="Second-order accuracy of the bootstrap-t (Hall, 1988)">
<p>Let $\theta = g(\E X)$ with $g$ smooth, $\hat\theta = g(\bar X)$, $\hat\sigma^2$ the delta-method variance estimate, and suppose $X$ has enough finite moments and satisfies Cramér's condition $\limsup_{|t|\to\infty}|\E e^{itX}|<1$. Then $\P(Z\le x) = \Phi(x) + n^{-1/2}q_1(x)\phi(x) + O(n^{-1})$ uniformly in $x$, the bootstrap law satisfies the same expansion with $q_1$ replaced by $\hat q_1$ (coefficients evaluated at $\hat F_n$), and
$$\P\big(\theta \le \hat\theta - \hat t_{1-\alpha}\hat\sigma\big) = \alpha + O(n^{-1}).$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The coefficients of $q_1$ are smooth functions of a few moments of $F$; their plug-in estimates differ from them by $O_p(n^{-1/2})$. Hence $\P^*(Z^*\le x) - \P(Z\le x) = n^{-1/2}(\hat q_1 - q_1)(x)\phi(x) + O_p(n^{-1}) = O_p(n^{-1})$. Inverting through Cornish–Fisher expansions, $\hat t_\beta = z_\beta + n^{-1/2}\hat q_{11}(z_\beta) + O_p(n^{-1})$ and the exact quantile is $t_\beta = z_\beta + n^{-1/2}q_{11}(z_\beta) + O(n^{-1})$, so $\hat t_\beta - t_\beta = O_p(n^{-1})$. Converting a quantile error of $O_p(n^{-1})$ into a coverage error of $O(n^{-1})$ requires one more step — expanding $\P(Z \le \hat t_\beta)$ jointly, since $\hat t_\beta$ is random and correlated with $Z$ — which contributes only $O(n^{-1})$ terms (Hall, 1992, §3.5). By contrast, the percentile method bootstraps $\sqrt n(\hat\theta-\theta)$ <em>unstudentized</em>, whose leading term $\Phi(x/\sigma)$ has its scale estimated with $O_p(n^{-1/2})$ error, and its $p_1$ polynomial differs from the one needed; both leave $O(n^{-1/2})$ errors.</p>
</details>
<div class="warn" data-name="When the bootstrap-t misbehaves">
<p>Studentizing helps only if $\hat\sigma^*$ is a stable function of the resample. For a correlation coefficient near $\pm1$, a ratio with a denominator near zero, or small $n$ with heavy tails, the $Z^*_b$ have occasional huge values ($\hat\sigma^*_b$ tiny), producing absurdly wide or out-of-range intervals. The bootstrap-$t$ is also <em>not</em> transformation-respecting: applying it to $\rho$ and to Fisher's $z = \tanh^{-1}\rho$ gives different answers, and the latter (variance-stabilized) is much better. A good rule: bootstrap-$t$ on a variance-stabilizing scale, then transform back.</p>
</div>`,
      },
      {
        title: 'BCa: the transformation argument',
        html: R`
<p>Efron (1987) asked: what if the percentile method were applied on a scale where it is <em>exactly</em> right, but we do not know that scale? Postulate an increasing transformation $m$, with $\phi = m(\theta)$, $\hat\phi = m(\hat\theta)$, such that</p>
$$\hat\phi = \phi + \sigma_\phi\,(Z - z_0), \qquad \sigma_\phi = 1 + a\phi, \qquad Z\sim\N(0,1). \tag{$\star$}$$
<p>Two constants describe how far this is from the ideal "$\hat\phi\sim\N(\phi,1)$" world in which the percentile method is exact. The <strong>bias-correction</strong> $z_0$ is the median bias of $\hat\phi$ in standard-deviation units: $\P(\hat\phi < \phi) = \Phi(z_0)$. The <strong>acceleration</strong> $a$ is the rate at which the standard deviation of $\hat\phi$ changes with $\phi$ (on the normalized scale), the source of skewness on the original scale.</p>
<div class="thm" data-kind="Theorem" data-name="Exact BCa endpoints under the model (Efron, 1987)">
<p>Under $(\star)$, with $1 + a\phi > 0$ and $1 - a(z_0 + z_\alpha) > 0$, let $\hat G(t) = \P^*(\hat\theta^*\le t)$ be the bootstrap CDF of $\hat\theta^*$ generated by the model at $\hat\phi$. Define
$$\alpha_1 = \Phi\Big(z_0 + \frac{z_0 + z_\alpha}{1 - a(z_0 + z_\alpha)}\Big), \qquad \hat\theta_{\text{BC}_a}[\alpha] = \hat G^{-1}(\alpha_1).$$
Then $\P(\theta\le\hat\theta_{\text{BC}_a}[\alpha]) = \alpha$ exactly, and $z_0 = \Phi^{-1}\big(\hat G(\hat\theta)\big)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Step 1: an exact endpoint on the $\phi$ scale.</em> Let $w = z_0 + z_\alpha$ and $\hat\phi[\alpha] = \hat\phi + (1 + a\hat\phi)\,\dfrac{w}{1 - aw}$. Put $u = Z - z_0$; then by $(\star)$, $\hat\phi = \phi + (1+a\phi)u$ and $1 + a\hat\phi = (1+a\phi)(1 + au)$. So
$$\hat\phi[\alpha] - \phi = (1+a\phi)\Big[u + \frac{(1+au)w}{1-aw}\Big] = \frac{(1+a\phi)(u + w)}{1-aw}.$$
With $1+a\phi>0$ and $1-aw>0$, $\phi\le\hat\phi[\alpha] \iff u + w \ge 0 \iff Z \ge -z_\alpha$, which has probability $\Phi(z_\alpha) = \alpha$.</p>
<p><em>Step 2: find that endpoint among the bootstrap percentiles.</em> Under the model at $\hat\phi$, the bootstrap replicate is $\hat\phi^* = \hat\phi + (1+a\hat\phi)(Z^* - z_0)$, so its $\beta$-quantile is $\hat\phi + (1 + a\hat\phi)(z_\beta - z_0)$. Setting this equal to $\hat\phi[\alpha]$ gives $z_\beta = z_0 + w/(1 - aw)$, i.e. $\beta = \alpha_1$.</p>
<p><em>Step 3: transform back.</em> Quantiles commute with increasing maps: the $\alpha_1$-quantile of $\hat\theta^* = m^{-1}(\hat\phi^*)$ is $m^{-1}$ of the $\alpha_1$-quantile of $\hat\phi^*$, i.e. $m^{-1}(\hat\phi[\alpha])$. Since $\theta\le m^{-1}(\hat\phi[\alpha]) \iff \phi\le\hat\phi[\alpha]$, coverage is exactly $\alpha$. Finally $\hat G(\hat\theta) = \P^*(\hat\phi^*\le\hat\phi) = \P(Z^*\le z_0) = \Phi(z_0)$. The transformation $m$ appears nowhere in the recipe.</p>
</details>
<p>With $a = z_0 = 0$ this is the percentile interval, which is therefore exact whenever some monotone transformation makes $\hat\theta$ normal, unbiased, and of constant variance. $a = 0$, $z_0\ne0$ gives the older BC interval. Real data never satisfy $(\star)$ exactly, but DiCiccio & Efron (1992) and Hall (1988) show that a general smooth problem can be approximated by $(\star)$ to the order that matters, so that with consistent estimates of $z_0$ and $a$ (error $O_p(n^{-1/2})$, while the constants themselves are $O(n^{-1/2})$) the BC$_a$ endpoints are second-order accurate.</p>`,
      },
      {
        title: 'Estimating $z_0$ and $a$',
        html: R`
<p>The <strong>bias-correction</strong> comes straight from the bootstrap replicates:</p>
$$\hat z_0 = \Phi^{-1}\Big(\frac{\#\{b:\hat\theta^*_b < \hat\theta\} + \tfrac12\#\{b:\hat\theta^*_b = \hat\theta\}}{B}\Big).$$
<p>If half the replicates fall below $\hat\theta$, $\hat z_0 = 0$. If 60% do, $\hat z_0 = \Phi^{-1}(0.6)\approx0.25$: by analogy, $\hat\theta$ tends to fall <em>below</em> its target, so BC$_a$ moves both endpoints to higher percentiles of the replicates.</p>
<p>The <strong>acceleration</strong> is subtler, because it measures how the standard error changes with the parameter, which a single bootstrap distribution does not reveal directly. In a one-parameter family Efron showed $a \approx \frac16\,\mathrm{SKEW}_{\theta=\hat\theta}(\dot\ell_\theta)$, one-sixth of the skewness of the score. The nonparametric analogue replaces the score with the influence function and estimates it by the <a href="#/t/jackknife">jackknife</a>:</p>
$$\hat a = \frac{\sum_{i=1}^n U_i^3}{6\big(\sum_{i=1}^n U_i^2\big)^{3/2}}, \qquad U_i = (n-1)\big(\hat\theta_{(\cdot)} - \hat\theta_{(i)}\big).$$
<p>(The factor $n-1$ cancels in the ratio; it is written so that $U_i$ approximates the empirical influence value $L(X_i;\hat F_n)$.) For the mean, $U_i = X_i - \bar X$ and $\hat a = \hat\gamma/(6\sqrt n)$ with $\hat\gamma$ the sample skewness — the $n^{-1/2}$ scaling shows that $a$ is a second-order quantity, as is $z_0$.</p>
<div class="algo" data-name="BCa interval">
<ol>
<li>Draw $B$ bootstrap replicates $\hat\theta^*_1,\dots,\hat\theta^*_B$ ($B\ge2000$ recommended: the adjusted levels can be far in the tails).</li>
<li>Compute $\hat z_0$ from the fraction below $\hat\theta$, and $\hat a$ from the $n$ jackknife values.</li>
<li>For $\alpha$ and $1-\alpha$, compute $\alpha_j = \Phi\big(\hat z_0 + (\hat z_0 + z_{\alpha})/(1 - \hat a(\hat z_0 + z_{\alpha}))\big)$ (with $z_{1-\alpha}$ for the upper end).</li>
<li>Return the empirical $\alpha_1$- and $\alpha_2$-quantiles of the replicates.</li>
</ol>
</div>
<div class="ex" data-name="Reading the adjustment">
<p>Suppose $\hat z_0 = 0.1$ and $\hat a = 0.05$, for a 90% interval ($z_{0.05} = -1.645$). Lower: $w = -1.545$, $w/(1-aw) = -1.545/1.077 = -1.434$, so $\alpha_1 = \Phi(0.1 - 1.434) = \Phi(-1.334)\approx0.091$. Upper: $w = 1.745$, $w/(1 - aw) = 1.745/0.913 = 1.912$, so $\alpha_2 = \Phi(2.012)\approx0.978$. Instead of the 5th and 95th percentiles, BC$_a$ uses the 9.1th and 97.8th: both endpoints move right, and the upper one stretches further into the tail to accommodate the skewness.</p>
</div>`,
        lab: 'bca-intervals-anatomy',
      },
      {
        title: 'Coverage in practice',
        html: R`
<p>Theory predicts that on skewed data the percentile and basic intervals miss asymmetrically, while studentized and BC$_a$ intervals balance their misses. For the mean of an exponential or lognormal sample with $n$ around 10–30, the effect is large and easy to see: the true mean lies above the upper limit far more often than nominal, because the sample mean is right-skewed and small samples rarely contain the large values that pull it up.</p>
<p>The lab repeats the experiment and reports left and right misses separately (target: 2.5% each for a 95% interval). Things to try:</p>
<ul>
<li><strong>Lognormal, mean, $n=15$.</strong> Percentile and basic intervals miss mostly on the right; BC$_a$ improves; bootstrap-$t$ comes closest to balance, at the price of much wider (sometimes wild) intervals.</li>
<li><strong>Variance as the statistic.</strong> The fourth moment drives the standard error, and all methods under-cover badly for small $n$ — second-order is an asymptotic statement, and the constants can be large.</li>
<li><strong>Normal data.</strong> Everything works; $z_0$ and $a$ are near zero and BC$_a$ reduces to the percentile interval.</li>
</ul>`,
        lab: 'bca-intervals-coverage',
      },
      {
        title: 'ABC, Monte Carlo error, and failure modes',
        html: R`
<p><strong>ABC intervals.</strong> DiCiccio & Efron (1992) observed that for statistics that are smooth functions of the resampling weights, $\hat\theta(w) = T(\sum_i w_i\delta_{X_i})$, all the ingredients of BC$_a$ — $\hat\sigma$, $z_0$, $a$, and the curvature of $\hat\theta$ along the least-favorable direction — can be computed analytically by numerical differentiation of $\hat\theta(w)$ around $w = (1/n,\dots,1/n)$. The resulting <em>approximate bootstrap confidence</em> (ABC) endpoints agree with BC$_a$ to second order but require only about $2n$ evaluations of the statistic and no Monte Carlo at all. The price is that $T$ must be expressible as a smooth function of weights (e.g. weighted moments), which rules out medians and other non-smooth statistics.</p>
<p><strong>Monte Carlo error.</strong> Endpoints are tail quantiles of $B$ replicates, whose sampling error is of order $\sqrt{\beta(1-\beta)/B}/\hat g(\hat\theta^*_\beta)$, much larger than the error in a standard error estimate. BC$_a$'s adjusted levels can sit at the 0.5th or 99.5th percentile, making things worse. Use $B\ge 2000$, and check stability by recomputing with a new seed (or with the jackknife-after-bootstrap).</p>
<div class="warn" data-name="Failure modes">
<ul>
<li><strong>Large $|\hat z_0|$.</strong> If nearly all replicates fall on one side of $\hat\theta$ (e.g. a parameter on a boundary, or a discrete statistic), $\hat z_0$ is huge and $\alpha_1$ collapses to 0 or 1; the interval degenerates to the range of the replicates.</li>
<li><strong>$1 - \hat a(\hat z_0 + z_\alpha)\le 0$.</strong> The formula breaks down for extreme acceleration; this signals that the normalizing-transformation model is a poor description.</li>
<li><strong>Non-smooth statistics.</strong> Second-order theory requires an Edgeworth expansion. For the median (lattice-like behaviour of the bootstrap distribution) and for the max (where even the plain bootstrap is inconsistent), BC$_a$ does not inherit its guarantees; see <a href="#/t/subsampling">subsampling</a>.</li>
<li><strong>Small $n$.</strong> "Second-order" means the error is $c/n$, not that it is small. With $n = 10$ and heavy skewness, every method under-covers; a well-chosen <a href="#/t/parametric-bootstrap">parametric</a> model may do much better.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why do normal and percentile two-sided intervals have coverage error $O(n^{-1})$ even though their one-sided errors are $O(n^{-1/2})$?`,
        a: R`The $n^{-1/2}$ term in the Edgeworth expansion is $p_1(x)\phi(x)$ with $p_1$ even, so it contributes the same amount at the upper and lower endpoints ($x = \pm z$) and cancels when the two one-sided probabilities are subtracted. The interval is wrong in both tails, in opposite directions, by $O(n^{-1/2})$ each.` },
      { q: R`In the BC$_a$ derivation, what are $z_0$ and $a$ on the normalized scale, and what do $z_0 = a = 0$ reduce to?`,
        a: R`On the scale $\phi = m(\theta)$ the model is $\hat\phi = \phi + (1+a\phi)(Z - z_0)$: $z_0$ is the median bias of $\hat\phi$ in SD units ($\P(\hat\phi<\phi) = \Phi(z_0)$) and $a$ is the rate at which the SD of $\hat\phi$ changes with $\phi$. With both zero, $\hat\phi\sim\N(\phi,1)$ and BC$_a$ is the percentile interval, which is then exact.` },
      { q: R`If 62% of the bootstrap replicates fall below $\hat\theta$ and $\hat a = 0$, which percentiles does a 95% BC interval use?`,
        a: R`$\hat z_0 = \Phi^{-1}(0.62)\approx 0.305$. Lower level $\Phi(2\hat z_0 - 1.96) = \Phi(-1.35)\approx 0.089$; upper level $\Phi(2\hat z_0 + 1.96) = \Phi(2.57)\approx0.995$. So BC uses the 8.9th and 99.5th percentiles instead of the 2.5th and 97.5th: both endpoints move <em>up</em> the bootstrap distribution. Replicates falling below $\hat\theta$ more than half the time suggest that $\hat\theta$ itself tends to fall below $\theta$, and the correction shifts the interval upward to compensate.` },
      { q: R`Why is the bootstrap-$t$ not transformation-respecting, and why does that matter for a correlation coefficient?`,
        a: R`It bootstraps $(\hat\theta-\theta)/\hat\sigma$, whose distribution and stability depend on the scale; the inverted interval for $g(\theta)$ is not $g$ of the interval for $\theta$. For $\rho$ near $\pm1$, $\hat\sigma^*$ (proportional to $1-\hat\rho^{*2}$) can be nearly zero, giving huge $Z^*$ values and erratic intervals that can exceed $[-1,1]$. Working on Fisher's $z$ scale stabilizes the variance, and transforming back fixes the range.` },
      { q: R`For the sample mean, express $\hat a$ in terms of the sample skewness.`,
        a: R`The jackknife influence values are $U_i = X_i - \bar X$, so $\hat a = \sum(X_i-\bar X)^3/\big(6(\sum(X_i-\bar X)^2)^{3/2}\big) = \hat\gamma/(6\sqrt n)$ where $\hat\gamma = n^{-1}\sum(X_i-\bar X)^3/\hat\sigma^3$. The $n^{-1/2}$ factor shows $a$ is an $O(n^{-1/2})$ correction.` },
    ],
  });
})();

/* ---------------------------------------------------------------- parametric-bootstrap */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'parametric-bootstrap',
    domain: 'resampling',
    order: 40,
    title: 'The Parametric Bootstrap',
    subtitle: 'Simulating from the fitted model: exact pivots, small samples, and the price of misspecification',
    level: 2,
    sources: { aos: '§9.11', other: 'Efron & Tibshirani (1993, Ch. 6, 21); Davison & Hinkley (1997, §2.2, §4.2); Beran (1997)' },
    prereqs: ['bootstrap', 'mle'],
    related: ['delta-method', 'fisher-information', 'likelihood-ratio-tests', 'bca-intervals', 'simulation-studies', 'gaussian-mixtures'],
    tags: ['plug-in', 'model-based simulation', 'pivot', 'Monte Carlo test', 'misspecification'],
    summary: R`The nonparametric bootstrap plugs in $\hat F_n$; the parametric bootstrap plugs in the fitted model $F_{\hat\theta}$ and simulates new data sets from it. When the model is right, this buys a great deal: exact answers whenever a pivot exists, smooth bootstrap distributions in small samples, sensible behaviour for extremes and tail quantities, and a general way to calibrate tests (likelihood ratios with non-standard limits, goodness-of-fit with estimated parameters). When the model is wrong, it confidently reports the variability the model <em>would</em> have, not the variability the data actually have.`,
    survey: {
      task: 'Sampling distributions, CIs and test calibration under a fitted parametric model',
      assumptions: R`Correct parametric family $\{F_\theta\}$; $\hat\theta$ consistent and regular (locally equivariant)`,
      output: R`Replicates $\hat\theta^*$ from data simulated under $F_{\hat\theta}$; SEs, intervals, Monte Carlo $p$-values`,
      strengths: 'Exact for pivotal quantities; works in tiny samples and for extremes; calibrates LRTs with non-chi-square limits',
      weaknesses: R`Inherits every misspecification of the model; fails for non-regular estimators (boundary, superefficiency)`,
      complexity: R`$B\times$(simulate + refit); refitting may require iterative optimization`,
    },
    sections: [
      {
        title: 'The recipe',
        html: R`
<p>Suppose the data are modelled as $X_1,\dots,X_n\iid F_\theta$, $\theta\in\Theta\subseteq\R^k$, and $\hat\theta$ is an estimator (usually the <a href="#/t/mle">MLE</a>). We want the sampling distribution of some statistic $T_n$ — perhaps $\hat\psi = g(\hat\theta)$, a pivot such as $(\hat\psi-\psi)/\widehat{\se}$, or a test statistic.</p>
<div class="algo" data-name="Parametric bootstrap">
<ol>
<li>Fit $\hat\theta$ to the observed data.</li>
<li>For $b = 1,\dots,B$: simulate $X^*_1,\dots,X^*_n\iid F_{\hat\theta}$; recompute the statistic (including refitting $\hat\theta^*_b$ if needed) to get $T^*_{n,b}$.</li>
<li>Use the empirical distribution of $T^*_{n,1},\dots,T^*_{n,B}$ in place of the unknown sampling distribution of $T_n$ under $F_\theta$.</li>
</ol>
</div>
<p>This is the plug-in principle once more, but the plug-in is $F_{\hat\theta}$, a smooth, model-respecting estimate of $F$, rather than the discrete $\hat F_n$. The two approximations stack as before: a statistical error ($F_{\hat\theta}$ vs. $F_\theta$, of order $n^{-1/2}$ when the model is right, and $O(1)$ when it is wrong) and a Monte Carlo error that vanishes as $B\to\infty$.</p>
<div class="note">
<p>The parametric bootstrap is how most of us first met simulation-based inference without calling it that: "fit the model, simulate from the fit, see how the estimator behaves." The only new element compared with a <a href="#/t/simulation-studies">simulation study</a> is that the parameter value used to simulate is the <em>estimate</em> from the data at hand.</p>
</div>`,
      },
      {
        title: 'Exactness under pivots',
        html: R`
<p>The cleanest justification for the parametric bootstrap needs no asymptotics at all.</p>
<div class="thm" data-kind="Proposition" data-name="Pivots are reproduced exactly">
<p>Let $Q = Q(X_1,\dots,X_n;\theta)$ be an exact pivot: its distribution $H$ under $F_\theta$ is the same for every $\theta\in\Theta$. Then the parametric bootstrap distribution of $Q^* = Q(X_1^*,\dots,X_n^*;\hat\theta)$ is exactly $H$, whatever the value of $\hat\theta$. Consequently a confidence set obtained by inverting $Q$ with bootstrap quantiles has exact coverage, up to Monte Carlo error.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Conditional on the data, $X^*_i\iid F_{\hat\theta}$, so the conditional law of $Q(X^*;\hat\theta)$ is the law of $Q(X;\vartheta)$ under $F_\vartheta$ evaluated at $\vartheta = \hat\theta$. Since that law is $H$ for every $\vartheta$, it is $H$. If $h_\beta$ denotes the $\beta$-quantile of $H$, then $\{\theta: h_{\alpha/2}\le Q(X;\theta)\le h_{1-\alpha/2}\}$ has probability $1-\alpha$ under $F_\theta$ (exactly, for continuous $H$), and the bootstrap supplies $h_\beta$ up to Monte Carlo error.</p>
</details>
<div class="ex" data-name="Normal mean: the bootstrap rediscovers Student">
<p>For $X_i\iid\N(\mu,\sigma^2)$, $Q = \sqrt n(\bar X - \mu)/s$ is pivotal. Parametric bootstrap-$t$ with $X^*\sim\N(\bar X, \hat\sigma^2)$ produces $Q^*\sim t_{n-1}$ exactly, and the bootstrap-$t$ interval is Student's $t$ interval. The nonparametric bootstrap-$t$ only approximates it.</p>
</div>
<div class="ex" data-name="Exponential rate: percentile vs. pivot">
<p>Let $X_i\iid\text{Exp}(\lambda)$ and $\hat\lambda = 1/\bar X$. Since $\lambda\bar X\sim\text{Gamma}(n,1/n)$ for every $\lambda$, the ratio $Q = \hat\lambda/\lambda$ is pivotal. The parametric bootstrap gives $\hat\lambda^*/\hat\lambda\sim Q$ exactly, so the interval
$$\Big(\frac{\hat\lambda}{h_{1-\alpha/2}},\ \frac{\hat\lambda}{h_{\alpha/2}}\Big) = \Big(\frac{\hat\lambda^2}{\hat\lambda^*_{(1-\alpha/2)}},\ \frac{\hat\lambda^2}{\hat\lambda^*_{(\alpha/2)}}\Big)$$
is exact. The parametric <em>percentile</em> interval $(\hat\lambda^*_{(\alpha/2)},\hat\lambda^*_{(1-\alpha/2)}) = (\hat\lambda h_{\alpha/2}, \hat\lambda h_{1-\alpha/2})$ uses the same draws but the wrong inversion: it would be exact only if $\log Q$ were symmetric about 0, and it is not ($Q$ is right-skewed, since $1/\text{Gamma}$ is). So even a perfect model does not rescue a poorly chosen interval construction — though a <a href="#/t/bca-intervals">BC$_a$</a> correction largely would, since it targets precisely this bias and skewness.</p>
</div>`,
        lab: 'parametric-bootstrap-exponential',
      },
      {
        title: 'Asymptotics: the delta method by simulation',
        html: R`
<p>Without a pivot, the parametric bootstrap is justified asymptotically. In a regular model the MLE satisfies $\sqrt n(\hat\theta-\theta)\dto\N(0, I(\theta)^{-1})$, and $\hat\psi = g(\hat\theta)$ has asymptotic variance $\nabla g^\top I(\theta)^{-1}\nabla g/n$ by the <a href="#/t/delta-method">delta method</a>. The parametric bootstrap reproduces this limit — and more.</p>
<div class="thm" data-kind="Theorem" data-name="Consistency of the parametric bootstrap (Beran, 1997)">
<p>Suppose $\hat\theta_n$ is <em>regular</em> at $\theta$: for every $h$, under $F_{\theta + h/\sqrt n}$ the law of $\sqrt n(\hat\theta_n - \theta - h/\sqrt n)$ converges to the same limit $J_\theta$, and $J_\vartheta$ is continuous in $\vartheta$. Then, in probability,
$$\mathcal L^*\big(\sqrt n(\hat\theta^*_n - \hat\theta_n)\big) \to J_\theta,$$
the limit law of $\sqrt n(\hat\theta_n - \theta)$. Beran also shows that, in locally asymptotically normal models, the failure of this local equivariance is essentially what makes the parametric bootstrap fail.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Since $\sqrt n(\hat\theta_n - \theta) = O_p(1)$, the bootstrap parameter $\hat\theta_n$ is of the form $\theta + h_n/\sqrt n$ with $h_n$ tight. Regularity (uniformly over $h$ in compacts, which follows from local asymptotic normality plus Le Cam's third lemma for MLEs in smooth models) gives that the law of $\sqrt n(\hat\theta^*_n - \hat\theta_n)$ under $F_{\theta + h_n/\sqrt n}$ is close to $J_\theta$ along any such sequence. A subsequence argument converts this into convergence in probability.</p>
</details>
<p>For a smooth $g$, applying the delta method in the bootstrap world shows that the parametric bootstrap SE of $g(\hat\theta)$ converges to the delta-method SE. So the parametric bootstrap is the delta method computed by simulation — but without the linearization at finite $n$. It captures skewness, bias, and the curvature of $g$ that a first-order delta method discards, which is why it often gives better intervals in small samples even when both are "asymptotically equivalent."</p>
<div class="warn" data-name="Non-regular estimators">
<p>Regularity is not a technicality. At a boundary (e.g. $\theta\ge0$ with $\theta = 0$ and $\hat\theta = \max(\bar X,0)$) or for superefficient estimators (Hodges), the limit law depends on the local parameter $h$, and simulating at $\hat\theta$ (which is a random $h$) gives the wrong answer. <a href="#/t/subsampling">Subsampling</a> and the $m$-out-of-$n$ bootstrap remain valid there.</p>
</div>`,
      },
      {
        title: 'When it beats the nonparametric bootstrap',
        html: R`
<p>When the model is correct, the parametric bootstrap estimates the same target as the nonparametric one, but typically better:</p>
<ul>
<li><strong>Small samples.</strong> $\hat F_n$ with $n = 8$ has eight atoms; $F_{\hat\theta}$ is a smooth distribution. Nonparametric bootstrap distributions of statistics like the median are lumpy (a handful of distinct values); parametric ones are continuous.</li>
<li><strong>Efficiency of the variance estimate.</strong> For an exponential sample, the parametric bootstrap SE of $\bar X$ is $\bar X/\sqrt n$; the nonparametric one is $\hat\sigma/\sqrt n$. Both are consistent, but $\Var(\bar X)$ involves only the second moment while $\Var(\hat\sigma)$ involves the fourth, so the model-based SE is far less noisy.</li>
<li><strong>Extremes and tails.</strong> Statistics that depend on the tail of $F$ beyond the data — the maximum, a high quantile, a return level — cannot be resampled from $\hat F_n$, which has no mass beyond $X_{(n)}$. For $X_i\iid\text{Unif}(0,\theta)$ and $\hat\theta = X_{(n)}$, the quantity $X_{(n)}/\theta$ is pivotal (it has the Beta$(n,1)$ law), so the parametric bootstrap is <em>exact</em> where the nonparametric bootstrap is inconsistent.</li>
<li><strong>Discrete and censored data.</strong> With censoring, resampling must respect the censoring mechanism; simulating from a fitted survival model and censoring distribution is often simpler.</li>
<li><strong>Testing</strong>: simulate from the fitted <em>null</em> model — next section.</li>
</ul>`,
        lab: 'parametric-bootstrap-max',
      },
      {
        title: 'Parametric bootstrap tests',
        html: R`
<p>To test $H_0:\theta\in\Theta_0$ with statistic $T$ (large values evidence against $H_0$), fit the model <em>under the null</em>, $\hat\theta_0$, simulate $B$ data sets from $F_{\hat\theta_0}$, recompute $T$ on each, and report
$$\hat p = \frac{1 + \#\{b: T^*_b\ge T_{\text{obs}}\}}{1 + B}.$$
If $T$ is pivotal under $H_0$ this is an exact Monte Carlo test (Barnard, 1963; see <a href="#/t/permutation-tests">permutation tests</a> for the $(1+\#)/(1+B)$ argument). If $T$ is only asymptotically pivotal, the error in rejection probability is typically of smaller order than the asymptotic $\chi^2$ approximation's (Beran, 1988).</p>
<div class="ex" data-name="Goodness of fit with estimated parameters">
<p>The Kolmogorov–Smirnov statistic $D = \sup_x|\hat F_n(x) - F_{\hat\theta}(x)|$ with $\hat\theta$ estimated from the same data is <em>not</em> distributed as the classical KS null law: fitting pulls $F_{\hat\theta}$ toward $\hat F_n$, so $D$ is stochastically smaller and the classical $p$-values are conservative (Lilliefors, 1967, tabulated the normal case by simulation). The parametric bootstrap automates Lilliefors for any family: simulate from $F_{\hat\theta}$, <em>re-estimate</em> $\hat\theta^*$ on each simulated sample, compute $D^* = \sup|\hat F^*_n - F_{\hat\theta^*}|$. Forgetting the re-estimation step reproduces the original error.</p>
</div>
<div class="ex" data-name="How many mixture components?">
<p>Testing one versus two components in a <a href="#/t/gaussian-mixtures">Gaussian mixture</a> by a <a href="#/t/likelihood-ratio-tests">likelihood ratio</a> violates Wilks' conditions: under $H_0$ the extra parameters are not identified (mixing weight at 0 or equal means), and $2\log\Lambda$ does not converge to $\chi^2$ with any obvious degrees of freedom. McLachlan (1987) proposed the parametric bootstrap LRT: simulate from the fitted one-component model, fit both models to each simulated set, and use the simulated $2\log\Lambda^*$ distribution. It is standard practice in mixture modelling.</p>
</div>`,
      },
      {
        title: 'Misspecification: the main risk',
        html: R`
<p>Everything above assumed $F = F_\theta$ for some $\theta$. If not, $\hat\theta$ converges to the pseudo-true value $\theta^\dagger$ minimizing $\KL(F\,\|\,F_\theta)$, and the true asymptotic variance of $\sqrt n(\hat\theta-\theta^\dagger)$ is the <em>sandwich</em> $A^{-1}BA^{-1}$, with $A = -\E_F\nabla^2\log f_{\theta^\dagger}(X)$ and $B = \E_F[\nabla\log f_{\theta^\dagger}\nabla\log f_{\theta^\dagger}^\top]$. The parametric bootstrap simulates from $F_{\hat\theta}$, under which the information equality $A = B$ holds, so it converges to $A^{-1}$ — the model-based variance, not the sandwich.</p>
<div class="ex" data-name="Overdispersed counts">
<p>Fit a Poisson model to counts whose true variance is $\phi\mu$ with $\phi = 3$. The MLE of the mean is $\bar X$, a perfectly good estimator, with true SE $\sqrt{3\mu/n}$. The parametric bootstrap simulates Poisson$(\bar X)$ samples and reports SE $\approx\sqrt{\bar X/n}$, too small by $\sqrt3$; nominal 95% intervals cover about $\P(|Z|\le1.96/\sqrt3) = \P(|Z|\le 1.13)\approx 74\%$ of the time. The nonparametric bootstrap, which does not care what the model says, gets $\sqrt{3\mu/n}$ right. See <a href="#/t/overdispersion">overdispersion</a>.</p>
</div>
<div class="warn" data-name="Practical advice">
<ul>
<li>Run both bootstraps. Agreement is reassuring; disagreement is a misspecification diagnostic (the same idea as White's (1982) information-matrix test, which compares $A$ with $B$).</li>
<li>Prefer the parametric bootstrap when the model is well-founded (physical mechanism, design-based randomness) or when the target is inherently model-based (tail quantiles, extrapolation).</li>
<li>For regression, the <a href="#/t/bootstrap-regression">residual bootstrap</a> is a semiparametric compromise: parametric mean structure, nonparametric errors.</li>
<li>A <em>smoothed</em> bootstrap (resample from a kernel density estimate) is another compromise; it helps for quantiles in small samples.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why is the parametric bootstrap distribution of a pivot exact, regardless of how far $\hat\theta$ is from $\theta$?`,
        a: R`Because the bootstrap simulates from $F_{\hat\theta}$, a member of the model, and the pivot's distribution is the same under every member of the model. The only thing estimation error could affect is which member we simulate from, and that does not matter for a pivot.` },
      { q: R`For exponential data, the parametric bootstrap reproduces the law of $\hat\lambda/\lambda$ exactly. Why is the parametric percentile interval $(\hat\lambda^*_{(\alpha/2)},\hat\lambda^*_{(1-\alpha/2)})$ nevertheless not exact?`,
        a: R`The exact interval inverts the pivot: $\lambda\in(\hat\lambda/h_{1-\alpha/2}, \hat\lambda/h_{\alpha/2})$. The percentile interval is $(\hat\lambda h_{\alpha/2}, \hat\lambda h_{1-\alpha/2})$, which coincides only if $h_{\alpha/2}h_{1-\alpha/2} = 1$, i.e. if $\log Q$ were symmetric. Since $Q = \hat\lambda/\lambda$ is an inverse-gamma variable, right-skewed, $h_{\alpha/2}h_{1-\alpha/2} > 1$ and the percentile interval sits too far right. With $n = 5$ and $\alpha = 0.05$: $h_{0.025}\approx0.49$ and $h_{0.975}\approx3.08$, so it misses with $\lambda$ below the interval about 10% of the time and above it almost never (under 0.1%).` },
      { q: R`Your Poisson regression's parametric bootstrap SEs are much smaller than the nonparametric (pairs) bootstrap SEs. What is the likely explanation and which should you report?`,
        a: R`Overdispersion: the parametric bootstrap simulates Poisson data, whose variance equals the mean, and so reports the model-based variance $A^{-1}$; the pairs bootstrap estimates the sandwich, which is correct under misspecification. Report the robust (nonparametric or sandwich) SEs, or refit with a model that allows overdispersion (quasi-Poisson, negative binomial).` },
      { q: R`In a parametric bootstrap goodness-of-fit test with estimated parameters, why must $\hat\theta$ be re-estimated on each simulated sample?`,
        a: R`The observed statistic $D = \sup|\hat F_n - F_{\hat\theta}|$ was computed with parameters fitted to the same data, which makes it systematically small. To reproduce its null distribution, each simulated $D^*$ must undergo the same fitting step, $D^* = \sup|\hat F^*_n - F_{\hat\theta^*}|$. Using the fixed $\hat\theta$ would compare against a stochastically larger null and give conservative $p$-values.` },
    ],
  });
})();

/* ---------------------------------------------------------------- bootstrap-regression */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'bootstrap-regression',
    domain: 'resampling',
    order: 50,
    title: 'Bootstrapping Regression: Pairs, Residual & Wild',
    subtitle: 'Three resampling schemes, three models of where the randomness lives',
    level: 2,
    sources: { other: 'Freedman (1981); Wu (1986); Liu (1988); Mammen (1993); Davison & Hinkley (1997, Ch. 6); Davidson & Flachaire (2008); MacKinnon (2013)' },
    prereqs: ['bootstrap', 'regression-inference'],
    related: ['ols-geometry', 'regression-diagnostics', 'weighted-gls', 'parametric-bootstrap', 'block-bootstrap', 'glm-inference'],
    tags: ['heteroscedasticity', 'sandwich', 'wild bootstrap', 'pairs bootstrap', 'residual bootstrap'],
    summary: R`In a regression $y_i = x_i^\top\beta + \varepsilon_i$ there is more than one thing to resample, and each choice encodes a different belief about the data-generating process. Resampling residuals treats the design as fixed and the errors as IID; it reproduces the classical $\sigma^2(X^\top X)^{-1}$ and fails under heteroscedasticity. Resampling $(x_i, y_i)$ pairs treats the rows as IID draws and estimates the heteroscedasticity-robust sandwich. The wild bootstrap keeps the design fixed but multiplies each residual by an independent random sign-like weight, matching the sandwich exactly while conditioning on $X$. Knowing which matrix each scheme estimates is the whole game.`,
    survey: {
      task: 'Standard errors, intervals and tests for regression coefficients without (or with weaker) distributional assumptions',
      assumptions: R`Residual: fixed $X$, IID errors. Pairs: IID rows. Wild: fixed $X$, independent (possibly heteroscedastic) errors`,
      output: R`Bootstrap distribution of $\hat\beta^*$; SEs, percentile / bootstrap-$t$ intervals, bootstrap tests`,
      strengths: 'Pairs and wild are valid under heteroscedasticity; wild bootstrap-t gives refinements over HC sandwich SEs',
      weaknesses: 'Residual fails under heteroscedasticity; pairs can produce singular designs; none handles dependence',
      complexity: R`$B$ least-squares fits; residual and wild reuse $(X^\top X)^{-1}X^\top$, so each costs $O(np)$`,
    },
    sections: [
      {
        title: 'What is random in a regression?',
        html: R`
<p>Let $y = X\beta + \varepsilon$ with $X$ an $n\times p$ design of full rank, OLS estimate $\hat\beta = (X^\top X)^{-1}X^\top y$, fitted values $\hat y = Hy$ with $H = X(X^\top X)^{-1}X^\top$, and residuals $e = y - \hat y$. Two sampling models are common:</p>
<div class="cols">
<div><h4>Fixed design</h4><p>$X$ is chosen by the experimenter (or we condition on it); only $\varepsilon$ is random. Inference is conditional on $X$. Natural for designed experiments.</p></div>
<div><h4>Random design</h4><p>Rows $(x_i, y_i)$ are IID draws from a joint distribution. $\beta$ is defined as the best linear predictor $\beta = (\E xx^\top)^{-1}\E xy$, whether or not the conditional mean is linear.</p></div>
</div>
<p>The OLS covariance has a different form depending on what we assume about the errors. Under homoscedastic IID errors, $\Cov(\hat\beta\mid X) = \sigma^2(X^\top X)^{-1}$. In general (independent errors with $\Var(\varepsilon_i\mid X) = \sigma_i^2$),</p>
$$\Cov(\hat\beta\mid X) = (X^\top X)^{-1}\Big(\sum_{i=1}^n\sigma_i^2 x_ix_i^\top\Big)(X^\top X)^{-1},$$
<p>estimated by the White (1980) sandwich $\widehat V_{\text{HC0}} = (X^\top X)^{-1}\big(\sum_i e_i^2x_ix_i^\top\big)(X^\top X)^{-1}$ (and its small-sample refinements HC2, HC3 that divide $e_i^2$ by $1-h_{ii}$ or $(1-h_{ii})^2$). The three bootstraps below estimate the classical matrix, the sandwich, and the sandwich, respectively.</p>`,
      },
      {
        title: 'The residual bootstrap',
        html: R`
<div class="algo" data-name="Residual bootstrap">
<ol>
<li>Fit OLS; form residuals $e_i$, centered if the model has no intercept, and optionally rescaled $\tilde e_i = e_i\sqrt{n/(n-p)}$ or $\tilde e_i = e_i/\sqrt{1-h_{ii}}$ (then re-centered).</li>
<li>For each $b$: draw $e^*_1,\dots,e^*_n$ IID from $\{\tilde e_i\}$; set $y^* = X\hat\beta + e^*$; compute $\hat\beta^* = (X^\top X)^{-1}X^\top y^*$.</li>
</ol>
</div>
<div class="thm" data-kind="Proposition" data-name="What the residual bootstrap estimates">
<p>With $e^*_i$ IID from the centered residuals $\{e_i\}$ (no rescaling), $\E^*\hat\beta^* = \hat\beta$ and
$$\Cov^*(\hat\beta^*) = \hat\sigma^2(X^\top X)^{-1},\qquad \hat\sigma^2 = \frac1n\sum_{i=1}^n e_i^2.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\hat\beta^* - \hat\beta = (X^\top X)^{-1}X^\top e^*$. The $e^*_i$ are IID with $\E^*e^*_i = \bar e = 0$ and $\Var^*e^*_i = \frac1n\sum e_i^2 = \hat\sigma^2$, so $\Cov^*(e^*) = \hat\sigma^2 I_n$ and $\Cov^*(\hat\beta^*) = (X^\top X)^{-1}X^\top(\hat\sigma^2 I)X(X^\top X)^{-1} = \hat\sigma^2(X^\top X)^{-1}$.</p>
</details>
<p>So the residual bootstrap reproduces the classical OLS covariance (with divisor $n$; the $\sqrt{n/(n-p)}$ rescaling gives the usual $s^2$). Its value lies not in the covariance but in the <em>shape</em>: with skewed or heavy-tailed errors and small $n$, the bootstrap distribution of $\hat\beta^*$ or of a studentized coefficient captures non-normality that the $t$ approximation ignores. Freedman (1981) proved consistency for fixed $p$ under IID errors with finite variance and $X^\top X/n\to$ a positive definite limit; Bickel & Freedman (1983) and Mammen (1993) extend it to $p\to\infty$ with $p/n\to0$ (with conditions on leverage).</p>
<div class="warn" data-name="Heteroscedasticity breaks it">
<p>Shuffling residuals across observations assigns a large residual from a high-variance region to a low-variance one and vice versa, homogenizing the variance. The bootstrap then estimates $\bar\sigma^2(X^\top X)^{-1}$ with $\bar\sigma^2$ the average error variance — wrong whenever $\sigma_i^2$ correlates with the leverage-weighted $x_ix_i^\top$.</p>
</div>
<div class="ex" data-name="How wrong?">
<p>Simple regression, slope $\beta_1$, with $d_i = x_i - \bar x$ and $\sigma_i^2 = c\,d_i^2$ (variance growing away from the center). The true variance is $\sum d_i^2\sigma_i^2/(\sum d_i^2)^2 = c\sum d_i^4/(\sum d_i^2)^2$; the classical (and residual-bootstrap) answer is approximately $\bar\sigma^2/\sum d_i^2 = c\,\frac1n\sum d_i^2/\sum d_i^2 = c/n$. Their ratio is
$$\frac{\text{true}}{\text{classical}} = \frac{n\sum d_i^4}{(\sum d_i^2)^2} = \text{kurtosis of the design},$$
which is 1.8 for a uniform design and 3 for a normal design. The classical SE is too small by $\sqrt3\approx1.73$ for normally spread $x$, and 95% intervals cover only about $\P(|Z|\le1.96/\sqrt3)\approx74\%$.</p>
</div>`,
      },
      {
        title: 'The pairs bootstrap',
        html: R`
<div class="algo" data-name="Pairs (case) bootstrap">
<ol>
<li>For each $b$: draw $n$ rows $(x^*_i, y^*_i)$ with replacement from $\{(x_i,y_i)\}$; compute $\hat\beta^* = (X^{*\top}X^*)^{-1}X^{*\top}y^*$.</li>
</ol>
</div>
<p>This is the ordinary <a href="#/t/bootstrap">bootstrap</a> applied to the rows, and its justification is the random-design model: $\hat\beta = \hat A^{-1}\hat c$ with $\hat A = \frac1n\sum x_ix_i^\top$, $\hat c = \frac1n\sum x_iy_i$, a smooth function of sample means.</p>
<div class="thm" data-kind="Theorem" data-name="Pairs bootstrap estimates the sandwich (Freedman, 1981)">
<p>Let $(x_i,y_i)$ be IID with $A = \E xx^\top$ positive definite, $\beta = A^{-1}\E xy$, $\varepsilon = y - x^\top\beta$, and $\E\|x\|^4<\infty$, $\E\varepsilon^4 < \infty$. Then $\sqrt n(\hat\beta-\beta)\dto\N(0, A^{-1}BA^{-1})$ with $B = \E[\varepsilon^2xx^\top]$, and conditionally on the data, almost surely, $\sqrt n(\hat\beta^*-\hat\beta)$ has the same limit.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Write $y^*_i = x^{*\top}_i\hat\beta + e^*_i$ where $(x^*_i, e^*_i)$ is a resampled (row, residual) pair. Then $\hat\beta^*-\hat\beta = \big(\frac1n\sum x^*_ix^{*\top}_i\big)^{-1}\frac1n\sum x^*_ie^*_i$. The normal equations give $\sum_ix_ie_i = 0$, so the resampled vectors $x^*_ie^*_i$ are IID with bootstrap mean zero and covariance $\frac1n\sum_ie_i^2x_ix_i^\top\to B$ a.s. A conditional (Lindeberg) CLT gives $n^{-1/2}\sum x^*_ie^*_i\dto\N(0,B)$, and $\frac1n\sum x^*_ix^{*\top}_i\to A$ in conditional probability; Slutsky finishes. On the real-world side the same algebra with $\varepsilon_i$ in place of $e_i$ gives the stated limit.</p>
</details>
<p>Two consequences. First, the pairs bootstrap is valid under arbitrary heteroscedasticity, because the variance pattern travels with each row. Second, it is valid even if $\E(y\mid x)$ is <em>not</em> linear: it then provides inference for the best linear predictor, with the misspecification error absorbed into $\varepsilon$. This is the bootstrap analogue of the sandwich's robustness.</p>
<div class="warn" data-name="Costs of resampling the design">
<ul>
<li>Inference is unconditional on $X$, which is inappropriate for designed experiments.</li>
<li>Resamples can be degenerate: a rare category, an isolated high-leverage point, or a binary covariate with few ones may be absent from $X^*$, making $X^{*\top}X^*$ singular. Common remedy: discard (and report) such resamples, or switch to the wild bootstrap.</li>
<li>With influential points, the bootstrap distribution can be multimodal (the point is in or out), which is honest but makes percentile intervals erratic.</li>
</ul>
</div>`,
      },
      {
        title: 'The wild bootstrap',
        html: R`
<p>The wild bootstrap (Wu, 1986; Liu, 1988; Mammen, 1993) keeps each residual attached to its own row — preserving the heteroscedasticity pattern — but randomizes its sign and scale.</p>
<div class="algo" data-name="Wild bootstrap">
<ol>
<li>Fit OLS; optionally transform residuals $\tilde e_i = e_i/\sqrt{1-h_{ii}}$ (HC2-type) or $e_i/(1-h_{ii})$ (HC3-type).</li>
<li>For each $b$: draw $v_1,\dots,v_n$ IID with $\E v = 0$, $\E v^2 = 1$; set $y^*_i = x_i^\top\hat\beta + \tilde e_iv_i$; refit with the <em>same</em> $X$.</li>
</ol>
</div>
<div class="thm" data-kind="Proposition" data-name="Wild bootstrap reproduces the sandwich exactly">
<p>$\E^*\hat\beta^* = \hat\beta$ and $\Cov^*(\hat\beta^*) = (X^\top X)^{-1}\big(\sum_i\tilde e_i^2x_ix_i^\top\big)(X^\top X)^{-1}$: the HC0 sandwich for $\tilde e = e$, HC2 or HC3 for the transformed residuals.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\hat\beta^*-\hat\beta = (X^\top X)^{-1}X^\top u^*$ with $u^*_i = \tilde e_iv_i$. The $u^*_i$ are independent with mean $\tilde e_i\E v = 0$ and variance $\tilde e_i^2\E v^2 = \tilde e_i^2$, so $\Cov^*(u^*) = \diag(\tilde e_i^2)$ and $\Cov^*(\hat\beta^*) = (X^\top X)^{-1}X^\top\diag(\tilde e_i^2)X(X^\top X)^{-1}$.</p>
</details>
<p>Why bootstrap at all, if the covariance is just the sandwich? Because the wild bootstrap reproduces more than the covariance. Choosing $v$ with $\E v^3 = 1$ also matches the third moment of $\tilde e_iv_i$ to that of $\tilde e_i$, which is what an Edgeworth argument needs for a refinement. Mammen's two-point distribution does this:</p>
$$v = \begin{cases}-(\sqrt5-1)/2 \approx -0.618 & \text{w.p. } (\sqrt5+1)/(2\sqrt5)\approx0.724,\\ (\sqrt5+1)/2\approx1.618 & \text{w.p. } (\sqrt5-1)/(2\sqrt5)\approx0.276,\end{cases}\qquad \E v = 0,\ \E v^2 = \E v^3 = 1.$$
<p>The <strong>Rademacher</strong> choice $v = \pm1$ with probability $\frac12$ has $\E v^3 = 0$ but $\E v^4 = 1$ (the smallest possible), and simulation evidence (Davidson & Flachaire, 2008) favours it for bootstrap-$t$ tests, especially with symmetric errors. The wild bootstrap-$t$ — studentizing each $\hat\beta^*$ with its own HC standard error — has smaller error in rejection probability than the HC sandwich $t$-test with normal critical values, whose finite-sample size distortions in the presence of high-leverage points are notorious.</p>
<div class="note" data-name="Wild cluster bootstrap">
<p>With clustered errors, draw one $v_g$ per <em>cluster</em> and multiply all residuals in cluster $g$ by it. This preserves arbitrary within-cluster correlation and is the standard remedy when the number of clusters is small (Cameron, Gelbach & Miller, 2008), where cluster-robust sandwich SEs badly over-reject.</p>
</div>`,
        lab: 'bootstrap-regression-hetero',
      },
      {
        title: 'Choosing a scheme; tests and prediction',
        html: R`
<table>
<thead><tr><th>Scheme</th><th>Bootstrap covariance</th><th>Valid under heteroscedasticity?</th><th>Conditions on $X$?</th><th>Typical use</th></tr></thead>
<tbody>
<tr><td>Residual</td><td>$\hat\sigma^2(X^\top X)^{-1}$</td><td>No</td><td>Yes</td><td>Designed experiments, homoscedastic non-normal errors</td></tr>
<tr><td>Pairs</td><td>$\approx$ sandwich</td><td>Yes</td><td>No</td><td>Observational data, misspecified mean</td></tr>
<tr><td>Wild</td><td>sandwich (exactly)</td><td>Yes</td><td>Yes</td><td>Heteroscedastic fixed design; small samples; clusters</td></tr>
<tr><td><a href="#/t/parametric-bootstrap">Parametric</a></td><td>model-based</td><td>Only if modelled</td><td>Yes</td><td>GLMs with trusted variance function</td></tr>
</tbody>
</table>
<p><strong>Bootstrap tests should impose the null.</strong> To test $H_0:\beta_j = 0$, fit the restricted model, and generate $y^*$ from the <em>restricted</em> fit and <em>restricted</em> residuals (residual or wild scheme). The bootstrap world then satisfies $H_0$, and the bootstrap distribution of the $t$-statistic approximates its null distribution. Resampling unrestricted residuals and centering at $\hat\beta_j$ is also valid asymptotically, but imposing the null typically gives better finite-sample size (Davidson & MacKinnon, 1999).</p>
<p><strong>Prediction intervals</strong> need two sources of variability: estimation error in $x_0^\top\hat\beta$ and the new error $\varepsilon_0$. The residual bootstrap handles both by computing $x_0^\top(\hat\beta-\hat\beta^*) + e^*_0$ with $e^*_0$ an extra draw from the (leverage-adjusted) residuals; under heteroscedasticity the new error must be drawn from residuals near $x_0$, or one turns to conformal methods.</p>
<div class="warn" data-name="Common pitfalls">
<ul>
<li>Using raw residuals in small samples: they are shrunk ($\Var e_i = \sigma^2(1-h_{ii})$), so bootstrap variances are biased down. Rescale by $\sqrt{1-h_{ii}}$.</li>
<li>Bootstrapping a model after data-driven variable selection without repeating the selection inside each resample: the SEs ignore selection uncertainty.</li>
<li>Any of the three schemes with serially correlated errors: resample blocks of residuals or rows instead (<a href="#/t/block-bootstrap">block bootstrap</a>), or use the wild cluster bootstrap with time-based clusters.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Which covariance matrix does each of the residual, pairs, and wild bootstraps estimate for $\hat\beta$?`,
        a: R`Residual: the classical $\hat\sigma^2(X^\top X)^{-1}$ (exactly, with $\hat\sigma^2 = n^{-1}\sum e_i^2$). Wild: the HC sandwich $(X^\top X)^{-1}\sum\tilde e_i^2x_ix_i^\top(X^\top X)^{-1}$, exactly. Pairs: the sandwich asymptotically, with the design also varying across resamples.` },
      { q: R`Why does the residual bootstrap underestimate the slope's SE when the error variance grows with $|x - \bar x|$?`,
        a: R`Residuals are reassigned at random across observations, so each bootstrap error has the average variance $\bar\sigma^2$ regardless of $x_i$. But the slope's variance weights $\sigma_i^2$ by $d_i^2$, and high-leverage (large $|d_i|$) points have the largest $\sigma_i^2$. Spreading variance evenly understates the variance exactly where it matters; the ratio of true to classical variance is the design kurtosis $n\sum d^4/(\sum d^2)^2$ when $\sigma_i^2\propto d_i^2$.` },
      { q: R`What properties should the wild bootstrap weights $v_i$ have, and what does $\E v^3 = 1$ buy?`,
        a: R`$\E v = 0$ (so the bootstrap errors have mean zero row by row) and $\E v^2 = 1$ (so their variance is $\tilde e_i^2$, matching the sandwich). $\E v^3 = 1$ additionally matches the third moment of each error, which is what an Edgeworth expansion of a studentized statistic needs for a second-order refinement, as with Mammen's two-point weights.` },
      { q: R`You bootstrap a logistic regression by resampling pairs and 3% of resamples produce perfect separation. What does this indicate, and what can you do?`,
        a: R`Some resamples omit the few observations that prevent separation (e.g. rare events in a category), so the MLE diverges. It signals that inference is fragile — the estimate depends heavily on a handful of points. Options: report the fraction and discard them (biasing the SE downward), use a penalized (Firth) estimator inside the bootstrap, or use a wild/score bootstrap that keeps the design fixed.` },
    ],
  });
})();

/* ---------------------------------------------------------------- block-bootstrap */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'block-bootstrap',
    domain: 'resampling',
    order: 60,
    title: 'Resampling Dependent Data: Block Bootstraps',
    subtitle: 'Moving, circular and stationary blocks, and the bias–variance tradeoff that sets the block length',
    level: 3,
    sources: { other: 'Carlstein (1986); Künsch (1989); Liu & Singh (1992); Politis & Romano (1992, 1994); Hall, Horowitz & Jing (1995); Lahiri (2003); Politis & White (2004)' },
    prereqs: ['bootstrap', 'time-series'],
    related: ['subsampling', 'mcmc-diagnostics', 'bootstrap-regression', 'stationary-distributions', 'kernel-density'],
    tags: ['time series', 'long-run variance', 'block length', 'mixing', 'stationary bootstrap'],
    summary: R`The IID bootstrap scrambles the time order and so destroys exactly the autocorrelation that determines the variance of a time-series statistic: for an AR(1) with $\phi = 0.8$ it underestimates the variance of the mean by a factor of nine. Block bootstraps resample contiguous stretches of the series, keeping dependence within blocks intact and letting blocks play the role of approximately independent units. The block length $\ell$ must grow, but slower than $n$: too short and the long-range correlation is lost (bias $\propto 1/\ell$); too long and there are too few distinct blocks (variance $\propto\ell/n$). Balancing the two gives the classical $\ell\propto n^{1/3}$ rule.`,
    survey: {
      task: 'Standard errors and confidence intervals for statistics of stationary, weakly dependent series',
      assumptions: R`Strict stationarity; mixing (dependence decays); $\ell\to\infty$, $\ell/n\to0$; finite $2+\delta$ moments`,
      output: R`Resampled series; bootstrap distribution of $T_n^*$; HAC-type variance estimate`,
      strengths: 'Model-free: no ARMA fit needed; handles nonlinear statistics; easy to implement',
      weaknesses: R`Block length matters and must be tuned; join points create artifacts; only modest higher-order gains; fails for long memory and nonstationarity`,
      complexity: R`$O(Bn)$ plus $B$ statistic evaluations`,
    },
    sections: [
      {
        title: 'Why the IID bootstrap fails for dependent data',
        html: R`
<p>Let $X_1,\dots,X_n$ be a stationary <a href="#/t/time-series">time series</a> with mean $\mu$ and autocovariance $\gamma(h) = \Cov(X_t, X_{t+h})$. The variance of the sample mean is</p>
$$n\Var(\bar X_n) = \sum_{|h| < n}\Big(1-\frac{|h|}{n}\Big)\gamma(h) \;\longrightarrow\; \sigma^2_\infty := \sum_{h=-\infty}^{\infty}\gamma(h) = \gamma(0) + 2\sum_{h\ge1}\gamma(h),$$
<p>the <strong>long-run variance</strong> (the limit holds by dominated convergence whenever $\sum|\gamma(h)|<\infty$; it equals $2\pi$ times the spectral density at frequency zero).</p>
<div class="thm" data-kind="Proposition" data-name="The IID bootstrap estimates the wrong variance">
<p>For the IID bootstrap of the mean, $n\Var^*(\bar X^*_n) = \hat\gamma(0) = \frac1n\sum(X_t-\bar X)^2\to\gamma(0)$, whereas the target is $\sigma^2_\infty$. For an AR(1) $X_t = \phi X_{t-1} + \eta_t$, $\gamma(h) = \gamma(0)\phi^{|h|}$ and
$$\frac{\sigma^2_\infty}{\gamma(0)} = 1 + 2\sum_{h\ge1}\phi^h = \frac{1+\phi}{1-\phi}.$$</p>
</div>
<p>With $\phi = 0.5$ the IID bootstrap variance is too small by a factor 3; with $\phi = 0.8$, by 9; with $\phi = 0.95$, by 39. The resulting intervals are far too narrow. The defect is not in the bootstrap principle but in the plug-in: $\hat F_n$ (the marginal empirical distribution) is the wrong estimate of the data-generating process, which is the joint law of the whole series.</p>
<div class="note" data-name="Two routes">
<p>One can model the dependence — fit an AR($p$) with $p$ growing (the <em>sieve bootstrap</em>, Bühlmann, 1997), then resample its residuals — or avoid modelling it by resampling <em>blocks</em> of consecutive observations, inside which the dependence is preserved automatically. Block methods are the subject of this page; the same idea underlies batch-means standard errors for <a href="#/t/mcmc-diagnostics">MCMC output</a>.</p>
</div>`,
      },
      {
        title: 'Block schemes',
        html: R`
<p>Fix a block length $\ell$ and let $k = \lceil n/\ell\rceil$ blocks be glued together (then truncated to length $n$). The schemes differ in which blocks are available.</p>
<div class="cols">
<div><h4>Non-overlapping (Carlstein, 1986)</h4><p>Blocks $(X_{(j-1)\ell+1},\dots,X_{j\ell})$, $j = 1,\dots,\lfloor n/\ell\rfloor$. Few distinct blocks, so more variable.</p></div>
<div><h4>Moving block, MBB (Künsch, 1989; Liu & Singh, 1992)</h4><p>All $n-\ell+1$ overlapping blocks $(X_i,\dots,X_{i+\ell-1})$. Observations near the ends appear in fewer blocks, so $\E^*\bar X^*\ne\bar X$.</p></div>
<div><h4>Circular, CBB (Politis & Romano, 1992)</h4><p>Wrap the series on a circle so that $X_{n+j} := X_j$; all $n$ blocks starting at $i = 1,\dots,n$. Every observation is equally likely and $\E^*\bar X^* = \bar X$ exactly.</p></div>
<div><h4>Stationary (Politis & Romano, 1994)</h4><p>Circular blocks with <em>random</em> lengths $L\sim\text{Geometric}(p)$, mean $1/p$. The resampled series is itself stationary conditional on the data.</p></div>
</div>
<div class="algo" data-name="Circular block bootstrap for a statistic $T_n$">
<ol>
<li>Extend the series circularly: $X_{n+j} = X_j$ for $j = 1,\dots,\ell-1$.</li>
<li>For each $b$: draw block starts $I_1,\dots,I_k$ IID uniform on $\{1,\dots,n\}$, concatenate $(X_{I_j},\dots,X_{I_j+\ell-1})$, truncate to length $n$, and compute $T_n^*$.</li>
<li>Use the $T^*_{n,b}$ as usual. For a bootstrap-$t$, studentize with a block-consistent variance estimate (e.g. the Bartlett/HAC estimate below), not the IID formula.</li>
</ol>
</div>
<p>For statistics that are functions of several lags (autocorrelations, regression on lagged values), apply the blocking to the vectors $Y_t = (X_t,\dots,X_{t+m-1})$ — the <em>blocks-of-blocks</em> device — so that resampled lag pairs are genuine pairs from the series rather than artifacts of a join between blocks.</p>`,
        lab: 'block-bootstrap-ar1',
      },
      {
        title: 'What the block bootstrap estimates',
        html: R`
<p>For the sample mean the circular block bootstrap variance can be computed exactly, which makes its behaviour transparent. Let $n = k\ell$, and let $M_i = \frac1\ell\sum_{j=0}^{\ell-1}X_{i+j}$ be the circular block means, $i = 1,\dots,n$.</p>
<div class="thm" data-kind="Proposition" data-name="CBB variance is a Bartlett lag-window estimator">
<p>$$n\Var^*(\bar X^*) = \frac{\ell}{n}\sum_{i=1}^n\big(M_i - \bar X\big)^2 = \sum_{|h|<\ell}\Big(1 - \frac{|h|}{\ell}\Big)\tilde\gamma(h),$$
where $\tilde\gamma(h) = \frac1n\sum_{t=1}^n(X_t-\bar X)(X_{t+h}-\bar X)$ is the circular sample autocovariance.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\bar X^* = \frac1k\sum_{j=1}^kM_{I_j}$ with $I_j$ IID uniform on $\{1,\dots,n\}$. Each $M_{I_j}$ has bootstrap mean $\frac1n\sum_iM_i = \bar X$ (on the circle every $X_t$ appears in exactly $\ell$ blocks) and variance $\frac1n\sum_i(M_i-\bar X)^2$. Independence gives $\Var^*(\bar X^*) = \frac1{kn}\sum_i(M_i-\bar X)^2$, and multiplying by $n = k\ell$ gives the first form. For the second, expand $(M_i-\bar X)^2 = \ell^{-2}\sum_{s,t=0}^{\ell-1}(X_{i+s}-\bar X)(X_{i+t}-\bar X)$ and sum over $i$: for each lag $h = t - s$ there are $\ell-|h|$ pairs $(s,t)$, each contributing $n\tilde\gamma(h)$ after summing over $i$ on the circle. Hence $\frac{\ell}{n}\sum_i(M_i-\bar X)^2 = \frac{\ell}{n}\cdot\frac{1}{\ell^2}\sum_{|h|<\ell}(\ell-|h|)n\tilde\gamma(h)$, which simplifies to the claim.</p>
</details>
<p>So the block bootstrap of the mean is the classical Bartlett-kernel HAC (Newey–West) estimator of $\sigma^2_\infty$ with truncation lag $\ell$. With $\ell = 1$ it collapses to $\tilde\gamma(0)$, the IID bootstrap. Its properties follow from spectral estimation theory:</p>
$$\text{Bias} \approx -\frac{\Gamma}{\ell}, \quad \Gamma := \sum_h|h|\gamma(h) = 2\sum_{h\ge1}h\gamma(h); \qquad \Var \approx \frac{4}{3}\cdot\frac{\ell}{n}\,\sigma^4_\infty.$$
<p>The bias comes from down-weighting lag $h$ by $|h|/\ell$ (plus truncating lags $\ge\ell$, which is smaller order when $\gamma$ decays geometrically); the variance comes from averaging over only about $n/\ell$ effectively independent blocks (the constant $4/3 = 2\int_{-1}^1(1-|u|)^2\d u$ is the Bartlett kernel's).</p>
<div class="thm" data-kind="Corollary" data-name="The $n^{1/3}$ block length (Künsch, 1989; Hall, Horowitz & Jing, 1995)">
<p>Minimizing $\text{MSE}(\ell) \approx \Gamma^2/\ell^2 + \frac43(\ell/n)\sigma^4_\infty$ over $\ell$ gives
$$\ell_{\text{opt}} = \Big(\frac{3\Gamma^2}{2\sigma^4_\infty}\Big)^{1/3}n^{1/3},\qquad \text{MSE}(\ell_{\text{opt}}) \asymp n^{-2/3}.$$
For an AR(1), $\Gamma/\sigma^2_\infty = 2\phi/(1-\phi^2)$, so $\ell_{\text{opt}} = \big(6\phi^2/(1-\phi^2)^2\big)^{1/3}n^{1/3}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Differentiate: $-2\Gamma^2/\ell^3 + \frac43\sigma^4_\infty/n = 0$ gives $\ell^3 = \frac32\Gamma^2n/\sigma^4_\infty$. Substituting back, both terms are of order $n^{-2/3}$. For the AR(1), $\sum_{h\ge1}h\phi^h = \phi/(1-\phi)^2$, so $\Gamma = 2\gamma(0)\phi/(1-\phi)^2$, and $\sigma^2_\infty = \gamma(0)(1+\phi)/(1-\phi)$; their ratio is $2\phi/((1-\phi)(1+\phi))$, and $\frac32\cdot4\phi^2/(1-\phi^2)^2 = 6\phi^2/(1-\phi^2)^2$.</p>
</details>
<p>Hall, Horowitz & Jing (1995) showed the optimal rate depends on the target: $\ell\propto n^{1/3}$ for bias and variance estimation, $n^{1/4}$ for one-sided distribution functions, and $n^{1/5}$ for two-sided ones. The constants depend on unknown autocovariances, which motivates plug-in selectors such as Politis & White (2004) (with the correction of Patton, Politis & White, 2009), which estimate $\Gamma$ and $\sigma^2_\infty$ with a flat-top lag window.</p>`,
        lab: 'block-bootstrap-tradeoff',
      },
      {
        title: 'Consistency and higher-order behaviour',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Consistency of the moving block bootstrap (Künsch, 1989; Liu & Singh, 1992)">
<p>Let $\{X_t\}$ be strictly stationary and $\alpha$-mixing with $\sum_k\alpha(k)^{\delta/(2+\delta)}<\infty$ and $\E|X_t|^{2+\delta}<\infty$ for some $\delta>0$, and $\sigma^2_\infty>0$. If $\ell\to\infty$ and $\ell/n\to0$, then
$$\sup_x\Big|\P^*\big(\sqrt n(\bar X^*_n - \E^*\bar X^*_n)\le x\big) - \P\big(\sqrt n(\bar X_n-\mu)\le x\big)\Big| \pto 0.$$
The same holds for smooth functions of means of finite-dimensional lag vectors, and (with more work) for Hadamard-differentiable functionals.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Conditionally on the data, $\sqrt n(\bar X^*-\E^*\bar X^*) = k^{-1/2}\sum_{j=1}^k\sqrt\ell(M_{I_j}-\E^*M_{I_1})$ is a normalized sum of $k\to\infty$ IID terms. Its variance is the block estimator above, consistent for $\sigma^2_\infty$ because bias $O(1/\ell)\to0$ and variance $O(\ell/n)\to0$. A Lindeberg condition for the block sums holds in probability under the moment and mixing assumptions, giving a conditional CLT with limit $\N(0,\sigma^2_\infty)$. The real-world mean has the same limit by the CLT for mixing sequences; Pólya's theorem gives uniformity.</p>
</details>
<p>The higher-order story is less rosy than in the IID case. Joining independent blocks creates artificial "independence at the seams," which introduces errors of order $\ell/n$ and $1/\ell$ that do not occur in the true sampling distribution. A suitably studentized block bootstrap does improve on the normal approximation (Götze & Künsch, 1996; Lahiri, 2003), but the error is of order $n^{-1/2}\cdot$(something tending to zero slowly), not $n^{-1}$. Practical implications: use the bootstrap-$t$ with a HAC standard error computed <em>with the same block structure</em> in the bootstrap world, and do not expect miracles in short series.</p>
<p>Among the schemes, the MBB and CBB have the same leading bias and variance; the non-overlapping version has variance larger by a factor $3/2$; the stationary bootstrap has the same bias but a larger variance constant than the MBB (Lahiri, 1999), and is less sensitive to the exact choice of mean block length.</p>`,
      },
      {
        title: 'Pitfalls and alternatives',
        html: R`
<div class="warn" data-name="When block bootstraps fail or mislead">
<ul>
<li><strong>Nonstationarity.</strong> Trends, level shifts, or changing variance mean blocks from different periods are not exchangeable. Detrend or difference first, or use a local block bootstrap.</li>
<li><strong>Long memory.</strong> If $\gamma(h)\sim h^{2d-1}$ with $0 < d < \frac12$, $\sum\gamma(h) = \infty$ and $\Var(\bar X)$ decays like $n^{2d-1}$, not $n^{-1}$. Block bootstraps (with $\ell = o(n)$) are inconsistent for the mean; <a href="#/t/subsampling">subsampling</a> with an estimated rate is one alternative (Lahiri, 1993; Hall, Jing & Lahiri, 1998).</li>
<li><strong>Seasonality.</strong> Blocks whose length is not a multiple of the period misalign the seasonal pattern; use seasonal blocks or remove the season first.</li>
<li><strong>Short series.</strong> With $n = 60$ and strong dependence, $\ell_{\text{opt}}$ may be 10–15, leaving only four to six effectively independent blocks. No bootstrap can manufacture information that is not in the series.</li>
<li><strong>Choosing $\ell$ by eye.</strong> A plot of $\widehat{\se}(\ell)$ versus $\ell$ typically rises and then plateaus; choosing the $\ell$ that gives the largest SE is a biased procedure (it picks noise). Prefer an automatic selector, and report sensitivity.</li>
</ul>
</div>
<p><strong>Alternatives.</strong> The <em>sieve bootstrap</em> fits AR($p_n$) with $p_n\to\infty$ and resamples residuals, highly efficient for linear processes; the <em>tapered block bootstrap</em> (Paparoditis & Politis, 2001) down-weights block ends to reduce bias to $O(\ell^{-2})$, improving the MSE rate to $n^{-4/5}$; the <em>dependent wild bootstrap</em> (Shao, 2010) multiplies residuals by a smooth dependent weight process; and for spatial data, blocks become tiles. For MCMC, batch means with batch size $\propto n^{1/3}$ is literally the non-overlapping block variance estimator.</p>`,
      },
    ],
    quiz: [
      { q: R`For an AR(1) with $\phi = 0.7$, by what factor does the IID bootstrap underestimate $\Var(\bar X)$ asymptotically?`,
        a: R`The ratio $\sigma^2_\infty/\gamma(0) = (1+\phi)/(1-\phi) = 1.7/0.3\approx5.67$. The IID bootstrap SE is too small by $\sqrt{5.67}\approx2.4$.` },
      { q: R`Show that the circular block bootstrap with $\ell = 1$ reduces to the IID bootstrap, and explain what goes wrong if $\ell = n$.`,
        a: R`With $\ell=1$ the block means are the observations, so $n\Var^* = \frac1n\sum(X_t - \bar X)^2 = \tilde\gamma(0)$, the IID bootstrap. With $\ell = n$ there is one block per resample, and every circular block of length $n$ contains the whole series, so $M_i = \bar X$ for all $i$ and $\Var^* = 0$: no variability is left. The useful regime is $1\ll\ell\ll n$.` },
      { q: R`Where does the $n^{1/3}$ rate for the block length come from?`,
        a: R`The block variance estimator has bias $\approx-\Gamma/\ell$ (lags are downweighted by $|h|/\ell$) and variance $\approx\frac43(\ell/n)\sigma^4_\infty$ (only about $n/\ell$ independent blocks). Squared bias $\propto\ell^{-2}$ plus variance $\propto\ell/n$ is minimized at $\ell^3\propto n$.` },
      { q: R`Why should a statistic like the lag-1 autocorrelation be bootstrapped with blocks of the pairs $(X_t, X_{t+1})$ rather than blocks of $X_t$?`,
        a: R`When blocks of $X_t$ are concatenated, the pair formed by the last point of one block and the first point of the next is not a real lag-1 pair — they come from unrelated parts of the series. Blocking the pair vectors ensures every resampled pair occurred in the data, removing these join artifacts (the "blocks of blocks" idea).` },
    ],
  });
})();

/* ---------------------------------------------------------------- subsampling */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'subsampling',
    domain: 'resampling',
    order: 70,
    title: 'Subsampling & the m-out-of-n Bootstrap',
    subtitle: 'Valid whenever a limit law exists: smaller samples, without replacement, with the right rate',
    level: 3,
    sources: { other: 'Politis & Romano (1994); Politis, Romano & Wolf (1999); Bickel, Götze & van Zwet (1997); Bickel & Sakov (2008); Hoeffding (1963)' },
    prereqs: ['bootstrap', 'convergence'],
    related: ['block-bootstrap', 'order-statistics', 'parametric-bootstrap', 'jackknife', 'block-maxima'],
    tags: ['bootstrap failure', 'U-statistics', 'rate of convergence', 'non-regular', 'extremes'],
    summary: R`The bootstrap can fail — for the sample maximum, at a boundary, with infinite variance, for superefficient or cube-root estimators — because $\hat F_n$ is too close to the data for a non-smooth statistic to behave in the bootstrap world as it does in the real one. Subsampling fixes this with a disarmingly simple observation: a subsample of size $m$ drawn <em>without</em> replacement is an honest IID sample of size $m$ from $F$ itself. If $\tau_n(\hat\theta_n - \theta)$ has <em>any</em> limit law and $m\to\infty$ with $m/n\to0$, the subsample distribution of $\tau_m(\hat\theta_m - \hat\theta_n)$ converges to it. No smoothness, no normality, no second moments. The $m$-out-of-$n$ bootstrap (with replacement) behaves the same way when $m$ is small enough that ties are rare.`,
    survey: {
      task: 'Confidence intervals and tests when the ordinary bootstrap fails or its validity is unknown',
      assumptions: R`$\tau_n(\hat\theta_n-\theta)\dto J$ (any non-degenerate law); $m\to\infty$, $m/n\to0$, $\tau_m/\tau_n\to0$; IID or stationary mixing`,
      output: R`Approximation $L_{n,m}$ to the law of $\tau_n(\hat\theta_n-\theta)$; intervals $\hat\theta_n - \tau_n^{-1}\big(L_{n,m}^{-1}(1-\alpha/2), L_{n,m}^{-1}(\alpha/2)\big)$`,
      strengths: 'Valid under minimal conditions; extends directly to time series and random fields; no Edgeworth theory needed',
      weaknesses: R`Slower convergence than the bootstrap when both work; requires knowing (or estimating) the rate $\tau_n$; sensitive to the choice of $m$`,
      complexity: R`$B$ statistic evaluations on samples of size $m$ (cheaper than the bootstrap)`,
    },
    sections: [
      {
        title: 'Why the bootstrap fails, and the subsampling idea',
        html: R`
<p>Recall the canonical failure (see <a href="#/t/bootstrap">the bootstrap</a>): for $X_i\iid\text{Unif}(0,\theta)$, $n(\theta - X_{(n)})/\theta\dto\text{Exp}(1)$, but a bootstrap sample contains the observed maximum with probability $1-(1-1/n)^n\to1-e^{-1}\approx0.632$, so the bootstrap law of $n(X_{(n)} - X^*_{(n)})$ has an atom of mass $0.632$ at zero that the true law does not. The bootstrap world reproduces the real world only when the statistic is insensitive to the $O(n^{-1/2})$, discrete perturbations that resampling applies — roughly, when it is Hadamard differentiable. The maximum is not.</p>
<p>Subsampling sidesteps the issue. Let $\hat\theta_m(S)$ denote the statistic computed on the subset $S\subset\{1,\dots,n\}$, $|S| = m$.</p>
<div class="def" data-name="Subsampling distribution">
<p>With $N = \binom nm$ and the sum over all size-$m$ subsets,
$$L_{n,m}(x) = \frac1N\sum_{|S|=m}\ind\Big\{\tau_m\big(\hat\theta_m(S) - \hat\theta_n\big)\le x\Big\}.$$
In practice the sum is replaced by an average over $B$ randomly chosen subsets (an incomplete $U$-statistic, with $O(B^{-1/2})$ Monte Carlo error).</p>
</div>
<div class="note">
<p>Each subset $S$ <em>is</em> an IID sample of size $m$ from $F$, so $\tau_m(\hat\theta_m(S)-\theta)$ has <em>exactly</em> the sampling distribution $J_m$ of the statistic at sample size $m$ — no plug-in approximation of $F$ at all. The only approximations are (i) $J_m\approx J_n\approx J$ (needs $m\to\infty$), (ii) centering at $\hat\theta_n$ instead of the unknown $\theta$ (needs $\tau_m/\tau_n\to0$, i.e. $\hat\theta_n$ much more accurate than $\hat\theta_m$), and (iii) the subsets overlap, so the average over $S$ is not an average of independent terms (needs $m/n\to0$ for a law of large numbers).</p>
</div>`,
      },
      {
        title: 'The Politis–Romano theorem',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Consistency of subsampling (Politis & Romano, 1994)">
<p>Let $X_1,\dots,X_n\iid F$ and suppose there is a normalizing sequence $\tau_n\to\infty$ and a distribution $J$ such that $J_n(x) = \P(\tau_n(\hat\theta_n-\theta)\le x)\to J(x)$ at continuity points of $J$. If $m = m_n\to\infty$, $m/n\to0$ and $\tau_m/\tau_n\to0$, then for every continuity point $x$ of $J$, $L_{n,m}(x)\pto J(x)$. If $J$ is continuous, $\sup_x|L_{n,m}(x) - J(x)|\pto0$, and the interval
$$\Big[\hat\theta_n - \tau_n^{-1}c_{1-\alpha/2},\ \hat\theta_n - \tau_n^{-1}c_{\alpha/2}\Big],\qquad c_\beta = L_{n,m}^{-1}(\beta),$$
has asymptotic coverage $1-\alpha$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Step 1 (replace $\hat\theta_n$ by $\theta$).</em> Define $U_n(x) = \frac1N\sum_S\ind\{\tau_m(\hat\theta_m(S)-\theta)\le x\}$. This is a $U$-statistic of order $m$ with a kernel bounded in $[0,1]$, and each term has expectation $J_m(x)$, so $\E U_n(x) = J_m(x)\to J(x)$.</p>
<p><em>Step 2 (variance of a $U$-statistic of growing order).</em> Hoeffding (1963) showed that a $U$-statistic of order $m$ is an average, over all permutations of the data, of $W = \frac1k\sum_{j=1}^k h(\text{block } j)$, where $k = \lfloor n/m\rfloor$ and the $k$ blocks are disjoint. Each $W$ is a mean of $k$ IID bounded terms with variance $\le\frac1{4k}$, and averaging cannot increase variance (Jensen), so $\Var U_n(x)\le\frac{1}{4\lfloor n/m\rfloor}\to0$. With Step 1, $U_n(x)\pto J(x)$.</p>
<p><em>Step 3 (the centering error is negligible).</em> $\tau_m(\hat\theta_m(S) - \hat\theta_n) = \tau_m(\hat\theta_m(S)-\theta) - \tau_m(\hat\theta_n-\theta)$, and $\tau_m(\hat\theta_n-\theta) = \frac{\tau_m}{\tau_n}\cdot\tau_n(\hat\theta_n - \theta) = o(1)\cdot O_p(1) = o_p(1)$. On the event $E_n = \{|\tau_m(\hat\theta_n - \theta)|\le\varepsilon\}$, whose probability tends to 1, $U_n(x-\varepsilon)\le L_{n,m}(x)\le U_n(x+\varepsilon)$. Choose $\varepsilon$ so that $x\pm\varepsilon$ are continuity points with $J(x+\varepsilon)-J(x-\varepsilon)$ arbitrarily small; then $L_{n,m}(x)\pto J(x)$.</p>
<p><em>Step 4.</em> Uniform convergence for continuous $J$ follows from pointwise convergence at finitely many points and monotonicity (the Pólya argument). Coverage: $\P\big(\theta\in\text{interval}\big) = \P\big(c_{\alpha/2}\le\tau_n(\hat\theta_n-\theta)\le c_{1-\alpha/2}\big)\to J(J^{-1}(1-\alpha/2)) - J(J^{-1}(\alpha/2)) = 1-\alpha$, using $c_\beta\pto J^{-1}(\beta)$ at points where $J$ is continuous and strictly increasing.</p>
</details>
<p>The hypotheses are remarkably weak: existence of a limit law, and knowledge of the rate. Nothing requires the limit to be normal, the statistic to be smooth, or the bootstrap to work. The proof also adapts directly to stationary $\alpha$-mixing time series if one uses the $n-m+1$ <em>contiguous</em> blocks $(X_i,\dots,X_{i+m-1})$ as subsamples — the subsampling cousin of the <a href="#/t/block-bootstrap">block bootstrap</a> (Politis, Romano & Wolf, 1999).</p>`,
      },
      {
        title: 'The m-out-of-n bootstrap',
        html: R`
<p>A closely related idea draws $m$ observations <em>with</em> replacement from $\hat F_n$ and uses the law of $\tau_m(\hat\theta^*_m - \hat\theta_n)$ (Bretagnolle, 1983; Swanepoel, 1986; Bickel, Götze & van Zwet, 1997).</p>
<div class="thm" data-kind="Proposition" data-name="m-out-of-n bootstrap is subsampling when ties are rare">
<p>Let $Q_{\text{with}}$ and $Q_{\text{without}}$ be the laws of the index multiset drawn by $m$ draws with and without replacement from $\{1,\dots,n\}$. Then
$$\big\|Q_{\text{with}} - Q_{\text{without}}\big\|_{TV}\le\P(\text{some index repeats})\le\frac{m(m-1)}{2n}.$$
Hence if $m^2/n\to0$, the $m$-out-of-$n$ bootstrap distribution and the subsampling distribution differ by $o(1)$ uniformly, for every statistic, and the $m$-out-of-$n$ bootstrap inherits the conclusion of the Politis–Romano theorem.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Couple the two schemes: draw indices $I_1,\dots,I_m$ with replacement; conditional on the event $D$ that they are all distinct, the set $\{I_1,\dots,I_m\}$ is uniformly distributed over size-$m$ subsets (by symmetry), which is exactly $Q_{\text{without}}$. So the two laws agree except on $D^c$, giving $\|Q_{\text{with}} - Q_{\text{without}}\|_{TV}\le\P(D^c)$. By the union bound over the $\binom m2$ pairs, each tied with probability $1/n$, $\P(D^c)\le m(m-1)/(2n)$. Because any statistic is a function of the drawn sample, the bootstrap and subsampling distributions of the statistic are within $\P(D^c)$ in total variation as well.</p>
</details>
<p>Under the weaker condition $m = o(n)$ the $m$-out-of-$n$ bootstrap is still consistent in many standard failure cases (including the maximum, where it suffices that $m/n\to0$: the probability that the resample contains $X_{(n)}$ is $1-(1-1/n)^m\approx m/n\to0$), but not universally; Bickel, Götze & van Zwet (1997) give the general picture. In smooth problems where the full bootstrap works, the $m$-out-of-$n$ version also works but throws away accuracy.</p>
<div class="ex" data-name="The uniform maximum, fixed">
<p>$\tau_n = n$, $\hat\theta_n = X_{(n)}$, $J = $ law of $-\theta E$ with $E\sim\text{Exp}(1)$ (for $n(\hat\theta_n - \theta)$). Subsampling uses $m(X_{(m)}(S) - X_{(n)})$; a subset of size $m$ contains the overall maximum with probability exactly $m/n\to0$, so the atom that ruined the bootstrap disappears. The condition $\tau_m/\tau_n = m/n\to0$ holds. Rescaling by $\hat\theta_n$ (a consistent estimate of the scale $\theta$) turns the interval into one for $\theta$ with asymptotically correct coverage. Try it in the lab below.</p>
</div>`,
        lab: 'subsampling-uniform-max',
      },
      {
        title: 'Choosing m, and the price of generality',
        html: R`
<p>The theorem permits any $m$ with $m\to\infty$, $m/n\to0$; finite-sample performance depends heavily on the choice. Two errors compete:</p>
<ul>
<li><strong>Too small $m$:</strong> $J_m$ is far from $J_n$. The subsample statistic has not "reached" its asymptotic regime, and the error $J_m - J$ is of the order of the rate at which $J_n\to J$ (e.g. $m^{-1/2}$ for smooth statistics, $1/m$ for the uniform max).</li>
<li><strong>Too large $m$:</strong> subsamples overlap heavily, $L_{n,m}$ is a poor average (variance of order $m/n$), and the centering error $\tau_m(\hat\theta_n - \theta)$ is no longer negligible.</li>
</ul>
<p>Data-driven rules exploit the fact that, for a good $m$, the subsampling distribution should be <em>stable</em> in $m$.</p>
<div class="algo" data-name="Bickel–Sakov adaptive rule (2008)">
<ol>
<li>Fix $q\in(0,1)$ and the grid $m_j = \lceil q^jn\rceil$, $j = 0,1,2,\dots$</li>
<li>Compute the $m$-out-of-$n$ bootstrap distributions $L_{m_j}$.</li>
<li>Choose $\hat m = \argmin_j \rho(L_{m_j}, L_{m_{j+1}})$ for a distance $\rho$ such as Kolmogorov's.</li>
</ol>
</div>
<p>The <em>minimum volatility</em> method of Politis, Romano & Wolf (1999) is similar: compute the interval endpoints over a range of $m$ and pick the $m$ where they vary least. The second lab shows coverage as a function of $m$ directly.</p>
<div class="note" data-name="Unknown rates">
<p>The rate $\tau_n$ must be known up to a constant. When it is not (e.g. heavy tails with unknown index, where $\tau_n = n^{1-1/\alpha}$), it can be estimated from the subsamples themselves: the spread of $L_{n,m}$ scales like $1/\tau_m$, so regressing the log interquartile range of the subsampling distribution on $\log m$ over several $m$ estimates the exponent (Bertail, Politis & Romano, 1999).</p>
</div>
<p>The price of generality is accuracy. When the bootstrap is valid and the statistic is smooth, the bootstrap-$t$ has coverage error $O(n^{-1})$; subsampling's error is governed by the much slower convergence of $J_m$ to $J$ and of the $U$-statistic average, and it is typically no better than the normal approximation. Use subsampling when it is needed, not by default.</p>`,
        lab: 'subsampling-choice-of-m',
      },
      {
        title: 'Where subsampling is the method of choice',
        html: R`
<table>
<thead><tr><th>Problem</th><th>Why the bootstrap fails</th><th>Rate $\tau_n$</th></tr></thead>
<tbody>
<tr><td>Endpoint of support (max/min of uniform, <a href="#/t/order-statistics">extreme order statistics</a>)</td><td>Atom at the observed extreme</td><td>$n$</td></tr>
<tr><td>Parameter on the boundary: $\theta\ge0$, $\hat\theta = \max(\bar X,0)$ at $\theta = 0$</td><td>Limit $\max(Z,0)$ depends on local parameter</td><td>$\sqrt n$</td></tr>
<tr><td>Mean with infinite variance (stable domain, $1<\alpha<2$)</td><td>Bootstrap law converges to a random limit (Athreya, 1987)</td><td>$n^{1-1/\alpha}$ (estimate)</td></tr>
<tr><td>Superefficient (Hodges) and post-selection estimators</td><td>Non-regular: limit not locally uniform</td><td>$\sqrt n$</td></tr>
<tr><td>Grenander (monotone density) and other cube-root estimators</td><td>Bootstrap inconsistent at a point (Kosorok, 2008; Sen, Banerjee & Woodroofe, 2010)</td><td>$n^{1/3}$</td></tr>
<tr><td>Unit root: AR(1) with $\phi = 1$</td><td>Non-standard Dickey–Fuller limit; residual bootstrap fails at $\phi=1$</td><td>$n$</td></tr>
</tbody>
</table>
<div class="warn" data-name="Pitfalls">
<ul>
<li><strong>Wrong rate.</strong> If the assumed rate $\tau_n$ is slower than the true rate $r_n$, the interval is mis-scaled by the factor $\tau_mr_n/(r_m\tau_n)$, which diverges: with $\tau_n = \sqrt n$ for the uniform max (true rate $n$) the factor is $\sqrt{n/m}$, the whole interval lies above $\theta$, and coverage tends to zero. Always check the rate.</li>
<li><strong>Uniformity.</strong> The theorem is pointwise in $\theta$. Near a boundary (e.g. $\theta$ small but positive with moderate $n$), subsampling intervals can under-cover uniformly; Andrews & Guggenberger (2009) analyze this and propose size-corrected versions.</li>
<li><strong>Discrete statistics.</strong> If $J$ has atoms, convergence holds only at continuity points, and quantiles may be unstable.</li>
<li><strong>Using $m$ close to $n$.</strong> With $m = n/2$ subsamples share half their points; the theory needs $m/n\to0$, and in practice $m$ between $n^{1/2}$ and $n^{2/3}$ is a common starting range.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why does subsampling not need the statistic to be smooth, while the bootstrap does?`,
        a: R`A subsample of size $m$ drawn without replacement is a genuine IID sample from $F$, so the subsample statistic has exactly the sampling distribution $J_m$; nothing has to be approximated by plugging $\hat F_n$ into a functional. The only requirements are that $J_m\to J$ (a limit law exists), that centering at $\hat\theta_n$ is negligible ($\tau_m/\tau_n\to0$), and that averaging over overlapping subsets satisfies a law of large numbers ($m/n\to0$). The bootstrap, by contrast, needs $T(\hat F^*_n) - T(\hat F_n)$ to mimic $T(\hat F_n) - T(F)$, which requires smoothness of $T$.` },
      { q: R`Bound the total variation distance between drawing $m = 30$ indices from $n = 10{,}000$ with and without replacement.`,
        a: R`$\le m(m-1)/(2n) = 30\cdot29/20{,}000 = 0.0435$. So for $m = 30$ the $m$-out-of-$n$ bootstrap is within about 4% (in TV) of subsampling, whatever the statistic.` },
      { q: R`In the uniform-max problem, what is the probability that a random subsample of size $m$ contains $X_{(n)}$, and that an $m$-out-of-$n$ bootstrap resample does?`,
        a: R`Subsample: exactly $m/n$. $m$-out-of-$n$ bootstrap: $1-(1-1/n)^m\approx m/n$. Both tend to 0 when $m/n\to0$, eliminating the atom at zero that makes the $n$-out-of-$n$ bootstrap fail (there the probability tends to $1-e^{-1}$).` },
      { q: R`What goes wrong if you subsample the uniform maximum using $\tau_n = \sqrt n$?`,
        a: R`The true fluctuations of $X_{(m)}(S)$ are of order $1/m$, so the quantiles $c_\beta$ of $\sqrt m(X_{(m)}(S) - X_{(n)})$ are of order $m^{-1/2}$, and the interval endpoints $X_{(n)} - c_\beta/\sqrt n$ sit at distance of order $(mn)^{-1/2}$ from $X_{(n)}$. But $\theta - X_{(n)}$ is of order $1/n$, which is smaller by the factor $\sqrt{m/n}\to0$. The interval is mis-scaled by $\sqrt{n/m}$: here it lies entirely above $\theta$ with probability tending to one, and coverage tends to zero. The rate must be right (here $\tau_n = n$), or estimated.` },
    ],
  });
})();

/* ---------------------------------------------------------------- permutation-tests */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'permutation-tests',
    domain: 'resampling',
    order: 80,
    title: 'Permutation Tests',
    subtitle: 'Exact tests from symmetry: exchangeability, randomization, and what the null really says',
    level: 2,
    sources: { aos: '§10.5', other: 'Fisher (1935); Pitman (1937); Hoeffding (1952); Lehmann & Romano (2005, Ch. 15); Phipson & Smyth (2010); Chung & Romano (2013)' },
    prereqs: ['hypothesis-testing'],
    related: ['rank-tests', 'multiple-testing', 'causal-inference', 'bootstrap', 'parametric-bootstrap'],
    tags: ['exact test', 'exchangeability', 'randomization inference', 'Monte Carlo p-value', 'studentization'],
    summary: R`If the null hypothesis implies that the data's distribution is unchanged by a group of transformations — relabelling which observations are "treatment" and which "control", flipping signs of paired differences, shuffling one variable against another — then the observed statistic is just one draw from the set of its transformed values, and its rank in that set is an exact $p$-value. No distributional assumptions, no asymptotics, valid at every $n$. Two subtleties matter in practice: Monte Carlo approximations must use $(1+\#)/(1+B)$ to stay exact, and the null that is tested exactly is equality of <em>distributions</em>; a test of equal means under unequal variances needs a studentized statistic to remain even asymptotically valid.`,
    survey: {
      task: 'Exact tests of no effect / no association / symmetry; randomization inference for experiments',
      assumptions: R`Randomization hypothesis: under $H_0$ the law of the data is invariant under a group $\mathcal G$ (e.g. exchangeability), or treatment was randomly assigned`,
      output: R`Exact $p$-value $\#\{g: T(gX)\ge T(X)\}/|\mathcal G|$ or Monte Carlo $(1+\#)/(1+B)$`,
      strengths: 'Exact finite-sample level for any statistic; design-based validity in randomized experiments; flexible choice of statistic',
      weaknesses: R`Tests only the null that justifies the symmetry; invalid for equal means with unequal variances unless studentized; limited for complex models`,
      complexity: R`$|\mathcal G|$ (e.g. $\binom{m+n}{m}$) evaluations exactly, or $B$ by Monte Carlo`,
    },
    sections: [
      {
        title: 'The idea: labels are arbitrary under the null',
        html: R`
<p>Observe $X_1,\dots,X_m\iid F$ and $Y_1,\dots,Y_n\iid G$ and test $H_0: F = G$. Pool the data into $Z = (Z_1,\dots,Z_N)$, $N = m+n$, the first $m$ coordinates being the $X$'s. Under $H_0$ the $Z_i$ are IID, so $Z$ is <strong>exchangeable</strong>: $(Z_{\pi(1)},\dots,Z_{\pi(N)})\overset{d}{=}(Z_1,\dots,Z_N)$ for every permutation $\pi$. The labelling "these $m$ are $X$'s" carries no information.</p>
<p>So take any statistic, say $T(Z) = \bar X - \bar Y$, and compare its observed value with the values $T(\pi Z)$ over all reassignments of labels. If the observed value is extreme among them, either something improbable happened or the labels matter — that is, $F\ne G$.</p>
<div class="def" data-name="Permutation p-value">
<p>Let $\mathcal G$ be a finite group acting on the sample space with $|\mathcal G| = M$. The permutation $p$-value for a statistic $T$ (large values against $H_0$) is
$$p(Z) = \frac1M\#\{g\in\mathcal G: T(gZ)\ge T(Z)\}.$$
For the two-sample problem, $\mathcal G$ is the symmetric group $S_N$; since $T$ depends only on which $m$ values are labelled $X$, one may equivalently enumerate the $\binom Nm$ subsets.</p>
</div>
<div class="ex" data-name="A tiny example by hand">
<p>Treatment: 7, 9, 12. Control: 3, 5, 6. $T = \bar X - \bar Y = 28/3 - 14/3 = 14/3$. There are $\binom63 = 20$ ways to choose which three values are "treatment". The observed split puts the three largest values in the treatment group, the most extreme possible, so exactly one of the 20 assignments has $T\ge14/3$: $p = 1/20 = 0.05$. With three per group no test of this kind can ever give $p < 0.05$ — the granularity $1/\binom Nm$ is a real limit in small samples.</p>
</div>`,
      },
      {
        title: 'Exactness: the randomization hypothesis',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Exactness of permutation tests (Fisher, 1935; Hoeffding, 1952; Lehmann & Romano, 2005, §15.2)">
<p>Suppose that under $H_0$, $gZ\overset{d}{=}Z$ for every $g\in\mathcal G$ (the <strong>randomization hypothesis</strong>). Then for every $u\in[0,1]$,
$$\P_{H_0}\big(p(Z)\le u\big)\le u.$$
If moreover the values $T(gZ)$, $g\in\mathcal G$, are almost surely distinct (no ties), then $\P_{H_0}(p(Z)\le k/M) = k/M$ for $k = 1,\dots,M$ — exact equality.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Reindexing.</em> For any $h\in\mathcal G$, $p(hZ) = \frac1M\#\{g: T(ghZ)\ge T(hZ)\} = \frac1M\#\{g': T(g'Z)\ge T(hZ)\}$, because $g\mapsto gh$ is a bijection of the group.</p>
<p><em>Averaging over the group.</em> Since $hZ\overset{d}{=}Z$, $\P(p(Z)\le u) = \P(p(hZ)\le u)$ for each $h$; averaging over $h$,
$$\P(p(Z)\le u) = \E\Big[\frac1M\#\{h: p(hZ)\le u\}\Big].$$</p>
<p><em>A deterministic counting fact.</em> Fix $z$, write $t_h = T(hz)$, and let $H_u = \{h: \#\{g: t_g\ge t_h\}\le uM\}$. If $H_u$ is nonempty, let $h_0\in H_u$ have the smallest $t$-value in $H_u$. Every $h\in H_u$ has $t_h\ge t_{h_0}$, so $H_u\subseteq\{g: t_g\ge t_{h_0}\}$, a set of size at most $uM$ (because $h_0\in H_u$). Hence $\#H_u\le uM$, and the expectation above is at most $u$.</p>
<p><em>Equality without ties.</em> If the $t_h$ are distinct, the "upper rank" $\#\{g: t_g\ge t_h\}$ takes each value $1,\dots,M$ exactly once as $h$ ranges over $\mathcal G$, so $\#\{h: p(hz)\le k/M\} = k$ exactly.</p>
</details>
<p>Notice what the proof did <em>not</em> use: any distributional form, any sample size condition, any property of $T$. The only input is invariance of the null distribution under the group. Common groups:</p>
<table>
<thead><tr><th>Problem</th><th>$H_0$</th><th>Group</th></tr></thead>
<tbody>
<tr><td>Two samples</td><td>$F = G$</td><td>Permutations of the pooled sample</td></tr>
<tr><td>Paired differences $D_i$</td><td>$D_i$ symmetric about 0 (independent)</td><td>Sign flips $D_i\mapsto\pm D_i$ ($2^n$ elements)</td></tr>
<tr><td>Independence of $X$ and $Y$</td><td>$X\indep Y$</td><td>Permute the $Y$'s against fixed $X$'s</td></tr>
<tr><td>$k$ samples / one-way layout</td><td>$F_1 = \cdots = F_k$</td><td>Permutations of all $N$ observations</td></tr>
<tr><td>Randomized experiment</td><td>Sharp null of no effect</td><td>The assignment mechanism's distribution</td></tr>
</tbody>
</table>
<div class="note" data-name="Randomization inference">
<p>In a randomized experiment there is a second, assumption-free justification. Under Fisher's <em>sharp</em> null $Y_i(1) = Y_i(0)$ for all units, the observed outcomes are fixed numbers that do not depend on the assignment; the only randomness is the assignment itself, chosen by the experimenter uniformly among $\binom Nm$ possibilities. Re-randomizing the assignment on paper and recomputing $T$ yields the exact null distribution. No IID sampling from any population is assumed: validity comes from the <em>design</em>. This is the foundation of design-based <a href="#/t/causal-inference">causal inference</a>.</p>
</div>`,
      },
      {
        title: R`Monte Carlo p-values: why $(1+\#)/(1+B)$`,
        html: R`
<p>$\binom{40}{20}\approx1.4\times10^{11}$, so complete enumeration is usually impossible. Draw $g_1,\dots,g_B$ IID uniformly from $\mathcal G$ instead and compute $T_b = T(g_bZ)$, $T_0 = T(Z)$.</p>
<div class="thm" data-kind="Theorem" data-name="Exact Monte Carlo permutation p-value">
<p>Under the randomization hypothesis,
$$\hat p = \frac{1 + \#\{b\ge1: T_b\ge T_0\}}{1+B}$$
satisfies $\P_{H_0}(\hat p\le u)\le u$ for all $u\in[0,1]$, for every finite $B$ — with equality at $u = k/(B+1)$ when ties have probability zero. The naive estimate $\#\{b: T_b\ge T_0\}/B$ does not have this property.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Exchangeability of $(T_0,\dots,T_B)$.</em> Let $h$ be uniform on $\mathcal G$, independent of $Z$ and of the $g_b$. Since $hZ\overset{d}{=}Z$ (for each fixed $h$, hence for random independent $h$), and the $g_b$ are independent of both,
$$(Z, g_1Z,\dots,g_BZ)\overset{d}{=}(hZ, g_1hZ,\dots,g_BhZ) = (U_0Z, U_1Z,\dots,U_BZ),$$
where $U_0 = h$ and $U_b = g_bh$. Conditional on $h$, the $g_bh$ are IID uniform (right multiplication is a bijection), so $U_0,\dots,U_B$ are IID uniform on $\mathcal G$ and independent of $Z$. A vector of the form $(U_0Z,\dots,U_BZ)$ with IID $U$'s independent of $Z$ is exchangeable, so $(T_0,\dots,T_B)$ is exchangeable.</p>
<p><em>Rank argument.</em> $\hat p\le k/(B+1)$ iff $T_0$ is among the $k$ largest of the $B+1$ values (counting ties against $T_0$). Applying the counting fact from the previous proof to the $B+1$ exchangeable values (with the symmetric group on positions in place of $\mathcal G$), the probability is at most $k/(B+1)$, with equality when ties are impossible because then $T_0$'s rank is uniform on $\{1,\dots,B+1\}$.</p>
<p><em>The naive version fails:</em> $\P(\#\{b: T_b\ge T_0\}/B = 0) = \P(T_0\text{ is the strict maximum}) = 1/(B+1) > 0$, so a test rejecting when the naive $p$-value is $0$ (which is $\le u$ for every $u > 0$) has size $1/(B+1)$ at every level $u < 1/(B+1)$.</p>
</details>
<p>Equivalently: the observed data set is itself one of the "permutations" (the identity), so it belongs in both numerator and denominator. Phipson & Smyth (2010) document published analyses — especially in genomics, with many tests and multiple-testing corrections that hinge on tiny $p$-values — that reported $p = 0$ from finite permutation samples.</p>
<div class="ex" data-name="How large should B be?">
<p>The Monte Carlo standard error of $\hat p$ is about $\sqrt{p(1-p)/B}$. Near $p = 0.05$ with $B = 999$, that is $0.007$; to distinguish 0.04 from 0.05 reliably you need $B$ in the tens of thousands. Choosing $B$ so that $\alpha(B+1)$ is an integer (e.g. $B = 999$ or $9999$ for $\alpha = 0.05$) makes the Monte Carlo test exactly of level $\alpha$. With multiple-testing corrections at level $\alpha/K$, $B$ must exceed $K/\alpha$ by a comfortable factor for any rejection to be possible.</p>
</div>`,
        lab: 'permutation-tests-null',
      },
      {
        title: 'What is being tested? Studentized permutation tests',
        html: R`
<p>Exactness holds for $H_0: F = G$. Practitioners often want $H_0:\mu_X = \mu_Y$, allowing $F$ and $G$ to differ otherwise. Then the randomization hypothesis fails, and with it exactness. Worse, the test need not even be asymptotically valid.</p>
<div class="thm" data-kind="Proposition" data-name="The unstudentized permutation test under unequal variances (Romano, 1990)">
<p>Let $\mu_X = \mu_Y$, $\sigma_X^2\ne\sigma_Y^2$, $m/N\to\lambda\in(0,1)$. The permutation distribution of $\sqrt N(\bar X - \bar Y)$ converges in probability to $\N\big(0, \bar\sigma^2/(\lambda(1-\lambda))\big)$ with $\bar\sigma^2 = \lambda\sigma_X^2 + (1-\lambda)\sigma_Y^2$ (the variance of the pooled population), whereas the true sampling distribution is $\N\big(0, \sigma_X^2/\lambda + \sigma_Y^2/(1-\lambda)\big)$. The two agree iff $\lambda = \frac12$ or $\sigma_X^2 = \sigma_Y^2$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Under a random relabelling, the pseudo-groups are random samples (without replacement) of sizes $m$ and $n$ from the pooled data, whose empirical variance converges to $\bar\sigma^2$ when the means are equal. By the finite-population CLT (Hájek), $\bar X_\pi - \bar Y_\pi$ is approximately normal with variance $\bar\sigma^2(1/m + 1/n) = \bar\sigma^2 N/(mn)$, i.e. $\sqrt N(\bar X_\pi - \bar Y_\pi)\approx\N(0,\bar\sigma^2/(\lambda(1-\lambda)))$. The true variance of $\sqrt N(\bar X-\bar Y)$ is $N(\sigma_X^2/m + \sigma_Y^2/n)\to\sigma_X^2/\lambda + \sigma_Y^2/(1-\lambda)$. The difference is
$$\frac{\lambda\sigma_X^2 + (1-\lambda)\sigma_Y^2}{\lambda(1-\lambda)} - \frac{\sigma_X^2}{\lambda} - \frac{\sigma_Y^2}{1-\lambda} = \sigma_X^2\Big(\frac1{1-\lambda}-\frac1\lambda\Big) + \sigma_Y^2\Big(\frac1\lambda - \frac1{1-\lambda}\Big) = \frac{(\sigma_X^2-\sigma_Y^2)(2\lambda - 1)}{\lambda(1-\lambda)},$$
which vanishes iff $\sigma_X^2 = \sigma_Y^2$ or $\lambda = \frac12$. Its sign gives the direction of the error.</p>
</details>
<p>The consequence: if the smaller group has the larger variance, the permutation distribution is too narrow and the test over-rejects; if the smaller group has the smaller variance, it under-rejects (and loses power). The fix is to permute a statistic that is asymptotically pivotal:</p>
<div class="thm" data-kind="Theorem" data-name="Studentized permutation tests (Janssen, 1997; Chung & Romano, 2013)">
<p>Let $T = (\bar X - \bar Y)/\sqrt{S_X^2/m + S_Y^2/n}$ (Welch's statistic). The permutation test based on $T$ is exact under $H_0: F = G$, and under $H_0:\mu_X = \mu_Y$ with finite but unequal variances it has asymptotic level $\alpha$: its permutation distribution converges in probability to $\N(0,1)$, which is also the true limiting null distribution of $T$.</p>
</div>
<p>This gives the best of both worlds: exact when distributions are equal, asymptotically correct otherwise. The same principle applies to other parameters (medians, correlations): permute a <em>studentized</em> estimate. A permutation test of the raw correlation, for instance, is exact for independence but not valid for $H_0:\rho = 0$ when $X$ and $Y$ are uncorrelated but dependent.</p>`,
        lab: 'permutation-tests-behrens-fisher',
      },
      {
        title: 'Choosing the statistic; power; extensions',
        html: R`
<p>Any statistic yields a valid test; the choice determines power. Some facts:</p>
<ul>
<li><strong>Difference of means</strong> vs. the $t$-test: Hoeffding (1952) showed the permutation test of $\bar X - \bar Y$ is asymptotically equivalent to the two-sample $t$-test under normality, so nothing is lost in large samples, and exactness is gained in small ones.</li>
<li><strong>Ranks</strong>: the permutation test of the rank-sum is the Wilcoxon–Mann–Whitney test (see <a href="#/t/rank-tests">rank tests</a>); its null distribution can be tabulated once because ranks are always $1,\dots,N$. Ranks give robustness to outliers and heavy tails.</li>
<li><strong>Omnibus statistics</strong> (Kolmogorov–Smirnov, energy distance, kernel MMD) detect any difference between $F$ and $G$, at the cost of power against location shifts.</li>
<li><strong>Max-$T$ for multiple testing</strong>: permuting the whole data matrix and recording $\max_j|T_j|$ across many outcomes gives the Westfall–Young step-down procedure, which controls the familywise error rate while exploiting the dependence among tests (see <a href="#/t/multiple-testing">multiple testing</a>).</li>
<li><strong>Confidence intervals by inversion</strong>: for a shift model $G(x) = F(x-\Delta)$, test $H_0:\Delta = \Delta_0$ by permuting $X_i$ and $Y_j - \Delta_0$; the set of non-rejected $\Delta_0$ is an exact confidence set.</li>
</ul>
<div class="warn" data-name="Pitfalls">
<ul>
<li><strong>Exchangeability can fail under $H_0$.</strong> Time series, clustered or repeated-measures data are not exchangeable even when there is no treatment effect. Permute at the level of the randomization unit (whole clusters, whole subjects), or use restricted permutations (within blocks, within strata).</li>
<li><strong>Testing one coefficient in a regression.</strong> Permuting $y$ tests that $y$ is independent of <em>all</em> covariates, not that $\beta_j = 0$. Approximate methods (Freedman–Lane: permute residuals of the reduced model; ter Braak) are valid only asymptotically.</li>
<li><strong>Permuting after data-dependent steps.</strong> If features were selected, models tuned, or outliers removed using all the data, those steps must be repeated inside each permutation, or the null distribution is wrong.</li>
<li><strong>Discreteness.</strong> With small $N$ or many ties, exact levels like 0.05 may be unattainable; report the attained $p$-value rather than a binary decision, or use a randomized test if exact size matters.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`With 4 treated and 4 control units and $T = \bar X - \bar Y$, what is the smallest possible one-sided permutation $p$-value?`,
        a: R`$1/\binom84 = 1/70\approx0.0143$, attained when the four treated values are the four largest. (If ties are present, possibly larger.)` },
      { q: R`Why is $(1+\#\{T_b\ge T_0\})/(1+B)$ exactly valid while $\#\{T_b\ge T_0\}/B$ is not?`,
        a: R`Under $H_0$ the observed statistic $T_0$ and the $B$ permuted statistics are exchangeable, so $T_0$'s rank among all $B+1$ values is uniform. The probability that at most $k-1$ of the $T_b$ exceed $T_0$ is $k/(B+1)$, which is exactly what $\hat p\le k/(B+1)$ says. The naive version omits $T_0$ itself; it equals 0 with probability $1/(B+1)$, so it is anti-conservative and can report impossible $p = 0$.` },
      { q: R`Group sizes 10 and 40; the small group has variance 9, the large group variance 1; means equal. Does a permutation test of $\bar X - \bar Y$ at level 0.05 over- or under-reject?`,
        a: R`Over-reject. The permutation distribution uses the pooled variance $\bar\sigma^2 = 0.2\cdot9 + 0.8\cdot1 = 2.6$, giving $\Var_\pi(\bar X - \bar Y)\approx2.6(1/10 + 1/40) = 0.325$. The true variance is $9/10 + 1/40 = 0.925$. The null distribution is far too narrow, so the observed difference looks extreme too often. Studentizing (Welch) fixes it.` },
      { q: R`In a randomized experiment, what justifies the permutation test if the units were not sampled from any population?`,
        a: R`The randomization itself. Under the sharp null $Y_i(1) = Y_i(0)$ for every unit, the observed outcomes are fixed regardless of assignment, and the assignment was drawn uniformly from all possible assignments by design. Recomputing $T$ over re-randomizations gives the exact null distribution of $T$ — design-based inference, no sampling model required.` },
    ],
  });
})();

/* ---------------------------------------------------------------- cross-validation */
(function () {
  const R = String.raw;

  Atlas.topic({
    id: 'cross-validation',
    domain: 'resampling',
    order: 90,
    title: 'Cross-Validation',
    subtitle: 'Estimating prediction error by holding data out — what it estimates, how well, and how to fool yourself',
    level: 2,
    sources: { aos: '§13.6, §20.2', other: 'Stone (1974, 1977); Allen (1974); Craven & Wahba (1979); Shao (1993); Hastie, Tibshirani & Friedman (2009, §7.10); Arlot & Celisse (2010); Bates, Hastie & Tibshirani (2024)' },
    prereqs: ['bias-variance', 'model-selection'],
    related: ['kernel-regression', 'ridge-regression', 'lasso', 'splines', 'classifier-evaluation', 'jackknife', 'bootstrap'],
    tags: ['prediction error', 'model selection', 'LOOCV', 'GCV', 'nested CV', 'data leakage'],
    summary: R`Training error is an optimistic estimate of how a fitted model will do on new data, because the same data chose the fit. Cross-validation (CV) refits the procedure with part of the data held out and scores it on the held-out part, rotating so every point is held out once. It is nearly unbiased for the expected prediction error of the procedure trained on slightly fewer points, it is the default tool for tuning parameters, and for linear smoothers its leave-one-out version costs a single fit thanks to a hat-matrix identity. But CV estimates the <em>average</em> performance of a procedure better than the performance of the particular model you fitted, its naive standard errors are too small, it is not consistent for selecting the true model, and it is easily invalidated by any data-dependent step performed outside the folds.`,
    survey: {
      task: 'Estimate out-of-sample prediction error; choose tuning parameters and models',
      assumptions: R`Exchangeable (IID) observations; the full modelling pipeline is re-run inside each fold`,
      output: R`$\widehat{\text{Err}}_{\text{CV}}$, CV curve over tuning parameters, fold-level errors`,
      strengths: 'Model-agnostic; nearly unbiased for expected error; LOOCV is free for linear smoothers (hat-matrix shortcut)',
      weaknesses: 'High variance; naive SEs too small; inconsistent for model selection (Shao); invalid with leakage or dependence',
      complexity: R`$K$ refits ($n$ for LOOCV, or one fit for linear smoothers); nested CV multiplies by the inner $K$`,
    },
    sections: [
      {
        title: 'What we want to estimate',
        html: R`
<p>Let $\mathcal T = \{(x_i,y_i)\}_{i=1}^n$ be IID from $P$, $\hat f_{\mathcal T}$ the fitted predictor, and $L$ a loss. Two targets are often conflated:</p>
<div class="def" data-name="Conditional and expected test error">
<p>$$\text{Err}_{\mathcal T} = \E_{(X,Y)\sim P}\big[L(Y,\hat f_{\mathcal T}(X))\,\big|\,\mathcal T\big], \qquad \text{Err}(n) = \E_{\mathcal T}\big[\text{Err}_{\mathcal T}\big].$$
$\text{Err}_{\mathcal T}$ is how well <em>this</em> model will do; $\text{Err}(n)$ is how well the <em>procedure</em> does on average over training sets of size $n$.</p>
</div>
<p>The training error $\overline{\text{err}} = \frac1n\sum_iL(y_i,\hat f(x_i))$ underestimates both. For squared loss and fixed $x_i$, the expected optimism has a clean form (Efron, 2004): if $\text{Err}_{\text{in}}$ denotes the error at the same $x_i$ with fresh responses,
$$\E\big[\text{Err}_{\text{in}} - \overline{\text{err}}\big] = \frac2n\sum_{i=1}^n\Cov(\hat y_i, y_i),$$
which for a linear smoother $\hat y = Sy$ with homoscedastic noise is $2\sigma^2\tr(S)/n$ — the basis of Mallows' $C_p$ and AIC (see <a href="#/t/model-selection">model selection</a>). Analytic corrections need a model for the noise and a notion of degrees of freedom; CV needs neither.</p>
<div class="algo" data-name="K-fold cross-validation">
<ol>
<li>Randomly partition $\{1,\dots,n\}$ into $K$ folds $F_1,\dots,F_K$ of (nearly) equal size.</li>
<li>For $k = 1,\dots,K$: fit the <em>entire</em> procedure on the data outside $F_k$, giving $\hat f^{(-k)}$; compute the fold error $\widehat{E}_k = \frac1{|F_k|}\sum_{i\in F_k}L(y_i,\hat f^{(-k)}(x_i))$.</li>
<li>Return $\text{CV}_K = \frac1n\sum_{k}\sum_{i\in F_k}L(y_i,\hat f^{(-k)}(x_i))$.</li>
</ol>
</div>
<p>$K = n$ is leave-one-out CV (LOOCV); $K = 5$ or $10$ are standard.</p>`,
      },
      {
        title: 'What CV actually estimates',
        html: R`
<div class="thm" data-kind="Proposition" data-name="CV is unbiased for a smaller sample size">
<p>If $n$ is divisible by $K$ and the data are IID, then $\E[\text{CV}_K] = \text{Err}\big(n(1-1/K)\big)$, the expected test error of the procedure trained on $n - n/K$ points.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For $i\in F_k$, $\hat f^{(-k)}$ is fitted on $n - n/K$ IID points that do not include $(x_i,y_i)$, and $(x_i,y_i)$ is an independent draw from $P$. Hence $\E[L(y_i,\hat f^{(-k)}(x_i))\mid\hat f^{(-k)}] = \text{Err}_{\mathcal T^{(-k)}}$, and taking expectations gives $\text{Err}(n - n/K)$ for each term. Averaging preserves the expectation. (With random fold assignment the same holds conditionally on the partition.)</p>
</details>
<p>Since learning curves usually decrease in $n$, CV is <strong>pessimistically biased</strong> for $\text{Err}(n)$, by roughly the slope of the learning curve times $n/K$. The bias is negligible when the curve is flat (large $n$, simple model) and serious when it is steep (small $n$, flexible model). LOOCV has the smallest bias.</p>
<div class="note" data-name="CV estimates the procedure, not your model">
<p>A surprising and important fact: $\text{CV}$ is nearly uncorrelated with $\text{Err}_{\mathcal T}$ — the error of the model actually fitted to these data — and in some regimes negatively correlated (Hastie, Tibshirani & Friedman, 2009, §7.12; Bates, Hastie & Tibshirani, 2024). Intuitively, a training set that happens to contain unusual points both degrades the fitted model and makes its held-out errors look large in ways that do not line up. So CV is best read as an estimate of $\text{Err}(n)$: "how good is this <em>recipe</em> on data like these?"</p>
</div>
<p><strong>Variance and standard errors.</strong> The fold errors $\widehat E_k$ are positively correlated (training sets overlap heavily), so the naive standard error $\text{sd}(\widehat E_1,\dots,\widehat E_K)/\sqrt K$ underestimates the true variability of $\text{CV}_K$. Bengio & Grandvalet (2004) showed there is no unbiased estimator of $\Var(\text{CV}_K)$ based on a single run of $K$-fold CV. Bates, Hastie & Tibshirani (2024) document that naive CV intervals for prediction error can cover far below nominal and propose a nested-CV estimate of the MSE of CV that restores coverage.</p>
<p><strong>How large should $K$ be?</strong> Folklore holds that LOOCV has high variance because its $n$ training sets are nearly identical. The truth is procedure-dependent (Arlot & Celisse, 2010): for stable procedures such as least squares, LOOCV's variance is typically no larger than 10-fold CV's, while for unstable procedures (trees, nearest neighbours with small $k$, discontinuous selection) it can be. $K = 5$–$10$ is a robust compromise of bias, variance, and computation, and repeating $K$-fold CV over several random partitions removes the extra variance due to the particular split.</p>`,
      },
      {
        title: 'LOOCV for linear smoothers, and GCV',
        html: R`
<p>Many fitting procedures are <strong>linear smoothers</strong>: $\hat y = Sy$ for a matrix $S$ depending on the $x$'s and a tuning parameter but not on $y$ — least squares ($S = H$), <a href="#/t/ridge-regression">ridge</a>, smoothing <a href="#/t/splines">splines</a>, <a href="#/t/kernel-regression">kernel regression</a>, local polynomials. For them, LOOCV needs no refitting.</p>
<div class="thm" data-kind="Theorem" data-name="Leave-one-out shortcut">
<p>Let $\hat\beta = (X^\top X + \Omega)^{-1}X^\top y$ for a fixed symmetric matrix $\Omega$ with $X^\top X + \Omega$ positive definite (OLS: $\Omega = 0$; ridge: $\Omega = \lambda I$ or $\lambda$ on non-intercept coordinates; penalized splines: $\Omega = \lambda K$). With $S = X(X^\top X+\Omega)^{-1}X^\top$ and $\hat y^{(-i)}_i = x_i^\top\hat\beta^{(-i)}$ the prediction at $x_i$ from the fit without observation $i$,
$$y_i - \hat y^{(-i)}_i = \frac{y_i - \hat y_i}{1 - S_{ii}}, \qquad\text{so}\qquad \text{CV}_n = \frac1n\sum_{i=1}^n\Big(\frac{y_i-\hat y_i}{1-S_{ii}}\Big)^2.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $A = X^\top X + \Omega$ and $h_i = S_{ii} = x_i^\top A^{-1}x_i$. Deleting row $i$ replaces $A$ by $A - x_ix_i^\top$ and $X^\top y$ by $X^\top y - x_iy_i$. The Sherman–Morrison formula gives
$$(A - x_ix_i^\top)^{-1} = A^{-1} + \frac{A^{-1}x_ix_i^\top A^{-1}}{1-h_i}.$$
Multiply by $x_i^\top$ on the left: $x_i^\top(A - x_ix_i^\top)^{-1} = x_i^\top A^{-1}\big(1 + \frac{h_i}{1-h_i}\big) = \frac{x_i^\top A^{-1}}{1-h_i}$. Therefore
$$\hat y_i^{(-i)} = \frac{x_i^\top A^{-1}(X^\top y - x_iy_i)}{1-h_i} = \frac{\hat y_i - h_iy_i}{1-h_i},$$
and $y_i - \hat y_i^{(-i)} = \frac{y_i(1-h_i) - \hat y_i + h_iy_i}{1-h_i} = \frac{y_i - \hat y_i}{1-h_i}$.</p>
</details>
<p>The same identity holds for the Nadaraya–Watson kernel smoother, $S_{ij} = K_{ij}/\sum_lK_{il}$: removing point $i$ gives $\hat y^{(-i)}_i = \sum_{j\ne i}K_{ij}y_j/\sum_{j\ne i}K_{ij} = (\hat y_i - S_{ii}y_i)/(1-S_{ii})$, which rearranges to the same formula. In each case the deleted-point prediction is what you get by refitting with $y_i$ replaced by $\hat y^{(-i)}_i$ — the "self-consistency" property behind the shortcut. The quantity $(y_i - \hat y_i)/(1-h_{ii})$ is the PRESS residual (Allen, 1974), and it is the same algebra as the <a href="#/t/jackknife">jackknife</a> of a least-squares fit.</p>
<div class="def" data-name="Generalized cross-validation (Craven & Wahba, 1979)">
<p>Replace each $S_{ii}$ by its average $\tr(S)/n$:
$$\text{GCV} = \frac{\frac1n\sum_i(y_i-\hat y_i)^2}{\big(1-\tr(S)/n\big)^2}.$$</p>
</div>
<p>GCV is invariant under orthogonal rotations of the response (LOOCV is not), avoids instability when some $S_{ii}$ are near 1, and needs only $\tr(S)$, the effective degrees of freedom. Expanding $(1-\nu/n)^{-2}\approx1 + 2\nu/n$ shows $\text{GCV}\approx\overline{\text{err}}\,(1 + 2\nu/n)$, close to $C_p$/AIC with $\nu = \tr(S)$ degrees of freedom.</p>`,
        lab: 'cross-validation-curves',
      },
      {
        title: 'CV for model selection: efficiency versus consistency',
        html: R`
<p>CV is used far more often to <em>choose</em> than to <em>estimate</em>: pick the tuning parameter or model minimizing the CV curve. Two different goals lead to different verdicts.</p>
<div class="thm" data-kind="Theorem" data-name="LOOCV is inconsistent for selecting a linear model (Shao, 1993)">
<p>Consider selecting among linear models with fixed $p$, where the smallest correct model $\alpha_0$ is in the candidate set and at least one strictly larger correct model is too. With leave-$n_v$-out CV (training size $n_c = n - n_v$), the probability of selecting $\alpha_0$ tends to 1 if and only if $n_v/n\to1$ and $n_c\to\infty$. In particular LOOCV ($n_v = 1$) and $K$-fold CV with fixed $K$ select an over-fitting correct model with probability bounded away from zero.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>For a correct model $\alpha$ of dimension $p_\alpha$ with hat matrix $H_\alpha$, residuals are $e = (I-H_\alpha)\varepsilon$, and when $\max_ih_{ii}\to0$ the shortcut formula expands as $\text{CV}_1(\alpha) = \frac1n\sum_ie_i^2(1 + 2h_{ii} + \cdots) = \frac1n\|\varepsilon\|^2 - \frac1n\varepsilon^\top H_\alpha\varepsilon + \frac{2p_\alpha\sigma^2}{n} + o_p(n^{-1})$, using $\sum_ih_{ii}\varepsilon_i^2\approx\sigma^2\tr H_\alpha = \sigma^2p_\alpha$. The difference between two nested correct models is then $\frac1n\big(2\sigma^2(p_\alpha - p_{\alpha_0}) - \varepsilon^\top(H_\alpha - H_{\alpha_0})\varepsilon\big) + o_p(n^{-1})$, where $\varepsilon^\top(H_\alpha - H_{\alpha_0})\varepsilon/\sigma^2\sim\chi^2_{p_\alpha - p_{\alpha_0}}$ under normality. This is exactly AIC's comparison: the penalty $2(p_\alpha-p_{\alpha_0})$ is of the same order as the $\chi^2$ fluctuation, so the larger model wins with fixed positive probability $\P(\chi^2_d > 2d)$. For leave-$n_v$-out CV the penalty term becomes of order $p_\alpha\sigma^2/n_c$ rather than $2p_\alpha\sigma^2/n$; when $n_c/n\to0$ it dominates the $O_p(1/n)$ fluctuation, and the smallest correct model wins with probability tending to one (Shao, 1993).</p>
</details>
<p>The analogy is exact in spirit: LOOCV behaves like AIC (Stone, 1977, proved their asymptotic equivalence), and leave-most-out CV behaves like BIC. AIC-like criteria are <strong>efficient</strong>: they choose the model with (asymptotically) the best prediction error, even when no candidate is correct (Li, 1987). BIC-like criteria are <strong>consistent</strong>: they find the true model when it is among the candidates. If your goal is prediction, LOOCV/$K$-fold is the right tool; if your goal is identifying the true sparse structure, it over-selects.</p>
<div class="note" data-name="The one-standard-error rule">
<p>Because the CV curve is noisy and often flat near its minimum, Breiman et al. (1984) recommended choosing the simplest model whose CV error is within one standard error of the minimum. It trades a little expected prediction error for parsimony and stability, and partially counteracts CV's tendency to over-fit in selection. The standard error used is the naive fold-based one, so the rule is a heuristic, not a test.</p>
</div>`,
      },
      {
        title: 'Nested CV and data leakage',
        html: R`
<p>CV is valid only if each held-out fold is untouched by every data-dependent decision made in producing $\hat f^{(-k)}$. Two violations are common enough to deserve names.</p>
<div class="warn" data-name="Selection outside the folds">
<p>In a classic demonstration (Ambroise & McLachlan, 2002), with thousands of noise features and a few dozen samples whose labels are pure noise, selecting the features most correlated with the label <em>using all the data</em> and then cross-validating a classifier on those features produces CV error rates near zero. Every fold's test points helped choose the features, so the held-out data are not held out. Repeating the selection within each training fold gives the honest 50%. The same applies to any supervised preprocessing: target encoding, oversampling (SMOTE) before splitting, imputation models fitted on all rows, outlier removal based on residuals, and choosing transformations by looking at the outcome. The second lab reproduces this.</p>
</div>
<div class="warn" data-name="Reporting the minimum of the CV curve">
<p>If you choose $\lambda$ (or a model, or an architecture) to minimize $\text{CV}(\lambda)$, then $\min_\lambda\text{CV}(\lambda)$ is an optimistically biased estimate of the error of the tuned procedure — a winner's curse, worse with many candidates and noisy curves (Varma & Simon, 2006; Cawley & Talbot, 2010). <strong>Nested CV</strong> fixes this: an outer loop holds out a test fold; within the remaining data an inner CV chooses $\lambda$; the tuned model is scored on the outer fold. The outer average estimates the error of the whole "tune then fit" procedure. It costs $K_{\text{outer}}\times K_{\text{inner}}\times|\Lambda|$ fits.</p>
</div>
<p><strong>Structure in the data.</strong> CV's justification is exchangeability of observations. When that fails, so does naive CV:</p>
<ul>
<li><em>Time series</em>: random folds let the model see the future and neighbouring correlated points. Use forward-chaining (rolling-origin) evaluation, or blocked folds with a gap (buffer) between train and test (cf. <a href="#/t/block-bootstrap">block bootstrap</a>).</li>
<li><em>Groups</em> (patients with multiple scans, images from the same site, repeated measures): put all of a group in the same fold (group $K$-fold), otherwise CV measures within-group interpolation rather than generalization to new groups.</li>
<li><em>Spatial data</em>: spatially blocked folds, for the same reason.</li>
<li><em>Distribution shift</em>: if deployment data differ from training data, no resampling of the training data can estimate deployment error.</li>
</ul>
<p>CV and the <a href="#/t/bootstrap">bootstrap</a> are cousins: the bootstrap's out-of-bag error (each point scored by models whose resamples omitted it) is a smoothed relative of LOOCV, and the .632 and .632+ estimators (Efron & Tibshirani, 1997) blend it with the training error to correct the pessimistic bias that comes from training on only about 63.2% of the distinct points.</p>`,
        lab: 'cross-validation-leakage',
      },
    ],
    quiz: [
      { q: R`What exactly does $K$-fold CV estimate without bias, and why is that pessimistic for the model trained on all $n$ points?`,
        a: R`$\E[\text{CV}_K] = \text{Err}(n - n/K)$: the expected test error of the procedure trained on $n(1-1/K)$ observations. Since more data usually means lower error, $\text{Err}(n - n/K)\ge\text{Err}(n)$, so CV is biased upward for the error at size $n$, by about the learning-curve slope times $n/K$.` },
      { q: R`Show that for OLS the leave-one-out residual is $e_i/(1-h_{ii})$. What does a leverage of $h_{ii} = 0.9$ imply?`,
        a: R`By Sherman–Morrison, $x_i^\top(X^\top X - x_ix_i^\top)^{-1} = x_i^\top(X^\top X)^{-1}/(1-h_{ii})$, so $\hat y^{(-i)}_i = (\hat y_i - h_{ii}y_i)/(1-h_{ii})$ and $y_i - \hat y_i^{(-i)} = (y_i-\hat y_i)/(1-h_{ii})$. With $h_{ii} = 0.9$ the deleted residual is 10 times the ordinary one: the point largely determines its own fit, and its in-sample residual says little about how well it is predicted.` },
      { q: R`Why can't you report $\min_\lambda\text{CV}(\lambda)$ as the test error of the tuned model?`,
        a: R`The minimum over many noisy estimates is biased downward: some $\lambda$ look good partly because their CV noise was favourable. The selection step used the same folds that produced the estimate. Nested CV — tuning inside each outer training set and scoring on the untouched outer fold — estimates the error of the full tune-and-fit procedure.` },
      { q: R`Why is LOOCV not consistent for selecting the true linear model, and what modification is?`,
        a: R`Comparing two correct nested models, LOOCV's effective penalty for $d$ extra parameters is $2d\sigma^2/n$, the same order as the $\chi^2_d\sigma^2/n$ reduction in residual sum of squares from fitting noise, so the larger model wins with probability $\P(\chi^2_d > 2d) > 0$ — like AIC. Shao (1993) showed leave-$n_v$-out CV with $n_v/n\to1$ (and $n - n_v\to\infty$) is consistent, because it inflates the penalty by $n/n_c\to\infty$ — like BIC.` },
      { q: R`You standardize all features using the full data's means and SDs, then run 10-fold CV. Is this leakage? What about selecting the 20 features most correlated with $y$ on the full data?`,
        a: R`Unsupervised standardization uses no label information; it leaks only a little (the test fold's feature distribution slightly influences the scaling) and is usually harmless, though doing it inside folds is cleaner. Supervised feature selection on the full data is serious leakage: test-fold labels influenced which features were chosen, and the CV error can be wildly optimistic (near 0% error on pure noise). Selection must be repeated within each training fold.` },
    ],
  });
})();
