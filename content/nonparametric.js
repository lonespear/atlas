/* Nonparametric Estimation: empirical CDF, functionals, density estimation, smoothing, rank tests, robustness. */
(function () {
  const R = String.raw;

  /* ================================================================== */
  Atlas.topic({
    id: 'empirical-cdf',
    domain: 'nonparametric',
    order: 10,
    title: 'The Empirical CDF: Glivenko–Cantelli & DKW',
    subtitle: 'Estimating a whole distribution with no model at all, uniformly and with honest error bars',
    level: 2,
    sources: { aos: '7.1', cb: '—', other: 'Dvoretzky, Kiefer & Wolfowitz (1956); Massart (1990); van der Vaart (1998), Ch. 19' },
    prereqs: ['lln', 'clt'],
    related: ['inequalities', 'bootstrap', 'statistical-functionals', 'order-statistics', 'vc-theory'],
    tags: ['empirical process', 'uniform convergence', 'confidence band', 'goodness of fit'],
    summary: R`Put mass $1/n$ on every observation and you get the empirical distribution $\hat F_n$, the most important object in nonparametric statistics. Pointwise it is a binomial proportion: unbiased with variance $F(1-F)/n$. Much more is true. The Glivenko–Cantelli theorem says $\hat F_n \to F$ <em>uniformly</em> with probability one, and the Dvoretzky–Kiefer–Wolfowitz inequality makes that quantitative with a bound, $2e^{-2n\epsilon^2}$, that holds for every $n$ and every $F$. Turned around, DKW gives a distribution-free confidence band for the entire CDF, and the same supremum distance is the Kolmogorov–Smirnov statistic.`,
    survey: {
      task: 'Estimate the CDF $F$ with no parametric assumptions; simultaneous bands; goodness-of-fit tests',
      assumptions: 'IID real-valued data. Nothing about $F$ (KS null distribution additionally needs $F_0$ continuous)',
      output: R`Step function $\hat F_n$; band $\hat F_n \pm \epsilon_n$, $\epsilon_n=\sqrt{\log(2/\alpha)/(2n)}$; statistic $D_n$`,
      strengths: 'Exact finite-sample guarantees; uniform consistency; foundation of plug-in and bootstrap',
      weaknesses: R`Rate only $n^{-1/2}$; band is constant-width so loose in the tails; gives no density`,
      complexity: R`$O(n\log n)$ to sort; $O(\log n)$ per evaluation`,
    },
    sections: [
      {
        title: 'The estimator and its pointwise behaviour',
        html: R`
<p>Let $X_1,\dots,X_n \iid F$ on $\R$. With no model for $F$, the natural estimate puts probability $1/n$ on each observed point.</p>
<div class="def" data-name="Empirical distribution function">
<p>$$\hat F_n(x) = \frac1n\sum_{i=1}^n \ind(X_i \le x), \qquad x\in\R.$$
Equivalently $\hat F_n$ is the CDF of the discrete <em>empirical measure</em> $\P_n = \frac1n\sum_i \delta_{X_i}$. It is a right-continuous step function rising by $1/n$ at each order statistic $X_{(1)}\le\dots\le X_{(n)}$ (by $k/n$ at a value repeated $k$ times).</p>
</div>
<p>Fix $x$. The indicators $\ind(X_i\le x)$ are IID Bernoulli$(F(x))$, so $n\hat F_n(x) \sim \text{Binomial}(n, F(x))$. Everything about $\hat F_n$ at a single point follows from that one observation.</p>
<div class="thm" data-kind="Theorem" data-name="Pointwise properties of the ECDF">
<p>For each fixed $x$:</p>
<ol>
<li>$\E\,\hat F_n(x) = F(x)$ (unbiased);</li>
<li>$\Var\,\hat F_n(x) = F(x)\{1-F(x)\}/n$, so $\MSE \to 0$ and $\hat F_n(x)\pto F(x)$;</li>
<li>$\hat F_n(x) \asto F(x)$ (strong law);</li>
<li>$\sqrt n\{\hat F_n(x) - F(x)\} \dto \N\big(0,\,F(x)\{1-F(x)\}\big)$;</li>
<li>for $x\le y$, $\Cov\{\hat F_n(x),\hat F_n(y)\} = F(x)\{1-F(y)\}/n$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1)–(4) are the mean, variance, SLLN and CLT for an average of IID Bernoulli$(F(x))$ variables. For (5), with $x\le y$, $\ind(X\le x)\ind(X\le y) = \ind(X\le x)$, so $\Cov\{\ind(X\le x),\ind(X\le y)\} = F(x) - F(x)F(y) = F(x)\{1-F(y)\}$; distinct observations are independent, so the covariance of the averages is this divided by $n$.</p>
</details>
<p>Item (5) says that $\sqrt n(\hat F_n - F)$ behaves, jointly over finitely many points, like a Gaussian process with covariance $F(x\wedge y) - F(x)F(y)$. That is the covariance of a <strong>Brownian bridge</strong> $\mathbb B$ evaluated at $F(x)$, a fact that will reappear when we ask for the distribution of the largest deviation.</p>
<div class="note">
<p>The variance $F(1-F)/n$ is largest at the median ($1/(4n)$) and vanishes in the tails, where $\hat F_n$ is pinned near 0 or 1. Pointwise intervals are therefore narrow in the tails — but a pointwise 95% interval at each $x$ is not a 95% statement about the whole curve. For that we need uniform control.</p>
</div>`,
      },
      {
        title: 'Uniform convergence: the Glivenko–Cantelli theorem',
        html: R`
<p>Pointwise almost-sure convergence at every $x$ does not by itself give convergence of $\sup_x|\hat F_n(x)-F(x)|$: there are uncountably many $x$, and the exceptional null sets could pile up. What saves us is monotonicity — both $\hat F_n$ and $F$ are nondecreasing and bounded, so controlling them on a finite grid controls them everywhere.</p>
<div class="thm" data-kind="Theorem" data-name="Glivenko–Cantelli (1933)">
<p>For any distribution $F$ on $\R$,
$$D_n := \sup_{x\in\R}\big|\hat F_n(x) - F(x)\big| \asto 0.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Fix an integer $k$ and set $t_j = F^{-1}(j/k)$ for $j=1,\dots,k-1$ (with $F^{-1}(p) = \inf\{x : F(x)\ge p\}$), $t_0=-\infty$, $t_k = +\infty$. By construction $F(t_j^-) \le j/k$ and $F(t_{j-1}) \ge (j-1)/k$, so
$$F(t_j^-) - F(t_{j-1}) \le 1/k \qquad (j = 1,\dots,k).$$
(Atoms of $F$ larger than $1/k$ sit exactly at grid points, so they cause no trouble.) Now take $t\in[t_{j-1}, t_j)$. Monotonicity of both functions gives
$$\hat F_n(t) - F(t) \le \hat F_n(t_j^-) - F(t_{j-1}) \le \hat F_n(t_j^-) - F(t_j^-) + \tfrac1k,$$
$$\hat F_n(t) - F(t) \ge \hat F_n(t_{j-1}) - F(t_j^-) \ge \hat F_n(t_{j-1}) - F(t_{j-1}) - \tfrac1k.$$
Hence $D_n \le \max_{j}\max\{|\hat F_n(t_j)-F(t_j)|,\,|\hat F_n(t_j^-)-F(t_j^-)|\} + 1/k$. The maximum is over finitely many points, and at each one the strong law applies (also to $\hat F_n(t^-) = \frac1n\sum\ind(X_it t)$). So outside a null set $N_k$, $\limsup_n D_n \le 1/k$. On the complement of the null set $\bigcup_k N_k$ this holds for every $k$, hence $D_n\to0$ almost surely.</p>
</details>
<p>Glivenko–Cantelli is called the "fundamental theorem of statistics" because it is the precise sense in which a sample reveals its population. It is also the template for the uniform laws of large numbers behind <a href="#/t/vc-theory">VC theory</a>: the class of half-lines $\{(-\infty,x]\}$ has VC dimension 1, and the bracketing argument above is a special case of the general bracketing-entropy proof.</p>`,
      },
      {
        title: 'DKW: a finite-sample, distribution-free rate',
        html: R`
<p>Glivenko–Cantelli says $D_n\to0$ but not how fast or with what probability at a given $n$. The answer is remarkably clean.</p>
<div class="thm" data-kind="Theorem" data-name="Dvoretzky–Kiefer–Wolfowitz inequality (sharp constant: Massart, 1990)">
<p>For every $n\ge1$, every $\epsilon>0$ and every distribution $F$ on $\R$,
$$\P\Big(\sup_x |\hat F_n(x) - F(x)| > \epsilon\Big) \le 2e^{-2n\epsilon^2}.$$</p>
</div>
<p>Compare Hoeffding's inequality for a single point: $\P(|\hat F_n(x)-F(x)|>\epsilon)\le 2e^{-2n\epsilon^2}$. DKW says that taking the supremum over the whole line costs <em>nothing</em> in the exponent or the constant. Dvoretzky, Kiefer and Wolfowitz (1956) proved it with an unspecified constant $C$; Massart showed $C=2$ suffices, which is best possible. The sharp proof is delicate, but a version with a slightly worse prefactor follows from the Glivenko–Cantelli grid argument in a few lines.</p>
<div class="thm" data-kind="Proposition" data-name="A DKW-type bound from Hoeffding plus a grid">
<p>For any integer $k$ with $1/k < \epsilon$,
$\;\P(D_n > \epsilon) \le 4(k+1)\exp\{-2n(\epsilon - 1/k)^2\}.$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>From the proof of Glivenko–Cantelli, $D_n \le M_n + 1/k$ where $M_n$ is a maximum of $|\hat F_n(s)-F(s)|$ over at most $2(k+1)$ values $\hat F_n(t_j), \hat F_n(t_j^-)$, each an average of $n$ IID indicators. If $D_n>\epsilon$ then $M_n > \epsilon - 1/k$, so by the union bound and two-sided Hoeffding, $\P(D_n > \epsilon) \le 2(k+1)\cdot 2e^{-2n(\epsilon-1/k)^2}$.</p>
</details>
<p>Choosing $k\approx\sqrt n$ gives a bound of order $\sqrt n\, e^{-2n\epsilon^2}$, already enough to conclude $D_n = O_p(n^{-1/2})$ up to a log factor, and (via Borel–Cantelli) to reprove Glivenko–Cantelli. Massart's theorem removes the polynomial prefactor entirely.</p>
<div class="def" data-name="DKW confidence band">
<p>Set the right-hand side equal to $\alpha$: $\epsilon_n = \sqrt{\log(2/\alpha)/(2n)}$. Then with
$$L(x) = \max\{\hat F_n(x) - \epsilon_n,\,0\},\qquad U(x) = \min\{\hat F_n(x)+\epsilon_n,\,1\},$$
$\P\big(L(x)\le F(x)\le U(x)\text{ for all }x\big) \ge 1-\alpha$ for every $F$ and every $n$.</p>
</div>
<div class="ex" data-name="How wide is it?">
<p>For $\alpha = 0.05$, $\log(40)/2 = 1.844$ so $\epsilon_n = 1.358/\sqrt n$: $0.136$ at $n=100$, $0.043$ at $n=1000$. To pin the whole CDF to within $\pm0.01$ you need $n \approx 18{,}400$. The number $1.358$ is also, by a happy coincidence of the sharp constant, essentially the asymptotic 95% point of the Kolmogorov distribution below — the DKW band is nearly exact for large $n$ when $F$ is continuous.</p>
</div>
<p>The band has constant width, which is its weakness: in the tails, where $F(1-F)$ is tiny, it is far wider than necessary. Weighted bands based on $\sup|\hat F_n-F|/\sqrt{F(1-F)}$ (Anderson–Darling-type) or on likelihood ratios (Owen, 1995) are tighter there.</p>`,
        lab: 'empirical-cdf-dkw',
      },
      {
        title: 'Kolmogorov–Smirnov: the same supremum as a test',
        html: R`
<p>To test $H_0: F = F_0$ for a fully specified $F_0$, use the distance the DKW inequality controls.</p>
<div class="def" data-name="Kolmogorov–Smirnov statistic">
<p>$D_n = \sup_x |\hat F_n(x) - F_0(x)|$. For continuous $F_0$ it is computed exactly from the order statistics:
$$D_n = \max_{1\le i\le n}\max\Big\{\tfrac in - F_0(X_{(i)}),\;F_0(X_{(i)}) - \tfrac{i-1}{n}\Big\}.$$</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Distribution-freeness of KS">
<p>If $F_0$ is continuous and $H_0$ holds, the law of $D_n$ does not depend on $F_0$: it equals the law of $\sup_{u\in[0,1]}|\hat G_n(u) - u|$ where $\hat G_n$ is the ECDF of $n$ IID Uniform$(0,1)$ variables.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $U_i = F_0(X_i)$. By the probability integral transform $U_i\iid\text{Unif}(0,1)$ when $F_0$ is continuous. Since $F_0$ is nondecreasing, $\{X_i\le x\}\subseteq\{U_i\le F_0(x)\}$, and the difference event $\{X_i>x,\,F_0(X_i)=F_0(x)\}$ has probability zero (it requires $X_i$ to land in a flat stretch of $F_0$, which carries no mass). So almost surely $\hat F_n(x) = \hat G_n(F_0(x))$ for all $x$ simultaneously, and $D_n = \sup_x|\hat G_n(F_0(x)) - F_0(x)|$. As $x$ ranges over $\R$, $F_0(x)$ ranges over a set containing $(0,1)$ (continuity plus the intermediate value theorem), so the supremum is $\sup_{u}|\hat G_n(u)-u|$.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="Kolmogorov (1933); Donsker (1952)">
<p>Under $H_0$ with $F_0$ continuous, $\sqrt n\,D_n \dto K := \sup_{0\le u\le1}|\mathbb B(u)|$, where $\mathbb B$ is a Brownian bridge, and
$$\P(K\le t) = 1 - 2\sum_{k=1}^\infty(-1)^{k-1}e^{-2k^2t^2},\qquad t>0.$$
The upper 5% point is $1.358$; the 1% point is $1.628$.</p>
</div>
<p>The proof is Donsker's functional CLT — $\sqrt n(\hat G_n(u) - u)$ converges as a process to $\mathbb B$, matching the covariance $u\wedge v - uv$ found in the first section — followed by the continuous mapping theorem for the sup-norm, and a reflection-principle computation for the Brownian bridge. Note the leading term: $\P(K>t)\approx 2e^{-2t^2}$, exactly the DKW bound with $\epsilon = t/\sqrt n$.</p>
<p>The <strong>two-sample</strong> version, $D_{m,n} = \sup_x|\hat F_m(x) - \hat G_n(x)|$, tests $F=G$ and satisfies $\sqrt{mn/(m+n)}\,D_{m,n}\dto K$ under $H_0$; it is distribution-free for the same reason, and is in fact a <a href="#/t/rank-tests">rank statistic</a>.</p>`,
        lab: 'empirical-cdf-ks',
      },
      {
        title: 'Pitfalls: estimated parameters, discreteness, and power',
        html: R`
<div class="warn" data-name="Estimating parameters breaks the null distribution">
<p>The most common misuse: testing normality by computing $D_n$ against $\N(\bar X, s^2)$ and comparing to Kolmogorov's table. Fitting the parameters pulls $F_{\hat\theta}$ toward $\hat F_n$, so $D_n$ is stochastically <em>smaller</em> than under a fixed $F_0$; the test is extremely conservative and loses power: for the normal family the correct 5% point of $\sqrt n D_n$ is about $0.89$ (Lilliefors), not $1.36$, so the nominal 5% test almost never rejects a true normal. The correct null distribution depends on the family and must be simulated — for the normal family this is the <strong>Lilliefors test</strong>; a <a href="#/t/parametric-bootstrap">parametric bootstrap</a> handles any family. Try it in the lab above.</p>
</div>
<div class="warn" data-name="Discrete F">
<p>DKW still holds for discrete $F$ (it is valid for every $F$), but the KS null distribution is no longer distribution-free: $F_0$ no longer maps onto all of $(0,1)$, the supremum is over fewer effective points, and the Kolmogorov critical values give a conservative test.</p>
</div>
<div class="warn" data-name="KS is insensitive in the tails">
<p>Because $\Var\hat F_n(x)$ is smallest in the tails, deviations there are a small fraction of the scale $1/\sqrt n$ that $D_n$ measures, and KS has little power against tail alternatives. Anderson–Darling, $n\int (\hat F_n - F_0)^2 /\{F_0(1-F_0)\}\d F_0$, reweights by the inverse pointwise variance and is usually preferred for goodness of fit; Cramér–von Mises, $n\int(\hat F_n-F_0)^2\d F_0$, sits in between.</p>
</div>
<div class="note" data-name="The bigger picture">
<p>$\hat F_n$ is the nonparametric MLE of $F$: among all distributions, it maximizes $\prod_i \P(\{X_i\})$. Every plug-in estimator $T(\hat F_n)$ of a <a href="#/t/statistical-functionals">statistical functional</a>, every <a href="#/t/bootstrap">bootstrap</a> procedure, and every sample quantile $\hat F_n^{-1}(p)$ inherits its behaviour from the results on this page. Its limitation is equally fundamental: $\hat F_n$ is a discrete measure and has no density, which is why density estimation (<a href="#/t/histograms">histograms</a>, <a href="#/t/kernel-density">kernels</a>) is a genuinely harder problem with slower rates.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`What is the exact distribution of $n\hat F_n(x)$ for fixed $x$, and what does it imply about the variance of $\hat F_n(x)$ at the median versus at the 1% quantile?`,
        a: R`$n\hat F_n(x)\sim\text{Binomial}(n,F(x))$, so $\Var\hat F_n(x) = F(1-F)/n$. At the median this is $0.25/n$; at the 1% quantile it is $0.0099/n$, about 25 times smaller. The ECDF is far more precise in the tails in absolute terms.` },
      { q: R`Why does pointwise a.s. convergence of $\hat F_n(x)$ for each $x$ not immediately give $\sup_x|\hat F_n - F|\to0$ a.s., and what property rescues the argument?`,
        a: R`Each $x$ has its own exceptional null set and there are uncountably many $x$, so the union need not be null; and pointwise convergence does not imply uniform convergence anyway. Monotonicity and boundedness of both functions let us sandwich $\hat F_n - F$ between its values on a finite grid with mesh $1/k$, so only finitely many SLLN statements are needed for each $k$.` },
      { q: R`Using DKW, how large must $n$ be to guarantee $\sup_x|\hat F_n - F|\le 0.05$ with probability at least 0.95?`,
        a: R`Need $2e^{-2n(0.05)^2}\le 0.05$, i.e. $n\ge \log(40)/(2\cdot 0.0025) = 3.689/0.005 \approx 738$. This holds for any $F$ whatever.` },
      { q: R`You test normality with the KS statistic against $\N(\bar X,s^2)$ and Kolmogorov's 5% critical value. Is the test's actual size larger or smaller than 5%, and why?`,
        a: R`Smaller (conservative). Estimating $\mu,\sigma$ from the same data makes the fitted CDF hug $\hat F_n$, so $D_n$ is stochastically smaller than the distribution-free null law assumes. The remedy is the Lilliefors table or a parametric bootstrap of $D_n$ with re-estimation inside each replicate.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'statistical-functionals',
    domain: 'nonparametric',
    order: 20,
    title: 'Statistical Functionals, Plug-in & Influence Functions',
    subtitle: 'Parameters as maps from distributions to numbers, and the derivative that governs their estimates',
    level: 2,
    sources: { aos: '7.2', cb: '§10.2', other: 'Hampel et al. (1986), Ch. 2; van der Vaart (1998), Ch. 20; Fernholz (1983)' },
    prereqs: ['empirical-cdf', 'delta-method'],
    related: ['bootstrap', 'jackknife', 'robust-statistics', 'method-of-moments'],
    tags: ['plug-in', 'influence function', 'von Mises expansion', 'standard error'],
    summary: R`In nonparametric inference a parameter is a <em>functional</em> $\theta = T(F)$: the mean, a quantile, a correlation, a trimmed mean. The plug-in principle estimates it by $T(\hat F_n)$. How good is that estimate? Differentiate $T$ with respect to $F$ in the direction of a point mass: the result, the <strong>influence function</strong> $L(x)$, measures how much one observation at $x$ moves the estimate. A first-order expansion $T(\hat F_n)\approx T(F) + \frac1n\sum L(X_i)$ — the nonparametric delta method — turns any smooth plug-in estimator into an average, giving its asymptotic normality and a standard error $\sqrt{\int L^2\d F/n}$ for free.`,
    survey: {
      task: R`Estimate $\theta = T(F)$ and its standard error without a parametric model`,
      assumptions: R`IID data; $T$ differentiable at $F$ (Hadamard for rigour); $\int L^2\d F<\infty$`,
      output: R`$\hat\theta = T(\hat F_n)$; influence function $L$; $\widehat{\se} = \hat\tau/\sqrt n$; Wald intervals`,
      strengths: 'One recipe for estimation and standard errors; IF also quantifies robustness',
      weaknesses: 'IF can be messy to derive; involves unknowns (e.g. a density for quantiles); fails for non-smooth $T$',
      complexity: R`$O(n)$ once $L$ is known; $O(n)$ evaluations of $T$ for an empirical (jackknife) version`,
    },
    sections: [
      {
        title: 'Parameters as functionals',
        html: R`
<p>Parametric inference indexes distributions by a finite vector. Nonparametric inference has no such index, so we describe the target directly as a function of the distribution.</p>
<div class="def" data-name="Statistical functional; plug-in estimator">
<p>A <em>statistical functional</em> is a map $T$ from a class of distributions to $\R$ (or $\R^k$). The <em>plug-in estimator</em> of $\theta = T(F)$ is $\hat\theta_n = T(\hat F_n)$.</p>
</div>
<table>
<thead><tr><th>Parameter</th><th>$T(F)$</th><th>Plug-in $T(\hat F_n)$</th></tr></thead>
<tbody>
<tr><td>Mean</td><td>$\int x\d F(x)$</td><td>$\bar X_n$</td></tr>
<tr><td>Variance</td><td>$\int (x-\mu_F)^2\d F$</td><td>$\hat\sigma^2 = \frac1n\sum(X_i-\bar X)^2$ (divisor $n$)</td></tr>
<tr><td>$p$-quantile</td><td>$F^{-1}(p)$</td><td>$\hat F_n^{-1}(p)$, an order statistic</td></tr>
<tr><td>CDF at $t$</td><td>$F(t)$</td><td>$\hat F_n(t)$</td></tr>
<tr><td>Correlation</td><td>$\Cov_F(X,Y)/\sqrt{\Var_F X\,\Var_F Y}$</td><td>sample correlation $r$</td></tr>
<tr><td>Skewness</td><td>$\E_F(X-\mu)^3/\sigma^3$</td><td>sample skewness</td></tr>
</tbody>
</table>
<p>A functional is <strong>linear</strong> if $T(F) = \int r(x)\d F(x)$ for some function $r$. Then $T(\hat F_n) = \frac1n\sum r(X_i)$ is an average, and the LLN, CLT and the variance $\Var r(X)/n$ are immediate. The mean and $F(t)$ are linear; variance, quantiles and correlation are not. The theory below reduces nonlinear functionals to linear ones by differentiation, exactly as the ordinary <a href="#/t/delta-method">delta method</a> reduces $g(\bar X)$ to $g'(\mu)\bar X$.</p>
<div class="note">
<p>Plug-in is consistent whenever $T$ is continuous with respect to a distance in which $\hat F_n\to F$. For the sup-distance, Glivenko–Cantelli supplies the convergence; the mean, however, is <em>not</em> sup-norm continuous (moving mass $\epsilon$ to $1/\epsilon^2$ changes the mean by $1/\epsilon$ while the CDF moves by at most $\epsilon$). This is the first hint that the mean is fragile — the theme of <a href="#/t/robust-statistics">robust statistics</a>.</p>
</div>`,
      },
      {
        title: 'The influence function',
        html: R`
<p>To linearize $T$ at $F$, we perturb $F$ in the most elementary direction available: toward a point mass.</p>
<div class="def" data-name="Influence function (Hampel, 1974)">
<p>Let $\delta_x$ be the point mass at $x$ and $F_\epsilon = (1-\epsilon)F + \epsilon\delta_x$. The influence function of $T$ at $F$ is the Gâteaux derivative
$$L_F(x) = \lim_{\epsilon\downarrow0}\frac{T\big((1-\epsilon)F+\epsilon\delta_x\big) - T(F)}{\epsilon}.$$
The <em>empirical influence function</em> is $\hat L(x) = L_{\hat F_n}(x)$.</p>
</div>
<p>$L_F(x)$ is the rate at which $T$ responds to contaminating $F$ with a little mass at $x$: roughly, $n$ times the change in $\hat\theta$ caused by adding one observation at $x$ to a large sample. If $T$ is differentiable in a suitable sense, the derivative in a general direction $G - F$ is $\int L_F\d G$ (linearity), and taking $G = F$ gives the key identity
$$\int L_F(x)\d F(x) = 0.$$</p>
<div class="ex" data-name="Four influence functions">
<p><strong>Linear functional</strong> $T(F)=\int r\d F$: $T(F_\epsilon) = (1-\epsilon)T(F) + \epsilon r(x)$, so $L(x) = r(x) - T(F)$. The mean has $L(x) = x-\mu$, unbounded in $x$.</p>
<p><strong>Variance.</strong> With $m_2 = \int x^2\d F$: $\mu_\epsilon = \mu + \epsilon(x-\mu)$, $m_{2,\epsilon} = m_2+\epsilon(x^2-m_2)$, and $\sigma^2_\epsilon = m_{2,\epsilon}-\mu_\epsilon^2$. Differentiating at 0: $x^2 - m_2 - 2\mu(x-\mu) = (x-\mu)^2 - \sigma^2$. Quadratic, even more sensitive to outliers than the mean.</p>
<p><strong>Quantile</strong> $\xi_p = F^{-1}(p)$, $F$ with density $f>0$ at $\xi_p$. Differentiate $F_\epsilon(\xi_{p,\epsilon}) = p$, i.e. $(1-\epsilon)F(\xi_{p,\epsilon}) + \epsilon\ind(x\le\xi_{p,\epsilon}) = p$, at $\epsilon=0$: $-p + f(\xi_p)\dot\xi + \ind(x\le\xi_p) = 0$, so
$$L(x) = \frac{p - \ind(x\le\xi_p)}{f(\xi_p)}.$$
Bounded: an observation's influence depends only on which side of $\xi_p$ it falls. For the median, $L(x) = \sgn(x-m)/\{2f(m)\}$.</p>
<p><strong>Chain rule.</strong> If $\theta = g(T(F))$ with $g$ differentiable, $L_{g\circ T} = g'(T(F))\,L_T$. So the standard deviation has $L(x) = \{(x-\mu)^2-\sigma^2\}/(2\sigma)$.</p>
</div>`,
        lab: 'statistical-functionals-influence',
      },
      {
        title: 'The nonparametric delta method',
        html: R`
<p>Expanding $T$ around $F$ along the path toward $\hat F_n$ (a "von Mises expansion"):
$$T(\hat F_n) = T(F) + \int L_F\d(\hat F_n - F) + R_n = T(F) + \frac1n\sum_{i=1}^n L_F(X_i) + R_n,$$
using $\int L_F\d F = 0$. The leading term is an average of IID mean-zero variables with variance $\tau^2 = \int L_F^2\d F$.</p>
<div class="thm" data-kind="Theorem" data-name="Nonparametric delta method">
<p>Suppose $T$ is Hadamard differentiable at $F$ (tangentially to the space of bounded functions, with respect to the sup-norm) with influence function $L_F$, and $\tau^2 = \int L_F^2\d F<\infty$. Then
$$\sqrt n\,\big\{T(\hat F_n) - T(F)\big\} \dto \N(0,\tau^2),$$
and if $\hat\tau^2 = \frac1n\sum_i \hat L(X_i)^2 \pto \tau^2$, the interval $T(\hat F_n)\pm z_{\alpha/2}\,\hat\tau/\sqrt n$ has asymptotic coverage $1-\alpha$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Hadamard differentiability means $\{T(F + t_nh_n) - T(F)\}/t_n \to T'_F(h)$ whenever $t_n\to0$ and $h_n\to h$ uniformly, with $T'_F$ linear and continuous. Take $t_n = n^{-1/2}$ and $h_n = \sqrt n(\hat F_n - F)$. By Donsker's theorem $h_n$ converges in distribution (in sup-norm) to $\mathbb B\circ F$. The functional delta method — the continuous mapping theorem applied to the difference quotient, via Skorokhod representation — gives $\sqrt n\{T(\hat F_n)-T(F)\}\dto T'_F(\mathbb B\circ F)$, a linear functional of a Gaussian process, hence normal. Its variance is identified by the linear term: $T'_F(\hat F_n - F) = \frac1n\sum L_F(X_i)$, whose variance is $\tau^2/n$. The remainder $R_n$ is $o_p(n^{-1/2})$ by the definition of the derivative.</p>
</details>
<p>Gâteaux differentiability (which defines $L$) is too weak to control $R_n$; Fréchet differentiability in sup-norm is too strong (it fails for quantiles). Hadamard differentiability is the right middle ground: quantiles at a point of positive density, M-estimators with smooth $\psi$, trimmed means, and smooth functions of moments all qualify.</p>
<div class="ex" data-name="Standard error of the sample median">
<p>From $L(x) = \sgn(x-m)/\{2f(m)\}$, $\tau^2 = 1/\{4f(m)^2\}$, so $\se(\hat m)\approx 1/\{2f(m)\sqrt n\}$. For $\N(\mu,\sigma^2)$, $f(m) = 1/(\sigma\sqrt{2\pi})$, giving $\se = \sigma\sqrt{\pi/(2n)}\approx 1.253\,\sigma/\sqrt n$: the median has asymptotic relative efficiency $2/\pi\approx0.637$ against the mean at the normal. Note the unknown $f(m)$ in the formula — estimating it requires <a href="#/t/kernel-density">density estimation</a>, which is why the <a href="#/t/bootstrap">bootstrap</a> is popular for medians.</p>
</div>
<div class="ex" data-name="Standard error of the sample variance">
<p>$L(x) = (x-\mu)^2-\sigma^2$ gives $\tau^2 = \mu_4 - \sigma^4$ with $\mu_4 = \E(X-\mu)^4$, so $\se(\hat\sigma^2)\approx\sqrt{(\mu_4-\sigma^4)/n}$, estimated by plugging in sample moments. Under normality $\mu_4 = 3\sigma^4$ and this reduces to $\sigma^2\sqrt{2/n}$, the textbook formula — but that textbook formula is wrong (often very wrong) for heavy-tailed data, while the influence-function version is not.</p>
</div>`,
      },
      {
        title: 'Empirical influence, the jackknife and the bootstrap',
        html: R`
<p>Three standard-error methods are three ways of getting at $\tau^2=\int L_F^2\d F$:</p>
<div class="cols">
<div><h4>Analytic IF</h4><p>Derive $L_F$, plug in $\hat F_n$, average $\hat L(X_i)^2$. Exact first-order theory, but needs a derivation and sometimes nuisance quantities like $f(m)$.</p></div>
<div><h4>Jackknife</h4><p>The <a href="#/t/jackknife">jackknife</a> replaces the derivative by a finite difference with $\epsilon = -1/(n-1)$: deleting $X_i$ moves $\hat F_n$ toward $-\delta_{X_i}$. The pseudo-values $n\hat\theta - (n-1)\hat\theta_{(-i)}\approx \hat\theta + \hat L(X_i)$.</p></div>
<div><h4>Bootstrap</h4><p>Simulates the whole distribution of $T(\hat F_n^*) - T(\hat F_n)$; to first order its variance is $\frac1n\int \hat L^2\d\hat F_n$. Captures higher-order terms (skewness) the linearization misses.</p></div>
</div>
<p>All three agree to first order for smooth $T$. They part ways exactly when $T$ is not smooth enough: the jackknife variance estimate is <em>inconsistent</em> for the median (the finite difference at $\epsilon\approx-1/n$ jumps between neighbouring order statistics, a step too coarse for a functional whose derivative involves $f(m)$), while the bootstrap remains consistent there.</p>
<div class="warn" data-name="When linearization fails">
<p>The maximum $T(F) = \sup\{x: F(x)<1\}$ is not differentiable: its "influence function" is $0$ below the endpoint and infinite above. The delta method gives nothing, and the sample maximum converges at rate $n^{-1}$ (not $n^{-1/2}$) to a non-normal limit; see <a href="#/t/order-statistics">order statistics</a>. Likewise, at a quantile where $f(\xi_p)=0$ the formula $1/f$ explodes and the rate changes.</p>
</div>`,
      },
      {
        title: 'Influence as a robustness diagnostic',
        html: R`
<p>The influence function has a second life: it quantifies how vulnerable an estimator is to a small fraction of bad data. If the data come from the contaminated model $F_\epsilon = (1-\epsilon)F+\epsilon\delta_x$, the asymptotic bias of $T$ is approximately $\epsilon L_F(x)$.</p>
<div class="def" data-name="Gross-error sensitivity">
<p>$\gamma^* = \sup_x |L_F(x)|$. An estimator is <em>B-robust</em> if $\gamma^*<\infty$.</p>
</div>
<p>The mean ($\gamma^*=\infty$) and the variance ($\gamma^*=\infty$, and quadratic) are not B-robust; the median ($\gamma^* = 1/\{2f(m)\}$) is. There is a tension: $\tau^2 = \int L^2\d F$ governs efficiency, $\sup|L|$ governs robustness, and the Cramér–Rao bound forces $\tau^2\ge 1/I(\theta)$ for estimators of a parametric location — the MLE attains it with $L\propto$ score, which is unbounded for the normal. Hampel's optimality problem (minimize $\tau^2$ subject to $\gamma^*\le c$) is solved by Huber's clipped score; see <a href="#/t/robust-statistics">M-estimators</a>.</p>
<p>The empirical counterpart, Tukey's <strong>sensitivity curve</strong>
$$SC_n(x) = n\big\{T_n(x_1,\dots,x_{n-1},x) - T_{n-1}(x_1,\dots,x_{n-1})\big\},$$
is what the lab above plots: drag $x$ and watch $SC_n$ trace out $L_F$.</p>`,
      },
    ],
    quiz: [
      { q: R`Show that the influence function of any linear functional $\int r\d F$ integrates to zero against $F$. Why must this hold for every (suitably differentiable) functional?`,
        a: R`$L(x) = r(x) - \int r\d F$, so $\int L\d F = \int r\d F - \int r\d F = 0$. In general, if the derivative in direction $G-F$ is $\int L\d G$ (linear in $G$), then the direction $G=F$ is the zero perturbation and must have derivative zero, so $\int L\d F = 0$. This is what makes $\frac1n\sum L(X_i)$ a mean-zero average.` },
      { q: R`Derive the asymptotic variance of the sample $p$-quantile from its influence function.`,
        a: R`$L(x) = \{p - \ind(x\le\xi_p)\}/f(\xi_p)$. The numerator is $p$ minus a Bernoulli$(p)$, with second moment $p(1-p)$. So $\tau^2 = p(1-p)/f(\xi_p)^2$ and $\sqrt n(\hat\xi_p - \xi_p)\dto\N(0,p(1-p)/f(\xi_p)^2)$, the classical result.` },
      { q: R`For data from a $t_5$ distribution, is the textbook standard error $\hat\sigma^2\sqrt{2/n}$ for the sample variance too small or too large? Use the influence function.`,
        a: R`Too small. The correct asymptotic variance is $(\mu_4-\sigma^4)/n$; the textbook formula assumes $\mu_4 = 3\sigma^4$. A $t_5$ has kurtosis $\mu_4/\sigma^4 = 9$, so $\mu_4 - \sigma^4 = 8\sigma^4$, four times the normal-theory $2\sigma^4$: the true standard error is about twice the textbook one.` },
      { q: R`Why is the jackknife variance estimate inconsistent for the sample median while the bootstrap is consistent?`,
        a: R`The jackknife uses a finite difference with step $\approx 1/n$; for the median deleting one point moves the estimate between adjacent order statistics, so the pseudo-values take essentially only two values, and the resulting variance estimate does not converge to $1/\{4f(m)^2 n\}$ (scaled appropriately, it converges in distribution to a non-degenerate random limit). The median is Hadamard differentiable but not smooth enough for the jackknife's finite-difference linearization; the bootstrap, which resamples the whole empirical distribution, remains consistent.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'histograms',
    domain: 'nonparametric',
    order: 30,
    title: 'Histograms & the Bias–Variance Trade-off of Binning',
    subtitle: 'The simplest density estimator, its risk, and why the optimal bin width shrinks like the cube root of n',
    level: 2,
    sources: { aos: '20.1–20.2', cb: '—', other: 'Scott (1979, 2015); Freedman & Diaconis (1981); Rudemo (1982)' },
    prereqs: ['empirical-cdf', 'bias-variance'],
    related: ['kernel-density', 'cross-validation', 'orthogonal-series'],
    tags: ['density estimation', 'smoothing', 'bin width', 'risk'],
    summary: R`A histogram estimates a density by counting points in bins and dividing by $nh$. It is the cleanest place to see the central tension of all smoothing: wide bins average many points (low variance) but blur the density (high bias); narrow bins track the density but are noisy. Balancing the integrated squared bias $\approx\frac{h^2}{12}\int (f')^2$ against the integrated variance $\approx 1/(nh)$ gives an optimal bin width $h^*\propto n^{-1/3}$ and a risk of order $n^{-2/3}$ — slower than the parametric $n^{-1}$, and slower than kernel estimators achieve. The bin width can be chosen from the data by cross-validation.`,
    survey: {
      task: 'Estimate a univariate density $f$; exploratory display',
      assumptions: R`IID; $f$ with absolutely continuous derivative, $\int (f')^2<\infty$`,
      output: R`Piecewise-constant $\hat f_n$; bin width $h$ (Scott, Freedman–Diaconis or CV)`,
      strengths: 'Transparent, fast, exact counts; risk theory is elementary',
      weaknesses: R`Discontinuous; depends on origin; rate $n^{-2/3}$ only; poor in $d>2$`,
      complexity: R`$O(n)$ after choosing bins; CV over a grid of $h$ is $O(n\cdot\#\text{grid})$`,
    },
    sections: [
      {
        title: 'Setup: the histogram as a density estimator',
        html: R`
<p>Suppose $f$ is supported on $[0,1]$ (rescale otherwise). Partition into $m$ bins $B_j = [(j-1)h, jh)$ of width $h = 1/m$. Let $\nu_j$ be the number of observations in $B_j$, $\hat p_j = \nu_j/n$ and $p_j = \int_{B_j}f$.</p>
<div class="def" data-name="Histogram estimator">
<p>$$\hat f_n(x) = \sum_{j=1}^m \frac{\hat p_j}{h}\,\ind(x\in B_j).$$
It is a genuine density (nonnegative, integrates to one) that is constant on each bin.</p>
</div>
<p>For $x\in B_j$, $n\hat p_j\sim\text{Binomial}(n,p_j)$, so
$$\E\hat f_n(x) = \frac{p_j}{h},\qquad \Var\hat f_n(x) = \frac{p_j(1-p_j)}{nh^2}.$$
The mean is the <em>average</em> of $f$ over the bin, not $f(x)$ — that is the bias. The variance has $h^2$ in the denominator but $p_j\approx hf(x)$ in the numerator, so $\Var\approx f(x)/(nh)$: shrinking the bin by half doubles the variance.</p>
<div class="note">
<p>The histogram is a local average: $\hat f_n(x) = \{\hat F_n(b_j)-\hat F_n(a_j)\}/h$ is a finite-difference derivative of the <a href="#/t/empirical-cdf">ECDF</a> over the bin $[a_j,b_j)$ containing $x$. Differentiating a noisy step function is ill-posed, and the step $h$ is precisely the regularization parameter.</p>
</div>`,
      },
      {
        title: 'Risk decomposition and the optimal bin width',
        html: R`
<p>Measure performance by the mean integrated squared error, $R(f,\hat f_n) = \E\int (\hat f_n - f)^2 = \int b^2(x)\d x + \int v(x)\d x$, the integrated squared bias plus integrated variance.</p>
<div class="thm" data-kind="Theorem" data-name="Histogram risk (Scott, 1979)">
<p>Assume $f'$ is absolutely continuous and $\int (f')^2<\infty$. As $h\to0$ and $nh\to\infty$,
$$R(f,\hat f_n) = \frac{h^2}{12}\int (f'(u))^2\d u + \frac{1}{nh} + o(h^2) + o\Big(\frac1{nh}\Big).$$
The minimizer of the leading terms and the resulting risk are
$$h^* = \Big(\frac{6}{n\int (f')^2}\Big)^{1/3},\qquad R^* \approx \Big(\frac34\Big)^{2/3}\Big(\int (f')^2\Big)^{1/3} n^{-2/3}.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><strong>Bias.</strong> For $x\in B_j$ with bin midpoint $c_j$, Taylor-expand around $x$: $p_j = \int_{B_j}f(u)\d u = h f(x) + h f'(x)(c_j - x) + O(h^3)$. So $b(x) = p_j/h - f(x) \approx f'(x)(c_j - x)$. Then
$$\int_{B_j} b^2 \approx (f'(\tilde x_j))^2\int_{B_j}(c_j-x)^2\d x = (f'(\tilde x_j))^2\frac{h^3}{12},$$
and summing over bins, $\sum_j h\,(f'(\tilde x_j))^2\cdot\frac{h^2}{12}\to\frac{h^2}{12}\int (f')^2$ as a Riemann sum.</p>
<p><strong>Variance.</strong> $\int v = \sum_j h\cdot\frac{p_j(1-p_j)}{nh^2} = \frac{1}{nh}\big(1 - \sum_j p_j^2\big)$, and $\sum_j p_j^2 \approx h\int f^2 \to 0$, so $\int v = \frac1{nh} + o(\frac1{nh})$.</p>
<p><strong>Optimization.</strong> Minimize $ah^2 + b/h$ with $a = \int(f')^2/12$, $b=1/n$: $2ah = b/h^2$ gives $h^3 = b/(2a) = 6/(n\int(f')^2)$. At the optimum $ah^2 = b/(2h)$, so $R^* = \frac{3b}{2h^*} = \frac32\cdot6^{-1/3}(\int(f')^2)^{1/3}n^{-2/3}$, and $\frac32\cdot6^{-1/3} = (3/4)^{2/3}\approx0.826$.</p>
</details>
<p>Notice the pointwise bias is $O(h)$ — the histogram's bins are not centred on $x$ — yet the integrated squared bias is $O(h^2)$ because the bias averages to zero over each bin. A kernel estimator centres its window at $x$, which kills the $O(h)$ term pointwise and improves the integrated bias to $O(h^4)$; that is the source of its better rate.</p>
<div class="ex" data-name="Normal reference rules">
<p>If $f$ is $\N(\mu,\sigma^2)$, then $\int (f')^2 = 1/(4\sqrt\pi\,\sigma^3)$ and $h^* = (24\sqrt\pi)^{1/3}\sigma n^{-1/3}\approx 3.49\,\sigma n^{-1/3}$ (<strong>Scott's rule</strong>). Replacing $\sigma$ by a robust spread gives the <strong>Freedman–Diaconis rule</strong> $h = 2\,\mathrm{IQR}\,n^{-1/3}$. Sturges' rule, $m = 1+\log_2 n$ bins, grows only logarithmically and oversmooths badly for large $n$; it is the default in some software and should be overridden.</p>
</div>
<p>Plug in numbers: with $n=1000$ standard normal points, Scott gives $h\approx0.35$ — about 20 bins across $\pm3.5$. The risk $\approx 0.826\cdot(0.141)^{1/3}\cdot 1000^{-2/3} \approx 0.0043$.</p>`,
        lab: 'histograms-risk',
      },
      {
        title: 'Choosing the bin width by cross-validation',
        html: R`
<p>The optimal $h^*$ depends on the unknown $\int (f')^2$. Instead, estimate the risk itself. Write
$$\int(\hat f_n - f)^2 = \underbrace{\int \hat f_n^2 - 2\int\hat f_n f}_{J(h)} + \int f^2,$$
and the last term does not involve $h$. The cross term $\int\hat f_n f = \E_X[\hat f_n(X)]$ for a new $X\sim f$ independent of the data, which leave-one-out estimates by $\frac1n\sum_i\hat f_{(-i)}(X_i)$.</p>
<div class="thm" data-kind="Proposition" data-name="Leave-one-out CV score for histograms (Rudemo, 1982)">
<p>The cross-validation estimate $\hat J(h) = \int\hat f_n^2 - \frac2n\sum_i \hat f_{(-i)}(X_i)$ has the closed form
$$\hat J(h) = \frac{2}{(n-1)h} - \frac{n+1}{(n-1)h}\sum_{j=1}^m\hat p_j^2,$$
and $\E\hat J(h) = \E J(h)$ computed with sample size $n-1$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\int\hat f_n^2 = \sum_j h(\nu_j/(nh))^2 = \sum_j\nu_j^2/(n^2h)$. If $X_i\in B_j$, deleting it leaves $\nu_j - 1$ points in that bin among $n-1$, so $\hat f_{(-i)}(X_i) = (\nu_j-1)/((n-1)h)$. Summing over the $\nu_j$ points in each bin, $\sum_i\hat f_{(-i)}(X_i) = \sum_j\nu_j(\nu_j-1)/((n-1)h)$. Hence
$$\hat J = \frac{\sum\nu_j^2}{n^2h} - \frac{2\sum\nu_j^2}{n(n-1)h} + \frac{2n}{n(n-1)h},$$
and substituting $\nu_j = n\hat p_j$, the coefficient of $\sum\hat p_j^2/h$ is $1 - 2n/(n-1) = -(n+1)/(n-1)$. Unbiasedness: for each $i$, $\hat f_{(-i)}$ is independent of $X_i$, so $\E\hat f_{(-i)}(X_i) = \E\int\hat f_{n-1}f$.</p>
</details>
<p>Minimizing $\hat J$ over a grid of $h$ gives a data-driven bin width. CV is unbiased for the risk (up to the constant) but noisy — the CV minimizer converges to $h^*$ very slowly in relative terms — so its choice can wander noticeably from sample to sample (try it in the lab below). See <a href="#/t/cross-validation">cross-validation</a> for the general theory.</p>`,
      },
      {
        title: 'The origin problem, and averaged shifted histograms',
        html: R`
<p>The risk theory treats only $h$, but a histogram has a second tuning parameter: the bin <em>origin</em> $x_0$. For moderate $n$, shifting the grid by a fraction of $h$ can create or erase a mode — an unsettling property for an exploratory tool.</p>
<div class="ex" data-name="Averaged shifted histogram (Scott, 1985)">
<p>Average $k$ histograms with the same width $h$ and origins $x_0 + \ell h/k$, $\ell = 0,\dots,k-1$. The result is a piecewise-constant estimate on a finer mesh $h/k$ whose value at $x$ is a weighted count of points within distance $h$, with triangular weights $1 - |\ell|/k$. As $k\to\infty$ the ASH converges to a <a href="#/t/kernel-density">kernel density estimator</a> with the triangular kernel and bandwidth $h$ — and inherits its $n^{-4/5}$ rate. The histogram's origin dependence is precisely the price of an off-centre window.</p>
</div>
<div class="warn" data-name="Practical pitfalls">
<ul>
<li><strong>Counts versus density.</strong> With unequal bins, plot $\hat p_j/h_j$, not $\nu_j$; otherwise wide bins look like modes.</li>
<li><strong>Discretized data.</strong> If values are rounded to a grid (ages, prices), bins narrower than the rounding unit, or misaligned with it, produce spurious comb patterns.</li>
<li><strong>Heavy tails.</strong> A single bin width cannot serve both the bulk and the tail; consider a transformation (log) before binning.</li>
<li><strong>Dimension.</strong> In $\R^d$ the integrated variance is $\approx1/(nh^d)$ and the optimal risk is $O(n^{-2/(2+d)})$ — the curse of dimensionality, worse even than for kernels.</li>
</ul>
</div>`,
        lab: 'histograms-binwidth',
      },
      {
        title: 'Where histograms sit in the theory',
        html: R`
<p>Three rates are worth keeping in mind for estimating a density with two derivatives:</p>
<table>
<thead><tr><th>Estimator</th><th>Integrated sq. bias</th><th>Integrated variance</th><th>Optimal $h$</th><th>Risk</th></tr></thead>
<tbody>
<tr><td>Histogram</td><td>$O(h^2)$</td><td>$O(1/(nh))$</td><td>$n^{-1/3}$</td><td>$n^{-2/3}$</td></tr>
<tr><td>Kernel (second-order)</td><td>$O(h^4)$</td><td>$O(1/(nh))$</td><td>$n^{-1/5}$</td><td>$n^{-4/5}$</td></tr>
<tr><td>Correct parametric model</td><td>0</td><td>$O(1/n)$</td><td>—</td><td>$n^{-1}$</td></tr>
</tbody>
</table>
<p>The $n^{-4/5}$ rate is minimax over densities with a bounded second derivative, so the histogram is rate-suboptimal for smooth densities; it is rate-optimal only over the larger class of Lipschitz densities (one derivative), where $n^{-2/3}$ is the minimax rate. In return it is simple, exact, and makes no pretence of smoothness — which is why it remains the default exploratory display, and why its risk calculation is the template for every smoother that follows.</p>`,
      },
    ],
    quiz: [
      { q: R`Halving the bin width $h$ changes the integrated squared bias and the integrated variance by what factors (to leading order)?`,
        a: R`Integrated squared bias $\propto h^2$ falls by a factor of 4; integrated variance $\propto 1/(nh)$ doubles. Whether halving helps depends on which term dominates at the current $h$.` },
      { q: R`If the sample size grows by a factor of 8, by what factor should the bin width shrink, and by what factor does the optimal risk fall?`,
        a: R`$h^*\propto n^{-1/3}$, so $h$ halves. $R^*\propto n^{-2/3}$, so the risk falls by $8^{2/3} = 4$. A parametric estimator's risk would fall by a factor of 8.` },
      { q: R`Why is the pointwise bias of a histogram $O(h)$, and its integrated squared bias $O(h^2)$, whereas a kernel estimator achieves $O(h^2)$ and $O(h^4)$?`,
        a: R`At $x$ the bias is $\approx f'(x)(c_j - x)$, linear in the offset from the bin centre, so it is $O(h)$ unless $x$ happens to be at the centre. Squaring and integrating over a bin gives $f'^2h^3/12$ per bin, i.e. $O(h^2)$ in total. A kernel estimator centres its window at $x$, cancelling the first-order term and leaving $O(h^2)$ pointwise and $O(h^4)$ integrated.` },
      { q: R`What quantity does cross-validation estimate unbiasedly, and why can it drop $\int f^2$?`,
        a: R`It estimates $J(h) = \E[\int\hat f^2 - 2\int\hat f f]$ (for sample size $n-1$), which equals the risk minus $\int f^2$. Since $\int f^2$ does not depend on $h$, minimizing $J$ and minimizing the risk select the same $h$.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'kernel-density',
    domain: 'nonparametric',
    order: 40,
    title: 'Kernel Density Estimation',
    subtitle: 'Smoothing the empirical distribution with a bump at every point, and the h⁴ versus 1/(nh) trade-off',
    level: 2,
    sources: { aos: '20.3', cb: '—', other: 'Silverman (1986); Wand & Jones (1995); Tsybakov (2009), Ch. 1' },
    prereqs: ['histograms'],
    related: ['kernel-regression', 'cross-validation', 'spectral-density-clustering', 'orthogonal-series'],
    tags: ['density estimation', 'bandwidth', 'AMISE', 'curse of dimensionality'],
    summary: R`A kernel density estimator places a small, smooth bump of width $h$ at every observation and adds them up. Because each bump is centred at its data point, the first-order bias that plagues the histogram cancels: the bias is $\frac{h^2}{2}\sigma_K^2 f''(x)$ and the variance $f(x)R(K)/(nh)$. Balancing them gives an optimal bandwidth $h\propto n^{-1/5}$ and risk $\propto n^{-4/5}$ — the minimax rate for twice-differentiable densities. The bandwidth matters enormously; the shape of the kernel barely matters. In $d$ dimensions the rate degrades to $n^{-4/(4+d)}$, the curse of dimensionality in its purest form.`,
    survey: {
      task: R`Estimate a smooth density $f$ on $\R$ or $\R^d$`,
      assumptions: R`IID; $f''$ continuous and square-integrable; $h\to0$, $nh\to\infty$`,
      output: R`Smooth $\hat f_h$; bandwidth via rule of thumb, LSCV or plug-in`,
      strengths: R`Smooth, origin-free; rate $n^{-4/5}$; simple theory; works for any kernel`,
      weaknesses: R`Single global $h$ over- or under-smooths parts of $f$; boundary bias; rate $n^{-4/(4+d)}$ in $\R^d$`,
      complexity: R`$O(n)$ per evaluation point (FFT binning: $O(n + g\log g)$ on a grid of $g$ points)`,
    },
    sections: [
      {
        title: 'From histograms to kernels',
        html: R`
<p>The histogram's flaw is that $\hat f(x)$ counts points in a bin that is not centred at $x$. Centring the window gives the "naive" estimator $\#\{i:|X_i-x|\le h\}/(2nh)$, and replacing the hard window with a smooth weight gives the general form.</p>
<div class="def" data-name="Kernel density estimator (Rosenblatt, 1956; Parzen, 1962)">
<p>Let $K$ be a <em>kernel</em>: $K\ge0$, $\int K = 1$, $\int uK(u)\d u = 0$, $\sigma_K^2 = \int u^2K(u)\d u\in(0,\infty)$, and write $R(K) = \int K^2$. With bandwidth $h>0$ and $K_h(u) = h^{-1}K(u/h)$,
$$\hat f_h(x) = \frac1n\sum_{i=1}^n K_h(x - X_i) = \frac{1}{nh}\sum_{i=1}^n K\Big(\frac{x-X_i}{h}\Big).$$</p>
</div>
<p>Two equivalent views. As a sum of bumps: each observation contributes a copy of $K_h$ with mass $1/n$. As a convolution: $\hat f_h = K_h * \P_n$, the <a href="#/t/empirical-cdf">empirical measure</a> smoothed by $K_h$. Since $\hat f_h$ is a mixture of densities it is itself a density, as smooth as $K$.</p>
<table>
<thead><tr><th>Kernel</th><th>$K(u)$</th><th>$\sigma_K^2$</th><th>$R(K)$</th><th>Efficiency</th></tr></thead>
<tbody>
<tr><td>Epanechnikov</td><td>$\tfrac34(1-u^2)$, $|u|\le1$</td><td>$1/5$</td><td>$3/5$</td><td>1</td></tr>
<tr><td>Biweight</td><td>$\tfrac{15}{16}(1-u^2)^2$, $|u|\le1$</td><td>$1/7$</td><td>$5/7$</td><td>0.994</td></tr>
<tr><td>Triangular</td><td>$1-|u|$, $|u|\le1$</td><td>$1/6$</td><td>$2/3$</td><td>0.986</td></tr>
<tr><td>Gaussian</td><td>$\phi(u)$</td><td>$1$</td><td>$1/(2\sqrt\pi)$</td><td>0.951</td></tr>
<tr><td>Uniform</td><td>$\tfrac12$, $|u|\le1$</td><td>$1/3$</td><td>$1/2$</td><td>0.930</td></tr>
</tbody>
</table>
<p>The efficiency column is explained below: it is the fraction of the sample the Epanechnikov kernel needs to match the optimal risk of the given kernel, and it shows that kernel choice is a second-order concern.</p>`,
      },
      {
        title: 'Bias and variance',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Pointwise bias and variance of the KDE">
<p>Let $f$ have two continuous derivatives in a neighbourhood of $x$, and let $h\to0$ with $nh\to\infty$. Then
$$\E\hat f_h(x) - f(x) = \frac{h^2}{2}\sigma_K^2 f''(x) + o(h^2),\qquad \Var\hat f_h(x) = \frac{f(x)R(K)}{nh} + o\Big(\frac1{nh}\Big).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><strong>Bias.</strong> The summands are IID, so $\E\hat f_h(x) = \E K_h(x-X) = \int h^{-1}K((x-y)/h)f(y)\d y$. Substitute $u = (x-y)/h$:
$$\E\hat f_h(x) = \int K(u)f(x-hu)\d u = \int K(u)\Big\{f(x) - hu f'(x) + \tfrac{h^2u^2}{2}f''(x) + o(h^2)\Big\}\d u.$$
The three conditions on $K$ make the terms $f(x)\cdot1$, $-hf'(x)\cdot0$ and $\frac{h^2}{2}f''(x)\sigma_K^2$. (The remainder is uniformly $o(h^2)$ by continuity of $f''$ if $K$ has compact support; for the Gaussian kernel use dominated convergence with bounded $f''$.)</p>
<p><strong>Variance.</strong> $\Var\hat f_h(x) = \frac1n\{\E K_h(x-X)^2 - (\E K_h(x-X))^2\}$. The second term is $O(1/n)$. The first:
$$\frac1n\int h^{-2}K^2\Big(\frac{x-y}{h}\Big)f(y)\d y = \frac1{nh}\int K^2(u)f(x-hu)\d u = \frac{f(x)R(K)}{nh} + o\Big(\frac1{nh}\Big).$$
Since $1/n = o(1/(nh))$, the claim follows.</p>
</details>
<div class="note">
<p>The bias is proportional to the <em>curvature</em> $f''$: the estimator fills in valleys ($f''>0$, positive bias) and shaves off peaks ($f''<0$, negative bias), because averaging over a window always pulls toward the local mean. Its variance is proportional to $f(x)$ itself, so the estimator is relatively noisiest in low-density regions: $\mathrm{sd}/f\approx\sqrt{R(K)/(nhf(x))}$.</p>
</div>
<p>Unlike the histogram, there is no $O(h)$ term at any $x$: symmetry ($\int uK = 0$) cancels it exactly because the window is centred. And $\hat f_h(x)$ is asymptotically normal, $\sqrt{nh}\{\hat f_h(x) - \E\hat f_h(x)\}\dto\N(0,f(x)R(K))$, by the Lindeberg CLT for triangular arrays.</p>`,
      },
      {
        title: 'AMISE, the optimal bandwidth and the n^(−4/5) rate',
        html: R`
<p>Integrating squared bias and variance over $x$ (with $R(f'') = \int (f'')^2<\infty$) gives the asymptotic mean integrated squared error.</p>
<div class="thm" data-kind="Theorem" data-name="AMISE and optimal bandwidth">
<p>$$\mathrm{AMISE}(h) = \frac{h^4}{4}\sigma_K^4R(f'') + \frac{R(K)}{nh}.$$
It is minimized by
$$h^* = \Big(\frac{R(K)}{\sigma_K^4R(f'')}\Big)^{1/5}n^{-1/5},\qquad \mathrm{AMISE}^* = \frac54\big(\sigma_KR(K)\big)^{4/5}R(f'')^{1/5}\,n^{-4/5}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>With $A = \sigma_K^4R(f'')/4$ and $B = R(K)/n$, minimize $Ah^4 + B/h$: setting the derivative $4Ah^3 - B/h^2$ to zero gives $h^5 = B/(4A) = R(K)/(n\sigma_K^4R(f''))$. At the optimum $Ah^4 = B/(4h)$, so the minimum is $\frac{5B}{4h^*} = \frac54 R(K)n^{-1}\big(n\sigma_K^4R(f'')/R(K)\big)^{1/5} = \frac54\sigma_K^{4/5}R(K)^{4/5}R(f'')^{1/5}n^{-4/5}$.</p>
</details>
<p>Three lessons fall out of this formula.</p>
<ol>
<li><strong>Rate.</strong> The risk is $O(n^{-4/5})$ — better than the histogram's $n^{-2/3}$, worse than parametric $n^{-1}$. Over the class of densities with $\|f''\|$ bounded this rate is minimax: no estimator can do better uniformly.</li>
<li><strong>Kernel shape is secondary.</strong> $K$ enters only through $C(K) = \sigma_KR(K)$, which is scale-invariant. Among nonnegative kernels it is minimized by the Epanechnikov kernel (Epanechnikov, 1969; the same parabola appears in Hodges & Lehmann, 1956); the ratio $C(K_{\text{Epa}})/C(K)$ is the efficiency in the table above. Even the uniform kernel loses only 7%.</li>
<li><strong>Bandwidth is primary.</strong> $h^*$ depends on the unknown roughness $R(f'')$. Getting $h$ wrong by a factor of 2 can cost more than any choice of kernel.</li>
</ol>
<div class="ex" data-name="Normal reference rule">
<p>If $f$ is $\N(\mu,\sigma^2)$ then $R(f'') = 3/(8\sqrt\pi\sigma^5)$. With the Gaussian kernel ($\sigma_K=1$, $R(K)=1/(2\sqrt\pi)$), $h^* = (4/3)^{1/5}\sigma n^{-1/5}\approx1.06\,\sigma n^{-1/5}$. <strong>Silverman's rule of thumb</strong>, $h = 0.9\min(\hat\sigma,\widehat{\mathrm{IQR}}/1.34)n^{-1/5}$, shrinks this a little and uses a robust spread so that it oversmooths less for skewed or bimodal data. It remains a normal-reference rule: for a bimodal density it is typically too large and can merge the modes.</p>
</div>`,
        lab: 'kernel-density-bandwidth',
      },
      {
        title: 'Choosing the bandwidth from data',
        html: R`
<p>As for histograms, decompose $\int(\hat f_h - f)^2 = \int\hat f_h^2 - 2\int\hat f_hf + \int f^2$ and estimate the $h$-dependent part by leave-one-out.</p>
<div class="def" data-name="Least-squares cross-validation (Rudemo, 1982; Bowman, 1984)">
<p>$$\mathrm{LSCV}(h) = \int\hat f_h^2(x)\d x - \frac2n\sum_{i=1}^n\hat f_{h,(-i)}(X_i),\qquad \hat f_{h,(-i)}(X_i) = \frac{1}{n-1}\sum_{j\ne i}K_h(X_i-X_j).$$
For the Gaussian kernel the first term is exact: $\int\hat f_h^2 = n^{-2}\sum_{i,j}\phi_{\sqrt2h}(X_i - X_j)$, since the convolution of two $\N(0,h^2)$ densities is $\N(0,2h^2)$.</p>
</div>
<p>$\E\,\mathrm{LSCV}(h) + \int f^2$ is exactly the MISE at sample size $n-1$, and the LSCV choice is asymptotically optimal: $\mathrm{ISE}(\hat h_{\text{CV}})/\min_h\mathrm{ISE}(h)\to1$ a.s. (Stone, 1984). The catch is variability — the relative error $\hat h_{\text{CV}}/h^*-1$ is of order $n^{-1/10}$ (Hall & Marron, 1987), so LSCV bandwidths wander from sample to sample, and LSCV tends to undersmooth data with ties or rounding. The alternatives:</p>
<ul>
<li><strong>Plug-in</strong> (Sheather & Jones, 1991): estimate $R(f'')$ with a pilot KDE (whose own bandwidth is chosen by a second-stage formula) and plug into $h^*$. Much less variable; the standard default in modern software.</li>
<li><strong>Likelihood CV</strong>: maximize $\sum_i\log\hat f_{h,(-i)}(X_i)$. Minimizes a Kullback–Leibler risk, but is ruined by heavy tails (a single isolated point drives $h$ up).</li>
</ul>
<div class="warn" data-name="Boundary bias">
<p>If $f$ is supported on $[0,\infty)$ and $f(0)>0$, kernels near 0 spill mass onto the negative axis: $\E\hat f_h(0)\to f(0)/2$ for a symmetric kernel, an $O(1)$ bias, and the bias is $O(h)$ or worse throughout the boundary region $[0,h)$. Fixes include reflecting the data about the boundary, boundary-corrected kernels, and estimating on a transformed scale (e.g. $\log X$) then transforming back. Try the Exponential population in the lab above.</p>
</div>`,
      },
      {
        title: 'Higher-order kernels and adaptivity',
        html: R`
<p>If $f$ has $\beta>2$ derivatives, a kernel of <em>order</em> $\beta$ — one with $\int u^jK = 0$ for $1\le j<\beta$ — pushes the bias to $O(h^\beta)$ and the risk to $O(n^{-2\beta/(2\beta+1)})$, again minimax over the corresponding smoothness class. The price: for $\beta>2$, $\int u^2K=0$ forces $K$ to take negative values, so $\hat f_h$ can be negative (truncate and renormalize), and the asymptotic gains often do not materialize at moderate $n$.</p>
<p>A deeper limitation is that one global $h$ must serve the whole line. A density with a sharp peak and long flat tails wants a small $h$ at the peak and a large one in the tails. <strong>Variable-bandwidth</strong> estimators (Abramson's square-root law $h_i\propto f(X_i)^{-1/2}$ from a pilot, or $k$-nearest-neighbour bandwidths) address this, as do the locally adaptive <a href="#/t/orthogonal-series">wavelet</a> methods, which are designed precisely for spatially inhomogeneous smoothness.</p>
<div class="note" data-name="KDE is everywhere">
<p>Mean shift and density-based clustering climb the gradient of a KDE (<a href="#/t/spectral-density-clustering">density-based clustering</a>); the Nadaraya–Watson regression estimator is a ratio of two KDEs (<a href="#/t/kernel-regression">next topic</a>); a smoothed bootstrap resamples from $\hat f_h$ instead of $\hat F_n$; kernel two-sample tests compare kernel mean embeddings in an RKHS norm.</p>
</div>`,
      },
      {
        title: 'The curse of dimensionality',
        html: R`
<p>In $\R^d$ use a product or radial kernel with a common bandwidth: $\hat f_h(x) = \frac{1}{nh^d}\sum_iK\big((x-X_i)/h\big)$. The same calculation gives bias still $O(h^2)$ (proportional to the Laplacian of $f$ for a spherical kernel) but variance $f(x)R(K)/(nh^d)$, because the window now has volume of order $h^d$ and contains about $nh^df(x)$ points.</p>
<div class="thm" data-kind="Corollary" data-name="Rate in d dimensions">
<p>$\mathrm{AMISE}(h) = c_1h^4 + c_2/(nh^d)$ is minimized at $h^*\asymp n^{-1/(d+4)}$, with risk $\asymp n^{-4/(d+4)}$.</p>
</div>
<p>To halve the risk you need $2^{(d+4)/4}$ times as many observations: about 2.4 times more in one dimension, 4 times more in four, 11 times more in ten. Silverman (1986, Table 4.2) computed the sample size needed to estimate a standard normal density at the origin with relative mean squared error 0.1 using the optimal bandwidth: 4 points in one dimension, 19 in two, 223 in four, and about 842,000 in ten. The second lab makes the same point by simulation.</p>
<div class="note">
<p>Geometrically: in high dimensions a neighbourhood that contains a fixed fraction of the data is not local. For data uniform on $[0,1]^d$, a cube capturing 10% of the points has side $0.1^{1/d}$ — that is $0.79$ in $d=10$, nearly the full range of each coordinate. Local averaging methods (kernels, <a href="#/t/knn">nearest neighbours</a>, local regression) all suffer the same fate; structural assumptions (additivity as in <a href="#/t/gam">GAMs</a>, low intrinsic dimension, sparsity) are the only escape.</p>
</div>`,
        lab: 'kernel-density-curse',
      },
    ],
    quiz: [
      { q: R`Where is the KDE biased upward and where downward, and why does the sign follow $f''$?`,
        a: R`$\Bias\approx\frac{h^2}{2}\sigma_K^2f''(x)$: positive where $f$ is convex (valleys, tails), negative where concave (peaks). Averaging $f$ over a symmetric window around $x$ gives $f(x)$ plus a curvature correction; at a peak the neighbours are lower, so the average is too low.` },
      { q: R`If you use $h = 2h^*$ instead of the AMISE-optimal $h^*$, by what factor does AMISE increase? What about $h = h^*/2$?`,
        a: R`At $h^*$ the bias part is $1/5$ of the total and the variance part $4/5$ (since $Ah^4 = B/(4h)$). With $h=2h^*$: bias $\times16$, variance $\times\frac12$, total $= \frac{16}5 + \frac25 = 3.6$ times. With $h=h^*/2$: bias $\times\frac1{16}$, variance $\times2$, total $= \frac1{80} + \frac85\approx1.61$ times. Oversmoothing is penalized more heavily than undersmoothing by the same factor.` },
      { q: R`Why does the choice of kernel matter so much less than the choice of bandwidth?`,
        a: R`The kernel enters the optimal AMISE only through the factor $(\sigma_KR(K))^{4/5}$, which varies by at most a few percent across all sensible nonnegative kernels (efficiencies 0.93–1). The bandwidth enters as $h^4$ and $1/h$, so a factor-of-two error changes the risk by 60% or more.` },
      { q: R`In $d = 8$, what is the KDE's optimal risk exponent, and how many times more data does it take to halve the risk?`,
        a: R`Risk $\asymp n^{-4/(d+4)} = n^{-1/3}$. Halving it requires multiplying $n$ by $2^3 = 8$.` },
      { q: R`Why can the Gaussian-kernel LSCV criterion be computed exactly without numerical integration?`,
        a: R`$\int\hat f_h^2 = n^{-2}\sum_{i,j}\int\phi_h(x-X_i)\phi_h(x-X_j)\d x = n^{-2}\sum_{i,j}\phi_{\sqrt2h}(X_i-X_j)$, because the integral is a convolution of two normal densities, which is normal with the variances added.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'kernel-regression',
    domain: 'nonparametric',
    order: 50,
    title: 'Nadaraya–Watson & Local Polynomial Regression',
    subtitle: 'Weighted local averages for E[Y | X = x], and why fitting a local line beats fitting a local constant',
    level: 2,
    sources: { aos: '20.4', cb: '—', other: 'Fan (1992, 1993); Fan & Gijbels (1996); Wand & Jones (1995), Ch. 5; Loader (1999)' },
    prereqs: ['kernel-density', 'simple-linear-regression'],
    related: ['splines', 'knn', 'gam', 'weighted-gls', 'cross-validation'],
    tags: ['smoothing', 'linear smoother', 'boundary bias', 'design bias', 'bandwidth'],
    summary: R`To estimate a regression function $r(x) = \E(Y\mid X=x)$ without a parametric form, average the responses of observations whose $x_i$ are near $x$, weighting by a kernel. The Nadaraya–Watson estimator fits a weighted local <em>constant</em>; its bias has an extra "design bias" term $h^2\sigma_K^2r'g'/g$ that depends on how the $x_i$ are spread, and near the edges of the data its bias is $O(h)$. Fitting a weighted local <em>line</em> instead removes both defects at no asymptotic cost in variance, which is why local linear regression is the default kernel smoother. Both are linear smoothers $\hat{\mathbf r} = L\by$, so effective degrees of freedom and leave-one-out cross-validation come for free.`,
    survey: {
      task: R`Estimate $r(x) = \E(Y\mid X=x)$ in low dimension`,
      assumptions: R`$Y_i = r(x_i)+\epsilon_i$, $\E\epsilon_i=0$, $\Var\epsilon_i = \sigma^2(x_i)$; $r''$ continuous; design density $g>0$`,
      output: R`$\hat r(x) = \sum_i\ell_i(x)Y_i$; smoother matrix $L$; df $=\tr L$; CV-chosen $h$`,
      strengths: R`Local linear: no design bias, automatic boundary correction, minimax-efficient; easy CV`,
      weaknesses: R`Single global $h$; variance blows up in sparse regions; curse of dimensionality`,
      complexity: R`$O(n)$ per evaluation point; $O(n^2)$ for all fitted values and CV`,
    },
    sections: [
      {
        title: 'Local averaging and linear smoothers',
        html: R`
<p>Observe $(x_1,Y_1),\dots,(x_n,Y_n)$ with $Y_i = r(x_i) + \epsilon_i$. The regression function $r$ is assumed only to be smooth. The simplest idea is to average the $Y_i$ with $x_i$ near $x$, with weights decaying in $|x_i - x|$.</p>
<div class="def" data-name="Nadaraya–Watson estimator (1964)">
<p>$$\hat r_{\mathrm{NW}}(x) = \sum_{i=1}^n w_i(x)Y_i,\qquad w_i(x) = \frac{K\big((x-x_i)/h\big)}{\sum_{j}K\big((x-x_j)/h\big)}.$$
Equivalently $\hat r_{\mathrm{NW}}(x) = \argmin_{c}\sum_i K_h(x-x_i)(Y_i - c)^2$: a weighted least-squares fit of a <em>constant</em>.</p>
</div>
<p>For random design, $\hat r_{\mathrm{NW}}(x) = \hat m(x)/\hat g(x)$ where $\hat g$ is the <a href="#/t/kernel-density">KDE</a> of the $x_i$ and $\hat m(x) = \frac1n\sum_iK_h(x-x_i)Y_i$ estimates $m = rg$. As $h\to0$ the fitted values approach the $Y_i$ themselves (interpolation); as $h\to\infty$ the fit becomes the constant $\bar Y$.</p>
<div class="def" data-name="Linear smoother; effective degrees of freedom">
<p>An estimator is a <em>linear smoother</em> if $\hat r(x) = \sum_i\ell_i(x)Y_i$ with weights not depending on the $Y$'s. Stacking fitted values, $\hat{\mathbf r} = L\by$ with $L_{ij} = \ell_j(x_i)$. The <em>effective degrees of freedom</em> is $\nu = \tr L$, which equals the number of parameters when $L$ is an OLS hat matrix.</p>
</div>
<p>Kernel smoothers, <a href="#/t/splines">smoothing splines</a>, <a href="#/t/ridge-regression">ridge regression</a> and <a href="#/t/ols-geometry">OLS</a> are all linear smoothers, which is why one theory of bias, variance and cross-validation covers them all.</p>`,
      },
      {
        title: 'Bias of Nadaraya–Watson: design bias and boundary bias',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Asymptotic bias and variance of Nadaraya–Watson">
<p>Let the $x_i$ be IID with density $g$, continuously differentiable and positive at an interior point $x$; let $r$ be twice continuously differentiable, $\Var(Y\mid X=x) = \sigma^2(x)$ continuous, $h\to0$, $nh\to\infty$. Then, conditionally on the design and to leading order,
$$\E\big[\hat r_{\mathrm{NW}}(x)\big] - r(x) \approx h^2\sigma_K^2\Big(\frac{r''(x)}{2} + \frac{r'(x)g'(x)}{g(x)}\Big),\qquad \Var\big[\hat r_{\mathrm{NW}}(x)\big]\approx\frac{\sigma^2(x)R(K)}{nhg(x)}.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Write $\hat r = \hat m/\hat g$ with $\hat m(x) = \frac1n\sum K_h(x-x_i)Y_i$. By the KDE bias calculation applied to $m = rg$ and to $g$,
$$\E\hat m(x)\approx (rg)(x) + \tfrac{h^2}{2}\sigma_K^2(rg)''(x),\qquad \E\hat g(x)\approx g(x) + \tfrac{h^2}{2}\sigma_K^2g''(x).$$
Linearizing the ratio, $\E\hat r\approx r + \frac{h^2\sigma_K^2}{2}\cdot\frac{(rg)'' - rg''}{g}$, and $(rg)'' - rg'' = r''g + 2r'g'$, giving the bias. For the variance, conditionally on the design $\Var\hat m(x) = \frac{1}{n^2}\sum K_h(x-x_i)^2\sigma^2(x_i)\approx\frac{\sigma^2(x)g(x)R(K)}{nh}$, and dividing by $\hat g(x)^2\approx g(x)^2$ gives the result.</p>
</details>
<p>The term $h^2\sigma_K^2r'g'/g$ is the <strong>design bias</strong>. Where the design density is increasing ($g'>0$) there are more points to the right of $x$ than to the left; if $r$ is also increasing, those extra points have larger responses, so the local average overshoots. The bias depends on where the <em>design</em> happens to put points, not only on the shape of $r$ — even a perfectly straight line $r(x) = a + bx$ is estimated with bias.</p>
<div class="ex" data-name="Boundary bias">
<p>At the edge of the design, say $x=0$ with all $x_i\ge0$, the window is one-sided. For $r(x_i)\approx r(0) + r'(0)x_i$ and uniform design,
$$\E\hat r_{\mathrm{NW}}(0)\approx r(0) + r'(0)\,h\,\frac{\int_0^\infty uK(u)\d u}{\int_0^\infty K(u)\d u},$$
an $O(h)$ bias — an order of magnitude worse than the interior $O(h^2)$. For the Gaussian kernel the factor is $2\phi(0)\approx0.80$, so a slope of 1 and $h = 0.1$ give a bias near $0.08$ at the edge. Every NW fit of a trend flattens at the ends.</p>
</div>`,
      },
      {
        title: 'Local linear regression fixes both',
        html: R`
<p>Instead of a weighted constant, fit a weighted line centred at $x$:
$$(\hat a,\hat b) = \argmin_{a,b}\sum_{i=1}^nK_h(x-x_i)\big(Y_i - a - b(x_i-x)\big)^2,\qquad\hat r_{\mathrm{LL}}(x) = \hat a.$$
Solving the $2\times2$ weighted normal equations with $S_j(x) = \sum_iK_h(x-x_i)(x_i-x)^j$ gives the explicit weights
$$\hat r_{\mathrm{LL}}(x) = \sum_i\ell_i(x)Y_i,\qquad \ell_i(x) = \frac{K_h(x-x_i)\{S_2(x) - (x_i-x)S_1(x)\}}{S_0(x)S_2(x) - S_1(x)^2}.$$</p>
<div class="thm" data-kind="Lemma" data-name="Local linear reproduces lines">
<p>The local linear weights satisfy $\sum_i\ell_i(x) = 1$ and $\sum_i\ell_i(x)(x_i - x) = 0$ for every $x$ and every design. Consequently, if $r$ is linear then $\E\hat r_{\mathrm{LL}}(x) = r(x)$ exactly.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\sum_i\ell_i = (S_0S_2 - S_1S_1)/(S_0S_2-S_1^2) = 1$ and $\sum_i\ell_i(x_i-x) = (S_1S_2 - S_2S_1)/(S_0S_2-S_1^2) = 0$. Then for $r(u) = r(x) + r'(x)(u-x)$, $\sum\ell_ir(x_i) = r(x)\cdot1 + r'(x)\cdot0$. (Alternatively: weighted least squares fits a line exactly when the data lie on one.)</p>
</details>
<p>Nadaraya–Watson weights satisfy only the first condition. The second is exactly what kills the design and boundary terms: expanding $r(x_i) = r(x) + r'(x)(x_i-x) + \frac12r''(x)(x_i-x)^2+\dots$, the bias of local linear is $\frac12r''(x)\sum_i\ell_i(x)(x_i-x)^2+\dots$, with no first-derivative contribution at all.</p>
<div class="thm" data-kind="Theorem" data-name="Local linear asymptotics (Fan, 1992, 1993)">
<p>Under the conditions above, at interior points
$$\Bias\big[\hat r_{\mathrm{LL}}(x)\big]\approx\frac{h^2}{2}\sigma_K^2r''(x),\qquad \Var\big[\hat r_{\mathrm{LL}}(x)\big]\approx\frac{\sigma^2(x)R(K)}{nhg(x)},$$
the same variance as NW but without the design-bias term. At a boundary point the bias remains $O(h^2)$ (with a different constant) and the variance $O(1/(nh))$ — the boundary correction is automatic. With the Epanechnikov kernel, local linear regression is asymptotically minimax-efficient among <em>all</em> linear smoothers (Fan, 1993).</p>
</div>
<p>Near a boundary the local linear weights become <em>negative</em> for points far from the edge: the fit extrapolates the local trend, and extrapolation needs negative weights. That is how it removes the $O(h)$ bias, and it is also why its variance near the boundary is somewhat larger. The lab lets you click on any $x_0$ to see both weight functions.</p>`,
        lab: 'kernel-regression-boundary',
      },
      {
        title: 'Local polynomials of higher degree',
        html: R`
<p>Fitting a degree-$p$ polynomial locally generalizes both: $p=0$ is NW, $p=1$ local linear. Then $\hat r^{(\nu)}(x) = \nu!\,\hat\beta_\nu$ estimates the $\nu$-th derivative as well. Fan & Gijbels' rule is to take $p - \nu$ odd. For estimating $r$ itself ($\nu=0$) the degrees pair up: $p=0$ and $p=1$ have the same asymptotic variance, but $p=1$ removes the design-bias term and the boundary bias; likewise $p=2$ and $p=3$ share a larger variance constant and an $O(h^4)$ interior bias when $r$ has four derivatives, but the odd degree again has the simpler, design-free bias and better boundary behaviour. Moving up to an even degree costs variance and buys nothing the next odd degree would not.</p>
<div class="note" data-name="Equivalent kernels">
<p>At interior points, local polynomial fitting of degree $p$ behaves like a kernel smoother with an implied "equivalent kernel" $K^*_p$ that does not depend on the design. For $p=1$ it is $K$ itself (hence no design bias); for $p=2,3$ it is the fourth-order kernel $K^*(u) = \frac{\mu_4 - \mu_2u^2}{\mu_4-\mu_2^2}K(u)$ with $\mu_j = \int u^jK$ — negative in its tails. Local polynomial fitting is how one builds higher-order kernels that adapt automatically to the design and the boundary.</p>
</div>
<p>LOESS/LOWESS (Cleveland, 1979) is local polynomial regression with a nearest-neighbour bandwidth (the window always contains a fixed fraction of the data, so it widens in sparse regions) and tricube kernel $K(u) = (1-|u|^3)^3$, plus optional robustness iterations that downweight large residuals — an M-estimation idea from <a href="#/t/robust-statistics">robust statistics</a>.</p>`,
      },
      {
        title: 'Choosing the bandwidth: cross-validation and degrees of freedom',
        html: R`
<p>The risk-optimal bandwidth balances $h^4$ squared bias against $1/(nh)$ variance exactly as for densities, so $h^*\asymp n^{-1/5}$ and the mean squared error is $O(n^{-4/5})$. The constants involve $\int r''^2$ and $\sigma^2$, so in practice $h$ is chosen by leave-one-out cross-validation.</p>
<div class="thm" data-kind="Proposition" data-name="Leave-one-out shortcut for kernel smoothers">
<p>For Nadaraya–Watson and local polynomial smoothers, deleting observation $i$ and refitting at $x_i$ gives
$$Y_i - \hat r_{(-i)}(x_i) = \frac{Y_i - \hat r(x_i)}{1 - L_{ii}},\qquad\text{so}\qquad \mathrm{CV}(h) = \frac1n\sum_{i=1}^n\Big(\frac{Y_i - \hat r(x_i)}{1 - L_{ii}}\Big)^2.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For NW, deleting $i$ renormalizes the remaining weights: $\hat r_{(-i)}(x_i) = \sum_{j\ne i}L_{ij}Y_j/(1-L_{ii}) = \{\hat r(x_i) - L_{ii}Y_i\}/(1-L_{ii})$. Then $Y_i - \hat r_{(-i)}(x_i) = \{Y_i(1-L_{ii}) - \hat r(x_i) + L_{ii}Y_i\}/(1-L_{ii}) = \{Y_i - \hat r(x_i)\}/(1-L_{ii})$. For local polynomials, $\hat r(x_i)$ is a weighted least-squares fit at the fixed target $x_i$, and the standard case-deletion identity for weighted least squares gives the same formula with $L_{ii}$ the leverage of point $i$ in that fit.</p>
</details>
<p><strong>Generalized cross-validation</strong> replaces each $L_{ii}$ by the average $\nu/n$, $\mathrm{GCV}(h) = \frac1n\sum(Y_i-\hat r(x_i))^2/(1-\nu/n)^2$, which is smoother in $h$ and invariant to rotations of the problem. Both criteria are discussed generally in <a href="#/t/cross-validation">cross-validation</a> and <a href="#/t/model-selection">model selection</a>.</p>`,
        lab: 'kernel-regression-cv',
      },
      {
        title: 'Inference and pitfalls',
        html: R`
<p><strong>Variance estimation.</strong> Since $\E\|\by - L\by\|^2 = \sigma^2\tr\{(I-L)^\top(I-L)\} + \|(I-L)\mathbf r\|^2$ under homoscedasticity, an approximately unbiased estimator (when the bias is small) is
$$\hat\sigma^2 = \frac{\sum_i(Y_i - \hat r(x_i))^2}{n - 2\nu + \tilde\nu},\qquad\nu = \tr L,\;\tilde\nu = \tr(L^\top L).$$
Pointwise standard errors are then $\hat\sigma\|\ell(x)\|$, since $\Var\hat r(x) = \sigma^2\sum_i\ell_i(x)^2$.</p>
<div class="warn" data-name="Bands cover the smoothed curve, not the curve">
<p>The interval $\hat r(x)\pm2\hat\sigma\|\ell(x)\|$ is centred on $\hat r(x)$, whose mean is $\bar r(x) = \sum\ell_i(x)r(x_i)$, not $r(x)$. At the MSE-optimal bandwidth the bias is of the same order as the standard error, so these are confidence bands for $\bar r$, a smoothed version of $r$, and they systematically miss peaks and valleys. Remedies: undersmooth ($h = o(n^{-1/5})$) so bias is negligible, explicitly correct the bias, or use bootstrap bands with a pilot oversmoothed fit.</p>
</div>
<div class="warn" data-name="Other pitfalls">
<ul>
<li><strong>Sparse regions.</strong> Variance $\propto 1/(nhg(x))$; where $g$ is small, a fixed-$h$ fit becomes erratic, and NW's denominator can vanish. Nearest-neighbour bandwidths help.</li>
<li><strong>Heteroscedasticity</strong> leaves the point estimates fine but invalidates a constant-$\sigma$ band; estimate $\sigma^2(x)$ by smoothing squared residuals.</li>
<li><strong>Dimension.</strong> With $d$ covariates the rate is $n^{-4/(4+d)}$, as for densities; beyond $d\approx3$ use structure such as additive models (<a href="#/t/gam">GAMs</a>), backfitting each coordinate with a univariate smoother.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Data lie exactly on the line $Y = 1 + 2x$ (no noise), with design points crowded toward the right. Which of NW and local linear reproduces the line exactly at interior points?`,
        a: R`Only local linear, which satisfies $\sum\ell_i(x_i - x) = 0$ and so fits lines exactly. NW averages $1 + 2x_i$ with weights that put more mass on the right (where points are denser), so its fit is pulled above the line where $g'>0$ — the design bias $h^2\sigma_K^2r'g'/g$ with $r'=2$.` },
      { q: R`What is the order of the Nadaraya–Watson bias at a boundary point, and what feature of the local linear weights eliminates it?`,
        a: R`$O(h)$, from the first-order Taylor term $r'(0)\sum w_i x_i$, which no longer cancels for a one-sided window. Local linear weights satisfy $\sum\ell_i(x)(x_i-x)=0$ at every $x$, including boundaries; near an edge they achieve this by taking negative values far from the edge (extrapolating the local slope).` },
      { q: R`For a linear smoother, how is the leave-one-out residual related to the ordinary residual, and why does this make cross-validation cheap?`,
        a: R`$Y_i - \hat r_{(-i)}(x_i) = (Y_i - \hat r(x_i))/(1-L_{ii})$ for kernel and local polynomial smoothers. One fit gives all $n$ leave-one-out residuals using the diagonal of the smoother matrix, instead of $n$ refits.` },
      { q: R`Why do pointwise confidence intervals at the CV-optimal bandwidth tend to undercover at peaks of $r$?`,
        a: R`At the optimal $h$, squared bias and variance are of the same order, and the bias $\frac{h^2}{2}\sigma_K^2r''(x)$ is negative at peaks. The intervals are centred on $\E\hat r(x)$, which sits below $r(x)$ by an amount comparable to the standard error, so they systematically miss.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'splines',
    domain: 'nonparametric',
    order: 60,
    title: 'Splines & Smoothing Splines',
    subtitle: 'Piecewise cubics, the roughness penalty, and why the optimal curve is always a natural cubic spline',
    level: 3,
    sources: { aos: '20.5', cb: '—', other: 'Green & Silverman (1994); Wahba (1990); de Boor (1978); Silverman (1984); Hastie, Tibshirani & Friedman (2009), Ch. 5' },
    prereqs: ['kernel-regression', 'ridge-regression'],
    related: ['gam', 'gaussian-processes', 'kernel-methods', 'model-selection', 'orthogonal-series'],
    tags: ['basis expansion', 'roughness penalty', 'natural cubic spline', 'GCV', 'equivalent kernel'],
    summary: R`A spline is a piecewise polynomial glued smoothly at knots. Regression splines fix a modest number of knots and fit by least squares, turning nonparametric regression into an ordinary linear model in a spline basis. Smoothing splines instead put a knot at every data point and control complexity with a penalty: minimize $\sum(Y_i - f(x_i))^2 + \lambda\int f''^2$ over <em>all</em> twice-differentiable $f$. The remarkable theorem is that this infinite-dimensional problem has a finite-dimensional answer, a natural cubic spline, computable as a generalized ridge regression. The result is a linear smoother that shrinks wiggly components most, behaves like a kernel smoother with bandwidth $\propto(\lambda/n)^{1/4}$, and has its complexity summarized by effective degrees of freedom.`,
    survey: {
      task: R`Estimate a smooth regression function; flexible terms in GAMs`,
      assumptions: R`$Y_i = r(x_i)+\epsilon_i$; $r$ with square-integrable $r''$; independent errors for GCV`,
      output: R`Natural cubic spline $\hat f_\lambda$; smoother $S_\lambda$; $\mathrm{df}_\lambda = \tr S_\lambda$; GCV-chosen $\lambda$`,
      strengths: 'Exact variational optimality; no knot selection; $O(n)$ algorithms; Bayesian and RKHS interpretations',
      weaknesses: R`Global $\lambda$ cannot adapt to varying smoothness; linear extrapolation; multivariate versions costly`,
      complexity: R`$O(n)$ per $\lambda$ with banded (Reinsch / B-spline) algorithms`,
    },
    sections: [
      {
        title: 'Splines and regression splines',
        html: R`
<div class="def" data-name="Spline of degree M">
<p>Given knots $\xi_1 < \dots < \xi_K$, a spline of degree $M$ is a function that is a polynomial of degree $\le M$ on each interval between consecutive knots and has $M-1$ continuous derivatives everywhere. <strong>Cubic splines</strong> ($M=3$) are continuous with continuous first and second derivatives; the eye cannot see the knots.</p>
</div>
<p>Counting: $K+1$ cubic pieces have $4(K+1)$ coefficients; each knot imposes 3 continuity constraints, leaving $K+4$ free parameters. A basis is the <strong>truncated power basis</strong>
$$1,\;x,\;x^2,\;x^3,\;(x-\xi_1)_+^3,\;\dots,\;(x-\xi_K)_+^3.$$
Each $(x-\xi_k)_+^3$ adds a jump in the third derivative at $\xi_k$, which is exactly the freedom a cubic spline has at a knot. The truncated power basis is numerically terrible (nearly collinear columns); in practice one uses the <strong>B-spline</strong> basis, which spans the same space but whose functions are supported on only $M+2$ consecutive knots, giving banded, well-conditioned design matrices.</p>
<div class="def" data-name="Natural cubic spline">
<p>A cubic spline that is <em>linear</em> beyond the boundary knots $\xi_1$ and $\xi_K$. The four extra constraints (second and third derivatives vanish outside both ends) leave $K$ free parameters.</p>
</div>
<p>A <strong>regression spline</strong> fixes the knots (often at quantiles of $x$) and fits by OLS: with basis matrix $B$, $\hat\beta = (B^\top B)^{-1}B^\top\by$ and the hat matrix has trace equal to the number of basis functions. All the machinery of <a href="#/t/ols-geometry">linear models</a> applies — but the number and placement of knots is now the tuning parameter, and that choice is discrete and awkward. Natural splines are often preferred because ordinary cubic splines have high variance near the boundaries, where a cubic is free to swing.</p>`,
        lab: 'splines-basis',
      },
      {
        title: 'The smoothing spline problem',
        html: R`
<p>Avoid knot selection altogether: search over all smooth functions, and penalize roughness.</p>
<div class="def" data-name="Smoothing spline criterion">
<p>Given data with distinct $a < x_1 < \dots < x_n < b$ and $\lambda>0$, minimize over $f$ in the Sobolev space $W_2[a,b] = \{f: f, f'\text{ absolutely continuous},\ f''\in L^2\}$:
$$\mathrm{PRSS}_\lambda(f) = \sum_{i=1}^n\big(Y_i - f(x_i)\big)^2 + \lambda\int_a^b f''(t)^2\d t.$$</p>
</div>
<p>$\int f''^2$ is (approximately) the bending energy of a thin elastic beam; linear functions are free. The extremes: $\lambda\to0$ gives the interpolant with minimal curvature; $\lambda\to\infty$ forces $f''\equiv0$, giving the least-squares line.</p>
<div class="thm" data-kind="Theorem" data-name="Natural cubic spline optimality (Schoenberg, 1964; Reinsch, 1967)">
<p>For $n\ge2$ and any $\lambda>0$, $\mathrm{PRSS}_\lambda$ has a unique minimizer over $W_2[a,b]$, and it is a natural cubic spline with knots at $x_1,\dots,x_n$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $g\in W_2$ be arbitrary and let $f$ be the natural cubic spline interpolating $g$ at the knots, $f(x_i) = g(x_i)$ (it exists and is unique for $n\ge2$). Then the residual sums of squares of $f$ and $g$ are equal, so it suffices to show $\int f''^2\le\int g''^2$ with equality only if $g = f$.</p>
<p>Put $\eta = g - f$, so $\eta(x_i) = 0$ for all $i$. Integrate by parts on $[x_1,x_n]$:
$$\int_{x_1}^{x_n}f''\eta'' = \big[f''\eta'\big]_{x_1}^{x_n} - \int_{x_1}^{x_n}f'''\eta' = 0 - \sum_{i=1}^{n-1}f'''(x_i^+)\int_{x_i}^{x_{i+1}}\eta' = -\sum_{i}f'''(x_i^+)\{\eta(x_{i+1}) - \eta(x_i)\} = 0.$$
The boundary term vanishes because $f''(x_1) = f''(x_n) = 0$ (natural); $f'''$ is constant on each interval because $f$ is cubic there; and $\eta$ vanishes at every knot. Since $f$ is linear outside $[x_1,x_n]$, $f''=0$ there too, so $\int_a^bf''\eta'' = 0$ and
$$\int_a^b g''^2 = \int_a^b(f''+\eta'')^2 = \int_a^bf''^2 + \int_a^b\eta''^2\ \ge\ \int_a^bf''^2.$$
Equality forces $\eta''\equiv0$, so $\eta$ is linear; vanishing at $n\ge2$ points makes it zero. Hence any $g$ that is not a natural cubic spline with these knots is strictly beaten by one that is. Uniqueness of the minimizer within the spline space follows from strict convexity, shown in the next section.</p>
</details>
<p>This is the first and simplest instance of a <strong>representer theorem</strong>: a penalized problem over an infinite-dimensional function space has its solution in a finite-dimensional span determined by the data. The general version, for any reproducing kernel Hilbert space, is in <a href="#/t/kernel-methods">kernel methods</a>; the smoothing spline is the case where the penalty is the squared RKHS seminorm $\int f''^2$, whose null space is the linear functions.</p>`,
      },
      {
        title: 'Computation: a generalized ridge regression',
        html: R`
<p>Restricting to natural cubic splines with knots at the $x_i$, write $f(x) = \sum_{j=1}^nN_j(x)\theta_j$ for a basis $N_1,\dots,N_n$ of that $n$-dimensional space. With $\mathbf N_{ij} = N_j(x_i)$ and $\Omega_{jk} = \int N_j''N_k''$,
$$\mathrm{PRSS}_\lambda = (\by - \mathbf N\theta)^\top(\by - \mathbf N\theta) + \lambda\theta^\top\Omega\theta,\qquad\hat\theta = (\mathbf N^\top\mathbf N + \lambda\Omega)^{-1}\mathbf N^\top\by,$$
exactly a <a href="#/t/ridge-regression">ridge regression</a> with a non-identity penalty matrix. $\mathbf N^\top\mathbf N$ is positive definite (a natural spline vanishing at all $n$ knots is zero) and $\Omega$ is positive semi-definite, so the criterion is strictly convex and the minimizer unique.</p>
<p>A cleaner parametrization uses the fitted values themselves, $\mathbf f = (f(x_1),\dots,f(x_n))$: then $\int f''^2 = \mathbf f^\top\mathbf K\mathbf f$ for a fixed positive semi-definite matrix $\mathbf K = Q R^{-1}Q^\top$ built from the knot spacings ($Q$ is $n\times(n-2)$ and $R$ is $(n-2)\times(n-2)$ tridiagonal; Green & Silverman, 1994, §2.1). The fit is
$$\hat{\mathbf f} = S_\lambda\by,\qquad S_\lambda = (I + \lambda\mathbf K)^{-1}\quad\text{(Reinsch form)}.$$
Banded structure gives $O(n)$ algorithms.</p>
<div class="thm" data-kind="Proposition" data-name="Demmler–Reinsch eigenstructure">
<p>Let $\mathbf K = \sum_k d_k\mathbf u_k\mathbf u_k^\top$ with $0 = d_1 = d_2 < d_3\le\dots\le d_n$. Then
$$S_\lambda = \sum_{k=1}^n\frac{1}{1+\lambda d_k}\mathbf u_k\mathbf u_k^\top,\qquad\mathrm{df}_\lambda = \tr S_\lambda = \sum_{k=1}^n\frac{1}{1+\lambda d_k}\in(2,n).$$
The eigenvectors do not depend on $\lambda$; $\mathbf u_1,\mathbf u_2$ span the linear functions of $x$, and $\mathbf u_k$ oscillates more as $k$ grows, with $d_k$ growing roughly like $k^4$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$I + \lambda\mathbf K$ shares the eigenvectors of $\mathbf K$ with eigenvalues $1+\lambda d_k$, so its inverse has eigenvalues $1/(1+\lambda d_k)$. The null space of $\mathbf K$ is $\{\mathbf f: \int f''^2 = 0\}$ for the interpolating natural spline, i.e. the values of linear functions — a 2-dimensional space. The trace is the sum of the eigenvalues; it tends to $2$ as $\lambda\to\infty$ and to $n$ as $\lambda\to0$.</p>
</details>
<p>So a smoothing spline is a <strong>shrinker</strong>, not a projector: it decomposes $\by$ in the Demmler–Reinsch basis, keeps the linear part intact, and shrinks the coefficient on the $k$-th component by $1/(1+\lambda d_k)$ — gently for smooth components, almost to zero for rough ones. Contrast a regression spline, whose hat matrix has eigenvalues exactly 0 or 1. The lab shows both the shrinkage factors and the equivalent kernel.</p>`,
        lab: 'splines-smoothing',
      },
      {
        title: 'The equivalent kernel',
        html: R`
<p>Row $i$ of $S_\lambda$ gives the weights the smoothing spline places on each $Y_j$ to produce $\hat f(x_i)$. Silverman (1984) showed that for large $n$, away from the boundary, these weights are approximately those of a kernel smoother with a design-dependent bandwidth.</p>
<div class="thm" data-kind="Theorem" data-name="Equivalent kernel (Silverman, 1984)">
<p>Let the design have a smooth positive density $g$. For $x$ in the interior, as $n\to\infty$ and $\lambda\to0$ suitably,
$$\hat f(x)\approx\frac{1}{n}\sum_j\frac{1}{g(x)\,h(x)}\,\kappa\Big(\frac{x - x_j}{h(x)}\Big)Y_j,\qquad h(x) = \Big(\frac{\lambda}{ng(x)}\Big)^{1/4},$$
with $\kappa(u) = \frac12e^{-|u|/\sqrt2}\sin\big(|u|/\sqrt2 + \pi/4\big)$.</p>
</div>
<details class="proof"><summary>Heuristic</summary>
<p>For a uniform design on a long interval, $\sum_i(Y_i - f(x_i))^2\approx n\int(y(t)-f(t))^2\d t$, and the criterion becomes $n\int(y-f)^2 + \lambda\int f''^2$. In the Fourier domain this decouples by frequency $\omega$: $\hat f(\omega) = \hat y(\omega)/\{1 + (\lambda/n)\omega^4\}$. So the smoother is a convolution whose transfer function is $1/(1+h^4\omega^4)$ with $h = (\lambda/n)^{1/4}$, and the inverse Fourier transform of $1/(1+\omega^4)$ is exactly $\kappa$. For a nonuniform design, a locally uniform approximation replaces $n$ by $ng(x)$.</p>
</details>
<p>Consequences worth knowing:</p>
<ul>
<li>$\kappa$ is symmetric, integrates to 1, and has $\int u^2\kappa = 0$ (its transfer function $1/(1+\omega^4)$ has zero second derivative at 0): it is a <em>fourth-order</em> kernel with small negative side lobes.</li>
<li>The bandwidth adapts to the design, but only weakly: $h(x)\propto g(x)^{-1/4}$, between a fixed bandwidth (exponent 0) and a nearest-neighbour bandwidth (exponent $-1$).</li>
<li>The effective bandwidth scales like $\lambda^{1/4}$: to halve it you must divide $\lambda$ by 16. That is why the lab's $\lambda$ slider is logarithmic.</li>
</ul>`,
      },
      {
        title: 'Choosing λ',
        html: R`
<p>Because $\hat{\mathbf f} = S_\lambda\by$ is a linear smoother, the leave-one-out identity applies exactly: $\mathrm{CV}(\lambda) = \frac1n\sum_i\{(Y_i - \hat f_\lambda(x_i))/(1-(S_\lambda)_{ii})\}^2$. More common is <strong>generalized cross-validation</strong> (Craven & Wahba, 1979):
$$\mathrm{GCV}(\lambda) = \frac{\frac1n\sum_i(Y_i - \hat f_\lambda(x_i))^2}{\big(1 - \mathrm{df}_\lambda/n\big)^2}.$$
It is invariant under orthogonal transformations of the data and has good asymptotic optimality properties. It is common to parametrize the fit by $\mathrm{df}_\lambda$ rather than $\lambda$, since df is interpretable and comparable across data sets and across smoothers ("a smoother with 5 df").</p>
<div class="note" data-name="Two other interpretations">
<p><strong>Bayesian</strong> (Wahba, 1978): the smoothing spline is the limit of the posterior mean of $f$ under a Gaussian-process prior consisting of a once-integrated Brownian motion plus a diffuse (flat) prior on linear functions, with noise-to-signal ratio $\lambda$. Posterior standard deviations give Bayesian credible bands with good frequentist coverage on average across $x$. See <a href="#/t/gaussian-processes">Gaussian processes</a>.</p>
<p><strong>Mixed model</strong>: writing the penalized coefficients as random effects, $\lambda = \sigma^2_\epsilon/\sigma^2_u$, so $\lambda$ can be estimated by REML. This is how modern GAM software (e.g. mgcv) often chooses smoothing parameters, and it tends to be more stable than GCV, which occasionally undersmooths badly.</p>
</div>
<p>With $\lambda$ chosen well the mean squared error is $O(n^{-4/5})$ when $\int r''^2<\infty$ — the same rate as second-order kernel methods, consistent with the bandwidth relation $h\propto(\lambda/n)^{1/4}$.</p>`,
      },
      {
        title: 'Extensions and pitfalls',
        html: R`
<ul>
<li><strong>Penalized regression splines (P-splines).</strong> Use a moderate number of B-spline basis functions (say 20–40) with a difference penalty on adjacent coefficients (Eilers & Marx, 1996). Nearly identical fits to a full smoothing spline at a fraction of the cost; the workhorse of GAM software.</li>
<li><strong>Multivariate.</strong> Thin-plate splines minimize $\sum(Y_i-f(\bx_i))^2 + \lambda J(f)$ with a rotation-invariant roughness functional; in $\R^2$ the solution involves radial basis functions $\|\bx - \bx_i\|^2\log\|\bx-\bx_i\|$. Tensor-product splines handle interactions with different scales. Additive models fit one spline per coordinate (<a href="#/t/gam">GAMs</a>).</li>
<li><strong>Generalized responses.</strong> Replace the residual sum of squares by a negative log-likelihood (binomial, Poisson) and solve by penalized IRLS.</li>
</ul>
<div class="warn" data-name="Pitfalls">
<ul>
<li><strong>Extrapolation.</strong> A natural spline is linear beyond the data. That is better than a cubic's wild swings, but still pure assumption.</li>
<li><strong>Spatially varying smoothness.</strong> One $\lambda$ must serve the whole curve: a function that is flat in one region and rapidly oscillating in another will be oversmoothed in one place or undersmoothed in the other. Adaptive penalties or <a href="#/t/orthogonal-series">wavelets</a> are better suited.</li>
<li><strong>Correlated errors.</strong> GCV assumes independent errors; positive autocorrelation makes the noise look like signal and GCV chooses $\lambda$ far too small (nearly interpolating the noise).</li>
<li><strong>Ties in $x$.</strong> Repeated $x$ values are handled by collapsing to unique values with weights equal to the multiplicities.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`How many free parameters does a cubic spline with $K$ interior knots have, and how many does a natural cubic spline with $K$ knots have? Why the difference?`,
        a: R`$K+4$ for a cubic spline ($4(K+1)$ coefficients minus $3K$ continuity constraints). A natural cubic spline must be linear beyond the boundary knots: the two outer pieces lose their quadratic and cubic coefficients, removing 4, leaving $K$.` },
      { q: R`In the proof of smoothing-spline optimality, which three facts make $\int f''\eta'' = 0$?`,
        a: R`(i) $f''$ vanishes at the end knots (natural spline), killing the boundary term of the integration by parts; (ii) $f'''$ is constant between knots (cubic pieces), so the remaining integral splits into $\sum_ic_i\{\eta(x_{i+1}) - \eta(x_i)\}$; (iii) $\eta = g - f$ vanishes at every knot, since $f$ interpolates $g$.` },
      { q: R`What are the limits of $\mathrm{df}_\lambda$ as $\lambda\to0$ and $\lambda\to\infty$, and what fits do they correspond to?`,
        a: R`$\lambda\to0$: $\mathrm{df}\to n$, the natural spline interpolant of the data. $\lambda\to\infty$: $\mathrm{df}\to2$, the least-squares straight line (the null space of the penalty is the linear functions, which are never shrunk).` },
      { q: R`By what factor must $\lambda$ change to halve the equivalent-kernel bandwidth of a smoothing spline?`,
        a: R`$h\propto(\lambda/n)^{1/4}$, so $\lambda$ must be divided by $2^4 = 16$.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'orthogonal-series',
    domain: 'nonparametric',
    order: 70,
    title: 'Orthogonal Series & Wavelets',
    subtitle: 'Estimating coefficients instead of curves: truncation, shrinkage and wavelet thresholding',
    level: 3,
    sources: { aos: '21', cb: '—', other: 'Efromovich (1999); Donoho & Johnstone (1994, 1995); Johnstone (2019), Gaussian Estimation; Tsybakov (2009), Ch. 1, 3' },
    prereqs: ['kernel-density'],
    related: ['splines', 'kernel-regression', 'lasso', 'decision-theory'],
    tags: ['basis expansion', 'sequence model', 'thresholding', 'sparsity', 'adaptivity'],
    summary: R`Expand the unknown function in an orthonormal basis, $f = \sum_j\beta_j\phi_j$, and the estimation problem becomes one of estimating a sequence of coefficients, each an ordinary mean. Keeping the first $J$ estimated coefficients trades variance $\approx J/n$ against the tail $\sum_{j>J}\beta_j^2$, recovering the $n^{-4/5}$ rate of kernels for twice-smooth functions. The coefficient view pays off with <strong>wavelets</strong>: bases localized in both position and scale, in which functions with jumps or spikes are sparse. Thresholding the noisy wavelet coefficients — keep the large ones, kill the small — gives estimators that adapt automatically to unknown and spatially varying smoothness, something no linear smoother can do.`,
    survey: {
      task: R`Estimate a density or regression function via its basis coefficients`,
      assumptions: R`$f\in L^2[0,1]$; coefficients decay (Sobolev) or are sparse (Besov); for wavelets, equispaced design with $n=2^J$`,
      output: R`$\hat f = \sum_j\hat\beta_j\phi_j$ with truncated, shrunk or thresholded $\hat\beta_j$`,
      strengths: R`Clean risk theory; wavelet thresholding is spatially adaptive and near-minimax over wide classes`,
      weaknesses: R`Fourier: Gibbs ringing, boundary effects; wavelets: dyadic, shift-variant, blocky (Haar)`,
      complexity: R`$O(nJ)$ for $J$ series terms; $O(n)$ for the fast wavelet transform`,
    },
    sections: [
      {
        title: 'Functions as coefficient sequences',
        html: R`
<p>Let $\{\phi_0,\phi_1,\dots\}$ be an orthonormal basis of $L^2[0,1]$. Every square-integrable $f$ has the expansion $f = \sum_j\beta_j\phi_j$ with $\beta_j = \int f\phi_j$, and by Parseval $\int(f-g)^2 = \sum_j(\beta_j - \gamma_j)^2$. Estimating $f$ in integrated squared error is <em>exactly</em> estimating the coefficient sequence in squared $\ell^2$ error.</p>
<div class="def" data-name="Cosine basis">
<p>$\phi_0(x) = 1$, $\phi_j(x) = \sqrt2\cos(\pi jx)$ for $j\ge1$. Unlike the full Fourier basis it does not force $f(0) = f(1)$, so it avoids wrap-around artefacts at the boundary.</p>
</div>
<p>Smoothness becomes coefficient decay. Integrating by parts, if $f$ has $p$ square-integrable derivatives (with boundary conditions appropriate to the basis), then $\sum_j(\pi j)^{2p}\beta_j^2 = \int(f^{(p)})^2<\infty$. This motivates the <strong>Sobolev ellipsoid</strong>
$$\Theta(p,c) = \Big\{\beta: \sum_{j\ge1}j^{2p}\beta_j^2\le c^2\Big\}.$$</p>
<div class="note">
<p>The basis-expansion view unifies the smoothers you have met: a <a href="#/t/splines">smoothing spline</a> is (nearly) a Fourier-type estimator that multiplies the $j$-th coefficient by $1/(1+\lambda d_j)$; a kernel estimator multiplies the $\omega$-th Fourier coefficient by the kernel's transfer function $\hat K(h\omega)$. Every linear smoother is a set of shrinkage weights in some basis. Truncation uses weights in $\{0,1\}$.</p>
</div>`,
      },
      {
        title: 'Projection density estimation',
        html: R`
<p>For a density on $[0,1]$, $\beta_j = \int\phi_jf = \E\,\phi_j(X)$, so the obvious estimate is a sample mean:
$$\hat\beta_j = \frac1n\sum_{i=1}^n\phi_j(X_i),\qquad\hat f_J(x) = 1 + \sum_{j=1}^J\hat\beta_j\phi_j(x).$$
($\beta_0 = 1$ for a density.) Each $\hat\beta_j$ is unbiased with $\Var\hat\beta_j = \sigma_j^2/n$, where $\sigma_j^2 = \Var\phi_j(X)$. For the cosine basis, $2\cos^2(\pi jx) = 1 + \cos(2\pi jx)$ gives $\E\phi_j^2(X) = 1 + \beta_{2j}/\sqrt2$, so $\sigma_j^2 = 1 + \beta_{2j}/\sqrt2 - \beta_j^2$, which is close to 1 for large $j$.</p>
<div class="thm" data-kind="Theorem" data-name="Risk of the projection estimator">
<p>$$R(J) = \E\int(\hat f_J - f)^2 = \frac1n\sum_{j=1}^J\sigma_j^2 + \sum_{j>J}\beta_j^2.$$
If $\beta\in\Theta(p,c)$ and $\sigma_j^2\le C$, then $R(J)\le CJ/n + c^2J^{-2p}$; choosing $J\asymp n^{1/(2p+1)}$ gives $R = O(n^{-2p/(2p+1)})$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By Parseval, $\int(\hat f_J - f)^2 = \sum_{j\le J}(\hat\beta_j - \beta_j)^2 + \sum_{j>J}\beta_j^2$. Taking expectations, the first sum is a sum of variances because each $\hat\beta_j$ is unbiased. For the bound, $\sum_{j>J}\beta_j^2\le J^{-2p}\sum_{j>J}j^{2p}\beta_j^2\le c^2J^{-2p}$. Minimizing $CJ/n + c^2J^{-2p}$ over $J$ gives $J^{2p+1}\asymp n$, and both terms are then of order $n^{-2p/(2p+1)}$.</p>
</details>
<p>For $p=2$ this is $n^{-4/5}$, matching the <a href="#/t/kernel-density">kernel estimator</a> for twice-differentiable densities, and the rate is minimax over $\Theta(p,c)$. Here the number of terms $J$ plays the role of $1/h$: variance grows linearly in $J$, squared bias falls polynomially.</p>
<div class="ex" data-name="Estimating the risk to choose J">
<p>Since $\E\hat\beta_j^2 = \beta_j^2 + \sigma_j^2/n$, the quantity $\hat\beta_j^2 - \hat\sigma_j^2/n$ is (nearly) unbiased for $\beta_j^2$, with $\hat\sigma_j^2$ the sample variance of $\phi_j(X_i)$. Hence an estimate of $R(J)$, up to a constant not depending on $J$, is
$$\hat R(J) = \sum_{j=1}^J\frac{\hat\sigma_j^2}{n} + \sum_{j=J+1}^{J_{\max}}\Big(\hat\beta_j^2 - \frac{\hat\sigma_j^2}{n}\Big)_+,$$
and we choose $\hat J = \argmin_J\hat R(J)$. Read it as: include coefficient $j$ only if its estimated squared size exceeds its variance — a signal-to-noise test, coefficient by coefficient.</p>
</div>
<div class="warn" data-name="Series estimates are not densities">
<p>$\hat f_J$ integrates to 1 but can be negative where $f$ is small, and truncated cosine series ring near discontinuities (Gibbs phenomenon: an overshoot of about 9% of the jump that does not shrink as $J$ grows, it only narrows). Clip negatives and renormalize; for jumps, prefer wavelets.</p>
</div>`,
        lab: 'orthogonal-series-cosine',
      },
      {
        title: 'Regression and the Gaussian sequence model',
        html: R`
<p>For regression on an equispaced design, $Y_i = r(x_i) + \sigma\epsilon_i$ with $\epsilon_i\iid\N(0,1)$, take empirical coefficients $Z_j = \frac1n\sum_iY_i\phi_j(x_i)$. If the basis is orthonormal on the design points (the discrete cosine basis at $x_i = (i-\frac12)/n$ is exactly orthonormal there), then
$$Z_j = \beta_j + \frac{\sigma}{\sqrt n}\,\epsilon'_j,\qquad\epsilon'_j\iid\N(0,1),\qquad j = 0,\dots,n-1.$$
This is the <strong>Gaussian sequence model</strong>: $n$ independent normal means, each observed once with noise level $\sigma/\sqrt n$. Nonparametric regression <em>is</em> the problem of estimating many normal means — the setting of <a href="#/t/decision-theory">Stein's paradox</a> — and the different smoothers are different estimators of a mean vector:</p>
<table>
<thead><tr><th>Estimator</th><th>Rule $\hat\beta_j = \eta_j(Z_j)$</th><th>Nature</th></tr></thead>
<tbody>
<tr><td>Projection / truncation</td><td>$Z_j\ind(j\le J)$</td><td>linear, keep-or-kill by index</td></tr>
<tr><td>Spline / ridge-type</td><td>$Z_j/(1+\lambda d_j)$</td><td>linear, smooth shrinkage by index</td></tr>
<tr><td>James–Stein (blockwise)</td><td>$\big(1 - \frac{(m-2)\sigma^2/n}{\|Z_B\|^2}\big)_+Z_j$ for $j$ in block $B$ of size $m$</td><td>nonlinear, data-driven shrinkage</td></tr>
<tr><td>Hard threshold</td><td>$Z_j\ind(|Z_j|>\lambda)$</td><td>nonlinear, keep-or-kill by size</td></tr>
<tr><td>Soft threshold</td><td>$\sgn(Z_j)(|Z_j|-\lambda)_+$</td><td>nonlinear, shrink-or-kill by size</td></tr>
</tbody>
</table>
<p>Pinsker (1980) showed that over a Sobolev ellipsoid the best <em>linear</em> estimator is asymptotically minimax among <em>all</em> estimators — linear smoothing is enough when the coefficients decay in a known order. The thresholding rules matter when they do not: when a few large coefficients are scattered unpredictably among many negligible ones.</p>`,
      },
      {
        title: 'Wavelets: localization in position and scale',
        html: R`
<p>A cosine basis function spreads over the entire interval, so a single jump in $f$ spreads energy over all frequencies: its coefficients decay only like $1/j$, and no truncation captures it efficiently. Wavelets are built to be local.</p>
<div class="def" data-name="Haar wavelets">
<p>Father wavelet (scaling function) $\varphi = \ind_{[0,1)}$; mother wavelet $\psi = \ind_{[0,1/2)} - \ind_{[1/2,1)}$. Dilations and translations
$$\psi_{jk}(x) = 2^{j/2}\psi(2^jx - k),\qquad j\ge0,\;k = 0,\dots,2^j-1,$$
together with $\varphi$ form an orthonormal basis of $L^2[0,1]$. The coefficient $\langle f,\psi_{jk}\rangle$ measures the change in local average of $f$ across the midpoint of the dyadic interval $[k2^{-j},(k+1)2^{-j})$.</p>
</div>
<p>The Haar system is the simplest example of a <strong>multiresolution analysis</strong>: $V_j$ = functions constant on dyadic intervals of length $2^{-j}$, with $V_{j+1} = V_j\oplus W_j$ and $W_j$ spanned by the $\psi_{jk}$. Daubechies (1988) constructed compactly supported orthonormal wavelets with any number of vanishing moments ($\int x^m\psi = 0$ for $mt M$) and corresponding smoothness, which give smooth reconstructions and make coefficients of smooth pieces of $f$ decay like $2^{-j(M+1/2)}$.</p>
<p>Two facts make wavelets powerful for estimation:</p>
<ul>
<li><strong>Sparsity.</strong> If $f$ is smooth except for a few jumps or spikes, then at level $j$ only $O(1)$ coefficients per singularity are large (those whose support straddles it); the rest are tiny. A function with $n$ samples is well represented by far fewer than $n$ coefficients, but <em>which</em> ones depends on $f$.</li>
<li><strong>Fast transform.</strong> On $n=2^J$ equispaced samples the discrete wavelet transform is an orthogonal $n\times n$ matrix computable in $O(n)$ operations (Mallat's pyramid algorithm). Orthogonality means white noise stays white: the sequence model holds exactly with noise level $\sigma$ per coefficient.</li>
</ul>`,
      },
      {
        title: 'Thresholding: adaptivity for free',
        html: R`
<p>In the sequence model $Z_j = \theta_j + \sigma\epsilon_j$, $j = 1,\dots,n$ (noise level $\sigma$ per coefficient after an orthogonal transform), the ideal "oracle" would keep $Z_j$ exactly when $|\theta_j|>\sigma$, with risk $\sum_j\min(\theta_j^2,\sigma^2)$. Thresholding imitates the oracle using only the data.</p>
<div class="thm" data-kind="Lemma" data-name="Maximum of Gaussian noise">
<p>If $\epsilon_1,\dots,\epsilon_n\iid\N(0,1)$, then $\P\big(\max_j|\epsilon_j| > \sqrt{2\log n}\big)\le\frac{1}{\sqrt{\pi\log n}}\to0.$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Mills' ratio gives $\P(|Z|>t)\le2\phi(t)/t$. With $t = \sqrt{2\log n}$, $\phi(t) = e^{-\log n}/\sqrt{2\pi} = 1/(n\sqrt{2\pi})$. The union bound gives $n\cdot2\phi(t)/t = 2/(\sqrt{2\pi}\sqrt{2\log n}) = 1/\sqrt{\pi\log n}$.</p>
</details>
<p>So the <strong>universal threshold</strong> $\lambda_n = \sigma\sqrt{2\log n}$ is, with high probability, above every pure-noise coefficient: thresholding at $\lambda_n$ produces a reconstruction that is noise-free where $f$ is smooth. Donoho and Johnstone (1994) proved that it nearly achieves the oracle:</p>
<div class="thm" data-kind="Theorem" data-name="Oracle inequality for soft thresholding (Donoho & Johnstone, 1994)">
<p>With $\hat\theta_j = \sgn(Z_j)(|Z_j|-\lambda_n)_+$ and $\lambda_n = \sigma\sqrt{2\log n}$,
$$\E\sum_{j=1}^n(\hat\theta_j - \theta_j)^2\le(2\log n + 1)\Big(\sigma^2 + \sum_{j=1}^n\min(\theta_j^2,\sigma^2)\Big).$$</p>
</div>
<p>The oracle risk is within a logarithmic factor, uniformly over all $\theta$ — no smoothness assumption at all. Translated back to function estimation, wavelet thresholding is simultaneously near-minimax (within $\log n$) over a huge range of Besov smoothness classes, including spatially inhomogeneous functions with jumps and spikes. For such classes (Besov $B^s_{p,q}$ with $p<2$), linear estimators — kernels, splines, truncated series with any choice of tuning — are rate-suboptimal (Donoho & Johnstone, 1998). This is the precise sense in which nonlinearity buys adaptivity.</p>
<div class="algo" data-name="Wavelet denoising (VisuShrink)">
<ol>
<li>Compute the discrete wavelet transform of $Y_1,\dots,Y_n$ ($n=2^J$).</li>
<li>Estimate $\sigma$ robustly from the finest-level details: $\hat\sigma = \mathrm{median}(|d_{J-1,k}|)/0.6745$ (most fine coefficients are pure noise; the MAD ignores the few that are not).</li>
<li>Soft- or hard-threshold all detail coefficients at $\hat\sigma\sqrt{2\log n}$; keep the coarse scaling coefficients.</li>
<li>Invert the transform.</li>
</ol>
</div>
<p>The universal threshold is conservative (it aims to kill <em>all</em> noise) and visibly oversmooths; SureShrink chooses a level-dependent threshold by minimizing Stein's unbiased risk estimate, and empirical-Bayes thresholds adapt to the sparsity of each level. Soft thresholding is exactly the <a href="#/t/lasso">lasso</a> with an orthogonal design, and hard thresholding is best-subset selection there.</p>`,
        lab: 'orthogonal-series-wavelet',
      },
      {
        title: 'Pitfalls and practicalities',
        html: R`
<div class="warn" data-name="Things that go wrong">
<ul>
<li><strong>Design.</strong> Fast wavelet methods assume equispaced $x_i$ and $n = 2^J$. For irregular designs, interpolate to a grid, pad, or use lifting schemes.</li>
<li><strong>Shift variance.</strong> Haar coefficients depend on where the jumps fall relative to the dyadic grid; the reconstruction shows pseudo-Gibbs artefacts and blockiness. Averaging over all circular shifts ("cycle spinning", Coifman & Donoho, 1995) removes most of it.</li>
<li><strong>Correlated noise.</strong> The universal threshold assumes white noise. With correlated errors the noise level differs by level; estimate $\sigma_j$ level by level.</li>
<li><strong>Boundary.</strong> Periodic wavelet transforms wrap the ends together; for non-periodic $f$ use boundary-corrected wavelets or reflection.</li>
<li><strong>Series density estimates</strong> can be negative and ring at jumps; their risk estimate $\hat R(J)$ is noisy when $n$ is small.</li>
</ul>
</div>
<p>The broad lesson outlives the specific bases: choose a representation in which the signal is <em>sparse</em>, then estimate by a rule that keeps large coefficients and discards small ones. That template reappears as the lasso, compressed sensing, and sparse dictionary learning.</p>`,
      },
    ],
    quiz: [
      { q: R`For the projection density estimator, why is the integrated variance approximately $J/n$ regardless of $f$?`,
        a: R`By Parseval the integrated variance is $\sum_{j\le J}\Var\hat\beta_j = \sum_{j\le J}\sigma_j^2/n$, and for the cosine basis $\sigma_j^2 = 1 + \beta_{2j}/\sqrt2 - \beta_j^2$, which tends to 1 as $j$ grows because the coefficients of an $L^2$ function decay. Each retained coefficient costs about $1/n$ of variance.` },
      { q: R`Why is $\sqrt{2\log n}$ (times the noise level) a natural threshold?`,
        a: R`It is the typical size of the maximum of $n$ independent standard normals; the probability that any pure-noise coefficient exceeds it tends to zero (at rate $1/\sqrt{\pi\log n}$). Thresholding there kills all noise with high probability while keeping coefficients that are clearly larger than the noise.` },
      { q: R`What does soft thresholding do to a large coefficient, and why might hard thresholding be preferred for it? Why might soft be preferred overall?`,
        a: R`Soft thresholding shrinks every surviving coefficient toward zero by $\lambda$, introducing bias $\lambda$ for large coefficients; hard thresholding leaves them untouched. Soft is continuous in the data (more stable, lower variance near the threshold) and admits the clean oracle inequality and Stein-unbiased risk estimation; hard is discontinuous and can be erratic for coefficients near $\lambda$.` },
      { q: R`Why can a kernel smoother not match wavelet thresholding on a signal that is flat with a few sharp spikes?`,
        a: R`A kernel smoother uses one bandwidth everywhere: small enough to resolve the spikes means noisy flat regions; large enough to smooth the flat regions blurs the spikes. In the wavelet basis this signal is sparse — a few large coefficients near the spikes — and thresholding keeps exactly those, effectively using a small bandwidth near spikes and a huge one elsewhere. Formally, linear estimators are rate-suboptimal over Besov classes with $p<2$.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'rank-tests',
    domain: 'nonparametric',
    order: 80,
    title: 'Rank Tests: Sign, Wilcoxon, Mann–Whitney & Kruskal–Wallis',
    subtitle: 'Exact distribution-free tests from ranks alone, and the surprisingly small price they pay',
    level: 2,
    sources: { aos: '—', cb: '—', other: 'Lehmann (1975), Nonparametrics; Hájek, Šidák & Sen (1999); Hodges & Lehmann (1956); Hollander, Wolfe & Chicken (2014)' },
    prereqs: ['hypothesis-testing'],
    related: ['permutation-tests', 'order-statistics', 'robust-statistics', 'empirical-cdf'],
    tags: ['distribution-free', 'ranks', 'Pitman efficiency', 'Hodges–Lehmann'],
    summary: R`Replace each observation by its rank and you throw away the scale of the data — and with it every dependence of the null distribution on the unknown $F$. Under the null hypothesis of exchangeability, the rank vector is uniform over all permutations, so rank statistics have exact, tabulable null distributions valid for every continuous $F$. The cost is astonishingly small: against normal shift alternatives the Wilcoxon test needs only about 5% more data than the optimal $t$-test (efficiency $3/\pi\approx0.955$), never more than 16% more for any distribution (Hodges–Lehmann bound $0.864$), and arbitrarily less for heavy-tailed data.`,
    survey: {
      task: 'Test location shifts / stochastic ordering; one-sample, paired, two-sample, $k$-sample',
      assumptions: R`Independent observations; continuous $F$ (ties handled by midranks); signed-rank also needs symmetry`,
      output: R`Statistic ($S$, $W^+$, $W$ or $U$, $H$); exact or normal-approximation p-value; Hodges–Lehmann estimate and CI`,
      strengths: R`Exact level for any continuous $F$; robust to outliers; ARE $\ge0.864$ vs $t$ for shifts`,
      weaknesses: R`Tests $\P(Xt Y)=1/2$, not means or medians in general; sensitive to unequal spreads; discrete null distribution`,
      complexity: R`$O(N\log N)$ to rank; exact null by DP in $O(mn\cdot\min(m,n))$`,
    },
    sections: [
      {
        title: 'Why ranks give distribution-free tests',
        html: R`
<p>Let $Z_1,\dots,Z_N$ be observations with no ties, and let $R_i$ be the rank of $Z_i$ among them. The rank vector $\mathbf R = (R_1,\dots,R_N)$ is a permutation of $\{1,\dots,N\}$.</p>
<div class="thm" data-kind="Theorem" data-name="Ranks are uniform under exchangeability">
<p>If $Z_1,\dots,Z_N$ are exchangeable with a continuous joint distribution (in particular, IID from any continuous $F$), then $\P(\mathbf R = \mathbf r) = 1/N!$ for every permutation $\mathbf r$. Hence any statistic $T(\mathbf R)$ has a null distribution that does not depend on $F$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Continuity makes ties a null event, so $\mathbf R$ is a.s. well defined. For any permutation $\pi$, exchangeability says $(Z_{\pi(1)},\dots,Z_{\pi(N)})$ has the same law as $(Z_1,\dots,Z_N)$. Permuting the observations permutes their ranks in the same way, so $\mathbf R\circ\pi$ has the same law as $\mathbf R$. Thus the distribution of $\mathbf R$ is invariant under composition with every permutation; since the permutation group acts transitively on itself, all $N!$ values are equally likely.</p>
</details>
<p>Rank tests are therefore a special case of <a href="#/t/permutation-tests">permutation tests</a> — but because they depend only on ranks, their null distribution can be computed once and tabulated, for every data set and every $F$. The probability integral transform shows why nothing is lost in the null: $F(Z_i)$ are IID uniform and have the same ranks as the $Z_i$.</p>`,
      },
      {
        title: 'One sample: sign and signed-rank tests',
        html: R`
<p>Test $H_0$: the median of $F$ is $\theta_0$ (shift so $\theta_0 = 0$).</p>
<div class="def" data-name="Sign test">
<p>$S = \#\{i: X_i>0\}$. Under $H_0$ with $F$ continuous, $S\sim\text{Binomial}(n,1/2)$ exactly.</p>
</div>
<p>The sign test assumes almost nothing, and it uses almost nothing: only which side of zero each point falls. Inverting it gives a distribution-free confidence interval for the median between two order statistics, $[X_{(k)},X_{(n-k+1)}]$, with $k$ from the binomial distribution.</p>
<div class="def" data-name="Wilcoxon signed-rank test">
<p>Rank $|X_1|,\dots,|X_n|$ and let $R_i^+$ be the rank of $|X_i|$. Then $W^+ = \sum_iR_i^+\ind(X_i>0)$, the sum of ranks belonging to positive observations.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Null distribution of the signed-rank statistic">
<p>If the $X_i$ are IID from a continuous distribution symmetric about 0, then $W^+\overset{d}{=}\sum_{k=1}^nk\,I_k$ with $I_k\iid\text{Bernoulli}(1/2)$. Hence
$$\E W^+ = \frac{n(n+1)}{4},\qquad\Var W^+ = \frac{n(n+1)(2n+1)}{24},$$
and $(W^+ - \E W^+)/\sqrt{\Var W^+}\dto\N(0,1)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For a continuous distribution symmetric about 0, $\sgn(X)$ and $|X|$ are independent and $\sgn(X)$ is $\pm1$ with probability $\frac12$ each. Hence the signs are IID fair coin flips independent of the vector $(|X_1|,\dots,|X_n|)$, and so independent of the ranks of the absolute values. Let $I_k$ be the indicator that the observation with the $k$-th smallest $|X|$ is positive; the $I_k$ are IID Bernoulli$(\frac12)$ and $W^+ = \sum_kkI_k$. Then $\E W^+ = \frac12\sum k = n(n+1)/4$ and $\Var W^+ = \frac14\sum k^2 = n(n+1)(2n+1)/24$. Asymptotic normality follows from the Lindeberg CLT (each term is bounded by $n$ while the standard deviation grows like $n^{3/2}$).</p>
</details>
<p>The signed-rank test uses the magnitudes too, so it is far more powerful than the sign test — but it needs <strong>symmetry</strong>. For a skewed $F$ with median 0, $W^+$ is not centred at $n(n+1)/4$; the test then rejects a true median hypothesis, because what it actually tests is whether the <em>pseudo-median</em> — the median of the distribution of $(X+X')/2$, estimated by the median of the Walsh averages $(X_i+X_j)/2$ — is zero; the two coincide only under symmetry. It is the natural test for <em>paired differences</em>, where symmetry under no treatment effect is plausible.</p>`,
      },
      {
        title: 'Two samples: Wilcoxon rank-sum and Mann–Whitney',
        html: R`
<p>Let $X_1,\dots,X_m\iid F$ and $Y_1,\dots,Y_n\iid G$, $N = m+n$; test $H_0: F = G$. Rank all $N$ observations together.</p>
<div class="def" data-name="Rank-sum and Mann–Whitney statistics">
<p>$W = \sum_{j=1}^nR(Y_j)$, the sum of the $Y$-ranks in the combined sample. Equivalently $U = \#\{(i,j): X_i < Y_j\} = W - n(n+1)/2$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Null moments of the rank-sum statistic">
<p>Under $H_0$ with continuous $F$, the set of $Y$-ranks is a uniformly random $n$-subset of $\{1,\dots,N\}$, and
$$\E W = \frac{n(N+1)}{2},\qquad\Var W = \frac{mn(N+1)}{12}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Uniformity follows from the previous theorem. Each $Y$-rank $R_j$ is marginally uniform on $\{1,\dots,N\}$, so $\E R_j = (N+1)/2$ and $\Var R_j = (N^2-1)/12$. For $j\ne k$, the pair is a uniform draw without replacement; since $\sum_{i=1}^N i$ is constant, $0 = \Var\big(\sum_{\text{all }N}R\big) = N\Var R_1 + N(N-1)\Cov(R_1,R_2)$, so $\Cov(R_j,R_k) = -(N+1)/12$. Then
$$\Var W = n\frac{N^2-1}{12} - n(n-1)\frac{N+1}{12} = \frac{n(N+1)}{12}\{(N-1)-(n-1)\} = \frac{mn(N+1)}{12}.$$</p>
</details>
<p>The exact null distribution of $U$ follows from a simple recursion: if $c_{m,n}(u)$ counts the arrangements with $U=u$, then conditioning on whether the largest observation is a $Y$ (which then exceeds all $m$ $X$'s) or an $X$ gives $c_{m,n}(u) = c_{m,n-1}(u-m) + c_{m-1,n}(u)$, with $\P(U = u) = c_{m,n}(u)/\binom{N}{n}$.</p>
<div class="note" data-name="What Mann–Whitney actually estimates">
<p>$U/(mn)$ is an unbiased estimator of $\theta = \P(Xt Y)$, and the test is consistent exactly against alternatives with $\theta\ne\frac12$. It is a test of "stochastic ordering" or of a location shift <em>when $F$ and $G$ differ only by a shift</em>. It is <strong>not</strong> a test of equal medians in general: two distributions with equal medians but different spreads or skewness can have $\theta\ne\frac12$, and two distributions with $\theta = \frac12$ can differ wildly. With unequal variances the test's size for "equal medians" can be badly off, just as the pooled $t$-test misbehaves in the Behrens–Fisher problem.</p>
</div>`,
        lab: 'rank-tests-null',
      },
      {
        title: 'Efficiency: 3/π and the Hodges–Lehmann bound',
        html: R`
<p>How much power does replacing data by ranks cost? Compare tests by <strong>Pitman asymptotic relative efficiency</strong>: the limiting ratio of sample sizes the two tests need to achieve the same power against the same sequence of alternatives $\theta_N = \delta/\sqrt N\to0$.</p>
<div class="thm" data-kind="Theorem" data-name="ARE of Wilcoxon versus t for shift alternatives">
<p>For two-sample shift alternatives $G(x) = F(x-\theta)$, with $F$ having density $f$ and variance $\sigma^2<\infty$,
$$\mathrm{ARE}(W,t) = 12\,\sigma^2\Big(\int f^2(x)\d x\Big)^2.$$
The same formula holds for the signed-rank test versus the one-sample $t$-test (symmetric $F$). The sign test has $\mathrm{ARE}(S,t) = 4\sigma^2f(0)^2$ for $F$ symmetric about 0.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>For a test statistic $T_N$ with $\E_\theta T_N\approx\mu(\theta)$ and null variance $\approx\tau^2/N$, asymptotically normal, the power against $\theta_N = \delta/\sqrt N$ is $\Phi(\delta\mu'(0)/\tau - z_\alpha)$. The <em>efficacy</em> $e = \mu'(0)/\tau$ governs power, and the ARE is $(e_1/e_2)^2$. Put $m = \lambda N$. For the $t$-test with $T = \bar Y - \bar X$: $\mu(\theta) = \theta$, $N\Var T = \sigma^2/\{\lambda(1-\lambda)\}$, so $e_t^2 = \lambda(1-\lambda)/\sigma^2$. For $T = U/(mn)$: $\mu(\theta) = \P(X< Y) = \int F(y+\theta)f(y)\d y$ when $Y = X'+\theta$, so $\mu'(0) = \int f^2$; the null variance is $(N+1)/(12mn)\approx1/\{12N\lambda(1-\lambda)\}$, so $e_W^2 = 12\lambda(1-\lambda)(\int f^2)^2$. The ratio is $12\sigma^2(\int f^2)^2$, free of $\lambda$.</p>
</details>
<table>
<thead><tr><th>$F$</th><th>Wilcoxon vs $t$</th><th>Sign vs $t$</th></tr></thead>
<tbody>
<tr><td>Normal</td><td>$3/\pi\approx0.955$</td><td>$2/\pi\approx0.637$</td></tr>
<tr><td>Uniform</td><td>$1$</td><td>$1/3$</td></tr>
<tr><td>Logistic</td><td>$\pi^2/9\approx1.097$</td><td>$\pi^2/12\approx0.822$</td></tr>
<tr><td>Laplace</td><td>$1.5$</td><td>$2$</td></tr>
<tr><td>$t_3$</td><td>$\approx1.90$</td><td>$\approx1.62$</td></tr>
<tr><td>Cauchy</td><td>$\infty$ ($t$-test invalid)</td><td>$\infty$</td></tr>
</tbody>
</table>
<p>For the normal: $\int\phi^2 = 1/(2\sqrt\pi)$, so $12\cdot\frac1{4\pi} = 3/\pi$. Note the sign test beats Wilcoxon for the Laplace, whose sharp peak at 0 makes "which side" unusually informative.</p>
<div class="thm" data-kind="Theorem" data-name="Hodges–Lehmann (1956) lower bound">
<p>For every $F$ with a density and finite variance, $\mathrm{ARE}(W,t)\ge108/125 = 0.864$, with equality for the parabolic density $f(x) = \frac{3}{20\sqrt5}(5-x^2)$ on $|x|\le\sqrt5$ (and its location-scale family).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$\sigma^2(\int f^2)^2$ is invariant under location and scale changes, so fix mean 0 and variance 1 and minimize $\int f^2$ subject to $f\ge0$, $\int f = 1$, $\int x^2f = 1$. This is a convex problem; minimizing the Lagrangian $\int\{f^2 - 2(a - bx^2)f\}$ pointwise over $f\ge0$ gives $f(x) = (a - bx^2)_+$. Imposing the two constraints yields support $|x|\le\sqrt5$ and $f(x) = \frac{3}{20\sqrt5}(5-x^2)$ (a rescaled Epanechnikov kernel), for which $\int f^2 = \frac{3}{5\sqrt5}$. Then $\mathrm{ARE} = 12\cdot\frac{9}{125} = \frac{108}{125}$.</p>
</details>
<p>So the worst case loses 13.6% of the sample; the normal case loses 4.5%; and the upside is unbounded. That asymmetric bet is the main argument for using Wilcoxon by default for shift problems. The lab below shows it in finite samples.</p>`,
        lab: 'rank-tests-power',
      },
      {
        title: 'k samples, ties, and estimation',
        html: R`
<div class="def" data-name="Kruskal–Wallis test">
<p>For $k$ groups of sizes $n_1,\dots,n_k$ ($N = \sum n_i$) with mean ranks $\bar R_i$ in the combined ranking,
$$H = \frac{12}{N(N+1)}\sum_{i=1}^kn_i\Big(\bar R_i - \frac{N+1}{2}\Big)^2\ \dto\ \chi^2_{k-1}\quad\text{under }H_0.$$
It is the one-way <a href="#/t/anova">ANOVA</a> $F$-statistic computed on ranks (up to a monotone transformation), and reduces to the two-sided rank-sum test when $k=2$.</p>
</div>
<p><strong>Ties.</strong> With ties (rounded data), assign each tied group the average of the ranks it occupies (midranks). The permutation null distribution is still exact conditionally on the tie pattern; for the normal approximation, the rank-sum variance becomes $\frac{mn}{12}\big\{(N+1) - \frac{\sum_g(t_g^3-t_g)}{N(N-1)}\big\}$ where $t_g$ are the tie-group sizes.</p>
<div class="def" data-name="Hodges–Lehmann estimators">
<p>Invert the rank test: the shift estimate is the value of $\theta$ that makes the test statistic computed on $(X_i, Y_j-\theta)$ most central. For two samples this is $\hat\Delta = \mathrm{median}\{Y_j - X_i\}$ over all $mn$ pairs; for one sample (signed-rank) it is the median of the Walsh averages $(X_i+X_j)/2$, $i\le j$. Confidence intervals come from order statistics of the same sets.</p>
</div>
<p>The Hodges–Lehmann estimator inherits the test's efficiency (its ARE relative to the mean is the same $12\sigma^2(\int f^2)^2$) and is robust: its breakdown point is $1 - 1/\sqrt2\approx0.29$, compared with 0 for the mean. See <a href="#/t/robust-statistics">robust statistics</a>.</p>
<div class="warn" data-name="Pitfalls">
<ul>
<li>Do not use the signed-rank test for a median when the distribution may be skewed — use the sign test.</li>
<li>Mann–Whitney is not a test for equal means or medians under unequal spreads; consider the Brunner–Munzel test, which targets $\P(Xt Y)+\frac12\P(X=Y) = \frac12$ without assuming equal shapes.</li>
<li>Rank tests are invariant to monotone transformations — a strength — but this means they cannot test hypotheses about the mean, which is not transformation-invariant.</li>
<li>With small samples the attainable significance levels are discrete; an exact p-value of 0.057 may be the closest to 0.05 available.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why is the null distribution of any rank statistic the same for every continuous $F$?`,
        a: R`Under $H_0$ the observations are IID (hence exchangeable) and ties have probability zero, so the rank vector is uniformly distributed over all $N!$ permutations whatever $F$ is. A statistic that depends on the data only through the ranks therefore has a fixed null distribution.` },
      { q: R`Compute $\E W$ and $\Var W$ for the rank-sum statistic with $m = 8$, $n = 12$.`,
        a: R`$N = 20$. $\E W = n(N+1)/2 = 12\cdot21/2 = 126$. $\Var W = mn(N+1)/12 = 8\cdot12\cdot21/12 = 168$, so $\sd(W)\approx12.96$.` },
      { q: R`Derive the ARE of the Wilcoxon test relative to the $t$-test when $F$ is normal.`,
        a: R`$\mathrm{ARE} = 12\sigma^2(\int f^2)^2$. For $\N(0,\sigma^2)$, $\int f^2 = 1/(2\sigma\sqrt\pi)$, so $(\int f^2)^2 = 1/(4\pi\sigma^2)$ and $\mathrm{ARE} = 12/(4\pi) = 3/\pi\approx0.955$.` },
      { q: R`Two populations have the same median but $G$ has three times the spread of $F$, both symmetric. Is Mann–Whitney a valid test of equal medians here?`,
        a: R`Approximately in terms of the target, since by symmetry about a common centre $\P(Xt Y) = \frac12$; but the null distribution of $U$ assumed $F=G$, and with unequal spreads the variance of $U$ differs from $mn(N+1)/12$, so the size is wrong (inflated or deflated depending on the sample-size imbalance). The rank test is not a general test of medians; use a Brunner–Munzel or studentized permutation approach.` },
      { q: R`What is the smallest possible Pitman ARE of Wilcoxon versus $t$ for shift alternatives, and which density attains it?`,
        a: R`$108/125 = 0.864$ (Hodges–Lehmann), attained by the parabolic density $f(x)\propto(5 - x^2)$ on $|x|\le\sqrt5$, after standardization — the shape of the Epanechnikov kernel.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'robust-statistics',
    domain: 'nonparametric',
    order: 90,
    title: 'Robust Estimation: M-Estimators & Breakdown',
    subtitle: 'Estimators that tolerate a fraction of bad data, and the influence and breakdown calculus that measures them',
    level: 3,
    sources: { aos: '—', cb: '§10.2', other: 'Huber (1964, 1981); Hampel, Ronchetti, Rousseeuw & Stahel (1986); Maronna, Martin, Yohai & Salibián-Barrera (2019)' },
    prereqs: ['statistical-functionals'],
    related: ['rank-tests', 'mle', 'regression-diagnostics', 'kernel-regression'],
    tags: ['M-estimation', 'influence function', 'breakdown point', 'Huber', 'contamination'],
    summary: R`Real data contain gross errors: transcription mistakes, instrument glitches, members of a different population. The sample mean, optimal under normality, is ruined by a single one. Robust statistics quantifies this fragility with two tools — the <strong>influence function</strong>, which measures the effect of infinitesimal contamination at a point, and the <strong>breakdown point</strong>, the largest fraction of arbitrary corruption an estimator survives — and builds estimators that do well on both while staying efficient at the model. <strong>M-estimators</strong>, solutions of $\sum\psi(X_i-\theta)=0$, are the central family: choosing a bounded $\psi$ such as Huber's buys bounded influence and 50% breakdown at a cost of only 5% efficiency at the normal.`,
    survey: {
      task: 'Estimate location, scale or regression coefficients when a fraction of data may be arbitrary',
      assumptions: R`Contamination model $(1-\epsilon)F_\theta + \epsilon G$; $F_\theta$ symmetric for location; auxiliary robust scale`,
      output: R`$\hat\theta$ solving $\sum\psi((X_i-\theta)/\hat s) = 0$; sandwich standard error $\sqrt{\E\psi^2/(\E\psi')^2/n}$`,
      strengths: 'Bounded influence; high breakdown; near-full efficiency at the model; downweights rather than deletes',
      weaknesses: R`Needs a scale estimate; redescending $\psi$ gives multiple roots; regression M-estimators break under leverage`,
      complexity: R`IRLS: a few weighted means (or weighted LS fits), $O(n)$ per iteration for location`,
    },
    sections: [
      {
        title: 'Contamination and the fragility of the mean',
        html: R`
<p>Model the data as coming from $F_\epsilon = (1-\epsilon)F_\theta + \epsilon G$: most observations follow the model $F_\theta$, a fraction $\epsilon$ come from an arbitrary distribution $G$. Hampel estimated that routine data typically contain 1–10% gross errors. We want estimators of $\theta$ that are (i) nearly optimal when $\epsilon = 0$, (ii) only slightly biased for small $\epsilon$, and (iii) not destroyed by larger $\epsilon$.</p>
<p>The mean fails (ii) and (iii) completely. From the <a href="#/t/statistical-functionals">influence function</a> $L(x) = x - \mu$, a point mass of size $\epsilon$ at $x$ shifts the mean by $\epsilon(x-\mu)$ — unbounded in $x$. One observation sent to infinity takes $\bar X$ with it. The variance and the correlation coefficient are worse. Even when nothing is "wrong", heavy tails make the mean inefficient: for the $t_3$ distribution the median has lower variance than the mean.</p>
<div class="note">
<p>"Robust" here means insensitive to small departures from the assumed model — a precise, quantitative notion, distinct from "nonparametric" (no model) and from "distribution-free" (exact null distribution for all $F$). A robust estimator is typically still tuned to a model (often the normal), but it degrades gracefully when the model is wrong.</p>
</div>
<p>Two quantitative tools measure robustness:</p>
<div class="cols">
<div><h4>Influence function</h4><p>Local: the effect of an infinitesimal amount of contamination at $x$. Summarized by the gross-error sensitivity $\gamma^* = \sup_x|L(x)|$. Approximate bias under $\epsilon$-contamination: $\epsilon L(x)$.</p></div>
<div><h4>Breakdown point</h4><p>Global: the largest fraction of the data that can be replaced by arbitrary values without the estimate becoming arbitrarily bad. The "distance to disaster".</p></div>
</div>`,
        lab: 'robust-statistics-contamination',
      },
      {
        title: 'Breakdown point',
        html: R`
<div class="def" data-name="Finite-sample replacement breakdown point (Donoho & Huber, 1983)">
<p>For a sample $\mathbf x$ of size $n$ and estimator $T_n$,
$$\epsilon^*_n(T_n,\mathbf x) = \min\Big\{\frac mn:\ \sup_{\mathbf x'}|T_n(\mathbf x') - T_n(\mathbf x)| = \infty\Big\},$$
the supremum over all samples $\mathbf x'$ obtained by replacing $m$ of the points with arbitrary values. The asymptotic breakdown point is the limit as $n\to\infty$.</p>
</div>
<table>
<thead><tr><th>Estimator</th><th>Finite-sample breakdown</th><th>Asymptotic</th></tr></thead>
<tbody>
<tr><td>Mean</td><td>$1/n$</td><td>$0$</td></tr>
<tr><td>$\alpha$-trimmed mean</td><td>$(\lfloor\alpha n\rfloor+1)/n$</td><td>$\alpha$</td></tr>
<tr><td>Median</td><td>$\lfloor(n+1)/2\rfloor/n$</td><td>$1/2$</td></tr>
<tr><td>Hodges–Lehmann (one-sample)</td><td>—</td><td>$1-1/\sqrt2\approx0.293$</td></tr>
<tr><td>Huber M-estimator with MAD scale</td><td>—</td><td>$1/2$</td></tr>
<tr><td>Standard deviation / IQR / MAD</td><td>$1/n$ / — / —</td><td>$0$ / $1/4$ / $1/2$</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Proposition" data-name="Breakdown of the median, and the upper bound">
<p>The sample median has $\epsilon^*_n = \lfloor(n+1)/2\rfloor/n$. No translation-equivariant location estimator can do better: $\epsilon^*_n\le\lfloor(n+1)/2\rfloor/n$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><strong>Median.</strong> If $n = 2k+1$ and at most $k$ points are replaced, at least $k+1$ originals remain, so the middle order statistic is bracketed by original values: bounded. Replacing $k+1$ points by a huge value $M$ makes the median $M$. If $n = 2k$, replacing $k$ points by $M$ makes the two middle order statistics an original and $M$, so the median is unbounded; replacing $k-1$ leaves both middle values original. In both cases breakdown occurs first at $m = \lfloor(n+1)/2\rfloor$.</p>
<p><strong>Upper bound.</strong> Let $m = \lfloor(n+1)/2\rfloor$, so $n - m\le m$, and fix $t\in\R$. Starting from $\mathbf x$, keep the first $n-m$ points and replace the other $m$ by $x_1+t,\dots,x_{n-m}+t$ (plus, when $n$ is odd, one extra arbitrary value $y$). Call the result $\mathbf x'$. The shifted sample $\mathbf x' - t$ consists of $x_1-t,\dots,x_{n-m}-t$, the originals $x_1,\dots,x_{n-m}$ and $y - t$: up to order, it is also $\mathbf x$ with $m$ points replaced. If $T$ stayed within distance $C$ of $T(\mathbf x)$ on all samples with $m$ replacements, then by equivariance $|t| = |T(\mathbf x') - T(\mathbf x'-t)|\le2C$ for every $t$ — impossible. So $m$ replacements always suffice to break a translation-equivariant estimator.</p>
</details>
<p>Breakdown and influence are different: the median and the 25%-trimmed mean both have bounded influence functions, but very different breakdown points; the median has the best breakdown possible yet its influence function jumps at the centre, which makes it inefficient. M-estimators let us trade these properties continuously. The second lab below lets you corrupt points one at a time and watch each estimator break.</p>`,
        lab: 'robust-statistics-breakdown',
      },
      {
        title: 'M-estimators',
        html: R`
<p>Generalize maximum likelihood (minimize $-\sum\log f(X_i-\theta)$) by replacing the negative log-density with a loss $\rho$ chosen for robustness.</p>
<div class="def" data-name="M-estimator of location (Huber, 1964)">
<p>$\hat\theta_n$ minimizes $\sum_i\rho(X_i - \theta)$, or more generally solves the estimating equation
$$\sum_{i=1}^n\psi(X_i - \hat\theta_n) = 0,\qquad\psi = \rho'.$$
The population version $\theta(F)$ solves $\int\psi(x-\theta)\d F(x) = 0$.</p>
</div>
<table>
<thead><tr><th>Estimator</th><th>$\rho(u)$</th><th>$\psi(u)$</th></tr></thead>
<tbody>
<tr><td>Mean</td><td>$u^2/2$</td><td>$u$</td></tr>
<tr><td>Median</td><td>$|u|$</td><td>$\sgn(u)$</td></tr>
<tr><td>Huber</td><td>$u^2/2$ if $|u|\le k$; $k|u| - k^2/2$ otherwise</td><td>$\max(-k,\min(k,u))$</td></tr>
<tr><td>Tukey biweight</td><td>$\frac{c^2}{6}\{1 - (1-(u/c)^2)^3\}$ if $|u|\le c$; $\frac{c^2}{6}$ otherwise</td><td>$u(1-(u/c)^2)^2\ind(|u|\le c)$</td></tr>
<tr><td>MLE for density $f$</td><td>$-\log f(u)$</td><td>$-f'(u)/f(u)$</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Theorem" data-name="Influence function and asymptotic normality of M-estimators">
<p>Let $\theta_0 = \theta(F)$, and suppose $\lambda(\theta) = \int\psi(x-\theta)\d F$ is differentiable at $\theta_0$ with $\lambda'(\theta_0) = -\E_F\psi'(X-\theta_0)\ne0$ (interpreting $\E\psi'$ as $-\lambda'$ when $\psi$ has jumps), and $\E\psi^2<\infty$. Then
$$L_F(x) = \frac{\psi(x-\theta_0)}{\E_F\psi'(X-\theta_0)},\qquad\sqrt n(\hat\theta_n - \theta_0)\dto\N\Big(0,\ \frac{\E_F\psi^2(X-\theta_0)}{\{\E_F\psi'(X-\theta_0)\}^2}\Big).$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><strong>IF.</strong> Differentiate $\int\psi(x'-\theta_\epsilon)\d F_\epsilon(x') = 0$, with $F_\epsilon = (1-\epsilon)F+\epsilon\delta_x$, at $\epsilon=0$: $-\E\psi(X-\theta_0) + \psi(x-\theta_0) - \dot\theta\,\E\psi'(X-\theta_0) = 0$; the first term is 0 by definition of $\theta_0$, giving $\dot\theta = \psi(x-\theta_0)/\E\psi'$.</p>
<p><strong>Normality.</strong> For monotone $\psi$, $\hat\theta_n$ is consistent because $\frac1n\sum\psi(X_i-\theta)$ is monotone in $\theta$ and converges pointwise to $\lambda(\theta)$, which changes sign at $\theta_0$. Expand $0 = \frac1n\sum\psi(X_i-\hat\theta_n)\approx\frac1n\sum\psi(X_i-\theta_0) - (\hat\theta_n-\theta_0)\frac1n\sum\psi'(X_i-\theta_0)$. So $\sqrt n(\hat\theta_n-\theta_0)\approx\frac{n^{-1/2}\sum\psi(X_i-\theta_0)}{\E\psi'}$, and the CLT gives the "sandwich" variance $\E\psi^2/(\E\psi')^2 = \int L_F^2\d F$, consistent with the <a href="#/t/statistical-functionals">nonparametric delta method</a>.</p>
</details>
<p>So $\psi$ is (up to a constant) the influence function: <strong>a bounded $\psi$ means bounded influence.</strong> When $\psi$ is the score of the true model, $\E\psi^2 = \E\psi' = I(\theta)$ and the variance is $1/I(\theta)$, the <a href="#/t/mle">MLE</a>'s Cramér–Rao bound. Robustness consists of deliberately choosing $\psi\ne$ score, paying a little variance at the model to bound the damage elsewhere.</p>`,
      },
      {
        title: 'Huber’s estimator and its minimax property',
        html: R`
<p>Huber's $\psi_k$ is linear in the middle (like the mean) and constant in the tails (like the median). For the normal model:</p>
<div class="ex" data-name="Efficiency of the Huber estimator at the normal">
<p>With $Z\sim\N(0,1)$: $\E\psi_k'(Z) = \P(|Z|\le k) = 2\Phi(k)-1$, and $\E\psi_k^2(Z) = (2\Phi(k)-1) - 2k\phi(k) + 2k^2\Phi(-k)$. The efficiency relative to the mean is $(\E\psi_k')^2/\E\psi_k^2$. At $k = 1.345$ it is $0.95$; at $k=1$ about $0.90$; as $k\to0$ it tends to the median's $2/\pi\approx0.64$, and as $k\to\infty$ to 1.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Huber's minimax theorem (1964)">
<p>Over the gross-error neighbourhood $\mathcal F_\epsilon = \{(1-\epsilon)\Phi + \epsilon G: G\text{ symmetric}\}$, the Huber M-estimator with $k$ solving
$$\frac{2\phi(k)}{k} - 2\Phi(-k) = \frac{\epsilon}{1-\epsilon}$$
minimizes the maximum asymptotic variance $\sup_{F\in\mathcal F_\epsilon}V(\psi,F)$ among all M-estimators of location; since that maximum equals the inverse Fisher information of the least favourable distribution, no regular estimator can do better uniformly over $\mathcal F_\epsilon$. The least favourable distribution is normal in the middle with exponential (Laplace-like) tails, and $\psi_k$ is its MLE score.</p>
</div>
<p>The theorem is a saddle point: the least favourable $F_0$ has density $\propto e^{-\rho_k(x)}$, Huber's $\psi_k$ is the MLE for $F_0$ (so it is optimal against $F_0$), and $V(\psi_k,F)\le V(\psi_k,F_0)$ for every $F\in\mathcal F_\epsilon$: the worst contamination puts its mass where $|\psi_k| = k$ is largest and $\psi_k' = 0$, which is what $F_0$'s tails already do. Hampel's complementary result: $\psi_k$ also minimizes the asymptotic variance at the normal subject to a bound on the gross-error sensitivity.</p>
<div class="warn" data-name="Scale is not optional">
<p>$\psi(X-\theta)$ is not scale-equivariant: multiplying the data by 10 changes which points count as "large". In practice one solves $\sum\psi((X_i-\theta)/\hat s) = 0$ with a robust scale estimate, typically $\hat s = \mathrm{MAD}/0.6745$, where $\mathrm{MAD} = \mathrm{median}|X_i - \mathrm{median}(X)|$ and the constant makes $\hat s$ consistent for $\sigma$ at the normal. Using the sample SD as the scale would reintroduce the mean's fragility through the back door.</p>
</div>
<div class="algo" data-name="Iteratively reweighted least squares (IRLS) for location">
<ol>
<li>Start from $\hat\theta^{(0)} = \mathrm{median}(X)$ and $\hat s = \mathrm{MAD}/0.6745$.</li>
<li>Compute residuals $u_i = (X_i - \hat\theta^{(t)})/\hat s$ and weights $w_i = \psi(u_i)/u_i$ (with $w_i = 1$ at $u_i = 0$ for Huber).</li>
<li>Update $\hat\theta^{(t+1)} = \sum w_iX_i/\sum w_i$, a weighted mean. Repeat to convergence.</li>
</ol>
<p>For Huber, $w_i = \min(1,k/|u_i|)$: points beyond $k$ scale units are downweighted in proportion to their distance, not deleted.</p>
</div>`,
      },
      {
        title: 'Redescending estimators and robust regression',
        html: R`
<p>Huber's $\psi$ never returns to zero, so a gross outlier still exerts influence $k$. <strong>Redescending</strong> M-estimators (Tukey's biweight, Hampel's three-part $\psi$) set $\psi = 0$ beyond a cutoff, so sufficiently wild points are ignored entirely. The biweight with $c = 4.685$ has 95% efficiency at the normal. The price: $\rho$ is non-convex, the estimating equation can have several roots, and the answer depends on the starting value — always start from a high-breakdown estimate such as the median.</p>
<p><strong>Regression.</strong> The M-estimator $\hat\beta = \argmin\sum\rho\big((Y_i - \bx_i^\top\beta)/\hat s\big)$ bounds the influence of large residuals but not of <em>leverage</em> points (outlying $\bx_i$): a single bad point far out in $x$-space can pull the fitted plane toward itself so that its own residual is small, and Huber regression has breakdown point $1/n$. Remedies:</p>
<ul>
<li><strong>High-breakdown estimators:</strong> least median of squares and least trimmed squares (Rousseeuw, 1984) minimize the median or a trimmed sum of squared residuals; breakdown near 1/2, but low efficiency and hard (combinatorial) computation.</li>
<li><strong>MM-estimators</strong> (Yohai, 1987): start from a high-breakdown S-estimate of scale and coefficients, then refine with a redescending M-step. They achieve breakdown 1/2 and, simultaneously, 95% efficiency at the normal — the modern default.</li>
<li><strong>GM-estimators</strong> also downweight high-leverage points (Mallows, Schweppe weights), bounding influence in $\bx$ as well as $y$.</li>
</ul>
<p>Robust fits double as diagnostics: points with near-zero final weight are the outliers, found without the masking problem that afflicts deletion diagnostics based on a non-robust fit (see <a href="#/t/regression-diagnostics">regression diagnostics</a>).</p>`,
      },
      {
        title: 'Pitfalls and perspective',
        html: R`
<div class="warn" data-name="Common mistakes">
<ul>
<li><strong>Deleting outliers, then using classical inference.</strong> "Reject points beyond 3 SD, then compute the mean and its $t$-interval" uses a non-robust SD to detect outliers (masking: several outliers inflate the SD and hide each other) and ignores the selection in the standard error.</li>
<li><strong>Asymmetric contamination.</strong> The theory above assumes symmetric $G$; with one-sided contamination every location estimator is biased. Robust estimators minimize the maximum bias (the median is minimax-bias), but they cannot remove it.</li>
<li><strong>What is being estimated?</strong> For asymmetric $F$ the Huber functional, the median and the mean are different parameters. Robustness is a property of how you estimate a well-defined target, so be clear which target the science needs.</li>
<li><strong>Standard errors.</strong> Use the sandwich formula $\hat\tau^2 = \frac1n\sum\psi^2(u_i)\hat s^2/\{\frac1n\sum\psi'(u_i)\}^2$ or a bootstrap — not $s/\sqrt n$.</li>
</ul>
</div>
<p>Rank-based procedures (<a href="#/t/rank-tests">Wilcoxon, Hodges–Lehmann</a>) are a parallel route to robustness, with the same flavour of guarantee: small loss at the normal, large gains under heavy tails or contamination. Both traditions teach that efficiency at an idealized model is a fragile virtue, and that a few percent of it is well spent on insurance.</p>`,
      },
    ],
    quiz: [
      { q: R`What is the influence function of the Huber M-estimator at a symmetric $F$, and what is its gross-error sensitivity?`,
        a: R`$L(x) = \psi_k(x-\theta)/\E\psi_k'(X-\theta) = \psi_k(x-\theta)/\P_F(|X-\theta|\le k)$. It is linear for $|x-\theta|\le k$ and constant beyond, so $\gamma^* = k/\P_F(|X-\theta|\le k)$, finite — unlike the mean's.` },
      { q: R`With $n = 20$, how many observations must be replaced to break the median? The 10% trimmed mean? The mean?`,
        a: R`Median: $\lfloor21/2\rfloor = 10$ (with 10 points sent to infinity, the average of the 10th and 11th order statistics is unbounded). 10% trimmed mean: it drops $\lfloor0.1\cdot20\rfloor = 2$ points from each end, so 3 replacements break it. Mean: 1.` },
      { q: R`Show that if $\psi$ is the score function of the true density, the M-estimator's asymptotic variance equals $1/I(\theta)$.`,
        a: R`With $\psi = -f'/f$ (location score), $\E\psi^2 = I$ by definition, and $\E\psi' = \int(-f'/f)'f = \int\{(f')^2/f - f''\} = I - 0 = I$ (since $\int f'' = 0$). So $V = \E\psi^2/(\E\psi')^2 = I/I^2 = 1/I$.` },
      { q: R`Why do robust regression M-estimators with Huber $\psi$ have breakdown point $1/n$ even though $\psi$ is bounded?`,
        a: R`A bounded $\psi$ limits the effect of large <em>residuals</em>, but a point with extreme $\bx$ (high leverage) can rotate the fit toward itself so that its residual stays small; its effective influence grows with $\|\bx\|$, which $\psi$ does not bound. One such point can carry the slope arbitrarily far. High-breakdown starts (LTS, S-estimators) and MM-estimators fix this.` },
    ],
  });
})();
