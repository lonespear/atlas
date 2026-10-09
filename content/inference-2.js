/* Statistical inference, part 2: Bayesian inference, testing, intervals, multiplicity, decisions, causality. */
(function () {
  const R = String.raw;

  /* ================================================================== */
  Atlas.topic({
    id: 'bayesian-inference',
    domain: 'inference',
    order: 100,
    title: 'Bayesian Inference & Conjugate Priors',
    subtitle: 'Treating the parameter as random and updating beliefs with Bayes’ theorem',
    level: 2,
    sources: { aos: 'Ch. 11', cb: '§7.2.3', other: 'Gelman et al., Bayesian Data Analysis (2013); van der Vaart (1998), Ch. 10' },
    prereqs: ['likelihood', 'conditional-probability'],
    related: ['exponential-families', 'decision-theory', 'mcmc-foundations', 'confidence-intervals', 'hierarchical-mcmc'],
    tags: ['prior', 'posterior', 'conjugacy', 'Jeffreys prior', 'Bernstein–von Mises', 'credible interval'],
    summary: R`Bayesian inference puts a probability distribution on the unknown parameter itself. A prior $\pi(\theta)$ encodes what is believed before the data; Bayes’ theorem combines it with the likelihood to give the posterior $\pi(\theta\mid x^n) \propto \mathcal L_n(\theta)\,\pi(\theta)$, and every inferential statement — point estimates, intervals, predictions, tests — is read off that one distribution. Conjugate families make the update a matter of adding sufficient statistics to prior "pseudo-data". The deep result that reconciles the two schools is the Bernstein–von Mises theorem: in regular parametric models the posterior forgets the prior and becomes the normal sampling distribution of the MLE, so credible intervals become confidence intervals.`,
    survey: {
      task: 'Full probabilistic inference about $\\theta$: estimates, intervals, predictions, model comparison',
      assumptions: R`A likelihood $f(x\mid\theta)$ and a prior $\pi(\theta)$; posterior must be proper`,
      output: R`Posterior $\pi(\theta\mid x^n)$; posterior mean / MAP; credible sets; posterior predictive`,
      strengths: 'Coherent; uses prior information; handles nuisance parameters by integration; exact small-sample statements given the model',
      weaknesses: 'Prior sensitivity at small $n$; computation (MCMC) for non-conjugate models; credible ≠ confidence in high dimensions',
      complexity: R`Closed form for conjugate models; otherwise integration / MCMC / VI`,
    },
    sections: [
      {
        title: 'The Bayesian recipe',
        html: R`
<p>In the frequentist view $\theta$ is a fixed unknown constant and probability describes only the data-generating process. The Bayesian view adds one ingredient: uncertainty about $\theta$ is also expressed by a probability distribution. Once that step is taken, inference is just conditional probability.</p>
<div class="def" data-name="Prior, posterior, marginal likelihood">
<p>Let $X_1,\dots,X_n \mid \theta \iid f(x\mid\theta)$ and let $\pi$ be a density on $\Theta$ (the <em>prior</em>). The <em>posterior</em> is the conditional density of $\theta$ given $x^n = (x_1,\dots,x_n)$:
$$\pi(\theta\mid x^n) = \frac{\mathcal L_n(\theta)\,\pi(\theta)}{m(x^n)}, \qquad \mathcal L_n(\theta) = \prod_{i=1}^n f(x_i\mid\theta), \qquad m(x^n) = \int_\Theta \mathcal L_n(\theta)\,\pi(\theta)\d\theta.$$
The normaliser $m(x^n)$ is the <em>marginal likelihood</em> (or evidence): the density of the data averaged over the prior.</p>
</div>
<p>Because $m(x^n)$ does not depend on $\theta$ we usually write $\pi(\theta\mid x^n) \propto \mathcal L_n(\theta)\pi(\theta)$ — "posterior is proportional to likelihood times prior". Two consequences are immediate. First, the data enter only through the <a href="#/t/likelihood">likelihood</a>, so Bayesian inference automatically obeys the likelihood principle, and any <a href="#/t/sufficiency">sufficient statistic</a> carries all the information the posterior uses. Second, updating is sequential: the posterior after $n$ observations is the prior for observation $n+1$, and the order of the observations is irrelevant.</p>
<p>Everything is then a summary of $\pi(\theta\mid x^n)$:</p>
<ul>
<li><strong>Point estimates:</strong> the posterior mean $\bar\theta_n = \int\theta\,\pi(\theta\mid x^n)\d\theta$, the posterior median, or the posterior mode (MAP). <a href="#/t/decision-theory">Decision theory</a> says which to use: they are the Bayes rules for squared, absolute, and (limiting) 0–1 loss.</li>
<li><strong>Credible sets:</strong> any $C$ with $\int_C \pi(\theta\mid x^n)\d\theta = 1-\alpha$. The <em>equal-tailed</em> interval cuts $\alpha/2$ from each tail; the <em>highest posterior density</em> (HPD) set $\{\theta : \pi(\theta\mid x^n) \ge c\}$ is the shortest region with that mass.</li>
<li><strong>Prediction:</strong> the posterior predictive density $f(x_{\text{new}}\mid x^n) = \int f(x_{\text{new}}\mid\theta)\,\pi(\theta\mid x^n)\d\theta$ integrates over parameter uncertainty instead of plugging in an estimate.</li>
<li><strong>Nuisance parameters:</strong> if $\theta = (\psi,\lambda)$, the marginal posterior $\pi(\psi\mid x^n) = \int \pi(\psi,\lambda\mid x^n)\d\lambda$ removes $\lambda$ by integration rather than maximisation.</li>
</ul>
<div class="note">
<p>A credible interval says "given the model, the prior and these data, $\theta$ lies in $C$ with probability 0.95". A confidence interval says "the procedure that produced $C$ traps the true $\theta$ in 95% of repeated samples". The first conditions on the data and averages over $\theta$; the second conditions on $\theta$ and averages over data. They answer different questions — which is why it is remarkable that, asymptotically, they give the same numbers.</p>
</div>`,
      },
      {
        title: 'Conjugate priors',
        html: R`
<p>For most priors the integral $m(x^n)$ has no closed form. A family $\mathcal P$ of priors is <em>conjugate</em> for a likelihood if the posterior is again in $\mathcal P$ for every data set. Then updating reduces to updating a few hyperparameters.</p>
<div class="thm" data-kind="Theorem" data-name="Conjugate priors for exponential families">
<p>Let $f(x\mid\theta) = h(x)\exp\{\eta(\theta)^\top T(x) - A(\theta)\}$ be an <a href="#/t/exponential-families">exponential family</a>. For hyperparameters $(\tau, n_0)$ define
$$\pi_{\tau,n_0}(\theta) \propto \exp\{\eta(\theta)^\top \tau - n_0 A(\theta)\},$$
whenever this is integrable. Then the posterior after $x_1,\dots,x_n$ is $\pi_{\tau', n_0'}$ with
$$\tau' = \tau + \sum_{i=1}^n T(x_i), \qquad n_0' = n_0 + n.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\mathcal L_n(\theta)\,\pi_{\tau,n_0}(\theta) \propto \exp\{\eta(\theta)^\top\sum_i T(x_i) - nA(\theta)\}\cdot\exp\{\eta(\theta)^\top\tau - n_0A(\theta)\} = \exp\{\eta(\theta)^\top(\tau + \sum_iT(x_i)) - (n_0+n)A(\theta)\}$, where factors $\prod h(x_i)$ not involving $\theta$ were absorbed in the proportionality constant. This is the kernel of $\pi_{\tau',n_0'}$.</p>
</details>
<p>The hyperparameters have a direct reading: the prior behaves like $n_0$ <em>pseudo-observations</em> whose sufficient statistics sum to $\tau$. The data add $n$ real observations. This is why conjugate posteriors are always compromises between a prior guess and the data, with weights proportional to the prior and actual sample sizes.</p>
<table>
<thead><tr><th>Likelihood</th><th>Conjugate prior</th><th>Posterior</th><th>Posterior mean as a compromise</th></tr></thead>
<tbody>
<tr><td>$\text{Bernoulli}(p)$, $k$ successes in $n$</td><td>$\text{Beta}(a,b)$</td><td>$\text{Beta}(a+k,\, b+n-k)$</td><td>$w\,\hat p + (1-w)\frac{a}{a+b}$, $w=\frac{n}{n+a+b}$</td></tr>
<tr><td>$\text{Poisson}(\lambda)$, total $s=\sum x_i$</td><td>$\text{Gamma}(\alpha,\text{rate }\beta)$</td><td>$\text{Gamma}(\alpha+s,\, \beta+n)$</td><td>$w\,\bar x + (1-w)\frac{\alpha}{\beta}$, $w=\frac{n}{n+\beta}$</td></tr>
<tr><td>$\N(\theta,\sigma^2)$, $\sigma^2$ known</td><td>$\N(\mu_0,\tau^2)$</td><td>$\N(\mu_n,\tau_n^2)$</td><td>$\mu_n = w\bar x + (1-w)\mu_0$, $w = \frac{n/\sigma^2}{n/\sigma^2 + 1/\tau^2}$</td></tr>
<tr><td>$\N(\mu,\sigma^2)$, $\mu$ known</td><td>$\text{Inv-Gamma}(\alpha,\beta)$ on $\sigma^2$</td><td>$\text{Inv-Gamma}(\alpha+\tfrac n2,\, \beta+\tfrac12\sum(x_i-\mu)^2)$</td><td>—</td></tr>
<tr><td>$\text{Multinomial}(p_1,\dots,p_K)$</td><td>$\text{Dirichlet}(\alpha_1,\dots,\alpha_K)$</td><td>$\text{Dirichlet}(\alpha_k + n_k)$</td><td>$\frac{\alpha_k+n_k}{\sum_j\alpha_j + n}$</td></tr>
</tbody>
</table>
<div class="ex" data-name="Normal mean with known variance">
<p>Completing the square in $-\frac{1}{2\sigma^2}\sum(x_i-\theta)^2 - \frac{1}{2\tau^2}(\theta-\mu_0)^2$ gives a normal kernel with
$$\frac{1}{\tau_n^2} = \frac{n}{\sigma^2} + \frac{1}{\tau^2}, \qquad \mu_n = \tau_n^2\Big(\frac{n\bar x}{\sigma^2} + \frac{\mu_0}{\tau^2}\Big).$$
<strong>Precisions add</strong>, and the posterior mean is the precision-weighted average of the data mean and the prior mean. As $n\to\infty$, $w\to1$ and $\tau_n^2 \approx \sigma^2/n$: the posterior concentrates at $\bar x$ with exactly the frequentist standard error. As $\tau^2\to\infty$ (a flat prior) the posterior is $\N(\bar x, \sigma^2/n)$ for every $n$, and the 95% credible interval $\bar x \pm 1.96\sigma/\sqrt n$ coincides with the textbook confidence interval.</p>
</div>
<div class="ex" data-name="Beta–Binomial and the rule of succession">
<p>With a $\text{Beta}(a,b)$ prior and $k$ successes in $n$ trials, the posterior predictive probability that the next trial succeeds is
$$\P(X_{n+1}=1\mid x^n) = \int_0^1 p\,\pi(p\mid x^n)\d p = \frac{a+k}{a+b+n}.$$
With the uniform prior $a=b=1$ this is Laplace’s rule of succession $(k+1)/(n+2)$: after observing 10 successes in 10 trials it predicts $11/12$, not the MLE’s certainty of 1. The prior’s pseudo-counts are exactly what keeps the estimate off the boundary.</p>
</div>`,
        lab: 'bayesian-inference-beta-binomial',
      },
      {
        title: 'Choosing a prior: flat priors and Jeffreys’ invariance',
        html: R`
<p>Where does $\pi$ come from? <em>Subjective</em> priors encode genuine expert knowledge. <em>Weakly informative</em> priors rule out absurd values (a log-odds of 40) while letting the data dominate. <em>Objective</em> or reference priors try to express "no information". The last is subtler than it sounds.</p>
<div class="warn" data-name="Flat is not invariant">
<p>A uniform prior on $p\in(0,1)$ is not uniform on the log-odds $\psi = \log\{p/(1-p)\}$: by the change-of-variables formula the induced density is $\pi(\psi) = e^\psi/(1+e^\psi)^2$, a logistic density concentrated near $\psi = 0$. "Ignorance" about $p$ is knowledge about $\psi$. Any rule that says "use the flat prior" therefore depends on an arbitrary choice of parameterisation.</p>
</div>
<p>Jeffreys (1946) proposed a rule whose output does not depend on that choice.</p>
<div class="thm" data-kind="Theorem" data-name="Invariance of the Jeffreys prior">
<p>For a one-parameter regular model with <a href="#/t/fisher-information">Fisher information</a> $I(\theta)$, define $\pi_J(\theta) \propto \sqrt{I(\theta)}$. If $\phi = g(\theta)$ for a smooth bijection $g$, then the Jeffreys prior computed directly in the $\phi$-parameterisation equals the density induced on $\phi$ by $\pi_J(\theta)$ via the change of variables. In $d$ dimensions the same holds for $\pi_J(\theta)\propto\sqrt{\det I(\theta)}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $\theta = g^{-1}(\phi)$. The score in $\phi$ is $\partial_\phi \log f = (\partial_\theta\log f)\,\frac{\d\theta}{\d\phi}$, so the information transforms as $I_\phi(\phi) = I_\theta(\theta)\big(\frac{\d\theta}{\d\phi}\big)^2$. Hence $\sqrt{I_\phi(\phi)} = \sqrt{I_\theta(\theta)}\,\big|\frac{\d\theta}{\d\phi}\big|$, which is exactly the Jacobian rule for transforming the density $\sqrt{I_\theta(\theta)}$ from $\theta$ to $\phi$. In $d$ dimensions, $I_\phi = J^\top I_\theta J$ with $J = \partial\theta/\partial\phi$, and $\sqrt{\det I_\phi} = \sqrt{\det I_\theta}\,|\det J|$.</p>
</details>
<table>
<thead><tr><th>Model</th><th>$I(\theta)$</th><th>Jeffreys prior</th><th>Remark</th></tr></thead>
<tbody>
<tr><td>$\text{Bernoulli}(p)$</td><td>$\frac{1}{p(1-p)}$</td><td>$\text{Beta}(\tfrac12,\tfrac12)$</td><td>proper; U-shaped</td></tr>
<tr><td>$\text{Poisson}(\lambda)$</td><td>$1/\lambda$</td><td>$\lambda^{-1/2}$</td><td>improper</td></tr>
<tr><td>$\N(\mu,\sigma^2)$, location $\mu$</td><td>$1/\sigma^2$</td><td>flat in $\mu$</td><td>improper; translation-invariant</td></tr>
<tr><td>$\N(\mu,\sigma^2)$, scale $\sigma$</td><td>$2/\sigma^2$</td><td>$1/\sigma$</td><td>flat in $\log\sigma$; scale-invariant</td></tr>
</tbody>
</table>
<p><strong>Improper priors</strong> (infinite total mass) are acceptable only if the resulting posterior is proper, i.e. $m(x^n)\lt\infty$. For the Poisson with prior $\lambda^{-1/2}$ the posterior is $\text{Gamma}(s+\tfrac12, n)$, proper for every data set. In other models (variance components in hierarchical models, mixtures) improper priors can produce improper posteriors that an MCMC sampler will happily, and meaninglessly, explore.</p>
<div class="warn" data-name="Jeffreys in several dimensions">
<p>For $\N(\mu,\sigma^2)$ with both unknown, $\sqrt{\det I}\propto 1/\sigma^2$, which yields marginal posteriors whose degrees of freedom do not match the classical $t$ interval. Jeffreys himself preferred the "independence" prior $1/\sigma$. Multiparameter objective priors (reference priors of Bernardo and Berger) order the parameters by interest and are not simply $\sqrt{\det I}$.</p>
</div>`,
      },
      {
        title: 'Large-sample theory: consistency and Bernstein–von Mises',
        html: R`
<p>Does the posterior find the truth? Suppose the data are in fact IID from $f(\cdot\mid\theta_0)$. We want $\Pi(\cdot\mid X^n)$ to concentrate on $\theta_0$.</p>
<div class="thm" data-kind="Proposition" data-name="Posterior consistency, finite parameter space">
<p>Let $\Theta = \{\theta_0,\theta_1,\dots,\theta_K\}$ be finite and identifiable ($f(\cdot\mid\theta_j)\ne f(\cdot\mid\theta_0)$ for $j\ne0$), with $\pi(\theta_0)\gt0$ and $\KL(f_{\theta_0}\,\|\,f_{\theta_j})\lt\infty$. Then $\pi(\theta_0\mid X^n)\to1$ almost surely, and each $\pi(\theta_j\mid X^n)$, $j\neq 0$, tends to 0 exponentially fast.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For $j\ne0$, the posterior odds are
$$\frac{\pi(\theta_j\mid X^n)}{\pi(\theta_0\mid X^n)} = \frac{\pi(\theta_j)}{\pi(\theta_0)}\exp\Big\{\sum_{i=1}^n \log\frac{f(X_i\mid\theta_j)}{f(X_i\mid\theta_0)}\Big\}.$$
By the strong law, $\frac1n\sum_i\log\frac{f(X_i\mid\theta_j)}{f(X_i\mid\theta_0)} \asto -\KL(f_{\theta_0}\|f_{\theta_j}) \lt 0$, the inequality being strict by identifiability (Gibbs’ inequality). So the odds behave like $e^{-n\,\KL}$ and tend to 0 a.s. Summing the finitely many odds, $\pi(\theta_0\mid X^n) = (1+\sum_{j\ne0}\text{odds}_j)^{-1}\to1$.</p>
</details>
<p>The infinite-dimensional story is subtler. Doob’s theorem (1949) guarantees consistency for $\pi$-almost every $\theta_0$ — reassuring to the Bayesian, useless to someone who wants a guarantee at the particular truth. Schwartz’s theorem (1965) gives consistency at $\theta_0$ provided the prior puts positive mass on every Kullback–Leibler neighbourhood of $f_{\theta_0}$ and the model admits uniformly consistent tests. For regular finite-dimensional models with a prior density continuous and positive at $\theta_0$, these hold. In nonparametric problems they can fail (Diaconis & Freedman, 1986), so priors must be chosen with care. The sharpest parametric result describes not just where the posterior goes but its shape.</p>
<div class="thm" data-kind="Theorem" data-name="Bernstein–von Mises">
<p>Let the model be differentiable in quadratic mean at an interior point $\theta_0\in\Theta\subset\R^d$ with nonsingular Fisher information $I(\theta_0)$, suppose there exist uniformly consistent tests of $\theta=\theta_0$ against $\{\|\theta-\theta_0\|\ge\epsilon\}$ for each $\epsilon\gt0$, and let the prior have a density continuous and positive at $\theta_0$. Then, with $\hth_n$ any efficient estimator (e.g. the MLE),
$$\Big\| \Pi(\cdot\mid X^n) - \N\big(\hth_n,\; \tfrac1n I(\theta_0)^{-1}\big)\Big\|_{\mathrm{TV}} \pto 0 \quad\text{under } \theta_0.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Reparameterise by the local parameter $h = \sqrt n(\theta-\hth_n)$. The posterior density of $h$ is proportional to $\exp\{\ell_n(\hth_n + h/\sqrt n) - \ell_n(\hth_n)\}\,\pi(\hth_n + h/\sqrt n)$. A second-order expansion of the log-likelihood around $\hth_n$, where the score vanishes, gives
$$\ell_n(\hth_n + h/\sqrt n) - \ell_n(\hth_n) = -\tfrac12 h^\top I(\theta_0)h + o_p(1)$$
uniformly on compact sets of $h$ (local asymptotic normality). Continuity of $\pi$ at $\theta_0$ and consistency of $\hth_n$ make $\pi(\hth_n + h/\sqrt n)\to\pi(\theta_0)$, a constant that cancels in normalisation. So on compacts the posterior of $h$ is approximately $\N(0, I(\theta_0)^{-1})$. The testing condition is what shows the posterior mass outside $\{\|h\|\le M\}$ is negligible; without it the tails of the posterior could wander. Scheffé’s lemma converts convergence of densities into total variation.</p>
</details>
<p>The consequences are what make the theorem important:</p>
<ul>
<li><strong>The prior washes out.</strong> Any two priors that are continuous and positive at $\theta_0$ lead to asymptotically identical posteriors.</li>
<li><strong>Credible sets are confidence sets.</strong> A $1-\alpha$ credible interval is $\hth_n \pm z_{\alpha/2}\,\widehat{\se} + o_p(n^{-1/2})$, whose frequentist coverage tends to $1-\alpha$. The second lab measures this exactly for the Beta–Binomial.</li>
<li><strong>The posterior mean is efficient.</strong> $\sqrt n(\bar\theta_n - \theta_0) \dto \N(0, I(\theta_0)^{-1})$, attaining the Cramér–Rao bound.</li>
</ul>`,
        lab: 'bayesian-inference-coverage',
      },
      {
        title: 'Where Bayesian inference goes wrong',
        html: R`
<div class="warn" data-name="Small samples and strong priors">
<p>Before BvM kicks in, the prior matters, and it should: that is its job. The danger is a prior that is strong <em>by accident</em>. A $\text{Beta}(20,20)$ prior is worth 40 observations; with $n=10$ the posterior mean is pulled 80% of the way to $1/2$. Always report the effective prior sample size and run a sensitivity analysis over plausible priors.</p>
</div>
<div class="warn" data-name="Bernstein–von Mises can fail">
<p>The theorem needs its conditions. It fails when $\theta_0$ is on the boundary of $\Theta$ (the posterior is a truncated normal), when the model is non-regular (e.g. $\text{Unif}(0,\theta)$, where the posterior contracts at rate $1/n$ with an exponential, not normal, shape), when parameters are not identified, and when the dimension grows with $n$. In high-dimensional and nonparametric models, credible sets can have frequentist coverage far from nominal — sometimes near zero (Freedman, 1999; Szabó, van der Vaart & van Zanten, 2015).</p>
</div>
<div class="warn" data-name="Bayes factors are prior-sensitive">
<p>Model comparison with $B_{10} = m_1(x^n)/m_0(x^n)$ depends on the prior <em>scale</em> under the alternative even asymptotically. A very diffuse prior on an alternative spreads its mass over values the data rule out, so $m_1$ shrinks and the Bayes factor favours the null regardless of the data: the Jeffreys–Lindley paradox, discussed in <a href="#/t/hypothesis-testing">hypothesis testing</a>. Improper priors make Bayes factors undefined (the arbitrary constant does not cancel).</p>
</div>
<div class="warn" data-name="The model is still a model">
<p>The posterior is a statement conditional on the likelihood being right. Under misspecification the posterior concentrates on the KL-closest parameter, but its spread is governed by the inverse Hessian rather than the sandwich variance, so credible intervals can be badly miscalibrated (Kleijn & van der Vaart, 2012). Posterior predictive checks are the Bayesian’s diagnostic.</p>
</div>`,
      },
      {
        title: 'Connections',
        html: R`
<ul>
<li><strong>Decision theory.</strong> Posterior summaries are <a href="#/t/decision-theory">Bayes rules</a>; Bayes rules with proper priors are admissible; shrinkage estimators such as James–Stein are empirical-Bayes posterior means.</li>
<li><strong>Computation.</strong> Outside conjugate families the posterior is known only up to $m(x^n)$, which is exactly the situation <a href="#/t/mcmc-foundations">Markov chain Monte Carlo</a> was built for; <a href="#/t/variational-inference">variational inference</a> trades exactness for speed.</li>
<li><strong>Hierarchical models.</strong> Putting priors on prior hyperparameters lets groups borrow strength from each other: see <a href="#/t/hierarchical-mcmc">hierarchical models</a> and <a href="#/t/mixed-models">mixed models</a>.</li>
<li><strong>Regularisation.</strong> MAP estimation with a Gaussian prior is <a href="#/t/ridge-regression">ridge regression</a>; with a Laplace prior it is the <a href="#/t/lasso">lasso</a> (though the Bayesian lasso posterior mean is not sparse).</li>
<li><strong>Gaussian processes</strong> are conjugate priors over functions: <a href="#/t/gaussian-processes">Gaussian process regression</a> is the normal–normal update in infinite dimensions.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`A $\text{Beta}(2,8)$ prior is updated with 30 successes in 40 trials. What is the posterior, its mean, and how much weight does the posterior mean put on the MLE?`,
        a: R`Posterior $\text{Beta}(32, 18)$ with mean $32/50 = 0.64$. The weight on $\hat p = 0.75$ is $w = n/(n+a+b) = 40/50 = 0.8$, and $0.8(0.75) + 0.2(0.2) = 0.64$. The prior acts like 10 pseudo-observations with 2 successes.` },
      { q: R`Why is a uniform prior on $p$ not "uninformative" in an absolute sense, and how does Jeffreys’ prior address this?`,
        a: R`Uniformity is not preserved under reparameterisation: uniform on $p$ induces a logistic density on the log-odds, so the same "ignorance" is informative about another parameter. The Jeffreys prior $\propto\sqrt{I(\theta)}$ transforms with exactly the Jacobian of the reparameterisation (because Fisher information transforms with the squared Jacobian), so applying the rule in any parameterisation gives the same prior. For the Bernoulli it is $\text{Beta}(\tfrac12,\tfrac12)$.` },
      { q: R`State in words what the Bernstein–von Mises theorem implies about the frequentist coverage of 95% credible intervals, and give one situation in which the conclusion fails.`,
        a: R`In a regular, finite-dimensional model with a prior continuous and positive at the truth, the posterior is asymptotically $\N(\hth_n, I(\theta_0)^{-1}/n)$, so a 95% credible interval is asymptotically the Wald interval and has frequentist coverage tending to 95%. It fails, for example, when $\theta_0$ is on the boundary, in non-regular models like $\text{Unif}(0,\theta)$, under misspecification, or when the number of parameters grows with $n$.` },
      { q: R`In the normal model with known $\sigma^2$ and prior $\N(\mu_0,\tau^2)$, what happens to the posterior as $\tau^2\to0$ and as $\tau^2\to\infty$?`,
        a: R`Posterior precision is $n/\sigma^2 + 1/\tau^2$. As $\tau^2\to0$ the prior precision dominates and the posterior collapses on $\mu_0$, ignoring the data. As $\tau^2\to\infty$ the posterior becomes $\N(\bar x,\sigma^2/n)$ — the flat-prior limit — and credible intervals equal classical confidence intervals exactly.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'hypothesis-testing',
    domain: 'inference',
    order: 110,
    title: 'Hypothesis Testing: Errors, Power & p-values',
    subtitle: 'Deciding between hypotheses with controlled error rates',
    level: 2,
    sources: { aos: '10.1–10.2', cb: '§8.1, §8.3.1, §8.3.4', other: 'Lehmann & Romano, Testing Statistical Hypotheses (2005); Wasserstein & Lazar (2016)' },
    prereqs: ['sampling-distributions'],
    related: ['neyman-pearson', 'likelihood-ratio-tests', 'confidence-intervals', 'multiple-testing', 'permutation-tests'],
    tags: ['type I error', 'power', 'p-value', 'size', 'sample size'],
    summary: R`A test partitions the sample space into "reject $H_0$" and "do not reject". Its quality is summarised by one function, the power function $\beta(\theta) = \P_\theta(\text{reject})$: we want it small on the null (type I error, bounded by the level $\alpha$) and large on the alternative (power). The p-value is the smallest level at which the observed data would lead to rejection; under a continuous simple null it is exactly uniform, which is both what makes it a valid error-rate device and why it cannot be read as the probability that $H_0$ is true.`,
    survey: {
      task: R`Decide between $H_0:\theta\in\Theta_0$ and $H_1:\theta\in\Theta_1$ with a guaranteed type I error rate`,
      assumptions: R`A model $\{P_\theta\}$; a test statistic with known (or asymptotically known) null distribution`,
      output: R`A reject / do-not-reject decision; a p-value; a power function $\beta(\theta)$`,
      strengths: 'Controls false-positive rate exactly or asymptotically; power analysis plans sample sizes',
      weaknesses: R`Dichotomises evidence; p-values are routinely misread as $\P(H_0\mid\text{data})$; significance ≠ importance`,
      complexity: R`Usually $O(n)$ for the statistic; exact tests may need enumeration or simulation`,
    },
    sections: [
      {
        title: 'The testing problem',
        html: R`
<p>Partition the parameter space into $\Theta_0$ and $\Theta_1$ and consider the hypotheses
$$H_0:\theta\in\Theta_0 \qquad\text{versus}\qquad H_1:\theta\in\Theta_1.$$
A hypothesis is <em>simple</em> if it pins down a single distribution ($\Theta_0=\{\theta_0\}$) and <em>composite</em> otherwise. A test is a rule that maps the data to a decision; most generally a <em>test function</em> $\phi(x)\in[0,1]$ is the probability of rejecting $H_0$ after seeing $x$. A non-randomised test has $\phi = \ind\{x\in\mathcal R\}$ for a <em>rejection region</em> $\mathcal R$, typically of the form $\{T(x) \gt c\}$ for a test statistic $T$ and critical value $c$.</p>
<table>
<thead><tr><th></th><th>Retain $H_0$</th><th>Reject $H_0$</th></tr></thead>
<tbody>
<tr><td>$H_0$ true</td><td>correct</td><td><strong>type I error</strong></td></tr>
<tr><td>$H_1$ true</td><td><strong>type II error</strong></td><td>correct (power)</td></tr>
</tbody>
</table>
<div class="def" data-name="Power function, size, level">
<p>The <em>power function</em> of a test $\phi$ is $\beta(\theta) = \E_\theta\,\phi(X)$, the rejection probability when $\theta$ is true. The test has <em>size</em> $\alpha$ if $\sup_{\theta\in\Theta_0}\beta(\theta) = \alpha$ and <em>level</em> $\alpha$ if $\sup_{\theta\in\Theta_0}\beta(\theta)\le\alpha$. On $\Theta_1$, $\beta(\theta)$ is the power and $1-\beta(\theta)$ the type II error probability.</p>
</div>
<p>The power function is the complete description of a test: one function on $\Theta$ whose values on $\Theta_0$ are error probabilities and on $\Theta_1$ are success probabilities. An ideal test would have $\beta = \ind_{\Theta_1}$; with finite data this is impossible because $\beta$ is typically continuous in $\theta$, so it cannot jump at the boundary between $\Theta_0$ and $\Theta_1$.</p>
<div class="note" data-name="Why the asymmetry">
<p>The Neyman–Pearson paradigm fixes the type I error at $\alpha$ and then maximises power. The hypotheses are therefore not interchangeable: $H_0$ is the claim that must be protected against false rejection — "the drug has no effect", "the new process is no worse". Failing to reject $H_0$ is not evidence for it; it may simply mean low power.</p>
</div>`,
      },
      {
        title: 'Power and sample size',
        html: R`
<p>The canonical example is the one-sample $z$ test. Let $X_1,\dots,X_n\iid\N(\mu,\sigma^2)$ with $\sigma$ known, and test $H_0:\mu\le\mu_0$ against $H_1:\mu\gt\mu_0$ by rejecting when $Z = \sqrt n(\bar X - \mu_0)/\sigma \gt z_\alpha$, where $z_\alpha = \Phi^{-1}(1-\alpha)$.</p>
<div class="thm" data-kind="Proposition" data-name="Power of the z test">
<p>The power function of the one-sided test is
$$\beta(\mu) = 1 - \Phi\Big(z_\alpha - \frac{\sqrt n(\mu-\mu_0)}{\sigma}\Big),$$
which is increasing in $\mu$, so the test has size $\beta(\mu_0) = \alpha$. For the two-sided test of $H_0:\mu=\mu_0$ that rejects when $|Z|\gt z_{\alpha/2}$, with $\delta = \sqrt n(\mu-\mu_0)/\sigma$,
$$\beta(\mu) = 1 - \Phi(z_{\alpha/2} - \delta) + \Phi(-z_{\alpha/2} - \delta).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Under $\mu$, $Z = \frac{\sqrt n(\bar X-\mu)}{\sigma} + \delta$ with the first term standard normal. So $\P_\mu(Z\gt z_\alpha) = \P(N \gt z_\alpha - \delta) = 1-\Phi(z_\alpha-\delta)$, increasing in $\delta$ and hence in $\mu$. The supremum over $\mu\le\mu_0$ is attained at $\mu_0$ where $\delta=0$, giving $\alpha$. The two-sided formula adds the probability $\P(N \lt -z_{\alpha/2}-\delta)$ of the lower tail.</p>
</details>
<p>The quantity $\delta$ is the <em>standardised effect</em> measured in standard errors. Power depends on $n$, $\sigma$ and $\mu-\mu_0$ only through $\delta$, so doubling the effect is worth quadrupling the sample size. Setting $\beta(\mu_1) = 1-\gamma$ and solving gives the classical sample-size formula for detecting a difference $\Delta = \mu_1-\mu_0$ with power $1-\gamma$:
$$n = \frac{\sigma^2(z_\alpha + z_\gamma)^2}{\Delta^2} \qquad(\text{two-sided: replace } z_\alpha \text{ by } z_{\alpha/2}\text{, ignoring the far tail}).$$</p>
<div class="ex" data-name="Planning a study">
<p>To detect a mean shift of half a standard deviation ($\Delta/\sigma = 0.5$) with a two-sided 5% test and 80% power: $z_{0.025}=1.960$, $z_{0.2} = 0.842$, so $n = (2.802)^2/0.25 \approx 31.4$ — 32 subjects. For a shift of $0.2\sigma$ the same calculation needs 197. Small effects are expensive: required $n$ scales as $1/\Delta^2$.</p>
</div>`,
        lab: 'hypothesis-testing-power',
      },
      {
        title: 'p-values and their null distribution',
        html: R`
<p>Reporting only "reject at 5%" throws information away: the same decision is reached whether the statistic barely crossed $c$ or exceeded it by a mile. A p-value reports how extreme the data are on a scale every reader can threshold at their own $\alpha$.</p>
<div class="def" data-name="p-value">
<p>Suppose large values of $T(X)$ are evidence against $H_0$. The p-value of observed data $x$ is
$$p(x) = \sup_{\theta\in\Theta_0}\P_\theta\big(T(X)\ge T(x)\big).$$
Equivalently, if we have a family of level-$\alpha$ tests with nested rejection regions $\mathcal R_\alpha$, $p(x) = \inf\{\alpha : x\in\mathcal R_\alpha\}$: the smallest level at which $H_0$ would be rejected.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Validity of p-values">
<p>For every $\theta\in\Theta_0$ and every $\alpha\in[0,1]$, $\P_\theta(p(X)\le\alpha)\le\alpha$ (the p-value is <em>super-uniform</em>). If $\Theta_0 = \{\theta_0\}$ is simple and $T(X)$ has a continuous distribution under $\theta_0$, then $p(X)\sim\text{Unif}(0,1)$ exactly.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Fix $\theta\in\Theta_0$ and let $G(t) = \P_\theta(T\ge t)$, a non-increasing, left-continuous function. By definition of the supremum, $p(X)\ge G(T(X))$, so it suffices to show $\P_\theta(G(T)\le\alpha)\le\alpha$. The set $\{t: G(t)\le\alpha\}$ is an up-set (if $t$ is in it, so is every $t'\gt t$), so it is either $[t^*,\infty)$ or $(t^*,\infty)$. In the first case $\P(G(T)\le\alpha) = \P(T\ge t^*) = G(t^*) \le\alpha$. In the second, $\P(G(T)\le\alpha) = \P(T\gt t^*) = \lim_{t\downarrow t^*}G(t)\le\alpha$, since each $G(t)\le\alpha$ for $t\gt t^*$.</p>
<p>If the null is simple, $p(X) = G(T) = 1 - F(T^-)$ where $F$ is the CDF of $T$; if $F$ is continuous this is $1-F(T)$, and $F(T)\sim\text{Unif}(0,1)$ by the probability integral transform.</p>
</details>
<p>Uniformity under $H_0$ is the operational meaning of a p-value: rejecting when $p\le\alpha$ is a level-$\alpha$ test. Three practical corollaries:</p>
<ul>
<li>For a <strong>discrete</strong> statistic (binomial, Fisher’s exact test), $p$ is stochastically larger than uniform and the test is conservative; mid-p values or randomisation restore exactness at a price in interpretability.</li>
<li>For a <strong>composite</strong> null, $p$ is uniform only at the least favourable boundary point; deep inside $\Theta_0$ it piles up near 1.</li>
<li>Under the <strong>alternative</strong>, the distribution of $p$ is pushed toward 0. Its CDF at $\alpha$ is exactly the power at level $\alpha$. The second lab shows the histogram morphing from flat to a spike at 0 as the effect grows.</li>
</ul>`,
        lab: 'hypothesis-testing-pvalues',
      },
      {
        title: 'What a p-value is not',
        html: R`
<p>A p-value is a probability computed <em>assuming</em> $H_0$; it is not the probability <em>of</em> $H_0$. These can differ enormously.</p>
<div class="ex" data-name="Calibrating p = 0.05">
<p>Sellke, Bayarri & Berger (2001) showed that, over a broad class of alternatives, the Bayes factor in favour of a point null satisfies $B_{01} \ge -e\,p\log p$ for $p\lt 1/e$. At $p=0.05$ the bound is $-e(0.05)\log(0.05)\approx 0.41$, so with even prior odds $\P(H_0\mid x)\ge 0.41/1.41 \approx 0.29$. A "significant" result can leave nearly a 30% chance the null is true. At $p=0.005$ the Bayes-factor bound is about $0.072$, and the posterior probability of the null is still at least about $0.067$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Jeffreys–Lindley paradox">
<p>Let $\bar X\sim\N(\mu,\sigma^2/n)$, test $H_0:\mu=0$ with prior mass $\pi_0$ on the null and $\mu\sim\N(0,\tau^2)$ under $H_1$. Hold the $z$ statistic $z = \sqrt n\bar x/\sigma$ fixed. Then $\P(H_0\mid\bar x)\to1$ as $n\to\infty$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Under $H_1$ the marginal of $\bar X$ is $\N(0, \tau^2 + \sigma^2/n)$. The Bayes factor is
$$B_{01} = \frac{\phi(\bar x;0,\sigma^2/n)}{\phi(\bar x;0,\tau^2+\sigma^2/n)} = \sqrt{1 + \frac{n\tau^2}{\sigma^2}}\;\exp\Big\{-\frac{z^2}{2}\cdot\frac{n\tau^2}{n\tau^2+\sigma^2}\Big\}.$$
With $z$ fixed the exponential tends to $e^{-z^2/2}$ while the square root grows like $\sqrt n$, so $B_{01}\to\infty$.</p>
</details>
<p>The same data — say $z=2.5$, $p\approx0.012$ — reject at 5% for every $n$, yet with $n$ huge they favour $H_0$. The resolution: a fixed $z$ at large $n$ corresponds to a tiny estimated effect $\bar x = z\sigma/\sqrt n$, far more plausible under the point null than under a diffuse alternative that spreads mass over effects of size $\tau$.</p>
<div class="warn" data-name="Statistical vs practical significance">
<p>With $n$ large, any non-zero effect, however trivial, eventually gives $p\to0$. A p-value conflates effect size and sample size. Report the estimate and a <a href="#/t/confidence-intervals">confidence interval</a>; test equivalence (two one-sided tests) when "no meaningful difference" is the claim of interest.</p>
</div>`,
      },
      {
        title: 'The standard tests, and the duality with intervals',
        html: R`
<table>
<thead><tr><th>Setting</th><th>Statistic</th><th>Null distribution</th></tr></thead>
<tbody>
<tr><td>Normal mean, $\sigma$ known</td><td>$Z = \sqrt n(\bar X-\mu_0)/\sigma$</td><td>$\N(0,1)$ exactly</td></tr>
<tr><td>Normal mean, $\sigma$ unknown</td><td>$T = \sqrt n(\bar X-\mu_0)/S$</td><td>$t_{n-1}$ exactly</td></tr>
<tr><td>Normal variance</td><td>$(n-1)S^2/\sigma_0^2$</td><td>$\chi^2_{n-1}$ (very non-robust)</td></tr>
<tr><td>Two normal means, equal variances</td><td>pooled $t$</td><td>$t_{n_1+n_2-2}$</td></tr>
<tr><td>Asymptotically normal $\hth$</td><td>Wald $W = (\hth-\theta_0)/\widehat{\se}$</td><td>$\N(0,1)$ asymptotically</td></tr>
<tr><td>Nested parametric models</td><td>$-2\log\lambda$</td><td>$\chi^2_r$ asymptotically (<a href="#/t/likelihood-ratio-tests">Wilks</a>)</td></tr>
<tr><td>Counts in $K$ cells</td><td>Pearson $\sum(O-E)^2/E$</td><td>$\chi^2_{K-1-q}$ asymptotically</td></tr>
<tr><td>Exchangeable groups</td><td>any statistic</td><td><a href="#/t/permutation-tests">permutation</a> distribution, exact</td></tr>
</tbody>
</table>
<p>Tests and <a href="#/t/confidence-intervals">confidence sets</a> are two descriptions of the same object. If $A(\theta_0)$ is the acceptance region of a level-$\alpha$ test of $\theta=\theta_0$, then $C(x) = \{\theta_0 : x\in A(\theta_0)\}$ covers the true $\theta$ with probability at least $1-\alpha$; conversely, "reject $\theta_0$ iff $\theta_0\notin C(x)$" is a level-$\alpha$ test. A 95% interval is the set of null values a 5% test would not reject — which is more informative than any single p-value.</p>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Optional stopping">
<p>Testing after every new observation and stopping at the first $p\lt0.05$ inflates the type I error toward 1: by the law of the iterated logarithm, $|Z_n|$ exceeds any fixed level infinitely often under $H_0$. Sequential designs (group-sequential boundaries, alpha-spending, e-values / anytime-valid p-values) are built to handle this.</p>
</div>
<div class="warn" data-name="Many tests">
<p>With 20 independent true nulls tested at 5%, the chance of at least one rejection is $1-0.95^{20}\approx 0.64$. See <a href="#/t/multiple-testing">multiple testing</a>.</p>
</div>
<div class="warn" data-name="Choosing the test after seeing the data">
<p>Switching from a two-sided to a one-sided test because the estimate came out positive doubles the effective type I error rate. Selecting covariates, outcomes or subgroups by looking at results has the same effect, less visibly ("the garden of forking paths").</p>
</div>
<div class="warn" data-name="Power after the fact">
<p>"Observed power" computed by plugging the estimated effect into the power formula is a deterministic, decreasing function of the p-value and adds no information. Power is a design quantity, computed for effects that matter scientifically.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A test has power function $\beta(\theta)$. How do you read off its size, and why must a test with continuous $\beta$ have power close to $\alpha$ for alternatives near the null boundary?`,
        a: R`Size is $\sup_{\theta\in\Theta_0}\beta(\theta)$. If $\beta$ is continuous and equals (at most) $\alpha$ at boundary points of $\Theta_0$, then for $\theta\in\Theta_1$ close to that boundary $\beta(\theta)$ is close to $\alpha$. No finite-sample test can distinguish alternatives arbitrarily close to the null.` },
      { q: R`Under a simple null with continuous test statistic, $p\sim\text{Unif}(0,1)$. What is the distribution of the p-value for a one-sided $z$ test when the standardised effect is $\delta$?`,
        a: R`$p = 1-\Phi(Z)$ with $Z\sim\N(\delta,1)$. So $\P(p\le u) = \P(Z\ge z_u) = 1-\Phi(z_u-\delta)$ where $z_u = \Phi^{-1}(1-u)$. This is the power at level $u$; for $\delta\gt0$ it exceeds $u$, so the density of $p$ is decreasing and piles up near 0.` },
      { q: R`How many subjects are needed for a one-sided 5% $z$ test to have 90% power against a shift of $0.25\sigma$?`,
        a: R`$n = (z_{0.05}+z_{0.10})^2/0.25^2 = (1.645+1.282)^2/0.0625 = 8.567/0.0625\approx137.1$, so 138.` },
      { q: R`Explain why $p = 0.01$ does not mean "there is a 1% chance the null hypothesis is true".`,
        a: R`The p-value is $\P(T\ge t_{\text{obs}}\mid H_0)$, a probability of data given the hypothesis. $\P(H_0\mid\text{data})$ requires a prior and the likelihood under the alternative; by Bayes’ theorem it can be far larger than $p$ (Sellke–Bayarri–Berger bound, Jeffreys–Lindley paradox).` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'neyman-pearson',
    domain: 'inference',
    order: 120,
    title: 'The Neyman–Pearson Lemma & UMP Tests',
    subtitle: 'Most powerful tests reject where the likelihood ratio is large',
    level: 2,
    sources: { aos: '10.10', cb: '§8.3.2', other: 'Neyman & Pearson (1933); Lehmann & Romano (2005), Ch. 3' },
    prereqs: ['hypothesis-testing', 'likelihood'],
    related: ['likelihood-ratio-tests', 'exponential-families', 'sufficiency', 'bayes-classifier', 'classifier-evaluation'],
    tags: ['most powerful test', 'UMP', 'monotone likelihood ratio', 'Karlin–Rubin', 'randomised test'],
    summary: R`Among all tests with type I error at most $\alpha$, which has the greatest power? For two simple hypotheses the Neyman–Pearson lemma gives a complete answer: reject when the likelihood ratio $f_1(x)/f_0(x)$ exceeds a threshold, randomising on ties to hit $\alpha$ exactly. The proof is three lines and the idea is a greedy knapsack: spend the type I budget on the outcomes that buy the most power per unit of null probability. When the likelihood ratio is monotone in a statistic $T$, the same test is optimal against every alternative on one side (Karlin–Rubin), giving uniformly most powerful one-sided tests; two-sided UMP tests typically do not exist.`,
    survey: {
      task: 'Construct optimal (most powerful) tests and characterise when uniformly most powerful tests exist',
      assumptions: R`Simple hypotheses, or a one-parameter family with monotone likelihood ratio for one-sided composite hypotheses`,
      output: R`Test $\phi(x) = \ind\{f_1(x) \gt kf_0(x)\}$ (+ randomisation); UMP one-sided tests`,
      strengths: 'Exact optimality; explains why standard tests have the form they do; same mathematics as the Bayes classifier and ROC curves',
      weaknesses: 'UMP tests rarely exist beyond one-sided one-parameter problems; randomisation needed for exact size with discrete data',
      complexity: 'Closed form whenever the likelihood ratio distribution is tractable',
    },
    sections: [
      {
        title: 'Most powerful tests as a knapsack problem',
        html: R`
<p>Take the simplest possible testing problem: $H_0: X\sim f_0$ versus $H_1: X\sim f_1$, both densities fully specified. A test $\phi$ has size $\E_0\phi(X) = \int\phi f_0$ and power $\E_1\phi(X) = \int\phi f_1$. We seek
$$\max_{\phi:\,X\to[0,1]}\int\phi\,f_1 \quad\text{subject to}\quad \int\phi\,f_0\le\alpha.$$</p>
<div class="note" data-name="Greedy intuition">
<p>Think of each point $x$ of the sample space as an item. Putting $x$ in the rejection region "costs" $f_0(x)$ of the type I error budget and "earns" $f_1(x)$ of power. This is a fractional knapsack problem, whose optimum is greedy: include items in decreasing order of value-per-cost $f_1(x)/f_0(x)$ until the budget $\alpha$ is used up, taking a fraction of the marginal item if needed. That is the Neyman–Pearson test.</p>
</div>
<div class="def" data-name="Most powerful test">
<p>A test $\phi^*$ of level $\alpha$ is <em>most powerful</em> (MP) for $H_0:\theta=\theta_0$ versus $H_1:\theta=\theta_1$ if $\E_{\theta_1}\phi^*\ge\E_{\theta_1}\phi$ for every test $\phi$ with $\E_{\theta_0}\phi\le\alpha$. For composite $H_1$, a level-$\alpha$ test is <em>uniformly most powerful</em> (UMP) if it is MP against every $\theta_1\in\Theta_1$ simultaneously.</p>
</div>`,
      },
      {
        title: 'The Neyman–Pearson lemma',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Neyman–Pearson lemma (1933)">
<p>Let $f_0, f_1$ be densities (with respect to a common measure) and $0\lt\alpha\lt1$.</p>
<p>(a) <em>Existence.</em> There exist $k\ge0$ and $\gamma\in[0,1]$ such that the test
$$\phi^*(x) = \begin{cases}1 & f_1(x) \gt k f_0(x)\\ \gamma & f_1(x) = k f_0(x)\\ 0 & f_1(x)\lt k f_0(x)\end{cases}$$
has $\E_0\phi^* = \alpha$.</p>
<p>(b) <em>Sufficiency.</em> Any test of this form with size $\alpha$ is most powerful at level $\alpha$.</p>
<p>(c) <em>Necessity.</em> If $\phi$ is MP at level $\alpha$, then $\phi=\phi^*$ almost everywhere on $\{f_1\ne kf_0\}$, and $\phi$ has size $\alpha$ unless it has power 1.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><strong>(b)</strong> Let $\phi$ be any test with $\E_0\phi\le\alpha$. At every $x$,
$$\big(\phi^*(x)-\phi(x)\big)\big(f_1(x)-kf_0(x)\big)\ge0,$$
because where $f_1\gt kf_0$ we have $\phi^*=1\ge\phi$, where $f_1\lt kf_0$ we have $\phi^*=0\le\phi$, and where they are equal the second factor is 0. Integrating,
$$\E_1\phi^* - \E_1\phi \;\ge\; k\big(\E_0\phi^* - \E_0\phi\big) = k(\alpha - \E_0\phi)\;\ge\;0.$$</p>
<p><strong>(a)</strong> Let $L = f_1(X)/f_0(X)$ (with $L=\infty$ where $f_0=0$) and $c(k) = \P_0(L\gt k)$. This is non-increasing and right-continuous in $k$, with $c(0^-)=1$ and $c(k)\to0$. Pick $k$ with $c(k)\le\alpha\le c(k^-) = \P_0(L\ge k)$. If $\P_0(L=k)=0$ take $\gamma=0$; otherwise $\gamma = (\alpha - c(k))/\P_0(L=k)\in[0,1]$. Then $\E_0\phi^* = c(k) + \gamma\P_0(L=k) = \alpha$.</p>
<p><strong>(c)</strong> If $\phi$ is also MP, the integrated inequality in (b) is an equality, so the non-negative integrand $(\phi^*-\phi)(f_1-kf_0)$ vanishes a.e., forcing $\phi=\phi^*$ where $f_1\ne kf_0$; and $k(\alpha - \E_0\phi)=0$ forces size $\alpha$ when $k\gt0$.</p>
</details>
<p>Three corollaries follow at once.</p>
<ul>
<li><strong>Sufficiency.</strong> If $T$ is sufficient, the factorisation theorem makes $f_1/f_0$ a function of $T$, so the MP test depends on the data only through $T$.</li>
<li><strong>Unbiasedness.</strong> Comparing $\phi^*$ with the trivial test $\phi\equiv\alpha$ shows $\E_1\phi^*\ge\alpha$: power is at least size, with equality only if $f_0=f_1$.</li>
<li><strong>Bayes connection.</strong> The Bayes classifier between two classes with priors $\pi_0,\pi_1$ assigns $x$ to class 1 when $f_1(x)/f_0(x)\gt\pi_0/\pi_1$ — the same likelihood-ratio form with a different threshold. Sweeping $k$ traces out the optimal ROC curve; no classifier has a higher true-positive rate at any false-positive rate (see <a href="#/t/bayes-classifier">the Bayes classifier</a> and <a href="#/t/classifier-evaluation">ROC analysis</a>).</li>
</ul>`,
        lab: 'neyman-pearson-region',
      },
      {
        title: 'Worked examples',
        html: R`
<div class="ex" data-name="Normal mean">
<p>$X_1,\dots,X_n\iid\N(\mu,\sigma^2)$, $\sigma$ known, $H_0:\mu=\mu_0$ vs $H_1:\mu=\mu_1$ with $\mu_1\gt\mu_0$. The log likelihood ratio is
$$\log\frac{f_1}{f_0} = \frac{\mu_1-\mu_0}{\sigma^2}\sum_i x_i - \frac{n(\mu_1^2-\mu_0^2)}{2\sigma^2},$$
an increasing function of $\bar x$. So "LR $\gt k$" is "$\bar x\gt c$", and size $\alpha$ forces $c = \mu_0 + z_\alpha\sigma/\sqrt n$. The rejection region <strong>does not depend on $\mu_1$</strong>: the same test is MP against every $\mu_1\gt\mu_0$, hence UMP for $H_1:\mu\gt\mu_0$.</p>
</div>
<div class="ex" data-name="Binomial and why randomisation appears">
<p>$X\sim\text{Bin}(10,p)$, $H_0: p=\tfrac12$ vs $H_1:p=p_1\gt\tfrac12$. The LR $\big(\tfrac{p_1}{1-p_1}\big)^x\big(\tfrac{1/2}{1-p_1}\big)^{10}\cdot$const is increasing in $x$, so we reject for large $x$. Under $H_0$, $\P(X\ge9) = 11/1024\approx0.0107$ and $\P(X\ge8) = 56/1024\approx0.0547$. No non-randomised test has size exactly $0.05$. The MP test rejects when $X\ge9$ and, when $X=8$, rejects with probability $\gamma = (0.05\cdot1024 - 11)/45 \approx 0.893$. In practice one reports the conservative test ($X\ge9$, size $0.011$) or a p-value; the randomised test exists to make the optimality theory clean.</p>
</div>
<div class="ex" data-name="Normal scale: a two-sided region from a one-sided problem">
<p>$X\sim\N(0,\sigma^2)$, $H_0:\sigma=1$ vs $H_1:\sigma=2$. The LR is $\tfrac12\exp\{\tfrac38x^2\}$, increasing in $|x|$, so the MP region is $\{|x|\gt c\}$ with $c=z_{\alpha/2}$: two tails, even though the alternative is "one-sided" in $\sigma$. The shape of the optimal region is dictated by the likelihood ratio, not by the geometry of the hypotheses. Try this in the lab.</p>
</div>`,
      },
      {
        title: 'Monotone likelihood ratio and the Karlin–Rubin theorem',
        html: R`
<p>The normal-mean example generalises: whenever the LR is monotone in one statistic, the MP region against $\theta_1$ does not depend on which $\theta_1\gt\theta_0$ is chosen.</p>
<div class="def" data-name="Monotone likelihood ratio (MLR)">
<p>A family $\{f(x\mid\theta):\theta\in\Theta\subseteq\R\}$ has MLR in a real statistic $T(x)$ if for every $\theta_1\lt\theta_2$ the ratio $f(x\mid\theta_2)/f(x\mid\theta_1)$ is a non-decreasing function of $T(x)$ on $\{x: f(x\mid\theta_1)\gt0 \text{ or } f(x\mid\theta_2)\gt0\}$.</p>
</div>
<p>One-parameter <a href="#/t/exponential-families">exponential families</a> $f(x\mid\theta) = h(x)c(\theta)e^{w(\theta)T(x)}$ with $w$ non-decreasing have MLR in $T$, since the ratio is $\frac{c(\theta_2)}{c(\theta_1)}e^{(w(\theta_2)-w(\theta_1))T(x)}$. This covers the binomial, Poisson, normal mean, normal variance (in $\sum(x_i-\mu)^2$), exponential and gamma-scale families. The $\text{Unif}(0,\theta)$ family has MLR in $X_{(n)}$, and the noncentral $t$ family has MLR in $t$; the Cauchy location family does <em>not</em> have MLR.</p>
<div class="thm" data-kind="Theorem" data-name="Karlin–Rubin">
<p>Suppose $\{f(x\mid\theta)\}$ has MLR in $T$. For testing $H_0:\theta\le\theta_0$ versus $H_1:\theta\gt\theta_0$, the test
$$\phi(x) = \begin{cases}1 & T(x)\gt t_0\\ \gamma & T(x)=t_0\\ 0 & T(x)\lt t_0\end{cases}\qquad\text{with }\E_{\theta_0}\phi = \alpha$$
is UMP of level $\alpha$. Moreover its power function is non-decreasing in $\theta$ (strictly where it lies in $(0,1)$).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Step 1 (MP against each point alternative).</em> Fix $\theta_1\gt\theta_0$. By MLR, $\{T\gt t_0\}\subseteq\{f_{\theta_1}/f_{\theta_0}\ge k\}$ and $\{T\lt t_0\}\subseteq\{f_{\theta_1}/f_{\theta_0}\le k\}$ for $k$ the ratio’s value at $T=t_0$, so $\phi$ is a Neyman–Pearson test of $\theta_0$ vs $\theta_1$ with size $\alpha$ (the lemma’s sufficiency argument only needs $\phi=1$ where the ratio exceeds $k$ and $\phi=0$ where it is below $k$). Hence $\phi$ is MP for each $\theta_1\gt\theta_0$.</p>
<p><em>Step 2 (monotone power).</em> For any $\theta'\lt\theta''$, the same reasoning shows $\phi$ is MP for $\theta'$ vs $\theta''$ at its own size $\beta(\theta')$; by the unbiasedness corollary, $\beta(\theta'')\ge\beta(\theta')$.</p>
<p><em>Step 3 (composite null).</em> By step 2, $\sup_{\theta\le\theta_0}\beta(\theta) = \beta(\theta_0)=\alpha$, so $\phi$ has level $\alpha$ for the composite null. Any competitor of level $\alpha$ for $\theta\le\theta_0$ is in particular level $\alpha$ for $\theta=\theta_0$, so by step 1 it cannot beat $\phi$ at any $\theta_1\gt\theta_0$.</p>
</details>
<div class="ex" data-name="Poisson rate">
<p>$X_i\iid\text{Poisson}(\lambda)$: MLR in $S=\sum X_i$, so the UMP test of $\lambda\le\lambda_0$ vs $\lambda\gt\lambda_0$ rejects for large $S$, with $S\sim\text{Poisson}(n\lambda_0)$ at the boundary. Exponential lifetimes with mean $\theta$: MLR in $\sum X_i$, and $2\sum X_i/\theta_0\sim\chi^2_{2n}$ gives exact critical values.</p>
</div>`,
      },
      {
        title: 'When UMP tests do not exist',
        html: R`
<div class="thm" data-kind="Proposition" data-name="No UMP two-sided test for a normal mean">
<p>For $X_1,\dots,X_n\iid\N(\mu,\sigma^2)$, $\sigma$ known, and $0\lt\alpha\lt1$, there is no UMP level-$\alpha$ test of $H_0:\mu=\mu_0$ versus $H_1:\mu\ne\mu_0$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Suppose $\phi$ were UMP. Against $\mu_1\gt\mu_0$ the MP test is $\ind\{\bar x\gt\mu_0+z_\alpha\sigma/\sqrt n\}$, and by part (c) of the lemma any MP test equals it almost everywhere (ties have probability zero here). Against $\mu_2\lt\mu_0$ the MP test is $\ind\{\bar x\lt\mu_0 - z_\alpha\sigma/\sqrt n\}$, again essentially unique. A UMP $\phi$ would have to equal both a.e., but these two regions are disjoint and each has positive probability. Contradiction.</p>
</details>
<p>The one-sided upper test has excellent power for $\mu\gt\mu_0$ and power below $\alpha$ (indeed tending to 0) for $\mu\lt\mu_0$ — it is <em>biased</em>. Restoring optimality requires restricting the class of tests:</p>
<ul>
<li><strong>Unbiasedness.</strong> Require $\beta(\theta)\ge\alpha$ on $\Theta_1$. Among unbiased tests, the symmetric two-sided $z$ test is UMP (UMPU). More generally, in one-parameter exponential families the UMPU two-sided test rejects outside an interval $[c_1,c_2]$ chosen so that the size is $\alpha$ and the power function has zero derivative at $\theta_0$. For multiparameter exponential families, conditioning on the sufficient statistics for nuisance parameters yields UMPU tests — this is the theoretical origin of the $t$ test and Fisher’s exact test.</li>
<li><strong>Invariance.</strong> Restrict to tests invariant under a group that leaves the problem unchanged (e.g. rescaling); UMP invariant tests include the $F$ test in linear models.</li>
<li><strong>Local optimality.</strong> Maximise the slope $\beta'(\theta_0)$ of the power function: the locally most powerful test rejects for large values of the score $\partial_\theta\log f(x\mid\theta_0)$ — the <a href="#/t/likelihood-ratio-tests">score test</a>.</li>
<li><strong>Give up exact optimality.</strong> The <a href="#/t/likelihood-ratio-tests">generalised likelihood ratio test</a> replaces the unknown $\theta_1$ by its MLE and is asymptotically optimal in a local sense.</li>
</ul>`,
      },
      {
        title: 'The ROC view',
        html: R`
<p>For a family of tests indexed by threshold, plot power $\E_1\phi$ against size $\E_0\phi$. The Neyman–Pearson lemma says the likelihood-ratio family traces the <em>upper envelope</em>: every other test lies on or below its curve. Three geometric facts follow from the lemma:</p>
<ul>
<li>The optimal curve is <strong>concave</strong> — randomising between two LR tests gives any point on the chord, so the achievable set is convex and its upper boundary concave.</li>
<li>Its <strong>slope at a point equals the threshold</strong> $k$ used there: $\d(\text{power})/\d(\text{size}) = k$, because the marginal outcomes added as the threshold drops have $f_1/f_0 = k$.</li>
<li>The diagonal is the power of a coin-flip test $\phi\equiv\alpha$; the area between the optimal curve and the diagonal measures how distinguishable $f_0$ and $f_1$ are. In the large-sample limit, Stein’s lemma says the best type II error at fixed $\alpha$ decays like $e^{-n\KL(f_0\|f_1)}$ (see <a href="#/t/entropy-kl">KL divergence</a>).</li>
</ul>
<p>The lab’s right panel draws the LR test’s ROC curve next to that of a naive "reject for large $x$" test. When $f_1/f_0$ is monotone in $x$ they coincide; when it is not (scale alternatives, heavy-tailed alternatives) the naive test is strictly dominated.</p>`,
      },
    ],
    quiz: [
      { q: R`In the proof of the Neyman–Pearson lemma, why is $(\phi^*(x)-\phi(x))(f_1(x)-kf_0(x))\ge0$ for every $x$?`,
        a: R`Where $f_1\gt kf_0$, $\phi^*=1$ so $\phi^*-\phi\ge0$ and the second factor is positive. Where $f_1\lt kf_0$, $\phi^*=0$ so $\phi^*-\phi\le0$ and the second factor is negative. Where they are equal the product is zero. Integrating then gives power$(\phi^*)-$power$(\phi)\ge k(\alpha-\E_0\phi)\ge0$.` },
      { q: R`For $X_1,\dots,X_n\iid\text{Unif}(0,\theta)$, describe the UMP test of $H_0:\theta\le\theta_0$ against $H_1:\theta\gt\theta_0$ at level $\alpha$.`,
        a: R`The family has MLR in $X_{(n)}$ (the ratio $f_{\theta_2}/f_{\theta_1}$ equals $(\theta_1/\theta_2)^n$ for $X_{(n)}\le\theta_1$ and $+\infty$ for $\theta_1\lt X_{(n)}\le\theta_2$). By Karlin–Rubin, reject when $X_{(n)}\gt c$ with $\P_{\theta_0}(X_{(n)}\gt c) = 1-(c/\theta_0)^n = \alpha$, i.e. $c = \theta_0(1-\alpha)^{1/n}$.` },
      { q: R`Why does no UMP test exist for $H_0:\mu=\mu_0$ vs $H_1:\mu\neq\mu_0$ in the normal model, and what restriction recovers optimality?`,
        a: R`The essentially unique MP tests against $\mu_1\gt\mu_0$ and against $\mu_1\lt\mu_0$ reject in opposite tails, so no single test can be MP against both. Restricting to unbiased tests ($\beta\ge\alpha$ on $H_1$) makes the symmetric two-sided $z$ test UMP unbiased.` },
      { q: R`Testing $f_0=\N(0,1)$ against $f_1=\N(0,4)$ from one observation, what does the MP region look like and what is its power at $\alpha=0.05$?`,
        a: R`The LR is increasing in $|x|$, so reject when $|x|\gt1.96$. Under $f_1$, $X/2\sim\N(0,1)$, so power is $\P(|N|\gt0.98) = 2(1-\Phi(0.98))\approx0.327$.` },
    ],
  });
  /* ================================================================== */
  Atlas.topic({
    id: 'likelihood-ratio-tests',
    domain: 'inference',
    order: 130,
    title: 'Likelihood Ratio, Wald & Score Tests',
    subtitle: 'Three asymptotically equivalent ways to ask whether the likelihood is far from the null',
    level: 2,
    sources: { aos: '10.3–10.6', cb: '§8.2.1, §10.3', other: 'Wilks (1938); Rao (1948); Brown, Cai & DasGupta (2001)' },
    prereqs: ['mle', 'fisher-information', 'hypothesis-testing'],
    related: ['neyman-pearson', 'confidence-intervals', 'deviance', 'glm-inference', 'delta-method'],
    tags: ['GLRT', 'Wilks theorem', 'Wald test', 'score test', 'chi-squared', 'Wilson interval'],
    summary: R`When no uniformly most powerful test exists, the generalised likelihood ratio test (GLRT) compares the best fit under the null with the best fit overall. Wilks’ theorem gives the universal answer for its null distribution in regular models: $-2\log\lambda \dto \chi^2_r$, with $r$ the number of restrictions. The Wald test measures the distance from $\hth$ to the null in standard-error units; the score test measures the slope of the log-likelihood at the null. All three are quadratic approximations to the same log-likelihood and agree to first order — but they differ in finite samples, in invariance, and in how badly they fail, as the binomial confidence interval shows.`,
    survey: {
      task: 'General-purpose tests (and, by inversion, intervals) for parametric hypotheses, including composite nulls with nuisance parameters',
      assumptions: R`Regular model: identifiable, smooth log-likelihood, interior true value, nonsingular Fisher information, nested hypotheses`,
      output: R`Statistics $-2\log\lambda$, $W$, $S$, each $\approx\chi^2_r$ under $H_0$; p-values; inverted intervals`,
      strengths: 'Universal recipe; nuisance parameters handled by profiling; LR and score are invariant to reparameterisation',
      weaknesses: 'Only asymptotic; Wilks fails at boundaries, under non-identifiability, and as dimension grows; Wald is not invariant',
      complexity: R`One (Wald, score) or two (LR) maximisations of the likelihood`,
    },
    sections: [
      {
        title: 'The generalised likelihood ratio',
        html: R`
<p>The Neyman–Pearson lemma tells us to reject when $f_1(x)/f_0(x)$ is large — but with composite hypotheses neither density is known. The natural surrogate replaces each by its best-fitting member.</p>
<div class="def" data-name="Likelihood ratio statistic">
<p>For $H_0:\theta\in\Theta_0$ versus $H_1:\theta\in\Theta\setminus\Theta_0$, the likelihood ratio statistic is
$$\lambda(x) = \frac{\sup_{\theta\in\Theta_0}\mathcal L(\theta)}{\sup_{\theta\in\Theta}\mathcal L(\theta)} = \frac{\mathcal L(\hth_0)}{\mathcal L(\hth)} \in[0,1],$$
where $\hth_0$ is the restricted MLE and $\hth$ the unrestricted <a href="#/t/mle">MLE</a>. The GLRT rejects when $\lambda$ is small, i.e. when $\Lambda = -2\log\lambda = 2\{\ell(\hth) - \ell(\hth_0)\}$ is large.</p>
</div>
<p>Using the full space $\Theta$ in the denominator rather than $\Theta_1$ changes nothing when $\Theta_0$ has lower dimension, and it makes $\lambda\le1$ and the asymptotics cleaner. Many classical tests turn out to be GLRTs in disguise.</p>
<div class="ex" data-name="The t test is a likelihood ratio test">
<p>$X_1,\dots,X_n\iid\N(\mu,\sigma^2)$ with both unknown; $H_0:\mu=\mu_0$. Unrestricted: $\hat\mu = \bar x$, $\hat\sigma^2 = \frac1n\sum(x_i-\bar x)^2$. Restricted: $\hat\sigma_0^2 = \frac1n\sum(x_i-\mu_0)^2 = \hat\sigma^2 + (\bar x-\mu_0)^2$. Plugging in, the exponential factors cancel and
$$\lambda = \Big(\frac{\hat\sigma^2}{\hat\sigma_0^2}\Big)^{n/2} = \Big(1 + \frac{T^2}{n-1}\Big)^{-n/2},\qquad T = \frac{\sqrt n(\bar x-\mu_0)}{S}.$$
Small $\lambda$ is large $|T|$: the GLRT is exactly the two-sided $t$ test, and its exact null distribution is known. Here we did not need asymptotics; Wilks’ theorem is what we use when we cannot do this algebra.</p>
</div>
<p>Nuisance parameters are eliminated by maximisation: if $\theta = (\psi,\eta)$ and $H_0:\psi=\psi_0$, then $\ell(\hth_0) = \sup_\eta\ell(\psi_0,\eta) =: \ell_p(\psi_0)$, the <em>profile log-likelihood</em>. So $\Lambda = 2\{\ell_p(\hat\psi) - \ell_p(\psi_0)\}$.</p>`,
      },
      {
        title: 'Wilks’ theorem',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Wilks (1938)">
<p>Let $X_1,\dots,X_n\iid f(x\mid\theta)$, $\theta\in\Theta\subseteq\R^k$ open, satisfying the usual MLE regularity conditions (identifiability, three times differentiable log-density with dominated third derivatives, interchange of differentiation and integration, nonsingular $I(\theta)$). Let $\Theta_0$ be a smooth $(k-r)$-dimensional submanifold (equivalently, $r$ independent smooth restrictions), and let the true $\theta^*$ be an interior point of $\Theta_0$. Then
$$\Lambda_n = -2\log\lambda(X^n) \dto \chi^2_r.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Scalar case</em> ($k=r=1$, $\Theta_0=\{\theta_0\}$). Expand $\ell$ around the MLE, where $\ell'(\hth)=0$:
$$\ell(\theta_0) = \ell(\hth) + \tfrac12\ell''(\tilde\theta)(\theta_0-\hth)^2$$
for some $\tilde\theta$ between them. Hence
$$\Lambda_n = -\ell''(\tilde\theta)(\hth-\theta_0)^2 = \Big[\frac{-\ell''(\tilde\theta)}{n I(\theta_0)}\Big]\cdot\big[\sqrt{nI(\theta_0)}(\hth-\theta_0)\big]^2.$$
The first bracket $\pto1$ because $\hth\pto\theta_0$ and $-\frac1n\ell''(\theta)\to I(\theta)$ uniformly near $\theta_0$ (LLN plus the dominating function for the third derivative). The second bracket is the square of a quantity that is asymptotically $\N(0,1)$ by asymptotic normality of the MLE. By Slutsky and the continuous mapping theorem, $\Lambda_n\dto Z^2\sim\chi^2_1$.</p>
<p><em>General case.</em> Write $Z_n = I(\theta^*)^{-1/2}\,n^{-1/2}\nabla\ell(\theta^*)\dto\N(0,I_k)$. Locally, the log-likelihood is a quadratic $\ell(\theta^*+h/\sqrt n) - \ell(\theta^*) \approx h^\top I^{1/2}Z_n - \tfrac12h^\top I h$. Maximising this quadratic over all $h$ gives $\tfrac12\|Z_n\|^2$; maximising over $h$ in the tangent space of $\Theta_0$ (a $(k-r)$-dimensional linear subspace after the local change of variables) gives $\tfrac12\|P Z_n\|^2$, where $P$ is the orthogonal projection onto the transformed subspace. So $\Lambda_n\approx\|Z_n\|^2-\|PZ_n\|^2 = \|(I-P)Z_n\|^2$, and $(I-P)$ is a projection of rank $r$, so the limit is $\chi^2_r$.</p>
</details>
<p>The degrees of freedom are a <strong>dimension count</strong>: free parameters under the full model minus free parameters under the null. For the multinomial goodness-of-fit test with $K$ cells and a $q$-parameter null model, $r = (K-1)-q$. For independence in an $a\times b$ table, $r = (ab-1) - (a-1) - (b-1) = (a-1)(b-1)$.</p>
<div class="note" data-name="Why a chi-square">
<p>Near the truth every regular log-likelihood looks like the log-likelihood of a normal mean with known covariance. For that model $-2\log\lambda$ is <em>exactly</em> a squared Mahalanobis distance from the data to the null subspace, which is $\chi^2_r$. Wilks’ theorem is local asymptotic normality plus Pythagoras.</p>
</div>`,
      },
      {
        title: 'Wald and score tests, and their equivalence',
        html: R`
<p>The log-likelihood ratio uses two fits. Two cheaper statistics need only one.</p>
<div class="cols">
<div><h4>Wald</h4>$$W = (\hth-\theta_0)^\top\,\big[n\,I(\hth)\big]\,(\hth-\theta_0)$$<p>Fit the full model only. Distance from the MLE to the null, in estimated standard errors. Scalar case: $W = \big((\hth-\theta_0)/\widehat{\se}\big)^2$.</p></div>
<div><h4>Score (Rao / Lagrange multiplier)</h4>$$S = U(\theta_0)^\top\big[n\,I(\theta_0)\big]^{-1}U(\theta_0)$$<p>Fit the null model only; $U = \nabla\ell$. How steeply the log-likelihood is still rising at the null. With nuisance parameters, evaluate at $\hth_0$.</p></div>
<div><h4>Likelihood ratio</h4>$$\Lambda = 2\{\ell(\hth)-\ell(\hth_0)\}$$<p>Fit both. The vertical drop in log-likelihood between the best fit and the best null fit.</p></div>
</div>
<p>Geometrically, for a scalar parameter: $W$ is a horizontal distance (on the $\theta$ axis, scaled by curvature at $\hth$), $S$ is a slope (at $\theta_0$, scaled by curvature there), and $\Lambda$ is a height. For an exactly quadratic log-likelihood the three coincide. The first lab draws all three on a real log-likelihood.</p>
<div class="thm" data-kind="Theorem" data-name="Asymptotic equivalence of the trinity">
<p>Under the conditions of Wilks’ theorem and $H_0$, $W - \Lambda\pto0$ and $S-\Lambda\pto0$, so all three are asymptotically $\chi^2_r$. Under local alternatives $\theta_n = \theta_0 + h/\sqrt n$, all three converge to the same noncentral $\chi^2_r(h^\top I(\theta_0)h)$ law, so they have the same asymptotic local power.</p>
</div>
<details class="proof"><summary>Proof sketch (scalar)</summary>
<p>From the scalar Wilks argument, $\Lambda = n I(\theta_0)(\hth-\theta_0)^2 + o_p(1)$. Since $I(\hth)\pto I(\theta_0)$, $W = nI(\hth)(\hth-\theta_0)^2$ differs by $o_p(1)$. For the score, expand $0 = \ell'(\hth) = \ell'(\theta_0) + \ell''(\bar\theta)(\hth-\theta_0)$, giving $U(\theta_0) = nI(\theta_0)(\hth-\theta_0)(1+o_p(1))$, so $S = U^2/(nI(\theta_0)) = nI(\theta_0)(\hth-\theta_0)^2 + o_p(1)$. Under contiguous local alternatives, $o_p(1)$ terms stay $o_p(1)$ (Le Cam’s third lemma), and $\sqrt{nI}(\hth-\theta_0)\dto\N(h\sqrt I,1)$.</p>
</details>
<table>
<thead><tr><th>Property</th><th>Wald</th><th>Score</th><th>LR</th></tr></thead>
<tbody>
<tr><td>Models fitted</td><td>full</td><td>null</td><td>both</td></tr>
<tr><td>Invariant to reparameterisation $\phi=g(\theta)$</td><td><strong>no</strong></td><td>yes</td><td>yes</td></tr>
<tr><td>Uses information at</td><td>$\hth$</td><td>$\theta_0$</td><td>— (no information matrix)</td></tr>
<tr><td>Inverted interval</td><td>symmetric, can leave $\Theta$</td><td>respects $\Theta$ (Wilson)</td><td>respects $\Theta$; follows likelihood shape</td></tr>
<tr><td>Typical small-sample behaviour</td><td>worst</td><td>good</td><td>good</td></tr>
</tbody>
</table>
<div class="warn" data-name="Wald’s non-invariance in practice">
<p>Testing $p = \tfrac12$ via $\hat p$ or via $\log\{\hat p/(1-\hat p)\}$ gives different Wald statistics from the same data. In logistic regression, the <em>Hauck–Donner effect</em> (1977) makes the Wald statistic for a coefficient <em>decrease</em> as the estimate moves far from zero, because the estimated standard error blows up faster than the estimate — very strong effects can look insignificant. LR and score tests do not suffer from this.</p>
</div>`,
        lab: 'likelihood-ratio-tests-geometry',
      },
      {
        title: 'Case study: intervals for a binomial proportion',
        html: R`
<p>Every test inverts to a <a href="#/t/confidence-intervals">confidence interval</a>. For $X\sim\text{Bin}(n,p)$, $\hat p = X/n$, the trinity gives:</p>
<ul>
<li><strong>Wald:</strong> $\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$. The textbook interval.</li>
<li><strong>Score (Wilson, 1927):</strong> $\{p: |\hat p - p|\le z\sqrt{p(1-p)/n}\}$. Solving the quadratic,
$$\frac{\hat p + \frac{z^2}{2n}}{1+\frac{z^2}{n}} \;\pm\; \frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}.$$
The centre is pulled toward $\tfrac12$, as if adding $z^2/2\approx2$ successes and 2 failures.</li>
<li><strong>Likelihood ratio:</strong> $\{p : 2[\ell(\hat p)-\ell(p)]\le\chi^2_{1,1-\alpha}\}$, found numerically.</li>
<li><strong>Clopper–Pearson (1934)</strong> inverts the <em>exact</em> binomial tests: endpoints are Beta quantiles, $[B_{\alpha/2}(x, n-x+1),\, B_{1-\alpha/2}(x+1,n-x)]$. Coverage is guaranteed $\ge1-\alpha$ for every $p$ — at the cost of being conservative.</li>
</ul>
<div class="warn" data-name="The Wald interval’s failure">
<p>Brown, Cai & DasGupta (2001) documented that the Wald interval’s coverage is poor not only for small $n$ or $p$ near the boundary, but erratically for all $n$: coverage oscillates as $n$ and $p$ vary because of the lattice structure of $X$, with "unlucky" $(n,p)$ pairs well below nominal even when $np$ and $n(1-p)$ are both large. Near the boundary it is catastrophic: if $X=0$ the interval is the single point $\{0\}$, so for $p$ small enough that $\P(X=0) = (1-p)^n$ is large, coverage collapses toward 0. The root cause is that the Wald test uses the curvature at $\hat p$, which is zero-variance at the boundary, rather than at the hypothesised $p$.</p>
</div>
<p>The second lab computes the exact coverage function $C(p) = \sum_{x=0}^n\binom nx p^x(1-p)^{n-x}\ind\{p\in I(x)\}$ for each interval, so you can see the oscillations directly. The practical conclusion, now standard: never use the Wald interval for a proportion; use Wilson (or Agresti–Coull, its simplified cousin) or Jeffreys’ Bayesian interval.</p>`,
        lab: 'likelihood-ratio-tests-coverage',
      },
      {
        title: 'Goodness of fit: Pearson’s chi-square as a score test',
        html: R`
<p>For counts $O_1,\dots,O_K$ from a multinomial with probabilities $p_1,\dots,p_K$, and a null specifying $p_j = p_j(\eta)$ with $q$ free parameters estimated by maximum likelihood, let $E_j = n\,p_j(\hat\eta)$. The two classic statistics are
$$G^2 = 2\sum_j O_j\log\frac{O_j}{E_j}\quad(\text{likelihood ratio}),\qquad X^2 = \sum_j\frac{(O_j-E_j)^2}{E_j}\quad(\text{Pearson}).$$</p>
<p>$G^2$ is the LR statistic of the null against the saturated multinomial. Pearson’s $X^2$ is the score statistic for the same hypotheses; a second-order Taylor expansion of $O\log(O/E)$ around $O=E$ shows $G^2 = X^2 + o_p(1)$. Both are asymptotically $\chi^2_{K-1-q}$. The approximation needs expected counts that are not too small (the folklore "all $E_j\ge5$" is conservative; Cochran’s rules are more nuanced); $X^2$ is usually closer to its limit than $G^2$ for sparse tables.</p>
<div class="ex" data-name="Degrees of freedom when parameters are estimated">
<p>Testing whether counts in 6 bins come from a Poisson with unknown mean: $K=6$, $q=1$, so $5-1 = 4$ degrees of freedom. Using $\chi^2_5$ instead would ignore that estimating $\lambda$ already pulled the $E_j$ toward the $O_j$, making the test conservative. (The correct reduction requires $\hat\eta$ to be the MLE from the <em>grouped</em> counts; using the ungrouped MLE gives a limit between $\chi^2_{K-1-q}$ and $\chi^2_{K-1}$ — Chernoff & Lehmann, 1954.)</p>
</div>`,
      },
      {
        title: 'When Wilks fails',
        html: R`
<div class="warn" data-name="Parameter on the boundary">
<p>Let $X_i\iid\N(\mu,1)$ with $\Theta=[0,\infty)$, and test $H_0:\mu=0$. The MLE is $\max(\bar X,0)$, so $\Lambda_n = n\bar X^2\,\ind\{\bar X\gt0\}$, whose exact null law is $\tfrac12\delta_0+\tfrac12\chi^2_1$ — a 50:50 mixture of a point mass at 0 and $\chi^2_1$. Using $\chi^2_1$ critical values halves the actual size. The same phenomenon governs tests of a variance component equal to zero in <a href="#/t/mixed-models">mixed models</a> (Self & Liang, 1987).</p>
</div>
<div class="warn" data-name="Parameters not identified under the null">
<p>Testing one normal component against a two-component mixture: under $H_0$ the mixing weight is 0 and the second component’s mean is then meaningless. The regularity conditions fail; $-2\log\lambda$ does not converge to any $\chi^2$, and in some versions diverges slowly (like $\log\log n$). Change-point and threshold models share the problem (Davies, 1977, 1987). Use parametric bootstrap calibration.</p>
</div>
<div class="warn" data-name="Many parameters">
<p>If the number of parameters $k$ grows with $n$, Wilks’ approximation degrades; for logistic regression with $k/n\to\kappa\gt0$, the LR statistic is asymptotically a <em>rescaled</em> $\chi^2$ with a factor greater than 1 (Sur, Chen & Candès, 2019), so nominal p-values are anti-conservative.</p>
</div>
<div class="warn" data-name="Non-nested models">
<p>Comparing two models neither of which contains the other (log-normal vs gamma) is outside Wilks’ theorem. Use Vuong’s test, information criteria (<a href="#/t/model-selection">model selection</a>), or cross-validation.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In a $3\times4$ contingency table, how many degrees of freedom does the likelihood ratio test of independence have, and where does the number come from?`,
        a: R`The saturated model has $12-1=11$ free cell probabilities. Independence $p_{ij}=a_ib_j$ has $(3-1)+(4-1)=5$. Wilks’ $r = 11-5 = 6 = (3-1)(4-1)$.` },
      { q: R`Why is the Wald interval for a binomial proportion degenerate when $X=0$, while the Wilson interval is not?`,
        a: R`Wald uses the estimated variance $\hat p(1-\hat p)/n$ at $\hat p=0$, which is 0, so the interval is $\{0\}$. Wilson inverts the score test, which uses the variance $p(1-p)/n$ at each hypothesised $p$; for $X=0$ it accepts all $p$ with $p\le z\sqrt{p(1-p)/n}$, i.e. $p\le z^2/(n+z^2)$, a non-degenerate interval.` },
      { q: R`Show that for $X_1,\dots,X_n\iid\N(\mu,1)$ and $H_0:\mu=\mu_0$, the Wald, score and LR statistics are identical.`,
        a: R`$\ell(\mu) = -\tfrac12\sum(x_i-\mu)^2$, $I=1$, $\hat\mu = \bar x$. LR: $2[\ell(\bar x)-\ell(\mu_0)] = n(\bar x-\mu_0)^2$. Wald: $n\cdot1\cdot(\bar x-\mu_0)^2$. Score: $U(\mu_0) = n(\bar x-\mu_0)$, so $U^2/n = n(\bar x-\mu_0)^2$. The log-likelihood is exactly quadratic, so all three coincide and are exactly $\chi^2_1$.` },
      { q: R`You test $H_0:\sigma^2_u = 0$ for a random-effect variance with the LR statistic and compare it to $\chi^2_1$. Is your test conservative or anti-conservative?`,
        a: R`Conservative. The null value lies on the boundary of $[0,\infty)$, and the null distribution is approximately $\tfrac12\chi^2_0+\tfrac12\chi^2_1$. Its upper tail probability at a given cut-off is half that of $\chi^2_1$, so using $\chi^2_1$ gives p-values twice too large.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'confidence-intervals',
    domain: 'inference',
    order: 140,
    title: 'Confidence Intervals & Pivotal Quantities',
    subtitle: 'Random sets that trap a fixed parameter with guaranteed frequency',
    level: 2,
    sources: { aos: '6.3.2', cb: '§9.1–9.3', other: 'Neyman (1937); Pratt (1961); Casella & Berger (2002), Ch. 9' },
    prereqs: ['hypothesis-testing', 'sampling-distributions'],
    related: ['likelihood-ratio-tests', 'bootstrap', 'bayesian-inference', 'delta-method', 'bca-intervals'],
    tags: ['coverage', 'pivot', 'test inversion', 'shortest interval', 'Wald interval'],
    summary: R`A $1-\alpha$ confidence set is a data-dependent set $C(X)$ with $\P_\theta(\theta\in C(X))\ge1-\alpha$ for every $\theta$. The randomness is in the set, not the parameter. There are two universal constructions: find a <em>pivot</em> — a function of data and parameter whose distribution is free of the parameter — and solve it for $\theta$; or <em>invert a family of tests</em>, collecting every null value a level-$\alpha$ test would not reject. Every reasonable interval is one of these. Among valid intervals we prefer short ones, and Pratt’s identity shows that short expected length is the same thing as rarely covering false values — the interval analogue of power.`,
    survey: {
      task: R`Quantify uncertainty about $\theta$ with a set that has guaranteed long-run coverage`,
      assumptions: R`A pivot with known distribution, a family of level-$\alpha$ tests, or an asymptotically normal estimator`,
      output: R`Interval $[L(X),U(X)]$ (or region) with coverage $\ge1-\alpha$ exactly or asymptotically`,
      strengths: 'Reports effect size and precision together; exact for pivotal models; duality with tests',
      weaknesses: 'Coverage is a pre-data average, not a post-data probability; asymptotic intervals can under-cover badly',
      complexity: 'Closed form for pivots; root-finding for test inversion; resampling for bootstrap intervals',
    },
    sections: [
      {
        title: 'What a confidence set is (and is not)',
        html: R`
<div class="def" data-name="Confidence set">
<p>A set-valued statistic $C(X)\subseteq\Theta$ is a $1-\alpha$ <em>confidence set</em> if
$$\inf_{\theta\in\Theta}\P_\theta\big(\theta\in C(X)\big)\ge 1-\alpha.$$
The left side is the <em>confidence coefficient</em>; $\P_\theta(\theta\in C(X))$ as a function of $\theta$ is the <em>coverage probability</em>. When $C(X) = [L(X),U(X)]$ we have a confidence interval; one-sided bounds have $U=\infty$ or $L=-\infty$.</p>
</div>
<p>The probability statement is about $X$, with $\theta$ held fixed: before the data are drawn, the procedure will trap the truth with probability at least $1-\alpha$. After the data are observed, the realised interval $[1.3, 2.7]$ either contains $\theta$ or it does not; there is no probability left to assign without a prior. The lab below draws 100 intervals from the same population — about 95 of them cover, and you cannot tell from any single interval whether it is one of the lucky ones.</p>
<div class="note" data-name="The infimum matters">
<p>Coverage must hold for <em>every</em> $\theta$. An interval with 95% coverage averaged over $\theta$ but 60% at some $\theta$ is not a 95% interval; this is exactly the standard the Wald binomial interval fails. Asymptotic intervals typically satisfy $\P_\theta(\theta\in C_n)\to1-\alpha$ for each fixed $\theta$ but not uniformly in $\theta$.</p>
</div>`,
        lab: 'confidence-intervals-dance',
      },
      {
        title: 'Pivotal quantities',
        html: R`
<div class="def" data-name="Pivot">
<p>A random variable $Q(X,\theta)$ is a <em>pivot</em> if its distribution is the same for all $\theta\in\Theta$.</p>
</div>
<p>If $Q$ is a pivot, pick constants $a\lt b$ with $\P(a\le Q\le b) = 1-\alpha$ (computable, since the law of $Q$ does not depend on $\theta$). Then $C(X) = \{\theta : a\le Q(X,\theta)\le b\}$ satisfies $\P_\theta(\theta\in C(X)) = \P_\theta(a\le Q(X,\theta)\le b) = 1-\alpha$ for <em>every</em> $\theta$ — exact coverage, no asymptotics. When $Q$ is monotone in $\theta$ the set is an interval.</p>
<table>
<thead><tr><th>Model</th><th>Pivot</th><th>Law</th><th>Interval</th></tr></thead>
<tbody>
<tr><td>$\N(\mu,\sigma^2)$, $\sigma$ known</td><td>$\sqrt n(\bar X-\mu)/\sigma$</td><td>$\N(0,1)$</td><td>$\bar X\pm z_{\alpha/2}\sigma/\sqrt n$</td></tr>
<tr><td>$\N(\mu,\sigma^2)$, both unknown</td><td>$\sqrt n(\bar X-\mu)/S$</td><td>$t_{n-1}$</td><td>$\bar X\pm t_{n-1,\alpha/2}S/\sqrt n$</td></tr>
<tr><td>$\N(\mu,\sigma^2)$, variance</td><td>$(n-1)S^2/\sigma^2$</td><td>$\chi^2_{n-1}$</td><td>$\big[\frac{(n-1)S^2}{b},\frac{(n-1)S^2}{a}\big]$</td></tr>
<tr><td>$\text{Exp}$ with mean $\theta$</td><td>$2\sum X_i/\theta$</td><td>$\chi^2_{2n}$</td><td>$\big[\frac{2\sum X_i}{b},\frac{2\sum X_i}{a}\big]$</td></tr>
<tr><td>$\text{Unif}(0,\theta)$</td><td>$X_{(n)}/\theta$</td><td>CDF $t^n$ on $[0,1]$</td><td>$\big[X_{(n)},\, X_{(n)}\alpha^{-1/n}\big]$</td></tr>
</tbody>
</table>
<p>In the variance and exponential rows, $a,b$ are quantiles of the pivot law with $\P(a\le Q\le b)=1-\alpha$; the equal-tailed choice is $a=\chi^2_{\alpha/2}$, $b=\chi^2_{1-\alpha/2}$ (lower quantiles). For the uniform, the one-sided choice is natural: $\P(X_{(n)}/\theta\ge\alpha^{1/n}) = 1-\alpha$, giving an interval whose lower end is the observed maximum, since $\theta$ can never be smaller.</p>
<div class="thm" data-kind="Proposition" data-name="Pivots in location–scale families">
<p>If $X_i = \mu+\sigma Z_i$ with $Z_1,\dots,Z_n\iid F$ for a known $F$, then $(\bar X-\mu)/S$ and $S/\sigma$ are pivots, as is $(T(X)-\mu)/S$ for any location-equivariant $T$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\bar X - \mu = \sigma\bar Z$ and $S_X = \sigma S_Z$, so $(\bar X-\mu)/S_X = \bar Z/S_Z$ and $S_X/\sigma = S_Z$, functions of $Z$ alone, whose law is fixed by $F$. The same cancellation works for any $T$ with $T(\mu+\sigma z) = \mu+\sigma T(z)$.</p>
</details>
<p>The general lesson: pivots arise from <strong>group structure</strong> (location, scale) in the model. When there is none — the binomial, the Poisson — exact pivots do not exist and we turn to test inversion or asymptotics.</p>`,
      },
      {
        title: 'Inverting a family of tests',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Duality of tests and confidence sets">
<p>(a) For each $\theta_0\in\Theta$ let $A(\theta_0)$ be the acceptance region of a level-$\alpha$ test of $H_0:\theta=\theta_0$. Then $C(x) = \{\theta_0 : x\in A(\theta_0)\}$ is a $1-\alpha$ confidence set.</p>
<p>(b) Conversely, if $C$ is a $1-\alpha$ confidence set, then $A(\theta_0) = \{x:\theta_0\in C(x)\}$ is the acceptance region of a level-$\alpha$ test of $H_0:\theta=\theta_0$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Both parts rest on the identity of events $\{\theta\in C(X)\} = \{X\in A(\theta)\}$, which holds by construction. Hence $\P_\theta(\theta\in C(X)) = \P_\theta(X\in A(\theta)) = 1-\P_\theta(\text{reject }\theta)\ge1-\alpha$ for every $\theta$, and the converse reads the same equation from right to left.</p>
</details>
<p>Test inversion is the master construction: all the pivot intervals above invert the corresponding pivot tests; Wilson and LR intervals invert <a href="#/t/likelihood-ratio-tests">score and LR tests</a>; Clopper–Pearson inverts exact binomial tests. Optimality transfers too: inverting UMP one-sided tests gives <em>uniformly most accurate</em> confidence bounds — bounds that minimise the probability of covering each false value below (or above) the truth.</p>
<div class="ex" data-name="Exact Poisson interval">
<p>$X\sim\text{Poisson}(\lambda)$, observed $x$. Invert the two one-sided exact tests: the upper limit $\lambda_U$ solves $\P_{\lambda_U}(X\le x) = \alpha/2$, the lower limit $\lambda_L$ solves $\P_{\lambda_L}(X\ge x) = \alpha/2$. Using the identity $\P_\lambda(X\le x) = \P(\text{Gamma}(x+1,1)\gt\lambda)$, the solutions are Gamma quantiles: $\lambda_L = \tfrac12\chi^2_{2x,\alpha/2}$ and $\lambda_U = \tfrac12\chi^2_{2x+2,1-\alpha/2}$. For $x=0$: $\lambda_L=0$, $\lambda_U = -\log(\alpha/2)\approx3.69$.</p>
</div>`,
      },
      {
        title: 'Pivoting the CDF and asymptotic intervals',
        html: R`
<p>Even without a natural pivot, any real statistic $T$ with continuous CDF $F_T(t\mid\theta)$ supplies one: by the probability integral transform, $F_T(T\mid\theta)\sim\text{Unif}(0,1)$ for every $\theta$.</p>
<div class="thm" data-kind="Theorem" data-name="Pivoting a CDF">
<p>Let $T$ have continuous CDF $F_T(t\mid\theta)$ that is strictly decreasing in $\theta$ for each $t$ (the family is stochastically increasing). Define $\theta_L(t)$, $\theta_U(t)$ by
$$F_T(t\mid\theta_U(t)) = \alpha/2,\qquad F_T(t\mid\theta_L(t)) = 1-\alpha/2.$$
Then $[\theta_L(T),\theta_U(T)]$ is a $1-\alpha$ confidence interval.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Because $F_T(t\mid\cdot)$ is decreasing, $F_T(t\mid\theta)\ge\alpha/2 \iff \theta\le\theta_U(t)$ and $F_T(t\mid\theta)\le1-\alpha/2\iff\theta\ge\theta_L(t)$. Hence $\{\theta_L(T)\le\theta\le\theta_U(T)\} = \{\alpha/2\le F_T(T\mid\theta)\le1-\alpha/2\}$, which has probability $1-\alpha$ since $F_T(T\mid\theta)$ is uniform.</p>
</details>
<p>For discrete $T$ the same construction with $\P_\theta(T\le t)$ and $\P_\theta(T\ge t)$ gives conservative intervals (Clopper–Pearson, the exact Poisson interval above).</p>
<p>When nothing exact is available, use asymptotic pivots. If $(\hth_n-\theta)/\widehat{\se}\dto\N(0,1)$ then $\hth_n\pm z_{\alpha/2}\widehat{\se}$ has asymptotic coverage $1-\alpha$ (the Wald interval). Improvements, roughly in order of reliability:</p>
<ul>
<li><strong>Transform first.</strong> Build the interval for $\phi = g(\theta)$ on a scale where the estimator is closer to normal and its variance does not depend on $\theta$ (variance-stabilising transforms via the <a href="#/t/delta-method">delta method</a>: $\sqrt{\cdot}$ for Poisson, $\arcsin\sqrt\cdot$ for binomial, Fisher’s $\tanh^{-1}$ for correlation), then map back.</li>
<li><strong>Invert LR or score tests</strong>, which respect the parameter space and the likelihood’s asymmetry.</li>
<li><strong>Bootstrap</strong>, especially studentised or <a href="#/t/bca-intervals">BC$_a$</a> intervals, which correct skewness to second order (<a href="#/t/bootstrap">the bootstrap</a>).</li>
</ul>`,
      },
      {
        title: 'Optimality: short intervals and false coverage',
        html: R`
<p>Validity is easy — $C(X)=\Theta$ has coverage 1. We want intervals that are valid <em>and</em> small.</p>
<div class="thm" data-kind="Theorem" data-name="Shortest interval from a unimodal pivot">
<p>Let $f$ be a unimodal density with CDF $F$. Among intervals $[a,b]$ with $F(b)-F(a) = 1-\alpha$, the shortest satisfies $f(a)=f(b)\gt0$ and $a\le x^*\le b$, where $x^*$ is a mode. Consequently, if a pivot has the form $Q = (T-\theta)/c$ with $c$ not depending on $\theta$, the interval with equal pivot-density endpoints is the shortest pivotal interval.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Minimise $b-a$ subject to $F(b)-F(a) = 1-\alpha$. With a Lagrange multiplier, $\partial_a: -1+\lambda f(a) = 0$ and $\partial_b: 1-\lambda f(b) = 0$, so $f(a)=f(b)$. That this stationary point is the minimum follows from unimodality: if $f(a)\lt f(b)$, sliding the interval toward $b$ gains mass at the right end faster than it loses it at the left, so for the same mass it can be made shorter; the argument is reversed if $f(a)\gt f(b)$.</p>
</details>
<p>The length that matters is in the $\theta$ scale, which can be a non-linear function of the pivot endpoints. For the normal variance, the interval $[(n-1)S^2/b,(n-1)S^2/a]$ has length proportional to $1/a-1/b$; minimising it subject to the $\chi^2_{n-1}$ mass constraint gives $a^2f(a) = b^2f(b)$, not $f(a)=f(b)$, and the shortest interval is noticeably asymmetric in tail probabilities. The second lab lets you slide the tail split and watch the length.</p>
<div class="thm" data-kind="Theorem" data-name="Pratt (1961)">
<p>For a confidence set $C(X)\subseteq\R$ with Lebesgue measure $|C(X)|$,
$$\E_\theta|C(X)| = \int_{\theta'\ne\theta}\P_\theta\big(\theta'\in C(X)\big)\d\theta'.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$|C(X)| = \int\ind\{\theta'\in C(X)\}\d\theta'$. Take $\E_\theta$ and exchange expectation and integral by Tonelli (the integrand is non-negative). Removing the single point $\theta'=\theta$ does not change a Lebesgue integral.</p>
</details>
<p>Pratt’s identity says expected length is the integrated probability of <strong>false coverage</strong>. Since $\P_\theta(\theta'\in C(X)) = 1-\text{power of the test of }\theta'$, short intervals come from powerful tests: inverting UMP (or UMPU) tests yields intervals that minimise expected length within the corresponding class.</p>`,
        lab: 'confidence-intervals-shortest',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Post-data probability">
<p>"There is a 95% probability that $\mu$ lies in $[1.3,2.7]$" is a Bayesian statement and generally false as a frequentist one. Under a flat prior in the normal model the two coincide numerically (see <a href="#/t/bayesian-inference">Bayesian inference</a>), which is why the confusion is so durable.</p>
</div>
<div class="warn" data-name="Relevant subsets: coverage is an average">
<p>Let $X_1,X_2\iid\text{Unif}(\theta-\tfrac12,\theta+\tfrac12)$ and take $C = [X_{(1)},X_{(2)}]$. It covers $\theta$ exactly when the two points fall on opposite sides of $\theta$, which has probability $\tfrac12$: a valid 50% interval. But if the observed range $X_{(2)}-X_{(1)}\ge\tfrac12$, both points within $\tfrac12$ of $\theta$ force $X_{(2)}-\tfrac12\le\theta\le X_{(1)}+\tfrac12$, an interval contained in $C$ — so $C$ covers with <em>certainty</em>. When the range is tiny, conditional coverage is close to 0. The unconditional 50% averages over recognisably different situations. Conditioning on ancillary statistics (here the range) is the frequentist remedy (Fisher; Cox, 1958).</p>
</div>
<div class="warn" data-name="Simultaneous vs marginal coverage">
<p>Ten separate 95% intervals do not jointly cover all ten parameters with probability 95%. Joint statements need simultaneous regions (Bonferroni, Scheffé, Tukey) — see <a href="#/t/multiple-testing">multiple testing</a>. Intervals reported only for "significant" parameters have lower coverage still; false coverage-rate (FCR) methods address selection.</p>
</div>
<div class="warn" data-name="Overlap is not a test">
<p>Two 95% intervals for $\mu_1$ and $\mu_2$ can overlap while the difference $\mu_1-\mu_2$ is significant at 5%. Non-overlap corresponds to a test at roughly the 0.6% level when the standard errors are equal. Build the interval for the difference.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`$X_1,\dots,X_{10}\iid\text{Exp}$ with mean $\theta$, and $\sum X_i = 25$. Give the equal-tailed 95% interval for $\theta$ in terms of chi-square quantiles.`,
        a: R`$2\sum X_i/\theta\sim\chi^2_{20}$. With $a=\chi^2_{20,0.025}\approx9.59$ and $b=\chi^2_{20,0.975}\approx34.17$, the interval is $[50/b,\,50/a]\approx[1.46,\,5.21]$.` },
      { q: R`State the duality between tests and confidence sets, and use it to explain why "the 95% CI excludes 0" is equivalent to "p < 0.05 for $H_0:\theta=0$".`,
        a: R`$C(x) = \{\theta_0: x\in A(\theta_0)\}$ where $A(\theta_0)$ is the acceptance region of a level-$\alpha$ test of $\theta=\theta_0$; the events $\{\theta\in C(X)\}$ and $\{X\in A(\theta)\}$ coincide. So $0\notin C(x)$ iff $x\notin A(0)$ iff the level-0.05 test rejects $\theta=0$ — provided the interval and the test come from the same family of tests.` },
      { q: R`What does Pratt’s theorem say, and why does it link short intervals to powerful tests?`,
        a: R`$\E_\theta|C(X)| = \int_{\theta'\ne\theta}\P_\theta(\theta'\in C(X))\d\theta'$: expected length equals the total probability of covering false values. Covering a false $\theta'$ is a failure of the inverted test to reject $\theta'$ (a type II error), so tests with higher power against every $\theta'$ give intervals with smaller expected length.` },
      { q: R`Why is the equal-tailed chi-square interval for $\sigma^2$ not the shortest one?`,
        a: R`Its length is proportional to $1/a-1/b$, which depends non-linearly on the pivot endpoints. Minimising subject to $F(b)-F(a)=1-\alpha$ gives $a^2f(a)=b^2f(b)$, which is not satisfied by equal tails: there $a^2f(a)\lt b^2f(b)$, because the $1/a$ term is very sensitive to small $a$. The shortest interval raises $a$, putting <em>more</em> than $\alpha/2$ in the lower tail of the pivot (and less in the upper), which shortens the long upper end of the interval for $\sigma^2$.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'multiple-testing',
    domain: 'inference',
    order: 150,
    title: 'Multiple Testing: FWER & the False Discovery Rate',
    subtitle: 'Controlling errors when thousands of hypotheses are tested at once',
    level: 2,
    sources: { aos: '10.7', cb: '—', other: 'Holm (1979); Benjamini & Hochberg (1995); Benjamini & Yekutieli (2001); Efron, Large-Scale Inference (2010)' },
    prereqs: ['hypothesis-testing'],
    related: ['confidence-intervals', 'permutation-tests', 'model-selection', 'bayesian-inference'],
    tags: ['FWER', 'Bonferroni', 'Holm', 'FDR', 'Benjamini–Hochberg', 'q-value'],
    summary: R`Test $m$ true null hypotheses each at level $\alpha$ and you should expect $m\alpha$ false rejections — 500 "discoveries" among 10,000 null genes at 5%. Multiple-testing procedures redefine the error being controlled. The family-wise error rate (FWER), the probability of even one false rejection, is controlled by Bonferroni and, uniformly more powerfully, by Holm’s step-down procedure — using nothing but the union bound. The false discovery rate (FDR), the expected fraction of rejections that are false, is a far less stringent target suited to screening; the Benjamini–Hochberg step-up procedure controls it at level $q$ under independence, by an elegant exact argument that gives $\text{FDR} = \pi_0 q$.`,
    survey: {
      task: R`Test $m$ hypotheses simultaneously while controlling a global error rate`,
      assumptions: R`Valid (super-uniform) p-values for true nulls; for BH, independence or positive dependence (PRDS)`,
      output: R`A rejection set; adjusted p-values or q-values`,
      strengths: 'Bonferroni/Holm need no dependence assumptions; BH retains power as $m$ grows; all are trivial to compute',
      weaknesses: R`FWER control is very conservative for large $m$; FDR is an expectation, not a per-study guarantee; dependence can matter`,
      complexity: R`$O(m\log m)$ (a sort)`,
    },
    sections: [
      {
        title: 'The problem and the error rates',
        html: R`
<p>Test $H_1,\dots,H_m$ with p-values $p_1,\dots,p_m$. Of the $m$ hypotheses, $m_0$ are true nulls (index set $\mathcal H_0$) and $m_1 = m-m_0$ are false. A procedure produces the following unobservable table:</p>
<table>
<thead><tr><th></th><th>Not rejected</th><th>Rejected</th><th>Total</th></tr></thead>
<tbody>
<tr><td>True null</td><td>$U$</td><td>$V$ (false discoveries)</td><td>$m_0$</td></tr>
<tr><td>False null</td><td>$T$</td><td>$S$ (true discoveries)</td><td>$m_1$</td></tr>
<tr><td>Total</td><td>$m-R$</td><td>$R$</td><td>$m$</td></tr>
</tbody>
</table>
<p>Only $R$ is observed. If each test is run at level $\alpha$ with no adjustment, $\E V = m_0\alpha$, and with independent tests $\P(V\ge1) = 1-(1-\alpha)^{m_0}$, which is $0.64$ for $m_0=20$ and essentially 1 for $m_0=1000$.</p>
<div class="def" data-name="Error rates">
<ul>
<li><strong>Family-wise error rate:</strong> $\text{FWER} = \P(V\ge1)$.</li>
<li><strong>False discovery proportion:</strong> $\text{FDP} = V/\max(R,1)$ (zero when nothing is rejected).</li>
<li><strong>False discovery rate:</strong> $\text{FDR} = \E[\text{FDP}]$ (Benjamini & Hochberg, 1995).</li>
</ul>
<p>Control is <em>strong</em> if it holds for every configuration of true and false nulls, and <em>weak</em> if only under the global null $m_0=m$.</p>
</div>
<p>Since $\text{FDP}\le\ind\{V\ge1\}$, $\text{FDR}\le\text{FWER}$: FDR control is the weaker requirement and buys more power. Under the global null $V=R$, so $\text{FDP} = \ind\{V\ge1\}$ and $\text{FDR}=\text{FWER}$; FDR control implies weak FWER control. Which one is appropriate depends on the cost of a single false positive. A confirmatory clinical trial with three endpoints wants FWER; a genome-wide screen whose hits will be followed up wants a list that is mostly right, i.e. FDR.</p>`,
      },
      {
        title: 'Controlling the FWER: Bonferroni and Holm',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Bonferroni">
<p>Rejecting $H_i$ when $p_i\le\alpha/m$ controls the FWER at level $\alpha$ in the strong sense, under any dependence among the p-values.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\P(V\ge1) = \P\big(\bigcup_{i\in\mathcal H_0}\{p_i\le\alpha/m\}\big)\le\sum_{i\in\mathcal H_0}\P(p_i\le\alpha/m)\le m_0\,\alpha/m\le\alpha$, using the union bound and super-uniformity of null p-values.</p>
</details>
<p>For independent tests, Šidák’s threshold $1-(1-\alpha)^{1/m}$ is exact and marginally larger than $\alpha/m$. More useful is a uniform improvement that costs nothing in assumptions.</p>
<div class="algo" data-name="Holm’s step-down procedure (1979)">
<ol>
<li>Order the p-values $p_{(1)}\le\dots\le p_{(m)}$ with hypotheses $H_{(1)},\dots,H_{(m)}$.</li>
<li>Find the smallest $j$ with $p_{(j)}\gt\alpha/(m-j+1)$.</li>
<li>Reject $H_{(1)},\dots,H_{(j-1)}$ (all, if no such $j$ exists).</li>
</ol>
</div>
<p>The first threshold is Bonferroni’s $\alpha/m$, but each subsequent one is less strict: $\alpha/(m-1)$, $\alpha/(m-2)$, …, $\alpha$. Holm rejects everything Bonferroni does and sometimes more.</p>
<div class="thm" data-kind="Theorem" data-name="Holm controls FWER">
<p>Holm’s procedure controls the FWER at level $\alpha$ in the strong sense, under arbitrary dependence.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Suppose $m_0\ge1$ and let $j^*$ be the position, in the sorted list, of the smallest true-null p-value. At most $m-m_0$ hypotheses (the false nulls) can precede it, so $j^*\le m-m_0+1$, i.e. $m-j^*+1\ge m_0$, and its threshold satisfies $\alpha/(m-j^*+1)\le\alpha/m_0$. A false rejection requires Holm to reach and reject position $j^*$, which requires $p_{(j^*)}\le\alpha/(m-j^*+1)\le\alpha/m_0$. Therefore
$$\P(V\ge1)\le\P\Big(\min_{i\in\mathcal H_0}p_i\le\frac{\alpha}{m_0}\Big)\le m_0\cdot\frac{\alpha}{m_0} = \alpha.$$</p>
</details>
<p>Both procedures can be phrased as <em>adjusted p-values</em> — the smallest $\alpha$ at which each hypothesis is rejected: Bonferroni $\tilde p_i = \min(1,mp_i)$; Holm $\tilde p_{(j)} = \max_{l\le j}\min(1,(m-l+1)p_{(l)})$. Hochberg’s step-up variant uses the same thresholds but is valid only under independence or positive dependence. All of these derive from the <em>closure principle</em> (Marcus, Peritz & Gabriel, 1976): reject $H_i$ iff every intersection hypothesis containing $i$ is rejected by a local level-$\alpha$ test.</p>`,
      },
      {
        title: 'The Benjamini–Hochberg procedure',
        html: R`
<div class="algo" data-name="Benjamini–Hochberg (BH) step-up at level q">
<ol>
<li>Order $p_{(1)}\le\dots\le p_{(m)}$.</li>
<li>Let $R = \max\{k : p_{(k)}\le qk/m\}$ (zero if the set is empty).</li>
<li>Reject $H_{(1)},\dots,H_{(R)}$ — equivalently, reject every $H_i$ with $p_i\le qR/m$.</li>
</ol>
</div>
<p>Graphically: plot the sorted p-values against rank and draw the line through the origin with slope $q/m$. Reject everything up to the <em>last</em> point below the line, even if some earlier points lie above it. That "step-up" feature is what gives BH its power: one very small p-value lowers the bar for the others.</p>
<div class="thm" data-kind="Theorem" data-name="BH controls the FDR (Benjamini & Hochberg, 1995)">
<p>If the true-null p-values are independent of each other and of the false-null p-values, and each is $\text{Unif}(0,1)$, then the BH procedure satisfies
$$\text{FDR} = \frac{m_0}{m}\,q\le q.$$
(If null p-values are merely super-uniform, $\le$ holds.)</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $V = \sum_{i\in\mathcal H_0}V_i$ with $V_i = \ind\{H_i\text{ rejected}\}$, so $\text{FDR} = \sum_{i\in\mathcal H_0}\E[V_i/\max(R,1)]$. It suffices to show each term equals $q/m$.</p>
<p><em>A characterisation.</em> For a threshold index $k$, let $N(k) = \#\{l : p_l\le qk/m\}$. Then $R = \max\{k: N(k)\ge k\}$, and $H_i$ is rejected iff $p_i\le qR/m$.</p>
<p><em>The leave-one-in trick.</em> Fix a null $i$ and let $R^{(i)}$ be the number of BH rejections when $p_i$ is replaced by 0, all else unchanged. $R^{(i)}$ is a function of $(p_l)_{l\ne i}$ only, hence independent of $p_i$. Claim: for each $k\ge1$,
$$\{p_i\le qk/m\}\cap\{R=k\} = \{p_i\le qk/m\}\cap\{R^{(i)}=k\}.$$
Indeed, on $\{p_i\le qk/m\}$, for every $j\ge k$ the threshold $qj/m\ge p_i$, so $p_i$ is counted in $N(j)$ whether its value is $p_i$ or 0; thus $N(j) = N^{(i)}(j)$ for all $j\ge k$. Both $R=k$ and $R^{(i)}=k$ mean "$N(k)\ge k$ and $N(j)\lt j$ for all $j\gt k$", conditions that only involve $j\ge k$, so they coincide.</p>
<p><em>Computation.</em> On $\{V_i=1\}$ we have $R\ge1$ and $p_i\le qR/m$. Therefore
$$\E\Big[\frac{V_i}{\max(R,1)}\Big] = \sum_{k=1}^m\frac1k\P\big(p_i\le\tfrac{qk}{m},\,R=k\big) = \sum_{k=1}^m\frac1k\P\big(p_i\le\tfrac{qk}{m}\big)\P\big(R^{(i)}=k\big) = \sum_{k=1}^m\frac1k\cdot\frac{qk}{m}\P\big(R^{(i)}=k\big) = \frac qm,$$
using the claim, independence of $p_i$ and $R^{(i)}$, uniformity of $p_i$, and $\sum_k\P(R^{(i)}=k) = 1$ (note $R^{(i)}\ge1$ always, since $p_i=0$ is rejected). Summing over the $m_0$ nulls gives $\text{FDR} = m_0q/m$.</p>
</details>
<p>The bound is tight and also suggests an improvement: since BH actually controls FDR at $\pi_0 q$ with $\pi_0 = m_0/m$, running it at level $q/\hat\pi_0$ with an estimate $\hat\pi_0$ recovers power when many hypotheses are non-null. Storey’s estimator $\hat\pi_0(\lambda) = \#\{p_i\gt\lambda\}/\{(1-\lambda)m\}$ exploits the flatness of the null p-value histogram above $\lambda$; the resulting adjusted values are Storey’s <em>q-values</em>.</p>
<div class="ex" data-name="Ten p-values">
<p>Sorted p-values: 0.001, 0.004, 0.006, 0.015, 0.021, 0.031, 0.10, 0.27, 0.48, 0.80, at $\alpha = q = 0.05$, $m=10$.</p>
<ul>
<li><strong>Bonferroni</strong> ($\le0.005$): rejects 2.</li>
<li><strong>Holm</strong>: thresholds $0.005, 0.00556, 0.00625, 0.00714,\dots$. $0.001\le0.005$, $0.004\le0.00556$, $0.006\le0.00625$, but $0.015\gt0.00714$: rejects 3.</li>
<li><strong>BH</strong>: thresholds $0.005k$: $0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035,\dots$. Comparing: $0.001, 0.004, 0.006, 0.015$ are below, $0.021\le0.025$ is below, $0.031\gt0.030$ is above, and nothing later is below its threshold. $R=5$: rejects 5.</li>
</ul>
</div>`,
        lab: 'multiple-testing-bh',
      },
      {
        title: 'Dependence',
        html: R`
<p>The independence assumption in the BH theorem is rarely literally true: test statistics for nearby genes, overlapping brain voxels or correlated outcomes are dependent.</p>
<div class="thm" data-kind="Theorem" data-name="Benjamini & Yekutieli (2001)">
<p>(a) If the test statistics are <em>positively regression dependent on the subset</em> of true nulls (PRDS) — e.g. one-sided tests from multivariate normal statistics with non-negative correlations — BH still controls the FDR at $\pi_0q$.</p>
<p>(b) Under arbitrary dependence, BH run at level $q/c_m$ with $c_m = \sum_{i=1}^m 1/i\approx\log m + 0.577$ controls the FDR at $q$.</p>
</div>
<p>The BY correction is severe ($c_m\approx9.8$ for $m=10^4$) and rarely necessary; in practice BH is remarkably robust to the forms of dependence that occur. What dependence does change is the <em>variability</em> of the FDP: with strongly correlated statistics, the realised FDP is either 0 or very large, so "FDR $=5\%$" can coexist with a substantial chance that half of a given study’s discoveries are false. The second lab lets you add equicorrelation and watch this.</p>
<div class="warn" data-name="FDR is an expectation">
<p>FDR control says nothing about the FDP in your study, only its average over hypothetical repetitions. Procedures controlling $\P(\text{FDP}\gt\gamma)$ (false discovery exceedance) exist when a per-study guarantee is needed.</p>
</div>`,
        lab: 'multiple-testing-rates',
      },
      {
        title: 'The empirical Bayes view',
        html: R`
<p>Efron’s <em>two-groups model</em> gives FDR a Bayesian reading. Suppose each hypothesis is null with probability $\pi_0$, and its $z$-value has density $f_0$ if null and $f_1$ otherwise, so the mixture is $f = \pi_0f_0+\pi_1f_1$. Then for a rejection region $\{z\ge c\}$, by Bayes’ theorem,
$$\text{Fdr}(c) = \P(\text{null}\mid Z\ge c) = \frac{\pi_0\,\bar F_0(c)}{\bar F(c)},$$
where $\bar F$ denotes the survival function. Replacing $\bar F(c)$ by its empirical estimate $R(c)/m$ and $\bar F_0(c)$ by the p-value threshold recovers the BH rule: BH rejects exactly where the estimated Bayesian Fdr is below $q$ (with $\pi_0$ bounded by 1). The <em>local</em> false discovery rate $\text{fdr}(z) = \pi_0f_0(z)/f(z)$ is the posterior null probability for an individual case. Large-scale inference thus sits naturally between the frequentist and <a href="#/t/bayesian-inference">Bayesian</a> paradigms: the many parallel tests estimate their own prior.</p>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="What counts as the family?">
<p>Error-rate control is relative to the set of tests adjusted for. Splitting an analysis across papers, or adjusting within subgroups but not across outcomes, silently changes the family. Pre-specify it.</p>
</div>
<div class="warn" data-name="Selection after testing">
<p>The effect estimates of hypotheses that survived a threshold are biased upward ("winner’s curse"), and their unadjusted confidence intervals under-cover. Selective-inference methods and false coverage-rate intervals (Benjamini & Yekutieli, 2005) correct this.</p>
</div>
<div class="warn" data-name="Invalid null p-values">
<p>Every procedure assumes null p-values are super-uniform. A misspecified null distribution (e.g. unmodelled correlation or overdispersion inflating all $z$-scores) breaks every guarantee at once; Efron’s <em>empirical null</em> estimates $f_0$ from the bulk of the $z$-values for this reason.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why is FDR control under the global null equivalent to weak FWER control?`,
        a: R`If all nulls are true then every rejection is false, so $V=R$ and $\text{FDP}=V/\max(V,1) = \ind\{V\ge1\}$. Taking expectations, $\text{FDR} = \P(V\ge1) = \text{FWER}$.` },
      { q: R`In the proof that Holm controls the FWER, where does the inequality $m-j^*+1\ge m_0$ come from?`,
        a: R`$j^*$ is the rank of the smallest true-null p-value; only the $m-m_0$ false nulls can have smaller p-values, so $j^*\le m-m_0+1$. Hence the threshold at that step, $\alpha/(m-j^*+1)$, is at most $\alpha/m_0$, and the union bound over the $m_0$ nulls gives $\alpha$.` },
      { q: R`With $m=1000$, $m_0=900$ independent tests and BH at $q=0.1$, what is the exact FDR, and how could you use this to gain power?`,
        a: R`$\text{FDR} = (m_0/m)q = 0.9\times0.1 = 0.09$. Estimating $\pi_0$ (e.g. Storey’s $\hat\pi_0$) and running BH at $q/\hat\pi_0\approx0.111$ brings the FDR back up to about $0.1$ and rejects more hypotheses.` },
      { q: R`Why does BH reject $H_{(3)}$ in a situation where $p_{(3)}\gt 3q/m$ but $p_{(5)}\le 5q/m$?`,
        a: R`BH is step-up: $R$ is the <em>largest</em> $k$ with $p_{(k)}\le qk/m$, here at least 5, and every hypothesis with $p_i\le qR/m$ is rejected. Since $p_{(3)}\le p_{(5)}\le 5q/m\le qR/m$, $H_{(3)}$ is rejected even though it is above its own line.` },
    ],
  });
  /* ================================================================== */
  Atlas.topic({
    id: 'decision-theory',
    domain: 'inference',
    order: 160,
    title: 'Decision Theory: Risk, Minimaxity & Admissibility',
    subtitle: 'Comparing statistical procedures by their risk functions',
    level: 3,
    sources: { aos: 'Ch. 12', cb: '§7.3.4', other: 'Lehmann & Casella, Theory of Point Estimation (1998), Ch. 4–5; Berger (1985); Efron & Morris (1973)' },
    prereqs: ['point-estimation', 'bayesian-inference'],
    related: ['mle', 'rao-blackwell', 'ridge-regression', 'hierarchical-mcmc', 'neyman-pearson'],
    tags: ['loss', 'risk', 'Bayes rule', 'minimax', 'admissibility', 'James–Stein', 'shrinkage'],
    summary: R`Decision theory puts every statistical procedure on a common footing: choose a loss $L(\theta,a)$, and a rule $\delta$ is judged by its risk function $R(\theta,\delta) = \E_\theta L(\theta,\delta(X))$. Risk functions cross, so there is no uniformly best rule; one must either average the risk over a prior (Bayes) or guard against the worst case (minimax). Bayes rules are found by minimising posterior expected loss one data set at a time; a Bayes rule with constant risk is minimax. The weakest sensible requirement — admissibility, not being dominated — produces the most famous surprise in statistics: in three or more dimensions the sample mean is inadmissible under squared error, beaten everywhere by the James–Stein shrinkage estimator.`,
    survey: {
      task: 'Compare and optimise estimators, tests and other procedures under an explicit loss',
      assumptions: R`A model $\{P_\theta\}$, an action space, a loss $L(\theta,a)\ge0$; for Bayes, a prior`,
      output: R`Risk functions; Bayes rules; minimax rules; admissibility verdicts`,
      strengths: 'Unifies estimation, testing and prediction; explains shrinkage; makes the role of the loss explicit',
      weaknesses: 'The answer depends on the loss; minimax can be pessimistic; admissibility alone is a weak criterion',
      complexity: 'Analytical; Bayes rules need posterior computations',
    },
    sections: [
      {
        title: 'Loss, risk, and why there is no best estimator',
        html: R`
<div class="def" data-name="Decision problem">
<p>A <em>decision rule</em> is a map $\delta$ from the sample space to an action space $\mathcal A$ (estimates, accept/reject, predictions). A <em>loss</em> $L:\Theta\times\mathcal A\to[0,\infty)$ measures the cost of action $a$ when the truth is $\theta$. The <em>risk function</em> of $\delta$ is
$$R(\theta,\delta) = \E_\theta\,L\big(\theta,\delta(X)\big) = \int L(\theta,\delta(x))\,f(x\mid\theta)\d x.$$</p>
</div>
<p>Common losses: squared error $(\theta-a)^2$, whose risk is the <a href="#/t/point-estimation">MSE</a>; absolute error $|\theta-a|$; weighted squared error $w(\theta)(\theta-a)^2$; the 0–1 loss of a test, $\ind\{\text{wrong decision}\}$, whose risk is the type I error probability on $\Theta_0$ and type II on $\Theta_1$; Kullback–Leibler loss for density estimation.</p>
<div class="ex" data-name="Crossing risk functions">
<p>$X\sim\text{Bin}(n,\theta)$ under squared error. The MLE $\delta_1 = X/n$ has risk $\theta(1-\theta)/n$. The silly rule $\delta_2\equiv\tfrac12$ has risk $(\theta-\tfrac12)^2$, which is <em>zero</em> at $\theta=\tfrac12$ — no rule can beat it there. Neither rule dominates. Any rule can be beaten at some $\theta$ by the constant rule that guesses that $\theta$, so a uniformly best rule cannot exist except in trivial problems.</p>
</div>
<p>There are three ways out. <strong>Restrict the class</strong> of rules (unbiased estimators — giving <a href="#/t/rao-blackwell">UMVUEs</a>; unbiased or invariant tests). <strong>Average</strong> the risk with respect to a weight on $\Theta$ — the Bayes approach. <strong>Take the worst case</strong> — the minimax approach. All three collapse the risk function to a single number or ordering.</p>
<div class="def" data-name="Dominance and admissibility">
<p>$\delta'$ <em>dominates</em> $\delta$ if $R(\theta,\delta')\le R(\theta,\delta)$ for all $\theta$, with strict inequality for some $\theta$. A rule is <em>admissible</em> if no rule dominates it, and <em>inadmissible</em> otherwise.</p>
</div>`,
      },
      {
        title: 'Bayes rules',
        html: R`
<div class="def" data-name="Bayes risk and Bayes rule">
<p>For a prior $\pi$ on $\Theta$, the <em>Bayes risk</em> of $\delta$ is $r(\pi,\delta) = \int_\Theta R(\theta,\delta)\,\pi(\theta)\d\theta$. A rule $\delta_\pi$ minimising $r(\pi,\cdot)$ is a <em>Bayes rule</em> with respect to $\pi$.</p>
</div>
<p>Minimising an integral over all functions $\delta$ looks hard. It is not, because the order of integration can be swapped so that the optimisation decouples across data sets.</p>
<div class="thm" data-kind="Theorem" data-name="Bayes rules minimise posterior expected loss">
<p>Suppose that for (almost) every $x$ there is an action $a^*(x)$ minimising the posterior expected loss $\rho(a\mid x) = \int L(\theta,a)\,\pi(\theta\mid x)\d\theta$, and that $\delta_\pi(x) = a^*(x)$ is measurable with $r(\pi,\delta_\pi)\lt\infty$. Then $\delta_\pi$ is a Bayes rule.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write the joint density as $f(x\mid\theta)\pi(\theta) = \pi(\theta\mid x)\,m(x)$. By Tonelli (the loss is non-negative),
$$r(\pi,\delta) = \int\!\!\int L(\theta,\delta(x))f(x\mid\theta)\pi(\theta)\d x\d\theta = \int\Big[\int L(\theta,\delta(x))\,\pi(\theta\mid x)\d\theta\Big]m(x)\d x = \int\rho(\delta(x)\mid x)\,m(x)\d x.$$
For every $x$, $\rho(\delta(x)\mid x)\ge\rho(a^*(x)\mid x)$, so integrating against $m(x)\ge0$ gives $r(\pi,\delta)\ge r(\pi,\delta_\pi)$.</p>
</details>
<table>
<thead><tr><th>Loss</th><th>Bayes rule</th><th>Why</th></tr></thead>
<tbody>
<tr><td>$(\theta-a)^2$</td><td>posterior mean</td><td>$\E[(\theta-a)^2\mid x] = \Var(\theta\mid x) + (\E[\theta\mid x]-a)^2$</td></tr>
<tr><td>$w(\theta)(\theta-a)^2$</td><td>$\E[w(\theta)\theta\mid x]/\E[w(\theta)\mid x]$</td><td>set derivative to zero</td></tr>
<tr><td>$|\theta-a|$</td><td>posterior median</td><td>derivative is $\P(\theta\lt a\mid x)-\P(\theta\gt a\mid x)$</td></tr>
<tr><td>$\ind\{|\theta-a|\gt\epsilon\}$</td><td>centre of the highest-mass window; $\to$ posterior mode as $\epsilon\to0$</td><td>maximise $\P(|\theta-a|\le\epsilon\mid x)$</td></tr>
<tr><td>Testing: cost $c_{\rm I}$ for a false rejection, $c_{\rm II}$ for a false acceptance</td><td>reject $H_0$ iff $\frac{\P(H_0\mid x)}{\P(H_1\mid x)}\lt\frac{c_{\rm II}}{c_{\rm I}}$</td><td>compare $c_{\rm I}\P(H_0\mid x)$ with $c_{\rm II}\P(H_1\mid x)$</td></tr>
</tbody>
</table>
<p>The last row is the Neyman–Pearson form again: posterior odds are prior odds times the likelihood ratio, so Bayes tests between simple hypotheses are <a href="#/t/neyman-pearson">likelihood-ratio tests</a> with a threshold set by costs and prior odds instead of by $\alpha$.</p>
<div class="ex" data-name="Bayes estimators for a binomial">
<p>With a $\text{Beta}(a,b)$ prior and squared error, the Bayes rule is $\delta_{a,b}(X) = (X+a)/(n+a+b)$. Its bias is $(a-(a+b)\theta)/(n+a+b)$ and variance $n\theta(1-\theta)/(n+a+b)^2$, so
$$R(\theta,\delta_{a,b}) = \frac{n\theta(1-\theta) + \big(a-(a+b)\theta\big)^2}{(n+a+b)^2}.$$
The first lab plots these curves.</p>
</div>`,
      },
      {
        title: 'Minimax rules',
        html: R`
<div class="def" data-name="Minimax">
<p>A rule $\delta^*$ is <em>minimax</em> if $\sup_\theta R(\theta,\delta^*) = \inf_\delta\sup_\theta R(\theta,\delta)$. A prior $\pi$ is <em>least favourable</em> if $r(\pi,\delta_\pi)\ge r(\pi',\delta_{\pi'})$ for every prior $\pi'$.</p>
</div>
<p>Minimax rules are hard to find directly. The key tool turns a Bayes computation into a minimax certificate.</p>
<div class="thm" data-kind="Theorem" data-name="Constant-risk Bayes rules are minimax">
<p>Let $\delta_\pi$ be Bayes for $\pi$ and suppose $R(\theta,\delta_\pi)\le r(\pi,\delta_\pi)$ for all $\theta$. Then $\delta_\pi$ is minimax and $\pi$ is least favourable. In particular, a Bayes rule with constant risk is minimax.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For any rule $\delta$,
$$\sup_\theta R(\theta,\delta)\;\ge\;\int R(\theta,\delta)\,\pi(\theta)\d\theta = r(\pi,\delta)\;\ge\; r(\pi,\delta_\pi)\;\ge\;\sup_\theta R(\theta,\delta_\pi).$$
The first inequality bounds an average by a maximum, the second is the Bayes property, the third the hypothesis. For least favourability: for any $\pi'$, $r(\pi',\delta_{\pi'})\le r(\pi',\delta_\pi)\le\sup_\theta R(\theta,\delta_\pi)\le r(\pi,\delta_\pi)$.</p>
</details>
<div class="ex" data-name="The minimax estimator of a binomial proportion">
<p>Look for $a,b$ making the risk of $\delta_{a,b}$ constant in $\theta$. The numerator of the risk is $n\theta-n\theta^2 + a^2 - 2a(a+b)\theta + (a+b)^2\theta^2$; the $\theta^2$ and $\theta$ coefficients vanish when $(a+b)^2 = n$ and $2a(a+b) = n$, i.e. $a = b = \sqrt n/2$. Then
$$\delta^*(X) = \frac{X+\sqrt n/2}{n+\sqrt n},\qquad R(\theta,\delta^*) = \frac{n/4}{(n+\sqrt n)^2} = \frac{1}{4(1+\sqrt n)^2}\quad\text{for all }\theta.$$
By the theorem, $\delta^*$ is minimax and $\text{Beta}(\sqrt n/2,\sqrt n/2)$ is least favourable. Compare the MLE’s maximum risk $1/(4n)$ at $\theta=\tfrac12$: the minimax rule wins near $\tfrac12$ and loses near 0 and 1, and the region where it wins shrinks as $n$ grows. Minimaxity is an insurance policy, priced by the worst case.</p>
</div>
<p>When no proper prior gives constant risk, use a sequence.</p>
<div class="thm" data-kind="Theorem" data-name="Minimaxity via limits of Bayes risks">
<p>If $\delta$ satisfies $\sup_\theta R(\theta,\delta) = \lim_k r(\pi_k,\delta_{\pi_k})$ for some sequence of priors $\pi_k$ with Bayes rules $\delta_{\pi_k}$, then $\delta$ is minimax.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For any $\delta'$ and every $k$, $\sup_\theta R(\theta,\delta')\ge r(\pi_k,\delta')\ge r(\pi_k,\delta_{\pi_k})$. Let $k\to\infty$: $\sup_\theta R(\theta,\delta')\ge\sup_\theta R(\theta,\delta)$.</p>
</details>
<div class="ex" data-name="The normal mean">
<p>$X_1,\dots,X_n\iid\N(\theta,\sigma^2)$, squared error. $\bar X$ has constant risk $\sigma^2/n$. Under the prior $\N(0,\tau^2)$ the Bayes rule is the posterior mean, with Bayes risk equal to the posterior variance $\big(n/\sigma^2+1/\tau^2\big)^{-1}\to\sigma^2/n$ as $\tau\to\infty$. Hence $\bar X$ is minimax in every dimension. There is no least favourable prior — the flat "prior" that would play that role is improper.</p>
</div>`,
        lab: 'decision-theory-risk',
      },
      {
        title: 'Admissibility',
        html: R`
<p>Admissibility is a minimal rationality requirement: an inadmissible rule can be replaced by one that is never worse and sometimes better. It is also a weak one: the constant rule $\delta\equiv\theta_0$ is typically admissible, since nothing else achieves zero risk at $\theta_0$.</p>
<div class="thm" data-kind="Theorem" data-name="Unique Bayes rules are admissible">
<p>If $\delta_\pi$ is the unique Bayes rule for $\pi$ (up to rules equal $P_\theta$-a.s. for all $\theta$) and $r(\pi,\delta_\pi)\lt\infty$, then $\delta_\pi$ is admissible.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>If $\delta'$ dominated $\delta_\pi$, then integrating $R(\theta,\delta')\le R(\theta,\delta_\pi)$ against $\pi$ gives $r(\pi,\delta')\le r(\pi,\delta_\pi)$, so $\delta'$ is also Bayes. By uniqueness $\delta'=\delta_\pi$ a.s., so their risk functions coincide — contradicting strict improvement somewhere.</p>
</details>
<p>Under squared error the Bayes rule is unique whenever the Bayes risk is finite (strict convexity of the loss), so <strong>every proper-prior posterior mean with finite Bayes risk is admissible</strong>. A variant drops uniqueness: if $\pi$ gives positive mass to every open set and all risk functions are continuous in $\theta$, a Bayes rule with finite Bayes risk is admissible, since a dominating rule would improve on an open set and hence strictly lower the Bayes risk.</p>
<p>A converse holds in great generality — Wald’s <em>complete class theorem</em>: under compactness and continuity conditions, every admissible rule is Bayes or a limit of Bayes rules (for some possibly improper prior). So, roughly, the admissible rules and the (generalised) Bayes rules are the same class. Admissibility of rules that are only limits of Bayes rules — such as $\bar X$ — must be checked case by case, typically by Blyth’s method; $\bar X$ is admissible for a normal mean in dimensions 1 and 2. In dimension 3 it is not.</p>`,
      },
      {
        title: 'Stein’s paradox and the James–Stein estimator',
        html: R`
<p>Let $X\sim\N_d(\theta,I_d)$ — one observation of each of $d$ unrelated normal means — under the loss $\|\delta-\theta\|^2$. The obvious estimator $\delta_0(X) = X$ is the MLE, the UMVUE, minimax, and equivariant, with constant risk $d$. Stein (1956) showed it is inadmissible for $d\ge3$; James and Stein (1961) gave an explicit improvement.</p>
<div class="thm" data-kind="Lemma" data-name="Stein’s identity">
<p>If $Y\sim\N(\mu,1)$ and $g:\R\to\R$ is absolutely continuous with $\E|g'(Y)|\lt\infty$, then $\E[(Y-\mu)g(Y)] = \E[g'(Y)]$. In $\R^d$ with $X\sim\N_d(\theta,I)$ and weakly differentiable $g:\R^d\to\R^d$: $\E[(X-\theta)^\top g(X)] = \E[\nabla\cdot g(X)]$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The standard normal density satisfies $\varphi'(z) = -z\varphi(z)$. Integrating by parts, $\E[g'(Y)] = \int g'(y)\varphi(y-\mu)\d y = \big[g\varphi\big]_{-\infty}^{\infty} + \int g(y)(y-\mu)\varphi(y-\mu)\d y$; the boundary term vanishes under the integrability condition. The multivariate version applies the scalar identity coordinate-wise, conditioning on the other coordinates.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="James–Stein (1961)">
<p>For $d\ge3$, the estimator
$$\delta_{JS}(X) = \Big(1-\frac{d-2}{\|X\|^2}\Big)X$$
has risk
$$R(\theta,\delta_{JS}) = d-(d-2)^2\,\E_\theta\Big[\frac{1}{\|X\|^2}\Big]\lt d = R(\theta,X)\quad\text{for every }\theta\in\R^d.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $\delta = X - g(X)$ with $g(x) = c\,x/\|x\|^2$. Then
$$\E\|\delta-\theta\|^2 = \E\|X-\theta\|^2 - 2\E[(X-\theta)^\top g(X)] + \E\|g(X)\|^2 = d - 2\E[\nabla\cdot g(X)] + c^2\E\big[\|X\|^{-2}\big].$$
Compute $\partial_i\big(x_i/\|x\|^2\big) = 1/\|x\|^2 - 2x_i^2/\|x\|^4$; summing over $i$, $\nabla\cdot g = c(d-2)/\|x\|^2$. (Weak differentiability and $\E\|X\|^{-2}\lt\infty$ both hold for $d\ge3$, since $1/\|x\|^2$ is integrable near the origin in three or more dimensions.) So
$$R(\theta,\delta) = d - \big(2c(d-2)-c^2\big)\E\big[\|X\|^{-2}\big],$$
minimised at $c = d-2$, where the bracket is $(d-2)^2\gt0$.</p>
</details>
<p>The improvement is largest at $\theta=0$, where $\|X\|^2\sim\chi^2_d$ and $\E[1/\chi^2_d] = 1/(d-2)$, giving risk exactly $2$ instead of $d$. As $\|\theta\|\to\infty$ the risk returns to $d$. In general $\|X\|^2$ is noncentral $\chi^2_d(\|\theta\|^2)$, a Poisson mixture of central chi-squares, which gives the exact formula
$$\E_\theta\big[\|X\|^{-2}\big] = \sum_{k=0}^\infty e^{-\lambda/2}\frac{(\lambda/2)^k}{k!}\cdot\frac{1}{d+2k-2},\qquad\lambda = \|\theta\|^2,$$
used in the second lab. The risk depends on $\theta$ only through $\|\theta\|$; the shrinkage target (here 0) can be any fixed point, or the grand mean $\bar X\mathbf 1$ (then $d-3$ replaces $d-2$ and $d\ge4$ is needed).</p>
<div class="note" data-name="Why shrinking unrelated means helps">
<p>$\E\|X\|^2 = \|\theta\|^2 + d$: the data vector is systematically <em>too long</em>. Pulling it back toward the origin trades a little bias for a large variance reduction, and the data themselves say how much to pull. Efron & Morris (1973) gave the empirical Bayes reading: if $\theta_i\iid\N(0,\tau^2)$, the Bayes rule is $\big(1-\tfrac1{1+\tau^2}\big)X$. Marginally $\|X\|^2\sim(1+\tau^2)\chi^2_d$, so $\E\big[(d-2)/\|X\|^2\big] = 1/(1+\tau^2)$ exactly — James–Stein plugs in an unbiased estimate of the unknown shrinkage factor. The coordinates are statistically unrelated, but they share a loss function, and that is enough.</p>
</div>
<div class="warn" data-name="James–Stein is itself inadmissible">
<p>When $\|X\|^2\lt d-2$ the factor $1-(d-2)/\|X\|^2$ is negative and $\delta_{JS}$ flips the sign of $X$ — absurd. The <em>positive-part</em> estimator $\big(1-\tfrac{d-2}{\|X\|^2}\big)_+X$ dominates $\delta_{JS}$, and is itself inadmissible (being non-smooth, it is not a generalised Bayes rule), though no practical improvement is known. Also, the gain is in <em>total</em> risk: individual coordinates — especially outlying ones — can have larger risk than under the MLE.</p>
</div>`,
        lab: 'decision-theory-james-stein',
      },
      {
        title: 'Connections',
        html: R`
<ul>
<li><strong>Shrinkage everywhere.</strong> <a href="#/t/ridge-regression">Ridge regression</a>, <a href="#/t/mixed-models">random-effect predictions</a> (BLUPs), and <a href="#/t/hierarchical-mcmc">hierarchical Bayesian models</a> all shrink many parameters toward a common value, and all inherit Stein’s insight: when estimating many things at once, borrowing strength reduces total error.</li>
<li><strong>Stein’s unbiased risk estimate (SURE).</strong> The proof above shows $\|g(X)\|^2 - 2\nabla\cdot g(X) + d$ is an unbiased estimate of the risk of $X - g(X)$ that can be computed from data alone; it is used to tune wavelet thresholds and other smoothing parameters.</li>
<li><strong>Testing.</strong> Under 0–1 loss the risk function of a test <em>is</em> its error-probability function; Neyman–Pearson optimality is a constrained minimax-like criterion, and Bayes tests are likelihood-ratio tests.</li>
<li><strong>Learning theory.</strong> Statistical learning replaces $\theta$ by an unknown distribution and uses minimax risk over function classes to define optimal rates of convergence (see <a href="#/t/learning-framework">the learning framework</a>).</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Why can no estimator have uniformly smallest risk under squared error in the binomial problem?`,
        a: R`For each $\theta_0$, the constant rule $\delta\equiv\theta_0$ has risk 0 at $\theta_0$. A uniformly best rule would have to have risk 0 at every $\theta$, i.e. estimate $\theta$ perfectly from data, which is impossible since $P_\theta$ overlap.` },
      { q: R`Prove that the posterior median is the Bayes rule under absolute-error loss.`,
        a: R`The posterior expected loss is $\rho(a) = \int|\theta-a|\pi(\theta\mid x)\d\theta$, a convex function of $a$ with derivative $\P(\theta\lt a\mid x)-\P(\theta\gt a\mid x)$ (where it exists). This is negative below the posterior median and positive above it, so $\rho$ is minimised at the median; by the posterior-risk theorem the median is Bayes.` },
      { q: R`Verify that the minimax binomial estimator $(X+\sqrt n/2)/(n+\sqrt n)$ has smaller maximum risk than $X/n$, and say where $X/n$ is better.`,
        a: R`Its risk is constant at $1/(4(1+\sqrt n)^2)$, below the MLE’s maximum risk $1/(4n)$ attained at $\theta=\tfrac12$ (since $(1+\sqrt n)^2\gt n$). The MLE’s risk $\theta(1-\theta)/n$ is smaller than the constant near $\theta=0$ and $\theta=1$; the crossing points move toward the endpoints as $n$ grows.` },
      { q: R`What is the risk of the James–Stein estimator at $\theta=0$ in dimension $d=10$, and why does the argument fail in $d=2$?`,
        a: R`At $\theta=0$, $\|X\|^2\sim\chi^2_{10}$ and $\E[1/\chi^2_{10}] = 1/8$, so the risk is $10 - 64/8 = 2$, compared with 10 for the MLE. In $d=2$ the factor $d-2$ is zero (no shrinkage), and $\E\|X\|^{-2} = \infty$ (since $\int_0 r^{-2}\cdot r\d r$ diverges), so Stein’s identity cannot be applied; indeed $X$ is admissible for $d\le2$.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'causal-inference',
    domain: 'inference',
    order: 170,
    title: 'Causal Inference: Potential Outcomes & DAGs',
    subtitle: 'When does an association identify the effect of an intervention?',
    level: 2,
    sources: { aos: 'Ch. 16', cb: '—', other: 'Hernán & Robins, Causal Inference: What If (2020); Pearl (2009); Imbens & Rubin (2015)' },
    prereqs: ['conditional-expectation', 'conditional-probability'],
    related: ['directed-graphical-models', 'permutation-tests', 'importance-sampling', 'simple-linear-regression', 'logistic-regression'],
    tags: ['potential outcomes', 'ATE', 'randomisation', 'confounding', 'back-door criterion', 'propensity score', 'Simpson’s paradox'],
    summary: R`Causal questions ask what would happen under an intervention, not what is observed alongside what. The potential-outcomes framework writes each unit’s response to treatment and to control as $Y(1)$ and $Y(0)$, only one of which is ever seen; the average treatment effect $\E[Y(1)-Y(0)]$ is then a well-defined target that the observed contrast $\E[Y\mid A=1]-\E[Y\mid A=0]$ generally misses. Randomisation closes the gap by making treatment independent of the potential outcomes. In observational data, identification requires that a measured set of covariates blocks every confounding path — the back-door criterion of causal DAGs — after which the effect is recovered by standardisation, inverse probability weighting, or matching. Simpson’s paradox shows the stakes: the right adjustment can reverse the sign of an association.`,
    survey: {
      task: R`Identify and estimate causal effects such as $\E[Y(1)-Y(0)]$ from experimental or observational data`,
      assumptions: R`Consistency/SUTVA; exchangeability (randomisation, or no unmeasured confounding given $X$); positivity`,
      output: 'ATE / ATT estimates via difference in means, standardisation (g-formula), IPW, matching, doubly robust estimators',
      strengths: 'Makes assumptions explicit and testable in part; randomised experiments need no modelling at all',
      weaknesses: 'Key observational assumptions (no unmeasured confounding) are untestable; positivity violations give unstable estimates',
      complexity: 'As for the underlying regression / propensity models',
    },
    sections: [
      {
        title: 'Potential outcomes and the fundamental problem',
        html: R`
<p>Let $A\in\{0,1\}$ be a treatment and $Y$ an outcome. For each unit imagine two <em>potential outcomes</em>: $Y(1)$, the outcome if treated, and $Y(0)$, the outcome if untreated. The individual causal effect is $Y(1)-Y(0)$.</p>
<div class="def" data-name="Causal estimands">
<p>The <em>average treatment effect</em> is $\text{ATE} = \E[Y(1)-Y(0)]$; the effect on the treated is $\text{ATT} = \E[Y(1)-Y(0)\mid A=1]$; the conditional effect is $\tau(x) = \E[Y(1)-Y(0)\mid X=x]$.</p>
</div>
<p>To connect potential outcomes with data we need <strong>consistency</strong>: the observed outcome is the potential outcome under the treatment actually received, $Y = AY(1)+(1-A)Y(0)$. This encodes SUTVA — no interference between units and a single well-defined version of each treatment. The <em>fundamental problem of causal inference</em> (Holland, 1986) is that for each unit we see one potential outcome and never the other: causal inference is a missing-data problem.</p>
<p>The naive comparison is the <em>association</em> $\alpha = \E[Y\mid A=1]-\E[Y\mid A=0]$. By consistency,
$$\alpha = \underbrace{\E[Y(1)-Y(0)\mid A=1]}_{\text{ATT}} \;+\; \underbrace{\E[Y(0)\mid A=1]-\E[Y(0)\mid A=0]}_{\text{selection bias}}.$$
The bias term asks whether the treated would have fared differently from the untreated <em>even without treatment</em>. If sicker patients get the drug, it is negative and can mask or reverse a real benefit.</p>
<div class="note" data-name="Causation is a contrast between worlds">
<p>$\E[Y\mid A=1]$ averages over the people who happened to be treated. $\E[Y(1)]$ averages over everyone, in a hypothetical world where everyone is treated. In Pearl’s notation the latter is $\E[Y\mid\text{do}(A=1)]$: conditioning on seeing versus conditioning on doing.</p>
</div>`,
      },
      {
        title: 'Randomisation identifies the causal effect',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Randomisation">
<p>If treatment is assigned independently of the potential outcomes, $A\indep(Y(0),Y(1))$, with $0\lt\P(A=1)\lt1$, then under consistency
$$\E[Y\mid A=1]-\E[Y\mid A=0] = \E[Y(1)]-\E[Y(0)] = \text{ATE}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\E[Y\mid A=1] = \E[Y(1)\mid A=1]$ by consistency, and $\E[Y(1)\mid A=1] = \E[Y(1)]$ by independence. Likewise for $A=0$. The selection-bias term above is zero because $\E[Y(0)\mid A=1] = \E[Y(0)] = \E[Y(0)\mid A=0]$.</p>
</details>
<p>The theorem requires independence of $A$ from the <em>potential outcomes</em>, not from the observed $Y$ (which depends on $A$ whenever there is an effect). Randomisation achieves this for every characteristic of the units, measured or not; that is why randomised experiments are the gold standard.</p>
<p>In a completely randomised experiment with $n_1$ treated and $n_0$ controls, the difference in means $\hat\tau = \bar Y_1-\bar Y_0$ is unbiased for the (sample) ATE. Neyman (1923) showed its randomisation variance is
$$\Var(\hat\tau) = \frac{S_1^2}{n_1}+\frac{S_0^2}{n_0}-\frac{S_\tau^2}{n},$$
where $S_1^2,S_0^2$ are the finite-population variances of the potential outcomes and $S_\tau^2$ that of the unit-level effects. The last term involves the joint distribution of $(Y(0),Y(1))$, never observed, so the usual estimate $s_1^2/n_1+s_0^2/n_0$ is <strong>conservative</strong> — exact when effects are constant. Alternatively, Fisher’s sharp null $Y_i(1)=Y_i(0)$ for all $i$ fixes every missing potential outcome, and the exact randomisation distribution of any statistic follows by re-randomising: the <a href="#/t/permutation-tests">permutation test</a>.</p>`,
      },
      {
        title: 'Observational studies: adjustment, weighting, propensity scores',
        html: R`
<p>Without randomisation we hope treatment is "as good as random" within levels of measured covariates $X$.</p>
<div class="def" data-name="Identifying assumptions">
<p><strong>Conditional exchangeability</strong> (no unmeasured confounding): $(Y(0),Y(1))\indep A\mid X$. <strong>Positivity</strong> (overlap): $0\lt e(x)\lt1$ for all $x$ in the support of $X$, where $e(x) = \P(A=1\mid X=x)$ is the <em>propensity score</em>. Plus consistency.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Identification by adjustment and by weighting">
<p>Under consistency, conditional exchangeability and positivity, for $a\in\{0,1\}$,
$$\E[Y(a)] = \E_X\big[\E[Y\mid A=a,X]\big] \qquad\text{(standardisation / g-formula)}$$
$$\E[Y(1)] = \E\Big[\frac{AY}{e(X)}\Big],\qquad \E[Y(0)] = \E\Big[\frac{(1-A)Y}{1-e(X)}\Big]\qquad\text{(inverse probability weighting)}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Standardisation.</em> $\E[Y(a)] = \E\big[\E[Y(a)\mid X]\big] = \E\big[\E[Y(a)\mid A=a,X]\big] = \E\big[\E[Y\mid A=a,X]\big]$: tower property, then exchangeability, then consistency. Positivity guarantees that $\E[Y\mid A=a,X=x]$ is defined for every $x$ in the support.</p>
<p><em>Weighting.</em> $AY = AY(1)$ by consistency. Conditioning on $X$ and $Y(1)$, $\E\big[\tfrac{AY(1)}{e(X)}\big] = \E\big[\tfrac{Y(1)}{e(X)}\E[A\mid X,Y(1)]\big] = \E\big[\tfrac{Y(1)}{e(X)}e(X)\big] = \E[Y(1)]$, where exchangeability gives $\E[A\mid X,Y(1)] = \E[A\mid X] = e(X)$ and positivity makes the division legitimate.</p>
</details>
<p>Standardisation models the outcome; IPW models treatment assignment, creating a pseudo-population in which $A$ is independent of $X$ (each treated unit stands for $1/e(X)$ people). <em>Augmented IPW</em> combines both and is <strong>doubly robust</strong>: consistent if either the outcome model or the propensity model is correct, and semiparametrically efficient when both are.</p>
<div class="thm" data-kind="Theorem" data-name="The propensity score is a balancing score (Rosenbaum & Rubin, 1983)">
<p>(a) $X\indep A\mid e(X)$. (b) If $(Y(0),Y(1))\indep A\mid X$, then $(Y(0),Y(1))\indep A\mid e(X)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(a) $\P(A=1\mid X,e(X)) = \P(A=1\mid X) = e(X)$, and $\P(A=1\mid e(X)) = \E[\P(A=1\mid X)\mid e(X)] = e(X)$; these agree, so $A$ is independent of $X$ given $e(X)$. (b) Write $Y(\cdot) = (Y(0),Y(1))$. Then $\P(A=1\mid Y(\cdot),e(X)) = \E\big[\P(A=1\mid Y(\cdot),X)\mid Y(\cdot),e(X)\big] = \E[e(X)\mid Y(\cdot),e(X)] = e(X)$, which does not depend on $Y(\cdot)$.</p>
</details>
<p>So adjustment for a possibly high-dimensional $X$ can be reduced to adjustment for a scalar: matching or stratifying on $e(X)$ suffices. In practice $e$ is estimated (e.g. by <a href="#/t/logistic-regression">logistic regression</a>) and covariate balance is checked after matching or weighting.</p>`,
      },
      {
        title: 'Simpson’s paradox and confounding',
        html: R`
<p>Consider a hypothetical observational study of a treatment where disease severity $Z$ influences both who is treated and who recovers.</p>
<table>
<thead><tr><th></th><th>Treated: recovered / total</th><th>Untreated: recovered / total</th></tr></thead>
<tbody>
<tr><td>Mild cases</td><td>45 / 50 = 90%</td><td>170 / 200 = 85%</td></tr>
<tr><td>Severe cases</td><td>120 / 200 = 60%</td><td>25 / 50 = 50%</td></tr>
<tr><td><strong>All cases</strong></td><td><strong>165 / 250 = 66%</strong></td><td><strong>195 / 250 = 78%</strong></td></tr>
</tbody>
</table>
<p>Treatment is better in each stratum, yet worse overall — <em>Simpson’s paradox</em>. The arithmetic is simple: the treated group is 80% severe cases and the untreated group 20%, and severity dominates the comparison. Standardising to the overall severity distribution (half mild, half severe) gives treated $0.5(0.90)+0.5(0.60) = 0.75$ and untreated $0.5(0.85)+0.5(0.50) = 0.675$: a benefit of 7.5 percentage points, against a crude difference of $-12$.</p>
<div class="warn" data-name="The data cannot tell you which table to believe">
<p>Exactly the same numbers could arise if $Z$ were a <em>consequence</em> of treatment — say, a side effect that makes cases severe. Then $Z$ is a mediator, the stratified comparison blocks part of the treatment’s effect, and the crude comparison is the right estimate of the total effect. Whether to adjust is a question about the causal structure, not about the joint distribution. This is the central lesson that DAGs formalise.</p>
</div>`,
        lab: 'causal-inference-simpson',
      },
      {
        title: 'Causal DAGs and the back-door criterion',
        html: R`
<p>A causal <a href="#/t/directed-graphical-models">directed acyclic graph</a> encodes assumptions: each variable is generated from its parents and independent noise (a structural causal model), and absent arrows are claims of no direct effect. The joint law factorises as $p(v) = \prod_j p(v_j\mid\text{pa}_j)$. An intervention $\text{do}(A=a)$ replaces $A$’s mechanism by the constant $a$, deleting the arrows into $A$, which gives the <em>truncated factorisation</em>
$$p(v\mid\text{do}(A=a)) = \prod_{j:\,V_j\ne A}p(v_j\mid\text{pa}_j)\Big|_{A=a}.$$</p>
<p>Three elementary structures determine whether a path between $A$ and $Y$ transmits association:</p>
<div class="cols">
<div><h4>Fork (confounder)</h4>$$A\leftarrow Z\rightarrow Y$$<p>$Z$ creates non-causal association. Path open; <strong>conditioning on $Z$ blocks it</strong>.</p></div>
<div><h4>Chain (mediator)</h4>$$A\rightarrow M\rightarrow Y$$<p>Part of the causal effect. Open; conditioning on $M$ blocks it — removing part of the effect you want.</p></div>
<div><h4>Collider</h4>$$A\rightarrow C\leftarrow Y$$<p>Path <em>blocked</em> by default; <strong>conditioning on $C$ (or a descendant) opens it</strong> and creates spurious association.</p></div>
</div>
<p>A path is <em>blocked</em> by a set $Z$ if it contains a non-collider in $Z$, or a collider such that neither it nor any descendant is in $Z$ (d-separation).</p>
<div class="thm" data-kind="Theorem" data-name="Back-door adjustment (Pearl, 1993)">
<p>A set $Z$ satisfies the <em>back-door criterion</em> relative to $(A,Y)$ if (i) no node in $Z$ is a descendant of $A$, and (ii) $Z$ blocks every path between $A$ and $Y$ that begins with an arrow into $A$. If $Z$ satisfies it, then
$$p(y\mid\text{do}(A=a)) = \sum_z p(y\mid A=a,Z=z)\,p(z),$$
and in potential-outcome terms $Y(a)\indep A\mid Z$.</p>
</div>
<details class="proof"><summary>Proof sketch (for $Z = \text{pa}(A)$)</summary>
<p>The parents of $A$ always satisfy the criterion. By truncated factorisation, the intervention changes only $A$’s factor $p(a\mid z)$. The marginal of $Z$ is unchanged, because $Z$ consists of non-descendants of $A$. And $p(y\mid a,z)$ is unchanged: given $A$ and its parents, $Y$ depends on $A$’s mechanism only through the value of $A$ (the mechanism’s noise is independent of everything else, and every back-door path goes through $\text{pa}(A)$). Hence $p(y\mid\text{do}(a)) = \sum_zp(z)p(y\mid a,z)$. The general case reduces to this one by showing that any back-door set renders $\text{pa}(A)$ ignorable via d-separation arguments.</p>
</details>
<p>The back-door formula is the g-formula of the previous section, now with a graphical test for which $Z$ to use. It also tells you what <em>not</em> to adjust for: descendants of $A$ (mediators, post-treatment variables) and colliders. When no back-door set is observed, other strategies — the front-door criterion, instrumental variables, do-calculus — may still identify the effect. The second lab simulates the three structures and compares adjusted and unadjusted regression estimates with the truth.</p>`,
        lab: 'causal-inference-dag',
      },
      {
        title: 'Pitfalls and what to do about them',
        html: R`
<div class="warn" data-name="Unmeasured confounding">
<p>Conditional exchangeability is untestable from the observed data. Sensitivity analysis quantifies how strong an unmeasured confounder would have to be to explain away an estimate; for example, the E-value of VanderWeele & Ding (2017) for an observed risk ratio $RR\gt1$ is $RR+\sqrt{RR(RR-1)}$: the minimum strength of association, on the risk-ratio scale, that a confounder would need with both treatment and outcome.</p>
</div>
<div class="warn" data-name="Adjusting for too much">
<p>Adding every available covariate is not safe. Conditioning on a collider — e.g. studying only hospitalised patients when both the exposure and the disease cause hospitalisation (Berkson’s bias) — manufactures association. Adjusting for a mediator answers a different question (a controlled direct effect, itself identified only under further assumptions). Adjusting for an instrument (a cause of $A$ that affects $Y$ only through $A$) inflates variance and amplifies any residual confounding bias.</p>
</div>
<div class="warn" data-name="Positivity">
<p>If some covariate strata almost never receive treatment, $1/e(X)$ explodes and IPW estimates become unstable; the g-formula then extrapolates a model into regions with no data. Trimming or changing the estimand (to the ATT, or to an overlap-weighted population) are honest responses.</p>
</div>
<div class="warn" data-name="The Table 2 fallacy">
<p>In a regression adjusting for confounders of $A\to Y$, the coefficients of the confounders themselves are generally not causal effects of anything — the adjustment set was chosen for $A$, not for them.</p>
</div>
<p>Beyond adjustment, the design-based toolkit — instrumental variables, regression discontinuity, difference-in-differences, synthetic controls — trades the no-unmeasured-confounding assumption for others that may be more credible in a given application.</p>`,
      },
    ],
    quiz: [
      { q: R`Decompose the observed difference $\E[Y\mid A=1]-\E[Y\mid A=0]$ into a causal effect and a bias term, and explain why randomisation removes the bias.`,
        a: R`It equals $\text{ATT} + \{\E[Y(0)\mid A=1]-\E[Y(0)\mid A=0]\}$. The second term compares the untreated potential outcomes of the treated and untreated groups. Under randomisation $A\indep Y(0)$, so both conditional means equal $\E[Y(0)]$ and the bias vanishes; also ATT = ATE.` },
      { q: R`Prove that $\E[AY/e(X)] = \E[Y(1)]$ under the identifying assumptions. Which assumption does each step use?`,
        a: R`$AY = AY(1)$ (consistency). $\E[AY(1)/e(X)] = \E\big[Y(1)\E[A\mid X,Y(1)]/e(X)\big]$ (tower property), $\E[A\mid X,Y(1)] = e(X)$ (conditional exchangeability), and the ratio $e(X)/e(X) = 1$ requires $e(X)\gt0$ (positivity).` },
      { q: R`In the DAG $A\to Y$, $A\to C\leftarrow Y$, what happens to the regression coefficient of $Y$ on $A$ if you add $C$ as a covariate?`,
        a: R`$C$ is a collider on the path $A\to C\leftarrow Y$. Unadjusted, this path is blocked and the coefficient estimates the causal effect. Conditioning on $C$ opens it, inducing an association between $A$ and $Y$ given $C$ that is not causal, so the adjusted coefficient is biased (typically toward a negative association when both arrows into $C$ are positive).` },
      { q: R`In the Simpson’s-paradox table, why is the standardised difference the right answer if severity is a pre-treatment confounder but not if it is caused by treatment?`,
        a: R`If severity precedes treatment and affects both treatment choice and recovery, it opens a back-door path $A\leftarrow Z\to Y$, which standardisation blocks. If treatment causes severity, $Z$ is a mediator (a descendant of $A$), violating the back-door criterion; stratifying on it removes part of the treatment’s effect, and the crude comparison (with no back-door paths to block) is the total causal effect.` },
    ],
  });
})();
