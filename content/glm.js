/* Generalized Linear Models. Topics: glm-framework, logistic-regression, poisson-regression, irls, deviance,
   glm-inference, overdispersion, multinomial-ordinal, glmm-gee, gam. Labs live in js/labs/glm.js. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'glm-framework',
    domain: 'glm',
    order: 10,
    title: 'The GLM Framework: Family, Linear Predictor & Link',
    subtitle: 'One likelihood machine for Gaussian, binary, count and positive responses',
    level: 2,
    sources: { aos: '13.7', cb: '§12.3', other: 'Nelder & Wedderburn (1972); McCullagh & Nelder (1989), Ch. 2; Agresti (2015), Ch. 4' },
    prereqs: ['exponential-families', 'mle'],
    related: ['ols-geometry', 'logistic-regression', 'poisson-regression', 'irls'],
    tags: ['exponential dispersion family', 'link function', 'variance function', 'canonical link'],
    summary: R`A generalized linear model has three separable parts: a <em>random component</em> (the response follows an exponential dispersion family with mean $\mu_i$), a <em>systematic component</em> (a linear predictor $\eta_i = \bx_i^\top\bbeta$), and a <em>link</em> $g(\mu_i) = \eta_i$ that joins them. The family fixes how the variance depends on the mean, $\Var(Y_i) = \phi\,V(\mu_i)/w_i$; the link fixes how covariates act on the mean. Because everything is driven by the cumulant function $b(\theta)$, one set of formulas — the score $X^\top W G(\by-\bmu)/\phi$, the information $X^\top W X/\phi$, the IRLS algorithm, the deviance — covers linear, logistic, Poisson, gamma and inverse-Gaussian regression at once.`,
    survey: {
      task: 'Regression for responses whose mean is constrained and whose variance depends on the mean',
      assumptions: R`Independent $Y_i$ from an exponential dispersion family; $g(\mu_i) = \bx_i^\top\bbeta$ correctly specified`,
      output: R`$\hat\bbeta$, $\widehat{\Cov}(\hat\bbeta) = \hat\phi(X^\top \hat W X)^{-1}$, fitted means $\hat\mu_i$, deviance`,
      strengths: 'Unified theory and software; respects the support of the response; interpretable coefficients on the link scale',
      weaknesses: 'Mean–variance relation is imposed by the family; link and linearity can be wrong; independence assumed',
      complexity: R`$O(np^2)$ per IRLS iteration; typically 4–10 iterations`,
    },
    sections: [
      {
        title: 'Why linear regression is not enough',
        html: R`
<p>Ordinary least squares models $\E(Y\mid\bx) = \bx^\top\bbeta$ with constant variance. Three kinds of data break this immediately:</p>
<ul>
<li><strong>Binary outcomes</strong> ($Y\in\{0,1\}$): the mean is a probability, so a linear predictor that ranges over $\R$ will produce fitted "probabilities" below 0 or above 1, and $\Var(Y\mid\bx) = \mu(1-\mu)$ is not constant.</li>
<li><strong>Counts</strong> ($Y\in\{0,1,2,\dots\}$): the mean must be positive, effects are naturally multiplicative, and the variance typically grows with the mean.</li>
<li><strong>Positive continuous data</strong> (costs, durations, concentrations): skewed, with standard deviation roughly proportional to the mean.</li>
</ul>
<p>The classical fix — transform $Y$ and run OLS — asks a single transformation to do two jobs at once: make the mean linear in $\bx$ <em>and</em> make the variance constant. There is no reason one function should do both, and the transformation changes the estimand: $\E(\log Y\mid\bx) \neq \log \E(Y\mid\bx)$ (Jensen), and $\log 0$ is undefined.</p>
<div class="note">
<p>Nelder and Wedderburn (1972) separated the two jobs. The <em>link</em> $g$ is applied to the <em>mean</em>, not to the data, and only has to make the mean linear. The <em>family</em> supplies the variance through a variance function $V(\mu)$. Fitting is by maximum likelihood, which turns out to be iteratively reweighted least squares. Linear regression is the special case "Normal family, identity link".</p>
</div>`,
      },
      {
        title: 'The random component: exponential dispersion families',
        html: R`
<p>The <a href="#/t/exponential-families">exponential family</a> topic develops one-parameter natural families $p(y\mid\theta) = h(y)\exp\{\theta T(y) - A(\theta)\}$. GLMs need a slight generalization that adds a nuisance scale.</p>
<div class="def" data-name="Exponential dispersion family (EDF)">
<p>A random variable $Y$ belongs to an exponential dispersion family if its density (with respect to Lebesgue or counting measure) is
$$f(y;\theta,\phi) = \exp\!\left\{\frac{y\theta - b(\theta)}{a(\phi)} + c(y,\phi)\right\},$$
where $\theta$ is the <em>natural (canonical) parameter</em>, $\phi > 0$ the <em>dispersion parameter</em>, $b$ the <em>cumulant function</em>, and usually $a(\phi) = \phi/w$ for a known prior weight $w$ (e.g. a binomial denominator).</p>
</div>
<p>For fixed $\phi$ this is a natural exponential family in $\theta$ with sufficient statistic $y$. All moments are encoded in $b$:</p>
<div class="thm" data-kind="Theorem" data-name="Mean and variance from the cumulant function">
<p>If $\theta$ lies in the interior of the natural parameter space, the cumulant generating function of $Y$ is
$$K(t) = \log \E e^{tY} = \frac{b(\theta + t\,a(\phi)) - b(\theta)}{a(\phi)}$$
for $t$ near 0. Consequently $\E Y = \mu = b'(\theta)$ and $\Var Y = a(\phi)\, b''(\theta)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $a = a(\phi)$ and $\theta' = \theta + ta$, which is still in the natural parameter space for small $|t|$. Then
$$\E e^{tY} = \int \exp\!\left\{\frac{y\theta - b(\theta)}{a} + ty + c(y,\phi)\right\}\d y = \exp\!\left\{\frac{b(\theta') - b(\theta)}{a}\right\}\int \exp\!\left\{\frac{y\theta' - b(\theta')}{a} + c(y,\phi)\right\}\d y.$$
The last integral is 1 because it integrates the EDF density at parameter $\theta'$. Taking logs gives $K(t)$. Differentiating, $K'(t) = b'(\theta + ta)$ and $K''(t) = a\,b''(\theta+ta)$; at $t=0$ these are the first two cumulants, the mean and variance.</p>
</details>
<p>Because $\Var Y > 0$ for a non-degenerate distribution, $b'' > 0$: $b$ is strictly convex, so $\mu = b'(\theta)$ is strictly increasing and invertible. The variance can therefore be written as a function of the mean.</p>
<div class="def" data-name="Variance function">
<p>$V(\mu) = b''\big((b')^{-1}(\mu)\big)$, so that $\Var(Y) = a(\phi) V(\mu) = \phi V(\mu)/w$. The variance function characterizes the natural exponential family within the class of NEFs (Morris, 1982, for quadratic $V$; Jørgensen, 1987, in general).</p>
</div>
<table>
<thead><tr><th>Family</th><th>$\theta$</th><th>$b(\theta)$</th><th>$\phi$</th><th>$V(\mu)$</th><th>Canonical link</th></tr></thead>
<tbody>
<tr><td>Normal $\N(\mu,\sigma^2)$</td><td>$\mu$</td><td>$\theta^2/2$</td><td>$\sigma^2$</td><td>$1$</td><td>identity</td></tr>
<tr><td>Binomial proportion $Y = S/m$</td><td>$\log\frac{\mu}{1-\mu}$</td><td>$\log(1+e^\theta)$</td><td>$1$ ($w = m$)</td><td>$\mu(1-\mu)$</td><td>logit</td></tr>
<tr><td>Poisson$(\mu)$</td><td>$\log\mu$</td><td>$e^\theta$</td><td>$1$</td><td>$\mu$</td><td>log</td></tr>
<tr><td>Gamma (mean $\mu$, shape $\nu$)</td><td>$-1/\mu$</td><td>$-\log(-\theta)$</td><td>$1/\nu$</td><td>$\mu^2$</td><td>inverse ($-1/\mu$)</td></tr>
<tr><td>Inverse Gaussian</td><td>$-1/(2\mu^2)$</td><td>$-\sqrt{-2\theta}$</td><td>$1/\lambda$</td><td>$\mu^3$</td><td>$1/\mu^2$</td></tr>
</tbody>
</table>
<div class="ex" data-name="Checking the Poisson and gamma rows">
<p>Poisson: $\log f = y\log\mu - \mu - \log y!$, so $\theta = \log\mu$, $b(\theta) = e^\theta$, $a = 1$. Then $b'(\theta) = e^\theta = \mu$ and $b''(\theta) = e^\theta = \mu$: mean equals variance.</p>
<p>Gamma with shape $\nu$ and mean $\mu$: $\log f = \nu(-y/\mu - \log\mu) + (\nu-1)\log y + \nu\log\nu - \log\Gamma(\nu)$. With $\theta = -1/\mu$ we have $-\log\mu = \log(-\theta)$, so $b(\theta) = -\log(-\theta)$ and $a(\phi) = 1/\nu$. Then $b'(\theta) = -1/\theta = \mu$ and $b''(\theta) = 1/\theta^2 = \mu^2$, so $\Var Y = \mu^2/\nu$: a constant coefficient of variation $1/\sqrt\nu$.</p>
</div>`,
        lab: 'glm-framework-cumulant',
      },
      {
        title: 'The systematic component and the link',
        html: R`
<div class="def" data-name="Generalized linear model">
<p>Responses $Y_1,\dots,Y_n$ are independent, $Y_i$ from an EDF with natural parameter $\theta_i$, common dispersion $\phi$, and known weights $w_i$. The mean $\mu_i = b'(\theta_i)$ is related to covariates by
$$g(\mu_i) = \eta_i = \bx_i^\top\bbeta,$$
where the <em>link</em> $g$ is a known, strictly monotone, twice-differentiable function from the range of $\mu$ onto (a subset of) $\R$.</p>
</div>
<p>The chain of maps is $\bbeta \mapsto \eta \overset{g^{-1}}{\mapsto} \mu \overset{(b')^{-1}}{\mapsto} \theta$. A coefficient $\beta_j$ is the change in $g(\mu)$ per unit change in $x_j$ holding the other covariates fixed, which is why the choice of link determines how a coefficient is read: additive on the mean (identity), multiplicative on the mean (log), multiplicative on the odds (logit).</p>
<table>
<thead><tr><th>Link</th><th>$g(\mu)$</th><th>$\mu = g^{-1}(\eta)$</th><th>Typical use</th></tr></thead>
<tbody>
<tr><td>identity</td><td>$\mu$</td><td>$\eta$</td><td>Normal; risk differences</td></tr>
<tr><td>log</td><td>$\log\mu$</td><td>$e^\eta$</td><td>counts, rates, positive responses; rate/risk ratios</td></tr>
<tr><td>logit</td><td>$\log\frac{\mu}{1-\mu}$</td><td>$\frac{1}{1+e^{-\eta}}$</td><td>binary; odds ratios</td></tr>
<tr><td>probit</td><td>$\Phi^{-1}(\mu)$</td><td>$\Phi(\eta)$</td><td>binary with a latent Gaussian threshold</td></tr>
<tr><td>complementary log-log</td><td>$\log\{-\log(1-\mu)\}$</td><td>$1 - \exp(-e^\eta)$</td><td>binary from a latent Poisson count; grouped survival</td></tr>
<tr><td>inverse</td><td>$1/\mu$</td><td>$1/\eta$</td><td>gamma</td></tr>
</tbody>
</table>
<p>Logit and probit are symmetric about $\mu = 1/2$; cloglog is not — it approaches 1 much faster than it leaves 0. The cloglog link arises exactly when $Y = \ind(N > 0)$ with $N\sim\text{Poisson}(e^{\eta})$: then $\P(Y=1) = 1 - \exp(-e^\eta)$, so $\beta_j$ is a log rate ratio for the underlying count.</p>
<div class="def" data-name="Canonical link">
<p>The link $g = (b')^{-1}$, for which $\theta_i = \eta_i = \bx_i^\top\bbeta$. The natural parameter itself is linear in the covariates.</p>
</div>
<p>The lab fits the same binary data with five links. Notice that within the data range the logit, probit and cloglog curves are hard to tell apart (their deviances differ little), while the identity link produces fitted values that leave $[0,1]$ and the log link can only fit a curve that bends one way.</p>`,
        lab: 'glm-framework-links',
      },
      {
        title: 'Likelihood, score and Fisher information',
        html: R`
<p>Let $a_i(\phi) = \phi/w_i$. Up to terms free of $\bbeta$ the log-likelihood is
$$\ell(\bbeta) = \sum_{i=1}^n \frac{w_i\{y_i\theta_i - b(\theta_i)\}}{\phi},\qquad \theta_i = (b')^{-1}\big(g^{-1}(\bx_i^\top\bbeta)\big).$$
Define the diagonal matrices $G = \diag\{g'(\mu_i)\}$ and
$$W = \diag\{W_i\},\qquad W_i = \frac{w_i}{V(\mu_i)\, g'(\mu_i)^2}.$$</p>
<div class="thm" data-kind="Theorem" data-name="Score and expected information">
<p>For any link,
$$U(\bbeta) = \frac{\partial\ell}{\partial\bbeta} = \frac1\phi\sum_{i=1}^n \frac{w_i (y_i - \mu_i)}{V(\mu_i)\,g'(\mu_i)}\,\bx_i = \frac1\phi X^\top W G(\by - \bmu),\qquad \mathcal I(\bbeta) = \Cov\big(U(\bbeta)\big) = \frac1\phi X^\top W X.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the chain rule, $\dfrac{\partial \ell_i}{\partial\beta_j} = \dfrac{\partial\ell_i}{\partial\theta_i}\dfrac{\partial\theta_i}{\partial\mu_i}\dfrac{\partial\mu_i}{\partial\eta_i}\dfrac{\partial\eta_i}{\partial\beta_j}$. The four factors are: $\partial\ell_i/\partial\theta_i = w_i(y_i - b'(\theta_i))/\phi = w_i(y_i-\mu_i)/\phi$; $\partial\mu_i/\partial\theta_i = b''(\theta_i) = V(\mu_i)$, so $\partial\theta_i/\partial\mu_i = 1/V(\mu_i)$; $\partial\mu_i/\partial\eta_i = 1/g'(\mu_i)$ since $\eta = g(\mu)$; and $\partial\eta_i/\partial\beta_j = x_{ij}$. Multiplying gives the score. Since $\E Y_i = \mu_i$ the score has mean zero, and by independence with $\Var Y_i = \phi V(\mu_i)/w_i$,
$$\Cov(U) = \sum_i \frac{w_i^2\,\phi V(\mu_i)/w_i}{\phi^2 V(\mu_i)^2 g'(\mu_i)^2}\bx_i\bx_i^\top = \frac1\phi\sum_i W_i\bx_i\bx_i^\top.$$
By the information identity this also equals $-\E\,\partial^2\ell/\partial\bbeta\partial\bbeta^\top$.</p>
</details>
<p>Two features of these formulas carry the rest of the domain. First, <strong>the score depends on the distribution only through $\mu_i$ and $V(\mu_i)$</strong>; the full density never appears. That is the seed of <a href="#/t/overdispersion">quasi-likelihood</a>, and it is why the MLE of $\bbeta$ remains consistent whenever the <em>mean</em> model is right. Second, the information has the weighted-least-squares form $X^\top W X$, which is why <a href="#/t/irls">Fisher scoring becomes iteratively reweighted least squares</a> and why the large-sample covariance of $\hat\bbeta$ is $\phi(X^\top W X)^{-1}$, the GLM analogue of $\sigma^2(X^\top X)^{-1}$.</p>
<div class="note" data-name="Why the weights look like that">
<p>Linearize the link around $\mu_i$: $g(Y_i) \approx \eta_i + g'(\mu_i)(Y_i - \mu_i)$. The "working" response on the linear-predictor scale therefore has variance $g'(\mu_i)^2\,\phi V(\mu_i)/w_i = \phi/W_i$. So $W_i$ is exactly the precision of observation $i$ on the scale where the model is linear, and $X^\top W X/\phi$ is the usual precision matrix of a weighted regression.</p>
</div>`,
      },
      {
        title: 'Canonical links: sufficiency and concavity',
        html: R`
<p>With the canonical link, $\theta_i = \eta_i$ and $g'(\mu) = 1/V(\mu)$, so $W_i = w_i V(\mu_i)$ and the log-likelihood collapses to
$$\ell(\bbeta) = \frac1\phi\Big\{\bbeta^\top X^\top \mathbf{w}\!\circ\!\by - \sum_i w_i\, b(\bx_i^\top\bbeta)\Big\}.$$</p>
<div class="thm" data-kind="Theorem" data-name="Properties of the canonical link">
<p>Under the canonical link, with $X$ of full column rank $p$:</p>
<ol>
<li>$T = X^\top \mathrm{diag}(w)\by$ (just $X^\top\by$ with unit weights) is a sufficient statistic for $\bbeta$, and the model is a $p$-parameter exponential family in $\bbeta$.</li>
<li>The Hessian is $-\phi^{-1}X^\top W X$, which does not involve $\by$: observed and expected information coincide, so Newton–Raphson and Fisher scoring are the same algorithm.</li>
<li>$\ell$ is strictly concave, so the MLE, if it exists, is unique and is the unique solution of the score equations $X^\top \mathrm{diag}(w)(\by - \hat\bmu) = \mathbf 0$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) The density of the sample is $\exp\{\phi^{-1}(\bbeta^\top T - \sum_i w_i b(\bx_i^\top\bbeta)) + \sum_i c(y_i,\phi)\}$; the factorization theorem gives sufficiency of $T$, and the form is a natural exponential family with natural parameter $\bbeta$. (2) Differentiating, $\partial\ell/\partial\bbeta = \phi^{-1}\sum_i w_i(y_i - b'(\bx_i^\top\bbeta))\bx_i$ and $\partial^2\ell/\partial\bbeta\partial\bbeta^\top = -\phi^{-1}\sum_i w_i b''(\bx_i^\top\bbeta)\bx_i\bx_i^\top = -\phi^{-1}X^\top W X$, with no $y$ in it. (3) $b'' > 0$ makes $W$ positive definite on its diagonal, so for $\mathbf v \neq 0$, $\mathbf v^\top X^\top W X\mathbf v = \sum_i W_i (\bx_i^\top\mathbf v)^2 > 0$ because $X\mathbf v \neq 0$ by full rank. A strictly concave function has at most one maximizer, and at an interior maximizer the gradient vanishes.</p>
</details>
<p>The score equations have a geometric reading worth remembering: the residual vector $\by - \hat\bmu$ is orthogonal to every column of $X$ (in the $w$-weighted inner product). With an intercept, $\sum_i w_i y_i = \sum_i w_i\hat\mu_i$; with a categorical factor, fitted totals match observed totals within every level. These "margin-matching" identities make canonical-link fits easy to sanity-check.</p>
<div class="warn" data-name="Canonical is convenient, not compulsory">
<p>The canonical link is mathematically tidy but carries no guarantee of fitting better. Probit and cloglog are routinely used for binary data, the log link for gamma data (the canonical inverse link does not keep $\mu > 0$ automatically), and the identity link for binomial data when risk differences are the estimand. For non-canonical links the Hessian contains terms in $(y_i - \mu_i)$, observed and expected information differ, and concavity can fail — though it holds for some important cases, such as the probit, whose $\log\Phi$ is concave.</p>
</div>`,
      },
      {
        title: 'Choosing a family and a link',
        html: R`
<p>The two choices answer different questions and should be made separately.</p>
<div class="cols">
<div><h4>Family = variance</h4><p>Ask how the spread scales with the mean. Constant: Normal. Proportional to $\mu$: Poisson (or quasi-Poisson). Proportional to $\mu^2$ (constant CV): gamma, or log-normal-like data. Proportional to $\mu(1-\mu)$: binomial. Plot $\log|y - \hat\mu|$ against $\log\hat\mu$ from a pilot fit: the slope estimates the power in $V(\mu)\propto\mu^k$ divided by two.</p></div>
<div><h4>Link = mean structure</h4><p>Ask on what scale covariates act additively. Multiplicative effects: log. Effects on odds: logit. A latent threshold: probit. A meaningful zero-intercept dose–response for "any event": cloglog. The link must map the support of $\mu$ onto the range of $\eta$, or else the fit needs constraints.</p></div>
</div>
<div class="ex" data-name="Gamma regression for costs versus OLS on log costs">
<p>Suppose hospital costs satisfy $\E(Y\mid\bx) = \exp(\bx^\top\bbeta)$ with constant coefficient of variation. A gamma GLM with log link estimates $\bbeta$ directly, and $e^{\beta_j}$ is the multiplicative change in <em>mean</em> cost. OLS on $\log Y$ estimates $\E(\log Y\mid\bx)$; if the shape of the error distribution changes with $\bx$ (as heteroscedasticity on the log scale implies), then $\E\log Y - \log\E Y$ varies with $\bx$ and the OLS slope does not estimate $\beta_j$. To report mean costs from the log-OLS fit one also needs a smearing correction (Duan, 1983), valid only under homoscedasticity on the log scale.</p>
</div>
<p>Two general facts reduce the stakes. If the mean model is right but the family is wrong, $\hat\bbeta$ is still consistent (the score has mean zero); only the standard errors need repair, via a dispersion estimate or the <a href="#/t/overdispersion">sandwich</a>. If the link is wrong, $\hat\bbeta$ converges to a pseudo-true value — the best approximation in Kullback–Leibler divergence — and for binary data Li and Duan (1989) showed that with elliptically distributed covariates the slope direction is still estimated consistently up to scale.</p>`,
      },
      {
        title: 'Roadmap of the GLM domain',
        html: R`
<table>
<thead><tr><th>Topic</th><th>What it adds</th></tr></thead>
<tbody>
<tr><td><a href="#/t/logistic-regression">Logistic regression</a></td><td>Binomial family, logit link: odds ratios, separation, case-control sampling</td></tr>
<tr><td><a href="#/t/poisson-regression">Poisson regression</a></td><td>Counts, offsets for exposure, log-linear models for tables</td></tr>
<tr><td><a href="#/t/irls">IRLS</a></td><td>Fisher scoring as repeated weighted least squares; convergence</td></tr>
<tr><td><a href="#/t/deviance">Deviance</a></td><td>The GLM analogue of the residual sum of squares; residuals; goodness of fit</td></tr>
<tr><td><a href="#/t/glm-inference">Inference</a></td><td>Wald, score and likelihood-ratio tests; profile intervals</td></tr>
<tr><td><a href="#/t/overdispersion">Overdispersion</a></td><td>Quasi-likelihood, sandwich, negative binomial</td></tr>
<tr><td><a href="#/t/multinomial-ordinal">Multinomial and ordinal</a></td><td>More than two categories; latent-variable models</td></tr>
<tr><td><a href="#/t/glmm-gee">GLMMs and GEE</a></td><td>Clustered data; conditional versus marginal effects</td></tr>
<tr><td><a href="#/t/gam">GAMs</a></td><td>Smooth functions in place of linear terms</td></tr>
</tbody>
</table>
<p>The common thread is that a GLM is a likelihood model built from two modelling choices plus a linear predictor, and each later topic relaxes one of the three: the variance (overdispersion), the independence (GLMM/GEE), the linearity (GAM), or the binary/count response type (multinomial).</p>`,
      },
    ],
    quiz: [
      { q: R`Using only $b(\theta) = \log(1+e^\theta)$ and $a(\phi) = 1/m$, derive the mean and variance of a binomial proportion $Y = S/m$.`,
        a: R`$\mu = b'(\theta) = e^\theta/(1+e^\theta)$, the success probability $\pi$. $b''(\theta) = e^\theta/(1+e^\theta)^2 = \pi(1-\pi)$, so $\Var Y = a(\phi)b''(\theta) = \pi(1-\pi)/m$, matching $\Var(S)/m^2 = m\pi(1-\pi)/m^2$.` },
      { q: R`Why do the score equations of a GLM depend on the assumed distribution only through its mean and variance function, and what practical consequence does this have?`,
        a: R`The chain rule gives $\partial\ell_i/\partial\bbeta = w_i(y_i-\mu_i)\bx_i/(\phi V(\mu_i)g'(\mu_i))$: the only distribution-specific ingredients are $\mu_i$ and $V(\mu_i)$ (because $\partial\mu/\partial\theta = b'' = V$). Hence any two families with the same $V$ give the same $\hat\bbeta$, and the estimating equations remain unbiased whenever the mean is correctly specified — the basis of quasi-likelihood and sandwich inference.` },
      { q: R`For a Poisson GLM with log link and an intercept, show that the sum of fitted values equals the sum of observed counts.`,
        a: R`The log link is canonical for the Poisson, so the score equations are $X^\top(\by - \hat\bmu) = \mathbf 0$. The row of this system corresponding to the intercept column of ones reads $\sum_i (y_i - \hat\mu_i) = 0$.` },
      { q: R`Give a reason to prefer the log link over the canonical (inverse) link for a gamma GLM.`,
        a: R`The canonical link $\theta = -1/\mu$ requires $\bx^\top\bbeta < 0$ (equivalently $1/\mu > 0$) for every observation, a constraint the linear predictor does not respect automatically, so the fit can step outside the parameter space. The log link maps $(0,\infty)$ onto $\R$ and gives multiplicative, easily interpreted effects on the mean.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'logistic-regression',
    domain: 'glm',
    order: 20,
    title: 'Logistic Regression',
    subtitle: 'Linear log-odds, odds ratios, separation and the case-control miracle',
    level: 2,
    sources: { aos: '13.7', cb: '§12.3', other: 'Agresti (2013), Ch. 4–7; Albert & Anderson (1984); Prentice & Pyke (1979); Firth (1993)' },
    prereqs: ['glm-framework'],
    related: ['irls', 'lda-qda', 'classifier-evaluation', 'glm-inference', 'multinomial-ordinal'],
    tags: ['binary response', 'odds ratio', 'separation', 'case-control', 'logit'],
    summary: R`Logistic regression models a binary response by making its log-odds linear: $\log\{\pi(\bx)/(1-\pi(\bx))\} = \bx^\top\bbeta$. It is the binomial GLM with canonical link, so its log-likelihood is concave with score $X^\top(\by - \boldsymbol\pi)$ and information $X^\top W X$, $W = \diag\{\pi_i(1-\pi_i)\}$. Coefficients are log odds ratios, which have a remarkable property: they are unchanged when sampling depends on the outcome, which is why the model dominates case-control epidemiology. Its characteristic failure is <em>separation</em>: when a hyperplane splits the 0s from the 1s, the likelihood increases forever and the MLE does not exist.`,
    survey: {
      task: R`Model $\P(Y=1\mid\bx)$ for binary or binomial outcomes; probabilistic classification`,
      assumptions: R`Independent Bernoulli/binomial responses; log-odds linear in $\bx$; no separation`,
      output: R`$\hat\bbeta$ (log odds ratios), $\hat\pi(\bx)$, standard errors from $(X^\top\hat W X)^{-1}$`,
      strengths: 'Concave likelihood; interpretable odds ratios; valid under case-control sampling; calibrated probabilities',
      weaknesses: 'MLE fails under separation; small-sample bias away from 0; odds ratios non-collapsible and often misread as risk ratios',
      complexity: R`$O(np^2)$ per Newton step`,
    },
    sections: [
      {
        title: 'The model and its latent-variable origin',
        html: R`
<p>Let $Y_i\in\{0,1\}$ with $\P(Y_i = 1\mid\bx_i) = \pi_i$. Logistic regression posits
$$\operatorname{logit}\pi_i = \log\frac{\pi_i}{1-\pi_i} = \bx_i^\top\bbeta,\qquad \pi_i = \frac{1}{1+e^{-\bx_i^\top\bbeta}}.$$
In the <a href="#/t/glm-framework">GLM framework</a> this is the Bernoulli family ($b(\theta) = \log(1+e^\theta)$, $V(\mu) = \mu(1-\mu)$, $\phi = 1$) with its canonical link. Grouped data with $m_i$ trials and $S_i$ successes use the same model with prior weights $m_i$.</p>
<p>A second derivation explains both the logit and its cousins. Suppose a continuous latent propensity $Y_i^* = \bx_i^\top\bbeta + \varepsilon_i$ and we observe only $Y_i = \ind(Y_i^* > 0)$. Then
$$\P(Y_i=1) = \P(\varepsilon_i > -\bx_i^\top\bbeta) = F(\bx_i^\top\bbeta)$$
when $\varepsilon$ has a symmetric CDF $F$. A standard logistic $\varepsilon$ (variance $\pi^2/3$) gives the logit model; a standard normal gives the probit. Since the latent scale is unidentified, the coefficients are only determined relative to the error SD: logit coefficients are roughly $1.6$–$1.8$ times probit coefficients fitted to the same data ($\pi/\sqrt3 \approx 1.81$ matches the error variances, $4\varphi(0)\approx1.6$ matches the slopes at the centre, and $1.7$ gives the best overall approximation $\operatorname{logit}^{-1}(1.7t)\approx\Phi(t)$).</p>
<div class="note">
<p>The latent view has a consequence that is easy to forget: adding an independent covariate to the model <em>changes</em> the other coefficients even when it is uncorrelated with them, because it shrinks the residual variance of $\varepsilon$ and thus rescales everything. This is the <strong>non-collapsibility</strong> of the odds ratio, and it reappears in <a href="#/t/glmm-gee">GLMMs</a> as attenuation of marginal effects.</p>
</div>`,
      },
      {
        title: 'Reading the coefficients: odds ratios and marginal effects',
        html: R`
<p>For a covariate $x_j$, compare two individuals identical except $x_j$ differs by one unit:
$$\frac{\text{odds}(\bx + \mathbf e_j)}{\text{odds}(\bx)} = \frac{e^{(\bx+\mathbf e_j)^\top\bbeta}}{e^{\bx^\top\bbeta}} = e^{\beta_j}.$$
So $e^{\beta_j}$ is a <strong>conditional odds ratio</strong>, the same at every value of $\bx$; this constancy is the model's central structural assumption. For a binary exposure, $e^{\beta_j}$ is the odds ratio in the $2\times2$ table of exposure by outcome within each stratum of the other covariates.</p>
<p>On the probability scale the effect is not constant:
$$\frac{\partial\pi(\bx)}{\partial x_j} = \beta_j\,\pi(\bx)\{1-\pi(\bx)\} \le \frac{\beta_j}{4},$$
with equality at $\pi = 1/2$ (the "divide by four" rule for a quick upper bound on the change in probability per unit of $x_j$).</p>
<div class="warn" data-name="Odds ratios are not risk ratios">
<p>$\text{OR} = \dfrac{\pi_1/(1-\pi_1)}{\pi_0/(1-\pi_0)} = \text{RR}\cdot\dfrac{1-\pi_0}{1-\pi_1}$. The two agree only when the outcome is rare in both groups. With $\pi_0 = 0.4$ and $\pi_1 = 0.6$, the risk ratio is $1.5$ but the odds ratio is $2.25$. When a risk ratio is the target, fit a log-binomial model or a Poisson GLM with robust standard errors (see <a href="#/t/poisson-regression">Poisson regression</a>).</p>
</div>
<div class="ex" data-name="A confidence interval for an odds ratio">
<p>Suppose a fitted smoking coefficient is $\hat\beta = 0.70$ with $\widehat\se = 0.15$. The estimated odds ratio is $e^{0.70}\approx 2.01$. Build the Wald interval on the coefficient scale, where the normal approximation is good, then exponentiate: $\exp(0.70 \pm 1.96\cdot0.15) = (e^{0.406}, e^{0.994}) \approx (1.50, 2.70)$. The interval is asymmetric about $2.01$, as it should be for a ratio.</p>
</div>`,
      },
      {
        title: 'Likelihood, concavity and fitting',
        html: R`
<p>For independent Bernoulli responses,
$$\ell(\bbeta) = \sum_{i=1}^n\Big\{y_i\,\bx_i^\top\bbeta - \log\big(1 + e^{\bx_i^\top\bbeta}\big)\Big\},\qquad U(\bbeta) = X^\top(\by - \boldsymbol\pi),\qquad -\nabla^2\ell = X^\top W X,$$
with $W = \diag\{\pi_i(1-\pi_i)\}$. These follow from $\frac{d}{d\eta}\log(1+e^\eta) = \pi$ and $\frac{d\pi}{d\eta} = \pi(1-\pi)$. Since $W$ has a positive diagonal, $\ell$ is concave, strictly so when $X$ has full rank. Newton's method is
$$\bbeta^{(t+1)} = \bbeta^{(t)} + (X^\top W X)^{-1}X^\top(\by - \boldsymbol\pi) = (X^\top W X)^{-1}X^\top W\mathbf z,\quad z_i = \eta_i + \frac{y_i - \pi_i}{\pi_i(1-\pi_i)},$$
which is the <a href="#/t/irls">IRLS</a> update. Note the absence of a closed form: even the simplest logistic model with one continuous covariate requires iteration.</p>
<p>In the lab, click to add points of either class. The left panel shows the fitted curve; the right panel shows the log-likelihood surface over $(\beta_0,\beta_1)$ with the MLE marked. Move a few points so that every 1 lies to the right of every 0 and watch the contours open into a ridge that runs off to infinity.</p>`,
        lab: 'logistic-regression-sandbox',
      },
      {
        title: 'Separation and the nonexistence of the MLE',
        html: R`
<div class="def" data-name="Separation (Albert & Anderson, 1984)">
<p>The data are <em>completely separated</em> if there is $\mathbf d\neq\mathbf 0$ with $\bx_i^\top\mathbf d > 0$ whenever $y_i = 1$ and $\bx_i^\top\mathbf d < 0$ whenever $y_i = 0$. They are <em>quasi-completely separated</em> if there is $\mathbf d\neq \mathbf 0$ with $\bx_i^\top\mathbf d\ge 0$ for all $y_i=1$, $\bx_i^\top\mathbf d\le0$ for all $y_i=0$, with equality for at least one case (and no complete separation). Otherwise the data <em>overlap</em>.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Existence of the logistic MLE (Albert & Anderson, 1984)">
<p>With $X$ of full column rank, a finite maximizer of $\ell$ exists (and is then unique) if and only if the data overlap.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $s_i = 2y_i - 1\in\{-1,+1\}$; then each term of $\ell$ is $-\log(1 + e^{-s_i\bx_i^\top\bbeta}) < 0$, so $\ell < 0$ everywhere.</p>
<p><em>Separation ⇒ no maximizer.</em> Suppose $\mathbf d$ separates (completely or quasi-completely), so $s_i\bx_i^\top\mathbf d \ge 0$ for all $i$. For any $\bbeta$, the function $c\mapsto \ell(\bbeta + c\mathbf d)$ is nondecreasing, since each term $-\log(1+e^{-s_i\bx_i^\top\bbeta - c\,s_i\bx_i^\top\mathbf d})$ is nondecreasing in $c$, and it is strictly increasing because at least one $s_i\bx_i^\top\mathbf d > 0$ (if all were 0 then $X\mathbf d = 0$, contradicting full rank). So no $\bbeta$ can be a maximizer. Under complete separation moreover $\ell(c\mathbf d)\to 0 = \sup\ell$.</p>
<p><em>Overlap ⇒ maximizer exists.</em> Overlap means every direction $\mathbf d\neq\mathbf 0$ has some $i$ with $s_i\bx_i^\top\mathbf d < 0$. For that $i$, the term $-\log(1+e^{-s_i\bx_i^\top(\bbeta + c\mathbf d)})\to-\infty$ linearly as $c\to\infty$, while all other terms are bounded above by 0. So $\ell\to-\infty$ along every ray. The function $\mathbf d\mapsto \min_i s_i\bx_i^\top\mathbf d$ is continuous and negative on the unit sphere, hence bounded above by some $-\delta < 0$, which makes the decay uniform over directions: $\ell$ is coercive, its superlevel sets are compact, and a continuous function attains its maximum on a compact set. Uniqueness follows from strict concavity.</p>
</details>
<p>What does a fitting routine do with separated data? Newton iterates march along $\mathbf d$, the fitted probabilities approach 0 and 1, $W\to0$, and $(X^\top WX)^{-1}$ explodes. Software stops at an arbitrary iteration and reports a large coefficient with an enormous standard error (and often a warning about fitted probabilities numerically 0 or 1). The reported Wald $z$ is small — a symptom of the <a href="#/t/glm-inference">Hauck–Donner effect</a> — even though the evidence for an effect may be overwhelming.</p>
<div class="warn" data-name="Separation is common, not exotic">
<p>It happens with small $n$, with rare outcomes, with many covariates (the probability of separation jumps from near 0 to near 1 as $p/n$ crosses a threshold; for Gaussian covariates and no signal the threshold is $p/n = 1/2$, Cover 1965), and with a categorical covariate having a level where all outcomes are equal.</p>
</div>
<p>Remedies, roughly in order of preference: <strong>Firth's penalized likelihood</strong> (1993), which maximizes $\ell(\bbeta) + \tfrac12\log\det\mathcal I(\bbeta)$ (the Jeffreys prior), always gives finite estimates and removes the $O(n^{-1})$ bias; <strong>weakly informative priors</strong> such as independent Cauchy(0, 2.5) on standardized coefficients (Gelman et al., 2008); a <strong>ridge or lasso penalty</strong>; or <strong>exact conditional logistic regression</strong> for small samples, which conditions on the sufficient statistics of nuisance parameters. Profile-likelihood intervals remain informative under separation (one endpoint is infinite) where Wald intervals are meaningless.</p>`,
      },
      {
        title: 'Case-control sampling: why slopes survive',
        html: R`
<p>In a case-control study one samples cases ($Y=1$) and controls ($Y=0$) separately, often in equal numbers, from a population where cases are rare. The covariate distribution is then sampled <em>conditionally on the outcome</em> — the reverse of the regression direction. The logistic model is uniquely suited to this design.</p>
<div class="thm" data-kind="Proposition" data-name="Invariance of logistic slopes under outcome-dependent sampling">
<p>Suppose $\operatorname{logit}\P(Y=1\mid\bx) = \beta_0 + \bx^\top\boldsymbol\gamma$ in the population, and an individual with outcome $y$ enters the sample with probability $s_y$ independently of $\bx$ (let $Z = 1$ denote being sampled). Then
$$\operatorname{logit}\P(Y=1\mid\bx, Z=1) = \beta_0 + \log\frac{s_1}{s_0} + \bx^\top\boldsymbol\gamma.$$
Only the intercept changes.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By Bayes' rule, using $\P(Z=1\mid Y=y,\bx) = s_y$,
$$\frac{\P(Y=1\mid\bx,Z=1)}{\P(Y=0\mid\bx,Z=1)} = \frac{\P(Z=1\mid Y=1,\bx)\P(Y=1\mid\bx)}{\P(Z=1\mid Y=0,\bx)\P(Y=0\mid\bx)} = \frac{s_1}{s_0}\,e^{\beta_0 + \bx^\top\boldsymbol\gamma}.$$
Take logs.</p>
</details>
<p>The proposition shows the model is correct in the sample, but the sample likelihood is really the "retrospective" likelihood $\prod_i p(\bx_i\mid y_i)$, not $\prod_i p(y_i\mid\bx_i)$. Prentice and Pyke (1979) proved the stronger result that maximizing the prospective logistic likelihood on case-control data gives the semiparametric MLE of $\boldsymbol\gamma$ with correct asymptotic standard errors, leaving the covariate distribution completely unspecified. The intercept is useless unless the sampling fractions are known, in which case one can subtract $\log(s_1/s_0)$ to recover absolute risks.</p>
<div class="warn" data-name="This is special to the logit">
<p>The argument used only that the logit of the conditional probability shifts by a constant under outcome-dependent sampling, which is true precisely because the odds factor multiplicatively. Probit or identity-link slopes are distorted by case-control sampling, and risk ratios or risk differences cannot be estimated from case-control data without external information on the baseline risk.</p>
</div>`,
        lab: 'logistic-regression-casecontrol',
      },
      {
        title: 'Connections and pitfalls',
        html: R`
<div class="cols">
<div><h4>Discriminative vs generative</h4><p>If $\bX\mid Y=k\sim\N(\bmu_k,\bSigma)$, the posterior log-odds are exactly linear in $\bx$ (<a href="#/t/lda-qda">LDA</a>). Logistic regression estimates the same conditional model without assuming Gaussian covariates. Efron (1975) showed LDA is more efficient when its assumptions hold (asymptotic relative efficiency of logistic about 1/2 to 2/3 for well-separated classes); logistic is more robust when they do not.</p></div>
<div><h4>Small-sample bias</h4><p>The MLE is biased away from zero, with bias of order $n^{-1}$ that can be substantial when events are few. The old rule of "10 events per variable" is a crude guard; Firth's correction removes the leading bias term.</p></div>
<div><h4>Probabilities, not decisions</h4><p>The model outputs $\hat\pi(\bx)$; a classification threshold is a separate, loss-dependent decision. Assess discrimination (AUC) and calibration separately (<a href="#/t/classifier-evaluation">classifier evaluation</a>).</p></div>
</div>
<p>Further pitfalls: interpreting the coefficient of a covariate that participates in an interaction as a "main effect"; comparing logit coefficients across models with different covariates as if they were on a common scale (they are not, because of non-collapsibility); and assuming the logit link is correct when the data cannot distinguish it from alternatives with different tail behaviour, which matters for extrapolation to extreme probabilities.</p>`,
      },
    ],
    quiz: [
      { q: R`A logistic model gives $\hat\beta_1 = 1.2$ for a covariate. Give two correct interpretations: one on the odds scale and one bound on the probability scale.`,
        a: R`A one-unit increase multiplies the odds by $e^{1.2}\approx 3.32$, holding the other covariates fixed. On the probability scale the change per unit is $\beta_1\pi(1-\pi)$, at most $1.2/4 = 0.3$, achieved where $\pi = 1/2$; far in the tails it is much smaller.` },
      { q: R`Your logistic fit reports $\hat\beta = 23.4$, $\widehat\se = 2900$, $z = 0.008$. What has likely happened, and is the covariate unimportant?`,
        a: R`The data are (quasi-)separated by this covariate: the likelihood keeps increasing along a direction, so the iterations stopped at an arbitrary large value, and the information is nearly singular. The tiny Wald $z$ is an artefact (Hauck–Donner); the covariate is likely extremely predictive. Use a likelihood-ratio test or profile interval, Firth's correction, or a weakly informative prior.` },
      { q: R`In a case-control study cases are sampled with probability $s_1 = 1$ and controls with probability $s_0 = 0.01$. How does the fitted intercept relate to the population intercept?`,
        a: R`It is shifted by $\log(s_1/s_0) = \log 100\approx 4.61$: $\hat\beta_0^{\text{sample}} \approx \beta_0 + 4.61$. The slopes estimate the population log odds ratios unchanged.` },
      { q: R`Why does the Hessian of the logistic log-likelihood not depend on $\by$, and why does that matter?`,
        a: R`With the canonical link, $\ell = \sum_i\{y_i\eta_i - b(\eta_i)\}$ is linear in $\by$ given $\bbeta$, so the second derivative $-\sum_i b''(\eta_i)\bx_i\bx_i^\top$ involves only $\bbeta$. Hence observed and expected information coincide, Newton equals Fisher scoring, and concavity holds for every data set.` },
      { q: R`Adding a covariate that is independent of $x_1$ to a logistic model changes $\hat\beta_1$. Why does this not happen in linear regression?`,
        a: R`In linear regression, omitting an independent covariate only adds to the error variance and leaves $\E(Y\mid x_1)$ linear with the same slope. In the logistic model, averaging $\operatorname{logit}^{-1}(\beta_1x_1 + \beta_2x_2)$ over $x_2$ does not give a logistic curve with slope $\beta_1$; it gives a flatter curve (attenuation). The odds ratio is non-collapsible, so conditional and marginal odds ratios differ even without confounding.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'poisson-regression',
    domain: 'glm',
    order: 30,
    title: 'Poisson Regression & Log-Linear Models',
    subtitle: 'Counts, rates, exposure offsets, and contingency tables as regressions',
    level: 2,
    sources: { aos: '13.7', cb: '§12.3', other: 'Agresti (2013), Ch. 9; Bishop, Fienberg & Holland (1975); Gourieroux, Monfort & Trognon (1984); Cameron & Trivedi (2013)' },
    prereqs: ['glm-framework'],
    related: ['overdispersion', 'poisson-process', 'deviance', 'logistic-regression'],
    tags: ['counts', 'rate ratio', 'offset', 'log-linear model', 'contingency table'],
    summary: R`Poisson regression models counts with $\log\mu_i = \bx_i^\top\bbeta$, so covariates act multiplicatively and $e^{\beta_j}$ is a rate ratio. When counts accumulate over different exposures $t_i$ (person-years, area, time at risk), the model for the <em>rate</em> becomes $\log\mu_i = \log t_i + \bx_i^\top\bbeta$, where $\log t_i$ is an <em>offset</em> — a covariate with coefficient fixed at 1. The same machinery fits log-linear models to contingency tables, where conditioning on the total turns the Poisson likelihood into the multinomial one. And because the log link is canonical, the Poisson score equations are unbiased whenever the mean is right, which makes Poisson pseudo-likelihood a robust tool for any nonnegative response with a log-linear mean.`,
    survey: {
      task: 'Regression for counts and rates; log-linear analysis of contingency tables',
      assumptions: R`Independent counts; $\log\E(Y\mid\bx) = \text{offset} + \bx^\top\bbeta$; for model-based SEs, $\Var(Y) = \E(Y)$`,
      output: R`Rate ratios $e^{\hat\beta_j}$; fitted counts; deviance and Pearson fit statistics`,
      strengths: 'Canonical link: concave likelihood, margin-matching fits; consistent under mean-only assumptions; offsets handle exposure',
      weaknesses: 'Equidispersion rarely holds (SEs too small); excess zeros; mean structure must be log-linear',
      complexity: R`$O(np^2)$ per IRLS iteration`,
    },
    sections: [
      {
        title: 'The model',
        html: R`
<p>For counts $Y_i\in\{0,1,2,\dots\}$ assume $Y_i\sim\text{Poisson}(\mu_i)$ independently with
$$\log\mu_i = \bx_i^\top\bbeta.$$
The log link is canonical ($\theta = \log\mu$, $b(\theta) = e^\theta$, $V(\mu) = \mu$, $\phi = 1$), so the log-likelihood, score and information are
$$\ell(\bbeta) = \sum_i\{y_i\bx_i^\top\bbeta - e^{\bx_i^\top\bbeta}\} - \sum_i\log y_i!,\qquad U = X^\top(\by - \bmu),\qquad \mathcal I = X^\top\diag(\mu_i)X.$$
Concavity, uniqueness and the margin-matching property all follow from the <a href="#/t/glm-framework">canonical-link theorem</a>.</p>
<p><strong>Interpretation.</strong> A unit increase in $x_j$ multiplies the expected count by $e^{\beta_j}$, the <em>rate ratio</em> (or incidence rate ratio). Unlike the odds ratio, the rate ratio is collapsible over covariates that are independent of $x_j$: averaging $e^{\beta_0 + \beta_1x_1 + \beta_2x_2}$ over $x_2$ gives $e^{\beta_1x_1}\cdot\text{const}$.</p>
<div class="note">
<p>Why the Poisson? If events occur as a <a href="#/t/poisson-process">Poisson process</a> with intensity $\lambda(\bx)$, the number in a window of length $t$ is Poisson with mean $t\lambda(\bx)$. Small independent chances of an event among many opportunities also give a Poisson limit (law of rare events). Both stories predict $\Var Y = \E Y$, which real data typically violate — see <a href="#/t/overdispersion">overdispersion</a>.</p>
</div>
<div class="warn" data-name="Existence of the MLE">
<p>As with logistic regression the Poisson MLE can fail to exist. The simplest case: if every count in some level of a factor is zero, that level's coefficient is driven to $-\infty$. In general the MLE exists if and only if there is no direction $\mathbf d$ with $X\mathbf d \le \mathbf 0$, $X\mathbf d
eq\mathbf 0$, and $(X\mathbf d)_i = 0$ whenever $y_i > 0$ — such a direction increases $\ell$ forever by pushing the zero-count means to 0 (Haberman, 1974).</p>
</div>`,
      },
      {
        title: 'Rates, exposure and offsets',
        html: R`
<p>Counts are often recorded over unequal exposures: deaths over different numbers of person-years, accidents over different traffic volumes, species over plots of different area. The natural model is for the <em>rate</em> $\lambda_i$ per unit exposure:
$$Y_i\sim\text{Poisson}(t_i\lambda_i),\qquad \log\lambda_i = \bx_i^\top\bbeta\quad\Longleftrightarrow\quad \log\mu_i = \log t_i + \bx_i^\top\bbeta.$$</p>
<div class="def" data-name="Offset">
<p>A known term added to the linear predictor with coefficient fixed at 1: $\eta_i = o_i + \bx_i^\top\bbeta$. Here $o_i = \log t_i$.</p>
</div>
<p>The score equations become $X^\top(\by - \bmu) = \mathbf 0$ with $\mu_i = t_i e^{\bx_i^\top\bbeta}$. With an intercept only, $\hat\lambda = \sum y_i/\sum t_i$ — total events over total exposure, the classical rate estimate. With a binary exposure group, the MLE of the rate ratio is the ratio of the two crude rates.</p>
<p>Three ways to handle exposure, only one of them right:</p>
<ul>
<li><strong>Ignore it.</strong> If $t_i$ is correlated with a covariate, that covariate absorbs the exposure effect and its coefficient is biased. Even if uncorrelated, the counts are more variable than the model allows.</li>
<li><strong>Model $Y_i/t_i$ as the response with OLS.</strong> Wrong variance (it is $\lambda_i/t_i$, not constant) and the fitted rates can be negative.</li>
<li><strong>Offset $\log t_i$.</strong> Exactly the rate model above. If you doubt proportionality, include $\log t_i$ as an ordinary covariate and test whether its coefficient is 1.</li>
</ul>
<p>In the lab, regions have exposures correlated with the covariate. Toggle the offset off and the slope estimate drifts toward the exposure–covariate association; toggle it on and it recovers the true rate ratio.</p>`,
        lab: 'poisson-regression-offset',
      },
      {
        title: 'Log-linear models for contingency tables',
        html: R`
<p>An $I\times J$ table of counts $n_{ij}$ can be analysed as a Poisson regression with one observation per cell and factor covariates for rows and columns. The <em>independence model</em> is
$$\log\mu_{ij} = \lambda + \alpha_i + \beta_j,$$
and the <em>saturated model</em> adds interactions $(\alpha\beta)_{ij}$, one per cell after identifiability constraints. Under the independence model the margin-matching score equations force $\hat\mu_{i+} = n_{i+}$ and $\hat\mu_{+j} = n_{+j}$, and the unique table of the form $\mu_{ij} = r_i c_j$ with those margins is
$$\hat\mu_{ij} = \frac{n_{i+}\,n_{+j}}{n}.$$
These are the familiar expected counts of the chi-squared test of independence, and the Pearson statistic for this Poisson fit is exactly Pearson's $X^2$. Interaction terms are log odds ratios: in a $2\times2$ table, $\log\frac{\mu_{11}\mu_{22}}{\mu_{12}\mu_{21}} = (\alpha\beta)_{11} + (\alpha\beta)_{22} - (\alpha\beta)_{12} - (\alpha\beta)_{21}$.</p>
<div class="thm" data-kind="Theorem" data-name="Poisson–multinomial equivalence (Birch, 1963)">
<p>Let $Y_1,\dots,Y_K$ be independent Poisson with $\log\mu_k = \alpha + \bx_k^\top\bbeta$ (the model includes a free intercept). Then the MLE of $\bbeta$ from the Poisson likelihood equals the MLE from the multinomial likelihood of $(Y_1,\dots,Y_K)$ given $N = \sum_k Y_k$, with cell probabilities $\pi_k = e^{\bx_k^\top\bbeta}/\sum_l e^{\bx_l^\top\bbeta}$, and the two give the same observed information for $\bbeta$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Reparameterize by $\tau = \sum_k\mu_k = e^\alpha\sum_l e^{\bx_l^\top\bbeta}$ and $\pi_k = \mu_k/\tau$, which depends only on $\bbeta$. The joint Poisson probability factorizes as
$$\prod_k\frac{e^{-\mu_k}\mu_k^{y_k}}{y_k!} = \underbrace{\frac{e^{-\tau}\tau^{N}}{N!}}_{N\sim\text{Poisson}(\tau)}\times\underbrace{\frac{N!}{\prod_k y_k!}\prod_k\pi_k^{y_k}}_{\text{multinomial}(N,\boldsymbol\pi(\bbeta))}.$$
Because the intercept is free, $(\tau,\bbeta)$ is a one-to-one reparameterization of $(\alpha,\bbeta)$ with $\tau$ variation-independent of $\bbeta$. The log-likelihood separates into a function of $\tau$ plus a function of $\bbeta$, so the maximization over $\bbeta$ is the multinomial maximization, and the information is block-diagonal in $(\tau,\bbeta)$.</p>
</details>
<p>The same argument with one free parameter per stratum shows that Poisson log-linear models also reproduce product-multinomial (stratified) sampling, as long as the model includes the terms that fix the sampled margins. This is why one can fit <a href="#/t/multinomial-ordinal">multinomial logit</a> models as Poisson regressions — the "Poisson trick" — and why log-linear models are agnostic to whether the table total was fixed by design.</p>`,
      },
      {
        title: 'Robustness: Poisson pseudo-maximum likelihood',
        html: R`
<p>Because the score $X^\top(\by - \bmu)$ has mean zero whenever $\E(Y_i\mid\bx_i) = \exp(\bx_i^\top\bbeta)$, the Poisson estimator is consistent for $\bbeta$ under that mean assumption alone — the response need not be Poisson, need not be a count, and its variance can be anything.</p>
<div class="thm" data-kind="Theorem" data-name="Pseudo-MLE consistency (Gourieroux, Monfort & Trognon, 1984)">
<p>Suppose $(\bx_i, Y_i)$ are IID with $Y_i\ge 0$, $\E(Y_i\mid\bx_i) = \exp(\bx_i^\top\bbeta_0)$, and standard moment and identifiability conditions hold. Then the Poisson pseudo-MLE $\hat\bbeta$ is consistent and
$$\sqrt n(\hat\bbeta - \bbeta_0)\dto\N\big(\mathbf 0,\; A^{-1}BA^{-1}\big),\quad A = \E[\mu_i\bx_i\bx_i^\top],\; B = \E[\Var(Y_i\mid\bx_i)\bx_i\bx_i^\top].$$
More generally, a pseudo-likelihood estimator is consistent for <em>every</em> distribution with the correct mean if and only if the assumed family is a linear exponential family in $\mu$ (Poisson, Normal, gamma, binomial and NB with known $k$ are all examples).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$\hat\bbeta$ solves $\frac1n\sum_i(Y_i - e^{\bx_i^\top\bbeta})\bx_i = \mathbf 0$. At $\bbeta_0$ its expectation is $\E[\bx_i\,\E(Y_i - \mu_i\mid\bx_i)] = \mathbf 0$, and the population objective $\E[Y\bx^\top\bbeta - e^{\bx^\top\bbeta}]$ is strictly concave, so $\bbeta_0$ is its unique maximizer; standard M-estimation arguments give consistency. A Taylor expansion of the estimating equation gives $\sqrt n(\hat\bbeta - \bbeta_0)\approx A^{-1}n^{-1/2}\sum_i(Y_i - \mu_i)\bx_i$, and the CLT supplies the normal limit with "meat" $B$. When $\Var(Y\mid\bx) = \mu$, $B = A$ and the sandwich reduces to $A^{-1}$.</p>
</details>
<p>Two widely used applications:</p>
<ul>
<li><strong>Log-linear models for continuous positive outcomes</strong> (trade flows in the gravity model, costs, wages). Santos Silva and Tenreyro (2006) showed that OLS on $\log y$ is inconsistent for the elasticities under heteroscedasticity, while Poisson PML is consistent. The second lab reproduces this.</li>
<li><strong>Risk ratios for binary outcomes</strong> ("modified Poisson", Zou 2004): fit a Poisson GLM with log link to 0/1 data. The point estimates are consistent for log risk ratios; the model-based SEs are too large (binary variance $\mu(1-\mu) < \mu$), so use the sandwich.</li>
</ul>
<p>In all of these, the sandwich covariance from the <a href="#/t/overdispersion">overdispersion</a> topic is mandatory.</p>`,
        lab: 'poisson-regression-ppml',
      },
      {
        title: 'Worked example and connections to survival',
        html: R`
<div class="ex" data-name="Comparing two incidence rates">
<p>A clinic observes 30 infections in 1,200 catheter-days on protocol A and 12 in 1,000 catheter-days on protocol B. Fit $\log\mu = \log t + \beta_0 + \beta_1\ind(\text{A})$. The MLEs are the crude log rates: $\hat\beta_0 = \log(12/1000) = -4.423$, and $\hat\beta_1 = \log\frac{30/1200}{12/1000} = \log(2.083) = 0.734$. The inverse information for a two-group Poisson model gives $\widehat{\Var}(\hat\beta_1) = 1/30 + 1/12 = 0.1167$, so $\widehat\se = 0.342$ and a Wald 95% interval for the rate ratio is $\exp(0.734\pm1.96\cdot0.342) = (1.07, 4.07)$. The form $1/y_A + 1/y_B$ comes from $\mathcal I = X^\top\diag(\hat\mu)X$ with $\hat\mu$ equal to the observed counts.</p>
</div>
<p><strong>The piecewise-exponential (Poisson) trick.</strong> Split follow-up time into intervals; let $d_{ik}\in\{0,1\}$ indicate an event for subject $i$ in interval $k$ and $t_{ik}$ the time at risk there. A proportional hazards model with piecewise-constant baseline hazard $\lambda_k$ has likelihood proportional to $\prod_{i,k}(\lambda_k e^{\bx_i^\top\bbeta})^{d_{ik}}\exp(-t_{ik}\lambda_k e^{\bx_i^\top\bbeta})$, which is (up to a constant) the Poisson likelihood for "counts" $d_{ik}$ with $\log\mu_{ik} = \log t_{ik} + \log\lambda_k + \bx_i^\top\bbeta$. A survival model is thus fitted by Poisson regression with interval factors and a log-exposure offset; letting the intervals shrink recovers the Cox partial likelihood estimates (Holford, 1980; Laird & Olivier, 1981).</p>
<div class="warn" data-name="Pitfalls">
<ul>
<li><strong>Overdispersion</strong> is the rule: compare the Pearson statistic with its degrees of freedom before trusting any standard error.</li>
<li><strong>Excess zeros</strong> (structural non-users, never-at-risk units) need zero-inflated or hurdle models; a Poisson GLM will underpredict zeros and overpredict small positive counts.</li>
<li><strong>Exposure on the wrong scale</strong>: the offset must be $\log t$, not $t$.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why is including $\log t_i$ as an offset equivalent to modelling the rate $\mu_i/t_i$ log-linearly, and how would you test whether exposure really acts proportionally?`,
        a: R`$\log\mu_i = \log t_i + \bx_i^\top\bbeta$ rearranges to $\log(\mu_i/t_i) = \bx_i^\top\bbeta$. To test proportionality, include $\log t_i$ as an ordinary covariate with free coefficient $\gamma$ and test $H_0:\gamma = 1$ (Wald or likelihood ratio against the offset model).` },
      { q: R`Show that for the independence log-linear model in a two-way table the fitted counts are $n_{i+}n_{+j}/n$.`,
        a: R`The model is canonical with row and column factors, so the score equations force fitted row totals and column totals to equal the observed ones. The model says $\mu_{ij} = r_ic_j$; summing gives $\mu_{i+} = r_ic_+ = n_{i+}$ and $\mu_{+j} = r_+c_j = n_{+j}$, and $\mu_{++} = r_+c_+ = n$. So $\mu_{ij} = r_ic_j = (n_{i+}/c_+)(n_{+j}/r_+) = n_{i+}n_{+j}/n$.` },
      { q: R`A colleague fits a Poisson GLM to 0/1 outcomes to estimate risk ratios. Is this legitimate?`,
        a: R`Yes for the point estimates: the Poisson score equations are unbiased whenever $\E(Y\mid\bx) = \exp(\bx^\top\bbeta)$, which is exactly the log-risk model. But the Poisson variance $\mu$ exceeds the Bernoulli variance $\mu(1-\mu)$, so model-based SEs are conservative; use the sandwich estimator. Fitted risks can also exceed 1.` },
      { q: R`Why does the Poisson–multinomial equivalence require the log-linear model to contain an intercept?`,
        a: R`The intercept lets the total $\tau = \sum\mu_k$ vary freely and independently of $\bbeta$. Without it, $\tau$ would be a function of $\bbeta$, the factorization into Poisson($\tau$) × multinomial($\boldsymbol\pi(\bbeta)$) would no longer separate the parameters, and the observed total $N$ would carry information about $\bbeta$ under the Poisson model but not under the multinomial one.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'irls',
    domain: 'glm',
    order: 40,
    title: 'Fitting GLMs: Fisher Scoring & IRLS',
    subtitle: 'Newton’s method in disguise as a sequence of weighted least-squares fits',
    level: 2,
    sources: { aos: '13.7', other: 'Nelder & Wedderburn (1972); McCullagh & Nelder (1989), §2.5; Green (1984); Marschner (2011)' },
    prereqs: ['glm-framework', 'fisher-information'],
    related: ['logistic-regression', 'weighted-gls', 'mle', 'gam'],
    tags: ['Newton–Raphson', 'Fisher scoring', 'weighted least squares', 'working response', 'optimization'],
    summary: R`GLM likelihoods have no closed-form maximizer, but they have a special structure: the score is $X^\top WG(\by-\bmu)/\phi$ and the expected information is $X^\top WX/\phi$. Plugging these into Fisher scoring, $\bbeta\leftarrow\bbeta + \mathcal I^{-1}U$, and rearranging gives $\bbeta\leftarrow(X^\top WX)^{-1}X^\top W\mathbf z$: a weighted least-squares regression of a <em>working response</em> $z_i = \eta_i + g'(\mu_i)(y_i-\mu_i)$ on $X$ with weights that are the inverse working variances. Iterating until the weights stop changing is IRLS. With the canonical link it is exactly Newton–Raphson, so convergence near the optimum is quadratic; the same template fits penalized models, GAMs and robust regressions.`,
    survey: {
      task: 'Compute the MLE (or quasi-likelihood estimate) of a GLM',
      assumptions: R`Full-rank $X$; MLE exists (no separation); reasonable starting values`,
      output: R`$\hat\bbeta$, final weights $\hat W$ (giving $\widehat\Cov(\hat\bbeta)$ and leverages), deviance`,
      strengths: 'Reuses least-squares software; needs no step size; quadratic convergence for canonical links',
      weaknesses: 'Not globally convergent without step-halving; diverges under separation; $O(np^2)$ memory-bound for huge $n$',
      complexity: R`$O(np^2 + p^3)$ per iteration, typically 4–10 iterations`,
    },
    sections: [
      {
        title: 'Newton–Raphson and Fisher scoring',
        html: R`
<p>To solve the score equation $U(\bbeta) = \mathbf 0$, Newton–Raphson linearizes $U$ around the current iterate:
$$U(\bbeta)\approx U(\bbeta^{(t)}) - J(\bbeta^{(t)})(\bbeta - \bbeta^{(t)}),\qquad J = -\frac{\partial^2\ell}{\partial\bbeta\partial\bbeta^\top}\ \text{(observed information)},$$
and sets the approximation to zero: $\bbeta^{(t+1)} = \bbeta^{(t)} + J^{-1}U$. Fisher (1925) proposed replacing $J$ by its expectation $\mathcal I = \E J$, giving <strong>Fisher scoring</strong>:
$$\bbeta^{(t+1)} = \bbeta^{(t)} + \mathcal I(\bbeta^{(t)})^{-1}\,U(\bbeta^{(t)}).$$
Scoring has two practical advantages: $\mathcal I$ is always positive semidefinite (it is a covariance matrix), so each step is an ascent direction for small enough steps, and $\mathcal I$ is often simpler than $J$.</p>
<p>For a GLM the observed information is
$$J(\bbeta) = \mathcal I(\bbeta) - \frac1\phi\sum_i (y_i - \mu_i)\,\frac{\partial}{\partial\eta_i}\!\left[\frac{w_i}{V(\mu_i)g'(\mu_i)}\right]\bx_i\bx_i^\top,$$
which differs from $\mathcal I$ by a term with mean zero. Under the canonical link $w_i/(V g')= w_i$ is constant in $\eta$, the correction vanishes, and <strong>Newton = Fisher scoring</strong>.</p>`,
      },
      {
        title: 'From Fisher scoring to iteratively reweighted least squares',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Fisher scoring is IRLS (Nelder & Wedderburn, 1972)">
<p>For a GLM, the Fisher scoring update can be written
$$\bbeta^{(t+1)} = \big(X^\top W X\big)^{-1}X^\top W\mathbf z,\qquad z_i = \eta_i + g'(\mu_i)(y_i - \mu_i),\qquad W_i = \frac{w_i}{V(\mu_i)g'(\mu_i)^2},$$
with $\eta_i,\mu_i,W$ evaluated at $\bbeta^{(t)}$. The dispersion $\phi$ cancels, so the algorithm does not need it.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>From the <a href="#/t/glm-framework">score and information formulas</a>, $U = \phi^{-1}X^\top WG(\by-\bmu)$ and $\mathcal I = \phi^{-1}X^\top WX$, where $G = \diag\{g'(\mu_i)\}$. Then
$$\mathcal I\bbeta^{(t+1)} = \mathcal I\bbeta^{(t)} + U = \phi^{-1}X^\top W\big(X\bbeta^{(t)} + G(\by - \bmu)\big) = \phi^{-1}X^\top W\mathbf z,$$
since $X\bbeta^{(t)} = \boldsymbol\eta$. Multiplying by $\phi\,(X^\top WX)^{-1}$ gives the claim.</p>
</details>
<p>The working response $z_i$ is the first-order Taylor expansion of $g(y_i)$ about $\mu_i$: $g(y_i)\approx g(\mu_i) + g'(\mu_i)(y_i - \mu_i)$. It is a "linearized data point" on the scale where the model is linear, and it avoids evaluating $g(y_i)$ itself (which may be infinite: $\log 0$, $\operatorname{logit} 1$). Its conditional variance given the current fit is $g'(\mu_i)^2\phi V(\mu_i)/w_i = \phi/W_i$, so the weights are exactly the inverse variances. IRLS is therefore the natural procedure: <em>linearize, fit the linearized model by the appropriate weighted least squares, update the weights, repeat</em>.</p>
<div class="algo" data-name="IRLS">
<ol>
<li><strong>Initialize on the mean scale</strong>, not with $\bbeta$: set $\mu_i^{(0)} = y_i$ adjusted away from the boundary (e.g. $(m_iy_i + 0.5)/(m_i+1)$ for binomial proportions, $y_i + 0.1$ for Poisson), and $\eta_i^{(0)} = g(\mu_i^{(0)})$.</li>
<li>Compute $z_i = \eta_i + g'(\mu_i)(y_i-\mu_i)$ and $W_i = w_i/\{V(\mu_i)g'(\mu_i)^2\}$.</li>
<li>Solve the weighted least-squares problem $\bbeta^{\text{new}} = \argmin_{\bbeta}\sum_i W_i(z_i - \bx_i^\top\bbeta)^2$ (by QR of $W^{1/2}X$, not by forming $X^\top WX$).</li>
<li>Update $\boldsymbol\eta = X\bbeta^{\text{new}}$ (plus any offset), $\mu_i = g^{-1}(\eta_i)$, and the deviance $D$.</li>
<li>Stop when $|D^{\text{new}} - D^{\text{old}}|/(|D^{\text{new}}| + 0.1) < \epsilon$ (R's default rule with $\epsilon = 10^{-8}$); otherwise return to step 2.</li>
</ol>
</div>
<div class="ex" data-name="IRLS for a Poisson mean, and why the starting value matters">
<p>Take an intercept-only Poisson model with $\bar y = 4$. Then $W_i = \mu$, $z_i = \beta + (y_i - \mu)/\mu$, and the WLS update is just the average working response: $\beta^{(t+1)} = \beta^{(t)} + (\bar y - e^{\beta^{(t)}})/e^{\beta^{(t)}}$ — Newton's method for $e^\beta = \bar y$. Starting at $\beta = 0$: $3 \to 2.199 \to 1.643 \to 1.4165 \to 1.38672 \to 1.386293$, with $\log 4 = 1.386294$. The first step overshoots badly (because $e^\beta$ is far from linear over $[0,3]$), the errors then shrink like $0.23, 0.03, 4\times10^{-4}, 10^{-7}$: each roughly the square of the previous, as Newton theory predicts. Starting from $\mu^{(0)} = \bar y$ would have converged in one step.</p>
</div>`,
        lab: 'irls-steps',
      },
      {
        title: 'Convergence: local rates and global safeguards',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Local convergence rates">
<p>Let $\hat\bbeta$ be a nondegenerate maximizer ($J(\hat\bbeta)$ positive definite) and let $\ell$ be three times continuously differentiable near $\hat\bbeta$.</p>
<ol>
<li>Newton–Raphson converges quadratically from any start close enough to $\hat\bbeta$: $\|\bbeta^{(t+1)} - \hat\bbeta\|\le C\|\bbeta^{(t)} - \hat\bbeta\|^2$.</li>
<li>Fisher scoring converges at least linearly, with asymptotic rate equal to the spectral radius of $\mathcal I(\hat\bbeta)^{-1}\{\mathcal I(\hat\bbeta) - J(\hat\bbeta)\}$.</li>
</ol>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Scoring is the fixed-point iteration $\bbeta\mapsto T(\bbeta) = \bbeta + \mathcal I(\bbeta)^{-1}U(\bbeta)$. Its Jacobian at $\hat\bbeta$ is $I_p + \mathcal I^{-1}\,\partial U/\partial\bbeta + (\partial\mathcal I^{-1})U(\hat\bbeta) = I_p - \mathcal I^{-1}J$, using $U(\hat\bbeta) = 0$ and $\partial U/\partial\bbeta = -J$. A smooth fixed-point map converges locally at the rate given by the spectral radius of its Jacobian when that is below 1. For Newton, $\mathcal I$ is replaced by $J$, the Jacobian vanishes, and a second-order Taylor expansion gives the quadratic bound.</p>
</details>
<p>For the canonical link the two coincide, so IRLS converges quadratically. For non-canonical links, $\mathcal I - J$ at $\hat\bbeta$ is a sum of mean-zero terms of order $\sqrt n$ while $\mathcal I$ is of order $n$, so the linear rate is $O_p(n^{-1/2})$ — in practice nearly as fast as Newton for moderate $n$.</p>
<p><strong>Global behaviour</strong> is another matter. Newton steps are only guaranteed to increase $\ell$ when they are short. With concave $\ell$ (canonical link) Newton plus a line search (step-halving until the deviance decreases) converges from any start to the maximizer, if one exists. R's <code>glm.fit</code> halves the step only when the deviance becomes non-finite or the linear predictor leaves the valid region; the <code>glm2</code> package (Marschner, 2011) also halves when the deviance increases, which is what fixes non-convergence for log-binomial and identity-link Poisson models whose parameter space is constrained.</p>
<div class="warn" data-name="When iterations diverge">
<ul>
<li><strong>Separation</strong> (binary) or all-zero cells (counts): the MLE does not exist, iterates head to infinity, weights collapse, and $(X^\top WX)^{-1}$ blows up. Convergence "warnings" are the symptom; see <a href="#/t/logistic-regression">logistic regression</a>.</li>
<li><strong>Constrained parameter spaces</strong>: with log link for binomial data, $\mu = e^\eta$ must stay below 1; the MLE may lie on the boundary where the score is not zero.</li>
<li><strong>Collinearity</strong>: $X^\top WX$ nearly singular; QR with pivoting detects rank deficiency.</li>
</ul>
</div>`,
      },
      {
        title: 'What the converged fit gives you for free',
        html: R`
<p>At convergence $\hat\bbeta = (X^\top\hat WX)^{-1}X^\top\hat W\hat{\mathbf z}$ is a fixed point, and the final WLS fit carries all the standard inferential quantities:</p>
<ul>
<li><strong>Covariance:</strong> $\widehat\Cov(\hat\bbeta) = \hat\phi(X^\top\hat WX)^{-1}$, the inverse expected information. For non-canonical links some software reports the inverse <em>observed</em> information instead; both are consistent.</li>
<li><strong>Hat matrix:</strong> $H = \hat W^{1/2}X(X^\top\hat WX)^{-1}X^\top\hat W^{1/2}$, a projection with $\tr H = p$. Its diagonal $h_{ii}$ gives leverages used to standardize residuals and in Cook's distance (<a href="#/t/deviance">deviance and residuals</a>). Unlike linear regression, leverage depends on $\hat\bmu$: points with fitted probabilities near 0 or 1 have small weight and hence small leverage, even when $\bx_i$ is extreme.</li>
<li><strong>Dispersion:</strong> $\hat\phi = \frac{1}{n-p}\sum_i \hat W_i(\hat z_i - \hat\eta_i)^2 = \frac{1}{n-p}\sum_i\frac{w_i(y_i-\hat\mu_i)^2}{V(\hat\mu_i)}$, the Pearson estimate — the weighted residual mean square of the final WLS fit.</li>
</ul>
<div class="note" data-name="A useful mental model">
<p>A converged GLM is "a weighted linear regression of $\hat{\mathbf z}$ on $X$ that happens to be self-consistent". Many diagnostics, approximations (one-step deletion estimates, leave-one-out formulas), and extensions are obtained by taking a linear-model formula and evaluating it on the final working response and weights.</p>
</div>`,
      },
      {
        title: 'Extensions of the IRLS template',
        html: R`
<p>Green (1984) observed that IRLS solves any problem of the form "maximize $\sum_i \ell_i(\eta_i)$ with $\boldsymbol\eta = X\bbeta$" by repeatedly solving WLS with weights $-\partial^2\ell_i/\partial\eta_i^2$ (or their expectations) and working response $\eta_i + (\partial\ell_i/\partial\eta_i)/W_i$. This covers:</p>
<table>
<thead><tr><th>Problem</th><th>Modification</th></tr></thead>
<tbody>
<tr><td>Ridge / smoothing penalty $\frac12\bbeta^\top P\bbeta$</td><td>Solve $(X^\top WX + P)\bbeta = X^\top W\mathbf z$ — penalized IRLS (P-IRLS), the engine of <a href="#/t/gam">GAMs</a></td></tr>
<tr><td>Lasso penalty</td><td>Coordinate descent on the WLS sub-problem (glmnet uses a quadratic approximation of exactly this form)</td></tr>
<tr><td>Quasi-likelihood</td><td>Same iterations, only $V$ and $g$ needed; see <a href="#/t/overdispersion">overdispersion</a></td></tr>
<tr><td>Robust M-estimation</td><td>Weights $\psi(r_i)/r_i$ from a robust loss (Huber, Tukey bisquare)</td></tr>
<tr><td>Generalized estimating equations</td><td>Block-diagonal weights $V_i^{-1}$ per cluster; see <a href="#/t/glmm-gee">GEE</a></td></tr>
</tbody>
</table>
<p>For very large $n$ or $p$, forming $X^\top WX$ at $O(np^2)$ per iteration becomes the bottleneck; quasi-Newton methods (L-BFGS) on $-\ell$ or stochastic gradient methods take over, giving up IRLS's quadratic convergence for cheaper iterations.</p>`,
      },
    ],
    quiz: [
      { q: R`Derive the working response and weights for logistic regression and show that IRLS is Newton's method.`,
        a: R`With the logit link $g'(\mu) = 1/\{\mu(1-\mu)\}$ and $V(\mu) = \mu(1-\mu)$, so $W_i = \mu_i(1-\mu_i)$ and $z_i = \eta_i + (y_i-\mu_i)/\{\mu_i(1-\mu_i)\}$. The update $(X^\top WX)^{-1}X^\top W\mathbf z = \bbeta + (X^\top WX)^{-1}X^\top(\by - \bmu)$ is the Newton step, because the Hessian of the logistic log-likelihood is exactly $-X^\top WX$.` },
      { q: R`Why initialize IRLS at $\mu^{(0)}\approx y$ rather than at $\bbeta^{(0)} = \mathbf 0$?`,
        a: R`Starting on the mean scale puts the first linearization near the data, where the model curve passes, so the first WLS step is already close to the answer; it also avoids evaluating $g(y)$ at the boundary by using a small adjustment. A start at $\bbeta = \mathbf 0$ can be far from the optimum on a scale where $g^{-1}$ is very nonlinear, producing overshoot (as in the Poisson example) or divergence.` },
      { q: R`For a probit model, does IRLS (Fisher scoring) converge quadratically? Explain.`,
        a: R`Not in general. The probit link is not canonical, so the observed and expected information differ by a mean-zero term involving $(y_i - \mu_i)$. Scoring converges linearly with rate equal to the spectral radius of $\mathcal I^{-1}(\mathcal I - J)$ at the optimum, which is $O_p(n^{-1/2})$, so it is fast but not quadratic. Newton with the observed Hessian would converge quadratically locally.` },
      { q: R`Why does $\phi$ not appear in the IRLS iterations, and how is it estimated afterwards?`,
        a: R`Both $U$ and $\mathcal I$ carry a factor $1/\phi$, which cancels in $\mathcal I^{-1}U$; $\hat\bbeta$ does not depend on $\phi$. After convergence, $\phi$ is estimated by the Pearson statistic $X^2/(n-p)$, the residual mean square of the final weighted least-squares fit on the working scale.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'deviance',
    domain: 'glm',
    order: 50,
    title: 'Deviance, Residuals & Goodness of Fit',
    subtitle: 'The likelihood-ratio distance to the saturated model, and when it can be trusted',
    level: 2,
    sources: { other: 'McCullagh & Nelder (1989), §2.3–2.4, Ch. 12; Jørgensen (1997); Dunn & Smyth (1996, 2018); Hosmer & Lemeshow (2013)' },
    prereqs: ['glm-framework', 'likelihood-ratio-tests'],
    related: ['glm-inference', 'overdispersion', 'regression-diagnostics', 'irls'],
    tags: ['deviance', 'saturated model', 'Pearson statistic', 'residuals', 'goodness of fit'],
    summary: R`The deviance measures how far a fitted GLM is from the <em>saturated</em> model that fits every observation exactly: $D = 2\phi\{\ell(\by;\by) - \ell(\hat\bmu;\by)\}$. For the Normal family it is the residual sum of squares, and differences of deviances between nested models are likelihood-ratio statistics with the usual $\chi^2$ limits. The deviance itself, however, is only approximately $\chi^2_{n-p}$ under <em>small-dispersion</em> asymptotics — many trials per binomial observation, large Poisson means. For ungrouped binary data it is a function of the fitted values alone and says nothing about fit. Pearson and deviance residuals, both normalized versions of $y_i - \hat\mu_i$, are the diagnostic tools that replace OLS residuals.`,
    survey: {
      task: 'Measure lack of fit; compare nested GLMs; diagnose individual observations',
      assumptions: R`Correct family for $\chi^2$ reference; small-dispersion regime (large counts / many trials) for absolute goodness of fit`,
      output: R`Deviance $D$, Pearson $X^2$, residual degrees of freedom $n-p$, residuals $r_{D,i}$, $r_{P,i}$, leverages`,
      strengths: 'Generalizes RSS and $F$-tests; additive analysis of deviance; residuals for any family',
      weaknesses: R`Absolute $\chi^2_{n-p}$ reference fails for sparse data (binary, small counts); residuals of discrete data are lumpy`,
      complexity: R`$O(n)$ given the fit`,
    },
    sections: [
      {
        title: 'The saturated model and the deviance',
        html: R`
<p>Among all models for $n$ independent observations from a given EDF, the largest is the <strong>saturated model</strong>, with one free mean per observation. Its MLE is $\tilde\mu_i = y_i$ (setting the score of each term to zero), and it attains the largest possible log-likelihood, $\ell(\by;\by)$. A fitted model with $p$ parameters can be judged by how much log-likelihood it gives up.</p>
<div class="def" data-name="Deviance">
<p>With $\tilde\theta_i = \theta(y_i)$ and $\hat\theta_i = \theta(\hat\mu_i)$, the (unscaled) deviance is
$$D(\by;\hat\bmu) = 2\phi\{\ell(\by;\by) - \ell(\hat\bmu;\by)\} = 2\sum_{i=1}^n w_i\Big\{y_i(\tilde\theta_i - \hat\theta_i) - b(\tilde\theta_i) + b(\hat\theta_i)\Big\},$$
and the <em>scaled deviance</em> is $D/\phi$. Because $\phi$ multiplies out, $D$ is computable without knowing $\phi$.</p>
</div>
<div class="thm" data-kind="Lemma" data-name="Integral form of the unit deviance">
<p>$D = \sum_i w_i\,d(y_i,\hat\mu_i)$ where the <em>unit deviance</em> is
$$d(y,\mu) = 2\int_\mu^y\frac{y - t}{V(t)}\d t\;\ge 0,$$
with equality iff $y = \mu$. Near $y=\mu$, $d(y,\mu) = (y-\mu)^2/V(\mu) + O(|y-\mu|^3)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For one observation with unit weight, $\phi\,\ell(\mu;y) = y\theta(\mu) - b(\theta(\mu)) + \text{const}$. Differentiating in $\mu$, using $d\theta/d\mu = 1/V(\mu)$ and $b'(\theta(\mu)) = \mu$: $\phi\,\partial\ell/\partial\mu = (y - \mu)/V(\mu)$. So $\phi\{\ell(y;y) - \ell(\mu;y)\} = \int_\mu^y (y-t)/V(t)\d t$. The integrand has the sign of $y - t$, which is the sign of the direction of integration, so the integral is nonnegative and zero only if $y=\mu$. For the expansion, substitute $t = \mu + s$ and expand $1/V(\mu+s) = 1/V(\mu) + O(s)$: $\int_0^{y-\mu}(y-\mu-s)\,\d s/V(\mu) = (y-\mu)^2/\{2V(\mu)\}$, and doubling gives the result.</p>
</details>
<table>
<thead><tr><th>Family</th><th>Unit deviance $d(y,\mu)$</th></tr></thead>
<tbody>
<tr><td>Normal</td><td>$(y-\mu)^2$</td></tr>
<tr><td>Poisson</td><td>$2\{y\log(y/\mu) - (y-\mu)\}$ (with $0\log0 = 0$)</td></tr>
<tr><td>Binomial proportion ($w = m$)</td><td>$2\{y\log(y/\mu) + (1-y)\log\frac{1-y}{1-\mu}\}$</td></tr>
<tr><td>Gamma</td><td>$2\{-\log(y/\mu) + (y-\mu)/\mu\}$</td></tr>
<tr><td>Inverse Gaussian</td><td>$(y-\mu)^2/(\mu^2y)$</td></tr>
</tbody>
</table>
<p>For the Normal family $D = \sum_i(y_i - \hat\mu_i)^2 = \text{RSS}$, and everything below reduces to familiar linear-model results.</p>`,
      },
      {
        title: 'Comparing nested models: analysis of deviance',
        html: R`
<p>Let model $M_0$ (with $p_0$ parameters) be nested in $M_1$ ($p_1 > p_0$). The saturated log-likelihood cancels in the difference:
$$\frac{D_0 - D_1}{\phi} = 2\{\ell(\hat\bmu_1;\by) - \ell(\hat\bmu_0;\by)\},$$
which is exactly the <a href="#/t/likelihood-ratio-tests">likelihood-ratio statistic</a>. With $p_0,p_1$ fixed and $n\to\infty$, Wilks' theorem applies: under $M_0$, $(D_0 - D_1)/\phi\dto\chi^2_{p_1-p_0}$. When $\phi$ is unknown (Normal, gamma, quasi-likelihood), replace it by the Pearson estimate from the larger model and use
$$F = \frac{(D_0 - D_1)/(p_1 - p_0)}{\hat\phi}\;\approx\;F_{p_1-p_0,\;n-p_1},$$
which is exact for the Normal linear model. A sequence of nested fits produces an <em>analysis of deviance</em> table, the GLM analogue of ANOVA; as in unbalanced ANOVA, sequential deviance reductions depend on the order of terms.</p>
<p>This use of the deviance is reliable: it is a statement about a fixed number of parameters. Using $D$ itself as a goodness-of-fit statistic is different, because the saturated model's dimension grows with $n$.</p>`,
      },
      {
        title: 'Is $D$ approximately $\chi^2_{n-p}$? Small-dispersion asymptotics',
        html: R`
<p>Wilks' theorem compares two models of <em>fixed</em> dimension. The saturated model has $n$ parameters, so letting $n\to\infty$ does not justify $D/\phi\approx\chi^2_{n-p}$. A different limit does.</p>
<div class="thm" data-kind="Theorem" data-name="Small-dispersion asymptotics (Jørgensen, 1987)">
<p>Fix $n$ and the design, and let the dispersion of each observation go to zero: $\phi/w_i\to0$ (binomial denominators $m_i\to\infty$, Poisson means $\mu_i\to\infty$, gamma shapes $\to\infty$). If the model is correct, then $D/\phi\dto\chi^2_{n-p}$ and $X^2/\phi\dto\chi^2_{n-p}$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>As $\phi/w_i\to 0$ each standardized observation $(Y_i - \mu_i)/\sqrt{\phi V(\mu_i)/w_i}$ becomes standard normal (CLT for sums of binomial trials, Poisson with large mean). By the lemma above, $w_i d(Y_i,\mu_i)/\phi = w_i(Y_i-\mu_i)^2/\{\phi V(\mu_i)\} + o_p(1)$, so to first order the GLM behaves like a weighted Normal linear model on the working scale with known weights. In that model the scaled residual sum of squares after fitting $p$ parameters is exactly $\chi^2_{n-p}$ (Cochran's theorem: projection onto an $(n-p)$-dimensional residual space).</p>
</details>
<p>The practical rule: compare $D$ or $X^2$ with $\chi^2_{n-p}$ only when most fitted counts are reasonably large (say $\hat\mu_i\ge 5$ for Poisson, $m_i\hat\pi_i$ and $m_i(1-\hat\pi_i)\ge5$ for binomial). Pearson's $X^2$ approaches its limit faster than $D$ for sparse data, which is why $X^2/(n-p)$ is the preferred dispersion estimate.</p>
<div class="thm" data-kind="Proposition" data-name="The deviance of ungrouped binary data is uninformative">
<p>For logistic regression on Bernoulli responses (with an intercept or any design), the deviance at the MLE is
$$D = -2\sum_{i=1}^n\big\{\hat\pi_i\log\hat\pi_i + (1-\hat\pi_i)\log(1-\hat\pi_i)\big\},$$
a function of the fitted probabilities alone. Given $\hat\bbeta$, it does not depend on the observed $y_i$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For binary data $\ell(\by;\by) = 0$ (each saturated fitted probability is 0 or 1), so $D = -2\ell(\hat\bbeta) = -2\sum_i\{y_i\hat\eta_i - \log(1 + e^{\hat\eta_i})\}$. The score equations give $X^\top\by = X^\top\hat{\boldsymbol\pi}$, hence $\sum_i y_i\hat\eta_i = \hat\bbeta^\top X^\top\by = \hat\bbeta^\top X^\top\hat{\boldsymbol\pi} = \sum_i\hat\pi_i\hat\eta_i$. Substituting $\hat\eta_i = \log\hat\pi_i - \log(1-\hat\pi_i)$ and $\log(1+e^{\hat\eta_i}) = -\log(1-\hat\pi_i)$ yields the formula.</p>
</details>
<p>Two data sets with the same fitted probabilities have the same deviance regardless of how badly the observed outcomes match them. For binary data, goodness of fit must come from grouping (the Hosmer–Lemeshow test groups observations into deciles of $\hat\pi$ and computes a Pearson statistic, referred to $\chi^2_{g-2}$ by simulation evidence rather than theory), from smoothed residual plots, or from tests against specific richer models. The lab shows the null distribution of $D$ collapsing away from $\chi^2_{n-p}$ as the binomial denominator $m$ shrinks to 1.</p>`,
        lab: 'deviance-gof',
      },
      {
        title: 'Residuals for GLMs',
        html: R`
<p>Raw residuals $y_i - \hat\mu_i$ have variance $\phi V(\mu_i)/w_i$, so they are not comparable across observations. Several standardizations are in use:</p>
<table>
<thead><tr><th>Residual</th><th>Definition</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Pearson</td><td>$r_{P,i} = \dfrac{y_i - \hat\mu_i}{\sqrt{V(\hat\mu_i)/w_i}}$</td><td>$\sum r_{P,i}^2 = X^2$; skewed for small counts</td></tr>
<tr><td>Deviance</td><td>$r_{D,i} = \sgn(y_i-\hat\mu_i)\sqrt{w_i\,d(y_i,\hat\mu_i)}$</td><td>$\sum r_{D,i}^2 = D$; closer to normal than Pearson</td></tr>
<tr><td>Standardized</td><td>$r_i/\sqrt{\hat\phi(1-h_{ii})}$</td><td>Corrects for leverage from the IRLS hat matrix</td></tr>
<tr><td>Randomized quantile</td><td>$\Phi^{-1}(u_i)$, $u_i\sim\text{Unif}\big(F(y_i^-;\hat\mu_i),F(y_i;\hat\mu_i)\big)$</td><td>Exactly $\N(0,1)$ under the true model (Dunn & Smyth, 1996)</td></tr>
</tbody>
</table>
<p>Why do deviance residuals look more normal? The signed root of a likelihood-ratio statistic has skewness $O(n^{-1/2})$ removed to first order, whereas the Pearson residual of a Poisson count inherits the count's skewness $1/\sqrt\mu$. For continuous families with small dispersion both are approximately $\N(0,\phi)$.</p>
<p>The randomized quantile residual deserves special mention. For a discrete response, $F(Y;\mu)$ is not uniform, but if $U$ is drawn uniformly on the jump of $F$ at the observed $y$, then $U\sim\text{Unif}(0,1)$ exactly when $\mu$ is the true mean (the probability integral transform with randomization). After the plug-in of $\hat\mu$ the residuals are approximately IID $\N(0,1)$, so the eye can read QQ-plots and residual-versus-fitted plots as in linear regression — free of the "stripes" that discrete Pearson and deviance residuals produce.</p>
<p>Leverage and influence carry over through the IRLS hat matrix $H = \hat W^{1/2}X(X^\top\hat WX)^{-1}X^\top\hat W^{1/2}$: Williams' approximation to Cook's distance is $C_i = r_{P,i}^2h_{ii}/\{p\hat\phi(1-h_{ii})^2\}$, from a one-step Newton approximation to the leave-one-out fit.</p>
<p>In the lab, generate Poisson-regression data from different truths and read the residual plot: a funnel or an overall spread well above 1 signals overdispersion; a curved trend signals a missing nonlinear term or wrong link; an excess of large negative residuals at small $\hat\mu$ signals zero inflation.</p>`,
        lab: 'deviance-residuals',
      },
      {
        title: 'Information criteria and other fit summaries',
        html: R`
<p>For comparing non-nested models, use $\text{AIC} = -2\ell(\hat\bbeta) + 2p$ (count $\phi$ as a parameter when it is estimated). Deviance differences are AIC differences up to the penalty: $\text{AIC}_1 - \text{AIC}_0 = (D_1 - D_0)/\phi + 2(p_1 - p_0)$ when $\phi$ is known. Quasi-likelihood models have no likelihood and need QAIC, which divides the deviance by an externally estimated $\hat\phi$.</p>
<p>Pseudo-$R^2$ measures abound; McFadden's $1 - \ell(\hat\bbeta)/\ell(\hat\beta_0)$ is the proportional reduction in deviance relative to the null model when $\ell(\by;\by) = 0$. None has the variance-explained interpretation of the linear-model $R^2$, and they should be compared only across models for the same data.</p>
<div class="warn" data-name="Common errors">
<ul>
<li>Declaring good fit because $D\approx n-p$ for binary data (the proposition above shows this is vacuous).</li>
<li>Dividing a Poisson deviance by $n-p$ and calling the result the dispersion for small counts; $X^2/(n-p)$ is less biased.</li>
<li>Reading discreteness stripes in residual plots as structure. Use randomized quantile residuals or binned residual plots.</li>
<li>Comparing deviances of models fitted to different subsets of the data (missing values dropped differently).</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Derive the Poisson unit deviance from the integral formula $d(y,\mu) = 2\int_\mu^y (y-t)/V(t)\d t$.`,
        a: R`With $V(t) = t$: $2\int_\mu^y (y/t - 1)\d t = 2\{y\log(y/\mu) - (y-\mu)\}$. For $y=0$ the first term is 0 and $d = 2\mu$.` },
      { q: R`Why is the difference of deviances between two nested GLMs approximately $\chi^2$ for large $n$, while the deviance itself may not be?`,
        a: R`The difference is a likelihood-ratio statistic between two models of fixed dimensions $p_0 < p_1$, so Wilks' theorem applies as $n\to\infty$. The deviance alone compares the fitted model with the saturated model, whose dimension $n$ grows with the sample; Wilks does not apply, and a $\chi^2_{n-p}$ approximation needs each observation to be nearly normal (small dispersion).` },
      { q: R`A logistic model fitted to 500 binary outcomes has deviance 487 on 495 degrees of freedom. What can you conclude about goodness of fit?`,
        a: R`Nothing. For ungrouped binary data the deviance is $-2\sum\{\hat\pi\log\hat\pi + (1-\hat\pi)\log(1-\hat\pi)\}$, a function of the fitted values only, so it cannot detect lack of fit. Use grouped tests (Hosmer–Lemeshow), binned residual plots, calibration curves, or likelihood-ratio tests against richer models.` },
      { q: R`What makes randomized quantile residuals exactly normal under the true model, and why randomize?`,
        a: R`For continuous $Y$, $F(Y;\mu)\sim\text{Unif}(0,1)$. For discrete $Y$, $F$ jumps, so draw $U$ uniformly between $F(y^-)$ and $F(y)$; then $\P(U\le u) = u$ exactly. Applying $\Phi^{-1}$ gives a standard normal. Randomization fills in the gaps created by discreteness, removing the artificial banding of Pearson and deviance residuals.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'glm-inference',
    domain: 'glm',
    order: 60,
    title: 'Inference in GLMs: Wald, Score & Likelihood-Ratio',
    subtitle: 'Three asymptotically equivalent tests that disagree exactly when it matters',
    level: 2,
    sources: { cb: '§10.3', other: 'Fahrmeir & Kaufmann (1985); Hauck & Donner (1977); Agresti (2013), §4.3–5.2; Pawitan (2001)' },
    prereqs: ['deviance', 'likelihood-ratio-tests', 'fisher-information'],
    related: ['confidence-intervals', 'delta-method', 'logistic-regression', 'irls'],
    tags: ['Wald test', 'score test', 'likelihood ratio', 'profile likelihood', 'Hauck–Donner'],
    summary: R`Large-sample GLM inference rests on $\hat\bbeta\approx\N\big(\bbeta,\phi(X^\top WX)^{-1}\big)$. From the log-likelihood there are three classical ways to test a hypothesis or build an interval: the <em>Wald</em> statistic measures the horizontal distance $\hat\beta - \beta_0$ in standard-error units; the <em>likelihood-ratio</em> statistic measures the vertical drop in log-likelihood; the <em>score</em> (Rao) statistic measures the slope at the null. They agree to first order, but in finite samples — especially for logistic models with large effects, rare events or near-separation — the Wald test can be badly misleading, while likelihood-ratio and profile-likelihood intervals respect the curvature and parameterization invariance of the likelihood.`,
    survey: {
      task: 'Hypothesis tests and confidence intervals for GLM coefficients and fitted means',
      assumptions: R`Correct model; $X^\top WX$ growing (information accumulates); MLE in interior`,
      output: R`$z$/$\chi^2$/$F$ statistics, $p$-values, Wald and profile-likelihood intervals`,
      strengths: 'Wald: one fit; Score: only the null fit; LR: invariant, accurate, uses the actual likelihood shape',
      weaknesses: 'Wald not invariant and fails near boundaries (Hauck–Donner); all three need large samples and fail under separation',
      complexity: R`Wald $O(p^3)$ once; LR one extra fit per test; profile CI a 1-D root search of refits`,
    },
    sections: [
      {
        title: 'Asymptotic normality of the MLE',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Asymptotics of the GLM MLE (Fahrmeir & Kaufmann, 1985)">
<p>For a GLM with canonical link, suppose the smallest eigenvalue of $\mathcal I_n(\bbeta_0) = \phi^{-1}X^\top W(\bbeta_0)X$ tends to infinity and the design satisfies a mild boundedness condition (no single $\bx_i$ dominates the information). Then with probability tending to 1 the MLE exists and is unique, $\hat\bbeta\pto\bbeta_0$, and
$$\mathcal I_n(\bbeta_0)^{1/2}(\hat\bbeta - \bbeta_0)\dto\N(\mathbf 0, I_p).$$
Analogous results hold for non-canonical links under slightly stronger conditions.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Expand the score at $\hat\bbeta$ around $\bbeta_0$: $\mathbf 0 = U(\hat\bbeta) = U(\bbeta_0) - J(\bar\bbeta)(\hat\bbeta - \bbeta_0)$. The score is a sum of independent mean-zero terms with covariance $\mathcal I_n$, so $\mathcal I_n^{-1/2}U(\bbeta_0)\dto\N(\mathbf 0, I_p)$ by the Lindeberg–Feller CLT (the boundedness condition is Lindeberg's condition). For the canonical link $J = \mathcal I$ is nonrandom and continuous in $\bbeta$, and the growing-information condition makes $\mathcal I_n(\bar\bbeta)\mathcal I_n(\bbeta_0)^{-1}\to I_p$ uniformly on shrinking neighbourhoods. Existence comes from showing that $\ell$ is, with high probability, smaller on the boundary of such a neighbourhood than at its centre; concavity then gives uniqueness.</p>
</details>
<p>In practice: $\hat\bbeta\approx\N(\bbeta,\hat\phi(X^\top\hat WX)^{-1})$, and $\widehat\se(\hat\beta_j)$ is the square root of the $j$-th diagonal entry. The relevant "sample size" is the information, not $n$. A logistic regression with 10,000 observations and 12 events carries roughly the information of 12 observations: $W_i = \pi_i(1-\pi_i)\approx\pi_i$ is tiny for the non-events.</p>`,
      },
      {
        title: 'The trinity: Wald, likelihood-ratio and score',
        html: R`
<p>Partition $\bbeta = (\bbeta_1,\bbeta_2)$ with $\bbeta_2\in\R^r$ and test $H_0:\bbeta_2 = \mathbf 0$. Let $\hat\bbeta$ be the unrestricted MLE and $\tilde\bbeta = (\tilde\bbeta_1,\mathbf 0)$ the restricted one. Write $[\mathcal I^{-1}]_{22}$ for the $r\times r$ block of the inverse information.</p>
<div class="cols">
<div><h4>Wald</h4>$$W = \hat\bbeta_2^\top\big([\mathcal I(\hat\bbeta)^{-1}]_{22}\big)^{-1}\hat\bbeta_2$$<p>Needs only the full fit. For one coefficient, $W = z^2$ with $z = \hat\beta_j/\widehat\se$.</p></div>
<div><h4>Likelihood ratio</h4>$$\Lambda = 2\{\ell(\hat\bbeta) - \ell(\tilde\bbeta)\} = \frac{D_0 - D_1}{\phi}$$<p>Needs both fits. Invariant to reparameterization.</p></div>
<div><h4>Score (Rao)</h4>$$S = U(\tilde\bbeta)^\top\mathcal I(\tilde\bbeta)^{-1}U(\tilde\bbeta)$$<p>Needs only the null fit; $U(\tilde\bbeta)$ is zero in the $\bbeta_1$ block.</p></div>
</div>
<div class="thm" data-kind="Theorem" data-name="Asymptotic equivalence">
<p>Under $H_0$ and the conditions above, $W$, $\Lambda$ and $S$ differ by $o_p(1)$ and each converges in distribution to $\chi^2_r$. Under local alternatives $\bbeta_2 = \mathbf h/\sqrt n$ they share the same noncentral $\chi^2_r$ limit.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Approximate $\ell$ near $\hat\bbeta$ by the quadratic $\ell(\hat\bbeta) - \frac12(\bbeta - \hat\bbeta)^\top\mathcal I(\bbeta-\hat\bbeta)$, which has error $o_p(1)$ on $n^{-1/2}$-neighbourhoods. For an exactly quadratic log-likelihood with fixed curvature, all three statistics equal the same quadratic form $\hat\bbeta_2^\top([\mathcal I^{-1}]_{22})^{-1}\hat\bbeta_2$: the Wald statistic by definition, $\Lambda$ by maximizing the quadratic over $\bbeta_1$ (a Schur complement computation), and $S$ because the score of a quadratic is linear in $\bbeta$. Asymptotic normality of $\hat\bbeta_2$ with covariance $[\mathcal I^{-1}]_{22}$ gives the $\chi^2_r$ limit.</p>
</details>
<div class="note">
<p>Picture the profile log-likelihood for a single coefficient. Wald measures the <em>horizontal</em> distance from the peak to $\beta_0$ using the curvature <em>at the peak</em>. LR measures the <em>vertical</em> drop between the peak and $\beta_0$, using the true curve. Score measures the <em>slope</em> at $\beta_0$, normalized by the curvature <em>at</em> $\beta_0$. When the log-likelihood is quadratic all three agree; when it is skewed, they differ, and the LR test, which uses the actual shape, is usually the most reliable.</p>
</div>
<p>Familiar tests are special cases. Pearson's $X^2$ for independence in a two-way table is the score test of the independence log-linear model against the saturated one; the Cochran–Armitage trend test is the score test for the slope in a logistic model with ordered dose scores; the log-rank test is the score test of the Cox model. Score tests are attractive for screening many candidate covariates, since only the null model is fitted. For the Normal linear model with estimated $\sigma^2$, the three satisfy $W\ge\Lambda\ge S$ (Berndt & Savin, 1977), so they can give conflicting conclusions at a fixed level.</p>
<p>When $\phi$ is estimated, replace $\chi^2_r$ by $r F_{r,n-p}$ (Wald or LR divided by $r\hat\phi$), and $z$ by Student $t_{n-p}$ for single coefficients.</p>`,
      },
      {
        title: 'Confidence intervals: Wald versus profile likelihood',
        html: R`
<p>Inverting each test gives an interval. The <strong>Wald interval</strong> $\hat\beta_j\pm z_{1-\alpha/2}\widehat\se$ is symmetric by construction. The <strong>profile-likelihood interval</strong> is
$$\Big\{\beta_j:\;2\big[\ell(\hat\bbeta) - \ell_p(\beta_j)\big]\le\chi^2_{1,1-\alpha}\Big\},\qquad \ell_p(\beta_j) = \max_{\bbeta_{-j}}\ell(\beta_j,\bbeta_{-j}),$$
found by a one-dimensional root search, refitting the model with $\beta_j$ held fixed (as an offset) at each trial value.</p>
<div class="cols">
<div><h4>Wald</h4><p>Cheap. Not invariant: a Wald interval for $\beta$ exponentiated is not the Wald interval for $e^\beta$, and Wald intervals for $\pi$ and $\operatorname{logit}\pi$ disagree. Can extend outside the parameter space. Ignores skewness of $\ell$.</p></div>
<div><h4>Profile likelihood</h4><p>Invariant under monotone reparameterization (the likelihood does not care how you label the parameter). Asymmetric when $\ell$ is. Coverage error is typically smaller. Remains meaningful under separation, where one endpoint is $\pm\infty$.</p></div>
</div>
<div class="thm" data-kind="Proposition" data-name="Hauck–Donner effect (Hauck & Donner, 1977)">
<p>In a binary logistic model, consider a sequence of data sets in which the estimated effect $|\hat\beta_j|$ grows (e.g. by moving an observation in a $2\times2$ table so that one cell count approaches zero). The Wald statistic $z_j = \hat\beta_j/\widehat\se(\hat\beta_j)$ is not monotone in $|\hat\beta_j|$: beyond some point it decreases, tending to 0 as $|\hat\beta_j|\to\infty$, while the likelihood-ratio statistic increases.</p>
</div>
<details class="proof"><summary>Proof sketch (a $2\times 2$ table)</summary>
<p>For a binary exposure with cell counts $a,b$ (exposed events, non-events) and $c,d$ (unexposed), $\hat\beta = \log\frac{ad}{bc}$ and $\widehat\se = \sqrt{1/a + 1/b + 1/c + 1/d}$. Let $b\downarrow0$ with the others fixed (interpreting the counts as continuous to trace the curve): $\hat\beta\sim\log(1/b)$ grows logarithmically while $\widehat\se\sim b^{-1/2}$ grows polynomially, so $z\to0$. The LR statistic, by contrast, increases monotonically toward a finite positive limit (with $a=20$, $c=10$, $d=20$ it rises from 14.4 at $b=4$ to 29.0 at $b=0.01$, while $z$ falls from 3.4 to 0.8): the evidence is strongest exactly where the Wald test reports none.</p>
</details>
<p>The lab plots the profile log-likelihood of a logistic slope, its Wald quadratic approximation, and the three intervals. Push the true slope up with a small $n$ and watch the quadratic approximation fail and the Wald $z$ shrink while the LR statistic grows.</p>`,
        lab: 'glm-inference-profile',
      },
      {
        title: 'Intervals for fitted means and predictions',
        html: R`
<p>For a new covariate vector $\bx_0$, $\hat\eta_0 = \bx_0^\top\hat\bbeta$ has $\widehat\se(\hat\eta_0) = \sqrt{\bx_0^\top\widehat\Cov(\hat\bbeta)\bx_0}$. The recommended interval for $\mu_0$ is built on the link scale and transformed:
$$\Big(g^{-1}\big(\hat\eta_0 - z\,\widehat\se(\hat\eta_0)\big),\; g^{-1}\big(\hat\eta_0 + z\,\widehat\se(\hat\eta_0)\big)\Big)$$
(for increasing $g$). It respects the range of $\mu$ — probabilities stay in $(0,1)$, rates stay positive — and inherits the better normal approximation on the $\eta$ scale. The <a href="#/t/delta-method">delta method</a> applied directly on the $\mu$ scale gives $\widehat\se(\hat\mu_0) = \widehat\se(\hat\eta_0)/|g'(\hat\mu_0)|$ and a symmetric interval that can cross 0 or 1; avoid it near the boundary.</p>
<div class="warn" data-name="Mean versus new observation">
<p>These are intervals for the <em>mean</em> $\mu_0$. A prediction interval for a new $Y_0$ must also include the family's variability ($\phi V(\mu_0)$); for a binary $Y_0$ the notion barely makes sense, and for counts it is best obtained by simulating from the fitted distribution with $\hat\bbeta$ drawn from its approximate sampling distribution.</p>
</div>`,
      },
      {
        title: 'Finite-sample behaviour and remedies',
        html: R`
<p>All three procedures are first-order approximations, and with few events they part ways. For a logistic slope, the Wald standard error inflates as $|\hat\beta_1|$ grows (the Hauck–Donner mechanism), so Wald tests are typically <em>conservative</em> and lose power; and because the Wald interval is symmetric where the log-likelihood is skewed, its errors pile up on one side. Its actual coverage swings with $n$ and the true $\beta_1$, often well above 95% for tiny samples, which is no virtue: the interval is long in the wrong place. Profile-likelihood intervals follow the shape of $\ell$ and stay within a point or two of nominal in the same settings, erring slightly on the liberal side because the LR statistic's $\chi^2_1$ calibration is itself only first-order. The second lab runs the experiment: repeated small samples from a logistic model, recording how often each nominal-95% interval covers the true slope.</p>
<p>Refinements when the sample is genuinely small:</p>
<ul>
<li><strong>Bartlett correction</strong> of $\Lambda$ (in regular continuous models): divide by its estimated mean $r(1 + b/n)$ to match the $\chi^2_r$ mean, reducing the error in the null distribution from $O(n^{-1})$ to $O(n^{-2})$.</li>
<li><strong>Exact conditional inference</strong>: condition on the sufficient statistics of nuisance parameters (available with canonical links, because the sufficient statistics are $X^\top\by$), giving tests whose null distribution is free of them — Fisher's exact test is the $2\times2$ case.</li>
<li><strong>Firth / Jeffreys-penalized likelihood</strong> with profile penalized-likelihood intervals, which behave well even under separation.</li>
<li><strong>Parametric bootstrap</strong> of the LR statistic (<a href="#/t/parametric-bootstrap">parametric bootstrap</a>).</li>
</ul>
<p>Finally, the model-based standard errors are only as good as the variance assumption: with <a href="#/t/overdispersion">overdispersion</a> they are too small by roughly $\sqrt\phi$, and all three tests become anti-conservative.</p>`,
        lab: 'glm-inference-coverage',
      },
    ],
    quiz: [
      { q: R`Which of the three tests requires fitting only the null model, and why is that useful?`,
        a: R`The score test: it evaluates the gradient and information of the full model at the restricted MLE $\tilde\bbeta$. This makes it cheap to screen many candidate additions (each needs no new fit), and it is well defined even when the alternative model's MLE does not exist, e.g. under separation.` },
      { q: R`Explain why the Wald interval for an odds ratio, $\exp(\hat\beta\pm1.96\,\widehat\se)$, is not the same as the Wald interval computed for $\psi = e^\beta$ directly via the delta method, and which is preferable.`,
        a: R`The delta-method Wald interval for $\psi$ is $e^{\hat\beta}\pm1.96\,e^{\hat\beta}\widehat\se$, symmetric on the ratio scale, which can include negative values and misrepresents the skewed sampling distribution of $\hat\psi$. Exponentiating the interval for $\beta$ uses the scale on which the normal approximation is better and respects positivity. Wald intervals are not invariant; profile-likelihood intervals would give the same answer on either scale.` },
      { q: R`In a $2\times2$ table, exposed cases $a=20$, exposed controls $b=1$, unexposed cases $c=10$, unexposed controls $d=20$. Compute the Wald $z$ for the log odds ratio and explain why it understates the evidence.`,
        a: R`$\hat\beta = \log(20\cdot20/(1\cdot10)) = \log 40\approx3.69$; $\widehat\se = \sqrt{1/20 + 1/1 + 1/10 + 1/20}\approx1.095$; $z\approx3.37$. The $1/b = 1$ term dominates the SE, and as $b\to0$ the SE grows faster than $\hat\beta$ (Hauck–Donner). The log-likelihood is far from quadratic here: the LR statistic is $G^2\approx22.9$, about twice $z^2\approx11.4$.` },
      { q: R`Why is the effective sample size for a rare-event logistic regression closer to the number of events than to $n$?`,
        a: R`Information is $X^\top WX$ with $W_i = \pi_i(1-\pi_i)\approx\pi_i$ when events are rare, so $\sum_i W_i\approx\sum_i\pi_i\approx$ the expected number of events. Each non-event contributes very little information, so standard errors scale like $1/\sqrt{\#\text{events}}$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'overdispersion',
    domain: 'glm',
    order: 70,
    title: 'Overdispersion, Quasi-Likelihood & Negative Binomial',
    subtitle: 'When the variance outruns the mean: repair the standard errors or change the model',
    level: 3,
    sources: { other: 'Wedderburn (1974); McCullagh (1983); White (1982); Cameron & Trivedi (2013), Ch. 3; Hilbe (2011); Dean & Lawless (1989)' },
    prereqs: ['poisson-regression', 'deviance'],
    related: ['glmm-gee', 'irls', 'glm-inference', 'robust-statistics'],
    tags: ['quasi-likelihood', 'sandwich estimator', 'negative binomial', 'gamma–Poisson mixture', 'estimating equations'],
    summary: R`Poisson and binomial models fix the variance as a function of the mean, $\Var Y = \mu$ or $m\pi(1-\pi)$, with no free scale. Real counts are almost always more variable — through unobserved heterogeneity, clustering or contagion — and when they are, the point estimates stay consistent but every model-based standard error is too small by a factor of about $\sqrt\phi$. There are three repairs with different assumptions: <em>quasi-likelihood</em> keeps the mean and the shape of the variance function but frees the scale, $\Var Y = \phi V(\mu)$; the <em>sandwich</em> estimator keeps only the mean model and estimates the variance empirically; the <em>negative binomial</em> builds a full likelihood from a gamma-distributed Poisson rate, giving $\Var Y = \mu + \mu^2/k$.`,
    survey: {
      task: 'Valid inference for count or binomial regressions whose variance exceeds the nominal one',
      assumptions: R`Correct mean model; quasi: $\Var = \phi V(\mu)$; NB: gamma–Poisson mixture ($\Var = \mu + \mu^2/k$); sandwich: independence only`,
      output: R`$\hat\bbeta$ with corrected SEs; $\hat\phi$ or $\hat k$; $F$-tests or LR tests`,
      strengths: 'Quasi and sandwich need no new likelihood; NB gives a full model, predictive distributions and efficient weights',
      weaknesses: 'Quasi has no likelihood (no AIC); sandwich is noisy in small samples; NB assumes a specific variance shape',
      complexity: R`As for IRLS; NB adds a 1-D optimization for $k$`,
    },
    sections: [
      {
        title: 'Diagnosis and consequences',
        html: R`
<p>Under a Poisson model the Pearson statistic $X^2 = \sum_i(y_i - \hat\mu_i)^2/\hat\mu_i$ has expectation close to $n - p$. A ratio $X^2/(n-p)$ well above 1 (say 1.5 or more with a reasonable sample size) indicates <strong>overdispersion</strong>; a ratio below 1, rarer, indicates underdispersion. Before blaming the variance, rule out a misspecified mean (missing covariates, wrong functional form, outliers), which also inflates $X^2$ — see the <a href="#/t/deviance">deviance and residuals</a> lab.</p>
<p>Mechanisms that produce it:</p>
<ul>
<li><strong>Unobserved heterogeneity</strong>: if $Y\mid\lambda\sim\text{Poisson}(\lambda)$ and $\lambda$ varies across units with the same $\bx$, then $\Var Y = \E\lambda + \Var\lambda > \E Y$ by the law of total variance.</li>
<li><strong>Clustering</strong>: binomial successes among correlated trials (litter mates, classrooms) have $\Var S = m\pi(1-\pi)\{1 + (m-1)\rho\}$ with intra-cluster correlation $\rho$.</li>
<li><strong>Contagion</strong>: one event raising the chance of the next.</li>
</ul>
<div class="thm" data-kind="Proposition" data-name="What overdispersion does and does not break">
<p>If $\E(Y_i\mid\bx_i) = g^{-1}(\bx_i^\top\bbeta)$ is correct but $\Var(Y_i\mid\bx_i) = \phi V(\mu_i)$ with $\phi > 1$, the Poisson (or binomial) MLE remains consistent and asymptotically normal, with covariance $\phi\,\mathcal I^{-1}$ instead of $\mathcal I^{-1}$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The estimating equation $\sum_i W_iG_{ii}(y_i - \mu_i)\bx_i = \mathbf 0$ has mean zero at the true $\bbeta$ whatever the variance. Its sensitivity $-\E\,\partial U/\partial\bbeta = \mathcal I$ is unchanged (it depends only on the mean), while $\Cov U$ is multiplied by $\phi$ because every $\Var Y_i$ is. The M-estimation sandwich gives $\mathcal I^{-1}(\phi\mathcal I)\mathcal I^{-1} = \phi\mathcal I^{-1}$.</p>
</details>
<p>So Wald intervals have actual coverage about $2\Phi(1.96/\sqrt\phi) - 1$: with $\phi = 4$, nominal 95% intervals cover only about 67% of the time, and $p$-values are wildly optimistic.</p>`,
      },
      {
        title: 'Quasi-likelihood',
        html: R`
<p>Wedderburn (1974) observed that the GLM score uses only the mean and variance function. Specify just those two:
$$\E Y_i = \mu_i(\bbeta),\qquad \Var Y_i = \phi V(\mu_i),$$
with no distribution, and define the <strong>quasi-likelihood</strong> of one observation by
$$Q(\mu;y) = \int_y^\mu\frac{y - t}{\phi V(t)}\d t.$$</p>
<div class="thm" data-kind="Proposition" data-name="The quasi-score behaves like a score">
<p>Let $q = \partial Q/\partial\mu = (Y-\mu)/\{\phi V(\mu)\}$. Then $\E q = 0$ and $\Var q = -\E\,\partial q/\partial\mu = 1/\{\phi V(\mu)\}$ — the two Bartlett identities a genuine log-likelihood derivative satisfies.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\E q = (\E Y - \mu)/\{\phi V(\mu)\} = 0$; $\Var q = \Var Y/\{\phi V(\mu)\}^2 = 1/\{\phi V(\mu)\}$; and $\partial q/\partial\mu = -1/\{\phi V(\mu)\} - (Y-\mu)V'(\mu)/\{\phi V(\mu)^2\}$, whose expectation is $-1/\{\phi V(\mu)\}$.</p>
</details>
<p>The maximum quasi-likelihood estimator solves $\sum_i\frac{\partial\mu_i}{\partial\bbeta}\frac{y_i - \mu_i}{V(\mu_i)} = \mathbf 0$ — identical to the Poisson/binomial score equations when $V$ matches, so <strong>$\hat\bbeta$ is unchanged</strong> and computed by the same IRLS. Only the covariance changes: $\widehat\Cov(\hat\bbeta) = \hat\phi(X^\top\hat WX)^{-1}$ with
$$\hat\phi = \frac{X^2}{n-p} = \frac{1}{n-p}\sum_i\frac{(y_i - \hat\mu_i)^2}{V(\hat\mu_i)}.$$
Quasi-Poisson standard errors are Poisson standard errors times $\sqrt{\hat\phi}$; nested models are compared by $F = \{(D_0 - D_1)/r\}/\hat\phi$.</p>
<div class="thm" data-kind="Theorem" data-name="Optimality of the quasi-score (Godambe, 1960; McCullagh, 1983)">
<p>Among all unbiased estimating functions that are linear in the data, $\mathbf g(\bbeta) = \sum_i\mathbf a_i(\bbeta)(y_i - \mu_i)$ with nonrandom $\mathbf a_i\in\R^p$, the quasi-score choice $\mathbf a_i = \mathbf D_i/V(\mu_i)$, $\mathbf D_i = \partial\mu_i/\partial\bbeta$, minimizes the asymptotic covariance of the root (in the positive semidefinite order), achieving $\phi\big(\sum_i\mathbf D_i\mathbf D_i^\top/V_i\big)^{-1}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The root of $\mathbf g$ has asymptotic covariance $A^{-1}BA^{-\top}$ with $A = \sum_i\mathbf a_i\mathbf D_i^\top$ (minus the expected derivative) and $B = \sum_i\mathbf a_i\mathbf a_i^\top\phi V_i$. Stack the rows $\mathbf m_i^\top = \sqrt{\phi V_i}\,\mathbf a_i^\top$ into $M$ and $\mathbf n_i^\top = \mathbf D_i^\top/\sqrt{\phi V_i}$ into $N$. Then $A = M^\top N$, $B = M^\top M$, and the claimed optimum is $(N^\top N)^{-1}$. We need $(M^\top N)^{-1}M^\top M(N^\top M)^{-1}\succeq(N^\top N)^{-1}$, equivalently $M^\top M\succeq M^\top N(N^\top N)^{-1}N^\top M$, i.e. $M^\top(I - P_N)M\succeq0$ with $P_N$ the projection onto the column space of $N$. This holds because $I - P_N$ is a projection. Equality holds when $M\propto N$, i.e. $\mathbf a_i\propto\mathbf D_i/V_i$.</p>
</details>
<p>So quasi-likelihood is the best you can do with second-moment assumptions alone — the analogue of Gauss–Markov for GLMs.</p>`,
      },
      {
        title: 'The sandwich: when even the variance function is wrong',
        html: R`
<p>If $V$ itself is misspecified, $\hat\bbeta$ from any GLM estimating equation with a correct mean is still consistent, but $\phi(X^\top WX)^{-1}$ is wrong in shape, not just scale. The robust (Huber–White) covariance estimates the variance of the score empirically.</p>
<div class="thm" data-kind="Theorem" data-name="Sandwich covariance (Huber, 1967; White, 1982)">
<p>Let $\hat\bbeta$ solve $\sum_i\boldsymbol\psi_i(\bbeta) = \mathbf 0$ with $\E\boldsymbol\psi_i(\bbeta_0) = \mathbf 0$, independent across $i$. Under regularity conditions,
$$\hat\bbeta\approx\N\big(\bbeta_0,\;A^{-1}BA^{-\top}\big),\quad A = \sum_i\E\Big[-\frac{\partial\boldsymbol\psi_i}{\partial\bbeta^\top}\Big],\quad B = \sum_i\E\big[\boldsymbol\psi_i\boldsymbol\psi_i^\top\big],$$
estimated consistently by plugging in $\hat\bbeta$ and replacing $B$ by $\hat B = \sum_i\boldsymbol\psi_i(\hat\bbeta)\boldsymbol\psi_i(\hat\bbeta)^\top$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Taylor-expand: $\mathbf 0 = \sum_i\boldsymbol\psi_i(\hat\bbeta) = \sum_i\boldsymbol\psi_i(\bbeta_0) - \hat A(\hat\bbeta - \bbeta_0) + o_p(\sqrt n)$, where $\hat A\approx A$ by a law of large numbers. So $\hat\bbeta - \bbeta_0\approx A^{-1}\sum_i\boldsymbol\psi_i(\bbeta_0)$, a sum of independent mean-zero vectors with covariance $B$; apply the CLT. $\hat B$ is consistent because $\E[\boldsymbol\psi_i\boldsymbol\psi_i^\top] = \Cov\boldsymbol\psi_i$ and the plug-in error is negligible.</p>
</details>
<p>For a Poisson log-linear fit, $\boldsymbol\psi_i = (y_i - \mu_i)\bx_i$, the "bread" is $A = X^\top\diag(\mu_i)X$ and the "meat" is $\hat B = \sum_i(y_i - \hat\mu_i)^2\bx_i\bx_i^\top$. When the model variance is right, $\E\hat B\approx A$ and the sandwich collapses to $A^{-1}$. With clustered data, $\boldsymbol\psi$ is summed within clusters before taking outer products — the cluster-robust sandwich used by <a href="#/t/glmm-gee">GEE</a>.</p>
<div class="warn" data-name="Small samples">
<p>$\hat B$ is built from squared residuals, which are small at high-leverage points (the fit is pulled towards them). The sandwich is biased downward and highly variable for small $n$; leverage-adjusted versions (HC2, HC3 — dividing residuals by $\sqrt{1-h_{ii}}$ or $1-h_{ii}$) and $t$ reference distributions help.</p>
</div>`,
      },
      {
        title: 'The negative binomial as a gamma–Poisson mixture',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Gamma mixture of Poissons">
<p>If $Y\mid\lambda\sim\text{Poisson}(\lambda)$ and $\lambda\sim\text{Gamma}$ with shape $k$ and mean $\mu$ (rate $k/\mu$), then marginally
$$\P(Y = y) = \frac{\Gamma(y+k)}{\Gamma(k)\,y!}\Big(\frac{k}{k+\mu}\Big)^k\Big(\frac{\mu}{k+\mu}\Big)^y,\qquad y = 0,1,2,\dots,$$
the negative binomial with $\E Y = \mu$ and $\Var Y = \mu + \mu^2/k$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Integrate the Poisson probability against the gamma density $\frac{(k/\mu)^k}{\Gamma(k)}\lambda^{k-1}e^{-\lambda k/\mu}$:
$$\P(Y=y) = \frac{(k/\mu)^k}{\Gamma(k)\,y!}\int_0^\infty\lambda^{y+k-1}e^{-\lambda(1 + k/\mu)}\d\lambda = \frac{(k/\mu)^k\,\Gamma(y+k)}{\Gamma(k)\,y!\,(1+k/\mu)^{y+k}}.$$
Since $(k/\mu)^k(1 + k/\mu)^{-(y+k)} = k^k\mu^y(\mu+k)^{-(y+k)}$, this is the stated form. For the moments use iterated expectation: $\E Y = \E\lambda = \mu$ and $\Var Y = \E\Var(Y\mid\lambda) + \Var\E(Y\mid\lambda) = \E\lambda + \Var\lambda = \mu + \mu^2/k$.</p>
</details>
<p>The parameter $\alpha = 1/k$ measures extra-Poisson variation; $\alpha\to0$ recovers the Poisson. Regression enters through $\log\mu_i = \bx_i^\top\bbeta$ (the "NB2" model). An alternative, "NB1", has $\Var Y = \mu(1+\alpha)$, the same shape as quasi-Poisson.</p>
<p><strong>Fitting.</strong> For fixed $k$ the NB is an exponential dispersion family in $\mu$ with $V(\mu) = \mu + \mu^2/k$ and $\phi = 1$, so $\bbeta$ is fitted by IRLS with log link: weights $W_i = \mu_i/(1 + \mu_i/k)$ and working response $z_i = \eta_i + (y_i - \mu_i)/\mu_i$. One then maximizes the profile likelihood over $k$ (a one-dimensional problem) and alternates. Because $\bbeta$ and $k$ are orthogonal parameters (the expected cross-information is zero), the standard errors of $\hat\bbeta$ are unaffected to first order by estimating $k$.</p>
<div class="cols">
<div><h4>Quasi-Poisson weights</h4><p>$W_i\propto\mu_i$. Large counts get proportionally more weight. Variance grows linearly in $\mu$.</p></div>
<div><h4>NB weights</h4><p>$W_i = \mu_i/(1 + \mu_i/k)\to k$ as $\mu_i\to\infty$. Large counts are capped in influence because their variance grows quadratically.</p></div>
</div>
<p>The two can give noticeably different $\hat\bbeta$ when the range of $\mu$ is wide (Ver Hoef & Boveng, 2007): the choice is a substantive claim about how variance scales with the mean, checkable by plotting $(y - \hat\mu)^2$ against $\hat\mu$.</p>`,
        lab: 'overdispersion-mixture',
      },
      {
        title: 'Comparing the remedies',
        html: R`
<p>The lab simulates negative binomial regressions and compares four nominal 95% Wald intervals for the slope: naive Poisson, quasi-Poisson, Poisson with sandwich SEs, and a negative binomial fit. As $\alpha$ grows the naive interval's coverage collapses; quasi-Poisson and sandwich recover most of it; NB does best and is also more <em>efficient</em> (smaller true SD), because its weights match the true variance.</p>
<table>
<thead><tr><th></th><th>Assumes</th><th>Point estimate</th><th>Likelihood?</th></tr></thead>
<tbody>
<tr><td>Quasi-Poisson</td><td>$\Var = \phi\mu$</td><td>= Poisson MLE</td><td>No ($F$-tests, QAIC)</td></tr>
<tr><td>Poisson + sandwich</td><td>Mean only</td><td>= Poisson MLE</td><td>No</td></tr>
<tr><td>Negative binomial</td><td>$\Var = \mu + \mu^2/k$, NB distribution</td><td>Reweighted</td><td>Yes (LR, AIC, prediction)</td></tr>
<tr><td>Poisson GLMM (random effects)</td><td>Lognormal heterogeneity</td><td>Conditional effects</td><td>Yes (integrated)</td></tr>
</tbody>
</table>`,
        lab: 'overdispersion-coverage',
      },
      {
        title: 'Testing, zeros and underdispersion',
        html: R`
<p><strong>Testing for overdispersion.</strong> Testing the Poisson against the NB means testing $H_0:\alpha = 0$, which lies on the boundary of the parameter space $\alpha\ge0$. The LR statistic then has null distribution $\frac12\chi^2_0 + \frac12\chi^2_1$ (Self & Liang, 1987), so the naive $\chi^2_1$ $p$-value should be halved. Score tests need only the Poisson fit; Dean & Lawless (1989) proposed
$$T = \frac{\sum_i\{(y_i - \hat\mu_i)^2 - y_i\}}{\sqrt{2\sum_i\hat\mu_i^2}}\;\approx\;\N(0,1)\ \text{under}\ H_0,$$
whose numerator estimates $\sum_i(\Var Y_i - \mu_i)$ and is large when the variance exceeds the mean quadratically.</p>
<p><strong>Excess zeros.</strong> A zero-inflated model mixes a point mass at zero with a count distribution: $\P(Y=0) = \omega + (1-\omega)f(0)$. A hurdle model uses a binary model for zero versus positive and a zero-truncated count model for the positives. Both are needed when a subpopulation can never produce events; the NB alone often absorbs moderate excess zeros through its larger $f(0)$.</p>
<p><strong>Binomial overdispersion</strong> is handled the same way: quasi-binomial with $\Var = \phi m\pi(1-\pi)$, or the beta-binomial, the binomial analogue of the gamma–Poisson mixture, with $\Var = m\pi(1-\pi)\{1 + (m-1)\rho\}$. Note that binary data ($m = 1$) cannot be overdispersed: a Bernoulli variable's variance is determined by its mean.</p>
<p><strong>Underdispersion</strong> ($\phi < 1$) arises with regular processes or bounded counts; quasi-likelihood handles it, the NB cannot (its variance always exceeds $\mu$), and generalized Poisson or Conway–Maxwell–Poisson models can.</p>`,
      },
    ],
    quiz: [
      { q: R`A Poisson regression has $X^2 = 600$ on $n - p = 200$ degrees of freedom. By what factor should the reported standard errors be inflated under quasi-Poisson, and what happens to the coefficients?`,
        a: R`$\hat\phi = 600/200 = 3$, so SEs are multiplied by $\sqrt3\approx1.73$. The coefficients are unchanged, because the quasi-score equations are identical to the Poisson score equations.` },
      { q: R`Show that for the gamma–Poisson mixture $\Var Y = \mu + \mu^2/k$ without computing the pmf.`,
        a: R`Law of total variance: $\Var Y = \E[\Var(Y\mid\lambda)] + \Var[\E(Y\mid\lambda)] = \E\lambda + \Var\lambda$. A gamma with shape $k$ and mean $\mu$ has variance $\mu^2/k$. So $\Var Y = \mu + \mu^2/k$.` },
      { q: R`Why should the $p$-value from the LR test of Poisson versus negative binomial be halved?`,
        a: R`The null $\alpha = 0$ is on the boundary of $\alpha\ge0$. Under $H_0$ the unconstrained estimate would be negative half the time; constrained to $\hat\alpha = 0$ then, the LR statistic is exactly 0. The other half of the time it behaves like $\chi^2_1$. Hence the null law is $\frac12\chi^2_0 + \frac12\chi^2_1$ and $\P(\Lambda > c) = \frac12\P(\chi^2_1 > c)$.` },
      { q: R`Quasi-Poisson and negative binomial fits to the same data give different slope estimates. Why, and how would you decide between them?`,
        a: R`Their estimating equations weight observations differently: quasi-Poisson weights $\propto\mu_i$, NB weights $\mu_i/(1+\mu_i/k)$, which cap the influence of large counts. If the mean model is correct both are consistent, but they differ in finite samples and, under misspecification, converge to different pseudo-true values. Decide by examining how the squared residuals scale with $\hat\mu$ (linear supports quasi-Poisson, quadratic supports NB), ideally with binned means of $(y-\hat\mu)^2$ against $\hat\mu$.` },
      { q: R`Can ungrouped binary data be overdispersed? Explain.`,
        a: R`No. For $Y\in\{0,1\}$, $\Var Y = \mu(1-\mu)$ exactly; there is no room for extra variation. Apparent overdispersion statistics for binary data reflect mean misspecification or reliance on the uninformative ungrouped deviance. Clustering of binary outcomes shows up instead as correlation, handled by GEE or GLMMs.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'multinomial-ordinal',
    domain: 'glm',
    order: 80,
    title: 'Multinomial & Ordinal Regression',
    subtitle: 'Baseline-category logits, random utility, and the proportional-odds latent variable',
    level: 2,
    sources: { other: 'McCullagh (1980); McFadden (1974); Agresti (2010, 2013 Ch. 8); Train (2009)' },
    prereqs: ['logistic-regression'],
    related: ['poisson-regression', 'glm-framework', 'lda-qda', 'classifier-evaluation'],
    tags: ['softmax', 'multinomial logit', 'proportional odds', 'cumulative logit', 'IIA', 'latent variable'],
    summary: R`For a response with $J > 2$ categories there are two families of models. If the categories are <em>nominal</em>, the baseline-category (multinomial) logit makes each log-odds versus a reference category linear, giving softmax probabilities; it is the multivariate canonical GLM and also the choice model implied by maximizing utilities with Gumbel noise (McFadden). If the categories are <em>ordered</em>, the proportional-odds model treats them as slices of a continuous latent variable $Y^* = \bx^\top\bbeta + \varepsilon$ cut at thresholds $\alpha_1 < \dots < \alpha_{J-1}$; one coefficient vector serves all cut points, which buys parsimony and invariance to merging categories at the cost of a testable "parallel lines" assumption.`,
    survey: {
      task: 'Regression for categorical responses with more than two levels, nominal or ordered',
      assumptions: R`Independent observations; linear logits; nominal: IIA; ordinal: common $\bbeta$ across cut points`,
      output: R`Category probabilities $\hat\pi_j(\bx)$; log odds ratios (per category or common); cut points $\hat\alpha_j$`,
      strengths: 'Multinomial: concave, flexible; ordinal: parsimonious, collapsible across categories, latent-variable interpretation',
      weaknesses: R`Multinomial: $(J-1)p$ parameters, IIA; ordinal: proportional-odds assumption often rejected in large samples`,
      complexity: R`Newton on $(J-1)p$ or $J-1+p$ parameters`,
    },
    sections: [
      {
        title: 'Baseline-category logits',
        html: R`
<p>Let $Y_i\in\{1,\dots,J\}$ with $\pi_{ij} = \P(Y_i = j\mid\bx_i)$. With category 1 as baseline, the multinomial logit model is
$$\log\frac{\pi_{ij}}{\pi_{i1}} = \bx_i^\top\bbeta_j,\quad j = 2,\dots,J,\qquad\text{equivalently}\qquad \pi_{ij} = \frac{e^{\bx_i^\top\bbeta_j}}{\sum_{k=1}^J e^{\bx_i^\top\bbeta_k}},\ \bbeta_1 = \mathbf 0.$$
The constraint $\bbeta_1 = \mathbf 0$ is needed because adding the same vector to every $\bbeta_k$ leaves all probabilities unchanged. Any pair of categories has linear log-odds: $\log(\pi_j/\pi_k) = \bx^\top(\bbeta_j - \bbeta_k)$, so the choice of baseline is cosmetic.</p>
<p>The multinomial distribution with one trial is a $(J-1)$-dimensional exponential family with natural parameters $\theta_j = \log(\pi_j/\pi_1)$, so this is its canonical-link GLM. Consequently, with indicator responses $y_{ij}$:</p>
<ul>
<li>the log-likelihood $\ell = \sum_i\big\{\sum_j y_{ij}\bx_i^\top\bbeta_j - \log\sum_k e^{\bx_i^\top\bbeta_k}\big\}$ is concave in $(\bbeta_2,\dots,\bbeta_J)$ (log-sum-exp is convex);</li>
<li>the score equations are $\sum_i(y_{ij} - \hat\pi_{ij})\bx_i = \mathbf 0$ for each $j$: fitted category counts match observed ones within every covariate pattern spanned by $X$;</li>
<li>the Hessian has blocks $-\sum_i\pi_{ij}(\delta_{jk} - \pi_{ik})\bx_i\bx_i^\top$, free of $\by$, so Newton = Fisher scoring;</li>
<li>by the <a href="#/t/poisson-regression">Poisson–multinomial equivalence</a>, the same estimates come from a Poisson log-linear model with a nuisance parameter for each observation.</li>
</ul>
<p>Separation has a multinomial version: if some category can be perfectly split from another by a hyperplane, the corresponding $\bbeta_j - \bbeta_k$ diverges.</p>`,
      },
      {
        title: 'Random utility and the IIA property',
        html: R`
<p>Economists derive the same model from choice behaviour. Agent $i$ assigns utility $U_{ij} = v_{ij} + \varepsilon_{ij}$ to alternative $j$, with systematic part $v_{ij} = \bx_i^\top\bbeta_j$ (or $\mathbf z_{ij}^\top\boldsymbol\gamma$ for alternative-specific covariates), and chooses the maximum.</p>
<div class="thm" data-kind="Theorem" data-name="Gumbel noise gives softmax choice (McFadden, 1974)">
<p>If $\varepsilon_{i1},\dots,\varepsilon_{iJ}$ are IID standard Gumbel, with CDF $F(e) = \exp(-e^{-e})$, then
$$\P(Y_i = j) = \P\big(U_{ij} > U_{ik}\ \forall k\neq j\big) = \frac{e^{v_{ij}}}{\sum_k e^{v_{ik}}}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Condition on $\varepsilon_{ij} = e$. Alternative $j$ wins iff $\varepsilon_{ik} < e + v_{ij} - v_{ik}$ for all $k\neq j$, which has probability $\prod_{k\neq j}\exp(-e^{-(e + v_{ij} - v_{ik})})$. The density of $\varepsilon_{ij}$ is $f(e) = e^{-e}\exp(-e^{-e})$, which is the factor for $k = j$ times $e^{-e}$. So
$$\P(Y_i=j) = \int_{-\infty}^\infty e^{-e}\exp\!\Big(-e^{-e}\sum_k e^{v_{ik} - v_{ij}}\Big)\d e.$$
Substitute $t = e^{-e}$, $\d t = -e^{-e}\d e$, and write $S = \sum_k e^{v_{ik} - v_{ij}}$: the integral becomes $\int_0^\infty e^{-tS}\d t = 1/S = e^{v_{ij}}/\sum_k e^{v_{ik}}$.</p>
</details>
<p>The model inherits a strong property: <strong>independence of irrelevant alternatives (IIA)</strong>. The ratio $\pi_j/\pi_k = e^{v_j - v_k}$ does not depend on what other alternatives exist. In the classic red-bus/blue-bus example, a traveller indifferent between car and a red bus ($\pi = 1/2$ each) faces a new blue bus identical to the red one. Common sense says car stays at $1/2$ and the buses split the rest; IIA forces $1/3$ each, because the model treats the two buses' errors as independent. Nested logit, multinomial probit (correlated normal errors) and mixed logit (random coefficients) relax IIA. In the regression setting with individual-specific covariates and a fixed category set, IIA is less of a practical concern than in choice modelling.</p>`,
      },
      {
        title: 'Ordinal responses: the proportional-odds model',
        html: R`
<p>Ordered categories (disease severity, Likert items, income bands) carry information the multinomial model wastes. The <strong>cumulative logit</strong> or <strong>proportional-odds</strong> model (McCullagh, 1980) models cumulative probabilities:
$$\operatorname{logit}\P(Y_i\le j\mid\bx_i) = \alpha_j - \bx_i^\top\bbeta,\qquad j = 1,\dots,J-1,\qquad \alpha_1 < \alpha_2 < \dots < \alpha_{J-1}.$$
Category probabilities are differences: $\pi_{ij} = F(\alpha_j - \bx_i^\top\bbeta) - F(\alpha_{j-1} - \bx_i^\top\bbeta)$ with $F$ the logistic CDF, $\alpha_0 = -\infty$, $\alpha_J = +\infty$. The minus sign makes positive $\beta$ shift mass toward higher categories (software differs on this convention).</p>
<div class="thm" data-kind="Proposition" data-name="Latent-variable derivation">
<p>Let $Y_i^* = \bx_i^\top\bbeta + \varepsilon_i$ with $\varepsilon_i$ IID with CDF $F$, and observe $Y_i = j$ iff $\alpha_{j-1} < Y_i^*\le\alpha_j$. Then $\P(Y_i\le j\mid\bx_i) = F(\alpha_j - \bx_i^\top\bbeta)$. With logistic $F$ this is the proportional-odds model; with $F = \Phi$, the ordered probit; with $F(t) = 1 - \exp(-e^t)$, the grouped proportional-hazards model.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\P(Y_i\le j) = \P(Y_i^*\le\alpha_j) = \P(\varepsilon_i\le\alpha_j - \bx_i^\top\bbeta) = F(\alpha_j - \bx_i^\top\bbeta)$. The thresholds are increasing so the events are nested.</p>
</details>
<p>Consequences:</p>
<ul>
<li><strong>Proportional odds.</strong> For covariate values $\bx$ and $\bx'$, the cumulative odds ratio $\frac{\text{odds}(Y\le j\mid\bx)}{\text{odds}(Y\le j\mid\bx')} = e^{-(\bx - \bx')^\top\bbeta}$ is the same for every $j$. One number summarizes the effect on the whole ordinal scale.</li>
<li><strong>Invariance to merging categories.</strong> Combining adjacent categories deletes one threshold and leaves $\bbeta$ unchanged — the latent variable is the same, only cut more coarsely. A study using a 5-point scale and one using a 3-point scale estimate the same $\bbeta$ (unlike the multinomial model, where $\bbeta_j$ depend on which categories exist).</li>
<li><strong>Parameter count</strong> is $J - 1 + p$ rather than $(J-1)(p+1)$.</li>
<li><strong>Scale.</strong> As for binary logit, $\bbeta$ is identified only relative to the latent error scale, and the intercept is absorbed into the thresholds.</li>
</ul>
<p>The likelihood is not that of a canonical GLM, but it is still concave in $(\boldsymbol\alpha,\bbeta)$ for log-concave $F$ (logistic, normal, extreme value) (Pratt, 1981), and the MLE is found by Newton's method subject to ordering of the thresholds, which is automatically satisfied at an interior optimum.</p>`,
        lab: 'multinomial-ordinal-cutpoints',
      },
      {
        title: 'Checking proportional odds, and alternatives',
        html: R`
<p>The assumption is that the $J-1$ binary logistic regressions "$Y\le j$ vs $Y > j$" share a slope. A natural alternative frees the slopes:
$$\operatorname{logit}\P(Y\le j\mid\bx) = \alpha_j - \bx^\top\bbeta_j.$$
The proportional-odds model is nested in it, so a likelihood-ratio test with $(J-2)p$ degrees of freedom checks parallelism (score and Brant-type Wald tests are alternatives). The non-parallel model has a defect: the cumulative logit lines can cross, giving negative category probabilities for some $\bx$, so it is usually fitted only within the data range or used as a diagnostic.</p>
<p>The lab simulates ordinal data from a model whose two slopes differ by $\delta$, fits both models, and reports the LR test. With $\delta = 0$ the test rejects about 5% of the time; as $\delta$ grows, and especially as $n$ grows, it rejects. With large $n$ even a trivially small departure is detected — a reason to judge the <em>size</em> of the departure (do fitted probabilities differ materially?) and not just the $p$-value.</p>
<table>
<thead><tr><th>Model</th><th>Logit of</th><th>Natural when</th></tr></thead>
<tbody>
<tr><td>Cumulative (proportional odds)</td><td>$\P(Y\le j)$</td><td>Discretized latent continuum</td></tr>
<tr><td>Partial proportional odds</td><td>$\P(Y\le j)$ with some $\beta$'s category-specific</td><td>A few covariates violate parallelism</td></tr>
<tr><td>Adjacent categories</td><td>$\pi_{j+1}/\pi_j$</td><td>Local comparisons; a constrained multinomial logit</td></tr>
<tr><td>Continuation ratio</td><td>$\P(Y = j\mid Y\ge j)$</td><td>Sequential stages (progression through grades); discrete-time survival</td></tr>
<tr><td>Multinomial logit</td><td>$\pi_j/\pi_1$</td><td>Ordering doubtful or irrelevant</td></tr>
</tbody>
</table>`,
        lab: 'multinomial-ordinal-potest',
      },
      {
        title: 'Worked interpretation and pitfalls',
        html: R`
<div class="ex" data-name="Reading a proportional-odds fit">
<p>A pain score (none, mild, moderate, severe) is regressed on treatment. Suppose $\hat\beta_{\text{trt}} = -0.8$ in the convention $\operatorname{logit}\P(Y\le j) = \alpha_j - \beta x$. For every cut, the treated have odds of being at or <em>below</em> category $j$ multiplied by $e^{0.8}\approx2.23$ relative to controls: treatment shifts the latent pain distribution down by $0.8$ logistic units (about $0.8/1.81\approx0.44$ latent SDs). With thresholds $\hat{\boldsymbol\alpha} = (-1, 0.5, 2)$, a control has $\P(\text{none}) = \operatorname{logit}^{-1}(-1) = 0.27$, while a treated patient has $\operatorname{logit}^{-1}(-1 + 0.8) = 0.45$.</p>
</div>
<div class="warn" data-name="Pitfalls">
<ul>
<li><strong>Sign conventions</strong> differ (R's <code>MASS::polr</code> and <code>ordinal::clm</code> use $\alpha_j - \bx^\top\bbeta$; others use $\alpha_j + \bx^\top\bbeta$). Always check which direction "positive" means.</li>
<li><strong>Treating ordinal scores as interval data</strong> (OLS on 1, 2, 3, 4) assumes equal spacing, can produce effects of the wrong sign when the latent variances differ between groups, and ignores floor and ceiling effects (Liddell & Kruschke, 2018).</li>
<li><strong>Sparse categories</strong> lead to unstable thresholds and separation; merging (which proportional odds tolerates) is a legitimate remedy.</li>
<li><strong>Heteroscedastic latent scales</strong>: if the latent error SD differs between groups, slopes and proportional odds are both distorted; location-scale ordinal models address this.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`In a multinomial logit model with $J = 3$, why is it harmless to choose category 3 instead of category 1 as baseline?`,
        a: R`The model only specifies differences $\bbeta_j - \bbeta_k$; changing the baseline reparameterizes $\bbeta_j\mapsto\bbeta_j - \bbeta_3$. The fitted probabilities, likelihood and all pairwise log-odds are identical; only which differences are reported as coefficients changes.` },
      { q: R`Explain how IIA follows from the softmax form, and give a setting where it is implausible.`,
        a: R`$\pi_j/\pi_k = e^{v_j - v_k}$ depends only on $j$ and $k$, so adding or removing other alternatives cannot change their relative odds. It is implausible when alternatives are close substitutes (red bus/blue bus): introducing a near-copy of one option should draw share mostly from that option, not proportionally from all.` },
      { q: R`Why does merging two adjacent categories leave the proportional-odds coefficients unchanged?`,
        a: R`In the latent-variable formulation the categories are intervals of $Y^* = \bx^\top\bbeta + \varepsilon$ between thresholds. Merging adjacent categories simply removes the threshold between them; the latent model, and hence $\bbeta$, is untouched. The remaining cumulative logits $\alpha_j - \bx^\top\bbeta$ are a subset of the original ones.` },
      { q: R`How many extra parameters does the non-parallel cumulative logit model have compared with proportional odds, for $J$ categories and $p$ covariates (excluding the intercept)? What is a drawback of fitting it?`,
        a: R`It has $(J-1)p$ slope parameters instead of $p$, so $(J-2)p$ extra; that is the df of the LR test of proportional odds. Its drawback is that the cumulative probability curves can cross, producing negative fitted category probabilities for some covariate values, so the model is not globally valid.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'glmm-gee',
    domain: 'glm',
    order: 90,
    title: 'Correlated Responses: GLMMs & GEE',
    subtitle: 'Subject-specific versus population-averaged effects, and why they differ for nonlinear links',
    level: 3,
    sources: { other: 'Liang & Zeger (1986); Zeger, Liang & Albert (1988); Breslow & Clayton (1993); Diggle, Heagerty, Liang & Zeger (2002); Pepe & Anderson (1994)' },
    prereqs: ['logistic-regression', 'mixed-models', 'overdispersion'],
    related: ['hierarchical-mcmc', 'mixed-models', 'irls', 'block-bootstrap'],
    tags: ['random effects', 'marginal model', 'generalized estimating equations', 'sandwich', 'attenuation', 'longitudinal data'],
    summary: R`Longitudinal and clustered data violate the GLM's independence assumption. Two strategies dominate. A <em>generalized linear mixed model</em> adds random effects inside the link, $g(\E[y_{ij}\mid\mathbf b_i]) = \bx_{ij}^\top\bbeta + \mathbf z_{ij}^\top\mathbf b_i$, giving a full likelihood and <em>subject-specific</em> coefficients. <em>Generalized estimating equations</em> model only the marginal mean $g(\E y_{ij}) = \bx_{ij}^\top\bbeta^{M}$, posit a working correlation, and obtain valid standard errors from a cluster-level sandwich even when that correlation is wrong. For the identity link the two $\bbeta$'s coincide; for the log link only intercepts differ; for the logit the marginal coefficients are <em>attenuated</em> toward zero by roughly $1/\sqrt{1 + 0.346\,\sigma_b^2}$. Choosing between them is choosing the estimand.`,
    survey: {
      task: 'Regression for clustered, longitudinal or repeated binary/count outcomes',
      assumptions: R`GLMM: correct conditional model and random-effect distribution; GEE: correct marginal mean, many clusters, MCAR missingness`,
      output: R`GLMM: $\hat\bbeta$ (conditional), $\hat\sigma_b^2$, predicted random effects; GEE: $\hat\bbeta^M$ with sandwich SEs`,
      strengths: 'GLMM: likelihood, MAR-valid, individual predictions; GEE: robust to correlation misspecification, simple, targets population averages',
      weaknesses: 'GLMM: intractable integrals, sensitive to random-effect assumptions; GEE: needs many clusters, no likelihood, MCAR',
      complexity: R`GLMM: quadrature or Laplace per cluster per iteration; GEE: IRLS with $m_i\times m_i$ blocks`,
    },
    sections: [
      {
        title: 'Two questions about clustered data',
        html: R`
<p>Data come in clusters $i = 1,\dots,K$ (patients, schools, litters) with observations $j = 1,\dots,m_i$ inside each. Observations within a cluster are correlated because they share unmeasured cluster characteristics. Ignoring this has the same consequence as <a href="#/t/overdispersion">overdispersion</a>: consistent point estimates (if the mean is right) with wrong standard errors — too small for cluster-level covariates, possibly too large for within-cluster covariates.</p>
<p>But a nonlinear link raises a deeper issue. There are two different regression questions:</p>
<div class="cols">
<div><h4>Subject-specific (conditional)</h4><p>How does the probability of an event change for <em>a given</em> individual if their $x$ changes? $g(\E[y_{ij}\mid\mathbf b_i,\bx_{ij}]) = \bx_{ij}^\top\bbeta^{S}$. Natural for clinical decisions about a patient.</p></div>
<div><h4>Population-averaged (marginal)</h4><p>How does the proportion of events in the <em>population</em> change between groups with different $x$? $g(\E[y_{ij}\mid\bx_{ij}]) = \bx_{ij}^\top\bbeta^{M}$. Natural for public-health and policy comparisons.</p></div>
</div>
<p>For linear models these coincide. For logistic and probit models they do not, even when both models are correct descriptions of the same data-generating process.</p>`,
      },
      {
        title: 'Generalized linear mixed models',
        html: R`
<div class="def" data-name="GLMM">
<p>Conditionally on random effects $\mathbf b_i\sim\N(\mathbf 0,\bSigma_b)$, the $y_{ij}$ are independent with an EDF distribution and $g(\E[y_{ij}\mid\mathbf b_i]) = \bx_{ij}^\top\bbeta + \mathbf z_{ij}^\top\mathbf b_i$.</p>
</div>
<p>The likelihood integrates the random effects out, one cluster at a time:
$$L(\bbeta,\bSigma_b) = \prod_{i=1}^K\int\prod_{j=1}^{m_i}f(y_{ij}\mid\mathbf b;\bbeta)\;\varphi(\mathbf b;\mathbf 0,\bSigma_b)\d\mathbf b.$$
Except for the Normal–identity case (the <a href="#/t/mixed-models">linear mixed model</a>) this integral has no closed form. Approaches:</p>
<ul>
<li><strong>Laplace approximation</strong>: expand the log-integrand to second order about its mode $\hat{\mathbf b}_i$; accurate when clusters are large.</li>
<li><strong>Adaptive Gauss–Hermite quadrature</strong>: quadrature centred and scaled at the mode; the gold standard for one or two random effects per cluster.</li>
<li><strong>Penalized quasi-likelihood (PQL)</strong> (Breslow & Clayton, 1993): linearize as in IRLS and fit a linear mixed model to the working response. Fast, but variance components and fixed effects are biased toward zero for binary data with small clusters.</li>
<li><strong>MCMC</strong> (<a href="#/t/hierarchical-mcmc">hierarchical models</a>): exact up to Monte Carlo error, with priors on $\bSigma_b$.</li>
</ul>
<p>Testing $\sigma_b^2 = 0$ is again a boundary problem (mixture $\chi^2$ null distribution), and the random-effects distribution is an assumption: misspecifying it biases the intercept and variance component, though slopes of within-cluster covariates are usually fairly robust (Neuhaus, McCulloch & Boylan, 2013).</p>`,
      },
      {
        title: 'Marginal versus conditional effects: attenuation',
        html: R`
<p>Integrate a random-intercept GLMM over $b\sim\N(0,\sigma^2)$ to get the implied marginal mean:
$$\E(y\mid\bx) = \int g^{-1}(\bx^\top\bbeta^S + b)\,\varphi(b;0,\sigma^2)\d b.$$</p>
<div class="thm" data-kind="Proposition" data-name="Marginalizing over a random intercept">
<ol>
<li><strong>Identity link:</strong> $\E(y\mid\bx) = \bx^\top\bbeta^S$; marginal and conditional coefficients agree.</li>
<li><strong>Log link:</strong> $\E(y\mid\bx) = \exp(\bx^\top\bbeta^S + \sigma^2/2)$; all slopes agree, the intercept shifts by $\sigma^2/2$.</li>
<li><strong>Probit link:</strong> $\E(y\mid\bx) = \Phi\big(\bx^\top\bbeta^S/\sqrt{1+\sigma^2}\big)$ exactly; marginal coefficients are $\bbeta^M = \bbeta^S/\sqrt{1+\sigma^2}$.</li>
<li><strong>Logit link:</strong> approximately $\E(y\mid\bx)\approx\operatorname{logit}^{-1}\big(\bx^\top\bbeta^S/\sqrt{1 + c^2\sigma^2}\big)$ with $c = 16\sqrt3/(15\pi)\approx0.588$, $c^2\approx0.346$ (Zeger, Liang & Albert, 1988).</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) Linearity of expectation with $\E b = 0$. (2) $\E e^{b} = e^{\sigma^2/2}$ (lognormal mean / normal MGF), so $\E\exp(\eta + b) = e^{\eta}e^{\sigma^2/2}$. (3) Let $Z\sim\N(0,1)$ independent of $b$. Then $\Phi(\eta + b) = \P(Z\le\eta + b\mid b)$, so $\E\Phi(\eta + b) = \P(Z - b\le\eta) = \Phi(\eta/\sqrt{1+\sigma^2})$ because $Z - b\sim\N(0,1+\sigma^2)$. (4) Approximate the logistic CDF by a probit with matched scale, $\operatorname{logit}^{-1}(t)\approx\Phi(ct)$; then by (3), $\E\Phi(c(\eta + b)) = \Phi\big(c\eta/\sqrt{1 + c^2\sigma^2}\big)\approx\operatorname{logit}^{-1}\big(\eta/\sqrt{1 + c^2\sigma^2}\big)$. The constant $c = 16\sqrt3/(15\pi)\approx 1/1.70$ is the familiar scale that makes $\Phi(ct)$ a close approximation to $\operatorname{logit}^{-1}(t)$; the approximation error is below about 0.01 uniformly in $t$, which is why the marginal curve is only approximately logistic.</p>
</details>
<div class="ex" data-name="How big is the gap?">
<p>With a conditional odds ratio of 2 ($\beta^S = 0.693$) and random-intercept SD $\sigma = 2$, the marginal coefficient is about $0.693/\sqrt{1 + 0.346\cdot4} = 0.693/1.544 = 0.449$: a marginal odds ratio of $1.57$. Neither number is "wrong": 2 is the odds ratio for one person whose $x$ changes; 1.57 compares the population proportions at two values of $x$, averaging over people with very different baseline risks.</p>
</div>
<p>The mechanism is averaging: a mixture of steep logistic curves with shifted centres is a shallower curve. It is the same mechanism as the non-collapsibility of the odds ratio described in <a href="#/t/logistic-regression">logistic regression</a>, and it operates whenever there is unmodelled heterogeneity, clustered or not. The lab draws conditional curves for several clusters together with the exact marginal curve.</p>`,
        lab: 'glmm-gee-attenuation',
      },
      {
        title: 'Generalized estimating equations',
        html: R`
<p>Liang and Zeger (1986) extended quasi-likelihood to clustered data. Specify only the marginal mean $\mu_{ij} = g^{-1}(\bx_{ij}^\top\bbeta)$, the variance function $\Var y_{ij} = \phi V(\mu_{ij})$, and a <em>working correlation</em> matrix $R_i(\alpha)$. Let $A_i = \diag\{V(\mu_{ij})\}$, $\mathbf V_i = \phi A_i^{1/2}R_i(\alpha)A_i^{1/2}$ and $D_i = \partial\bmu_i/\partial\bbeta^\top$ ($m_i\times p$). The GEE estimator solves
$$U(\bbeta) = \sum_{i=1}^K D_i^\top\mathbf V_i^{-1}(\by_i - \bmu_i) = \mathbf 0,$$
with $\alpha$ and $\phi$ estimated from Pearson residuals between updates (e.g. exchangeable: $\hat\alpha$ = average product of residual pairs within clusters divided by $\hat\phi$).</p>
<div class="thm" data-kind="Theorem" data-name="Robustness to the working correlation (Liang & Zeger, 1986)">
<p>Suppose the marginal mean model is correct, clusters are independent, $K\to\infty$ with bounded $m_i$, and $\hat\alpha$ is $\sqrt K$-consistent for some $\alpha^*$ (not necessarily the truth) given $\bbeta$, with $\hat\phi$ similarly behaved. Then $\hat\bbeta$ is consistent and
$$\sqrt K(\hat\bbeta - \bbeta)\dto\N\big(\mathbf 0,\;\lim K\,H^{-1}MH^{-1}\big),\qquad H = \sum_i D_i^\top\mathbf V_i^{-1}D_i,\quad M = \sum_i D_i^\top\mathbf V_i^{-1}\Cov(\by_i)\mathbf V_i^{-1}D_i,$$
estimated by the sandwich with $\Cov(\by_i)$ replaced by $(\by_i - \hat\bmu_i)(\by_i - \hat\bmu_i)^\top$. If $R_i(\alpha^*)$ is the true correlation, $M = H$ and the estimator is optimal among estimators based on these estimating equations.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Each cluster's contribution has mean zero at the true $\bbeta$ for <em>any</em> fixed weight matrix, because $\E(\by_i - \bmu_i) = \mathbf 0$; the working correlation only changes efficiency. Expanding around the truth, $\hat\bbeta - \bbeta\approx H^{-1}\sum_i D_i^\top\mathbf V_i^{-1}(\by_i - \bmu_i)$, a sum of $K$ independent mean-zero terms; apply the CLT across clusters. The plug-in $\hat\alpha$ has no first-order effect because $\partial U/\partial\alpha$ is a sum of terms proportional to $(\by_i - \bmu_i)$, hence has mean zero and is $O_p(\sqrt K)$, while $\hat\alpha - \alpha^* = O_p(K^{-1/2})$; their product is $O_p(1)$, negligible against the $O_p(\sqrt K)$ score. Optimality when $R$ is correct follows exactly as in the quasi-score optimality theorem, with cluster-level vectors in place of scalars.</p>
</details>
<p>Common working correlations: independence ($R = I$: an ordinary GLM fit with a cluster-robust sandwich), exchangeable ($\alpha$ for every pair), AR(1) ($\alpha^{|j-k|}$), and unstructured. Exchangeable or AR(1) gains efficiency for within-cluster covariates when the correlation is substantial; for cluster-level covariates the gain is small, and with equal cluster sizes and a purely cluster-level covariate the exchangeable and independence estimates coincide exactly (every observation in a cluster then receives the same weight). The lab compares naive GLM standard errors, independence GEE and exchangeable GEE in repeated simulations.</p>`,
        lab: 'glmm-gee-sandwich',
      },
      {
        title: 'Caveats for GEE',
        html: R`
<div class="warn" data-name="Few clusters">
<p>The theory is asymptotic in the number of clusters $K$, not observations. With $K < 40$ or so the sandwich is biased downward; use small-sample corrections (Mancl & DeRouen, 2001; Kauermann & Carroll, 2001) and $t_{K-p}$ reference distributions.</p>
</div>
<div class="warn" data-name="Time-varying covariates and non-diagonal working correlation">
<p>The estimating equation is unbiased only if $\E(y_{ij}\mid\bx_{i1},\dots,\bx_{im_i}) = \E(y_{ij}\mid\bx_{ij})$ — the "full covariate conditional mean" assumption (Pepe & Anderson, 1994). With a non-independence working correlation, residuals at time $k$ are weighted into the equation for time $j$; if past outcomes affect future covariates (feedback), this introduces bias. Independence working correlation is safe in that setting.</p>
</div>
<div class="warn" data-name="Missing data">
<p>GEE is valid under missing completely at random. Under missing at random (dropout depending on observed past outcomes) it is biased unless inverse-probability weighted (Robins, Rotnitzky & Zhao, 1995). Likelihood-based GLMMs remain valid under MAR, a substantial practical advantage.</p>
</div>`,
      },
      {
        title: 'Choosing between them',
        html: R`
<table>
<thead><tr><th></th><th>GLMM</th><th>GEE</th></tr></thead>
<tbody>
<tr><td>Estimand</td><td>Subject-specific $\bbeta^S$</td><td>Population-averaged $\bbeta^M$</td></tr>
<tr><td>Assumptions</td><td>Full conditional distribution + random-effect law</td><td>Marginal mean; working correlation may be wrong</td></tr>
<tr><td>Inference</td><td>Likelihood: LR tests, AIC, profile intervals</td><td>Wald with sandwich; no likelihood</td></tr>
<tr><td>Missing data</td><td>Valid under MAR</td><td>MCAR (or weighted GEE)</td></tr>
<tr><td>Needs</td><td>Numerical integration</td><td>Many clusters</td></tr>
<tr><td>Predictions for a cluster</td><td>Yes (BLUP-like $\hat{\mathbf b}_i$)</td><td>No</td></tr>
</tbody>
</table>
<p>The deciding question is the estimand. If the scientific statement is about what happens to an individual when their covariate changes (a within-person treatment effect), the GLMM targets it. If it is about how the prevalence differs between subpopulations, the marginal model targets it directly and more robustly. A GLMM can also produce marginal effects by numerical integration of its fitted conditional model, provided the random-effect distribution is right.</p>`,
      },
    ],
    quiz: [
      { q: R`For a random-intercept probit model with conditional slope 1.0 and $\sigma_b = 1$, what is the marginal slope? Why is the result exact?`,
        a: R`$1/\sqrt{1 + 1} \approx 0.707$. Exact because $\E_b\Phi(\eta + b) = \P(Z - b\le\eta)$ with $Z - b\sim\N(0, 1+\sigma^2)$, which is again a probit curve with scaled argument.` },
      { q: R`Why do marginal and conditional slopes coincide for a log-link Poisson GLMM with a random intercept, but not for a logistic GLMM?`,
        a: R`With the log link the random intercept enters multiplicatively: $\E e^{\eta + b} = e^{\eta}\E e^b$, so integrating only rescales the mean by a constant, absorbed into the intercept. The inverse logit is not multiplicative; averaging $\operatorname{logit}^{-1}(\eta + b)$ over $b$ flattens the curve, shrinking the slope.` },
      { q: R`State the key reason GEE point estimates are consistent even if the working correlation is wrong.`,
        a: R`The estimating function $\sum_i D_i^\top\mathbf V_i^{-1}(\by_i - \bmu_i)$ has mean zero at the true $\bbeta$ for any fixed weight matrix $\mathbf V_i^{-1}$, because only the marginal mean enters through $\E(\by_i - \bmu_i) = \mathbf 0$. The working correlation affects efficiency and the variance (handled by the sandwich), not unbiasedness.` },
      { q: R`A trial randomizes 12 clinics (cluster-level treatment) with 200 patients each. A colleague fits an ordinary logistic regression. What goes wrong, and is GEE with a sandwich the full fix?`,
        a: R`The naive SEs treat 2,400 patients as independent and are far too small for a clinic-level treatment; the effective sample size is closer to the number of clinics. GEE with a sandwich corrects the form of the variance, but with only 12 clusters the sandwich is itself biased downward and noisy, so small-sample corrections and $t_{10}$ references (or a GLMM, or cluster-level analysis) are needed.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'gam',
    domain: 'glm',
    order: 100,
    title: 'Generalized Additive Models',
    subtitle: 'Smooth functions in the linear predictor, fitted by backfitting or penalized IRLS',
    level: 3,
    sources: { aos: 'Ch. 20', other: 'Hastie & Tibshirani (1990); Buja, Hastie & Tibshirani (1989); Eilers & Marx (1996); Wood (2017)' },
    prereqs: ['splines', 'irls'],
    related: ['kernel-regression', 'model-selection', 'ridge-regression', 'cross-validation', 'mixed-models'],
    tags: ['additive model', 'backfitting', 'P-splines', 'penalized likelihood', 'GCV', 'REML', 'effective degrees of freedom'],
    summary: R`A generalized additive model replaces each linear term of a GLM with an unknown smooth function: $g(\mu_i) = \beta_0 + \sum_j f_j(x_{ij})$. Additivity sidesteps the curse of dimensionality — each $f_j$ is estimated at a one-dimensional rate — while keeping the interpretability of separate covariate effects. Classical fitting uses <em>backfitting</em>, a Gauss–Seidel sweep that smooths partial residuals against one covariate at a time, nested inside IRLS for non-Gaussian responses. Modern fitting represents each $f_j$ by a spline basis with a roughness penalty and maximizes a penalized likelihood by penalized IRLS, with smoothing parameters chosen by GCV, UBRE/AIC or REML.`,
    survey: {
      task: 'Flexible regression for any GLM family with interpretable per-covariate effects',
      assumptions: R`Additive structure on the link scale (plus chosen interactions); smooth $f_j$; GLM family assumptions`,
      output: R`Estimated curves $\hat f_j$ with pointwise bands; effective degrees of freedom; smoothing parameters`,
      strengths: 'One-dimensional convergence rates; visual, interpretable; automatic smoothness selection; reuses GLM machinery',
      weaknesses: 'Misses interactions unless specified; concurvity destabilizes estimates; inference is approximate (conditional on $\\lambda$)',
      complexity: R`P-IRLS: $O(nq^2)$ per iteration for $q$ total basis functions; backfitting: $O(n)$–$O(n\log n)$ per smoother per sweep`,
    },
    sections: [
      {
        title: 'From linear predictors to additive predictors',
        html: R`
<p>A GLM forces $\eta = \bx^\top\bbeta$. If the true effect of age on log-odds is U-shaped, no single coefficient captures it. Fully nonparametric regression on $d$ covariates escapes this but pays the curse of dimensionality: for $s$-times-differentiable functions the optimal mean-squared error rate is $n^{-2s/(2s+d)}$ (Stone, 1982), hopeless for $d$ beyond 3 or 4. The additive model
$$g(\mu_i) = \beta_0 + f_1(x_{i1}) + \dots + f_d(x_{id})$$
is a compromise. Stone (1985) proved that if the truth is additive, each $f_j$ can be estimated at the <em>one-dimensional</em> rate $n^{-2s/(2s+1)}$, whatever $d$. Additivity is an assumption, but a checkable one, and selected interactions $f_{jk}(x_j,x_k)$ (tensor-product smooths) can be added where needed.</p>
<p><strong>Identifiability.</strong> Constants can be shifted between $\beta_0$ and the $f_j$, so each function is centred: $\sum_i f_j(x_{ij}) = 0$. Each $\hat f_j$ is then interpreted as the effect of $x_j$ on the link scale relative to its average, holding the other terms fixed.</p>`,
      },
      {
        title: 'Backfitting',
        html: R`
<p>For the Gaussian additive model $y_i = \beta_0 + \sum_j f_j(x_{ij}) + \varepsilon_i$, suppose each $f_j$ is to be estimated by a linear smoother $S_j$ (an $n\times n$ matrix: smoothing spline, P-spline, local linear, running mean). If the other functions were known, we would smooth the <em>partial residual</em> $\by - \beta_0 - \sum_{k\neq j}\mathbf f_k$ against $x_j$. Backfitting does exactly this, cyclically.</p>
<div class="algo" data-name="Backfitting (Friedman & Stuetzle, 1981; Buja, Hastie & Tibshirani, 1989)">
<ol>
<li>Set $\hat\beta_0 = \bar y$ and $\hat{\mathbf f}_j = \mathbf 0$ for all $j$.</li>
<li>Cycle over $j = 1,\dots,d$: $\hat{\mathbf f}_j\leftarrow S_j\big(\by - \hat\beta_0 - \sum_{k\neq j}\hat{\mathbf f}_k\big)$, then centre $\hat{\mathbf f}_j$.</li>
<li>Repeat until the functions change by less than a tolerance.</li>
</ol>
</div>
<p>At a fixed point the functions satisfy the "normal equations" $\mathbf f_j = S_j(\by - \beta_0 - \sum_{k\neq j}\mathbf f_k)$ for all $j$, a linear system of size $nd$; backfitting is its Gauss–Seidel solution, using only the $O(n)$ action of each smoother.</p>
<div class="thm" data-kind="Theorem" data-name="Convergence of backfitting (two smoothers)">
<p>Let $S_1,S_2$ be symmetric with eigenvalues in $[0,1]$ and $\|S_1S_2\| < 1$ (no exact concurvity). Then backfitting converges from any start to the unique solution of the normal equations, geometrically with ratio at most $\|S_1S_2\|$ per sweep. If $S_1,S_2$ are orthogonal projections onto subspaces meeting only at $\mathbf 0$, the solution is the least-squares fit onto $\mathcal L_1 + \mathcal L_2$ and the error contracts per sweep by $\cos^2\gamma$, where $\gamma$ is the angle between the subspaces.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Let $(\mathbf f_1^*,\mathbf f_2^*)$ solve the normal equations and write errors $\mathbf e_j^{(t)} = \hat{\mathbf f}_j^{(t)} - \mathbf f_j^*$. Subtracting the fixed-point equations from the updates $\hat{\mathbf f}_1\leftarrow S_1(\by - \hat{\mathbf f}_2)$, $\hat{\mathbf f}_2\leftarrow S_2(\by - \hat{\mathbf f}_1)$ gives $\mathbf e_1^{(t+1)} = -S_1\mathbf e_2^{(t)}$ and $\mathbf e_2^{(t+1)} = -S_2\mathbf e_1^{(t+1)} = S_2S_1\mathbf e_2^{(t)}$. So $\mathbf e_2^{(t)} = (S_2S_1)^t\mathbf e_2^{(0)}\to\mathbf 0$ geometrically when $\|S_2S_1\| = \|S_1S_2\| < 1$, and then $\mathbf e_1^{(t)}\to\mathbf 0$. Existence and uniqueness of the fixed point follow from invertibility of $I - S_2S_1$. For projections, $\|P_2P_1P_2\| = \cos^2\gamma$ (von Neumann's alternating-projection theorem with Aronszajn's rate); the fixed point satisfies $P_1(\by - \mathbf f_1 - \mathbf f_2) = 0$ and $P_2(\by - \mathbf f_1 - \mathbf f_2) = 0$, the least-squares normal equations for $\mathcal L_1 + \mathcal L_2$.</p>
</details>
<p>The rate formula has a statistical meaning. When $x_1$ and $x_2$ are strongly correlated, functions of $x_1$ are nearly expressible as functions of $x_2$ — <strong>concurvity</strong>, the nonparametric analogue of collinearity. Then $\|S_1S_2\|$ is near 1, backfitting crawls, and the individual $\hat f_j$ are poorly determined even if their sum is well estimated. The lab shows this: raise the correlation between the covariates and count sweeps to convergence.</p>`,
        lab: 'gam-backfitting',
      },
      {
        title: 'Penalized regression splines',
        html: R`
<p>Modern GAM software represents each smooth by a moderate basis expansion $f_j(x) = \sum_{k=1}^{q_j}B_{jk}(x)\gamma_{jk}$ (cubic regression splines, thin-plate splines, B-splines) and controls wiggliness with a quadratic penalty $\lambda_j\boldsymbol\gamma_j^\top P_j\boldsymbol\gamma_j$. Two common penalties:</p>
<ul>
<li><strong>Integrated squared second derivative</strong> $\int f_j''(x)^2\d x = \boldsymbol\gamma_j^\top P_j\boldsymbol\gamma_j$ with $(P_j)_{kl} = \int B_{jk}''B_{jl}''$ — the smoothing-spline penalty (see <a href="#/t/splines">splines</a>, where the minimizer over all functions is shown to be a natural cubic spline).</li>
<li><strong>P-splines</strong> (Eilers & Marx, 1996): B-splines on equally spaced knots with a difference penalty $\|\Delta^2\boldsymbol\gamma\|^2 = \boldsymbol\gamma^\top D^\top D\boldsymbol\gamma$ on adjacent coefficients, a discrete version of the same idea that is trivial to construct.</li>
</ul>
<p>Stacking all bases into a model matrix $X$ (with parametric terms) and the penalties into a block-diagonal $S_\lambda = \sum_j\lambda_jP_j$, the Gaussian GAM solves a ridge-type problem:
$$\hat\bbeta = \argmin_{\bbeta}\|\by - X\bbeta\|^2 + \bbeta^\top S_\lambda\bbeta = (X^\top X + S_\lambda)^{-1}X^\top\by,\qquad \hat\by = A_\lambda\by,\quad A_\lambda = X(X^\top X + S_\lambda)^{-1}X^\top.$$</p>
<div class="def" data-name="Effective degrees of freedom">
<p>$\text{edf} = \tr(A_\lambda)$. It ranges from the dimension of the penalty's null space (a straight line for a second-derivative penalty) as $\lambda\to\infty$ to the basis dimension $q$ as $\lambda\to0$. Per-term edf are the corresponding diagonal blocks of $(X^\top X + S_\lambda)^{-1}X^\top X$.</p>
</div>
<p>The basis dimension matters little once it is generous enough (say 10–40 per smooth): the penalty, not the knots, controls smoothness. This is what makes penalized regression splines a practical replacement for full smoothing splines.</p>
<div class="note" data-name="Smooths are random effects">
<p>The penalty $\lambda\boldsymbol\gamma^\top P\boldsymbol\gamma$ is, up to a constant, the negative log-density of the improper Gaussian prior $\boldsymbol\gamma\sim\N(\mathbf 0,\sigma^2(\lambda P)^-)$. The penalized estimate is the posterior mode, and also the BLUP in a <a href="#/t/mixed-models">linear mixed model</a> whose random-effect variance is $\sigma^2/\lambda$. Smoothing-parameter selection then becomes variance-component estimation, which is the basis of REML smoothness selection and of Bayesian credible bands for $\hat f_j$.</p>
</div>`,
      },
      {
        title: 'Non-Gaussian responses: penalized IRLS',
        html: R`
<p>For a GLM family, maximize the penalized log-likelihood
$$\ell_p(\bbeta) = \ell(\bbeta) - \frac{1}{2\phi}\bbeta^\top S_\lambda\bbeta.$$
The <a href="#/t/irls">IRLS derivation</a> goes through with the penalty added to the information: each iteration solves
$$\bbeta^{(t+1)} = (X^\top WX + S_\lambda)^{-1}X^\top W\mathbf z,$$
with the usual working response $\mathbf z$ and weights $W$. This is <em>penalized IRLS</em> (P-IRLS). The older <em>local scoring</em> algorithm of Hastie and Tibshirani (1990) is the same idea with backfitting used to solve each weighted additive fit: compute $\mathbf z$ and $W$, backfit $\mathbf z$ on the covariates with weights $W$, update, repeat.</p>
<p>At convergence, $A_\lambda = X(X^\top\hat WX + S_\lambda)^{-1}X^\top\hat W$ is the influence matrix of the final weighted fit, and the approximate posterior covariance $\hat\phi(X^\top\hat WX + S_\lambda)^{-1}$ gives pointwise credible bands for each $\hat f_j$. Nychka (1988) showed such bands have close to nominal coverage <em>averaged across the function</em>, though not pointwise at every $x$ (they undercover where the function is most curved and overcover elsewhere).</p>`,
      },
      {
        title: 'Choosing the smoothing parameters',
        html: R`
<p>Each $\lambda_j$ trades bias against variance. Criteria:</p>
<ul>
<li><strong>GCV</strong> (Craven & Wahba, 1979): $\mathcal V_g = \dfrac{n\,\|\by - A_\lambda\by\|^2}{\{n - \tr(A_\lambda)\}^2}$ — leave-one-out cross-validation with the leverages $A_{ii}$ replaced by their average, which makes it invariant to rotations of the problem. For GLMs, the deviance replaces the residual sum of squares.</li>
<li><strong>UBRE / Mallows' $C_p$ / AIC</strong> when $\phi$ is known (Poisson, binomial): $\mathcal V_u = D(\hat\bbeta)/n - \phi + 2\phi\,\tr(A_\lambda)/n$, an unbiased estimate of predictive risk up to a constant.</li>
<li><strong>REML / marginal likelihood</strong> (Wood, 2011): treat the smooths as random effects and maximize the restricted likelihood of $\lambda$. Less prone than GCV to occasional severe undersmoothing and with a smoother objective; the default in modern software.</li>
</ul>
<p>In the lab a single smooth is fitted to Gaussian or binary data. Drag $\lambda$ and watch edf and the criterion curve; the optimum balances a curve that tracks noise (small $\lambda$, high edf) against one that flattens real structure (large $\lambda$, edf near 2).</p>`,
        lab: 'gam-smoothing',
      },
      {
        title: 'Pitfalls and extensions',
        html: R`
<div class="warn" data-name="Inference conditional on the smoothing parameters">
<p>Standard errors, $p$-values and edf are computed treating $\hat\lambda$ as known; uncertainty in $\lambda$ is ignored. Wood's corrections add it approximately. Approximate $p$-values for "is $f_j$ zero?" use a Wald statistic with a rank based on edf and are roughly calibrated, but not exact.</p>
</div>
<div class="warn" data-name="Concurvity and extrapolation">
<p>With strongly related covariates the decomposition into separate $f_j$ is unstable even if the fit is fine: report concurvity measures before interpreting shapes. Outside the data range a spline extrapolates linearly (second-derivative penalty) or worse; never read the tails of $\hat f_j$ where data are sparse — the bands widen there for a reason.</p>
</div>
<p><strong>Extensions</strong>: varying-coefficient models $f(z)\,x$; tensor-product smooths $f(x_1,x_2)$ for interactions with different scales; GAMs for location, scale and shape (GAMLSS), where every distributional parameter gets its own additive predictor; and GAMMs, which combine smooths with ordinary random effects, unified by the mixed-model view of penalties. Boosting with componentwise smoothers (<a href="#/t/boosting">boosting</a>) fits additive models by a greedy, regularized cousin of backfitting.</p>`,
      },
    ],
    quiz: [
      { q: R`Why does an additive model escape the curse of dimensionality, and what does it give up?`,
        a: R`If the regression function is additive, each component is a one-dimensional function and can be estimated at the one-dimensional rate $n^{-2s/(2s+1)}$ regardless of $d$ (Stone, 1985). It gives up interactions: effects of one covariate cannot depend on the level of another unless an interaction smooth is explicitly included.` },
      { q: R`Show that for two smoothers backfitting errors satisfy $\mathbf e_2^{(t+1)} = S_2S_1\mathbf e_2^{(t)}$, and interpret slow convergence.`,
        a: R`Subtracting the fixed-point relations from the updates gives $\mathbf e_1^{(t+1)} = -S_1\mathbf e_2^{(t)}$ and $\mathbf e_2^{(t+1)} = -S_2\mathbf e_1^{(t+1)} = S_2S_1\mathbf e_2^{(t)}$. Convergence is slow when $\|S_2S_1\|$ is near 1, i.e. when some function is almost reproduced by smoothing on $x_1$ and then on $x_2$: the covariates are nearly concurve, so the split of the fit between $f_1$ and $f_2$ is poorly determined.` },
      { q: R`What is the effective degrees of freedom of a penalized spline fit, and what are its limits as $\lambda\to0$ and $\lambda\to\infty$ for a second-difference P-spline with $q$ basis functions?`,
        a: R`$\text{edf} = \tr A_\lambda = \tr\{X(X^\top X + \lambda D^\top D)^{-1}X^\top\}$. As $\lambda\to0$ it tends to $q$ (unpenalized regression spline, assuming full rank). As $\lambda\to\infty$ the coefficients are forced into the null space of the second-difference penalty, which is linear sequences, so the fit becomes a straight line and edf $\to2$.` },
      { q: R`Why is the penalized IRLS update $(X^\top WX + S_\lambda)^{-1}X^\top W\mathbf z$?`,
        a: R`The penalized log-likelihood $\ell(\bbeta) - \frac1{2\phi}\bbeta^\top S_\lambda\bbeta$ has score $\phi^{-1}\{X^\top WG(\by-\bmu) - S_\lambda\bbeta\}$ and expected information $\phi^{-1}(X^\top WX + S_\lambda)$. A Fisher scoring step from $\bbeta^{(t)}$ therefore satisfies $(X^\top WX + S_\lambda)\bbeta^{(t+1)} = (X^\top WX + S_\lambda)\bbeta^{(t)} + X^\top WG(\by-\bmu) - S_\lambda\bbeta^{(t)} = X^\top W\mathbf z$.` },
    ],
  });
})();
