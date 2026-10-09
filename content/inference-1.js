/* Statistical Inference I: models, estimation, likelihood, sufficiency, exponential families,
 * maximum likelihood, Fisher information, and optimal unbiased estimation. Labs: js/labs/inference-1.js */
(function () {
  const R = String.raw;

  /* ================================================================== */
  Atlas.topic({
    id: 'statistical-models',
    domain: 'inference',
    order: 10,
    title: 'Statistical Models & the Inference Problem',
    subtitle: 'Running probability backwards: families of distributions, parameters, and what data can identify',
    level: 1,
    sources: { aos: '§6.1–6.2', cb: '§6.1', other: 'Cox (2006), Principles of Statistical Inference, Ch. 1–2; Box (1976)' },
    prereqs: ['random-variables', 'distribution-families'],
    related: ['point-estimation', 'likelihood', 'bayesian-inference', 'statistical-functionals'],
    tags: ['model', 'parameter', 'identifiability', 'misspecification', 'frequentist', 'Bayesian'],
    summary: R`Probability theory starts from a known distribution and asks what data will look like; statistical inference starts from data and asks which distribution produced them. That inverse problem has no answer until we commit to a <strong>statistical model</strong> — a family $\mathcal P = \{P_\theta : \theta \in \Theta\}$ assumed to contain (or approximate) the truth. The model decides what is learnable: a parameter that two different members of the family share cannot be recovered from any amount of data (non-identifiability), and when the truth lies outside the family, estimators quietly converge to the member closest to it in Kullback–Leibler divergence rather than to anything "true".`,
    survey: {
      task: 'Formalize an inference question as a family of distributions plus a target functional',
      assumptions: R`Data $X \sim P$ for some $P$ in (or near) $\mathcal P$; the target $\psi = T(P)$ is identified`,
      output: R`A model $\mathcal P$, parameter space $\Theta$, estimand $\psi(\theta)$, and a notion of repeated sampling`,
      strengths: 'Makes assumptions explicit; separates what is estimated from how',
      weaknesses: 'Every conclusion is conditional on the model; misspecification shifts the target silently',
      complexity: '—',
    },
    sections: [
      {
        title: 'From probability to inference',
        html: R`
<p>In probability we are handed a distribution $P$ and compute things about the random outcome $X$: the chance that a sample mean exceeds 3, the expected number of records in 100 draws. Statistics reverses the arrow. We observe one realization $x$ of $X$ and want to say something about the $P$ that produced it.</p>
<p>Stated that baldly the problem is hopeless: <em>every</em> distribution that puts positive probability (or density) near $x$ is compatible with it. Inference becomes possible only once we restrict the candidates. That restriction is the model, and it carries most of the assumptions in any analysis — usually far more than the choice of estimator.</p>
<p>The classical tasks, all defined relative to a model, are:</p>
<ul>
<li><strong>Point estimation</strong> — a single best guess $\hat\psi$ for a quantity $\psi$ ("the failure rate is 0.031 per hour"). See <a href="#/t/point-estimation">point estimation</a>.</li>
<li><strong>Set estimation</strong> — a random set $C(X)$ with $\P_\theta(\psi(\theta)\in C(X)) \ge 1-\alpha$ for every $\theta$. See <a href="#/t/confidence-intervals">confidence intervals</a>.</li>
<li><strong>Hypothesis testing</strong> — decide between $\theta\in\Theta_0$ and $\theta\in\Theta_1$ with controlled error rates. See <a href="#/t/hypothesis-testing">hypothesis testing</a>.</li>
<li><strong>Prediction</strong> — say something about a future $X_{n+1}$, which is random even when $\theta$ is known.</li>
</ul>
<div class="note">
<p>A useful mental picture: the data-generating process is a machine with a dial set to an unknown $\theta$. You see one batch of output. Each inferential statement is a claim about the dial, and its quality is judged by how such claims would fare if the machine were run again and again at that same (unknown) setting.</p>
</div>`,
      },
      {
        title: 'Models, parameters and estimands',
        html: R`
<div class="def" data-name="Statistical model">
<p>A <em>statistical model</em> for data $X$ taking values in a sample space $\mathcal X$ is a set $\mathcal P$ of probability distributions on $\mathcal X$. It is <em>parametric</em> if it can be indexed smoothly by a finite-dimensional parameter, $\mathcal P = \{P_\theta: \theta\in\Theta\}$ with $\Theta\subseteq\R^k$; <em>nonparametric</em> if no such finite indexing exists (e.g. "all distributions on $\R$ with finite variance"); and <em>semiparametric</em> if it has a finite-dimensional parameter of interest together with an infinite-dimensional nuisance component.</p>
</div>
<p>Most of this domain works with a <strong>dominated</strong> parametric family: every $P_\theta$ has a density $f(x;\theta)$ with respect to a common measure (Lebesgue for continuous data, counting measure for discrete data). This is what makes the <a href="#/t/likelihood">likelihood function</a> well defined.</p>
<table>
<thead><tr><th>Model</th><th>$\Theta$</th><th>Type</th></tr></thead>
<tbody>
<tr><td>$X_1,\dots,X_n \iid \N(\mu,\sigma^2)$</td><td>$\R\times(0,\infty)$</td><td>parametric, $k=2$</td></tr>
<tr><td>$X_1,\dots,X_n \iid \text{Poisson}(\lambda)$</td><td>$(0,\infty)$</td><td>parametric, $k=1$</td></tr>
<tr><td>$X_i = \theta + \varepsilon_i$, $\varepsilon_i \iid g$ with $g$ any density symmetric about 0</td><td>$\R\times\{\text{symmetric } g\}$</td><td>semiparametric</td></tr>
<tr><td>$X_1,\dots,X_n \iid F$, $F$ any CDF</td><td>all CDFs</td><td>nonparametric</td></tr>
</tbody>
</table>
<p>We rarely care about the whole of $\theta$. The quantity we actually want is an <strong>estimand</strong> $\psi = g(\theta)$ — or, nonparametrically, a <a href="#/t/statistical-functionals">functional</a> $\psi = T(P)$ such as the mean $\int x\,\d P$ or the median $P^{-1}(1/2)$. The remaining components are <strong>nuisance parameters</strong>: needed to specify the distribution, irrelevant to the question. In $\N(\mu,\sigma^2)$ with interest in $\mu$, $\sigma$ is a nuisance; it still affects how precisely $\mu$ can be estimated, and handling it well (e.g. the $t$ distribution instead of the normal) is much of the craft.</p>
<div class="def" data-name="Statistic, estimator, sampling distribution">
<p>A <em>statistic</em> is any (measurable) function $T = t(X)$ of the data that does not depend on the unknown parameter. An <em>estimator</em> of $\psi$ is a statistic used to guess $\psi$. Because $X$ is random, $T$ is random; its distribution under $P_\theta$ is the <em>sampling distribution</em> of $T$, and it generally depends on $\theta$.</p>
</div>
<p>The requirement "does not depend on $\theta$" is the whole point: $\bar X - \mu$ is not a statistic, because you cannot compute it. Quantities like it, whose <em>distribution</em> is free of $\theta$, are called pivots and drive interval estimation.</p>`,
      },
      {
        title: 'Identifiability: what the data can and cannot tell you',
        html: R`
<div class="def" data-name="Identifiability">
<p>The parameterization $\theta\mapsto P_\theta$ is <em>identifiable</em> if it is injective: $\theta_1 \neq \theta_2 \Rightarrow P_{\theta_1}\neq P_{\theta_2}$. An estimand $\psi = g(\theta)$ is identified if $P_{\theta_1}=P_{\theta_2}\Rightarrow g(\theta_1)=g(\theta_2)$, i.e. if $\psi$ is a function of the distribution $P$ alone.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="No consistency without identification">
<p>Suppose $P_{\theta_1} = P_{\theta_2}$ (for all sample sizes) but $\psi_1 = g(\theta_1) \ne g(\theta_2) = \psi_2$. Then no sequence of estimators $\hat\psi_n$ can be consistent for $\psi$ at both $\theta_1$ and $\theta_2$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Because $\hat\psi_n$ is a function of the data alone, it has the same distribution under $\theta_1$ and under $\theta_2$. Let $\varepsilon = |\psi_1-\psi_2|/3$ and let $A_n = \{|\hat\psi_n - \psi_1| \le \varepsilon\}$. Consistency at $\theta_1$ forces $\P_{\theta_1}(A_n)\to1$, hence $\P_{\theta_2}(A_n)\to 1$ too. But on $A_n$ we have $|\hat\psi_n - \psi_2| \ge 2\varepsilon$, so $\P_{\theta_2}(|\hat\psi_n-\psi_2|>\varepsilon)\to 1$, contradicting consistency at $\theta_2$.</p>
</details>
<p>Non-identifiability is not exotic; it is built into many workhorse models.</p>
<div class="ex" data-name="Four ways to lose identification">
<ul>
<li><strong>Only a combination is visible.</strong> If $X_i \iid \N(\alpha+\beta, 1)$, only $\alpha+\beta$ is identified; the likelihood is constant along lines $\alpha+\beta = c$.</li>
<li><strong>Label switching.</strong> In a two-component mixture $p\,\N(\mu_1,1)+(1-p)\N(\mu_2,1)$, the parameter $(p,\mu_1,\mu_2)$ and $(1-p,\mu_2,\mu_1)$ give the same distribution. The family is identifiable only up to permutation of labels, or after imposing $\mu_1 \lt \mu_2$. If $\mu_1=\mu_2$, $p$ is not identified at all.</li>
<li><strong>Collinearity.</strong> In regression $Y = X\beta+\varepsilon$, $\beta$ is identified iff $X$ has full column rank; otherwise $\beta$ and $\beta + v$ with $Xv=0$ are indistinguishable (see <a href="#/t/ols-geometry">least squares as projection</a>).</li>
<li><strong>Latent structure.</strong> In factor analysis the loadings are identified only up to rotation; with competing risks, the dependence between latent failure times is not identified from the observed first failure (Tsiatis, 1975).</li>
</ul>
</div>
<p>Two refinements matter in practice. <em>Local</em> identifiability (injectivity in a neighbourhood) holds under mild conditions whenever the <a href="#/t/fisher-information">Fisher information</a> matrix is nonsingular (Rothenberg, 1971). And <em>weak</em> identification — distinct parameters giving distributions that are different but very close — is the finite-sample shadow of non-identifiability: the likelihood is nearly flat along a direction, estimates are wildly variable, and asymptotic approximations are poor. The lab lets you watch a mixture move from identified to weakly identified to unidentified.</p>`,
        lab: 'statistical-models-identifiability',
      },
      {
        title: 'Two readings of probability',
        html: R`
<p>Once a model is fixed, there are two broad ways to reason about $\theta$, and they differ in what is treated as random.</p>
<div class="cols">
<div><h4>Frequentist</h4><p>$\theta$ is a fixed unknown constant; the data are random. A procedure $\delta(X)$ is judged by its behaviour over hypothetical repetitions of the experiment at each fixed $\theta$: bias, variance, risk, coverage, error rates. Guarantees must hold <em>for every</em> $\theta\in\Theta$ (or be stated as functions of $\theta$).</p></div>
<div><h4>Bayesian</h4><p>Uncertainty about $\theta$ is expressed by a prior distribution $\pi(\theta)$; after seeing $x$ we condition, obtaining the posterior $\pi(\theta\mid x)\propto f(x;\theta)\pi(\theta)$. Statements are about $\theta$ given the observed data, not about repetitions. See <a href="#/t/bayesian-inference">Bayesian inference</a>.</p></div>
</div>
<p>The distinction shows most clearly in interval statements. A frequentist 95% confidence interval $C(X)$ satisfies $\P_\theta(\theta\in C(X))\ge 0.95$ for every $\theta$: the probability refers to the random interval, and once $x$ is observed, $C(x)$ either contains $\theta$ or it does not. A 95% Bayesian credible set satisfies $\Pi(\theta\in C(x)\mid x) = 0.95$: the probability refers to $\theta$, given these data. The two often agree numerically in large samples (Bernstein–von Mises), but they answer different questions.</p>
<p>Both views share the same model $\mathcal P$, and both use the <a href="#/t/likelihood">likelihood</a> $\theta\mapsto f(x;\theta)$; they differ in how they summarize it. <a href="#/t/decision-theory">Decision theory</a> provides a common language: a loss $L(\theta,a)$, the frequentist risk $R(\theta,\delta) = \E_\theta L(\theta,\delta(X))$, and the Bayes risk $\int R(\theta,\delta)\,\pi(\theta)\d\theta$.</p>`,
      },
      {
        title: 'All models are wrong: misspecification',
        html: R`
<p>"All models are wrong, but some are useful" (Box). If the true distribution $P$ is not in $\mathcal P$, what does a model-based estimator estimate? For likelihood-based methods the answer is clean.</p>
<div class="thm" data-kind="Proposition" data-name="The pseudo-true parameter">
<p>Let $X_1,\dots,X_n\iid P$ with density $p$, and fit the family $\{f(\cdot;\theta)\}$. Define the <em>pseudo-true</em> parameter
$$\theta^* = \argmin_{\theta\in\Theta} \KL(p\,\|\,f_\theta) = \argmax_{\theta\in\Theta}\E_P \log f(X;\theta).$$
Then for each $\theta$, $\frac1n\sum_i \log f(X_i;\theta) \asto \E_P\log f(X;\theta) = \E_P \log p(X) - \KL(p\,\|\,f_\theta)$, so the average log-likelihood is maximized, in the limit, at $\theta^*$. Under the regularity conditions of the <a href="#/t/mle">MLE</a> consistency theorem, $\hth_n \pto \theta^*$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The identity is algebra: $\E_P\log f_\theta = \E_P\log p - \E_P\log(p/f_\theta)$, and the first term does not involve $\theta$. The convergence is the strong law of large numbers applied to the IID variables $\log f(X_i;\theta)$, valid whenever $\E_P|\log f(X;\theta)|<\infty$. Upgrading pointwise convergence of the criterion to convergence of its maximizer requires the uniform argument given in the MLE topic (Huber, 1967; White, 1982).</p>
</details>
<div class="ex" data-name="Fitting a normal to anything">
<p>Fit $\N(\mu,\sigma^2)$ to data from any $P$ with mean $m$ and variance $v$. Then
$-\E_P\log f(X;\mu,\sigma^2) = \tfrac12\log(2\pi\sigma^2) + \frac{v + (m-\mu)^2}{2\sigma^2},$
minimized at $\mu^* = m$, $\sigma^{2*} = v$. The KL projection onto the normal family matches the first two moments, so the normal MLE $(\bar X, \hat\sigma^2)$ is consistent for the true mean and variance no matter the shape of $P$. What breaks is everything that depends on the normal shape: tail probabilities, quantiles, and the model-based standard error of $\hat\sigma^2$ (which assumes kurtosis 3). Fitting an Exponential$(\lambda)$ model instead gives $\lambda^* = 1/m$.</p>
</div>
<p>Two practical lessons. First, a misspecified estimator can be a perfectly good estimator of a meaningful quantity ($\theta^*$ often has an interpretation, like "best linear predictor" in regression). Second, its uncertainty must be assessed without trusting the model: the <a href="#/t/mle">sandwich variance</a> or the <a href="#/t/bootstrap">nonparametric bootstrap</a> remain valid where the inverse Fisher information does not. The lab shows the fitted model converging, as $n$ grows, to the KL projection rather than to the truth.</p>`,
        lab: 'statistical-models-misspec',
      },
      {
        title: 'The price of flexibility',
        html: R`
<p>Why not always use a nonparametric model and avoid misspecification altogether? Because flexibility is paid for in precision. In a regular parametric model, the best estimators have error of order $n^{-1/2}$, with a constant given by the <a href="#/t/fisher-information">Fisher information</a>. Estimating a whole density nonparametrically with twice-differentiable $f$ achieves at best mean squared error of order $n^{-4/5}$ — root-MSE $n^{-2/5}$ — and in $d$ dimensions $n^{-4/(4+d)}$ (see <a href="#/t/kernel-density">kernel density estimation</a>). Smooth functionals such as the mean or a quantile remain $\sqrt n$-estimable nonparametrically, which is why the <a href="#/t/empirical-cdf">empirical CDF</a> and the <a href="#/t/bootstrap">bootstrap</a> are so useful.</p>
<table>
<thead><tr><th></th><th>Parametric</th><th>Semiparametric</th><th>Nonparametric</th></tr></thead>
<tbody>
<tr><td>Typical rate for the target</td><td>$n^{-1/2}$</td><td>$n^{-1/2}$ (for the finite-dim part)</td><td>slower for curves; $n^{-1/2}$ for smooth functionals</td></tr>
<tr><td>Risk if assumptions fail</td><td>bias toward $\theta^*$; wrong SEs</td><td>moderate</td><td>low</td></tr>
<tr><td>Interpretability</td><td>high</td><td>high for $\theta$</td><td>often lower</td></tr>
</tbody>
</table>
<p>The rest of this domain develops the parametric toolkit: how to reduce data without losing information (<a href="#/t/sufficiency">sufficiency</a>, <a href="#/t/exponential-families">exponential families</a>), how to construct estimators (<a href="#/t/method-of-moments">moments</a>, <a href="#/t/mle">maximum likelihood</a>), and how to measure and bound their precision (<a href="#/t/point-estimation">MSE</a>, <a href="#/t/fisher-information">Cramér–Rao</a>, <a href="#/t/rao-blackwell">Rao–Blackwell and Lehmann–Scheffé</a>).</p>`,
      },
    ],
    quiz: [
      { q: R`Is $\bar X - \mu$ a statistic? Is $(\bar X-\mu)/(S/\sqrt n)$? What is the role of the second quantity?`,
        a: R`Neither is a statistic, because both depend on the unknown $\mu$ and cannot be computed from data. The second is a <em>pivot</em>: under the normal model its distribution ($t_{n-1}$) is free of $(\mu,\sigma)$, which is what lets us invert it into a confidence interval.` },
      { q: R`In the model $X_i\iid\N(\alpha\beta,1)$ with $(\alpha,\beta)\in\R^2$, which functions of the parameter are identified? Can any estimator be consistent for $\alpha$?`,
        a: R`Only functions of the product $\alpha\beta$ are identified, since $P_{(\alpha,\beta)}$ depends on $(\alpha,\beta)$ only through $\alpha\beta$. Because $(\alpha,\beta)=(1,2)$ and $(2,1)$ give the same distribution but different $\alpha$, no estimator can be consistent for $\alpha$ at both points.` },
      { q: R`You fit an Exponential$(\lambda)$ model by maximum likelihood to data that are actually Gamma$(3, \text{scale } 2)$. What does $\hat\lambda$ converge to, and which model-based conclusions remain trustworthy?`,
        a: R`$\hat\lambda = 1/\bar X\to 1/\E X = 1/6$, the KL projection $\lambda^*$. The fitted mean is therefore right, but the model-based variance of $\hat\lambda$ (which uses $\Var X = \E X^2$ for the exponential, i.e. coefficient of variation 1) is wrong — the true CV is $1/\sqrt3$ — and so are fitted tail probabilities. Use sandwich or bootstrap standard errors.` },
      { q: R`A two-component normal mixture is fitted with $\mu_1$ and $\mu_2$ unconstrained. Why does the likelihood always have at least two global maximizers, and why can things get much worse when the true $\mu_1\approx\mu_2$?`,
        a: R`Swapping the component labels, $(p,\mu_1,\mu_2)\mapsto(1-p,\mu_2,\mu_1)$, leaves the distribution unchanged, so every maximizer has a mirror image. When $\mu_1=\mu_2$ the mixing weight $p$ is completely unidentified, and near that point the likelihood is almost flat in a whole direction: estimates are highly variable and the usual asymptotic theory (which needs nonsingular information) fails.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'point-estimation',
    domain: 'inference',
    order: 20,
    title: 'Point Estimation: Bias, Variance, MSE & Consistency',
    subtitle: 'How to judge a guess you will only ever make once',
    level: 1,
    sources: { aos: '§6.3', cb: '§7.1, §7.3.1', other: 'Lehmann & Casella (1998), Theory of Point Estimation, Ch. 1–2' },
    prereqs: ['statistical-models', 'expectation'],
    related: ['mle', 'rao-blackwell', 'decision-theory', 'bias-variance', 'lln'],
    tags: ['bias', 'variance', 'MSE', 'consistency', 'shrinkage', 'risk'],
    summary: R`An estimator is a rule applied to random data, so it has a sampling distribution, and its quality is a property of that distribution, not of any single estimate. The mean squared error splits exactly into squared bias plus variance; trading a little bias for a larger cut in variance is often a good deal, which is why unbiasedness is a constraint rather than a virtue in itself. Consistency asks only that the sampling distribution collapse onto the truth as $n\to\infty$, and the <em>rate</em> of that collapse ($n^{-1/2}$ in regular problems, faster or slower in others) is as important as the fact. No estimator is best at every $\theta$, which forces the choices — restrict, average, or bound the risk — that organize the rest of estimation theory.`,
    survey: {
      task: R`Evaluate and compare estimators $\hth$ of $\theta$ (or $\psi(\theta)$)`,
      assumptions: R`A model $\{P_\theta\}$; finite second moments for MSE`,
      output: R`$\Bias_\theta$, $\Var_\theta$, $\MSE_\theta$ as functions of $\theta$; consistency and rate`,
      strengths: 'MSE decomposition is exact and universal; consistency needs very little',
      weaknesses: 'Risk functions usually cross: no uniformly best estimator exists',
      complexity: 'Analytic or Monte Carlo over a grid of θ',
    },
    sections: [
      {
        title: 'Estimators and their sampling distributions',
        html: R`
<p>Let $X_1,\dots,X_n$ come from $P_\theta$. A <strong>point estimator</strong> of $\theta$ is any statistic $\hth_n = t(X_1,\dots,X_n)$. The number $t(x_1,\dots,x_n)$ you compute from the observed data is the <em>estimate</em>; the random variable $\hth_n$ is the <em>estimator</em>. Everything we can say about the quality of an estimate is really a statement about the estimator: you will see one draw from its sampling distribution, and you want that distribution concentrated near $\theta$ <em>whatever $\theta$ is</em>.</p>
<div class="def" data-name="Standard error">
<p>The <em>standard error</em> of $\hth$ is $\se_\theta(\hth) = \sqrt{\Var_\theta(\hth)}$, the spread of its sampling distribution. Since it typically depends on the unknown $\theta$, in practice we report an <em>estimated</em> standard error $\widehat{\se}$, obtained by plugging in $\hth$, by the <a href="#/t/delta-method">delta method</a>, from <a href="#/t/fisher-information">Fisher information</a>, or by the <a href="#/t/bootstrap">bootstrap</a>.</p>
</div>
<div class="ex" data-name="Proportion">
<p>For $X_i\iid\text{Bernoulli}(p)$, $\hat p = \bar X$ has $\E\hat p = p$ and $\se(\hat p) = \sqrt{p(1-p)/n}$, estimated by $\sqrt{\hat p(1-\hat p)/n}$. Note the standard error is largest at $p=1/2$ and vanishes at $p\in\{0,1\}$ — where the plug-in estimate $\widehat\se = 0$ is badly overconfident (if you see 0 successes in 20 trials, you have not learned that $p=0$).</p>
</div>`,
      },
      {
        title: 'Bias, variance and the MSE decomposition',
        html: R`
<div class="def" data-name="Bias and mean squared error">
<p>$\Bias_\theta(\hth) = \E_\theta\hth - \theta$. The estimator is <em>unbiased</em> if $\Bias_\theta(\hth) = 0$ for every $\theta\in\Theta$. The <em>mean squared error</em> is $\MSE_\theta(\hth) = \E_\theta(\hth-\theta)^2$, the risk under squared-error loss.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Bias–variance decomposition">
<p>If $\E_\theta\hth^2<\infty$, then $\MSE_\theta(\hth) = \Bias_\theta(\hth)^2 + \Var_\theta(\hth)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $m = \E_\theta\hth$. Then $\hth - \theta = (\hth - m) + (m-\theta)$, and squaring and taking expectations gives $\E(\hth-m)^2 + 2(m-\theta)\E(\hth-m) + (m-\theta)^2$. The cross term vanishes because $\E(\hth - m)=0$, leaving $\Var_\theta(\hth) + \Bias_\theta(\hth)^2$.</p>
</details>
<p>The decomposition is exact and model-free, and it is the reason biased estimators can win. Two families of examples recur throughout statistics.</p>
<div class="ex" data-name="Which divisor for the variance?">
<p>For $X_i\iid\N(\mu,\sigma^2)$ let $SS = \sum(X_i-\bar X)^2$, so $SS/\sigma^2\sim\chi^2_{n-1}$ with mean $n-1$ and variance $2(n-1)$. For the family $cSS$,
$$\MSE(c\,SS) = \sigma^4\Big[\,2(n-1)c^2 + \big((n-1)c - 1\big)^2\Big].$$
Setting the derivative in $c$ to zero gives $c = 1/(n+1)$. Hence</p>
<table>
<thead><tr><th>Divisor</th><th>Bias$/\sigma^2$</th><th>MSE$/\sigma^4$</th></tr></thead>
<tbody>
<tr><td>$n-1$ (unbiased $S^2$)</td><td>$0$</td><td>$2/(n-1)$</td></tr>
<tr><td>$n$ (MLE)</td><td>$-1/n$</td><td>$(2n-1)/n^2$</td></tr>
<tr><td>$n+1$</td><td>$-2/(n+1)$</td><td>$2/(n+1)$ (minimum)</td></tr>
</tbody>
</table>
<p>The unbiased estimator has the <em>largest</em> MSE of the three, for every $(\mu,\sigma)$. The optimal divisor depends on the normal kurtosis; under heavier tails it shifts.</p>
</div>
<div class="ex" data-name="Shrinking a mean">
<p>For $X_i\iid\N(\theta,\sigma^2)$ consider $\hth_c = c\bar X$ with $0\le c\le1$. Then $\MSE_\theta(\hth_c) = c^2\sigma^2/n + (1-c)^2\theta^2$. Shrinkage beats $\bar X$ exactly when $\theta^2 \lt \frac{1+c}{1-c}\,\frac{\sigma^2}{n}$: near the shrinkage target the bias is small and the variance saving is real; far away the bias dominates. The lab plots these risk curves and lets you choose the competitor.</p>
</div>`,
        lab: 'point-estimation-risk',
      },
      {
        title: 'Unbiasedness: virtue and limits',
        html: R`
<p>Unbiasedness is attractive — on average the estimator is right — and it is the restriction that makes the elegant theory of <a href="#/t/rao-blackwell">uniformly minimum-variance unbiased estimation</a> possible. But it has serious limitations.</p>
<div class="warn" data-name="Unbiasedness is not preserved by transformation">
<p>If $\E\hth=\theta$ and $g$ is nonlinear, $\E g(\hth)\ne g(\theta)$ in general; for strictly convex $g$, Jensen gives $\E g(\hth) > g(\theta)$. Thus $S^2$ is unbiased for $\sigma^2$ but $S$ underestimates $\sigma$ on average ($\E S \lt \sigma$ by Jensen applied to the concave square root). The <a href="#/t/mle">MLE</a>, by contrast, is equivariant: the MLE of $g(\theta)$ is $g(\hth)$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Unbiased estimators need not exist">
<p>Let $X\sim\text{Binomial}(n,p)$, $p\in(0,1)$. There is no unbiased estimator of $1/p$ (nor of any function of $p$ that is not a polynomial of degree at most $n$).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Any estimator is a function $t(X)$ on $\{0,\dots,n\}$, so $\E_p t(X) = \sum_{k=0}^n t(k)\binom nk p^k(1-p)^{n-k}$ is a polynomial in $p$ of degree at most $n$. It is bounded on $(0,1)$, whereas $1/p\to\infty$ as $p\downarrow0$; so they cannot agree on all of $(0,1)$. The same argument applies to any non-polynomial target.</p>
</details>
<div class="ex" data-name="An unbiased estimator nobody should use">
<p>Let $X\sim\text{Poisson}(\lambda)$ (a single observation) and target $\psi=e^{-2\lambda}$. The estimator $t(X) = (-1)^X$ satisfies $\E_\lambda(-1)^X = e^{-\lambda}\sum_k (-\lambda)^k/k! = e^{-2\lambda}$, so it is unbiased — and since $X$ is a complete sufficient statistic, it is the <em>only</em> unbiased estimator. Yet it reports $-1$ for a positive quantity half the time. The biased $e^{-2X}$ is far better in MSE for every $\lambda$.</p>
</div>
<p>Alternatives exist: <em>median-unbiasedness</em> ($\P_\theta(\hth\le\theta)\ge\frac12$ and $\P_\theta(\hth\ge\theta)\ge\frac12$) is preserved by monotone transformations; <em>asymptotic</em> unbiasedness ($\Bias\to0$) is what most modern estimators satisfy.</p>`,
      },
      {
        title: 'Consistency and rates',
        html: R`
<div class="def" data-name="Consistency">
<p>A sequence $\hth_n$ is (weakly) <em>consistent</em> for $\theta$ if $\hth_n\pto\theta$ under $P_\theta$, for every $\theta\in\Theta$; <em>strongly consistent</em> if $\hth_n\asto\theta$. It is <em>$r_n$-consistent</em> if $r_n(\hth_n-\theta) = O_p(1)$, and <em>asymptotically normal</em> if $\sqrt n(\hth_n-\theta)\dto\N(0,v(\theta))$; then $v(\theta)$ is the <em>asymptotic variance</em>.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="MSE consistency implies consistency">
<p>If $\MSE_\theta(\hth_n)\to0$ — equivalently $\Bias_\theta(\hth_n)\to0$ and $\Var_\theta(\hth_n)\to0$ — then $\hth_n\pto\theta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By Markov's inequality applied to $(\hth_n-\theta)^2$: $\P_\theta(|\hth_n-\theta|>\varepsilon)\le \MSE_\theta(\hth_n)/\varepsilon^2\to0$. The equivalence is the bias–variance decomposition.</p>
</details>
<p>The converse is false: consistency is about the bulk of the distribution, MSE about its tails. With $X_i\iid\N(\theta,1)$, the estimator $\hth_n = \bar X_n + 1/(nX_1)$ is consistent (the second term is $O_p(1/n)$), but $\E|1/X_1| = \infty$ because the density of $X_1$ is positive at 0, so its MSE is infinite for every $n$. Likewise, the asymptotic variance $v(\theta)$ is the variance of the limit law, which need not equal $\lim n\Var(\hth_n)$.</p>
<p>Consistency propagates: if $\hth_n\pto\theta$ and $g$ is continuous at $\theta$, then $g(\hth_n)\pto g(\theta)$ (continuous mapping). Asymptotic normality propagates through differentiable $g$ by the <a href="#/t/delta-method">delta method</a>.</p>
<div class="ex" data-name="Not every rate is root-n">
<p>For $X_i\iid\text{Unif}(0,\theta)$, the moment estimator $2\bar X$ has variance $\theta^2/(3n)$: $\sqrt n$-consistent and asymptotically normal. The maximum $X_{(n)}$ has $\P(X_{(n)}\le\theta - t) = (1-t/\theta)^n$, so $n(\theta - X_{(n)})/\theta\dto\text{Exp}(1)$: it converges at rate $n$, an order of magnitude faster, to a non-normal limit. This happens because the support depends on $\theta$, so the data's edge carries sharp information. The second lab shows the two sampling distributions shrinking at different speeds.</p>
</div>`,
        lab: 'point-estimation-rates',
      },
      {
        title: 'No uniformly best estimator',
        html: R`
<p>We would like an estimator whose MSE is smallest at every $\theta$ simultaneously. In any nontrivial model none exists.</p>
<div class="thm" data-kind="Proposition" data-name="Constant estimators are unbeatable somewhere">
<p>Fix $\theta_0\in\Theta$. The estimator $\hth\equiv\theta_0$ has $\MSE_{\theta_0}=0$. Hence an estimator $\delta$ that is at least as good as every other estimator at every $\theta$ must have $\MSE_{\theta}(\delta)=0$ for all $\theta$, i.e. $\delta(X)=\theta$ almost surely under each $P_\theta$ — impossible unless the $P_\theta$ have mutually singular supports that reveal $\theta$ exactly.</p>
</div>
<p>The constant estimator is absurd, but it proves the point: risk functions cross, and choosing among estimators needs an additional principle. There are three classical ways out.</p>
<div class="cols">
<div><h4>Restrict the class</h4><p>Require unbiasedness, then minimize variance uniformly: the UMVUE, via <a href="#/t/rao-blackwell">Rao–Blackwell and Lehmann–Scheffé</a>. Or require equivariance under a group (e.g. location shifts).</p></div>
<div><h4>Summarize the risk</h4><p>Average it against a prior (Bayes) or take its worst case (minimax). See <a href="#/t/decision-theory">decision theory</a>.</p></div>
<div><h4>Go asymptotic</h4><p>Compare limiting variances; the <a href="#/t/fisher-information">Cramér–Rao bound</a> gives a benchmark that the <a href="#/t/mle">MLE</a> attains asymptotically.</p></div>
</div>
<div class="ex" data-name="A constant-risk competitor for a proportion">
<p>For $X\sim\text{Binomial}(n,p)$, the estimators $(X+a)/(n+b)$ have bias $(a-bp)/(n+b)$ and variance $np(1-p)/(n+b)^2$. Choosing $a=\sqrt n/2$, $b=\sqrt n$ makes the MSE <em>constant</em>:
$$\MSE_p = \frac{n\big[(\tfrac12-p)^2 + p(1-p)\big]}{(n+\sqrt n)^2} = \frac{1}{4(1+\sqrt n)^2}.$$
This is the minimax estimator (Hodges & Lehmann, 1950). It beats $X/n$, whose MSE is $p(1-p)/n$, near $p=1/2$ and loses near the edges. Try it in the first lab.</p>
</div>
<p>In more than two dimensions even the "obvious" estimator can be uniformly beaten: for $\bX\sim\N_d(\btheta, I)$ with $d\ge3$, the James–Stein estimator $\big(1-(d-2)/\|\bX\|^2\big)\bX$ has strictly smaller total MSE than $\bX$ at every $\btheta$ (Stein, 1956; James & Stein, 1961). Shrinkage is not a trick; it is often forced by optimality.</p>`,
      },
      {
        title: 'Pitfalls in judging estimators',
        html: R`
<div class="warn" data-name="Risk is a function, not a number">
<p>"Estimator A has lower MSE than B" is meaningful only for a stated $\theta$ or region of $\Theta$. Simulation studies that evaluate at one convenient parameter value can reverse their conclusions elsewhere — check a grid that covers the plausible range, including boundaries.</p>
</div>
<div class="warn" data-name="Squared error is a choice">
<p>MSE penalizes large errors heavily and is not scale-free. For variances, a loss such as $\hat\sigma^2/\sigma^2 - \log(\hat\sigma^2/\sigma^2) - 1$ (Stein's loss) is scale-invariant and, unlike squared error, penalizes severe underestimation without bound; it selects a different best divisor ($n-1$ under normality) than MSE does ($n+1$). Under absolute-error loss the Bayes estimator is the posterior median rather than the mean.</p>
</div>
<div class="warn" data-name="Asymptotics are a guide, not a guarantee">
<p>Two consistent, asymptotically normal estimators with the same asymptotic variance can differ greatly at $n=20$; and an estimator with infinite MSE can be perfectly consistent. Always ask whether $n$ is large <em>relative to the difficulty</em> (e.g. $np$ for rare events, or the number of parameters).</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`For normal data, rank the divisors $n-1$, $n$, $n+1$ for $SS=\sum(X_i-\bar X)^2$ by MSE as estimators of $\sigma^2$. Why does the unbiased one lose?`,
        a: R`MSE$/\sigma^4$ is $2/(n-1)$, $(2n-1)/n^2$, $2/(n+1)$ respectively, so $n+1 \succ n \succ n-1$. Shrinking $SS$ toward 0 introduces a bias of order $1/n$ (squared: order $1/n^2$) while cutting the variance, which is of order $1/n$, by a factor $(n-1)^2/(n+1)^2$ — a first-order saving bought with a second-order cost.` },
      { q: R`Give an estimator that is consistent but has infinite MSE for every $n$, and explain why this is possible.`,
        a: R`With $X_i\iid\N(\theta,1)$, $\bar X_n + 1/(nX_1)$. The perturbation is $O_p(1/n)$, so the estimator is consistent, but $1/X_1$ has no finite mean (the normal density is positive at 0). Consistency constrains where most of the probability lies; MSE is sensitive to rare, huge errors.` },
      { q: R`Why is there no unbiased estimator of $1/p$ from $X\sim\text{Binomial}(n,p)$? Does the same obstacle apply to $p(1-p)$?`,
        a: R`$\E_p t(X)$ is always a polynomial of degree at most $n$ in $p$, and $1/p$ is not (it is unbounded on $(0,1)$). The target $p(1-p)$ is a quadratic polynomial, so for $n\ge2$ an unbiased estimator does exist: $X(n-X)/(n(n-1))$.` },
      { q: R`For $X_i\iid\text{Unif}(0,\theta)$, compare the rates of convergence of $2\bar X$ and $X_{(n)}$. What feature of the model allows the faster rate?`,
        a: R`$2\bar X - \theta = O_p(n^{-1/2})$ with a normal limit, while $\theta - X_{(n)} = O_p(n^{-1})$ with an exponential limit. The support $[0,\theta]$ depends on the parameter, so the sample edge pins $\theta$ down sharply; regular models (support free of $\theta$, smooth densities) cannot beat $n^{-1/2}$.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'method-of-moments',
    domain: 'inference',
    order: 30,
    title: 'The Method of Moments',
    subtitle: 'Equate what the model predicts with what the sample shows, and solve',
    level: 1,
    sources: { aos: '§9.2', cb: '§7.2.1', other: 'Hansen (1982) for GMM; van der Vaart (1998), Asymptotic Statistics, Ch. 4' },
    prereqs: ['point-estimation', 'delta-method'],
    related: ['mle', 'statistical-functionals', 'lln', 'robust-statistics', 'exponential-families'],
    tags: ['moments', 'plug-in', 'estimating equations', 'GMM', 'efficiency'],
    summary: R`The method of moments (Pearson, 1894) estimates a $k$-dimensional parameter by solving the $k$ equations "population moment = sample moment". It needs no likelihood, only the ability to compute a few expectations, and the law of large numbers plus the delta method give its consistency and asymptotic normality almost for free. Its weaknesses are just as instructive: it can be markedly less efficient than maximum likelihood (the Gamma shape is the classic case), it can return values outside the parameter space, and it relies on higher moments that are noisy or nonexistent in heavy-tailed data. In exponential families, matching the moments of the sufficient statistic <em>is</em> maximum likelihood; elsewhere the method is best seen as a fast, robust-enough starting point and as the root of the general theory of estimating equations.`,
    survey: {
      task: R`Estimate $\theta\in\R^k$ from IID data`,
      assumptions: R`Moments $\mu_j(\theta)=\E_\theta X^j$, $j\le k$, exist and the map $\theta\mapsto(\mu_1,\dots,\mu_k)$ is invertible; finite $2k$-th moment for asymptotic normality`,
      output: R`$\tilde\theta = \mu^{-1}(\hat m_1,\dots,\hat m_k)$ with sandwich-type asymptotic variance`,
      strengths: 'Closed forms; no likelihood needed; good starting values; consistent under mild conditions',
      weaknesses: 'Often inefficient; may leave the parameter space; sensitive to outliers through high moments',
      complexity: R`$O(nk)$ to compute moments, plus solving a $k\times k$ system`,
    },
    sections: [
      {
        title: 'The idea: match moments',
        html: R`
<p>For $X\sim P_\theta$ with $\theta\in\Theta\subseteq\R^k$, the population moments $\mu_j(\theta) = \E_\theta X^j$ are known functions of $\theta$. The sample moments $\hat m_j = \frac1n\sum_i X_i^j$ converge to them by the <a href="#/t/lln">law of large numbers</a>. So it is natural to choose $\theta$ to make them agree.</p>
<div class="def" data-name="Method-of-moments estimator">
<p>The method-of-moments (MoM) estimator $\tilde\theta_n$ is a solution of the system
$$\mu_j(\theta) = \hat m_j,\qquad j=1,\dots,k.$$
More generally, for any $k$ functions $g_1,\dots,g_k$, one may solve $\E_\theta g_j(X) = \frac1n\sum_i g_j(X_i)$ (<em>generalized</em> moments). Central moments, e.g. $\Var_\theta X = \hat\sigma^2$, can replace raw ones since they are smooth functions of them.</p>
</div>
<div class="note">
<p>The method of moments is the <a href="#/t/statistical-functionals">plug-in principle</a> applied to a parameter defined implicitly: $\theta$ is the value whose moments equal those of the true $F$, so we replace $F$ by the empirical distribution $\hat F_n$ and solve. Everything we prove below follows from two facts about $\hat F_n$: its moments converge (LLN), and they fluctuate normally around their limits (CLT).</p>
</div>`,
      },
      {
        title: 'Worked examples',
        html: R`
<div class="ex" data-name="Normal">
<p>$\mu_1 = \mu$, $\mu_2 = \sigma^2+\mu^2$. Solving: $\tilde\mu=\bar X$, $\tilde\sigma^2 = \hat m_2 - \bar X^2 = \frac1n\sum(X_i-\bar X)^2$. Identical to the MLE.</p>
</div>
<div class="ex" data-name="Gamma">
<p>For shape $\alpha$ and scale $\beta$, $\E X = \alpha\beta$ and $\Var X = \alpha\beta^2$. With $\hat\sigma^2 = \frac1n\sum(X_i-\bar X)^2$,
$$\tilde\alpha = \frac{\bar X^2}{\hat\sigma^2},\qquad \tilde\beta = \frac{\hat\sigma^2}{\bar X}.$$
Closed form, whereas the MLE of $\alpha$ solves $\log\alpha - \psi(\alpha) = \log\bar X - \overline{\log X}$ numerically ($\psi$ = digamma). We will see the closed form costs a lot of efficiency when $\alpha$ is small.</p>
</div>
<div class="ex" data-name="Uniform(0, θ)">
<p>$\E X = \theta/2$, so $\tilde\theta = 2\bar X$. It is unbiased with variance $\theta^2/(3n)$. But it ignores the most informative feature of the data: it can be <em>smaller than the largest observation</em>, an estimate that the data themselves rule out. (Take $n=3$ and data $0.1, 0.1, 0.9$: $2\bar X\approx 0.73 \lt 0.9$.)</p>
</div>
<div class="ex" data-name="Overdispersed counts">
<p>A negative binomial with mean $\mu$ and size $r$ has $\Var X = \mu+\mu^2/r$. Matching mean and variance: $\tilde\mu = \bar X$, $\tilde r = \bar X^2/(\hat\sigma^2-\bar X)$. If the sample happens to be <em>under</em>dispersed ($\hat\sigma^2 \lt \bar X$), which occurs with substantial probability when $r$ is large or $n$ small, then $\tilde r \lt 0$: outside the parameter space. The MoM equations need not have a solution in $\Theta$.</p>
</div>`,
      },
      {
        title: 'Consistency and asymptotic normality',
        html: R`
<p>Write $\mu(\theta) = (\mu_1(\theta),\dots,\mu_k(\theta))$ and $\hat m = (\hat m_1,\dots,\hat m_k)$. If $\mu$ is a bijection from $\Theta$ onto an open set $M$ with inverse $g = \mu^{-1}$, then $\tilde\theta_n = g(\hat m)$ whenever $\hat m\in M$.</p>
<div class="thm" data-kind="Theorem" data-name="Asymptotics of the method of moments">
<p>Let $X_i\iid P_{\theta_0}$, $\theta_0$ interior. Suppose $g=\mu^{-1}$ is continuously differentiable in a neighbourhood of $\mu(\theta_0)$.</p>
<ol>
<li>If $\E|X|^k \lt \infty$, then $\tilde\theta_n\asto\theta_0$.</li>
<li>If $\E X^{2k} \lt \infty$, then $\sqrt n(\tilde\theta_n - \theta_0)\dto\N_k\big(0,\ G\Sigma G^\top\big)$, where $\Sigma = \Cov_{\theta_0}(X, X^2,\dots,X^k)$ and $G = \nabla g(\mu(\theta_0)) = [\nabla\mu(\theta_0)]^{-1}$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) By the strong law, $\hat m\asto\mu(\theta_0)$ coordinatewise. Since $M$ is open, $\hat m\in M$ eventually, almost surely, and $g$ is continuous at $\mu(\theta_0)$, so $g(\hat m)\asto g(\mu(\theta_0)) = \theta_0$.</p>
<p>(2) The vectors $Y_i = (X_i,\dots,X_i^k)$ are IID with mean $\mu(\theta_0)$ and covariance $\Sigma$ (finite because $\E X^{2k}<\infty$). The multivariate CLT gives $\sqrt n(\hat m - \mu(\theta_0))\dto\N_k(0,\Sigma)$. The multivariate <a href="#/t/delta-method">delta method</a> with the differentiable map $g$ yields the stated limit; the inverse function theorem gives $\nabla g(\mu(\theta_0)) = [\nabla\mu(\theta_0)]^{-1}$.</p>
</details>
<div class="ex" data-name="Gamma shape: the asymptotic variance">
<p>Because $\tilde\alpha = \bar X^2/\hat\sigma^2$ is scale-invariant, set $\beta=1$. Gamma$(\alpha,1)$ has central moments $\sigma^2 = \alpha$, $\kappa_3 = 2\alpha$, $\mu_4 = 3\alpha^2+6\alpha$, so $\sqrt n(\bar X - \alpha,\ \hat\sigma^2 - \alpha)$ is asymptotically normal with covariance
$$\begin{pmatrix}\alpha & 2\alpha\\ 2\alpha & \mu_4-\sigma^4\end{pmatrix} = \begin{pmatrix}\alpha & 2\alpha\\ 2\alpha & 2\alpha^2+6\alpha\end{pmatrix}.$$
The gradient of $(m,v)\mapsto m^2/v$ at $(\alpha,\alpha)$ is $(2,-1)$, so
$$n\,\Var(\tilde\alpha)\to 4\alpha - 2\cdot2\cdot2\alpha + (2\alpha^2+6\alpha) = 2\alpha(\alpha+1).$$</p>
</div>`,
      },
      {
        title: 'Efficiency: what does matching moments give up?',
        html: R`
<div class="def" data-name="Asymptotic relative efficiency">
<p>If $\sqrt n(\hth_{1,n} - \theta)\dto\N(0,v_1)$ and $\sqrt n(\hth_{2,n}-\theta)\dto\N(0,v_2)$, the <em>asymptotic relative efficiency</em> of $\hth_1$ to $\hth_2$ is $\text{ARE} = v_2/v_1$. An ARE of 0.5 means $\hth_1$ needs about twice as many observations to match $\hth_2$'s precision.</p>
</div>
<p>The <a href="#/t/mle">MLE</a> has asymptotic variance equal to the inverse <a href="#/t/fisher-information">Fisher information</a>, the smallest possible for regular estimators, so ARE(MoM : MLE) $\le 1$. For the Gamma shape with unknown scale the per-observation information matrix in $(\alpha,\beta)$ is
$$I(\alpha,\beta) = \begin{pmatrix}\psi'(\alpha) & 1/\beta\\ 1/\beta & \alpha/\beta^2\end{pmatrix},$$
whose inverse has $(1,1)$ entry $\alpha/(\alpha\psi'(\alpha)-1)$, with $\psi'$ the trigamma function. Therefore
$$\text{ARE}(\tilde\alpha:\hat\alpha_{\text{MLE}}) = \frac{\alpha/(\alpha\psi'(\alpha)-1)}{2\alpha(\alpha+1)} = \frac{1}{2(\alpha+1)\big(\alpha\psi'(\alpha)-1\big)}.$$
Using $\psi'(\alpha) = \alpha^{-1} + \tfrac12\alpha^{-2} + O(\alpha^{-3})$, the ARE tends to 1 as $\alpha\to\infty$ (the Gamma becomes normal, where MoM = MLE), and since $\psi'(\alpha)\sim\alpha^{-2}$ as $\alpha\to0$, the ARE tends to 0 for small shapes. At $\alpha=1$ (exponential data) it is $1/(4(\pi^2/6-1))\approx 0.39$: MoM wastes about 60% of the sample. This was the example Fisher (1922) used to argue for maximum likelihood.</p>
<p>Why the loss? The Gamma is an <a href="#/t/exponential-families">exponential family</a> with sufficient statistic $(\sum X_i, \sum\log X_i)$. The MLE matches the moments of the <em>sufficient</em> statistic; the MoM matches $\E X^2$ instead, which is not a function of the sufficient statistic and carries less information about $\alpha$ — especially when the distribution is very skewed.</p>
<div class="thm" data-kind="Proposition" data-name="When MoM is MLE">
<p>In a full-rank exponential family with natural sufficient statistic $T$, the MLE solves $\E_\eta T(X) = \frac1n\sum_i T(X_i)$. Hence the generalized method of moments using $g = T$ is exactly maximum likelihood.</p>
</div>
<p>The proof is one line from the score equation and is given in the <a href="#/t/exponential-families">exponential families</a> topic. The lab compares the two estimators for Gamma data by simulation and plots the theoretical ARE curve.</p>`,
        lab: 'method-of-moments-gamma',
      },
      {
        title: 'Estimating equations and the generalized method of moments',
        html: R`
<p>The method of moments is the simplest instance of a <em>Z-estimator</em>: $\hth$ solves $\frac1n\sum_i\phi(X_i;\theta) = 0$ for an estimating function with $\E_{\theta}\phi(X;\theta)=0$. For MoM, $\phi_j(x;\theta) = x^j - \mu_j(\theta)$; for the MLE, $\phi$ is the score. Under regularity, every such estimator satisfies
$$\sqrt n(\hth-\theta_0)\dto\N\big(0,\ A^{-1}BA^{-\top}\big),\quad A = \E\,\nabla_\theta\phi(X;\theta_0),\ \ B = \E\,\phi\phi^\top,$$
the <em>sandwich</em> form, valid even if the model is wrong provided the moment condition holds at $\theta_0$. This is also the language of <a href="#/t/robust-statistics">M-estimation</a>.</p>
<p>When there are more moment conditions than parameters ($m > k$), the equations generally have no exact solution. Hansen's (1982) <strong>generalized method of moments</strong> minimizes a quadratic form $\bar\phi(\theta)^\top W\bar\phi(\theta)$, with $\bar\phi(\theta) = \frac1n\sum_i\phi(X_i;\theta)\in\R^m$. With $G = \E\nabla_\theta\phi$ ($m\times k$) and $\Sigma = \E\phi\phi^\top$, the asymptotic variance is
$$(G^\top WG)^{-1}G^\top W\Sigma WG\,(G^\top WG)^{-1},$$
minimized by $W = \Sigma^{-1}$, giving $(G^\top\Sigma^{-1}G)^{-1}$. In practice $\Sigma$ is estimated from a first-stage fit (two-step GMM). The minimized objective, scaled by $n$, is asymptotically $\chi^2_{m-k}$ under correct specification — Hansen's $J$ test of overidentifying restrictions. Adding valid moments never hurts asymptotically; with the score included, GMM recovers full efficiency.</p>`,
      },
      {
        title: 'Pitfalls, and the one-step fix',
        html: R`
<div class="warn" data-name="Estimates outside the parameter space">
<p>MoM equations ignore constraints: negative variance components, negative dispersion sizes, a uniform endpoint below the sample maximum. Truncating to the boundary restores validity but not efficiency. The likelihood respects the support automatically.</p>
</div>
<div class="warn" data-name="Moments that do not exist, or barely exist">
<p>For Cauchy data no moment exists, and the method is meaningless. For a Student $t_4$, the variance exists but its MoM estimate has infinite variance (that needs $\E X^4 \lt \infty$, and $t_\nu$ has moments only of order below $\nu$). Rule of thumb: using the $j$-th moment requires $\E|X|^{2j} \lt \infty$ for root-$n$ behaviour, and even then high sample moments converge slowly and are dominated by a few extreme points.</p>
</div>
<div class="warn" data-name="Non-uniqueness">
<p>If $\theta\mapsto\mu(\theta)$ is not injective, the equations can have several solutions; e.g. matching three moments of a two-component mixture can yield multiple roots (Pearson's original crab-measurement fit led to a ninth-degree polynomial).</p>
</div>
<p>MoM estimates remain valuable as starting points. A classic result makes this precise.</p>
<div class="thm" data-kind="Theorem" data-name="One-step estimator (Le Cam)">
<p>Under the regularity conditions for MLE asymptotic normality, if $\tilde\theta_n$ is $\sqrt n$-consistent then one Newton step on the log-likelihood,
$$\hth^{(1)}_n = \tilde\theta_n - \frac{\ell_n'(\tilde\theta_n)}{\ell_n''(\tilde\theta_n)},$$
satisfies $\sqrt n(\hth^{(1)}_n-\theta_0)\dto\N(0,1/I(\theta_0))$: it is asymptotically as efficient as the MLE.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Expand $\ell_n'(\tilde\theta_n) = \ell_n'(\theta_0) + (\tilde\theta_n-\theta_0)\ell_n''(\theta_0) + O_p(n)(\tilde\theta_n-\theta_0)^2$ using a bounded third derivative. Since $\tilde\theta_n-\theta_0 = O_p(n^{-1/2})$ the last term is $O_p(1)$, and $\ell''_n(\tilde\theta_n)/n\pto -I(\theta_0)$. Substituting, $\sqrt n(\hth^{(1)}_n-\theta_0) = \frac{n^{-1/2}\ell_n'(\theta_0)}{I(\theta_0)} + o_p(1)$, whose limit is $\N(0,1/I)$ by the CLT for the score. The dependence on the starting value cancels to first order.</p>
</details>`,
      },
    ],
    quiz: [
      { q: R`Derive the MoM estimators for Beta$(a,b)$ using the mean $m$ and variance $v$. What constraint on $(m,v)$ is needed for a valid solution?`,
        a: R`$\E X = a/(a+b)$, $\Var X = m(1-m)/(a+b+1)$. So $a+b = m(1-m)/v - 1$, $\tilde a = m\,[m(1-m)/v-1]$, $\tilde b = (1-m)[m(1-m)/v-1]$, with $m,v$ replaced by $\bar X,\hat\sigma^2$. Validity requires $\hat\sigma^2 \lt \bar X(1-\bar X)$; otherwise $a+b\le0$.` },
      { q: R`Why is the MoM estimator of a Gamma shape so inefficient when $\alpha$ is small, while for normal data MoM equals the MLE?`,
        a: R`The Gamma's sufficient statistic involves $\sum\log X_i$, and the MLE matches its expectation; MoM uses $\sum X_i^2$ instead, which is not sufficient and, for highly skewed (small-$\alpha$) data, is dominated by a few large values. ARE $=1/[2(\alpha+1)(\alpha\psi'(\alpha)-1)]\to0$ as $\alpha\to0$. For the normal, the sufficient statistic is $(\sum X_i,\sum X_i^2)$, exactly the first two moments, so MoM and MLE coincide.` },
      { q: R`What moment condition does the delta-method argument need for $\sqrt n$-asymptotic normality of a MoM estimator built from the first $k$ moments, and why?`,
        a: R`$\E X^{2k}<\infty$. The CLT is applied to the vector $(X,\dots,X^k)$, whose covariance matrix contains $\Var(X^k) = \E X^{2k} - (\E X^k)^2$.` },
      { q: R`State the asymptotic variance of a GMM estimator with weight $W$ and the optimal choice of $W$.`,
        a: R`$(G^\top WG)^{-1}G^\top W\Sigma WG(G^\top WG)^{-1}$ with $G=\E\nabla\phi$, $\Sigma = \E\phi\phi^\top$. Choosing $W=\Sigma^{-1}$ gives the smallest variance, $(G^\top\Sigma^{-1}G)^{-1}$ (in the positive semidefinite order).` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'likelihood',
    domain: 'inference',
    order: 40,
    title: 'The Likelihood Function & the Likelihood Principle',
    subtitle: 'The probability of the data, read as a function of the parameter',
    level: 2,
    sources: { aos: '§9.3', cb: '§6.3', other: 'Royall (1997), Statistical Evidence; Birnbaum (1962); Pawitan (2001), In All Likelihood' },
    prereqs: ['statistical-models'],
    related: ['mle', 'sufficiency', 'fisher-information', 'bayesian-inference', 'likelihood-ratio-tests'],
    tags: ['likelihood', 'relative likelihood', 'likelihood principle', 'profile likelihood', 'evidence'],
    summary: R`Fix the observed data and let the parameter vary: the joint density $f(x;\theta)$ becomes the likelihood $L(\theta)$, a function that ranks parameter values by how well they predicted what actually happened. It is defined only up to a constant factor, it is not a probability distribution over $\theta$, and yet it is the common currency of almost all parametric inference — maximum likelihood, likelihood-ratio tests, Fisher information and Bayesian posteriors are all built from it. Log-likelihoods add across independent data, so the likelihood sharpens as data accumulate, and the information inequality guarantees it piles up at the true parameter. The likelihood principle — that two experiments yielding proportional likelihoods carry the same evidence — follows from two compelling premises (Birnbaum, 1962) and is violated by p-values and confidence intervals, which is why stopping rules matter to a frequentist and not to a Bayesian.`,
    survey: {
      task: R`Summarize what data say about $\theta$; basis for MLE, LRT, Bayes`,
      assumptions: R`A dominated parametric model with densities $f(x;\theta)$`,
      output: R`$L(\theta)\propto f(x;\theta)$; relative likelihood $L(\theta)/L(\hth)$; likelihood intervals; profile likelihoods`,
      strengths: 'Parameterization-invariant; additive across independent data; uses all the information in the model',
      weaknesses: 'Only as good as the model; nuisance parameters complicate it; not a probability over θ',
      complexity: R`$O(n)$ per evaluation of $\ell(\theta)$`,
    },
    sections: [
      {
        title: 'Definition and first properties',
        html: R`
<div class="def" data-name="Likelihood">
<p>Let $X$ have density (or pmf) $f(x;\theta)$, $\theta\in\Theta$. Having observed $X = x$, the <em>likelihood function</em> is $L(\theta) = L(\theta;x) = c(x)\,f(x;\theta)$ for any $c(x) > 0$ not depending on $\theta$; the <em>log-likelihood</em> is $\ell(\theta) = \log L(\theta)$. For $X_1,\dots,X_n$ independent with densities $f_i$,
$$L(\theta) = \prod_{i=1}^n f_i(x_i;\theta),\qquad \ell(\theta) = \sum_{i=1}^n\log f_i(x_i;\theta).$$</p>
</div>
<p>The arbitrary constant $c(x)$ is not a technicality. It says that only <em>ratios</em> $L(\theta_1)/L(\theta_2)$ carry meaning: "the data were $3.2$ times as probable under $\theta_1$ as under $\theta_2$." Several consequences follow.</p>
<ul>
<li><strong>Not a density in $\theta$.</strong> $\int L(\theta)\d\theta$ need not be 1, or even finite. Treating $L$ as a distribution requires a measure on $\Theta$ — which is exactly what a prior provides (<a href="#/t/bayesian-inference">Bayesian inference</a>).</li>
<li><strong>Invariant to reparameterization.</strong> If $\eta = g(\theta)$ with $g$ one-to-one, the likelihood for $\eta$ is $L^*(\eta) = L(g^{-1}(\eta))$ — the values are carried over with no Jacobian. Hence the maximizer transforms equivariantly and likelihood-based intervals transform correctly. Contrast a density in $\theta$, which would pick up $|dg^{-1}/d\eta|$.</li>
<li><strong>Invariant to transformation of the data.</strong> If $y = h(x)$ with $h$ smooth and one-to-one, $f_Y(y;\theta) = f_X(h^{-1}(y);\theta)\,|J(y)|$, and the Jacobian does not involve $\theta$. Whether you record times or log-times, the likelihood is the same up to the constant $c$.</li>
</ul>
<div class="note" data-name="Where the density comes from">
<p>For continuous data, the density likelihood is an approximation: an observation recorded to precision $\delta$ has probability $\P_\theta(X\in[x-\delta/2,x+\delta/2])\approx f(x;\theta)\,\delta$, and $\delta$ cancels in ratios. The approximation is excellent unless $f(x;\theta)$ varies on the scale of $\delta$ — which is precisely what happens when a mixture component collapses onto a single point, producing an <a href="#/t/mle">unbounded likelihood</a>. The exact interval-probability likelihood is always bounded by 1.</p>
</div>`,
      },
      {
        title: 'Reading a likelihood: ratios, intervals and curvature',
        html: R`
<p>Because only ratios matter, standardize by the maximum: the <strong>relative likelihood</strong> $R(\theta) = L(\theta)/L(\hth)\in[0,1]$, where $\hth$ is the <a href="#/t/mle">maximum likelihood estimate</a>. A <strong>likelihood interval</strong> at level $c$ is $\{\theta : R(\theta)\ge c\}$, the parameter values that explain the data at least a fraction $c$ as well as the best one.</p>
<p>Two calibrations of $c$ are common. Royall (1997) advocates fixed evidential benchmarks such as $c = 1/8$ ("fairly strong") and $c = 1/32$ ("strong"). Alternatively, Wilks' theorem (see <a href="#/t/likelihood-ratio-tests">likelihood-ratio tests</a>) says $-2\log R(\theta_0)\dto\chi^2_1$ under $\theta_0$ in regular one-parameter models, so $c = \exp(-\chi^2_{1,0.95}/2) = e^{-1.92}\approx 0.147$ gives an asymptotic 95% confidence interval.</p>
<p>Near its maximum, a smooth log-likelihood is approximately quadratic:
$$\ell(\theta)\approx\ell(\hth) - \tfrac12 J(\hth)(\theta-\hth)^2,\qquad J(\hth) = -\ell''(\hth).$$
The <strong>observed information</strong> $J(\hth)$ is the curvature at the peak: a sharply curved log-likelihood pins $\theta$ down, a flat one does not. Under the quadratic approximation the $c=0.147$ likelihood interval becomes the Wald interval $\hth\pm1.96/\sqrt{J(\hth)}$. See <a href="#/t/fisher-information">Fisher information</a> for the expected version of this curvature.</p>
<div class="ex" data-name="When the quadratic picture fails">
<p>One success in $n=20$ Bernoulli trials: $\hat p = 0.05$ and $J(\hat p) = n/(\hat p(1-\hat p))\approx 421$. The Wald interval is $0.05\pm 0.096 = (-0.046,\ 0.146)$, which includes impossible negative values. The likelihood interval $\{p: R(p)\ge0.147\}$ is $(0.0029,\ 0.202)$: asymmetric, inside $(0,1)$, and better calibrated — because the log-likelihood $\log p + 19\log(1-p)$ is far from quadratic near the boundary. Likelihood intervals inherit parameterization invariance; Wald intervals do not.</p>
</div>`,
      },
      {
        title: 'Accumulation: why the likelihood finds the truth',
        html: R`
<p>For independent observations, log-likelihoods add: $\ell_{n+1}(\theta) = \ell_n(\theta)+\log f(x_{n+1};\theta)$. Each new datum reshapes the curve by adding its own log-density, and the curve grows sharper roughly linearly in $n$. The following inequality is the reason it sharpens around the right place.</p>
<div class="thm" data-kind="Lemma" data-name="Information inequality">
<p>For any $\theta$, $\E_{\theta_0}\log f(X;\theta)\le\E_{\theta_0}\log f(X;\theta_0)$ whenever both sides are finite, with equality iff $f(\cdot;\theta) = f(\cdot;\theta_0)$ almost everywhere. Equivalently, $\KL(f_{\theta_0}\,\|\,f_\theta)\ge0$ with equality iff the distributions coincide.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $S_0 = \{x : f(x;\theta_0) > 0\}$. By Jensen's inequality for the strictly concave $\log$,
$$\E_{\theta_0}\log\frac{f(X;\theta)}{f(X;\theta_0)}\le\log\E_{\theta_0}\frac{f(X;\theta)}{f(X;\theta_0)} = \log\int_{S_0} f(x;\theta)\d x\le\log1 = 0.$$
Equality in Jensen requires the ratio to be a.s. constant on $S_0$, and equality in the last step requires $f_\theta$ to put all its mass on $S_0$; together these force $f_\theta = f_{\theta_0}$ a.e.</p>
</details>
<div class="thm" data-kind="Corollary" data-name="The truth wins, eventually and with high probability">
<p>Let $X_i\iid f(\cdot;\theta_0)$ and $\theta\neq\theta_0$ with $P_\theta\ne P_{\theta_0}$ (identifiability). Then (i) $\frac1n\log\frac{L_n(\theta_0)}{L_n(\theta)}\asto\KL(f_{\theta_0}\|f_\theta) > 0$, so $L_n(\theta_0) > L_n(\theta)$ for all large $n$, almost surely; and (ii) for every $n$ and $k>1$, $\P_{\theta_0}\big(L_n(\theta)/L_n(\theta_0)\ge k\big)\le 1/k$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(i) is the strong law applied to the IID log-ratios, whose mean is the KL divergence, positive by the lemma. (ii) The ratio $L_n(\theta)/L_n(\theta_0)$ is nonnegative with $\E_{\theta_0}$ at most $\prod_i\int_{S_0}f(x;\theta)\d x\le1$, so Markov's inequality applies. Note that (ii) holds for every $n$ and needs no regularity: the probability of being misled by a factor $k$ is never more than $1/k$ (the "universal bound").</p>
</details>
<p>Pointwise superiority over each fixed $\theta$ is not quite enough to make the <em>maximizer</em> converge — that needs a uniform version, developed in the <a href="#/t/mle">MLE</a> topic — but it is the heart of the argument. Watch it happen in the lab, including the Cauchy location model, whose likelihood can be multimodal for small samples.</p>`,
        lab: 'likelihood-accumulate',
      },
      {
        title: 'The likelihood principle',
        html: R`
<p>An <em>experiment</em> $E = (X, \{f(x;\theta)\})$ specifies the sample space and model; write $\text{Ev}(E,x)$ for the evidence about $\theta$ obtained by observing $x$ in $E$ (whatever inference you would draw).</p>
<div class="def" data-name="Three principles">
<ul>
<li><strong>Sufficiency principle (S).</strong> If $T$ is <a href="#/t/sufficiency">sufficient</a> in $E$ and $T(x)=T(y)$, then $\text{Ev}(E,x) = \text{Ev}(E,y)$.</li>
<li><strong>Weak conditionality principle (C).</strong> If $E^*$ consists of choosing $E_1$ or $E_2$ by a fair coin (independent of $\theta$) and then performing it, then $\text{Ev}(E^*,(j,x_j)) = \text{Ev}(E_j,x_j)$: experiments not performed are irrelevant.</li>
<li><strong>Likelihood principle (L).</strong> If $x_1$ in $E_1$ and $x_2$ in $E_2$ (same parameter $\theta$) have proportional likelihoods, $L_1(\theta;x_1) = c\,L_2(\theta;x_2)$ for all $\theta$, then $\text{Ev}(E_1,x_1) = \text{Ev}(E_2,x_2)$.</li>
</ul>
</div>
<div class="thm" data-kind="Theorem" data-name="Birnbaum (1962)">
<p>(S) and (C) together imply (L).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Let $x_1^*, x_2^*$ have proportional likelihoods in $E_1, E_2$. Form the mixture $E^*$ as in (C). In $E^*$ define $T(j,x_j)$ to equal a common label when $(j,x_j)\in\{(1,x_1^*),(2,x_2^*)\}$ and to equal $(j,x_j)$ otherwise. The likelihood of $E^*$ at $(j,x_j)$ is $\tfrac12 f_j(x_j;\theta)$, so given $T$ equals the common label, the conditional probabilities of $(1,x^*_1)$ and $(2,x^*_2)$ are $c/(1+c)$ and $1/(1+c)$, free of $\theta$: $T$ is sufficient. By (S), $\text{Ev}(E^*,(1,x_1^*)) = \text{Ev}(E^*,(2,x_2^*))$. By (C) each side equals $\text{Ev}(E_j,x_j^*)$.</p>
</details>
<div class="ex" data-name="Same likelihood, different p-values">
<p>Experimenter A plans $n=12$ Bernoulli trials and observes 3 successes. Experimenter B plans to continue until the 3rd success, which arrives on trial 12. Both likelihoods are proportional to $p^3(1-p)^9$, so by (L) the evidence is identical. Yet testing $H_0: p=1/2$ against $p \lt 1/2$:</p>
<ul>
<li>A (binomial): $\P(X\le3) = 299/4096\approx0.073$.</li>
<li>B (negative binomial): $\P(N\ge12) = \P(\le 2\text{ successes in 11 trials}) = 67/2048\approx0.033$.</li>
</ul>
<p>One rejects at 5%, the other does not, because p-values sum over outcomes that did not occur, and the two designs have different unobserved outcomes. Bayesian posteriors and likelihood intervals, which depend on the data only through $L$, are identical for A and B. The lab lets you vary the counts.</p>
</div>
<p>Frequentist procedures — p-values, confidence intervals, unbiasedness — violate (L), most visibly through their dependence on <strong>stopping rules</strong>. Bayesian inference satisfies (L) automatically. Whether this indicts frequentism or Birnbaum's premises is a long-running debate: (C) seems hard to reject, yet the derivation has been challenged (e.g. Evans, 2013; Mayo, 2014) on the grounds that (S) and (C) must be applied in a way that does not respect the structure of the mixed experiment. What is not in dispute is the practical point: under (L), optional stopping is harmless; under frequentist error control, it must be accounted for.</p>`,
        lab: 'likelihood-principle',
      },
      {
        title: 'Nuisance parameters: profile, marginal and conditional likelihoods',
        html: R`
<p>With $\theta = (\psi,\lambda)$ and interest in $\psi$ only, we need a function of $\psi$ alone.</p>
<div class="def" data-name="Profile likelihood">
<p>$L_p(\psi) = \sup_\lambda L(\psi,\lambda) = L(\psi,\hat\lambda_\psi)$, where $\hat\lambda_\psi$ maximizes over the nuisance with $\psi$ held fixed.</p>
</div>
<p>The profile is maximized at the full MLE $\hat\psi$, its curvature gives the correct asymptotic variance of $\hat\psi$ (the $(\psi,\psi)$ block of the inverse information), and Wilks' theorem holds for $-2\log[L_p(\psi_0)/L_p(\hat\psi)]$. But the profile treats $\hat\lambda_\psi$ as if it were known, and when there are many nuisance parameters it can be badly biased and overconfident.</p>
<div class="ex" data-name="Profile vs marginal likelihood for a normal variance">
<p>With $X_i\iid\N(\mu,\sigma^2)$, profiling out $\mu$ (set $\mu = \bar x$) gives $L_p(\sigma^2)\propto\sigma^{-n}\exp(-SS/2\sigma^2)$, maximized at $SS/n$. Alternatively, use only $SS$, whose distribution $\sigma^2\chi^2_{n-1}$ is free of $\mu$: the <em>marginal</em> likelihood $\propto\sigma^{-(n-1)}\exp(-SS/2\sigma^2)$ is maximized at $SS/(n-1)$. The marginal likelihood "charges" one degree of freedom for estimating $\mu$ — the idea behind REML in <a href="#/t/mixed-models">mixed models</a>. With many strata, each with its own mean, the gap is not a small-sample nuisance but an inconsistency: see Neyman–Scott in the <a href="#/t/mle">MLE</a> topic.</p>
</div>
<p><strong>Conditional likelihood</strong> conditions on a statistic that is sufficient for $\lambda$ when $\psi$ is fixed, removing $\lambda$ exactly — Fisher's exact test (conditioning on margins) and conditional logistic regression for matched pairs are examples. When no exact factorization exists, the <em>modified profile likelihood</em> (Barndorff-Nielsen, 1983) and the Cox–Reid (1987) adjustment for orthogonal parameters approximate it.</p>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Likelihood is not posterior probability">
<p>$R(\theta_1) = 0.5$ does not mean $\theta_1$ is "half as probable" as $\hth$. Probability statements about $\theta$ require a prior; integrating $L$ without one gives results that change with the parameterization (since $L$ has no Jacobian, $\int L(\theta)\d\theta \ne \int L^*(\eta)\d\eta$).</p>
</div>
<div class="warn" data-name="Comparing likelihoods across different data or models">
<p>Likelihood ratios compare parameter values given the same data and the same model of those data. Comparing $\max L$ across models with different numbers of parameters needs a penalty or a calibration (<a href="#/t/likelihood-ratio-tests">LRT</a>, <a href="#/t/model-selection">AIC/BIC</a>); comparing likelihoods for differently transformed responses requires the Jacobian, which is no longer a constant when the transformation itself is a model choice (e.g. Box–Cox).</p>
</div>
<div class="warn" data-name="The model is part of the likelihood">
<p>The likelihood measures relative support <em>within</em> the model. A sharply peaked likelihood under a wrong model is precise about the pseudo-true parameter (<a href="#/t/statistical-models">KL projection</a>), not about reality, and its curvature then misstates the sampling variability.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Two analysts record the same survival data, one in days and one in log-days. Do their likelihood functions for $\theta$ differ?`,
        a: R`Only by a factor that does not depend on $\theta$ (the Jacobian of the one-to-one transformation). Since the likelihood is defined up to such constants, they carry identical information: same MLE, same likelihood ratios, same intervals.` },
      { q: R`Why does $\P_{\theta_0}(L_n(\theta)/L_n(\theta_0)\ge k)\le1/k$ hold for every $n$ without regularity conditions?`,
        a: R`The likelihood ratio is nonnegative with $\E_{\theta_0}[L_n(\theta)/L_n(\theta_0)] = \int_{\{f_{\theta_0}>0\}} f_\theta\le1$, so Markov's inequality gives the bound directly.` },
      { q: R`In the binomial vs negative-binomial example (3 successes, 12 trials), which quantities are identical across the two designs and which differ?`,
        a: R`Identical: the likelihood ($\propto p^3(1-p)^9$), the MLE $1/4$, likelihood intervals, and any Bayesian posterior under the same prior. Different: p-values (0.073 vs 0.033 for $H_0: p=1/2$), confidence intervals, and unbiased estimators (e.g. $X/n$ is unbiased for the binomial, while $(r-1)/(N-1)$ is unbiased for the negative binomial), because they depend on unobserved outcomes in the sample space.` },
      { q: R`Why does the profile likelihood for $\sigma^2$ in a normal model peak at $SS/n$ while the marginal likelihood based on $SS$ peaks at $SS/(n-1)$?`,
        a: R`The profile plugs in $\hat\mu = \bar x$ as though it were the truth, ignoring that fitting $\mu$ absorbs one degree of freedom of variability. The marginal likelihood uses the distribution of $SS$, $\sigma^2\chi^2_{n-1}$, which accounts for that loss explicitly.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'sufficiency',
    domain: 'inference',
    order: 50,
    title: 'Sufficiency, Completeness & Ancillarity',
    subtitle: 'Compressing data without losing information about the parameter',
    level: 2,
    sources: { aos: '§9.13', cb: '§6.2', other: 'Lehmann & Casella (1998), §1.6; Basu (1955); Lehmann & Scheffé (1950)' },
    prereqs: ['likelihood', 'conditional-expectation'],
    related: ['exponential-families', 'rao-blackwell', 'mle', 'order-statistics'],
    tags: ['sufficient statistic', 'factorization', 'minimal sufficiency', 'completeness', 'ancillary', 'Basu'],
    summary: R`A statistic $T(X)$ is sufficient for $\theta$ when, once you know $T$, the rest of the data is pure noise whose distribution does not involve $\theta$. The factorization theorem turns this into a one-line check — the likelihood depends on the data only through $T$ — and minimal sufficient statistics compress as far as possible, which is exactly the information carried by the shape of the likelihood. Ancillary statistics are the opposite: their distribution is free of $\theta$. Completeness, the requirement that no nontrivial function of $T$ has mean zero for every $\theta$, is the technical key to uniqueness of unbiased estimators (Lehmann–Scheffé) and to Basu's theorem, which delivers independence results such as $\bar X\indep S^2$ with almost no computation.`,
    survey: {
      task: 'Reduce data to a low-dimensional summary that loses no information about θ',
      assumptions: R`A dominated model $\{f(x;\theta)\}$; conclusions are model-relative`,
      output: R`Sufficient / minimal sufficient statistic $T$; complete and ancillary statistics`,
      strengths: 'Exact data reduction; foundation for UMVUE theory, conditional inference, and independence results',
      weaknesses: 'Fragile to misspecification; outside exponential families little reduction is possible',
      complexity: '—',
    },
    sections: [
      {
        title: 'Data reduction without loss',
        html: R`
<div class="def" data-name="Sufficient statistic">
<p>A statistic $T = T(X)$ is <em>sufficient</em> for $\theta$ (in the model $\{P_\theta\}$) if the conditional distribution of $X$ given $T=t$ does not depend on $\theta$, for every $t$.</p>
</div>
<p>The operational meaning is striking. If you know $T$ but not $X$, you can generate a surrogate data set $X'$ by drawing from the ($\theta$-free) conditional law of $X$ given $T$. Then $X'$ has exactly the same distribution as $X$ under every $\theta$, so any procedure applied to $X'$ has exactly the same risk function as when applied to $X$. Nothing about $\theta$ has been lost; the discarded part of the data is an independent coin flip that a statistician could have made herself.</p>
<div class="ex" data-name="Bernoulli trials">
<p>For $X_1,\dots,X_n\iid\text{Bernoulli}(p)$ and $T = \sum X_i$, any sequence $x$ with $\sum x_i = t$ has $\P(X = x\mid T = t) = p^t(1-p)^{n-t}\big/\binom nt p^t(1-p)^{n-t} = 1/\binom nt$. Given the number of successes, every arrangement is equally likely whatever $p$ is: the order carries no information about $p$.</p>
</div>
<div class="ex" data-name="Poisson counts: what conditioning on T does">
<p>For $X_i\iid\text{Poisson}(\lambda)$ and $T = \sum X_i\sim\text{Poisson}(n\lambda)$, the conditional law of $(X_1,\dots,X_n)$ given $T=t$ is Multinomial$(t; 1/n,\dots,1/n)$, free of $\lambda$; in particular $X_1\mid T=t\sim\text{Binomial}(t,1/n)$. Conditioning instead on the maximum, which is not sufficient, leaves a conditional law that still depends on $\lambda$. The lab computes both exactly.</p>
</div>`,
        lab: 'sufficiency-conditional',
      },
      {
        title: 'The factorization theorem',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Fisher–Neyman factorization">
<p>Let $\{f(x;\theta)\}$ be densities with respect to a common $\sigma$-finite measure. $T$ is sufficient for $\theta$ if and only if there exist nonnegative functions $g$ and $h$ such that
$$f(x;\theta) = g\big(T(x);\theta\big)\,h(x)\qquad\text{for all } x,\theta.$$</p>
</div>
<details class="proof"><summary>Proof (discrete case)</summary>
<p>($\Leftarrow$) Suppose the factorization holds and $\P_\theta(T=t)>0$. For $x$ with $T(x)=t$,
$$\P_\theta(X=x\mid T=t) = \frac{g(t;\theta)h(x)}{\sum_{y:T(y)=t}g(t;\theta)h(y)} = \frac{h(x)}{\sum_{y:T(y)=t}h(y)},$$
free of $\theta$; for $T(x)\ne t$ it is 0.</p>
<p>($\Rightarrow$) If $T$ is sufficient, let $k(x) = \P(X=x\mid T=T(x))$, which does not depend on $\theta$. Then $f(x;\theta) = \P_\theta(X=x) = \P_\theta(T = T(x))\,k(x)$, which is the factorization with $g(t;\theta) = \P_\theta(T=t)$ and $h = k$.</p>
<p>The general (dominated) case is due to Halmos & Savage (1949); the proof replaces sums by conditional expectations with respect to a suitably chosen dominating probability measure.</p>
</details>
<p>An immediate and important corollary: <strong>if $T(x) = T(y)$, then $L(\theta;x)\propto L(\theta;y)$</strong>. Data sets with the same sufficient statistic produce the same likelihood function, up to a constant. So any likelihood-based method — MLE, likelihood ratios, posteriors — depends on the data only through a sufficient statistic.</p>
<table>
<thead><tr><th>Model ($n$ IID observations)</th><th>Sufficient statistic</th><th>Factorization</th></tr></thead>
<tbody>
<tr><td>$\N(\mu,\sigma^2)$</td><td>$(\bar X, SS)$</td><td>$\sum(x_i-\mu)^2 = SS + n(\bar x-\mu)^2$</td></tr>
<tr><td>Poisson$(\lambda)$</td><td>$\sum X_i$</td><td>$e^{-n\lambda}\lambda^{\sum x_i}\cdot\prod 1/x_i!$</td></tr>
<tr><td>Unif$(0,\theta)$</td><td>$X_{(n)}$</td><td>$\theta^{-n}\ind\{x_{(n)}\le\theta\}\cdot\ind\{x_{(1)}\ge0\}$</td></tr>
<tr><td>Unif$(\theta,\theta+1)$</td><td>$(X_{(1)},X_{(n)})$</td><td>$\ind\{\theta\le x_{(1)}\}\ind\{x_{(n)}\le\theta+1\}$</td></tr>
<tr><td>Any $f(x;\theta)$</td><td>order statistics</td><td>the product is symmetric in the $x_i$</td></tr>
<tr><td><a href="#/t/exponential-families">Exponential family</a></td><td>$\sum_i T(X_i)$</td><td>$\prod h(x_i)\cdot\exp\{\eta^\top\sum T(x_i) - nA(\eta)\}$</td></tr>
</tbody>
</table>
<p>The lab exploits the corollary: it constructs genuinely different data sets that share a sufficient statistic and shows their likelihoods coincide — and contrasts statistics that are not sufficient.</p>`,
        lab: 'sufficiency-same-likelihood',
      },
      {
        title: 'Minimal sufficiency',
        html: R`
<p>The whole data set is always sufficient, so sufficiency alone does not compress. We want the coarsest sufficient statistic.</p>
<div class="def" data-name="Minimal sufficient statistic">
<p>A sufficient statistic $T$ is <em>minimal</em> if it is a function of every other sufficient statistic: for any sufficient $U$, $U(x)=U(y)\Rightarrow T(x)=T(y)$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Likelihood-ratio criterion (Lehmann & Scheffé, 1950)">
<p>Suppose that for all sample points $x,y$:
$$f(x;\theta)/f(y;\theta)\ \text{is constant in }\theta\quad\Longleftrightarrow\quad T(x) = T(y).$$
Then $T$ is minimal sufficient.</p>
</div>
<details class="proof"><summary>Proof (assuming common support)</summary>
<p><em>Sufficiency.</em> For each value $t$ in the range of $T$ fix a representative $x_t$ with $T(x_t)=t$. For any $x$, by the "$\Leftarrow$" direction, $h(x) := f(x;\theta)/f(x_{T(x)};\theta)$ does not depend on $\theta$. So $f(x;\theta) = f(x_{T(x)};\theta)\,h(x) = g(T(x);\theta)h(x)$, and the factorization theorem applies.</p>
<p><em>Minimality.</em> Let $U$ be any sufficient statistic, with factorization $f(x;\theta) = g_U(U(x);\theta)h_U(x)$. If $U(x)=U(y)$ then $f(x;\theta)/f(y;\theta) = h_U(x)/h_U(y)$ is constant in $\theta$, so by "$\Rightarrow$" $T(x)=T(y)$. Hence $T$ is a function of $U$.</p>
</details>
<div class="ex" data-name="Applying the criterion">
<ul>
<li><strong>Normal:</strong> $\log[f(x;\mu,\sigma)/f(y;\mu,\sigma)] = -\frac{1}{2\sigma^2}\big(\sum x_i^2-\sum y_i^2\big)+\frac{\mu}{\sigma^2}\big(\sum x_i-\sum y_i\big)$, constant in $(\mu,\sigma)$ iff $\sum x_i = \sum y_i$ and $\sum x_i^2=\sum y_i^2$. So $(\bar X, SS)$ is minimal.</li>
<li><strong>Uniform$(\theta,\theta+1)$:</strong> the ratio is $\ind\{x_{(n)}-1\le\theta\le x_{(1)}\}/\ind\{y_{(n)}-1\le\theta\le y_{(1)}\}$, constant (where defined) iff the two intervals coincide. So $(X_{(1)},X_{(n)})$ is minimal: a two-dimensional summary for a one-dimensional parameter.</li>
<li><strong>Cauchy location:</strong> the ratio $\prod_i\frac{1+(y_i-\theta)^2}{1+(x_i-\theta)^2}$ is a ratio of degree-$2n$ polynomials in $\theta$; it is constant iff the polynomials have the same complex roots $\{x_i\pm \mathrm{i}\}$ and $\{y_i\pm\mathrm{i}\}$, i.e. iff the $y$'s are a permutation of the $x$'s. The order statistics are minimal: <em>no</em> reduction beyond discarding order is possible.</li>
</ul>
</div>
<div class="note">
<p>The criterion says that the minimal sufficient statistic is, in essence, the <strong>shape of the likelihood function</strong>: the map $x\mapsto L(\cdot\,;x)$ modulo multiplicative constants. A model admits a low-dimensional sufficient statistic exactly when its likelihood shapes form a low-dimensional family — which, for IID sampling with $\theta$-free support, essentially characterizes exponential families (Pitman–Koopman–Darmois).</p>
</div>`,
      },
      {
        title: 'Ancillarity and completeness',
        html: R`
<div class="def" data-name="Ancillary statistic">
<p>$V(X)$ is <em>ancillary</em> if its distribution does not depend on $\theta$.</p>
</div>
<p>In a location family $X_i = \theta+\varepsilon_i$, any function of the differences $X_i - X_j$ is ancillary (the range, the residuals $X_i-\bar X$, the sample variance). In a scale family $X_i = \sigma\varepsilon_i$, any function of the ratios $X_i/X_j$ is ancillary. An ancillary statistic alone tells you nothing about $\theta$ — but combined with others it can tell you <em>how much</em> the data tell you. In Uniform$(\theta,\theta+1)$ the range $R = X_{(n)}-X_{(1)}$ is ancillary, yet given $R$ close to 1 the sample pins $\theta$ down almost exactly ($\theta\in[X_{(n)}-1, X_{(1)}]$, an interval of length $1-R$). Fisher's conditionality argument says inference should condition on such an ancillary.</p>
<div class="def" data-name="Completeness">
<p>A statistic $T$ (or its family of distributions) is <em>complete</em> if, for every measurable $g$,
$$\E_\theta\,g(T) = 0\ \text{ for all }\theta\in\Theta\quad\Longrightarrow\quad\P_\theta(g(T)=0)=1\ \text{ for all }\theta.$$
It is <em>boundedly complete</em> if this holds for bounded $g$.</p>
</div>
<p>Completeness says the family $\{P_\theta^T\}$ is rich enough that the only unbiased estimator of zero based on $T$ is zero itself. Two consequences: unbiased estimators that are functions of $T$ are unique, and a complete sufficient statistic contains no ancillary "noise" (any nonconstant ancillary function of $T$ would give a nonzero $g$ with mean zero).</p>
<div class="ex" data-name="Proving completeness">
<ul>
<li><strong>Binomial.</strong> $T\sim\text{Bin}(n,p)$: $\E_p g(T) = (1-p)^n\sum_{k=0}^n g(k)\binom nk r^k$ with $r = p/(1-p)$ ranging over $(0,\infty)$. A polynomial in $r$ vanishing on an interval has all coefficients zero, so $g(k)=0$ for every $k$.</li>
<li><strong>Uniform maximum.</strong> $T = X_{(n)}$ from Unif$(0,\theta)$ has density $nt^{n-1}/\theta^n$ on $(0,\theta)$. If $\int_0^\theta g(t)t^{n-1}\d t=0$ for all $\theta>0$, then differentiating in $\theta$ (Lebesgue differentiation) gives $g(\theta)\theta^{n-1}=0$ for almost every $\theta$, so $g=0$ a.e.</li>
<li><strong>Not complete: Uniform$(\theta,\theta+1)$.</strong> The minimal sufficient $(X_{(1)},X_{(n)})$ has the ancillary function $R$, with $\E R = (n-1)/(n+1)$ for every $\theta$. Then $g = R - (n-1)/(n+1)$ is a nonzero function with mean zero.</li>
<li><strong>Full-rank exponential families</strong> are complete; see <a href="#/t/exponential-families">exponential families</a>.</li>
</ul>
</div>
<p>Complete sufficient statistics are automatically minimal whenever a minimal sufficient statistic exists (Bahadur, 1957). The converse fails, as the Uniform$(\theta,\theta+1)$ example shows.</p>`,
      },
      {
        title: "Basu's theorem",
        html: R`
<div class="thm" data-kind="Theorem" data-name="Basu (1955)">
<p>If $T$ is a boundedly complete sufficient statistic and $V$ is ancillary, then $T$ and $V$ are independent under every $P_\theta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Fix a measurable set $A$ and let $p_A = \P_\theta(V\in A)$, which is the same for every $\theta$ because $V$ is ancillary. Because $T$ is sufficient, the conditional probability $q_A(T) = \P(V\in A\mid T)$ does not depend on $\theta$, so $g(T) = q_A(T)-p_A$ is a genuine statistic. It is bounded, and $\E_\theta g(T) = \P_\theta(V\in A) - p_A = 0$ for all $\theta$. Bounded completeness gives $q_A(T) = p_A$ almost surely: the conditional law of $V$ given $T$ equals its marginal law, which is independence.</p>
</details>
<p>Basu's theorem converts a hard distributional fact into a structural one, and it is often the quickest route to an independence result or an expectation.</p>
<div class="ex" data-name="Independence of mean and variance">
<p>For $X_i\iid\N(\mu,\sigma^2)$, fix $\sigma=\sigma_0$ and regard only $\mu$ as unknown. Then $\bar X$ is complete sufficient (a full-rank one-parameter exponential family), and $S^2$ is ancillary (it is a function of the differences $X_i - X_j$, whose distribution does not involve $\mu$). Basu gives $\bar X\indep S^2$ under $\N(\mu,\sigma_0^2)$ for every $\mu$; since $\sigma_0$ was arbitrary, the independence holds for all $(\mu,\sigma^2)$. The same argument shows $\bar X$ is independent of the range, the MAD, and the vector of residuals.</p>
</div>
<div class="ex" data-name="An expectation for free">
<p>Let $X_i\iid\text{Exp}(\text{mean }\theta)$, $T = \sum X_i$ (complete sufficient), and $V = X_{(1)}/T$ (scale-invariant, hence ancillary). By Basu, $\E[X_{(1)}] = \E[V\,T] = \E V\cdot\E T$, so
$$\E\Big[\frac{X_{(1)}}{\sum X_i}\Big] = \frac{\E X_{(1)}}{\E T} = \frac{\theta/n}{n\theta} = \frac1{n^2},$$
using $X_{(1)}\sim\text{Exp}(\text{mean }\theta/n)$. Computing the ratio's distribution directly would be much harder.</p>
</div>
<div class="warn" data-name="Completeness is essential">
<p>In Uniform$(\theta,\theta+1)$ the minimal sufficient statistic $(X_{(1)},X_{(n)})$ is not independent of the ancillary range — the range is a function of it. Basu's conclusion needs completeness, not just minimality.</p>
</div>`,
      },
      {
        title: 'Sufficiency is model-relative',
        html: R`
<p>Every statement above is conditional on the model. If the model is wrong, reducing to "the" sufficient statistic can discard exactly the evidence that would reveal the problem.</p>
<div class="warn" data-name="Reduction hides misfit">
<p>Under a normal model, $(\bar X, SS)$ is sufficient, so skewness, outliers and bimodality are "irrelevant" — they live in the ancillary part. Model checking must therefore look at the data given $T$: residual plots, or exact goodness-of-fit tests that use the $\theta$-free conditional distribution of the data given $T$ (as in Fisher's exact test, or conditional tests of Poisson dispersion given $\sum X_i$).</p>
</div>
<div class="warn" data-name="Sufficiency is fragile">
<p>Contaminate a normal model with a small proportion $\varepsilon$ of outliers from another distribution, and the minimal sufficient statistic jumps from two numbers to the full order statistic. Low-dimensional sufficiency is a special property of <a href="#/t/exponential-families">exponential families</a>; the Pitman–Koopman–Darmois theorem says that, among IID families with $\theta$-free support and smooth densities, only exponential families admit a sufficient statistic of fixed dimension for every $n$.</p>
</div>
<p>The positive message: within a trusted model, sufficiency licenses compression (store $\sum X_i$, not the data), justifies conditioning (Rao–Blackwell, conditional tests), and underpins the optimality theory of <a href="#/t/rao-blackwell">UMVUEs</a>.</p>`,
      },
    ],
    quiz: [
      { q: R`For $X_i\iid\text{Unif}(-\theta,\theta)$, find a minimal sufficient statistic.`,
        a: R`$f(x;\theta) = (2\theta)^{-n}\ind\{\max_i|x_i|\le\theta\}$, so $T = \max_i|X_i|$ is sufficient by factorization, and the likelihood ratio $f(x;\theta)/f(y;\theta)$ is constant in $\theta$ iff $\max|x_i| = \max|y_i|$, so it is minimal.` },
      { q: R`Use Basu's theorem to show that for normal data $\bar X$ is independent of $\sum|X_i - \bar X|$.`,
        a: R`With $\sigma$ fixed, $\bar X$ is complete sufficient for $\mu$, and $\sum|X_i-\bar X|$ is a function of the differences $X_i - X_j$, whose joint law does not depend on $\mu$ (location-invariant), so it is ancillary. Basu gives independence for each $\sigma$, hence for all $(\mu,\sigma)$.` },
      { q: R`Why is the order statistic minimal sufficient for the Cauchy location model, and what does this imply about data reduction there?`,
        a: R`The likelihood ratio is a ratio of polynomials in $\theta$ with roots $x_i\pm\mathrm i$ and $y_i\pm\mathrm i$; it is constant only when the root sets agree, i.e. when $y$ is a permutation of $x$. So beyond discarding the order of the observations no reduction is possible without losing information.` },
      { q: R`Give a minimal sufficient statistic that is not complete, and the nonzero unbiased estimator of zero that shows it.`,
        a: R`Uniform$(\theta,\theta+1)$: $T=(X_{(1)},X_{(n)})$ is minimal sufficient, but $R = X_{(n)}-X_{(1)}$ is ancillary with $\E R=(n-1)/(n+1)$, so $g(T) = R-(n-1)/(n+1)$ has mean zero for every $\theta$ without being zero.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'exponential-families',
    domain: 'inference',
    order: 60,
    title: 'Exponential Families',
    subtitle: 'One convex function that generates moments, sufficiency, completeness and the MLE',
    level: 2,
    sources: { aos: '§9.13.3', cb: '§3.4', other: 'Brown (1986), Fundamentals of Statistical Exponential Families; Barndorff-Nielsen (1978)' },
    prereqs: ['sufficiency', 'mgf'],
    related: ['mle', 'glm-framework', 'fisher-information', 'bayesian-inference', 'entropy-kl', 'rao-blackwell'],
    tags: ['natural parameter', 'cumulant function', 'sufficient statistic', 'completeness', 'convexity', 'tilting'],
    summary: R`An exponential family writes every density as $h(x)\exp\{\eta^\top T(x) - A(\eta)\}$: a fixed base measure tilted by a linear function of a sufficient statistic $T$. Almost every textbook distribution — normal, Poisson, binomial, gamma, beta, multinomial — has this form, and the form explains why they are so tractable. The normalizer $A(\eta)$ is convex and is the cumulant generating function of $T$: its gradient is the mean, its Hessian is the covariance and the Fisher information. Consequently the log-likelihood is concave, the MLE solves "observed $\bar T$ = expected $T$", full-rank families have complete sufficient statistics (so UMVUEs and Basu's theorem apply), and KL divergences between members are Bregman divergences of $A$. They are the probabilistic backbone of GLMs, conjugate priors and maximum-entropy modelling.`,
    survey: {
      task: 'A unifying parametric class with closed-form sufficiency, moments, and likelihood geometry',
      assumptions: R`Support free of $\theta$; density of the form $h(x)e^{\eta^\top T(x)-A(\eta)}$`,
      output: R`Natural parameter $\eta$, cumulant function $A$, sufficient statistic $\sum T(X_i)$, mean map $\mu = \nabla A(\eta)$`,
      strengths: 'Concave log-likelihood; complete sufficient statistics; moments by differentiation; conjugacy',
      weaknesses: 'Excludes support-dependent, heavy-tailed location, and mixture models; MLE may not exist at the boundary',
      complexity: R`One pass to compute $\sum T(X_i)$; MLE by Newton on a concave function`,
    },
    sections: [
      {
        title: 'Definition and the standard members',
        html: R`
<div class="def" data-name="Exponential family">
<p>A family $\{f(x;\theta):\theta\in\Theta\}$ is a $k$-parameter <em>exponential family</em> if
$$f(x;\theta) = h(x)\,c(\theta)\exp\Big\{\sum_{j=1}^k w_j(\theta)T_j(x)\Big\}.$$
Reparameterizing by the <em>natural parameter</em> $\eta = (w_1(\theta),\dots,w_k(\theta))$ gives the <em>canonical form</em>
$$f(x;\eta) = h(x)\exp\{\eta^\top T(x) - A(\eta)\},\qquad A(\eta) = \log\int h(x)e^{\eta^\top T(x)}\d\mu(x).$$
$A$ is the <em>cumulant</em> (or log-partition) function. The representation is <em>minimal</em> if neither the $T_j$ nor the $\eta_j$ satisfy a linear constraint, and the family is <em>full rank</em> if, in a minimal representation, the parameter set contains a $k$-dimensional open set. If $\eta$ is confined to a lower-dimensional curve or surface, the family is <em>curved</em>.</p>
</div>
<p>Because $h(x)$ alone decides where the density is positive, the support of an exponential family does not depend on the parameter.</p>
<table>
<thead><tr><th>Family</th><th>$T(x)$</th><th>$\eta$</th><th>$A(\eta)$</th><th>$h(x)$</th></tr></thead>
<tbody>
<tr><td>Bernoulli$(p)$</td><td>$x$</td><td>$\log\frac{p}{1-p}$</td><td>$\log(1+e^\eta)$</td><td>1</td></tr>
<tr><td>Binomial$(m,p)$, $m$ known</td><td>$x$</td><td>$\log\frac{p}{1-p}$</td><td>$m\log(1+e^\eta)$</td><td>$\binom mx$</td></tr>
<tr><td>Poisson$(\lambda)$</td><td>$x$</td><td>$\log\lambda$</td><td>$e^\eta$</td><td>$1/x!$</td></tr>
<tr><td>Exponential (rate $\lambda$)</td><td>$x$</td><td>$-\lambda$</td><td>$-\log(-\eta)$</td><td>$\ind\{x>0\}$</td></tr>
<tr><td>$\N(\mu,\sigma^2)$</td><td>$(x,x^2)$</td><td>$\big(\frac{\mu}{\sigma^2},-\frac1{2\sigma^2}\big)$</td><td>$-\frac{\eta_1^2}{4\eta_2}-\frac12\log(-2\eta_2)$</td><td>$\frac1{\sqrt{2\pi}}$</td></tr>
<tr><td>Gamma (shape $\alpha$, rate $\beta$)</td><td>$(\log x, x)$</td><td>$(\alpha,-\beta)$</td><td>$\log\Gamma(\eta_1)-\eta_1\log(-\eta_2)$</td><td>$\frac1x\ind\{x>0\}$</td></tr>
<tr><td>Beta$(a,b)$</td><td>$(\log x,\log(1-x))$</td><td>$(a,b)$</td><td>$\log B(\eta_1,\eta_2)$</td><td>$\frac{1}{x(1-x)}$</td></tr>
</tbody>
</table>
<div class="warn" data-name="Not exponential families">
<p>Uniform$(0,\theta)$ and any model whose support moves with $\theta$; location families of the Cauchy, $t$, logistic or Laplace (their minimal sufficient statistics are the order statistics or nearly so); finite mixtures; the Weibull with unknown shape. The model $\N(\theta,\theta^2)$ <em>is</em> an exponential family, but a curved one: $\eta = (1/\theta, -1/(2\theta^2))$ traces a parabola in $\R^2$.</p>
</div>`,
      },
      {
        title: 'The natural parameter space and the cumulant function',
        html: R`
<div class="def" data-name="Natural parameter space">
<p>$\mathcal N = \{\eta\in\R^k : \int h(x)e^{\eta^\top T(x)}\d\mu(x)<\infty\}$, the set of all $\eta$ for which the canonical form defines a probability density.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Convexity">
<p>$\mathcal N$ is convex and $A$ is convex on $\mathcal N$. If the representation is minimal, $A$ is strictly convex.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\eta_1,\eta_2\in\mathcal N$, $\lambda\in(0,1)$. Hölder's inequality with exponents $1/\lambda$ and $1/(1-\lambda)$ applied to $e^{\lambda\eta_1^\top T}\cdot e^{(1-\lambda)\eta_2^\top T}$ under the measure $h\,\d\mu$ gives
$$\int h\,e^{(\lambda\eta_1+(1-\lambda)\eta_2)^\top T}\d\mu\le\Big(\int he^{\eta_1^\top T}\d\mu\Big)^{\lambda}\Big(\int he^{\eta_2^\top T}\d\mu\Big)^{1-\lambda}<\infty.$$
So the convex combination is in $\mathcal N$, and taking logs gives $A(\lambda\eta_1+(1-\lambda)\eta_2)\le\lambda A(\eta_1)+(1-\lambda)A(\eta_2)$. Equality in Hölder requires $e^{\eta_1^\top T}\propto e^{\eta_2^\top T}$ a.e., i.e. $(\eta_1-\eta_2)^\top T(x)$ constant a.e., which minimality rules out for $\eta_1\ne\eta_2$.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="A generates the cumulants of T">
<p>For $\eta$ in the interior of $\mathcal N$, the moment generating function of $T(X)$ under $f(\cdot;\eta)$ is $M_T(s) = \exp\{A(\eta+s)-A(\eta)\}$ for $s$ near 0. Hence $A$ is infinitely differentiable there and
$$\E_\eta T = \nabla A(\eta),\qquad \Cov_\eta T = \nabla^2A(\eta),$$
and more generally the $r$-th cumulants of $T$ are the $r$-th derivatives of $A$ at $\eta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$M_T(s) = \int e^{s^\top T}h\,e^{\eta^\top T - A(\eta)}\d\mu = e^{-A(\eta)}\int h\,e^{(\eta+s)^\top T}\d\mu = e^{A(\eta+s)-A(\eta)}$, finite for $\eta+s\in\mathcal N$, which holds for small $s$ when $\eta$ is interior. An MGF finite on a neighbourhood of 0 is analytic there, so $K(s) = A(\eta+s)-A(\eta)$, the cumulant generating function of $T$, is analytic, and its derivatives at $s=0$ — which are the derivatives of $A$ at $\eta$ — are the cumulants.</p>
</details>
<div class="ex" data-name="Reading off moments">
<p>Poisson: $A(\eta) = e^\eta$, so every derivative equals $e^\eta = \lambda$: all cumulants of a Poisson equal its mean. Bernoulli: $A'(\eta) = e^\eta/(1+e^\eta) = p$, $A''(\eta) = p(1-p)$. Normal: $\partial A/\partial\eta_1 = -\eta_1/(2\eta_2) = \mu = \E X$, and $\partial A/\partial\eta_2 = \eta_1^2/(4\eta_2^2) - 1/(2\eta_2) = \mu^2+\sigma^2 = \E X^2$.</p>
</div>
<p>Strict convexity makes the <strong>mean map</strong> $\eta\mapsto\mu(\eta) = \nabla A(\eta)$ one-to-one on the interior, with positive-definite Jacobian $\nabla^2A = \Cov T$. So a full-rank family can be parameterized equally well by its <em>mean-value parameter</em> $\mu = \E T$. The lab shows $A$ for several families: the slope of the tangent is the mean, its curvature is the variance.</p>`,
        lab: 'exponential-families-cumulant',
      },
      {
        title: 'Sufficiency, minimality and completeness',
        html: R`
<p>For an IID sample, the joint density is $\prod_ih(x_i)\cdot\exp\{\eta^\top S - nA(\eta)\}$ with $S = \sum_iT(X_i)$. By factorization $S$ is sufficient, whatever $n$ is: the dimension of the summary never grows. Moreover, $S$ itself has a density of exponential-family form, $h_n(s)\exp\{\eta^\top s-nA(\eta)\}$, with the same natural parameter.</p>
<div class="thm" data-kind="Theorem" data-name="Completeness of full-rank exponential families">
<p>If the natural parameter set contains a nonempty open subset of $\R^k$, then $S=\sum_iT(X_i)$ is complete (and minimal) sufficient.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Suppose $\E_\eta g(S)=0$ for all $\eta$ in an open set $U$. Writing $g = g^+-g^-$ and $\nu$ for the dominating measure of $S$ (incorporating $h_n$),
$$\int g^+(s)e^{\eta^\top s}\d\nu(s) = \int g^-(s)e^{\eta^\top s}\d\nu(s)\quad\text{for all }\eta\in U.$$
Fix $\eta_0\in U$ and divide both sides by their common value at $\eta_0$ (if it is zero, $g=0$ a.e. already). The two sides become moment generating functions of two probability measures, $\propto g^\pm e^{\eta_0^\top s}\d\nu$, that agree on a neighbourhood of 0. By uniqueness of MGFs (via analytic continuation to a characteristic function), the measures coincide, so $g^+=g^-$ $\nu$-a.e., i.e. $g(S)=0$ a.s. Minimality follows from the likelihood-ratio criterion, since $(\eta_1-\eta_2)^\top(s-s')=0$ for all $\eta_1,\eta_2$ in an open set forces $s=s'$.</p>
</details>
<p>This is the engine behind most textbook UMVUEs: in a full-rank family, any unbiased function of $S$ is the <a href="#/t/rao-blackwell">unique minimum-variance unbiased estimator</a>, and any ancillary statistic is independent of $S$ by <a href="#/t/sufficiency">Basu's theorem</a>.</p>
<div class="ex" data-name="Curved families lose completeness">
<p>For $X_i\iid\N(\theta,\theta^2)$, $\theta>0$, the minimal sufficient statistic is still $(\bar X, S^2)$, but the natural parameter lies on a curve, not an open set. Since $\E\bar X^2 = \theta^2(1+1/n)$ and $\E S^2 = \theta^2$, the statistic $\frac{n}{n+1}\bar X^2 - S^2$ has mean zero for every $\theta$ without being zero: $(\bar X, S^2)$ is not complete.</p>
</div>
<p>The converse direction is the <strong>Pitman–Koopman–Darmois theorem</strong>: if an IID family with support not depending on $\theta$ (and suitably smooth densities) admits a sufficient statistic whose dimension stays bounded as $n\to\infty$, it is an exponential family. Exponential families are exactly the models in which data can be compressed losslessly into a fixed number of sums.</p>`,
      },
      {
        title: 'Maximum likelihood: moment matching on a concave surface',
        html: R`
<p>The log-likelihood in the natural parameter is
$$\ell_n(\eta) = \eta^\top S - nA(\eta) + \text{const},\qquad \nabla\ell_n(\eta) = S - n\nabla A(\eta),\qquad \nabla^2\ell_n(\eta) = -n\nabla^2A(\eta) = -n\Cov_\eta T.$$
It is concave (strictly, for a minimal representation), so any stationary point is the unique global maximum, and Newton's method is reliable.</p>
<div class="thm" data-kind="Theorem" data-name="MLE in a full-rank exponential family">
<p>Let the family be regular ($\mathcal N$ open) and of full rank. The MLE $\hat\eta$ exists if and only if $\bar T = S/n$ lies in the interior of the convex hull of the support of $T$; it is then unique and solves the moment equation
$$\nabla A(\hat\eta) = \bar T,\qquad\text{i.e.}\qquad \E_{\hat\eta}T(X) = \frac1n\sum_iT(X_i).$$
</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Uniqueness and the moment equation follow from strict concavity and the gradient formula. For existence, the range of the mean map $\nabla A$ over the interior of $\mathcal N$ is, for a regular family (one with $\mathcal N$ open), exactly the interior of the convex support of $T$ (Barndorff-Nielsen, 1978). If $\bar T$ is in that set, some $\eta$ solves the equation; if $\bar T$ is on the boundary, no $\eta$ does: $\ell_n$ keeps increasing along some ray toward its supremum without ever attaining it.</p>
</details>
<div class="ex" data-name="When the MLE runs away">
<p>Bernoulli with all $n$ observations equal to 0: $\bar T = 0$, the boundary of $[0,1]$, and $\ell(\eta) = -n\log(1+e^\eta)$ increases as $\eta\to-\infty$. The MLE of $p$ is 0 (in the closure), but $\hat\eta = -\infty$ does not exist. In logistic regression this is <em>separation</em>: if a hyperplane perfectly separates successes from failures, the sufficient statistic lies on the boundary of its convex support and the coefficients diverge. The lab marks this case.</p>
</div>
<p>Two further gifts. The observed information equals the expected information, $-\nabla^2\ell_n(\eta) = n\nabla^2A(\eta)$, since it involves no data. And in the mean parameterization the information is $n[\Cov T]^{-1}$, so $\bar T$, which is unbiased for $\mu = \E T$ with covariance $\Cov(T)/n$, attains the <a href="#/t/fisher-information">Cramér–Rao bound</a> exactly. That is not a coincidence: the bound is attained only in exponential families, only for the mean-value parameter.</p>`,
      },
      {
        title: 'Tilting, maximum entropy and divergence',
        html: R`
<p>Start from any base density $h$ and any statistic $T$. The family $f_\eta(x) = h(x)e^{\eta T(x)-A(\eta)}$ is obtained by <em>exponential tilting</em>: reweighting $h$ by $e^{\eta T}$ and renormalizing. Raising $\eta$ shifts mass toward large $T$, and by the cumulant theorem the mean $\E_\eta T = A'(\eta)$ increases strictly. Tilting appears in importance sampling, large deviations (Cramér's theorem uses the tilted law to make a rare event typical), and saddlepoint approximations.</p>
<div class="thm" data-kind="Proposition" data-name="Exponential families maximize entropy">
<p>Among all densities $p$ with $\E_pT = t$, the KL divergence $\KL(p\,\|\,h)$ is minimized by the exponential-family member $f_\eta$ with $\nabla A(\eta) = t$ (when such $\eta$ exists). With $h$ uniform (or Lebesgue), this says $f_\eta$ maximizes entropy subject to the moment constraint.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For any such $p$, $\log(p/h) = \log(p/f_\eta) + \eta^\top T - A(\eta)$. Taking $\E_p$:
$$\KL(p\,\|\,h) = \KL(p\,\|\,f_\eta) + \eta^\top t - A(\eta).$$
The last two terms are the same for every admissible $p$, and $\KL(p\|f_\eta)\ge0$ with equality iff $p=f_\eta$.</p>
</details>
<p>So the normal is the maximum-entropy law with given mean and variance, the exponential is the one on $(0,\infty)$ with given mean, and so on. The same computation gives a closed form for divergence between members:
$$\KL(f_{\eta_1}\,\|\,f_{\eta_2}) = A(\eta_2) - A(\eta_1) - (\eta_2-\eta_1)^\top\nabla A(\eta_1),$$
the <strong>Bregman divergence</strong> of $A$: the gap between $A$ and its tangent plane at $\eta_1$, evaluated at $\eta_2$. Its second-order expansion is $\tfrac12(\eta_2-\eta_1)^\top\nabla^2A(\eta_1)(\eta_2-\eta_1)$, recovering <a href="#/t/fisher-information">Fisher information</a> as the local metric. The lab tilts an arbitrary base density.</p>`,
        lab: 'exponential-families-tilting',
      },
      {
        title: 'Where exponential families reappear',
        html: R`
<ul>
<li><strong>Generalized linear models.</strong> A GLM puts the natural parameter of each response in an exponential (dispersion) family equal to a linear predictor, $\eta_i = x_i^\top\beta$ for the canonical link. Concavity, IRLS, deviance and $X^\top y$ as sufficient statistic all descend from this section: see <a href="#/t/glm-framework">the GLM framework</a> and <a href="#/t/irls">IRLS</a>.</li>
<li><strong>Conjugate priors.</strong> The prior $\pi(\eta)\propto\exp\{\eta^\top\tau - n_0A(\eta)\}$ is closed under updating: the posterior replaces $(\tau,n_0)$ by $(\tau+S, n_0+n)$, so the prior acts like $n_0$ pseudo-observations with total sufficient statistic $\tau$. See <a href="#/t/bayesian-inference">Bayesian inference</a>.</li>
<li><strong>Graphical models.</strong> Gaussian graphical models and Ising/Potts models are exponential families whose sufficient statistics are pairwise products; zero natural parameters encode conditional independence (<a href="#/t/undirected-graphical-models">undirected graphical models</a>).</li>
<li><strong>Variational inference</strong> exploits the conjugate duality between $A$ and the negative entropy (Wainwright & Jordan, 2008).</li>
</ul>
<div class="warn" data-name="Mind the boundary">
<p>Most pathologies of exponential families sit on the boundary of the natural parameter space: nonexistent MLEs (zero counts, separation), information that degenerates as $p\to0$ or $1$, and non-regular behaviour when $\mathcal N$ is not open (e.g. families with $\mathcal N$ closed at an endpoint where moments of $T$ fail to exist).</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Put the geometric distribution $f(x;p) = (1-p)^{x}p$, $x=0,1,2,\dots$, in canonical form and use $A$ to find its mean and variance.`,
        a: R`$f = \exp\{x\log(1-p)+\log p\}$, so $T=x$, $\eta = \log(1-p)\lt 0$, and $A(\eta) = -\log p = -\log(1-e^\eta)$. Then $A'(\eta) = e^\eta/(1-e^\eta) = (1-p)/p$ and $A''(\eta) = e^\eta/(1-e^\eta)^2 = (1-p)/p^2$.` },
      { q: R`Why is the log-likelihood of a full-rank exponential family concave in $\eta$, and what does the MLE equation say in words?`,
        a: R`$\ell_n(\eta) = \eta^\top S - nA(\eta)$ is linear minus a convex function ($A$ is convex by Hölder, with Hessian $\Cov_\eta T\succeq0$). Setting the gradient to zero gives $\nabla A(\hat\eta) = \bar T$: choose the member of the family whose expected sufficient statistic equals the observed average.` },
      { q: R`Show that the $\N(\theta,\theta^2)$ family is not complete. Which hypothesis of the completeness theorem fails?`,
        a: R`$\E[\tfrac{n}{n+1}\bar X^2 - S^2] = \tfrac{n}{n+1}\theta^2(1+\tfrac1n)-\theta^2 = 0$ for all $\theta$, yet the statistic is not identically 0. The natural parameter $(1/\theta, -1/(2\theta^2))$ lies on a curve, so the parameter set contains no open subset of $\R^2$.` },
      { q: R`Express $\KL(\text{Poisson}(\lambda_1)\,\|\,\text{Poisson}(\lambda_2))$ using the Bregman formula.`,
        a: R`With $A(\eta)=e^\eta$ and $\eta_j = \log\lambda_j$: $\KL = e^{\eta_2}-e^{\eta_1}-(\eta_2-\eta_1)e^{\eta_1} = \lambda_2 - \lambda_1 - \lambda_1\log(\lambda_2/\lambda_1) = \lambda_1\log(\lambda_1/\lambda_2) + \lambda_2 - \lambda_1$.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'mle',
    domain: 'inference',
    order: 70,
    title: 'Maximum Likelihood Estimation',
    subtitle: 'The parameter value that made the data most probable, and why it is usually the right default',
    level: 2,
    sources: { aos: '§9.3–9.8', cb: '§7.2.2, §10.1', other: 'van der Vaart (1998), Asymptotic Statistics, Ch. 5; Ferguson (1996), A Course in Large Sample Theory' },
    prereqs: ['likelihood', 'clt'],
    related: ['fisher-information', 'exponential-families', 'em-algorithm', 'likelihood-ratio-tests', 'delta-method', 'irls', 'parametric-bootstrap'],
    tags: ['MLE', 'consistency', 'asymptotic normality', 'invariance', 'sandwich', 'Neyman–Scott'],
    summary: R`The maximum likelihood estimator $\hth = \argmax_\theta L(\theta)$ is the default estimator of parametric statistics, and for good reason: it is equivariant under reparameterization, consistent under mild conditions (because the expected log-likelihood is maximized at the truth), and, in regular models, asymptotically normal with variance equal to the inverse Fisher information — the smallest achievable. Its standard error comes free from the curvature of the log-likelihood. The same theory, read carefully, also says when it fails: when the support moves with $\theta$, when the parameter sits on the boundary, when the number of parameters grows with $n$ (Neyman–Scott), and when the likelihood is unbounded (normal mixtures). Under misspecification it still converges — to the KL projection — but its variance is the sandwich, not the inverse information.`,
    survey: {
      task: R`Estimate $\theta$ in a parametric model by maximizing $\ell_n(\theta) = \sum\log f(X_i;\theta)$`,
      assumptions: R`Identifiable model; for asymptotic normality: $\theta$-free support, smooth log-density, interior $\theta_0$, $0 \lt I(\theta_0) \lt \infty$`,
      output: R`$\hth$, with $\widehat{\se} = 1/\sqrt{J_n(\hth)}$; $\sqrt n(\hth-\theta_0)\dto\N(0,I(\theta_0)^{-1})$`,
      strengths: 'Equivariant; asymptotically efficient; automatic standard errors; general recipe',
      weaknesses: 'Can be biased in small samples; fails with incidental parameters, unbounded likelihoods, boundaries; model-dependent SEs',
      complexity: R`Iterative optimization; $O(nk^2)$ per Newton step for $k$ parameters`,
    },
    sections: [
      {
        title: 'Definition, computation and first examples',
        html: R`
<div class="def" data-name="Maximum likelihood estimator">
<p>Given data $x$ and a model $\{f(x;\theta):\theta\in\Theta\}$, a <em>maximum likelihood estimate</em> is any $\hth\in\Theta$ with $L(\hth;x) = \sup_{\theta\in\Theta}L(\theta;x)$. When $\ell$ is differentiable and the maximum is interior, $\hth$ solves the <em>likelihood (score) equation</em> $\ell'(\theta)=0$.</p>
</div>
<div class="ex" data-name="Four shapes of a maximization">
<ul>
<li><strong>Normal, both parameters.</strong> $\ell(\mu,\sigma^2) = -\frac n2\log\sigma^2-\frac{1}{2\sigma^2}\big[SS + n(\bar x-\mu)^2\big]$. Maximize in $\mu$ first ($\hat\mu = \bar x$ for any $\sigma^2$), then in $\sigma^2$: $\hat\sigma^2 = SS/n$, biased downward by the factor $(n-1)/n$.</li>
<li><strong>Uniform$(0,\theta)$.</strong> $L(\theta) = \theta^{-n}\ind\{\theta\ge x_{(n)}\}$ is decreasing where positive, so $\hth = X_{(n)}$ — a boundary maximum, not a root of the score equation.</li>
<li><strong>Laplace location.</strong> $\ell(\theta) = -\sum|x_i-\theta|$ is piecewise linear; it is maximized by the sample median, which is not unique for even $n$.</li>
<li><strong>Gamma shape.</strong> With unknown shape $\alpha$ and rate $\beta$, profiling out $\beta = \alpha/\bar x$ leaves $\log\alpha-\psi(\alpha) = \log\bar x - \overline{\log x}$, which has a unique root (the left side decreases from $\infty$ to 0) but no closed form.</li>
</ul>
</div>
<p>Numerically, the standard tool is <strong>Newton–Raphson</strong>, $\theta^{(t+1)} = \theta^{(t)} - [\nabla^2\ell(\theta^{(t)})]^{-1}\nabla\ell(\theta^{(t)})$, or <strong>Fisher scoring</strong>, which replaces the observed Hessian by its expectation $-nI(\theta^{(t)})$ and is more stable far from the optimum (in GLMs it is <a href="#/t/irls">IRLS</a>). Latent-variable models use the <a href="#/t/em-algorithm">EM algorithm</a>. Outside concave problems, the likelihood may have several local maxima — the Cauchy location model is the classic example — so start from a consistent estimator (a <a href="#/t/method-of-moments">moment</a> or quantile estimator) or from several points.</p>`,
      },
      {
        title: 'Invariance',
        html: R`
<p>Suppose we want the MLE of $\tau = g(\theta)$ for a function $g$ that need not be one-to-one (e.g. $\tau = \sigma$ from $(\mu,\sigma^2)$, or $\tau = \P_\theta(X>c)$). Define the <em>induced likelihood</em> $L^*(\tau) = \sup_{\{\theta:\,g(\theta)=\tau\}}L(\theta)$ — a profile likelihood for $\tau$.</p>
<div class="thm" data-kind="Theorem" data-name="Invariance of the MLE (Zehna, 1966)">
<p>If $\hth$ is an MLE of $\theta$, then $g(\hth)$ maximizes the induced likelihood $L^*$, i.e. it is an MLE of $\tau=g(\theta)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For any $\tau$, $L^*(\tau) = \sup_{g(\theta)=\tau}L(\theta)\le\sup_\theta L(\theta) = L(\hth)$. And $L(\hth)\le\sup_{g(\theta)=g(\hth)}L(\theta) = L^*(g(\hth))$, because $\hth$ is one of the $\theta$'s in that set. So $L^*(g(\hth)) = L(\hth)\ge L^*(\tau)$ for all $\tau$.</p>
</details>
<p>Thus the MLE of $\sigma$ is $\sqrt{SS/n}$, the MLE of a normal tail probability $\P(X>c)$ is $1-\Phi\big((c-\bar x)/\hat\sigma\big)$, and the MLE of an odds $p/(1-p)$ is $\hat p/(1-\hat p)$. No other common estimation principle is equivariant in this complete way: unbiasedness is destroyed by nonlinear $g$ (<a href="#/t/point-estimation">point estimation</a>), and a posterior mean of $g(\theta)$ is not $g$ of the posterior mean. The price of equivariance is that the MLE cannot be simultaneously unbiased in all parameterizations; its small-sample bias depends on which one you care about.</p>`,
      },
      {
        title: 'Consistency',
        html: R`
<p>The <a href="#/t/likelihood">information inequality</a> showed that the normalized log-likelihood $M_n(\theta) = \frac1n\ell_n(\theta)$ converges pointwise to $M(\theta) = \E_{\theta_0}\log f(X;\theta)$, uniquely maximized at $\theta_0$. Consistency of the maximizer needs this convergence to be uniform and the maximum to be well separated.</p>
<div class="thm" data-kind="Theorem" data-name="Consistency of the MLE (Wald, 1949; argmax form)">
<p>Let $X_i\iid f(\cdot;\theta_0)$ and suppose: (i) the model is identifiable; (ii) $\Theta$ is compact; (iii) $\theta\mapsto\log f(x;\theta)$ is continuous for every $x$; (iv) $\E_{\theta_0}\sup_{\theta\in\Theta}|\log f(X;\theta)| \lt \infty$. Then any MLE $\hth_n$ satisfies $\hth_n\asto\theta_0$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Step 1 (uniform LLN).</em> Conditions (ii)–(iv) imply $\sup_\theta|M_n(\theta)-M(\theta)|\asto0$ and that $M$ is continuous (a standard bracketing argument: cover $\Theta$ by finitely many small balls, on each of which $\sup$ and $\inf$ of $\log f$ have means close to each other by dominated convergence).</p>
<p><em>Step 2 (well-separated maximum).</em> By the information inequality and (i), $M(\theta) \lt M(\theta_0)$ for $\theta\ne\theta_0$. For $\varepsilon>0$ the set $\{|\theta-\theta_0|\ge\varepsilon\}$ is compact and $M$ is continuous, so $\sup_{|\theta-\theta_0|\ge\varepsilon}M(\theta) = M(\theta_0)-\delta$ for some $\delta>0$.</p>
<p><em>Step 3.</em> Write $\Delta_n = \sup|M_n-M|$. Then $M(\hth_n)\ge M_n(\hth_n)-\Delta_n\ge M_n(\theta_0)-\Delta_n\ge M(\theta_0)-2\Delta_n$. Once $2\Delta_n \lt \delta$ (eventually, a.s.), this forces $|\hth_n-\theta_0| \lt \varepsilon$.</p>
</details>
<p>Compactness can be replaced by concavity: if $\ell_n$ is concave (as in <a href="#/t/exponential-families">exponential families</a> in the natural parameter), pointwise convergence of concave functions is automatically uniform on compacts, and concavity prevents the maximizer from escaping to infinity. Without some such control, things can go wrong even in smooth, identifiable, one-dimensional models: Ferguson (1982) constructed one whose MLE is inconsistent. Most practical failures, though, come from the specific mechanisms in the last section.</p>`,
      },
      {
        title: 'Asymptotic normality and efficiency',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Asymptotic normality (Cramér, 1946)">
<p>Let $X_i\iid f(\cdot;\theta_0)$, $\theta\in\Theta\subseteq\R$, and assume:</p>
<ol>
<li>$\Theta$ is open and contains $\theta_0$; the support $\{x: f(x;\theta)>0\}$ does not depend on $\theta$;</li>
<li>$\log f(x;\theta)$ is three times continuously differentiable in $\theta$ for every $x$;</li>
<li>$\int f(x;\theta)\d x$ can be differentiated twice under the integral sign;</li>
<li>$0 \lt I(\theta_0) = \E_{\theta_0}[\partial_\theta\log f(X;\theta_0)]^2 \lt \infty$;</li>
<li>$|\partial^3_\theta\log f(x;\theta)|\le K(x)$ for $\theta$ in a neighbourhood of $\theta_0$, with $\E_{\theta_0}K(X)<\infty$.</li>
</ol>
<p>Then every consistent sequence of roots $\hth_n$ of the likelihood equation satisfies
$$\sqrt n(\hth_n-\theta_0)\dto\N\big(0,\ I(\theta_0)^{-1}\big).$$
For $\theta\in\R^k$ the same holds with the information matrix: $\sqrt n(\hth_n-\theta_0)\dto\N_k(0, I(\theta_0)^{-1})$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Expand the score around $\theta_0$ with Lagrange remainder: for some $\theta_n^*$ between $\hth_n$ and $\theta_0$,
$$0 = \ell_n'(\hth_n) = \ell_n'(\theta_0) + (\hth_n-\theta_0)\ell_n''(\theta_0) + \tfrac12(\hth_n-\theta_0)^2\ell_n'''(\theta^*_n).$$
Rearranging,
$$\sqrt n(\hth_n-\theta_0) = \frac{n^{-1/2}\ell_n'(\theta_0)}{-n^{-1}\ell_n''(\theta_0) - \tfrac12(\hth_n-\theta_0)\,n^{-1}\ell_n'''(\theta^*_n)}.$$
<em>Numerator:</em> the score $\ell_n'(\theta_0)$ is a sum of IID terms with mean 0 and variance $I(\theta_0)$ (by condition 3; see <a href="#/t/fisher-information">Fisher information</a>), so the CLT gives $\N(0,I(\theta_0))$. <em>Denominator, first term:</em> by the LLN and the information identity, $-n^{-1}\ell_n''(\theta_0)\pto I(\theta_0)$. <em>Second term:</em> on the event $\{\theta^*_n$ in the neighbourhood$\}$, whose probability tends to 1 by consistency, $|n^{-1}\ell_n'''(\theta_n^*)|\le n^{-1}\sum K(X_i)\pto\E K<\infty$, and $\hth_n-\theta_0\pto0$, so this term $\pto0$. Slutsky's theorem gives the limit $\N(0,I)/I = \N(0,1/I)$.</p>
</details>
<p>In practice, $\widehat\se(\hth) = 1/\sqrt{nI(\hth)}$ (expected information) or $1/\sqrt{J_n(\hth)}$ (observed information, $J_n = -\ell_n''$). Both are consistent; the observed version adapts to the realized curvature and is often more accurate. For a function $g(\theta)$, the <a href="#/t/delta-method">delta method</a> gives $\widehat\se(g(\hth)) = |g'(\hth)|\,\widehat\se(\hth)$. Wald intervals $\hth\pm z_{\alpha/2}\widehat\se$ follow; likelihood-ratio intervals are usually better in small samples (<a href="#/t/likelihood-ratio-tests">LRT</a>).</p>
<div class="note" data-name="Why this is optimal">
<p>The asymptotic variance $1/I(\theta_0)$ equals the <a href="#/t/fisher-information">Cramér–Rao bound</a> for unbiased estimators. Asymptotically, no "regular" estimator can do better: by Hájek's convolution theorem (1970), the limit law of any regular estimator is $\N(0,I^{-1})$ convolved with some other distribution. Superefficient estimators that beat $1/I$ at isolated points exist (Hodges) but pay with terrible risk nearby.</p>
</div>
<p>The lab checks the theorem by simulation — and lets you break it: try the Uniform (moving support), a Bernoulli $p$ near 0 with small $n$, or the Cauchy (multimodal likelihood but still regular).</p>`,
        lab: 'mle-asymptotics',
      },
      {
        title: 'Misspecification and the sandwich',
        html: R`
<p>If the data come from a density $p$ outside the model, the <a href="#/t/statistical-models">pseudo-true value</a> $\theta^* = \argmax_\theta\E_p\log f(X;\theta)$ takes the place of $\theta_0$. Running the proof above with $p$ in place of $f_{\theta_0}$, the CLT still applies to the score (it has mean zero at $\theta^*$ by definition of the maximizer), but the information identity fails: the variance of the score and the expected negative Hessian are now different matrices.</p>
<div class="thm" data-kind="Theorem" data-name="MLE under misspecification (Huber, 1967; White, 1982)">
<p>Under regularity conditions analogous to the above, with $A = -\E_p\nabla^2_\theta\log f(X;\theta^*)$ and $B = \E_p[\nabla_\theta\log f(X;\theta^*)\nabla_\theta\log f(X;\theta^*)^\top]$,
$$\sqrt n(\hth_n-\theta^*)\dto\N\big(0,\ A^{-1}BA^{-1}\big).$$
If the model is correct, $A=B=I(\theta_0)$ and the sandwich collapses to $I^{-1}$.</p>
</div>
<div class="ex" data-name="Exponential model, Gamma data">
<p>Fit Exponential(rate $\lambda$) to data with mean $m$ and variance $v$. Then $\log f = \log\lambda-\lambda x$, $\lambda^* = 1/m$, $A = 1/\lambda^{*2} = m^2$, $B = \Var_p(X) = v$, so the sandwich variance of $\sqrt n(\hat\lambda-\lambda^*)$ is $v/m^4$, whereas the model-based $I^{-1} = \lambda^{*2} = 1/m^2$ implicitly assumes $v = m^2$. For Gamma data with shape $k$, $v/m^2 = 1/k$: the model-based standard error is too large by a factor $\sqrt k$. For overdispersed data ($v>m^2$) it is too small — the common failure in <a href="#/t/overdispersion">count regression</a>.</p>
</div>
<p>The empirical sandwich $\hat A^{-1}\hat B\hat A^{-1}$, with $\hat A = J_n(\hth)/n$ and $\hat B = \frac1n\sum_i\nabla\log f(X_i;\hth)\nabla\log f(X_i;\hth)^\top$, is the "robust" or "Huber–White" standard error reported by regression software. It protects the variance, not the target: $\theta^*$ may or may not be the quantity you care about.</p>`,
      },
      {
        title: 'When maximum likelihood fails',
        html: R`
<p>Each standard failure violates an identifiable hypothesis of the theorems above. Recognizing which one is the practical skill.</p>
<div class="warn" data-name="Incidental parameters: Neyman–Scott (1948)">
<p>Observe pairs $X_{i1},X_{i2}\iid\N(\mu_i,\sigma^2)$, $i=1,\dots,K$, each pair with its own mean. The MLE is $\hat\mu_i = \bar X_i$ and
$$\hat\sigma^2 = \frac{1}{2K}\sum_{i=1}^K\sum_{j=1}^2(X_{ij}-\bar X_i)^2 = \frac1{4K}\sum_{i=1}^K(X_{i1}-X_{i2})^2.$$
Since $X_{i1}-X_{i2}\sim\N(0,2\sigma^2)$, the LLN gives $\hat\sigma^2\pto2\sigma^2/4 = \sigma^2/2$. The MLE is <em>inconsistent</em>, off by a factor of two no matter how many pairs we collect. The number of parameters grows with $n$, so the asymptotics above (fixed $k$, $n\to\infty$) never apply; each $\hat\mu_i$ is estimated from 2 points and its error is absorbed into the residuals. The cure is a likelihood free of the incidental parameters — the marginal likelihood of the differences gives $2\hat\sigma^2$, which is consistent (REML in <a href="#/t/mixed-models">mixed models</a>; conditional likelihood in matched designs).</p>
</div>
<div class="warn" data-name="Unbounded likelihood: normal mixtures">
<p>For $p\,\N(\mu_1,\sigma_1^2)+(1-p)\N(\mu_2,\sigma_2^2)$ with $0 \lt p \lt 1$, set $\mu_1 = x_1$ and let $\sigma_1\to0$. The first observation contributes at least $p\,\phi(0)/\sigma_1\to\infty$, while every other observation contributes at least $(1-p)f_2(x_i)>0$. So $\sup L = \infty$ and the global MLE does not exist; the degenerate "fit" puts a spike on one data point. Remedies: bound the variances below or their ratio (Hathaway, 1985), penalize, or use a <a href="#/t/bayesian-inference">prior</a>. A consistent local maximizer still exists (Kiefer, 1978), which is what <a href="#/t/em-algorithm">EM</a> from good starting values typically finds. Recall the rounding remark in <a href="#/t/likelihood">likelihood</a>: the exact interval likelihood is bounded.</p>
</div>
<div class="warn" data-name="Parameter on the boundary">
<p>For $X_i\iid\N(\mu,1)$ with $\Theta = [0,\infty)$ and true $\mu_0 = 0$, the MLE is $\hat\mu = \max(\bar X,0)$, so $\sqrt n\hat\mu\dto\max(Z,0)$: a 50% atom at 0 plus a half-normal. The interior-point assumption fails, Wald intervals misbehave, and the likelihood-ratio statistic converges to $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ rather than $\chi^2_1$ (Chernoff, 1954; Self & Liang, 1987). This is the situation in every test of "is this variance component zero?"</p>
</div>
<div class="warn" data-name="Support depending on θ">
<p>For Unif$(0,\theta)$, $\hth = X_{(n)}$ converges at rate $n$, not $\sqrt n$, to a non-normal limit: $n(\theta-\hth)\dto\theta\cdot\text{Exp}(1)$. It is also biased; $(n+1)X_{(n)}/n$ removes the bias. Fisher information and the Cramér–Rao machinery do not apply.</p>
</div>
<div class="warn" data-name="Nonexistence and non-uniqueness">
<p>All-zero Bernoulli data or separable logistic data push the natural parameter to $\pm\infty$ (see <a href="#/t/exponential-families">exponential families</a>); the Laplace location MLE is an interval for even $n$; the Cauchy location likelihood may have many local maxima for small $n$, though the probability of a unique consistent root tends to 1.</p>
</div>`,
        lab: 'mle-failures',
      },
    ],
    quiz: [
      { q: R`State the invariance property of the MLE and use it to find the MLE of the coefficient of variation $\sigma/\mu$ from normal data.`,
        a: R`If $\hth$ is the MLE of $\theta$ then $g(\hth)$ maximizes the induced likelihood of $g(\theta)$. With $\hat\mu = \bar X$ and $\hat\sigma = \sqrt{SS/n}$, the MLE of $\sigma/\mu$ is $\sqrt{SS/n}/\bar X$.` },
      { q: R`In the proof of asymptotic normality, which conditions are used for the numerator and which for the denominator?`,
        a: R`Numerator: the score has mean 0 and variance $I(\theta_0)$ — needs differentiation under the integral (and $\theta$-free support) plus finite positive $I$ — then the CLT. Denominator: the LLN for $-\ell''/n$ together with the information identity $-\E\ell'' = I$ (again differentiation under the integral), and the third-derivative bound plus consistency to kill the remainder.` },
      { q: R`In the Neyman–Scott problem, what does $\hat\sigma^2$ converge to, and why does the usual consistency theorem not apply?`,
        a: R`$\sigma^2/2$. The theorem assumes a fixed finite-dimensional parameter while $n\to\infty$; here the dimension $K+1$ grows proportionally to $n=2K$, and each incidental mean is estimated from only two observations, so its error never averages out of $\hat\sigma^2$.` },
      { q: R`You fit a Poisson regression to counts whose variance is three times their mean. Is $\hat\beta$ consistent? Are the model-based standard errors right?`,
        a: R`If the mean model is correct, the Poisson score still has mean zero at the true $\beta$, so $\hat\beta$ is consistent. But $B = 3A$, so the sandwich variance is $A^{-1}BA^{-1} = 3A^{-1}$: model-based standard errors are too small by $\sqrt3$. Use sandwich SEs or a quasi-Poisson/negative-binomial model.` },
      { q: R`Why does the likelihood of a two-component normal mixture with unequal unknown variances have no global maximum?`,
        a: R`Centering one component on a data point and shrinking its variance to 0 sends that point's density contribution to infinity, while the other component keeps every other point's contribution bounded away from 0. The supremum is $+\infty$ and is approached only by degenerate fits.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'fisher-information',
    domain: 'inference',
    order: 80,
    title: 'Fisher Information & the Cramér–Rao Bound',
    subtitle: 'How sharply the data can locate a parameter, and the variance floor that follows',
    level: 2,
    sources: { aos: '§9.7', cb: '§7.3.2', other: 'Lehmann & Casella (1998), §2.5–2.6; van der Vaart (1998), Ch. 8' },
    prereqs: ['likelihood', 'point-estimation'],
    related: ['mle', 'rao-blackwell', 'exponential-families', 'entropy-kl', 'decision-theory', 'bayesian-inference'],
    tags: ['score', 'information', 'Cramér–Rao', 'efficiency', 'superefficiency', 'Jeffreys'],
    summary: R`The score — the derivative of the log-likelihood — has mean zero at the true parameter, and its variance is the Fisher information $I(\theta)$, which also equals the expected curvature of the log-likelihood. Information measures how quickly the distribution changes as $\theta$ moves (it is the local curvature of KL divergence), it adds over independent observations, and it cannot be increased by processing the data (with equality exactly for sufficient statistics). Through a single application of Cauchy–Schwarz it bounds the variance of every unbiased estimator: $\Var(W)\ge\psi'(\theta)^2/(nI(\theta))$. The bound is attained only in exponential families, for the mean-value parameter; elsewhere the best unbiased estimator sits strictly above it, and biased or superefficient estimators can dip below it — but the MLE attains it asymptotically, which is the precise sense in which maximum likelihood is efficient.`,
    survey: {
      task: 'Quantify the information about θ in data; lower-bound estimator variance',
      assumptions: R`$\theta$-free support; differentiation under the integral sign; finite second moment of the score`,
      output: R`$I(\theta) = \Var_\theta(\text{score})$; bound $\Var_\theta W\ge\psi'(\theta)^2/I_n(\theta)$`,
      strengths: 'Universal benchmark for efficiency; gives asymptotic SEs; geometric meaning via KL',
      weaknesses: 'Fails for support-dependent models; not attained in general; biased estimators can beat it',
      complexity: 'Closed form for standard families; otherwise numerical or Monte Carlo',
    },
    sections: [
      {
        title: 'The score and its two moments',
        html: R`
<div class="def" data-name="Score and Fisher information">
<p>For a single observation with density $f(x;\theta)$, $\theta\in\R$, the <em>score</em> is $s(\theta;x) = \partial_\theta\log f(x;\theta)$, and the <em>Fisher information</em> is $I(\theta) = \E_\theta[s(\theta;X)^2]$. For data $X = (X_1,\dots,X_n)$ the score is $\ell_n'(\theta)=\sum_is(\theta;X_i)$ and the information is $I_n(\theta) = \E_\theta[\ell_n'(\theta)^2]$. The <em>observed information</em> is $J_n(\theta) = -\ell_n''(\theta)$.</p>
</div>
<div class="thm" data-kind="Lemma" data-name="Score identities">
<p>Suppose the support of $f(\cdot;\theta)$ does not depend on $\theta$ and $\int f(x;\theta)\d x$ can be differentiated twice under the integral. Then
$$\E_\theta\,s(\theta;X) = 0,\qquad I(\theta) = \Var_\theta s(\theta;X) = -\E_\theta\,\partial_\theta^2\log f(X;\theta).$$
For IID data, $I_n(\theta) = nI(\theta)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Differentiate $\int f(x;\theta)\d x = 1$: $0 = \int\partial_\theta f = \int(\partial_\theta\log f)f = \E_\theta s$. Differentiate again: $0 = \int\partial_\theta\big[(\partial_\theta\log f)f\big] = \int(\partial^2_\theta\log f)f + \int(\partial_\theta\log f)^2f$, which is $\E\,\partial_\theta^2\log f + I(\theta) = 0$. For independent observations the scores are independent with mean zero, so variances add.</p>
</details>
<p>Both identities have content. The first says the true parameter is a root of the <em>expected</em> score, which is why the MLE, a root of the sample score, is consistent. The second says the expected curvature of the log-likelihood equals the variance of its slope; under misspecification the two separate, and that gap is the <a href="#/t/mle">sandwich</a>.</p>
<table>
<thead><tr><th>Model (one observation)</th><th>$I(\theta)$</th></tr></thead>
<tbody>
<tr><td>Bernoulli$(p)$</td><td>$1/[p(1-p)]$</td></tr>
<tr><td>Poisson$(\lambda)$</td><td>$1/\lambda$</td></tr>
<tr><td>Exponential (rate $\lambda$)</td><td>$1/\lambda^2$</td></tr>
<tr><td>$\N(\mu,\sigma^2)$, $\mu$ ($\sigma$ known)</td><td>$1/\sigma^2$</td></tr>
<tr><td>$\N(\mu,\sigma^2)$, $\sigma^2$ ($\mu$ known)</td><td>$1/(2\sigma^4)$</td></tr>
<tr><td>Location: Laplace / logistic / Cauchy / $t_\nu$ (unit scale)</td><td>$1$ / $1/3$ / $1/2$ / $(\nu+1)/(\nu+3)$</td></tr>
<tr><td>Uniform$(0,\theta)$</td><td>not defined: support depends on $\theta$</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Proposition" data-name="Information cannot be created by processing">
<p>For any statistic $T = T(X)$, under regularity conditions $I_T(\theta)\le I_X(\theta)$, with equality for all $\theta$ if and only if $T$ is <a href="#/t/sufficiency">sufficient</a>.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Factor $f_X(x;\theta) = f_T(t;\theta)f_{X\mid T}(x\mid t;\theta)$, so the score splits as $s_X = s_T + s_{X\mid T}$. The conditional score has conditional mean zero given $T$ (apply the first identity to the conditional density), so it is uncorrelated with any function of $T$, in particular $s_T$. Hence $I_X = I_T + \E\Var(s_{X|T}\mid T)\ge I_T$, with equality iff $s_{X\mid T}=0$ a.s. for all $\theta$, i.e. iff the conditional law of $X$ given $T$ is free of $\theta$.</p>
</details>`,
        lab: 'fisher-information-score',
      },
      {
        title: 'Information as curvature and as distinguishability',
        html: R`
<p>Information measures how fast the distribution changes with $\theta$. Expanding the KL divergence to second order,
$$\KL\big(f_\theta\,\|\,f_{\theta+\delta}\big) = \tfrac12I(\theta)\,\delta^2 + o(\delta^2),$$
because the zeroth-order term vanishes, the first-order term is $-\delta\,\E_\theta s = 0$, and the second-order term is $-\tfrac12\delta^2\E_\theta\partial_\theta^2\log f = \tfrac12\delta^2I(\theta)$. With $n$ observations the divergence is $\tfrac12nI\delta^2$; two parameter values become statistically distinguishable once this is of order 1, i.e. when $|\delta|\gtrsim1/\sqrt{nI(\theta)}$. That is the $\sqrt n$ rate and its constant, in one line. In multiparameter models, $I(\theta)$ is a Riemannian metric on the parameter space ("information geometry").</p>
<div class="thm" data-kind="Proposition" data-name="Reparameterization">
<p>If $\theta = h(\eta)$ with $h$ differentiable, the information about $\eta$ is $I^{*}(\eta) = I(h(\eta))\,h'(\eta)^2$. In $k$ dimensions, $I^*(\eta) = D^\top I(\theta)D$ with $D = \partial\theta/\partial\eta$.</p>
</div>
<p>This follows from the chain rule for the score, $s^*(\eta) = s(\theta)h'(\eta)$. Two classic uses:</p>
<ul>
<li><strong>Variance-stabilizing transformations.</strong> Choose $\eta = \int^\theta\sqrt{I(u)}\d u$, so that $I^*(\eta)\equiv1$ and $\hat\eta$ has asymptotic variance $1/n$ regardless of $\theta$. Poisson: $\eta = 2\sqrt\lambda$. Bernoulli: $\eta = 2\arcsin\sqrt p$.</li>
<li><strong>Jeffreys prior</strong> $\pi(\theta)\propto\sqrt{\det I(\theta)}$, which transforms consistently under reparameterization (see <a href="#/t/bayesian-inference">Bayesian inference</a>).</li>
</ul>`,
      },
      {
        title: 'Several parameters and the cost of nuisance',
        html: R`
<p>For $\theta\in\R^k$, the information matrix is $I(\theta) = \E[s s^\top] = -\E\nabla^2\log f$. Partition $\theta = (\psi,\lambda)$ with $\psi$ scalar of interest. If $\lambda$ were known, the information about $\psi$ would be $I_{\psi\psi}$. With $\lambda$ unknown, the asymptotic variance of the MLE $\hat\psi$ is the $(\psi,\psi)$ entry of $I^{-1}$, and by the block-inverse formula
$$\big[(I^{-1})_{\psi\psi}\big]^{-1} = I_{\psi\psi} - I_{\psi\lambda}I_{\lambda\lambda}^{-1}I_{\lambda\psi}\ \le\ I_{\psi\psi}.$$
The left side is the <strong>efficient information</strong> for $\psi$: what remains after projecting the $\psi$-score onto the orthogonal complement of the nuisance scores. Not knowing $\lambda$ costs nothing asymptotically iff $I_{\psi\lambda}=0$ — the parameters are <em>orthogonal</em>.</p>
<div class="ex" data-name="Orthogonal and non-orthogonal pairs">
<p>Normal $(\mu,\sigma^2)$: $I = \diag(1/\sigma^2,\ 1/(2\sigma^4))$, orthogonal, so estimating $\sigma$ does not slow down $\hat\mu$ asymptotically. Gamma (shape $\alpha$, scale $\beta$): $I = \begin{pmatrix}\psi'(\alpha) & 1/\beta\\ 1/\beta & \alpha/\beta^2\end{pmatrix}$, so with $\beta$ known the variance bound for $\alpha$ is $1/(n\psi'(\alpha))$, while with $\beta$ unknown it is $\alpha/(n(\alpha\psi'(\alpha)-1))$, which is larger (often much larger for large $\alpha$). Reparameterizing by the mean $m = \alpha\beta$ instead of $\beta$ makes the pair $(\alpha,m)$ orthogonal (Cox & Reid, 1987).</p>
</div>`,
      },
      {
        title: 'The Cramér–Rao lower bound',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Cramér–Rao inequality">
<p>Let $W = W(X)$ have $\E_\theta W = \psi(\theta)$ and $\Var_\theta W<\infty$, and suppose $\frac{d}{d\theta}\int W(x)f(x;\theta)\d x = \int W(x)\,\partial_\theta f(x;\theta)\d x$ together with the score identities. Then
$$\Var_\theta W\ \ge\ \frac{\psi'(\theta)^2}{I_n(\theta)}.$$
In particular, for unbiased estimators of $\theta$ from IID data, $\Var_\theta W\ge1/(nI(\theta))$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $S = \ell_n'(\theta)$, with $\E_\theta S=0$ and $\Var_\theta S = I_n(\theta)$. Then
$$\Cov_\theta(W,S) = \E_\theta[WS] = \int W\,\frac{\partial_\theta f}{f}\,f = \int W\partial_\theta f = \psi'(\theta).$$
By Cauchy–Schwarz, $\psi'(\theta)^2 = \Cov(W,S)^2\le\Var(W)\Var(S) = \Var(W)I_n(\theta)$.</p>
</details>
<div class="thm" data-kind="Corollary" data-name="Attainment">
<p>Equality holds at every $\theta$ if and only if $S = a(\theta)\big(W-\psi(\theta)\big)$ almost surely for some function $a$ (necessarily $a = I_n/\psi'$). Then $f(x;\theta)$ is a one-parameter <a href="#/t/exponential-families">exponential family</a> with sufficient statistic $W$, and $\psi(\theta) = \E_\theta W$ is (an affine function of) its mean-value parameter. Thus an unbiased estimator attaining the bound exists only for that one parameter, up to affine maps.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Equality in Cauchy–Schwarz holds iff $S-\E S$ and $W-\E W$ are linearly dependent, i.e. $S = a(\theta)(W-\psi(\theta))$ since $\E S=0$. Integrating $\partial_\theta\log f(x;\theta) = a(\theta)W(x) - a(\theta)\psi(\theta)$ in $\theta$ gives $\log f(x;\theta) = \eta(\theta)W(x) - B(\theta) + \log h(x)$ with $\eta' = a$, $B' = a\psi$ — an exponential family in which $W$ is the sufficient statistic, and $\E_\theta W = \psi(\theta)$ by assumption. Conversely, in such a family the score is $\eta'(\theta)(W - \E_\theta W)$, so equality holds.</p>
</details>
<div class="ex" data-name="Attained, not attained, not applicable">
<ul>
<li><strong>Attained.</strong> Poisson: $\Var\bar X = \lambda/n = 1/(nI(\lambda))$. Normal variance with known mean: $\frac1n\sum(X_i-\mu)^2$ has variance $2\sigma^4/n$, the bound.</li>
<li><strong>Not attained.</strong> Normal variance with unknown mean: the bound is still $2\sigma^4/n$, but the best unbiased estimator $S^2$ has variance $2\sigma^4/(n-1)$. Poisson $e^{-\lambda}$: bound $\lambda e^{-2\lambda}/n$, best unbiased variance $e^{-2\lambda}(e^{\lambda/n}-1)$, strictly larger. In both cases the <a href="#/t/rao-blackwell">UMVUE</a> exists but sits above the bound, because the target is not the mean-value parameter.</li>
<li><strong>Not applicable.</strong> Uniform$(0,\theta)$: the unbiased $\frac{n+1}{n}X_{(n)}$ has variance $\theta^2/(n(n+2))$, of order $n^{-2}$. Differentiating under the integral fails because the support moves with $\theta$.</li>
</ul>
</div>
<p>The multiparameter version: if $\E_\theta W = \psi(\theta)\in\R^m$ with Jacobian $D_\psi$, then $\Cov_\theta W\succeq D_\psi\,I_n(\theta)^{-1}D_\psi^\top$ in the positive semidefinite order. The lab compares the variances of several location estimators against $1/(nI)$ across five error distributions.</p>`,
        lab: 'fisher-information-crlb',
      },
      {
        title: 'Beating the bound, and why the MLE is still efficient',
        html: R`
<p>The Cramér–Rao bound constrains <em>unbiased</em> estimators. For an estimator with bias $b(\theta)$, $\psi(\theta) = \theta+b(\theta)$ gives $\MSE_\theta\ge b(\theta)^2 + (1+b'(\theta))^2/I_n(\theta)$, and a biased estimator with $b'<0$ (shrinkage) can have MSE below $1/I_n$ over a region of $\Theta$ — sometimes everywhere: for normal data $SS/(n+1)$ has MSE $2\sigma^4/(n+1)$, below the bound $2\sigma^4/n$ for unbiased estimators of $\sigma^2$ at every parameter value. For the normal mean in one dimension, by contrast, $\bar X$ is admissible and minimax, so no estimator beats $1/(nI)$ everywhere (see <a href="#/t/decision-theory">decision theory</a>).</p>
<div class="ex" data-name="Hodges' superefficient estimator (1951)">
<p>For $X_i\iid\N(\theta,1)$ let $T_n = \bar X_n$ if $|\bar X_n|\ge n^{-1/4}$ and $T_n=0$ otherwise. At any fixed $\theta\ne0$, $T_n=\bar X_n$ with probability tending to 1, so $\sqrt n(T_n-\theta)\dto\N(0,1)$. At $\theta = 0$, $T_n = 0$ with probability tending to 1, so $\sqrt n\,T_n\dto0$: asymptotic variance 0, below $1/I = 1$. But the gain is fake: for $\theta_n$ near $n^{-1/4}$ the estimator often reports 0 when the truth is $n^{-1/4}$ away, so $\sup_\theta n\,\MSE_\theta(T_n)\to\infty$. The third lab plots the exact risk $n\MSE_\theta(T_n)$.</p>
</div>
<p>Le Cam (1953) showed that superefficiency can occur only on a set of Lebesgue measure zero. Hájek's <strong>convolution theorem</strong> (1970) says that any <em>regular</em> estimator (one whose limit law is stable under $n^{-1/2}$ perturbations of $\theta$) has limit $\N(0,I^{-1})*L$ for some law $L$, and the <strong>local asymptotic minimax theorem</strong> (Hájek, 1972) says $\liminf_n\sup_{|\theta-\theta_0|\le c/\sqrt n}n\MSE_\theta\ge1/I(\theta_0)$ as $c\to\infty$. Together they make precise the sense in which the <a href="#/t/mle">MLE</a>, with limit exactly $\N(0,I^{-1})$, is asymptotically optimal.</p>`,
        lab: 'fisher-information-hodges',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Information is a model property">
<p>$I(\theta)$ is computed under the assumed model. Under misspecification the inverse information is the wrong variance; use the sandwich. Under dependence (time series, clustered data) the information is not $nI(\theta)$; it must be computed from the joint likelihood.</p>
</div>
<div class="warn" data-name="Expected vs observed">
<p>$nI(\hth)$ and $J_n(\hth)$ agree to first order, but they differ in finite samples, and the observed version reflects ancillary information about precision actually achieved in this sample (Efron & Hinkley, 1978). In exponential families in the natural parameter they coincide exactly.</p>
</div>
<div class="warn" data-name="Regularity is not decoration">
<p>The bound can fail when the support depends on $\theta$ (uniform), when the density is not smooth enough (the Laplace location model still has $I=1$ and the median attains it asymptotically, but the classical proof of MLE normality needs modification), and at the boundary of the parameter space, where $I(\theta)$ may be infinite or zero (e.g. $p\to0$ in the Bernoulli model, where the information $1/(p(1-p))$ explodes).</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Compute the Fisher information for $\sigma$ (not $\sigma^2$) in $\N(0,\sigma^2)$, using the reparameterization rule.`,
        a: R`$I(\sigma^2) = 1/(2\sigma^4)$. With $\sigma^2 = h(\sigma) = \sigma^2$, $h'(\sigma) = 2\sigma$, so $I^*(\sigma) = \frac{1}{2\sigma^4}\cdot4\sigma^2 = 2/\sigma^2$. Direct check: $\partial_\sigma\log f = -1/\sigma + x^2/\sigma^3$, whose variance is $\Var(X^2)/\sigma^6 = 2\sigma^4/\sigma^6 = 2/\sigma^2$.` },
      { q: R`Why does the Cramér–Rao bound for unbiased estimation of $\sigma^2$ (with $\mu$ unknown) equal $2\sigma^4/n$, while $S^2$ — the UMVUE — has variance $2\sigma^4/(n-1)$? Is there a contradiction?`,
        a: R`No. With $\mu$ unknown the bound is the $(\sigma^2,\sigma^2)$ element of $I_n^{-1}$, which is $2\sigma^4/n$ because $\mu$ and $\sigma^2$ are orthogonal. Attainment would require the score for $\sigma^2$ to be affine in an unbiased estimator, which fails when $\mu$ is unknown. The bound is simply not sharp here; the UMVUE exceeds it.` },
      { q: R`For which parameters of a one-parameter family can an unbiased estimator attain the Cramér–Rao bound?`,
        a: R`Only if the family is an exponential family $h(x)\exp\{\eta(\theta)T(x)-B(\theta)\}$, and only for $\psi(\theta) = a+b\,\E_\theta T$ (affine functions of the mean-value parameter), estimated by $a+b\bar T$.` },
      { q: R`Explain why Hodges' estimator does not contradict the asymptotic optimality of the MLE.`,
        a: R`It beats $1/I$ only at the single point $\theta=0$ (a null set, consistent with Le Cam's theorem), and it is not regular: under local alternatives $\theta_n = h/\sqrt n$ its limit law shifts, and its worst-case scaled risk over shrinking neighbourhoods of 0 is unbounded. Optimality theorems (convolution, local asymptotic minimax) are stated for regular estimators or in a local-minimax sense, where the MLE is optimal.` },
    ],
  });

  /* ================================================================== */
  Atlas.topic({
    id: 'rao-blackwell',
    domain: 'inference',
    order: 90,
    title: 'Rao–Blackwell, Lehmann–Scheffé & the UMVUE',
    subtitle: 'Conditioning on a sufficient statistic never hurts; with completeness it gives the best unbiased estimator',
    level: 2,
    sources: { cb: '§7.3.3', other: 'Rao (1945); Blackwell (1947); Lehmann & Scheffé (1950); Lehmann & Casella (1998), Ch. 2' },
    prereqs: ['sufficiency', 'point-estimation'],
    related: ['fisher-information', 'exponential-families', 'decision-theory', 'conditional-expectation', 'variance-reduction', 'gibbs-sampling'],
    tags: ['UMVUE', 'conditioning', 'completeness', 'unbiasedness', 'Rao–Blackwellization'],
    summary: R`Take any estimator $W$ and replace it by its conditional expectation given a sufficient statistic $T$. Because $T$ is sufficient, $\E[W\mid T]$ does not depend on $\theta$, so it is still an estimator; it has the same mean, and by the law of total variance it has variance no larger — strictly smaller unless $W$ was already a function of $T$. That is the Rao–Blackwell theorem: averaging out irrelevant noise can only help, for any convex loss. If $T$ is also complete, there is only one unbiased function of $T$, so every unbiased estimator Rao–Blackwellizes to the same thing: the uniformly minimum-variance unbiased estimator (Lehmann–Scheffé). The theory is elegant and constructive, but it inherits the limits of unbiasedness — UMVUEs can be inadmissible, absurd, or nonexistent — and its most widespread modern use is in Monte Carlo, where conditioning reduces simulation variance.`,
    survey: {
      task: 'Construct the uniformly minimum-variance unbiased estimator (UMVUE) of τ(θ)',
      assumptions: R`A sufficient statistic $T$ (complete, for uniqueness/optimality); an unbiased starting estimator or solvable $\E_\theta g(T)=\tau(\theta)$`,
      output: R`$\phi(T) = \E[W\mid T]$; unique UMVUE when $T$ is complete`,
      strengths: 'Constructive; exact finite-sample optimality among unbiased estimators; variance reduction in simulation',
      weaknesses: 'Unbiasedness may be impossible or harmful; UMVUE can be inadmissible under MSE',
      complexity: R`Analytic conditional expectation; in Monte Carlo, one conditional mean per draw`,
    },
    sections: [
      {
        title: 'Conditioning reduces variance',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Rao–Blackwell (Rao, 1945; Blackwell, 1947)">
<p>Let $T$ be sufficient for $\theta$ and $W$ an estimator with $\E_\theta|W|<\infty$. Define $\phi(T) = \E[W\mid T]$. Then $\phi(T)$ is an estimator (it does not depend on $\theta$), $\E_\theta\phi(T) = \E_\theta W$, and
$$\Var_\theta\phi(T)\le\Var_\theta W\qquad\text{for all }\theta,$$
with strict inequality unless $W = \phi(T)$ almost surely. More generally, $\E_\theta L(\theta,\phi(T))\le\E_\theta L(\theta,W)$ for any loss $L(\theta,\cdot)$ that is convex in its second argument.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Because $T$ is sufficient, the conditional law of $X$ (hence of $W$) given $T$ is free of $\theta$, so $\phi(T)$ is computable from data. The tower property gives $\E_\theta\phi(T) = \E_\theta W$. The law of total variance gives
$$\Var_\theta W = \Var_\theta\big(\E[W\mid T]\big) + \E_\theta\big[\Var(W\mid T)\big]\ \ge\ \Var_\theta\phi(T),$$
with equality iff $\Var(W\mid T)=0$ a.s., i.e. $W = \phi(T)$ a.s. For convex loss, the conditional Jensen inequality gives $L(\theta,\E[W\mid T])\le\E[L(\theta,W)\mid T]$; take expectations.</p>
</details>
<div class="note">
<p>Think of $W$ as "$\phi(T)$ plus noise": the part of $W$ not explained by $T$ is, by sufficiency, a function of the $\theta$-free residual randomness in the data. Adding pure noise to an estimator can only inflate its variance. Rao–Blackwellization strips that noise off. Sufficiency matters: conditioning on a non-sufficient statistic would give $\E_\theta[W\mid U]$, a function of $\theta$ — not an estimator.</p>
</div>`,
      },
      {
        title: 'Worked examples',
        html: R`
<div class="ex" data-name="Poisson: probability of a zero count">
<p>$X_i\iid\text{Poisson}(\lambda)$; target $\tau = \P(X=0) = e^{-\lambda}$. Crude unbiased estimator: $W = \ind\{X_1=0\}$. Sufficient: $T=\sum X_i$. Given $T=t$, $X_1\sim\text{Binomial}(t,1/n)$, so
$$\phi(T) = \P(X_1=0\mid T) = \Big(1-\frac1n\Big)^T.$$
Variances: $\Var W = e^{-\lambda}(1-e^{-\lambda})$, which does not shrink with $n$ at all, while, using the Poisson MGF $\E s^T = e^{n\lambda(s-1)}$,
$$\Var\phi(T) = e^{-n\lambda(1-(1-1/n)^2)} - e^{-2\lambda} = e^{-2\lambda}\big(e^{\lambda/n}-1\big)\approx\frac{\lambda e^{-2\lambda}}{n}.$$
The Cramér–Rao bound is $\lambda e^{-2\lambda}/n$; since $e^{x}-1>x$, $\phi(T)$ sits strictly above it for every $n$, yet (by Lehmann–Scheffé below) nothing unbiased does better. The MLE $e^{-\bar X}$ is slightly biased and has nearly the same variance.</p>
</div>
<div class="ex" data-name="Uniform: from one observation to the maximum">
<p>$X_i\iid\text{Unif}(0,\theta)$. $W = 2X_1$ is unbiased with variance $\theta^2/3$. The sufficient statistic is $M = X_{(n)}$. Given $M=m$, $X_1 = m$ with probability $1/n$ and is otherwise Unif$(0,m)$, so $\E[X_1\mid M] = \frac{M}{n}+\frac{n-1}{n}\frac M2 = \frac{(n+1)M}{2n}$ and
$$\phi(M) = \frac{n+1}{n}X_{(n)},\qquad\Var\phi(M) = \frac{\theta^2}{n(n+2)}.$$
Conditioning turned a variance that never shrinks into one of order $n^{-2}$. Note $2\bar X$ is also unbiased (variance $\theta^2/(3n)$) and Rao–Blackwellizes to the same $\phi(M)$.</p>
</div>
<div class="ex" data-name="Bernoulli: estimating p²">
<p>$W = X_1X_2$ is unbiased for $p^2$. Given $T = \sum X_i = t$, the chance that both of the first two trials are successes is $\frac{t(t-1)}{n(n-1)}$, so $\phi(T) = \frac{T(T-1)}{n(n-1)}$. Unlike the plug-in $\hat p^2 = T^2/n^2$, it is exactly unbiased.</p>
</div>
<p>The first lab simulates these sampling distributions side by side; the second shows conditioning geometrically, as the collapse of a scatter onto its regression curve.</p>`,
        lab: 'rao-blackwell-variance',
      },
      {
        title: 'Lehmann–Scheffé: completeness gives uniqueness',
        html: R`
<p>Rao–Blackwell improves an estimator, but different starting points could, in principle, lead to different improved estimators. <a href="#/t/sufficiency">Completeness</a> rules that out.</p>
<div class="thm" data-kind="Theorem" data-name="Lehmann–Scheffé (1950)">
<p>Let $T$ be complete and sufficient. If $\phi(T)$ is unbiased for $\tau(\theta)$ and has finite variance, then $\phi(T)$ is the unique (a.s.) uniformly minimum-variance unbiased estimator (UMVUE) of $\tau(\theta)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $W$ be any unbiased estimator of $\tau$ with finite variance. By Rao–Blackwell, $\phi_W(T) = \E[W\mid T]$ is unbiased with $\Var\phi_W(T)\le\Var W$. Both $\phi_W(T)$ and $\phi(T)$ are unbiased functions of $T$, so $\E_\theta[\phi_W(T)-\phi(T)] = 0$ for all $\theta$, and completeness forces $\phi_W(T) = \phi(T)$ a.s. Hence $\Var_\theta\phi(T)\le\Var_\theta W$ for every $\theta$ and every unbiased $W$. Uniqueness: if $W$ is another UMVUE, then $\Var W = \Var\phi(T)=\Var\phi_W(T)$, which by the equality case of Rao–Blackwell means $W = \phi_W(T) = \phi(T)$ a.s.</p>
</details>
<div class="algo" data-name="Two routes to a UMVUE">
<ol>
<li>Find a complete sufficient statistic $T$ (in a full-rank <a href="#/t/exponential-families">exponential family</a>, $T=\sum T(X_i)$).</li>
<li><em>Either</em> take any simple unbiased $W$ and compute $\E[W\mid T]$, <em>or</em> solve $\E_\theta g(T) = \tau(\theta)$ for $g$ directly, e.g. by matching power-series coefficients.</li>
</ol>
</div>
<table>
<thead><tr><th>Model</th><th>Target</th><th>UMVUE</th></tr></thead>
<tbody>
<tr><td>$\N(\mu,\sigma^2)$</td><td>$\mu$, $\sigma^2$</td><td>$\bar X$, $S^2$</td></tr>
<tr><td>$\N(\mu,\sigma^2)$</td><td>$\mu^2$</td><td>$\bar X^2 - S^2/n$</td></tr>
<tr><td>$\N(\mu,\sigma^2)$</td><td>$\sigma$</td><td>$S\,\sqrt{\tfrac{n-1}{2}}\,\Gamma(\tfrac{n-1}{2})/\Gamma(\tfrac n2)$</td></tr>
<tr><td>Bernoulli$(p)$</td><td>$p(1-p)$</td><td>$T(n-T)/[n(n-1)]$</td></tr>
<tr><td>Poisson$(\lambda)$</td><td>$\P(X=k)$</td><td>$\binom Tk n^{-k}(1-1/n)^{T-k}$</td></tr>
<tr><td>Exponential (rate $\lambda$)</td><td>$\lambda$</td><td>$(n-1)/\sum X_i$</td></tr>
<tr><td>Unif$(0,\theta)$</td><td>$\theta$</td><td>$\frac{n+1}{n}X_{(n)}$</td></tr>
</tbody>
</table>
<p>(For the exponential: $T = \sum X_i\sim\text{Gamma}(n,\text{rate }\lambda)$ has $\E[1/T] = \lambda/(n-1)$; so the MLE $n/T$ is biased upward by the factor $n/(n-1)$.)</p>`,
        lab: 'rao-blackwell-conditioning',
      },
      {
        title: 'A characterization without completeness',
        html: R`
<p>Completeness is sufficient for the existence of a UMVUE, not necessary. The general criterion is about correlation with noise.</p>
<div class="thm" data-kind="Theorem" data-name="UMVUE characterization">
<p>Let $W$ be unbiased for $\tau(\theta)$ with finite variance. Then $W$ is a UMVUE if and only if $\Cov_\theta(W,U)=0$ for every $\theta$ and every $U$ with $\E_\theta U = 0$ for all $\theta$ and finite variance (every "unbiased estimator of zero"). A UMVUE, if it exists, is unique.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>($\Leftarrow$) Any other unbiased $W'$ can be written $W' = W+U$ with $U = W'-W$ an unbiased estimator of zero; then $\Var W' = \Var W+\Var U+2\Cov(W,U) = \Var W+\Var U\ge\Var W$.
($\Rightarrow$) If $\Cov_{\theta_0}(W,U)=c\ne0$ for some $\theta_0$ and $U$, then $W_a = W+aU$ is unbiased with $\Var_{\theta_0}W_a = \Var_{\theta_0}W + 2ac + a^2\Var_{\theta_0}U$, which is less than $\Var_{\theta_0}W$ for $a = -c/\Var_{\theta_0}U$. So $W$ is not UMVUE.
Uniqueness: if $W,W'$ are both UMVUEs, $U = W'-W$ satisfies $\Cov(W,U)=0$, so $\Var W' = \Var W+\Var U$; equality of variances forces $\Var_\theta U = 0$, i.e. $W'=W$ a.s.</p>
</details>
<p>In a complete family the only unbiased estimator of zero based on $T$ is $0$, and any unbiased estimator of zero $U$ has $\E[U\mid T]$ an unbiased estimator of zero based on $T$, hence 0; so $\Cov(\phi(T), U) = \E[\phi(T)\E(U\mid T)]=0$, recovering Lehmann–Scheffé. In incomplete families the criterion explains why UMVUEs often fail to exist: there are too many unbiased estimators of zero to be uncorrelated with all of them at every $\theta$ simultaneously. Uniform$(\theta,\theta+1)$ is a standard example in which no UMVUE of $\theta$ exists.</p>`,
      },
      {
        title: 'The limits of unbiased optimality',
        html: R`
<p>"Best among unbiased estimators" is a strong statement inside a restricted class. Outside it, the UMVUE can look poor.</p>
<div class="warn" data-name="Dominated in mean squared error">
<p>For normal data, $S^2$ is the UMVUE of $\sigma^2$, yet $SS/(n+1)$ has smaller MSE at every $(\mu,\sigma^2)$ (see <a href="#/t/point-estimation">point estimation</a>). The UMVUE is <a href="#/t/decision-theory">inadmissible</a> under squared-error loss.</p>
</div>
<div class="warn" data-name="Absurd or out-of-range values">
<p>The UMVUE of $\mu^2$, $\bar X^2 - S^2/n$, is negative with positive probability, especially when $|\mu|$ is small relative to $\sigma/\sqrt n$; truncating at 0 introduces bias but strictly reduces MSE. The UMVUE of $e^{-2\lambda}$ from a single Poisson count is $(-1)^X$.</p>
</div>
<div class="warn" data-name="Nonexistence">
<p>No unbiased estimator of $1/p$ exists in the binomial model, nor of the odds $p/(1-p)$; then there is no UMVUE to find. Consistent, slightly biased estimators (MLE, Bayes) are the practical answer.</p>
</div>
<p>Rao–Blackwell itself is immune to these criticisms: conditioning on a sufficient statistic improves <em>any</em> estimator under <em>any</em> convex loss, biased or not. It is unbiasedness, not conditioning, that is the restrictive ingredient. This is why the Rao–Blackwell idea survives in decision theory (complete classes of estimators are functions of sufficient statistics) while the UMVUE is mostly a classical benchmark.</p>`,
      },
      {
        title: 'Rao–Blackwellization in Monte Carlo',
        html: R`
<p>The same law of total variance is a workhorse of simulation. To estimate $\mu = \E h(X)$ from draws $(X_t, Y_t)$, if $g(y) = \E[h(X)\mid Y=y]$ is available in closed form, then $\frac1N\sum_t g(Y_t)$ is unbiased with variance $\Var(g(Y))/N\le\Var(h(X))/N$ for independent draws.</p>
<ul>
<li><strong>Gibbs samplers.</strong> Average conditional expectations (or conditional densities) instead of raw draws (Gelfand & Smith, 1990). For Markov-chain output the inequality is not automatic, because autocorrelations change too, but for the two-block data-augmentation sampler it holds (Liu, Wong & Kong, 1994). See <a href="#/t/gibbs-sampling">Gibbs sampling</a>.</li>
<li><strong>Rao–Blackwellized particle filters</strong> integrate out linear-Gaussian components analytically with Kalman filters and simulate only the rest (<a href="#/t/particle-filters">sequential Monte Carlo</a>).</li>
<li><strong>Conditional Monte Carlo</strong> more broadly is one of the standard <a href="#/t/variance-reduction">variance-reduction</a> techniques: integrate out whatever you can do exactly.</li>
</ul>
<div class="ex" data-name="Estimating a tail probability by conditioning">
<p>To estimate $\P(X+Y>c)$ with $X\sim\N(0,1)$ independent of $Y$, simulate only $Y$ and average $1-\Phi(c-Y_t)$ instead of the indicator $\ind\{X_t+Y_t>c\}$. The estimator is smoother, never has larger variance, and is far better when the event is rare.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why must the conditioning statistic in Rao–Blackwell be sufficient? What goes wrong if you condition on $X_{(1)}$ in the Poisson model?`,
        a: R`The improved estimator is $\E_\theta[W\mid T]$; it is a usable estimator only if this conditional expectation does not depend on $\theta$, which is what sufficiency guarantees. $X_{(1)}$ is not sufficient for the Poisson, so $\E_\lambda[W\mid X_{(1)}]$ generally depends on $\lambda$ and cannot be computed from the data.` },
      { q: R`Find the UMVUE of $p^2$ from $n$ Bernoulli trials and verify it is unbiased.`,
        a: R`$\phi(T) = T(T-1)/(n(n-1))$. Since $\E[T(T-1)] = \E T^2-\E T = np(1-p)+n^2p^2-np = n(n-1)p^2$, it is unbiased; it is a function of the complete sufficient $T$, so by Lehmann–Scheffé it is the UMVUE.` },
      { q: R`For $X_i\iid\text{Unif}(0,\theta)$, three unbiased estimators of $\theta$ are $2X_1$, $2\bar X$, and $\frac{n+1}{n}X_{(n)}$. What are their variances, and what do the first two Rao–Blackwellize to?`,
        a: R`$\theta^2/3$, $\theta^2/(3n)$, and $\theta^2/(n(n+2))$. Both $2X_1$ and $2\bar X$ condition (on the complete sufficient $X_{(n)}$) to $\frac{n+1}{n}X_{(n)}$, as Lehmann–Scheffé requires.` },
      { q: R`The UMVUE of $\sigma^2$ in the normal model is $S^2$. Name an estimator with uniformly smaller MSE, and say what this shows.`,
        a: R`$SS/(n+1)$ has MSE $2\sigma^4/(n+1) \lt 2\sigma^4/(n-1)$ for all parameters. Minimum variance among unbiased estimators does not imply admissibility under squared error: the unbiasedness constraint excludes better (biased) estimators.` },
      { q: R`Show that $W$ UMVUE and $U$ an unbiased estimator of zero imply $\Cov_\theta(W,U)=0$.`,
        a: R`Otherwise, for $a = -\Cov_\theta(W,U)/\Var_\theta U$ the unbiased estimator $W+aU$ would have variance $\Var_\theta W - \Cov_\theta(W,U)^2/\Var_\theta U \lt \Var_\theta W$ at that $\theta$, contradicting uniform minimality.` },
    ],
  });
})();
