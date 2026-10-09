/* Supervised learning I: the framework, bias–variance, Bayes-optimal classification, generative
   classifiers, nearest neighbours, VC theory and evaluation. Labs live in js/labs/supervised-1.js. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'learning-framework',
    domain: 'supervised',
    order: 10,
    title: 'The Statistical Learning Framework: Loss, Risk & ERM',
    subtitle: 'What it means to learn a predictor from data, and the two errors every learner makes',
    level: 2,
    sources: { aos: '22.1', other: 'ESL Ch. 2 and §7.1–7.4; Shalev-Shwartz & Ben-David Ch. 2–4; Bishop §1.5' },
    prereqs: ['conditional-expectation'],
    related: ['decision-theory', 'bias-variance', 'vc-theory', 'model-selection', 'cross-validation'],
    tags: ['risk', 'loss', 'ERM', 'generalization', 'approximation error', 'estimation error'],
    summary: R`Supervised learning is decision theory with an unknown distribution. We want a predictor $f$ with small <em>risk</em> $R(f) = \E\,L(Y, f(X))$, but the expectation is over a law $P$ we only see through $n$ samples, so we minimize the <em>empirical</em> risk instead, over a restricted class $\mathcal H$. Everything in the subject follows from that substitution: the minimizer of the true risk is a conditional functional of $Y$ given $X$ (a mean, a median, a mode), the training error is optimistically biased, and the excess risk of any learner splits into an approximation error (the price of $\mathcal H$ being too small) and an estimation error (the price of $\mathcal H$ being too large for $n$).`,
    survey: {
      task: 'Formalize prediction: choose $f$ from data to minimize expected loss on new draws',
      assumptions: R`$(X_i,Y_i) \iid P$; test point drawn from the same $P$; a loss $L$ fixed in advance`,
      output: R`A predictor $\hat f \in \mathcal H$; its training error $\hat R_n(\hat f)$ and (estimated) risk $R(\hat f)$`,
      strengths: 'One vocabulary for regression and classification; clean decomposition of what can go wrong',
      weaknesses: 'IID assumption is fragile (distribution shift); $R(\hat f)$ is itself unknown and must be estimated',
      complexity: R`Depends on $\mathcal H$: closed form (least squares) to NP-hard (0–1 ERM over half-spaces)`,
    },
    sections: [
      {
        title: 'Prediction as a decision problem',
        html: R`
<p>We observe pairs $(X_1,Y_1),\dots,(X_n,Y_n) \iid P$ on $\mathcal X \times \mathcal Y$, and must produce a function $f:\mathcal X \to \mathcal Y$ (or into some action space $\mathcal A$) that will be applied to a <em>new</em> draw $(X,Y)\sim P$ for which only $X$ is revealed. How good a prediction $a$ is when the truth is $y$ is measured by a <strong>loss</strong> $L(y,a) \ge 0$.</p>
<div class="def" data-name="Risk and Bayes risk">
<p>The <em>risk</em> (expected loss, generalization error, test error) of a fixed $f$ is
$$R(f) = \E_{(X,Y)\sim P}\, L\big(Y, f(X)\big).$$
The <em>Bayes risk</em> is $R^* = \inf_f R(f)$, the infimum over all measurable $f$, and a minimizer $f^*$ (when it exists) is a <em>Bayes predictor</em>.</p>
</div>
<p>The loss encodes what you care about; changing it changes the target. The common choices:</p>
<table>
<thead><tr><th>Loss</th><th>$L(y,a)$</th><th>Setting</th><th>Bayes predictor $f^*(x)$</th></tr></thead>
<tbody>
<tr><td>Squared</td><td>$(y-a)^2$</td><td>Regression</td><td>$\E[Y\mid X=x]$</td></tr>
<tr><td>Absolute</td><td>$|y-a|$</td><td>Robust regression</td><td>$\operatorname{median}(Y\mid X=x)$</td></tr>
<tr><td>Pinball ($\tau$)</td><td>$(y-a)(\tau - \ind\{y < a\})$</td><td>Quantile regression</td><td>$\tau$-quantile of $Y\mid X=x$</td></tr>
<tr><td>0–1</td><td>$\ind\{y\ne a\}$</td><td>Classification</td><td>$\argmax_k \P(Y=k\mid X=x)$</td></tr>
<tr><td>Log loss</td><td>$-\log a(y)$, $a$ a distribution</td><td>Probabilistic prediction</td><td>$P(Y=\cdot\mid X=x)$ itself</td></tr>
</tbody>
</table>
<div class="note">
<p>This is exactly the <a href="#/t/decision-theory">decision-theoretic</a> setup, with one twist. In classical decision theory the risk is averaged over data for a fixed parameter; here the object being chosen is a whole function, the averaging is over a fresh test point, and nobody pretends $P$ belongs to a small parametric family. A learning <em>algorithm</em> is a map from samples to predictors, $D_n \mapsto \hat f_n$, and its risk $R(\hat f_n)$ is a random variable (random through $D_n$).</p>
</div>`,
      },
      {
        title: 'What the risk minimizer looks like',
        html: R`
<p>Because $R(f) = \E\big[\E[L(Y,f(X))\mid X]\big]$, we can minimize the risk one $x$ at a time: $f^*(x) \in \argmin_a \E[L(Y,a)\mid X=x]$, the minimizer of the <em>conditional risk</em>. (Measurability of the pointwise minimizer needs a measurable-selection argument in general; for the losses above it is explicit.) Each loss therefore picks out a functional of the conditional law of $Y$ given $X$.</p>
<div class="thm" data-name="The regression function minimizes squared risk">
<p>Suppose $\E Y^2 < \infty$ and let $m(x) = \E[Y\mid X=x]$. For every measurable $f$ with $\E f(X)^2 < \infty$,
$$\E\big(Y - f(X)\big)^2 = \E\big(Y - m(X)\big)^2 + \E\big(m(X) - f(X)\big)^2.$$
Hence $m$ is the Bayes predictor under squared loss, $R^* = \E\Var(Y\mid X)$, and the excess risk of any $f$ is its $L^2(P_X)$ distance from $m$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $Y - f(X) = (Y - m(X)) + (m(X) - f(X))$ and expand the square. The cross term is
$$2\,\E\big[(Y-m(X))(m(X)-f(X))\big] = 2\,\E\Big[(m(X)-f(X))\,\E\big[Y - m(X)\mid X\big]\Big] = 0,$$
using the tower property and the fact that $m(X)-f(X)$ is a function of $X$, so it comes out of the inner conditional expectation; and $\E[Y-m(X)\mid X] = m(X)-m(X) = 0$. All terms are finite by Cauchy–Schwarz. The second term on the right is nonnegative and zero iff $f = m$ $P_X$-a.e.</p>
</details>
<p>Geometrically, $m(X)$ is the orthogonal projection of $Y$ onto the closed subspace of $L^2$ functions of $X$ (see <a href="#/t/conditional-expectation">conditional expectation</a>); the theorem is Pythagoras.</p>
<div class="thm" data-kind="Proposition" data-name="Absolute loss targets the conditional median">
<p>If $\E|Y| < \infty$, any conditional median $\operatorname{med}(Y\mid X=x)$ minimizes $a \mapsto \E[|Y-a|\mid X=x]$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Fix $x$ and let $G$ be the conditional CDF. The function $g(a) = \E|Y-a|$ is convex with right derivative $g'_+(a) = \P(Y\le a) - \P(Y > a) = 2G(a)-1$ (differentiate under the integral; the kink at $Y=a$ contributes the boundary mass to the right derivative). A convex function is minimized where its right derivative changes sign from negative to nonnegative, i.e. at any $a$ with $G(a^-) \le 1/2 \le G(a)$: a median.</p>
</details>
<p>For 0–1 loss the conditional risk of predicting class $k$ is $1 - \P(Y=k\mid X=x)$, so the Bayes rule picks the most probable class; this is developed in <a href="#/t/bayes-classifier">the Bayes classifier</a>. For log loss, the conditional risk of reporting distribution $q$ is $H(p_x) + \KL(p_x\,\|\,q)$ where $p_x = P(Y=\cdot\mid X=x)$, minimized uniquely at $q = p_x$ — log loss is a <em>strictly proper scoring rule</em> (<a href="#/t/classifier-evaluation">evaluation</a>).</p>`,
      },
      {
        title: 'Empirical risk minimization',
        html: R`
<p>The risk cannot be computed, because $P$ is unknown. Its natural plug-in estimate replaces $P$ by the empirical distribution of the sample.</p>
<div class="def" data-name="Empirical risk, ERM">
<p>The <em>empirical risk</em> (training error) of $f$ is $\hat R_n(f) = \frac1n\sum_{i=1}^n L(Y_i, f(X_i))$. Given a <em>hypothesis class</em> $\mathcal H$, <em>empirical risk minimization</em> returns $\hat f \in \argmin_{f\in\mathcal H} \hat R_n(f)$. Write $f_{\mathcal H} \in \argmin_{f\in\mathcal H} R(f)$ for the best-in-class predictor (assume it exists).</p>
</div>
<p>Least squares is ERM with squared loss over linear functions; maximum likelihood is ERM with log loss over a parametric family; the perceptron, SVMs, trees and neural networks are all (regularized, surrogate-loss, approximate) ERM.</p>
<p>For any <em>fixed</em> $f$, $\hat R_n(f)$ is an average of IID terms, unbiased for $R(f)$ and consistent by the <a href="#/t/lln">law of large numbers</a>. The trouble is that $\hat f$ is not fixed: it was chosen because its training error is small, and that selection biases the training error downward.</p>
<div class="thm" data-kind="Proposition" data-name="Training error is optimistic">
<p>For ERM over $\mathcal H$,
$$\E\,\hat R_n(\hat f) \;\le\; R(f_{\mathcal H}) \;\le\; \E\,R(\hat f).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By definition of ERM, $\hat R_n(\hat f) \le \hat R_n(f_{\mathcal H})$ pointwise in the sample. Taking expectations, and using that $f_{\mathcal H}$ does not depend on the data, $\E\hat R_n(\hat f) \le \E \hat R_n(f_{\mathcal H}) = R(f_{\mathcal H})$. The right inequality holds pointwise because $\hat f\in\mathcal H$ and $f_{\mathcal H}$ minimizes $R$ over $\mathcal H$.</p>
</details>
<div class="ex" data-name="The memorizer">
<p>Let $\mathcal H$ be all functions and $X$ continuous. The rule "$\hat f(X_i) = Y_i$ on the training points, $\hat f(x) = 0$ elsewhere" has $\hat R_n(\hat f) = 0$ for 0–1 loss, yet for a fresh $X$ (which almost surely is not a training point) it predicts 0, so its risk is $\P(Y\ne 0)$, as bad as ignoring the data. ERM over an unrestricted class does not learn. Restricting $\mathcal H$ (or penalizing complexity) is not a technicality; it is where the inductive bias lives. The <a href="#/t/vc-theory">VC theory</a> makes "restricted enough" precise.</p>
</div>`,
      },
      {
        title: 'Approximation and estimation error',
        html: R`
<p>Fix a class $\mathcal H$ and let $\hat f$ be any learner with values in $\mathcal H$. Its excess risk splits telescopically:
$$R(\hat f) - R^* = \underbrace{R(\hat f) - R(f_{\mathcal H})}_{\text{estimation error}\;\ge 0} \;+\; \underbrace{R(f_{\mathcal H}) - R^*}_{\text{approximation error}\;\ge 0}.$$
The approximation error is deterministic and shrinks as $\mathcal H$ grows. The estimation error is random, and for a fixed $n$ it typically grows with $\mathcal H$, because there are more ways to fit noise. Choosing $\mathcal H$ is choosing a point on this trade-off; the squared-loss analogue at a single $x$ is the <a href="#/t/bias-variance">bias–variance decomposition</a>.</p>
<div class="thm" data-kind="Lemma" data-name="Estimation error is controlled by uniform deviation">
<p>For ERM over $\mathcal H$,
$$R(\hat f) - R(f_{\mathcal H}) \;\le\; 2\sup_{f\in\mathcal H}\big|\hat R_n(f) - R(f)\big|.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Add and subtract empirical risks:
$$R(\hat f) - R(f_{\mathcal H}) = \big[R(\hat f) - \hat R_n(\hat f)\big] + \big[\hat R_n(\hat f) - \hat R_n(f_{\mathcal H})\big] + \big[\hat R_n(f_{\mathcal H}) - R(f_{\mathcal H})\big].$$
The middle bracket is $\le 0$ by the definition of ERM, and each outer bracket is at most $\sup_{f\in\mathcal H}|\hat R_n(f) - R(f)|$.</p>
</details>
<p>This lemma is the bridge from statistics to learning theory: a sample is "good for ERM" if the empirical risk is uniformly close to the true risk over the whole class. Uniformity is the point — the LLN for each $f$ separately is not enough, because ERM gets to choose $f$ after seeing the data. Bounding $\sup_{f\in\mathcal H}|\hat R_n - R|$ is the subject of <a href="#/t/vc-theory">uniform convergence theory</a>.</p>
<div class="note" data-name="Structural risk minimization">
<p>Take a nested sequence $\mathcal H_1 \subset \mathcal H_2 \subset \cdots$ (polynomials of degree $d$, trees of depth $d$, networks of width $d$). Approximation error decreases in $d$; the uniform-deviation term increases. Vapnik's <em>structural risk minimization</em> picks $d$ to minimize training error plus a complexity penalty that upper-bounds the deviation; in practice the penalty is usually replaced by a held-out or <a href="#/t/cross-validation">cross-validated</a> estimate of risk. The lab shows the trade-off with polynomials: training error falls monotonically in degree, test error is U-shaped.</p>
</div>`,
        lab: 'learning-framework-complexity',
      },
      {
        title: 'How optimistic is the training error?',
        html: R`
<p>For squared loss with a fixed design there is an exact formula for the optimism. Let $x_1,\dots,x_n$ be fixed, $Y_i = m(x_i) + \varepsilon_i$ with $\varepsilon_i$ independent, mean zero, variance $\sigma^2$, and let $\hat Y_i = \hat f(x_i)$ be fitted values. Define the training error $\overline{\text{err}} = \frac1n\sum(Y_i-\hat Y_i)^2$ and the in-sample prediction error $\text{Err}_{\text{in}} = \frac1n\sum \E_{Y^0}(Y_i^0 - \hat Y_i)^2$, where $Y^0_i$ is an independent fresh response at the same $x_i$.</p>
<div class="thm" data-kind="Proposition" data-name="Covariance form of the optimism (Efron, 1986)">
<p>$$\E\,\text{Err}_{\text{in}} - \E\,\overline{\text{err}} = \frac2n\sum_{i=1}^n \Cov(\hat Y_i, Y_i).$$
For a linear smoother $\hat{\mathbf Y} = S\mathbf Y$ the right side is $2\sigma^2\tr(S)/n$; for least squares with $p$ linearly independent columns, $\tr(S) = p$ and the optimism is $2\sigma^2 p/n$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For each $i$, $Y_i^0$ is independent of $\hat Y_i$ and has the same mean and variance as $Y_i$. Expanding,
$$\E(Y_i^0 - \hat Y_i)^2 = \Var Y_i + \Var\hat Y_i + (\E Y_i - \E\hat Y_i)^2,$$
$$\E(Y_i - \hat Y_i)^2 = \Var Y_i + \Var\hat Y_i - 2\Cov(Y_i,\hat Y_i) + (\E Y_i - \E\hat Y_i)^2.$$
Subtract and average over $i$. For $\hat{\mathbf Y} = S\mathbf Y$ with $S$ fixed, $\Cov(\hat Y_i, Y_i) = \sum_j S_{ij}\Cov(Y_j,Y_i) = \sigma^2 S_{ii}$. For least squares $S = X(X^\top X)^{-1}X^\top$ is a projection of rank $p$, so $\tr S = p$.</p>
</details>
<p>Notice the result holds whether or not the model is correct — bias cancels. The quantity $\sum_i\Cov(\hat Y_i,Y_i)/\sigma^2$ is the <em>effective degrees of freedom</em> of the fitting procedure, and adding $2\sigma^2\,\text{df}/n$ to the training error is exactly Mallows' $C_p$ (and, for Gaussian likelihoods, AIC); see <a href="#/t/model-selection">model selection</a>.</p>
<div class="ex" data-name="Holdout estimation is honest">
<p>If $\hat f$ is trained on one sample and evaluated on an independent test set of size $m$ with a loss bounded in $[0,1]$, then conditional on $\hat f$ the test error is an average of $m$ IID terms with mean $R(\hat f)$, and Hoeffding's inequality gives $\P\big(|\hat R_{\text{test}} - R(\hat f)| > \epsilon\big) \le 2e^{-2m\epsilon^2}$. With $m = 1000$, $\epsilon = 0.043$ at 95%. This guarantee evaporates the moment the test set is used to choose among models: each look is a selection, and the minimum of many test errors is optimistic for the same reason training error is.</p>
</div>`,
      },
      {
        title: 'Surrogate losses, regularization, and failure modes',
        html: R`
<p>Two departures from pure ERM are nearly universal in practice.</p>
<p><strong>Surrogates.</strong> 0–1 ERM over half-spaces is NP-hard in the agnostic case, so classifiers of the form $h(x) = \sgn g(x)$ are trained by minimizing a convex <em>margin loss</em> $\phi(Y g(X))$ with $Y\in\{-1,+1\}$: hinge $\phi(t) = (1-t)_+$ (SVM), logistic $\log(1+e^{-t})$, exponential $e^{-t}$ (AdaBoost), squared $(1-t)^2$. Is it safe to optimize the wrong loss? The answer is a clean theorem.</p>
<div class="thm" data-name="Classification calibration (Zhang, 2004; Bartlett, Jordan & McAuliffe, 2006)">
<p>A convex $\phi$ is <em>classification-calibrated</em> — its pointwise minimizer $g^*(x)$ has the sign of $2\eta(x)-1$ whenever $\eta(x)\ne 1/2$ — if and only if $\phi$ is differentiable at $0$ with $\phi'(0) < 0$. In that case there is a nondecreasing $\psi$ with $\psi(0)=0$ such that for every $g$,
$$\psi\big(R_{0\text{–}1}(\sgn g) - R^*_{0\text{–}1}\big) \le R_\phi(g) - R^*_\phi.$$
For hinge loss $\psi(\theta) = |\theta|$; for squared loss $\psi(\theta) = \theta^2$.</p>
</div>
<p>So driving the surrogate excess risk to zero drives the 0–1 excess risk to zero, at a rate governed by $\psi$. All four losses above are calibrated.</p>
<p><strong>Regularization.</strong> Rather than fix a class, minimize $\hat R_n(f) + \lambda J(f)$ over a large class with a complexity penalty $J$ (a norm, a roughness integral, the number of leaves). By Lagrange duality this is ERM over the ball $\{J(f) \le c(\lambda)\}$, so $\lambda$ indexes a nested family of classes and plays the role of the complexity knob; see <a href="#/t/ridge-regression">ridge</a> and the <a href="#/t/lasso">lasso</a>.</p>
<div class="warn" data-name="When the framework's assumptions fail">
<p><strong>Distribution shift:</strong> every guarantee here is for test points from the training $P$. Covariate shift ($P_X$ changes, $P_{Y\mid X}$ does not) leaves the Bayes predictor intact but changes which regions of $\mathcal X$ matter; label shift and concept drift change the target itself. <strong>Leakage:</strong> preprocessing (feature selection, imputation, scaling) fitted on all the data before splitting makes the test set part of training. <strong>Dependence:</strong> grouped, spatial, or time-ordered data violate the IID premise; random splits then overstate performance and grouped or blocked splits are needed.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Under squared loss, the Bayes risk is $\E\Var(Y\mid X)$. What is the Bayes risk under absolute loss, and why can it not in general be expressed through variances?`,
        a: R`It is $\E\big[\E(|Y - \operatorname{med}(Y\mid X)|\;\big|\; X)\big]$, the expected conditional mean absolute deviation about the conditional median. Each loss has its own conditional "spread" functional; the variance belongs to squared loss because squared loss is minimized by the mean and the minimum value of $\E(Y-a)^2$ is $\Var Y$.` },
      { q: R`Explain why $\E\,\hat R_n(\hat f) \le R(f_{\mathcal H})$ even though $\hat R_n(f)$ is unbiased for $R(f)$ for every fixed $f$.`,
        a: R`Unbiasedness holds for a data-independent $f$. ERM picks $\hat f$ to minimize $\hat R_n$, so $\hat R_n(\hat f) \le \hat R_n(f_{\mathcal H})$ on every sample; taking expectations gives the inequality. The training error of the selected predictor is the minimum of many noisy estimates, and minima of noisy estimates are biased low.` },
      { q: R`You enlarge the hypothesis class from $\mathcal H$ to $\mathcal H' \supset \mathcal H$. Which term of the excess-risk decomposition can only go down, and which can go up?`,
        a: R`The approximation error $R(f_{\mathcal H}) - R^*$ can only decrease (the infimum is over a larger set). The estimation error $R(\hat f) - R(f_{\mathcal H})$ is not monotone in general but typically increases, because the uniform deviation $\sup_{f\in\mathcal H'}|\hat R_n - R|$ that controls it is a supremum over a larger set.` },
      { q: R`For least squares with $p = 20$ predictors, $n = 100$ observations and noise variance $\sigma^2 = 4$, by how much does the expected in-sample prediction error exceed the expected training error? Does the answer require the linear model to be correct?`,
        a: R`By $2\sigma^2 p/n = 2\cdot 4\cdot 20/100 = 1.6$. No: the covariance formula $\frac2n\sum\Cov(\hat Y_i,Y_i)$ holds whatever the mean function is, because the bias terms appear identically in both expected errors and cancel.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'bias-variance',
    domain: 'supervised',
    order: 20,
    title: 'The Bias–Variance Decomposition',
    subtitle: 'Why the expected test error of a learner splits into noise, systematic error, and instability',
    level: 2,
    sources: { aos: '20.1, 22.8', other: 'ESL §2.9, §7.3; Bishop §3.2; Geman, Bienenstock & Doursat (1992)' },
    prereqs: ['learning-framework', 'point-estimation'],
    related: ['ridge-regression', 'knn', 'kernel-regression', 'bagging-random-forests', 'regularization-deep'],
    tags: ['bias', 'variance', 'model complexity', 'overfitting', 'ridge', 'k-NN'],
    summary: R`Under squared loss, the expected prediction error of a learning procedure at a point $x$ — averaged over the training sets it might have seen and over the noise in the new response — is exactly <em>noise</em> $+$ <em>squared bias</em> $+$ <em>variance</em>. Noise is the Bayes risk and cannot be touched. Bias measures how far the average fitted function is from the truth; variance measures how much the fit moves when the training set is redrawn. Flexible procedures trade the first for the second, and nearly every tuning parameter in statistics (bandwidth, $k$, $\lambda$, degree, depth) is a dial on this trade-off.`,
    survey: {
      task: 'Explain and quantify the expected test error of a procedure as a function of its complexity',
      assumptions: R`Squared loss; new $(x, Y)$ independent of the training set; $\Var(Y\mid X=x) = \sigma^2(x)$ finite`,
      output: R`$\E(Y-\hat f(x))^2 = \sigma^2(x) + \Bias^2 + \Var$; optimal complexity balancing the two`,
      strengths: 'Exact identity; explains overfitting, shrinkage, bagging, and smoothing rates',
      weaknesses: 'Not additive for 0–1 loss; the components are unobservable from one training set',
      complexity: 'Conceptual; estimated by simulation with many training sets',
    },
    sections: [
      {
        title: 'The decomposition at a point',
        html: R`
<p>Let $Y = f(X) + \varepsilon$ with $\E[\varepsilon\mid X] = 0$ and $\Var(\varepsilon\mid X = x) = \sigma^2(x)$. A learning procedure maps a training set $D = \{(X_i,Y_i)\}_{i=1}^n$ to a function $\hat f_D$. Fix a query point $x$ and draw a new response $Y = f(x)+\varepsilon$ independent of $D$. The quantity we care about is the expected squared prediction error, averaged over both sources of randomness: the training set and the new noise.</p>
<div class="thm" data-name="Bias–variance–noise decomposition">
<p>Let $\bar f(x) = \E_D \hat f_D(x)$ and suppose $\E_D \hat f_D(x)^2 < \infty$. Then
$$\E_{D,\varepsilon}\big(Y - \hat f_D(x)\big)^2 = \underbrace{\sigma^2(x)}_{\text{noise}} + \underbrace{\big(\bar f(x) - f(x)\big)^2}_{\text{bias}^2} + \underbrace{\E_D\big(\hat f_D(x) - \bar f(x)\big)^2}_{\text{variance}}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $Y - \hat f_D(x) = \varepsilon + \big(f(x) - \bar f(x)\big) + \big(\bar f(x) - \hat f_D(x)\big)$, a sum of three terms $A + B + C$. Square and take expectations. $\E A^2 = \sigma^2(x)$; $B$ is a constant; $\E C^2$ is the variance.</p>
<p>Cross terms: $\E[AB] = B\,\E\varepsilon = 0$. $\E[AC] = \E[\varepsilon]\,\E[C] = 0$ because the new noise $\varepsilon$ is independent of $D$, and $C$ is a function of $D$ alone. $\E[BC] = B\,\E_D[\bar f(x) - \hat f_D(x)] = 0$ by the definition of $\bar f$. Only the three squares survive.</p>
</details>
<p>The middle two terms are exactly the <a href="#/t/point-estimation">MSE of $\hat f_D(x)$ as an estimator</a> of the number $f(x)$: $\MSE = \Bias^2 + \Var$. The noise term is what is left even when $\hat f = f$; it is the <a href="#/t/learning-framework">Bayes risk</a> at $x$.</p>
<div class="note">
<p>The variance is <em>not</em> the noise variance. It is the variance of the procedure's output across hypothetical training sets — how much your fitted curve would wiggle if you collected the data again. A procedure can have zero variance (always predict $0$) and enormous bias, or near-zero bias (interpolate) and enormous variance.</p>
</div>`,
      },
      {
        title: 'Integrating over the input distribution',
        html: R`
<p>The expected test error of the procedure is the average of the pointwise identity over a test point $X\sim P_X$:
$$\E_D\, R(\hat f_D) = \E_X\sigma^2(X) + \E_X\big(\bar f(X) - f(X)\big)^2 + \E_X\Var_D\big(\hat f_D(X)\big).$$
This is an <em>unconditional</em> error: it averages over training sets. Any one analyst has a single $D$ and cares about the conditional error $R(\hat f_D)$; the decomposition describes what the procedure does on average and explains why some procedures are reliably better than others.</p>
<p>Two remarks on what is random. In a <strong>fixed design</strong> the training inputs $x_1,\dots,x_n$ are held fixed and only the responses are redrawn; bias and variance are then conditional on the design, which makes calculations clean (both worked examples below). In a <strong>random design</strong> the inputs are redrawn too, so the variance includes how the fit changes when the design changes — for local methods this is substantial wherever data are sparse.</p>
<div class="warn" data-name="Decomposing a single training set is impossible">
<p>Bias and variance are properties of a <em>distribution</em> over training sets. From one data set you cannot compute them; you can only estimate their sum (minus noise) by held-out data or <a href="#/t/cross-validation">cross-validation</a>, or estimate them by resampling — e.g. the <a href="#/t/bootstrap">bootstrap</a> variance of $\hat f(x)$ — under extra assumptions. The decomposition guides design; it is not a diagnostic you read off one fit.</p>
</div>`,
      },
      {
        title: 'Two exact decompositions: k-NN and ridge',
        html: R`
<p><strong>$k$-nearest-neighbour regression.</strong> Fixed design, homoscedastic noise. The fit at $x$ is $\hat f(x) = \frac1k\sum_{\ell=1}^k Y_{(\ell)}$, where $(\ell)$ indexes the $\ell$-th nearest training input $x_{(\ell)}$ to $x$. Because the design is fixed, the neighbour set is fixed, and
$$\E\big(Y - \hat f(x)\big)^2 = \sigma^2 + \Big(f(x) - \frac1k\sum_{\ell=1}^k f(x_{(\ell)})\Big)^2 + \frac{\sigma^2}{k}.$$
The variance is $\sigma^2/k$ (an average of $k$ independent noises). The bias is the difference between $f(x)$ and the average of $f$ over the neighbourhood, which grows with $k$ as the neighbourhood widens. $k$ is a pure bias–variance dial; $k = 1$ has variance $\sigma^2$, as much as the noise itself.</p>
<p><strong>Ridge regression.</strong> Fixed design $X$ ($n\times p$), $\mathbf Y = X\beta + \boldsymbol\varepsilon$, $\Cov\boldsymbol\varepsilon = \sigma^2 I$, and $\hat\beta_\lambda = (X^\top X + \lambda I)^{-1}X^\top \mathbf Y$. Take the SVD $X = UDV^\top$ with singular values $d_1,\dots,d_p > 0$ and rotate: $\theta = V^\top\beta$, $\hat\theta_\lambda = V^\top\hat\beta_\lambda$. Then the problem decouples into $p$ independent scalar problems:</p>
$$\hat\theta_{\lambda,j} = \frac{d_j^2}{d_j^2 + \lambda}\,\hat\theta_{\text{OLS},j},\qquad \Bias(\hat\theta_{\lambda,j}) = -\frac{\lambda}{d_j^2 + \lambda}\,\theta_j,\qquad \Var(\hat\theta_{\lambda,j}) = \frac{\sigma^2 d_j^2}{(d_j^2+\lambda)^2}.$$
<p>(Derivation: $\hat\theta_{\text{OLS},j} = \theta_j + u_j^\top\boldsymbol\varepsilon/d_j$ has mean $\theta_j$ and variance $\sigma^2/d_j^2$; multiply by the shrinkage factor $s_j = d_j^2/(d_j^2+\lambda)$.) Directions with small singular values — poorly determined by the design — are shrunk hardest, which is precisely where OLS variance $\sigma^2/d_j^2$ is largest.</p>
<div class="thm" data-name="Ridge beats least squares for some λ (Hoerl & Kennard, 1970)">
<p>Let $M(\lambda) = \E\|\hat\beta_\lambda - \beta\|^2$. If $\sigma^2 > 0$, then $M'(0) < 0$; hence there is $\lambda > 0$ with $M(\lambda) < M(0) = \MSE(\hat\beta_{\text{OLS}})$, for every $\beta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$V$ is orthogonal, so $M(\lambda) = \sum_j \E(\hat\theta_{\lambda,j}-\theta_j)^2 = \sum_j \dfrac{\lambda^2\theta_j^2 + \sigma^2 d_j^2}{(d_j^2+\lambda)^2}$. Differentiate term by term:
$$\frac{\d}{\d\lambda}\frac{\lambda^2\theta_j^2 + \sigma^2 d_j^2}{(d_j^2+\lambda)^2} = \frac{2\lambda\theta_j^2(d_j^2 + \lambda) - 2(\lambda^2\theta_j^2 + \sigma^2 d_j^2)}{(d_j^2+\lambda)^3} = \frac{2 d_j^2(\lambda\theta_j^2 - \sigma^2)}{(d_j^2+\lambda)^3}.$$
At $\lambda = 0$ this is $-2\sigma^2/d_j^4 < 0$ for each $j$. Since $M$ is differentiable on $[0,\infty)$ with $M'(0) < 0$, it decreases on some interval $(0,\lambda_0)$.</p>
</details>
<p>The derivative also shows each coordinate's MSE is minimized at $\lambda_j = \sigma^2/\theta_j^2$ — the inverse signal-to-noise ratio — which is why ridge is the posterior mean under a Gaussian prior $\beta\sim\N(0,(\sigma^2/\lambda)I)$. See <a href="#/t/ridge-regression">ridge regression</a>.</p>`,
      },
      {
        title: 'Watching the trade-off',
        html: R`
<p>The lab simulates the whole thought experiment: draw many training sets from the same world, fit the same procedure to each, and look at the fan of fitted curves. The spread of the fan at each $x$ is the variance; the gap between the fan's centre and the truth is the bias. The right panel integrates both over $x$ and plots them against complexity for three procedures — polynomial degree, $k$-NN (complexity increases as $k$ decreases), and ridge on a degree-10 polynomial basis (complexity increases as $\lambda$ decreases).</p>
<p>Things to try: shrink $n$ and watch the variance curve rise and the optimum move toward simpler models; raise the noise and watch the same thing happen; switch to a smooth truth and see that low-degree fits have little bias to begin with.</p>`,
        lab: 'bias-variance-decomposition',
      },
      {
        title: 'Optimal complexity and rates of convergence',
        html: R`
<p>The decomposition explains not only <em>that</em> there is an optimal complexity but how fast the best achievable error shrinks with $n$. Consider $k$-NN regression with inputs uniform on $[0,1]^d$, $f$ Lipschitz with constant $L$, and noise variance $\sigma^2$. The $k$ nearest neighbours of $x$ lie within distance $r_k \asymp (k/n)^{1/d}$ (a ball containing $k$ of $n$ uniform points has volume about $k/n$). Hence
$$\Bias^2 \lesssim L^2 r_k^2 \asymp L^2 (k/n)^{2/d}, \qquad \Var \approx \frac{\sigma^2}{k}.$$
Minimizing $\sigma^2/k + L^2(k/n)^{2/d}$ over $k$ gives $k^* \asymp n^{2/(d+2)}$ and
$$\MSE^* \asymp n^{-2/(d+2)}.$$
In one dimension this is $n^{-2/3}$; in ten dimensions $n^{-1/6}$ — to halve the error you need $2^6 = 64$ times as much data. This is the <a href="#/t/knn">curse of dimensionality</a> in quantitative form, and the same calculation with a bandwidth $h$ in place of $(k/n)^{1/d}$ gives the classic rates for <a href="#/t/kernel-density">kernel density</a> and <a href="#/t/kernel-regression">kernel regression</a> estimators ($n^{-4/(d+4)}$ under two derivatives of smoothness, because a symmetric kernel cancels the first-order bias term).</p>
<div class="note" data-name="Pattern">
<p>Every nonparametric rate follows the same template: bias decreasing as a power of the smoothing parameter, variance increasing as a power of it, and the optimum where the two powers balance. More smoothness of $f$ makes the bias smaller and the rate faster; more dimensions make each neighbourhood emptier and the rate slower.</p>
</div>`,
      },
      {
        title: 'Beyond squared loss and the classical U-curve',
        html: R`
<p><strong>Classification.</strong> Under 0–1 loss there is no additive decomposition. Consider a plug-in classifier $\hat h(x) = \ind\{\hat\eta(x) > 1/2\}$. At a point where the truth is $\eta(x) > 1/2$, the error is driven by $\P(\hat\eta(x) < 1/2)$. If $\hat\eta(x)$ is approximately normal, this is about $\Phi\big((1/2 - \E\hat\eta(x))/\operatorname{sd}\hat\eta(x)\big)$. When the average estimate is on the correct side of $1/2$ (the bias is "harmless"), shrinking the variance drives the error to zero; when the average estimate is on the wrong side, <em>more</em> variance actually helps. Bias and variance interact multiplicatively (Friedman, 1997). This is one reason heavily biased classifiers like <a href="#/t/naive-bayes">naive Bayes</a> can classify well: only the side of $1/2$ matters, not the size of the bias.</p>
<p><strong>Variance reduction by averaging.</strong> If $\hat f_1,\dots,\hat f_B$ are identically distributed with pairwise correlation $\rho$ and variance $v$, their average has variance $\rho v + (1-\rho)v/B$ and the same bias. <a href="#/t/bagging-random-forests">Bagging and random forests</a> exploit this: grow low-bias, high-variance trees and average them, decorrelating to lower $\rho$.</p>
<div class="warn" data-name="The U-curve is not a law">
<p>The textbook picture — test error U-shaped in complexity — describes classical regimes. For models with more parameters than observations, the test error of the minimum-norm interpolating fit often peaks at the interpolation threshold ($p \approx n$) and then <em>decreases</em> again as $p$ grows: <strong>double descent</strong> (Belkin, Hsu, Ma & Mandal, 2019; Hastie, Montanari, Rosset & Tibshirani, 2022). The decomposition still holds exactly at every $p$; what fails is the heuristic that "complexity" is measured by parameter count. Past the threshold, the minimum-norm solution is implicitly regularized and its variance falls. See <a href="#/t/regularization-deep">regularizing deep networks</a>.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In the decomposition $\E(Y - \hat f_D(x))^2 = \sigma^2 + \Bias^2 + \Var$, which cross term vanishes because the test response is independent of the training set, and which vanishes by the definition of $\bar f$?`,
        a: R`$\E[\varepsilon\,(\bar f(x) - \hat f_D(x))] = \E\varepsilon\cdot\E(\bar f(x)-\hat f_D(x)) = 0$ uses independence of the new noise from $D$. $\E[(f(x) - \bar f(x))(\bar f(x) - \hat f_D(x))] = 0$ because $f(x)-\bar f(x)$ is constant and $\E_D\hat f_D(x) = \bar f(x)$.` },
      { q: R`For fixed-design $k$-NN regression with $\sigma^2 = 1$, what is the variance of $\hat f(x)$ for $k=1$ and $k=25$? What happens to the bias as $k \to n$?`,
        a: R`$\sigma^2/k$: $1$ and $0.04$. As $k\to n$ the fit becomes the global mean $\bar Y$ at every $x$, so $\bar f(x) \to \frac1n\sum f(x_i)$ and the bias tends to $f(x)$ minus the average of $f$ over the design — large wherever $f$ is not constant.` },
      { q: R`In the ridge eigenbasis, which coordinate of $\beta$ is shrunk most, and why is that a sensible place to shrink?`,
        a: R`The coordinate along the right singular vector with the smallest singular value $d_j$, shrunk by $d_j^2/(d_j^2+\lambda)$. That direction is the least determined by the design: its OLS variance $\sigma^2/d_j^2$ is the largest, so a little bias buys the biggest reduction in variance.` },
      { q: R`Why does the optimal $k$-NN error rate $n^{-2/(d+2)}$ deteriorate with dimension? Identify which of bias or variance is responsible.`,
        a: R`The bias. For a fixed number $k$ of neighbours (hence fixed variance $\sigma^2/k$), the neighbourhood radius scales as $(k/n)^{1/d}$, which shrinks very slowly with $n$ when $d$ is large; the bias, of order $L^2 (k/n)^{2/d}$, therefore decays slowly, forcing a compromise with a worse exponent.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'bayes-classifier',
    domain: 'supervised',
    order: 30,
    title: 'The Bayes Classifier & Optimal Decisions',
    subtitle: 'The best possible classifier, why it is best, and how close a plug-in estimate gets',
    level: 2,
    sources: { aos: '22.2', other: 'Devroye, Györfi & Lugosi (1996) Ch. 2; ESL §2.4; Bishop §1.5' },
    prereqs: ['learning-framework', 'conditional-probability'],
    related: ['lda-qda', 'naive-bayes', 'knn', 'decision-theory', 'neyman-pearson', 'classifier-evaluation'],
    tags: ['Bayes risk', '0–1 loss', 'posterior', 'plug-in', 'cost-sensitive', 'margin'],
    summary: R`If you knew the joint law of $(X,Y)$, the best classifier would be obvious: predict the most probable class given $X$. For two classes this is $h^*(x) = \ind\{\eta(x) > 1/2\}$ with $\eta(x) = \P(Y=1\mid X=x)$, and its error $R^*$ — the <em>Bayes error</em> — is the floor beneath every classifier. The proof gives more than optimality: it expresses the excess risk of any classifier as an integral of $|2\eta - 1|$ over the region where it disagrees with $h^*$. That identity shows how priors and costs move the threshold, and why a plug-in rule $\ind\{\hat\eta > 1/2\}$ has excess risk at most $2\,\E|\hat\eta(X) - \eta(X)|$: classification is no harder than regression, and often much easier.`,
    survey: {
      task: 'Characterize the optimal classifier and the irreducible error under 0–1 (or cost-weighted) loss',
      assumptions: R`The joint law of $(X,Y)$ is known (theory) or $\eta$ is estimated (plug-in)`,
      output: R`$h^*(x) = \argmax_k \P(Y=k\mid X=x)$; Bayes error $R^* = \E\min(\eta,1-\eta)$`,
      strengths: 'Benchmark for every classifier; reduces classification to estimating $\eta$ or a likelihood ratio',
      weaknesses: R`Requires $\eta$, which is unknown; estimating it well in high dimension is hard`,
      complexity: R`Evaluating $h^*$ costs one posterior computation; plug-in cost is that of estimating $\hat\eta$`,
    },
    sections: [
      {
        title: 'The setting and the regression function η',
        html: R`
<p>Let $(X,Y)$ take values in $\mathcal X\times\{0,1\}$. Everything about the joint law that matters for classification is captured by two objects: the marginal $P_X$ and the <strong>regression function</strong>
$$\eta(x) = \P(Y = 1\mid X = x) = \E[Y\mid X=x].$$
Equivalently, with class priors $\pi_k = \P(Y=k)$ and class-conditional densities $f_k$ (with respect to a common dominating measure), Bayes' theorem gives
$$\eta(x) = \frac{\pi_1 f_1(x)}{\pi_0 f_0(x) + \pi_1 f_1(x)}.$$
The risk of a classifier $h:\mathcal X\to\{0,1\}$ under 0–1 loss is its misclassification probability $R(h) = \P(h(X)\ne Y)$.</p>
<div class="def" data-name="Bayes classifier, Bayes error">
<p>The <em>Bayes classifier</em> is $h^*(x) = \ind\{\eta(x) > 1/2\}$ (ties may be broken arbitrarily). The <em>Bayes error</em> is $R^* = R(h^*)$.</p>
</div>
<p>Conditional on $X = x$, predicting $1$ errs with probability $1 - \eta(x)$ and predicting $0$ errs with probability $\eta(x)$. So the conditional error of any $h$ at $x$ is $\eta(x)\ind\{h(x)=0\} + (1-\eta(x))\ind\{h(x)=1\}$, and the Bayes rule picks whichever of the two is smaller.</p>`,
      },
      {
        title: 'Optimality of the Bayes rule',
        html: R`
<div class="thm" data-name="Excess-risk identity">
<p>For every classifier $h$,
$$R(h) - R^* = \E\Big[\,|2\eta(X) - 1|\;\ind\{h(X)\ne h^*(X)\}\Big] \;\ge\; 0.$$
In particular $h^*$ minimizes $R$ over all measurable classifiers, and $R^* = \E\min\{\eta(X), 1-\eta(X)\}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the tower property, $R(h) = \E\big[\eta(X)\ind\{h(X)=0\} + (1-\eta(X))\ind\{h(X)=1\}\big]$. Subtract the same expression for $h^*$ and look at a fixed $x$. If $h(x) = h^*(x)$ the integrands agree. If $h(x) = 1$, $h^*(x) = 0$ (so $\eta(x)\le 1/2$), the difference is $(1-\eta(x)) - \eta(x) = 1 - 2\eta(x) = |2\eta(x)-1|$. If $h(x)=0$, $h^*(x)=1$ (so $\eta(x) > 1/2$), the difference is $\eta(x) - (1-\eta(x)) = |2\eta(x)-1|$. Integrating over $P_X$ gives the identity. Finally the conditional error of $h^*$ at $x$ is $\min(\eta(x),1-\eta(x))$.</p>
</details>
<p>The identity tells you <em>where</em> mistakes hurt. Disagreeing with the Bayes rule costs nothing where $\eta(x) = 1/2$ (a coin flip either way) and costs most where $\eta$ is near $0$ or $1$. This is why classification can be easier than estimating $\eta$: errors in $\hat\eta$ matter only near the boundary, and there they are cheap.</p>
<p>In density form, $h^*(x) = 1$ iff $\pi_1 f_1(x) > \pi_0 f_0(x)$, and
$$R^* = \int \min\{\pi_0 f_0(x),\, \pi_1 f_1(x)\}\d x.$$
With equal priors, $\min(a,b) = (a+b-|a-b|)/2$ gives $R^* = \tfrac12\big(1 - \operatorname{TV}(f_0,f_1)\big)$, where $\operatorname{TV}(f_0,f_1) = \frac12\int|f_0 - f_1|$: the Bayes error is a direct measure of how distinguishable the classes are.</p>
<div class="ex" data-name="Two Gaussians with common covariance">
<p>Let $X\mid Y=k \sim \N(\mu_k, \Sigma)$ and $\Delta^2 = (\mu_1-\mu_0)^\top\Sigma^{-1}(\mu_1-\mu_0)$ (squared Mahalanobis distance). The log-likelihood ratio is
$$\delta(x) = \log\frac{f_1(x)}{f_0(x)} = \Big(x - \tfrac{\mu_0+\mu_1}{2}\Big)^\top\Sigma^{-1}(\mu_1-\mu_0),$$
and $h^*(x) = \ind\{\delta(x) > c\}$ with $c = \log(\pi_0/\pi_1)$. Under $Y=1$, $\delta(X)\sim\N(\Delta^2/2, \Delta^2)$; under $Y=0$, $\delta(X)\sim\N(-\Delta^2/2,\Delta^2)$. Therefore
$$R^* = \pi_1\,\Phi\!\Big(\frac{c}{\Delta} - \frac{\Delta}{2}\Big) + \pi_0\,\Phi\!\Big(-\frac{c}{\Delta} - \frac{\Delta}{2}\Big),$$
which reduces to $\Phi(-\Delta/2)$ for equal priors. The Bayes error depends on the means and covariance only through one number, $\Delta$. With $\Delta = 2$, $R^* = \Phi(-1) \approx 0.159$.</p>
</div>`,
      },
      {
        title: 'Priors, costs, many classes, and rejection',
        html: R`
<p><strong>Unequal costs.</strong> Suppose a false positive (predict 1, truth 0) costs $c_{\text{FP}}$ and a false negative costs $c_{\text{FN}}$. The conditional expected cost of predicting $1$ is $c_{\text{FP}}(1-\eta(x))$ and of predicting $0$ is $c_{\text{FN}}\eta(x)$. Repeating the proof above with these weights, the Bayes rule is
$$h^*_c(x) = \ind\Big\{\eta(x) > \frac{c_{\text{FP}}}{c_{\text{FP}} + c_{\text{FN}}}\Big\} \iff \frac{f_1(x)}{f_0(x)} > \frac{c_{\text{FP}}\,\pi_0}{c_{\text{FN}}\,\pi_1}.$$
Costs and priors enter only through the product $c_{\text{FP}}\pi_0/(c_{\text{FN}}\pi_1)$: doubling the cost of a missed positive is indistinguishable from doubling its prior.</p>
<div class="note" data-name="Every Bayes rule is a likelihood-ratio test">
<p>As priors and costs vary, the Bayes rules sweep out the family $\{\ind\{f_1/f_0 > t\}: t\ge 0\}$ — exactly the family of most powerful tests in the <a href="#/t/neyman-pearson">Neyman–Pearson lemma</a>. The Bayesian chooses $t$ from priors and costs; the Neyman–Pearson statistician chooses it to hit a size $\alpha$. Plotting the error rates of this family gives the optimal <a href="#/t/classifier-evaluation">ROC curve</a>.</p>
</div>
<p><strong>$K$ classes.</strong> With $p_k(x) = \P(Y=k\mid X=x)$, the conditional error of predicting $k$ is $1 - p_k(x)$, so $h^*(x) = \argmax_k p_k(x) = \argmax_k \pi_k f_k(x)$, and $R^* = 1 - \E\max_k p_k(X)$. With a general loss matrix $C_{jk}$ (cost of predicting $k$ when the truth is $j$), $h^*(x) = \argmin_k\sum_j C_{jk}\,p_j(x)$.</p>
<p><strong>Reject option (Chow, 1970).</strong> If abstaining costs $d \in (0, 1/2)$, the optimal rule predicts $\argmax_k p_k(x)$ when $\max_k p_k(x) \ge 1 - d$ and abstains otherwise — it hands off exactly the cases where it is least sure, as measured by the posterior.</p>`,
      },
      {
        title: 'A Gaussian world you can push around',
        html: R`
<p>Below, class 0 is $\N(0,1)$ and class 1 is $\N(\mu, s^2)$ on the real line. The left panel shows the cost-weighted joint densities $c_{\text{FP}}\pi_0 f_0$ and $c_{\text{FN}}\pi_1 f_1$; the Bayes rule predicts whichever is higher, and the shaded area under the lower envelope is the Bayes risk. The right panel shows $\eta(x)$ and the cost threshold. You also control a simple rule "predict 1 when $x > t$" and can see its excess risk.</p>
<p>Try: (i) $s = 1$ — the Bayes region is a half-line, a single threshold achieves $R^*$; (ii) $s \ne 1$ — the log-likelihood ratio is quadratic, the Bayes region becomes an interval or its complement, and <em>no</em> single threshold is optimal; (iii) shift the prior or the cost ratio and watch the boundary slide by $\log(c_{\text{FP}}\pi_0/c_{\text{FN}}\pi_1)$ in log-likelihood-ratio units.</p>`,
        lab: 'bayes-classifier-gaussians',
      },
      {
        title: 'Plug-in classifiers and their excess risk',
        html: R`
<p>In practice $\eta$ is unknown. The <strong>plug-in</strong> strategy estimates it by any regression method and thresholds: $\hat h(x) = \ind\{\hat\eta(x) > 1/2\}$. How good must $\hat\eta$ be?</p>
<div class="thm" data-name="Plug-in excess risk bound">
<p>For any measurable $\hat\eta:\mathcal X\to[0,1]$ and $\hat h = \ind\{\hat\eta > 1/2\}$,
$$R(\hat h) - R^* \;\le\; 2\,\E\big|\hat\eta(X) - \eta(X)\big| \;\le\; 2\sqrt{\E\big(\hat\eta(X) - \eta(X)\big)^2}.$$
(For a data-dependent $\hat\eta$, the bound holds conditionally on the training data, with $X$ an independent test point.)</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the excess-risk identity it suffices to show $|2\eta(x) - 1| \le 2|\hat\eta(x) - \eta(x)|$ whenever $\hat h(x)\ne h^*(x)$. Disagreement means $\eta(x)$ and $\hat\eta(x)$ lie on (weakly) opposite sides of $1/2$, so $1/2$ lies between them and $|\eta(x) - 1/2| \le |\eta(x) - \hat\eta(x)|$. Multiply by $2$. Then
$$R(\hat h) - R^* = \E\big[|2\eta - 1|\,\ind\{\hat h\ne h^*\}\big] \le 2\,\E\big[|\hat\eta - \eta|\,\ind\{\hat h\ne h^*\}\big] \le 2\,\E|\hat\eta - \eta|.$$
The last step is Cauchy–Schwarz (or Jensen).</p>
</details>
<p>Consequently any $L^1$- or $L^2$-consistent regression estimate yields a consistent classifier, and the classification rate is at least as fast as the regression rate. Usually it is much faster, because the indicator $\ind\{\hat h\ne h^*\}$ — which the proof throws away — is zero except near the boundary $\{\eta = 1/2\}$.</p>
<div class="thm" data-kind="Corollary" data-name="Faster rates under a margin condition (Mammen & Tsybakov, 1999; Audibert & Tsybakov, 2007)">
<p>Suppose $\P\big(0 < |\eta(X) - 1/2| \le t\big) \le C t^{\alpha}$ for all $t > 0$ (Tsybakov's margin condition with exponent $\alpha\ge 0$). If $\sup_x|\hat\eta(x) - \eta(x)| \le \tau$, then $R(\hat h) - R^* \le 2C\,\tau^{1+\alpha}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>On $\{\hat h \ne h^*\}$ we showed $|\eta - 1/2| \le |\hat\eta - \eta| \le \tau$, so $|2\eta - 1| \le 2\tau$ there, and points with $\eta = 1/2$ contribute zero. Hence the excess risk is at most $2\tau\,\P(0 < |\eta(X)-1/2|\le\tau) \le 2C\tau^{1+\alpha}$.</p>
</details>
<p>If $\eta$ crosses $1/2$ steeply (large $\alpha$), the sup-norm error $\tau$ is raised to a power above one, and plug-in classification converges faster than the regression estimate it is built on. The lab below estimates $\eta$ on $[0,1]$ with a histogram and compares the actual excess risk to the bound $2\E|\hat\eta - \eta|$ as $n$ grows.</p>`,
        lab: 'bayes-classifier-plugin',
      },
      {
        title: 'Three roads to a classifier',
        html: R`
<p>The Bayes classifier organizes the whole field. Every practical method approximates it by one of three routes:</p>
<div class="cols">
<div><h4>Generative</h4><p>Model $\pi_k$ and $f_k$, then apply Bayes' theorem. Gaussian $f_k$ give <a href="#/t/lda-qda">LDA and QDA</a>; independent coordinates give <a href="#/t/naive-bayes">naive Bayes</a>. Efficient when the model is right; uses unlabeled $X$ naturally.</p></div>
<div><h4>Discriminative</h4><p>Model $\eta(x)$ directly, ignoring $P_X$: <a href="#/t/logistic-regression">logistic regression</a>, <a href="#/t/knn">$k$-NN</a> voting, <a href="#/t/decision-trees">trees</a>, neural nets with softmax output. Fewer assumptions; probabilities available for thresholding.</p></div>
<div><h4>Direct</h4><p>Skip probabilities and minimize (a surrogate for) the empirical 0–1 risk over a class of boundaries: the perceptron, <a href="#/t/svm">SVMs</a>, boosting. Theory via <a href="#/t/vc-theory">uniform convergence</a>; probabilities need post-hoc calibration.</p></div>
</div>
<div class="warn" data-name="The Bayes error is not zero, and it is not known">
<p>$R^*$ is a property of the population, not of any model: if classes overlap, no amount of data or model capacity can beat it. A training error well below a credible estimate of $R^*$ signals overfitting, not brilliance. Estimating $R^*$ itself is hard; the <a href="#/t/knn">Cover–Hart</a> inequality $R^*\le R_{1\text{NN}}\le 2R^*(1-R^*)$ gives a rough bracket from the asymptotic nearest-neighbour error.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that the Bayes error satisfies $R^* \le \min(\pi_0, \pi_1)$, and say when equality holds.`,
        a: R`$R^* = \E\min(\eta, 1-\eta) \le \min(\E\eta, \E(1-\eta)) = \min(\pi_1,\pi_0)$, since the minimum of expectations dominates the expectation of the minimum. Equality holds iff $\eta(X) - 1/2$ has a constant sign a.s. — the features never push the posterior across $1/2$, so always predicting the majority class is optimal (e.g. when $X$ is independent of $Y$).` },
      { q: R`A screening test should flag a disease (prevalence 2%) whenever the expected cost of not flagging exceeds that of flagging. A missed case costs 50 times a false alarm. What posterior threshold should be used, and what likelihood-ratio threshold?`,
        a: R`Flag when $\eta(x) > c_{\text{FP}}/(c_{\text{FP}}+c_{\text{FN}}) = 1/51 \approx 0.0196$. Equivalently $f_1/f_0 > c_{\text{FP}}\pi_0/(c_{\text{FN}}\pi_1) = 0.98/(50\cdot 0.02) = 0.98$: flag whenever the evidence even slightly favours disease.` },
      { q: R`In the plug-in bound proof, why is $|\eta(x) - 1/2| \le |\hat\eta(x) - \eta(x)|$ on the disagreement set?`,
        a: R`Disagreement means $\hat\eta(x)$ and $\eta(x)$ are on opposite sides of $1/2$ (one is $> 1/2$, the other $\le 1/2$). Then $1/2$ lies in the closed interval between them, so the distance from $\eta(x)$ to $1/2$ cannot exceed the distance from $\eta(x)$ to $\hat\eta(x)$.` },
      { q: R`Two Gaussian classes have common covariance and equal priors. The Mahalanobis distance between the means is $\Delta = 3$. What is the Bayes error, and how does it change if you add a feature that is independent of everything else?`,
        a: R`$R^* = \Phi(-3/2) \approx 0.067$. An independent feature has the same distribution in both classes, contributes nothing to $\Delta$, and leaves $R^*$ unchanged. (It can still hurt a <em>learned</em> classifier through extra variance.) Adding features can never increase the Bayes error, since $h^*$ could ignore them.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'lda-qda',
    domain: 'supervised',
    order: 40,
    title: 'Linear & Quadratic Discriminant Analysis',
    subtitle: 'Gaussian class models, the linear boundary they imply, and Fisher’s projection view',
    level: 2,
    sources: { aos: '22.3', other: 'ESL §4.3; Bishop §4.1.4–4.1.6, §4.2.2; Fisher (1936); Friedman (1989)' },
    prereqs: ['bayes-classifier', 'multivariate'],
    related: ['naive-bayes', 'logistic-regression', 'pca', 'classifier-evaluation', 'gaussian-mixtures'],
    tags: ['generative', 'Gaussian', 'discriminant', 'Fisher', 'reduced rank', 'Mahalanobis'],
    summary: R`Model each class as a multivariate normal, estimate the means and covariances, and apply the <a href="#/t/bayes-classifier">Bayes rule</a>. With class-specific covariances the decision boundaries are quadrics (QDA). With a shared covariance the quadratic terms cancel and the boundaries are hyperplanes (LDA) — the log-odds is exactly linear, the same functional form as logistic regression but fitted through the joint likelihood. LDA has a second, distribution-free life: it is the projection that maximizes between-class separation relative to within-class spread (Fisher's criterion), and with $K$ classes the relevant information lives in a subspace of dimension at most $K-1$, which yields optimal low-dimensional views of the data.`,
    survey: {
      task: 'Multiclass classification; supervised dimension reduction (discriminant coordinates)',
      assumptions: R`$X\mid Y=k\sim\N(\mu_k,\Sigma_k)$; LDA additionally $\Sigma_k\equiv\Sigma$`,
      output: R`Discriminant functions $\delta_k(x)$, posterior probabilities, up to $K-1$ discriminant directions`,
      strengths: 'Closed form, stable, fast; LDA excellent with small $n$; interpretable low-dimensional views',
      weaknesses: R`Sensitive to outliers and heavy tails; QDA needs $O(Kp^2)$ parameters; singular $\hat\Sigma$ when $p > n$`,
      complexity: R`$O(np^2 + p^3)$ to fit; $O(Kp)$ (LDA) or $O(Kp^2)$ (QDA) per prediction`,
    },
    sections: [
      {
        title: 'Gaussian class models and the quadratic discriminant',
        html: R`
<p>The generative route to a classifier: posit $X\mid Y=k\sim\N(\mu_k,\Sigma_k)$ with priors $\pi_k$, $k=1,\dots,K$, and classify by $\argmax_k \pi_k f_k(x)$. Taking logs and dropping the constant $-\frac p2\log 2\pi$ common to all classes, the Bayes rule maximizes the <strong>quadratic discriminant function</strong>
$$\delta_k(x) = \log\pi_k - \tfrac12\log|\Sigma_k| - \tfrac12(x-\mu_k)^\top\Sigma_k^{-1}(x-\mu_k).$$
The boundary between classes $k$ and $l$, $\{x:\delta_k(x) = \delta_l(x)\}$, is a quadric surface: the term $-\frac12 x^\top(\Sigma_k^{-1} - \Sigma_l^{-1})x$ survives. Depending on the covariances it can be an ellipse, a hyperbola, a parabola, or a pair of lines. In one dimension with unequal variances the class-1 region is an interval or the complement of one — try $s\ne1$ in the <a href="#/t/bayes-classifier">Bayes classifier</a> lab.</p>
<p>Read $\delta_k$ as a prior-adjusted, volume-penalized Mahalanobis distance: assign $x$ to the class whose centre is nearest in that class's own metric, with a bonus for common classes and a penalty $\frac12\log|\Sigma_k|$ for diffuse classes (a spread-out density cannot be high anywhere).</p>`,
      },
      {
        title: 'Shared covariance: why the boundary becomes linear',
        html: R`
<div class="thm" data-name="LDA has linear boundaries and linear log-odds">
<p>If $\Sigma_k = \Sigma$ for all $k$, the Bayes rule maximizes the <strong>linear discriminant function</strong>
$$\delta_k(x) = x^\top\Sigma^{-1}\mu_k - \tfrac12\mu_k^\top\Sigma^{-1}\mu_k + \log\pi_k,$$
and for any two classes the log-posterior odds is affine in $x$:
$$\log\frac{\P(Y=k\mid x)}{\P(Y=l\mid x)} = \log\frac{\pi_k}{\pi_l} - \tfrac12(\mu_k+\mu_l)^\top\Sigma^{-1}(\mu_k-\mu_l) + x^\top\Sigma^{-1}(\mu_k-\mu_l).$$
The boundary between $k$ and $l$ is the hyperplane where this vanishes, with normal vector $\Sigma^{-1}(\mu_k-\mu_l)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Expand the quadratic form: $(x-\mu_k)^\top\Sigma^{-1}(x-\mu_k) = x^\top\Sigma^{-1}x - 2x^\top\Sigma^{-1}\mu_k + \mu_k^\top\Sigma^{-1}\mu_k$ (using symmetry of $\Sigma^{-1}$). With a shared $\Sigma$, both $\frac12\log|\Sigma|$ and $\frac12x^\top\Sigma^{-1}x$ are the same for every class, so they do not affect the $\argmax$ and can be dropped, leaving $\delta_k$. The log-odds is $\log(\pi_kf_k/\pi_lf_l) = \delta_k(x) - \delta_l(x)$, and
$$-\tfrac12\mu_k^\top\Sigma^{-1}\mu_k + \tfrac12\mu_l^\top\Sigma^{-1}\mu_l = -\tfrac12(\mu_k+\mu_l)^\top\Sigma^{-1}(\mu_k-\mu_l),$$
again by symmetry of $\Sigma^{-1}$ (the cross terms $\mu_k^\top\Sigma^{-1}\mu_l$ cancel).</p>
</details>
<p>Two readings of the boundary. Geometrically, with equal priors it passes through the midpoint $\frac12(\mu_k+\mu_l)$ and is <em>not</em> perpendicular to $\mu_k - \mu_l$ unless $\Sigma\propto I$; the normal is tilted by $\Sigma^{-1}$, away from directions in which the classes are noisy. Probabilistically, the theorem says that under the LDA model the posterior is exactly a <a href="#/t/logistic-regression">logistic regression</a>: $\P(Y=1\mid x) = \sigma(\beta_0 + \beta^\top x)$ with $\beta = \Sigma^{-1}(\mu_1-\mu_0)$.</p>
<div class="note" data-name="LDA vs. logistic regression: same family, different fits">
<p>Both produce linear log-odds. Logistic regression maximizes the <em>conditional</em> likelihood $\prod_i\P(y_i\mid x_i)$ and leaves $P_X$ unmodelled; LDA maximizes the <em>joint</em> likelihood $\prod_i\pi_{y_i}f_{y_i}(x_i)$, borrowing strength from the Gaussian shape of $P_X$. When the Gaussian model is right, LDA is more efficient — Efron (1975) found logistic regression to be only between one half and two thirds as efficient (asymptotically, in excess error rate) for statistically interesting parameter values. When it is wrong (skewed or discrete features, outliers far from the boundary that drag the estimated means and covariance), logistic regression is more robust. LDA also behaves well when classes are perfectly separable, where the logistic MLE does not exist.</p>
</div>`,
      },
      {
        title: 'Estimation and the price of parameters',
        html: R`
<p>Plug in the maximum-likelihood (or unbiased) estimates: $\hat\pi_k = n_k/n$, $\hat\mu_k = \frac1{n_k}\sum_{i:y_i=k}x_i$,
$$\hat\Sigma_k = \frac1{n_k-1}\sum_{i:y_i=k}(x_i-\hat\mu_k)(x_i-\hat\mu_k)^\top \;\;\text{(QDA)},\qquad \hat\Sigma = \frac1{n-K}\sum_{k=1}^K\sum_{i:y_i=k}(x_i-\hat\mu_k)(x_i-\hat\mu_k)^\top\;\;\text{(LDA, pooled)}.$$</p>
<p>The models differ sharply in what they must estimate. The LDA classifier depends on $(K-1)(p+1)$ numbers (the differences $\delta_k - \delta_K$); the QDA classifier on $(K-1)\big(p(p+3)/2 + 1\big)$. For $K = 2$, $p = 50$ that is $51$ versus $1326$. LDA's bias (wrong if covariances differ) is often a good trade for its far smaller variance, which is why it remains a strong baseline even when its assumptions are visibly false.</p>
<div class="ex" data-name="Regularized discriminant analysis (Friedman, 1989)">
<p>Interpolate between the two: $\hat\Sigma_k(\alpha) = \alpha\hat\Sigma_k + (1-\alpha)\hat\Sigma$, and, to stabilize the pooled estimate itself, shrink it toward a scalar matrix, $\hat\Sigma(\gamma) = \gamma\hat\Sigma + (1-\gamma)\hat\sigma^2 I$. Both $\alpha$ and $\gamma$ are chosen by <a href="#/t/cross-validation">cross-validation</a>. When $p > n$, $\hat\Sigma$ is singular and some shrinkage is mandatory; $\gamma = 0$ gives the nearest-centroid classifier in Euclidean distance, and its sparse version (nearest shrunken centroids) is a staple of genomics.</p>
</div>
<div class="warn" data-name="Unequal class sizes and the intercept">
<p>For two classes, regressing a coded response on $x$ by least squares gives a coefficient vector proportional to $\hat\Sigma^{-1}(\hat\mu_1-\hat\mu_0)$ — the LDA direction — but the least-squares intercept does not equal the LDA intercept unless $n_0 = n_1$. The direction is robust; the threshold depends on priors and should be set by the Bayes formula or chosen to minimize validation error under the costs that matter (<a href="#/t/classifier-evaluation">classifier evaluation</a>).</p>
</div>`,
      },
      {
        title: 'Fisher’s discriminant: the projection view',
        html: R`
<p>Fisher (1936) asked a question with no Gaussian assumption in it: find a direction $a$ such that the projected data $a^\top x$ separate the classes as much as possible, measured by between-class spread relative to within-class spread. Let $W = \sum_k\sum_{i:y_i=k}(x_i - \hat\mu_k)(x_i-\hat\mu_k)^\top$ (within-class scatter) and $B = \sum_k n_k(\hat\mu_k - \hat\mu)(\hat\mu_k - \hat\mu)^\top$ (between-class scatter), with $\hat\mu$ the grand mean. The <strong>Fisher criterion</strong> is the Rayleigh quotient
$$J(a) = \frac{a^\top B a}{a^\top W a}.$$</p>
<div class="thm" data-name="Fisher's direction for two classes">
<p>For $K=2$, with $d = \hat\mu_1 - \hat\mu_0$ and $W$ positive definite, $J$ is maximized by $a^* \propto W^{-1}d$, which is the LDA normal vector $\hat\Sigma^{-1}(\hat\mu_1 - \hat\mu_0)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For two classes, $B = \frac{n_0n_1}{n}\,d\,d^\top$ (direct computation: $\hat\mu_k - \hat\mu$ equals $\frac{n_{1-k}}{n}$ times $\pm d$). So $J(a) \propto (a^\top d)^2/(a^\top W a)$. By the Cauchy–Schwarz inequality in the inner product $\langle u,v\rangle_W = u^\top W v$,
$$(a^\top d)^2 = \big\langle a,\, W^{-1}d\big\rangle_W^2 \le (a^\top W a)\,(d^\top W^{-1}d),$$
so $J(a) \propto (a^\top d)^2/(a^\top Wa) \le d^\top W^{-1}d$, with equality iff $a \propto W^{-1}d$. Since $\hat\Sigma = W/(n-2)$, this is the LDA direction up to scale.</p>
</details>
<p>The equivalence is striking: one derivation assumes Gaussians and computes the Bayes rule; the other assumes nothing and maximizes a signal-to-noise ratio. They agree because the Gaussian log-likelihood ratio depends on $x$ only through $a^{*\top}x$. Under the Gaussian model, projected classes are $\N(a^\top\mu_k, a^\top\Sigma a)$, and with equal priors the best threshold on the projection has error $\Phi\big(-\tfrac12|a^\top(\mu_1-\mu_0)|/\sqrt{a^\top\Sigma a}\big)$: maximizing Fisher's (population) criterion <em>is</em> minimizing the error over all linear projections. The maximized value is $\Delta^2$, giving the Bayes error $\Phi(-\Delta/2)$.</p>
<p>In the lab, rotate a projection direction by hand and watch the projected class histograms overlap or separate. The naive direction $\mu_1 - \mu_0$ is optimal only when $\Sigma\propto I$; with correlated features, Fisher's direction can be far from it and much better.</p>`,
        lab: 'lda-qda-fisher',
      },
      {
        title: 'Reduced-rank LDA and discriminant coordinates',
        html: R`
<p>With $K$ classes, the LDA rule has a geometric description that reveals hidden low-dimensional structure. <strong>Sphere</strong> the data: $x^* = \Sigma^{-1/2}x$, so that each class has identity covariance. Then
$$\delta_k(x) = -\tfrac12\|x^* - \mu_k^*\|^2 + \log\pi_k + \text{(terms common to all }k).$$
LDA is nearest-centroid classification in sphered space, adjusted for priors.</p>
<div class="thm" data-kind="Proposition" data-name="LDA lives in a (K−1)-dimensional subspace">
<p>Let $\mathcal M$ be the affine span of the sphered centroids $\mu^*_1,\dots,\mu^*_K$, of dimension at most $K-1$, and $P$ the orthogonal projection onto its direction space. Then $\delta_k(x)$ depends on $x$ only through $P(x^* - \bar\mu^*)$, where $\bar\mu^*$ is any point of $\mathcal M$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $x^* - \mu_k^* = (x^* - \bar\mu^*) - (\mu_k^* - \bar\mu^*)$ and split $x^*-\bar\mu^* = P(x^*-\bar\mu^*) + (I-P)(x^*-\bar\mu^*)$. Since $\mu^*_k - \bar\mu^*$ lies in the range of $P$, Pythagoras gives $\|x^*-\mu^*_k\|^2 = \|P(x^*-\bar\mu^*) - (\mu^*_k - \bar\mu^*)\|^2 + \|(I-P)(x^*-\bar\mu^*)\|^2$. The second term is the same for every $k$ and drops out of the $\argmax$.</p>
</details>
<p>For $p = 100$ and $K = 3$, the classification happens in a plane. To go further and find the best $L$-dimensional subspace, $L < K-1$, take the leading principal components of the sphered centroids (weighted by $\pi_k$), i.e. the top eigenvectors $v$ of the sphered between-class covariance. Mapping back, $a = \Sigma^{-1/2}v$ solves the generalized eigenproblem
$$B a = \lambda W a,$$
whose eigenvectors are exactly the successive maximizers of Fisher's $J(a)$ subject to $W$-orthogonality to the previous ones. These <strong>discriminant coordinates</strong> (canonical variates) give the most informative low-dimensional plots of labelled data — a supervised counterpart of <a href="#/t/pca">PCA</a>, which ignores labels and can easily choose directions that do not separate the classes at all. Restricting LDA to the first $L$ coordinates (reduced-rank LDA) is itself a regularizer and is sometimes more accurate than full LDA.</p>`,
      },
      {
        title: 'LDA versus QDA in practice',
        html: R`
<p>The lab draws two Gaussian classes in the plane. You control how different the covariance of class 1 is from class 0 and how many training points there are. It shades the decision regions of the fitted LDA or QDA classifier and draws both fitted boundaries alongside the true Bayes boundary, with test errors estimated on a large independent sample.</p>
<p>Things to see: with equal covariances and small $n$, QDA's boundary curves for no reason and its test error exceeds LDA's (pure variance). As the covariances diverge, LDA's straight line cannot follow the curved Bayes boundary and its error plateaus above $R^*$ (pure bias), while QDA approaches $R^*$ once $n$ is large enough to estimate six covariance parameters.</p>`,
        lab: 'lda-qda-regions',
      },
      {
        title: 'Pitfalls and extensions',
        html: R`
<div class="warn" data-name="What breaks discriminant analysis">
<p><strong>Outliers and heavy tails:</strong> sample means and covariances have breakdown point $0$; one gross outlier can rotate the boundary. Robust covariance estimates (MCD) fix this. <strong>Multimodal classes:</strong> a class made of two clusters is badly summarized by one mean and covariance — use mixture discriminant analysis (a <a href="#/t/gaussian-mixtures">Gaussian mixture</a> per class). <strong>$p$ near or above $n$:</strong> $\hat\Sigma^{-1}$ amplifies noise along small-eigenvalue directions; regularize. <strong>Probability estimates:</strong> LDA's posteriors inherit the Gaussian tails and are overconfident far from the data when the model is wrong — check <a href="#/t/classifier-evaluation">calibration</a>.</p>
</div>
<p><strong>Extensions.</strong> Flexible discriminant analysis replaces the linear regressions implicit in LDA with nonparametric ones; kernel Fisher discriminant analysis applies Fisher's criterion in a <a href="#/t/kernel-methods">feature space</a>. Within the generative family, relaxing in the other direction — keeping class-specific variances but forcing $\Sigma_k$ diagonal — gives Gaussian <a href="#/t/naive-bayes">naive Bayes</a>.</p>`,
      },
    ],
    quiz: [
      { q: R`Two classes, equal priors, $\Sigma = \begin{pmatrix}1 & 0.8\\ 0.8 & 1\end{pmatrix}$, $\mu_0 = (0,0)$, $\mu_1 = (1,1)$. Is the LDA boundary perpendicular to $\mu_1-\mu_0$? Compute the Bayes error.`,
        a: R`$\Sigma^{-1}(1,1)^\top = (1,1)^\top/1.8$, which is parallel to $\mu_1-\mu_0$ because $(1,1)$ is an eigenvector of $\Sigma$; so yes, here the boundary is perpendicular to the mean difference. $\Delta^2 = (1,1)\Sigma^{-1}(1,1)^\top = 2/1.8 \approx 1.111$, $\Delta\approx 1.054$, $R^* = \Phi(-0.527)\approx 0.299$. The positive correlation along the mean-difference direction makes separation hard.` },
      { q: R`Why does the term $-\frac12\log|\Sigma_k|$ appear in QDA but not in LDA?`,
        a: R`It is part of the Gaussian log-density's normalizing constant. In QDA it differs across classes and must be kept (a diffuse class has lower density everywhere near its centre). In LDA all classes share $\Sigma$, so it is a common additive constant and does not affect the $\argmax$.` },
      { q: R`State the Fisher criterion and explain why its maximizer coincides with the LDA direction even though Fisher made no distributional assumption.`,
        a: R`$J(a) = a^\top Ba/a^\top Wa$. For two classes $B\propto dd^\top$, so $J\propto (a^\top d)^2/a^\top Wa$, maximized at $a\propto W^{-1}d$ by Cauchy–Schwarz. Under the shared-covariance Gaussian model, the log-likelihood ratio is affine in $x$ with exactly that normal vector; both criteria reduce to the same second-moment computation.` },
      { q: R`With $K = 4$ classes in $p = 30$ dimensions, how many dimensions does LDA actually use, and how would you find the best 2-D view for classification?`,
        a: R`At most $K-1 = 3$: after sphering, only the projection onto the affine span of the 4 centroids matters. The best 2-D view uses the top two generalized eigenvectors of $Ba = \lambda Wa$ (the first two discriminant coordinates).` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'naive-bayes',
    domain: 'supervised',
    order: 50,
    title: 'Naive Bayes',
    subtitle: 'A deliberately wrong independence model that is cheap, scalable, and often surprisingly accurate',
    level: 1,
    sources: { aos: '22.6', other: 'ESL §6.6.3; Bishop §8.2.2; Domingos & Pazzani (1997); Ng & Jordan (2002)' },
    prereqs: ['bayes-classifier'],
    related: ['lda-qda', 'logistic-regression', 'bayesian-inference', 'directed-graphical-models', 'classifier-evaluation'],
    tags: ['generative', 'conditional independence', 'Laplace smoothing', 'text classification', 'calibration'],
    summary: R`Naive Bayes estimates each class-conditional density as a product of one-dimensional marginals, $f_k(x) = \prod_j f_{kj}(x_j)$ — it assumes the features are independent <em>given the class</em>. The assumption is almost always false, yet the classifier is a fixture in text classification and a hard-to-beat baseline when $p$ is large relative to $n$. The reasons are instructive: the independence model trades large bias for very low variance; 0–1 loss rewards only getting the <em>side</em> of the decision boundary right, so the posterior can be badly distorted while the classification is unchanged; and with discrete features the classifier is linear in the features, the generative partner of logistic regression. The price is paid in probabilities, which are systematically overconfident.`,
    survey: {
      task: 'Classification, especially with many discrete features (text, counts, indicators)',
      assumptions: R`$X_1,\dots,X_p$ conditionally independent given $Y$; a 1-D model for each $X_j\mid Y$`,
      output: R`Class scores $\log\hat\pi_k + \sum_j\log\hat f_{kj}(x_j)$; (miscalibrated) posteriors`,
      strengths: R`One pass over the data; $O(Kp)$ parameters; works with $n \ll p$; handles missing features trivially`,
      weaknesses: 'Overconfident probabilities; double-counts correlated evidence; zero counts need smoothing',
      complexity: R`$O(np)$ to fit, $O(Kp)$ per prediction`,
    },
    sections: [
      {
        title: 'The independence assumption',
        html: R`
<p>The Bayes classifier needs $\pi_k$ and the class-conditional joint density $f_k$. With $p$ binary features, a fully general $f_k$ is a table with $2^p - 1$ free entries per class — hopeless for $p=1000$ word indicators. Naive Bayes assumes
$$X_1,\dots,X_p \text{ are mutually independent given } Y, \qquad\text{so}\qquad f_k(x) = \prod_{j=1}^p f_{kj}(x_j),$$
cutting the parameter count to $p$ per class. The classifier is
$$\hat h(x) = \argmax_k\Big\{\log\hat\pi_k + \sum_{j=1}^p\log\hat f_{kj}(x_j)\Big\},$$
always computed in log space (products of hundreds of small probabilities underflow), with posteriors recovered by a <code>logsumexp</code> normalization.</p>
<div class="note">
<p>In <a href="#/t/directed-graphical-models">graphical-model</a> terms, naive Bayes is the star-shaped DAG with $Y$ as the only parent of every $X_j$. Conditional independence given $Y$ is very different from marginal independence: the features are strongly correlated marginally (all words in a sports article are associated with each other through the topic), and naive Bayes attributes <em>all</em> of that correlation to the class.</p>
</div>
<p>Each $f_{kj}$ is estimated separately from the $n_k$ observations in class $k$ — $p$ one-dimensional problems instead of one $p$-dimensional problem. That is the entire computational story and most of the statistical one: one-dimensional density estimation does not suffer from the <a href="#/t/knn">curse of dimensionality</a>.</p>`,
      },
      {
        title: 'Three flavours, and why two of them are linear',
        html: R`
<div class="cols">
<div><h4>Gaussian</h4><p>$X_j\mid Y=k\sim\N(\mu_{kj},\sigma^2_{kj})$. This is <a href="#/t/lda-qda">QDA</a> with diagonal $\Sigma_k$; with $\sigma^2_{kj}=\sigma^2_j$ shared across classes, it is LDA with diagonal $\Sigma$ and the boundary is linear.</p></div>
<div><h4>Bernoulli</h4><p>$X_j\in\{0,1\}$, $\P(X_j=1\mid Y=k) = \theta_{kj}$. Models presence/absence of each vocabulary word; absent words contribute evidence too.</p></div>
<div><h4>Multinomial</h4><p>$x$ is a vector of word counts with total $N$; the document is $N$ draws from class-specific word probabilities $\phi_{k1},\dots,\phi_{kV}$. Only words that occur contribute.</p></div>
</div>
<div class="thm" data-kind="Proposition" data-name="Discrete naive Bayes is a linear classifier">
<p>For two classes, the Bernoulli naive Bayes log-posterior odds is
$$\log\frac{\P(Y=1\mid x)}{\P(Y=0\mid x)} = w_0 + \sum_{j=1}^p w_jx_j,\qquad w_j = \log\frac{\theta_{1j}(1-\theta_{0j})}{\theta_{0j}(1-\theta_{1j})},\quad w_0 = \log\frac{\pi_1}{\pi_0} + \sum_j\log\frac{1-\theta_{1j}}{1-\theta_{0j}}.$$
For multinomial naive Bayes, the log-odds is $\log(\pi_1/\pi_0) + \sum_v x_v\log(\phi_{1v}/\phi_{0v})$, also linear in the counts.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For a binary feature, $\log\P(X_j = x_j\mid Y=k) = x_j\log\theta_{kj} + (1-x_j)\log(1-\theta_{kj}) = \log(1-\theta_{kj}) + x_j\log\frac{\theta_{kj}}{1-\theta_{kj}}$. Subtract the class-0 expression from the class-1 expression and sum over $j$; add the prior log-odds. For the multinomial, the multinomial coefficient is the same for both classes and cancels, leaving $\sum_v x_v\log\phi_{kv}$.</p>
</details>
<p>So discrete naive Bayes produces a classifier in the same family as <a href="#/t/logistic-regression">logistic regression</a> on the same features. The two differ only in how the weights are chosen: naive Bayes sets each $w_j$ from feature $j$'s class-conditional frequencies in isolation; logistic regression fits all the weights jointly to maximize the conditional likelihood, so it can discount redundant features. They form a <strong>generative–discriminative pair</strong> (Ng &amp; Jordan, 2002): logistic regression has lower (or equal) asymptotic error, but naive Bayes approaches its own asymptote much faster — with a number of samples logarithmic in $p$, versus linear in $p$ for logistic regression. With small $n$ the "worse" model frequently wins.</p>`,
      },
      {
        title: 'Estimation and Laplace smoothing',
        html: R`
<p>The maximum-likelihood estimates are relative frequencies: $\hat\theta_{kj} = n_{kj}/n_k$, where $n_{kj}$ is the number of class-$k$ training examples with $X_j = 1$. The <strong>zero-frequency problem</strong>: if word $j$ never appears in the class-$k$ training documents, $\hat\theta_{kj} = 0$, and any test document containing word $j$ gets $\log\hat f_k(x) = -\infty$. A single unseen word vetoes the class, regardless of all other evidence. With large vocabularies this happens constantly.</p>
<div class="def" data-name="Additive (Laplace / Lidstone) smoothing">
<p>Bernoulli: $\hat\theta_{kj} = \dfrac{n_{kj} + \alpha}{n_k + 2\alpha}$. Multinomial: $\hat\phi_{kv} = \dfrac{N_{kv}+\alpha}{N_k + \alpha V}$, where $N_{kv}$ is the total count of word $v$ in class $k$, $N_k = \sum_v N_{kv}$, and $V$ is the vocabulary size. $\alpha = 1$ is Laplace smoothing; $0 < \alpha < 1$ is Lidstone smoothing.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Smoothing is a posterior mean">
<p>If $\theta_{kj}\sim\mathrm{Beta}(\alpha,\alpha)$ a priori and the $n_k$ class-$k$ observations of $X_j$ are IID Bernoulli($\theta_{kj}$), the posterior mean of $\theta_{kj}$ is $(n_{kj}+\alpha)/(n_k+2\alpha)$. Likewise, a $\mathrm{Dirichlet}(\alpha,\dots,\alpha)$ prior on $\phi_k$ gives posterior mean $(N_{kv}+\alpha)/(N_k+\alpha V)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The likelihood is $\theta^{n_{kj}}(1-\theta)^{n_k - n_{kj}}$ and the prior density is $\propto\theta^{\alpha-1}(1-\theta)^{\alpha-1}$, so the posterior is $\mathrm{Beta}(n_{kj}+\alpha,\; n_k-n_{kj}+\alpha)$, whose mean is $\frac{n_{kj}+\alpha}{n_k+2\alpha}$. The Dirichlet–multinomial case is identical with $V$ categories: the posterior is $\mathrm{Dirichlet}(N_{k1}+\alpha,\dots,N_{kV}+\alpha)$, with mean $(N_{kv}+\alpha)/(N_k+V\alpha)$. See <a href="#/t/bayesian-inference">conjugate priors</a>.</p>
</details>
<p>With $\alpha = 1$ and a uniform prior this is Laplace's <em>rule of succession</em>: having seen $s$ successes in $m$ trials, predict the next success with probability $(s+1)/(m+2)$. Smoothing is shrinkage toward $1/2$ (or toward uniform), strongest for rare classes and rare features. Its effect on <em>classification</em> is real but bounded — a veto at worst turns a decision into a coin flip or a wrong call — while its effect on <em>log loss</em> is unbounded, because without it the predicted probabilities include exact zeros. Once zero counts are rare (large $n_k$), the choice of $alpha$ barely matters for either. The lab below shows both with a 60-feature Bernoulli model.</p>`,
        lab: 'naive-bayes-smoothing',
      },
      {
        title: 'Why it classifies well with bad probabilities',
        html: R`
<p>For 0–1 loss, the classifier is correct at $x$ exactly when the estimated posterior lands on the same side of $1/2$ as the true one (<a href="#/t/bayes-classifier">plug-in theory</a>). The <em>magnitude</em> of the error in $\hat\eta$ is irrelevant once the side is right. Naive Bayes exploits this tolerance.</p>
<div class="ex" data-name="Duplicated evidence">
<p>Let $X_1$ be a binary feature with log-likelihood ratio $\ell(x_1) = \log\frac{\P(X_1 = x_1\mid Y=1)}{\P(X_1 = x_1\mid Y=0)}$, and let $X_2 = X_1$ be an exact copy. The true posterior log-odds is $\log\frac{\pi_1}{\pi_0} + \ell(x_1)$: the copy carries no new information. Naive Bayes, treating them as conditionally independent, computes $\log\frac{\pi_1}{\pi_0} + 2\ell(x_1)$.</p>
<ul>
<li>With equal priors, $\sgn(2\ell) = \sgn(\ell)$: the classification is <em>identical</em> to the Bayes classifier, yet the probability is badly wrong. If $\ell = 1.5$, the truth is $\sigma(1.5) = 0.82$ and naive Bayes reports $\sigma(3) = 0.95$.</li>
<li>With unequal priors, the decisions differ only for $x$ where $\ell(x_1)$ and the prior log-odds $c$ have opposite signs and $|\ell| < |c| < 2|\ell|$ — a narrow band.</li>
</ul>
</div>
<p>More generally, positively dependent features cause naive Bayes to count shared evidence several times. That inflates $|\log\text{odds}|$, pushing probabilities toward $0$ and $1$ (overconfidence), but if the inflation is roughly uniform across directions it changes the decision boundary little. Domingos and Pazzani (1997) showed that naive Bayes is in fact Bayes-optimal under 0–1 loss for a far larger set of distributions than those satisfying conditional independence, and gave examples with strong dependence where it remains optimal. It fails when dependence distorts the evidence <em>unevenly</em> — when one group of redundant features is overweighted relative to another group pointing the other way.</p>
<p>The lab makes this concrete with two correlated Gaussian features. When the mean difference lies along the direction of correlation, the naive boundary is close to the true one and the error rate barely moves as $\rho\to 1$, while the predicted probabilities become overconfident — the naive log-odds is $(1+\rho)$ times the true one — and the log loss rises above LDA’s. Rotate the mean difference and the boundary itself goes wrong.</p>`,
        lab: 'naive-bayes-correlation',
      },
      {
        title: 'Bias, variance, and when to use it',
        html: R`
<p>Naive Bayes is a high-bias, low-variance procedure. Each of its $Kp$ parameters is estimated from a full class's data with a one-dimensional estimator, so its variance is tiny; its bias is whatever the independence approximation costs. By the <a href="#/t/bias-variance">bias–variance</a> logic, it wins when $n$ is small relative to the complexity of the alternatives and loses when $n$ is large enough for a discriminative model to estimate the dependence. Learning curves typically cross: naive Bayes is ahead early, logistic regression later.</p>
<table>
<thead><tr><th>Situation</th><th>Naive Bayes</th></tr></thead>
<tbody>
<tr><td>Text / bag-of-words, $p \sim 10^4$–$10^6$</td><td>Strong baseline (multinomial, with smoothing and often TF–IDF or log counts)</td></tr>
<tr><td>Small $n$, many weak features</td><td>Often beats discriminative models</td></tr>
<tr><td>Need calibrated probabilities</td><td>Recalibrate (Platt scaling, isotonic regression) or use another model</td></tr>
<tr><td>Strongly redundant features with uneven redundancy</td><td>Can be poor; consider feature selection or logistic regression</td></tr>
<tr><td>Missing features at prediction time</td><td>Natural: drop the missing factors from the product (exact marginalization under the model)</td></tr>
</tbody>
</table>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Common mistakes">
<p><strong>Trusting the probabilities.</strong> Use naive Bayes scores for ranking and classification; if probabilities matter (expected-cost thresholds, risk scores), recalibrate on held-out data and check a reliability diagram (<a href="#/t/classifier-evaluation">evaluation</a>). <strong>Wrong marginal models.</strong> Gaussian naive Bayes on skewed or multimodal features misplaces the boundary; per-feature kernel density estimates or discretization help. <strong>No smoothing.</strong> Unsmoothed estimates turn unseen feature values into vetoes. <strong>Smoothing tuned for accuracy only.</strong> $\alpha$ interacts with vocabulary size; tune it by cross-validated log loss, not just error rate. <strong>Document length.</strong> In multinomial NB, long documents produce extreme log-odds simply because more terms are summed — another source of overconfidence.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`How many free parameters does a Bernoulli naive Bayes model with $K = 3$ classes and $p = 1000$ binary features have, and how many would the unrestricted class-conditional model have?`,
        a: R`Naive Bayes: $K-1 = 2$ prior parameters plus $Kp = 3000$ feature probabilities, 3002 in total. Unrestricted: $2 + 3(2^{1000}-1)$. The independence assumption replaces an exponential table by a linear number of 1-D estimates.` },
      { q: R`A spam filter trained without smoothing has never seen the word "invoice" in ham. A legitimate email contains "invoice" and twenty strongly ham-like words. What does unsmoothed naive Bayes predict, and how does Laplace smoothing change this?`,
        a: R`Unsmoothed: $\hat\theta_{\text{ham},\text{invoice}} = 0$, so the ham log-score is $-\infty$ and the email is classified as spam regardless of the other evidence. With smoothing the ham probability of "invoice" is small but positive, contributing a finite negative term that the twenty ham-like words can outweigh.` },
      { q: R`Explain, with the duplicated-feature example, why naive Bayes can have the Bayes error rate but a poor log loss.`,
        a: R`If $X_2 = X_1$, naive Bayes's log-odds is $c + 2\ell(x_1)$ instead of $c + \ell(x_1)$. With equal priors ($c = 0$) the sign — hence the classification — is unchanged, so its error equals the Bayes error. But its probabilities $\sigma(2\ell)$ are more extreme than the truth $\sigma(\ell)$, and log loss, a strictly proper scoring rule, penalizes any deviation from the true posterior.` },
      { q: R`Show that Gaussian naive Bayes with class-independent variances $\sigma_j^2$ yields a linear decision boundary.`,
        a: R`The log-odds is $\log\frac{\pi_1}{\pi_0} + \sum_j\big[-\frac{(x_j-\mu_{1j})^2}{2\sigma_j^2} + \frac{(x_j-\mu_{0j})^2}{2\sigma_j^2}\big]$. The $x_j^2/(2\sigma_j^2)$ terms cancel, leaving $\sum_j x_j(\mu_{1j}-\mu_{0j})/\sigma_j^2$ plus a constant — affine in $x$. It is LDA with a diagonal covariance.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'knn',
    domain: 'supervised',
    order: 60,
    title: 'k-Nearest Neighbours',
    subtitle: 'The simplest consistent classifier, its asymptotic error, and its collapse in high dimension',
    level: 2,
    sources: { aos: '22.10', other: 'Cover & Hart (1967); Stone (1977); Devroye, Györfi & Lugosi (1996) Ch. 5–6, 11; ESL §2.3, §13.3' },
    prereqs: ['bayes-classifier', 'bias-variance'],
    related: ['kernel-regression', 'cross-validation', 'vc-theory', 'decision-trees', 'kernel-density'],
    tags: ['nonparametric', 'consistency', 'Cover–Hart', 'Stone', 'curse of dimensionality', 'instance-based'],
    summary: R`To classify $x$, find the $k$ training points closest to it and take a majority vote. There is no training step and no model; the neighbourhood is the model. This rule estimates $\eta(x)$ by a local average, so it inherits the full <a href="#/t/bias-variance">bias–variance</a> trade-off with $k$ as the dial. Two classical theorems frame it. Cover and Hart (1967): with $k = 1$ and infinite data, its error is at most twice the Bayes error — half of all the information is in the nearest neighbour. Stone (1977): if $k\to\infty$ and $k/n\to 0$, it is <em>universally consistent</em>, reaching the Bayes error for every distribution. Both are asymptotic, and in high dimension the asymptotics arrive extremely slowly: neighbourhoods stop being local.`,
    survey: {
      task: 'Classification and regression by local averaging',
      assumptions: R`A meaningful metric on $\mathcal X$; features on comparable scales; smooth $\eta$ for good rates`,
      output: R`Vote fractions $\hat\eta_k(x)$; class prediction; decision regions that are unions of Voronoi-cell pieces`,
      strengths: 'No training; universally consistent; adapts to any boundary shape; multiclass for free',
      weaknesses: 'Curse of dimensionality; sensitive to scaling and irrelevant features; slow and memory-hungry at prediction',
      complexity: R`$O(np)$ per query by brute force; $O(\log n)$ with KD-trees in low $d$; approximate NN in high $d$`,
    },
    sections: [
      {
        title: 'The rule',
        html: R`
<p>Given training data $(X_1,Y_1),\dots,(X_n,Y_n)$ and a metric (usually Euclidean after standardizing each feature), let $X_{(1)}(x),\dots,X_{(n)}(x)$ be the training inputs ordered by distance to $x$, with labels $Y_{(1)}(x),\dots$. The $k$-NN estimates are
$$\hat\eta_k(x) = \frac1k\sum_{i=1}^k Y_{(i)}(x)\;\;\text{(classification, } Y\in\{0,1\}),\qquad \hat f_k(x) = \frac1k\sum_{i=1}^k Y_{(i)}(x)\;\;\text{(regression)},$$
and the classifier is the plug-in rule $\hat h_k(x) = \ind\{\hat\eta_k(x) > 1/2\}$ (for $K$ classes, the plurality vote). Distance ties are broken at random or by index; odd $k$ avoids vote ties for two classes.</p>
<p>With $k=1$ the decision regions are unions of <strong>Voronoi cells</strong> of the training points — the boundary is piecewise linear and passes between every pair of adjacent points with different labels, so it fits the training data perfectly (training error $0$). As $k$ grows, the vote averages over a larger region and the boundary smooths out. For regression, $k$-NN is a linear smoother with $\tr S = n/k$ (each training point receives weight $1/k$ in its own fit), so its effective degrees of freedom are $n/k$: $k$ is a direct complexity parameter.</p>
<div class="note">
<p>$k$-NN is the discriminative cousin of kernel density estimation: $\hat\eta_k(x)$ is a ratio of two $k$-NN density estimates, and $k$-NN regression is <a href="#/t/kernel-regression">Nadaraya–Watson</a> with a uniform kernel whose bandwidth adapts to the local density of data — wide where data are sparse, narrow where they are dense.</p>
</div>`,
      },
      {
        title: 'k as a complexity dial',
        html: R`
<p>Below, each class is a mixture of Gaussian blobs in the plane, so the true $\eta$ and the Bayes boundary are known. The shading is the $k$-NN vote fraction; click on the plot to add training points of the selected class and watch the regions reorganize. Test error is estimated on a large independent sample, and the Bayes error is computed from the true $\eta$.</p>
<p>With $k = 1$, the regions are fragmented by individual noisy points (high variance). With $k$ near $n$, the classifier predicts the majority class nearly everywhere (high bias). Somewhere between, the boundary tracks the Bayes boundary. A single mislabelled point added by clicking flips a whole Voronoi cell when $k = 1$ and does nothing when $k = 15$.</p>`,
        lab: 'knn-regions',
      },
      {
        title: 'The asymptotic 1-NN error: Cover & Hart',
        html: R`
<div class="thm" data-name="Cover & Hart (1967)">
<p>Let $X$ take values in $\R^d$ and assume $\eta$ is continuous $P_X$-almost everywhere. Then the expected risk of the 1-NN rule converges:
$$\lim_{n\to\infty}\E\,R(\hat h_{1,n}) = R_{1\text{NN}} := \E\big[2\eta(X)(1-\eta(X))\big],$$
and
$$R^* \;\le\; R_{1\text{NN}} \;\le\; 2R^*(1-R^*) \;\le\; 2R^*.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Limit.</em> First, the nearest neighbour converges to the query: if $x$ is in the support of $P_X$, then $\P(\|X_{(1)}(x) - x\| > \epsilon) = (1 - P_X(B(x,\epsilon)))^n \to 0$ since $P_X(B(x,\epsilon)) > 0$, and $P_X(\text{support}) = 1$. Conditional on $X = x$ and on the neighbour $X' = X_{(1)}(x)$, the test label $Y\sim\mathrm{Bern}(\eta(x))$ and the neighbour's label $Y'\sim\mathrm{Bern}(\eta(X'))$ are independent, and the 1-NN rule errs iff $Y\ne Y'$, with probability
$$\eta(x)(1-\eta(X')) + (1-\eta(x))\eta(X').$$
As $X'\to x$ and $\eta$ is continuous at $x$, this tends to $2\eta(x)(1-\eta(x))$. Dominated convergence (everything is bounded by 1) gives the limit.</p>
<p><em>Bounds.</em> Let $r(x) = \min(\eta(x),1-\eta(x))\in[0,1/2]$, so $R^* = \E r(X)$ and $2\eta(1-\eta) = 2r(1-r)$. Lower: $2r(1-r)\ge r$ because $1 - r\ge 1/2$. Upper: by Jensen ($\E r^2\ge(\E r)^2$),
$$R_{1\text{NN}} = 2\E r - 2\E r^2 \le 2R^* - 2(R^*)^2 = 2R^*(1-R^*).$$</p>
</details>
<p>The continuity assumption can be removed — the limit holds for every distribution on $\R^d$ (Devroye, Györfi &amp; Lugosi, 1996, Ch. 5). For $K$ classes, the bound becomes $R^*\le R_{1\text{NN}}\le R^*\big(2 - \tfrac{K}{K-1}R^*\big)$.</p>
<div class="note" data-name="Reading the bound">
<p>The 1-NN rule errs when the test label and the neighbour's label disagree — two independent draws from (nearly) the same Bernoulli. When the problem is nearly deterministic ($R^*$ small) the rule is nearly optimal: $R_{1\text{NN}}\approx 2R^*$. When it is noisy, the bound is loose. Note also what the theorem does <em>not</em> say: the 1-NN rule is not consistent (its limit exceeds $R^*$ whenever $0 < \eta < 1$ on a set of positive probability), because a single neighbour's label never stops being noisy. That is the motivation for letting $k$ grow.</p>
</div>`,
      },
      {
        title: 'Universal consistency: Stone’s theorem',
        html: R`
<div class="thm" data-name="Stone (1977)">
<p>If $k = k_n\to\infty$ and $k_n/n\to 0$, then for every distribution of $(X,Y)$ on $\R^d\times\{0,1\}$ (with distance ties broken at random), $\E R(\hat h_{k,n})\to R^*$. The $k$-NN rule is <strong>universally consistent</strong>. (Devroye, Györfi, Krzyżak &amp; Lugosi (1994) showed strong universal consistency, $R(\hat h_{k,n})\to R^*$ a.s., under the additional condition $k_n/\log n\to\infty$.)</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>By the <a href="#/t/bayes-classifier">plug-in bound</a>, $\E R(\hat h_k) - R^*\le 2\E|\hat\eta_k(X) - \eta(X)|$, so it suffices to show $\hat\eta_k\to\eta$ in $L^1$. Introduce the "noise-free" local average $\tilde\eta_k(x) = \frac1k\sum_{i=1}^k\eta(X_{(i)}(x))$ and split
$$\E|\hat\eta_k - \eta| \le \E|\hat\eta_k - \tilde\eta_k| + \E|\tilde\eta_k - \eta|.$$
<em>Variance.</em> Conditionally on all the inputs, the neighbours' labels are independent Bernoullis with means $\eta(X_{(i)})$, so $\E[(\hat\eta_k - \tilde\eta_k)^2\mid X, X_1,\dots,X_n]\le \frac1{4k}$. By Jensen, $\E|\hat\eta_k - \tilde\eta_k|\le 1/(2\sqrt k)\to 0$ because $k\to\infty$.</p>
<p><em>Bias.</em> Approximate $\eta$ in $L^1(P_X)$ by a continuous, compactly supported $\eta_\epsilon$ with $\E|\eta - \eta_\epsilon|(X) < \epsilon$. Then $|\eta(X_{(i)}) - \eta(X)|\le|\eta - \eta_\epsilon|(X_{(i)}) + |\eta_\epsilon(X_{(i)}) - \eta_\epsilon(X)| + |\eta_\epsilon - \eta|(X)$. The middle term vanishes in the limit because $k/n\to 0$ forces the $k$-th neighbour distance $\|X_{(k)}(X) - X\|\to 0$ in probability, and $\eta_\epsilon$ is uniformly continuous. The last term has expectation $< \epsilon$. The first term is handled by <em>Stone's lemma</em>, a geometric fact about $\R^d$: any point can be among the $k$ nearest neighbours of at most $k\gamma_d$ of the other points, where $\gamma_d$ depends only on $d$ (it is a cone-covering number). Hence $\frac1k\sum_{i\le k}\E|g(X_{(i)}(X))|\le\gamma_d\,\E|g(X)|$ for any $g$, giving a bound $\gamma_d\epsilon$. Let $\epsilon\to 0$.</p>
</details>
<p>Both conditions are necessary in general. Fixed $k$ leaves the variance term at order $1/\sqrt k$ — the rule converges to an asymptotic risk $R_k$ with $R_k - R^* = O(1/\sqrt k)$ — and $k/n\not\to 0$ means the neighbourhood never shrinks, leaving bias. A rate needs smoothness: if $\eta$ is Lipschitz, the calculation in <a href="#/t/bias-variance">bias–variance</a> gives $\E|\hat\eta_k - \eta|\lesssim k^{-1/2} + (k/n)^{1/d}$, optimized by $k\asymp n^{2/(d+2)}$. Without assumptions there is no rate at all: for any rule and any rate, some distribution makes convergence slower (Devroye, 1982) — a <a href="#/t/vc-theory">no-free-lunch</a> phenomenon.</p>`,
      },
      {
        title: 'The curse of dimensionality',
        html: R`
<p>Every argument above assumed neighbours are near. In high dimension, they are not.</p>
<div class="ex" data-name="Neighbourhoods are not local">
<p>Data uniform on $[0,1]^d$. A sub-cube capturing a fraction $r$ of the data has edge length $r^{1/d}$. To capture $1\%$ of the data in $d = 10$, the edge must be $0.01^{1/10}\approx 0.63$ — the "local" neighbourhood spans 63% of the range of every coordinate. In $d = 100$, it is $0.955$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Distance concentration">
<p>Let $X, X'$ be independent with IID coordinates having variance $\sigma^2$ and finite fourth moment. Then
$$\frac{\|X - X'\|^2}{2d\sigma^2}\pto 1\quad\text{as } d\to\infty.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\|X - X'\|^2 = \sum_{j=1}^d(X_j - X'_j)^2$ is a sum of $d$ IID terms with mean $2\sigma^2$ and finite variance $v$ (finite fourth moment). Its mean is $2d\sigma^2$ and its standard deviation is $\sqrt{dv}$, so the ratio has mean $1$ and standard deviation $\sqrt{v}/(2\sigma^2\sqrt d)\to 0$; apply Chebyshev.</p>
</details>
<p>Relative to their typical size, all pairwise distances become nearly equal; the "nearest" neighbour is barely nearer than the farthest (Beyer et al., 1999). If only a few coordinates carry signal and the rest are noise, the noise dominates the distance and the neighbours are essentially random with respect to the signal. The lab below adds pure-noise dimensions to a 2-D problem and plots test error against $k$: with no noise dimensions, the optimal $k$-NN error approaches the Bayes rate; with twenty, it degrades badly at every $k$. Real data escape the curse only to the extent that they concentrate near low-dimensional structure and the metric respects it.</p>`,
        lab: 'knn-error-curve',
      },
      {
        title: 'Practice: metrics, scaling, and computation',
        html: R`
<ul>
<li><strong>Standardize</strong> features (or use a principled metric). Euclidean distance on raw features lets the feature with the largest units decide everything.</li>
<li><strong>Choose $k$ by cross-validation</strong>, typically on a log grid. The leave-one-out error of $k$-NN is cheap: exclude each point from its own neighbour list.</li>
<li><strong>Metric learning</strong> (e.g. NCA, LMNN, or a Mahalanobis metric from <a href="#/t/lda-qda">discriminant analysis</a>) stretches discriminative directions and shrinks noise directions — a direct attack on the curse.</li>
<li><strong>Weighted votes</strong> (inverse distance, kernel weights) reduce sensitivity to $k$ and connect to <a href="#/t/kernel-regression">kernel smoothing</a>.</li>
<li><strong>Computation.</strong> Brute force costs $O(np)$ per query. KD-trees and ball trees give roughly logarithmic queries in low dimension but degrade toward brute force beyond $d\approx 20$; approximate nearest-neighbour methods (locality-sensitive hashing, graph-based indices such as HNSW) trade exactness for speed at scale.</li>
</ul>
<div class="warn" data-name="Pitfalls">
<p>Training error of $k$-NN computed with each point counted as its own neighbour is meaningless at small $k$ (it is $0$ for $k=1$). Class imbalance biases votes toward the majority class in sparse regions; adjust the threshold or reweight. Irrelevant or redundant features are not ignored by the metric; select or reweight them. Finally, a low $k$-NN error does not mean the features are "simple" — it is a statement about local smoothness of $\eta$ under the chosen metric.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`The Bayes error of a problem is $0.10$. Between which values must the asymptotic 1-NN error lie? What if $R^* = 0.4$?`,
        a: R`$0.10\le R_{1\text{NN}}\le 2(0.1)(0.9) = 0.18$. For $R^* = 0.4$: $0.40\le R_{1\text{NN}}\le 2(0.4)(0.6) = 0.48$. The upper bound is close to $2R^*$ for small $R^*$ and becomes much tighter than $2R^*$ (which would be $0.8$) for noisy problems.` },
      { q: R`Why is the 1-NN rule not consistent even though its nearest neighbour converges to the query point?`,
        a: R`Convergence of $X_{(1)}$ to $x$ makes the neighbour's label distribution converge to $\mathrm{Bern}(\eta(x))$, but the label itself remains a single random draw. The rule errs whenever two independent $\mathrm{Bern}(\eta(x))$ draws differ, with probability $2\eta(1-\eta) > \min(\eta,1-\eta)$ for $0 < \eta < 1$, $\eta\ne1/2$. Averaging $k\to\infty$ labels is needed to kill this variance.` },
      { q: R`In Stone's theorem, which condition controls variance and which controls bias? Give a choice of $k_n$ satisfying both.`,
        a: R`$k\to\infty$ controls variance (the label noise in the vote is of order $1/\sqrt k$); $k/n\to 0$ controls bias (the neighbourhood must shrink so that $\eta$ at the neighbours approaches $\eta(x)$). For example $k_n = \lceil\sqrt n\rceil$, or $n^{2/(d+2)}$ for the Lipschitz-optimal rate.` },
      { q: R`You add 50 standard-normal noise features to a data set where two features carry all the signal (each with variance 1). Roughly what fraction of the squared Euclidean distance between two random points is due to the signal features?`,
        a: R`Each coordinate difference has expected square $2$, so the signal contributes about $2\cdot 2 = 4$ of a total $2\cdot 52 = 104$, under 4%. The neighbour ordering is dominated by noise, which is why $k$-NN degrades; standardization does not help, feature selection or metric learning does.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'vc-theory',
    domain: 'supervised',
    order: 70,
    title: 'Generalization: VC Dimension & Uniform Convergence',
    subtitle: 'When does a small training error certify a small test error, and how many samples does that take',
    level: 3,
    sources: { aos: '22.8', other: 'Shalev-Shwartz & Ben-David Ch. 2–6, 26–28; Devroye, Györfi & Lugosi (1996) Ch. 12–13; Vapnik (1998)' },
    prereqs: ['learning-framework', 'inequalities'],
    related: ['knn', 'svm', 'model-selection', 'deep-learning-theory', 'empirical-cdf'],
    tags: ['PAC', 'VC dimension', 'shattering', 'Sauer–Shelah', 'Rademacher', 'no free lunch'],
    summary: R`ERM chooses the hypothesis with the smallest training error, so the training error of the chosen hypothesis is a biased estimate of its risk. The bias is controlled if the empirical risk converges to the true risk <em>uniformly</em> over the hypothesis class. For finite classes, Hoeffding's inequality and a union bound give deviations of order $\sqrt{\log|\mathcal H|/n}$. For infinite classes, the right measure of size is combinatorial: the number of distinct labelings the class can produce on $n$ points. The Sauer–Shelah lemma shows this <em>growth function</em> is either $2^n$ for all $n$ or polynomial of degree $d$ — the VC dimension — and the VC inequality turns that into deviations of order $\sqrt{d\log n/n}$. The punchline, the fundamental theorem of statistical learning: a binary class is learnable if and only if its VC dimension is finite, with sample complexity $\Theta\big((d + \log(1/\delta))/\epsilon^2\big)$.`,
    survey: {
      task: 'Distribution-free guarantees on $R(\hat f) - \hat R_n(\hat f)$ and on the excess risk of ERM',
      assumptions: R`IID data; bounded loss; class $\mathcal H$ fixed before seeing data; no assumption on $P$`,
      output: R`High-probability bounds $\sup_{\mathcal H}|\hat R_n - R|\lesssim\sqrt{(d\log n + \log(1/\delta))/n}$; sample complexities`,
      strengths: 'Holds for every distribution; characterizes learnability exactly; explains overfitting quantitatively',
      weaknesses: 'Worst-case bounds are loose; vacuous for heavily overparameterized models that generalize well in practice',
      complexity: 'Theoretical; computing the VC dimension of a class can itself be hard',
    },
    sections: [
      {
        title: 'Why uniform convergence is the right question',
        html: R`
<p>From <a href="#/t/learning-framework">the learning framework</a>, ERM's estimation error obeys $R(\hat f) - R(f_{\mathcal H})\le 2\sup_{f\in\mathcal H}|\hat R_n(f) - R(f)|$, and the same supremum bounds the gap between training and test error of the selected predictor. The law of large numbers applies to each $f$ separately, but ERM chooses $f$ <em>after</em> seeing the data, and among many hypotheses some will have training errors that are low by luck. The supremum is the price of that search.</p>
<div class="ex" data-name="A class where uniform convergence fails">
<p>Let $X\sim\text{Unif}(0,1)$, $Y = 1$ always, and $\mathcal H = \{\ind_A: A\subset[0,1]\text{ finite}\}$. Every $h = \ind_A\in\mathcal H$ has true risk $R(h) = \P(X\notin A) = 1$, yet the member $h = \ind_{\{X_1,\dots,X_n\}}$ has training error $0$. For every $n$, $\sup_h|\hat R_n(h) - R(h)| = 1$. The class is so rich that it can memorize any sample, so the training error says nothing.</p>
</div>
<p>When the supremum does go to zero — for every distribution, at a known rate — the class is said to satisfy <strong>uniform convergence</strong>, and ERM over it is a sound learning algorithm. The same question, for the class of half-lines $\{\ind\{x\le t\}\}$, is the <a href="#/t/empirical-cdf">Glivenko–Cantelli theorem</a>; VC theory is its generalization to arbitrary classes of sets.</p>`,
      },
      {
        title: 'Finite classes: Hoeffding plus a union bound',
        html: R`
<div class="thm" data-name="Finite-class generalization bound">
<p>Let $\mathcal H$ be finite and the loss take values in $[0,1]$. For every $\delta\in(0,1)$, with probability at least $1-\delta$,
$$\sup_{h\in\mathcal H}\big|\hat R_n(h) - R(h)\big|\le\sqrt{\frac{\log(2|\mathcal H|/\delta)}{2n}},$$
and consequently ERM satisfies $R(\hat h)\le R(h_{\mathcal H}) + 2\sqrt{\log(2|\mathcal H|/\delta)/(2n)}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For a fixed $h$, $\hat R_n(h)$ is an average of $n$ IID $[0,1]$-valued variables with mean $R(h)$, so by <a href="#/t/inequalities">Hoeffding's inequality</a> $\P(|\hat R_n(h) - R(h)| > \epsilon)\le 2e^{-2n\epsilon^2}$. The event that the supremum exceeds $\epsilon$ is the union over $h$ of these events, so by the union bound its probability is at most $2|\mathcal H|e^{-2n\epsilon^2}$. Set this equal to $\delta$ and solve for $\epsilon$. The ERM statement follows from the estimation-error lemma.</p>
</details>
<p>The dependence on the class is only through $\log|\mathcal H|$ — the number of bits needed to name a hypothesis. A class of $10^6$ hypotheses costs only $\log 10^6\approx 14$ "effective samples" in the numerator. This is the first appearance of the idea that the complexity of a class is about how many <em>distinguishable</em> hypotheses it contains, not how many parameters it has.</p>
<div class="thm" data-kind="Proposition" data-name="Realizable case: faster rate">
<p>Suppose some $h\in\mathcal H$ has $R(h) = 0$. Then with probability at least $1-\delta$, every $h\in\mathcal H$ with $\hat R_n(h) = 0$ has $R(h)\le\dfrac{\log|\mathcal H| + \log(1/\delta)}{n}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Fix $\epsilon$. A "bad" hypothesis with $R(h) > \epsilon$ survives $n$ independent tests with probability $(1 - R(h))^n < (1-\epsilon)^n\le e^{-n\epsilon}$. By the union bound over at most $|\mathcal H|$ bad hypotheses, the probability that any of them has zero training error is at most $|\mathcal H|e^{-n\epsilon}$. Setting this to $\delta$ gives the claim.</p>
</details>
<p>The realizable rate is $1/n$, not $1/\sqrt n$: when there is no noise, a single mistake on the training set suffices to eliminate a bad hypothesis, and the probability of no mistake decays exponentially. With noise, distinguishing risk $\epsilon$ from $0$ needs a variance-limited $1/\epsilon^2$ samples.</p>`,
      },
      {
        title: 'Shattering and the VC dimension',
        html: R`
<p>For infinite classes, $\log|\mathcal H| = \infty$ and the union bound is useless. But on a fixed sample $x_1,\dots,x_n$, a class of binary classifiers can only produce finitely many labelings, and those are all that the training error can see.</p>
<div class="def" data-name="Growth function, shattering, VC dimension">
<p>The restriction of $\mathcal H$ to points $x_1,\dots,x_n$ is $\mathcal H_{x_{1:n}} = \{(h(x_1),\dots,h(x_n)): h\in\mathcal H\}\subseteq\{0,1\}^n$. The <em>growth function</em> is $\Pi_{\mathcal H}(n) = \max_{x_1,\dots,x_n}|\mathcal H_{x_{1:n}}|$. A set of $n$ points is <em>shattered</em> if $|\mathcal H_{x_{1:n}}| = 2^n$ — the class realizes every labeling of it. The <em>VC dimension</em> $\operatorname{VC}(\mathcal H)$ is the largest $n$ such that some set of $n$ points is shattered (or $\infty$ if arbitrarily large sets are).</p>
</div>
<p>Note the quantifiers: VC dimension $\ge d$ needs <em>one</em> shattered set of size $d$; VC dimension $< d$ needs that <em>no</em> set of size $d$ is shattered.</p>
<table>
<thead><tr><th>Class</th><th>VC dimension</th><th>Why</th></tr></thead>
<tbody>
<tr><td>Thresholds on $\R$: $\ind\{x\ge t\}$</td><td>1</td><td>For $x_1 < x_2$, the labeling $(1,0)$ is impossible</td></tr>
<tr><td>Intervals on $\R$: $\ind\{a\le x\le b\}$</td><td>2</td><td>For $x_1 < x_2 < x_3$, $(1,0,1)$ is impossible</td></tr>
<tr><td>Half-planes in $\R^2$</td><td>3</td><td>Special case of half-spaces below</td></tr>
<tr><td>Affine half-spaces in $\R^d$</td><td>$d+1$</td><td>Theorem below</td></tr>
<tr><td>Axis-aligned rectangles in $\R^2$</td><td>4</td><td>Among 5 points, one is not extreme in any axis direction; label it 0, the rest 1</td></tr>
<tr><td>Convex polygons in $\R^2$</td><td>$\infty$</td><td>Any subset of points on a circle is cut out by its convex hull</td></tr>
<tr><td>$\{\ind\{\sin(\omega x) > 0\}:\omega > 0\}$</td><td>$\infty$</td><td>One parameter, infinite VC: dimension is not parameter count</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Theorem" data-name="Half-spaces in R^d have VC dimension d + 1">
<p>The class $\{x\mapsto\ind\{w^\top x + b > 0\}: w\in\R^d, b\in\R\}$ has VC dimension $d+1$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Lower bound.</em> Take $x_0 = 0$ and $x_j = e_j$, $j=1,\dots,d$. Given target signs $s_0,\dots,s_d\in\{-1,+1\}$, set $b = s_0/2$ and $w_j = 2s_j - b$. Then $w^\top x_0 + b = b$ has sign $s_0$ and $w^\top x_j + b = 2s_j$ has sign $s_j$. All $2^{d+1}$ labelings are realized.</p>
<p><em>Upper bound.</em> By <strong>Radon's theorem</strong>, any $d+2$ points in $\R^d$ can be partitioned into two sets $A, B$ whose convex hulls intersect. (Proof: the $d+2$ vectors $(x_i, 1)\in\R^{d+1}$ are linearly dependent, $\sum_i\lambda_i(x_i,1) = 0$ with $\lambda\ne 0$; let $A$ be the indices with $\lambda_i > 0$; then $\sum_A\lambda_i = \sum_{B}(-\lambda_i) =: \Lambda > 0$ and $\sum_A\frac{\lambda_i}{\Lambda}x_i = \sum_B\frac{-\lambda_i}{\Lambda}x_i$ is a common point of both hulls.) Label $A$ with 1 and $B$ with 0. A half-space containing $A$ contains its convex hull, and the complementary (closed) half-space containing $B$ contains its hull; the common point would lie in both, a contradiction. So no $d+2$ points are shattered.</p>
</details>
<p>In the lab you place points and the class is tested against all $2^m$ labelings. Try to shatter four points with half-planes (impossible: either one point is inside the triangle of the others, or the four are in convex position and the "diagonal" labeling fails), and five with rectangles.</p>`,
        lab: 'vc-theory-shatter',
      },
      {
        title: 'The Sauer–Shelah lemma',
        html: R`
<p>A finite VC dimension $d$ says the class cannot realize all $2^n$ labelings once $n > d$. The remarkable fact is that it then realizes only <em>polynomially</em> many.</p>
<div class="thm" data-name="Sauer–Shelah lemma (Sauer, 1972; Shelah, 1972; Vapnik & Chervonenkis, 1971)">
<p>If $\operatorname{VC}(\mathcal H) = d < \infty$, then for all $n$,
$$\Pi_{\mathcal H}(n)\le\Phi_d(n) := \sum_{i=0}^d\binom ni,\qquad\text{and for } n\ge d,\quad \Phi_d(n)\le\Big(\frac{en}{d}\Big)^d.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>We prove the stronger statement: for any finite set $S$ of $n$ points and any family $C\subseteq\{0,1\}^S$, $|C|\le\#\{T\subseteq S: C\text{ shatters }T\}$. Since $C$ shatters no set larger than $d$, the right side is at most $\Phi_d(n)$.</p>
<p>Induct on $n$; $n=1$ is immediate. Pick a point $x\in S$ and let $S' = S\setminus\{x\}$. Let $C_1 = \{c|_{S'}: c\in C\}$ (restrictions) and $C_2 = \{c'\in\{0,1\}^{S'}: \text{both extensions } (c',0),(c',1)\in C\}$. Each restriction in $C_1$ comes from one or two members of $C$, and from two exactly when it lies in $C_2$, so $|C| = |C_1| + |C_2|$. By induction, $|C_1|\le\#\{T\subseteq S': C_1\text{ shatters }T\}$, and each such $T$ is also shattered by $C$. Also $|C_2|\le\#\{T\subseteq S': C_2\text{ shatters }T\}$, and if $C_2$ shatters $T$ then $C$ shatters $T\cup\{x\}$ (both values at $x$ are available for each pattern on $T$). The two collections — sets not containing $x$, and sets of the form $T\cup\{x\}$ — are disjoint, so $|C|\le\#\{T\subseteq S: C\text{ shatters }T\}$.</p>
<p>For the numerical bound with $n\ge d$: $\sum_{i\le d}\binom ni\le\big(\frac nd\big)^d\sum_{i\le d}\binom ni\big(\frac dn\big)^i\le\big(\frac nd\big)^d\big(1+\frac dn\big)^n\le\big(\frac nd\big)^de^d$.</p>
</details>
<p>The growth function therefore has a sharp dichotomy: $\Pi_{\mathcal H}(n) = 2^n$ for all $n$ (infinite VC dimension), or $\Pi_{\mathcal H}(n) = O(n^d)$. There is nothing in between. Effectively, a class with VC dimension $d$ behaves on any sample like a finite class with $\log|\mathcal H|\approx d\log(en/d)$.</p>`,
      },
      {
        title: 'The VC inequality',
        html: R`
<div class="thm" data-name="VC inequality (Vapnik & Chervonenkis, 1971)">
<p>For any class $\mathcal H$ of binary classifiers, any distribution, and $\epsilon > 0$,
$$\P\Big(\sup_{h\in\mathcal H}\big|\hat R_n(h) - R(h)\big| > \epsilon\Big)\le 8\,\Pi_{\mathcal H}(n)\,e^{-n\epsilon^2/32}.$$
Hence, with probability at least $1-\delta$, if $\operatorname{VC}(\mathcal H) = d$ and $n\ge d$,
$$\sup_{h\in\mathcal H}\big|\hat R_n(h) - R(h)\big|\le\sqrt{\frac{32}{n}\Big(d\log\frac{en}{d} + \log\frac8\delta\Big)}.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Symmetrization.</em> Introduce an independent "ghost" sample of size $n$ with empirical risk $\hat R_n'$. For $n\epsilon^2\ge 2$, Chebyshev shows that if $h$ deviates by more than $\epsilon$ on the real sample, it deviates by more than $\epsilon/2$ from its ghost-sample error with probability at least $1/2$; so $\P(\sup|\hat R_n - R| > \epsilon)\le 2\P(\sup|\hat R_n - \hat R_n'| > \epsilon/2)$. The right side involves only the $2n$ observed points — the unknown $R$ is gone.</p>
<p><em>Random signs.</em> Swapping the $i$-th real and ghost points does not change the joint distribution, so we may multiply each difference by an independent random sign $\sigma_i$. Now condition on the $2n$ points: $\mathcal H$ acts on them through at most $\Pi_{\mathcal H}(2n)$ labelings, a finite class. For each labeling, $\frac1n\sum\sigma_i(\cdot)$ is a sum of bounded independent signs, and Hoeffding plus a union bound over the labelings finishes the argument. This route yields a bound of the same form with $\Pi_{\mathcal H}(2n)$ in place of $\Pi_{\mathcal H}(n)$ and slightly different constants; the version stated is the one in AoS Thm. 22.18 and Devroye, Györfi &amp; Lugosi Thm. 12.5. Constants vary across textbooks and are not the point.</p>
</details>
<p>Plugging into the estimation-error lemma: ERM over a VC class has excess risk over $h_{\mathcal H}$ of order $\sqrt{d\log n/n}$ with high probability, <em>for every distribution</em>. A more careful argument (chaining) removes the $\log n$, giving $O(\sqrt{d/n})$, which is optimal.</p>
<div class="note" data-name="How loose?">
<p>The bounds are worst-case over all distributions and all members of the class simultaneously, and the constants are generous. For $d = 3$ and $\delta = 0.05$ the VC bound above drops below $0.1$ only once $n$ exceeds roughly $1.3\times10^5$. Their value is in the <em>shape</em>: deviation grows with $\sqrt{d}$, shrinks with $\sqrt n$, and depends on the confidence only logarithmically. The second lab compares the bounds with the actual uniform deviation for the class of thresholds, where it can be computed exactly.</p>
</div>`,
        lab: 'vc-theory-bound',
      },
      {
        title: 'Rademacher complexity: a data-dependent measure',
        html: R`
<p>VC dimension is a property of the class alone. A finer measure takes the distribution into account: how well can the class correlate with pure noise <em>on data like ours</em>?</p>
<div class="def" data-name="Rademacher complexity">
<p>For a class $\mathcal F$ of functions $z\mapsto f(z)$ and IID $Z_1,\dots,Z_n$, with independent Rademacher signs $\sigma_i$ ($\pm1$ with probability $1/2$),
$$\mathfrak R_n(\mathcal F) = \E\sup_{f\in\mathcal F}\frac1n\sum_{i=1}^n\sigma_if(Z_i).$$
The empirical version $\hat{\mathfrak R}_S(\mathcal F)$ conditions on the sample and averages only over $\sigma$.</p>
</div>
<div class="thm" data-name="Rademacher generalization bound">
<p>If every $f\in\mathcal F$ maps into $[0,1]$, then with probability at least $1-\delta$, for all $f\in\mathcal F$ simultaneously,
$$\E f(Z)\le\frac1n\sum_{i=1}^nf(Z_i) + 2\mathfrak R_n(\mathcal F) + \sqrt{\frac{\log(1/\delta)}{2n}}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\Phi(S) = \sup_{f}\big(\E f - \hat\E_Sf\big)$. Changing one $Z_i$ changes $\hat\E_Sf$ by at most $1/n$ for every $f$, hence $\Phi$ by at most $1/n$. McDiarmid's bounded-differences inequality gives $\Phi(S)\le\E\Phi(S) + \sqrt{\log(1/\delta)/(2n)}$ with probability $\ge 1-\delta$. For the expectation, introduce a ghost sample $S'$: $\E f = \E\hat\E_{S'}f$, so by Jensen ($\sup$ of an expectation $\le$ expectation of $\sup$)
$$\E\Phi(S)\le\E\sup_f\frac1n\sum_i\big(f(Z_i') - f(Z_i)\big) = \E\sup_f\frac1n\sum_i\sigma_i\big(f(Z_i') - f(Z_i)\big)\le 2\mathfrak R_n(\mathcal F),$$
where the equality holds because swapping $Z_i\leftrightarrow Z_i'$ leaves the distribution unchanged, and the last step splits the supremum of a sum.</p>
</details>
<p>For classification take $f(x,y) = \ind\{h(x)\ne y\}$. On a sample, the vectors $(f(Z_i))_i$ take at most $\Pi_{\mathcal H}(n)$ values in $\{0,1\}^n$, each of Euclidean norm at most $\sqrt n$, and <strong>Massart's lemma</strong> — for a finite $A\subset\R^n$, $\E\max_{a\in A}\frac1n\sum\sigma_ia_i\le\max_a\|a\|_2\sqrt{2\log|A|}/n$, proved by bounding $\E e^{\lambda\max}$ by a sum of moment generating functions — gives
$$\mathfrak R_n\le\sqrt{\frac{2\log\Pi_{\mathcal H}(n)}{n}}\le\sqrt{\frac{2d\log(en/d)}{n}}.$$
So Rademacher bounds recover VC bounds with better constants, and they go further: for linear predictors with $\|w\|_2\le B$ on data with $\|x\|_2\le r$, $\hat{\mathfrak R}_S\le Br/\sqrt n$ regardless of the dimension — the basis of margin bounds for <a href="#/t/svm">SVMs</a> and norm-based bounds for networks.</p>`,
      },
      {
        title: 'PAC learning, the fundamental theorem, and no free lunch',
        html: R`
<div class="def" data-name="(Agnostic) PAC learnability">
<p>$\mathcal H$ is <em>agnostic PAC learnable</em> if there exist an algorithm $A$ and a function $m_{\mathcal H}(\epsilon,\delta)$ such that for every distribution $P$ and all $\epsilon,\delta\in(0,1)$, given $n\ge m_{\mathcal H}(\epsilon,\delta)$ IID samples, $R(A(S))\le\min_{h\in\mathcal H}R(h) + \epsilon$ with probability at least $1-\delta$. In the <em>realizable</em> version $P$ is restricted to have $\min_{\mathcal H}R = 0$. (Valiant, 1984.)</p>
</div>
<div class="thm" data-name="Fundamental theorem of statistical learning">
<p>For binary classification with 0–1 loss, the following are equivalent: (i) $\mathcal H$ has the uniform convergence property; (ii) ERM is an agnostic PAC learner for $\mathcal H$; (iii) $\mathcal H$ is agnostic PAC learnable; (iv) $\mathcal H$ is PAC learnable (realizable); (v) $\operatorname{VC}(\mathcal H) < \infty$. Quantitatively, with $d = \operatorname{VC}(\mathcal H)$ and absolute constants $C_1, C_2$,
$$\text{agnostic: } C_1\frac{d + \log(1/\delta)}{\epsilon^2}\le m_{\mathcal H}(\epsilon,\delta)\le C_2\frac{d + \log(1/\delta)}{\epsilon^2};\qquad \text{realizable: } C_1\frac{d + \log(1/\delta)}{\epsilon}\le m_{\mathcal H}\le C_2\frac{d\log(1/\epsilon) + \log(1/\delta)}{\epsilon}.$$
(Shalev-Shwartz &amp; Ben-David, Thm. 6.7–6.8.)</p>
</div>
<p>The direction (v)$\Rightarrow$(i) is the VC inequality. The converse rests on a lower-bound construction that is also the cleanest form of the <strong>no-free-lunch theorem</strong>.</p>
<div class="thm" data-name="No free lunch (Shalev-Shwartz & Ben-David, Thm. 5.1)">
<p>Let $A$ be any learning algorithm for binary classification on a domain $\mathcal X$, and $n < |\mathcal X|/2$. There is a distribution $P$ on $\mathcal X\times\{0,1\}$ such that some $f:\mathcal X\to\{0,1\}$ has $R_P(f) = 0$, yet with probability at least $1/7$ over $S\sim P^n$, $R_P(A(S))\ge 1/8$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Fix $C\subseteq\mathcal X$ with $|C| = 2n$ and consider the $2^{2n}$ labelings $f$ of $C$, each with $P_f$ uniform on $C$ and labels given by $f$. Average the expected risk over a uniformly random $f$. A sample of size $n$ sees at most $n$ distinct points, so at least half of $C$ is unseen; on unseen points the random labeling is independent of everything the learner saw, so the learner is wrong there with probability $1/2$. Hence the average expected risk is at least $\frac12\cdot\frac12 = \frac14$, so some $f$ has $\E R_{P_f}(A(S))\ge 1/4$. Since $R\in[0,1]$, $\P(R\ge 1/8)\ge(1/4 - 1/8)/(1 - 1/8) = 1/7$.</p>
</details>
<p>A learner with no inductive bias — one that must handle every labeling of a domain — cannot generalize. If $\mathcal H$ shatters arbitrarily large sets, the same construction lives inside $\mathcal H$ and ERM over $\mathcal H$ fails; that is (iii)$\Rightarrow$(v). Every successful learner encodes prior knowledge: a restricted class, a penalty, or a smoothness assumption.</p>
<div class="warn" data-name="Where VC theory stops explaining">
<p>Modern networks have VC dimensions far exceeding $n$ and can fit random labels (Zhang et al., 2017), so the uniform bounds are vacuous — yet the same networks generalize on real labels. The resolution must involve the <em>algorithm</em> and the <em>data</em>, not just the class: margin and norm-based Rademacher bounds, PAC-Bayes bounds, algorithmic stability, and the implicit bias of gradient descent. See <a href="#/t/deep-learning-theory">theory of deep learning</a>. VC theory remains exactly right about what it claims: worst-case, distribution-free guarantees for ERM over a fixed class.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A finite class has $|\mathcal H| = 2^{20}$ hypotheses and the loss is 0–1. How large must $n$ be to guarantee, with probability $0.95$, that every hypothesis's training error is within $0.05$ of its risk?`,
        a: R`Need $\sqrt{\log(2|\mathcal H|/\delta)/(2n)}\le 0.05$, i.e. $n\ge\log(2\cdot 2^{20}/0.05)/(2\cdot 0.0025) = (21\log 2 + \log 20)/0.005\approx(14.56 + 3.00)/0.005\approx 3512$.` },
      { q: R`Show that intervals on the line have VC dimension exactly 2.`,
        a: R`Two points $x_1 < x_2$ are shattered: $[x_1-1, x_1-\tfrac12]$ gives $(0,0)$; tiny intervals around one point give $(1,0)$ or $(0,1)$; $[x_1,x_2]$ gives $(1,1)$. For three points $x_1 < x_2 < x_3$, the labeling $(1,0,1)$ is impossible: an interval containing $x_1$ and $x_3$ contains $x_2$.` },
      { q: R`The Sauer–Shelah lemma bounds $\Pi_{\mathcal H}(n)$ by $\sum_{i\le d}\binom ni$. Evaluate the bound for $d = 3$, $n = 10$, and compare with $2^{10}$. Is the bound attained by half-planes in $\R^2$?`,
        a: R`$1 + 10 + 45 + 120 = 176$ versus $1024$. For half-planes on 10 points in general position, the number of labelings is $2\binom{10}{2} + 2 = 92$, below 176. Sauer–Shelah is tight for some classes (e.g. all sets of size at most $d$) but not for every class of that VC dimension.` },
      { q: R`Why is the realizable sample complexity of order $1/\epsilon$ while the agnostic one is of order $1/\epsilon^2$?`,
        a: R`In the realizable case one only needs to rule out hypotheses with risk above $\epsilon$; such a hypothesis survives $n$ samples without a mistake with probability at most $(1-\epsilon)^n\approx e^{-n\epsilon}$. In the agnostic case one must estimate risks to accuracy $\epsilon$ when they are bounded away from $0$, and the standard deviation of an empirical frequency is of order $1/\sqrt n$, requiring $n\sim1/\epsilon^2$.` },
      { q: R`What does the no-free-lunch theorem say about a learner that is guaranteed to have small risk for every distribution on an infinite domain?`,
        a: R`No such learner exists with a fixed sample size: for any algorithm and any $n$, some realizable distribution forces risk at least $1/8$ with probability at least $1/7$. Guarantees require restricting the target (a finite-VC class, a smoothness or margin assumption); without such prior knowledge, learning from finite data is impossible.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'classifier-evaluation',
    domain: 'supervised',
    order: 80,
    title: 'Evaluating Classifiers: ROC, Calibration & Proper Scoring Rules',
    subtitle: 'Ranking, thresholding, and probability quality are three different questions',
    level: 2,
    sources: { aos: '22.8', other: 'Fawcett (2006); Gneiting & Raftery (2007); Hand (2009); ESL §9.2.5; Bishop §1.5' },
    prereqs: ['bayes-classifier'],
    related: ['neyman-pearson', 'rank-tests', 'logistic-regression', 'cross-validation', 'naive-bayes'],
    tags: ['ROC', 'AUC', 'precision–recall', 'calibration', 'Brier score', 'log loss', 'cost-sensitive'],
    summary: R`A probabilistic classifier produces a score $s(x)$; turning it into decisions requires a threshold. Three distinct questions follow. <em>Discrimination</em>: does the score rank positives above negatives? The ROC curve and its area answer this, and AUC is exactly the probability that a random positive outscores a random negative. <em>Decision quality</em>: at the threshold you will actually use, given the costs and the prevalence, what is the expected cost? <em>Calibration</em>: when the model says 0.8, does the event happen 80% of the time? Proper scoring rules — log loss and the Brier score — are the losses that reward reporting your true belief, and they decompose into calibration and discrimination components. Accuracy, by contrast, answers none of these questions well.`,
    survey: {
      task: 'Assess and compare scoring classifiers; choose thresholds; check probability quality',
      assumptions: R`Held-out data from the deployment distribution (or a known prevalence shift)`,
      output: R`ROC/PR curves, AUC, cost-optimal threshold, reliability diagram, Brier score and log loss`,
      strengths: 'Separates ranking from thresholding from calibration; proper scores are principled model-selection criteria',
      weaknesses: 'AUC ignores costs and calibration; PR metrics depend on prevalence; binned calibration measures are noisy',
      complexity: R`$O(m\log m)$ for ROC/AUC on $m$ test points (one sort)`,
    },
    sections: [
      {
        title: 'The confusion matrix and its many rates',
        html: R`
<p>Fix a threshold $t$ and predict positive when $s(x) > t$. On a population with prevalence $\pi = \P(Y=1)$, the four outcomes have rates</p>
<table>
<thead><tr><th></th><th>Predicted positive</th><th>Predicted negative</th></tr></thead>
<tbody>
<tr><td>$Y = 1$</td><td>True positive: $\text{TPR} = \P(s > t\mid Y=1)$ (sensitivity, recall)</td><td>False negative: $1 - \text{TPR}$</td></tr>
<tr><td>$Y = 0$</td><td>False positive: $\text{FPR} = \P(s > t\mid Y=0)$</td><td>True negative: $1-\text{FPR}$ (specificity)</td></tr>
</tbody>
</table>
<p>TPR and FPR condition on the truth and do not depend on $\pi$. Quantities that condition on the <em>prediction</em> do:
$$\text{precision} = \text{PPV} = \P(Y=1\mid s > t) = \frac{\pi\,\text{TPR}}{\pi\,\text{TPR} + (1-\pi)\,\text{FPR}},\qquad\text{accuracy} = \pi\,\text{TPR} + (1-\pi)(1-\text{FPR}).$$</p>
<div class="ex" data-name="The base-rate trap">
<p>A test with sensitivity $0.99$ and false-positive rate $0.05$ is applied where $\pi = 0.01$. Then $\text{PPV} = \frac{0.0099}{0.0099 + 0.0495}\approx 0.167$: five of six positive results are false. The test is excellent at separating the classes; the population is simply dominated by negatives. Meanwhile, the trivial rule "always negative" has accuracy $0.99$, beating the test's $0.95$ — accuracy rewards ignoring the minority class.</p>
</div>`,
      },
      {
        title: 'ROC curves',
        html: R`
<p>Sweeping the threshold from $+\infty$ to $-\infty$ traces the <strong>receiver operating characteristic</strong> $t\mapsto(\text{FPR}(t),\text{TPR}(t))$ from $(0,0)$ to $(1,1)$. It describes the score's ability to separate the classes at every possible operating point simultaneously.</p>
<ul>
<li><strong>Invariance.</strong> The ROC depends only on the ordering of scores: any strictly increasing transformation of $s$ leaves it unchanged. It is also unchanged by the prevalence, since TPR and FPR condition on the class.</li>
<li><strong>Chance.</strong> A score independent of $Y$ gives the diagonal $\text{TPR} = \text{FPR}$.</li>
<li><strong>Randomization.</strong> Flipping a biased coin between two thresholds achieves any point on the segment joining their ROC points, so the achievable set is the convex hull of the curve (the ROCCH).</li>
</ul>
<div class="thm" data-kind="Proposition" data-name="Slope of the ROC and the optimal curve">
<p>If the score has class-conditional densities $g_0, g_1$, then at threshold $t$ the ROC has slope $\dfrac{\d\,\text{TPR}}{\d\,\text{FPR}} = \dfrac{g_1(t)}{g_0(t)}$. The ROC of the likelihood ratio $\Lambda(x) = f_1(x)/f_0(x)$ (or any increasing function of it, such as $\eta(x)$) lies weakly above the ROC of every other score, and it is concave.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\text{TPR}(t) = \int_t^\infty g_1$ and $\text{FPR}(t) = \int_t^\infty g_0$, so $\frac{\d\,\text{TPR}}{\d t} = -g_1(t)$, $\frac{\d\,\text{FPR}}{\d t} = -g_0(t)$, and the chain rule gives the slope. Dominance: for any FPR level $\alpha$, the TPR is the power of a test of $H_0: Y=0$ against $H_1: Y=1$ at size $\alpha$, and by the <a href="#/t/neyman-pearson">Neyman–Pearson lemma</a> the likelihood-ratio test has maximal power at every size. Concavity: when the score is $\Lambda$ itself, the density ratio of $\Lambda$ at value $t$ equals $t$ (a likelihood ratio of a likelihood ratio is itself), so the slope at threshold $t$ is $t$, which decreases as the threshold decreases — that is, as we move right along the curve.</p>
</details>
<p>A non-concave ROC therefore signals a score that is not monotone in the likelihood ratio somewhere — some regions of score space are ranked in the wrong order — and the optimal ROC is the upper boundary of everything achievable.</p>`,
      },
      {
        title: 'AUC is a probability',
        html: R`
<div class="thm" data-name="AUC as a concordance probability">
<p>Let $S^+$ and $S^-$ be independent scores of a random positive and a random negative. Then the area under the ROC curve is
$$\text{AUC} = \P(S^+ > S^-) + \tfrac12\P(S^+ = S^-).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Assume first that $S^-$ has a density $g_0$. Parametrize the curve by the threshold: as $t$ decreases from $+\infty$ to $-\infty$, $x = \text{FPR}(t) = \P(S^- > t)$ increases from $0$ to $1$, with $\d x = g_0(t)\,\d t$ (in absolute value). Then
$$\text{AUC} = \int_0^1\text{TPR}\,\d x = \int_{-\infty}^{\infty}\P(S^+ > t)\,g_0(t)\,\d t = \P(S^+ > S^-),$$
the last step by conditioning on $S^- = t$ and using independence. For atoms, the ROC joins the points before and after a jump by a straight segment; the trapezoid over a tie at value $t$ contributes $\P(S^+ > t)\P(S^-=t) + \frac12\P(S^+=t)\P(S^-=t)$, giving the half-credit term.</p>
</details>
<p>On a test set with $m_+$ positives and $m_-$ negatives, the empirical AUC is therefore the fraction of the $m_+m_-$ positive–negative pairs that are correctly ordered — exactly the Mann–Whitney statistic $U/(m_+m_-)$ (<a href="#/t/rank-tests">rank tests</a>), which supplies standard errors and tests for comparing AUCs.</p>
<div class="ex" data-name="Binormal scores">
<p>If $S^-\sim\N(\mu_0,\sigma_0^2)$ and $S^+\sim\N(\mu_1,\sigma_1^2)$ independently, then $S^+ - S^-\sim\N(\mu_1-\mu_0,\sigma_0^2+\sigma_1^2)$ and $\text{AUC} = \Phi\big((\mu_1-\mu_0)/\sqrt{\sigma_0^2+\sigma_1^2}\big)$. With $\sigma_0 = \sigma_1 = 1$ and $\mu_1 - \mu_0 = 1$, AUC $= \Phi(0.707)\approx 0.76$.</p>
</div>
<div class="warn" data-name="What AUC does not measure">
<p>AUC averages performance over all thresholds, including ones you would never use; two classifiers with crossing ROCs can have equal AUC and very different value at the relevant operating point. It is blind to calibration (it is invariant to monotone transformations of the score). And as an implicit cost model it is incoherent: Hand (2009) showed that AUC corresponds to averaging the expected cost over a distribution of cost ratios that depends on the classifier itself. Use it to compare <em>rankers</em>; use expected cost at the operating point to compare <em>decision rules</em>.</p>
</div>`,
      },
      {
        title: 'Choosing an operating point: costs and priors',
        html: R`
<p>With costs $c_{\text{FP}}$ and $c_{\text{FN}}$ and prevalence $\pi$, the expected cost per case at threshold $t$ is
$$C(t) = c_{\text{FP}}(1-\pi)\,\text{FPR}(t) + c_{\text{FN}}\,\pi\,\big(1-\text{TPR}(t)\big).$$
In ROC space, the level sets of $C$ are parallel <strong>iso-cost lines</strong> with slope
$$m = \frac{c_{\text{FP}}(1-\pi)}{c_{\text{FN}}\,\pi},$$
and the optimal operating point is where the highest such line touches the ROC (convex hull): the point where the ROC's slope equals $m$. By the slope proposition, that means $g_1(t)/g_0(t) = m$ — the likelihood-ratio threshold of the cost-sensitive <a href="#/t/bayes-classifier">Bayes rule</a>. If the score is a calibrated probability $\hat\eta(x)$, the optimal threshold is directly $\hat\eta > c_{\text{FP}}/(c_{\text{FP}} + c_{\text{FN}})$, with no ROC needed.</p>
<div class="note" data-name="Prior shift">
<p>If the deployment prevalence $\pi'$ differs from the training prevalence $\pi$ but the class-conditionals do not change (label shift), the ROC is unchanged and only the iso-cost slope moves. For a calibrated posterior, adjust the odds: $\frac{\eta'}{1-\eta'} = \frac{\eta}{1-\eta}\cdot\frac{\pi'/(1-\pi')}{\pi/(1-\pi)}$.</p>
</div>
<p>In the lab, two Gaussian score distributions generate an ROC. Move the threshold and watch the operating point slide along the curve; change the costs or prevalence and watch the iso-cost line rotate and the optimal point move. Switch to the precision–recall view and change the prevalence: the ROC does not move, but the PR curve collapses.</p>`,
        lab: 'classifier-evaluation-roc',
      },
      {
        title: 'Precision, recall, and class imbalance',
        html: R`
<p>The <strong>precision–recall curve</strong> plots precision against recall ($=\text{TPR}$) as the threshold varies. Because precision involves $\pi$, the PR curve of a fixed score changes with prevalence even though its ROC does not. That is a feature when the positive class is rare and false positives are what users experience (search, fraud alerts, screening): a ROC near the top-left corner can still correspond to a precision of a few percent.</p>
<ul>
<li><strong>Summaries.</strong> Average precision (AP) is the area under the PR curve, computed as $\sum_k(R_k - R_{k-1})P_k$ over thresholds. $F_1 = 2PR/(P+R)$, the harmonic mean, depends on the threshold and implicitly fixes a cost ratio.</li>
<li><strong>Baseline.</strong> A random score has precision $\pi$ at every recall, so AP should be compared with $\pi$, not $1/2$.</li>
<li><strong>No linear interpolation.</strong> Points between two PR operating points are <em>not</em> on the straight line between them; interpolating linearly overstates performance (Davis &amp; Goadrich, 2006). Interpolate in ROC space and map back with the precision formula.</li>
</ul>
<p>There is a one-to-one map between ROC and PR curves for a given $\pi$, and a curve dominates in ROC space iff it dominates in PR space. The two views carry the same information about the score; they weight the regions of the curve differently.</p>`,
      },
      {
        title: 'Proper scoring rules',
        html: R`
<p>To evaluate probabilities themselves, use a loss $S(p, y)$ for reporting probability $p$ that $Y = 1$ when the outcome is $y\in\{0,1\}$. If $Y\sim\mathrm{Bern}(q)$, the expected loss is $\bar S(p, q) = q\,S(p,1) + (1-q)\,S(p,0)$.</p>
<div class="def" data-name="Proper scoring rule">
<p>$S$ is <em>proper</em> if $\bar S(q,q)\le\bar S(p,q)$ for all $p, q$ — honesty is optimal — and <em>strictly proper</em> if equality forces $p = q$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Brier score and log loss are strictly proper">
<p>For the Brier score $S(p,y) = (p-y)^2$, $\;\bar S(p,q) = (p-q)^2 + q(1-q)$. For the log loss $S(p,y) = -y\log p - (1-y)\log(1-p)$, $\;\bar S(p,q) = H(q) + \KL\big(\mathrm{Bern}(q)\,\|\,\mathrm{Bern}(p)\big)$, where $H(q)$ is the binary entropy. Both are uniquely minimized at $p = q$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Brier: $\bar S(p,q) = q(1-p)^2 + (1-q)p^2 = p^2 - 2pq + q = (p-q)^2 + q - q^2$. Log loss: $\bar S(p,q) = -q\log p - (1-q)\log(1-p) = \big[-q\log q - (1-q)\log(1-q)\big] + \big[q\log\frac qp + (1-q)\log\frac{1-q}{1-p}\big]$, and the second bracket is a KL divergence, which is $\ge 0$ with equality iff $p = q$ (Gibbs' inequality; <a href="#/t/entropy-kl">entropy and KL</a>).</p>
</details>
<div class="ex" data-name="An improper score">
<p>The absolute error $S(p,y) = |p - y|$ has $\bar S(p,q) = q(1-p) + (1-q)p = q + p(1-2q)$, linear in $p$. It is minimized at $p = 1$ if $q > 1/2$ and $p = 0$ if $q < 1/2$: it rewards exaggerating to certainty. Accuracy (0–1 loss of the thresholded forecast) is proper but not strictly so — any $p$ on the correct side of $1/2$ is equally good — which is why it cannot assess probability quality.</p>
</div>
<p><strong>Characterization (Savage, 1971; Gneiting &amp; Raftery, 2007).</strong> Every proper scoring rule for a binary outcome has the form $S(p,y) = G(p) + G'(p)(y - p)$ for a concave function $G$ (with $G'$ a supergradient); $G(q) = \bar S(q,q)$ is the minimum achievable expected loss, a generalized entropy. Brier corresponds to $G(q) = q(1-q)$, log loss to Shannon entropy. Strict concavity gives strict propriety. For outcomes with three or more categories, log loss is essentially the only strictly proper rule that depends on the forecast only through the probability assigned to the realized outcome (Bernardo, 1979). It punishes confident mistakes without bound, while Brier is bounded and more forgiving.</p>`,
      },
      {
        title: 'Calibration and the decomposition of the Brier score',
        html: R`
<div class="def" data-name="Calibration">
<p>A probabilistic classifier $\hat p(x)$ is <em>calibrated</em> if $\P(Y=1\mid\hat p(X) = p) = p$ for all $p$ in its range. Write $c(p) = \P(Y=1\mid\hat p(X) = p)$ for the <em>calibration function</em>; calibration means $c(p) = p$.</p>
</div>
<p>Calibration and discrimination are separate: the constant forecast $\hat p\equiv\pi$ is perfectly calibrated and useless for ranking; the score $\hat p = \eta^{10}$ ranks perfectly (same ROC as $\eta$) and is badly miscalibrated. A <strong>reliability diagram</strong> bins the forecasts and plots the observed frequency in each bin against the mean forecast; points on the diagonal indicate calibration. The expected calibration error $\text{ECE} = \sum_b\frac{m_b}{m}|\bar y_b - \bar p_b|$ summarizes it, though it depends on the binning and is biased upward by noise in small bins.</p>
<div class="thm" data-kind="Proposition" data-name="Calibration–refinement decomposition">
<p>$$\E(\hat p(X) - Y)^2 = \underbrace{\E\big(\hat p(X) - c(\hat p(X))\big)^2}_{\text{calibration error}} + \underbrace{\E\,c(\hat p(X))\big(1 - c(\hat p(X))\big)}_{\text{refinement}},$$
and the refinement term equals $\pi(1-\pi) - \Var\big(c(\hat p(X))\big)$ (uncertainty minus resolution; Murphy, 1973).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Condition on $\hat p(X) = p$. Then $Y\sim\mathrm{Bern}(c(p))$, so $\E[(p - Y)^2\mid\hat p = p] = (p - c(p))^2 + \Var(Y\mid\hat p = p) = (p-c(p))^2 + c(p)(1-c(p))$ by the bias–variance identity. Average over $\hat p(X)$. For the second statement, $\E c(\hat p(X)) = \E Y = \pi$ by the tower property, so $\E[c(1-c)] = \pi - \E c^2 = \pi - (\Var c + \pi^2)$.</p>
</details>
<p>So a proper score rewards two things: being calibrated (first term zero) and being <em>refined</em> — sorting cases into groups whose outcome frequencies are far from the base rate (large resolution). Recalibration methods fix only the first term: <strong>Platt scaling</strong> fits a logistic regression of $Y$ on the score; <strong>isotonic regression</strong> fits a monotone step function (the pool-adjacent-violators algorithm); <strong>temperature scaling</strong> divides logits by a single fitted constant. All are monotone, so they leave the ROC and AUC unchanged while improving log loss and Brier. Fit them on data not used for training.</p>
<p>In the lab, the forecasts are a distorted version of the true posterior: a temperature makes them over- or underconfident and a shift biases them. Watch AUC stay fixed while Brier score, log loss and the reliability diagram respond.</p>`,
        lab: 'classifier-evaluation-calibration',
      },
    ],
    quiz: [
      { q: R`Prove that AUC is unchanged if every score is replaced by $\log s$ (for positive scores), and explain why the Brier score is not.`,
        a: R`AUC $= \P(S^+ > S^-) + \frac12\P(S^+ = S^-)$, and a strictly increasing transformation preserves every strict inequality and equality between pairs. The Brier score uses the numerical value of the forecast as a probability; transforming it changes $(p - y)^2$ and generally breaks calibration.` },
      { q: R`Positives are 5% of cases, a missed positive costs 10, and a false alarm costs 1. What is the slope of the iso-cost lines in ROC space, and what posterior threshold is optimal for a calibrated model?`,
        a: R`$m = c_{\text{FP}}(1-\pi)/(c_{\text{FN}}\pi) = 1\cdot 0.95/(10\cdot 0.05) = 1.9$. The optimal point is where the ROC has slope $1.9$. For a calibrated posterior, predict positive when $\hat\eta > c_{\text{FP}}/(c_{\text{FP}}+c_{\text{FN}}) = 1/11\approx 0.091$.` },
      { q: R`Show that the expected absolute-error score $\E|p - Y|$ is improper, and identify the forecast it rewards when $q = 0.7$.`,
        a: R`$\bar S(p,q) = q(1-p) + (1-q)p = q + p(1-2q)$. For $q = 0.7$ the slope in $p$ is $1 - 1.4 = -0.4 < 0$, so the expected score is minimized at $p = 1$, not at the true $0.7$: it rewards overstating confidence.` },
      { q: R`A model has AUC 0.85 but its predicted probabilities are all between 0.4 and 0.6. Which terms of the Brier decomposition does this hurt, and what fixes it?`,
        a: R`The ranking is good, so the model has resolution available, but the forecasts are underconfident: $c(\hat p)$ ranges much more widely than $\hat p$, giving a large calibration-error term. A monotone recalibration (Platt or isotonic) fitted on held-out data stretches the forecasts, removes most of the calibration error, and leaves the AUC unchanged.` },
    ],
  });
})();
