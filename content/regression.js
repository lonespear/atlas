/* Regression & Linear Models. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'simple-linear-regression',
    domain: 'regression',
    order: 10,
    title: 'Simple Linear Regression',
    subtitle: 'One predictor, one line, and everything that can be said about it exactly',
    level: 1,
    sources: { aos: '13.1–13.2', cb: '§11.3', other: 'Seber & Lee, Linear Regression Analysis (2003), Ch. 1–4' },
    prereqs: ['expectation', 'point-estimation'],
    related: ['ols-geometry', 'regression-inference', 'mle', 'conditional-expectation'],
    tags: ['least squares', 'regression function', 'correlation', 'R²'],
    summary: R`Simple linear regression models the conditional mean of a response $Y$ as a straight line in a single predictor $x$: $\E(Y\mid x)=\beta_0+\beta_1x$. Least squares picks the line minimizing the sum of squared vertical deviations and has a closed form, $\hat\beta_1 = S_{xy}/S_{xx}$, $\hat\beta_0=\bar y-\hat\beta_1\bar x$. Because both estimates are <em>linear</em> in the responses, their means and variances follow from two lines of algebra, and under normal errors their exact sampling distributions are available. The simple case is worth mastering on its own: every idea in multiple regression — projection, leverage, the role of the design, the gap between prediction and causation — is already visible here with one predictor.`,
    survey: {
      task: R`Estimate a linear conditional mean $\E(Y\mid x)$; predict $Y$ at new $x$`,
      assumptions: R`Linearity of the mean; uncorrelated errors with constant variance $\sigma^2$; normality only for exact small-sample inference`,
      output: R`$\hat\beta_0,\hat\beta_1$, $\hat\sigma^2 = \text{RSS}/(n-2)$, fitted line, $R^2$`,
      strengths: 'Closed form; exact finite-sample theory; transparent role of design spread',
      weaknesses: 'Sensitive to outliers and leverage; linearity is an assumption; association is not causation',
      complexity: R`$O(n)$`,
    },
    sections: [
      {
        title: 'The regression function and the best linear predictor',
        html: R`
<p>Let $(X,Y)$ be a random pair with $\E Y^2<\infty$. The <strong>regression function</strong> is $r(x)=\E(Y\mid X=x)$. It is the object regression estimates, and it earns that status through an optimality property.</p>
<div class="thm" data-kind="Proposition" data-name="The conditional mean is the best predictor">
<p>Among all (measurable) functions $g$, the mean squared prediction error $\E\big(Y-g(X)\big)^2$ is minimized by $g=r$, and
$$\E\big(Y-g(X)\big)^2=\E\big(Y-r(X)\big)^2+\E\big(r(X)-g(X)\big)^2.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Add and subtract $r(X)$ and expand. The cross term is $2\E\big[(Y-r(X))(r(X)-g(X))\big]$. Condition on $X$: the factor $r(X)-g(X)$ is fixed, and $\E[Y-r(X)\mid X]=0$, so by the tower property the cross term vanishes. The remaining two terms are non-negative and only the second depends on $g$.</p>
</details>
<p>If we restrict $g$ to straight lines, the minimizer is the <strong>best linear predictor</strong> (the linear projection of $Y$ on $(1,X)$). Setting the derivatives of $\E(Y-b_0-b_1X)^2$ to zero gives
$$b_1^\star=\frac{\Cov(X,Y)}{\Var(X)},\qquad b_0^\star=\E Y-b_1^\star\,\E X .$$
This line exists for <em>any</em> joint distribution with finite second moments. It coincides with $r$ exactly when $r$ happens to be linear — for example when $(X,Y)$ is bivariate normal (see <a href="#/t/conditional-expectation">conditional expectation</a>).</p>
<div class="note">
<p>Two different things are both called "the regression line": a <em>model</em> asserting that $r(x)$ is linear, and the <em>projection</em> that is the best line whether or not $r$ is linear. Least squares estimates the projection consistently in either case. What the linearity assumption buys is interpretability ($\beta_1$ is the change in mean response per unit of $x$, everywhere) and the exact inference of later sections.</p>
</div>
<div class="def" data-name="Simple linear regression model">
<p>Observations $(x_i,Y_i)$, $i=1,\dots,n$, with the $x_i$ treated as fixed (or conditioned on), satisfy
$$Y_i=\beta_0+\beta_1x_i+\varepsilon_i,\qquad \E\varepsilon_i=0,\quad \Var\varepsilon_i=\sigma^2,\quad \Cov(\varepsilon_i,\varepsilon_j)=0\ (i\ne j).$$
The <em>normal</em> model adds $\varepsilon_i\iid\N(0,\sigma^2)$.</p>
</div>`,
      },
      {
        title: 'Least squares',
        html: R`
<p>The least-squares estimates minimize the residual sum of squares
$$\text{RSS}(b_0,b_1)=\sum_{i=1}^n\big(Y_i-b_0-b_1x_i\big)^2 .$$
This is a convex quadratic in $(b_0,b_1)$, so the stationary point is the global minimum. Write $\bar x,\bar Y$ for sample means and
$$S_{xx}=\sum_i(x_i-\bar x)^2,\qquad S_{xy}=\sum_i(x_i-\bar x)(Y_i-\bar Y)=\sum_i(x_i-\bar x)Y_i .$$</p>
<div class="thm" data-kind="Proposition" data-name="Normal equations and their solution">
<p>If $S_{xx}>0$ (the $x_i$ are not all equal), the unique minimizer of RSS is
$$\hat\beta_1=\frac{S_{xy}}{S_{xx}},\qquad \hat\beta_0=\bar Y-\hat\beta_1\bar x .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The partial derivatives give the <em>normal equations</em>
$$\sum_i(Y_i-b_0-b_1x_i)=0,\qquad \sum_i x_i(Y_i-b_0-b_1x_i)=0 .$$
The first gives $b_0=\bar Y-b_1\bar x$. Substituting into the second and subtracting $\bar x$ times the first, $\sum_i(x_i-\bar x)\big(Y_i-\bar Y-b_1(x_i-\bar x)\big)=0$, i.e. $S_{xy}=b_1S_{xx}$. The Hessian is $2\begin{pmatrix}n&\sum x_i\\ \sum x_i&\sum x_i^2\end{pmatrix}$, whose determinant $4nS_{xx}$ is positive, so the critical point is the strict global minimum.</p>
</details>
<p>The normal equations are worth reading as statements about the <strong>residuals</strong> $e_i=Y_i-\hat\beta_0-\hat\beta_1x_i$: they say $\sum_ie_i=0$ and $\sum_ix_ie_i=0$. The residual vector is orthogonal to the constant vector and to the predictor vector. That is the one-predictor shadow of the projection geometry in <a href="#/t/ols-geometry">multiple regression</a>. Two immediate consequences: the fitted line passes through $(\bar x,\bar Y)$, and the residuals are uncorrelated (in-sample) with the fitted values.</p>
<p>Why <em>vertical</em> squared deviations? Because the target is $\E(Y\mid x)$ and the error lives in $Y$. Minimizing perpendicular distances (total least squares, the first principal component) answers a different question and gives a different line; it is appropriate when both coordinates are measured with comparable error.</p>`,
        lab: 'simple-linear-regression-drag',
      },
      {
        title: 'Sampling properties: unbiasedness and variance',
        html: R`
<p>The key structural fact is that $\hat\beta_1$ is a <strong>linear combination of the responses</strong> with weights fixed by the design:
$$\hat\beta_1=\sum_{i=1}^n c_iY_i,\qquad c_i=\frac{x_i-\bar x}{S_{xx}},\qquad \sum_ic_i=0,\quad \sum_ic_ix_i=1,\quad \sum_ic_i^2=\frac1{S_{xx}} .$$</p>
<div class="thm" data-kind="Theorem" data-name="Moments of the least-squares estimates">
<p>Under the (not necessarily normal) model above,
$$\E\hat\beta_1=\beta_1,\quad \Var\hat\beta_1=\frac{\sigma^2}{S_{xx}},\qquad \E\hat\beta_0=\beta_0,\quad \Var\hat\beta_0=\sigma^2\Big(\frac1n+\frac{\bar x^2}{S_{xx}}\Big),\qquad \Cov(\hat\beta_0,\hat\beta_1)=-\frac{\bar x\,\sigma^2}{S_{xx}} .$$
Moreover $\hat\sigma^2=\text{RSS}/(n-2)$ is unbiased for $\sigma^2$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\E\hat\beta_1=\sum c_i(\beta_0+\beta_1x_i)=\beta_0\sum c_i+\beta_1\sum c_ix_i=\beta_1$. Uncorrelated errors give $\Var\hat\beta_1=\sigma^2\sum c_i^2=\sigma^2/S_{xx}$. Next, $\bar Y$ and $\hat\beta_1$ are uncorrelated because $\Cov(\bar Y,\sum c_iY_i)=\frac{\sigma^2}{n}\sum c_i=0$; so $\Var\hat\beta_0=\Var\bar Y+\bar x^2\Var\hat\beta_1$ and $\Cov(\hat\beta_0,\hat\beta_1)=-\bar x\Var\hat\beta_1$.</p>
<p>For $\hat\sigma^2$: $e_i=(Y_i-\bar Y)-\hat\beta_1(x_i-\bar x)$, so $\text{RSS}=\sum(Y_i-\bar Y)^2-\hat\beta_1^2S_{xx}$. Now $\E\sum(Y_i-\bar Y)^2=(n-1)\sigma^2+\beta_1^2S_{xx}$ and $\E\hat\beta_1^2S_{xx}=\sigma^2+\beta_1^2S_{xx}$. Subtracting, $\E\,\text{RSS}=(n-2)\sigma^2$. Two degrees of freedom are spent on the two estimated coefficients.</p>
</details>
<p>The formula $\Var\hat\beta_1=\sigma^2/S_{xx}$ is a statement about <strong>experimental design</strong>. Precision depends on the spread of the $x$'s, not just on $n$: doubling the range of $x$ quarters the variance, the same as quadrupling $n$. If you control $x$, the variance-minimizing design puts half the points at each end of the feasible range — at the cost of having no ability to detect curvature. The intercept is estimated worst when $\bar x$ is far from $0$, because $\hat\beta_0$ is then an extrapolation to $x=0$; centering $x$ makes $\hat\beta_0=\bar Y$ and decorrelates the two estimates.</p>`,
        lab: 'simple-linear-regression-sampling',
      },
      {
        title: 'Normal errors: likelihood and exact distributions',
        html: R`
<p>Under $\varepsilon_i\iid\N(0,\sigma^2)$, the log-likelihood is
$$\ell(\beta_0,\beta_1,\sigma^2)=-\frac n2\log(2\pi\sigma^2)-\frac{1}{2\sigma^2}\text{RSS}(\beta_0,\beta_1).$$
For every fixed $\sigma^2$ it is maximized in $(\beta_0,\beta_1)$ by minimizing RSS, so the <a href="#/t/mle">MLE</a> of the coefficients <em>is</em> least squares. Profiling then gives $\hat\sigma^2_{\text{MLE}}=\text{RSS}/n$, which is biased downward by the factor $(n-2)/n$.</p>
<div class="thm" data-kind="Theorem" data-name="Exact sampling distributions">
<p>In the normal model,
$$\hat\beta_1\sim\N\!\Big(\beta_1,\frac{\sigma^2}{S_{xx}}\Big),\qquad \frac{(n-2)\hat\sigma^2}{\sigma^2}\sim\chi^2_{n-2},\qquad \hat\beta_1\indep\hat\sigma^2,$$
and therefore
$$T=\frac{\hat\beta_1-\beta_1}{\hat\sigma/\sqrt{S_{xx}}}\sim t_{n-2}.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$\hat\beta_1$ is a linear combination of independent normals, hence normal with the moments already computed. The residual vector $e$ and $(\bar Y,\hat\beta_1)$ are jointly normal and uncorrelated (each $e_i$ is orthogonal to the weight vectors $\mathbf 1/n$ and $c$ — that is the normal equations again), hence independent; RSS is a function of $e$. The $\chi^2_{n-2}$ law follows because RSS is $\sigma^2$ times the squared length of the projection of a standard normal vector onto an $(n-2)$-dimensional subspace. The general proof is in <a href="#/t/regression-inference">regression inference</a>; the ratio is then $t$ by the definition in <a href="#/t/sampling-distributions">sampling distributions</a>.</p>
</details>
<p>This yields the familiar interval $\hat\beta_1\pm t_{n-2,\alpha/2}\,\hat\sigma/\sqrt{S_{xx}}$ and the $t$-test of $H_0:\beta_1=0$. Without normality the same statements hold asymptotically provided $S_{xx}\to\infty$ and no single $x_i$ dominates ($\max_i(x_i-\bar x)^2/S_{xx}\to0$), by the Lindeberg CLT applied to $\sum c_i\varepsilon_i$.</p>`,
      },
      {
        title: 'Correlation, R², and regression toward the mean',
        html: R`
<p>Let $r$ be the sample correlation and $s_x,s_y$ the sample standard deviations. Then
$$\hat\beta_1=r\,\frac{s_y}{s_x},\qquad\text{equivalently}\qquad \frac{\hat y-\bar y}{s_y}=r\,\frac{x-\bar x}{s_x}.$$
In standard units the fitted line has slope $r$, with $|r|\le1$. A point one standard deviation above average in $x$ is predicted to be only $r$ standard deviations above average in $y$. This is <strong>regression toward the mean</strong>, and it is pure arithmetic, not a causal force: it appears whenever $|r|<1$, in either direction of prediction.</p>
<div class="thm" data-kind="Proposition" data-name="Sum-of-squares decomposition">
<p>With an intercept in the model,
$$\underbrace{\sum_i(Y_i-\bar Y)^2}_{\text{SST}}=\underbrace{\sum_i(\hat Y_i-\bar Y)^2}_{\text{SSR}}+\underbrace{\sum_ie_i^2}_{\text{RSS}},\qquad R^2:=\frac{\text{SSR}}{\text{SST}}=r^2 .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$Y_i-\bar Y=(\hat Y_i-\bar Y)+e_i$ and $\hat Y_i-\bar Y=\hat\beta_1(x_i-\bar x)$. The cross term is $\hat\beta_1\sum(x_i-\bar x)e_i=0$ by the normal equations. Finally $\text{SSR}=\hat\beta_1^2S_{xx}=S_{xy}^2/S_{xx}$, and dividing by $\text{SST}=S_{yy}$ gives $S_{xy}^2/(S_{xx}S_{yy})=r^2$.</p>
</details>
<div class="ex" data-name="The two regression lines">
<p>Suppose standardized scores on two exams have correlation $r=0.6$. Regressing exam 2 on exam 1 gives slope $0.6$; regressing exam 1 on exam 2 also gives slope $0.6$ <em>in its own coordinates</em>. Drawn in the (exam 1, exam 2) plane, the second line has slope $1/0.6\approx1.67$. The two lines coincide only if $|r|=1$. Neither is "the" relationship; each answers a different prediction question. Students who did best on exam 1 will, on average, do less spectacularly on exam 2 — and students who did best on exam 2 did, on average, less spectacularly on exam 1.</p>
</div>
<p>$R^2$ measures the fraction of in-sample variation captured by the line. It is <em>not</em> a measure of whether the linear model is correct, and it depends on the design: for a fixed true line and noise level, spreading the $x$'s out increases $R^2$ because SST grows while RSS does not.</p>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Extrapolation">
<p>The fit constrains the mean function only over the range of observed $x$. The variance of $\hat\beta_0+\hat\beta_1x_0$ is $\sigma^2\big(\tfrac1n+\tfrac{(x_0-\bar x)^2}{S_{xx}}\big)$, which grows quadratically away from $\bar x$ — and that formula assumes linearity continues to hold out there, which the data cannot check.</p>
</div>
<div class="warn" data-name="Errors in the predictor attenuate the slope">
<p>If we observe $W_i=x_i+U_i$ with measurement error $U_i$ of variance $\sigma_U^2$ independent of everything else, and the $x_i$ have variance $\sigma_x^2$, regressing $Y$ on $W$ gives $\hat\beta_1\pto\beta_1\,\sigma_x^2/(\sigma_x^2+\sigma_U^2)$. The slope is biased toward zero by the <em>reliability ratio</em>, and more data does not help.</p>
</div>
<div class="warn" data-name="One influential point">
<p>Because $\hat\beta_1=\sum c_iY_i$ with $c_i\propto x_i-\bar x$, points far from $\bar x$ get the largest weights. A single point at an extreme $x$ can rotate the line almost arbitrarily. Try it in the first lab, and see <a href="#/t/regression-diagnostics">leverage and influence</a>.</p>
</div>
<div class="warn" data-name="A good fit is not a mechanism">
<p>$\beta_1$ describes how the conditional mean of $Y$ changes across units with different $x$. It equals the effect of <em>intervening</em> on $x$ only under assumptions — randomization, or no unmeasured confounding — discussed in <a href="#/t/causal-inference">causal inference</a>.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`You can choose the $x$ values for $n=20$ runs anywhere in $[0,10]$. Which design minimizes $\Var\hat\beta_1$, and what is the price?`,
        a: R`Ten runs at $x=0$ and ten at $x=10$. This maximizes $S_{xx}=\sum(x_i-\bar x)^2$ (here $20\cdot25=500$), and $\Var\hat\beta_1=\sigma^2/S_{xx}$. The price is that with only two distinct $x$ values the data cannot detect any departure from linearity.` },
      { q: R`Why is $\text{RSS}/(n-2)$ rather than $\text{RSS}/n$ the unbiased estimator of $\sigma^2$?`,
        a: R`Because $\E\,\text{RSS}=(n-2)\sigma^2$. The residuals satisfy two linear constraints ($\sum e_i=0$, $\sum x_ie_i=0$), so they live in an $(n-2)$-dimensional subspace; fitting two coefficients absorbs two dimensions' worth of noise. $\text{RSS}/n$ is the MLE and is biased by the factor $(n-2)/n$.` },
      { q: R`A company assigns a training course to its lowest-performing employees, and their next-quarter scores improve on average. Why is this not evidence the course works?`,
        a: R`Regression toward the mean. If quarter-to-quarter scores have correlation $r<1$, those selected for being far below average in one quarter will on average be closer to the mean in the next, course or no course. A randomized control group of low performers is needed to separate the course's effect from this artifact.` },
      { q: R`Is the least-squares line still meaningful if $\E(Y\mid x)$ is not linear?`,
        a: R`Yes: it consistently estimates the best linear predictor $b_1^\star=\Cov(X,Y)/\Var(X)$, $b_0^\star=\E Y-b_1^\star\E X$ (for a random design). But $\hat\beta_1$ no longer has the interpretation "change in mean response per unit $x$ at every $x$", the residuals are no longer mean-zero given $x$, and the usual standard errors can be wrong.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'ols-geometry',
    domain: 'regression',
    order: 20,
    title: 'Multiple Regression: Least Squares as Projection',
    subtitle: 'The normal equations, the hat matrix, and what a coefficient means when predictors are correlated',
    level: 2,
    sources: { aos: '13.5', cb: '§11.3', other: 'Seber & Lee (2003), Ch. 3; Golub & Van Loan, Matrix Computations, Ch. 5' },
    prereqs: ['simple-linear-regression'],
    related: ['gauss-markov', 'regression-diagnostics', 'ridge-regression', 'pca', 'multivariate'],
    tags: ['normal equations', 'hat matrix', 'orthogonal projection', 'Frisch–Waugh–Lovell', 'collinearity'],
    summary: R`With $p$ predictors the model is $y=X\beta+\varepsilon$ for an $n\times p$ design matrix $X$. Least squares has a clean geometric meaning: the fitted vector $\hat y$ is the orthogonal projection of $y$ onto the column space of $X$, and the residual is what is left over, perpendicular to every column. The projection is the <em>hat matrix</em> $H=X(X^\top X)^{-1}X^\top$ — symmetric, idempotent, with trace $p$. That one picture explains the normal equations, the Pythagorean decomposition behind $R^2$ and ANOVA, why residuals have $n-p$ degrees of freedom, and — through the Frisch–Waugh–Lovell theorem — exactly what a coefficient means when the predictors are correlated.`,
    survey: {
      task: R`Fit a linear mean $X\beta$ with $p$ predictors; decompose $y$ into fit and residual`,
      assumptions: R`$X$ of full column rank $p$ (else $\hat\beta$ is not unique, $\hat y$ still is)`,
      output: R`$\hat\beta=(X^\top X)^{-1}X^\top y$, $H$, $\hat y=Hy$, $e=(I-H)y$`,
      strengths: 'Exact geometry; closed form; coefficients have a precise partial-effect meaning (FWL)',
      weaknesses: R`Collinearity inflates variances; normal equations square the condition number`,
      complexity: R`$O(np^2)$ via QR`,
    },
    sections: [
      {
        title: 'The model in matrix form and the normal equations',
        html: R`
<p>Stack the responses into $y\in\R^n$ and the predictors (including a column of ones for the intercept) into the design matrix $X\in\R^{n\times p}$. The linear model is
$$y=X\beta+\varepsilon,\qquad \E\varepsilon=0,\quad \Var\varepsilon=\sigma^2I_n .$$
Least squares minimizes $Q(b)=\|y-Xb\|^2$ over $b\in\R^p$. Write $\mathcal C(X)=\{Xb:b\in\R^p\}$ for the column space.</p>
<div class="thm" data-kind="Theorem" data-name="Normal equations">
<p>$\hat\beta$ minimizes $Q$ if and only if it solves the normal equations
$$X^\top X\,\hat\beta=X^\top y,\qquad\text{equivalently}\qquad X^\top(y-X\hat\beta)=0 .$$
Solutions always exist. If $\operatorname{rank}X=p$ the solution is unique, $\hat\beta=(X^\top X)^{-1}X^\top y$. In every case the fitted vector $\hat y=X\hat\beta$ is unique.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\hat\beta$ satisfy $X^\top(y-X\hat\beta)=0$. For any $b$, write $y-Xb=(y-X\hat\beta)+X(\hat\beta-b)$. The two pieces are orthogonal because $(X(\hat\beta-b))^\top(y-X\hat\beta)=(\hat\beta-b)^\top X^\top(y-X\hat\beta)=0$. Hence
$$\|y-Xb\|^2=\|y-X\hat\beta\|^2+\|X(\hat\beta-b)\|^2\ \ge\ \|y-X\hat\beta\|^2,$$
with equality iff $Xb=X\hat\beta$. So every solution of the normal equations is a minimizer, all minimizers share the same fitted vector, and if $X$ has full column rank ($Xv=0\Rightarrow v=0$) the minimizer is unique. Existence: $X^\top y\in\mathcal C(X^\top)=\mathcal C(X^\top X)$ (these spaces coincide because $X^\top Xv=0\Rightarrow\|Xv\|^2=0\Rightarrow Xv=0$), so the system is always consistent. Conversely, any minimizer has zero gradient $-2X^\top(y-Xb)$.</p>
</details>
<p>The proof is the whole story in miniature: the residual $e=y-X\hat\beta$ must be orthogonal to every column of $X$, and Pythagoras does the rest. Calculus is not needed.</p>`,
      },
      {
        title: 'The hat matrix is an orthogonal projection',
        html: R`
<div class="def" data-name="Hat matrix">
<p>For full-rank $X$, $H=X(X^\top X)^{-1}X^\top$. It "puts the hat on $y$": $\hat y=Hy$. The residual maker is $M=I-H$, with $e=My$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Properties of H">
<ol>
<li>$H^\top=H$ and $H^2=H$ (symmetric and idempotent), so $H$ is the orthogonal projection onto $\mathcal C(X)$; $M$ is the orthogonal projection onto $\mathcal C(X)^\perp$.</li>
<li>$HX=X$, $MX=0$, $HM=0$.</li>
<li>Every eigenvalue of $H$ is $0$ or $1$; $\operatorname{rank}H=\tr H=p$ and $\tr M=n-p$.</li>
<li>$\|y\|^2=\|\hat y\|^2+\|e\|^2$ and $\hat y^\top e=0$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) Symmetry is immediate. $H^2=X(X^\top X)^{-1}X^\top X(X^\top X)^{-1}X^\top=H$. A symmetric idempotent matrix is the orthogonal projection onto its range: for any $v$, $v=Hv+(I-H)v$ and $(Hv)^\top(I-H)w=v^\top(H-H^2)w=0$. The range of $H$ is contained in $\mathcal C(X)$ by its form and contains it because $HX=X$. (2) Direct multiplication. (3) If $Hv=\lambda v$ with $v\ne0$ then $\lambda v=H^2v=\lambda^2v$, so $\lambda\in\{0,1\}$. For a diagonalizable matrix with eigenvalues in $\{0,1\}$, rank equals the number of unit eigenvalues equals the trace. By the cyclic property, $\tr H=\tr\big((X^\top X)^{-1}X^\top X\big)=\tr I_p=p$. (4) $\hat y^\top e=y^\top HMy=0$, then Pythagoras.</p>
</details>
<p>Property 3 is the source of every "$n-p$" in regression. The residual vector is confined to an $(n-p)$-dimensional subspace, so $\E\|e\|^2=\E\,\varepsilon^\top M\varepsilon=\sigma^2\tr M=\sigma^2(n-p)$, using $My=M\varepsilon$ and $\E\,\varepsilon^\top A\varepsilon=\sigma^2\tr A$. Hence $\hat\sigma^2=\|e\|^2/(n-p)$ is unbiased with no distributional assumption at all.</p>
<p>The diagonal entries $h_{ii}$ are the <strong>leverages</strong>: $h_{ii}=\partial\hat y_i/\partial y_i$ measures how much observation $i$ pulls its own fitted value. They average $p/n$ and drive the diagnostics in <a href="#/t/regression-diagnostics">regression diagnostics</a>.</p>`,
        lab: 'ols-geometry-projection',
      },
      {
        title: 'Computing least squares: QR and SVD',
        html: R`
<p>Forming $X^\top X$ and solving is the textbook formula but rarely the right algorithm. The 2-norm condition number satisfies $\kappa(X^\top X)=\kappa(X)^2$, so in floating point the normal equations lose twice as many digits as necessary when predictors are nearly collinear.</p>
<div class="cols">
<div><h4>QR</h4><p>Factor $X=QR$ with $Q\in\R^{n\times p}$ orthonormal columns and $R$ upper triangular. Then $H=QQ^\top$ and $R\hat\beta=Q^\top y$ is solved by back-substitution. Householder QR is backward stable; this is what standard software does.</p></div>
<div><h4>SVD</h4><p>$X=UDV^\top$ with singular values $d_1\ge\dots\ge d_p>0$. Then $\hat\beta=VD^{-1}U^\top y$, $H=UU^\top$. Small $d_j$ expose collinearity directly: $\Var\hat\beta=\sigma^2VD^{-2}V^\top$.</p></div>
<div><h4>Rank deficiency</h4><p>If $\operatorname{rank}X=r&lt;p$, the minimizers form an affine set; the minimum-norm one is $\hat\beta^+=X^+y=\sum_{j\le r}v_jd_j^{-1}u_j^\top y$. Only estimable functions $c^\top\beta$ (with $c\in\mathcal C(X^\top)$) are determined.</p></div>
</div>
<p>All three produce the same $\hat y$ — the projection does not depend on which basis of $\mathcal C(X)$ you use. Reparameterizing (centering, rescaling, replacing dummy codings) changes $\hat\beta$ but never $\hat y$, $e$, or RSS.</p>`,
      },
      {
        title: 'What a coefficient means: Frisch–Waugh–Lovell',
        html: R`
<p>Partition $X=[X_1\;X_2]$ and $\beta=(\beta_1,\beta_2)$. What is $\hat\beta_2$ in the joint fit, in terms of simpler regressions?</p>
<div class="thm" data-kind="Theorem" data-name="Frisch–Waugh–Lovell">
<p>Let $M_1=I-X_1(X_1^\top X_1)^{-1}X_1^\top$ be the residual maker for $X_1$, and let $\tilde y=M_1y$, $\tilde X_2=M_1X_2$ be $y$ and $X_2$ with $X_1$ "partialled out". Then the coefficient $\hat\beta_2$ from regressing $y$ on $[X_1\;X_2]$ equals the coefficient from regressing $\tilde y$ on $\tilde X_2$:
$$\hat\beta_2=(\tilde X_2^\top\tilde X_2)^{-1}\tilde X_2^\top\tilde y=(X_2^\top M_1X_2)^{-1}X_2^\top M_1y,$$
and the residuals of the two regressions are identical.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write the full fit as $y=X_1\hat\beta_1+X_2\hat\beta_2+e$ with $X_1^\top e=0$ and $X_2^\top e=0$. Apply $M_1$: since $M_1X_1=0$ and $M_1e=e$ (because $e\perp\mathcal C(X_1)$),
$$M_1y=M_1X_2\hat\beta_2+e .$$
Moreover $(M_1X_2)^\top e=X_2^\top M_1e=X_2^\top e=0$. So $e$ is orthogonal to the columns of $\tilde X_2$, which means $\hat\beta_2$ satisfies the normal equations for the regression of $\tilde y$ on $\tilde X_2$, and its residual is $e$.</p>
</details>
<p>So $\hat\beta_j$ is the slope of $y$ on <em>the part of $x_j$ not linearly explained by the other predictors</em>. "Holding the others fixed" is a slogan for this partialling-out operation; it does not require that the other predictors could actually be held fixed in the world. The scatterplot of $\tilde y$ against $\tilde x_j$ is the <strong>added-variable plot</strong>, whose least-squares slope is exactly $\hat\beta_j$.</p>
<div class="thm" data-kind="Corollary" data-name="Omitted-variable bias">
<p>If the truth is $\E y=X_1\beta_1+X_2\beta_2$ but we regress $y$ on $X_1$ alone, then
$$\E\tilde\beta_1=\beta_1+(X_1^\top X_1)^{-1}X_1^\top X_2\,\beta_2 .$$
The bias is the coefficient matrix from regressing the omitted $X_2$ on the included $X_1$, times $\beta_2$. It vanishes iff $X_1^\top X_2\beta_2=0$ — e.g. if the omitted predictors are orthogonal to the included ones, or irrelevant.</p>
</div>`,
        lab: 'ols-geometry-fwl',
      },
      {
        title: 'Intercepts, R², and collinearity',
        html: R`
<p>When $\mathbf 1\in\mathcal C(X)$, the projection onto $\mathcal C(X)$ contains the projection onto $\operatorname{span}(\mathbf 1)$, and $y-\bar y\mathbf 1=(\hat y-\bar y\mathbf 1)+e$ is an orthogonal split. Thus SST = SSR + RSS and $R^2=\text{SSR}/\text{SST}$ equals the squared sample correlation between $y$ and $\hat y$. Because enlarging $\mathcal C(X)$ can only shorten the residual, $R^2$ never decreases when a predictor is added — even a column of pure noise. The <em>adjusted</em> $R^2=1-\frac{\text{RSS}/(n-p)}{\text{SST}/(n-1)}$ compares unbiased variance estimates and can fall.</p>
<div class="thm" data-kind="Proposition" data-name="Variance inflation">
<p>With an intercept, let $R_j^2$ be the $R^2$ from regressing predictor $x_j$ on all other predictors, and $S_{jj}=\sum_i(x_{ij}-\bar x_j)^2$. Then
$$\Var\hat\beta_j=\frac{\sigma^2}{S_{jj}}\cdot\frac{1}{1-R_j^2}.$$
The factor $\text{VIF}_j=1/(1-R_j^2)$ is the <strong>variance inflation factor</strong>.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By FWL, $\hat\beta_j=\tilde x_j^\top y/\|\tilde x_j\|^2$ where $\tilde x_j$ is the residual of $x_j$ on the other columns (including $\mathbf 1$). This is linear in $y$ with weight vector $\tilde x_j/\|\tilde x_j\|^2$, so $\Var\hat\beta_j=\sigma^2/\|\tilde x_j\|^2$. Finally $\|\tilde x_j\|^2$ is the residual sum of squares of that auxiliary regression, which is $(1-R_j^2)S_{jj}$.</p>
</details>
<div class="warn" data-name="Collinearity is a property of the design, not a bug">
<p>When $R_j^2$ is near one, the data contain little variation in $x_j$ <em>independent of the others</em>, so they carry little information about its partial effect. Individual coefficients become unstable and their $t$-tests weak even when the joint $F$-test is overwhelming (see <a href="#/t/regression-inference">regression inference</a>). Predictions within the span of the data remain fine. Remedies are more informative data, a different question (a combination of coefficients that <em>is</em> well determined), or bias: <a href="#/t/ridge-regression">ridge regression</a>.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Prove that $\tr H=p$ without computing any eigenvalues.`,
        a: R`$\tr H=\tr\big(X(X^\top X)^{-1}X^\top\big)=\tr\big((X^\top X)^{-1}X^\top X\big)=\tr I_p=p$, using $\tr(AB)=\tr(BA)$.` },
      { q: R`You rescale one predictor from metres to centimetres and add a constant to another. Which of $\hat\beta$, $\hat y$, $H$, RSS change?`,
        a: R`Only $\hat\beta$ (the rescaled coefficient divides by 100, and the intercept adjusts for the shift). The column space $\mathcal C(X)$ is unchanged by invertible column operations that keep the intercept, so $H$, $\hat y$, the residuals and RSS are identical.` },
      { q: R`In a regression of wage on education and experience, the education coefficient is $0.08$. Using FWL, describe exactly which variation in education identifies this number.`,
        a: R`The variation in education left over after linearly regressing education on experience (and the intercept). $0.08$ is the least-squares slope of wage (equivalently, of wage residualized on experience) on that residual. People whose education is unusual <em>given their experience</em> drive the estimate.` },
      { q: R`Adding a column of pure noise to $X$ raised $R^2$ from $0.40$ to $0.41$. Is that surprising?`,
        a: R`No. $\mathcal C(X)$ grew, so the projection of $y$ can only get longer and RSS can only shrink; with one extra random direction, the expected drop in RSS is about $\text{RSS}/(n-p)$, i.e. roughly a $(1-R^2)/(n-p)$ increase in $R^2$. Adjusted $R^2$ or a formal criterion (<a href="#/t/model-selection">model selection</a>) is needed to penalize this.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'gauss-markov',
    domain: 'regression',
    order: 30,
    title: 'The Gauss–Markov Theorem',
    subtitle: 'Why least squares is the best linear unbiased estimator — and exactly how much that claim covers',
    level: 2,
    sources: { aos: '13.3', cb: '§11.3.2', other: 'Rao, Linear Statistical Inference (1973), §4a; Puntanen & Styan (1989) on OLS = GLS' },
    prereqs: ['ols-geometry', 'point-estimation'],
    related: ['weighted-gls', 'ridge-regression', 'rao-blackwell', 'decision-theory', 'fisher-information'],
    tags: ['BLUE', 'Loewner order', 'efficiency', 'Aitken', 'unbiasedness'],
    summary: R`Gauss–Markov says that under uncorrelated, equal-variance errors, the least-squares estimator has the smallest covariance matrix among all estimators that are linear in $y$ and unbiased for $\beta$. The proof is two lines once you write any competitor as OLS plus a perturbation that annihilates $X$. Its value lies as much in its limits as in its conclusion: it needs no normality, but it says nothing about biased estimators (ridge, James–Stein), nothing about nonlinear ones (robust regression under heavy tails), and fails as soon as the errors are heteroscedastic or correlated — where generalized least squares takes over.`,
    survey: {
      task: R`Characterize the optimal linear unbiased estimator of $\beta$`,
      assumptions: R`$\E y=X\beta$, $\Var y=\sigma^2I$, $X$ fixed, full column rank; no normality`,
      output: R`$\Var(\tilde\beta)-\Var(\hat\beta_{\text{OLS}})\succeq0$ for every linear unbiased $\tilde\beta$`,
      strengths: 'Distribution-free optimality; extends to any linear combination and to GLS (Aitken)',
      weaknesses: 'Silent about biased or nonlinear estimators; fails under heteroscedasticity or correlation',
      complexity: '—',
    },
    sections: [
      {
        title: 'The claim and the ordering it uses',
        html: R`
<p>Assume the <strong>Gauss–Markov conditions</strong>: $y=X\beta+\varepsilon$ with $X$ a fixed $n\times p$ matrix of rank $p$, $\E\varepsilon=0$, and $\Var\varepsilon=\sigma^2I_n$ — errors uncorrelated with a common variance. Nothing is assumed about the shape of the error distribution.</p>
<div class="def" data-name="Linear unbiased estimator; BLUE">
<p>An estimator $\tilde\beta$ is <em>linear</em> if $\tilde\beta=Cy$ for a fixed $p\times n$ matrix $C$, and <em>unbiased</em> if $\E\tilde\beta=\beta$ for every $\beta\in\R^p$. Since $\E Cy=CX\beta$, unbiasedness for all $\beta$ is equivalent to $CX=I_p$. It is <em>best</em> (BLUE) if its covariance is minimal among all linear unbiased estimators in the Loewner order: $A\preceq B$ means $B-A$ is positive semidefinite.</p>
</div>
<p>The Loewner order is the strongest sensible notion of "smaller covariance". If $\Var\tilde\beta-\Var\hat\beta\succeq0$ then for <em>every</em> vector $c$, $\Var(c^\top\tilde\beta)\ge\Var(c^\top\hat\beta)$: each coefficient, each contrast, each predicted mean $x_0^\top\beta$ is estimated at least as precisely by OLS. It also implies the scalar summaries $\tr$ (total MSE) and $\det$ (generalized variance) are ordered.</p>`,
      },
      {
        title: 'The theorem and its proof',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Gauss–Markov">
<p>Under the Gauss–Markov conditions, $\hat\beta=(X^\top X)^{-1}X^\top y$ is linear and unbiased with $\Var\hat\beta=\sigma^2(X^\top X)^{-1}$, and for any linear unbiased $\tilde\beta=Cy$,
$$\Var\tilde\beta-\Var\hat\beta=\sigma^2DD^\top\succeq0,\qquad D=C-(X^\top X)^{-1}X^\top .$$
Equality holds iff $D=0$, i.e. $\tilde\beta=\hat\beta$. Consequently, for every $c\in\R^p$, $c^\top\hat\beta$ is the unique minimum-variance linear unbiased estimator of $c^\top\beta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $A=(X^\top X)^{-1}X^\top$ so $\hat\beta=Ay$, with $AX=I$ and $\Var\hat\beta=\sigma^2AA^\top=\sigma^2(X^\top X)^{-1}$. Any linear unbiased $C$ has $CX=I$; write $C=A+D$. Then $DX=CX-AX=0$. Now
$$\Var(Cy)=\sigma^2(A+D)(A+D)^\top=\sigma^2\big(AA^\top+AD^\top+DA^\top+DD^\top\big).$$
The cross term vanishes: $DA^\top=DX(X^\top X)^{-1}=0$. So $\Var(Cy)=\Var\hat\beta+\sigma^2DD^\top$, and $DD^\top\succeq0$ since $v^\top DD^\top v=\|D^\top v\|^2\ge0$. Equality of covariances forces $\tr DD^\top=\|D\|_F^2=0$.</p>
</details>
<div class="note">
<p>Geometrically: an unbiased linear estimator of $c^\top\beta$ is $a^\top y$ with $X^\top a=c$. All such $a$ form an affine set; the variance is $\sigma^2\|a\|^2$, so we want the shortest $a$ in that set — the one orthogonal to the set's direction $\{d:X^\top d=0\}=\mathcal C(X)^\perp$. The shortest $a$ therefore lies in $\mathcal C(X)$, and the unique such $a$ is $X(X^\top X)^{-1}c$, which gives $c^\top\hat\beta$. Least squares wins because it uses no weight outside the column space; any weight in $\mathcal C(X)^\perp$ only adds noise.</p>
</div>
<p>The same argument with $\E\,\varepsilon^\top M\varepsilon=\sigma^2\tr M$ shows $\hat\sigma^2=\text{RSS}/(n-p)$ is unbiased. (It is <em>not</em> claimed to be best in any sense without further assumptions.)</p>`,
      },
      {
        title: 'Seeing the theorem: competing slope estimators',
        html: R`
<p>In simple regression, here are three linear unbiased estimators of the slope, each a weighted combination $\sum a_iY_i$ with $\sum a_i=0$ and $\sum a_ix_i=1$:</p>
<table>
<thead><tr><th>Estimator</th><th>Weights</th><th>Variance</th></tr></thead>
<tbody>
<tr><td>OLS</td><td>$a_i=(x_i-\bar x)/S_{xx}$</td><td>$\sigma^2/S_{xx}$</td></tr>
<tr><td>Endpoints</td><td>$(Y_{(n)}-Y_{(1)})/(x_{(n)}-x_{(1)})$, using the points with extreme $x$</td><td>$2\sigma^2/(x_{(n)}-x_{(1)})^2$</td></tr>
<tr><td>Grouping (Wald)</td><td>(mean $Y$ of upper half − mean of lower half) / (difference of mean $x$'s)</td><td>$\dfrac{4\sigma^2/n}{(\bar x_U-\bar x_L)^2}$ for even $n$</td></tr>
</tbody>
</table>
<div class="ex" data-name="Equally spaced design">
<p>Take $x_i=i$, $i=1,\dots,20$. Then $S_{xx}=\sum(i-10.5)^2=665$, so $\Var_{\text{OLS}}=\sigma^2/665\approx0.00150\,\sigma^2$. The endpoint estimator has $2\sigma^2/19^2\approx0.00554\,\sigma^2$, about 3.7 times larger. The grouping estimator has $\bar x_U-\bar x_L=10$ and variance $(4/20)\sigma^2/100=0.002\,\sigma^2$, about 1.33 times OLS — a respectable 75% efficiency, because grouping keeps most of the information in an evenly spread design.</p>
</div>
<p>The lab lets you race these estimators by simulation and then <em>break the assumptions</em>: under heteroscedastic errors a correctly weighted estimator beats OLS, and under heavy-tailed errors a nonlinear estimator (Theil–Sen, the median of pairwise slopes) beats it — neither contradicts the theorem, because each violates one of its premises.</p>`,
        lab: 'gauss-markov-race',
      },
      {
        title: 'What the theorem does not say',
        html: R`
<div class="warn" data-name="Biased estimators can have smaller MSE">
<p>Gauss–Markov restricts attention to unbiased estimators. Allowing a little bias can buy a large variance reduction: <a href="#/t/ridge-regression">ridge regression</a> beats OLS in total mean squared error for some $\lambda>0$ whenever $\sigma^2>0$. Under normal errors with $p\ge3$ and the prediction loss $\|X(\tilde\beta-\beta)\|^2$ — which reduces the problem to estimating a $p$-dimensional normal mean with spherical covariance — OLS is even <em>inadmissible</em> (Stein, 1956); James–Stein shrinkage dominates it (see <a href="#/t/decision-theory">decision theory</a>).</p>
</div>
<div class="warn" data-name="Nonlinear estimators can be far better">
<p>The class of competitors is only <em>linear</em> estimators. With Laplace errors, least absolute deviations has asymptotic variance $1/(4f(0)^2)=b^2$ per unit of design (for scale $b$), versus $2b^2$ for OLS — twice as efficient. With Cauchy-like tails OLS is not even consistent while the median-based estimators are. See <a href="#/t/robust-statistics">robust estimation</a>.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Under normality, OLS is UMVU">
<p>If in addition $\varepsilon\sim\N(0,\sigma^2I)$, then $c^\top\hat\beta$ has the smallest variance among <em>all</em> unbiased estimators of $c^\top\beta$, linear or not, and $\Var\hat\beta=\sigma^2(X^\top X)^{-1}$ equals the Cramér–Rao bound.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The normal linear model is a full-rank exponential family with natural sufficient statistic $(X^\top y,\;y^\top y)$, which is complete. $\hat\beta$ is an unbiased function of it, so by <a href="#/t/rao-blackwell">Lehmann–Scheffé</a> it is UMVU. Directly: the Fisher information for $\beta$ is $X^\top X/\sigma^2$, whose inverse is $\Var\hat\beta$.</p>
</details>
<p>So normality upgrades "best linear" to "best unbiased", and Gauss–Markov can be read as saying that, for linear estimators, second-moment assumptions suffice to get the conclusion that normality would give.</p>`,
      },
      {
        title: 'When the error covariance is not spherical',
        html: R`
<p>Suppose $\Var\varepsilon=\sigma^2\Omega$ with $\Omega$ known and positive definite. OLS is still linear and unbiased, but no longer best.</p>
<div class="thm" data-kind="Theorem" data-name="Aitken (1935)">
<p>Under $\E y=X\beta$, $\Var y=\sigma^2\Omega$, the BLUE of $\beta$ is the generalized least-squares estimator
$$\hat\beta_{\text{GLS}}=(X^\top\Omega^{-1}X)^{-1}X^\top\Omega^{-1}y,\qquad \Var\hat\beta_{\text{GLS}}=\sigma^2(X^\top\Omega^{-1}X)^{-1}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\Omega=LL^\top$ (Cholesky). Then $L^{-1}y=L^{-1}X\beta+L^{-1}\varepsilon$ with $\Var(L^{-1}\varepsilon)=\sigma^2I$, so the transformed model satisfies the Gauss–Markov conditions. Linear unbiased estimators of $\beta$ based on $y$ and on $L^{-1}y$ are the same class (since $L$ is invertible), and OLS on the transformed data is $\big((L^{-1}X)^\top L^{-1}X\big)^{-1}(L^{-1}X)^\top L^{-1}y=\hat\beta_{\text{GLS}}$.</p>
</details>
<p>There is an exact characterization of when you lose nothing by ignoring $\Omega$:</p>
<div class="thm" data-kind="Theorem" data-name="When OLS is still BLUE (Zyskind, Kruskal)">
<p>$\hat\beta_{\text{OLS}}=\hat\beta_{\text{GLS}}$ for every $y$ if and only if $\mathcal C(\Omega X)\subseteq\mathcal C(X)$, i.e. $\Omega X=XQ$ for some $p\times p$ matrix $Q$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The two estimators coincide for all $y$ iff $(X^\top X)^{-1}X^\top=(X^\top\Omega^{-1}X)^{-1}X^\top\Omega^{-1}$, iff the OLS residual maker satisfies $X^\top\Omega^{-1}M=0$, iff $\mathcal C(\Omega^{-1}X)\subseteq\mathcal C(X)$. Because $\Omega^{-1}$ maps $\mathcal C(X)$ into itself iff $\Omega$ does (both are invertible and $\mathcal C(X)$ is finite-dimensional), this is equivalent to $\mathcal C(\Omega X)\subseteq\mathcal C(X)$.</p>
</details>
<div class="ex" data-name="Equicorrelated errors">
<p>If $\Omega=(1-\rho)I+\rho\mathbf 1\mathbf 1^\top$ (exchangeable errors, as with a shared random effect) and the model contains an intercept, then $\Omega X=(1-\rho)X+\rho\mathbf 1(\mathbf 1^\top X)$, whose columns lie in $\mathcal C(X)$ because $\mathbf 1$ does. So OLS is still BLUE — but its <em>reported</em> standard errors, computed as $\hat\sigma^2(X^\top X)^{-1}$, are wrong. Being BLUE and having correct naive standard errors are separate questions; see <a href="#/t/weighted-gls">weighted & generalized least squares</a>.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`State precisely what "best" means in BLUE, and give one consequence for predicting a mean response.`,
        a: R`Minimal covariance in the Loewner order: $\Var\tilde\beta-\Var\hat\beta\succeq0$ for every linear unbiased $\tilde\beta$. Consequently for any $x_0$, $\Var(x_0^\top\tilde\beta)\ge\Var(x_0^\top\hat\beta)$, so OLS gives the most precise linear unbiased estimate of the mean response at every $x_0$ simultaneously.` },
      { q: R`In the proof, why does the cross term $DA^\top$ vanish?`,
        a: R`Unbiasedness of both $C=A+D$ and $A$ gives $DX=0$, and $A^\top=X(X^\top X)^{-1}$, so $DA^\top=DX(X^\top X)^{-1}=0$. Geometrically, the rows of $D$ lie in $\mathcal C(X)^\perp$ while the rows of $A$ lie in $\mathcal C(X)$.` },
      { q: R`Errors are i.i.d. Laplace with scale $b$. Does Gauss–Markov guarantee OLS is the best estimator of $\beta$? What is?`,
        a: R`It guarantees only that OLS is best among <em>linear</em> unbiased estimators. Least absolute deviations (median regression), which is nonlinear and is the MLE here, has half the asymptotic variance of OLS for Laplace errors.` },
      { q: R`Give a nontrivial error covariance $\Omega\ne I$ for which OLS is still BLUE.`,
        a: R`Exchangeable errors $\Omega=(1-\rho)I+\rho\mathbf 1\mathbf 1^\top$ when the model has an intercept: $\Omega X\subseteq\mathcal C(X)$, so OLS equals GLS (Zyskind's condition). The naive OLS standard errors are nonetheless incorrect.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'regression-inference',
    domain: 'regression',
    order: 40,
    title: 'Inference for Coefficients: t, F & Prediction Intervals',
    subtitle: 'Exact distribution theory for the normal linear model, and what survives without normality',
    level: 2,
    sources: { aos: '13.4', cb: '§11.3.4–11.3.5', other: 'Scheffé, The Analysis of Variance (1959), Ch. 2–3; Seber & Lee (2003), Ch. 5' },
    prereqs: ['gauss-markov', 'sampling-distributions'],
    related: ['confidence-intervals', 'hypothesis-testing', 'anova', 'bootstrap-regression', 'likelihood-ratio-tests', 'multiple-testing'],
    tags: ['t-test', 'F-test', 'confidence ellipsoid', 'prediction interval', 'Working–Hotelling'],
    summary: R`Under normal errors the linear model enjoys an exact distribution theory: $\hat\beta$ is multivariate normal around $\beta$, the residual sum of squares is an independent scaled $\chi^2_{n-p}$, and every pivot you need — $t$ for a single linear combination, $F$ for several at once — follows. This topic derives that theory from the projection geometry, then builds the tools: coefficient intervals and tests, joint confidence ellipsoids, intervals for the mean response, and prediction intervals for a new observation. The last distinction matters most in practice: a confidence band for the regression line shrinks to zero width as $n$ grows; a prediction band never does, and it is the only one of these procedures that depends on normality even asymptotically.`,
    survey: {
      task: R`Confidence intervals and tests for $c^\top\beta$ or $L\beta$; intervals for $\E(Y\mid x_0)$ and for a new $Y_0$`,
      assumptions: R`Normal linear model for exact results; asymptotically, independent errors with constant variance and $\max_ih_{ii}\to0$`,
      output: R`$t_{n-p}$ and $F_{q,n-p}$ pivots; ellipsoids; CI and PI bands`,
      strengths: 'Exact finite-sample coverage; elegant unified geometry; $t$, $F$ and LRT agree',
      weaknesses: 'PI needs normal errors at any $n$; heteroscedasticity breaks the SEs; invalid after data-driven model choice',
      complexity: R`$O(np^2)$`,
    },
    sections: [
      {
        title: 'The distribution theory of the normal linear model',
        html: R`
<p>Assume $y\sim\N(X\beta,\sigma^2I_n)$ with $X$ fixed and of rank $p$. Two facts about Gaussian vectors do all the work: a linear map of a Gaussian is Gaussian, and jointly Gaussian blocks are independent iff uncorrelated.</p>
<div class="thm" data-kind="Lemma" data-name="Quadratic forms in projections">
<p>If $z\sim\N(0,I_n)$ and $P$ is a symmetric idempotent matrix of rank $r$, then $z^\top Pz\sim\chi^2_r$. If $P_1,P_2$ are such projections with $P_1P_2=0$, then $z^\top P_1z$ and $z^\top P_2z$ are independent.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Diagonalize $P=Q\Lambda Q^\top$ with $Q$ orthogonal and $\Lambda=\diag(1,\dots,1,0,\dots,0)$ ($r$ ones). Then $w=Q^\top z\sim\N(0,I_n)$ by rotation invariance, and $z^\top Pz=\sum_{j\le r}w_j^2\sim\chi^2_r$. For the second claim, $P_1z$ and $P_2z$ are jointly Gaussian with $\Cov(P_1z,P_2z)=P_1P_2^\top=P_1P_2=0$, hence independent; and $z^\top P_kz=\|P_kz\|^2$.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="Sampling distribution of the normal linear model">
<p>$$\hat\beta\sim\N\big(\beta,\ \sigma^2(X^\top X)^{-1}\big),\qquad \frac{\text{RSS}}{\sigma^2}=\frac{(n-p)\hat\sigma^2}{\sigma^2}\sim\chi^2_{n-p},\qquad \hat\beta\indep\hat\sigma^2 .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\hat\beta=\beta+(X^\top X)^{-1}X^\top\varepsilon$ is an affine image of a Gaussian, with the stated mean and covariance. The residual is $e=M\varepsilon$, so $\text{RSS}/\sigma^2=z^\top Mz$ with $z=\varepsilon/\sigma\sim\N(0,I)$ and $M$ a projection of rank $n-p$; apply the lemma. Finally $\Cov\big((X^\top X)^{-1}X^\top\varepsilon,\ M\varepsilon\big)=\sigma^2(X^\top X)^{-1}X^\top M=0$ because $MX=0$; jointly Gaussian and uncorrelated means independent, and $\hat\sigma^2$ is a function of $e$ alone.</p>
</details>
<p>Every inferential procedure below is a corollary. The theorem also shows <em>why</em> they are exact: no plug-in approximation is involved, because $\hat\sigma^2$ enters only through a ratio whose law is known.</p>`,
      },
      {
        title: 't-intervals and t-tests for linear combinations',
        html: R`
<div class="thm" data-kind="Corollary" data-name="t pivot">
<p>For any fixed $c\in\R^p$,
$$T_c=\frac{c^\top\hat\beta-c^\top\beta}{\hat\sigma\sqrt{c^\top(X^\top X)^{-1}c}}\sim t_{n-p}.$$
Hence $c^\top\hat\beta\pm t_{n-p,\alpha/2}\,\hat\sigma\sqrt{c^\top(X^\top X)^{-1}c}$ is an exact $1-\alpha$ interval, and rejecting $H_0:c^\top\beta=m$ when $|T_c|>t_{n-p,\alpha/2}$ is an exact level-$\alpha$ test.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The numerator divided by $\sigma\sqrt{c^\top(X^\top X)^{-1}c}$ is $\N(0,1)$; $\hat\sigma^2/\sigma^2$ is an independent $\chi^2_{n-p}/(n-p)$. Their ratio is $t_{n-p}$ by definition, and the unknown $\sigma$ cancels.</p>
</details>
<p>Taking $c=e_j$ gives the coefficient table every package prints: $\hat\beta_j/\widehat{\se}(\hat\beta_j)$ with $\widehat{\se}(\hat\beta_j)=\hat\sigma\sqrt{[(X^\top X)^{-1}]_{jj}}$. Its null hypothesis is $\beta_j=0$ <em>with all other predictors in the model</em> — by <a href="#/t/ols-geometry">Frisch–Waugh–Lovell</a>, a test of whether the part of $x_j$ orthogonal to the others predicts $y$. Dropping or adding another predictor changes the hypothesis being tested, not just the estimate.</p>
<div class="ex" data-name="Difference of two coefficients">
<p>To test whether two predictors have equal effects, take $c=e_j-e_k$: the standard error is $\hat\sigma\sqrt{v_{jj}+v_{kk}-2v_{jk}}$ with $v=(X^\top X)^{-1}$. Comparing the two individual intervals by eye ignores the covariance $v_{jk}$, which is large and positive (making the difference <em>more</em> precise than it looks) or negative (less precise) depending on the correlation between the predictors.</p>
</div>`,
      },
      {
        title: 'The mean response versus a new observation',
        html: R`
<p>At a new covariate vector $x_0$ there are two different targets.</p>
<div class="cols">
<div><h4>Confidence interval for $\mu_0=x_0^\top\beta$</h4>$$x_0^\top\hat\beta\pm t_{n-p,\alpha/2}\,\hat\sigma\sqrt{x_0^\top(X^\top X)^{-1}x_0}$$<p>Uncertainty about where the regression surface is. Width $\to0$ as $n\to\infty$.</p></div>
<div><h4>Prediction interval for $Y_0=x_0^\top\beta+\varepsilon_0$</h4>$$x_0^\top\hat\beta\pm t_{n-p,\alpha/2}\,\hat\sigma\sqrt{1+x_0^\top(X^\top X)^{-1}x_0}$$<p>Also the irreducible noise of the new point. Width $\to2z_{\alpha/2}\sigma$, never zero.</p></div>
</div>
<p>The prediction interval follows because $Y_0-x_0^\top\hat\beta=\varepsilon_0-x_0^\top(\hat\beta-\beta)$ is a sum of independent Gaussians (the new error is independent of the training data) with variance $\sigma^2\big(1+x_0^\top(X^\top X)^{-1}x_0\big)$, and is independent of $\hat\sigma^2$. In simple regression $x_0^\top(X^\top X)^{-1}x_0=\tfrac1n+\tfrac{(x_0-\bar x)^2}{S_{xx}}$, which is why both bands flare out away from $\bar x$.</p>
<p>The pointwise band covers the true line at each fixed $x_0$ with probability $1-\alpha$, but not at all $x_0$ simultaneously. For a <strong>simultaneous</strong> band, replace $t_{n-p,\alpha/2}$ by $\sqrt{pF_{p,n-p;\alpha}}$:</p>
<div class="thm" data-kind="Theorem" data-name="Working–Hotelling–Scheffé band">
<p>$$\P\Big(\big|x^\top\hat\beta-x^\top\beta\big|\le\sqrt{pF_{p,n-p;\alpha}}\;\hat\sigma\sqrt{x^\top(X^\top X)^{-1}x}\ \text{ for all }x\in\R^p\Big)=1-\alpha .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $A=X^\top X$ and $d=\hat\beta-\beta$. By Cauchy–Schwarz in the inner product defined by $A^{-1}$, $(x^\top d)^2=\big((A^{-1/2}x)^\top(A^{1/2}d)\big)^2\le(x^\top A^{-1}x)(d^\top Ad)$, with equality for $x\propto Ad$. So $\sup_x (x^\top d)^2/(x^\top A^{-1}x)=d^\top Ad$, and the event "all inequalities hold" is exactly $d^\top Ad\le p\hat\sigma^2F_{p,n-p;\alpha}$. Since $d^\top Ad/\sigma^2\sim\chi^2_p$ independent of $\hat\sigma^2$, the ratio $d^\top Ad/(p\hat\sigma^2)\sim F_{p,n-p}$.</p>
</details>
<div class="warn" data-name="Prediction intervals depend on normality forever">
<p>The CI for $x_0^\top\beta$ is asymptotically valid under any error distribution with finite variance: $x_0^\top\hat\beta$ is a weighted average of many errors and the CLT applies. The PI is not: its width is dominated by the single new error $\varepsilon_0$, which is not averaged over anything. With skewed or heavy-tailed errors, a "95%" normal-theory PI can have badly wrong coverage in each tail no matter how large $n$ is. Empirical residual quantiles or <a href="#/t/bootstrap-regression">bootstrap</a> and conformal methods are the remedy.</p>
</div>`,
        lab: 'regression-inference-bands',
      },
      {
        title: 'F-tests for several coefficients',
        html: R`
<p>To test $q$ restrictions at once — say $H_0:\beta_{j_1}=\dots=\beta_{j_q}=0$ — compare the residual sum of squares of the full model with that of the <em>reduced</em> model that imposes the restrictions:
$$F=\frac{(\text{RSS}_0-\text{RSS})/q}{\text{RSS}/(n-p)}\ \overset{H_0}{\sim}\ F_{q,n-p}.$$
The proof, for general linear restrictions $L\beta=m$, is in <a href="#/t/anova">ANOVA and the general linear hypothesis</a>. Three special cases:</p>
<ul>
<li><strong>Overall regression test</strong>: all slopes zero, reduced model = intercept only, $F=\dfrac{\text{SSR}/(p-1)}{\text{RSS}/(n-p)}=\dfrac{R^2/(p-1)}{(1-R^2)/(n-p)}$.</li>
<li><strong>One coefficient</strong>: $q=1$ and $F=T^2$, since $F_{1,\nu}$ is the law of the square of a $t_\nu$.</li>
<li><strong>Nested models</strong>: any comparison of a model with a sub-model obtained by deleting or constraining terms.</li>
</ul>
<p>Inverting the $F$-test gives the <strong>joint confidence ellipsoid</strong>
$$\Big\{\beta:(\hat\beta-\beta)^\top X^\top X(\hat\beta-\beta)\le p\,\hat\sigma^2F_{p,n-p;\alpha}\Big\},$$
(or its $q$-dimensional analogue for a subvector, using the inverse of the corresponding block of $(X^\top X)^{-1}$). Its axes are the eigenvectors of $X^\top X$, and its length along each is inversely proportional to the singular value. Under collinearity the ellipse is a long thin cigar.</p>
<div class="warn" data-name="Joint significance without individual significance">
<p>When two predictors are highly correlated, $\hat\beta_1$ and $\hat\beta_2$ are strongly negatively correlated. The confidence ellipse is a thin diagonal cigar that may cross both axes (so neither individual $t$-test rejects) while lying well away from the origin (so the joint $F$-test rejects decisively). The data say "at least one of these matters" while being unable to say which. This is not a contradiction; individual $t$-tests and the $F$-test answer different questions. The reverse can also happen: a single significant $t$ can coexist with a non-significant overall $F$ when many useless predictors dilute the numerator.</p>
</div>`,
        lab: 'regression-inference-ellipse',
      },
      {
        title: 'Without normality: what remains valid',
        html: R`
<p>Drop normality but keep independent errors with mean zero and constant variance $\sigma^2$.</p>
<div class="thm" data-kind="Theorem" data-name="Asymptotic normality of least squares (Eicker, 1963; Huber, 1973)">
<p>If the errors are i.i.d. with finite variance and the maximum leverage $\max_ih_{ii}\to0$, then for every fixed $c$,
$$\frac{c^\top\hat\beta-c^\top\beta}{\sigma\sqrt{c^\top(X^\top X)^{-1}c}}\dto\N(0,1),$$
and since $\hat\sigma^2\pto\sigma^2$, the $t$ pivot is asymptotically standard normal. The $F$-statistic satisfies $qF\dto\chi^2_q$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$c^\top(\hat\beta-\beta)=\sum_ia_i\varepsilon_i$ with $a=X(X^\top X)^{-1}c$ and $\|a\|^2=c^\top(X^\top X)^{-1}c$. Lindeberg's condition for a weighted sum of i.i.d. finite-variance terms holds whenever $\max_ia_i^2/\|a\|^2\to0$ (split $\E[a_i^2\varepsilon_i^2\ind(|a_i\varepsilon_i|>\eta\|a\|)]\le a_i^2\,\E[\varepsilon^2\ind(|\varepsilon|>\eta\|a\|/\max_j|a_j|)]$ and use dominated convergence). By Cauchy–Schwarz, $a_i^2\le h_{ii}\,c^\top(X^\top X)^{-1}c$, so $\max_ih_{ii}\to0$ suffices.</p>
</details>
<p>The condition $\max_ih_{ii}\to0$ says no single observation controls the fit; a design with a few extreme points can violate it at any $n$. Under heteroscedasticity, the point estimate stays consistent but $\hat\sigma^2(X^\top X)^{-1}$ is the wrong variance — use <a href="#/t/weighted-gls">sandwich standard errors</a>. Under dependence (clustered or serially correlated errors), the effective sample size is smaller than $n$ and naive intervals are too narrow.</p>
<div class="warn" data-name="Inference after model selection">
<p>All of the above treats the model — the columns of $X$ — as fixed in advance. If predictors were chosen because they looked significant in the same data, the reported $p$-values are too small and the intervals too narrow. See <a href="#/t/model-selection">model selection</a> for the problem and its partial remedies.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why are $\hat\beta$ and $\hat\sigma^2$ independent in the normal linear model? Would they be independent with non-normal errors?`,
        a: R`$\hat\beta-\beta$ is a linear function of $H\varepsilon$ and $\hat\sigma^2$ is a function of $M\varepsilon$; the two are uncorrelated because $HM=0$, and for jointly Gaussian vectors uncorrelated implies independent. With non-normal errors they are still uncorrelated but generally dependent (e.g. skewed errors make $\hat\beta$ and RSS correlated in higher moments).` },
      { q: R`As $n\to\infty$ with the design spreading normally, what happens to the widths of the 95% CI for the mean response and the 95% PI at a fixed $x_0$?`,
        a: R`The CI width is proportional to $\hat\sigma\sqrt{x_0^\top(X^\top X)^{-1}x_0}=O(n^{-1/2})\to0$. The PI width tends to $2\cdot1.96\,\sigma$ because of the irreducible $1$ under the square root.` },
      { q: R`Two predictors with correlation $0.98$ both have $|t|<1$, but the overall $F$-test has $p<10^{-6}$. Explain using the confidence ellipse.`,
        a: R`High positive correlation between predictors makes $\hat\beta_1,\hat\beta_2$ strongly negatively correlated, so the joint confidence region is a thin ellipse along the direction $\beta_1+\beta_2=\text{const}$. It can straddle both axes (each coefficient individually consistent with zero) while excluding the origin, because the sum $\beta_1+\beta_2$ is precisely estimated and clearly nonzero.` },
      { q: R`Show that the $F$-test of a single restriction $\beta_j=0$ is identical to the two-sided $t$-test.`,
        a: R`With $L=e_j^\top$, the general linear hypothesis numerator is $\hat\beta_j^2/[(X^\top X)^{-1}]_{jj}$, so $F=\hat\beta_j^2/(\hat\sigma^2[(X^\top X)^{-1}]_{jj})=T_j^2$. Since $T_j\sim t_{n-p}$ implies $T_j^2\sim F_{1,n-p}$ and $\P(T^2>t_{\alpha/2}^2)=\P(|T|>t_{\alpha/2})$, the tests reject on the same samples.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'anova',
    domain: 'regression',
    order: 50,
    title: 'ANOVA & the General Linear Hypothesis',
    subtitle: 'Decomposing variation into orthogonal pieces, and the one F-test behind every nested comparison',
    level: 2,
    sources: { cb: '§11.2', other: 'Scheffé (1959); Searle, Linear Models (1971), Ch. 5–7' },
    prereqs: ['regression-inference'],
    related: ['multiple-testing', 'mixed-models', 'rank-tests', 'permutation-tests', 'likelihood-ratio-tests'],
    tags: ['one-way ANOVA', 'F-test', 'Cochran', 'contrasts', 'Tukey', 'Scheffé'],
    summary: R`Analysis of variance is regression with categorical predictors, read through the geometry of nested subspaces. The total variation around the grand mean splits orthogonally into variation between group means and variation within groups; the ratio of their mean squares is an $F$ statistic whose null distribution is exact under normal errors. The same idea in full generality is the <em>general linear hypothesis</em> $L\beta=m$: fit the model with and without the restriction, and the increase in residual sum of squares, scaled by $\hat\sigma^2$, is $F$-distributed. Every $t$-test, nested-model comparison, and ANOVA table is a special case, and the $F$-test is also the likelihood-ratio test.`,
    survey: {
      task: R`Test equality of group means; test any linear hypothesis $L\beta=m$ in a linear model`,
      assumptions: R`Normal, independent, homoscedastic errors (exact); balanced designs make the test robust`,
      output: R`ANOVA table (SS, df, MS), $F_{q,n-p}$ statistic and $p$-value, contrast intervals`,
      strengths: 'Exact; unified (t, nested F, LRT coincide); orthogonal decomposition in balanced designs',
      weaknesses: 'Global test says nothing about which groups differ; unequal variances with unequal n distort it',
      complexity: R`$O(np^2)$`,
    },
    sections: [
      {
        title: 'ANOVA is a linear model',
        html: R`
<p>Observe $y_{ij}$, $j=1,\dots,n_i$, in groups $i=1,\dots,k$, with $N=\sum n_i$. The <strong>one-way model</strong> is
$$y_{ij}=\mu_i+\varepsilon_{ij}=\mu+\alpha_i+\varepsilon_{ij},\qquad \varepsilon_{ij}\iid\N(0,\sigma^2).$$
In the cell-means form the design matrix is the $N\times k$ indicator matrix $X$ with $X_{(ij),l}=\ind(i=l)$, so $X^\top X=\diag(n_1,\dots,n_k)$ and least squares gives $\hat\mu_i=\bar y_{i\cdot}$, the group means. The effects form $\mu+\alpha_i$ has $k+1$ parameters and only rank $k$: it is <em>overparameterized</em>.</p>
<div class="def" data-name="Estimable function">
<p>In a rank-deficient model, $c^\top\beta$ is <em>estimable</em> if it has a linear unbiased estimator, which happens iff $c\in\mathcal C(X^\top)$ (the row space of $X$). Estimable functions have the same least-squares estimate whichever solution of the normal equations is used.</p>
</div>
<p>In the effects model, $\mu+\alpha_i$ is estimable but $\alpha_i$ alone is not; every <strong>contrast</strong> $\sum c_i\alpha_i$ with $\sum c_i=0$ is estimable (it equals $\sum c_i\mu_i$). Constraints such as $\sum_i\alpha_i=0$, $\sum n_i\alpha_i=0$, or $\alpha_1=0$ (reference coding) pick a particular solution; they change the reported $\hat\alpha_i$ but not the fitted values, residuals, contrasts, or $F$-tests.</p>`,
      },
      {
        title: 'The variance decomposition and the F-test',
        html: R`
<div class="thm" data-kind="Proposition" data-name="One-way decomposition">
<p>With $\bar y_{\cdot\cdot}$ the grand mean,
$$\underbrace{\sum_{i,j}(y_{ij}-\bar y_{\cdot\cdot})^2}_{\text{SST},\ N-1\ \text{df}}=\underbrace{\sum_in_i(\bar y_{i\cdot}-\bar y_{\cdot\cdot})^2}_{\text{SSB},\ k-1\ \text{df}}+\underbrace{\sum_{i,j}(y_{ij}-\bar y_{i\cdot})^2}_{\text{SSW},\ N-k\ \text{df}} .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $y_{ij}-\bar y_{\cdot\cdot}=(\bar y_{i\cdot}-\bar y_{\cdot\cdot})+(y_{ij}-\bar y_{i\cdot})$ and square. The cross term is $2\sum_i(\bar y_{i\cdot}-\bar y_{\cdot\cdot})\sum_j(y_{ij}-\bar y_{i\cdot})=0$ since deviations from a group mean sum to zero within the group. Geometrically: let $V_0=\operatorname{span}(\mathbf 1)\subset V=\mathcal C(X)$. Then $y-P_{V_0}y=(P_V-P_{V_0})y+(I-P_V)y$, and $P_V-P_{V_0}$, $I-P_V$ are projections onto orthogonal subspaces of dimensions $k-1$ and $N-k$.</p>
</details>
<p>Let $\text{MSB}=\text{SSB}/(k-1)$ and $\text{MSW}=\text{SSW}/(N-k)=\hat\sigma^2$. Their expectations reveal the logic of the test:
$$\E\,\text{MSW}=\sigma^2,\qquad \E\,\text{MSB}=\sigma^2+\frac{1}{k-1}\sum_in_i(\mu_i-\bar\mu)^2,\qquad \bar\mu=\frac1N\sum_in_i\mu_i .$$
Under $H_0:\mu_1=\dots=\mu_k$ both estimate $\sigma^2$; otherwise MSB is inflated.</p>
<div class="thm" data-kind="Theorem" data-name="One-way ANOVA F-test">
<p>$F=\text{MSB}/\text{MSW}$ has the noncentral $F_{k-1,N-k}(\lambda)$ distribution with noncentrality $\lambda=\sum_in_i(\mu_i-\bar\mu)^2/\sigma^2$. Under $H_0$, $\lambda=0$ and $F\sim F_{k-1,N-k}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the lemma on quadratic forms (applied to $y/\sigma$, with the noncentral extension for nonzero mean), $\text{SSB}/\sigma^2=\|(P_V-P_{V_0})y\|^2/\sigma^2$ is noncentral $\chi^2_{k-1}$ with noncentrality $\|(P_V-P_{V_0})\mu\|^2/\sigma^2=\lambda$, and $\text{SSW}/\sigma^2=\|(I-P_V)y\|^2/\sigma^2\sim\chi^2_{N-k}$ (central, because $(I-P_V)\mu=0$). The two projections are orthogonal, so the quadratic forms are independent.</p>
</details>
<p>The noncentrality parameter gives the <strong>power</strong>: $\P(F_{k-1,N-k}(\lambda)>F_{k-1,N-k;\alpha})$. Power depends on the means only through $\lambda$, which grows linearly in the group sizes — the basis of sample-size calculations.</p>`,
        lab: 'anova-decomposition',
      },
      {
        title: 'The general linear hypothesis',
        html: R`
<p>Return to a full-rank model $y\sim\N(X\beta,\sigma^2I)$ with $p$ columns, and a hypothesis $H_0:L\beta=m$ where $L$ is $q\times p$ of rank $q$. Let $V=(X^\top X)^{-1}$.</p>
<div class="thm" data-kind="Theorem" data-name="F-test of the general linear hypothesis">
<p>The restricted least-squares estimate (minimizing RSS subject to $L\beta=m$) is
$$\hat\beta_0=\hat\beta-VL^\top(LVL^\top)^{-1}(L\hat\beta-m),$$
and
$$\text{RSS}_0-\text{RSS}=(L\hat\beta-m)^\top(LVL^\top)^{-1}(L\hat\beta-m).$$
Consequently
$$F=\frac{(L\hat\beta-m)^\top(LVL^\top)^{-1}(L\hat\beta-m)/q}{\hat\sigma^2}\sim F_{q,n-p}(\delta),\qquad \delta=\frac{(L\beta-m)^\top(LVL^\top)^{-1}(L\beta-m)}{\sigma^2},$$
central under $H_0$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>From the normal-equations proof, for any $b$: $\text{RSS}(b)=\text{RSS}+(b-\hat\beta)^\top X^\top X(b-\hat\beta)$. So we minimize the quadratic $(b-\hat\beta)^\top V^{-1}(b-\hat\beta)$ subject to $Lb=m$. The Lagrangian stationarity condition is $2V^{-1}(b-\hat\beta)=L^\top\gamma$, i.e. $b=\hat\beta+\tfrac12VL^\top\gamma$; imposing $Lb=m$ gives $\tfrac12\gamma=-(LVL^\top)^{-1}(L\hat\beta-m)$, which yields $\hat\beta_0$. Substituting, $(\hat\beta_0-\hat\beta)^\top V^{-1}(\hat\beta_0-\hat\beta)=(L\hat\beta-m)^\top(LVL^\top)^{-1}LVV^{-1}VL^\top(LVL^\top)^{-1}(L\hat\beta-m)$, which simplifies to the stated form.</p>
<p>Distribution: $L\hat\beta-m\sim\N(L\beta-m,\sigma^2LVL^\top)$, so the numerator quadratic form divided by $\sigma^2$ is (noncentral) $\chi^2_q$ — whiten with $(LVL^\top)^{-1/2}$. It is a function of $\hat\beta$, hence independent of $\hat\sigma^2$, and $(n-p)\hat\sigma^2/\sigma^2\sim\chi^2_{n-p}$.</p>
</details>
<p>The theorem unifies everything: $q=1$ gives the $t$-test squared; $L$ selecting a block of coefficients gives the nested-model $F$; the one-way ANOVA is $L$ encoding $\mu_1-\mu_k=\dots=\mu_{k-1}-\mu_k=0$.</p>
<div class="thm" data-kind="Proposition" data-name="The F-test is the likelihood-ratio test">
<p>The generalized likelihood ratio for $H_0:L\beta=m$ (with $\sigma^2$ unknown) is $\Lambda=(\text{RSS}/\text{RSS}_0)^{n/2}$, a decreasing function of $F$: $\Lambda^{-2/n}=1+\tfrac{q}{n-p}F$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Maximizing the Gaussian likelihood over $\sigma^2$ for given $\beta$ gives $\sigma^2=\text{RSS}(\beta)/n$ and maximized value proportional to $\text{RSS}(\beta)^{-n/2}$. Maximizing over $\beta$ unrestricted and restricted gives the ratio. Then $\text{RSS}_0/\text{RSS}=1+(\text{RSS}_0-\text{RSS})/\text{RSS}=1+qF/(n-p)$.</p>
</details>
<p>So the exact $F$-test is the finite-sample version of the $\chi^2_q$ asymptotics of <a href="#/t/likelihood-ratio-tests">Wilks' theorem</a>, with $qF\dto\chi^2_q$ as $n-p\to\infty$.</p>`,
      },
      {
        title: "Cochran's theorem and multi-way layouts",
        html: R`
<div class="thm" data-kind="Theorem" data-name="Cochran (1934)">
<p>Let $y\sim\N(\mu,\sigma^2I_n)$ and $I_n=A_1+\dots+A_K$ with each $A_k$ symmetric of rank $r_k$. If $\sum_kr_k=n$, then the $A_k$ are mutually orthogonal projections, and the quadratic forms $y^\top A_ky/\sigma^2$ are independent noncentral $\chi^2_{r_k}$ with noncentralities $\mu^\top A_k\mu/\sigma^2$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The key linear-algebra step: $\mathcal C(A_1)+\dots+\mathcal C(A_K)=\R^n$ (since the $A_k$ sum to $I$) and the dimensions add to $n$, so the sum is direct. For $v\in\mathcal C(A_l)$ we have $v=\sum_kA_kv$ with $A_kv\in\mathcal C(A_k)$; uniqueness of the direct-sum decomposition forces $A_kv=0$ for $k\ne l$ and $A_lv=v$. Hence $A_kA_l=0$ for $k\ne l$ and $A_l^2=A_l$. Symmetric idempotent, mutually annihilating matrices project onto orthogonal subspaces; apply the quadratic-forms lemma.</p>
</details>
<p>Cochran's theorem licenses every ANOVA table in which sums of squares add up and degrees of freedom add up. In a <strong>balanced two-way layout</strong> $y_{ijl}=\mu+\alpha_i+\gamma_j+(\alpha\gamma)_{ij}+\varepsilon_{ijl}$ with $m$ replicates per cell, the subspaces for main effect A, main effect B, interaction, and error are mutually orthogonal, so
$$\text{SST}=\text{SS}_A+\text{SS}_B+\text{SS}_{AB}+\text{SSE},\qquad (abm-1)=(a-1)+(b-1)+(a-1)(b-1)+ab(m-1),$$
and each effect has an unambiguous $F$-test.</p>
<div class="warn" data-name="Unbalanced designs: the order of terms matters">
<p>With unequal cell counts, the A and B subspaces (after removing the mean) are no longer orthogonal, and "the sum of squares for A" is ambiguous. <em>Sequential</em> (Type I) SS test A ignoring B, then B adjusting for A; <em>Type II</em> tests each main effect adjusting for the other but not the interaction; <em>Type III</em> tests each term adjusting for all others under sum-to-zero coding. They test different hypotheses about cell means. Each is a particular general linear hypothesis — write down $L$ to know which question you are asking.</p>
</div>`,
      },
      {
        title: 'After the F-test: contrasts and multiple comparisons',
        html: R`
<p>A significant global $F$ says the means are not all equal; it does not say which differ. For a single pre-specified contrast $\psi=\sum c_i\mu_i$ ($\sum c_i=0$), the interval $\sum c_i\bar y_{i\cdot}\pm t_{N-k,\alpha/2}\,\hat\sigma\sqrt{\sum c_i^2/n_i}$ is exact. For many comparisons the family-wise error rate must be controlled (see <a href="#/t/multiple-testing">multiple testing</a>):</p>
<table>
<thead><tr><th>Method</th><th>Family</th><th>Critical multiplier</th></tr></thead>
<tbody>
<tr><td>Bonferroni</td><td>$m$ pre-specified contrasts</td><td>$t_{N-k,\alpha/(2m)}$</td></tr>
<tr><td>Tukey HSD</td><td>all $\binom k2$ pairwise differences</td><td>$q_{k,N-k;\alpha}/\sqrt2$ (studentized range); exact for balanced designs</td></tr>
<tr><td>Scheffé</td><td>all contrasts, even data-suggested ones</td><td>$\sqrt{(k-1)F_{k-1,N-k;\alpha}}$</td></tr>
</tbody>
</table>
<p>Scheffé's method is the Working–Hotelling argument restricted to the $(k-1)$-dimensional space of contrasts: the supremum over all contrasts of the squared $t$-statistic is $(k-1)F$. Hence the global $F$-test rejects at level $\alpha$ <em>if and only if</em> at least one contrast is Scheffé-significant — a pleasing consistency between the omnibus test and its follow-ups.</p>
<div class="ex" data-name="Which multiplier to use">
<p>With $k=5$ groups, $N-k=45$, $\alpha=0.05$: the unadjusted $t$ multiplier is about $2.01$; Tukey's is $q_{5,45;0.05}/\sqrt2\approx4.02/1.414\approx2.84$; Bonferroni for all 10 pairs is $t_{45,0.0025}\approx2.95$; Scheffé is $\sqrt{4F_{4,45;0.05}}\approx\sqrt{4\times2.58}\approx3.21$. Tukey is shortest for pairwise comparisons; Scheffé pays for its much larger family.</p>
</div>`,
      },
      {
        title: 'Robustness and alternatives',
        html: R`
<ul>
<li><strong>Non-normality.</strong> The $F$-test is fairly robust in level when group sizes are moderate, because group means are approximately normal by the CLT and MSW is consistent. Power suffers with heavy tails; <a href="#/t/rank-tests">Kruskal–Wallis</a> or a <a href="#/t/permutation-tests">permutation test</a> of the $F$ statistic is safer.</li>
<li><strong>Unequal variances.</strong> With equal $n_i$, the level is only mildly affected. With unequal $n_i$ it can be badly wrong: if the smallest groups have the largest variances, MSW underestimates the relevant noise and the test is liberal (and conservative in the reverse pairing). Welch's ANOVA weights groups by $n_i/s_i^2$ and uses approximate degrees of freedom.</li>
<li><strong>Dependence.</strong> Repeated measurements on the same units violate independence; use repeated-measures or <a href="#/t/mixed-models">mixed-effects models</a>.</li>
<li><strong>Random factors.</strong> If the groups are a sample from a population of groups, the question becomes "is the between-group variance $\tau^2$ zero?" The same $F$ statistic tests it, with $\E\,\text{MSB}=\sigma^2+n\tau^2$ in a balanced design — the starting point for <a href="#/t/mixed-models">variance components</a>.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`In a one-way layout, why are $\alpha_1,\dots,\alpha_k$ not estimable while $\alpha_1-\alpha_2$ is?`,
        a: R`The row space of $X$ in the effects parameterization $(\mu,\alpha_1,\dots,\alpha_k)$ is spanned by vectors $(1,e_i)$. $\alpha_1$ corresponds to $(0,e_1)$, not in that span, so adding a constant to all $\alpha_i$ and subtracting it from $\mu$ leaves the mean of the data unchanged. $\alpha_1-\alpha_2$ corresponds to $(1,e_1)-(1,e_2)$, which is in the span: it equals $\mu_1-\mu_2$.` },
      { q: R`Derive $\E\,\text{MSB}$ when $k=2$, $n_1=n_2=n$.`,
        a: R`$\text{SSB}=n\big[(\bar y_1-\bar y)^2+(\bar y_2-\bar y)^2\big]=\tfrac n2(\bar y_1-\bar y_2)^2$. $\E(\bar y_1-\bar y_2)^2=2\sigma^2/n+(\mu_1-\mu_2)^2$. So $\E\,\text{MSB}=\E\,\text{SSB}=\sigma^2+\tfrac n2(\mu_1-\mu_2)^2$, matching $\sigma^2+\sum n_i(\mu_i-\bar\mu)^2/(k-1)$.` },
      { q: R`Show that the generalized likelihood ratio test of $L\beta=m$ rejects for large $F$.`,
        a: R`Profiling out $\sigma^2$ gives maximized likelihood $\propto\text{RSS}(\beta)^{-n/2}$, so $\Lambda=(\text{RSS}/\text{RSS}_0)^{n/2}$, and $\text{RSS}_0/\text{RSS}=1+qF/(n-p)$. $\Lambda$ is decreasing in $F$, so "reject for small $\Lambda$" is "reject for large $F$", and the exact $F_{q,n-p}$ null distribution calibrates it.` },
      { q: R`After seeing the data, you notice groups 1 and 2 look higher than groups 3–5 and want to test $\tfrac12(\mu_1+\mu_2)-\tfrac13(\mu_3+\mu_4+\mu_5)$. Which interval procedure keeps the error rate honest?`,
        a: R`Scheffé's: it controls the family-wise error rate over <em>all</em> contrasts simultaneously, including ones suggested by the data. Bonferroni and Tukey are valid only for families fixed in advance.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'regression-diagnostics',
    domain: 'regression',
    order: 60,
    title: 'Diagnostics: Residuals, Leverage & Influence',
    subtitle: 'Leave-one-out algebra, Cook’s distance, and the plots that reveal a wrong model',
    level: 2,
    sources: { aos: '13.5', other: 'Cook & Weisberg, Residuals and Influence in Regression (1982); Belsley, Kuh & Welsch (1980)' },
    prereqs: ['ols-geometry', 'regression-inference'],
    related: ['robust-statistics', 'weighted-gls', 'bootstrap-regression', 'cross-validation', 'model-selection'],
    tags: ['residuals', 'leverage', "Cook's distance", 'Sherman–Morrison', 'studentized residuals'],
    summary: R`Every inferential statement about a regression rests on assumptions that the fitted model can partly check through its residuals. But residuals are not the errors: they are correlated, have unequal variances, and are shrunk most at the points that pull hardest on the fit. This topic develops the exact algebra — leverage $h_{ii}$, standardized and studentized residuals, and the leave-one-out identities obtained from the Sherman–Morrison formula — that lets you compute what would happen if each observation were deleted without refitting $n$ times. From it come Cook's distance, DFFITS, the PRESS statistic, and the graphical diagnostics that reveal nonlinearity, heteroscedasticity, and heavy tails.`,
    survey: {
      task: 'Check model assumptions; find outliers, high-leverage and influential observations',
      assumptions: R`A fitted linear model; normal errors for exact null distributions of studentized residuals`,
      output: R`$h_{ii}$, $r_i$, $t_i$, Cook's $D_i$, DFFITS, residual plots, QQ plots`,
      strengths: 'Closed-form leave-one-out quantities at the cost of one fit; interpretable plots',
      weaknesses: 'Single-case deletion misses masking by groups of outliers; thresholds are heuristics',
      complexity: R`$O(np^2)$ for all $n$ deletion diagnostics`,
    },
    sections: [
      {
        title: 'Residuals are not errors',
        html: R`
<p>Under the model, $e=My=M\varepsilon$ with $M=I-H$, so
$$\E e=0,\qquad \Var e=\sigma^2(I-H),\qquad \Var e_i=\sigma^2(1-h_{ii}),\qquad \Cov(e_i,e_j)=-\sigma^2h_{ij}\ (i\ne j).$$
The residuals are correlated (they satisfy $p$ linear constraints $X^\top e=0$) and heteroscedastic even when the errors are not: high-leverage points have <em>small</em> residual variance, because they drag the fit toward themselves. Raw residuals are therefore poorly suited to comparing observations.</p>
<div class="def" data-name="Standardized and studentized residuals">
<p>The <em>(internally) standardized</em> residual is $r_i=\dfrac{e_i}{\hat\sigma\sqrt{1-h_{ii}}}$. The <em>(externally) studentized</em> residual uses the variance estimate from the fit without case $i$:
$$t_i=\frac{e_i}{\hat\sigma_{(i)}\sqrt{1-h_{ii}}},\qquad (n-p-1)\,\hat\sigma^2_{(i)}=(n-p)\hat\sigma^2-\frac{e_i^2}{1-h_{ii}} .$$</p>
</div>
<p>Under the normal model $t_i\sim t_{n-p-1}$ exactly, because $e_i$ is then independent of $\hat\sigma_{(i)}$ (shown below via the deletion formulas). The internally standardized $r_i$ does not have a $t$ law — its square is bounded: $r_i^2\le n-p$ — and the two are related by $t_i=r_i\sqrt{(n-p-1)/(n-p-r_i^2)}$, so $t_i$ is the more sensitive outlier detector. Testing the most extreme $t_i$ requires a Bonferroni correction over $n$ cases.</p>
<p>Plot residuals against <strong>fitted values</strong>, never against $y$: $\Cov(e,\hat y)=\sigma^2MH=0$, while $\Cov(e,y)=\sigma^2M$, so a plot of $e$ against $y$ shows a spurious positive slope even when the model is correct.</p>`,
      },
      {
        title: 'Leverage',
        html: R`
<p>The leverage $h_{ii}=x_i^\top(X^\top X)^{-1}x_i$ is the $i$-th diagonal of the hat matrix. Since $\hat y_i=\sum_jh_{ij}y_j$, it is the weight observation $i$ puts on its own fitted value: $h_{ii}=\partial\hat y_i/\partial y_i$.</p>
<div class="thm" data-kind="Proposition" data-name="Properties of leverage">
<ol>
<li>$0\le h_{ii}\le1$ and $\sum_ih_{ii}=p$, so the average leverage is $p/n$. If the model has an intercept, $h_{ii}\ge1/n$.</li>
<li>$h_{ii}=1$ iff the fit passes exactly through $y_i$ for every $y$ (the point determines its own parameter).</li>
<li>With an intercept, $h_{ii}=\dfrac1n+\dfrac{1}{n-1}\,(x_i^\circ-\bar x^\circ)^\top S^{-1}(x_i^\circ-\bar x^\circ)$, where $x^\circ$ are the non-constant predictors and $S$ their sample covariance: leverage is a Mahalanobis distance from the centroid of the design. In simple regression, $h_{ii}=\tfrac1n+\tfrac{(x_i-\bar x)^2}{S_{xx}}$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) $H=H^2=HH^\top$ gives $h_{ii}=\sum_jh_{ij}^2\ge h_{ii}^2$, so $0\le h_{ii}\le1$; the trace is $p$. With an intercept, $H-\tfrac1n\mathbf 1\mathbf 1^\top$ is the projection onto the orthogonal complement of $\mathbf 1$ inside $\mathcal C(X)$, which has non-negative diagonal. (2) $h_{ii}=1$ forces $h_{ij}=0$ for $j\ne i$, so $\hat y_i=y_i$. (3) Reparameterize by centering the predictors (this does not change $H$); the centered design $[\mathbf 1\;X_c]$ has block-diagonal Gram matrix $\diag(n,X_c^\top X_c)$ and $X_c^\top X_c=(n-1)S$.</p>
</details>
<p>A common flag is $h_{ii}>2p/n$. High leverage is <em>potential</em> influence: a high-leverage point lying on the trend of the others is harmless (indeed helpful — it is what makes $S_{xx}$ large); one lying off the trend can rotate the fit. Leverage depends only on $X$, never on $y$.</p>`,
      },
      {
        title: 'Deleting one case without refitting',
        html: R`
<p>Let $\hat\beta_{(i)}$ be the estimate with case $i$ removed. Refitting $n$ times is unnecessary.</p>
<div class="thm" data-kind="Lemma" data-name="Sherman–Morrison">
<p>If $A$ is invertible and $1-v^\top A^{-1}v\ne0$, then $(A-vv^\top)^{-1}=A^{-1}+\dfrac{A^{-1}vv^\top A^{-1}}{1-v^\top A^{-1}v}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Multiply: $(A-vv^\top)\big(A^{-1}+\tfrac{A^{-1}vv^\top A^{-1}}{1-v^\top A^{-1}v}\big)=I-vv^\top A^{-1}+\tfrac{vv^\top A^{-1}-v(v^\top A^{-1}v)v^\top A^{-1}}{1-v^\top A^{-1}v}=I-vv^\top A^{-1}+vv^\top A^{-1}=I$.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="Case-deletion identities">
<p>With $A=X^\top X$ and $h_{ii}<1$:
$$\hat\beta-\hat\beta_{(i)}=\frac{A^{-1}x_i\,e_i}{1-h_{ii}},\qquad y_i-x_i^\top\hat\beta_{(i)}=\frac{e_i}{1-h_{ii}} .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Removing case $i$ replaces $A$ by $A-x_ix_i^\top$ and $X^\top y$ by $X^\top y-x_iy_i$. With $h=h_{ii}=x_i^\top A^{-1}x_i$ and Sherman–Morrison,
$$\hat\beta_{(i)}=\Big(A^{-1}+\frac{A^{-1}x_ix_i^\top A^{-1}}{1-h}\Big)(X^\top y-x_iy_i)=\hat\beta+\frac{A^{-1}x_i}{1-h}\Big(x_i^\top\hat\beta-(1-h)y_i-hy_i\Big)=\hat\beta-\frac{A^{-1}x_i\,e_i}{1-h}.$$
Then $y_i-x_i^\top\hat\beta_{(i)}=y_i-\hat y_i+\tfrac{h\,e_i}{1-h}=\tfrac{e_i}{1-h}$.</p>
</details>
<p>The <strong>deleted (predicted) residual</strong> $e_i/(1-h_{ii})$ is the error made predicting $y_i$ from the other $n-1$ points. Summing their squares gives the PRESS statistic, i.e. exact leave-one-out <a href="#/t/cross-validation">cross-validation</a> from a single fit — used in <a href="#/t/model-selection">model selection</a>. The identity also explains $t_i\sim t_{n-p-1}$: the deleted residual is a function of $y_i$ and $\hat\beta_{(i)}$, which are independent of the residuals of the reduced fit that determine $\hat\sigma_{(i)}$.</p>`,
      },
      {
        title: "Influence: Cook's distance and DFFITS",
        html: R`
<div class="def" data-name="Cook's distance">
<p>$$D_i=\frac{(\hat\beta-\hat\beta_{(i)})^\top X^\top X(\hat\beta-\hat\beta_{(i)})}{p\,\hat\sigma^2}=\frac{\|\hat y-\hat y_{(i)}\|^2}{p\,\hat\sigma^2},$$
the squared change in all fitted values from deleting case $i$, in units of the confidence-ellipsoid metric.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Closed form">
<p>$$D_i=\frac{r_i^2}{p}\cdot\frac{h_{ii}}{1-h_{ii}} .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the deletion identity, $(\hat\beta-\hat\beta_{(i)})^\top A(\hat\beta-\hat\beta_{(i)})=\frac{e_i^2}{(1-h_{ii})^2}x_i^\top A^{-1}AA^{-1}x_i=\frac{e_i^2h_{ii}}{(1-h_{ii})^2}$. Divide by $p\hat\sigma^2$ and use $r_i^2=e_i^2/(\hat\sigma^2(1-h_{ii}))$.</p>
</details>
<p>The formula is the moral of the topic: <strong>influence = outlyingness × leverage</strong>. A large residual at low leverage moves the fit little; a high-leverage point that agrees with the trend moves it little; it takes both. Because $D_i$ compares the shift to the confidence ellipsoid, $D_i=1$ means deleting case $i$ moves $\hat\beta$ to roughly the edge of a 50% joint confidence region ($F_{p,n-p;0.5}\approx1$). Common flags are $D_i>1$ or $D_i>4/n$.</p>
<p>Related measures: $\text{DFFITS}_i=t_i\sqrt{h_{ii}/(1-h_{ii})}$ (the change in $\hat y_i$, in standard errors, using $\hat\sigma_{(i)}$), and $\text{DFBETAS}_{ij}$, the change in a single coefficient in units of its standard error.</p>`,
        lab: 'regression-diagnostics-influence',
      },
      {
        title: 'Graphical diagnostics',
        html: R`
<table>
<thead><tr><th>Plot</th><th>Checks</th><th>Pattern that signals trouble</th></tr></thead>
<tbody>
<tr><td>Residuals vs fitted</td><td>Linearity of the mean</td><td>Curvature (U or arch shape)</td></tr>
<tr><td>Scale–location ($\sqrt{|r_i|}$ vs fitted)</td><td>Constant variance</td><td>Fan or funnel shape</td></tr>
<tr><td>Normal QQ plot of $t_i$</td><td>Normal errors</td><td>S-shapes (tails), curvature (skew)</td></tr>
<tr><td>Residuals vs leverage, Cook contours</td><td>Influence</td><td>Points beyond $D=0.5$ or $1$ contours</td></tr>
<tr><td>Added-variable plot for $x_j$</td><td>Form and influence for one coefficient</td><td>Slope driven by one or two points</td></tr>
<tr><td>Residuals vs time / order</td><td>Independence</td><td>Runs, trends, autocorrelation</td></tr>
</tbody>
</table>
<p>Each plot targets one assumption; none certifies the model. Their purpose is to find structure the model missed. Because the residuals have covariance $\sigma^2M$ rather than $\sigma^2I$, small patterns in small samples are expected; calibrate your eye by simulating residual plots from the fitted model.</p>`,
        lab: 'regression-diagnostics-residuals',
      },
      {
        title: 'Remedies and pitfalls',
        html: R`
<ul>
<li><strong>Nonlinearity</strong>: transform $x$ (log, polynomial, <a href="#/t/splines">splines</a>) or $y$; Box–Cox chooses a power transformation of $y$ by profile likelihood.</li>
<li><strong>Heteroscedasticity</strong>: transform $y$ (log for multiplicative errors), model the variance (<a href="#/t/weighted-gls">WLS</a>), or keep OLS and report sandwich standard errors.</li>
<li><strong>Heavy tails / outliers</strong>: <a href="#/t/robust-statistics">M-estimators</a> (Huber) or high-breakdown methods (LTS, MM) downweight extreme residuals automatically.</li>
</ul>
<div class="warn" data-name="Masking and swamping">
<p>Single-case deletion diagnostics can fail when outliers come in groups. Two high-leverage points together pull the line toward themselves; deleting either one alone leaves the other still holding the line, so neither shows a large $D_i$ (<em>masking</em>). Conversely, a cluster of outliers can make good points look bad (<em>swamping</em>). High-breakdown fits, or deletion of subsets, are needed.</p>
</div>
<div class="warn" data-name="Deleting points is not a remedy">
<p>An influential point is a finding, not a nuisance. It may be a recording error (fix it), a different population (model it), or the most informative observation you have (it is where the model is tested hardest). Report results with and without it, and never iterate "delete the worst point and refit" until the diagnostics look clean — the resulting standard errors are meaningless.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why do high-leverage points tend to have small raw residuals?`,
        a: R`$\Var e_i=\sigma^2(1-h_{ii})$: as $h_{ii}\to1$, the fit is forced through $y_i$ and the residual variance vanishes. The point "pulls the line toward itself", so its own residual understates how unusual it is; the deleted residual $e_i/(1-h_{ii})$ corrects for this.` },
      { q: R`Derive Cook's distance in terms of $r_i$ and $h_{ii}$, and explain what it implies about an outlier at the centroid of the design.`,
        a: R`$D_i=r_i^2h_{ii}/(p(1-h_{ii}))$ from the deletion identity $\hat\beta-\hat\beta_{(i)}=A^{-1}x_ie_i/(1-h_{ii})$. At the centroid (with an intercept) $h_{ii}=1/n$, so $D_i\approx r_i^2/(pn)$: even a large residual there mostly shifts the intercept by $e_i/n$ and has little influence on the fit.` },
      { q: R`How can you compute leave-one-out cross-validation error of a linear regression from one fit?`,
        a: R`The deleted residual is $y_i-\hat y_{(i)}=e_i/(1-h_{ii})$, so $\text{LOOCV}=\frac1n\sum_i\big(e_i/(1-h_{ii})\big)^2$ (the PRESS statistic divided by $n$), requiring only the residuals and leverages of the full fit.` },
      { q: R`A residual-versus-$y$ plot of a correct model shows a clear positive trend. Why, and what should be plotted instead?`,
        a: R`$\Cov(e,y)=\Cov(M\varepsilon,X\beta+\varepsilon)=\sigma^2M$, whose diagonal $\sigma^2(1-h_{ii})$ is positive, so residuals correlate positively with $y$ by construction. Residuals are uncorrelated with fitted values ($\Cov(e,\hat y)=\sigma^2MH=0$), so plot $e$ against $\hat y$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'weighted-gls',
    domain: 'regression',
    order: 70,
    title: 'Weighted & Generalized Least Squares; Robust Standard Errors',
    subtitle: 'Two answers to non-spherical errors: model the covariance, or make the standard errors robust to it',
    level: 2,
    sources: { cb: '§11.3', other: 'White (1980), Econometrica 48; Long & Ervin (2000); Cameron & Miller (2015) on clustering' },
    prereqs: ['gauss-markov', 'regression-inference'],
    related: ['regression-diagnostics', 'bootstrap-regression', 'mixed-models', 'glmm-gee', 'irls', 'time-series'],
    tags: ['heteroscedasticity', 'GLS', 'WLS', 'sandwich', 'HC3', 'cluster-robust'],
    summary: R`When errors have unequal variances or are correlated, $\Var\varepsilon=\sigma^2\Omega\ne\sigma^2I$. Least squares remains unbiased, but it is no longer efficient and — worse in practice — its textbook standard errors are wrong, sometimes badly. There are two cures. <em>Generalized least squares</em> models $\Omega$, whitens the data, and recovers efficiency and exact inference (Aitken's theorem). <em>Sandwich</em> (heteroscedasticity- or cluster-robust) standard errors keep the OLS point estimate and estimate its true variance without modeling $\Omega$ at all. The first is optimal when the variance model is right; the second is valid when it is unknown. Good practice often combines them.`,
    survey: {
      task: R`Estimate $\beta$ and valid standard errors when $\Var\varepsilon=\sigma^2\Omega$`,
      assumptions: R`GLS: $\Omega$ known up to scale or a correct parametric model for it. Sandwich: independence (or independent clusters), large $n$`,
      output: R`$\hat\beta_{\text{GLS}}=(X^\top\Omega^{-1}X)^{-1}X^\top\Omega^{-1}y$; $\widehat{\Var}_{\text{HC}}(\hat\beta_{\text{OLS}})$`,
      strengths: 'GLS: BLUE, exact tests. Sandwich: no variance model needed, consistent under arbitrary heteroscedasticity',
      weaknesses: 'GLS: wrong Ω gives wrong SEs. Sandwich: downward-biased in small samples and with high leverage or few clusters',
      complexity: R`$O(np^2)$ (diagonal $\Omega$); $O(n^3)$ for a dense $\Omega$ without structure`,
    },
    sections: [
      {
        title: 'What goes wrong with OLS',
        html: R`
<p>Let $\E y=X\beta$ and $\Var y=\sigma^2\Omega$, with $\Omega$ positive definite. OLS is still linear and unbiased, but its true covariance is
$$\Var\hat\beta_{\text{OLS}}=(X^\top X)^{-1}X^\top(\sigma^2\Omega)X(X^\top X)^{-1},$$
a "sandwich" with $X^\top\Omega X$ as the meat, rather than $\sigma^2(X^\top X)^{-1}$. And $\E\hat\sigma^2=\sigma^2\tr(M\Omega)/(n-p)$ is not $\sigma^2$ either. The naive standard errors can be too small or too large.</p>
<div class="ex" data-name="Variance that grows with the predictor">
<p>Simple regression with $\Var\varepsilon_i=\sigma^2x_i^2$ and $x_i$ spread over $[1,10]$. The extreme $x$'s, which carry most of the weight in $\hat\beta_1$ (weights $\propto x_i-\bar x$), are also the noisiest. The true variance of $\hat\beta_1$ is $\sigma^2\sum(x_i-\bar x)^2x_i^2/S_{xx}^2$, while the naive formula averages the noise over all points: $\approx\sigma^2\overline{x^2}/S_{xx}$. The naive standard error is too small and $t$-intervals under-cover. The lab below lets you watch it happen.</p>
</div>
<div class="ex" data-name="Autocorrelated errors">
<p>If errors follow a stationary AR(1) with coefficient $\phi>0$ and the regressor is itself positively autocorrelated with lag-one correlation $\rho$, then for large $n$ the ratio of the true to the naive variance of the slope is approximately $(1+\phi\rho)/(1-\phi\rho)$. With $\phi=\rho=0.8$ that is $4.6$: reported standard errors are less than half of what they should be. Trending time series are the classic source of "spurious" significance.</p>
</div>`,
      },
      {
        title: 'Generalized least squares and the whitening transform',
        html: R`
<p>If $\Omega$ is known, factor $\Omega=LL^\top$ (Cholesky, or any square root) and premultiply by $L^{-1}$:
$$\underbrace{L^{-1}y}_{y^\ast}=\underbrace{L^{-1}X}_{X^\ast}\beta+\underbrace{L^{-1}\varepsilon}_{\varepsilon^\ast},\qquad \Var\varepsilon^\ast=\sigma^2L^{-1}\Omega L^{-\top}=\sigma^2I .$$
The whitened model satisfies the Gauss–Markov conditions. OLS on it is
$$\hat\beta_{\text{GLS}}=(X^{\ast\top}X^\ast)^{-1}X^{\ast\top}y^\ast=(X^\top\Omega^{-1}X)^{-1}X^\top\Omega^{-1}y .$$</p>
<div class="thm" data-kind="Theorem" data-name="Properties of GLS">
<ol>
<li>(Aitken) $\hat\beta_{\text{GLS}}$ is BLUE, with $\Var\hat\beta_{\text{GLS}}=\sigma^2(X^\top\Omega^{-1}X)^{-1}\preceq\Var\hat\beta_{\text{OLS}}$.</li>
<li>It minimizes the Mahalanobis criterion $(y-Xb)^\top\Omega^{-1}(y-Xb)$, and under $y\sim\N(X\beta,\sigma^2\Omega)$ it is the MLE.</li>
<li>All of the exact normal-theory inference of <a href="#/t/regression-inference">regression inference</a> applies to the whitened model, with $\hat\sigma^2=\|y^\ast-X^\ast\hat\beta_{\text{GLS}}\|^2/(n-p)$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) is Gauss–Markov applied to the whitened model; linear unbiased estimators of $\beta$ from $y$ and from $y^\ast=L^{-1}y$ are the same class because $L$ is invertible (this is Aitken's theorem, proved in <a href="#/t/gauss-markov">Gauss–Markov</a>). (2) $\|y^\ast-X^\ast b\|^2=(y-Xb)^\top L^{-\top}L^{-1}(y-Xb)=(y-Xb)^\top\Omega^{-1}(y-Xb)$, and the Gaussian log-likelihood is $-\tfrac1{2\sigma^2}$ times this plus terms free of $b$. (3) is immediate.</p>
</details>
<div class="def" data-name="Weighted least squares">
<p>When $\Omega=\diag(1/w_1,\dots,1/w_n)$ (independent errors, $\Var\varepsilon_i=\sigma^2/w_i$), GLS becomes WLS: minimize $\sum_iw_i(y_i-x_i^\top b)^2$, i.e. $\hat\beta_{\text{WLS}}=(X^\top WX)^{-1}X^\top Wy$. Whitening multiplies row $i$ by $\sqrt{w_i}$.</p>
</div>
<p>Natural weights arise when each response is an average of $m_i$ raw observations ($w_i=m_i$), or when the variance is a known function of the mean (counts, proportions — the gateway to <a href="#/t/irls">IRLS</a> for GLMs). The weights need only be known up to a constant.</p>`,
      },
      {
        title: 'Feasible GLS: estimating the covariance',
        html: R`
<p>In practice $\Omega=\Omega(\theta)$ depends on unknown parameters. <strong>Feasible GLS</strong> estimates $\theta$ and plugs in.</p>
<div class="algo" data-name="Feasible GLS">
<ol>
<li>Fit OLS (consistent for $\beta$ under the usual conditions) and form residuals $e$.</li>
<li>Estimate $\theta$ from $e$ — e.g. regress $\log e_i^2$ on covariates for a variance function $\Var\varepsilon_i=\sigma^2\exp(z_i^\top\gamma)$; or estimate the lag-one autocorrelation $\hat\phi$ for AR(1) errors.</li>
<li>Compute $\hat\beta_{\text{FGLS}}=(X^\top\Omega(\hat\theta)^{-1}X)^{-1}X^\top\Omega(\hat\theta)^{-1}y$. Optionally iterate steps 2–3 (iterated FGLS; at convergence under normality, this is the MLE).</li>
</ol>
</div>
<p>For AR(1) errors $\varepsilon_t=\phi\varepsilon_{t-1}+u_t$, whitening is the <strong>Prais–Winsten</strong> transform: $y_1^\ast=\sqrt{1-\phi^2}\,y_1$ and $y_t^\ast=y_t-\phi y_{t-1}$ for $t\ge2$ (same for each column of $X$). Under regularity conditions — $\hat\theta$ consistent, $\Omega(\theta)$ smooth — FGLS has the same asymptotic distribution as GLS with the true $\Omega$: estimating a finite-dimensional $\theta$ costs nothing to first order. In small samples the uncertainty in $\hat\theta$ makes FGLS standard errors somewhat optimistic.</p>
<div class="warn" data-name="A wrong variance model">
<p>If the assumed $\Omega$ is wrong, GLS with that $\Omega$ is still unbiased (as is any linear estimator with $CX=I$) but no longer optimal, and the model-based covariance $\sigma^2(X^\top\hat\Omega^{-1}X)^{-1}$ is wrong. The cure is to pair the weighted estimator with a sandwich covariance — the strategy of <a href="#/t/glmm-gee">GEE</a>.</p>
</div>`,
      },
      {
        title: 'Sandwich estimators: robust standard errors without a variance model',
        html: R`
<p>Suppose errors are independent with arbitrary variances $\sigma_i^2$. The true covariance of OLS is
$$\Var\hat\beta=(X^\top X)^{-1}\Big(\sum_i\sigma_i^2x_ix_i^\top\Big)(X^\top X)^{-1}.$$
We cannot estimate each $\sigma_i^2$, but we do not need to: we need only the $p\times p$ average $\frac1n\sum\sigma_i^2x_ix_i^\top$, and $e_i^2$ is a (nearly) unbiased, if very noisy, estimate of $\sigma_i^2$.</p>
<div class="def" data-name="Heteroscedasticity-consistent (HC) covariance estimators">
<p>$$\widehat{\Var}_{\text{HC}}(\hat\beta)=(X^\top X)^{-1}\Big(\sum_i\omega_i\,x_ix_i^\top\Big)(X^\top X)^{-1}$$
with $\omega_i=e_i^2$ (HC0, White 1980), $\tfrac{n}{n-p}e_i^2$ (HC1), $\tfrac{e_i^2}{1-h_{ii}}$ (HC2), or $\tfrac{e_i^2}{(1-h_{ii})^2}$ (HC3).</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Consistency of the sandwich (White, 1980)">
<p>If observations are independent, $\frac1nX^\top X\to Q\succ0$, and suitable moments are bounded (e.g. uniformly bounded $\E|\varepsilon_i|^{4+\delta}$ and $\|x_i\|$), then
$$\frac1n\sum_ie_i^2x_ix_i^\top-\frac1n\sum_i\sigma_i^2x_ix_i^\top\pto0,$$
so $n\,\widehat{\Var}_{\text{HC0}}$ consistently estimates the asymptotic covariance of $\sqrt n(\hat\beta-\beta)$, and Wald tests using it are asymptotically valid under heteroscedasticity of unknown form.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Write $e_i=\varepsilon_i-x_i^\top(\hat\beta-\beta)$, so $e_i^2=\varepsilon_i^2-2\varepsilon_ix_i^\top(\hat\beta-\beta)+\big(x_i^\top(\hat\beta-\beta)\big)^2$. The average of $(\varepsilon_i^2-\sigma_i^2)x_ix_i^\top$ is $o_p(1)$ by a weak law for independent, non-identical terms with bounded $(1+\delta/2)$-th moments. The other terms are bounded by $\|\hat\beta-\beta\|=O_p(n^{-1/2})$ times averages that are $O_p(1)$. Note that we estimate a $p\times p$ matrix, not $n$ variances — that is why it works.</p>
</details>
<p>Why the leverage corrections? Under homoscedasticity $\E e_i^2=\sigma^2(1-h_{ii})$, so HC0 is biased downward, most for high-leverage points — exactly the points that dominate the sandwich. HC2 is <em>exactly unbiased</em> under homoscedasticity; HC3 over-corrects slightly, approximating a jackknife, and has the best small-sample coverage in simulations (Long & Ervin, 2000). HC3 is a sensible default for $n$ below a few hundred.</p>`,
        lab: 'weighted-gls-sandwich',
      },
      {
        title: 'Clusters, serial correlation, and testing for heteroscedasticity',
        html: R`
<p>The sandwich idea generalizes to any structure where the "meat" can be estimated by summing over independent units:</p>
<ul>
<li><strong>Cluster-robust</strong>: with independent clusters $g=1,\dots,G$ but arbitrary correlation within, the meat is $\sum_gX_g^\top e_ge_g^\top X_g$. Consistency requires $G\to\infty$; with fewer than about 30–50 clusters, use small-sample corrections or a wild cluster bootstrap.</li>
<li><strong>HAC (Newey–West)</strong>: for time series, the meat sums autocovariances of $x_te_t$ up to lag $\ell$ with weights $1-j/(\ell+1)$, which keep the estimate positive semidefinite; consistency requires $\ell\to\infty$ slowly.</li>
</ul>
<p>To test for heteroscedasticity, the <strong>Breusch–Pagan</strong> test (in Koenker's studentized form) regresses $e_i^2$ on the predictors (or on chosen variables $z_i$) and compares $nR^2$ to $\chi^2_q$; White's test adds squares and cross-products. But a non-significant test is not a license to use naive standard errors, and a significant one does not tell you the right weights. Many practitioners simply report robust standard errors by default.</p>`,
      },
      {
        title: 'Choosing between modeling and robustifying',
        html: R`
<div class="cols">
<div><h4>Model the variance (WLS/GLS)</h4><p>Efficient if right — can dramatically shrink standard errors when variances differ by orders of magnitude. Gives exact small-sample inference and better prediction intervals (which need the variance at $x_0$!). Wrong model: inefficient estimator, wrong SEs.</p></div>
<div><h4>Robust SEs (OLS + sandwich)</h4><p>No variance model; valid asymptotically under any heteroscedasticity. Point estimates unchanged, so no gain in efficiency. Noisy and downward-biased in small samples; useless for prediction intervals.</p></div>
<div><h4>Both</h4><p>Fit WLS with a plausible variance function <em>and</em> report a sandwich covariance for it: $(X^\top WX)^{-1}(\sum w_i^2e_i^2x_ix_i^\top)(X^\top WX)^{-1}$. Efficiency if the model is roughly right, validity if it is not.</p></div>
</div>
<div class="warn" data-name="Robust SEs do not fix a wrong mean">
<p>Heteroscedasticity-robust standard errors protect inference about the <em>best linear approximation</em> coefficients against unequal variances. They do nothing about a misspecified mean function, omitted confounders, or dependence not captured by the clustering. A dramatic gap between naive and robust standard errors is itself a diagnostic worth investigating.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that WLS with weights $w_i$ is OLS on rows rescaled by $\sqrt{w_i}$.`,
        a: R`$\sum_iw_i(y_i-x_i^\top b)^2=\sum_i(\sqrt{w_i}y_i-\sqrt{w_i}x_i^\top b)^2=\|W^{1/2}y-W^{1/2}Xb\|^2$. Minimizing gives $(X^\top WX)^{-1}X^\top Wy$. This is the whitening transform with $L^{-1}=W^{1/2}$.` },
      { q: R`Why is HC0 biased downward under homoscedasticity, and which variant fixes this exactly?`,
        a: R`Because $\E e_i^2=\sigma^2(1-h_{ii})<\sigma^2$: residuals are shrunk, most at high-leverage points. HC2, which uses $e_i^2/(1-h_{ii})$, is exactly unbiased under homoscedasticity: $\E\,\widehat{\Var}_{\text{HC2}}=(X^\top X)^{-1}\big(\sum\sigma^2x_ix_i^\top\big)(X^\top X)^{-1}=\sigma^2(X^\top X)^{-1}$.` },
      { q: R`If you know $\Var\varepsilon_i=\sigma^2x_i^2$, why might you still prefer WLS over OLS with HC3 standard errors?`,
        a: R`WLS with $w_i=1/x_i^2$ is BLUE: its variance $\sigma^2(X^\top WX)^{-1}$ can be much smaller than that of OLS, so intervals are shorter while remaining exact under normality. HC3 only repairs the standard error of the less efficient OLS estimate, and is approximate.` },
      { q: R`A panel study has 12 states observed over 20 years. Is cluster-robust inference by state reliable?`,
        a: R`Not with the plain cluster sandwich: its consistency is in the number of clusters $G$, and with $G=12$ it is badly biased downward and the normal reference is poor. Use small-$G$ corrections (e.g. $t_{G-1}$ reference with bias-reduced meat) or a wild cluster bootstrap.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'model-selection',
    domain: 'regression',
    order: 80,
    title: 'Model Selection: Cp, AIC, BIC & Cross-Validation',
    subtitle: 'Estimating prediction risk honestly, and why the two classic criteria disagree',
    level: 2,
    sources: { aos: '13.6', other: 'Mallows (1973); Akaike (1974); Schwarz (1978); Efron (2004), JASA 99; Yang (2005), Biometrika 92' },
    prereqs: ['ols-geometry', 'bias-variance'],
    related: ['cross-validation', 'lasso', 'ridge-regression', 'likelihood-ratio-tests', 'bayesian-inference', 'entropy-kl'],
    tags: ['Mallows Cp', 'AIC', 'BIC', 'optimism', 'degrees of freedom', 'post-selection inference'],
    summary: R`Adding predictors always lowers training error, so training error cannot choose a model. What we want is prediction risk, and the gap between risk and training error — the <em>optimism</em> — can be computed: for a linear fit it is $2\sigma^2$ times the number of parameters. Correcting for it gives Mallows' $C_p$ and its likelihood generalization, AIC, both unbiased estimates of predictive risk. BIC instead approximates the log marginal likelihood by Laplace's method and penalizes each parameter by $\log n$. The two answer different questions: AIC aims at good prediction and tends to overfit when a small true model exists; BIC aims at identifying that model and is consistent when it exists. Cross-validation estimates risk directly. And whatever you choose, inference on the chosen model is no longer straightforward.`,
    survey: {
      task: 'Choose among candidate models (subsets, polynomial degrees, tuning parameters)',
      assumptions: R`Cp: $\sigma^2$ known or estimated from a big model. AIC/BIC: likelihood, regularity, fixed $p$. CV: exchangeable data`,
      output: R`A criterion value per model; the minimizer`,
      strengths: 'Cp/AIC: unbiased risk estimates, cheap. BIC: consistent selection. CV: assumption-light',
      weaknesses: 'AIC overfits nested truths; BIC underfits when truth is complex; all invalidate naive post-selection inference',
      complexity: R`Criterion per model is one fit; exhaustive subsets cost $2^p$ fits`,
    },
    sections: [
      {
        title: 'Prediction risk and the trade-off',
        html: R`
<p>Fix the design $X$ (all $n\times P$ candidate columns) and let $\mu=\E y$, $\Var y=\sigma^2I$. For a subset $S$ of columns with $|S|$ elements, the fit is $\hat y_S=H_Sy$. Measure quality by the expected error predicting a fresh response vector $y^\ast$ at the same design points, independent of $y$:
$$R(S)=\E\|y^\ast-\hat y_S\|^2 .$$
We do not assume any candidate model is correct: $\mu$ is arbitrary.</p>
<div class="thm" data-kind="Proposition" data-name="Risk decomposition for a subset">
<p>$$R(S)=\underbrace{n\sigma^2}_{\text{irreducible}}+\underbrace{\|(I-H_S)\mu\|^2}_{\text{squared bias}}+\underbrace{|S|\,\sigma^2}_{\text{variance}},\qquad \E\,\text{RSS}_S=\|(I-H_S)\mu\|^2+(n-|S|)\sigma^2 .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$y^\ast-\hat y_S=(y^\ast-\mu)+(I-H_S)\mu-H_S\varepsilon$, three uncorrelated pieces with zero-mean random parts; their expected squared norms are $n\sigma^2$, $\|(I-H_S)\mu\|^2$, and $\sigma^2\tr H_S=|S|\sigma^2$. For training error, $y-\hat y_S=(I-H_S)\mu+(I-H_S)\varepsilon$ and $\E\|(I-H_S)\varepsilon\|^2=\sigma^2(n-|S|)$.</p>
</details>
<p>Adding a column can only decrease the bias term (the column space grows) and always increases variance by exactly $\sigma^2$. Adding a column is worthwhile iff it reduces squared bias by more than $\sigma^2$. Training error hides this, because it <em>decreases</em> by $\sigma^2$ per parameter on average instead of increasing — this is the sense in which it is optimistic (see <a href="#/t/bias-variance">bias–variance</a>).</p>`,
      },
      {
        title: 'Optimism, degrees of freedom, and Mallows’ Cp',
        html: R`
<p>Subtracting the two formulas: $R(S)-\E\,\text{RSS}_S=2|S|\sigma^2$. This is a special case of a general identity.</p>
<div class="thm" data-kind="Theorem" data-name="Covariance penalty (Efron, 2004)">
<p>For any fitting rule $\hat y=\hat y(y)$ and $y^\ast$ an independent copy of $y$,
$$\E\|y^\ast-\hat y\|^2-\E\|y-\hat y\|^2=2\sum_{i=1}^n\Cov(\hat y_i,y_i)=:2\sigma^2\,\text{df}(\hat y).$$
For a linear smoother $\hat y=Sy$ (with $S$ not depending on $y$), $\text{df}=\tr S$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Coordinatewise, $\E(y_i^\ast-\hat y_i)^2-\E(y_i-\hat y_i)^2=\E(y_i^{\ast2})-\E(y_i^2)-2\E(y_i^\ast)\E(\hat y_i)+2\E(y_i\hat y_i)$, using independence of $y^\ast$ and $\hat y$. Since $y_i^\ast$ and $y_i$ have the same distribution, this is $2\big(\E(y_i\hat y_i)-\E y_i\,\E\hat y_i\big)=2\Cov(y_i,\hat y_i)$. For $\hat y=Sy$, $\sum_i\Cov(\hat y_i,y_i)=\tr\Cov(Sy,y)=\sigma^2\tr S$.</p>
</details>
<p>Degrees of freedom so defined measure how strongly the fit chases its own data. They equal the parameter count for least squares, $\sum d_j^2/(d_j^2+\lambda)$ for <a href="#/t/ridge-regression">ridge</a>, and (in expectation) the number of nonzero coefficients for the <a href="#/t/lasso">lasso</a>. For best-subset selection they <em>exceed</em> the size of the chosen model, because the selection itself adapts to $y$.</p>
<div class="def" data-name="Mallows’ Cp">
<p>With $\hat\sigma^2$ estimated from the largest model (unbiased if that model contains $\mu$'s column space),
$$\hat R(S)=\text{RSS}_S+2|S|\hat\sigma^2,\qquad\text{or in Mallows' normalization}\qquad C_p(S)=\frac{\text{RSS}_S}{\hat\sigma^2}-n+2|S| .$$
$\hat R(S)$ is unbiased for $R(S)$ when $\sigma^2$ is known; models with negligible bias have $C_p\approx|S|$.</p>
</div>`,
      },
      {
        title: 'AIC: estimating expected out-of-sample log-likelihood',
        html: R`
<p>For a parametric model $\{f(\cdot;\theta):\theta\in\R^k\}$ fit by maximum likelihood, the natural predictive target is $\E_{y^\ast}\log f(y^\ast;\hat\theta)$ — larger means smaller <a href="#/t/entropy-kl">Kullback–Leibler</a> divergence from the truth. The maximized log-likelihood $\ell(\hat\theta)$ overestimates it, because $\hat\theta$ was tuned to the same data.</p>
<div class="thm" data-kind="Theorem" data-name="Akaike (1974)">
<p>Under standard regularity conditions and if the model is correctly specified,
$$\E\big[\ell(\hat\theta)\big]-\E\Big[\E_{y^\ast}\log f(y^\ast;\hat\theta)\Big]=k+o(1),$$
so $\text{AIC}=-2\ell(\hat\theta)+2k$ is an asymptotically unbiased estimate of $-2$ times expected out-of-sample log-likelihood.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Let $\theta_0$ be the true parameter, $L(\theta)=\E_{y^\ast}\log f(y^\ast;\theta)$, and $I$ the per-observation Fisher information. Two second-order expansions, each around a maximizer:</p>
<p>(a) $L$ is maximized at $\theta_0$: $L(\hat\theta)\approx L(\theta_0)-\tfrac n2(\hat\theta-\theta_0)^\top I(\hat\theta-\theta_0)$.</p>
<p>(b) $\ell$ is maximized at $\hat\theta$: $\ell(\theta_0)\approx\ell(\hat\theta)-\tfrac n2(\hat\theta-\theta_0)^\top I(\hat\theta-\theta_0)$.</p>
<p>Since $n(\hat\theta-\theta_0)^\top I(\hat\theta-\theta_0)\dto\chi^2_k$ with mean $k$, and $\E\ell(\theta_0)=L(\theta_0)$, taking expectations gives $\E\ell(\hat\theta)\approx L(\theta_0)+k/2$ and $\E L(\hat\theta)\approx L(\theta_0)-k/2$. The difference is $k$. Under misspecification the penalty becomes $\tr(JI^{-1})$ (Takeuchi's TIC), with $J$ the variance of the score.</p>
</details>
<p>For a Gaussian linear model with known $\sigma^2$, $-2\ell=\text{RSS}/\sigma^2+\text{const}$, so AIC is $C_p$ up to scaling. With $\sigma^2$ estimated by maximum likelihood, $\text{AIC}=n\log(\text{RSS}_S/n)+2|S|+\text{const}$. In small samples Hurvich & Tsai's correction $\text{AIC}_c=\text{AIC}+\tfrac{2k(k+1)}{n-k-1}$ reduces the tendency to overfit.</p>`,
      },
      {
        title: 'BIC: the Laplace approximation to the evidence',
        html: R`
<p>The Bayesian approach compares models by their marginal likelihood $p(y\mid M)=\int f(y;\theta)\pi(\theta)\d\theta$ (see <a href="#/t/bayesian-inference">Bayesian inference</a>). With equal prior model probabilities, the posterior favours the model with the largest evidence.</p>
<div class="thm" data-kind="Theorem" data-name="Schwarz (1978)">
<p>For i.i.d. data, a smooth prior positive at the MLE, and the usual regularity conditions,
$$\log p(y\mid M)=\ell(\hat\theta)-\frac k2\log n+O_p(1),$$
so $\text{BIC}=-2\ell(\hat\theta)+k\log n$ approximates $-2\log p(y\mid M)$ up to a bounded error.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Expand $\ell(\theta)\approx\ell(\hat\theta)-\tfrac n2(\theta-\hat\theta)^\top\hat I(\theta-\hat\theta)$ where $\hat I=-\tfrac1n\nabla^2\ell(\hat\theta)$. The integrand concentrates in an $O(n^{-1/2})$ neighbourhood of $\hat\theta$, where $\pi$ is nearly constant, so Laplace's method gives
$$p(y\mid M)\approx e^{\ell(\hat\theta)}\pi(\hat\theta)\int e^{-\frac n2(\theta-\hat\theta)^\top\hat I(\theta-\hat\theta)}\d\theta=e^{\ell(\hat\theta)}\pi(\hat\theta)\Big(\frac{2\pi}{n}\Big)^{k/2}|\hat I|^{-1/2}.$$
Taking logs: $\ell(\hat\theta)-\tfrac k2\log n+\big[\log\pi(\hat\theta)+\tfrac k2\log2\pi-\tfrac12\log|\hat I|\big]$. The bracket is $O_p(1)$; dropping it gives BIC.</p>
</details>
<div class="thm" data-kind="Proposition" data-name="Consistency versus efficiency">
<p>Suppose a finite list of nested models contains a smallest correct one, $M_0$. Comparing $M_0$ with a larger correct model $M_1$ having one extra parameter, the likelihood-ratio statistic $2(\ell_1-\ell_0)\dto\chi^2_1$. AIC picks $M_1$ when it exceeds $2$, which happens with limiting probability $\P(\chi^2_1>2)\approx0.157$. BIC picks $M_1$ when it exceeds $\log n\to\infty$, with probability $\to0$. Underfitting a model with a fixed nonzero effect costs $O_p(n)$ in log-likelihood, which outweighs either penalty. Hence BIC is <em>consistent</em> and AIC is not.</p>
</div>
<p>But consistency is not everything. When the truth is not in the candidate list (e.g. an infinite-dimensional $\mu$ approximated by growing models), AIC and $C_p$ are <em>asymptotically efficient</em>: their risk approaches that of the best candidate (Shibata, 1981), while BIC can be slower. Yang (2005) showed no criterion can be both consistent and minimax-rate optimal for prediction. Choose by the question: interpretation of a sparse truth, or prediction.</p>`,
      },
      {
        title: 'Cross-validation and the criteria in action',
        html: R`
<p><a href="#/t/cross-validation">Cross-validation</a> estimates prediction error directly by holding out data. For linear smoothers leave-one-out CV costs a single fit, by the deletion identity of <a href="#/t/regression-diagnostics">regression diagnostics</a>:
$$\text{LOOCV}=\frac1n\sum_{i=1}^n\Big(\frac{y_i-\hat y_i}{1-h_{ii}}\Big)^2,\qquad \text{GCV}=\frac{\frac1n\text{RSS}}{\big(1-\tr H/n\big)^2}.$$
GCV replaces each $h_{ii}$ by the average $\tr H/n$; expanding $(1-\tr H/n)^{-2}\approx1+2\tr H/n$ shows $n\cdot\text{GCV}\approx\text{RSS}+2\,\tr H\cdot\text{RSS}/n$, which is $C_p$ with $\text{RSS}/n$ standing in for $\sigma^2$. Stone (1977) showed LOOCV and AIC are asymptotically equivalent; leave-$n_v$-out CV with $n_v/n\to1$ behaves like BIC (Shao, 1993).</p>
<p>The first lab fits polynomials of increasing degree and plots training error, $C_p$, BIC, LOOCV and the true risk on one scale. The second repeats the experiment many times and tallies which degree each criterion selects as $n$ grows — making the consistency results visible.</p>`,
        lab: 'model-selection-curves',
      },
      {
        title: 'Search, and inference after selection',
        html: R`
<p>With $P$ candidate predictors there are $2^P$ subsets. Exhaustive search (leaps-and-bounds) is feasible to $P\approx40$; beyond, greedy forward or backward stepwise search, or convex relaxations such as the <a href="#/t/lasso">lasso</a>, are used.</p>
<div class="warn" data-name="Post-selection inference">
<p>After choosing a model using the data, the usual $t$-intervals and $p$-values for the chosen coefficients are invalid: a coefficient was kept <em>because</em> its estimate looked large, so its estimate is biased away from zero and its naive interval undercovers (a winner's curse). <strong>Freedman's paradox</strong> (1983): with $n=100$ and $50$ pure-noise predictors, screening at $p<0.25$ and refitting typically produces a highly "significant" overall $F$ and several significant coefficients.</p>
</div>
<p>Remedies, roughly in increasing sophistication:</p>
<ul>
<li><strong>Sample splitting</strong>: select on one half, infer on the other. Valid and simple; costs power and makes results depend on the split.</li>
<li><strong>Simultaneous inference over all models</strong> (PoSI; Berk et al., 2013): widen intervals enough to cover every submodel's coefficients at once. Valid for any selection rule, conservative.</li>
<li><strong>Selective inference</strong> (e.g. Lee et al., 2016 for the lasso): condition on the selection event, which for many procedures is a polyhedron in $y$-space, and use truncated-normal pivots.</li>
<li><strong>Honest reporting</strong>: if selection was exploratory, say so, and validate on new data.</li>
</ul>`,
        lab: 'model-selection-frequency',
      },
    ],
    quiz: [
      { q: R`Training RSS decreases by about $\sigma^2$ per added parameter even when the parameter is useless. Use the risk decomposition to explain why this means training error is biased by $-2\sigma^2$ per parameter as an estimate of risk.`,
        a: R`$\E\,\text{RSS}_S=\text{bias}^2+(n-|S|)\sigma^2$ while $R(S)=n\sigma^2+\text{bias}^2+|S|\sigma^2$. A useless parameter leaves the bias unchanged, lowers expected RSS by $\sigma^2$, and raises risk by $\sigma^2$. The gap $R-\E\,\text{RSS}=2|S|\sigma^2$, i.e. $2\sigma^2$ per parameter.` },
      { q: R`Compute the effective degrees of freedom of a fit that sets $\hat y=y$ (interpolation), and of $\hat y=\bar y\mathbf 1$.`,
        a: R`$\hat y=Iy$ has $\text{df}=\tr I=n$; its training error is $0$ but its risk is $2n\sigma^2$. $\hat y=\bar y\mathbf 1=\tfrac1n\mathbf 1\mathbf 1^\top y$ has $\text{df}=\tr(\tfrac1n\mathbf 1\mathbf 1^\top)=1$.` },
      { q: R`For $n=1000$, how large must the likelihood-ratio statistic for one extra parameter be for AIC to prefer the bigger model? For BIC?`,
        a: R`AIC: $2\Delta\ell>2$. BIC: $2\Delta\ell>\log1000\approx6.91$. Under the null, $2\Delta\ell\approx\chi^2_1$, so AIC adds a useless parameter about $15.7\%$ of the time and BIC about $0.9\%$ (and that rate goes to zero as $n$ grows).` },
      { q: R`Why do the $O_p(1)$ terms dropped in the BIC derivation (including the prior) not matter for consistency?`,
        a: R`Differences between correct nested models in $\ell(\hat\theta)$ are $O_p(1)$ while the penalty difference $\tfrac12\log n\to\infty$; differences between a correct and an incorrect model are of order $n$. Bounded terms cannot change which side wins in either comparison as $n\to\infty$.` },
      { q: R`You choose 5 of 40 predictors by forward stepwise and report the refitted OLS $p$-values. What is wrong and what is a simple valid alternative?`,
        a: R`The predictors were chosen because they looked strong in these data, so the refitted estimates are biased away from zero and their $p$-values too small (the null distribution ignores selection). Splitting the sample — select on one half, fit and test on the other — gives valid inference for the selected model.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'ridge-regression',
    domain: 'regression',
    order: 90,
    title: 'Ridge Regression & Shrinkage',
    subtitle: 'Trading a little bias for a lot of variance, one principal direction at a time',
    level: 2,
    sources: { aos: '13.6', other: 'Hoerl & Kennard (1970), Technometrics 12; Hastie, Tibshirani & Friedman, ESL (2009), §3.4' },
    prereqs: ['ols-geometry', 'gauss-markov'],
    related: ['lasso', 'model-selection', 'pca', 'bayesian-inference', 'kernel-methods', 'gaussian-processes', 'bias-variance'],
    tags: ['shrinkage', 'regularization', 'SVD', 'effective degrees of freedom', 'Tikhonov'],
    summary: R`Ridge regression adds a penalty $\lambda\|\beta\|^2$ to the least-squares objective. The solution $(X^\top X+\lambda I)^{-1}X^\top y$ always exists, even when predictors are collinear or $p>n$. Through the singular value decomposition its action is transparent: it shrinks the component of the fit along each principal direction of the design by the factor $d_j^2/(d_j^2+\lambda)$, so the poorly determined, low-variance directions are shrunk most. The resulting bias is more than paid for by the reduced variance — Hoerl and Kennard proved that some $\lambda>0$ always beats least squares in mean squared error. Ridge is also the posterior mean under a Gaussian prior and the simplest instance of the kernel trick.`,
    survey: {
      task: R`Stabilize linear regression under collinearity or $p\gtrsim n$; improve prediction MSE`,
      assumptions: R`Standardized predictors (penalty is not scale-invariant); intercept unpenalized`,
      output: R`$\hat\beta_\lambda=(X^\top X+\lambda I)^{-1}X^\top y$; a path over $\lambda$; $\text{df}(\lambda)=\sum d_j^2/(d_j^2+\lambda)$`,
      strengths: 'Closed form; always unique; dominates OLS in MSE for small λ; cheap paths via one SVD',
      weaknesses: 'Never sets coefficients exactly to zero; biased, so naive inference is invalid; needs λ tuning',
      complexity: R`One SVD $O(np\min(n,p))$, then $O(p^2)$ per $\lambda$`,
    },
    sections: [
      {
        title: 'The estimator',
        html: R`
<p>Center $y$ and the columns of $X$, and scale each column to unit variance (the penalty treats all coefficients alike, so the units of the predictors matter). The intercept is then $\bar y$ and is not penalized. Ridge regression solves
$$\hat\beta_\lambda=\argmin_b\ \|y-Xb\|^2+\lambda\|b\|^2=(X^\top X+\lambda I_p)^{-1}X^\top y,\qquad\lambda\ge0 .$$
The objective is strictly convex for $\lambda>0$ and $X^\top X+\lambda I\succeq\lambda I$ is invertible regardless of the rank of $X$: ridge is well-defined under exact collinearity and for $p>n$.</p>
<p>By Lagrangian duality the problem is equivalent to minimizing RSS subject to $\|b\|^2\le t$ for some $t=t(\lambda)$, decreasing in $\lambda$. Geometrically, the solution is where an elliptical RSS contour first touches a sphere centered at the origin. It is also least squares on augmented data: append $\sqrt\lambda\,I_p$ to $X$ and $p$ zeros to $y$ — "pseudo-observations" that pull every coefficient toward zero.</p>
<div class="note">
<p>Under collinearity the RSS surface has a long flat valley: many coefficient vectors fit almost equally well, and OLS picks one by chasing noise. The penalty adds a bowl centered at the origin that picks the shortest vector along the valley. The flatter the valley in some direction, the more the penalty dominates there.</p>
</div>`,
      },
      {
        title: 'The SVD view: shrinkage along principal directions',
        html: R`
<p>Let $X=UDV^\top$ be the thin SVD, with $d_1\ge\dots\ge d_r>0$ and $r=\operatorname{rank}X$. The columns $v_j$ of $V$ are the principal-component directions of the (centered) design, and $d_j^2/(n-1)$ is the sample variance along $v_j$ (see <a href="#/t/pca">PCA</a>).</p>
<div class="thm" data-kind="Proposition" data-name="Ridge in the SVD basis">
<p>$$\hat\beta_\lambda=\sum_{j=1}^r v_j\,\frac{d_j}{d_j^2+\lambda}\,u_j^\top y,\qquad X\hat\beta_\lambda=\sum_{j=1}^ru_j\,\frac{d_j^2}{d_j^2+\lambda}\,u_j^\top y .$$
OLS ($\lambda=0$, full rank) has factors $1/d_j$ and $1$ respectively.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For full column rank, $X^\top X+\lambda I=V(D^2+\lambda I)V^\top$ with $V$ square orthogonal, so $\hat\beta_\lambda=V(D^2+\lambda I)^{-1}DU^\top y$. If $r&lt;p$, extend $V$ by an orthonormal basis of the null space of $X$; on those directions $X^\top y$ has no component, so the solution has none either.</p>
</details>
<p>So the fitted values are the OLS fit with its component along $u_j$ shrunk by $d_j^2/(d_j^2+\lambda)\in(0,1)$. Directions with large $d_j$ (well-spread data, well-determined coefficients) are barely touched; directions with $d_j^2\ll\lambda$ (collinear combinations) are nearly zeroed. <strong>Principal components regression</strong> does the hard-threshold version: keep the top $m$ components, drop the rest. Ridge is a smooth version that never discards anything entirely.</p>
<div class="def" data-name="Effective degrees of freedom">
<p>Ridge is a linear smoother $\hat y=H_\lambda y$ with $H_\lambda=X(X^\top X+\lambda I)^{-1}X^\top$, so by the covariance-penalty definition of <a href="#/t/model-selection">model selection</a>,
$$\text{df}(\lambda)=\tr H_\lambda=\sum_{j=1}^r\frac{d_j^2}{d_j^2+\lambda},$$
which decreases continuously from $r$ at $\lambda=0$ to $0$ as $\lambda\to\infty$. Note $H_\lambda$ is not a projection: its eigenvalues lie strictly between 0 and 1.</p>
</div>`,
        lab: 'ridge-regression-path',
      },
      {
        title: 'Bias, variance, and the Hoerl–Kennard theorem',
        html: R`
<p>Assume $y=X\beta+\varepsilon$, $\Var\varepsilon=\sigma^2I$, $X$ of full column rank. Work in principal coordinates $\theta=V^\top\beta$, $\hat\theta=V^\top\hat\beta_\lambda$; since $V$ is orthogonal, $\E\|\hat\beta_\lambda-\beta\|^2=\E\|\hat\theta-\theta\|^2$.</p>
<div class="thm" data-kind="Proposition" data-name="Componentwise bias and variance">
<p>$\hat\theta_j=\frac{d_j}{d_j^2+\lambda}u_j^\top y$ are uncorrelated across $j$ with
$$\E\hat\theta_j-\theta_j=-\frac{\lambda}{d_j^2+\lambda}\theta_j,\qquad \Var\hat\theta_j=\frac{\sigma^2d_j^2}{(d_j^2+\lambda)^2},$$
$$\MSE(\lambda):=\E\|\hat\beta_\lambda-\beta\|^2=\sum_{j=1}^p\frac{\sigma^2d_j^2+\lambda^2\theta_j^2}{(d_j^2+\lambda)^2}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$u_j^\top y=u_j^\top UDV^\top\beta+u_j^\top\varepsilon=d_j\theta_j+u_j^\top\varepsilon$, and the $u_j^\top\varepsilon$ are uncorrelated with variance $\sigma^2$ because the $u_j$ are orthonormal. Multiply by $d_j/(d_j^2+\lambda)$.</p>
</details>
<p>At $\lambda=0$ the MSE is $\sigma^2\sum_j1/d_j^2$, which explodes when any $d_j$ is small. Ridge caps each variance term at $\sigma^2d_j^2/(d_j^2+\lambda)^2\le\sigma^2/(4\lambda)$ (attained at $d_j^2=\lambda$), at the cost of bias proportional to the size of $\theta_j$.</p>
<div class="thm" data-kind="Theorem" data-name="Hoerl & Kennard (1970)">
<p>If $\sigma^2>0$, then $\MSE(\lambda)$ is strictly decreasing on $\big[0,\ \sigma^2/\max_j\theta_j^2\big]$. In particular there exists $\lambda>0$ with $\MSE(\lambda)<\MSE(0)$: ridge strictly beats least squares for every $\beta$ — but the good range of $\lambda$ depends on the unknown $\beta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Differentiate one term $f_j(\lambda)=(\sigma^2d_j^2+\lambda^2\theta_j^2)/(d_j^2+\lambda)^2$:
$$f_j'(\lambda)=\frac{2\lambda\theta_j^2(d_j^2+\lambda)-2(\sigma^2d_j^2+\lambda^2\theta_j^2)}{(d_j^2+\lambda)^3}=\frac{2d_j^2\,(\lambda\theta_j^2-\sigma^2)}{(d_j^2+\lambda)^3}.$$
This is negative whenever $\lambda\theta_j^2<\sigma^2$ (always if $\theta_j=0$). For $\lambda<\sigma^2/\max_j\theta_j^2$ every term is strictly decreasing, so the sum is too.</p>
</details>
<p>The derivative also gives the optimum. With a separate penalty per direction ("generalized ridge"), the best choice is $\lambda_j=\sigma^2/\theta_j^2$. With a single $\lambda$ and equal $d_j$, setting $\sum_jf_j'=0$ gives $\lambda^\star=p\sigma^2/\|\theta\|^2$ — the same signal-to-noise logic as James–Stein shrinkage. This does not contradict <a href="#/t/gauss-markov">Gauss–Markov</a>: ridge is linear but biased.</p>`,
        lab: 'ridge-regression-bias-variance',
      },
      {
        title: 'Bayesian, kernel, and noise-injection views',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Ridge as a posterior mean">
<p>If $y\mid\beta\sim\N(X\beta,\sigma^2I)$ and $\beta\sim\N(0,\tau^2I)$, then $\beta\mid y$ is Gaussian with mean $\hat\beta_\lambda$ for $\lambda=\sigma^2/\tau^2$ and covariance $\sigma^2(X^\top X+\lambda I)^{-1}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The log posterior is $-\tfrac{1}{2\sigma^2}\|y-X\beta\|^2-\tfrac{1}{2\tau^2}\|\beta\|^2+\text{const}=-\tfrac{1}{2\sigma^2}\big(\|y-X\beta\|^2+\lambda\|\beta\|^2\big)+\text{const}$, a quadratic in $\beta$ with Hessian $-(X^\top X+\lambda I)/\sigma^2$. For a Gaussian the mean equals the mode, the ridge solution.</p>
</details>
<p>The penalty encodes the belief that coefficients are exchangeable and moderate — which is why standardizing first matters. Empirical Bayes estimates $\tau^2$ by maximizing the marginal likelihood $y\sim\N(0,\sigma^2I+\tau^2XX^\top)$, an alternative to cross-validation for choosing $\lambda$ (and the same structure as the <a href="#/t/mixed-models">random-effects</a> BLUP).</p>
<p><strong>Kernel form.</strong> The "push-through" identity $(X^\top X+\lambda I_p)^{-1}X^\top=X^\top(XX^\top+\lambda I_n)^{-1}$ gives
$$\hat\beta_\lambda=X^\top(XX^\top+\lambda I_n)^{-1}y,\qquad \hat f(x_0)=x_0^\top\hat\beta_\lambda=k_0^\top(K+\lambda I)^{-1}y,$$
with $K=XX^\top$ the Gram matrix of inner products. Only inner products appear, so replacing them by a kernel $k(x,x')$ gives kernel ridge regression (<a href="#/t/kernel-methods">kernel methods</a>); its predictions coincide with the posterior mean of a <a href="#/t/gaussian-processes">Gaussian process</a>. When $p\gg n$, solving the $n\times n$ system is also cheaper.</p>
<p><strong>Noise injection.</strong> Training least squares on inputs perturbed by independent noise of variance $\lambda/n$ per coordinate gives, in expectation over the noise, the objective $\|y-Xb\|^2+\lambda\|b\|^2$ (for centered, standardized design): ridge is what you get by insisting the fit be stable under small perturbations of the predictors.</p>`,
      },
      {
        title: 'Choosing λ',
        html: R`
<p>Because ridge is a linear smoother, every criterion of <a href="#/t/model-selection">model selection</a> is available in closed form along the entire path once the SVD is computed: with $s_j(\lambda)=d_j^2/(d_j^2+\lambda)$ and $z_j=u_j^\top y$,
$$\text{RSS}(\lambda)=\|y\|^2-\sum_j\big(2s_j-s_j^2\big)z_j^2,\quad \text{df}(\lambda)=\sum_js_j,\quad \text{GCV}(\lambda)=\frac{\text{RSS}(\lambda)/n}{(1-\text{df}(\lambda)/n)^2}.$$
Exact leave-one-out CV also needs only the diagonal of $H_\lambda$, $h_{ii}(\lambda)=\sum_jU_{ij}^2s_j$. A practical convention is to plot the criterion against $\text{df}(\lambda)$ rather than $\lambda$, because $\text{df}$ is interpretable across datasets.</p>
<div class="ex" data-name="When does ridge help most?">
<p>Ridge's advantage is largest when (i) the design is ill-conditioned (some small $d_j$), (ii) the signal is spread across many coefficients of similar size, and (iii) the signal-to-noise ratio is modest. If the true $\beta$ is sparse with a few large coefficients, the <a href="#/t/lasso">lasso</a> usually predicts better and is far easier to interpret. If the design is orthogonal and $n\gg p$, the gain is small: $\lambda^\star=p\sigma^2/\|\theta\|^2$ is then small relative to $d_j^2\propto n$.</p>
</div>`,
      },
      {
        title: 'Pitfalls and limits',
        html: R`
<div class="warn" data-name="Scale and coding dependence">
<p>Ridge is not equivariant under rescaling of predictors: multiplying a column by 10 makes its coefficient 10 times smaller and so 100 times cheaper to keep. Standardize (or choose scales deliberately). Dummy-coded factors raise the question of how to scale a 0/1 column; group-wise scalings or penalties on contrasts are common.</p>
</div>
<div class="warn" data-name="Inference">
<p>$\hat\beta_\lambda$ is biased, so an interval $\hat\beta_{\lambda,j}\pm1.96\,\widehat{\se}$ centered on it does not cover $\beta_j$ at the nominal rate even with the correct standard error; and when $\lambda$ is chosen from the data the variance formula $\sigma^2(X^\top X+\lambda I)^{-1}X^\top X(X^\top X+\lambda I)^{-1}$ ignores the selection. Ridge is a prediction tool; use it for inference only with care (e.g. the Bayesian interpretation, which gives credible intervals under the prior).</p>
</div>
<div class="warn" data-name="The ridgeless limit when p > n">
<p>As $\lambda\to0^+$ with $p>n$, $\hat\beta_\lambda\to X^+y$, the minimum-norm interpolator. It fits the training data perfectly, yet its test error can be surprisingly reasonable when $p\gg n$ — the "double descent" phenomenon discussed in <a href="#/t/regularization-deep">regularizing deep networks</a>. Near $p\approx n$, by contrast, the smallest nonzero $d_j$ approaches zero and the variance $\sigma^2\sum1/d_j^2$ explodes; a positive $\lambda$ is essential there.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that ridge shrinks the fitted values, not only the coefficients: $\|X\hat\beta_\lambda\|\le\|X\hat\beta_{\text{OLS}}\|$.`,
        a: R`$X\hat\beta_\lambda=\sum_ju_j\,s_j\,u_j^\top y$ with $s_j=d_j^2/(d_j^2+\lambda)\in(0,1]$, while $X\hat\beta_{\text{OLS}}=\sum_ju_ju_j^\top y$. By orthonormality, $\|X\hat\beta_\lambda\|^2=\sum s_j^2(u_j^\top y)^2\le\sum(u_j^\top y)^2$.` },
      { q: R`Two predictors have correlation $0.99$ (standardized, $n=100$). Which principal direction does ridge shrink hardest, and what happens to $\hat\beta_1-\hat\beta_2$ versus $\hat\beta_1+\hat\beta_2$?`,
        a: R`The eigenvalues of $X^\top X$ are about $99(1\pm0.99)$, i.e. $197$ along $(1,1)/\sqrt2$ and $\approx1$ along $(1,-1)/\sqrt2$. Ridge shrinks the $(1,-1)$ direction hardest, so $\hat\beta_1-\hat\beta_2$ is pulled strongly toward zero (the two coefficients are pulled together) while $\hat\beta_1+\hat\beta_2$, which is well determined, is barely changed.` },
      { q: R`Why does the Hoerl–Kennard theorem not give a usable rule for choosing $\lambda$?`,
        a: R`The guaranteed-improvement interval $(0,\sigma^2/\max_j\theta_j^2]$ and the optimum depend on the unknown $\theta=V^\top\beta$. Plugging in $\hat\theta_{\text{OLS}}$ is possible but noisy; in practice $\lambda$ is chosen by CV, GCV or empirical Bayes, and then the dominance guarantee no longer strictly holds.` },
      { q: R`What is $\text{df}(\lambda)$ when $X$ has orthonormal columns ($X^\top X=I_p$)?`,
        a: R`All $d_j=1$, so $\text{df}(\lambda)=p/(1+\lambda)$, and $\hat\beta_\lambda=\hat\beta_{\text{OLS}}/(1+\lambda)$: uniform proportional shrinkage.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'lasso',
    domain: 'regression',
    order: 100,
    title: 'The Lasso, Sparsity & the Elastic Net',
    subtitle: 'An ℓ1 penalty that selects variables and shrinks them in one convex problem',
    level: 3,
    sources: { aos: '13.6', other: 'Tibshirani (1996), JRSS-B 58; Efron et al. (2004), Ann. Statist. 32 (LARS); Hastie, Tibshirani & Wainwright, Statistical Learning with Sparsity (2015)' },
    prereqs: ['ridge-regression'],
    related: ['model-selection', 'cross-validation', 'undirected-graphical-models', 'multiple-testing', 'bayesian-inference'],
    tags: ['ℓ1 penalty', 'soft thresholding', 'KKT', 'coordinate descent', 'LARS', 'elastic net', 'sparsity'],
    summary: R`The lasso replaces ridge's squared penalty with the $\ell_1$ norm, $\lambda\sum_j|\beta_j|$. That single change makes the solution <em>sparse</em>: for large enough $\lambda$ many coefficients are exactly zero, so the lasso performs variable selection and shrinkage simultaneously while remaining a convex problem that can be solved for thousands of predictors. In an orthonormal design it is soft-thresholding of the least-squares coefficients; in general its solution is characterized by subgradient (KKT) conditions, its path in $\lambda$ is piecewise linear, and coordinate descent computes it with remarkable speed. With $p\gg n$ and a sparse truth it achieves near-oracle prediction error, but its selections can be unstable under correlation — the motivation for the elastic net.`,
    survey: {
      task: R`Sparse linear regression; variable selection with $p$ large (even $p\gg n$)`,
      assumptions: R`Standardized predictors; sparse or approximately sparse $\beta$; for selection consistency, an irrepresentable condition`,
      output: R`$\hat\beta_\lambda$ with many exact zeros; the full path over $\lambda$`,
      strengths: 'Convex; sparse and interpretable; scales to huge p; good prediction under sparsity',
      weaknesses: R`Biased toward zero for large effects; unstable selection with correlated predictors; at most $n$ nonzeros; hard post-selection inference`,
      complexity: R`Coordinate descent $O(np)$ per sweep; whole path $\approx$ cost of a few least-squares fits`,
    },
    sections: [
      {
        title: 'The ℓ1 penalty and why it produces zeros',
        html: R`
<p>With centered $y$ and standardized columns of $X$, the lasso solves
$$\hat\beta_\lambda\in\argmin_b\ \tfrac12\|y-Xb\|^2+\lambda\|b\|_1,\qquad \|b\|_1=\sum_{j=1}^p|b_j| .$$
Equivalently (by convex duality) minimize RSS subject to $\|b\|_1\le t$. The constraint region is a cross-polytope — a diamond in two dimensions — whose corners lie on the coordinate axes. An elliptical RSS contour expanding from the least-squares point typically first touches the diamond at a corner or an edge of lower dimension, where some coordinates are exactly zero. The ridge constraint, a sphere, has no corners: the contact point generically has every coordinate nonzero.</p>
<p>An equivalent one-dimensional explanation: near zero, the ridge penalty $\lambda b^2$ has slope $0$, so any nonzero data gradient moves the coefficient off zero. The lasso penalty $\lambda|b|$ has slope $\pm\lambda$, so a coefficient stays at zero until the data gradient exceeds $\lambda$ in magnitude.</p>`,
        lab: 'lasso-geometry',
      },
      {
        title: 'Orthonormal design: soft thresholding',
        html: R`
<p>Suppose $X^\top X=I$ and let $z=X^\top y=\hat\beta_{\text{OLS}}$. Then $\|y-Xb\|^2=\|y-Xz\|^2+\|z-b\|^2$, and every penalized problem separates into $p$ scalar problems in $b_j$ with data $z_j$.</p>
<div class="thm" data-kind="Proposition" data-name="Three estimators as thresholding rules">
<p>For $X^\top X=I$:</p>
<table>
<thead><tr><th>Objective</th><th>Solution</th><th>Rule</th></tr></thead>
<tbody>
<tr><td>$\tfrac12\|y-Xb\|^2+\lambda\|b\|_1$ (lasso)</td><td>$\hat\beta_j=\sgn(z_j)\big(|z_j|-\lambda\big)_+=:S_\lambda(z_j)$</td><td>soft threshold</td></tr>
<tr><td>$\tfrac12\|y-Xb\|^2+\tfrac\lambda2\|b\|^2$ (ridge)</td><td>$\hat\beta_j=z_j/(1+\lambda)$</td><td>proportional shrinkage</td></tr>
<tr><td>$\tfrac12\|y-Xb\|^2+\lambda\|b\|_0$ (best subset)</td><td>$\hat\beta_j=z_j\,\ind\big(|z_j|>\sqrt{2\lambda}\big)$</td><td>hard threshold</td></tr>
</tbody>
</table>
</div>
<details class="proof"><summary>Proof</summary>
<p>Lasso: minimize $g(b)=\tfrac12(b-z)^2+\lambda|b|$. For $b>0$, $g'(b)=b-z+\lambda=0$ gives $b=z-\lambda$, admissible iff $z>\lambda$; symmetrically $b=z+\lambda$ iff $z<-\lambda$. At $b=0$ the subdifferential is $[-z-\lambda,\,-z+\lambda]$, which contains $0$ iff $|z|\le\lambda$. Exactly one of these cases holds, giving $S_\lambda(z)$. Ridge: $g'(b)=b-z+\lambda b$. Subset: $b=z$ costs $\lambda$, $b=0$ costs $z^2/2$; keep iff $z^2/2>\lambda$.</p>
</details>
<p>Soft thresholding both selects (zeros below $\lambda$) and shrinks (survivors are moved toward zero by $\lambda$). The shrinkage of large coefficients is a bias the hard threshold avoids, but the hard threshold is discontinuous in the data, hence unstable, and its general-design version (best subset) is a combinatorial problem. The lasso is the convex problem closest to best subset — $\|b\|_1$ is the convex envelope of $\|b\|_0$ on the unit $\ell_\infty$ ball.</p>`,
      },
      {
        title: 'KKT conditions and what they imply',
        html: R`
<p>For general $X$, the objective is convex but not differentiable at zero coordinates. Optimality is characterized by the subgradient.</p>
<div class="thm" data-kind="Theorem" data-name="Lasso optimality (KKT) conditions">
<p>$\hat\beta$ solves the lasso problem if and only if
$$X^\top(y-X\hat\beta)=\lambda\hat s,\qquad \hat s_j\begin{cases}=\sgn(\hat\beta_j)&\hat\beta_j\ne0,\\ \in[-1,1]&\hat\beta_j=0.\end{cases}$$
In words: every active predictor has correlation with the residual exactly $\pm\lambda$, with the sign of its coefficient; every inactive predictor has correlation at most $\lambda$ in absolute value.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For a convex function, $\hat\beta$ is a minimizer iff $0$ belongs to the subdifferential at $\hat\beta$. The smooth part contributes gradient $-X^\top(y-X\hat\beta)$, and the subdifferential of $\lambda\|b\|_1$ is $\lambda$ times the set of vectors $s$ with $s_j=\sgn(b_j)$ on nonzero coordinates and $s_j\in[-1,1]$ on zeros. The subdifferential of the sum is the sum because one term is differentiable.</p>
</details>
<p>Consequences:</p>
<ul>
<li><strong>Entry point.</strong> $\hat\beta=0$ iff $\|X^\top y\|_\infty\le\lambda$. So $\lambda_{\max}=\max_j|x_j^\top y|$ is where the path starts.</li>
<li><strong>Uniqueness of the fit.</strong> All solutions share the same fitted values $X\hat\beta$ and the same $\|\hat\beta\|_1$. (If two solutions had different fits, their average would have strictly smaller RSS by strict convexity of $u\mapsto\|y-u\|^2$ and no larger penalty — a contradiction.) If the columns of $X$ are in general position (true with probability one for continuously distributed predictors), the solution itself is unique and has at most $\min(n,p)$ nonzeros (Tibshirani, 2013).</li>
<li><strong>Piecewise linearity.</strong> On an interval of $\lambda$ where the active set $A$ and signs $s_A$ are fixed, the KKT equations read $X_A^\top(y-X_A\beta_A)=\lambda s_A$, so
$$\hat\beta_A(\lambda)=(X_A^\top X_A)^{-1}\big(X_A^\top y-\lambda s_A\big),$$
linear in $\lambda$. The path is piecewise linear, with kinks where a variable enters (an inactive correlation reaches $\pm\lambda$) or leaves (an active coefficient hits zero).</li>
</ul>`,
      },
      {
        title: 'Computing the path: LARS and coordinate descent',
        html: R`
<p>Piecewise linearity means the whole path can be traced exactly by computing the breakpoints. That is <strong>least angle regression</strong> (LARS; Efron, Hastie, Johnstone & Tibshirani, 2004): start at $\beta=0$, move the coefficients of the active set in the equiangular direction that keeps their residual correlations tied and decreasing, and add a variable whenever its correlation catches up. With the modification "drop a variable when its coefficient crosses zero", LARS produces exactly the lasso path, at roughly the cost of a single least-squares fit when $p&lt;n$.</p>
<div class="algo" data-name="Cyclic coordinate descent (pathwise, with warm starts)">
<ol>
<li>Choose a decreasing grid $\lambda_{\max}=\lambda_1>\lambda_2>\dots>\lambda_K$ (typically log-spaced down to $10^{-3}\lambda_{\max}$).</li>
<li>For each $\lambda_k$, initialize at the solution for $\lambda_{k-1}$ (a warm start). Cycle over $j=1,\dots,p$:
$$\beta_j\leftarrow\frac{S_\lambda\big(x_j^\top r_{(j)}\big)}{\|x_j\|^2},\qquad r_{(j)}=y-\sum_{k\ne j}x_k\beta_k,$$
updating the residual incrementally after each change. Repeat until the coefficients stop changing.</li>
</ol>
</div>
<p>Each update exactly minimizes the objective in one coordinate (it is the orthonormal-case soft threshold applied to the partial residual). Because the nondifferentiable part is <em>separable</em> across coordinates, cyclic coordinate descent converges to a global minimizer (Tseng, 2001). Warm starts along a fine grid, together with "active-set" iterations that sweep only nonzero coordinates and screening rules that discard predictors certain to be zero, make this the fastest known method for large problems (the algorithm behind glmnet).</p>`,
        lab: 'lasso-path',
      },
      {
        title: 'What the lasso achieves: prediction, estimation, selection',
        html: R`
<p>Suppose $y=X\beta^\ast+\varepsilon$ with Gaussian noise, $\beta^\ast$ having $s$ nonzero entries, and columns scaled so $\|x_j\|^2=n$. Three different goals have three different conditions.</p>
<ul>
<li><strong>Prediction.</strong> With $\lambda\asymp\sigma\sqrt{n\log p}$ — just large enough to dominate $\|X^\top\varepsilon\|_\infty$, which is of that order by a union bound over $p$ Gaussian maxima — the in-sample prediction error $\|X(\hat\beta-\beta^\ast)\|^2/n$ is $O(\sigma^2s\log p/n)$ under a restricted-eigenvalue (compatibility) condition on $X$ (Bickel, Ritov & Tsybakov, 2009). Compare $\sigma^2s/n$ for an oracle that knows the support: the price for not knowing which $s$ of $p$ variables matter is only a $\log p$ factor.</li>
<li><strong>Estimation.</strong> Under the same condition, $\|\hat\beta-\beta^\ast\|_2^2=O(s\sigma^2\log p/n)$ with high probability, with constants depending on the restricted-eigenvalue constant.</li>
<li><strong>Selection.</strong> Recovering the exact support with correct signs requires much more: a minimum signal strength ("beta-min") and the <em>irrepresentable condition</em> $\big\|X_{S^c}^\top X_S(X_S^\top X_S)^{-1}\sgn(\beta_S^\ast)\big\|_\infty\le1-\eta$ (Zhao & Yu, 2006) — the irrelevant predictors must not be too well explained by the relevant ones. When it fails the lasso tends to include false positives that proxy for true variables.</li>
</ul>
<p><strong>Degrees of freedom.</strong> For fixed $\lambda$, the number of nonzero coefficients is an unbiased estimate of the lasso's degrees of freedom in the covariance-penalty sense (Zou, Hastie & Tibshirani, 2007). Surprisingly, the adaptive selection does not inflate df beyond the active-set size — shrinkage of the survivors exactly compensates. This makes $C_p$/AIC-type tuning straightforward, although CV is more common.</p>
<div class="warn" data-name="Bias and its fixes">
<p>Large coefficients are shrunk by about $\lambda$, so the lasso underestimates strong effects and, to compensate, tends to choose a small $\lambda$ that admits extra noise variables. The <em>relaxed lasso</em> refits (or partially refits) least squares on the selected set; the <em>adaptive lasso</em> uses weights $w_j=1/|\tilde\beta_j|^\gamma$ from an initial estimate, penalizing small coefficients more, and enjoys selection consistency under weaker conditions (Zou, 2006).</p>
</div>`,
      },
      {
        title: 'Correlated predictors and the elastic net',
        html: R`
<p>When predictors are highly correlated, the lasso tends to pick one of a group almost arbitrarily and ignore the rest; small perturbations of the data swap which one. In the extreme case of two identical columns, any split of a coefficient between them with matching signs has the same objective value, so the solution is not unique. And with $p>n$ the lasso can select at most $n$ variables.</p>
<div class="def" data-name="Elastic net (Zou & Hastie, 2005)">
<p>With standardized columns ($\tfrac1n\sum_ix_{ij}^2=1$),
$$\hat\beta=\argmin_b\ \frac1{2n}\|y-Xb\|^2+\lambda\Big(\alpha\|b\|_1+\frac{1-\alpha}{2}\|b\|^2\Big),\qquad \alpha\in[0,1],$$
interpolating between ridge ($\alpha=0$) and lasso ($\alpha=1$). The coordinate-descent update becomes $\beta_j\leftarrow S_{\lambda\alpha}\big(\tfrac1nx_j^\top r_{(j)}\big)\big/\big(1+\lambda(1-\alpha)\big)$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Grouping effect">
<p>For $\alpha<1$ and $\lambda>0$ the objective is strictly convex, so the elastic-net solution is unique; if $x_i=x_j$ then $\hat\beta_i=\hat\beta_j$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The ridge term makes the objective strictly convex, so the minimizer is unique. Swapping coordinates $i$ and $j$ of any $b$ leaves the objective unchanged when $x_i=x_j$. Applying the swap to the unique minimizer yields another minimizer, which must be the same point, so $\hat\beta_i=\hat\beta_j$.</p>
</details>
<p>More generally, highly correlated predictors receive similar coefficients and enter or leave the model together. The elastic net can select more than $n$ variables and is usually more stable than the lasso; the cost is an extra tuning parameter. When groups are known in advance (the dummies of a factor, genes in a pathway), the <em>group lasso</em> penalty $\sum_g\|b_g\|_2$ selects whole groups.</p>
<div class="warn" data-name="Inference after the lasso">
<p>Coefficients that survive the lasso are selected <em>and</em> shrunk using the same data, so naive standard errors from refitting OLS on the selected set are invalid (see <a href="#/t/model-selection">model selection</a>). Valid options include sample splitting, selective inference conditioned on the lasso's polyhedral selection event (Lee et al., 2016), and the debiased (desparsified) lasso for individual coefficients in high dimensions.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In an orthonormal design with $z=X^\top y=(3,\,-0.5,\,1.2)$ and $\lambda=1$, give the lasso, ridge ($\tfrac\lambda2\|b\|^2$ penalty), and best-subset ($\lambda\|b\|_0$) estimates.`,
        a: R`Lasso: $S_1(z)=(2,\,0,\,0.2)$. Ridge: $z/2=(1.5,\,-0.25,\,0.6)$. Best subset keeps $|z_j|>\sqrt2\approx1.414$: $(3,\,0,\,0)$.` },
      { q: R`Using the KKT conditions, find the smallest $\lambda$ at which the lasso solution is entirely zero.`,
        a: R`$\hat\beta=0$ requires $|x_j^\top(y-0)|\le\lambda$ for all $j$, so $\lambda\ge\lambda_{\max}=\|X^\top y\|_\infty$. Below it, the predictor with the largest $|x_j^\top y|$ enters first.` },
      { q: R`Why is the lasso path piecewise linear in $\lambda$, while the ridge path is not?`,
        a: R`On a stretch where the active set and signs are fixed, the KKT equations $X_A^\top(y-X_A\beta_A)=\lambda s_A$ are linear in $(\beta_A,\lambda)$, giving $\beta_A(\lambda)=(X_A^\top X_A)^{-1}(X_A^\top y-\lambda s_A)$. Ridge gives $(X^\top X+\lambda I)^{-1}X^\top y$, a rational function of $\lambda$ (in the SVD basis, factors $d_j/(d_j^2+\lambda)$).` },
      { q: R`Two predictors are identical copies. What do the lasso and the elastic net do?`,
        a: R`The lasso objective depends only on $\beta_1+\beta_2$ (when they share a sign), so any split is optimal — the solution is non-unique and algorithms pick one arbitrarily. The elastic net's ridge component makes the objective strictly convex and symmetric in the two coordinates, so its unique solution splits the coefficient equally.` },
      { q: R`Explain why the lasso's prediction error rate $s\log p/n$ is "near-oracle".`,
        a: R`An oracle told the true support would fit OLS on $s$ variables and get prediction error of order $\sigma^2s/n$. The lasso, without knowing the support, achieves $\sigma^2s\log p/n$: it pays only a logarithmic factor in the number of candidate variables, which is the price of controlling the maximum of $p$ noise correlations.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'mixed-models',
    domain: 'regression',
    order: 110,
    title: 'Hierarchical & Mixed-Effects Models',
    subtitle: 'Random effects, partial pooling, BLUP shrinkage, and REML',
    level: 3,
    sources: { cb: '§11.2', other: 'Robinson (1991), Statist. Sci. 6 (BLUP); Pinheiro & Bates, Mixed-Effects Models in S (2000); Gelman & Hill (2007)' },
    prereqs: ['anova', 'weighted-gls'],
    related: ['glmm-gee', 'bayesian-inference', 'hierarchical-mcmc', 'em-algorithm', 'ridge-regression', 'decision-theory'],
    tags: ['random effects', 'partial pooling', 'BLUP', 'REML', 'intraclass correlation', 'shrinkage'],
    summary: R`Data often come in groups — students in schools, measurements on patients, plots in fields. Treating every observation as independent understates uncertainty; fitting a separate parameter to each group wastes information and overfits small groups. A mixed-effects model treats the group effects as draws from a common distribution with variance $\tau^2$. This induces a correlated-error linear model, estimated by GLS, and yields predictions of each group's effect that <em>partially pool</em>: each group's own mean is shrunk toward the overall mean by a factor that depends on how much data the group has and on the ratio $\sigma^2/\tau^2$. The resulting best linear unbiased predictor (BLUP) is the frequentist face of the Bayesian posterior mean and a close relative of ridge regression. Variance components are estimated by REML, which corrects maximum likelihood's downward bias.`,
    survey: {
      task: 'Regression with grouped / clustered / repeated-measures data; predicting group-level effects',
      assumptions: R`Group effects exchangeable, normal, independent of covariates; normal within-group errors`,
      output: R`$\hat\beta$ (GLS), $\hat\tau^2,\hat\sigma^2$ (REML), BLUPs $\hat b_i$ with shrinkage $n_i\tau^2/(n_i\tau^2+\sigma^2)$`,
      strengths: 'Correct SEs for clustered data; borrows strength across groups; handles unbalanced designs',
      weaknesses: 'Bias if random effects correlate with covariates; few groups → poor τ² estimates; df for tests approximate',
      complexity: R`Sparse Cholesky per likelihood evaluation; random intercepts $O(n)$`,
    },
    sections: [
      {
        title: 'Why groups need special treatment',
        html: R`
<p>Consider $G$ groups with $n_i$ observations in group $i$. Three ways to estimate the group means $\mu_i$:</p>
<div class="cols">
<div><h4>Complete pooling</h4><p>Ignore groups; estimate every $\mu_i$ by the grand mean. Low variance, but biased whenever groups truly differ, and its standard error pretends all $N$ observations are independent.</p></div>
<div><h4>No pooling</h4><p>Estimate each $\mu_i$ by its own mean $\bar y_i$ (fixed effects, one-way ANOVA). Unbiased, but variance $\sigma^2/n_i$ is huge for small groups; extreme estimates are mostly noise.</p></div>
<div><h4>Partial pooling</h4><p>A weighted compromise, with the weight on $\bar y_i$ increasing in $n_i$ and in the between-group variance. This is what a random-effects model delivers.</p></div>
</div>
<p>Even when only the overall mean or a covariate effect is of interest, ignoring grouping is dangerous. In a balanced design with $m$ observations per group and <strong>intraclass correlation</strong> $\rho=\tau^2/(\tau^2+\sigma^2)$, the variance of the overall mean is inflated by the <em>design effect</em> $1+(m-1)\rho$ relative to the independent-data formula. With $m=30$ and $\rho=0.1$ that is $3.9$: naive standard errors are half what they should be.</p>`,
      },
      {
        title: 'The linear mixed model',
        html: R`
<div class="def" data-name="Random-intercept model">
<p>$$y_{ij}=x_{ij}^\top\beta+b_i+\varepsilon_{ij},\qquad b_i\iid\N(0,\tau^2),\quad \varepsilon_{ij}\iid\N(0,\sigma^2),\quad b\indep\varepsilon .$$
$\beta$ are <em>fixed effects</em> (population-level coefficients); $b_i$ are <em>random effects</em> (group deviations).</p>
</div>
<p>Integrating out $b_i$, the observations in group $i$ are jointly normal with
$$y_i\sim\N\big(X_i\beta,\ V_i\big),\qquad V_i=\sigma^2I_{n_i}+\tau^2\mathbf 1\mathbf 1^\top,$$
independent across groups: an exchangeable correlation $\rho$ within a group. The general form is
$$y=X\beta+Zb+\varepsilon,\qquad b\sim\N(0,G),\quad \varepsilon\sim\N(0,R),\qquad\text{so}\qquad y\sim\N(X\beta,\ V),\quad V=ZGZ^\top+R,$$
which covers random slopes ($b_i$ a vector with a covariance matrix), nested and crossed effects, and structured $R$ (e.g. AR(1) within subject).</p>
<p>For fixed variance parameters, the marginal model is a <a href="#/t/weighted-gls">GLS</a> problem, so
$$\hat\beta=(X^\top V^{-1}X)^{-1}X^\top V^{-1}y,\qquad \Var\hat\beta=(X^\top V^{-1}X)^{-1}.$$
In the intercept-only random-intercept model this gives $\hat\mu=\sum_iw_i\bar y_i/\sum_iw_i$ with $w_i=1/(\tau^2+\sigma^2/n_i)$: large groups get more weight, but with diminishing returns — once $\sigma^2/n_i\ll\tau^2$, a group counts as one draw of $b_i$ no matter how big it is.</p>`,
      },
      {
        title: 'BLUP: predicting the random effects',
        html: R`
<p>The $b_i$ are random, so we <em>predict</em> them rather than estimate them. If $\beta$ and the variances were known, the best predictor under squared error is the conditional mean (<a href="#/t/conditional-expectation">conditional expectation</a>), and for jointly Gaussian $(b,y)$,
$$\E[b\mid y]=GZ^\top V^{-1}(y-X\beta).$$</p>
<div class="thm" data-kind="Proposition" data-name="Shrinkage in the random-intercept model">
<p>For the random-intercept model, with $r_i=\bar y_i-\bar x_i^\top\beta$ the group's mean residual,
$$\E[b_i\mid y]=\underbrace{\frac{n_i\tau^2}{n_i\tau^2+\sigma^2}}_{1-B_i}\,r_i,\qquad \Var(b_i\mid y)=\frac{\tau^2\,\sigma^2/n_i}{\tau^2+\sigma^2/n_i} .$$
The predicted group mean $\bar x_i^\top\beta+\E[b_i\mid y]=B_i\,\bar x_i^\top\beta+(1-B_i)\,\bar y_i$ is a weighted average of the model prediction and the group's own mean, with shrinkage factor $B_i=\dfrac{\sigma^2/n_i}{\tau^2+\sigma^2/n_i}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Only group $i$'s data are informative about $b_i$, and given $\beta$ they enter through the sufficient statistic $r_i=b_i+\bar\varepsilon_i$, where $\bar\varepsilon_i\sim\N(0,\sigma^2/n_i)$ is independent of $b_i$. So $(b_i,r_i)$ is bivariate normal with $\Var b_i=\tau^2$, $\Var r_i=\tau^2+\sigma^2/n_i$, $\Cov(b_i,r_i)=\tau^2$. The conditional mean is $\frac{\tau^2}{\tau^2+\sigma^2/n_i}r_i$ and the conditional variance $\tau^2-\frac{\tau^4}{\tau^2+\sigma^2/n_i}$, which simplify to the stated forms.</p>
</details>
<p>Small groups ($\sigma^2/n_i\gg\tau^2$) are shrunk almost entirely to the model prediction; large groups keep their own mean. Comparing mean squared errors for predicting $\mu_i=\bar x_i^\top\beta+b_i$ with $\beta$ known: no pooling has $\sigma^2/n_i$; partial pooling has $\Var(b_i\mid y)=(1-B_i)\sigma^2/n_i$, strictly smaller. The gain is largest exactly for the small groups where no pooling is worst.</p>
<div class="thm" data-kind="Theorem" data-name="BLUP (Henderson; Goldberger, 1962)">
<p>Replace $\beta$ by its GLS estimate: $\hat b=GZ^\top V^{-1}(y-X\hat\beta)$. Among all predictors of $b$ (and of any $c^\top\beta+d^\top b$) that are linear in $y$ and unbiased ($\E[\tilde b-b]=0$ for all $\beta$), $\hat b$ has the smallest mean squared prediction error. Equivalently, $(\hat\beta,\hat b)$ solve Henderson's mixed-model equations
$$\begin{pmatrix}X^\top R^{-1}X&X^\top R^{-1}Z\\ Z^\top R^{-1}X&Z^\top R^{-1}Z+G^{-1}\end{pmatrix}\begin{pmatrix}\hat\beta\\ \hat b\end{pmatrix}=\begin{pmatrix}X^\top R^{-1}y\\ Z^\top R^{-1}y\end{pmatrix},$$
which are the normal equations for minimizing $(y-X\beta-Zb)^\top R^{-1}(y-X\beta-Zb)+b^\top G^{-1}b$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>A linear unbiased predictor is $\tilde b=Ay$ with $AX=0$. Write $\tilde b-b=(\tilde b-\hat b)+(\hat b-b)$. One checks that $\hat b-b$ is uncorrelated with every linear function $Cy$ having $CX=0$ — the same orthogonality as in the Gauss–Markov proof, now in the $V^{-1}$ inner product. Hence the MSE of $\tilde b$ is that of $\hat b$ plus a positive semidefinite term. For the mixed-model equations: the penalized criterion is $-2\log$ of the joint density $f(y\mid b)f(b)$ up to constants; eliminating $b$ from its normal equations (using the Woodbury identity $V^{-1}=R^{-1}-R^{-1}Z(Z^\top R^{-1}Z+G^{-1})^{-1}Z^\top R^{-1}$) recovers the GLS equation for $\beta$ and the BLUP for $b$.</p>
</details>
<p>The penalized-least-squares form shows the kinship with <a href="#/t/ridge-regression">ridge</a>: random effects are coefficients with a quadratic penalty $b^\top G^{-1}b$, i.e. ridge with $\lambda=\sigma^2/\tau^2$ on the group dummies, fixed effects unpenalized. And with a flat prior on $\beta$, $\hat b$ is the posterior mean of a <a href="#/t/bayesian-inference">Bayesian</a> hierarchical model given the variances.</p>`,
        lab: 'mixed-models-pooling',
      },
      {
        title: 'Estimating variance components: ML, REML and ANOVA',
        html: R`
<p>The marginal log-likelihood of $\theta=(\tau^2,\sigma^2,\dots)$ after profiling out $\beta$ is
$$\ell(\theta)=-\tfrac12\Big[\log|V_\theta|+(y-X\hat\beta_\theta)^\top V_\theta^{-1}(y-X\hat\beta_\theta)\Big]+\text{const}.$$
Its maximizer underestimates variances because it ignores the degrees of freedom spent estimating $\beta$ — the familiar $\text{RSS}/n$ bias, generalized.</p>
<div class="def" data-name="Restricted (residual) maximum likelihood (Patterson & Thompson, 1971)">
<p>Base the likelihood on $n-p$ linearly independent <em>error contrasts</em> $K^\top y$ with $K^\top X=0$, whose distribution $\N(0,K^\top V_\theta K)$ does not involve $\beta$. Up to a constant not depending on the choice of $K$ (Harville, 1974),
$$\ell_R(\theta)=-\tfrac12\Big[\log|V_\theta|+\log|X^\top V_\theta^{-1}X|+(y-X\hat\beta_\theta)^\top V_\theta^{-1}(y-X\hat\beta_\theta)\Big].$$</p>
</div>
<p>The extra $\log|X^\top V^{-1}X|$ term is the correction. For i.i.d. normal data REML gives $\sum(y_i-\bar y)^2/(n-1)$ rather than $/n$; in a balanced one-way random-effects model REML coincides with the ANOVA (method-of-moments) estimators whenever the latter are positive. Those ANOVA estimators use $\E\,\text{MSW}=\sigma^2$ and $\E\,\text{MSB}=\sigma^2+m\tau^2$ (see <a href="#/t/anova">ANOVA</a>), giving $\hat\tau^2=(\text{MSB}-\text{MSW})/m$, which can be negative and is then truncated at zero.</p>
<p>Computation: Newton or Fisher scoring on $\ell_R$, the <a href="#/t/em-algorithm">EM algorithm</a> treating $b$ as missing data, or (as in lme4) profiling everything except the relative covariance factor and solving penalized least squares with a sparse Cholesky factorization.</p>
<div class="warn" data-name="Testing a variance component">
<p>$H_0:\tau^2=0$ puts the parameter on the boundary of its space, so the likelihood-ratio statistic is <em>not</em> asymptotically $\chi^2_1$. For a single variance component with many independent groups its null limit is the mixture $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ (Self & Liang, 1987); using $\chi^2_1$ halves the true size. Equivalently, halve the naive $p$-value. The exact $F$-test from the one-way ANOVA is available in balanced designs.</p>
</div>`,
      },
      {
        title: 'Random slopes, fixed versus random, and the Mundlak device',
        html: R`
<p>A random-slope model lets each group have its own coefficient on a covariate: $y_{ij}=\beta_0+\beta_1x_{ij}+b_{0i}+b_{1i}x_{ij}+\varepsilon_{ij}$ with $(b_{0i},b_{1i})\sim\N(0,G)$. Within-group correlation now depends on $x$, and the BLUP shrinks each group's fitted line toward the population line — strongly for groups with few points or little spread in $x$.</p>
<p><strong>When are effects "random"?</strong> Not when they are uninteresting, but when the groups can be regarded as exchangeable draws from a population about which we want to generalize, and when we are willing to assume the random effects are <em>independent of the covariates</em>. That last assumption bites:</p>
<div class="warn" data-name="Random effects correlated with covariates">
<p>If $b_i$ is correlated with $\bar x_i$ (e.g. better schools attract richer students), the random-effects estimate of $\beta$ mixes within-group and between-group associations and is biased for the within-group effect. The <em>fixed-effects</em> (within) estimator, which uses group dummies and so identifies $\beta$ only from variation within groups, is unaffected. The <strong>Mundlak device</strong> adds the group means $\bar x_i$ as extra covariates in the random-effects model; the coefficient on $x_{ij}$ then equals the within estimator, while the model retains partial pooling of intercepts. The Hausman test compares the two estimators.</p>
</div>`,
      },
      {
        title: 'Practical pitfalls',
        html: R`
<ul>
<li><strong>Few groups.</strong> $\tau^2$ is estimated from roughly $G$ group effects; with $G<10$ it is very uncertain, often estimated at exactly zero ("singular fit"), and the BLUP shrinkage is then complete. Consider a Bayesian model with a weakly informative prior on $\tau$ (<a href="#/t/hierarchical-mcmc">hierarchical MCMC</a>).</li>
<li><strong>Degrees of freedom for fixed effects.</strong> Wald $t$-tests with $\hat V$ plugged in are not exactly $t$; small-sample approximations (Satterthwaite, Kenward–Roger) estimate effective denominator df, which can be close to $G$ rather than $N$ for group-level covariates.</li>
<li><strong>BLUPs are not estimates of fixed quantities.</strong> Their spread is smaller than that of the true $b_i$ ($\Var\hat b_i=\tau^2(1-B_i)<\tau^2$ when $\beta$ is known), so a histogram of BLUPs understates between-group variation, and ranking groups by BLUP differs from ranking by raw means — by design.</li>
<li><strong>Non-normal responses.</strong> For counts and binary outcomes, the analogous models are <a href="#/t/glmm-gee">GLMMs</a>, whose fixed effects have a <em>conditional</em> (subject-specific) interpretation that differs from GEE's population-averaged coefficients.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Group A has 2 observations with mean 90; group B has 200 with mean 70. The population mean is 75, $\sigma^2=100$, $\tau^2=25$. Compute both BLUP-predicted group means.`,
        a: R`$B_A=\frac{100/2}{25+100/2}=\frac{50}{75}=\tfrac23$, so the prediction is $\tfrac23\cdot75+\tfrac13\cdot90=80$. $B_B=\frac{0.5}{25.5}\approx0.0196$, so $\approx0.0196\cdot75+0.9804\cdot70\approx70.1$. The small group is pulled most of the way to 75; the large group barely moves.` },
      { q: R`Why does ML underestimate variance components, and how does REML fix it?`,
        a: R`ML evaluates the likelihood at $\hat\beta$ as if it were the true $\beta$, so residuals are too small by the $p$ degrees of freedom spent fitting $\beta$ (the $\text{RSS}/n$ bias). REML uses only error contrasts $K^\top y$ with $K^\top X=0$, whose distribution does not depend on $\beta$; its likelihood adds $\tfrac12\log|X^\top V^{-1}X|$ and yields, e.g., division by $n-1$ in the i.i.d. case.` },
      { q: R`Show that the BLUP in a random-intercept model is a ridge estimate.`,
        a: R`Henderson's equations minimize $\|y-X\beta-Zb\|^2/\sigma^2+\|b\|^2/\tau^2$ when $R=\sigma^2I$, $G=\tau^2I$; multiplying by $\sigma^2$, this is least squares in $(\beta,b)$ with ridge penalty $\lambda\|b\|^2$, $\lambda=\sigma^2/\tau^2$, on the group-dummy coefficients only.` },
      { q: R`A researcher tests $\tau^2=0$ with a likelihood-ratio statistic of $2.9$ and reports $p=0.089$ from $\chi^2_1$. What is the better $p$-value?`,
        a: R`Under $H_0$ the statistic is asymptotically $\tfrac12\chi^2_0+\tfrac12\chi^2_1$ (boundary parameter), so $p\approx\tfrac12\P(\chi^2_1>2.9)\approx0.044$.` },
    ],
  });
})();
