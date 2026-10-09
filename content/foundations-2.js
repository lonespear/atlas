/* Foundations II: inequalities, information, convergence, limit theorems, sampling distributions, order statistics. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'inequalities',
    domain: 'foundations',
    order: 100,
    title: 'Probability Inequalities',
    subtitle: 'Distribution-free tail bounds, from Markov to Hoeffding, and the convexity inequalities behind them',
    level: 2,
    sources: { aos: 'Ch. 4', cb: '§3.6, §4.7', other: 'Boucheron, Lugosi & Massart, Concentration Inequalities (2013); Vershynin, High-Dimensional Probability (2018)' },
    prereqs: ['expectation'],
    related: ['mgf', 'lln', 'entropy-kl', 'vc-theory', 'empirical-cdf'],
    tags: ['tail bound', 'concentration', 'convexity', 'finite-sample'],
    summary: R`A probability inequality trades knowledge of a distribution for a guarantee that holds for <em>every</em> distribution satisfying a few weak conditions — a known mean, a bounded variance, a bounded range. Markov and Chebyshev need only moments and give polynomial tails; the Chernoff method exploits the moment generating function and gives exponential tails; Hoeffding and Bernstein turn this into finite-sample guarantees for averages. Jensen, Cauchy–Schwarz and Hölder are the convexity tools that make the proofs (and much of the rest of statistics) work. Together they are how you prove that an estimator is close to its target with high probability, at a specific $n$, not just in the limit.`,
    survey: {
      task: 'Bound tail probabilities and expectations without knowing the distribution',
      assumptions: R`Nonnegativity (Markov); finite variance (Chebyshev); finite MGF (Chernoff); bounded range (Hoeffding); bounded range + variance (Bernstein)`,
      output: R`Upper bounds on $\P(X\ge t)$ or $\P(|\bar X_n-\mu|\ge\epsilon)$; sample-size guarantees $n(\epsilon,\delta)$`,
      strengths: 'Non-asymptotic, distribution-free, composable via union bounds; the backbone of learning theory',
      weaknesses: 'Conservative constants; Markov/Chebyshev can be loose by orders of magnitude; exponential bounds need light tails',
      complexity: 'Closed form, or a one-dimensional convex optimisation (Chernoff)',
    },
    sections: [
      {
        title: 'Why bound instead of compute?',
        html: R`
<p>Suppose you estimate a click-through probability $p$ by the fraction $\bar X_n$ of $n$ users who click. You would like to say "with probability at least $0.95$, $|\bar X_n - p| \le 0.03$". The <a href="#/t/clt">central limit theorem</a> gives an approximate answer, but it is a statement about $n\to\infty$; it says nothing certain about $n=500$, and its error depends on the unknown $p$. A <strong>probability inequality</strong> gives a statement that is <em>true</em> at every $n$ for every distribution in a broad class, at the price of some looseness.</p>
<p>That trade — exactness for universality — is the defining feature of this chapter. Three questions organise it:</p>
<ol>
<li><strong>What do you know?</strong> Only a mean? A variance? A bounded range? The full moment generating function? More information buys sharper bounds.</li>
<li><strong>How fast does the bound decay?</strong> Polynomially ($t^{-1}$, $t^{-2}$) or exponentially ($e^{-ct^2}$)? For averages, this decides whether the required sample size scales like $1/\delta$ or like $\log(1/\delta)$ in the failure probability $\delta$.</li>
<li><strong>Can it be attained?</strong> A bound that is attained by some distribution in its class cannot be improved without adding assumptions.</li>
</ol>
<p>The second family of inequalities in this topic — Jensen, Cauchy–Schwarz, Hölder — bound <em>expectations</em> rather than probabilities. They are the algebraic engine behind the tail bounds, behind the nonnegativity of <a href="#/t/entropy-kl">KL divergence</a>, behind the Cramér–Rao bound and behind countless consistency proofs.</p>`,
      },
      {
        title: 'Markov, Chebyshev and Cantelli',
        html: R`
<div class="thm" data-name="Markov's inequality">
<p>If $X \ge 0$ and $t > 0$, then $\displaystyle \P(X \ge t) \le \frac{\E X}{t}.$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Pointwise, $X \ge t\,\ind(X \ge t)$: if $X\ge t$ the right side is $t \le X$, otherwise it is $0 \le X$. Take expectations: $\E X \ge t\,\P(X\ge t)$.</p>
</details>
<p>The proof is one line, and it contains the whole method: <em>dominate an indicator by a function whose expectation you know</em>. Every inequality in this section is Markov applied to a cleverly chosen transform $\psi(X)$ with $\psi$ nonnegative and nondecreasing:
$$\P(X \ge t) \le \P(\psi(X) \ge \psi(t)) \le \frac{\E\psi(X)}{\psi(t)}.$$</p>
<div class="thm" data-name="Chebyshev's inequality">
<p>If $\E X = \mu$ and $\Var X = \sigma^2 < \infty$, then for $t > 0$, $\displaystyle \P(|X-\mu| \ge t) \le \frac{\sigma^2}{t^2}$. Equivalently $\P(|X-\mu| \ge k\sigma) \le 1/k^2$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Apply Markov to the nonnegative variable $(X-\mu)^2$ at level $t^2$: $\P(|X-\mu|\ge t) = \P((X-\mu)^2 \ge t^2) \le \E(X-\mu)^2/t^2$.</p>
</details>
<div class="thm" data-kind="Proposition" data-name="Cantelli's one-sided inequality">
<p>Under the same conditions, for $t>0$, $\displaystyle \P(X - \mu \ge t) \le \frac{\sigma^2}{\sigma^2 + t^2}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For any $u \ge 0$, $\P(X-\mu \ge t) \le \P\big((X-\mu+u)^2 \ge (t+u)^2\big) \le \dfrac{\sigma^2 + u^2}{(t+u)^2}$, using $\E(X-\mu+u)^2 = \sigma^2+u^2$. The right side is minimised at $u = \sigma^2/t$, giving $\sigma^2/(\sigma^2+t^2)$.</p>
</details>
<div class="ex" data-name="These bounds are sharp">
<p>Markov is attained by the two-point law $X \in \{0, t\}$ with $\P(X=t) = \E X/t$. Chebyshev is attained by the three-point law putting mass $1/(2k^2)$ at $\mu \pm k\sigma$ and the rest at $\mu$. So without further assumptions, neither can be improved. The looseness you see on familiar distributions (the normal puts $4.6\%$ outside $2\sigma$, Chebyshev allows $25\%$) is the price of covering these extreme cases.</p>
</div>
<p>Higher moments help: $\P(|X-\mu|\ge t) \le \E|X-\mu|^k/t^k$ for any $k$ with a finite moment. When all moments exist, choosing the best $k$ is surprisingly powerful — see the moment bound below.</p>`,
        lab: 'inequalities-tails',
      },
      {
        title: 'The Chernoff method',
        html: R`
<p>Apply Markov to $\psi(x) = e^{\lambda x}$ for any $\lambda > 0$, then optimise over $\lambda$:
$$\P(X \ge t) \le \inf_{\lambda > 0} e^{-\lambda t}\,\E e^{\lambda X} = \exp\Big(-\sup_{\lambda>0}\big[\lambda t - \log M_X(\lambda)\big]\Big).$$
The exponent is the <strong>Legendre transform</strong> (convex conjugate) of the cumulant generating function $K(\lambda) = \log M_X(\lambda)$. The method needs the <a href="#/t/mgf">MGF</a> to be finite for some $\lambda>0$ — that is, a right tail that decays at least exponentially.</p>
<div class="ex" data-name="Gaussian and exponential tails">
<p>For $Z \sim \N(0,1)$, $K(\lambda) = \lambda^2/2$, so $\sup_\lambda(\lambda t - \lambda^2/2) = t^2/2$ and $\P(Z\ge t) \le e^{-t^2/2}$. The true tail is $\approx \phi(t)/t$, so the Chernoff bound is off only by the polynomial factor $t\sqrt{2\pi}$ — exponentially tight.</p>
<p>For $X \sim \text{Exp}(1)$, $K(\lambda) = -\log(1-\lambda)$ for $\lambda < 1$; the optimum is $\lambda = 1 - 1/t$ and the bound is $t\,e^{1-t}$ for $t > 1$, again matching the true tail $e^{-t}$ up to a factor linear in $t$.</p>
</div>
<p>The Chernoff bound is at its best for <em>sums</em>, because MGFs of independent sums factor: if $S_n = \sum X_i$ with $X_i$ IID, then $K_{S_n}(\lambda) = nK(\lambda)$ and
$$\P(\bar X_n \ge a) \le \exp\big(-n\,I(a)\big), \qquad I(a) = \sup_{\lambda}\big[\lambda a - K(\lambda)\big].$$
Cramér's theorem says this is the right exponent: $\frac1n\log \P(\bar X_n \ge a) \to -I(a)$ for $a > \mu$. For Bernoulli($p$) data, $I(a) = \KL(\text{Ber}(a)\,\|\,\text{Ber}(p))$ — the <a href="#/t/entropy-kl">KL divergence</a>, which is where information theory and large deviations meet.</p>
<div class="thm" data-kind="Proposition" data-name="Moments beat the MGF">
<p>If $X \ge 0$ and $t>0$, then $\displaystyle \min_{k = 0,1,2,\dots} \frac{\E X^k}{t^k} \le \inf_{\lambda > 0} \E e^{\lambda(X - t)}.$ The best polynomial-moment Markov bound is never worse than the Chernoff bound.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $m = \min_k \E X^k/t^k$. By monotone convergence, $\E e^{\lambda X} = \sum_k \lambda^k \E X^k/k! \ge m\sum_k (\lambda t)^k/k! = m\,e^{\lambda t}$. Divide by $e^{\lambda t}$ and take the infimum over $\lambda$.</p>
</details>
<p>Chernoff's advantage is therefore not sharpness but <em>tractability</em>: MGFs multiply under independence, moments do not.</p>`,
      },
      {
        title: 'Hoeffding and Bernstein: concentration of averages',
        html: R`
<p>To use the Chernoff method on averages without knowing the distribution, we need a universal bound on the MGF. Boundedness is enough.</p>
<div class="thm" data-kind="Lemma" data-name="Hoeffding's lemma">
<p>If $a \le X \le b$ almost surely and $\E X = 0$, then $\E e^{\lambda X} \le \exp\!\big(\lambda^2 (b-a)^2/8\big)$ for all $\lambda\in\R$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By convexity of $x\mapsto e^{\lambda x}$ on $[a,b]$, $e^{\lambda x} \le \frac{b-x}{b-a}e^{\lambda a} + \frac{x-a}{b-a}e^{\lambda b}$. Taking expectations and using $\E X=0$: $\E e^{\lambda X} \le \frac{b}{b-a}e^{\lambda a} - \frac{a}{b-a}e^{\lambda b}$. Write $\theta = -a/(b-a) \in [0,1]$ and $u = \lambda(b-a)$; the right side equals $e^{\varphi(u)}$ with $\varphi(u) = -\theta u + \log(1-\theta+\theta e^u)$. Then $\varphi(0) = 0$, $\varphi'(0) = 0$, and $\varphi''(u) = \rho(1-\rho)$ where $\rho = \theta e^u/(1-\theta+\theta e^u) \in [0,1]$, so $\varphi'' \le 1/4$. By Taylor's theorem with remainder, $\varphi(u) \le u^2/8$.</p>
</details>
<div class="thm" data-name="Hoeffding's inequality (1963)">
<p>Let $X_1,\dots,X_n$ be independent with $a_i \le X_i \le b_i$. Then for $t>0$,
$$\P\Big(\sum_{i=1}^n (X_i - \E X_i) \ge t\Big) \le \exp\!\Big(-\frac{2t^2}{\sum_i (b_i-a_i)^2}\Big).$$
In particular, if all $X_i \in [a,b]$, then $\P(|\bar X_n - \mu| \ge \epsilon) \le 2\exp\!\big(-2n\epsilon^2/(b-a)^2\big)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Chernoff plus independence plus the lemma: $\P(\sum(X_i-\E X_i)\ge t) \le e^{-\lambda t}\prod_i \E e^{\lambda(X_i-\E X_i)} \le \exp\big(-\lambda t + \lambda^2 \sum(b_i-a_i)^2/8\big)$. Minimising the quadratic at $\lambda = 4t/\sum(b_i-a_i)^2$ gives the result. For the average put $t = n\epsilon$; for the two-sided version apply the same bound to $-X_i$ and add.</p>
</details>
<div class="ex" data-name="How many users?">
<p>To guarantee $\P(|\bar X_n - p| \ge 0.03) \le 0.05$ for Bernoulli data:</p>
<ul>
<li><strong>Chebyshev</strong> with $\Var X \le 1/4$: $n \ge 1/(4\epsilon^2\delta) = 5{,}556$.</li>
<li><strong>Hoeffding</strong>: $n \ge \log(2/\delta)/(2\epsilon^2) = 2{,}050$.</li>
<li><strong>CLT heuristic</strong> (not a guarantee): $n \approx z_{0.975}^2/(4\epsilon^2) = 1{,}068$.</li>
</ul>
<p>Hoeffding's dependence on $\delta$ is $\log(1/\delta)$ rather than $1/\delta$. This matters enormously when you need many guarantees simultaneously: a union bound over $m$ events replaces $\delta$ by $\delta/m$, which costs Hoeffding only an additive $\log m$. That is the mechanism behind uniform convergence in <a href="#/t/vc-theory">learning theory</a> and the DKW inequality for the <a href="#/t/empirical-cdf">empirical CDF</a>.</p>
</div>
<p>Hoeffding uses only the range, so it is loose when the variance is much smaller than $(b-a)^2/4$ — e.g. Bernoulli($p$) with small $p$. <strong>Bernstein's inequality</strong> repairs this: if $|X_i - \E X_i| \le b$ and $\frac1n\sum\Var X_i = \sigma^2$, then
$$\P(\bar X_n - \mu \ge t) \le \exp\!\Big(-\frac{n t^2}{2\sigma^2 + 2bt/3}\Big).$$
For small $t$ this behaves like the Gaussian $e^{-nt^2/(2\sigma^2)}$ — the CLT rate with the true variance — and for large $t$ it degrades gracefully to $e^{-3nt/(2b)}$. The second lab compares all of these with the exact binomial tail.</p>`,
        lab: 'inequalities-concentration',
      },
      {
        title: 'Jensen and the convexity inequalities',
        html: R`
<div class="thm" data-name="Jensen's inequality">
<p>If $g$ is convex and $X$ and $g(X)$ are integrable, then $g(\E X) \le \E g(X)$. If $g$ is concave the inequality reverses. If $g$ is strictly convex, equality holds iff $X$ is almost surely constant.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\mu = \E X$. A convex function lies above a supporting line at every interior point of its domain: there is a $c$ with $g(x) \ge g(\mu) + c(x-\mu)$ for all $x$. Substitute $X$ and take expectations: $\E g(X) \ge g(\mu) + c\cdot 0$. For strict convexity the supporting-line inequality is strict for $x\neq\mu$, so equality forces $\P(X=\mu) = 1$.</p>
</details>
<p>Jensen produces a remarkable number of standard facts:</p>
<ul>
<li>$\E X^2 \ge (\E X)^2$, i.e. $\Var X \ge 0$ ($g(x) = x^2$).</li>
<li>$\E\log X \le \log \E X$: the geometric mean is at most the arithmetic mean.</li>
<li>$\KL(P\|Q) \ge 0$, by applying Jensen to $-\log$ — the foundation of <a href="#/t/entropy-kl">information theory</a>, of the consistency of the <a href="#/t/mle">MLE</a> and of the EM algorithm's monotonicity.</li>
<li>The sample standard deviation is biased downward: $\E S = \E\sqrt{S^2} \le \sqrt{\E S^2} = \sigma$, with strict inequality unless $S$ is degenerate.</li>
<li>$1/\bar X$ overestimates $1/\mu$ on average when $\bar X > 0$, since $1/x$ is convex on $(0,\infty)$.</li>
</ul>
<div class="thm" data-name="Cauchy–Schwarz">
<p>$(\E XY)^2 \le \E X^2 \,\E Y^2$, with equality iff $X$ and $Y$ are linearly dependent almost surely. Consequently $|\Corr(X,Y)| \le 1$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For every real $s$, $0 \le \E(sX - Y)^2 = s^2\E X^2 - 2s\E XY + \E Y^2$. A nonnegative quadratic in $s$ has nonpositive discriminant: $4(\E XY)^2 - 4\E X^2\E Y^2 \le 0$. Equality means the quadratic has a real root $s_0$, i.e. $\E(s_0X-Y)^2 = 0$, so $Y = s_0X$ a.s. Apply to centred variables for the correlation statement.</p>
</details>
<p><strong>Hölder's inequality</strong> generalises Cauchy–Schwarz: for $p,q > 1$ with $1/p + 1/q = 1$, $\E|XY| \le (\E|X|^p)^{1/p}(\E|Y|^q)^{1/q}$. Its corollary <strong>Lyapunov's inequality</strong> says the $L^r$ norms are ordered: for $0 < r < s$,
$$\big(\E|X|^r\big)^{1/r} \le \big(\E|X|^s\big)^{1/s},$$
which follows from Jensen with the convex map $u \mapsto u^{s/r}$ applied to $|X|^r$. So a finite $s$-th moment implies finite moments of every lower order — the reason "finite third moment" in <a href="#/t/clt">Berry–Esseen</a> automatically includes finite variance.</p>`,
      },
      {
        title: 'Choosing a bound, and how they mislead',
        html: R`
<table>
<thead><tr><th>Inequality</th><th>Assumes</th><th>Bound on the tail at distance $t$</th><th>Sample size for $(\epsilon,\delta)$</th></tr></thead>
<tbody>
<tr><td>Markov</td><td>$X\ge 0$, $\E X$</td><td>$\E X / t$</td><td>—</td></tr>
<tr><td>Chebyshev</td><td>$\sigma^2$</td><td>$\sigma^2/t^2$</td><td>$\sigma^2/(\epsilon^2\delta)$</td></tr>
<tr><td>Chernoff</td><td>MGF finite</td><td>$e^{-nI(\mu+t)}$</td><td>$\log(1/\delta)/I$</td></tr>
<tr><td>Hoeffding</td><td>$X\in[a,b]$</td><td>$e^{-2nt^2/(b-a)^2}$</td><td>$(b-a)^2\log(2/\delta)/(2\epsilon^2)$</td></tr>
<tr><td>Bernstein</td><td>$|X-\mu|\le b$, $\sigma^2$</td><td>$e^{-nt^2/(2\sigma^2+2bt/3)}$</td><td>$\approx 2\sigma^2\log(2/\delta)/\epsilon^2$ for small $\epsilon$</td></tr>
</tbody>
</table>
<div class="warn" data-name="Markov needs nonnegativity">
<p>Applying $\P(X \ge t) \le \E X/t$ to a variable that can be negative is simply wrong: if $\E X = 0$ the "bound" would say every positive tail is zero. Apply Markov to $|X|$, $X^2$, $e^{\lambda X}$ or $X - \inf X$ instead.</p>
</div>
<div class="warn" data-name="Heavy tails defeat Chernoff">
<p>A Pareto or lognormal variable has $\E e^{\lambda X} = \infty$ for every $\lambda>0$, so the Chernoff bound is the trivial $1$. Polynomial-moment bounds still work (up to the order of the highest finite moment). Bounded-range assumptions are often violated in practice; truncation or median-of-means estimators restore exponential concentration with only a variance assumption.</p>
</div>
<div class="warn" data-name="A bound is not an estimate">
<p>Hoeffding's $2e^{-2n\epsilon^2}$ is not the probability of a large deviation; for moderate $n$ it can exceed the true probability by a factor of 10 or 100. Use inequalities to <em>guarantee</em>, the CLT or simulation to <em>estimate</em>. A useful companion fact: if $X_1,\dots,X_m$ are each $\sigma^2$-sub-Gaussian (i.e. $\E e^{\lambda X_i} \le e^{\lambda^2\sigma^2/2}$), then $\E\max_i X_i \le \sigma\sqrt{2\log m}$ <em>without independence</em> — the price of a maximum over $m$ things grows only like $\sqrt{\log m}$.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that for $X\sim\text{Exp}(1)$ the Chebyshev bound on $\P(X \ge 1 + t)$ eventually becomes worse than Markov's bound on the same event. What does this say about using "more information"?`,
        a: R`Chebyshev gives $\P(|X-1|\ge t) \le 1/t^2$, while Markov gives $\P(X \ge 1+t) \le 1/(1+t)$. Since $1/t^2 < 1/(1+t)$ iff $t^2 > 1+t$, i.e. $t > 1.618$, Chebyshev is better for large $t$ and worse for small $t$. More information gives a better bound only in the regime where the extra moment matters; for small $t$, Chebyshev "wastes" itself on the left tail too. (Cantelli's one-sided $1/(1+t^2)$ is uniformly better than Chebyshev here.)` },
      { q: R`Why does Hoeffding's inequality give a required sample size proportional to $\log(1/\delta)$ while Chebyshev gives one proportional to $1/\delta$? Why does this matter for a union bound over $m$ hypotheses?`,
        a: R`Chebyshev's tail $\sigma^2/(n\epsilon^2)$ decays polynomially in $n$, so solving $\le\delta$ gives $n\propto 1/\delta$. Hoeffding's tail $2e^{-2n\epsilon^2}$ decays exponentially, so $n \propto \log(2/\delta)$. A union bound over $m$ events replaces $\delta$ with $\delta/m$: Chebyshev's $n$ grows linearly in $m$, Hoeffding's only by $\log m/(2\epsilon^2)$.` },
      { q: R`Use Jensen to show that if $\hat\theta$ is unbiased for $\theta > 0$ and not degenerate, then $\log\hat\theta$ is biased downward for $\log\theta$ (assume $\hat\theta>0$).`,
        a: R`$\log$ is strictly concave, so $\E\log\hat\theta < \log\E\hat\theta = \log\theta$ unless $\hat\theta$ is a.s. constant. The bias is roughly $-\Var(\hat\theta)/(2\theta^2)$ by a second-order Taylor expansion — compare the bias formula in <a href="#/t/delta-method">the delta method</a>.` },
      { q: R`For $X\ge0$ the best moment bound $\min_k \E X^k/t^k$ is never worse than the Chernoff bound. Why, then, is Chernoff the standard tool for sums of independent variables?`,
        a: R`Because $\E e^{\lambda S_n} = \prod_i \E e^{\lambda X_i}$ factorises under independence, so the Chernoff bound for a sum is computable from the one-variable MGF and yields the clean exponent $nI(a)$. Moments of a sum do not factorise; $\E S_n^k$ is a messy combinatorial sum of mixed moments. Chernoff wins on tractability, not tightness.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'entropy-kl',
    domain: 'foundations',
    order: 110,
    title: 'Entropy, KL Divergence & Information',
    subtitle: 'Measuring uncertainty and the mismatch between distributions, and why statistics keeps meeting them',
    level: 2,
    sources: { other: 'Cover & Thomas, Elements of Information Theory (2nd ed., 2006), Ch. 2, 8, 11; MacKay, Information Theory, Inference and Learning Algorithms (2003)' },
    prereqs: ['expectation'],
    related: ['inequalities', 'mle', 'fisher-information', 'variational-inference', 'exponential-families', 'classifier-evaluation'],
    tags: ['entropy', 'KL divergence', 'mutual information', 'maximum entropy', 'cross-entropy'],
    summary: R`Entropy $H(P) = -\E\log p(X)$ is the average surprise of a random variable: the number of bits (or nats) needed, on average, to encode it. The Kullback–Leibler divergence $\KL(P\|Q) = \E_P\log(p/q)$ is the extra cost of encoding $P$-data with a code built for $Q$. KL is nonnegative (by Jensen), zero only when $P=Q$, and strikingly asymmetric. It is the quantity that maximum likelihood minimises, that controls the power of likelihood-ratio tests, whose curvature is Fisher information, and whose two directions give the mass-covering and mode-seeking behaviours of approximate inference.`,
    survey: {
      task: 'Quantify uncertainty, dependence, and discrepancy between distributions',
      assumptions: R`$P \ll Q$ for finite $\KL(P\|Q)$; densities w.r.t. a common measure`,
      output: R`$H(X)$, $h(f)$, $\KL(P\|Q)$, $I(X;Y)$ in nats (natural log) or bits ($\log_2$)`,
      strengths: 'Additive over independent coordinates; invariant (KL, MI) under invertible reparameterisation; links coding, testing and estimation',
      weaknesses: 'KL is not a metric (asymmetric, no triangle inequality) and is infinite under support mismatch; differential entropy is not invariant',
      complexity: 'Closed form within exponential families; otherwise numerical integration or Monte Carlo',
    },
    sections: [
      {
        title: 'Entropy as average surprise',
        html: R`
<p>Observing an event of probability $p$ is "surprising" to the degree $-\log p$: certain events carry no surprise, rare events a lot, and surprises of independent events add because probabilities multiply. Entropy is the expected surprise.</p>
<div class="def" data-name="Entropy (discrete)">
<p>For a discrete random variable with pmf $p$ on a set $\mathcal X$,
$$H(X) = -\sum_{x\in\mathcal X} p(x)\log p(x) = \E\big[-\log p(X)\big],$$
with $0\log 0 = 0$. With natural logs the unit is the <em>nat</em>; with $\log_2$, the <em>bit</em>.</p>
</div>
<p>The coding interpretation makes this concrete. A prefix-free binary code assigns codeword lengths $\ell(x)$ satisfying the Kraft inequality $\sum_x 2^{-\ell(x)} \le 1$. Shannon's source coding theorem says the optimal expected length $L^*$ satisfies $H_2(X) \le L^* < H_2(X)+1$: entropy is the irreducible cost of describing $X$, and the ideal codeword length for $x$ is $-\log_2 p(x)$.</p>
<div class="thm" data-kind="Proposition" data-name="Range of entropy">
<p>If $|\mathcal X| = K$, then $0 \le H(X) \le \log K$, with $H = 0$ iff $X$ is degenerate and $H = \log K$ iff $X$ is uniform.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Each term $-p\log p \ge 0$ for $p\in[0,1]$, and is zero only at $p\in\{0,1\}$. For the upper bound, let $u$ be the uniform pmf. Then $0 \le \KL(p\|u) = \sum p(x)\log\big(p(x)K\big) = \log K - H(X)$, using the nonnegativity of KL proved below, with equality iff $p = u$.</p>
</details>
<p>For a pair $(X,Y)$, the <strong>joint entropy</strong> is $H(X,Y) = -\E\log p(X,Y)$ and the <strong>conditional entropy</strong> is $H(Y\mid X) = -\E\log p(Y\mid X)$. Because $p(x,y) = p(x)p(y\mid x)$, logs turn this product into the <strong>chain rule</strong> $H(X,Y) = H(X) + H(Y\mid X)$. Conditioning never increases entropy on average: $H(Y\mid X) \le H(Y)$, with equality iff $X \indep Y$ (proved via mutual information below).</p>
<div class="ex" data-name="Binary entropy">
<p>For $X\sim\text{Bernoulli}(p)$, $H(X) = h(p) = -p\log p - (1-p)\log(1-p)$, maximised at $p = 1/2$ with value $\log 2$ (one bit). A coin with $p=0.11$ carries only half a bit per toss — runs of such tosses can be compressed to about half their length.</p>
</div>`,
      },
      {
        title: 'Differential entropy and maximum entropy',
        html: R`
<p>For a continuous random variable with density $f$, the analogue is the <strong>differential entropy</strong> $h(f) = -\int f(x)\log f(x)\d x$. It behaves differently from discrete entropy in two important ways:</p>
<ul>
<li><strong>It can be negative.</strong> $\text{Unif}(0,a)$ has $h = \log a$, negative for $a < 1$. Differential entropy is not a count of bits; it is the limit of $H(\text{quantised }X) + \log\Delta$ as the bin width $\Delta\to0$, and the $+\log\Delta$ diverges.</li>
<li><strong>It is not invariant.</strong> $h(aX + b) = h(X) + \log|a|$; more generally, for a smooth bijection $Y = g(X)$, $h(Y) = h(X) + \E\log|g'(X)|$ (by the change-of-variables formula in <a href="#/t/transformations">transformations</a>).</li>
</ul>
<p>Standard values: $h(\N(\mu,\sigma^2)) = \tfrac12\log(2\pi e\sigma^2)$; $h(\text{Exp}(\lambda)) = 1 - \log\lambda$; $h(\N_d(\bmu,\bSigma)) = \tfrac12\log\big((2\pi e)^d\det\bSigma\big)$.</p>
<div class="thm" data-name="The normal maximises entropy at fixed variance">
<p>Among all densities on $\R$ with variance $\sigma^2$, the $\N(\mu,\sigma^2)$ density has the largest differential entropy, $\tfrac12\log(2\pi e\sigma^2)$, uniquely.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $f$ have mean $\mu$ and variance $\sigma^2$, and let $\phi$ be the $\N(\mu,\sigma^2)$ density. Since $-\log\phi(x) = \tfrac12\log(2\pi\sigma^2) + (x-\mu)^2/(2\sigma^2)$ is a quadratic, its expectation under $f$ depends only on the first two moments of $f$:
$$0 \le \KL(f\|\phi) = -h(f) - \int f\log\phi = -h(f) + \tfrac12\log(2\pi\sigma^2) + \tfrac12 = -h(f) + \tfrac12\log(2\pi e\sigma^2).$$
Equality holds iff $f = \phi$ a.e.</p>
</details>
<p>The same argument proves a general principle: maximising entropy subject to moment constraints $\E T_j(X) = t_j$ yields a density of the form $f(x) \propto \exp\big(\sum_j\eta_j T_j(x)\big)$ — an <a href="#/t/exponential-families">exponential family</a>. Fixing the mean on $[0,\infty)$ yields the exponential distribution; fixing $\E\log X$ and $\E X$ yields the gamma. Maximum entropy is one principled answer to "what distribution assumes nothing beyond these constraints?"</p>`,
      },
      {
        title: 'KL divergence: definition and properties',
        html: R`
<div class="def" data-name="Kullback–Leibler divergence">
<p>For distributions $P, Q$ with densities (or pmfs) $p, q$,
$$\KL(P\|Q) = \E_P\Big[\log\frac{p(X)}{q(X)}\Big] = \int p(x)\log\frac{p(x)}{q(x)}\d x,$$
defined to be $+\infty$ unless $P$ is absolutely continuous with respect to $Q$ ($q = 0 \Rightarrow p = 0$).</p>
</div>
<p>Writing $\KL(P\|Q) = \big(-\E_P\log q\big) - \big(-\E_P\log p\big)$ exhibits it as <strong>cross-entropy minus entropy</strong>: the expected excess code length when $P$-distributed data are encoded with a code optimised for $Q$.</p>
<div class="thm" data-name="Gibbs' inequality">
<p>$\KL(P\|Q) \ge 0$, with equality iff $P = Q$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Assume $P\ll Q$ (otherwise $\KL = \infty$). Since $-\log$ is strictly convex, Jensen gives
$$\KL(P\|Q) = \E_P\Big[-\log\frac{q(X)}{p(X)}\Big] \ge -\log\E_P\frac{q(X)}{p(X)} = -\log\int_{p>0} q \ge -\log 1 = 0.$$
Equality in Jensen requires $q/p$ to be $P$-a.s. constant, and equality in the last step requires $Q(p>0)=1$; together they force $p = q$.</p>
</details>
<p>Useful properties and formulas:</p>
<ul>
<li><strong>Not symmetric:</strong> in general $\KL(P\|Q) \ne \KL(Q\|P)$, and it fails the triangle inequality. It is a divergence, not a distance.</li>
<li><strong>Additivity:</strong> $\KL(P_1\otimes P_2\,\|\,Q_1\otimes Q_2) = \KL(P_1\|Q_1)+\KL(P_2\|Q_2)$, so $n$ IID observations carry $n\,\KL(P\|Q)$.</li>
<li><strong>Invariance:</strong> KL is unchanged by any bijective transformation of the sample space (the Jacobians cancel in the ratio), unlike differential entropy.</li>
<li><strong>Data processing:</strong> passing both distributions through the same channel (any Markov kernel, any statistic $T$) cannot increase KL: $\KL(P^T\|Q^T) \le \KL(P\|Q)$, with equality when $T$ is sufficient for $\{P, Q\}$.</li>
<li><strong>Pinsker's inequality:</strong> total variation is controlled by KL, $\sup_A|P(A)-Q(A)| \le \sqrt{\KL(P\|Q)/2}$.</li>
<li><strong>Normals:</strong> $\KL\big(\N(\mu_1,\sigma_1^2)\,\|\,\N(\mu_2,\sigma_2^2)\big) = \log\dfrac{\sigma_2}{\sigma_1} + \dfrac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \dfrac12.$</li>
</ul>
<div class="ex" data-name="Asymmetry in one line">
<p>Take $P = \N(0,1)$ and $Q = \N(0,\sigma^2)$. Then $\KL(P\|Q) = \log\sigma + \frac{1}{2\sigma^2} - \frac12$ and $\KL(Q\|P) = -\log\sigma + \frac{\sigma^2}{2} - \frac12$. As $\sigma\to0$ the first blows up like $1/(2\sigma^2)$ (a narrow $Q$ cannot explain $P$'s spread) while the second only grows like $\log(1/\sigma)$. Asymmetric penalties for "too narrow" and "too wide" are the whole story of the next two sections.</p>
</div>`,
        lab: 'entropy-kl-normals',
      },
      {
        title: 'Why statisticians keep meeting KL',
        html: R`
<p><strong>Maximum likelihood is minimum KL.</strong> Let $X_1,\dots,X_n \iid P$ and fit a model $\{f_\theta\}$. The average log-likelihood satisfies, by the <a href="#/t/lln">law of large numbers</a>,
$$\frac1n\sum_{i=1}^n\log f_\theta(X_i) \;\asto\; \E_P\log f_\theta(X) = -H(P) - \KL(P\|f_\theta).$$
Since $H(P)$ does not depend on $\theta$, maximising the likelihood asymptotically minimises $\KL(P\|f_\theta)$. If the model is correct the minimiser is the true $\theta_0$ (Gibbs), which is the heart of the consistency proof of the <a href="#/t/mle">MLE</a>. If the model is misspecified, the MLE converges to the <em>pseudo-true</em> parameter, the KL projection of $P$ onto the model (White, 1982).</p>
<p><strong>Fisher information is KL curvature.</strong> For a smooth regular family,
$$\KL(f_\theta\,\|\,f_{\theta+h}) = \tfrac12\,I(\theta)\,h^2 + o(h^2).$$
Sketch: expand $\log f_{\theta+h}$ to second order in $h$; the first-order term has expectation zero (the score has mean zero), and the second-order term has expectation $-\tfrac12 I(\theta)h^2$. So <a href="#/t/fisher-information">Fisher information</a> measures how quickly nearby parameter values become distinguishable.</p>
<p><strong>KL is the exponent of testing.</strong> In a test of $P_0$ against $P_1$ from $n$ IID observations, the log-likelihood ratio $\sum\log(p_1/p_0)(X_i)$ has mean $-n\KL(P_0\|P_1)$ under $P_0$ and $+n\KL(P_1\|P_0)$ under $P_1$. The Chernoff–Stein lemma makes this precise: holding the type I error below any fixed $\alpha\in(0,1)$, the smallest achievable type II error $\beta_n$ satisfies $\frac1n\log\beta_n\to-\KL(P_0\|P_1)$. The asymmetry of KL is the asymmetry between the two error types in <a href="#/t/neyman-pearson">Neyman–Pearson</a> testing.</p>
<p><strong>Cross-entropy loss.</strong> Training a classifier by minimising the average of $-\log q_\theta(y\mid x)$ minimises $H(p) + \KL(p\|q_\theta)$, i.e. KL from the data distribution to the model; this is why log loss is a <a href="#/t/classifier-evaluation">strictly proper scoring rule</a>.</p>
<div class="def" data-name="Mutual information">
<p>$I(X;Y) = \KL\big(P_{XY}\,\|\,P_X\otimes P_Y\big) = H(X) - H(X\mid Y) = H(X)+H(Y)-H(X,Y).$</p>
</div>
<p>Mutual information is nonnegative and zero iff $X\indep Y$ — unlike correlation, it detects every kind of dependence. It is invariant under bijections of $X$ and of $Y$ separately. For a bivariate normal with correlation $\rho$, $I = -\tfrac12\log(1-\rho^2)$, which diverges as $|\rho|\to1$: a deterministic linear relationship between continuous variables carries infinite information.</p>`,
      },
      {
        title: 'Forward vs. reverse KL: mass-covering and mode-seeking',
        html: R`
<p>When we approximate a complicated $p$ by a simple family $q\in\mathcal Q$, the direction of KL decides what "good approximation" means.</p>
<div class="cols">
<div><h4>Forward: $\min_q \KL(p\|q)$</h4>
<p>The integrand $p\log(p/q)$ explodes wherever $p > 0$ but $q\approx 0$. So $q$ is forced to <strong>cover all the mass</strong> of $p$, even at the cost of putting mass where $p$ has none. For an exponential-family $q$, the optimum <strong>matches the expected sufficient statistics</strong>: a Gaussian $q$ takes $p$'s mean and variance. This is what maximum likelihood does, and what expectation propagation does locally.</p></div>
<div><h4>Reverse: $\min_q \KL(q\|p)$</h4>
<p>Now the integrand $q\log(q/p)$ explodes wherever $q>0$ but $p\approx0$. So $q$ is forced to <strong>avoid regions where $p$ is small</strong> ("zero-forcing"), and is happy to ignore whole modes of $p$. The optimum <strong>locks onto one mode</strong> and typically underestimates spread. This is the objective of <a href="#/t/variational-inference">variational inference</a>, chosen because it needs only expectations under $q$ and an unnormalised $p$.</p></div>
</div>
<div class="thm" data-kind="Proposition" data-name="Forward KL projection onto the Gaussians">
<p>If $p$ has mean $m$ and variance $s^2 < \infty$, then $\argmin_{\mu,\sigma}\KL\big(p\,\|\,\N(\mu,\sigma^2)\big) = (m, s)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\KL(p\|q) = -h(p) + \tfrac12\log(2\pi\sigma^2) + \E_p(X-\mu)^2/(2\sigma^2)$ and $\E_p(X-\mu)^2 = s^2 + (m-\mu)^2$. Minimising in $\mu$ gives $\mu=m$; then $\frac{d}{d\sigma}\big[\log\sigma + s^2/(2\sigma^2)\big] = 1/\sigma - s^2/\sigma^3 = 0$ gives $\sigma = s$.</p>
</details>
<p>Reverse KL has no such closed form for a mixture target, and generally has several local minima — one near each mode, and for well-separated modes often a broad "straddling" compromise as well — which is a practical headache in variational inference: where the optimiser ends up depends on where it starts. The lab lets you watch the two projections diverge as a bimodal target separates.</p>`,
        lab: 'entropy-kl-fit',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Support mismatch makes KL infinite">
<p>If $Q$ assigns zero probability to an event that $P$ can produce, $\KL(P\|Q)=\infty$. Empirical distributions have finite support, so $\KL(\hat P_n\|Q)$ is fine but $\KL(Q\|\hat P_n)$ is infinite for any continuous $Q$. This is why plug-in KL estimators need smoothing, and why GAN-style objectives often use other divergences.</p>
</div>
<div class="warn" data-name="Differential entropy is not information">
<p>Comparing differential entropies across different units or parameterisations is meaningless ($h$ shifts by $\log|a|$ under rescaling). Mutual information and KL, which are differences of such quantities, are invariant and safe to compare.</p>
</div>
<div class="warn" data-name="Bits vs. nats">
<p>Formulas with $\log$ are in nats; divide by $\log 2 \approx 0.693$ to get bits. Pinsker's inequality as stated above uses nats.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that $I(X;Y) \ge 0$ and deduce that conditioning cannot increase entropy on average.`,
        a: R`$I(X;Y) = \KL(P_{XY}\|P_X\otimes P_Y) \ge 0$ by Gibbs' inequality. Since $I(X;Y) = H(X) - H(X\mid Y)$, we get $H(X\mid Y)\le H(X)$, with equality iff $P_{XY} = P_X\otimes P_Y$, i.e. independence. (A particular observed $Y=y$ can increase uncertainty; only the average decreases.)` },
      { q: R`Compute $\KL(\text{Ber}(a)\|\text{Ber}(p))$ and explain its role in bounding $\P(\bar X_n \ge a)$ for Bernoulli($p$) data with $a > p$.`,
        a: R`$\KL = a\log\frac{a}{p} + (1-a)\log\frac{1-a}{1-p}$. It is the Legendre transform of the Bernoulli cumulant generating function, so the Chernoff bound reads $\P(\bar X_n\ge a)\le e^{-n\KL(\text{Ber}(a)\|\text{Ber}(p))}$, and by Cramér's theorem this exponent is exact.` },
      { q: R`A target $p$ is an equal mixture of $\N(-3,1)$ and $\N(3,1)$. Describe the best Gaussian approximation under forward KL and under reverse KL.`,
        a: R`Forward KL matches moments: mean $0$ and variance $1 + 9 = 10$, a broad Gaussian straddling both modes and putting much of its mass near $0$ where $p$ is tiny. Reverse KL picks one mode: approximately $\N(\pm3, 1)$ (two symmetric global optima), ignoring the other mode entirely, and its reverse KL is close to $\log 2$ — the price of missing half the mass.` },
      { q: R`Why does maximum likelihood under a misspecified model still estimate something meaningful?`,
        a: R`The average log-likelihood converges to $-H(P) - \KL(P\|f_\theta)$, so the MLE converges (under regularity) to $\theta^* = \argmin_\theta\KL(P\|f_\theta)$, the member of the model closest to the truth in forward KL. Inference about $\theta^*$ then requires the sandwich variance rather than the inverse Fisher information.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'convergence',
    domain: 'foundations',
    order: 120,
    title: 'Modes of Convergence',
    subtitle: 'Almost sure, in probability, in mean and in distribution — what each means and how they relate',
    level: 2,
    sources: { aos: '5.1–5.2', cb: '§5.5.1–5.5.3', other: 'Billingsley, Probability and Measure (1995), §20–25; van der Vaart, Asymptotic Statistics (1998), Ch. 2' },
    prereqs: ['random-variables', 'expectation'],
    related: ['lln', 'clt', 'delta-method', 'mgf', 'inequalities'],
    tags: ['asymptotics', 'Slutsky', 'Borel–Cantelli', 'uniform integrability'],
    summary: R`A random variable is a function on a sample space, so "$X_n\to X$" can mean several things: pointwise for almost every outcome (almost surely), with vanishing probability of a deviation (in probability), with vanishing mean error (in $L^r$), or only in the sense that the distributions approach each other (in distribution). These modes form a strict hierarchy, and the counterexamples that separate them are as instructive as the theorems. Slutsky's theorem and the continuous mapping theorem are the workhorses that let you combine limits; uniform integrability is the condition that lets you pass expectations to the limit.`,
    survey: {
      task: 'Make precise statements about the limiting behaviour of estimators and statistics',
      assumptions: R`All $X_n$ on a common probability space (except for convergence in distribution)`,
      output: R`Statements $X_n\asto X$, $X_n\pto X$, $X_n\xrightarrow{L^r}X$, $X_n\dto X$; $O_p$/$o_p$ rates`,
      strengths: 'Clean implication hierarchy; Slutsky and continuous mapping make asymptotics compositional',
      weaknesses: 'Limits say nothing about any finite $n$; convergence in distribution does not imply convergence of moments',
      complexity: '—',
    },
    sections: [
      {
        title: 'Four ways to be close',
        html: R`
<p>For numbers, $x_n \to x$ has one meaning. For random variables there are several, because a random variable $X_n:\Omega\to\R$ is a whole function, and "closeness of functions" can be measured in different ways. Throughout, $X_1, X_2, \dots$ and $X$ are random variables; for the first three modes they must live on a common probability space $(\Omega,\mathcal F,\P)$.</p>
<div class="def" data-name="Modes of convergence">
<ul>
<li><strong>Almost surely:</strong> $X_n \asto X$ if $\P\big(\{\omega: X_n(\omega)\to X(\omega)\}\big) = 1$.</li>
<li><strong>In probability:</strong> $X_n \pto X$ if for every $\epsilon>0$, $\P(|X_n - X| > \epsilon) \to 0$.</li>
<li><strong>In $L^r$</strong> ($r\ge1$; $r=2$ is "in quadratic mean"): $X_n \xrightarrow{L^r} X$ if $\E|X_n - X|^r \to 0$.</li>
<li><strong>In distribution</strong> (weakly, in law): $X_n \dto X$ if $F_{X_n}(x) \to F_X(x)$ at every $x$ where $F_X$ is continuous.</li>
</ul>
</div>
<p>The first is a statement about individual sample paths: for almost every $\omega$, the sequence of numbers $X_n(\omega)$ converges. The second is a statement about a sequence of probabilities, one for each $n$; it says nothing about how a single path behaves across $n$. The third concerns a sequence of expectations. The fourth only involves the marginal distributions of each $X_n$ separately — the $X_n$ need not even be defined on the same space.</p>
<div class="note" data-name="Why only continuity points?">
<p>Let $X_n = 1/n$ (a constant). Surely $X_n \to 0$ in every reasonable sense. But $F_{X_n}(0) = 0$ for all $n$ while $F_0(0) = 1$. Requiring convergence at $x=0$, the jump point of the limit, would reject this obvious convergence. Distributions can approach a jump from either side, so we demand nothing exactly there.</p>
</div>
<p>A useful reformulation of almost sure convergence, which the lab visualises directly:
$$X_n\asto X \iff \text{for every }\epsilon>0,\quad \P\Big(\sup_{m\ge n}|X_m - X| > \epsilon\Big) \to 0.$$
Compare with convergence in probability, which only needs $\P(|X_n - X|>\epsilon)\to0$ — no supremum over the future. Almost sure convergence controls the whole tail of the path; convergence in probability controls one time at a time.</p>`,
      },
      {
        title: 'The hierarchy and its proofs',
        html: R`
$$\begin{aligned}
X_n\asto X \;&\Longrightarrow\; X_n\pto X \;\Longrightarrow\; X_n\dto X,\\
X_n\xrightarrow{L^r} X \;&\Longrightarrow\; X_n\pto X, \qquad X_n \dto c \text{ (constant)} \;\Longrightarrow\; X_n\pto c.
\end{aligned}$$
<div class="thm" data-name="Implications">
<p>(i) a.s. $\Rightarrow$ in probability; (ii) $L^r$ $\Rightarrow$ in probability; (iii) in probability $\Rightarrow$ in distribution; (iv) in distribution to a constant $\Rightarrow$ in probability; (v) $L^s \Rightarrow L^r$ for $s > r \ge 1$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(i) $\P(|X_n-X|>\epsilon) \le \P(\sup_{m\ge n}|X_m-X|>\epsilon) \to 0$ by the reformulation above (which itself follows from continuity of $\P$ along the decreasing events $\{\sup_{m\ge n}|X_m-X|>\epsilon\}$, whose intersection is contained in the null set where $X_n\not\to X$).</p>
<p>(ii) Markov: $\P(|X_n-X|>\epsilon) \le \E|X_n-X|^r/\epsilon^r \to 0$.</p>
<p>(iii) Fix a continuity point $x$ of $F$ and $\epsilon>0$. If $X_n \le x$ then either $X \le x+\epsilon$ or $|X_n-X|>\epsilon$, so $F_n(x) \le F(x+\epsilon) + \P(|X_n-X|>\epsilon)$. Symmetrically, $F(x-\epsilon) \le F_n(x) + \P(|X_n-X|>\epsilon)$. Letting $n\to\infty$: $F(x-\epsilon) \le \liminf F_n(x) \le \limsup F_n(x) \le F(x+\epsilon)$. Let $\epsilon\downarrow0$ and use continuity of $F$ at $x$.</p>
<p>(iv) $c\pm\epsilon$ are continuity points of the limit CDF $\ind(x\ge c)$, so $\P(|X_n-c|>\epsilon) \le 1 - F_n(c+\epsilon) + F_n(c-\epsilon) \to 1 - 1 + 0 = 0$.</p>
<p>(v) Lyapunov's inequality (see <a href="#/t/inequalities">inequalities</a>): $(\E|Y|^r)^{1/r}\le(\E|Y|^s)^{1/s}$.</p>
</details>
<p>The tool for proving almost sure convergence is the Borel–Cantelli lemma.</p>
<div class="thm" data-kind="Lemma" data-name="Borel–Cantelli">
<p>Let $A_1, A_2,\dots$ be events and $\{A_n\text{ i.o.}\} = \bigcap_n\bigcup_{m\ge n}A_m$ the event that infinitely many occur. (I) If $\sum_n\P(A_n)<\infty$ then $\P(A_n\text{ i.o.}) = 0$. (II) If the $A_n$ are independent and $\sum_n\P(A_n)=\infty$ then $\P(A_n\text{ i.o.})=1$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(I) $\P(A_n\text{ i.o.}) \le \P(\bigcup_{m\ge n}A_m) \le \sum_{m\ge n}\P(A_m)\to0$, the tail of a convergent series. (II) By independence and $1-x\le e^{-x}$, $\P(\bigcap_{m=n}^N A_m^c) = \prod_{m=n}^N(1-\P(A_m)) \le \exp(-\sum_{m=n}^N\P(A_m)) \to 0$ as $N\to\infty$. So $\P(\bigcup_{m\ge n}A_m) = 1$ for every $n$, and the intersection over $n$ also has probability one.</p>
</details>
<p>Applying (I) with $A_n = \{|X_n - X|>\epsilon\}$ gives the most-used sufficient condition: <strong>if $\sum_n\P(|X_n-X|>\epsilon)<\infty$ for every $\epsilon>0$, then $X_n\asto X$</strong> ("complete convergence"). It also gives a partial converse to (i): if $X_n\pto X$, choose $n_k$ with $\P(|X_{n_k}-X|>2^{-k})<2^{-k}$; the probabilities are summable, so <em>some subsequence</em> converges almost surely.</p>`,
      },
      {
        title: 'Counterexamples that separate the modes',
        html: R`
<p>Every converse in the hierarchy fails, and the failures are not pathological curiosities — they are exactly the phenomena that make asymptotic arguments delicate.</p>
<div class="ex" data-name="The typewriter: in probability (and $L^r$), not almost surely">
<p>Let $U\sim\text{Unif}(0,1)$. Write $n = 2^k + j$ with $0\le j < 2^k$ and set $X_n = \ind\big(U\in[j2^{-k},(j+1)2^{-k})\big)$. As $n$ runs through block $k$, the indicator's window sweeps across $[0,1)$ in $2^k$ steps of width $2^{-k}$. Then $\P(X_n = 1) = \E|X_n|^r = 2^{-k}\to0$, so $X_n\to0$ in probability and in every $L^r$. But every $U$ is hit exactly once in every block, so $X_n(\omega) = 1$ infinitely often for <em>every</em> $\omega$: the sequence converges almost surely nowhere.</p>
</div>
<div class="ex" data-name="Independent rare events: in probability, not almost surely">
<p>Let $X_n\sim\text{Bernoulli}(1/n)$ independent. Then $\P(X_n=1) = 1/n\to0$, but $\sum 1/n = \infty$, so by Borel–Cantelli II, $X_n = 1$ infinitely often with probability one. Replace $1/n$ by $1/n^2$ and the sum converges: now $X_n\asto0$. The boundary between the two is exactly summability.</p>
</div>
<div class="ex" data-name="Almost surely, but not in $L^1$">
<p>Let $X_n = n\,\ind(U < 1/n)$. For each $\omega$ with $U(\omega)>0$, $X_n(\omega) = 0$ once $n > 1/U(\omega)$, so $X_n\asto0$. Yet $\E X_n = 1$ for all $n$. A vanishing-probability event carries a non-vanishing amount of mass — the failure of <em>uniform integrability</em> discussed below.</p>
</div>
<div class="ex" data-name="In distribution, not in probability">
<p>Let $X\sim\N(0,1)$ and $X_n = -X$ for all $n$. Each $X_n$ has exactly the law of $X$, so $X_n\dto X$ trivially, but $|X_n - X| = 2|X|$ does not shrink. Convergence in distribution is a statement about laws, not about random variables.</p>
</div>
<p>The lab draws many sample paths of each sequence. Each row is one $\omega$; a mark appears at time $n$ when $|X_n|>\epsilon$. Convergence in probability means the <em>columns</em> thin out; almost sure convergence means each <em>row</em> eventually goes silent.</p>`,
        lab: 'convergence-paths',
      },
      {
        title: 'Working with convergence in distribution',
        html: R`
<p>Convergence in distribution is the mode of the <a href="#/t/clt">central limit theorem</a>, so we need tools to manipulate it. Three equivalent characterisations are useful:</p>
<div class="thm" data-name="Portmanteau and Lévy continuity (selected)">
<p>The following are equivalent: (a) $X_n\dto X$; (b) $\E g(X_n)\to\E g(X)$ for every bounded continuous $g$; (c) $\limsup_n\P(X_n\in C)\le\P(X\in C)$ for every closed $C$; (d) the characteristic functions converge pointwise, $\E e^{itX_n}\to\E e^{itX}$ for all $t$ (Lévy's continuity theorem).</p>
</div>
<p>Characterisation (d) is how the CLT is proved; (b) is the natural definition in higher dimensions and on function spaces. For random vectors, the <strong>Cramér–Wold device</strong> reduces everything to one dimension: $\mathbf X_n\dto\mathbf X$ in $\R^d$ iff $\mathbf a^\top\mathbf X_n\dto\mathbf a^\top\mathbf X$ for every fixed $\mathbf a\in\R^d$.</p>
<div class="thm" data-name="Continuous mapping theorem">
<p>If $g$ is continuous at every point of a set $C$ with $\P(X\in C)=1$, then $X_n\to X$ almost surely, in probability, or in distribution implies $g(X_n)\to g(X)$ in the same mode.</p>
</div>
<div class="thm" data-name="Slutsky's theorem">
<p>If $X_n\dto X$ and $Y_n\pto c$ for a constant $c$, then $X_n+Y_n\dto X+c$, $\;Y_nX_n\dto cX$, and $X_n/Y_n\dto X/c$ if $c\ne0$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The key step is joint convergence: $(X_n, Y_n)\dto(X, c)$. Because the second limit is a constant, for bounded Lipschitz $g$, $|\E g(X_n,Y_n) - \E g(X_n,c)| \le L\epsilon + 2\|g\|_\infty\P(|Y_n-c|>\epsilon)$, which is small, and $\E g(X_n,c)\to\E g(X,c)$ by marginal convergence; bounded Lipschitz functions suffice to characterise weak convergence. Then apply the continuous mapping theorem to $(x,y)\mapsto x+y,\;xy,\;x/y$.</p>
</details>
<div class="warn" data-name="Slutsky needs a constant limit">
<p>If $Y_n \dto Y$ with $Y$ random, marginal convergence does not determine joint convergence, and $X_n + Y_n$ need not converge to $X+Y$. With $X_n = Z$, $Y_n = -Z$ and $Z\sim\N(0,1)$, both marginals converge to $\N(0,1)$ but $X_n+Y_n=0$, not $\N(0,2)$.</p>
</div>
<p>Slutsky is what lets us replace unknown quantities by consistent estimates: since $\sqrt n(\bar X_n-\mu)/\sigma\dto\N(0,1)$ and $\sigma/S_n\pto1$, the studentised statistic $\sqrt n(\bar X_n-\mu)/S_n\dto\N(0,1)$ for any distribution with finite variance.</p>
<div class="def" data-name="Stochastic order notation">
<p>$X_n = o_p(a_n)$ means $X_n/a_n\pto0$. $X_n = O_p(a_n)$ means $X_n/a_n$ is <em>bounded in probability</em> (tight): for every $\epsilon$ there is $M$ with $\sup_n\P(|X_n/a_n|>M)<\epsilon$. Any sequence converging in distribution is $O_p(1)$; rules such as $O_p(1)\,o_p(1) = o_p(1)$ and $o_p(1)+O_p(1) = O_p(1)$ follow from Slutsky-type arguments.</p>
</div>
<p>In this notation the CLT says $\bar X_n - \mu = O_p(n^{-1/2})$, and the <a href="#/t/delta-method">delta method</a> is the statement $g(\bar X_n) - g(\mu) - g'(\mu)(\bar X_n-\mu) = o_p(n^{-1/2})$.</p>`,
      },
      {
        title: 'Uniform integrability: when expectations follow',
        html: R`
<p>None of the modes except $L^r$ convergence say anything about moments. This matters: "the asymptotic variance of $\hat\theta_n$" usually means the variance of the limit distribution, which need not be the limit of $\Var\hat\theta_n$ — the latter may even be infinite for every $n$ (e.g. $1/\bar X_n$ for normal data).</p>
<div class="def" data-name="Uniform integrability">
<p>A family $\{X_n\}$ is uniformly integrable (UI) if $\displaystyle\lim_{M\to\infty}\sup_n\E\big[|X_n|\,\ind(|X_n|>M)\big] = 0.$</p>
</div>
<div class="thm" data-name="Vitali's convergence theorem">
<p>If $X_n\pto X$, then $\E|X_n-X|\to0$ iff $\{X_n\}$ is uniformly integrable. If $X_n\dto X$ and $\{X_n\}$ is UI, then $\E X_n\to\E X$.</p>
</div>
<p>UI forbids exactly what happens in the $n\ind(U<1/n)$ example, where a mass of $1$ escapes to infinity on an event of probability $1/n$. A convenient sufficient condition is a uniformly bounded higher moment: if $\sup_n\E|X_n|^{1+\delta}<\infty$ for some $\delta>0$, then $\{X_n\}$ is UI (because $\E[|X|;|X|>M]\le\E|X|^{1+\delta}/M^\delta$). Dominated convergence is the special case $|X_n|\le Y$ with $\E Y<\infty$.</p>`,
      },
      {
        title: 'Summary and pitfalls',
        html: R`
<table>
<thead><tr><th>Mode</th><th>Controls</th><th>Typical tool</th><th>Typical theorem</th></tr></thead>
<tbody>
<tr><td>a.s.</td><td>each path, all future times</td><td>Borel–Cantelli</td><td>Strong <a href="#/t/lln">LLN</a>, Glivenko–Cantelli</td></tr>
<tr><td>in probability</td><td>one time at a time</td><td>Chebyshev / Markov</td><td>Weak LLN, consistency</td></tr>
<tr><td>$L^r$</td><td>mean error</td><td>variance + bias</td><td>MSE consistency</td></tr>
<tr><td>in distribution</td><td>marginal laws only</td><td>characteristic functions</td><td><a href="#/t/clt">CLT</a>, <a href="#/t/delta-method">delta method</a></td></tr>
</tbody>
</table>
<div class="warn" data-name="Consistency by MSE vs. by probability">
<p>$\MSE(\hat\theta_n)\to0$ implies consistency, but not conversely: an estimator can be consistent while its MSE is infinite (heavy-tailed estimators) or does not vanish (rare huge errors). Statements like "the estimator is consistent because its variance goes to zero" should check for bias too.</p>
</div>
<div class="warn" data-name="Asymptotic statements hide constants">
<p>$X_n\pto X$ is compatible with $\P(|X_n-X|>\epsilon) = 0.9$ at $n = 10^6$. Rates (Chebyshev, Hoeffding, Berry–Esseen) are what turn limits into usable approximations.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Give a sequence that converges in probability but not almost surely, and explain using Borel–Cantelli why it fails.`,
        a: R`Independent $X_n\sim\text{Bernoulli}(1/n)$: $\P(|X_n|>\epsilon) = 1/n\to0$, but the events $\{X_n=1\}$ are independent with $\sum1/n=\infty$, so by Borel–Cantelli II they occur infinitely often almost surely; each path has infinitely many ones and does not converge to $0$. (The typewriter sequence also works and does not need independence.)` },
      { q: R`Why does convergence in distribution to a constant imply convergence in probability, but not convergence in distribution to a non-degenerate limit?`,
        a: R`For a constant $c$, $\P(|X_n - c|>\epsilon)$ is determined by the marginal CDF $F_n$ at $c\pm\epsilon$, which converges to $0$ and $1$. For a random limit $X$, $\P(|X_n - X|>\epsilon)$ depends on the <em>joint</em> law of $(X_n,X)$, which convergence in distribution does not constrain — e.g. $X_n = -X$ with $X$ symmetric.` },
      { q: R`Use Slutsky's theorem to show that $\sqrt n(\bar X_n - \mu)/S_n\dto\N(0,1)$ for any IID data with $0<\sigma^2<\infty$.`,
        a: R`By the CLT, $\sqrt n(\bar X_n-\mu)/\sigma\dto\N(0,1)$. By the WLLN applied to $X_i$ and $X_i^2$ and the continuous mapping theorem, $S_n^2\pto\sigma^2$, hence $\sigma/S_n\pto1$. Slutsky (product with a sequence converging in probability to a constant) gives the result.` },
      { q: R`$X_n = n\,\ind(U<1/n)$ converges to $0$ almost surely. Does $\E X_n\to0$? Which condition fails?`,
        a: R`No: $\E X_n = 1$ for every $n$. Uniform integrability fails: $\E[X_n;X_n>M] = 1$ whenever $n>M$, so $\sup_n\E[X_n;X_n>M] = 1$ for all $M$. By Vitali, a.s. convergence plus UI would have been needed for $\E X_n\to\E X$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'lln',
    domain: 'foundations',
    order: 130,
    title: 'The Laws of Large Numbers',
    subtitle: 'Why averages settle down, how fast, and what it takes to stop them',
    level: 2,
    sources: { aos: '5.3', cb: '§5.5.1–5.5.2', other: 'Durrett, Probability: Theory and Examples (5th ed.), §2.2–2.4; Etemadi (1981)' },
    prereqs: ['convergence', 'inequalities'],
    related: ['clt', 'monte-carlo-integration', 'empirical-cdf', 'mle', 'stationary-distributions'],
    tags: ['consistency', 'Borel–Cantelli', 'Cauchy', 'ergodic'],
    summary: R`The law of large numbers says the sample mean of IID observations converges to the population mean: in probability (the weak law) and, more strongly, along almost every sample path (the strong law). The weak law falls out of Chebyshev's inequality in one line when the variance is finite; the strong law holds under the sharp condition $\E|X|<\infty$, and that condition cannot be weakened — for the Cauchy distribution the sample mean never settles at all. The LLN is the reason estimators are consistent, Monte Carlo works, and the empirical distribution recovers the truth.`,
    survey: {
      task: R`Justify $\bar X_n\to\mu$: consistency of moment-based estimators and Monte Carlo averages`,
      assumptions: R`IID (or weaker: uncorrelated with bounded variance; stationary ergodic); $\E|X|<\infty$`,
      output: R`$\bar X_n\pto\mu$ (WLLN), $\bar X_n\asto\mu$ (SLLN); with $\sigma^2$, the rate $O_p(n^{-1/2})$`,
      strengths: 'Minimal assumptions; extends to functions of the data, empirical CDFs, ergodic chains',
      weaknesses: 'No rate without more moments; fails for $\E|X|=\infty$ and for non-ergodic dependence',
      complexity: '—',
    },
    sections: [
      {
        title: 'What the law says',
        html: R`
<p>Let $X_1,X_2,\dots$ be IID with mean $\mu$ and let $\bar X_n = \frac1n\sum_{i=1}^n X_i$. The law of large numbers is the theorem that turns the frequency intuition of probability — "in many repetitions, the fraction of heads approaches $1/2$" — into mathematics. It comes in two strengths, matching two <a href="#/t/convergence">modes of convergence</a>:</p>
<div class="cols">
<div><h4>Weak law (WLLN)</h4><p>$\bar X_n\pto\mu$: for every $\epsilon>0$, $\P(|\bar X_n-\mu|>\epsilon)\to0$. At any single large $n$, the average is probably close to $\mu$.</p></div>
<div><h4>Strong law (SLLN)</h4><p>$\bar X_n\asto\mu$: with probability one the whole sequence of averages converges. Once a path gets close, it eventually <em>stays</em> close.</p></div>
</div>
<p>The distinction matters when you watch a single experiment unfold — a Monte Carlo run, a sequential trial, an online learner. The weak law alone permits each path to make rare but infinitely many excursions; the strong law rules that out.</p>
<p>A useful way to read both laws is through the <strong>variance of the mean</strong>: if $\Var X = \sigma^2 < \infty$ then $\Var\bar X_n = \sigma^2/n$, so the average concentrates at rate $1/\sqrt n$. The <a href="#/t/clt">central limit theorem</a> describes the shape of those $1/\sqrt n$ fluctuations; the law of large numbers only says they vanish.</p>`,
      },
      {
        title: 'The weak law',
        html: R`
<div class="thm" data-name="WLLN with finite variance">
<p>If $X_1,\dots,X_n$ are pairwise uncorrelated with common mean $\mu$ and $\Var X_i\le\sigma^2<\infty$, then $\P(|\bar X_n-\mu|\ge\epsilon)\le\dfrac{\sigma^2}{n\epsilon^2}\to0$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Uncorrelatedness gives $\Var\bar X_n = \frac1{n^2}\sum_i\Var X_i\le\sigma^2/n$, and Chebyshev's inequality does the rest. Note the proof also shows $\bar X_n\to\mu$ in $L^2$.</p>
</details>
<p>Independence is not needed — only that the covariances vanish — and the identical-distribution assumption is not needed either. The finite variance, however, is doing real work: the result gives a rate. Remarkably, a weak law survives without any second moment.</p>
<div class="thm" data-name="Khinchin's WLLN">
<p>If $X_1,X_2,\dots$ are IID with $\E|X_1|<\infty$ and $\E X_1=\mu$, then $\bar X_n\pto\mu$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\varphi(t) = \E e^{itX_1}$. Because $\E|X_1|<\infty$, $\varphi$ is differentiable at $0$ with $\varphi'(0) = i\mu$, so $\varphi(t) = 1 + i\mu t + o(t)$ as $t\to0$. (Use $|e^{ix}-1-ix|\le\min(2|x|, x^2/2)$ and dominated convergence.) By independence,
$$\E e^{it\bar X_n} = \varphi(t/n)^n = \Big(1 + \frac{i\mu t}{n} + o\big(\tfrac1n\big)\Big)^n \to e^{i\mu t},$$
the characteristic function of the constant $\mu$. By Lévy's continuity theorem $\bar X_n\dto\mu$, and convergence in distribution to a constant implies convergence in probability.</p>
</details>
<p>(The limit of $(1+z_n/n)^n$ for complex $z_n\to z$ is $e^z$; this is checked by taking the principal logarithm, valid once $|z_n/n|<1$.)</p>`,
      },
      {
        title: 'The strong law',
        html: R`
<div class="thm" data-name="Kolmogorov's SLLN (Etemadi's form)">
<p>If $X_1,X_2,\dots$ are pairwise independent and identically distributed with $\E|X_1|<\infty$, then $\bar X_n\asto\E X_1$. Conversely, if the $X_i$ are IID and $\E|X_1|=\infty$, then $\limsup_n|\bar X_n| = \infty$ almost surely.</p>
</div>
<p>The full proof (Etemadi, 1981) truncates $X_i$ at level $i$, controls the truncated variances, proves convergence along a geometric subsequence $k_n = \lfloor\alpha^n\rfloor$ with Chebyshev + Borel–Cantelli, and fills in the gaps by monotonicity (treating positive and negative parts separately). The essential mechanism is visible in a cleaner special case.</p>
<div class="thm" data-kind="Proposition" data-name="SLLN with a fourth moment">
<p>If the $X_i$ are IID with $\E X_1^4<\infty$, then $\bar X_n\asto\mu$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $Y_i = X_i-\mu$ and $S_n = \sum_{i\le n}Y_i$. Expanding $\E S_n^4 = \sum_{i,j,k,l}\E Y_iY_jY_kY_l$, every term containing some index exactly once vanishes by independence and $\E Y_i = 0$. What survives are the $n$ terms with all four indices equal and the $3n(n-1)$ terms with two distinct pairs, so
$$\E S_n^4 = n\,\E Y_1^4 + 3n(n-1)\sigma^4 \le Cn^2.$$
By Markov's inequality applied to $S_n^4$, $\P(|\bar X_n-\mu|>\epsilon) = \P(S_n^4>n^4\epsilon^4)\le\dfrac{C}{n^2\epsilon^4}$, which is summable in $n$. By the first Borel–Cantelli lemma, $|\bar X_n-\mu|>\epsilon$ happens only finitely often, almost surely. Intersecting over $\epsilon = 1/k$, $k\in\N$ (a countable intersection of probability-one events) gives $\bar X_n\to\mu$ a.s.</p>
</details>
<p>Compare with the Chebyshev bound $\sigma^2/(n\epsilon^2)$: it is <em>not</em> summable, which is why a variance alone does not yield the strong law by this route. The fourth moment buys a $1/n^2$ tail — the general proof buys summability along a sparse subsequence instead.</p>
<details class="proof"><summary>Proof of the converse ($\E|X_1|=\infty$)</summary>
<p>$\sum_n\P(|X_1|>n)\ge\int_1^\infty\P(|X_1|>t)\d t = \E|X_1| - \int_0^1 \P(|X_1|>t)\d t = \infty$. The events $\{|X_n|>n\}$ are independent with the same probabilities, so by Borel–Cantelli II, $|X_n|>n$ infinitely often a.s. But $X_n/n = \bar X_n - \frac{n-1}{n}\bar X_{n-1}$, so if $\bar X_n$ converged to a finite limit we would have $X_n/n\to0$. Hence $\bar X_n$ cannot converge to a finite limit; a refinement of the same argument gives $\limsup|\bar X_n|=\infty$.</p>
</details>
<div class="note" data-name="How close does a path stay?">
<p>The SLLN says the deviation $\bar X_n-\mu$ goes to zero along each path; the <strong>law of the iterated logarithm</strong> (Hartman–Wintner, 1941) gives its exact size when $\sigma^2<\infty$: $\limsup_n\dfrac{\sqrt n(\bar X_n-\mu)}{\sigma\sqrt{2\log\log n}} = 1$ almost surely. So each path keeps touching the envelope $\pm\sigma\sqrt{2\log\log n/n}$, slightly wider than the CLT's $\pm\sigma/\sqrt n$ scale. The lab can draw this envelope.</p>
</div>`,
        lab: 'lln-paths',
      },
      {
        title: 'When the law fails',
        html: R`
<div class="ex" data-name="The Cauchy distribution: averaging does nothing">
<p>Let $X_i$ be IID standard Cauchy, with density $1/(\pi(1+x^2))$ and characteristic function $\varphi(t) = e^{-|t|}$. Then
$$\E e^{it\bar X_n} = \varphi(t/n)^n = \big(e^{-|t|/n}\big)^n = e^{-|t|},$$
so $\bar X_n$ is <em>exactly</em> standard Cauchy for every $n$. Averaging a million Cauchy observations gives an estimate of the centre no better than a single observation. The obstruction is $\E|X| = \int|x|/(\pi(1+x^2))\d x = \infty$: rare enormous observations arrive often enough to drag the mean around forever. (The sample <em>median</em> is consistent for the centre, at rate $1/\sqrt n$ — see <a href="#/t/order-statistics">order statistics</a>.)</p>
</div>
<div class="ex" data-name="Infinite variance but finite mean">
<p>For a Pareto distribution with tail index $\alpha\in(1,2)$, $\E X<\infty$ but $\Var X=\infty$. The SLLN still holds, but convergence is slow and jumpy: the fluctuations of $\bar X_n$ are of order $n^{1/\alpha-1}$, not $n^{-1/2}$, and paths show long quiet stretches punctuated by jumps from single huge observations. For $\alpha\le1$ the mean is infinite and $\bar X_n\to\infty$ a.s.</p>
</div>
<div class="warn" data-name="Dependence without ergodicity">
<p>Let $Z\sim\N(0,1)$ and $X_i = Z+\varepsilon_i$ with $\varepsilon_i$ IID $\N(0,1)$ independent of $Z$. Each $X_i$ has mean $0$, yet $\bar X_n\asto Z$, a random limit. Every path converges — but to its own value, not to $\mu=0$. Exchangeable sequences behave this way in general (de Finetti): they are IID conditionally on a random "parameter", and the LLN recovers that parameter, not the unconditional mean. For stationary processes, the right condition is <em>ergodicity</em>, which yields the ergodic theorem (Birkhoff) — the LLN behind <a href="#/t/stationary-distributions">Markov chain</a> and MCMC averages.</p>
</div>`,
      },
      {
        title: 'Extensions you will use constantly',
        html: R`
<p>The LLN applies to any integrable function of the data, since $g(X_1), g(X_2),\dots$ are again IID. This one observation underlies a large part of statistics:</p>
<ul>
<li><strong>Consistency of moment estimators.</strong> $\frac1n\sum X_i^k\asto\E X^k$, and by the continuous mapping theorem any continuous function of sample moments converges to the same function of population moments — e.g. $S_n^2\asto\sigma^2$. This is the logic of the <a href="#/t/method-of-moments">method of moments</a>.</li>
<li><strong>Monte Carlo integration.</strong> To compute $\int h(x)f(x)\d x$, draw $X_i\iid f$ and average $h(X_i)$; the SLLN guarantees convergence whenever $\E|h(X)|<\infty$ (<a href="#/t/monte-carlo-integration">Monte Carlo integration</a>).</li>
<li><strong>The empirical CDF.</strong> For each fixed $x$, $\hat F_n(x) = \frac1n\sum\ind(X_i\le x)\asto F(x)$. The Glivenko–Cantelli theorem upgrades this to uniform convergence $\sup_x|\hat F_n(x)-F(x)|\asto0$ (<a href="#/t/empirical-cdf">empirical CDF</a>).</li>
<li><strong>Uniform laws and M-estimators.</strong> Consistency of the <a href="#/t/mle">MLE</a> needs $\frac1n\sum\log f_\theta(X_i)\to\E\log f_\theta(X)$ <em>uniformly</em> in $\theta$, a uniform LLN; the same idea in learning theory is uniform convergence of empirical risk (<a href="#/t/vc-theory">VC theory</a>).</li>
<li><strong>Non-identical summands.</strong> Kolmogorov's criterion: if the $X_i$ are independent with $\sum_n\Var(X_n)/n^2<\infty$, then $\frac1n\sum(X_i-\E X_i)\asto0$.</li>
</ul>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="The LLN gives no rate">
<p>Without a variance, $\bar X_n$ converges but arbitrarily slowly. With a variance, the error is $O_p(\sigma/\sqrt n)$ — each extra decimal digit of accuracy costs a factor of $100$ in sample size.</p>
</div>
<div class="warn" data-name="The gambler's fallacy">
<p>The LLN does not say that deviations are "corrected". After 10 extra heads, the expected excess of heads after $n$ more fair tosses is still 10. The <em>proportion</em> converges because the fixed excess is divided by a growing $n$, not because future tosses compensate.</p>
</div>
<div class="warn" data-name="Sample averages of heavy-tailed data">
<p>A finite theoretical mean is not enough for practical stability. If the tail index is near $1$, a sample of a million may still be dominated by its largest few observations. Inspect the tail (e.g. a log-log survival plot) before trusting a mean; consider medians, trimmed means, or <a href="#/t/tail-index">tail-index</a> methods.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why does Chebyshev's inequality prove the weak law but not (directly) the strong law, even with finite variance?`,
        a: R`Chebyshev gives $\P(|\bar X_n-\mu|>\epsilon)\le\sigma^2/(n\epsilon^2)$, which tends to $0$ (weak law) but is not summable in $n$, so Borel–Cantelli I cannot be applied to all $n$. One needs either a faster tail (a fourth moment gives $C/n^2$) or the subsequence trick: along $n_k = \lfloor\alpha^k\rfloor$ the Chebyshev bounds are summable, and the gaps are filled by a monotonicity argument.` },
      { q: R`Show that the sample mean of $n$ IID standard Cauchy variables is standard Cauchy. Which hypothesis of the SLLN fails?`,
        a: R`The Cauchy characteristic function is $e^{-|t|}$, so the ch.f. of $\bar X_n$ is $(e^{-|t|/n})^n = e^{-|t|}$. The hypothesis $\E|X|<\infty$ fails ($\int|x|/(1+x^2)\d x$ diverges), and by the converse to the SLLN $\bar X_n$ cannot converge a.s.` },
      { q: R`Let $X_i = Z+\varepsilon_i$ with a common random $Z$. What does $\bar X_n$ converge to, and why doesn't this contradict the LLN?`,
        a: R`$\bar X_n = Z+\bar\varepsilon_n\asto Z$. The $X_i$ are identically distributed but not independent (they share $Z$; $\Cov(X_i,X_j)=\Var Z>0$ for all $i\neq j$), so neither law applies. Conditionally on $Z$ they are IID with mean $Z$, and the LLN applied conditionally gives exactly this limit.` },
      { q: R`Using the LLN and the continuous mapping theorem, show that the sample correlation coefficient is consistent.`,
        a: R`$r_n$ is a continuous function of the five sample moments $\frac1n\sum X_i,\frac1n\sum Y_i,\frac1n\sum X_i^2,\frac1n\sum Y_i^2,\frac1n\sum X_iY_i$ (continuous wherever the variances are positive). Each converges a.s. to its population counterpart by the SLLN (given finite second moments), so $r_n\asto\rho$ by continuous mapping.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'clt',
    domain: 'foundations',
    order: 140,
    title: 'The Central Limit Theorem',
    subtitle: 'Why sums are approximately normal, how good the approximation is, and when it breaks',
    level: 2,
    sources: { aos: '5.4', cb: '§5.5.3', other: 'Billingsley (1995), §27; Petrov, Sums of Independent Random Variables (1975); Shevtsova (2011) for the Berry–Esseen constant' },
    prereqs: ['lln', 'mgf'],
    related: ['delta-method', 'confidence-intervals', 'sampling-distributions', 'bootstrap', 'evt-intro', 'multivariate'],
    tags: ['asymptotic normality', 'characteristic functions', 'Lindeberg', 'Berry–Esseen', 'Edgeworth'],
    summary: R`If $X_1,\dots,X_n$ are IID with mean $\mu$ and finite variance $\sigma^2$, then $\sqrt n(\bar X_n-\mu)/\sigma$ converges in distribution to $\N(0,1)$, whatever the shape of the original distribution. The proof is a two-term Taylor expansion of the characteristic function: only the mean and variance survive the scaling, and everything else washes out. Lindeberg's condition extends the result to non-identical summands; Berry–Esseen bounds the error by $C\rho/(\sigma^3\sqrt n)$; Edgeworth expansions show that the leading error is a skewness correction. The theorem fails without a finite variance, and it governs the centre of the distribution, not its far tails.`,
    survey: {
      task: 'Approximate the distribution of sums, means and (via the delta method) smooth statistics',
      assumptions: R`Independence; finite variance; Lindeberg condition for non-identical summands`,
      output: R`$\sqrt n(\bar X_n-\mu)\dto\N(0,\sigma^2)$; multivariate $\N(\mathbf 0,\bSigma)$`,
      strengths: 'Universal; needs only two moments; underlies nearly every standard error and Wald interval',
      weaknesses: R`Error $O(n^{-1/2})$ driven by skewness; poor in the tails; fails for infinite variance or strong dependence`,
      complexity: '—',
    },
    sections: [
      {
        title: 'Statement and meaning',
        html: R`
<div class="thm" data-name="Lindeberg–Lévy CLT">
<p>Let $X_1,X_2,\dots$ be IID with $\E X_1 = \mu$ and $0<\Var X_1 = \sigma^2<\infty$. Then
$$Z_n = \frac{\sqrt n(\bar X_n-\mu)}{\sigma} = \frac{S_n-n\mu}{\sigma\sqrt n}\dto\N(0,1).$$
Equivalently, $\P(Z_n\le z)\to\Phi(z)$ for every $z$, and (Pólya) the convergence is uniform in $z$.</p>
</div>
<p>Three features deserve emphasis. First, the scaling: the <a href="#/t/lln">LLN</a> says $\bar X_n-\mu\to0$; the CLT magnifies by $\sqrt n$ — exactly the factor that keeps the variance at $\sigma^2$ — and finds a nondegenerate limit. Second, <strong>universality</strong>: the limit does not depend on the distribution of $X$ beyond $\mu$ and $\sigma^2$. Third, the statement is about distributions, not paths: $Z_n$ does not converge to anything in probability (indeed $\Corr(Z_n,Z_{4n}) = 1/2$ for every $n$, so the sequence never settles).</p>
<p>Why the normal? Because it is the unique finite-variance distribution that is <em>stable under averaging</em>: if $X, X'$ are IID $\N(0,1)$ then $(X+X')/\sqrt2$ is again $\N(0,1)$. The CLT says that repeated averaging (with $\sqrt n$ rescaling) drives any finite-variance distribution to this fixed point. In the language of <a href="#/t/entropy-kl">information theory</a>, entropy increases monotonically along the normalised sums toward its maximum at the normal (Artstein, Ball, Barthe & Naor, 2004).</p>
<p>The <strong>multivariate CLT</strong> follows from the univariate one by the Cramér–Wold device: if $\mathbf X_i\in\R^d$ are IID with mean $\bmu$ and covariance $\bSigma$, then $\sqrt n(\bar{\mathbf X}_n-\bmu)\dto\N_d(\mathbf 0,\bSigma)$, because every projection $\mathbf a^\top\mathbf X_i$ satisfies the univariate CLT with variance $\mathbf a^\top\bSigma\mathbf a$.</p>`,
      },
      {
        title: 'Proof via characteristic functions',
        html: R`
<p>Standardise: let $Y_i = (X_i-\mu)/\sigma$, so $\E Y=0$, $\E Y^2=1$, and $Z_n = n^{-1/2}\sum Y_i$. Let $\varphi(t)=\E e^{itY}$. By independence $\E e^{itZ_n} = \varphi(t/\sqrt n)^n$; by <a href="#/t/mgf">Lévy's continuity theorem</a> it suffices to show this tends to $e^{-t^2/2}$, the $\N(0,1)$ characteristic function.</p>
<div class="thm" data-kind="Lemma" data-name="Second-order expansion">
<p>If $\E Y^2<\infty$, then $\varphi(t) = 1 + it\E Y - \tfrac12t^2\E Y^2 + o(t^2)$ as $t\to0$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For real $x$, Taylor's theorem with integral remainder gives $\big|e^{ix} - (1+ix-\tfrac12x^2)\big|\le\min\big(|x|^3/6,\;|x|^2\big)$. Apply with $x = tY$ and take expectations:
$$\big|\varphi(t)-1-it\E Y+\tfrac12t^2\E Y^2\big|\le t^2\,\E\big[\min(|t||Y|^3/6,\;|Y|^2)\big].$$
The expectation on the right tends to $0$ as $t\to0$ by dominated convergence (the integrand is dominated by $|Y|^2$, integrable, and tends to $0$ pointwise). Hence the error is $o(t^2)$.</p>
</details>
<div class="thm" data-kind="Lemma" data-name="Products of complex numbers">
<p>If $z_1,\dots,z_n$ and $w_1,\dots,w_n$ are complex with all moduli $\le1$, then $\big|\prod z_i - \prod w_i\big|\le\sum|z_i-w_i|$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Induction: $\prod_{i\le n}z_i - \prod_{i\le n}w_i = z_n\big(\prod_{i\le n-1}z_i-\prod_{i\le n-1}w_i\big) + (z_n-w_n)\prod_{i\le n-1}w_i$, and both multipliers have modulus at most $1$.</p>
</details>
<details class="proof"><summary>Proof of the CLT</summary>
<p>By the first lemma with $\E Y=0$, $\E Y^2=1$: $\varphi(t/\sqrt n) = 1 - \frac{t^2}{2n} + o(1/n)$ for fixed $t$. For $n > t^2/2$, both $\varphi(t/\sqrt n)$ and $w_n = 1-t^2/(2n)$ have modulus at most $1$, so by the second lemma
$$\big|\varphi(t/\sqrt n)^n - w_n^n\big|\le n\,\big|\varphi(t/\sqrt n)-w_n\big| = n\cdot o(1/n)\to0.$$
And $w_n^n = (1-t^2/(2n))^n\to e^{-t^2/2}$. Therefore $\E e^{itZ_n}\to e^{-t^2/2}$ for every $t$, and Lévy's theorem gives $Z_n\dto\N(0,1)$.</p>
</details>
<p>The proof shows precisely why universality holds: after scaling by $\sqrt n$, the third and higher cumulants enter $\log\varphi(t/\sqrt n)^n$ with factors $n\cdot n^{-k/2}\to0$ for $k\ge3$. Only the second cumulant — the variance — survives. The first omitted term, of order $\kappa_3/\sqrt n$, is the skewness correction that Edgeworth expansions retain.</p>`,
      },
      {
        title: 'Beyond identical distributions: Lindeberg and Lyapunov',
        html: R`
<p>Many statistics are sums of independent but <em>non-identical</em> terms — a regression coefficient is $\sum a_i\varepsilon_i$ with design-dependent weights. Consider a triangular array: for each $n$, independent $X_{n,1},\dots,X_{n,k_n}$ with $\E X_{n,i}=0$ and $\sum_i\Var X_{n,i}=1$.</p>
<div class="thm" data-name="Lindeberg–Feller CLT">
<p>(Lindeberg) If for every $\epsilon>0$
$$L_n(\epsilon) = \sum_{i=1}^{k_n}\E\big[X_{n,i}^2\,\ind(|X_{n,i}|>\epsilon)\big]\to0,$$
then $\sum_iX_{n,i}\dto\N(0,1)$. (Feller) Conversely, if $\max_i\Var X_{n,i}\to0$ and $\sum_iX_{n,i}\dto\N(0,1)$, then the Lindeberg condition holds.</p>
</div>
<p>The Lindeberg condition says no single summand contributes a non-negligible share of the variance through large values: the sum is the accumulation of many individually small effects. The proof is the same characteristic-function argument, with the Lindeberg condition used to bound the third-order remainder term by term.</p>
<p><strong>Lyapunov's condition</strong> is easier to check: if $\sum_i\E|X_{n,i}|^{2+\delta}\to0$ for some $\delta>0$, then Lindeberg holds, because $\E[X^2\ind(|X|>\epsilon)]\le\E|X|^{2+\delta}/\epsilon^\delta$.</p>
<div class="ex" data-name="Weighted sums and regression">
<p>Let $\varepsilon_i$ be IID with mean $0$, variance $\sigma^2$, and let $T_n = \sum_ia_{n,i}\varepsilon_i$. Then $T_n/(\sigma\|a_n\|)\dto\N(0,1)$ provided $\max_ia_{n,i}^2/\sum_ja_{n,j}^2\to0$ (Hájek–Šidák). Check Lindeberg: the summands are $c_i\varepsilon_i$ with $c_i = a_{n,i}/(\sigma\|a_n\|)$, so $\sum_ic_i^2 = 1/\sigma^2$ and, for any $\eta>0$,
$$L_n(\eta) = \sum_ic_i^2\,\E\big[\varepsilon^2;\,|\varepsilon|>\eta/|c_i|\big]\le\frac{1}{\sigma^2}\,\E\big[\varepsilon^2;\,|\varepsilon|>\eta/\textstyle\max_j|c_j|\big]\to0,$$
because $\max_j|c_j|\to0$ pushes the truncation level to infinity and $\E\varepsilon^2<\infty$. In <a href="#/t/ols-geometry">least squares</a> the condition becomes "no observation has non-vanishing leverage": a single high-leverage point keeps its own non-normal error distribution in the estimate forever.</p>
</div>`,
      },
      {
        title: 'How good is the approximation?',
        html: R`
<div class="thm" data-name="Berry–Esseen">
<p>If the $X_i$ are IID with $\sigma^2>0$ and $\rho = \E|X-\mu|^3<\infty$, then
$$\sup_z\big|\P(Z_n\le z)-\Phi(z)\big|\le\frac{C\,\rho}{\sigma^3\sqrt n},$$
with a universal constant $C$. The best known value is $C\le0.4748$ (Shevtsova, 2011); $C$ cannot be below $0.4097$ (Esseen, 1956).</p>
</div>
<p>The rate $n^{-1/2}$ cannot be improved in general — it is attained by lattice distributions such as Bernoulli($1/2$), where the CDF of $Z_n$ has jumps of height about $0.8/\sqrt n$ at the centre that no continuous $\Phi$ can follow. The ratio $\rho/\sigma^3\ge1$ is a measure of how "non-normal" a single observation is; it is large for skewed or heavy-tailed data.</p>
<div class="thm" data-name="One-term Edgeworth expansion">
<p>If in addition $X$ is non-lattice, then uniformly in $z$,
$$\P(Z_n\le z) = \Phi(z) - \frac{\gamma}{6\sqrt n}(z^2-1)\,\phi(z) + o(n^{-1/2}),\qquad \gamma = \frac{\E(X-\mu)^3}{\sigma^3}.$$</p>
</div>
<p>The leading error is a <strong>skewness correction</strong>: for right-skewed data ($\gamma>0$), the true CDF lies <em>above</em> $\Phi$ near the centre ($z^2<1$), so too much mass sits left of zero — the distribution of $\bar X_n$ is itself right-skewed, with a longer right tail and a shorter left tail than the normal. This is why normal-theory intervals for skewed data miss asymmetrically, and why the studentised <a href="#/t/bootstrap">bootstrap</a>, which estimates $\gamma$ implicitly, is second-order accurate.</p>
<div class="ex" data-name="How large is large?">
<p>For exponential data ($\gamma=2$, $\rho/\sigma^3 = 12/e - 2\approx2.41$), Berry–Esseen guarantees $\sup|F_n-\Phi|\le0.4748\times2.41/\sqrt n\approx1.15/\sqrt n$, which is below $0.01$ only for $n\gtrsim13{,}000$. The actual error is much smaller: the Edgeworth term is largest at $z=0$, where it equals $\gamma\phi(0)/(6\sqrt n)\approx0.133/\sqrt n$ — about $0.013$ at $n=100$, nearly nine times below the guarantee. Berry–Esseen is a worst-case bound over all distributions with the same $\rho/\sigma^3$; the Edgeworth term is the actual leading error for this one. The "$n\ge30$" rule of thumb has no theoretical basis: the required $n$ scales like $\gamma^2$.</p>
</div>`,
        lab: 'clt-sums',
      },
      {
        title: 'Watching the error decay',
        html: R`
<p>Berry–Esseen predicts a slope of $-\tfrac12$ on a log-log plot of the Kolmogorov distance $\sup_z|F_n(z)-\Phi(z)|$ against $n$. The Edgeworth expansion refines this: for non-lattice data, subtracting the skewness term leaves an error of order $1/n$ (slope $-1$, given a finite fourth moment and Cramér's condition). For lattice data, it does not — the discreteness itself contributes an irreducible $n^{-1/2}$ term, which a continuity correction can partly remove.</p>
<p>The lab computes these distances <em>exactly</em> for two families where the distribution of the sum is known in closed form: gamma summands (the sum is gamma) and Bernoulli summands (the sum is binomial).</p>`,
        lab: 'clt-berry-esseen',
      },
      {
        title: 'When the CLT fails or misleads',
        html: R`
<div class="warn" data-name="Infinite variance: stable limits">
<p>If $\P(|X|>x)\sim cx^{-\alpha}$ with $0<\alpha<2$, then $\Var X=\infty$ and the normal limit fails. The <strong>generalised CLT</strong> says $(S_n-b_n)/n^{1/\alpha}$ converges to an $\alpha$-stable law — heavy-tailed, with the same tail index. The Cauchy distribution is the $\alpha=1$ case and is its own limit. In the first lab, choose a Pareto source with $\alpha<2$ and watch the standardised sums refuse to become normal.</p>
</div>
<div class="warn" data-name="The CLT is about the centre, not the tails">
<p>$\P(Z_n>z)\approx1-\Phi(z)$ has small <em>absolute</em> error, but for large $z$ the <em>relative</em> error can be enormous: probabilities like $\P(\bar X_n>\mu+c)$ for fixed $c$ are governed by <a href="#/t/inequalities">large deviations</a> ($e^{-nI(\mu+c)}$), not by $\Phi$. Tail risk calculations built on the CLT routinely underestimate extreme sums. For maxima rather than sums, the relevant limit theory is <a href="#/t/evt-intro">extreme value theory</a>.</p>
</div>
<div class="warn" data-name="Dependence changes the variance">
<p>For a stationary sequence with autocovariances $\gamma_k$, $n\Var\bar X_n\to\sigma^2_\infty = \gamma_0+2\sum_{k\ge1}\gamma_k$ (when summable), and under mixing conditions $\sqrt n(\bar X_n-\mu)\dto\N(0,\sigma^2_\infty)$. Using $\gamma_0=\sigma^2$ instead — treating the data as IID — gives standard errors that are too small for positively correlated data. With long-range dependence ($\sum|\gamma_k|=\infty$), even the $\sqrt n$ rate fails.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In the characteristic-function proof, where exactly is $\E X^2<\infty$ used, and why do the higher moments not matter?`,
        a: R`It is used to expand $\varphi(t) = 1 - t^2/2 + o(t^2)$ (via dominated convergence with dominating function $|Y|^2$). After scaling $t\to t/\sqrt n$ and raising to the $n$-th power, the $k$-th cumulant contributes a term of order $n\cdot n^{-k/2}$, which vanishes for $k\ge3$; so only the variance survives in the limit.` },
      { q: R`For Bernoulli($p$) summands, compute the Berry–Esseen ratio $\rho/\sigma^3$ and explain why the normal approximation to the binomial is poor for small $p$ at moderate $n$.`,
        a: R`$\rho = \E|X-p|^3 = p(1-p)\big[p^2+(1-p)^2\big]$ and $\sigma^3 = (p(1-p))^{3/2}$, so $\rho/\sigma^3 = \big[p^2+(1-p)^2\big]/\sqrt{p(1-p)}$, which behaves like $1/\sqrt p$ as $p\to0$. The bound is $C/\sqrt{np}$ in that regime: what matters is the expected number of successes $np$, not $n$. (The skewness $(1-2p)/\sqrt{p(1-p)}$ blows up the same way; the Poisson approximation is better there.)` },
      { q: R`State the Lindeberg condition and give an example of independent summands with finite variances for which the CLT fails.`,
        a: R`$\sum_i\E[X_{n,i}^2\ind(|X_{n,i}|>\epsilon)]\to0$ for all $\epsilon$ (summands normalised so $\sum\Var=1$). Counterexample: $X_{n,1}\sim\text{Unif}\{-1,1\}\cdot\sqrt{1/2}$ carrying half the variance and $n$ small normal terms carrying the rest — the sum converges to a non-normal mixture because one summand never becomes negligible. Feller's converse shows that when all variances are uniformly small, Lindeberg is necessary.` },
      { q: R`For right-skewed data, is the normal-approximation confidence interval $\bar X\pm1.96\,s/\sqrt n$ more likely to miss on the left or on the right of $\mu$? Use the Edgeworth expansion.`,
        a: R`For right-skewed data, $\bar X$ and $s$ are positively correlated: samples with a large mean tend to have a large $s$ (a wide interval), samples with a small mean a small $s$. So $T = \sqrt n(\bar X-\mu)/s$ is <em>left</em>-skewed; its Edgeworth expansion is $\P(T\le z) = \Phi(z)+\gamma(2z^2+1)\phi(z)/(6\sqrt n)+o(n^{-1/2})$, which puts more than $2.5\%$ below $-1.96$ and less than $2.5\%$ above $1.96$. The interval therefore misses mostly by lying entirely <em>below</em> $\mu$. (With $\sigma$ known the skew of $Z_n$ is to the right, $\P(Z_n\le z) = \Phi(z)-\gamma(z^2-1)\phi(z)/(6\sqrt n)+\dots$, and the misses flip to the other side — studentising reverses the asymmetry.)` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'delta-method',
    domain: 'foundations',
    order: 150,
    title: 'The Delta Method',
    subtitle: 'Asymptotic distributions of smooth functions of asymptotically normal statistics',
    level: 2,
    sources: { aos: '5.5, 9.9', cb: '§5.5.4', other: 'van der Vaart, Asymptotic Statistics (1998), Ch. 3 and §20 (functional delta method)' },
    prereqs: ['clt', 'convergence'],
    related: ['mle', 'fisher-information', 'bootstrap', 'confidence-intervals', 'statistical-functionals', 'order-statistics'],
    tags: ['asymptotic variance', 'linearisation', 'variance stabilisation', 'Taylor expansion'],
    summary: R`If $\sqrt n(T_n-\theta)\dto\N(0,\sigma^2)$ and $g$ is differentiable at $\theta$, then $\sqrt n(g(T_n)-g(\theta))\dto\N(0,g'(\theta)^2\sigma^2)$. Locally, a smooth function is linear, and a linear function of a normal is normal. This one line gives standard errors for log-odds, rates, ratios, correlations and every smooth function of an MLE; it suggests variance-stabilising transformations; and when $g'(\theta)=0$ it fails in an instructive way, giving a scaled chi-square limit at rate $n$ instead of a normal at rate $\sqrt n$.`,
    survey: {
      task: R`Standard errors and limit laws for $g(\hat\theta)$ given those of $\hat\theta$`,
      assumptions: R`$\sqrt n(T_n-\theta)\dto\N(0,\Sigma)$; $g$ differentiable at $\theta$ with $\nabla g(\theta)\neq0$`,
      output: R`$\N\big(g(\theta),\,\nabla g^\top\Sigma\nabla g/n\big)$ approximation; Wald intervals for $g(\theta)$`,
      strengths: 'Analytic, instant, works for any smooth function; basis of variance-stabilising transforms',
      weaknesses: R`First-order only: poor when $g$ is strongly curved over the spread of $T_n$, near $g'(\theta)=0$, or near boundaries`,
      complexity: R`One gradient evaluation; $O(d^2)$ for the quadratic form`,
    },
    sections: [
      {
        title: 'The problem',
        html: R`
<p>The <a href="#/t/clt">CLT</a> hands us the asymptotic distribution of a sample mean. But the quantities we report are rarely means. A proportion $\hat p$ is a mean, yet we may want the log-odds $\log\frac{\hat p}{1-\hat p}$; an exponential sample has mean $1/\lambda$, yet we want the rate $\lambda$; a two-sample study gives two means, yet we want their ratio; a <a href="#/t/mle">maximum likelihood estimate</a> $\hat\theta$ is asymptotically normal, yet we want a derived quantity $g(\hat\theta)$ such as a median survival time. In each case the target is a smooth function $g$ of an asymptotically normal statistic.</p>
<div class="note">
<p>Zoom in. As $n$ grows, $T_n$ concentrates in a window of width $O(n^{-1/2})$ around $\theta$. On such a small window any differentiable $g$ is indistinguishable from its tangent line $g(\theta)+g'(\theta)(t-\theta)$. A linear map scales a normal by $|g'(\theta)|$. So $g(T_n)$ is approximately normal with standard deviation $|g'(\theta)|$ times that of $T_n$. That is the entire method; the rest is making "indistinguishable" precise.</p>
</div>`,
      },
      {
        title: 'The first-order delta method',
        html: R`
<div class="thm" data-name="Delta method">
<p>Suppose $\sqrt n(T_n-\theta)\dto\N(0,\sigma^2)$ and $g$ is differentiable at $\theta$. Then
$$\sqrt n\big(g(T_n)-g(\theta)\big)\dto\N\big(0,\;g'(\theta)^2\sigma^2\big).$$
(If $g'(\theta)=0$ the limit is the point mass at $0$ — see the second-order method below.)</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Define $h(t) = \dfrac{g(t)-g(\theta)}{t-\theta}-g'(\theta)$ for $t\neq\theta$ and $h(\theta)=0$. Differentiability of $g$ at $\theta$ is exactly continuity of $h$ at $\theta$. Then the identity
$$\sqrt n\big(g(T_n)-g(\theta)\big) = \big(g'(\theta)+h(T_n)\big)\cdot\sqrt n(T_n-\theta)$$
holds exactly. Since $\sqrt n(T_n-\theta) = O_p(1)$, we have $T_n-\theta = O_p(n^{-1/2})\pto0$, so $h(T_n)\pto h(\theta)=0$ by the continuous mapping theorem (continuity at the single point $\theta$ suffices). By Slutsky, the right side converges in distribution to $g'(\theta)\cdot\N(0,\sigma^2)$.</p>
</details>
<p>Notice what the proof does <em>not</em> need: $g$ need not be differentiable anywhere except at $\theta$, and $g(T_n)$ need not have finite moments. The theorem is about distributions, not variances — "the asymptotic variance of $g(T_n)$" is $g'(\theta)^2\sigma^2/n$ even when $\Var g(T_n)$ is infinite.</p>
<p>In practice $\theta$ is unknown, so we estimate the standard error by plugging in: $\widehat\se\big(g(T_n)\big) = |g'(T_n)|\,\hat\sigma/\sqrt n$. If $g'$ is continuous at $\theta$ and $\hat\sigma\pto\sigma$, Slutsky justifies the Wald interval $g(T_n)\pm z_{\alpha/2}\,\widehat\se$.</p>
<div class="ex" data-name="Log-odds">
<p>For $\hat p = \bar X_n$ with Bernoulli($p$) data, $\sigma^2 = p(1-p)$ and $g(p) = \log\frac{p}{1-p}$ has $g'(p) = \frac1{p(1-p)}$. So $\sqrt n(\text{logit}\,\hat p-\text{logit}\,p)\dto\N\big(0,\frac{1}{p(1-p)}\big)$, and $\widehat\se = 1/\sqrt{n\hat p(1-\hat p)}$. This is the standard error behind <a href="#/t/logistic-regression">logistic regression</a> coefficients in the one-sample case.</p>
</div>
<div class="ex" data-name="An exponential rate">
<p>For $X_i\iid\text{Exp}(\lambda)$, $\bar X_n$ has mean $1/\lambda$ and variance $1/(n\lambda^2)$. The MLE $\hat\lambda = 1/\bar X_n$ uses $g(x) = 1/x$ with $g'(1/\lambda) = -\lambda^2$, so $\sqrt n(\hat\lambda-\lambda)\dto\N(0,\lambda^4\cdot\lambda^{-2}) = \N(0,\lambda^2)$ — agreeing with the inverse <a href="#/t/fisher-information">Fisher information</a> $I(\lambda)^{-1} = \lambda^2$, as it must, since the delta method preserves efficiency.</p>
</div>`,
      },
      {
        title: 'Several parameters',
        html: R`
<div class="thm" data-name="Multivariate delta method">
<p>If $\sqrt n(\mathbf T_n-\btheta)\dto\N_k(\mathbf 0,\bSigma)$ and $g:\R^k\to\R^m$ is differentiable at $\btheta$ with Jacobian $J = \partial g/\partial\btheta^\top$ ($m\times k$), then $\sqrt n\big(g(\mathbf T_n)-g(\btheta)\big)\dto\N_m(\mathbf 0,\,J\bSigma J^\top)$. For scalar $g$: variance $\nabla g(\btheta)^\top\bSigma\,\nabla g(\btheta)$.</p>
</div>
<p>The proof is identical, with the scalar remainder replaced by $g(\mathbf t)-g(\btheta)-J(\mathbf t-\btheta) = o(\|\mathbf t-\btheta\|)$.</p>
<div class="ex" data-name="A ratio of means">
<p>Let $(X_i,Y_i)$ be IID pairs with means $(\mu_X,\mu_Y)$, $\mu_Y\neq0$, variances $\sigma_X^2,\sigma_Y^2$ and covariance $\sigma_{XY}$. For $R = \bar X/\bar Y$ and $g(x,y)=x/y$, $\nabla g = (1/\mu_Y,\,-\mu_X/\mu_Y^2)$, so with $r=\mu_X/\mu_Y$,
$$n\,\Var_{\text{asy}}(R) = \frac{\sigma_X^2-2r\,\sigma_{XY}+r^2\sigma_Y^2}{\mu_Y^2}.$$
Positive correlation between numerator and denominator <em>reduces</em> the variance of the ratio — the principle behind ratio estimators in survey sampling.</p>
</div>
<div class="ex" data-name="The sample variance and kurtosis">
<p>$S^2$ is a smooth function of $(\bar X,\overline{X^2})$. Working through the gradient gives $\sqrt n(S^2-\sigma^2)\dto\N(0,\mu_4-\sigma^4)$, where $\mu_4 = \E(X-\mu)^4$. For normal data $\mu_4 = 3\sigma^4$ and the variance is $2\sigma^4$; for heavy-tailed data the variance of $S^2$ can be many times larger. This is why <a href="#/t/sampling-distributions">chi-square tests for a variance</a> are not robust, while $t$-tests for a mean are.</p>
</div>`,
      },
      {
        title: 'Variance-stabilising transformations',
        html: R`
<p>Often the asymptotic variance depends on the parameter: $\sqrt n(T_n-\theta)\dto\N(0,\sigma^2(\theta))$. Then the width of a Wald interval depends on the unknown $\theta$, and the normal approximation degrades where $\sigma(\theta)$ changes fast. The delta method suggests a fix: pick $g$ with
$$g'(\theta) = \frac{c}{\sigma(\theta)},\qquad\text{i.e.}\qquad g(\theta) = \int^\theta\frac{c}{\sigma(u)}\d u,$$
so that $\sqrt n(g(T_n)-g(\theta))\dto\N(0,c^2)$, free of $\theta$.</p>
<table>
<thead><tr><th>Setting</th><th>$\sigma^2(\theta)$</th><th>Stabilising $g$</th><th>Limit variance</th></tr></thead>
<tbody>
<tr><td>Poisson mean $\lambda$</td><td>$\lambda$</td><td>$\sqrt\lambda$</td><td>$1/4$</td></tr>
<tr><td>Binomial proportion $p$</td><td>$p(1-p)$</td><td>$\arcsin\sqrt p$</td><td>$1/4$</td></tr>
<tr><td>Normal correlation $\rho$</td><td>$(1-\rho^2)^2$</td><td>$\tanh^{-1}\rho$ (Fisher's $z$)</td><td>$1$</td></tr>
<tr><td>Normal variance $\sigma^2$</td><td>$2\sigma^4$</td><td>$\log\sigma^2$</td><td>$2$</td></tr>
<tr><td>Exponential mean $\mu$</td><td>$\mu^2$</td><td>$\log\mu$</td><td>$1$</td></tr>
</tbody>
</table>
<p>For Fisher's $z$: $\int d\rho/(1-\rho^2) = \tanh^{-1}\rho = \frac12\log\frac{1+\rho}{1-\rho}$. Build the interval on the stabilised scale, $\tanh^{-1}r\pm z_{\alpha/2}/\sqrt{n}$ (in practice $\sqrt{n-3}$), and map back with $\tanh$. The resulting interval respects the boundary $|\rho|<1$ and is asymmetric in the right direction. Stabilising transforms also tend to <em>symmetrise</em>, which improves the normal approximation beyond what the first-order theory promises.</p>`,
      },
      {
        title: 'The second-order delta method',
        html: R`
<p>When $g'(\theta)=0$, the first-order theorem gives the degenerate limit $\N(0,0)$: $g(T_n)-g(\theta)$ is smaller than $O_p(n^{-1/2})$. The quadratic term takes over.</p>
<div class="thm" data-name="Second-order delta method">
<p>Suppose $\sqrt n(T_n-\theta)\dto\N(0,\sigma^2)$, $g'(\theta)=0$, and $g''(\theta)$ exists and is nonzero. Then
$$n\big(g(T_n)-g(\theta)\big)\dto\frac{\sigma^2\,g''(\theta)}{2}\,\chi^2_1.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By Taylor's theorem (Peano form) at $\theta$, $g(t) = g(\theta)+\tfrac12g''(\theta)(t-\theta)^2 + r(t)(t-\theta)^2$ with $r(t)\to0$ as $t\to\theta$. Multiply by $n$:
$$n\big(g(T_n)-g(\theta)\big) = \Big(\tfrac12g''(\theta)+r(T_n)\Big)\big(\sqrt n(T_n-\theta)\big)^2.$$
Since $T_n\pto\theta$, $r(T_n)\pto0$; by the continuous mapping theorem $\big(\sqrt n(T_n-\theta)\big)^2\dto\sigma^2Z^2 = \sigma^2\chi^2_1$; Slutsky finishes.</p>
</details>
<div class="ex" data-name="Variance of a Bernoulli at $p=1/2$">
<p>Estimate $\tau = p(1-p)$ by $\hat p(1-\hat p)$. Here $g'(p) = 1-2p$, so for $p\neq1/2$ the first-order method gives $\sqrt n(\hat\tau-\tau)\dto\N(0,(1-2p)^2p(1-p))$. At $p=1/2$, $g'=0$, $g''=-2$, $\sigma^2=1/4$, and $n(\hat\tau-1/4)\dto-\tfrac14\chi^2_1$: the estimate is always below $1/4$, and its error shrinks at rate $1/n$, not $1/\sqrt n$.</p>
</div>
<p>Near (but not at) a stationary point, both regimes compete: the linear term has size $|g'(\theta)|\sigma/\sqrt n$ and the quadratic term $|g''(\theta)|\sigma^2/(2n)$. The first-order approximation is accurate only when the linear term dominates, i.e. when $\sqrt n\,|g'(\theta)|\gg|g''(\theta)|\sigma$. The lab draws both approximations; move $\mu$ towards a stationary point of $g$ and watch the normal approximation lose to the skewed second-order law.</p>`,
        lab: 'delta-method-lab',
      },
      {
        title: 'Bias, moments and the bigger picture',
        html: R`
<p>A second-order expansion also gives the leading <strong>bias</strong>: when the moments exist, $\E g(\bar X_n) = g(\mu)+\frac{g''(\mu)\sigma^2}{2n}+O(n^{-2})$. Convex $g$ biases upward, concave $g$ downward — Jensen's inequality, quantified. Subtracting an estimate of this term gives a bias-corrected estimator, with the same first-order variance.</p>
<div class="warn" data-name="The asymptotic variance need not be a variance">
<p>For normal data, $1/\bar X_n$ has $\E|1/\bar X_n| = \infty$ for every $n$ (the density of $\bar X_n$ is positive at $0$), yet the delta method correctly gives $\sqrt n(1/\bar X_n-1/\mu)\dto\N(0,\sigma^2/\mu^4)$ for $\mu\neq0$. The approximation describes the bulk of the distribution, not its moments. Simulation-based standard errors of such statistics can be wildly unstable.</p>
</div>
<div class="warn" data-name="Curvature over the sampling spread">
<p>The method linearises over a window of width $\approx\sigma/\sqrt n$. If $g$ changes slope substantially over that window — $\log$ near $0$, $1/x$ near $0$, logit near $0$ or $1$ — the linear approximation is poor even if $n$ is "large". A useful diagnostic is the ratio of the two Taylor terms, $|g''|\sigma/(\sqrt n|g'|)$.</p>
</div>
<p><strong>The functional delta method.</strong> Many statistics are functionals $T(\hat F_n)$ of the empirical distribution — medians, quantiles, trimmed means. Since $\sqrt n(\hat F_n-F)$ converges to a Brownian bridge (Donsker), a suitable notion of differentiability of $T$ (Hadamard) gives $\sqrt n(T(\hat F_n)-T(F))\dto\N\big(0,\E\,\mathrm{IF}(X)^2\big)$, where $\mathrm{IF}$ is the <a href="#/t/statistical-functionals">influence function</a>. For the median $m$ of a density $f$ with $f(m)>0$, this yields $\sqrt n(\hat m_n-m)\dto\N\big(0,1/(4f(m)^2)\big)$ (derived directly in <a href="#/t/order-statistics">order statistics</a>). The same differentiability is what makes the <a href="#/t/bootstrap">bootstrap</a> consistent for these statistics.</p>`,
      },
    ],
    quiz: [
      { q: R`Let $X_i\iid\text{Poisson}(\lambda)$. Find the asymptotic distribution of $\sqrt n(\sqrt{\bar X_n}-\sqrt\lambda)$ and explain why this is useful.`,
        a: R`$g(x)=\sqrt x$, $g'(\lambda) = 1/(2\sqrt\lambda)$, $\sigma^2=\lambda$, so the limit is $\N(0,\lambda\cdot\frac1{4\lambda}) = \N(0,1/4)$. The variance no longer depends on $\lambda$: intervals $\sqrt{\bar X}\pm z_{\alpha/2}/(2\sqrt n)$ have the same width everywhere and can be squared back to an interval for $\lambda$.` },
      { q: R`For $X_i\iid\N(0,1)$, what is the limit of $\sqrt n\,\bar X_n^2$? Of $n\bar X_n^2$?`,
        a: R`$g(x)=x^2$ has $g'(0)=0$, so $\sqrt n\,\bar X_n^2\pto0$ (first-order delta method gives a degenerate limit). The second-order method with $g''=2$, $\sigma^2=1$: $n\bar X_n^2\dto\chi^2_1$ — in fact exactly $\chi^2_1$ for every $n$, since $\sqrt n\bar X_n\sim\N(0,1)$.` },
      { q: R`Derive the asymptotic variance of the sample correlation's Fisher transform, given $\sqrt n(r-\rho)\dto\N(0,(1-\rho^2)^2)$ under bivariate normality.`,
        a: R`$g(\rho)=\tanh^{-1}\rho$, $g'(\rho) = 1/(1-\rho^2)$. The asymptotic variance is $(1-\rho^2)^2/(1-\rho^2)^2 = 1$, so $\sqrt n(\tanh^{-1}r-\tanh^{-1}\rho)\dto\N(0,1)$, free of $\rho$.` },
      { q: R`Why does the delta-method Wald interval for $p(1-p)$ have poor coverage near $p=1/2$ even for large $n$?`,
        a: R`The plug-in standard error is $|1-2\hat p|\sqrt{\hat p(1-\hat p)/n}$, which is nearly $0$ when $\hat p\approx1/2$, so the interval collapses to a point. Meanwhile the true sampling distribution is a reflected $\frac1{4n}\chi^2_1$ entirely below $1/4$: the first-order approximation is degenerate exactly where the second-order term dominates.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'sampling-distributions',
    domain: 'foundations',
    order: 160,
    title: 'Sampling from the Normal: χ², t and F',
    subtitle: 'Exact finite-sample laws of the mean and variance of a normal sample, and the families they generate',
    level: 2,
    sources: { aos: '2.4', cb: '§5.3', other: 'Student (1908); Cochran (1934); Box (1953) on non-robustness of variance tests' },
    prereqs: ['distribution-families', 'transformations', 'multivariate'],
    related: ['confidence-intervals', 'regression-inference', 'anova', 'hypothesis-testing', 'clt', 'mgf'],
    tags: ['chi-square', "Student's t", 'F distribution', 'Cochran', 'exact inference'],
    summary: R`For a normal sample, the sample mean and sample variance are independent, $\bar X\sim\N(\mu,\sigma^2/n)$ and $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$ — exactly, at every $n$. From these two facts come Student's $t$ (a normal divided by an independent root-chi-square) and Snedecor's $F$ (a ratio of independent chi-squares), the reference distributions for nearly all classical exact tests and intervals. The engine behind all of it is rotational invariance of the standard normal: an orthogonal change of coordinates splits a sum of squares into independent pieces, which is Cochran's theorem. Independence of $\bar X$ and $S^2$ actually characterises the normal, so these results are exactly as fragile as that assumption.`,
    survey: {
      task: 'Exact sampling distributions for normal-theory tests and intervals',
      assumptions: R`$X_i\iid\N(\mu,\sigma^2)$ (or Gaussian errors in a linear model)`,
      output: R`$\chi^2_k$, $t_k$, $F_{m,k}$ reference laws; $t$-, $\chi^2$- and $F$-pivots`,
      strengths: 'Exact at every $n$; elegant geometry (projections and orthogonal decompositions)',
      weaknesses: 'Variance-based procedures ($\\chi^2$, $F$ for variances) are badly non-robust to non-normal kurtosis',
      complexity: 'Closed-form densities; quantiles by numerical inversion',
    },
    sections: [
      {
        title: 'The setting and the goal',
        html: R`
<p>Let $X_1,\dots,X_n\iid\N(\mu,\sigma^2)$, with sample mean $\bar X$ and sample variance $S^2 = \frac{1}{n-1}\sum(X_i-\bar X)^2$. The <a href="#/t/clt">CLT</a> tells us about $\bar X$ for large $n$ under almost any distribution. Under normality we can do much better: we can write down the <em>exact</em> joint distribution of $(\bar X,S^2)$ at every $n$, including $n=3$. That is what makes exact $t$-intervals, $\chi^2$ intervals for a variance, and the $F$-tests of <a href="#/t/anova">ANOVA</a> and <a href="#/t/regression-inference">regression</a> possible.</p>
<p>The plan: define the chi-square family; prove the fundamental theorem about $(\bar X,S^2)$ by a rotation; generalise the rotation to Cochran's theorem; then build $t$ and $F$ as ratios.</p>`,
      },
      {
        title: 'The chi-square family',
        html: R`
<div class="def" data-name="Chi-square distribution">
<p>If $Z_1,\dots,Z_k\iid\N(0,1)$, then $V = \sum_{i=1}^kZ_i^2$ has the chi-square distribution with $k$ degrees of freedom, $V\sim\chi^2_k$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Chi-square facts">
<p>(i) $\chi^2_k$ is the $\text{Gamma}(k/2,\,\text{scale }2)$ distribution, with density $f(v) = \dfrac{v^{k/2-1}e^{-v/2}}{2^{k/2}\Gamma(k/2)}$ for $v>0$. (ii) Its MGF is $(1-2t)^{-k/2}$ for $t<1/2$; its mean is $k$ and variance $2k$. (iii) If $V_1\sim\chi^2_{k_1}$ and $V_2\sim\chi^2_{k_2}$ are independent, $V_1+V_2\sim\chi^2_{k_1+k_2}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For $k=1$: $\P(Z^2\le v) = \P(-\sqrt v\le Z\le\sqrt v)$, and differentiating gives $f(v) = 2\phi(\sqrt v)\cdot\frac{1}{2\sqrt v} = \frac{1}{\sqrt{2\pi}}v^{-1/2}e^{-v/2}$, the Gamma$(1/2, 2)$ density (using $\Gamma(1/2)=\sqrt\pi$). Its MGF is $\E e^{tZ^2} = \int\frac{1}{\sqrt{2\pi}}e^{-(1-2t)z^2/2}\d z = (1-2t)^{-1/2}$. By independence the MGF of a sum of $k$ squares is $(1-2t)^{-k/2}$, which is the Gamma$(k/2,2)$ MGF; MGFs determine distributions (see <a href="#/t/mgf">MGFs</a>), giving (i) and (ii). (iii) follows by multiplying MGFs. Mean and variance: a Gamma$(\alpha,\beta)$ has mean $\alpha\beta = k$ and variance $\alpha\beta^2 = 2k$.</p>
</details>
<p>By the CLT, $(\chi^2_k-k)/\sqrt{2k}\dto\N(0,1)$, but slowly — the skewness is $\sqrt{8/k}$. The Wilson–Hilferty approximation $(V/k)^{1/3}\approx\N\big(1-\frac{2}{9k},\frac{2}{9k}\big)$ is far more accurate for moderate $k$, another instance of a symmetrising transformation (cf. the <a href="#/t/delta-method">delta method</a>).</p>`,
      },
      {
        title: 'The fundamental theorem: independence of the mean and variance',
        html: R`
<div class="thm" data-name="Mean and variance of a normal sample">
<p>If $X_1,\dots,X_n\iid\N(\mu,\sigma^2)$ with $n\ge2$, then (a) $\bar X$ and $S^2$ are independent; (b) $\bar X\sim\N(\mu,\sigma^2/n)$; (c) $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $\mathbf X = (X_1,\dots,X_n)^\top\sim\N_n(\mu\mathbf 1,\sigma^2I)$. Let $H$ be any $n\times n$ orthogonal matrix whose first row is $\mathbf 1^\top/\sqrt n$; one exists by completing $\mathbf 1/\sqrt n$ to an orthonormal basis (Gram–Schmidt; an explicit choice is the Helmert matrix). Set $\mathbf Y = H\mathbf X$. Because $H$ is orthogonal, $\mathbf Y\sim\N_n(\mu H\mathbf 1,\sigma^2HH^\top) = \N_n(\mu H\mathbf 1,\sigma^2I)$ — its coordinates are <em>independent</em> normals with common variance $\sigma^2$. Moreover:</p>
<ul>
<li>$Y_1 = \mathbf 1^\top\mathbf X/\sqrt n = \sqrt n\,\bar X$, and $\E Y_1 = \mu\sqrt n$.</li>
<li>For $i\ge2$, row $i$ of $H$ is orthogonal to $\mathbf 1$, so $\E Y_i = \mu\,(H\mathbf 1)_i = 0$.</li>
<li>Orthogonal maps preserve length: $\sum_{i=1}^nY_i^2 = \sum_{i=1}^nX_i^2$, so $\sum_{i\ge2}Y_i^2 = \sum X_i^2-n\bar X^2 = \sum(X_i-\bar X)^2 = (n-1)S^2$.</li>
</ul>
<p>Hence $\bar X$ is a function of $Y_1$ alone and $S^2$ a function of $(Y_2,\dots,Y_n)$ alone; independence of the coordinates gives (a). $Y_1\sim\N(\mu\sqrt n,\sigma^2)$ gives (b). And $(n-1)S^2/\sigma^2 = \sum_{i=2}^n(Y_i/\sigma)^2$ is a sum of $n-1$ independent squared standard normals, giving (c).</p>
</details>
<div class="note" data-name="The geometry">
<p>$\bar X\mathbf 1$ is the orthogonal projection of $\mathbf X$ onto the line spanned by $\mathbf 1$; the residual vector $\mathbf X-\bar X\mathbf 1$ lives in the $(n-1)$-dimensional orthogonal complement. A spherical normal distribution has independent components along any orthogonal directions, so the projection and the residual are independent, and the squared length of the residual is $\sigma^2$ times a chi-square with as many degrees of freedom as the dimension of its subspace. "Losing a degree of freedom by estimating $\mu$" is literally losing one dimension.</p>
</div>
<p>The independence is special to the normal. <strong>Geary (1936) and Lukacs (1942)</strong> proved the converse: if $X_1,\dots,X_n$ are IID with finite variance and $\bar X\indep S^2$, the distribution is normal. In general $\Cov(\bar X,S^2) = \mu_3/n$, where $\mu_3 = \E(X-\mu)^3$, so for skewed data the mean and variance are correlated — strongly so for exponential data, where the correlation tends to $1/\sqrt2\approx0.71$. The first lab shows this.</p>`,
        lab: 'sampling-distributions-xbar-s2',
      },
      {
        title: "Cochran's theorem: splitting sums of squares",
        html: R`
<p>The proof above split $\|\mathbf X\|^2$ into two orthogonal pieces. ANOVA and regression split it into several. The general statement is about quadratic forms in a standard normal vector.</p>
<div class="thm" data-kind="Lemma" data-name="Quadratic forms">
<p>Let $\mathbf Z\sim\N_n(\mathbf 0,I)$ and $A$ symmetric. Then $\mathbf Z^\top A\mathbf Z\sim\chi^2_r$ if and only if $A$ is idempotent ($A^2 = A$) of rank $r$, i.e. an orthogonal projection onto an $r$-dimensional subspace.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Diagonalise $A = Q\Lambda Q^\top$ with $Q$ orthogonal. Then $\mathbf W = Q^\top\mathbf Z\sim\N(\mathbf 0,I)$ and $\mathbf Z^\top A\mathbf Z = \sum_j\lambda_jW_j^2$, with MGF $\prod_j(1-2\lambda_jt)^{-1/2}$. This equals $(1-2t)^{-r/2}$ in a neighbourhood of $0$ iff exactly $r$ eigenvalues are $1$ and the rest are $0$ (compare the zeros/poles of the two analytic functions), i.e. iff $A$ is a rank-$r$ projection.</p>
</details>
<div class="thm" data-name="Cochran's theorem (1934)">
<p>Let $\mathbf Z\sim\N_n(\mathbf 0,I)$ and $I = A_1+\dots+A_k$ with each $A_j$ symmetric of rank $r_j$. If $r_1+\dots+r_k = n$, then the quadratic forms $Q_j = \mathbf Z^\top A_j\mathbf Z$ are independent and $Q_j\sim\chi^2_{r_j}$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The rank condition forces the decomposition to be orthogonal. Every $\mathbf x = \sum_jA_j\mathbf x$, so $\R^n = \text{col}(A_1)+\dots+\text{col}(A_k)$; since the dimensions add to exactly $n$, the sum is direct. For $\mathbf x\in\text{col}(A_j)$, $\mathbf x = \sum_iA_i\mathbf x$ and directness give $A_j\mathbf x = \mathbf x$ and $A_i\mathbf x = \mathbf 0$ for $i\ne j$. So each $A_j$ is the identity on its own column space and annihilates the others: $A_j^2 = A_j$ and $A_iA_j = 0$. Choose an orthonormal basis adapted to the (now mutually orthogonal) column spaces; in those coordinates $\mathbf W = Q^\top\mathbf Z\sim\N(\mathbf 0,I)$ and each $Q_j$ is the sum of squares of a disjoint block of $r_j$ coordinates of $\mathbf W$.</p>
</details>
<div class="ex" data-name="One-way ANOVA">
<p>With $g$ groups of sizes $n_1,\dots,n_g$ and total $N$, the identity
$$\sum_{j,i}(X_{ji}-\bar X)^2 = \sum_jn_j(\bar X_j-\bar X)^2 + \sum_{j,i}(X_{ji}-\bar X_j)^2$$
is a decomposition of a projection of rank $N-1$ into projections of ranks $g-1$ and $N-g$. Under the null hypothesis of equal means, Cochran gives independent $\sigma^2\chi^2_{g-1}$ and $\sigma^2\chi^2_{N-g}$ pieces, and their scaled ratio is $F_{g-1,N-g}$ — the <a href="#/t/anova">ANOVA</a> $F$-test. Regression $F$-tests are the same argument with hat matrices (<a href="#/t/ols-geometry">least squares as projection</a>).</p>
</div>`,
      },
      {
        title: "Student's t",
        html: R`
<div class="def" data-name="t distribution">
<p>If $Z\sim\N(0,1)$ and $V\sim\chi^2_k$ are independent, then $T = Z/\sqrt{V/k}$ has Student's $t$ distribution with $k$ degrees of freedom, with density
$$f_k(t) = \frac{\Gamma\big(\frac{k+1}{2}\big)}{\sqrt{k\pi}\;\Gamma\big(\frac k2\big)}\Big(1+\frac{t^2}{k}\Big)^{-(k+1)/2}.$$</p>
</div>
<details class="proof"><summary>Derivation sketch</summary>
<p>Condition on $V=v$: $T\mid V=v\sim\N(0,k/v)$. Integrate this normal density against the $\chi^2_k$ density of $V$; the integrand in $v$ is a gamma kernel $v^{(k+1)/2-1}e^{-v(1+t^2/k)/2}$, which integrates to $\Gamma\big(\frac{k+1}2\big)\big(2/(1+t^2/k)\big)^{(k+1)/2}$. Collecting constants gives $f_k$.</p>
</details>
<p>By the fundamental theorem, $\sqrt n(\bar X-\mu)/\sigma\sim\N(0,1)$ is independent of $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$, so the unknown $\sigma$ cancels in the ratio:
$$T = \frac{\sqrt n(\bar X-\mu)}{S}\sim t_{n-1}.$$
This is the pivot behind the one-sample $t$-interval $\bar X\pm t_{n-1,\alpha/2}\,S/\sqrt n$ (Student, 1908). Properties worth knowing:</p>
<ul>
<li><strong>Heavier tails than the normal:</strong> $f_k(t)\sim c\,|t|^{-(k+1)}$, so $\E|T|^m<\infty$ iff $m< k$. $t_1$ is the Cauchy distribution; $t_2$ has infinite variance; for $k>2$, $\Var T = k/(k-2)$.</li>
<li><strong>Normal limit:</strong> $V/k\pto1$ by the LLN, so $T\dto\N(0,1)$ by Slutsky. The approach is slow in the tails: $t_{5,0.975} = 2.571$, $t_{30,0.975} = 2.042$, versus $1.960$.</li>
<li><strong>A scale mixture of normals:</strong> $T\mid V\sim\N(0,k/V)$ — which is why the $t$ appears as a robust error model and as a posterior marginal in <a href="#/t/bayesian-inference">Bayesian</a> normal models with unknown variance.</li>
</ul>`,
        lab: 'sampling-distributions-t',
      },
      {
        title: "Snedecor's F",
        html: R`
<div class="def" data-name="F distribution">
<p>If $U\sim\chi^2_m$ and $V\sim\chi^2_k$ are independent, $F = \dfrac{U/m}{V/k}\sim F_{m,k}$.</p>
</div>
<p>Facts: $1/F\sim F_{k,m}$; if $T\sim t_k$ then $T^2\sim F_{1,k}$; $\E F = k/(k-2)$ for $k>2$; and $\dfrac{(m/k)F}{1+(m/k)F}\sim\text{Beta}(m/2,k/2)$, which is how $F$ tail probabilities are computed (through the incomplete beta function). As $k\to\infty$, $mF_{m,k}\dto\chi^2_m$.</p>
<div class="ex" data-name="Comparing two variances">
<p>With independent normal samples of sizes $n_1,n_2$, $\dfrac{S_1^2/\sigma_1^2}{S_2^2/\sigma_2^2}\sim F_{n_1-1,n_2-1}$, a pivot for $\sigma_1^2/\sigma_2^2$. It is exact under normality — and, as the next section explains, one of the least robust procedures in classical statistics.</p>
</div>`,
      },
      {
        title: 'Robustness: means survive, variances do not',
        html: R`
<p>What happens when the data are not normal? The answer differs sharply between procedures about means and procedures about variances.</p>
<div class="cols">
<div><h4>The $t$ statistic</h4><p>By the CLT and Slutsky, $\sqrt n(\bar X-\mu)/S\dto\N(0,1)$ for any finite-variance distribution, and $t_{n-1}\to\N(0,1)$ too. So the $t$-test is <em>asymptotically valid</em> for any population; non-normality costs an $O(n^{-1/2})$ skewness error (see the <a href="#/t/clt">Edgeworth expansion</a>), noticeable for strongly skewed data and small $n$.</p></div>
<div><h4>The $\chi^2$ and $F$ statistics for variances</h4><p>For general IID data, $\Var(S^2) = \sigma^4\Big(\dfrac{2}{n-1}+\dfrac{\kappa}{n}\Big)$ where $\kappa = \mu_4/\sigma^4-3$ is the excess kurtosis. The $\chi^2_{n-1}$ reference assumes $\kappa=0$. If $\kappa\ne0$ the error does <em>not</em> vanish as $n\to\infty$: $\sqrt n(S^2/\sigma^2-1)\dto\N(0,2+\kappa)$, not $\N(0,2)$.</p></div>
</div>
<div class="warn" data-name="Variance tests are wrong at every sample size">
<p>For exponential data, $\kappa=6$, so $(n-1)S^2/\sigma^2$ has asymptotically <em>four times</em> the variance the $\chi^2_{n-1}$ reference claims. A nominal $5\%$ test of $\sigma^2=\sigma_0^2$ rejects far more often than $5\%$, and collecting more data makes the size converge to the wrong value rather than to $5\%$. Box (1953) showed the same for Bartlett's and the $F$ test of equal variances. Use robust alternatives (Levene/Brown–Forsythe, bootstrap, permutation) for variances.</p>
</div>
<div class="warn" data-name="Independence of $\bar X$ and $S$ is not a harmless detail">
<p>Procedures that rely on it — e.g. using $S$ as if it were a fixed scale in sequential designs — inherit the correlation $\Corr(\bar X,S^2)\approx\gamma/\sqrt{\kappa+2}$ for non-normal data ($\gamma$ = skewness). The first lab shows the exponential case, where this is about $0.71$.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why does the sample variance have $n-1$ degrees of freedom? Answer geometrically.`,
        a: R`$(n-1)S^2 = \|\mathbf X-\bar X\mathbf 1\|^2$ is the squared length of the projection of $\mathbf X$ onto the $(n-1)$-dimensional orthogonal complement of $\mathbf 1$. Under a spherical normal (after centring), the squared length of the projection onto an $r$-dimensional subspace is $\sigma^2\chi^2_r$. Estimating $\mu$ uses up the one dimension along $\mathbf 1$.` },
      { q: R`Show that if $T\sim t_k$ then $T^2\sim F_{1,k}$, and deduce the relation between two-sided $t$ quantiles and $F$ quantiles.`,
        a: R`$T^2 = Z^2/(V/k) = (Z^2/1)/(V/k)$ with $Z^2\sim\chi^2_1$ independent of $V\sim\chi^2_k$, which is the definition of $F_{1,k}$. Hence $\P(|T|>c) = \P(T^2>c^2)$, so $t_{k,\alpha/2}^2 = F_{1,k,\alpha}$ (upper-tail quantiles).` },
      { q: R`For exponential data, is a $\chi^2$-based 95% confidence interval for $\sigma^2$ too wide or too narrow, and does increasing $n$ fix it?`,
        a: R`Too narrow: it assumes $\Var\big((n-1)S^2/\sigma^2\big)\approx2(n-1)$, but with excess kurtosis $\kappa=6$ the true variance is about $(2+6)(n-1)$, four times larger. Increasing $n$ does not fix it, because the ratio of true to assumed variance tends to $(2+\kappa)/2 = 4$; coverage converges to $\P(|Z|\le1.96/2)\approx0.67$, not $0.95$.` },
      { q: R`What is $\Cov(\bar X,S^2)$ for IID data with finite third moment, and what does it imply about the normal?`,
        a: R`$\Cov(\bar X,S^2) = \mu_3/n$ where $\mu_3 = \E(X-\mu)^3$. It is zero for any symmetric distribution, so zero covariance is not enough for independence; but full independence of $\bar X$ and $S^2$ holds only for the normal (Geary–Lukacs). For skewed data the two are correlated, which is why the $t$ statistic is skewed even when $\bar X$ is nearly normal.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'order-statistics',
    domain: 'foundations',
    order: 170,
    title: 'Order Statistics',
    subtitle: 'The distribution of sorted samples: extremes, quantiles, ranges and spacings',
    level: 2,
    sources: { cb: '§5.4', other: 'David & Nagaraja, Order Statistics (3rd ed., 2003); Arnold, Balakrishnan & Nagaraja, A First Course in Order Statistics (1992)' },
    prereqs: ['random-variables', 'transformations'],
    related: ['evt-intro', 'fisher-tippett', 'empirical-cdf', 'robust-statistics', 'rank-tests', 'delta-method'],
    tags: ['quantiles', 'median', 'extremes', 'beta distribution', 'spacings'],
    summary: R`Sorting a sample $X_1,\dots,X_n$ gives the order statistics $X_{(1)}\le\dots\le X_{(n)}$: minima, maxima, medians, ranges and sample quantiles are all functions of them. A counting argument gives everything: $X_{(k)}\le x$ exactly when at least $k$ observations fall below $x$, a binomial event. This yields the density of the $k$-th order statistic, the joint density of any pair, and the joint density of the whole sorted sample. For uniform data the order statistics are beta-distributed; for exponential data their spacings are independent. Central order statistics are asymptotically normal with variance $p(1-p)/(nf(\xi_p)^2)$; extreme ones are not, and lead to extreme value theory.`,
    survey: {
      task: 'Distributions of minima, maxima, medians, quantiles, ranges and spacings',
      assumptions: R`IID sample from a continuous $F$ (no ties); density $f$ for the density formulas`,
      output: R`$f_{(k)}(x)$, joint densities, $\E X_{(k)}$; asymptotic normality of sample quantiles`,
      strengths: 'Exact, distribution-free structure via $U_{(k)} = F(X_{(k)})$; basis of robust and rank methods',
      weaknesses: 'Quantile variance depends on the unknown density $f(\\xi_p)$; extremes need separate (non-normal) asymptotics',
      complexity: R`Sorting $O(n\log n)$; formulas closed-form given $F$ and $f$`,
    },
    sections: [
      {
        title: 'Definitions and why they matter',
        html: R`
<p>Let $X_1,\dots,X_n$ be IID with continuous CDF $F$ and density $f$. Sorting gives the <strong>order statistics</strong> $X_{(1)}< X_{(2)}<\dots< X_{(n)}$; with a continuous $F$, ties have probability zero. They are not independent (they are sorted!) and not identically distributed. Familiar statistics are functions of them:</p>
<ul>
<li>minimum $X_{(1)}$ and maximum $X_{(n)}$ — reliability of series and parallel systems, record values, <a href="#/t/evt-intro">extremes</a>;</li>
<li>sample median $X_{((n+1)/2)}$ ($n$ odd), sample quantiles, interquartile range — <a href="#/t/robust-statistics">robust</a> summaries;</li>
<li>range $X_{(n)}-X_{(1)}$, trimmed means, $L$-estimators $\sum c_iX_{(i)}$;</li>
<li>ranks, which are the positions of the original observations among the order statistics — the raw material of <a href="#/t/rank-tests">rank tests</a>.</li>
</ul>
<p>A powerful reduction: since $F$ is continuous and nondecreasing, $U_i = F(X_i)$ are IID uniform and $F(X_{(k)}) = U_{(k)}$. Conversely, $X_{(k)}\overset{d}{=}F^{-1}(U_{(k)})$. Many results need only be proved for uniform samples and then transported by the quantile function — which is also why the distribution of $F(X_{(k)})$ is the same for every continuous $F$, the fact behind distribution-free tolerance intervals.</p>`,
      },
      {
        title: 'Extremes: the minimum and maximum',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Distribution of the max and min">
<p>$\P(X_{(n)}\le x) = F(x)^n$ and $\P(X_{(1)}>x) = (1-F(x))^n$, so $f_{(n)}(x) = nF(x)^{n-1}f(x)$ and $f_{(1)}(x) = n(1-F(x))^{n-1}f(x)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The maximum is at most $x$ iff every observation is: by independence the probability is $\prod_i\P(X_i\le x) = F(x)^n$. The minimum exceeds $x$ iff every observation does. Differentiate for the densities.</p>
</details>
<div class="ex" data-name="Two classics">
<p><strong>Exponential minimum.</strong> If $X_i\iid\text{Exp}(\lambda)$, $\P(X_{(1)}>x) = e^{-n\lambda x}$, so $X_{(1)}\sim\text{Exp}(n\lambda)$: a series system of $n$ identical memoryless components fails $n$ times as fast.</p>
<p><strong>Uniform maximum.</strong> If $U_i\iid\text{Unif}(0,1)$, $U_{(n)}\sim\text{Beta}(n,1)$, with $\E U_{(n)} = n/(n+1)$. Rescaling the gap to the endpoint, $\P(n(1-U_{(n)})>t) = (1-t/n)^n\to e^{-t}$: the gap is of order $1/n$, not $1/\sqrt n$, and its limit is exponential, not normal. This is the simplest instance of the <a href="#/t/fisher-tippett">extremal types theorem</a>.</p>
</div>`,
      },
      {
        title: 'The k-th order statistic',
        html: R`
<div class="thm" data-name="Distribution of $X_{(k)}$">
<p>For $1\le k\le n$,
$$\P(X_{(k)}\le x) = \sum_{j=k}^n\binom nj F(x)^j\big(1-F(x)\big)^{n-j},\qquad f_{(k)}(x) = \frac{n!}{(k-1)!\,(n-k)!}\,F(x)^{k-1}\big(1-F(x)\big)^{n-k}f(x).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>CDF.</em> Let $N(x) = \#\{i:X_i\le x\}\sim\text{Binomial}(n,F(x))$. The $k$-th smallest value is at most $x$ iff at least $k$ observations are at most $x$: $\{X_{(k)}\le x\} = \{N(x)\ge k\}$. This holds for any $F$, continuous or not.</p>
<p><em>Density.</em> Heuristically, $X_{(k)}\in[x,x+\d x]$ requires $k-1$ observations below $x$, one in $[x,x+\d x]$, and $n-k$ above; the multinomial coefficient $\frac{n!}{(k-1)!\,1!\,(n-k)!}$ counts the assignments, giving the formula. Rigorously, differentiate the CDF: the derivative of $\sum_{j\ge k}\binom njF^j(1-F)^{n-j}$ telescopes. Indeed, with $u = F(x)$, $\frac{d}{du}\P(\text{Bin}(n,u)\ge k) = n\binom{n-1}{k-1}u^{k-1}(1-u)^{n-k}$ (the classical identity linking binomial tails and the incomplete beta function), and the chain rule supplies the factor $f(x)$.</p>
</details>
<p>For uniform data, $f_{(k)}(u) = \frac{n!}{(k-1)!(n-k)!}u^{k-1}(1-u)^{n-k}$ on $(0,1)$: $U_{(k)}\sim\text{Beta}(k,n-k+1)$, with
$$\E U_{(k)} = \frac{k}{n+1},\qquad\Var U_{(k)} = \frac{k(n-k+1)}{(n+1)^2(n+2)}.$$
So the $n$ order statistics of a uniform sample sit, on average, at the $n+1$-equally-spaced points $1/(n+1),\dots,n/(n+1)$ — the origin of the plotting positions $k/(n+1)$ used in QQ plots, and of the binomial identity $\P(\text{Bin}(n,u)\ge k) = I_u(k,n-k+1)$.</p>
<div class="ex" data-name="A distribution-free interval for the median">
<p>For any continuous $F$ with median $m$, $\P(X_{(j)}\le m< X_{(k)}) = \P(j\le N(m)< k) = \sum_{i=j}^{k-1}\binom ni2^{-n}$. For $n=20$, $(X_{(6)},X_{(15)})$ covers the median with probability $0.959$, whatever $F$ is. This is the inversion of the sign test.</p>
</div>`,
        lab: 'order-statistics-density',
      },
      {
        title: 'Joint distributions, ranges and spacings',
        html: R`
<div class="thm" data-name="Joint densities">
<p>(a) The joint density of all order statistics is $f_{(1),\dots,(n)}(x_1,\dots,x_n) = n!\prod_{i=1}^nf(x_i)$ on $x_1<\dots< x_n$, and zero elsewhere.</p>
<p>(b) For $i< j$, the joint density of $(X_{(i)},X_{(j)})$ at $u< v$ is
$$\frac{n!}{(i-1)!\,(j-i-1)!\,(n-j)!}\,F(u)^{i-1}\big[F(v)-F(u)\big]^{j-i-1}\big[1-F(v)\big]^{n-j}f(u)f(v).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(a) The map from $(X_1,\dots,X_n)$ to the sorted vector is $n!$-to-one: each sorted configuration $x_1<\dots< x_n$ arises from exactly $n!$ permutations, each with density $\prod f(x_i)$, and the map is a permutation of coordinates on each piece (Jacobian $\pm1$). Summing over the $n!$ preimages gives (a).</p>
<p>(b) Integrate (a) over the other coordinates, or count: $i-1$ observations below $u$, one at $u$, $j-i-1$ between, one at $v$, $n-j$ above, with the multinomial coefficient counting labelled assignments.</p>
</details>
<div class="ex" data-name="The uniform range">
<p>For a uniform sample, (b) with $i=1$, $j=n$ gives the joint density $n(n-1)(v-u)^{n-2}$ on $0< u< v<1$. Changing variables to $R = V-U$ and integrating out $U\in(0,1-r)$: $f_R(r) = n(n-1)r^{n-2}(1-r)$, i.e. $R\sim\text{Beta}(n-1,2)$ with $\E R = (n-1)/(n+1)$.</p>
</div>
<div class="thm" data-name="Rényi's representation for exponential spacings">
<p>If $X_i\iid\text{Exp}(1)$, the normalised spacings $D_k = (n-k+1)\big(X_{(k)}-X_{(k-1)}\big)$, $k=1,\dots,n$ (with $X_{(0)}=0$), are IID $\text{Exp}(1)$. Equivalently, $X_{(k)}\overset d= \sum_{j=1}^k\dfrac{E_j}{n-j+1}$ for IID $E_j\sim\text{Exp}(1)$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>By (a) the sorted sample has density $n!\,e^{-\sum x_i}$ on $0< x_1<\dots< x_n$. The linear map to the spacings $d_k = (n-k+1)(x_k-x_{k-1})$ has Jacobian determinant $n!$ (triangular, diagonal $n,n-1,\dots,1$) and satisfies $\sum_kx_k = \sum_kd_k$. So the spacings have density $e^{-\sum d_k}$ on $(0,\infty)^n$: IID unit exponentials. (Memorylessness says the same in words: after the $(k-1)$-th failure, $n-k+1$ fresh exponentials race.)</p>
</details>
<p>Consequences: $\E X_{(n)} = \sum_{j=1}^n1/j = H_n\approx\log n+0.5772$, so the maximum of exponential data grows logarithmically, and $X_{(n)}-\log n\dto$ Gumbel. For uniform data the analogous fact is that the $n+1$ spacings $(U_{(1)},U_{(2)}-U_{(1)},\dots,1-U_{(n)})$ are jointly Dirichlet$(1,\dots,1)$, i.e. $U_{(k)}\overset d=\Gamma_k/\Gamma_{n+1}$ where $\Gamma_k$ is a sum of $k$ IID unit exponentials.</p>`,
      },
      {
        title: 'Asymptotics of sample quantiles',
        html: R`
<div class="thm" data-name="Asymptotic normality of central order statistics">
<p>Let $0< p<1$, let $\xi_p = F^{-1}(p)$, and suppose $f$ is continuous and positive at $\xi_p$. If $k_n/n = p+o(n^{-1/2})$, then
$$\sqrt n\big(X_{(k_n)}-\xi_p\big)\dto\N\Big(0,\;\frac{p(1-p)}{f(\xi_p)^2}\Big).$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Use the counting identity. For fixed $t$, with $x_n = \xi_p+t/\sqrt n$,
$$\P\big(\sqrt n(X_{(k_n)}-\xi_p)\le t\big) = \P\big(N(x_n)\ge k_n\big),\qquad N(x_n)\sim\text{Bin}\big(n,F(x_n)\big).$$
Since $F(x_n) = p+f(\xi_p)t/\sqrt n+o(n^{-1/2})$, standardising the binomial gives
$$\P\Big(\frac{N(x_n)-nF(x_n)}{\sqrt{nF(x_n)(1-F(x_n))}}\ge\frac{k_n-nF(x_n)}{\sqrt{nF(x_n)(1-F(x_n))}}\Big),$$
and the right-hand threshold tends to $-f(\xi_p)t/\sqrt{p(1-p)}$. By the CLT for the binomial (a triangular array satisfying Lindeberg), the probability tends to $\Phi\big(f(\xi_p)t/\sqrt{p(1-p)}\big)$, the CDF of $\N\big(0,p(1-p)/f(\xi_p)^2\big)$ at $t$.</p>
</details>
<p>The formula is a <a href="#/t/delta-method">delta method</a> in disguise: the empirical CDF at $\xi_p$ has variance $p(1-p)/n$, and inverting $F$ near $\xi_p$ divides by the slope $f(\xi_p)$. Where the density is low, a small error in probability becomes a large error in $x$.</p>
<div class="ex" data-name="Median vs. mean">
<p>For the median ($p=1/2$), $\Var_{\text{asy}} = 1/(4f(m)^2n)$. Asymptotic relative efficiency of the median to the mean, $\sigma^2\cdot4f(m)^2$:</p>
<table>
<thead><tr><th>Population</th><th>$n\Var(\bar X)$</th><th>$n\Var_{\text{asy}}(\hat m)$</th><th>ARE (median vs. mean)</th></tr></thead>
<tbody>
<tr><td>Normal$(0,1)$</td><td>$1$</td><td>$\pi/2\approx1.571$</td><td>$2/\pi\approx0.637$</td></tr>
<tr><td>Laplace (scale 1)</td><td>$2$</td><td>$1$</td><td>$2$</td></tr>
<tr><td>$t_3$</td><td>$3$</td><td>$\approx1.850$</td><td>$\approx1.62$</td></tr>
<tr><td>Uniform$(0,1)$</td><td>$1/12$</td><td>$1/4$</td><td>$1/3$</td></tr>
<tr><td>Cauchy</td><td>$\infty$</td><td>$\pi^2/4\approx2.467$</td><td>$\infty$</td></tr>
</tbody>
</table>
<p>Under normality the median wastes about 36% of the data; under heavy tails it wins, decisively so for the Cauchy, where the mean does not converge at all (<a href="#/t/lln">LLN</a>). This trade-off motivates trimmed means and M-estimators.</p>
</div>`,
        lab: 'order-statistics-efficiency',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Extremes are not asymptotically normal">
<p>The quantile theorem needs $p$ fixed in $(0,1)$. For $k$ fixed (or $k/n\to0$) the normalisation and the limit change: $n(1-U_{(n)})\to\text{Exp}(1)$, and for general $F$ the maximum, suitably normalised, converges to a GEV law (<a href="#/t/fisher-tippett">extremal types theorem</a>). Plugging an extreme order statistic into a normal-theory interval gives nonsense; so does the naive <a href="#/t/bootstrap">bootstrap</a> of the maximum.</p>
</div>
<div class="warn" data-name="Discrete data and ties">
<p>With a discrete $F$, ties occur with positive probability, the density formulas fail, and $F(X_{(k)})$ is no longer a uniform order statistic. The CDF formula via $N(x)\ge k$ remains valid. Rank tests need tie corrections for the same reason.</p>
</div>
<div class="warn" data-name="Which sample quantile?">
<p>"The $p$-th sample quantile" has at least nine standard definitions (Hyndman & Fan, 1996), interpolating between order statistics differently. They agree to first order — all satisfy the asymptotic theorem — but differ noticeably for small $n$ and in the tails. State which one you use.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Derive the density of the sample median of $n=2m+1$ IID Uniform$(0,1)$ observations and give its mean and variance.`,
        a: R`It is $U_{(m+1)}\sim\text{Beta}(m+1,m+1)$ with density $\frac{(2m+1)!}{m!\,m!}u^m(1-u)^m$; mean $1/2$; variance $\frac{(m+1)^2}{(2m+2)^2(2m+3)} = \frac{1}{4(2m+3)} = \frac1{4(n+2)}$, which matches the asymptotic $p(1-p)/(nf^2) = 1/(4n)$.` },
      { q: R`For IID Exp(1) data, find $\E X_{(n)}$ and explain why it grows like $\log n$.`,
        a: R`By Rényi's representation, $X_{(n)} = \sum_{j=1}^nE_j/(n-j+1)$, so $\E X_{(n)} = \sum_{i=1}^n1/i = H_n\approx\log n+\gamma$. Each successive gap between order statistics is the waiting time for the first of the remaining exponentials, which shrinks like $1/(\text{number remaining})$; summing gives the harmonic series.` },
      { q: R`Why is $(X_{(1)},X_{(n)})$ a distribution-free confidence interval for the median with coverage $1-2^{1-n}$?`,
        a: R`$m$ lies outside $[X_{(1)},X_{(n)}]$ only if all observations fall on the same side of $m$, which happens with probability $2\cdot2^{-n}$ for any continuous $F$ (each observation is below $m$ with probability $1/2$, independently).` },
      { q: R`For which populations is the sample median asymptotically more efficient than the sample mean? Give the criterion.`,
        a: R`The median wins when $4f(m)^2\sigma^2>1$, i.e. when the density at the median is high relative to the spread measured by the variance — typical of heavy-tailed or sharply peaked distributions (Laplace: ARE $2$; $t_3$: $\approx1.62$; Cauchy: $\infty$). It loses for light-tailed or flat distributions (normal: $2/\pi$; uniform: $1/3$).` },
    ],
  });
})();
