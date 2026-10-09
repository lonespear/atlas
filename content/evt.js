/* Extreme Value Theory. Anchors: Coles (2001); Embrechts, Klüppelberg & Mikosch (1997); de Haan & Ferreira (2006);
 * Resnick (1987); Leadbetter, Lindgren & Rootzén (1983). Labs live in js/labs/evt.js. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'evt-intro',
    domain: 'evt',
    order: 10,
    title: 'Why Extremes Need Their Own Theory',
    subtitle: 'The maximum degenerates, the CLT is blind to tails, and the right normalisation changes everything',
    level: 2,
    sources: { cb: '§5.4 (order statistics)', other: 'Coles (2001) Ch. 1–3; Embrechts, Klüppelberg & Mikosch (1997) Ch. 3; Leadbetter, Lindgren & Rootzén (1983) Ch. 1' },
    prereqs: ['order-statistics', 'convergence'],
    related: ['clt', 'fisher-tippett', 'poisson-process'],
    tags: ['maxima', 'tails', 'extrapolation', 'Poisson approximation'],
    summary: R`Engineers, insurers and risk managers routinely need probabilities of events larger than anything in the record: the 1-in-10,000-year storm surge, the loss that bankrupts a reinsurer. Two naive routes fail. The empirical distribution puts zero mass beyond the sample maximum, and a model fitted to the bulk of the data (a normal, say) dictates its tail from its centre. The exact law of the maximum, $F^n$, collapses onto the right endpoint of $F$ as $n$ grows, so something must be normalised — and the central limit theorem, which governs sums, says nothing about how. Extreme value theory supplies the missing limit theorems: under a single affine normalisation the maximum has only one possible family of limits, and that universality is what licenses extrapolation.`,
    survey: {
      task: 'Understand why maxima and tail probabilities require dedicated asymptotics',
      assumptions: R`IID $X_i \sim F$ (relaxed later); interest in $M_n = \max_i X_i$ or $\P(X > x)$ for large $x$`,
      output: R`Exact law $F^n$; the Poisson lemma $F^n(u_n)\to e^{-\tau} \iff n\bar F(u_n)\to\tau$`,
      strengths: 'Explains degeneracy, motivates normalisation, connects maxima to exceedance counts',
      weaknesses: 'Exact law needs the unknown tail of F; nothing here yet tells us the limiting shape',
      complexity: '—',
    },
    sections: [
      {
        title: 'The questions extremes ask',
        html: R`
<p>After the North Sea flood of 1953 killed more than 1,800 people in the Netherlands, the Delta Committee asked for sea defences that would fail with probability about $10^{-4}$ per year. The available record of storm surges was roughly a century long. The question was therefore not "what is typical?" but "how high is the level that the sea exceeds once in ten thousand years?" — an extrapolation roughly two orders of magnitude beyond the data.</p>
<p>The same structure appears everywhere risk is concentrated in the tail:</p>
<ul>
<li><strong>Hydrology and climate:</strong> design floods, extreme rainfall, heat waves, wind loads on structures.</li>
<li><strong>Finance and insurance:</strong> value-at-risk and expected shortfall at the 99.5% level, catastrophe reinsurance layers, the largest single claim in a portfolio.</li>
<li><strong>Engineering and materials:</strong> a chain breaks at its weakest link, so strength is a <em>minimum</em>; corrosion pits, fatigue cracks and inclusions are extremes of spatial fields.</li>
<li><strong>Machine learning and monitoring:</strong> anomaly scores, worst-case latency, out-of-distribution detection thresholds.</li>
</ul>
<p>Minima need no separate theory: $\min_i X_i = -\max_i(-X_i)$, so everything below is stated for maxima.</p>
<div class="note">
<p>The defining feature of these problems is <strong>extrapolation</strong>. The empirical CDF $\hat F_n$ assigns probability zero to $(X_{(n)}, \infty)$, so any purely nonparametric answer to "how likely is a value larger than all we have seen?" is "impossible", which is certainly wrong. Fitting a convenient parametric family to all the data is no better: its tail is then a by-product of how well it fits the centre, where almost all the observations live. We need a model for the tail that is justified by something other than convenience. Extreme value theory justifies it by <em>asymptotics</em>, exactly as the central limit theorem justifies the normal model for averages.</p>
</div>`,
      },
      {
        title: 'The exact law of the maximum, and why it degenerates',
        html: R`
<p>Let $X_1,\dots,X_n \iid F$ and $M_n = \max(X_1,\dots,X_n) = X_{(n)}$, the largest <a href="#/t/order-statistics">order statistic</a>. By independence,
$$\P(M_n \le x) = \P(X_1\le x,\dots,X_n\le x) = F(x)^n,$$
and if $F$ has density $f$, then $M_n$ has density $nF(x)^{n-1}f(x)$. This is exact, but it is not useful as it stands, for two reasons. First, it depends on $F$ in precisely the region where $F$ is unknown: $F(x)^n$ is not small only where $F(x)$ is within about $1/n$ of $1$, i.e. in the far tail. Small relative errors in $1-F$ become large errors in $F^n$. Second, its behaviour as $n\to\infty$ is trivial.</p>
<div class="def" data-name="Right endpoint">
<p>The right endpoint of $F$ is $x_F = \sup\{x : F(x) < 1\} \in (-\infty, \infty]$. Write $\bar F = 1 - F$ for the survival function (tail).</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Degeneracy of the raw maximum">
<p>$M_n \asto x_F$. In particular $F^n(x) \to 0$ for every $x < x_F$ and $F^n(x) = 1$ for $x \ge x_F$, so the law of $M_n$ converges to a point mass at $x_F$ (or escapes to $+\infty$ if $x_F = \infty$).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Fix $x < x_F$. Then $F(x) < 1$, so $\P(M_n \le x) = F(x)^n \to 0$. The sequence $M_n$ is nondecreasing in $n$, so $M_\infty = \lim_n M_n$ exists (possibly $+\infty$) and $\{M_\infty \le x\} \subseteq \{M_n \le x\}$ for every $n$; hence $\P(M_\infty \le x) = 0$. Taking $x$ along a countable sequence increasing to $x_F$ gives $M_\infty \ge x_F$ almost surely. Conversely $X_i \le x_F$ a.s. for every $i$ when $x_F < \infty$, so $M_n \le x_F$. Thus $M_n \to x_F$ almost surely.</p>
</details>
<p>The same happens to sums: $S_n/n \to \mu$ by the law of large numbers, a degenerate limit. The CLT recovers a nondegenerate law by centring and scaling, $(S_n - n\mu)/(\sigma\sqrt n)$. Extreme value theory does the same for $M_n$: find constants $a_n > 0$, $b_n$ such that
$$\P\!\left(\frac{M_n - b_n}{a_n} \le x\right) = F^n(a_n x + b_n) \;\longrightarrow\; G(x)$$
for some nondegenerate distribution function $G$. Whether such constants exist, how they must grow, and what $G$ can be, are the subject of the next two topics. The lab below shows the degeneracy and the rescue: the curves $F^n$ march off to the right (and, depending on $F$, spread out, sharpen or keep their width), while after normalisation they collapse onto a single limit.</p>`,
        lab: 'evt-intro-degenerate',
      },
      {
        title: 'The Poisson lemma: maxima and exceedance counts are the same problem',
        html: R`
<p>Before asking what $G$ can be, there is a simple and remarkably useful equivalence. Choose levels $u_n$ that grow with $n$, and let $N_n = \sum_{i=1}^n \ind(X_i > u_n)$ count exceedances. Then
$$\{M_n \le u_n\} = \{N_n = 0\}, \qquad N_n \sim \text{Binomial}\big(n, \bar F(u_n)\big).$$
If $u_n$ is chosen so that the expected number of exceedances $n\bar F(u_n)$ settles down to a constant $\tau$, the law of rare events says $N_n$ is approximately Poisson($\tau$), so $\P(M_n \le u_n) \approx e^{-\tau}$. The converse also holds.</p>
<div class="thm" data-kind="Lemma" data-name="Poisson approximation (Leadbetter, Lindgren & Rootzén, 1983, Thm 1.5.1)">
<p>Let $0 \le \tau \le \infty$ and let $(u_n)$ be any sequence of reals. Then
$$n\,\bar F(u_n) \to \tau \quad\Longleftrightarrow\quad \P(M_n \le u_n) \to e^{-\tau}.$$
Moreover $N_n \dto \text{Poisson}(\tau)$ when $\tau < \infty$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>($\Rightarrow$), $\tau<\infty$: then $p_n := \bar F(u_n) \to 0$, and $\log \P(M_n\le u_n) = n\log(1-p_n) = -np_n\,(1 + O(p_n)) \to -\tau$. If $\tau = \infty$, then $\log F^n(u_n) = n\log(1-p_n) \le -np_n \to -\infty$.</p>
<p>($\Leftarrow$), $\tau<\infty$: $F(u_n)^n \to e^{-\tau} > 0$ forces $F(u_n) \to 1$ (along any subsequence with $F(u_n) \le 1-\delta$ we would have $F(u_n)^n \le (1-\delta)^n \to 0$). Then $-\log F(u_n) = p_n(1+o(1))$, and $n\log F(u_n) \to -\tau$ gives $np_n \to \tau$. If $\tau = \infty$ and $np_n \not\to\infty$, a subsequence has $np_n \to \tau' < \infty$, and by the first part $F^n(u_n) \to e^{-\tau'} > 0$ along it, a contradiction.</p>
<p>The Poisson limit of $N_n$ is the classical binomial-to-Poisson limit, since $N_n \sim \text{Bin}(n,p_n)$ with $np_n \to \tau$.</p>
</details>
<p>This turns the search for a limit law into a statement about the <em>tail</em> alone. To get $F^n(a_n x + b_n) \to G(x)$ we need exactly
$$n\,\bar F(a_n x + b_n) \;\to\; -\log G(x) \qquad\text{for each } x \text{ with } G(x) > 0.$$
So: a limit for the maximum exists if and only if the tail $\bar F$, viewed through the affine "lens" $x \mapsto a_n x + b_n$ and multiplied by $n$, converges. Only the behaviour of $\bar F$ near $x_F$ matters.</p>
<div class="ex" data-name="Three parents, three limits">
<p><strong>Exponential</strong>, $\bar F(x) = e^{-x}$. Take $a_n = 1$, $b_n = \log n$: $n\bar F(x + \log n) = e^{-x}$ exactly, so $\P(M_n - \log n \le x) \to \exp(-e^{-x})$ for all real $x$. This is the <strong>Gumbel</strong> law $\Lambda$. The maximum grows like $\log n$ but its fluctuations stay $O(1)$.</p>
<p><strong>Pareto</strong>, $\bar F(x) = x^{-\alpha}$ for $x\ge1$. Take $a_n = n^{1/\alpha}$, $b_n = 0$: $n\bar F(n^{1/\alpha}x) = x^{-\alpha}$ for $x > 0$ (once $n^{1/\alpha}x\ge 1$), so $\P(n^{-1/\alpha}M_n \le x) \to \exp(-x^{-\alpha})$, the <strong>Fréchet</strong> law $\Phi_\alpha$. Location and spread grow together, like $n^{1/\alpha}$.</p>
<p><strong>Uniform(0,1)</strong>, $\bar F(x) = 1-x$ on $[0,1]$. Take $a_n = 1/n$, $b_n = 1$: for $x<0$, $n\bar F(1 + x/n) = -x$, so $\P(n(M_n - 1) \le x) \to e^{x}$, the (reverse) <strong>Weibull</strong> law $\Psi_1$ supported on $(-\infty,0]$. The maximum approaches the endpoint at rate $1/n$.</p>
</div>
<p>Three parents, three qualitatively different limits. The extremal types theorem (next topic) says there are no others.</p>`,
      },
      {
        title: 'Why the central limit theorem is the wrong tool',
        html: R`
<p>It is tempting to think the normal distribution, justified by the <a href="#/t/clt">CLT</a>, is a safe default model for anything. For tail questions it is not, and the reason is structural rather than a matter of degree.</p>
<p>The CLT is universal because it <em>forgets</em>: for any $F$ with $\E X^2 < \infty$, $(S_n - n\mu)/(\sigma\sqrt n) \dto \N(0,1)$, and the only features of $F$ that survive are $\mu$ and $\sigma^2$. Those two numbers are determined by the bulk of $F$. Two distributions with the same mean and variance — a standard normal and a Student $t_4$ rescaled to variance one, say — produce the same limit for sums but behave completely differently in the tail:</p>
<table>
<thead><tr><th></th><th>Normal $\N(0,1)$</th><th>$t_4/\sqrt2$ (variance 1)</th></tr></thead>
<tbody>
<tr><td>Tail $\bar F(x)$</td><td>$\sim \phi(x)/x$, faster than any power</td><td>$\sim c\,x^{-4}$, a power law</td></tr>
<tr><td>Growth of $M_n$</td><td>$\sim\sqrt{2\log n}$</td><td>$\asymp n^{1/4}$</td></tr>
<tr><td>Limit of normalised $M_n$</td><td>Gumbel ($\xi = 0$)</td><td>Fréchet with $\alpha=4$ ($\xi = 1/4$)</td></tr>
<tr><td>$\P(X > 6)$</td><td>$\approx 10^{-9}$</td><td>$\approx 5\times 10^{-4}$</td></tr>
</tbody>
</table>
<p>A normal model fitted to the second sample would match its mean and variance perfectly and still understate the probability of a six-sigma event by more than five orders of magnitude. Nothing in the data's first two moments could have warned us; the information is in the tail, which a moment fit ignores.</p>
<div class="note" data-name="Sums versus maxima">
<p>For a sum, every observation contributes a little and the bulk dominates. For a maximum, a single observation determines the answer and the bulk is irrelevant. These are opposite regimes, and they have different universal limits: the normal (or more generally stable) laws for sums, and the extreme value laws for maxima. When the tail is heavy enough the two regimes meet: for subexponential $F$ (Pareto, lognormal, $t$) one has $\P(S_n > x) \sim \P(M_n > x) \sim n\bar F(x)$ as $x\to\infty$ — a large sum happens because of one large jump, so even questions about sums become questions about maxima.</p>
</div>
<p>The lab makes this concrete: draw data from a heavy-tailed parent, fit a normal to it by matching moments, and compare the normal's tail with the truth on a logarithmic scale. Within the range of the data the discrepancy is modest; beyond it the two diverge without bound.</p>`,
        lab: 'evt-intro-tails',
      },
      {
        title: 'Not every distribution has an extreme-value limit',
        html: R`
<p>The Poisson lemma also explains when the program fails. To have $F^n(a_nx+b_n) \to G(x)$ with $G$ continuous (we will see all extreme value limits are continuous), the tail must be "tunable": we need levels $u_n$ with $n\bar F(u_n) \to \tau$ for every $\tau \in (0,\infty)$. If $\bar F$ jumps by a non-vanishing factor at its atoms, this is impossible.</p>
<div class="thm" data-kind="Theorem" data-name="Leadbetter, Lindgren & Rootzén (1983), Thm 1.7.13">
<p>Let $0 < \tau < \infty$. There exist $u_n$ with $n\bar F(u_n) \to \tau$ if and only if
$$\frac{\bar F(x)}{\bar F(x^-)} \to 1 \qquad \text{as } x \uparrow x_F,$$
where $\bar F(x^-) = \P(X \ge x)$. In particular, if this fails then no affine normalisation produces a nondegenerate limit for $M_n$.</p>
</div>
<div class="ex" data-name="Geometric and Poisson maxima">
<p>For a geometric variable on $\{1,2,\dots\}$, $\bar F(k) = q^k$, so $\bar F(k)/\bar F(k^-) = q^k/q^{k-1} = q$ for all $k$: the tail drops by the fixed factor $q$ at every atom, $n\bar F(u_n)$ can only take values in a geometric lattice, and it cannot converge to an arbitrary $\tau$. For the Poisson, $\bar F(k)/\bar F(k^-) = \P(X > k)/\P(X\ge k) \to 0$, even worse. The maximum of IID geometric variables oscillates: $M_n - \log_{1/q} n$ is tight but does not converge in law.</p>
</div>
<p>In practice, discretised continuous data (rainfall to the nearest millimetre, say) are fine when the discretisation is fine relative to the scale of the extremes. Genuinely discrete counts with light tails need other tools.</p>`,
      },
      {
        title: 'A roadmap: four ways to use the theory',
        html: R`
<p>The Poisson lemma already hints that there are two equivalent ways of looking at extremes — through maxima, or through exceedances of a high level — and the rest of this domain develops both, plus their common generalisation and the extensions needed for real data.</p>
<table>
<thead><tr><th>Approach</th><th>Data used</th><th>Limit model</th><th>Topic</th></tr></thead>
<tbody>
<tr><td>Block maxima</td><td>One maximum per block (e.g. per year)</td><td>GEV$(\mu,\sigma,\xi)$</td><td><a href="#/t/block-maxima">Block maxima</a>, <a href="#/t/return-levels">return levels</a></td></tr>
<tr><td>Peaks over threshold</td><td>All excesses over a high $u$</td><td>Generalised Pareto$(\sigma_u,\xi)$</td><td><a href="#/t/peaks-over-threshold">POT</a>, <a href="#/t/threshold-selection">thresholds</a></td></tr>
<tr><td>Point process</td><td>Times and sizes of exceedances</td><td>Poisson process, intensity from GEV</td><td><a href="#/t/point-process-evt">Point-process view</a></td></tr>
<tr><td>Tail index</td><td>Top $k$ order statistics</td><td>Regularly varying tail $x^{-1/\xi}L(x)$</td><td><a href="#/t/tail-index">Hill and Pickands</a></td></tr>
</tbody>
</table>
<p>All rest on the same shape parameter $\xi$, which measures tail heaviness: $\xi>0$ for power-law tails, $\xi=0$ for exponential-type tails, $\xi<0$ for a finite upper endpoint. The independence assumption is relaxed in <a href="#/t/extremal-index">dependent extremes</a> (where clustering enters through the extremal index $\theta$), and the univariate theory is extended in <a href="#/t/multivariate-extremes">multivariate extremes</a>.</p>
<div class="warn" data-name="What asymptotics can and cannot buy">
<p>Extreme value models are justified by limit theorems as $n\to\infty$ (block size, or threshold level). They are therefore approximations whose accuracy at the block sizes and thresholds you actually use is not guaranteed, and the convergence can be slow (logarithmic for normal parents). The theory replaces an arbitrary tail assumption with a principled one; it does not abolish the uncertainty of extrapolation, which shows up as wide confidence intervals for long return periods. Report them.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that for $X_i \iid \text{Exp}(1)$, $\E M_n = \sum_{k=1}^n 1/k$, and reconcile this with the Gumbel limit of $M_n - \log n$.`,
        a: R`By the Rényi representation, the spacings of exponential order statistics are independent: $M_n \overset{d}{=} \sum_{k=1}^n E_k/k$ with $E_k \iid \text{Exp}(1)$, so $\E M_n = H_n = \log n + \gamma + o(1)$ ($\gamma \approx 0.5772$, Euler's constant). The Gumbel law has mean $\gamma$, consistent with $M_n - \log n \dto \Lambda$ (and uniform integrability).` },
      { q: R`Why is $\P(M_n \le x) = F(x)^n$, though exact, a poor basis for estimating the distribution of an annual maximum of daily data?`,
        a: R`With $n = 365$, $F^n(x)$ is only away from 0 where $\bar F(x) \lesssim$ a few times $1/365$, i.e. deep in the tail of the daily distribution. A model for $F$ fitted to all days is dominated by typical days, and a small relative error in $\bar F$ there is raised to the 365th power. In addition daily data are rarely independent or identically distributed across seasons. Modelling the maxima directly with their asymptotic family avoids needing $F$ at all.` },
      { q: R`Using the Poisson lemma, find $a_n, b_n$ such that $(M_n - b_n)/a_n$ has a nondegenerate limit when $\bar F(x) = 2/x^2$ for $x\ge\sqrt2$.`,
        a: R`We need $n\bar F(a_n x) \to x^{-2}$: $n\cdot 2/(a_n x)^2 = x^{-2}$ when $a_n = \sqrt{2n}$. So $M_n/\sqrt{2n} \dto \Phi_2$, i.e. $\P(M_n \le \sqrt{2n}\,x) \to \exp(-x^{-2})$, with $b_n = 0$.` },
      { q: R`Two models have identical mean and variance. Which features of the data determine whether their predictions of a 1-in-1000-year level agree?`,
        a: R`Only the behaviour of $\bar F$ in the far tail, summarised asymptotically by the shape $\xi$ (and the scale of the tail). Mean and variance are bulk features that the CLT retains; they do not constrain $\bar F$ near $x_F$. Two distributions can share all moments that exist and still have different $\xi$ (for instance, normal and $t_5$ scaled to the same variance).` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'fisher-tippett',
    domain: 'evt',
    order: 20,
    title: 'The Extremal Types Theorem',
    subtitle: 'Max-stability forces the limit of normalised maxima into one three-parameter family',
    level: 3,
    sources: { other: 'Fisher & Tippett (1928); Gnedenko (1943); de Haan & Ferreira (2006) §1.1; Resnick (1987) Ch. 0–1; Embrechts et al. (1997) §3.2' },
    prereqs: ['evt-intro'],
    related: ['gev-distribution', 'convergence', 'clt'],
    tags: ['max-stability', 'convergence of types', 'Gumbel', 'Fréchet', 'Weibull'],
    summary: R`If the normalised maximum $(M_n - b_n)/a_n$ has a nondegenerate limit $G$, then $G$ must be <em>max-stable</em>: the maximum of $k$ IID copies of $G$ is again $G$ up to an affine change of variable. That single functional equation has very few solutions. Solving it shows that $G$ must be, up to location and scale, $\exp\{-(1+\xi x)^{-1/\xi}\}$ for some real $\xi$ — the Gumbel ($\xi=0$), Fréchet ($\xi>0$) and reversed Weibull ($\xi<0$) laws in one formula. This is the Fisher–Tippett–Gnedenko theorem, the extreme-value analogue of the central limit theorem.`,
    survey: {
      task: 'Characterise all possible limits of affinely normalised maxima',
      assumptions: R`IID; $F^n(a_n x + b_n) \to G$ nondegenerate for some $a_n>0$, $b_n$`,
      output: R`$G(x) = \exp\{-(1+\xi\frac{x-\mu}{\sigma})^{-1/\xi}\}$: Gumbel, Fréchet or Weibull type`,
      strengths: 'Universality: one parametric family justified by asymptotics alone',
      weaknesses: 'Says nothing about existence of the limit or the rate of convergence (normal: O(1/log n))',
      complexity: '—',
    },
    sections: [
      {
        title: 'Posing the question precisely',
        html: R`
<p>From <a href="#/t/evt-intro">the previous topic</a>: the raw maximum $M_n$ degenerates, but suitable affine normalisations can produce nondegenerate limits, and three examples produced three different-looking limits. We now ask what <em>all</em> possible limits are.</p>
<div class="def" data-name="Max-domain of attraction; types">
<p>$F$ is in the <em>max-domain of attraction</em> of a nondegenerate distribution $G$, written $F \in \mathcal D(G)$, if there exist $a_n > 0$ and $b_n \in \R$ with
$$F^n(a_n x + b_n) \to G(x) \quad\text{at every continuity point of } G.$$
Two distribution functions $G_1, G_2$ are of the same <em>type</em> if $G_2(x) = G_1(ax + b)$ for some $a > 0$, $b\in\R$.</p>
</div>
<p>Since $a_n, b_n$ are ours to choose, $G$ can only be determined up to type: if $(M_n - b_n)/a_n \dto G$ then $(M_n - b_n')/a_n' \dto G(a\,\cdot + b)$ whenever $a_n'/a_n \to 1/a$ and $(b_n' - b_n)/a_n \to -b/a$. The next result says this is the <em>only</em> freedom.</p>
<div class="thm" data-kind="Theorem" data-name="Convergence of types (Khintchine)">
<p>Let $U_n$ be random variables, $a_n, \alpha_n > 0$, and suppose $(U_n - b_n)/a_n \dto G$ and $(U_n - \beta_n)/\alpha_n \dto G^*$ with $G, G^*$ nondegenerate. Then there are $A > 0$, $B\in\R$ with
$$\frac{\alpha_n}{a_n} \to A,\qquad \frac{\beta_n - b_n}{a_n} \to B,\qquad G^*(x) = G(Ax + B).$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Weak convergence of distribution functions implies convergence of their quantile functions at continuity points of the limit quantile. Write $V_n = (U_n - b_n)/a_n$; then $(U_n - \beta_n)/\alpha_n = (a_n/\alpha_n) V_n + (b_n - \beta_n)/\alpha_n$. Pick two continuity points $p_1 < p_2$ of both limiting quantile functions with $G^{\leftarrow}(p_1) < G^{\leftarrow}(p_2)$ (possible because $G$ is nondegenerate). The $p_i$-quantiles of the second sequence are an affine image of those of the first: $q^*_n(p_i) = (a_n/\alpha_n) q_n(p_i) + (b_n-\beta_n)/\alpha_n$. Two equations in two unknowns: since $q_n(p_i) \to G^{\leftarrow}(p_i)$ and $q^*_n(p_i) \to G^{*\leftarrow}(p_i)$, the coefficients $a_n/\alpha_n$ and $(b_n-\beta_n)/\alpha_n$ converge, and the limit of the slope is positive because $G^*$ is nondegenerate. The relation $G^*(x) = G(Ax+B)$ follows by passing to the limit.</p>
</details>`,
      },
      {
        title: 'Max-stability',
        html: R`
<div class="def" data-name="Max-stable distribution">
<p>A nondegenerate distribution $G$ is <em>max-stable</em> if for every $k \ge 1$ there are $\alpha_k > 0$, $\beta_k$ with
$$G^k(\alpha_k x + \beta_k) = G(x)\qquad\text{for all } x.$$
Equivalently: if $Y_1,\dots,Y_k \iid G$ then $\max_i Y_i \overset{d}{=} \alpha_k Y_1 + \beta_k$.</p>
</div>
<p>This is the exact analogue of stability for sums (a sum of IID stable variables is an affine copy of one of them). It is also exactly what one should expect of a limit: a maximum of $nk$ variables can be computed as a maximum of $k$ maxima of $n$ variables, so if maxima of $n$ variables are approximately $G$, so are maxima of $nk$, and the two descriptions must agree up to normalisation.</p>
<div class="thm" data-kind="Theorem" data-name="Limits are exactly the max-stable laws">
<p>A nondegenerate $G$ is the limit of $F^n(a_nx+b_n)$ for some $F$ and constants $a_n>0$, $b_n$ if and only if $G$ is max-stable. Moreover, in that case for every real $t > 0$ there exist $A(t)>0$ and $B(t)$ with
$$G^t\big(A(t)\,x + B(t)\big) = G(x).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>($\Leftarrow$) Take $F = G$, $a_n = \alpha_n$, $b_n = \beta_n$: then $F^n(a_nx+b_n) = G(x)$ for every $n$.</p>
<p>($\Rightarrow$) Suppose $F^n(a_nx+b_n)\to G(x)$. Fix real $t>0$ and write $m = \lfloor nt\rfloor$. Along the subsequence of indices $m$, $F^{m}(a_{m}x + b_{m}) \to G(x)$. On the other hand,
$$F^{m}(a_n x + b_n) = \big(F^n(a_nx+b_n)\big)^{m/n} \to G^t(x),$$
because $m/n \to t$ and the base converges. So the distribution functions $F^{m}$ have two affine normalisations, $(a_m, b_m)$ and $(a_n, b_n)$, each producing a nondegenerate limit ($G^t$ is nondegenerate whenever $G$ is). By convergence of types, $a_n/a_m \to A(t)$, $(b_n - b_m)/a_m \to B(t)$ and $G^t(A(t)x + B(t)) = G(x)$. Taking $t = k$ integer gives max-stability.</p>
</details>
<div class="note">
<p>Max-stability for all real $t$ is a one-parameter family of affine symmetries of $G$. Such strong symmetry leaves very little room. The next section turns it into a functional equation and solves it.</p>
</div>`,
      },
      {
        title: 'The Fisher–Tippett–Gnedenko theorem',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Extremal types (Fisher & Tippett, 1928; Gnedenko, 1943)">
<p>If $F \in \mathcal D(G)$ for a nondegenerate $G$, then $G$ is of the same type as
$$G_\xi(x) = \exp\!\big\{-(1+\xi x)^{-1/\xi}\big\},\qquad 1 + \xi x > 0,$$
for some $\xi \in \R$, where the case $\xi = 0$ is read as the limit $G_0(x) = \exp(-e^{-x})$, $x\in\R$. Equivalently $G$ is of one of the three types
$$\Lambda(x) = \exp(-e^{-x}),\qquad \Phi_\alpha(x) = \exp(-x^{-\alpha})\ (x>0),\qquad \Psi_\alpha(x) = \exp(-(-x)^{\alpha})\ (x<0),$$
with $\alpha > 0$ (and $\Phi_\alpha = 0$ for $x\le0$, $\Psi_\alpha = 1$ for $x\ge 0$).</p>
</div>
<details class="proof"><summary>Proof sketch (via the max-stability functional equation)</summary>
<p><strong>Step 1: linearise.</strong> On the interval where $0 < G < 1$, define $\varphi(x) = -\log(-\log G(x))$, which is nondecreasing, and let $\psi$ be its (left-continuous) inverse. Taking $-\log$ twice in $G^t(A(t)x + B(t)) = G(x)$ gives
$$\varphi\big(A(t)x+B(t)\big) = \varphi(x) + \log t.$$
Put $s = \log t$, $\tilde A(s) = A(e^s)$, $\tilde B(s) = B(e^s)$. Inverting, $\psi(y + s) = \tilde A(s)\,\psi(y) + \tilde B(s)$ for all real $y, s$.</p>
<p><strong>Step 2: remove the constant.</strong> Setting $y = 0$ gives $\psi(s) = \tilde A(s)\psi(0) + \tilde B(s)$; subtracting, the function $\chi(y) = \psi(y) - \psi(0)$ satisfies
$$\chi(y+s) = \tilde A(s)\,\chi(y) + \chi(s).$$
Exchanging the roles of $y$ and $s$ and comparing, $\chi(s)\,(1 - \tilde A(y)) = \chi(y)\,(1-\tilde A(s))$ for all $y,s$.</p>
<p><strong>Step 3a: $\tilde A \equiv 1$.</strong> Then $\chi(y+s) = \chi(y)+\chi(s)$, Cauchy's equation; $\chi$ is monotone, so $\chi(y) = cy$ with $c > 0$. Hence $\psi(y) = \mu + cy$, $\varphi(x) = (x-\mu)/c$, and $G(x) = \exp\{-e^{-(x-\mu)/c}\}$: the Gumbel type.</p>
<p><strong>Step 3b: $\tilde A(s_0) \ne 1$ for some $s_0$.</strong> Step 2 with $y=s_0$ gives $\chi(s) = \kappa\,(1 - \tilde A(s))$ with $\kappa = \chi(s_0)/(1-\tilde A(s_0)) \ne 0$. Substituting back into the equation of Step 2 yields $\tilde A(y+s) = \tilde A(y)\tilde A(s)$, a multiplicative Cauchy equation whose monotone solutions are $\tilde A(s) = e^{\xi s}$ with $\xi \ne 0$. So $\psi(y) = \mu + \kappa(1 - e^{\xi y})$, and monotonicity of $\psi$ forces $\kappa\xi < 0$; write $\sigma = -\kappa\xi > 0$. Then
$$x = \mu + \sigma\,\frac{e^{\xi y} - 1}{\xi} \iff y = \frac1\xi\log\!\Big(1 + \xi\,\frac{x-\mu}{\sigma}\Big),$$
and $G(x) = \exp(-e^{-y}) = \exp\{-(1+\xi\frac{x-\mu}{\sigma})^{-1/\xi}\}$ on $\{1 + \xi(x-\mu)/\sigma > 0\}$.</p>
<p>The omitted details are routine but not trivial: that $G$ is continuous with an interval as the interior of its support (so $\varphi$ is a bijection onto $\R$), and the measurability or monotonicity needed to solve Cauchy's equations. See de Haan &amp; Ferreira (2006, Thm 1.1.3) for a complete argument in this spirit.</p>
</details>
<p>Notice what the proof did <em>not</em> use: any property of $F$ beyond the existence of the limit. The family of possible limits is determined by the algebra of the maximum operation alone, just as the stable laws are determined by the algebra of addition. That is the source of the universality that lets us model maxima parametrically without knowing $F$.</p>
<div class="note" data-name="Why exactly one shape parameter">
<p>In the proof, $\xi$ is the exponent in $A(t) = t^{\xi}$: when you take the maximum of $t$ times as many observations, the spread of the maximum is multiplied by $t^\xi$. For Fréchet-type tails ($\xi > 0$) the spread grows; for Gumbel ($\xi = 0$) it is unchanged and only the location shifts (by $\log t$ times the scale); for Weibull-type ($\xi<0$) the maximum crowds against the endpoint and its spread shrinks.</p>
</div>`,
      },
      {
        title: 'The three types, and why one formula is better than three',
        html: R`
<div class="cols">
<div><h4>Gumbel $\Lambda$ ($\xi = 0$)</h4><p>$\Lambda(x) = \exp(-e^{-x})$ on $\R$. Light upper tail: $1-\Lambda(x)\sim e^{-x}$. All moments finite; mean $\gamma \approx 0.5772$, variance $\pi^2/6$. Limit for normal, exponential, gamma, lognormal parents.</p></div>
<div><h4>Fréchet $\Phi_\alpha$ ($\xi = 1/\alpha > 0$)</h4><p>$\Phi_\alpha(x) = \exp(-x^{-\alpha})$ on $(0,\infty)$. Power tail $1-\Phi_\alpha(x)\sim x^{-\alpha}$; moments of order $\ge \alpha$ infinite. Limit for Pareto, Student $t_\nu$ ($\alpha = \nu$), Cauchy ($\alpha=1$).</p></div>
<div><h4>Reversed Weibull $\Psi_\alpha$ ($\xi = -1/\alpha < 0$)</h4><p>$\Psi_\alpha(x) = \exp(-(-x)^{\alpha})$ on $(-\infty,0)$. Finite upper endpoint. Limit for uniform ($\alpha=1$), beta$(a,b)$ ($\alpha = b$), any law with a polynomially-approached endpoint.</p></div>
</div>
<p>The types are linked by simple transformations: if $Y \sim \Phi_\alpha$ then $\alpha\log Y \sim \Lambda$ and $-1/Y \sim \Psi_\alpha$. And the unified form $G_\xi$ contains them all after relocation and rescaling: for $\xi>0$, $G_\xi(x) = \Phi_{1/\xi}(1 + \xi x)$; for $\xi<0$, $G_\xi(x) = \Psi_{-1/\xi}(-(1+\xi x))$.</p>
<p>The unification (von Mises, 1936; Jenkinson, 1955) is not cosmetic. In data analysis we do not know in advance which type applies, and choosing among three separate models is a discrete decision whose uncertainty is lost once made. With $\xi$ as a continuous parameter, the data choose the type, and the uncertainty about $\xi$ — which usually dominates the uncertainty of long-range extrapolation — propagates into confidence intervals. The family $G_\xi$ is continuous in $\xi$ at $0$: $(1+\xi x)^{-1/\xi} \to e^{-x}$.</p>
<div class="warn" data-name="Weibull is an overloaded name">
<p>The extreme-value Weibull type $\Psi_\alpha$ is a law for <em>maxima</em> with a finite upper endpoint. The Weibull distribution of reliability engineering, $F(x) = 1 - e^{-(x/\lambda)^k}$ on $x>0$, is the law of a <em>minimum</em> (it is $-\Psi$ reflected and shifted), and its own maxima are in the <em>Gumbel</em> domain. Mixing the two up is a classic error.</p>
</div>`,
      },
      {
        title: 'Watching the limit emerge',
        html: R`
<p>To see convergence in practice we need concrete normalising constants. A choice that works for all the standard parents (justified on the <a href="#/t/gev-distribution">next topic</a> via the von Mises conditions) is
$$b_n = F^{\leftarrow}(1 - 1/n),\qquad a_n = \frac{\bar F(b_n)}{f(b_n)} = \frac{1}{n f(b_n)},$$
the $1-1/n$ quantile and the reciprocal hazard there. With these constants the limit is the standard $G_\xi$ for <em>every</em> parent, which makes different parents directly comparable. In the lab, maxima are drawn exactly by inverse transform: $M_n \overset{d}{=} F^{\leftarrow}(U^{1/n})$ for $U\sim\text{Unif}(0,1)$, so even block sizes of $10^6$ cost nothing.</p>
<p>Things to try: the exponential converges almost immediately; the Cauchy and Pareto converge quickly to their Fréchet limits; the uniform's limit $\Psi_1$ is reached essentially exactly. The normal is the instructive case. At $n = 365$ the fitted shape of the simulated maxima is about $-0.08$; at $n = 10^6$ the histogram is visually indistinguishable from the Gumbel density and passes a Kolmogorov–Smirnov test with 4000 maxima, yet the fitted $\hat\xi$ is still about $-0.045 \pm 0.01$ — clearly negative, and close to the penultimate shape discussed next.</p>`,
        lab: 'fisher-tippett-convergence',
      },
      {
        title: 'Rates of convergence and penultimate approximations',
        html: R`
<p>The theorem says nothing about how fast $F^n(a_nx+b_n)\to G(x)$, and the answer varies enormously.</p>
<ul>
<li><strong>Exponential:</strong> $(1-e^{-x}/n)^n - \exp(-e^{-x}) = O(1/n)$ uniformly.</li>
<li><strong>Pareto:</strong> with $a_n = n^{1/\alpha}$ the error is again $O(1/n)$.</li>
<li><strong>Normal:</strong> for the best affine constants the uniform error is of exact order $1/\log n$ (Hall, 1979). At $n = 365$, $\log n \approx 5.9$; even at $n = 10^6$ it is under $14$.</li>
</ul>
<p>The normal's slowness has a clean explanation in terms of the next topic's von Mises function $r = \bar F/f$: the "local shape" of the tail at level $b_n$ is $r'(b_n)$, which for the normal is about $-1/b_n^2 \approx -1/(2\log n)$, not $0$. A GEV with this <em>penultimate</em> shape $\xi_n = r'(b_n)$ approximates the law of $M_n$ much better than the Gumbel limit does (Fisher &amp; Tippett noticed this already in 1928; see Gomes &amp; de Haan, 1999).</p>
<div class="warn" data-name="Practical consequence">
<p>When annual maxima of approximately normal daily data are fitted with a GEV, an estimate $\hat\xi$ slightly below zero (about $-0.1$ for $n = 365$) is what the theory predicts at finite $n$, not evidence of a bounded tail. Conversely, extrapolating with the penultimate fitted $\hat\xi$ far beyond the record is internally consistent: maxima over longer periods have a penultimate shape closer to zero, but that drift is slow.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show directly that the Gumbel law is max-stable and find $\alpha_k$, $\beta_k$.`,
        a: R`$\Lambda^k(x) = \exp(-k e^{-x}) = \exp(-e^{-(x - \log k)})= \Lambda(x - \log k)$. So $\Lambda^k(x + \log k) = \Lambda(x)$: $\alpha_k = 1$, $\beta_k = \log k$. The maximum of $k$ Gumbel variables is a Gumbel shifted by $\log k$, with unchanged spread ($\xi = 0$).` },
      { q: R`Show that $\Phi_\alpha$ is max-stable, and identify the exponent $\xi$ in $A(t) = t^\xi$.`,
        a: R`$\Phi_\alpha^k(x) = \exp(-kx^{-\alpha}) = \exp(-(k^{-1/\alpha}x)^{-\alpha}) = \Phi_\alpha(k^{-1/\alpha}x)$, so $\Phi_\alpha^k(k^{1/\alpha}x) = \Phi_\alpha(x)$: $\alpha_k = k^{1/\alpha}$, $\beta_k = 0$. Thus $A(t) = t^{1/\alpha}$ and $\xi = 1/\alpha$.` },
      { q: R`The convergence of types theorem is the key step showing a limit must be max-stable. Where exactly is it used, and what would go wrong without it?`,
        a: R`For $m = \lfloor nt\rfloor$ we know two normalisations of the same distribution function $F^m$ with nondegenerate limits $G$ and $G^t$. Convergence of types guarantees the two normalisations differ asymptotically by an affine map, giving $G^t(A x + B) = G(x)$. Without it, the two limits could in principle be unrelated, and we would have no functional equation to solve.` },
      { q: R`Does the extremal types theorem imply that the normalised maximum of every continuous distribution converges to a GEV?`,
        a: R`No. It says that <em>if</em> a nondegenerate limit exists, it is GEV. Existence requires a regularity condition on the tail (regular variation, or the de Haan condition for the Gumbel domain). There are continuous distributions, with sufficiently irregular tails, in no domain of attraction; and discrete laws such as the Poisson and geometric fail the condition $\bar F(x)/\bar F(x^-) \to 1$.` },
      { q: R`Why does a normal parent converge so slowly to the Gumbel law, and what does that imply for the sign of a fitted GEV shape?`,
        a: R`The local tail shape at level $b_n$, $r'(b_n)$ with $r = \bar F/f$ the Mills ratio, is about $-1/b_n^2 \approx -1/(2\log n)$: it tends to $0$ only logarithmically. Finite-$n$ maxima therefore look like a GEV with a slightly negative shape, and fitted $\hat\xi$ will tend to be slightly negative.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'gev-distribution',
    domain: 'evt',
    order: 30,
    title: 'The GEV Distribution & Domains of Attraction',
    subtitle: 'The generalised extreme value family, and which parents lead to which tail shape',
    level: 2,
    sources: { other: 'Coles (2001) §3.1–3.3; Embrechts et al. (1997) §3.3; de Haan & Ferreira (2006) §1.1–1.2; Resnick (1987) Ch. 1' },
    prereqs: ['fisher-tippett'],
    related: ['distribution-families', 'peaks-over-threshold', 'tail-index'],
    tags: ['GEV', 'regular variation', 'von Mises conditions', 'normalising constants'],
    summary: R`The generalised extreme value family $\text{GEV}(\mu,\sigma,\xi)$, with CDF $\exp\{-[1+\xi(z-\mu)/\sigma]^{-1/\xi}\}$, is the location–scale closure of the extremal types. Its shape $\xi$ sets the tail: a finite endpoint when $\xi<0$, an exponential tail when $\xi = 0$, a power tail with only moments of order $< 1/\xi$ when $\xi > 0$. Which $\xi$ a parent $F$ leads to is decided by its tail alone: regularly varying tails give $\xi>0$, tails that vanish polynomially at a finite endpoint give $\xi<0$, and a broad intermediate class gives $\xi = 0$. The von Mises condition $\frac{\d}{\d x}\big(\bar F/f\big)\to\xi$ is a convenient sufficient test, and it also produces the normalising constants.`,
    survey: {
      task: 'Parametrise extreme value limits; decide which limit a given parent leads to',
      assumptions: R`$\sigma > 0$; support $1+\xi(z-\mu)/\sigma > 0$; domain-of-attraction conditions on $\bar F$ near $x_F$`,
      output: R`CDF, quantiles $z_p$, moments (finite iff order $< 1/\xi$); constants $a_n, b_n$`,
      strengths: R`Continuous in $\xi$; closed under taking maxima (max-stability) with explicit parameter map`,
      weaknesses: R`Support depends on parameters (non-regular likelihood for $\xi \le -1/2$); $\xi$ describes only the far tail`,
      complexity: R`Closed-form CDF and quantile`,
    },
    sections: [
      {
        title: 'The GEV family',
        html: R`
<div class="def" data-name="Generalised extreme value distribution">
<p>For $\mu\in\R$, $\sigma>0$, $\xi\in\R$,
$$G(z;\mu,\sigma,\xi) = \exp\!\left\{-\left[1 + \xi\,\frac{z-\mu}{\sigma}\right]_+^{-1/\xi}\right\},$$
defined on $\{z : 1 + \xi(z-\mu)/\sigma > 0\}$, with the Gumbel case $\xi = 0$ read as $\exp\{-e^{-(z-\mu)/\sigma}\}$. The density is
$$g(z) = \frac1\sigma\, t(z)^{\xi + 1} e^{-t(z)},\qquad t(z) = \Big[1+\xi\tfrac{z-\mu}{\sigma}\Big]^{-1/\xi}\ (\text{or } e^{-(z-\mu)/\sigma} \text{ if } \xi=0).$$</p>
</div>
<p>The support is $(\mu - \sigma/\xi, \infty)$ when $\xi>0$, all of $\R$ when $\xi = 0$, and $(-\infty, \mu - \sigma/\xi)$ when $\xi < 0$: so for $\xi < 0$ there is a finite upper endpoint $z^+ = \mu + \sigma/|\xi|$, and for $\xi > 0$ a finite lower endpoint. Note that $\mu$ and $\sigma$ are location and scale but <em>not</em> mean and standard deviation.</p>
<table>
<thead><tr><th>Quantity</th><th>$\xi \ne 0$</th><th>$\xi = 0$</th></tr></thead>
<tbody>
<tr><td>Quantile $z_p$ with $G(z_p) = 1-p$</td><td>$\mu - \frac{\sigma}{\xi}\big[1 - \{-\log(1-p)\}^{-\xi}\big]$</td><td>$\mu - \sigma\log\{-\log(1-p)\}$</td></tr>
<tr><td>Mean (needs $\xi<1$)</td><td>$\mu + \sigma\,\frac{\Gamma(1-\xi) - 1}{\xi}$</td><td>$\mu + \gamma\sigma$, $\gamma\approx0.5772$</td></tr>
<tr><td>Variance (needs $\xi < 1/2$)</td><td>$\sigma^2\,\frac{\Gamma(1-2\xi) - \Gamma(1-\xi)^2}{\xi^2}$</td><td>$\sigma^2\pi^2/6$</td></tr>
<tr><td>$\E|Z|^k<\infty$</td><td>iff $k < 1/\xi$ (all $k$ if $\xi<0$)</td><td>all $k$</td></tr>
</tbody>
</table>
<p>The moment formulas come from writing $Z = \mu + \sigma(E^{-\xi}-1)/\xi$ with $E \sim \text{Exp}(1)$ — the GEV is a power transform of an exponential — and $\E E^{-\xi k} = \Gamma(1-\xi k)$.</p>
<div class="thm" data-kind="Proposition" data-name="Max-stability in GEV coordinates">
<p>If $Z \sim \text{GEV}(\mu,\sigma,\xi)$ then the maximum of $k$ IID copies is $\text{GEV}(\mu_k,\sigma_k,\xi)$ with the <em>same</em> shape and
$$\sigma_k = \sigma k^{\xi},\qquad \mu_k = \mu + \sigma\,\frac{k^{\xi} - 1}{\xi}\quad(\mu_k = \mu + \sigma\log k \text{ if } \xi = 0).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$G^k(z) = \exp\{-k[1 + \xi(z-\mu)/\sigma]^{-1/\xi}\} = \exp\{-[k^{-\xi} + k^{-\xi}\xi(z-\mu)/\sigma]^{-1/\xi}\}$. We need $k^{-\xi}(1+\xi(z-\mu)/\sigma) = 1 + \xi(z-\mu_k)/\sigma_k$ for all $z$. Matching the coefficient of $z$ gives $\sigma_k = \sigma k^{\xi}$; matching constants gives $1 - \xi\mu_k/\sigma_k = k^{-\xi} - \xi\mu/\sigma_k$, i.e. $\mu_k - \mu = \sigma_k(1 - k^{-\xi})/\xi = \sigma(k^\xi - 1)/\xi$.</p>
</details>
<p>Practically: if monthly maxima are GEV, annual maxima are GEV with the same $\xi$ and the parameters above with $k = 12$ (under independence). Block size changes $\mu$ and $\sigma$ but not $\xi$, which is why $\xi$ is the fundamental parameter.</p>`,
        lab: 'gev-distribution-explorer',
      },
      {
        title: 'Domains of attraction: the characterisation theorems',
        html: R`
<p>Which parents lead to which $\xi$? The answer is phrased through slowly and regularly varying functions.</p>
<div class="def" data-name="Regular variation">
<p>A positive measurable function $L$ on $(0,\infty)$ is <em>slowly varying</em> if $L(tx)/L(t) \to 1$ as $t\to\infty$ for every $x > 0$ (examples: constants, $\log t$, $\log\log t$). A function $h$ is <em>regularly varying with index $\rho$</em>, $h \in RV_\rho$, if $h(tx)/h(t) \to x^\rho$; equivalently $h(t) = t^\rho L(t)$ with $L$ slowly varying.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Gnedenko (1943); de Haan (1970)">
<ol>
<li><strong>Fréchet</strong> ($\xi = 1/\alpha > 0$): $F \in \mathcal D(\Phi_\alpha)$ iff $x_F = \infty$ and $\bar F(x) = x^{-\alpha}L(x)$ with $L$ slowly varying. One may take $b_n = 0$, $a_n = F^{\leftarrow}(1-1/n)$.</li>
<li><strong>Weibull</strong> ($\xi = -1/\alpha < 0$): $F \in \mathcal D(\Psi_\alpha)$ iff $x_F < \infty$ and $\bar F(x_F - 1/x) = x^{-\alpha}L(x)$ with $L$ slowly varying. One may take $b_n = x_F$, $a_n = x_F - F^{\leftarrow}(1-1/n)$.</li>
<li><strong>Gumbel</strong> ($\xi=0$): $F \in \mathcal D(\Lambda)$ iff there is a positive function $a(\cdot)$ with
$$\lim_{t\uparrow x_F}\frac{\bar F(t + x\,a(t))}{\bar F(t)} = e^{-x}\qquad\text{for all } x\in\R.$$
One may take $a(t) = \E[X - t \mid X > t]$, the mean excess function, and $b_n = F^{\leftarrow}(1-1/n)$, $a_n = a(b_n)$.</li>
</ol>
</div>
<details class="proof"><summary>Proof of sufficiency in the Fréchet case</summary>
<p>Let $a_n = F^{\leftarrow}(1-1/n) \to \infty$. Regular variation of $\bar F$ implies $n\bar F(a_n) \to 1$ (the jumps of $\bar F$ are asymptotically negligible relative to $\bar F$). Then for $x > 0$
$$n\bar F(a_n x) = n\bar F(a_n)\cdot\frac{\bar F(a_n x)}{\bar F(a_n)} \to 1\cdot x^{-\alpha},$$
and the <a href="#/t/evt-intro">Poisson lemma</a> gives $F^n(a_nx) \to \exp(-x^{-\alpha}) = \Phi_\alpha(x)$. For $x\le 0$, $F^n(a_nx) \le F^n(a_n\varepsilon) \to \Phi_\alpha(\varepsilon)$ for every $\varepsilon>0$, which tends to $0$ as $\varepsilon\downarrow0$. The Weibull case follows by the transformation $X \mapsto 1/(x_F - X)$, which maps a Weibull-type tail at $x_F$ to a Fréchet-type tail at $\infty$.</p>
</details>
<p>Three things to notice. First, only the tail matters: modifying $F$ on any bounded set below $x_F$ does not change its domain. Second, in the Fréchet domain $\xi$ is a property of $\bar F$ alone, $\xi = 1/\alpha$, with no location or scale involved. Third, the unified statement of all three cases is
$$F\in\mathcal D(G_\xi) \iff \lim_{t\uparrow x_F}\frac{\bar F(t + x\,a(t))}{\bar F(t)} = (1 + \xi x)^{-1/\xi}\ \text{ for some } a(t)>0,$$
which says: the conditional distribution of a scaled excess over a high threshold converges to a generalised Pareto law. That is the Pickands–Balkema–de Haan theorem underlying <a href="#/t/peaks-over-threshold">peaks over threshold</a>.</p>`,
      },
      {
        title: 'The von Mises conditions: a practical test',
        html: R`
<p>Checking regular variation or the de Haan condition directly can be awkward. For distributions with a density there is a simple sufficient condition in terms of the <em>reciprocal hazard</em> $r(x) = \bar F(x)/f(x)$.</p>
<div class="thm" data-kind="Theorem" data-name="von Mises (1936)">
<p>Suppose $F$ has a positive density $f$ on a left neighbourhood of $x_F$, differentiable there, and
$$\lim_{x\uparrow x_F} r'(x) = \xi,\qquad r = \bar F/f.$$
Then $F\in\mathcal D(G_\xi)$, with normalising constants $b_n = F^{\leftarrow}(1-1/n)$ and $a_n = r(b_n) = 1/(nf(b_n))$, and the limit is the standard $G_\xi$ itself. Equivalent forms: $\xi>0$ if $x f(x)/\bar F(x) \to \alpha = 1/\xi$ with $x_F=\infty$; $\xi<0$ if $(x_F-x)f(x)/\bar F(x) \to \alpha = -1/\xi$ with $x_F<\infty$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Since $(\log\bar F)' = -f/\bar F = -1/r$, we have the representation $\bar F(x) = \bar F(x_0)\exp\{-\int_{x_0}^x \d s/r(s)\}$. Hence for fixed $x$ with $1 + \xi x > 0$,
$$\frac{\bar F(t + x\,r(t))}{\bar F(t)} = \exp\Big\{-\int_0^x \frac{r(t)}{r(t + u\,r(t))}\,\d u\Big\},$$
after substituting $s = t + u\,r(t)$. By the mean value theorem $r(t+u\,r(t)) - r(t) = r'(\eta)\,u\,r(t)$ for some $\eta$ between $t$ and $t + u r(t)$, so $r(t+u\,r(t))/r(t) \to 1 + \xi u$ locally uniformly in $u$ (using $r(t)/t \to \xi$ when $x_F=\infty$, or $r(t)\to0$ when $x_F < \infty$, so that $\eta \uparrow x_F$). Therefore the integral tends to $\int_0^x \d u/(1+\xi u) = \xi^{-1}\log(1+\xi x)$, and the ratio tends to $(1+\xi x)^{-1/\xi}$. With $t = b_n$, $n\bar F(b_n) = 1$ and the Poisson lemma, $F^n(b_n + a_n x) \to G_\xi(x)$.</p>
</details>
<p>The function $r'(x)$ is a "local shape" of the tail. Evaluated at the level $b_n$ relevant for blocks of size $n$, it is the penultimate shape discussed in <a href="#/t/fisher-tippett">the previous topic</a>. The lab plots $r'(b_n)$ against the block size $n$ for several parents: the curves settle at the limiting $\xi$, some within a few observations, the normal and lognormal only at astronomically large $n$.</p>`,
        lab: 'gev-distribution-vonmises',
      },
      {
        title: 'A catalogue of parents and constants',
        html: R`
<table>
<thead><tr><th>Parent $F$</th><th>Tail</th><th>$\xi$</th><th>$b_n$</th><th>$a_n$</th></tr></thead>
<tbody>
<tr><td>Exponential(1)</td><td>$e^{-x}$</td><td>$0$</td><td>$\log n$</td><td>$1$</td></tr>
<tr><td>Gamma$(k,1)$</td><td>$\sim x^{k-1}e^{-x}/\Gamma(k)$</td><td>$0$</td><td>$\log n + (k-1)\log\log n - \log\Gamma(k)$</td><td>$1$</td></tr>
<tr><td>Weibull (reliability) $e^{-x^k}$</td><td>$e^{-x^k}$</td><td>$0$</td><td>$(\log n)^{1/k}$</td><td>$k^{-1}(\log n)^{1/k - 1}$</td></tr>
<tr><td>Normal $\N(0,1)$</td><td>$\sim\phi(x)/x$</td><td>$0$</td><td>$\sqrt{2\log n} - \dfrac{\log\log n + \log 4\pi}{2\sqrt{2\log n}}$</td><td>$(2\log n)^{-1/2}$</td></tr>
<tr><td>Lognormal</td><td>$\sim \phi(\log x)/\log x$</td><td>$0$</td><td>$e^{b_n^{\N}}$</td><td>$a_n^{\N}e^{b_n^{\N}}$</td></tr>
<tr><td>Pareto$(\alpha)$</td><td>$x^{-\alpha}$</td><td>$1/\alpha$</td><td>$0$</td><td>$n^{1/\alpha}$</td></tr>
<tr><td>Student $t_\nu$</td><td>$\sim c_\nu x^{-\nu}$</td><td>$1/\nu$</td><td>$0$</td><td>$(c_\nu n)^{1/\nu}$</td></tr>
<tr><td>Cauchy</td><td>$\sim 1/(\pi x)$</td><td>$1$</td><td>$0$</td><td>$n/\pi$</td></tr>
<tr><td>Uniform$(0,1)$</td><td>$1-x$</td><td>$-1$</td><td>$1$</td><td>$1/n$</td></tr>
<tr><td>Beta$(a,b)$</td><td>$\sim c\,(1-x)^{b}$</td><td>$-1/b$</td><td>$1$</td><td>$(cn)^{-1/b}$</td></tr>
<tr><td>$1-e^{-x/(1-x)}$ on $(0,1)$</td><td>finite endpoint</td><td>$0$</td><td colspan="2">Gumbel despite $x_F = 1$</td></tr>
<tr><td>Poisson, geometric</td><td>atoms</td><td colspan="3">no nondegenerate limit</td></tr>
</tbody>
</table>
<p>In the lognormal row, $a_n^{\N}, b_n^{\N}$ are the normal constants; the approximation uses $a_n^{\N}\to 0$. The last-but-one row shows that a finite endpoint does not imply $\xi<0$: here $r(x) = (1-x)^2$, $r'(x) = -2(1-x) \to 0$.</p>
<div class="ex" data-name="Deriving the normal constants">
<p>Solve $n\bar\Phi(b_n) = 1$ using Mills' ratio $\bar\Phi(x) = \phi(x)x^{-1}(1 + O(x^{-2}))$. Taking logs: $\log n - b_n^2/2 - \log b_n - \tfrac12\log 2\pi + o(1) = 0$. To first order $b_n^2 \approx 2\log n$; substituting $\log b_n \approx \tfrac12\log(2\log n)$ into the correction gives $b_n^2 = 2\log n - \log\log n - \log 4\pi + o(1)$, and taking the square root yields the tabulated $b_n$. Since $r(x) = \bar\Phi(x)/\phi(x) \sim 1/x$, $a_n = r(b_n) \sim 1/b_n \sim (2\log n)^{-1/2}$. Because $b_n$ is only determined up to $o(a_n)$ by this expansion, the error terms are of relative order $1/\log n$ — the slow convergence again.</p>
</div>`,
      },
      {
        title: 'Tail heaviness, moments and the meaning of ξ',
        html: R`
<p>For $F \in \mathcal D(\Phi_\alpha)$, Karamata's theorem gives $\E X_+^k < \infty$ for $k < \alpha$ and $= \infty$ for $k>\alpha$. So $\xi = 1/\alpha$ directly counts moments: $\xi = 0.25$ means a finite variance but infinite fourth moment; $\xi \ge 0.5$ means infinite variance; $\xi \ge 1$ means no mean. Financial returns typically show $\xi\approx 0.2$–$0.4$; insurance losses (fire, wind, cyber) often $0.5$–$1$; many environmental variables are close to $0$ or slightly negative.</p>
<p>The Gumbel domain is surprisingly broad. It contains distributions with all moments finite and very light tails (normal), distributions with all moments finite but heavy, subexponential tails (lognormal), and even distributions with a finite endpoint. So $\xi = 0$ is a statement about the <em>asymptotic</em> rate at which the tail decays relative to its own mean excess, not about "lightness" in any everyday sense.</p>
<div class="warn" data-name="Pitfalls">
<ul>
<li>$\xi$ is a property of the far tail only. Two parents with identical $\xi$ can have wildly different bulk and very different finite-$n$ behaviour (Pareto versus $t$ with the same index).</li>
<li>The domain of attraction says nothing about rate. A lognormal sample of size $10^4$ will usually yield a clearly positive fitted $\xi$.</li>
<li>Do not confuse the GEV $\mu,\sigma$ with mean and standard deviation; for $\xi\ge 1/2$ the variance does not exist at all.</li>
<li>Domain membership is not preserved by every transformation: $\log X$ of a Fréchet-domain variable is in the Gumbel domain, and $\exp(X)$ of a Gumbel-domain variable can be in any domain.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Verify the von Mises condition for the Pareto$(\alpha)$ and identify $a_n, b_n$ from the theorem. Compare with the "Fréchet" constants $a_n = n^{1/\alpha}$, $b_n = 0$.`,
        a: R`$r(x) = \bar F/f = x^{-\alpha}/(\alpha x^{-\alpha-1}) = x/\alpha$, so $r' = 1/\alpha = \xi$. The theorem gives $b_n = n^{1/\alpha}$, $a_n = r(b_n) = n^{1/\alpha}/\alpha$, and $(M_n - b_n)/a_n \dto G_{1/\alpha}$. This is the same type as $\Phi_\alpha$: indeed $(M_n - b_n)/a_n = \alpha(M_n/n^{1/\alpha} - 1)$, an affine function of $M_n/n^{1/\alpha} \dto \Phi_\alpha$, and $G_\xi(x) = \Phi_{1/\xi}(1+\xi x)$.` },
      { q: R`If daily maxima are $\text{GEV}(\mu,\sigma,\xi)$ and independent, what is the distribution of the maximum over 30 days? Evaluate for $\xi = 0$.`,
        a: R`$\text{GEV}(\mu_{30},\sigma_{30},\xi)$ with $\sigma_{30} = 30^\xi\sigma$ and $\mu_{30} = \mu + \sigma(30^\xi-1)/\xi$. For $\xi = 0$: $\sigma_{30} = \sigma$, $\mu_{30} = \mu + \sigma\log 30 \approx \mu + 3.40\sigma$ — pure shift.` },
      { q: R`Explain why $F(x) = 1 - e^{-x/(1-x)}$ on $(0,1)$ has a finite endpoint but is in the Gumbel domain.`,
        a: R`Its density is $f(x) = e^{-x/(1-x)}/(1-x)^2$, so $r = \bar F/f = (1-x)^2$ and $r'(x) = -2(1-x) \to 0$ as $x\uparrow1$. By von Mises, $F\in\mathcal D(\Lambda)$. The tail vanishes faster than any power of $(1-x)$ at the endpoint, so $\bar F(1-1/x)$ is not regularly varying, ruling out the Weibull domain.` },
      { q: R`For a $t_3$ parent, which moments of the annual maximum exist, and what is $\xi$?`,
        a: R`The $t_\nu$ tail is $\sim c_\nu x^{-\nu}$, so it is in $\mathcal D(\Phi_\nu)$ with $\xi = 1/\nu = 1/3$. The GEV limit has finite moments of order $k < 3$ only: mean and variance exist, the third moment does not. (The same holds for the parent itself.)` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'block-maxima',
    domain: 'evt',
    order: 40,
    title: 'Block Maxima: Fitting the GEV',
    subtitle: 'Maximum likelihood, its unusual regularity conditions, and L-moment alternatives',
    level: 2,
    sources: { other: 'Coles (2001) Ch. 3; Smith (1985), Biometrika 72; Hosking, Wallis & Wood (1985), Technometrics 27; Prescott & Walden (1980)' },
    prereqs: ['gev-distribution', 'mle'],
    related: ['fisher-information', 'method-of-moments', 'return-levels', 'parametric-bootstrap'],
    tags: ['GEV', 'maximum likelihood', 'non-regular models', 'L-moments', 'model checking'],
    summary: R`Split the record into blocks — usually years — keep each block's maximum, and model those maxima as IID $\text{GEV}(\mu,\sigma,\xi)$; the unknown normalising constants are absorbed into $\mu$ and $\sigma$. Maximum likelihood is the default fit, but the GEV is not a regular model: its support depends on the parameters. Smith (1985) showed the usual $\sqrt m$ asymptotic normality holds only when $\xi > -1/2$, that the MLE may not exist when $\xi < -1$, and the middle range is non-standard. L-moment (probability-weighted moment) estimators are a robust alternative that excel in small samples. The block size embodies a bias–variance trade-off that never goes away.`,
    survey: {
      task: R`Estimate $(\mu,\sigma,\xi)$ from $m$ block maxima`,
      assumptions: R`Blocks long enough for the GEV approximation; maxima IID (stationarity across blocks)`,
      output: R`$(\hat\mu,\hat\sigma,\hat\xi)$, observed-information standard errors, diagnostic plots`,
      strengths: 'Simple, interpretable (annual scale), robust to within-block dependence and seasonality',
      weaknesses: R`Wasteful (one value per block); non-regular for $\xi\le -1/2$; small $m$ gives very uncertain $\xi$`,
      complexity: R`$O(m)$ per likelihood evaluation; 3-parameter numerical optimisation`,
    },
    sections: [
      {
        title: 'From a limit theorem to a statistical model',
        html: R`
<p>The <a href="#/t/fisher-tippett">extremal types theorem</a> says that for large $n$,
$$\P(M_n \le z) \approx G\big((z - b_n)/a_n\big) = G(z;\mu,\sigma,\xi)$$
where the unknown constants $a_n, b_n$ have been absorbed into location and scale. That is the entire modelling step: we never need to know $a_n$ and $b_n$, or even $F$. The model is used as follows.</p>
<div class="algo" data-name="Block maxima method">
<ol>
<li>Partition the observations into $m$ blocks of length $n$ (for daily data, $n = 365$ and blocks are years).</li>
<li>Extract the block maxima $Z_1,\dots,Z_m$.</li>
<li>Fit $Z_j \iid \text{GEV}(\mu,\sigma,\xi)$, usually by maximum likelihood.</li>
<li>Check fit (probability, quantile and return-level plots) and report return levels with honest intervals.</li>
</ol>
</div>
<p>Annual blocks are popular for good reasons: they make the result directly interpretable ("annual maximum"), they absorb seasonality (each year contains every season), and within-block dependence matters little — by the <a href="#/t/extremal-index">extremal index</a> theory, dependent maxima are still approximately GEV with adjusted $\mu,\sigma$.</p>
<div class="note" data-name="The block-size trade-off">
<p>Long blocks make the GEV approximation accurate (small bias, since the limit theorem kicks in as $n\to\infty$) but leave few maxima (large variance). Short blocks give many maxima but each is only weakly "extreme", so the GEV is a poorer approximation and systematic error creeps in. With a fixed record of $N = mn$ observations, there is no free lunch; the <a href="#/t/peaks-over-threshold">peaks-over-threshold</a> approach uses the data more efficiently, but faces the same trade-off through the threshold.</p>
</div>`,
      },
      {
        title: 'Maximum likelihood',
        html: R`
<p>For $\xi\ne0$ the log-likelihood of $z_1,\dots,z_m$ is
$$\ell(\mu,\sigma,\xi) = -m\log\sigma - \Big(1+\frac1\xi\Big)\sum_{i=1}^m \log\Big[1+\xi\frac{z_i-\mu}{\sigma}\Big] - \sum_{i=1}^m\Big[1+\xi\frac{z_i-\mu}{\sigma}\Big]^{-1/\xi},$$
valid only if $1 + \xi(z_i-\mu)/\sigma > 0$ for <em>every</em> $i$; outside this set the likelihood is zero. For $\xi = 0$,
$$\ell(\mu,\sigma) = -m\log\sigma - \sum_i \frac{z_i-\mu}{\sigma} - \sum_i \exp\Big(-\frac{z_i-\mu}{\sigma}\Big).$$
There is no closed form for the GEV MLE; it is found numerically. In practice: optimise over $(\mu,\log\sigma,\xi)$, return $-\infty$ (or a large penalty) outside the support constraint, start from moment or L-moment estimates, and treat $|\xi|$ very near $0$ with the Gumbel formula to avoid cancellation. Standard errors come from inverting the observed information (the Hessian of $-\ell$) at the optimum, and are justified by the asymptotic theory of the <a href="#/t/mle">MLE</a> — under the condition discussed in the next section.</p>
<div class="ex" data-name="The Gumbel likelihood equations">
<p>For $\xi = 0$, setting $\partial\ell/\partial\mu = 0$ gives $\sum_i e^{-(z_i-\mu)/\sigma} = m$, i.e.
$$\hat\mu = -\hat\sigma\log\Big(\frac1m\sum_i e^{-z_i/\hat\sigma}\Big).$$
Substituting this into $\partial\ell/\partial\sigma = 0$, which reads $m\sigma = \sum_i (z_i-\mu)\big(1 - e^{-(z_i-\mu)/\sigma}\big)$, leaves a single equation in $\sigma$:
$$\hat\sigma = \bar z - \frac{\sum_i z_i e^{-z_i/\hat\sigma}}{\sum_i e^{-z_i/\hat\sigma}},$$
the sample mean minus a weighted mean that up-weights the smaller maxima. It is solved by one-dimensional root finding. The GEV case has no such reduction, but the same logic — $\mu$ can be profiled out cheaply — is a useful numerical device.</p>
</div>
<p>In the lab, simulate a record from a chosen parent (maxima are drawn exactly via $F^{\leftarrow}(U^{1/n})$), fit the GEV by maximum likelihood and by L-moments, and inspect the fit through the density and a quantile plot. Change the block length to see the trade-off between a better approximation and fewer maxima.</p>`,
        lab: 'block-maxima-fit',
      },
      {
        title: 'Regularity: Smith (1985) and the ξ = −1/2 boundary',
        html: R`
<p>The textbook theorem that the MLE is consistent, asymptotically normal and efficient assumes, among other things, that the support of the density does not depend on the parameter. The GEV violates this: the endpoint $\mu - \sigma/\xi$ moves with the parameters. Whether the standard theory survives depends on how fast the density vanishes at the endpoint.</p>
<div class="thm" data-kind="Theorem" data-name="Smith (1985)">
<p>For IID GEV data with true shape $\xi$:</p>
<ol>
<li>If $\xi > -1/2$, the MLE is regular: $\sqrt m(\hat\btheta - \btheta) \dto \N\big(0, I(\btheta)^{-1}\big)$ with finite, positive definite Fisher information $I$, and it is asymptotically efficient.</li>
<li>If $-1 < \xi \le -1/2$, a local maximiser exists with probability tending to one, but it is not asymptotically normal at the usual rate (the endpoint is estimated faster than $m^{-1/2}$).</li>
<li>If $\xi < -1$, the likelihood is unbounded: the MLE in the usual sense does not exist.</li>
</ol>
</div>
<details class="proof"><summary>Proof sketch (why −1/2 and −1)</summary>
<p>Let $\xi<0$ and write $z^+ = \mu - \sigma/\xi$ for the endpoint. Near $z^+$, $t(z) = [1+\xi(z-\mu)/\sigma]^{-1/\xi} \to 0$ and the density behaves like
$$g(z) \asymp (z^+ - z)^{-1/\xi - 1}.$$
<em>Unboundedness:</em> if $\xi<-1$ then $-1/\xi - 1 \in (-1, 0)$, so $g$ is infinite at the endpoint. Moving $z^+$ down to $\max_i z_i$ makes one factor of the likelihood infinite while the others stay positive: $\sup\ell = \infty$.</p>
<p><em>Fisher information:</em> the score with respect to the endpoint behaves like $\partial \log g/\partial z^+ \asymp (z^+ - z)^{-1}$. Its second moment is finite iff $\int_0 s^{-2}\, s^{-1/\xi - 1}\d s < \infty$, i.e. iff $-1/\xi - 3 > -1$, i.e. $\xi > -1/2$. So the information about the endpoint is finite exactly when $\xi > -1/2$; at and below that boundary the endpoint is "super-efficiently" estimable and the quadratic expansion behind asymptotic normality breaks down.</p>
</details>
<p>Fortunately, in most applications $\xi > -1/2$: shapes as negative as $-0.5$ imply a very sharply bounded tail and are rare in environmental and financial data. But the theorem has a practical sting at small $m$: $\hat\xi$ is highly variable, its sampling distribution is skewed, and Wald intervals $\hat\xi \pm 1.96\,\widehat{\se}$ can have poor coverage even when the truth is comfortably regular. The second lab shows this by repeating the fit thousands of times.</p>
<div class="note" data-name="Testing ξ = 0">
<p>The Gumbel model is the GEV with $\xi = 0$, an interior point of the parameter space, so the deviance $2\{\ell(\hat\mu,\hat\sigma,\hat\xi) - \ell(\tilde\mu,\tilde\sigma,0)\}$ is asymptotically $\chi^2_1$ under $\xi=0$ (a standard <a href="#/t/likelihood-ratio-tests">likelihood ratio test</a>). Be cautious about adopting the Gumbel model just because the test does not reject: with $m=30$–$50$ the test has little power, and fixing $\xi=0$ throws away the dominant source of uncertainty in long-range return levels, producing intervals that are far too narrow.</p>
</div>`,
      },
      {
        title: 'Probability-weighted moments and L-moments',
        html: R`
<p>Classical moments are poor estimators for heavy-tailed data (they may not exist and are dominated by the largest values). <em>Probability-weighted moments</em> (Greenwood et al., 1979) and their linear combinations, <em>L-moments</em> (Hosking, 1990), weight the ordered sample by polynomials in the ranks and need only a finite mean.</p>
<div class="def" data-name="PWMs and L-moments">
<p>$\beta_r = \E[Z\,G(Z)^r]$, $r = 0,1,2,\dots$ With order statistics $z_{(1)}\le\dots\le z_{(m)}$, an unbiased estimator is
$$b_r = \frac1m\sum_{i=1}^m \frac{(i-1)(i-2)\cdots(i-r)}{(m-1)(m-2)\cdots(m-r)}\,z_{(i)}.$$
The first three L-moments are $\lambda_1 = \beta_0$, $\lambda_2 = 2\beta_1 - \beta_0$, $\lambda_3 = 6\beta_2 - 6\beta_1 + \beta_0$, and the L-skewness is $\tau_3 = \lambda_3/\lambda_2$.</p>
</div>
<p>For the GEV with $\xi<1$, $\xi\neq0$ (Hosking, Wallis &amp; Wood, 1985; in their notation $k = -\xi$):
$$\lambda_1 = \mu + \sigma\frac{\Gamma(1-\xi) - 1}{\xi},\qquad \lambda_2 = \sigma\,\frac{(2^{\xi} - 1)\,\Gamma(1-\xi)}{\xi},\qquad \tau_3 = \frac{2(3^{\xi}-1)}{2^{\xi}-1} - 3.$$
$\tau_3$ depends only on $\xi$, so $\xi$ is estimated by inverting it; Hosking et al. give the accurate approximation $k \approx 7.8590\,c + 2.9554\,c^2$ with $c = 2/(3 + \hat\tau_3) - \log 2/\log 3$, valid for $-0.5 < \tau_3 < 0.5$. Then $\hat\sigma$ and $\hat\mu$ follow from $\lambda_2$ and $\lambda_1$.</p>
<div class="cols">
<div><h4>Maximum likelihood</h4><p>Asymptotically efficient for $\xi>-1/2$; extends naturally to covariates (non-stationary models), censoring, and profile-likelihood intervals; can be erratic for $m \lesssim 20$.</p></div>
<div><h4>L-moments</h4><p>Always defined (given $\xi < 1$), no optimisation, low variance for small $m$, robust to a single wild maximum; not efficient for large $m$; awkward with covariates; standard errors need the bootstrap.</p></div>
</div>
<p>Hosking et al.'s simulations showed the PWM/L-moment estimators often achieving smaller mean squared error than the MLE in small samples (a few tens of maxima) for $\xi$ in the usual range; as $m$ grows the efficiency of the MLE takes over. Many practitioners use the L-moment fit as a starting value for the likelihood optimiser — as the labs here do.</p>`,
        lab: 'block-maxima-sampling',
      },
      {
        title: 'Model checking',
        html: R`
<p>With fitted $\hat G$ and ordered maxima $z_{(1)}\le\dots\le z_{(m)}$, using empirical probabilities $p_i = i/(m+1)$:</p>
<ul>
<li><strong>Probability plot:</strong> $\{(p_i,\ \hat G(z_{(i)}))\}$ — should lie on the diagonal; sensitive in the centre.</li>
<li><strong>Quantile plot:</strong> $\{(\hat G^{-1}(p_i),\ z_{(i)})\}$ — should lie on the diagonal; sensitive in the tails, which is where it matters.</li>
<li><strong>Return-level plot:</strong> $\hat z_p$ against $-\log(-\log(1-p))$ or against the return period on a log scale, with the empirical points overlaid (see <a href="#/t/return-levels">return levels</a>).</li>
<li><strong>Density plot:</strong> histogram against $\hat g$ — the least informative, but useful for spotting bimodality (mixed storm types, for instance).</li>
</ul>
<div class="warn" data-name="Two failure modes to look for">
<p>A quantile plot that bends upward at the top can mean the tail is heavier than fitted — or that the largest values come from a different mechanism (hurricanes among frontal storms; a regime change in a financial series). Mixed populations are not covered by the IID theory and should be modelled separately. A systematic trend in the residuals against time indicates non-stationarity; then make $\mu$ (and perhaps $\log\sigma$) a function of covariates, e.g. $\mu(t) = \beta_0 + \beta_1 t$, and fit by likelihood.</p>
</div>`,
      },
      {
        title: 'Practicalities and extensions',
        html: R`
<ul>
<li><strong>Uncertainty for derived quantities.</strong> Return levels are nonlinear in $(\mu,\sigma,\xi)$. Use the <a href="#/t/delta-method">delta method</a>, profile likelihood, or a <a href="#/t/parametric-bootstrap">parametric bootstrap</a>. Resampling the $m$ block maxima nonparametrically is legitimate (they are IID), unlike bootstrapping the maximum of a raw sample, which fails.</li>
<li><strong>Using more than one value per block.</strong> The $r$-largest order statistics model uses the top $r$ values in each block; its likelihood comes from the <a href="#/t/point-process-evt">point-process limit</a>. It shares parameters with the block-maximum GEV and can sharpen estimates, at the risk of bias if $r$ is too large.</li>
<li><strong>Missing data.</strong> A year with half the days missing has a maximum that is stochastically smaller. Either discard incomplete blocks or adjust via max-stability ($\sigma_k = \sigma k^\xi$ with $k$ the observed fraction).</li>
<li><strong>Bayesian fitting.</strong> Weakly informative priors on $\xi$ (e.g. concentrated on $(-0.5, 0.5)$) stabilise small-sample estimates, and the posterior predictive distribution integrates parameter uncertainty into return-level statements.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Why can the GEV likelihood be unbounded, and for which $\xi$?`,
        a: R`For $\xi<0$ the density near the upper endpoint $z^+ = \mu - \sigma/\xi$ behaves like $(z^+ - z)^{-1/\xi - 1}$. When $\xi < -1$ the exponent is negative, so $g$ blows up at $z^+$; setting $z^+ = \max_i z_i$ makes the likelihood infinite. This is Smith's third regime.` },
      { q: R`State the condition for regular (root-$m$, asymptotically normal) MLE behaviour in the GEV, and explain where it comes from.`,
        a: R`$\xi > -1/2$. The score for the endpoint grows like $(z^+-z)^{-1}$; its variance under a density $\asymp (z^+-z)^{-1/\xi-1}$ is finite iff $-1/\xi - 3 > -1$, i.e. $\xi > -1/2$. Finite Fisher information is what the usual quadratic expansion of the log-likelihood needs.` },
      { q: R`You halve the block length (from years to half-years) on a fixed record. What happens to the bias and the variance of $\hat\xi$, qualitatively?`,
        a: R`Twice as many maxima reduce the variance (roughly halving it). But each maximum is now taken over $n/2$ observations, further from the asymptotic regime, so the GEV approximation is poorer and the bias of $\hat\xi$ (relative to the limiting shape) typically grows. Seasonality may also make half-year maxima non-identically distributed.` },
      { q: R`Why is the L-moment shape estimate a function of $\hat\tau_3$ alone?`,
        a: R`L-moment ratios are invariant to location and scale: $\tau_3 = \lambda_3/\lambda_2$ is unchanged by $Z \mapsto a + bZ$ ($b>0$). For the GEV, $\mu$ and $\sigma$ are location and scale, so $\tau_3$ is a function of $\xi$ only — $2(3^\xi-1)/(2^\xi-1) - 3$ — and inverting it gives $\hat\xi$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'return-levels',
    domain: 'evt',
    order: 50,
    title: 'Return Levels & Return Periods',
    subtitle: 'The quantities engineers design for, their uncertainty, and how they are misread',
    level: 2,
    sources: { other: 'Coles (2001) §3.3–3.4, §4.4; Rootzén & Katz (2013), Water Resour. Res. 49; Langbein (1949)' },
    prereqs: ['block-maxima', 'delta-method'],
    related: ['likelihood-ratio-tests', 'confidence-intervals', 'peaks-over-threshold', 'parametric-bootstrap'],
    tags: ['return level', 'return period', 'profile likelihood', 'delta method', 'design life'],
    summary: R`The $T$-year return level $z_p$, with $p = 1/T$, is the level exceeded by the annual maximum with probability $p$ each year — the $(1-p)$ quantile of the fitted GEV. It is what a dike, dam or reserve is designed to. Its point estimate is easy; its uncertainty is not, because $z_p$ depends nonlinearly and steeply on the poorly-determined shape $\xi$. Delta-method intervals are symmetric and too optimistic on the upper side; profile likelihood intervals capture the strong right skew. And the phrase "100-year event" is routinely misread: such an event has a 63% chance of occurring at least once in any 100-year window, and a plug-in estimate of it is exceeded more often than 1% of years.`,
    survey: {
      task: 'Estimate high quantiles of the annual-maximum distribution and quantify their uncertainty',
      assumptions: R`Fitted GEV (or POT) model valid out to the return period; stationarity`,
      output: R`$\hat z_p$, delta-method SE, profile-likelihood CI, return-level plot`,
      strengths: 'Directly answers design questions; profile CIs are asymmetric and transformation-respecting',
      weaknesses: R`Extrapolation error grows rapidly with $T$; ill-defined under non-stationarity`,
      complexity: 'Profile CI: one 2-D optimisation per grid point',
    },
    sections: [
      {
        title: 'Definitions and what they actually mean',
        html: R`
<div class="def" data-name="Return level and return period">
<p>If annual maxima are $\text{GEV}(\mu,\sigma,\xi)$, the <em>return level</em> associated with <em>return period</em> $T = 1/p$ years is the level $z_p$ with $G(z_p) = 1 - p$:
$$z_p = \begin{cases}\mu - \dfrac{\sigma}{\xi}\Big[1 - y_p^{-\xi}\Big], & \xi\ne0,\\[2mm] \mu - \sigma\log y_p, & \xi=0,\end{cases}\qquad y_p = -\log(1-p).$$</p>
</div>
<p>Three interpretations, all correct under stationarity and independence across years:</p>
<ol>
<li><strong>Annual exceedance probability.</strong> In any given year, $\P(Z > z_p) = p$.</li>
<li><strong>Mean waiting time.</strong> The number of years $W$ until the first exceedance is geometric with success probability $p$, so $\E W = 1/p = T$.</li>
<li><strong>Long-run frequency.</strong> Over $L$ years the expected number of exceedances is $L/T$.</li>
</ol>
<p>What it does <em>not</em> mean is "happens once every $T$ years". The probability of at least one exceedance in a window of $L$ years is
$$\P(\text{at least one in } L) = 1 - (1-1/T)^L \approx 1 - e^{-L/T}.$$
For $L = T = 100$ this is $0.634$; the probability of <em>two or more</em> 100-year events in a century is $0.264$. Conversely, to keep the risk over a design life $L$ below $r$ one needs $T = 1/\{1 - (1-r)^{1/L}\} \approx L/r$. The well-known "475-year earthquake" of seismic codes is exactly this calculation for $r = 10\%$ in $L = 50$ years.</p>
<div class="note" data-name="Annual-maximum versus partial-duration return periods">
<p>If exceedances of a level $x$ occur as a Poisson process with rate $1/T_{\text{PD}}$ per year (the peaks-over-threshold or "partial duration" view), then the annual maximum exceeds $x$ with probability $1 - e^{-1/T_{\text{PD}}}$, so $T_{\text{AM}} = 1/(1 - e^{-1/T_{\text{PD}}})$ (Langbein, 1949). The two agree to within half a year for $T \gtrsim 10$ ($T_{\text{AM}}\approx T_{\text{PD}} + 1/2$) but differ substantially for frequent events: $T_{\text{PD}} = 1$ corresponds to $T_{\text{AM}}\approx 1.58$.</p>
</div>`,
      },
      {
        title: 'The return-level plot',
        html: R`
<p>Plot $z_p$ against $\log y_p$, or equivalently against the return period on a logarithmic axis (for large $T$, $-\log y_p \approx \log T$). Since $z_p = \mu - \sigma\log y_p$ in the Gumbel case, the curve is a straight line when $\xi = 0$; it is convex and unbounded when $\xi>0$; and it is concave with asymptote $z^+ = \mu - \sigma/\xi$ when $\xi<0$. Overlaying the empirical points $(T_i, z_{(i)})$ with plotting positions $T_i = 1/(1 - i/(m+1))$ gives a visual check of the fit precisely where the extrapolation starts.</p>
<p>In the lab, the shaded band is the pointwise 95% delta-method interval; the vertical bar at the chosen $T$ is the 95% profile-likelihood interval. Increase $T$ beyond the record length and watch the two diverge: the profile interval stretches upward, the delta interval stays symmetric.</p>`,
        lab: 'return-levels-plot',
      },
      {
        title: 'Delta-method standard errors',
        html: R`
<p>With $\hat\btheta = (\hat\mu,\hat\sigma,\hat\xi)$ asymptotically normal with covariance $V$ (inverse observed information, valid when $\xi>-1/2$), the <a href="#/t/delta-method">delta method</a> gives
$$\Var(\hat z_p) \approx \nabla z_p^\top\, V\, \nabla z_p,$$
$$\nabla z_p = \Big(\frac{\partial z_p}{\partial\mu},\frac{\partial z_p}{\partial\sigma},\frac{\partial z_p}{\partial\xi}\Big) = \Big(1,\ -\frac{1 - y_p^{-\xi}}{\xi},\ \frac{\sigma(1 - y_p^{-\xi})}{\xi^2} - \frac{\sigma}{\xi}\,y_p^{-\xi}\log y_p\Big).$$</p>
<details class="proof"><summary>Derivation of the ξ-component</summary>
<p>Write $z_p = \mu - \sigma\xi^{-1}(1-y_p^{-\xi})$. Using $\frac{\d}{\d\xi}y_p^{-\xi} = -y_p^{-\xi}\log y_p$, the product rule gives
$\partial z_p/\partial\xi = \sigma\xi^{-2}(1-y_p^{-\xi}) - \sigma\xi^{-1}\,y_p^{-\xi}\log y_p$. As $\xi\to0$ this tends to $+\tfrac{\sigma}{2}(\log y_p)^2$ (expand $y_p^{-\xi} = e^{-\xi\log y_p}$ to second order): raising $\xi$ always raises the return level.</p>
</details>
<p>For large $T$, $y_p \approx 1/T$ is small and $\log y_p \approx -\log T$ is large in magnitude, so the $\xi$-component dominates: for $\xi\approx 0$ it grows like $\tfrac{\sigma}{2}(\log T)^2$. Since $\Var(\hat\xi)$ is typically the largest relative variance, uncertainty about the shape is the main source of uncertainty about long return levels.</p>
<div class="warn" data-name="Why the symmetric interval is wrong">
<p>$\hat z_p \pm 1.96\,\widehat{\se}$ relies on $\hat z_p$ being approximately normal. But $z_p$ is a convex, rapidly growing function of $\xi$: an error of $+0.1$ in $\hat\xi$ raises $\hat z_{0.001}$ much more than an error of $-0.1$ lowers it. The sampling distribution of $\hat z_p$ is therefore right-skewed, and the symmetric interval understates the upper risk — the side that matters for design.</p>
</div>`,
      },
      {
        title: 'Profile likelihood intervals',
        html: R`
<p>The better approach treats $z_p$ as a parameter. Re-express $\mu$ in terms of $(z_p,\sigma,\xi)$:
$$\mu = z_p + \frac{\sigma}{\xi}\Big[1 - y_p^{-\xi}\Big]\qquad(\mu = z_p + \sigma\log y_p \text{ if } \xi=0),$$
and define the <em>profile log-likelihood</em>
$$\ell_p(z) = \max_{\sigma,\,\xi}\ \ell\big(\mu(z,\sigma,\xi), \sigma, \xi\big).$$</p>
<div class="thm" data-kind="Theorem" data-name="Profile likelihood ratio (Wilks)">
<p>Under the regularity conditions of the MLE ($\xi > -1/2$), at the true $z_p$,
$$D(z_p) = 2\big\{\ell(\hat\btheta) - \ell_p(z_p)\big\} \dto \chi^2_1.$$
Hence $\{z : D(z) \le \chi^2_{1,1-\alpha}\}$ is an asymptotic $(1-\alpha)$ confidence interval ($\chi^2_{1,0.95} = 3.841$).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Reparametrise $\btheta \mapsto (z_p, \sigma, \xi)$, a smooth bijection on the parameter space, so the MLE and its asymptotic normality carry over. The profile likelihood ratio for a scalar interest parameter in the presence of nuisance parameters is the standard <a href="#/t/likelihood-ratio-tests">likelihood ratio statistic</a> for $H_0: z_p = z$, whose null distribution is $\chi^2_1$ by a second-order expansion of $\ell$ around $\hat\btheta$ and the asymptotic normality of the score.</p>
</details>
<p>Two virtues: the interval is not forced to be symmetric, and it is invariant under reparametrisation (the interval for $\log z_p$ is the log of the interval for $z_p$). In practice the profile interval is markedly asymmetric for long return periods, with a long upper arm, reflecting exactly the convexity in $\xi$ described above. Computation: on a grid of $z$ values, maximise over $(\sigma,\xi)$ (warm-starting from the neighbouring grid point), and read off where $D$ crosses $3.841$.</p>`,
      },
      {
        title: 'The plug-in level is exceeded too often',
        html: R`
<p>Suppose you estimate $\hat z_p$ and build to that level. What is the probability that next year's maximum exceeds it? It is $1 - G(\hat z_p)$, a random quantity because $\hat z_p$ is. Its expectation is not $p$. Since $1-G$ is convex in the upper tail, Jensen's inequality pushes $\E[1 - G(\hat z_p)]$ above $1 - G(\E\hat z_p)$; and the right skew of $\hat z_p$ means many estimates fall below the truth. The net effect, for typical record lengths, is that the realised exceedance probability of a plug-in $T$-year level is noticeably larger than $1/T$.</p>
<p>The quantity a designer actually cares about is the <em>predictive</em> probability $\P(Z_{\text{new}} > z \mid \text{data})$, which integrates over parameter uncertainty. A Bayesian posterior predictive distribution does this directly; frequentist calibration (choosing the design level so that the realised exceedance probability is $1/T$ on average) can be done by parametric bootstrap.</p>
<p>The lab repeats the experiment: generate an $m$-year record from a known GEV, estimate the $T$-year level, and compute the true probability that it is exceeded. Compare the average realised probability with $1/T$, and the realised risk over a design life with the nominal risk.</p>`,
        lab: 'return-levels-risk',
      },
      {
        title: 'Non-stationarity and other caveats',
        html: R`
<ul>
<li><strong>Non-stationarity.</strong> If $\mu$ drifts (sea-level rise, urbanisation of a catchment), "the 100-year level" is not a single number. Rootzén &amp; Katz (2013) propose design-life levels: the level $z$ with $\P(\max_{t\in\text{design life}} Z_t \le z) = 1-r$ under the fitted non-stationary model.</li>
<li><strong>Extrapolation range.</strong> A 1000-year level from 40 years of data is an extrapolation by a factor of 25 in return period. The interval will be wide; that is the honest answer, not a defect.</li>
<li><strong>Return periods of joint events.</strong> "Both rivers exceed their 100-year levels" has a return period anywhere between 100 years (complete dependence) and 10,000 years (independence); see <a href="#/t/multivariate-extremes">multivariate extremes</a>.</li>
<li><strong>Return levels from POT.</strong> With a <a href="#/t/peaks-over-threshold">GPD</a> fit the $N$-year level is $x_N = u + \frac{\sigma}{\xi}[(N n_y\zeta_u)^\xi - 1]$, with $n_y$ observations per year and $\zeta_u = \P(X>u)$; the same delta-method and profile machinery apply.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`What is the probability that a 50-year flood occurs at least once in the next 10 years? At least once in 50 years?`,
        a: R`$1 - (1 - 1/50)^{10} = 1 - 0.98^{10} \approx 0.183$; $1 - 0.98^{50} \approx 0.636$.` },
      { q: R`Why does the uncertainty of $\hat z_p$ grow so fast with $T$, and which parameter is responsible?`,
        a: R`The gradient component $\partial z_p/\partial\xi$ grows like $\tfrac{\sigma}{2}(\log T)^2$ near $\xi=0$ (and like $T^{\xi}\log T$ for $\xi>0$), while the $\mu$ and $\sigma$ components grow at most like $\log T$. $\Var(\hat\xi)$ is also relatively large. So the shape parameter dominates long-range uncertainty.` },
      { q: R`Give two reasons to prefer profile-likelihood over delta-method intervals for return levels.`,
        a: R`(1) The profile interval can be asymmetric, capturing the right skew of $\hat z_p$ caused by convexity in $\xi$; the delta interval is symmetric by construction and understates the upper limit. (2) It is invariant to reparametrisation, whereas delta intervals depend on the scale (e.g. $z_p$ versus $\log z_p$).` },
      { q: R`A seismic code specifies a ground motion with 2% probability of exceedance in 50 years. What return period is this?`,
        a: R`$T = 1/\{1 - 0.98^{1/50}\} \approx 2475$ years — the "2475-year" motion. The approximation $T \approx L/r = 50/0.02 = 2500$ is close.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'peaks-over-threshold',
    domain: 'evt',
    order: 60,
    title: 'Peaks Over Threshold & the Generalized Pareto',
    subtitle: 'Excesses over a high threshold are asymptotically generalised Pareto, whatever the parent',
    level: 2,
    sources: { other: 'Pickands (1975); Balkema & de Haan (1974); Davison & Smith (1990), JRSS-B 52; Coles (2001) Ch. 4; Embrechts et al. (1997) §6.5' },
    prereqs: ['gev-distribution', 'mle'],
    related: ['threshold-selection', 'point-process-evt', 'extremal-index', 'poisson-process'],
    tags: ['GPD', 'threshold exceedances', 'Pickands–Balkema–de Haan', 'declustering'],
    summary: R`Instead of one maximum per block, use every observation above a high threshold $u$. The Pickands–Balkema–de Haan theorem says that, for any parent in the domain of attraction of a GEV with shape $\xi$, the excess $X - u$ given $X>u$ is approximately generalised Pareto with the same $\xi$ and a scale $\sigma_u$ that grows linearly in $u$. Combined with a Poisson model for the number of exceedances, this is equivalent to the GEV for annual maxima, but typically uses several times as much data. The GPD is threshold-stable — exceedances of a higher threshold are again GPD with the same shape — which underpins both the diagnostics for choosing $u$ and the method's internal consistency. Clustering of exceedances in time series must be handled by declustering.`,
    survey: {
      task: R`Model the tail $\P(X>x)$ for $x$ above a high threshold; estimate extreme quantiles`,
      assumptions: R`$F\in\mathcal D(G_\xi)$; $u$ high enough; exceedances (or cluster maxima) independent`,
      output: R`$(\hat\sigma_u,\hat\xi)$, $\hat\zeta_u$; tail estimate $\hat\zeta_u(1+\hat\xi(x-u)/\hat\sigma_u)^{-1/\hat\xi}$; return levels`,
      strengths: 'Uses all large observations; natural for tail risk measures (VaR, ES); threshold-stable',
      weaknesses: R`Threshold choice (bias–variance); temporal clustering; same non-regularity as GEV for $\xi\le-1/2$`,
      complexity: R`$O(k)$ per likelihood evaluation, $k$ = number of exceedances`,
    },
    sections: [
      {
        title: 'Why thresholds',
        html: R`
<p>Block maxima discard information: if a year contains three of the five largest storms in the record, only one of them enters the analysis, while the maximum of a calm year is used even though it is not extreme. A threshold method keeps exactly the observations that are extreme. Fix a high level $u$ and study the <em>conditional excess distribution</em>
$$F_u(y) = \P(X - u \le y \mid X > u) = \frac{F(u+y) - F(u)}{\bar F(u)},\qquad 0\le y < x_F - u.$$
The tail of $F$ above $u$ is then reconstructed as $\bar F(u + y) = \bar F(u)\,\bar F_u(y)$: a rate times a conditional shape. The rate $\zeta_u = \bar F(u)$ is estimated by the fraction of exceedances; the shape requires a limit theorem.</p>`,
      },
      {
        title: 'The Pickands–Balkema–de Haan theorem',
        html: R`
<div class="def" data-name="Generalised Pareto distribution">
<p>$\text{GPD}(\sigma,\xi)$, $\sigma>0$, has distribution function
$$H(y) = 1 - \Big(1 + \frac{\xi y}{\sigma}\Big)^{-1/\xi},\qquad y > 0,\ 1 + \xi y/\sigma > 0,$$
with $H(y) = 1 - e^{-y/\sigma}$ for $\xi = 0$. The support is $(0,\infty)$ for $\xi\ge0$ and $(0, -\sigma/\xi)$ for $\xi<0$. Special cases: $\xi=0$ exponential, $\xi=-1$ uniform on $(0,\sigma)$, $\xi>0$ a shifted Pareto (Lomax). Mean $\sigma/(1-\xi)$ for $\xi<1$; variance $\sigma^2/\{(1-\xi)^2(1-2\xi)\}$ for $\xi<1/2$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Pickands (1975); Balkema & de Haan (1974)">
<p>$F \in \mathcal D(G_\xi)$ if and only if there is a positive function $\sigma(u)$ such that
$$\lim_{u\uparrow x_F}\ \sup_{0\le y < x_F - u}\ \big|F_u(y) - H_{\sigma(u),\xi}(y)\big| = 0.$$</p>
</div>
<details class="proof"><summary>Heuristic derivation from the GEV (Coles, 2001, §4.2)</summary>
<p>If $F^n \approx \text{GEV}(\mu,\sigma,\xi)$ for large $n$, then $n\log F(z) \approx -[1+\xi(z-\mu)/\sigma]^{-1/\xi}$. For $z$ in the upper tail, $\log F(z) \approx -\bar F(z)$, so
$$\bar F(z) \approx \frac1n\Big[1 + \xi\frac{z-\mu}{\sigma}\Big]^{-1/\xi}.$$
Take the ratio at $z = u+y$ and $z = u$:
$$\bar F_u(y) = \frac{\bar F(u+y)}{\bar F(u)} \approx \left[\frac{1 + \xi(u + y - \mu)/\sigma}{1 + \xi(u-\mu)/\sigma}\right]^{-1/\xi} = \Big[1 + \frac{\xi y}{\tilde\sigma}\Big]^{-1/\xi},\qquad \tilde\sigma = \sigma + \xi(u - \mu).$$
The rigorous version is the unified domain-of-attraction condition $\bar F(t + x\,a(t))/\bar F(t) \to (1+\xi x)^{-1/\xi}$ from <a href="#/t/gev-distribution">the GEV topic</a>, with $\sigma(u) = a(u)$, upgraded to uniform convergence because the limit is continuous and monotone.</p>
</details>
<p>The shape $\xi$ is the same as in the GEV for block maxima of the same parent: the two approaches estimate the same tail index. The scale $\tilde\sigma$ depends on $u$ — linearly, as the next section shows — which is the key to both checking the model and choosing $u$.</p>`,
      },
      {
        title: 'Threshold stability and the mean excess function',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Threshold stability">
<p>If excesses over $u_0$ are exactly $\text{GPD}(\sigma_{u_0},\xi)$, then for every $u > u_0$ (inside the support) the excesses over $u$ are $\text{GPD}(\sigma_u,\xi)$ with
$$\sigma_u = \sigma_{u_0} + \xi(u - u_0).$$
In particular the shape is constant in $u$, and the <em>modified scale</em> $\sigma^* = \sigma_u - \xi u$ is constant in $u$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $d = u - u_0$. For $y > 0$,
$$\P(X - u > y\mid X>u) = \frac{\bar H_{u_0}(d+y)}{\bar H_{u_0}(d)} = \left[\frac{1+\xi(d+y)/\sigma_{u_0}}{1+\xi d/\sigma_{u_0}}\right]^{-1/\xi} = \Big[1 + \frac{\xi y}{\sigma_{u_0} + \xi d}\Big]^{-1/\xi}.$$
(For $\xi=0$ the ratio is $e^{-y/\sigma_{u_0}}$: memorylessness of the exponential.)</p>
</details>
<p>The GPD is the only family with this property (it is to excesses what max-stability is to maxima). Its consequence for the mean excess function $e(u) = \E[X - u\mid X>u]$ is immediate: for $\xi<1$,
$$e(u) = \frac{\sigma_u}{1-\xi} = \frac{\sigma_{u_0} + \xi(u-u_0)}{1-\xi},\qquad u>u_0,$$
a <strong>linear</strong> function of $u$ with slope $\xi/(1-\xi)$. The sample version, the mean residual life plot, is one of the tools of <a href="#/t/threshold-selection">threshold selection</a>. An upward slope suggests $\xi>0$, flat suggests $\xi=0$, downward $\xi<0$.</p>`,
      },
      {
        title: 'Fitting: the POT likelihood and tail estimates',
        html: R`
<p>Given exceedances $x_{(1)},\dots,x_{(k)}$ of $u$ among $n$ observations, with excesses $y_j = x_j - u$, the GPD log-likelihood is
$$\ell(\sigma,\xi) = -k\log\sigma - \Big(1+\frac1\xi\Big)\sum_{j=1}^k\log\Big(1+\frac{\xi y_j}{\sigma}\Big),\qquad 1+\xi y_j/\sigma > 0\ \forall j,$$
and $-k\log\sigma - \sum_j y_j/\sigma$ when $\xi = 0$. The regularity story is the same as for the GEV (Smith, 1985): standard asymptotics for $\xi > -1/2$. In that range the inverse expected information per exceedance is
$$k\,\Var\begin{pmatrix}\hat\sigma\\ \hat\xi\end{pmatrix} \approx (1+\xi)\begin{pmatrix} 2\sigma^2 & -\sigma\\ -\sigma & 1+\xi\end{pmatrix},$$
so $\se(\hat\xi)\approx (1+\xi)/\sqrt k$ and the estimates are negatively correlated. The exceedance rate is estimated by $\hat\zeta_u = k/n$, asymptotically independent of $(\hat\sigma,\hat\xi)$, with variance $\zeta_u(1-\zeta_u)/n$.</p>
<p>The fitted tail is
$$\widehat{\P}(X > x) = \hat\zeta_u\Big[1 + \hat\xi\,\frac{x-u}{\hat\sigma}\Big]^{-1/\hat\xi},\qquad x>u,$$
and inverting it gives the level exceeded on average once every $m$ observations,
$$x_m = u + \frac{\sigma}{\xi}\Big[(m\zeta_u)^{\xi} - 1\Big]\qquad(x_m = u + \sigma\log(m\zeta_u) \text{ if } \xi = 0),$$
valid when $m\zeta_u > 1$. With $n_y$ observations per year, the $N$-year return level uses $m = Nn_y$. In finance the same formula with $m = 1/(1-q)$ is the POT estimate of value-at-risk at level $q$.</p>
<p>In the lab, move the threshold and watch three things: the number of exceedances $k$, the stability of $\hat\xi$, and the estimate of a far-tail quantile compared with the truth (known because the data are simulated). The tail plot shows the fitted GPD tail extended beyond the data on a log scale.</p>`,
        lab: 'peaks-over-threshold-fit',
      },
      {
        title: 'The Poisson–GPD model and its equivalence with the GEV',
        html: R`
<p>The threshold approach and the block maxima approach are two views of one model. Model the number of exceedances of $u$ in a year as Poisson($\lambda$) and their sizes as IID GPD, independent of the count.</p>
<div class="thm" data-kind="Proposition" data-name="Poisson–GPD implies GEV maxima">
<p>Let $N\sim\text{Poisson}(\lambda)$ and, given $N$, let $Y_1,\dots,Y_N \iid \text{GPD}(\sigma_u,\xi)$. Then $Z = u + \max_{j\le N} Y_j$ (with $Z \le u$ if $N=0$) satisfies, for $z > u$,
$$\P(Z\le z) = \exp\Big\{-\lambda\Big[1+\xi\frac{z-u}{\sigma_u}\Big]^{-1/\xi}\Big\},$$
the GEV distribution function with $\sigma = \sigma_u\lambda^{\xi}$ and $\mu = u - \frac{\sigma_u}{\xi}(1 - \lambda^{\xi})$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Conditioning on $N$: $\P(Z\le z) = \sum_{k\ge0} e^{-\lambda}\frac{\lambda^k}{k!}H(z-u)^k = \exp\{-\lambda(1 - H(z-u))\} = \exp\{-\lambda\bar H(z-u)\}$, and $\bar H(z-u) = [1+\xi(z-u)/\sigma_u]^{-1/\xi}$. Writing $\lambda[\cdots]^{-1/\xi} = [\lambda^{-\xi}(1 + \xi(z-u)/\sigma_u)]^{-1/\xi}$ and matching $\lambda^{-\xi}(1+\xi(z-u)/\sigma_u) = 1 + \xi(z-\mu)/\sigma$ coefficient by coefficient gives $\sigma = \sigma_u\lambda^\xi$ and $\mu - u = \sigma_u(\lambda^\xi - 1)/\xi$.</p>
</details>
<p>So a POT fit implies an annual-maximum GEV with the same $\xi$; and conversely a GEV for annual maxima implies a GPD for excesses above any threshold in its support with $\sigma_u = \sigma + \xi(u-\mu)$. The two methods should give compatible answers; when they do not, either the threshold is too low or the blocks too short. This unification is made fully explicit by the <a href="#/t/point-process-evt">point-process view</a>.</p>`,
      },
      {
        title: 'Dependence and declustering',
        html: R`
<p>In a time series, extremes cluster: a storm raises wave heights for several consecutive hours, a market crash produces several large losses in a row. Exceedances are then not independent, and treating them as such has two consequences. The marginal tail estimate $(\hat\sigma,\hat\xi)$ is still roughly right (the GPD describes the marginal excess distribution), but its standard errors are too small because the effective number of independent observations is less than $k$. And the rate of extreme <em>events</em> is overstated, since one event produces several exceedances.</p>
<div class="algo" data-name="Runs declustering">
<ol>
<li>Choose a threshold $u$ and a run length $r$.</li>
<li>Scan the series; a cluster begins at an exceedance and ends when $r$ consecutive observations fall below $u$.</li>
<li>Keep the maximum of each cluster and fit the GPD to the cluster-maximum excesses.</li>
<li>Estimate the rate of clusters $n_c/n$; the ratio $n_c/k$ estimates the <a href="#/t/extremal-index">extremal index</a> $\theta$.</li>
</ol>
</div>
<p>Return levels then use the cluster rate, $x_m = u + \frac{\sigma}{\xi}[(m\zeta_u\theta)^\xi - 1]$. The answer depends on $r$, so check sensitivity; alternatively keep all exceedances for the marginal fit and use a <a href="#/t/block-bootstrap">block bootstrap</a> to get standard errors that respect dependence.</p>
<div class="warn" data-name="Pitfalls">
<ul>
<li>Threshold too low: the GPD is not yet a good approximation and $\hat\xi$ is biased (for normal or lognormal data, typically biased toward the penultimate value). Too high: few exceedances, huge variance. See <a href="#/t/threshold-selection">threshold selection</a>.</li>
<li>Seasonality: a single threshold may be extreme in winter and ordinary in summer; consider seasonal thresholds or covariates in $\sigma$.</li>
<li>Rounded data produce ties among small excesses, which can distort the fit near $u$.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`If the excesses over $u_0 = 10$ are $\text{GPD}(\sigma=4,\xi=0.2)$, what is the distribution of excesses over $u = 20$, and what is the mean excess at $u = 20$?`,
        a: R`Threshold stability gives $\text{GPD}(\sigma_{20}, 0.2)$ with $\sigma_{20} = 4 + 0.2\times10 = 6$. Mean excess $= \sigma_{20}/(1-\xi) = 6/0.8 = 7.5$.` },
      { q: R`Why is the GPD shape for excesses the same as the GEV shape for block maxima of the same parent?`,
        a: R`Both arise from the same domain-of-attraction condition $\bar F(t + x\,a(t))/\bar F(t)\to(1+\xi x)^{-1/\xi}$: the GEV via the Poisson lemma ($F^n(b_n + a_n x)\to\exp\{-(1+\xi x)^{-1/\xi}\}$), the GPD by reading the same ratio as a conditional survival function. Equivalently, a Poisson number of GPD excesses has a GEV maximum with the same $\xi$.` },
      { q: R`Derive the POT estimate of the level exceeded once every $m$ observations.`,
        a: R`Solve $\zeta_u[1+\xi(x-u)/\sigma]^{-1/\xi} = 1/m$: $1 + \xi(x-u)/\sigma = (m\zeta_u)^{\xi}$, so $x_m = u + \frac\sigma\xi[(m\zeta_u)^\xi - 1]$. For $\xi=0$, $\zeta_u e^{-(x-u)/\sigma} = 1/m$ gives $u + \sigma\log(m\zeta_u)$.` },
      { q: R`Hourly wave heights exceed a threshold 600 times in 10 years, in 150 distinct storms. How should this affect a 100-year return level calculation?`,
        a: R`The extremal index is about $150/600 = 0.25$: exceedances come in clusters of mean size 4. Using all 600 as independent events overstates the event rate by a factor of about 4 and understates standard errors. Decluster (fit to storm maxima, rate = 15 per year) or include $\theta$ in the return level formula, and use cluster-aware uncertainty (e.g. block bootstrap).` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'threshold-selection',
    domain: 'evt',
    order: 70,
    title: 'Threshold Selection: Mean Residual Life & Stability Plots',
    subtitle: 'Choosing where the tail begins, a bias–variance problem with graphical diagnostics',
    level: 2,
    sources: { other: 'Coles (2001) §4.3; Davison & Smith (1990); Scarrott & MacDonald (2012), REVSTAT 10; Northrop & Coleman (2014), Extremes 17' },
    prereqs: ['peaks-over-threshold'],
    related: ['bias-variance', 'tail-index', 'model-selection'],
    tags: ['threshold', 'mean residual life', 'parameter stability', 'bias–variance'],
    summary: R`The GPD is an asymptotic model for excesses over a threshold, so the threshold $u$ must be high enough for the approximation to hold, yet low enough to leave data to fit. Two diagnostics exploit threshold stability. If the GPD holds above $u_0$, the mean excess $\E[X-u\mid X>u]$ is linear in $u$ for $u>u_0$ (mean residual life plot), and the fitted shape $\hat\xi_u$ and modified scale $\hat\sigma_u - \hat\xi_u u$ are constant for $u>u_0$ (parameter stability plots). One chooses the lowest threshold above which these look linear or constant within sampling error. The underlying trade-off is a classical bias–variance problem whose optimum depends on second-order tail behaviour that is hard to estimate, which is why judgement and sensitivity analysis remain part of the method.`,
    survey: {
      task: R`Choose the threshold $u$ (equivalently the number of exceedances $k$) for a POT analysis`,
      assumptions: R`GPD holds above some unknown $u_0$; IID or declustered exceedances`,
      output: R`MRL plot with CIs; $\hat\xi_u$, $\hat\sigma^*_u$ against $u$; a chosen $u$ and sensitivity range`,
      strengths: 'Model-based diagnostics with clear theoretical targets (linearity, constancy)',
      weaknesses: 'Subjective; plots are noisy and serially correlated across u; MRL invalid if ξ ≥ 1/2',
      complexity: R`One GPD fit per candidate threshold`,
    },
    sections: [
      {
        title: 'The threshold as a bias–variance dial',
        html: R`
<p>Let $k$ be the number of exceedances of $u$ in a sample of size $n$. The <a href="#/t/peaks-over-threshold">Pickands–Balkema–de Haan theorem</a> is a limit as $u\uparrow x_F$, i.e. $k/n\to0$; to estimate anything we also need $k\to\infty$. At any finite threshold the excess distribution differs from a GPD, and estimates of $\xi$ inherit a <strong>bias</strong> that shrinks as $u$ rises. Their <strong>variance</strong> is roughly $(1+\xi)^2/k$ and grows as $u$ rises. The mean squared error
$$\MSE(\hat\xi_u) \approx \text{bias}(u)^2 + \frac{(1+\xi)^2}{k(u)}$$
is minimised at an intermediate threshold — exactly the <a href="#/t/bias-variance">bias–variance</a> trade-off of smoothing, with $k$ playing the role of a bandwidth.</p>
<p>The bias is governed by <em>second-order</em> tail behaviour: how fast $\bar F(t + x a(t))/\bar F(t)$ approaches its GPD limit. If that rate is a power $t^{\rho}$ of the "return period" $t = n/k$ with $\rho<0$, then $\text{bias} \propto (n/k)^{\rho}$ and the MSE-optimal number of exceedances grows like
$$k_{\text{opt}} \propto n^{-2\rho/(1-2\rho)},$$
a vanishing fraction of the sample. When $|\rho|$ is large (e.g. the exponential, or the exact Pareto) high bias is never a concern and low thresholds are fine; when $\rho$ is close to zero (normal, lognormal: the convergence is logarithmic) every practical threshold carries bias. Since $\rho$ is harder to estimate than $\xi$, plug-in optimal thresholds are unstable, and graphical diagnostics remain standard.</p>`,
      },
      {
        title: 'The mean residual life plot',
        html: R`
<p>By threshold stability, if the GPD holds above $u_0$ with $\xi<1$, then $e(u) = \E[X-u\mid X>u] = (\sigma_{u_0} + \xi(u-u_0))/(1-\xi)$ is linear in $u$ for $u>u_0$. The empirical version
$$\hat e(u) = \frac{1}{k(u)}\sum_{i:\,x_i>u}(x_i - u)$$
plotted against $u$, with pointwise intervals $\hat e(u) \pm 1.96\,s_u/\sqrt{k(u)}$ ($s_u$ the standard deviation of the excesses), is the <em>mean residual life</em> (MRL) or mean excess plot (Davison &amp; Smith, 1990). Choose $u_0$ as the lowest threshold above which the plot is consistent with a straight line.</p>
<table>
<thead><tr><th>Parent</th><th>Theoretical $e(u)$ for large $u$</th><th>MRL shape</th></tr></thead>
<tbody>
<tr><td>Exponential / GPD $\xi=0$</td><td>constant $\sigma$</td><td>flat</td></tr>
<tr><td>Pareto / GPD $\xi\in(0,1)$</td><td>$\propto u$, slope $\xi/(1-\xi)$</td><td>rising line</td></tr>
<tr><td>Uniform / GPD $\xi<0$</td><td>linear, decreasing to $0$ at $x_F$</td><td>falling line</td></tr>
<tr><td>Normal</td><td>$\sim 1/u$</td><td>slowly falling, curved</td></tr>
<tr><td>Lognormal</td><td>$\sim u/\log u$</td><td>rising, slightly concave</td></tr>
</tbody>
</table>
<div class="warn" data-name="Limitations of the MRL plot">
<ul>
<li>The right end is based on very few exceedances and wanders wildly; ignore it.</li>
<li>For $\xi\ge1/2$ the excesses have infinite variance, so the intervals are meaningless; for $\xi\ge1$ the mean excess is infinite and the plot estimates nothing.</li>
<li>Linearity is a necessary consequence of the GPD, not a sufficient one. Curvature (lognormal) can be invisible at the resolution the noise permits.</li>
</ul>
</div>`,
      },
      {
        title: 'Parameter stability plots',
        html: R`
<p>The second consequence of threshold stability is that, above $u_0$, $\xi$ does not depend on $u$ and neither does the modified scale
$$\sigma^* = \sigma_u - \xi u.$$
So fit the GPD at a grid of thresholds and plot $\hat\xi_u$ and $\hat\sigma^*_u = \hat\sigma_u - \hat\xi_u u$ against $u$, with intervals. The variance of $\hat\sigma^*_u$ follows from the <a href="#/t/delta-method">delta method</a>:
$$\Var(\hat\sigma^*_u) \approx \Var(\hat\sigma_u) - 2u\,\Cov(\hat\sigma_u,\hat\xi_u) + u^2\Var(\hat\xi_u).$$
Choose the lowest $u$ above which both estimates are constant within sampling error.</p>
<div class="note" data-name="Reading stability plots honestly">
<p>Estimates at neighbouring thresholds use nested data — the exceedances of a higher threshold are a subset of those of a lower one — so they are strongly positively correlated. Wiggles therefore persist across many thresholds and are easily mistaken for structure. The pointwise intervals are not simultaneous bands. Look for a sustained trend in the lower part of the plot (bias) giving way to a plateau whose fluctuations are comparable to the interval widths.</p>
</div>
<p>In the lab, the left panel is the MRL plot and the right panel a stability plot; the vertical line is the threshold whose GPD fit is summarised in the tiles. Try the GPD parent (stable everywhere), the Student $t$ (stable above a moderate threshold), and the lognormal, whose fitted shape is clearly positive and drifts slowly downward — a Gumbel-domain distribution impersonating a heavy tail.</p>`,
        lab: 'threshold-selection-plots',
      },
      {
        title: 'Worked example: the lognormal trap',
        html: R`
<p>Lognormal data are in the Gumbel domain ($\xi=0$), yet GPD fits to them almost always return $\hat\xi>0$. This is not a failure of the estimator; it is the penultimate shape. With $t = \log x$ and $m(t) = \bar\Phi(t)/\phi(t)$ the Mills ratio, the reciprocal hazard of the standard lognormal is $r(x) = x\,m(t)$, and
$$r'(x) = m(t) + m'(t) = (1+t)\,m(t) - 1,$$
using $m'(t) = t\,m(t) - 1$. Since $m(t) = 1/t - 1/t^3 + O(t^{-5})$, $r'(x) = 1/t + O(t^{-2})$: the local shape decays like $1/\log x$.</p>
<div class="ex" data-name="Numbers">
<p>At the $99.87\%$ quantile, $t = 3$: $m(3) = 0.00135/0.00443 \approx 0.305$ and $r' \approx 4(0.305) - 1 \approx 0.22$. So at thresholds anyone would use, the excesses of a lognormal look like a GPD with $\xi\approx0.2$–$0.3$. The local shape reaches $0.1$ only around $t\approx 9$, a quantile of order $1 - 10^{-19}$.</p>
</div>
<p>Two lessons. For extrapolation <em>moderately</em> beyond the threshold, the fitted positive $\hat\xi$ is the right model locally — that is exactly the penultimate approximation, and a Gumbel fit would understate the tail. For extrapolation <em>far</em> beyond the data, the positive shape overstates the tail, but nothing in the data can reveal this: the curvature that distinguishes a lognormal from a GPD lives at levels where there are no observations. This is the irreducible model uncertainty of extrapolation.</p>`,
      },
      {
        title: 'Quantifying the trade-off by simulation',
        html: R`
<p>With a known parent we can compute the bias–variance trade-off directly: repeatedly simulate a sample, fit a GPD at each of a grid of thresholds, and record the error of an extreme quantile estimate. The lab below does this for the $0.9999$ quantile — an extrapolation beyond samples of a few thousand — and plots squared bias, variance and MSE against the fraction of data used. For the exact GPD parent the bias is negligible at every threshold and the lowest threshold (largest $k$) wins. For the Student $t_4$ there is a clear interior optimum, around the top 15% of a sample of 2000, beyond which the squared bias grows rapidly. The normal also has an interior optimum at a fairly low threshold, where the squared bias is already a sizeable fraction (roughly a third) of the MSE: with logarithmically slow convergence no threshold is free of bias, so the best one can do is balance it.</p>`,
        lab: 'threshold-selection-tradeoff',
      },
      {
        title: 'Other approaches and practical advice',
        html: R`
<ul>
<li><strong>Rules of thumb.</strong> A fixed upper fraction (e.g. the top 5–10%; DuMouchel, 1983, suggested 10% for tail-index estimation) is simple and often reasonable, but ignores the data's own second-order behaviour.</li>
<li><strong>Goodness-of-fit sequences.</strong> Test the GPD at increasing thresholds (Anderson–Darling, Cramér–von Mises) and take the first that is not rejected. Multiple testing and dependence between tests make the error rates approximate.</li>
<li><strong>Score and likelihood-ratio tests of shape constancy</strong> over a piecewise-constant-$\xi$ model (Northrop &amp; Coleman, 2014), which formalise the stability plot.</li>
<li><strong>Extreme-value mixture models</strong> splice a parametric or nonparametric bulk to a GPD tail and estimate the threshold as a parameter (Scarrott &amp; MacDonald, 2012). Uncertainty about $u$ then propagates into return levels.</li>
<li><strong>Cross-validated predictive performance</strong> of the tail fit, e.g. leave-one-out scores on exceedances, offers a <a href="#/t/model-selection">model-selection</a> view of the same problem.</li>
</ul>
<div class="warn" data-name="Whatever you choose">
<p>Report results for a range of plausible thresholds. If a return level changes materially between defensible choices of $u$, that variability is part of the uncertainty and belongs in the conclusions.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`An MRL plot is approximately linear with slope $0.25$ above $u_0$. What shape does this suggest?`,
        a: R`The slope is $\xi/(1-\xi) = 0.25$, so $\xi = 0.25/1.25 = 0.2$.` },
      { q: R`Why plot the modified scale $\sigma_u - \xi u$ rather than $\sigma_u$ itself?`,
        a: R`By threshold stability $\sigma_u = \sigma_{u_0} + \xi(u-u_0)$ changes linearly with $u$ unless $\xi=0$, so non-constancy of $\hat\sigma_u$ is expected and uninformative. The combination $\sigma_u - \xi u = \sigma_{u_0} - \xi u_0$ is constant above $u_0$ whenever the GPD holds, giving a clean target.` },
      { q: R`Explain why the optimal number of exceedances satisfies $k\to\infty$ but $k/n\to0$.`,
        a: R`Variance of the shape estimate is of order $1/k$, so $k\to\infty$ is needed for consistency. The GPD is only a limit as the threshold approaches $x_F$, i.e. as the fraction $k/n$ of data used goes to zero; a fixed fraction would leave a non-vanishing bias.` },
      { q: R`Your GPD fit to insurance claims gives $\hat\xi\approx0.6$. Which threshold diagnostic should you distrust?`,
        a: R`The MRL plot's confidence intervals: with $\xi\ge1/2$ the excesses have infinite variance, so $s_u/\sqrt{k}$ does not describe the variability of $\hat e(u)$, and the empirical mean excess converges slowly and erratically. Rely on parameter stability plots instead.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'tail-index',
    domain: 'evt',
    order: 80,
    title: 'Tail Index Estimation: Hill & Pickands',
    subtitle: 'Semiparametric estimation of tail heaviness from the top order statistics',
    level: 3,
    sources: { other: 'Hill (1975), Ann. Statist. 3; Pickands (1975); Mason (1982); Dekkers, Einmahl & de Haan (1989); de Haan & Ferreira (2006) Ch. 3–4; Resnick (2007)' },
    prereqs: ['gev-distribution', 'order-statistics'],
    related: ['peaks-over-threshold', 'threshold-selection', 'mle'],
    tags: ['Hill estimator', 'Pickands estimator', 'regular variation', 'Hill plot', 'Weissman'],
    summary: R`For heavy-tailed data with $\bar F(x) = x^{-1/\gamma}L(x)$, the tail index $\gamma = \xi>0$ can be estimated from the $k$ largest observations without modelling the rest of the distribution. Hill's estimator — the average log-excess of the top $k$ values over the $(k+1)$-th — is the conditional MLE for a Pareto tail and the empirical analogue of a Karamata limit. It is consistent when $k\to\infty$ and $k/n\to0$, asymptotically normal with variance $\gamma^2/k$, and biased at a rate set by the slowly varying $L$. The Hill plot exposes the bias–variance trade-off in $k$. Pickands' estimator and the moment estimator extend to all real $\xi$, at the cost of variance; Weissman's estimator converts $\hat\gamma$ into extreme quantiles.`,
    survey: {
      task: R`Estimate $\gamma = \xi > 0$ (or any real $\xi$) and extreme quantiles from the top $k$ order statistics`,
      assumptions: R`Regularly varying tail (Hill); $F\in\mathcal D(G_\xi)$ (Pickands, moment); $k\to\infty$, $k/n\to0$`,
      output: R`$\hat\gamma_{k}$ as a function of $k$ (Hill plot); asymptotic CI $\hat\gamma \pm 1.96\hat\gamma/\sqrt k$`,
      strengths: 'Closed form, no optimisation; minimal assumptions; Hill has the smallest variance among common estimators',
      weaknesses: R`Choice of $k$; bias from $L$; Hill not location-invariant and only for $\gamma>0$`,
      complexity: R`$O(n\log n)$ for the sort, then $O(n)$ for all $k$`,
    },
    sections: [
      {
        title: 'The semiparametric tail problem',
        html: R`
<p>In the <a href="#/t/gev-distribution">Fréchet domain</a>, $\bar F(x) = x^{-1/\gamma}L(x)$ with $L$ slowly varying and $\gamma = \xi = 1/\alpha > 0$. The index $\gamma$ is the single most important number about such a distribution: it determines which moments exist, how fast maxima grow ($\asymp n^{\gamma}$), and how fast extreme quantiles grow ($\bar F^{-1}(p) \approx p^{-\gamma}$ up to slowly varying factors). Nothing about $\gamma$ depends on the bulk of $F$, so it is natural to estimate it from the largest observations only, treating $L$ as an unknown nuisance function. Write the order statistics in <em>decreasing</em> order, $X_{(1)}\ge X_{(2)}\ge\dots\ge X_{(n)}$, and let $k$ be the number of top values used.</p>`,
      },
      {
        title: 'The Hill estimator: three derivations',
        html: R`
<div class="def" data-name="Hill estimator (Hill, 1975)">
<p>$$\hat\gamma^{H}_{k} = \frac1k\sum_{i=1}^k \log X_{(i)} - \log X_{(k+1)},\qquad 1\le k < n,$$
for positive data.</p>
</div>
<p><strong>(a) Conditional maximum likelihood.</strong> If above a threshold $u$ the tail is exactly Pareto, $\P(X > x\mid X>u) = (x/u)^{-1/\gamma}$, then $\log(X/u)$ given $X>u$ is exponential with mean $\gamma$, and the MLE of $\gamma$ from the exceedances is their average log-excess. Taking the random threshold $u = X_{(k+1)}$ gives Hill's estimator.</p>
<p><strong>(b) Empirical Karamata limit.</strong> For regularly varying $\bar F$,
$$\E[\log X - \log u\mid X>u] = \int_u^\infty \frac{\bar F(x)}{\bar F(u)}\frac{\d x}{x} = \int_1^\infty \frac{\bar F(us)}{\bar F(u)}\frac{\d s}{s}\ \longrightarrow\ \int_1^\infty s^{-1/\gamma}\frac{\d s}{s} = \gamma$$
as $u\to\infty$ (dominated convergence is justified by Potter's bounds). Hill's estimator is this mean log-excess with $F$ replaced by $\hat F_n$ and $u$ by $X_{(k+1)}$.</p>
<p><strong>(c) Pareto quantile plot.</strong> Plot $\log X_{(i)}$ against $\log(n/i)$. For a regularly varying tail the top points lie near a line of slope $\gamma$, since $\log \bar F^{-1}(i/n) \approx \gamma\log(n/i) + $ slowly varying terms; Hill's estimator is (asymptotically) a slope estimate through the anchor point $X_{(k+1)}$.</p>`,
      },
      {
        title: 'Consistency, asymptotic normality and bias',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Exact Pareto case">
<p>If $X_i \iid$ Pareto with $\bar F(x) = x^{-1/\gamma}$, $x\ge1$, then $k\,\hat\gamma^{H}_{k}/\gamma \sim \text{Gamma}(k,1)$; hence $\hat\gamma^{H}_{k}$ is unbiased with variance $\gamma^2/k$ for every $k$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$E_i = \gamma^{-1}\log X_i$ are IID Exp(1). By the Rényi representation, the normalised spacings of exponential order statistics $j(E_{(j)} - E_{(j+1)})$, $j = 1,\dots,k$ (decreasing order), are IID Exp(1). Then
$$\frac{k\hat\gamma^{H}_{k}}{\gamma} = \sum_{i=1}^k (E_{(i)} - E_{(k+1)}) = \sum_{i=1}^k\sum_{j=i}^{k}(E_{(j)} - E_{(j+1)}) = \sum_{j=1}^k j\,(E_{(j)} - E_{(j+1)}),$$
a sum of $k$ IID Exp(1) variables.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="Consistency (Mason, 1982) and asymptotic normality (Hall, 1982; de Haan & Ferreira, 2006, Thm 3.2.5)">
<p>(i) If $\bar F$ is regularly varying with index $-1/\gamma$ and $k = k_n\to\infty$, $k/n\to0$, then $\hat\gamma^{H}_k \pto \gamma$.</p>
<p>(ii) Suppose in addition the second-order condition: for some $\rho<0$ and a function $A$ with $|A| \in RV_\rho$, $A(t)\to0$,
$$\frac{U(tx)/U(t) - x^{\gamma}}{A(t)} \to x^{\gamma}\,\frac{x^\rho - 1}{\rho},\qquad U(t) = F^{\leftarrow}(1 - 1/t).$$
If $\sqrt k\,A(n/k) \to \lambda \in\R$, then
$$\sqrt k\big(\hat\gamma^{H}_k - \gamma\big) \dto \N\Big(\frac{\lambda}{1-\rho},\ \gamma^2\Big).$$</p>
</div>
<p>Consistency comes from (b): the empirical mean log-excess converges to its population version at the random threshold $X_{(k+1)}\to\infty$, which needs $k/n\to0$, while the averaging needs $k\to\infty$. The second-order condition quantifies how far $L$ is from constant. In the Hall class $\bar F(x) = Cx^{-1/\gamma}(1 + Dx^{\rho/\gamma} + o(x^{\rho/\gamma}))$, $A(t)\propto t^{\rho}$, so
$$\text{bias} \approx \frac{c\,(n/k)^{\rho}}{1-\rho},\qquad \text{variance} \approx \frac{\gamma^2}{k},\qquad k_{\text{opt}} \propto n^{-2\rho/(1-2\rho)}.$$
Examples: Student $t_\nu$ has $\gamma = 1/\nu$ and $\rho = -2/\nu$, so the $t_4$ ($\rho=-1/2$) gives $k_{\text{opt}} \propto n^{1/2}$ while the Cauchy ($\rho = -2$) gives $n^{4/5}$. The exact Pareto has no bias at all, and the optimal $k$ is "as large as possible".</p>`,
      },
      {
        title: 'The Hill plot',
        html: R`
<p>The <em>Hill plot</em> graphs $\{(k, \hat\gamma^H_k)\}$ with pointwise intervals $\hat\gamma^H_k(1 \pm 1.96/\sqrt k)$. Small $k$: high variance. Large $k$: the estimator is pulled by the bulk of the distribution (bias). One looks for a stable region in between. In the lab, compare the exact Pareto (flat), the Student $t$ (stable at small $k$, then drifting), and the normal (in the Gumbel domain, $\gamma=0$ is outside Hill's scope; the plot declines slowly toward zero and never stabilises). The Pickands and moment estimators are shown for comparison.</p>
<div class="warn" data-name="Hill horror plots">
<p>When $L$ varies very slowly, the plot can be flat at the wrong value. For $\bar F(x) = 1/(x\log x)$ (so $\gamma = 1$), the bias decays like $1/\log(n/k)$ and the Hill plot sits stably around values well below $1$ over the whole visible range (Resnick, 2007, calls these "Hill horror plots"). Plotting against $\log k$ (the "alt-Hill" plot of Drees, de Haan &amp; Resnick, 2000) spreads out the informative small-$k$ region.</p>
</div>
<div class="warn" data-name="Not location invariant">
<p>Hill's estimator uses $\log X$, so adding a constant to the data changes it at every finite $k$, although not its limit. Data with a shifted origin (losses net of a deductible, say) can produce very different Hill plots. Toggle the shift in the lab to see this.</p>
</div>`,
        lab: 'tail-index-hill',
      },
      {
        title: 'Beyond ξ > 0: the Pickands and moment estimators',
        html: R`
<div class="def" data-name="Pickands estimator (Pickands, 1975)">
<p>$$\hat\xi^{P}_k = \frac{1}{\log 2}\,\log\frac{X_{(k)} - X_{(2k)}}{X_{(2k)} - X_{(4k)}},\qquad 4k\le n.$$</p>
</div>
<p>Its motivation is a quantile identity. With $U(t) = F^{\leftarrow}(1-1/t)$, the domain-of-attraction condition can be written $\{U(tx) - U(t)\}/a(t) \to (x^{\xi}-1)/\xi$. Then
$$\frac{U(4t) - U(2t)}{U(2t) - U(t)} \to \frac{(4^\xi - 2^\xi)}{(2^\xi - 1)} = 2^\xi,$$
and $X_{(k)}, X_{(2k)}, X_{(4k)}$ estimate $U(n/k)$, $U(n/2k)$, $U(n/4k)$. Pickands' estimator is consistent for every real $\xi$ when $k\to\infty$, $k/n\to0$, and it is invariant to location and scale. The price is a much larger asymptotic variance than Hill's, and visible erratic behaviour in practice.</p>
<div class="def" data-name="Moment estimator (Dekkers, Einmahl & de Haan, 1989)">
<p>With $M^{(j)}_k = \frac1k\sum_{i=1}^k\big(\log X_{(i)} - \log X_{(k+1)}\big)^j$ (so $M^{(1)}_k$ is Hill),
$$\hat\xi^{M}_k = M^{(1)}_k + 1 - \frac12\Big(1 - \frac{(M^{(1)}_k)^2}{M^{(2)}_k}\Big)^{-1}.$$</p>
</div>
<p>It corrects Hill so as to be consistent for all real $\xi$ (for positive data), with variance between Hill's and Pickands'. Both are useful when the sign of $\xi$ is in doubt; for clearly heavy-tailed data Hill (or the GPD MLE, which is also consistent under the same $k$ conditions) is preferred.</p>`,
      },
      {
        title: 'From tail index to extreme quantiles',
        html: R`
<p>The point of estimating $\gamma$ is usually to estimate a quantile $x_p = \bar F^{-1}(p)$ with $p < 1/n$, beyond the data. Regular variation suggests anchoring at $X_{(k+1)} \approx x_{k/n}$ and scaling: $\bar F(x)/\bar F(X_{(k+1)}) \approx (x/X_{(k+1)})^{-1/\gamma}$. Setting this equal to $p/(k/n)$:</p>
<div class="thm" data-kind="Proposition" data-name="Weissman (1978) estimator">
<p>$$\hat x_p = X_{(k+1)}\Big(\frac{k}{np}\Big)^{\hat\gamma_k}.$$
Its relative error is dominated by that of $\hat\gamma_k$ multiplied by $\log(k/(np))$: $\log\hat x_p - \log x_p \approx (\hat\gamma_k - \gamma)\log\frac{k}{np}$ plus lower-order terms.</p>
</div>
<p>The factor $\log(k/(np))$ is the extrapolation distance on the log scale. It is the semiparametric counterpart of the $(\log T)^2$ growth in GEV return-level variance — the same story told twice: extrapolation amplifies the uncertainty in tail heaviness.</p>
<p>The lab below estimates the MSE of Hill's estimator as a function of $k$ by simulation, decomposed into squared bias and variance, for parents with different second-order parameters. The optimal $k$ moves from "all the data" (Pareto) to a small fraction (Student $t$ with many degrees of freedom).</p>`,
        lab: 'tail-index-mse',
      },
    ],
    quiz: [
      { q: R`Show that for exact Pareto data the Hill estimator is unbiased, and give its variance.`,
        a: R`By the Rényi representation $k\hat\gamma^H_k/\gamma$ is a sum of $k$ IID Exp(1) variables (the normalised spacings of the top $k+1$ log-observations), so $\E\hat\gamma^H_k = \gamma$ and $\Var\hat\gamma^H_k = \gamma^2/k$.` },
      { q: R`Why does Hill's estimator need both $k\to\infty$ and $k/n\to0$?`,
        a: R`$k\to\infty$ makes the average of $k$ log-excesses concentrate (variance $\gamma^2/k$). $k/n\to0$ makes the random threshold $X_{(k+1)}$ go to infinity, so the excesses come from the region where $\bar F$ behaves like $x^{-1/\gamma}$ and the slowly varying factor $L$ no longer matters.` },
      { q: R`For the Student $t_\nu$, how does the optimal $k$ scale with $n$, and why do larger $\nu$ make tail-index estimation harder?`,
        a: R`$\rho = -2/\nu$, so $k_{\text{opt}}\propto n^{-2\rho/(1-2\rho)} = n^{4/(\nu+4)}$. As $\nu$ grows, $\rho\to0$ and the exponent falls: a vanishing fraction of the data is usable, and the MSE rate $n^{\rho/(1-2\rho)}$ slows. The $t$ approaches the normal, whose tail is not regularly varying at all.` },
      { q: R`What does the Pickands estimator gain and lose relative to Hill?`,
        a: R`Gains: consistency for all real $\xi$ (not only $\xi>0$) and invariance to location and scale. Loses: much larger variance, erratic behaviour in $k$, and it uses only three order statistics at a time.` },
      { q: R`With $n = 10{,}000$, $k = 200$ and $\hat\gamma = 0.4$, how far does an error of $0.05$ in $\hat\gamma$ move the Weissman estimate of the $p = 10^{-5}$ quantile?`,
        a: R`$\log(k/(np)) = \log(200/0.1) = \log 2000 \approx 7.6$, so the log-quantile moves by $0.05\times7.6 = 0.38$: a factor $e^{0.38}\approx1.46$, i.e. a 46% error in the quantile from a 12.5% error in $\gamma$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'point-process-evt',
    domain: 'evt',
    order: 90,
    title: 'The Point-Process View of Extremes',
    subtitle: 'Exceedances form a Poisson process whose intensity contains the GEV, the GPD and the r-largest model',
    level: 3,
    sources: { other: 'Pickands (1971); Leadbetter, Lindgren & Rootzén (1983) Ch. 5; Resnick (1987) Ch. 3–4; Smith (1989), Stat. Sci. 4; Coles (2001) Ch. 7' },
    prereqs: ['peaks-over-threshold', 'poisson-process'],
    related: ['block-maxima', 'return-levels', 'extremal-index'],
    tags: ['Poisson process', 'exceedances', 'r-largest', 'unification', 'non-stationarity'],
    summary: R`Plot each observation as a point (time, normalised value). As $n\to\infty$, the points that land above any fixed level form, in the limit, a Poisson process on time × level, with intensity measure $\Lambda([t_1,t_2]\times(z,\infty)) = (t_2-t_1)[1+\xi(z-\mu)/\sigma]^{-1/\xi}$. This single statement implies the GEV for maxima (no points above $z$), the GPD for excesses (ratio of intensities), and the joint law of the $r$ largest order statistics. As a likelihood it combines the efficiency of threshold methods with the interpretability of annual GEV parameters, and it handles non-stationarity naturally through time-varying parameters.`,
    survey: {
      task: 'Model the times and sizes of all exceedances of a high threshold jointly',
      assumptions: R`$F\in\mathcal D(G_\xi)$; independence (or a cluster extension); threshold high enough`,
      output: R`Annual-scale $(\mu,\sigma,\xi)$ from exceedance data; $r$-largest likelihood`,
      strengths: 'Unifies block maxima and POT; parameters invariant to threshold choice; covariates natural',
      weaknesses: 'Same threshold sensitivity as POT; clustering must be handled separately',
      complexity: R`$O(k)$ per likelihood evaluation`,
    },
    sections: [
      {
        title: 'Extremes as points in a plane',
        html: R`
<p>So far we have extracted one statistic from the sample — the maximum, or the set of excesses. A richer object keeps track of <em>when</em> and <em>how large</em>. For a sequence $X_1,\dots,X_n$ and normalising constants $a_n, b_n$, define the point process
$$P_n = \sum_{i=1}^n \delta_{\left(\frac{i}{n+1},\ \frac{X_i - b_n}{a_n}\right)}$$
on $(0,1)\times\R$. Most points pile up near the bottom edge (the normalised bulk tends to the lower endpoint of $G$, often $-\infty$); we restrict attention to a region $A_u = (0,1)\times(u,\infty)$ above a fixed level $u$. The number of points of $P_n$ in $A_u$ is binomial with mean $n\bar F(a_nu + b_n)$, which converges by the domain-of-attraction condition. The natural guess — and the truth — is that the whole configuration of points in $A_u$ becomes a <a href="#/t/poisson-process">Poisson process</a>.</p>`,
      },
      {
        title: 'The limit theorem',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Pickands (1971); Leadbetter et al. (1983, Ch. 5); Resnick (1987, Ch. 4)">
<p>Let $X_i\iid F$ with $F^n(a_nz + b_n)\to G(z) = \exp\{-[1+\xi(z-\mu)/\sigma]^{-1/\xi}\}$. Then on $(0,1)\times(u,\infty)$, for any $u$ with $1+\xi(u-\mu)/\sigma>0$, $P_n$ converges in distribution to a Poisson process $P$ with intensity measure
$$\Lambda\big([t_1,t_2]\times(z,\infty)\big) = (t_2 - t_1)\Big[1+\xi\,\frac{z-\mu}{\sigma}\Big]^{-1/\xi}.$$
Equivalently, the intensity density is $\lambda(t,z) = \sigma^{-1}[1+\xi(z-\mu)/\sigma]^{-1/\xi-1}$: homogeneous in time, power-law (or exponential) in level.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>By Kallenberg's theorem, for a simple limit process it suffices to check (i) convergence of mean measures on rectangles $R = (t_1,t_2]\times(z,\infty)$ and (ii) convergence of void probabilities $\P(P_n(B)=0) \to e^{-\Lambda(B)}$ for finite unions $B$ of such rectangles. For (i), $\E P_n(R) = \#\{i: i/(n+1)\in(t_1,t_2]\}\cdot\bar F(a_nz+b_n) \approx n(t_2-t_1)\bar F(a_nz+b_n) \to (t_2-t_1)\,(-\log G(z))$ by the Poisson lemma. For (ii), a finite union of rectangles is a disjoint union of time slabs, within each of which "no points" means all the corresponding $X_i$ fall below a level; by independence the probability factorises across slabs, and each factor converges to $\exp\{-(\text{slab length})(-\log G(\text{level}))\}$ by the same lemma.</p>
</details>
<p>Time enters only through the slab length because the $X_i$ are identically distributed; levels enter through $-\log G$. In the dependent case, the limit becomes a <em>compound</em> Poisson process (clusters of points at the same time), which is the subject of <a href="#/t/extremal-index">the extremal index</a>.</p>`,
      },
      {
        title: 'One model, three corollaries',
        html: R`
<p>Write $\Lambda(z) = [1+\xi(z-\mu)/\sigma]^{-1/\xi}$ for the intensity per unit time above level $z$.</p>
<div class="cols">
<div><h4>Block maxima → GEV</h4><p>$\{M_n \le z\}$ is the event that no point lies in $(0,1)\times(z,\infty)$, so
$$\P(M_n\le z) \to e^{-\Lambda(z)} = G(z).$$</p></div>
<div><h4>Excesses → GPD</h4><p>Given a point above $u$, its level exceeds $u+y$ with probability
$$\frac{\Lambda(u+y)}{\Lambda(u)} = \Big[1 + \frac{\xi y}{\sigma + \xi(u-\mu)}\Big]^{-1/\xi},$$
a GPD with $\sigma_u = \sigma + \xi(u-\mu)$.</p></div>
<div><h4>$r$ largest → joint density</h4><p>The top $r$ points $z^{(1)}\ge\dots\ge z^{(r)}$ have joint density
$$e^{-\Lambda(z^{(r)})}\prod_{k=1}^r \lambda(z^{(k)}),$$
$\lambda = -\Lambda'$: $r$ points at those levels, none elsewhere above $z^{(r)}$.</p></div>
</div>
<p>The number of exceedances of $u$ is Poisson with mean $\Lambda(u)$, independent of their sizes — exactly the <a href="#/t/peaks-over-threshold">Poisson–GPD model</a>. All the univariate methods are marginal or conditional views of one process. The <strong>$r$-largest order statistic model</strong> (Smith, 1986; Tawn, 1988) uses the third corollary as a likelihood over blocks, with $r$ chosen by the same bias–variance reasoning as a threshold: too large an $r$ includes values not yet in the asymptotic regime.</p>
<div class="ex" data-name="Writing the r-largest density explicitly">
<p>For one block with ordered top values $z^{(1)}\ge\dots\ge z^{(r)}$ and $s_k = 1+\xi(z^{(k)}-\mu)/\sigma > 0$,
$$f(z^{(1)},\dots,z^{(r)}) = \exp\big\{-s_r^{-1/\xi}\big\}\prod_{k=1}^r \sigma^{-1}s_k^{-1/\xi - 1}.$$
With $r=1$ this is the GEV density, as it must be.</p>
</div>`,
      },
      {
        title: 'The point-process likelihood',
        html: R`
<p>Scale time so that one unit is one year (or one block), let the record span $n_y$ years, fix a threshold $u$ and let $x_1,\dots,x_N$ be the exceedances. The likelihood of a Poisson process with intensity $\lambda$ observed on a region $A$ is $\exp\{-\Lambda(A)\}\prod_i \lambda(x_i)$. Here $A = (0,n_y)\times(u,\infty)$, so
$$\ell(\mu,\sigma,\xi) = -n_y\Big[1+\xi\,\frac{u-\mu}{\sigma}\Big]^{-1/\xi} - N\log\sigma - \Big(1+\frac1\xi\Big)\sum_{i=1}^N \log\Big[1+\xi\,\frac{x_i-\mu}{\sigma}\Big].$$</p>
<div class="note" data-name="Why practitioners like it">
<ul>
<li>The parameters are those of the <strong>annual-maximum GEV</strong>, directly interpretable and comparable to a block-maxima fit, yet estimated from all exceedances (as efficient as POT).</li>
<li>In principle $(\mu,\sigma,\xi)$ do not depend on $u$ — unlike the GPD scale $\sigma_u$ — so stability across thresholds is checked on fixed targets.</li>
<li>Non-stationarity is easy: let $\mu(t)$, $\log\sigma(t)$ depend on covariates and integrate the intensity over time, $\Lambda(A) = \int_0^{n_y}[1+\xi(u-\mu(t))/\sigma(t)]^{-1/\xi}\d t$. A time-varying threshold is also allowed.</li>
</ul>
</div>
<p>The usual regularity caveat applies ($\xi>-1/2$), and the model assumes independent exceedances; with clustering, decluster first or include the extremal index.</p>`,
      },
      {
        title: 'Seeing the Poisson limit',
        html: R`
<p>The lab draws a sample of size $n$ from a chosen parent, normalises it with the von Mises constants $b_n = F^{\leftarrow}(1-1/n)$, $a_n = 1/(nf(b_n))$, and plots the points above a level $u$ in the (time, level) plane. Repeating the experiment, it tallies the number of points that fall in a fixed region $A = (t_1, t_2)\times(z_0,\infty)$ and compares the histogram with the Poisson pmf of mean $\Lambda(A) = (t_2-t_1)(1+\xi z_0)^{-1/\xi}$. For the exponential or Pareto the match is excellent even at small $n$; for the normal at small $n$ the mean is visibly off — the same slow convergence seen for maxima, now in a different guise.</p>`,
        lab: 'point-process-evt-poisson',
      },
      {
        title: 'Why the point-process view matters',
        html: R`
<ul>
<li><strong>Unification.</strong> Block maxima, POT and $r$-largest methods are not competing theories but different likelihoods for one limiting process, which explains why their parameter estimates must agree.</li>
<li><strong>Dependence.</strong> For stationary sequences under mixing, the limit becomes a compound Poisson process: cluster positions are Poisson with rate $\theta\Lambda$, and each cluster carries a random number of points. This is the cleanest way to understand the <a href="#/t/extremal-index">extremal index</a> and declustering.</li>
<li><strong>Multivariate extremes.</strong> The point process of normalised vectors converges to a Poisson process on a cone whose intensity — the <em>exponent measure</em> — encodes extremal dependence, which is the starting point of <a href="#/t/multivariate-extremes">multivariate extreme value theory</a>.</li>
<li><strong>Spatial and functional extremes.</strong> Max-stable processes are built from Poisson processes of random functions (de Haan, 1984); the same machinery powers models of spatial rainfall and heat extremes.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Using the point-process limit, derive the probability that the second-largest of $n$ normalised observations is at most $z$.`,
        a: R`$\{X_{(2)}\le z\}$ means at most one point in $(0,1)\times(z,\infty)$, a Poisson count with mean $\Lambda(z) = -\log G(z)$. So the limit is $e^{-\Lambda(z)}(1+\Lambda(z)) = G(z)\{1 - \log G(z)\}$.` },
      { q: R`In the point-process likelihood, which term accounts for the absence of exceedances outside the observed ones, and why does it involve only the threshold?`,
        a: R`$-n_y[1+\xi(u-\mu)/\sigma]^{-1/\xi} = -\Lambda(A)$, the expected number of points in the observation region $A = (0,n_y)\times(u,\infty)$. The Poisson likelihood is $e^{-\Lambda(A)}\prod\lambda(x_i)$; $\Lambda(A)$ depends on $A$'s lower boundary $u$ and its time length, not on where the points fell.` },
      { q: R`Show that the excess-over-threshold distribution derived from the point process has a scale linear in $u$.`,
        a: R`$\Lambda(u+y)/\Lambda(u) = \{[1+\xi(u+y-\mu)/\sigma]/[1+\xi(u-\mu)/\sigma]\}^{-1/\xi} = [1 + \xi y/(\sigma + \xi(u-\mu))]^{-1/\xi}$, a GPD with $\sigma_u = \sigma + \xi(u-\mu)$, linear in $u$ with slope $\xi$.` },
      { q: R`Why might one prefer the point-process likelihood over the GPD likelihood when the goal is an annual-maximum return level?`,
        a: R`It estimates the annual GEV parameters directly from all exceedances, so return levels come straight from the GEV quantile formula with a single covariance matrix; the exceedance rate is built into the model rather than estimated separately and combined via the delta method; and non-stationary versions are straightforward.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'extremal-index',
    domain: 'evt',
    order: 100,
    title: 'Dependent Extremes & the Extremal Index',
    subtitle: 'Stationary series still have GEV maxima, but extremes arrive in clusters of mean size 1/θ',
    level: 3,
    sources: { other: 'Leadbetter (1983), Z. Wahrsch. 65; Leadbetter, Lindgren & Rootzén (1983) Ch. 3; Hsing, Hüsler & Leadbetter (1988); Ferro & Segers (2003), JRSS-B 65; Coles (2001) Ch. 5' },
    prereqs: ['fisher-tippett', 'time-series'],
    related: ['point-process-evt', 'peaks-over-threshold', 'block-bootstrap', 'markov-chains'],
    tags: ['stationary sequences', 'D(u_n) condition', 'clustering', 'extremal index', 'declustering'],
    summary: R`Real extremes come from time series: heat waves last days, storms last hours, crashes come in bursts. For stationary sequences satisfying Leadbetter's long-range mixing condition $D(u_n)$, normalised maxima can still only converge to a GEV. Local dependence enters through one number, the extremal index $\theta\in[0,1]$: $\P(M_n\le u_n)\approx F^{n\theta}(u_n)$, so the series behaves like $n\theta$ independent observations as far as its maximum is concerned. Under further conditions $1/\theta$ is the mean size of a cluster of exceedances. The ARMAX process has $\theta = 1-\alpha$ exactly, while a Gaussian AR(1), however strongly correlated, has $\theta=1$: correlation is not the same as clustering of extremes.`,
    survey: {
      task: 'Extend extreme value theory to stationary sequences; quantify clustering',
      assumptions: R`Stationarity; $D(u_n)$ (long-range asymptotic independence at extreme levels)`,
      output: R`$\theta$; dependent max $\approx F^{n\theta}$; GEV with same $\xi$, adjusted $\mu,\sigma$`,
      strengths: 'One-number summary of local extremal dependence; GEV for block maxima still valid',
      weaknesses: R`$\theta$ estimators need a threshold and (often) a run or block length; $\theta$ alone does not give the cluster-size distribution`,
      complexity: R`$O(n)$ for runs, blocks and intervals estimators`,
    },
    sections: [
      {
        title: 'Extremes of time series',
        html: R`
<p>Let $\{X_t\}$ be a strictly stationary sequence with marginal distribution $F$, and $M_n = \max_{t\le n}X_t$. Two questions arise. Does $(M_n - b_n)/a_n$ still have a GEV limit? And how does the dependence change the answer compared with an IID sequence $\{\tilde X_t\}$ with the same marginal $F$ (the <em>associated</em> independent sequence)?</p>
<p>Two kinds of dependence need to be distinguished. <strong>Long-range</strong> dependence — extremes far apart in time influencing each other — could destroy the limit theory altogether (take $X_t \equiv X_1$: then $M_n = X_1$ for all $n$). <strong>Short-range</strong> dependence — an extreme value making nearby values extreme too — leaves the GEV family intact but causes clustering. Leadbetter's conditions separate the two.</p>`,
      },
      {
        title: "Leadbetter's D(u_n) condition and the extremal types theorem",
        html: R`
<div class="def" data-name="Condition D(u_n) (Leadbetter, 1974)">
<p>Write $F_{i_1\dots i_p}(u) = \P(X_{i_1}\le u,\dots,X_{i_p}\le u)$. The condition $D(u_n)$ holds if for all $1\le i_1 < \dots < i_p < j_1 < \dots < j_q\le n$ with $j_1 - i_p \ge l$,
$$\big|F_{i_1\dots i_p j_1\dots j_q}(u_n) - F_{i_1\dots i_p}(u_n)F_{j_1\dots j_q}(u_n)\big| \le \alpha_{n,l},$$
where $\alpha_{n,l_n}\to0$ for some sequence $l_n = o(n)$.</p>
</div>
<p>$D(u_n)$ asks only that events of the specific form "everything below $u_n$" become approximately independent across a gap $l_n$ that is negligible compared with $n$. It is much weaker than strong mixing and holds for most stationary models of interest.</p>
<div class="thm" data-kind="Theorem" data-name="Leadbetter (1974, 1983)">
<p>If $\P((M_n - b_n)/a_n\le x)\to G(x)$ nondegenerate, and $D(u_n)$ holds for $u_n = a_nx+b_n$ for every real $x$, then $G$ is a GEV.</p>
</div>
<details class="proof"><summary>Proof sketch (blocking)</summary>
<p>Divide $\{1,\dots,n\}$ into $k$ blocks of length $n/k$ and remove from the end of each block a short gap of length $l_n = o(n)$. The gaps carry negligible probability of containing the maximum. By $D(u_n)$ applied repeatedly, the events "block $j$ stays below $u_n$" are asymptotically independent, so $\P(M_n\le u_n) - \P(M_{n/k}\le u_n)^k \to 0$. Thus $\P(M_{n/k}\le u_n)\to G^{1/k}(x)$ for each fixed $k$, and the argument of <a href="#/t/fisher-tippett">max-stability</a> applies verbatim: $G$ is max-stable, hence GEV.</p>
</details>
<div class="def" data-name="Condition D′(u_n) (anti-clustering)">
<p>$\displaystyle\lim_{k\to\infty}\limsup_{n\to\infty}\ n\sum_{j=2}^{\lfloor n/k\rfloor}\P(X_1>u_n,\ X_j>u_n) = 0.$</p>
</div>
<p>$D'(u_n)$ forbids two exceedances close together (within a block of length $n/k$) more often than chance. Under $D(u_n)$ and $D'(u_n)$, the maximum has <em>the same</em> limit as the associated IID sequence (Leadbetter, 1974). Without $D'$, it does not, in a very structured way.</p>`,
      },
      {
        title: 'The extremal index',
        html: R`
<div class="def" data-name="Extremal index">
<p>The stationary sequence $\{X_t\}$ has extremal index $\theta\in[0,1]$ if for every $\tau>0$ there is a sequence $u_n(\tau)$ with $n\bar F(u_n(\tau))\to\tau$ and
$$\P\big(M_n\le u_n(\tau)\big)\to e^{-\theta\tau}.$$</p>
</div>
<p>For the associated IID sequence the limit would be $e^{-\tau}$ (Poisson lemma). So with dependence
$$\P(M_n\le u_n) \approx F^{n\theta}(u_n):$$
the maximum of $n$ dependent observations behaves like the maximum of $n\theta$ independent ones. If $\tilde M_n$ (IID) satisfies $\P((\tilde M_n - b_n)/a_n\le x)\to G(x)$, then $\P((M_n - b_n)/a_n \le x)\to G^\theta(x)$, and by max-stability $G^\theta$ is a GEV with the <em>same shape</em> $\xi$ and
$$\sigma_\theta = \sigma\theta^{\xi},\qquad \mu_\theta = \mu - \frac{\sigma}{\xi}\big(1 - \theta^{\xi}\big).$$
This is why block maxima of dependent data can be fitted with a GEV directly: the dependence is absorbed into $\mu$ and $\sigma$. Under $D'(u_n)$, $\theta = 1$.</p>
<div class="thm" data-kind="Theorem" data-name="Clusters (Hsing, Hüsler & Leadbetter, 1988); runs formula (O'Brien, 1987)">
<p>Under mixing conditions, the point process of exceedance times of $u_n$, with time scaled by $1/n$, converges to a compound Poisson process: clusters occur at rate $\theta\tau$, and cluster sizes are IID with a distribution $\pi$ whose mean is $1/\theta$ (when finite). Moreover, for suitable $r_n\to\infty$, $r_n = o(n)$,
$$\theta = \lim_{n\to\infty}\P\big(\max(X_2,\dots,X_{r_n})\le u_n\ \big|\ X_1>u_n\big):$$
$\theta$ is the limiting probability that an exceedance is the last of its cluster.</p>
</div>
<p>Thus $\theta = 1/(\text{mean cluster size})$: $\theta = 0.25$ means extreme events come in bursts of four exceedances on average. Note the consistency: the expected number of exceedances is $\tau$ = (number of clusters $\theta\tau$) × (mean size $1/\theta$).</p>`,
      },
      {
        title: 'Two instructive examples',
        html: R`
<div class="ex" data-name="ARMAX: θ = 1 − α exactly">
<p>Let $Z_t\iid$ unit Fréchet, $\P(Z\le z) = e^{-1/z}$, and for $0\le\alpha<1$
$$X_t = \max\big(\alpha X_{t-1},\ (1-\alpha)Z_t\big).$$
<em>Stationary marginal:</em> if $X_{t-1}$ is unit Fréchet, $\P(X_t\le x) = e^{-\alpha/x}\,e^{-(1-\alpha)/x} = e^{-1/x}$.
<em>Maximum:</em> since $X_t = \max(\alpha^{t-1}X_1, \max_{2\le j\le t}\alpha^{t-j}(1-\alpha)Z_j)$, all $X_t\le x$ for $t\le n$ iff $X_1\le x$ and $(1-\alpha)Z_j\le x$ for $2\le j\le n$. Hence
$$\P(M_n\le x) = e^{-1/x}e^{-(n-1)(1-\alpha)/x},\qquad \P(M_n\le nx)\to e^{-(1-\alpha)/x} = \Phi_1(x)^{1-\alpha}.$$
For the associated IID sequence the limit is $\Phi_1(x)$, so $\theta = 1-\alpha$. The clusters are visible in the recursion: after a large innovation, $X$ decays geometrically by factor $\alpha$, staying above a high $u$ for a geometric number of steps; $\P(\text{size}\ge j+1)\to\alpha^j$, mean $\sum_j\alpha^j = 1/(1-\alpha) = 1/\theta$.</p>
</div>
<div class="ex" data-name="Gaussian AR(1): θ = 1 despite strong correlation">
<p>For a stationary Gaussian sequence with autocorrelation $r_n$, Berman's condition $r_n\log n\to0$ implies $D(u_n)$ and $D'(u_n)$ at the normal levels, so $\theta = 1$ (Berman, 1964). The AR(1) $X_t = \phi X_{t-1} + \varepsilon_t$ with $|\phi|<1$ has $r_n = \phi^n$, so even with $\phi = 0.9$ extremes do not cluster asymptotically. The reason: given $X_t = x$ large, $X_{t+1}\approx\phi x$, which is far below $x$ on the scale $a_n\approx 1/x$ of normal extremes. At finite thresholds there is some clustering (the empirical $\hat\theta$ is below one), which fades slowly as the threshold rises.</p>
</div>
<p>In the lab, compare the two. The ARMAX series shows plateaus of exceedances decaying geometrically; the Gaussian AR(1) wanders smoothly but its exceedances of a high threshold are mostly isolated.</p>`,
        lab: 'extremal-index-armax',
      },
      {
        title: 'Estimating θ',
        html: R`
<p>All estimators use a high threshold $u$; let $N$ be the number of exceedances among $n$ observations.</p>
<ul>
<li><strong>Blocks estimator.</strong> Split the series into $b$ blocks of length $r$; let $K$ be the number of blocks containing at least one exceedance. Then $\hat\theta_B = K/N$ (clusters ≈ blocks with exceedances). A refinement, motivated by $\P(M_r\le u)\approx F^{r\theta}(u)$, is $\hat\theta = \frac{\log(1 - K/b)}{r\log(1-N/n)}$ (Smith &amp; Weissman, 1994).</li>
<li><strong>Runs estimator.</strong> Declare a new cluster whenever an exceedance follows at least $m$ consecutive non-exceedances; $\hat\theta_R = (\text{number of clusters})/N$. This is runs declustering (as in <a href="#/t/peaks-over-threshold">POT</a>) read as an estimator.</li>
<li><strong>Intervals estimator (Ferro &amp; Segers, 2003).</strong> Let $T_1,\dots,T_{N-1}$ be the inter-exceedance times. In the limit, normalised inter-exceedance times are $0$ with probability $1-\theta$ (within a cluster) and exponential with mean $1/\theta$ with probability $\theta$ (between clusters), so $\theta = 2(\E T)^2/\E T^2$. Estimating moments with a small-sample bias correction,
$$\hat\theta = \begin{cases}\min\Big(1,\ \dfrac{2(\sum T_i)^2}{(N-1)\sum T_i^2}\Big), & \max T_i\le 2,\\[3mm] \min\Big(1,\ \dfrac{2\{\sum(T_i-1)\}^2}{(N-1)\sum(T_i-1)(T_i-2)}\Big), & \max T_i>2.\end{cases}$$
It needs no block or run length, only the threshold.</li>
</ul>
<p>Uncertainty for $\hat\theta$ is best assessed with a <a href="#/t/block-bootstrap">block bootstrap</a> (resampling whole clusters or blocks), since the exceedances are dependent by construction.</p>`,
      },
      {
        title: 'Consequences for inference',
        html: R`
<p>The second lab simulates many series and compares the empirical distribution of $M_n$ with the IID prediction $F^n$ and the dependent prediction $F^{n\theta}$, making the "$n\theta$ effective observations" statement visible.</p>
<div data-lab="extremal-index-maxima"></div>
<ul>
<li><strong>Block maxima:</strong> fit the GEV as usual; $\theta$ is absorbed. Standard errors are fine if the block maxima themselves are approximately independent.</li>
<li><strong>POT:</strong> the marginal GPD is fine, but the event rate is $\theta\zeta_u$ and return levels must use $x_m = u + \frac\sigma\xi[(m\zeta_u\theta)^\xi - 1]$. Ignoring $\theta$ in an $m$-observation return level overstates it.</li>
<li><strong>Return periods of events vs exceedances:</strong> "the level is exceeded on average $T$ times per 1000 years" and "an event exceeding the level occurs on average every $T$ years" differ by the factor $1/\theta$.</li>
<li><strong>Warning:</strong> $\theta<1$ is not implied by correlation (Gaussian AR(1)), and $\theta=1$ does not mean independence. Heavy-tailed financial returns often show $\theta\approx0.5$–$0.8$ because of volatility clustering (GARCH processes have $\theta<1$).</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`An ARMAX process has $\alpha = 0.75$. What are $\theta$, the mean cluster size, and the approximate probability that $\max_{t\le 1000}X_t\le 2000$?`,
        a: R`$\theta = 0.25$; mean cluster size $4$. $\P(M_n\le x)\approx\exp\{-n\theta/x\} = \exp(-1000\times0.25/2000) = e^{-0.125}\approx0.88$, versus $e^{-0.5}\approx0.61$ for an IID unit-Fréchet sequence.` },
      { q: R`If IID annual maxima would be $\text{GEV}(\mu,\sigma,\xi)$ but the series has extremal index $\theta$, what is the GEV for the actual annual maxima?`,
        a: R`$G^\theta$, which is $\text{GEV}(\mu_\theta,\sigma_\theta,\xi)$ with $\sigma_\theta = \sigma\theta^\xi$ and $\mu_\theta = \mu - \frac{\sigma}{\xi}(1-\theta^\xi)$ (and $\mu_\theta = \mu + \sigma\log\theta$ for $\xi=0$). Since $\theta\le1$, the maxima are stochastically smaller: clustering wastes extreme observations on the same event.` },
      { q: R`Why does a Gaussian AR(1) with $\phi = 0.9$ have extremal index 1?`,
        a: R`Its autocorrelations $\phi^n$ satisfy Berman's condition $r_n\log n\to0$, which yields $D'(u_n)$. Intuitively, if $X_t = x$ is extreme then $X_{t+1}\approx 0.9x$, and $x - 0.9x = 0.1x$ is huge compared with the scale $a_n\sim 1/x$ of normal extremes, so a second exceedance of a level near $x$ is asymptotically unlikely.` },
      { q: R`Explain the identity $\theta = 2(\E T)^2/\E T^2$ behind the intervals estimator.`,
        a: R`In the limit, a normalised inter-exceedance time $T$ is $0$ with probability $1-\theta$ (the next exceedance is in the same cluster) and $\text{Exp}$ with mean $1/\theta$ with probability $\theta$. Then $\E T = \theta\cdot\theta^{-1} = 1$ and $\E T^2 = \theta\cdot 2\theta^{-2} = 2/\theta$, so $2(\E T)^2/\E T^2 = \theta$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'multivariate-extremes',
    domain: 'evt',
    order: 110,
    title: 'Multivariate Extremes & Tail Dependence',
    subtitle: 'Componentwise maxima, the exponent measure, and the difference between correlation and joint extremes',
    level: 3,
    sources: { other: 'de Haan & Resnick (1977); Pickands (1981); Coles, Heffernan & Tawn (1999), Extremes 2; Ledford & Tawn (1996), Biometrika 83; Coles (2001) Ch. 8; Beirlant et al. (2004) Ch. 8–9' },
    prereqs: ['gev-distribution', 'multivariate'],
    related: ['point-process-evt', 'peaks-over-threshold', 'return-levels'],
    tags: ['componentwise maxima', 'exponent measure', 'Pickands dependence function', 'tail dependence', 'asymptotic independence'],
    summary: R`Joint extremes — two rivers flooding together, simultaneous losses in a portfolio — are governed by the dependence between variables in their joint tail, which correlation does not measure. Limits of componentwise maxima, after transforming margins to a common scale, are max-stable distributions $\exp\{-V(z_1,z_2)\}$; unlike the univariate case there is no finite-parameter family, but the exponent function $V$ is characterised by a spectral measure, or equivalently by the convex Pickands dependence function $A$. The coefficient $\chi = \lim_{u\to1}\P(V>u\mid U>u)$ separates asymptotic dependence from asymptotic independence, and $\bar\chi$ measures the residual dependence in the latter. The Gaussian copula is asymptotically independent for every $\rho<1$, so Gaussian models systematically understate joint extremes when data are asymptotically dependent.`,
    survey: {
      task: 'Model and measure dependence between extremes of several variables',
      assumptions: R`Margins in some $\mathcal D(G_{\xi_j})$; joint domain of attraction; margins standardised (e.g. to unit Fréchet)`,
      output: R`Exponent function $V$, Pickands function $A$, extremal coefficient $2A(1/2)$, $\chi$, $\bar\chi$`,
      strengths: 'Separates marginal tails from tail dependence; quantifies joint-exceedance risk',
      weaknesses: 'Nonparametric dependence class; max-stable models cannot represent asymptotic independence',
      complexity: R`Pairwise summaries $O(n)$; full likelihoods grow quickly with dimension`,
    },
    sections: [
      {
        title: 'What is a multivariate extreme?',
        html: R`
<p>There is no total order on $\R^d$, so "the maximum" of a sample of vectors is not defined. The classical choice is the vector of <strong>componentwise maxima</strong>,
$$\mathbf M_n = \big(\max_i X_{i1},\ \dots,\ \max_i X_{id}\big),$$
which need not be an observed point — the largest river flow and the largest rainfall in a year may occur in different months. It is nonetheless the right object for questions like "will both sea walls hold this year?", which ask whether every component stays below its design level. Other questions (the probability that a portfolio loss exceeds $x$, the probability that at least one site floods) need the joint tail of $\mathbf X$ itself; the same theory, phrased through the point-process limit, answers those too.</p>
<p>Two different things drive joint extreme risk. The <strong>marginal tails</strong> are handled by the univariate theory (GEV or GPD for each component). The <strong>tail dependence</strong> — how often the components are extreme together — is the new ingredient, and it is invisible to correlation, which is dominated by the bulk.</p>`,
      },
      {
        title: 'Limits of componentwise maxima',
        html: R`
<p>Since each margin can be transformed to any continuous distribution without changing the dependence, standardise to <strong>unit Fréchet</strong>, $Z_j = -1/\log F_j(X_j)$, so $\P(Z_j\le z) = e^{-1/z}$. Then maxima of $n$ copies divided by $n$ are again unit Fréchet, and the question is what joint limit $\P(\max_i Z_{i1}/n\le z_1,\ \max_i Z_{i2}/n\le z_2)$ can have.</p>
<div class="thm" data-kind="Theorem" data-name="de Haan & Resnick (1977); Pickands (1981)">
<p>Every bivariate limit of componentwise maxima with unit Fréchet margins has the form
$$G(z_1,z_2) = \exp\{-V(z_1,z_2)\},\qquad V(z_1,z_2) = 2\int_0^1\max\Big(\frac{w}{z_1},\ \frac{1-w}{z_2}\Big)\,\d H(w),$$
where $H$ is a probability measure on $[0,1]$ with $\int_0^1 w\,\d H(w) = 1/2$ (the <em>spectral</em> or <em>angular</em> measure). Conversely every such $H$ gives a valid limit.</p>
</div>
<p>Properties of the exponent function $V$:</p>
<ul>
<li><strong>Homogeneity:</strong> $V(tz_1,tz_2) = t^{-1}V(z_1,z_2)$. Hence $G^t(tz_1,tz_2) = G(z_1,z_2)$: bivariate max-stability.</li>
<li><strong>Margins:</strong> $V(z,\infty) = V(\infty,z) = 1/z$, which is what the mean constraint on $H$ ensures.</li>
<li><strong>Bounds:</strong> $\max(1/z_1,1/z_2)\le V(z_1,z_2)\le 1/z_1+1/z_2$. The lower bound (all mass of $H$ at $w=1/2$) is complete dependence; the upper bound (mass $1/2$ at each of $0$ and $1$) is independence.</li>
</ul>
<p>The key difference from the univariate case: $H$ can be any probability measure with mean $1/2$, so the class of limits is <em>nonparametric</em>. There is no extremal types theorem that reduces dependence to finitely many parameters. Parametric sub-families are chosen for convenience, the most popular being the <strong>logistic</strong> (Gumbel, 1960):
$$V(z_1,z_2) = \big(z_1^{-1/\alpha} + z_2^{-1/\alpha}\big)^{\alpha},\qquad 0<\alpha\le1,$$
with $\alpha=1$ independence and $\alpha\to0$ complete dependence.</p>`,
      },
      {
        title: 'The Pickands dependence function',
        html: R`
<p>Because of homogeneity, $V$ is determined by its values on a line. Writing the max-stable law as an extreme-value copula,
$$C(u,v) = \exp\Big\{\log(uv)\,A\Big(\frac{\log v}{\log(uv)}\Big)\Big\},\qquad u,v\in(0,1),$$
defines the <strong>Pickands dependence function</strong> $A:[0,1]\to[1/2,1]$; in Fréchet terms $V(z_1,z_2) = (z_1^{-1}+z_2^{-1})\,A\big(z_1/(z_1+z_2)\big)$.</p>
<div class="thm" data-kind="Proposition" data-name="Characterisation of A (Pickands, 1981)">
<p>$A$ is a valid Pickands dependence function if and only if it is convex and $\max(w,1-w)\le A(w)\le1$ (so $A(0)=A(1)=1$). $A\equiv1$ is independence; $A(w)=\max(w,1-w)$ is complete dependence. For the logistic model $A(w) = \{w^{1/\alpha}+(1-w)^{1/\alpha}\}^{\alpha}$.</p>
</div>
<details class="proof"><summary>Check of the two extremes</summary>
<p>Let $a = -\log u$, $b = -\log v$, so $\log(uv) = -(a+b)$ and $w = b/(a+b)$. If $A\equiv1$, $C = e^{-(a+b)} = uv$. If $A(w) = \max(w,1-w) = \max(a,b)/(a+b)$, then $C = e^{-\max(a,b)} = \min(u,v)$, the comonotone copula. Convexity of $A$ is inherited from the integral representation of $V$ (a mixture of the convex functions $w\mapsto\max(\cdot)$).</p>
</details>
<p>The <strong>extremal coefficient</strong> $\theta_E = V(1,1) = 2A(1/2)\in[1,2]$ has a direct meaning: $\P(M_1\le z, M_2\le z) = \P(M_1\le z)^{\theta_E}$ on a common marginal scale — the pair behaves like $\theta_E$ independent variables (1 = completely dependent, 2 = independent). For the logistic, $\theta_E = 2^\alpha$.</p>
<p>A simple nonparametric estimator (Pickands, 1981) uses the fact that if $(S,T)$ have unit exponential margins and extreme-value copula with function $A$, then $\min\{S/(1-w), T/w\}$ is exponential with rate $A(w)$: $\hat A(w) = n\big/\sum_i\min\{S_i/(1-w), T_i/w\}$. It need not be convex or respect the bounds; corrected versions (Capéraà, Fougères &amp; Genest, 1997, and convex projections) are preferred in practice. The lab compares the estimate from simulated componentwise maxima with the truth.</p>`,
        lab: 'multivariate-extremes-pickands',
      },
      {
        title: 'Asymptotic dependence: the coefficient χ',
        html: R`
<p>Max-stable models describe the limit of maxima. For the joint tail of the raw data, the basic question is whether the probability that one variable is extreme, given the other is, vanishes or not at the most extreme levels. With $U = F_1(X_1)$, $V = F_2(X_2)$ uniform:</p>
<div class="def" data-name="Tail dependence coefficient">
<p>$$\chi = \lim_{u\to1}\P(V>u\mid U>u),$$
when the limit exists. $X_1,X_2$ are <em>asymptotically dependent</em> if $\chi>0$ and <em>asymptotically independent</em> if $\chi=0$. A finite-level version (Coles, Heffernan &amp; Tawn, 1999) is
$$\chi(u) = 2 - \frac{\log\P(U < u, V < u)}{\log u},\qquad \chi(u)\to\chi\ \text{as } u\to1.$$</p>
</div>
<p>For a bivariate extreme-value distribution, $\chi = 2 - \theta_E = 2\{1 - A(1/2)\}$; the logistic gives $\chi = 2 - 2^\alpha>0$ for every $\alpha<1$. Max-stable dependence is always either asymptotic dependence or exact independence; there is nothing in between. That is a modelling problem, because much real data sits in between.</p>`,
      },
      {
        title: 'Asymptotic independence, the Gaussian case, and χ̄',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Sibuya (1960)">
<p>If $(X,Y)$ is bivariate normal with correlation $\rho<1$, then $\chi = 0$: the Gaussian copula is asymptotically independent.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Standardise to $\N(0,1)$ margins. Then $\P(X>x, Y>x)\le\P(X+Y>2x) = \bar\Phi\big(2x/\sqrt{2(1+\rho)}\big) = \bar\Phi(cx)$ with $c = \sqrt{2/(1+\rho)}>1$. By Mills' ratio, $\bar\Phi(cx)/\bar\Phi(x)\sim c^{-1}e^{-(c^2-1)x^2/2}\to0$. Hence $\P(Y>x\mid X>x)\to0$.</p>
</details>
<p>The same calculation shows the joint tail decays like $\P(X>x)^{2/(1+\rho)}$, faster than the marginal tail but slower than under independence ($\P(X>x)^2$). Ledford &amp; Tawn (1996) formalised this: on unit Fréchet margins,
$$\P(Z_1>z, Z_2>z)\sim L(z)\,z^{-1/\eta},\qquad \eta\in(0,1],$$
with $L$ slowly varying. The <strong>coefficient of tail dependence</strong> $\eta$ is $1$ under asymptotic dependence (with $L\to\chi$), $1/2$ under exact independence, and $(1+\rho)/2$ for the Gaussian. Coles et al. (1999) use the companion measure
$$\bar\chi = \lim_{u\to1}\bar\chi(u),\qquad \bar\chi(u) = \frac{2\log\P(U>u)}{\log\P(U>u,V>u)} - 1 = 2\eta - 1\in(-1,1].$$
The pair $(\chi,\bar\chi)$ classifies tail behaviour: $\bar\chi = 1$ and $\chi>0$ for asymptotic dependence; $\chi=0$ and $\bar\chi<1$ for asymptotic independence, with $\bar\chi$ measuring how much dependence remains ($\bar\chi = \rho$ for the Gaussian).</p>
<div class="ex" data-name="The t copula">
<p>The Student $t$ copula with $\nu$ degrees of freedom and correlation $\rho$ is asymptotically dependent for every $\rho>-1$, with
$$\chi = 2\,t_{\nu+1}\Big(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\Big)$$
(Embrechts, McNeil &amp; Straumann, 2002). Even at $\rho = 0$ — uncorrelated! — joint extremes occur with positive limiting conditional probability, because a common random scale (the chi-square mixing variable) inflates both components at once.</p>
</div>
<p>In the lab, simulate pairs from Gaussian, logistic and $t$ dependence and watch the empirical $\chi(u)$ and $\bar\chi(u)$ as $u\to1$. The Gaussian's $\chi(u)$ declines toward zero, though slowly enough to be mistaken for a positive limit at moderate $u$; its $\bar\chi(u)$ settles near $\rho$. The logistic and $t$ have $\chi(u)$ levelling off at a positive value and $\bar\chi(u)\to1$.</p>`,
        lab: 'multivariate-extremes-chi',
      },
      {
        title: 'Consequences and modern directions',
        html: R`
<div class="warn" data-name="Model choice changes the answer">
<p>Suppose two sites each have 100-year levels and the question is the probability that both are exceeded in the same year. Under independence it is $10^{-4}$; under complete dependence $10^{-2}$. A Gaussian copula fitted to data with $\rho = 0.7$ will put the answer near the independent end at very high levels, because $\chi = 0$; a logistic model fitted to the same data will put it at a fixed fraction of $10^{-2}$, because $\chi>0$. Which is right is a question about the data's tail dependence ($\chi$, $\bar\chi$), not about its correlation — and fitting a max-stable model to asymptotically independent data overstates joint risk at high levels, while a Gaussian model for asymptotically dependent data understates it.</p>
</div>
<ul>
<li><strong>Multivariate threshold models.</strong> The multivariate generalised Pareto distribution (Rootzén &amp; Tajvidi, 2006) is the threshold counterpart of the max-stable law, for vectors with at least one large component.</li>
<li><strong>Conditional extremes.</strong> Heffernan &amp; Tawn (2004) model $\mathbf X_{-j}$ given $X_j$ large via a regression-type limit that covers both asymptotic dependence and independence.</li>
<li><strong>Spatial extremes.</strong> Max-stable processes (Smith, Schlather, Brown–Resnick models) extend the exponent measure to random fields, with extremal coefficients as functions of distance.</li>
<li><strong>High dimensions.</strong> Extremal graphical models (Engelke &amp; Hitz, 2020) bring conditional-independence structure to multivariate Pareto distributions.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`For the logistic model with $\alpha = 0.5$, compute $A(1/2)$, the extremal coefficient and $\chi$.`,
        a: R`$A(1/2) = (2\cdot(1/2)^{2})^{1/2} = (1/2)^{1/2}\approx0.707$. $\theta_E = 2A(1/2) = 2^{0.5}\approx1.414$. $\chi = 2-\theta_E\approx0.586$.` },
      { q: R`Two variables have correlation 0.9 under a Gaussian copula. What is $\chi$? What is $\bar\chi$? Interpret.`,
        a: R`$\chi = 0$ (Sibuya), $\bar\chi = \rho = 0.9$. They are strongly dependent at moderately high levels, but the conditional probability of joint exceedance of a level $u$ tends to zero as $u\to1$; at the most extreme levels they behave as if independent, though the approach is slow.` },
      { q: R`Why is there no finite-parameter "multivariate extremal types theorem"?`,
        a: R`Max-stability only forces $V$ to be homogeneous of order $-1$ with unit-Fréchet margins; any spectral probability measure $H$ on $[0,1]$ with mean $1/2$ produces such a $V$. That is an infinite-dimensional family, so dependence must be modelled nonparametrically or with a chosen parametric subfamily.` },
      { q: R`Explain why the extremal coefficient satisfies $\P(M_1\le z, M_2\le z) = \P(M_1\le z)^{\theta_E}$ for unit Fréchet margins.`,
        a: R`$\P(M_1\le z, M_2\le z) = \exp\{-V(z,z)\} = \exp\{-V(1,1)/z\}$ by homogeneity, and $\P(M_1\le z) = e^{-1/z}$. So the joint probability is $(e^{-1/z})^{V(1,1)}$, with $\theta_E = V(1,1)$.` },
    ],
  });
})();
