/* Markov Chain Monte Carlo: foundations, Metropolis–Hastings, Gibbs, diagnostics, slice, Langevin,
 * Hamiltonian MC, tempering, hierarchical reparameterization, and variational inference. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'mcmc-foundations',
    domain: 'mcmc',
    order: 10,
    title: 'Why MCMC Works: Detailed Balance & Ergodicity',
    subtitle: 'Building a Markov chain whose long-run behaviour is the distribution you cannot sample',
    level: 2,
    sources: { aos: '§24.4', cb: '§5.6.3', other: 'Tierney (1994); Roberts & Rosenthal (2004); Meyn & Tweedie (2009)' },
    prereqs: ['stationary-distributions', 'monte-carlo-integration'],
    related: ['markov-chains', 'chain-classification', 'metropolis-hastings', 'mcmc-diagnostics', 'clt'],
    tags: ['Markov chain', 'detailed balance', 'ergodic theorem', 'CLT', 'asymptotic variance'],
    summary: R`Monte Carlo integration needs independent draws from $\pi$; for a posterior known only up to its normalizing constant we usually cannot produce them. Markov chain Monte Carlo replaces independence with a weaker and far more constructible property: a Markov chain whose <em>invariant</em> distribution is $\pi$. Detailed balance gives a local, checkable recipe for invariance; irreducibility and aperiodicity turn invariance into convergence; and the Markov chain ergodic theorem and CLT say that time averages along a single dependent path still estimate $\E_\pi f$ at rate $n^{-1/2}$, with the variance inflated by the factor $1 + 2\sum_k \rho_k$.`,
    survey: {
      task: R`Estimate $\E_\pi f$ for a target $\pi$ known up to a constant`,
      assumptions: R`Chain is $\pi$-invariant, $\pi$-irreducible (aperiodic for convergence in law); CLT needs e.g. geometric ergodicity + moments`,
      output: R`Dependent draws $X_1,\dots,X_n$; ergodic averages $\bar f_n$ with $\se = \sigma_f/\sqrt n$`,
      strengths: 'Works in high dimension with unnormalized targets; one framework covers every MCMC algorithm',
      weaknesses: R`Correlation inflates variance by $\tau = 1+2\sum\rho_k$; convergence can be slow and is hard to certify`,
      complexity: R`$O(n)$ target evaluations; effective sample size $n/\tau$`,
    },
    sections: [
      {
        title: 'The problem: integrals against unnormalized densities',
        html: R`
<p>Bayesian computation, statistical physics, and much of modern machine learning reduce to integrals of the form
$$\E_\pi f = \int f(x)\,\pi(x)\d x, \qquad \pi(x) = \frac{\tilde\pi(x)}{Z},\quad Z = \int \tilde\pi(x)\d x,$$
where $\tilde\pi$ can be evaluated pointwise but $Z$ cannot. For a posterior, $\tilde\pi(\theta) = p(y\mid\theta)p(\theta)$ and $Z = p(y)$ is the marginal likelihood, a $d$-dimensional integral as hard as the one we wanted in the first place.</p>
<p><a href="#/t/monte-carlo-integration">Plain Monte Carlo</a> needs IID draws from $\pi$. <a href="#/t/rejection-sampling">Rejection sampling</a> and <a href="#/t/importance-sampling">importance sampling</a> produce them (or weighted surrogates) from a proposal $q$, but their efficiency is governed by $\sup \pi/q$ or by $\E_q[(\pi/q)^2]$, quantities that typically grow <em>exponentially</em> in the dimension. A Gaussian proposal that is off by 10% in scale in each of 100 coordinates gives importance weights with astronomically large variance.</p>
<div class="note">
<p>MCMC gives up on independence. Instead of trying to hit $\pi$ globally in one shot, it makes many small, local moves, each of which only needs to know whether a nearby point is more or less probable than the current one, a ratio $\tilde\pi(y)/\tilde\pi(x)$ in which $Z$ cancels. Locality is what beats the curse of dimensionality; the price is that consecutive draws are correlated.</p>
</div>
<p>The theory then has to answer three questions: (1) how to design a chain whose long-run distribution is $\pi$; (2) when the chain actually gets there, from an arbitrary start; and (3) how accurate the resulting averages are. The answers are, respectively, detailed balance, ergodicity, and the Markov chain CLT.</p>`,
      },
      {
        title: 'Markov kernels, invariance, and detailed balance',
        html: R`
<div class="def" data-name="Markov kernel; invariant distribution">
<p>A <em>Markov transition kernel</em> on a state space $\mathcal X$ is a map $P(x, A)$ that is a probability measure in $A$ for each $x$ and measurable in $x$ for each $A$. A chain with kernel $P$ satisfies $\P(X_{t+1}\in A\mid X_0,\dots,X_t) = P(X_t, A)$. A distribution $\pi$ is <em>invariant</em> (stationary) for $P$ if
$$\pi P = \pi, \qquad\text{i.e.}\qquad \int_{\mathcal X} \pi(\d x)\,P(x, A) = \pi(A)\quad\text{for all } A.$$</p>
</div>
<p>Invariance says: if $X_t \sim \pi$ then $X_{t+1}\sim\pi$. It is a global balance condition (total flow into every set equals total flow out), which is hard to verify directly for a kernel defined by an algorithm. The workhorse is a stronger, <em>local</em> condition.</p>
<div class="def" data-name="Reversibility (detailed balance)">
<p>$P$ is <em>reversible</em> with respect to $\pi$ if
$$\pi(\d x)\,P(x,\d y) = \pi(\d y)\,P(y,\d x)$$
as measures on $\mathcal X\times\mathcal X$; that is, $\int_A \pi(\d x) P(x,B) = \int_B \pi(\d y) P(y, A)$ for all measurable $A, B$. For a discrete space this reads $\pi_i P_{ij} = \pi_j P_{ji}$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Detailed balance implies invariance">
<p>If $P$ is reversible with respect to $\pi$, then $\pi$ is invariant for $P$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Take $B = \mathcal X$ in the measure form of detailed balance: $\int_A \pi(\d x)P(x,\mathcal X) = \int_{\mathcal X}\pi(\d y)P(y,A)$. Since $P(x,\mathcal X) = 1$, the left side is $\pi(A)$ and the right side is $(\pi P)(A)$. Hence $\pi P = \pi$. In the discrete case: $\sum_i \pi_i P_{ij} = \sum_i \pi_j P_{ji} = \pi_j\sum_i P_{ji} = \pi_j$.</p>
</details>
<p>Two more facts make detailed balance a design tool rather than a curiosity:</p>
<ul>
<li><strong>Composition and mixtures.</strong> If $P_1,\dots,P_m$ each leave $\pi$ invariant, so do the composition $P_1P_2\cdots P_m$ and any mixture $\sum_k w_k P_k$. Mixtures of reversible kernels are reversible; compositions generally are <em>not</em> (the palindrome $P_1P_2P_1$ is). This is how Gibbs samplers, Metropolis-within-Gibbs, and HMC (momentum refresh followed by a trajectory) are assembled from simple pieces.</li>
<li><strong>Self-adjointness.</strong> Reversibility is exactly the statement that $P$, acting on $L^2(\pi)$ by $Pf(x) = \int f(y)P(x,\d y)$, is self-adjoint: $\langle f, Pg\rangle_\pi = \langle Pf, g\rangle_\pi$. Its spectrum is then real, in $[-1,1]$, and spectral theory controls both the convergence rate and the asymptotic variance (below).</li>
</ul>
<div class="warn" data-name="Invariance is not convergence">
<p>The identity kernel $P(x,\cdot) = \delta_x$ is reversible for <em>every</em> $\pi$. So is a chain that only ever moves within one of two disconnected components. Detailed balance guarantees the right fixed point; it says nothing about whether the chain ever reaches it.</p>
</div>`,
        lab: 'mcmc-foundations-chain',
      },
      {
        title: 'Ergodicity: from invariance to convergence',
        html: R`
<p>On a general state space, single points typically have probability zero, so "can reach every state" must be phrased in terms of sets of positive measure.</p>
<div class="def" data-name="Irreducibility, aperiodicity, Harris recurrence">
<p>A $\pi$-invariant chain is <em>$\pi$-irreducible</em> if for every $x$ and every $A$ with $\pi(A)>0$ there is an $n$ with $P^n(x,A) > 0$. It is <em>periodic</em> with period $k\ge 2$ if the space splits into disjoint sets $D_1,\dots,D_k$ visited cyclically; otherwise <em>aperiodic</em>. It is <em>Harris recurrent</em> if every set with $\pi(A) > 0$ is visited infinitely often almost surely, from every starting point.</p>
</div>
<p>In practice irreducibility is usually immediate: a random-walk Metropolis chain with a Gaussian proposal and a target with positive density everywhere can reach any open set in one step. Aperiodicity is also cheap: any Metropolis–Hastings chain that rejects with positive probability has $P(x,\{x\})>0$ and cannot cycle.</p>
<div class="thm" data-name="Convergence in total variation (Tierney, 1994)">
<p>Let $P$ have invariant distribution $\pi$ and be $\pi$-irreducible and aperiodic. Then for $\pi$-almost every $x$,
$$\|P^n(x,\cdot) - \pi\|_{\mathrm{TV}} = \sup_A |P^n(x,A) - \pi(A)| \longrightarrow 0.$$
If in addition the chain is Harris recurrent, this holds for <em>every</em> starting point $x$. (Metropolis–Hastings chains that are $\pi$-irreducible are automatically Harris recurrent.)</p>
</div>
<details class="proof"><summary>Proof sketch (finite state space)</summary>
<p>For a finite irreducible aperiodic chain there is an $m$ with $P^m_{ij} \ge \delta\,\pi_j$ for all $i,j$ and some $\delta > 0$ (a <em>Doeblin minorization</em>). Write $P^m = \delta\,\mathbf 1\pi + (1-\delta)Q$ with $Q$ stochastic. Couple two copies, one started at $x$ and one from $\pi$: every $m$ steps, with probability $\delta$ both draw their next state from $\pi$ jointly and coalesce. Hence $\|P^{n}(x,\cdot) - \pi\|_{\mathrm{TV}} \le \P(\text{not coalesced by time } n) \le (1-\delta)^{\lfloor n/m\rfloor}$. The general-state proof replaces the global minorization by one on a "small set" that the chain returns to (Nummelin splitting), which is where Harris recurrence enters.</p>
</details>
<p>The rate matters as much as the limit. A chain is <strong>geometrically ergodic</strong> if $\|P^n(x,\cdot)-\pi\|_{\mathrm{TV}} \le M(x)\,r^n$ for some $r\lt 1$ and $\pi$-integrable $M$, and <strong>uniformly ergodic</strong> if $M$ can be taken constant. The finite-state coupling argument above already proves uniform ergodicity. For reversible chains, geometric ergodicity is equivalent to a spectral gap: the spectrum of $P$ on $L^2_0(\pi) = \{f : \E_\pi f = 0\}$ stays bounded away from $\pm 1$ (Roberts & Rosenthal, 1997).</p>
<div class="ex" data-name="Two-state chain">
<p>On $\{0,1\}$ let $P_{01} = a$, $P_{10} = b$. Then $\pi = (b, a)/(a+b)$, detailed balance $\pi_0 a = \pi_1 b$ holds, and the second eigenvalue is $\lambda = 1-a-b$. Thus $P^n_{00} - \pi_0 = \pi_1\lambda^n$: convergence is geometric at rate $|\lambda|$. If $a=b=1$ the chain alternates deterministically, $\lambda = -1$, and $P^n$ never converges although $\pi = (\frac12, \frac12)$ is invariant: periodicity in its purest form. If $a, b$ are tiny the chain is "sticky", $\lambda\approx 1$, and convergence is slow even though nothing is formally wrong.</p>
</div>`,
      },
      {
        title: 'The ergodic theorem and the Markov chain CLT',
        html: R`
<p>We almost never use the law of $X_n$ directly. We use the <strong>ergodic average</strong> $\bar f_n = \frac1n\sum_{t=1}^n f(X_t)$, and want a law of large numbers and a CLT for it.</p>
<div class="thm" data-name="Ergodic theorem for Markov chains">
<p>If $P$ is $\pi$-invariant and $\pi$-irreducible, then for every $f\in L^1(\pi)$,
$$\bar f_n = \frac1n\sum_{t=1}^n f(X_t) \;\asto\; \E_\pi f$$
for $\pi$-almost every starting point, and for every starting point if the chain is Harris recurrent. Aperiodicity is <em>not</em> needed.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Started in stationarity, $(f(X_t))$ is a stationary sequence and Birkhoff's ergodic theorem gives convergence to $\E[f(X_0)\mid\mathcal I]$, where $\mathcal I$ is the invariant $\sigma$-field. Irreducibility forces every invariant set to have $\pi$-measure $0$ or $1$, so the conditional expectation is the constant $\E_\pi f$. Harris recurrence then transfers the statement from $\pi$-a.e. start to every start: the chain started anywhere a.s. hits a regeneration set, after which it behaves like a stationary chain, and a finite initial segment does not affect a Cesàro limit. (Periodicity does not hurt averages: a period-2 chain spends half its time in each cyclic class.)</p>
</details>
<p>This is why <strong>burn-in is not required for consistency</strong>: the starting point is forgotten in the limit. It is discarded in practice because a finite-$n$ average starting far from the bulk of $\pi$ carries an $O(1/n)$ bias that can be large in absolute terms.</p>
<div class="thm" data-name="Markov chain CLT">
<p>Suppose $X_t$ is a stationary $\pi$-invariant chain, $f\in L^2(\pi)$, and let $\gamma_k = \Cov_\pi(f(X_0), f(X_k))$ and $\rho_k = \gamma_k/\gamma_0$. If either (a) the chain is reversible and geometrically ergodic (Roberts & Rosenthal, 1997), or (b) it is geometrically ergodic and $\E_\pi|f|^{2+\delta}\lt\infty$ for some $\delta>0$, or (c) it is uniformly ergodic, then
$$\sqrt n\,(\bar f_n - \E_\pi f) \dto \N(0, \sigma_f^2),\qquad \sigma^2_f = \gamma_0\Big(1 + 2\sum_{k=1}^\infty \rho_k\Big) = \Var_\pi(f)\,\tau_f,$$
and the same limit holds from any initial distribution. The factor $\tau_f = 1 + 2\sum_{k\ge1}\rho_k$ is the <strong>integrated autocorrelation time</strong>, and $n/\tau_f$ is the <strong>effective sample size</strong>.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>The variance.</em> By stationarity,
$$\Var\Big(\sum_{t=1}^n f(X_t)\Big) = n\gamma_0 + 2\sum_{k=1}^{n-1}(n-k)\gamma_k,\qquad\text{so}\qquad n\Var(\bar f_n) = \gamma_0 + 2\sum_{k=1}^{n-1}\Big(1-\frac kn\Big)\gamma_k.$$
Geometric ergodicity gives $|\gamma_k| \le C r^k$ for $f\in L^2$ in the reversible case (the spectral gap bounds $|\langle \bar f, P^k \bar f\rangle_\pi|$, $\bar f = f - \E_\pi f$), so dominated convergence yields $n\Var(\bar f_n)\to\gamma_0 + 2\sum_{k\ge1}\gamma_k = \sigma^2_f$.</p>
<p><em>The normality.</em> Solve the Poisson equation $g - Pg = \bar f$; under geometric ergodicity $g = \sum_{k\ge0}P^k\bar f$ converges in $L^2(\pi)$. Then $\bar f(X_t) = [g(X_{t}) - Pg(X_{t-1})] + [Pg(X_{t-1}) - Pg(X_t)]$. The first bracket is a stationary, square-integrable martingale difference sequence; the second telescopes and contributes $O_p(1)$ to the sum. The martingale CLT gives $n^{-1/2}\sum_t \bar f(X_t) \dto \N(0, \E_\pi[(g(X_1) - Pg(X_0))^2])$, and a short computation shows this variance equals $\sigma_f^2$ (Gordin's method; Kipnis &amp; Varadhan, 1986, handle reversible chains with only $\sigma^2_f\lt\infty$).</p>
</details>
<div class="note" data-name="Reading the formula">
<p>$\sigma^2_f/\Var_\pi f = \tau_f$ is "how many correlated draws are worth one independent draw". Positive autocorrelation, the usual case for local samplers, makes $\tau_f\gt 1$. Antithetic chains with $\rho_1 \lt 0$ can make $\tau_f \lt 1$ and beat IID sampling, though this is rare in practice. For a reversible chain with spectrum of $P$ on $L^2_0(\pi)$ contained in $[-1, \lambda]$, the spectral theorem gives
$$\tau_f = \frac{\int \frac{1+s}{1-s}\, \mu_f(\d s)}{\int \mu_f(\d s)} \le \frac{1+\lambda}{1-\lambda},$$
where $\mu_f$ is the spectral measure of $\bar f$. A spectral gap $1-\lambda$ thus caps the variance inflation uniformly over $f$.</p>
</div>
<div class="ex" data-name="AR(1) as a model chain">
<p>The Gaussian autoregression $X_{t+1} = \phi X_t + \sqrt{1-\phi^2}\,\varepsilon_t$ has invariant law $\N(0,1)$, is reversible, and has $\rho_k = \phi^k$. For $f(x)=x$, $\tau = 1 + 2\sum_{k\ge1}\phi^k = \frac{1+\phi}{1-\phi}$. At $\phi = 0.9$, $\tau = 19$: ten thousand draws are worth about 526 independent ones. Many MCMC samplers behave locally like an AR(1) in each slow direction, which is why this formula is a good mental model. The <a href="#/t/gibbs-sampling">Gibbs sampler on a bivariate normal</a> is exactly an AR(1) with $\phi = \rho^2$.</p>
</div>`,
        lab: 'mcmc-foundations-clt',
      },
      {
        title: 'What can go wrong, and what the theory does not promise',
        html: R`
<div class="warn" data-name="Asymptotic guarantees, finite runs">
<p>Every theorem above is a statement about $n\to\infty$. A chain on a bimodal target with a narrow proposal is irreducible, aperiodic, Harris recurrent, and even geometrically ergodic, yet it may need $10^{12}$ steps to cross between modes. Within any feasible run it reports one mode with complete confidence. Diagnostics (<a href="#/t/mcmc-diagnostics">ESS, $\hat R$</a>) can detect some such failures, never all of them.</p>
</div>
<div class="warn" data-name="Heavy tails and missing CLTs">
<p>The ergodic theorem needs only $f\in L^1(\pi)$, but the CLT needs more: second moments <em>and</em> fast enough mixing. Random-walk Metropolis on a heavy-tailed target (polynomial tails) is not geometrically ergodic (Jarner &amp; Tweedie, 2003), and the ergodic averages can converge at a slower-than-$\sqrt n$ rate. A reported Monte Carlo standard error is then meaningless.</p>
</div>
<div class="warn" data-name="Adaptation breaks the Markov property">
<p>Tuning the proposal using the chain's own history makes the process non-Markov, and invariance can fail. Adaptive MCMC is valid under conditions such as <em>diminishing adaptation</em> plus <em>containment</em> (Roberts &amp; Rosenthal, 2007); the common practical solution is to adapt only during warm-up and then freeze the kernel, as Stan does.</p>
</div>
<p>Finally, two points of interpretation. First, the CLT variance $\sigma_f^2$ depends on $f$: a chain can estimate the posterior mean well (high ESS) while estimating a tail quantile poorly. Second, convergence <em>in distribution</em> of $X_n$ (which needs aperiodicity) and convergence of <em>averages</em> (which does not) are different statements, and the one that matters for estimation is the second.</p>`,
      },
      {
        title: 'Connections',
        html: R`
<table>
<thead><tr><th>Algorithm</th><th>How it achieves $\pi$-invariance</th><th>Reversible?</th></tr></thead>
<tbody>
<tr><td><a href="#/t/metropolis-hastings">Metropolis–Hastings</a></td><td>Accept/reject to enforce detailed balance</td><td>Yes</td></tr>
<tr><td><a href="#/t/gibbs-sampling">Gibbs sampling</a></td><td>Each conditional update is MH with acceptance 1</td><td>Random scan yes; systematic scan no</td></tr>
<tr><td><a href="#/t/slice-sampling">Slice sampling</a></td><td>Gibbs on an augmented uniform distribution</td><td>Yes</td></tr>
<tr><td><a href="#/t/langevin-mala">MALA</a></td><td>MH correction of a discretized diffusion</td><td>Yes</td></tr>
<tr><td><a href="#/t/hamiltonian-mc">Hamiltonian MC</a></td><td>Volume-preserving involution + MH, then momentum Gibbs step</td><td>Each piece yes; composition no</td></tr>
<tr><td><a href="#/t/tempering">Parallel tempering</a></td><td>Detailed balance on a product space</td><td>Yes</td></tr>
</tbody>
</table>
<p>The common skeleton (design a kernel by local balance, argue irreducibility, quantify mixing by autocorrelation) is the same for all of them. What differs is how far each move travels and how much information about $\pi$ (values, gradients, conditionals) it uses to get there. Compare this with the theory of <a href="#/t/stationary-distributions">stationary distributions</a> for finite chains, where the logic runs in the opposite direction: there the kernel is given and $\pi$ is unknown; here $\pi$ is given and we are free to invent the kernel.</p>`,
      },
    ],
    quiz: [
      { q: R`A kernel $P$ satisfies $\pi P = \pi$ but not detailed balance. Can it still be a valid MCMC kernel for $\pi$?`,
        a: R`Yes. Invariance (plus irreducibility) is what the ergodic theorem needs; detailed balance is a sufficient condition that is easy to verify, not a necessary one. Systematic-scan Gibbs samplers and HMC's full update are invariant but not reversible, and non-reversible samplers (lifted chains, underdamped dynamics) can even mix faster.` },
      { q: R`Why does the ergodic theorem not need aperiodicity, while convergence of $P^n(x,\cdot)$ does?`,
        a: R`A period-$k$ chain cycles through classes $D_1,\dots,D_k$, so the law of $X_n$ oscillates and cannot converge. But each class has $\pi$-mass $1/k$ and the chain spends exactly a fraction $1/k$ of its time in each, so time averages still converge to $\E_\pi f$. Averaging washes out the periodicity.` },
      { q: R`A chain produces $n = 20{,}000$ draws with lag-$k$ autocorrelations approximately $0.8^k$. What are the integrated autocorrelation time, the effective sample size, and the factor by which a naive IID standard error understates the truth?`,
        a: R`$\tau = (1+0.8)/(1-0.8) = 9$, so ESS $= 20000/9 \approx 2222$. The true standard error is $\sqrt{\tau}=3$ times the naive $s/\sqrt n$.` },
      { q: R`If the chain is consistent from any starting point, why discard a burn-in at all?`,
        a: R`Consistency is asymptotic. In a finite run, early draws from a poor start sit in low-probability regions and bias the average by an amount of order (distance from the bulk) $\times$ (time to get there)$/n$. Discarding the transient removes this bias at a small cost in variance. It is a finite-sample correction, not a requirement of the theory.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'metropolis-hastings',
    domain: 'mcmc',
    order: 20,
    title: 'The Metropolis–Hastings Algorithm',
    subtitle: 'Propose anything, then accept with exactly the probability that restores detailed balance',
    level: 2,
    sources: { aos: '§24.4', cb: '§5.6.3', other: 'Metropolis et al. (1953); Hastings (1970); Roberts, Gelman & Gilks (1997); Mengersen & Tweedie (1996)' },
    prereqs: ['mcmc-foundations'],
    related: ['rejection-sampling', 'importance-sampling', 'gibbs-sampling', 'langevin-mala', 'bayesian-inference', 'mcmc-diagnostics'],
    tags: ['Metropolis', 'Hastings ratio', 'random walk', 'independence sampler', 'optimal scaling', '0.234'],
    summary: R`Metropolis–Hastings turns <em>any</em> proposal mechanism $q(x,\cdot)$ into a $\pi$-reversible Markov chain by accepting a proposed move $x\to y$ with probability $\min\{1, \pi(y)q(y,x)/[\pi(x)q(x,y)]\}$. Only ratios of $\pi$ appear, so the normalizing constant is never needed. The correctness proof is two lines; the art lies in the proposal. Random-walk proposals must be scaled to the target (in high dimension, aim for about 23% acceptance), and independence proposals are excellent when they cover the target's tails and catastrophic when they do not.`,
    survey: {
      task: R`Sample a target $\pi$ known up to a constant`,
      assumptions: R`Can evaluate $\tilde\pi$ pointwise; proposal makes the chain $\pi$-irreducible`,
      output: R`Correlated draws from $\pi$ (asymptotically)`,
      strengths: 'Universal, simple, gradient-free; exact invariance for any proposal',
      weaknesses: R`Random walk scales badly with dimension ($O(d)$ steps per effective draw) and with conditioning`,
      complexity: R`One $\tilde\pi$ evaluation per step; RWM needs step size $\propto d^{-1/2}$`,
    },
    sections: [
      {
        title: 'The algorithm',
        html: R`
<p>Fix a target density $\pi$ on $\R^d$ (known up to a constant) and a proposal density $q(x, y)$, a conditional density in $y$ for each $x$.</p>
<div class="algo" data-name="Metropolis–Hastings">
<ol>
<li>Start at some $X_0$. For $t = 0,1,2,\dots$:</li>
<li>Draw a proposal $Y\sim q(X_t,\cdot)$.</li>
<li>Compute the <em>Hastings ratio</em> and the acceptance probability
$$\alpha(X_t, Y) = \min\left\{1,\; \frac{\pi(Y)\,q(Y, X_t)}{\pi(X_t)\,q(X_t, Y)}\right\}.$$</li>
<li>With probability $\alpha$ set $X_{t+1} = Y$ (accept); otherwise set $X_{t+1} = X_t$ (reject, and <em>record $X_t$ again</em>).</li>
</ol>
</div>
<p>The repeated value on rejection is essential, not a bookkeeping detail: it is the mass $r(x)\delta_x$ in the kernel that makes detailed balance hold. Discarding rejected steps biases the output toward low-probability regions.</p>
<p>Two special cases cover most of practice.</p>
<div class="cols">
<div><h4>Random-walk Metropolis (RWM)</h4><p>$Y = x + \sigma Z$ with $Z$ symmetric (usually $\N(0, \Sigma)$). Then $q(x,y) = q(y,x)$ and the ratio is just $\pi(y)/\pi(x)$: always accept uphill moves, accept downhill moves with probability equal to the density ratio. This is the original algorithm of Metropolis et al. (1953).</p></div>
<div><h4>Independence sampler</h4><p>$q(x,y) = g(y)$ ignores the current state. The ratio becomes $w(y)/w(x)$ with $w = \pi/g$, the importance weight. It is a Markov-chain cousin of <a href="#/t/importance-sampling">importance sampling</a> and <a href="#/t/rejection-sampling">rejection sampling</a> that needs no bound $M$ on $\pi/g$.</p></div>
</div>
<p>Everything else (Langevin proposals, Hamiltonian proposals, Gibbs updates, reversible-jump moves between dimensions) is Metropolis–Hastings with a cleverer $q$.</p>`,
      },
      {
        title: 'Correctness: the kernel satisfies detailed balance',
        html: R`
<p>The MH transition kernel has an absolutely continuous part (accepted moves) and an atom at the current point (rejections):
$$P(x, \d y) = q(x,y)\,\alpha(x,y)\d y + r(x)\,\delta_x(\d y),\qquad r(x) = 1 - \int q(x,z)\alpha(x,z)\d z.$$</p>
<div class="thm" data-name="Metropolis–Hastings is π-reversible">
<p>For any proposal $q$, the MH kernel $P$ satisfies detailed balance with respect to $\pi$, and therefore leaves $\pi$ invariant.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>We must show $\int_A \pi(x) P(x,B)\d x = \int_B \pi(y)P(y,A)\d y$ for all measurable $A,B$. Split $P$ into its two parts.</p>
<p><em>Continuous part.</em> Define $k(x,y) = \pi(x)q(x,y)\alpha(x,y)$. If $\pi(x)q(x,y) > 0$ then
$$k(x,y) = \pi(x)q(x,y)\min\Big\{1, \frac{\pi(y)q(y,x)}{\pi(x)q(x,y)}\Big\} = \min\{\pi(x)q(x,y),\; \pi(y)q(y,x)\},$$
and if $\pi(x)q(x,y) = 0$ then $k(x,y) = 0$ while (with the convention that $\alpha(y,x) = 0$ when the ratio's numerator vanishes) $k(y,x) = 0$ too. In all cases $k(x,y) = \min\{\pi(x)q(x,y), \pi(y)q(y,x)\}$, which is <strong>symmetric in $(x,y)$</strong>. Hence
$$\int_A\!\int_B k(x,y)\d y\d x = \int_B\!\int_A k(y,x)\d x\d y.$$</p>
<p><em>Atom.</em> $\int_A \pi(x)r(x)\delta_x(B)\d x = \int_{A\cap B}\pi(x)r(x)\d x$, which is symmetric in $(A,B)$.</p>
<p>Adding the two parts gives detailed balance, and invariance follows from the <a href="#/t/mcmc-foundations">detailed-balance proposition</a>.</p>
</details>
<div class="note" data-name="Why the minimum">
<p>Without correction, the flow from $x$ to $y$ is $\pi(x)q(x,y)$ and from $y$ to $x$ is $\pi(y)q(y,x)$. Accept/reject thins the larger of the two flows down to the smaller one, so they balance. Any $\alpha$ with $\pi(x)q(x,y)\alpha(x,y)$ symmetric works (Barker's rule $\alpha = \frac{\pi(y)q(y,x)}{\pi(x)q(x,y) + \pi(y)q(y,x)}$ is another). Peskun (1973) showed the MH choice, which makes the larger acceptance probability equal to 1, minimizes the asymptotic variance among all such rules for every $f$: fewer rejections means less "stickiness".</p>
</div>
<p>Invariance needs nothing about $q$. Convergence needs irreducibility, which does: if $q(x,y)>0$ whenever $\pi(y)>0$, or if $q$ is a random walk with density positive near zero and $\pi$ has a connected open support, the chain is $\pi$-irreducible; since $r(x)>0$ somewhere, it is aperiodic; and $\pi$-irreducible MH chains are Harris recurrent (Tierney, 1994). The <a href="#/t/mcmc-foundations">ergodic theorem</a> then applies from every starting point.</p>`,
      },
      {
        title: 'Random walk vs. independence proposals',
        html: R`
<p>The two classical proposals have strikingly different convergence theory.</p>
<div class="thm" data-name="Independence sampler (Mengersen & Tweedie, 1996)">
<p>Let $q(x,y) = g(y)$ and $w(x) = \pi(x)/g(x)$ with both densities normalized. If $W^* = \sup_x w(x) \lt \infty$, the chain is uniformly ergodic:
$$\|P^n(x,\cdot) - \pi\|_{\mathrm{TV}} \le \Big(1 - \frac{1}{W^*}\Big)^n\quad\text{for all }x.$$
If $w$ is unbounded ($\pi$-essentially), the chain is not even geometrically ergodic.</p>
</div>
<details class="proof"><summary>Proof of the bound</summary>
<p>For $y\ne x$, $P(x,\d y) \ge g(y)\min\{1, w(y)/w(x)\}\d y = \min\{g(y), \pi(y)/w(x)\}\d y \ge \pi(y)/W^*\,\d y$, because $g(y) = \pi(y)/w(y)\ge\pi(y)/W^*$ and $\pi(y)/w(x) \ge \pi(y)/W^*$. So $P(x,\cdot)\ge \frac1{W^*}\pi(\cdot)$ for every $x$: a global Doeblin minorization with constant $\delta = 1/W^*$. The coupling argument from the <a href="#/t/mcmc-foundations">foundations</a> gives the stated bound.</p>
</details>
<p>The condition $\sup \pi/g \lt \infty$ is exactly the one rejection sampling needs, and $1/W^*$ is exactly rejection sampling's acceptance rate. The difference: the independence sampler does not need to <em>know</em> $W^*$. When $w$ is unbounded (proposal tails lighter than the target's), the chain gets stuck: having reached an $x$ with huge $w(x)$, it accepts a move only with probability about $\E_g[\min(1, w(Y)/w(x))]\approx 1/w(x)$ and may sit there for thousands of iterations. Try the independence proposal with a small scale in the lab below.</p>
<p>For random-walk Metropolis, ergodicity depends on the target's tails rather than on the proposal. In one dimension RWM is geometrically ergodic essentially when $\pi$ has exponentially decaying tails (Mengersen &amp; Tweedie, 1996; Roberts &amp; Tweedie, 1996 for $d\gt1$, with an additional condition on the contours); a necessary condition is $\int e^{s\|x\|}\pi(\d x) \lt\infty$ for some $s>0$ (Jarner &amp; Tweedie, 2003). Heavy-tailed targets such as a Cauchy posterior are therefore handled poorly: the chain wanders into the tails and returns slowly.</p>
<div class="ex" data-name="Target with an unnormalized posterior">
<p>Suppose $y_1,\dots,y_n\iid$ Cauchy$(\theta, 1)$ with a $\N(0, 10^2)$ prior. The posterior $\tilde\pi(\theta) = e^{-\theta^2/200}\prod_i[1 + (y_i - \theta)^2]^{-1}$ has no closed-form normalizer and can be multimodal when the data contain outliers. RWM needs only $\log\tilde\pi(\theta') - \log\tilde\pi(\theta)$, an $O(n)$ computation, per step. Working on the log scale is mandatory in practice: products of many likelihood terms underflow, and the acceptance test $\log U \lt \log\tilde\pi(Y) - \log\tilde\pi(X)$ is numerically stable.</p>
</div>`,
        lab: 'metropolis-hastings-rw',
      },
      {
        title: 'Tuning the random walk: optimal scaling and 0.234',
        html: R`
<p>The proposal scale $\sigma$ trades off two failure modes. If $\sigma$ is tiny, almost everything is accepted but the chain moves by $O(\sigma)$ per step: a slow random walk. If $\sigma$ is huge, proposals land in the tails and are almost always rejected: the chain sits still. Somewhere in between the <em>expected squared jump distance</em> $\E\|X_{t+1}-X_t\|^2$ is maximized. Remarkably, the optimum is characterized by the acceptance rate alone in high dimension.</p>
<div class="thm" data-name="Optimal scaling of RWM (Roberts, Gelman & Gilks, 1997)">
<p>Let $\pi_d(x) = \prod_{i=1}^d f(x_i)$ with $f$ smooth and $I = \E_f[((\log f)')^2] \lt \infty$, start in stationarity, and propose $Y = x + \frac{\ell}{\sqrt d}Z$ with $Z\sim\N(0, I_d)$. Let $U^d_t = X^{(d)}_{\lfloor td\rfloor, 1}$ be the first coordinate, speeded up by a factor $d$. Then $U^d\Rightarrow U$, where $U$ is the Langevin diffusion
$$\d U_t = \sqrt{h(\ell)}\d B_t + \tfrac12 h(\ell)\,(\log f)'(U_t)\d t,\qquad h(\ell) = 2\ell^2\,\Phi\!\Big(-\frac{\ell\sqrt I}{2}\Big).$$
The asymptotic acceptance rate is $a(\ell) = 2\Phi(-\ell\sqrt I/2)$, the speed $h(\ell) = \ell^2 a(\ell)$ is maximized at $\hat\ell \approx 2.38/\sqrt I$, and at the optimum $a(\hat\ell) \approx 0.234$.</p>
</div>
<p>Three lessons. (i) <strong>Step size must shrink like $d^{-1/2}$</strong>, so RWM needs $O(d)$ iterations per effectively independent draw. (ii) <strong>The optimal acceptance rate, 0.234, does not depend on $f$</strong>, which makes it a practical tuning target even when $I$ is unknown. (iii) The efficiency curve is <strong>flat near the optimum</strong>: anything between roughly 0.15 and 0.4 loses little. In low dimension the optimum is higher (about 0.44 for $d=1$).</p>
<p>For correlated targets, isotropic proposals are crippled by the narrowest direction. With target covariance $\Sigma$, the right proposal is $\N(0, (2.38^2/d)\,\Sigma)$; adaptive Metropolis (Haario, Saksman &amp; Tamminen, 2001) estimates $\Sigma$ from the chain's history. Without such preconditioning, the cost grows with the condition number of $\Sigma$.</p>
<div class="note" data-name="Why the acceptance rate decouples from f">
<p>In high dimension the log acceptance ratio $\log\pi(Y)-\log\pi(x)$ is a sum of $d$ small, nearly independent terms, so by a CLT it is approximately $\N(-\frac{\ell^2 I}{2}, \ell^2 I)$: mean equal to minus half the variance, as for any log-likelihood ratio between nearby points. $\E\min(1, e^{W})$ for $W\sim\N(-s^2/2, s^2)$ equals $2\Phi(-s/2)$. Everything about $f$ enters only through $s = \ell\sqrt I$, so optimizing over $\ell$ is the same as optimizing over $s$, and the optimal acceptance $2\Phi(-\hat s/2)$ is universal.</p>
</div>`,
        lab: 'metropolis-hastings-scaling',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Forgetting the Hastings correction">
<p>With an asymmetric proposal (a log-normal step on a positive parameter, a drift toward a mode, a proposal that depends on the current point's gradient), the $q(y,x)/q(x,y)$ factor is mandatory. Omitting it gives a perfectly well-behaved chain that converges to the wrong distribution. The <a href="#/t/mcmc-foundations">first lab of the foundations topic</a> shows this happening.</p>
</div>
<div class="warn" data-name="Transforming parameters">
<p>To sample a positive $\sigma$ with a Gaussian random walk on $\eta=\log\sigma$, the target in $\eta$-space is $\pi_\sigma(e^\eta)\,e^{\eta}$: include the Jacobian. Equivalently, a random walk on $\log\sigma$ viewed in $\sigma$-space is an asymmetric proposal with $q(\sigma',\sigma)/q(\sigma,\sigma') = \sigma/\sigma'$.</p>
</div>
<div class="warn" data-name="Proposals outside the support">
<p>If $Y$ falls where $\pi(Y)=0$, reject (and record the current state again). Do <em>not</em> redraw until you land inside: that changes $q$ into a truncated proposal whose normalizing constant depends on $x$, and the Hastings ratio would need it.</p>
</div>
<div class="warn" data-name="Acceptance rate is a means, not the goal">
<p>A 23% acceptance rate does not certify a good sampler: a chain trapped in one mode of a bimodal target can be perfectly tuned for that mode. Judge samplers by <a href="#/t/mcmc-diagnostics">effective sample size per unit cost</a> and by between-chain agreement.</p>
</div>`,
      },
      {
        title: 'Extensions',
        html: R`
<ul>
<li><strong>Metropolis-within-Gibbs.</strong> Update one coordinate or block at a time with its own MH step; each leaves $\pi$ invariant, so the sweep does. This is how most conditionally non-conjugate models were fit before HMC.</li>
<li><strong>Gradient-informed proposals.</strong> <a href="#/t/langevin-mala">MALA</a> drifts toward higher density ($d^{-1/3}$ step scaling, optimal acceptance $\approx 0.574$); <a href="#/t/hamiltonian-mc">HMC</a> follows Hamiltonian trajectories ($d^{-1/4}$, $\approx 0.651$).</li>
<li><strong>Reversible jump</strong> (Green, 1995): MH between spaces of different dimension, with the Hastings ratio including a Jacobian for the dimension-matching map. Used for mixture models with unknown numbers of components and for variable selection.</li>
<li><strong>Delayed rejection</strong> (Tierney &amp; Mira, 1999): after a rejection, try a second, more conservative proposal, with a modified acceptance probability that preserves detailed balance.</li>
<li><strong>Pseudo-marginal MH</strong> (Andrieu &amp; Roberts, 2009): replace $\pi(y)$ by an unbiased nonnegative estimate (e.g., from a <a href="#/t/particle-filters">particle filter</a>); the chain remains exact, a surprising and very useful fact.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Show that the Metropolis–Hastings acceptance probability does not depend on the normalizing constant of $\pi$.`,
        a: R`$\alpha(x,y) = \min\{1, \pi(y)q(y,x)/[\pi(x)q(x,y)]\}$ involves $\pi$ only through the ratio $\pi(y)/\pi(x) = \tilde\pi(y)/\tilde\pi(x)$; the constant $Z$ cancels. This is the single property that makes MCMC usable for posteriors.` },
      { q: R`An independence sampler uses a $\N(0,1)$ proposal for a $t_3$ target. Is it uniformly ergodic? What will you see in the trace?`,
        a: R`No. $w(x) = \pi(x)/g(x)\propto e^{x^2/2}(1+x^2/3)^{-2}\to\infty$, so $\sup w = \infty$ and the chain is not even geometrically ergodic. The trace looks fine most of the time but occasionally reaches a tail point with huge weight and then sticks there for a very long run of rejections. Reversing the roles (a $t_3$ proposal for a normal target) gives bounded $w$ and uniform ergodicity.` },
      { q: R`In $d = 400$ dimensions, an isotropic RWM on an IID standard normal target is run with $\sigma = 0.5$. Roughly what acceptance rate do you expect, and is $\sigma$ too large or too small?`,
        a: R`$\ell = \sigma\sqrt d = 10$, so $a \approx 2\Phi(-\ell/2) = 2\Phi(-5)\approx 6\times10^{-7}$: essentially every proposal is rejected. The optimal $\sigma$ is $2.38/\sqrt{400}\approx 0.12$, giving acceptance $\approx 0.234$.` },
      { q: R`Why must a rejected proposal be recorded as a repeat of the current state rather than simply skipped?`,
        a: R`The kernel is $q\alpha\,\d y + r(x)\delta_x$; the atom $r(x)\delta_x$ is part of what balances flows. Skipping rejections replaces the chain by the "jump chain", whose invariant law is $\pi(x)(1-r(x))/\text{const}$, overweighting states from which it is easy to leave, typically low-density regions.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'gibbs-sampling',
    domain: 'mcmc',
    order: 30,
    title: 'Gibbs Sampling',
    subtitle: 'Sample each block from its full conditional, and let the conditionals assemble the joint',
    level: 2,
    sources: { aos: '§24.5', other: 'Geman & Geman (1984); Gelfand & Smith (1990); Besag (1974); Liu, Wong & Kong (1994); Roberts & Sahu (1997)' },
    prereqs: ['metropolis-hastings'],
    related: ['undirected-graphical-models', 'em-algorithm', 'gaussian-mixtures', 'hierarchical-mcmc', 'slice-sampling', 'multivariate'],
    tags: ['full conditional', 'conjugacy', 'data augmentation', 'blocking', 'Hammersley–Clifford'],
    summary: R`When a joint distribution is hard but each coordinate's <em>full conditional</em> $\pi(x_i\mid x_{-i})$ is a standard distribution, the Gibbs sampler cycles through the coordinates and draws each from its conditional. Every such update is a Metropolis–Hastings step that is always accepted, so $\pi$ is invariant with no tuning at all. The catch is geometry: Gibbs moves only along coordinate axes, and when the coordinates are strongly correlated it zig-zags in tiny steps. For a bivariate normal with correlation $\rho$ the chain is an AR(1) with coefficient $\rho^2$. Blocking, collapsing, and reparameterization are the cures.`,
    survey: {
      task: R`Sample $\pi(x_1,\dots,x_d)$ via its full conditionals`,
      assumptions: R`Full conditionals available to sample from (conjugacy); positivity for the conditionals to determine $\pi$`,
      output: R`Correlated draws from the joint`,
      strengths: 'No tuning, no rejections; natural for conjugate hierarchical models, mixtures, missing data',
      weaknesses: R`Slow when blocks are strongly dependent (rate $\approx$ squared maximal correlation); needs conjugacy`,
      complexity: 'One conditional draw per coordinate per sweep',
    },
    sections: [
      {
        title: 'The algorithm and why it is Metropolis–Hastings',
        html: R`
<p>Partition $x = (x_1,\dots,x_d)$ into components or blocks, and write $\pi_i(\cdot\mid x_{-i})$ for the full conditional of $x_i$ given all the others.</p>
<div class="algo" data-name="Systematic-scan Gibbs sampler">
<ol>
<li>Initialize $x^{(0)}$. For $t=0,1,\dots$:</li>
<li>Draw $x_1^{(t+1)}\sim\pi_1(\cdot\mid x_2^{(t)},\dots,x_d^{(t)})$.</li>
<li>Draw $x_2^{(t+1)}\sim\pi_2(\cdot\mid x_1^{(t+1)}, x_3^{(t)},\dots,x_d^{(t)})$, and so on, always conditioning on the most recent values.</li>
<li>Finish with $x_d^{(t+1)}\sim\pi_d(\cdot\mid x_1^{(t+1)},\dots,x_{d-1}^{(t+1)})$.</li>
</ol>
</div>
<p>The <em>random-scan</em> variant instead picks a coordinate $i$ uniformly at random at each step and updates only it.</p>
<div class="thm" data-kind="Proposition" data-name="A Gibbs update is an MH step with acceptance 1">
<p>The kernel $K_i$ that replaces $x_i$ by a draw from $\pi_i(\cdot\mid x_{-i})$ is the Metropolis–Hastings kernel for the proposal $q_i(x,y) = \pi_i(y_i\mid x_{-i})\,\ind\{y_{-i} = x_{-i}\}$, and its acceptance probability is identically 1. Consequently each $K_i$ is $\pi$-reversible, systematic scan $K_1K_2\cdots K_d$ is $\pi$-invariant, and random scan $\frac1d\sum_i K_i$ is $\pi$-reversible.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For a proposed $y$ with $y_{-i} = x_{-i}$, factor $\pi(x) = \pi_i(x_i\mid x_{-i})\,\pi_{-i}(x_{-i})$ and $\pi(y) = \pi_i(y_i\mid x_{-i})\,\pi_{-i}(x_{-i})$. Then
$$\frac{\pi(y)\,q_i(y,x)}{\pi(x)\,q_i(x,y)} = \frac{\pi_i(y_i\mid x_{-i})\pi_{-i}(x_{-i})\cdot\pi_i(x_i\mid x_{-i})}{\pi_i(x_i\mid x_{-i})\pi_{-i}(x_{-i})\cdot\pi_i(y_i\mid x_{-i})} = 1.$$
So every proposal is accepted and $K_i$ is exactly the MH kernel; <a href="#/t/metropolis-hastings">MH kernels are $\pi$-reversible</a>. Invariance is preserved under composition, and reversibility under mixtures.</p>
</details>
<p>Note what this does <em>not</em> give: systematic scan is generally not reversible (its time reversal scans in the opposite order), and no single $K_i$ is irreducible on its own. Irreducibility of the sweep must be checked separately.</p>
<div class="ex" data-name="Normal model with semi-conjugate prior">
<p>Let $y_1,\dots,y_n\iid\N(\mu,\sigma^2)$ with independent priors $\mu\sim\N(\mu_0,\tau_0^2)$ and $\sigma^2\sim\text{Inv-Gamma}(a,b)$. The joint posterior is not a standard family (unlike the fully conjugate normal–inverse-gamma prior), but both conditionals are:
$$\mu\mid\sigma^2,y \sim \N\!\left(\frac{\mu_0/\tau_0^2 + n\bar y/\sigma^2}{1/\tau_0^2 + n/\sigma^2},\; \frac{1}{1/\tau_0^2 + n/\sigma^2}\right),\qquad \sigma^2\mid\mu,y\sim\text{Inv-Gamma}\!\Big(a + \frac n2,\; b + \frac12\sum_i (y_i-\mu)^2\Big).$$
Each follows from writing the joint density and keeping only the factors that involve the variable being updated: "conditional $\propto$ joint" is the only rule needed. The second lab below runs exactly this sampler.</p>
</div>`,
      },
      {
        title: 'Do the conditionals determine the joint? Hammersley–Clifford',
        html: R`
<p>Gibbs sampling presupposes that the full conditionals carry all the information in $\pi$. Under a positivity condition, they do.</p>
<div class="thm" data-name="Brook's lemma / Hammersley–Clifford (Besag, 1974)">
<p>Suppose $\pi$ satisfies the <em>positivity condition</em>: if each $x_i$ is individually possible ($\pi_i(x_i)>0$), then $\pi(x_1,\dots,x_d)>0$. Then for any fixed reference point $y$,
$$\frac{\pi(x)}{\pi(y)} = \prod_{i=1}^d \frac{\pi_i(x_i\mid x_1,\dots,x_{i-1},y_{i+1},\dots,y_d)}{\pi_i(y_i\mid x_1,\dots,x_{i-1},y_{i+1},\dots,y_d)},$$
so the full conditionals determine $\pi$ up to normalization. (The Hammersley–Clifford theorem proper is the corollary that a positive distribution is Markov with respect to an undirected graph iff it factorizes over the graph's cliques; see <a href="#/t/undirected-graphical-models">undirected graphical models</a>.)</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Telescope through the intermediate points $z^{(i)} = (x_1,\dots,x_i, y_{i+1},\dots,y_d)$, with $z^{(0)} = y$ and $z^{(d)} = x$; positivity makes each $\pi(z^{(i)})>0$. Consecutive points differ only in coordinate $i$, so
$$\frac{\pi(z^{(i)})}{\pi(z^{(i-1)})} = \frac{\pi_i(x_i\mid z^{(i)}_{-i})\,\pi_{-i}(z^{(i)}_{-i})}{\pi_i(y_i\mid z^{(i)}_{-i})\,\pi_{-i}(z^{(i)}_{-i})},$$
since $z^{(i)}_{-i} = z^{(i-1)}_{-i}$. The marginal factors cancel, and the product over $i$ telescopes to $\pi(x)/\pi(y)$.</p>
</details>
<div class="warn" data-name="Positivity matters, and so does compatibility">
<p>Let $\pi$ be uniform on the two squares $[0,1]^2\cup[2,3]^2$. Each full conditional is a uniform on one unit interval, and the Gibbs sampler started in one square never leaves it: positivity fails and the chain is reducible. Conversely, one can write down a set of conditionals that are <em>incompatible</em> (no joint has them, e.g. $x_1\mid x_2\sim\N(x_2, 1)$ and $x_2\mid x_1\sim\N(x_1,1)$, whose would-be joint is improper). A Gibbs sampler built from them produces numbers, but they do not converge to anything. Improper priors can produce exactly this situation silently (Hobert &amp; Casella, 1996).</p>
</div>`,
      },
      {
        title: 'Correlation slows Gibbs: the bivariate normal',
        html: R`
<p>The geometric weakness of Gibbs sampling can be computed exactly in the simplest case. Let $(X,Y)$ be bivariate normal with zero means, unit variances, and correlation $\rho$. The conditionals are $X\mid Y=y\sim\N(\rho y, 1-\rho^2)$ and $Y\mid X=x\sim\N(\rho x, 1-\rho^2)$.</p>
<div class="thm" data-kind="Proposition" data-name="Gibbs on a bivariate normal is an AR(1)">
<p>Under systematic scan the $X$-subsequence satisfies $X_{t+1} = \rho^2 X_t + \sqrt{1-\rho^4}\,\varepsilon_t$ with $\varepsilon_t\iid\N(0,1)$. Hence $\Corr(X_t, X_{t+k}) = \rho^{2k}$, and the integrated autocorrelation time of $f(x,y) = x$ is
$$\tau = \frac{1+\rho^2}{1-\rho^2}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$Y_{t+1} = \rho X_t + \sqrt{1-\rho^2}\,\xi_t$ and $X_{t+1} = \rho Y_{t+1} + \sqrt{1-\rho^2}\,\eta_t$, so $X_{t+1} = \rho^2 X_t + \rho\sqrt{1-\rho^2}\,\xi_t + \sqrt{1-\rho^2}\,\eta_t$. The noise is Gaussian with variance $(1-\rho^2)(\rho^2+1) = 1-\rho^4$, independent of the past. This is a stationary AR(1) with coefficient $\phi=\rho^2$ and unit variance, for which $\rho_k = \phi^k$ and $\tau = (1+\phi)/(1-\phi)$, as in the <a href="#/t/mcmc-foundations">foundations</a>.</p>
</details>
<p>At $\rho = 0.99$, $\tau\approx 99.5$; at $\rho=0.999$, $\tau\approx 999.5$. The picture explains why: the conditional standard deviation $\sqrt{1-\rho^2}$ is the width of the ridge, while the ridge's length is about 1, so each axis-aligned step covers a fraction $\sqrt{1-\rho^2}$ of the distance it needs to travel, and the walk along the ridge is diffusive. Watch the zig-zag collapse in the lab.</p>
<p>More generally, for a two-block sampler (data augmentation) the geometric rate of convergence equals the square of the <em>maximal correlation</em> between the blocks, $\gamma^2 = \sup_{f,g}\Corr(f(X_1), g(X_2))^2$ (Liu, Wong &amp; Kong, 1994), and for a Gaussian target the convergence rate of the systematic-scan sampler is the spectral radius of a matrix built from the precision matrix (Amit, 1991; Roberts &amp; Sahu, 1997). The common theme: dependence between blocks is the enemy.</p>`,
        lab: 'gibbs-sampling-zigzag',
      },
      {
        title: 'Cures: blocking, collapsing, reparameterizing',
        html: R`
<ul>
<li><strong>Blocking.</strong> Update strongly dependent coordinates <em>jointly</em>. In the bivariate normal, drawing $(X,Y)$ as one block gives IID samples. In linear mixed models, drawing all regression coefficients from their multivariate normal conditional (rather than one at a time) removes the dependence among them. Liu, Wong &amp; Kong (1994) give operator-norm comparisons showing that grouping and collapsing reduce autocorrelation under broad conditions; the price is that the block conditional must be tractable.</li>
<li><strong>Collapsing.</strong> Integrate a variable out analytically and run Gibbs on the rest (e.g., integrate out the mixture weights in a Dirichlet–multinomial mixture). The collapsed sampler has smaller autocorrelation (Liu, 1994).</li>
<li><strong>Reparameterization.</strong> Change variables so the posterior is closer to having independent coordinates: centering or non-centering a <a href="#/t/hierarchical-mcmc">hierarchical model</a>, orthogonalizing regression predictors, or rotating by an estimated covariance.</li>
<li><strong>Metropolis-within-Gibbs.</strong> When a conditional is not standard, replace its exact draw by any $\pi_i$-invariant kernel: an MH step, an adaptive rejection sampler, or a <a href="#/t/slice-sampling">slice sampler</a>.</li>
</ul>
<div class="ex" data-name="Data augmentation for mixtures and probit">
<p>Gibbs sampling shines when <em>adding</em> variables makes conditionals tractable. For a <a href="#/t/gaussian-mixtures">Gaussian mixture</a>, introduce the latent labels $z_i$: given labels, the component means and weights have conjugate updates; given parameters, each $z_i$ is categorical with probabilities proportional to $w_k\phi(y_i;\mu_k,\sigma_k)$. This is the sampling analogue of the <a href="#/t/em-algorithm">EM algorithm</a>, with the E-step's expectations replaced by draws. For probit regression, Albert &amp; Chib (1993) introduce $z_i\sim\N(x_i^\top\beta, 1)$ with $y_i = \ind\{z_i>0\}$: $z_i\mid\beta,y$ is truncated normal and $\beta\mid z$ is normal.</p>
</div>`,
        lab: 'gibbs-sampling-normal',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Label switching">
<p>Mixture posteriors are invariant under permutations of component labels and so have $K!$ symmetric modes. A Gibbs sampler that explores only one of them gives sensible component-specific summaries; one that switches labels averages them into nonsense. Neither is "wrong" as a sampler of the joint; the problem lies in summarizing a non-identified parameter. Use permutation-invariant summaries or relabeling algorithms.</p>
</div>
<div class="warn" data-name="Data augmentation can be slow when data are informative">
<p>The rate of a two-block sampler is the fraction of missing information. When the latent variables carry most of the information about the parameters (many latent variables per parameter, or nearly separable classes), the sampler, like EM, slows to a crawl. Parameter expansion (PX-DA; Liu &amp; Wu, 1999) and interweaving (ASIS; Yu &amp; Meng, 2011) attack exactly this problem.</p>
</div>
<div class="warn" data-name="Absorbing states from improper or degenerate conditionals">
<p>In hierarchical models with a variance component $\tau^2$ and an improper prior $p(\tau^2)\propto 1/\tau^2$, the posterior can be improper while all conditionals are proper. The sampler then drifts toward $\tau\to 0$ and gets stuck. Always check propriety of the joint, not just of the conditionals.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that a Gibbs update of coordinate $i$ has Metropolis–Hastings acceptance probability 1.`,
        a: R`The proposal is $q(x,y) = \pi_i(y_i\mid x_{-i})$ with $y_{-i} = x_{-i}$. The Hastings ratio is $\frac{\pi(y)\pi_i(x_i\mid x_{-i})}{\pi(x)\pi_i(y_i\mid x_{-i})}$; writing $\pi(y) = \pi_i(y_i\mid x_{-i})\pi_{-i}(x_{-i})$ and $\pi(x) = \pi_i(x_i\mid x_{-i})\pi_{-i}(x_{-i})$, everything cancels to 1.` },
      { q: R`For a bivariate normal with $\rho = 0.95$, compute the lag-1 autocorrelation of the $X$-chain under systematic-scan Gibbs and the effective sample size of $10^4$ sweeps for estimating $\E X$.`,
        a: R`Lag-1 autocorrelation is $\rho^2 = 0.9025$. $\tau = (1+0.9025)/(1-0.9025) \approx 19.5$, so ESS $\approx 10^4/19.5\approx 513$.` },
      { q: R`Give an example where every full conditional is a proper distribution but the Gibbs sampler is useless, and say why.`,
        a: R`(1) Uniform on two disjoint squares: conditionals are proper uniforms but positivity fails, so the chain is reducible and never visits the other square. (2) $x_1\mid x_2 \sim \N(x_2,1)$, $x_2\mid x_1\sim\N(x_1, 1)$: conditionals are proper but no proper joint exists (the implied joint is $\propto e^{-(x_1-x_2)^2/2}$, improper), so the chain is null recurrent and wanders off.` },
      { q: R`Why can blocking two highly correlated coordinates into one joint update help so dramatically?`,
        a: R`Single-site updates move along axes in steps of the conditional width $\sqrt{1-\rho^2}$, so they diffuse slowly along the correlated ridge. A joint draw samples the block from its exact conditional distribution, which already accounts for the correlation, so it can move along the ridge in one step. In the bivariate normal it gives IID draws.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'mcmc-diagnostics',
    domain: 'mcmc',
    order: 40,
    title: 'Convergence Diagnostics: Autocorrelation, ESS & R-hat',
    subtitle: 'How long is long enough, and how precise is the answer',
    level: 2,
    sources: { aos: '§24.4', other: 'Geyer (1992); Gelman & Rubin (1992); Flegal & Jones (2010); Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021)' },
    prereqs: ['mcmc-foundations'],
    related: ['time-series', 'block-bootstrap', 'metropolis-hastings', 'hamiltonian-mc', 'hierarchical-mcmc'],
    tags: ['effective sample size', 'autocorrelation time', 'batch means', 'Gelman–Rubin', 'R-hat', 'MCSE'],
    summary: R`An MCMC run produces numbers whether or not it has converged, so every analysis needs two separate checks. <em>Mixing</em>: have the chains forgotten their starting points and explored the whole target? Multiple chains from dispersed starts and the potential scale reduction $\hat R$ (in its modern rank-normalized split form) address this. <em>Precision</em>: given mixing, how accurate is $\bar f_n$? The Monte Carlo standard error $\sqrt{\hat\sigma_f^2/n}$, estimated by the autocorrelation sum, batch means, or spectral methods, answers it, and its reciprocal-square form is the effective sample size. Diagnostics can prove non-convergence but never convergence.`,
    survey: {
      task: 'Assess convergence and Monte Carlo error of MCMC output',
      assumptions: R`Markov chain CLT holds; several chains from overdispersed starts for $\hat R$`,
      output: R`$\hat R$, bulk/tail ESS, MCSE, trace and autocorrelation plots`,
      strengths: 'Cheap, general, catches most practical failures (stuck chains, poor mixing, heavy autocorrelation)',
      weaknesses: 'Necessary, not sufficient: all chains can miss the same mode; estimates are noisy for short runs',
      complexity: R`$O(n\log n)$ per parameter (FFT autocorrelations, ranks)`,
    },
    sections: [
      {
        title: 'Two questions, kept separate',
        html: R`
<p>Suppose a sampler has produced $X_1,\dots,X_n$ and we report $\bar f_n$ for $\E_\pi f$. There are two distinct ways the report can be wrong.</p>
<div class="cols">
<div><h4>Bias from non-convergence</h4><p>The chain has not yet reached (or has not fully explored) $\pi$: it is still drifting from its starting point, or it has found only some of the modes. This is a <em>systematic</em> error that more draws from the same stuck chain do not fix.</p></div>
<div><h4>Variance from finite length</h4><p>The chain is mixing fine, but $n$ correlated draws carry only $n/\tau_f$ draws' worth of information. This is a <em>random</em> error, quantified by the <a href="#/t/mcmc-foundations">Markov chain CLT</a>: $\bar f_n\approx\N(\E_\pi f, \sigma^2_f/n)$.</p></div>
</div>
<p>The first question is attacked by comparing several chains, the second by estimating $\sigma_f^2$ from within the chains. A complete MCMC report gives, for every quantity of interest, $\hat R$, an effective sample size, and a Monte Carlo standard error (MCSE).</p>
<div class="note" data-name="Graphical checks first">
<p>A <strong>trace plot</strong> ($X_t$ against $t$) of a healthy chain looks like stationary noise: a "fat hairy caterpillar" with no trends, no long excursions, and no visible plateaus. Overlaid traces of several chains should be indistinguishable. An <strong>autocorrelation plot</strong> should decay to zero within a modest number of lags. Trends signal unfinished burn-in; flat stretches signal a rejecting MH chain or a sticky region; chains that never overlap signal multimodality or extreme slowness.</p>
</div>`,
      },
      {
        title: 'Autocorrelation, integrated autocorrelation time, and ESS',
        html: R`
<p>Recall from the CLT that $\sigma^2_f = \gamma_0\,\tau_f$ with $\tau_f = 1 + 2\sum_{k\ge1}\rho_k$. Define the <strong>effective sample size</strong> $\text{ESS} = n/\tau_f$, so that $\text{MCSE} = \sqrt{\gamma_0/\text{ESS}}$ looks exactly like an IID standard error.</p>
<p>The obvious plug-in estimator, $\hat\tau = 1 + 2\sum_{k=1}^{n-1}\hat\rho_k$ with sample autocorrelations $\hat\rho_k$, is <em>inconsistent</em>: each $\hat\rho_k$ has variance of order $1/n$, and summing $n$ of them leaves $O(1)$ noise. The sum must be truncated where the true $\rho_k$ have become negligible but before the noise accumulates.</p>
<div class="thm" data-name="Initial sequence estimators (Geyer, 1992)">
<p>For a stationary reversible chain, define $\Gamma_m = \gamma_{2m} + \gamma_{2m+1}$. Then $\Gamma_m$ is strictly positive, strictly decreasing, and strictly convex in $m$ (when $\gamma_k$ does not vanish identically), and $\sigma_f^2 = -\gamma_0 + 2\sum_{m\ge0}\Gamma_m$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>For a reversible chain, $\gamma_k = \langle \bar f, P^k\bar f\rangle_\pi = \int s^k\,\mu_f(\d s)$ with $\mu_f$ a positive measure on $[-1,1]$ (spectral theorem). Then $\Gamma_m = \int s^{2m}(1+s)\,\mu_f(\d s)$: the integrand is nonnegative since $1+s\ge 0$, and is decreasing and convex in $m$ since $s^{2m}$ is, for $|s|\le 1$. The variance identity is the CLT formula with the terms paired: $\gamma_0 + 2\sum_{k\ge1}\gamma_k = -\gamma_0 + 2\sum_{m\ge0}(\gamma_{2m}+\gamma_{2m+1})$.</p>
</details>
<p>This yields the <strong>initial positive sequence</strong> estimator: sum the estimated $\hat\Gamma_m$ until the first one that is not positive (they are positive in truth, so a nonpositive estimate means noise has taken over). The <em>initial monotone</em> variant additionally replaces each $\hat\Gamma_m$ by the minimum of those before it. Stan's ESS uses this, computed across chains (below). The labs report it as "ESS".</p>
<div class="def" data-name="Batch means">
<p>Split the run into $a$ consecutive batches of length $b$ ($n = ab$), with batch means $\bar Y_1,\dots,\bar Y_a$. Each $\bar Y_j$ is approximately $\N(\E_\pi f, \sigma^2_f/b)$ and nearly independent of the others when $b\gg\tau_f$, so
$$\hat\sigma^2_{\mathrm{BM}} = \frac{b}{a-1}\sum_{j=1}^a (\bar Y_j - \bar f_n)^2$$
estimates $\sigma_f^2$. It is consistent if both $b\to\infty$ and $a\to\infty$ (Jones, Haran, Caffo &amp; Neath, 2006); $b = \lfloor\sqrt n\rfloor$ is the standard choice (Flegal &amp; Jones, 2010).</p>
</div>
<p>The tension is visible in the second lab below: batches much shorter than $\tau_f$ are still correlated with each other, so $\hat\sigma^2_{\mathrm{BM}}$ is biased toward $\gamma_0$ (the naive IID variance); batches that are too long leave too few of them, and the estimate is noisy. This is the same bias–variance tradeoff that governs the block length in the <a href="#/t/block-bootstrap">block bootstrap</a> and bandwidths in <a href="#/t/time-series">spectral density estimation</a>, and for good reason: $\sigma^2_f = 2\pi s_f(0)$ is the spectral density at frequency zero.</p>`,
        lab: 'mcmc-diagnostics-acf',
      },
      {
        title: 'Multiple chains and the Gelman–Rubin R-hat',
        html: R`
<p>A single chain cannot tell you about the regions it has not visited. Gelman &amp; Rubin (1992) proposed running $M$ chains from <em>overdispersed</em> starting points and comparing between-chain and within-chain variation: if all chains have forgotten their starts, they should look alike.</p>
<div class="def" data-name="Potential scale reduction factor">
<p>With $M$ chains of length $N$, chain means $\bar\theta_{\cdot m}$, overall mean $\bar\theta_{\cdot\cdot}$, and within-chain variances $s_m^2$, let
$$B = \frac{N}{M-1}\sum_{m=1}^M(\bar\theta_{\cdot m} - \bar\theta_{\cdot\cdot})^2,\qquad W = \frac1M\sum_{m=1}^M s_m^2,\qquad \widehat{\Var}^+ = \frac{N-1}{N}W + \frac1N B,$$
and $\hat R = \sqrt{\widehat{\Var}^+/W}$.</p>
</div>
<p>The logic: $W$ underestimates $\Var_\pi\theta$ before convergence (each chain has explored only part of the space), while $\widehat{\Var}^+$ overestimates it when the starts are overdispersed. Both converge to $\Var_\pi\theta$, so $\hat R\to 1$, and $\hat R$ well above 1 says that running longer would still reduce the uncertainty about the scale of $\theta$, hence the name. In stationarity, $\E\widehat{\Var}^+ \approx \Var_\pi\theta$ up to $O(\tau/N)$ terms, because $B/N$ is an estimate of the variance of a chain mean.</p>
<p>The classic statistic has two blind spots, both addressed by Vehtari, Gelman, Simpson, Carpenter &amp; Bürkner (2021):</p>
<ul>
<li><strong>Split chains.</strong> A chain that drifts slowly (first half low, second half high) can have the same mean as the others. Splitting every chain in half and computing $\hat R$ on $2M$ half-chains turns within-chain non-stationarity into between-chain disagreement.</li>
<li><strong>Rank normalization.</strong> $\hat R$ is built on means and variances, so it is unreliable (or undefined) for heavy-tailed draws, and blind to chains that agree in location but differ in scale. Replace all draws by their pooled ranks $r$, map to normal scores $z = \Phi^{-1}\big((r - 3/8)/(S + 1/4)\big)$ where $S$ is the total number of draws, and compute split-$\hat R$ on the $z$'s ("bulk" $\hat R$). Repeat on the <em>folded</em> draws $|\theta - \text{median}(\theta)|$ to detect scale differences, and report the maximum.</li>
</ul>
<p>The recommended thresholds are $\hat R \lt 1.01$ (the older 1.1 is too lenient) and a bulk- and tail-ESS of at least about 100 per chain. The multi-chain ESS combines the chains through $\hat\rho_k = 1 - \frac{W - \frac1M\sum_m s_m^2\hat\rho_{k,m}}{\widehat{\Var}^+}$, which pulls autocorrelations up when the chains disagree, then applies Geyer's truncation; tail-ESS is the minimum ESS of the 5% and 95% quantile indicators.</p>`,
        lab: 'mcmc-diagnostics-rhat',
      },
      {
        title: 'Burn-in, thinning, and run length',
        html: R`
<p><strong>Burn-in (warm-up).</strong> Discarding an initial segment removes the transient bias discussed in the <a href="#/t/mcmc-foundations">foundations</a>. Modern practice discards the first half of each chain, which is also when adaptive samplers tune their step sizes and mass matrices (Stan's warm-up). The choice is pragmatic: if half the run is not enough to forget the start, the second half is unlikely to have converged either, and $\hat R$ will say so.</p>
<p><strong>Thinning.</strong> Keeping every $k$-th draw reduces storage but, for a fixed run length, never reduces the variance of $\bar f_n$ (Geyer, 1992; MacEachern &amp; Berliner, 1994): the discarded draws are correlated with the kept ones, but they still carry positive information. Thin only for memory or post-processing cost.</p>
<p><strong>Run length.</strong> Decide in terms of the precision you need. If you want the posterior mean to within $\pm 0.01$ posterior standard deviations (MCSE $=0.005\,\text{sd}$), you need ESS $\approx 1/0.005^2 = 40{,}000$; for 2 significant digits on the mean relative to its sd, ESS $\approx 400$ suffices. A "fixed-width" stopping rule (run until MCSE $\lt\epsilon$) is valid when $\hat\sigma^2_f$ is strongly consistent (Jones et al., 2006).</p>
<div class="ex" data-name="Reading a report">
<p>Four chains of 1,000 post-warm-up draws give, for a parameter $\beta$: posterior mean 1.37, sd 0.42, bulk-ESS 310, tail-ESS 180, $\hat R = 1.004$. MCSE of the mean $= 0.42/\sqrt{310}\approx 0.024$, so report "1.37" (the second decimal is roughly at the Monte Carlo noise level). The 90% interval endpoints rest on a tail-ESS of 180: adequate, but not for a third significant figure.</p>
</div>`,
      },
      {
        title: 'What diagnostics cannot do',
        html: R`
<div class="warn" data-name="Convergence cannot be proven from output">
<p>Every diagnostic compares the draws with each other. If all chains miss a region (a second mode separated by a deep valley, or the narrow neck of a <a href="#/t/hierarchical-mcmc">funnel</a>), they agree perfectly and $\hat R = 1.00$. Use dispersed initializations, multiple algorithms, sampler-specific signals (HMC's <a href="#/t/hamiltonian-mc">divergences</a>), simulation-based calibration (Talts et al., 2018), and, above all, understanding of the model's geometry.</p>
</div>
<div class="warn" data-name="ESS is per functional">
<p>A chain can have ESS 4,000 for a mean and 40 for a variance component or a 99th percentile. Report diagnostics for every quantity you interpret, including derived ones. Quantities that depend on the tails (quantiles, probabilities of rare events) need tail-ESS.</p>
</div>
<div class="warn" data-name="Noisy diagnostics for short runs">
<p>$\hat R$ and ESS are themselves estimates. With 50 draws per chain, $\hat R=1.01$ means little. Geweke's (1992) comparison of early and late segment means, and the Heidelberger–Welch stationarity test, are subject to the same caveat.</p>
</div>
<p>Rigorous alternatives exist: drift-and-minorization bounds (Rosenthal, 1995) give explicit, if usually very conservative, burn-in lengths for specific models, and coupling-based methods (Jacob, O'Leary &amp; Atchadé, 2020) produce unbiased estimators and computable bounds on the distance to stationarity from the meeting times of coupled chains.</p>`,
      },
    ],
    quiz: [
      { q: R`Why is $1 + 2\sum_{k=1}^{n-1}\hat\rho_k$ a bad estimator of $\tau_f$, and how does Geyer's initial positive sequence fix it?`,
        a: R`Each $\hat\rho_k$ has variance about $1/n$, so summing $n$ of them accumulates $O(1)$ noise and the estimator is inconsistent (in fact the full sum of sample autocorrelations is identically $-1/2$ with mean-centering). For reversible chains the pair sums $\Gamma_m = \gamma_{2m}+\gamma_{2m+1}$ are positive and decreasing, so truncating at the first nonpositive estimated $\hat\Gamma_m$ stops exactly when noise dominates.` },
      { q: R`Four chains are started at the same point and give $\hat R = 1.00$. What does this tell you?`,
        a: R`Less than it seems. $\hat R$ relies on overdispersed starting points: chains started together and stuck in the same region will agree. It rules out within-chain drift (if split $\hat R$ is used) but not missing modes. Restart from dispersed initializations.` },
      { q: R`Batch means with batch size $b = 5$ is applied to a chain with $\tau_f\approx 50$. In which direction is the estimated MCSE wrong?`,
        a: R`Too small. Batches much shorter than the autocorrelation time are positively correlated with each other, so their sample variance underestimates $\sigma_f^2/b$, and the estimate is biased toward the naive IID value $\gamma_0$. Batches should be much longer than $\tau_f$; $b\approx\sqrt n$ is the standard choice.` },
      { q: R`Does thinning by 10 improve the precision of a posterior mean computed from a fixed run of $10^5$ iterations?`,
        a: R`No. For a fixed number of iterations, the full chain always gives a variance at least as small as the thinned one; thinning discards information. Thinning per se is only useful to save storage or downstream computation.` },
      { q: R`Why is the rank-normalized $\hat R$ computed both on the draws and on the folded draws $|\theta - \text{median}|$?`,
        a: R`Ranks of the draws detect chains that disagree in location (the bulk). Chains that agree in location but differ in spread (one chain exploring tails, another not) have similar rank means; folding about the median converts a scale difference into a location difference, which split-$\hat R$ then detects.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'slice-sampling',
    domain: 'mcmc',
    order: 50,
    title: 'Slice Sampling & Auxiliary Variables',
    subtitle: 'Sample uniformly under the graph of the density, and the right marginal falls out',
    level: 2,
    sources: { other: 'Neal (2003), Slice sampling, Ann. Statist.; Damlen, Wakefield & Walker (1999); Murray, Adams & MacKay (2010); Mira & Tierney (2002)' },
    prereqs: ['gibbs-sampling'],
    related: ['rejection-sampling', 'metropolis-hastings', 'hamiltonian-mc', 'tempering'],
    tags: ['auxiliary variable', 'stepping out', 'shrinkage', 'elliptical slice', 'tuning-free'],
    summary: R`Sampling from $\pi(x)\propto f(x)$ is the same as sampling a point uniformly from the region under the graph of $f$ and keeping its $x$ coordinate. Slice sampling does this with a two-step Gibbs sampler on $(x,u)$: draw a height $u$ uniformly under $f(x)$, then draw $x$ uniformly from the "slice" $\{x: f(x)>u\}$. Neal's stepping-out and shrinkage procedures implement the second step without knowing the slice, using only evaluations of $f$, and remain exactly valid. The result adapts its step size automatically and is far less sensitive to tuning than random-walk Metropolis.`,
    survey: {
      task: R`Sample univariate (or coordinate-wise) conditionals with only $f$ evaluations`,
      assumptions: R`Can evaluate unnormalized $f$; slice (or its relevant component) found by interval search`,
      output: R`Correlated draws, no rejections in the Metropolis sense`,
      strengths: R`Robust to the width parameter $w$; adapts to local scale; no acceptance-rate tuning`,
      weaknesses: R`Coordinate-wise use inherits Gibbs's trouble with correlation; can miss distant modes`,
      complexity: R`$O(\text{width}/w + \log(w/\text{width}))$ evaluations per update`,
    },
    sections: [
      {
        title: 'The auxiliary-variable idea',
        html: R`
<p>Let $\pi(x) = f(x)/Z$ with $f\ge0$. Introduce a height variable $u$ and the joint density
$$p(x, u) = \frac{1}{Z}\,\ind\{0 \lt u \lt f(x)\},$$
the uniform distribution on the region $\mathcal A = \{(x,u): 0\lt u\lt f(x)\}$, whose area is $Z$.</p>
<div class="thm" data-kind="Proposition" data-name="Marginal of the uniform under the graph">
<p>Under $p$, the marginal of $x$ is $\pi$, and the conditionals are
$$u\mid x\sim\text{Unif}(0, f(x)),\qquad x\mid u\sim\text{Unif}(S_u),\quad S_u = \{x: f(x) \gt u\}.$$
Hence the Gibbs sampler that alternates these two draws has $p$ as its invariant distribution, and its $x$-marginal chain has $\pi$ invariant.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\int_0^\infty p(x,u)\d u = \frac1Z\int_0^{f(x)}\d u = f(x)/Z = \pi(x)$. For fixed $x$, $p(x,u)$ is constant in $u$ on $(0,f(x))$ and zero elsewhere, giving the first conditional. For fixed $u$, $p(x,u)$ is constant in $x$ on $S_u$ and zero elsewhere, giving the second. The <a href="#/t/gibbs-sampling">Gibbs sampler</a> leaves $p$ invariant; if $(x_t,u_t)\sim p$ then $x_{t+1}\sim\pi$.</p>
</details>
<p>This is the same geometry as <a href="#/t/rejection-sampling">rejection sampling</a>, which also samples uniformly under a graph, but globally, from an enclosing envelope. Slice sampling samples under the graph <em>locally</em>, by a Markov chain, and therefore needs no envelope. The first step is free; all the work is in step two, since $S_u$ is generally unknown (and in several dimensions it can be a complicated, possibly disconnected set).</p>
<div class="note" data-name="Why it adapts">
<p>The size of the move in $x$ is set by the width of the slice at the current height, which is set by the target itself: in a narrow peak, slices are narrow and moves are small; in a broad tail, slices are wide and moves are large. A random-walk proposal has one fixed scale everywhere.</p>
</div>
<div class="ex" data-name="An exact slice: the exponential">
<p>For $f(x) = e^{-x}$ on $x>0$, the slice at height $u$ is $(0, -\log u)$, explicitly available. The sampler is: $u\sim\text{Unif}(0, e^{-x})$, then $x'\sim\text{Unif}(0, -\log u)$. Writing $u = e^{-x}V$ with $V\sim\text{Unif}(0,1)$, the new point is $x' = W(x + E)$ with $W\sim\text{Unif}(0,1)$ and $E = -\log V \sim\text{Exp}(1)$. One can check directly that if $x\sim\text{Exp}(1)$ then $x+E\sim\text{Gamma}(2,1)$ and $W(x+E)\sim\text{Exp}(1)$, confirming invariance.</p>
</div>`,
      },
      {
        title: 'Finding the slice: stepping out and shrinkage',
        html: R`
<p>Neal (2003) showed that it is enough to replace "uniform on $S_u$" by any update that leaves the uniform distribution on $S_u$ invariant, and gave a practical one for univariate $x$ using a rough scale $w$.</p>
<div class="algo" data-name="Univariate slice sampler with stepping out and shrinkage (Neal, 2003)">
<ol>
<li><strong>Height.</strong> Draw $u\sim\text{Unif}(0, f(x_0))$ (in practice $\log u = \log f(x_0) - E$, $E\sim\text{Exp}(1)$).</li>
<li><strong>Stepping out.</strong> Place an interval of width $w$ randomly around $x_0$: $L = x_0 - wV$, $R = L + w$ with $V\sim\text{Unif}(0,1)$. While $f(L)\gt u$, set $L\leftarrow L-w$; while $f(R)\gt u$, set $R\leftarrow R + w$ (optionally with a cap of $m$ total steps, randomly split between the two sides).</li>
<li><strong>Shrinkage.</strong> Repeat: draw $x_1\sim\text{Unif}(L, R)$. If $f(x_1)\gt u$, accept and stop. Otherwise shrink the interval toward $x_0$: if $x_1\lt x_0$ set $L\leftarrow x_1$, else $R\leftarrow x_1$.</li>
</ol>
</div>
<div class="thm" data-name="Correctness of stepping out + shrinkage (Neal, 2003)">
<p>The update $x_0\mapsto x_1$ above is reversible with respect to the uniform distribution on $S_u$. Hence the full slice sampler leaves $\pi$ invariant.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>It suffices to show that the transition density $T(x_0\to x_1)$ for $x_0,x_1\in S_u$ is symmetric. Consider any particular sequence of random choices (initial position $V$, the intervals generated by stepping out, the rejected shrinkage points) that leads from $x_0$ to $x_1$. Key fact: stepping out from $x_0$ yields a final interval $I$ with exactly the same probability as stepping out from $x_1$, because the grid of possible endpoints $\{L + kw\}$ is the same and the stopping rule ("expand while the endpoint is inside the slice") depends only on the endpoints, not on where inside $I$ one started, provided $x_1\in I$ (with a cap $m$, the random split of the step budget preserves this). Similarly, a shrinkage sequence of rejected points $z_1, z_2,\dots$ that leads from $I$ toward $x_0$ and finally accepts $x_1$ has the same probability density as the mirror sequence from $x_1$ toward $x_0$: each rejected $z_j$ lies outside $S_u$, so it would have been rejected and would have cut the interval on the same side either way, as long as it does not separate $x_0$ from $x_1$, which shrinkage guarantees. Hence $T(x_0\to x_1) = T(x_1\to x_0)$, which is detailed balance for the uniform distribution.</p>
</details>
<p><strong>Cost and robustness.</strong> If the slice has width $\ell$, stepping out costs about $\ell/w$ evaluations when $w\ll\ell$, while shrinkage costs about $\log_2(w/\ell)$ evaluations when $w\gg\ell$. Logarithmic in the overshoot, linear in the undershoot: so err on the side of a <em>large</em> $w$. Compare random-walk Metropolis, where a step 10 times too large reduces the acceptance rate (and efficiency) by an order of magnitude. Neal also gives a <em>doubling</em> procedure (double the interval's width on a random side) that finds wide slices in $O(\log(\ell/w))$ steps, at the price of an extra acceptability check for reversibility.</p>`,
        lab: 'slice-sampling-stepout',
      },
      {
        title: 'Multivariate slice sampling',
        html: R`
<ul>
<li><strong>Coordinate-wise.</strong> Apply the univariate update to each coordinate in turn (each is a valid $\pi$-invariant update of a full conditional). This is how JAGS and many Gibbs-style samplers handle non-conjugate conditionals. It inherits the <a href="#/t/gibbs-sampling">correlation problems of Gibbs sampling</a>.</li>
<li><strong>Random directions.</strong> Choose a random direction $e$ and slice-sample along the line $x_0 + te$; correctness is the same argument applied to $t$.</li>
<li><strong>Hyperrectangles.</strong> Place a randomly positioned box around $x_0$ and shrink it toward $x_0$ coordinate-wise after rejections. Stepping out in $d$ dimensions is impractical, so the box size must be set adequately.</li>
<li><strong>Elliptical slice sampling</strong> (Murray, Adams &amp; MacKay, 2010). For a target $\propto L(x)\,\N(x; 0, \Sigma)$ (a Gaussian prior times a likelihood, as in <a href="#/t/gaussian-processes">Gaussian process</a> models), draw $\nu\sim\N(0,\Sigma)$ and slice-sample the angle $\theta$ on the ellipse $x(\theta) = x_0\cos\theta + \nu\sin\theta$ with respect to $L$ only, shrinking a bracket in $\theta$. The update is tuning-free, never rejects in the MH sense, and handles strong prior correlation because the ellipse respects $\Sigma$.</li>
</ul>
<p>The idealized ("simple") slice sampler that draws exactly from $S_u$ has strong theoretical properties: it is uniformly ergodic when $f$ is bounded and its support has finite Lebesgue measure (Mira &amp; Tierney, 2002), and Roberts &amp; Rosenthal (1999) give explicit convergence bounds for log-concave targets in terms of the dimension. The practical versions with stepping out approximate this behaviour as long as $w$ is not tiny.</p>`,
      },
      {
        title: 'Auxiliary variables more broadly',
        html: R`
<p>Slice sampling is one instance of a general principle: enlarge the space so that the extended target has easy conditionals or easy dynamics, and marginalize at the end.</p>
<table>
<thead><tr><th>Method</th><th>Auxiliary variable</th><th>What it buys</th></tr></thead>
<tbody>
<tr><td>Slice sampling</td><td>Height $u$ under the density</td><td>Adaptive step size, uniform conditionals</td></tr>
<tr><td>Data augmentation (<a href="#/t/gibbs-sampling">Albert–Chib</a>, mixture labels)</td><td>Latent variables of the model</td><td>Conjugate conditionals</td></tr>
<tr><td><a href="#/t/hamiltonian-mc">Hamiltonian MC</a></td><td>Momentum $p\sim\N(0,M)$</td><td>Long-distance deterministic moves</td></tr>
<tr><td>Swendsen–Wang (Ising/Potts)</td><td>Bond variables on edges</td><td>Cluster flips; escape from critical slowing</td></tr>
<tr><td><a href="#/t/tempering">Parallel tempering</a></td><td>Copies at other temperatures</td><td>Mode hopping</td></tr>
<tr><td>Damlen–Wakefield–Walker</td><td>One $u_i$ per likelihood factor</td><td>Product of simple constraints in Bayesian GLMs</td></tr>
</tbody>
</table>
<p>The Damlen, Wakefield &amp; Walker (1999) idea is worth spelling out: for $\pi(x)\propto g(x)\prod_i \ell_i(x)$, introduce $u_i\mid x\sim\text{Unif}(0,\ell_i(x))$. Then $x\mid u\propto g(x)\prod_i\ind\{\ell_i(x)\gt u_i\}$: the prior restricted to an intersection of sets, which for monotone $\ell_i$ (e.g. logistic likelihoods) is an interval or a polytope.</p>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Multimodality and gaps">
<p>Stepping out stops as soon as both endpoints are outside the slice. If the slice consists of two separated pieces and the interval grows past a gap only by luck, the sampler may never see the other mode. In the lab, choose the bimodal target with a small $w$, then increase $w$ beyond the gap to see mode jumps appear. Slice sampling is local, like any Markov chain.</p>
</div>
<div class="warn" data-name="Shrink toward the current point">
<p>Shrinkage must keep $x_0$ inside the interval. A tempting "optimization" that shrinks around the rejected point or recentres the interval breaks the symmetry argument, and the sampler becomes biased.</p>
</div>
<div class="warn" data-name="Numerical underflow">
<p>Work with $\log f$ and draw $\log u = \log f(x_0) - E$, $E\sim\text{Exp}(1)$. In the tails $f(x_0)$ may underflow to zero in floating point, and $u\sim\text{Unif}(0, 0)$ is degenerate.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that the $x$-marginal of the uniform distribution on $\{(x,u): 0\lt u\lt f(x)\}$ is $\pi = f/Z$.`,
        a: R`The joint density is $\frac1Z\ind\{0\lt u\lt f(x)\}$; integrating over $u$ gives $f(x)/Z$. The normalizing constant is the area under $f$, i.e. $Z$.` },
      { q: R`You set $w$ ten times too large relative to the typical slice width. Roughly how many extra function evaluations per update does this cost? What if $w$ were ten times too small?`,
        a: R`Too large: stepping out stops immediately, and shrinkage halves the interval roughly every rejection, so about $\log_2 10\approx 3.3$ extra evaluations. Too small: stepping out needs about 10 expansions. Either way the update remains exact, unlike a mis-scaled Metropolis step, whose efficiency degrades proportionally.` },
      { q: R`Why does slice sampling need no accept/reject step, and is there a sense in which it still "rejects"?`,
        a: R`Every update is a valid Gibbs step on the augmented $(x,u)$ space and always moves to a new point in the slice. Shrinkage proposals that fall outside the slice are discarded, which costs evaluations, but they are internal to constructing one draw: the chain never repeats its state because of a rejection.` },
      { q: R`What makes elliptical slice sampling particularly suited to Gaussian-process models?`,
        a: R`The prior $\N(0,\Sigma)$ is typically strongly correlated and high-dimensional. ESS proposes along ellipses $x_0\cos\theta + \nu\sin\theta$ with $\nu$ drawn from the prior, which leaves the prior exactly invariant, so only the likelihood enters the slice. The proposals automatically respect prior correlations, need no step size, and every update moves.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'langevin-mala',
    domain: 'mcmc',
    order: 60,
    title: 'Langevin Dynamics & MALA',
    subtitle: 'Follow the gradient of the log density, add noise, and correct the discretization with Metropolis',
    level: 3,
    sources: { other: 'Roberts & Tweedie (1996); Roberts & Rosenthal (1998); Dalalyan (2017); Durmus & Moulines (2017); Welling & Teh (2011)' },
    prereqs: ['metropolis-hastings'],
    related: ['stochastic-calculus', 'brownian-motion', 'hamiltonian-mc', 'sgd-optimizers', 'diffusion-models'],
    tags: ['Langevin diffusion', 'Fokker–Planck', 'ULA', 'MALA', 'gradient-based MCMC', 'SGLD'],
    summary: R`The overdamped Langevin diffusion $\d X_t = \frac12\nabla\log\pi(X_t)\d t + \d W_t$ has $\pi$ as its stationary distribution: the drift pushes toward high density and the noise spreads mass out, and the two balance exactly at $\pi$. Discretizing it with step $h$ gives the <em>unadjusted Langevin algorithm</em> (ULA), which is cheap but biased (its stationary law is not $\pi$) and can even diverge. Treating the same Euler step as a Metropolis–Hastings proposal gives MALA, which is exact and, in high dimensions, needs only $O(d^{1/3})$ steps per effective sample versus $O(d)$ for random-walk Metropolis.`,
    survey: {
      task: R`Sample smooth targets on $\R^d$ using gradients of $\log\pi$`,
      assumptions: R`$\log\pi$ differentiable; tails roughly between Gaussian and exponential for geometric ergodicity of MALA`,
      output: 'Correlated draws (MALA: exact invariance; ULA: biased stationary law)',
      strengths: R`Uses local geometry; step size $\propto d^{-1/3}$; foundation of SGLD and diffusion models`,
      weaknesses: 'ULA biased and possibly transient; MALA fragile for light- or heavy-tailed targets; isotropic noise needs preconditioning',
      complexity: R`One $\log\pi$ and one gradient evaluation per step`,
    },
    sections: [
      {
        title: 'The Langevin diffusion has π as its stationary law',
        html: R`
<p>Let $\pi(x)\propto e^{-U(x)}$ on $\R^d$ with $U$ smooth. Consider the stochastic differential equation (see <a href="#/t/stochastic-calculus">stochastic calculus</a>)
$$\d X_t = \tfrac12\nabla\log\pi(X_t)\d t + \d W_t = -\tfrac12\nabla U(X_t)\d t + \d W_t,$$
where $W$ is a standard <a href="#/t/brownian-motion">Brownian motion</a> in $\R^d$. Its generator, the operator describing $\frac{\d}{\d t}\E[g(X_t)]$, is
$$\mathcal L g = \tfrac12\Delta g + \tfrac12\nabla\log\pi\cdot\nabla g.$$
If $X_t$ has density $\rho_t$, then $\rho_t$ evolves by the <strong>Fokker–Planck equation</strong>
$$\partial_t\rho = \mathcal L^*\rho = -\nabla\cdot\big(\tfrac12\rho\,\nabla\log\pi\big) + \tfrac12\Delta\rho = \tfrac12\nabla\cdot\Big(\rho\,\nabla\log\frac{\rho}{\pi}\Big).$$</p>
<div class="thm" data-kind="Proposition" data-name="π is stationary, and the diffusion is reversible">
<p>$\mathcal L^*\pi = 0$, so $\pi$ is a stationary density. Moreover $\mathcal L$ is self-adjoint in $L^2(\pi)$:
$$\int g\,(\mathcal L h)\,\pi\d x = -\tfrac12\int\nabla g\cdot\nabla h\;\pi\d x = \int (\mathcal Lg)\,h\,\pi\d x,$$
which is the continuous-time form of detailed balance.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>From the last form of Fokker–Planck, $\rho = \pi$ gives $\nabla\log(\pi/\pi) = 0$, so $\partial_t\rho = 0$. Directly: $\pi\nabla\log\pi = \nabla\pi$, so $-\nabla\cdot(\frac12\pi\nabla\log\pi) + \frac12\Delta\pi = -\frac12\Delta\pi + \frac12\Delta\pi = 0$. For self-adjointness, note $\pi\,\mathcal Lh = \frac12(\pi\Delta h + \nabla\pi\cdot\nabla h) = \frac12\nabla\cdot(\pi\nabla h)$. Integrating against $g$ by parts (boundary terms vanish under mild decay conditions) gives $\int g\,\mathcal Lh\,\pi = -\frac12\int\pi\,\nabla g\cdot\nabla h$, which is symmetric in $g,h$.</p>
</details>
<p>The Fokker–Planck form also shows <em>why</em> the process converges: along the flow, $\frac{\d}{\d t}\KL(\rho_t\|\pi) = -\frac12\int\rho_t\,\|\nabla\log(\rho_t/\pi)\|^2 \le 0$, with equality only at $\rho_t = \pi$. Langevin dynamics is gradient flow of the KL divergence in the Wasserstein geometry (Jordan, Kinderlehrer &amp; Otto, 1998), a fact that connects MCMC to <a href="#/t/variational-inference">variational inference</a>.</p>
<div class="thm" data-name="Exponential convergence under strong log-concavity">
<p>If $U$ is $m$-strongly convex ($\nabla^2 U\succeq mI$), then for two solutions driven by the same Brownian motion, $\|X_t - Y_t\| \le e^{-mt/2}\|X_0 - Y_0\|$. Consequently $W_2(\mathcal L(X_t),\pi)\le e^{-mt/2}\,W_2(\mathcal L(X_0),\pi)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>With a shared Brownian motion the noise cancels in the difference: $\frac{\d}{\d t}(X_t-Y_t) = -\frac12(\nabla U(X_t) - \nabla U(Y_t))$. Hence $\frac{\d}{\d t}\|X_t-Y_t\|^2 = -(X_t-Y_t)\cdot(\nabla U(X_t)-\nabla U(Y_t))\le -m\|X_t-Y_t\|^2$ by strong convexity, and Grönwall gives $\|X_t-Y_t\|^2\le e^{-mt}\|X_0-Y_0\|^2$. Starting $Y_0\sim\pi$ (so $Y_t\sim\pi$ for all $t$) with $(X_0,Y_0)$ optimally coupled yields the $W_2$ bound.</p>
</details>`,
      },
      {
        title: 'Discretization: the unadjusted Langevin algorithm',
        html: R`
<p>The Euler–Maruyama discretization with step $h$ is
$$X_{k+1} = X_k + \frac h2\nabla\log\pi(X_k) + \sqrt h\,\xi_k,\qquad \xi_k\iid\N(0, I_d).$$
Run as a Markov chain without correction, this is <strong>ULA</strong>. Its stationary distribution $\pi_h$ (if one exists) is <em>not</em> $\pi$.</p>
<div class="ex" data-name="ULA on a Gaussian: exact bias">
<p>Take $\pi = \N(0, \sigma^2)$ in one dimension, so $\nabla\log\pi(x) = -x/\sigma^2$ and $X_{k+1} = (1 - \frac{h}{2\sigma^2})X_k + \sqrt h\,\xi_k$, an AR(1). It is stable iff $|1-\frac{h}{2\sigma^2}|\lt 1$, i.e. $h\lt 4\sigma^2$, and then its stationary law is Gaussian with variance
$$v_h = \frac{h}{1-(1-\frac{h}{2\sigma^2})^2} = \frac{h}{\frac{h}{\sigma^2} - \frac{h^2}{4\sigma^4}} = \frac{\sigma^2}{1 - h/(4\sigma^2)}.$$
So ULA inflates the variance by a factor $1/(1-h/4\sigma^2) = 1 + \frac{h}{4\sigma^2} + O(h^2)$: a first-order bias in $h$. At $h = 2\sigma^2$ the variance is doubled; for $h\ge 4\sigma^2$ the chain explodes.</p>
</div>
<p>In general, under smoothness and strong log-concavity, the bias of $\pi_h$ vanishes as $h\to 0$, with explicit non-asymptotic bounds in total variation and Wasserstein distance that depend polynomially on $d$ (Dalalyan, 2017; Durmus &amp; Moulines, 2017). These results made ULA popular in machine learning: it is an optimization-like iteration with a guaranteed approximation error. But two failure modes are qualitative, not quantitative:</p>
<div class="warn" data-name="Transience for light tails">
<p>If $\pi\propto e^{-x^4/4}$, then $\nabla\log\pi(x) = -x^3$ and the ULA map is $x\mapsto x - \frac h2 x^3 + \sqrt h\xi$. For $|x|$ large, $|x - \frac h2x^3|\approx\frac h2|x|^3\gg|x|$: the iteration overshoots ever more violently. The chain is transient for every $h>0$ (Roberts &amp; Tweedie, 1996): with positive probability the noise sends it far enough out that it never returns. Gradients that grow faster than linearly make explicit discretizations unstable; this is the light-tailed target in the lab.</p>
</div>`,
      },
      {
        title: 'MALA: Metropolis-adjusting the Langevin step',
        html: R`
<p>Use the ULA step as a <a href="#/t/metropolis-hastings">Metropolis–Hastings</a> proposal:
$$q(x, y) = \N\Big(y;\; x + \tfrac h2\nabla\log\pi(x),\; hI\Big) \propto \exp\Big(-\frac{1}{2h}\big\|y - x - \tfrac h2\nabla\log\pi(x)\big\|^2\Big).$$</p>
<div class="algo" data-name="Metropolis-adjusted Langevin algorithm (MALA)">
<ol>
<li>Propose $Y = x + \frac h2\nabla\log\pi(x) + \sqrt h\,\xi$.</li>
<li>Accept with probability $\min\Big\{1, \dfrac{\pi(Y)\,q(Y, x)}{\pi(x)\,q(x,Y)}\Big\}$; otherwise stay.</li>
</ol>
</div>
<p>The proposal is <em>not</em> symmetric (the drift at $Y$ differs from that at $x$), so the $q$-ratio is essential. MALA is exactly $\pi$-invariant by the MH theorem, for every $h$: the Metropolis step removes ULA's bias completely, at the cost of rejections. On the Gaussian example, MALA's histogram matches $\pi$ for every step size in the lab, while ULA's spreads out.</p>
<div class="thm" data-name="Optimal scaling of MALA (Roberts & Rosenthal, 1998)">
<p>For product targets $\pi_d = \prod_{i=1}^d f(x_i)$ with $f$ sufficiently smooth, and step $h = \ell^2 d^{-1/3}$, the (suitably speeded-up) first coordinate converges to a Langevin diffusion whose speed is maximized when the asymptotic acceptance rate is approximately $0.574$.</p>
</div>
<p>Compare <a href="#/t/metropolis-hastings">RWM</a>: step $\propto d^{-1/2}$, $O(d)$ iterations per effective sample, optimal acceptance $0.234$. MALA: step$^2\propto d^{-1/3}$, $O(d^{1/3})$ iterations. The gradient information is what buys the improvement: the drift moves the proposal toward where the mass is, so a larger step survives the accept/reject test. <a href="#/t/hamiltonian-mc">HMC</a> goes further, to $O(d^{1/4})$.</p>
<div class="warn" data-name="MALA's fragile ergodicity">
<p>MALA is not geometrically ergodic for targets with tails lighter than Gaussian (the overshoot of the previous section now produces proposals in regions with $q(Y,x)\approx 0$, so the chain gets stuck in the tails), nor for heavy tails where the gradient vanishes (Roberts &amp; Tweedie, 1996). It is also sensitive to the start: in transient phases far from the mode, gradients are large and the optimal stationary step size can be catastrophically large, so practical MALA needs warm-up with smaller steps. Truncating the drift (the MALTA variant) restores robustness.</p>
</div>`,
        lab: 'langevin-mala-bias',
      },
      {
        title: 'Preconditioning and geometry',
        html: R`
<p>With isotropic noise, the step size is limited by the stiffest direction of $\pi$, exactly as for RWM. For a Gaussian target $\N(\mu,\Sigma)$, stability of ULA requires $h\lt 4\lambda_{\min}(\Sigma)$, while moving along the widest direction needs time of order $\lambda_{\max}(\Sigma)$: the cost scales with the condition number. The remedy is a preconditioner $A$ (a fixed positive definite matrix):
$$\d X_t = \tfrac12 A\nabla\log\pi(X_t)\d t + A^{1/2}\d W_t$$
still has $\pi$ invariant (repeat the Fokker–Planck computation with $\nabla\cdot(A\cdots)$). Choosing $A\approx\Sigma$ (an estimate of the posterior covariance) equalizes the scales. <em>Riemannian manifold MALA</em> (Girolami &amp; Calderhead, 2011) lets $A$ depend on position, typically the inverse Fisher information plus the prior precision, which requires an extra divergence term in the drift to preserve invariance.</p>`,
      },
      {
        title: 'Stochastic-gradient Langevin and connections to machine learning',
        html: R`
<p>For a posterior with $N$ data points, $\nabla\log\pi(\theta) = \nabla\log p(\theta) + \sum_{i=1}^N\nabla\log p(y_i\mid\theta)$ costs $O(N)$. <strong>Stochastic gradient Langevin dynamics</strong> (Welling &amp; Teh, 2011) replaces it with a minibatch estimate $\nabla\log p(\theta) + \frac N n\sum_{i\in B}\nabla\log p(y_i\mid\theta)$ and uses ULA with decreasing step sizes $h_k$ satisfying $\sum h_k=\infty$, $\sum h_k^2\lt\infty$. It is <a href="#/t/sgd-optimizers">SGD</a> with injected Gaussian noise of the right scale. Early on, the minibatch noise dominates and the iteration behaves like SGD; as $h_k\to0$ the injected $\sqrt{h_k}$ noise dominates the $h_k$-scaled gradient noise, and the iterates approach posterior samples. With a fixed step size, SGLD has an irreducible bias, and no Metropolis correction is possible without full-data evaluations.</p>
<p>Langevin dynamics also powers <a href="#/t/diffusion-models">score-based diffusion models</a>: if a neural network approximates the score $\nabla\log p_\sigma(x)$ of noise-smoothed data distributions, annealed Langevin dynamics (or the reverse-time SDE) turns noise into samples. The ingredient is the same as here, a score function; the difference is that it is learned rather than computed.</p>
<div class="note" data-name="Underdamped Langevin">
<p>Adding a momentum variable gives the underdamped (kinetic) Langevin equation $\d q = p\d t$, $\d p = -\nabla U(q)\d t - \gamma p\d t + \sqrt{2\gamma}\d W$, whose invariant law is $\pi(q)\N(p;0,I)$. Its limits are the overdamped diffusion ($\gamma\to\infty$, with time rescaled) and pure <a href="#/t/hamiltonian-mc">Hamiltonian dynamics</a> ($\gamma\to0$). HMC with partial momentum refreshment interpolates between them.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Verify that $\pi$ is stationary for $\d X = \frac12\nabla\log\pi\,\d t + \d W$ using the Fokker–Planck equation.`,
        a: R`$\partial_t\rho = -\nabla\cdot(\frac12\rho\nabla\log\pi) + \frac12\Delta\rho$. At $\rho = \pi$: $\pi\nabla\log\pi = \nabla\pi$, so the right side is $-\frac12\Delta\pi + \frac12\Delta\pi = 0$.` },
      { q: R`ULA with $h = 0.4$ is run on $\N(0, 1)$. What variance does its stationary distribution have? What does MALA with the same $h$ produce?`,
        a: R`$v_h = 1/(1 - h/4) = 1/0.9 \approx 1.111$, an 11% overestimate. MALA is exactly $\N(0,1)$-invariant for any $h$; the step only affects its acceptance rate and mixing.` },
      { q: R`Why must the MALA acceptance ratio include $q(Y, x)/q(x, Y)$ while RWM's does not?`,
        a: R`RWM proposals are symmetric, $q(x,y) = q(y,x)$. MALA's proposal mean $x + \frac h2\nabla\log\pi(x)$ depends on the starting point's gradient, so the reverse move from $Y$ has a different mean and $q(Y,x)\neq q(x,Y)$ in general.` },
      { q: R`How do the step size and iterations-per-effective-sample scale with dimension for RWM, MALA, and HMC on IID product targets, and what are the optimal acceptance rates?`,
        a: R`RWM: step $\propto d^{-1/2}$, $O(d)$ iterations, acceptance $0.234$. MALA: step size $h\propto d^{-1/3}$ (proposal sd $\propto d^{-1/6}$), $O(d^{1/3})$ iterations, acceptance $0.574$. HMC: leapfrog step $\propto d^{-1/4}$, $O(d^{1/4})$, acceptance $0.651$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'hamiltonian-mc',
    domain: 'mcmc',
    order: 70,
    title: 'Hamiltonian Monte Carlo & NUTS',
    subtitle: 'Long, nearly energy-conserving trajectories that the Metropolis test almost always accepts',
    level: 3,
    sources: { other: 'Duane, Kennedy, Pendleton & Roweth (1987); Neal (2011), MCMC using Hamiltonian dynamics; Hoffman & Gelman (2014); Betancourt (2017); Beskos et al. (2013)' },
    prereqs: ['metropolis-hastings'],
    related: ['langevin-mala', 'mcmc-diagnostics', 'hierarchical-mcmc', 'slice-sampling', 'variational-inference'],
    tags: ['Hamiltonian dynamics', 'leapfrog', 'symplectic', 'NUTS', 'divergences', 'mass matrix'],
    summary: R`HMC treats $-\log\pi(q)$ as a potential energy, attaches a Gaussian momentum $p$, and moves by simulating Hamiltonian dynamics. Exact dynamics conserve energy, preserve volume, and are time-reversible; the leapfrog integrator keeps the last two properties exactly and the first approximately, which is precisely what makes the Metropolis correction both valid (volume preservation + reversibility) and nearly always accepted (small energy error). The result is distant proposals with high acceptance, costing $O(d^{1/4})$ gradient evaluations per effective sample. The No-U-Turn Sampler chooses trajectory lengths automatically, and when the integrator fails in regions of high curvature, HMC tells you: the energy error explodes and the transition is flagged as <em>divergent</em>.`,
    survey: {
      task: R`Sample smooth, continuous targets in moderate to high dimension`,
      assumptions: R`$\log\pi$ differentiable on $\R^d$ (constrained parameters transformed); curvature not too variable`,
      output: 'Weakly correlated draws, plus diagnostics (divergences, energy, tree depth)',
      strengths: R`Long-range moves; $O(d^{1/4})$ scaling; robust defaults via NUTS + adaptation; built-in failure signal`,
      weaknesses: R`Needs gradients; no discrete parameters; struggles with multimodality and varying curvature (funnels)`,
      complexity: R`$L$ gradient evaluations per iteration; NUTS uses up to $2^{\text{depth}}$`,
    },
    sections: [
      {
        title: 'Hamiltonian dynamics on an augmented space',
        html: R`
<p>Write the target as $\pi(q)\propto e^{-U(q)}$ with <em>potential energy</em> $U = -\log\tilde\pi$, and introduce an independent momentum $p\sim\N(0, M)$ with kinetic energy $K(p) = \frac12p^\top M^{-1}p$. The joint density on phase space is
$$\pi(q,p)\propto e^{-H(q,p)},\qquad H(q,p) = U(q) + K(p),$$
whose $q$-marginal is $\pi$. Hamilton's equations define a flow $\Phi_t$ on $(q,p)$:
$$\frac{\d q}{\d t} = \frac{\partial H}{\partial p} = M^{-1}p,\qquad \frac{\d p}{\d t} = -\frac{\partial H}{\partial q} = -\nabla U(q).$$
Physically: a frictionless puck on the surface $U$, sliding downhill into high-probability regions, gaining kinetic energy, and coasting back up the other side.</p>
<div class="thm" data-name="Three properties of the Hamiltonian flow">
<p>(i) <strong>Energy conservation</strong>: $H(\Phi_t(q,p)) = H(q,p)$. (ii) <strong>Volume preservation</strong> (Liouville): $|\det D\Phi_t| = 1$. (iii) <strong>Reversibility</strong>: with $F(q,p) = (q,-p)$ the momentum flip, $F\circ\Phi_t\circ F = \Phi_t^{-1}$. Consequently $\Phi_t$ maps $e^{-H}$ to itself: the exact flow leaves $\pi(q,p)$ invariant.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(i) $\frac{\d H}{\d t} = \nabla_q H\cdot\dot q + \nabla_p H\cdot\dot p = \nabla_q H\cdot\nabla_p H - \nabla_p H\cdot\nabla_q H = 0$.</p>
<p>(ii) The vector field $v = (\nabla_p H, -\nabla_q H)$ has divergence $\sum_i\big(\partial_{q_i}\partial_{p_i}H - \partial_{p_i}\partial_{q_i}H\big) = 0$. By Liouville's formula $\frac{\d}{\d t}\det D\Phi_t = (\nabla\cdot v)\det D\Phi_t = 0$, and $\det D\Phi_0 = 1$.</p>
<p>(iii) If $(q(t),p(t))$ solves Hamilton's equations, so does $(\tilde q(t),\tilde p(t)) = (q(-t), -p(-t))$: $\dot{\tilde q}(t) = -\dot q(-t) = -M^{-1}p(-t) = M^{-1}\tilde p(t)$ and $\dot{\tilde p}(t) = \dot p(-t) = -\nabla U(\tilde q(t))$, using that $H$ is even in $p$. So running time forward from the flipped endpoint retraces the path: $\Phi_t(F\Phi_t(z)) = F(z)$, i.e. $F\Phi_tF = \Phi_t^{-1}$.</p>
<p>Invariance: for $z' = \Phi_t(z)$, the change-of-variables formula gives the pushforward density $e^{-H(\Phi_t^{-1}z')}/|\det D\Phi_t| = e^{-H(z')}$ by (i) and (ii).</p>
</details>
<p>If we could integrate exactly, the algorithm would be: draw $p\sim\N(0,M)$ (a Gibbs step for the momentum), flow for time $T$, and keep $q$. No accept/reject would be needed. Since $H$ is conserved, each trajectory stays on one energy level set; the momentum refresh is what moves the chain <em>between</em> level sets, and the interplay of the two determines how well HMC explores the energy distribution (Betancourt, 2017).</p>`,
      },
      {
        title: 'The leapfrog integrator and the Metropolis correction',
        html: R`
<p>Exact flows are unavailable, so HMC uses the <strong>leapfrog</strong> (Störmer–Verlet) scheme with step $\varepsilon$:
$$p_{1/2} = p - \tfrac\varepsilon2\nabla U(q),\qquad q' = q + \varepsilon M^{-1}p_{1/2},\qquad p' = p_{1/2} - \tfrac\varepsilon2\nabla U(q').$$
Composing $L$ steps (adjacent half-steps merge, so the cost is $L$ gradient evaluations) gives the map $\Psi_{\varepsilon,L}$.</p>
<div class="thm" data-kind="Proposition" data-name="Leapfrog is volume-preserving and reversible">
<p>Each leapfrog step has Jacobian determinant 1 and satisfies $F\circ\Psi\circ F = \Psi^{-1}$. It does <em>not</em> conserve $H$ exactly, but it is symplectic, so it exactly conserves a "shadow" Hamiltonian $\tilde H = H + O(\varepsilon^2)$ up to exponentially small terms over exponentially long times (backward error analysis; Hairer, Lubich &amp; Wanner). The energy error therefore stays $O(\varepsilon^2)$ along a trajectory instead of drifting.</p>
</div>
<details class="proof"><summary>Proof of volume preservation and reversibility</summary>
<p>Each of the three substeps is a <em>shear</em>: it changes one block of variables by an amount depending only on the other block. E.g. $(q,p)\mapsto(q, p - \frac\varepsilon2\nabla U(q))$ has Jacobian $\begin{pmatrix} I & 0\\ * & I\end{pmatrix}$, determinant 1. Compositions of determinant-1 maps have determinant 1. For reversibility, the step is a palindrome: kick$(\varepsilon/2)$, drift$(\varepsilon)$, kick$(\varepsilon/2)$. Conjugating a kick or a drift by $F$ gives its inverse (flipping $p$ reverses a drift; $F\circ\text{kick}\circ F$ subtracts instead of adds the gradient term on the flipped momentum, which is the inverse kick). Hence $F\Psi F = \text{kick}^{-1}\text{drift}^{-1}\text{kick}^{-1}$ in the same palindromic order, which equals $\Psi^{-1}$.</p>
</details>
<p>Now combine: the proposal map $\Psi^* = F\circ\Psi_{\varepsilon,L}$ (integrate, then flip the momentum) is a volume-preserving <strong>involution</strong>: $\Psi^*\circ\Psi^* = \text{id}$ by reversibility.</p>
<div class="thm" data-name="MH with a deterministic volume-preserving involution">
<p>Let $\Psi^*$ be a volume-preserving involution of phase space. The kernel that moves $z\mapsto\Psi^*(z)$ with probability $\alpha(z) = \min\{1, e^{-H(\Psi^* z) + H(z)}\}$ and otherwise stays at $z$ satisfies detailed balance with respect to $\pi(z)\propto e^{-H(z)}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The rejection part is symmetric as in the <a href="#/t/metropolis-hastings">MH proof</a>. For the move part, we need $\int_A\pi(z)\alpha(z)\ind_B(\Psi^*z)\d z = \int_B\pi(w)\alpha(w)\ind_A(\Psi^*w)\d w$. Substitute $z = \Psi^*w$ (valid since $\Psi^*$ is its own inverse) with $|\det D\Psi^*| = 1$:
$$\int_A\pi(z)\alpha(z)\ind_B(\Psi^*z)\d z = \int\pi(\Psi^*w)\,\alpha(\Psi^*w)\,\ind_A(\Psi^*w)\,\ind_B(w)\d w.$$
And $\pi(\Psi^*w)\alpha(\Psi^*w) = \min\{\pi(\Psi^*w), \pi(w)\} = \pi(w)\alpha(w)$. Hence the two sides agree.</p>
</details>
<p>Since $H$ is even in $p$, the final flip does not affect the acceptance probability and can be dropped, because the next iteration resamples $p$ anyway. The complete HMC iteration is a Gibbs refresh of $p$ followed by this MH step; both leave $\pi(q,p)$ invariant. <strong>This is why HMC works so well</strong>: volume preservation removes any Jacobian from the acceptance ratio, and near-conservation of $H$ makes $e^{-\Delta H}\approx 1$ even for proposals far from the start.</p>
<div class="ex" data-name="Leapfrog on a Gaussian: stability and the shadow Hamiltonian">
<p>For $U(q) = q^2/2$ ($\N(0,1)$), $M=1$, one leapfrog step is linear: $q' = (1-\frac{\varepsilon^2}2)q + \varepsilon p$, $p' = -\varepsilon(1-\frac{\varepsilon^2}{4})q + (1-\frac{\varepsilon^2}2)p$. The matrix has determinant 1 and trace $2-\varepsilon^2$, so its eigenvalues lie on the unit circle iff $|2-\varepsilon^2|\lt 2$, i.e. $\varepsilon \lt 2$; for $\varepsilon\gt2$ they are real with one exceeding 1 and trajectories blow up exponentially. A direct check shows the quadratic form $\tilde H = \frac12p^2 + \frac12(1-\frac{\varepsilon^2}{4})q^2$ is <em>exactly</em> conserved, so $H - \tilde H = \frac{\varepsilon^2}{8}q^2$: the energy error oscillates with amplitude $O(\varepsilon^2)$ and never accumulates. For a Gaussian with standard deviations $\sigma_i$, stability needs $\varepsilon\lt 2\min_i\sigma_i$: the narrowest direction caps the step size.</p>
</div>`,
        lab: 'hamiltonian-mc-integrator',
      },
      {
        title: 'Tuning: step size, trajectory length, and mass matrix',
        html: R`
<p><strong>Step size $\varepsilon$.</strong> Too large and $\Delta H$ grows, acceptance collapses, and beyond the stability limit trajectories diverge; too small and each trajectory costs many gradients. Beskos, Pillai, Roberts, Sanz-Serna &amp; Stuart (2013) showed that for product targets $\varepsilon$ should scale as $d^{-1/4}$, with optimal acceptance about $0.651$; in practice a target of $0.8$ (Stan's default <code>adapt_delta</code>) is more robust for non-product targets. Stan tunes $\varepsilon$ during warm-up by dual averaging (Nesterov's primal-dual method applied to the acceptance statistic).</p>
<p><strong>Trajectory length $T = L\varepsilon$.</strong> Too short and HMC degenerates to a Langevin-like random walk ($L=1$ HMC <em>is</em> MALA); too long and trajectories curve back on themselves (for a Gaussian, the motion is periodic), wasting computation and possibly returning close to the start. The ideal length differs across the target, so no single $L$ is right.</p>
<p><strong>Mass matrix $M$.</strong> The momentum covariance acts as a preconditioner: choosing $M^{-1}\approx\Cov_\pi(q)$ makes the dynamics see an approximately isotropic target, which equalizes the stability limits across directions. Stan estimates a diagonal (or dense) $M^{-1}$ from warm-up draws.</p>
<div class="algo" data-name="The No-U-Turn Sampler (Hoffman & Gelman, 2014), in outline">
<ol>
<li>Refresh $p$. Build a trajectory by repeated doubling: at each stage, choose a direction (forward or backward in time) uniformly and extend the trajectory by as many leapfrog steps as it already has.</li>
<li>Stop when the trajectory starts to turn back on itself: when, for the extreme states $(q^-,p^-)$ and $(q^+,p^+)$, $(q^+-q^-)\cdot p^- \lt 0$ or $(q^+-q^-)\cdot p^+\lt 0$ (checked also on all sub-trees), or when a divergence occurs, or a maximum tree depth is reached.</li>
<li>Select the next state from the trajectory's points with probabilities proportional to $e^{-H}$ (multinomial sampling, with a bias toward the most recently added subtree), in a way that preserves detailed balance because the trajectory-building process is symmetric with respect to which point one started from.</li>
</ol>
</div>
<p>NUTS removes the most delicate tuning parameter and adapts $T$ locally. Combined with step-size dual averaging and mass-matrix adaptation, it gives the "push-button" HMC of Stan, PyMC, and NumPyro.</p>`,
      },
      {
        title: 'Divergences: when HMC tells you it failed',
        html: R`
<p>In regions where the curvature of $U$ is much larger than the step size can resolve (the neck of a <a href="#/t/hierarchical-mcmc">hierarchical funnel</a>, a boundary where $U\to\infty$), the leapfrog integrator becomes unstable: the energy error grows without bound within a few steps. Implementations flag a transition as <strong>divergent</strong> when $H$ along the trajectory exceeds its initial value by a large threshold (1000 in Stan).</p>
<div class="warn" data-name="Divergences mean biased samples">
<p>A divergence is not a numerical nuisance; it is evidence that there is a region of the posterior the sampler cannot enter with this step size. Draws are then biased away from that region, often precisely the region of scientific interest (small group-level variances). Because the error is a missing region, the bias does not show up in $\hat R$ or ESS. Remedies, in order: reparameterize (e.g., <a href="#/t/hierarchical-mcmc">non-centering</a>), increase <code>adapt_delta</code> (smaller $\varepsilon$), which helps only if the problem is mild, or reconsider the model (stronger priors).</p>
</div>
<p>Divergences are one of HMC's great practical advantages over random-walk methods: a random-walk chain stuck outside a narrow region simply does not visit it, silently. HMC's integrator error is a geometric probe of the target, and the diagnostic comes for free.</p>`,
        lab: 'hamiltonian-mc-trajectory',
      },
      {
        title: 'Limitations and extensions',
        html: R`
<ul>
<li><strong>No discrete parameters.</strong> HMC needs gradients on $\R^d$. Discrete latent variables must be marginalized (mixtures: sum over labels in the likelihood) or handled with other kernels in a Gibbs-type composition.</li>
<li><strong>Constraints.</strong> Positive or bounded parameters are transformed to $\R$ (log, logit), with the log-Jacobian added to $\log\pi$.</li>
<li><strong>Multimodality.</strong> Energy conservation means a trajectory cannot climb an energy barrier higher than its kinetic energy, which is about $\chi^2_d/2$. HMC crosses modest barriers far better than RWM but does not solve multimodality; see <a href="#/t/tempering">tempering</a>.</li>
<li><strong>Riemannian HMC</strong> (Girolami &amp; Calderhead, 2011) uses a position-dependent mass matrix $M(q)$ adapted to local curvature, requiring an implicit (generalized leapfrog) integrator. It handles funnels better but is expensive.</li>
<li><strong>Heavy tails.</strong> With heavy-tailed targets, trajectories reaching the tails have tiny gradients and take very long to return; HMC is not geometrically ergodic for polynomial tails (Livingstone, Betancourt, Byrne &amp; Girolami, 2019).</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Why does the leapfrog integrator's volume preservation matter for the acceptance probability?`,
        a: R`For a deterministic proposal map $\Psi^*$, the MH ratio contains the Jacobian $|\det D\Psi^*|$. Volume preservation makes it 1, so the acceptance probability reduces to $\min\{1, e^{-\Delta H}\}$. A non-volume-preserving integrator (e.g., explicit Euler) would require computing the Jacobian, and without it would be biased.` },
      { q: R`For an isotropic Gaussian target with standard deviation $\sigma$, what is the largest stable leapfrog step size with $M = I$? What happens just beyond it?`,
        a: R`$\varepsilon \lt 2\sigma$. Beyond it, the one-step matrix has real eigenvalues $\lambda, 1/\lambda$ with $|\lambda|\gt 1$, so trajectories grow exponentially, $\Delta H$ explodes, and essentially every proposal is rejected (and flagged as divergent).` },
      { q: R`HMC with $L = 1$ leapfrog step is equivalent to which algorithm?`,
        a: R`MALA. With $p\sim\N(0,I)$, one leapfrog step gives $q' = q - \frac{\varepsilon^2}2\nabla U(q) + \varepsilon p$, a Langevin proposal with $h = \varepsilon^2$, and the HMC acceptance ratio $e^{-\Delta H}$ equals the MALA Hastings ratio.` },
      { q: R`A model fit reports $\hat R = 1.00$, high ESS, and 3% divergent transitions concentrated at small values of a variance parameter. Should you trust the posterior?`,
        a: R`No. Divergences indicate a high-curvature region the integrator cannot resolve; the sampler under-represents it, biasing estimates (typically overestimating the variance parameter). $\hat R$ and ESS cannot detect a region that no chain visits. Reparameterize (e.g., non-centered) or reduce the step size and recheck.` },
      { q: R`Why does NUTS stop when $(q^+ - q^-)\cdot p^+ \lt 0$?`,
        a: R`$\frac{\d}{\d t}\frac12\|q^+ - q^-\|^2 = (q^+ - q^-)\cdot\dot q^+$, which (with $M=I$) is $(q^+-q^-)\cdot p^+$. When it becomes negative, extending the trajectory further would shrink the distance between its ends: the trajectory has started to make a U-turn, so further computation would mostly retrace ground already covered.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'tempering',
    domain: 'mcmc',
    order: 80,
    title: 'Tempering & Multimodal Targets',
    subtitle: 'Flatten the landscape, let hot chains cross barriers, and pass their discoveries down the ladder',
    level: 3,
    sources: { other: 'Swendsen & Wang (1986); Geyer (1991); Marinari & Parisi (1992); Neal (2001); Atchadé, Roberts & Rosenthal (2011); Woodard, Schmidler & Huber (2009)' },
    prereqs: ['metropolis-hastings'],
    related: ['importance-sampling', 'gaussian-mixtures', 'particle-filters', 'hamiltonian-mc', 'slice-sampling'],
    tags: ['parallel tempering', 'replica exchange', 'simulated tempering', 'annealing', 'thermodynamic integration'],
    summary: R`Local samplers cross between well-separated modes with probability exponentially small in the height of the barrier. Tempering raises the target to a power $\beta\lt1$: $\pi^\beta$ has the same modes but lower barriers, so a chain at high "temperature" $1/\beta$ moves freely. Parallel tempering runs chains at a ladder of temperatures simultaneously and proposes swaps between neighbours; the swap acceptance probability $\min\{1, \exp[(\beta_i-\beta_j)(\log\pi(x_j)-\log\pi(x_i))]\}$ makes the product chain exactly invariant, so the cold chain samples $\pi$ while inheriting the hot chains' mobility.`,
    survey: {
      task: 'Sample multimodal or rugged targets; estimate normalizing constants',
      assumptions: R`Tempered targets $\pi^{\beta}$ (or prior $\times$ likelihood$^\beta$) are proper and mix fast at high temperature`,
      output: R`Draws from $\pi$ (cold chain); draws from all $\pi_\beta$ (for thermodynamic integration)`,
      strengths: 'Exact, simple to add on top of any sampler, embarrassingly parallel, crosses energy barriers',
      weaknesses: R`Needs $K\propto\sqrt d$ temperatures; fails when modes have different widths (weights shift with $\beta$)`,
      complexity: R`$K\times$ the cost of a single chain`,
    },
    sections: [
      {
        title: 'Why local samplers fail on multimodal targets',
        html: R`
<p>Consider $\pi = \frac12\N(-\mu, \sigma^2) + \frac12\N(\mu,\sigma^2)$ with $\mu\gg\sigma$. A random-walk or Hamiltonian chain tuned to the width $\sigma$ of each mode must pass through the valley around 0, where the density is lower than at the modes by a factor $e^{-\mu^2/2\sigma^2}$. The expected time between mode switches is roughly the inverse of the valley's relative probability, exponential in the barrier height $\Delta U = \mu^2/2\sigma^2$ (Arrhenius' law of statistical physics). With $\mu/\sigma = 6$, $e^{18}\approx 6.6\times 10^7$.</p>
<p>The chain is still irreducible and geometrically ergodic, so the theory holds, but the spectral gap is exponentially small: convergence is <em>metastable</em>. Worse, within any practical run the chain looks perfectly converged in one mode. Mixture models, Ising-type models at low temperature, phylogenetic trees, and many latent-variable models exhibit this behaviour.</p>
<div class="note" data-name="Flattening by temperature">
<p>For $\beta\in(0,1]$ define the tempered target $\pi_\beta(x)\propto\pi(x)^\beta = e^{-\beta U(x)}$. Barriers shrink: $\beta\Delta U$. At $\beta = 1/36$ in the example above, the barrier is $0.5$ and a random walk crosses easily. The modes are still in the same places, so the hot chain knows <em>where</em> they are; it just doesn't have the right <em>weights</em> or shapes. Tempering methods pass this information down to $\beta=1$ without biasing it.</p>
</div>`,
      },
      {
        title: 'Parallel tempering (replica exchange)',
        html: R`
<p>Choose a ladder $1 = \beta_1\gt\beta_2\gt\dots\gt\beta_K\gt 0$ and target the product distribution on $\mathcal X^K$
$$\Pi(x_1,\dots,x_K) = \prod_{k=1}^K\pi_{\beta_k}(x_k).$$
Its first marginal is $\pi$.</p>
<div class="algo" data-name="Parallel tempering (Swendsen & Wang, 1986; Geyer, 1991)">
<ol>
<li><strong>Local moves.</strong> Update each $x_k$ with any $\pi_{\beta_k}$-invariant kernel (RWM with scale tuned per temperature, HMC, Gibbs).</li>
<li><strong>Swap moves.</strong> Pick a pair of adjacent temperatures $(i, j = i+1)$ and propose to exchange their states, $(x_i, x_j)\to(x_j, x_i)$. Accept with probability
$$\alpha_{\text{swap}} = \min\Big\{1, \frac{\pi_{\beta_i}(x_j)\,\pi_{\beta_j}(x_i)}{\pi_{\beta_i}(x_i)\,\pi_{\beta_j}(x_j)}\Big\} = \min\Big\{1, \exp\big[(\beta_i - \beta_j)\big(\log\pi(x_j) - \log\pi(x_i)\big)\big]\Big\}.$$</li>
<li>Record the $\beta_1 = 1$ chain as draws from $\pi$.</li>
</ol>
</div>
<div class="thm" data-kind="Proposition" data-name="Swaps satisfy detailed balance on the product space">
<p>The swap move is $\Pi$-reversible, and normalizing constants of the tempered densities cancel. Hence the parallel-tempering chain is $\Pi$-invariant and its cold component has invariant law $\pi$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The swap is a Metropolis–Hastings move on $\mathcal X^K$ with a deterministic proposal $S_{ij}$ (transpose coordinates $i$ and $j$). $S_{ij}$ is an involution and a permutation of coordinates, hence volume-preserving, so by the <a href="#/t/hamiltonian-mc">involution lemma</a> (or directly: the proposal is symmetric) the acceptance probability is $\min\{1, \Pi(S_{ij}x)/\Pi(x)\}$. All factors with $k\ne i,j$ cancel, leaving
$$\frac{\Pi(S_{ij}x)}{\Pi(x)} = \frac{\pi_{\beta_i}(x_j)\pi_{\beta_j}(x_i)}{\pi_{\beta_i}(x_i)\pi_{\beta_j}(x_j)} = \frac{\pi(x_j)^{\beta_i}\pi(x_i)^{\beta_j}}{\pi(x_i)^{\beta_i}\pi(x_j)^{\beta_j}},$$
in which the normalizing constants $Z(\beta_i), Z(\beta_j)$ appear once in numerator and denominator and cancel. Taking logs gives the stated exponent. The local moves are $\Pi$-invariant because each updates one coordinate with a kernel invariant for that coordinate's factor. Compositions of $\Pi$-invariant kernels are $\Pi$-invariant.</p>
</details>
<p>Interpretation: if the hotter chain $j$ currently sits at a better point ($\log\pi(x_j)\gt\log\pi(x_i)$) the swap is always accepted, so good states discovered by hot chains flow down to cold ones, and cold chains' states flow up to be "re-mixed". A useful mental model is to follow one <em>replica</em> (state) as it random-walks up and down the ladder: each round trip from cold to hot and back gives the cold chain a fresh, nearly independent state.</p>`,
        lab: 'tempering-ladder',
      },
      {
        title: 'Designing the temperature ladder',
        html: R`
<p>Swaps between $\beta$ and $\beta' = \beta - \delta$ succeed only if typical states of $\pi_\beta$ and $\pi_{\beta'}$ overlap. Write $E = -\log\pi(x)$ for the energy. Under $\pi_\beta$, $E$ has mean $\mu(\beta)$ and variance $s^2(\beta) = -\mu'(\beta)$. The log acceptance ratio for a swap is $\delta\,(E_{\text{cold}} - E_{\text{hot}})$, whose mean is about $-\delta^2 s^2$ and whose spread is about $\delta s\sqrt2$, so acceptance stays bounded away from zero iff $\delta\,s(\beta) = O(1)$.</p>
<ul>
<li>For a $d$-dimensional target with roughly independent components, $s^2(\beta)\propto d/\beta^2$ (each degree of freedom contributes $\frac{1}{2\beta^2}$ for Gaussian-like modes), so spacing must be $\delta\propto\beta/\sqrt d$ and the number of temperatures needed to span $[\beta_{\min},1]$ grows like $\sqrt d\,\log(1/\beta_{\min})$. Geometric spacing of $\beta$ is therefore natural.</li>
<li>Atchadé, Roberts &amp; Rosenthal (2011) showed, under product-form assumptions, that the efficiency of the ladder is maximized when the swap acceptance rate between neighbours is about $0.234$, echoing the <a href="#/t/metropolis-hastings">RWM scaling result</a>. Adaptive schemes tune spacings toward a target swap rate during warm-up.</li>
<li>$\beta_{\min}$ should be small enough that the hottest chain crosses barriers rapidly, and not much smaller: each extra temperature costs a chain.</li>
</ul>
<div class="warn" data-name="Modes of different widths">
<p>Tempering preserves mode <em>locations</em> but not mode <em>weights</em>. If one mode is narrow and tall and another broad and low, then as $\beta\to0$ the broad mode's share of $\pi^\beta$ grows, because the integral of $\pi^\beta$ over a mode scales like height$^\beta\times$width. The hot chains then spend almost no time in the narrow mode, and information about it cannot reach the cold chain. Woodard, Schmidler &amp; Huber (2009) prove that parallel and simulated tempering can be torpidly (exponentially slowly) mixing in such cases. In $d$ dimensions the hot-chain mass of a mode scales like $(\text{width})^{d(1-\beta)}$, so the imbalance grows exponentially with dimension. In the one-dimensional lab, the "narrow + wide" option makes the narrow mode visibly shrink on the hot rungs; the cold chain still copes, but with fewer mode switches.</p>
</div>`,
      },
      {
        title: 'Simulated tempering, annealing, and normalizing constants',
        html: R`
<p><strong>Simulated tempering</strong> (Marinari &amp; Parisi, 1992) uses a single chain on $(x, k)$ with target $\propto\pi(x)^{\beta_k}c_k$, moving $k$ up and down by MH. To spend comparable time at each level, the pseudo-prior weights $c_k$ must approximate $1/Z(\beta_k)$, which must be learned (e.g., by Wang–Landau or stochastic approximation). It needs only one chain, but the constants are a nuisance.</p>
<p><strong>Annealed importance sampling</strong> (Neal, 2001) and <strong>sequential Monte Carlo samplers</strong> (Del Moral, Doucet &amp; Jasra, 2006) instead move a population of particles through $\beta_K\to\dots\to\beta_1$, applying an MCMC kernel at each level and reweighting by $\pi(x)^{\beta_{k-1}-\beta_k}$. The weights give an unbiased estimate of $Z(1)/Z(\beta_K)$; resampling (as in <a href="#/t/particle-filters">particle filters</a>) controls weight degeneracy.</p>
<div class="thm" data-name="Thermodynamic integration">
<p>For the power posterior $\pi_\beta(\theta)\propto p(y\mid\theta)^\beta p(\theta)$ with normalizer $Z(\beta) = \int p(y\mid\theta)^\beta p(\theta)\d\theta$, we have $Z(0) = 1$, $Z(1) = p(y)$, and
$$\log p(y) = \int_0^1 \E_{\pi_\beta}\big[\log p(y\mid\theta)\big]\d\beta.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Differentiate under the integral: $\frac{\d}{\d\beta}\log Z(\beta) = \frac{1}{Z(\beta)}\int\log p(y\mid\theta)\,p(y\mid\theta)^\beta p(\theta)\d\theta = \E_{\pi_\beta}[\log p(y\mid\theta)]$. Integrate from 0 to 1 and use $\log Z(0) = 0$.</p>
</details>
<p>The parallel-tempering chains at the ladder's temperatures provide exactly the expectations needed, so the marginal likelihood for Bayesian model comparison comes almost for free (Friel &amp; Pettitt, 2008). Note that tempering only the likelihood keeps every $\pi_\beta$ proper as long as the prior is.</p>
<p><strong>Simulated annealing</strong> (Kirkpatrick, Gelatt &amp; Vecchi, 1983) runs a single chain while increasing $\beta\to\infty$, concentrating on the global <em>maximum</em>. It is an optimization method, not a sampler: its output is not distributed as $\pi$.</p>`,
      },
      {
        title: 'Pitfalls and practice',
        html: R`
<div class="warn" data-name="Only the cold chain is a sample from π">
<p>Hot chains sample $\pi^{\beta}$, not $\pi$. Their draws can be reused by importance reweighting with weights $\propto\pi(x)^{1-\beta}$, but these weights are typically highly variable. Report the cold chain.</p>
</div>
<div class="warn" data-name="Diagnosing the ladder">
<p>Check the swap acceptance rate for every adjacent pair (a near-zero rate anywhere breaks the communication of the whole ladder) and the number of replica round trips. Few round trips means the cold chain's mode proportions rest on few independent visits, even if each cold-chain segment looks well mixed.</p>
</div>
<p>In practice, parallel tempering is the standard tool for phylogenetics (MrBayes's "Metropolis-coupled MCMC"), for spin glasses, and for multimodal posteriors in astrophysics and cosmology. For moderate multimodality in continuous models, using several chains from dispersed starting points and checking with <a href="#/t/mcmc-diagnostics">$\hat R$</a> is often the first line of defence, but it reveals multimodality rather than resolving it: chains in different modes give $\hat R\gg1$ and no information about the modes' relative weights.</p>`,
      },
    ],
    quiz: [
      { q: R`Derive the swap acceptance probability for exchanging states between temperatures $\beta_i$ and $\beta_j$, and explain why $Z(\beta)$ is not needed.`,
        a: R`The product target is $\prod_k\pi(x_k)^{\beta_k}/Z(\beta_k)$. The swap is a symmetric deterministic proposal, so $\alpha = \min\{1, \Pi(\text{swapped})/\Pi(\text{current})\} = \min\{1, \pi(x_j)^{\beta_i}\pi(x_i)^{\beta_j}/[\pi(x_i)^{\beta_i}\pi(x_j)^{\beta_j}]\}$. Each $Z(\beta_k)$ appears once in numerator and denominator because each temperature still holds exactly one state after the swap.` },
      { q: R`A bimodal target has barrier height $\Delta U = 20$ (in units of $\log\pi$). At what $\beta$ does the barrier fall to about 1, and why is it not enough to run a single chain at that $\beta$?`,
        a: R`$\beta\approx 1/20$. A single chain at $\beta = 0.05$ samples $\pi^{0.05}$, not $\pi$: the modes are much wider and their relative weights can differ from $\pi$'s. The ladder plus swaps is what transfers the hot chain's mobility to a chain that samples $\pi$ exactly.` },
      { q: R`Why does the number of temperatures need to grow with dimension?`,
        a: R`The energy's standard deviation under $\pi_\beta$ grows like $\sqrt d/\beta$. Swap acceptance between neighbours stays reasonable only if $\Delta\beta\cdot\text{sd}(E) = O(1)$, so $\Delta\beta\propto\beta/\sqrt d$, and the number of rungs spanning a fixed range grows like $\sqrt d$.` },
      { q: R`Give the thermodynamic integration identity for $\log p(y)$ and say where its integrand comes from in a parallel-tempering run with power posteriors.`,
        a: R`$\log p(y) = \int_0^1\E_{\pi_\beta}[\log p(y\mid\theta)]\d\beta$ with $\pi_\beta\propto p(y\mid\theta)^\beta p(\theta)$. Each rung's chain gives a Monte Carlo estimate of $\E_{\pi_{\beta_k}}[\log p(y\mid\theta)]$, and the integral over $\beta$ is approximated by a quadrature rule over the ladder.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'hierarchical-mcmc',
    domain: 'mcmc',
    order: 90,
    title: 'Hierarchical Models in Practice: Funnels & Reparameterization',
    subtitle: 'The same model, written two ways, can be easy or impossible to sample',
    level: 3,
    sources: { other: 'Neal (2003); Papaspiliopoulos, Roberts & Sköld (2007); Betancourt & Girolami (2015); Gelman et al., Bayesian Data Analysis, Ch. 5 & 12; Rubin (1981)' },
    prereqs: ['hamiltonian-mc', 'gibbs-sampling'],
    related: ['mixed-models', 'glmm-gee', 'mcmc-diagnostics', 'bayesian-inference', 'variational-inference'],
    tags: ["Neal's funnel", 'centered', 'non-centered', 'eight schools', 'partial pooling', 'divergences'],
    summary: R`In a hierarchical model, group-level effects $\theta_j\sim\N(\mu,\tau^2)$ are tied together by a population scale $\tau$. When $\tau$ is small the $\theta_j$ are squeezed together, when it is large they spread out, so the posterior of $(\tau,\theta)$ has the shape of a funnel whose curvature changes by orders of magnitude. No single step size fits both the neck and the mouth, samplers avoid the neck, and estimates of $\tau$ are biased upward. Writing $\theta_j = \mu + \tau z_j$ with $z_j\sim\N(0,1)$ (the <em>non-centered</em> parameterization) makes the prior geometry isotropic. Which parameterization is better depends on how informative the data are: centered for strong data, non-centered for weak data.`,
    survey: {
      task: R`Sample posteriors of multilevel models $\theta_j\sim\N(\mu,\tau^2)$, $y_j\mid\theta_j$`,
      assumptions: R`Gaussian (or approximately) random effects; ability to rewrite the model`,
      output: 'Unbiased posterior draws for both group effects and variance components',
      strengths: 'Reparameterization is free, exact, and often turns divergent fits into clean ones',
      weaknesses: 'Right choice depends on data informativeness, which can differ across groups',
      complexity: 'Same cost per gradient; far fewer divergences and higher ESS when chosen well',
    },
    sections: [
      {
        title: 'The model and its geometry',
        html: R`
<p>The canonical two-level normal model (see <a href="#/t/mixed-models">mixed models</a>) has $J$ groups with estimates $y_j$ and known standard errors $\sigma_j$:
$$y_j\mid\theta_j\sim\N(\theta_j,\sigma_j^2),\qquad \theta_j\mid\mu,\tau\sim\N(\mu,\tau^2),\qquad (\mu,\tau)\sim p(\mu)p(\tau).$$
The posterior shrinks each $\theta_j$ toward $\mu$ by an amount controlled by $\tau$: <em>partial pooling</em>. In this <strong>centered</strong> parameterization the sampler works directly with $(\mu,\tau,\theta_1,\dots,\theta_J)$.</p>
<p>Look at the prior part of the log density, which dominates when the data are weak:
$$\log p(\theta,\tau\mid\mu) = -J\log\tau - \frac{1}{2\tau^2}\sum_j(\theta_j-\mu)^2 + \log p(\tau).$$
The conditional standard deviation of each $\theta_j$ is $\tau$, so the Hessian in the $\theta$ directions is $\tau^{-2}I$. As $\tau$ ranges over, say, $[e^{-4}, e^{3}]$, the curvature ranges over a factor $e^{14}\approx 10^6$. Meanwhile the $-J\log\tau$ term pulls mass toward small $\tau$ when the $\theta_j$ are close together. The result is <strong>Neal's funnel</strong>.</p>
<div class="def" data-name="Neal's funnel (Neal, 2003)">
<p>$v\sim\N(0, 3^2)$ and $x_i\mid v\iid\N(0, e^{v})$ for $i=1,\dots,n$. The log density is $-\frac{v^2}{18} - \frac{n v}{2} - \frac{e^{-v}}{2}\sum_i x_i^2$. In the $(x,v)$ plane the density is a funnel: wide for large $v$, extremely narrow for small $v$, with a neck where the conditional standard deviation of $x$ is $e^{v/2}$, about $0.01$ at $v = -9$.</p>
</div>
<p>The funnel is the prior geometry of every centered hierarchical model with $v = 2\log\tau$. Its marginal in $v$ is a simple $\N(0,9)$, so a sampler that fails to reach $v\approx-6$ is visibly biased in a direction we can check, which is what makes it a benchmark.</p>`,
      },
      {
        title: 'Why samplers fail in the neck',
        html: R`
<p>Every local sampler must choose a step size. For <a href="#/t/metropolis-hastings">RWM</a>, a scale appropriate for the mouth ($e^{v/2}\approx 20$ at $v = 6$) is rejected almost always in the neck; a scale for the neck makes exploration of the mouth hopeless. Similarly for <a href="#/t/hamiltonian-mc">HMC</a>: the leapfrog step must satisfy $\varepsilon\lt 2e^{v/2}$ for stability (the Gaussian stability bound), so any step size tuned during warm-up on the bulk of the distribution becomes unstable deep in the neck. The trajectory diverges, the proposal is rejected, and the chain is repelled from the neck.</p>
<div class="warn" data-name="The bias is systematic, not just slow">
<p>Because the chain is repelled from small $\tau$, it does not merely mix slowly there; it under-represents that region for any feasible run length, and estimates of $\tau$ (and of the shrinkage it implies) are biased upward. In HMC the evidence is a cluster of <a href="#/t/hamiltonian-mc">divergent transitions</a> near the neck; in RWM or Gibbs there is often no warning at all, except perhaps long sticky stretches in the trace of $\log\tau$.</p>
</div>
<p>Gibbs sampling has the mirror-image problem (Section 4): in the centered parameterization, $\tau\mid\theta$ and $\theta\mid\tau$ are strongly dependent when data are weak, so the chain moves in tiny steps along the funnel.</p>`,
        lab: 'hierarchical-mcmc-neal',
      },
      {
        title: 'The non-centered parameterization',
        html: R`
<p>Introduce standardized effects $z_j$ and define $\theta_j$ deterministically:
$$z_j\iid\N(0,1),\qquad \theta_j = \mu + \tau z_j,\qquad y_j\mid z_j,\mu,\tau\sim\N(\mu + \tau z_j, \sigma_j^2).$$
This is the same model (the implied joint law of $(\mu,\tau,\theta,y)$ is identical), but the sampler now works in $(\mu,\tau,z)$. A priori $z$ is independent of $\tau$ with unit scale, so the prior geometry is a product of simple distributions: the funnel is gone. For Neal's funnel itself, $x_i = e^{v/2}z_i$ with $z_i\sim\N(0,1)$ turns the target into an isotropic Gaussian in $(z,v)$.</p>
<p>The cost is moved into the likelihood: now $y_j$ depends on the product $\tau z_j$. If the data are <em>informative</em> ($\sigma_j\ll\tau$), the likelihood pins $\theta_j = \mu + \tau z_j$ tightly, which in $(\tau, z_j)$ coordinates is a thin curved ridge $z_j\approx(y_j - \mu)/\tau$: a funnel of its own, now caused by the data. Hence the rule of thumb:</p>
<div class="cols">
<div><h4>Weak data per group</h4><p>Few observations per group, $\sigma_j\gtrsim\tau$: the prior dominates the geometry. Use <strong>non-centered</strong>.</p></div>
<div><h4>Strong data per group</h4><p>Many observations, $\sigma_j\ll\tau$: the likelihood determines each $\theta_j$ well. Use <strong>centered</strong>.</p></div>
</div>
<p>When groups differ (some well measured, some not), one can center some groups and non-center others, or use <em>partial</em> non-centering $\theta_j = \mu + \tau^{1-w_j}\tilde\theta_j$ with $w_j\in[0,1]$ (Papaspiliopoulos, Roberts &amp; Sköld, 2007). Ancillarity–sufficiency interweaving (ASIS; Yu &amp; Meng, 2011) alternates between the two parameterizations within a Gibbs sweep and is robust to the choice.</p>`,
      },
      {
        title: 'A theorem: the two parameterizations are complementary',
        html: R`
<p>The trade-off can be computed exactly for Gibbs sampling in the simplest model: one group, $y\mid\theta\sim\N(\theta,\sigma^2)$, $\theta\mid\mu\sim\N(\mu,\tau^2)$, flat prior on $\mu$, with $\sigma$ and $\tau$ known.</p>
<div class="thm" data-name="Convergence rates of centered and non-centered Gibbs (Papaspiliopoulos, Roberts & Sköld, 2007)">
<p>The two-block Gibbs sampler on $(\theta,\mu)$ (centered) has lag-one autocorrelation (and geometric rate) $\kappa_C = \frac{\sigma^2}{\sigma^2+\tau^2}$ for the $\mu$-chain; on $(z,\mu)$ with $\theta = \mu + \tau z$ (non-centered) it has $\kappa_{NC} = \frac{\tau^2}{\sigma^2+\tau^2}$. In particular $\kappa_C + \kappa_{NC} = 1$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Centered.</em> $\mu\mid\theta\sim\N(\theta,\tau^2)$ (flat prior). $\theta\mid\mu,y\sim\N\big(w y + (1-w)\mu,\; (\sigma^{-2}+\tau^{-2})^{-1}\big)$ with $w = \frac{\tau^2}{\sigma^2+\tau^2}$, the precision-weighted average. So $\mu_{t+1} = \theta_{t+1} + \text{noise} = wy + (1-w)\mu_t + \text{noise}$: an AR(1) with coefficient $1-w = \frac{\sigma^2}{\sigma^2+\tau^2}$.</p>
<p><em>Non-centered.</em> Given $z$, $y\sim\N(\mu + \tau z,\sigma^2)$, so $\mu\mid z,y\sim\N(y-\tau z,\sigma^2)$. Given $\mu$, $z$ has prior $\N(0,1)$ and likelihood $\N(y-\mu;\tau z,\sigma^2)$, so $\E[z\mid\mu,y] = \frac{\tau(y-\mu)}{\sigma^2+\tau^2}$. Then $\mu_{t+1} = y - \tau z_{t+1} + \text{noise}$, with mean $y - \frac{\tau^2}{\sigma^2+\tau^2}(y - \mu_t)$: an AR(1) with coefficient $\frac{\tau^2}{\sigma^2+\tau^2}$.</p>
</details>
<p>When data are informative relative to the prior ($\sigma\ll\tau$), centered is nearly IID and non-centered nearly frozen; when data are weak ($\sigma\gg\tau$), the reverse. The sum being 1 means one of the two is always at least as fast as $1/2$, but which one depends on the unknown balance of $\sigma$ and $\tau$. This is the argument for interweaving the two parameterizations within one sweep, which is never worse than the better of the two and is often much faster than either (Yu &amp; Meng, 2011).</p>`,
      },
      {
        title: 'Worked example: eight schools',
        html: R`
<div class="ex" data-name="Eight schools (Rubin, 1981)">
<p>Estimated effects of coaching programs on SAT-V scores in eight schools, with standard errors:</p>
<table>
<thead><tr><th>School</th><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th><th>F</th><th>G</th><th>H</th></tr></thead>
<tbody>
<tr><td>$y_j$</td><td>28</td><td>8</td><td>−3</td><td>7</td><td>−1</td><td>1</td><td>18</td><td>12</td></tr>
<tr><td>$\sigma_j$</td><td>15</td><td>10</td><td>16</td><td>11</td><td>9</td><td>11</td><td>10</td><td>18</td></tr>
</tbody>
</table>
<p>The between-school spread of the $y_j$ is about what the standard errors alone would produce, so the data are consistent with anything from $\tau = 0$ (complete pooling) to $\tau\approx 20$: the likelihood for $\tau$ is flat near zero and the group-level data are weak ($\sigma_j\gtrsim\tau$). This is the funnel regime. With priors $\mu\sim\N(0,5^2)$ and $\tau\sim$ half-Cauchy$(0,5)$, centered HMC reports divergences concentrated at small $\tau$ and tends to overestimate $\tau$; the non-centered version runs cleanly. The lab lets you shrink the standard errors by a factor $k$ to move into the strong-data regime, where the conclusion reverses.</p>
</div>
<p>In code the two versions differ by two lines. In Stan, the non-centered model declares <code>vector[J] z;</code> with <code>z ~ std_normal();</code> and defines <code>theta = mu + tau * z;</code> as a transformed parameter. Because $\theta$ is a deterministic function, no Jacobian is needed: we place the density on $z$, which is the variable being sampled.</p>`,
        lab: 'hierarchical-mcmc-eight',
      },
      {
        title: 'Beyond the normal–normal model',
        html: R`
<ul>
<li><strong>GLMMs</strong> (<a href="#/t/glmm-gee">generalized linear mixed models</a>): random intercepts $b_j\sim\N(0,\tau^2)$ in a logistic or Poisson model are non-centered as $b_j = \tau z_j$. With few observations per group (binary data especially) non-centering is almost always right.</li>
<li><strong>Multivariate random effects</strong> $b_j\sim\N(0,\Sigma)$: write $b_j = \diag(\tau)\,L z_j$ with $L$ the Cholesky factor of a correlation matrix (LKJ prior). This is the non-centered form in matrix guise.</li>
<li><strong>Scale mixtures</strong> (horseshoe priors, Student-$t$ effects as normals with gamma-distributed precisions): $\beta_j = \tau\lambda_j z_j$ non-centers both global and local scales; centered horseshoes are notoriously funnel-shaped.</li>
<li><strong>Time series and GPs</strong>: latent processes $f = L z$ with $L$ a Cholesky factor of the prior covariance (the "whitened" parameterization) are non-centered; it is the same principle that <a href="#/t/slice-sampling">elliptical slice sampling</a> exploits.</li>
</ul>
<div class="note" data-name="The general lesson">
<p>MCMC efficiency is a property of the <em>parameterization</em>, not just of the model. Two parameterizations of the same posterior are related by a change of variables; a sampler with a fixed metric (isotropic steps, a constant mass matrix) is efficient in one and inefficient in the other. Good parameterizations make the posterior approximately a product of unit-scale independent components. <a href="#/t/variational-inference">Mean-field variational families</a> are sensitive to the parameterization for the same reason.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In Neal's funnel with $v\sim\N(0,9)$, $x\mid v\sim\N(0,e^v)$, what is the conditional standard deviation of $x$ at $v=-6$ and at $v=6$, and why does this break a fixed-step sampler?`,
        a: R`$e^{v/2}$: about $0.05$ at $v=-6$ and $20$ at $v=6$, a factor of 400. A fixed RWM scale or leapfrog step is either far too large in the neck (rejections, divergences, the chain avoids the region) or far too small in the mouth (slow random walk).` },
      { q: R`Write the non-centered version of $\theta_j\sim\N(\mu,\tau^2)$, $y_j\sim\N(\theta_j,\sigma_j^2)$. Is a Jacobian adjustment needed?`,
        a: R`$z_j\sim\N(0,1)$, $\theta_j = \mu+\tau z_j$, $y_j\sim\N(\mu + \tau z_j, \sigma_j^2)$. No Jacobian is needed because the density is specified on $z$, the variable being sampled; $\theta$ is a derived quantity. (A Jacobian would be needed only if we placed a density on $\theta$ while sampling $z$.)` },
      { q: R`In the one-group normal model, centered Gibbs has rate $\sigma^2/(\sigma^2+\tau^2)$. If $\sigma = 1$ and $\tau = 10$, which parameterization should you use?`,
        a: R`Centered: $\kappa_C = 1/101\approx 0.01$ (nearly IID), whereas non-centered has $\kappa_{NC} = 100/101\approx 0.99$. The data are much more informative than the prior about $\theta$.` },
      { q: R`Why do divergences in a centered hierarchical model typically lead to <em>over</em>estimation of $\tau$?`,
        a: R`The divergences occur in the neck (small $\tau$), where the curvature is too high for the step size. Trajectories entering the neck are rejected, so the chain under-samples small $\tau$, shifting the posterior for $\tau$ upward and understating the amount of pooling.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'variational-inference',
    domain: 'mcmc',
    order: 100,
    title: 'Variational Inference',
    subtitle: 'Replace sampling with optimization: the closest tractable distribution in KL divergence',
    level: 2,
    sources: { other: 'Jordan, Ghahramani, Jaakkola & Saul (1999); Blei, Kucukelbir & McAuliffe (2017); Kingma & Welling (2014); Kucukelbir et al. (2017); Yao, Vehtari, Simpson & Gelman (2018)' },
    prereqs: ['bayesian-inference', 'entropy-kl'],
    related: ['em-algorithm', 'autoencoders-vae', 'gibbs-sampling', 'importance-sampling', 'gaussian-mixtures', 'hamiltonian-mc'],
    tags: ['ELBO', 'KL divergence', 'mean field', 'CAVI', 'reparameterization trick', 'ADVI'],
    summary: R`Variational inference approximates a posterior $p(\theta\mid y)$ by the member of a tractable family $\mathcal Q$ that minimizes $\KL(q\,\|\,p(\cdot\mid y))$. The KL itself involves the unknown evidence $p(y)$, but $\log p(y) = \text{ELBO}(q) + \KL(q\|p)$, so minimizing KL is the same as maximizing the evidence lower bound, which needs only the joint density. Mean-field families give closed-form coordinate updates (CAVI); generic families are fitted by stochastic gradients, made low-variance by the reparameterization trick. VI is fast and scales to huge data, but the reverse KL makes it mode-seeking and systematically underestimates posterior variance.`,
    survey: {
      task: R`Approximate a posterior (or any unnormalized density) by an optimized tractable $q$`,
      assumptions: R`Variational family $\mathcal Q$; joint $\log p(y,\theta)$ evaluable (and differentiable for BBVI)`,
      output: R`$q^*\in\mathcal Q$ and a lower bound on $\log p(y)$`,
      strengths: 'Orders of magnitude faster than MCMC; scales with stochastic optimization; amortizable (VAEs)',
      weaknesses: 'Biased with no asymptotic exactness; underestimates variance; local optima; hard to assess accuracy',
      complexity: R`Optimization: iterations $\times$ cost of (stochastic) ELBO gradients`,
    },
    sections: [
      {
        title: 'From sampling to optimization: the ELBO',
        html: R`
<p>MCMC is asymptotically exact but can be slow, and assessing its convergence is hard. VI takes the opposite trade: choose a family $\mathcal Q$ of distributions over $\theta$ that are easy to work with, and find the best one. "Best" is measured by the <a href="#/t/entropy-kl">Kullback–Leibler divergence</a> from $q$ to the posterior,
$$q^* = \argmin_{q\in\mathcal Q}\KL\big(q(\theta)\,\|\,p(\theta\mid y)\big),\qquad \KL(q\|p) = \E_q\big[\log q(\theta) - \log p(\theta\mid y)\big].$$
Evaluating this KL requires $\log p(\theta\mid y) = \log p(y,\theta) - \log p(y)$, which contains the intractable evidence. But the evidence does not depend on $q$:</p>
<div class="thm" data-kind="Proposition" data-name="Evidence decomposition">
<p>For any density $q$ with $q\ll p(\cdot\mid y)$,
$$\log p(y) = \underbrace{\E_q\big[\log p(y,\theta)\big] - \E_q\big[\log q(\theta)\big]}_{\text{ELBO}(q)} + \KL\big(q\,\|\,p(\cdot\mid y)\big).$$
Since $\KL\ge0$, $\text{ELBO}(q)\le\log p(y)$, with equality iff $q = p(\cdot\mid y)$ a.e. Maximizing the ELBO over $\mathcal Q$ is equivalent to minimizing the KL.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\KL(q\|p(\cdot\mid y)) = \E_q[\log q(\theta)] - \E_q[\log p(y,\theta)] + \log p(y)$, using $\log p(\theta\mid y) = \log p(y,\theta) - \log p(y)$ and $\E_q[\log p(y)] = \log p(y)$. Rearranging gives the identity. Nonnegativity of KL, with equality iff the distributions agree, is Gibbs' inequality (Jensen applied to $-\log$).</p>
</details>
<p>Equivalently, by Jensen, $\log p(y) = \log\E_q[p(y,\theta)/q(\theta)]\ge\E_q[\log p(y,\theta)/q(\theta)]$. The ELBO splits as an <em>energy</em> term $\E_q\log p(y,\theta)$, which rewards placing $q$'s mass where the joint is large, plus the <em>entropy</em> $H(q) = -\E_q\log q$, which rewards spreading out. The same decomposition underlies the <a href="#/t/em-algorithm">EM algorithm</a>: EM is coordinate ascent on the ELBO in which the E-step sets $q$ to the exact conditional of the latent variables, closing the gap completely.</p>`,
      },
      {
        title: 'Mean-field VI and coordinate ascent (CAVI)',
        html: R`
<p>The classical family is <strong>mean field</strong>: $\theta$ is split into blocks and $q$ factorizes, $q(\theta) = \prod_{j=1}^m q_j(\theta_j)$, with each $q_j$ otherwise unrestricted.</p>
<div class="thm" data-name="Optimal mean-field factor">
<p>Holding $q_{-j} = \prod_{k\ne j}q_k$ fixed, the ELBO is maximized over $q_j$ by
$$q_j^*(\theta_j)\;\propto\;\exp\Big(\E_{q_{-j}}\big[\log p(y,\theta_j,\theta_{-j})\big]\Big).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $\text{ELBO}(q) = \E_{q_j}\big[\E_{q_{-j}}\log p(y,\theta)\big] - \E_{q_j}\log q_j - \sum_{k\ne j}\E_{q_k}\log q_k$. Define $\log\tilde q_j(\theta_j) = \E_{q_{-j}}[\log p(y,\theta)] - \log C$ with $C$ normalizing $\tilde q_j$. Then the terms depending on $q_j$ are $\E_{q_j}[\log\tilde q_j] - \E_{q_j}[\log q_j] + \log C = -\KL(q_j\|\tilde q_j) + \log C$, maximized uniquely at $q_j = \tilde q_j$.</p>
</details>
<p>Coordinate ascent variational inference (CAVI) cycles through $j$, applying this update. Each step cannot decrease the ELBO, so CAVI converges to a local optimum. In conditionally conjugate models (exponential-family complete conditionals, the same models where <a href="#/t/gibbs-sampling">Gibbs sampling</a> works), $q_j^*$ is in the same family as the complete conditional, with natural parameter equal to the $q_{-j}$-expectation of the conditional's natural parameter. CAVI is "Gibbs sampling with expectations in place of draws".</p>
<div class="ex" data-name="Mean field on a correlated Gaussian">
<p>Let the target be $\N(\mu,\Lambda^{-1})$ in $\R^2$, with precision matrix $\Lambda$. Then $\log p = -\frac12(\theta-\mu)^\top\Lambda(\theta-\mu) + c$, and the optimal factor for $\theta_1$ is Gaussian:
$$q_1^* = \N\Big(m_1,\;\Lambda_{11}^{-1}\Big),\qquad m_1 = \mu_1 - \Lambda_{11}^{-1}\Lambda_{12}(m_2 - \mu_2),$$
and symmetrically for $\theta_2$. The fixed point is $m = \mu$: the means are exact. The errors $e_j = m_j - \mu_j$ satisfy $e_1\leftarrow -\frac{\Lambda_{12}}{\Lambda_{11}}e_2$, $e_2\leftarrow-\frac{\Lambda_{21}}{\Lambda_{22}}e_1$, contracting per sweep by $\frac{\Lambda_{12}^2}{\Lambda_{11}\Lambda_{22}} = \rho^2$, exactly the Gibbs sampler's rate. The variances, however, are the <em>conditional</em> variances $1/\Lambda_{jj} = \Var(\theta_j\mid\theta_{-j})$, not the marginal ones.</p>
</div>`,
        lab: 'variational-inference-meanfield',
      },
      {
        title: 'What reverse KL does: mode seeking and variance underestimation',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Mean field underestimates marginal variances (Gaussian case)">
<p>For a Gaussian target, the mean-field optimum has $\Var_q(\theta_j) = 1/\Lambda_{jj}\le\Sigma_{jj} = \Var_p(\theta_j)$, with equality iff $\theta_j$ is uncorrelated with the other coordinates. For a bivariate normal with correlation $\rho$, the ratio of standard deviations is $\sqrt{1-\rho^2}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For a Gaussian, $1/\Lambda_{jj}$ is the conditional variance $\Var(\theta_j\mid\theta_{-j})$ (Schur complement). By the law of total variance, $\Var(\theta_j) = \E\Var(\theta_j\mid\theta_{-j}) + \Var\E(\theta_j\mid\theta_{-j})\ge\Var(\theta_j\mid\theta_{-j})$, with equality iff the conditional mean does not depend on $\theta_{-j}$, i.e. zero correlation. In the bivariate case $\Var(\theta_1\mid\theta_2) = \sigma_1^2(1-\rho^2)$.</p>
</details>
<p>The general mechanism is the direction of the KL. $\KL(q\|p) = \E_q[\log q/p]$ is huge wherever $q$ puts mass and $p$ does not: $q$ is penalized for <em>covering</em> regions of low posterior density, and not penalized for <em>missing</em> regions of high posterior density. It is <strong>zero-forcing</strong>: the optimal $q$ hides inside the posterior, fitting one mode of a multimodal target and squeezing into a correlated one. The forward divergence $\KL(p\|q)$ (used by expectation propagation) is the opposite, <strong>mass-covering</strong>: for $q$ in an exponential family, minimizing $\KL(p\|q)$ is moment matching, $\E_q T(\theta) = \E_p T(\theta)$, because $\KL(p\|q) = -\E_p\log q + \text{const}$ and the gradient in the natural parameter is $\E_q T - \E_p T$.</p>
<div class="cols">
<div><h4>Reverse KL $\KL(q\|p)$ (VI)</h4><p>Tractable (expectations under $q$). Mode-seeking; underdispersed; picks one mode; gives a lower bound on the evidence.</p></div>
<div><h4>Forward KL $\KL(p\|q)$ (EP)</h4><p>Needs expectations under $p$ (the hard part). Mass-covering; overdispersed for multimodal $p$; matches moments.</p></div>
</div>
<p>Asymptotically, under regularity conditions, VI posteriors are consistent and concentrate at the right rate around the truth (Wang &amp; Blei, 2019), so point estimates are often fine; it is the <em>uncertainty</em> that suffers, with credible intervals from mean-field VI too narrow when parameters are correlated a posteriori.</p>`,
      },
      {
        title: 'Black-box VI and the reparameterization trick',
        html: R`
<p>CAVI needs conjugacy. For general models, parameterize $q_\lambda$ (e.g., a Gaussian with mean $m$ and Cholesky factor $C$) and maximize $\mathcal L(\lambda) = \E_{q_\lambda}[\log p(y,\theta) - \log q_\lambda(\theta)]$ by stochastic gradient ascent. Two unbiased gradient estimators exist.</p>
<div class="cols">
<div><h4>Score function (REINFORCE)</h4>$$\nabla_\lambda\mathcal L = \E_{q_\lambda}\big[(\log p(y,\theta) - \log q_\lambda(\theta))\,\nabla_\lambda\log q_\lambda(\theta)\big]$$<p>Needs only evaluations of $\log p$; works for discrete $\theta$; typically very high variance (control variates and Rao–Blackwellization help).</p></div>
<div><h4>Reparameterization (pathwise)</h4>$$\theta = g_\lambda(\varepsilon),\ \varepsilon\sim r:\quad \nabla_\lambda\mathcal L = \E_r\big[\nabla_\theta\big(\log p(y,\theta)-\log q_\lambda(\theta)\big)\big|_{\theta = g_\lambda(\varepsilon)}\nabla_\lambda g_\lambda(\varepsilon)\big]$$<p>Needs $\nabla_\theta\log p$ and a differentiable $g$; usually much lower variance.</p></div>
</div>
<div class="thm" data-kind="Proposition" data-name="Reparameterization gradient for a Gaussian family">
<p>For $q = \N(m, s^2)$ in one dimension, write $\theta = m + s\varepsilon$, $\varepsilon\sim\N(0,1)$, and use the closed-form entropy $H(q) = \log s + \frac12\log(2\pi e)$. Then
$$\frac{\partial\mathcal L}{\partial m} = \E_\varepsilon\big[\nabla\log p(y, m+s\varepsilon)\big],\qquad \frac{\partial\mathcal L}{\partial\log s} = \E_\varepsilon\big[\nabla\log p(y, m+s\varepsilon)\,s\varepsilon\big] + 1.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\mathcal L(m,s) = \E_\varepsilon[\log p(y,m+s\varepsilon)] + \log s + \text{const}$. The distribution of $\varepsilon$ does not depend on $(m,s)$, so the gradient passes inside the expectation (dominated convergence, given integrable $\nabla\log p$): $\partial_m$ gives $\E[\nabla\log p]$; $\partial_s$ gives $\E[\nabla\log p\cdot\varepsilon] + 1/s$; and $\partial_{\log s} = s\,\partial_s$.</p>
</details>
<p>The score-function estimator's variance does not shrink as $q$ approaches a good fit, because the log-ratio still fluctuates; the reparameterized estimator's noise comes only from the variation of $\nabla\log p$ across the spread of $q$, which is small whenever $\log p$ is smooth on that scale. Automatic differentiation variational inference (ADVI; Kucukelbir et al., 2017) combines reparameterization gradients, a Gaussian family on an unconstrained transformation of $\theta$, and step-size adaptation. Stochastic VI (Hoffman et al., 2013) additionally subsamples data, scaling to millions of observations. Variational autoencoders (<a href="#/t/autoencoders-vae">VAEs</a>) amortize: a neural network maps each $y_i$ to its local $q$'s parameters, trained jointly with the model by the same reparameterized ELBO.</p>`,
        lab: 'variational-inference-reparam',
      },
      {
        title: 'Assessing and improving a variational fit',
        html: R`
<p>Unlike MCMC, VI has no asymptotic exactness to fall back on: more iterations converge to $q^*$, not to $p$. So the error $\KL(q^*\|p)$ must be checked independently.</p>
<ul>
<li><strong>Importance weights.</strong> Draw $\theta_s\sim q$ and form $w_s = p(y,\theta_s)/q(\theta_s)$. Their variability measures the mismatch. Pareto-smoothed importance sampling diagnostics (Yao, Vehtari, Simpson &amp; Gelman, 2018) fit a generalized Pareto to the largest weights; a shape estimate $\hat k\gt 0.7$ says $q$ is too narrow in some direction for any reweighting to be reliable. When $\hat k$ is small, <a href="#/t/importance-sampling">importance sampling</a> with $q$ as the proposal corrects VI's bias.</li>
<li><strong>Richer families.</strong> Full-rank Gaussians capture correlations (at $O(d^2)$ parameters); mixtures capture several modes; normalizing flows give flexible families with tractable densities; structured mean field keeps dependent blocks together, the VI counterpart of <a href="#/t/gibbs-sampling">blocking</a>.</li>
<li><strong>Other divergences.</strong> Rényi-$\alpha$ and $\chi^2$ divergences interpolate toward mass-covering behaviour; importance-weighted bounds (IWAE) $\E\log\frac1K\sum_k w_k$ tighten the ELBO and reduce underdispersion as $K$ grows.</li>
</ul>
<div class="warn" data-name="Pitfalls">
<p>(1) The ELBO is not comparable across variational families as a measure of fit to the posterior; it measures $\log p(y) - \KL$, and both terms matter. (2) Multimodal posteriors give multiple local optima; the answer depends on initialization. (3) Parameterization matters: a mean-field Gaussian in the <a href="#/t/hierarchical-mcmc">centered</a> hierarchical parameterization is a much worse approximation than in the non-centered one. (4) Stochastic optimization noise can be mistaken for convergence; monitor a running average of the ELBO and the parameter trajectories.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Derive $\log p(y) = \text{ELBO}(q) + \KL(q\|p(\cdot\mid y))$ and explain why we can maximize the ELBO without knowing $p(y)$.`,
        a: R`$\KL(q\|p(\cdot\mid y)) = \E_q\log q - \E_q\log p(y,\theta) + \log p(y)$. Rearranging gives the identity. Since $\log p(y)$ does not depend on $q$, maximizing $\text{ELBO}(q) = \E_q\log p(y,\theta) - \E_q\log q$ (which needs only the joint) is equivalent to minimizing the KL.` },
      { q: R`A bivariate normal posterior has unit marginal variances and correlation $0.9$. What marginal standard deviations does mean-field VI report, and what does $\KL(p\|q)$ minimization over factorized Gaussians report?`,
        a: R`Mean-field VI uses the conditional variances $1-\rho^2 = 0.19$, so sd $\approx 0.436$: less than half the true value of 1. Minimizing the forward KL over factorized Gaussians matches the marginal moments, giving sd $=1$ exactly (but ignoring the correlation).` },
      { q: R`Why is $\KL(q\|p)$ called zero-forcing?`,
        a: R`$\KL(q\|p) = \int q\log(q/p)$ is infinite if $q\gt0$ where $p = 0$, and large where $q\gg p$. So the optimal $q$ must vanish wherever $p$ is (near) zero, which pushes it inside the support of $p$, typically onto one mode and with too little spread. Missing mass of $p$ costs nothing directly.` },
      { q: R`Write the reparameterization-gradient estimator for the mean $m$ of a Gaussian $q = \N(m, s^2)$, using $K$ Monte Carlo samples.`,
        a: R`Draw $\varepsilon_1,\dots,\varepsilon_K\iid\N(0,1)$ and set $\hat g_m = \frac1K\sum_k\nabla_\theta\log p(y,\theta)\big|_{\theta = m + s\varepsilon_k}$. It is unbiased for $\partial\mathcal L/\partial m$ because the entropy does not depend on $m$ and the sampling distribution of $\varepsilon$ does not depend on $(m,s)$.` },
      { q: R`How is the CAVI update related to the Gibbs sampler for the same model?`,
        a: R`Both use the complete conditionals $p(\theta_j\mid\theta_{-j},y)$. Gibbs draws $\theta_j$ from the conditional given the current values of the others; CAVI sets $q_j\propto\exp\E_{q_{-j}}[\log p(\theta_j\mid\theta_{-j},y)]$, averaging the log conditional over the other factors. In conjugate models this replaces the other variables' current values by their variational expectations (of natural parameters).` },
    ],
  });
})();
