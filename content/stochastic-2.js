/* Stochastic Processes II: continuous time — Poisson processes, CTMCs, queues, martingales,
   Brownian motion, Itô calculus, and stationary time series. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'poisson-process',
    domain: 'stochastic',
    order: 70,
    title: 'The Poisson Process',
    subtitle: 'Completely random points in time: three definitions, one object',
    level: 2,
    sources: { aos: '23.3', cb: '—', other: 'Kingman, Poisson Processes (1993); Ross, Stochastic Processes, Ch. 2' },
    prereqs: ['distribution-families', 'order-statistics'],
    related: ['ctmc', 'queueing', 'point-process-evt', 'poisson-regression'],
    tags: ['counting process', 'exponential', 'thinning', 'superposition', 'point process'],
    summary: R`The Poisson process is the canonical model for events that occur "completely at random" in time: arrivals at a server, radioactive decays, insurance claims, exceedances of a high threshold. It can be defined in three apparently different ways — by independent Poisson increments, by IID exponential gaps, or as the limit of rare independent Bernoulli trials — and the fact that these coincide is what makes it so robust as a model. Its arrival times, given their number, are uniform order statistics; it is closed under random thinning and under superposition; and a deterministic time change turns any non-homogeneous version into the standard one. These properties make it the building block of continuous-time Markov chains, queues, and the point-process view of extremes.`,
    survey: {
      task: 'Model random event times; count events in windows; simulate arrivals',
      assumptions: R`Independent increments; no simultaneous events (orderliness); intensity $\lambda(t)$ deterministic`,
      output: R`Counts $N(t)\sim\text{Poisson}(\Lambda(t))$; arrival times; MLE $\hat\lambda = N(T)/T$`,
      strengths: 'Exactly solvable; closed under thinning, superposition and time change; a universal limit of rare events',
      weaknesses: 'No clustering or self-excitation: real counts are often overdispersed (variance > mean)',
      complexity: R`Simulation $O(N(T))$; NHPP by thinning $O(\lambda^* T)$`,
    },
    sections: [
      {
        title: 'Counting processes and three equivalent definitions',
        html: R`
<p>A <strong>counting process</strong> $\{N(t): t\ge 0\}$ records the number of events in $[0,t]$: it is integer-valued, $N(0)=0$, non-decreasing, and right-continuous. Write $S_n$ for the time of the $n$-th event and $T_n = S_n - S_{n-1}$ (with $S_0 = 0$) for the inter-arrival times, so that $\{N(t)\ge n\} = \{S_n \le t\}$. That last identity is the bridge between the "counts" view and the "times" view and is used constantly below.</p>
<div class="def" data-name="Poisson process (increments definition)">
<p>$N$ is a (homogeneous) <strong>Poisson process with rate $\lambda>0$</strong> if (i) $N(0)=0$; (ii) it has independent increments: for $0\le t_0< t_1<\dots< t_k$ the counts $N(t_1)-N(t_0),\dots,N(t_k)-N(t_{k-1})$ are independent; and (iii) $N(s+t)-N(s)\sim\text{Poisson}(\lambda t)$ for all $s,t\ge 0$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Equivalent characterizations">
<p>For a counting process $N$ and $\lambda>0$ the following are equivalent:</p>
<ol>
<li><strong>(Increments)</strong> $N$ is a Poisson process of rate $\lambda$ in the sense above.</li>
<li><strong>(Gaps)</strong> The inter-arrival times $T_1,T_2,\dots$ are IID $\text{Exp}(\lambda)$, and $N(t)=\max\{n: S_n\le t\}$.</li>
<li><strong>(Infinitesimal)</strong> $N(0)=0$, $N$ has stationary independent increments, $\P(N(h)=1)=\lambda h+o(h)$ and $\P(N(h)\ge 2)=o(h)$ as $h\downarrow 0$.</li>
</ol>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><strong>(3 ⇒ 1).</strong> Let $p_n(t)=\P(N(t)=n)$. By independent stationary increments, $p_0(t+h)=p_0(t)p_0(h)=p_0(t)(1-\lambda h+o(h))$, so $p_0'=-\lambda p_0$ and $p_0(t)=e^{-\lambda t}$. For $n\ge1$, conditioning on the count in $(t,t+h]$,
$$p_n(t+h)=p_n(t)(1-\lambda h)+p_{n-1}(t)\lambda h+o(h)\;\Longrightarrow\; p_n'=-\lambda p_n+\lambda p_{n-1}.$$
Then $(e^{\lambda t}p_n)'=\lambda e^{\lambda t}p_{n-1}$, and induction from $e^{\lambda t}p_0=1$ gives $e^{\lambda t}p_n(t)=(\lambda t)^n/n!$, i.e. $N(t)\sim\text{Poisson}(\lambda t)$. Stationarity transfers this to every increment.</p>
<p><strong>(1 ⇒ 2).</strong> $\P(T_1>t)=\P(N(t)=0)=e^{-\lambda t}$. For the joint law, take $0< s_1<\dots< s_n$ and small disjoint windows $[s_i,s_i+h_i]$. "One arrival in each window and none elsewhere in $[0,s_n+h_n]$" has probability, by independent Poisson increments, $\prod_i(\lambda h_i e^{-\lambda h_i})\cdot e^{-\lambda(s_n-\sum h_i)} = \lambda^n e^{-\lambda s_n}\prod h_i\,(1+o(1))$. So $(S_1,\dots,S_n)$ has density $\lambda^n e^{-\lambda s_n}$ on $\{0< s_1<\dots< s_n\}$. The linear map $(s_1,\dots,s_n)\mapsto(s_1,s_2-s_1,\dots,s_n-s_{n-1})$ has Jacobian $1$, giving density $\prod_i \lambda e^{-\lambda t_i}$ for the gaps: IID exponentials.</p>
<p><strong>(2 ⇒ 1).</strong> $S_n$ is a sum of $n$ IID $\text{Exp}(\lambda)$, so $S_n\sim\text{Gamma}(n,\lambda)$. Then $\P(N(t)=n)=\P(S_n\le t)-\P(S_{n+1}\le t)$, whose $t$-derivative is $\lambda e^{-\lambda t}\big[\tfrac{(\lambda t)^{n-1}}{(n-1)!}-\tfrac{(\lambda t)^n}{n!}\big]=\tfrac{\d}{\d t}\big[e^{-\lambda t}(\lambda t)^n/n!\big]$; both vanish at $t=0$, so $N(t)\sim\text{Poisson}(\lambda t)$. Independent increments follow from memorylessness: at any fixed time $s$, the residual wait to the next arrival is $\text{Exp}(\lambda)$ independent of $\{N(u):u\le s\}$, and later gaps are fresh, so the post-$s$ process is an independent copy. (3) follows from (1) by expanding the Poisson pmf at small $\lambda h$.</p>
</details>
<p>A fourth view explains <em>why</em> the Poisson process is ubiquitous. Cut $[0,t]$ into $n$ slots of width $t/n$ and let each slot independently contain an event with probability $\lambda t/n$. The count is $\text{Bin}(n,\lambda t/n)$, whose pgf $\big(1+\tfrac{\lambda t}{n}(z-1)\big)^n\to e^{\lambda t(z-1)}$, the Poisson pgf. Disjoint intervals use disjoint slots, hence independence; the slot of the first event is geometric, and a scaled geometric converges to an exponential. This is the <strong>law of rare events</strong>, and it holds quantitatively and without identical probabilities:</p>
<div class="thm" data-kind="Theorem" data-name="Le Cam's inequality (1960)">
<p>If $Y_1,\dots,Y_n$ are independent Bernoulli$(p_i)$ and $\mu=\sum p_i$, then the total-variation distance between the law of $\sum Y_i$ and $\text{Poisson}(\mu)$ is at most $\sum_i p_i^2$.</p>
</div>
<p>So any superposition of many independent, individually rare event sources looks Poisson, regardless of their individual mechanisms. That is the modelling justification, and also a hint about when it fails: when events are <em>not</em> independent (aftershocks, contagion, batch arrivals).</p>`,
      },
      {
        title: 'Arrival times: Gamma waits and the order-statistics property',
        html: R`
<p>From the gaps definition, $S_n\sim\text{Gamma}(n,\lambda)$ with density $\lambda^n s^{n-1}e^{-\lambda s}/(n-1)!$ and mean $n/\lambda$. The most useful structural fact, however, is what the arrival times look like <em>once we know how many there were</em>.</p>
<div class="thm" data-kind="Theorem" data-name="Order-statistics property">
<p>Conditional on $N(t)=n$, the arrival times $(S_1,\dots,S_n)$ have the same joint law as the order statistics $(U_{(1)},\dots,U_{(n)})$ of $n$ IID $\text{Uniform}(0,t)$ variables, i.e. density $n!/t^n$ on $\{0< s_1<\dots< s_n< t\}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>From the previous proof, $(S_1,\dots,S_{n+1})$ has density $\lambda^{n+1}e^{-\lambda s_{n+1}}$ on the ordered cone. The event $\{N(t)=n\}$ is $\{S_n\le t< S_{n+1}\}$. Integrating $s_{n+1}$ over $(t,\infty)$ gives the joint density of $(S_1,\dots,S_n)$ on $\{N(t)=n\}$: $\lambda^n e^{-\lambda t}$ for $0< s_1<\dots< s_n< t$. Dividing by $\P(N(t)=n)=e^{-\lambda t}(\lambda t)^n/n!$ leaves $n!/t^n$, which is the density of uniform <a href="#/t/order-statistics">order statistics</a>.</p>
</details>
<p>Equivalently: given the count, the <em>unordered</em> set of times is an IID uniform sample. This gives a second simulation algorithm (draw $N(T)\sim\text{Poisson}(\lambda T)$, then sort $N(T)$ uniforms) and a powerful computational device:</p>
<div class="ex" data-name="Campbell's formula and a discounted-claims example">
<p>For any integrable $g$, $\E\big[\sum_{i=1}^{N(t)} g(S_i)\big]=\E\big[\E[\sum_{i=1}^{N(t)} g(U_i)\mid N(t)]\big]=\E N(t)\cdot\tfrac1t\int_0^t g=\lambda\int_0^t g(s)\d s$. For instance, if claims of mean size $m$ arrive at rate $\lambda$ and are discounted at rate $r$, the expected present value of claims in $[0,t]$ is $\lambda m\int_0^t e^{-rs}\d s=\lambda m(1-e^{-rt})/r$, assuming claim sizes independent of the times.</p>
</div>
<p>The order-statistics property is also the basis of goodness-of-fit testing: given $N(T)=n$, the rescaled times $S_i/T$ should look like a sorted uniform sample, so a Kolmogorov–Smirnov test against $\text{Uniform}(0,1)$ checks homogeneity, and a dispersion test on counts in sub-windows checks the Poisson variance.</p>`,
        lab: 'poisson-process-timeline',
      },
      {
        title: 'Thinning, superposition and marking',
        html: R`
<p>Two operations leave the Poisson class invariant, and together they explain most of its algebra.</p>
<div class="thm" data-kind="Theorem" data-name="Thinning (colouring)">
<p>Let $N$ be Poisson($\lambda$) and independently label each event type 1 with probability $p$ and type 2 otherwise. Then the type-1 and type-2 counting processes $N_1,N_2$ are <strong>independent</strong> Poisson processes with rates $\lambda p$ and $\lambda(1-p)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>On a single interval of length $t$:
$$\P(N_1(t)=j,N_2(t)=k)=\P(N(t)=j+k)\binom{j+k}{j}p^j(1-p)^k=\frac{e^{-\lambda p t}(\lambda p t)^j}{j!}\cdot\frac{e^{-\lambda(1-p)t}(\lambda(1-p)t)^k}{k!},$$
which factorizes into the product of two Poisson pmfs. Applying this to each of finitely many disjoint intervals, which are independent under $N$ and receive independent labels, shows that all the increments of $N_1$ and $N_2$ over disjoint intervals are mutually independent with the right Poisson laws.</p>
</details>
<p>The independence of $N_1$ and $N_2$ is the surprising part: learning that many type-1 events occurred tells you nothing about type 2. It fails for any other count distribution (this independence in fact characterizes the Poisson). The same proof works when the retention probability depends on the time, $p=p(s)$: the kept process is then a non-homogeneous Poisson process with intensity $\lambda p(s)$.</p>
<div class="thm" data-kind="Theorem" data-name="Superposition">
<p>If $N_1,\dots,N_k$ are independent Poisson processes with rates $\lambda_1,\dots,\lambda_k$, then $N=\sum_i N_i$ is a Poisson process with rate $\lambda=\sum_i\lambda_i$, and each event of $N$ comes from source $i$ with probability $\lambda_i/\lambda$, independently of everything else.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Increments of $N$ over disjoint intervals are sums of independent Poisson variables, hence independent and Poisson with summed means. For the labelling: by the gaps definition, the first event of $N$ is $\min_i E_i$ with $E_i\sim\text{Exp}(\lambda_i)$ independent; $\P(\min_i E_i>t, E_1 \text{ is the min})=\int_t^\infty\lambda_1e^{-\lambda_1 s}e^{-(\lambda-\lambda_1)s}\d s=\tfrac{\lambda_1}{\lambda}e^{-\lambda t}$, which factorizes: the minimum is $\text{Exp}(\lambda)$, the index is independent of it with probabilities $\lambda_i/\lambda$. Memorylessness restarts the competition after each event.</p>
</details>
<p>This "competing exponential clocks" lemma is exactly the mechanism of <a href="#/t/ctmc">continuous-time Markov chains</a>. More generally, attaching an independent random <em>mark</em> $Y_i$ (claim size, customer type, location) to each event produces a Poisson process on the product space $[0,\infty)\times\mathcal Y$ with intensity $\lambda\,\d t\,F_Y(\d y)$ — the marking theorem.</p>
<div class="ex" data-name="The M/G/∞ queue is Poisson">
<p>Customers arrive at rate $\lambda$; each stays for an independent time with CDF $G$; there are infinitely many servers. A customer arriving at $s\le t$ is still present at $t$ with probability $1-G(t-s)$. By time-dependent thinning, the number present at $t$ is Poisson with mean $\lambda\int_0^t(1-G(u))\d u\to\lambda\E[S]$. The stationary distribution depends on $G$ only through its mean — an <em>insensitivity</em> result that reappears in <a href="#/t/queueing">queueing theory</a>.</p>
</div>`,
      },
      {
        title: 'Non-homogeneous Poisson processes and simulation by thinning',
        html: R`
<p>Real arrival rates vary: traffic peaks at rush hour, failures increase with age. Let $\lambda(t)\ge0$ be locally integrable with cumulative intensity $\Lambda(t)=\int_0^t\lambda(s)\d s$.</p>
<div class="def" data-name="Non-homogeneous Poisson process (NHPP)">
<p>$N$ is an NHPP with intensity $\lambda(\cdot)$ if $N(0)=0$, it has independent increments, and $N(t)-N(s)\sim\text{Poisson}(\Lambda(t)-\Lambda(s))$ for $s< t$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Time change">
<p>If $\Lambda$ is continuous and strictly increasing, $\tilde N(u)=N(\Lambda^{-1}(u))$ is a homogeneous Poisson process with rate 1. Conversely, if $\tilde N$ is rate 1, then $N(t)=\tilde N(\Lambda(t))$ is an NHPP with intensity $\lambda$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Increments of $\tilde N$ over disjoint $u$-intervals are increments of $N$ over disjoint $t$-intervals, hence independent, and $\tilde N(u_2)-\tilde N(u_1)\sim\text{Poisson}(\Lambda(\Lambda^{-1}(u_2))-\Lambda(\Lambda^{-1}(u_1)))=\text{Poisson}(u_2-u_1)$.</p>
</details>
<p>The time change gives an inversion sampler (simulate rate-1 arrivals $\tilde S_i$ and map $S_i=\Lambda^{-1}(\tilde S_i)$) but requires $\Lambda^{-1}$. Thinning needs only a bound on $\lambda$:</p>
<div class="algo" data-name="Lewis–Shedler thinning (1979)">
<ol>
<li>Choose $\lambda^*\ge\sup_{t\le T}\lambda(t)$. Simulate a homogeneous Poisson($\lambda^*$) process on $[0,T]$: candidate times $s_1< s_2<\dots$.</li>
<li>Keep candidate $s_i$ independently with probability $\lambda(s_i)/\lambda^*$.</li>
<li>The kept points form an NHPP with intensity $\lambda(t)$. The expected number of candidates is $\lambda^*T$; the acceptance rate is $\Lambda(T)/(\lambda^*T)$.</li>
</ol>
</div>
<p>Correctness is time-dependent thinning with $p(s)=\lambda(s)/\lambda^*$. The efficiency loss is the same as in <a href="#/t/rejection-sampling">accept–reject sampling</a>: a loose envelope wastes candidates, so piecewise-constant envelopes are used when $\lambda$ is peaked.</p>
<div class="note" data-name="The likelihood of a point pattern">
<p>Observing an NHPP on $[0,T]$ with events at $t_1<\dots< t_n$, the likelihood (density with respect to a unit-rate process) is
$$L(\lambda)=\Big[\prod_{i=1}^n\lambda(t_i)\Big]\exp\Big(-\int_0^T\lambda(t)\d t\Big).$$
Each event contributes its intensity; the exponential is the probability of no events anywhere else. For constant $\lambda$ this is $\lambda^ne^{-\lambda T}$, so $\hat\lambda=n/T$ with Fisher information $T/\lambda$. For a parametric $\lambda_\theta(t)$, e.g. $\log\lambda_\theta(t)=\theta_0+\theta_1 t$, maximizing $\log L$ is a <a href="#/t/poisson-regression">Poisson regression</a> in disguise — binning time finely and fitting a Poisson GLM to the bin counts approximates exactly this likelihood.</p>
</div>`,
        lab: 'poisson-process-thinning',
      },
      {
        title: 'Inference, compound processes, and the inspection paradox',
        html: R`
<p><strong>Estimating the rate.</strong> With $n=N(T)$ events, $\hat\lambda=n/T$ is the MLE and is unbiased with $\Var(\hat\lambda)=\lambda/T$. An exact (conservative) $1-\alpha$ interval comes from the Poisson–Gamma duality $\P(N(T)\ge n)=\P(S_n\le T)$:
$$\Big[\frac{\chi^2_{2n,\,\alpha/2}}{2T},\;\frac{\chi^2_{2(n+1),\,1-\alpha/2}}{2T}\Big],$$
where $\chi^2_{k,q}$ is the $q$ quantile (Garwood, 1936). Note it is the <em>time on test</em> $T$, not the number of intervals, that controls precision.</p>
<p><strong>Compound Poisson.</strong> If events carry IID marks $Y_i$ independent of $N$, the total $X(t)=\sum_{i=1}^{N(t)}Y_i$ has
$$\E X(t)=\lambda t\,\E Y,\qquad \Var X(t)=\lambda t\,\E[Y^2],\qquad \E e^{sX(t)}=\exp\big(\lambda t(M_Y(s)-1)\big).$$
(The variance uses $\E[Y^2]$, not $\Var Y$: the law of total variance gives $\E N\Var Y+\Var N(\E Y)^2$ and $\Var N=\E N$.) This is the Cramér–Lundberg model of insurance risk and a Lévy process.</p>
<div class="warn" data-name="The inspection paradox">
<p>Arrive at a bus stop at a "random" large time $t$ when buses follow a Poisson($\lambda$) process. The mean gap between buses is $1/\lambda$, yet the gap <em>you land in</em> has mean close to $2/\lambda$. Reason: the time since the last bus (the age, truncated at $t$) and the time until the next (the residual) are independent $\text{Exp}(\lambda)$, by memorylessness looking forward and the reversibility of the Poisson process looking backward. You are more likely to land in a long gap: length-biased sampling. The same bias hits any renewal process — the sampled interval has density $x f(x)/\E T$ — and is the reason the mean residual service time in the <a href="#/t/queueing">M/G/1 queue</a> is $\E[S^2]/(2\E S)$, not $\E S/2$.</p>
</div>
<div class="warn" data-name="Overdispersion and clustering">
<p>The Poisson process forces $\Var N(t)=\E N(t)$. Real event counts (accidents, defaults, tweets) are often overdispersed because the rate itself is random or events trigger more events. A <em>Cox process</em> (doubly stochastic Poisson, rate $\Lambda(t)$ random) has $\Var N=\E N+\Var\Lambda$; a <em>Hawkes process</em> lets each event raise the intensity. Check the dispersion index of window counts before assuming Poisson; see <a href="#/t/overdispersion">overdispersion</a> for the regression analogue.</p>
</div>`,
      },
      {
        title: 'Beyond the line: Poisson random measures',
        html: R`
<p>Nothing above used the ordering of time except the gaps definition. A <strong>Poisson random measure</strong> on a space $E$ with mean measure $\mu$ is a random counting measure $N$ such that $N(A)\sim\text{Poisson}(\mu(A))$ and counts of disjoint sets are independent. Thinning, superposition, marking, and the conditional-uniformity property (given $N(A)=n$, the points are IID with law $\mu(\cdot\cap A)/\mu(A)$) all carry over unchanged. Two important instances:</p>
<ul>
<li><strong>Spatial point patterns</strong> (trees in a forest, galaxies): "complete spatial randomness" is a homogeneous Poisson process in the plane, the null model against which clustering is tested.</li>
<li><strong>Extremes</strong>: the points $\{(i/n, (X_i-b_n)/a_n)\}$ of a normalized IID sample converge to a Poisson process on $[0,1]\times(u,\infty)$. This single statement contains the <a href="#/t/fisher-tippett">Fisher–Tippett</a> limit (no points above $x$) and the <a href="#/t/peaks-over-threshold">peaks-over-threshold</a> model; see <a href="#/t/point-process-evt">the point-process approach to extremes</a>.</li>
</ul>
<p>Within stochastic processes, the Poisson process is the driving noise of every <a href="#/t/ctmc">continuous-time Markov chain</a> (each transition is a ringing exponential clock) and, after centring and scaling, converges to <a href="#/t/brownian-motion">Brownian motion</a>: $(N(nt)-\lambda nt)/\sqrt{\lambda n}\Rightarrow B(t)$.</p>`,
      },
    ],
    quiz: [
      { q: R`Customers arrive as a Poisson process of rate 6 per hour. Given that exactly 3 arrived between 9:00 and 10:00, what is the probability that all three arrived before 9:30?`,
        a: R`By the order-statistics property, given $N=3$ the three times are IID uniform on the hour, so the probability is $(1/2)^3=1/8$. The rate 6 is irrelevant once the count is known — that is the content of the theorem.` },
      { q: R`Calls arrive at rate $\lambda$; each is independently a sales call with probability $0.3$. In an hour there were 40 support calls. What is the expected number of sales calls in that hour?`,
        a: R`$0.3\lambda$. By the thinning theorem the sales and support streams are <em>independent</em> Poisson processes, so the support count carries no information about the sales count. (With a fixed total — e.g. binomial thinning of a fixed $n$ — the two counts would be negatively correlated.)` },
      { q: R`Why is $\Var\big(\sum_{i=1}^{N(t)}Y_i\big)=\lambda t\,\E[Y^2]$ rather than $\lambda t\Var(Y)$?`,
        a: R`Total variance: $\Var X=\E[\Var(X\mid N)]+\Var(\E[X\mid N])=\E N\Var Y+(\E Y)^2\Var N$. For the Poisson, $\Var N=\E N=\lambda t$, so the sum is $\lambda t(\Var Y+(\E Y)^2)=\lambda t\E[Y^2]$. The randomness of the number of terms contributes the $(\E Y)^2$ part.` },
      { q: R`You want to simulate an NHPP with $\lambda(t)=5+4\sin t$ on $[0,20]$ by thinning a homogeneous process of rate $\lambda^*=9$. What fraction of candidates is accepted, approximately, and what goes wrong if you use $\lambda^*=7$?`,
        a: R`Acceptance rate $\Lambda(T)/(\lambda^*T)=\big(100+4(1-\cos 20)\big)/180\approx 0.57$ (with $t$ in radians). With $\lambda^*=7<\sup\lambda=9$ the "probability" $\lambda(s)/\lambda^*$ exceeds 1 near the peaks; capping it at 1 produces an NHPP with intensity $\min(\lambda,7)$ — the peaks are silently clipped and the simulated process is wrong.` },
      { q: R`Show that the first-arrival time of the superposition of independent Poisson processes with rates $\lambda_1,\lambda_2$ is independent of which process produced it.`,
        a: R`$\P(\min(E_1,E_2)>t,\ E_1< E_2)=\int_t^\infty\lambda_1e^{-\lambda_1 s}e^{-\lambda_2 s}\d s=\frac{\lambda_1}{\lambda_1+\lambda_2}e^{-(\lambda_1+\lambda_2)t}$, a product of a function of the index and a function of $t$. So the minimum is $\text{Exp}(\lambda_1+\lambda_2)$ and the identity of the winner is independent of it.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'ctmc',
    domain: 'stochastic',
    order: 80,
    title: 'Continuous-Time Markov Chains & Birth–Death Processes',
    subtitle: 'Exponential holding times, the generator Q, and the matrix exponential',
    level: 2,
    sources: { aos: '—', cb: '—', other: 'Norris, Markov Chains (1997), Ch. 2–3; Ross, Introduction to Probability Models, Ch. 6' },
    prereqs: ['markov-chains', 'poisson-process'],
    related: ['stationary-distributions', 'queueing', 'branching-processes', 'chain-classification'],
    tags: ['generator', 'Kolmogorov equations', 'matrix exponential', 'Gillespie', 'detailed balance'],
    summary: R`A continuous-time Markov chain (CTMC) jumps between discrete states at random times: it sits in state $i$ for an exponential time with rate $q_i$, then jumps to $j$ with probability proportional to a rate $q_{ij}$. All of this is encoded in a single <em>generator matrix</em> $Q$, whose rows sum to zero. The transition probabilities solve the Kolmogorov differential equations $P'=QP=PQ$, so on a finite space $P(t)=e^{tQ}$; stationary distributions solve $\pi Q=0$; and for birth–death chains a one-line cut argument gives $\pi$ in closed form. CTMCs are the language of queues, chemical kinetics, epidemics, population genetics, and phylogenetic substitution models.`,
    survey: {
      task: 'Model discrete-state systems evolving in continuous time; compute transient and long-run behaviour',
      assumptions: R`Markov property; time-homogeneous rates; non-explosive (finitely many jumps in finite time)`,
      output: R`Generator $Q$; $P(t)=e^{tQ}$; stationary $\pi$ with $\pi Q=0$; simulated paths`,
      strengths: 'Exponential clocks make everything tractable; birth–death chains solved in closed form',
      weaknesses: 'Exponential holding times are often unrealistic; infinite state spaces can explode',
      complexity: R`$e^{tQ}$ costs $O(d^3)$; Gillespie simulation $O(\text{number of jumps}\times d)$`,
    },
    sections: [
      {
        title: 'From discrete to continuous time',
        html: R`
<p>A process $\{X_t:t\ge0\}$ on a countable state space $\mathcal S$ with right-continuous piecewise-constant paths is a <strong>(time-homogeneous) CTMC</strong> if for all $s,t\ge0$
$$\P(X_{s+t}=j\mid X_u,\,u\le s)=P_{ij}(t)\quad\text{on }\{X_s=i\}.$$
The matrices $P(t)=(P_{ij}(t))$ are stochastic and form a <strong>semigroup</strong>: $P(0)=I$ and $P(s+t)=P(s)P(t)$ (Chapman–Kolmogorov, by conditioning on $X_s$). Unlike the discrete-time case there is no "one-step" matrix to start from; the semigroup is generated by its derivative at zero.</p>
<div class="thm" data-kind="Proposition" data-name="Holding times are exponential">
<p>Started at $i$, the time $\tau=\inf\{t:X_t\ne i\}$ until the first jump is exponential with some rate $q_i\in[0,\infty)$ (with $q_i=0$ meaning $i$ is absorbing).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $g(t)=\P_i(\tau>t)$. The event $\{\tau>s\}$ is determined by the path up to $s$ and implies $X_s=i$, so by the Markov property $\P_i(\tau>s+t\mid\tau>s)=\P_i(\tau>t)$, i.e. $g(s+t)=g(s)g(t)$. Right-continuity of paths makes $g$ right-continuous with $g(0)=1$; the only such non-increasing solutions of Cauchy's equation are $g(t)=e^{-q_it}$.</p>
</details>
<p>Memorylessness is forced: a non-exponential holding time would let the process "remember" how long it has been in a state, violating the Markov property. Processes with general holding times are <em>semi-Markov</em>.</p>`,
      },
      {
        title: 'The generator and the jump-chain construction',
        html: R`
<div class="def" data-name="Generator (Q-matrix)">
<p>The generator is $Q=P'(0)$, with entries $q_{ij}=\lim_{h\downarrow0}P_{ij}(h)/h\ge0$ for $j\ne i$ (the <em>rate</em> of jumping from $i$ to $j$) and $q_{ii}=-q_i$. For a conservative chain the rows sum to zero: $\sum_{j\ne i}q_{ij}=q_i$. Thus $P(h)=I+hQ+o(h)$.</p>
</div>
<p>$Q$ has a direct probabilistic meaning. Define the <strong>jump chain</strong> $J_{ij}=q_{ij}/q_i$ for $j\ne i$ (and $J_{ii}=0$ when $q_i>0$).</p>
<div class="thm" data-kind="Theorem" data-name="Jump-chain / holding-time construction">
<p>Let $Y_0,Y_1,\dots$ be a discrete-time Markov chain with transition matrix $J$, and given $Y$, let $E_0,E_1,\dots$ be independent with $E_k\sim\text{Exp}(q_{Y_k})$. Setting $X_t=Y_k$ for $E_0+\dots+E_{k-1}\le t< E_0+\dots+E_k$ produces a CTMC with generator $Q$ (up to the explosion time, see below). Equivalently: in state $i$, start independent clocks $\text{Exp}(q_{ij})$, one per target $j$; jump to whichever rings first.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The two descriptions agree by the competing-exponentials lemma from <a href="#/t/poisson-process">the Poisson process</a>: the minimum of independent $\text{Exp}(q_{ij})$ is $\text{Exp}(q_i)$, the argmin is $j$ with probability $q_{ij}/q_i$, and the two are independent. Markov property: at a fixed time $s$ in state $i$, the residual holding time is again $\text{Exp}(q_i)$ by memorylessness, independent of the past, so the future evolves as a fresh copy started at $i$. The small-$h$ behaviour $\P_i(X_h=j)=q_{ij}h+o(h)$ follows since one jump to $j$ in $[0,h]$ has probability $(1-e^{-q_ih})q_{ij}/q_i=q_{ij}h+o(h)$ and two or more jumps have probability $O(h^2)$.</p>
</details>
<p>This construction is also the standard exact simulation method, known in chemistry as the Gillespie algorithm (1977):</p>
<div class="algo" data-name="Gillespie / stochastic simulation algorithm">
<ol>
<li>In state $i$ with total rate $q_i=\sum_{j\ne i}q_{ij}$, draw the holding time $E\sim\text{Exp}(q_i)$; advance $t\leftarrow t+E$.</li>
<li>Choose the next state $j$ with probability $q_{ij}/q_i$; set $X\leftarrow j$. Repeat until $t>T$.</li>
</ol>
</div>
<div class="warn" data-name="Explosion">
<p>On an infinite state space the jump times may accumulate. A pure birth process with rates $\lambda_n$ has explosion time $\zeta=\sum_n E_n$, $E_n\sim\text{Exp}(\lambda_n)$; since $\E\zeta=\sum_n1/\lambda_n$, if this is finite (e.g. $\lambda_n=n^2$) the chain makes infinitely many jumps before a finite time almost surely. Conversely, $\sum 1/\lambda_n=\infty$ implies $\zeta=\infty$ a.s. A chain is <em>non-explosive</em> if $\zeta=\infty$ a.s. from every state; this holds on finite spaces, when $\sup_i q_i<\infty$, or when the jump chain is recurrent. All results below assume non-explosion.</p>
</div>`,
        lab: 'ctmc-gillespie',
      },
      {
        title: 'Kolmogorov equations and the matrix exponential',
        html: R`
<p>Differentiate the semigroup property on either side:</p>
<div class="thm" data-kind="Theorem" data-name="Kolmogorov backward and forward equations">
<p>For a non-explosive chain,
$$\text{backward: }P'(t)=QP(t),\qquad\text{forward: }P'(t)=P(t)Q,$$
with $P(0)=I$. On a finite state space both hold and the unique solution is
$$P(t)=e^{tQ}=\sum_{k=0}^\infty\frac{t^kQ^k}{k!}.$$</p>
</div>
<details class="proof"><summary>Proof (finite state space)</summary>
<p>$P(t+h)-P(t)=(P(h)-I)P(t)=P(t)(P(h)-I)$. Dividing by $h$ and using $(P(h)-I)/h\to Q$ (finitely many entries, so no interchange issues) gives both equations. The linear ODE $P'=QP$, $P(0)=I$ has the unique solution $e^{tQ}$; that it is stochastic follows from $Q\mathbf 1=0$ (so $e^{tQ}\mathbf 1=\mathbf1$) and non-negativity of $e^{tQ}=e^{-ct}e^{t(Q+cI)}$ for $c\ge\max_iq_i$, since $Q+cI$ has non-negative entries.</p>
</details>
<p>The names come from which variable is differentiated: the backward equation conditions on the <em>first</em> jump (the "backward" time $0$), the forward equation on the <em>last</em> jump before $t$. In words, the forward equation reads $\frac{\d}{\d t}P_{ij}=\sum_{k\ne j}P_{ik}q_{kj}-P_{ij}q_j$: probability flows into $j$ from every $k$ at rate $q_{kj}$ and leaks out at rate $q_j$. For infinite state spaces the backward equation holds for the minimal chain under very general conditions, while the forward equation can fail without extra assumptions (e.g. $\sup_i q_i<\infty$) — a classical subtlety due to Feller.</p>
<div class="ex" data-name="The two-state chain">
<p>With $Q=\begin{pmatrix}-\alpha&\alpha\\\beta&-\beta\end{pmatrix}$, the eigenvalues are $0$ and $-(\alpha+\beta)$, and
$$P(t)=\frac{1}{\alpha+\beta}\begin{pmatrix}\beta+\alpha e^{-(\alpha+\beta)t}&\alpha-\alpha e^{-(\alpha+\beta)t}\\\beta-\beta e^{-(\alpha+\beta)t}&\alpha+\beta e^{-(\alpha+\beta)t}\end{pmatrix}.$$
Each row converges to $\pi=\big(\tfrac{\beta}{\alpha+\beta},\tfrac{\alpha}{\alpha+\beta}\big)$ exponentially fast at rate $\alpha+\beta$, the <em>spectral gap</em>. In general, if $Q$ is diagonalizable, $P(t)=\sum_k e^{\lambda_kt}r_k\ell_k^\top$ and the slowest nonzero eigenvalue sets the relaxation time.</p>
</div>
<div class="note" data-name="Uniformization">
<p>Choose $\gamma\ge\max_iq_i$ and set $\tilde P=I+Q/\gamma$, a stochastic matrix. Since $Q=\gamma(\tilde P-I)$ and $I$ commutes with everything,
$$P(t)=e^{-\gamma t}e^{\gamma t\tilde P}=\sum_{k=0}^\infty e^{-\gamma t}\frac{(\gamma t)^k}{k!}\tilde P^k.$$
Probabilistically: run a discrete chain $\tilde P$ (which may "jump" to itself) at the epochs of a Poisson($\gamma$) process. This is a numerically stable way to compute $e^{tQ}$ (all terms non-negative) and the basis of exact simulation and of proofs that transfer discrete-time results to continuous time.</p>
</div>`,
        lab: 'ctmc-transient',
      },
      {
        title: 'Stationary distributions and the ergodic theorem',
        html: R`
<p>A distribution $\pi$ is <strong>stationary</strong> if $\pi P(t)=\pi$ for all $t$. On a finite space this is equivalent to the <strong>global balance</strong> equations
$$\pi Q=0\iff\sum_{i\ne j}\pi_iq_{ij}=\pi_jq_j\quad\text{for every }j,$$
(differentiate $\pi P(t)=\pi$ at $0$; conversely $\pi Q=0$ gives $\pi e^{tQ}=\pi$ term by term): the probability flux into each state equals the flux out.</p>
<div class="thm" data-kind="Proposition" data-name="Relation to the jump chain">
<p>$\pi Q=0$ if and only if $\nu_i\propto\pi_iq_i$ is stationary for the jump chain $J$. Equivalently $\pi_j\propto\nu_j/q_j$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$(\pi Q)_j=\sum_{i\ne j}\pi_iq_{ij}-\pi_jq_j=\sum_{i\ne j}(\pi_iq_i)J_{ij}-\pi_jq_j$. So $\pi Q=0$ iff $\mu=(\pi_iq_i)_i$ satisfies $\mu J=\mu$.</p>
</details>
<p>The two stationary distributions differ because the jump chain counts <em>visits</em> while $\pi$ measures <em>time</em>: a state visited often but left quickly (large $q_j$) has large $\nu_j$ and small $\pi_j$. A consequence that trips people up: positive recurrence of the jump chain is neither necessary nor sufficient for positive recurrence of the CTMC, because the holding-time weights $1/q_j$ can be unbounded.</p>
<div class="thm" data-kind="Theorem" data-name="Ergodic theorem for CTMCs">
<p>Let $X$ be irreducible and non-explosive with a stationary distribution $\pi$ (equivalently, positive recurrent). Then for all $i,j$:
$$P_{ij}(t)\to\pi_j\quad\text{as }t\to\infty,\qquad \frac1t\int_0^t\ind(X_s=j)\d s\asto\pi_j=\frac{1}{q_j\,m_j},$$
where $m_j$ is the mean return time to $j$. There is no periodicity condition.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>For any $h>0$ the skeleton chain $Z_n=X_{nh}$ is a discrete-time chain with matrix $P(h)$; it is irreducible and <em>aperiodic</em> because $P_{ii}(h)\ge e^{-q_ih}>0$ (stay put with positive probability), and $\pi$ is stationary for it. The discrete <a href="#/t/stationary-distributions">convergence theorem</a> gives $P_{ij}(nh)\to\pi_j$; uniform continuity of $t\mapsto P_{ij}(t)$ (since $|P_{ij}(t+s)-P_{ij}(t)|\le1-e^{-q_is}$) fills in between grid points. For the time average, successive returns to $j$ split the path into IID cycles; in each, the time spent at $j$ has mean $1/q_j$ and the cycle length has mean $m_j$, so the renewal-reward theorem gives the limit $(1/q_j)/m_j$.</p>
</details>`,
      },
      {
        title: 'Birth–death processes',
        html: R`
<p>A <strong>birth–death process</strong> on $\{0,1,2,\dots\}$ only moves to neighbours: $q_{n,n+1}=\lambda_n$ (births), $q_{n,n-1}=\mu_n$ (deaths, $\mu_0=0$). Queues, population sizes, epidemics in their early phase, and the number of busy phone lines are all birth–death chains.</p>
<div class="thm" data-kind="Theorem" data-name="Stationary distribution of a birth–death chain">
<p>Suppose all $\lambda_n,\mu_{n+1}>0$. A stationary distribution exists if and only if
$$\sum_{n=0}^\infty\theta_n<\infty,\qquad\theta_0=1,\quad\theta_n=\prod_{k=1}^n\frac{\lambda_{k-1}}{\mu_k},$$
in which case $\pi_n=\theta_n/\sum_m\theta_m$, and $\pi$ satisfies <strong>detailed balance</strong> $\pi_n\lambda_n=\pi_{n+1}\mu_{n+1}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Global balance at $0$ reads $\pi_0\lambda_0=\pi_1\mu_1$. At $n\ge1$: $\pi_{n-1}\lambda_{n-1}+\pi_{n+1}\mu_{n+1}=\pi_n(\lambda_n+\mu_n)$. If $\pi_{n-1}\lambda_{n-1}=\pi_n\mu_n$ (induction hypothesis), these terms cancel and $\pi_{n+1}\mu_{n+1}=\pi_n\lambda_n$. So any solution of $\pi Q=0$ satisfies detailed balance, which forces $\pi_n=\pi_0\theta_n$; normalizability is exactly $\sum\theta_n<\infty$. A more transparent argument: in equilibrium, the long-run rate of crossings of the "cut" between $\{0,\dots,n\}$ and $\{n+1,\dots\}$ upward ($\pi_n\lambda_n$) must equal the rate downward ($\pi_{n+1}\mu_{n+1}$), because up- and down-crossings alternate.</p>
</details>
<table>
<thead><tr><th>Model</th><th>$\lambda_n$</th><th>$\mu_n$</th><th>Stationary law</th></tr></thead>
<tbody>
<tr><td>Single-server queue (M/M/1)</td><td>$\lambda$</td><td>$\mu$</td><td>Geometric: $(1-\rho)\rho^n$, $\rho=\lambda/\mu<1$</td></tr>
<tr><td>Immigration–death (M/M/∞)</td><td>$\lambda$</td><td>$n\mu$</td><td>$\text{Poisson}(\lambda/\mu)$</td></tr>
<tr><td>Finite source (Ehrenfest type)</td><td>$\lambda(N-n)$</td><td>$n\mu$</td><td>$\text{Bin}\big(N,\tfrac{\lambda}{\lambda+\mu}\big)$</td></tr>
<tr><td>Linear birth–death with immigration</td><td>$\nu+nb$</td><td>$nd$</td><td>Negative binomial, size $\nu/b$, needs $b< d$</td></tr>
</tbody>
</table>
<div class="ex" data-name="Verifying the Poisson case">
<p>For $\lambda_n=\lambda$, $\mu_n=n\mu$: $\theta_n=\prod_{k=1}^n\frac{\lambda}{k\mu}=\frac{(\lambda/\mu)^n}{n!}$, so $\pi_n=e^{-\lambda/\mu}(\lambda/\mu)^n/n!$. Compare the M/G/∞ result from <a href="#/t/poisson-process">thinning</a>: the Poisson stationary law does not even need exponential service.</p>
</div>
<p>Detailed balance means the stationary chain is <strong>time-reversible</strong>: $(X_t)$ and $(X_{T-t})$ have the same law. Every birth–death chain with a stationary distribution is reversible. This is the same property that <a href="#/t/metropolis-hastings">Metropolis–Hastings</a> engineers in discrete time, and its continuous-time analogue underlies <a href="#/t/langevin-mala">Langevin dynamics</a>. A useful consequence (Kelly, Burke): the departure process of a stationary M/M/1 queue, being the time-reversal of its arrival process, is itself Poisson($\lambda$).</p>`,
      },
      {
        title: 'Where CTMCs appear, and what to watch for',
        html: R`
<ul>
<li><strong>Phylogenetics.</strong> DNA substitution models (Jukes–Cantor, HKY, GTR) are 4-state reversible CTMCs; the likelihood of a tree multiplies matrices $e^{tQ}$ along branches.</li>
<li><strong>Chemical kinetics and epidemics.</strong> Reaction networks and the stochastic SIR model are CTMCs on counts, simulated with Gillespie; their large-population limits are ODEs (Kurtz's theorem), with Gaussian fluctuations described by <a href="#/t/stochastic-calculus">SDEs</a>.</li>
<li><strong>Branching.</strong> Continuous-time <a href="#/t/branching-processes">branching processes</a> (each individual splits at rate $b$, dies at rate $d$) are linear birth–death chains; extinction is certain iff $b\le d$.</li>
<li><strong>Survival and reliability.</strong> Multi-state models (healthy → ill → dead) are CTMCs; their transition intensities are hazard functions.</li>
</ul>
<div class="warn" data-name="Common errors">
<p>(1) Using the jump chain's stationary distribution as the time-fraction distribution: it must be reweighted by mean holding times $1/q_j$. (2) Simulating by small fixed time steps with $\P(\text{jump})\approx q_ih$ — biased and slow; Gillespie is exact. (3) Computing $e^{tQ}$ by naive series summation for large $t\|Q\|$ — catastrophic cancellation; use uniformization or scaling-and-squaring. (4) Assuming non-explosion on infinite spaces with superlinear rates.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A 3-state chain has $q_1=1$, $q_2=10$, $q_3=1$ and its jump chain is symmetric random walk on the triangle (each jump to either other state with probability $1/2$). What fraction of time does the CTMC spend in state 2?`,
        a: R`The jump chain's stationary law is uniform, $\nu=(1/3,1/3,1/3)$. Reweight by mean holding time: $\pi_j\propto\nu_j/q_j=(1,0.1,1)$, so $\pi_2=0.1/2.1\approx0.048$. State 2 is visited as often as the others but left ten times faster.` },
      { q: R`Why does the ergodic theorem for CTMCs not need an aperiodicity assumption?`,
        a: R`Holding times are continuous, so returns can occur at any time; formally, $P_{ii}(t)\ge e^{-q_it}>0$ for every $t$, so every skeleton chain $X_{nh}$ is aperiodic. Periodicity is an artefact of a discrete clock.` },
      { q: R`For the birth–death chain with $\lambda_n=\lambda$ and $\mu_n=\mu$ (all $n\ge1$), find the condition for a stationary distribution and the distribution itself.`,
        a: R`$\theta_n=(\lambda/\mu)^n$, summable iff $\rho=\lambda/\mu<1$; then $\pi_n=(1-\rho)\rho^n$, a geometric law on $\{0,1,\dots\}$ with mean $\rho/(1-\rho)$. This is the M/M/1 queue.` },
      { q: R`What is uniformization, and why is it preferred to summing $\sum_k t^kQ^k/k!$ directly?`,
        a: R`Write $P(t)=\sum_k e^{-\gamma t}\frac{(\gamma t)^k}{k!}\tilde P^k$ with $\tilde P=I+Q/\gamma$ stochastic for $\gamma\ge\max q_i$. Every term is non-negative, so there is no cancellation, truncation error is controlled by a Poisson tail, and it has a probabilistic reading (a discrete chain driven by Poisson($\gamma$) epochs). The raw series alternates in sign because $Q$ has negative diagonal and suffers catastrophic cancellation for large $t$.` },
      { q: R`A pure birth process has $\lambda_n=2^n$. Starting from $0$, what is the expected time to reach "infinity", and what does this imply?`,
        a: R`$\E\zeta=\sum_{n\ge0}2^{-n}=2<\infty$, so the process explodes in finite time almost surely. Without specifying what happens after $\zeta$, $Q$ does not determine a unique honest transition function; non-explosion must be checked before using the Kolmogorov equations or ergodic results.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'queueing',
    domain: 'stochastic',
    order: 90,
    title: 'Queueing Theory',
    subtitle: "Little's law, the M/M/1 queue, Erlang formulas, and why congestion explodes as load approaches capacity",
    level: 2,
    sources: { aos: '—', cb: '—', other: 'Kleinrock, Queueing Systems Vol. 1 (1975); Harchol-Balter, Performance Modeling and Design of Computer Systems (2013)' },
    prereqs: ['ctmc'],
    related: ['poisson-process', 'simulation-studies', 'block-bootstrap'],
    tags: ["Little's law", 'M/M/1', 'Erlang C', 'Pollaczek–Khinchine', 'heavy traffic'],
    summary: R`Queueing theory studies systems where customers arrive, wait, get served, and leave — call centres, CPUs, hospital beds, network routers. Its first law, $L=\lambda W$, holds for essentially any stable system and links the average number present to the average time spent. Its first model, the M/M/1 queue, is a birth–death chain with a geometric stationary law, giving mean delay $1/(\mu-\lambda)$: delay grows without bound as utilization $\rho=\lambda/\mu\to1$. Multi-server systems lead to the Erlang B and C formulas used to staff call centres, and the Pollaczek–Khinchine formula shows that <em>variability</em> of service times, not just their mean, drives congestion.`,
    survey: {
      task: 'Predict waiting times, queue lengths, blocking probabilities; size capacity',
      assumptions: R`Stationarity ($\rho<1$); for closed forms: Poisson arrivals, often exponential service, FCFS`,
      output: R`$L,\,L_q,\,W,\,W_q$; stationary queue-length law; $P(\text{wait})$, $P(\text{block})$`,
      strengths: "Little's law is distribution-free; exact formulas for Markovian queues; sharp heavy-traffic insight",
      weaknesses: 'Closed forms need Poisson/exponential assumptions; transients and non-stationary loads need simulation',
      complexity: R`Formulas $O(c)$; simulation needs long runs since relaxation time $\sim(1-\rho)^{-2}$`,
    },
    sections: [
      {
        title: 'Vocabulary and Kendall notation',
        html: R`
<p>A queue is described by <strong>A/S/c</strong> (Kendall, 1953): the inter-arrival distribution A, the service-time distribution S, and the number of servers $c$, with M = Markovian (exponential), D = deterministic, G = general. Extra fields give capacity and discipline (FCFS unless stated). The basic quantities:</p>
<table>
<thead><tr><th>Symbol</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td>$\lambda$, $\mu$</td><td>arrival rate; service rate per server (mean service $\E S=1/\mu$)</td></tr>
<tr><td>$\rho=\lambda/(c\mu)$</td><td>utilization: long-run fraction of time a server is busy</td></tr>
<tr><td>$L$, $L_q$</td><td>mean number in system; mean number waiting (not in service)</td></tr>
<tr><td>$W$, $W_q$</td><td>mean sojourn time (wait + service); mean waiting time</td></tr>
</tbody>
</table>
<p>The stability condition is $\rho<1$: the servers must be able to work faster than work arrives. At $\rho\ge1$ the queue length is not positive recurrent (null recurrent at $\rho=1$ in the M/M/1 case, transient above) and no stationary regime exists, no matter how large the waiting room.</p>`,
      },
      {
        title: "Little's law",
        html: R`
<div class="thm" data-kind="Theorem" data-name="Little's law (Little, 1961; Stidham, 1974)">
<p>Consider any system that customers enter and leave. Let $N(t)$ be the number present, $A(t)$ the number of arrivals by $t$, and $W_k$ the time customer $k$ spends inside. If the limits $\lambda=\lim A(t)/t$ and $W=\lim\frac1n\sum_{k\le n}W_k$ exist and are finite (and the system empties often enough that $W_k/k\to0$), then $L=\lim\frac1t\int_0^tN(s)\d s$ exists and
$$L=\lambda W.$$</p>
</div>
<details class="proof"><summary>Proof sketch (sample path)</summary>
<p>Picture each customer as a horizontal bar of length $W_k$ from arrival to departure. The area under $N(s)$ on $[0,t]$ is the total length of bar inside $[0,t]$. Every customer that has both arrived and departed by $t$ contributes fully, and only customers arriving by $t$ contribute at all:
$$\sum_{k:\,\text{departed by }t}W_k\;\le\;\int_0^tN(s)\d s\;\le\;\sum_{k\le A(t)}W_k.$$
Divide by $t$. The right side is $\frac{A(t)}{t}\cdot\frac{1}{A(t)}\sum_{k\le A(t)}W_k\to\lambda W$; the condition $W_k/k\to0$ ensures the customers still inside at $t$ are negligible, so the left side has the same limit.</p>
</details>
<p>The law requires no distributional assumptions, no particular discipline, and applies to any "box": the whole system ($L=\lambda W$), the waiting room ($L_q=\lambda W_q$), or the servers alone (mean number in service $=\lambda\E S$, so for one server $\P(\text{busy})=\lambda/\mu=\rho$). It is the most-used identity in operations: if a warehouse holds 2,000 pallets and ships 400 a day, pallets stay 5 days on average.</p>
<div class="warn" data-name="Rates must match the box">
<p>$\lambda$ is the rate of customers that actually <em>enter</em> the box. In a system with blocking or balking, use the effective arrival rate $\lambda(1-P_{\text{block}})$, not the offered rate.</p>
</div>`,
      },
      {
        title: 'The M/M/1 queue',
        html: R`
<p>Poisson arrivals at rate $\lambda$, exponential service at rate $\mu$, one server. The number in system is the <a href="#/t/ctmc">birth–death chain</a> with $\lambda_n=\lambda$, $\mu_n=\mu$, so for $\rho=\lambda/\mu<1$:</p>
$$\pi_n=(1-\rho)\rho^n,\qquad L=\frac{\rho}{1-\rho},\qquad \Var(N)=\frac{\rho}{(1-\rho)^2},\qquad L_q=\frac{\rho^2}{1-\rho}.$$
<p>By Little's law, $W=L/\lambda=\dfrac{1}{\mu-\lambda}$ and $W_q=\dfrac{\rho}{\mu-\lambda}$. The factor $1/(1-\rho)$ is the whole story of congestion: going from 80% to 90% utilization doubles the mean delay; from 90% to 99% multiplies it by ten.</p>
<div class="thm" data-kind="Theorem" data-name="PASTA (Wolff, 1982)">
<p><strong>P</strong>oisson <strong>A</strong>rrivals <strong>S</strong>ee <strong>T</strong>ime <strong>A</strong>verages: if arrivals form a Poisson process and future arrivals are independent of the past state of the system (lack of anticipation), the long-run fraction of arrivals that find the system in state $n$ equals the long-run fraction of time it spends in state $n$.</p>
</div>
<p>PASTA is not automatic. In a D/D/1 queue with arrivals at integer times and service time $1/2$, the time-average number present is $1/2$ but every arrival finds the system empty. Poisson arrivals are "uninformed observers" because their independent increments cannot synchronize with the system.</p>
<div class="thm" data-kind="Proposition" data-name="Sojourn time distribution in M/M/1 (FCFS)">
<p>In stationarity, the sojourn time of an arriving customer is $\text{Exp}(\mu-\lambda)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By PASTA the arrival finds $n$ customers with probability $(1-\rho)\rho^n$. By memorylessness the one in service has a fresh $\text{Exp}(\mu)$ residual, so the sojourn time is a sum of $n+1$ IID $\text{Exp}(\mu)$ variables. $K=n+1$ is geometric on $\{1,2,\dots\}$ with success probability $1-\rho$, and a geometric($p$) sum of IID $\text{Exp}(\mu)$ variables is $\text{Exp}(p\mu)$ (compare Laplace transforms: $\sum_k p(1-p)^{k-1}\big(\tfrac{\mu}{\mu+s}\big)^k=\tfrac{p\mu}{p\mu+s}$). With $p=1-\rho$, the rate is $\mu-\lambda$.</p>
</details>
<div class="note" data-name="Heavy traffic">
<p>As $\rho\uparrow1$, $(1-\rho)N$ converges in distribution to an $\text{Exp}(1)$ variable: queue lengths are of order $1/(1-\rho)$ and fluctuate on that scale. Worse, the relaxation time — how long the queue takes to forget its initial state — is $1/(\sqrt\mu-\sqrt\lambda)^2\approx4/(\mu(1-\rho)^2)$, the inverse spectral gap of the chain. A simulation at $\rho=0.95$ needs on the order of $10^3$ mean service times just to warm up, and time averages over a run are strongly autocorrelated: confidence intervals need batch means or a <a href="#/t/block-bootstrap">block bootstrap</a>, never IID formulas.</p>
</div>`,
        lab: 'queueing-mm1',
      },
      {
        title: 'Many servers: Erlang B and Erlang C',
        html: R`
<p>With $c$ servers, each at rate $\mu$, the death rate is $\mu_n=\min(n,c)\mu$. Let $a=\lambda/\mu$ be the <strong>offered load</strong> in Erlangs (mean number of busy servers if there were infinitely many) and $\rho=a/c$.</p>
<div class="thm" data-kind="Proposition" data-name="M/M/c and the Erlang C formula">
<p>For $\rho<1$, $\pi_n=\pi_0a^n/n!$ for $n\le c$ and $\pi_n=\pi_0\frac{a^c}{c!}\rho^{n-c}$ for $n\ge c$, where $\pi_0^{-1}=\sum_{k=0}^{c-1}\frac{a^k}{k!}+\frac{a^c}{c!(1-\rho)}$. The probability an arrival must wait is
$$C(c,a)=\P(N\ge c)=\frac{\frac{a^c}{c!(1-\rho)}}{\sum_{k=0}^{c-1}\frac{a^k}{k!}+\frac{a^c}{c!(1-\rho)}},\qquad L_q=C(c,a)\frac{\rho}{1-\rho},\qquad W_q=\frac{C(c,a)}{c\mu-\lambda}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\theta_n=\prod_{k=1}^n\lambda/(\min(k,c)\mu)$ equals $a^n/n!$ for $n\le c$ and $\frac{a^c}{c!}\rho^{n-c}$ beyond; summing the geometric tail gives $\pi_0$. By PASTA, $\P(\text{wait})=\sum_{n\ge c}\pi_n=\pi_0\frac{a^c}{c!(1-\rho)}$. Then $L_q=\sum_{n\ge c}(n-c)\pi_n=\pi_0\frac{a^c}{c!}\frac{\rho}{(1-\rho)^2}=C\frac{\rho}{1-\rho}$, and $W_q=L_q/\lambda$ simplifies to $C/(c\mu-\lambda)$.</p>
</details>
<p>If instead there is <em>no</em> waiting room (M/M/c/c, the telephone trunk model), an arrival finding all servers busy is lost. Truncating the Poisson-type law at $c$ gives the <strong>Erlang B</strong> (loss) formula
$$B(c,a)=\frac{a^c/c!}{\sum_{k=0}^c a^k/k!},$$
computed stably by the recursion $B(0,a)=1$, $B(c,a)=\frac{aB(c-1,a)}{c+aB(c-1,a)}$, with $C=\frac{cB}{c-a(1-B)}$. Remarkably, Erlang B is <strong>insensitive</strong>: it holds for M/G/c/c with any service distribution of mean $1/\mu$ (Sevastyanov, 1957).</p>
<div class="ex" data-name="Economies of scale: pooling">
<p>Compare $c$ separate M/M/1 queues, each with arrival rate $\lambda/c$, against one shared M/M/c queue with the same total load. Both have utilization $\rho$, but in the separate system a customer can wait while another server idles. With $c=10$ and $\rho=0.9$, each separate queue has $W_q=\rho/(\mu(1-\rho))=9/\mu$, while the pooled $W_q=C(10,9)/(10\mu-9\mu)=C(10,9)/\mu\approx0.67/\mu$, more than ten times smaller. Square-root staffing (Halfin & Whitt, 1981) makes this precise: with $c=a+\beta\sqrt a$, the waiting probability converges to a constant in $(0,1)$ as $a\to\infty$, so the spare capacity needed grows only like $\sqrt a$.</p>
</div>`,
        lab: 'queueing-erlang',
      },
      {
        title: 'General service: M/G/1 and the role of variability',
        html: R`
<p>Exponential service is a modelling convenience; real service times (file sizes, surgery durations) are often far more variable. For Poisson arrivals and IID general service $S$ with $\rho=\lambda\E S<1$:</p>
<div class="thm" data-kind="Theorem" data-name="Pollaczek–Khinchine mean-value formula">
<p>$$W_q=\frac{\lambda\E[S^2]}{2(1-\rho)}=\frac{\rho\,\E S}{1-\rho}\cdot\frac{1+c_s^2}{2},\qquad c_s^2=\frac{\Var S}{(\E S)^2}.$$</p>
</div>
<details class="proof"><summary>Proof sketch (mean-value argument)</summary>
<p>An arriving customer (FCFS) waits for the $N_q$ customers in the queue, each requiring a full service, plus the residual service $R$ of the customer being served, if any. By PASTA and independence of service times from the queue length, $W_q=\E[N_q]\E S+\E[\text{residual}]$. By PASTA the server is busy with probability $\rho$, and the residual of a service in progress is length-biased (the inspection paradox): its mean is $\E[S^2]/(2\E S)$. So $W_q=L_q\E S+\rho\frac{\E[S^2]}{2\E S}=\rho W_q+\frac{\lambda\E[S^2]}{2}$ using $L_q=\lambda W_q$. Solve for $W_q$.</p>
</details>
<p>Deterministic service ($c_s^2=0$) halves the waiting time of M/M/1 ($c_s^2=1$); heavy-tailed service with $\E[S^2]=\infty$ gives infinite mean waiting at <em>any</em> load. This is why a few huge jobs can wreck the latency of a FCFS server, and why size-based scheduling (shortest remaining processing time) or processor sharing helps. For general arrivals, <strong>Kingman's approximation</strong> (1961) extends the same structure:
$$W_q\approx\E S\cdot\frac{\rho}{1-\rho}\cdot\frac{c_a^2+c_s^2}{2},$$
exact in the heavy-traffic limit, where $c_a^2$ is the squared coefficient of variation of inter-arrival times. The three factors — time scale, utilization, variability — are the "VUT" equation of operations management.</p>`,
      },
      {
        title: 'Pitfalls and extensions',
        html: R`
<div class="warn" data-name="Designing for average load">
<p>Because delay scales like $1/(1-\rho)$, a system sized to run at 95% of capacity on average will be overwhelmed by modest fluctuations in demand. Time-varying arrival rates (an NHPP) make it worse: a queue that is stable on average can be overloaded during the peak, and the stationary formulas evaluated at the peak rate underestimate the lagged build-up.</p>
</div>
<div class="warn" data-name="Simulating queues naively">
<p>Starting a simulation empty biases time averages downward (initialization bias); discard a warm-up period. Successive queue lengths are highly correlated, so the effective sample size of a run is much smaller than its length. Use batch means or regenerative cycles (busy periods are IID) for honest standard errors; see <a href="#/t/simulation-studies">simulation studies</a>.</p>
</div>
<p><strong>Networks.</strong> Jackson networks (Poisson external arrivals, exponential servers, Markovian routing) have a product-form stationary distribution: each node behaves like an independent M/M/1 with its total arrival rate from the traffic equations. This rests on Burke's theorem (departures from a stationary M/M/1 are Poisson) and on reversibility. Beyond product form, one turns to fluid and diffusion limits — scaled queue lengths converge to reflected Brownian motion (see <a href="#/t/brownian-motion">Brownian motion</a>) — or to simulation.</p>`,
      },
    ],
    quiz: [
      { q: R`A coffee shop serves on average 60 customers per hour and there are on average 9 people inside. How long does a customer stay, on average? Which assumptions did you need?`,
        a: R`$W=L/\lambda=9/60$ hour $=9$ minutes. Little's law needs only that the long-run averages exist (stationarity); no assumptions on arrival or service distributions, number of servers, or discipline.` },
      { q: R`An M/M/1 server runs at $\rho=0.9$. Arrival rate rises by 5%. By what factor does the mean sojourn time change?`,
        a: R`$W=1/(\mu-\lambda)=1/(\mu(1-\rho))$. New $\rho=0.945$, so the factor is $(1-0.9)/(1-0.945)=0.1/0.055\approx1.82$. A 5% increase in load nearly doubles the delay.` },
      { q: R`For M/G/1 with mean service 1 minute and $\rho=0.8$, compare $W_q$ for deterministic, exponential, and service with $c_s^2=4$.`,
        a: R`$W_q=\frac{\rho}{1-\rho}\E S\frac{1+c_s^2}{2}=4\cdot\frac{1+c_s^2}{2}$ minutes: 2, 4, and 10 minutes respectively. Same mean, same load — fivefold difference driven purely by variability.` },
      { q: R`Why do PASTA and the Pollaczek–Khinchine argument both fail for a D/D/1 queue?`,
        a: R`Deterministic arrivals are synchronized with the system: with inter-arrival 1 and service 1/2, every arrival finds an empty system although the server is busy half the time. Arrivals do not sample the state "at random", so the fraction of arrivals seeing state $n$ differs from the fraction of time in $n$, which is the step the mean-value argument relies on.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'martingales',
    domain: 'stochastic',
    order: 100,
    title: 'Martingales & Optional Stopping',
    subtitle: 'Fair games, stopping times, convergence, and concentration',
    level: 3,
    sources: { aos: '—', cb: '—', other: 'Williams, Probability with Martingales (1991); Durrett, Probability: Theory and Examples, Ch. 4' },
    prereqs: ['conditional-expectation', 'random-walks'],
    related: ['inequalities', 'branching-processes', 'brownian-motion', 'stochastic-calculus', 'hypothesis-testing'],
    tags: ['filtration', 'stopping time', 'optional stopping', "Wald's identity", 'Azuma–Hoeffding', 'Doob'],
    summary: R`A martingale is the mathematical form of a fair game: given everything known so far, the expected next value equals the current value. That single condition, $\E[X_{n+1}\mid\mathcal F_n]=X_n$, has remarkable consequences. You cannot turn a fair game into a favourable one by clever bet sizing or by choosing when to stop — <em>provided</em> the stopping rule and stakes are suitably bounded (optional stopping), and the doubling strategy shows exactly what goes wrong when they are not. Bounded martingales converge (Doob), martingales with bounded increments concentrate like sums of independent variables (Azuma–Hoeffding), and the identity $\E\sum_{i\le\tau}X_i=\E\tau\,\E X$ (Wald) follows in a line. Martingales are the backbone of sequential analysis, stochastic calculus, and modern concentration-of-measure arguments.`,
    survey: {
      task: 'Analyse fair games, hitting probabilities, stopped sums, limits, and deviations of adaptive processes',
      assumptions: R`Integrability; adaptedness to a filtration; boundedness / UI conditions for stopping and convergence`,
      output: R`Identities $\E X_\tau=\E X_0$; a.s. limits; tail bounds $e^{-t^2/(2\sum c_k^2)}$`,
      strengths: 'No independence needed; handles adaptive, sequential, data-dependent procedures',
      weaknesses: 'Conditions (boundedness, UI) are easy to violate silently; the doubling strategy is the canonical trap',
      complexity: R`Analytic tool — no computation; Azuma bounds are closed-form`,
    },
    sections: [
      {
        title: 'Filtrations and martingales',
        html: R`
<p>A <strong>filtration</strong> $\mathcal F_0\subseteq\mathcal F_1\subseteq\cdots$ is an increasing sequence of σ-fields: $\mathcal F_n$ is the information available at time $n$. A process $X_n$ is <strong>adapted</strong> if $X_n$ is $\mathcal F_n$-measurable (its value is known at time $n$).</p>
<div class="def" data-name="Martingale">
<p>An adapted, integrable process $(X_n)$ is a <strong>martingale</strong> if $\E[X_{n+1}\mid\mathcal F_n]=X_n$ for all $n$; a <strong>submartingale</strong> if $\E[X_{n+1}\mid\mathcal F_n]\ge X_n$; a <strong>supermartingale</strong> if $\le$. By the tower property a martingale has $\E[X_m\mid\mathcal F_n]=X_n$ for all $m\ge n$, hence constant mean $\E X_n=\E X_0$.</p>
</div>
<p>All of this uses <a href="#/t/conditional-expectation">conditional expectation</a> given a σ-field. (Super- is "unfavourable": a supermartingale is what a casino offers you.) The examples worth knowing:</p>
<ul>
<li><strong>Sums.</strong> $S_n=\sum_{i\le n}\xi_i$ with independent mean-zero $\xi_i$. If $\Var\xi_i=\sigma^2$, then $S_n^2-n\sigma^2$ is also a martingale.</li>
<li><strong>Products.</strong> $M_n=\prod_{i\le n}Y_i$ with independent $Y_i\ge0$, $\E Y_i=1$. Special case: the <strong>exponential martingale</strong> $e^{\theta S_n}/m(\theta)^n$ with $m(\theta)=\E e^{\theta\xi}$.</li>
<li><strong>Likelihood ratios.</strong> If $X_i\iid f$ and $g$ is another density, $L_n=\prod_{i\le n}g(X_i)/f(X_i)$ is a mean-one martingale under $f$, since $\E_f[g(X)/f(X)]=\int g=1$.</li>
<li><strong>Doob (information) martingale.</strong> For any integrable $Z$, $X_n=\E[Z\mid\mathcal F_n]$ is a martingale: the best current guess of $Z$ as information accumulates. Every <a href="#/t/bayesian-inference">posterior mean</a> sequence is of this form.</li>
<li><strong>Pólya's urn.</strong> Start with $a$ red and $b$ blue balls; draw one, return it with another of the same colour. The fraction of red balls is a martingale.</li>
<li><strong>Branching.</strong> In a <a href="#/t/branching-processes">Galton–Watson process</a> with mean offspring $m$, $Z_n/m^n$ is a martingale.</li>
</ul>`,
      },
      {
        title: "Stopping times and why you can't beat a fair game",
        html: R`
<div class="def" data-name="Stopping time">
<p>A random variable $\tau\in\{0,1,2,\dots,\infty\}$ is a <strong>stopping time</strong> if $\{\tau=n\}\in\mathcal F_n$ for every $n$: the decision to stop at $n$ uses only information available at $n$. "The first time the walk hits 10" is a stopping time; "the time of the walk's maximum over $[0,100]$" is not (it requires the future).</p>
</div>
<p>A gambler's stake on round $n$ must be decided before the round is played: a <strong>predictable</strong> process $H_n\in\mathcal F_{n-1}$. If $X$ is the price of a fair game, the gambler's wealth is the <strong>martingale transform</strong>
$$(H\cdot X)_n=\sum_{k=1}^nH_k(X_k-X_{k-1}).$$</p>
<div class="thm" data-kind="Theorem" data-name="No system beats a fair game">
<p>If $X$ is a martingale and $H$ is predictable and bounded, then $H\cdot X$ is a martingale. (For a supermartingale and $H\ge0$ bounded, $H\cdot X$ is a supermartingale.)</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$(H\cdot X)_n$ is adapted and integrable (bounded $H$, integrable $X$). And $\E[(H\cdot X)_{n+1}-(H\cdot X)_n\mid\mathcal F_n]=\E[H_{n+1}(X_{n+1}-X_n)\mid\mathcal F_n]=H_{n+1}\E[X_{n+1}-X_n\mid\mathcal F_n]=0$, pulling out the known factor $H_{n+1}\in\mathcal F_n$.</p>
</details>
<p>Taking $H_k=\ind(\tau\ge k)$, which is predictable because $\{\tau\ge k\}=\{\tau\le k-1\}^c\in\mathcal F_{k-1}$, gives $(H\cdot X)_n=X_{\tau\wedge n}-X_0$:</p>
<div class="thm" data-kind="Corollary" data-name="Stopped martingales">
<p>If $X$ is a martingale and $\tau$ a stopping time, the stopped process $X_{\tau\wedge n}$ is a martingale, so $\E X_{\tau\wedge n}=\E X_0$ for every $n$.</p>
</div>`,
      },
      {
        title: 'The optional stopping theorem and the doubling strategy',
        html: R`
<p>The corollary is about $\tau\wedge n$. The question is whether we can let $n\to\infty$ and conclude $\E X_\tau=\E X_0$. Not always.</p>
<div class="thm" data-kind="Theorem" data-name="Optional stopping (Doob)">
<p>Let $X$ be a martingale and $\tau$ a stopping time. Then $X_\tau$ is integrable and $\E X_\tau=\E X_0$ if any of the following holds:</p>
<ol>
<li>$\tau$ is bounded: $\tau\le N$ a.s. for a constant $N$;</li>
<li>$\tau< \infty$ a.s. and the stopped process is bounded: $|X_{\tau\wedge n}|\le K$ for all $n$;</li>
<li>$\E\tau< \infty$ and the increments are bounded: $|X_n-X_{n-1}|\le K$ for all $n$ (or merely $\E[|X_{n}-X_{n-1}|\mid\mathcal F_{n-1}]\le K$ on $\{\tau\ge n\}$).</li>
</ol>
<p>More generally it suffices that $\{X_{\tau\wedge n}\}$ is uniformly integrable and $\tau< \infty$ a.s.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Always $\E X_{\tau\wedge n}=\E X_0$. (1) Take $n=N$. (2) $X_{\tau\wedge n}\to X_\tau$ a.s. because $\tau< \infty$; dominated convergence with bound $K$. (3) $|X_{\tau\wedge n}-X_0|=\big|\sum_{k=1}^{\tau\wedge n}(X_k-X_{k-1})\big|\le\sum_{k=1}^\infty|X_k-X_{k-1}|\ind(\tau\ge k)=:Y$, and $\E Y=\sum_k\E\big[\ind(\tau\ge k)\E[|X_k-X_{k-1}|\mid\mathcal F_{k-1}]\big]\le K\sum_k\P(\tau\ge k)=K\E\tau< \infty$. Dominated convergence again.</p>
</details>
<div class="warn" data-name="The doubling strategy">
<p>Bet 1 unit on a fair coin; after each loss, double the stake; stop at the first win. If the first win is at round $\tau$, you have lost $1+2+\dots+2^{\tau-2}=2^{\tau-1}-1$ and then win $2^{\tau-1}$: net gain exactly $+1$, <em>always</em>. Since $\tau\sim\text{Geometric}(1/2)$ is a.s. finite (indeed $\E\tau=2$), it seems $\E X_\tau=X_0+1\ne X_0$ — a sure profit from a fair game. Which hypothesis fails? $\tau$ is unbounded (1 fails); the wealth before stopping is unbounded below (2 fails); $\E\tau< \infty$ but the increments $2^{n-1}$ are unbounded (3 fails). The loss you must be able to absorb before the win has expectation $\E[2^{\tau-1}-1]=\sum_{n\ge1}(2^{n-1}-1)2^{-n}=\infty$.</p>
<p>With any finite bankroll the paradox disappears. If you can survive at most $m$ losses (bankroll $2^m-1$), then with probability $1-2^{-m}$ you gain 1 and with probability $2^{-m}$ you lose $2^m-1$: expected gain $(1-2^{-m})-2^{-m}(2^m-1)=0$ exactly, as condition (1) demands. The strategy converts a fair game into frequent small wins and rare catastrophic losses — the profile of many real trading strategies. Try it in the lab.</p>
</div>
<div class="ex" data-name="Gambler's ruin by optional stopping">
<p>A simple symmetric <a href="#/t/random-walks">random walk</a> starts at $k\in\{0,\dots,N\}$ and stops at $\tau$, the first hit of $0$ or $N$. The stopped walk is bounded, so (2) gives $k=\E S_\tau=N\,\P(\text{hit }N)$, i.e. $\P(\text{hit }N)=k/N$. Applying (3) to the martingale $S_n^2-n$ (with $\E\tau< \infty$, which holds since from any state the walk exits within $N$ steps with probability at least $2^{-N}$): $\E S_\tau^2-\E\tau=k^2$, so $\E\tau=N^2\cdot\tfrac kN-k^2=k(N-k)$. For a biased walk ($p\ne q$), $(q/p)^{S_n}$ is a martingale and gives $\P(\text{hit }N)=\frac{1-(q/p)^k}{1-(q/p)^N}$.</p>
</div>`,
        lab: 'martingales-betting',
      },
      {
        title: "Wald's identities",
        html: R`
<div class="thm" data-kind="Theorem" data-name="Wald's identity (1944)">
<p>Let $\xi_1,\xi_2,\dots$ be IID with $\E|\xi|< \infty$, $\mu=\E\xi$, and let $\tau$ be a stopping time with respect to $\mathcal F_n=\sigma(\xi_1,\dots,\xi_n)$ with $\E\tau< \infty$. Then
$$\E\Big[\sum_{i=1}^\tau\xi_i\Big]=\mu\,\E\tau.$$
If moreover $\Var\xi=\sigma^2< \infty$ and $\mu=0$, then $\E\big[(\sum_{i\le\tau}\xi_i)^2\big]=\sigma^2\E\tau$ (Wald's second identity).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $\sum_{i=1}^\tau\xi_i=\sum_{i=1}^\infty\xi_i\ind(\tau\ge i)$. The event $\{\tau\ge i\}\in\mathcal F_{i-1}$ is independent of $\xi_i$. Hence $\E[|\xi_i|\ind(\tau\ge i)]=\E|\xi|\,\P(\tau\ge i)$ and $\sum_i\E|\xi|\P(\tau\ge i)=\E|\xi|\,\E\tau< \infty$, which justifies exchanging sum and expectation (Fubini). Then $\E\sum_i\xi_i\ind(\tau\ge i)=\mu\sum_i\P(\tau\ge i)=\mu\E\tau$. Equivalently: optional stopping, condition (3) in its conditional form, applied to the martingale $S_n-n\mu$.</p>
</details>
<p>Wald's identity is the workhorse of sequential analysis: in Wald's sequential probability ratio test, the expected sample size is approximated by dividing the expected log-likelihood ratio at the boundary by the per-observation drift $\pm\KL$. It also gives renewal-theory results: with $\tau=N(t)+1$ (a stopping time, unlike $N(t)$ itself), $\E S_{N(t)+1}=\mu(\E N(t)+1)$, the starting point of the elementary renewal theorem.</p>
<div class="warn" data-name="Stopping times only">
<p>$\E\sum_{i\le N}\xi_i=\mu\E N$ is false for general random $N$ correlated with the $\xi_i$. Take $N=1$ if $\xi_1>0$ and $N=2$ otherwise: $N$ looks at $\xi_1$ only, so it <em>is</em> a stopping time and Wald holds. But "$N=1$ if $\xi_2>0$" peeks ahead and breaks it.</p>
</div>`,
      },
      {
        title: 'Convergence: upcrossings, Doob, and uniform integrability',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Doob's martingale convergence theorem">
<p>If $X$ is a submartingale with $\sup_n\E X_n^+< \infty$ (for a martingale: $\sup_n\E|X_n|< \infty$), then $X_n\to X_\infty$ almost surely, with $\E|X_\infty|< \infty$. In particular every non-negative supermartingale converges a.s.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Fix $a< b$ and let $U_n[a,b]$ be the number of upcrossings of $[a,b]$ completed by time $n$. Doob's <strong>upcrossing inequality</strong> says $(b-a)\E U_n[a,b]\le\E(X_n-a)^+$. Proof idea: consider the strategy "buy when $X$ drops below $a$, sell when it rises above $b$"; each completed upcrossing earns at least $b-a$, and since the strategy is a predictable 0/1 stake on a submartingale, its expected gain is bounded by the final position's value $\E(X_n-a)^+$. So $\E U_\infty[a,b]< \infty$ and there are finitely many upcrossings a.s. If $X_n$ did not converge, there would be rationals $a< b$ with $\liminf X_n< a< b< \limsup X_n$, forcing infinitely many upcrossings — a null event for each of countably many pairs. Fatou gives $\E|X_\infty|\le\liminf\E|X_n|< \infty$.</p>
</details>
<p>Almost sure convergence does <em>not</em> imply $\E X_\infty=\E X_0$. The missing ingredient is uniform integrability:</p>
<div class="thm" data-kind="Theorem" data-name="L¹ convergence">
<p>For a martingale the following are equivalent: (a) $\{X_n\}$ is uniformly integrable; (b) $X_n\to X_\infty$ in $L^1$; (c) there is an integrable $Z$ with $X_n=\E[Z\mid\mathcal F_n]$ (and then $X_\infty=\E[Z\mid\mathcal F_\infty]$, Lévy's upward theorem). If $\sup_n\E X_n^2< \infty$, convergence also holds in $L^2$.</p>
</div>
<div class="ex" data-name="A martingale that converges to the wrong place">
<p>Likelihood ratio $L_n=\prod_{i\le n}g(X_i)/f(X_i)$ under $f\ne g$: non-negative, mean 1, so it converges a.s. By the SLLN, $\frac1n\log L_n\to-\KL(f\,\|\,g)< 0$, so $L_n\to0$ a.s. while $\E L_n=1$ for every $n$. The mass escapes to rare, huge values: not UI. Statistically this is good news — under the null, the likelihood ratio against a false alternative dies — and <strong>Ville's inequality</strong> $\P(\sup_nL_n\ge1/\alpha)\le\alpha$ (Doob's maximal inequality for non-negative supermartingales) gives a test that may be checked after <em>every</em> observation without inflating type I error: the basis of anytime-valid sequential testing. Contrast <a href="#/t/hypothesis-testing">fixed-$n$ tests</a>, whose error rates are destroyed by optional stopping on p-values.</p>
</div>
<div class="ex" data-name="Pólya's urn">
<p>The red fraction $R_n$ is a martingale in $[0,1]$, hence bounded, UI, and convergent a.s. and in $L^1$. Its limit is random: with $a$ red and $b$ blue initially, $R_\infty\sim\text{Beta}(a,b)$. Each urn "locks in" early luck. (Exchangeability makes the urn a Beta mixture of IID coin flips, by de Finetti.)</p>
</div>`,
        lab: 'martingales-polya',
      },
      {
        title: 'Concentration: Azuma–Hoeffding and bounded differences',
        html: R`
<p>Martingale differences behave like independent mean-zero increments for the purposes of exponential tail bounds.</p>
<div class="thm" data-kind="Theorem" data-name="Azuma–Hoeffding inequality">
<p>Let $X$ be a martingale with $|X_k-X_{k-1}|\le c_k$ a.s. Then for $t>0$,
$$\P(X_n-X_0\ge t)\le\exp\Big(-\frac{t^2}{2\sum_{k=1}^nc_k^2}\Big),$$
and the same bound holds for the lower tail, so the two-sided probability is at most twice this.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $D_k=X_k-X_{k-1}$, so $\E[D_k\mid\mathcal F_{k-1}]=0$ and $D_k\in[-c_k,c_k]$. Hoeffding's lemma (a mean-zero variable in $[a,b]$ has $\E e^{sD}\le e^{s^2(b-a)^2/8}$), applied conditionally, gives $\E[e^{sD_k}\mid\mathcal F_{k-1}]\le e^{s^2c_k^2/2}$. Peeling off one conditional expectation at a time,
$$\E e^{s(X_n-X_0)}=\E\big[e^{s(X_{n-1}-X_0)}\E[e^{sD_n}\mid\mathcal F_{n-1}]\big]\le e^{s^2c_n^2/2}\E e^{s(X_{n-1}-X_0)}\le\dots\le e^{s^2\sum c_k^2/2}.$$
Markov's inequality: $\P(X_n-X_0\ge t)\le e^{-st+s^2\sum c_k^2/2}$; minimize at $s=t/\sum c_k^2$.</p>
</details>
<div class="thm" data-kind="Corollary" data-name="McDiarmid's bounded-differences inequality (1989)">
<p>If $Y_1,\dots,Y_n$ are independent and changing the $i$-th argument of $f$ changes $f$ by at most $c_i$, then $\P(f(Y)-\E f(Y)\ge t)\le\exp(-2t^2/\sum c_i^2)$.</p>
</div>
<p>The proof applies the Azuma argument to the Doob martingale $X_k=\E[f(Y)\mid Y_1,\dots,Y_k]$, whose $k$-th increment lies in an interval of length $c_k$ (the sharper form of Hoeffding's lemma gives the factor 2). This is how one proves concentration of the chromatic number of a random graph, of empirical risk uniformly over a class (<a href="#/t/vc-theory">VC theory</a>), and of the bootstrap. See also <a href="#/t/inequalities">probability inequalities</a> for the independent-sum case.</p>
<div class="note" data-name="Martingales in continuous time">
<p>Everything above has a continuous-time version for right-continuous martingales. <a href="#/t/brownian-motion">Brownian motion</a> $B_t$, $B_t^2-t$ and $e^{\theta B_t-\theta^2t/2}$ are martingales, and the <a href="#/t/stochastic-calculus">Itô integral</a> is built precisely so that $\int H\d B$ is a martingale transform in continuous time.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A simple symmetric random walk starts at 0. Let $\tau$ be the first time it hits $+1$. Then $\tau< \infty$ a.s. and $S_\tau=1$, so $\E S_\tau=1\ne0=\E S_0$. Which optional-stopping hypothesis fails?`,
        a: R`All of them: $\tau$ is unbounded; the stopped walk is unbounded below; and $\E\tau=\infty$ (the walk is null recurrent), so condition (3) fails even though increments are bounded. Indeed, if $\E\tau$ were finite, Wald's identity would give $\E S_\tau=0\cdot\E\tau=0$, a contradiction — a slick proof that $\E\tau=\infty$.` },
      { q: R`With the doubling strategy and a bankroll that covers at most 10 consecutive losses (starting stake 1), what are the probability of ruin, the loss when ruined, and the expected net gain?`,
        a: R`Ruin probability $2^{-10}\approx0.001$; loss $2^{10}-1=1023$; gain $+1$ otherwise. Expected gain $(1-2^{-10})\cdot1-2^{-10}\cdot1023=0$: bounded stopping time, so optional stopping holds exactly.` },
      { q: R`Why does the non-negative martingale $L_n$ (a likelihood ratio under the null) converge to 0 a.s. even though $\E L_n=1$? What theorem distinguishes these two facts?`,
        a: R`A.s. convergence follows from Doob's theorem (non-negative supermartingale), and $\frac1n\log L_n\to-\KL< 0$ identifies the limit as 0. Expectations need not converge because $\{L_n\}$ is not uniformly integrable: the mean is carried by rare paths with enormous $L_n$. UI is equivalent to $L^1$ convergence, which fails here.` },
      { q: R`A fair die is rolled until the first 6. What is the expected sum of the rolls (including the 6)?`,
        a: R`The stopping time is Geometric(1/6) with $\E\tau=6$, and $\E\xi=3.5$, so Wald gives $\E\sum_{i\le\tau}\xi_i=21$. (Conditioning shows the same thing: the five non-final rolls are uniform on $\{1,\dots,5\}$, mean 3, giving $5\cdot3+6=21$.)` },
      { q: R`Use Azuma–Hoeffding to bound $\P(|S_n|\ge t)$ for a $\pm1$ random walk, and compare with the CLT.`,
        a: R`$c_k=1$, so $\P(|S_n|\ge t)\le2e^{-t^2/(2n)}$. With $t=x\sqrt n$ this is $2e^{-x^2/2}$, which matches the Gaussian tail $\P(|Z|\ge x)\approx\frac{2}{x\sqrt{2\pi}}e^{-x^2/2}$ up to the polynomial prefactor — but holds non-asymptotically and for any bounded-increment martingale, not just independent sums.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'brownian-motion',
    domain: 'stochastic',
    order: 110,
    title: 'Brownian Motion',
    subtitle: 'The universal scaling limit of random walks, and its strange paths',
    level: 3,
    sources: { aos: '—', cb: '—', other: 'Mörters & Peres, Brownian Motion (2010); Durrett, Probability, Ch. 8; Karatzas & Shreve (1991), Ch. 2' },
    prereqs: ['random-walks', 'clt', 'martingales'],
    related: ['stochastic-calculus', 'empirical-cdf', 'poisson-process', 'diffusion-models'],
    tags: ['Wiener process', 'Donsker', 'quadratic variation', 'reflection principle', 'scaling'],
    summary: R`Brownian motion $B_t$ is the continuous-time process with independent, stationary, Gaussian increments and continuous paths. It is what every finite-variance random walk looks like from far away (Donsker's invariance principle), which is why it appears in physics, finance, statistics (the Kolmogorov–Smirnov limit), and machine learning (Langevin samplers and diffusion models). Its paths are continuous but nowhere differentiable, have infinite length on every interval, and yet accumulate squared increments at the deterministic rate $\d t$ — quadratic variation $[B]_t=t$, the fact that makes Itô calculus differ from ordinary calculus. Symmetry plus the strong Markov property give the reflection principle: the running maximum $M_t$ has the same law as $|B_t|$.`,
    survey: {
      task: 'Model continuous random fluctuation; describe limits of random walks and empirical processes',
      assumptions: R`Independent stationary Gaussian increments, $\Var(B_t-B_s)=t-s$; continuous paths`,
      output: R`Gaussian process with $\Cov(B_s,B_t)=\min(s,t)$; laws of maxima, hitting times, exit probabilities`,
      strengths: 'Universal (Donsker); exact formulas via reflection and martingales; scale invariant',
      weaknesses: 'Gaussian, light-tailed, no jumps; paths too rough for classical calculus',
      complexity: R`Simulation on a grid of $n$ points: $O(n)$; maxima need fine grids (discretization bias $\sim n^{-1/2}$)`,
    },
    sections: [
      {
        title: 'Definition and existence',
        html: R`
<div class="def" data-name="Standard Brownian motion (Wiener process)">
<p>A real process $(B_t)_{t\ge0}$ is a standard Brownian motion if (i) $B_0=0$; (ii) for $0\le t_0< t_1< \dots< t_k$ the increments $B_{t_1}-B_{t_0},\dots,B_{t_k}-B_{t_{k-1}}$ are independent; (iii) $B_t-B_s\sim\N(0,t-s)$ for $s< t$; (iv) $t\mapsto B_t$ is continuous almost surely.</p>
</div>
<p>Equivalently, $B$ is a centred <a href="#/t/gaussian-processes">Gaussian process</a> with continuous paths and covariance $\Cov(B_s,B_t)=\min(s,t)$: for $s\le t$, $\Cov(B_s,B_t)=\Cov(B_s,B_s+(B_t-B_s))=s$. This characterization is the quickest way to verify that a transformed process is again Brownian motion.</p>
<p>Existence is not obvious: (i)–(iii) specify finite-dimensional laws, which by Kolmogorov's extension theorem define a process, but continuity (iv) is a property of uncountably many coordinates. Two classical routes: <strong>Kolmogorov's continuity criterion</strong> — if $\E|X_t-X_s|^p\le C|t-s|^{1+\beta}$ there is a version with $\alpha$-Hölder paths for every $\alpha< \beta/p$; Gaussian moments $\E|B_t-B_s|^{2m}=c_m|t-s|^m$ give $\alpha< (m-1)/(2m)$ for every $m$, i.e. every $\alpha< 1/2$ — and <strong>Lévy's construction</strong>, which builds $B$ on dyadic grids by midpoint refinement: given $B$ at $k2^{-n}$, the midpoint is the average of its neighbours plus independent $\N(0,2^{-n-2})$ noise (the Brownian bridge variance at a midpoint), and the resulting piecewise-linear interpolants converge uniformly a.s. The same idea in series form is $B_t=\sum_k Z_k\int_0^t\phi_k$ for any orthonormal basis $\phi_k$ of $L^2[0,1]$ (with the Haar basis this is Lévy–Ciesielski; with sines it is the Karhunen–Loève expansion).</p>`,
      },
      {
        title: "Brownian motion as a scaling limit: Donsker's theorem",
        html: R`
<p>Let $\xi_1,\xi_2,\dots$ be IID with mean 0 and variance $\sigma^2$, $S_k=\sum_{i\le k}\xi_i$. Speed up time by $n$ and shrink space by $\sqrt n$: define the continuous, piecewise-linear process
$$W_n(t)=\frac{1}{\sigma\sqrt n}\Big(S_{\lfloor nt\rfloor}+(nt-\lfloor nt\rfloor)\,\xi_{\lfloor nt\rfloor+1}\Big),\qquad t\in[0,1].$$</p>
<div class="thm" data-kind="Theorem" data-name="Donsker's invariance principle (1951)">
<p>$W_n\Rightarrow B$ in distribution as random elements of $C[0,1]$ with the sup norm. Consequently $\phi(W_n)\dto\phi(B)$ for every functional $\phi:C[0,1]\to\R$ that is continuous (or continuous at almost every Brownian path).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Two ingredients, as for any weak convergence in function space. <em>Finite-dimensional convergence</em>: for fixed $t_1< \dots< t_k$, the increments $W_n(t_j)-W_n(t_{j-1})$ are (up to a negligible interpolation term) normalized sums over disjoint blocks of $\xi$'s, hence independent, and by the <a href="#/t/clt">CLT</a> each converges to $\N(0,t_j-t_{j-1})$. <em>Tightness</em>: one shows the modulus of continuity is controlled, $\lim_{\delta\to0}\limsup_n\P(\sup_{|s-t|\le\delta}|W_n(s)-W_n(t)|>\epsilon)=0$, via a maximal inequality for partial sums (Kolmogorov/Ottaviani or Doob). Prokhorov's theorem then upgrades fidi convergence to convergence in $C[0,1]$. An alternative proof embeds the walk in a Brownian motion (Skorokhod embedding) so that the convergence holds almost surely in sup norm.</p>
</details>
<p>"Invariance" refers to the limit not depending on the law of $\xi$, only on $\sigma^2$. The payoff is that functionals like $\max_{k\le n}S_k/(\sigma\sqrt n)$, the fraction of time the walk is positive, or the last zero before $n$ all converge to the corresponding Brownian quantities, which can be computed exactly. The same theorem applied to the uniform empirical process gives the <a href="#/t/empirical-cdf">Kolmogorov–Smirnov</a> limit: $\sqrt n(F_n-F)\Rightarrow\mathbb B\circ F$, where $\mathbb B_t=B_t-tB_1$ is the <strong>Brownian bridge</strong>.</p>`,
        lab: 'brownian-motion-donsker',
      },
      {
        title: 'Symmetries and the Markov property',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Invariances">
<p>If $B$ is standard Brownian motion, so are:</p>
<ol>
<li>$-B_t$ (reflection);</li>
<li>$B_{s+t}-B_s$ for fixed $s\ge0$, independent of $\{B_u:u\le s\}$ (simple Markov property);</li>
<li>$c^{-1/2}B_{ct}$ for any $c>0$ (Brownian scaling);</li>
<li>$tB_{1/t}$ for $t>0$, extended by $0$ at $t=0$ (time inversion).</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>Each is a centred Gaussian process; check covariances. (3): $\Cov(c^{-1/2}B_{cs},c^{-1/2}B_{ct})=c^{-1}\min(cs,ct)=\min(s,t)$. (4): $\Cov(sB_{1/s},tB_{1/t})=st\min(1/s,1/t)=\min(s,t)$. Continuity is clear for $t>0$; at $t=0$ for (4) it amounts to $B_u/u\to0$ as $u\to\infty$, which holds by the SLLN along integers plus a bound on fluctuations between integers. (2) uses independence of increments.</p>
</details>
<p>Scaling is the signature fact: Brownian motion has no characteristic scale. Zoom in on a path by a factor $c$ in time and $\sqrt c$ in space and you see another Brownian path. That immediately says paths are rough: over time $h$, fluctuations are of order $\sqrt h\gg h$.</p>
<p>The <strong>strong Markov property</strong> extends (2) to stopping times: if $\tau$ is an a.s. finite stopping time, then $(B_{\tau+t}-B_\tau)_{t\ge0}$ is a Brownian motion independent of $\mathcal F_\tau$. Applied to hitting times $\tau_a=\inf\{t:B_t=a\}$, it drives the reflection principle below. Large-scale behaviour is pinned down by the <strong>law of the iterated logarithm</strong>: $\limsup_{t\to\infty}B_t/\sqrt{2t\log\log t}=1$ a.s., and by time inversion the same rate governs $t\to0$ with $\log\log(1/t)$.</p>`,
      },
      {
        title: 'Path properties: quadratic variation and roughness',
        html: R`
<p>For a partition $\Pi=\{0=t_0< t_1< \dots< t_n=t\}$ with mesh $|\Pi|=\max_i(t_{i+1}-t_i)$, consider the sum of squared increments.</p>
<div class="thm" data-kind="Theorem" data-name="Quadratic variation">
<p>$Q(\Pi)=\sum_i(B_{t_{i+1}}-B_{t_i})^2\to t$ in $L^2$ as $|\Pi|\to0$, and almost surely along any sequence of partitions with $\sum|\Pi_n|< \infty$ (e.g. dyadic partitions). We write $[B]_t=t$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\Delta_i=t_{i+1}-t_i$ and $D_i=(B_{t_{i+1}}-B_{t_i})^2-\Delta_i$. The $D_i$ are independent with mean 0 and $\Var D_i=\Var(\Delta_iZ^2)=2\Delta_i^2$. So $\E(Q(\Pi)-t)^2=\sum_i2\Delta_i^2\le2t|\Pi|\to0$. For the a.s. statement, Chebyshev gives $\P(|Q(\Pi_n)-t|>\epsilon)\le2t|\Pi_n|/\epsilon^2$, summable, so Borel–Cantelli applies.</p>
</details>
<p>For any continuously differentiable $f$, $\sum_i(f(t_{i+1})-f(t_i))^2\le\max_i|f(t_{i+1})-f(t_i)|\cdot\int_0^t|f'|\to0$: smooth functions have zero quadratic variation. Brownian paths are therefore not smooth, and more:</p>
<div class="thm" data-kind="Corollary" data-name="Infinite variation">
<p>Almost surely, the total variation $\sup_\Pi\sum_i|B_{t_{i+1}}-B_{t_i}|$ is infinite on every interval.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$Q(\Pi)\le\max_i|B_{t_{i+1}}-B_{t_i}|\cdot\sum_i|B_{t_{i+1}}-B_{t_i}|$. The max tends to 0 by uniform continuity of paths on $[0,t]$, while $Q(\Pi)\to t>0$ along dyadic partitions. If the total variation were finite the right side would tend to 0.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="Paley–Wiener–Zygmund (1933)">
<p>Almost surely, $t\mapsto B_t$ is differentiable at no $t\ge0$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>If $B$ were differentiable at some $s\in[0,1]$ with $|B'(s)|< M$, then for all large $n$ and $k=\lfloor ns\rfloor$, the three consecutive increments $|B_{(j+1)/n}-B_{j/n}|$, $j=k+1,k+2,k+3$, would all be at most $CM/n$ for a constant $C$ (each endpoint is within $4/n$ of $s$). The probability that <em>some</em> $k\le n$ has three such small consecutive increments is at most $n\cdot\P(|Z|\le CM/\sqrt n)^3\le n\,(CM/\sqrt n)^3\to0$, using independence and $\P(|Z|\le x)\le x$. Taking $M\to\infty$ over integers finishes it.</p>
</details>
<p>Brownian paths are $\alpha$-Hölder for every $\alpha< 1/2$ but not for $\alpha=1/2$: Lévy's modulus of continuity is exactly $\sqrt{2h\log(1/h)}$. These facts are why $\int f(B_t)\d B_t$ cannot be defined path by path as a Riemann–Stieltjes integral, and why the second-order term in Itô's formula survives: $(\d B)^2=\d t$.</p>`,
        lab: 'brownian-motion-qv',
      },
      {
        title: 'The reflection principle and the running maximum',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Reflection principle">
<p>Let $M_t=\max_{s\le t}B_s$ and $a>0$. Then
$$\P(M_t\ge a)=2\,\P(B_t\ge a)=\P(|B_t|\ge a),$$
so $M_t\overset{d}{=}|B_t|$. More precisely, for $b\le a$, $\P(M_t\ge a,\,B_t\le b)=\P(B_t\ge2a-b)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\tau_a=\inf\{s:B_s=a\}$; $\{M_t\ge a\}=\{\tau_a\le t\}$ by continuity. Define the reflected path $\tilde B_s=B_s$ for $s\le\tau_a$ and $\tilde B_s=2a-B_s$ for $s>\tau_a$. By the strong Markov property, $B_{\tau_a+u}-a$ is a Brownian motion independent of $\mathcal F_{\tau_a}$, and so is its negative; hence $\tilde B$ is again a Brownian motion, with the same hitting time $\tau_a$. For $b\le a$: $\P(\tau_a\le t,B_t\le b)=\P(\tau_a\le t,\tilde B_t\ge2a-b)=\P(B_t\ge2a-b)$, the last step because $\tilde B$ is BM and $\{\tilde B_t\ge2a-b\}\subseteq\{\tau_a\le t\}$. With $b=a$: $\P(M_t\ge a,B_t\le a)=\P(B_t\ge a)$; adding $\P(M_t\ge a,B_t>a)=\P(B_t>a)$ gives $2\P(B_t\ge a)$.</p>
</details>
<p>Consequences, each a short calculation:</p>
<ul>
<li><strong>Hitting times.</strong> $\P(\tau_a\le t)=2(1-\Phi(a/\sqrt t))$, so $\tau_a$ has density $\frac{a}{\sqrt{2\pi}\,t^{3/2}}e^{-a^2/(2t)}$ (the Lévy distribution, a stable law with index 1/2). Every level is hit ($\P(\tau_a< \infty)=1$) but $\E\tau_a=\infty$ because of the $t^{-3/2}$ tail — the continuous analogue of the null recurrence of the symmetric <a href="#/t/random-walks">random walk</a>.</li>
<li><strong>Exit from an interval</strong> (by optional stopping on the bounded <a href="#/t/martingales">martingales</a> $B_t$ and $B_t^2-t$): started at 0, $B$ leaves $(-b,a)$ at $a$ with probability $b/(a+b)$, and the expected exit time is $ab$.</li>
<li><strong>Brownian motion with drift</strong> $X_t=\mu t+\sigma B_t$, $\mu< 0$: the all-time maximum is exponential, $\P(\sup_tX_t\ge a)=e^{-2|\mu|a/\sigma^2}$, from the exponential martingale $e^{\theta X_t-(\theta\mu+\theta^2\sigma^2/2)t}$ with $\theta=2|\mu|/\sigma^2$. This is the heavy-traffic waiting-time law in <a href="#/t/queueing">queueing</a> and the ruin probability in diffusion risk models.</li>
</ul>
<div class="warn" data-name="Discretization bias for maxima">
<p>Monitoring a simulated path only at grid points $k/n$ underestimates the maximum: the continuous path can exceed $a$ between grid points. The error in $\E M_1$ is of order $n^{-1/2}$ (Asmussen, Glynn & Pitman, 1995), much larger than the $O(n^{-1})$ one might expect. The Brownian-bridge correction — conditional on endpoints $x,y< a$ over a step of length $h$, the bridge crosses $a$ with probability $\exp(-2(a-x)(a-y)/h)$ — removes most of it. The Donsker lab shows the same bias for random walks.</p>
</div>`,
      },
      {
        title: 'Martingales, higher dimensions, and where Brownian motion leads',
        html: R`
<p>Three martingales generate most Brownian calculations: $B_t$, $B_t^2-t$, and $\exp(\theta B_t-\theta^2t/2)$. Conversely, <strong>Lévy's characterization</strong>: a continuous martingale $M$ with $M_0=0$ and $M_t^2-t$ a martingale (i.e. $[M]_t=t$) is a standard Brownian motion. Gaussianity is a consequence, not an assumption.</p>
<p>In $\R^d$, a Brownian motion has independent coordinate Brownian motions. Pólya's dichotomy carries over: in $d\le2$ it is (neighbourhood-)recurrent, returning arbitrarily close to its start infinitely often; in $d\ge3$ it is transient, $|B_t|\to\infty$. The function $u(x)=\P_x(\text{hit }A\text{ before }C)$ is harmonic ($\Delta u=0$), linking Brownian motion to the Dirichlet problem and, via the generator $\frac12\Delta$, to the heat equation: $u(t,x)=\E f(x+B_t)$ solves $\partial_tu=\frac12\Delta u$.</p>
<p>Where it leads in this atlas:</p>
<ul>
<li><a href="#/t/stochastic-calculus">Itô calculus</a> integrates against $\d B_t$ and builds diffusions $\d X=\mu(X)\d t+\sigma(X)\d B$ — including geometric Brownian motion and the Ornstein–Uhlenbeck process.</li>
<li><a href="#/t/langevin-mala">Langevin MCMC</a> discretizes $\d X=\nabla\log\pi(X)\d t+\sqrt2\,\d B$, whose stationary law is $\pi$.</li>
<li><a href="#/t/diffusion-models">Diffusion models</a> corrupt data by a Brownian-driven SDE and learn to run it backwards.</li>
<li>The <a href="#/t/poisson-process">Poisson process</a> and any finite-variance <a href="#/t/random-walks">random walk</a> converge to it after centring and scaling.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Compute $\Cov(B_2,B_5)$ and $\Var(B_5-2B_2)$.`,
        a: R`$\Cov(B_2,B_5)=\min(2,5)=2$. $\Var(B_5-2B_2)=\Var B_5+4\Var B_2-4\Cov(B_5,B_2)=5+8-8=5$. (Alternatively $B_5-2B_2=(B_5-B_2)-B_2$, a sum of independent $\N(0,3)$ and $\N(0,2)$.)` },
      { q: R`What is $\P(\max_{s\le4}B_s\ge2)$?`,
        a: R`$2\P(B_4\ge2)=2\P(Z\ge1)=2(1-\Phi(1))\approx0.317$.` },
      { q: R`Why does the sum of squared increments of a Brownian path over a fine partition of $[0,t]$ converge to $t$, while for a differentiable function it converges to 0?`,
        a: R`For BM, each squared increment has mean $\Delta_i$ and variance $2\Delta_i^2$, so the sum has mean $t$ and variance $\le2t|\Pi|\to0$ — a law of large numbers over increments. For $C^1$ functions the increments are $O(\Delta_i)$, so the sum of squares is $O(\max\Delta_i)\to0$. Increments of size $\sqrt\Delta$ versus $\Delta$ is the whole difference.` },
      { q: R`Brownian motion started at 0 runs until it exits $(-1,3)$. Find the probability it exits at 3 and the expected exit time.`,
        a: R`Optional stopping on $B_t$ (bounded before exit): $3p-(1-p)=0$, so $p=1/4$. On $B_t^2-t$: $\E\tau=\E B_\tau^2=9\cdot\tfrac14+1\cdot\tfrac34=3=ab$ with $a=3,b=1$.` },
      { q: R`Is $X_t=\sqrt2\,B_{t/2}$ a standard Brownian motion? What about $Y_t=B_{2t}-B_t$?`,
        a: R`$X$: yes, by scaling with $c=1/2$: $\Cov(X_s,X_t)=2\min(s/2,t/2)=\min(s,t)$. $Y$: no. $\Var Y_t=t$ is right, but for $s< t$, $\Cov(Y_s,Y_t)=\Cov(B_{2s}-B_s,B_{2t}-B_t)=\min(2s,2t)-\min(2s,t)-\min(s,2t)+\min(s,t)$; with $s=1,t=1.5$ this is $2-1.5-1+1=0.5\ne1$. (Its increments are not independent.)` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'stochastic-calculus',
    domain: 'stochastic',
    order: 120,
    title: 'Itô Calculus & Stochastic Differential Equations',
    subtitle: 'Integrating against Brownian motion, where (dB)² = dt',
    level: 3,
    sources: { aos: '—', cb: '—', other: 'Øksendal, Stochastic Differential Equations (6th ed., 2003); Kloeden & Platen, Numerical Solution of SDEs (1992)' },
    prereqs: ['brownian-motion', 'martingales'],
    related: ['langevin-mala', 'diffusion-models', 'time-series', 'gaussian-processes'],
    tags: ['Itô integral', 'Itô isometry', "Itô's lemma", 'GBM', 'Ornstein–Uhlenbeck', 'Euler–Maruyama'],
    summary: R`To model a quantity driven by continuous noise — a stock price, a particle, the state of a diffusion-model sampler — we want equations like $\d X_t=\mu(X_t)\d t+\sigma(X_t)\d B_t$. Brownian paths have infinite variation, so $\int\sigma\d B$ cannot be an ordinary Stieltjes integral, and different Riemann sums give different limits. Itô's choice — evaluate the integrand at the left endpoint — makes the integral a martingale and gives the isometry $\E(\int H\d B)^2=\E\int H^2\d t$. The price is a new chain rule: because $(\d B)^2=\d t$, Itô's lemma has an extra second-derivative term, $\d f(X)=f'(X)\d X+\tfrac12f''(X)\sigma^2\d t$. With it one solves geometric Brownian motion and the Ornstein–Uhlenbeck process exactly, and analyses the Euler–Maruyama scheme used to simulate everything else.`,
    survey: {
      task: 'Define, solve, and simulate differential equations driven by Brownian noise',
      assumptions: R`Adapted integrands with $\E\int H^2< \infty$; Lipschitz and linear-growth coefficients for existence/uniqueness`,
      output: R`Strong solutions $X_t$; moments and laws (GBM lognormal, OU Gaussian); simulated paths`,
      strengths: "Itô's lemma turns SDE questions into calculus; Fokker–Planck links paths to PDEs",
      weaknesses: 'Itô vs Stratonovich ambiguity; Euler–Maruyama converges slowly (strong order 1/2) and can blow up for superlinear drift',
      complexity: R`Euler–Maruyama: $O(T/\Delta)$ per path; strong error $O(\Delta^{1/2})$, weak error $O(\Delta)$`,
    },
    sections: [
      {
        title: 'Why ordinary calculus fails: three answers for $\int B\,\d B$',
        html: R`
<p>Try to compute $\int_0^TB_t\d B_t$ as a limit of Riemann sums over a partition $0=t_0< \dots< t_n=T$, writing $\Delta B_i=B_{t_{i+1}}-B_{t_i}$. Using the identity $B_{t_i}\Delta B_i=\frac12(B_{t_{i+1}}^2-B_{t_i}^2)-\frac12(\Delta B_i)^2$ and telescoping:
$$\sum_iB_{t_i}\Delta B_i=\tfrac12B_T^2-\tfrac12\sum_i(\Delta B_i)^2\;\longrightarrow\;\tfrac12(B_T^2-T),$$
because the sum of squared increments converges to the <a href="#/t/brownian-motion">quadratic variation</a> $T$. Evaluating at the right endpoint instead gives $\sum B_{t_{i+1}}\Delta B_i\to\frac12(B_T^2+T)$, and the average of the two (the trapezoid rule) gives exactly $\frac12B_T^2$ for every partition.</p>
<p>For a function of bounded variation all three would agree; here they differ by $\pm\frac T2$, a <em>deterministic</em> amount coming from $\sum(\Delta B_i)^2\to T$. We must choose. The two standard choices:</p>
<div class="cols">
<div><h4>Itô (left point)</h4><p>$\int_0^TB\d B=\frac12(B_T^2-T)$. Integrand uses only the past: the integral is a martingale (mean zero). Natural for finance, filtering, statistics, and anything causal.</p></div>
<div><h4>Stratonovich (midpoint)</h4><p>$\int_0^TB\circ\d B=\frac12B_T^2$. Ordinary chain rule holds. Arises as the limit of ODEs driven by smooth approximations of noise (Wong–Zakai); natural in physics and on manifolds.</p></div>
</div>
<p>They are interconvertible: the Stratonovich SDE $\d X=\mu\d t+\sigma(X)\circ\d B$ is the Itô SDE $\d X=\big(\mu+\tfrac12\sigma\sigma'\big)\d t+\sigma(X)\d B$. Below, everything is Itô.</p>`,
        lab: 'stochastic-calculus-riemann',
      },
      {
        title: 'Constructing the Itô integral',
        html: R`
<p>Fix a filtration $\mathcal F_t$ to which $B$ is adapted and with $B_{t+s}-B_t$ independent of $\mathcal F_t$. A <strong>simple</strong> process is $H_t=\sum_{i}h_i\ind(t_i< t\le t_{i+1})$ with each $h_i$ being $\mathcal F_{t_i}$-measurable and square-integrable — the stake is set before the increment is revealed, exactly as in a <a href="#/t/martingales">martingale transform</a>. Define
$$\int_0^TH_t\d B_t=\sum_ih_i\,(B_{t_{i+1}}-B_{t_i}).$$</p>
<div class="thm" data-kind="Theorem" data-name="Itô isometry">
<p>For simple $H$, $\E\int_0^TH\d B=0$ and
$$\E\Big[\Big(\int_0^TH_t\d B_t\Big)^2\Big]=\E\int_0^TH_t^2\d t.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $\Delta_i=B_{t_{i+1}}-B_{t_i}$, independent of $\mathcal F_{t_i}$ with mean 0 and variance $t_{i+1}-t_i$. The mean: $\E[h_i\Delta_i]=\E[h_i\E(\Delta_i\mid\mathcal F_{t_i})]=0$. Expanding the square, diagonal terms give $\E[h_i^2\Delta_i^2]=\E[h_i^2\E(\Delta_i^2\mid\mathcal F_{t_i})]=\E[h_i^2](t_{i+1}-t_i)$. For $i< j$, $h_ih_j\Delta_i$ is $\mathcal F_{t_j}$-measurable, so $\E[h_ih_j\Delta_i\Delta_j]=\E[h_ih_j\Delta_i\E(\Delta_j\mid\mathcal F_{t_j})]=0$. Summing, $\E(\int H\d B)^2=\sum_i\E[h_i^2](t_{i+1}-t_i)=\E\int_0^TH^2\d t$.</p>
</details>
<p>The isometry says the map $H\mapsto\int H\d B$ preserves the $L^2$ norm, from $L^2(\Omega\times[0,T])$ to $L^2(\Omega)$. Simple processes are dense in the space $\mathcal H^2$ of adapted (progressively measurable) $H$ with $\E\int_0^TH^2\d t< \infty$. So for general $H\in\mathcal H^2$, take simple $H^{(n)}\to H$; the integrals form a Cauchy sequence in $L^2(\Omega)$ by the isometry, and the limit defines $\int H\d B$, independent of the approximating sequence. Properties inherited from the simple case:</p>
<ul>
<li>$t\mapsto\int_0^tH\d B$ has a continuous version and is a square-integrable <strong>martingale</strong>, with mean zero.</li>
<li>Its quadratic variation is $\int_0^tH_s^2\d s$ — the "volatility clock".</li>
<li>For deterministic $H=h(t)$, $\int_0^Th\d B\sim\N\big(0,\int_0^Th^2\big)$ (a Wiener integral).</li>
</ul>
<p>By localization (stopping when $\int_0^tH^2$ exceeds $n$), the integral extends to adapted $H$ with $\int_0^TH^2< \infty$ a.s.; the result is then only a <em>local</em> martingale, and the isometry may fail. This gap is responsible for "strict local martingales" and bubbles in finance.</p>`,
      },
      {
        title: "Itô's lemma",
        html: R`
<p>An <strong>Itô process</strong> is $X_t=X_0+\int_0^t\mu_s\d s+\int_0^t\sigma_s\d B_s$, written $\d X_t=\mu_t\d t+\sigma_t\d B_t$. Its quadratic variation is $[X]_t=\int_0^t\sigma_s^2\d s$: the $\d t$ part has zero quadratic variation and the cross term vanishes. This is summarized by the multiplication table
$$(\d t)^2=0,\qquad\d t\,\d B=0,\qquad(\d B)^2=\d t,\qquad\text{so}\quad(\d X)^2=\sigma^2\d t.$$</p>
<div class="thm" data-kind="Theorem" data-name="Itô's lemma (Itô, 1951)">
<p>If $f(t,x)$ is $C^1$ in $t$ and $C^2$ in $x$, and $\d X=\mu\d t+\sigma\d B$, then $Y_t=f(t,X_t)$ is an Itô process with
$$\d f(t,X_t)=\Big(\partial_tf+\mu\,\partial_xf+\tfrac12\sigma^2\,\partial_{xx}f\Big)\d t+\sigma\,\partial_xf\,\d B_t.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Take a partition of $[0,t]$ and Taylor expand each increment to second order:
$$f(t_{i+1},X_{t_{i+1}})-f(t_i,X_{t_i})=\partial_tf\,\Delta t_i+\partial_xf\,\Delta X_i+\tfrac12\partial_{xx}f\,(\Delta X_i)^2+R_i,$$
with derivatives evaluated at $(t_i,X_{t_i})$ and $R_i$ containing $(\Delta t_i)^2$, $\Delta t_i\Delta X_i$ and $o((\Delta X_i)^2)$ terms. Summing: the first term converges to $\int\partial_tf\d s$; the second to $\int\partial_xf\d X$ (left-point sums, so an Itô integral); the third to $\frac12\int\partial_{xx}f\,\sigma^2\d s$, by the same $L^2$ argument as for quadratic variation, now weighted by the adapted factor $\partial_{xx}f$ — for $X=B$, the variance of $\sum g_i[(\Delta B_i)^2-\Delta t_i]$ is $2\sum\E g_i^2\Delta t_i^2\to0$. The remainders vanish because $\sum|\Delta X_i|^3\le\max|\Delta X_i|\sum(\Delta X_i)^2\to0\cdot[X]_t$. (Rigorous versions localize so that $f$ and its derivatives are bounded.)</p>
</details>
<p>The only difference from the ordinary chain rule is the $\frac12\sigma^2f''$ term — the "Itô correction". Jensen-type intuition: if $f$ is convex, noise pushes $f(X)$ up on average, at rate $\frac12\sigma^2f''$.</p>
<div class="ex" data-name="Quick checks">
<p>(a) $f(x)=x^2$, $X=B$: $\d(B_t^2)=2B_t\d B_t+\d t$, so $\int_0^TB\d B=\frac12(B_T^2-T)$, matching the Riemann-sum computation. (b) $Z_t=\exp(\theta B_t-\theta^2t/2)$: $\partial_t=-\frac{\theta^2}{2}Z$, $\partial_x=\theta Z$, $\partial_{xx}=\theta^2Z$, so the $\d t$ terms cancel and $\d Z=\theta Z\d B$ — the exponential martingale. (c) Product rule: $\d(XY)=X\d Y+Y\d X+\d X\,\d Y$, the last term being $\sigma_X\sigma_Y\d t$ for processes driven by the same $B$.</p>
</div>`,
      },
      {
        title: 'Solving SDEs: geometric Brownian motion and Ornstein–Uhlenbeck',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Existence and uniqueness">
<p>Consider $\d X_t=\mu(t,X_t)\d t+\sigma(t,X_t)\d B_t$, $X_0$ independent of $B$ with $\E X_0^2< \infty$. If $|\mu(t,x)-\mu(t,y)|+|\sigma(t,x)-\sigma(t,y)|\le K|x-y|$ (Lipschitz) and $|\mu(t,x)|+|\sigma(t,x)|\le K(1+|x|)$ (linear growth), there is a unique continuous adapted solution, with $\E\sup_{t\le T}X_t^2< \infty$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Picard iteration $X^{(k+1)}_t=X_0+\int_0^t\mu(s,X^{(k)}_s)\d s+\int_0^t\sigma(s,X^{(k)}_s)\d B_s$. The Itô isometry (with Doob's $L^2$ maximal inequality for the sup) and Cauchy–Schwarz give $\E\sup_{s\le t}|X^{(k+1)}_s-X^{(k)}_s|^2\le C\int_0^t\E\sup_{u\le s}|X^{(k)}_u-X^{(k-1)}_u|^2\d s$, so the differences are bounded by $C^kt^k/k!$: the iteration converges. Uniqueness: the same estimate and Gronwall's lemma applied to the difference of two solutions.</p>
</details>
<p><strong>Geometric Brownian motion</strong>, $\d S_t=\mu S_t\d t+\sigma S_t\d B_t$, models a quantity whose <em>relative</em> changes are noisy (the Black–Scholes stock price). Apply Itô to $\log S$: with $f(x)=\log x$, $f'=1/x$, $f''=-1/x^2$,
$$\d\log S_t=\frac{\d S_t}{S_t}-\frac12\frac{\sigma^2S_t^2}{S_t^2}\d t=\Big(\mu-\frac{\sigma^2}{2}\Big)\d t+\sigma\d B_t\;\Longrightarrow\;S_t=S_0\exp\Big(\big(\mu-\tfrac{\sigma^2}{2}\big)t+\sigma B_t\Big).$$
So $S_t$ is lognormal with $\E S_t=S_0e^{\mu t}$ and $\Var S_t=S_0^2e^{2\mu t}(e^{\sigma^2t}-1)$, but the <em>median</em> is $S_0e^{(\mu-\sigma^2/2)t}$. Since $B_t/t\to0$, if $0< \mu< \sigma^2/2$ then $S_t\to0$ almost surely while $\E S_t\to\infty$: the mean is carried by ever rarer paths. Volatility drag $\sigma^2/2$ is the reason a portfolio's long-run growth rate is below its mean return.</p>
<p><strong>Ornstein–Uhlenbeck</strong>, $\d X_t=-\theta(X_t-m)\d t+\sigma\d B_t$ with $\theta>0$, is mean-reverting noise (interest rates, velocities of particles, the forward process of variance-preserving diffusion models). Multiply by the integrating factor $e^{\theta t}$; by the product rule $\d(e^{\theta t}X_t)=e^{\theta t}(\theta X_t\d t+\d X_t)=e^{\theta t}(\theta m\d t+\sigma\d B_t)$, so
$$X_t=m+(X_0-m)e^{-\theta t}+\sigma\int_0^te^{-\theta(t-s)}\d B_s.$$
The stochastic integral has a deterministic integrand, so given $X_0$, $X_t$ is Gaussian with mean $m+(X_0-m)e^{-\theta t}$ and (by the isometry) variance $\sigma^2\int_0^te^{-2\theta(t-s)}\d s=\frac{\sigma^2}{2\theta}(1-e^{-2\theta t})$. The stationary law is $\N\big(m,\sigma^2/(2\theta)\big)$, and the stationary autocovariance is $\frac{\sigma^2}{2\theta}e^{-\theta|h|}$. Sampled at spacing $\Delta$, the OU process is <em>exactly</em> an AR(1) series,
$$X_{k+1}=m+e^{-\theta\Delta}(X_k-m)+\sqrt{\tfrac{\sigma^2}{2\theta}(1-e^{-2\theta\Delta})}\;Z_{k+1},$$
connecting continuous-time diffusions to <a href="#/t/time-series">ARMA models</a>.</p>`,
      },
      {
        title: 'Simulation: Euler–Maruyama and Milstein',
        html: R`
<div class="algo" data-name="Euler–Maruyama">
<ol>
<li>Grid $t_k=k\Delta$, $\Delta=T/N$; draw $\Delta B_k\sim\N(0,\Delta)$ independently.</li>
<li>$\hat X_{k+1}=\hat X_k+\mu(t_k,\hat X_k)\Delta+\sigma(t_k,\hat X_k)\Delta B_k$.</li>
</ol>
</div>
<p>This is the left-point (Itô) Riemann sum turned into a recursion. Two notions of accuracy matter:</p>
<ul>
<li><strong>Strong</strong> (pathwise) error $\E|X_T-\hat X_T|$, relevant when the path itself matters (hedging, filtering). Under Lipschitz/linear-growth conditions EM has strong order $1/2$: error $\le C\Delta^{1/2}$.</li>
<li><strong>Weak</strong> error $|\E g(X_T)-\E g(\hat X_T)|$ for smooth $g$, relevant for expectations and distributions (pricing, Monte Carlo). EM has weak order $1$ (Talay & Tubaro, 1990).</li>
</ul>
<p>The strong order is limited by the neglected term $\frac12\sigma\sigma'((\Delta B)^2-\Delta)$ in the Itô–Taylor expansion, which has size $O(\Delta)$ per step but random sign, adding up to $O(\Delta^{1/2})$. Including it gives the <strong>Milstein scheme</strong>,
$$\hat X_{k+1}=\hat X_k+\mu\Delta+\sigma\Delta B_k+\tfrac12\sigma\sigma'\big((\Delta B_k)^2-\Delta\big),$$
with strong order 1. For additive noise ($\sigma$ constant, as in OU) the correction vanishes and EM already has strong order 1. To measure strong error in practice, simulate the fine and coarse schemes on the <em>same</em> Brownian path (sum fine increments to get coarse ones), as the lab does.</p>
<div class="warn" data-name="When Euler–Maruyama fails">
<p>(1) <strong>Superlinear coefficients.</strong> For $\d X=-X^3\d t+\d B$ the SDE is perfectly stable, but EM moments diverge: $\E|\hat X_T|^p\to\infty$ as $\Delta\to0$ (Hutzenthaler, Jentzen & Kloeden, 2011), because rare large Gaussian increments are amplified by the explicit cubic drift. Tamed or implicit schemes fix it. The same instability afflicts unadjusted <a href="#/t/langevin-mala">Langevin</a> samplers on light-tailed targets. (2) <strong>Positivity.</strong> EM for square-root diffusions such as CIR, $\d X=\kappa(\theta-X)\d t+\sigma\sqrt X\d B$, can go negative. (3) <strong>Stiffness.</strong> For OU with large $\theta$, EM is stable only if $|1-\theta\Delta|< 1$, i.e. $\Delta< 2/\theta$; beyond that the recursion explodes, while the exact AR(1) update is stable for every $\Delta$. Try it in the lab.</p>
</div>`,
        lab: 'stochastic-calculus-em',
      },
      {
        title: 'From paths to densities, and the reverse-time SDE',
        html: R`
<p>Each SDE $\d X=\mu(X)\d t+\sigma(X)\d B$ has a <strong>generator</strong> $\mathcal Lf=\mu f'+\frac12\sigma^2f''$ — the drift of $f(X_t)$ in Itô's lemma — so that $f(X_t)-\int_0^t\mathcal Lf(X_s)\d s$ is a martingale (Dynkin's formula). Its adjoint governs the density $p(t,x)$ of $X_t$: the <strong>Fokker–Planck</strong> (forward Kolmogorov) equation
$$\partial_tp=-\partial_x(\mu p)+\tfrac12\partial_{xx}(\sigma^2p),$$
the continuous-space analogue of $P'=PQ$ for <a href="#/t/ctmc">CTMCs</a>. Two consequences used throughout modern computational statistics:</p>
<ul>
<li><strong>Langevin dynamics.</strong> For $\d X=\nabla\log\pi(X)\d t+\sqrt2\,\d B$, the density $\pi$ is stationary: in one dimension the right side of Fokker–Planck is $-(\pi'/\pi\cdot\pi)'+\pi''=0$. Discretizing with EM and correcting with a Metropolis step is <a href="#/t/langevin-mala">MALA</a>.</li>
<li><strong>Time reversal</strong> (Anderson, 1982). If $\d X=f(X,t)\d t+g(t)\d B$ has marginal densities $p_t$, the reversed process satisfies $\d X=\big[f(X,t)-g(t)^2\nabla_x\log p_t(X)\big]\d t+g(t)\d\bar B$, run backward in time with a backward Brownian motion $\bar B$. Score-based <a href="#/t/diffusion-models">diffusion models</a> learn $\nabla\log p_t$ with a neural network and integrate this reverse SDE from noise to data.</li>
</ul>
<p>Finally, <strong>Girsanov's theorem</strong> describes changes of measure: if $\theta_t$ is adapted and $Z_T=\exp(\int_0^T\theta\d B-\frac12\int_0^T\theta^2\d t)$ is a true martingale (e.g. under Novikov's condition $\E e^{\frac12\int\theta^2}< \infty$), then under $\d\mathbb Q=Z_T\d\P$ the process $B_t-\int_0^t\theta_s\d s$ is a Brownian motion. Adding drift is equivalent to reweighting paths — the foundation of risk-neutral pricing, of likelihoods for discretely observed diffusions, and of importance sampling for rare events.</p>`,
      },
    ],
    quiz: [
      { q: R`Use Itô's lemma to compute $\d(B_t^3)$ and deduce $\E B_t^4$ from $\d(B_t^4)$.`,
        a: R`$\d(B^3)=3B^2\d B+3B\d t$. $\d(B^4)=4B^3\d B+6B^2\d t$, so taking expectations (the $\d B$ term is a martingale with mean 0), $\E B_t^4=6\int_0^t\E B_s^2\d s=6\int_0^ts\d s=3t^2$, the Gaussian fourth moment.` },
      { q: R`For GBM with $\mu=0.05$ and $\sigma=0.4$ (per year), what happens to $S_t$ as $t\to\infty$? What happens to $\E S_t$?`,
        a: R`$\log S_t=\log S_0+(\mu-\sigma^2/2)t+\sigma B_t$ with $\mu-\sigma^2/2=0.05-0.08=-0.03< 0$, so $S_t\to0$ almost surely (since $B_t/t\to0$). Yet $\E S_t=S_0e^{0.05t}\to\infty$. Typical paths decay while the mean is driven by exponentially rare large paths.` },
      { q: R`What is the stationary distribution of $\d X=-2X\d t+3\d B$, and what is $\Corr(X_t,X_{t+1})$ in stationarity?`,
        a: R`OU with $\theta=2$, $\sigma=3$, $m=0$: stationary $\N(0,\sigma^2/(2\theta))=\N(0,9/4)$, autocorrelation $e^{-\theta|h|}=e^{-2}\approx0.135$ at lag 1.` },
      { q: R`Why does the Itô isometry require $H$ to be adapted? Give an example where it fails without adaptedness.`,
        a: R`The cross terms vanish because $h_i$ is independent of the future increment $\Delta_i$. With the anticipating "integrand" $H_t=B_T$ on $[0,T]$, the left-point sums give $B_T\cdot B_T=B_T^2$, whose mean is $T\ne0$; the mean-zero and isometry properties both fail ($\E B_T^4=3T^2\ne\E\int_0^TB_T^2\d t=T^2$).` },
      { q: R`Euler–Maruyama for GBM has strong order 1/2 but for OU it has strong order 1. Why?`,
        a: R`The leading neglected term in the Itô–Taylor expansion is $\frac12\sigma\sigma'((\Delta B)^2-\Delta)$. For OU the noise is additive, $\sigma'=0$, so this term is absent and EM coincides with Milstein (strong order 1). For GBM $\sigma(x)=\sigma x$, $\sigma\sigma'=\sigma^2x\ne0$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'time-series',
    domain: 'stochastic',
    order: 130,
    title: 'Stationary Time Series & ARMA Models',
    subtitle: 'Autocovariance, causality, ACF/PACF identification, and Yule–Walker estimation',
    level: 2,
    sources: { aos: '—', cb: '—', other: 'Brockwell & Davis, Time Series: Theory and Methods (2nd ed., 1991); Shumway & Stoffer, Time Series Analysis and Its Applications' },
    prereqs: ['multivariate', 'lln'],
    related: ['block-bootstrap', 'stochastic-calculus', 'hidden-markov', 'rnn-lstm', 'markov-chains'],
    tags: ['stationarity', 'autocovariance', 'AR', 'MA', 'ARMA', 'PACF', 'Wold', 'Yule–Walker'],
    summary: R`Time series observations are dependent, and the dependence is the object of study. Weak stationarity — constant mean and an autocovariance $\gamma(h)$ depending only on the lag — is the minimal structure that makes averaging over time meaningful. Within it, ARMA models $\phi(B)X_t=\theta(B)Z_t$ are finite-parameter approximations of the general linear process promised by Wold's decomposition. Whether an ARMA equation has a stationary, causal, invertible solution is decided by where the roots of two polynomials lie relative to the unit circle; the sample ACF and PACF identify orders (MA($q$): ACF cuts off after $q$; AR($p$): PACF cuts off after $p$); and the Yule–Walker equations estimate AR coefficients from sample autocovariances.`,
    survey: {
      task: 'Model, identify, estimate and forecast serially correlated data',
      assumptions: R`Weak stationarity (after differencing/detrending); linear dynamics; uncorrelated innovations`,
      output: R`$\hat\gamma(h)$, $\hat\rho(h)$, PACF; ARMA coefficients; forecasts with error variance`,
      strengths: 'Clean theory (roots, Wold, Yule–Walker); small number of interpretable parameters; fast',
      weaknesses: 'Linear and Gaussian-minded; nonstationarity (trends, unit roots, breaks) invalidates it silently',
      complexity: R`ACF $O(nH)$; Yule–Walker via Durbin–Levinson $O(p^2)$; exact ARMA likelihood $O(n)$ by Kalman filter`,
    },
    sections: [
      {
        title: 'Stationarity and the autocovariance function',
        html: R`
<div class="def" data-name="Stationarity">
<p>$(X_t)_{t\in\mathbb Z}$ is <strong>strictly stationary</strong> if $(X_{t_1+h},\dots,X_{t_k+h})\overset{d}{=}(X_{t_1},\dots,X_{t_k})$ for all $k$, $t_i$, $h$. It is <strong>weakly (covariance) stationary</strong> if $\E X_t^2< \infty$, $\E X_t=\mu$ is constant, and $\Cov(X_{t+h},X_t)=\gamma(h)$ depends only on $h$. The <strong>autocorrelation function</strong> (ACF) is $\rho(h)=\gamma(h)/\gamma(0)$.</p>
</div>
<p>Strict stationarity with finite variance implies weak; the converse holds for Gaussian processes. <strong>White noise</strong> $Z_t\sim\text{WN}(0,\sigma^2)$ means uncorrelated, mean zero, variance $\sigma^2$ — weaker than IID, an important distinction for financial returns, which are nearly uncorrelated but far from independent (their squares are autocorrelated).</p>
<div class="thm" data-kind="Proposition" data-name="Characterization of autocovariance functions">
<p>$\gamma:\mathbb Z\to\R$ is the autocovariance of a stationary process iff it is even and <strong>non-negative definite</strong>: $\sum_{i,j}a_ia_j\gamma(t_i-t_j)\ge0$ for all $a\in\R^k$, $t\in\mathbb Z^k$. In particular $\gamma(0)\ge0$ and $|\gamma(h)|\le\gamma(0)$. By Herglotz's theorem $\gamma(h)=\int_{(-\pi,\pi]}e^{ih\omega}\d F(\omega)$ for a finite spectral measure $F$; if $\sum|\gamma(h)|< \infty$, $F$ has the spectral density $f(\omega)=\frac1{2\pi}\sum_h\gamma(h)e^{-ih\omega}\ge0$.</p>
</div>
<details class="proof"><summary>Proof (necessity)</summary>
<p>$0\le\Var\big(\sum_ia_iX_{t_i}\big)=\sum_{i,j}a_ia_j\gamma(t_i-t_j)$. Evenness: $\gamma(-h)=\Cov(X_{t-h},X_t)=\Cov(X_t,X_{t+h})=\gamma(h)$. Cauchy–Schwarz gives $|\gamma(h)|\le\gamma(0)$. Sufficiency: a Gaussian process with this covariance exists by Kolmogorov's extension theorem.</p>
</details>
<p><strong>Estimation.</strong> The sample autocovariance is $\hat\gamma(h)=\frac1n\sum_{t=1}^{n-h}(X_{t+h}-\bar X)(X_t-\bar X)$. The divisor $n$ (not $n-h$) makes $\hat\Gamma_n=[\hat\gamma(i-j)]$ non-negative definite, which matters for Yule–Walker below. For IID noise with finite variance, $\sqrt n\,\hat\rho(h)\dto\N(0,1)$ for each $h\ge1$, so $\pm1.96/\sqrt n$ bands flag lags inconsistent with white noise. More generally <strong>Bartlett's formula</strong> gives the asymptotic covariance of $\hat\rho$; for an MA($q$) process at lags $h>q$, $\Var\hat\rho(h)\approx\frac1n\big(1+2\sum_{k=1}^q\rho(k)^2\big)$ — wider than the white-noise band.</p>
<div class="note" data-name="Why stationarity buys you a law of large numbers">
<p>$\Var(\bar X_n)=\frac1n\sum_{|h|< n}\big(1-\tfrac{|h|}{n}\big)\gamma(h)$. If $\gamma(h)\to0$ then $\bar X_n\to\mu$ in $L^2$; if $\sum|\gamma(h)|< \infty$, $n\Var(\bar X_n)\to\sum_h\gamma(h)=2\pi f(0)$, the <strong>long-run variance</strong>. With positive autocorrelation this exceeds $\gamma(0)$, often by a lot: for AR(1) the ratio is $(1+\phi)/(1-\phi)$, which is 19 at $\phi=0.9$. Ignoring it is the classic error behind overconfident inference on dependent data, and the reason the <a href="#/t/block-bootstrap">block bootstrap</a> exists.</p>
</div>`,
      },
      {
        title: 'AR(1): the stationarity condition',
        html: R`
<p>The first-order autoregression $X_t=\phi X_{t-1}+Z_t$, $Z_t\sim\text{WN}(0,\sigma^2)$, already displays every phenomenon.</p>
<div class="thm" data-kind="Theorem" data-name="Stationary solutions of AR(1)">
<ol>
<li>If $|\phi|< 1$, there is a unique stationary solution, $X_t=\sum_{j=0}^\infty\phi^jZ_{t-j}$ (converging in $L^2$ and a.s.), and $\gamma(h)=\dfrac{\sigma^2\phi^{|h|}}{1-\phi^2}$.</li>
<li>If $|\phi|>1$, there is a unique stationary solution, $X_t=-\sum_{j=1}^\infty\phi^{-j}Z_{t+j}$, which depends on the <em>future</em> noise.</li>
<li>If $|\phi|=1$ (unit root), there is no stationary solution.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) Iterating $k$ times, $X_t=\phi^kX_{t-k}+\sum_{j=0}^{k-1}\phi^jZ_{t-j}$. If $X$ is stationary, $\E(\phi^kX_{t-k})^2=\phi^{2k}(\gamma(0)+\mu^2)\to0$, so $X_t$ must equal the $L^2$ limit $\sum_j\phi^jZ_{t-j}$: uniqueness. Conversely that series converges since $\sum\phi^{2j}< \infty$, is stationary, and satisfies the recursion. Covariance: $\gamma(h)=\sigma^2\sum_j\phi^j\phi^{j+h}=\sigma^2\phi^h/(1-\phi^2)$ for $h\ge0$. (2) Rewrite as $X_{t}=\phi^{-1}X_{t+1}-\phi^{-1}Z_{t+1}$ and iterate forward, now with contraction factor $|\phi^{-1}|< 1$. (3) For $\phi=1$, $X_t-X_{t-n}=\sum_{j=0}^{n-1}Z_{t-j}$ has variance $n\sigma^2$, but for a stationary process $\Var(X_t-X_{t-n})=2\gamma(0)-2\gamma(n)\le4\gamma(0)$: contradiction for large $n$. Similarly for $\phi=-1$.</p>
</details>
<p>Case 2 is the important subtlety: "$|\phi|>1$ means explosive" is true only if you start the recursion at a fixed time and run it forward. The stationary solution exists but is <strong>non-causal</strong> — useless for forecasting from the past — and it has the same autocovariance as the causal AR(1) with coefficient $1/\phi$ and noise variance $\sigma^2/\phi^2$. Second-order properties cannot distinguish them, so we restrict attention to causal models. Case 3 is the random walk: the subject of unit-root testing, explored in the second lab.</p>`,
      },
      {
        title: 'ARMA processes: causality and invertibility',
        html: R`
<p>With the backshift operator $BX_t=X_{t-1}$ and polynomials $\phi(z)=1-\phi_1z-\dots-\phi_pz^p$, $\theta(z)=1+\theta_1z+\dots+\theta_qz^q$, an <strong>ARMA($p,q$)</strong> process satisfies
$$\phi(B)X_t=\theta(B)Z_t,\qquad Z_t\sim\text{WN}(0,\sigma^2),$$
assuming $\phi$ and $\theta$ have no common roots (otherwise factors cancel and the model is over-parameterized).</p>
<div class="thm" data-kind="Theorem" data-name="Existence, causality, invertibility">
<ol>
<li>A stationary solution exists (and is unique) iff $\phi(z)\ne0$ for all $|z|=1$.</li>
<li>It is <strong>causal</strong> — $X_t=\sum_{j\ge0}\psi_jZ_{t-j}$ with $\sum|\psi_j|< \infty$ — iff $\phi(z)\ne0$ for $|z|\le1$. Then $\psi(z)=\sum\psi_jz^j=\theta(z)/\phi(z)$ for $|z|\le1$.</li>
<li>It is <strong>invertible</strong> — $Z_t=\sum_{j\ge0}\pi_jX_{t-j}$ with $\sum|\pi_j|< \infty$ — iff $\theta(z)\ne0$ for $|z|\le1$. Then $\pi(z)=\phi(z)/\theta(z)$.</li>
</ol>
</div>
<details class="proof"><summary>Proof sketch (causality)</summary>
<p>If $\phi$ has no roots in the closed unit disc, $1/\phi(z)$ is analytic on a disc of radius $1+\epsilon$, so its power series $\sum\xi_jz^j$ has geometrically decaying coefficients. Apply the absolutely summable filter $\xi(B)$ to both sides: $\xi(B)\phi(B)=1$ as operators on stationary sequences (products of absolutely summable filters correspond to products of power series), giving $X_t=\xi(B)\theta(B)Z_t=\psi(B)Z_t$. Conversely, if $X=\psi(B)Z$ is causal, then $\phi(B)\psi(B)Z=\theta(B)Z$; comparing the covariances of both sides with $Z$ forces $\phi(z)\psi(z)=\theta(z)$ on $|z|\le1$, so a root of $\phi$ inside the disc would have to be a root of $\theta$, excluded by the no-common-roots assumption. Invertibility is the same argument with the roles of $\phi$ and $\theta$ swapped.</p>
</details>
<p>The coefficients $\psi_j$ are found by matching powers in $\phi(z)\psi(z)=\theta(z)$: $\psi_0=1$, $\psi_j=\theta_j+\sum_{k=1}^{\min(j,p)}\phi_k\psi_{j-k}$ (with $\theta_j=0$ for $j>q$). Then $\gamma(h)=\sigma^2\sum_j\psi_j\psi_{j+|h|}$.</p>
<div class="ex" data-name="AR(2) and MA(1)">
<p><strong>AR(2).</strong> $X_t=\phi_1X_{t-1}+\phi_2X_{t-2}+Z_t$ is causal iff $(\phi_1,\phi_2)$ lies in the triangle $\phi_1+\phi_2< 1$, $\phi_2-\phi_1< 1$, $|\phi_2|< 1$. When $\phi_1^2+4\phi_2< 0$ the roots are complex and the ACF is a damped sinusoid, giving pseudo-cycles (sunspots, business cycles) with period $2\pi/\arccos\big(\phi_1/(2\sqrt{-\phi_2})\big)$.</p>
<p><strong>MA(1).</strong> $X_t=Z_t+\theta Z_{t-1}$: $\gamma(0)=\sigma^2(1+\theta^2)$, $\gamma(1)=\sigma^2\theta$, $\gamma(h)=0$ for $h\ge2$. So $\rho(1)=\theta/(1+\theta^2)$, which is unchanged by $\theta\mapsto1/\theta$ and satisfies $|\rho(1)|\le1/2$. Every MA(1) autocovariance arises from exactly one invertible representation ($|\theta|< 1$; the boundary $|\theta|=1$ is the non-invertible edge case); invertibility is the identifiability convention that also makes the innovations recoverable from the observed past.</p>
</div>`,
      },
      {
        title: 'Identification: ACF and PACF',
        html: R`
<div class="def" data-name="Partial autocorrelation">
<p>The PACF at lag $h$ is $\alpha(h)=\phi_{hh}$, the last coefficient in the best linear predictor $\hat X_{h+1}=\phi_{h1}X_h+\dots+\phi_{hh}X_1$ of $X_{h+1}$ from $X_1,\dots,X_h$. Equivalently, it is the correlation between $X_{h+1}$ and $X_1$ after removing the linear effect of the intermediate $X_2,\dots,X_h$ from both.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Cut-off properties">
<p>For a causal AR($p$), $\alpha(h)=0$ for $h>p$ and $\alpha(p)=\phi_p$. For an MA($q$), $\rho(h)=0$ for $h>q$. For mixed ARMA, both tail off (geometrically).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>AR($p$), $h\ge p$: $X_{h+1}=\sum_{j=1}^p\phi_jX_{h+1-j}+Z_{h+1}$, and by causality $Z_{h+1}$ is uncorrelated with $X_1,\dots,X_h$. So $\sum_{j\le p}\phi_jX_{h+1-j}$ is the best linear predictor; its coefficient on $X_1$ is 0 when $h>p$ and $\phi_p$ when $h=p$. MA($q$): $X_{t+h}$ and $X_t$ share no $Z$ terms when $h>q$.</p>
</details>
<table>
<thead><tr><th>Model</th><th>ACF</th><th>PACF</th></tr></thead>
<tbody>
<tr><td>AR($p$)</td><td>tails off (damped exponentials/sinusoids)</td><td>cuts off after lag $p$</td></tr>
<tr><td>MA($q$)</td><td>cuts off after lag $q$</td><td>tails off</td></tr>
<tr><td>ARMA($p,q$)</td><td>tails off after lag $q-p$</td><td>tails off after lag $p-q$</td></tr>
</tbody>
</table>
<p>The PACF is computed from the ACF by the <strong>Durbin–Levinson recursion</strong>: $\phi_{11}=\rho(1)$ and
$$\phi_{hh}=\frac{\rho(h)-\sum_{k=1}^{h-1}\phi_{h-1,k}\,\rho(h-k)}{1-\sum_{k=1}^{h-1}\phi_{h-1,k}\,\rho(k)},\qquad \phi_{hk}=\phi_{h-1,k}-\phi_{hh}\phi_{h-1,h-k}.$$
Plugging in $\hat\rho$ gives the sample PACF; for an AR($p$), $\sqrt n\,\hat\alpha(h)\dto\N(0,1)$ for $h>p$ (Quenouille), so the same $\pm1.96/\sqrt n$ bands apply. In practice, Box–Jenkins identification reads candidate orders from these plots, then compares fitted models by AIC/BIC (<a href="#/t/model-selection">model selection</a>) and checks that residuals look like white noise (Ljung–Box test).</p>`,
        lab: 'time-series-arma',
      },
      {
        title: 'The Wold decomposition',
        html: R`
<p>Why should linear models with white-noise inputs be general enough? Because every stationary process is one, up to a perfectly predictable part. Let $P_{t-1}$ denote $L^2$ projection onto the closed span of $\{X_s:s\le t-1\}$.</p>
<div class="thm" data-kind="Theorem" data-name="Wold decomposition (Wold, 1938)">
<p>Let $X_t$ be weakly stationary with mean zero and one-step prediction error $\sigma^2=\E(X_t-P_{t-1}X_t)^2>0$. Then
$$X_t=\sum_{j=0}^\infty\psi_jZ_{t-j}+V_t,$$
where $Z_t=X_t-P_{t-1}X_t$ is white noise with variance $\sigma^2$ (the <em>innovations</em>), $\psi_0=1$ and $\sum\psi_j^2< \infty$, $\Cov(Z_s,V_t)=0$ for all $s,t$, and $V_t$ is <em>deterministic</em>: perfectly predictable from the infinitely remote past. The $\psi_j$ and $Z_t$ are uniquely determined.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The innovations $Z_t$ are orthogonal to the past $\overline{\text{span}}\{X_s:s< t\}$, which contains every $Z_s$ with $s< t$; hence they are mutually orthogonal, and by stationarity they have common variance $\sigma^2$. Project $X_t$ onto the closed span of $\{Z_{t-j}:j\ge0\}$: the coefficients are $\psi_j=\Cov(X_t,Z_{t-j})/\sigma^2$ (independent of $t$ by stationarity), $\psi_0=1$, and Bessel's inequality gives $\sum\psi_j^2< \infty$. The residual $V_t$ is orthogonal to all innovations and lies in $\bigcap_s\overline{\text{span}}\{X_u:u\le s\}$, the remote past.</p>
</details>
<p>Caveats that matter: the $Z_t$ are uncorrelated, not independent, so the Wold representation is a statement about second-order structure only — a nonlinear (e.g. GARCH) process has a trivial Wold decomposition, $\psi_j=0$ for $j\ge1$, while being far from IID. And $\psi(z)$ may need infinitely many coefficients. ARMA models approximate $\psi(z)$ by a rational function $\theta(z)/\phi(z)$, which can mimic slowly decaying $\psi_j$ with very few parameters; that parsimony is the practical case for ARMA.</p>`,
      },
      {
        title: 'Estimation: Yule–Walker and its asymptotics',
        html: R`
<p>For a causal AR($p$), multiply $X_t=\sum_{j=1}^p\phi_jX_{t-j}+Z_t$ by $X_{t-k}$ and take expectations. Causality makes $\E[Z_tX_{t-k}]=0$ for $k\ge1$ and $\sigma^2$ for $k=0$, so
$$\gamma(k)=\sum_{j=1}^p\phi_j\gamma(k-j)\ (k=1,\dots,p),\qquad\gamma(0)=\sum_{j=1}^p\phi_j\gamma(j)+\sigma^2.$$
In matrix form, the <strong>Yule–Walker equations</strong> are $\Gamma_p\boldsymbol\phi=\boldsymbol\gamma_p$ and $\sigma^2=\gamma(0)-\boldsymbol\phi^\top\boldsymbol\gamma_p$, with $\Gamma_p=[\gamma(i-j)]_{i,j=1}^p$. The method-of-moments estimator replaces $\gamma$ by $\hat\gamma$ (solved in $O(p^2)$ by Durbin–Levinson).</p>
<div class="thm" data-kind="Theorem" data-name="Asymptotics of Yule–Walker">
<p>If $X$ is a causal AR($p$) with IID noise of variance $\sigma^2$, then
$$\sqrt n(\hat{\boldsymbol\phi}-\boldsymbol\phi)\dto\N\big(0,\ \sigma^2\Gamma_p^{-1}\big),\qquad\hat\sigma^2\pto\sigma^2.$$
For AR(1): $\sqrt n(\hat\phi-\phi)\dto\N(0,1-\phi^2)$. This is the same limit as the Gaussian MLE and least squares, so Yule–Walker is asymptotically efficient for pure AR models.</p>
</div>
<p>Two finite-sample facts. Because $\hat\Gamma_p$ (divisor $n$) is positive definite whenever $\hat\gamma(0)>0$, the fitted Yule–Walker AR is always causal — a guarantee least squares lacks. But Yule–Walker is biased toward zero for persistent series ($\phi$ near 1), sometimes badly in short samples; Burg's method or least squares do better there. For MA and mixed models, moment equations are nonlinear and inefficient; use the Gaussian likelihood, evaluated exactly in $O(n)$ by the innovations algorithm or the Kalman filter on a state-space form.</p>
<div class="warn" data-name="Near and at the unit root">
<p>The normal approximation $\N(\phi,(1-\phi^2)/n)$ degenerates as $\phi\to1$. At $\phi=1$ the OLS estimator is <em>super-consistent</em>, $n(\hat\phi-1)\dto\frac{\frac12(W(1)^2-1)}{\int_0^1W(r)^2\d r}$ for a <a href="#/t/brownian-motion">Brownian motion</a> $W$ — the Dickey–Fuller distribution, skewed to the left, whose numerator is the Itô integral $\int_0^1W\d W$. Standard $t$-tests for $\phi=1$ are invalid; use Dickey–Fuller critical values. Relatedly, regressing one independent random walk on another yields "significant" slopes and high $R^2$ most of the time (spurious regression; Granger & Newbold, 1974). Difference first, or test for cointegration.</p>
</div>
<p>Connections: an AR(1) is an <a href="#/t/stochastic-calculus">Ornstein–Uhlenbeck process</a> sampled at equal spacing; AR models are <a href="#/t/markov-chains">Markov chains</a> on a continuous state space, and their stationary distributions are the subject of <a href="#/t/stationary-distributions">stationarity theory</a>; <a href="#/t/hidden-markov">hidden Markov models</a> and state-space models add observation noise; and <a href="#/t/rnn-lstm">recurrent neural networks</a> are nonlinear autoregressions with a learned state.</p>`,
        lab: 'time-series-unit-root',
      },
    ],
    quiz: [
      { q: R`Is $X_t=Z_t+2Z_{t-1}$ invertible? Find an invertible MA(1) with the same autocovariance.`,
        a: R`No: $\theta(z)=1+2z$ has root $z=-1/2$ inside the unit disc. The original has $\gamma(0)=5\sigma^2$, $\gamma(1)=2\sigma^2$. The model $X_t=W_t+\tfrac12W_{t-1}$ with $\Var W=4\sigma^2$ has $\gamma(0)=4\sigma^2\cdot\frac54=5\sigma^2$ and $\gamma(1)=4\sigma^2\cdot\frac12=2\sigma^2$ — identical second-order structure.` },
      { q: R`A sample ACF decays geometrically while the sample PACF has significant spikes at lags 1 and 2 only. What model do you propose, and why?`,
        a: R`AR(2): an AR($p$) has a PACF that cuts off after lag $p$ and an ACF that tails off. An MA would show the reverse pattern.` },
      { q: R`For the AR(1) with $\phi=0.8$, $\sigma^2=1$: find $\gamma(0)$, $\rho(3)$, and the long-run variance $\lim n\Var\bar X_n$.`,
        a: R`$\gamma(0)=1/(1-0.64)\approx2.78$; $\rho(3)=0.8^3=0.512$; long-run variance $\sigma^2/(1-\phi)^2=1/0.04=25$, nine times $\gamma(0)$: positive autocorrelation makes $\bar X$ far less precise than the IID formula suggests.` },
      { q: R`Is $(\phi_1,\phi_2)=(0.5,0.6)$ a causal AR(2)?`,
        a: R`No: $\phi_1+\phi_2=1.1>1$ violates the triangle. Equivalently $\phi(z)=1-0.5z-0.6z^2$ has $\phi(1)=-0.1< 0< \phi(0)=1$, so it has a real root in $(0,1)$, inside the unit disc.` },
      { q: R`Why does the Yule–Walker estimator use divisor $n$ in $\hat\gamma(h)$?`,
        a: R`With divisor $n$, $\hat\Gamma_p$ is the covariance matrix of a genuine (empirical) stationary sequence and hence non-negative definite — positive definite if $\hat\gamma(0)>0$. This guarantees a solution and a causal fitted AR. Divisor $n-h$ is less biased lag by lag but can produce non-NND matrices and non-causal or invalid fits.` },
    ],
  });
})();
