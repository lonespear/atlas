/* Stochastic processes, part 1: discrete-time Markov chains, classification, stationarity & mixing,
   random walks, branching processes, hidden Markov models. Labs live in js/labs/stochastic-1.js. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'markov-chains',
    domain: 'stochastic',
    order: 10,
    title: 'Discrete-Time Markov Chains',
    subtitle: 'Processes that forget everything except where they are now',
    level: 2,
    sources: { aos: '§23.2', other: 'Norris, Markov Chains (1997), Ch. 1; Grimmett & Stirzaker (2001), Ch. 6' },
    prereqs: ['conditional-probability'],
    related: ['chain-classification', 'stationary-distributions', 'mcmc-foundations', 'ctmc'],
    tags: ['Markov property', 'transition matrix', 'Chapman–Kolmogorov', 'first-step analysis'],
    summary: R`A Markov chain is a random sequence $X_0, X_1, X_2,\dots$ whose next state depends on the past only through the present state. That single conditional-independence assumption collapses the entire law of the process into two objects — an initial distribution $\mu$ and a transition matrix $P$ — and turns probability questions into linear algebra: the distribution at time $n$ is $\mu P^n$, expectations of $f(X_n)$ are $P^n f$, and hitting probabilities and expected hitting times solve linear systems. Markov chains are the workhorse model behind queues, genetics, ranking algorithms, language models, and every MCMC sampler.`,
    survey: {
      task: 'Model a random sequence with one-step memory; compute $n$-step laws, hitting probabilities, hitting times',
      assumptions: R`Markov property; time-homogeneous transition matrix $P$; countable state space`,
      output: R`Distribution $\mu P^n$; $P^n f$; solutions of first-step equations $h = Ph$ on $A^c$`,
      strengths: 'Whole law in one matrix; linear-algebraic computation; rich limit theory',
      weaknesses: 'Memory beyond one step must be encoded in the state (state-space blow-up); homogeneity often unrealistic',
      complexity: R`$O(|S|^2)$ per step of $\mu \mapsto \mu P$; $O(|S|^3 \log n)$ for $P^n$ by squaring`,
    },
    sections: [
      {
        title: 'The Markov property',
        html: R`
<p>Most random sequences we meet in statistics are IID, which is the extreme of no memory at all. Real systems remember: tomorrow's weather depends on today's, a queue's length next minute depends on its length now, the next word in a sentence depends on the words before it. Allowing <em>arbitrary</em> dependence on the past makes a process intractable — the law of $(X_0,\dots,X_n)$ would need a separate conditional distribution for every history. The Markov property is the minimal useful compromise: the past matters, but only through the present.</p>
<div class="def" data-name="Markov chain">
<p>Let $S$ be a countable set (the <em>state space</em>). A sequence of $S$-valued random variables $(X_n)_{n\ge 0}$ is a (time-homogeneous) <strong>Markov chain</strong> with initial distribution $\mu$ and transition matrix $P = (p_{ij})_{i,j\in S}$ if $\P(X_0 = i) = \mu_i$ and, for every $n\ge0$ and all states $i_0,\dots,i_{n-1}, i, j$ with $\P(X_0=i_0,\dots,X_n=i)>0$,
$$\P(X_{n+1}=j \mid X_n = i, X_{n-1}=i_{n-1},\dots,X_0=i_0) = \P(X_{n+1}=j\mid X_n=i) = p_{ij}.$$
We write $(X_n) \sim \text{Markov}(\mu, P)$. The matrix $P$ is <strong>stochastic</strong>: $p_{ij}\ge 0$ and $\sum_j p_{ij} = 1$ for every row $i$.</p>
</div>
<p>Two assumptions are packed into this definition. The <em>Markov property</em> is the first equality: conditional on the present, the future is independent of the past. <em>Time-homogeneity</em> is the second: the transition probabilities do not depend on $n$. Inhomogeneous chains exist and are sometimes needed (annealing schedules, seasonal models), but almost all of the theory below is about the homogeneous case.</p>
<div class="note">
<p>Think of $P$ as a weighted directed graph: an arrow $i\to j$ with weight $p_{ij}$ whenever $p_{ij}>0$, and the out-weights of each node summing to one. The chain is a random walker on this graph that, at each tick, picks an outgoing arrow with the indicated probabilities — consulting nothing but the node it is standing on.</p>
</div>
<p>The Markov property is a statement about a <em>representation</em>, not about the physical phenomenon. A process with two steps of memory, $\P(Y_{n+1}\mid Y_n, Y_{n-1})$, is not Markov in $Y$, but the pair $X_n = (Y_n, Y_{n-1})$ is a Markov chain on $S^2$. Any finite-memory process can be made Markov by enlarging the state; the price is the size of the state space, which grows exponentially in the memory length. Conversely, a function of a Markov chain is usually <em>not</em> Markov — merging states destroys the property unless the chain is "lumpable" — which is precisely the situation <a href="#/t/hidden-markov">hidden Markov models</a> exploit.</p>`,
      },
      {
        title: 'The law of a path, and matrix notation',
        html: R`
<p>The definition determines the probability of every finite path, and hence (by Kolmogorov's extension theorem) the law of the whole sequence.</p>
<div class="thm" data-kind="Proposition" data-name="Path probabilities">
<p>$(X_n)\sim\text{Markov}(\mu,P)$ if and only if, for every $n$ and all $i_0,\dots,i_n\in S$,
$$\P(X_0=i_0, X_1=i_1,\dots,X_n=i_n) = \mu_{i_0}\, p_{i_0 i_1}\, p_{i_1 i_2}\cdots p_{i_{n-1} i_n}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>($\Rightarrow$) By the chain rule of conditional probability,
$\P(X_0=i_0,\dots,X_n=i_n) = \P(X_0=i_0)\prod_{k=1}^n \P(X_k=i_k\mid X_{k-1}=i_{k-1},\dots,X_0=i_0)$, and the Markov property replaces each factor by $p_{i_{k-1}i_k}$. (If some prefix has probability zero, both sides are zero.)</p>
<p>($\Leftarrow$) Summing the product formula over $i_n$ shows it holds for $n-1$ as well, so the conditional probability $\P(X_{n}=j\mid X_{n-1}=i, \dots, X_0=i_0)$ is the ratio of two such products, which is $p_{ij}$ — independent of $i_0,\dots,i_{n-2}$ and of $n$.</p>
</details>
<p>The bookkeeping becomes painless once we fix a convention: <strong>measures are row vectors, functions are column vectors</strong>. A distribution $\mu$ on $S$ is a row vector; a function $f: S\to\R$ is a column vector. Then</p>
<ul>
<li>$\mu P$ is the distribution of $X_1$ when $X_0\sim\mu$: $(\mu P)_j = \sum_i \mu_i p_{ij}$.</li>
<li>$Pf$ is the function $i\mapsto \E[f(X_1)\mid X_0=i] = \sum_j p_{ij}f(j)$.</li>
<li>$\mu P f = \E_\mu f(X_1)$ is a number, and the two ways of bracketing it — push the distribution forward, or pull the function back — agree.</li>
</ul>
<p>This duality is worth internalizing: "evolve the distribution" ($\mu\mapsto\mu P$, the forward or Fokker–Planck picture) and "evolve the observable" ($f\mapsto Pf$, the backward or Kolmogorov picture) are the same computation read in different directions.</p>
<div class="ex" data-name="A two-state chain">
<p>Let $S=\{0,1\}$ and $P = \begin{pmatrix}1-\alpha & \alpha\\ \beta & 1-\beta\end{pmatrix}$ with $\alpha,\beta\in[0,1]$, not both zero. Think of 0 = "machine working", 1 = "machine broken", with failure probability $\alpha$ and repair probability $\beta$ per day. The path $0\,0\,1\,1\,0$ started at $0$ has probability $(1-\alpha)\,\alpha\,(1-\beta)\,\beta$. Notice that the time spent in a state before leaving is <em>geometric</em> — sojourn in state 0 lasts $k$ steps with probability $(1-\alpha)^{k-1}\alpha$. Geometric holding times are the fingerprint of the Markov property in discrete time.</p>
</div>`,
        lab: 'markov-chains-simulator',
      },
      {
        title: 'Chapman–Kolmogorov: n-step transitions are matrix powers',
        html: R`
<p>Write $p^{(n)}_{ij} = \P(X_{m+n}=j\mid X_m=i)$ for the $n$-step transition probability (it does not depend on $m$, by homogeneity).</p>
<div class="thm" data-name="Chapman–Kolmogorov equations">
<p>For all $m,n\ge0$ and $i,j\in S$,
$$p^{(m+n)}_{ij} = \sum_{k\in S} p^{(m)}_{ik}\,p^{(n)}_{kj}.$$
Consequently $p^{(n)}_{ij} = (P^n)_{ij}$, and if $X_0\sim\mu$ then $X_n\sim\mu P^n$ and $\E_\mu f(X_n) = \mu P^n f$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Condition on the state at the intermediate time $m$ and use the law of total probability:
$$\P(X_{m+n}=j\mid X_0=i) = \sum_k \P(X_m = k\mid X_0=i)\,\P(X_{m+n}=j\mid X_m=k, X_0=i).$$
By the Markov property (in the form of the path formula, summed over the intermediate states) the last factor equals $\P(X_{m+n}=j\mid X_m=k) = p^{(n)}_{kj}$. Hence $P^{(m+n)} = P^{(m)}P^{(n)}$ as matrices; since $P^{(1)} = P$ and $P^{(0)} = I$, induction gives $P^{(n)} = P^n$. The statements about $\mu$ and $f$ follow by multiplying on the left or right.</p>
</details>
<p>The equations say that the transition matrices form a <em>semigroup</em> under multiplication. That is the algebraic heart of the subject: long-run behaviour of the chain is the behaviour of $P^n$ as $n\to\infty$, which for a finite chain is governed by the eigenvalues of $P$.</p>
<div class="ex" data-name="Closed form for the two-state chain">
<p>For the two-state chain above with $\alpha+\beta>0$, the eigenvalues of $P$ are $1$ and $\lambda = 1-\alpha-\beta$. Diagonalizing gives
$$P^n = \frac{1}{\alpha+\beta}\begin{pmatrix}\beta & \alpha\\ \beta & \alpha\end{pmatrix} + \frac{(1-\alpha-\beta)^n}{\alpha+\beta}\begin{pmatrix}\alpha & -\alpha\\ -\beta & \beta\end{pmatrix}.$$
(Check $n=0$ and $n=1$ directly.) If $|1-\alpha-\beta|<1$, every row converges to $\big(\tfrac{\beta}{\alpha+\beta}, \tfrac{\alpha}{\alpha+\beta}\big)$ geometrically fast at rate $|1-\alpha-\beta|^n$: the chain forgets its starting state. If $\alpha=\beta=1$ then $\lambda=-1$ and $P^n$ alternates forever between $I$ and $P$ — a periodic chain that never settles. If $\alpha=\beta=0$ then $P=I$ and nothing moves. These three behaviours — geometric convergence, periodic oscillation, and reducibility — are exactly what the <a href="#/t/chain-classification">classification theory</a> and the <a href="#/t/stationary-distributions">convergence theorem</a> sort out in general.</p>
</div>
<p>Computationally, $\mu P^n$ costs $O(n|S|^2)$ by repeated vector–matrix products, while $P^n$ itself can be had in $O(|S|^3\log n)$ operations by repeated squaring. For huge sparse chains (web graphs) one never forms $P^n$; one iterates $\mu\mapsto\mu P$, which is the power method.</p>`,
        lab: 'markov-chains-nstep',
      },
      {
        title: 'The strong Markov property and the random-mapping construction',
        html: R`
<p>The Markov property at a <em>fixed</em> time $m$ says: conditional on $X_m=i$, the post-$m$ process $(X_{m+n})_{n\ge0}$ is $\text{Markov}(\delta_i,P)$ and independent of $X_0,\dots,X_m$. Most arguments need this at <em>random</em> times — the first time the chain hits a state, the $k$-th return — and it remains true provided the random time does not peek into the future.</p>
<div class="def" data-name="Stopping time">
<p>A random variable $T$ with values in $\{0,1,2,\dots\}\cup\{\infty\}$ is a <strong>stopping time</strong> if for each $n$ the event $\{T=n\}$ is determined by $X_0,\dots,X_n$. Hitting times $H_A = \inf\{n\ge0: X_n\in A\}$ and return times $T_i = \inf\{n\ge1: X_n = i\}$ are stopping times; "the last time the chain visits $i$" is not.</p>
</div>
<div class="thm" data-name="Strong Markov property">
<p>Let $T$ be a stopping time. Conditional on $T<\infty$ and $X_T=i$, the process $(X_{T+n})_{n\ge0}$ is $\text{Markov}(\delta_i,P)$ and is independent of $X_0,\dots,X_T$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Let $B$ be an event determined by $X_0,\dots,X_T$. Then $B\cap\{T=m\}$ is determined by $X_0,\dots,X_m$, so by the simple Markov property at the fixed time $m$,
$\P(\{X_T=j_0,\dots,X_{T+n}=j_n\}\cap B\cap\{T=m\}\cap\{X_T=i\}) = \P_i(X_0=j_0,\dots,X_n=j_n)\,\P(B\cap\{T=m\}\cap\{X_T=i\}).$
Summing over $m<\infty$ and dividing by $\P(T<\infty, X_T=i)$ gives the claim.</p>
</details>
<p>The strong Markov property is what makes <em>excursion</em> and <em>regeneration</em> arguments work: each return to a state $i$ starts a fresh, independent copy of the chain from $i$. That is the engine behind the recurrence criterion, the formula $\pi_i = 1/\E_i T_i$, and the ergodic theorem.</p>
<p>A second structural fact is that every Markov chain can be built from IID noise.</p>
<div class="thm" data-kind="Proposition" data-name="Random mapping representation">
<p>Given $P$ on a countable $S$, there is a function $\phi: S\times[0,1]\to S$ such that if $U_1,U_2,\dots\iid\text{Unif}(0,1)$ are independent of $X_0\sim\mu$, then $X_{n+1} = \phi(X_n, U_{n+1})$ defines a $\text{Markov}(\mu,P)$ chain.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Enumerate $S = \{s_1, s_2,\dots\}$ and let $\phi(i,u) = s_k$ when $\sum_{l< k}p_{i s_l} < u \le \sum_{l\le k} p_{i s_l}$ — inverse-CDF sampling from row $i$. Then $\P(\phi(i,U)=j) = p_{ij}$, and because $U_{n+1}$ is independent of $(X_0,\dots,X_n)$, $\P(X_{n+1}=j\mid X_n=i,\text{past}) = \P(\phi(i,U_{n+1})=j) = p_{ij}$.</p>
</details>
<p>This is how every simulator works (including the labs on this page), and it is the starting point for <em>coupling</em>: drive two copies of the chain with the same or cleverly correlated noise and study when they meet. Coupling-from-the-past (Propp–Wilson) uses exactly this representation to draw <em>exact</em> samples from a stationary distribution.</p>`,
      },
      {
        title: 'First-step analysis: hitting probabilities and hitting times',
        html: R`
<p>Many practical questions are about what the chain does <em>before</em> it reaches some set $A$: will it get there at all, and how long will it take? Conditioning on the first step turns these into linear equations.</p>
<div class="thm" data-name="Hitting probabilities and expected hitting times">
<p>Let $A\subseteq S$, $H_A = \inf\{n\ge0: X_n\in A\}$, $h_i = \P_i(H_A<\infty)$ and $k_i = \E_i H_A$. Then
$$h_i = 1\ (i\in A),\qquad h_i = \sum_j p_{ij}h_j\ (i\notin A);$$
$$k_i = 0\ (i\in A),\qquad k_i = 1+\sum_j p_{ij}k_j\ (i\notin A).$$
Moreover $h$ (respectively $k$) is the <strong>minimal non-negative solution</strong> of its system.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For $i\notin A$, $H_A\ge1$ and, by the Markov property at time 1, the process $(X_{1+n})$ is a chain started from $X_1$. So $\P_i(H_A<\infty\mid X_1=j) = h_j$ and $\E_i[H_A\mid X_1=j] = 1+k_j$; average over $X_1\sim p_{i\cdot}$.</p>
<p>Minimality for $h$: let $x\ge0$ solve the system. For $i\notin A$, iterate the equation: $x_i = \sum_{j\in A}p_{ij} + \sum_{j\notin A}p_{ij}x_j = \P_i(H_A=1) + \sum_{j\notin A}p_{ij}\big(\sum_{k\in A}p_{jk} + \sum_{k\notin A}p_{jk}x_k\big) = \P_i(H_A\le 2) + \sum_{j,k\notin A}p_{ij}p_{jk}x_k.$ After $n$ substitutions $x_i = \P_i(H_A\le n) + (\text{non-negative remainder}) \ge \P_i(H_A\le n)$, and letting $n\to\infty$ gives $x_i\ge h_i$. The argument for $k$ is identical.</p>
</details>
<p>Minimality matters on infinite state spaces, where the system typically has many solutions (for instance $x\equiv1$ always solves the $h$-system). On finite spaces with $A$ reachable from everywhere, the solution is unique.</p>
<div class="ex" data-name="Waiting for two heads in a row">
<p>Toss a fair coin until two consecutive heads appear. Track the state "current run of heads": $0$, $1$, or $2$ (absorbing). From $0$: heads $\to1$, tails $\to0$. From $1$: heads $\to2$, tails $\to0$. With $k_2=0$,
$$k_0 = 1+\tfrac12 k_1+\tfrac12 k_0,\qquad k_1 = 1+\tfrac12\cdot 0+\tfrac12 k_0.$$
The first gives $k_0 = 2+k_1$; substituting into the second, $k_1 = 1+1+\tfrac12k_1$, so $k_1=4$ and $k_0 = 6$. (For "HT" the same method gives 4: patterns that overlap themselves take longer to appear — a fact that surprises most people and that a martingale argument explains elegantly.)</p>
</div>
<p>First-step analysis is the discrete ancestor of the backward Kolmogorov equation: the unknown function satisfies $(P-I)h = 0$ off $A$, i.e. $h$ is <em>harmonic</em> for the chain there. The same idea drives <a href="#/t/random-walks">gambler's ruin</a>, absorption probabilities in <a href="#/t/chain-classification">reducible chains</a>, and value iteration in Markov decision processes.</p>`,
      },
      {
        title: 'Pitfalls and connections',
        html: R`
<div class="warn" data-name="Functions of Markov chains are rarely Markov">
<p>Merge states $2$ and $3$ of a three-state chain into a single label. The resulting label process is Markov only if every state in the merged block has the same total probability of jumping to each other block (Kemeny–Snell lumpability). Otherwise the label's future depends on <em>which</em> member of the block you are in, which depends on the past. Coarse-graining a model — binning a continuous variable, collapsing categories — generally manufactures memory.</p>
</div>
<div class="warn" data-name="Row versus column conventions">
<p>Probabilists use row-stochastic $P$ with distributions as row vectors ($\mu\mapsto\mu P$). Some numerical and engineering texts use column-stochastic matrices ($\mu\mapsto P\mu$). Mixing the two silently transposes the chain. When you compute a stationary distribution by an eigen-solver, make sure you are taking a <em>left</em> eigenvector of a row-stochastic matrix.</p>
</div>
<div class="warn" data-name="Estimating P from data">
<p>Given one observed path, the MLE of $p_{ij}$ is $\hat p_{ij} = N_{ij}/N_{i\cdot}$, the fraction of departures from $i$ that went to $j$ (the likelihood is a product of multinomials, one per row, conditioning on $X_0$). Its precision is governed by the number of <em>visits</em> to $i$, not the path length — rarely visited states are estimated poorly, and states never visited are not estimated at all.</p>
</div>
<p>Where this leads:</p>
<ul>
<li><a href="#/t/chain-classification">Classification</a> — which states communicate, which are visited infinitely often, and with what period.</li>
<li><a href="#/t/stationary-distributions">Stationary distributions</a> — the fixed points $\pi = \pi P$, why $\mu P^n\to\pi$, and how fast.</li>
<li><a href="#/t/mcmc-foundations">MCMC</a> — run the logic backwards: design $P$ so that a given target is stationary, then simulate.</li>
<li><a href="#/t/ctmc">Continuous-time chains</a> replace $P^n$ by $e^{tQ}$, and <a href="#/t/hidden-markov">hidden Markov models</a> observe a chain only through noise.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Let $(Y_n)$ be IID fair coin flips and $X_n = Y_n + Y_{n-1}$. Is $(X_n)$ a Markov chain?`,
        a: R`No. $X_n=1$ can arise from $(Y_{n-1},Y_n) = (0,1)$ or $(1,0)$, and the past reveals which: if $X_{n-1}=2$ then $Y_{n-1}=1$, so $X_n=1$ forces $Y_n=0$ and $X_{n+1}\in\{0,1\}$; if $X_{n-1}=0$ then $X_n=1$ forces $Y_n=1$ and $X_{n+1}\in\{1,2\}$. So $\P(X_{n+1}=2\mid X_n=1, X_{n-1})$ depends on $X_{n-1}$. The pair $(Y_{n-1},Y_n)$ <em>is</em> Markov; $X_n$ is a non-lumpable function of it.` },
      { q: R`If $X_0\sim\mu$, what are the distribution of $X_n$ and the value of $\E[f(X_n)\mid X_0=i]$, in matrix notation? Why does the order of multiplication differ?`,
        a: R`$X_n\sim\mu P^n$ (row vector times matrix) and $\E[f(X_n)\mid X_0=i] = (P^nf)(i)$ (matrix times column vector). Distributions are pushed forward along the arrows, so they multiply on the left; functions are pulled back (we average $f$ over where we go next), so they multiply on the right. Both are consequences of Chapman–Kolmogorov.` },
      { q: R`For the two-state chain with $\alpha = 0.1$, $\beta = 0.3$, what is $\lim_n p^{(n)}_{00}$ and how fast is it approached?`,
        a: R`The limit is $\beta/(\alpha+\beta) = 0.75$. The error is $\frac{\alpha}{\alpha+\beta}(1-\alpha-\beta)^n = 0.25\cdot 0.6^n$, so it shrinks geometrically at rate equal to the second eigenvalue $0.6$.` },
      { q: R`Why is the solution of the hitting-probability equations required to be the <em>minimal</em> non-negative one?`,
        a: R`On infinite state spaces the equations $h = Ph$ off $A$, $h=1$ on $A$, can have several solutions — $h\equiv1$ always works. The probabilistic quantity is the smallest, because iterating the equations shows any non-negative solution dominates $\P_i(H_A\le n)$ for every $n$. For a transient random walk drifting away from $A$, the true $h$ is strictly less than 1 and only minimality picks it out.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'chain-classification',
    domain: 'stochastic',
    order: 20,
    title: 'Classifying States: Communication, Recurrence & Periodicity',
    subtitle: 'Which states the chain keeps coming back to, and on what rhythm',
    level: 2,
    sources: { aos: '§23.2', other: 'Norris (1997), §§1.2–1.8; Feller, Vol. I (1968), Ch. XV' },
    prereqs: ['markov-chains'],
    related: ['stationary-distributions', 'random-walks', 'martingales'],
    tags: ['communicating classes', 'recurrence', 'transience', 'period', 'absorption'],
    summary: R`Before asking where a Markov chain settles, we must ask which states it can reach, which it returns to forever, and whether those returns happen on a fixed rhythm. States split into communicating classes; each class is either recurrent (returns are certain, and happen infinitely often) or transient (visited finitely often), with the clean criterion $\sum_n p^{(n)}_{ii} = \infty$ for recurrence. Recurrent states further split into positive recurrent (finite mean return time) and null recurrent. Finally each class has a period $d$, and only aperiodic classes can converge. These three properties — irreducibility, positive recurrence, aperiodicity — are exactly the hypotheses of the convergence theorem.`,
    survey: {
      task: 'Decompose a chain into classes; decide recurrence/transience, positive/null recurrence, and period',
      assumptions: 'Time-homogeneous Markov chain on a countable state space',
      output: R`Partition $S = T\cup C_1\cup C_2\cup\cdots$; period $d(i)$; absorption probabilities $B = (I-Q)^{-1}R$`,
      strengths: 'All three properties are class properties: check one state, know the class',
      weaknesses: R`On infinite spaces, recurrence needs analysis (e.g. asymptotics of $p^{(n)}_{ii}$), not just graph structure`,
      complexity: R`Classes: $O(|S|+|E|)$ (strongly connected components); period: BFS levels; absorption: $O(|T|^3)$`,
    },
    sections: [
      {
        title: 'Communicating classes',
        html: R`
<p>Classification starts from the transition <em>graph</em> — which moves are possible — and ignores the numerical values of $p_{ij}$.</p>
<div class="def" data-name="Accessibility and communication">
<p>State $j$ is <strong>accessible</strong> from $i$, written $i\to j$, if $p^{(n)}_{ij}>0$ for some $n\ge0$; equivalently there is a path $i=i_0,i_1,\dots,i_n=j$ with every $p_{i_{k-1}i_k}>0$. States $i$ and $j$ <strong>communicate</strong>, $i\leftrightarrow j$, if $i\to j$ and $j\to i$.</p>
</div>
<p>Communication is reflexive ($n=0$), symmetric by definition, and transitive because $p^{(m+n)}_{ik}\ge p^{(m)}_{ij}p^{(n)}_{jk}$ by Chapman–Kolmogorov. So it is an equivalence relation, and $S$ splits into disjoint <strong>communicating classes</strong> — the strongly connected components of the transition graph.</p>
<div class="def" data-name="Closed classes and irreducibility">
<p>A class $C$ is <strong>closed</strong> if $i\in C$ and $i\to j$ imply $j\in C$: once in, the chain never leaves. A state $i$ with $p_{ii}=1$ is <strong>absorbing</strong> (a closed class of size one). The chain is <strong>irreducible</strong> if $S$ is a single class.</p>
</div>
<p>Collapsing each class to a node yields a directed acyclic graph; closed classes are its sinks. A chain started in a non-closed class eventually leaves it (on a finite space; see below) and falls into some closed class, never to return. The interesting long-run behaviour therefore lives on the closed classes, and the interesting transient behaviour is the question of <em>which</em> closed class captures the chain.</p>`,
        lab: 'chain-classification-explorer',
      },
      {
        title: 'Recurrence and transience',
        html: R`
<p>Let $T_i = \inf\{n\ge1: X_n=i\}$ be the first return time to $i$ and $f_i = \P_i(T_i<\infty)$ the probability of ever returning.</p>
<div class="def" data-name="Recurrent and transient states">
<p>State $i$ is <strong>recurrent</strong> if $f_i = 1$ and <strong>transient</strong> if $f_i<1$.</p>
</div>
<p>The dichotomy is sharper than it looks, because of the strong Markov property. Let $V_i = \sum_{n\ge1}\ind\{X_n=i\}$ be the number of returns.</p>
<div class="thm" data-name="Recurrence criterion">
<p>Under $\P_i$, $V_i$ is geometric: $\P_i(V_i\ge k) = f_i^k$ for $k\ge0$. Hence</p>
<ul>
<li>if $f_i=1$: $\P_i(V_i=\infty)=1$ and $\sum_{n\ge1}p^{(n)}_{ii} = \infty$;</li>
<li>if $f_i<1$: $\P_i(V_i<\infty)=1$ and $\sum_{n\ge1}p^{(n)}_{ii} = \dfrac{f_i}{1-f_i}<\infty$.</li>
</ul>
<p>In particular, $i$ is recurrent if and only if $\sum_{n} p^{(n)}_{ii} = \infty$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $T^{(k)}_i$ be the time of the $k$-th return. $\{V_i\ge k+1\} = \{T^{(k)}_i<\infty,\ T^{(k+1)}_i<\infty\}$. On $\{T^{(k)}_i<\infty\}$ we have $X_{T^{(k)}_i}=i$, and by the strong Markov property the process after $T^{(k)}_i$ is a fresh chain from $i$, independent of the past; so it returns again with probability $f_i$. Thus $\P_i(V_i\ge k+1) = f_i\,\P_i(V_i\ge k)$, and $\P_i(V_i\ge k) = f_i^k$ by induction.</p>
<p>Now compute $\E_iV_i$ two ways. By Tonelli, $\E_iV_i = \sum_{n\ge1}\P_i(X_n=i) = \sum_{n\ge1}p^{(n)}_{ii}$. By the geometric law, $\E_iV_i = \sum_{k\ge1}\P_i(V_i\ge k) = \sum_{k\ge1}f_i^k$, which is $f_i/(1-f_i)$ if $f_i<1$ and $\infty$ if $f_i=1$. If $f_i=1$ then $\P_i(V_i\ge k)=1$ for all $k$, so $V_i=\infty$ a.s.</p>
</details>
<p>So there is no middle ground: a state is either visited infinitely often almost surely, or finitely often almost surely. The criterion $\sum_n p^{(n)}_{ii}=\infty$ converts a question about path behaviour into one about the asymptotics of a single sequence of numbers — which is how Pólya's theorem on <a href="#/t/random-walks">random walks</a> is proved.</p>
<div class="thm" data-kind="Proposition" data-name="Recurrence is a class property">
<p>If $i\leftrightarrow j$ and $i$ is recurrent, then $j$ is recurrent. Moreover every recurrent class is closed, and every finite closed class is recurrent.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Choose $r,s$ with $p^{(r)}_{ji}>0$, $p^{(s)}_{ij}>0$. Then $p^{(r+n+s)}_{jj}\ge p^{(r)}_{ji}\,p^{(n)}_{ii}\,p^{(s)}_{ij}$, so $\sum_n p^{(n)}_{jj}\ge p^{(r)}_{ji}p^{(s)}_{ij}\sum_n p^{(n)}_{ii} = \infty$.</p>
<p>Closedness: suppose $i$ is recurrent and $i\to j$ but $j\not\to i$. Then with positive probability the chain from $i$ reaches $j$ and thereafter can never return to $i$, contradicting $f_i=1$. So $j\to i$, i.e. $j$ is in the same class.</p>
<p>Finite closed class $C$: started in $C$ the chain stays in $C$ forever, so over infinitely many steps some state of the finite set $C$ is visited infinitely often with positive probability. That state cannot be transient (transient states are visited finitely often a.s. from any start, by the strong Markov property at the first visit), so it is recurrent, and hence so is all of $C$.</p>
</details>
<div class="note">
<p>On a finite state space classification is pure graph theory: closed classes are recurrent, all other states are transient. Numbers only matter on infinite state spaces, where an irreducible chain can be transient (a random walk with drift, or simple random walk in $\mathbb Z^3$) because there is always "more room" to escape to.</p>
</div>`,
      },
      {
        title: 'Positive and null recurrence',
        html: R`
<p>Certain return is not the same as prompt return. Let $m_i = \E_i T_i$ be the mean return time.</p>
<div class="def" data-name="Positive and null recurrence">
<p>A recurrent state $i$ is <strong>positive recurrent</strong> if $m_i<\infty$ and <strong>null recurrent</strong> if $m_i=\infty$.</p>
</div>
<p>Positive recurrence is again a class property, and for an irreducible chain it is equivalent to the existence of a stationary distribution, with $\pi_i = 1/m_i$ — this is proved on the <a href="#/t/stationary-distributions">stationary distributions</a> page. Finite irreducible chains are always positive recurrent. Null recurrence is a genuinely infinite-state phenomenon: the chain returns to every state infinitely often, yet spends an asymptotically vanishing fraction of time at each one, and $p^{(n)}_{ij}\to0$.</p>
<div class="ex" data-name="A reflecting walk on the half-line">
<p>On $S=\{0,1,2,\dots\}$ let the chain step up with probability $p$ and down with probability $q=1-p$, except that at $0$ a "down" step means staying at $0$. Write $\rho = p/q$.</p>
<ul>
<li>$p<\tfrac12$: the drift pulls toward $0$. Detailed balance $\pi_k p = \pi_{k+1}q$ gives the geometric law $\pi_k = (1-\rho)\rho^k$, so the chain is positive recurrent with $m_0 = 1/\pi_0 = q/(q-p)$.</li>
<li>$p=\tfrac12$: by comparison with simple random walk on $\mathbb Z$, the chain is recurrent, but the only invariant measure is constant, which cannot be normalized — null recurrent. Here $\P_0(T_0>n)\asymp n^{-1/2}$, whose sum diverges.</li>
<li>$p>\tfrac12$: the drift carries the chain to infinity; by the strong law $X_n/n\to p-q>0$, so it visits $0$ only finitely often — transient.</li>
</ul>
<p>The second lab below simulates return times to $0$ and plots their survival function on log–log axes: a fast-decaying tail, a $t^{-1/2}$ power law, or a plateau at the escape probability.</p>
</div>`,
        lab: 'chain-classification-return',
      },
      {
        title: 'Periodicity',
        html: R`
<p>Even a positive recurrent chain can fail to converge if its returns are locked to a rhythm.</p>
<div class="def" data-name="Period">
<p>The <strong>period</strong> of state $i$ is $d(i) = \gcd\{n\ge1: p^{(n)}_{ii}>0\}$. The state is <strong>aperiodic</strong> if $d(i)=1$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Period is a class property">
<p>If $i\leftrightarrow j$ then $d(i)=d(j)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Take $r,s$ with $p^{(r)}_{ij}>0$ and $p^{(s)}_{ji}>0$. Then $p^{(r+s)}_{ii}>0$, so $d(i)\mid r+s$. If $p^{(n)}_{jj}>0$ then $p^{(r+n+s)}_{ii}\ge p^{(r)}_{ij}p^{(n)}_{jj}p^{(s)}_{ji}>0$, so $d(i)\mid r+n+s$ and therefore $d(i)\mid n$. Thus $d(i)$ divides every return time of $j$, so $d(i)\mid d(j)$. By symmetry $d(j)\mid d(i)$.</p>
</details>
<div class="thm" data-kind="Lemma" data-name="Aperiodic means eventually always possible">
<p>If $i$ is aperiodic, then $p^{(n)}_{ii}>0$ for all sufficiently large $n$. If the chain is irreducible and aperiodic, then for each pair $i,j$, $p^{(n)}_{ij}>0$ for all large $n$; on a finite space, $P^n$ has all entries positive for some $n$ (the chain is <em>regular</em>).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The set $A = \{n\ge1: p^{(n)}_{ii}>0\}$ is closed under addition (Chapman–Kolmogorov). A standard number-theoretic fact: an additively closed set of positive integers with gcd 1 contains every sufficiently large integer. (Finitely many elements $a_1,\dots,a_k\in A$ already have gcd 1; by Bézout, $1 = \sum c_l a_l$ with integer $c_l$; splitting into positive and negative parts gives $x,\ x+1\in A$, and then every $n\ge x^2$ is a non-negative combination of $x$ and $x+1$.) For $i\ne j$, take $r$ with $p^{(r)}_{ij}>0$ and use $p^{(n+r)}_{ij}\ge p^{(n)}_{ii}p^{(r)}_{ij}$.</p>
</details>
<p>An irreducible chain of period $d>1$ has a <strong>cyclic decomposition</strong>: $S$ splits into $d$ sets $C_0,\dots,C_{d-1}$ such that the chain moves deterministically $C_0\to C_1\to\cdots\to C_{d-1}\to C_0$. Then $P^d$ restricted to each $C_r$ is irreducible and aperiodic. Bipartite graphs give $d=2$: a random walk on a chessboard alternates colours.</p>
<div class="note">
<p>A single self-loop ($p_{ii}>0$ for one $i$) makes an irreducible chain aperiodic. That is why MCMC practitioners speak of "lazy" chains: replacing $P$ by $\tfrac12(I+P)$ kills periodicity, preserves the stationary distribution, and makes all eigenvalues non-negative — at the cost of moving half as often. Metropolis–Hastings is automatically lazy whenever proposals can be rejected.</p>
</div>`,
      },
      {
        title: 'The canonical decomposition and absorption',
        html: R`
<div class="thm" data-name="Decomposition theorem">
<p>The state space partitions uniquely as $S = T\cup C_1\cup C_2\cup\cdots$, where $T$ is the set of transient states and the $C_k$ are closed recurrent classes. A chain started in $C_k$ stays there forever and visits every state of $C_k$ infinitely often. A chain started in $T$ either stays in $T$ forever (possible only if $T$ is infinite) or enters some $C_k$ and stays.</p>
</div>
<p>For a finite chain, order the states transient-first and write
$$P = \begin{pmatrix} Q & R\\ 0 & P_{\text{rec}}\end{pmatrix},$$
where $Q$ holds transitions among transient states and $R$ transitions from transient to recurrent states. Since the transient states are eventually left, $Q^n\to0$, so $I-Q$ is invertible with
$$N = (I-Q)^{-1} = \sum_{n\ge0}Q^n.$$
This <strong>fundamental matrix</strong> has a direct meaning: $N_{ij}$ is the expected number of visits to transient $j$ starting from transient $i$ (sum the probabilities $(Q^n)_{ij}$ of being at $j$ at time $n$). Consequently:</p>
<ul>
<li>expected time to absorption from $i$: $(N\mathbf 1)_i$;</li>
<li>probability of being absorbed at recurrent state $k$: $B = NR$, i.e. $B_{ik} = \sum_j N_{ij}R_{jk}$ (last transient visit at $j$, then jump to $k$).</li>
</ul>
<p>Both follow from first-step analysis too: $B = R + QB$ rearranges to $B=(I-Q)^{-1}R$.</p>
<div class="ex" data-name="A small absorbing chain">
<p>States $\{1,2,3,4\}$; $1$ and $4$ absorbing; from $2$: to $1$ w.p. $\tfrac12$, to $3$ w.p. $\tfrac12$; from $3$: to $2$ w.p. $\tfrac13$, to $4$ w.p. $\tfrac23$. Then $Q = \begin{pmatrix}0 & \tfrac12\\ \tfrac13 & 0\end{pmatrix}$, $\det(I-Q) = 1-\tfrac16 = \tfrac56$, and
$$N = \tfrac65\begin{pmatrix}1 & \tfrac12\\ \tfrac13 & 1\end{pmatrix},\qquad N\mathbf 1 = \begin{pmatrix}9/5\\ 8/5\end{pmatrix},\qquad B = N\begin{pmatrix}\tfrac12 & 0\\ 0 & \tfrac23\end{pmatrix} = \begin{pmatrix}3/5 & 2/5\\ 1/5 & 4/5\end{pmatrix}.$$
From state 2 the chain is absorbed at 1 with probability $3/5$, after $9/5$ steps on average. Rows of $B$ sum to one, as they must.</p>
</div>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Irreducible is not enough for convergence">
<p>An irreducible, positive recurrent chain has a unique stationary distribution, but $p^{(n)}_{ij}$ need not converge — the deterministic 3-cycle has $\pi = (\tfrac13,\tfrac13,\tfrac13)$ while $P^n$ cycles forever. Aperiodicity is a separate hypothesis. Time averages still converge (the <a href="#/t/stationary-distributions">ergodic theorem</a> needs no aperiodicity); marginal laws do not.</p>
</div>
<div class="warn" data-name="Recurrent is not positive recurrent">
<p>Simple symmetric random walk on $\mathbb Z$ returns to the origin with probability one, but the expected return time is infinite. Sample averages of return times do not stabilize, and no stationary distribution exists. Simulation can be badly misleading here: any finite run shows finite return times.</p>
</div>
<div class="warn" data-name="Graph structure only on finite spaces">
<p>On an infinite irreducible chain, recurrence depends on the numerical transition probabilities, not just on which are non-zero: the same graph ($\mathbb Z$ with nearest-neighbour edges) is recurrent at $p=\tfrac12$ and transient at $p\ne\tfrac12$.</p>
</div>
<div class="warn" data-name="Transient chains in MCMC">
<p>A sampler whose target is improper (a posterior that does not integrate) can still be run; it simply corresponds to a null-recurrent or transient chain, and its output wanders without ever settling. Trace plots that drift indefinitely are a symptom worth taking seriously — see <a href="#/t/mcmc-diagnostics">MCMC diagnostics</a>.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In a finite chain, can every state be transient?`,
        a: R`No. If all states were transient, each would be visited only finitely often almost surely, but the chain must be somewhere at every one of infinitely many times, so some state in the finite set is visited infinitely often — contradiction. Equivalently: the class DAG has at least one sink, and finite closed classes are recurrent.` },
      { q: R`State $i$ has $p^{(n)}_{ii}\sim c\,n^{-3/2}$. Is $i$ recurrent? What if $p^{(n)}_{ii}\sim c\,n^{-1/2}$?`,
        a: R`With $n^{-3/2}$ the series $\sum_n p^{(n)}_{ii}$ converges, so $i$ is transient and the expected number of returns is finite. With $n^{-1/2}$ the series diverges, so $i$ is recurrent. These are the 3-D and 1-D simple random walk rates (along even $n$).` },
      { q: R`A chain on $\{1,2,3,4\}$ moves $1\to2$, $2\to3$, $3\to4$, $4\to1$ and also $2\to4$ (each with positive probability). What is its period?`,
        a: R`Return loops through 1: $1\to2\to3\to4\to1$ has length 4 and $1\to2\to4\to1$ has length 3. $\gcd(3,4)=1$, so the chain is aperiodic even though it has no self-loops.` },
      { q: R`What does the entry $N_{ij}$ of the fundamental matrix $(I-Q)^{-1}$ count, and why does $(I-Q)^{-1} = \sum_n Q^n$ hold?`,
        a: R`$N_{ij}$ is the expected number of time steps spent in transient state $j$ when starting from transient state $i$ (including time 0 if $i=j$), because $(Q^n)_{ij} = \P_i(X_n=j)$ for transient $i,j$ (paths staying in $T$). The Neumann series converges because the transient part is eventually left, so $Q^n\to0$ geometrically and the spectral radius of $Q$ is below 1.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'stationary-distributions',
    domain: 'stochastic',
    order: 30,
    title: 'Stationary Distributions, Convergence & Mixing',
    subtitle: 'Where a chain settles, why it gets there, and how long it takes',
    level: 2,
    sources: { aos: '§23.2', other: 'Norris (1997), §§1.7–1.10; Levin, Peres & Wilmer, Markov Chains and Mixing Times (2017), Chs. 4, 5, 12' },
    prereqs: ['chain-classification'],
    related: ['mcmc-foundations', 'metropolis-hastings', 'mcmc-diagnostics', 'lln', 'clt'],
    tags: ['stationary distribution', 'coupling', 'ergodic theorem', 'detailed balance', 'spectral gap', 'mixing time'],
    summary: R`A stationary distribution is a probability vector with $\pi P = \pi$: start the chain there and it stays there in law forever. For an irreducible chain, one exists if and only if the chain is positive recurrent; it is then unique and given by $\pi_i = 1/\E_i T_i$. Two limit theorems make $\pi$ operational: the convergence theorem ($\mu P^n\to\pi$ in total variation, for aperiodic chains, proved by coupling) and the ergodic theorem (time averages $\frac1n\sum f(X_k)\to\sum_i\pi_i f(i)$, no aperiodicity needed). Detailed balance gives an easy sufficient condition for stationarity, and for reversible chains the spectral gap of $P$ controls the speed of convergence — the theory behind every MCMC algorithm.`,
    survey: {
      task: R`Find the long-run law $\pi$ of a chain; bound the distance of $\mu P^n$ from $\pi$; justify time averages`,
      assumptions: 'Irreducible and positive recurrent (existence/ergodic theorem); plus aperiodic (convergence); plus reversible (spectral bounds)',
      output: R`$\pi$ solving $\pi P=\pi$; $d(n)=\max_i\|P^n(i,\cdot)-\pi\|_{TV}$; $t_{\text{mix}}$; asymptotic variance $\sigma_f^2$`,
      strengths: 'Exact characterization; coupling gives intuitive and quantitative proofs; detailed balance makes MCMC design easy',
      weaknesses: R`Spectral bounds can be loose by the factor $\log(1/\pi_{\min})$; non-reversible chains need other tools`,
      complexity: R`Solving $\pi(P-I)=0$: $O(|S|^3)$ dense; power iteration $O(|S|^2)$ per step`,
    },
    sections: [
      {
        title: 'Stationary distributions and invariant measures',
        html: R`
<div class="def" data-name="Stationary distribution">
<p>A row vector $\pi$ with $\pi_i\ge0$, $\sum_i\pi_i = 1$ and $\pi P = \pi$ (that is, $\pi_j = \sum_i\pi_ip_{ij}$ for all $j$) is a <strong>stationary</strong> (or invariant, or equilibrium) distribution. A non-negative, not identically zero $\lambda$ with $\lambda P=\lambda$ but possibly infinite total mass is an <strong>invariant measure</strong>.</p>
</div>
<p>If $X_0\sim\pi$ then $X_n\sim\pi P^n = \pi$ for all $n$, and in fact the whole process is stationary: $(X_m, X_{m+1},\dots)$ has the same law as $(X_0,X_1,\dots)$. The equation $\pi_j = \sum_i\pi_i p_{ij}$ is a <em>global balance</em> condition: in equilibrium the probability flowing into $j$ per step equals the probability sitting at $j$ (which is also what flows out, since rows of $P$ sum to one).</p>
<p>Three things can go wrong, and the examples are worth keeping in mind:</p>
<ul>
<li><strong>Non-uniqueness.</strong> If the chain has two closed classes, each supports its own stationary distribution, and every mixture is stationary. Uniqueness needs (essentially) irreducibility.</li>
<li><strong>Non-existence.</strong> Simple symmetric random walk on $\mathbb Z$ has the invariant measure $\lambda_i\equiv1$ but it cannot be normalized; there is no stationary <em>distribution</em>. This is null recurrence.</li>
<li><strong>Non-convergence.</strong> The flip chain $P=\begin{pmatrix}0&1\\1&0\end{pmatrix}$ has unique $\pi = (\tfrac12,\tfrac12)$, yet $\delta_0P^n$ alternates and never converges. This is periodicity.</li>
</ul>
<p>On a finite space $\pi$ is a left eigenvector of $P$ for eigenvalue $1$; $1$ is always an eigenvalue because $P\mathbf 1 = \mathbf 1$. Numerically, solve the linear system $\pi(P-I)=0$ with one equation replaced by $\sum_i\pi_i=1$, or iterate $\mu\mapsto\mu P$ (the power method) when the chain is aperiodic. PageRank is exactly this computation for a random surfer on the web graph, with a small "teleportation" probability added to make the chain irreducible and aperiodic.</p>`,
      },
      {
        title: 'Existence, uniqueness and the return-time formula',
        html: R`
<p>The key construction counts how much time the chain spends in each state during one excursion away from a fixed reference state $i$. For recurrent $i$, define
$$\gamma^i_j = \E_i\sum_{n=0}^{T_i-1}\ind\{X_n = j\},$$
the expected number of visits to $j$ before the first return to $i$. Note $\gamma^i_i = 1$ and $\sum_j\gamma^i_j = \E_iT_i = m_i$.</p>
<div class="thm" data-name="Invariant measure from excursions">
<p>Let $P$ be irreducible and recurrent and fix $i$. Then $\gamma^i$ is an invariant measure with $0<\gamma^i_j<\infty$ for all $j$, and every invariant measure $\lambda$ with $\lambda_i=1$ equals $\gamma^i$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Invariance.</em> Because $X_0 = X_{T_i} = i$ under $\P_i$ (and $T_i<\infty$ a.s.), we may shift the window by one: $\gamma^i_j = \E_i\sum_{n=1}^{T_i}\ind\{X_n=j\} = \sum_{n\ge1}\P_i(X_n=j,\ n\le T_i)$. The event $\{n\le T_i\} = \{T_i\le n-1\}^c$ is determined by $X_0,\dots,X_{n-1}$, so by the Markov property at time $n-1$,
$$\P_i(X_{n-1}=k,\ n\le T_i,\ X_n=j) = \P_i(X_{n-1}=k,\ n-1\le T_i-1)\,p_{kj}.$$
Summing over $k$ and $n$: $\gamma^i_j = \sum_k p_{kj}\sum_{m\ge0}\P_i(X_m=k,\ m\le T_i-1) = \sum_k\gamma^i_kp_{kj}$.</p>
<p><em>Positivity and finiteness.</em> By irreducibility pick $r,s$ with $p^{(r)}_{ij}>0$, $p^{(s)}_{ji}>0$. Invariance iterated gives $\gamma^i_j\ge\gamma^i_ip^{(r)}_{ij}>0$ and $1 = \gamma^i_i\ge\gamma^i_jp^{(s)}_{ji}$, so $\gamma^i_j<\infty$.</p>
<p><em>Uniqueness (sketch).</em> If $\lambda$ is invariant with $\lambda_i=1$, expand $\lambda_j = \sum_k\lambda_kp_{kj}$ by repeatedly splitting off the $k=i$ term; after $n$ expansions $\lambda_j\ge\sum_{m=1}^{n}\P_i(X_m=j,\ T_i\ge m)$, so $\lambda\ge\gamma^i$. Then $\mu = \lambda-\gamma^i$ is a non-negative invariant measure with $\mu_i=0$; since $0 = \mu_i\ge\mu_jp^{(s)}_{ji}$ for every $j$, $\mu\equiv0$.</p>
</details>
<div class="thm" data-name="Existence and uniqueness of the stationary distribution">
<p>For an irreducible chain the following are equivalent: (a) some state is positive recurrent; (b) every state is positive recurrent; (c) a stationary distribution exists. When they hold, the stationary distribution is unique and
$$\pi_i = \frac{1}{m_i} = \frac{1}{\E_iT_i}\qquad\text{for every }i.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(a)$\Rightarrow$(c): if $m_i<\infty$, then $\gamma^i$ has finite total mass $m_i$, so $\pi = \gamma^i/m_i$ is a stationary distribution, with $\pi_i = \gamma^i_i/m_i = 1/m_i$.</p>
<p>(c)$\Rightarrow$(b): let $\pi$ be stationary. If the chain were transient, $p^{(n)}_{ji}\to0$ for all $j$ (the series $\sum_n p^{(n)}_{ji}$ converges), and dominated convergence in $\pi_i = \sum_j\pi_jp^{(n)}_{ji}$ would give $\pi_i = 0$ for all $i$ — impossible. So the chain is recurrent. Every $\pi_k>0$ (by irreducibility, as above), so $\lambda = \pi/\pi_k$ is invariant with $\lambda_k = 1$; by uniqueness $\lambda = \gamma^k$, whence $m_k = \sum_j\gamma^k_j = 1/\pi_k<\infty$. This holds for every $k$, and also proves uniqueness and the formula. (b)$\Rightarrow$(a) is trivial.</p>
</details>
<div class="note">
<p>$\pi_i = 1/m_i$ is a renewal statement: if visits to $i$ are separated by gaps of mean $m_i$, then in the long run a fraction $1/m_i$ of time is spent at $i$. The excursion measure $\gamma^i$ refines this: within a typical excursion from $i$, the expected time at $j$ is $\pi_j/\pi_i$.</p>
</div>
<div class="ex" data-name="Ehrenfest urn and the arrow of time">
<p>$N$ molecules are split between two chambers; at each step a uniformly chosen molecule switches chambers. With $X_n$ the count in chamber 1, $p_{k,k-1} = k/N$ and $p_{k,k+1} = (N-k)/N$. The binomial law $\pi_k = \binom Nk2^{-N}$ is stationary (check detailed balance below). So the expected time to return to the state "all molecules in chamber 1" is $1/\pi_N = 2^N$ — for $N = 100$ that is about $10^{30}$ steps. Boltzmann's answer to Zermelo's recurrence objection is exactly this: the chain <em>is</em> recurrent, but the return times to low-entropy states are astronomically long. (The chain has period 2, which matters for convergence but not for $\pi$.)</p>
</div>`,
      },
      {
        title: 'The convergence theorem, by coupling',
        html: R`
<p>Total variation distance between distributions on $S$ is $\|\mu-\nu\|_{TV} = \sup_{A\subseteq S}|\mu(A)-\nu(A)| = \tfrac12\sum_j|\mu_j-\nu_j|$.</p>
<div class="thm" data-name="Convergence to equilibrium">
<p>Let $P$ be irreducible, aperiodic, and positive recurrent with stationary distribution $\pi$. Then for every initial distribution $\mu$,
$$\|\mu P^n - \pi\|_{TV}\to0\qquad (n\to\infty);$$
in particular $p^{(n)}_{ij}\to\pi_j$ for all $i,j$.</p>
</div>
<details class="proof"><summary>Proof (Doeblin's coupling)</summary>
<p>Run two chains: $X\sim\text{Markov}(\mu,P)$ and $Y\sim\text{Markov}(\pi,P)$, independently. The pair $W_n = (X_n,Y_n)$ is a Markov chain on $S\times S$ with transitions $p_{ik}p_{jl}$.</p>
<p><em>$W$ is irreducible.</em> By aperiodicity and irreducibility, $p^{(n)}_{ik}>0$ and $p^{(n)}_{jl}>0$ for all $n$ large enough (the lemma on the <a href="#/t/chain-classification">classification</a> page), so $(i,j)\to(k,l)$. This is the one place aperiodicity is used: for the flip chain, $X$ and $Y$ started on different states swap in lockstep and never meet.</p>
<p><em>$W$ is positive recurrent.</em> It has the stationary distribution $\pi_i\pi_j$, so by the existence theorem it is positive recurrent, hence recurrent.</p>
<p><em>Coupling.</em> Fix $b\in S$ and let $T = \inf\{n: X_n=Y_n=b\}$; by recurrence of $W$, $T<\infty$ a.s. Define $Z_n = X_n$ for $n< T$ and $Z_n=Y_n$ for $n\ge T$. By the strong Markov property at $T$ (both coordinates are at $b$, and $X$ and $Y$ have the same transition law from there), $Z\sim\text{Markov}(\mu,P)$. Hence
$$|\P(X_n=j)-\pi_j| = |\P(Z_n = j)-\P(Y_n=j)| \le \P(Z_n\ne Y_n) \le \P(T>n)\to0,$$
and the same argument applied to $\P(Z_n\in A)-\P(Y_n\in A)$ gives $\|\mu P^n-\pi\|_{TV}\le\P(T>n)$.</p>
</details>
<p>The final inequality is the <strong>coupling inequality</strong>: for <em>any</em> coupling of two copies of the chain that stay together after meeting,
$$\|\mu P^n-\nu P^n\|_{TV}\le\P(T_{\text{couple}}>n).$$
The independent coupling used in the proof is crude; cleverer couplings (driving both chains with shared randomness) give sharp quantitative mixing bounds, and the second lab on this page compares a coupling bound with the exact distance.</p>
<div class="note">
<p>Periodic chains still "converge" in a weaker sense: the Cesàro averages $\frac1n\sum_{k< n}\mu P^k\to\pi$, and $P^d$ restricted to each cyclic class converges. Only the raw sequence $\mu P^n$ oscillates. In the lab, pick the "near-cycle" preset and then the "deterministic cycle" preset: two eigenvalues move out to the complex cube roots of unity on the unit circle, and $d(n)$ stops decaying.</p>
</div>`,
        lab: 'stationary-distributions-convergence',
      },
      {
        title: 'The ergodic theorem and the Markov chain CLT',
        html: R`
<p>Statistical uses of Markov chains — above all <a href="#/t/mcmc-foundations">MCMC</a> — rest on time averages along a single path rather than on the law of $X_n$ at a fixed time.</p>
<div class="thm" data-name="Ergodic theorem (strong law for Markov chains)">
<p>Let $P$ be irreducible, with any initial distribution. Then for each state $i$, the fraction of time spent at $i$ satisfies
$$\frac{1}{n}\sum_{k=0}^{n-1}\ind\{X_k=i\}\asto\frac{1}{m_i}$$
(interpreted as $0$ if $m_i=\infty$). If the chain is positive recurrent with stationary distribution $\pi$ and $f:S\to\R$ is bounded (or merely $\sum_i\pi_i|f(i)|<\infty$), then
$$\frac{1}{n}\sum_{k=0}^{n-1}f(X_k)\asto\pi f = \sum_i\pi_if(i).$$
No aperiodicity is required.</p>
</div>
<details class="proof"><summary>Proof sketch (regeneration)</summary>
<p>If the chain is transient, visits to $i$ are finitely many and both sides are zero. Otherwise the chain hits $i$ in finite time; by the strong Markov property, the successive excursions from $i$ are IID. Let $L_r$ be the length of the $r$-th excursion ($\E L_r = m_i$) and $V_n$ the number of visits to $i$ before time $n$. Up to the (a.s. finite) time of the first visit, $L_1+\dots+L_{V_n-1}\le n \le L_1+\dots+L_{V_n}$; dividing by $V_n\to\infty$ and applying the strong <a href="#/t/lln">law of large numbers</a> to the $L_r$ gives $n/V_n\to m_i$.</p>
<p>For $f\ge0$, let $Y_r = \sum_{k\text{ in excursion } r}f(X_k)$. The $Y_r$ are IID with mean $\sum_j\gamma^i_jf(j) = m_i\,\pi f$. Then $\frac1n\sum_{k< n}f(X_k)$ is squeezed between $\frac{1}{n}\sum_{r< V_n}Y_r$ and $\frac{1}{n}\sum_{r\le V_n}Y_r$ (up to the initial segment), and both tend to $\frac{1}{m_i}\cdot m_i\pi f$. Split a general $f$ into positive and negative parts.</p>
</details>
<p>The regeneration argument also delivers a central limit theorem, with the twist that consecutive terms are correlated.</p>
<div class="thm" data-name="Markov chain CLT (statement)">
<p>Suppose the chain is positive recurrent with stationary $\pi$, $\pi f^2<\infty$, and the chain mixes fast enough — for example it is geometrically ergodic and $\pi|f|^{2+\delta}<\infty$ for some $\delta>0$, or it is reversible and the series below converges (Kipnis–Varadhan). Then
$$\sqrt n\Big(\frac1n\sum_{k< n}f(X_k)-\pi f\Big)\dto\N(0,\sigma_f^2),\qquad \sigma_f^2 = \Var_\pi f(X_0) + 2\sum_{k\ge1}\Cov_\pi\big(f(X_0),f(X_k)\big).$$</p>
</div>
<p>The ratio $n\Var_\pi f/\sigma_f^2$ is the <em>effective sample size</em>. Positive autocorrelation inflates $\sigma_f^2$ above the IID variance, which is why naive standard errors from correlated MCMC output are too small, and why <a href="#/t/mcmc-diagnostics">batch means and spectral estimators</a> exist. Compare the <a href="#/t/block-bootstrap">block bootstrap</a>, which faces the same problem from the resampling side.</p>`,
      },
      {
        title: 'Detailed balance and reversibility',
        html: R`
<p>Global balance $\pi P=\pi$ is a system of $|S|$ coupled equations. A much stronger but far easier condition asks that the flow balance across every single edge.</p>
<div class="thm" data-kind="Proposition" data-name="Detailed balance implies stationarity">
<p>If a probability vector $\pi$ satisfies $\pi_ip_{ij} = \pi_jp_{ji}$ for all $i,j$, then $\pi$ is stationary for $P$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\sum_i\pi_ip_{ij} = \sum_i\pi_jp_{ji} = \pi_j\sum_ip_{ji} = \pi_j$.</p>
</details>
<div class="def" data-name="Reversible chain">
<p>$P$ is <strong>reversible</strong> with respect to $\pi$ if detailed balance holds. Equivalently, if $X_0\sim\pi$, then for every $n$ the reversed path $(X_n,X_{n-1},\dots,X_0)$ has the same law as $(X_0,\dots,X_n)$: a movie of the stationary chain looks statistically identical played backwards.</p>
</div>
<p>(The equivalence: the time-reversal of a stationary chain is Markov with transitions $\hat p_{ij} = \pi_jp_{ji}/\pi_i$, and $\hat P=P$ is detailed balance.) Kolmogorov's criterion characterizes reversibility without knowing $\pi$: an irreducible $P$ is reversible iff $p_{i_0i_1}p_{i_1i_2}\cdots p_{i_ki_0} = p_{i_0i_k}\cdots p_{i_2i_1}p_{i_1i_0}$ for every cycle — no net circulation around any loop.</p>
<div class="ex" data-name="Three reversible families">
<ul>
<li><strong>Birth–death chains</strong> (nearest-neighbour moves on $\{0,\dots,N\}$ or $\mathbb Z_{\ge0}$) are always reversible: there is only one edge between $k$ and $k+1$, so the net flow across it must vanish in equilibrium. Detailed balance gives $\pi_{k+1}/\pi_k = p_{k,k+1}/p_{k+1,k}$. For Ehrenfest, $\pi_{k+1}/\pi_k = (N-k)/(k+1)$, which is the binomial ratio.</li>
<li><strong>Random walk on a graph</strong> — move to a uniformly chosen neighbour — has $\pi_v = \deg(v)/2|E|$, since $\pi_u\cdot\frac{1}{\deg u} = \frac{1}{2|E|}$ is symmetric in $u,v$.</li>
<li><strong>Metropolis–Hastings</strong> chains are <em>constructed</em> to satisfy detailed balance with respect to an arbitrary target $\pi$ known up to a constant; see <a href="#/t/metropolis-hastings">Metropolis–Hastings</a>. This is why detailed balance, a sufficient condition, is the workhorse of MCMC design.</li>
</ul>
</div>
<div class="warn" data-name="Stationary does not mean reversible">
<p>A chain that cycles $1\to2\to3\to1$ with probability $0.9$ (and steps backwards with probability $0.1$) has uniform $\pi$ but a net clockwise current: $\pi_1p_{12} = 0.3$ while $\pi_2p_{21} = 0.0\overline{3}$. Non-reversible chains are common and sometimes mix faster (lifted chains, non-reversible MCMC), but the clean spectral theory below needs reversibility.</p>
</div>`,
      },
      {
        title: 'Spectral gap and mixing time',
        html: R`
<p>The convergence theorem is qualitative. To know <em>how many</em> steps suffice we measure</p>
$$d(n) = \max_{i\in S}\|P^n(i,\cdot)-\pi\|_{TV},\qquad t_{\text{mix}}(\varepsilon) = \min\{n: d(n)\le\varepsilon\},$$
<p>with $t_{\text{mix}} = t_{\text{mix}}(1/4)$ by convention. $d$ is non-increasing, and $d(kn)\le(2d(n))^k$, so after $t_{\text{mix}}$ the distance decays geometrically: $t_{\text{mix}}(\varepsilon)\le\lceil\log_2(1/\varepsilon)\rceil\,t_{\text{mix}}$.</p>
<p>For a reversible $P$ on a finite space, $P$ is self-adjoint in $L^2(\pi)$ with inner product $\langle f,g\rangle_\pi = \sum_i\pi_if(i)g(i)$, because $\langle Pf,g\rangle_\pi = \sum_{i,j}\pi_ip_{ij}f(j)g(i)$ is symmetric under detailed balance. So $P$ has real eigenvalues $1=\lambda_1\ge\lambda_2\ge\dots\ge\lambda_{|S|}\ge-1$ with a $\pi$-orthonormal eigenbasis. Irreducibility makes $\lambda_1 = 1$ simple; aperiodicity excludes $\lambda_{|S|}=-1$. Define the <strong>absolute spectral gap</strong> $\gamma_* = 1-\lambda_*$ with $\lambda_* = \max(|\lambda_2|,|\lambda_{|S|}|)$, and the <strong>relaxation time</strong> $t_{\text{rel}} = 1/\gamma_*$.</p>
<div class="thm" data-name="Spectral bounds on mixing">
<p>Let $P$ be reversible, irreducible and aperiodic on a finite space. Then for every $i$ and $n$,
$$\|P^n(i,\cdot)-\pi\|_{TV}\le\frac12\sqrt{\frac{1-\pi_i}{\pi_i}}\;\lambda_*^n.$$
Conversely, for <em>any</em> finite chain and any eigenvalue $\lambda\ne1$ of $P$, $d(n)\ge\tfrac12|\lambda|^n$. Consequently, for reversible chains,
$$t_{\text{mix}}(\varepsilon)\ \le\ t_{\text{rel}}\,\log\frac{1}{2\varepsilon\sqrt{\pi_{\min}}},\qquad t_{\text{mix}}(\varepsilon)\ \ge\ (t_{\text{rel}}-1)\log\frac{1}{2\varepsilon}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Upper bound.</em> Let $g = \ind_{\{i\}}/\pi_i$, the density of $\delta_i$ with respect to $\pi$. Reversibility gives $p^{(n)}_{ij}/\pi_j = p^{(n)}_{ji}/\pi_i = (P^ng)(j)$, so the density of $P^n(i,\cdot)$ is $P^ng$. Then
$$2\|P^n(i,\cdot)-\pi\|_{TV} = \sum_j\pi_j|(P^ng)(j)-1| \le \|P^ng-\mathbf 1\|_{L^2(\pi)} = \|P^n(g-\mathbf 1)\|_{L^2(\pi)}$$
by Cauchy–Schwarz and $P\mathbf1=\mathbf1$. Since $\langle g-\mathbf1,\mathbf1\rangle_\pi = 0$, $g-\mathbf 1$ lies in the span of eigenvectors $2,\dots,|S|$, on which $P^n$ has operator norm $\lambda_*^n$. Finally $\|g-\mathbf1\|^2_{L^2(\pi)} = \pi g^2-1 = 1/\pi_i-1$.</p>
<p><em>Lower bound.</em> Let $Pf = \lambda f$, $\lambda\ne1$ (possibly complex). Then $\pi f = \pi Pf = \lambda\,\pi f$, so $\pi f=0$, and $\lambda^nf(i) = (P^nf)(i) = \sum_j(p^{(n)}_{ij}-\pi_j)f(j)$, whose modulus is at most $2\|f\|_\infty\|P^n(i,\cdot)-\pi\|_{TV}$. Choose $i$ with $|f(i)| = \|f\|_\infty$.</p>
<p>The upper mixing-time bound follows by using $\lambda_*^n\le e^{-n\gamma_*}$ and $(1-\pi_i)/\pi_i\le1/\pi_{\min}$. The lower one follows from $\tfrac12\lambda_*^n\le\varepsilon$ at $n = t_{\text{mix}}(\varepsilon)$ together with $\log(1/\lambda_*)\le(1-\lambda_*)/\lambda_* = 1/(t_{\text{rel}}-1)$.</p>
</details>
<div class="ex" data-name="Lazy random walk on a cycle">
<p>On $\mathbb Z_N$, stay put with probability $a$ and otherwise step $\pm1$ with equal probability. The eigenvectors are the Fourier characters $x\mapsto e^{2\pi\mathrm{i}kx/N}$ with eigenvalues $a+(1-a)\cos(2\pi k/N)$. For $a=\tfrac12$, $\lambda_2 = \tfrac12(1+\cos\frac{2\pi}{N})\approx1-\frac{\pi^2}{N^2}$, so $t_{\text{rel}}\approx N^2/\pi^2$: mixing on a cycle is diffusive, $t_{\text{mix}}\asymp N^2$, because a random walk needs about $N^2$ steps to travel distance $N$. The second lab shows this scaling and compares the exact $d(n)$ with the coupling bound.</p>
</div>
<p>Other routes to $\gamma_*$ are often easier than computing eigenvalues: <em>conductance</em> (Cheeger's inequality $\Phi^2/2\le1-\lambda_2\le2\Phi$, where $\Phi$ measures the worst bottleneck), canonical paths, and coupling. Bottlenecks — multimodal targets separated by low-probability regions — are exactly what make MCMC slow, and are what <a href="#/t/tempering">tempering</a> methods attack.</p>`,
        lab: 'stationary-distributions-coupling',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Slow mixing looks like convergence">
<p>A chain trapped in one mode of a bimodal $\pi$ produces stable-looking averages for a very long time. Convergence diagnostics test necessary, not sufficient, conditions; only multiple dispersed starts or structural knowledge (bottleneck bounds) can reveal metastability.</p>
</div>
<div class="warn" data-name="Second eigenvalue of a non-reversible chain">
<p>For non-reversible $P$ the eigenvalues can be complex and $P$ need not be diagonalizable; $|\lambda_2|^n$ still governs the eventual geometric rate, but transient behaviour can be far worse than the spectrum suggests (non-normal matrices). Use the multiplicative reversibilization $PP^*$ or direct coupling arguments instead.</p>
</div>
<div class="warn" data-name="Burn-in does not fix exploration">
<p>The ergodic theorem holds from any start, so the initial transient contributes $O(1/n)$ bias to time averages; discarding a burn-in reduces that bias but cannot fix a chain that has not explored the space. Conversely, a periodic chain gives perfectly valid time averages even though $X_n$ itself never converges in law.</p>
</div>
<div class="warn" data-name="Null recurrence and empirical frequencies">
<p>For a null recurrent chain, $\frac1n\#\{k< n: X_k=i\}\to0$ for every $i$, even though every state is visited infinitely often. Empirical occupation frequencies from a long simulation look like they converge to something, but they converge to zero, slowly.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`An irreducible finite chain has $\pi = (0.5, 0.3, 0.2)$. Starting from state 3, what is the expected time to return to state 3? What is the expected number of visits to state 1 during one such excursion?`,
        a: R`$m_3 = 1/\pi_3 = 5$ steps. By the excursion construction, the expected number of visits to $j$ before returning to $i$ is $\gamma^i_j = \pi_j/\pi_i$, so $0.5/0.2 = 2.5$ visits to state 1.` },
      { q: R`Where exactly does the coupling proof of the convergence theorem use aperiodicity, and what goes wrong for the flip chain?`,
        a: R`It is used to show that the product chain $(X_n,Y_n)$ is irreducible, via $p^{(n)}_{ik}>0$ for all large $n$. For the flip chain started at $(0,1)$, the product chain alternates between $(0,1)$ and $(1,0)$ and never reaches the diagonal, so the coupling time is infinite and indeed $\delta_0P^n$ does not converge.` },
      { q: R`Show that the stationary distribution of a random walk on a connected undirected graph is proportional to vertex degree. Is the walk necessarily aperiodic?`,
        a: R`Detailed balance with $\pi_v\propto\deg(v)$: $\pi_up_{uv} = \frac{\deg u}{2|E|}\cdot\frac{1}{\deg u} = \frac{1}{2|E|} = \pi_vp_{vu}$. It is aperiodic iff the graph is not bipartite (an odd cycle exists); on a bipartite graph the period is 2, and one uses the lazy walk to obtain convergence.` },
      { q: R`For a reversible chain with $\lambda_* = 0.99$ and $\pi_{\min} = 10^{-6}$, roughly how many steps guarantee $d(n)\le 0.01$ by the spectral bound?`,
        a: R`Need $\frac12\pi_{\min}^{-1/2}\lambda_*^n\le0.01$, i.e. $n\ge\log\big(\frac{1}{2\cdot0.01\cdot10^{-3}}\big)/\log(1/0.99) = \log(5\times10^4)/0.01005\approx10.82/0.01005\approx1077$. The relaxation time is $100$; the $\log(1/\pi_{\min})$ factor accounts for most of the multiplier.` },
      { q: R`Why does the ergodic theorem not need aperiodicity, while the convergence theorem does?`,
        a: R`Time averages average over phases of the cycle: a period-$d$ chain spends asymptotically the right fraction of time in each cyclic class even though at any particular time it is in one specific class. The convergence theorem is about the law at a fixed time $n$, which for a periodic chain depends on $n \bmod d$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'random-walks',
    domain: 'stochastic',
    order: 40,
    title: "Random Walks & Gambler's Ruin",
    subtitle: 'Sums of independent steps: recurrence, ruin, reflection and the arcsine law',
    level: 2,
    sources: { aos: '§23.1', other: 'Feller, Vol. I (1968), Chs. III, XIV; Durrett, Probability: Theory and Examples (2019), §4.2' },
    prereqs: ['markov-chains'],
    related: ['chain-classification', 'martingales', 'brownian-motion', 'clt'],
    tags: ['simple random walk', 'Pólya', 'gambler’s ruin', 'reflection principle', 'ballot theorem', 'arcsine law'],
    summary: R`The simple random walk $S_n = \xi_1+\dots+\xi_n$ with IID $\pm1$ steps is the simplest non-trivial Markov chain and a laboratory for nearly every idea in stochastic processes. Pólya's theorem says it returns to the origin with probability one in dimensions 1 and 2 but escapes forever with positive probability in dimension 3 and above. First-step analysis solves gambler's ruin exactly. Path counting with the reflection principle yields the ballot theorem and the distribution of the maximum, and the arcsine law shows that "fair" fluctuations are wildly lopsided: one player is usually ahead for most of the game.`,
    survey: {
      task: 'Analyse sums of IID steps: return to origin, hitting/ruin probabilities, maxima and occupation times',
      assumptions: R`IID increments (simple walk: $\pm1$ with probabilities $p, q=1-p$); lattice $\mathbb Z^d$`,
      output: R`Return probabilities, $h_i = \frac{1-(q/p)^i}{1-(q/p)^N}$, expected duration, laws of $\max_{k\le n}S_k$ and of the last zero`,
      strengths: 'Exact combinatorial answers; prototype for martingale and Brownian-motion arguments',
      weaknesses: R`Exact formulas are special to nearest-neighbour steps; general steps need renewal / Wiener–Hopf tools`,
      complexity: R`Closed forms; $O(N)$ for ruin quantities on $\{0,\dots,N\}$`,
    },
    sections: [
      {
        title: 'Simple random walk and return to the origin',
        html: R`
<p>Let $\xi_1,\xi_2,\dots$ be IID with $\P(\xi=+1)=p$, $\P(\xi=-1)=q=1-p$, and $S_n = S_0+\xi_1+\cdots+\xi_n$. Then $(S_n)$ is a Markov chain on $\mathbb Z$ with $p_{i,i+1}=p$, $p_{i,i-1}=q$. It is irreducible with period 2 (returns to the start take an even number of steps).</p>
<p>Returning to $0$ in $2n$ steps requires exactly $n$ up-steps among $2n$, so
$$u_{2n} := p^{(2n)}_{00} = \binom{2n}{n}p^nq^n\sim\frac{(4pq)^n}{\sqrt{\pi n}}$$
by Stirling's formula $\binom{2n}{n}\sim 4^n/\sqrt{\pi n}$. Since $4pq\le1$ with equality iff $p=\tfrac12$:</p>
<ul>
<li>$p=\tfrac12$: $u_{2n}\sim1/\sqrt{\pi n}$, the series diverges, and the walk is <strong>recurrent</strong>.</li>
<li>$p\ne\tfrac12$: the series converges geometrically, and the walk is <strong>transient</strong> (as the strong law $S_n/n\to p-q\neq 0$ also shows).</li>
</ul>
<p>The symmetric walk is <em>null</em> recurrent: the counting measure is invariant but not normalizable, so $\E_0T_0 = \infty$. Indeed $\P_0(T_0>2n) = u_{2n}\sim1/\sqrt{\pi n}$ (proved below), and $\E T_0 = \sum_n\P(T_0>n) = \infty$. The expected number of returns by time $2n$ is $\sum_{k=1}^n u_{2k}\sim2\sqrt{n/\pi}$ — infinitely many returns in the long run, but ever more widely spaced.</p>
<div class="note">
<p>Why should the answer be $n^{-1/2}$? After $2n$ steps, $S_{2n}$ has standard deviation $\sqrt{2n}$ and, by the local <a href="#/t/clt">CLT</a>, is spread nearly uniformly over the $O(\sqrt n)$ even sites near the origin. The chance of being at one particular site is therefore of order $n^{-1/2}$.</p>
</div>`,
      },
      {
        title: "Pólya's theorem: recurrence depends on dimension",
        html: R`
<p>Simple random walk on $\mathbb Z^d$ moves to one of the $2d$ nearest neighbours uniformly at random.</p>
<div class="thm" data-name="Pólya (1921)">
<p>Simple random walk on $\mathbb Z^d$ is recurrent for $d=1,2$ and transient for $d\ge3$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the recurrence criterion, it suffices to decide whether $\sum_n p^{(2n)}_{00}$ diverges.</p>
<p><em>$d=1$:</em> done above, $p^{(2n)}_{00}\sim(\pi n)^{-1/2}$.</p>
<p><em>$d=2$:</em> rotate coordinates. If $(X_n,Y_n)$ is the walk, set $U_n = X_n+Y_n$ and $V_n = X_n-Y_n$. Each of the four steps $(\pm1,0),(0,\pm1)$ changes $(U,V)$ by one of $(\pm1,\pm1)$, each with probability $\tfrac14$ — that is, $U$ and $V$ are <em>independent</em> one-dimensional simple random walks. The walk is at the origin iff $U_n=V_n=0$, so
$$p^{(2n)}_{00} = \Big(\binom{2n}{n}4^{-n}\Big)^2\sim\frac{1}{\pi n},$$
and the harmonic series diverges.</p>
<p><em>$d=3$:</em> a return in $2n$ steps uses $i,j,k$ steps in the positive direction along the three axes and the same numbers back, $i+j+k=n$. Counting arrangements,
$$p^{(2n)}_{00} = \sum_{i+j+k=n}\frac{(2n)!}{(i!\,j!\,k!)^2}6^{-2n} = \binom{2n}{n}2^{-2n}\sum_{i+j+k=n}\Big(\frac{n!}{i!\,j!\,k!}3^{-n}\Big)^2.$$
The terms $\frac{n!}{i!j!k!}3^{-n}$ are the probabilities of a multinomial distribution, so they sum to 1; hence the sum of their squares is at most their maximum, which is attained near $i=j=k=n/3$ and is $O(1/n)$ by Stirling. So $p^{(2n)}_{00} = O(n^{-1/2})\cdot O(n^{-1}) = O(n^{-3/2})$, which is summable.</p>
<p><em>$d\ge4$:</em> the projection onto the first three coordinates, observed at the times it moves, is a 3-D simple random walk, which visits the origin of $\mathbb Z^3$ only finitely often; hence the full walk visits the origin of $\mathbb Z^d$ only finitely often. (Alternatively, the local CLT gives $p^{(2n)}_{00}\asymp n^{-d/2}$.)</p>
</details>
<p>The return probability in three dimensions is $f\approx0.3405$ (Watson's integral): a 3-D walker returns at all with only about one chance in three, and the expected number of visits to the origin (counting time 0) is $1/(1-f)\approx1.516$. Kakutani's quip: "A drunk man will find his way home, but a drunk bird may get lost forever."</p>
<div class="note">
<p>The heuristic behind the dimension threshold: after $n$ steps the walk is spread over a ball of radius $\sqrt n$, containing about $n^{d/2}$ sites, so the chance of being at the origin is about $n^{-d/2}$. Summable iff $d/2>1$. The same scaling governs recurrence of <a href="#/t/brownian-motion">Brownian motion</a> (neighbourhood-recurrent in $d\le2$, transient in $d\ge3$) and of electrical networks (Pólya's theorem is equivalent to the infinite effective resistance from the origin to infinity in the $\mathbb Z^2$ lattice).</p>
</div>`,
        lab: 'random-walks-return',
      },
      {
        title: "Gambler's ruin",
        html: R`
<p>A gambler with fortune $i$ bets one unit at a time, winning with probability $p$, until she reaches a target $N$ or goes broke. Her fortune is a simple random walk absorbed at $0$ and $N$. Let $h_i = \P_i(\text{hit }N\text{ before }0)$ and $D_i = \E_i[\text{duration}]$.</p>
<div class="thm" data-name="Gambler's ruin">
<p>With $r = q/p$,
$$h_i = \begin{cases}\dfrac{1-r^i}{1-r^N}, & p\ne\tfrac12,\\[2mm] \dfrac iN, & p=\tfrac12,\end{cases}\qquad D_i = \begin{cases}\dfrac{i - Nh_i}{q-p}, & p\ne\tfrac12,\\[2mm] i(N-i), & p=\tfrac12.\end{cases}$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>First-step analysis gives $h_i = ph_{i+1}+qh_{i-1}$ for $0< i< N$, with $h_0=0$, $h_N=1$. Writing $h_i = ph_i+qh_i$ and rearranging, $p(h_{i+1}-h_i) = q(h_i-h_{i-1})$. So the increments $\Delta_i = h_{i+1}-h_i$ satisfy $\Delta_i = r\Delta_{i-1} = r^i\Delta_0$, and $h_i = \Delta_0\sum_{k< i}r^k$. The boundary condition $h_N=1$ fixes $\Delta_0 = 1/\sum_{k< N}r^k$; summing the geometric series gives the formula (and $i/N$ when $r=1$).</p>
<p>For the duration, $D_i = 1+pD_{i+1}+qD_{i-1}$, $D_0=D_N=0$. When $p=\tfrac12$ the second difference of $D$ is constant ($-2$), so $D$ is the quadratic vanishing at $0$ and $N$: $D_i = i(N-i)$. When $p\ne\tfrac12$ a cleaner route is a martingale: $M_n = S_n-n(p-q)$ is a martingale with bounded increments and the duration $\tau$ has finite mean (it is dominated by a geometric number of blocks of $N$ steps), so optional stopping gives $\E_iS_\tau - (p-q)\E_i\tau = i$. Since $\E_iS_\tau = Nh_i$, $D_i = (Nh_i-i)/(p-q)$.</p>
</details>
<p>The ruin formula can also be read off from the martingale $r^{S_n} = (q/p)^{S_n}$: optional stopping gives $r^i = h_ir^N+(1-h_i)$. See <a href="#/t/martingales">martingales</a> for the general machinery.</p>
<div class="ex" data-name="Timid play in a subfair game">
<p>Betting one unit on red at American roulette has $p = 18/38$, so $r = 10/9$. Starting with 10 units and aiming for 20: $h_{10} = \frac{1-r^{10}}{1-r^{20}} = \frac{1}{1+r^{10}} = \frac{1}{1+2.868}\approx0.259$, after about $(10-20\cdot0.259)/(2/38)\approx92$ bets on average. Starting with 100 units and aiming for 200: $h_{100} = 1/(1+r^{100})\approx 2.7\times10^{-5}$. A tiny per-bet edge compounds over the many bets of timid play; in a subfair game the gambler maximizes her chance of doubling by betting <em>boldly</em> (Dubins–Savage).</p>
</div>
<p>Letting $N\to\infty$: against an infinitely rich adversary, ruin is certain when $p\le\tfrac12$, while for $p>\tfrac12$ the ruin probability is $r^i = (q/p)^i$. Even a fair game ($p=\tfrac12$) ruins the gambler with probability one — yet the expected duration is $\lim_N i(N-i) = \infty$. This is null recurrence in disguise.</p>`,
        lab: 'random-walks-ruin',
      },
      {
        title: 'The reflection principle and the ballot theorem',
        html: R`
<p>For the symmetric walk every path of length $n$ has probability $2^{-n}$, so probabilities are path counts. Let $N_n(a,b)$ be the number of $n$-step paths from $a$ to $b$; this is $\binom{n}{(n+b-a)/2}$ when $n+b-a$ is even and $|b-a|\le n$, and zero otherwise.</p>
<div class="thm" data-kind="Lemma" data-name="Reflection principle">
<p>For $a,b>0$, the number of $n$-step paths from $a$ to $b$ that touch or cross $0$ equals $N_n(-a,b)$, the number of all paths from $-a$ to $b$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Given a path from $a$ to $b$ that hits $0$, let $k$ be its first hitting time of $0$ and reflect the segment before $k$ in the time axis. This produces a path from $-a$ to $b$. Conversely, any path from $-a$ to $b$ must cross $0$; reflecting its segment before the first hit recovers a path from $a$ to $b$ that touches $0$. The two maps are inverse bijections.</p>
</details>
<div class="thm" data-name="Ballot theorem (Bertrand, 1887)">
<p>In an election, candidate A receives $a$ votes and B receives $b< a$ votes, counted in uniformly random order. The probability that A is strictly ahead throughout the count is $\dfrac{a-b}{a+b}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $n=a+b$; counts are paths from $0$ to $a-b$, all $\binom na$ equally likely. A good path must start with an up-step and then go from $1$ to $a-b$ in $n-1$ steps without touching $0$. By reflection the number of such paths is
$$N_{n-1}(1,a-b)-N_{n-1}(-1,a-b) = \binom{n-1}{a-1}-\binom{n-1}{a} = \binom na\Big(\frac an-\frac bn\Big),$$
using $\binom{n-1}{a-1} = \frac an\binom na$ and $\binom{n-1}{a} = \frac bn\binom na$. Divide by $\binom na$.</p>
</details>
<p>Two consequences for the symmetric walk from $0$:</p>
<ul>
<li><strong>Maximum.</strong> With $M_n = \max_{k\le n}S_k$ and $a\ge1$: paths with $M_n\ge a$ and $S_n< a$ are in bijection (reflect after the first hit of $a$) with paths having $S_n>a$. Hence $\P(M_n\ge a) = \P(S_n\ge a)+\P(S_n>a)$, about twice the tail of $S_n$. Its Brownian limit is $\P(\max_{t\le1}B_t\ge x) = 2\P(B_1\ge x)$.</li>
<li><strong>No return.</strong> $\P(S_1\ne0,\dots,S_{2n}\ne0) = u_{2n}$. By symmetry this is twice the probability of staying strictly positive. A strictly positive path starts with an up-step and ends at some $2r\ge2$; by reflection the count is $\sum_{r\ge1}\big[N_{2n-1}(1,2r)-N_{2n-1}(-1,2r)\big]$, which telescopes to $N_{2n-1}(0,1) = \binom{2n-1}{n} = \tfrac12\binom{2n}{n}$. Thus $\P(T_0>2n) = u_{2n}\sim(\pi n)^{-1/2}$, and the first-return law is $\P(T_0 = 2n) = u_{2n-2}-u_{2n} = \frac{u_{2n}}{2n-1}$.</li>
</ul>`,
      },
      {
        title: 'The arcsine laws',
        html: R`
<p>In a fair game, how often does the lead change? Intuition says each player should be ahead about half the time. The truth is the opposite.</p>
<div class="thm" data-name="Arcsine law for the last zero">
<p>Let $L_{2n} = \max\{k\le2n: S_k=0\}$ be the time of the last visit to $0$ up to time $2n$. Then
$$\P(L_{2n}=2k) = u_{2k}\,u_{2n-2k},\qquad k=0,\dots,n,$$
and $\P(L_{2n}\le 2nx)\to\frac2\pi\arcsin\sqrt x$ for $0\le x\le1$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\{L_{2n}=2k\} = \{S_{2k}=0\}\cap\{S_j\ne0\text{ for }2k< j\le2n\}$. By the Markov property at time $2k$ its probability is $u_{2k}\cdot\P_0(S_1\ne0,\dots,S_{2n-2k}\ne0) = u_{2k}u_{2n-2k}$ by the "no return" identity. For the limit, $u_{2k}u_{2n-2k}\approx\frac{1}{\pi\sqrt{k(n-k)}} = \frac1n\cdot\frac{1}{\pi\sqrt{x(1-x)}}$ with $x=k/n$, a Riemann sum for the arcsine density, whose CDF is $\frac2\pi\arcsin\sqrt x$.</p>
</details>
<div class="thm" data-name="Arcsine law for occupation time (Chung–Feller; Lévy)">
<p>Let $\Pi_{2n}$ be the number of steps $j\le2n$ during which the path lies above the axis (i.e. $S_{j-1}>0$ or $S_j>0$). Then $\P(\Pi_{2n}=2k) = u_{2k}u_{2n-2k}$ as well, so the fraction of time the walk is positive also converges in law to the arcsine distribution.</p>
</div>
<p>The arcsine density $\frac{1}{\pi\sqrt{x(1-x)}}$ is U-shaped: values near $0$ and $1$ are the most likely, and $\tfrac12$ is the <em>least</em> likely. For instance $\P(\text{fraction positive}\le0.1) \approx \frac2\pi\arcsin\sqrt{0.1}\approx0.205$, so with probability about $0.41$ one of two players in a fair coin-tossing game leads for more than 90% of the time. Long leads are not evidence of skill. The lab below lets you check this.</p>`,
        lab: 'random-walks-arcsine',
      },
      {
        title: 'Pitfalls and connections',
        html: R`
<div class="warn" data-name="The gambler's fallacy, in reverse">
<p>Recurrence guarantees a return to the break-even point, but the expected waiting time is infinite and the typical excursion is enormous. "I will surely get back to even" is true and useless: by the time it happens, the bankroll or the patience is likely gone, and in any subfair game recurrence fails anyway.</p>
</div>
<div class="warn" data-name="Simulation hides null recurrence">
<p>Every simulated return time is finite, so the sample mean of return times is finite — but it does not stabilize as you collect more; it grows with the simulation length. Heavy tails with infinite mean are a recurring theme (compare <a href="#/t/evt-intro">extreme value theory</a>).</p>
</div>
<div class="warn" data-name="Periodicity in counting">
<p>The simple walk has period 2: $p^{(n)}_{00}=0$ for odd $n$. Asymptotic formulas such as $p^{(2n)}_{00}\sim(\pi n)^{-1/2}$ are along even times only; forgetting this introduces factors of 2.</p>
</div>
<p>Connections:</p>
<ul>
<li><strong>Brownian motion.</strong> Donsker's invariance principle: $S_{\lfloor nt\rfloor}/\sqrt n\Rightarrow B_t$. The reflection principle, the arcsine laws and ruin probabilities all have continuum analogues, often with cleaner formulas — see <a href="#/t/brownian-motion">Brownian motion</a>.</li>
<li><strong>Martingales.</strong> $S_n$ (fair case), $S_n^2-n$, and $(q/p)^{S_n}$ are martingales; optional stopping reproduces gambler's ruin in two lines — see <a href="#/t/martingales">martingales</a>.</li>
<li><strong>Sequential testing.</strong> The log-likelihood ratio in Wald's sequential probability ratio test is a random walk, and its error probabilities are ruin probabilities.</li>
<li><strong>Queues and branching.</strong> The queue length of a single-server <a href="#/t/queueing">queue</a> and the exploration process of a <a href="#/t/branching-processes">branching process</a> are random walks reflected or stopped at 0.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Why does the rotation $U=X+Y$, $V=X-Y$ make the 2-D walk easy, and why does no such trick work in 3-D?`,
        a: R`In 2-D the four steps map bijectively to the four sign patterns $(\pm1,\pm1)$ of $(\Delta U,\Delta V)$, each with probability $\frac14$, so $U$ and $V$ are independent 1-D walks and $p^{(2n)}_{00} = u_{2n}^2$. In 3-D there are 6 steps, but three independent $\pm1$ coordinates would need $8$ equally likely sign patterns, so no linear change of coordinates turns the walk into independent 1-D walks; one bounds the multinomial sum instead.` },
      { q: R`In a fair game ($p=\frac12$) with $i=30$ and $N=100$, what is the probability of reaching 100, and the expected number of bets?`,
        a: R`$h_{30} = 30/100 = 0.3$, and $D_{30} = 30\cdot70 = 2100$ bets. The probability is the one that makes the stopped fortune a fair game: $\E S_\tau = 100\cdot0.3 = 30$.` },
      { q: R`Votes are 6 for A and 4 for B. What is the probability A is strictly ahead throughout the count? Verify it with the reflection count.`,
        a: R`$(6-4)/(6+4) = 0.2$. Directly: good paths number $\binom{9}{5}-\binom96 = 126-84 = 42$ out of $\binom{10}{6} = 210$ total, and $42/210 = 0.2$.` },
      { q: R`For the symmetric walk over $2n = 1000$ steps, which is more likely: that the last visit to 0 happens at one of the times $0,2,\dots,10$, or at one of the times $496,\dots,504$?`,
        a: R`The first. $\P(L_{1000}=2k) = u_{2k}u_{1000-2k}$ is U-shaped in $k$. For $k=0,\dots,5$ the terms are about $u_{2k}\cdot u_{1000}$ with $u_0+u_2+\dots+u_{10} = 1+0.5+0.375+0.3125+0.273+0.246\approx2.71$ and $u_{1000}\approx(500\pi)^{-1/2}\approx0.025$, giving about $0.068$. Near the middle, five terms each $\approx u_{500}^2\approx(0.0357)^2\approx0.0013$ sum to about $0.006$ — more than ten times smaller.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'branching-processes',
    domain: 'stochastic',
    order: 50,
    title: 'Branching Processes',
    subtitle: 'Galton–Watson trees, generating functions, and the probability of extinction',
    level: 2,
    sources: { other: 'Athreya & Ney, Branching Processes (1972); Grimmett & Stirzaker (2001), §5.4; Lyons & Peres, Probability on Trees and Networks (2016), Ch. 5' },
    prereqs: ['markov-chains', 'mgf'],
    related: ['martingales', 'random-walks', 'chain-classification'],
    tags: ['Galton–Watson', 'probability generating function', 'extinction', 'criticality', 'epidemics'],
    summary: R`A Galton–Watson process models a population in which every individual independently leaves a random number of offspring with a fixed law. The population sizes $Z_n$ form a Markov chain with an absorbing state at zero, and the whole theory runs on one object: the offspring probability generating function $G(s) = \E s^\xi$. The $n$-th generation has generating function $G$ composed with itself $n$ times, its mean is $m^n$, and the extinction probability is the smallest fixed point of $G$ in $[0,1]$ — equal to one exactly when $m\le 1$ (apart from the trivial case of exactly one child). The model underlies epidemic thresholds ($R_0 = m$), the giant component of random graphs, and the survival of family names.`,
    survey: {
      task: 'Model population growth from independent reproduction; compute extinction probability and growth rate',
      assumptions: R`IID offspring counts across individuals and generations; no interaction or density dependence`,
      output: R`$G_n = G^{\circ n}$; $\E Z_n = m^n$; extinction probability $q = \min\{s\in[0,1]: G(s)=s\}$`,
      strengths: 'Exact fixed-point characterization; sharp threshold at $m=1$; clean martingale limit theory',
      weaknesses: 'Independence and infinite resources are unrealistic at large population sizes; ignores spatial and network structure',
      complexity: R`$q$ by fixed-point iteration $s_{n+1} = G(s_n)$ from $s_0=0$ (linear convergence when $m\ne1$)`,
    },
    sections: [
      {
        title: 'The Galton–Watson process',
        html: R`
<p>In 1873 Francis Galton asked why so many aristocratic family names were dying out; Henry Watson's answer (1874) introduced the model, though with an error in the conclusion, and Irénée-Jules Bienaymé had stated the correct criterion already in 1845. The model is the simplest one in which randomness compounds multiplicatively.</p>
<div class="def" data-name="Galton–Watson process">
<p>Let $(p_k)_{k\ge0}$ be a distribution on $\{0,1,2,\dots\}$ (the <em>offspring law</em>) and let $\{\xi_{n,i}: n\ge0, i\ge1\}$ be IID with law $(p_k)$. Set $Z_0 = 1$ and
$$Z_{n+1} = \sum_{i=1}^{Z_n}\xi_{n,i},$$
with the empty sum equal to $0$. $Z_n$ is the size of generation $n$; individual $i$ of generation $n$ has $\xi_{n,i}$ children. Write $m = \E\xi = \sum_kkp_k$ and $\sigma^2 = \Var\xi$.</p>
</div>
<p>$(Z_n)$ is a Markov chain on $\{0,1,2,\dots\}$: given $Z_n = k$, $Z_{n+1}$ is a sum of $k$ fresh IID offspring counts, with law $(p_\cdot)^{*k}$, the $k$-fold convolution. State $0$ is absorbing. The <strong>extinction</strong> event is $\{Z_n = 0\text{ for some }n\}$ and $q$ denotes its probability.</p>
<div class="thm" data-kind="Proposition" data-name="Extinction or explosion">
<p>If $p_1<1$, then almost surely either $Z_n = 0$ eventually or $Z_n\to\infty$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>If $p_0>0$: from any state $k\ge1$ the chain jumps to the absorbing state $0$ in one step with probability $p_0^k>0$ and then never returns, so every $k\ge1$ is transient and is visited only finitely often a.s. by the <a href="#/t/chain-classification">recurrence criterion</a>. Hence $Z_n$ eventually leaves every finite set unless it is absorbed at $0$. If $p_0 = 0$ then $Z_n$ is non-decreasing and strictly increases with probability at least $1-p_1^{Z_n}\ge1-p_1>0$ at each step, so $Z_n\to\infty$ a.s.</p>
</details>
<p>There is no stable intermediate population. This is a property of the model (no density dependence), not of nature: real populations are regulated by resources, but in the early phase of an invasion or epidemic, when resources are not limiting, the branching approximation is excellent.</p>`,
      },
      {
        title: 'Generating functions and moments',
        html: R`
<p>Because $Z_{n+1}$ is a random sum of IID variables, the natural tool is the <strong>probability generating function</strong> (pgf), $G(s) = \E s^\xi = \sum_kp_ks^k$ for $s\in[0,1]$ — the discrete cousin of the <a href="#/t/mgf">moment generating function</a>. On $[0,1]$, $G$ is increasing and convex (all coefficients are non-negative), with $G(0) = p_0$, $G(1) = 1$ and $G'(1) = m$ (possibly $+\infty$).</p>
<div class="thm" data-name="Generation pgfs compose">
<p>Let $G_n(s) = \E s^{Z_n}$. Then $G_{n+1} = G_n\circ G = G\circ G_n$, so $G_n = G\circ G\circ\cdots\circ G$ ($n$ times). Consequently
$$\E Z_n = m^n,\qquad \Var Z_n = \begin{cases}\sigma^2m^{n-1}\dfrac{m^n-1}{m-1}, & m\ne1,\\[1mm] n\sigma^2, & m=1.\end{cases}$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Conditioning on $Z_n$ and using independence of the offspring counts,
$$\E[s^{Z_{n+1}}\mid Z_n] = \prod_{i=1}^{Z_n}\E s^{\xi_{n,i}} = G(s)^{Z_n},$$
so $G_{n+1}(s) = \E[G(s)^{Z_n}] = G_n(G(s))$. (Conditioning instead on the first generation, each of the $Z_1$ children founds an independent copy of the process, giving $G_{n+1} = G\circ G_n$.)</p>
<p>Moments: the tower property gives $\E Z_{n+1} = \E[\E(Z_{n+1}\mid Z_n)] = m\E Z_n$, so $\E Z_n = m^n$. For the variance, the conditional variance formula gives $V_{n+1} = \E[\Var(Z_{n+1}\mid Z_n)] + \Var(\E[Z_{n+1}\mid Z_n]) = \sigma^2m^n + m^2V_n$ with $V_0 = 0$. The stated formula satisfies this recursion: $\sigma^2m^n + m^2\sigma^2m^{n-1}\frac{m^n-1}{m-1} = \sigma^2m^n\frac{(m-1) + m^{n+1}-m}{m-1} = \sigma^2m^n\frac{m^{n+1}-1}{m-1}$; the $m=1$ case is the limit.</p>
</details>
<p>The mean grows or decays geometrically, but the coefficient of variation tells a richer story. For $m>1$, $\sqrt{\Var Z_n}/\E Z_n\to\sigma/\sqrt{m(m-1)}$, a positive constant: the randomness of the first few generations is never averaged away; it is <em>amplified</em> and frozen into the eventual growth constant. For $m=1$ the mean is stuck at $1$ while the variance grows linearly — the signature of a process that almost always dies but occasionally becomes large.</p>`,
      },
      {
        title: 'The extinction probability',
        html: R`
<div class="thm" data-name="Extinction probability">
<p>The extinction probability $q$ is the smallest root of $G(s) = s$ in $[0,1]$. Moreover, if $p_1<1$:</p>
<ul>
<li>$m\le1\ \Rightarrow\ q = 1$ (extinction is certain);</li>
<li>$m>1\ \Rightarrow\ q<1$, and $q$ is the unique root of $G(s)=s$ in $[0,1)$.</li>
</ul>
<p>Furthermore $\P(Z_n = 0) = G_n(0)$ increases to $q$: iterating $s_{k+1} = G(s_k)$ from $s_0=0$ converges to $q$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>$q$ is a fixed point.</em> The events $\{Z_n = 0\}$ increase to the extinction event, so $q = \lim_n\P(Z_n=0) = \lim_nG_n(0)$. Since $G_{n+1}(0) = G(G_n(0))$ and $G$ is continuous on $[0,1]$, letting $n\to\infty$ gives $q = G(q)$.</p>
<p><em>$q$ is the smallest.</em> Let $r\in[0,1]$ satisfy $G(r) = r$. Since $G$ is increasing, $G_1(0) = G(0)\le G(r) = r$, and inductively $G_{n+1}(0) = G(G_n(0))\le G(r) = r$. Letting $n\to\infty$, $q\le r$.</p>
<p><em>Which root.</em> Let $h(s) = G(s)-s$, convex on $[0,1]$ with $h(1)=0$ and $h(0) = p_0\ge0$. If $m\le1$: when $p_0+p_1<1$, $G$ is strictly convex, so $G'(s)< G'(1) = m\le1$ for $s<1$; then $h$ is strictly decreasing on $[0,1]$ and has no root in $[0,1)$, so $q=1$. When $p_0+p_1=1$ (with $p_1<1$), $h(s) = p_0(1-s)>0$ on $[0,1)$, again $q=1$. If $m>1$: $h'(1) = m-1>0$, so $h<0$ just left of $1$; since $h(0)\ge0$ there is a root in $[0,1)$, and strict convexity (forced by $m>1$) allows at most two roots in $[0,1]$, one of which is $1$.</p>
</details>
<div class="note" data-name="The picture">
<p>Draw $y = G(s)$ and the diagonal $y=s$ on the unit square. Both pass through $(1,1)$. If the slope of $G$ at $1$ (that is, $m$) is at most 1, the convex curve stays above the diagonal and the cobweb iteration from $0$ climbs to $1$. If $m>1$, the curve dips below the diagonal before reaching $(1,1)$, crossing it once at $q$; the cobweb stalls there. The first lab draws exactly this.</p>
</div>
<p>When the population starts from $Z_0 = k$ individuals, the $k$ subtrees are independent and the extinction probability is $q^k$. This is why a small number of introductions of an invasive species or a pathogen with $R_0$ only slightly above one usually fizzles out, while a large number almost surely takes hold.</p>`,
        lab: 'branching-processes-pgf',
      },
      {
        title: 'Criticality and the martingale limit',
        html: R`
<p>The mean $m$ splits the behaviour into three regimes.</p>
<div class="cols">
<div><h4>Subcritical, $m<1$</h4><p>$\P(Z_n>0)\le\E Z_n = m^n$ by Markov's inequality: extinction comes fast, geometrically. The total progeny $\sum_nZ_n$ has mean $\sum_nm^n = 1/(1-m)$.</p></div>
<div><h4>Critical, $m=1$</h4><p>Extinction is certain, but slow. If $\sigma^2<\infty$, Kolmogorov's estimate $\P(Z_n>0)\sim\frac{2}{n\sigma^2}$ holds, and conditioned on survival $Z_n/n$ converges to an exponential with mean $\sigma^2/2$ (Yaglom). The total progeny has infinite mean.</p></div>
<div><h4>Supercritical, $m>1$</h4><p>With probability $1-q$ the population grows like $m^n$ times a random constant; with probability $q$ it dies.</p></div>
</div>
<p>The precise statement of supercritical growth uses a martingale.</p>
<div class="thm" data-name="The Galton–Watson martingale">
<p>If $0< m<\infty$, then $W_n = Z_n/m^n$ is a non-negative <a href="#/t/martingales">martingale</a> and converges almost surely to a limit $W$ with $\E W\le1$. If $m>1$ and $\sigma^2<\infty$, the convergence also holds in $L^2$, $\E W = 1$, $\Var W = \sigma^2/(m(m-1))$, and $\{W=0\}$ coincides a.s. with extinction. (Kesten–Stigum, 1966: for $m>1$, $\P(W>0) = 1-q$ if and only if $\E[\xi\log\xi]<\infty$; otherwise $W=0$ a.s.)</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$\E[Z_{n+1}\mid Z_0,\dots,Z_n] = mZ_n$, so $\E[W_{n+1}\mid\mathcal F_n] = W_n$; a non-negative martingale converges a.s. by the martingale convergence theorem, and Fatou gives $\E W\le1$. With finite variance, $\E W_n^2 = 1+\Var Z_n/m^{2n}\to1+\sigma^2/(m(m-1))$ by the variance formula, so $(W_n)$ is bounded in $L^2$ and converges in $L^2$, giving $\E W=1$ and the stated variance. Finally $\P(W=0)$ is a fixed point of $G$ (condition on the first generation: $W=0$ iff every child's subtree has limit zero), it is not $1$ because $\E W=1$, so it equals $q$; since extinction implies $W=0$, the two events coincide a.s.</p>
</details>
<div class="warn" data-name="The mean is a poor summary in the critical case">
<p>For $m=1$, $\E Z_n = 1$ for every $n$, yet $Z_n\to0$ a.s. The martingale $W_n = Z_n$ converges almost surely to $0$ while its expectation stays at $1$ — the textbook example of a martingale that is not uniformly integrable. All of the mean is carried by rare, large surviving families.</p>
</div>`,
        lab: 'branching-processes-generations',
      },
      {
        title: 'Worked examples',
        html: R`
<div class="ex" data-name="Poisson offspring">
<p>$\xi\sim\text{Poisson}(m)$ gives $G(s) = e^{m(s-1)}$, so $q$ solves $q = e^{-m(1-q)}$. For $m=2$, iteration from $0$ gives $q\approx0.2032$; for $m=1.5$, $q\approx0.417$. The survival probability $1-q$ solves $1-q = 1-e^{-m(1-q)}$, which is the same equation as the size of the <em>giant component</em> of an Erdős–Rényi random graph with mean degree $m$: exploring the component of a vertex is, to first order, a Poisson branching process.</p>
</div>
<div class="ex" data-name="Geometric offspring: a closed form">
<p>Let $p_k = (1-a)a^k$, $k\ge0$, so $m = a/(1-a)$ and $G(s) = \frac{1-a}{1-as}$. The fixed-point equation $as^2-s+(1-a) = 0$ has roots $s=1$ and $s = (1-a)/a = 1/m$. Hence $q = \min(1, 1/m)$. With $m=2$, half of all lineages die out — far more than under Poisson offspring with the same mean, because the geometric law puts mass $p_0 = 1/3$ on zero children.</p>
</div>
<div class="ex" data-name="Superspreading">
<p>Epidemiologists model individual reproduction numbers with a negative binomial of mean $R$ and dispersion $k$: $G(s) = \big(1+\tfrac Rk(1-s)\big)^{-k}$. Small $k$ means most cases infect nobody and a few infect many. With $R = 2$: Poisson ($k=\infty$) gives $q\approx0.20$, $k=0.5$ gives $q\approx0.64$, and $k=0.1$ gives $q\approx0.89$. Same mean, wildly different chances that an introduction sparks an outbreak (Lloyd-Smith et al., 2005). Means alone do not determine extinction; the whole pgf does.</p>
</div>
<div class="ex" data-name="Total progeny via a random walk">
<p>Explore the family tree one individual at a time, keeping a queue of individuals not yet explored. Starting with 1 in the queue, each exploration removes one individual and adds its $\xi_i$ children, so the queue length after $n$ explorations is $1+\sum_{i\le n}(\xi_i-1)$ — a random walk with steps $\xi-1$ and drift $m-1$. The total progeny is the first time this walk hits $0$. The hitting time theorem (Otter–Dwass) then gives
$$\P(\text{total progeny} = n) = \frac1n\,\P(\xi_1+\dots+\xi_n = n-1).$$
For Poisson offspring this is the Borel distribution $e^{-mn}(mn)^{n-1}/n!$. Extinction ($q=1$) is exactly recurrence of this walk to $0$, which by the strong law happens a.s. iff its drift $m-1\le0$ — another proof of the threshold.</p>
</div>`,
      },
      {
        title: 'Applications, extensions and pitfalls',
        html: R`
<ul>
<li><strong>Epidemics.</strong> In the early phase of an outbreak each case infects a random number of others with mean $R_0$; $R_0>1$ is the branching-process threshold. Interventions that reduce $R$ below 1 make the process subcritical.</li>
<li><strong>Nuclear chain reactions.</strong> Neutrons per fission play the role of $\xi$; "criticality" in reactor physics is literally $m=1$.</li>
<li><strong>Cell biology and genetics.</strong> Survival of a new beneficial mutation with selective advantage $s$ is a slightly supercritical branching process; Haldane's approximation gives survival probability about $2s$.</li>
<li><strong>Multitype processes</strong> track several kinds of individual; the threshold becomes the Perron–Frobenius eigenvalue of the mean offspring matrix.</li>
<li><strong>Continuous time.</strong> Exponential lifetimes give a <a href="#/t/ctmc">continuous-time Markov chain</a> (birth–death process); critical processes rescaled in space and time converge to Feller's diffusion.</li>
</ul>
<div class="warn" data-name="Growth in expectation is not growth in probability">
<p>$\E Z_n = m^n\to\infty$ for $m>1$, yet the process can still die with high probability (geometric offspring with $m=1.1$ dies with probability $1/1.1\approx0.91$). Expected-value reasoning overstates how reliably a small founding population takes off.</p>
</div>
<div class="warn" data-name="Estimation from one lineage">
<p>Observing generation sizes $Z_0,\dots,Z_n$ of a single tree, the natural estimator $\hat m = \sum_{k\ge1}Z_k/\sum_{k< n}Z_k$ (Harris) is consistent on the survival event, but the variance $\sigma^2$ and the full offspring law are not consistently estimable from generation totals alone without more structure, and on the extinction event nothing is consistent — the information is finite.</p>
</div>
<div class="warn" data-name="Independence is doing the work">
<p>The fixed-point result assumes individuals reproduce independently with the same law. Household structure, depletion of susceptibles, or heterogeneity that is correlated within families all break the pgf composition rule; the model then is at best an early-phase approximation.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Offspring law: $p_0 = 0.25$, $p_1 = 0.25$, $p_2 = 0.5$. Find $m$ and the extinction probability.`,
        a: R`$m = 0.25 + 2\cdot0.5 = 1.25>1$. Solve $s = 0.25+0.25s+0.5s^2$, i.e. $0.5s^2-0.75s+0.25 = 0$, i.e. $2s^2-3s+1 = 0$, with roots $s = 1$ and $s = 1/2$. So $q = 1/2$.` },
      { q: R`Why is the extinction probability the <em>smallest</em> fixed point rather than, say, the largest?`,
        a: R`$q = \lim G_n(0)$ is approached from below by iterating the increasing map $G$ starting at $0$, and such an iteration can never jump over a fixed point $r$: $G_n(0)\le r$ implies $G_{n+1}(0) = G(G_n(0))\le G(r) = r$. Since $1$ is always a fixed point, the largest one would always give $q=1$, which is wrong for $m>1$.` },
      { q: R`In the critical case $m=1$ with $\sigma^2 = 1$, roughly what is $\P(Z_{100}>0)$, and how large is $Z_{100}$ typically given that it is positive?`,
        a: R`Kolmogorov's estimate gives $\P(Z_n>0)\approx2/(n\sigma^2) = 0.02$. Since $\E Z_{100} = 1 = \P(Z_{100}>0)\,\E[Z_{100}\mid Z_{100}>0]$, the conditional mean is about $50$, consistent with Yaglom's limit (exponential with mean $n\sigma^2/2 = 50$).` },
      { q: R`A process starts with 5 independent founders and has Poisson(2) offspring. What is the probability it eventually dies out?`,
        a: R`Each founder's line is an independent copy, so $q^5$ with $q\approx0.2032$: about $0.2032^5\approx3.5\times10^{-4}$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'hidden-markov',
    domain: 'stochastic',
    order: 60,
    title: 'Hidden Markov Models',
    subtitle: 'Inferring a latent Markov chain from noisy observations: forward–backward, Viterbi, Baum–Welch',
    level: 3,
    sources: { other: 'Rabiner (1989), Proc. IEEE 77(2); Durbin, Eddy, Krogh & Mitchison, Biological Sequence Analysis (1998), Ch. 3; Cappé, Moulines & Rydén, Inference in Hidden Markov Models (2005)' },
    prereqs: ['markov-chains', 'em-algorithm'],
    related: ['particle-filters', 'directed-graphical-models', 'gaussian-mixtures', 'rnn-lstm', 'time-series'],
    tags: ['latent variables', 'filtering', 'smoothing', 'dynamic programming', 'EM'],
    summary: R`A hidden Markov model (HMM) is a Markov chain $Z_1,\dots,Z_T$ that we never see, together with observations $X_t$ that depend only on the current hidden state. The observed sequence is not Markov, yet the chain structure makes exact inference cheap: the forward–backward algorithm computes the likelihood and every posterior marginal $\P(Z_t = k\mid x_{1:T})$ in $O(TK^2)$ time instead of $O(K^T)$; the Viterbi algorithm finds the single most probable hidden path by dynamic programming; and Baum–Welch fits all parameters by EM, with expected sufficient statistics supplied by forward–backward. HMMs are the prototype of state-space models and of message passing on graphical models.`,
    survey: {
      task: 'Sequence modelling with a discrete latent regime: likelihood, filtering/smoothing, decoding, parameter estimation',
      assumptions: R`Latent first-order Markov chain; observations conditionally independent given the hidden states; $X_t$ depends only on $Z_t$`,
      output: R`$p(x_{1:T})$; posteriors $\gamma_t(k)$, $\xi_t(k,l)$; Viterbi path $\hat z_{1:T}$; MLE (local) of $(\pi, A, B)$`,
      strengths: 'Exact inference in linear time in $T$; interpretable regimes; EM updates in closed form',
      weaknesses: 'Geometric state durations; first-order memory; EM finds local optima; label switching; $K$ must be chosen',
      complexity: R`$O(TK^2)$ per forward–backward or Viterbi pass; Baum–Welch: that per iteration`,
    },
    sections: [
      {
        title: 'The model',
        html: R`
<p>Many sequences are driven by a regime that we cannot observe: a speaker's intended phoneme behind an acoustic signal, a gene's coding/non-coding status behind a DNA sequence, a market's calm/turbulent state behind daily returns, a casino's fair/loaded die behind a sequence of rolls. An HMM posits that the regime is a Markov chain and that each observation is a noisy readout of the current regime.</p>
<div class="def" data-name="Hidden Markov model">
<p>Hidden states $Z_1,\dots,Z_T\in\{1,\dots,K\}$ form a Markov chain with initial distribution $\pi$ and transition matrix $A = (a_{kl})$. Observations $X_1,\dots,X_T$ satisfy
$$p(x_t\mid z_{1:T}, x_{1:t-1}, x_{t+1:T}) = b_{z_t}(x_t),$$
for emission densities (or pmfs) $b_1,\dots,b_K$. The joint law factorizes as
$$p(z_{1:T}, x_{1:T}) = \pi_{z_1}b_{z_1}(x_1)\prod_{t=2}^T a_{z_{t-1}z_t}\,b_{z_t}(x_t).$$
Write $\theta = (\pi, A, b)$.</p>
</div>
<p>As a <a href="#/t/directed-graphical-models">directed graphical model</a> the HMM is a chain $Z_1\to Z_2\to\cdots\to Z_T$ with a pendant arrow $Z_t\to X_t$ at each time. Two conditional independences drive every algorithm below: given $Z_t$, the past $(Z_{1:t-1}, X_{1:t})$ is independent of the future $(Z_{t+1:T}, X_{t+1:T})$; and $X_t$ depends on everything else only through $Z_t$.</p>
<div class="note">
<p>The observed process $(X_t)$ alone is <em>not</em> Markov — the hidden state carries memory, so a long run of sixes makes the next six more likely even though each roll depends only on the current die. An HMM is thus a parsimonious way to get long-range dependence in $X$ from a first-order latent chain. A <a href="#/t/gaussian-mixtures">mixture model</a> is the special case where the $Z_t$ are IID ($A$ has identical rows); the HMM adds temporal persistence of the mixture component.</p>
</div>
<p>Rabiner's tutorial framed three canonical problems: (1) <strong>evaluation</strong> — compute $p(x_{1:T}\mid\theta)$; (2) <strong>decoding</strong> — infer the hidden states, either marginally ($\P(Z_t=k\mid x_{1:T})$) or jointly (the most probable path); (3) <strong>learning</strong> — estimate $\theta$. The naive sum $p(x) = \sum_{z_{1:T}}p(z,x)$ has $K^T$ terms. The chain structure collapses this to $O(TK^2)$.</p>`,
      },
      {
        title: 'Filtering and the forward algorithm',
        html: R`
<p>Define the <strong>forward variables</strong> $\alpha_t(k) = p(x_{1:t},\ Z_t = k)$.</p>
<div class="thm" data-name="Forward recursion">
<p>$\alpha_1(k) = \pi_kb_k(x_1)$, and for $t = 1,\dots,T-1$,
$$\alpha_{t+1}(l) = b_l(x_{t+1})\sum_{k=1}^K\alpha_t(k)\,a_{kl}.$$
The likelihood is $p(x_{1:T}) = \sum_k\alpha_T(k)$, and the filtering distribution is $\P(Z_t=k\mid x_{1:t}) = \alpha_t(k)/\sum_j\alpha_t(j)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Sum over the previous state and factor using the conditional independences:
$$\alpha_{t+1}(l) = \sum_kp(x_{1:t}, Z_t=k)\;p(Z_{t+1}=l\mid Z_t = k, x_{1:t})\;p(x_{t+1}\mid Z_{t+1}=l, Z_t=k, x_{1:t}).$$
The second factor is $a_{kl}$ because $Z_{t+1}$ depends on the past only through $Z_t$; the third is $b_l(x_{t+1})$ because $X_{t+1}$ depends only on $Z_{t+1}$. The likelihood is the marginal of $\alpha_T$, and the filter is Bayes' rule.</p>
</details>
<p>Each step is a vector–matrix product ($K^2$ operations) followed by elementwise multiplication, so the total cost is $O(TK^2)$. The recursion is the discrete form of the <em>predict–update</em> cycle of Bayesian filtering: $\sum_k\alpha_t(k)a_{kl}$ predicts $Z_{t+1}$ from data up to $t$, and multiplying by $b_l(x_{t+1})$ updates with the new observation. With continuous linear-Gaussian states it becomes the Kalman filter; with intractable nonlinear dynamics, the <a href="#/t/particle-filters">particle filter</a> approximates it by simulation.</p>
<div class="warn" data-name="Underflow">
<p>$\alpha_t$ is a joint probability of $t$ observations and decays geometrically; for $T$ in the hundreds it underflows double precision. Always normalize: carry $\hat\alpha_t = \alpha_t/c_{1:t}$ where $c_t = \sum_l(\text{unnormalized }\alpha_t)(l) = p(x_t\mid x_{1:t-1})$, so that $\hat\alpha_t$ is the filtering distribution and $\log p(x_{1:T}) = \sum_t\log c_t$. Alternatively work in log space with log-sum-exp.</p>
</div>
<div class="ex" data-name="Two sixes at the casino">
<p>A casino switches between a fair die (F) and a loaded die (L) with $\P(6\mid L) = \tfrac12$ and the other faces $\tfrac1{10}$ each; $a_{FL} = 0.05$, $a_{LF} = 0.1$, and $\pi = (\tfrac12,\tfrac12)$. Observe $x = (6,6)$. Then $\alpha_1 = (\tfrac12\cdot\tfrac16,\ \tfrac12\cdot\tfrac12) = (0.0833,\ 0.25)$ and
$$\alpha_2(F) = \tfrac16(0.0833\cdot0.95+0.25\cdot0.1) = 0.01736,\qquad \alpha_2(L) = \tfrac12(0.0833\cdot0.05+0.25\cdot0.9) = 0.11458.$$
So $p(6,6) = 0.1319$ and the filter puts $\P(Z_2 = L\mid 6,6) = 0.1146/0.1319 = 33/38\approx0.868$ on the loaded die.</p>
</div>`,
      },
      {
        title: 'Smoothing: the backward pass',
        html: R`
<p>Filtering uses only the past. To use the whole sequence, define the <strong>backward variables</strong> $\beta_t(k) = p(x_{t+1:T}\mid Z_t = k)$, with $\beta_T(k) = 1$.</p>
<div class="thm" data-name="Backward recursion and posterior marginals">
<p>For $t = T-1,\dots,1$,
$$\beta_t(k) = \sum_{l=1}^Ka_{kl}\,b_l(x_{t+1})\,\beta_{t+1}(l).$$
The smoothed marginals and pairwise marginals are
$$\gamma_t(k) := \P(Z_t=k\mid x_{1:T}) = \frac{\alpha_t(k)\beta_t(k)}{p(x_{1:T})},\qquad \xi_t(k,l) := \P(Z_t=k, Z_{t+1}=l\mid x_{1:T}) = \frac{\alpha_t(k)\,a_{kl}\,b_l(x_{t+1})\,\beta_{t+1}(l)}{p(x_{1:T})}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Backward: condition on $Z_{t+1}$: $\beta_t(k) = \sum_l\P(Z_{t+1}=l\mid Z_t=k)\,p(x_{t+1}\mid Z_{t+1}=l)\,p(x_{t+2:T}\mid Z_{t+1}=l)$, each factorization licensed by the Markov and emission structure. Posterior: $p(x_{1:T}, Z_t = k) = p(x_{1:t}, Z_t=k)\,p(x_{t+1:T}\mid Z_t=k, x_{1:t}) = \alpha_t(k)\beta_t(k)$ because the future observations are independent of the past ones given $Z_t$. The pairwise formula follows the same way, splitting the sequence at the edge $(t,t+1)$.</p>
</details>
<p>Forward–backward is the sum-product (belief propagation) algorithm on a chain: $\alpha$ is the message passed rightwards, $\beta$ the message passed leftwards, and the marginal at a node is the normalized product of the incoming messages. With scaled forward variables, use the same constants $c_t$ to scale $\beta$; then $\gamma_t = \hat\alpha_t\odot\hat\beta_t$ needs no further normalization.</p>
<div class="note">
<p>Smoothing can reverse a filtering decision. Immediately after a switch to the loaded die, the filter is still unsure — it has seen only one or two suspicious rolls — but the smoother has also seen the run of sixes that follows and assigns the switch to the right time. The lab below shows filtered and smoothed curves side by side.</p>
</div>`,
        lab: 'hidden-markov-casino',
      },
      {
        title: 'Decoding: the Viterbi algorithm',
        html: R`
<p>Posterior marginals answer "what was the state at time $t$?". Often we want a single coherent explanation of the whole sequence: $\hat z_{1:T} = \argmax_{z_{1:T}}p(z_{1:T}\mid x_{1:T})$. Replacing sums by maxima in the forward recursion gives a dynamic program (Viterbi, 1967).</p>
<div class="thm" data-name="Viterbi recursion">
<p>Let $\delta_t(l) = \max_{z_{1:t-1}}p(z_{1:t-1},\ Z_t=l,\ x_{1:t})$. Then $\delta_1(l) = \pi_lb_l(x_1)$ and
$$\delta_{t+1}(l) = b_l(x_{t+1})\max_k\delta_t(k)a_{kl},\qquad \psi_{t+1}(l) = \argmax_k\delta_t(k)a_{kl}.$$
The most probable path is recovered by $\hat z_T = \argmax_l\delta_T(l)$ and backtracking $\hat z_t = \psi_{t+1}(\hat z_{t+1})$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the factorization, $p(z_{1:t+1}, x_{1:t+1}) = p(z_{1:t}, x_{1:t})\,a_{z_tz_{t+1}}\,b_{z_{t+1}}(x_{t+1})$. Maximizing over $z_{1:t}$ with $z_{t+1}=l$ fixed, first maximize over $z_{1:t-1}$ with $z_t = k$ fixed — this gives $\delta_t(k)$, since the remaining factors do not involve $z_{1:t-1}$ — then over $k$. This is Bellman's principle of optimality: every prefix of an optimal path ending at $(t,k)$ is itself optimal among paths ending at $(t,k)$. Backtracking reads off the maximizers.</p>
</details>
<p>Work in log space: products become sums and the recursion is a shortest-path computation on the trellis graph with $K$ nodes per time slice and edge weights $-\log a_{kl}-\log b_l(x_{t+1})$.</p>
<div class="cols">
<div><h4>Viterbi path</h4><p>Maximizes the joint posterior $p(z_{1:T}\mid x)$: the best single sequence. Always a feasible path (never uses a zero-probability transition). Minimizes the expected 0–1 loss on the <em>entire</em> sequence.</p></div>
<div><h4>Posterior decoding</h4><p>$\hat z_t = \argmax_k\gamma_t(k)$ separately for each $t$. Minimizes the expected number of incorrectly labelled positions (Hamming loss). The resulting sequence can be impossible under $A$.</p></div>
</div>
<p>The two often agree but differ around uncertain stretches. In the lab, compare their accuracy against the true states; posterior decoding usually wins on per-position accuracy, which is the criterion it optimizes, while Viterbi produces cleaner segments.</p>`,
      },
      {
        title: 'Learning: Baum–Welch is EM',
        html: R`
<p>If the hidden states were observed, the MLE would be a matter of counting: transition frequencies for $A$, emission frequencies for $b$. They are not observed, so we use the <a href="#/t/em-algorithm">EM algorithm</a>, replacing counts by their posterior expectations. Baum and colleagues derived the procedure (1970) before EM was formalized (Dempster, Laird & Rubin, 1977).</p>
<p>The complete-data log-likelihood is
$$\log p(z,x\mid\theta) = \sum_k\ind\{z_1=k\}\log\pi_k + \sum_{t=1}^{T-1}\sum_{k,l}\ind\{z_t=k,z_{t+1}=l\}\log a_{kl} + \sum_{t=1}^T\sum_k\ind\{z_t=k\}\log b_k(x_t).$$
Taking the expectation under $p(z\mid x,\theta^{\text{old}})$ replaces indicators by $\gamma$ and $\xi$:
$$Q(\theta\mid\theta^{\text{old}}) = \sum_k\gamma_1(k)\log\pi_k + \sum_{t< T}\sum_{k,l}\xi_t(k,l)\log a_{kl} + \sum_t\sum_k\gamma_t(k)\log b_k(x_t).$$</p>
<div class="algo" data-name="Baum–Welch">
<ol>
<li><strong>E-step.</strong> Run forward–backward with the current $\theta$ to get $\gamma_t(k)$ and $\xi_t(k,l)$.</li>
<li><strong>M-step.</strong> Maximize $Q$ term by term (Lagrange multipliers for the sum-to-one constraints):
$$\pi_k = \gamma_1(k),\qquad a_{kl} = \frac{\sum_{t< T}\xi_t(k,l)}{\sum_{t< T}\gamma_t(k)},\qquad b_k(v) = \frac{\sum_t\gamma_t(k)\ind\{x_t=v\}}{\sum_t\gamma_t(k)}\ \ \text{(discrete emissions)}.$$
For Gaussian emissions: $\mu_k = \sum_t\gamma_t(k)x_t/\sum_t\gamma_t(k)$ and $\sigma_k^2 = \sum_t\gamma_t(k)(x_t-\mu_k)^2/\sum_t\gamma_t(k)$.</li>
<li>Repeat until the log-likelihood $\sum_t\log c_t$ stops increasing.</li>
</ol>
</div>
<div class="thm" data-kind="Proposition" data-name="Monotonicity">
<p>Each Baum–Welch iteration does not decrease $p(x_{1:T}\mid\theta)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>This is the general EM ascent property: $\log p(x\mid\theta) = Q(\theta\mid\theta^{\text{old}}) + H(\theta\mid\theta^{\text{old}})$ with $H(\theta\mid\theta^{\text{old}}) = -\E_{\theta^{\text{old}}}[\log p(z\mid x,\theta)\mid x]$. By Gibbs' inequality, $H(\theta\mid\theta^{\text{old}})\ge H(\theta^{\text{old}}\mid\theta^{\text{old}})$ for every $\theta$ (the difference is a <a href="#/t/entropy-kl">KL divergence</a>). The M-step ensures $Q(\theta^{\text{new}}\mid\theta^{\text{old}})\ge Q(\theta^{\text{old}}\mid\theta^{\text{old}})$; adding the two inequalities gives $\log p(x\mid\theta^{\text{new}})\ge\log p(x\mid\theta^{\text{old}})$.</p>
</details>
<p>With several independent sequences, sum the expected counts across sequences before normalizing. The second lab runs Baum–Welch on simulated casino data from a random start and plots the log-likelihood climbing — and, with an unlucky start, plateauing at an inferior local maximum.</p>`,
        lab: 'hidden-markov-baumwelch',
      },
      {
        title: 'Pitfalls and connections',
        html: R`
<div class="warn" data-name="Local maxima and label switching">
<p>The likelihood is multimodal. Any permutation of the state labels gives the same likelihood ($K!$ equivalent maxima), so estimated states have no intrinsic names — identify them afterwards by their emission parameters. Beyond relabelling, genuinely different local maxima exist; use several random starts and keep the best. A parameter initialized at exactly zero (a transition or emission probability) stays zero forever under EM.</p>
</div>
<div class="warn" data-name="Geometric durations">
<p>The time spent in a hidden state before leaving is geometric with mean $1/(1-a_{kk})$, whose mode is 1. Regimes with characteristic durations (phonemes, exons, recessions) are poorly fit; <em>hidden semi-Markov models</em> attach an explicit duration distribution to each state at extra computational cost.</p>
</div>
<div class="warn" data-name="Choosing the number of states">
<p>The likelihood always increases with $K$. Standard asymptotics for likelihood-ratio tests of $K$ versus $K+1$ fail (parameters on the boundary, non-identifiability under the null), so use penalized criteria such as BIC, cross-validated predictive likelihood, or substantive knowledge.</p>
</div>
<div class="warn" data-name="Viterbi training is not maximum likelihood">
<p>"Hard EM" — decode with Viterbi, then count — is faster but optimizes a different objective (the joint likelihood of the single best path). It is biased, especially for transition probabilities, and tends to over-segment or under-segment depending on the emission overlap.</p>
</div>
<p>Connections:</p>
<ul>
<li><strong>State-space models.</strong> Continuous hidden state with linear-Gaussian dynamics gives the Kalman filter and RTS smoother — the same forward–backward structure with Gaussian messages. Nonlinear versions use <a href="#/t/particle-filters">particle filters</a>.</li>
<li><strong>Graphical models.</strong> Forward–backward and Viterbi are sum-product and max-product on a chain; on trees and junction trees the same messages give exact inference. See <a href="#/t/undirected-graphical-models">undirected graphical models</a> and conditional random fields, the discriminative counterpart of HMMs.</li>
<li><strong>Sequence models in ML.</strong> <a href="#/t/rnn-lstm">Recurrent networks</a> replace the discrete hidden state with a continuous, deterministically updated one; HMMs remain competitive when data are scarce and interpretability matters.</li>
<li><strong>Bayesian HMMs.</strong> A <a href="#/t/gibbs-sampling">Gibbs sampler</a> alternates forward-filtering backward-sampling of $z_{1:T}$ with conjugate updates of $\theta$; the hierarchical Dirichlet process HMM lets $K$ be learned.</li>
</ul>`,
      },
    ],
    quiz: [
      { q: R`Why is the observed sequence $X_1, X_2,\dots$ of an HMM generally not a Markov chain, and when is it one?`,
        a: R`$X_{t+1}$ depends on $Z_{t+1}$, whose conditional law given the past observations depends on the whole history $x_{1:t}$ through the filter $\P(Z_t\mid x_{1:t})$, not just on $x_t$. It is Markov if, for example, the emissions are deterministic and injective (then $X_t$ reveals $Z_t$), or if the states are IID (a mixture: $X$ is then IID).` },
      { q: R`What do the forward and backward variables represent, and why does their product give the smoothed marginal?`,
        a: R`$\alpha_t(k) = p(x_{1:t}, Z_t=k)$ summarizes the past; $\beta_t(k) = p(x_{t+1:T}\mid Z_t=k)$ summarizes the future. Given $Z_t$, past and future observations are independent, so $p(x_{1:T}, Z_t=k) = \alpha_t(k)\beta_t(k)$; dividing by $p(x_{1:T}) = \sum_k\alpha_t(k)\beta_t(k)$ gives $\gamma_t(k)$.` },
      { q: R`Give a situation where the posterior-decoding sequence $\argmax_k\gamma_t(k)$ is not a valid path, while Viterbi's is.`,
        a: R`Suppose $a_{12} = 0$ (state 1 can never move directly to state 2). Posterior decoding might pick state 1 at time $t$ (where $\gamma_t(1)$ is slightly larger) and state 2 at $t+1$, because it maximizes each position separately without regard to transitions; the resulting sequence has probability zero. Viterbi maximizes the joint probability, which is zero for any path using a forbidden transition, so it never does this.` },
      { q: R`In Baum–Welch, why is $a_{kl}$ updated as $\sum_t\xi_t(k,l)/\sum_t\gamma_t(k)$ rather than as a ratio of Viterbi counts?`,
        a: R`The M-step maximizes the <em>expected</em> complete-data log-likelihood, in which the transition count $\sum_t\ind\{z_t=k, z_{t+1}=l\}$ is replaced by its posterior expectation $\sum_t\xi_t(k,l)$, and the number of departures from $k$ by $\sum_{t< T}\gamma_t(k)$. Using Viterbi counts would plug in a single (mode) configuration, discarding posterior uncertainty and losing the guarantee that the likelihood increases.` },
      { q: R`What is the computational cost of forward–backward, and what would brute-force marginalization cost for $K=2$ states and $T=300$?`,
        a: R`$O(TK^2)$: about $300\cdot4 = 1200$ multiply-adds per pass. Brute force sums over $2^{300}\approx 2\times10^{90}$ paths — far more than the number of atoms in the observable universe.` },
    ],
  });
})();
