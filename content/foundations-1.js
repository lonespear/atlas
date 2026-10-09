/* Probability Foundations I: probability spaces through conditional expectation. See AUTHORING.md. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'probability-spaces',
    domain: 'foundations',
    order: 10,
    title: 'Probability Spaces & the Axioms',
    subtitle: 'Sample spaces, σ-fields and the three axioms everything else is built on',
    level: 1,
    sources: { aos: '1.1–1.5', cb: '§1.1–1.2', other: 'Billingsley, Probability and Measure (1995), Ch. 1–2; Feller Vol. I, Ch. 1–4' },
    prereqs: [],
    related: ['conditional-probability', 'random-variables', 'pseudo-random'],
    tags: ['axioms', 'sigma-field', 'counting', 'inclusion–exclusion'],
    summary: R`A probability model is a triple $(\Omega, \mathcal F, \P)$: a set of outcomes, a collection of events we are allowed to ask about, and a function that assigns them numbers obeying Kolmogorov's three axioms. Everything in statistics — likelihoods, sampling distributions, posteriors — is a statement about some such triple. The axioms look innocuous, but two of their ingredients carry real content: <em>countable</em> additivity, which gives probability its continuity and makes limits work, and the σ-field, which is forced on us because on uncountable spaces not every set can be assigned a probability. Phrases like "chosen at random" are meaningless until the triple is specified, as Bertrand's paradox makes vivid.`,
    survey: {
      task: 'Formalize uncertainty: say precisely which events have probabilities and what they are',
      assumptions: R`Kolmogorov axioms: $\P \ge 0$, $\P(\Omega)=1$, countable additivity on a σ-field`,
      output: R`A probability space $(\Omega,\mathcal F,\P)$; derived rules (complement, inclusion–exclusion, continuity)`,
      strengths: 'One framework covers discrete, continuous and mixed models; makes limits rigorous',
      weaknesses: 'Silent on interpretation (frequency vs belief); "uniform at random" must still be modelled',
      complexity: R`Counting problems can be combinatorially hard; inclusion–exclusion has $2^n$ terms`,
    },
    sections: [
      {
        title: 'Why we need a formal model',
        html: R`
<p>Ask "draw a chord of a circle at random; what is the probability it is longer than the side of the inscribed equilateral triangle?" and three careful people can get three different answers. Joseph Bertrand posed this in 1889. The point was not that probability is incoherent, but that the phrase <em>at random</em> does not name a probability distribution. Each answer corresponds to a different, perfectly legitimate model:</p>
<ul>
<li><strong>Random endpoints.</strong> Fix one endpoint; the other is uniform on the circle. The chord is long when the second endpoint falls in the arc of $120°$ opposite the first: probability $1/3$.</li>
<li><strong>Random radius.</strong> Choose a radius uniformly in direction, then a point uniformly along it as the chord's midpoint. The chord is long iff the midpoint is within distance $1/2$ of the centre (for a unit circle): probability $1/2$.</li>
<li><strong>Random midpoint.</strong> Choose the midpoint uniformly over the disk. Long iff it lies in the concentric disk of radius $1/2$, area ratio $1/4$.</li>
</ul>
<p>The three procedures induce different distributions on the same set of outcomes (chords). The lesson is the founding one of the subject: <strong>before computing a probability, specify the experiment</strong> — the outcome space, the events, and the measure. Kolmogorov's 1933 axiomatization supplies the language for doing so.</p>
<div class="note"><p>A probability model does not tell you what the "right" distribution is; it tells you what you have committed to once you choose one. Modelling is the choice; probability theory is the bookkeeping that follows.</p></div>`,
        lab: 'probability-spaces-bertrand',
      },
      {
        title: 'Sample spaces, events and σ-fields',
        html: R`
<p>The <strong>sample space</strong> $\Omega$ is the set of possible outcomes $\omega$ of an experiment. Events are subsets of $\Omega$; we say $A$ <em>occurs</em> when the realized $\omega \in A$. Set operations translate logic: $A \cup B$ is "$A$ or $B$", $A\cap B$ is "both", $A^c$ is "not $A$", $A \subseteq B$ is "$A$ implies $B$", and $A\cap B=\varnothing$ means $A$ and $B$ are mutually exclusive. De Morgan's laws, $(\bigcup_i A_i)^c = \bigcap_i A_i^c$ and $(\bigcap_i A_i)^c = \bigcup_i A_i^c$, hold for arbitrary (even uncountable) families.</p>
<div class="def" data-name="σ-field (σ-algebra)">
<p>A collection $\mathcal F$ of subsets of $\Omega$ is a <em>σ-field</em> if (i) $\Omega \in \mathcal F$; (ii) $A \in \mathcal F \Rightarrow A^c \in \mathcal F$; (iii) $A_1, A_2, \dots \in \mathcal F \Rightarrow \bigcup_{i=1}^\infty A_i \in \mathcal F$. The pair $(\Omega,\mathcal F)$ is a <em>measurable space</em> and members of $\mathcal F$ are the <em>events</em>.</p>
</div>
<p>By De Morgan, a σ-field is also closed under countable intersections, and it contains $\varnothing = \Omega^c$. Closure under <em>countable</em> (not merely finite) operations is what lets us talk about events like "the coin eventually lands heads" $= \bigcup_n \{\text{heads on toss } n\}$ or "the sample mean converges" — sets defined by limits.</p>
<p>Why not simply take $\mathcal F = 2^\Omega$, all subsets? For finite or countable $\Omega$ we do. For $\Omega = [0,1)$ we cannot, if we want the natural "length" measure:</p>
<div class="thm" data-kind="Theorem" data-name="Vitali (1905)">
<p>There is no function $\P$ defined on <em>all</em> subsets of $[0,1)$ that is countably additive, has $\P([0,1)) = 1$, and is invariant under translation modulo 1.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Call $x \sim y$ when $x - y \in \mathbb Q$. Using the axiom of choice, pick one representative from each equivalence class to form a set $V$. For each rational $q \in [0,1)$ let $V_q = \{(v + q) \bmod 1 : v \in V\}$. The $V_q$ are pairwise disjoint (two representatives in the same class would be equal) and their union is $[0,1)$ (every $x$ is equivalent to some $v$). Translation invariance forces $\P(V_q) = \P(V) = c$ for all $q$. Countable additivity over the countably many rationals gives $1 = \sum_q c$, which is $0$ if $c=0$ and $\infty$ if $c>0$. Contradiction.</p>
</details>
<p>So on continuous spaces we restrict to a σ-field that is rich enough for every set we will ever meet but excludes pathological ones. The standard choice on $\R$ is the <strong>Borel σ-field</strong> $\mathcal B(\R)$, the smallest σ-field containing all intervals. (The smallest σ-field containing a collection $\mathcal C$, written $\sigma(\mathcal C)$, exists because an intersection of σ-fields is a σ-field.) Every set you can describe by countably many unions, intersections and complements of intervals is Borel.</p>
<div class="ex" data-name="Information as a σ-field">
<p>Roll a die, $\Omega=\{1,\dots,6\}$, but you are only told whether the result is even. The events you can decide are $\mathcal G = \{\varnothing, \{2,4,6\}, \{1,3,5\}, \Omega\}$, the σ-field generated by the partition $\{\text{even},\text{odd}\}$. A partition into $k$ cells generates a σ-field with $2^k$ members. This reading of a σ-field as <em>available information</em> is exactly what <a href="#/t/conditional-expectation">conditional expectation</a> formalizes later.</p>
</div>`,
      },
      {
        title: 'The axioms and their first consequences',
        html: R`
<div class="def" data-name="Probability measure (Kolmogorov, 1933)">
<p>A <em>probability measure</em> on $(\Omega,\mathcal F)$ is a function $\P:\mathcal F \to \R$ with</p>
<ol>
<li>$\P(A) \ge 0$ for every $A \in \mathcal F$;</li>
<li>$\P(\Omega) = 1$;</li>
<li><strong>countable additivity:</strong> if $A_1, A_2, \dots \in \mathcal F$ are pairwise disjoint, $\P\big(\bigcup_{i} A_i\big) = \sum_{i} \P(A_i)$.</li>
</ol>
<p>The triple $(\Omega, \mathcal F, \P)$ is a <em>probability space</em>.</p>
</div>
<p>Everything else is derived. Taking $A_1 = \Omega$ and $A_i=\varnothing$ for $i \ge 2$ in axiom 3 forces $\P(\varnothing)=0$; then padding any finite disjoint list with empty sets gives finite additivity. From these:</p>
<div class="thm" data-kind="Proposition" data-name="Elementary rules">
<p>For events $A, B, A_1, A_2,\dots$:</p>
<ol>
<li>$\P(A^c) = 1 - \P(A)$;</li>
<li>$A \subseteq B \Rightarrow \P(A) \le \P(B)$, and $\P(B \setminus A) = \P(B) - \P(A)$;</li>
<li>$\P(A\cup B) = \P(A)+\P(B)-\P(A\cap B)$;</li>
<li><strong>Boole's inequality</strong> (union bound): $\P(\bigcup_i A_i) \le \sum_i \P(A_i)$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) $\Omega = A \cup A^c$ disjointly. (2) $B = A \cup (B\setminus A)$ disjointly, and $\P(B\setminus A)\ge0$. (3) $A\cup B = A \cup (B \setminus A)$ and $B = (A\cap B) \cup (B\setminus A)$, both disjoint; subtract. (4) Disjointify: $B_1 = A_1$, $B_i = A_i \setminus \bigcup_{j\lt i} A_j$. The $B_i$ are disjoint, $B_i \subseteq A_i$, and $\bigcup B_i = \bigcup A_i$, so $\P(\bigcup A_i) = \sum \P(B_i) \le \sum \P(A_i)$ by (2).</p>
</details>
<p>The union bound is crude but astonishingly useful: it needs no knowledge of how the events overlap. It is the engine behind Bonferroni corrections in <a href="#/t/multiple-testing">multiple testing</a> and behind uniform convergence arguments in <a href="#/t/vc-theory">learning theory</a>.</p>
<div class="thm" data-kind="Theorem" data-name="Inclusion–exclusion and Bonferroni bounds">
<p>Let $S_k = \sum_{i_1 \lt \cdots \lt i_k} \P(A_{i_1}\cap\cdots\cap A_{i_k})$ for events $A_1,\dots,A_n$. Then
$$\P\Big(\bigcup_{i=1}^n A_i\Big) = S_1 - S_2 + S_3 - \cdots + (-1)^{n+1} S_n,$$
and the partial sums alternate around the truth: truncating after an odd number of terms gives an upper bound, after an even number a lower bound.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The sets cut $\Omega$ into at most $2^n$ disjoint cells $\bigcap_i A_i^{\epsilon_i}$ (each $A_i^{\epsilon_i}$ is $A_i$ or $A_i^c$), and every quantity in the theorem is a sum of cell probabilities. Fix a cell that lies inside exactly $m \ge 1$ of the sets. Its probability is counted once on the left side and $\binom mk$ times in $S_k$ (once for each $k$-subset of its $m$ sets), so its total weight in the partial sum through $r$ terms is $\sum_{k=1}^r (-1)^{k+1}\binom mk$. The identity $\sum_{k=0}^{r}(-1)^k\binom mk = (-1)^r\binom{m-1}{r}$ (induct on $r$ using Pascal's rule) gives
$$\sum_{k=1}^r (-1)^{k+1}\binom mk = 1 - (-1)^r\binom{m-1}{r},$$
which equals $1$ when $r \ge m$, is $\ge 1$ for odd $r$ and $\le 1$ for even $r$. The cell lying in none of the sets contributes $0$ to both sides. Summing over cells gives the identity (take $r=n$) and both bounds.</p>
</details>`,
      },
      {
        title: 'Continuity: what countable additivity buys',
        html: R`
<p>Finite additivity would be enough for most finite calculations. Countable additivity is what makes probability behave well under limits, and it is equivalent to a continuity property.</p>
<div class="thm" data-kind="Theorem" data-name="Continuity of probability">
<p>If $A_1 \subseteq A_2 \subseteq \cdots$ and $A = \bigcup_n A_n$ (written $A_n \uparrow A$), then $\P(A_n) \uparrow \P(A)$. If $A_1 \supseteq A_2 \supseteq \cdots$ and $A = \bigcap_n A_n$, then $\P(A_n)\downarrow \P(A)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For the increasing case let $B_1=A_1$ and $B_k = A_k\setminus A_{k-1}$. These are disjoint, $\bigcup_{k\le n} B_k = A_n$ and $\bigcup_k B_k = A$. By countable additivity,
$$\P(A) = \sum_{k=1}^\infty \P(B_k) = \lim_{n\to\infty}\sum_{k=1}^n \P(B_k) = \lim_{n\to\infty}\P(A_n).$$
For the decreasing case apply this to the complements $A_n^c \uparrow A^c$ and use $\P(A^c) = 1-\P(A)$.</p>
</details>
<p>Conversely, a finitely additive $\P$ that is continuous from above at $\varnothing$ (i.e. $A_n \downarrow \varnothing \Rightarrow \P(A_n)\to0$) is countably additive. So the third axiom is precisely "probability is continuous".</p>
<div class="ex" data-name="A fair coin never stays tails forever">
<p>Toss a fair coin forever and let $A_n$ be "the first $n$ tosses are all tails", so $\P(A_n)=2^{-n}$. The event "tails forever" is $\bigcap_n A_n$, and continuity from above gives probability $\lim 2^{-n} = 0$. Note the event is not empty — the sequence TTT… is a legitimate outcome — so <strong>probability zero does not mean impossible</strong>. In continuous models every single outcome has probability zero, yet one of them occurs.</p>
</div>
<p>Continuity is used constantly downstream: it shows that a CDF is right-continuous (<a href="#/t/random-variables">random variables</a>), it underlies the Borel–Cantelli lemmas and almost-sure <a href="#/t/convergence">convergence</a>, and it is why expectations can be computed as limits of simple approximations.</p>`,
      },
      {
        title: 'Equally likely outcomes and counting',
        html: R`
<p>When $\Omega$ is finite and symmetry justifies treating outcomes as equally likely, $\P(A) = |A|/|\Omega|$ and probability becomes counting. The basic tool is the multiplication principle: a $k$-stage experiment with $n_j$ choices at stage $j$ has $n_1 n_2\cdots n_k$ outcomes. Drawing $k$ items from $n$ distinguishable items gives four counts:</p>
<table>
<thead><tr><th></th><th>Without replacement</th><th>With replacement</th></tr></thead>
<tbody>
<tr><td>Ordered</td><td>$n!/(n-k)!$</td><td>$n^k$</td></tr>
<tr><td>Unordered</td><td>$\binom nk$</td><td>$\binom{n+k-1}{k}$</td></tr>
</tbody>
</table>
<div class="warn" data-name="Counted ≠ equally likely">
<p>The bottom-right cell ("stars and bars") counts multisets, but under ordinary independent sampling those multisets are <em>not</em> equally likely. Two dice have $\binom{6+2-1}{2}=21$ unordered outcomes, yet $\{1,1\}$ has probability $1/36$ while $\{1,2\}$ has $2/36$. Using $1/21$ is the classic error (historically, physicists found that photons actually obey this "Bose–Einstein" counting; dice do not). Which outcomes are equally likely is a modelling assumption, not a fact about counting.</p>
</div>
<div class="ex" data-name="Shared birthdays">
<p>With $k$ people and 365 equally likely birthdays (ignoring leap years), the ordered without-replacement count gives
$$\P(\text{all distinct}) = \prod_{j=0}^{k-1}\Big(1-\frac{j}{365}\Big) \approx \exp\Big(-\frac{k(k-1)}{730}\Big),$$
using $\log(1-x)\approx -x$. The probability of a shared birthday passes $1/2$ at $k=23$. The intuition: what matters is the number of <em>pairs</em>, $\binom{23}{2}=253$, each with a $1/365$ chance of matching. The same quadratic-in-$k$ effect drives hash collisions and makes many simultaneous tests turn up spurious "coincidences".</p>
</div>
<div class="ex" data-name="The matching problem">
<p>Shuffle $n$ cards labelled $1,\dots,n$. Let $A_i$ be "card $i$ lands in position $i$". Any $k$ specified positions all match with probability $(n-k)!/n!$, and there are $\binom nk$ ways to choose them, so $S_k = 1/k!$. Inclusion–exclusion gives
$$\P(\text{at least one match}) = \sum_{k=1}^n \frac{(-1)^{k+1}}{k!} \;\xrightarrow[n\to\infty]{}\; 1 - e^{-1} \approx 0.632.$$
The answer is essentially independent of $n$ beyond $n\approx 6$, which surprises almost everyone. The number of matches converges in distribution to Poisson(1) — a first glimpse of the <a href="#/t/distribution-families">law of rare events</a>.</p>
</div>`,
        lab: 'probability-spaces-matching',
      },
      {
        title: 'Interpretations, and pitfalls',
        html: R`
<p>The axioms are deliberately silent about what probability <em>means</em>. Two readings dominate statistics, and both satisfy the axioms:</p>
<div class="cols">
<div><h4>Frequentist</h4><p>$\P(A)$ is the long-run relative frequency of $A$ in repetitions of the experiment. Natural for sampling and measurement error; awkward for one-off events. The <a href="#/t/lln">law of large numbers</a> is the theorem that makes this reading self-consistent.</p></div>
<div><h4>Subjective / Bayesian</h4><p>$\P(A)$ is a degree of belief, calibrated by willingness to bet. Coherence arguments (de Finetti's Dutch book) show beliefs must obey the axioms — at least finite additivity. This reading underlies <a href="#/t/bayesian-inference">Bayesian inference</a>.</p></div>
</div>
<div class="warn" data-name="Zero is not impossible, one is not certain">
<p>Events of probability zero can occur (any particular real number drawn from a continuous distribution). Statements that hold "with probability one" are said to hold <em>almost surely</em>; the qualifier matters, for instance in the strong law of large numbers.</p>
</div>
<div class="warn" data-name="Uncountable unions">
<p>Countable additivity says nothing about uncountable unions: $[0,1] = \bigcup_{x} \{x\}$ is a union of probability-zero sets with total probability 1. Arguments that sum over uncountably many events are invalid.</p>
</div>
<div class="warn" data-name="&quot;At random&quot; is not a model">
<p>Bertrand's chord, the "random point on a sphere", the "random integer": each requires an explicit choice of measure, and some natural-sounding requests (a uniform distribution on all integers) are impossible under countable additivity, since $\sum_n c$ cannot equal 1.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Show that a σ-field is closed under countable intersections, and explain why closure under countable (rather than finite) unions is needed for statistics.`,
        a: R`If $A_i \in \mathcal F$ then $A_i^c \in \mathcal F$, so $\bigcup A_i^c \in \mathcal F$ and its complement $\bigcap A_i$ is too (De Morgan). Countable operations are needed because the events statisticians care about are defined by limits: "$\bar X_n \to \mu$" is $\bigcap_k \bigcup_N \bigcap_{n\ge N}\{|\bar X_n-\mu|\lt 1/k\}$, which requires countable unions and intersections.` },
      { q: R`Use the union bound to show: if each of 20 hypothesis tests has false-rejection probability $0.0025$, the chance of at least one false rejection is at most $0.05$. When is this bound nearly tight?`,
        a: R`$\P(\bigcup_i A_i) \le \sum_i \P(A_i) = 20 \times 0.0025 = 0.05$. Inclusion–exclusion shows the gap is $S_2 - S_3 + \cdots$, so the bound is nearly tight when pairwise intersections are small — e.g. independent tests with small levels, where $S_2 \approx \binom{20}{2}(0.0025)^2 \approx 0.0012$.` },
      { q: R`Why is there no "uniform distribution on the positive integers"?`,
        a: R`Uniformity requires $\P(\{n\}) = c$ for every $n$. Countable additivity then gives $1 = \P(\N) = \sum_n c$, which is $0$ if $c=0$ and $\infty$ if $c\gt 0$. (Finitely additive "densities" such as natural density exist, but they are not probability measures in Kolmogorov's sense.)` },
      { q: R`In the matching problem, $\P(\text{no match}) \to e^{-1}$. Which property of $S_k$ makes the limit essentially independent of $n$?`,
        a: R`$S_k = \binom nk \frac{(n-k)!}{n!} = \frac1{k!}$ does not depend on $n$ at all; $n$ only determines where the alternating series $\sum_k (-1)^k/k!$ is truncated. Since $1/k!$ decays super-exponentially, the truncation error after $n$ terms is below $1/(n+1)!$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'conditional-probability',
    domain: 'foundations',
    order: 20,
    title: R`Conditional Probability, Independence & Bayes' Theorem`,
    subtitle: 'Updating a probability model on partial information',
    level: 1,
    sources: { aos: '1.6–1.7', cb: '§1.3', other: 'Gigerenzer & Hoffrage (1995) on natural frequencies; Pearl, Causality (2009) on Simpson' },
    prereqs: ['probability-spaces'],
    related: ['bayesian-inference', 'naive-bayes', 'conditional-expectation', 'causal-inference'],
    tags: ['Bayes', 'independence', 'base rate', 'Simpson'],
    summary: R`Conditioning on an event $B$ replaces the probability model by a new one in which $B$ is certain: restrict to $B$ and renormalize. From this single definition follow the multiplication rule, the law of total probability and Bayes' theorem, which inverts conditional probabilities and is the arithmetic of learning from evidence. Independence is the special case where conditioning changes nothing. The mathematics is elementary; the traps are not — base-rate neglect, confusing $\P(A\mid B)$ with $\P(B\mid A)$, ignoring how the information was generated (Monty Hall), and aggregation reversals (Simpson's paradox) catch experts routinely.`,
    survey: {
      task: 'Revise probabilities given that an event has occurred; invert conditional probabilities',
      assumptions: R`$\P(B) \gt 0$ for elementary conditioning; a correct model of how evidence was generated`,
      output: R`$\P(A\mid B)$, posterior probabilities and odds, independence structure`,
      strengths: 'Exact, mechanical, and the basis of all Bayesian reasoning and probabilistic classifiers',
      weaknesses: 'Garbage priors or a wrong data-generating story (protocol) give confidently wrong posteriors',
      complexity: R`Bayes over $k$ hypotheses: $O(k)$; joint models can be exponential in the number of variables`,
    },
    sections: [
      {
        title: 'Conditioning as renormalization',
        html: R`
<div class="def" data-name="Conditional probability">
<p>For events $A,B$ with $\P(B) \gt 0$,
$$\P(A \mid B) = \frac{\P(A\cap B)}{\P(B)}.$$</p>
</div>
<p>Read it geometrically: knowing $B$ occurred, the outcomes outside $B$ are eliminated; within $B$, outcomes keep their <em>relative</em> weights, so we rescale by $1/\P(B)$ to make the total mass 1 again. In particular, conditioning never changes the ratio $\P(A_1\cap B)/\P(A_2\cap B)$ of two outcomes consistent with $B$.</p>
<div class="thm" data-kind="Proposition" data-name="Conditioning yields a probability measure">
<p>For fixed $B$ with $\P(B)\gt0$, the map $A\mapsto \P(A\mid B)$ is a probability measure on $(\Omega,\mathcal F)$, concentrated on $B$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Nonnegativity is clear and $\P(\Omega\mid B) = \P(B)/\P(B) = 1$. If $A_1,A_2,\dots$ are disjoint, so are $A_i\cap B$, hence $\P(\bigcup A_i\mid B) = \P(\bigcup (A_i\cap B))/\P(B) = \sum \P(A_i \cap B)/\P(B) = \sum \P(A_i\mid B)$. Finally $\P(B^c\mid B)=0$.</p>
</details>
<p>Consequently every rule from <a href="#/t/probability-spaces">the axioms</a> holds conditionally: $\P(A^c\mid B) = 1-\P(A\mid B)$, conditional inclusion–exclusion, and so on. (But $\P(A \mid B^c) \ne 1 - \P(A\mid B)$ in general — the complement is taken on the left of the bar, not the right.)</p>
<p>Rearranging the definition gives the <strong>multiplication rule</strong> $\P(A\cap B) = \P(A\mid B)\P(B)$, and by induction the <strong>chain rule</strong>
$$\P(A_1\cap\cdots\cap A_n) = \P(A_1)\,\P(A_2\mid A_1)\,\P(A_3\mid A_1\cap A_2)\cdots \P(A_n\mid A_1\cap\cdots\cap A_{n-1}).$$
The chain rule is how joint distributions are built in practice: as a sequence of conditional steps. It is the factorization underlying <a href="#/t/markov-chains">Markov chains</a>, <a href="#/t/directed-graphical-models">directed graphical models</a> and autoregressive language models.</p>`,
      },
      {
        title: 'Total probability and Bayes\' theorem',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Law of total probability; Bayes' theorem">
<p>Let $B_1,B_2,\dots$ be a (finite or countable) partition of $\Omega$ with $\P(B_j)\gt0$. For any event $A$,
$$\P(A) = \sum_j \P(A\mid B_j)\,\P(B_j),$$
and if $\P(A)\gt0$,
$$\P(B_k\mid A) = \frac{\P(A\mid B_k)\,\P(B_k)}{\sum_j \P(A\mid B_j)\,\P(B_j)}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$A = \bigcup_j (A\cap B_j)$ disjointly, so by countable additivity and the multiplication rule $\P(A) = \sum_j \P(A\cap B_j) = \sum_j \P(A\mid B_j)\P(B_j)$. Bayes is then $\P(B_k\mid A) = \P(A\cap B_k)/\P(A)$ with numerator by the multiplication rule and denominator by total probability.</p>
</details>
<p>In inferential language the $B_j$ are hypotheses, $\P(B_j)$ the <strong>prior</strong>, $\P(A\mid B_j)$ the <strong>likelihood</strong> of the evidence $A$ under hypothesis $j$, and $\P(B_k\mid A)$ the <strong>posterior</strong>. The denominator is just the normalizing constant. For two hypotheses, dividing Bayes' formula for $H$ by that for $H^c$ cancels it:</p>
$$\underbrace{\frac{\P(H\mid A)}{\P(H^c\mid A)}}_{\text{posterior odds}} = \underbrace{\frac{\P(A\mid H)}{\P(A\mid H^c)}}_{\text{likelihood ratio}} \times \underbrace{\frac{\P(H)}{\P(H^c)}}_{\text{prior odds}}.$$
<p>On the log scale, evidence <em>adds</em>: each independent piece of evidence contributes its log-likelihood ratio. This is the form to carry in your head.</p>
<div class="ex" data-name="A screening test">
<p>A condition has prevalence $1\%$. A test has sensitivity $\P(+\mid D) = 0.95$ and specificity $\P(-\mid D^c) = 0.90$. Then
$$\P(D\mid +) = \frac{0.95 \times 0.01}{0.95\times0.01 + 0.10\times 0.99} = \frac{0.0095}{0.1085} \approx 0.088.$$
Fewer than 9% of positives are true positives. In odds form: prior odds $1:99$, likelihood ratio $0.95/0.10 = 9.5$, posterior odds $9.5:99 \approx 0.096$. In <em>natural frequencies</em>: of 1000 people, 10 are ill and about 9.5 test positive; of the 990 healthy, 99 test positive. A positive result puts you among about 108.5 people of whom 9.5 are ill.</p>
<p>A second, conditionally independent positive test multiplies the odds by 9.5 again: $0.096 \times 9.5 \approx 0.91$, i.e. $\P(D\mid ++) \approx 0.48$. The independence needed is <em>conditional</em> on disease status — two tests sharing a failure mode do not deliver this.</p>
</div>`,
        lab: 'conditional-probability-base-rate',
      },
      {
        title: 'Independence',
        html: R`
<div class="def" data-name="Independence">
<p>Events $A$ and $B$ are <em>independent</em> if $\P(A\cap B) = \P(A)\P(B)$. A family $\{A_i : i\in I\}$ is (mutually) independent if $\P(\bigcap_{i\in J} A_i) = \prod_{i\in J}\P(A_i)$ for <strong>every finite</strong> subset $J\subseteq I$.</p>
</div>
<p>When $\P(B)\gt0$ this is equivalent to $\P(A\mid B) = \P(A)$: learning $B$ does not change the probability of $A$. The product form is preferred as the definition because it is symmetric and makes sense even when $\P(B)=0$.</p>
<div class="thm" data-kind="Proposition" data-name="Complements preserve independence">
<p>If $A$ and $B$ are independent, so are $A$ and $B^c$, $A^c$ and $B$, and $A^c$ and $B^c$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\P(A\cap B^c) = \P(A) - \P(A\cap B) = \P(A) - \P(A)\P(B) = \P(A)(1-\P(B)) = \P(A)\P(B^c)$. Apply the same argument with roles swapped for the others.</p>
</details>
<div class="ex" data-name="Pairwise is not mutual">
<p>Toss two fair coins. Let $A$ = "first is heads", $B$ = "second is heads", $C$ = "the two coins agree". Each has probability $1/2$, and each pairwise intersection is the single outcome HH, probability $1/4 = \frac12\cdot\frac12$. So the events are pairwise independent. But $\P(A\cap B\cap C) = 1/4 \ne 1/8$: any two of them determine the third. Mutual independence requires checking all $2^n - n - 1$ intersection conditions, not just the pairs.</p>
</div>
<div class="warn" data-name="Disjoint is (almost) the opposite of independent">
<p>If $A\cap B=\varnothing$ and both have positive probability, then $\P(A\mid B) = 0 \ne \P(A)$: learning $B$ tells you $A$ did not happen. Disjoint events with positive probability are always dependent.</p>
</div>
<div class="def" data-name="Conditional independence">
<p>$A$ and $B$ are conditionally independent given $C$ (with $\P(C)\gt0$) if $\P(A\cap B\mid C) = \P(A\mid C)\P(B\mid C)$.</p>
</div>
<p>Neither independence nor conditional independence implies the other:</p>
<ul>
<li><strong>Conditionally independent but dependent.</strong> Pick one of two coins at random, with heads probabilities $0.9$ and $0.1$, and toss it twice. Given the coin, the tosses are independent; unconditionally, a first head makes the "0.9" coin more likely and hence raises the chance of a second head ($\P(H_2\mid H_1) = 0.82 \ne 0.5$).</li>
<li><strong>Independent but conditionally dependent.</strong> Two independent fair coins, and $C$ = "at least one head". Then $\P(H_1\mid C)=\P(H_2\mid C) = 2/3$ but $\P(H_1\cap H_2\mid C) = 1/3 \ne 4/9$. Conditioning on a common effect induces dependence ("explaining away", or collider bias in <a href="#/t/causal-inference">causal inference</a>).</li>
</ul>`,
      },
      {
        title: 'The protocol matters: Monty Hall',
        html: R`
<p>Conditioning on an event requires knowing <em>which</em> event you have learned — and that depends on the process that revealed the information, not just its content.</p>
<div class="ex" data-name="Monty Hall, done with Bayes">
<p>A car is behind one of three doors, uniformly. You pick door 1. The host, who knows where the car is, opens a door that is neither yours nor the car's, choosing at random if he has a choice. He opens door 3. Let $C$ be the car's location and $H$ the door opened:</p>
$$\P(H=3\mid C=1) = \tfrac12,\quad \P(H=3\mid C=2) = 1,\quad \P(H=3\mid C=3) = 0.$$
<p>With prior $1/3$ each, $\P(H=3) = \frac13(\frac12+1+0) = \frac12$, so
$$\P(C=1\mid H=3) = \frac{\frac12\cdot\frac13}{\frac12} = \frac13,\qquad \P(C=2\mid H=3) = \frac{1\cdot\frac13}{\frac12} = \frac23.$$
Switching wins with probability $2/3$. The host's action is evidence because it was <em>constrained</em> by the car's location: when the car is behind door 2 he had to open door 3, whereas with the car behind door 1 he only did so half the time.</p>
</div>
<p>Now change only the protocol. Suppose the host does not know where the car is and opens one of the other doors at random; it happens to show a goat. The observed event is "door 3 opened and showed a goat", with likelihoods $\frac12, \frac12, 0$ under $C = 1,2,3$. The posterior is now $\frac12$ vs $\frac12$: switching does not help. Same words, same door, different evidence — because the probability of the observation under each hypothesis is different.</p>
<div class="note"><p>With $n$ doors, a knowing host who opens $k \le n-2$ goat doors leaves your door with probability $1/n$ and each of the $n-1-k$ other closed doors with $\frac{n-1}{n}\cdot\frac{1}{n-1-k}$. An ignorant host who happens to reveal $k$ goats leaves <em>every</em> closed door with probability $1/(n-k)$. The lab lets you check both by simulation.</p></div>`,
        lab: 'conditional-probability-monty-hall',
      },
      {
        title: 'Pitfalls: inverted conditionals and aggregation',
        html: R`
<div class="warn" data-name="The prosecutor's fallacy">
<p>"The chance of this DNA match if the defendant were innocent is one in a million, so the chance he is innocent is one in a million." This confuses $\P(\text{match}\mid \text{innocent})$ with $\P(\text{innocent}\mid\text{match})$. Bayes' theorem needs the prior: in a city of ten million plausible sources, about ten innocent people would also match, so the evidence alone gives odds of roughly 1 to 10 for guilt. The same inversion error produces misreadings of p-values as probabilities that the null is true (see <a href="#/t/hypothesis-testing">hypothesis testing</a>).</p>
</div>
<div class="warn" data-name="Simpson's paradox">
<p>Conditional relationships can reverse on aggregation. Consider two treatments in mild and severe cases (invented numbers):</p>
<table>
<thead><tr><th></th><th>Mild</th><th>Severe</th><th>Overall</th></tr></thead>
<tbody>
<tr><td>Treatment A</td><td>90/100 = 90%</td><td>120/400 = 30%</td><td>210/500 = 42%</td></tr>
<tr><td>Treatment B</td><td>340/400 = 85%</td><td>20/100 = 20%</td><td>360/500 = 72%</td></tr>
</tbody>
</table>
<p>A is better within each severity stratum yet worse overall, because A was mostly given to severe cases. Algebraically, $\P(S\mid A) = \sum_g \P(S\mid A, g)\P(g\mid A)$: the overall rate is a mixture whose weights differ between treatments. Which comparison is "right" is not a probability question but a causal one: if severity affects both treatment choice and outcome, compare within strata (see <a href="#/t/causal-inference">causal inference</a>).</p>
</div>
<div class="warn" data-name="Conditioning on probability-zero events">
<p>$\P(A\mid B)$ is undefined when $\P(B)=0$, yet we constantly want "the distribution of $Y$ given $X = x$" for continuous $X$. Defining it requires densities or the measure-theoretic conditional expectation, and the answer can depend on how the null event is approached (the Borel–Kolmogorov paradox; see <a href="#/t/conditional-expectation">conditional expectation</a>).</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A test has sensitivity 0.99 and specificity 0.99. For what prevalence $\pi$ is $\P(D\mid +)=1/2$?`,
        a: R`Posterior odds = likelihood ratio × prior odds = $99 \cdot \pi/(1-\pi)$. Setting this to 1 gives $\pi = 1/100$. Even an excellent test yields a coin-flip posterior when the condition is as rare as the test's false-positive rate.` },
      { q: R`Can two events be independent of each other and also mutually exclusive?`,
        a: R`Only if at least one has probability zero: independence requires $\P(A\cap B) = \P(A)\P(B)$, and exclusivity makes the left side 0, so $\P(A)\P(B)=0$.` },
      { q: R`In Monty Hall, why does the host's knowledge change the answer even though in both versions you see a goat behind door 3?`,
        a: R`The posterior depends on the likelihood of the <em>observation</em> under each hypothesis. A knowing host must open door 3 when the car is behind door 2 (likelihood 1) but only half the time when it is behind door 1 (likelihood 1/2): a 2:1 likelihood ratio favouring door 2. An ignorant host opens door 3 and reveals a goat with probability 1/2 under both hypotheses: likelihood ratio 1, posterior unchanged.` },
      { q: R`Show that if $A$ and $B$ are conditionally independent given $C$ and given $C^c$, they need not be independent. Construct the example.`,
        a: R`Take $C$ = "biased coin chosen" in the two-coin example: given either coin the tosses are independent, but mixing over which coin was chosen creates dependence: $\P(H_1\cap H_2) = \frac12(0.81+0.01) = 0.41 \ne 0.25 = \P(H_1)\P(H_2)$. A shared latent cause induces marginal dependence.` },
      { q: R`In the Simpson's table, compute $\P(\text{Severe}\mid A)$ and $\P(\text{Severe}\mid B)$ and use them to explain the reversal.`,
        a: R`$\P(\text{Severe}\mid A) = 400/500 = 0.8$ and $\P(\text{Severe}\mid B) = 100/500 = 0.2$. A's overall rate is $0.2(0.9) + 0.8(0.3) = 0.42$; B's is $0.8(0.85)+0.2(0.2) = 0.72$. The overall rates average stratum-specific rates with very different weights, and severity has a far larger effect on recovery than the treatment does.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'random-variables',
    domain: 'foundations',
    order: 30,
    title: 'Random Variables, CDFs & Densities',
    subtitle: 'Numbers attached to outcomes, and the distribution functions that describe them',
    level: 1,
    sources: { aos: '2.1–2.4', cb: '§1.4–1.6', other: 'Billingsley (1995) §12, §14; Durrett, Probability: Theory and Examples, §1.2' },
    prereqs: ['probability-spaces'],
    related: ['distribution-families', 'transformations', 'inverse-transform', 'empirical-cdf'],
    tags: ['CDF', 'pmf', 'pdf', 'quantile function', 'probability integral transform'],
    summary: R`A random variable is a function $X:\Omega\to\R$ that is measurable, so that statements like $\{X\le x\}$ are events with probabilities. Its distribution is fully captured by the cumulative distribution function $F(x)=\P(X\le x)$, and the functions that can arise as CDFs are exactly the nondecreasing, right-continuous functions running from 0 to 1. Discrete variables have probability mass functions, absolutely continuous ones have densities, and some have neither. The generalized inverse of $F$ — the quantile function — turns a uniform random number into any distribution and, run backwards, turns any continuous variable into a uniform one.`,
    survey: {
      task: 'Describe the distribution of a numerical quantity; move between CDF, pmf/pdf and quantiles',
      assumptions: R`Measurability of $X$; for densities, absolute continuity of $F$`,
      output: R`$F$, $p$ or $f$, quantile function $F^{-1}$; probabilities of intervals`,
      strengths: 'The CDF always exists and determines the law; quantiles give universal simulation (inverse transform)',
      weaknesses: R`Densities need not exist (singular laws); a density is not a probability and may exceed 1`,
      complexity: R`$O(1)$ per evaluation for closed forms; numerical inversion for $F^{-1}$ otherwise`,
    },
    sections: [
      {
        title: 'Random variables and their distributions',
        html: R`
<p>Outcomes $\omega$ are often complicated objects — a sequence of coin tosses, a patient, a sentence. Statistics works with numbers computed from them.</p>
<div class="def" data-name="Random variable">
<p>A <em>random variable</em> on $(\Omega,\mathcal F,\P)$ is a function $X : \Omega \to \R$ such that $\{\omega : X(\omega)\le x\} \in \mathcal F$ for every $x\in\R$. Equivalently, $X^{-1}(B)\in\mathcal F$ for every Borel set $B$.</p>
</div>
<p>The measurability requirement is what makes $\P(X\le x)$ meaningful; in practice every function you will write down is measurable. (The equivalence holds because the sets $B$ with $X^{-1}(B)\in\mathcal F$ form a σ-field, and if it contains the half-lines $(-\infty,x]$ it contains $\sigma(\text{half-lines}) = \mathcal B(\R)$.)</p>
<p>A random variable transports $\P$ from $\Omega$ to the real line: its <strong>distribution</strong> (or law) is the probability measure $P_X(B) = \P(X\in B) = \P(X^{-1}(B))$ on $(\R,\mathcal B)$. Once we have $P_X$, the original $\Omega$ is usually irrelevant — which is why statisticians rarely mention it.</p>
<div class="ex" data-name="Two dice">
<p>$\Omega = \{1,\dots,6\}^2$ with 36 equally likely outcomes; $X(\omega) = \omega_1+\omega_2$. Then $\P(X=k) = (6-|k-7|)/36$ for $k=2,\dots,12$. Many different $\omega$'s map to the same value: the distribution of $X$ forgets which ones.</p>
</div>
<div class="warn" data-name="Equal in distribution is not equal">
<p>$X \overset{d}{=} Y$ means $P_X = P_Y$; it says nothing about $X(\omega)$ and $Y(\omega)$ being close. If $Z\sim\N(0,1)$ then $-Z \overset{d}{=} Z$, yet $Z - (-Z) = 2Z$ is far from zero. Results "in distribution" (like the <a href="#/t/clt">CLT</a>) are statements about laws, not about the variables themselves.</p>
</div>`,
      },
      {
        title: 'The cumulative distribution function',
        html: R`
<div class="def" data-name="CDF">
<p>The <em>cumulative distribution function</em> of $X$ is $F_X(x) = \P(X\le x)$, $x\in\R$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Characterization of CDFs">
<p>A function $F:\R\to[0,1]$ is the CDF of some random variable if and only if (i) $F$ is nondecreasing; (ii) $\lim_{x\to-\infty} F(x) = 0$ and $\lim_{x\to\infty}F(x) = 1$; (iii) $F$ is right-continuous: $F(x) = \lim_{y\downarrow x}F(y)$. Moreover, the CDF determines the distribution: $F_X = F_Y$ implies $P_X = P_Y$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Necessity.</em> (i) $x\le y$ gives $\{X\le x\}\subseteq\{X\le y\}$. (iii) If $y_n \downarrow x$ then $\{X\le y_n\}\downarrow\{X\le x\}$, so $F(y_n)\to F(x)$ by <a href="#/t/probability-spaces">continuity from above</a>. (ii) $\{X\le -n\}\downarrow\varnothing$ and $\{X\le n\}\uparrow\Omega$, using that $X$ is real-valued.</p>
<p><em>Sufficiency.</em> Given such an $F$, take $\Omega = (0,1)$ with Lebesgue measure and $X(u) = F^{-1}(u)$, the quantile function defined below; the lemma in the next-but-one section shows $\P(X\le x) = F(x)$.</p>
<p><em>Uniqueness.</em> $F$ fixes $P_X$ on intervals $(a,b]$, which form a collection closed under finite intersection that generates $\mathcal B$. Two probability measures agreeing on such a "π-system" agree on the σ-field it generates (Dynkin's π–λ theorem).</p>
</details>
<p>Every probability of interest can be read from $F$:</p>
$$\P(a \lt X\le b) = F(b)-F(a),\qquad \P(X \lt x) = F(x^-),\qquad \P(X=x) = F(x) - F(x^-),$$
<p>where $F(x^-) = \lim_{y\uparrow x}F(y)$ (by continuity from below, since $\{X\le y\}\uparrow\{X\lt x\}$). So jumps of $F$ are exactly the atoms of the distribution. A nondecreasing function has at most countably many jumps (at most $k$ jumps of size exceeding $1/k$), so every distribution has at most countably many atoms.</p>`,
      },
      {
        title: 'Mass functions, densities, and neither',
        html: R`
<div class="cols">
<div><h4>Discrete</h4><p>$X$ takes values in a countable set $\{x_1,x_2,\dots\}$. Its <em>probability mass function</em> is $p(x) = \P(X=x)$, with $\sum_i p(x_i)=1$. $F$ is a step function jumping $p(x_i)$ at each $x_i$.</p></div>
<div><h4>Absolutely continuous</h4><p>There is $f\ge0$ with $F(x) = \int_{-\infty}^x f(t)\d t$. Then $f$ is the <em>density</em>; $f = F'$ wherever $F$ is differentiable (almost everywhere), and $\P(X\in B) = \int_B f$.</p></div>
</div>
<p>A density is not a probability. It is probability <em>per unit length</em>: $\P(x \lt X\le x+h) \approx f(x)h$ for small $h$ when $f$ is continuous at $x$. Densities can exceed 1 (Uniform$(0, 0.1)$ has density 10) and can even be unbounded (Beta$(\frac12,\frac12)$). And a density is only defined up to sets of measure zero: changing $f$ at countably many points changes no probability.</p>
<div class="def" data-name="Mixed distributions">
<p>Many real quantities are neither discrete nor continuous. Daily rainfall is exactly 0 with positive probability and continuous otherwise: $F = w\,\ind\{x\ge0\} + (1-w)G$ with $G$ continuous. Insurance claims, censored lifetimes and zero-inflated counts behave similarly.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Lebesgue decomposition">
<p>Every CDF can be written uniquely as $F = \alpha_d F_d + \alpha_{ac} F_{ac} + \alpha_s F_s$ with nonnegative weights summing to 1, where $F_d$ is discrete, $F_{ac}$ is absolutely continuous, and $F_s$ is <em>singular continuous</em>: continuous, yet its derivative is 0 almost everywhere.</p>
</div>
<p>The singular part is not a curiosity of pathology alone. The Cantor distribution — write $X = \sum_{k\ge1} 2B_k/3^k$ with $B_k$ independent fair bits — has a continuous CDF (no atoms), but $X$ lives on the Cantor set, which has length zero, so no density can integrate to its probabilities. Such laws are the reason the general theory is phrased with CDFs and measures rather than densities.</p>
<p>The lab below builds a mixed distribution: drag the atom weight and watch the jump in $F$ — and see how the quantile function handles it.</p>`,
        lab: 'random-variables-cdf',
      },
      {
        title: 'Quantiles and the probability integral transform',
        html: R`
<p>When $F$ is continuous and strictly increasing, it has an ordinary inverse. In general it has flat stretches (no mass) and jumps (atoms), and we need a convention.</p>
<div class="def" data-name="Quantile function (generalized inverse)">
<p>$F^{-1}(u) = \inf\{x : F(x) \ge u\}$ for $u\in(0,1)$. The median is $F^{-1}(1/2)$; the quartiles are $F^{-1}(1/4), F^{-1}(3/4)$.</p>
</div>
<div class="thm" data-kind="Lemma" data-name="Galois inequality">
<p>For $u\in(0,1)$ and $x\in\R$: $\;F^{-1}(u)\le x \iff u\le F(x)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The set $\{y : F(y)\ge u\}$ is nonempty and bounded below (because $F\to1$ and $F\to0$ at $\pm\infty$), and it is an up-set by monotonicity. By right-continuity it contains its infimum, so it equals $[F^{-1}(u),\infty)$. Hence $F(x)\ge u$ iff $x$ belongs to it iff $x\ge F^{-1}(u)$.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="Inverse transform and probability integral transform">
<p>(a) If $U\sim\text{Unif}(0,1)$ then $X = F^{-1}(U)$ has CDF $F$ — for <em>any</em> CDF $F$.<br>(b) If $X$ has a <em>continuous</em> CDF $F$, then $F(X)\sim\text{Unif}(0,1)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(a) By the lemma, $\P(F^{-1}(U)\le x) = \P(U\le F(x)) = F(x)$.</p>
<p>(b) Fix $u\in(0,1)$ and let $q = F^{-1}(u)$. By definition of the infimum, $x\lt q$ implies $F(x)\lt u$, while $x\ge q$ implies $F(x)\ge F(q)\ge u$. So $\{F(X)\lt u\} = \{X\lt q\}$, and $\P(F(X)\lt u) = F(q^-) = F(q) = u$, using continuity of $F$ and then $F(q) = u$ (continuity again: $F$ cannot jump over $u$). A variable with $\P(V\lt u) = u$ for all $u \in (0,1)$ is uniform.</p>
</details>
<p>Part (a) is the basis of <a href="#/t/inverse-transform">inverse transform sampling</a>: a uniform random number generator plus a quantile function simulates anything. Part (b), the <strong>probability integral transform</strong> (PIT), is the basis of goodness-of-fit diagnostics: if a model $G$ is correct, $G(X_1),\dots,G(X_n)$ should look uniform; systematic departures (U-shapes, humps, slopes) diagnose under-dispersion, over-dispersion and bias. For discrete $X$, $F(X)$ is <em>not</em> uniform — it takes only countably many values — but the <em>randomized</em> PIT $V = F(X^-) + W\,[F(X)-F(X^-)]$ with independent $W\sim\text{Unif}(0,1)$ restores exact uniformity.</p>
<div class="ex" data-name="Exponential via inverse transform">
<p>$F(x) = 1-e^{-\lambda x}$ gives $F^{-1}(u) = -\log(1-u)/\lambda$; since $1-U\sim U$, $-\log(U)/\lambda \sim \text{Exp}(\lambda)$. For a Bernoulli$(p)$, $F^{-1}(u) = \ind\{u \gt 1-p\}$: the generalized inverse handles the jumps automatically.</p>
</div>`,
        lab: 'random-variables-pit',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Left versus right limits">
<p>For discrete variables $\P(X\lt x)$ and $\P(X\le x)$ differ by the atom at $x$. Statistical software disagrees on which tail a p-value includes for discrete tests; always check. With the convention $F(x)=\P(X\le x)$, $\P(X\ge k) = 1-F(k-1)$ for integer-valued $X$.</p>
</div>
<div class="warn" data-name="&quot;Continuous random variable&quot;">
<p>Textbooks often say "continuous" to mean "has a density". The precise term is <em>absolutely continuous</em>; a continuous CDF only rules out atoms. The Cantor distribution is continuous but has no density, and results requiring densities (change of variables, likelihoods) do not apply to it.</p>
</div>
<div class="warn" data-name="Sample quantiles have conventions">
<p>The quantile function of the empirical CDF, $\hat F_n^{-1}(u)$, is a step function; software offers many interpolating alternatives (R has nine "types"). They agree asymptotically but can differ noticeably for small $n$ — relevant for bootstrap percentile intervals and Q–Q plots. See <a href="#/t/empirical-cdf">the empirical CDF</a> and <a href="#/t/order-statistics">order statistics</a>.</p>
</div>
<div class="warn" data-name="Densities on transformed scales">
<p>The density of $X$ at its mode is not the density of $g(X)$ at $g(\text{mode})$; densities change by a Jacobian factor under reparameterization (see <a href="#/t/transformations">transformations</a>). Modes are not invariant; quantiles are: the median of $g(X)$ is $g(\text{median of } X)$ for increasing $g$.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Let $F(x) = 0$ for $x\lt0$, $F(x) = 0.3 + 0.7(1-e^{-x})$ for $x\ge0$. Find $\P(X=0)$, $\P(X \gt 1)$ and the median.`,
        a: R`$\P(X=0) = F(0)-F(0^-) = 0.3$. $\P(X\gt1) = 1-F(1) = 0.7e^{-1}\approx 0.258$. The median solves $0.3+0.7(1-e^{-m}) = 0.5$, i.e. $e^{-m} = 5/7$, $m = \log(7/5)\approx0.336$.` },
      { q: R`Why must a CDF be right-continuous but need not be left-continuous?`,
        a: R`Right-continuity follows from continuity from above: $\{X\le x+1/n\}\downarrow\{X\le x\}$. Approaching from the left, $\{X\le x-1/n\}\uparrow\{X\lt x\}$, so $F(x^-) = \P(X\lt x)$, which differs from $F(x)$ by $\P(X=x)$. The asymmetry comes from the convention $\le$ in the definition.` },
      { q: R`If $X\sim$ Poisson(3), is $F(X)$ uniform? What would you do to build a PIT histogram for a count model?`,
        a: R`No: $F(X)$ takes only the countably many values $F(0), F(1),\dots$, so its distribution is discrete. Use the randomized PIT $V = F(X-1) + W\,p(X)$ with $W\sim$ Unif(0,1) independent, which is exactly uniform when the model is correct (or the non-randomized "mid-PIT" histogram of Czado, Gneiting & Held, 2009).` },
      { q: R`Can a density exceed 1? Can a pmf?`,
        a: R`A density can (e.g. Uniform$(0,1/2)$ has $f = 2$), because it is probability per unit length; only its integral is constrained. A pmf cannot: $p(x) = \P(X=x)\le1$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'distribution-families',
    domain: 'foundations',
    order: 40,
    title: 'The Common Distribution Families',
    subtitle: 'The standard parametric models, the stories behind them, and how they connect',
    level: 1,
    sources: { aos: '2.3–2.4', cb: '§3.1–3.3', other: 'Johnson, Kotz & Balakrishnan, Continuous Univariate Distributions; Leemis & McQueston (2008), "Univariate distribution relationships"' },
    prereqs: ['random-variables'],
    related: ['exponential-families', 'mgf', 'transformations', 'poisson-process', 'sampling-distributions'],
    tags: ['binomial', 'Poisson', 'gamma', 'beta', 'normal', 'law of rare events'],
    summary: R`A handful of parametric families account for most of applied probability, and each earns its place through a <em>generative story</em>: the binomial counts successes in independent trials, the Poisson counts rare events, the exponential is the only memoryless waiting time, the gamma sums exponentials, the beta describes proportions and uniform order statistics, the normal arises from sums. Knowing the stories tells you when a family is appropriate; knowing the relationships among them — limits, sums, ratios, transformations — turns a catalogue into a connected map and supplies most of the distributional results used in inference.`,
    survey: {
      task: 'Choose a parametric model for counts, waiting times, proportions or measurements',
      assumptions: 'The generative story (independence, constant rate, fixed trials, …) approximately holds',
      output: 'pmf/pdf, CDF, moments; distributional identities between families',
      strengths: 'Few parameters, interpretable, closed-form moments; many exact sampling results',
      weaknesses: 'Real data are often overdispersed, heavy-tailed or mixed relative to the textbook family',
      complexity: R`Closed-form densities; CDFs may need special functions ($\Gamma$, incomplete beta)`,
    },
    sections: [
      {
        title: 'Discrete families and their stories',
        html: R`
<table>
<thead><tr><th>Family</th><th>pmf</th><th>Support</th><th>Mean</th><th>Variance</th></tr></thead>
<tbody>
<tr><td>Bernoulli$(p)$</td><td>$p^x(1-p)^{1-x}$</td><td>$\{0,1\}$</td><td>$p$</td><td>$p(1-p)$</td></tr>
<tr><td>Binomial$(n,p)$</td><td>$\binom nx p^x(1-p)^{n-x}$</td><td>$0,\dots,n$</td><td>$np$</td><td>$np(1-p)$</td></tr>
<tr><td>Geometric$(p)$ (trials)</td><td>$(1-p)^{x-1}p$</td><td>$1,2,\dots$</td><td>$1/p$</td><td>$(1-p)/p^2$</td></tr>
<tr><td>Neg. binomial$(r,p)$ (failures)</td><td>$\binom{x+r-1}{x}p^r(1-p)^x$</td><td>$0,1,\dots$</td><td>$r(1-p)/p$</td><td>$r(1-p)/p^2$</td></tr>
<tr><td>Hypergeometric$(N,K,n)$</td><td>$\binom Kx\binom{N-K}{n-x}/\binom Nn$</td><td>$\max(0,n{-}N{+}K),\dots,\min(n,K)$</td><td>$nK/N$</td><td>$n\frac KN\frac{N-K}N\frac{N-n}{N-1}$</td></tr>
<tr><td>Poisson$(\lambda)$</td><td>$e^{-\lambda}\lambda^x/x!$</td><td>$0,1,\dots$</td><td>$\lambda$</td><td>$\lambda$</td></tr>
</tbody>
</table>
<ul>
<li><strong>Binomial</strong>: the number of successes in $n$ independent trials with common success probability $p$ — a sum of $n$ IID Bernoullis. The two assumptions (independence, constant $p$) are the ones that fail in practice; failure typically shows up as variance larger than $np(1-p)$ (<a href="#/t/overdispersion">overdispersion</a>).</li>
<li><strong>Geometric</strong>: the trial on which the first success occurs. It is <em>memoryless</em>: $\P(X\gt m+k\mid X\gt m) = (1-p)^{m+k}/(1-p)^m = \P(X\gt k)$. Having waited does not bring success closer.</li>
<li><strong>Negative binomial</strong>: failures before the $r$-th success, a sum of $r$ independent (failure-count) geometrics. It also arises as a Poisson whose rate is gamma-distributed, which is why it is the default model for overdispersed counts; there $r$ need not be an integer.</li>
<li><strong>Hypergeometric</strong>: successes in a sample of $n$ drawn <em>without replacement</em> from $N$ items of which $K$ are successes. The factor $\frac{N-n}{N-1}$ is the <em>finite-population correction</em>: sampling without replacement induces negative dependence among draws and reduces variance. As $N\to\infty$ with $K/N\to p$ it tends to Binomial$(n,p)$.</li>
<li><strong>Poisson</strong>: counts of events that are individually rare but have many opportunities to occur — the content of the next theorem. Mean equals variance, a sharp, testable property.</li>
</ul>
<div class="warn" data-name="Parameterizations differ">
<p>"Geometric" may count trials (support $1,2,\dots$) or failures (support $0,1,\dots$); "negative binomial" may count failures or trials, and is parameterized by $(r,p)$ or by mean and dispersion. Gamma uses shape–scale or shape–rate. Always check the support and the mean before plugging a formula into code.</p>
</div>`,
      },
      {
        title: 'The law of rare events',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Poisson limit of the binomial">
<p>If $X_n\sim\text{Binomial}(n,p_n)$ with $np_n\to\lambda\in(0,\infty)$, then for each fixed $k$, $\P(X_n=k)\to e^{-\lambda}\lambda^k/k!$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $\lambda_n = np_n$. Then
$$\P(X_n=k) = \frac{\lambda_n^k}{k!}\cdot\underbrace{\frac{n(n-1)\cdots(n-k+1)}{n^k}}_{\to 1}\cdot\underbrace{\Big(1-\frac{\lambda_n}{n}\Big)^{n}}_{\to e^{-\lambda}}\cdot\underbrace{\Big(1-\frac{\lambda_n}{n}\Big)^{-k}}_{\to1}.$$
The second factor is a product of $k$ terms each tending to 1; the third uses $n\log(1-\lambda_n/n) = -\lambda_n + O(\lambda_n^2/n)\to-\lambda$.</p>
</details>
<p>The theorem is far more robust than its proof suggests. Neither identical probabilities nor exact independence is needed:</p>
<div class="thm" data-kind="Theorem" data-name="Le Cam's inequality (1960)">
<p>Let $S = \sum_{i=1}^n B_i$ with independent $B_i\sim\text{Bernoulli}(p_i)$ and $\lambda = \sum_i p_i$. Then
$$\sum_{k=0}^\infty\Big|\P(S=k) - \frac{e^{-\lambda}\lambda^k}{k!}\Big| \le 2\sum_{i=1}^n p_i^2.$$</p>
</div>
<p>So the total-variation distance $d_{TV} = \frac12\sum_k|\cdot|$ is at most $\sum p_i^2$; for the binomial this is $np^2 = \lambda^2/n$, an $O(1/n)$ rate. The proof is by coupling: build each $B_i$ together with a Poisson$(p_i)$ variable so that they disagree with probability at most $p_i^2$, then apply the union bound. Stein–Chen methods extend the result to weakly dependent indicators, which is why Poisson approximations work for matches in a shuffled deck (the number of fixed points of a random permutation is approximately Poisson(1)), for birthday coincidences, and for counts of rare words in long texts.</p>
<p>Contrast the <em>normal</em> approximation to Binomial$(n,p)$ with $p$ fixed, which has Kolmogorov-distance error of order $1/\sqrt{np(1-p)}$ (Berry–Esseen). The lab below plots both error rates against $n$ on a log–log scale: slopes of $-1$ and $-1/2$.</p>`,
        lab: 'distribution-families-limits',
      },
      {
        title: 'Continuous families and their stories',
        html: R`
<table>
<thead><tr><th>Family</th><th>Density</th><th>Support</th><th>Mean</th><th>Variance</th></tr></thead>
<tbody>
<tr><td>Uniform$(a,b)$</td><td>$\frac1{b-a}$</td><td>$(a,b)$</td><td>$\frac{a+b}2$</td><td>$\frac{(b-a)^2}{12}$</td></tr>
<tr><td>Normal$(\mu,\sigma^2)$</td><td>$\frac1{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$</td><td>$\R$</td><td>$\mu$</td><td>$\sigma^2$</td></tr>
<tr><td>Exponential (rate $\lambda$)</td><td>$\lambda e^{-\lambda x}$</td><td>$(0,\infty)$</td><td>$1/\lambda$</td><td>$1/\lambda^2$</td></tr>
<tr><td>Gamma$(\alpha,\beta)$ (scale $\beta$)</td><td>$\frac{x^{\alpha-1}e^{-x/\beta}}{\Gamma(\alpha)\beta^\alpha}$</td><td>$(0,\infty)$</td><td>$\alpha\beta$</td><td>$\alpha\beta^2$</td></tr>
<tr><td>Beta$(a,b)$</td><td>$\frac{x^{a-1}(1-x)^{b-1}}{B(a,b)}$</td><td>$(0,1)$</td><td>$\frac a{a+b}$</td><td>$\frac{ab}{(a+b)^2(a+b+1)}$</td></tr>
<tr><td>Lognormal$(\mu,\sigma^2)$</td><td>$\frac{1}{x\sigma\sqrt{2\pi}}e^{-(\log x-\mu)^2/2\sigma^2}$</td><td>$(0,\infty)$</td><td>$e^{\mu+\sigma^2/2}$</td><td>$(e^{\sigma^2}-1)e^{2\mu+\sigma^2}$</td></tr>
<tr><td>Student $t_\nu$</td><td>$\propto(1+x^2/\nu)^{-(\nu+1)/2}$</td><td>$\R$</td><td>$0$ ($\nu\gt1$)</td><td>$\frac\nu{\nu-2}$ ($\nu\gt2$)</td></tr>
<tr><td>Cauchy$(0,1)$</td><td>$\frac{1}{\pi(1+x^2)}$</td><td>$\R$</td><td>undefined</td><td>undefined</td></tr>
<tr><td>Pareto$(\alpha,x_m)$</td><td>$\alpha x_m^\alpha x^{-\alpha-1}$</td><td>$(x_m,\infty)$</td><td>$\frac{\alpha x_m}{\alpha-1}$ ($\alpha\gt1$)</td><td>$\frac{\alpha x_m^2}{(\alpha-1)^2(\alpha-2)}$ ($\alpha\gt2$)</td></tr>
</tbody>
</table>
<p>Here $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1}e^{-t}\d t$ (with $\Gamma(n) = (n-1)!$ and $\Gamma(\frac12)=\sqrt\pi$) and $B(a,b) = \Gamma(a)\Gamma(b)/\Gamma(a+b)$. The chi-square with $k$ degrees of freedom is Gamma$(k/2, 2)$.</p>
<ul>
<li><strong>Normal</strong>: the limit of standardized sums (<a href="#/t/clt">CLT</a>), and the maximum-entropy law for given mean and variance. Its tails decay like $e^{-x^2/2}$, so it is a poor model for anything with outliers. About 68.3%, 95.4% and 99.7% of mass lies within 1, 2, 3 standard deviations.</li>
<li><strong>Exponential</strong>: waiting time with a constant hazard (instantaneous failure rate). It is the continuous memoryless law — see the theorem below — and the inter-arrival distribution of the <a href="#/t/poisson-process">Poisson process</a>.</li>
<li><strong>Gamma</strong>: for integer $\alpha$, the waiting time until the $\alpha$-th event of a Poisson process, i.e. a sum of $\alpha$ independent exponentials. Shape $\alpha\lt1$ gives a density that is infinite at 0; $\alpha\gt1$ a unimodal hump; large $\alpha$ looks normal.</li>
<li><strong>Beta</strong>: the natural family on $[0,1]$. The $k$-th smallest of $n$ IID uniforms is Beta$(k,n-k+1)$ (see <a href="#/t/order-statistics">order statistics</a>), and the beta is conjugate to the binomial in <a href="#/t/bayesian-inference">Bayesian inference</a>.</li>
<li><strong>Lognormal</strong>: products of many positive factors (by the CLT applied to logs). Right-skewed with all moments finite, yet no MGF — a key counterexample in the <a href="#/t/mgf">MGF</a> topic.</li>
<li><strong>Student $t$, Cauchy, Pareto</strong>: polynomial tails. $t_\nu$ has finite moments only of order $\lt\nu$; Cauchy ($t_1$) has no mean; Pareto moments exist only below $\alpha$. See <a href="#/t/expectation">expectation</a> for what this does to sample averages.</li>
</ul>
<div class="thm" data-kind="Theorem" data-name="Memorylessness characterizes the exponential">
<p>If $X\gt0$ has a continuous distribution with $\P(X\gt s+t\mid X\gt s) = \P(X\gt t)$ for all $s,t\ge0$, then $X\sim\text{Exp}(\lambda)$ for some $\lambda\gt0$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $G(t) = \P(X\gt t)$. The hypothesis says $G(s+t) = G(s)G(t)$. Then $G(m/n) = G(1/n)^m$ and $G(1) = G(1/n)^n$, so $G(q) = G(1)^q$ for every positive rational $q$. $G(1)\in(0,1)$: if $G(1)=0$ then $G(1/n)=0$ for all $n$ so $\P(X\gt0)=0$, a contradiction; if $G(1)=1$ then $G\equiv1$ on the rationals, impossible for a proper distribution. Set $\lambda = -\log G(1)\gt0$. $G$ is monotone and (being a survival function of a continuous law) continuous, so $G(t) = e^{-\lambda t}$ for all real $t\ge0$ by density of the rationals.</p>
</details>`,
      },
      {
        title: 'The web of relationships',
        html: R`
<p>Most distributional facts used in inference are relationships between these families. Each line below is a theorem; several are proved in <a href="#/t/transformations">transformations</a> and <a href="#/t/mgf">MGFs</a>.</p>
<table>
<thead><tr><th>Relationship</th><th>Mechanism</th></tr></thead>
<tbody>
<tr><td>$\sum_{i=1}^n \text{Bern}(p) = \text{Bin}(n,p)$</td><td>sum of IID</td></tr>
<tr><td>$\text{Poi}(\lambda_1)+\text{Poi}(\lambda_2) = \text{Poi}(\lambda_1+\lambda_2)$</td><td>independent sum (MGF)</td></tr>
<tr><td>$\sum_{i=1}^k\text{Exp}(\lambda) = \text{Gamma}(k, 1/\lambda)$</td><td>independent sum</td></tr>
<tr><td>$\text{Gamma}(\alpha,\beta)+\text{Gamma}(\alpha',\beta) = \text{Gamma}(\alpha+\alpha',\beta)$</td><td>same scale</td></tr>
<tr><td>$X/(X+Y)\sim\text{Beta}(\alpha,\alpha')$, independent of $X+Y$</td><td>$X\sim\text{Gamma}(\alpha,1)$, $Y\sim \text{Gamma}(\alpha',1)$</td></tr>
<tr><td>$Z^2\sim\chi^2_1$; $\sum_{i=1}^k Z_i^2\sim\chi^2_k$</td><td>squares of standard normals</td></tr>
<tr><td>$Z/\sqrt{V/\nu}\sim t_\nu$</td><td>$Z\indep V\sim\chi^2_\nu$</td></tr>
<tr><td>$(V_1/d_1)/(V_2/d_2)\sim F_{d_1,d_2}$</td><td>independent chi-squares</td></tr>
<tr><td>$t_1 = $ Cauchy; $Z_1/Z_2\sim$ Cauchy</td><td>ratio of normals</td></tr>
<tr><td>$-\log U\sim\text{Exp}(1)$; $e^{N(\mu,\sigma^2)}\sim$ Lognormal</td><td>monotone transform</td></tr>
<tr><td>$\text{Bin}(n,\lambda/n)\to\text{Poi}(\lambda)$; $t_\nu\to\N(0,1)$; $\text{Gamma}(\alpha,1)$ standardized $\to\N(0,1)$</td><td>limits</td></tr>
<tr><td>Poisson$(\Lambda)$ with $\Lambda\sim$ Gamma = negative binomial</td><td>mixture</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Proposition" data-name="Gamma–Poisson duality">
<p>For integer $k\ge1$ and $\lambda,t\gt0$: $\;\P(\text{Gamma}(k,1/\lambda)\le t) = \P(\text{Poisson}(\lambda t)\ge k)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $T_k$ be the $k$-th arrival time of a rate-$\lambda$ Poisson process and $N(t)$ the number of arrivals in $[0,t]$. Then $T_k\sim\text{Gamma}(k,1/\lambda)$ (a sum of $k$ exponential inter-arrival times), $N(t)\sim\text{Poisson}(\lambda t)$, and the events $\{T_k\le t\}$ and $\{N(t)\ge k\}$ are identical. (An analytic proof: differentiate $\sum_{j\ge k}e^{-\lambda t}(\lambda t)^j/j!$ in $t$; the sum telescopes to the Gamma density $\lambda^k t^{k-1}e^{-\lambda t}/(k-1)!$.)</p>
</details>
<p>A similar identity links binomial tails to beta CDFs: $\P(\text{Bin}(n,p)\ge k) = \P(\text{Beta}(k,n-k+1)\le p)$, because the $k$-th smallest of $n$ uniforms is below $p$ exactly when at least $k$ of them are. This is how exact (Clopper–Pearson) <a href="#/t/confidence-intervals">binomial confidence intervals</a> are computed.</p>
<p>Use the explorer to see each family's pmf/pdf and CDF side by side, and toggle a normal with matched mean and variance to see where the normal approximation is and is not adequate.</p>`,
        lab: 'distribution-families-explorer',
      },
      {
        title: 'Choosing and checking a family',
        html: R`
<p>A family is a hypothesis about mechanism, and each comes with testable fingerprints:</p>
<table>
<thead><tr><th>Family</th><th>Fingerprint</th><th>Typical failure</th></tr></thead>
<tbody>
<tr><td>Poisson</td><td>variance = mean</td><td>variance ≫ mean (clustering, heterogeneity) → negative binomial</td></tr>
<tr><td>Binomial</td><td>variance = $np(1-p)$</td><td>correlated trials or varying $p$ → beta-binomial</td></tr>
<tr><td>Exponential</td><td>constant hazard; $\text{sd} = \text{mean}$</td><td>wear-out or burn-in → Weibull, gamma</td></tr>
<tr><td>Normal</td><td>skewness 0, kurtosis 3</td><td>heavy tails → $t$; skew → log-transform</td></tr>
</tbody>
</table>
<div class="ex" data-name="Counts that are not Poisson">
<p>Suppose accident counts per driver have mean 0.5 and variance 0.9. A Poisson model would understate uncertainty by nearly half. If each driver has their own rate $\Lambda$ drawn from a Gamma with mean $\mu$ and variance $\tau^2$, then by the <a href="#/t/conditional-expectation">law of total variance</a> the count has variance $\mu + \tau^2$; matching gives $\tau^2 = 0.4$. The resulting marginal is negative binomial — overdispersion explained as unobserved heterogeneity.</p>
</div>
<div class="warn" data-name="The normal approximation lives in the centre">
<p>Approximations like Binomial ≈ Normal are accurate in <em>absolute</em> error near the centre but can be off by orders of magnitude in <em>relative</em> error in the tails, exactly where rare-event probabilities, p-values and risk measures live. A normal approximation to $\P(\text{Bin}(100,0.02)\ge 8)$ is poor; Poisson(2) is far better.</p>
</div>
<div class="warn" data-name="Mean and variance do not pin down a family">
<p>Matching two moments makes the normal approximation look plausible but says nothing about tails or skew. Compare whole distributions — Q–Q plots, PIT histograms (see <a href="#/t/random-variables">random variables</a>) — not just summaries.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A website gets on average 3 sign-ups per hour. Assuming a Poisson process, what is the probability that the third sign-up after noon happens before 1 pm?`,
        a: R`By gamma–Poisson duality, $\P(T_3\le1) = \P(N(1)\ge3)$ with $N(1)\sim$ Poisson(3): $1 - e^{-3}(1+3+4.5) = 1-8.5e^{-3}\approx 0.577$.` },
      { q: R`Use Le Cam's inequality to bound the total-variation error of approximating Binomial$(1000, 0.002)$ by Poisson(2).`,
        a: R`$d_{TV}\le np^2 = 1000\times(0.002)^2 = 0.004$. Every event's probability is approximated to within 0.004.` },
      { q: R`Why is the hypergeometric variance smaller than the binomial variance with $p = K/N$?`,
        a: R`Sampling without replacement makes draws negatively correlated: drawing a success leaves fewer successes. $\Var = np(1-p)\frac{N-n}{N-1}$, and the factor is $\lt1$ for $n\gt1$; it is 0 when $n=N$ (a census has no sampling variability).` },
      { q: R`Show that if $X\sim$ Geometric$(p)$ (trials) then $\P(X\gt m+k\mid X\gt m) = \P(X\gt k)$. Why can no other distribution on $\{1,2,\dots\}$ have this property?`,
        a: R`$\P(X\gt j) = (1-p)^j$, so the ratio is $(1-p)^{m+k}/(1-p)^m = (1-p)^k$. Conversely, memorylessness gives $G(m+k) = G(m)G(k)$ for $G(j) = \P(X\gt j)$, hence $G(j) = G(1)^j$; setting $1-p = G(1)$ forces the geometric.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'transformations',
    domain: 'foundations',
    order: 50,
    title: 'Transformations & Change of Variables',
    subtitle: 'Finding the distribution of a function of random variables',
    level: 2,
    sources: { aos: '2.11–2.12', cb: '§2.1, §4.3', other: 'Rudin, Real and Complex Analysis, Thm 7.26 (change of variables)' },
    prereqs: ['random-variables'],
    related: ['distribution-families', 'multivariate', 'transformation-methods', 'inverse-transform', 'delta-method'],
    tags: ['Jacobian', 'CDF method', 'convolution', 'Box–Muller'],
    summary: R`If $X$ has a known distribution and $Y = g(X)$, what is the distribution of $Y$? The fully general answer is the CDF method: compute $\P(g(X)\le y)$ by finding the set of $x$ that map below $y$. For smooth invertible maps this collapses to the change-of-variables formula, where densities pick up a Jacobian factor because probability is conserved while lengths (areas, volumes) are stretched. The same idea, in several dimensions, gives the convolution formula for sums, the Cauchy as a ratio of normals, Box–Muller sampling, and the independence of gamma sums and beta proportions.`,
    survey: {
      task: R`Distribution of $Y=g(X)$, or of sums, products, ratios, maxima of random variables`,
      assumptions: R`For the density formula: $g$ differentiable and (piecewise) invertible; $X$ has a density`,
      output: R`$F_Y$ or $f_Y(y) = \sum_{x: g(x)=y} f_X(x)/|g'(x)|$`,
      strengths: 'Exact; the multivariate version handles sums, ratios and coordinate changes',
      weaknesses: 'Requires solving for preimages; non-invertible maps need auxiliary variables or the CDF method',
      complexity: R`Analytic; numerically $O(\text{grid})$ per evaluation for convolution integrals`,
    },
    sections: [
      {
        title: 'The CDF method',
        html: R`
<p>Let $Y = g(X)$. For any $y$,
$$F_Y(y) = \P(g(X)\le y) = \P\big(X\in g^{-1}((-\infty,y])\big).$$
This works for any $g$ and any $X$ — discrete, continuous or mixed. The only work is describing the preimage set. If $X$ is discrete, it is even simpler: $p_Y(y) = \sum_{x : g(x)=y} p_X(x)$.</p>
<div class="ex" data-name="Square of a standard normal">
<p>$Y = Z^2$ with $Z\sim\N(0,1)$. For $y\gt0$, $F_Y(y) = \P(-\sqrt y\le Z\le\sqrt y) = 2\Phi(\sqrt y)-1$. Differentiating,
$$f_Y(y) = 2\phi(\sqrt y)\cdot\frac{1}{2\sqrt y} = \frac{1}{\sqrt{2\pi y}}e^{-y/2},\quad y\gt0,$$
which is the Gamma$(\frac12, 2)$ = $\chi^2_1$ density. Note the $y^{-1/2}$ singularity at 0: squaring compresses a neighbourhood of 0, piling up probability.</p>
</div>
<div class="ex" data-name="Maximum and minimum">
<p>For IID $X_1,\dots,X_n$ with CDF $F$: $\{\max_i X_i\le y\} = \bigcap_i\{X_i\le y\}$, so $F_{\max}(y) = F(y)^n$; similarly $\P(\min_i X_i\gt y) = (1-F(y))^n$. For Exp$(\lambda)$ the minimum is Exp$(n\lambda)$ — competing exponential clocks — and for Uniform$(0,1)$ the maximum is Beta$(n,1)$. The general theory is <a href="#/t/order-statistics">order statistics</a>; the extreme cases drive <a href="#/t/evt-intro">extreme value theory</a>.</p>
</div>
<div class="ex" data-name="Log of a uniform">
<p>$Y = -\log U$ for $U\sim$ Unif$(0,1)$: $\P(Y\le y) = \P(U\ge e^{-y}) = 1-e^{-y}$. So $Y\sim\text{Exp}(1)$, and $-\frac1\lambda\log U\sim$ Exp$(\lambda)$ — the simplest instance of <a href="#/t/inverse-transform">inverse transform sampling</a>.</p>
</div>`,
      },
      {
        title: 'The univariate change-of-variables formula',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Change of variables (monotone case)">
<p>Let $X$ have density $f_X$, supported on an open interval $I$, and let $g:I\to\R$ be strictly monotone and continuously differentiable with $g'\ne0$ on $I$. Then $Y = g(X)$ has density
$$f_Y(y) = f_X\big(g^{-1}(y)\big)\,\left|\frac{\d}{\d y}g^{-1}(y)\right| = \frac{f_X(x)}{|g'(x)|}\Big|_{x=g^{-1}(y)},\qquad y\in g(I),$$
and $f_Y(y)=0$ outside $g(I)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>If $g$ is increasing, $F_Y(y) = \P(X\le g^{-1}(y)) = F_X(g^{-1}(y))$; differentiate with the chain rule. If $g$ is decreasing, $F_Y(y) = \P(X\ge g^{-1}(y)) = 1-F_X(g^{-1}(y))$ (no atoms), whose derivative is $-f_X(g^{-1}(y))\,(g^{-1})'(y)$, which is positive because $(g^{-1})'\lt0$. Both cases are captured by the absolute value. The inverse function theorem gives $(g^{-1})'(y) = 1/g'(g^{-1}(y))$.</p>
</details>
<div class="note"><p>Conservation of probability: the mass in a small interval $[x, x+\d x]$ is $f_X(x)\,\d x$, and it lands in an interval of length $|g'(x)|\,\d x$ around $y=g(x)$. Equating masses, $f_Y(y)\,|g'(x)|\,\d x = f_X(x)\d x$. Where $g$ stretches ($|g'|$ large), density thins out; where $g$ compresses, density piles up. The Jacobian factor is the stretch.</p></div>
<div class="thm" data-kind="Corollary" data-name="Piecewise-monotone maps">
<p>If $I$ splits into finitely many intervals $I_1,\dots,I_m$ on each of which $g$ is strictly monotone and $C^1$ with nonvanishing derivative, then
$$f_Y(y) = \sum_{j : y\in g(I_j)} \frac{f_X(x_j(y))}{|g'(x_j(y))|},$$
where $x_j(y)$ is the unique preimage of $y$ in $I_j$. (Finitely many points where $g'=0$ do not matter: they carry no probability.)</p>
</div>
<p>The $\chi^2_1$ example is this corollary with two branches $x=\pm\sqrt y$, each contributing $\phi(\sqrt y)/(2\sqrt y)$.</p>
<div class="ex" data-name="Location–scale families">
<p>For $Y = a + bX$ with $b\ne0$: $f_Y(y) = \frac{1}{|b|}f_X\big(\frac{y-a}{b}\big)$. Every normal, Cauchy, logistic or uniform is a location–scale transform of a standard one, and this formula is why densities in such families have the form $\frac1\sigma f_0(\frac{x-\mu}{\sigma})$.</p>
</div>
<p>The lab pushes samples through $g$. Switch on the naive formula (no Jacobian) to see that $f_X(g^{-1}(y))$ alone is not even a density: its integral over the window is not the probability of the window.</p>`,
        lab: 'transformations-jacobian',
      },
      {
        title: 'Several variables: the Jacobian determinant',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Multivariate change of variables">
<p>Let $\bX$ have density $f_{\bX}$ on an open set $A\subseteq\R^k$, and let $g:A\to B$ be a bijection onto an open set $B$, continuously differentiable with continuously differentiable inverse $h = g^{-1}$. Then $\bY = g(\bX)$ has density
$$f_{\bY}(\by) = f_{\bX}(h(\by))\,\big|\det J_h(\by)\big|,\qquad J_h(\by) = \Big[\frac{\partial h_i}{\partial y_j}(\by)\Big]_{i,j},$$
for $\by\in B$, and 0 elsewhere. Equivalently, $|\det J_h(\by)| = 1/|\det J_g(h(\by))|$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>For any Borel $C\subseteq B$, $\P(\bY\in C) = \P(\bX\in h(C)) = \int_{h(C)}f_{\bX}(\bx)\d\bx$. The substitution $\bx = h(\by)$ in the multivariable integral (the analytic change-of-variables theorem) turns this into $\int_C f_{\bX}(h(\by))\,|\det J_h(\by)|\d\by$. Since this holds for every $C$, the integrand is the density. Geometrically, $|\det J_h|$ is the factor by which $h$ scales small volumes.</p>
</details>
<p>If $g$ maps $\R^k\to\R^m$ with $m\lt k$ (e.g. we only want $Y_1 = X_1+X_2$), complete it to a bijection by adding auxiliary coordinates (say $Y_2 = X_2$), apply the theorem, then integrate the auxiliaries out.</p>
<div class="ex" data-name="Linear maps">
<p>$\bY = A\bX + \mathbf b$ with $A$ invertible: $f_{\bY}(\by) = f_{\bX}(A^{-1}(\by-\mathbf b))/|\det A|$. Applied to a standard normal vector this gives the <a href="#/t/multivariate">multivariate normal</a> density with $\Sigma = AA^\top$.</p>
</div>
<div class="ex" data-name="Polar coordinates and Box–Muller">
<p>Let $X,Y\iid\N(0,1)$ and write $X = R\cos\Theta$, $Y = R\sin\Theta$. The inverse map $(r,\theta)\mapsto(r\cos\theta, r\sin\theta)$ has Jacobian determinant $r$, so
$$f_{R,\Theta}(r,\theta) = \frac{1}{2\pi}e^{-r^2/2}\cdot r = \underbrace{\frac{1}{2\pi}}_{\Theta\sim\text{Unif}(0,2\pi)}\cdot\underbrace{re^{-r^2/2}}_{R\text{ Rayleigh}},\quad r\gt0,\ \theta\in(0,2\pi).$$
The density factorizes, so $R$ and $\Theta$ are independent, and $R^2\sim\text{Exp}(\text{rate } \tfrac12)=\chi^2_2$. Reversing: with $U_1,U_2$ uniform, $R = \sqrt{-2\log U_1}$, $\Theta = 2\pi U_2$ produces two independent standard normals (Box & Muller, 1958). See <a href="#/t/transformation-methods">transformation methods</a>.</p>
</div>
<div class="ex" data-name="Gamma sums and beta proportions">
<p>Let $X\sim\text{Gamma}(\alpha,1)$ and $Y\sim\text{Gamma}(\beta,1)$ be independent, and set $U = X+Y$, $V = X/(X+Y)$. The inverse is $x = uv$, $y = u(1-v)$ with $|\det J| = |{-uv} - u(1-v)| = u$. Then
$$f_{U,V}(u,v) = \frac{(uv)^{\alpha-1}e^{-uv}}{\Gamma(\alpha)}\frac{(u(1-v))^{\beta-1}e^{-u(1-v)}}{\Gamma(\beta)}\,u = \underbrace{\frac{u^{\alpha+\beta-1}e^{-u}}{\Gamma(\alpha+\beta)}}_{\text{Gamma}(\alpha+\beta,1)}\cdot\underbrace{\frac{v^{\alpha-1}(1-v)^{\beta-1}}{B(\alpha,\beta)}}_{\text{Beta}(\alpha,\beta)}.$$
So $U$ and $V$ are independent, with the stated laws. This explains the Dirichlet construction (normalize independent gammas) and is the core of the independence of $\bar X$ and $S^2$-type results (see <a href="#/t/sampling-distributions">sampling distributions</a>).</p>
</div>`,
      },
      {
        title: 'Sums, products and ratios',
        html: R`
<p>Applying the multivariate theorem with an auxiliary coordinate gives integral formulas for the most common combinations of independent $X$ and $Y$ with densities $f_X, f_Y$:</p>
<table>
<thead><tr><th>$Z$</th><th>Density of $Z$</th><th>Auxiliary map</th></tr></thead>
<tbody>
<tr><td>$X+Y$</td><td>$\int f_X(x)f_Y(z-x)\d x$ (convolution)</td><td>$(x,y)\mapsto(x, x+y)$, $|J|=1$</td></tr>
<tr><td>$X-Y$</td><td>$\int f_X(x)f_Y(x-z)\d x$</td><td>$(x,y)\mapsto(x,x-y)$</td></tr>
<tr><td>$XY$</td><td>$\int f_X(x)f_Y(z/x)\frac{1}{|x|}\d x$</td><td>$(x,y)\mapsto(x,xy)$, $|J_h| = 1/|x|$</td></tr>
<tr><td>$X/Y$</td><td>$\int |y|\,f_X(zy)f_Y(y)\d y$</td><td>$(x,y)\mapsto(x/y,y)$, $|J_h|=|y|$</td></tr>
<tr><td>$\max(X,Y)$</td><td>$f_X(z)F_Y(z)+F_X(z)f_Y(z)$</td><td>CDF method</td></tr>
<tr><td>$\min(X,Y)$</td><td>$f_X(z)(1-F_Y(z)) + (1-F_X(z))f_Y(z)$</td><td>CDF method</td></tr>
</tbody>
</table>
<div class="ex" data-name="Two uniforms make a triangle">
<p>$X,Y\iid$ Unif$(0,1)$: $f_{X+Y}(z) = \int_0^1\ind\{0\lt z-x\lt1\}\d x = $ length of $[0,1]\cap(z-1,z)$, which is $z$ on $[0,1]$ and $2-z$ on $[1,2]$. Each further convolution smooths: the sum of $n$ uniforms is a piecewise polynomial of degree $n-1$ (Irwin–Hall), already visually normal at $n=3$ or 4.</p>
</div>
<div class="ex" data-name="A ratio of normals is Cauchy">
<p>$X,Y\iid\N(0,1)$, $Z = X/Y$:
$$f_Z(z) = \int_{-\infty}^\infty|y|\,\phi(zy)\phi(y)\d y = \frac{2}{2\pi}\int_0^\infty y\,e^{-y^2(1+z^2)/2}\d y = \frac{1}{\pi(1+z^2)}.$$
Two perfectly well-behaved variables produce one with no mean. Near $Y=0$ the ratio explodes, and the tails decay only like $z^{-2}$. The same phenomenon plagues ratio estimators (e.g. of a cost-effectiveness ratio) whenever the denominator can be near zero.</p>
</div>
<p>The lab computes these integrals numerically for several pairs of families and compares them with simulated draws.</p>`,
        lab: 'transformations-algebra',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Forgetting the Jacobian">
<p>The most common error: writing $f_Y(y) = f_X(g^{-1}(y))$. It is not a density (it generally does not integrate to 1), and it is not a matter of normalization either: the shape is wrong wherever $|g'|$ varies. Example: if $X\sim$ Unif$(0,1)$ and $Y = X^2$, the naive formula gives $f_Y\equiv1$ on $(0,1)$, whereas the truth is $1/(2\sqrt y)$, piling up near zero.</p>
</div>
<div class="warn" data-name="Densities are not invariant; modes move">
<p>Because of the Jacobian, the mode of $g(X)$ is generally not $g(\text{mode of }X)$. This matters for MAP estimation in Bayesian statistics: the posterior mode depends on the parameterization, whereas posterior quantiles and the <a href="#/t/mle">MLE</a> are equivariant.</p>
</div>
<div class="warn" data-name="Expectation does not commute with g">
<p>You rarely need $f_Y$ to compute $\E g(X)$: the law of the unconscious statistician gives $\E g(X) = \int g(x)f_X(x)\d x$ directly (see <a href="#/t/expectation">expectation</a>). But $\E g(X)\ne g(\E X)$ unless $g$ is affine; Jensen's inequality says which way the error goes for convex $g$, and the <a href="#/t/delta-method">delta method</a> says how large it is approximately.</p>
</div>
<div class="warn" data-name="Supports and branches">
<p>Track the support. For $Y = 1/X$ with $X\sim\N(0,1)$, the map has two monotone branches (negative and positive $x$), and the density of $Y$ is $\phi(1/y)/y^2$, which vanishes at 0 and has Cauchy-like tails. For non-injective maps, missing a branch loses probability mass; double-counting one creates it.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Find the density of $Y = e^X$ when $X\sim\N(\mu,\sigma^2)$.`,
        a: R`$g^{-1}(y) = \log y$, $|(g^{-1})'(y)| = 1/y$, so $f_Y(y) = \frac{1}{y\sigma\sqrt{2\pi}}\exp\{-(\log y-\mu)^2/2\sigma^2\}$ for $y\gt0$: the lognormal.` },
      { q: R`If $U\sim$ Unif$(0,1)$, what is the distribution of $\tan(\pi(U-\tfrac12))$, and why?`,
        a: R`Standard Cauchy. The map is increasing from $(0,1)$ onto $\R$ with inverse $u = \frac12+\frac1\pi\arctan y$, whose derivative is $\frac{1}{\pi(1+y^2)}$. Equivalently it is $F^{-1}(U)$ for the Cauchy CDF.` },
      { q: R`In the gamma–beta example, why does the factorization of $f_{U,V}$ imply independence, and what would break it?`,
        a: R`A joint density that factors as $a(u)b(v)$ on a product set (here $(0,\infty)\times(0,1)$) gives independence. It depends on both gammas having the <em>same scale</em>: with different scales the exponent $-uv/\theta_1 - u(1-v)/\theta_2$ couples $u$ and $v$.` },
      { q: R`For independent $X,Y\sim$ Exp(1), derive the density of $X-Y$.`,
        a: R`$f(z) = \int_0^\infty e^{-x}e^{-(x-z)}\ind\{x-z\gt0\}\d x$. For $z\ge0$: $e^{z}\int_z^\infty e^{-2x}\d x = \frac12e^{-z}$. For $z\lt0$: $e^{z}\int_0^\infty e^{-2x}\d x = \frac12e^{z}$. So $f(z) = \frac12e^{-|z|}$, the Laplace distribution.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'expectation',
    domain: 'foundations',
    order: 60,
    title: 'Expectation, Variance & Moments',
    subtitle: 'Averages under a distribution: when they exist, how they combine, and what they hide',
    level: 1,
    sources: { aos: '3.1–3.4', cb: '§2.2–2.3', other: 'Durrett (2019) §1.6; Williams, Probability with Martingales, Ch. 5–6' },
    prereqs: ['random-variables'],
    related: ['mgf', 'inequalities', 'lln', 'conditional-expectation', 'bias-variance'],
    tags: ['linearity', 'variance', 'covariance', 'Jensen', 'heavy tails'],
    summary: R`The expectation $\E X$ is the probability-weighted average of $X$ — the centre of mass of its distribution — and it exists only when the weighted sum of $|X|$ converges. Linearity, which holds with no independence assumption, makes expectation the single most useful computational tool in probability. Variance and covariance measure spread and co-movement; Cauchy–Schwarz bounds correlation by 1; Jensen's inequality controls nonlinear functions. When tails are heavy enough that moments do not exist, sample averages misbehave in ways no finite-variance intuition prepares you for.`,
    survey: {
      task: 'Summarize location, spread and dependence; compute averages of functions of random variables',
      assumptions: R`$\E|X|\lt\infty$ for the mean; $\E X^2\lt\infty$ for variance and covariance`,
      output: R`$\E X$, $\E g(X)$, $\Var X$, $\Cov(X,Y)$, higher moments`,
      strengths: 'Linearity with no independence needed; indicators turn probabilities into expectations',
      weaknesses: 'May not exist (Cauchy, Pareto); sensitive to tails; summaries hide shape',
      complexity: 'Sums/integrals; often closed form via linearity or LOTUS',
    },
    sections: [
      {
        title: 'Definition and existence',
        html: R`
<div class="def" data-name="Expectation">
<p>For discrete $X$, $\E X = \sum_x x\,p(x)$; for $X$ with density $f$, $\E X = \int x f(x)\d x$; in general $\E X = \int x\d F(x)$, a Lebesgue–Stieltjes integral. The expectation <strong>exists</strong> (is finite) when $\E|X| = \int|x|\d F(x)\lt\infty$. More generally, with $X^+ = \max(X,0)$, $X^- = \max(-X,0)$, define $\E X = \E X^+ - \E X^-$ whenever at least one of them is finite.</p>
</div>
<p>Absolute convergence is not pedantry. A conditionally convergent series can be rearranged to sum to anything (Riemann), and the outcomes of a random variable have no natural order, so we insist on a definition that does not depend on one.</p>
<div class="ex" data-name="Two expectations that fail">
<p><strong>Cauchy.</strong> $\int_0^\infty x\cdot\frac{1}{\pi(1+x^2)}\d x = \infty$ and likewise for the negative part, so $\E X$ is undefined (not zero, despite symmetry). The principal value $\lim_{a\to\infty}\int_{-a}^a$ is $0$, but $\lim\int_{-a}^{2a}$ is $\frac{\log 4}{2\pi}$: the answer depends on how you take the limit.</p>
<p><strong>St. Petersburg.</strong> Win $2^k$ if the first head appears on toss $k$: $\E X = \sum_k 2^k 2^{-k} = \infty$. Here $\E X = +\infty$ is well-defined, but no finite entry price makes the game fair in any practical sense.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Law of the unconscious statistician (LOTUS)">
<p>For measurable $g$, $\E g(X) = \sum_x g(x)p(x)$ or $\int g(x)f(x)\d x$, whenever $\E|g(X)|\lt\infty$. No need to find the distribution of $g(X)$ first.</p>
</div>
<details class="proof"><summary>Proof (discrete case)</summary>
<p>Let $Y=g(X)$. Grouping the $x$'s by the value of $g$, $\sum_y y\,\P(Y=y) = \sum_y y\sum_{x:g(x)=y}p(x) = \sum_y\sum_{x:g(x)=y}g(x)p(x) = \sum_x g(x)p(x)$; the regrouping is justified by absolute convergence. The density case follows from the change-of-variables formula for smooth monotone $g$, and in general from the abstract change-of-variables theorem $\int g\d P_X = \int g\circ X\d\P$.</p>
</details>`,
      },
      {
        title: 'Linearity and the indicator method',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Properties of expectation">
<p>If the expectations exist: (i) $\E(aX+bY) = a\E X+b\E Y$; (ii) $X\le Y$ a.s. implies $\E X\le\E Y$; (iii) if $X$ and $Y$ are independent, $\E(XY) = \E X\,\E Y$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>(i) For discrete $(X,Y)$ with joint pmf $p$: $\E(aX+bY) = \sum_{x,y}(ax+by)p(x,y) = a\sum_x x\,p_X(x) + b\sum_y y\,p_Y(y)$, marginalizing each sum. The general case follows by approximating with simple random variables and passing to the limit (monotone convergence). Crucially, <em>no independence</em> is used. (ii) $\E(Y-X)\ge0$ for a nonnegative variable. (iii) With densities, $\E XY = \iint xy f_X(x)f_Y(y)\d x\d y = \int xf_X\int yf_Y$ by Fubini, justified because $\E|XY| = \E|X|\E|Y|\lt\infty$ by the same computation applied to $|x||y|$.</p>
</details>
<p>Since $\E\ind_A = \P(A)$, any count written as a sum of indicators has expectation equal to the sum of probabilities, however dependent the indicators are. This is the <strong>indicator method</strong>, and it solves problems that are hopeless by direct enumeration.</p>
<div class="ex" data-name="Fixed points of a random permutation">
<p>Let $N$ be the number of fixed points of a uniformly random permutation of $n$ items. $N = \sum_i\ind_{A_i}$ with $\P(A_i) = 1/n$, so $\E N = 1$ for every $n$. For the variance, $\E N^2 = \sum_i\P(A_i) + \sum_{i\ne j}\P(A_i\cap A_j) = 1 + n(n-1)\cdot\frac{1}{n(n-1)} = 2$, so $\Var N = 1$ — matching the Poisson(1) limit from the <a href="#/t/probability-spaces">matching problem</a>.</p>
</div>
<div class="ex" data-name="Coupon collector">
<p>To collect all $n$ coupon types, let $T_k$ be the number of draws needed to get a new type when $k-1$ are already held; $T_k\sim$ Geometric$(\frac{n-k+1}{n})$ with mean $\frac{n}{n-k+1}$. By linearity, the expected total is $n\sum_{j=1}^n\frac1j = nH_n\approx n\log n$. (The $T_k$ happen to be independent here, but linearity did not need that.)</p>
</div>`,
      },
      {
        title: 'Variance, covariance and moments',
        html: R`
<div class="def" data-name="Variance and covariance">
<p>For $\E X^2\lt\infty$ with $\mu = \E X$: $\Var X = \E(X-\mu)^2 = \E X^2-\mu^2$. For $X,Y$ with finite second moments, $\Cov(X,Y) = \E[(X-\mu_X)(Y-\mu_Y)] = \E XY - \mu_X\mu_Y$, and $\Corr(X,Y) = \Cov(X,Y)/(\sigma_X\sigma_Y)$.</p>
</div>
<p>Covariance is bilinear and symmetric, $\Var X = \Cov(X,X)$, and therefore
$$\Var\Big(\sum_i a_iX_i\Big) = \sum_i a_i^2\Var X_i + 2\sum_{i\lt j}a_ia_j\Cov(X_i,X_j).$$
With uncorrelated summands the cross terms vanish: $\Var(\bar X_n) = \sigma^2/n$ for IID data, the most-used formula in statistics. With positively correlated data (time series, clustered samples) the cross terms add up and naive standard errors are too small.</p>
<div class="thm" data-kind="Theorem" data-name="Cauchy–Schwarz; correlation bound">
<p>$(\E XY)^2\le\E X^2\,\E Y^2$. Consequently $|\Corr(X,Y)|\le1$, with equality iff $Y = a+bX$ a.s. for some constants with $b\ne0$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For every real $t$, $0\le\E(Y-tX)^2 = \E Y^2 - 2t\E XY + t^2\E X^2$. A nonnegative quadratic in $t$ has nonpositive discriminant: $4(\E XY)^2 - 4\E X^2\E Y^2\le0$. (If $\E X^2 = 0$ then $X=0$ a.s. and both sides vanish.) Applying this to the centred variables gives $|\Cov(X,Y)|\le\sigma_X\sigma_Y$. Equality means the discriminant is 0, so for some $t_0$, $\E(Y-\mu_Y - t_0(X-\mu_X))^2=0$, i.e. the centred $Y$ is exactly $t_0$ times the centred $X$ almost surely.</p>
</details>
<div class="warn" data-name="Uncorrelated is not independent">
<p>Correlation measures <em>linear</em> association only. If $X\sim\N(0,1)$ and $Y = X^2$, then $\Cov(X,Y) = \E X^3 = 0$, yet $Y$ is a deterministic function of $X$. Independence implies zero covariance (when it exists), never the converse — except for jointly normal vectors (see <a href="#/t/multivariate">multivariate normal</a>).</p>
</div>
<p>Higher moments describe shape: the <em>skewness</em> $\E(X-\mu)^3/\sigma^3$ and the <em>kurtosis</em> $\E(X-\mu)^4/\sigma^4$ (3 for the normal; "excess kurtosis" subtracts 3). Moments are nested:</p>
<div class="thm" data-kind="Corollary" data-name="Lyapunov's inequality">
<p>For $0\lt p\lt q$: $(\E|X|^p)^{1/p}\le(\E|X|^q)^{1/q}$. In particular, a finite $q$-th moment implies finite moments of all lower orders.</p>
</div>
<p>(Proof via Jensen, below: $x\mapsto x^{q/p}$ is convex on $[0,\infty)$.)</p>
<div class="thm" data-kind="Proposition" data-name="Best constant predictors">
<p>$\E(X-c)^2$ is minimized over $c$ by $c = \E X$, with minimum $\Var X$. $\E|X-c|$ is minimized by any median of $X$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\E(X-c)^2 = \Var X + (\E X - c)^2$, expanding around $\mu$ (the cross term has mean zero). For the absolute loss, for $c\lt c'$ one checks $\E|X-c'| - \E|X-c| = \int_c^{c'}\big(\P(X\le t) - \P(X\gt t)\big)\d t$, so moving $c$ upward helps exactly while $\P(X\le t)\lt\frac12$; the minimum is attained where $F$ crosses $\frac12$.</p>
</details>
<p>Squared error and the mean, absolute error and the median: this pairing reappears in <a href="#/t/conditional-expectation">conditional expectation</a> (the best predictor), in <a href="#/t/robust-statistics">robust statistics</a> and in quantile regression.</p>`,
      },
      {
        title: 'Jensen\'s inequality',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Jensen's inequality">
<p>If $\varphi$ is convex and $\E|X|,\E|\varphi(X)|\lt\infty$, then $\varphi(\E X)\le\E\varphi(X)$. If $\varphi$ is strictly convex, equality holds iff $X$ is a.s. constant.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>A convex function lies above each of its supporting lines: at $\mu = \E X$ there is a slope $b$ with $\varphi(x)\ge\varphi(\mu) + b(x-\mu)$ for all $x$. Substitute $X$ and take expectations: $\E\varphi(X)\ge\varphi(\mu) + b\,\E(X-\mu) = \varphi(\mu)$. For strict convexity, the inequality $\varphi(x)\gt\varphi(\mu)+b(x-\mu)$ is strict for $x\ne\mu$, so equality forces $X=\mu$ a.s.</p>
</details>
<p>A huge number of inequalities are Jensen in disguise: $\E X^2\ge(\E X)^2$ ($\varphi = x^2$, i.e. $\Var\ge0$); $\E e^X\ge e^{\E X}$; the AM–GM inequality ($\varphi = -\log$); Lyapunov's inequality; the nonnegativity of KL divergence (<a href="#/t/entropy-kl">entropy and KL</a>); the fact that the EM algorithm never decreases the likelihood (<a href="#/t/em-algorithm">EM</a>).</p>
<div class="note" data-name="How big is the gap?">
<p>Taylor-expand $\varphi$ around $\mu$: $\varphi(X)\approx\varphi(\mu)+\varphi'(\mu)(X-\mu)+\tfrac12\varphi''(\mu)(X-\mu)^2$. Taking expectations, $\E\varphi(X)-\varphi(\E X)\approx\tfrac12\varphi''(\mu)\Var X$. The Jensen gap is curvature times variance. This is the second-order <a href="#/t/delta-method">delta method</a>, and it explains, for example, why $\exp(\widehat{\E\log Y})$ underestimates $\E Y$ by roughly a factor $e^{-\sigma^2/2}$ for lognormal data.</p>
</div>`,
        lab: 'expectation-jensen',
      },
      {
        title: 'Tails and the existence of moments',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Tail-integral formula">
<p>If $X\ge0$, then $\E X = \int_0^\infty\P(X\gt t)\d t$, and more generally $\E X^p = \int_0^\infty p\,t^{p-1}\P(X\gt t)\d t$ for $p\gt0$ (both sides possibly infinite). For integer-valued $X\ge0$, $\E X = \sum_{k\ge1}\P(X\ge k)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Pointwise, $X^p = \int_0^X p\,t^{p-1}\d t = \int_0^\infty p\,t^{p-1}\ind\{X\gt t\}\d t$. Take expectations and swap $\E$ and $\int$ by Tonelli's theorem (the integrand is nonnegative): $\E X^p = \int_0^\infty p\,t^{p-1}\P(X\gt t)\d t$.</p>
</details>
<p>So the existence of moments is a statement about tail decay. If $\P(|X|\gt t)\sim Ct^{-\alpha}$ (a polynomial, "Pareto-type" tail), then $\E|X|^p\lt\infty$ iff $p\lt\alpha$. The Student $t_\nu$ has $\alpha = \nu$; the Cauchy $\alpha=1$; Pareto$(\alpha)$ has exactly $\alpha$. Exponential and normal tails have all moments.</p>
<p>What happens to the sample mean $\bar X_n$ in each regime?</p>
<ul>
<li>$\alpha\gt2$: finite variance; $\bar X_n$ concentrates at rate $n^{-1/2}$ and the <a href="#/t/clt">CLT</a> applies.</li>
<li>$1\lt\alpha\le2$: finite mean, infinite variance; the <a href="#/t/lln">LLN</a> still holds but convergence is slow ($n^{1/\alpha-1}$) and punctuated by jumps; the limit law is stable, not normal.</li>
<li>$\alpha\le1$: no mean; $\bar X_n$ does not converge at all. For the Cauchy, $\bar X_n$ has exactly the same Cauchy distribution as a single observation for every $n$.</li>
</ul>
<p>A telling diagnostic is the share of the sum contributed by the single largest observation, $\max_i|X_i|/\sum_i|X_i|$. It tends to 0 when the mean is finite, but stays of order 1 when $\alpha\lt1$: one observation dominates the total forever. The lab tracks it.</p>`,
        lab: 'expectation-heavy-tails',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="E of a ratio is not a ratio of Es">
<p>$\E(1/X)\ne1/\E X$ (Jensen: $\E(1/X)\ge1/\E X$ for $X\gt0$), and $\E(X/Y)\ne\E X/\E Y$ even under independence. For $X\sim$ Poisson, $\E(1/X)$ is infinite because $\P(X=0)\gt0$. Plug-in estimators of nonlinear quantities are generally biased.</p>
</div>
<div class="warn" data-name="Interchanging limits and expectations">
<p>$X_n\to X$ does not imply $\E X_n\to\E X$. Take $X_n = n\ind\{U\lt1/n\}$: $X_n\to0$ a.s., yet $\E X_n = 1$. The interchange needs a condition — monotonicity (monotone convergence), domination by an integrable variable (dominated convergence) or uniform integrability.</p>
</div>
<div class="warn" data-name="Sample moments always exist">
<p>Every finite sample has a finite mean and variance, even from a Cauchy. Their existence says nothing about the population. Heavy tails show up as instability across subsamples and as a few observations dominating the sum — check before reporting a standard error.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A fair die is rolled until a 6 appears. Use the tail-sum formula to find the expected number of rolls.`,
        a: R`$\P(N\ge k) = (5/6)^{k-1}$, so $\E N = \sum_{k\ge1}(5/6)^{k-1} = 6$.` },
      { q: R`Each of $n$ people throws their hat in a box and draws one at random. Let $N$ be the number who get their own hat. Find $\E N$ and $\Var N$ without finding the distribution of $N$.`,
        a: R`Indicators: $\E N = n\cdot\frac1n = 1$. $\E N(N-1) = \sum_{i\ne j}\P(A_i\cap A_j) = n(n-1)\frac{1}{n(n-1)} = 1$, so $\E N^2 = 2$ and $\Var N = 1$ (for $n\ge2$).` },
      { q: R`If $\Var X = 4$, $\Var Y = 9$, what is the range of possible values of $\Var(X+Y)$?`,
        a: R`$\Var(X+Y) = 13 + 2\Cov(X,Y)$ with $|\Cov|\le\sigma_X\sigma_Y = 6$ by Cauchy–Schwarz, so $\Var(X+Y)\in[1,25]$; the extremes need $Y = -\frac32X + c$ and $Y = \frac32X + c$.` },
      { q: R`For which $\nu$ does the Student $t_\nu$ have a finite variance, and what does the sample variance do when it does not?`,
        a: R`$\P(|T|\gt t)\asymp t^{-\nu}$, so $\E T^2\lt\infty$ iff $\nu\gt2$. For $\nu\le2$ the sample variance does not converge; it drifts upward without bound as occasional huge observations arrive.` },
      { q: R`Show that $\E\log X\le\log\E X$ for positive $X$, and give the approximate size of the gap.`,
        a: R`$\log$ is concave, so by Jensen (applied to $-\log$) $\E\log X\le\log\E X$. With $\varphi'' = -1/x^2$, the gap is approximately $\frac12\Var X/\mu^2 = \frac12\text{CV}^2$, half the squared coefficient of variation.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'mgf',
    domain: 'foundations',
    order: 70,
    title: 'Moment Generating & Characteristic Functions',
    subtitle: 'Transforms that turn sums into products and convergence into calculus',
    level: 2,
    sources: { aos: '3.6', cb: '§2.3–2.4', other: 'Billingsley (1995) §26 & §30; Durrett (2019) §3.3; Heyde (1963) on the lognormal moment problem' },
    prereqs: ['expectation', 'distribution-families'],
    related: ['clt', 'convergence', 'exponential-families', 'inequalities'],
    tags: ['MGF', 'cumulants', 'characteristic function', 'uniqueness', 'moment problem'],
    summary: R`The moment generating function $M_X(t) = \E e^{tX}$ packs every moment of $X$ into a single function: its derivatives at 0 are the moments. Its real power is structural. Independent sums have MGFs that multiply, so distributions of sums can be identified at a glance; and when $M$ is finite near 0 it determines the distribution uniquely. The MGF need not exist — heavy-tailed laws, and even the lognormal, have none — which is why the characteristic function $\varphi_X(t) = \E e^{itX}$, which always exists, is the tool of choice for limit theorems. The proof of the central limit theorem is, at heart, a two-term Taylor expansion of $\varphi$.`,
    survey: {
      task: 'Compute moments, identify distributions of sums, prove limit theorems',
      assumptions: R`MGF: $\E e^{tX}\lt\infty$ for $|t|\lt h$, some $h\gt0$. CF: none`,
      output: R`$M_X$, $K_X = \log M_X$ (cumulants), $\varphi_X$; uniqueness and continuity theorems`,
      strengths: 'Sums become products; uniqueness and Lévy continuity make limit proofs short',
      weaknesses: 'MGF may not exist; moments need not determine a distribution; inversion is analytic, not elementary',
      complexity: R`Closed forms for standard families; empirical MGF/CF are $O(n)$ per $t$`,
    },
    sections: [
      {
        title: 'Generating functions',
        html: R`
<p>A transform encodes a distribution as a function on which some operation of interest is easy. For a variable on $\{0,1,2,\dots\}$, the <strong>probability generating function</strong> $G(s) = \E s^X = \sum_k p(k)s^k$, $|s|\le1$, is a power series whose coefficients are the probabilities: $p(k) = G^{(k)}(0)/k!$. Its derivatives at $s=1$ give factorial moments, $G'(1) = \E X$ and $G''(1) = \E X(X-1)$. For Poisson$(\lambda)$, $G(s) = \sum_k e^{-\lambda}(\lambda s)^k/k! = e^{\lambda(s-1)}$; for Bernoulli$(p)$, $G(s) = 1-p+ps$. PGFs are the natural tool for <a href="#/t/branching-processes">branching processes</a>, where generations compose as $G\circ G\circ\cdots$.</p>
<p>Substituting $s = e^t$ gives a transform that works for any real-valued variable.</p>
<div class="def" data-name="Moment generating function">
<p>$M_X(t) = \E e^{tX}$, defined for those $t$ where the expectation is finite. We say $X$ <em>has an MGF</em> if $M_X(t)\lt\infty$ for all $t$ in some open interval $(-h,h)$, $h\gt0$.</p>
</div>
<p>$M_X(0) = 1$ always; the domain $\{t: M_X(t)\lt\infty\}$ is an interval containing 0 (by convexity of $t\mapsto e^{tx}$), and $M_X$ is convex on it. Finiteness of $M_X(t)$ for $t\gt0$ requires the right tail to decay at least exponentially: $\P(X\gt x)\le M_X(t)e^{-tx}$ by Markov's inequality — the starting point of Chernoff bounds (<a href="#/t/inequalities">inequalities</a>).</p>
<table>
<thead><tr><th>Distribution</th><th>$M_X(t)$</th><th>Domain</th></tr></thead>
<tbody>
<tr><td>Bernoulli$(p)$</td><td>$1-p+pe^t$</td><td>$\R$</td></tr>
<tr><td>Binomial$(n,p)$</td><td>$(1-p+pe^t)^n$</td><td>$\R$</td></tr>
<tr><td>Poisson$(\lambda)$</td><td>$\exp\{\lambda(e^t-1)\}$</td><td>$\R$</td></tr>
<tr><td>Geometric$(p)$, trials</td><td>$\dfrac{pe^t}{1-(1-p)e^t}$</td><td>$t\lt-\log(1-p)$</td></tr>
<tr><td>Normal$(\mu,\sigma^2)$</td><td>$\exp\{\mu t+\sigma^2t^2/2\}$</td><td>$\R$</td></tr>
<tr><td>Exponential (rate $\lambda$)</td><td>$\lambda/(\lambda-t)$</td><td>$t\lt\lambda$</td></tr>
<tr><td>Gamma$(\alpha,\beta)$, scale $\beta$</td><td>$(1-\beta t)^{-\alpha}$</td><td>$t\lt1/\beta$</td></tr>
<tr><td>Uniform$(a,b)$</td><td>$\dfrac{e^{tb}-e^{ta}}{t(b-a)}$ ($t\ne0$)</td><td>$\R$</td></tr>
<tr><td>Lognormal, $t_\nu$, Cauchy, Pareto</td><td>$\infty$ for every $t\gt0$</td><td>no MGF</td></tr>
</tbody>
</table>`,
      },
      {
        title: 'Moments and cumulants',
        html: R`
<div class="thm" data-kind="Theorem" data-name="MGFs generate moments">
<p>If $M_X(t)\lt\infty$ for $|t|\lt h$, then $\E|X|^k\lt\infty$ for all $k$, and for $|t|\lt h$
$$M_X(t) = \sum_{k=0}^\infty\frac{\E X^k}{k!}t^k,\qquad\text{so}\qquad M_X^{(k)}(0) = \E X^k.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Fix $0\lt s\lt h$. Since $e^{s|x|}\le e^{sx}+e^{-sx}$, we have $\E e^{s|X|}\le M_X(s)+M_X(-s)\lt\infty$. Each term of $e^{s|X|} = \sum_k s^k|X|^k/k!$ is nonnegative, so $\E|X|^k\le k!\,s^{-k}\E e^{s|X|}\lt\infty$. For $|t|\le s$ the partial sums of $\sum_k(tX)^k/k!$ are dominated by $e^{s|X|}$, which is integrable, so dominated convergence lets us exchange $\E$ and $\sum$. A convergent power series can be differentiated term by term inside its radius of convergence, giving $M^{(k)}(0) = \E X^k$.</p>
</details>
<p>The logarithm is often more useful. The <strong>cumulant generating function</strong> $K_X(t) = \log M_X(t)$ has Taylor coefficients $\kappa_j$, the <em>cumulants</em>: $K(t) = \sum_{j\ge1}\kappa_jt^j/j!$, with
$$\kappa_1 = \E X,\quad \kappa_2 = \Var X,\quad\kappa_3 = \E(X-\mu)^3,\quad\kappa_4 = \E(X-\mu)^4-3\sigma^4.$$
(To check $\kappa_2$: $K' = M'/M$ and $K'' = M''/M - (M'/M)^2$, which at 0 is $\E X^2-(\E X)^2$.) Cumulants are additive over independent sums and shift-invariant beyond the first, which makes them the right coordinates for describing departures from normality: the normal has $K(t) = \mu t+\sigma^2t^2/2$, so all cumulants beyond the second vanish. Skewness is $\kappa_3/\kappa_2^{3/2}$ and excess kurtosis $\kappa_4/\kappa_2^2$. For a Poisson, $K(t) = \lambda(e^t-1)$, so <em>every</em> cumulant equals $\lambda$.</p>
<div class="ex" data-name="Gamma moments in two lines">
<p>For Gamma$(\alpha,\beta)$, $K(t) = -\alpha\log(1-\beta t)$, so $K'(t) = \alpha\beta/(1-\beta t)$ and $K''(t) = \alpha\beta^2/(1-\beta t)^2$. At 0: mean $\alpha\beta$, variance $\alpha\beta^2$. Generally $\kappa_j = (j-1)!\,\alpha\beta^j$, so skewness $= 2/\sqrt\alpha\to0$ as $\alpha\to\infty$ — the gamma becomes normal.</p>
</div>
<p>The lab plots $M$ and $K$ with their second-order approximations, and an <em>empirical</em> MGF $\hat M(t) = \frac1n\sum_i e^{tX_i}$. Push $t$ away from 0 and watch $\hat M$ become hostage to the single largest observation.</p>`,
        lab: 'mgf-explorer',
      },
      {
        title: 'Uniqueness and sums',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Uniqueness">
<p>If $M_X(t) = M_Y(t)\lt\infty$ for all $t$ in an open interval around 0, then $X\overset{d}{=}Y$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>A finite MGF on $(-h,h)$ extends to an analytic function $\E e^{zX}$ on the strip $|\operatorname{Re} z|\lt h$. Two analytic functions that agree on a real interval agree on the whole strip, in particular on the imaginary axis, where they are the characteristic functions. Characteristic functions determine the distribution by the inversion formula below.</p>
</details>
<div class="thm" data-kind="Proposition" data-name="Sums and affine maps">
<p>If $X$ and $Y$ are independent, $M_{X+Y}(t) = M_X(t)M_Y(t)$. For constants, $M_{a+bX}(t) = e^{at}M_X(bt)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$e^{tX}$ and $e^{tY}$ are independent, so $\E e^{t(X+Y)} = \E e^{tX}\E e^{tY}$ (the product rule for expectations of independent variables, valid here for nonnegative variables even if infinite). The affine rule is $\E e^{t(a+bX)} = e^{at}\E e^{(bt)X}$.</p>
</details>
<p>Together these identify the distribution of many sums immediately:</p>
<ul>
<li>Poisson$(\lambda_1)$ + Poisson$(\lambda_2)$: $e^{\lambda_1(e^t-1)}e^{\lambda_2(e^t-1)} = e^{(\lambda_1+\lambda_2)(e^t-1)}$ — Poisson$(\lambda_1+\lambda_2)$.</li>
<li>$\N(\mu_1,\sigma_1^2)+\N(\mu_2,\sigma_2^2)$ independent: exponents add, giving $\N(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$.</li>
<li>$n$ IID Exp$(\lambda)$: $(\lambda/(\lambda-t))^n = (1-t/\lambda)^{-n}$ — Gamma$(n,1/\lambda)$.</li>
<li>$\bar X_n$ for IID normals: $M_{\bar X}(t) = M(t/n)^n = \exp\{\mu t+\sigma^2t^2/(2n)\}$, so $\bar X_n\sim\N(\mu,\sigma^2/n)$ exactly.</li>
<li>Sum of squares of $k$ IID standard normals: $M_{Z^2}(t) = \E e^{tZ^2} = (1-2t)^{-1/2}$ for $t\lt\frac12$, so the sum has $(1-2t)^{-k/2}$ — Gamma$(k/2,2) = \chi^2_k$.</li>
</ul>`,
      },
      {
        title: 'Moments do not always determine the distribution',
        html: R`
<p>If $X$ has an MGF, its moments determine it (via uniqueness, since the moments determine the power series). Without an MGF, two different distributions can share <em>all</em> their moments.</p>
<div class="thm" data-kind="Proposition" data-name="Heyde's lognormal example (1963)">
<p>Let $f$ be the standard lognormal density, $f(x) = \frac{1}{x\sqrt{2\pi}}e^{-(\log x)^2/2}$, $x\gt0$. For every $a\in[-1,1]$, $f_a(x) = f(x)\big[1+a\sin(2\pi\log x)\big]$ is a density with exactly the same moments $\E X^k = e^{k^2/2}$, $k = 1,2,\dots$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$f_a\ge0$ since $|a\sin|\le1$. It suffices to show $\int_0^\infty x^kf(x)\sin(2\pi\log x)\d x = 0$ for $k=0,1,2,\dots$ (the case $k=0$ shows $f_a$ integrates to 1). Substitute $y=\log x$: the integral becomes $\int e^{ky}\phi(y)\sin(2\pi y)\d y$. Complete the square, $e^{ky}\phi(y) = e^{k^2/2}\phi(y-k)$, and set $u=y-k$: we get $e^{k^2/2}\int\phi(u)\sin(2\pi u+2\pi k)\d u = e^{k^2/2}\int\phi(u)\sin(2\pi u)\d u = 0$, since $k$ is an integer and the integrand is odd.</p>
</details>
<p>The lognormal's moments grow like $e^{k^2/2}$ — too fast for the moment sequence to pin down the distribution. A sufficient condition for determinacy is <em>Carleman's condition</em> $\sum_k(\E X^{2k})^{-1/(2k)} = \infty$, which every distribution with an MGF satisfies. The practical lesson: matching moments (as in the <a href="#/t/method-of-moments">method of moments</a> or moment-based approximations) does not by itself match distributions, especially in the tails.</p>`,
      },
      {
        title: 'Characteristic functions and the CLT',
        html: R`
<div class="def" data-name="Characteristic function">
<p>$\varphi_X(t) = \E e^{itX} = \E\cos(tX) + i\,\E\sin(tX)$, $t\in\R$.</p>
</div>
<p>Because $|e^{itX}| = 1$, the CF exists for every distribution. It satisfies $\varphi(0) = 1$, $|\varphi(t)|\le1$, $\varphi(-t) = \overline{\varphi(t)}$; it is uniformly continuous; it is real-valued iff $X\overset{d}{=}-X$; and the sum and affine rules carry over unchanged: $\varphi_{X+Y} = \varphi_X\varphi_Y$ for independent summands, $\varphi_{a+bX}(t) = e^{iat}\varphi_X(bt)$.</p>
<table>
<thead><tr><th>Distribution</th><th>$\varphi(t)$</th></tr></thead>
<tbody>
<tr><td>$\N(\mu,\sigma^2)$</td><td>$e^{i\mu t-\sigma^2t^2/2}$</td></tr>
<tr><td>Cauchy$(0,1)$</td><td>$e^{-|t|}$</td></tr>
<tr><td>Uniform$(-a,a)$</td><td>$\sin(at)/(at)$</td></tr>
<tr><td>Exponential(1)</td><td>$1/(1-it)$</td></tr>
<tr><td>Laplace$(0,1)$</td><td>$1/(1+t^2)$</td></tr>
<tr><td>Rademacher $\pm1$</td><td>$\cos t$</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Theorem" data-name="Inversion and Lévy continuity">
<p>(a) <em>Inversion.</em> If $a\lt b$ are continuity points of $F$, $F(b)-F(a) = \lim_{T\to\infty}\frac1{2\pi}\int_{-T}^T\frac{e^{-ita}-e^{-itb}}{it}\varphi(t)\d t$; if $\int|\varphi|\lt\infty$, $X$ has the bounded continuous density $f(x) = \frac1{2\pi}\int e^{-itx}\varphi(t)\d t$. Hence $\varphi$ determines the distribution.<br>
(b) <em>Continuity theorem.</em> $X_n\dto X$ iff $\varphi_{X_n}(t)\to\varphi_X(t)$ for every $t$. Moreover, if $\varphi_{X_n}(t)\to g(t)$ pointwise and $g$ is continuous at 0, then $g$ is the CF of some $X$ and $X_n\dto X$.</p>
</div>
<p>Smoothness of $\varphi$ at 0 mirrors tail behaviour: if $\E|X|^k\lt\infty$ then $\varphi$ is $k$ times continuously differentiable with $\varphi^{(k)}(0) = i^k\E X^k$. The Cauchy CF $e^{-|t|}$ has a corner at 0: no mean.</p>
<div class="thm" data-kind="Theorem" data-name="Central limit theorem (via CFs)">
<p>If $X_1,X_2,\dots$ are IID with mean 0 and variance 1, then $S_n/\sqrt n\dto\N(0,1)$, where $S_n = X_1+\cdots+X_n$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Finite variance makes $\varphi$ twice differentiable with $\varphi(t) = 1 - t^2/2 + o(t^2)$ as $t\to0$. By independence and the affine rule, $\varphi_{S_n/\sqrt n}(t) = \varphi(t/\sqrt n)^n = \big(1-\frac{t^2}{2n}+o(\frac1n)\big)^n\to e^{-t^2/2}$ for each fixed $t$ (using $(1+c_n/n)^n\to e^c$ when $c_n\to c$, valid for complex $c_n$). The limit is the $\N(0,1)$ CF, so Lévy's continuity theorem finishes the proof. See <a href="#/t/clt">the CLT</a> for refinements (Lindeberg, Berry–Esseen).</p>
</details>
<p>Contrast the Cauchy: $\varphi_{S_n/n}(t) = (e^{-|t|/n})^n = e^{-|t|}$ for every $n$, so the sample mean of Cauchy data is exactly standard Cauchy — no averaging gain at all. With the $\sqrt n$ scaling, $\varphi_{S_n/\sqrt n}(t) = e^{-\sqrt n|t|}\to\ind\{t=0\}$, a limit discontinuous at 0: probability escapes to infinity and there is no limiting distribution. The lab shows all three behaviours.</p>`,
        lab: 'mgf-cf-clt',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Existence is not automatic">
<p>Before differentiating an MGF, check that it is finite on a neighbourhood of 0. A derivation that "computes" the MGF of a $t$ or lognormal variable has gone wrong somewhere; use the CF instead.</p>
</div>
<div class="warn" data-name="Pointwise convergence of transforms needs care">
<p>Convergence of MGFs on a neighbourhood of 0 implies convergence in distribution (Curtiss, 1942), but pointwise CF convergence to a limit that is <em>not</em> continuous at 0 means mass is escaping (the sequence is not tight), as in the Cauchy example. The continuity condition in Lévy's theorem is not a technicality.</p>
</div>
<div class="warn" data-name="Empirical MGFs away from zero">
<p>$\hat M(t) = \frac1n\sum e^{tX_i}$ is unbiased for $M(t)$, but its variance $\frac1n[M(2t)-M(t)^2]$ explodes as $t$ approaches half the edge of the domain — and is infinite beyond it. Saddlepoint and Chernoff-type methods that plug in $\hat M$ far from 0 inherit this instability.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Use MGFs to show that if $X\sim$ Binomial$(m,p)$ and $Y\sim$ Binomial$(n,p)$ are independent, then $X+Y\sim$ Binomial$(m+n,p)$. What if the $p$'s differ?`,
        a: R`$M_{X+Y}(t) = (1-p+pe^t)^m(1-p+pe^t)^n = (1-p+pe^t)^{m+n}$, the Binomial$(m+n,p)$ MGF, and uniqueness applies. With different $p$'s the product is not of binomial form; the sum is a "Poisson-binomial" variable.` },
      { q: R`Find the third cumulant of Poisson$(\lambda)$ and of $\N(\mu,\sigma^2)$, and interpret.`,
        a: R`Poisson: $K(t) = \lambda(e^t-1)$, so $\kappa_3 = \lambda$ (skewness $\lambda^{-1/2}\to0$). Normal: $K$ is quadratic, so $\kappa_3 = 0$. The Poisson becomes symmetric as $\lambda$ grows, consistent with its normal limit.` },
      { q: R`Why does the CF of a symmetric distribution take only real values?`,
        a: R`If $X\overset{d}{=}-X$ then $\varphi(t) = \E e^{-itX} = \overline{\varphi(t)}$, so $\varphi$ is real. Equivalently, $\E\sin(tX) = 0$ because $\sin$ is odd.` },
      { q: R`The MGF of $X$ is $M(t) = (1-2t)^{-3}$ for $t\lt\frac12$. Identify the distribution and give its mean and variance.`,
        a: R`It is Gamma$(3,2) = \chi^2_6$. From $K(t) = -3\log(1-2t)$: $K'(0) = 6$, $K''(0) = 12$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'multivariate',
    domain: 'foundations',
    order: 80,
    title: 'Joint Distributions & the Multivariate Normal',
    subtitle: 'Several random variables at once: marginals, conditionals, covariance and the Gaussian',
    level: 2,
    sources: { aos: '2.5–2.10, Ch. 14', cb: '§4.1–4.6', other: 'Anderson, An Introduction to Multivariate Statistical Analysis (2003), Ch. 2; Tong, The Multivariate Normal Distribution' },
    prereqs: ['expectation', 'transformations'],
    related: ['conditional-expectation', 'simple-linear-regression', 'pca', 'gaussian-processes', 'lda-qda'],
    tags: ['joint density', 'independence', 'covariance matrix', 'multivariate normal', 'Schur complement'],
    summary: R`A joint distribution describes how several random variables vary together. Marginals are obtained by summing or integrating out, conditionals by slicing and renormalizing, and independence is factorization. Second-order structure is summarized by the covariance matrix, which transforms as $A\Sigma A^\top$ under linear maps. The multivariate normal is the one family where this summary is complete: it is closed under linear maps, marginalization and conditioning; zero correlation means independence; and conditional means are linear with constant conditional variance given by a Schur complement — the probabilistic foundation of linear regression, Kalman filtering and Gaussian processes.`,
    survey: {
      task: 'Model several variables jointly; derive marginal and conditional laws; describe dependence',
      assumptions: R`For MVN results: joint normality (not merely normal marginals); $\Sigma$ positive definite for a density`,
      output: R`Joint/marginal/conditional laws; $\bmu$, $\Sigma$; $\E[\bX_1\mid\bX_2]$ and conditional covariance`,
      strengths: 'MVN: closed under all linear operations; conditioning is linear algebra',
      weaknesses: 'Normal marginals do not imply joint normality; correlation misses nonlinear dependence',
      complexity: R`MVN density and conditioning: $O(k^3)$ (Cholesky / solve)`,
    },
    sections: [
      {
        title: 'Joint, marginal and independent',
        html: R`
<p>A random vector $\bX = (X_1,\dots,X_k)$ is a measurable map into $\R^k$. Its law is described by the joint CDF $F(\bx) = \P(X_1\le x_1,\dots,X_k\le x_k)$, or by a joint pmf $p(\bx)$ or joint density $f(\bx)$ with $\P(\bX\in A) = \int_A f$.</p>
<p><strong>Marginals</strong> come from summing or integrating out the other coordinates: for two variables, $f_X(x) = \int f(x,y)\d y$. The joint determines the marginals, never the reverse: many joint distributions share the same marginals, and the leftover information — the dependence structure — can be isolated as a <em>copula</em> (Sklar's theorem).</p>
<div class="thm" data-kind="Proposition" data-name="Independence as factorization">
<p>$X$ and $Y$ are independent iff $F(x,y) = F_X(x)F_Y(y)$ for all $x,y$. If $(X,Y)$ has a joint density, they are independent iff $f(x,y) = g(x)h(y)$ for some functions $g,h$ (for almost every $(x,y)$), in which case $g,h$ are proportional to the marginals.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Independence means $\P(X\in A, Y\in B) = \P(X\in A)\P(Y\in B)$ for all Borel $A,B$; taking half-lines gives the CDF factorization, and conversely CDF factorization on half-lines extends to all Borel rectangles by the π–λ argument. For densities: if $f = gh$, then $\int_A\int_B f = (\int_A g)(\int_B h)$, and setting $A$ or $B$ to $\R$ identifies the factors as multiples of the marginals.</p>
</details>
<div class="warn" data-name="The support must be a product set">
<p>$f(x,y) = 2$ on the triangle $0\lt x\lt y\lt1$ looks constant, hence "factorizable", but it is $2\cdot\ind\{0\lt x\lt y\lt1\}$, and the indicator does not split into a function of $x$ times a function of $y$. $X$ and $Y$ are dependent: knowing $X = 0.9$ forces $Y\in(0.9,1)$.</p>
</div>
<p>Expectations of functions of several variables use the multivariate LOTUS, $\E g(X,Y) = \iint g(x,y)f(x,y)\d x\d y$, and the distribution of transformed vectors uses the Jacobian formula from <a href="#/t/transformations">transformations</a>.</p>`,
      },
      {
        title: 'Conditional distributions',
        html: R`
<div class="def" data-name="Conditional density">
<p>If $(X,Y)$ has joint density $f$ and $f_X(x)\gt0$, the conditional density of $Y$ given $X=x$ is
$$f_{Y\mid X}(y\mid x) = \frac{f(x,y)}{f_X(x)}.$$
In the discrete case, $p_{Y\mid X}(y\mid x) = p(x,y)/p_X(x)$ is an ordinary conditional probability.</p>
</div>
<p>Geometrically: take the slice of the joint density along the vertical line at $x$, and rescale it to integrate to 1. Even though $\P(X=x)=0$, the definition is the limit of $\P(Y\le y\mid x\le X\le x+h)$ as $h\downarrow0$ at continuity points, which is what makes it the right object. The product rule $f(x,y) = f_X(x)f_{Y\mid X}(y\mid x)$ and the continuous Bayes formula
$$f_{X\mid Y}(x\mid y) = \frac{f_{Y\mid X}(y\mid x)f_X(x)}{\int f_{Y\mid X}(y\mid u)f_X(u)\d u}$$
follow immediately; the latter is the engine of <a href="#/t/bayesian-inference">Bayesian inference</a>, with $X$ the parameter and $Y$ the data. Independence is equivalent to $f_{Y\mid X}(y\mid x) = f_Y(y)$ for (almost) all $x$.</p>
<div class="ex" data-name="Hierarchical models">
<p>Let $P\sim$ Beta$(a,b)$ and, given $P=p$, $Y\sim$ Binomial$(n,p)$. Then $f(p,y) = \text{Beta}(p;a,b)\binom ny p^y(1-p)^{n-y}$; integrating out $p$ gives the beta-binomial marginal $\P(Y=y) = \binom ny\frac{B(a+y,b+n-y)}{B(a,b)}$, and dividing gives $P\mid Y=y\sim\text{Beta}(a+y,b+n-y)$. Specifying a joint law through a marginal and a conditional is the standard way to build models — and the subject of <a href="#/t/mixed-models">hierarchical models</a>.</p>
</div>`,
      },
      {
        title: 'Mean vectors and covariance matrices',
        html: R`
<p>For a random vector with finite second moments, $\bmu = \E\bX$ (componentwise) and
$$\Sigma = \Cov(\bX) = \E\big[(\bX-\bmu)(\bX-\bmu)^\top\big],\qquad\Sigma_{ij} = \Cov(X_i,X_j).$$</p>
<div class="thm" data-kind="Proposition" data-name="Linear transformation rule">
<p>For a fixed $m\times k$ matrix $A$ and vector $\mathbf b$: $\E(A\bX+\mathbf b) = A\bmu+\mathbf b$ and $\Cov(A\bX+\mathbf b) = A\Sigma A^\top$. In particular $\Var(\mathbf a^\top\bX) = \mathbf a^\top\Sigma\mathbf a\ge0$, so $\Sigma$ is symmetric positive semidefinite.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Linearity of expectation applied entrywise. Then $A\bX+\mathbf b-(A\bmu+\mathbf b) = A(\bX-\bmu)$, so the covariance is $\E[A(\bX-\bmu)(\bX-\bmu)^\top A^\top] = A\Sigma A^\top$. Taking $A = \mathbf a^\top$ gives a variance, which is nonnegative.</p>
</details>
<p>$\Sigma$ is singular exactly when some nontrivial linear combination $\mathbf a^\top\bX$ is a.s. constant — the distribution lives on a lower-dimensional affine subspace and has no density on $\R^k$. The spectral decomposition $\Sigma = Q\Lambda Q^\top$ gives the principal axes of variation, which is <a href="#/t/pca">PCA</a>.</p>
<div class="def" data-name="Multinomial distribution">
<p>Throw $n$ independent balls into $k$ boxes with probabilities $p_1,\dots,p_k$; let $N_j$ be the count in box $j$. Then $\P(N_1=n_1,\dots,N_k=n_k) = \frac{n!}{n_1!\cdots n_k!}p_1^{n_1}\cdots p_k^{n_k}$ for $\sum n_j = n$. Each $N_j\sim$ Binomial$(n,p_j)$ and $\Cov(N_i,N_j) = -np_ip_j$ for $i\ne j$: the counts compete for a fixed total, so the covariance matrix $n(\diag(\mathbf p)-\mathbf p\mathbf p^\top)$ is singular.</p>
</div>`,
      },
      {
        title: 'The multivariate normal',
        html: R`
<div class="def" data-name="Multivariate normal">
<p>$\bX\sim\N_k(\bmu,\Sigma)$ if $\bX \overset{d}{=} \bmu + A\mathbf Z$ where $\mathbf Z$ is a vector of IID $\N(0,1)$ and $AA^\top = \Sigma$. Equivalently: every linear combination $\mathbf a^\top\bX$ is univariate normal (possibly degenerate). Its MGF is $\E e^{\mathbf t^\top\bX} = \exp\{\mathbf t^\top\bmu+\frac12\mathbf t^\top\Sigma\mathbf t\}$, and when $\Sigma$ is positive definite it has density
$$f(\bx) = (2\pi)^{-k/2}\det(\Sigma)^{-1/2}\exp\Big\{-\tfrac12(\bx-\bmu)^\top\Sigma^{-1}(\bx-\bmu)\Big\}.$$</p>
</div>
<p>The equivalences: the MGF of $\bmu + A\mathbf Z$ is $e^{\mathbf t^\top\bmu}\E e^{(A^\top\mathbf t)^\top\mathbf Z} = e^{\mathbf t^\top\bmu + \frac12\|A^\top\mathbf t\|^2}$, which depends on $A$ only through $AA^\top=\Sigma$; and the "every linear combination" characterization follows from the Cramér–Wold device (a distribution on $\R^k$ is determined by the laws of all its one-dimensional projections). The density is the Jacobian formula applied to $\bx = \bmu+A\mathbf z$ with $|\det A| = \det(\Sigma)^{1/2}$.</p>
<div class="thm" data-kind="Theorem" data-name="Closure properties">
<p>Let $\bX\sim\N_k(\bmu,\Sigma)$. (i) $B\bX+\mathbf c\sim\N_m(B\bmu+\mathbf c, B\Sigma B^\top)$. (ii) Every sub-vector is multivariate normal with the corresponding sub-vector of $\bmu$ and sub-block of $\Sigma$. (iii) Two sub-vectors are independent iff their cross-covariance block is zero. (iv) If $\Sigma$ is invertible, $(\bX-\bmu)^\top\Sigma^{-1}(\bX-\bmu)\sim\chi^2_k$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(i) $B\bX+\mathbf c = (B\bmu+\mathbf c) + (BA)\mathbf Z$, and $(BA)(BA)^\top = B\Sigma B^\top$. (ii) is (i) with $B$ a coordinate-selection matrix. (iii) If $\Sigma_{12}=0$ the joint MGF factorizes, $\exp\{\mathbf t_1^\top\bmu_1+\frac12\mathbf t_1^\top\Sigma_{11}\mathbf t_1\}\exp\{\mathbf t_2^\top\bmu_2+\frac12\mathbf t_2^\top\Sigma_{22}\mathbf t_2\}$, which is the MGF of independent sub-vectors; uniqueness of MGFs gives independence. The converse holds for any distribution. (iv) Take $A = \Sigma^{1/2}$; then $\mathbf Z = \Sigma^{-1/2}(\bX-\bmu)$ is standard normal and the quadratic form is $\|\mathbf Z\|^2 = \sum Z_i^2\sim\chi^2_k$.</p>
</details>
<p>Contours of the density are the ellipsoids $(\bx-\bmu)^\top\Sigma^{-1}(\bx-\bmu) = c$, with axes along the eigenvectors of $\Sigma$ and half-lengths $\sqrt{c\lambda_j}$. By (iv), the ellipse with $c = \chi^2_{k,0.95}$ contains 95% of the probability — the basis of confidence ellipsoids and Mahalanobis-distance outlier detection.</p>`,
      },
      {
        title: 'Conditioning a Gaussian',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Conditional distributions of the MVN">
<p>Partition $\bX = (\bX_1,\bX_2)$ with $\bmu = (\bmu_1,\bmu_2)$ and $\Sigma = \begin{pmatrix}\Sigma_{11}&\Sigma_{12}\\\Sigma_{21}&\Sigma_{22}\end{pmatrix}$, $\Sigma_{22}$ invertible. Then
$$\bX_1\mid\bX_2 = \bx_2\;\sim\;\N\Big(\bmu_1+\Sigma_{12}\Sigma_{22}^{-1}(\bx_2-\bmu_2),\;\;\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\Big).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $B = \Sigma_{12}\Sigma_{22}^{-1}$ and $\mathbf W = \bX_1 - B\bX_2$. The pair $(\mathbf W,\bX_2)$ is a linear function of $\bX$, hence jointly normal, and $\Cov(\mathbf W,\bX_2) = \Sigma_{12}-B\Sigma_{22} = 0$. By closure property (iii), $\mathbf W$ is independent of $\bX_2$. Writing $\bX_1 = \mathbf W + B\bX_2$, conditioning on $\bX_2 = \bx_2$ leaves $\mathbf W$'s law unchanged, so $\bX_1\mid\bx_2\sim\N(\E\mathbf W + B\bx_2, \Cov\mathbf W)$. Finally $\E\mathbf W = \bmu_1-B\bmu_2$ and $\Cov\mathbf W = \Sigma_{11}-B\Sigma_{21}-\Sigma_{12}B^\top+B\Sigma_{22}B^\top = \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$.</p>
</details>
<p>Three features are special to the Gaussian and worth memorizing: the conditional mean is <strong>linear</strong> in $\bx_2$; the conditional covariance (the Schur complement) <strong>does not depend on</strong> $\bx_2$; and it is never larger than the unconditional $\Sigma_{11}$ — conditioning on Gaussian information always reduces uncertainty.</p>
<div class="ex" data-name="The bivariate case and regression to the mean">
<p>For $(X,Y)$ bivariate normal with correlation $\rho$:
$$\E[Y\mid X=x] = \mu_Y+\rho\frac{\sigma_Y}{\sigma_X}(x-\mu_X),\qquad\Var(Y\mid X=x) = \sigma_Y^2(1-\rho^2).$$
In standard units the predicted $Y$ is $\rho$ times the observed $X$: since $|\rho|\lt1$, predictions are pulled toward the mean. Tall parents have tall children, but less extremely so — Galton's "regression to mediocrity", which gave regression its name. Note the regression line of $Y$ on $X$ (slope $\rho\sigma_Y/\sigma_X$) is <em>not</em> the major axis of the ellipse, and not the inverse of the regression of $X$ on $Y$ (slope $\sigma_Y/(\rho\sigma_X)$ in the same axes). The fraction of variance explained is $\rho^2$.</p>
</div>`,
        lab: 'multivariate-bivariate-normal',
      },
      {
        title: 'Normal marginals are not enough',
        html: R`
<p>Every property above assumes <em>joint</em> normality. It is easy to construct vectors with normal marginals that violate all of them.</p>
<div class="ex" data-name="Uncorrelated, normal, and completely dependent">
<p>Let $X\sim\N(0,1)$, fix $c\gt0$, and set $Y = X$ if $|X|\le c$ and $Y = -X$ if $|X|\gt c$. By symmetry of the normal, $Y\sim\N(0,1)$. Its correlation with $X$ is
$$\rho(c) = \E[X^2\ind\{|X|\le c\}]-\E[X^2\ind\{|X|\gt c\}] = 2\big[(2\Phi(c)-1)-2c\,\phi(c)\big]-1,$$
which increases continuously from $-1$ at $c=0$ to $1$ as $c\to\infty$, crossing 0 near $c\approx1.54$. At that $c$, $X$ and $Y$ are each standard normal, uncorrelated, and yet $|Y| = |X|$ always. And $X+Y = 2X\ind\{|X|\le c\}$ has an atom at 0, so it is not normal: the pair is not bivariate normal.</p>
</div>
<div class="warn" data-name="Two common errors">
<p>(1) "$X$ and $Y$ are normal and uncorrelated, therefore independent" — true only under joint normality. (2) "$X$ and $Y$ are normal, so $X+Y$ is normal" — also needs joint normality (or independence). Regression theory, for instance, assumes normal <em>errors</em> jointly, not just marginally.</p>
</div>`,
        lab: 'multivariate-not-jointly-normal',
      },
    ],
    quiz: [
      { q: R`Let $(X,Y)$ be bivariate normal with $\sigma_X = 2$, $\sigma_Y = 3$, $\rho = 0.5$ and zero means. Find $\E[Y\mid X=4]$ and $\Var(Y\mid X=4)$.`,
        a: R`$\E[Y\mid X=4] = 0.5\cdot\frac32\cdot4 = 3$. $\Var(Y\mid X) = 9(1-0.25) = 6.75$, the same for every $x$.` },
      { q: R`If $\bX\sim\N_3(\mathbf 0, I)$, what is the distribution of $X_1-X_2$ and of $(X_1+X_2, X_1-X_2)$?`,
        a: R`$X_1-X_2\sim\N(0,2)$. The pair is $B\bX$ with $B = \begin{pmatrix}1&1&0\\1&-1&0\end{pmatrix}$, so it is bivariate normal with covariance $BB^\top = 2I$: the sum and difference are independent $\N(0,2)$ variables.` },
      { q: R`Show that $f(x,y) = 6x$ on $0\lt x\lt y\lt1$ is a density and determine whether $X$ and $Y$ are independent.`,
        a: R`$\int_0^1\int_0^y6x\d x\d y = \int_0^13y^2\d y = 1$. The support is a triangle, not a product set, so they are dependent (e.g. $\P(X\gt\frac12, Y\lt\frac12) = 0$ while both marginal probabilities are positive).` },
      { q: R`In the multinomial, why is $\Cov(N_i,N_j)$ negative, and why is the covariance matrix singular?`,
        a: R`The counts share a fixed total: more balls in box $i$ leaves fewer for box $j$. Since $\sum_j N_j = n$ is constant, $\Var(\mathbf 1^\top\mathbf N) = \mathbf 1^\top\Sigma\mathbf 1 = 0$, so $\mathbf 1$ is in the null space of $\Sigma$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'conditional-expectation',
    domain: 'foundations',
    order: 90,
    title: 'Conditional Expectation & the Iterated Laws',
    subtitle: 'The best prediction from partial information, and how averages and variances decompose',
    level: 2,
    sources: { aos: '3.5', cb: '§4.4', other: 'Williams, Probability with Martingales (1991), Ch. 9; Blitzstein & Hwang, Introduction to Probability, Ch. 9' },
    prereqs: ['multivariate', 'conditional-probability'],
    related: ['martingales', 'rao-blackwell', 'em-algorithm', 'simple-linear-regression', 'bias-variance', 'overdispersion'],
    tags: ['tower property', 'law of total variance', 'best predictor', 'projection', 'random sums'],
    summary: R`$\E[Y\mid X]$ is the average of $Y$ once the value of $X$ is known — a function $m(X)$ of $X$, and therefore itself a random variable. Two laws make it indispensable: the tower property $\E[\E[Y\mid X]] = \E Y$, which computes hard expectations by conditioning on the right thing, and the law of total variance $\Var Y = \E\Var(Y\mid X) + \Var\E[Y\mid X]$, which splits uncertainty into unexplained and explained parts. Geometrically, $\E[Y\mid X]$ is the orthogonal projection of $Y$ onto the space of functions of $X$, and hence the mean-squared-error-optimal predictor: the target that every regression method is estimating.`,
    survey: {
      task: 'Predict $Y$ from $X$; compute expectations and variances by conditioning',
      assumptions: R`$\E|Y|\lt\infty$ (existence); $\E Y^2\lt\infty$ for the projection and variance results`,
      output: R`The regression function $m(x) = \E[Y\mid X=x]$; variance decomposition`,
      strengths: 'Tower property and total variance solve many problems in two lines; defines the regression target',
      weaknesses: R`Defined only up to null sets; conditioning on probability-zero events is ambiguous without a σ-field`,
      complexity: 'Analytic; estimating $m$ from data is the whole of nonparametric regression',
    },
    sections: [
      {
        title: 'From a number to a random variable',
        html: R`
<p>Given $X=x$, the conditional distribution of $Y$ (a pmf $p_{Y\mid X}(\cdot\mid x)$ or density $f_{Y\mid X}(\cdot\mid x)$, see <a href="#/t/multivariate">joint distributions</a>) has a mean:
$$m(x) = \E[Y\mid X=x] = \int y\,f_{Y\mid X}(y\mid x)\d y\quad\text{or}\quad\sum_y y\,p_{Y\mid X}(y\mid x).$$
This is a number for each $x$ — an ordinary function, the <strong>regression function</strong>. Plugging in the random $X$ gives the random variable
$$\E[Y\mid X] = m(X).$$
It is random because $X$ is: before observing $X$ we do not know which conditional mean will apply.</p>
<div class="ex" data-name="Rolling until a value">
<p>Roll a die; let $X$ be the result and, given $X=x$, toss $x$ fair coins and let $Y$ count heads. Then $\E[Y\mid X=x] = x/2$, so $\E[Y\mid X] = X/2$, a random variable taking values $\frac12,1,\dots,3$ with probability $\frac16$ each. Its mean is $\E X/2 = 1.75$ — which is $\E Y$, by the theorem that follows.</p>
</div>
<div class="note"><p>$\E[Y\mid X]$ depends on $X$ only through the information it carries: if $g$ is one-to-one, $\E[Y\mid g(X)] = \E[Y\mid X]$. This suggests the measure-theoretic definition below, where one conditions on a σ-field (an information set) rather than on a particular variable.</p></div>`,
      },
      {
        title: 'The tower property',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Law of iterated expectations (tower property)">
<p>If $\E|Y|\lt\infty$, then $\E\big[\E[Y\mid X]\big] = \E Y$.</p>
</div>
<details class="proof"><summary>Proof (density case)</summary>
<p>$\E[m(X)] = \int m(x)f_X(x)\d x = \int\Big(\int y\frac{f(x,y)}{f_X(x)}\d y\Big)f_X(x)\d x = \iint y\,f(x,y)\d y\d x = \E Y$, the interchange justified by Fubini since $\iint|y|f(x,y) = \E|Y|\lt\infty$. The discrete case replaces integrals with sums; the general case is built into the definition below.</p>
</details>
<p>Equivalently, $\E Y = \sum_x\E[Y\mid X=x]\P(X=x)$ — the law of total probability for averages. Its use is strategic: <strong>condition on whatever makes the problem easy</strong>.</p>
<div class="ex" data-name="Random sums (Wald's identity)">
<p>Let $N$ be a nonnegative integer variable, independent of IID $X_1,X_2,\dots$ with mean $\mu$, and $S = \sum_{i=1}^NX_i$ (total insurance claims, total customers' spending). Given $N=n$, $\E[S\mid N=n] = n\mu$; so $\E[S\mid N] = N\mu$ and $\E S = \mu\,\E N$. (Wald's identity extends this to stopping times $N$ that may depend on the past $X_i$ but not the future.)</p>
</div>
<div class="ex" data-name="Waiting for two heads in a row">
<p>Let $e$ be the expected number of fair-coin tosses to see HH, and $e_1$ the expected number still needed when the last toss was H. Conditioning on the next toss: $e = 1+\frac12e_1+\frac12e$ and $e_1 = 1+\frac12\cdot0+\frac12e$. Solving, $e = 6$. (For HT the answer is 4: after an H, a T finishes and another H keeps you one step away, never sending you back to the start.) This "first-step analysis" is the tower property applied to a <a href="#/t/markov-chains">Markov chain</a>.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Working rules">
<p>Assuming the expectations exist: (i) linearity, $\E[aY_1+bY_2\mid X] = a\E[Y_1\mid X]+b\E[Y_2\mid X]$; (ii) <em>taking out what is known</em>, $\E[g(X)Y\mid X] = g(X)\E[Y\mid X]$; (iii) if $Y\indep X$, $\E[Y\mid X] = \E Y$; (iv) <em>tower, general form</em>, $\E\big[\E[Y\mid X,Z]\mid X\big] = \E[Y\mid X]$ — the coarser conditioning wins.</p>
</div>
<p>Rule (ii) is intuitive: once $X$ is known, $g(X)$ is a constant. Rule (iv) says that predicting your future best prediction, using less information, gives your current best prediction — the defining property of a <a href="#/t/martingales">martingale</a>.</p>`,
      },
      {
        title: 'The best predictor',
        html: R`
<p>Among all functions $g$, which $g(X)$ best predicts $Y$ in mean squared error?</p>
<div class="thm" data-kind="Theorem" data-name="Conditional expectation minimizes MSE">
<p>If $\E Y^2\lt\infty$, then for every function $g$ with $\E g(X)^2\lt\infty$,
$$\E\big(Y-g(X)\big)^2 = \E\big(Y-m(X)\big)^2+\E\big(m(X)-g(X)\big)^2,$$
where $m(X) = \E[Y\mid X]$. Hence $m$ is the unique (up to null sets) minimizer.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $Y-g = (Y-m)+(m-g)$ and expand. The cross term is $2\E[(Y-m(X))h(X)]$ with $h = m-g$. Condition on $X$ and take out what is known: $\E[(Y-m(X))h(X)\mid X] = h(X)\big(\E[Y\mid X]-m(X)\big) = 0$, so by the tower property the cross term vanishes.</p>
</details>
<p>The key step is the <strong>orthogonality</strong> $\E[(Y-m(X))h(X)] = 0$ for every $h$: the prediction error is uncorrelated with every function of $X$. In the Hilbert space $L^2$ of finite-variance variables with inner product $\langle U,V\rangle = \E UV$, the functions of $X$ form a closed subspace, and $m(X)$ is the orthogonal projection of $Y$ onto it. The theorem is Pythagoras.</p>
<div class="cols">
<div><h4>Best predictor</h4><p>$m(X) = \E[Y\mid X]$: projection onto <em>all</em> functions of $X$. Error is orthogonal to every $h(X)$.</p></div>
<div><h4>Best linear predictor</h4><p>$\alpha+\beta X$ with $\beta = \Cov(X,Y)/\Var X$: projection onto $\text{span}\{1,X\}$. Error is uncorrelated with $X$ only.</p></div>
<div><h4>Best constant</h4><p>$\E Y$: projection onto constants. MSE $= \Var Y$.</p></div>
</div>
<p>Larger subspaces give smaller error, so $\Var Y\ge\text{MSE}_{\text{lin}}\ge\text{MSE}_m$. The best linear predictor coincides with $m$ exactly when $m$ is linear — always true for the <a href="#/t/multivariate">bivariate normal</a>, not in general. Statistical regression (<a href="#/t/simple-linear-regression">linear</a>, <a href="#/t/kernel-regression">kernel</a>, <a href="#/t/decision-trees">trees</a>, neural networks) consists of different ways of estimating $m$ from data, trading the bias of a restricted class against the variance of estimating a flexible one (<a href="#/t/bias-variance">bias–variance</a>). The lab estimates $m$ by binned averages and compares it with the three population predictors by exact test error.</p>`,
        lab: 'conditional-expectation-projection',
      },
      {
        title: 'The law of total variance',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Law of total variance (Eve's law)">
<p>If $\E Y^2\lt\infty$, then
$$\Var Y = \underbrace{\E\big[\Var(Y\mid X)\big]}_{\text{within / unexplained}} + \underbrace{\Var\big(\E[Y\mid X]\big)}_{\text{between / explained}},$$
where $\Var(Y\mid X) = \E[(Y-\E[Y\mid X])^2\mid X]$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>With $m = m(X)$, $Y-\E Y = (Y-m)+(m-\E Y)$. Square and take expectations. The first term is $\E(Y-m)^2 = \E[\E[(Y-m)^2\mid X]] = \E\Var(Y\mid X)$ by the tower property. The second is $\E(m-\E Y)^2 = \Var(m)$, because $\E m = \E Y$. The cross term $2\E[(Y-m)(m-\E Y)]$ vanishes by orthogonality, $m-\E Y$ being a function of $X$.</p>
</details>
<p>This is the Pythagorean theorem from the previous section with $g\equiv\E Y$. The ratio $\Var(\E[Y\mid X])/\Var Y$ is the population <em>correlation ratio</em> $\eta^2$, the fraction of variance explained by the best possible predictor; its sample version is the one-way <a href="#/t/anova">ANOVA</a> decomposition $SS_{\text{total}} = SS_{\text{within}}+SS_{\text{between}}$. For bivariate normals $\eta^2 = \rho^2$; in general $\eta^2\ge\rho^2$.</p>
<div class="ex" data-name="Variance of a random sum">
<p>In the random-sum example, with $\Var X_i = \sigma^2$: $\E[S\mid N] = N\mu$ and $\Var(S\mid N) = N\sigma^2$, so
$$\Var S = \E[N\sigma^2] + \Var(N\mu) = \sigma^2\E N+\mu^2\Var N.$$
Uncertainty in the number of claims contributes even if each claim were known exactly.</p>
</div>
<div class="ex" data-name="Overdispersion from heterogeneity">
<p>Let $\Lambda$ have mean $\mu$ and variance $\tau^2$, and $Y\mid\Lambda\sim$ Poisson$(\Lambda)$. Then $\E Y = \mu$ and $\Var Y = \E\Lambda+\Var\Lambda = \mu+\tau^2\gt\mu$. Unobserved heterogeneity in rates always inflates count variance beyond the Poisson; with a gamma $\Lambda$ the marginal is negative binomial (<a href="#/t/overdispersion">overdispersion</a>). The same decomposition is behind random-effects models, where $\tau^2$ is the between-group variance component.</p>
</div>`,
        lab: 'conditional-expectation-total-variance',
      },
      {
        title: 'The general definition',
        html: R`
<p>Conditioning on a variable with a density used ratios of densities, which are only defined almost everywhere and say nothing about conditioning on events like "$X$ is rational". Kolmogorov's definition conditions on information directly.</p>
<div class="def" data-name="Conditional expectation given a σ-field">
<p>Let $\E|Y|\lt\infty$ and let $\mathcal G\subseteq\mathcal F$ be a σ-field. A random variable $Z$ is a version of $\E[Y\mid\mathcal G]$ if (i) $Z$ is $\mathcal G$-measurable and integrable, and (ii) $\E[Z\ind_G] = \E[Y\ind_G]$ for every $G\in\mathcal G$. We write $\E[Y\mid X] = \E[Y\mid\sigma(X)]$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Existence and uniqueness">
<p>$\E[Y\mid\mathcal G]$ exists, and any two versions agree almost surely.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Uniqueness:</em> if $Z,Z'$ both satisfy (i)–(ii), take $G = \{Z-Z'\gt\epsilon\}\in\mathcal G$; then $0 = \E[(Z-Z')\ind_G]\ge\epsilon\P(G)$, so $\P(G) = 0$ for every $\epsilon\gt0$, and symmetrically. <em>Existence:</em> for $Y\ge0$, $\nu(G) = \E[Y\ind_G]$ is a finite measure on $\mathcal G$ that vanishes on $\P$-null sets, so the Radon–Nikodym theorem provides a $\mathcal G$-measurable density $Z = \d\nu/\d\P$. (For square-integrable $Y$, an alternative proof takes the $L^2$ projection onto the closed subspace of $\mathcal G$-measurable variables.)</p>
</details>
<p>Condition (ii) is the tower property localized to every event in the information set. The Doob–Dynkin lemma says a $\sigma(X)$-measurable variable is $h(X)$ for some Borel $h$, recovering $\E[Y\mid X] = m(X)$.</p>
<div class="warn" data-name="The Borel–Kolmogorov paradox">
<p>Choose a point uniformly on the sphere and condition on it lying on a given great circle (an event of probability zero). If the great circle is treated as the equator ("latitude 0"), the conditional distribution along it is uniform; if the same circle is a meridian ("longitude fixed"), the conditional density is proportional to $\cos(\text{latitude})$. Both are correct: conditioning on a null event is only meaningful relative to the family of events (the σ-field, or the limiting procedure) it belongs to. Densities hide the choice; the general definition makes it explicit.</p>
</div>`,
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="E[Y | X = x] versus E[Y | X]">
<p>The first is a number (a function value), the second a random variable. Statements like "$\E[\E[Y\mid X]] = \E Y$" only make sense for the second. Software that "predicts at $x$" returns the first.</p>
</div>
<div class="warn" data-name="Conditioning is not intervening">
<p>$\E[Y\mid X=x]$ describes what $Y$ tends to be among units that <em>happen</em> to have $X=x$. It is not what $Y$ would be if we <em>set</em> $X=x$ — that is a causal quantity, equal to the conditional mean only without confounding (<a href="#/t/causal-inference">causal inference</a>).</p>
</div>
<div class="warn" data-name="Constant conditional variance is special">
<p>$\Var(Y\mid X)$ is generally a function of $X$ (heteroscedasticity). The MVN's constant conditional variance is the exception, and methods that assume it (ordinary least squares standard errors) need correction otherwise (<a href="#/t/weighted-gls">robust standard errors</a>).</p>
</div>
<div class="warn" data-name="Regression is not symmetric">
<p>$\E[Y\mid X]$ and $\E[X\mid Y]$ are different functions and are not inverses of each other. Even for the bivariate normal, the two regression lines differ unless $|\rho| = 1$; regressing in the "wrong" direction and inverting gives a biased slope (the errors-in-variables problem).</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A store gets $N\sim$ Poisson(50) customers a day; each spends an independent amount with mean 20 and standard deviation 10 (independent of $N$). Find the mean and variance of daily revenue.`,
        a: R`$\E S = 20\cdot50 = 1000$. $\Var S = \sigma^2\E N+\mu^2\Var N = 100\cdot50+400\cdot50 = 25{,}000$ (equivalently $\E N\cdot\E X^2 = 50\cdot500$, the compound-Poisson formula). SD $\approx158$.` },
      { q: R`Why does $\E[(Y-\E[Y\mid X])\,X^3] = 0$ but not necessarily $\E[(Y-\alpha-\beta X)X^3] = 0$ for the best linear predictor?`,
        a: R`The conditional-mean residual is orthogonal to <em>every</em> function of $X$ (it is a projection onto all of them). The linear-predictor residual is orthogonal only to $1$ and $X$, the span it projects onto; $X^3$ lies outside that span.` },
      { q: R`Let $X\sim$ Unif$(0,1)$ and, given $X=x$, $Y\sim$ Unif$(0,x)$. Find $\E Y$ and $\Var Y$ using the iterated laws.`,
        a: R`$\E[Y\mid X] = X/2$ and $\Var(Y\mid X) = X^2/12$. $\E Y = \E X/2 = 1/4$. $\Var Y = \E X^2/12+\Var(X)/4 = \frac{1}{36}+\frac{1}{48} = \frac{7}{144}$.` },
      { q: R`If $\E[Y\mid X] = \E Y$ (mean independence), are $X$ and $Y$ independent? Are they uncorrelated?`,
        a: R`Uncorrelated, yes: $\E XY = \E[X\E[Y\mid X]] = \E X\E Y$. Independent, not necessarily: e.g. $Y = XZ$ with $Z$ independent of $X$, $\E Z = 0$ — the conditional mean of $Y$ is 0 for every $X$, but the conditional variance $X^2\Var Z$ depends on $X$.` },
    ],
  });
})();
