/* Simulation Theory: random variate generation, Monte Carlo integration, variance reduction,
 * quasi-Monte Carlo, sequential Monte Carlo, and the design of simulation studies. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'pseudo-random',
    domain: 'simulation',
    order: 10,
    title: 'Pseudo-Random Number Generation',
    subtitle: 'Deterministic recurrences that are good enough to pass for independent uniforms',
    level: 2,
    sources: { aos: '§24.1', cb: '§5.6', other: 'Knuth, TAOCP Vol. 2, Ch. 3 (1997); L’Ecuyer & Simard (2007); O’Neill (2014)' },
    prereqs: ['random-variables'],
    related: ['inverse-transform', 'monte-carlo-integration', 'simulation-studies', 'quasi-monte-carlo'],
    tags: ['random number generator', 'LCG', 'lattice', 'period', 'spectral test', 'TestU01'],
    summary: R`Every simulation in statistics rests on a stream of numbers that are supposed to behave like independent $\text{Unif}(0,1)$ draws, yet are produced by a deterministic recurrence $s_{n+1} = T(s_n)$, $u_n = G(s_n)$ on a finite state space. Such a sequence is necessarily periodic and has deterministic structure; the art is to make that structure invisible to every computation you are likely to run. Linear congruential generators show the issues in their purest form: number theory decides their period (Hull–Dobell), and geometry decides their quality, because their $d$-tuples lie on a lattice and hence on a small number of parallel hyperplanes. Modern generators (Mersenne Twister, xoshiro, PCG, counter-based Philox) push that structure far beyond what statistical test batteries can detect, which is precisely and only what "passing the tests" certifies.`,
    survey: {
      task: R`Produce a reproducible stream $u_1,u_2,\dots$ that imitates IID $\text{Unif}(0,1)$`,
      assumptions: R`None statistical; quality judged by period, $k$-dimensional equidistribution, and empirical test batteries`,
      output: 'A deterministic sequence fully determined by the seed',
      strengths: 'Fast, reproducible, jumpable into independent streams; good generators pass TestU01 BigCrush',
      weaknesses: 'Finite period and hidden lattice/linear structure; weak generators (RANDU, low LCG bits) bias results silently',
      complexity: R`$O(1)$ per draw; state from 64–128 bits (LCG, PCG) to about 2.5 KB (MT19937)`,
    },
    sections: [
      {
        title: 'What a generator is, and what we ask of it',
        html: R`
<p>A computer cannot produce a random number; it can only run an algorithm. What simulation actually requires is a sequence that, for the purposes of the computation at hand, cannot be told apart from an IID uniform sample. The formal object is simple.</p>
<div class="def" data-name="Pseudo-random number generator">
<p>A generator is a tuple $(\mathcal S, s_0, T, G)$: a finite state space $\mathcal S$, a seed $s_0 \in \mathcal S$, a transition map $T:\mathcal S\to\mathcal S$, and an output map $G:\mathcal S\to[0,1)$. It produces $s_n = T(s_{n-1})$ and $u_n = G(s_n)$. The <em>period</em> $\rho$ is the smallest $\rho\ge1$ such that $s_{n+\rho} = s_n$ for all $n$ beyond some point.</p>
</div>
<p>Because $\mathcal S$ is finite, the sequence of states is eventually periodic and $\rho \le |\mathcal S|$. If $T$ is a bijection, the sequence is purely periodic, so every seed lies on a cycle. Three requirements follow from how the numbers are used:</p>
<ol>
<li><strong>Long period.</strong> A simulation that consumes a sizeable fraction of the period sees the sequence's global structure. For LCGs the birthday-spacings test exposes the lattice once a small multiple of $\rho^{1/3}$ outputs has been drawn, so a conservative rule for such generators is to stay well below $\rho^{1/3}$.</li>
<li><strong>Uniformity in many dimensions.</strong> A Monte Carlo estimator that uses $d$ uniforms per replication is integrating over $[0,1)^d$, so it is the joint behaviour of the overlapping $d$-tuples $(u_n,\dots,u_{n+d-1})$ that matters, not just the one-dimensional histogram.</li>
<li><strong>Speed, reproducibility and splittability.</strong> The same seed must reproduce the same stream, and parallel computations need streams that are provably far apart (or independent by construction).</li>
</ol>
<div class="note">
<p>A useful way to think about it: a generator is good for a given application if the application's output, as a function of the stream, has the same distribution it would have under truly IID uniforms. No generator is good for all functions, because one of those functions is "run the generator's own recurrence and check whether the next value matches." Quality is always relative to a class of tests one considers natural.</p>
</div>`,
      },
      {
        title: 'Linear congruential generators and the Hull–Dobell theorem',
        html: R`
<p>The classical generator (Lehmer, 1949) is the <strong>linear congruential generator</strong> (LCG)
$$x_{n+1} = (a x_n + c) \bmod m,\qquad u_n = x_n/m,$$
with modulus $m$, multiplier $a$ and increment $c$. When $c=0$ it is a <em>multiplicative</em> (Lehmer) generator. Since the state is a single residue, $\rho\le m$. Unrolling the recurrence gives the closed form that drives everything below:
$$x_n \equiv a^n x_0 + c\,(1 + a + \dots + a^{n-1}) \pmod m.$$</p>
<div class="thm" data-name="Hull–Dobell (1962)">
<p>The LCG with $c\neq0$ has full period $m$ (for every seed) if and only if
(i) $\gcd(c,m) = 1$; (ii) every prime $p$ dividing $m$ divides $a-1$; (iii) if $4\mid m$ then $4 \mid a-1$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Necessity.</em> (i) If $d=\gcd(c,m)>1$, starting from $x_0=0$ every iterate is a multiple of $d$, so the residue $1$ is never visited. (ii) If $p\mid m$ but $p\nmid a-1$, then $a-1$ is invertible mod $p$ and the reduced map $x\mapsto ax+c \pmod p$ has a fixed point $x^\star$. Started at any $x_0\equiv x^\star \pmod p$, the sequence stays in that residue class mod $p$ and cannot visit all of $\mathbb Z_m$. (iii) If $4\mid m$ and $a\equiv 3\pmod 4$, then two steps of the map mod 4 give $x\mapsto 9x+4c\equiv x$, so the sequence mod 4 has period at most 2 and misses residues.</p>
<p><em>Sufficiency.</em> The map is a bijection under (ii), so the orbit of $0$ is a cycle; it suffices that its length is $m$. From $x_0=0$, $x_n = c\,S_n$ with $S_n = (a^n-1)/(a-1)$ as an integer. By (i), $x_n\equiv0$ iff $m\mid S_n$. Factor $m=\prod p^{e}$. For each prime $p\mid m$, conditions (ii)–(iii) are exactly the hypotheses of the lifting-the-exponent lemma, which gives $v_p(a^n-1) = v_p(a-1)+v_p(n)$, i.e. $v_p(S_n) = v_p(n)$ (when $p=2$ and $4\nmid m$ one only needs $S_n$ odd iff $n$ odd, true because $S_n$ is a sum of $n$ odd terms). Hence $p^e\mid S_n \iff p^e\mid n$ for every prime power in $m$, so the first return to $0$ is at $n=m$.</p>
</details>
<p>For $m = 2^k$ the conditions read: $c$ odd and $a\equiv1\pmod 4$. For multiplicative generators the best possible period is $m-1$, attained when $m$ is prime and $a$ is a primitive root mod $m$; the "minimal standard" of Park and Miller (1988), $m = 2^{31}-1$, $a = 16807$, is the canonical example.</p>
<div class="thm" data-kind="Proposition" data-name="Low-order bits of power-of-two LCGs">
<p>If $m = 2^k$, the bit $j$ (counting from $j=0$ for the least significant) of $x_n$ has period at most $2^{j+1}$. In particular the lowest bit of a full-period LCG simply alternates.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Because $2^{j+1}\mid 2^k$, reducing the recurrence mod $2^{j+1}$ gives $x_{n+1}\bmod 2^{j+1} = (a\,(x_n\bmod 2^{j+1}) + c)\bmod 2^{j+1}$, an LCG on $2^{j+1}$ states. Its period is at most $2^{j+1}$, and bit $j$ is a function of $x_n \bmod 2^{j+1}$.</p>
</details>
<div class="warn" data-name="Never use the low bits">
<p>Code such as <code>x % 6</code> to roll a die from a power-of-two LCG inherits the short periods of the low bits; parity alternates perfectly. Use the high bits (multiply $u_n$ by 6 and take the floor), or a generator whose output function scrambles the bits.</p>
</div>`,
        lab: 'pseudo-random-lcg-lattice',
      },
      {
        title: 'Geometry: lattices, hyperplanes, and the spectral test',
        html: R`
<p>Full period only says each residue occurs once per cycle; it says nothing about how consecutive values relate. For an LCG they relate very rigidly.</p>
<div class="thm" data-name="Lattice structure (Marsaglia, 1968)">
<p>For an LCG, all overlapping $d$-tuples $\mathbf u_n = (u_n,\dots,u_{n+d-1})$ lie in $(\mathbf b + L_d)\cap[0,1)^d$, where $L_d$ is the lattice $\{\, (k/m)\,\mathbf v + \mathbf z : k\in\mathbb Z, \mathbf z\in\mathbb Z^d\,\}$ with $\mathbf v = (1, a, a^2,\dots,a^{d-1})$, and $\mathbf b$ is a fixed offset (zero when $c=0$). Consequently they lie on a family of parallel hyperplanes; Marsaglia showed one can always find such a family containing all the points with at most $(d!\,m)^{1/d}$ hyperplanes.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>From the closed form, $x_{n+j}\equiv a^jx_n + c_j \pmod m$ with constants $c_j$ not depending on $n$. Hence $(x_n,\dots,x_{n+d-1}) = x_n\mathbf v + \mathbf c + m\mathbf z$ for some integer vector $\mathbf z$; dividing by $m$ gives the lattice statement. Now take any nonzero integer vector $\mathbf h$ with $\mathbf h\cdot\mathbf v\equiv0 \pmod m$ (the <em>dual lattice</em>). For every tuple, $\mathbf h\cdot\mathbf u_n = (x_n\,\mathbf h\cdot\mathbf v)/m + \mathbf h\cdot\mathbf c/m + \mathbf h\cdot\mathbf z \in \mathbf h\cdot\mathbf c/m + \mathbb Z$. So the points sit on the hyperplanes $\{\mathbf h\cdot\mathbf u = \text{const} + k\}$, $k\in\mathbb Z$, which are $1/\lVert\mathbf h\rVert_2$ apart; at most $\lVert\mathbf h\rVert_1+1$ of them meet the unit cube. Marsaglia's bound comes from Minkowski's convex body theorem applied to the dual lattice, which has index $m$ in $\mathbb Z^d$ (determinant $m$), so the cross-polytope $\{\lVert\mathbf h\rVert_1\le (d!\,m)^{1/d}\}$, of volume $2^d m$, must contain a nonzero dual vector.</p>
</details>
<p>Plugging in $m = 2^{32}$: in three dimensions at most about 2,950 planes, in ten dimensions at most about 41. Those 41 hyperplanes, with gaps between them containing no points at all, are a gross departure from uniformity for anyone integrating a function of 10 consecutive uniforms.</p>
<div class="def" data-name="Spectral test">
<p>The <em>spectral test</em> (Coveyou & MacPherson, 1967) measures the largest gap: $\nu_d = \min\{\lVert\mathbf h\rVert_2 : \mathbf h\in\mathbb Z^d\setminus\{0\},\ \mathbf h\cdot\mathbf v\equiv 0 \pmod m\}$, so the maximal distance between adjacent covering hyperplanes is $1/\nu_d$. Since $\nu_d \le \gamma_d^{1/2} m^{1/d}$ (Hermite's constant $\gamma_d$), a multiplier is judged by how close $\nu_d$ comes to that bound for $d = 2,\dots,8$.</p>
</div>
<div class="ex" data-name="RANDU: fifteen planes">
<p>IBM's RANDU used $m = 2^{31}$, $a = 65539 = 2^{16}+3$, $c=0$. Then $a^2 = 2^{32} + 6\cdot2^{16} + 9 \equiv 6(2^{16}+3) - 9 = 6a - 9 \pmod{2^{31}}$, so
$$x_{n+2} \equiv 6x_{n+1} - 9x_n \pmod{2^{31}}.$$
Thus $\mathbf h = (9,-6,1)$ is in the dual lattice and $9u_n - 6u_{n+1} + u_{n+2}$ is always an integer. With each $u\in[0,1)$ that integer lies in $\{-5,\dots,9\}$: every triple falls on one of just 15 parallel planes, spaced $1/\sqrt{118}\approx0.092$ apart. RANDU passes one-dimensional uniformity tests perfectly and was in wide use for a decade. The second lab below lets you rotate the cube until the planes snap into view.</p>
</div>`,
        lab: 'pseudo-random-randu',
      },
      {
        title: 'Modern generators',
        html: R`
<p>The lesson of the LCG era is that linear structure is unavoidable in fast recurrences, so modern designs either make the structure astronomically fine or hide it behind a nonlinear output map.</p>
<table>
<thead><tr><th>Generator</th><th>Idea</th><th>State / period</th><th>Caveats</th></tr></thead>
<tbody>
<tr><td>Mersenne Twister MT19937 (Matsumoto & Nishimura, 1998)</td><td>Twisted generalized feedback shift register; linear over $\mathbb F_2$, primitive characteristic polynomial</td><td>19,937 bits; $2^{19937}-1$; 623-dimensionally equidistributed to 32 bits</td><td>Fails linear-complexity and binary-rank tests in BigCrush; recovers slowly from states with many zero bits; large state</td></tr>
<tr><td>xorshift / xoshiro (Marsaglia, 2003; Blackman & Vigna, 2021)</td><td>Shifts and XORs of the state words ($\mathbb F_2$-linear), followed by a nonlinear scrambler (+, *, rotate)</td><td>128–256 bits; $2^{128}-1$ or $2^{256}-1$</td><td>Raw xorshift fails linearity tests; scrambled versions pass BigCrush; fast jump-ahead</td></tr>
<tr><td>PCG (O’Neill, 2014)</td><td>A 64- or 128-bit power-of-two LCG for the state, then a permutation of the output (xorshift plus a data-dependent rotation) that hides the weak low bits</td><td>64–128 bits; $2^{64}$ or $2^{128}$</td><td>Default in NumPy since 1.17 (PCG64)</td></tr>
<tr><td>Counter-based: Philox, Threefry (Salmon et al., 2011)</td><td>$u_i = G_k(i)$: a keyed bijection (reduced-round block cipher) applied to a counter</td><td>Period $2^{128}$ or more per key</td><td>Trivially parallel: stream $k$, position $i$ is computed directly, no sequential state</td></tr>
</tbody>
</table>
<p>$\mathbb F_2$-linear generators deserve a remark. Their state is a bit vector $\mathbf s\in\mathbb F_2^k$ updated by $\mathbf s_{n+1} = A\mathbf s_n$. If the characteristic polynomial of $A$ is primitive, every nonzero state lies on a single cycle of length $2^k-1$, the analogue of a primitive-root multiplier. The analogue of the spectral test is <em>equidistribution</em>: MT19937 is constructed so that, for every $d\le 623$, the $d$-tuples of 32-bit outputs hit every cell of the $2^{32d}$ partition of $[0,1)^d$ equally often over the period (excluding the all-zero state). This is the bit-level counterpart of Marsaglia's hyperplanes, made fine enough to be irrelevant.</p>
<div class="note" data-name="Converting bits to doubles">
<p>The standard conversion takes the top 53 bits $b$ of a 64-bit output and returns $b\cdot2^{-53}\in[0,1)$. Two consequences: $u=0$ is possible (so $-\log u$ can overflow; use $1-u$ or reject zero), and the smallest positive value is $2^{-53}$, so inversion cannot produce exponential variates beyond about $53\log2\approx 36.7$. That is harmless for most purposes and fatal for naive rare-event simulation (see <a href="#/t/importance-sampling">importance sampling</a>).</p>
</div>`,
      },
      {
        title: 'What "passing the tests" means',
        html: R`
<p>An empirical test is a statistic $T(u_1,\dots,u_N)$ whose distribution under the null hypothesis "the $u_i$ are IID uniform" is known, at least asymptotically. The generator is run, $T$ computed, and its $p$-value examined.</p>
<ul>
<li><strong>Equidistribution / serial tests:</strong> chi-square counts of $d$-tuples in $k^d$ cells.</li>
<li><strong>Gap, run, poker, coupon-collector tests:</strong> classic tests in Knuth's TAOCP Vol. 2.</li>
<li><strong>Birthday spacings</strong> (Marsaglia): drop $n$ points into $N$ cells; the number of repeated spacings is approximately Poisson with mean $n^3/(4N)$. Lattice generators produce far too many repeats.</li>
<li><strong>Linear complexity and matrix rank</strong> over $\mathbb F_2$: designed to catch $\mathbb F_2$-linear generators.</li>
</ul>
<p>Batteries bundle hundreds of such tests: Marsaglia's Diehard, NIST SP 800-22, <strong>TestU01</strong> (L’Ecuyer & Simard, 2007) with SmallCrush, Crush and BigCrush, and PractRand. A generator "fails" a test when the $p$-value is extreme beyond reasonable doubt, typically outside $[10^{-10}, 1-10^{-10}]$, and reproducibly so on re-running with fresh seeds.</p>
<div class="warn" data-name="Interpreting a battery">
<p>Under the null, $p$-values are uniform, so in a battery of 160 tests one expects a couple of $p$-values below $0.01$ from a perfect generator. A single moderately small $p$-value is not a failure; the logic is that of <a href="#/t/multiple-testing">multiple testing</a>. Conversely, passing means only that the particular statistics in the battery could not detect the structure at that sample size. It is evidence about a class of tests, not a proof of randomness: every generator has a test it fails (the one that knows its algorithm).</p>
</div>
<p>Practical rules that follow:</p>
<ol>
<li>Use a modern default (PCG64, xoshiro256**, Philox, or MT19937 where tradition requires it) and never a homemade LCG or a library <code>rand()</code> of unknown provenance.</li>
<li>Seed once, record the seed, and for parallel work use the library's stream-splitting facility (jump-ahead, or counter-based keys). Seeding parallel workers with consecutive integers can produce correlated streams for some generators.</li>
<li>Count how many uniforms you consume relative to the period, especially with 32-bit generators: $2^{32}\approx 4.3\times10^9$ draws is a few seconds of modern computation.</li>
<li>When results look suspicious, rerun with a structurally different generator. If conclusions change, the generator was not adequate for your application.</li>
</ol>`,
      },
      {
        title: 'Beyond uniform: true randomness and its uses',
        html: R`
<p>Physical sources (thermal noise, radioactive decay, the CPU's <code>RDRAND</code> instruction, the operating system's entropy pool) give "true" random bits but are slow, not reproducible, and need post-processing to remove bias. Cryptographically secure generators (ChaCha20, AES in counter mode) promise that no polynomial-time test can distinguish them from random bits, a much stronger guarantee than passing a battery. Counter-based generators such as Philox borrow their design. For statistical simulation the priorities are different: reproducibility trumps unpredictability, so a high-quality seeded PRNG is the right tool, with physical entropy used only to choose seeds.</p>
<p>Everything else in this domain consumes uniforms. <a href="#/t/inverse-transform">Inversion</a>, <a href="#/t/rejection-sampling">rejection</a> and <a href="#/t/transformation-methods">transformations</a> turn them into other distributions; <a href="#/t/monte-carlo-integration">Monte Carlo integration</a> averages functions of them; and <a href="#/t/quasi-monte-carlo">quasi-Monte Carlo</a> deliberately replaces them with deterministic point sets that are <em>more</em> uniform than random, inverting the goal of this topic: there, regular structure is the point.</p>`,
      },
    ],
    quiz: [
      { q: R`An LCG has $m = 2^{32}$, $a = 1664525$, $c = 1013904223$. Does it have full period?`,
        a: R`Hull–Dobell with $m=2^{32}$ requires $c$ odd and $a\equiv1\pmod4$. Here $c$ is odd and $1664525 = 4\cdot416131+1$, so yes: period $2^{32}$ (these are the Numerical Recipes constants). Full period, of course, says nothing about lattice quality.` },
      { q: R`Why do the triples of RANDU fall on exactly 15 planes, and why would no one-dimensional test notice?`,
        a: R`Because $a^2\equiv 6a-9 \pmod{2^{31}}$, the integer $9u_n - 6u_{n+1}+u_{n+2}$ is determined up to the range $\{-5,\dots,9\}$: 15 values, each a plane. The one-dimensional marginal is still essentially uniform (the $2^{29}$ values in a cycle are spread evenly over $[0,1)$), so a histogram of $u_n$ looks perfect; the defect lives only in the joint law of consecutive values.` },
      { q: R`A power-of-two LCG is used to simulate fair coin flips via <code>x % 2</code>. What sequence results?`,
        a: R`The least significant bit has period at most 2, and for a full-period generator it alternates $0,1,0,1,\dots$ The "coin" is perfectly balanced and perfectly predictable. Use the top bit, i.e. $\lfloor 2u_n\rfloor$, instead.` },
      { q: R`A generator passes all of BigCrush. Can it still bias your simulation?`,
        a: R`Yes. Passing means the specific statistics in the battery, at the sample sizes used, detected nothing. If your simulation's output happens to be sensitive to a structure the battery does not probe (say, a linear relation over $\mathbb F_2$ for a raw xorshift, or correlations between badly seeded parallel streams), the result can still be biased. The practical check is to rerun with a structurally different generator.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'inverse-transform',
    domain: 'simulation',
    order: 20,
    title: 'Inverse Transform Sampling',
    subtitle: 'The probability integral transform in both directions, and the generalized inverse that makes it exact',
    level: 1,
    sources: { aos: '§24.1', cb: '§5.6.1, Thm. 2.1.10', other: 'Devroye, Non-Uniform Random Variate Generation (1986), Ch. II–III' },
    prereqs: ['random-variables', 'pseudo-random'],
    related: ['rejection-sampling', 'transformations', 'empirical-cdf', 'quasi-monte-carlo', 'variance-reduction'],
    tags: ['probability integral transform', 'quantile function', 'generalized inverse', 'alias method'],
    summary: R`If $U\sim\text{Unif}(0,1)$ and $F$ is any distribution function, then $X = F^{-}(U)$, with the generalized inverse $F^{-}(u) = \inf\{x: F(x)\ge u\}$, has distribution exactly $F$. Conversely, if $F$ is continuous and $X\sim F$, then $F(X)$ is uniform. These two facts, the probability integral transform, make the quantile function a universal recipe for random variate generation: one uniform in, one variate out, with a monotone map. The method is exact for discrete, continuous and mixed laws alike; its practical limits are only the cost of evaluating $F^{-}$ and the floating-point resolution in the tails. Its monotonicity is what makes it the method of choice for common random numbers, antithetic variates, copulas and quasi-Monte Carlo.`,
    survey: {
      task: R`Generate $X\sim F$ from one uniform`,
      assumptions: R`Ability to evaluate $F^{-}(u)=\inf\{x:F(x)\ge u\}$ (closed form, numerical root, or table)`,
      output: R`Exact draws $X = F^{-}(U)$`,
      strengths: 'Exact for any $F$; one uniform per draw; monotone (preserves correlation structure, works with QMC)',
      weaknesses: 'Needs the quantile function; numerical inversion is slow; tail resolution limited by floating point',
      complexity: R`$O(1)$ closed form; $O(\log K)$ binary search or $O(1)$ alias table for $K$-point laws`,
    },
    sections: [
      {
        title: 'The generalized inverse',
        html: R`
<p>A distribution function $F$ is nondecreasing and right-continuous, but it need not be strictly increasing (flat stretches where there is no mass) or continuous (jumps at atoms). So $F^{-1}$ need not exist. The right replacement is the quantile function.</p>
<div class="def" data-name="Generalized inverse (quantile function)">
<p>For $u\in(0,1)$, $F^{-}(u) = \inf\{x\in\mathbb R : F(x) \ge u\}$. It is nondecreasing and left-continuous. A jump of $F$ of size $p$ at $x_0$ becomes a flat stretch of $F^{-}$ of length $p$ at height $x_0$; a flat stretch of $F$ becomes a jump of $F^{-}$.</p>
</div>
<div class="thm" data-kind="Lemma" data-name="Galois inequality">
<p>For all $u\in(0,1)$ and $x\in\mathbb R$: $\;F^{-}(u)\le x \iff u\le F(x)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>($\Leftarrow$) If $u\le F(x)$ then $x$ belongs to the set $\{y : F(y)\ge u\}$, so its infimum satisfies $F^{-}(u)\le x$.
($\Rightarrow$) The set $\{y: F(y)\ge u\}$ is a nonempty interval $[F^{-}(u),\infty)$ that contains its left endpoint: it is unbounded above because $F\to1>u$, bounded below because $F\to0< u$, and closed on the left by right-continuity, since $F(F^{-}(u)) = \lim_{y\downarrow F^{-}(u)} F(y) \ge u$. So if $F^{-}(u)\le x$, monotonicity gives $F(x)\ge F(F^{-}(u))\ge u$.</p>
</details>
<p>Two by-products are worth remembering: $F(F^{-}(u))\ge u$ with equality iff $u$ is in the range of $F$, and $F^{-}(F(x))\le x$ with equality unless $x$ lies inside a flat stretch of $F$.</p>`,
      },
      {
        title: 'The probability integral transform, both ways',
        html: R`
<div class="thm" data-name="Inversion (quantile transform)">
<p>Let $F$ be any distribution function and $U\sim\text{Unif}(0,1)$. Then $X = F^{-}(U)$ has distribution function $F$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the Galois inequality, $\P(X\le x) = \P(F^{-}(U)\le x) = \P(U\le F(x)) = F(x)$ for every $x$.</p>
</details>
<div class="thm" data-name="Probability integral transform">
<p>If $X\sim F$ and $F$ is continuous, then $F(X)\sim\text{Unif}(0,1)$. If $F$ has an atom, $F(X)$ is not uniform; but with $V\sim\text{Unif}(0,1)$ independent of $X$, the randomized transform $\tilde U = F(X^-) + V\,[F(X) - F(X^-)]$ is exactly uniform for every $F$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Continuous case. Fix $u\in(0,1)$. By the Galois inequality, $F(X)\ge u \iff X\ge F^{-}(u)$, so $\P(F(X)\ge u) = \P(X\ge F^{-}(u)) = 1-F(F^{-}(u)^-) = 1 - F(F^{-}(u))$ by continuity. Continuity and the intermediate value theorem put every $u\in(0,1)$ in the range of $F$, so $F(F^{-}(u)) = u$ and $\P(F(X)\ge u) = 1-u$. Thus $\P(F(X)< u) = u$ for all $u$, and letting $u'\downarrow u$, $\P(F(X)\le u) = u$.</p>
<p>Randomized case. Condition on $X = x$: $\tilde U$ is uniform on $[F(x^-), F(x)]$, an interval of length $\P(X=x)$ (or a point if $x$ is not an atom). The intervals attached to the atoms, together with the values $F(x)$ at continuity points, tile $(0,1)$ in order; integrating the conditional law over the law of $X$ yields Lebesgue measure on $(0,1)$. Equivalently, $\tilde U$ is $X$ "spread out" uniformly over the gap its atom occupies in $F$.</p>
</details>
<div class="note">
<p>The two directions are dual. Inversion says every law is a monotone image of the uniform: the uniform is the "universal" source. The PIT says a continuous law can be pushed back to the uniform, which is why $p$-values are uniform under a simple continuous null, why goodness-of-fit can be checked by testing $F_0(X_i)$ for uniformity, and why a copula $C(u_1,\dots,u_d)$ captures dependence once margins are transformed away. The randomized version underlies randomized quantile residuals for discrete regression models.</p>
</div>`,
        lab: 'inverse-transform-mapping',
      },
      {
        title: 'Closed-form inverses and what they buy',
        html: R`
<table>
<thead><tr><th>Law</th><th>$F(x)$</th><th>$F^{-}(u)$</th></tr></thead>
<tbody>
<tr><td>Exponential($\lambda$)</td><td>$1-e^{-\lambda x}$</td><td>$-\log(1-u)/\lambda$ (or $-\log u/\lambda$, since $1-U\sim U$)</td></tr>
<tr><td>Weibull($k,\lambda$)</td><td>$1-e^{-(x/\lambda)^k}$</td><td>$\lambda(-\log(1-u))^{1/k}$</td></tr>
<tr><td>Cauchy</td><td>$\tfrac12 + \tfrac1\pi\arctan x$</td><td>$\tan(\pi(u-\tfrac12))$</td></tr>
<tr><td>Logistic</td><td>$1/(1+e^{-x})$</td><td>$\log\!\big(u/(1-u)\big)$</td></tr>
<tr><td>Pareto($\alpha$, $x_m$)</td><td>$1-(x_m/x)^\alpha$</td><td>$x_m(1-u)^{-1/\alpha}$</td></tr>
<tr><td>Gumbel</td><td>$\exp(-e^{-x})$</td><td>$-\log(-\log u)$</td></tr>
<tr><td>Geometric($p$) on $\{1,2,\dots\}$</td><td>$1-(1-p)^{\lfloor x\rfloor}$</td><td>$\lceil \log(1-u)/\log(1-p)\rceil$</td></tr>
</tbody>
</table>
<p>The geometric entry shows inversion handling a discrete law through a continuous formula: it is the exponential quantile, discretized by a ceiling.</p>
<div class="ex" data-name="Truncation without rejection">
<p>To sample $X\sim F$ conditioned on $a< X\le b$, note that the conditional CDF is $(F(x)-F(a))/(F(b)-F(a))$ on $(a,b]$, whose quantile function is $F^{-}\big(F(a) + u\,[F(b)-F(a)]\big)$. One uniform per draw, no matter how small $F(b)-F(a)$ is, whereas naive rejection ("draw from $F$ until it lands in the interval") costs $1/(F(b)-F(a))$ draws on average. The catch is numerical: for a normal truncated to $(8,\infty)$, $F(8)$ rounds to 1 in double precision. Work with the survival function $\bar F = 1-F$ (or its logarithm) instead: $X = \bar F^{-1}(u\,\bar F(a))$.</p>
</div>
<div class="ex" data-name="Order statistics directly">
<p>The maximum of $n$ IID uniforms has CDF $u^n$, so it is $U^{1/n}$: one draw instead of $n$. More generally, $F^{-}(U_{(k)})$ is distributed as the $k$-th <a href="#/t/order-statistics">order statistic</a> of an $F$-sample, because $F^{-}$ is monotone. Sampling $U_{(k)}\sim\text{Beta}(k, n-k+1)$ gives any single order statistic of any law in one shot.</p>
</div>`,
      },
      {
        title: 'Discrete distributions: search, tables and aliases',
        html: R`
<p>For $X$ on $x_1< x_2< \dots$ with probabilities $p_k$ and cumulative sums $P_k$, inversion returns the smallest $k$ with $U\le P_k$. The algorithms differ only in how they find $k$.</p>
<div class="algo" data-name="Sequential search">
<ol>
<li>Draw $U$. Set $k=1$, $P=p_1$.</li>
<li>While $U > P$: $k\leftarrow k+1$, $P\leftarrow P+p_k$.</li>
<li>Return $x_k$.</li>
</ol>
</div>
<p>The number of comparisons is exactly the index $K$ returned, so the expected cost is $\E K$. For a Poisson($\lambda$) searched from 0 that is $\lambda+1$: fine for small means, poor for large ones. Starting the search at the mode and moving up or down cuts it to $O(\sqrt\lambda)$. For a fixed finite support of size $K$, binary search on the table $(P_k)$ costs $O(\log K)$, and a guide table (Chen & Asau, 1974) gives $O(1)$ expected time.</p>
<div class="thm" data-kind="Proposition" data-name="Alias decomposition (Walker, 1977)">
<p>Any distribution $(p_1,\dots,p_K)$ can be written as an equal-weight mixture of $K$ distributions, each supported on at most two points: $p = \frac1K\sum_{j=1}^K q_j$, where $q_j$ puts mass $\pi_j$ on $j$ and $1-\pi_j$ on an "alias" $a_j$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Induction on $K$; $K=1$ is trivial. Since the $p_k$ average $1/K$, there is an index $i$ with $p_i\le 1/K$ and an index $j\ne i$ with $p_j\ge 1/K$ (if all equal $1/K$, any pair works). Let $q_i$ put mass $Kp_i$ on $i$ and $1-Kp_i$ on alias $j$. Removing $\frac1K q_i$ from $p$ zeroes coordinate $i$ and leaves $p_j - (1/K - p_i)\ge 0$ at $j$, because $p_j\ge1/K$. What remains is a nonnegative vector on $K-1$ points with total mass $(K-1)/K$; rescaled by $K/(K-1)$ it is a distribution, and the induction hypothesis decomposes it into $K-1$ two-point pieces of weight $1/(K-1)$ each, i.e. $1/K$ each on the original scale.</p>
</details>
<p>Sampling is then $O(1)$: draw $J$ uniform on $\{1,\dots,K\}$, and return $J$ with probability $\pi_J$, else $a_J$. Two uniforms (or one, split into integer and fractional parts of $KU$) per draw after $O(K)$ setup. The alias method is not monotone in $U$, which is the price for constant time.</p>`,
      },
      {
        title: 'Numerical inversion and its pitfalls',
        html: R`
<p>Many important laws (normal, gamma, beta, Student-$t$) have no closed-form quantile function. Options are: a high-accuracy rational approximation (Wichura's AS 241 gives the normal quantile to about $10^{-16}$); root finding on $F(x) = u$ by bisection, secant or Newton ($x\leftarrow x - (F(x)-u)/f(x)$, quadratically convergent near the root if $f$ is positive and smooth there); or interpolation of a precomputed table (the polynomial-interpolation inversion of Derflinger, Hörmann & Leydold, 2010, achieves near machine accuracy after a one-time setup). When none is cheap, <a href="#/t/rejection-sampling">rejection</a> or <a href="#/t/transformation-methods">transformations</a> are usually faster.</p>
<div class="warn" data-name="Floating-point tails">
<p>Uniform doubles are spaced $2^{-53}$ apart near 1 (and the generator's grid is usually no finer near 0). So $-\log(1-U)$ never exceeds about $36.7$, and normal draws by inversion never exceed roughly $8.2$ in absolute value. For ordinary work this is invisible; for estimating $\P(X>10)$ it is the whole answer. Compute in the log or survival scale and use <a href="#/t/importance-sampling">importance sampling</a> for rare events.</p>
</div>
<div class="warn" data-name="Off-by-one at atoms">
<p>With a discrete $F$, the comparison must be $U\le P_k$ (smallest $k$ with $P_k\ge U$). Using $U< P_k$ changes nothing in exact arithmetic (ties have probability zero) but a hand-coded search over integer-valued $x$ that tests $F(x)>U$ while incrementing from the wrong starting point often returns $x+1$. Always validate a discrete sampler by comparing empirical frequencies with the pmf.</p>
</div>
<p><strong>Why monotonicity matters.</strong> Because $F^{-}$ is nondecreasing, inversion maps comonotone uniforms to comonotone variates and antithetic uniforms $(U,1-U)$ to negatively dependent variates. This is exactly what <a href="#/t/variance-reduction">common random numbers and antithetic variates</a> need, and it is the only way to feed <a href="#/t/quasi-monte-carlo">low-discrepancy points</a> into a simulation without destroying their uniformity: each coordinate of a QMC point is mapped by a fixed monotone function, and the discrepancy structure carries over. Rejection methods consume a random number of uniforms and break this correspondence.</p>`,
        lab: 'inverse-transform-pit',
      },
    ],
    quiz: [
      { q: R`Let $X$ be $0$ with probability $0.3$ and Exponential(1) otherwise. Write down $F^{-}(u)$ and check that it returns 0 for 30% of uniforms.`,
        a: R`$F(x) = 0.3 + 0.7(1-e^{-x})$ for $x\ge0$ and $0$ for $x<0$. For $u\le0.3$ the smallest $x$ with $F(x)\ge u$ is $x=0$ (the jump). For $u>0.3$, solve $0.3+0.7(1-e^{-x}) = u$: $x = -\log\big((1-u)/0.7\big)$. So $F^{-}(U) = 0$ exactly when $U\le 0.3$, i.e. with probability $0.3$.` },
      { q: R`If $X\sim\text{Binomial}(10, 0.5)$, is $F(X)$ uniform? What is its distribution, and how do you fix it?`,
        a: R`No. $F(X)$ takes only the 11 values $F(0),\dots,F(10)$, with $\P(F(X)=F(k)) = p_k$; e.g. it equals 1 with probability $2^{-10}$ and never takes values below $F(0)\approx0.001$. The randomized PIT $F(X-1) + V p_X$ with independent $V\sim U(0,1)$ is exactly uniform.` },
      { q: R`You need $10^6$ draws from a fixed 1000-point distribution. Compare sequential search, binary search and the alias method.`,
        a: R`Sequential search costs $\E K$ comparisons per draw (up to $1000$; about $500$ for a near-uniform law). Binary search costs about $\log_2 1000\approx10$. The alias method costs $O(1000)$ setup then two table lookups per draw. For $10^6$ draws the alias method wins; if the draws must be monotone in $U$ (e.g. for QMC or common random numbers), binary search on the CDF table is the right compromise.` },
      { q: R`Why does $\max(U_1,\dots,U_n) \overset{d}{=} U^{1/n}$, and why is this inversion?`,
        a: R`$\P(\max_i U_i\le u) = u^n$, so the max has CDF $G(u)=u^n$ on $[0,1]$, with inverse $G^{-1}(v) = v^{1/n}$. Applying it to one uniform is inversion of $G$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'rejection-sampling',
    domain: 'simulation',
    order: 30,
    title: 'Accept–Reject Sampling',
    subtitle: 'Sampling uniformly under a curve by throwing darts under a bigger one',
    level: 2,
    sources: { aos: '§24.1', cb: '§5.6.2', other: 'von Neumann (1951); Gilks & Wild (1992); Devroye (1986), Ch. II.3' },
    prereqs: ['inverse-transform'],
    related: ['importance-sampling', 'metropolis-hastings', 'slice-sampling', 'transformation-methods'],
    tags: ['envelope', 'acceptance rate', 'squeeze', 'adaptive rejection', 'log-concave'],
    summary: R`To sample from a density $f$ known perhaps only up to a constant, find a proposal $g$ you can sample and a constant $M$ with $f\le Mg$ everywhere. Draw $X\sim g$ and accept it with probability $f(X)/(Mg(X))$. Accepted draws are exactly distributed as $f$, each attempt succeeds with probability $1/M$, and the number of attempts is geometric. The method is exact, needs no normalizing constant, and is the engine behind most fast gamma, beta and normal generators. Its weaknesses are just as clear: $M$ must be a true upper bound (an underestimate silently biases the output), the proposal must have tails at least as heavy as the target, and in high dimension the best possible $M$ grows exponentially. Squeezes and adaptive envelopes for log-concave targets reduce the cost of each attempt toward a single cheap comparison.`,
    survey: {
      task: R`Exact draws from $f$ (possibly unnormalized) using a proposal $g$`,
      assumptions: R`$f(x)\le M g(x)$ for all $x$; $g$ easy to sample; $f$ evaluable pointwise`,
      output: R`IID draws from $f$; attempts per draw $\sim\text{Geometric}(1/M)$`,
      strengths: 'Exact; normalizing constant not needed; simple to verify; squeezes make it very fast',
      weaknesses: R`Needs a global bound; heavy-tailed $g$ required; $M$ grows exponentially with dimension`,
      complexity: R`Expected $M$ proposals and at most $M$ evaluations of $f$ per accepted draw`,
    },
    sections: [
      {
        title: 'The fundamental theorem of simulation',
        html: R`
<p>Rejection sampling rests on a geometric fact: sampling a density is the same as sampling a point uniformly from the region under its graph.</p>
<div class="thm" data-kind="Lemma" data-name="Uniform under the graph">
<p>Let $\tilde f\ge0$ be integrable with $Z = \int\tilde f$, and $A = \{(x,y): 0< y< \tilde f(x)\}$. If $(X,Y)$ is uniform on $A$, then $X$ has density $f = \tilde f/Z$. Conversely, if $X\sim f$ and $V\sim\text{Unif}(0,1)$ independently, then $(X, V\tilde f(X))$ is uniform on $A$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$A$ has area $Z$, so the uniform density on $A$ is $1/Z$. The marginal of $X$ is $\int_0^{\tilde f(x)} \frac1Z\d y = \tilde f(x)/Z$. Conversely, the joint density of $(X, Y=V\tilde f(X))$ is $f(x)\cdot\frac{1}{\tilde f(x)}\ind\{0< y< \tilde f(x)\} = \frac1Z\ind_A(x,y)$.</p>
</details>
<p>So if we can produce uniform points under <em>some</em> curve $Mg\ge\tilde f$, the ones that also lie under $\tilde f$ are uniform under $\tilde f$ (restricting a uniform distribution to a subset leaves it uniform), and their $x$-coordinates are draws from $f$. Uniform points under $Mg$ are easy: draw $X\sim g$ and set $Y = U\cdot Mg(X)$ by the converse part of the lemma.</p>`,
      },
      {
        title: 'The algorithm and its correctness',
        html: R`
<div class="algo" data-name="Accept–reject (von Neumann, 1951)">
<ol>
<li>Draw $X\sim g$ and $U\sim\text{Unif}(0,1)$ independently.</li>
<li>If $U \le \tilde f(X)/(M g(X))$, return $X$. Otherwise go to 1.</li>
</ol>
</div>
<div class="thm" data-name="Correctness and cost">
<p>Suppose $\tilde f(x)\le Mg(x)$ for all $x$, with $\int\tilde f = Z$. Then (a) the returned $X$ has density $f=\tilde f/Z$; (b) each attempt is accepted with probability $Z/M$, which equals $1/M$ when $\tilde f=f$ is normalized; (c) the number of attempts $N$ is $\text{Geometric}(Z/M)$, independent of the returned value, with $\E N = M/Z$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For a single attempt,
$$\P(X\le x,\ \text{accept}) = \int_{-\infty}^x g(y)\,\frac{\tilde f(y)}{Mg(y)}\d y = \frac{1}{M}\int_{-\infty}^x \tilde f(y)\d y.$$
Letting $x\to\infty$ gives $\P(\text{accept}) = Z/M$. Dividing, $\P(X\le x\mid\text{accept}) = \int_{-\infty}^x f$, which is (a). Attempts are IID, so the first success occurs at a geometric time, and since each attempt's accepted value has the same conditional law whatever happened before, the returned value is independent of $N$.</p>
</details>
<p>The condition $\tilde f\le Mg$ has two hidden requirements. First, $g>0$ wherever $f>0$. Second, $f/g$ must be bounded, which in practice means $g$ must have <strong>tails at least as heavy as $f$</strong>. A Cauchy proposal can envelope a normal target; a normal proposal can never envelope a Cauchy, because $f/g\propto e^{x^2/2}/(1+x^2)\to\infty$.</p>
<div class="note">
<p>Because the algorithm only ever uses the ratio $\tilde f/(Mg)$, any unknown normalizing constant of $f$ is simply absorbed into $M$. This is why rejection sampling (and its cousin <a href="#/t/metropolis-hastings">Metropolis–Hastings</a>) applies directly to posteriors known only up to $p(y)$.</p>
</div>`,
        lab: 'rejection-sampling-darts',
      },
      {
        title: 'Choosing the envelope',
        html: R`
<p>For a given $g$, the best constant is $M^\star = \sup_x f(x)/g(x)$, and the acceptance rate $1/M^\star$ then measures how well $g$ matches $f$'s shape. Within a parametric family $g_\lambda$ one chooses $\lambda$ to minimize $\sup_x f/g_\lambda$, a minimax problem that is usually solvable by calculus.</p>
<div class="ex" data-name="Normal from Cauchy and from Laplace">
<p><em>Cauchy proposal.</em> $f(x)/g(x) = \sqrt{\pi/2}\,(1+x^2)e^{-x^2/2}$. Differentiating, $(1+x^2)e^{-x^2/2}$ has derivative $x(1-x^2)e^{-x^2/2}$, maximized at $x=\pm1$ with value $2e^{-1/2}$. So $M^\star = \sqrt{2\pi/e}\approx1.520$, acceptance $\approx0.658$.</p>
<p><em>Laplace proposal</em> $g_\lambda(x) = \frac\lambda2 e^{-\lambda|x|}$. Here $f/g_\lambda = \frac{2}{\lambda\sqrt{2\pi}}\exp(-x^2/2+\lambda|x|)$, maximized at $|x|=\lambda$ with value $\frac{2}{\lambda\sqrt{2\pi}}e^{\lambda^2/2}$. Minimizing $\lambda^2/2-\log\lambda$ gives $\lambda=1$ and $M^\star = \sqrt{2e/\pi}\approx1.315$, acceptance $\approx0.760$. The Laplace wins because its tails, while heavier than normal, are far lighter than Cauchy, so less proposal mass is wasted out there.</p>
</div>
<div class="ex" data-name="Beta from uniform">
<p>For $f = \text{Beta}(a,b)$ with $a,b\ge1$ and $g$ uniform on $(0,1)$, $M^\star$ is the density at the mode $(a-1)/(a+b-2)$. For Beta(2,2), $M^\star = 1.5$; for Beta(20,20), $M^\star\approx5.0$. The uniform proposal wastes effort as the target concentrates, which is why practical beta generators instead use ratios of gammas or tailored envelopes (Cheng, 1978).</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Dimension">
<p>Let $f = \N(0,I_d)$ and $g = \N(0,\sigma^2I_d)$ with $\sigma\ge1$. Then $M^\star = \sigma^d$, so the acceptance probability is $\sigma^{-d}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$f(x)/g(x) = \sigma^d\exp\!\big(-\tfrac{\lVert x\rVert^2}{2}(1-\sigma^{-2})\big)$, whose supremum, at $x=0$, is $\sigma^d$ since $1-\sigma^{-2}\ge0$.</p>
</details>
<p>Even a proposal only 10% wider than the target in each coordinate accepts with probability $1.1^{-100}\approx 7\times10^{-5}$ in $d=100$. Exact proposal matching is impossible in practice, so rejection sampling is a low-dimensional tool; in high dimension one turns to <a href="#/t/mcmc-foundations">MCMC</a>, which trades exactness and independence for local moves.</p>`,
      },
      {
        title: 'Squeezes: avoid evaluating the target',
        html: R`
<p>Often evaluating $f$ is the expensive step (it may involve logs, special functions or a likelihood over many data points). Marsaglia (1977) proposed a cheap <strong>lower bound</strong> $s(x)\le f(x)$, the squeeze.</p>
<div class="algo" data-name="Accept–reject with squeeze">
<ol>
<li>Draw $X\sim g$, $U\sim\text{Unif}(0,1)$; set $W = U\,Mg(X)$.</li>
<li>If $W\le s(X)$: accept (no evaluation of $f$).</li>
<li>Else if $W\le f(X)$: accept. Else reject and repeat.</li>
</ol>
</div>
<div class="thm" data-kind="Proposition" data-name="Cost with a squeeze">
<p>With $f$ normalized, $0\le s\le f\le Mg$, the output is still exactly $f$-distributed, and the expected number of evaluations of $f$ per returned draw is $M - \int s(x)\d x$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The acceptance event is unchanged (step 2 only accepts points already under $f$), so correctness is as before. On each attempt, $f$ is evaluated iff $W> s(X)$, which has probability $1 - \int g\cdot\frac{s}{Mg} = 1-\frac{\int s}{M}$. The number of attempts $N$ has mean $M$ and is a stopping time for the IID attempt sequence, so by Wald's identity the expected number of evaluations is $M(1 - \int s/M) = M-\int s$.</p>
</details>
<p>With a tight envelope ($M\approx1$) and a tight squeeze ($\int s\approx1$), $f$ is almost never evaluated. The Marsaglia–Tsang (2000) gamma generator, used in many libraries, is a rejection sampler on a transformed normal proposal whose squeeze $1-0.0331x^4$ accepts the large majority of proposals before any logarithm is taken.</p>`,
      },
      {
        title: 'Adaptive rejection sampling for log-concave targets',
        html: R`
<p>When $h=\log\tilde f$ is concave, an envelope and a squeeze can be built automatically and refined as sampling proceeds (Gilks & Wild, 1992). The method was designed for full conditionals in <a href="#/t/gibbs-sampling">Gibbs sampling</a>, which are frequently log-concave, and it powered the original BUGS software.</p>
<p>Given abscissae $x_1< \dots< x_k$ with $h(x_j)$ and $h'(x_j)$ computed:</p>
<ul>
<li><strong>Upper hull</strong> $u_k(x) = \min_j \{h(x_j) + h'(x_j)(x-x_j)\}$, the lower envelope of the tangent lines. Concavity gives $h(x)\le h(x_j)+h'(x_j)(x-x_j)$ for each $j$, hence $h\le u_k$.</li>
<li><strong>Lower hull</strong> $l_k(x)$: the chords between consecutive $(x_j, h(x_j))$ on $[x_1,x_k]$, and $-\infty$ outside. Concavity puts the graph above each chord, hence $l_k\le h$.</li>
</ul>
<p>The envelope $e^{u_k}$ is piecewise exponential, so it can be normalized in closed form and sampled exactly by choosing a piece with probability proportional to its mass, then inverting within the piece. The squeeze $e^{l_k}$ is free to evaluate.</p>
<div class="algo" data-name="Adaptive rejection sampling (tangent version)">
<ol>
<li>Draw $X\propto e^{u_k}$, $W\sim\text{Unif}(0,1)$.</li>
<li>If $W\le \exp(l_k(X) - u_k(X))$, accept.</li>
<li>Else evaluate $h(X)$, $h'(X)$; accept if $W\le\exp(h(X)-u_k(X))$. In either case add $X$ to the abscissae, updating $u$ and $l$.</li>
</ol>
</div>
<p>Each attempt is a valid rejection step for the current envelope, and the current envelope is a function of the past only. So, conditionally on everything that happened before, each accepted value has law $f$: the accepted draws are exactly IID from $f$ even though the proposal keeps changing. As abscissae accumulate where the target has mass, $\int e^{u_k}\big/\int e^{h}\to1$ and evaluations of $h$ become rare. Without derivatives, secants through adjacent points give a valid (slightly looser) hull (Gilks, 1992). For targets that are not log-concave, ARMS (Gilks, Best & Tan, 1995) adds a Metropolis–Hastings correction; the output is then a Markov chain rather than independent draws.</p>`,
        lab: 'rejection-sampling-ars',
      },
      {
        title: 'Failure modes and connections',
        html: R`
<div class="warn" data-name="An envelope that is not an envelope">
<p>If $M$ is too small, so that $\tilde f>Mg$ on some set, the algorithm does not fail loudly. It returns draws from the density proportional to $\min(\tilde f, Mg)$: wherever the target sticks out above the envelope, it is clipped. Proof: repeat the correctness calculation with acceptance probability $\min\{1, \tilde f/(Mg)\}$. The bias concentrates near the modes or tails, exactly where $M$ was underestimated. Bound $\sup f/g$ analytically, or with a generous safety factor plus a run-time check that flags any $\tilde f(X)/(Mg(X))>1$.</p>
</div>
<div class="warn" data-name="Unbounded cost">
<p>The expected cost is $M$, but the tail of the geometric number of attempts matters in vectorized code: with $M=20$, a batch of $10^6$ draws will contain attempts needing hundreds of tries. Also, rejection consumes a random number of uniforms, which breaks the synchronization needed for <a href="#/t/variance-reduction">common random numbers</a> and <a href="#/t/quasi-monte-carlo">QMC</a>.</p>
</div>
<p><strong>Relatives.</strong> <a href="#/t/importance-sampling">Importance sampling</a> uses the same ratio $f/g$ but keeps every draw with a weight instead of discarding some; it needs no bound $M$ but yields weighted, not exact, samples. The <a href="#/t/metropolis-hastings">independence Metropolis–Hastings sampler</a> with proposal $g$ accepts with probability $\min\{1, w(X')/w(X)\}$, $w=f/g$, and works even when $f/g$ is unbounded, at the price of dependence. <a href="#/t/slice-sampling">Slice sampling</a> is the Markov-chain version of the "uniform under the graph" lemma. <a href="#/t/transformation-methods">Ratio-of-uniforms</a> is rejection sampling in a cleverly transformed plane.</p>`,
      },
    ],
    quiz: [
      { q: R`You sample a Gamma(3,1) target with an Exponential(rate $1/3$) proposal. Is there a finite $M$? What if the proposal rate is $1$?`,
        a: R`$f/g\propto x^2e^{-x}/e^{-x/3} = x^2e^{-2x/3}$, bounded on $(0,\infty)$, so a finite $M$ exists. With rate 1, $f/g\propto x^2$, unbounded: the proposal's tail is not heavier than the target's, so no envelope exists. (Rate $1/\alpha$ is the optimal choice for Gamma($\alpha$) with $\alpha\ge1$.)` },
      { q: R`Accept–reject is run with an unnormalized target $\tilde f$ with $\int\tilde f = 7$ and $\tilde f\le 10g$. What fraction of proposals is accepted?`,
        a: R`$Z/M = 7/10 = 0.7$. The normalizing constant enters the acceptance rate but not the correctness of the output.` },
      { q: R`Someone uses $M = 1.2$ to sample a standard normal from a Cauchy proposal. What distribution do they actually get?`,
        a: R`Since $\sup f/g = 1.52>1.2$, they get the density proportional to $\min\{f, 1.2g\}$: the normal is clipped by $1.2\times$Cauchy near $x=\pm1$ where $f/g$ peaks, producing a distribution with too little mass around $|x|\approx1$ and correspondingly too much elsewhere after renormalization. Acceptance rate looks like roughly $1/1.2$, giving no warning.` },
      { q: R`Why is adaptive rejection sampling restricted (in its exact form) to log-concave densities?`,
        a: R`The upper hull built from tangent lines is an upper bound for $\log f$ only if $\log f$ is concave (tangents lie above a concave function). For non-log-concave targets the tangents can cut below the density, so the "envelope" is not one; ARMS corrects for this with a Metropolis–Hastings step, at the cost of exactness and independence.` },
      { q: R`With a squeeze $s$ satisfying $\int s = 0.95$ and envelope constant $M = 1.05$, how many target evaluations are needed per accepted draw on average?`,
        a: R`$M-\int s = 1.05 - 0.95 = 0.10$: one evaluation of $f$ per ten draws.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'transformation-methods',
    domain: 'simulation',
    order: 40,
    title: 'Transformation Methods: Box–Muller, Composition & Ratio-of-Uniforms',
    subtitle: 'Change of variables as a generator: polar coordinates, mixtures, and a rejection region in the ratio plane',
    level: 2,
    sources: { cb: '§5.6, §4.3, §4.6', other: 'Box & Muller (1958); Marsaglia & Bray (1964); Kinderman & Monahan (1977); Devroye (1986)' },
    prereqs: ['transformations', 'inverse-transform'],
    related: ['rejection-sampling', 'multivariate', 'distribution-families', 'pseudo-random'],
    tags: ['Box–Muller', 'polar method', 'mixture', 'composition', 'ratio-of-uniforms', 'Cholesky'],
    summary: R`Inversion needs the quantile function and rejection needs an envelope; a third route is to find a map $T$ that carries an easy distribution exactly onto the target. The multivariate change-of-variables formula is both the design tool and the proof. Box–Muller turns two uniforms into two independent normals by reading them as polar coordinates; Marsaglia's polar variant removes the trigonometry with a cheap rejection step. Composition samples a mixture by first sampling the component, which turns scale mixtures (Student-$t$, negative binomial) and piecewise decompositions (the ziggurat) into simple two-stage recipes. Ratio-of-uniforms combines a transformation with rejection: draw a point uniformly from a bounded planar region, return the ratio of its coordinates.`,
    survey: {
      task: 'Exact variates via an explicit map from simple inputs',
      assumptions: R`A known transformation, mixture representation, or bounded ratio-of-uniforms region $C_h$`,
      output: 'Exact draws; often several per input vector (Box–Muller gives two normals)',
      strengths: 'Exact, fast, often branch-free; multivariate laws (MVN, Dirichlet, sphere) follow from univariate ones',
      weaknesses: 'Each law needs its own representation; sensitive to generator defects (Neave effect); some maps are not monotone',
      complexity: R`$O(1)$ per draw; $O(d^3)$ setup for a $d$-variate normal (Cholesky), then $O(d^2)$ per draw`,
    },
    sections: [
      {
        title: 'Change of variables as a design principle',
        html: R`
<p>All transformation methods are applications of one theorem from <a href="#/t/transformations">transformations of random variables</a>.</p>
<div class="thm" data-name="Multivariate change of variables">
<p>Let $\mathbf X$ have density $f_{\mathbf X}$ on an open set $A\subseteq\R^d$ and let $T:A\to B$ be a bijection with continuously differentiable inverse. Then $\mathbf Y = T(\mathbf X)$ has density $f_{\mathbf Y}(\mathbf y) = f_{\mathbf X}(T^{-1}(\mathbf y))\,\lvert\det J_{T^{-1}}(\mathbf y)\rvert$ on $B$.</p>
</div>
<p>To build a generator one reads the theorem backwards: start from the target density, find coordinates in which it factorizes into pieces with easy samplers, and map back. The classic constructions are summarized below; each follows from the theorem or from the definition of the target.</p>
<table>
<thead><tr><th>Target</th><th>Representation</th></tr></thead>
<tbody>
<tr><td>$\chi^2_k$</td><td>$\sum_{i=1}^k Z_i^2$, or $2\,\text{Gamma}(k/2)$</td></tr>
<tr><td>Gamma($n$), integer $n$</td><td>$-\sum_{i=1}^n\log U_i = -\log\prod U_i$</td></tr>
<tr><td>Beta($a,b$)</td><td>$G_a/(G_a+G_b)$ with independent Gamma($a$), Gamma($b$)</td></tr>
<tr><td>Dirichlet($\boldsymbol\alpha$)</td><td>$(G_1,\dots,G_K)/\sum G_k$, $G_k\sim\text{Gamma}(\alpha_k)$</td></tr>
<tr><td>Student $t_\nu$</td><td>$Z/\sqrt{V/\nu}$, $V\sim\chi^2_\nu$ independent of $Z$</td></tr>
<tr><td>Cauchy</td><td>$Z_1/Z_2$</td></tr>
<tr><td>$\N_d(\bmu,\bSigma)$</td><td>$\bmu + L\mathbf Z$ with $\bSigma = LL^\top$ (Cholesky)</td></tr>
<tr><td>Uniform on the sphere $S^{d-1}$</td><td>$\mathbf Z/\lVert\mathbf Z\rVert$</td></tr>
<tr><td>Uniform order statistics</td><td>$U_{(k)} = S_k/S_{n+1}$, $S_k = E_1+\dots+E_k$, $E_i\iid\text{Exp}(1)$</td></tr>
</tbody>
</table>
<p>The multivariate normal row is worth a line of proof: $\Cov(\bmu+L\mathbf Z) = L\,\Cov(\mathbf Z)\,L^\top = LL^\top = \bSigma$, and an affine image of a Gaussian vector is Gaussian. Any square root works (the symmetric one, or $Q\Lambda^{1/2}$ from an eigendecomposition, which also handles singular $\bSigma$); Cholesky is the cheapest. The sphere row uses rotation invariance: the density of $\mathbf Z$ depends only on $\lVert\mathbf Z\rVert$, so $\mathbf Z/\lVert\mathbf Z\rVert$ is invariant under every orthogonal map, and the uniform law is the only rotation-invariant law on the sphere.</p>`,
      },
      {
        title: 'Box–Muller and the polar method',
        html: R`
<div class="thm" data-name="Box–Muller (1958)">
<p>Let $U_1, U_2\iid\text{Unif}(0,1)$, $R = \sqrt{-2\log U_1}$ and $\Theta = 2\pi U_2$. Then $Z_1 = R\cos\Theta$ and $Z_2 = R\sin\Theta$ are independent $\N(0,1)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Run the argument forwards from the answer. If $Z_1,Z_2\iid\N(0,1)$, their joint density is $\frac1{2\pi}e^{-(z_1^2+z_2^2)/2}$. In polar coordinates $(z_1,z_2) = (r\cos\theta, r\sin\theta)$, with Jacobian $r$, the density of $(R,\Theta)$ is $\frac{1}{2\pi}\cdot re^{-r^2/2}$ on $(0,\infty)\times(0,2\pi)$. It factorizes: $\Theta\sim\text{Unif}(0,2\pi)$ independent of $R$, which has density $re^{-r^2/2}$. Then $R^2$ has density $\frac12 e^{-s/2}$, i.e. $R^2\sim\text{Exp}(\text{rate }1/2)=\chi^2_2$, whose quantile function gives $R^2 = -2\log(1-U_1)\overset{d}{=}-2\log U_1$. Since the polar map is a bijection off a null set, generating $(R,\Theta)$ this way and mapping back produces exactly the joint law of two independent normals.</p>
</details>
<p>Box–Muller uses one log, one square root and two trigonometric evaluations per pair. Marsaglia and Bray (1964) replaced the trigonometry with rejection.</p>
<div class="thm" data-kind="Proposition" data-name="Polar method (Marsaglia–Bray)">
<p>Let $(V_1,V_2)$ be uniform on the unit disk minus the origin, $S = V_1^2+V_2^2$. Then $Z_i = V_i\sqrt{-2\log S/S}$, $i=1,2$, are independent $\N(0,1)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $(V_1,V_2) = \sqrt S(\cos\Theta,\sin\Theta)$. For a uniform point in the disk, $\P(S\le s) = \pi s/\pi = s$, so $S\sim\text{Unif}(0,1)$; by rotational symmetry $\Theta$ is uniform on $(0,2\pi)$ and independent of $S$. Then $Z_1 = \sqrt{-2\log S}\cos\Theta$, $Z_2 = \sqrt{-2\log S}\sin\Theta$, which is Box–Muller with $U_1 = S$ and $U_2 = \Theta/2\pi$.</p>
</details>
<p>A uniform point in the disk is obtained by rejection from the square $[-1,1]^2$, accepting with probability $\pi/4\approx0.785$. On average $8/\pi\approx2.55$ uniforms buy two normals with no trigonometric calls; the cosine and sine of $\Theta$ come for free as $V_i/\sqrt S$.</p>
<div class="warn" data-name="The Neave effect">
<p>Box–Muller is a nonlinear map that amplifies dependence between consecutive uniforms. Neave (1973) showed that feeding it successive outputs of a multiplicative LCG makes the pairs $(Z_1,Z_2)$ lie on a spiral and visibly distorts the frequencies of tail values. With a modern generator the issue disappears; the lesson is that transformation methods expose two-dimensional structure in the uniform stream that a histogram of the uniforms would never show. The first lab lets you see this with a deliberately poor LCG.</p>
</div>`,
        lab: 'transformation-methods-box-muller',
      },
      {
        title: 'Composition: sampling mixtures',
        html: R`
<p>If the target is a mixture,
$$f(x) = \sum_{j} p_j\, f_j(x)\qquad\text{or}\qquad f(x) = \int g(x\mid y)\,h(y)\d y,$$
then sampling the latent label first is exact.</p>
<div class="thm" data-kind="Proposition" data-name="Composition method">
<p>If $Y\sim h$ and, given $Y=y$, $X\sim g(\cdot\mid y)$, then $X$ has density $f(x) = \int g(x\mid y)h(y)\d y$ (and the discrete case with $Y=J\sim(p_j)$).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\P(X\in A) = \E[\P(X\in A\mid Y)] = \int\!\int_A g(x\mid y)\d x\,h(y)\d y = \int_A f(x)\d x$ by Tonelli.</p>
</details>
<div class="ex" data-name="Scale and rate mixtures">
<p><em>Student-$t$:</em> if $W\sim\text{Gamma}(\nu/2,\ \text{rate }\nu/2)$ and $X\mid W\sim\N(0, 1/W)$, then $X\sim t_\nu$. This is the table's $Z/\sqrt{V/\nu}$ read as a mixture, and it is how $t$ errors enter Gibbs samplers as latent precisions.<br>
<em>Negative binomial:</em> if $\Lambda\sim\text{Gamma}(\text{shape } r,\ \text{scale }(1-p)/p)$ and $X\mid\Lambda\sim\text{Poisson}(\Lambda)$, then $X$ is negative binomial, the number of failures before the $r$-th success with success probability $p$ (compare generating functions: $(1+\theta(1-s))^{-r}$ with $\theta = (1-p)/p$ equals $\big(p/(1-(1-p)s)\big)^r$). This is the standard model for <a href="#/t/overdispersion">overdispersed counts</a>.<br>
<em>Laplace:</em> an Exp(1) with an independent random sign.</p>
</div>
<p>Composition is also a design tool for speed: split a hard density into a big easy piece plus small awkward pieces, $f = p_1f_1 + (1-p_1)f_2$ with $p_1$ close to 1 and $f_1$ cheap. The <strong>ziggurat</strong> method (Marsaglia & Tsang, 2000) covers a monotone density with a stack of equal-area horizontal rectangles; with about 98–99% probability a draw lands strictly inside a rectangle and is returned after one table lookup, one multiplication and one comparison; only the remaining cases fall back to a rejection step or a tail algorithm. It is the default normal and exponential generator in several numerical libraries.</p>`,
      },
      {
        title: 'Ratio-of-uniforms',
        html: R`
<p>Kinderman and Monahan (1977) found a transformation that turns sampling a density into sampling a uniform point from a planar region, which is then done by rejection from a rectangle.</p>
<div class="thm" data-name="Ratio-of-uniforms (Kinderman & Monahan, 1977)">
<p>Let $h\ge0$ be integrable and $C_h = \{(u,v): 0< u\le\sqrt{h(v/u)}\}$. Then $C_h$ has area $\tfrac12\int h$, and if $(U,V)$ is uniform on $C_h$, then $X = V/U$ has density $h/\int h$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Change variables $(u,v)\mapsto(u,x)$ with $x = v/u$, i.e. $v = ux$; the Jacobian of $(u,x)\mapsto(u,v)$ is $\lvert\partial(u,v)/\partial(u,x)\rvert = u$. The region becomes $\{(u,x): 0< u\le\sqrt{h(x)}\}$ and
$$\text{area}(C_h) = \int\!\!\int_0^{\sqrt{h(x)}} u\d u\d x = \int\tfrac12 h(x)\d x.$$
If $(U,V)$ is uniform on $C_h$, the joint density of $(U,X)$ is $u/\text{area}(C_h)$ on that region; integrating out $u$ gives $\tfrac12h(x)/\text{area}(C_h) = h(x)/\int h$.</p>
</details>
<p>If $h(x)$ and $x^2h(x)$ are bounded, $C_h$ fits in the rectangle
$$0< u\le a = \sup_x\sqrt{h(x)},\qquad b_- = \inf_x x\sqrt{h(x)}\ \le v\le\ b_+ = \sup_x x\sqrt{h(x)},$$
because $v = xu\le x\sqrt{h(x)}$ when $x\ge0$ (and symmetrically for $x< 0$). Sample uniformly in the rectangle and accept iff $u^2\le h(v/u)$; the acceptance probability is $\tfrac12\int h\big/[a(b_+-b_-)]$.</p>
<div class="ex" data-name="Three regions">
<p><em>Normal</em>, $h = e^{-x^2/2}$: $a=1$ and $b_\pm = \pm\sqrt{2/e}$ (maximize $xe^{-x^2/4}$ at $x=\sqrt2$). The acceptance test $u^2\le e^{-v^2/(2u^2)}$ simplifies to $v^2\le -4u^2\log u$, and the acceptance rate is $\frac{\sqrt{2\pi}/2}{2\sqrt{2/e}} = \sqrt{\pi e}/4\approx0.731$.<br>
<em>Cauchy</em>, $h = 1/(1+x^2)$: the condition $u^2\le u^2/(u^2+v^2)$ is $u^2+v^2\le1$. $C_h$ is a half-disk, and $V/U$ for a uniform point in a half-disk is Cauchy: the polar method's geometry in a different guise. Acceptance from the rectangle $[0,1]\times[-1,1]$ is $\pi/4$.<br>
<em>Gamma(3)</em>, $h = x^2e^{-x}$: $a = 2/e$ (at $x=2$), $b_- = 0$, $b_+ = 16/e^2$ (at $x=4$), $\int h = 2$; acceptance $1/(ab_+) = e^3/32\approx0.628$.</p>
</div>
<p>The region $C_h$ is convex whenever $-1/\sqrt{h}$ is concave (in particular whenever $h$ is log-concave), which is the basis of automatic, table-driven ratio-of-uniforms samplers (Leydold, 2000). Cheap squeezes (inner polygons) and tighter bounding regions push the effective cost close to a single comparison.</p>`,
        lab: 'transformation-methods-rou',
      },
      {
        title: 'Choosing among methods',
        html: R`
<table>
<thead><tr><th>Method</th><th>Exact?</th><th>Uniforms per draw</th><th>Monotone in $U$?</th><th>Best for</th></tr></thead>
<tbody>
<tr><td><a href="#/t/inverse-transform">Inversion</a></td><td>yes</td><td>1</td><td>yes</td><td>QMC, common random numbers, truncation, discrete laws</td></tr>
<tr><td><a href="#/t/rejection-sampling">Rejection</a></td><td>yes</td><td>random, mean $2M$</td><td>no</td><td>Densities with a good envelope; unnormalized targets</td></tr>
<tr><td>Transformation</td><td>yes</td><td>fixed (Box–Muller: 1 per normal)</td><td>not in general</td><td>Families with structural representations: normal, gamma, beta, MVN, sphere</td></tr>
<tr><td>Composition</td><td>yes</td><td>1 + those of the component</td><td>no</td><td>Mixtures, latent-variable models, ziggurat-style speedups</td></tr>
<tr><td>Ratio-of-uniforms</td><td>yes</td><td>random, mean $2/\text{acc.}$</td><td>no</td><td>Log-concave and $T$-concave densities, automatic samplers</td></tr>
</tbody>
</table>
<div class="warn" data-name="Exactness is about the math, not the arithmetic">
<p>Every method above is exact for exact uniforms and exact arithmetic. In floating point, Box–Muller never produces $\lvert Z\rvert$ beyond $\sqrt{-2\log 2^{-53}}\approx 8.57$ (with 53-bit uniforms), and $U_1 = 0$ must be excluded. Library implementations are carefully engineered at these edges; reimplementations often are not.</p>
</div>
<p>In practice, use library generators for standard families; they encode decades of tuning. Transformation thinking is still indispensable for nonstandard targets: a Bayesian model's full conditional, a copula, a distribution on a manifold, or a mixture you need to sample inside a larger simulation.</p>`,
      },
    ],
    quiz: [
      { q: R`In Box–Muller, why must the two outputs $Z_1, Z_2$ be independent, even though both are computed from the same $R$?`,
        a: R`Because $(R,\Theta)$ has exactly the joint law of the polar coordinates of two independent standard normals: $\Theta$ uniform and independent of $R$, $R^2\sim\chi^2_2$. The polar map back to Cartesian coordinates reproduces the joint density $\frac1{2\pi}e^{-(z_1^2+z_2^2)/2}$, which factorizes. Sharing $R$ is what the joint law requires, not a source of dependence.` },
      { q: R`How many uniforms does the polar method need, on average, per normal variate?`,
        a: R`Each attempt uses 2 uniforms and succeeds with probability $\pi/4$, so a successful pair costs $2\cdot4/\pi = 8/\pi\approx2.55$ uniforms and yields 2 normals: about $1.27$ uniforms per normal.` },
      { q: R`Show that if $(U,V)$ is uniform on the half-disk $\{u>0, u^2+v^2\le1\}$, then $V/U$ is standard Cauchy.`,
        a: R`For $h(x) = 1/(1+x^2)$, $C_h = \{0< u\le\sqrt{h(v/u)}\} = \{u^2\le u^2/(u^2+v^2)\} = \{u^2+v^2\le1, u>0\}$, the half-disk. By the ratio-of-uniforms theorem $V/U$ has density $\propto 1/(1+x^2)$. Equivalently, $V/U = \tan\Phi$ for an angle $\Phi$ uniform on $(-\pi/2,\pi/2)$, which is inversion of the Cauchy CDF.` },
      { q: R`To generate $\N_2(\mathbf 0, \bSigma)$ with $\bSigma = \begin{pmatrix}1&\rho\\\rho&1\end{pmatrix}$, give an explicit $L$.`,
        a: R`The Cholesky factor is $L = \begin{pmatrix}1&0\\\rho&\sqrt{1-\rho^2}\end{pmatrix}$, so $X_1 = Z_1$, $X_2 = \rho Z_1 + \sqrt{1-\rho^2}Z_2$. Check: $\Var X_2 = \rho^2 + 1-\rho^2 = 1$ and $\Cov(X_1,X_2) = \rho$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'monte-carlo-integration',
    domain: 'simulation',
    order: 50,
    title: 'Monte Carlo Integration & Its Error',
    subtitle: 'Integrals as expectations, and an error rate of n^(-1/2) that does not care about dimension',
    level: 1,
    sources: { aos: '§24.2', cb: '§5.6', other: 'Robert & Casella, Monte Carlo Statistical Methods (2004), Ch. 3; Owen, Monte Carlo Theory, Methods and Examples (2013)' },
    prereqs: ['lln', 'clt'],
    related: ['importance-sampling', 'variance-reduction', 'quasi-monte-carlo', 'simulation-studies', 'bootstrap'],
    tags: ['law of large numbers', 'CLT', 'standard error', 'curse of dimensionality', 'quadrature'],
    summary: R`Write the integral you want as an expectation, $I = \int h(x)f(x)\d x = \E_f h(X)$, and estimate it by the average of $h$ over IID draws from $f$. The estimator is unbiased, converges almost surely by the strong law, and by the central limit theorem has error of order $\sigma/\sqrt n$ with $\sigma^2 = \Var_f h(X)$, which you estimate from the same draws and report as a standard error. The rate $n^{-1/2}$ is slow, but it does not depend on the dimension of $x$, whereas product-grid quadrature degrades as $n^{-r/d}$. That is the whole reason Monte Carlo dominates high-dimensional integration. The constant $\sigma$ is where the dimension hides, and attacking it is the subject of variance reduction.`,
    survey: {
      task: R`Estimate $I = \E_f h(X)$ (any integral, after rewriting)`,
      assumptions: R`IID draws from $f$; $\E\lvert h\rvert< \infty$ for consistency, $\sigma^2< \infty$ for the CLT error bar`,
      output: R`$\hat I_n = \frac1n\sum h(X_i)$ with $\widehat{\se} = s/\sqrt n$ and CLT interval`,
      strengths: 'Unbiased; dimension-free rate; trivial error estimate; works for nonsmooth integrands',
      weaknesses: R`Slow $n^{-1/2}$; constant can grow with dimension; rare events need $n\gg1/p$; fails silently if $\sigma^2=\infty$`,
      complexity: R`$O(n\cdot\text{cost}(h) )$; one more digit costs $100\times$ the samples`,
    },
    sections: [
      {
        title: 'Integrals are expectations',
        html: R`
<p>Every integral can be read as an expectation. If $f$ is a density on $\mathcal X$, then $I = \int h f = \E_f[h(X)]$. If you just have $\int_A g(x)\d x$ over a bounded set $A$, take $X$ uniform on $A$: $\int_A g = \text{vol}(A)\,\E[g(X)]$. More generally, for any density $p$ positive where $g\ne0$, $\int g = \E_p[g(X)/p(X)]$, which is the starting point of <a href="#/t/importance-sampling">importance sampling</a>.</p>
<div class="def" data-name="Crude Monte Carlo estimator">
<p>Given $X_1,\dots,X_n\iid f$, $\hat I_n = \frac1n\sum_{i=1}^n h(X_i)$. Its sample variance $s_n^2 = \frac1{n-1}\sum(h(X_i)-\hat I_n)^2$ estimates $\sigma^2 = \Var_f h(X)$, and $\widehat{\se} = s_n/\sqrt n$ is the Monte Carlo standard error.</p>
</div>
<p>Expectations of all kinds are integrals: probabilities ($h = \ind_A$), moments, posterior means, expected utilities, option prices, $p$-values of complicated tests, the risk of an estimator. In each case Monte Carlo replaces analysis with averaging.</p>`,
      },
      {
        title: 'Unbiasedness, consistency, and the CLT error bar',
        html: R`
<div class="thm" data-name="Properties of the crude estimator">
<p>Let $X_i\iid f$.</p>
<ol>
<li>If $\E_f\lvert h(X)\rvert< \infty$: $\E\hat I_n = I$ and $\hat I_n\asto I$.</li>
<li>If additionally $\sigma^2< \infty$: $\Var\hat I_n = \sigma^2/n$, $\sqrt n(\hat I_n - I)\dto\N(0,\sigma^2)$, and $s_n^2\asto\sigma^2$, so $\hat I_n\pm z_{1-\alpha/2}\,s_n/\sqrt n$ is an asymptotic $1-\alpha$ confidence interval for $I$.</li>
<li>If $a\le h\le b$: for every $\varepsilon>0$, $\P(\lvert\hat I_n - I\rvert\ge\varepsilon)\le 2\exp\!\big(-2n\varepsilon^2/(b-a)^2\big)$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) Linearity of expectation and the <a href="#/t/lln">strong law of large numbers</a>. (2) Variances of independent terms add: $\Var(\frac1n\sum h(X_i)) = n\sigma^2/n^2$. The <a href="#/t/clt">CLT</a> gives the limit law; the strong law applied to $h(X_i)$ and $h(X_i)^2$ gives $s_n^2\asto\sigma^2$, and Slutsky's theorem gives $\sqrt n(\hat I_n-I)/s_n\dto\N(0,1)$. (3) Hoeffding's inequality for bounded independent summands.</p>
</details>
<p>Three remarks carry most of the practical content.</p>
<ul>
<li><strong>The error estimate is free.</strong> The same draws give $\hat I_n$ and $\widehat{\se}$. Always report both; an estimate with no standard error is not a Monte Carlo result.</li>
<li><strong>The rate is $n^{-1/2}$.</strong> Halving the error costs four times the work; one more decimal digit costs a hundred times.</li>
<li><strong>The rate does not mention $d$.</strong> Nothing above used the dimension of $X$. Only $\sigma^2$ depends on the problem.</li>
</ul>
<div class="ex" data-name="Hit-or-miss versus crude: estimating π">
<p>Two estimators of $\pi$ from $U, U_1,U_2\iid\text{Unif}(0,1)$:</p>
<p><em>Hit-or-miss:</em> $4\,\ind\{U_1^2+U_2^2\le1\}$, a Bernoulli($\pi/4$) times 4, with variance $16\cdot\frac\pi4(1-\frac\pi4) = \pi(4-\pi)\approx2.697$.</p>
<p><em>Crude:</em> $4\sqrt{1-U^2}$, since $\int_0^1\sqrt{1-u^2}\d u = \pi/4$. Its variance is $16\,\E(1-U^2) - \pi^2 = \tfrac{32}{3}-\pi^2\approx0.797$.</p>
<p>The crude estimator needs $3.4$ times fewer draws for the same accuracy. This is no accident: the crude integrand is the conditional expectation of the hit-or-miss indicator given $U_1$, so by the law of total variance it can only have smaller variance. This is <a href="#/t/variance-reduction">conditional Monte Carlo</a>, a form of <a href="#/t/rao-blackwell">Rao–Blackwellization</a>.</p>
</div>`,
        lab: 'monte-carlo-integration-convergence',
      },
      {
        title: 'Monte Carlo versus quadrature: where the dimension goes',
        html: R`
<p>Deterministic quadrature in one dimension is spectacularly better than Monte Carlo: the composite trapezoid rule has error $O(m^{-2})$ with $m$ nodes for $C^2$ integrands, Simpson's rule $O(m^{-4})$, Gaussian rules exponentially small for analytic ones. The trouble is the extension to $d$ dimensions.</p>
<div class="thm" data-kind="Proposition" data-name="Product rules in d dimensions">
<p>A product rule built from a one-dimensional rule of order $r$ with $m$ nodes per axis uses $N = m^d$ points and, for integrands with bounded mixed derivatives of order $r$, has error $O(m^{-r}) = O(N^{-r/d})$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Apply the one-dimensional error bound one coordinate at a time: replacing the integral in coordinate $j$ by the rule commits error $O(m^{-r})$ uniformly in the other coordinates, and the rule's weights are bounded in total, so the $d$ errors add to $O(d\,m^{-r})$. Expressing $m = N^{1/d}$ gives the rate.</p>
</details>
<p>So trapezoid ($r=2$) loses to Monte Carlo's $N^{-1/2}$ once $d>4$, Simpson ($r=4$) once $d>8$. This is not a weakness of product rules alone. Bakhvalov (1959) showed that for the class of functions with bounded derivatives of order $r$ on $[0,1]^d$, <em>no</em> deterministic rule with $N$ points can guarantee worst-case error better than order $N^{-r/d}$, while randomized rules can achieve $N^{-1/2-r/d}$. Randomness is what breaks the curse of dimensionality for integration, in the precise sense of information-based complexity.</p>
<div class="warn" data-name="Dimension-free rate, dimension-dependent constant">
<p>The constant $\sigma^2$ can grow exponentially with $d$. Estimating the volume $V_d$ of the unit $d$-ball by hit-or-miss in $[-1,1]^d$ gives a Bernoulli with success probability $V_d/2^d$: about $0.0025$ for $d=10$ and $2.5\times10^{-8}$ for $d=20$. The relative standard error $\sqrt{(1-p)/(np)}$ requires $n\gg 1/p$ draws to get even one hit. Dimension-free means "no worse than $\sigma/\sqrt n$", not "easy". The second lab shows both effects: MC overtakes the product rule as $d$ grows, and its own constant grows too.</p>
</div>`,
        lab: 'monte-carlo-integration-dimension',
      },
      {
        title: 'Functions of means, quantiles, and how many draws to take',
        html: R`
<p>Many Monte Carlo targets are not plain means.</p>
<ul>
<li><strong>Ratios</strong> such as $\E[A]/\E[B]$ (conditional expectations, self-normalized estimators): use the ratio of sample means, which is consistent and has a <a href="#/t/delta-method">delta-method</a> standard error $\frac{1}{\sqrt n\,\lvert\bar B\rvert}\,\text{sd}(A_i - \hat r B_i)$, $\hat r = \bar A/\bar B$, with bias of order $1/n$.</li>
<li><strong>Quantiles</strong> $q_p = F^{-1}(p)$: the sample quantile satisfies $\sqrt n(\hat q_p - q_p)\dto\N\!\big(0, p(1-p)/f(q_p)^2\big)$; the density in the denominator makes tail quantiles expensive. A distribution-free interval comes from order statistics: $(X_{(j)}, X_{(k)})$ with $j,k$ from Binomial($n,p$) quantiles.</li>
<li><strong>Probabilities</strong> $p = \P(A)$: $\se = \sqrt{p(1-p)/n}$, so the <em>relative</em> error is $\sqrt{(1-p)/(np)}$. To estimate $p = 10^{-6}$ to 10% relative error needs about $10^8$ draws. Rare events call for <a href="#/t/importance-sampling">importance sampling</a>.</li>
</ul>
<p><strong>Choosing $n$.</strong> To achieve a target standard error $\varepsilon$, take $n\approx\sigma^2/\varepsilon^2$ with $\sigma$ from a pilot run. Sequential procedures that stop when $s_n/\sqrt n< \varepsilon$ are asymptotically valid (Chow & Robbins, 1965) but are biased in small samples because the stopping time depends on $s_n$; start with a reasonable minimum $n$.</p>
<div class="note" data-name="Reporting">
<p>A Monte Carlo estimate has a precision you choose. Report it with its standard error and only as many digits as that standard error supports. Two methods compared by simulation should be compared using the <em>same</em> random inputs where possible (common random numbers), and the difference reported with its own standard error. See <a href="#/t/simulation-studies">designing simulation studies</a>.</p>
</div>`,
      },
      {
        title: 'When the error bar lies: infinite variance',
        html: R`
<p>The CLT needs $\sigma^2< \infty$. Without it the strong law may still give consistency, but the error no longer shrinks like $n^{-1/2}$, and, worse, the <em>estimated</em> standard error looks perfectly reasonable most of the time.</p>
<div class="ex" data-name="A finite mean with infinite variance">
<p>Take $h(u) = u^{-1/2}$ on $(0,1)$: $\int_0^1 u^{-1/2}\d u = 2$, but $\int_0^1u^{-1}\d u = \infty$. The crude estimator converges to 2 almost surely, yet it is a sum of variables with tail $\P(h(U)>t) = t^{-2}$: this is the boundary case of the generalized CLT: the sum is still asymptotically normal, but only after normalizing by $\sqrt{n\log n}$ instead of $\sqrt n$, so errors decay like $\sqrt{\log n/n}$. For heavier tails, index $\alpha\in(1,2)$, errors decay like $n^{-(1-1/\alpha)}$. The sample variance, meanwhile, is dominated by the largest draw so far and jumps upward each time a new record appears; between jumps it suggests a stable, overly optimistic error bar.</p>
</div>
<div class="warn" data-name="Diagnosing heavy tails">
<p>Symptoms: running means with occasional jumps that do not diminish; $s_n$ that keeps growing with $n$; a few draws accounting for a large share of $\sum h(X_i)$. Remedies are mathematical, not computational: change variables to remove the singularity ($u = v^2$ turns $\int_0^1u^{-1/2}\d u$ into $\int_0^1 2\d v$), use importance sampling with a proposal that matches the singularity, or subtract an analytically integrable part (a control variate). The first lab includes this integrand.</p>
</div>
<p>The same phenomenon appears in <a href="#/t/importance-sampling">importance sampling</a> whenever the proposal has lighter tails than the target: the weights have infinite variance and the estimate is consistent but erratic.</p>`,
      },
    ],
    quiz: [
      { q: R`A crude MC estimate with $n = 10^4$ gives $\hat I = 0.4127$ with $s = 0.38$. Give a 95% interval and say how many draws you need to report three correct decimals with reasonable confidence.`,
        a: R`$\widehat{\se} = 0.38/100 = 0.0038$, so $0.4127\pm0.0074$, i.e. $[0.405, 0.420]$. For a half-width of $0.0005$ you need $1.96\cdot0.38/\sqrt n\le0.0005$, i.e. $n\approx(1490)^2\approx2.2\times10^6$.` },
      { q: R`Why does the composite trapezoid rule lose to Monte Carlo in dimension 6, even though it is far better in dimension 1?`,
        a: R`With $N$ total points on a product grid, each axis gets $N^{1/6}$ nodes and the error is $O(N^{-2/6}) = O(N^{-1/3})$, worse than Monte Carlo's $O(N^{-1/2})$. The trapezoid rule's per-axis accuracy is diluted because the number of nodes per axis grows only as the $d$-th root of the budget.` },
      { q: R`Is the hit-or-miss estimator ever better than the crude estimator for $\int_0^1 g(u)\d u$ with $0\le g\le1$?`,
        a: R`No. The crude integrand $g(U_1) = \E[\ind\{U_2\le g(U_1)\}\mid U_1]$ is the conditional expectation of the hit-or-miss indicator, so by the law of total variance $\Var g(U_1)\le\Var\ind\{U_2\le g(U_1)\}$, with equality only if the indicator is already a function of $U_1$ (i.e. $g$ takes only values 0 and 1).` },
      { q: R`You estimate $\E[1/X]$ for $X\sim\text{Unif}(0,1)$ by Monte Carlo. What happens?`,
        a: R`$\E[1/X] = \int_0^1 x^{-1}\d x = \infty$. The running average diverges (slowly, like $\log n$), occasionally jumping when a tiny $X$ appears. Monte Carlo cannot detect infiniteness reliably from finite samples; this is why integrability must be checked analytically.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'importance-sampling',
    domain: 'simulation',
    order: 60,
    title: 'Importance Sampling',
    subtitle: 'Sample from the wrong distribution on purpose, and correct with likelihood-ratio weights',
    level: 2,
    sources: { aos: '§24.3', other: 'Owen (2013), Ch. 9; Kong (1992); Hesterberg (1995); Chatterjee & Diaconis (2018); Vehtari et al. (2024)' },
    prereqs: ['monte-carlo-integration'],
    related: ['rejection-sampling', 'particle-filters', 'variance-reduction', 'bayesian-inference', 'entropy-kl'],
    tags: ['likelihood ratio', 'weights', 'effective sample size', 'rare events', 'self-normalized', 'exponential tilting'],
    summary: R`Importance sampling rewrites $\E_f h(X) = \E_g[h(X)w(X)]$ with $w = f/g$, so that draws from a proposal $g$ of your choosing can estimate an expectation under $f$. A good $g$ concentrates draws where $\lvert h\rvert f$ is large; the ideal $g\propto\lvert h\rvert f$ attains zero variance when $h\ge0$. A bad $g$ is catastrophic in a quiet way: if its tails are lighter than $f$'s, the weights have infinite variance and the estimate converges at an unknown, erratic rate while looking stable. When $f$ is known only up to a constant, normalizing the weights gives a consistent but biased self-normalized estimator, and the weights' spread, summarized by the effective sample size, is the main diagnostic. Importance sampling is the workhorse of rare-event simulation and the foundation of sequential Monte Carlo.`,
    survey: {
      task: R`Estimate $\E_f h(X)$ using draws from a different density $g$`,
      assumptions: R`$g>0$ wherever $hf\ne0$; for a CLT, $\E_g[h^2w^2]< \infty$; $f$ evaluable (up to a constant for SNIS)`,
      output: R`$\hat I = \frac1n\sum h(X_i)w(X_i)$ or the self-normalized $\sum w_ih_i/\sum w_i$; ESS diagnostic`,
      strengths: 'Variance can drop by orders of magnitude; reuses one sample for many targets; no bound $M$ needed',
      weaknesses: 'Light-tailed proposals give infinite variance; collapses in high dimension; ESS cannot see unvisited regions',
      complexity: R`$O(n)$; needed $n$ grows roughly like $e^{\KL(f\Vert g)}$`,
    },
    sections: [
      {
        title: 'The identity and the estimator',
        html: R`
<p>Let $g$ be a density with $g(x)>0$ whenever $h(x)f(x)\neq0$. Then
$$I = \int h(x)f(x)\d x = \int h(x)\frac{f(x)}{g(x)}\,g(x)\d x = \E_g\big[h(X)w(X)\big],\qquad w(x) = \frac{f(x)}{g(x)}.$$
The function $w$ is the <em>likelihood ratio</em> or importance weight.</p>
<div class="thm" data-name="Importance sampling estimator">
<p>With $X_1,\dots,X_n\iid g$, the estimator $\hat I_g = \frac1n\sum_{i=1}^n h(X_i)w(X_i)$ is unbiased for $I$, is strongly consistent whenever $\E_f\lvert h\rvert< \infty$, and has variance
$$\Var\hat I_g = \frac1n\Big(\int\frac{h(x)^2f(x)^2}{g(x)}\d x - I^2\Big) = \frac{\sigma_g^2}{n}.$$
If $\sigma_g^2< \infty$, $\sqrt n(\hat I_g - I)\dto\N(0,\sigma_g^2)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Unbiasedness is the identity above; consistency and the CLT are those of <a href="#/t/monte-carlo-integration">crude Monte Carlo</a> applied to the IID variables $h(X_i)w(X_i)$, whose mean is $I$ and whose absolute mean is $\E_f\lvert h\rvert$. The second moment is $\E_g[h^2w^2] = \int h^2f^2/g$.</p>
</details>
<p>The support condition is essential: if $g = 0$ on a set where $hf\ne0$, the estimator is unbiased for the wrong integral (the part of $I$ outside that set), and no amount of sampling reveals it.</p>`,
      },
      {
        title: 'The optimal proposal',
        html: R`
<div class="thm" data-name="Zero-variance proposal (Kahn & Marshall, 1953)">
<p>Among all densities $g$, $\sigma_g^2$ is minimized by $g^\star(x) = \lvert h(x)\rvert f(x)\big/\int\lvert h\rvert f$, with minimum value $\big(\int\lvert h\rvert f\big)^2 - I^2$. If $h\ge0$, then $\sigma_{g^\star}^2 = 0$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By Cauchy–Schwarz (or Jensen) under $g$,
$$\E_g\big[h^2w^2\big]\ \ge\ \big(\E_g\lvert h\rvert w\big)^2 = \Big(\int\lvert h\rvert f\Big)^2,$$
for every admissible $g$, with equality iff $\lvert h\rvert w$ is $g$-a.s. constant, i.e. $g\propto\lvert h\rvert f$. Substituting $g^\star$, $\lvert h\rvert w = \int\lvert h\rvert f$ identically. When $h\ge0$ the bound $(\int hf)^2 = I^2$ makes the variance zero: every draw returns exactly $I$.</p>
</details>
<p>The optimum is useless as an algorithm, since normalizing $g^\star$ requires the very integral we want. It is invaluable as a <strong>design principle</strong>: choose $g$ roughly proportional to $\lvert h\rvert f$. For a tail probability $\P_f(X>c)$, that means a proposal concentrated just beyond $c$, not around the mode of $f$. Good importance sampling is about the <em>product</em> $\lvert h\rvert f$, not about $f$ alone. Adaptive schemes (the cross-entropy method of Rubinstein, 1997; population Monte Carlo) iteratively fit a parametric $g_\theta$ to weighted samples to approximate $g^\star$.</p>
<div class="ex" data-name="Gaussian tail by mean shift">
<p>Estimate $p = \P(Z>c)$ for $Z\sim\N(0,1)$ with $g = \N(\theta,1)$. The weight is $w(x) = \phi(x)/\phi(x-\theta) = e^{-\theta x + \theta^2/2}$ (exponential tilting). Completing the square, $\phi(x)^2/\phi(x-\theta) = e^{\theta^2}\phi(x+\theta)$, so
$$\E_g\big[\ind\{X>c\}w^2\big] = e^{\theta^2}\,\bar\Phi(c+\theta),\qquad \sigma_g^2 = e^{\theta^2}\bar\Phi(c+\theta) - p^2.$$
At $\theta = c$ and large $c$, Mills' ratio $\bar\Phi(t)\approx\phi(t)/t$ gives relative variance $\sigma_g^2/p^2\approx c\sqrt{2\pi}/2-1$: it grows only linearly in $c$, whereas crude Monte Carlo's relative variance $(1-p)/p$ grows like $e^{c^2/2}$. For $c = 5$ ($p\approx2.9\times10^{-7}$), crude MC needs about $3.5\times10^{8}$ draws for 10% relative error; the shifted proposal needs about $600$.</p>
</div>`,
        lab: 'importance-sampling-tail',
      },
      {
        title: 'Finite variance and the tails of the proposal',
        html: R`
<p>Unbiasedness holds for any admissible $g$; finite variance does not. The integral $\int h^2f^2/g$ is finite only if $g$ does not decay much faster than $hf$ anywhere.</p>
<div class="thm" data-kind="Proposition" data-name="A Gaussian example">
<p>Let $f = \N(0,1)$, $g = \N(0,\sigma^2)$, $h\equiv1$. Then $\E_g w^2 = \sigma^2/\sqrt{2\sigma^2-1}$ if $\sigma^2>1/2$, and $\E_gw^2 = \infty$ if $\sigma^2\le1/2$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$f(x)^2/g(x) = \frac{\sigma}{\sqrt{2\pi}}\exp\!\big(-x^2(1-\tfrac{1}{2\sigma^2})\big)$. This is integrable iff $1-\frac1{2\sigma^2}>0$, i.e. $\sigma^2>1/2$, in which case the Gaussian integral gives $\frac{\sigma}{\sqrt{2\pi}}\sqrt{\pi/(1-\frac1{2\sigma^2})} = \sigma^2/\sqrt{2\sigma^2-1}$.</p>
</details>
<p>Note the asymmetry: a proposal that is too wide ($\sigma = 3$: $\E w^2 = 9/\sqrt{17}\approx2.18$) costs a modest constant factor; one that is too narrow by a factor $\sqrt2$ is infinitely bad. With $\sigma^2 < 1/2$ the estimator still converges almost surely, but most of the time it underestimates, and occasionally a draw far in $g$'s tail carries an enormous weight. A running estimate shows a sawtooth that never settles into the $n^{-1/2}$ envelope.</p>
<div class="note" data-name="Safe proposals">
<p>Use proposals with heavier tails than the target: a Student-$t$ with few degrees of freedom in place of a normal, or a <strong>defensive mixture</strong> $g = \alpha f + (1-\alpha)g_0$ (Hesterberg, 1995), which guarantees $w\le1/\alpha$ and hence variance at most $1/\alpha$ times that of crude Monte Carlo, whatever $g_0$ does. The general rule: $\sup_x f(x)/g(x)< \infty$ is sufficient (for bounded $h$), and that is exactly the <a href="#/t/rejection-sampling">rejection-sampling</a> envelope condition; when it holds you could also run accept–reject, and IS will typically be more efficient.</p>
</div>`,
        lab: 'importance-sampling-weights',
      },
      {
        title: 'Self-normalized importance sampling',
        html: R`
<p>In Bayesian computation $f = \tilde f/Z$ with $Z$ unknown, so $w$ is only available up to a constant: $\tilde w = \tilde f/g = Zw$. Normalize the weights instead:
$$\tilde I_n = \frac{\sum_{i=1}^n\tilde w(X_i)h(X_i)}{\sum_{i=1}^n\tilde w(X_i)} = \sum_i W_i h(X_i),\qquad W_i = \frac{\tilde w_i}{\sum_j\tilde w_j}.$$
The unknown $Z$ cancels. As a byproduct, $\frac1n\sum\tilde w_i$ is an unbiased estimator of $Z$, the marginal likelihood.</p>
<div class="thm" data-name="Self-normalized IS">
<p>If $g>0$ wherever $f>0$ and $\E_f\lvert h\rvert< \infty$, then $\tilde I_n\asto I$. If $\E_g[w^2(1+h^2)]< \infty$,
$$\sqrt n(\tilde I_n - I)\dto\N\big(0,\ \E_g[w^2(h-I)^2]\big),$$
and the bias is $O(1/n)$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$\tilde I_n = A_n/B_n$ with $A_n = \frac1n\sum w_ih_i\asto I$ and $B_n = \frac1n\sum w_i\asto1$ (normalized $w$; the constant $Z$ cancels in the ratio). Consistency follows. For the CLT, write $\tilde I_n - I = \frac{1}{B_n}\cdot\frac1n\sum w_i(h_i - I)$; the summands have mean $\E_f[h-I] = 0$ and variance $\E_g[w^2(h-I)^2]$, and $B_n\to1$, so Slutsky's theorem gives the limit. A second-order expansion of $A/B$ around $(I,1)$ gives bias $\frac1n\big(I\,\Var_g w - \Cov_g(wh, w)\big) + o(1/n) = -\frac1n\E_g[w^2(h-I)] + o(1/n)$.</p>
</details>
<p>The asymptotic variance $\E_g[w^2(h-I)^2]$ differs from the unnormalized $\E_g[w^2h^2] - I^2$: self-normalization can be better (it removes the variance from fluctuations of $\sum w_i$ when $h$ is nearly constant) or worse. Its optimal proposal is $g\propto\lvert h-I\rvert f$ (Hesterberg), which never gives zero variance: half its mass must sit where $h>I$, half where $h< I$.</p>`,
      },
      {
        title: 'Effective sample size and diagnostics',
        html: R`
<div class="def" data-name="Kish effective sample size">
<p>$\text{ESS} = \dfrac{\big(\sum_i w_i\big)^2}{\sum_i w_i^2} = \dfrac{1}{\sum_i W_i^2}\in[1,n].$ It equals $n$ when all weights are equal and $1$ when one weight dominates.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="ESS and the chi-square divergence (Kong, 1992)">
<p>With normalized $w$ ($\E_gw = 1$), $\text{ESS}/n\asto 1/\E_g[w^2] = 1/\big(1+\Var_g w\big) = 1/\big(1+\chi^2(f\Vert g)\big)$. Moreover, to first order and when $h$ is roughly uncorrelated with $w^2$, $\Var\tilde I_n\approx\Var_f h(X)/\text{ESS}$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$\text{ESS}/n = (\frac1n\sum w_i)^2/(\frac1n\sum w_i^2)\to(\E w)^2/\E w^2$ by the strong law. For the second claim, the SNIS asymptotic variance is $\E_g[w^2(h-I)^2]/n$; under the approximation $\E_g[w^2(h-I)^2]\approx\E_g[w^2]\,\E_f[(h-I)^2]$ it equals $\Var_f(h)\cdot\E_gw^2/n\approx\Var_f(h)/\text{ESS}$. The approximation ignores the dependence between $h$ and $w$, so ESS is a heuristic, not a theorem about any particular $h$.</p>
</details>
<div class="warn" data-name="What ESS cannot see">
<p>ESS is computed from the weights you drew. If $g$ entirely misses a mode of $f$ (or a region where $h$ matters), all the drawn weights can be similar and ESS can look excellent while the estimate is badly wrong. ESS detects weight degeneracy among visited points; it says nothing about unvisited regions. Combine it with knowledge of where $f$ puts mass and with a tail diagnostic for the weights.</p>
</div>
<p><strong>Pareto-smoothed IS.</strong> Vehtari, Simpson, Gelman, Yao and Gabry fit a generalized Pareto distribution to the largest weights (see <a href="#/t/peaks-over-threshold">peaks over threshold</a>). The estimated shape $\hat k$ governs the number of finite moments of $w$ (moments of order below $1/k$ exist): $k< 1/2$ means finite variance; their recommended threshold is $\hat k< 0.7$ (or a sample-size-dependent version) for usable estimates. Replacing the largest weights by GPD quantiles also stabilizes the estimate. Chatterjee and Diaconis (2018) give a complementary rule of thumb: under concentration conditions, a sample size of about $\exp(\KL(f\Vert g))$ is necessary and sufficient for accurate IS, which makes clear why IS degrades exponentially as dimension grows: KL divergences between product measures add across coordinates.</p>`,
      },
      {
        title: 'Where importance sampling is used',
        html: R`
<ul>
<li><strong>Rare events</strong> in reliability, finance and queueing: exponential tilting toward the dominating point (Siegmund, 1976, for random-walk boundary crossing), with asymptotic efficiency guarantees.</li>
<li><strong>Bayesian computation</strong>: posterior expectations from a Laplace or variational approximation used as proposal; marginal likelihood estimation; leave-one-out cross-validation by reweighting posterior draws by $1/p(y_i\mid\theta)$ (PSIS-LOO).</li>
<li><strong>Off-policy evaluation</strong> in reinforcement learning and causal inference: inverse propensity weighting is importance sampling with $w = \pi_{\text{target}}/\pi_{\text{behaviour}}$ (see <a href="#/t/causal-inference">causal inference</a>), with exactly the same variance pathologies when propensities are small.</li>
<li><strong>Sequential Monte Carlo</strong>: <a href="#/t/particle-filters">particle filters</a> apply importance sampling one time step at a time and resample to fight weight degeneracy.</li>
<li><strong>Reusing simulations</strong>: one sample from $g$ estimates $\E_{f_\theta}h$ for a whole family $f_\theta$ by reweighting, useful for sensitivity analysis and likelihood surfaces (but each $\theta$ far from the sampling point inherits the degeneracy problem).</li>
</ul>
<div class="warn" data-name="Dimension">
<p>For product targets $f = \prod_j f_1(x_j)$ and proposals $g = \prod_j g_1(x_j)$, $\E_g w^2 = (\E_{g_1}w_1^2)^d$ grows exponentially in $d$ unless $g_1 = f_1$. Even a tiny per-coordinate mismatch is fatal in hundreds of dimensions. Importance sampling in high dimension works only when $g$ is tailored to $f$ globally (e.g. a well-fitted Gaussian approximation for a nearly Gaussian posterior), or when the problem is broken into a sequence of low-dimensional steps, as SMC does.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`For $h\ge0$, the zero-variance proposal is $g^\star = hf/I$. Why can't we just use it?`,
        a: R`Evaluating $g^\star$ (to sample from it or compute weights) requires $I = \int hf$, the quantity being estimated. Its value is as a target to approximate: proposals close to $hf$ in shape give small variance.` },
      { q: R`To estimate $\E_f[X^2]$ for $f = \N(0,1)$, you use $g = \N(0, 0.4)$. Is the estimator unbiased? Is its variance finite?`,
        a: R`It is unbiased (support condition holds). The variance requires $\int x^4f^2/g< \infty$; $f^2/g\propto\exp(-x^2(1 - 1/(2\cdot0.4))) = \exp(+x^2/4)$, which is not integrable. Infinite variance: the estimate converges but erratically, with no valid CLT error bar.` },
      { q: R`Explain why self-normalized IS is biased but plain IS is not.`,
        a: R`Plain IS is a sample mean of unbiased terms. Self-normalized IS is a ratio of two sample means, $\bar A/\bar B$; the expectation of a ratio is not the ratio of expectations, giving bias of order $1/n$ (from the covariance of $\bar A$ and $\bar B$ and the variance of $\bar B$). It is still consistent and is necessary when $f$ is known only up to a constant.` },
      { q: R`You get ESS $= 950$ from $n = 1000$ importance draws. Is your estimate reliable?`,
        a: R`Not necessarily. High ESS shows the weights of the draws you obtained are nearly equal, but if $g$ never visits a region where $f$ (or $hf$) has substantial mass, those regions contribute nothing and ESS cannot detect it. Check the proposal's coverage of $f$'s support and tails (e.g. heavier tails than $f$) and the $\hat k$ tail diagnostic.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'variance-reduction',
    domain: 'simulation',
    order: 70,
    title: 'Variance Reduction: Antithetic, Control Variates & Stratification',
    subtitle: 'Same expectation, smaller variance: using what you know to shrink the Monte Carlo error',
    level: 2,
    sources: { other: 'Glasserman, Monte Carlo Methods in Financial Engineering (2004), Ch. 4; Owen (2013), Ch. 8; Ross, Simulation (2013), Ch. 9; McKay, Beckman & Conover (1979)' },
    prereqs: ['monte-carlo-integration'],
    related: ['importance-sampling', 'quasi-monte-carlo', 'rao-blackwell', 'simple-linear-regression', 'conditional-expectation'],
    tags: ['antithetic variates', 'control variates', 'stratified sampling', 'Neyman allocation', 'Latin hypercube', 'conditional Monte Carlo', 'common random numbers'],
    summary: R`The Monte Carlo error is $\sigma/\sqrt n$, and while $n$ is bought with computer time, $\sigma$ can often be reduced with thought. Antithetic variates pair each draw with a negatively correlated mirror image and provably help when the integrand is monotone. Control variates subtract a correlated quantity with known mean, cutting the variance by the factor $1-\rho^2$ at the optimal coefficient, which is just a regression slope. Stratification forces the sample to cover the domain in fixed proportions, removing the between-strata component of variance, and Neyman allocation spends more draws where the integrand varies most. Conditional Monte Carlo integrates part of the randomness analytically, a Rao–Blackwell step. All four keep the estimator unbiased (or nearly so) and leave the $n^{-1/2}$ rate unchanged; what changes is the constant.`,
    survey: {
      task: R`Reduce $\Var\hat I$ at fixed cost while keeping (near) unbiasedness`,
      assumptions: R`Antithetic: monotone $h$. Control: known $\E C$ and correlation with $h$. Stratified: sampler within strata. Conditional: tractable $\E[h\mid Y]$`,
      output: 'An estimator with the same mean and variance reduced by a computable factor',
      strengths: R`Often 10–1000$\times$ efficiency gains; methods combine; the gain is estimable from the same run`,
      weaknesses: 'Problem-specific; antithetic can double the variance for symmetric $h$; estimated coefficients add $O(1/n)$ bias',
      complexity: R`Same order as crude MC; control variates add an $O(nk^2)$ regression for $k$ controls`,
    },
    sections: [
      {
        title: 'Efficiency, not just variance',
        html: R`
<p>Two unbiased estimators should be compared by the variance they achieve per unit of computing time. If estimator $A$ has per-sample variance $\sigma_A^2$ and costs $c_A$ per sample, its error at budget $T$ is $\sigma_A\sqrt{c_A/T}$. The <strong>relative efficiency</strong> of $A$ over $B$ is
$$\text{eff}(A{:}B) = \frac{\sigma_B^2\,c_B}{\sigma_A^2\,c_A}.$$
A control variate that halves the variance but triples the cost is a loss. Every method below can be assessed from the same simulation run that uses it: estimate both variances and both costs, and report the ratio.</p>
<div class="note">
<p>All variance reduction techniques exploit <em>knowledge</em>: of symmetry (antithetic), of a related quantity's exact mean (control), of the structure of the domain (stratification), or of a partial analytic solution (conditioning). Monte Carlo is the method of last resort for the part of the problem you cannot do analytically, and the less you leave to it, the better it performs.</p>
</div>`,
      },
      {
        title: 'Antithetic variates',
        html: R`
<p>With $I = \E h(U)$, $U\sim\text{Unif}(0,1)^d$, note that $\mathbf 1 - U$ has the same law as $U$. The antithetic estimator averages over $n/2$ pairs:
$$\hat I_{\text{anti}} = \frac{1}{n}\sum_{i=1}^{n/2}\big[h(U_i) + h(\mathbf 1-U_i)\big].$$
It is unbiased, and with $\rho = \Corr(h(U), h(\mathbf1-U))$,
$$\Var\hat I_{\text{anti}} = \frac{\sigma^2}{n}(1+\rho),$$
compared with $\sigma^2/n$ for $n$ independent draws. It helps iff $\rho< 0$. For normal inputs the analogous pairing is $(Z,-Z)$.</p>
<div class="thm" data-name="Monotone integrands give negative correlation">
<p>If $h$ is monotone (nondecreasing or nonincreasing) in each coordinate separately, then $\Cov\big(h(U), h(\mathbf1-U)\big)\le0$, so the antithetic estimator is never worse than crude Monte Carlo with the same number of evaluations.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>One dimension.</em> Let $a$ be nondecreasing and $b$ nonincreasing, and $U,U'$ IID. Then $(a(U)-a(U'))(b(U)-b(U'))\le0$ pointwise, since the two factors always have opposite signs. Taking expectations, $2\Cov(a(U),b(U))\le0$. Apply with $a = h$ and $b(u) = h(1-u)$ when $h$ is nondecreasing (and to $-h$ otherwise).</p>
<p><em>$d$ dimensions.</em> Flipping $U_j\to1-U_j$ in coordinates where $h$ is nonincreasing does not change the law, so assume $h$ nondecreasing in every coordinate; then $b(u) = -h(\mathbf1-u)$ is also nondecreasing in every coordinate. Harris's inequality states that two coordinatewise nondecreasing functions of independent variables are nonnegatively correlated; it is proved by induction on $d$: conditional on $U_d$, the induction hypothesis gives $\E[ab\mid U_d]\ge\E[a\mid U_d]\E[b\mid U_d]$, and both conditional means are nondecreasing functions of $U_d$, so the one-dimensional case finishes the step. Hence $\Cov(h(U), -h(\mathbf 1-U))\ge0$.</p>
</details>
<div class="ex" data-name="An exponential integral">
<p>For $I = \int_0^1e^u\d u = e-1$: $\sigma^2 = \Var e^U = \frac{e^2-1}{2}-(e-1)^2\approx0.2420$ and $\Cov(e^U,e^{1-U}) = e-(e-1)^2\approx-0.2342$, so $\rho\approx-0.968$. The antithetic estimator's variance factor is $1+\rho\approx0.032$: about 31 times more efficient than crude MC at the same number of function evaluations.</p>
</div>
<div class="warn" data-name="Symmetric integrands">
<p>If $h(u) = h(1-u)$, then $\rho = 1$ and antithetic sampling doubles the variance: each pair contains the same value twice. For $h(u) = (u-\tfrac12)^2$ that is exactly what happens. Antithetic variates exploit odd (monotone) structure; even (symmetric) structure works against them.</p>
</div>`,
      },
      {
        title: 'Control variates',
        html: R`
<p>Suppose that alongside $Y = h(X)$ we can compute $C = c(X)$ whose mean $\mu_C$ is known exactly. For any $\beta$,
$$\hat I_\beta = \frac1n\sum_{i=1}^n\big[Y_i - \beta(C_i - \mu_C)\big]$$
is unbiased, because the added term has mean zero.</p>
<div class="thm" data-name="Optimal control coefficient">
<p>$\Var\hat I_\beta = \frac1n\big(\sigma_Y^2 - 2\beta\sigma_{YC} + \beta^2\sigma_C^2\big)$ is minimized at $\beta^\star = \Cov(Y,C)/\Var(C)$, where it equals $\frac{\sigma_Y^2}{n}(1-\rho_{YC}^2)$. With $k$ controls $\mathbf C$, $\boldsymbol\beta^\star = \bSigma_{CC}^{-1}\bSigma_{CY}$ and the variance factor is $1-R^2$, where $R^2$ is the population squared multiple correlation of $Y$ on $\mathbf C$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The variance is a convex quadratic in $\beta$; setting the derivative $-2\sigma_{YC}+2\beta\sigma_C^2$ to zero gives $\beta^\star$. Substituting, $\sigma_Y^2 - \sigma_{YC}^2/\sigma_C^2 = \sigma_Y^2(1-\rho^2)$. The vector case is identical with matrices: $\sigma_Y^2 - \bSigma_{YC}\bSigma_{CC}^{-1}\bSigma_{CY} = \sigma_Y^2(1-R^2)$.</p>
</details>
<p>$\beta^\star$ is the population slope of the <a href="#/t/simple-linear-regression">regression</a> of $Y$ on $C$, and the residual variance is what remains. In practice $\beta^\star$ is estimated by the least-squares slope from the same draws. Then $\hat I_{\hat\beta} = \bar Y - \hat\beta(\bar C - \mu_C)$, which is exactly the fitted value of the least-squares line at $C = \mu_C$. Estimating $\beta$ introduces bias of order $1/n$ and leaves the asymptotic variance unchanged; the regression's residual standard error gives the standard error directly. (Splitting the sample, estimating $\beta$ on a pilot, removes the bias if it matters.)</p>
<div class="ex" data-name="Same integral, control U">
<p>For $Y = e^U$ with control $C = U$ ($\mu_C = 1/2$, $\sigma_C^2 = 1/12$), $\Cov(e^U,U) = 1-\frac{e-1}{2}\approx0.1409$ and $\rho^2\approx0.984$: the variance shrinks by a factor of about 61. Here the control works better than antithetics because $e^u$ is very nearly linear on $[0,1]$, and the linear part is exactly what a control $U$ removes.</p>
</div>
<p>Good controls are simplified versions of the problem with known answers: the geometric-average Asian option as a control for the arithmetic-average one (Kemna & Vorst, 1990), a linearized model for a nonlinear one, the first terms of a Taylor expansion, or the input variables themselves (whose means are known by construction). The more of $Y$'s variability the controls explain, the bigger the gain.</p>`,
        lab: 'variance-reduction-compare',
      },
      {
        title: 'Stratified sampling and Neyman allocation',
        html: R`
<p>Partition the domain into strata $A_1,\dots,A_K$ with known probabilities $p_k = \P(X\in A_k)$, and suppose we can sample from $f$ restricted to each stratum. Let $\mu_k$, $\sigma_k^2$ be the conditional mean and variance of $h(X)$ given $X\in A_k$. Take $n_k$ draws in stratum $k$ ($\sum n_k = n$) and form
$$\hat I_{\text{strat}} = \sum_{k=1}^K p_k\,\bar Y_k,\qquad \Var\hat I_{\text{strat}} = \sum_{k=1}^K\frac{p_k^2\sigma_k^2}{n_k}.$$</p>
<div class="thm" data-name="Proportional and Neyman allocation">
<p>(a) <em>Proportional</em> allocation $n_k = np_k$ gives $\Var = \frac1n\sum_kp_k\sigma_k^2\le\frac{\sigma^2}{n}$, with deficit $\frac1n\sum_kp_k(\mu_k-I)^2$.<br>
(b) <em>Neyman</em> allocation $n_k = n\,p_k\sigma_k\big/\sum_jp_j\sigma_j$ minimizes the variance over all allocations, giving $\Var = \frac1n\big(\sum_kp_k\sigma_k\big)^2$, which improves on proportional allocation by $\frac1n\sum_kp_k(\sigma_k-\bar\sigma)^2$, $\bar\sigma = \sum_kp_k\sigma_k$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(a) Substitute $n_k = np_k$. The law of total variance gives $\sigma^2 = \sum_kp_k\sigma_k^2 + \sum_kp_k(\mu_k-I)^2$: stratification removes the between-strata term.<br>
(b) By Cauchy–Schwarz,
$$\Big(\sum_kp_k\sigma_k\Big)^2 = \Big(\sum_k\frac{p_k\sigma_k}{\sqrt{n_k}}\sqrt{n_k}\Big)^2\le\Big(\sum_k\frac{p_k^2\sigma_k^2}{n_k}\Big)\Big(\sum_kn_k\Big),$$
so $\Var\ge(\sum p_k\sigma_k)^2/n$ for every allocation, with equality iff $\sqrt{n_k}\propto p_k\sigma_k/\sqrt{n_k}$, i.e. $n_k\propto p_k\sigma_k$. The gap to proportional allocation is $\frac1n[\sum p_k\sigma_k^2 - (\sum p_k\sigma_k)^2] = \frac1n\sum p_k(\sigma_k-\bar\sigma)^2$.</p>
</details>
<p>So proportional stratification can never hurt, and Neyman allocation helps further when the strata variances differ. Neyman needs the $\sigma_k$, which are estimated from a pilot run; a poor pilot can make "optimal" allocation worse than proportional, so a common compromise is to cap the ratio $n_k/(np_k)$. With costs $c_k$ per draw, the optimum becomes $n_k\propto p_k\sigma_k/\sqrt{c_k}$. In one dimension with $K = n$ equal strata and one draw each, the variance becomes $O(n^{-3})$ for smooth $h$: stratification then behaves like a randomized quadrature rule.</p>
<div class="def" data-name="Latin hypercube sampling (McKay, Beckman & Conover, 1979)">
<p>In $d$ dimensions, full stratification into $K^d$ cells is impossible. LHS stratifies each coordinate's margin into $n$ equal intervals and uses each interval exactly once per coordinate, pairing them by independent random permutations. It removes the variance of the additive (main-effect) part of $h$ and, by a result of Owen (1997), is never much worse than IID sampling: the variance with $n$ points is at most that of crude Monte Carlo with $n-1$ points.</p>
</div>`,
        lab: 'variance-reduction-neyman',
      },
      {
        title: 'Conditional Monte Carlo and common random numbers',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Conditioning reduces variance">
<p>If $Y = h(X)$ and $\psi(Z) = \E[Y\mid Z]$ can be computed for some random element $Z$, then $\E\psi(Z) = I$ and $\Var\psi(Z) = \Var Y - \E\Var(Y\mid Z)\le\Var Y$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Iterated expectations and the law of total variance; this is the <a href="#/t/rao-blackwell">Rao–Blackwell</a> argument with Monte Carlo draws in place of data.</p>
</details>
<div class="ex" data-name="A sum exceeding a threshold">
<p>To estimate $\P(X+Y>c)$ with independent $X\sim F$, $Y\sim G$, the crude estimator averages $\ind\{X_i+Y_i>c\}$. Conditioning on $Y$ gives $\bar F(c-Y)$ instead: average $\bar F(c-Y_i)$ over draws of $Y$ only. It is smooth in $Y$, never 0 or 1, and for rare events it can be orders of magnitude better. The <a href="#/t/monte-carlo-integration">hit-or-miss versus crude</a> comparison is the same idea.</p>
</div>
<p><strong>Common random numbers.</strong> To estimate a <em>difference</em> $\E h_1(X) - \E h_2(X)$, for example the performance of two policies or two estimators, use the same inputs for both: $\Var(h_1(X)-h_2(X)) = \sigma_1^2+\sigma_2^2-2\Cov(h_1(X),h_2(X))$, which is small when the two systems respond similarly to the same randomness. By the same Harris inequality as above, if $h_1$ and $h_2$ are both nondecreasing (in each input), the covariance is nonnegative and CRN cannot hurt. Synchronization matters: the $i$-th uniform must drive the same random element in both systems, which is why <a href="#/t/inverse-transform">inversion</a> (one uniform per variate) is preferred over rejection there. CRN is the default in <a href="#/t/simulation-studies">simulation studies</a> comparing methods on the same datasets.</p>
<div class="note" data-name="Combining methods">
<p>Methods stack. Stratify the first coordinate and use a control variate within strata; use antithetic pairs inside a stratified design; condition analytically on part of the randomness and use QMC for the rest. <a href="#/t/importance-sampling">Importance sampling</a> is also a variance reduction method, the only one in this family that can make variance arbitrarily small, and the only one that can make it infinite.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`For $h(u) = u^2$ on $(0,1)$, will antithetic variates help? For $h(u) = (u-0.5)^2$?`,
        a: R`$u^2$ is monotone on $(0,1)$, so the correlation of $U^2$ and $(1-U)^2$ is negative (it is $-7/8$), and antithetics help. $(u-0.5)^2$ is symmetric about $1/2$, so $h(U) = h(1-U)$ exactly, $\rho = 1$, and antithetics double the variance relative to two independent draws.` },
      { q: R`A control variate has correlation $0.9$ with the integrand. By what factor does the optimal control variate estimator reduce the variance, and what if the correlation is $-0.9$?`,
        a: R`The factor is $1-\rho^2 = 0.19$ (about $5.3\times$ more efficient) in both cases; the sign of $\rho$ only changes the sign of $\beta^\star$.` },
      { q: R`Three strata have $p = (0.5, 0.3, 0.2)$ and $\sigma = (1, 1, 10)$. Compare per-draw variances of proportional and Neyman allocation.`,
        a: R`Proportional: $\sum p_k\sigma_k^2 = 0.5+0.3+20 = 20.8$. Neyman: $(\sum p_k\sigma_k)^2 = (0.5+0.3+2)^2 = 7.84$. Neyman puts $n_k\propto(0.5, 0.3, 2)$, i.e. about 71% of the draws in the small, highly variable third stratum, and is about $2.65\times$ more efficient.` },
      { q: R`Why is conditional Monte Carlo guaranteed not to increase variance, while antithetic sampling is not?`,
        a: R`Conditioning is an orthogonal projection in $L^2$: $\Var(Y) = \Var(\E[Y\mid Z]) + \E\Var(Y\mid Z)$, and the second term is nonnegative. Antithetic sampling introduces a correlation whose sign depends on $h$; only for monotone $h$ is it guaranteed nonpositive.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'quasi-monte-carlo',
    domain: 'simulation',
    order: 80,
    title: 'Quasi-Monte Carlo & Low-Discrepancy Sequences',
    subtitle: 'Replacing random points by deliberately even ones, and randomizing them back for error bars',
    level: 3,
    sources: { other: 'Niederreiter, Random Number Generation and Quasi-Monte Carlo Methods (1992); Dick & Pillichshammer, Digital Nets and Sequences (2010); Owen (1997); L’Ecuyer & Lemieux (2002)' },
    prereqs: ['monte-carlo-integration'],
    related: ['variance-reduction', 'pseudo-random', 'inverse-transform', 'empirical-cdf'],
    tags: ['discrepancy', 'Koksma–Hlawka', 'Halton', 'Sobol', 'digital nets', 'randomized QMC', 'scrambling'],
    summary: R`Random points clump and leave holes, and those fluctuations are exactly the $n^{-1/2}$ Monte Carlo error. Quasi-Monte Carlo replaces them by deterministic point sets that fill $[0,1)^d$ as evenly as possible, measured by the discrepancy: the worst gap between the fraction of points in an anchored box and its volume. The Koksma–Hlawka inequality bounds the integration error by the integrand's variation times the discrepancy, and constructions of Halton, Sobol, Faure and Niederreiter achieve discrepancy $O((\log n)^d/n)$, asymptotically far better than $n^{-1/2}$. The asymptotics are misleading in high dimension, yet QMC often works there anyway because practical integrands have low effective dimension. Randomized QMC (random shifts, Owen scrambling) restores unbiasedness and gives honest error bars while keeping, and for smooth integrands improving, the faster rate.`,
    survey: {
      task: R`Integrate $f$ over $[0,1)^d$ (or any domain, via a monotone transform) faster than $n^{-1/2}$`,
      assumptions: R`$f$ of bounded Hardy–Krause variation for the deterministic bound; smoothness and low effective dimension for practical gains`,
      output: R`$\frac1n\sum f(\mathbf x_i)$; with RQMC, $R$ independent replicates and a standard error`,
      strengths: R`Error near $O(n^{-1})$ for smooth, low-effective-dimension integrands; scrambled nets reach $O(n^{-3/2})$ RMSE`,
      weaknesses: 'No error estimate without randomization; discontinuities and high effective dimension erase the gain; needs inversion, not rejection',
      complexity: R`$O(nd)$ point generation; same function evaluation count as MC`,
    },
    sections: [
      {
        title: 'Why random points are not the most uniform points',
        html: R`
<p>In <a href="#/t/monte-carlo-integration">Monte Carlo integration</a> the error $\frac1n\sum f(\mathbf x_i) - \int f$ is the discrepancy between the empirical measure of the points and Lebesgue measure, tested against $f$. Random points make that discrepancy of order $n^{-1/2}$ because independent points ignore each other: some regions get too many, others too few. If we only need the empirical measure to be close to uniform, and not the points to be independent, we can do much better by placing points deliberately.</p>
<p>A grid is the naive answer, but it is the product rule of <a href="#/t/monte-carlo-integration">quadrature</a> with its $N^{-r/d}$ curse: a grid with $m$ points per axis projects onto only $m$ distinct values in each coordinate, wasting $m^{d-1}$ evaluations on every one of them. Low-discrepancy sequences are designed so that <em>every</em> projection, onto every coordinate and every subset of coordinates, is itself well spread, and so that $n$ can be increased one point at a time.</p>`,
      },
      {
        title: 'Discrepancy and the Koksma–Hlawka inequality',
        html: R`
<div class="def" data-name="Star discrepancy">
<p>For a point set $P = \{\mathbf x_1,\dots,\mathbf x_n\}\subset[0,1)^d$,
$$D_n^*(P) = \sup_{\mathbf t\in[0,1]^d}\Big\lvert\frac{\#\{i:\mathbf x_i\in[\mathbf 0,\mathbf t)\}}{n} - \prod_{j=1}^dt_j\Big\rvert.$$
In one dimension this is the Kolmogorov–Smirnov distance between the <a href="#/t/empirical-cdf">empirical CDF</a> of the points and the uniform CDF.</p>
</div>
<p>For IID uniform points, $\sqrt nD_n^*$ converges in law to the Kolmogorov distribution in $d=1$ (and is of order $\sqrt{\log\log n}$ almost surely, by the law of the iterated logarithm), so random points have discrepancy $\approx n^{-1/2}$. In one dimension the minimum possible is $D_n^* = 1/(2n)$, attained by the centred grid $\{(2i-1)/(2n)\}$.</p>
<div class="thm" data-name="Koksma–Hlawka inequality">
<p>For $f$ of bounded variation $V_{\text{HK}}(f)$ in the sense of Hardy and Krause,
$$\Big\lvert\frac1n\sum_{i=1}^nf(\mathbf x_i) - \int_{[0,1]^d}f\Big\rvert\ \le\ V_{\text{HK}}(f)\,D_n^*(P).$$
In $d=1$ (Koksma, 1942), $V_{\text{HK}}$ is the total variation $\int_0^1\lvert f'\rvert$ for absolutely continuous $f$.</p>
</div>
<details class="proof"><summary>Proof ($d=1$, $f$ absolutely continuous)</summary>
<p>Let $F_n(t) = \frac1n\#\{i: x_i< t\}$. For each point, $\int_0^1\ind\{x_i< t\}f'(t)\d t = f(1)-f(x_i)$, so averaging, $\int_0^1F_n(t)f'(t)\d t = f(1) - \frac1n\sum f(x_i)$. Integration by parts gives $\int_0^1t\,f'(t)\d t = f(1) - \int_0^1f$. Subtracting,
$$\frac1n\sum_{i}f(x_i) - \int_0^1f = -\int_0^1\big(F_n(t)-t\big)f'(t)\d t,$$
whose absolute value is at most $\sup_t\lvert F_n(t)-t\rvert\int_0^1\lvert f'\rvert = D_n^*\,V(f)$. The multivariate proof applies the same idea to all $2^d-1$ coordinate projections (Hlawka, 1961), which is why the Hardy–Krause variation is a sum over faces of the cube.</p>
</details>
<p>The bound separates the integrand (variation) from the points (discrepancy), so a construction with small discrepancy is good for every integrand of bounded variation. It is a worst-case bound, usually very loose, and neither factor is easy to compute. Note also that indicator functions of sets not aligned with the axes (a disk, a half-space $x_1+x_2\le1$) have <em>infinite</em> Hardy–Krause variation in $d\ge2$: the guarantee evaporates for discontinuous integrands, and in practice the QMC advantage shrinks to roughly $n^{-1/2-1/(2d)}$ for such functions.</p>`,
      },
      {
        title: 'Constructions: van der Corput, Halton, Sobol, nets',
        html: R`
<div class="def" data-name="Radical inverse and van der Corput sequence">
<p>Write $i = \sum_{k\ge0}a_kb^k$ in base $b$. The radical inverse reflects the digits about the radix point: $\phi_b(i) = \sum_{k\ge0}a_kb^{-k-1}$. The van der Corput sequence is $\phi_b(0),\phi_b(1),\phi_b(2),\dots$; in base 2: $0, \tfrac12, \tfrac14, \tfrac34, \tfrac18, \tfrac58, \tfrac38, \tfrac78,\dots$</p>
</div>
<p>Each new point lands in the largest remaining gap, and any $b^m$ consecutive points starting at a multiple of $b^m$ contain exactly one point in each interval $[jb^{-m},(j+1)b^{-m})$. The discrepancy is $O(\log n/n)$.</p>
<ul>
<li><strong>Halton (1960):</strong> $\mathbf x_i = (\phi_{p_1}(i),\dots,\phi_{p_d}(i))$ with the first $d$ primes. Coprime bases make the coordinates jointly well distributed; $D_n^* = O((\log n)^d/n)$. For large primes the early points of two coordinates are strongly correlated (with bases 29 and 31 the first 29 points lie on a line), so Halton is usually scrambled or leaped in moderate dimension.</li>
<li><strong>Hammersley (1964):</strong> for fixed $n$, $(i/n, \phi_{p_1}(i),\dots,\phi_{p_{d-1}}(i))$ achieves $O((\log n)^{d-1}/n)$, at the cost of not being extensible in $n$.</li>
<li><strong>Sobol (1967), Faure (1982), Niederreiter (1987):</strong> <em>digital</em> sequences, in which coordinate $j$ is obtained by multiplying the digit vector of $i$ by a generator matrix $C_j$ over $\mathbb F_b$. Sobol uses $b=2$ (fast bitwise XOR implementation) and matrices built from primitive polynomials and "direction numbers" chosen to optimize low-dimensional projections (Joe & Kuo, 2008).</li>
</ul>
<div class="def" data-name="(t, m, s)-net">
<p>A set of $b^m$ points in $[0,1)^s$ is a $(t,m,s)$-net in base $b$ if every elementary box $\prod_j[a_jb^{-k_j},(a_j+1)b^{-k_j})$ of volume $b^{t-m}$ contains exactly $b^t$ points. Smaller $t$ means finer equidistribution. A $(t,s)$-sequence is one whose consecutive blocks of $b^m$ points are $(t,m,s)$-nets for every $m$; Sobol's sequence is a $(t,s)$-sequence in base 2.</p>
</div>
<p>All these constructions satisfy $D_n^* \le C_d(\log n)^d/n + O((\log n)^{d-1}/n)$. It is conjectured that $(\log n)^d/n$ is the optimal order for infinite sequences (equivalently $(\log n)^{d-1}/n$ for $n$-point sets), but this is proved only in the lowest dimension (Schmidt, 1972); Roth's (1954) general lower bound for point sets, of order $(\log n)^{(d-1)/2}/n$, is far from it.</p>`,
        lab: 'quasi-monte-carlo-points',
      },
      {
        title: 'The asymptotic rate and the high-dimensional paradox',
        html: R`
<p>$(\log n)^d/n$ beats $n^{-1/2}$ eventually, but "eventually" can be absurd. The function $(\log n)^d/n$ is increasing in $n$ until $n = e^d$, and it drops below $n^{-1/2}$ only when $(\log n)^d< \sqrt n$; for $d = 10$ that requires $\log n\approx90$, i.e. $n\approx10^{39}$. Taken at face value, Koksma–Hlawka says QMC is useless beyond a handful of dimensions.</p>
<p>Yet QMC routinely beats MC in finance with $d$ in the hundreds (Paskov & Traub, 1995). The resolution is <strong>effective dimension</strong> (Caflisch, Morokoff & Owen, 1997). Decompose $f$ by the ANOVA (Hoeffding) decomposition $f = \sum_{u\subseteq\{1..d\}}f_u$ into pieces depending only on the coordinates in $u$, with variances $\sigma_u^2$ summing to $\sigma^2$.</p>
<ul>
<li>$f$ has <em>effective dimension $s$ in the superposition sense</em> if the components with $\lvert u\rvert\le s$ carry, say, 99% of the variance.</li>
<li>$f$ has <em>effective dimension $s$ in the truncation sense</em> if the components with $u\subseteq\{1,\dots,s\}$ carry 99%.</li>
</ul>
<p>Low-discrepancy points are extremely well distributed in their low-dimensional projections, especially the leading coordinates, so integrands dominated by low-order and leading-coordinate effects are integrated at nearly the one- or two-dimensional rate. Constructions such as the Brownian bridge or principal-component path generation reorder a simulation's inputs so that the important variation sits in the first coordinates. Sloan and Woźniakowski (1998) formalized this with <em>weighted</em> function spaces in which coordinate $j$ has importance $\gamma_j$: if $\sum_j\gamma_j< \infty$, there are QMC rules whose worst-case error is bounded independently of $d$.</p>
<div class="note">
<p>QMC therefore sits between quadrature and Monte Carlo. Like quadrature it exploits smoothness; like Monte Carlo it is indifferent to nominal dimension as long as the integrand's variation lives in a few dimensions. Its failure modes are the reverse: genuinely high-dimensional interactions and discontinuities not aligned with the axes.</p>
</div>`,
      },
      {
        title: 'Randomized QMC: error bars and better rates',
        html: R`
<p>A deterministic QMC estimate comes with no usable error estimate: Koksma–Hlawka is not computable in practice. Randomization fixes this while preserving the structure.</p>
<div class="thm" data-kind="Proposition" data-name="Random shift (Cranley & Patterson, 1976)">
<p>Let $\mathbf\Delta\sim\text{Unif}[0,1)^d$ and $\tilde{\mathbf x}_i = (\mathbf x_i+\mathbf\Delta)\bmod1$. Then each $\tilde{\mathbf x}_i$ is uniform on $[0,1)^d$, so $\hat I = \frac1n\sum f(\tilde{\mathbf x}_i)$ is unbiased. With $R$ independent shifts, the replicate estimates $\hat I^{(1)},\dots,\hat I^{(R)}$ are IID, and their mean and sample standard deviation over $\sqrt R$ give an unbiased estimate and a valid ($t_{R-1}$-based) confidence interval.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For fixed $\mathbf x$, $\mathbf x+\mathbf\Delta\bmod1$ is uniform on the torus because translation preserves Lebesgue measure. Linearity gives unbiasedness for each replicate; independence of the shifts makes the replicates IID, and the usual $t$ interval applies (exactly if they are normal, asymptotically in $R$ otherwise).</p>
</details>
<p>The shifted points keep their relative geometry, so the variance of each replicate is typically far below $\sigma^2/n$. <strong>Owen's scrambling</strong> (1995) randomly permutes digits in a nested way that preserves the $(t,m,s)$-net property while making each point uniform. Owen proved that scrambled-net variance is $o(1/n)$ for every square-integrable $f$, never more than a constant (depending on $t,b,s$) times the MC variance, and $O(n^{-3}(\log n)^{d-1})$ for sufficiently smooth $f$: an RMSE near $n^{-3/2}$, better than deterministic QMC, because random errors in different elementary boxes cancel.</p>
<div class="warn" data-name="Using RQMC correctly">
<ul>
<li>Use sample sizes that respect the construction: powers of 2 for Sobol nets. Truncating at an arbitrary $n$ breaks the net property and can cost an order of magnitude.</li>
<li>Do not drop the first point of a Sobol sequence (the origin) as was once common; with scrambling it is harmless and dropping it destroys the net structure (Owen, 2022).</li>
<li>Spend the budget as $R\approx 10$–$30$ replicates of a large $n$, not many replicates of a small $n$; the rate gain is in $n$.</li>
<li>Map to non-uniform inputs by <a href="#/t/inverse-transform">inversion</a> coordinate by coordinate. Rejection sampling consumes a random number of coordinates and scrambles the correspondence between points and variates.</li>
</ul>
</div>`,
        lab: 'quasi-monte-carlo-error',
      },
    ],
    quiz: [
      { q: R`Compute the first eight points of the base-2 van der Corput sequence starting at $i=0$, and explain why it fills gaps.`,
        a: R`$0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875$. Reversing the binary digits of $i$ means the least significant bit of $i$, which alternates fastest, controls the most significant position of the point: consecutive points alternate between halves, then quarters, and so on, so every block of $2^m$ consecutive points (aligned at a multiple of $2^m$) has one point in each dyadic interval of length $2^{-m}$.` },
      { q: R`Why does the Koksma–Hlawka inequality give no guarantee for estimating the area of a disk inside $[0,1)^2$ by QMC?`,
        a: R`The indicator of a disk has infinite Hardy–Krause variation, because its discontinuity is not aligned with the coordinate axes. The bound becomes $\infty\cdot D_n^*$. QMC still converges (the points are uniformly distributed) but the rate is slower than for smooth integrands.` },
      { q: R`What does a random shift buy you, and what does it cost?`,
        a: R`It makes each point uniform, hence the estimator unbiased, and independent replicates give a valid standard error. It costs the replicates: the effective sample per replicate is $n$, and you need $R$ of them. The relative geometry of the points (and thus their low discrepancy modulo the torus) is preserved, so the variance per replicate remains small.` },
      { q: R`QMC beats MC on a 360-dimensional finance integral. Does this contradict the $(\log n)^d/n$ rate analysis?`,
        a: R`No. That rate is a worst-case bound over all functions of bounded variation. The finance integrand has low effective dimension: most of its variance comes from a few leading coordinates or low-order interactions, on which the low-discrepancy points are nearly optimally spread. The relevant dimension is effective, not nominal.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'particle-filters',
    domain: 'simulation',
    order: 90,
    title: 'Sequential Monte Carlo & Particle Filters',
    subtitle: 'Importance sampling one time step at a time, with resampling to keep the weights alive',
    level: 3,
    sources: { other: 'Gordon, Salmond & Smith (1993); Doucet, de Freitas & Gordon (2001); Del Moral (2004); Chopin & Papaspiliopoulos, An Introduction to Sequential Monte Carlo (2020)' },
    prereqs: ['importance-sampling', 'hidden-markov'],
    related: ['mcmc-foundations', 'markov-chains', 'bootstrap', 'gibbs-sampling'],
    tags: ['state-space model', 'filtering', 'sequential importance sampling', 'resampling', 'weight degeneracy', 'bootstrap filter', 'particle MCMC'],
    summary: R`In a state-space model a hidden Markov process $X_t$ is observed through noisy measurements $Y_t$, and the filtering distribution $p(x_t\mid y_{1:t})$ must be updated as each observation arrives. Except for linear-Gaussian models (Kalman) and finite state spaces (the HMM forward algorithm), the recursion has no closed form. Particle filters represent the filter by a weighted cloud of $N$ samples, propagate them through the dynamics, reweight by the likelihood of the new observation, and resample. Without resampling, sequential importance sampling suffers weight degeneracy: the variance of the weights grows exponentially in time until one particle carries all the weight. Resampling trades a little variance now for stability later, and the resulting bootstrap filter gives consistent filter estimates with error $O(N^{-1/2})$, uniformly in time under mixing conditions, plus an unbiased estimate of the likelihood that powers particle MCMC.`,
    survey: {
      task: R`Online estimation of $p(x_t\mid y_{1:t})$ and of the likelihood $p(y_{1:T})$ in nonlinear/non-Gaussian state-space models`,
      assumptions: R`Can simulate $X_t\mid X_{t-1}$ (or a proposal) and evaluate $g(y_t\mid x_t)$; low-to-moderate state dimension`,
      output: R`Weighted particles $\{(x_t^i, W_t^i)\}$ approximating the filter; unbiased $\hat p(y_{1:T})$`,
      strengths: 'Handles nonlinearity, multimodality and non-Gaussian noise; online; simple to implement; unbiased likelihood',
      weaknesses: 'Degenerates in high state dimension; path degeneracy for smoothing; Monte Carlo noise in likelihood',
      complexity: R`$O(NT)$ time; $O(N)$ memory for filtering`,
    },
    sections: [
      {
        title: 'State-space models and the filtering recursion',
        html: R`
<div class="def" data-name="State-space model (hidden Markov model)">
<p>A latent Markov chain $X_0\sim\mu$, $X_t\mid X_{t-1} = x\sim f(\cdot\mid x)$, observed through $Y_t\mid X_t = x\sim g(\cdot\mid x)$, with the $Y_t$ conditionally independent given the states. The finite-state case is the <a href="#/t/hidden-markov">hidden Markov model</a>; continuous states arise in tracking, econometrics (stochastic volatility), epidemiology and robotics.</p>
</div>
<p>The <strong>filtering distribution</strong> $\pi_t(x_t) = p(x_t\mid y_{1:t})$ obeys a two-step recursion:</p>
$$\begin{aligned}
\text{predict:}&\quad p(x_t\mid y_{1:t-1}) = \int f(x_t\mid x_{t-1})\,\pi_{t-1}(x_{t-1})\d x_{t-1},\\
\text{update:}&\quad \pi_t(x_t) = \frac{g(y_t\mid x_t)\,p(x_t\mid y_{1:t-1})}{p(y_t\mid y_{1:t-1})},\qquad p(y_t\mid y_{1:t-1}) = \int g(y_t\mid x_t)\,p(x_t\mid y_{1:t-1})\d x_t.
\end{aligned}$$
<p>The normalizing constants multiply to the likelihood: $p(y_{1:T}) = \prod_{t=1}^Tp(y_t\mid y_{1:t-1})$, needed for parameter estimation. When $f$ and $g$ are linear-Gaussian, every distribution in the recursion is Gaussian and the Kalman filter propagates means and covariances exactly. When the state space is finite, the integrals are sums (the forward algorithm). Otherwise the predict step is an intractable integral, which is where Monte Carlo enters.</p>
<div class="ex" data-name="A standard nonlinear benchmark">
<p>The model of Kitagawa (1996) and Gordon et al. (1993):
$$X_t = \tfrac12X_{t-1} + \frac{25X_{t-1}}{1+X_{t-1}^2} + 8\cos(1.2t) + V_t,\qquad Y_t = \frac{X_t^2}{20} + W_t,$$
with $V_t\sim\N(0,10)$ and $W_t\sim\N(0,1)$. Because $Y_t$ depends on $X_t^2$, the filter is frequently bimodal (the sign of $X_t$ is ambiguous), and Gaussian approximations such as the extended Kalman filter fail badly. The lab tracks this model.</p>
</div>`,
      },
      {
        title: 'Sequential importance sampling',
        html: R`
<p>Work on path space. The target at time $t$ is the joint smoothing density
$$\pi_t(x_{0:t})\ \propto\ \gamma_t(x_{0:t}) = \mu(x_0)\prod_{s=1}^tf(x_s\mid x_{s-1})\,g(y_s\mid x_s),$$
known up to the constant $p(y_{1:t})$. Choose a proposal that builds paths sequentially, $q_t(x_{0:t}) = q_0(x_0)\prod_{s=1}^tq(x_s\mid x_{s-1},y_s)$.</p>
<div class="thm" data-kind="Proposition" data-name="Weight recursion">
<p>The importance weights $w_t = \gamma_t/q_t$ satisfy
$$w_t(x_{0:t}) = w_{t-1}(x_{0:t-1})\cdot\alpha_t(x_{t-1},x_t),\qquad\alpha_t = \frac{f(x_t\mid x_{t-1})\,g(y_t\mid x_t)}{q(x_t\mid x_{t-1},y_t)}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Both $\gamma_t$ and $q_t$ are products whose first $t-1$ factors are those of $\gamma_{t-1}$ and $q_{t-1}$; dividing, the common factors give $w_{t-1}$ and the new factors give $\alpha_t$.</p>
</details>
<p>So each particle can be extended by one step and its weight updated by a local "incremental weight", at $O(1)$ cost per particle per time step regardless of $t$. The weighted particles $\{(x_t^i, W_t^i)\}$, $W_t^i\propto w_t^i$, approximate $\pi_t$ by <a href="#/t/importance-sampling">self-normalized importance sampling</a>, and $\frac1N\sum_iw_t^i$ is an unbiased estimate of $p(y_{1:t})$.</p>
<div class="thm" data-kind="Proposition" data-name="Weight degeneracy">
<p>Suppose the incremental weights along a particle's path are independent across time with $\E\alpha_s = 1$ and $\E\alpha_s^2 = 1+c$, $c>0$. Then $\E w_t^2/(\E w_t)^2 = (1+c)^t$, so the effective sample size satisfies $\text{ESS}/N\to(1+c)^{-t}$ as $N\to\infty$: it decays exponentially in $t$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By independence, $\E w_t = \prod_s\E\alpha_s = 1$ and $\E w_t^2 = \prod_s\E\alpha_s^2 = (1+c)^t$. The ESS limit is $(\E w)^2/\E w^2$ (see <a href="#/t/importance-sampling">ESS and the chi-square divergence</a>).</p>
</details>
<p>The independence assumption is a simplification, but the conclusion is general: Kong, Liu and Wong (1994) showed that the unconditional variance of SIS weights increases over time. Equivalently, $\log w_t$ is a sum of $t$ terms, so by the CLT its spread grows like $\sqrt t$, and the largest of $N$ lognormal-like weights soon dominates the sum. After a few dozen steps, a plain SIS filter typically has one particle with nearly all the weight: the estimate is effectively based on a single path.</p>`,
      },
      {
        title: 'Resampling',
        html: R`
<p>The fix (Gordon, Salmond & Smith, 1993) is to <strong>resample</strong>: replace the weighted cloud by an unweighted one, drawing particles in proportion to their weights, so that low-weight particles die and high-weight particles split. Afterwards all weights are reset to $1/N$.</p>
<div class="def" data-name="Unbiased resampling schemes">
<p>A resampling scheme produces offspring counts $(N^1,\dots,N^N)$ with $\sum_iN^i = N$ and $\E[N^i\mid W] = NW^i$.</p>
<ul>
<li><strong>Multinomial:</strong> $N$ IID draws from the categorical distribution $W$.</li>
<li><strong>Stratified</strong> (Kitagawa, 1996): one independent uniform in each of the $N$ intervals $[k/N,(k+1)/N)$, inverted through the cumulative weights.</li>
<li><strong>Systematic</strong> (Carpenter, Clifford & Fearnhead, 1999): a single uniform $U\in[0,1/N)$ and the grid $U + k/N$; each $N^i\in\{\lfloor NW^i\rfloor,\lceil NW^i\rceil\}$.</li>
<li><strong>Residual:</strong> keep $\lfloor NW^i\rfloor$ copies deterministically, fill the remainder multinomially.</li>
</ul>
</div>
<p>Unbiasedness ($\E[N^i\mid W] = NW^i$) guarantees that $\frac1N\sum_i\varphi(\tilde x^i)$ after resampling has the same conditional mean as $\sum_iW^i\varphi(x^i)$ before. Resampling always <em>adds</em> variance at the current step; residual and stratified resampling add provably less than multinomial (Douc & Cappé, 2005), and systematic resampling is the cheapest and usually the best in practice, though it lacks a general dominance guarantee.</p>
<p>Because resampling adds noise, it is done adaptively: resample only when $\text{ESS}< \tau N$, commonly $\tau = 1/2$ (Liu & Chen, 1995). The point of resampling is not to improve the present estimate; it is to stop the future weights from being products of ever more factors. After resampling, particles are concentrated where the filter has mass, so subsequent propagation spends effort where it matters, which is precisely what the exponential decay of SIS could not do.</p>
<div class="warn" data-name="Path degeneracy">
<p>Resampling cures weight degeneracy of the current state but creates <em>path degeneracy</em>: tracing the ancestral lineages of today's particles back in time, they coalesce, and for $s\ll t$ all particles share one or a few ancestors. The particle approximation of the smoothing distribution $p(x_s\mid y_{1:t})$ then collapses. Filtering is fine; smoothing needs fixed-lag approximations, forward-filtering backward-smoothing (Godsill, Doucet & West, 2004), or particle MCMC.</p>
</div>`,
      },
      {
        title: 'The bootstrap particle filter',
        html: R`
<div class="algo" data-name="Bootstrap particle filter (Gordon, Salmond & Smith, 1993)">
<ol>
<li>Draw $x_0^i\sim\mu$, set $W_0^i = 1/N$, $\hat\ell = 0$.</li>
<li>For $t = 1,\dots,T$:
<ol>
<li>If $\text{ESS}_{t-1}< \tau N$: resample ancestor indices $a^i$ with probabilities $W_{t-1}$ and set $W_{t-1}^i = 1/N$; otherwise $a^i = i$.</li>
<li>Propagate: $x_t^i\sim f(\cdot\mid x_{t-1}^{a^i})$.</li>
<li>Weight: $w_t^i = W_{t-1}^{a^i}\,g(y_t\mid x_t^i)$ (with $W^{a^i}_{t-1} = 1/N$ after resampling).</li>
<li>Likelihood increment: $\hat\ell \mathrel{+}= \log\sum_iw_t^i$; normalize $W_t^i = w_t^i/\sum_jw_t^j$.</li>
</ol></li>
<li>Filter estimates: $\widehat{\E}[\varphi(X_t)\mid y_{1:t}] = \sum_iW_t^i\varphi(x_t^i)$; likelihood estimate $\hat p(y_{1:T}) = e^{\hat\ell}$.</li>
</ol>
</div>
<p>The proposal is the prior dynamics $q = f$, so the incremental weight is just the likelihood $g(y_t\mid x_t)$. This is simple, needs only the ability to <em>simulate</em> $f$ (not evaluate it), and fails when observations are very informative relative to the dynamics: then few prior particles land where $g$ is large. Better proposals use $y_t$:</p>
<ul>
<li>The <strong>locally optimal proposal</strong> $q(x_t\mid x_{t-1},y_t) = p(x_t\mid x_{t-1},y_t)$ minimizes the variance of the incremental weight conditional on the past, which becomes $p(y_t\mid x_{t-1})$ (Doucet, Godsill & Andrieu, 2000). It is available in closed form for Gaussian $f$ with linear-Gaussian $g$.</li>
<li>The <strong>auxiliary particle filter</strong> (Pitt & Shephard, 1999) resamples using a look-ahead weight approximating $p(y_t\mid x_{t-1})$, so that particles likely to explain $y_t$ are selected before propagation.</li>
<li><strong>Rao–Blackwellized</strong> filters integrate conditionally linear-Gaussian components with Kalman filters and sample only the rest: the <a href="#/t/variance-reduction">conditional Monte Carlo</a> principle.</li>
</ul>`,
        lab: 'particle-filters-tracking',
      },
      {
        title: 'Theory: consistency, unbiased likelihoods, and the curse of dimension',
        html: R`
<div class="thm" data-name="Convergence of particle filters">
<p>For bounded $\varphi$ and $g$ bounded and positive, the bootstrap filter (with multinomial resampling at every step) satisfies
$$\E\Big[\Big(\sum_iW_t^i\varphi(x_t^i) - \E[\varphi(X_t)\mid y_{1:t}]\Big)^2\Big]\le\frac{C_t\lVert\varphi\rVert_\infty^2}{N},$$
and a CLT holds with an explicit asymptotic variance (Del Moral, 2004; Chopin, 2004). Under mixing conditions on the hidden chain (the filter forgets its initial condition exponentially fast), $C_t$ can be bounded uniformly in $t$ (Del Moral & Guionnet, 2001).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Induct over time. Each step consists of a resampling (an empirical measure of $N$ conditionally IID draws, contributing error $O(N^{-1/2})$ by a conditional variance bound), a propagation (likewise), and a reweighting, which is a ratio of two integrals against the current particle measure and transmits the previous error with a multiplicative constant depending on $\sup g/\int g\,\d\pi$. Accumulating gives $C_t$. Uniform-in-time bounds use the fact that, under mixing, errors introduced at time $s$ are contracted by the filter's forgetting as $t-s$ grows, so the accumulated error is a convergent geometric series.</p>
</details>
<div class="thm" data-name="Unbiased likelihood (Del Moral, 2004)">
<p>For any $N\ge1$ and any unbiased resampling scheme, $\E[\hat p(y_{1:T})] = p(y_{1:T})$.</p>
</div>
<p>By Jensen, $\E\log\hat p\le\log p$: the log-likelihood estimate is biased downward, by about half its variance when it is approximately normal. Unbiasedness of $\hat p$ itself is what makes <strong>particle MCMC</strong> exact (Andrieu, Doucet & Holenstein, 2010): a <a href="#/t/metropolis-hastings">Metropolis–Hastings</a> chain on parameters that uses $\hat p_\theta(y_{1:T})$ in place of the intractable likelihood still targets the exact posterior, the "pseudo-marginal" principle. Efficiency is best when $N$ is chosen so that $\Var\log\hat p\approx1$ (Doucet, Pitt, Deligiannidis & Kohn, 2015).</p>
<div class="warn" data-name="High-dimensional states">
<p>Particle filters do not escape the dimension problem of importance sampling. For the bootstrap filter in a $d$-dimensional state with informative observations in every coordinate, the number of particles needed to avoid collapse grows exponentially in $d$ (more precisely, in the variance of the log-likelihood; Bengtsson, Bickel & Li, 2008; Snyder et al., 2008). Weather-scale data assimilation therefore uses ensemble Kalman filters and localization, not plain particle filters.</p>
</div>`,
        lab: 'particle-filters-likelihood',
      },
      {
        title: 'Beyond filtering: SMC samplers',
        html: R`
<p>Nothing above requires the sequence of targets to come from time. Any sequence $\pi_0,\pi_1,\dots,\pi_K$ of distributions that bridges an easy one to a hard one can be traversed by reweighting, resampling and moving particles with an MCMC kernel invariant for the current target (Del Moral, Doucet & Jasra, 2006). Examples: tempering $\pi_k\propto\pi_0^{1-\beta_k}\pi^{\beta_k}$ from prior to posterior (compare <a href="#/t/tempering">tempering in MCMC</a>), data tempering (adding observations one at a time), and rare-event splitting (raising a threshold step by step). The output includes an unbiased estimate of the normalizing constant ratio, i.e. the marginal likelihood, which plain MCMC does not give.</p>
<div class="note">
<p>The recurring pattern of this whole domain shows up in SMC: <a href="#/t/importance-sampling">importance sampling</a> is exact in expectation but its variance explodes when the proposal and target drift apart; resampling (a <a href="#/t/bootstrap">bootstrap</a> of the weighted cloud) and MCMC moves (rejuvenation) keep each step's mismatch small, so that the difficulty is spread over many easy steps rather than one impossible one.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In the bootstrap filter, what is the incremental weight, and why can the filter work even if $f(x_t\mid x_{t-1})$ cannot be evaluated?`,
        a: R`With proposal $q = f$, the incremental weight $f\,g/q$ reduces to $g(y_t\mid x_t)$, the observation likelihood. The transition density cancels, so only the ability to simulate from the dynamics is needed (useful for models defined by simulators, e.g. stochastic differential equations solved numerically).` },
      { q: R`Why does resampling at every step not make the estimate of $\E[\varphi(X_t)\mid y_{1:t}]$ at the current step better?`,
        a: R`Resampling is unbiased but adds variance: the resampled average $\frac1N\sum\varphi(\tilde x^i)$ has the same conditional mean as $\sum W^i\varphi(x^i)$ plus extra multinomial noise. Its benefit is for the future: it removes particles with negligible weight so that subsequent propagation is not wasted on them, preventing weights from becoming products of many factors.` },
      { q: R`If the log-likelihood estimate $\log\hat p$ is approximately $\N(m, s^2)$ and $\hat p$ is unbiased, what is $m$?`,
        a: R`$\E\hat p = e^{m+s^2/2} = p$, so $m = \log p - s^2/2$. The log-likelihood estimate is biased downward by half its variance, even though $\hat p$ is unbiased.` },
      { q: R`What is path degeneracy, and which inferential target suffers from it?`,
        a: R`After repeated resampling, the ancestral lines of the current particles coalesce, so the particle approximation of early states $x_s$, $s\ll t$, rests on very few distinct values. Filtering ($x_t$ given $y_{1:t}$) is unaffected; smoothing ($x_s$ given $y_{1:t}$) and parameter learning through the path suffer.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'simulation-studies',
    domain: 'simulation',
    order: 100,
    title: 'Designing Simulation Studies',
    subtitle: 'Simulation studies are experiments: aims, data-generating mechanisms, estimands, methods, performance measures, and their Monte Carlo error',
    level: 2,
    sources: { other: 'Morris, White & Crowther, Statistics in Medicine (2019); Koehler, Brown & Haneuse, Am. Statistician (2009); Brown, Cai & DasGupta (2001)' },
    prereqs: ['monte-carlo-integration', 'confidence-intervals'],
    related: ['hypothesis-testing', 'bootstrap', 'pseudo-random', 'point-estimation', 'variance-reduction'],
    tags: ['ADEMP', 'Monte Carlo standard error', 'coverage', 'bias', 'power', 'number of replications', 'reproducibility'],
    summary: R`Much of statistical methodology is evaluated by simulation: draw data from a known mechanism, apply the methods, and measure bias, variance, coverage or power. A simulation study is an experiment with the computer as laboratory and deserves the same discipline as any other: a stated aim, deliberately chosen data-generating mechanisms, a precisely defined estimand, a fixed set of methods, and performance measures each reported with its own Monte Carlo standard error. Morris, White and Crowther's ADEMP framework structures this. The number of replications should be chosen to make those standard errors small enough to answer the question; a coverage of 93.8% from 500 replications cannot be distinguished from 95%. Reproducibility (seeds, streams, stored states) and fairness (common random numbers, no tuning on the evaluation data) complete the design.`,
    survey: {
      task: 'Evaluate statistical methods (bias, precision, coverage, size, power) where theory is unavailable or asymptotic',
      assumptions: 'A data-generating mechanism with known truth; methods that can be run automatically on each dataset',
      output: 'Performance estimates per method and scenario, each with a Monte Carlo standard error',
      strengths: 'Exact finite-sample truth is known; any method, any scenario; reveals failures asymptotics hide',
      weaknesses: 'Conclusions limited to the scenarios simulated; easily biased toward a favoured method; MC error often ignored',
      complexity: R`$n_{\text{sim}}\times(\text{scenarios})\times(\text{cost of all methods})$`,
    },
    sections: [
      {
        title: 'Why simulate, and why design',
        html: R`
<p>Theory tells us how estimators behave as $n\to\infty$; practitioners use $n = 40$. Bootstrap intervals, robust estimators, small-sample corrections, model selection procedures, and machine learning pipelines all have finite-sample properties that are hard or impossible to derive. Simulation provides ground truth: the data are generated from a mechanism with known parameters, so the error of every estimate is known.</p>
<p>Because simulation studies are cheap, they are often run carelessly: a handful of scenarios chosen to flatter a new method, 100 replications, results reported to three decimals with no uncertainty. Every performance measure is itself a <a href="#/t/monte-carlo-integration">Monte Carlo estimate</a> of an expectation over the data-generating mechanism, and should be treated as such: with a target precision, a standard error, and an honest description of the population of scenarios it speaks for.</p>`,
      },
      {
        title: 'The ADEMP framework',
        html: R`
<p>Morris, White and Crowther (2019) organize the planning of a simulation study around five elements, to be written down before any code is run.</p>
<table>
<thead><tr><th>Element</th><th>Question</th><th>Typical choices</th></tr></thead>
<tbody>
<tr><td><strong>A</strong>ims</td><td>What is the study for?</td><td>Check a method works as intended; compare methods; find where a method breaks; check software</td></tr>
<tr><td><strong>D</strong>ata-generating mechanisms</td><td>How are datasets simulated?</td><td>Parametric models with factors varied (sample size, effect size, correlation, error distribution, missingness); resampling from a real dataset ("plasmode")</td></tr>
<tr><td><strong>E</strong>stimands</td><td>What is the target of inference?</td><td>A parameter of the DGM (a mean difference, a regression coefficient, a marginal risk ratio); a null hypothesis; a prediction</td></tr>
<tr><td><strong>M</strong>ethods</td><td>What is applied to each dataset?</td><td>The estimators/tests/intervals compared, including a well-established benchmark</td></tr>
<tr><td><strong>P</strong>erformance measures</td><td>How is a method judged?</td><td>Bias, empirical SE, MSE, coverage, interval width, type I error, power, relative error of model-based SE, convergence failures</td></tr>
</tbody>
</table>
<div class="note" data-name="Estimands deserve special care">
<p>Many apparently contradictory simulation results come from methods that target different estimands: a conditional versus a marginal odds ratio, a population average versus a sample average treatment effect, a parameter of the true model versus the best approximation within a misspecified one. Define the estimand as an explicit functional of the DGM, and compute its true value (analytically or by a very large separate simulation) before evaluating bias or coverage.</p>
</div>
<p><strong>Designing the DGMs.</strong> Vary factors in a factorial design when interactions are plausible (a method may fail only with small $n$ <em>and</em> heavy tails); use one-at-a-time variation around a base case when the number of factors is large. Include scenarios where the competitor should win, and scenarios that break assumptions, not only those that showcase the new method.</p>`,
      },
      {
        title: 'Performance measures and their Monte Carlo standard errors',
        html: R`
<p>Let $\hat\theta_1,\dots,\hat\theta_{R}$ be the estimates from $R = n_{\text{sim}}$ independent replications, $\bar\theta$ their mean, $\theta$ the true estimand, $\widehat{\se}_r$ the model-based standard errors and $C_r$ the indicator that interval $r$ covers $\theta$.</p>
<table>
<thead><tr><th>Measure</th><th>Estimate</th><th>Monte Carlo SE</th></tr></thead>
<tbody>
<tr><td>Bias</td><td>$\bar\theta - \theta$</td><td>$\sqrt{\frac{1}{R(R-1)}\sum_r(\hat\theta_r-\bar\theta)^2}$</td></tr>
<tr><td>Empirical SE</td><td>$\text{EmpSE} = \sqrt{\frac1{R-1}\sum_r(\hat\theta_r-\bar\theta)^2}$</td><td>$\text{EmpSE}/\sqrt{2(R-1)}$ (normal approximation)</td></tr>
<tr><td>MSE</td><td>$\frac1R\sum_r(\hat\theta_r-\theta)^2$</td><td>$\sqrt{\frac{1}{R(R-1)}\sum_r\big[(\hat\theta_r-\theta)^2-\widehat{\MSE}\big]^2}$</td></tr>
<tr><td>Coverage</td><td>$\hat c = \frac1R\sum_rC_r$</td><td>$\sqrt{\hat c(1-\hat c)/R}$</td></tr>
<tr><td>Rejection rate (size or power)</td><td>$\hat\beta = \frac1R\sum_r\ind\{\text{reject}_r\}$</td><td>$\sqrt{\hat\beta(1-\hat\beta)/R}$</td></tr>
<tr><td>Average model SE</td><td>$\text{ModSE} = \sqrt{\frac1R\sum_r\widehat{\se}_r^2}$</td><td>$\sqrt{\widehat\Var(\widehat{\se}_r^2)/(4R\,\text{ModSE}^2)}$ (delta method)</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Proposition" data-name="Where the formulas come from">
<p>Each performance measure is a sample mean (or a smooth function of sample means) over IID replications, so its Monte Carlo SE is the usual standard error of a mean, propagated through the <a href="#/t/delta-method">delta method</a> when needed.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Bias is the mean of $\hat\theta_r-\theta$, whose variance is $\Var(\hat\theta)/R$. Coverage and rejection rates are means of Bernoulli variables, variance $c(1-c)/R$. MSE is the mean of $(\hat\theta_r-\theta)^2$, with variance estimated by the sample variance of those squares over $R$. For EmpSE $= \sqrt{s^2}$: if the $\hat\theta_r$ are normal, $(R-1)s^2/\sigma^2\sim\chi^2_{R-1}$, so $\Var(s^2) = 2\sigma^4/(R-1)$, and the delta method with $\frac{\d}{\d v}\sqrt v = \frac{1}{2\sqrt v}$ gives $\Var(s)\approx\sigma^2/(2(R-1))$. For non-normal $\hat\theta$, replace $2\sigma^4$ by $\mu_4-\sigma^4$ (approximately). ModSE is the square root of a mean of $\widehat{\se}_r^2$, handled by the same delta-method step.</p>
</details>
<p>Comparisons between methods should use the <em>paired</em> differences across replications when the methods were applied to the same datasets: the MCSE of a difference in bias is $\text{sd}_r(\hat\theta_{A,r}-\hat\theta_{B,r})/\sqrt R$, often much smaller than either method's own MCSE because the datasets' idiosyncrasies affect both methods alike. This is <a href="#/t/variance-reduction">common random numbers</a> at the level of whole datasets.</p>`,
        lab: 'simulation-studies-coverage',
      },
      {
        title: 'How many replications?',
        html: R`
<p>Choose $R$ so that the Monte Carlo SE of the key performance measure is small compared with differences that matter.</p>
<ul>
<li><strong>Coverage.</strong> $R = c(1-c)/\text{MCSE}^2$. For nominal $c = 0.95$ and a desired MCSE of $0.5$ percentage points: $R = 0.95\cdot0.05/0.005^2 = 1900$. The worst case $c = 0.5$ (useful when coverage may be poor) needs $R = 10{,}000$ for the same MCSE.</li>
<li><strong>Type I error.</strong> At $\alpha = 0.05$ with $R = 1000$, the MCSE is $\sqrt{0.05\cdot0.95/1000}\approx0.0069$. An observed size of $0.058$ is only $1.2$ MCSEs above nominal: no evidence of anti-conservativeness.</li>
<li><strong>Bias.</strong> $R = \Var(\hat\theta)/\text{MCSE}^2$. If $\text{SD}(\hat\theta)\approx0.5$ and you want to detect bias of $0.02$ with MCSE $0.005$, you need $R = 0.25/0.000025 = 10{,}000$.</li>
</ul>
<p>Estimate $\Var(\hat\theta)$ from a small pilot. Because the MCSE shrinks only like $R^{-1/2}$, the same remarks as for any Monte Carlo computation apply: one extra digit costs a hundredfold more replications, and <a href="#/t/variance-reduction">variance reduction</a> (control variates using a method with known performance, common datasets across methods) is often cheaper than brute force.</p>
<div class="warn" data-name="Multiplicity across scenarios">
<p>With 40 scenarios each estimated with an MCSE of $0.5$ points, about two coverages will fall more than 2 MCSE from their true value by chance alone. A method whose coverage is 93.9% in one scenario and 95.1% everywhere else is probably fine; a method that is below nominal in every small-$n$ scenario is not. Look for patterns across the design, not isolated deviations. The second lab shows how slowly a running coverage estimate settles.</p>
</div>`,
        lab: 'simulation-studies-mcse',
      },
      {
        title: 'Good practice',
        html: R`
<ol>
<li><strong>Write a protocol</strong> with the ADEMP elements before running anything, and report deviations.</li>
<li><strong>Reproducibility.</strong> Use a modern <a href="#/t/pseudo-random">generator</a>, set and record the seed, and store the generator state at the start of each replication (or use independent streams per replication). Then any single replication, for instance one where a method failed, can be re-created in isolation.</li>
<li><strong>Common datasets.</strong> Apply all methods to the same simulated datasets and analyse paired differences.</li>
<li><strong>Failures.</strong> Record non-convergence, errors and warnings; report their frequency per method and scenario. Silently discarding failed replications biases performance toward "datasets where the method worked", and comparing methods on different subsets of datasets is unfair.</li>
<li><strong>Check the code.</strong> Verify the DGM by simulating one huge dataset and checking that parameters are recovered; include a method whose performance is known (e.g. the MLE in a correctly specified model).</li>
<li><strong>Do not tune on the test scenarios.</strong> Hyperparameters of a new method tuned by looking at simulation results are a form of overfitting; tune on separate scenarios or seeds.</li>
<li><strong>Exact computation where possible.</strong> Some performance measures can be computed without simulation. The coverage of an interval for a binomial proportion is $\sum_k\binom nk p^k(1-p)^{n-k}\ind\{p\in\text{CI}(k)\}$, an exact finite sum.</li>
<li><strong>Present results with their MCSEs.</strong> Tables with MCSEs, or plots with error bars; for many scenarios, nested-loop plots (Rücker & Schwarzer, 2014) or lattice plots by factor.</li>
</ol>
<div class="ex" data-name="The Wald interval's chaotic coverage">
<p>Brown, Cai and DasGupta (2001) showed that the Wald interval $\hat p\pm z\sqrt{\hat p(1-\hat p)/n}$ for a binomial proportion has coverage that oscillates wildly in $p$ and $n$, falling well below 95% even for $n$ in the hundreds at some $p$, and not improving monotonically with $n$. The oscillation is a lattice effect: $\hat p$ takes only $n+1$ values, and the coverage function jumps as $p$ crosses interval endpoints. Simulation with a coarse grid of $p$ values and a few hundred replications can miss this entirely or misattribute it to Monte Carlo noise. The first lab lets you compare simulated coverage (with error bars) against the exact coverage curve for Wald, Wilson and Agresti–Coull intervals.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A new interval shows coverage 0.938 in $R = 500$ replications. Is there evidence it undercovers?`,
        a: R`MCSE $\approx\sqrt{0.938\cdot0.062/500}\approx0.0108$. The deficit $0.012$ is about $1.1$ MCSE: no evidence. To resolve a 1-point deficit reliably you would want MCSE $\approx0.003$, i.e. $R\approx0.95\cdot0.05/0.003^2\approx5300$.` },
      { q: R`Two estimators are applied to the same 1000 simulated datasets. Their biases are $0.031$ (MCSE $0.010$) and $0.012$ (MCSE $0.010$). Can you conclude the first is more biased?`,
        a: R`Not from the separate MCSEs, which would suggest a difference of $0.019$ with SE about $0.014$. Because the datasets are shared, compute the MCSE of the paired difference $\hat\theta_{A,r}-\hat\theta_{B,r}$: if the estimators are highly correlated across datasets, that MCSE might be $0.002$, making the difference clearly significant.` },
      { q: R`What goes wrong if replications where a method failed to converge are dropped only for that method?`,
        a: R`The method is then evaluated on a selected subset of datasets (those where it converged), which are typically "easier", so its performance is overstated; and comparisons with other methods are no longer on the same datasets. Report failure rates and compare on common subsets or with a pre-specified fallback.` },
      { q: R`Which ADEMP element is most often responsible for conflicting conclusions between published simulation studies?`,
        a: R`Often the estimand: studies may compare methods targeting different quantities (marginal vs conditional effects, model parameters under misspecification), or define "truth" inconsistently, and then report bias relative to different targets. DGMs that favour a particular method are the other frequent cause.` },
    ],
  });
})();
