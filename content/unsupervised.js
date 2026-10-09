/* Unsupervised Learning: clustering, mixtures & EM, dimension reduction, graphical models. */
(function () {
  const R = String.raw;

  /* ===================================================================== k-means */
  Atlas.topic({
    id: 'kmeans',
    domain: 'unsupervised',
    order: 10,
    title: "k-Means & Lloyd's Algorithm",
    subtitle: 'Partitioning by within-cluster sum of squares, and the alternating minimisation that almost solves it',
    level: 2,
    sources: { aos: '—', cb: '—', other: 'ESL §14.3.6–14.3.11; Bishop PRML §9.1; Murphy (2012) Ch. 25; Arthur & Vassilvitskii (2007); Pollard (1981)' },
    prereqs: ['multivariate'],
    related: ['gaussian-mixtures', 'em-algorithm', 'hierarchical-clustering', 'spectral-density-clustering'],
    tags: ['clustering', 'vector quantization', 'Voronoi', 'NP-hard', 'k-means++'],
    summary: R`k-means chooses $K$ centres and assigns each point to its nearest one so that the total within-cluster sum of squared distances is as small as possible. The exact problem is NP-hard, but Lloyd's algorithm — alternate "assign to the nearest centre" and "move each centre to its cluster mean" — decreases the objective at every step and stops in finitely many steps at a fixed point. What it finds is a local optimum whose quality depends heavily on initialisation; $k$-means++ seeding gives an $O(\log K)$ approximation in expectation. Statistically, k-means is the zero-variance limit of EM for an isotropic Gaussian mixture, which tells you exactly which cluster shapes it can and cannot find.`,
    survey: {
      task: 'Partition $n$ points in $\\R^p$ into $K$ groups; vector quantisation',
      assumptions: 'Euclidean geometry is meaningful; clusters roughly spherical, similar spread; $K$ given',
      output: 'Centres $\\mu_1,\\dots,\\mu_K$ and a hard assignment (Voronoi partition)',
      strengths: 'Fast ($O(nKp)$ per pass), simple, scales to huge $n$; strong baseline and seeding for richer models',
      weaknesses: 'Local optima; scale-dependent; fails on elongated, nested or unequal-size clusters; $K$ must be chosen',
      complexity: '$O(nKp)$ per Lloyd iteration; exact optimum NP-hard',
    },
    sections: [
      {
        title: 'The problem: compress data into $K$ prototypes',
        html: R`
<p>Given $x_1,\dots,x_n \in \R^p$, a <em>clustering</em> into $K$ groups is a map $C:\{1,\dots,n\}\to\{1,\dots,K\}$. Without labels there is no ground truth to fit, so every clustering method is defined by what it calls a good partition. k-means takes the most classical answer: a partition is good if each group is tightly packed around its own mean.</p>
<div class="def" data-name="k-means objective">
<p>For a partition $C$ with clusters $C_k = \{i : C(i)=k\}$, sizes $n_k$ and means $\bar x_k$, the <strong>within-cluster sum of squares</strong> is
$$W(C) = \sum_{k=1}^K \sum_{i\in C_k} \lVert x_i - \bar x_k\rVert^2 .$$
k-means seeks $C^\star = \argmin_C W(C)$. Equivalently it minimises over centres and assignments jointly,
$$\min_{\mu_1,\dots,\mu_K}\;\min_{C}\; J(C,\mu) ,\qquad J(C,\mu)=\sum_{i=1}^n \lVert x_i - \mu_{C(i)}\rVert^2 .$$</p>
</div>
<p>Two other readings of $W$ are worth having in hand. First, a pairwise form that never mentions centres:</p>
<div class="thm" data-kind="Lemma" data-name="Pairwise form of the within-cluster scatter">
<p>For each cluster, $\displaystyle\sum_{i\in C_k}\lVert x_i-\bar x_k\rVert^2 = \frac{1}{2n_k}\sum_{i\in C_k}\sum_{i'\in C_k}\lVert x_i - x_{i'}\rVert^2$. Hence $W(C) = \sum_k \frac{1}{2n_k}\sum_{i,i'\in C_k}\lVert x_i-x_{i'}\rVert^2$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $x_i - x_{i'} = (x_i - \bar x_k) - (x_{i'} - \bar x_k)$ and expand: $\sum_{i,i'}\lVert x_i - x_{i'}\rVert^2 = 2n_k\sum_i \lVert x_i-\bar x_k\rVert^2 - 2\big\lVert\sum_i (x_i-\bar x_k)\big\rVert^2$. The last sum is zero by definition of the mean.</p>
</details>
<p>Second, the multivariate ANOVA identity $T = W(C) + B(C)$, where $T = \sum_i \lVert x_i - \bar x\rVert^2$ is the total scatter (fixed by the data) and $B(C) = \sum_k n_k \lVert \bar x_k - \bar x\rVert^2$ is the between-cluster scatter. Minimising within-cluster spread is therefore the same as maximising separation of the cluster means, weighted by cluster size.</p>
<div class="note">
<p>k-means is <strong>vector quantisation</strong>: replace each point by its codeword $\mu_{C(i)}$ and pay squared-error distortion. $W(C)/n$ is the mean distortion of a $K$-word codebook. That framing explains both its strength (an excellent compressor) and its main bias: it would rather split one big round blob into equal pieces than respect a long thin cluster, because the codebook only cares about squared distance.</p>
</div>`,
      },
      {
        title: "Lloyd's algorithm and why it converges",
        html: R`
<p>Joint minimisation over $(C,\mu)$ is hard, but each half is trivial with the other held fixed. Minimising $J$ over $C$ with $\mu$ fixed decouples across points: each $x_i$ goes to its nearest centre. Minimising over $\mu$ with $C$ fixed decouples across clusters, and the minimiser is the mean, by the bias–variance identity
$$\sum_{i\in C_k}\lVert x_i - m\rVert^2 = \sum_{i\in C_k}\lVert x_i - \bar x_k\rVert^2 + n_k\lVert \bar x_k - m\rVert^2 .$$</p>
<div class="algo" data-name="Lloyd's algorithm (Lloyd 1957/1982; Forgy 1965)">
<ol>
<li>Initialise centres $\mu_1^{(0)},\dots,\mu_K^{(0)}$ (e.g. $K$ random data points, or $k$-means++).</li>
<li><strong>Assignment:</strong> $C^{(t)}(i) = \argmin_k \lVert x_i - \mu_k^{(t)}\rVert^2$, ties broken by a fixed rule (e.g. keep the current label if it is among the minimisers).</li>
<li><strong>Update:</strong> $\mu_k^{(t+1)} = \frac{1}{n_k}\sum_{i : C^{(t)}(i)=k} x_i$ (if a cluster empties, re-seed it, e.g. at the point with the largest residual).</li>
<li>Repeat until the assignment no longer changes.</li>
</ol>
</div>
<p>The assignment step partitions space into the <strong>Voronoi cells</strong> of the centres, so k-means boundaries are always piecewise linear: the boundary between clusters $j$ and $k$ is the perpendicular bisector hyperplane $\{x: 2x^\top(\mu_k-\mu_j) = \lVert\mu_k\rVert^2-\lVert\mu_j\rVert^2\}$.</p>
<div class="thm" data-kind="Theorem" data-name="Monotone descent and finite termination">
<p>Let $J_t = J(C^{(t)},\mu^{(t+1)})$. Then $J_{t+1} \le J_t$ for all $t$, and with a tie-breaking rule that never changes a label without strict improvement, Lloyd's algorithm terminates after finitely many iterations at a partition $C^\dagger$ that is a fixed point: each point is closest to the mean of its own cluster.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Chain the two exact minimisations:
$$J(C^{(t+1)},\mu^{(t+2)}) \;\le\; J(C^{(t+1)},\mu^{(t+1)}) \;\le\; J(C^{(t)},\mu^{(t+1)}).$$
The right inequality holds because $C^{(t+1)}$ minimises $J(\cdot,\mu^{(t+1)})$; the left because the means minimise $J(C^{(t+1)},\cdot)$. So $J_t$ is non-increasing. If the assignment changes at step $t+1$, some point moved to a strictly closer centre (by the tie rule), so the right inequality is strict, and $J_{t+1} \lt J_t$. Since $J_t = W(C^{(t)})$ is a function of the partition alone, a strictly decreasing sequence of values of $W$ can never revisit a partition. There are at most $K^n$ partitions, so the algorithm stops; at the stopping point the assignment step reproduces $C^\dagger$, which is exactly the fixed-point condition.</p>
</details>
<p>Finite is not fast in the worst case: Vattani (2011) constructed planar inputs where Lloyd takes $2^{\Omega(n)}$ iterations. In practice it converges in tens of iterations, and Arthur, Manthey &amp; Röglin (2011) showed polynomial <em>smoothed</em> complexity (after tiny random perturbations of the data).</p>
<div class="warn" data-name="Fixed point is not a local minimum in every sense">
<p>A Lloyd fixed point is a local minimum only with respect to the two block moves. It can still be improved by moving a single point: transferring $x$ from cluster $a$ to cluster $b$ changes $W$ by exactly
$$\Delta W = \frac{n_b}{n_b+1}\lVert x - \bar x_b\rVert^2 - \frac{n_a}{n_a-1}\lVert x-\bar x_a\rVert^2 ,$$
because moving a point also drags both means. Lloyd only compares $\lVert x-\bar x_b\rVert^2$ with $\lVert x-\bar x_a\rVert^2$, ignoring the size factors. <strong>Hartigan's method</strong> uses the exact $\Delta W$ and its fixed points are a strict subset of Lloyd's (Telgarsky &amp; Vattani 2010).</p>
</div>`,
        lab: 'kmeans-lloyd',
      },
      {
        title: 'Hardness, initialisation and $k$-means++',
        html: R`
<p>Minimising $W$ exactly is NP-hard even for $K=2$ when the dimension is part of the input (Dasgupta 2008; Aloise et al. 2009), and NP-hard in the plane when $K$ is part of the input (Mahajan, Nimbhorkar &amp; Varadarajan 2012). For fixed $K$ and $p$ it can be solved in polynomial time $O(n^{Kp+1})$ (Inaba, Katoh &amp; Imai 1994) by enumerating Voronoi partitions, which is useless beyond toy sizes. So practice is: a good heuristic, repeated from many starts, keeping the best $W$.</p>
<p>Random initialisation fails in a characteristic way: two seeds land in one true cluster and none in another, and Lloyd cannot repair this, because a centre can only move to the mean of points that are already closest to it. With $K$ well-separated equal clusters, the probability that uniform random seeds hit every cluster once is $K!/K^K \approx \sqrt{2\pi K}\,e^{-K}$, about $0.09\%$ at $K=9$.</p>
<div class="algo" data-name="k-means++ seeding (Arthur & Vassilvitskii 2007)">
<ol>
<li>Choose $\mu_1$ uniformly at random from the data.</li>
<li>For $j=2,\dots,K$: let $D(x)$ be the distance from $x$ to the nearest centre chosen so far, and choose $\mu_j = x_i$ with probability $D(x_i)^2 / \sum_{i'} D(x_{i'})^2$.</li>
<li>Run Lloyd from these seeds.</li>
</ol>
</div>
<div class="thm" data-kind="Theorem" data-name="k-means++ guarantee (Arthur & Vassilvitskii 2007)">
<p>Let $W_{\text{opt}}$ be the optimal objective. The seeding alone (before any Lloyd iterations) produces centres whose cost $W_{\text{seed}}$ satisfies
$$\E\,[W_{\text{seed}}] \le 8(\ln K + 2)\, W_{\text{opt}} .$$
Lloyd iterations can only lower the cost further.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>(i) If the first centre is drawn uniformly from an optimal cluster $A$, the expected cost of $A$ is exactly $2\,W_{\text{opt}}(A)$ (the pairwise identity above: the mean squared distance to a random member is twice the scatter about the mean). (ii) If a centre is drawn by $D^2$-sampling and happens to land in a not-yet-covered cluster $A$, then, conditionally, $A$'s expected cost is at most $8\,W_{\text{opt}}(A)$ — a triangle-inequality argument shows $D^2$ sampling within $A$ is no worse than uniform sampling up to a factor $4$. (iii) The remaining work is to control clusters that are never "covered"; an induction on the number of uncovered clusters and remaining picks shows their total contribution inflates the bound by a harmonic number $H_{K} \le 1+\ln K$, giving $8(\ln K+2)$. The bound is tight up to constants: there are inputs where $D^2$ seeding is $\Omega(\log K)$ from optimal in expectation.</p>
</details>
<p>The intuition behind $D^2$ weighting: a point far from all current centres is likely to lie in an uncovered cluster, so it should be likely to become a seed — but squaring distances (rather than taking the farthest point) avoids systematically seeding on outliers.</p>`,
        lab: 'kmeans-init',
      },
      {
        title: 'k-means as a statistical model',
        html: R`
<p>What population quantity does k-means estimate, and what assumptions is it implicitly making? Two complementary answers.</p>
<p><strong>The population objective.</strong> For $X\sim P$ with $\E\lVert X\rVert^2 \lt \infty$, define $\Phi(\mu_1,\dots,\mu_K) = \E \min_k\lVert X-\mu_k\rVert^2$. Sample k-means minimises the empirical version $\Phi_n$. Pollard (1981) proved <em>strong consistency</em>: if $\Phi$ has a unique minimiser (up to relabelling), the optimal empirical centres converge to it almost surely; Pollard (1982) gave a $\sqrt n$ central limit theorem under further regularity. Note the target is the optimal quantiser of $P$, which need not have anything to do with "true clusters" — for a single Gaussian blob the population optimum still has $K$ distinct centres.</p>
<p><strong>The generative reading.</strong> k-means is the limit of a likelihood method for a specific mixture.</p>
<div class="thm" data-kind="Proposition" data-name="k-means is hard EM for an isotropic Gaussian mixture">
<p>Consider the mixture $p(x) = \sum_k \pi_k\,\N(x\mid \mu_k, \sigma^2 I)$ with fixed $\pi_k \gt 0$ and common known $\sigma^2$. As $\sigma^2\to 0$, the EM responsibilities $r_{ik}$ converge to the nearest-centre indicator $\ind\{k = \argmin_j \lVert x_i-\mu_j\rVert\}$ (for points without ties), the M-step for $\mu_k$ becomes the Lloyd update, and $-2\sigma^2$ times the expected complete-data log-likelihood converges to $J(C,\mu)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The responsibility is
$$r_{ik} = \frac{\pi_k \exp(-\lVert x_i-\mu_k\rVert^2/2\sigma^2)}{\sum_j \pi_j\exp(-\lVert x_i-\mu_j\rVert^2/2\sigma^2)} = \Big[1 + \sum_{j\neq k}\tfrac{\pi_j}{\pi_k}\exp\!\big(-(\lVert x_i-\mu_j\rVert^2-\lVert x_i-\mu_k\rVert^2)/2\sigma^2\big)\Big]^{-1}.$$
If $\mu_k$ is the unique nearest centre, every exponent tends to $-\infty$ and $r_{ik}\to 1$; otherwise some term diverges and $r_{ik}\to 0$. The fixed weights $\pi_j/\pi_k$ do not matter in the limit. The M-step $\mu_k = \sum_i r_{ik}x_i/\sum_i r_{ik}$ then becomes the cluster mean. Finally $\sum_{i,k} r_{ik}\log[\pi_k \N(x_i\mid\mu_k,\sigma^2 I)] = -\frac{1}{2\sigma^2}\sum_{i,k} r_{ik}\lVert x_i-\mu_k\rVert^2 + O(\log\sigma^2)$, and multiplying by $-2\sigma^2$ leaves $J$ in the limit.</p>
</details>
<p>This tells you which shapes k-means "believes in": spherical clusters with a common spread. Because the mixing weights drop out of the hard-assignment limit, k-means also places boundaries at perpendicular bisectors regardless of cluster sizes, which systematically steals points from a large diffuse cluster and gives them to a small tight neighbour. The full <a href="#/t/gaussian-mixtures">Gaussian mixture</a> fitted by <a href="#/t/em-algorithm">EM</a> relaxes all of these assumptions.</p>
<div class="ex" data-name="Unequal sizes">
<p>Take 900 points from $\N((0,0),I)$ and 100 from $\N((3,0),0.1\,I)$. The Bayes boundary between the two components sits close to the small cluster, because the large one has both more mass and more spread. k-means puts its boundary near the perpendicular bisector of the two means, at roughly $x_1\approx 1.5$, and so assigns the right tail of the big cluster (about $7\%$ of it, since $\P(Z \gt 1.5)\approx 0.067$) to the small one. Try the "unequal" dataset in the first lab.</p>
</div>`,
      },
      {
        title: 'Choosing $K$ and other practical matters',
        html: R`
<p>The optimal $W_K$ is non-increasing in $K$ (split any cluster to do at least as well) and reaches $0$ at $K=n$, so $W$ itself cannot pick $K$. Common devices:</p>
<ul>
<li><strong>Elbow:</strong> plot $W_K$ against $K$ and look for a kink. Informal and often ambiguous.</li>
<li><strong>Gap statistic</strong> (Tibshirani, Walther &amp; Hastie 2001): compare $\log W_K$ with its expectation under a reference null distribution (uniform over the data's bounding box or principal-axis box), estimated by simulation; choose the smallest $K$ with $\text{Gap}(K)\ge \text{Gap}(K+1) - s_{K+1}$.</li>
<li><strong>Silhouette</strong> (Rousseeuw 1987): for each point, $s_i = (b_i - a_i)/\max(a_i,b_i)$ where $a_i$ is its mean distance to its own cluster and $b_i$ the smallest mean distance to another cluster; average over $i$ and maximise over $K$.</li>
<li><strong>Model-based:</strong> fit a Gaussian mixture and use BIC (<a href="#/t/gaussian-mixtures">next topic</a>); or stability under resampling (Ben-Hur et al. 2002), with the caveat that stability can favour wrong $K$ for symmetric data.</li>
</ul>
<div class="warn" data-name="Scale and invariance">
<p>k-means is invariant to translations and rotations but <em>not</em> to rescaling individual coordinates: measure income in thousands instead of dollars and the clustering changes. Standardise deliberately, or better, think about which metric is meaningful. A Mahalanobis metric with a shared covariance $\Sigma$ is equivalent to running k-means on $\Sigma^{-1/2}x$.</p>
</div>
<div class="cols">
<div><h4>k-medoids (PAM)</h4><p>Centres restricted to data points; works with any dissimilarity; robust to outliers. Cost $O(K(n-K)^2)$ per swap pass.</p></div>
<div><h4>Kernel k-means</h4><p>Run k-means in a feature space via the kernel trick; the pairwise form of $W$ needs only inner products. Closely related to <a href="#/t/spectral-density-clustering">spectral clustering</a> (Dhillon, Guan &amp; Kulis 2004).</p></div>
<div><h4>Mini-batch k-means</h4><p>Stochastic-approximation updates on small batches (Sculley 2010); the standard choice for $n$ in the millions.</p></div>
</div>`,
      },
      {
        title: 'Where k-means breaks',
        html: R`
<p>Every failure follows from the objective, not from the algorithm. If $W$ prefers a different partition than the one you want, no amount of restarts will help.</p>
<div class="warn" data-name="Non-convex and nested clusters">
<p>Voronoi cells are convex, so k-means can never return two interleaved half-moons or concentric rings as separate clusters. On two concentric circles with $K=2$ it cuts the plane in half through the centre. Graph-based methods (<a href="#/t/spectral-density-clustering">spectral clustering, DBSCAN</a>) or <a href="#/t/hierarchical-clustering">single linkage</a> are needed.</p>
</div>
<div class="warn" data-name="Anisotropy and unequal spread">
<p>Elongated clusters are split across their long axis whenever that lowers squared error, and boundaries ignore cluster variance and size. A Gaussian mixture with full covariances handles both; k-means on whitened data handles a <em>shared</em> anisotropy.</p>
</div>
<div class="warn" data-name="High dimensions">
<p>When $p$ is large and most coordinates are noise, pairwise distances concentrate and the noise coordinates dominate $\lVert x_i-\mu_k\rVert^2$. Reduce dimension first (e.g. <a href="#/t/pca">PCA</a>), or use sparse k-means (Witten &amp; Tibshirani 2010), which learns feature weights.</p>
</div>
<p>Finally, k-means always returns $K$ clusters, including on data with no cluster structure at all. Whether the clusters are "real" is a separate inferential question (e.g. the gap statistic, or the selective-inference tests of Gao, Bien &amp; Witten 2022, which correct for the fact that the clusters were chosen from the same data used to test them).</p>`,
      },
    ],
    quiz: [
      { q: R`Why does Lloyd's algorithm terminate in finitely many steps, and why does this argument say nothing about how many steps?`,
        a: R`Each iteration does an exact block minimisation of $J$, so the objective is non-increasing, and strictly decreasing whenever the partition changes (with a sensible tie rule). Because the objective at a Lloyd step is a function of the partition, no partition can be revisited, and there are at most $K^n$ partitions. The bound $K^n$ is astronomically loose, and in fact some inputs force $2^{\Omega(n)}$ iterations; the finiteness argument is a counting argument, not a rate.` },
      { q: R`Give the exact change in $W$ when a point $x$ moves from cluster $a$ (size $n_a$) to cluster $b$ (size $n_b$). What does Lloyd's rule ignore?`,
        a: R`$\Delta W = \frac{n_b}{n_b+1}\lVert x-\bar x_b\rVert^2 - \frac{n_a}{n_a-1}\lVert x-\bar x_a\rVert^2$. Lloyd compares the raw squared distances, ignoring that moving $x$ also shifts both means. A Lloyd fixed point can therefore still be improved by a single-point move; Hartigan's method exploits this.` },
      { q: R`In what sense is k-means a special case of fitting a Gaussian mixture, and what does this imply about cluster shapes and sizes?`,
        a: R`It is the $\sigma^2\to0$ limit of EM for $\sum_k\pi_k\N(\mu_k,\sigma^2 I)$: responsibilities become hard nearest-centre indicators and the M-step becomes the cluster mean. It therefore assumes spherical clusters with a common spread, and since $\pi_k$ drops out of the hard limit, its boundaries are perpendicular bisectors that ignore cluster size and variance.` },
      { q: R`With $K=9$ well-separated equal-size clusters, roughly how often does uniform random seeding place exactly one seed in each cluster? What does $k$-means++ guarantee instead?`,
        a: R`$9!/9^9 \approx 0.00094$, i.e. about once in a thousand starts. k-means++ ($D^2$ seeding) guarantees $\E W_{\text{seed}} \le 8(\ln K+2)\,W_{\text{opt}}$; on well-separated data it covers all clusters with high probability because points in uncovered clusters carry almost all the $D^2$ mass.` },
    ],
  });

  /* ===================================================================== GMM */
  Atlas.topic({
    id: 'gaussian-mixtures',
    domain: 'unsupervised',
    order: 20,
    title: 'Gaussian Mixture Models',
    subtitle: 'Soft clustering and flexible density estimation with a likelihood that is unbounded, multimodal and still useful',
    level: 2,
    sources: { aos: '—', cb: '—', other: 'Bishop PRML §9.2; ESL §6.8, §14.3.7; Murphy (2012) Ch. 11; McLachlan & Peel (2000) Finite Mixture Models' },
    prereqs: ['multivariate', 'mle'],
    related: ['em-algorithm', 'kmeans', 'model-selection', 'kernel-density', 'hidden-markov'],
    tags: ['mixture model', 'latent variable', 'soft clustering', 'BIC', 'identifiability'],
    summary: R`A Gaussian mixture models data as coming from one of $K$ Gaussian sub-populations chosen at random with probabilities $\pi_k$. It is simultaneously a probabilistic clustering method (each point gets posterior membership probabilities rather than a hard label) and a universal density estimator. Its likelihood is a textbook case of irregularity: invariant under the $K!$ relabellings of components, unbounded because a component can collapse onto a single point, and without closed-form maximisers. Yet EM fitting, a few safeguards against degeneracy, and BIC for choosing $K$ make it one of the most widely used models in statistics.`,
    survey: {
      task: 'Soft clustering; density estimation; modelling heterogeneous populations',
      assumptions: 'Data from a finite mixture of Gaussians (or well approximated by one); $K$ chosen by a criterion',
      output: R`$\hat\pi_k,\hat\mu_k,\hat\Sigma_k$; responsibilities $r_{ik}=\hat\P(Z_i=k\mid x_i)$; a density`,
      strengths: 'Elliptical clusters of any orientation and size; calibrated memberships; likelihood-based model choice',
      weaknesses: 'Unbounded likelihood; local optima; $O(Kp^2)$ parameters; Gaussian tails; label switching',
      complexity: '$O(nKp^2 + Kp^3)$ per EM iteration (full covariances)',
    },
    sections: [
      {
        title: 'The model',
        html: R`
<div class="def" data-name="Finite Gaussian mixture">
<p>A random vector $X\in\R^p$ follows a $K$-component Gaussian mixture if
$$p(x\mid\theta) = \sum_{k=1}^K \pi_k\,\N(x\mid\mu_k,\Sigma_k),\qquad \pi_k \ge 0,\ \sum_k\pi_k = 1,$$
with $\theta = (\pi_1,\dots,\pi_K,\mu_1,\dots,\mu_K,\Sigma_1,\dots,\Sigma_K)$. Equivalently, there is a latent label $Z\sim\text{Categorical}(\pi)$ and $X\mid Z=k\sim\N(\mu_k,\Sigma_k)$.</p>
</div>
<p>The latent-variable form is the useful one. It makes sampling trivial (draw $Z$, then $X$), it gives cluster memberships by Bayes' theorem,
$$r_k(x) = \P(Z=k\mid X=x) = \frac{\pi_k\,\N(x\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\,\N(x\mid\mu_j,\Sigma_j)},$$
and it turns fitting into a missing-data problem that the <a href="#/t/em-algorithm">EM algorithm</a> solves. The <em>responsibility</em> $r_k(x)$ is a soft assignment; the induced classifier $\argmax_k r_k(x)$ is the plug-in Bayes rule of <a href="#/t/lda-qda">QDA</a> with unknown labels. Its decision boundaries are quadrics, and become linear when all $\Sigma_k$ are equal.</p>
<p>Mixtures are also a nonparametric density estimator in disguise: finite Gaussian mixtures are dense (in $L^1$ and in the weak topology) in the set of densities on $\R^p$, and a <a href="#/t/kernel-density">kernel density estimate</a> with Gaussian kernel is the extreme case $K=n$, $\mu_k = x_k$, $\Sigma_k = h^2 I$, $\pi_k = 1/n$.</p>
<table>
<thead><tr><th>Covariance structure</th><th>$\Sigma_k$</th><th>Free covariance parameters</th><th>Cluster shapes</th></tr></thead>
<tbody>
<tr><td>Spherical</td><td>$\sigma_k^2 I$</td><td>$K$</td><td>round, varying size</td></tr>
<tr><td>Diagonal</td><td>$\diag(\sigma^2_{k1},\dots,\sigma^2_{kp})$</td><td>$Kp$</td><td>axis-aligned ellipses</td></tr>
<tr><td>Tied (shared)</td><td>$\Sigma$</td><td>$p(p+1)/2$</td><td>same ellipse everywhere; linear boundaries</td></tr>
<tr><td>Full</td><td>$\Sigma_k$</td><td>$Kp(p+1)/2$</td><td>arbitrary ellipses</td></tr>
</tbody>
</table>
<p>Adding $K-1$ weights and $Kp$ means gives the total parameter count $d_K$ used by BIC below. Banfield &amp; Raftery (1993) parametrise $\Sigma_k = \lambda_k D_k A_k D_k^\top$ (volume, orientation, shape) and constrain each factor separately, giving the family of models in the <code>mclust</code> software.</p>`,
      },
      {
        title: 'The likelihood and its pathologies',
        html: R`
<p>For IID data the log-likelihood is
$$\ell(\theta) = \sum_{i=1}^n \log\Big(\sum_{k=1}^K \pi_k\,\N(x_i\mid\mu_k,\Sigma_k)\Big).$$
The log of a sum does not separate, so setting the gradient to zero gives coupled equations with no closed form. Those score equations are worth writing down, because EM's updates are exactly their fixed-point form: with $r_{ik} = r_k(x_i)$,
$$\frac{\partial\ell}{\partial\mu_k} = \sum_i r_{ik}\,\Sigma_k^{-1}(x_i-\mu_k) = 0 \iff \mu_k = \frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}},$$
and similarly for $\Sigma_k$ and $\pi_k$ (with a Lagrange multiplier for $\sum\pi_k=1$). The equations are implicit since $r_{ik}$ depends on $\theta$.</p>
<h4>1. Label switching and identifiability</h4>
<p>Permuting component labels leaves $p(x\mid\theta)$ unchanged, so every maximiser comes in $K!$ copies and the parameter is identified only up to permutation. That is the best one can hope for:</p>
<div class="thm" data-kind="Theorem" data-name="Identifiability of Gaussian mixtures (Teicher 1963; Yakowitz & Spragins 1968)">
<p>If $\sum_{k=1}^K\pi_k\N(\mu_k,\Sigma_k) = \sum_{k=1}^{K'}\pi'_k\N(\mu'_k,\Sigma'_k)$ as distributions, with all $\pi_k,\pi'_k \gt 0$ and the pairs $(\mu_k,\Sigma_k)$ distinct within each side, then $K=K'$ and the two sets of triples $(\pi_k,\mu_k,\Sigma_k)$ coincide up to a permutation.</p>
</div>
<details class="proof"><summary>Proof sketch (univariate)</summary>
<p>Compare moment generating functions: $\sum_k \pi_k e^{\mu_k t + \sigma_k^2 t^2/2}$. As $t\to\infty$ the term with the largest $\sigma_k$ (ties broken by largest $\mu_k$) dominates every other term exponentially, so both sides must share their dominant component with equal weight. Subtract it and induct on $K$. The multivariate case reduces to the univariate one by projecting onto a direction $a$ for which all the projected pairs $(a^\top\mu_k, a^\top\Sigma_k a)$ are distinct, which holds for all $a$ outside a finite union of proper algebraic subsets.</p>
</details>
<p>For frequentist point estimation label switching is harmless (report any one copy). It matters for Bayesian inference by MCMC, where a well-mixing sampler visits all $K!$ modes and component-specific posterior means average over labels into meaningless values; remedies are relabelling algorithms (Stephens 2000) or ordering constraints, the latter of which can distort the posterior.</p>
<h4>2. The likelihood is unbounded</h4>
<div class="thm" data-kind="Proposition" data-name="Degeneracy">
<p>Let $K\ge 2$ and suppose the data are not all identical. Then $\sup_\theta \ell(\theta) = +\infty$. In particular, a global maximum likelihood estimator does not exist.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Set $\mu_1 = x_1$, $\Sigma_1 = \varepsilon I$, $\pi_1 = 1/2$, and let component 2 be any fixed Gaussian with $\pi_2 = 1/2$ (e.g. $\mu_2 = \bar x$, $\Sigma_2 = I$). The first term of $\ell$ is $\log\big(\tfrac12(2\pi\varepsilon)^{-p/2} + \tfrac12\N(x_1\mid\mu_2,\Sigma_2)\big)\ge \log\tfrac12 - \tfrac p2\log(2\pi\varepsilon)\to+\infty$ as $\varepsilon\to 0$. Every other term is bounded below by $\log\big(\tfrac12\N(x_i\mid\mu_2,\Sigma_2)\big)$, a finite constant not depending on $\varepsilon$. So $\ell\to+\infty$.</p>
</details>
<p>Component 2 is what makes this work: it keeps the other points from contributing $-\infty$. With $K=1$ the Gaussian likelihood is bounded (as long as the data are not degenerate), which is why the singularity is special to mixtures. So "the MLE" for a mixture means a well-behaved <em>local</em> maximiser in the interior of the parameter space. Kiefer (1978) showed that for univariate normal mixtures a consistent, asymptotically efficient root of the likelihood equations exists; Hathaway (1985) showed that constraining $\min_{j,k}\sigma_j/\sigma_k \ge c \gt 0$ makes the constrained global MLE exist and be consistent.</p>
<div class="note" data-name="Remedies in practice">
<p>Add a small ridge to each covariance ($\Sigma_k \leftarrow \Sigma_k + \epsilon I$, the default in most software); bound eigenvalue ratios; use a MAP estimate with an inverse-Wishart prior, which adds pseudo-scatter to every $\Sigma_k$ and makes the posterior bounded; or tie covariances across components. When EM drifts toward a spike, the run is discarded or the component re-seeded.</p>
</div>
<h4>3. Multimodality and irregular asymptotics</h4>
<p>Apart from the $K!$ copies, the likelihood typically has genuinely distinct local maxima, so EM must be restarted from many seeds. And when the true number of components is smaller than $K$, the parameter is not identified (an extra component can have $\pi_k = 0$ with arbitrary $\mu_k$, or duplicate another component), the Fisher information is singular, and the usual quadratic expansion of the likelihood fails. Consequently the likelihood ratio statistic for $K$ versus $K+1$ components does <em>not</em> have the $\chi^2$ limit that <a href="#/t/likelihood-ratio-tests">Wilks' theorem</a> would predict (Hartigan 1985 showed it can even diverge in some formulations); calibrate it by a <a href="#/t/parametric-bootstrap">parametric bootstrap</a> instead (McLachlan 1987).</p>`,
      },
      {
        title: 'Fitting by EM',
        html: R`
<p>Treat the labels $z_i$ as missing. With them, the complete-data log-likelihood separates into a categorical part and $K$ independent Gaussian parts, each with a closed-form MLE. EM replaces the unknown indicators $\ind\{z_i=k\}$ by their conditional expectations $r_{ik}$ and solves the weighted problem. Derivations and the general convergence theory are in <a href="#/t/em-algorithm">the EM topic</a>; here is the algorithm.</p>
<div class="algo" data-name="EM for a Gaussian mixture">
<ol>
<li>Initialise $\theta^{(0)}$, e.g. from a k-means solution (means, within-cluster covariances, cluster proportions).</li>
<li><strong>E-step:</strong> $r_{ik} = \pi_k\N(x_i\mid\mu_k,\Sigma_k)\big/\sum_j\pi_j\N(x_i\mid\mu_j,\Sigma_j)$, computed in log space with log-sum-exp.</li>
<li><strong>M-step:</strong> with $N_k = \sum_i r_{ik}$,
$$\pi_k = \frac{N_k}{n},\qquad \mu_k = \frac1{N_k}\sum_i r_{ik}x_i,\qquad \Sigma_k = \frac1{N_k}\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top (+\,\epsilon I).$$</li>
<li>Evaluate $\ell(\theta)$; stop when the relative increase is below a tolerance. $\ell$ never decreases (absent the ridge).</li>
</ol>
</div>
<p>For the constrained families the M-step changes in the obvious way: diagonal models keep only the diagonal of the weighted scatter; tied models pool, $\Sigma = \frac1n\sum_k\sum_i r_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top$; spherical models use $\sigma_k^2 = \tr(\cdot)/(pN_k)$.</p>
<div class="ex" data-name="A one-step calculation">
<p>Univariate data $\{-1, 0, 4, 5\}$, $K=2$, start with $\mu_1 = 0$, $\mu_2 = 4$, $\sigma_1=\sigma_2=1$, $\pi_1=\pi_2=\tfrac12$. The responsibility of component 1 for $x=-1$ is $1/(1+e^{-(25-1)/2}) = 1/(1+e^{-12}) \approx 1-6\times10^{-6}$; for $x=0$ it is $1/(1+e^{-8})\approx 0.99966$; for $x=4$ it is $1/(1+e^{8})\approx 3.4\times10^{-4}$; for $x=5$, $1/(1+e^{12})\approx 6\times10^{-6}$. The M-step gives $\mu_1\approx -0.50$, $\mu_2\approx 4.50$, $\pi_1\approx\pi_2\approx 0.5$ and $\sigma_k^2 \approx 0.25$ each, already essentially the k-means answer because the clusters are well separated. With overlapping clusters the soft weights matter and EM can take hundreds of iterations.</p>
</div>`,
        lab: 'gaussian-mixtures-em',
      },
      {
        title: 'Choosing the number of components',
        html: R`
<p>Because $\ell$ increases with $K$, model choice needs a penalty. The <strong>Bayesian information criterion</strong> (Schwarz 1978) is the standard one for mixtures:
$$\text{BIC}(K) = -2\,\ell(\hat\theta_K) + d_K\log n,\qquad d_K = (K-1) + Kp + K\tfrac{p(p+1)}{2}\ \text{(full covariances)},$$
and one picks the $K$ that minimises it. (Some software, including <code>mclust</code>, reports $2\ell - d_K\log n$ and maximises; same thing.) BIC is $-2$ times a Laplace approximation to the log marginal likelihood $\log p(x\mid K)$, dropping terms that are $O(1)$.</p>
<div class="warn" data-name="The theory is shakier than for regular models">
<p>The Laplace approximation behind BIC assumes a regular model with a non-singular Fisher information, exactly what fails when $K$ exceeds the true number of components. Keribin (2000) nonetheless proved that penalised-likelihood estimators of $K$, including BIC, are consistent for the number of components under conditions including a bounded parameter space. In practice BIC works well for selecting $K$ <em>as a density estimator</em>; but if the true clusters are not Gaussian, BIC will happily use several Gaussian components to model one skewed or heavy-tailed cluster. Integrated completed likelihood, ICL (Biernacki, Celeux &amp; Govaert 2000), adds an entropy penalty $-2\sum_{i,k}r_{ik}\log r_{ik}$ that favours well-separated components and so targets "clusters" rather than "density".</p>
</div>
<p>AIC ($2d_K$ penalty) tends to overestimate $K$ for mixtures. Cross-validated log-likelihood is a robust, assumption-light alternative when $n$ is moderate.</p>`,
        lab: 'gaussian-mixtures-bic',
      },
      {
        title: 'Extensions and connections',
        html: R`
<div class="cols">
<div><h4>Bayesian mixtures</h4><p>Dirichlet prior on $\pi$, Normal–inverse-Wishart priors on components; fit by <a href="#/t/gibbs-sampling">Gibbs sampling</a> (data augmentation with the $z_i$) or <a href="#/t/variational-inference">variational Bayes</a>, which can switch off unneeded components. The Dirichlet process mixture lets $K$ grow with $n$.</p></div>
<div><h4>Mixtures of experts and HMMs</h4><p>Let $\pi$ depend on covariates (mixture of experts) or on the previous label (a <a href="#/t/hidden-markov">hidden Markov model</a>, whose EM is Baum–Welch). Mixtures of regressions model heterogeneous slopes.</p></div>
<div><h4>Heavier tails, high dimension</h4><p>Mixtures of multivariate $t$ distributions resist outliers. When $p$ is large, mixtures of <a href="#/t/factor-analysis-ica">factor analysers</a> or probabilistic PCA give each component a low-rank-plus-diagonal covariance.</p></div>
</div>
<p>A last caution on interpretation: a mixture component is not automatically a "type" in the population. A single skewed cluster is well fitted by two overlapping Gaussians; a clear bimodality in a fitted density is a stronger signal than the number of components chosen by BIC.</p>`,
      },
    ],
    quiz: [
      { q: R`Show that the GMM log-likelihood is unbounded above when $K\ge2$. Why does the argument fail for $K=1$?`,
        a: R`Centre one component at a data point $x_1$ with covariance $\varepsilon I$: its density at $x_1$ is $(2\pi\varepsilon)^{-p/2}\to\infty$, while a second fixed component keeps every other point's density bounded below. With $K=1$ there is no second component: shrinking $\Sigma$ to make $x_1$ likely sends every other point's density to $0$ and their log terms to $-\infty$ faster, so the single-Gaussian likelihood stays bounded (for non-degenerate data).` },
      { q: R`What does "identifiable up to label switching" mean, and when does label switching cause real trouble?`,
        a: R`Two parameter values give the same distribution iff they differ only by permuting components, so the likelihood has $K!$ symmetric copies of every mode. Point estimation is unaffected (pick any copy). It causes trouble for Bayesian MCMC: a sampler that mixes well jumps between the $K!$ modes, so marginal posteriors of component-specific parameters are mixtures of all components and their posterior means are meaningless without relabelling.` },
      { q: R`For a full-covariance GMM in $p=10$ dimensions with $K=4$, how many free parameters enter BIC? Why might BIC choose $K=4$ when there are only two real groups?`,
        a: R`$d = (K-1) + Kp + Kp(p+1)/2 = 3 + 40 + 4\cdot55 = 263$. BIC selects a good <em>density</em> approximation: if each real group is non-Gaussian (skewed, heavy-tailed, curved), several Gaussian components per group can improve the likelihood enough to pay the penalty. ICL or merging components is needed if "clusters" rather than "components" are the target.` },
      { q: R`Why is the likelihood ratio test of $K$ versus $K+1$ components not calibrated by a $\chi^2$ distribution?`,
        a: R`Under the null ($K$ components), the $(K+1)$-component model is not identified: the extra component can have weight $0$ (its mean and covariance arbitrary) or coincide with an existing one. The null lies on the boundary and in a region where the Fisher information is singular, so the regularity conditions behind Wilks' theorem fail. Use a parametric bootstrap of the LRT instead.` },
    ],
  });

  /* ===================================================================== EM */
  Atlas.topic({
    id: 'em-algorithm',
    domain: 'unsupervised',
    order: 30,
    title: 'The EM Algorithm',
    subtitle: 'Maximum likelihood with missing data by repeatedly maximising a lower bound that touches the likelihood',
    level: 2,
    sources: { aos: '9.13.4', cb: '7.2.4', other: 'Dempster, Laird & Rubin (1977); Wu (1983); McLachlan & Krishnan, The EM Algorithm and Extensions (2008); Bishop PRML §9.3–9.4; ESL §8.5; Koller & Friedman Ch. 19' },
    prereqs: ['mle', 'entropy-kl'],
    related: ['gaussian-mixtures', 'variational-inference', 'hidden-markov', 'factor-analysis-ica', 'fisher-information'],
    tags: ['latent variables', 'missing data', 'ELBO', 'MM algorithm', 'missing information'],
    summary: R`Many likelihoods are hard only because some variables are unobserved: cluster labels, censored lifetimes, hidden states, latent factors. If the "complete data" were seen, the MLE would be easy. EM alternates two steps: compute the expected complete-data log-likelihood under the current posterior of the missing data (E), then maximise it (M). The key identity $\ell(\theta) = \mathcal L(q,\theta) + \KL(q\,\Vert\, p(z\mid x,\theta))$ shows each E-step makes a lower bound tight and each M-step pushes the bound up, so the likelihood never decreases. The price is speed: convergence is linear at a rate equal to the fraction of information that is missing.`,
    survey: {
      task: 'Maximum likelihood (or MAP) with latent variables or missing data',
      assumptions: 'Complete-data MLE tractable; conditional law of missing data given observed computable',
      output: 'A stationary point of $\\ell(\\theta)$ (usually a local max); posterior of latent variables',
      strengths: 'Monotone; no step size; respects constraints automatically; simple closed-form steps',
      weaknesses: 'Local optima; linear (possibly very slow) convergence; no standard errors without extra work',
      complexity: 'Per iteration: one E-step + one complete-data MLE; iterations $\\propto 1/(1-\\text{missing fraction})$',
    },
    sections: [
      {
        title: 'Missing data and the complete-data trick',
        html: R`
<p>Let $x$ be the observed data and $z$ the unobserved part, with a joint model $p(x,z\mid\theta)$. The quantity to maximise is the observed-data, or <em>incomplete-data</em>, log-likelihood
$$\ell(\theta) = \log p(x\mid\theta) = \log\int p(x,z\mid\theta)\d z .$$
The integral (or sum) inside the log is what makes life hard. In contrast the <em>complete-data</em> log-likelihood $\log p(x,z\mid\theta)$ is often a nice <a href="#/t/exponential-families">exponential-family</a> expression whose maximiser is a closed-form function of sufficient statistics.</p>
<p>Examples of the pattern:</p>
<ul>
<li><strong>Mixtures:</strong> $z_i$ is the component label (<a href="#/t/gaussian-mixtures">Gaussian mixtures</a>).</li>
<li><strong>Censoring:</strong> $z_i$ is the unobserved exact lifetime beyond a censoring time.</li>
<li><strong>Hidden Markov models:</strong> $z_{1:T}$ is the hidden state path; EM is the Baum–Welch algorithm (<a href="#/t/hidden-markov">HMMs</a>).</li>
<li><strong>Factor analysis, PPCA:</strong> $z_i$ is the latent factor vector (<a href="#/t/factor-analysis-ica">factor analysis</a>).</li>
<li><strong>Random effects:</strong> $z$ is the vector of random effects in a <a href="#/t/mixed-models">mixed model</a>.</li>
<li><strong>Genuinely missing values</strong> in a multivariate normal sample, under missing at random.</li>
</ul>
<div class="note">
<p>EM's idea is: we cannot use $\log p(x,z\mid\theta)$ because we do not know $z$, so use our best current guess of it — not a single imputed value, but its whole conditional distribution given $x$ and the current $\theta$, and average the complete-data log-likelihood over it. Then pretend that average is the likelihood and maximise. The remarkable fact is that this cannot make the true likelihood worse.</p>
</div>`,
      },
      {
        title: 'Derivation: the evidence lower bound',
        html: R`
<p>Let $q(z)$ be any distribution over the missing data with $q \gt 0$ wherever $p(z\mid x,\theta) \gt 0$.</p>
<div class="thm" data-kind="Lemma" data-name="Decomposition of the log-likelihood">
<p>For every $q$ and $\theta$,
$$\ell(\theta) = \underbrace{\E_q\!\big[\log p(x,Z\mid\theta)\big] - \E_q\!\big[\log q(Z)\big]}_{\mathcal L(q,\theta)\ \text{(ELBO)}} \;+\; \KL\big(q\,\Vert\,p(\cdot\mid x,\theta)\big).$$
Since $\KL\ge 0$ with equality iff $q = p(\cdot\mid x,\theta)$ a.e., $\mathcal L(q,\theta)\le \ell(\theta)$ with equality iff $q$ is the exact posterior.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For every $z$, $\log p(x\mid\theta) = \log p(x,z\mid\theta) - \log p(z\mid x,\theta) = \big[\log p(x,z\mid\theta) - \log q(z)\big] + \big[\log q(z) - \log p(z\mid x,\theta)\big]$. The left side does not depend on $z$, so take $\E_q$ of both sides. The second bracket's expectation is $\KL(q\,\Vert\,p(\cdot\mid x,\theta))$, which is non-negative by Gibbs' inequality (Jensen applied to the convex function $-\log$: $-\KL = \E_q\log\frac{p}{q}\le\log\E_q\frac pq\le \log 1 = 0$).</p>
</details>
<p>The same bound can be reached in one line from Jensen directly: $\ell(\theta) = \log\E_q\big[\tfrac{p(x,Z\mid\theta)}{q(Z)}\big]\ge\E_q\log\tfrac{p(x,Z\mid\theta)}{q(Z)} = \mathcal L(q,\theta)$. The decomposition is better because it names the gap.</p>
<p>EM is coordinate ascent on $\mathcal L(q,\theta)$ (Neal &amp; Hinton 1998):</p>
<div class="algo" data-name="EM">
<ol>
<li><strong>E-step:</strong> $q^{(t+1)} = \argmax_q \mathcal L(q,\theta^{(t)}) = p(z\mid x,\theta^{(t)})$. The gap closes: $\mathcal L(q^{(t+1)},\theta^{(t)}) = \ell(\theta^{(t)})$. In practice one computes only what is needed for
$$Q(\theta\mid\theta^{(t)}) = \E\big[\log p(x,Z\mid\theta)\,\big|\,x,\theta^{(t)}\big],$$
typically the conditional expectations of complete-data sufficient statistics.</li>
<li><strong>M-step:</strong> $\theta^{(t+1)} = \argmax_\theta \mathcal L(q^{(t+1)},\theta) = \argmax_\theta Q(\theta\mid\theta^{(t)})$ (the entropy term $-\E_q\log q$ does not involve $\theta$).</li>
</ol>
</div>
<p>Viewed as a function of $\theta$, the bound $\theta\mapsto\mathcal L(q^{(t+1)},\theta) = Q(\theta\mid\theta^{(t)}) + \text{const}$ is a <em>minorant</em> of $\ell$: it lies everywhere below $\ell$ and touches it at $\theta^{(t)}$. EM is therefore a special case of the minorise–maximise (MM) principle (Lange, Hunter &amp; Yang 2000). The lab draws this picture.</p>`,
        lab: 'em-algorithm-bound',
      },
      {
        title: 'Monotonicity and convergence',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Ascent property (Dempster, Laird & Rubin 1977)">
<p>Every EM iteration satisfies $\ell(\theta^{(t+1)})\ge\ell(\theta^{(t)})$, with equality only if $Q(\theta^{(t+1)}\mid\theta^{(t)}) = Q(\theta^{(t)}\mid\theta^{(t)})$ and $p(z\mid x,\theta^{(t+1)}) = p(z\mid x,\theta^{(t)})$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Using the lemma twice,
$$\ell(\theta^{(t+1)}) \;\ge\; \mathcal L(q^{(t+1)},\theta^{(t+1)}) \;\ge\; \mathcal L(q^{(t+1)},\theta^{(t)}) \;=\; \ell(\theta^{(t)}).$$
The first inequality is the lower bound (it is strict unless $q^{(t+1)}$ is also the posterior at $\theta^{(t+1)}$); the second is the M-step; the equality is the E-step. An equivalent classical form: write $\ell(\theta) = Q(\theta\mid\theta^{(t)}) - H(\theta\mid\theta^{(t)})$ with $H(\theta\mid\theta^{(t)}) = \E[\log p(Z\mid x,\theta)\mid x,\theta^{(t)}]$. Gibbs' inequality gives $H(\theta\mid\theta^{(t)})\le H(\theta^{(t)}\mid\theta^{(t)})$ for all $\theta$, so any increase of $Q$ is an increase of $\ell$.</p>
</details>
<p>The proof uses only that the M-step <em>increases</em> $Q$, not that it maximises it. Any algorithm with $Q(\theta^{(t+1)}\mid\theta^{(t)})\ge Q(\theta^{(t)}\mid\theta^{(t)})$ is a <strong>generalised EM</strong> (GEM) and keeps the ascent property — for instance a single Newton or gradient step in the M-step, or the <strong>ECM</strong> algorithm (Meng &amp; Rubin 1993), which replaces a hard joint maximisation by a sequence of conditional maximisations over blocks of $\theta$.</p>
<p>Monotone and bounded sequences converge, but convergence of $\ell(\theta^{(t)})$ is not convergence of $\theta^{(t)}$, and neither implies reaching a maximum.</p>
<div class="thm" data-kind="Theorem" data-name="Convergence to stationary points (Wu 1983)">
<p>Suppose $\ell$ is continuous, the level set $\{\theta:\ell(\theta)\ge\ell(\theta^{(0)})\}$ is compact, $Q(\theta\mid\theta')$ is continuous in both arguments, and $\theta^{(t)}$ lies in the interior of the parameter space. Then every limit point of an EM sequence is a stationary point of $\ell$, and $\ell(\theta^{(t)})$ converges monotonically to $\ell(\theta^\star)$ for some stationary $\theta^\star$. If in addition stationary points are isolated and $\lVert\theta^{(t+1)}-\theta^{(t)}\rVert\to0$, then $\theta^{(t)}$ itself converges.</p>
</div>
<p>Stationary points include local maxima and saddle points (EM rarely gets stuck at a saddle in practice since a random perturbation escapes it, but it can stall there for a long time). The compactness hypothesis is not decorative: for <a href="#/t/gaussian-mixtures">Gaussian mixtures</a> the level sets are not compact because $\ell$ is unbounded, and EM sequences really can run off to a degenerate solution with a collapsing covariance.</p>`,
      },
      {
        title: 'Rate of convergence: the missing-information principle',
        html: R`
<p>Near a local maximum $\theta^\star$, EM is a fixed-point iteration $\theta^{(t+1)} = M(\theta^{(t)})$ with $M(\theta^\star)=\theta^\star$. Its speed is governed by the Jacobian $DM(\theta^\star)$. Three information matrices control everything. Evaluated at $\theta^\star$:</p>
<ul>
<li>observed information $I_{\text{obs}} = -\nabla^2\ell(\theta^\star)$;</li>
<li>complete information $I_c = \E\big[-\nabla^2\log p(x,Z\mid\theta^\star)\,\big|\,x,\theta^\star\big] = -\nabla^2_{\theta}Q(\theta\mid\theta^\star)\big|_{\theta^\star}$;</li>
<li>missing information $I_m = \E\big[-\nabla^2\log p(Z\mid x,\theta^\star)\,\big|\,x,\theta^\star\big] = \Var\big(\nabla\log p(Z\mid x,\theta^\star)\,\big|\,x,\theta^\star\big)$.</li>
</ul>
<p>Since $\log p(x,z) = \log p(x) + \log p(z\mid x)$, taking $-\nabla^2$ and the posterior expectation gives the <strong>missing-information principle</strong> (Orchard &amp; Woodbury 1972; Louis 1982):
$$I_{\text{obs}} = I_c - I_m .$$
"Observed information equals complete information minus what the missing data would have told us."</p>
<div class="thm" data-kind="Theorem" data-name="Linear rate of EM (Dempster, Laird & Rubin 1977)">
<p>Under smoothness, $DM(\theta^\star) = I_c^{-1}I_m = I - I_c^{-1}I_{\text{obs}}$, so
$$\theta^{(t+1)} - \theta^\star \approx I_c^{-1}I_m\,(\theta^{(t)}-\theta^\star),$$
and the asymptotic rate of convergence is the largest eigenvalue of $I_c^{-1}I_m$, the largest <strong>fraction of missing information</strong> in any direction. It lies in $[0,1)$ at a strict local maximum.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>Write $Q^{10}=\nabla_\theta Q(\theta\mid\theta')$ and similarly $Q^{20}, Q^{11}$. The M-step is $Q^{10}(\theta^{(t+1)}\mid\theta^{(t)})=0$, and $Q^{10}(\theta^\star\mid\theta^\star)=0$. Expanding around $(\theta^\star,\theta^\star)$:
$$0 \approx Q^{20}(\theta^\star\mid\theta^\star)(\theta^{(t+1)}-\theta^\star) + Q^{11}(\theta^\star\mid\theta^\star)(\theta^{(t)}-\theta^\star).$$
By definition $Q^{20} = -I_c$. For $Q^{11}$, differentiate $Q(\theta\mid\theta') = \int \log p(x,z\mid\theta)\,p(z\mid x,\theta')\d z$ in $\theta'$: $Q^{11} = \E[s_c(\theta)\,s_m(\theta')^\top\mid x,\theta']$, where $s_c=\nabla\log p(x,z\mid\theta)$ and $s_m = \nabla\log p(z\mid x,\theta')$. At $\theta=\theta'=\theta^\star$, $s_c = s_{\text{obs}} + s_m$ with $s_{\text{obs}}$ non-random given $x$, and $\E[s_m\mid x]=0$, so $Q^{11} = \E[s_m s_m^\top\mid x] = I_m$. Hence $\theta^{(t+1)}-\theta^\star\approx I_c^{-1}I_m(\theta^{(t)}-\theta^\star)$. Since $I_c^{-1}I_m = I - I_c^{-1}I_{\text{obs}}$ and $I_{\text{obs}}$ is positive definite at a strict maximum, the eigenvalues (real, as $I_c^{-1}I_m$ is similar to the symmetric $I_c^{-1/2}I_m I_c^{-1/2}$) lie in $[0,1)$.</p>
</details>
<div class="ex" data-name="Censored exponential: rate = censoring fraction">
<p>Let lifetimes be $\text{Exp}(\lambda)$, with $d$ failures observed exactly and $m$ observations right-censored at times $c_j$; $n = d+m$, and let $T$ be the total observed time (failure times plus censoring times). The observed-data MLE is $\hat\lambda = d/T$. Complete data would be the true lifetimes, with MLE $n/\sum(\text{lifetimes})$. By memorylessness, $\E[Y\mid Y \gt c,\lambda^{(t)}] = c + 1/\lambda^{(t)}$, so
$$\lambda^{(t+1)} = \frac{n}{T + m/\lambda^{(t)}} .$$
The fixed point solves $\lambda T + m = n$, i.e. $\lambda^\star = d/T$, as it should. Differentiating, $M'(\lambda^\star) = \frac{nm/\lambda^{\star2}}{(T+m/\lambda^\star)^2} = \frac{m}{n}$ since $T+m/\lambda^\star = n/\lambda^\star$. The rate is exactly the censoring fraction. Information check: $I_c = n/\lambda^2$, $I_{\text{obs}} = d/\lambda^2$, $I_m = m/\lambda^2$, and $I_c^{-1}I_m = m/n$. With $90\%$ censoring, each iteration removes only $10\%$ of the remaining error.</p>
</div>
<p>Two practical consequences. First, stopping when $\ell$ barely changes can be premature when the rate is close to 1; monitor the parameter change too, or use acceleration (Aitken, SQUAREM, quasi-Newton EM). Second, EM does not give standard errors for free: compute $I_{\text{obs}}$ via Louis' formula $I_{\text{obs}} = I_c - I_m$, or via the SEM algorithm (Meng &amp; Rubin 1991), which estimates $DM$ numerically from the EM sequence and uses $I_{\text{obs}}^{-1} = I_c^{-1}(I - DM)^{-1}$.</p>`,
      },
      {
        title: 'Worked example: EM for a mixture with an unknown mean',
        html: R`
<p>Take the simplest mixture with one free parameter: $X\sim\tfrac12\N(0,1)+\tfrac12\N(\mu,1)$, $\mu$ unknown. Let $z_i = 1$ if $x_i$ came from the second component.</p>
<p><strong>Complete-data log-likelihood:</strong> $\log p(x,z\mid\mu) = \sum_i\big[z_i\log\phi(x_i-\mu) + (1-z_i)\log\phi(x_i)\big] + n\log\tfrac12$.</p>
<p><strong>E-step:</strong> the only needed expectation is $r_i = \P(z_i=1\mid x_i,\mu^{(t)}) = \frac{\phi(x_i-\mu^{(t)})}{\phi(x_i)+\phi(x_i-\mu^{(t)})}$, so
$$Q(\mu\mid\mu^{(t)}) = -\tfrac12\sum_i r_i(x_i-\mu)^2 + \text{const}.$$</p>
<p><strong>M-step:</strong> a weighted mean, $\mu^{(t+1)} = \sum_i r_ix_i\big/\sum_i r_i$.</p>
<p><strong>The bound.</strong> Adding the entropy of the Bernoulli$(r_i)$ posteriors, $\mathcal L(q^{(t+1)},\mu) = Q(\mu\mid\mu^{(t)}) + \sum_i H(r_i)$ is a downward parabola in $\mu$ that touches $\ell(\mu)$ at $\mu^{(t)}$. Each M-step jumps to the vertex of the parabola.</p>
<p><strong>Rate.</strong> When the components are well separated the posteriors $r_i$ are nearly 0 or 1, the missing information is tiny, and EM converges in a few steps. As $\mu\to0$ the two components overlap, $r_i\approx\tfrac12$ for most points, almost all information about $\mu$ is "missing", and the rate approaches 1. This is the regime in which EM is painfully slow and in which the likelihood surface itself is flat — the two difficulties are the same phenomenon.</p>
<div class="warn" data-name="Local maxima">
<p>Even this one-parameter likelihood can be bimodal (try a small sample with the true $\mu$ moderate and start $\mu^{(0)}$ on the negative side in the lab). EM climbs to whichever mode's basin contains the start. Multiple starts, or deterministic annealing (raising the E-step posterior to a power $\beta \lt 1$ and increasing $\beta$ to 1), are the standard defences.</p>
</div>`,
      },
      {
        title: 'Variants and relatives',
        html: R`
<table>
<thead><tr><th>Variant</th><th>What changes</th><th>Use when</th></tr></thead>
<tbody>
<tr><td>GEM / ECM / ECME</td><td>M-step only increases $Q$, or maximises in blocks</td><td>No closed-form joint M-step</td></tr>
<tr><td>Monte Carlo EM, stochastic EM</td><td>E-step expectation approximated by sampling $z$</td><td>Posterior of $z$ intractable but samplable (e.g. by <a href="#/t/mcmc-foundations">MCMC</a>)</td></tr>
<tr><td>Variational EM</td><td>$q$ restricted to a tractable family; maximises $\mathcal L$, a true lower bound</td><td>Posterior intractable (<a href="#/t/variational-inference">variational inference</a>)</td></tr>
<tr><td>Hard / classification EM</td><td>$q$ is a point mass at the most likely $z$</td><td>Speed; k-means is the isotropic-Gaussian case</td></tr>
<tr><td>Online / incremental EM</td><td>Update sufficient statistics one point or batch at a time</td><td>Streaming or very large $n$ (Neal &amp; Hinton 1998; Cappé &amp; Moulines 2009)</td></tr>
<tr><td>MAP-EM</td><td>Add $\log p(\theta)$ to $Q$</td><td>Regularisation; avoids GMM singularities</td></tr>
</tbody>
</table>
<p>EM versus direct optimisation: Newton's method on $\ell$ converges quadratically but needs $\nabla^2\ell$, ignores constraints and can diverge. EM is stable, automatically respects constraints (probabilities stay in the simplex, covariances stay positive definite) and has cheap iterations, but only linear convergence. Hybrid schemes start with EM and finish with Newton or quasi-Newton.</p>
<p>The EM viewpoint — maximise a lower bound by alternating over a distribution on latent variables and the parameters — is the template for <a href="#/t/variational-inference">variational inference</a> and for training <a href="#/t/autoencoders-vae">variational autoencoders</a>, where the E-step is replaced by an amortised encoder network and both steps become stochastic gradient steps on the same ELBO.</p>`,
      },
    ],
    quiz: [
      { q: R`State the identity that decomposes $\ell(\theta)$ into the ELBO and a KL term, and explain how it yields the monotonicity of EM in three inequalities.`,
        a: R`$\ell(\theta) = \mathcal L(q,\theta) + \KL(q\Vert p(z\mid x,\theta))$ with $\mathcal L = \E_q\log p(x,Z\mid\theta) - \E_q\log q$. Then $\ell(\theta^{(t+1)})\ge\mathcal L(q^{(t+1)},\theta^{(t+1)})\ge\mathcal L(q^{(t+1)},\theta^{(t)}) = \ell(\theta^{(t)})$: lower bound (KL $\ge0$), M-step, and tightness of the bound after the E-step (KL $=0$ when $q$ is the exact posterior).` },
      { q: R`For right-censored exponential data with $m$ of $n$ observations censored, what is the EM convergence rate, and why does this make sense?`,
        a: R`Exactly $m/n$. The EM map is $\lambda\mapsto n/(T+m/\lambda)$ with derivative $m/n$ at the fixed point; equivalently $I_c^{-1}I_m = (m/\lambda^2)/(n/\lambda^2)$. The rate is the fraction of information that is missing: with heavy censoring most of what complete data would reveal about $\lambda$ is unobserved, and EM moves slowly.` },
      { q: R`What is a generalised EM algorithm and why does it still have the ascent property?`,
        a: R`One whose M-step merely increases $Q$, $Q(\theta^{(t+1)}\mid\theta^{(t)})\ge Q(\theta^{(t)}\mid\theta^{(t)})$, rather than maximising it. The ascent proof only uses the inequality $\mathcal L(q^{(t+1)},\theta^{(t+1)})\ge\mathcal L(q^{(t+1)},\theta^{(t)})$, which is exactly an increase in $Q$ (the entropy term does not depend on $\theta$).` },
      { q: R`Your EM log-likelihood changes by less than $10^{-8}$ per iteration. Can you conclude you are near a maximum? Name two reasons why not.`,
        a: R`No. (1) When the missing-information fraction is near 1, steps are tiny but the remaining distance to $\theta^\star$ can be large: the error shrinks by a factor close to 1 per step, so a small change per step can coexist with large remaining error. (2) EM converges to stationary points, which can be saddle points or poor local maxima, and in mixtures it can drift toward a degenerate (unbounded-likelihood) solution. Check parameter changes, use multiple starts, and inspect the Hessian.` },
      { q: R`Why does the M-step for a Gaussian mixture never produce a negative-definite covariance or mixing weights outside the simplex, while a Newton step on $\ell$ might?`,
        a: R`The M-step solves a weighted complete-data MLE: weights $N_k/n$ are non-negative and sum to one, and a weighted scatter matrix with non-negative weights is positive semi-definite. EM moves within the parameter space by construction; Newton's method takes an unconstrained quadratic step that can leave it.` },
    ],
  });
  /* ===================================================================== Hierarchical */
  Atlas.topic({
    id: 'hierarchical-clustering',
    domain: 'unsupervised',
    order: 40,
    title: 'Hierarchical Clustering',
    subtitle: 'Nested partitions at every scale, dendrograms as ultrametrics, and single linkage as a minimum spanning tree',
    level: 2,
    sources: { aos: '—', cb: '—', other: 'ESL §14.3.12; Murphy (2012) Ch. 25; Jain & Dubes (1988); Gower & Ross (1969); Kleinberg (2002)' },
    prereqs: ['kmeans'],
    related: ['spectral-density-clustering', 'kernel-density', 'manifold-learning'],
    tags: ['agglomerative', 'linkage', 'dendrogram', 'ultrametric', 'MST'],
    summary: R`Instead of committing to one $K$, hierarchical clustering builds a whole tree of nested partitions: start with every point alone and repeatedly merge the two closest clusters. The definition of "closest" — the linkage — is the entire method: single linkage is the minimum spanning tree and finds elongated, chained clusters; complete linkage favours compact clusters of small diameter; average and Ward linkage interpolate. The resulting dendrogram is exactly an ultrametric on the data, and cutting it at a height gives a flat clustering. It is greedy (no global objective is optimised) and expensive for large $n$, but it shows structure at every scale at once.`,
    survey: {
      task: 'A hierarchy of nested clusterings; exploratory analysis; taxonomy',
      assumptions: 'A meaningful dissimilarity between points; a linkage that matches the notion of cluster',
      output: 'Dendrogram (binary tree with merge heights) = ultrametric; flat clusters by cutting',
      strengths: 'No $K$ needed up front; any dissimilarity; shows multi-scale structure; deterministic',
      weaknesses: 'Greedy and irrevocable merges; $O(n^2)$ memory; single linkage chains, complete linkage breaks large clusters',
      complexity: '$O(n^2)$ (single via MST/SLINK) to $O(n^2\\log n)$ (general, with a priority queue)',
    },
    sections: [
      {
        title: 'Agglomerative clustering and linkage functions',
        html: R`
<p>Let $d(i,j)$ be a dissimilarity on $n$ objects (Euclidean distance, $1-$correlation, an edit distance — anything symmetric and non-negative). Agglomerative clustering never needs coordinates, only $d$.</p>
<div class="algo" data-name="Agglomerative hierarchical clustering">
<ol>
<li>Start with $n$ singleton clusters, all at height $0$.</li>
<li>Find the pair of current clusters $(A,B)$ minimising the linkage $D(A,B)$.</li>
<li>Merge them into $A\cup B$, recording the merge height $h = D(A,B)$.</li>
<li>Update $D(A\cup B, C)$ for every other cluster $C$; repeat until one cluster remains ($n-1$ merges).</li>
</ol>
</div>
<table>
<thead><tr><th>Linkage</th><th>$D(A,B)$</th><th>Character</th></tr></thead>
<tbody>
<tr><td>Single (nearest neighbour)</td><td>$\min_{a\in A,b\in B} d(a,b)$</td><td>Follows chains; finds arbitrary shapes; very sensitive to noise bridges</td></tr>
<tr><td>Complete (farthest neighbour)</td><td>$\max_{a\in A,b\in B} d(a,b)$</td><td>Compact clusters of small diameter; can split large clusters</td></tr>
<tr><td>Average (UPGMA)</td><td>$\frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B} d(a,b)$</td><td>Compromise; reasonably robust</td></tr>
<tr><td>Ward</td><td>$\frac{|A||B|}{|A|+|B|}\lVert\bar x_A-\bar x_B\rVert^2$</td><td>Increase in within-cluster sum of squares; greedy <a href="#/t/kmeans">k-means</a>; spherical clusters</td></tr>
<tr><td>Centroid</td><td>$\lVert\bar x_A-\bar x_B\rVert^2$</td><td>Can produce inversions (non-monotone heights)</td></tr>
</tbody>
</table>
<p>The Ward formula is not a guess: merging $A$ and $B$ increases the k-means objective $W$ by exactly $\frac{|A||B|}{|A|+|B|}\lVert\bar x_A-\bar x_B\rVert^2$ (apply the ANOVA decomposition to the merged cluster). So Ward's method is the greedy algorithm for $W$, one merge at a time, and its flat cuts are good k-means initialisations.</p>
<div class="thm" data-kind="Proposition" data-name="Lance–Williams recurrence (1967)">
<p>For single, complete, average and Ward linkage (Ward with squared Euclidean $d$), after merging $A$ and $B$ the linkage to any other cluster $C$ satisfies
$$D(A\cup B,C) = \alpha_A D(A,C) + \alpha_B D(B,C) + \beta D(A,B) + \gamma\,|D(A,C)-D(B,C)|$$
with: single $\alpha_A=\alpha_B=\tfrac12,\ \beta=0,\ \gamma=-\tfrac12$; complete the same but $\gamma=+\tfrac12$; average $\alpha_A=\frac{|A|}{|A|+|B|}$, $\alpha_B=\frac{|B|}{|A|+|B|}$, $\beta=\gamma=0$; Ward $\alpha_A = \frac{|A|+|C|}{|A|+|B|+|C|}$, $\alpha_B=\frac{|B|+|C|}{|A|+|B|+|C|}$, $\beta = -\frac{|C|}{|A|+|B|+|C|}$, $\gamma=0$.</p>
</div>
<details class="proof"><summary>Proof (single and complete)</summary>
<p>$\tfrac12(u+v) - \tfrac12|u-v| = \min(u,v)$ and $\tfrac12(u+v)+\tfrac12|u-v| = \max(u,v)$, and $\min_{x\in A\cup B}\min_{c\in C} d = \min(D(A,C),D(B,C))$; likewise for max. Average linkage follows by splitting the double sum over $A\cup B$. The Ward case is an algebraic identity in the cluster means using $\bar x_{A\cup B} = (|A|\bar x_A+|B|\bar x_B)/(|A|+|B|)$.</p>
</details>
<p>The recurrence means the algorithm needs only the current dissimilarity matrix, updated in $O(n)$ per merge: $O(n^2)$ memory and $O(n^2\log n)$ time with heaps (or $O(n^3)$ naively). Specialised algorithms reach $O(n^2)$ time: SLINK for single linkage (Sibson 1973), and nearest-neighbour-chain algorithms for any "reducible" linkage, which includes complete, average and Ward.</p>`,
        lab: 'hierarchical-clustering-dendrogram',
      },
      {
        title: 'Dendrograms are ultrametrics',
        html: R`
<p>A dendrogram records each merge height. Define the <strong>cophenetic distance</strong> $u(i,j)$ as the height at which $i$ and $j$ first fall into the same cluster. If merge heights never decrease (no inversions), $u$ has a special structure.</p>
<div class="def" data-name="Ultrametric">
<p>A metric $u$ is an <strong>ultrametric</strong> if it satisfies the strong triangle inequality $u(i,k)\le\max\{u(i,j),u(j,k)\}$ for all $i,j,k$. Equivalently, every triangle is isosceles with the two longest sides equal.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Dendrograms ↔ ultrametrics (Johnson 1967; Jardine & Sibson 1971)">
<p>(a) The cophenetic distance of a monotone dendrogram is an ultrametric. (b) Conversely, for every ultrametric $u$ on a finite set, the relations $i\sim_h j \iff u(i,j)\le h$ are equivalence relations whose classes are nested as $h$ grows, i.e. they form a dendrogram whose cophenetic distance is $u$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(a) Let $h = \max\{u(i,j),u(j,k)\}$. At height $h$, $i$ and $j$ are in one cluster and $j$ and $k$ are in one cluster; since the partition at height $h$ is a partition, these are the same cluster, so $i$ and $k$ have merged by height $h$: $u(i,k)\le h$. (b) $\sim_h$ is reflexive and symmetric; transitivity is exactly the strong triangle inequality. If $h\le h'$ then $u\le h$ implies $u\le h'$, so the classes at $h$ refine those at $h'$, giving a nested family.</p>
</details>
<p>So hierarchical clustering is "fit an ultrametric to a dissimilarity". The <strong>cophenetic correlation</strong> $\Corr(d(i,j),u(i,j))$ over pairs measures how faithfully a dendrogram represents the original distances. Different linkages give different ultrametric fits, and only one of them solves an optimisation problem exactly:</p>
<div class="thm" data-kind="Theorem" data-name="Single linkage = minimum spanning tree = subdominant ultrametric (Gower & Ross 1969)">
<p>Let $T$ be a minimum spanning tree (MST) of the complete graph on the points with edge weights $d(i,j)$. Then (i) the single-linkage merges are exactly the edges of $T$, processed in increasing order of weight, with merge heights equal to the edge weights; (ii) the single-linkage cophenetic distance is the <em>minimax path distance</em>
$$u_{\text{SL}}(i,j) = \min_{\text{paths } i=v_0,\dots,v_m=j}\ \max_{l} d(v_{l-1},v_l);$$
(iii) $u_{\text{SL}}$ is the largest ultrametric that lies below $d$ (the <em>subdominant</em> ultrametric): if $u$ is any ultrametric with $u\le d$ then $u\le u_{\text{SL}}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(i) Kruskal's algorithm builds an MST by scanning edges in increasing weight and adding an edge iff it joins two different components. Single linkage merges the two clusters with the smallest $\min_{a,b} d(a,b)$, i.e. it adds the lightest edge joining two different current clusters. Inductively the clusters are Kruskal's components and the merge heights are the accepted edge weights. (ii) $i$ and $j$ are joined at the first height $h$ at which the graph of edges with weight $\le h$ connects them, which is the minimax path value. (iii) $u_{\text{SL}}\le d$ (take the one-edge path) and it is an ultrametric by (a) of the previous theorem. If $u\le d$ is an ultrametric and $v_0,\dots,v_m$ is any path from $i$ to $j$, iterating the strong triangle inequality gives $u(i,j)\le\max_l u(v_{l-1},v_l)\le\max_l d(v_{l-1},v_l)$; minimise over paths.</p>
</details>
<p>So single linkage is the one linkage with a clean variational characterisation, and it can be computed in $O(n^2)$ time by Prim's algorithm. Complete linkage, by contrast, is a heuristic for a hard problem: finding the best ultrametric approximation in $\ell_1$ or $\ell_2$ error is NP-hard (Křivánek &amp; Morávek 1986), although the $\ell_\infty$ version happens to be solvable in polynomial time (Farach, Kannan &amp; Warnow 1995).</p>`,
      },
      {
        title: 'Chaining, compactness and what each linkage optimises',
        html: R`
<div class="ex" data-name="Chaining">
<p>Take two dense blobs joined by a sparse line of points spaced $0.3$ apart, while points within each blob are $0.2$ apart and the blobs are $3$ apart. Single linkage joins everything at height about $0.3$ because each bridge point is within $0.3$ of its neighbour: the minimax path between the blobs never takes a step longer than $0.3$. Complete and Ward linkage see the blobs as two compact groups, and assign the bridge points to whichever blob is nearer. The lab's "bridge" dataset shows exactly this.</p>
</div>
<p>Single linkage's chaining is the flip side of its strength. Viewed through density, the $\le h$ components of single linkage are the connected components of a union of balls of radius $h/2$ around the data — a crude estimate of the level sets of the density. Hartigan (1981) showed single linkage is only "fractionally consistent" for high-density clusters when $p\ge 2$: a few low-density points form bridges. <strong>Robust single linkage</strong> (Chaudhuri &amp; Dasgupta 2010) and HDBSCAN fix this by first discarding points whose $k$-th nearest-neighbour distance is large, making the method consistent for Hartigan's cluster tree (see <a href="#/t/spectral-density-clustering">density-based clustering</a>).</p>
<div class="cols">
<div><h4>Single</h4><p>Merge height $\le h$ iff connected by a path of steps $\le h$. Detects non-convex shapes; noise-sensitive.</p></div>
<div><h4>Complete</h4><p>Every cluster at height $h$ has diameter $\le h$. Tends to equal-diameter clusters; sensitive to outliers (they raise maxima).</p></div>
<div><h4>Average</h4><p>Merges when the mean between-cluster dissimilarity is small. Often the best cophenetic correlation.</p></div>
<div><h4>Ward</h4><p>Minimises the increase in $W$. Equal-size spherical clusters, like k-means.</p></div>
</div>
<div class="warn" data-name="Inversions">
<p>Centroid and median linkage can merge two clusters at a height <em>lower</em> than an earlier merge (the new centroid can be closer to a third cluster than either part was), so the dendrogram has crossings and its cophenetic distance is not an ultrametric. Single, complete, average and Ward are monotone: the Lance–Williams coefficients satisfy $\alpha_A+\alpha_B+\beta\ge1$, $\alpha_A,\alpha_B\ge0$, $\gamma\ge-\min(\alpha_A,\alpha_B)$, a sufficient condition for monotonicity (Milligan 1979).</p>
</div>`,
      },
      {
        title: 'Cutting the tree, and divisive methods',
        html: R`
<p>A flat clustering comes from cutting the dendrogram: at a fixed height $h$ (clusters are the classes of $u\le h$), or at the height that yields exactly $K$ clusters. Long vertical stretches of the dendrogram — large gaps between successive merge heights — indicate scales at which the partition is stable, and are the usual visual cue for a natural $K$. Inconsistency coefficients (merge height relative to the heights of the merges below it) automate this, and dynamic tree cutting (Langfelder et al. 2008) cuts different branches at different heights.</p>
<div class="warn" data-name="Greedy and irrevocable">
<p>A merge is never undone. One early mistake (two points from different true clusters that happen to be close) propagates up the whole tree. The flat clustering obtained at height $h$ therefore need not be optimal for any criterion at that height — Ward's cut at $K$ is usually worse than k-means with $K$ clusters started from it.</p>
</div>
<p><strong>Divisive</strong> (top-down) clustering starts with one cluster and recursively splits. Exact best splits are exponential in cluster size, so divisive methods use heuristics: DIANA (Kaufman &amp; Rousseeuw) splits off a "splinter group" around the most dissimilar point; bisecting k-means applies 2-means recursively; spectral bisection uses the Fiedler vector of the <a href="#/t/spectral-density-clustering">graph Laplacian</a>. Divisive methods get the top of the tree right at the expense of the bottom, the opposite of agglomerative methods.</p>`,
      },
      {
        title: 'Axioms and impossibility',
        html: R`
<p>Is there a "right" clustering method? Kleinberg (2002) formalised three natural properties for a clustering function $f$ mapping a dissimilarity $d$ on a fixed set to a partition:</p>
<ul>
<li><strong>Scale invariance:</strong> $f(\alpha d) = f(d)$ for all $\alpha \gt 0$.</li>
<li><strong>Richness:</strong> every partition is $f(d)$ for some $d$.</li>
<li><strong>Consistency:</strong> if $d'$ shrinks within-cluster distances and expands between-cluster distances of $f(d)$, then $f(d') = f(d)$.</li>
</ul>
<div class="thm" data-kind="Theorem" data-name="Impossibility (Kleinberg 2002)">
<p>For $n\ge2$ there is no clustering function satisfying all three properties. Each pair is achievable, by single linkage with an appropriate stopping rule: stop at $K$ clusters (scale invariance + consistency), stop at a fixed distance $r$ (richness + consistency), or stop at a fixed fraction of the maximum pairwise distance (scale invariance + richness).</p>
</div>
<p>The lesson is not that clustering is impossible but that a clustering method must commit to something — a number of clusters, a scale, or a model. Hierarchical clustering sidesteps the issue partly by returning all scales and leaving the cut to the analyst. Carlsson &amp; Mémoli (2010) showed that if one asks for a <em>hierarchy</em> rather than a partition, with analogous axioms, single linkage is the unique method satisfying them — another sense in which it is canonical despite its practical flaws.</p>`,
      },
    ],
    quiz: [
      { q: R`Prove that the cophenetic distance of a monotone dendrogram satisfies $u(i,k)\le\max\{u(i,j),u(j,k)\}$.`,
        a: R`At height $h=\max\{u(i,j),u(j,k)\}$, $i$ and $j$ share a cluster and $j$ and $k$ share a cluster. Clusters at a given height form a partition, so these are the same cluster and $i$, $k$ have merged by height $h$.` },
      { q: R`Why is single linkage equivalent to Kruskal's minimum spanning tree algorithm?`,
        a: R`Single linkage merges the two current clusters with the smallest minimum pairwise distance, which is the lightest edge connecting two different components. Kruskal adds edges in increasing order, accepting exactly those that connect different components. By induction the clusters are Kruskal's components and the merge heights are the MST edge weights; cutting the $K-1$ heaviest MST edges gives the single-linkage $K$-clustering.` },
      { q: R`What does Ward's linkage measure, and why are its flat cuts good starting points for k-means?`,
        a: R`$\frac{|A||B|}{|A|+|B|}\lVert\bar x_A-\bar x_B\rVert^2$ is exactly the increase in within-cluster sum of squares $W$ caused by merging $A$ and $B$. Ward is therefore the greedy bottom-up algorithm for the k-means objective, so its $K$-cluster cut has small $W$ and lies near a good k-means local optimum; Lloyd iterations then refine it.` },
      { q: R`Kleinberg's theorem says no clustering function is scale-invariant, rich and consistent. Which property does "single linkage stopped at $K$ clusters" give up, and why?`,
        a: R`Richness. With $K$ fixed, the output always has exactly $K$ clusters, so partitions with a different number of clusters can never be produced. It is scale-invariant (rescaling distances does not change the MST order) and consistent (shrinking within- and stretching between-cluster distances keeps the same $K-1$ heaviest MST edges).` },
    ],
  });

  /* ===================================================================== Spectral & density */
  Atlas.topic({
    id: 'spectral-density-clustering',
    domain: 'unsupervised',
    order: 50,
    title: 'Spectral & Density-Based Clustering',
    subtitle: 'Clusters as weakly connected pieces of a similarity graph, or as dense regions separated by sparse ones',
    level: 3,
    sources: { aos: '—', cb: '—', other: 'von Luxburg (2007) A tutorial on spectral clustering; Shi & Malik (2000); Ng, Jordan & Weiss (2002); Ester et al. (1996); ESL §14.5.3; Murphy (2012) Ch. 25' },
    prereqs: ['kmeans'],
    related: ['hierarchical-clustering', 'kernel-density', 'manifold-learning', 'kernel-methods', 'pca'],
    tags: ['graph Laplacian', 'normalized cut', 'eigengap', 'DBSCAN', 'level sets'],
    summary: R`k-means sees clusters as round blobs around centres. Two families of methods drop that geometry. Spectral clustering builds a similarity graph on the data and looks for a partition that cuts few edges relative to cluster volume; relaxing this combinatorial problem leads to the bottom eigenvectors of a graph Laplacian, whose zero eigenspace exactly indexes the connected components. Density-based methods such as DBSCAN define clusters as connected regions where the estimated density exceeds a threshold, and declare everything else noise. Both find non-convex shapes — moons, rings, spirals — where centroid methods fail.`,
    survey: {
      task: 'Clustering with non-convex shapes; graph partitioning; clustering with noise',
      assumptions: 'Spectral: a good similarity graph (scale $\\sigma$ or $k$-NN). DBSCAN: clusters are density-connected regions at one density level',
      output: 'Spectral: $K$ clusters via eigenvectors + k-means. DBSCAN: clusters, border points, noise points',
      strengths: 'Arbitrary shapes; spectral has a clean variational motivation; DBSCAN finds $K$ itself and flags outliers',
      weaknesses: 'Sensitive to $\\sigma$ / $k$ / $\\varepsilon$; spectral costs $O(n^3)$ dense; DBSCAN fails with varying densities',
      complexity: 'Spectral $O(n^3)$ dense, near-linear with sparse graphs + Lanczos; DBSCAN $O(n\\log n)$ with spatial index',
    },
    sections: [
      {
        title: 'Similarity graphs and the graph Laplacian',
        html: R`
<p>Spectral clustering starts from a weighted undirected graph on the $n$ data points with symmetric weights $w_{ij}\ge0$. Common choices:</p>
<ul>
<li><strong>Gaussian (fully connected):</strong> $w_{ij} = \exp(-\lVert x_i-x_j\rVert^2/2\sigma^2)$.</li>
<li><strong>$\varepsilon$-neighbourhood:</strong> $w_{ij}=1$ if $\lVert x_i-x_j\rVert\le\varepsilon$.</li>
<li><strong>$k$-nearest-neighbour:</strong> connect $i$ and $j$ if either is among the other's $k$ nearest neighbours (symmetrised), possibly with Gaussian weights; usually the most robust choice.</li>
</ul>
<p>Let $d_i = \sum_j w_{ij}$ be the degree, $D=\diag(d_i)$, and $\text{vol}(A) = \sum_{i\in A}d_i$.</p>
<div class="def" data-name="Graph Laplacians">
<p>Unnormalised: $L = D - W$. Symmetric normalised: $L_{\text{sym}} = I - D^{-1/2}WD^{-1/2}$. Random-walk: $L_{\text{rw}} = I - D^{-1}W$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Properties of $L$">
<p>(i) For every $f\in\R^n$, $f^\top Lf = \tfrac12\sum_{i,j}w_{ij}(f_i-f_j)^2$. (ii) $L$ is symmetric positive semi-definite. (iii) $L\mathbf 1 = 0$, so $0$ is the smallest eigenvalue. (iv) $\lambda$ is an eigenvalue of $L_{\text{rw}}$ with eigenvector $u$ iff $Lu = \lambda Du$ iff $\lambda$ is an eigenvalue of $L_{\text{sym}}$ with eigenvector $D^{1/2}u$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(i) $f^\top Lf = \sum_i d_if_i^2 - \sum_{i,j}w_{ij}f_if_j = \tfrac12\big(\sum_{i,j}w_{ij}f_i^2 - 2\sum_{i,j}w_{ij}f_if_j + \sum_{i,j}w_{ij}f_j^2\big) = \tfrac12\sum_{i,j}w_{ij}(f_i-f_j)^2$. (ii) follows from (i) and symmetry of $W$; (iii) row sums of $L$ are $d_i - \sum_j w_{ij}=0$. (iv) $L_{\text{rw}}u=\lambda u \iff D^{-1}Lu = \lambda u\iff Lu=\lambda Du$, and substituting $u = D^{-1/2}v$ gives $D^{-1/2}LD^{-1/2}v = \lambda v$, i.e. $L_{\text{sym}}v=\lambda v$.</p>
</details>
<p>The quadratic form (i) is the key: $f^\top Lf$ is small exactly when $f$ is nearly constant across heavy edges. A function that is constant on each tightly connected piece and changes only across light edges has small "energy".</p>
<div class="thm" data-kind="Theorem" data-name="Zero eigenvalues count connected components">
<p>The multiplicity of the eigenvalue $0$ of $L$ equals the number $c$ of connected components $A_1,\dots,A_c$ of the graph, and the null space is spanned by the indicator vectors $\ind_{A_1},\dots,\ind_{A_c}$. The same holds for $L_{\text{rw}}$; for $L_{\text{sym}}$ the null space is spanned by $D^{1/2}\ind_{A_j}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>If $Lf = 0$ then $0 = f^\top L f = \tfrac12\sum_{i,j}w_{ij}(f_i-f_j)^2$, so $f_i = f_j$ whenever $w_{ij} \gt 0$. Following edges, $f$ is constant on each connected component. Conversely each $\ind_{A_j}$ satisfies $(L\ind_{A_j})_i = d_i\ind\{i\in A_j\} - \sum_{l\in A_j}w_{il} = 0$ for all $i$ (for $i\in A_j$ all its edges stay in $A_j$; for $i\notin A_j$ both terms vanish). So the null space is exactly the span of the $c$ indicators, which are linearly independent (disjoint supports). Hence the multiplicity of $0$ — algebraic equals geometric, as $L$ is symmetric — is $c$. The statements for $L_{\text{rw}}$ and $L_{\text{sym}}$ follow from part (iv) of the proposition.</p>
</details>
<div class="note">
<p>If the clusters are perfectly separated (no edges between them), the bottom $K$ eigenvectors span cluster indicators, and the rows of the $n\times K$ eigenvector matrix take only $K$ distinct values — one per cluster. Any clustering of those rows, even trivial k-means, recovers the clusters exactly. With a few weak between-cluster edges, $L$ is a small perturbation of a block-diagonal matrix, and by the Davis–Kahan theorem the eigenvectors move by an amount proportional to the perturbation divided by the <em>eigengap</em> $\lambda_{K+1}-\lambda_K$. That is the whole theory of why spectral clustering works.</p>
</div>`,
      },
      {
        title: 'Graph cuts and their spectral relaxation',
        html: R`
<p>Clustering as graph partitioning: find disjoint $A_1,\dots,A_K$ covering the vertices that cut little weight. With $\text{cut}(A,B) = \sum_{i\in A,j\in B}w_{ij}$, the naive minimum cut tends to slice off a single vertex. Balanced criteria fix this:
$$\text{RatioCut}(A_1,\dots,A_K) = \sum_k\frac{\text{cut}(A_k,\bar A_k)}{|A_k|},\qquad \text{Ncut}(A_1,\dots,A_K) = \sum_k\frac{\text{cut}(A_k,\bar A_k)}{\text{vol}(A_k)} .$$
Both are NP-hard to minimise exactly (Wagner &amp; Wagner 1993). Spectral clustering is their continuous relaxation.</p>
<div class="thm" data-kind="Proposition" data-name="Ncut as a constrained Rayleigh quotient (Shi & Malik 2000)">
<p>For $K=2$ and a partition $(A,B)$, define $f_i = \sqrt{\text{vol}(B)/\text{vol}(A)}$ for $i\in A$ and $f_i = -\sqrt{\text{vol}(A)/\text{vol}(B)}$ for $i\in B$. Then
$$(Df)^\top\mathbf 1 = 0,\qquad f^\top Df = \text{vol}(V),\qquad f^\top Lf = \text{vol}(V)\cdot\text{Ncut}(A,B).$$
Hence minimising Ncut is minimising $f^\top Lf/f^\top Df$ over such two-valued $f$ with $Df\perp\mathbf1$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $a=\text{vol}(A)$, $b=\text{vol}(B)$. Then $(Df)^\top\mathbf1 = \sum_{A}d_i\sqrt{b/a} - \sum_B d_i\sqrt{a/b} = \sqrt{ab}-\sqrt{ab}=0$; $f^\top Df = a\cdot\frac ba + b\cdot\frac ab = a+b$. By the quadratic-form identity only edges between $A$ and $B$ contribute, each with $(f_i-f_j)^2 = (\sqrt{b/a}+\sqrt{a/b})^2 = \frac ba+\frac ab+2 = \frac{(a+b)^2}{ab}$, so $f^\top Lf = \text{cut}(A,B)\frac{(a+b)^2}{ab} = (a+b)\,\text{cut}(A,B)\big(\tfrac1a+\tfrac1b\big) = \text{vol}(V)\,\text{Ncut}(A,B)$.</p>
</details>
<p>Now <strong>relax</strong>: drop the two-valued constraint and minimise $f^\top Lf/f^\top Df$ over all real $f$ with $Df\perp\mathbf1$. By the Courant–Fischer theorem for the generalised problem $Lf = \lambda Df$, the solution is the eigenvector of $L_{\text{rw}}$ with the second-smallest eigenvalue (the smallest is $\mathbf 1$ with eigenvalue $0$, excluded by the constraint). Threshold it (at $0$, or at the value minimising Ncut) to get a partition. For general $K$ one takes the bottom $K$ eigenvectors and clusters their rows. The analogous calculation with $f_i=\sqrt{|B|/|A|}$ and $-\sqrt{|A|/|B|}$ shows that RatioCut relaxes to the bottom eigenvectors of the unnormalised $L$.</p>
<div class="algo" data-name="Normalised spectral clustering (Ng, Jordan & Weiss 2002)">
<ol>
<li>Build $W$ (e.g. Gaussian with scale $\sigma$, or symmetric $k$-NN) and $D$.</li>
<li>Compute the $K$ eigenvectors $v_1,\dots,v_K$ of $L_{\text{sym}}$ with smallest eigenvalues; stack them as columns of $V\in\R^{n\times K}$.</li>
<li>Normalise each row of $V$ to unit length, giving $y_i\in\R^K$.</li>
<li>Run k-means on $y_1,\dots,y_n$; assign $x_i$ to the cluster of $y_i$.</li>
</ol>
</div>
<p>The relaxation has no approximation guarantee in general (Guattery &amp; Miller 1998 give graphs where the spectral cut is far from optimal), but there is a deep inequality connecting them. <strong>Cheeger's inequality</strong> for the normalised Laplacian says $\lambda_2/2\le h_G\le\sqrt{2\lambda_2}$, where $h_G = \min_A \text{cut}(A,\bar A)/\min(\text{vol}A,\text{vol}\bar A)$ is the conductance; and thresholding the second eigenvector ("sweep cut") achieves conductance at most $\sqrt{2\lambda_2}$.</p>
<div class="note" data-name="Random-walk view">
<p>$P = D^{-1}W$ is the transition matrix of a random walk on the graph, with stationary distribution $\pi_i = d_i/\text{vol}(V)$. Meilă &amp; Shi (2001) showed $\text{Ncut}(A,\bar A) = \P(X_1\in\bar A\mid X_0\in A) + \P(X_1\in A\mid X_0\in\bar A)$ when $X_0\sim\pi$: a good normalised cut is a pair of sets the walk rarely leaves. The bottom eigenvectors of $L_{\text{rw}}$ are the slowest-decaying modes of the walk, the "metastable" sets.</p>
</div>`,
        lab: 'spectral-density-clustering-spectral',
      },
      {
        title: 'Practical choices for spectral clustering',
        html: R`
<p><strong>Choosing $K$:</strong> the eigengap heuristic chooses $K$ so that $\lambda_1,\dots,\lambda_K$ are small and $\lambda_{K+1}$ is markedly larger — the perturbation theory above says the clustering is stable exactly when this gap is large.</p>
<p><strong>Choosing the graph:</strong> this matters more than anything else. With Gaussian weights, $\sigma$ too small disconnects the graph into many tiny components (many near-zero eigenvalues); $\sigma$ too large makes $W$ nearly constant and destroys the structure (spectral clustering degenerates toward something like k-means on a kernel embedding). Self-tuning (Zelnik-Manor &amp; Perona 2004) uses local scales $\sigma_i$ = distance to the 7th neighbour and $w_{ij} = \exp(-\lVert x_i-x_j\rVert^2/\sigma_i\sigma_j)$. A symmetric $k$-NN graph with $k\sim\log n$ is a common robust default.</p>
<p><strong>Which Laplacian:</strong> von Luxburg (2007) recommends the normalised versions, particularly $L_{\text{rw}}$: Ncut balances clusters by volume (which reflects within-cluster connectivity) rather than by size, and normalised spectral clustering is statistically consistent as $n\to\infty$ under general conditions, while the unnormalised version can fail to be (von Luxburg, Belkin &amp; Bousquet 2008).</p>
<p><strong>Scale:</strong> a dense eigendecomposition is $O(n^3)$. With a sparse $k$-NN graph, Lanczos or LOBPCG gives the bottom eigenvectors in roughly $O(nk\cdot\text{iterations})$; the Nyström approximation samples landmark columns of $W$.</p>
<div class="cols">
<div><h4>Kernel k-means</h4><p>Weighted kernel k-means with kernel $D^{-1}WD^{-1}$ has the same objective as Ncut (Dhillon, Guan &amp; Kulis 2004): spectral clustering and kernel k-means are two algorithms for one problem.</p></div>
<div><h4>Laplacian eigenmaps</h4><p>Using the bottom nonzero eigenvectors as coordinates is a nonlinear embedding (Belkin &amp; Niyogi 2003) — see <a href="#/t/manifold-learning">manifold learning</a>.</p></div>
</div>`,
      },
      {
        title: 'Density-based clustering and DBSCAN',
        html: R`
<p>A different definition of a cluster, due to Hartigan (1975): given a density $f$ and level $\lambda$, the clusters at level $\lambda$ are the connected components of the upper level set $\{x : f(x)\ge\lambda\}$. As $\lambda$ varies these components form a tree, the <strong>cluster tree</strong>. Points in low-density regions belong to no cluster. Density-based methods estimate this from data.</p>
<p>DBSCAN (Ester, Kriegel, Sander &amp; Xu 1996) takes $\lambda$ implicitly through two parameters, a radius $\varepsilon$ and a count $m$ (minPts). Let $N_\varepsilon(x) = \{x_j : \lVert x_j - x\rVert\le\varepsilon\}$, including $x$ itself.</p>
<div class="def" data-name="DBSCAN point types">
<p>A data point $x_i$ is a <strong>core point</strong> if $|N_\varepsilon(x_i)|\ge m$. A non-core point within $\varepsilon$ of some core point is a <strong>border point</strong>. All other points are <strong>noise</strong>. Point $y$ is <em>directly density-reachable</em> from a core point $x$ if $y\in N_\varepsilon(x)$; <em>density-reachable</em> if there is a chain of such steps through core points; $x$ and $y$ are <em>density-connected</em> if both are density-reachable from a common core point. A cluster is a maximal set of mutually density-connected points.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="DBSCAN as connected components">
<p>The DBSCAN clusters restricted to core points are exactly the connected components of the graph on core points with edges $\lVert x_i - x_j\rVert\le\varepsilon$. Each border point is attached to (one of) the components of the core points within $\varepsilon$ of it; noise points are unassigned. The core-point clustering is unique; only border points within $\varepsilon$ of two different clusters can depend on processing order.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>If core points $x,y$ are joined by a chain of core points with consecutive distances $\le\varepsilon$, each is density-reachable from the other, so they are in one cluster. Conversely, density-reachability between core points can only pass through core points (the chain's intermediate points must be core to continue), so two core points in one cluster are joined by such a chain. Hence core clusters are the graph components, determined by the data alone. A border point is density-reachable from every core point within $\varepsilon$; if those lie in two components, it is density-connected to both, and the classical algorithm assigns it to whichever cluster reaches it first.</p>
</details>
<p>The density connection: $|N_\varepsilon(x)|/(n\,\text{vol}(B_\varepsilon))$ is a uniform-kernel <a href="#/t/kernel-density">density estimate</a> $\hat f_\varepsilon(x)$, so core points are the sample points where $\hat f_\varepsilon\ge\lambda := m/(n\,\text{vol}(B_\varepsilon))$, and the clusters approximate the components of $\{\hat f_\varepsilon\ge\lambda\}$. DBSCAN is a single horizontal slice of the cluster tree.</p>
<div class="cols">
<div><h4>HDBSCAN</h4><p>Varies the level: builds the single-linkage tree under the "mutual reachability" distance $\max\{\text{core}_m(a),\text{core}_m(b),d(a,b)\}$ and extracts the most persistent clusters (Campello, Moulavi &amp; Sander 2013). Handles clusters of different densities.</p></div>
<div><h4>Mean shift</h4><p>Moves each point uphill on a kernel density estimate until it reaches a mode; clusters are basins of attraction of the modes (Fukunaga &amp; Hostetler 1975; Comaniciu &amp; Meer 2002). $K$ is determined by the bandwidth.</p></div>
<div><h4>OPTICS</h4><p>Orders points so that every DBSCAN clustering for $\varepsilon'\le\varepsilon$ can be read off a "reachability plot" (Ankerst et al. 1999).</p></div>
</div>`,
        lab: 'spectral-density-clustering-dbscan',
      },
      {
        title: 'Pitfalls',
        html: R`
<div class="warn" data-name="Spectral: the graph is the model">
<p>All the cleverness of spectral clustering is downstream of $W$. On data where the clusters touch, no graph separates them and spectral clustering returns arbitrary cuts; with a badly chosen $\sigma$ it can do worse than k-means. Inspect the eigenvalues: no clear gap means no clear clusters.</p>
</div>
<div class="warn" data-name="Spectral: outliers and unbalanced clusters">
<p>A point far from everything has tiny degree and forms its own near-zero eigenvalue, spending one of the $K$ clusters. Clusters of very different sizes or densities break the balanced-cut assumption.</p>
</div>
<div class="warn" data-name="DBSCAN: one density level">
<p>A single $(\varepsilon,m)$ cannot simultaneously resolve a dense cluster and a sparse one — $\varepsilon$ small enough to separate two nearby dense clusters makes the sparse cluster all noise. Use HDBSCAN or OPTICS. A $k$-distance plot (sorted distance to the $m$-th neighbour) helps choose $\varepsilon$ at its "knee".</p>
</div>
<div class="warn" data-name="DBSCAN: high dimension">
<p>Density estimation suffers the curse of dimensionality: in high dimensions $\varepsilon$-balls are either empty or contain almost everything. Reduce dimension first.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Prove that $\ind_A$ is in the null space of $L=D-W$ when $A$ is a union of connected components, and that every null vector is constant on components.`,
        a: R`For $i\in A$, $(L\ind_A)_i = d_i - \sum_{j\in A}w_{ij} = 0$ because all of $i$'s neighbours are in $A$; for $i\notin A$ both terms are $0$. If $Lf=0$ then $0=f^\top Lf=\tfrac12\sum w_{ij}(f_i-f_j)^2$, forcing $f_i=f_j$ across every edge, hence constancy on each component.` },
      { q: R`What combinatorial problem does the second eigenvector of $L_{\text{rw}}$ solve approximately, and what exactly is relaxed?`,
        a: R`Normalised cut, $\text{cut}(A,B)(1/\text{vol}A + 1/\text{vol}B)$. With the two-valued indicator $f$ (values $\sqrt{\text{vol}B/\text{vol}A}$ and $-\sqrt{\text{vol}A/\text{vol}B}$), Ncut equals $f^\top Lf/f^\top Df$ with $Df\perp\mathbf1$. Relaxing $f$ to arbitrary real vectors makes this a generalised Rayleigh quotient minimised by the second eigenvector of $Lf=\lambda Df$.` },
      { q: R`Two concentric rings: why does k-means fail, why can spectral clustering succeed, and what happens to spectral clustering if $\sigma$ is much larger than the gap between rings?`,
        a: R`k-means' Voronoi cells are convex, and no convex partition separates nested rings. With a Gaussian graph at small $\sigma$, within-ring edges are strong and between-ring edges negligible, so the graph is nearly two components and the bottom two eigenvectors are nearly ring indicators. If $\sigma$ exceeds the gap, between-ring weights are comparable to within-ring ones, the near-block structure disappears, the eigengap closes, and the partition reverts to something like a straight cut.` },
      { q: R`In DBSCAN, why is the assignment of core points order-independent while that of border points is not?`,
        a: R`Core clusters are the connected components of the $\varepsilon$-graph on core points, a property of the data alone. A border point is attached to whichever cluster first reaches it; if it is within $\varepsilon$ of core points from two different components, both are valid and the processing order decides.` },
    ],
  });

  /* ===================================================================== PCA */
  Atlas.topic({
    id: 'pca',
    domain: 'unsupervised',
    order: 60,
    title: 'Principal Component Analysis',
    subtitle: 'The directions of maximal variance are the best low-rank approximation, and both come from one SVD',
    level: 2,
    sources: { aos: '—', cb: '—', other: 'ESL §14.5.1; Bishop PRML §12.1–12.2; Murphy (2012) Ch. 12; Jolliffe, Principal Component Analysis (2002); Tipping & Bishop (1999)' },
    prereqs: ['multivariate'],
    related: ['factor-analysis-ica', 'manifold-learning', 'ridge-regression', 'autoencoders-vae', 'spectral-density-clustering'],
    tags: ['dimension reduction', 'SVD', 'Eckart–Young', 'scree plot', 'probabilistic PCA'],
    summary: R`PCA finds an orthonormal set of directions ordered by how much variance of the data they capture. Two apparently different goals — keep as much variance as possible in $q$ dimensions, and lose as little as possible when reconstructing the data from $q$ dimensions — have the same answer, by Pythagoras: the top $q$ eigenvectors of the sample covariance. Computationally everything flows from the singular value decomposition of the centred data matrix, and the Eckart–Young theorem makes truncated SVD the optimal rank-$q$ approximation. Probabilistic PCA recasts PCA as maximum likelihood in a latent Gaussian model, and random-matrix theory tells us when a sample eigenvalue reflects signal rather than noise.`,
    survey: {
      task: 'Linear dimension reduction; decorrelation; compression; visualisation; denoising',
      assumptions: 'Variance is the signal of interest; linear structure; variables on comparable scales',
      output: 'Orthonormal loadings $v_1,\\dots,v_q$, variances $\\lambda_j$, scores $Xv_j$, rank-$q$ reconstruction',
      strengths: 'Optimal (Eckart–Young); unique (given distinct eigenvalues); fast; no tuning beyond $q$',
      weaknesses: 'Only linear structure; scale-dependent; outlier-sensitive; variance $\\neq$ relevance; noisy in $p \\gtrsim n$',
      complexity: '$O(np\\min(n,p))$ full SVD; $O(npq)$ for the top $q$ by randomized / Lanczos methods',
    },
    sections: [
      {
        title: 'Two goals, one answer',
        html: R`
<p>Let $X\in\R^{n\times p}$ have centred columns (subtract the column means), rows $x_i^\top$, and sample covariance $S = \frac1n X^\top X$ (use $n-1$ if you prefer; it changes nothing below). Let $S = V\Lambda V^\top$ with $\lambda_1\ge\dots\ge\lambda_p\ge0$ and orthonormal eigenvectors $v_1,\dots,v_p$.</p>
<p><strong>Goal 1 — maximal variance.</strong> Among unit vectors $a$, which direction maximises the sample variance of the projections $a^\top x_i$, namely $a^\top Sa$?</p>
<div class="thm" data-kind="Theorem" data-name="Variance maximisation">
<p>$\max_{\lVert a\rVert=1}a^\top Sa = \lambda_1$, attained at $a=v_1$. Inductively, the maximum of $a^\top Sa$ over unit $a$ orthogonal to $v_1,\dots,v_{j-1}$ is $\lambda_j$, attained at $v_j$. The resulting scores $Xv_j$ are uncorrelated, with variances $\lambda_j$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Expand $a = \sum_k c_kv_k$ with $\sum c_k^2 = 1$. Then $a^\top Sa = \sum_k\lambda_kc_k^2\le\lambda_1\sum c_k^2 = \lambda_1$, with equality at $c = e_1$. Orthogonality to $v_1,\dots,v_{j-1}$ forces $c_1=\dots=c_{j-1}=0$, and the same argument gives $\lambda_j$. Finally $\frac1n(Xv_j)^\top(Xv_k) = v_j^\top Sv_k = \lambda_k\ind\{j=k\}$.</p>
</details>
<p><strong>Goal 2 — best reconstruction.</strong> Among $q$-dimensional subspaces, with orthogonal projector $P$, which minimises the reconstruction error $\sum_i\lVert x_i - Px_i\rVert^2$?</p>
<div class="thm" data-kind="Theorem" data-name="Best-fitting subspace (Pearson 1901)">
<p>$\min_{P}\sum_i\lVert x_i-Px_i\rVert^2 = n\sum_{j\gt q}\lambda_j$ over rank-$q$ orthogonal projectors, attained at $P = V_qV_q^\top$ with $V_q = [v_1,\dots,v_q]$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By Pythagoras, $\lVert x_i\rVert^2 = \lVert Px_i\rVert^2 + \lVert(I-P)x_i\rVert^2$, so minimising the residual is maximising $\sum_i\lVert Px_i\rVert^2 = n\tr(PS)$. Write $P = UU^\top$ with $U\in\R^{p\times q}$, $U^\top U = I_q$, so $\tr(PS) = \tr(U^\top SU) = \sum_k\lambda_k w_k$ with $w_k = \lVert U^\top v_k\rVert^2$. Now $0\le w_k\le1$ (norm of a projection of a unit vector) and $\sum_k w_k = \tr(U^\top VV^\top U) = \tr(U^\top U) = q$. Maximising $\sum_k\lambda_kw_k$ over such weights puts $w_k = 1$ on the $q$ largest $\lambda_k$ (a linear function on this polytope is maximised at a vertex). So $\tr(PS)\le\sum_{j\le q}\lambda_j$ (Ky Fan's inequality), with equality at $U=V_q$, and the residual is $n(\tr S - \sum_{j\le q}\lambda_j) = n\sum_{j\gt q}\lambda_j$.</p>
</details>
<div class="note">
<p>The two goals coincide because the total variance $\tr S$ is fixed: variance captured plus squared residual equals total. In the lab, rotate a candidate axis and watch the captured variance and the mean squared projection distance trade off exactly, the sum constant.</p>
</div>`,
        lab: 'pca-projection',
      },
      {
        title: 'The SVD and the Eckart–Young theorem',
        html: R`
<p>Compute PCA from the thin SVD $X = UDV^\top$ with $U\in\R^{n\times r}$, $V\in\R^{p\times r}$ orthonormal columns and $D = \diag(d_1\ge\dots\ge d_r \gt 0)$, $r = \text{rank}(X)$. Then $X^\top X = VD^2V^\top$, so</p>
<ul>
<li>the <strong>loadings</strong> (principal directions) are the right singular vectors $v_j$;</li>
<li>the <strong>variances</strong> are $\lambda_j = d_j^2/n$;</li>
<li>the <strong>scores</strong> are $Xv_j = d_ju_j$: the left singular vectors scaled by singular values.</li>
</ul>
<p>Working with $X$ directly is numerically better than forming $X^\top X$, which squares the condition number. When $p\gg n$, $XX^\top = UD^2U^\top$ is the smaller matrix — the "Gram trick" behind <a href="#/t/manifold-learning">classical MDS</a> and kernel PCA.</p>
<div class="thm" data-kind="Theorem" data-name="Eckart–Young–Mirsky (1936)">
<p>Let $X_q = \sum_{j\le q}d_ju_jv_j^\top$. For every matrix $B$ of rank at most $q$,
$$\lVert X - B\rVert_F^2\ \ge\ \lVert X - X_q\rVert_F^2 = \sum_{j\gt q}d_j^2,\qquad \lVert X-B\rVert_2\ \ge\ \lVert X-X_q\rVert_2 = d_{q+1}.$$</p>
</div>
<details class="proof"><summary>Proof (Frobenius norm)</summary>
<p>The row space of $B$ is contained in some $q$-dimensional subspace with projector $P$. For any such $B$, $\lVert X-B\rVert_F^2 = \sum_i\lVert x_i - b_i\rVert^2\ge\sum_i\lVert x_i - Px_i\rVert^2$ because $Px_i$ is the closest point of the subspace to $x_i$. By the best-fitting subspace theorem the right side is at least $n\sum_{j\gt q}\lambda_j = \sum_{j\gt q}d_j^2$, attained with $P = V_qV_q^\top$, i.e. $B = XV_qV_q^\top = X_q$. (Spectral norm: any rank-$q$ $B$ has a null space of dimension $p-q$, which meets $\text{span}(v_1,\dots,v_{q+1})$ in a unit vector $w$; then $\lVert(X-B)w\rVert = \lVert Xw\rVert\ge d_{q+1}$.)</p>
</details>
<p>So PCA is simultaneously optimal compression of the rows, of the columns, and of the matrix as a whole. The same theorem underlies low-rank matrix completion, latent semantic analysis, and the analysis of <a href="#/t/ridge-regression">ridge regression</a>, which shrinks the fit along principal direction $j$ by $d_j^2/(d_j^2+\lambda)$: ridge is a soft version of principal-components regression.</p>`,
      },
      {
        title: 'How many components? Scree plots and random-matrix theory',
        html: R`
<p>The fraction of variance explained by $q$ components is $\sum_{j\le q}\lambda_j/\sum_j\lambda_j$. Common rules: keep enough for $80$–$90\%$; look for an elbow in the <strong>scree plot</strong> of $\lambda_j$ against $j$ (Cattell 1966); Kaiser's rule (keep $\lambda_j \gt 1$ for correlation-matrix PCA); <strong>parallel analysis</strong> (Horn 1965), which keeps components whose eigenvalues exceed those obtained from data with columns independently permuted. The last has a theoretical basis.</p>
<div class="thm" data-kind="Theorem" data-name="Marchenko–Pastur law (1967)">
<p>Let the rows of $X$ be IID with independent entries of mean $0$ and variance $\sigma^2$ (and finite fourth moment), and let $n,p\to\infty$ with $p/n\to\gamma\in(0,1]$. Then the empirical distribution of the eigenvalues of $S = X^\top X/n$ converges almost surely to a deterministic law with density supported on $[\sigma^2(1-\sqrt\gamma)^2,\ \sigma^2(1+\sqrt\gamma)^2]$; moreover the largest eigenvalue converges to the upper edge $\sigma^2(1+\sqrt\gamma)^2$ (Geman 1980; Bai &amp; Yin 1988).</p>
</div>
<p>So even with <em>no</em> structure at all ($\Sigma=\sigma^2I$), sample eigenvalues spread out: with $p=100$, $n=400$ the top one is near $(1.5)^2 = 2.25$ times the truth. A scree plot of pure noise slopes down convincingly. Signal must be judged against this edge.</p>
<div class="thm" data-kind="Theorem" data-name="BBP phase transition (Baik, Ben Arous & Péché 2005; Paul 2007)">
<p>In the spiked model $\Sigma = I + \ell\,vv^\top$ ($\sigma^2=1$, $\lVert v\rVert=1$, Gaussian data), with $p/n\to\gamma$: if $\ell\le\sqrt\gamma$ the top sample eigenvalue converges to $(1+\sqrt\gamma)^2$ and the top sample eigenvector $\hat v$ is asymptotically orthogonal to $v$; if $\ell \gt \sqrt\gamma$,
$$\hat\lambda_1\to(1+\ell)\Big(1+\frac\gamma\ell\Big) \gt (1+\sqrt\gamma)^2,\qquad \langle\hat v,v\rangle^2\to\frac{1-\gamma/\ell^2}{1+\gamma/\ell}.$$</p>
</div>
<p>Below the threshold a genuine signal direction is <em>undetectable</em> by PCA, however you choose $q$; above it the sample eigenvalue overshoots the true $1+\ell$ and the eigenvector is only partially aligned. This is why PCA in high dimensions needs either $n\gg p$, structural assumptions (sparse PCA), or shrinkage of eigenvalues. Explore it in the second lab.</p>`,
        lab: 'pca-spiked',
      },
      {
        title: 'Probabilistic PCA',
        html: R`
<p>Tipping &amp; Bishop (1999) and Roweis (1998) gave PCA a likelihood. Model
$$x = \mu + Wz + \varepsilon,\qquad z\sim\N(0,I_q),\quad\varepsilon\sim\N(0,\sigma^2I_p),\quad z\indep\varepsilon,$$
so marginally $x\sim\N(\mu, C)$ with $C = WW^\top + \sigma^2I$: a covariance that is "low rank plus isotropic noise".</p>
<div class="thm" data-kind="Theorem" data-name="MLE for probabilistic PCA (Tipping & Bishop 1999)">
<p>With $S = V\Lambda V^\top$ the sample covariance, the maximum likelihood estimates are $\hat\mu=\bar x$,
$$\hat\sigma^2 = \frac{1}{p-q}\sum_{j=q+1}^p\lambda_j,\qquad \hat W = V_q(\Lambda_q - \hat\sigma^2I)^{1/2}R$$
for any orthogonal $R\in\R^{q\times q}$, assuming $\lambda_q \gt \lambda_{q+1}$. All other stationary points are saddle points.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The log-likelihood is $-\frac n2\big[p\log2\pi + \log|C| + \tr(C^{-1}S)\big]$. Setting $\partial/\partial W = 0$ gives $SC^{-1}W = W$. Write $W = ULR^\top$ (SVD). Then each column of $U$ satisfies $SU_j = (\sigma^2 + l_j^2)U_j$ — it is an eigenvector of $S$ with eigenvalue $\lambda = \sigma^2+l_j^2$, so $l_j = (\lambda-\sigma^2)^{1/2}$. Substituting back, the likelihood depends on which $q$ eigenvalues were chosen through $\sum_{\text{discarded}}\log\lambda_j$ and the noise term; it is maximised by keeping the largest $q$ and setting $\sigma^2$ to the mean of the discarded ones. Choosing any other eigenvectors gives a saddle.</p>
</details>
<p>Consequences. The noise variance is the average variance "left over"; the rotation $R$ is not identified (the model only sees $WW^\top$), the same indeterminacy that plagues <a href="#/t/factor-analysis-ica">factor analysis</a>. The posterior of the latent is $z\mid x\sim\N\big(M^{-1}W^\top(x-\mu),\ \sigma^2M^{-1}\big)$ with $M = W^\top W+\sigma^2I_q$; as $\sigma^2\to0$ the posterior mean becomes $(W^\top W)^{-1}W^\top(x-\mu)$, an orthogonal projection onto the principal subspace — classical PCA is the zero-noise limit. The probabilistic view buys: a likelihood for choosing $q$ (by BIC or the Bayesian PCA of Bishop 1999), principled handling of missing entries via <a href="#/t/em-algorithm">EM</a>, mixtures of PPCA (Tipping &amp; Bishop 1999), and a density for anomaly detection.</p>`,
      },
      {
        title: 'Using PCA well',
        html: R`
<div class="warn" data-name="Scale">
<p>PCA on the covariance matrix is not invariant to rescaling variables: a variable measured in grams instead of kilograms dominates PC1. If variables are in different units, use the correlation matrix (PCA of standardised data). If they share units and their variances are meaningful (e.g. pixel intensities), do not standardise.</p>
</div>
<div class="warn" data-name="Variance is not relevance">
<p>The direction of largest variance need not be the direction that matters for a response. In regression, principal-components regression can discard a low-variance direction that carries all the signal (Jolliffe 1982 gives real examples). Supervised alternatives: partial least squares, or simply <a href="#/t/ridge-regression">ridge</a>.</p>
</div>
<div class="warn" data-name="Outliers and nonlinearity">
<p>Variance is a second moment, so one gross outlier can rotate PC1 toward itself. Robust PCA via robust covariance (MCD) or the low-rank-plus-sparse decomposition (Candès et al. 2011) resists this. Curved structure — a circle, a Swiss roll — needs <a href="#/t/manifold-learning">nonlinear methods</a>; kernel PCA (Schölkopf, Smola &amp; Müller 1998) does PCA in a feature space using only the kernel Gram matrix.</p>
</div>
<div class="warn" data-name="Signs and near-ties">
<p>Each $v_j$ is determined only up to sign, and when $\lambda_j\approx\lambda_{j+1}$ the individual eigenvectors are unstable (only their span is well determined; by Davis–Kahan the error scales like noise / eigengap). Do not over-interpret loadings of components with nearly equal variance.</p>
</div>
<p><strong>Sparse PCA</strong> (Zou, Hastie &amp; Tibshirani 2006; d'Aspremont et al. 2007) adds an $\ell_1$ penalty to the loadings, giving interpretable components and consistency in high dimensions when the true loadings are sparse — at the cost of a non-convex, computationally hard problem.</p>`,
      },
    ],
    quiz: [
      { q: R`Prove that maximising captured variance and minimising squared reconstruction error over $q$-dimensional subspaces give the same subspace.`,
        a: R`For an orthogonal projector $P$, $\lVert x_i\rVert^2 = \lVert Px_i\rVert^2 + \lVert(I-P)x_i\rVert^2$. Summing, total scatter $=$ captured scatter $+$ residual, and the total does not depend on $P$. So maximising $\sum\lVert Px_i\rVert^2 = n\tr(PS)$ is equivalent to minimising $\sum\lVert(I-P)x_i\rVert^2$; by Ky Fan, both are solved by the span of the top $q$ eigenvectors.` },
      { q: R`Given the thin SVD $X=UDV^\top$ of the centred data, express the principal directions, the component variances and the scores.`,
        a: R`Directions: columns of $V$. Variances: $d_j^2/n$ (or $d_j^2/(n-1)$). Scores: $XV = UD$, i.e. the $j$-th score vector is $d_ju_j$.` },
      { q: R`With $p=200$ variables, $n=800$ observations and pure white noise of unit variance, roughly where will the largest sample eigenvalue be? Is a population spike of size $\ell=0.4$ detectable?`,
        a: R`$\gamma = 0.25$, so the top eigenvalue sits near $(1+\sqrt{0.25})^2 = 2.25$, though every true eigenvalue is $1$. The BBP threshold is $\sqrt\gamma = 0.5$; $\ell=0.4$ is below it, so asymptotically the top eigenvalue still sits at the bulk edge and the top eigenvector carries no information about the spike direction.` },
      { q: R`In probabilistic PCA, what is $\hat\sigma^2$, and what happens to the posterior mean of $z$ as $\sigma^2\to0$?`,
        a: R`$\hat\sigma^2$ is the average of the $p-q$ discarded eigenvalues of $S$. As $\sigma^2\to0$, $\E[z\mid x] = (W^\top W+\sigma^2I)^{-1}W^\top(x-\mu)\to(W^\top W)^{-1}W^\top(x-\mu)$, the coordinates of the orthogonal projection of $x-\mu$ onto the column space of $W$ — classical PCA.` },
    ],
  });

  /* ===================================================================== FA & ICA */
  Atlas.topic({
    id: 'factor-analysis-ica',
    domain: 'unsupervised',
    order: 70,
    title: 'Factor Analysis & Independent Components',
    subtitle: 'Latent linear models, the rotation that Gaussianity cannot fix, and the non-Gaussianity that fixes it',
    level: 3,
    sources: { aos: '—', cb: '—', other: 'ESL §14.7; Bishop PRML §12.2.4, §12.4.1; Murphy (2012) Ch. 12; Hyvärinen, Karhunen & Oja, Independent Component Analysis (2001); Comon (1994)' },
    prereqs: ['pca', 'em-algorithm'],
    related: ['entropy-kl', 'gaussian-mixtures', 'clt', 'manifold-learning'],
    tags: ['latent variables', 'rotation', 'identifiability', 'negentropy', 'blind source separation'],
    summary: R`Factor analysis explains the correlations among $p$ observed variables by $q \lt p$ unobserved common factors plus variable-specific noise: $x = \mu + \Lambda f + \varepsilon$. Under Gaussian assumptions the loadings $\Lambda$ are identified only up to an orthogonal rotation, since $\Lambda\Lambda^\top$ is all the data can see — hence the long tradition of choosing "interpretable" rotations. Independent component analysis attacks the same indeterminacy from the other side: if the latent sources are independent and <em>non-Gaussian</em>, the rotation is identified (up to order, sign and scale) by Comon's theorem, and it can be found by maximising the non-Gaussianity of projections. Gaussianity is precisely what makes rotations invisible.`,
    survey: {
      task: 'FA: model covariance by few latent factors. ICA: blind source separation, recover independent sources',
      assumptions: 'FA: $x=\\mu+\\Lambda f+\\varepsilon$, Gaussian $f$, diagonal $\\Psi$. ICA: $x = As$, independent sources, at most one Gaussian',
      output: 'FA: loadings $\\Lambda$ (up to rotation), uniquenesses $\\Psi$, factor scores. ICA: unmixing $W\\approx A^{-1}$ and sources',
      strengths: 'FA: scale-equivariant, explicit noise model, likelihood tests. ICA: identifies directions PCA cannot',
      weaknesses: 'FA: rotation indeterminacy, Heywood cases. ICA: order/sign/scale ambiguity; fails with Gaussian sources; noiseless model',
      complexity: 'FA by EM: $O(npq + p^3)$ per iteration on $S$ (or $O(pq^2)$ with Woodbury). FastICA: $O(np)$ per unit per iteration',
    },
    sections: [
      {
        title: 'The factor model',
        html: R`
<div class="def" data-name="Factor analysis model">
<p>For $x\in\R^p$, $q \lt p$ common factors $f\sim\N(0,I_q)$ and independent specific errors $\varepsilon\sim\N(0,\Psi)$ with $\Psi = \diag(\psi_1,\dots,\psi_p)$,
$$x = \mu + \Lambda f + \varepsilon,\qquad\text{so}\qquad x\sim\N(\mu,\ \Sigma),\quad \Sigma = \Lambda\Lambda^\top + \Psi.$$
$\Lambda\in\R^{p\times q}$ holds the <strong>loadings</strong>; $h_j^2 = \sum_k\Lambda_{jk}^2$ is the <em>communality</em> of variable $j$ (variance shared with others) and $\psi_j$ its <em>uniqueness</em>.</p>
</div>
<p>The structural claim is that, conditional on $f$, the observed variables are independent: all covariation is routed through $q$ factors. Off-diagonal covariances are fitted by the low-rank $\Lambda\Lambda^\top$ alone, while the diagonal can absorb variable-specific noise of any size. This is the crucial difference from <a href="#/t/pca">probabilistic PCA</a>, which forces $\Psi = \sigma^2I$.</p>
<div class="cols">
<div><h4>Factor analysis</h4><p>Heteroscedastic diagonal noise. <strong>Scale-equivariant</strong>: rescaling $x_j$ by $c_j$ rescales row $j$ of $\Lambda$ and $\psi_j$, with no change in the fitted structure. Models <em>covariance</em> (the off-diagonals).</p></div>
<div><h4>(P)PCA</h4><p>Isotropic noise. Not scale-equivariant: a variable with large variance dominates. Models <em>variance</em>; a single noisy variable can become its own component.</p></div>
</div>
<div class="note">
<p>A variable with huge idiosyncratic noise but no correlation with anything else becomes PC1 in PCA (if its variance is the largest). In factor analysis it is absorbed into its uniqueness $\psi_j$ and gets loadings near zero. Factor analysis asks "what do the variables have in common?", PCA asks "where is the variance?".</p>
</div>`,
      },
      {
        title: 'Rotation and identifiability',
        html: R`
<p>For any orthogonal $R\in\R^{q\times q}$, $(\Lambda R)(\Lambda R)^\top = \Lambda\Lambda^\top$. Since the Gaussian model's likelihood depends on $\Lambda$ only through $\Sigma$, the loadings $\Lambda$ and $\Lambda R$ are observationally equivalent: <strong>the loadings are identified only up to rotation</strong>. In terms of the generative model, $f' = R^\top f$ is again $\N(0,I_q)$, so rotating the factors produces exactly the same distribution of $x$.</p>
<p>Counting parameters shows when even $\Lambda\Lambda^\top$ and $\Psi$ can be pinned down. $\Sigma$ has $p(p+1)/2$ free entries; the model has $pq + p$ parameters minus $q(q-1)/2$ for the rotation. The difference
$$\text{df} = \tfrac12\big[(p-q)^2 - (p+q)\big]$$
must be non-negative for the model to be a genuine restriction (the Ledermann bound): with $p=6$, at most $q=3$ factors. Anderson &amp; Rubin (1956) gave sufficient conditions for identifiability of $\Lambda\Lambda^\top$ and $\Psi$ (e.g. that deleting any row of $\Lambda$ leaves two disjoint submatrices of rank $q$).</p>
<div class="def" data-name="Rotations">
<p>Since the data cannot choose $R$, the analyst chooses one for interpretability. <strong>Varimax</strong> (Kaiser 1958) maximises the variance of the squared loadings within each factor, $\sum_k\big[\frac1p\sum_j\Lambda_{jk}^4 - (\frac1p\sum_j\Lambda_{jk}^2)^2\big]$, pushing loadings toward $0$ or $\pm$large ("simple structure"). <strong>Oblique</strong> rotations (promax, oblimin) allow correlated factors, $\Cov(f)\neq I$.</p>
</div>
<div class="warn" data-name="Rotation is a choice, not a finding">
<p>Varimax loadings look like discoveries ("factor 1 is verbal ability"), but any rotation fits the data equally well. Interpretations that depend on the rotation are statements about the analyst's criterion, not about the population — unless the factors are non-Gaussian, which is ICA's point of departure.</p>
</div>`,
      },
      {
        title: 'Fitting factor analysis',
        html: R`
<p>The Gaussian log-likelihood with $\hat\mu = \bar x$ is $\ell(\Lambda,\Psi) = -\frac n2\big[\log|\Sigma| + \tr(\Sigma^{-1}S)\big] + \text{const}$, with no closed-form maximiser. Treat $f_i$ as missing data and use <a href="#/t/em-algorithm">EM</a> (Rubin &amp; Thayer 1982). With centred data, $f\mid x$ is Gaussian:
$$\E[f\mid x] = \beta x,\quad \beta = \Lambda^\top\Sigma^{-1},\qquad \E[ff^\top\mid x] = I - \beta\Lambda + \beta xx^\top\beta^\top.$$</p>
<div class="algo" data-name="EM for factor analysis">
<ol>
<li><strong>E-step:</strong> $\beta = \Lambda^\top(\Lambda\Lambda^\top+\Psi)^{-1}$ (via Woodbury, only a $q\times q$ inverse is needed).</li>
<li><strong>M-step:</strong> with sufficient statistic $S$,
$$\Lambda_{\text{new}} = S\beta^\top\big(I - \beta\Lambda + \beta S\beta^\top\big)^{-1},\qquad \Psi_{\text{new}} = \diag\big(S - \Lambda_{\text{new}}\beta S\big).$$</li>
</ol>
</div>
<p>These are the regression of $x$ on the imputed factors: $\Lambda_{\text{new}} = \big(\sum_ix_i\E[f_i\mid x_i]^\top\big)\big(\sum_i\E[f_if_i^\top\mid x_i]\big)^{-1}$, and the uniquenesses are the residual variances. The likelihood ratio test of "$q$ factors" against an unrestricted covariance has $\text{df} = \frac12[(p-q)^2-(p+q)]$ degrees of freedom (Bartlett's correction improves its small-sample behaviour).</p>
<div class="warn" data-name="Heywood cases">
<p>The MLE often drives some $\psi_j\to0$ (a "Heywood case"): the factor model then claims variable $j$ is a perfect linear function of the factors. Causes include too many factors, too small $n$, or model misspecification. Software usually bounds $\psi_j$ away from zero, but a boundary estimate is a warning, and standard errors there are unreliable.</p>
</div>
<p>Factor scores (estimates of $f_i$) are not parameters but predictions of random variables. Thomson's regression scores $\beta x_i$ are the posterior means; Bartlett's scores $(\Lambda^\top\Psi^{-1}\Lambda)^{-1}\Lambda^\top\Psi^{-1}x_i$ are the GLS estimates treating $f_i$ as fixed. Neither recovers $f_i$ exactly, however large $n$ is, because each $f_i$ is informed by only $p$ numbers.</p>`,
      },
      {
        title: 'ICA: why non-Gaussianity identifies the rotation',
        html: R`
<p>The cocktail-party problem: $p$ microphones record $x = As$, linear mixtures of $p$ independent sources $s = (s_1,\dots,s_p)$ (speakers). Recover $s$ without knowing $A$ — "blind" source separation. ICA assumes $A$ is square and invertible and that the components of $s$ are mutually independent.</p>
<p><strong>Step 1, whitening.</strong> Fix scale by assuming $\Cov(s) = I$ (the scale of each source is unidentifiable anyway: $A s = (AC)(C^{-1}s)$ for diagonal $C$). Let $z = \Sigma_x^{-1/2}(x-\mu)$, so $\Cov(z)=I$. Then $z = \tilde As$ with $\tilde A = \Sigma_x^{-1/2}A$ and $\tilde A\tilde A^\top = \Cov(z) = I$: <em>after whitening, the mixing matrix is orthogonal</em>. Second-order statistics have done all they can; a rotation with $p(p-1)/2$ parameters remains.</p>
<p><strong>Step 2, the Gaussian obstruction.</strong> If $s\sim\N(0,I)$ then $Rs\sim\N(0,I)$ for every orthogonal $R$: rotated sources are again independent with the same distribution. No method can recover $\tilde A$. This is the factor-analysis rotation problem again.</p>
<div class="thm" data-kind="Theorem" data-name="Identifiability of ICA (Comon 1994)">
<p>Let $x = As$ with $A$ invertible and $s$ having mutually independent, non-degenerate components, at most one of which is Gaussian. If $y = Wx$ has mutually independent components for some invertible $W$, then $WA = PC$ for a permutation matrix $P$ and an invertible diagonal matrix $C$. That is, the sources are recovered up to order and scale (including sign).</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The key tool is the <strong>Darmois–Skitovich theorem</strong>: if $s_1,\dots,s_p$ are independent and $L_1 = \sum_j a_js_j$, $L_2 = \sum_jb_js_j$ are independent, then every $s_j$ with $a_jb_j\neq0$ is Gaussian. Let $B = WA$, so $y = Bs$. If some column $j$ of $B$ had two nonzero entries, say in rows $k$ and $l$, then $y_k$ and $y_l$ — independent by assumption — would both load on $s_j$, forcing $s_j$ Gaussian. So every non-Gaussian source appears in exactly one $y_k$. With at most one Gaussian source, a short counting argument using invertibility of $B$ shows each column of $B$ has exactly one nonzero entry, i.e. $B = PC$. Two Gaussian sources would break this: any rotation of the pair keeps them independent.</p>
</details>
<div class="note" data-name="Why maximise non-Gaussianity?">
<p>A projection $y = w^\top z$ of the whitened data is $y = q^\top s$ with $q = \tilde A^\top w$, $\lVert q\rVert = 1$. By a CLT heuristic, a sum of several independent sources is "more Gaussian" than any single one. So the projections that are <em>least</em> Gaussian should be those with $q = \pm e_j$, a single source. The next result makes this exact for kurtosis.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Kurtosis contrast">
<p>Let $\kappa_j = \E s_j^4 - 3$ be the excess kurtosis of source $j$ (unit variance), all $\kappa_j\neq0$. For unit $q$, $\kappa(q^\top s) = \sum_jq_j^4\kappa_j$, and $\max_{\lVert q\rVert=1}|\kappa(q^\top s)| = \max_j|\kappa_j|$, attained at $q = \pm e_{j^\star}$, a single source.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The fourth cumulant is additive over independent summands and homogeneous of degree 4: $\kappa_4(\sum_jq_js_j) = \sum_jq_j^4\kappa_4(s_j)$; with unit variance, the fourth cumulant is the excess kurtosis. Then $|\sum_jq_j^4\kappa_j|\le\max_j|\kappa_j|\sum_jq_j^4\le\max_j|\kappa_j|\big(\sum_jq_j^2\big)^2 = \max_j|\kappa_j|$, and $\sum_jq_j^4 = 1$ on the unit sphere only at the coordinate vectors $\pm e_j$.</p>
</details>
<p>Kurtosis is easy to analyse but sensitive to outliers (a fourth moment). Practical ICA uses more robust contrasts, which brings us to negentropy.</p>`,
        lab: 'factor-analysis-ica-unmix',
      },
      {
        title: 'Negentropy, FastICA and maximum likelihood',
        html: R`
<p>Among all distributions with a given variance, the Gaussian has the largest <a href="#/t/entropy-kl">differential entropy</a>. So
$$J(y) = H(y_{\text{gauss}}) - H(y)\ \ge\ 0,$$
the <strong>negentropy</strong>, with $y_{\text{gauss}}$ Gaussian of the same variance, is zero iff $y$ is Gaussian and is invariant to invertible linear transforms of $y$. It is the information-theoretically natural measure of non-Gaussianity. Exact negentropy needs a density estimate, so Hyvärinen (1998) uses the approximation
$$J(y)\ \propto\ \big[\E G(y) - \E G(\nu)\big]^2,\qquad \nu\sim\N(0,1),$$
with a non-quadratic $G$, typically $G(u) = \log\cosh u$ (robust, general-purpose) or $G(u) = -e^{-u^2/2}$. ($G(u)=u^4$ recovers kurtosis.)</p>
<div class="algo" data-name="FastICA, one unit (Hyvärinen 1999)">
<ol>
<li>Whiten the data: $z_i = \hat\Sigma^{-1/2}(x_i - \bar x)$.</li>
<li>Choose a random unit $w$. With $g = G'$ (e.g. $g = \tanh$), iterate the fixed point
$$w\leftarrow\E\big[z\,g(w^\top z)\big] - \E\big[g'(w^\top z)\big]\,w,\qquad w\leftarrow w/\lVert w\rVert,$$
until $|w^\top w_{\text{old}}|\approx1$.</li>
<li>For several components, either deflate (Gram–Schmidt each new $w$ against previous ones) or update all $w$'s in parallel and symmetrically orthogonalise $W\leftarrow(WW^\top)^{-1/2}W$.</li>
</ol>
</div>
<p>The update is an approximate Newton step for optimising $\E G(w^\top z)$ on the sphere, which is why it converges in a handful of iterations without a step size. Under the ICA model its convergence is locally at least quadratic.</p>
<p><strong>Maximum likelihood.</strong> With unmixing $W = A^{-1}$ and source densities $p_j$, the likelihood is
$$\ell(W) = \sum_{i=1}^n\sum_{j=1}^p\log p_j(w_j^\top x_i) + n\log|\det W| .$$
Infomax (Bell &amp; Sejnowski 1995) maximises the same objective. The source densities do not have to be known: it suffices to use the right "type" (sub- vs super-Gaussian), since the score functions only need to correlate correctly with the true ones. Minimising the mutual information $\sum_jH(y_j) - H(y)$ between outputs is equivalent too, because for whitened data and orthogonal $W$ the joint entropy $H(y)$ is constant, leaving $\sum_jH(y_j)$ — i.e. maximising total negentropy.</p>
<div class="warn" data-name="Limits of ICA">
<ul>
<li>Order, sign and scale of the components are arbitrary.</li>
<li>With two or more Gaussian sources, only the non-Gaussian ones are identified; the Gaussian subspace is not.</li>
<li>The basic model is square and noiseless. Noisy ICA and overcomplete ICA ($\#\text{sources} \gt p$) are much harder.</li>
<li>Independence is a strong assumption. Applied to natural data (images, EEG), ICA returns the "most independent" directions available, which need not be physically meaningful.</li>
</ul>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why are factor loadings identified only up to an orthogonal rotation in the Gaussian factor model, and why does this indeterminacy disappear in ICA?`,
        a: R`The distribution of $x$ depends on $\Lambda$ only through $\Lambda\Lambda^\top$, and $(\Lambda R)(\Lambda R)^\top = \Lambda\Lambda^\top$; equivalently $R^\top f\sim\N(0,I)$ is indistinguishable from $f$. In ICA the sources are non-Gaussian: a rotation of independent non-Gaussian variables is no longer independent (Darmois–Skitovich), so only the true rotation (up to permutation and sign) yields independent outputs.` },
      { q: R`For $p=10$ observed variables, what is the largest number of factors for which the factor model imposes restrictions on $\Sigma$?`,
        a: R`Need $(p-q)^2\ge p+q$. $q=6$: $16\ge16$ (df $=0$, just-identified count, no restriction). $q=5$: $25\ge15$, df $=5$. So $q=5$ is the largest with positive degrees of freedom; at $q=6$ the parameter count equals the number of covariances and the model has no testable restriction left by counting.` },
      { q: R`After whitening, why is the remaining mixing matrix orthogonal, and how many parameters remain to be found?`,
        a: R`With $\Cov(s)=I$, $z=\tilde As$ has $\Cov(z) = \tilde A\tilde A^\top$, and whitening makes $\Cov(z) = I$, so $\tilde A$ is orthogonal. An orthogonal $p\times p$ matrix has $p(p-1)/2$ free parameters, which second-order statistics cannot determine; higher-order statistics (non-Gaussianity) must.` },
      { q: R`Show that for independent unit-variance sources and unit $q$, $|\kappa(q^\top s)|\le\max_j|\kappa_j|$, with equality only at a single source.`,
        a: R`Cumulant additivity and homogeneity give $\kappa(q^\top s) = \sum_jq_j^4\kappa_j$. Then $|\sum q_j^4\kappa_j|\le\max|\kappa_j|\sum q_j^4\le\max|\kappa_j|(\sum q_j^2)^2 = \max|\kappa_j|$. The second inequality is equality only if one $q_j^2=1$, i.e. $q=\pm e_j$.` },
      { q: R`A variable with very large idiosyncratic variance, uncorrelated with all others, is included in a data set. How do PCA and factor analysis treat it?`,
        a: R`PCA, if that variable's variance exceeds the other eigenvalues, makes it (nearly) a principal component on its own, since PCA chases variance. Factor analysis absorbs it into the uniqueness $\psi_j$ and gives it near-zero loadings, since it shares no covariance with other variables.` },
    ],
  });

  /* ===================================================================== Manifold learning */
  Atlas.topic({
    id: 'manifold-learning',
    domain: 'unsupervised',
    order: 80,
    title: 'Manifold Learning: MDS, Isomap, t-SNE & UMAP',
    subtitle: 'Low-dimensional maps that preserve distances, geodesics, or neighbourhoods — and what each one gives up',
    level: 3,
    sources: { aos: '—', cb: '—', other: 'ESL §14.8–14.9; Borg & Groenen, Modern Multidimensional Scaling (2005); Tenenbaum, de Silva & Langford (2000); van der Maaten & Hinton (2008); McInnes, Healy & Melville (2018); Murphy (2022) Ch. 20' },
    prereqs: ['pca'],
    related: ['spectral-density-clustering', 'entropy-kl', 'kernel-methods', 'autoencoders-vae', 'hierarchical-clustering'],
    tags: ['MDS', 'double centering', 'geodesic distance', 't-SNE', 'UMAP', 'visualisation'],
    summary: R`When data in $\R^p$ lie near a curved low-dimensional surface, linear projection (PCA) folds distant parts of the surface on top of each other. Manifold-learning methods compute a low-dimensional map that preserves something intrinsic. Classical MDS recovers coordinates from Euclidean distances by double centering and an eigendecomposition. Isomap feeds MDS with geodesic distances estimated by shortest paths in a nearest-neighbour graph, unrolling surfaces like the Swiss roll. t-SNE and UMAP abandon global distances altogether and preserve neighbourhoods, using heavy-tailed similarities in the embedding to counter the crowding problem. The result is striking pictures of cluster structure — and pictures whose sizes and distances should not be read literally.`,
    survey: {
      task: 'Nonlinear dimension reduction; visualisation in 2–3 dimensions',
      assumptions: 'MDS: Euclidean distances. Isomap: isometric to a convex subset of $\\R^d$, dense sampling. t-SNE/UMAP: local neighbourhoods are meaningful',
      output: 'Coordinates $y_1,\\dots,y_n\\in\\R^d$ (no explicit out-of-sample map for MDS/Isomap/t-SNE)',
      strengths: 'MDS/Isomap: global geometry, eigen-solution, no local optima. t-SNE/UMAP: excellent at revealing local cluster structure',
      weaknesses: 'Isomap: short-circuits, holes. t-SNE/UMAP: non-convex; global distances, cluster sizes and densities not preserved',
      complexity: 'MDS $O(n^2)$ memory, $O(n^3)$ eigen; Isomap $+O(n^2\\log n)$ shortest paths; t-SNE $O(n\\log n)$ per iter (Barnes–Hut)',
    },
    sections: [
      {
        title: 'Classical multidimensional scaling',
        html: R`
<p>Suppose all we have is a matrix of pairwise distances $\Delta = (\delta_{ij})$. Can we find points $y_1,\dots,y_n\in\R^d$ with $\lVert y_i-y_j\rVert\approx\delta_{ij}$? If the $\delta_{ij}$ really are Euclidean distances among unknown points $x_i$, classical MDS (Torgerson 1952; Gower 1966) recovers the points exactly, up to rotation and translation.</p>
<p>Let $\Delta^{(2)} = (\delta_{ij}^2)$ and $J = I - \frac1n\mathbf1\mathbf1^\top$ the centring matrix.</p>
<div class="thm" data-kind="Theorem" data-name="Double centering recovers the Gram matrix">
<p>If $\delta_{ij} = \lVert x_i - x_j\rVert$ for points with $\sum_ix_i = 0$, then
$$B := -\tfrac12J\Delta^{(2)}J = XX^\top ,$$
the Gram matrix of inner products $b_{ij} = x_i^\top x_j$. Consequently $B$ is positive semi-definite of rank $\le p$, and with $B = U\Lambda U^\top$ the coordinates $Y = U_d\Lambda_d^{1/2}$ reproduce the configuration up to an orthogonal transformation when $d\ge\text{rank}(B)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\delta_{ij}^2 = \lVert x_i\rVert^2 + \lVert x_j\rVert^2 - 2x_i^\top x_j$, i.e. $\Delta^{(2)} = c\mathbf1^\top + \mathbf1c^\top - 2XX^\top$ with $c_i = \lVert x_i\rVert^2$. Since $J\mathbf1 = 0$, double centering kills the first two terms: $J\Delta^{(2)}J = -2JXX^\top J$. Because the $x_i$ are centred, $JX = X$, so $-\frac12J\Delta^{(2)}J = XX^\top$. If $XX^\top = YY^\top$ with $Y = U_d\Lambda_d^{1/2}$, then $X$ and $Y$ have the same Gram matrix and hence differ by an orthogonal map.</p>
</details>
<p><strong>Schoenberg's theorem</strong> (1935) gives the converse: a symmetric $\Delta$ with zero diagonal is a Euclidean distance matrix iff $-\frac12J\Delta^{(2)}J$ is PSD. For non-Euclidean dissimilarities $B$ has negative eigenvalues, which are simply discarded; the truncated solution minimises the "strain" $\lVert B - YY^\top\rVert_F$ over rank-$d$ $Y$ (Eckart–Young applied to $B$).</p>
<div class="note">
<p>Applied to Euclidean distances, classical MDS returns exactly the <a href="#/t/pca">principal component scores</a>: $XX^\top = UD^2U^\top$, and $U_dD_d$ are the scores. MDS is PCA via the $n\times n$ Gram matrix instead of the $p\times p$ covariance. Everything interesting in nonlinear MDS comes from feeding in <em>different</em> distances.</p>
</div>
<p>Metric MDS instead minimises a <em>stress</em> $\sum_{i \lt j}(\delta_{ij} - \lVert y_i-y_j\rVert)^2$ (by majorisation, SMACOF), optionally weighted — Sammon mapping weights by $1/\delta_{ij}$ to emphasise small distances. Non-metric MDS (Kruskal 1964) preserves only the rank order of dissimilarities.</p>`,
      },
      {
        title: 'Isomap: geodesics through a neighbourhood graph',
        html: R`
<p>On a curved surface such as a Swiss roll, two points on adjacent layers are close in $\R^3$ but far apart <em>along</em> the surface. Euclidean MDS (equivalently PCA) uses the former and squashes the layers together. Isomap (Tenenbaum, de Silva &amp; Langford 2000) estimates the latter.</p>
<div class="algo" data-name="Isomap">
<ol>
<li>Build a neighbourhood graph: connect each point to its $k$ nearest neighbours (or all points within $\varepsilon$), with edge length $\lVert x_i-x_j\rVert$.</li>
<li>Compute all shortest-path distances $\delta^G_{ij}$ in the graph (Dijkstra from every node, $O(n^2(k+\log n))$, or Floyd–Warshall $O(n^3)$).</li>
<li>Apply classical MDS to $\Delta^G$.</li>
</ol>
</div>
<p>Why it works: locally, a smooth manifold is nearly flat, so short Euclidean hops approximate geodesic arc length, and a shortest path made of short hops approximates the geodesic. Bernstein, de Silva, Langford &amp; Tenenbaum (2000) proved that with sufficiently dense sampling (relative to the manifold's curvature and its "branch separation", the minimum distance between points that are far apart along the manifold) graph distances approximate geodesic distances within any prescribed factor $1\pm\epsilon$. If the manifold is <em>isometric to a convex subset of $\R^d$</em>, geodesic distances are Euclidean distances in the intrinsic coordinates and classical MDS recovers those coordinates.</p>
<div class="warn" data-name="Short-circuits and holes">
<p>If $k$ (or $\varepsilon$) is too large relative to the gap between layers, one edge jumps between layers and every shortest path through it collapses — a topological failure, not a small error. If the parameter domain is not convex (a hole in the sheet), geodesics bend around the hole and are longer than intrinsic Euclidean distances, so MDS distorts the embedding. Isomap also cannot unroll intrinsically curved surfaces (a sphere has no isometric flat embedding). The lab lets you increase $k$ until the roll short-circuits.</p>
</div>
<div class="cols">
<div><h4>Locally linear embedding</h4><p>Reconstruct each point from its neighbours with weights summing to one, then find low-dimensional points reconstructed by the same weights; a sparse eigenproblem (Roweis &amp; Saul 2000).</p></div>
<div><h4>Laplacian eigenmaps</h4><p>Bottom nonzero eigenvectors of a neighbourhood-graph Laplacian; minimises $\sum w_{ij}\lVert y_i-y_j\rVert^2$ under a scale constraint (Belkin &amp; Niyogi 2003). The graph Laplacian converges to the Laplace–Beltrami operator of the manifold. Close cousin of <a href="#/t/spectral-density-clustering">spectral clustering</a>; diffusion maps (Coifman &amp; Lafon 2006) generalise it.</p></div>
</div>`,
        lab: 'manifold-learning-isomap',
      },
      {
        title: 't-SNE: matching neighbourhood distributions',
        html: R`
<p>SNE (Hinton &amp; Roweis 2002) and t-SNE (van der Maaten &amp; Hinton 2008) give up on distances and preserve <em>who is near whom</em>, expressed as probability distributions.</p>
<p><strong>High-dimensional affinities.</strong> For each $i$, a Gaussian kernel defines a conditional distribution over neighbours,
$$p_{j\mid i} = \frac{\exp(-\lVert x_i-x_j\rVert^2/2\sigma_i^2)}{\sum_{k\neq i}\exp(-\lVert x_i-x_k\rVert^2/2\sigma_i^2)},\qquad p_{i\mid i}=0,$$
with $\sigma_i$ chosen by bisection so that the <strong>perplexity</strong> $2^{H(P_i)}$, $H(P_i) = -\sum_jp_{j\mid i}\log_2p_{j\mid i}$, equals a user value (typically 5–50): an effective number of neighbours. Adaptive $\sigma_i$ means dense regions get small bandwidths and sparse regions large ones. Symmetrise: $p_{ij} = (p_{j\mid i}+p_{i\mid j})/2n$, so $\sum_{i\neq j}p_{ij}=1$.</p>
<p><strong>Low-dimensional affinities</strong> use a Student-$t$ kernel with one degree of freedom (a Cauchy kernel):
$$q_{ij} = \frac{(1+\lVert y_i-y_j\rVert^2)^{-1}}{\sum_{k\neq l}(1+\lVert y_k-y_l\rVert^2)^{-1}} .$$</p>
<p><strong>Objective:</strong> minimise $C(Y) = \KL(P\,\Vert\,Q) = \sum_{i\neq j}p_{ij}\log(p_{ij}/q_{ij})$ by gradient descent with momentum, where
$$\frac{\partial C}{\partial y_i} = 4\sum_{j\neq i}(p_{ij}-q_{ij})\,\frac{y_i-y_j}{1+\lVert y_i-y_j\rVert^2} .$$
Each pair acts as a spring: attractive if $p_{ij} \gt q_{ij}$ (too far apart in the map), repulsive otherwise.</p>
<div class="note" data-name="What KL(P‖Q) cares about">
<p>The KL divergence is asymmetric. A pair with large $p_{ij}$ (true neighbours) and small $q_{ij}$ (placed far apart) costs a lot: $p_{ij}\log(p_{ij}/q_{ij})$ is large. A pair with small $p_{ij}$ and large $q_{ij}$ (non-neighbours placed together) costs little. So t-SNE strongly preserves local neighbourhoods and only weakly penalises false neighbours or misplaced distant points. Global geometry is not part of the objective.</p>
</div>
<h4>The crowding problem and the heavy tail</h4>
<p>In $p$ dimensions, a point can have many equidistant neighbours (up to $p+1$ mutually equidistant points), and the volume of a ball at radius $r$ grows like $r^p$, so there is far more room at moderate distance than in 2D. Mapping a $p$-dimensional neighbourhood into the plane, the moderately distant points cannot all keep their distances: there is not enough area. With a Gaussian kernel in the map (symmetric SNE), they get crushed toward the centre and clusters merge into a central blob. The Cauchy kernel's heavy tail means a moderate dissimilarity in $P$ is matched by a <em>much larger</em> distance in the map: for large distances $q\propto\lVert y_i-y_j\rVert^{-2}$, so the map can push dissimilar points far apart at little cost. This both relieves crowding and produces the well-separated clusters t-SNE is known for. The lab has a toggle to swap the Cauchy kernel for a Gaussian one.</p>
<p><strong>Optimisation details matter.</strong> "Early exaggeration" multiplies $P$ by a factor (e.g. 12) for the first few hundred iterations, which makes clusters form tight, well-separated groups early; Arora, Hu &amp; Kothari (2018) and Linderman &amp; Steinerberger (2019) analyse this phase as a spectral-clustering-like dynamic. Barnes–Hut (van der Maaten 2014) and FFT-interpolation (Linderman et al. 2019) approximations reduce the cost from $O(n^2)$ to $O(n\log n)$ or $O(n)$ per iteration.</p>`,
        lab: 'manifold-learning-tsne',
      },
      {
        title: 'UMAP in brief',
        html: R`
<p>UMAP (McInnes, Healy &amp; Melville 2018) has the same overall shape — high-dimensional neighbour affinities, low-dimensional heavy-tailed affinities, and an optimisation matching the two — with different choices motivated by fuzzy simplicial sets:</p>
<ul>
<li>High-dimensional weights on a $k$-NN graph: $v_{j\mid i} = \exp\!\big(-(d(x_i,x_j)-\rho_i)/\sigma_i\big)$, where $\rho_i$ is the distance to the nearest neighbour (so every point has at least one neighbour with weight 1) and $\sigma_i$ is set so that $\sum_jv_{j\mid i} = \log_2k$. Symmetrised by fuzzy union: $v_{ij} = v_{j\mid i}+v_{i\mid j}-v_{j\mid i}v_{i\mid j}$.</li>
<li>Low-dimensional weights $w_{ij} = (1 + a\lVert y_i-y_j\rVert^{2b})^{-1}$, with $a,b$ fitted from a "min_dist" parameter ($a=b=1$ is the t-SNE kernel).</li>
<li>Objective: a fuzzy-set cross-entropy $\sum_{i\neq j}\big[v_{ij}\log\frac{v_{ij}}{w_{ij}} + (1-v_{ij})\log\frac{1-v_{ij}}{1-w_{ij}}\big]$, <em>without</em> a global normalisation, optimised by stochastic gradient descent with negative sampling, which makes it fast and lets it scale to millions of points.</li>
</ul>
<p>UMAP typically runs faster than t-SNE and is often said to preserve more global structure, but later analyses (Kobak &amp; Linderman 2021) traced much of that difference to initialisation (UMAP starts from a Laplacian eigenmap; t-SNE traditionally from random noise), and Damrich &amp; Hamprecht (2021) showed that negative sampling means UMAP effectively optimises a different objective from the stated one. With matched initialisation and attraction–repulsion balance, the two behave similarly.</p>`,
      },
      {
        title: 'Reading embeddings responsibly',
        html: R`
<div class="warn" data-name="What t-SNE and UMAP plots do not show">
<ul>
<li><strong>Cluster sizes</strong> are not meaningful. Adaptive bandwidths equalise local densities: a tight cluster and a diffuse one come out about the same size.</li>
<li><strong>Distances between clusters</strong> are at best weakly meaningful. Two clusters far apart in the map may be close in the data, and vice versa.</li>
<li><strong>Apparent clusters can come from noise</strong>, particularly at low perplexity, and continuous structure (a trajectory, a gradient) can be broken into islands.</li>
<li><strong>Different runs and perplexities give different pictures.</strong> Always look at several.</li>
</ul>
<p>See Wattenberg, Viégas &amp; Johnson, "How to use t-SNE effectively" (Distill, 2016), for many illustrations.</p>
</div>
<p>A reasonable rule: use t-SNE/UMAP to <em>generate hypotheses</em> about local structure, and test them with methods whose output has a definite meaning (clustering in the original space, distances, held-out prediction). Quantities computed <em>on</em> the 2D embedding — cluster areas, distances, density — should not be used as data.</p>
<table>
<thead><tr><th>Method</th><th>Preserves</th><th>Solution</th><th>Out-of-sample</th></tr></thead>
<tbody>
<tr><td>PCA / classical MDS</td><td>Global Euclidean distances (linear)</td><td>Eigenvectors; unique</td><td>Projection</td></tr>
<tr><td>Isomap</td><td>Geodesic distances</td><td>Eigenvectors; unique</td><td>Nyström extension</td></tr>
<tr><td>Laplacian eigenmaps / LLE</td><td>Local neighbourhoods (smoothness)</td><td>Sparse eigenproblem</td><td>Nyström extension</td></tr>
<tr><td>t-SNE</td><td>Neighbour probabilities, strongly local</td><td>Non-convex gradient descent</td><td>None (parametric variants exist)</td></tr>
<tr><td>UMAP</td><td>Fuzzy $k$-NN graph</td><td>SGD with negative sampling</td><td>Approximate transform</td></tr>
<tr><td><a href="#/t/autoencoders-vae">Autoencoder</a></td><td>Reconstruction</td><td>SGD; local optima</td><td>Encoder network</td></tr>
</tbody>
</table>`,
      },
    ],
    quiz: [
      { q: R`Show that double centering the squared-distance matrix of centred points gives their Gram matrix.`,
        a: R`$\Delta^{(2)} = c\mathbf1^\top + \mathbf1c^\top - 2XX^\top$ with $c_i = \lVert x_i\rVert^2$. Since $J\mathbf1 = 0$, $J\Delta^{(2)}J = -2JXX^\top J = -2XX^\top$ because $JX = X$ for centred $X$. Hence $-\frac12J\Delta^{(2)}J = XX^\top$.` },
      { q: R`Why does Isomap fail catastrophically, rather than gradually, when $k$ is too large on a Swiss roll?`,
        a: R`A single neighbourhood edge between adjacent layers creates a shortcut, and every shortest path that can use it becomes much shorter, changing the global distance structure at once. Geodesic estimates for a large fraction of pairs collapse, so MDS folds the roll back onto itself — a topological error, not a small perturbation.` },
      { q: R`What is the crowding problem, and how does the Student-$t$ kernel in t-SNE address it?`,
        a: R`In high dimensions there is far more volume at moderate distance than in 2D, so a 2D map cannot keep all moderately distant neighbours at faithful distances; with a Gaussian map kernel they are crushed together and clusters merge. The heavy-tailed Cauchy kernel lets a moderate high-dimensional dissimilarity correspond to a much larger map distance (its similarity decays polynomially), so dissimilar points can be placed far apart cheaply, creating room and separating clusters.` },
      { q: R`Two clusters in a t-SNE plot have very different areas. What can you conclude about their spreads in the original space?`,
        a: R`Essentially nothing. Perplexity calibration chooses $\sigma_i$ per point to give a fixed effective number of neighbours, which equalises apparent density: tight and diffuse clusters are rescaled to similar map sizes. Compare spreads in the original space.` },
    ],
  });

  /* ===================================================================== Directed graphical models */
  Atlas.topic({
    id: 'directed-graphical-models',
    domain: 'unsupervised',
    order: 90,
    title: 'Directed Graphical Models',
    subtitle: 'Bayesian networks: factorisation along a DAG, d-separation, and what arrows do and do not mean',
    level: 2,
    sources: { aos: 'Ch. 17', cb: '—', other: 'Koller & Friedman, Probabilistic Graphical Models (2009) Ch. 3, 21; Bishop PRML §8.1–8.2; Lauritzen, Graphical Models (1996); Pearl, Causality (2009); Murphy (2012) Ch. 10' },
    prereqs: ['conditional-probability', 'multivariate'],
    related: ['undirected-graphical-models', 'causal-inference', 'hidden-markov', 'naive-bayes', 'gibbs-sampling'],
    tags: ['Bayesian network', 'DAG', 'd-separation', 'Markov blanket', 'v-structure', 'causal'],
    summary: R`A directed acyclic graph (DAG) encodes a factorisation of a joint distribution into one conditional per variable given its parents, $p(x) = \prod_v p(x_v\mid x_{\text{pa}(v)})$. The graph then implies a set of conditional independencies that can be read off by a purely graphical criterion, d-separation: chains and forks are blocked by conditioning on the middle node, while a collider is blocked unless it or a descendant is conditioned on — the source of "explaining away". These rules make high-dimensional models tractable to specify, estimate and compute with. Read causally, a DAG says more: it predicts the effect of interventions, which the probabilistic reading alone does not license.`,
    survey: {
      task: 'Represent and reason with a joint distribution over many variables; causal modelling',
      assumptions: 'Each variable depends on the rest only through its parents (local Markov property); acyclicity',
      output: 'Factorised joint; implied conditional independencies; posterior marginals via inference algorithms',
      strengths: 'Modular, interpretable, compact; parameters estimable locally; natural generative and causal semantics',
      weaknesses: 'Structure learning hard and identified only up to Markov equivalence; exact inference NP-hard in general',
      complexity: 'd-separation $O(|V|+|E|)$; exact inference exponential in treewidth',
    },
    sections: [
      {
        title: 'Factorisation along a DAG',
        html: R`
<p>Let $G = (V,E)$ be a directed acyclic graph whose vertices index random variables $X_v$. Write $\text{pa}(v)$ for the parents of $v$, $\text{de}(v)$ for its descendants, and $\text{nd}(v) = V\setminus(\text{de}(v)\cup\{v\})$ for its non-descendants.</p>
<div class="def" data-name="Bayesian network">
<p>A distribution $P$ with density (or mass function) $p$ <strong>factorises according to $G$</strong> if
$$p(x) = \prod_{v\in V}p\big(x_v\mid x_{\text{pa}(v)}\big).$$
The pair $(G,\{p(x_v\mid x_{\text{pa}(v)})\})$ is a Bayesian network; the factors are its conditional probability distributions (CPDs).</p>
</div>
<p>Any joint distribution factorises according to a <em>complete</em> DAG, by the chain rule $p(x) = \prod_jp(x_j\mid x_1,\dots,x_{j-1})$ in any order. The content of a DAG lies in its <em>missing</em> edges: each says some variable does not need some predecessor once its parents are known. The payoff is economy: $n$ binary variables need $2^n-1$ numbers in general but only $\sum_v2^{|\text{pa}(v)|}$ under the DAG.</p>
<div class="thm" data-kind="Theorem" data-name="Factorisation ⇔ local Markov property">
<p>$p$ factorises according to $G$ if and only if every variable is independent of its non-descendants given its parents: $X_v\indep X_{\text{nd}(v)}\mid X_{\text{pa}(v)}$ for all $v$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>($\Leftarrow$) Order the vertices topologically (parents before children). By the chain rule $p(x) = \prod_vp(x_v\mid x_{\text{pred}(v)})$, where $\text{pred}(v)$ are the earlier vertices. In a topological order $\text{pa}(v)\subseteq\text{pred}(v)\subseteq\text{nd}(v)$, so the local Markov property gives $p(x_v\mid x_{\text{pred}(v)}) = p(x_v\mid x_{\text{pa}(v)})$. ($\Rightarrow$) Fix $v$. Choose a topological order in which all of $\text{nd}(v)$ comes before $v$ and all descendants after (possible since no non-descendant is a descendant of $v$). Summing the factorisation over the descendants, leaf by leaf in reverse order (each sum of a CPD over its own variable is 1), leaves $p(x_v, x_{\text{nd}(v)}) = p(x_v\mid x_{\text{pa}(v)})\prod_{u\in\text{nd}(v)}p(x_u\mid x_{\text{pa}(u)})$, where the product does not involve $x_v$ (no non-descendant has $v$ as a parent). Dividing by the marginal of $x_{\text{nd}(v)}$ shows $p(x_v\mid x_{\text{nd}(v)}) = p(x_v\mid x_{\text{pa}(v)})$.</p>
</details>
<div class="ex" data-name="Familiar models as DAGs">
<p><a href="#/t/naive-bayes">Naive Bayes</a> is $Y\to X_1,\dots,Y\to X_p$. A <a href="#/t/hidden-markov">hidden Markov model</a> is a chain $Z_1\to Z_2\to\cdots$ with emissions $Z_t\to X_t$. A <a href="#/t/gaussian-mixtures">mixture model</a> is $Z_i\to X_i$ with parameter nodes shared across $i$, drawn compactly with a <em>plate</em>. A hierarchical Bayesian model is a DAG whose top nodes are hyperparameters.</p>
</div>`,
      },
      {
        title: 'Three building blocks: chains, forks and colliders',
        html: R`
<p>Every path in a DAG is built from consecutive triples. Their behaviour is the whole of d-separation.</p>
<div class="cols">
<div><h4>Chain $A\to B\to C$</h4><p>$p = p(a)p(b\mid a)p(c\mid b)$. Then $p(c\mid a,b) = p(c\mid b)$: $A\indep C\mid B$. Marginally $A$ and $C$ are typically dependent. Conditioning on the middle <strong>blocks</strong>.</p></div>
<div><h4>Fork $A\leftarrow B\to C$</h4><p>$p = p(b)p(a\mid b)p(c\mid b)$, so $p(a,c\mid b) = p(a\mid b)p(c\mid b)$: $A\indep C\mid B$. A common cause induces marginal dependence. Conditioning <strong>blocks</strong>.</p></div>
<div><h4>Collider $A\to B\leftarrow C$</h4><p>$p = p(a)p(c)p(b\mid a,c)$. Summing out $b$: $p(a,c) = p(a)p(c)$, so $A\indep C$ marginally. But $p(a,c\mid b)\propto p(a)p(c)p(b\mid a,c)$ does not factor in general. Conditioning on the collider (or a descendant) <strong>opens</strong> the path.</p></div>
</div>
<div class="ex" data-name="Explaining away">
<p>Let $A$ = "sprinkler on", $C$ = "it rained", independent, and $B$ = "grass wet" $= A\lor C$. Marginally $A\indep C$. Given $B=1$, learning $C=1$ makes $A=1$ <em>less</em> likely: rain already explains the wet grass. With $\P(A)=\P(C)=\tfrac12$: $\P(A=1\mid B=1) = \frac{1/2}{3/4} = \frac23$, but $\P(A=1\mid B=1,C=1) = \frac12$. Conditioning on a common effect induces negative dependence between causes. The same mechanism produces selection bias (Berkson's paradox): if a hospital admits patients who have disease $A$ or disease $C$, the two diseases appear negatively associated among inpatients.</p>
</div>`,
      },
      {
        title: 'd-separation and the global Markov property',
        html: R`
<div class="def" data-name="d-separation (Pearl 1988)">
<p>A path (a sequence of distinct adjacent vertices, ignoring edge directions) between $X$ and $Y$ is <strong>blocked</strong> by a set $Z$ if it contains a vertex $m$ such that either
(i) $m$ is a chain or fork vertex on the path ($\to m\to$, $\leftarrow m\leftarrow$, or $\leftarrow m\to$) and $m\in Z$; or
(ii) $m$ is a collider on the path ($\to m\leftarrow$) and neither $m$ nor any descendant of $m$ is in $Z$.
Sets $A$ and $B$ are <strong>d-separated</strong> by $Z$, written $A\perp_GB\mid Z$, if every path from a vertex of $A$ to a vertex of $B$ is blocked. A path that is not blocked is <em>active</em>.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Soundness and completeness of d-separation (Verma & Pearl 1988; Geiger & Pearl 1990; Meek 1995)">
<p>(a) <strong>Soundness (global Markov property):</strong> if $p$ factorises according to $G$, then $A\perp_GB\mid Z$ implies $X_A\indep X_B\mid X_Z$. (b) <strong>Completeness:</strong> if $A$ and $B$ are not d-separated by $Z$, there is a distribution factorising according to $G$ in which $X_A\not\indep X_B\mid X_Z$. In fact, for discrete or linear-Gaussian networks, the set of parameter values for which some such dependence vanishes has Lebesgue measure zero (Meek 1995), so d-separation characterises the independencies of almost every distribution factorising over $G$.</p>
</div>
<details class="proof"><summary>Proof sketch of soundness</summary>
<p>Use the equivalent <strong>moralisation criterion</strong> (Lauritzen, Dawid, Larsen &amp; Leimer 1990): $A\perp_GB\mid Z$ iff $Z$ separates $A$ from $B$ in the moral graph of the ancestral set $\text{An}(A\cup B\cup Z)$ — the undirected graph obtained by keeping only these vertices and their ancestors, "marrying" all pairs of parents with a common child, and dropping directions. The marginal of $p$ on an ancestral set factorises over the induced sub-DAG (sum out non-ancestors from the leaves upward). Each factor $p(x_v\mid x_{\text{pa}(v)})$ is a function of a clique $\{v\}\cup\text{pa}(v)$ of the moral graph, so the marginal factorises over that undirected graph, and the factorisation-implies-separation argument for <a href="#/t/undirected-graphical-models">undirected models</a> gives the conditional independence. The equivalence of the two criteria is a combinatorial check: marrying parents is what opens colliders, and restricting to ancestors is what keeps colliders with no conditioned descendants closed.</p>
</details>
<p>The theorem means the graph is a complete, sound summary of the model's conditional independence structure, and that it can be queried by graph algorithms without touching a single probability.</p>
<div class="thm" data-kind="Proposition" data-name="Markov blanket">
<p>The Markov blanket of $v$ — its parents, its children, and its children's other parents (co-parents) — renders $X_v$ independent of all other variables: $p(x_v\mid x_{V\setminus v})\propto p(x_v\mid x_{\text{pa}(v)})\prod_{c\in\text{ch}(v)}p(x_c\mid x_{\text{pa}(c)})$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$p(x_v\mid x_{V\setminus v}) = p(x)/\sum_{x_v}p(x)$. In the factorisation, the only factors involving $x_v$ are $v$'s own CPD and those of its children; all other factors cancel between numerator and denominator. The surviving factors involve exactly $v$, its parents, its children and the children's parents.</p>
</details>
<p>This is what makes <a href="#/t/gibbs-sampling">Gibbs sampling</a> on Bayesian networks local: each full conditional needs only the Markov blanket.</p>`,
        lab: 'directed-graphical-models-dsep',
      },
      {
        title: 'Testing d-separation: Bayes-ball',
        html: R`
<p>Enumerating all paths is exponential. Shachter's (1998) <strong>Bayes-ball</strong> algorithm decides d-separation in linear time by a reachability search over (vertex, direction) states. Imagine a ball sent from $X$; it can pass through, bounce off, or be blocked by each vertex according to the three building blocks:</p>
<table>
<thead><tr><th>Vertex $m$</th><th>Ball arrives from a child (travelling up)</th><th>Ball arrives from a parent (travelling down)</th></tr></thead>
<tbody>
<tr><td>$m\notin Z$</td><td>Passes to parents and bounces to other children (chain upward / fork)</td><td>Passes to children only (chain); does not go to other parents (closed collider)</td></tr>
<tr><td>$m\in Z$</td><td>Blocked (conditioned chain/fork)</td><td>Bounces back to parents (opened collider)</td></tr>
</tbody>
</table>
<p>The rule for a ball from a parent at an unconditioned $m$ that has a conditioned descendant is handled automatically: the ball travels down to the conditioned descendant, bounces back up, and reaches $m$'s other parents through the "from a child" rule. $Y$ is d-connected to $X$ given $Z$ iff the ball can reach $Y$. Each (vertex, direction) state is visited at most once, so the cost is $O(|V|+|E|)$.</p>
<div class="algo" data-name="Reachable set (Koller & Friedman, Alg. 3.1, in outline)">
<ol>
<li>Phase I: mark $Z$ and all ancestors of $Z$ (these are the colliders that will be open).</li>
<li>Phase II: breadth-first search over states $(v,\uparrow)$ / $(v,\downarrow)$ starting at $(X,\uparrow)$. From $(v,\uparrow)$ with $v\notin Z$: go to parents as $\uparrow$ and children as $\downarrow$. From $(v,\downarrow)$: if $v\notin Z$ go to children as $\downarrow$; if $v$ is marked (in $Z$ or an ancestor of $Z$) go to parents as $\uparrow$.</li>
<li>Every vertex visited (and not in $Z$) is d-connected to $X$ given $Z$.</li>
</ol>
</div>`,
      },
      {
        title: 'Markov equivalence and structure learning',
        html: R`
<p>Different DAGs can encode exactly the same independencies: $A\to B\to C$, $A\leftarrow B\leftarrow C$ and $A\leftarrow B\to C$ all say only $A\indep C\mid B$. Only the collider $A\to B\leftarrow C$ differs.</p>
<div class="thm" data-kind="Theorem" data-name="Markov equivalence (Verma & Pearl 1990)">
<p>Two DAGs imply the same set of d-separations iff they have the same skeleton (undirected edges) and the same v-structures (colliders $a\to b\leftarrow c$ with $a$ and $c$ non-adjacent).</p>
</div>
<p>Consequently observational data, however plentiful, can identify a DAG only up to its Markov equivalence class, represented by a CPDAG (completed partially directed graph) in which edges whose direction varies within the class are left undirected. Structure-learning algorithms either test conditional independencies (the PC algorithm of Spirtes &amp; Glymour, consistent under faithfulness) or search over graphs maximising a penalised score (BIC, BDe), e.g. greedy equivalence search (Chickering 2002). Learning the optimal structure by score is NP-hard (Chickering 1996).</p>
<div class="note" data-name="Parameter learning is easy">
<p>With complete data, the log-likelihood of a Bayesian network splits into one term per CPD: $\ell(\theta) = \sum_v\sum_i\log p(x_{iv}\mid x_{i,\text{pa}(v)};\theta_v)$. Each $\theta_v$ is estimated separately — counts for discrete CPDs, a <a href="#/t/simple-linear-regression">regression</a> of the child on its parents for linear-Gaussian CPDs. With latent variables, use <a href="#/t/em-algorithm">EM</a>.</p>
</div>
<p><strong>Inference</strong> — computing $p(x_A\mid x_E)$ — is NP-hard in general (Cooper 1990), but variable elimination and the junction-tree algorithm are exact with cost exponential in the treewidth of the moral graph, linear for trees and polytrees (belief propagation, Pearl 1988). Otherwise one uses <a href="#/t/mcmc-foundations">MCMC</a> or <a href="#/t/variational-inference">variational</a> approximations.</p>`,
      },
      {
        title: 'Probabilistic versus causal DAGs',
        html: R`
<p>Everything so far concerns the <em>probabilistic</em> reading: the DAG is a bookkeeping device for conditional independencies, and any member of a Markov equivalence class is as good as another. An arrow $A\to B$ does not mean $A$ causes $B$ — $B\to A$ might encode exactly the same distribution.</p>
<p>A <strong>causal</strong> DAG asserts more: each CPD $p(x_v\mid x_{\text{pa}(v)})$ is an autonomous mechanism that would remain unchanged if we intervened on other variables. An intervention $\text{do}(X_A = a)$ replaces the mechanisms of the intervened variables by constants and leaves the rest alone, giving the <strong>truncated factorisation</strong>
$$p\big(x\mid\text{do}(X_A=a)\big) = \prod_{v\notin A}p\big(x_v\mid x_{\text{pa}(v)}\big)\Big|_{x_A=a}.$$
In general $p(y\mid\text{do}(x))\neq p(y\mid x)$: conditioning selects units with $X=x$, including whatever common causes made them so; intervening sets $X$ regardless. When a set $S$ of non-descendants of $X$ blocks every "back-door" path from $X$ to $Y$ (those starting with an arrow into $X$), the <strong>back-door adjustment</strong> gives $p(y\mid\text{do}(x)) = \sum_sp(y\mid x,s)p(s)$ (Pearl 1993). This is the bridge to <a href="#/t/causal-inference">causal inference</a>.</p>
<div class="warn" data-name="Two different claims">
<p>A DAG learned from observational data is a claim about conditional independencies, identifiable only up to Markov equivalence and only under faithfulness. Reading its arrows causally requires additional assumptions: no unmeasured common causes (causal sufficiency), correct temporal or mechanistic knowledge, or interventional data. Conditioning on a collider — for example by selecting a sample on an outcome — can create associations that no causal story supports.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In the DAG $A\to B\leftarrow C$, $B\to D$, is $A\indep C\mid D$? Explain with d-separation.`,
        a: R`No (generically). The only path $A\to B\leftarrow C$ has the collider $B$. $B\notin\{D\}$, but its descendant $D$ is conditioned on, so the collider is open and the path is active; $A$ and $C$ are d-connected given $D$. Observing an effect of the common effect induces dependence between its causes.` },
      { q: R`What is the Markov blanket of a node in a DAG, and why does it include co-parents?`,
        a: R`Parents, children, and the other parents of its children. The full conditional of $x_v$ is proportional to $p(x_v\mid x_{\text{pa}(v)})\prod_{c\in\text{ch}(v)}p(x_c\mid x_{\text{pa}(c)})$, and each child's CPD involves the child's other parents. Graphically, conditioning on a child opens the collider between $v$ and a co-parent, so co-parents must be conditioned on too.` },
      { q: R`Which of these DAGs are Markov equivalent: (i) $A\to B\to C$, (ii) $A\leftarrow B\to C$, (iii) $A\to B\leftarrow C$, (iv) $A\leftarrow B\leftarrow C$?`,
        a: R`(i), (ii) and (iv) share the skeleton $A-B-C$ and have no v-structure, so they are equivalent (all encode only $A\indep C\mid B$). (iii) has a v-structure at $B$ with $A$, $C$ non-adjacent; it encodes $A\indep C$ marginally and not given $B$, so it is in a class of its own.` },
      { q: R`In the DAG $Z\to X\to Y$, $Z\to Y$, why does $p(y\mid\text{do}(x))$ generally differ from $p(y\mid x)$, and how do you compute it?`,
        a: R`Conditioning on $X=x$ also changes the distribution of the confounder $Z$ (units with that $x$ tend to have particular $z$), and $Z$ affects $Y$ directly. Intervening leaves $Z$'s distribution unchanged. $\{Z\}$ blocks the back-door path $X\leftarrow Z\to Y$, so $p(y\mid\text{do}(x)) = \sum_zp(y\mid x,z)p(z)$, whereas $p(y\mid x) = \sum_zp(y\mid x,z)p(z\mid x)$.` },
    ],
  });

  /* ===================================================================== Undirected graphical models */
  Atlas.topic({
    id: 'undirected-graphical-models',
    domain: 'unsupervised',
    order: 100,
    title: 'Undirected Graphical Models & the Graphical Lasso',
    subtitle: 'Markov random fields, the Hammersley–Clifford theorem, and conditional independence as zeros of a precision matrix',
    level: 3,
    sources: { aos: 'Ch. 18', cb: '—', other: 'ESL Ch. 17; Lauritzen, Graphical Models (1996); Koller & Friedman (2009) Ch. 4, 7; Friedman, Hastie & Tibshirani (2008); Wainwright & Jordan (2008); Murphy (2012) Ch. 19, 26' },
    prereqs: ['directed-graphical-models', 'multivariate'],
    related: ['lasso', 'gibbs-sampling', 'variational-inference', 'exponential-families', 'mle'],
    tags: ['Markov random field', 'Hammersley–Clifford', 'Gaussian graphical model', 'precision matrix', 'graphical lasso', 'Ising'],
    summary: R`An undirected graph encodes conditional independence by separation: if every path from $A$ to $B$ passes through $S$, then $X_A\indep X_B\mid X_S$. The Hammersley–Clifford theorem says that, for strictly positive distributions, this Markov property is equivalent to the joint density factorising into non-negative potentials over the cliques of the graph. For a multivariate Gaussian the graph is read directly from the precision matrix $\Omega = \Sigma^{-1}$: $X_u\indep X_v$ given all the others exactly when $\Omega_{uv}=0$. Estimating a sparse precision matrix, by the convex graphical lasso, therefore estimates the graph, and works even when $p \gt n$.`,
    survey: {
      task: 'Model and estimate conditional dependence structure (networks) among many variables',
      assumptions: 'Positivity (for Hammersley–Clifford); Gaussianity for GGM; sparsity of $\\Omega$ for high-dimensional estimation',
      output: 'Graph (edge set) and potentials; for GGM, sparse precision $\\hat\\Omega$ and partial correlations',
      strengths: 'Symmetric relations (no arrows needed); convex estimation for GGM; scales to $p\\gg n$',
      weaknesses: 'Partition function intractable for discrete MRFs; Gaussian assumption; $\\lambda$ choice; conditional, not causal',
      complexity: 'Graphical lasso $O(p^3)$ per sweep (much less with sparsity / screening)',
    },
    sections: [
      {
        title: 'Markov properties on undirected graphs',
        html: R`
<p>Let $G=(V,E)$ be an undirected graph and $X = (X_v)_{v\in V}$. For disjoint $A,B,S\subseteq V$, $S$ <strong>separates</strong> $A$ from $B$ if every path from $A$ to $B$ passes through $S$. Unlike d-separation there are no colliders: conditioning never opens a path.</p>
<div class="def" data-name="Markov properties">
<p>$P$ satisfies the <strong>pairwise</strong> Markov property if $X_u\indep X_v\mid X_{V\setminus\{u,v\}}$ for every non-adjacent pair; the <strong>local</strong> property if $X_v\indep X_{V\setminus(\{v\}\cup\text{ne}(v))}\mid X_{\text{ne}(v)}$ for all $v$ (the neighbours form the Markov blanket); the <strong>global</strong> property if $X_A\indep X_B\mid X_S$ whenever $S$ separates $A$ from $B$.</p>
</div>
<p>Always global $\Rightarrow$ local $\Rightarrow$ pairwise. The reverse implications hold when $P$ has a strictly positive density, because then the <em>intersection property</em> holds: $X\indep Y\mid(Z,W)$ and $X\indep W\mid(Z,Y)$ imply $X\indep(Y,W)\mid Z$ (Pearl 1988; Lauritzen 1996). Without positivity they can fail: if $X=Y=W$ almost surely, then $X\indep Y\mid W$ and $X\indep W\mid Y$ (each is a.s. constant given the other) yet $X$ and $(Y,W)$ are dependent.</p>
<div class="note">
<p>The pairwise property is the one we can estimate edge by edge (is $u$ informative about $v$ beyond everything else?); the global property is the one we want to use (separation sets give conditional independence of whole blocks). Positivity connects them.</p>
</div>`,
      },
      {
        title: 'Gibbs factorisation and the Hammersley–Clifford theorem',
        html: R`
<div class="def" data-name="Gibbs distribution / Markov random field">
<p>$P$ <strong>factorises over $G$</strong> if its density can be written
$$p(x) = \frac1Z\prod_{C\in\mathcal C}\psi_C(x_C),\qquad Z = \int\prod_C\psi_C(x_C)\d x,$$
where $\mathcal C$ is the set of (maximal) cliques of $G$ and each <em>potential</em> $\psi_C\ge0$ depends only on the variables in $C$. Equivalently, for positive $p$, $p(x)\propto\exp\big(-\sum_C E_C(x_C)\big)$ with clique energies $E_C$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Factorisation implies the global Markov property">
<p>If $p$ factorises over $G$, then $p$ satisfies the global Markov property.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $S$ separate $A$ from $B$. Let $\tilde A$ be the union of $A$ and all vertices connected to $A$ in the graph with $S$ removed, and $\tilde B = V\setminus(\tilde A\cup S)\supseteq B$. A clique cannot contain both a vertex of $\tilde A$ and one of $\tilde B$ (they would be adjacent, so connected outside $S$). Hence every clique lies within $\tilde A\cup S$ or within $\tilde B\cup S$, and $p(x) = f(x_{\tilde A},x_S)\,g(x_{\tilde B},x_S)$. A density of this product form satisfies $X_{\tilde A}\indep X_{\tilde B}\mid X_S$, and independence of the sub-vectors $X_A$, $X_B$ follows by marginalisation.</p>
</details>
<div class="thm" data-kind="Theorem" data-name="Hammersley–Clifford (1971; Besag 1974)">
<p>If $p(x) \gt 0$ for all $x$ and $p$ satisfies the pairwise Markov property with respect to $G$, then $p$ factorises over the cliques of $G$. Hence, for positive distributions, pairwise, local and global Markov properties and clique factorisation are all equivalent.</p>
</div>
<details class="proof"><summary>Proof sketch (Möbius inversion)</summary>
<p>Fix a reference configuration $x^*$. For $A\subseteq V$ let $H_A(x) = \log p(x_A, x^*_{V\setminus A})$ (values in $A$ from $x$, the rest from $x^*$), and define
$$\phi_A(x) = \sum_{B\subseteq A}(-1)^{|A\setminus B|}H_B(x).$$
By Möbius inversion on the subset lattice, $H_A = \sum_{B\subseteq A}\phi_B$, so $\log p(x) = H_V(x) = \sum_{A\subseteq V}\phi_A(x)$, and each $\phi_A$ depends only on $x_A$. It remains to show $\phi_A\equiv0$ unless $A$ is a clique. Suppose $u,v\in A$ are not adjacent and let $C = A\setminus\{u,v\}$. Group the subsets $B\subseteq A$ by their intersection $W\subseteq C$ with $C$:
$$\phi_A = \sum_{W\subseteq C}(-1)^{|C\setminus W|}\big[H_{W\cup\{u,v\}} - H_{W\cup\{u\}} - H_{W\cup\{v\}} + H_W\big].$$
Each bracket is $\log\frac{p(x_u,x_v,x_W,x^*_\cdot)\,p(x_u^*,x_v^*,x_W,x^*_\cdot)}{p(x_u,x_v^*,x_W,x^*_\cdot)\,p(x_u^*,x_v,x_W,x^*_\cdot)}$, with all other coordinates fixed at $x^*$. By the pairwise property, $p(x_u,x_v\mid\text{rest}) = p(x_u\mid\text{rest})p(x_v\mid\text{rest})$, so the joint is a product of a function of $x_u$ and a function of $x_v$ (for fixed rest), and this cross-ratio equals $1$. So $\phi_A\equiv0$, and $p = \exp\sum_{A\text{ clique}}\phi_A$; group the $\phi_A$ into maximal cliques. Positivity is used to take logarithms.</p>
</details>
<p>Without positivity the theorem fails: Moussouris (1974) gave a distribution on a 4-cycle that is globally Markov but does not factorise. Note also that the factorisation over maximal cliques does not identify how a clique potential splits into pairwise and higher-order terms, which is why one often specifies a <em>pairwise</em> MRF explicitly or uses a factor graph.</p>
<div class="ex" data-name="The Ising model">
<p>For $x\in\{-1,+1\}^p$, $p(x) = \frac1{Z(\theta)}\exp\big(\sum_v\theta_vx_v + \sum_{(u,v)\in E}\theta_{uv}x_ux_v\big)$. It is a pairwise MRF and an <a href="#/t/exponential-families">exponential family</a>. The conditional of one site is logistic: $\P(x_v=1\mid x_{-v}) = \sigma\big(2\theta_v + 2\sum_{u\in\text{ne}(v)}\theta_{uv}x_u\big)$. $Z(\theta)$ is a sum over $2^p$ states and is #P-hard to compute in general, so MLE is replaced by pseudo-likelihood (Besag 1975), MCMC-based gradients, or neighbourhood selection by $\ell_1$-penalised logistic regression of each node on the others (Ravikumar, Wainwright &amp; Lafferty 2010).</p>
</div>`,
      },
      {
        title: 'Gaussian graphical models: zeros of the precision matrix',
        html: R`
<p>Let $X\sim\N_p(\mu,\Sigma)$ with $\Sigma \gt 0$, and let $\Omega = \Sigma^{-1}$ be the <strong>precision</strong> (concentration) matrix. The density is $\propto\exp\big(-\frac12(x-\mu)^\top\Omega(x-\mu)\big)$, a product of pairwise factors $\exp(-\Omega_{uv}(x_u-\mu_u)(x_v-\mu_v))$ — one for every nonzero off-diagonal $\Omega_{uv}$. The graph is therefore visible in $\Omega$.</p>
<div class="thm" data-kind="Theorem" data-name="Conditional independence ⇔ zero precision entry">
<p>For $u\neq v$, $\ X_u\indep X_v\mid X_{V\setminus\{u,v\}}\iff\Omega_{uv} = 0$. Moreover the partial correlation is
$$\rho_{uv\cdot\text{rest}} = \Corr(X_u,X_v\mid X_{\text{rest}}) = -\frac{\Omega_{uv}}{\sqrt{\Omega_{uu}\Omega_{vv}}}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $a = \{u,v\}$ and $b$ = the rest. For a Gaussian, $X_a\mid X_b$ is Gaussian with covariance given by the Schur complement $\Sigma_{a\mid b} = \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$, and the block-inverse formula says that this Schur complement is exactly $(\Omega_{aa})^{-1}$. For the $2\times2$ block,
$$(\Omega_{aa})^{-1} = \frac{1}{\Omega_{uu}\Omega_{vv}-\Omega_{uv}^2}\begin{pmatrix}\Omega_{vv} & -\Omega_{uv}\\ -\Omega_{uv} & \Omega_{uu}\end{pmatrix}.$$
The conditional covariance of $X_u$ and $X_v$ is therefore zero iff $\Omega_{uv}=0$, and for jointly Gaussian variables zero (conditional) covariance is equivalent to (conditional) independence. Dividing the off-diagonal entry by the square root of the product of the diagonal ones gives the partial correlation formula.</p>
</details>
<p>The same fact appears in regression form: regressing $X_u$ on all the others,
$$X_u = \mu_u + \sum_{v\neq u}\beta_{uv}(X_v-\mu_v) + \varepsilon_u,\qquad \beta_{uv} = -\frac{\Omega_{uv}}{\Omega_{uu}},\quad\Var(\varepsilon_u) = \frac1{\Omega_{uu}},$$
so $v$ has a nonzero coefficient in the regression for $u$ iff $\Omega_{uv}\neq0$. Meinshausen &amp; Bühlmann (2006) estimate the graph by running a <a href="#/t/lasso">lasso</a> regression of each variable on the rest (<em>neighbourhood selection</em>) and proved its consistency in high dimensions.</p>
<div class="warn" data-name="Correlation is not the graph">
<p>Zeros of $\Sigma$ encode <em>marginal</em> independence; zeros of $\Omega$ encode <em>conditional</em> independence. In a chain $X_1 - X_2 - X_3$ ($\Omega_{13}=0$), $X_1$ and $X_3$ are marginally correlated, often strongly. Thresholding a correlation matrix builds a "relevance network", which answers a different question and links everything to everything through common neighbours. The lab shows both matrices side by side.</p>
</div>`,
        lab: 'undirected-graphical-models-ggm',
      },
      {
        title: 'Estimation: covariance selection and the graphical lasso',
        html: R`
<p>The Gaussian log-likelihood in terms of $\Omega$, after profiling out $\mu$ ($\hat\mu = \bar x$), is
$$\ell(\Omega) = \frac n2\big[\log\det\Omega - \tr(S\Omega)\big] + \text{const},$$
concave in $\Omega$. Without constraints the MLE is $\hat\Omega = S^{-1}$, which exists only if $n \gt p$ and has no exact zeros. Dempster (1972) introduced <strong>covariance selection</strong>: maximise $\ell$ subject to $\Omega_{uv}=0$ for non-edges. The solution satisfies $\hat\Sigma_{uv} = S_{uv}$ on the diagonal and on edges, $\hat\Omega_{uv}=0$ off edges — a positive-definite completion of the observed entries — and is computed by iterative proportional scaling or by regression updates.</p>
<p>When the graph is unknown, penalise. The <strong>graphical lasso</strong> (Yuan &amp; Lin 2007; Banerjee, El Ghaoui &amp; d'Aspremont 2008; Friedman, Hastie &amp; Tibshirani 2008) solves
$$\hat\Omega_\lambda = \argmax_{\Omega \gt 0}\ \Big\{\log\det\Omega - \tr(S\Omega) - \lambda\sum_{u,v}|\Omega_{uv}|\Big\}$$
(some versions leave the diagonal unpenalised). The problem is convex, has a unique solution for every $\lambda \gt 0$ even when $p \gt n$, and the $\ell_1$ penalty sets entries exactly to zero.</p>
<div class="thm" data-kind="Proposition" data-name="Optimality conditions">
<p>With $W = \hat\Omega^{-1}$, the solution satisfies $W - S - \lambda\Gamma = 0$, where $\Gamma_{uv}\in\partial|\Omega_{uv}|$: $\Gamma_{uv}=\sgn(\Omega_{uv})$ if $\Omega_{uv}\neq0$, $\Gamma_{uv}\in[-1,1]$ otherwise. In particular $W_{uu} = S_{uu}+\lambda$, $|W_{uv}-S_{uv}|\le\lambda$ everywhere, with equality on the estimated edges.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The gradient of $\log\det\Omega$ is $\Omega^{-1}$, that of $\tr(S\Omega)$ is $S$, and the subdifferential of $\lambda\sum|\Omega_{uv}|$ is $\lambda\Gamma$. The objective is concave, so $0\in W - S - \lambda\Gamma$ characterises the maximiser. Diagonal entries of a positive-definite $\Omega$ are positive, so $\Gamma_{uu}=1$.</p>
</details>
<div class="algo" data-name="Graphical lasso by block coordinate descent (Friedman, Hastie & Tibshirani 2008)">
<ol>
<li>Initialise $W = S + \lambda I$.</li>
<li>Cycle over columns $j$. Partition $W$ and $S$ into the block $W_{11}$ (all rows and columns except $j$), the column $w_{12}$, and $w_{22}$. Solve the lasso problem
$$\hat\beta = \argmin_\beta\ \tfrac12\big\lVert W_{11}^{1/2}\beta - W_{11}^{-1/2}s_{12}\big\rVert^2 + \lambda\lVert\beta\rVert_1$$
by coordinate descent (it needs only $W_{11}$ and $s_{12}$), and set $w_{12} = W_{11}\hat\beta$.</li>
<li>Repeat until $W$ stops changing. Recover $\hat\Omega$ column by column: $\hat\omega_{22} = 1/(w_{22} - w_{12}^\top\hat\beta)$, $\hat\omega_{12} = -\hat\beta\,\hat\omega_{22}$.</li>
</ol>
</div>
<p>Each column update is the KKT condition for the $j$-th row of the optimality conditions, rewritten via the block-inverse formula, so the algorithm is block coordinate ascent on the dual. The sparsity pattern of $\hat\beta$ is the estimated neighbourhood of $j$.</p>
<p><strong>Theory.</strong> Under an irrepresentability (incoherence) condition on the Fisher information $\Sigma\otimes\Sigma$, with maximum degree $d$ and $n\gtrsim d^2\log p$, the graphical lasso with $\lambda\asymp\sqrt{\log p/n}$ recovers the exact edge set with high probability, and $\max_{uv}|\hat\Omega_{uv}-\Omega_{uv}| = O_P(\sqrt{\log p/n})$ (Ravikumar, Wainwright, Raskutti &amp; Yu 2011). In practice $\lambda$ is chosen by cross-validated likelihood (good for estimation, tends to over-select edges), by extended BIC (Foygel &amp; Drton 2010), or by stability selection (Meinshausen &amp; Bühlmann 2010; StARS, Liu, Roeder &amp; Wasserman 2010).</p>`,
      },
      {
        title: 'Directed versus undirected, and beyond',
        html: R`
<table>
<thead><tr><th></th><th>Directed (Bayesian network)</th><th>Undirected (MRF)</th></tr></thead>
<tbody>
<tr><td>Factors</td><td>Normalised CPDs $p(x_v\mid x_{\text{pa}(v)})$</td><td>Unnormalised clique potentials; global $Z$</td></tr>
<tr><td>Independence criterion</td><td>d-separation (colliders)</td><td>Plain graph separation</td></tr>
<tr><td>Can express</td><td>Explaining away ($A\indep C$, not given $B$)</td><td>Cyclic symmetric dependence (a 4-cycle: $A\indep C\mid B,D$ and $B\indep D\mid A,C$)</td></tr>
<tr><td>Parameter learning</td><td>Decomposes locally (complete data)</td><td>Coupled through $Z$ (except Gaussian, decomposable)</td></tr>
<tr><td>Natural for</td><td>Generative and causal stories</td><td>Spatial fields, images, networks of symmetric associations</td></tr>
</tbody>
</table>
<p>Neither family contains the other: the v-structure $A\to B\leftarrow C$ has no perfect undirected map, and the 4-cycle has no perfect DAG. Converting a DAG to an MRF requires <strong>moralisation</strong> (marrying co-parents), which loses the marginal independence of the parents. Graphs that are representable both ways are exactly the <em>chordal</em> (decomposable) graphs, for which MLEs have closed forms and the junction-tree algorithm is efficient.</p>
<div class="warn" data-name="Interpretation">
<p>An edge in an estimated GGM means non-zero partial correlation given <em>all other measured variables</em>. Adding or removing variables changes the graph; omitted common causes create edges; conditioning on a common effect (if it is among the variables) creates edges between its causes. A sparse precision matrix is a statistical summary, not a causal diagram.</p>
</div>
<p>Extensions include nonparanormal models (Gaussian copulas with unknown marginal transforms; Liu, Lafferty &amp; Wasserman 2009), mixed discrete–continuous graphical models, time-varying graphs, and joint estimation across groups (joint graphical lasso). Approximate inference in MRFs — loopy belief propagation, mean-field and other <a href="#/t/variational-inference">variational</a> methods (Wainwright &amp; Jordan 2008), <a href="#/t/gibbs-sampling">Gibbs sampling</a> — is a field of its own.</p>`,
      },
    ],
    quiz: [
      { q: R`Prove that for a Gaussian vector, $X_u\indep X_v\mid X_{\text{rest}}$ iff $\Omega_{uv}=0$.`,
        a: R`The conditional covariance of $(X_u,X_v)$ given the rest is the Schur complement $\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$, which equals $(\Omega_{aa})^{-1}$ by block inversion. For a $2\times2$ matrix, the inverse's off-diagonal is $-\Omega_{uv}/\det(\Omega_{aa})$, zero iff $\Omega_{uv}=0$. Conditionally on the rest, $(X_u,X_v)$ is bivariate normal, so zero covariance is independence.` },
      { q: R`In a Gaussian chain $X_1 - X_2 - X_3$, which entries of $\Sigma$ and of $\Omega$ are zero?`,
        a: R`$\Omega_{13}=0$ (conditional independence given $X_2$); all other $\Omega$ entries generically nonzero. $\Sigma$ generally has no zeros: $X_1$ and $X_3$ are marginally correlated through $X_2$ (e.g. $\Corr(X_1,X_3) = \Corr(X_1,X_2)\Corr(X_2,X_3)$ for the standardised chain).` },
      { q: R`Why does Hammersley–Clifford need positivity? Give the role it plays in the proof.`,
        a: R`The proof takes logarithms of $p$ (defining $H_A = \log p(x_A,x^*_{-A})$) and uses ratios of densities, which require $p \gt 0$ everywhere. More fundamentally, positivity gives the intersection property needed to pass from pairwise to global Markov; without it there are pairwise- or even globally-Markov distributions that do not factorise over the cliques (Moussouris 1974).` },
      { q: R`Why is the graphical lasso well defined when $p \gt n$ even though the Gaussian MLE is not?`,
        a: R`When $p \gt n$, $S$ is singular and $\log\det\Omega - \tr(S\Omega)$ is unbounded above (send $\Omega\to\infty$ along the null space of $S$). The $\ell_1$ penalty, which includes the diagonal, grows linearly in $\Omega$ while $\log\det$ grows only logarithmically, making the objective coercive; it is strictly concave, so there is a unique maximiser, and the KKT conditions give $W = \hat\Omega^{-1}$ with $W_{uu} = S_{uu}+\lambda \gt 0$.` },
    ],
  });
})();
