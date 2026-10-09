/* Supervised Learning II: trees, ensembles, boosting, SVMs, kernels and Gaussian processes. */
(function () {
  const R = String.raw;

  /* ===================================================================== */
  Atlas.topic({
    id: 'decision-trees',
    domain: 'supervised',
    order: 90,
    title: 'Classification & Regression Trees',
    subtitle: 'Greedy recursive partitioning, impurity, and cost-complexity pruning',
    level: 2,
    sources: { aos: '22.4', other: 'Breiman, Friedman, Olshen & Stone (1984); ESL §9.2' },
    prereqs: ['learning-framework', 'bias-variance'],
    related: ['bagging-random-forests', 'boosting', 'knn', 'cross-validation', 'histograms'],
    tags: ['CART', 'recursive partitioning', 'Gini', 'entropy', 'pruning', 'instability'],
    summary: R`A tree predicts with a piecewise-constant function on a partition of feature space into axis-aligned boxes, the boxes being chosen by the data through a sequence of yes/no questions $x_j \le s$. Finding the best tree is computationally hopeless, so CART grows one greedily, choosing at each node the split that most reduces an impurity (squared error, Gini, or entropy), then prunes the overgrown tree back by trading training error against the number of leaves. The result is interpretable, invariant to monotone transformations of the features, and handles interactions automatically — but it is also a high-variance, unstable estimator, which is exactly what makes it the ideal ingredient for bagging, random forests, and boosting.`,
    survey: {
      task: 'Regression and classification by recursive axis-aligned partitioning',
      assumptions: 'Essentially none: nonparametric; implicitly favours functions that are piecewise constant on boxes',
      output: R`A tree $\hat f(x)=\sum_m c_m \ind(x\in R_m)$; class proportions per leaf`,
      strengths: 'Interpretable, handles mixed types and missing values, invariant to monotone feature transforms, finds interactions',
      weaknesses: 'High variance and instability; greedy; staircase approximation of smooth or oblique structure; biased variable selection',
      complexity: R`Growing: $O(p\,n\log n)$ per level with presorting; pruning: $O(|T|^2)$ worst case`,
    },
    sections: [
      {
        title: 'Trees as adaptive partitions',
        html: R`
<p>Suppose we want to predict $Y$ from $X\in\R^p$. A histogram-style estimator chops feature space into fixed cells and averages $Y$ in each cell. The cells are chosen without looking at $Y$, and with $p$ features the number of cells needed explodes. A <strong>decision tree</strong> keeps the averaging but lets the response choose the cells. The fitted model is
$$\hat f(x) = \sum_{m=1}^{M} c_m\,\ind(x\in R_m),$$
where $R_1,\dots,R_M$ are disjoint axis-aligned boxes covering the feature space. The boxes are not arbitrary: each comes from a sequence of binary questions of the form "is $x_j \le s$?", so the partition can be drawn as a binary tree whose internal nodes are questions and whose leaves are the regions $R_m$.</p>
<p>Once the partition is fixed, the leaf values are trivial. Under squared error the best constant on $R_m$ is the leaf mean $c_m = \bar y_m$. For $K$-class classification the leaf stores the class proportions $\hat p_{mk}$ and predicts the majority class. All the difficulty is in the partition. Finding the partition with minimal training error among all trees with $M$ leaves is NP-complete (Hyafil &amp; Rivest, 1976), so CART (Breiman, Friedman, Olshen &amp; Stone, 1984) settles for a greedy, top-down construction followed by bottom-up pruning.</p>
<div class="note">
<p>Three properties follow straight from the form of the model. (1) <em>Invariance:</em> a split $x_j\le s$ depends only on the ordering of $x_j$, so any strictly monotone transformation of a feature gives the same tree. No standardisation is needed and outliers in $X$ are harmless. (2) <em>Interactions:</em> a split on $x_2$ inside the left branch of an $x_1$ split is an interaction, found without being specified. (3) <em>Locality:</em> the prediction at $x$ is an average of the $y_i$ in one cell. A tree is therefore a nearest-neighbour method whose neighbourhoods were chosen by the response.</p>
</div>`,
      },
      {
        title: 'Recursive binary splitting',
        html: R`
<p>Start with all data in one node. For a candidate variable $j$ and split point $s$ define the half-spaces $R_L(j,s) = \{x: x_j\le s\}$ and $R_R(j,s)=\{x: x_j \gt s\}$. In regression CART chooses
$$ (\hat j,\hat s) = \argmin_{j,s}\Big[\sum_{x_i\in R_L}(y_i-\bar y_L)^2 + \sum_{x_i\in R_R}(y_i-\bar y_R)^2\Big],$$
then applies the same rule recursively to each child. The search is cheap. Only the $n_t-1$ midpoints between consecutive distinct values of $x_j$ in the node need checking. After sorting, one sweep updates $\sum y$ and $\sum y^2$ on each side and evaluates $\mathrm{SSE} = \sum y^2 - (\sum y)^2/n$ in $O(1)$ per candidate. A node therefore costs $O(p\,n_t\log n_t)$, and a level of the tree costs $O(p\,n\log n)$.</p>
<div class="thm" data-kind="Proposition" data-name="What a regression split maximises">
<p>Splitting node $t$ (with $n_t$ points and mean $\bar y_t$) into children of sizes $n_L,n_R$ and means $\bar y_L,\bar y_R$ reduces the residual sum of squares by
$$\Delta = \mathrm{SSE}_t - \mathrm{SSE}_L - \mathrm{SSE}_R = \frac{n_L\,n_R}{n_t}\,(\bar y_L-\bar y_R)^2 .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>This is the one-way ANOVA identity: the total sum of squares about $\bar y_t$ splits into within-child sums plus the between-child term $n_L(\bar y_L-\bar y_t)^2 + n_R(\bar y_R - \bar y_t)^2$. Since $\bar y_t = (n_L\bar y_L + n_R\bar y_R)/n_t$, we have $\bar y_L - \bar y_t = \frac{n_R}{n_t}(\bar y_L-\bar y_R)$ and $\bar y_R-\bar y_t = -\frac{n_L}{n_t}(\bar y_L-\bar y_R)$. Substituting gives $\big(n_Ln_R^2 + n_Rn_L^2\big)(\bar y_L-\bar y_R)^2/n_t^2 = \frac{n_Ln_R}{n_t}(\bar y_L-\bar y_R)^2$.</p>
</details>
<p>So the best split maximises a two-sample statistic over all variables and cut points. That is a <em>maximally selected</em> statistic, and it has a consequence we return to below. Under the null of no signal, a variable with many distinct values offers many more chances to find a large $\Delta$ by luck than a binary variable does.</p>
<div class="algo" data-name="CART growing (regression or classification)">
<ol>
<li>Put all observations in the root node.</li>
<li>For the current node $t$, if a stopping rule holds (node pure, $n_t$ below a minimum, or maximum depth reached), make $t$ a leaf.</li>
<li>Otherwise, for each variable $j$ sort the node's values and evaluate every midpoint $s$. Pick $(j,s)$ maximising the impurity decrease $\Delta i(j,s,t)$.</li>
<li>Send observations with $x_j\le s$ to the left child and the rest to the right. Recurse on both children.</li>
<li>Grow deliberately large, then prune (next sections).</li>
</ol>
</div>
<p><strong>Categorical predictors.</strong> With $q$ unordered levels there are $2^{q-1}-1$ binary partitions. For squared-error regression, and for two-class classification with Gini or entropy, it suffices to order the levels by their mean response (or class-1 proportion) and treat the variable as ordered. The best of the $q-1$ resulting splits is optimal among all $2^{q-1}-1$ (Fisher, 1958; Breiman et al., 1984). No such shortcut exists for $K\ge3$ classes in general.</p>`,
        lab: 'decision-trees-cart',
      },
      {
        title: 'Impurity measures and why concavity matters',
        html: R`
<p>For classification, let $\hat p_k(t)$ be the proportion of class $k$ in node $t$. An impurity $i(t)=\phi(\hat p(t))$ is a function on the probability simplex that is zero at the vertices (pure nodes) and maximal at the uniform distribution. The three standard choices:</p>
<table>
<thead><tr><th>Name</th><th>$\phi(p)$</th><th>Two classes, $p = p_1$</th><th>Interpretation</th></tr></thead>
<tbody>
<tr><td>Gini index</td><td>$\sum_k p_k(1-p_k) = 1-\sum_k p_k^2$</td><td>$2p(1-p)$</td><td>Chance that two labels drawn independently from the node differ; total variance of one-hot labels</td></tr>
<tr><td>Entropy (deviance)</td><td>$-\sum_k p_k\log p_k$</td><td>$-p\log p-(1-p)\log(1-p)$</td><td>Node's maximised multinomial log-likelihood is $-n_t\,\phi$</td></tr>
<tr><td>Misclassification</td><td>$1-\max_k p_k$</td><td>$\min(p,1-p)$</td><td>Training error of the majority vote</td></tr>
</tbody>
</table>
<p>The split criterion is the weighted decrease $\Delta i = i(t) - \frac{n_L}{n_t}i(L)-\frac{n_R}{n_t}i(R)$. Two of the interpretations connect trees to the rest of statistics. With one-hot coding of $Y$, the Gini decrease is the regression SSE decrease of the previous proposition summed over the $K$ indicator columns (divided by $n_t$). The entropy decrease times $n_t$ equals $n_t$ times the empirical mutual information between the class and the side of the split, which is $G^2/2$, half the likelihood-ratio statistic for independence in the $2\times K$ table.</p>
<div class="thm" data-kind="Theorem" data-name="Concave impurities never increase under splitting">
<p>Let $\phi$ be concave on the simplex and let $w_L = n_L/n_t$, $w_R=n_R/n_t$. Then $\Delta i \ge 0$ for every split. If $\phi$ is strictly concave, $\Delta i = 0$ if and only if $\hat p(L) = \hat p(R)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Counting class-$k$ observations in the parent gives $n_t\hat p_k(t) = n_L\hat p_k(L) + n_R\hat p_k(R)$, i.e. $\hat p(t) = w_L\hat p(L)+w_R\hat p(R)$ is a convex combination of the children's proportion vectors. Concavity gives $\phi(\hat p(t)) \ge w_L\phi(\hat p(L)) + w_R\phi(\hat p(R))$. Strict concavity makes the inequality strict unless the two points coincide.</p>
</details>
<p>Gini and entropy are strictly concave, so every split that changes the class mix in the children registers as progress. Misclassification error is concave but only piecewise linear, and that is a real defect when growing trees:</p>
<div class="thm" data-kind="Lemma" data-name="Misclassification is blind to same-majority splits">
<p>If both children have the same majority class as the parent, the misclassification decrease is exactly zero.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $k$ be the common majority class. The children make $n_L(1-\hat p_k(L)) + n_R(1-\hat p_k(R)) = n_t - n_t\hat p_k(t)$ errors, which is the parent's error count.</p>
</details>
<div class="ex" data-name="Gini sees what misclassification cannot">
<p>A node holds 60 of each class. Split A gives children $(45,15)$ and $(15,45)$. Split B gives $(30,60)$ and $(30,0)$. Both leave 30 training errors, so misclassification rates them equal. Gini: split A leaves $\tfrac12\cdot 2(\tfrac14)(\tfrac34)+\tfrac12\cdot 2(\tfrac14)(\tfrac34) = 0.375$. Split B leaves $\tfrac34\cdot 2(\tfrac13)(\tfrac23) + \tfrac14\cdot 0 \approx 0.333$, so Gini prefers B, which produces a <em>pure</em> node that will never need further splitting. Entropy agrees with Gini. The usual practice is therefore to grow with Gini or entropy and to prune with misclassification error (or deviance), since that is the loss we will be judged on.</p>
</div>
<p>In the lab, choose XOR and the misclassification criterion. At the root no single axis-aligned cut changes the majority in either half, so the tree may refuse to split at all, even though two levels of splits solve the problem exactly.</p>`,
      },
      {
        title: 'Cost-complexity pruning',
        html: R`
<p>Stopping early is a bad way to control size. A split with little immediate gain can enable excellent splits below it (XOR again). CART instead grows a large tree $T_0$ and prunes. A <em>subtree</em> $T\preceq T_0$ is obtained by collapsing internal nodes into leaves. Write $|T|$ for its number of leaves and $R(T)=\sum_{\text{leaves } t}R(t)$ for its training risk (SSE for regression; misclassification count or deviance for classification). For a complexity parameter $\alpha \ge 0$ define
$$R_\alpha(T) = R(T) + \alpha\,|T| .$$</p>
<div class="thm" data-kind="Theorem" data-name="Nested optimal subtrees (Breiman et al., 1984)">
<p>(a) For every $\alpha\ge0$ there is a unique smallest subtree $T(\alpha)\preceq T_0$ minimising $R_\alpha$. (b) If $\alpha\le\beta$ then $T(\beta)\preceq T(\alpha)$. (c) Hence there are finitely many breakpoints $0=\alpha_0 \lt \alpha_1 \lt \cdots \lt \alpha_K$ and a nested sequence $T_0'\succ T_1\succ\cdots\succ T_K=\{\text{root}\}$ with $T(\alpha)=T_k$ for $\alpha\in[\alpha_k,\alpha_{k+1})$. (d) The sequence is produced by <em>weakest-link pruning</em>: for each internal node $t$ of $T_k$ with branch $T_{k,t}$ let
$$g_k(t) = \frac{R(t)-R(T_{k,t})}{|T_{k,t}|-1};$$
then $\alpha_{k+1}=\min_t g_k(t)$, and $T_{k+1}$ collapses every node attaining the minimum.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Dynamic programming.</em> $R_\alpha$ is additive over disjoint branches. Let $S_\alpha(t)$ be the least value of $R_\alpha$ over pruned versions of the branch at $t$. Then $S_\alpha(t)=R(t)+\alpha$ for a leaf of $T_0$, and otherwise
$$S_\alpha(t)=\min\{R(t)+\alpha,\; S_\alpha(t_L)+S_\alpha(t_R)\}.$$
Define $T(\alpha)$ top-down: collapse $t$ whenever $R(t)+\alpha\le S_\alpha(t_L)+S_\alpha(t_R)$, breaking ties toward collapsing. This attains the minimum, and every other minimiser keeps a superset of its nodes, giving (a).</p>
<p><em>Monotonicity.</em> $S_\alpha(t_L)+S_\alpha(t_R)$ is the minimum over pruned branches $T'$ that keep the split at $t$ of the affine functions $R(T')+\alpha|T'|$, each with slope $|T'|\ge2$. A minimum of affine functions is concave and piecewise linear, and here every slope is at least $2$. Subtracting $R(t)+\alpha$ (slope $1$) leaves $D_t(\alpha)=S_\alpha(t_L)+S_\alpha(t_R)-R(t)-\alpha$ with all slopes at least $1$, so $D_t$ is strictly increasing. Thus $\{\alpha: t \text{ is collapsed}\} = \{\alpha: D_t(\alpha)\ge0\}$ is an interval $[a_t,\infty)$. A node collapsed at $\alpha$ stays collapsed at every $\beta \gt \alpha$, which is (b). There are finitely many subtrees, giving (c).</p>
<p><em>Weakest link.</em> Collapsing $t$ in $T_k$ changes $R_\alpha$ by $R(t)-R(T_{k,t}) - \alpha(|T_{k,t}|-1)$, which is non-positive exactly when $\alpha\ge g_k(t)$. Starting from $T_k$, which is optimal on $[\alpha_k,\alpha_{k+1})$, the first value of $\alpha$ at which any collapse becomes worthwhile is $\min_t g_k(t)$.</p>
</details>
<p>(The full tree $T_0$ may contain splits with zero training gain. Collapsing those first gives the smallest minimiser at $\alpha=0$, written $T_0'$ above.) The pruning sequence is cheap, usually a few dozen trees. It converts the impossible search over all subtrees into a one-dimensional family indexed by $\alpha$.</p>
<div class="algo" data-name="Choosing α by K-fold cross-validation">
<ol>
<li>On the full data, grow $T_0$ and compute the breakpoints $\alpha_0,\dots,\alpha_K$ and subtrees $T_k$.</li>
<li>Represent the interval $[\alpha_k,\alpha_{k+1})$ by its geometric midpoint $\beta_k=\sqrt{\alpha_k\alpha_{k+1}}$.</li>
<li>For each fold $v$: grow and prune on the other folds, take the fold's optimal subtree at each $\beta_k$, and record its loss on fold $v$.</li>
<li>Average to get $\mathrm{CV}(\beta_k)$ with standard errors. Choose the minimiser, or (the <em>1-SE rule</em>) the smallest $T_k$ whose CV error is within one SE of the minimum.</li>
</ol>
</div>
<p>Cross-validation selects the <em>complexity parameter</em>, not a tree: the fold trees have different structures, and only $\alpha$ is transferred to the full-data sequence. See <a href="#/t/cross-validation">cross-validation</a> and <a href="#/t/model-selection">model selection</a>.</p>`,
        lab: 'decision-trees-pruning',
      },
      {
        title: 'Variance, instability and bias',
        html: R`
<p>A deep tree has low bias, because its cells are small, but each leaf averages only a handful of responses. For a leaf with $n_m$ points the conditional variance of the prediction is roughly $\sigma^2/n_m$. That number <em>understates</em> the real variability, because the cell boundaries are themselves random functions of the data. This second source of variance is what makes trees notoriously <strong>unstable</strong>. A small perturbation of the sample can change the root split, and since everything below is conditional on it, the whole tree changes. The prediction $\hat f(x)$ is a discontinuous function of the data: as an observation is moved, a split point can jump to a different variable. (In the bagging lab, press "New training sample" and watch the single tree.)</p>
<p>The bias has a characteristic geometry too. A piecewise-constant approximation on cells of diameter $h$ has error of order $h\,\|\nabla f\|$ for smooth $f$. An oblique decision boundary needs a staircase whose number of steps grows as the approximation improves. If the leaves hold about $k$ points each, cells in $d$ effective dimensions have diameter of order $(k/n)^{1/d}$. That is the same curse of dimensionality that afflicts <a href="#/t/histograms">histograms</a>, softened only to the extent that the tree concentrates its splits on the few directions that matter.</p>
<div class="note" data-name="Consistency, roughly">
<p>Partitioning estimators are consistent when, as $n\to\infty$, the cell containing $X$ shrinks to a point (in probability) while the number of observations in it grows without bound. For data-independent partitions this is classical. For data-dependent partitions like CART's, analogous results need extra conditions, such as a controlled number of cells relative to $n$ (Lugosi &amp; Nobel, 1996). Fully grown, unpruned trees violate "growing number of points per cell" and are typically <em>not</em> consistent on their own. Averaging many of them can restore consistency, which is part of why forests work.</p>
</div>
<p>Because the partition is chosen adaptively, naive inference on a fitted tree is invalid. Leaf means look significantly different because the splits were chosen to make them so. Honest estimation, which uses one subsample to choose splits and another to estimate leaf values, is the standard fix when trees are used for inference (Athey &amp; Imbens, 2016).</p>`,
      },
      {
        title: 'Pitfalls and practice',
        html: R`
<div class="warn" data-name="Greedy myopia">
<p>Greedy growth judges a split only by its immediate gain. In XOR-like structure every single split looks useless at the root. Growing large and pruning mitigates this. No greedy procedure eliminates it, because the global problem is NP-hard.</p>
</div>
<div class="warn" data-name="Variable-selection bias">
<p>Because the split is a maximum over candidate cut points, a pure-noise continuous variable or a categorical variable with many levels wins splits more often than a noise variable with few distinct values. Variable importance inherits this bias. Conditional inference trees (Hothorn, Hornik &amp; Zeileis, 2006) and GUIDE (Loh, 2002) separate the choice of variable, made by an unbiased test, from the choice of split point.</p>
</div>
<div class="warn" data-name="No extrapolation">
<p>Outside the range of the training data a tree predicts the constant of the boundary leaf, however strong the trend in the data. Regression trees are poor at smooth additive structure, such as $y=x_1+x_2+x_3$, which needs a staircase in every variable at once.</p>
</div>
<p><strong>Missing values</strong> are handled gracefully. CART keeps <em>surrogate splits</em> at each node: splits on other variables that best mimic the primary split. An observation missing the primary variable is routed by the best available surrogate. <strong>Interpretability</strong> is real only for small trees, and because of instability a different sample may tell a different story.</p>
<p><strong>Where trees go next.</strong> Their low bias and high variance make them ideal base learners for ensembles. <a href="#/t/bagging-random-forests">Bagging and random forests</a> average many deep trees to cut variance. <a href="#/t/boosting">Boosting</a> adds many shallow trees to cut bias. Smooth variants include MARS (Friedman, 1991), which replaces step functions by hinge functions, and oblique trees, which split on linear combinations.</p>`,
      },
    ],
    quiz: [
      { q: R`A regression node has 100 points. Split A sends 50 points each way with child means differing by 1. Split B sends 10 and 90 points with child means differing by 2. Which split does CART prefer?`,
        a: R`By $\Delta=\frac{n_Ln_R}{n_t}(\bar y_L-\bar y_R)^2$: split A gives $\frac{50\cdot50}{100}\cdot1 = 25$; split B gives $\frac{10\cdot90}{100}\cdot4=36$. CART prefers B. The criterion trades the size of the mean difference against the balance of the split: unbalanced splits pay a factor $n_Ln_R/n_t$ that is largest for equal sizes, but a large enough difference in means wins.` },
      { q: R`Why can a tree grown with the misclassification criterion stall on data that a tree grown with Gini fits easily?`,
        a: R`Misclassification is piecewise linear in the class proportions. Any split whose children keep the parent's majority class produces exactly zero decrease, however much purer the children are. Gini and entropy are strictly concave, so any change in the class mix counts as a strictly positive decrease. Gini therefore makes progress through intermediate splits that set up later pure nodes.` },
      { q: R`Show that if $\alpha$ increases, the number of leaves of the optimal pruned tree $T(\alpha)$ cannot increase.`,
        a: R`By the nesting theorem $T(\beta)\preceq T(\alpha)$ for $\beta\ge\alpha$, and a pruned subtree has no more leaves than the tree it was pruned from. Directly: for $T(\alpha)$ and $T(\beta)$ optimal, $R(T_\alpha)+\alpha|T_\alpha|\le R(T_\beta)+\alpha|T_\beta|$ and $R(T_\beta)+\beta|T_\beta|\le R(T_\alpha)+\beta|T_\alpha|$. Adding gives $(\beta-\alpha)(|T_\beta|-|T_\alpha|)\le0$.` },
      { q: R`You replace a feature $x_3$ by $\log x_3$ (all values positive). How does the fitted CART tree change? What if instead you replace $x_3$ by $x_3+x_4$?`,
        a: R`The log is strictly monotone, so the ordering of $x_3$ and hence every candidate partition is unchanged. The tree is identical except that split points on $x_3$ are reported on the log scale. Replacing $x_3$ by $x_3+x_4$ changes the set of available half-spaces: it adds oblique splits in the $(x_3,x_4)$ plane. The tree can change completely.` },
      { q: R`Why is a fully grown tree (one observation per leaf) a high-variance estimator even when $n$ is very large?`,
        a: R`Each prediction is a single noisy response $y_i$, so its variance is about $\sigma^2$ no matter how large $n$ is. On top of that, the cell boundaries are random and unstable. More data shrinks the cells (reducing bias) but does not average away the leaf noise. That noise only shrinks if leaves are forced to contain many points, or if many trees are averaged.` },
    ],
  });

  /* ===================================================================== */
  Atlas.topic({
    id: 'bagging-random-forests',
    domain: 'supervised',
    order: 100,
    title: 'Bagging & Random Forests',
    subtitle: 'Averaging unstable trees, and decorrelating them so the average helps more',
    level: 2,
    sources: { aos: '22.9', other: 'Breiman (1996, 2001); ESL §8.7 and Ch. 15' },
    prereqs: ['decision-trees', 'bootstrap'],
    related: ['boosting', 'bias-variance', 'cross-validation', 'knn'],
    tags: ['ensemble', 'variance reduction', 'out-of-bag', 'variable importance'],
    summary: R`Bagging fits the same unstable learner to many bootstrap resamples and averages the results. Averaging cannot change the expected prediction, but it removes the part of the variance that comes from the particular resample. What it cannot remove is the correlation between learners: an average of $B$ predictors with variance $\sigma^2$ and pairwise correlation $\rho$ has variance $\rho\sigma^2+(1-\rho)\sigma^2/B$. Random forests attack the floor $\rho\sigma^2$ by letting each split consider only a random subset of $m$ features, which decorrelates the trees at a small cost in individual accuracy. Out-of-bag observations give a free estimate of test error, and permutation and impurity importances (each with known biases) summarise which inputs the forest relies on.`,
    survey: {
      task: 'Regression and classification; variance reduction for unstable learners',
      assumptions: 'IID data; base learner low-bias and unstable (deep trees)',
      output: R`Averaged prediction / vote proportions; OOB error; variable importances`,
      strengths: 'Strong off-the-shelf accuracy, few tuning parameters, free OOB error, parallel, does not overfit as $B$ grows',
      weaknesses: 'Loses tree interpretability; biased importances; poor extrapolation; probabilities may be miscalibrated',
      complexity: R`$O(B\,m\,n\log^2 n)$ to grow $B$ trees (typical); prediction $O(B\cdot\text{depth})$`,
    },
    sections: [
      {
        title: 'Bootstrap aggregation',
        html: R`
<p>Let $\hat f(x;D)$ be a learner fitted to the training set $D$. <strong>Bagging</strong> (Breiman, 1996) draws $B$ bootstrap samples $D^*_1,\dots,D^*_B$ (each $n$ draws with replacement from $D$; see <a href="#/t/bootstrap">the bootstrap</a>) and averages:
$$\hat f_{\text{bag}}(x) = \frac1B\sum_{b=1}^B \hat f(x;D^*_b).$$
For classification one either takes a majority vote of the $B$ predicted labels or averages the trees' class-probability estimates. Averaging probabilities is usually preferred: it has lower variance and yields a probability.</p>
<p>The rationale is an idealisation. Define the <em>aggregated predictor</em> $f_A(x) = \E_D\,\hat f(x;D)$, the average over training sets drawn from the true distribution $P$.</p>
<div class="thm" data-kind="Proposition" data-name="Aggregation never hurts under squared error (Breiman, 1996)">
<p>For any fixed $(x,y)$, $\E_D\big(y-\hat f(x;D)\big)^2 = \big(y-f_A(x)\big)^2 + \Var_D\,\hat f(x;D) \;\ge\; \big(y-f_A(x)\big)^2$. Integrating over $(X,Y)$, the aggregated predictor's expected squared error is no larger than the average error of a single fit, and the gain is exactly the variance $\E_X \Var_D \hat f(X;D)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Add and subtract $f_A(x)$. The cross term $2(y-f_A(x))\,\E_D[f_A(x)-\hat f(x;D)]$ vanishes by the definition of $f_A$.</p>
</details>
<p>We cannot draw fresh training sets from $P$, so bagging draws them from $\hat P_n$ and in effect computes the "ideal bagged" predictor $\E_{D^*\sim\hat P_n}\hat f(x;D^*)$. This is not $f_A$. It is a bootstrap estimate of it, and it can differ from $\hat f(x;D)$ in its centre as well as in its variability. Three consequences follow.</p>
<ul>
<li><strong>Smooth learners gain nothing.</strong> For the sample mean, $\E^*\bar y^* = \bar y$ exactly. For least squares and other approximately linear statistics the bagged and unbagged fits nearly coincide. Bagging helps <em>unstable</em> procedures: trees, subset selection, hard thresholds. Bühlmann &amp; Yu (2002) make this precise: bagging replaces a hard indicator decision by a smoothed one, sharply reducing variance near the threshold.</li>
<li><strong>Bias is (essentially) unchanged.</strong> Averaging does not move the mean of the predictor, so the base learner should be low-bias: deep, unpruned trees.</li>
<li><strong>Classification can get worse.</strong> Under 0–1 loss, aggregation is a majority vote, and the inequality above has no analogue. Suppose that at some $x$, $P(Y=1\mid x) = 0.9$, but a classifier trained on a random $D$ predicts $1$ only 40% of the time. Its expected error at $x$ is $0.4(0.1)+0.6(0.9)=0.58$. The aggregated vote always predicts $0$, with error $0.9$. Voting helps only where the base classifier is right more often than not ("order-correct", in Breiman's term).</li>
</ul>`,
      },
      {
        title: 'The variance of an average of correlated trees',
        html: R`
<p>Bagged trees are not independent: they are grown from overlapping resamples of the same data. That correlation sets a floor on what averaging can achieve.</p>
<div class="thm" data-kind="Proposition" data-name="Variance of a correlated average">
<p>Let $T_1,\dots,T_B$ be identically distributed with variance $\sigma^2$ and pairwise correlation $\rho$. Then
$$\Var\Big(\frac1B\sum_{b=1}^B T_b\Big) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2 .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\Var(\sum_b T_b) = \sum_b\Var T_b + \sum_{b\ne b'}\Cov(T_b,T_{b'}) = B\sigma^2 + B(B-1)\rho\sigma^2$. Dividing by $B^2$ gives $\sigma^2/B + (1-1/B)\rho\sigma^2 = \rho\sigma^2+(1-\rho)\sigma^2/B$.</p>
</details>
<p>For an ensemble, the randomness in tree $b$ has two sources: the training set $D$, shared by all trees, and the tree's own randomisation $\Theta_b$ (its bootstrap sample and, for forests, its feature subsets), which is IID across trees given $D$. Write $T(x;\Theta,D)$ for a single tree's prediction at a fixed $x$. The correlation then has a clean meaning.</p>
<div class="thm" data-kind="Proposition" data-name="What ρ measures">
<p>If $\Theta_1,\Theta_2$ are IID and independent of $D$, then
$$\rho(x) = \Corr\big(T(x;\Theta_1,D),T(x;\Theta_2,D)\big) = \frac{\Var_D\big(\E[T\mid D]\big)}{\Var_D\big(\E[T\mid D]\big) + \E_D\big(\Var[T\mid D]\big)}\in[0,1].$$
It is the fraction of a single tree's variance that is due to the training sample rather than to the tree's own randomisation.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the law of total covariance, $\Cov(T_1,T_2) = \E_D\Cov(T_1,T_2\mid D) + \Cov_D(\E[T_1\mid D],\E[T_2\mid D])$. Given $D$, $T_1$ and $T_2$ are independent, so the first term is $0$. The two conditional means are the same function of $D$, so the second term is $\Var_D(\E[T\mid D])$. The denominator $\sigma^2=\Var T$ follows from the law of total variance.</p>
</details>
<div class="note">
<p>Read the formula as $\Var(\hat f_{\text{bag}}) = \Var_D(\E[T\mid D]) + \E_D\Var[T\mid D]/B$. Increasing $B$ kills only the within-$D$ randomisation variance. That is just Monte Carlo error, so adding trees cannot cause overfitting (Breiman, 2001, proves the generalisation error converges almost surely as $B\to\infty$). The floor $\rho\sigma^2$ is the variance of the infinite ensemble. The only way to lower it is to make $\E[T\mid D]$ less sensitive to $D$, which in practice means making the trees less alike.</p>
</div>`,
      },
      {
        title: 'Random forests: decorrelating the trees',
        html: R`
<p>Suppose one predictor is much stronger than the rest. Every bagged tree will split on it at the root, and usually on the same handful of variables just below. The trees are near-copies, $\rho$ is large, and bagging stalls. Random forests (Breiman, 2001; building on Amit &amp; Geman, 1997, and Ho, 1998) inject randomness into the growing itself.</p>
<div class="algo" data-name="Random forest">
<ol>
<li>For $b=1,\dots,B$: draw a bootstrap sample $D^*_b$.</li>
<li>Grow a tree on $D^*_b$. At <em>every</em> node, select $m$ of the $p$ variables at random and choose the best split among those $m$ only. Grow deep: stop at a minimum node size (commonly 1 for classification, 5 for regression), and do not prune.</li>
<li>Predict by averaging (regression) or by voting or averaging probabilities (classification).</li>
</ol>
</div>
<p>With $m=p$ this is bagging. Smaller $m$ lowers $\rho$, because different trees are forced to build on different variables. It raises the single-tree variance $\sigma^2$ and usually the bias a little, because each tree is weaker. The net effect on $\rho\sigma^2+(1-\rho)\sigma^2/B$ is typically favourable. Default choices are $m=\lfloor\sqrt p\rfloor$ for classification and $m=\lfloor p/3\rfloor$ for regression, and $m$ can be tuned by out-of-bag error. Small $m$ fails when relevant variables are rare. If $r$ of $p$ variables matter, the chance that a node sees none of them is $\binom{p-r}{m}\big/\binom{p}{m}$, which is close to $1$ when $r\ll p$ and $m$ is small.</p>
<div class="thm" data-kind="Theorem" data-name="Strength and correlation (Breiman, 2001)">
<p>For classification, let the margin of the forest be $\mathrm{mg}(X,Y) = P_\Theta(h(X,\Theta)=Y)-\max_{j\ne Y}P_\Theta(h(X,\Theta)=j)$, its strength $s=\E\,\mathrm{mg}(X,Y) \gt 0$, and $\bar\rho$ the mean correlation between the raw margin functions of two independent trees. Then the generalisation error satisfies
$$PE^* \le \frac{\bar\rho\,(1-s^2)}{s^2}.$$</p>
</div>
<p>The bound is loose, but it names the two levers: make individual trees accurate (high strength) and make them disagree (low correlation). The proof is Chebyshev's inequality applied to the margin, with the variance of the margin bounded via $\bar\rho$. Pushing the randomisation further gives <em>extremely randomised trees</em> (Geurts, Ernst &amp; Wehenkel, 2006), which also draw the cut point at random.</p>`,
        lab: 'bagging-random-forests-surfaces',
      },
      {
        title: 'Out-of-bag error',
        html: R`
<p>Each bootstrap sample omits a given observation with probability $(1-1/n)^n\to e^{-1}\approx0.368$. So for each $i$ about $0.368B$ trees never saw $(x_i,y_i)$. The <strong>out-of-bag</strong> (OOB) prediction for $i$ aggregates only those trees,
$$\hat f_{\text{oob}}(x_i) = \frac{1}{|\{b: i\notin D^*_b\}|}\sum_{b:\,i\notin D^*_b}\hat f(x_i;D^*_b),\qquad \widehat{\mathrm{Err}}_{\text{oob}}=\frac1n\sum_i L\big(y_i,\hat f_{\text{oob}}(x_i)\big).$$
Every OOB prediction is a genuine out-of-sample prediction, so $\widehat{\mathrm{Err}}_{\text{oob}}$ behaves like a <a href="#/t/cross-validation">cross-validation</a> estimate at no extra fitting cost. Two caveats. It describes an ensemble of about $0.37B$ trees, each fitted to about $63\%$ of the distinct points, so it is slightly pessimistic for the full forest, noticeably so when $B$ is small. And like CV it assumes exchangeable observations: with grouped or serially dependent data the OOB points are not truly "new", and the estimate is optimistic.</p>`,
        lab: 'bagging-random-forests-ntrees',
      },
      {
        title: 'Variable importance and its biases',
        html: R`
<p>Two importance measures are standard.</p>
<div class="cols">
<div><h4>Mean decrease in impurity (MDI)</h4><p>Sum the weighted impurity decreases $p(t)\,\Delta i(t)$ over all nodes $t$ that split on $X_j$, where $p(t)$ is the fraction of samples reaching $t$, and average over trees. Free to compute, but evaluated on the <em>training</em> (in-bag) data.</p></div>
<div><h4>Permutation importance (MDA)</h4><p>For tree $b$ compute the OOB error $e_b$. Randomly permute $X_j$ among the OOB observations and recompute it as $e_b^{(j)}$. Report $\frac1B\sum_b(e^{(j)}_b-e_b)$: how much worse predictions get when $X_j$'s link to $Y$ (and to the other $X$'s) is broken.</p></div>
</div>
<div class="warn" data-name="MDI favours many split points">
<p>Deep trees split noise variables too, and every split registers a positive training-impurity decrease. A noise variable with many distinct values offers more candidate cuts, so it collects more spurious decrease than a binary one (Strobl, Boulesteix, Zeileis &amp; Hothorn, 2007). MDI can rank an irrelevant continuous ID-like variable above a relevant binary one.</p>
</div>
<div class="warn" data-name="Correlated predictors confuse both measures">
<p>If $X_4$ and $X_6$ are near-copies, each split picks one, so their MDI is shared between them and each looks half as important. Permuting $X_6$ alone hurts little, because the forest can fall back on $X_4$. It also creates impossible points (an $X_6$ far from its twin) where the forest's behaviour is arbitrary. Conditional permutation schemes (Strobl et al., 2008) permute within strata of the correlated variables. Toggle the twin in the second lab to see the split.</p>
</div>
<p>For a cleaner theory: in an idealised setting (totally randomised trees with $m=1$, categorical inputs, infinite sample, entropy impurity), Louppe et al. (2013) show that the MDI importances sum to the mutual information $I(X_1,\dots,X_p;Y)$, and that an importance is zero exactly for variables irrelevant in their sense. With finite samples and greedy splits, the biases above dominate. Either way, an importance describes how much <em>this fitted model</em> relies on a variable. It is not a causal effect, and not a property of the variable alone.</p>`,
      },
      {
        title: 'Forests as adaptive nearest neighbours',
        html: R`
<p>A regression forest's prediction is a weighted average of training responses,
$$\hat f(x) = \sum_{i=1}^n w_i(x)\,y_i,\qquad w_i(x) = \frac1B\sum_{b=1}^B \frac{N_{bi}\,\ind\big(x_i\in L_b(x)\big)}{\sum_{k} N_{bk}\,\ind\big(x_k\in L_b(x)\big)},$$
where $L_b(x)$ is the leaf of tree $b$ containing $x$ and $N_{bi}$ counts how often $i$ appears in $D^*_b$. The weights are non-negative and sum to one, so a forest is a <a href="#/t/knn">nearest-neighbour</a> smoother whose metric is learned from $Y$ (Lin &amp; Jeon, 2006). This view pays off. Quantile regression forests (Meinshausen, 2006) use the same weights to estimate the whole conditional distribution $\hat F(y\mid x)=\sum_i w_i(x)\ind(y_i\le y)$. "Honest" forests built on subsamples are asymptotically normal, enabling confidence intervals (Wager &amp; Athey, 2018).</p>
<div class="warn" data-name="Practical pitfalls">
<p>(1) A forest cannot extrapolate: the predictions lie in the convex hull of the training $y_i$. (2) Training error of deep forests is near zero and meaningless, so use OOB or held-out data. (3) Vote proportions are often poorly calibrated, typically pulled toward $\tfrac12$; check a reliability curve (<a href="#/t/classifier-evaluation">classifier evaluation</a>). (4) Strong class imbalance needs stratified or balanced bootstrap samples. (5) The OOB estimate inherits CV's failure under dependence.</p>
</div>
<p>Bagging and forests reduce <em>variance</em> by averaging low-bias learners in parallel. <a href="#/t/boosting">Boosting</a> takes the complementary route: it reduces <em>bias</em> by adding high-bias learners in sequence.</p>`,
      },
    ],
    quiz: [
      { q: R`Single trees have variance $\sigma^2=1$ at some $x$ and pairwise correlation $\rho=0.3$. What is the variance of the average of 10 trees, and of infinitely many?`,
        a: R`$\rho\sigma^2+(1-\rho)\sigma^2/B = 0.3+0.7/10=0.37$ for $B=10$, and $0.3$ as $B\to\infty$. Ten trees already capture most of the achievable reduction. Getting below $0.3$ requires lowering $\rho$ (e.g. smaller $m$), not adding trees.` },
      { q: R`Why does bagging the sample mean change nothing, while bagging a regression tree helps a lot?`,
        a: R`The bootstrap expectation of the sample mean is exactly $\bar y$, so the bagged estimator equals the original, and averaging removes no instability because the mean is a smooth (linear) statistic. A tree is a highly nonlinear, discontinuous function of the data. Its resample-to-resample variation is large, and averaging over resamples smooths the hard split decisions and removes much of that variance.` },
      { q: R`With $B=500$ trees, roughly how many trees contribute to a given observation's OOB prediction? Why is the OOB error slightly pessimistic?`,
        a: R`About $500\,e^{-1}\approx184$. It is pessimistic because it evaluates a smaller ensemble (about 184 rather than 500 trees), each grown on only about 63% of the distinct points, so each OOB prediction is a bit noisier and more biased than the full forest's prediction.` },
      { q: R`A random noise variable with 1000 distinct values receives higher impurity importance than a relevant binary variable. Explain.`,
        a: R`MDI accumulates training-set impurity decreases. Deep trees keep splitting, and among candidate splits, a variable with 1000 cut points can almost always find one with a sizeable spurious decrease in a small node, while a binary variable has a single cut. The decreases are measured on in-bag data, so overfitting registers as importance. OOB permutation importance or conditional-inference trees reduce this bias.` },
      { q: R`If adding trees never causes overfitting, can a random forest overfit at all?`,
        a: R`Yes, through the trees themselves, not through $B$. Deep trees with tiny leaves, a poorly chosen $m$, or leakage between training and test data can give a forest whose limiting ($B\to\infty$) prediction overfits. The almost-sure convergence as $B\to\infty$ concerns only the Monte Carlo randomness. It says nothing about the variance $\rho\sigma^2$ that comes from the training sample, nor about bias.` },
    ],
  });

  /* ===================================================================== */
  Atlas.topic({
    id: 'boosting',
    domain: 'supervised',
    order: 110,
    title: 'Boosting: AdaBoost & Gradient Boosting',
    subtitle: 'Stagewise additive models: reweighting, residual fitting, and functional gradient descent',
    level: 3,
    sources: { aos: '22.9', other: 'Freund & Schapire (1997); Friedman, Hastie & Tibshirani (2000); Friedman (2001); Chen & Guestrin (2016); ESL Ch. 10' },
    prereqs: ['decision-trees', 'learning-framework'],
    related: ['bagging-random-forests', 'logistic-regression', 'lasso', 'svm', 'classifier-evaluation'],
    tags: ['AdaBoost', 'exponential loss', 'margins', 'gradient boosting', 'shrinkage', 'XGBoost'],
    summary: R`Boosting builds a strong predictor as a sum of many weak ones, $F_M(x)=\sum_m \beta_m h_m(x)$, adding one term at a time, with each new term chosen to fix what the current sum gets wrong. AdaBoost does this by reweighting the training points; its training error falls geometrically, bounded by $\prod_m 2\sqrt{\varepsilon_m(1-\varepsilon_m)}$. It turns out to be forward stagewise minimisation of the exponential loss, whose population minimiser is half the log-odds. Gradient boosting generalises the idea to any differentiable loss by fitting each new tree to the negative gradient of the loss, which is gradient descent in function space. Shrinkage (a small learning rate), subsampling and early stopping are the regularisers. Second-order (Newton) steps with penalised leaf weights give XGBoost.`,
    survey: {
      task: 'Classification and regression with any differentiable loss',
      assumptions: R`IID data; a weak learner slightly better than chance (AdaBoost); differentiable loss (gradient boosting)`,
      output: R`Additive model $F_M(x)=\sum_m \nu\,h_m(x)$; for classification, a score estimating (half) the log-odds`,
      strengths: 'State-of-the-art accuracy on tabular data; flexible losses; captures interactions up to tree depth',
      weaknesses: R`Sequential (slower to train); overfits if $M$ is too large; exponential loss is fragile under label noise; several tuning parameters`,
      complexity: R`$O(M \cdot \text{cost of one tree})$; prediction $O(M\cdot\text{depth})$`,
    },
    sections: [
      {
        title: 'From weak learners to a strong one',
        html: R`
<p>Kearns and Valiant asked in 1988 whether a "weak" learner, one guaranteed only to do slightly better than random guessing on any distribution, could be turned into a "strong" learner with arbitrarily small error. Schapire (1990) showed it could. Freund and Schapire's <strong>AdaBoost</strong> (1997) made the construction practical: fit the weak learner, increase the weight of the points it got wrong, fit again, and take a weighted vote.</p>
<div class="algo" data-name="AdaBoost (labels y ∈ {−1, +1})">
<ol>
<li>Initialise weights $w^{(1)}_i = 1/n$.</li>
<li>For $m=1,\dots,M$:
<ol>
<li>Fit a classifier $G_m(x)\in\{-1,+1\}$ to the training data with weights $w^{(m)}$.</li>
<li>Compute its weighted error $\varepsilon_m=\sum_i w^{(m)}_i\ind\big(y_i\ne G_m(x_i)\big)$. If $\varepsilon_m\ge\tfrac12$, stop.</li>
<li>Set $\alpha_m = \tfrac12\log\frac{1-\varepsilon_m}{\varepsilon_m} \gt 0$.</li>
<li>Update $w^{(m+1)}_i = w^{(m)}_i\exp\big(-\alpha_m y_iG_m(x_i)\big)/Z_m$, where $Z_m$ normalises the weights to sum to one.</li>
</ol></li>
<li>Output $\hat G(x)=\sgn F_M(x)$ with $F_M(x)=\sum_{m=1}^M\alpha_mG_m(x)$.</li>
</ol>
</div>
<p>Correctly classified points are multiplied by $e^{-\alpha_m}$ and misclassified ones by $e^{\alpha_m}$. (ESL writes $\alpha_m=\log\frac{1-\varepsilon_m}{\varepsilon_m}$, twice ours, and multiplies only the misclassified weights by $e^{\alpha_m}$. After normalisation the weights are identical and the final classifier has the same sign.) The reweighting is calibrated in a striking way:</p>
<div class="thm" data-kind="Proposition" data-name="The last learner is made useless">
<p>Under the updated weights $w^{(m+1)}$, the classifier $G_m$ has weighted error exactly $\tfrac12$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$Z_m=(1-\varepsilon_m)e^{-\alpha_m}+\varepsilon_me^{\alpha_m}$. With $e^{\alpha_m}=\sqrt{(1-\varepsilon_m)/\varepsilon_m}$ both terms equal $\sqrt{\varepsilon_m(1-\varepsilon_m)}$, so $Z_m = 2\sqrt{\varepsilon_m(1-\varepsilon_m)}$. The new weight on the misclassified points is $\varepsilon_me^{\alpha_m}/Z_m=\sqrt{\varepsilon_m(1-\varepsilon_m)}/Z_m=\tfrac12$.</p>
</details>
<p>So the next round cannot simply reuse $G_m$: the weights have been set so that $G_m$ is a coin flip under them, and $G_{m+1}$ must find structure that $G_m$ missed.</p>`,
      },
      {
        title: 'The training-error bound',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Training error of AdaBoost (Freund & Schapire, 1997)">
<p>Write $\gamma_m=\tfrac12-\varepsilon_m$ for the edge of the $m$-th learner over random guessing. Then
$$\frac1n\sum_{i=1}^n\ind\big(y_i\ne\hat G(x_i)\big)\;\le\;\prod_{m=1}^M 2\sqrt{\varepsilon_m(1-\varepsilon_m)} \;=\; \prod_{m=1}^M\sqrt{1-4\gamma_m^2}\;\le\;\exp\Big(-2\sum_{m=1}^M\gamma_m^2\Big).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Step 1 (0–1 loss is below exponential loss).</em> If $\hat G(x_i)\ne y_i$ then $y_iF_M(x_i)\le0$, so $\ind(y_i\ne\hat G(x_i))\le e^{-y_iF_M(x_i)}$.</p>
<p><em>Step 2 (unroll the weights).</em> From the update rule and $w^{(1)}_i=1/n$,
$$w^{(M+1)}_i=\frac{1}{n}\,\frac{\exp\big(-y_i\sum_m\alpha_mG_m(x_i)\big)}{\prod_m Z_m}=\frac{e^{-y_iF_M(x_i)}}{n\prod_mZ_m}.$$
The weights sum to one, so $\frac1n\sum_ie^{-y_iF_M(x_i)}=\prod_mZ_m$.</p>
<p><em>Step 3 (each factor).</em> $Z_m=(1-\varepsilon_m)e^{-\alpha_m}+\varepsilon_me^{\alpha_m}$, which equals $2\sqrt{\varepsilon_m(1-\varepsilon_m)}$ at the chosen $\alpha_m$. Indeed, that $\alpha_m$ is precisely the minimiser of $Z_m$ over $\alpha$, so AdaBoost greedily minimises the bound. Finally $2\sqrt{\varepsilon(1-\varepsilon)}=\sqrt{1-4\gamma^2}\le e^{-2\gamma^2}$ by $1-u\le e^{-u}$.</p>
</details>
<p>If every weak learner has edge at least $\gamma \gt 0$, the training error is at most $e^{-2M\gamma^2}$. Since training error is a multiple of $1/n$, it is exactly zero once $M \gt \log n/(2\gamma^2)$. The weak-learning assumption alone forces the ensemble to fit the training data perfectly. Whether that is good is a generalisation question, taken up below.</p>`,
        lab: 'boosting-adaboost',
      },
      {
        title: 'AdaBoost is stagewise exponential-loss minimisation',
        html: R`
<p>Friedman, Hastie &amp; Tibshirani (2000) explained <em>what</em> AdaBoost estimates. Consider an additive model $F(x)=\sum_m\beta_mG_m(x)$ with base classifiers $G\in\{-1,+1\}$, and the <strong>exponential loss</strong> $L(y,F)=e^{-yF}$. <em>Forward stagewise</em> fitting adds one term at a time without revisiting earlier ones:
$$(\beta_m,G_m)=\argmin_{\beta,G}\sum_{i=1}^n\exp\big(-y_i[F_{m-1}(x_i)+\beta G(x_i)]\big).$$</p>
<div class="thm" data-kind="Proposition" data-name="The stagewise solution is AdaBoost">
<p>With $w_i^{(m)}\propto e^{-y_iF_{m-1}(x_i)}$, the minimiser is: $G_m$ = the classifier minimising the weighted error $\varepsilon_m$; $\beta_m=\tfrac12\log\frac{1-\varepsilon_m}{\varepsilon_m}$; and the next weights are $w_i^{(m+1)}\propto w_i^{(m)}e^{-\beta_my_iG_m(x_i)}$. These are exactly AdaBoost's steps.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The criterion is $\sum_iw_i^{(m)}e^{-\beta y_iG(x_i)}$, since the factor $e^{-y_iF_{m-1}(x_i)}$ does not involve $(\beta,G)$. Because $y_iG(x_i)=\pm1$, it equals
$$e^{-\beta}\!\!\sum_{y_i=G(x_i)}\!\!w_i+e^{\beta}\!\!\sum_{y_i\ne G(x_i)}\!\!w_i=(e^{\beta}-e^{-\beta})\sum_iw_i\ind(y_i\ne G(x_i))+e^{-\beta}\sum_iw_i.$$
For any $\beta \gt 0$ this is increasing in the weighted error, so $G_m$ minimises the weighted error regardless of $\beta$. With normalised weights and error $\varepsilon$, minimise $(1-\varepsilon)e^{-\beta}+\varepsilon e^{\beta}$: the derivative vanishes when $e^{2\beta}=(1-\varepsilon)/\varepsilon$. The weight update is the definition $w^{(m+1)}_i\propto e^{-y_iF_m(x_i)}=w^{(m)}_ie^{-\beta_my_iG_m(x_i)}$.</p>
</details>
<div class="thm" data-kind="Proposition" data-name="Population minimiser of exponential loss">
<p>$f^*(x)=\argmin_f\E\big[e^{-Yf(x)}\mid X=x\big]=\tfrac12\log\frac{P(Y=1\mid x)}{P(Y=-1\mid x)}$, so $P(Y=1\mid x)=1/(1+e^{-2f^*(x)})$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>With $\eta=P(Y=1\mid x)$ the conditional risk is $\eta e^{-f}+(1-\eta)e^{f}$, which is strictly convex in $f$. Setting the derivative $-\eta e^{-f}+(1-\eta)e^{f}$ to zero gives $e^{2f}=\eta/(1-\eta)$.</p>
</details>
<p>So AdaBoost is an estimator of half the <a href="#/t/logistic-regression">log-odds</a>, the same target as logistic regression's binomial deviance $\log(1+e^{-2yF})$. (The AdaBoost lab shades the plane with $1/(1+e^{-2F})$.) As functions of the margin $z=yF$, the two losses agree near $z=0$ but differ sharply for badly misclassified points ($z\to-\infty$). The exponential loss grows like $e^{-z}$, while the deviance grows only linearly. A few mislabelled points therefore acquire exponentially large weight and dominate later rounds. That is the main practical weakness of AdaBoost under label noise, and the motivation for LogitBoost and for gradient boosting with the deviance.</p>`,
      },
      {
        title: 'Margins: why test error keeps falling',
        html: R`
<p>A puzzle from early experiments: AdaBoost's test error often keeps <em>decreasing</em> for hundreds of rounds after the training error reaches zero, although the model keeps growing more complex. The explanation of Schapire, Freund, Bartlett &amp; Lee (1998) is the <strong>normalised margin</strong>
$$\mathrm{mg}(x_i,y_i)=\frac{y_iF_M(x_i)}{\sum_m\alpha_m}\in[-1,1],$$
the weighted vote's net confidence in the correct label. Once every point is classified correctly, further rounds still increase the small margins.</p>
<div class="thm" data-kind="Theorem" data-name="Margin bound (Schapire et al., 1998)">
<p>Let the base classifiers come from a class of VC dimension $d$. With probability at least $1-\delta$ over a sample of size $n$, every convex combination $f=\sum_m\alpha_mG_m/\sum_m\alpha_m$ and every $\theta \gt 0$ satisfy
$$P\big(Yf(X)\le0\big)\;\le\;\hat P_n\big(Yf(X)\le\theta\big)+O\!\left(\sqrt{\frac{d\log^2(n/d)}{n\,\theta^2}+\frac{\log(1/\delta)}{n}}\right).$$
Moreover, AdaBoost drives the empirical margin distribution up: $\hat P_n(Yf(X)\le\theta)\le\prod_{m=1}^M2\sqrt{\varepsilon_m^{1-\theta}(1-\varepsilon_m)^{1+\theta}}$.</p>
</div>
<p>The bound does not depend on $M$, the number of terms. Complexity is measured by the base class and the margin, not by how many classifiers are combined. If the edges $\gamma_m$ stay bounded away from zero, the second inequality shows the fraction of points with margin below some $\theta \gt 0$ goes to zero, and the bound improves even after the training error is zero. The <em>Margins</em> view in the lab shows the empirical CDF of margins moving right round after round.</p>
<div class="warn" data-name="The margin story is incomplete">
<p>Breiman's arc-gv (1999) maximised the <em>minimum</em> margin more aggressively than AdaBoost and yet often generalised worse. That suggests the whole margin distribution matters, not only its minimum (later refinements: Reyzin &amp; Schapire, 2006; Gao &amp; Zhou, 2013). And with noisy labels, running AdaBoost forever does overfit. The statistically consistent versions of AdaBoost stop early, after a number of rounds growing more slowly than $n$ (Jiang, 2004; Bartlett &amp; Traskin, 2007).</p>
</div>`,
      },
      {
        title: 'Gradient boosting: descent in function space',
        html: R`
<p>Forward stagewise fitting is only tractable for special losses. Friedman (2001) and Mason et al. (2000) noticed that it can be read as <strong>gradient descent</strong>. Think of the fitted values $\mathbf F=(F(x_1),\dots,F(x_n))$ as the parameters. The steepest-descent direction for the empirical risk $\sum_iL(y_i,F(x_i))$ is the negative gradient
$$r_{im}=-\left.\frac{\partial L(y_i,F)}{\partial F}\right|_{F=F_{m-1}(x_i)}.$$
That direction is defined only at the training points. To generalise, project it onto the base-learner class by fitting a regression tree to the pairs $(x_i,r_{im})$ by least squares. Then take a step.</p>
<div class="algo" data-name="Gradient tree boosting (Friedman, 2001)">
<ol>
<li>Initialise $F_0(x)=\argmin_c\sum_iL(y_i,c)$.</li>
<li>For $m=1,\dots,M$:
<ol>
<li>Compute pseudo-residuals $r_{im}$ (the negative gradient at $F_{m-1}$).</li>
<li>Fit a $J$-leaf regression tree to $\{(x_i,r_{im})\}$, giving regions $R_{jm}$.</li>
<li>For each leaf, take the exact line search $\gamma_{jm}=\argmin_\gamma\sum_{x_i\in R_{jm}}L(y_i,F_{m-1}(x_i)+\gamma)$.</li>
<li>Update $F_m(x)=F_{m-1}(x)+\nu\sum_j\gamma_{jm}\ind(x\in R_{jm})$ with learning rate $0 \lt \nu\le1$.</li>
</ol></li>
</ol>
</div>
<table>
<thead><tr><th>Loss $L(y,F)$</th><th>Negative gradient $r$</th><th>Leaf value $\gamma$</th></tr></thead>
<tbody>
<tr><td>Squared $\tfrac12(y-F)^2$</td><td>$y-F$ (ordinary residual)</td><td>mean of residuals</td></tr>
<tr><td>Absolute $|y-F|$</td><td>$\sgn(y-F)$</td><td>median of residuals</td></tr>
<tr><td>Huber$_\delta$</td><td>$y-F$ if $|y-F|\le\delta$, else $\delta\sgn(y-F)$</td><td>robust location (approx.)</td></tr>
<tr><td>Binomial deviance, $y\in\{0,1\}$, $p=\sigma(F)$</td><td>$y-p$</td><td>one Newton step: $\sum r_i/\sum p_i(1-p_i)$</td></tr>
<tr><td>Exponential $e^{-yF}$, $y\in\{\pm1\}$</td><td>$ye^{-yF}$</td><td>recovers AdaBoost-type steps</td></tr>
</tbody>
</table>
<p>With squared loss, gradient boosting is literally "fit the residuals, add a fraction of the fit, repeat". With absolute loss the tree is fitted to signs, so each point's influence is bounded, and the leaf values are medians. That makes the method robust to outliers in $y$, which the lab lets you check.</p>
<p><strong>Regularisation.</strong> Three knobs control overfitting. (1) The <em>number of rounds</em> $M$: unlike a random forest, boosting does overfit as $M$ grows, so $M$ is chosen by validation or early stopping. (2) <em>Shrinkage</em> $\nu$: Friedman found empirically that small values ($\nu\le0.1$) give lower test error than $\nu=1$ at the price of larger $M$. A theoretical reading: in the limit $\nu\to0$, incremental forward stagewise fitting traces a path closely related to the <a href="#/t/lasso">lasso</a> path (identical under monotonicity conditions; Efron, Hastie, Johnstone &amp; Tibshirani, 2004). Small steps behave like a smoothly increasing $\ell_1$ budget. (3) <em>Subsampling</em>: fitting each tree on a random fraction $\eta\approx0.5$ of the data (stochastic gradient boosting; Friedman, 2002) adds bagging-like variance reduction and speeds things up.</p>
<p><strong>Tree size.</strong> A tree with $J$ leaves can represent interactions of order at most $J-1$. With stumps ($J=2$) the boosted model is <em>additive</em> in the inputs, a fitted generalised additive model. Values $4\le J\le8$ work well in most problems.</p>`,
        lab: 'boosting-gradient',
      },
      {
        title: 'Second-order steps and XGBoost',
        html: R`
<p>Gradient boosting takes a first-order step. XGBoost (Chen &amp; Guestrin, 2016) and related systems take a <em>Newton</em> step with an explicit penalty on tree complexity. At round $m$, add a tree $h$ with $T$ leaves and leaf weights $w_1,\dots,w_T$, and minimise
$$\sum_{i=1}^nL\big(y_i,F_{m-1}(x_i)+h(x_i)\big)+\Omega(h),\qquad\Omega(h)=\gamma T+\tfrac12\lambda\sum_{j=1}^Tw_j^2.$$
Expand the loss to second order with $g_i=\partial_FL(y_i,F_{m-1}(x_i))$ and $h_i=\partial_F^2L(y_i,F_{m-1}(x_i))$:
$$\sum_i\Big[g_ih(x_i)+\tfrac12h_ih(x_i)^2\Big]+\Omega(h)+\text{const}=\sum_{j=1}^T\Big[G_jw_j+\tfrac12(H_j+\lambda)w_j^2\Big]+\gamma T+\text{const},$$
where $G_j=\sum_{i\in I_j}g_i$ and $H_j=\sum_{i\in I_j}h_i$ over the points in leaf $j$.</p>
<div class="thm" data-kind="Proposition" data-name="Optimal leaf weights and split gain">
<p>For a fixed tree structure the minimising leaf weights and the attained objective are
$$w_j^*=-\frac{G_j}{H_j+\lambda},\qquad \text{obj}^*=-\frac12\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma T.$$
Splitting a leaf into $L$ and $R$ therefore reduces the objective by
$$\text{Gain}=\frac12\left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right]-\gamma .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The objective separates over leaves into quadratics $G_jw+\frac12(H_j+\lambda)w^2$. When $H_j+\lambda \gt 0$ (true for convex losses with $\lambda \gt 0$), each is minimised at $w=-G_j/(H_j+\lambda)$ with value $-\frac12G_j^2/(H_j+\lambda)$. The gain is the before-minus-after difference for one leaf replaced by two, which adds one leaf and so costs $\gamma$.</p>
</details>
<p>For squared loss $g_i=F-y_i$ and $h_i=1$, so $w_j^*$ is the leaf's mean residual shrunk toward zero by $\lambda$. For the logistic loss, $g_i=p_i-y_i$ and $h_i=p_i(1-p_i)$, and $w^*_j$ is a penalised Newton step. The gain formula replaces the impurity criterion of CART, and $\gamma$ acts as a built-in pruning threshold: splits with negative gain are not made. Practical systems add column subsampling, histogram-binned split finding (LightGBM; Ke et al., 2017), and sparsity-aware handling of missing values.</p>`,
      },
      {
        title: 'Boosting vs. bagging, and pitfalls',
        html: R`
<div class="cols">
<div><h4>Bagging / random forests</h4><p>Parallel; averages <em>low-bias, high-variance</em> deep trees; reduces variance; cannot overfit by adding trees; few tuning parameters.</p></div>
<div><h4>Boosting</h4><p>Sequential; adds <em>high-bias, low-variance</em> shallow trees; reduces bias (and some variance); overfits as $M$ grows; tune $M,\nu,J$, subsample.</p></div>
</div>
<div class="warn" data-name="Label noise and exponential loss">
<p>Mislabelled points get exponentially growing weight under AdaBoost and can dominate the later rounds. Prefer the deviance or a robust loss (Huber, absolute) when labels or responses are noisy.</p>
</div>
<div class="warn" data-name="Early stopping is the model selection">
<p>The number of rounds is a smoothing parameter. Choose it by validation (with a fixed small $\nu$) and report validation error from data not used to choose it. Boosting's training loss goes to zero for flexible base learners and says nothing about generalisation.</p>
</div>
<div class="warn" data-name="Scores are not probabilities">
<p>AdaBoost's $F$ estimates half the log-odds in the population, but finite-sample boosted scores are often over-confident. Calibrate them (Platt scaling or isotonic regression) before reading them as probabilities; see <a href="#/t/classifier-evaluation">classifier evaluation</a>.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`A weak learner has weighted error $\varepsilon_m=0.3$. Compute $\alpha_m$ and the factor by which this round multiplies the training-error bound.`,
        a: R`$\alpha_m=\tfrac12\log(0.7/0.3)\approx0.424$. The bound is multiplied by $Z_m=2\sqrt{0.3\cdot0.7}=2\sqrt{0.21}\approx0.917$. Ten such rounds give a bound of about $0.917^{10}\approx0.42$; fifty rounds give about $0.013$.` },
      { q: R`Why can AdaBoost never select the same weak classifier in two consecutive rounds?`,
        a: R`After the update, $G_m$'s weighted error under $w^{(m+1)}$ is exactly $\tfrac12$ (the misclassified points carry total weight $\varepsilon e^{\alpha}/Z=\tfrac12$). A classifier with error $\tfrac12$ has edge zero; AdaBoost would either pick something strictly better or stop.` },
      { q: R`Exponential loss and binomial deviance have the same population minimiser. Why, then, is AdaBoost more fragile than LogitBoost under label noise?`,
        a: R`The population minimiser concerns the conditional risk at each $x$ with the true $\eta(x)$. In a finite sample, a mislabelled point has a large negative margin $z=yF$. Exponential loss charges it $e^{-z}$ and gives it weight proportional to that, so it grows exponentially. Deviance charges about $-2z$, with gradient bounded by $1$. One point can dominate an exponential-loss fit, but not a deviance fit.` },
      { q: R`In gradient boosting with absolute loss, the tree is fitted to $\sgn(y_i-F_{m-1}(x_i))$, yet the leaf values are medians of the raw residuals. Why both?`,
        a: R`The sign is the negative gradient of $|y-F|$, so fitting a least-squares tree to it chooses the partition along the steepest-descent direction. The step size, though, should minimise the actual loss within each leaf. For absolute loss, $\argmin_\gamma\sum_{i\in R_j}|r_i-\gamma|$ over the residuals $r_i=y_i-F_{m-1}(x_i)$ is their median: the exact line search.` },
      { q: R`For logistic loss with $\lambda=1$, a leaf contains four points with $p_i=0.5$, and labels $y=(1,1,1,0)$. What leaf weight does XGBoost assign?`,
        a: R`$g_i=p_i-y_i=(-0.5,-0.5,-0.5,0.5)$, so $G=-1$. $h_i=p_i(1-p_i)=0.25$, so $H=1$. Then $w^*=-G/(H+\lambda)=1/2=0.5$: a step of $0.5$ on the log-odds scale toward class 1, shrunk from the unpenalised Newton step $1$ by $\lambda$.` },
    ],
  });

  /* ===================================================================== */
  Atlas.topic({
    id: 'svm',
    domain: 'supervised',
    order: 120,
    title: 'Support Vector Machines & the Margin',
    subtitle: 'Maximal margins, Lagrangian duality, and the hinge loss',
    level: 3,
    sources: { aos: '22.7', other: 'Vapnik (1995); Schölkopf & Smola (2002) Ch. 7; ESL §4.5.2, 12.2–12.3' },
    prereqs: ['learning-framework', 'vc-theory'],
    related: ['kernel-methods', 'logistic-regression', 'boosting', 'lda-qda', 'classifier-evaluation'],
    tags: ['maximal margin', 'convex duality', 'KKT', 'support vectors', 'hinge loss', 'SMO'],
    summary: R`A support vector machine separates two classes with the hyperplane that leaves the widest possible empty "street" between them. Maximising that margin is a convex quadratic program. Its Lagrangian dual involves the data only through inner products, and its KKT conditions show that the solution depends only on the points on or inside the margin: the support vectors. Allowing violations at a cost $C$ gives the soft-margin SVM, which is exactly regularised empirical risk minimisation with the hinge loss $[1-yf(x)]_+$ and a ridge penalty. The hinge loss targets the Bayes decision $\sgn(2\eta-1)$ directly rather than the probability $\eta$. Generalisation is controlled by the margin relative to the data radius, not by the dimension, which is what lets the kernel trick move the problem into enormous feature spaces safely.`,
    survey: {
      task: 'Binary classification (extensions: regression, multiclass, novelty detection)',
      assumptions: R`IID data; features on comparable scales; for nonlinear boundaries, a suitable kernel`,
      output: R`Decision function $f(x)=\sum_{i\in SV}\alpha_iy_iK(x_i,x)+b$; a label, no probability`,
      strengths: 'Convex (unique optimum), sparse in the data, dimension-free margin bounds, kernelisable',
      weaknesses: R`No native probabilities; sensitive to feature scaling and to $C,\gamma$; training $O(n^2)$–$O(n^3)$`,
      complexity: R`Training roughly $O(n^2)$ to $O(n^3)$ (SMO); prediction $O(\#SV\cdot d)$`,
    },
    sections: [
      {
        title: 'The maximal-margin hyperplane',
        html: R`
<p>Take labels $y_i\in\{-1,+1\}$ and a hyperplane $\{x: w^\top x+b=0\}$. It classifies by $\sgn(w^\top x+b)$, and the signed distance of $x_i$ from it is $y_i(w^\top x_i+b)/\|w\|$, positive when $x_i$ is on the correct side. If the data are linearly separable, infinitely many hyperplanes separate them. The SVM picks the one whose smallest distance to the data, the <strong>margin</strong>, is largest:
$$\max_{w,b}\;\min_i\frac{y_i(w^\top x_i+b)}{\|w\|}.$$
The pair $(w,b)$ can be rescaled freely without moving the hyperplane, so fix the scale by requiring $\min_iy_i(w^\top x_i+b)=1$. The margin is then $1/\|w\|$, the street has width $2/\|w\|$, and the problem becomes a convex quadratic program:</p>
$$\min_{w,b}\ \tfrac12\|w\|^2\quad\text{subject to}\quad y_i(w^\top x_i+b)\ge1,\quad i=1,\dots,n.$$
<div class="thm" data-kind="Proposition" data-name="Existence and uniqueness">
<p>If the data are linearly separable and both classes are present, the hard-margin problem has a unique solution $(w^*,b^*)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The feasible set is a non-empty closed convex polyhedron in $(w,b)$, and $\tfrac12\|w\|^2$ is continuous and coercive in $w$. Moreover, for bounded $w$, $b$ is confined to a bounded interval by the constraints for the two classes. So a minimiser exists. The objective is strictly convex in $w$, so $w^*$ is unique: two different minimisers $w_1\ne w_2$ would make $\tfrac12(w_1+w_2)$, which is feasible with $\tfrac12(b_1+b_2)$, strictly better. Given $w^*$, the active constraints include points of both classes (otherwise $b$ could be shifted to make all constraints slack, and then $w$ scaled down). Equality on one point of each class pins down $b^*$.</p>
</details>
<p>Why the widest street? Intuitively, a wide margin is robust: every training point can be perturbed by up to the margin without changing its label. More formally, the class of wide-margin separators is small in a sense that does not depend on the dimension (see the generalisation section below).</p>`,
      },
      {
        title: 'Lagrangian duality and the KKT conditions',
        html: R`
<p>Introduce multipliers $\alpha_i\ge0$ for the constraints:
$$\mathcal L(w,b,\alpha)=\tfrac12\|w\|^2-\sum_{i=1}^n\alpha_i\big[y_i(w^\top x_i+b)-1\big].$$
The primal problem is $\min_{w,b}\max_{\alpha\ge0}\mathcal L$. The dual swaps the order. For fixed $\alpha$, minimising over $(w,b)$ gives the stationarity conditions
$$\nabla_w\mathcal L=0\;\Rightarrow\;w=\sum_i\alpha_iy_ix_i,\qquad \partial_b\mathcal L=0\;\Rightarrow\;\sum_i\alpha_iy_i=0 .$$
(If $\sum_i\alpha_iy_i\ne0$, the infimum over $b$ is $-\infty$.) Substituting back:</p>
$$\max_{\alpha}\ \sum_{i=1}^n\alpha_i-\frac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\,x_i^\top x_j\quad\text{s.t.}\quad\alpha_i\ge0,\ \sum_i\alpha_iy_i=0 .$$
<div class="thm" data-kind="Theorem" data-name="Strong duality and KKT for the SVM">
<p>If the hard-margin problem is feasible, the primal and dual optimal values coincide. $(w^*,b^*)$ and $\alpha^*$ are optimal if and only if they satisfy the KKT conditions: primal feasibility $y_i(w^{*\top}x_i+b^*)\ge1$; dual feasibility $\alpha^*_i\ge0$; stationarity $w^*=\sum_i\alpha^*_iy_ix_i$, $\sum_i\alpha_i^*y_i=0$; and complementary slackness
$$\alpha_i^*\big[y_i(w^{*\top}x_i+b^*)-1\big]=0\quad\text{for all }i.$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The objective is convex and the constraints are affine. For such problems feasibility alone implies strong duality and the existence of dual optimal multipliers (the refined Slater condition, which needs no strict feasibility for affine constraints). Given zero duality gap, $\tfrac12\|w^*\|^2=\mathcal L(w^*,b^*,\alpha^*)$. Since $\tfrac12\|w^*\|^2 - \mathcal L = \sum_i\alpha_i^*[y_i(w^{*\top}x_i+b^*)-1]$ is a sum of non-negative terms, each term must vanish. That is complementary slackness. Conversely, any point satisfying KKT has zero gap and is therefore optimal for both problems.</p>
</details>
<p>Complementary slackness is the heart of the method. If $x_i$ lies strictly outside the margin, $y_if(x_i) \gt 1$, then $\alpha_i^*=0$, and it does not appear in $w^*=\sum_i\alpha_i^*y_ix_i$. Only the <strong>support vectors</strong>, the points with $\alpha_i^* \gt 0$, which lie exactly on the margin, determine the solution. Delete or move any other point (without crossing the margin) and nothing changes. The offset is recovered from any support vector: $b^*=y_s-w^{*\top}x_s$. Numerically it is better to average over all free support vectors.</p>
<div class="note">
<p>The dual depends on the inputs only through the inner products $x_i^\top x_j$, and so does the decision function $f(x)=\sum_i\alpha_iy_i\,x_i^\top x+b$. Replacing $x^\top x'$ by a kernel $K(x,x')$ fits a maximal-margin hyperplane in a feature space we never compute. This is the <a href="#/t/kernel-methods">kernel trick</a>.</p>
</div>`,
      },
      {
        title: 'The soft margin',
        html: R`
<p>Real data are rarely separable, and even when they are, the widest street may be dictated by one outlier. The <strong>soft-margin SVM</strong> (Cortes &amp; Vapnik, 1995) introduces slack variables $\xi_i\ge0$ that let points enter the street or cross it, at linear cost:
$$\min_{w,b,\xi}\ \tfrac12\|w\|^2+C\sum_{i=1}^n\xi_i\quad\text{s.t.}\quad y_i(w^\top x_i+b)\ge1-\xi_i,\ \ \xi_i\ge0 .$$
With multipliers $\alpha_i$ for the margin constraints and $\mu_i$ for $\xi_i\ge0$, stationarity in $\xi_i$ gives $C-\alpha_i-\mu_i=0$. Since $\mu_i\ge0$, this means $\alpha_i\le C$. Everything else goes through as before, and the dual changes only by a <em>box constraint</em>:
$$\max_\alpha\ \sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_jK(x_i,x_j)\quad\text{s.t.}\quad0\le\alpha_i\le C,\ \sum_i\alpha_iy_i=0 .$$</p>
<p>Complementary slackness now sorts the training points into three groups (write $f=w^\top x+b$):</p>
<table>
<thead><tr><th>Dual value</th><th>Primal condition</th><th>Geometry</th></tr></thead>
<tbody>
<tr><td>$\alpha_i=0$</td><td>$y_if(x_i)\ge1$</td><td>outside the street, correct side; irrelevant to the fit</td></tr>
<tr><td>$0 \lt \alpha_i \lt C$</td><td>$y_if(x_i)=1$, $\xi_i=0$</td><td>exactly on the margin ("free" SV)</td></tr>
<tr><td>$\alpha_i=C$</td><td>$y_if(x_i)\le1$, $\xi_i=1-y_if(x_i)$</td><td>inside the street or misclassified ("bounded" SV)</td></tr>
</tbody>
</table>
<p>The first row follows from $\alpha_i=0\Rightarrow\mu_i=C \gt 0\Rightarrow\xi_i=0$. In the second, $\mu_i \gt 0$ forces $\xi_i=0$ and $\alpha_i \gt 0$ forces the constraint to be tight. In the third, $\mu_i=0$ allows $\xi_i \gt 0$. The offset $b$ is computed from the free support vectors, for which $y_if(x_i)=1$ exactly.</p>
<div class="algo" data-name="Sequential minimal optimisation (Platt, 1998)">
<ol>
<li>The constraint $\sum_i\alpha_iy_i=0$ means a single $\alpha_i$ cannot move alone, but a pair can.</li>
<li>Pick the pair $(i,j)$ that most violates the KKT conditions: the largest gap between $-y_t\nabla_t$ over the coordinates that can move "up" and over those that can move "down".</li>
<li>Minimise the dual objective exactly along the feasible line for $(\alpha_i,\alpha_j)$ and clip to the box $[0,C]^2$.</li>
<li>Update the gradient (cost $O(n)$), and stop when the maximal violation falls below a tolerance.</li>
</ol>
</div>
<p>The lab's solver is exactly this (with maximal-violating-pair selection, as in LIBSVM). Click points into the plane. Moving a non-support vector leaves the fit unchanged, while dragging $C$ down widens the street and turns more points into bounded support vectors.</p>`,
        lab: 'svm-margin',
      },
      {
        title: 'Hinge loss: the SVM as regularised risk minimisation',
        html: R`
<p>At the optimum each slack is as small as allowed: $\xi_i=\max\big(0,1-y_if(x_i)\big)=[1-y_if(x_i)]_+$. Substituting and dividing by $C$ gives an unconstrained problem:
$$\min_{w,b}\ \sum_{i=1}^n\big[1-y_i(w^\top x_i+b)\big]_+ +\frac{\lambda}{2}\|w\|^2,\qquad\lambda=\frac1C .$$
This is ridge-penalised <a href="#/t/learning-framework">empirical risk minimisation</a> with the <strong>hinge loss</strong> $\phi(z)=[1-z]_+$ of the margin $z=yf(x)$. Seen this way, the SVM sits in a family of convex surrogates for the 0–1 loss $\ind(z\le0)$:</p>
<table>
<thead><tr><th>Method</th><th>Loss $\phi(z)$, $z=yf$</th><th>Population minimiser $f^*(x)$</th></tr></thead>
<tbody>
<tr><td>SVM</td><td>$[1-z]_+$</td><td>$\sgn(2\eta(x)-1)$</td></tr>
<tr><td>Logistic regression</td><td>$\log(1+e^{-z})$</td><td>$\log\frac{\eta}{1-\eta}$</td></tr>
<tr><td>AdaBoost</td><td>$e^{-z}$</td><td>$\frac12\log\frac{\eta}{1-\eta}$</td></tr>
<tr><td>Squared hinge</td><td>$[1-z]_+^2$</td><td>$2\eta-1$</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Theorem" data-name="The hinge loss targets the Bayes rule (Lin, 2002)">
<p>Let $\eta(x)=P(Y=1\mid X=x)$. The minimiser over $f(x)\in\R$ of $\E\big[[1-Yf(x)]_+\mid X=x\big]$ is $f^*(x)=\sgn(2\eta(x)-1)$ (any value in $[-1,1]$ when $\eta=\tfrac12$). Hence $\sgn f^*$ is the <a href="#/t/bayes-classifier">Bayes classifier</a>.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The conditional risk is $r(f)=\eta[1-f]_++(1-\eta)[1+f]_+$. For $f\ge1$, $r(f)=(1-\eta)(1+f)\ge2(1-\eta)=r(1)$. For $f\le-1$, $r(f)=\eta(1-f)\ge2\eta=r(-1)$. On $[-1,1]$, $r(f)=1+f(1-2\eta)$ is linear, minimised at $f=1$ if $\eta \gt \tfrac12$ and at $f=-1$ if $\eta \lt \tfrac12$.</p>
</details>
<p>Two lessons. First, the hinge loss is <em>classification-calibrated</em>: minimising it leads to the Bayes decision. Bartlett, Jordan &amp; McAuliffe (2006) quantify this. For the hinge loss the excess 0–1 risk is bounded by the excess hinge risk, $R(\sgn f)-R^*\le R_\phi(f)-R_\phi^*$. Second, the hinge loss does <em>not</em> estimate $\eta(x)$: its minimiser jumps from $-1$ to $+1$, and $|f(x)|$ carries no probability information. That is why SVM outputs need Platt scaling (a logistic regression of $y$ on $f(x)$, fitted on held-out data) when probabilities are wanted. It also explains sparsity. Points with $yf \gt 1$ contribute zero loss and zero gradient, so they drop out entirely, which never happens with the logistic loss.</p>`,
        lab: 'svm-regularization-path',
      },
      {
        title: 'Why large margins generalise',
        html: R`
<p>An SVM with an RBF kernel works in an infinite-dimensional feature space, where VC theory for arbitrary hyperplanes gives nothing. The saving grace is that capacity can be measured by the margin.</p>
<div class="thm" data-kind="Proposition" data-name="Leave-one-out bound (Vapnik)">
<p>Suppose the SVM solution is unique. Its leave-one-out error is at most $N_{SV}/n$, the fraction of support vectors. Consequently the expected test error of the SVM trained on $n-1$ points satisfies $\E\,R(f_{n-1})\le\E[N_{SV}(n)]/n$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>If $\alpha_i^*=0$, removing $x_i$ leaves the remaining coordinates of $\alpha^*$ feasible for the reduced dual. The KKT conditions for the remaining points involve only $w^*$ and $b^*$, which are unchanged, so the reduced solution is the same classifier. By the table above, $\alpha_i^*=0$ means $y_if(x_i)\ge1 \gt 0$, so the left-out point is classified correctly. Leave-one-out errors can therefore only occur at support vectors. The second statement uses the fact that the leave-one-out error on $n$ points is an unbiased estimate of the expected risk of a classifier trained on $n-1$ points (Luntz &amp; Brailovsky, 1969).</p>
</details>
<p>A solution with few support vectors cannot overfit badly, whatever the dimension of the feature space. A complementary uniform bound uses Rademacher complexity (<a href="#/t/vc-theory">generalisation theory</a>):</p>
<div class="thm" data-kind="Theorem" data-name="Margin bound for kernel classifiers">
<p>Let $\mathcal H=\{x\mapsto\langle w,\varphi(x)\rangle:\|w\|\le\Lambda\}$ with $K(x,x)\le r^2$. Fix $\rho \gt 0$. With probability at least $1-\delta$, every $h\in\mathcal H$ satisfies
$$P\big(Yh(X)\le0\big)\le\frac1n\sum_{i=1}^n\min\Big(1,\Big[1-\frac{y_ih(x_i)}{\rho}\Big]_+\Big)+\frac{2r\Lambda}{\rho\sqrt n}+3\sqrt{\frac{\log(2/\delta)}{2n}} .$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The ramp loss $\min(1,[1-z/\rho]_+)$ dominates $\ind(z\le0)$ and is $1/\rho$-Lipschitz. The standard Rademacher generalisation bound plus Talagrand's contraction lemma gives the empirical ramp loss $+\,2\hat{\mathfrak R}_n(\mathcal H)/\rho+3\sqrt{\log(2/\delta)/(2n)}$. By Cauchy–Schwarz and Jensen, $\hat{\mathfrak R}_n(\mathcal H)=\frac1n\E_\sigma\sup_{\|w\|\le\Lambda}\langle w,\sum_i\sigma_i\varphi(x_i)\rangle=\frac{\Lambda}{n}\E_\sigma\|\sum_i\sigma_i\varphi(x_i)\|\le\frac{\Lambda}{n}\sqrt{\sum_iK(x_i,x_i)}\le\frac{\Lambda r}{\sqrt n}$.</p>
</details>
<p>Only $r\Lambda/\rho$ appears: the ratio of the data radius to the margin, as in Vapnik's classical result that hyperplanes with margin $\rho$ on data in a ball of radius $r$ have VC dimension at most $\min(\lceil r^2/\rho^2\rceil,d)+1$. The dimension $d$ of the feature space need not enter. The trade-off in $C$ is visible in the bound. A small $\|w\|$ (large margin) shrinks the complexity term, while a small slack sum keeps the empirical term low. The soft-margin objective balances exactly these two quantities.</p>`,
      },
      {
        title: 'Practice and pitfalls',
        html: R`
<div class="warn" data-name="Scale your features">
<p>Unlike trees, SVMs depend on Euclidean geometry. A feature measured in thousands dominates both the inner product and the RBF distance. Standardise first, using statistics from the training data only.</p>
</div>
<div class="warn" data-name="Tune C and γ jointly, on a log grid">
<p>For the RBF kernel $K(x,x')=e^{-\gamma\|x-x'\|^2}$, large $\gamma$ makes every point its own island. Training error goes to zero, nearly all points become support vectors, and test error rises. Small $\gamma$ approaches a linear (or constant) classifier. $C$ and $\gamma$ interact strongly, so cross-validate over a two-dimensional grid of powers of ten.</p>
</div>
<div class="warn" data-name="Cost">
<p>Kernel SVM training scales between $O(n^2)$ and $O(n^3)$ and needs the kernel matrix, or a cache of it. Beyond about $10^5$ points use a linear SVM (coordinate descent, cost linear in $n$) or approximate kernels (random features, Nyström; see <a href="#/t/kernel-methods">kernel methods</a>).</p>
</div>
<p><strong>Extensions.</strong> Multiclass problems use one-vs-rest or one-vs-one voting, or the Crammer–Singer joint formulation. <em>Support vector regression</em> uses the $\varepsilon$-insensitive loss $[|y-f(x)|-\varepsilon]_+$, which keeps the sparsity: points inside the tube are not support vectors. The $\nu$-SVM reparameterises $C$ by $\nu\in(0,1]$, an upper bound on the fraction of margin errors and a lower bound on the fraction of support vectors. Class imbalance is handled with class-specific costs $C_+,C_-$.</p>
<p><strong>Compared with logistic regression</strong>, the two solve nearly the same regularised problem with different surrogate losses. Logistic regression gives calibrated probabilities and uses every point. The SVM gives a sparse solution and estimates only the decision boundary. With good tuning their accuracies are usually close.</p>`,
      },
    ],
    quiz: [
      { q: R`In the hard-margin dual you find $\alpha^*$ with three non-zero entries. What can you say about the other $n-3$ training points, and what happens to the classifier if you delete one of them?`,
        a: R`By complementary slackness, the points with $\alpha_i^*=0$ satisfy $y_if(x_i)\ge1$: they are on the correct side, on or outside the margin, and in the generic case strictly outside. Since $w^*=\sum\alpha_i^*y_ix_i$ involves only the three support vectors, and $b^*$ is computed from them, deleting a non-support vector leaves the solution unchanged.` },
      { q: R`Derive the box constraint $0\le\alpha_i\le C$ of the soft-margin dual.`,
        a: R`The Lagrangian has $-\sum\alpha_i[y_if(x_i)-1+\xi_i]-\sum\mu_i\xi_i+C\sum\xi_i$. Stationarity in $\xi_i$ gives $C-\alpha_i-\mu_i=0$. With $\mu_i\ge0$ and $\alpha_i\ge0$ this yields $0\le\alpha_i\le C$. The $\xi_i$ then cancel from the Lagrangian, leaving the same dual objective as in the hard-margin case.` },
      { q: R`Why can't you read a class probability from $|f(x)|$ of a trained SVM, while you can (approximately) for logistic regression?`,
        a: R`The population minimiser of the hinge loss is $\sgn(2\eta(x)-1)$: it takes only the values $\pm1$, whatever $\eta(x)$ is, so the hinge loss gives no incentive to make $f$ track $\eta$. The logistic loss's minimiser is the log-odds, a strictly increasing function of $\eta$, so $\hat f$ estimates it. For SVM probabilities, fit a separate calibration map such as Platt scaling.` },
      { q: R`An RBF-SVM achieves zero training error with 95 of 100 points as support vectors. What does the leave-one-out bound say, and what would you change?`,
        a: R`The bound $N_{SV}/n=0.95$ is vacuous. The model may be memorising, since nearly every point is needed to define the boundary. Reduce $\gamma$ (smoother kernel) and/or reduce $C$, choosing both by cross-validation. A good fit typically needs a much smaller fraction of support vectors unless the classes overlap heavily.` },
      { q: R`Write the soft-margin SVM as $\min_f \sum_i L(y_i,f(x_i))+\frac\lambda2\|w\|^2$. What is $\lambda$ in terms of $C$, and what happens as $C\to\infty$ on separable data?`,
        a: R`$L$ is the hinge loss and $\lambda=1/C$. As $C\to\infty$ the penalty on $\|w\|$ vanishes relative to the hinge loss, so any positive slack is infinitely costly. On separable data the solution converges to the hard-margin SVM.` },
    ],
  });

  /* ===================================================================== */
  Atlas.topic({
    id: 'kernel-methods',
    domain: 'supervised',
    order: 130,
    title: 'Kernels, RKHS & the Representer Theorem',
    subtitle: 'Inner products in feature spaces, positive definiteness, and kernel ridge regression',
    level: 3,
    sources: { aos: '22.7', other: 'Schölkopf & Smola (2002) Ch. 2–4; Rasmussen & Williams (2006) §6.1–6.2; ESL §5.8, 12.3; Aronszajn (1950)' },
    prereqs: ['svm', 'ridge-regression'],
    related: ['gaussian-processes', 'splines', 'kernel-regression', 'pca'],
    tags: ['kernel trick', 'Mercer', 'RKHS', 'representer theorem', 'kernel ridge regression', 'random features'],
    summary: R`A kernel $K(x,x')$ is an inner product $\langle\varphi(x),\varphi(x')\rangle$ in some feature space, computed without ever forming $\varphi$. Any algorithm that touches the data only through inner products (SVMs, ridge regression, PCA, distance-based methods) can therefore be run in a feature space of enormous or infinite dimension at the cost of an $n\times n$ Gram matrix. Which functions are kernels? Exactly the symmetric positive-definite ones, and each such kernel generates a unique Hilbert space of functions, its RKHS, in which evaluation is an inner product with $K(\cdot,x)$. The representer theorem then says that regularised risk minimisation over this infinite-dimensional space has a solution in the $n$-dimensional span of $K(\cdot,x_1),\dots,K(\cdot,x_n)$. For squared loss this gives kernel ridge regression in closed form, $\hat f(x)=k(x)^\top(K+\lambda I)^{-1}y$.`,
    survey: {
      task: 'Nonlinear regression/classification/unsupervised learning through linear methods in feature space',
      assumptions: R`A symmetric positive-definite kernel encoding the notion of similarity; IID data`,
      output: R`$\hat f(x)=\sum_i\alpha_iK(x,x_i)$; for KRR $\alpha=(K+\lambda I)^{-1}y$`,
      strengths: 'Convex problems with global optima; principled function spaces; modular (kernel = prior knowledge); closed forms',
      weaknesses: R`$O(n^2)$ memory and $O(n^3)$ time; kernel and bandwidth choice matter; local kernels suffer in high dimension`,
      complexity: R`Exact: $O(n^3)$ time, $O(n^2)$ memory; random features or Nyström with rank $m$: $O(nm^2)$`,
    },
    sections: [
      {
        title: 'Feature maps and the kernel trick',
        html: R`
<p>Linear methods are well understood, convex, and cheap, but linear in the raw inputs is often too restrictive. The classical remedy is a feature map $\varphi:\mathcal X\to\mathcal F$ followed by a linear method in $\mathcal F$, as in polynomial regression. The obstacle is dimension: all monomials of degree at most $d$ in $p$ variables number $\binom{p+d}{d}$, which is over $10^{10}$ for $p=100$ and $d=6$.</p>
<p>The kernel trick removes the obstacle whenever the method needs only inner products $\langle\varphi(x),\varphi(x')\rangle$. Take $x\in\R^2$ and $\varphi(x)=(x_1^2,\sqrt2\,x_1x_2,x_2^2)$. Then
$$\langle\varphi(x),\varphi(x')\rangle=x_1^2x_1'^2+2x_1x_2x_1'x_2'+x_2^2x_2'^2=(x^\top x')^2 .$$
More generally $K(x,x')=(x^\top x'+c)^d$ is the inner product of a feature vector containing all (suitably weighted) monomials up to degree $d$, evaluated in $O(p)$ operations instead of $O(p^d)$. The Gaussian (RBF) kernel goes further:
$$e^{-\|x-x'\|^2/(2\ell^2)}=e^{-\|x\|^2/(2\ell^2)}\,e^{-\|x'\|^2/(2\ell^2)}\sum_{k=0}^\infty\frac{(x^\top x')^k}{\ell^{2k}\,k!},$$
an inner product between infinite feature vectors containing monomials of every degree.</p>
<p>Many algorithms are "inner-product only". These include the <a href="#/t/svm">SVM</a> dual, ridge regression (via the identity below), PCA (via the Gram matrix), and anything based on distances, since $\|\varphi(x)-\varphi(x')\|^2=K(x,x)-2K(x,x')+K(x',x')$. Replacing $x^\top x'$ by $K(x,x')$ "kernelises" them. The rest of this page answers three questions. Which $K$ are legitimate? In what function space does the kernelised method operate? Why do its solutions take the form $\sum_i\alpha_iK(\cdot,x_i)$?</p>`,
      },
      {
        title: 'Positive-definite kernels',
        html: R`
<div class="def" data-name="Positive-definite kernel">
<p>A symmetric function $K:\mathcal X\times\mathcal X\to\R$ is <em>positive definite</em> (p.d.; some authors say positive semi-definite) if for every $n$, every $x_1,\dots,x_n\in\mathcal X$ and every $c\in\R^n$,
$$\sum_{i,j}c_ic_jK(x_i,x_j)\ge0,$$
i.e. every Gram matrix $\mathbf K=[K(x_i,x_j)]$ is positive semi-definite.</p>
</div>
<p>Every feature-map kernel is p.d.: $\sum_{i,j}c_ic_j\langle\varphi(x_i),\varphi(x_j)\rangle=\|\sum_ic_i\varphi(x_i)\|^2\ge0$. The converse, that every p.d. kernel is an inner product in some space, is the content of the next section. First, the tools for building kernels.</p>
<div class="thm" data-kind="Proposition" data-name="Closure properties">
<p>If $K_1,K_2$ are p.d., then so are: $aK_1+bK_2$ for $a,b\ge0$; the product $K_1K_2$; $g(x)K_1(x,x')g(x')$ for any function $g$; pointwise limits of p.d. kernels; $\exp(K_1)$; and $K_1(\psi(x),\psi(x'))$ for any map $\psi$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Sums, non-negative scalings and limits preserve the defining inequality. For $g K_1 g$, substitute $c_i\mapsto c_ig(x_i)$. For composition with $\psi$, apply the definition at the points $\psi(x_i)$. For products: the Gram matrix of $K_1K_2$ is the Hadamard product $\mathbf A\circ\mathbf B$ of two p.s.d. matrices. Write $\mathbf B=\sum_k\lambda_kv_kv_k^\top$ with $\lambda_k\ge0$. Then $c^\top(\mathbf A\circ\mathbf B)c=\sum_k\lambda_k(c\circ v_k)^\top\mathbf A(c\circ v_k)\ge0$ (Schur's product theorem). Finally, $\exp(K_1)=\sum_k K_1^k/k!$ is a limit of sums of products.</p>
</details>
<div class="ex" data-name="The Gaussian kernel is positive definite">
<p>$x^\top x'/\ell^2$ is p.d. (a linear kernel), so $\exp(x^\top x'/\ell^2)$ is p.d. Multiplying by $g(x)g(x')$ with $g(x)=e^{-\|x\|^2/(2\ell^2)}$ gives $e^{-\|x-x'\|^2/(2\ell^2)}$, which is therefore p.d.</p>
</div>
<p>For stationary kernels $K(x,x')=k(x-x')$ there is a complete characterisation.</p>
<div class="thm" data-kind="Theorem" data-name="Bochner">
<p>A continuous function $k$ on $\R^d$ is positive definite if and only if it is the Fourier transform of a finite non-negative measure $\mu$: $k(\tau)=\int e^{i\omega^\top\tau}\d\mu(\omega)$. When $\mu$ has a density $S(\omega)$ this is the <em>spectral density</em> of the kernel.</p>
</div>
<p>Bochner's theorem makes positive definiteness checkable. The Gaussian kernel has a Gaussian spectral density, and the Laplacian $e^{-|\tau|/\ell}$ has a Cauchy-shaped one. The "box" $\ind(|\tau| \lt \ell)$ has Fourier transform $2\sin(\ell\omega)/\omega$, which takes negative values, so it is <em>not</em> a kernel. The <em>power exponential</em> $e^{-|\tau/\ell|^q}$ is p.d. exactly when $0 \lt q\le2$. For $q\le2$ it is the characteristic function of a symmetric stable law. For $q \gt 2$, if it were p.d., Bochner would make it a characteristic function with $k''(0)=0$, i.e. of a random variable with zero variance, forcing $k\equiv1$, which is a contradiction. The lab lets you watch negative Gram eigenvalues appear as $q$ crosses 2. The popular "sigmoid kernel" $\tanh(a\,x^\top x'+c)$ is not p.d. for most parameter values.</p>
<div class="thm" data-kind="Theorem" data-name="Mercer">
<p>Let $\mathcal X$ be compact, $\nu$ a finite Borel measure with full support, and $K$ continuous and p.d. Then there are an orthonormal basis $\{e_k\}$ of $L^2(\nu)$ and eigenvalues $\lambda_1\ge\lambda_2\ge\cdots\ge0$ with $\int K(x,x')e_k(x')\d\nu(x')=\lambda_ke_k(x)$, such that
$$K(x,x')=\sum_k\lambda_ke_k(x)e_k(x'),$$
with absolute and uniform convergence. In particular $\varphi(x)=(\sqrt{\lambda_k}\,e_k(x))_{k\ge1}$ is an explicit feature map into $\ell^2$.</p>
</div>`,
        lab: 'kernel-methods-gram',
      },
      {
        title: 'Reproducing kernel Hilbert spaces',
        html: R`
<div class="def" data-name="RKHS">
<p>A Hilbert space $\mathcal H$ of functions $\mathcal X\to\R$ is a <em>reproducing kernel Hilbert space</em> with kernel $K$ if (i) $K_x:=K(\cdot,x)\in\mathcal H$ for every $x$, and (ii) the <strong>reproducing property</strong> holds: $\langle f,K_x\rangle_{\mathcal H}=f(x)$ for all $f\in\mathcal H$, $x\in\mathcal X$.</p>
</div>
<p>Applying (ii) to $f=K_{x'}$ gives $\langle K_{x'},K_x\rangle=K(x,x')$. The canonical feature map is $\varphi(x)=K_x$, and the kernel is literally an inner product of features, each of which is a function.</p>
<div class="thm" data-kind="Theorem" data-name="Moore–Aronszajn (Aronszajn, 1950)">
<p>Every positive-definite kernel $K$ is the reproducing kernel of exactly one RKHS $\mathcal H_K$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Construction.</em> Let $\mathcal H_0$ be the span of $\{K_x\}$ and define $\langle\sum_ia_iK_{x_i},\sum_jb_jK_{y_j}\rangle=\sum_{i,j}a_ib_jK(x_i,y_j)$. This is well defined because it equals $\sum_jb_jf(y_j)$ with $f=\sum_ia_iK_{x_i}$, so it does not depend on how $g=\sum_jb_jK_{y_j}$ is represented, and symmetrically for $f$. It is symmetric and bilinear, and $\langle f,f\rangle\ge0$ by positive definiteness. The reproducing property $\langle f,K_x\rangle=f(x)$ holds by construction. <em>Definiteness.</em> By Cauchy–Schwarz, $|f(x)|=|\langle f,K_x\rangle|\le\langle f,f\rangle^{1/2}K(x,x)^{1/2}$, so $\langle f,f\rangle=0$ forces $f\equiv0$. <em>Completion.</em> The same inequality shows that a Cauchy sequence in $\mathcal H_0$ converges pointwise. Adding these pointwise limits completes $\mathcal H_0$ to a Hilbert space of functions in which the reproducing property persists. <em>Uniqueness.</em> Any RKHS with kernel $K$ contains $\mathcal H_0$ with the same inner product, and $\mathcal H_0$ is dense in it (a function orthogonal to every $K_x$ vanishes everywhere).</p>
</details>
<p>Consequences worth internalising:</p>
<ul>
<li><strong>Evaluation is continuous</strong>: $|f(x)-g(x)|\le\|f-g\|_{\mathcal H}\sqrt{K(x,x)}$. Norm convergence implies pointwise (and, for bounded $K$, uniform) convergence. This fails in $L^2$, where functions are only defined almost everywhere.</li>
<li><strong>The norm measures smoothness</strong>: $|f(x)-f(x')|\le\|f\|_{\mathcal H}\,\|K_x-K_{x'}\|_{\mathcal H}$, so functions of small norm cannot vary faster than the kernel allows. In Mercer coordinates $f=\sum_kf_ke_k$ one has $\|f\|^2_{\mathcal H}=\sum_kf_k^2/\lambda_k$. Components along eigenfunctions with tiny eigenvalues, typically high-frequency ones, are expensive. For a stationary kernel, $\|f\|^2_{\mathcal H}$ is proportional to $\int|\hat f(\omega)|^2/S(\omega)\d\omega$: frequencies where the spectral density is small are heavily penalised.</li>
<li><strong>Examples.</strong> Linear kernel $x^\top x'$: $\mathcal H$ is the linear functions $f(x)=w^\top x$ with $\|f\|=\|w\|$. Brownian-motion kernel $\min(s,t)$ on $[0,1]$: $\mathcal H=\{f: f(0)=0,\ f'\in L^2\}$ with $\|f\|^2=\int_0^1f'(t)^2\d t$. Matérn-$\nu$ kernels on $\R^d$: $\mathcal H$ is norm-equivalent to the Sobolev space of order $\nu+d/2$. The Gaussian kernel's RKHS is much smaller: its functions are real-analytic.</li>
</ul>`,
      },
      {
        title: 'The representer theorem',
        html: R`
<p>We want to fit $f\in\mathcal H_K$ by regularised empirical risk minimisation. $\mathcal H_K$ may be infinite-dimensional, but the solution is not.</p>
<div class="thm" data-kind="Theorem" data-name="Representer theorem (Kimeldorf & Wahba, 1971; Schölkopf, Herbrich & Smola, 2001)">
<p>Let $L$ be any loss, $\Omega:[0,\infty)\to\R$ strictly increasing, and consider
$$\min_{f\in\mathcal H_K}\ \sum_{i=1}^nL\big(y_i,f(x_i)\big)+\Omega\big(\|f\|_{\mathcal H_K}\big).$$
Every minimiser has the form $\hat f=\sum_{i=1}^n\alpha_iK(\cdot,x_i)$ for some $\alpha\in\R^n$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $V=\mathrm{span}\{K_{x_1},\dots,K_{x_n}\}$, a closed (finite-dimensional) subspace. Decompose any $f\in\mathcal H_K$ as $f=f_V+f_\perp$ with $f_V\in V$ and $f_\perp\perp V$. By the reproducing property, $f(x_i)=\langle f,K_{x_i}\rangle=\langle f_V,K_{x_i}\rangle+\langle f_\perp,K_{x_i}\rangle=f_V(x_i)$, because $f_\perp$ is orthogonal to each $K_{x_i}$. So the loss term depends only on $f_V$. By Pythagoras $\|f\|^2=\|f_V\|^2+\|f_\perp\|^2$, so $\Omega(\|f\|)\ge\Omega(\|f_V\|)$, with strict inequality if $f_\perp\ne0$. Replacing $f$ by $f_V$ therefore never increases the objective, and strictly decreases it unless $f_\perp=0$. Every minimiser lies in $V$.</p>
</details>
<p>The theorem turns an infinite-dimensional optimisation into an $n$-dimensional one. With $\hat f=\sum_j\alpha_jK_{x_j}$ we have $\hat f(x_i)=(\mathbf K\alpha)_i$ and $\|\hat f\|^2=\alpha^\top\mathbf K\alpha$. The SVM's $w=\sum_i\alpha_iy_i\varphi(x_i)$ is a special case, with hinge loss and $\Omega(t)=\lambda t^2/2$. Two variants matter in practice. If an unpenalised parametric part is added, such as an intercept or the linear functions in a <a href="#/t/splines">smoothing spline</a>, the solution becomes $\sum_i\alpha_iK_{x_i}$ plus an element of that finite-dimensional space (the semiparametric representer theorem). If $\Omega$ is merely non-decreasing, <em>some</em> minimiser has the representer form.</p>`,
      },
      {
        title: 'Kernel ridge regression',
        html: R`
<p>Squared loss and $\Omega(t)=\lambda t^2$ give <strong>kernel ridge regression</strong> (KRR):
$$\hat f=\argmin_{f\in\mathcal H_K}\ \sum_{i=1}^n\big(y_i-f(x_i)\big)^2+\lambda\|f\|^2_{\mathcal H_K}.$$</p>
<div class="thm" data-kind="Proposition" data-name="KRR in closed form">
<p>The solution is $\hat f(x)=\sum_i\hat\alpha_iK(x,x_i)=k(x)^\top(\mathbf K+\lambda I)^{-1}y$, where $k(x)=(K(x,x_1),\dots,K(x,x_n))^\top$ and $\hat\alpha=(\mathbf K+\lambda I)^{-1}y$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the representer theorem, minimise $J(\alpha)=\|y-\mathbf K\alpha\|^2+\lambda\alpha^\top\mathbf K\alpha$, a convex quadratic. Its gradient is $2\mathbf K\big[(\mathbf K+\lambda I)\alpha-y\big]$, which vanishes at $\hat\alpha=(\mathbf K+\lambda I)^{-1}y$. The inverse exists for $\lambda \gt 0$ because $\mathbf K\succeq0$. If $\mathbf K$ is singular other minimisers exist, but they differ from $\hat\alpha$ by elements of the null space of $\mathbf K$. Such $\alpha$ define the zero function, since $\|\sum_i\alpha_iK_{x_i}\|^2=\alpha^\top\mathbf K\alpha=0$, so the fitted function is unique.</p>
</details>
<p><strong>The primal view.</strong> With an explicit finite feature matrix $\Phi$ (rows $\varphi(x_i)^\top$), ordinary <a href="#/t/ridge-regression">ridge regression</a> gives $\hat w=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top y$. The "push-through" identity $(\Phi^\top\Phi+\lambda I)\Phi^\top=\Phi^\top(\Phi\Phi^\top+\lambda I)$ gives $\hat w=\Phi^\top(\Phi\Phi^\top+\lambda I)^{-1}y=\Phi^\top\hat\alpha$, and so $\hat f(x)=\varphi(x)^\top\hat w=k(x)^\top(\mathbf K+\lambda I)^{-1}y$. The primal costs $O(D^3)$ in the feature dimension $D$, the dual $O(n^3)$ in the sample size. KRR is the dual, valid even when $D=\infty$.</p>
<p><strong>Shrinkage and degrees of freedom.</strong> The fitted values are $\hat y=\mathbf S_\lambda y$ with smoother matrix $\mathbf S_\lambda=\mathbf K(\mathbf K+\lambda I)^{-1}$. Diagonalising $\mathbf K=\sum_k\mu_ku_ku_k^\top$,
$$\hat y=\sum_k\frac{\mu_k}{\mu_k+\lambda}\,u_k\,(u_k^\top y),\qquad \mathrm{df}(\lambda)=\tr\mathbf S_\lambda=\sum_k\frac{\mu_k}{\mu_k+\lambda}.$$
KRR keeps the components of $y$ along the leading eigenvectors of the Gram matrix (the smooth directions) and shrinks the rest, exactly as ridge regression shrinks along principal components. As $\lambda\to0$ (with $\mathbf K$ invertible) the fit interpolates, with $\mathrm{df}=n$. As $\lambda\to\infty$ it shrinks to zero.</p>
<div class="ex" data-name="Polynomial kernel = polynomial ridge regression">
<p>With $K(x,x')=(1+xx')^d$ in one dimension, $\mathcal H_K$ is the polynomials of degree at most $d$, so $\mathbf K$ has rank at most $d+1$, and as $\lambda\to0$, $\mathrm{df}\to d+1$. That is the lab's polynomial setting, where df stays near $d+1$ until $\lambda$ is large. High degrees with tiny $\lambda$ reproduce the wild edge behaviour of high-order polynomial regression.</p>
</div>
<p><strong>Scaling up.</strong> Solving with $\mathbf K+\lambda I$ costs $O(n^3)$ (Cholesky) and storing $\mathbf K$ costs $O(n^2)$. The Nyström method approximates $\mathbf K$ by a rank-$m$ matrix built from $m$ landmark points. <em>Random Fourier features</em> (Rahimi &amp; Recht, 2007) use Bochner's theorem. If $\omega$ is drawn from the normalised spectral measure and $b\sim\mathrm{Unif}[0,2\pi]$, then $\E\big[2\cos(\omega^\top x+b)\cos(\omega^\top x'+b)\big]=\E\cos\big(\omega^\top(x-x')\big)=k(x-x')/k(0)$. So $D$ such random features give an explicit finite feature map whose inner products approximate the kernel, and ordinary ridge regression on them costs $O(nD^2)$.</p>`,
        lab: 'kernel-methods-krr',
      },
      {
        title: 'Choosing kernels, and where kernels lead',
        html: R`
<p>The kernel is the model. It fixes the hypothesis space and, through the RKHS norm, the notion of simplicity. Kernels can be composed to encode structure. Sums of kernels on different inputs give additive models, products give interactions, a periodic kernel times a squared-exponential gives locally periodic functions, and kernels on strings, graphs and sets bring the machinery to non-vector data. Hyperparameters (bandwidth $\ell$, degree, $\lambda$) are chosen by <a href="#/t/cross-validation">cross-validation</a>, which for KRR has a closed-form leave-one-out shortcut, $y_i-\hat f^{(-i)}(x_i)=(y_i-\hat y_i)/(1-[\mathbf S_\lambda]_{ii})$. They can also be chosen by the marginal likelihood of the equivalent <a href="#/t/gaussian-processes">Gaussian process</a>.</p>
<div class="warn" data-name="Local kernels in high dimension">
<p>The Gaussian kernel with bandwidth $\ell$ effectively averages over neighbourhoods of radius about $\ell$. In $d$ dimensions such neighbourhoods contain a vanishing fraction of the data unless $\ell$ is large, at which point the kernel is nearly constant locally. Kernel methods are not immune to the curse of dimensionality. They do well when the function is smooth in the kernel's sense or when the data lie near a low-dimensional set.</p>
</div>
<div class="warn" data-name="Indefinite 'kernels'">
<p>A similarity matrix that is not p.s.d. has no feature space behind it. The SVM dual becomes non-convex, and KRR's $\mathbf K+\lambda I$ can be singular. Clipping negative eigenvalues produces a valid kernel matrix but a different model, and it is not defined out of sample without care.</p>
</div>
<div class="warn" data-name="Ill-conditioning">
<p>Smooth kernels produce Gram matrices with eigenvalues decaying very fast (exponentially for the Gaussian), so $\mathbf K$ is numerically singular even when it is p.d. in theory. Never invert $\mathbf K$ directly. Solve with $\mathbf K+\lambda I$ via Cholesky, adding a small "jitter" when $\lambda$ is tiny.</p>
</div>
<p><strong>Connections.</strong> Smoothing <a href="#/t/splines">splines</a> are KRR in a Sobolev RKHS with an unpenalised linear part (Wahba, 1990). The <a href="#/t/gaussian-processes">GP</a> posterior mean is KRR with $\lambda=\sigma_n^2$. Kernel PCA is PCA on the centred Gram matrix. The maximum mean discrepancy compares distributions through mean embeddings $\E K_X$ in an RKHS. The <a href="#/t/kernel-regression">Nadaraya–Watson estimator</a> also uses a "kernel", but a smoothing kernel (a density), not necessarily a p.d. one. It is a local average, not an RKHS method, though both are linear smoothers $\hat y=\mathbf Sy$.</p>`,
      },
    ],
    quiz: [
      { q: R`Show directly that $K(x,x')=(x^\top x'+1)^2$ on $\R^2$ is an inner product of explicit features. How many features are there?`,
        a: R`Expanding, $(x_1x_1'+x_2x_2'+1)^2=x_1^2x_1'^2+x_2^2x_2'^2+2x_1x_2x_1'x_2'+2x_1x_1'+2x_2x_2'+1$, which is $\langle\varphi(x),\varphi(x')\rangle$ for $\varphi(x)=(x_1^2,x_2^2,\sqrt2x_1x_2,\sqrt2x_1,\sqrt2x_2,1)$: six features, $\binom{2+2}{2}=6$.` },
      { q: R`Is $K(x,x')=\min(x,x')$ on $(0,\infty)$ a valid kernel? Give a one-line argument.`,
        a: R`Yes. $\min(x,x')=\int_0^\infty\ind(t\le x)\ind(t\le x')\d t$ is an inner product in $L^2(0,\infty)$ of the features $\varphi(x)=\ind_{[0,x]}$. (It is the covariance of Brownian motion.)` },
      { q: R`In the proof of the representer theorem, where exactly is the reproducing property used, and what breaks without it?`,
        a: R`It shows that the loss depends only on the projection: $f(x_i)=\langle f,K_{x_i}\rangle=f_V(x_i)$ because $f_\perp\perp K_{x_i}$. In a general Hilbert space of functions where evaluation is not an inner product with a fixed element (e.g. $L^2$, where point evaluation is not even defined), $f_\perp$ could change the values $f(x_i)$, and restricting to a finite span would not be justified.` },
      { q: R`For KRR with a strictly p.d. Gram matrix, what are $\mathrm{df}(\lambda)$ in the limits $\lambda\to0$ and $\lambda\to\infty$? What do the fitted values do?`,
        a: R`$\mathrm{df}(\lambda)=\sum_k\mu_k/(\mu_k+\lambda)\to n$ as $\lambda\to0$: the fit interpolates the data ($\hat y\to y$). As $\lambda\to\infty$, $\mathrm{df}\to0$ and $\hat y\to0$, because the penalty dominates and $\hat f$ shrinks to the zero function (there is no unpenalised intercept unless you add one).` },
      { q: R`Why is $e^{-|x-x'|^3}$ not a valid kernel on $\R$, even though it is symmetric, bounded and peaked at $x=x'$?`,
        a: R`If it were p.d. and continuous, Bochner's theorem would make $k(\tau)=e^{-|\tau|^3}$ the characteristic function of some random variable $X$. But $k''(0)=0$, which would give $\E X^2=-k''(0)=0$, so $X\equiv0$ and $k\equiv1$, a contradiction. Concretely, some Gram matrices have negative eigenvalues, as the lab shows.` },
    ],
  });

  /* ===================================================================== */
  Atlas.topic({
    id: 'gaussian-processes',
    domain: 'supervised',
    order: 140,
    title: 'Gaussian Process Regression',
    subtitle: 'A prior over functions, exact Bayesian updating, and the marginal likelihood',
    level: 3,
    sources: { other: 'Rasmussen & Williams (2006) Ch. 2, 4, 5; ESL §5.8; Kanagawa et al. (2018) on GPs and kernel methods' },
    prereqs: ['kernel-methods', 'multivariate', 'bayesian-inference'],
    related: ['splines', 'ridge-regression', 'kernel-regression', 'model-selection'],
    tags: ['Bayesian nonparametrics', 'covariance function', 'posterior predictive', 'marginal likelihood', 'kriging'],
    summary: R`A Gaussian process is a probability distribution over functions under which the values at any finite set of inputs are jointly Gaussian, with covariances given by a positive-definite kernel. Put a GP prior on the regression function, observe it with Gaussian noise, and the posterior is again a GP, computed exactly by conditioning a multivariate normal. The posterior mean is $k_*^\top(K+\sigma_n^2I)^{-1}y$, identical to kernel ridge regression with $\lambda=\sigma_n^2$, and the posterior variance quantifies how far each prediction is from the data. Kernel hyperparameters are fitted by maximising the marginal likelihood, which automatically trades data fit against model complexity. The price is an $O(n^3)$ Cholesky factorisation.`,
    survey: {
      task: 'Nonparametric regression with calibrated uncertainty (also: classification, optimisation, emulation)',
      assumptions: R`Gaussian noise; a covariance function whose smoothness and length-scales suit $f$; usually stationarity`,
      output: R`Posterior mean and variance at any $x_*$; posterior sample paths; marginal likelihood`,
      strengths: 'Exact Bayesian inference, principled uncertainty, hyperparameters by evidence, works well with small $n$',
      weaknesses: R`$O(n^3)$ time and $O(n^2)$ memory; non-Gaussian likelihoods need approximation; results sensitive to the kernel`,
      complexity: R`$O(n^3)$ to factorise; then $O(n)$ per predictive mean and $O(n^2)$ per predictive variance`,
    },
    sections: [
      {
        title: 'A prior over functions',
        html: R`
<div class="def" data-name="Gaussian process">
<p>A stochastic process $\{f(x):x\in\mathcal X\}$ is a <em>Gaussian process</em> with mean function $m$ and covariance function $k$, written $f\sim\mathcal{GP}(m,k)$, if for every finite set $x_1,\dots,x_n$
$$\big(f(x_1),\dots,f(x_n)\big)\sim\N\big((m(x_i))_i,\;[k(x_i,x_j)]_{ij}\big).$$</p>
</div>
<p>The finite-dimensional distributions are consistent (marginalising a Gaussian drops rows and columns), so by Kolmogorov's extension theorem such a process exists exactly when $k$ is positive definite. That is the same condition as for a <a href="#/t/kernel-methods">kernel</a>. A GP is a "Gaussian distribution over functions" specified entirely by $m$ (usually $0$ after centring $y$) and $k$.</p>
<p><strong>The weight-space view.</strong> Bayesian linear regression is a GP. If $f(x)=\varphi(x)^\top w$ with $w\sim\N(0,\Sigma_p)$, then $f$ is Gaussian at any finite set of points, with $\Cov(f(x),f(x'))=\varphi(x)^\top\Sigma_p\varphi(x')$. Conversely, by Mercer's theorem, a GP with kernel $k$ is a linear model with infinitely many features $\sqrt{\lambda_j}e_j(x)$ and independent standard normal weights (the Karhunen–Loève expansion). GP regression is Bayesian linear regression in the kernel's feature space, computed in the dual.</p>
<p><strong>The kernel encodes the prior.</strong> For stationary $k(x,x')=\sigma_f^2\,\kappa(|x-x'|/\ell)$, $\sigma_f^2$ sets the vertical scale and $\ell$ the horizontal one: roughly the distance over which $f$ changes appreciably. The family sets smoothness:</p>
<table>
<thead><tr><th>Kernel</th><th>$k(r)$, $r=|x-x'|$</th><th>Mean-square differentiability</th></tr></thead>
<tbody>
<tr><td>Squared exponential</td><td>$\sigma_f^2\exp(-r^2/2\ell^2)$</td><td>infinitely often</td></tr>
<tr><td>Matérn $\nu=5/2$</td><td>$\sigma_f^2\big(1+\tfrac{\sqrt5r}{\ell}+\tfrac{5r^2}{3\ell^2}\big)e^{-\sqrt5r/\ell}$</td><td>twice</td></tr>
<tr><td>Matérn $\nu=3/2$</td><td>$\sigma_f^2\big(1+\tfrac{\sqrt3r}{\ell}\big)e^{-\sqrt3r/\ell}$</td><td>once</td></tr>
<tr><td>Exponential (Matérn $\tfrac12$, Ornstein–Uhlenbeck)</td><td>$\sigma_f^2e^{-r/\ell}$</td><td>not differentiable</td></tr>
<tr><td>Periodic</td><td>$\sigma_f^2\exp\big(-2\sin^2(\pi r/p)/\ell^2\big)$</td><td>infinitely often; period $p$</td></tr>
</tbody>
</table>
<p>A Matérn-$\nu$ process is $k$ times mean-square differentiable iff $\nu \gt k$. Stein (1999) argued that the squared exponential's infinite smoothness is unrealistic for many physical processes, and Matérn 3/2 or 5/2 is often the safer default.</p>`,
      },
      {
        title: 'Conditioning a Gaussian',
        html: R`
<p>All of GP regression rests on one fact about the <a href="#/t/multivariate">multivariate normal</a>.</p>
<div class="thm" data-kind="Lemma" data-name="Gaussian conditioning">
<p>Let $\begin{pmatrix}a\\ b\end{pmatrix}\sim\N\left(\begin{pmatrix}\mu_a\\ \mu_b\end{pmatrix},\begin{pmatrix}\Sigma_{aa}&\Sigma_{ab}\\ \Sigma_{ba}&\Sigma_{bb}\end{pmatrix}\right)$ with $\Sigma_{bb}$ invertible. Then
$$a\mid b\;\sim\;\N\Big(\mu_a+\Sigma_{ab}\Sigma_{bb}^{-1}(b-\mu_b),\;\;\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\Big).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $A=\Sigma_{ab}\Sigma_{bb}^{-1}$ and $z=a-Ab$. Then $(z,b)$ is jointly Gaussian (a linear map of $(a,b)$) and $\Cov(z,b)=\Sigma_{ab}-A\Sigma_{bb}=0$. Uncorrelated jointly Gaussian vectors are independent, so the conditional law of $z$ given $b$ is its marginal $\N(\mu_a-A\mu_b,\;\Sigma_{aa}-A\Sigma_{ba}-\Sigma_{ab}A^\top+A\Sigma_{bb}A^\top)$, and the covariance simplifies to $\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}$. Since $a=z+Ab$, conditionally on $b$ we get $a\mid b\sim\N(\mu_a+A(b-\mu_b),\ \Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba})$.</p>
</details>
<p>Two features of this formula drive everything that follows. The conditional mean is <em>linear</em> in the observed $b$. The conditional covariance does <em>not depend on</em> $b$ at all, only on which variables were observed, and it is never larger than the prior covariance (in the p.s.d. order), since $\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\succeq0$.</p>`,
      },
      {
        title: 'The posterior and predictive distributions',
        html: R`
<p>Model $y_i=f(x_i)+\varepsilon_i$ with $f\sim\mathcal{GP}(0,k)$ and $\varepsilon_i\iid\N(0,\sigma_n^2)$ independent of $f$. Write $\mathbf K=[k(x_i,x_j)]$, $k_*=(k(x_*,x_1),\dots,k(x_*,x_n))^\top$ and $k_{**}=k(x_*,x_*)$. The observations and the latent value at a test point are jointly Gaussian:
$$\begin{pmatrix}y\\ f(x_*)\end{pmatrix}\sim\N\left(0,\begin{pmatrix}\mathbf K+\sigma_n^2I&k_*\\ k_*^\top&k_{**}\end{pmatrix}\right).$$
Applying the conditioning lemma:</p>
<div class="thm" data-kind="Theorem" data-name="GP regression posterior">
<p>$f(x_*)\mid y\sim\N\big(\bar f_*,\ \Var f_*\big)$ with
$$\bar f_*=k_*^\top(\mathbf K+\sigma_n^2I)^{-1}y,\qquad\Var f_*=k_{**}-k_*^\top(\mathbf K+\sigma_n^2I)^{-1}k_* .$$
Jointly over test points $X_*$, the posterior is a GP with mean $\bar f(x)=k(x)^\top(\mathbf K+\sigma_n^2I)^{-1}y$ and covariance $k(x,x')-k(x)^\top(\mathbf K+\sigma_n^2I)^{-1}k(x')$. A new <em>observation</em> $y_*=f(x_*)+\varepsilon_*$ has predictive variance $\Var f_*+\sigma_n^2$.</p>
</div>
<p>Everything about the theorem is interpretable. The mean is a linear smoother, $\bar f(x)=\sum_i\alpha_ik(x,x_i)$ with $\alpha=(\mathbf K+\sigma_n^2I)^{-1}y$, a weighted sum of kernel bumps centred at the data. The variance starts at the prior $k_{**}$ and subtracts what the data explain. Near the data it collapses (down to about $\sigma_n^2$-scale), far away it returns to the prior, and it is the same for any $y$. A GP is confident where it has data, whatever those data say. This makes it useful for experimental design and Bayesian optimisation, and it is also a limitation, since a GP cannot flag data that are surprising given the model.</p>
<div class="algo" data-name="Numerically stable GP regression (Rasmussen & Williams, Alg. 2.1)">
<ol>
<li>$L=\operatorname{chol}(\mathbf K+\sigma_n^2I)$ (lower triangular), cost $O(n^3/3)$.</li>
<li>$\alpha=L^\top\backslash(L\backslash y)$ by two triangular solves, $O(n^2)$.</li>
<li>For each test point: $\bar f_*=k_*^\top\alpha$ in $O(n)$; $v=L\backslash k_*$ and $\Var f_*=k_{**}-v^\top v$ in $O(n^2)$.</li>
<li>$\log p(y\mid X)=-\tfrac12y^\top\alpha-\sum_i\log L_{ii}-\tfrac n2\log2\pi$.</li>
</ol>
</div>
<p>Never form $(\mathbf K+\sigma_n^2I)^{-1}$ explicitly. For posterior samples on a grid, factorise the posterior covariance (plus a tiny jitter) and draw $\bar f+L_*z$ with $z\sim\N(0,I)$.</p>`,
        lab: 'gaussian-processes-regression',
      },
      {
        title: 'Hyperparameters and the marginal likelihood',
        html: R`
<p>The kernel has hyperparameters $\theta$, such as $(\ell,\sigma_f,\sigma_n)$. The Bayesian answer is to integrate $f$ out. Marginally, $y\sim\N(0,\mathbf K_y)$ with $\mathbf K_y=\mathbf K_\theta+\sigma_n^2I$, since $y$ is the sum of independent Gaussians $f(X)\sim\N(0,\mathbf K_\theta)$ and $\varepsilon\sim\N(0,\sigma_n^2I)$. So
$$\log p(y\mid X,\theta)=\underbrace{-\tfrac12y^\top\mathbf K_y^{-1}y}_{\text{data fit}}\;\underbrace{-\tfrac12\log|\mathbf K_y|}_{\text{complexity penalty}}\;-\;\tfrac n2\log2\pi .$$
<strong>Type-II maximum likelihood</strong> (empirical Bayes) picks $\hat\theta=\argmax_\theta\log p(y\mid X,\theta)$.</p>
<div class="note" data-name="Automatic Occam's razor">
<p>A short length-scale makes $\mathbf K_y$ close to diagonal: the model can fit anything, so the data-fit term is good, but $|\mathbf K_y|$ is large, because the prior spreads its mass over a huge set of possible data sets. A very long length-scale concentrates the prior on nearly constant functions. $|\mathbf K_y|$ is small, but wiggly data fit badly. The evidence is the probability the model assigned to the data actually observed, and it peaks at the simplest model that explains them. No validation set is needed.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Gradient of the log marginal likelihood">
<p>With $\alpha=\mathbf K_y^{-1}y$,
$$\frac{\partial}{\partial\theta_j}\log p(y\mid X,\theta)=\frac12\tr\!\Big(\big(\alpha\alpha^\top-\mathbf K_y^{-1}\big)\frac{\partial\mathbf K_y}{\partial\theta_j}\Big).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Use $\partial(\mathbf K^{-1})=-\mathbf K^{-1}(\partial\mathbf K)\mathbf K^{-1}$ and $\partial\log|\mathbf K|=\tr(\mathbf K^{-1}\partial\mathbf K)$. Then $\partial(-\tfrac12y^\top\mathbf K_y^{-1}y)=\tfrac12y^\top\mathbf K_y^{-1}(\partial\mathbf K_y)\mathbf K_y^{-1}y=\tfrac12\tr(\alpha\alpha^\top\partial\mathbf K_y)$, and $\partial(-\tfrac12\log|\mathbf K_y|)=-\tfrac12\tr(\mathbf K_y^{-1}\partial\mathbf K_y)$.</p>
</details>
<p>After the $O(n^3)$ factorisation, each gradient component costs $O(n^2)$ given $\mathbf K_y^{-1}$, so quasi-Newton optimisation over a handful of hyperparameters is cheap. Two cautions. The surface is not concave, and it often has several local optima: typically a "smooth function plus lots of noise" explanation competes with a "wiggly function plus little noise" one. The second lab maps exactly this landscape. Use several restarts, or put priors on $\theta$ and sample them with <a href="#/t/mcmc-foundations">MCMC</a>, which also propagates hyperparameter uncertainty into predictions. A useful check: if $\sigma_f$ and $\sigma_n$ are both free (so the overall scale of $\mathbf K_y$ is free), then at the optimum $y^\top\mathbf K_y^{-1}y=n$ exactly, so the data-fit term equals $-n/2$. You can see this in the first lab after "Maximise evidence".</p>`,
        lab: 'gaussian-processes-evidence',
      },
      {
        title: 'GPs and kernel ridge regression',
        html: R`
<p>Compare the GP posterior mean with <a href="#/t/kernel-methods">kernel ridge regression</a>: $k_*^\top(\mathbf K+\sigma_n^2I)^{-1}y$ versus $k_*^\top(\mathbf K+\lambda I)^{-1}y$. They are the same with $\lambda=\sigma_n^2$. This is no coincidence:</p>
<div class="thm" data-kind="Proposition" data-name="Posterior mean as a regularised estimator">
<p>The GP posterior mean is the minimiser over the RKHS $\mathcal H_k$ of
$$\frac{1}{2\sigma_n^2}\sum_{i=1}^n\big(y_i-f(x_i)\big)^2+\frac12\|f\|^2_{\mathcal H_k}.$$
It is also the posterior mode (MAP) of $f(X)$ under the GP prior.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Multiply by $2\sigma_n^2$: this is KRR with $\lambda=\sigma_n^2$, whose solution is $k(x)^\top(\mathbf K+\sigma_n^2I)^{-1}y$. For the MAP statement: with $\mathbf f=f(X)\sim\N(0,\mathbf K)$, the negative log posterior is $\frac1{2\sigma_n^2}\|y-\mathbf f\|^2+\frac12\mathbf f^\top\mathbf K^{-1}\mathbf f$ (for invertible $\mathbf K$). Writing $\mathbf f=\mathbf K\alpha$ gives $\frac12\mathbf f^\top\mathbf K^{-1}\mathbf f=\frac12\alpha^\top\mathbf K\alpha=\frac12\|f\|_{\mathcal H}^2$, so it is the same objective. For a Gaussian posterior, mode and mean coincide.</p>
</details>
<p>So the two frameworks share their point predictions. The GP adds a calibrated (under the model) variance and the marginal likelihood for tuning, while KRR offers a frequentist reading. The first lab's overlay shows the two curves coinciding. A further bridge links the GP variance to worst-case error in the RKHS.</p>
<div class="thm" data-kind="Proposition" data-name="Posterior SD bounds the worst-case interpolation error">
<p>In the noise-free case ($\sigma_n=0$, $\mathbf K$ invertible), let $\bar f$ be the posterior mean computed from the values $f(x_1),\dots,f(x_n)$ of some function $f\in\mathcal H_k$. Then for every $x$,
$$|f(x)-\bar f(x)|\le\sigma(x)\,\|f\|_{\mathcal H_k},\qquad \sigma^2(x)=k(x,x)-k(x)^\top\mathbf K^{-1}k(x),$$
and $\sigma(x)$ is the largest such error over the unit ball of $\mathcal H_k$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>With $c=\mathbf K^{-1}k(x)$, $\bar f(x)=\sum_ic_if(x_i)$. By the reproducing property, $f(x)-\bar f(x)=\langle f,\,k_x-\sum_ic_ik_{x_i}\rangle$. Cauchy–Schwarz gives the bound with $\|k_x-\sum_ic_ik_{x_i}\|^2=k(x,x)-2c^\top k(x)+c^\top\mathbf Kc=k(x,x)-k(x)^\top\mathbf K^{-1}k(x)$. Equality holds for $f$ proportional to $k_x-\sum_ic_ik_{x_i}$.</p>
</details>
<div class="warn" data-name="Sample paths are rougher than the RKHS">
<p>The posterior mean lies in $\mathcal H_k$, but when $\mathcal H_k$ is infinite-dimensional, GP sample paths almost surely do <em>not</em> (Driscoll's zero–one law; Lukić &amp; Beder, 2001). For example, a Matérn-$\nu$ process in $d$ dimensions has paths roughly $d/2$ orders of smoothness below its RKHS. The "prior over functions" and the "regulariser" are related, but they are not the same object.</p>
</div>`,
      },
      {
        title: 'Scaling, pitfalls, and extensions',
        html: R`
<div class="warn" data-name="Cubic cost">
<p>Exact inference needs $O(n^3)$ time and $O(n^2)$ memory: comfortable to $n\approx10^4$ on a laptop, painful beyond. Remedies: <em>inducing-point</em> approximations with $m\ll n$ pseudo-inputs, at $O(nm^2)$ (FITC, Snelson &amp; Ghahramani, 2006; variational, Titsias, 2009; stochastic variational, Hensman et al., 2013); structure-exploiting algebra (Toeplitz or Kronecker for gridded inputs); iterative solvers with GPU matrix-vector products (Gardner et al., 2018); random features (see <a href="#/t/kernel-methods">kernel methods</a>).</p>
</div>
<div class="warn" data-name="Numerical conditioning">
<p>With a smooth kernel, small noise and closely spaced inputs, $\mathbf K+\sigma_n^2I$ is nearly singular. Add a jitter (about $10^{-8}\sigma_f^2$) to the diagonal, keep $\sigma_n$ bounded away from zero during optimisation, and always work through the Cholesky factor.</p>
</div>
<div class="warn" data-name="The model's uncertainty is only as good as the model">
<p>Posterior bands are calibrated if the prior is right. A stationary kernel assumes the same length-scale everywhere. Data with a jump or with varying wiggliness get a compromise $\ell$: too smooth in one place and too rough in another, with bands of the wrong width. Far from the data the mean reverts to the prior mean (usually $0$), so centre or detrend $y$ first. Heavy-tailed noise or outliers violate the Gaussian likelihood.</p>
</div>
<p><strong>Extensions.</strong> <em>GP classification</em> passes $f$ through a link (logistic or probit). The posterior is then no longer Gaussian and needs the Laplace approximation, expectation propagation, variational inference or MCMC. <em>Automatic relevance determination</em> uses one length-scale per input, $k=\sigma_f^2\exp(-\sum_j(x_j-x'_j)^2/2\ell_j^2)$; a large fitted $\ell_j$ marks an irrelevant input. In geostatistics GP regression is <em>kriging</em> (Matheron, 1963). In Bayesian optimisation the posterior mean and variance drive the choice of where to evaluate an expensive function next. In computer experiments GPs serve as emulators of slow simulators. Finally, wide neural networks with random weights converge to GPs (Neal, 1996), a bridge to the <a href="#/t/deep-learning-theory">theory of deep learning</a>.</p>`,
      },
    ],
    quiz: [
      { q: R`You add a new observation at $x_{n+1}$. Without knowing its $y$ value, which of these change: the posterior mean, the posterior variance, the marginal likelihood?`,
        a: R`Only the posterior variance can be computed without $y_{n+1}$. It depends only on the input locations, and it decreases (weakly) everywhere, most near $x_{n+1}$. The posterior mean and the marginal likelihood both depend on the observed value.` },
      { q: R`For a single observation $y_1$ at $x_1$ and a test point $x_*$ with $k(x_*,x_1)=\rho\sigma_f^2$, $k(x,x)=\sigma_f^2$, write the posterior mean and variance at $x_*$.`,
        a: R`$\bar f_*=\frac{\rho\sigma_f^2}{\sigma_f^2+\sigma_n^2}y_1$ and $\Var f_*=\sigma_f^2-\frac{\rho^2\sigma_f^4}{\sigma_f^2+\sigma_n^2}$. With no noise and $\rho=1$ (i.e. $x_*=x_1$), the mean is $y_1$ and the variance is $0$. As $\rho\to0$ the posterior reverts to the prior: mean $0$, variance $\sigma_f^2$.` },
      { q: R`Why does maximising the marginal likelihood not simply choose the most flexible model (tiny length-scale, zero noise)?`,
        a: R`The log evidence contains $-\frac12\log|\mathbf K_y|$. A flexible prior spreads its probability over many possible data sets, so its density at any particular $y$ (including the observed one) is low. Tiny length-scales make $\mathbf K_y$ close to $\sigma_f^2I$, with a large determinant penalty relative to a smoother model that explains the data equally well. The evidence balances fit against this automatic complexity penalty.` },
      { q: R`Kernel ridge regression and GP regression give the same mean. Name two things the GP gives you that plain KRR does not, and one thing it costs.`,
        a: R`(1) A pointwise predictive variance, and samples of whole functions, for uncertainty quantification. (2) The marginal likelihood, a differentiable criterion for fitting hyperparameters without cross-validation. The cost: these are only as trustworthy as the Gaussian prior and likelihood. Computing variances also needs $O(n^2)$ per test point, versus $O(n)$ for the mean.` },
      { q: R`A GP with the squared-exponential kernel and fitted $\ell=0.3$ is used to predict at $x_*=10$, far from all training inputs in $[0,2]$. What does it predict, and is the reported uncertainty sensible?`,
        a: R`At distance $\gg\ell$ every $k(x_*,x_i)\approx0$, so the mean reverts to the prior mean (0) and the variance to $\sigma_f^2$. The band is "honest" under the prior, but the prediction ignores any trend in the data. A stationary zero-mean GP cannot extrapolate a trend. Use a mean function or a linear-plus-stationary kernel if extrapolation matters.` },
    ],
  });
})();
