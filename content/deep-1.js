/* Deep Learning I: perceptron & MLP, universal approximation, backpropagation, optimizers,
 * initialization & normalization, regularization, and the theory of deep learning. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'perceptron-mlp',
    domain: 'deep',
    order: 10,
    title: 'From the Perceptron to the Multilayer Network',
    subtitle: 'Linear threshold units, their limits, and compositions of nonlinear layers',
    level: 2,
    sources: { aos: '22.11', other: 'Rosenblatt (1958); Novikoff (1962); Minsky & Papert (1969); Rumelhart, Hinton & Williams (1986); Goodfellow, Bengio & Courville (2016) Ch. 6; Prince, Understanding Deep Learning (2023) Ch. 3–5' },
    prereqs: ['logistic-regression', 'learning-framework'],
    related: ['svm', 'glm-framework', 'mle', 'universal-approximation', 'backpropagation'],
    tags: ['perceptron', 'MLP', 'activation function', 'cross-entropy', 'linear separability'],
    summary: R`A neural network is a composition of affine maps and fixed elementwise nonlinearities, $f = A_L\circ\phi\circ A_{L-1}\circ\cdots\circ\phi\circ A_1$, fitted by minimizing a loss. Its ancestor, Rosenblatt's perceptron, is a single linear threshold unit with a beautifully simple learning rule whose number of mistakes on separable data is bounded by $(R/\gamma)^2$ regardless of dimension. A single unit cannot represent XOR; one hidden layer of nonlinear units can. Statistically, the familiar losses are negative log-likelihoods — squared error for a Gaussian, cross-entropy for a categorical — so a network with no hidden layer is exactly a GLM, and a deep network is a GLM whose features are learned.`,
    survey: {
      task: 'Classification and regression with learned nonlinear features',
      assumptions: R`IID data; a differentiable loss (a negative log-likelihood); enough data for the parameter count`,
      output: R`A parametric predictor $f_\theta$: class probabilities (softmax) or a conditional mean`,
      strengths: 'Learns its own features; one recipe for regression, classification, structured outputs; scales with data and compute.',
      weaknesses: 'Non-convex, non-identifiable, hyperparameter-heavy; needs lots of data; poorly calibrated without care.',
      complexity: R`Forward pass $O\big(\sum_\ell n_\ell n_{\ell-1}\big)$ per example; training = many passes`,
    },
    sections: [
      {
        title: 'The linear threshold unit and the perceptron rule',
        html: R`
<p>Take inputs $x\in\R^d$ and labels $y\in\{-1,+1\}$. A <strong>linear threshold unit</strong> predicts $\hat y = \sgn(w^\top x + b)$: it splits $\R^d$ by the hyperplane $\{x : w^\top x + b = 0\}$, with $w$ the normal vector pointing into the positive half-space. Absorb the bias by augmenting $\tilde x = (x, 1)$ and $\tilde w = (w, b)$, so the classifier is $\sgn(\tilde w^\top \tilde x)$ and the hyperplane passes through the origin of $\R^{d+1}$.</p>
<p>Rosenblatt (1958) proposed learning $\tilde w$ by cycling through the data and correcting mistakes.</p>
<div class="algo" data-name="Perceptron">
<ol>
<li>Initialize $\tilde w_0 = 0$, $k = 0$.</li>
<li>Repeat passes over the data. For each $(\tilde x_i, y_i)$: if $y_i\,\tilde w_k^\top \tilde x_i \le 0$ (a mistake, or on the boundary), set $\tilde w_{k+1} = \tilde w_k + y_i\tilde x_i$ and $k \leftarrow k+1$.</li>
<li>Stop after a full pass with no mistakes.</li>
</ol>
</div>
<p>The update has a clean reading: after correcting on $(\tilde x_i,y_i)$, the margin of that point increases by $y_i^2\|\tilde x_i\|^2 = \|\tilde x_i\|^2 \gt 0$. It is also stochastic (sub)gradient descent with step size 1 on the <em>perceptron loss</em> $\ell(\tilde w; \tilde x, y) = \max(0, -y\,\tilde w^\top\tilde x)$, whose subgradient is $-y\tilde x$ on mistakes and $0$ otherwise. Because this loss is positively homogeneous, the step size is irrelevant: scaling $\tilde w$ never changes predictions.</p>
<div class="note">
<p>The perceptron is the zero-temperature limit of logistic regression's stochastic gradient: the logistic loss $\log(1+e^{-y\tilde w^\top \tilde x})$ has gradient $-y\tilde x\,\sigma(-y\tilde w^\top\tilde x)$, a soft version of "update by $y\tilde x$ only if wrong". The perceptron replaces the sigmoid weight by a hard indicator.</p>
</div>`,
      },
      {
        title: "Novikoff's theorem: mistakes are bounded by the margin",
        html: R`
<div class="def" data-name="Linear separability and margin">
<p>Data $\{(\tilde x_i, y_i)\}_{i=1}^n$ are <em>linearly separable with margin $\gamma \gt 0$</em> if some unit vector $u$ (with $\|u\|=1$) satisfies $y_i\, u^\top \tilde x_i \ge \gamma$ for all $i$. Write $R = \max_i \|\tilde x_i\|$.</p>
</div>
<div class="thm" data-name="Perceptron convergence (Novikoff, 1962)">
<p>If the data are separable with margin $\gamma$, the perceptron makes at most $(R/\gamma)^2$ updates, in any order of presentation and for any number of passes. In particular it terminates with a separating hyperplane.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Track two quantities after the $k$th update, which happened on some $(\tilde x_i,y_i)$ with $y_i\tilde w_{k-1}^\top \tilde x_i \le 0$.</p>
<p><em>Alignment grows linearly.</em> $u^\top \tilde w_k = u^\top \tilde w_{k-1} + y_i u^\top \tilde x_i \ge u^\top \tilde w_{k-1} + \gamma$. Starting from $\tilde w_0=0$, induction gives $u^\top\tilde w_k \ge k\gamma$.</p>
<p><em>Norm grows at most like $\sqrt k$.</em> $\|\tilde w_k\|^2 = \|\tilde w_{k-1}\|^2 + 2y_i\tilde w_{k-1}^\top \tilde x_i + \|\tilde x_i\|^2 \le \|\tilde w_{k-1}\|^2 + R^2$, because the middle term is $\le 0$ on a mistake. So $\|\tilde w_k\|^2 \le kR^2$.</p>
<p>Combine with Cauchy–Schwarz: $k\gamma \le u^\top\tilde w_k \le \|u\|\,\|\tilde w_k\| \le \sqrt k\,R$. Hence $\sqrt k \le R/\gamma$, i.e. $k \le (R/\gamma)^2$. Since each pass without termination contains at least one update, the algorithm stops after at most $(R/\gamma)^2 + 1$ passes.</p>
</details>
<p>Three features of this bound deserve emphasis. It does not depend on $n$ or on the dimension $d$ — only on the geometry ratio $R/\gamma$, which is the same quantity that controls the VC-type generalization bounds for large-margin classifiers (see <a href="#/t/svm">support vector machines</a>). It is worst-case over the presentation order, so it also bounds the <em>online</em> mistake count against an adversary. And it gives no guarantee about <em>which</em> separator is found: the perceptron stops at the first one, typically with a margin far below the maximum. The SVM asks for the max-margin separator instead; remarkably, gradient descent on the logistic loss finds it in the limit (<a href="#/t/deep-learning-theory">implicit bias</a>).</p>
<div class="ex" data-name="The bound can be attained">
<p>Work without a bias term and take the standard basis vectors $e_1,\dots,e_d\in\R^d$, all labelled $+1$. The unit vector $u = (1,\dots,1)/\sqrt d$ separates them with margin $\gamma = 1/\sqrt d$, and $R = 1$, so the bound is $(R/\gamma)^2 = d$. Starting from $w_0 = 0$, the perceptron misclassifies every $e_i$ on the first pass, because each is orthogonal to all previous updates: exactly $d$ mistakes. No mistake-driven rule of this kind can beat $(R/\gamma)^2$ in the worst case.</p>
</div>
<div class="warn" data-name="Non-separable data">
<p>Without separability the perceptron never stops and its weight vector does not converge; it wanders in a bounded region (the <em>perceptron cycling theorem</em>, Block & Levin, 1970). Practical fixes average the iterates (the <em>averaged</em> or <em>voted</em> perceptron of Freund & Schapire, 1999), which inherits margin-type guarantees, or switch to a smooth loss such as the logistic loss.</p>
</div>
<p>In the lab, compare the count of updates with the bound as you shrink the enforced margin, then switch to overlapping or XOR data.</p>`,
        lab: 'perceptron-mlp-perceptron',
      },
      {
        title: 'What one unit cannot do: XOR and the case for hidden layers',
        html: R`
<div class="thm" data-kind="Proposition" data-name="XOR is not linearly separable">
<p>There are no $w\in\R^2$, $b\in\R$ such that $\sgn(w^\top x + b)$ equals $+1$ on $\{(0,1),(1,0)\}$ and $-1$ on $\{(0,0),(1,1)\}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Suppose such $w,b$ existed. Then $w_2 + b \gt 0$ and $w_1 + b\gt 0$, while $b \le 0$ and $w_1+w_2+b \le 0$ (allowing ties on the boundary to count as $-1$ only strengthens the argument). Adding the first two, $w_1+w_2+2b \gt 0$; adding the last two, $w_1+w_2+2b \le 0$. Contradiction. Geometrically: the midpoint $(\tfrac12,\tfrac12)$ is the average of both positive points and of both negative points, and a half-space is convex, so it cannot contain one pair while excluding the other.</p>
</details>
<p>Minsky and Papert (1969) made this kind of limitation famous. The general fact is Cover's (1965) <em>function-counting theorem</em>: of the $2^n$ labelings of $n$ points in general position in $\R^d$, exactly $C(n,d) = 2\sum_{k=0}^{d-1}\binom{n-1}{k}$ are realizable by homogeneous linear separators. When $n \le d$ all of them are; once $n \gt 2d$ the fraction falls rapidly to zero. A fixed linear unit runs out of capacity quickly.</p>
<p>The cure is a nonlinear feature map. For XOR, the single feature $x_1x_2$ suffices; but the point of neural networks is to <em>learn</em> the features. One hidden layer of two ReLU units, $\phi(t) = \max(t,0)$, does it exactly:</p>
$$h_1 = \phi(x_1 + x_2),\qquad h_2 = \phi(x_1 + x_2 - 1),\qquad f(x) = h_1 - 2h_2.$$
<p>On the four corners, $f(0,0) = 0$, $f(1,0)=f(0,1)=1$, $f(1,1) = 2 - 2 = 0$, so thresholding $f$ at $\tfrac12$ computes XOR. The hidden layer has bent the input space so that the classes become linearly separable in $(h_1,h_2)$ coordinates.</p>
<div class="note">
<p>This is the whole idea in miniature: hidden layers compute a representation in which the final linear model works. Kernel methods fix the representation in advance (<a href="#/t/kernel-methods">kernels</a>); neural networks fit it jointly with the linear readout.</p>
</div>`,
      },
      {
        title: 'The multilayer perceptron as a composition',
        html: R`
<div class="def" data-name="Multilayer perceptron (fully connected network)">
<p>Fix widths $n_0 = d, n_1,\dots,n_L$ and an activation $\phi:\R\to\R$ applied elementwise. With parameters $\theta = (W_\ell, b_\ell)_{\ell=1}^L$, $W_\ell\in\R^{n_\ell\times n_{\ell-1}}$, define
$$h_0 = x,\qquad z_\ell = W_\ell h_{\ell-1} + b_\ell,\qquad h_\ell = \phi(z_\ell)\ (\ell \lt L),\qquad f_\theta(x) = z_L .$$
$L-1$ is the number of hidden layers; $z_\ell$ are pre-activations, $h_\ell$ activations. The parameter count is $\sum_\ell n_\ell(n_{\ell-1}+1)$.</p>
</div>
<table>
<thead><tr><th>Activation</th><th>$\phi(t)$</th><th>$\phi'(t)$</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Logistic sigmoid</td><td>$1/(1+e^{-t})$</td><td>$\phi(1-\phi)\le \tfrac14$</td><td>Saturates; gradients vanish through depth</td></tr>
<tr><td>tanh</td><td>$\tanh t$</td><td>$1-\phi^2 \le 1$</td><td>Zero-centred; $\tanh t = 2\sigma(2t)-1$</td></tr>
<tr><td>ReLU</td><td>$\max(t,0)$</td><td>$\ind(t\gt 0)$</td><td>Piecewise linear; "dead" units if $z\lt 0$ always</td></tr>
<tr><td>Leaky ReLU</td><td>$\max(t,\alpha t)$</td><td>$1$ or $\alpha$</td><td>Never fully dead</td></tr>
<tr><td>GELU</td><td>$t\,\Phi(t)$</td><td>$\Phi(t)+t\varphi(t)$</td><td>Smooth ReLU; standard in transformers</td></tr>
</tbody>
</table>
<div class="thm" data-kind="Proposition" data-name="Without a nonlinearity, depth is useless">
<p>If $\phi$ is affine, every MLP computes an affine function of $x$; with $\phi(t)=t$ the network $W_L\cdots W_1 x + (\text{const})$ has rank at most $\min_\ell n_\ell$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Compositions of affine maps are affine: $W_2(W_1x+b_1)+b_2 = (W_2W_1)x + (W_2b_1+b_2)$; induct on $L$. The product $W_L\cdots W_1$ factors through each $\R^{n_\ell}$, which bounds its rank.</p>
</details>
<p>(A deep <em>linear</em> network still has a non-convex loss surface and nontrivial training dynamics — it is a useful theoretical model — but its function class is just low-rank linear maps.)</p>
<p>With ReLU, the network is a <strong>continuous piecewise-linear</strong> function: each pattern of active units fixes a linear map, and the input space is tiled by convex polytopes ("linear regions"). The number of regions can grow exponentially with depth (Montúfar et al., 2014), one route to the depth-separation results in <a href="#/t/universal-approximation">universal approximation</a>.</p>
<div class="warn" data-name="Non-identifiability">
<p>The map $\theta\mapsto f_\theta$ is massively many-to-one. Permuting the hidden units of a layer (with the matching rows/columns of adjacent weight matrices) gives $\prod_\ell n_\ell!$ equivalent parameter vectors. For ReLU, rescaling $(W_\ell,b_\ell)\to(cW_\ell, cb_\ell)$ and $W_{\ell+1}\to W_{\ell+1}/c$ for $c\gt 0$ leaves $f$ unchanged, since $\phi(ct) = c\phi(t)$. So individual weights have no meaning, the Fisher information is singular, and classical asymptotics for $\hat\theta$ (<a href="#/t/mle">MLE theory</a>) do not apply; inference must target functionals of $f_\theta$.</p>
</div>`,
      },
      {
        title: 'Losses are negative log-likelihoods',
        html: R`
<p>Fitting a network means minimizing an empirical risk $\frac1n\sum_i \ell(f_\theta(x_i), y_i)$. The standard choices are not arbitrary: each is the negative log-likelihood of a conditional model whose parameter is the network output (see <a href="#/t/mle">maximum likelihood</a>).</p>
<div class="cols">
<div><h4>Regression</h4><p>Model $Y\mid x \sim \N(f_\theta(x), \sigma^2)$. Then $-\log p(y\mid x) = \frac{1}{2\sigma^2}(y - f_\theta(x))^2 + \tfrac12\log(2\pi\sigma^2)$, so maximum likelihood is least squares, for any fixed $\sigma$.</p></div>
<div><h4>Binary</h4><p>Model $Y\mid x\sim \text{Bernoulli}(\sigma(z))$, $z = f_\theta(x)$. Then $-\log p(y\mid x) = \log(1+e^{z}) - yz$: the binary cross-entropy "with logits".</p></div>
<div><h4>Multiclass</h4><p>Model $Y\mid x\sim\text{Categorical}(\operatorname{softmax}(z))$, $p_k = e^{z_k}/\sum_j e^{z_j}$. Then $-\log p(y\mid x) = -z_y + \log\sum_j e^{z_j}$: softmax cross-entropy.</p></div>
</div>
<p>So a network with no hidden layer and a sigmoid output is literally <a href="#/t/logistic-regression">logistic regression</a>, and with a softmax output it is <a href="#/t/multinomial-ordinal">multinomial logistic regression</a>. In <a href="#/t/glm-framework">GLM</a> language, the output layer is the inverse canonical link and the hidden layers construct the design matrix.</p>
<div class="thm" data-kind="Proposition" data-name="The canonical-link gradient">
<p>For softmax cross-entropy $\ell(z,y) = -z_y + \log\sum_j e^{z_j}$, $\nabla_z \ell = p - e_y$, where $e_y$ is the one-hot vector. For the Gaussian loss $\frac12(y-z)^2$, $\nabla_z\ell = z - y$. In both cases: <em>prediction minus target</em>.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\partial_{z_k}\log\sum_j e^{z_j} = e^{z_k}/\sum_j e^{z_j} = p_k$, and $\partial_{z_k}(-z_y) = -\ind(k=y)$. The Gaussian case is immediate. This is the general exponential-family identity: for the canonical link, the score with respect to the natural parameter is $T(y) - \E[T(Y)]$ (see <a href="#/t/exponential-families">exponential families</a>).</p>
</details>
<p>This identity is why the output layer and loss should be paired: combining a softmax with squared error, or computing $\log(\operatorname{softmax}(z))$ in two separate steps, produces gradients that vanish when the prediction is confidently wrong, and numerical overflow. Implementations fuse the two and evaluate $\log\sum_j e^{z_j}$ by the log-sum-exp trick.</p>
<div class="ex" data-name="Heteroscedastic regression">
<p>The likelihood view also says what to do when $\sigma$ varies with $x$: let the network output two numbers, $(\mu(x), s(x))$, set $\sigma(x) = e^{s(x)}$, and minimize $\sum_i \big[(y_i-\mu(x_i))^2 e^{-2s(x_i)}/2 + s(x_i)\big]$. Squared error is the special case of a constant $s$. Any distributional regression model (Poisson counts, mixture densities) works the same way.</p>
</div>
<p>In the playground, set zero hidden layers to recover logistic regression's linear boundary, then add a hidden layer and watch the boundary bend around XOR, circles or spirals.</p>`,
        lab: 'perceptron-mlp-playground',
      },
      {
        title: 'How to think about a network statistically',
        html: R`
<p>A one-hidden-layer network $f(x) = \sum_{j=1}^m v_j\,\phi(w_j^\top x + b_j)$ is an <strong>adaptive basis expansion</strong>: a linear model in the features $\phi(w_j^\top x+b_j)$, where the features themselves are fitted. Each feature is a <em>ridge function</em> — constant along directions orthogonal to $w_j$ — which links networks to projection pursuit regression (Friedman & Stuetzle, 1981). Freezing the $w_j,b_j$ at random values gives a random-features model, a finite-dimensional approximation to a <a href="#/t/kernel-methods">kernel method</a>; deeper analyses of this connection lead to the neural tangent kernel (<a href="#/t/deep-learning-theory">theory of deep learning</a>).</p>
<p>The consequences are worth stating plainly:</p>
<ul>
<li><strong>Approximation:</strong> with enough units the class is dense in continuous functions (<a href="#/t/universal-approximation">universal approximation</a>), and adaptivity of the basis can beat any fixed basis in high dimension.</li>
<li><strong>Optimization:</strong> the empirical risk is non-convex in $\theta$ (permutation symmetry alone creates many equivalent minima), so we rely on <a href="#/t/sgd-optimizers">stochastic gradient methods</a> and good <a href="#/t/initialization-normalization">initialization</a>, with gradients computed by <a href="#/t/backpropagation">backpropagation</a>.</li>
<li><strong>Estimation:</strong> parameter counts often exceed $n$, yet test error can be small. Classical capacity measures do not explain this; the role of explicit and implicit <a href="#/t/regularization-deep">regularization</a> does much of the work.</li>
</ul>
<div class="warn" data-name="Practical pitfalls">
<p>Standardize inputs (the scale of $x$ interacts with the initialization scale). Do not read probabilities from an overfit network at face value: cross-entropy training to near-zero loss produces overconfident predictions; check <a href="#/t/classifier-evaluation">calibration</a>. And remember that a network is a <em>conditional</em> model: it says nothing about $p(x)$ and can be arbitrarily wrong off the support of the training data.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In the proof of Novikoff's theorem, which inequality uses the fact that the update was triggered by a mistake, and what would break without it?`,
        a: R`The norm bound $\|\tilde w_k\|^2 \le \|\tilde w_{k-1}\|^2 + R^2$ drops the cross term $2y_i\tilde w_{k-1}^\top\tilde x_i$ because it is $\le 0$ on a mistake. If we also updated on correctly classified points, the norm could grow linearly in $k$, matching the linear growth of $u^\top \tilde w_k$, and the Cauchy–Schwarz squeeze would give nothing.` },
      { q: R`Multiply every augmented input $\tilde x_i$ (including its constant coordinate) by $c \gt 0$. What happens to the Novikoff bound and to the perceptron's run?`,
        a: R`Both $R$ and $\gamma$ are multiplied by $c$, so $(R/\gamma)^2$ is unchanged. The run is identical too: by induction $\tilde w_k$ becomes $c\,\tilde w_k$, so the signs of $y_i\tilde w_k^\top\tilde x_i$ and hence the sequence of mistakes do not change. Rescaling only the raw features (not the constant coordinate) does change the augmented geometry, and can change both the bound and the run — one reason to centre and standardize inputs.` },
      { q: R`Why is a network with sigmoid output trained with squared error a worse idea than the same network trained with cross-entropy, even though both can fit the data?`,
        a: R`With cross-entropy, $\partial\ell/\partial z = \sigma(z) - y$, which is large when the prediction is confidently wrong. With squared error on $\sigma(z)$, $\partial\ell/\partial z = (\sigma(z)-y)\,\sigma(z)(1-\sigma(z))$, and the factor $\sigma(1-\sigma)$ vanishes when $|z|$ is large — exactly when the model is confidently wrong — so learning stalls. Cross-entropy is also the Bernoulli negative log-likelihood, hence a proper scoring rule.` },
      { q: R`Give two distinct sources of non-identifiability in a ReLU MLP and one statistical consequence.`,
        a: R`(1) Permutations of hidden units within a layer; (2) positive rescaling of a unit's incoming weights and bias by $c$ with its outgoing weights divided by $c$. Consequence: the Fisher information is singular and $\hat\theta$ is not consistent for any particular "true" parameter, so standard Wald-type inference on weights is meaningless; one targets functionals such as predictions instead.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'universal-approximation',
    domain: 'deep',
    order: 20,
    title: 'Universal Approximation & the Power of Depth',
    subtitle: 'What networks can represent, how fast, and why depth can beat width',
    level: 3,
    sources: { other: 'Cybenko (1989); Hornik (1991); Leshno, Lin, Pinkus & Schocken (1993); Barron (1993); Telgarsky (2016); Pinkus (1999) survey; Prince, Understanding Deep Learning (2023) Ch. 3–4' },
    prereqs: ['perceptron-mlp'],
    related: ['splines', 'orthogonal-series', 'kernel-methods', 'vc-theory', 'bias-variance'],
    tags: ['approximation theory', 'density', 'Barron space', 'depth separation', 'ReLU'],
    summary: R`Universal approximation theorems say that one hidden layer of a non-polynomial activation suffices to approximate any continuous function on a compact set to any accuracy. The constructive one-dimensional proof for ReLU is just piecewise-linear interpolation, and Stone–Weierstrass lifts it to any dimension. Such density results say nothing about how many units are needed: for generic smooth functions the count grows like $\varepsilon^{-d/s}$ (the curse of dimensionality), but Barron showed that functions with an integrable Fourier moment are approximated at rate $O(1/\sqrt m)$ in any dimension. Depth adds something width cannot cheaply replicate: a composed sawtooth with $2^k$ linear pieces needs only $O(k)$ units deep but exponentially many shallow.`,
    survey: {
      task: 'Characterize the function classes networks can represent and at what cost',
      assumptions: R`Compact domain; non-polynomial continuous activation; target continuous (density) or with bounded Fourier moment (Barron rates)`,
      output: R`Existence results and approximation rates in sup, $L^p$ or $L^2(\mu)$ norm`,
      strengths: 'Rules out representational limits as an excuse; explains dimension-free rates for Barron-class targets and the value of depth.',
      weaknesses: 'Existence only: no statement about learnability by gradient descent, sample complexity, or practical network sizes.',
      complexity: R`Units needed: $O(\varepsilon^{-1/2})$ in 1D for $C^2$; $\varepsilon^{-d/s}$ for $C^s$ in $d$ dims; $O(C_f^2/\varepsilon^2)$ for Barron $L^2$ error $\varepsilon$`,
    },
    sections: [
      {
        title: 'What "universal" means',
        html: R`
<p>Let $K\subset\R^d$ be compact and let $\mathcal N_\phi$ be the set of one-hidden-layer networks
$$g(x) = \sum_{j=1}^m v_j\,\phi(w_j^\top x + b_j),\qquad m\in\mathbb{N},\ v_j,b_j\in\R,\ w_j\in\R^d.$$
$\mathcal N_\phi$ is <em>universal</em> on $K$ if it is dense in $C(K)$ with the sup norm: for every continuous $f$ and $\varepsilon\gt 0$ there is $g\in\mathcal N_\phi$ with $\sup_{x\in K}|f(x)-g(x)| \lt \varepsilon$.</p>
<div class="thm" data-name="Universal approximation">
<ol>
<li>(Cybenko, 1989) If $\phi$ is continuous and sigmoidal ($\phi(t)\to 1$ as $t\to\infty$, $\phi(t)\to 0$ as $t\to-\infty$), then $\mathcal N_\phi$ is dense in $C([0,1]^d)$.</li>
<li>(Hornik, 1991) If $\phi$ is bounded and non-constant, $\mathcal N_\phi$ is dense in $L^p(\mu)$ for every finite measure $\mu$ on $\R^d$, $1\le p\lt\infty$, and in $C(K)$ if $\phi$ is also continuous.</li>
<li>(Leshno, Lin, Pinkus & Schocken, 1993) If $\phi$ is continuous, $\mathcal N_\phi$ is dense in $C(K)$ for every compact $K$ <strong>if and only if $\phi$ is not a polynomial</strong>.</li>
</ol>
</div>
<p>The "only if" in (3) is easy and instructive: if $\phi$ is a polynomial of degree $r$, every $g\in\mathcal N_\phi$ is a polynomial of degree $\le r$, a finite-dimensional — hence closed and proper — subspace of $C(K)$. Everything else (sigmoids, tanh, ReLU, GELU, even $\sin$) is universal.</p>
<div class="warn" data-name="What these theorems do not say">
<p>They give no bound on $m$, no guarantee that gradient descent finds the approximant, and nothing about how much data is needed to estimate it. Universality is a property shared by polynomials, splines, Fourier series and Gaussian kernel machines; it is the floor, not the explanation of deep learning's success. The interesting questions are <em>rates</em> and the role of <em>depth</em>.</p>
</div>`,
      },
      {
        title: 'A constructive proof in one dimension',
        html: R`
<p>For ReLU, $\phi(t) = (t)_+ = \max(t,0)$, the one-dimensional case is elementary and gives explicit rates.</p>
<div class="thm" data-kind="Lemma" data-name="Piecewise-linear functions are ReLU networks">
<p>Every continuous piecewise-linear $g$ on $[a,b]$ with knots $a = t_0\lt t_1\lt\cdots\lt t_N = b$ can be written as
$$g(x) = g(t_0) + \sum_{k=0}^{N-1} c_k\,(x - t_k)_+,\qquad c_0 = s_0,\quad c_k = s_k - s_{k-1},$$
where $s_k$ is the slope of $g$ on $[t_k, t_{k+1}]$. This is a one-hidden-layer ReLU network with $N$ units.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The right-hand side is continuous and piecewise linear with knots at the $t_k$, equals $g(t_0)$ at $x = t_0$, and on $(t_k, t_{k+1})$ its slope is $\sum_{j\le k} c_j = s_k$ (a telescoping sum). Two continuous functions with the same value at $t_0$ and the same derivative on every piece coincide.</p>
</details>
<div class="thm" data-name="1D universal approximation with rates">
<p>Let $f\in C[a,b]$ with modulus of continuity $\omega_f(h) = \sup_{|x-y|\le h}|f(x)-f(y)|$, and let $g_N$ be its linear interpolant on $N$ equal cells of width $h = (b-a)/N$. Then $g_N$ is a ReLU network with $N$ units and
$$\|f - g_N\|_\infty \le \omega_f(h)\ \xrightarrow{N\to\infty}\ 0 .$$
If $f\in C^2$, then $\|f-g_N\|_\infty \le \tfrac{h^2}{8}\|f''\|_\infty$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>On a cell $[t_k,t_{k+1}]$, $g_N(x)$ is a convex combination $\lambda f(t_k) + (1-\lambda) f(t_{k+1})$, so $|f(x) - g_N(x)| \le \lambda|f(x)-f(t_k)| + (1-\lambda)|f(x)-f(t_{k+1})| \le \omega_f(h)$. Uniform continuity on a compact interval gives $\omega_f(h)\to0$. For $C^2$, the interpolation error $e = f - g_N$ vanishes at both ends of the cell and $e'' = f''$; the standard Rolle argument gives $e(x) = \tfrac12 f''(\xi)(x-t_k)(x-t_{k+1})$, and $|(x-t_k)(x-t_{k+1})|\le h^2/4$.</p>
</details>
<p>So $\varepsilon$ sup-accuracy for a $C^2$ target costs $N = O(\varepsilon^{-1/2})$ units; a function with a kink costs $O(\varepsilon^{-1})$; and a <em>discontinuous</em> target cannot be approximated in sup norm at all (though it can in $L^p$). Another view of the same construction uses "hat" bumps $\varphi_k(x) = \frac1h\big[(x-t_{k-1})_+ - 2(x-t_k)_+ + (x-t_{k+1})_+\big]$, each made of three ReLUs, so that $g_N = \sum_k f(t_k)\varphi_k$ — a partition of unity, exactly as in <a href="#/t/splines">linear splines</a>.</p>`,
        lab: 'universal-approximation-bumps',
      },
      {
        title: 'Lifting to $d$ dimensions',
        html: R`
<p>A short route from 1D to $\R^d$ uses ridge functions and the Stone–Weierstrass theorem; it proves universality for ReLU (and, by the same argument, for any activation that is universal in 1D).</p>
<div class="thm" data-name="Universality of one-hidden-layer ReLU networks on compacta">
<p>For compact $K\subset\R^d$, one-hidden-layer ReLU networks are dense in $C(K)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p><em>Step 1 (Stone–Weierstrass).</em> Let $\mathcal T$ be the linear span of $\{\cos(w^\top x + b) : w\in\R^d, b\in\R\}$. It contains constants ($w=0$), separates points (if $x\ne y$, choose $w$ with $w^\top(x-y)\notin 2\pi\mathbb{Z}$ and $b=0$, using also $\sin = \cos(\cdot - \pi/2)$), and is closed under multiplication by the product-to-sum identity $\cos\alpha\cos\beta = \tfrac12[\cos(\alpha+\beta)+\cos(\alpha-\beta)]$. So $\mathcal T$ is a subalgebra of $C(K)$, hence dense. Given $f$ and $\varepsilon$, pick $T(x) = \sum_{k=1}^r a_k\cos(w_k^\top x + b_k)$ with $\|f - T\|_\infty\lt \varepsilon/2$.</p>
<p><em>Step 2 (1D approximation of each ridge).</em> As $x$ ranges over $K$, $t_k = w_k^\top x + b_k$ ranges over a compact interval $I_k$. By the 1D theorem there is a ReLU network $\psi_k(t) = \beta_k + \sum_j c_{kj}(t - \tau_{kj})_+$ with $\sup_{t\in I_k}|\cos t - \psi_k(t)| \lt \varepsilon/(2r\max_k|a_k|)$.</p>
<p><em>Step 3 (assemble).</em> $g(x) = \sum_k a_k\psi_k(w_k^\top x + b_k)$ is a one-hidden-layer ReLU network (each unit $(w_k^\top x + b_k - \tau_{kj})_+$ is a ReLU of an affine function of $x$, and constants are absorbed into an output bias), and $\|f-g\|_\infty \le \|f-T\|_\infty + \sum_k|a_k|\,\|\cos - \psi_k\|_{I_k} \lt \varepsilon$.</p>
</details>
<p>Cybenko's original proof is non-constructive and functional-analytic: if the closure of $\mathcal N_\phi$ were a proper closed subspace of $C(K)$, Hahn–Banach would supply a nonzero bounded linear functional vanishing on it, which by the Riesz representation theorem is a nonzero signed measure $\mu$ with $\int\phi(w^\top x + b)\,\mu(\d x) = 0$ for all $w,b$. A sigmoidal $\phi$ is "discriminatory" — this forces $\mu = 0$ — a contradiction.</p>
<div class="note">
<p>Every proof of this kind manufactures a huge network. The Stone–Weierstrass route needs as many ridge directions as a trigonometric approximation of $f$, which for a generic smooth function in $d$ dimensions grows exponentially in $d$. Universality is cheap; efficiency is not.</p>
</div>`,
      },
      {
        title: 'Rates, the curse of dimensionality, and Barron space',
        html: R`
<p>For classical smoothness classes there is no escape from dimension. If $f$ has $s$ bounded derivatives on $[0,1]^d$, any method that depends continuously on $f$ through $m$ parameters has worst-case error of order $m^{-s/d}$ (DeVore, Howard & Micchelli, 1989): halving the error multiplies the parameter count by $2^{d/s}$. Networks do not beat this for generic $C^s$ functions.</p>
<p>Barron identified a different kind of regularity under which one hidden layer is dimension-free.</p>
<div class="thm" data-name="Barron (1993)">
<p>Let $f:\R^d\to\R$ have Fourier representation $f(x) = \int e^{i\omega^\top x}\hat f(\omega)\,\d\omega$ and finite first moment $C_f = \int\|\omega\|\,|\hat f(\omega)|\,\d\omega$. Let $B$ be the ball of radius $r$ and $\mu$ any probability measure on $B$. For every $m$ there is a one-hidden-layer network $f_m$ with $m$ sigmoidal units such that
$$\int_B \big(f(x) - f_m(x)\big)^2\,\mu(\d x) \;\le\; \frac{(2rC_f)^2}{m}.$$</p>
</div>
<p>The rate $m^{-1/2}$ in $L^2$ does not depend on $d$ (though $C_f$ can). By contrast, Barron also showed that any <em>fixed</em> basis of $m$ functions — linear approximation, as with polynomials or a fixed dictionary of features — has worst-case error over the same class at least of order $C_f/(d\,m^{1/d})$. Adaptivity of the basis is what buys the improvement.</p>
<div class="thm" data-kind="Lemma" data-name="Maurey's sampling lemma">
<p>Let $g$ lie in the closed convex hull of a set $G$ in a Hilbert space with $\|h\|\le b$ for all $h\in G$. Then for every $m$ there are $h_1,\dots,h_m\in G$ with $\big\|g - \tfrac1m\sum_{j}h_j\big\|^2 \le \frac{b^2 - \|g\|^2}{m}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>First suppose $g = \E[h]$ for a random element $h$ supported on finitely many points of $G$ (a finite convex combination). Draw $h_1,\dots,h_m$ IID from that law. Then $\E\big\|\tfrac1m\sum_j h_j - g\big\|^2 = \tfrac1m\E\|h - g\|^2 = \tfrac1m(\E\|h\|^2 - \|g\|^2) \le \tfrac{b^2-\|g\|^2}{m}$, so some realization does at least as well as the average — the probabilistic method. For $g$ in the closure, approximate by finite convex combinations and pass to the limit.</p>
</details>
<details class="proof"><summary>Proof sketch of Barron's theorem</summary>
<p>Writing $\hat f(\omega) = |\hat f(\omega)|e^{i\theta(\omega)}$ and subtracting $f(0)$, the Fourier representation expresses $f(x)-f(0)$ as an average, under the probability density $\propto\|\omega\||\hat f(\omega)|$, of functions $\frac{C_f}{\|\omega\|}\big[\cos(\omega^\top x+\theta(\omega)) - \cos\theta(\omega)\big]$, each bounded by $C_f r$ on $B$. So $f - f(0)$ lies in the closed convex hull of ridge functions of sup norm $\le C_f r$ on $B$. Each such ridge function is itself in the closed convex hull of (scaled, shifted) step functions of $\omega^\top x$, and steps are limits of steep sigmoids; tracking constants gives the bound $2rC_f$. Apply Maurey's lemma in $L^2(\mu)$ with $b = 2rC_f$.</p>
</details>
<div class="note">
<p>The same convex-hull-plus-sampling structure underlies the analysis of random features, boosting and Monte Carlo integration: averaging $m$ IID draws costs variance$/m$, with no dimension in sight. What Barron's class excludes is functions whose energy sits at high frequencies in many directions.</p>
</div>`,
      },
      {
        title: 'Depth separation: the sawtooth',
        html: R`
<p>Depth composes functions, and composition can multiply complexity. The cleanest example is Telgarsky's (2016) sawtooth. Let
$$T(x) = 2(x)_+ - 4\big(x-\tfrac12\big)_+ ,\qquad x\in[0,1],$$
the tent map: $T(x) = 2x$ on $[0,\tfrac12]$ and $2-2x$ on $[\tfrac12,1]$. It maps $[0,1]$ onto itself using two ReLUs. Its $k$-fold composition $T^k = T\circ\cdots\circ T$ is a sawtooth with $2^{k-1}$ teeth, i.e. $2^k$ linear pieces, computed by a ReLU network of depth $k$ and width 2.</p>
<div class="thm" data-kind="Lemma" data-name="Piece counting">
<p>In one input dimension, a ReLU network with $L$ hidden layers of width at most $m$ computes a continuous piecewise-linear function with at most $(m+1)^L$ pieces.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $B_\ell$ be the set of breakpoints of all units in layers $\le \ell$, and $P_\ell = |B_\ell|+1$ the number of intervals they define. Every pre-activation in layer $\ell$ is a linear combination of layer-$(\ell-1)$ outputs, hence affine on each of these $P_{\ell-1}$ intervals. Applying a ReLU to an affine function adds at most one breakpoint per interval, so each of the $\le m$ units adds at most $P_{\ell-1}$ breakpoints: $P_\ell \le P_{\ell-1} + mP_{\ell-1} = (m+1)P_{\ell-1}$. With $P_0 = 1$, $P_L\le(m+1)^L$; the output is affine on each interval.</p>
</details>
<p>So representing $T^k$ <em>exactly</em> with one hidden layer requires $m+1\ge 2^k$ units, versus $2k$ units with depth $k$. Exact representation is a weak notion, but a counting argument makes the separation robust.</p>
<div class="thm" data-kind="Proposition" data-name="Few pieces cannot approximate the sawtooth">
<p>If $g:[0,1]\to\R$ is continuous and piecewise linear with $p$ pieces, then
$$\int_0^1 |g(x) - T^k(x)|\,\d x \;\ge\; \frac{1 - (2p+2)\,2^{-k}}{8}.$$
In particular, if $p \le 2^{k-2}-1$ the $L^1$ error is at least $1/16$. Combined with the lemma, a one-hidden-layer ReLU net needs $m\gtrsim 2^{k-2}$ units, and an $L$-layer net needs width $m\gtrsim 2^{(k-2)/L}$, to beat that error.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The level $\tfrac12$ cuts the graph of $T^k$ into $2^k - 1$ full triangles (plus two half-triangles at the ends), alternating above and below the line $y=\tfrac12$; each full triangle has base $2^{-k}$, height $\tfrac12$ and area $2^{-k-2}$. Split $[0,1]$ into maximal intervals on which $g\gt\tfrac12$ or $g\le\tfrac12$. An affine piece crosses the level at most once, so there are at most $p$ switches and at most $p+1$ such intervals. At most $p$ triangles contain a switch point; the others lie inside one interval, and within any interval consecutive triangles alternate sides, so at least $\lfloor c/2\rfloor$ of the $c$ triangles it contains lie on the side opposite to $g$. On such a triangle $|g - T^k|\ge|T^k - \tfrac12|$, contributing its full area. Summing, at least $\tfrac12\big[(2^k - 1 - p) - (p+1)\big]$ triangles are mismatched, so the integral is at least $\tfrac12(2^k - 2p - 2)\,2^{-k-2}$.</p>
</details>
<p>Telgarsky turned this into a general theorem: there are functions computed by ReLU networks with $\Theta(k^3)$ layers and $\Theta(k^3)$ units that no network with $O(k)$ layers and fewer than order $2^k$ units can approximate within a constant in $L^1$. Similar separations exist between depth 2 and depth 3 in high dimension (Eldan & Shamir, 2016). The lab shows the counting at work.</p>`,
        lab: 'universal-approximation-sawtooth',
      },
      {
        title: 'Width, depth, and what approximation theory leaves open',
        html: R`
<p>Several complementary facts round out the picture.</p>
<ul>
<li><strong>Narrow-but-deep is universal.</strong> If the width exceeds the input dimension by a small constant, deep ReLU networks are universal: width $d+1$ suffices for $C(K)$ (Hanin & Sellke, 2017) and width $d+4$ for $L^1(\R^d)$ (Lu et al., 2017), while width $\le d$ is not enough.</li>
<li><strong>Depth helps for smooth functions too.</strong> Yarotsky (2017) showed deep ReLU networks approximate $x\mapsto x^2$ to accuracy $\varepsilon$ with $O(\log(1/\varepsilon))$ units (by composing sawtooths), hence products and polynomials efficiently; this gives near-optimal rates for Sobolev balls with depth growing like $\log(1/\varepsilon)$.</li>
<li><strong>Compositional structure beats the curse.</strong> If $f$ is a composition of functions each depending on few variables (a hierarchical structure), deep networks matching the hierarchy achieve rates governed by the local dimension rather than $d$ (Poggio et al., 2017; Schmidt-Hieber, 2020 for nonparametric regression).</li>
</ul>
<div class="warn" data-name="Approximation is one third of the story">
<p>Excess risk decomposes into approximation, estimation and optimization error (<a href="#/t/bias-variance">bias–variance</a>, <a href="#/t/vc-theory">uniform convergence</a>). A class can approximate well yet be hard to estimate (too rich for the sample size) or hard to optimize (the good approximant exists but gradient descent does not find it — learning parities with networks is a known hard case). Approximation results are best read as "the representational capacity is not the bottleneck".</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Why can a one-hidden-layer network with activation $\phi(t) = t^3$ never be universal, no matter how many units it has?`,
        a: R`Every unit $v(w^\top x + b)^3$ is a polynomial of degree $\le 3$ in $x$, so every network lies in the finite-dimensional space of polynomials of degree $\le 3$. That subspace is closed and proper in $C(K)$, so it cannot be dense (the easy direction of Leshno et al.).` },
      { q: R`How many ReLU units does the interpolation construction need to approximate $f(x)=\sin(2\pi x)$ on $[0,1]$ to sup-error $10^{-4}$?`,
        a: R`With $h = 1/N$, the bound $\tfrac{h^2}{8}\|f''\|_\infty = \tfrac{4\pi^2}{8N^2}$ must be $\le 10^{-4}$, so $N \ge \sqrt{4\pi^2/(8\cdot10^{-4})} = \pi\sqrt{5000}\approx 222$ units.` },
      { q: R`Barron's bound gives error $O(m^{-1/2})$ independent of $d$. Why does this not contradict the $m^{-s/d}$ lower bound for $C^s$ functions?`,
        a: R`They concern different function classes. The $m^{-s/d}$ lower bound is worst-case over a smoothness ball in $C^s$; Barron's class (finite $\int\|\omega\||\hat f|$) is a different, much more restrictive class in high dimension. Also the lower bound is for methods depending continuously on $f$; the guarantee is only that for Barron functions, networks achieve a dimension-free rate.` },
      { q: R`Using the piece-counting lemma, what is the minimum width of a 3-hidden-layer ReLU net that represents $T^{12}$ exactly?`,
        a: R`$T^{12}$ has $2^{12} = 4096$ pieces, so we need $(m+1)^3 \ge 4096$, i.e. $m+1\ge 16$ and $m\ge 15$. A depth-12 width-2 net does it with 24 units, while one hidden layer needs at least 4095.` },
    ],
  });
})();
