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

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'backpropagation',
    domain: 'deep',
    order: 30,
    title: 'Backpropagation & Automatic Differentiation',
    subtitle: 'Reverse-mode differentiation of a computational graph, and why deep gradients vanish',
    level: 2,
    sources: { other: 'Rumelhart, Hinton & Williams (1986); Linnainmaa (1970); Baur & Strassen (1983); Griewank & Walther, Evaluating Derivatives (2008); Baydin et al. (2018) survey; Bengio, Simard & Frasconi (1994); Goodfellow, Bengio & Courville (2016) §6.5; Prince, Understanding Deep Learning (2023) Ch. 7' },
    prereqs: ['perceptron-mlp'],
    related: ['sgd-optimizers', 'initialization-normalization', 'mle', 'irls'],
    tags: ['chain rule', 'reverse mode', 'automatic differentiation', 'vanishing gradients', 'Jacobian'],
    summary: R`Training a network needs the gradient of a scalar loss with respect to millions of parameters. Backpropagation is reverse-mode automatic differentiation: write the computation as a directed acyclic graph of primitive operations, run it forward storing intermediate values, then sweep backward propagating <em>adjoints</em> $\bar u = \partial L/\partial u$, each node multiplying the incoming adjoint by its local derivative. The whole gradient costs a small constant multiple of one forward pass, independent of the number of parameters — the "cheap gradient principle". For an MLP the backward sweep is two matrix products per layer, and it reveals why gradients in deep networks vanish or explode: they are products of many Jacobians.`,
    survey: {
      task: 'Exact gradients of a scalar function of many parameters',
      assumptions: R`Function given as a program of differentiable (a.e.) primitives; enough memory to store or recompute intermediates`,
      output: R`$\nabla_\theta L$ to machine precision; more generally vector–Jacobian products $u^\top J$`,
      strengths: 'Cost independent of parameter count; exact (no truncation error); mechanical, so frameworks do it automatically.',
      weaknesses: 'Memory proportional to the forward computation; long products of Jacobians vanish/explode; non-smooth points need conventions.',
      complexity: R`Time $\le c\cdot$cost$(L)$ with small constant ($\approx 2$–$3$ in practice); memory $O(\text{forward activations})$`,
    },
    sections: [
      {
        title: 'Three ways to compute a gradient',
        html: R`
<p>Let $L:\R^P\to\R$ be the loss as a function of the parameters. There are three generic ways to obtain $\nabla L(\theta)$.</p>
<div class="cols">
<div><h4>Symbolic</h4><p>Differentiate the formula by hand or computer algebra. Exact, but expressions swell (the derivative of a product of $L$ factors has $L$ terms; nested compositions repeat subexpressions), and the result is a formula rather than an efficient program.</p></div>
<div><h4>Finite differences</h4><p>$\partial_j L \approx [L(\theta + he_j) - L(\theta - he_j)]/(2h)$. Needs $2P$ evaluations, and accuracy is limited: truncation error $O(h^2)$ against rounding error $O(\epsilon_{\text{mach}}/h)$, optimized at $h\sim\epsilon_{\text{mach}}^{1/3}$ with error $\sim\epsilon_{\text{mach}}^{2/3}\approx 10^{-11}$.</p></div>
<div><h4>Automatic differentiation</h4><p>Apply the chain rule to the <em>program</em>, numerically, primitive by primitive. Exact up to rounding, and in reverse mode the whole gradient costs a constant multiple of one evaluation of $L$.</p></div>
</div>
<p>For $P = 10^8$ parameters, finite differences would need $2\times 10^8$ forward passes per step; backpropagation needs about three passes' worth of work. That difference is the reason deep learning is computationally possible. Finite differences remain indispensable as a <em>check</em> on hand-written gradients.</p>`,
      },
      {
        title: 'Computational graphs and the adjoint recursion',
        html: R`
<div class="def" data-name="Evaluation trace">
<p>A computation is a sequence $v_1,\dots,v_N$ where $v_1,\dots,v_P$ are inputs and each later $v_j = \psi_j(v_{\mathrm{pa}(j)})$ applies a primitive $\psi_j$ (add, multiply, $\exp$, matrix product, ...) to earlier variables, its <em>parents</em>. The output is $L = v_N$. The variables and parent relations form a directed acyclic graph. The <strong>adjoint</strong> of $v_i$ is the total derivative $\bar v_i = \partial L/\partial v_i$: the sensitivity of $L$ to a perturbation of $v_i$ when everything downstream is recomputed.</p>
</div>
<div class="thm" data-name="Adjoint recursion (reverse-mode chain rule)">
<p>$\bar v_N = 1$, and for $i\lt N$,
$$\bar v_i = \sum_{j\,:\, i\in \mathrm{pa}(j)} \bar v_j\, \frac{\partial \psi_j}{\partial v_i}.$$
Processing nodes in reverse topological order computes every adjoint, in particular the gradient $(\bar v_1,\dots,\bar v_P)$, in one backward sweep.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the multivariate chain rule applied inductively along the topological order, the total derivative of $v_N$ with respect to $v_i$ is the <em>path sum</em>: the sum, over all directed paths $i = k_0\to k_1\to\cdots\to k_r = N$, of the products of local partials $\prod_{s}\partial\psi_{k_{s}}/\partial v_{k_{s-1}}$. (Induct on the position of the output: a perturbation of $v_i$ reaches $v_N$ only through the parents of $v_N$, and to first order these contributions add.) Every path from $i$ starts with an edge $i\to j$ to some child $j$ and continues with a path from $j$ to $N$. Grouping the path sum by that first edge gives $\sum_j (\partial\psi_j/\partial v_i)\cdot(\text{path sum from } j) = \sum_j(\partial\psi_j/\partial v_i)\,\bar v_j$. Children come later in the order, so their adjoints are available when $\bar v_i$ is computed.</p>
</details>
<p>Grouping the same path sum by the <em>last</em> edge instead gives <strong>forward mode</strong>: propagate tangents $\dot v_j = \sum_{i\in\mathrm{pa}(j)}(\partial\psi_j/\partial v_i)\dot v_i$ alongside the evaluation, which yields one directional derivative $J\dot\theta$ per sweep. For $F:\R^n\to\R^m$ with Jacobian $J$:</p>
<table>
<thead><tr><th></th><th>One sweep computes</th><th>Full Jacobian needs</th><th>Best when</th></tr></thead>
<tbody>
<tr><td>Forward mode</td><td>$J u$ (Jacobian–vector product)</td><td>$n$ sweeps</td><td>few inputs, many outputs</td></tr>
<tr><td>Reverse mode</td><td>$u^\top J$ (vector–Jacobian product)</td><td>$m$ sweeps</td><td>many inputs, few outputs</td></tr>
</tbody>
</table>
<div class="thm" data-name="Cheap gradient principle (Baur & Strassen, 1983; Griewank & Walther, 2008)">
<p>If each primitive's local partials cost a bounded multiple of the primitive itself, then for $f:\R^P\to\R$ reverse mode computes $\nabla f$ with arithmetic cost at most a small constant times the cost of evaluating $f$ — a constant independent of $P$.</p>
</div>
<p>The price is <strong>memory</strong>: the backward sweep needs the forward intermediates (or the ability to recompute them), so memory scales with the length of the computation. Checkpointing stores only some intermediates and recomputes the rest; for a chain of $L$ layers, keeping $\sqrt L$ checkpoints gives $O(\sqrt L)$ memory for about one extra forward pass (Griewank, 1992; Chen et al., 2016).</p>`,
      },
      {
        title: 'Backpropagation through an MLP, in matrix form',
        html: R`
<p>Use the notation of <a href="#/t/perceptron-mlp">the MLP</a>: $z_\ell = W_\ell h_{\ell-1} + b_\ell$, $h_\ell = \phi(z_\ell)$, output $z_L$, loss $L = \ell(z_L, y)$. Define the error signals $\delta_\ell = \partial L/\partial z_\ell\in\R^{n_\ell}$.</p>
<div class="thm" data-name="Backpropagation equations">
<p>$$\delta_L = \nabla_{z_L}\ell,\qquad \delta_\ell = \phi'(z_\ell)\odot\big(W_{\ell+1}^\top\delta_{\ell+1}\big)\quad(\ell \lt L),$$
$$\frac{\partial L}{\partial W_\ell} = \delta_\ell\,h_{\ell-1}^\top,\qquad \frac{\partial L}{\partial b_\ell} = \delta_\ell .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Work with differentials. By definition of $\delta_\ell$, $\d L = \delta_\ell^\top \d z_\ell$ for any perturbation of $z_\ell$ that propagates downstream. With the weights fixed, a perturbation $\d z_\ell$ gives $\d h_\ell = \phi'(z_\ell)\odot \d z_\ell = \diag(\phi'(z_\ell))\,\d z_\ell$ and $\d z_{\ell+1} = W_{\ell+1}\d h_\ell$. Hence $\d L = \delta_{\ell+1}^\top W_{\ell+1}\diag(\phi'(z_\ell))\,\d z_\ell$, so $\delta_\ell = \diag(\phi'(z_\ell))W_{\ell+1}^\top\delta_{\ell+1}$. For the weights, perturbing $W_\ell$ alone gives $\d z_\ell = (\d W_\ell)h_{\ell-1}$, so $\d L = \delta_\ell^\top(\d W_\ell) h_{\ell-1} = \tr\big(h_{\ell-1}\delta_\ell^\top \d W_\ell\big) = \langle \delta_\ell h_{\ell-1}^\top, \d W_\ell\rangle_F$, which identifies the gradient. The bias case is $\d z_\ell = \d b_\ell$.</p>
</details>
<p>For a minibatch of $B$ examples stacked as rows ($H_\ell\in\R^{B\times n_\ell}$, $\Delta_\ell\in\R^{B\times n_\ell}$) and $L$ the mean loss:</p>
<div class="algo" data-name="Minibatch backpropagation">
<ol>
<li>Forward: $Z_\ell = H_{\ell-1}W_\ell^\top + \mathbf 1 b_\ell^\top$, $H_\ell = \phi(Z_\ell)$; store $H_\ell$ and $Z_\ell$.</li>
<li>Output: $\Delta_L = \tfrac1B\nabla_{Z_L}\ell$ (e.g. $\tfrac1B(\hat P - Y)$ for softmax cross-entropy).</li>
<li>For $\ell = L,\dots,1$: $\nabla_{W_\ell} = \Delta_\ell^\top H_{\ell-1}$, $\nabla_{b_\ell} = \Delta_\ell^\top\mathbf 1$; if $\ell\gt 1$, $\Delta_{\ell-1} = (\Delta_\ell W_\ell)\odot\phi'(Z_{\ell-1})$.</li>
</ol>
</div>
<p>The forward pass costs one matrix product per layer; the backward pass costs two (one for $\Delta_{\ell-1}$, one for $\nabla_{W_\ell}$). So a training step costs about three forward passes, in line with the cheap gradient principle.</p>
<div class="ex" data-name="A 2–2–1 network by hand">
<p>Take $x = (1, -0.5)$, $W = \begin{pmatrix}0.8&-0.6\\0.4&0.9\end{pmatrix}$, $b = (0.1,-0.2)$, $\phi = \tanh$, output $\hat y = v^\top h + c$ with $v = (1.1,-0.7)$, $c = 0.05$, target $y=1$ and $L = \tfrac12(\hat y - y)^2$.</p>
<p><em>Forward.</em> $z = (1.2, -0.25)$, $h = (0.8337, -0.2449)$, $\hat y = 1.1385$, $L = 0.00959$.</p>
<p><em>Backward.</em> $\bar{\hat y} = \hat y - y = 0.1385$. Then $\bar v = \bar{\hat y}\,h = (0.1154, -0.0339)$ and $\bar h = \bar{\hat y}\,v = (0.1523, -0.0969)$. Through tanh, $\phi'(z) = 1 - h^2 = (0.3050, 0.9400)$, so $\delta = \bar z = (0.0465, -0.0911)$. Finally $\bar W = \delta x^\top = \begin{pmatrix}0.0465 & -0.0232\\ -0.0911 & 0.0456\end{pmatrix}$, $\bar b = \delta$, and $\bar x = W^\top\delta = (0.0007, -0.1099)$.</p>
<p>These are the default numbers in the lab below; step through it and compare the backprop column with the finite-difference column.</p>
</div>`,
        lab: 'backpropagation-graph',
      },
      {
        title: 'Vector–Jacobian products: the autodiff toolbox',
        html: R`
<p>Modern frameworks never form Jacobians. Each primitive $y = \psi(x)$ registers a <strong>vector–Jacobian product</strong> (VJP) $\bar y\mapsto \bar x = J_\psi(x)^\top \bar y$, and reverse mode chains VJPs. A few that every practitioner should be able to derive:</p>
<table>
<thead><tr><th>Primitive</th><th>Forward</th><th>VJP (adjoints of inputs)</th></tr></thead>
<tbody>
<tr><td>Matrix product</td><td>$Y = AB$</td><td>$\bar A = \bar Y B^\top$, $\ \bar B = A^\top\bar Y$</td></tr>
<tr><td>Elementwise</td><td>$y = \phi(x)$</td><td>$\bar x = \phi'(x)\odot\bar y$</td></tr>
<tr><td>Sum / broadcast</td><td>$s = \sum_i x_i$; $\ y_i = s$</td><td>$\bar x_i = \bar s$; $\ \bar s = \sum_i\bar y_i$</td></tr>
<tr><td>Softmax</td><td>$p = \operatorname{softmax}(z)$</td><td>$\bar z = p\odot(\bar p - p^\top\bar p\,\mathbf 1)$</td></tr>
<tr><td>Log-sum-exp</td><td>$s = \log\sum_j e^{z_j}$</td><td>$\bar z = \bar s\, p$</td></tr>
<tr><td>Inverse</td><td>$Y = A^{-1}$</td><td>$\bar A = -Y^\top\bar Y Y^\top$</td></tr>
</tbody>
</table>
<p>Two structural rules complete the picture. <strong>Fan-out sums:</strong> if a variable is used in several places, its adjoint is the sum of the adjoints flowing back from each use (this is the sum in the adjoint recursion; summation and broadcasting are adjoint to each other). <strong>Parameter sharing sums:</strong> a weight used at every time step of a recurrent network, or at every position of a convolution, receives the sum of the per-use gradients.</p>
<div class="note" data-name="Second derivatives cheaply">
<p>A Hessian–vector product $\nabla^2L(\theta)\,u$ is the directional derivative of $\nabla L$ along $u$: run forward mode over the reverse-mode gradient computation ("forward-over-reverse", Pearlmutter, 1994). It costs a small multiple of one gradient — never forming the $P\times P$ Hessian — which is what makes Newton–CG, Lanczos estimates of curvature, and influence functions feasible for large models.</p>
</div>`,
      },
      {
        title: 'Vanishing and exploding gradients',
        html: R`
<p>Unrolling the recursion for $\delta_\ell$ from the output back to the first layer,
$$\delta_1 = D_1W_2^\top D_2W_3^\top\cdots D_{L-1}W_L^\top\,\delta_L,\qquad D_\ell = \diag(\phi'(z_\ell)).$$
The first-layer gradient is a product of $L-1$ Jacobian transposes applied to the output error, and products of many matrices behave geometrically.</p>
<div class="thm" data-kind="Proposition" data-name="Geometric decay bound">
<p>$\|\delta_1\| \le \Big(\prod_{\ell=1}^{L-1}\|D_\ell\|\,\|W_{\ell+1}\|\Big)\|\delta_L\|$ in the operator norm. For the logistic sigmoid, $\|D_\ell\|\le\tfrac14$, so if every $\|W_\ell\|\le c\lt 4$ the first-layer error signal is at most $(c/4)^{L-1}\|\delta_L\|$ — exponentially small in depth.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Submultiplicativity of the operator norm, $\|AB\|\le\|A\|\|B\|$, and $\|W^\top\| = \|W\|$; for the sigmoid, $\phi' = \phi(1-\phi)\le\tfrac14$ so the diagonal matrix $D_\ell$ has norm at most $\tfrac14$.</p>
</details>
<p>Upper bounds explain vanishing; the typical behaviour at a random initialization is captured by a mean-field heuristic. If $W_{\ell+1}$ has IID entries of variance $\sigma_w^2/n$ independent of the forward pass, then $\E\|W_{\ell+1}^\top D\delta\|^2\approx\sigma_w^2\,\E[\phi'(z)^2]\,\|\delta\|^2$ for wide layers, so the squared signal is multiplied by roughly
$$\chi = \sigma_w^2\,\E[\phi'(z)^2]$$
per layer. If $\chi\lt 1$ gradients vanish geometrically; if $\chi\gt 1$ they explode. For ReLU, $\E[\phi'(z)^2] = \tfrac12$, so $\chi = 1$ exactly at $\sigma_w^2 = 2$ — the He initialization. This calculation is developed in <a href="#/t/initialization-normalization">initialization and normalization</a>.</p>
<p>Recurrent networks are the extreme case: the same matrix $W$ appears at every step, so the backpropagated signal behaves like powers of $W^\top D$ and is governed by its spectral radius (Hochreiter, 1991; Bengio, Simard & Frasconi, 1994). The remedies form much of the modern toolkit: variance-preserving initialization, normalization layers, residual connections (an identity path in every Jacobian), gated recurrences (LSTM/GRU), and <strong>gradient clipping</strong> — rescale $g$ to norm $\tau$ when $\|g\|\gt\tau$ — against occasional explosions (Pascanu, Mikolov & Bengio, 2013).</p>`,
        lab: 'backpropagation-vanishing',
      },
      {
        title: 'Practicalities and pitfalls',
        html: R`
<div class="warn" data-name="Checking a gradient">
<p>Compare backprop with central differences using the <em>relative</em> error $|g - \hat g|/(|g|+|\hat g|)$, in double precision, on a handful of random coordinates. Values around $10^{-7}$–$10^{-10}$ are healthy. Turn off randomness first (fix dropout masks and data order), and expect isolated failures at non-differentiable points: if a ReLU pre-activation lies within $h$ of zero, the finite difference straddles the kink.</p>
</div>
<div class="warn" data-name="Non-smooth primitives">
<p>ReLU, max-pooling, absolute value and sorting are not differentiable everywhere. Frameworks return a fixed element of the subdifferential (e.g. $\phi'(0) = 0$ for ReLU). Since the kinks have measure zero this is harmless for training, but the computed "gradient" of a composition is not always a true (Clarke) subgradient, and theory for SGD on such functions needs care.</p>
</div>
<div class="warn" data-name="Numerical stability">
<p>Differentiate stable formulas, not naive ones: compute $\log\operatorname{softmax}$ via log-sum-exp, fuse sigmoid with binary cross-entropy (gradient $\sigma(z)-y$), and avoid $\log(\sigma(z))$ computed in two steps, whose gradient underflows exactly when the model is confidently wrong.</p>
</div>
<p>Two conceptual remarks. Backpropagation is <em>only</em> a way to compute derivatives; the learning rule (SGD, Adam, ...) is a separate choice. And for a statistician the gradient it computes is familiar: for a log-likelihood loss, the backpropagated gradient is the (negative) <a href="#/t/fisher-information">score</a> summed over the minibatch, and the same reverse sweep applied to any differentiable statistical model — a GLM fitted by <a href="#/t/irls">IRLS</a>, a state-space model, a hierarchical model in an <a href="#/t/hamiltonian-mc">HMC</a> sampler — supplies its gradients at the cost of a few likelihood evaluations.</p>`,
      },
    ],
    quiz: [
      { q: R`For $F:\R^{1000}\to\R^{3}$, how many sweeps do forward and reverse mode need to obtain the full Jacobian, and which would you use?`,
        a: R`Forward mode needs one sweep per input direction, $n = 1000$; reverse mode needs one sweep per output, $m = 3$. Use reverse mode (three VJPs with $u = e_1, e_2, e_3$).` },
      { q: R`In the minibatch algorithm, why must the forward pass store $Z_\ell$ (or $H_\ell$), and what is the memory cost for a network with $L$ layers of width $n$ and batch size $B$?`,
        a: R`The backward pass needs $\phi'(Z_{\ell})$ for the error recursion and $H_{\ell-1}$ for the weight gradient $\Delta_\ell^\top H_{\ell-1}$. Storing them costs $O(LBn)$ numbers — often more than the $O(Ln^2)$ parameters for large batches. Checkpointing trades this memory for recomputation.` },
      { q: R`Derive the VJP of $y = Ax$ with respect to $A$ when $x$ is a vector.`,
        a: R`$\d L = \bar y^\top \d y = \bar y^\top(\d A)x = \tr(x\bar y^\top \d A) = \langle \bar y x^\top, \d A\rangle_F$, so $\bar A = \bar y\,x^\top$, an outer product — the single-example form of $\partial L/\partial W_\ell = \delta_\ell h_{\ell-1}^\top$.` },
      { q: R`A 50-layer tanh network is initialized with weights of variance $0.5/n_{\text{in}}$. Using $\chi = \sigma_w^2\E[\phi'(z)^2]$, predict what happens to first-layer gradients, and name two remedies.`,
        a: R`Since $\tanh'\le 1$, $\chi\le 0.5$, so the squared error signal shrinks by at least a factor 2 per layer: $\|\delta_1\|^2$ is roughly $2^{-49}$ times $\|\delta_{50}\|^2$ — vanishing gradients, and the early layers barely train. Remedies: initialize at $\sigma_w^2\approx 1$ (LeCun/Glorot) so $\chi\approx1$; add normalization layers or residual connections.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'sgd-optimizers',
    domain: 'deep',
    order: 40,
    title: 'Stochastic Gradient Descent & Adaptive Optimizers',
    subtitle: 'Convergence guarantees, the role of noise, momentum, and Adam',
    level: 3,
    sources: { other: 'Robbins & Monro (1951); Polyak (1964); Nesterov (1983); Polyak & Juditsky (1992); Bottou, Curtis & Nocedal (2018) survey; Duchi, Hazan & Singer (2011); Kingma & Ba (2015); Reddi, Kale & Kumar (2018); Goodfellow, Bengio & Courville (2016) Ch. 8; Prince, Understanding Deep Learning (2023) Ch. 6' },
    prereqs: ['backpropagation', 'learning-framework'],
    related: ['mle', 'irls', 'lln', 'ridge-regression', 'regularization-deep'],
    tags: ['SGD', 'momentum', 'Nesterov', 'Adam', 'learning rate', 'stochastic approximation'],
    summary: R`Stochastic gradient descent replaces the full gradient of the empirical risk by the gradient on a random minibatch — an unbiased but noisy estimate costing $B/n$ of the price. For convex Lipschitz objectives the averaged iterate converges at rate $O(1/\sqrt T)$; for strongly convex smooth ones a constant step converges linearly to a noise ball of radius proportional to the step size, and decaying steps or iterate averaging remove the ball. Momentum accelerates progress along ill-conditioned valleys (rate $1-1/\sqrt\kappa$ instead of $1-1/\kappa$). Adaptive methods (AdaGrad, RMSProp, Adam) rescale each coordinate by a running estimate of its gradient magnitude. The noise is not just a nuisance: it acts as an implicit regularizer that shapes which minimum is found.`,
    survey: {
      task: 'Minimize an average of many losses with cheap noisy gradients',
      assumptions: R`Unbiased stochastic gradients with bounded variance; (for guarantees) convexity or smoothness; tuned step-size schedule`,
      output: R`An approximate minimizer $\theta_T$ (or average $\bar\theta_T$)`,
      strengths: 'Per-step cost independent of $n$; scales to huge data and models; noise helps escape saddles and sharp minima.',
      weaknesses: 'Slow final convergence; sensitive to step size; guarantees for non-convex nets are only to stationary points.',
      complexity: R`$O(B\cdot\text{cost of one gradient})$ per step; $O(1/\varepsilon^2)$ steps (convex), $O(\kappa\log(1/\varepsilon))$ (strongly convex, deterministic)`,
    },
    sections: [
      {
        title: 'From empirical risk to stochastic gradients',
        html: R`
<p>Training minimizes the <a href="#/t/learning-framework">empirical risk</a> $F(\theta) = \frac1n\sum_{i=1}^n f_i(\theta)$, $f_i(\theta) = \ell(f_\theta(x_i), y_i)$. Full-batch gradient descent $\theta_{t+1} = \theta_t - \eta\nabla F(\theta_t)$ costs $n$ per-example gradients per step. <strong>Stochastic gradient descent</strong> instead samples a minibatch $S_t$ of size $B$ and steps along
$$g_t = \frac1B\sum_{i\in S_t}\nabla f_i(\theta_t),\qquad \E[g_t\mid\theta_t] = \nabla F(\theta_t),\qquad \Cov(g_t\mid\theta_t) \approx \frac{\Sigma(\theta_t)}{B},$$
where $\Sigma(\theta)$ is the covariance of a single per-example gradient (exactly $\Sigma/B$ for sampling with replacement).</p>
<p>The idea is older than neural networks: <strong>stochastic approximation</strong> (Robbins & Monro, 1951) seeks a root of $h(\theta) = \E[H(\theta,\xi)]$ using noisy evaluations $H(\theta_t,\xi_t)$, and proved convergence with steps satisfying
$$\sum_t\eta_t = \infty,\qquad \sum_t \eta_t^2\lt\infty,$$
e.g. $\eta_t\propto 1/t$. The first condition lets the iterates travel arbitrarily far; the second makes the accumulated noise $\sum_t\eta_t\varepsilon_t$ a convergent martingale. In streaming form (one fresh example per step) SGD minimizes the <em>population</em> risk directly, which is why its analysis connects to statistical efficiency below.</p>
<div class="note">
<p>Why noisy steps can beat exact ones: early in training, all examples "agree" on the descent direction, so a gradient from 32 examples is nearly as informative as one from a million, at a tiny fraction of the cost. The noise only matters near a minimum, where the per-example gradients disagree — which is exactly where schedules decay the step.</p>
</div>`,
      },
      {
        title: 'Convergence guarantees in the convex case',
        html: R`
<div class="thm" data-name="SGD for convex Lipschitz objectives">
<p>Let $F$ be convex with a minimizer $\theta^\star$, $\|\theta_0-\theta^\star\|\le D$, and let $g_t$ satisfy $\E[g_t\mid\theta_t]\in\partial F(\theta_t)$ and $\E\|g_t\|^2\le G^2$. With constant step $\eta = D/(G\sqrt T)$ and $\bar\theta_T = \frac1T\sum_{t=0}^{T-1}\theta_t$,
$$\E F(\bar\theta_T) - F(\theta^\star) \;\le\; \frac{DG}{\sqrt T}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Expand $\|\theta_{t+1}-\theta^\star\|^2 = \|\theta_t-\theta^\star\|^2 - 2\eta\,g_t^\top(\theta_t-\theta^\star) + \eta^2\|g_t\|^2$. Condition on $\theta_t$: $\E[g_t^\top(\theta_t-\theta^\star)\mid\theta_t] = s_t^\top(\theta_t-\theta^\star)\ge F(\theta_t)-F(\theta^\star)$ for the subgradient $s_t$, by convexity. Taking expectations and summing over $t = 0,\dots,T-1$, the distances telescope:
$$2\eta\sum_{t}\E[F(\theta_t)-F^\star]\le \|\theta_0-\theta^\star\|^2 + \eta^2TG^2 \le D^2+\eta^2TG^2.$$
Divide by $2\eta T$ and use Jensen ($F(\bar\theta_T)\le\frac1T\sum_tF(\theta_t)$): $\E F(\bar\theta_T)-F^\star\le\frac{D^2}{2\eta T}+\frac{\eta G^2}{2}$. The choice $\eta = D/(G\sqrt T)$ balances the two terms, each equal to $DG/(2\sqrt T)$.</p>
</details>
<p>The rate $1/\sqrt T$ cannot be improved in general for this class (Nemirovski & Yudin, 1983). With more structure the picture sharpens and the role of the noise becomes explicit.</p>
<div class="thm" data-name="Constant step on strongly convex smooth objectives">
<p>Let $F$ be $\mu$-strongly convex and $L$-smooth, and let $g_t$ be unbiased with $\E[\|g_t-\nabla F(\theta_t)\|^2\mid\theta_t]\le\sigma^2$. For $0\lt\eta\le 1/L$,
$$\E\|\theta_t-\theta^\star\|^2\;\le\;(1-\eta\mu)^t\,\|\theta_0-\theta^\star\|^2\;+\;\frac{\eta\sigma^2}{\mu}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Conditioning on $\theta_t$ and using unbiasedness, $\E\|\theta_{t+1}-\theta^\star\|^2 = \|\theta_t-\theta^\star\|^2 - 2\eta\langle\nabla F(\theta_t),\theta_t-\theta^\star\rangle + \eta^2\big(\|\nabla F(\theta_t)\|^2+\E\|g_t-\nabla F(\theta_t)\|^2\big)$. For $L$-smooth convex $F$ with $\nabla F(\theta^\star)=0$, co-coercivity gives $\|\nabla F(\theta)\|^2\le L\langle\nabla F(\theta),\theta-\theta^\star\rangle$, so the gradient terms total at most $-\eta(2-\eta L)\langle\nabla F,\theta_t-\theta^\star\rangle\le-\eta\langle\nabla F,\theta_t-\theta^\star\rangle$ when $\eta\le1/L$. Strong convexity gives $\langle\nabla F(\theta),\theta-\theta^\star\rangle\ge\mu\|\theta-\theta^\star\|^2$. Hence $\E\|\theta_{t+1}-\theta^\star\|^2\le(1-\eta\mu)\E\|\theta_t-\theta^\star\|^2+\eta^2\sigma^2$. Unrolling, the noise contributions sum to at most $\eta^2\sigma^2\sum_{s\ge0}(1-\eta\mu)^s = \eta\sigma^2/\mu$.</p>
</details>
<p>Two regimes are visible: fast linear convergence until the iterates reach a <strong>noise ball</strong> of squared radius $\propto\eta\sigma^2/\mu$ (and $\sigma^2\propto1/B$), then a plateau. Removing the plateau requires shrinking the step — with $\eta_t\propto 1/(\mu t)$ one gets $\E\|\theta_t-\theta^\star\|^2 = O(1/t)$ — or averaging.</p>
<div class="thm" data-name="Polyak–Ruppert averaging (Polyak & Juditsky, 1992)">
<p>For a smooth strongly convex population risk with Hessian $H$ at $\theta^\star$ and per-sample gradient covariance $S$ at $\theta^\star$, streaming SGD with slowly decaying steps ($\eta_t\propto t^{-\alpha}$, $\tfrac12\lt\alpha\lt1$) and averaged iterate $\bar\theta_t$ satisfies $\sqrt t(\bar\theta_t-\theta^\star)\dto\N\big(0,\;H^{-1}SH^{-1}\big)$.</p>
</div>
<p>That covariance is the sandwich covariance of the corresponding M-estimator; for a correctly specified log-likelihood $S = H = I(\theta^\star)$, so one pass of averaged SGD over $n$ fresh examples is asymptotically as efficient as the <a href="#/t/mle">MLE</a>. For non-convex smooth $F$ the guarantees weaken to stationarity: with $\eta\propto1/\sqrt T$, $\min_{t\lt T}\E\|\nabla F(\theta_t)\|^2 = O(1/\sqrt T)$ (Ghadimi & Lan, 2013).</p>`,
        lab: 'sgd-optimizers-noise',
      },
      {
        title: 'Noise as an implicit regularizer',
        html: R`
<p>For small steps, SGD is well approximated by the stochastic differential equation (Li, Tai & E, 2017)
$$\d\theta = -\nabla F(\theta)\,\d t + \sqrt{\eta/B}\;\Sigma(\theta)^{1/2}\,\d W_t ,$$
with time $t = k\eta$ after $k$ steps. The <em>temperature</em> is $\eta/B$: doubling the batch size has roughly the same effect on the noise as halving the step — the origin of the "linear scaling rule" $\eta\propto B$ for large-batch training (Goyal et al., 2017).</p>
<p>Near a minimum where $F\approx F^\star+\tfrac12(\theta-\theta^\star)^\top H(\theta-\theta^\star)$, the stationary covariance $C$ of the iterates solves the Lyapunov equation $HC + CH = \frac{\eta}{B}\Sigma$. Two consequences:</p>
<ul>
<li>When $\Sigma\approx s^2H$ (as for well-specified least squares or log-likelihoods near the optimum), $C = \frac{\eta s^2}{2B}I$: the iterates fluctuate isotropically, and the expected excess loss is $\tfrac12\tr(HC) = \frac{\eta s^2}{4B}\tr H$ — <em>larger at sharper minima</em>. At fixed temperature, SGD therefore spends less time in sharp basins and is biased toward flat ones, which correlates with better generalization (Keskar et al., 2017, on large-batch training reaching sharper minima).</li>
<li>Even without noise, a finite step changes the trajectory. Backward error analysis shows full-batch GD with step $\eta$ follows, to second order in $\eta$, the gradient flow of the modified loss
$$\tilde F(\theta) = F(\theta) + \frac{\eta}{4}\|\nabla F(\theta)\|^2$$
(Barrett & Dherin, 2021) — an implicit penalty on gradient norm. For SGD there is an additional penalty growing with the minibatch gradient variance, i.e. with $\eta/B$ (Smith et al., 2021).</li>
</ul>
<details class="proof"><summary>Proof sketch of the modified loss</summary>
<p>Seek a flow $\dot\theta = -\nabla F(\theta)+\eta\,v(\theta)$ whose solution after time $\eta$ agrees with one GD step $\theta-\eta\nabla F(\theta)$ up to $O(\eta^3)$. Taylor-expanding the flow, $\theta(\eta) = \theta + \eta(-\nabla F + \eta v) + \frac{\eta^2}{2}\nabla^2F\,\nabla F + O(\eta^3)$, since $\ddot\theta = -\nabla^2F\,\dot\theta = \nabla^2F\nabla F + O(\eta)$. Matching requires $v = -\tfrac12\nabla^2F\,\nabla F = -\tfrac14\nabla\|\nabla F\|^2$, so the flow is $-\nabla\big(F+\tfrac\eta4\|\nabla F\|^2\big)$.</p>
</details>
<div class="warn" data-name="Edge of stability">
<p>Classical theory requires $\eta\lt 2/\lambda_{\max}(\nabla^2F)$ for stability. In practice full-batch training of networks drives the sharpness $\lambda_{\max}$ <em>up</em> until it hovers at about $2/\eta$, with the loss decreasing non-monotonically yet still making progress (Cohen et al., 2021). Step size, curvature and generalization are coupled in ways the convex picture does not capture.</p>
</div>`,
      },
      {
        title: 'Momentum and acceleration',
        html: R`
<p>On an ill-conditioned problem plain GD zigzags across the steep directions while creeping along the shallow ones. <strong>Heavy-ball momentum</strong> (Polyak, 1964) adds a fraction of the previous step:
$$v_{t+1} = \beta v_t - \eta\nabla F(\theta_t),\qquad \theta_{t+1} = \theta_t + v_{t+1}.$$
<strong>Nesterov's accelerated gradient</strong> evaluates the gradient at the look-ahead point: $v_{t+1} = \beta v_t - \eta\nabla F(\theta_t+\beta v_t)$.</p>
<div class="thm" data-name="Heavy ball on a quadratic">
<p>Let $F(\theta) = \tfrac12\theta^\top H\theta$ with eigenvalues in $[\mu, L]$, $\kappa = L/\mu$. With
$$\eta = \frac{4}{(\sqrt L+\sqrt\mu)^2},\qquad \beta = \Big(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\Big)^2,$$
every eigencomponent contracts at rate $\sqrt\beta = \frac{\sqrt\kappa-1}{\sqrt\kappa+1}\approx 1 - 2/\sqrt\kappa$ per step (up to a factor polynomial in $t$), compared with $\frac{\kappa-1}{\kappa+1}\approx1-2/\kappa$ for optimally tuned GD.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>In the eigenbasis the coordinates decouple. A coordinate $x$ with eigenvalue $\lambda$ obeys $x_{t+1} = (1+\beta-\eta\lambda)x_t - \beta x_{t-1}$, with characteristic polynomial $z^2-(1+\beta-\eta\lambda)z+\beta$. If the discriminant is $\le 0$, i.e. $|1+\beta-\eta\lambda|\le2\sqrt\beta$, the roots are complex conjugates (or a double root) with product $\beta$, so both have modulus $\sqrt\beta$ and $|x_t| = O(t\,\beta^{t/2})$. The condition is equivalent to $(1-\sqrt\beta)^2\le\eta\lambda\le(1+\sqrt\beta)^2$. Requiring it at both ends, $\eta\mu = (1-\sqrt\beta)^2$ and $\eta L = (1+\sqrt\beta)^2$; taking square roots, adding and subtracting gives $\sqrt\eta(\sqrt L+\sqrt\mu) = 2$ and $\sqrt\beta = (\sqrt L-\sqrt\mu)/(\sqrt L+\sqrt\mu)$, as claimed. For GD, the per-step factor $\max_\lambda|1-\eta\lambda|$ is minimized at $\eta = 2/(L+\mu)$ with value $(\kappa-1)/(\kappa+1)$.</p>
</details>
<p>For $\kappa = 10^4$, GD needs about $\kappa/2 = 5000$ steps per $e$-fold reduction, momentum about $\sqrt\kappa/2 = 50$. For general $L$-smooth convex functions Nesterov's method achieves $F(\theta_t)-F^\star\le 2L\|\theta_0-\theta^\star\|^2/(t+1)^2$, versus $O(1/t)$ for GD, and this $1/t^2$ rate is optimal among first-order methods (Nemirovski & Yudin; Nesterov, 2004). Heavy-ball momentum, by contrast, can fail to accelerate (or even converge) on some non-quadratic strongly convex functions (Lessard, Recht & Packard, 2016).</p>
<div class="note">
<p>With stochastic gradients, momentum's acceleration is largely lost once noise dominates: $v_t$ is an exponential moving average of gradients with effective step $\eta/(1-\beta)$, so momentum mostly acts as an averaging filter and a rescaling of the learning rate. Its real value in deep learning is damping oscillations across steep directions so that a larger step is stable.</p>
</div>`,
        lab: 'sgd-optimizers-race',
      },
      {
        title: 'Adaptive methods: AdaGrad, RMSProp and Adam',
        html: R`
<p>A single global step size is a poor fit when coordinates have very different curvature or gradient scales (embeddings of rare words, the last layer versus the first). Adaptive methods precondition with a diagonal matrix built from past gradients. All operations below are elementwise, with a small $\epsilon$ (e.g. $10^{-8}$) for numerical safety.</p>
<div class="cols">
<div><h4>AdaGrad</h4>$$r_t = r_{t-1}+g_t^2,\quad \theta_{t+1} = \theta_t - \frac{\eta\,g_t}{\sqrt{r_t}+\epsilon}$$<p>Duchi, Hazan & Singer (2011). Accumulated squares: steps shrink fastest on frequently-updated coordinates. Good regret bounds for sparse convex problems; the step decays too aggressively for deep nets.</p></div>
<div><h4>RMSProp</h4>$$s_t = \rho s_{t-1}+(1-\rho)g_t^2,\quad \theta_{t+1} = \theta_t-\frac{\eta\,g_t}{\sqrt{s_t}+\epsilon}$$<p>Tieleman & Hinton (2012). An exponential moving average forgets old gradients, so the effective step does not vanish.</p></div>
<div><h4>Adam</h4>$$\begin{aligned}m_t &= \beta_1m_{t-1}+(1-\beta_1)g_t\\ v_t &= \beta_2v_{t-1}+(1-\beta_2)g_t^2\\ \theta_{t} &= \theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}\end{aligned}$$<p>Kingma & Ba (2015): momentum plus RMSProp, with bias correction $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$. Defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$.</p></div>
</div>
<div class="thm" data-kind="Proposition" data-name="Why Adam divides by $1-\beta^t$">
<p>With $m_0 = 0$, $m_t = (1-\beta_1)\sum_{s=1}^t\beta_1^{t-s}g_s$. If the gradients have a common mean $\E g_s = \bar g$, then $\E m_t = (1-\beta_1^t)\,\bar g$, so $\hat m_t = m_t/(1-\beta_1^t)$ is unbiased for $\bar g$. Likewise $\hat v_t$ is unbiased for $\E[g^2]$ when the second moments are constant.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Unroll the recursion from $m_0 = 0$ to get the sum. Then $\E m_t = (1-\beta_1)\bar g\sum_{s=1}^t\beta_1^{t-s} = (1-\beta_1)\bar g\,\frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\bar g$. The same computation applies to $v_t$ with $g_s^2$.</p>
</details>
<p>Without the correction, the zero initialization biases both averages toward zero early on; the bias in $v_t$ is the dangerous one, since $\beta_2 = 0.999$ makes $v_t$ about $1000$ times too small at $t = 1$, and dividing by $\sqrt{v_t}$ would produce huge first steps. With it, $|\Delta\theta|$ is roughly bounded by $\eta$ per coordinate, and Adam is invariant to rescaling the gradient by a constant — useful in practice, but it also means Adam behaves like a smoothed <em>sign</em> descent: tiny but consistent gradients get full-sized steps.</p>
<div class="warn" data-name="Adam is not a convergent method in general">
<p>Reddi, Kale & Kumar (2018) built simple convex online problems on which Adam converges to the worst point, because the moving average $v_t$ can shrink and enlarge the effective step exactly when a rare large gradient matters. AMSGrad (use $\max_{s\le t}\hat v_s$) restores convergence guarantees. Separately, adding an L2 penalty to the loss is <em>not</em> weight decay under Adam, since the penalty's gradient is rescaled by $1/\sqrt{\hat v_t}$; decoupled weight decay (AdamW) applies the shrinkage directly — see <a href="#/t/regularization-deep">regularization</a>.</p>
</div>`,
      },
      {
        title: 'Learning-rate schedules in practice',
        html: R`
<p>Theory prescribes $\eta_t\propto1/t$ (strongly convex) or $1/\sqrt t$ (convex); practice uses schedules shaped by the two-phase picture above — make fast progress with a large step, then reduce the step to shrink the noise ball.</p>
<table>
<thead><tr><th>Schedule</th><th>Form</th><th>Comment</th></tr></thead>
<tbody>
<tr><td>Step decay</td><td>$\eta_t = \eta_0\gamma^{\lfloor t/T_0\rfloor}$</td><td>Classic for CNNs; loss drops visibly at each decay</td></tr>
<tr><td>Cosine</td><td>$\eta_t = \eta_{\min}+\tfrac12(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$</td><td>Smooth anneal to $\eta_{\min}$ (Loshchilov & Hutter, 2017)</td></tr>
<tr><td>Linear warmup</td><td>$\eta_t = \eta_0\,t/T_w$ for $t\le T_w$</td><td>Avoids early instability, especially with Adam and large batches</td></tr>
<tr><td>Inverse square root</td><td>$\eta_t\propto1/\sqrt t$ after warmup</td><td>Original transformer schedule</td></tr>
<tr><td>Iterate averaging / EMA</td><td>$\bar\theta_t = \alpha\bar\theta_{t-1}+(1-\alpha)\theta_t$</td><td>Reduces noise without reducing the step; cheap and effective</td></tr>
</tbody>
</table>
<div class="warn" data-name="Pitfalls">
<p>The best learning rate depends on batch size, optimizer, initialization scale and normalization; tune it on a log grid first, before anything else. A loss that falls and then explodes signals a step past the stability limit $2/\lambda_{\max}$; a loss that plateaus high signals a step too small or a noise floor. When comparing optimizers, tune each one's step separately — Adam and SGD with momentum have different natural scales, and "Adam beats SGD" claims often reflect unequal tuning.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In the strongly convex bound, what happens to the noise floor $\eta\sigma^2/\mu$ if you double the minibatch size, and if you halve the step? Which is cheaper per unit of progress?`,
        a: R`Doubling $B$ halves $\sigma^2$ and hence the floor, at twice the cost per step; halving $\eta$ also halves the floor but slows the linear phase $(1-\eta\mu)^t$ by a factor 2. Both lower the "temperature" $\eta/B$ by 2; in the noise-dominated phase they are roughly equivalent per gradient evaluation, while larger batches parallelize better.` },
      { q: R`Why do the step-size conditions $\sum\eta_t=\infty$ and $\sum\eta_t^2\lt\infty$ both matter?`,
        a: R`If $\sum\eta_t\lt\infty$ the iterates can move only a bounded total distance and may stop short of the optimum. If $\sum\eta_t^2=\infty$, the accumulated noise $\sum\eta_t\varepsilon_t$ (variance $\propto\sum\eta_t^2$) does not converge, so the iterates keep fluctuating. $\eta_t\propto1/t$ satisfies both.` },
      { q: R`Show that without bias correction Adam's first step has magnitude about $\eta(1-\beta_1)/\sqrt{1-\beta_2}$ per coordinate, and evaluate it for the defaults.`,
        a: R`At $t=1$, $m_1 = (1-\beta_1)g_1$ and $v_1 = (1-\beta_2)g_1^2$, so $m_1/\sqrt{v_1} = \sgn(g_1)(1-\beta_1)/\sqrt{1-\beta_2}$. With $\beta_1 = 0.9$, $\beta_2 = 0.999$: $0.1/\sqrt{0.001}\approx3.16$, a step about three times $\eta$. With correction, $\hat m_1/\sqrt{\hat v_1} = \sgn(g_1)$, a step of exactly $\eta$.` },
      { q: R`For a quadratic with $\kappa = 100$, compare the per-step contraction of optimally tuned GD and heavy-ball momentum, and the number of steps to reduce the error by $10^{-6}$.`,
        a: R`GD: $(\kappa-1)/(\kappa+1) = 99/101\approx0.980$, needing $\ln(10^6)/(-\ln 0.980)\approx 690$ steps. Heavy ball: $(\sqrt\kappa-1)/(\sqrt\kappa+1) = 9/11\approx0.818$, needing about $\ln(10^6)/0.2007\approx 69$ steps — a factor $\approx\sqrt\kappa$ fewer.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'initialization-normalization',
    domain: 'deep',
    order: 50,
    title: 'Initialization, Normalization & Training Dynamics',
    subtitle: 'Keeping signals and gradients at the right scale through depth',
    level: 2,
    sources: { other: 'LeCun et al. (1998); Glorot & Bengio (2010); He, Zhang, Ren & Sun (2015, 2016); Saxe, McClelland & Ganguli (2014); Poole et al. (2016); Schoenholz et al. (2017); Ioffe & Szegedy (2015); Ba, Kiros & Hinton (2016); Santurkar et al. (2018); De & Smith (2020); Goodfellow, Bengio & Courville (2016) §8.4, §8.7.1; Prince, Understanding Deep Learning (2023) Ch. 7, 11' },
    prereqs: ['backpropagation'],
    related: ['sgd-optimizers', 'deep-learning-theory', 'expectation', 'clt'],
    tags: ['He initialization', 'Glorot initialization', 'batch normalization', 'layer normalization', 'residual connections', 'signal propagation'],
    summary: R`A deep network at initialization is a random function, and whether it can be trained depends on how that random function propagates signals forward and gradients backward. Choosing the weight variance so that the second moment of pre-activations is preserved layer to layer gives LeCun/Glorot initialization ($\sigma_w^2 = 1/n_{\text{in}}$, or $2/(n_{\text{in}}+n_{\text{out}})$) for tanh-like units and He initialization ($2/n_{\text{in}}$) for ReLU. Normalization layers (batch norm, layer norm) enforce the right scale by construction and make the loss invariant to weight scale, which turns weight norm into an implicit learning-rate schedule. Residual connections add an identity path to every Jacobian, so gradients reach early layers however deep the network is.`,
    survey: {
      task: 'Make deep networks trainable: stable forward signals and backward gradients',
      assumptions: R`Independent zero-mean initial weights; wide layers (for the mean-field calculations); minibatches large enough for batch statistics (BN)`,
      output: R`Weight variances $\sigma_w^2$ per layer; normalized and residual architectures`,
      strengths: 'Cheap, principled fixes that enable training at depth 100+ and larger learning rates.',
      weaknesses: 'Calculations are at initialization only; BN couples examples in a batch and behaves differently at train and test time.',
      complexity: R`Initialization free; normalization $O(\text{activations})$ per layer`,
    },
    sections: [
      {
        title: 'Why the starting point matters',
        html: R`
<p>Gradient-based training starts from $\theta_0$ and only ever explores nearby. Two things can go wrong at the very first step.</p>
<div class="thm" data-kind="Proposition" data-name="Symmetric initializations stay symmetric">
<p>Suppose that in some hidden layer all units have identical incoming weights and biases and identical outgoing weights (for example, everything initialized to the same constant). Then, under gradient descent or SGD with any data, those units remain identical forever: the layer behaves like a single unit.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Swapping two such units $j$ and $k$ (permuting rows of $W_\ell$, entries of $b_\ell$ and columns of $W_{\ell+1}$) leaves $\theta$ unchanged, and the network function is invariant to this permutation for all parameter values. Hence the loss is invariant too, and its gradient is equivariant: the gradient with respect to unit $j$'s parameters equals that with respect to unit $k$'s. An identical update preserves the equality; induct over steps.</p>
</details>
<p>Random initialization breaks the symmetry. The remaining question is the <em>scale</em>. Too small, and the signal shrinks layer by layer until the output is constant and gradients vanish (and tanh units sit in their linear regime, so the network is effectively linear). Too large, and pre-activations blow up: tanh and sigmoid saturate (derivatives near zero) while ReLU activations explode. The backpropagation analysis showed the per-layer gradient gain $\chi = \sigma_w^2\E[\phi'(z)^2]$; initialization schemes choose $\sigma_w^2$ to make the relevant gains equal to one.</p>`,
      },
      {
        title: 'Variance-preserving initialization: LeCun, Glorot and He',
        html: R`
<p>Consider one layer $z = Wh$ with $W\in\R^{n_{\text{out}}\times n_{\text{in}}}$ (biases initialized at zero), where the $W_{ij}$ are IID with mean $0$ and variance $\sigma_w^2$, independent of the incoming activations $h$.</p>
<div class="thm" data-kind="Proposition" data-name="Forward second-moment propagation">
<p>Each pre-activation has mean zero and
$$\Var(z_i) = n_{\text{in}}\,\sigma_w^2\;\E[h_j^2]$$
(assuming the $h_j$ have a common second moment). If moreover $h = \phi(z')$ for a pre-activation $z'$ symmetric about $0$ and $\phi = \mathrm{ReLU}$, then $\E[h_j^2] = \tfrac12\Var(z'_j)$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$z_i = \sum_jW_{ij}h_j$ and $\E[W_{ij}h_j] = \E[W_{ij}]\E[h_j] = 0$. For the variance, $\E[z_i^2] = \sum_{j,k}\E[W_{ij}W_{ik}]\E[h_jh_k] = \sum_j\sigma_w^2\E[h_j^2]$, since $\E[W_{ij}W_{ik}] = 0$ for $j\ne k$. Note it is the second moment $\E[h^2]$ that appears, not $\Var(h)$; for ReLU outputs, which have positive mean, the difference matters. For symmetric $z'$, $\E[\mathrm{ReLU}(z')^2] = \E[z'^2\ind(z'\gt0)] = \tfrac12\E[z'^2]$.</p>
</details>
<p>Chaining layers: for ReLU networks, $\Var(z_\ell) = \tfrac12 n_{\ell-1}\sigma_\ell^2\Var(z_{\ell-1})$, so the pre-activation scale is preserved exactly when
$$\sigma_\ell^2 = \frac{2}{n_{\ell-1}}\qquad\text{(He et al., 2015).}$$
For tanh (or any activation with $\phi(0)=0$, $\phi'(0)=1$) near the origin $\phi(z)\approx z$, giving $\sigma_\ell^2 = 1/n_{\ell-1}$ (LeCun et al., 1998).</p>
<p>The <strong>backward</strong> pass gives a second condition. From $\delta_{\ell-1} = \phi'(z_{\ell-1})\odot W_\ell^\top\delta_\ell$, the same computation (treating the backward weights as independent of the forward pass, which is accurate for wide layers) gives $\E[\delta_{\ell-1,j}^2] = n_\ell\,\sigma_\ell^2\,\E[\phi'(z)^2]\,\E[\delta_{\ell,i}^2]$. Preserving gradient scale requires $\sigma_\ell^2 = 1/n_\ell$ (linear regime) or $2/n_\ell$ (ReLU). <strong>Glorot & Bengio (2010)</strong> compromise between $1/n_{\text{in}}$ and $1/n_{\text{out}}$:
$$\sigma_w^2 = \frac{2}{n_{\text{in}}+n_{\text{out}}},\qquad\text{e.g. } W_{ij}\sim\mathrm{Unif}\Big[-\sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}},\ \sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}}\Big],$$
using $\Var(\mathrm{Unif}[-a,a]) = a^2/3$. He et al. observed that for a deep net it hardly matters which of the forward or backward condition is imposed: if every layer satisfies one of them, the other is off by the product $\prod_\ell n_{\ell-1}/n_\ell = n_0/n_L$, a constant independent of depth.</p>
<div class="ex" data-name="What goes wrong with the 'obvious' choice">
<p>Initializing a 30-layer ReLU network with $\sigma_w^2 = 1/n_{\text{in}}$ (correct for tanh) halves the second moment at every layer: after 30 layers the activations are $2^{-15}\approx 3\times10^{-5}$ times smaller than the inputs, and the gradients shrink similarly. With $\sigma_w^2 = 4/n_{\text{in}}$ they grow by $2^{15}$ instead. Exactly this observation motivated He et al.'s derivation for very deep rectifier networks.</p>
</div>`,
      },
      {
        title: 'Beyond variance: order, chaos and dynamical isometry',
        html: R`
<p>Preserving the length of a signal is necessary but not sufficient: the network must also keep <em>different inputs distinguishable</em>. Mean-field analyses of wide random networks (Poole et al., 2016; Schoenholz et al., 2017) track, for two inputs, the second moment $q_\ell$ of their pre-activations and their correlation $c_\ell$. In the infinite-width limit these obey deterministic maps; for weights $\N(0,\sigma_w^2/n)$ and biases $\N(0,\sigma_b^2)$,
$$q_{\ell} = \sigma_w^2\,\E_{z\sim\N(0,q_{\ell-1})}[\phi(z)^2]+\sigma_b^2,\qquad c_\ell = \frac{\sigma_w^2\,\E[\phi(u)\phi(v)]+\sigma_b^2}{q^\star},$$
with $(u,v)$ jointly Gaussian, variances $q^\star$ (the fixed point of the $q$-map) and correlation $c_{\ell-1}$. The point $c = 1$ is always a fixed point; its stability is governed by the slope
$$\chi_1 = \sigma_w^2\,\E_{z\sim\N(0,q^\star)}\big[\phi'(z)^2\big],$$
the same quantity that multiplies squared gradients per layer.</p>
<div class="cols">
<div><h4>Ordered ($\chi_1\lt1$)</h4><p>All inputs converge to the same representation exponentially fast; the output barely depends on the input and gradients vanish.</p></div>
<div><h4>Chaotic ($\chi_1\gt1$)</h4><p>Nearby inputs decorrelate exponentially fast; the function becomes extremely rough and gradients explode.</p></div>
<div><h4>Critical ($\chi_1 = 1$)</h4><p>The "edge of chaos": correlation and gradient signals propagate over depths that diverge as $\chi_1\to1$, so very deep nets train.</p></div>
</div>
<p>For ReLU with $\sigma_b = 0$, He initialization is exactly critical ($\chi_1 = 2\cdot\tfrac12 = 1$), yet correlations still drift to $1$, only polynomially fast rather than exponentially; very deep plain ReLU stacks therefore map all inputs to nearly parallel vectors (rank collapse). The stronger requirement of <strong>dynamical isometry</strong> asks that <em>all</em> singular values of the input–output Jacobian be near 1, not just their mean square. For deep linear networks, orthogonal weight matrices achieve it exactly and give depth-independent learning speed (Saxe, McClelland & Ganguli, 2014); for tanh nets at criticality, orthogonal (not Gaussian) weights achieve it approximately, while ReLU nets cannot (Pennington, Schoenholz & Ganguli, 2017).</p>
<p>The lab propagates 128 random inputs through a width-64 network at initialization. Note the mean cosine between pairs of inputs at the last layer: they start nearly orthogonal and, in the ordered phase or for deep ReLU stacks, end nearly parallel.</p>`,
        lab: 'initialization-normalization-signal',
      },
      {
        title: 'Batch normalization',
        html: R`
<p>Instead of hoping the scale is right, enforce it. <strong>Batch normalization</strong> (Ioffe & Szegedy, 2015) standardizes each pre-activation over the minibatch $\mathcal B$ and then restores a learnable affine map:</p>
<div class="algo" data-name="BatchNorm for one unit">
<ol>
<li>$\mu_{\mathcal B} = \frac1B\sum_{i\in\mathcal B}z_i$, $\quad\sigma^2_{\mathcal B} = \frac1B\sum_{i\in\mathcal B}(z_i-\mu_{\mathcal B})^2$.</li>
<li>$\hat z_i = (z_i-\mu_{\mathcal B})/\sqrt{\sigma^2_{\mathcal B}+\epsilon}$, $\quad y_i = \gamma\hat z_i+\beta$.</li>
<li>Backpropagate through $\mu_{\mathcal B}$ and $\sigma^2_{\mathcal B}$ too — they depend on every example in the batch.</li>
<li>At test time replace batch statistics with running averages of $\mu_{\mathcal B}$ and $\sigma^2_{\mathcal B}$ collected during training.</li>
</ol>
</div>
<div class="thm" data-kind="Proposition" data-name="Scale invariance and its consequences">
<p>Let a weight vector $w$ feed a batch-normalized unit, so that (ignoring $\epsilon$) the loss satisfies $L(cw) = L(w)$ for all $c\gt0$. Then
$$\langle\nabla L(w), w\rangle = 0\qquad\text{and}\qquad\nabla L(cw) = \tfrac1c\nabla L(w).$$
Under gradient descent, $\|w_{t+1}\|^2 = \|w_t\|^2+\eta^2\|\nabla L(w_t)\|^2$, so the weight norm never decreases.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>BN removes any positive rescaling: $(cz - c\mu)/(c\sigma) = (z-\mu)/\sigma$. Differentiating $L(cw) = L(w)$ in $c$ at $c=1$ gives $\langle\nabla L(w),w\rangle = 0$; differentiating in $w$ gives $c\nabla L(cw) = \nabla L(w)$. Finally $\|w-\eta\nabla L\|^2 = \|w\|^2-2\eta\langle w,\nabla L\rangle+\eta^2\|\nabla L\|^2$ and the cross term is zero.</p>
</details>
<p>Because only the direction $w/\|w\|$ matters, and the gradient scales like $1/\|w\|$, the angular step is about $\eta\|\nabla L(w/\|w\|)\|/\|w\|^2$: the <strong>effective learning rate</strong> is $\eta/\|w\|^2$. Growing norms automatically anneal it; weight decay shrinks norms and keeps the effective step up. With BN, weight decay therefore acts mainly as a learning-rate controller rather than as a capacity penalty (van Laarhoven, 2017; Zhang et al., 2019).</p>
<p><em>Why</em> BN helps is still debated. The original motivation, reducing "internal covariate shift" (changes in each layer's input distribution during training), was challenged by Santurkar et al. (2018), who found that BN's benefit survives deliberately injected distribution shift and instead makes the loss and its gradients smoother, permitting larger learning rates. The batch-statistics noise also acts as a regularizer.</p>
<div class="warn" data-name="Batch dependence">
<p>BN makes each example's output depend on the others in its batch. Small batches give noisy statistics and degrade accuracy; non-IID batches (sorted data, a few long sequences, a single sample at inference) break the train/test correspondence; and forgetting to switch to running statistics at evaluation time is a classic bug.</p>
</div>`,
      },
      {
        title: 'Layer normalization and other variants',
        html: R`
<p>Normalization schemes differ only in which axes the statistics are computed over.</p>
<table>
<thead><tr><th>Method</th><th>Statistics over</th><th>Batch-dependent?</th><th>Typical use</th></tr></thead>
<tbody>
<tr><td>BatchNorm</td><td>examples in the batch (per unit/channel)</td><td>yes</td><td>CNNs with large batches</td></tr>
<tr><td>LayerNorm (Ba, Kiros & Hinton, 2016)</td><td>units of one example</td><td>no</td><td>Transformers, RNNs</td></tr>
<tr><td>RMSNorm (Zhang & Sennrich, 2019)</td><td>units of one example, no centring: $z/\sqrt{\tfrac1n\sum_k z_k^2}$</td><td>no</td><td>Large language models</td></tr>
<tr><td>GroupNorm (Wu & He, 2018)</td><td>groups of channels of one example</td><td>no</td><td>Vision with small batches</td></tr>
<tr><td>Weight normalization (Salimans & Kingma, 2016)</td><td>reparametrize $w = g\,v/\|v\|$</td><td>no</td><td>Decouples direction and scale</td></tr>
</tbody>
</table>
<p>Layer normalization computes $\hat z_k = (z_k-\bar z)/\sqrt{s^2+\epsilon}$ with $\bar z, s^2$ the mean and variance over the $n$ units of a single example, then applies per-unit $\gamma_k, \beta_k$. It behaves identically at training and test time and is insensitive to batch composition, which is why sequence models use it. It enjoys the same scale invariance as BN with respect to the whole incoming weight matrix.</p>
<div class="note" data-name="Where to put the normalization">
<p>In a residual block, "post-norm" normalizes after adding the skip connection, $h\leftarrow\mathrm{LN}(h+F(h))$; "pre-norm" normalizes only the branch input, $h\leftarrow h+F(\mathrm{LN}(h))$, keeping a clean identity path. Pre-norm transformers have well-behaved gradients at initialization and train without the learning-rate warmup that post-norm ones need (Xiong et al., 2020).</p>
</div>`,
      },
      {
        title: 'Residual connections',
        html: R`
<p>A residual block computes $h_{\ell+1} = h_\ell+F_\ell(h_\ell)$, where the branch $F_\ell$ is a small network (He, Zhang, Ren & Sun, 2016). The Jacobian of the whole stack is
$$\frac{\partial h_L}{\partial h_\ell} = \prod_{k=\ell}^{L-1}\Big(I+\frac{\partial F_k}{\partial h_k}\Big),$$
which, expanded, is a sum over all $2^{L-\ell}$ subsets of branches and always contains the identity term. Gradients therefore reach early layers without passing through every nonlinearity; residual networks behave in part like ensembles of shallower paths (Veit, Wilber & Belongie, 2016).</p>
<div class="thm" data-kind="Proposition" data-name="Variance growth in residual stacks">
<p>Suppose at initialization each branch output is uncorrelated with its input, $\E[h_\ell^\top F_\ell(h_\ell)] = 0$ (true when the branch's last weight matrix has independent zero-mean entries), and $\E\|F_\ell(h_\ell)\|^2 = \alpha\,\E\|h_\ell\|^2$. Then $\E\|h_L\|^2 = (1+\alpha)^{L}\E\|h_0\|^2$. If instead the branch normalizes its input so that $\E\|F_\ell(h_\ell)\|^2 = \beta$ is constant, then $\E\|h_L\|^2 = \E\|h_0\|^2+L\beta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\E\|h_{\ell+1}\|^2 = \E\|h_\ell\|^2+2\E[h_\ell^\top F_\ell(h_\ell)]+\E\|F_\ell(h_\ell)\|^2$. The cross term vanishes by assumption, leaving $(1+\alpha)\E\|h_\ell\|^2$ in the first case and $\E\|h_\ell\|^2+\beta$ in the second; induct on $\ell$.</p>
</details>
<p>An unnormalized residual stack with He-initialized branches has $\alpha\approx1$ and its activations double every block — exponential growth with depth. Remedies scale the branches down: initialize the last layer (or the last BN's $\gamma$) of every branch at zero so each block starts as the identity (Goyal et al., 2017; Fixup, Zhang, Dauphin & Ma, 2019; SkipInit, De & Smith, 2020), or multiply branches by $1/\sqrt L$. With normalization inside the branch (pre-activation ResNets, pre-norm transformers) the variance grows only linearly, so block $\ell$ contributes a fraction $\approx1/\ell$ of the signal: at initialization the deep network is effectively shallow, and depth is "switched on" gradually by training (De & Smith, 2020). Toggle "residual" in the lab with and without BatchNorm to see both regimes.</p>
<div class="warn" data-name="Initialization analyses describe step zero">
<p>Everything here concerns the random network before training. Normalized and residual architectures keep the forward and backward signals healthy throughout training in practice, but the mean-field calculations do not prove it; the dynamics after the first few thousand steps — feature learning, sharpness growth, the edge of stability — require different tools (<a href="#/t/deep-learning-theory">theory of deep learning</a>).</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`Derive the He initialization variance for a layer with $n_{\text{in}} = 512$ ReLU inputs, and the Glorot uniform bound for a layer with $n_{\text{in}} = 512$, $n_{\text{out}} = 256$.`,
        a: R`He: $\sigma_w^2 = 2/512$, so $\sigma_w\approx0.0625$. Glorot uniform: $a = \sqrt{6/(512+256)} = \sqrt{0.0078125}\approx0.0884$, i.e. $W_{ij}\sim\mathrm{Unif}[-0.088, 0.088]$, variance $a^2/3 = 2/768$.` },
      { q: R`Why is it $\E[h^2]$ and not $\Var(h)$ that enters the variance of the next pre-activation, and why does the distinction matter for ReLU but not tanh?`,
        a: R`$\Var(\sum_j W_{ij}h_j) = \sum_j\Var(W_{ij})\E[h_j^2]$ because $W_{ij}$ has mean zero and is independent of $h_j$; the mean of $h_j$ contributes through $\E[h_j^2] = \Var(h_j)+(\E h_j)^2$. For tanh with symmetric input, $\E h = 0$ so the two coincide. ReLU outputs have positive mean ($\E[\mathrm{ReLU}(z)] = \sigma/\sqrt{2\pi}$ for $z\sim\N(0,\sigma^2)$), so using the variance would understate the scale.` },
      { q: R`A weight vector feeding a BatchNorm unit has its norm doubled at some step. What happens to the network function, to the gradient with respect to that vector, and to the effective learning rate?`,
        a: R`The function is unchanged (BN removes the scale). The gradient is halved, $\nabla L(2w) = \tfrac12\nabla L(w)$, and it remains orthogonal to $w$. The angular change per step scales like $\eta/\|w\|^2$, so the effective learning rate drops by a factor 4.` },
      { q: R`In a 50-block unnormalized residual network with He-initialized branches ($\alpha\approx1$), how much larger is the output scale than the input scale? What one-line change fixes it?`,
        a: R`$\E\|h_{50}\|^2\approx2^{50}\E\|h_0\|^2$, so the RMS grows by about $2^{25}\approx3\times10^7$. Initializing the last layer (or BN scale $\gamma$) of each branch at zero makes every block the identity at initialization (or scale each branch by $1/\sqrt{50}$).` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'regularization-deep',
    domain: 'deep',
    order: 60,
    title: 'Regularizing Deep Networks: Weight Decay, Dropout & Double Descent',
    subtitle: 'Explicit and implicit controls on overparameterized models, and the risk curve beyond interpolation',
    level: 3,
    sources: { other: 'Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov (2014); Wager, Wang & Liang (2013); Gal & Ghahramani (2016); Loshchilov & Hutter (2019); Bishop (1995); Ali, Kolter & Tibshirani (2019); Zhang et al. (2017); Belkin, Hsu, Ma & Mandal (2019); Hastie, Montanari, Rosset & Tibshirani (2022); Nakkiran et al. (2020); Goodfellow, Bengio & Courville (2016) Ch. 7; Prince, Understanding Deep Learning (2023) Ch. 8–9' },
    prereqs: ['ridge-regression', 'bias-variance'],
    related: ['cross-validation', 'model-selection', 'bagging-random-forests', 'bayesian-inference', 'deep-learning-theory', 'sgd-optimizers'],
    tags: ['weight decay', 'AdamW', 'dropout', 'early stopping', 'data augmentation', 'double descent', 'interpolation'],
    summary: R`Deep networks can fit random labels, so their good generalization must come from somewhere other than limited capacity. The explicit regularizers have clean statistical meanings in simple models: weight decay is ridge regression (a Gaussian prior), early stopping of gradient descent on least squares is a spectral filter within a factor of two of ridge, dropout on the inputs of a linear model is ridge with a data-dependent diagonal penalty, and training on noisy inputs is Tikhonov regularization. In networks these effects combine with the implicit biases of the optimizer. And the classical U-shaped risk curve is only half the story: past the interpolation threshold, risk can fall again — double descent — with the minimum-norm interpolant as the protagonist.`,
    survey: {
      task: 'Control overfitting in models with more parameters than data',
      assumptions: R`A validation set to tune strength; for the clean equivalences, linear or least-squares models`,
      output: R`A trained network with better test risk; hyperparameters $\lambda$, dropout rate $p$, stopping time`,
      strengths: 'Cheap, composable, interpretable in linear models; dropout gives an approximate predictive distribution.',
      weaknesses: 'Effects interact with optimizer and normalization (L2 is not weight decay under Adam or BN); theory beyond linear models is partial.',
      complexity: R`Negligible overhead; tuning needs repeated training runs`,
    },
    sections: [
      {
        title: 'The puzzle: capacity without overfitting',
        html: R`
<p>Zhang et al. (2017) trained standard image-classification networks on the same images with <em>randomly permuted labels</em>. The networks reached zero training error — they can memorize arbitrary labelings — yet on real labels the same architectures generalize well. So uniform-convergence bounds based on the size of the function class (<a href="#/t/vc-theory">VC dimension</a>, which for ReLU networks scales roughly like parameters times layers) cannot explain generalization: the class is rich enough to shatter the training set. What matters is <em>which</em> of the many interpolating solutions training selects.</p>
<p>The selection is shaped by explicit regularizers (penalties, dropout, augmentation) and implicit ones (the initialization, the optimizer and its noise, early stopping, architecture). The cleanest way to understand each is in a model where it can be computed exactly; that is the strategy of this page. The classical <a href="#/t/bias-variance">bias–variance</a> picture then needs one amendment, double descent, to cover models that interpolate.</p>`,
      },
      {
        title: 'Weight decay: ridge, priors, and the AdamW correction',
        html: R`
<p>Adding $\frac\lambda2\|\theta\|^2$ to the loss is <a href="#/t/ridge-regression">ridge regression</a> in parameter space. If the loss is a summed negative log-likelihood, the penalized minimizer is the MAP estimate under the prior $\theta\sim\N(0,\lambda^{-1}I)$ (<a href="#/t/bayesian-inference">Bayesian inference</a>). With plain (S)GD the penalty's gradient $\lambda\theta$ turns each step into
$$\theta_{t+1} = (1-\eta\lambda)\,\theta_t-\eta\,g_t,$$
a multiplicative <em>decay</em> of the weights — hence the name. For SGD, "L2 penalty" and "weight decay" are the same algorithm.</p>
<p>For adaptive methods they are not. Adam with an L2 penalty feeds $g_t+\lambda\theta_t$ into the moment estimates, so the decay term is divided by $\sqrt{\hat v_t}$: coordinates with large historical gradients are barely regularized, and those with tiny gradients are decayed hard. <strong>Decoupled weight decay</strong> (AdamW, Loshchilov & Hutter, 2019) applies the shrinkage outside the preconditioner:
$$\theta_{t} = \theta_{t-1}-\eta\Big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\,\theta_{t-1}\Big).$$
This is the standard optimizer for transformers. The two choices define genuinely different estimators, with different good values of $\lambda$; the dropout lab below lets you compare them.</p>
<div class="note">
<p>In networks with normalization layers, most weights are scale-invariant (<a href="#/t/initialization-normalization">initialization & normalization</a>), so shrinking them does not change the function at all. There weight decay works by keeping weight norms small and hence the <em>effective learning rate</em> $\eta/\|w\|^2$ large — an optimization effect rather than a capacity penalty.</p>
</div>`,
      },
      {
        title: 'Early stopping is approximately ridge',
        html: R`
<p>Run gradient descent from $\theta_0 = 0$ on $F(\theta) = \frac1{2n}\|y-X\theta\|^2$. Let $G = X^\top X/n = \sum_i s_iv_iv_i^\top$ (eigenvalues $s_i\ge0$) and $r = X^\top y/n$.</p>
<div class="thm" data-name="Spectral filters of gradient descent and ridge">
<p>With step $\eta$, after $t$ steps
$$\theta_t = \sum_{i:s_i\gt0}\frac{1-(1-\eta s_i)^t}{s_i}\,(v_i^\top r)\,v_i,\qquad\text{while ridge gives}\qquad\hat\theta_\lambda = \sum_i\frac{1}{s_i+\lambda}\,(v_i^\top r)\,v_i .$$
Relative to least squares ($1/s_i$), GD applies the filter $\varphi_t(s) = 1-(1-\eta s)^t$ and ridge the filter $\psi_\lambda(s) = s/(s+\lambda)$. If $\eta s_{\max}\le1$ and $\lambda = 1/(\eta t)$, then for every eigenvalue
$$\psi_\lambda(s)\;\le\;\varphi_t(s)\;\le\;2\,\psi_\lambda(s).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The iteration is $\theta_{t+1} = \theta_t-\eta(G\theta_t-r) = (I-\eta G)\theta_t+\eta r$; in the eigenbasis each coordinate obeys $a_{t+1} = (1-\eta s)a_t+\eta c$ with $c = v^\top r$, solved by $a_t = \eta c\sum_{k\lt t}(1-\eta s)^k = c\,[1-(1-\eta s)^t]/s$. (Components with $s = 0$ have $c = 0$, since $r$ lies in the column space of $G$.) Ridge solves $(G+\lambda I)\theta = r$. For the comparison put $x = \eta st$, so $\psi_\lambda(s) = x/(1+x)$. Lower bound: $(1-\eta s)^t\le e^{-x}$, so $\varphi_t\ge1-e^{-x}\ge x/(1+x)$ because $e^{x}\ge1+x$. Upper bound: Bernoulli's inequality $(1-\eta s)^t\ge1-x$ gives $\varphi_t\le\min(1,x)$, and $\min(1,x)\le2x/(1+x)$ (check $x\le1$ and $x\ge1$ separately).</p>
</details>
<p>Both methods leave directions with large eigenvalue (well-determined by the data) essentially unshrunk and suppress directions with small eigenvalue (where the noise lives), with the transition near $s\approx\lambda = 1/(\eta t)$. Training time is an inverse regularization parameter. Ali, Kolter & Tibshirani (2019) sharpened this to a risk statement: for least squares, the excess risk of gradient flow at time $t$ is within a factor of about $1.69$ of ridge's at $\lambda = 1/t$, uniformly over $t$. Monitoring validation loss and stopping at its minimum is therefore a legitimate, nearly free way to choose a regularization level.</p>`,
        lab: 'regularization-deep-early-stopping',
      },
      {
        title: 'Dropout: ensembles, adaptive ridge, and approximate Bayes',
        html: R`
<p><strong>Dropout</strong> (Srivastava et al., 2014) multiplies each hidden unit, independently at every training step, by a random mask: with drop rate $p$, the unit is zeroed with probability $p$ and otherwise scaled by $1/(1-p)$ ("inverted dropout", so that the expected activation is unchanged). At test time the full network is used without masks.</p>
<p><em>Ensemble view.</em> Each mask selects one of $2^{N}$ subnetworks that share weights; dropout trains this exponentially large ensemble with one sampled member per step, and the test-time network approximates averaging their predictions. For a single linear layer followed by a softmax, the approximation is exact in a geometric sense:</p>
<div class="thm" data-kind="Proposition" data-name="Weight scaling equals the geometric-mean ensemble">
<p>Let $z(m) = W(m\odot x/(1-p))+b$ with mask distribution $\pi(m)$. The normalized geometric mean of the subnetwork predictions, $G_k\propto\prod_m\operatorname{softmax}(z(m))_k^{\pi(m)}$, equals $\operatorname{softmax}(Wx+b)$, the prediction of the unmasked network.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\log\prod_m\operatorname{softmax}(z(m))_k^{\pi(m)} = \sum_m\pi(m)\big[z_k(m)-\log\sum_je^{z_j(m)}\big]$. The second term does not depend on $k$ and disappears on normalization. The first is $\E_m[z_k(m)] = (W\,\E[m\odot x]/(1-p)+b)_k = (Wx+b)_k$ since $\E[m_j] = 1-p$. So $G\propto\exp(Wx+b)$, i.e. $G = \operatorname{softmax}(Wx+b)$.</p>
</details>
<p><em>Regularization view.</em> In a linear model the expectation over masks can be computed, and dropout is a form of ridge.</p>
<div class="thm" data-kind="Proposition" data-name="Dropout in linear regression is adaptive ridge">
<p>For inverted dropout on the inputs of a linear model, with independent masks per example and coordinate,
$$\E_{\text{masks}}\sum_{i=1}^n\Big(y_i-\sum_jx_{ij}\tfrac{m_{ij}}{1-p}\theta_j\Big)^2 = \|y-X\theta\|^2+\frac{p}{1-p}\sum_j\theta_j^2\,\|X_{\cdot j}\|^2 .$$
So dropout training minimizes a ridge objective whose penalty on $\theta_j$ is proportional to the energy $\sum_ix_{ij}^2$ of feature $j$ — equivalently, ridge on standardized features.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $\tilde m_{ij} = m_{ij}/(1-p)$, so $\E\tilde m_{ij} = 1$ and $\Var(\tilde m_{ij}) = \frac{1-p}{(1-p)^2}-1 = \frac{p}{1-p}$. For each $i$, $\sum_jx_{ij}\tilde m_{ij}\theta_j$ has mean $x_i^\top\theta$ and variance $\frac{p}{1-p}\sum_jx_{ij}^2\theta_j^2$ by independence. Since $\E(y_i-U)^2 = (y_i-\E U)^2+\Var(U)$, summing over $i$ gives the result.</p>
</details>
<p>Wager, Wang & Liang (2013) extend this to GLMs: to second order, dropout adds the penalty $\frac{p}{2(1-p)}\sum_j\theta_j^2\sum_i \mathrm{Var}_i\,x_{ij}^2$, where $\mathrm{Var}_i$ is the model variance at $x_i$ — a ridge penalty adapted to both feature scale and local curvature.</p>
<p><em>Bayesian view.</em> Gal & Ghahramani (2016) showed that training with dropout and L2 can be read as variational inference for a Bayesian network whose approximate posterior over each weight row is a mixture of a point mass at zero and a narrow Gaussian. Keeping dropout <em>on</em> at test time and averaging $T$ stochastic passes ("MC dropout") then approximates the posterior predictive mean, and their spread gives an uncertainty estimate. The approximation is crude — the variational family is rigid, the implied posterior does not concentrate as data grow in the usual way, and the dropout rate is not learned — so treat MC-dropout uncertainties as heuristic rather than calibrated.</p>`,
        lab: 'regularization-deep-dropout',
      },
      {
        title: 'Data augmentation and noise injection',
        html: R`
<p>Augmentation trains on transformed copies of the data — crops, flips, colour jitter for images; noise for signals — encoding the prior belief that the label is invariant to these transformations. Statistically it replaces the empirical distribution by a smoothed "vicinal" distribution around each training point (Chapelle et al., 2001), adding information that no penalty on $\theta$ can supply.</p>
<div class="thm" data-kind="Proposition" data-name="Input noise is Tikhonov regularization (Bishop, 1995)">
<p>For a linear model with input noise $\varepsilon\sim(0,\sigma^2I)$ independent of $(x,y)$, $\E_\varepsilon\big(y-\theta^\top(x+\varepsilon)\big)^2 = (y-\theta^\top x)^2+\sigma^2\|\theta\|^2$: training on noisy inputs is exactly ridge regression. For a smooth nonlinear $f$ and small $\sigma$, $\E_\varepsilon(y-f(x+\varepsilon))^2 = (y-f(x))^2+\sigma^2\big(\|\nabla f(x)\|^2-(y-f(x))\,\Delta f(x)\big)+O(\sigma^4)$, and near a good fit the leading penalty is $\sigma^2\|\nabla f(x)\|^2$, a smoothness penalty.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Linear case: expand the square; the cross term has mean zero and $\E(\theta^\top\varepsilon)^2 = \sigma^2\|\theta\|^2$. Nonlinear case: Taylor-expand $f(x+\varepsilon) = f(x)+\nabla f^\top\varepsilon+\tfrac12\varepsilon^\top\nabla^2f\,\varepsilon+O(\|\varepsilon\|^3)$, square, and take expectations using $\E\varepsilon = 0$, $\E\varepsilon\varepsilon^\top = \sigma^2I$ (so $\E\,\varepsilon^\top\nabla^2f\varepsilon = \sigma^2\Delta f$), dropping odd moments for symmetric noise.</p>
</details>
<p>Related techniques: <em>label smoothing</em> (train against $(1-\epsilon)e_y+\epsilon/K$, penalizing overconfident logits), <em>mixup</em> (train on convex combinations of input pairs and their labels), and the noise injected by small minibatches (<a href="#/t/sgd-optimizers">SGD</a>) and by batch statistics.</p>`,
      },
      {
        title: 'Double descent and the minimum-norm interpolant',
        html: R`
<p>The classical U-curve says test risk first falls as capacity grows (bias decreases) and then rises (variance increases). Belkin et al. (2019) observed that if capacity keeps growing past the <strong>interpolation threshold</strong> — the point where the model can fit the training data exactly — risk peaks at the threshold and then <em>descends again</em>. The overparameterized regime is not governed by the classical tradeoff, because among the many interpolants the training procedure picks a particular one: for least squares started at zero, gradient descent converges to the <strong>minimum-norm</strong> interpolant (<a href="#/t/deep-learning-theory">implicit bias</a>).</p>
<p>The phenomenon can be computed exactly in the simplest setting.</p>
<div class="thm" data-name="Risk of least squares across the threshold (Gaussian design)">
<p>Let $x\sim\N(0,I_p)$, $y = x^\top\beta+\varepsilon$ with $\Var(\varepsilon) = \sigma^2$, $\|\beta\| = r$, and $n$ training points. Let $\hat\beta$ be ordinary least squares if $p\lt n$ and the minimum-norm interpolant $\hat\beta = X^\top(XX^\top)^{-1}y$ if $p\gt n$. The excess test risk $R = \E\|\hat\beta-\beta\|^2$ is
$$R = \begin{cases}\sigma^2\,\dfrac{p}{n-p-1}, & p\le n-2,\\[2ex] r^2\Big(1-\dfrac np\Big)+\sigma^2\,\dfrac{n}{p-n-1}, & p\ge n+2 .\end{cases}$$
Both branches diverge as $p\to n$. As $n,p\to\infty$ with $p/n\to\gamma$: $R\to\sigma^2\gamma/(1-\gamma)$ for $\gamma\lt1$ and $r^2(1-1/\gamma)+\sigma^2/(\gamma-1)$ for $\gamma\gt1$ (Hastie, Montanari, Rosset & Tibshirani, 2022).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For a fresh $x_0\sim\N(0,I)$ the excess risk is $\E(x_0^\top(\hat\beta-\beta))^2 = \E\|\hat\beta-\beta\|^2$. <em>Underparameterized.</em> OLS is unbiased with $\hat\beta-\beta = (X^\top X)^{-1}X^\top\varepsilon$, so $\E\|\hat\beta-\beta\|^2 = \sigma^2\E\tr(X^\top X)^{-1}$. Since $X^\top X$ is Wishart$_p(n,I)$, $\E(X^\top X)^{-1} = I/(n-p-1)$ for $n\gt p+1$, giving $\sigma^2p/(n-p-1)$. <em>Overparameterized.</em> $\hat\beta = X^\top(XX^\top)^{-1}(X\beta+\varepsilon) = P\beta+X^\top(XX^\top)^{-1}\varepsilon$ where $P$ is the orthogonal projection onto the row space of $X$. The bias part is $\|(I-P)\beta\|^2$; by rotational invariance of the Gaussian design $P$ is a uniformly random rank-$n$ projection, $\E P = (n/p)I$, so $\E\|(I-P)\beta\|^2 = r^2(1-n/p)$. The noise part has mean-zero cross terms and $\E\|X^\top(XX^\top)^{-1}\varepsilon\|^2 = \sigma^2\E\tr(XX^\top)^{-1} = \sigma^2n/(p-n-1)$, since $XX^\top$ is Wishart$_n(p,I)$.</p>
</details>
<p>The spike at $p\approx n$ is a variance explosion: with as many parameters as data points, the design is nearly singular and the interpolant must chase the noise with huge coefficients. Past the threshold, extra dimensions give the interpolant room to fit the noise with <em>small</em> norm, spreading it over many directions where it does little harm at test time. In this isotropic model the second descent never beats the null risk $r^2$; in misspecified models, where additional features carry additional signal, or with random features, the overparameterized minimum can be the global one (Belkin, Hsu & Xu, 2020; Mei & Montanari, 2022). Optimally tuned ridge removes the peak altogether, and double descent also appears as a function of training time and of sample size in deep networks (Nakkiran et al., 2020).</p>
<p>The lab fits random ReLU features (a two-layer network with a frozen first layer) by minimum-norm least squares. Watch the spike at $p = n$, the second descent, and the effect of a small ridge penalty.</p>`,
        lab: 'regularization-deep-double-descent',
      },
    ],
    quiz: [
      { q: R`Why are "add $\frac\lambda2\|\theta\|^2$ to the loss" and "multiply the weights by $1-\eta\lambda$ each step" the same for SGD but different for Adam?`,
        a: R`For SGD the penalty's gradient $\lambda\theta$ is added to $g_t$ and multiplied by $\eta$, giving exactly $\theta\leftarrow(1-\eta\lambda)\theta-\eta g_t$. Adam divides the whole gradient, including $\lambda\theta$, by $\sqrt{\hat v_t}$ coordinatewise (and mixes it into the momentum), so the effective decay differs across coordinates and over time. Decoupled decay (AdamW) applies the shrinkage outside the preconditioner.` },
      { q: R`Using the filter comparison, roughly what ridge penalty corresponds to 500 gradient-descent steps with $\eta = 0.01$, and which eigendirections are fitted by then?`,
        a: R`$\lambda\approx1/(\eta t) = 1/5 = 0.2$. Directions with eigenvalue $s\gg0.2$ have filter near 1 (fully fitted); those with $s\ll0.2$ are shrunk toward zero ($\varphi\approx\eta st = 5s$).` },
      { q: R`For dropout on the inputs of linear regression with $p = 0.5$, what is the equivalent penalty, and why is it scale-equivariant in a way plain ridge is not?`,
        a: R`$\frac{p}{1-p} = 1$, so the penalty is $\sum_j\theta_j^2\|X_{\cdot j}\|^2$. If feature $j$ is multiplied by $c$, the fitted $\theta_j$ is divided by $c$ and the penalty term $\theta_j^2\|X_{\cdot j}\|^2$ is unchanged — the fit is the same as before rescaling. Plain ridge penalizes $\theta_j^2$ directly and so depends on the units of each feature.` },
      { q: R`In the Gaussian-design theorem with $n = 100$, $\sigma^2 = 1$, $r^2 = 1$, compute the risk at $p = 50$, $p = 98$, $p = 102$ and $p = 400$.`,
        a: R`$p=50$: $50/49\approx1.02$. $p=98$: $98/1 = 98$. $p=102$: $1\cdot(1-100/102)+100/1\approx100.02$. $p=400$: $(1-0.25)+100/299\approx0.75+0.33 = 1.08$. The risk explodes near $p = n$ and comes back down on both sides.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'deep-learning-theory',
    domain: 'deep',
    order: 70,
    title: 'Theory of Deep Learning: NTK, Implicit Bias & Overparameterization',
    subtitle: 'Infinite-width limits, lazy training, the solutions gradient descent prefers, and benign overfitting',
    level: 3,
    sources: { other: 'Neal (1996); Lee et al. (2018); Matthews et al. (2018); Jacot, Gabriel & Hongler (2018); Lee et al. (2019); Chizat, Oyallon & Bach (2019); Du, Zhai, Póczos & Singh (2019); Mei, Montanari & Nguyen (2018); Soudry, Hoffer, Nacson, Gunasekar & Srebro (2018); Lyu & Li (2020); Bartlett, Long, Lugosi & Tsigler (2020); Belkin et al. (2019); Bartlett, Montanari & Rakhlin (2021) survey; Prince, Understanding Deep Learning (2023) Ch. 20' },
    prereqs: ['sgd-optimizers', 'kernel-methods', 'gaussian-processes'],
    related: ['regularization-deep', 'svm', 'vc-theory', 'bias-variance', 'initialization-normalization', 'ridge-regression'],
    tags: ['neural tangent kernel', 'lazy training', 'NNGP', 'implicit bias', 'max margin', 'benign overfitting', 'mean field'],
    summary: R`Three ideas make overparameterized networks partly tractable. At infinite width a randomly initialized network is a Gaussian process (Neal), and — in the neural-tangent scaling — gradient descent barely moves the weights, so training is equivalent to kernel regression with the neural tangent kernel (NTK): "lazy training". Gradient descent also has an <em>implicit bias</em>: on underdetermined least squares it converges to the minimum-norm interpolant, and on separable classification with logistic loss its direction converges to the hard-margin SVM. Finally, interpolating the training data need not hurt: in high dimension, the minimum-norm interpolant can be consistent when the covariance has many small directions to absorb the noise — benign overfitting. None of these is the whole story; real networks learn features in a way the kernel limit cannot.`,
    survey: {
      task: 'Explain optimization and generalization of overparameterized networks',
      assumptions: R`Very wide layers with specific parameter scalings (NTK/NNGP); small steps (gradient flow); separable data or squared loss; high-dimensional covariance structure (benign overfitting)`,
      output: R`Limiting kernels (NNGP, NTK); convergence guarantees; characterizations of the solution GD selects; risk of interpolants`,
      strengths: 'Exact, testable predictions in their regimes; connects networks to kernels, GPs, SVMs and ridge.',
      weaknesses: 'Lazy regime excludes feature learning; most results are asymptotic in width or time and for simple architectures.',
      complexity: R`NTK regression $O(n^3)$ in the sample size; exact kernels computable layer by layer`,
    },
    sections: [
      {
        title: 'Three puzzles',
        html: R`
<p>Classical statistics and optimization predict three things that deep learning violates:</p>
<ol>
<li><strong>Non-convex optimization should get stuck.</strong> Yet gradient descent from random initialization routinely drives the training loss of large networks to zero.</li>
<li><strong>More parameters than data should overfit.</strong> Yet networks with $P\gg n$ that interpolate the training data — even noisy data — often generalize well, and test error frequently <em>improves</em> as width grows (<a href="#/t/regularization-deep">double descent</a>).</li>
<li><strong>Capacity bounds should explain generalization.</strong> Yet those same networks can fit random labels, so any bound depending only on the function class is vacuous.</li>
</ol>
<p>The resolutions below share a theme: in the overparameterized regime, the question is not whether a good solution exists or whether a bad one exists — both do — but <em>which solution the training algorithm reaches</em>. Width makes the optimization landscape benign near initialization, and the algorithm's geometry selects low-complexity interpolants.</p>`,
      },
      {
        title: 'Infinite width at initialization: networks as Gaussian processes',
        html: R`
<div class="thm" data-name="Neal (1996)">
<p>Let $f(x) = b+\frac{1}{\sqrt m}\sum_{j=1}^mv_j\,\phi(w_j^\top x+b_j)$ with all parameters independent, $(w_j,b_j)$ IID, $v_j$ IID with mean $0$ and variance $\sigma_v^2$, $b\sim\N(0,\sigma_b^2)$, and $\E\,\phi(w^\top x+b_1)^2\lt\infty$. As $m\to\infty$, for every finite set of inputs $x^{(1)},\dots,x^{(k)}$, the vector $(f(x^{(1)}),\dots,f(x^{(k)}))$ converges in distribution to $\N(0,K)$ with
$$K(x,x') = \sigma_b^2+\sigma_v^2\,\E\big[\phi(w^\top x+b_1)\,\phi(w^\top x'+b_1)\big].$$
That is, the random function $f$ converges to a centred <a href="#/t/gaussian-processes">Gaussian process</a> with covariance kernel $K$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Let $U_j = v_j\big(\phi(w_j^\top x^{(1)}+b_j),\dots,\phi(w_j^\top x^{(k)}+b_j)\big)\in\R^k$. The $U_j$ are IID; $\E U_j = \E[v_j]\,\E[\cdots] = 0$ by independence; and $\Cov(U_j)_{ab} = \sigma_v^2\E[\phi(w^\top x^{(a)}+b_1)\phi(w^\top x^{(b)}+b_1)]$, finite by Cauchy–Schwarz. The multivariate <a href="#/t/clt">CLT</a> gives $m^{-1/2}\sum_jU_j\dto\N(0,\sigma_v^2\,\E[\cdots])$, and adding the independent Gaussian output bias adds $\sigma_b^2$ to every entry.</p>
</details>
<p>For deep networks the same argument applies layer by layer (taking widths to infinity sequentially, or jointly with more work: Lee et al., 2018; Matthews et al., 2018). Each layer's pre-activations become a GP whose kernel is obtained from the previous one by
$$K_{\ell}(x,x') = \sigma_b^2+\sigma_w^2\,\E_{(u,v)\sim\N(0,\,K_{\ell-1}|_{\{x,x'\}})}\big[\phi(u)\phi(v)\big],$$
the <strong>NNGP kernel</strong>. For ReLU the expectation is explicit (Cho & Saul, 2009): if $u,v$ have variances $q_1,q_2$ and correlation $\cos\vartheta$,
$$\E[\mathrm{ReLU}(u)\,\mathrm{ReLU}(v)] = \frac{\sqrt{q_1q_2}}{2\pi}\big(\sin\vartheta+(\pi-\vartheta)\cos\vartheta\big),$$
the <em>arc-cosine kernel</em>; at $\vartheta = 0$ it gives $q/2$, matching $\E[\mathrm{ReLU}(z)^2] = \Var(z)/2$. The correlation map $c\mapsto$ (normalized kernel) is exactly the order/chaos map of <a href="#/t/initialization-normalization">initialization theory</a>.</p>
<div class="note">
<p>Consequence: exact Bayesian inference for an infinitely wide network with Gaussian priors on the weights and Gaussian likelihood is GP regression with the NNGP kernel — posterior mean $K(x,X)(K(X,X)+\sigma^2I)^{-1}y$. Nothing is learned about features: the kernel is fixed by the architecture and the prior. This is the first sign that infinite-width limits trade away representation learning for tractability.</p>
</div>`,
      },
      {
        title: 'The neural tangent kernel and lazy training',
        html: R`
<p>Now train. For any parametric model $f(x;\theta)$ and squared loss $L(\theta) = \frac12\sum_{i=1}^n(f(x_i;\theta)-y_i)^2$, gradient flow $\dot\theta = -\nabla L(\theta)$ induces dynamics on the function itself:</p>
$$\dot f_t(x) = \nabla_\theta f(x;\theta_t)^\top\dot\theta_t = -\sum_{i=1}^n\Theta_t(x,x_i)\,\big(f_t(x_i)-y_i\big),\qquad\Theta_t(x,x') = \nabla_\theta f(x;\theta_t)^\top\nabla_\theta f(x';\theta_t).$$
<p>This identity is exact. $\Theta_t$ is the <strong>neural tangent kernel</strong> — the Gram kernel of the parameter gradients — and it generally changes during training. Everything simplifies if it does not.</p>
<div class="thm" data-name="NTK limit (Jacot, Gabriel & Hongler, 2018)">
<p>For a fully connected network in the <em>NTK parametrization</em> (standard-normal weights, each layer's output multiplied by $1/\sqrt{\text{fan-in}}$) and a Lipschitz, smooth activation, as the hidden widths tend to infinity: (i) $\Theta_0$ converges in probability to a deterministic kernel $\Theta_\infty$ computable layer by layer, and (ii) for any fixed training horizon $T$, $\sup_{t\le T}\|\Theta_t-\Theta_0\|\to0$. Training is then a linear ODE in function space.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Training the linearized model">
<p>If $\Theta_t\equiv\Theta$ (equivalently, if we train the linearization $f^{\mathrm{lin}}(x;\theta) = f(x;\theta_0)+\nabla_\theta f(x;\theta_0)^\top(\theta-\theta_0)$), then with $\Theta = \Theta(X,X)$ positive definite,
$$f_t(X)-y = e^{-\Theta t}\big(f_0(X)-y\big),\qquad f_t(x) = f_0(x)-\Theta(x,X)\,\Theta^{-1}\big(I-e^{-\Theta t}\big)\big(f_0(X)-y\big).$$
As $t\to\infty$ the network interpolates the data and predicts $f_\infty(x) = f_0(x)+\Theta(x,X)\Theta^{-1}(y-f_0(X))$: kernel regression (minimum-RKHS-norm interpolation) with the NTK, applied to the residual of the random initial function.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>On the training inputs, the identity above reads $\frac{\d}{\d t}(f_t(X)-y) = -\Theta(f_t(X)-y)$, a linear ODE with solution $e^{-\Theta t}(f_0(X)-y)$. For any $x$, $\dot f_t(x) = -\Theta(x,X)e^{-\Theta s}(f_0(X)-y)$ at time $s$; integrate from $0$ to $t$ using $\int_0^te^{-\Theta s}\d s = \Theta^{-1}(I-e^{-\Theta t})$. Positive definiteness gives $\|f_t(X)-y\|\le e^{-\lambda_{\min}(\Theta)t}\|f_0(X)-y\|$, so the training loss converges to zero linearly.</p>
</details>
<p>Why does the kernel freeze? In the NTK scaling each of the $\sim m$ weights in a layer affects the output through a factor $m^{-1/2}$. To change the outputs by $O(1)$, the parameter vector must move $O(1)$ in Euclidean norm in total — only $O(m^{-1/2})$ per weight — while the Hessian of $f$ with respect to $\theta$ has operator norm $O(m^{-1/2})$ (Liu, Zhu & Belkin, 2020). So the Jacobian, and with it $\Theta$, changes by $O(m^{-1/2})$: the network is close to its linearization along the whole trajectory. This yields <strong>global convergence</strong> proofs for finite but large width: for two-layer ReLU networks, GD drives the training loss to zero at a linear rate once the width is polynomial in $n$ and $1/\lambda_{\min}(\Theta_\infty)$ (Du, Zhai, Póczos & Singh, 2019), with extensions to deep and residual networks. Chizat, Oyallon & Bach (2019) showed laziness is not about width per se: any model trained with a large output scale becomes lazy.</p>
<p>The lab trains five widths on the same data with gradient descent and compares each trained network with its own linearization trained for the same number of steps; the relative movement of weights and of the empirical NTK falls like $m^{-1/2}$.</p>`,
        lab: 'deep-learning-theory-lazy',
      },
      {
        title: 'Beyond the kernel regime: feature learning',
        html: R`
<p>If the lazy limit were the whole truth, deep networks would be no better than kernel machines with a particular kernel — and empirically, finite networks usually beat their own NTK regression, sometimes by a wide margin. The difference is <strong>feature learning</strong>: in practice the hidden representations (and the empirical NTK) change substantially during training, aligning with the target.</p>
<p>A different scaling makes this tractable. In the <strong>mean-field parametrization</strong>, $f(x) = \frac1m\sum_{j=1}^mv_j\phi(w_j^\top x)$ (note $1/m$, not $1/\sqrt m$), each neuron moves by $O(1)$ during training. As $m\to\infty$ the empirical distribution of neurons $\rho_t$ of $(v_j,w_j)$ evolves by a Wasserstein gradient flow of the risk viewed as a functional of $\rho$ (Mei, Montanari & Nguyen, 2018; Chizat & Bach, 2018; Rotskoff & Vanden-Eijnden, 2018). The risk is convex in $\rho$, and under homogeneity and support conditions the flow converges to a global minimizer — a non-convex problem in $\theta$ becomes a convex one in distribution space. For deep networks, the "maximal update" parametrization ($\mu$P; Yang & Hu, 2021) chooses per-layer scalings so that every layer learns features at infinite width, and makes optimal hyperparameters transfer across widths.</p>
<div class="ex" data-name="Where kernels provably lose">
<p>Rotation-invariant kernels on $\R^d$, the NTK included, need on the order of $d^k$ samples to learn a degree-$k$ polynomial in high dimension, even when the target depends on the input only through a single direction $u^\top x$ (Ghorbani, Mei, Misiakiewicz & Montanari, 2021). A network that learns the direction $u$ in its first layer adapts to the one-dimensional structure; random-features models with fixed first layers cannot even efficiently approximate a single ReLU neuron (Yehudai & Shamir, 2019). Adaptivity to low-dimensional structure is exactly what Barron-type results (<a href="#/t/universal-approximation">universal approximation</a>) promise and kernels cannot deliver (Bach, 2017).</p>
</div>`,
      },
      {
        title: 'Implicit bias: which solution gradient descent picks',
        html: R`
<p>When the loss has many global minimizers, the algorithm decides. The two cleanest results concern linear models, where the answer is a classical estimator.</p>
<div class="thm" data-name="Gradient descent on least squares finds the minimum-norm interpolant">
<p>Let $X\in\R^{n\times p}$ have rank $n\lt p$. Gradient descent on $\frac12\|y-X\theta\|^2$ from $\theta_0$ with step $0\lt\eta\lt2/\lambda_{\max}(X^\top X)$ converges to the interpolant closest to $\theta_0$; from $\theta_0 = 0$ it converges to
$$\hat\theta_{\mathrm{mn}} = X^\top(XX^\top)^{-1}y = \argmin\{\|\theta\|_2:\ X\theta = y\}.$$
The same holds for SGD and for any method whose updates lie in the row space of $X$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Each gradient $X^\top(X\theta-y)$ lies in the row space $\mathcal R = \mathrm{range}(X^\top)$, so $\theta_t-\theta_0\in\mathcal R$ for all $t$. On $\mathcal R$ the objective is strongly convex (since $X$ has full row rank), and with the stated step GD converges linearly to its unique minimizer over $\theta_0+\mathcal R$, which interpolates. Every interpolant has the form $\hat\theta_{\mathrm{mn}}+u$ with $u\in\ker X = \mathcal R^\perp$, and $\hat\theta_{\mathrm{mn}}\in\mathcal R$; so the interpolant in $\theta_0+\mathcal R$ is the orthogonal projection of $\theta_0$ onto the affine set of interpolants, and for $\theta_0 = 0$ it is the one with smallest norm, by Pythagoras: $\|\hat\theta_{\mathrm{mn}}+u\|^2 = \|\hat\theta_{\mathrm{mn}}\|^2+\|u\|^2$.</p>
</details>
<p>This is ridgeless regression, the $\lambda\to0^+$ limit of <a href="#/t/ridge-regression">ridge</a>, and it is the estimator whose risk curve exhibits <a href="#/t/regularization-deep">double descent</a>. Via the NTK, the same statement says a lazily trained wide network converges to minimum-RKHS-norm interpolation.</p>
<div class="thm" data-name="Logistic regression on separable data converges to the max-margin direction (Soudry, Hoffer, Nacson, Gunasekar & Srebro, 2018)">
<p>Let $\{(x_i,y_i)\}$ be linearly separable (through the origin) and let $L(w) = \sum_i\ell(y_iw^\top x_i)$ with $\ell$ the logistic or exponential loss. Gradient descent with a sufficiently small constant step, from any initialization, satisfies $\|w_t\|\to\infty$ and
$$\frac{w_t}{\|w_t\|}\to\frac{\hat w}{\|\hat w\|},\qquad\hat w = \argmin\{\|w\|_2:\ y_iw^\top x_i\ge1\ \forall i\},$$
the hard-margin <a href="#/t/svm">SVM</a> solution. Typically $w_t = \hat w\log t+\rho_t$ with $\rho_t$ bounded, so the directional error decays only like $1/\log t$, while the loss decays like $1/t$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>No finite minimizer exists, since scaling a separator up drives the loss to zero; GD drives $L\to0$, which forces every margin $y_iw_t^\top x_i\to\infty$ and $\|w_t\|\to\infty$. For the exponential tail, $-\nabla L(w) = \sum_i e^{-y_iw^\top x_i}y_ix_i$ is dominated by the points with the smallest margins, since all others are exponentially down-weighted. So any limit direction $\bar w$ is proportional to a non-negative combination $\sum_i\alpha_iy_ix_i$ supported on the points that attain the minimum normalized margin — precisely the KKT conditions of the hard-margin SVM, which characterize $\hat w$. The rate follows from substituting the ansatz $w_t\approx\hat w\log t$: support vectors (margin $1$) then contribute $e^{-\log t} = 1/t$ to the gradient, others $t^{-y_i\hat w^\top x_i} = o(1/t)$, so $\dot w\approx\hat w/t$, consistent with $w = \hat w\log t$.</p>
</details>
<p>The result is a striking instance of "the algorithm is the regularizer": no penalty appears anywhere, yet the iterates select the maximum-margin classifier, whose generalization is controlled by the margin rather than the dimension. It extends to homogeneous networks — GD on exponential-tailed losses converges in direction to a KKT point of the max-margin problem $\min\|\theta\|^2$ s.t. $y_if(x_i;\theta)\ge1$ (Lyu & Li, 2020; Ji & Telgarsky, 2020). The geometry of the optimizer matters: steepest descent with respect to a norm $\|\cdot\|$ finds the max-margin solution for <em>that</em> norm (Gunasekar, Lee, Soudry & Srebro, 2018), so sign-based methods such as Adam can select different solutions from GD.</p>`,
        lab: 'deep-learning-theory-implicit-bias',
      },
      {
        title: 'Benign overfitting and what generalization theory can say',
        html: R`
<p>When can an interpolant of noisy data still have near-optimal risk? For the minimum-norm interpolant in linear regression with covariance $\Sigma$ (eigenvalues $\lambda_1\ge\lambda_2\ge\cdots$), Bartlett, Long, Lugosi & Tsigler (2020) gave a sharp answer in terms of two effective ranks of the tail of the spectrum,
$$r_k(\Sigma) = \frac{\sum_{i\gt k}\lambda_i}{\lambda_{k+1}},\qquad R_k(\Sigma) = \frac{\big(\sum_{i\gt k}\lambda_i\big)^2}{\sum_{i\gt k}\lambda_i^2}.$$
With $k^\star$ the smallest $k$ for which $r_k(\Sigma)\ge bn$ (a constant $b$), the noise contribution to the excess risk is of order $\sigma^2\big(k^\star/n+n/R_{k^\star}(\Sigma)\big)$, and overfitting is <strong>benign</strong> — risk tends to the noise-free optimum — essentially when $r_0(\Sigma)/n\to0$, $k^\star/n\to0$ and $n/R_{k^\star}(\Sigma)\to0$.</p>
<div class="note">
<p>Reading the conditions: a few large eigenvalues carry the signal and are estimated well; a long tail of many small eigenvalues provides "room" in which the interpolant hides the noise. Each noisy label is fitted by a tiny spike spread across many unimportant directions, which barely affects predictions on new points. This requires the effective dimension of the tail to be much larger than $n$ — a high-dimensional phenomenon.</p>
</div>
<div class="warn" data-name="In low dimension, interpolation hurts">
<p>In fixed input dimension there is no tail to absorb noise. Minimum-norm interpolation with, e.g., the Laplace kernel is inconsistent in fixed dimension (Rakhlin & Zhai, 2019), and interpolating methods are typically "tempered" (risk bounded away from optimal) or "catastrophic" (Mallinar et al., 2022). The one-dimensional double-descent lab shows this: past the threshold the risk falls, but the min-norm fit never matches a well-regularized one.</p>
</div>
<p>For networks themselves, the most informative generalization bounds replace parameter counts by <em>norms and margins</em>: Bartlett (1998) showed that for networks with small weights the size of the weights, not their number, controls generalization; spectrally normalized margin bounds scale with $\prod_\ell\|W_\ell\|_2$ times a sum of norm ratios, divided by the margin and $\sqrt n$ (Bartlett, Foster & Telgarsky, 2017); PAC-Bayes bounds that optimize a posterior around the trained weights can be non-vacuous on real tasks (Dziugaite & Roy, 2017); and large empirical studies find flatness-based measures most predictive of the generalization gap (Jiang et al., 2020). These measures fit the picture above: the implicit biases of training keep norms small and margins large, and the noise of SGD favours flat minima (<a href="#/t/sgd-optimizers">SGD</a>).</p>
<div class="warn" data-name="Which regime are you in?">
<p>Each theory describes a regime: NNGP for untrained infinite-width priors, NTK for lazy training at large width and output scale, mean-field and $\mu$P for feature learning, implicit-bias results for long training on separable or underdetermined problems, benign overfitting for high-dimensional linear structure. Practical networks with finite width, large learning rates, normalization and data augmentation sit between these regimes. Treat each result as a precise statement about a model of training, and check its predictions — as the labs do — rather than as a description of all deep learning.</p>
</div>`,
      },
    ],
    quiz: [
      { q: R`In Neal's theorem, why is the output layer scaled by $1/\sqrt m$, and what goes wrong with $1/m$ or with no scaling?`,
        a: R`The output is a sum of $m$ IID mean-zero terms; dividing by $\sqrt m$ keeps its variance fixed, giving a nondegenerate Gaussian limit by the CLT. With $1/m$ the variance is $O(1/m)$ and the prior collapses to the constant function (the law of large numbers regime used by mean-field theory); with no scaling the variance grows like $m$.` },
      { q: R`Write the NTK of the linear model $f(x;\theta) = \theta^\top\psi(x)$ and say what "lazy training" means for it.`,
        a: R`$\nabla_\theta f = \psi(x)$, so $\Theta(x,x') = \psi(x)^\top\psi(x')$, constant in $\theta$. A linear model is always exactly "lazy": gradient flow on squared loss is kernel gradient flow with this fixed kernel, converging (from $\theta_0 = 0$, for $p\gt n$) to the minimum-norm interpolant — kernel ridgeless regression.` },
      { q: R`Logistic regression on separable data: the loss goes to zero like $1/t$, but the direction converges like $1/\log t$. What does this imply for early stopping based on training loss?`,
        a: R`The training loss becomes tiny long before the direction is close to the max-margin solution, so stopping when the loss is small yields a separator whose margin may still be far from maximal. The margin keeps improving (slowly) long after the loss and the training error look converged, which is one reason continued training can still improve test accuracy.` },
      { q: R`State the two qualitative requirements for benign overfitting in linear regression, and explain why neither holds for interpolation in a fixed low dimension.`,
        a: R`(1) The signal must live in a few high-variance directions that $n$ samples estimate well ($k^\star/n\to0$, $r_0/n\to0$); (2) there must be many low-variance directions whose effective rank greatly exceeds $n$ ($n/R_{k^\star}\to0$) to absorb the noise harmlessly. In fixed dimension $d$, the tail has at most $d\ll n$ directions, so the noise must be fitted by distorting the function along directions that matter.` },
      { q: R`In the lazy-training lab, relative weight change scales like $m^{-1/2}$. Give the heuristic argument.`,
        a: R`Each weight enters the output with a factor $m^{-1/2}$, so $\|\nabla_\theta f\| = O(1)$ while $\|\theta_0\|\sim\sqrt m$. Fitting $O(1)$ residuals needs a total parameter displacement $\|\theta_t-\theta_0\| = O(1)$, hence $\|\theta_t-\theta_0\|/\|\theta_0\| = O(m^{-1/2})$; and since the Hessian of $f$ is $O(m^{-1/2})$, the kernel changes by the same order.` },
    ],
  });
})();
