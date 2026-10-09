/* Deep Learning II: architectures and deep generative models.
 * Topics: cnn, rnn-lstm, attention-transformers, autoencoders-vae, gans, diffusion-models.
 * Labs live in js/labs/deep-2.js. */
(function () {
  const R = String.raw;

  /* ------------------------------------------------------------------ */
  /*  Convolutional networks                                             */
  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'cnn',
    domain: 'deep',
    order: 80,
    title: 'Convolutional Networks',
    subtitle: 'Locality, weight sharing and the symmetry they buy',
    level: 2,
    sources: { other: 'Goodfellow, Bengio & Courville (2016), Ch. 9; Prince, Understanding Deep Learning (2023), Ch. 10; LeCun et al. (1998); Simonyan & Zisserman (2015); He et al. (2016); Zhang (2019) on aliasing' },
    prereqs: ['backpropagation', 'perceptron-mlp'],
    related: ['initialization-normalization', 'regularization-deep', 'universal-approximation', 'orthogonal-series'],
    tags: ['convolution', 'equivariance', 'pooling', 'receptive field', 'weight sharing', 'images'],
    summary: R`A convolutional layer is a linear map with two constraints built in: each output looks only at a small neighbourhood of the input (<em>locality</em>), and the same weights are used at every position (<em>weight sharing</em>). These two constraints shrink the parameter count by many orders of magnitude, and they are not arbitrary: on periodic signals the convolutions are <em>exactly</em> the linear maps that commute with translation. Stacking convolutions, pointwise nonlinearities and pooling gives a network whose features move when the input moves (equivariance) and whose final summary can ignore where a pattern appeared (invariance), with a receptive field that grows predictably with depth.`,
    survey: {
      task: 'Learning from grid-structured signals: images, audio, spectrograms, volumes, sequences',
      assumptions: 'Local correlations matter; statistics are (approximately) translation-stationary',
      output: 'Hierarchical feature maps; class scores, dense predictions or embeddings',
      strengths: R`Massive parameter savings ($C'(Ck^2+1)$ vs. $CHW\cdot C'H'W'$); built-in translation equivariance; efficient on GPUs`,
      weaknesses: 'Not rotation/scale equivariant by default; striding causes aliasing; boundary effects; limited global context without depth',
      complexity: R`$O(H'W'\,C\,C'\,k^2)$ multiply-adds per layer`,
    },
    sections: [
      {
        title: 'Why a fully connected layer is the wrong prior for images',
        html: R`
<p>A <a href="#/t/perceptron-mlp">multilayer perceptron</a> treats its input as an unordered vector. Feed it a $224\times224$ colour image ($150{,}528$ numbers) and ask for a hidden layer of the same spatial size with 64 channels, and the weight matrix has about $150{,}528 \times 3{,}211{,}264 \approx 4.8\times 10^{11}$ entries. Even if memory were free, the statistics would not be: a model with that many free parameters has no reason to learn that an edge detector useful in the top-left corner is also useful in the bottom-right.</p>
<p>Natural signals have two structural regularities that a good architecture should hard-code rather than learn:</p>
<ul>
<li><strong>Locality.</strong> The pixels that determine whether location $(i,j)$ contains an edge are the pixels near $(i,j)$. Correlations decay with distance.</li>
<li><strong>Stationarity.</strong> The local statistics (what edges, textures and corners look like) are roughly the same everywhere in the image. A detector that works at one position should work at all of them.</li>
</ul>
<p>Locality says each output unit should connect only to a small $k\times k$ patch. Stationarity says all those patch-detectors should share the same weights. The resulting layer is a convolution. In statistical language, this is a drastic restriction of the hypothesis class that is (nearly) correct for the data, which is exactly the situation in which a restriction reduces variance without much bias (see <a href="#/t/bias-variance">bias–variance</a>).</p>
<div class="note">
<p>The convolutional layer is not "a different kind of neuron". It is an ordinary linear layer $\mathbf y = W\mathbf x + \mathbf b$ whose matrix $W$ is forced to be sparse (locality) and to have many tied entries (sharing). Everything else about training — losses, <a href="#/t/backpropagation">backpropagation</a>, <a href="#/t/sgd-optimizers">SGD</a> — is unchanged.</p>
</div>`,
      },
      {
        title: 'Convolution as a structured linear map',
        html: R`
<div class="def" data-name="Discrete convolution layer (as used in deep learning)">
<p>Let the input be a tensor $X\in\R^{C\times H\times W}$ ($C$ channels). A layer with $C'$ output channels, kernel size $k$, stride $s$, zero-padding $p$ and dilation $d$ has weights $K\in\R^{C'\times C\times k\times k}$ and biases $b\in\R^{C'}$, and computes
$$Y[c', i, j] = b_{c'} + \sum_{c=1}^{C}\sum_{u=0}^{k-1}\sum_{v=0}^{k-1} K[c', c, u, v]\; \tilde X[c,\; s i + d u,\; s j + d v],$$
where $\tilde X$ is $X$ surrounded by $p$ rows/columns of zeros.</p>
</div>
<p>Strictly, this is <em>cross-correlation</em>; mathematical convolution flips the kernel, $(h * x)_i = \sum_m h_m x_{i-m}$. Since the kernel is learned, the flip is immaterial (the network simply learns the flipped kernel), and deep-learning libraries universally call cross-correlation "convolution".</p>
<p>Flatten $X$ and $Y$ into vectors. Each output coordinate is a linear function of the input, so $\mathbf y = W\mathbf x + \mathbf b$ for some matrix $W$. In one dimension with $s=1$, $p=0$, kernel $(w_0,w_1,w_2)$ and input length 6:
$$W = \begin{pmatrix} w_0 & w_1 & w_2 & 0 & 0 & 0\\ 0 & w_0 & w_1 & w_2 & 0 & 0\\ 0&0&w_0&w_1&w_2&0\\ 0&0&0&w_0&w_1&w_2\end{pmatrix}.$$
This is a <strong>Toeplitz</strong> matrix: constant along diagonals. In 2D the matrix is doubly block-Toeplitz; with $C$ input and $C'$ output channels it is a $C'\times C$ grid of such blocks. Each row has at most $Ck^2$ non-zeros, and only $C'Ck^2$ distinct values appear anywhere in it.</p>
<div class="thm" data-kind="Proposition" data-name="Parameter count">
<p>A convolutional layer $\R^{C\times H\times W}\to\R^{C'\times H'\times W'}$ has $C'(Ck^2 + 1)$ parameters, independent of the image size. An unconstrained affine map between the same spaces has $CHW\cdot C'H'W' + C'H'W'$.</p>
</div>
<p>For $3\times3$ kernels from RGB to 64 channels this is $64\cdot(27+1) = 1{,}792$ parameters versus roughly $4.8\times10^{11}$. Ignoring biases, the ratio is $H'W'\cdot HW/k^2$: the number of output positions (sharing) times the number of patch-sized regions in the input (locality) — a factor that grows with the square of the image area.</p>
<p>Why this particular structure? The next theorem says that, on a periodic domain, Toeplitz (circulant) structure is not one choice among many: it is forced by translation symmetry.</p>
<div class="thm" data-kind="Theorem" data-name="Convolutions are the shift-commuting linear maps">
<p>Index $\R^n$ cyclically ($i \in \mathbb Z_n$) and let $S$ be the cyclic shift $(S\mathbf x)_i = x_{i-1}$. A linear map $L:\R^n\to\R^n$ satisfies $LS = SL$ if and only if $L$ is a circular convolution: there exists $\mathbf h\in\R^n$ with $(L\mathbf x)_i = \sum_{j} h_{i-j}\,x_j$ (indices mod $n$).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>($\Leftarrow$) $(L S\mathbf x)_i = \sum_j h_{i-j} x_{j-1} = \sum_{m} h_{i-1-m} x_m = (L\mathbf x)_{i-1} = (S L \mathbf x)_i$, substituting $m = j-1$.</p>
<p>($\Rightarrow$) Let $\mathbf e_j$ be the standard basis, so $\mathbf e_j = S^j \mathbf e_0$, and set $\mathbf h = L\mathbf e_0$. Since $L$ commutes with $S$ it commutes with every power $S^j$, so $L\mathbf e_j = L S^j\mathbf e_0 = S^j L\mathbf e_0 = S^j\mathbf h$, whose $i$-th entry is $h_{i-j}$. By linearity $L\mathbf x = \sum_j x_j L\mathbf e_j$ has $i$-th entry $\sum_j h_{i-j}x_j$. $\square$</p>
</details>
<p>The same argument works on $\mathbb Z_n^2$ (images on a torus) and for multi-channel maps (each channel pair gets its own kernel). Locality is a <em>further</em> restriction: the kernel $\mathbf h$ has support of size $k \ll n$. So a convolutional layer is "the most general translation-commuting linear map, with compact kernel support". This is also why the Fourier basis diagonalizes convolutions: circulant matrices share the eigenvectors $e^{2\pi \mathrm{i}\, \omega j/n}$, which is the convolution theorem and the basis of FFT-based implementations (and of the connection to <a href="#/t/orthogonal-series">orthogonal series</a>).</p>`,
        lab: 'cnn-conv-explorer',
      },
      {
        title: 'Equivariance, pooling and (approximate) invariance',
        html: R`
<p>Let $T_\tau$ translate an image by $\tau\in\mathbb Z^2$: $(T_\tau X)[c,\mathbf i] = X[c,\mathbf i-\tau]$. A map $\Phi$ is <strong>equivariant</strong> if $\Phi(T_\tau X) = T_\tau \Phi(X)$ (moving the input moves the output by the same amount) and <strong>invariant</strong> if $\Phi(T_\tau X) = \Phi(X)$.</p>
<div class="thm" data-kind="Theorem" data-name="Translation equivariance of stride-1 convolutional networks">
<p>On $\mathbb Z^2$ (infinite or cyclic domain), let $\Phi = \sigma_L\circ \mathcal C_L\circ\cdots\circ \sigma_1\circ\mathcal C_1$ where each $\mathcal C_\ell$ is a stride-1 convolution with bias and each $\sigma_\ell$ is a pointwise nonlinearity applied identically at every position. Then $\Phi(T_\tau X) = T_\tau\Phi(X)$ for every $\tau$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>For one convolution, $\mathcal C(T_\tau X)[c',\mathbf i] = b_{c'} + \sum_{c,\mathbf u} K[c',c,\mathbf u]\,X[c,\mathbf i+\mathbf u-\tau] = \mathcal C(X)[c',\mathbf i-\tau] = (T_\tau\mathcal C X)[c',\mathbf i]$; the bias is the same at every position, so it is unaffected. A pointwise map satisfies $\sigma(T_\tau Z)[\mathbf i] = \sigma(Z[\mathbf i-\tau]) = T_\tau\sigma(Z)[\mathbf i]$. Equivariant maps compose: if $A T = T A$ and $B T = T B$ then $AB\,T = A T B = T AB$. Induct over layers. $\square$</p>
</details>
<p>Two consequences follow immediately. First, any <strong>global pooling</strong> (sum, mean or max over all positions) after an equivariant map is <em>exactly invariant</em>, since it is a symmetric function of a set of values that translation merely permutes. Second, the intermediate feature maps are equivariant, not invariant: invariance has to be created by some aggregation over positions, and that is a feature — the network knows <em>where</em> things are until it chooses to forget.</p>
<div class="thm" data-kind="Proposition" data-name="What local pooling does and does not buy">
<p>Let $P$ be $p\times p$ max-pooling with stride $p$ applied to a feature map $Z$. (i) $P$ is equivariant to shifts by multiples of $p$: $P(T_{p\tau}Z) = T_\tau P(Z)$. (ii) $P(Z)$ is unchanged by any rearrangement of values within each pooling cell; in particular, it is unchanged by a small shift that keeps every cell's maximizing activation inside that cell. (iii) For shifts that are not multiples of $p$ there is no exact relation in general.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>(i) Cell $\mathbf c$ of $P(T_{p\tau}Z)$ is the max over $\{p\mathbf c+\mathbf u - p\tau : \mathbf u\in[0,p)^2\}$, which is cell $\mathbf c-\tau$ of $P(Z)$. (ii) A max is a symmetric function of its arguments. (iii) A one-pixel shift of a one-hot feature map lying on a cell boundary moves its mass into a neighbouring cell, so $P(T_1Z)$ is not a translate of $P(Z)$ by any integer amount. $\square$</p>
</details>
<p>So the oft-repeated claim that "pooling makes CNNs translation invariant" needs care. Local pooling buys robustness to <em>small deformations</em> (jitter within a cell), but any striding — in pooling or in convolution — subsamples, and subsampling a signal that contains frequencies above the new Nyquist limit causes <em>aliasing</em>: the output can change substantially under a one-pixel shift (Azulay &amp; Weiss, 2019). Low-pass filtering before subsampling ("blur-pool", Zhang, 2019) restores much of the lost stability, exactly as the sampling theorem predicts. The equivariance lab below lets you measure all of this.</p>
<div class="warn" data-name="Boundaries break the symmetry">
<p>The theorem holds on a torus or on $\mathbb Z^2$. Real images are finite and are zero-padded, so translation is not a symmetry of the domain: activations near the border see the padding, and deep networks can exploit this to encode absolute position (Islam, Jia &amp; Bruce, 2020). Equivariance is exact only away from the border by at least the receptive-field radius.</p>
</div>`,
        lab: 'cnn-equivariance',
      },
      {
        title: 'Output size and receptive field arithmetic',
        html: R`
<p>Two small formulas govern every CNN architecture diagram.</p>
<div class="thm" data-kind="Proposition" data-name="Output size">
<p>Along one axis, an input of length $n$ convolved with kernel size $k$, stride $s$, padding $p$ and dilation $d$ produces
$$n_{\text{out}} = \left\lfloor \frac{n + 2p - d(k-1) - 1}{s}\right\rfloor + 1 .$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>A dilated kernel spans $k_{\text{eff}} = d(k-1)+1$ input positions. Output index $i\ge 0$ reads padded positions $si,\dots,si+k_{\text{eff}}-1$, which must lie in $\{0,\dots,n+2p-1\}$. Hence $si \le n + 2p - k_{\text{eff}}$, i.e. $i \le \lfloor (n+2p-k_{\text{eff}})/s\rfloor$, and the count of valid $i$ is that bound plus one. $\square$</p>
</details>
<p>"Same" padding for odd $k$, $s=1$, $d=1$ is $p=(k-1)/2$, giving $n_{\text{out}}=n$. Pooling layers obey the same formula with their window and stride.</p>
<div class="def" data-name="Receptive field">
<p>The <em>receptive field</em> $r_\ell$ of a unit in layer $\ell$ is the side length of the input region that can influence it. The <em>jump</em> $j_\ell$ is the distance, in input pixels, between adjacent units of layer $\ell$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Receptive field recursion">
<p>With $r_0 = 1$, $j_0 = 1$ (the input itself), and layer $\ell$ having kernel $k_\ell$, stride $s_\ell$, dilation $d_\ell$:
$$j_\ell = j_{\ell-1}\, s_\ell, \qquad r_\ell = r_{\ell-1} + d_\ell(k_\ell - 1)\, j_{\ell-1}.$$
Consequently, for stride-1 undilated layers, $r_L = 1 + \sum_{\ell}(k_\ell - 1)$ (linear growth); with $s_\ell = 2$ throughout, the receptive field grows geometrically.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Induction on $\ell$. Adjacent units in layer $\ell-1$ are $j_{\ell-1}$ input pixels apart; a unit in layer $\ell$ reads $k_\ell$ of them spaced $d_\ell$ apart, so the centres of the first and last are $d_\ell(k_\ell-1)j_{\ell-1}$ input pixels apart. Each of those units sees $r_{\ell-1}$ input pixels, so the union spans $r_{\ell-1} + d_\ell(k_\ell-1)j_{\ell-1}$ pixels. Adjacent layer-$\ell$ units are $s_\ell$ layer-$(\ell-1)$ units apart, i.e. $s_\ell j_{\ell-1}$ input pixels. $\square$</p>
</details>
<div class="ex" data-name="A LeNet-style network on 28×28 digits">
<p>Input $1\times 28\times 28$. (1) Conv $5\times5$, 6 channels, $p=2$: output $6\times28\times28$, parameters $6(1\cdot25+1)=156$, $r=5$, $j=1$. (2) Max-pool $2\times2$, stride 2: $6\times14\times14$, $r = 5 + 1\cdot1 = 6$, $j=2$. (3) Conv $5\times5$, 16 channels, $p=0$: $\lfloor(14-5)/1\rfloor+1 = 10$, so $16\times10\times10$, parameters $16(6\cdot25+1) = 2{,}416$, $r = 6 + 4\cdot2 = 14$, $j=2$. (4) Pool $2\times2$: $16\times5\times5$, $r = 14 + 1\cdot 2 = 16$, $j=4$. Each of the 400 final features sees a $16\times16$ window of the 28×28 digit. A dense layer replacing step (1) alone would need $784\times 4{,}704 \approx 3.7$ million weights.</p>
</div>
<div class="ex" data-name="Why stacks of 3×3 kernels won">
<p>Three stride-1 $3\times3$ layers have $r = 1+2+2+2 = 7$, the same as one $7\times7$ layer. With $C$ channels in and out, the stack costs $3\cdot 9C^2 = 27C^2$ weights versus $49C^2$, and interleaves three nonlinearities instead of one (Simonyan &amp; Zisserman, 2015). The catch: the <em>theoretical</em> receptive field overstates what is used. The <em>effective</em> receptive field — the region where $\partial y/\partial x$ is non-negligible — is roughly Gaussian and grows only like $\sqrt{L}$ for $L$ stacked layers (Luo et al., 2016), for the same reason a sum of $L$ independent steps spreads like $\sqrt L$.</p>
</div>`,
      },
      {
        title: 'Backpropagation through a convolution',
        html: R`
<p>Because a convolution is linear, its gradients are themselves convolutions, which is why the forward and backward passes cost the same order of work. Work in one dimension, one channel, stride 1: $y_i = \sum_{u} w_u x_{i+u}$, and let $\delta_i = \partial\mathcal L/\partial y_i$ be the upstream gradient from <a href="#/t/backpropagation">backpropagation</a>.</p>
<div class="thm" data-kind="Proposition" data-name="Gradients of a convolutional layer">
<p>$$\frac{\partial\mathcal L}{\partial w_u} = \sum_i \delta_i\, x_{i+u}, \qquad \frac{\partial\mathcal L}{\partial x_m} = \sum_u \delta_{m-u}\, w_u .$$
The weight gradient is the cross-correlation of the input with $\delta$; the input gradient is the (true, flipped) convolution of $\delta$ with $w$ — a "full" convolution that pads $\delta$ by $k-1$. With stride $s$, the input gradient is a convolution of $\delta$ upsampled by inserting $s-1$ zeros between entries; this operation is the <em>transposed convolution</em> $W^\top\delta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the chain rule, $\partial\mathcal L/\partial w_u = \sum_i \delta_i\,\partial y_i/\partial w_u = \sum_i\delta_i x_{i+u}$. For the input, $x_m$ enters $y_i$ exactly when $i+u = m$, i.e. $u = m-i$, with coefficient $w_{m-i}$; so $\partial\mathcal L/\partial x_m = \sum_i \delta_i w_{m-i} = \sum_u \delta_{m-u}w_u$. In matrix form the second identity is $\nabla_{\mathbf x}\mathcal L = W^\top\boldsymbol\delta$, and the transpose of a (strided) Toeplitz matrix is again Toeplitz with the kernel reversed (and zeros interleaved for stride). $\square$</p>
</details>
<p>The weight gradient sums over <em>all positions</em> $i$: every position contributes evidence about the shared kernel. This is weight sharing seen from the statistical side — one kernel is estimated from $H'W'$ "replicates" per image, which is why CNNs are so sample-efficient compared with dense networks.</p>
<div class="note" data-name="Transposed convolutions and checkerboards">
<p>Decoders in autoencoders and GANs upsample with transposed convolutions. When the kernel size is not divisible by the stride, output pixels receive unequal numbers of contributions, producing the familiar checkerboard artefacts (Odena, Dumoulin &amp; Olah, 2016); resize-then-convolve avoids them.</p>
</div>`,
      },
      {
        title: 'Architectural building blocks',
        html: R`
<table>
<thead><tr><th>Block</th><th>What it is</th><th>Why it exists</th></tr></thead>
<tbody>
<tr><td>$1\times1$ convolution</td><td>A per-pixel linear map across channels</td><td>Mixes or reduces channels cheaply; "network in network"</td></tr>
<tr><td>Depthwise-separable</td><td>Per-channel $k\times k$ conv then $1\times1$</td><td>Cost $C k^2 + CC'$ vs $CC'k^2$ per pixel (MobileNet, Xception)</td></tr>
<tr><td>Dilated (atrous)</td><td>Kernel taps spaced $d$ apart</td><td>Exponential receptive-field growth without downsampling</td></tr>
<tr><td>Residual block</td><td>$\mathbf x + F(\mathbf x)$</td><td>Identity path keeps the Jacobian near $I$; trains 100+ layers (He et al., 2016)</td></tr>
<tr><td>Batch normalization</td><td>Standardize each channel over batch and space</td><td>Conditioning, larger learning rates (see <a href="#/t/initialization-normalization">initialization &amp; normalization</a>)</td></tr>
<tr><td>Global average pooling</td><td>Mean over all positions per channel</td><td>Exact translation invariance; replaces huge dense heads</td></tr>
</tbody>
</table>
<p>The residual block deserves a remark because it reappears in transformers and in the analysis of recurrent networks. If $\mathbf y = \mathbf x + F(\mathbf x)$ then $\partial\mathbf y/\partial\mathbf x = I + \partial F/\partial\mathbf x$, and the Jacobian of a stack of $L$ blocks is $\prod_\ell(I + J_\ell)$. When the $J_\ell$ are small this product stays near the identity instead of shrinking or growing geometrically — the same "additive path" idea that makes LSTMs trainable (see <a href="#/t/rnn-lstm">recurrent networks</a>).</p>
<div class="note" data-name="Beyond translations">
<p>The theorem of the previous section generalizes: for any group $G$ acting on the domain, the linear maps commuting with the action are <em>group convolutions</em>. Group-equivariant CNNs (Cohen &amp; Welling, 2016) build in rotations and reflections the same way ordinary CNNs build in translations; graph neural networks do the analogous thing for permutations of nodes, as self-attention does for permutations of tokens (see <a href="#/t/attention-transformers">attention</a>).</p>
</div>`,
      },
      {
        title: 'Pitfalls and limits',
        html: R`
<div class="warn" data-name="Equivariance is to translations only">
<p>A rotated or rescaled object produces feature maps that are not simple transforms of the original ones. Data augmentation teaches approximate invariance to such transformations statistically; group convolutions build it in exactly.</p>
</div>
<div class="warn" data-name="Striding and aliasing">
<p>Strided convolution and pooling are not shift-equivariant, and classification outputs of standard CNNs can flip under one-pixel shifts. Anti-aliasing (blur before subsample) helps.</p>
</div>
<div class="warn" data-name="Texture bias and shortcut features">
<p>The locality prior makes local texture statistics very easy to exploit. ImageNet-trained CNNs rely on texture more than humans do (Geirhos et al., 2019), and will happily learn spurious local cues (watermarks, scanner artefacts) if these predict the label.</p>
</div>
<div class="warn" data-name="Theoretical vs. effective receptive field">
<p>A large computed receptive field does not mean the network integrates information over it. If a task needs global context, check the effective receptive field, add global pooling or attention, or downsample earlier.</p>
</div>
<p>When the stationarity assumption fails — positions carry meaning, as in aligned face images or tabular data laid out as a grid — the convolutional prior is a liability rather than an asset. Matching the architecture's symmetry to the data's symmetry is the whole game.</p>`,
      },
    ],
    quiz: [
      { q: R`A layer maps $64\times 56\times 56$ to $128\times 56\times 56$ with $3\times3$ kernels and biases. How many parameters? How many would a dense layer between the same spaces need (ignore biases)?`,
        a: R`Convolution: $128\,(64\cdot 9 + 1) = 73{,}856$. Dense: $(64\cdot56^2)(128\cdot 56^2) = 200{,}704 \times 401{,}408 \approx 8.1\times10^{10}$. The convolution has roughly a million times fewer parameters because each output sees only a $3\times3$ patch and the weights are shared across all $3{,}136$ positions.` },
      { q: R`Compute the output side length for input 32, kernel 3, padding 1, stride 2, dilation 1; and for input 32, kernel 3, padding 2, stride 1, dilation 2.`,
        a: R`First: $\lfloor (32 + 2 - 2 - 1)/2\rfloor + 1 = \lfloor 31/2\rfloor + 1 = 16$. Second: the dilated kernel spans $2\cdot 2+1 = 5$ pixels, so $\lfloor(32 + 4 - 4 - 1)/1\rfloor + 1 = 32$ — padding $p=d(k-1)/2$ preserves size for odd $k$.` },
      { q: R`Why is a CNN followed by global average pooling exactly translation invariant on a torus, while one with $2\times2$ max-pooling in the middle is not?`,
        a: R`Stride-1 convolutions and pointwise nonlinearities are translation equivariant, so a shift of the input permutes the positions of the final feature map; averaging over all positions is invariant to permutation. A stride-2 pool is only equivariant to shifts by even amounts: an odd shift changes which pixels share a pooling cell, so the pooled map is not a translate of the original and the subsequent layers see genuinely different inputs (aliasing).` },
      { q: R`Show that a linear map $L$ on $\R^n$ (cyclic indices) that commutes with the cyclic shift is determined by the single vector $L\mathbf e_0$.`,
        a: R`Since $\mathbf e_j = S^j\mathbf e_0$ and $L S^j = S^j L$, we get $L\mathbf e_j = S^j(L\mathbf e_0)$. The columns of $L$ are therefore cyclic shifts of $\mathbf h = L\mathbf e_0$, so $L$ is the circulant matrix generated by $\mathbf h$, i.e. circular convolution with $\mathbf h$.` },
      { q: R`Four stride-1 $3\times3$ convolutions are followed by a $2\times2$ stride-2 pool and two more stride-1 $3\times3$ convolutions. What is the receptive field of the last layer?`,
        a: R`After four convs: $r = 1 + 4\cdot2 = 9$, $j = 1$. Pool: $r = 9 + 1\cdot 1 = 10$, $j = 2$. Two convs: each adds $2\cdot j = 4$, giving $r = 18$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  /*  Recurrent networks & LSTMs                                         */
  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'rnn-lstm',
    domain: 'deep',
    order: 90,
    title: 'Recurrent Networks & LSTMs',
    subtitle: 'Weight sharing in time, products of Jacobians, and the additive memory path',
    level: 2,
    sources: { other: 'Goodfellow, Bengio & Courville (2016), Ch. 10; Hochreiter & Schmidhuber (1997); Gers, Schmidhuber & Cummins (2000); Bengio, Simard & Frasconi (1994); Pascanu, Mikolov & Bengio (2013); Cho et al. (2014)' },
    prereqs: ['backpropagation'],
    related: ['time-series', 'hidden-markov', 'attention-transformers', 'initialization-normalization'],
    tags: ['sequences', 'BPTT', 'vanishing gradients', 'exploding gradients', 'gating', 'GRU'],
    summary: R`A recurrent network processes a sequence by repeatedly applying the same transition $\mathbf h_t = \phi(W\mathbf h_{t-1} + U\mathbf x_t + \mathbf b)$, carrying a state vector forward in time. Unrolled, it is a very deep feed-forward network with tied weights, trained by backpropagation through time. The gradient linking step $t$ to step $k$ is a product of $t-k$ Jacobians, so its norm behaves roughly like $(\gamma\,\lVert W\rVert)^{t-k}$: it vanishes or explodes exponentially in the time lag, which is why vanilla RNNs cannot learn long-range dependencies. The LSTM fixes this by routing memory through an additive, gated cell state whose Jacobian is a diagonal matrix of forget gates that can sit near the identity.`,
    survey: {
      task: 'Sequence modelling: language, speech, time series, control',
      assumptions: 'Markov state summary is adequate; dynamics are time-homogeneous (shared weights)',
      output: R`Hidden states $\mathbf h_t$; per-step predictions or a sequence summary`,
      strengths: R`Arbitrary-length input with fixed parameters; $O(1)$ memory per step at inference; natural for streaming`,
      weaknesses: 'Vanishing/exploding gradients; inherently sequential (no parallelism over time); finite-precision memory',
      complexity: R`$O(T\,n(n+m))$ per sequence for hidden size $n$, input size $m$; BPTT stores all $T$ states`,
    },
    sections: [
      {
        title: 'Sharing weights across time',
        html: R`
<p>A sequence $\mathbf x_1,\dots,\mathbf x_T$ has the same kind of structure in time that an image has in space: what matters at step $t$ is usually similar to what matters at step $t+1$. A recurrent network hard-codes this stationarity by applying one transition function at every step:</p>
$$\mathbf a_t = W\mathbf h_{t-1} + U\mathbf x_t + \mathbf b,\qquad \mathbf h_t = \phi(\mathbf a_t),\qquad \hat{\mathbf y}_t = V\mathbf h_t + \mathbf c,$$
<p>with $\mathbf h_t\in\R^n$, $\mathbf x_t\in\R^m$, $\phi$ typically $\tanh$, and $\mathbf h_0$ fixed (often $\mathbf 0$). The parameters $(W,U,V,\mathbf b,\mathbf c)$ do not depend on $t$, so the model handles sequences of any length with $n(n+m+1)$ recurrent parameters.</p>
<div class="note">
<p>Statistically, the RNN is a nonlinear, deterministic state-space model. Compare a <a href="#/t/hidden-markov">hidden Markov model</a> (discrete stochastic state, exact inference by the forward algorithm) or an ARMA model (<a href="#/t/time-series">time series</a>, linear state). The RNN's state is continuous and learned end-to-end: whatever summary of the past best predicts the future is what $\mathbf h_t$ is trained to hold. The linear RNN with $\phi = \text{id}$ and Gaussian noise is exactly a linear-Gaussian state-space model.</p>
</div>
<p><strong>Unrolling.</strong> Writing out the recursion for $T$ steps gives a $T$-layer feed-forward network in which every layer has the same weights. Everything we know about deep networks — in particular, how gradients behave through many layers — applies, with the twist that the "depth" is the sequence length and the weights are tied, so the same matrix $W$ is multiplied in over and over.</p>
<p>Expressively, RNNs are powerful: with rational weights and unbounded precision they are Turing complete (Siegelmann &amp; Sontag, 1995). The practical limitation is not what they can represent but what gradient descent can find, and that is a statement about gradients through long products.</p>`,
      },
      {
        title: 'Backpropagation through time',
        html: R`
<p>Let the loss be a sum over steps, $\mathcal L = \sum_{t=1}^T \ell_t(\hat{\mathbf y}_t, \mathbf y_t)$. Backpropagation on the unrolled graph is called <strong>backpropagation through time</strong> (BPTT).</p>
<div class="thm" data-kind="Proposition" data-name="BPTT gradient">
<p>Let $J_j = \partial\mathbf h_j/\partial\mathbf h_{j-1} = \diag(\phi'(\mathbf a_j))\,W$. Then
$$\frac{\partial \mathcal L}{\partial W} = \sum_{t=1}^T\sum_{k=1}^{t} \frac{\partial\ell_t}{\partial\mathbf h_t}\,\underbrace{\Big(\prod_{j=k+1}^{t} J_j\Big)}_{\partial\mathbf h_t/\partial\mathbf h_k}\,\frac{\partial^+\mathbf h_k}{\partial W},$$
where $\partial^+\mathbf h_k/\partial W$ is the "immediate" derivative holding $\mathbf h_{k-1}$ fixed (with entries $\phi'(a_{k,i})\,h_{k-1,j}$ for $W_{ij}$), and the product is ordered $J_tJ_{t-1}\cdots J_{k+1}$ (empty product $=I$).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$W$ influences $\ell_t$ through every $\mathbf h_k$ with $k\le t$. Treat the $W$ appearing at step $k$ as a separate copy $W^{(k)}$; then $\partial\mathcal L/\partial W = \sum_k \partial\mathcal L/\partial W^{(k)}$ (weight tying sums the gradients of the copies). The copy $W^{(k)}$ affects $\mathbf h_k$ directly, and $\mathbf h_k$ affects $\ell_t$ ($t\ge k$) through the chain $\mathbf h_k\to\mathbf h_{k+1}\to\cdots\to\mathbf h_t$, whose Jacobian is the stated product by the chain rule. Summing over $k\le t$ and over $t$ gives the formula. $\square$</p>
</details>
<p>In practice one never forms the products explicitly. The backward recursion
$$\boldsymbol\delta_t = \frac{\partial\ell_t}{\partial\mathbf h_t}^{\!\top} + J_{t+1}^\top\boldsymbol\delta_{t+1}, \qquad \boldsymbol\delta_{T+1} = \mathbf 0,$$
computes $\boldsymbol\delta_t = \partial\mathcal L/\partial\mathbf h_t$ in one sweep, and then $\partial\mathcal L/\partial W = \sum_t (\boldsymbol\delta_t\odot\phi'(\mathbf a_t))\,\mathbf h_{t-1}^\top$. Cost is $O(T n^2)$ time and $O(Tn)$ memory for the stored states. <strong>Truncated BPTT</strong> runs the backward pass only $\tau$ steps, trading bias (dependencies longer than $\tau$ get no gradient) for memory and speed.</p>
<p>The formula makes the central difficulty plain: the contribution of a dependency of lag $t-k$ is weighted by a product of $t-k$ Jacobians.</p>`,
      },
      {
        title: 'Vanishing and exploding gradients',
        html: R`
<div class="thm" data-kind="Theorem" data-name="Exponential bound (Bengio et al., 1994; Pascanu et al., 2013)">
<p>Suppose $|\phi'|\le\gamma$ (for $\tanh$, $\gamma = 1$; for the logistic sigmoid, $\gamma=1/4$). Then for $t > k$,
$$\Big\lVert\frac{\partial\mathbf h_t}{\partial\mathbf h_k}\Big\rVert_2 \le \big(\gamma\,\sigma_{\max}(W)\big)^{t-k}.$$
Hence if $\sigma_{\max}(W) < 1/\gamma$, long-range gradient contributions vanish exponentially in the lag. Conversely, exploding gradients ($\lVert\partial\mathbf h_t/\partial\mathbf h_k\rVert\to\infty$) are possible only if $\sigma_{\max}(W) > 1/\gamma$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>The operator 2-norm is submultiplicative, and $\lVert\diag(\phi'(\mathbf a_j))\rVert_2 = \max_i|\phi'(a_{j,i})|\le\gamma$. Hence $\lVert\prod_{j=k+1}^t J_j\rVert_2 \le \prod_{j}\lVert\diag(\phi'(\mathbf a_j))\rVert_2\lVert W\rVert_2 \le (\gamma\sigma_{\max}(W))^{t-k}$. The converse statement is the contrapositive of the bound. $\square$</p>
</details>
<p>For the <em>linear</em> RNN ($\phi = $ id) the Jacobian product is exactly $W^{t-k}$, and Gelfand's formula $\lVert W^m\rVert^{1/m}\to\rho(W)$ says the spectral radius sets the asymptotic rate in both directions: if $\rho(W)>1$ the gradient explodes along the leading eigenvector; if $\rho(W)<1$ it vanishes. With $\tanh$, saturation makes things worse for vanishing: $\phi'(a) = 1-\tanh^2(a) < 1$, so the effective per-step factor is $\rho(W)$ times a typical value of $1-h^2$, which shrinks as the state saturates.</p>
<div class="ex" data-name="A scalar RNN">
<p>Take $h_t = \tanh(w h_{t-1} + u x_t)$ with $w = 0.9$. Even in the most favourable case (no saturation, $\phi'=1$), $\partial h_{t}/\partial h_{t-50} = 0.9^{50} \approx 0.005$; with typical saturation $1-h^2\approx 0.7$, it is $(0.63)^{50}\approx 10^{-10}$. A dependency 50 steps back contributes ten orders of magnitude less gradient than one at lag 1 — gradient descent effectively never sees it. With $w = 1.2$ and inputs small enough that the state stays near zero, the factor is $1.2^{50} \approx 9{,}100$ and one update can throw the parameters into a region where the loss is enormous.</p>
</div>
<p>The two failure modes are not symmetric in practice. <strong>Exploding</strong> gradients are rare events (the loss surface has steep "cliffs") and are handled by <strong>gradient norm clipping</strong>: if $\lVert\mathbf g\rVert > c$, replace $\mathbf g$ by $c\,\mathbf g/\lVert\mathbf g\rVert$ (Pascanu et al., 2013). <strong>Vanishing</strong> gradients are the default and cannot be fixed by rescaling the step: the long-range signal is buried below the short-range one in the same gradient vector. Fixing it requires changing the architecture so that some path through time has a Jacobian close to the identity.</p>
<p>Setting $W$ to an orthogonal matrix makes $\sigma_{\max}(W) = 1$ and is a common initialization, but $\tanh$ still contracts. Unitary and orthogonally-constrained RNNs (Arjovsky, Shah &amp; Bengio, 2016) and linear recurrent units revisit this idea; gating, below, is the solution that won.</p>`,
        lab: 'rnn-lstm-gradients',
      },
      {
        title: 'The LSTM: an additive, gated memory',
        html: R`
<p>Hochreiter &amp; Schmidhuber (1997) diagnosed the problem above and proposed a fix: keep a memory $\mathbf c_t$ that is updated <em>additively</em>, so that $\partial\mathbf c_t/\partial\mathbf c_{t-1}$ is (close to) the identity — the "constant error carousel". Gates, computed from the current input and previous output, decide what to write and what to read. The modern form, with the forget gate of Gers, Schmidhuber &amp; Cummins (2000), is:</p>
$$\begin{aligned}
\mathbf f_t &= \sigma(W_f\mathbf x_t + U_f\mathbf h_{t-1} + \mathbf b_f) && \text{forget gate}\\
\mathbf i_t &= \sigma(W_i\mathbf x_t + U_i\mathbf h_{t-1} + \mathbf b_i) && \text{input gate}\\
\mathbf o_t &= \sigma(W_o\mathbf x_t + U_o\mathbf h_{t-1} + \mathbf b_o) && \text{output gate}\\
\mathbf g_t &= \tanh(W_g\mathbf x_t + U_g\mathbf h_{t-1} + \mathbf b_g) && \text{candidate}\\
\mathbf c_t &= \mathbf f_t\odot\mathbf c_{t-1} + \mathbf i_t\odot\mathbf g_t && \text{cell state}\\
\mathbf h_t &= \mathbf o_t\odot\tanh(\mathbf c_t) && \text{output}
\end{aligned}$$
<div class="thm" data-kind="Proposition" data-name="The cell-state path">
<p>Holding the gates and candidate fixed (i.e. along the direct path $\mathbf c_{t-1}\to\mathbf c_t$),
$$\frac{\partial^{\text{direct}}\mathbf c_t}{\partial\mathbf c_{k}} = \prod_{j=k+1}^{t}\diag(\mathbf f_j) = \diag\Big(\prod_{j=k+1}^t\mathbf f_j\Big).$$
No weight matrix and no derivative of a squashing function appears; the factor for unit $i$ is the product of its forget-gate values, which the network controls.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\mathbf c_j = \mathbf f_j\odot\mathbf c_{j-1} + \mathbf i_j\odot\mathbf g_j$, so with $\mathbf f_j, \mathbf i_j,\mathbf g_j$ held fixed $\partial\mathbf c_j/\partial\mathbf c_{j-1} = \diag(\mathbf f_j)$. Chain over $j = k+1,\dots,t$; products of diagonal matrices are diagonal with the elementwise product on the diagonal. $\square$</p>
</details>
<p>The full Jacobian $\partial\mathbf c_t/\partial\mathbf c_{t-1}$ also contains indirect terms, because the gates depend on $\mathbf h_{t-1} = \mathbf o_{t-1}\odot\tanh(\mathbf c_{t-1})$. Those terms behave like a vanilla RNN and can still vanish. But the gradient is a <em>sum</em> over paths, and the direct path alone carries signal across $t-k$ steps with factor $\prod_j f_{j,i}$. If a unit learns $f\approx 1$ (and $i\approx 0$) it holds its value indefinitely: the lab above shows the contrast with $\prod_j \diag(\phi'(\mathbf a_j))W$. In the original 1997 LSTM there was no forget gate ($\mathbf f\equiv 1$), so the carousel was exact, but the cell could never reset; the forget gate trades a little decay for the ability to clear memory.</p>
<div class="warn" data-name="Initialize the forget-gate bias high">
<p>With $\mathbf b_f = \mathbf 0$ the forget gates start near $\sigma(0) = 0.5$, and $0.5^{50}\approx 10^{-15}$ — a vanilla-RNN-like decay. Initializing $\mathbf b_f \approx 1$ or larger ($\sigma(1)\approx 0.73$, $\sigma(3)\approx 0.95$) makes the carousel open by default; Jozefowicz, Zaremba &amp; Sutskever (2015) found this single change closes much of the gap between LSTMs and alternatives.</p>
</div>
<p>The structural lesson — <em>make the default path through depth additive, so its Jacobian is identity-plus-something</em> — is the same as in residual networks and in the residual stream of a <a href="#/t/attention-transformers">transformer</a>. An LSTM with $\mathbf f = \mathbf 1$ is a residual network in time.</p>`,
      },
      {
        title: 'The GRU and a comparison',
        html: R`
<p>The gated recurrent unit (Cho et al., 2014) merges the cell and hidden state and uses two gates:</p>
$$\begin{aligned}
\mathbf z_t &= \sigma(W_z\mathbf x_t + U_z\mathbf h_{t-1} + \mathbf b_z) && \text{update gate}\\
\mathbf r_t &= \sigma(W_r\mathbf x_t + U_r\mathbf h_{t-1} + \mathbf b_r) && \text{reset gate}\\
\tilde{\mathbf h}_t &= \tanh\big(W\mathbf x_t + U(\mathbf r_t\odot\mathbf h_{t-1}) + \mathbf b\big)\\
\mathbf h_t &= (\mathbf 1-\mathbf z_t)\odot\mathbf h_{t-1} + \mathbf z_t\odot\tilde{\mathbf h}_t .
\end{aligned}$$
<p>(Conventions differ on whether $\mathbf z$ or $\mathbf 1-\mathbf z$ multiplies the old state.) The new state is a convex combination of the old state and a candidate — a learned, input-dependent <em>leaky integrator</em>, or exponential moving average with data-dependent smoothing. The direct-path Jacobian is $\diag(\mathbf 1-\mathbf z_t)$, playing the role of the LSTM's forget gate.</p>
<table>
<thead><tr><th>Cell</th><th>Recurrent parameters</th><th>Direct-path Jacobian</th><th>Notes</th></tr></thead>
<tbody>
<tr><td>Vanilla (Elman)</td><td>$n(n+m+1)$</td><td>$\diag(\phi'(\mathbf a_t))W$</td><td>Vanishes/explodes geometrically</td></tr>
<tr><td>LSTM</td><td>$4n(n+m+1)$</td><td>$\diag(\mathbf f_t)$</td><td>Separate memory $\mathbf c_t$ and output $\mathbf h_t$</td></tr>
<tr><td>GRU</td><td>$3n(n+m+1)$</td><td>$\diag(\mathbf 1-\mathbf z_t)$</td><td>Fewer parameters; similar accuracy on many tasks</td></tr>
</tbody>
</table>
<p>Large empirical comparisons (Greff et al., 2017; Jozefowicz et al., 2015) found no variant reliably better than the LSTM with a forget-gate bias, and that the forget gate and output nonlinearity are its most important parts.</p>
<div class="ex" data-name="How long can an LSTM remember?">
<p>A unit with constant forget gate $f$ retains a fraction $f^\tau$ of its content after $\tau$ steps: a <em>memory half-life</em> of $\tau_{1/2} = \log 2/(-\log f)$. For $f=\sigma(3)\approx 0.953$ that is about 14 steps; for $f = \sigma(5)\approx0.9933$, about 103 steps. Different units can learn different half-lives, giving the network a spectrum of timescales — a multiscale exponential smoother whose smoothing constants depend on the input.</p>
</div>`,
      },
      {
        title: 'Practice, pitfalls and the road to attention',
        html: R`
<div class="warn" data-name="Exposure bias">
<p>Sequence models are trained with <em>teacher forcing</em> (the true previous token is fed in), but at generation time they consume their own predictions. Errors compound along a distribution the model never saw in training. Scheduled sampling and sequence-level training are partial remedies.</p>
</div>
<div class="warn" data-name="The fixed-size bottleneck">
<p>An encoder–decoder RNN for translation compresses the whole source sentence into the final state $\mathbf h_T$. Performance degrades with sentence length because a fixed-size vector cannot hold an arbitrarily long sentence at fixed fidelity. Bahdanau, Cho &amp; Bengio (2015) let the decoder look back at <em>all</em> encoder states with learned weights — attention — and the bottleneck disappeared. Follow this thread to <a href="#/t/attention-transformers">attention and transformers</a>.</p>
</div>
<div class="warn" data-name="Sequential computation">
<p>$\mathbf h_t$ needs $\mathbf h_{t-1}$, so training cannot parallelize over time, and the path between distant positions has length $O(T)$. Transformers trade this for $O(T^2)$ parallel work and $O(1)$ path length.</p>
</div>
<p><strong>Other practical notes.</strong> Bidirectional RNNs run a second network backwards and concatenate states, when the whole sequence is available. Stacking RNN layers adds depth per step. Dropout must use the same mask at every time step on recurrent connections (Gal &amp; Ghahramani, 2016), or it destroys memory. Layer normalization stabilizes hidden-state scale. And recently, linear recurrences with carefully parameterized diagonal transitions (state-space models such as S4 and Mamba) have revived recurrence by making the "$\rho(W)$ near 1" regime explicit and the computation parallelizable via scans.</p>`,
      },
    ],
    quiz: [
      { q: R`Why does weight tying across time make the gradient a sum over time steps rather than a single term?`,
        a: R`Because $W$ is used at every step, the loss depends on $W$ through $T$ copies. The derivative of a function of several copies of the same parameter is the sum of the partial derivatives with respect to each copy, evaluated at the shared value. Each copy's contribution flows back through a different number of Jacobians, which is why lags are weighted so unequally.` },
      { q: R`For a $\tanh$ RNN with $\sigma_{\max}(W) = 0.95$, bound the gradient factor $\lVert\partial\mathbf h_t/\partial\mathbf h_{t-100}\rVert$. Can the gradient explode?`,
        a: R`$\le 0.95^{100}\approx 0.0059$, and smaller with saturation. Exploding is impossible because $\gamma\sigma_{\max}(W) = 0.95 < 1$, so every product of Jacobians is a contraction.` },
      { q: R`Gradient clipping fixes exploding gradients. Why can't an analogous "gradient amplification" fix vanishing ones?`,
        a: R`Clipping rescales the whole gradient vector, which preserves its direction. When gradients vanish, the problem is the direction: the long-lag contributions are many orders of magnitude smaller than the short-lag ones in the same sum, so scaling the sum up just scales up the short-range signal. One needs a path whose Jacobian does not shrink — the LSTM cell state or a residual connection.` },
      { q: R`In an LSTM, what is $\partial\mathbf c_t/\partial\mathbf c_{t-k}$ along the direct cell path, and what bias initialization keeps it from vanishing early in training?`,
        a: R`$\diag(\prod_{j=t-k+1}^{t}\mathbf f_j)$, the elementwise product of forget gates. Initializing $\mathbf b_f$ to about 1–3 makes $\mathbf f_j$ start near 0.73–0.95 instead of 0.5, so the product decays slowly over the lags the task needs.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  /*  Attention & the Transformer                                        */
  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'attention-transformers',
    domain: 'deep',
    order: 100,
    title: 'Attention & the Transformer',
    subtitle: 'Content-based weighted averaging, and the architecture built entirely from it',
    level: 2,
    sources: { other: 'Bahdanau, Cho & Bengio (2015); Vaswani et al. (2017); Prince, Understanding Deep Learning (2023), Ch. 12; Tsai et al. (2019) on attention as kernel smoothing; Su et al. (2021) RoPE; Katharopoulos et al. (2020) linear attention' },
    prereqs: ['rnn-lstm', 'kernel-regression'],
    related: ['kernel-methods', 'cnn', 'initialization-normalization', 'backpropagation'],
    tags: ['attention', 'transformer', 'self-attention', 'positional encoding', 'permutation equivariance', 'softmax'],
    summary: R`Attention computes, for each query, a weighted average of value vectors, with weights given by a softmax of query–key similarities: $\mathrm{Attn}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$. Statistically this is a Nadaraya–Watson kernel smoother whose kernel is learned, and the $1/\sqrt{d_k}$ factor is a bandwidth chosen so that logits have unit variance at initialization. Self-attention lets every position of a sequence read every other position in one step, but it is blind to order — it is exactly permutation equivariant — so transformers add positional information explicitly. Stacking attention and per-position MLPs with residual connections and layer normalization gives the transformer, whose cost is quadratic in sequence length but fully parallel.`,
    survey: {
      task: 'Sequence and set modelling: language, vision (patches), audio, proteins, tabular sets',
      assumptions: 'Relevant context can be retrieved by learned similarity; order supplied by positional encodings',
      output: 'Contextualized token representations; next-token distributions',
      strengths: R`$O(1)$ path length between any two positions; parallel over the sequence; scales well with data and compute`,
      weaknesses: R`$O(n^2)$ time and memory in sequence length; no built-in locality prior; length extrapolation is fragile`,
      complexity: R`$O(n^2 d + n d^2)$ per layer for $n$ tokens of width $d$`,
    },
    sections: [
      {
        title: 'From a bottleneck to a lookup',
        html: R`
<p>An encoder–decoder <a href="#/t/rnn-lstm">RNN</a> squeezes a source sentence into one vector. Bahdanau, Cho &amp; Bengio (2015) instead kept all encoder states $\mathbf h_1,\dots,\mathbf h_n$ and let the decoder, at each output step $t$, form a context vector as a weighted average
$$\mathbf c_t = \sum_{j=1}^n \alpha_{tj}\mathbf h_j,\qquad \alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{l}\exp(e_{tl})},\qquad e_{tj} = \mathbf v^\top\tanh(W\mathbf s_{t-1} + U\mathbf h_j),$$
where $\mathbf s_{t-1}$ is the decoder state. The weights are a soft, differentiable version of "look up the source word aligned with this target word", and they are computed from <em>content</em>, not position.</p>
<p>Abstracting away the RNN, attention is a soft dictionary lookup. There are <strong>queries</strong> $\mathbf q$, and a memory of <strong>key</strong>–<strong>value</strong> pairs $(\mathbf k_j,\mathbf v_j)$. A hard dictionary returns the value whose key equals the query; attention returns
$$\mathrm{Attn}(\mathbf q; \{(\mathbf k_j,\mathbf v_j)\}) = \sum_j \frac{\exp s(\mathbf q,\mathbf k_j)}{\sum_l\exp s(\mathbf q,\mathbf k_l)}\;\mathbf v_j$$
for a score function $s$. Bahdanau's $s$ is a small MLP ("additive attention"); the transformer uses a scaled dot product.</p>
<div class="def" data-name="Scaled dot-product attention (Vaswani et al., 2017)">
<p>For $Q\in\R^{n_q\times d_k}$, $K\in\R^{n\times d_k}$, $V\in\R^{n\times d_v}$,
$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}\Big)V,$$
where softmax acts on each row. Row $i$ of the result is a convex combination of the rows of $V$, with weights $A_{ij}$ determined by how well query $i$ matches key $j$.</p>
</div>`,
      },
      {
        title: 'Why divide by √d: a variance argument',
        html: R`
<div class="thm" data-kind="Proposition" data-name="Dot products grow like √d">
<p>Let $\mathbf q,\mathbf k\in\R^d$ be independent with independent coordinates of mean 0 and variance 1. Then $\E[\mathbf q^\top\mathbf k] = 0$ and $\Var(\mathbf q^\top\mathbf k) = d$. Hence $\mathbf q^\top\mathbf k/\sqrt d$ has variance 1, whatever $d$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\mathbf q^\top\mathbf k = \sum_{i=1}^d q_ik_i$. Each term has $\E[q_ik_i] = \E q_i\,\E k_i = 0$ and $\Var(q_ik_i) = \E[q_i^2]\E[k_i^2] - 0 = 1$. The terms are independent across $i$, so the variances add to $d$. $\square$</p>
</details>
<p>Why does the scale of the logits matter? Because the softmax's gradient dies when its input is spread out.</p>
<div class="thm" data-kind="Proposition" data-name="Softmax saturation">
<p>For $\mathbf p = \mathrm{softmax}(\mathbf s)$, the Jacobian is $\partial p_i/\partial s_j = p_i(\delta_{ij} - p_j)$, i.e. $\diag(\mathbf p) - \mathbf p\mathbf p^\top$. If the logits are multiplied by $c\to\infty$ and the maximum logit is unique, $\mathbf p$ approaches a one-hot vector exponentially fast in $c$ and every entry of the Jacobian tends to 0.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$\partial p_i/\partial s_j = \partial/\partial s_j\,[e^{s_i}/Z] = \delta_{ij}e^{s_i}/Z - e^{s_i}e^{s_j}/Z^2 = p_i(\delta_{ij}-p_j)$. Let the maximum be at $i^*$ with gap $\Delta>0$ to the second-largest logit. With logits $c\,\mathbf s$, every $p_j$ with $j\ne i^*$ is at most $e^{-c\Delta}$ and $1-p_{i^*}\le(n-1)e^{-c\Delta}$. The diagonal entries $p_i(1-p_i)$ and off-diagonal entries $-p_ip_j$ each contain at least one of these factors, so all are $O(n e^{-c\Delta})$. $\square$</p>
</details>
<p>Put together: with unscaled dot products in $d_k = 512$ dimensions, logits at initialization have standard deviation about $\sqrt{512}\approx 23$, the attention weights are essentially one-hot, and almost no gradient reaches the queries and keys. Dividing by $\sqrt{d_k}$ keeps the logits at unit scale, so attention starts diffuse and learnable. (Additive attention does not need the factor because $\tanh$ bounds its scores.) The lab shows the effect as a function of $d$.</p>`,
        lab: 'attention-transformers-scaling',
      },
      {
        title: 'Attention is a kernel smoother',
        html: R`
<p>Recall the <a href="#/t/kernel-regression">Nadaraya–Watson estimator</a>: given data $(x_j,y_j)$ and a kernel $K_h$, predict at $x$ by
$$\hat m(x) = \sum_j \frac{K_h(x-x_j)}{\sum_l K_h(x-x_l)}\,y_j.$$
Compare it with attention: the query is $x$, the keys are the $x_j$, the values are the $y_j$, and the kernel is $\kappa(\mathbf q,\mathbf k) = \exp s(\mathbf q,\mathbf k)$. Softmax attention <em>is</em> Nadaraya–Watson smoothing with a kernel built from the score function (Tsai et al., 2019). The query, key and value maps $W^Q,W^K,W^V$ learn the feature space in which to smooth, what counts as similar, and what to average.</p>
<div class="thm" data-kind="Proposition" data-name="Dot-product attention with equal-norm keys is Gaussian-kernel smoothing">
<p>If all keys have the same norm, $\lVert\mathbf k_j\rVert = r$, then for any $\tau>0$
$$\mathrm{softmax}_j\!\Big(\frac{\mathbf q^\top\mathbf k_j}{\tau}\Big) = \frac{\exp\!\big(-\lVert\mathbf q-\mathbf k_j\rVert^2/(2\tau)\big)}{\sum_l\exp\!\big(-\lVert\mathbf q-\mathbf k_l\rVert^2/(2\tau)\big)},$$
so dot-product attention is Nadaraya–Watson regression with a Gaussian kernel of bandwidth $h = \sqrt{\tau}$. The usual choice $\tau = \sqrt{d_k}$ is a bandwidth $h = d_k^{1/4}$ in key space.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$-\lVert\mathbf q - \mathbf k_j\rVert^2/(2\tau) = \mathbf q^\top\mathbf k_j/\tau - \lVert\mathbf q\rVert^2/(2\tau) - r^2/(2\tau)$. The last two terms do not depend on $j$, and softmax is invariant to adding a constant to every logit. $\square$</p>
</details>
<p>Keys never have exactly equal norms, but layer normalization pushes token vectors toward a common scale, so the Gaussian picture is a good first approximation. When norms differ, key $j$ gets an extra multiplicative weight $\exp(\lVert\mathbf k_j\rVert^2/(2\tau))$ — attention then behaves like a smoother with a <em>data-dependent prior weight</em> on each point, one mechanism behind "attention sinks" that soak up weight regardless of the query.</p>
<p>The kernel view imports everything we know about smoothing. The temperature is a bandwidth: too small and each query copies its nearest key (low bias, high variance — a hard lookup); too large and every query returns the global mean of the values (high bias). With one-dimensional positions as keys and a unit-circle embedding $\mathbf k = (\cos\omega x, \sin\omega x)$, the score $\beta\,\mathbf q^\top\mathbf k = \beta\cos(\omega(x_q - x))$ is a von Mises kernel; since $\cos u\approx 1-u^2/2$, it is approximately Gaussian with $h = 1/(\omega\sqrt\beta)$ for nearby points. In contrast, a raw scalar dot product $\beta\,x_q x$ is <em>not</em> a local kernel at all: it is monotone in $x$, so every query attends to the extremes. The lab makes all three concrete.</p>
<div class="note">
<p>This view also explains <em>linear attention</em> (Katharopoulos et al., 2020): replace $\exp(\mathbf q^\top\mathbf k)$ by a kernel with a finite feature map, $\kappa(\mathbf q,\mathbf k) = \varphi(\mathbf q)^\top\varphi(\mathbf k)$. Then $\sum_j \kappa(\mathbf q,\mathbf k_j)\mathbf v_j^\top = \varphi(\mathbf q)^\top\big(\sum_j\varphi(\mathbf k_j)\mathbf v_j^\top\big)$, the bracket is computed once, and cost becomes linear in $n$ — the same trick as random-feature approximations in <a href="#/t/kernel-methods">kernel methods</a>.</p>
</div>`,
        lab: 'attention-transformers-kernel',
      },
      {
        title: 'Self-attention, multiple heads, and the symmetry problem',
        html: R`
<p>In <strong>self-attention</strong>, queries, keys and values all come from the same sequence $X\in\R^{n\times d}$ (one row per token): $Q = XW^Q$, $K = XW^K$, $V = XW^V$. A <strong>multi-head</strong> layer runs $h$ attentions in parallel in lower-dimensional subspaces ($d_k = d_v = d/h$) and recombines:
$$\mathrm{MHA}(X) = \big[\mathrm{head}_1,\dots,\mathrm{head}_h\big]W^O,\qquad \mathrm{head}_i = \mathrm{Attn}(XW_i^Q, XW_i^K, XW_i^V).$$
With $d_k = d/h$ the parameter count is $4d^2$ (plus biases), the same as a single full-width head, but each head can implement a different kernel — one may attend to the previous token, another to syntactically related words. A single softmax produces one convex combination per query; multiple heads let one token gather several different averages at once.</p>
<div class="thm" data-kind="Theorem" data-name="Self-attention is permutation equivariant">
<p>Let $P$ be an $n\times n$ permutation matrix and $\mathrm{SA}(X) = \mathrm{softmax}(XW^Q(XW^K)^\top/\sqrt{d_k})\,XW^V$. Then $\mathrm{SA}(PX) = P\,\mathrm{SA}(X)$. The same holds for multi-head attention and for any network built from self-attention, row-wise maps (MLPs, layer normalization) and residual connections.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Write $S = XW^Q W^{K\top}X^\top/\sqrt{d_k}$. Replacing $X$ by $PX$ gives scores $PSP^\top$, whose $(i,j)$ entry is $S_{\pi(i)\pi(j)}$ for the permutation $\pi$ represented by $P$. Row-wise softmax commutes with permuting rows and with permuting columns within every row, so $\mathrm{softmax}(PSP^\top) = P\,\mathrm{softmax}(S)\,P^\top$. Then $\mathrm{SA}(PX) = P\,\mathrm{softmax}(S)P^\top P XW^V = P\,\mathrm{softmax}(S)XW^V = P\,\mathrm{SA}(X)$, using $P^\top P = I$. Concatenation of heads and right-multiplication by $W^O$ act on columns and preserve the identity. Row-wise maps $f$ satisfy $f(PX) = Pf(X)$ trivially, residual sums of equivariant maps are equivariant, and compositions of equivariant maps are equivariant. $\square$</p>
</details>
<p>So a transformer without positional information treats its input as a <em>set</em> (or multiset): "dog bites man" and "man bites dog" yield the same token representations, permuted. This is perfect for genuinely set-valued data and fatal for language. Position must be injected.</p>
<div class="def" data-name="Sinusoidal positional encoding (Vaswani et al., 2017)">
<p>Add to token $p$'s embedding the vector $\mathrm{PE}_p$ with $\mathrm{PE}_{p,2i} = \sin(\omega_i p)$, $\mathrm{PE}_{p,2i+1} = \cos(\omega_i p)$, $\omega_i = 10000^{-2i/d}$.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Relative offsets are linear in the encoding">
<p>For each frequency, $(\sin\omega(p+\Delta), \cos\omega(p+\Delta))^\top = R_\Delta\,(\sin\omega p,\cos\omega p)^\top$ with $R_\Delta = \begin{pmatrix}\cos\omega\Delta & \sin\omega\Delta\\ -\sin\omega\Delta & \cos\omega\Delta\end{pmatrix}$, independent of $p$. Hence $\mathrm{PE}_{p+\Delta} = M_\Delta\mathrm{PE}_p$ for a fixed block-diagonal rotation $M_\Delta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Angle-addition formulas: $\sin(a+b) = \sin a\cos b + \cos a\sin b$ and $\cos(a+b) = -\sin a\sin b + \cos a\cos b$ with $a = \omega p$, $b=\omega\Delta$. $\square$</p>
</details>
<p>So a linear map in the query/key projections can, in principle, express "attend $\Delta$ positions back". <strong>Rotary embeddings</strong> (RoPE; Su et al., 2021) make relative position exact: rotate each 2-D slice of the query at position $p$ by $R_p$ and of the key at position $p'$ by $R_{p'}$. Since $R_p^\top R_{p'} = R_{p'-p}$, the score $(R_p\mathbf q)^\top(R_{p'}\mathbf k) = \mathbf q^\top R_{p'-p}\mathbf k$ depends on positions only through the offset — the attention kernel becomes translation-equivariant along the sequence, like a <a href="#/t/cnn">convolution</a> with a content-dependent kernel.</p>`,
      },
      {
        title: 'Causal masking and the transformer block',
        html: R`
<p>For autoregressive modelling, $p(x_1,\dots,x_n) = \prod_i p(x_i\mid x_{\lt i})$, the representation at position $i$ must not see positions $j>i$. A <strong>causal mask</strong> adds $M_{ij} = -\infty$ for $j>i$ (and 0 otherwise) to the scores before the softmax, so $A_{ij} = 0$ for $j>i$.</p>
<div class="thm" data-kind="Proposition" data-name="Causal masking enforces autoregressive structure">
<p>In a stack of causally masked self-attention layers interleaved with row-wise maps and residual connections, output row $i$ is a function of input rows $1,\dots,i$ only.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Induction on depth. Inputs trivially satisfy the claim. If every row $j$ of $Z$ depends only on inputs $1..j$, then the masked attention output at row $i$ is $\sum_{j\le i}A_{ij}\mathbf v_j$, where $A_{ij}$ depends on $\mathbf q_i$ and $\mathbf k_1,\dots,\mathbf k_i$ (the normalization runs only over unmasked $j\le i$) and $\mathbf v_j$ on row $j$ of $Z$; all involve inputs $1..i$ only. Row-wise maps and residual additions preserve the property. $\square$</p>
</details>
<p>This lets one forward pass compute all $n$ next-token predictions of a training sequence in parallel — teacher forcing for free — while generation proceeds one token at a time (caching past keys and values). Note that the mask also breaks permutation equivariance: order now matters even without positional encodings.</p>
<div class="def" data-name="Transformer block (pre-norm form)">
<p>$$\begin{aligned} X' &= X + \mathrm{MHA}(\mathrm{LN}(X)),\\ Y &= X' + \mathrm{FFN}(\mathrm{LN}(X')),\qquad \mathrm{FFN}(\mathbf x) = W_2\,\phi(W_1\mathbf x + \mathbf b_1) + \mathbf b_2,\end{aligned}$$
with $\mathrm{LN}(\mathbf x) = \boldsymbol\gamma\odot(\mathbf x - \bar x\mathbf 1)/\sqrt{s_x^2 + \epsilon} + \boldsymbol\beta$ (mean and variance over the $d$ features of one token), $W_1\in\R^{4d\times d}$, and $\phi$ = ReLU or GELU. The original paper used post-norm, $\mathrm{LN}(X + \mathrm{Sublayer}(X))$, which is harder to train deep without learning-rate warm-up.</p>
</div>
<p>Each block costs about $4d^2$ (attention) $+\,8d^2$ (FFN) $= 12d^2$ parameters. The roles are complementary: attention <em>moves information between positions</em> (it is the only operation that mixes rows), the FFN <em>transforms information within a position</em>. The residual stream — the running sum $X, X', Y,\dots$ — is the additive path that keeps the end-to-end Jacobian near $I + \cdots$, just as in ResNets and the LSTM cell.</p>
<div class="ex" data-name="Counting">
<p>A model with $d = 768$, 12 layers, 12 heads and a vocabulary of $50{,}257$ (the GPT-2 small configuration): per layer $12\cdot768^2\approx 7.1$M, so about 85M in the blocks, plus a $50{,}257\times768\approx 38.6$M token-embedding matrix shared with the output layer and a small position table — roughly 124M in total. At $n=1024$ tokens, each head's attention matrix has about $10^6$ entries per layer.</p>
</div>`,
      },
      {
        title: 'Cost, comparisons and pitfalls',
        html: R`
<table>
<thead><tr><th>Layer type</th><th>Work per layer</th><th>Sequential steps</th><th>Max path length</th></tr></thead>
<tbody>
<tr><td>Self-attention</td><td>$O(n^2 d)$</td><td>$O(1)$</td><td>$O(1)$</td></tr>
<tr><td>Recurrent</td><td>$O(n d^2)$</td><td>$O(n)$</td><td>$O(n)$</td></tr>
<tr><td>Convolution (kernel $k$)</td><td>$O(k\,n d^2)$</td><td>$O(1)$</td><td>$O(n/k)$, or $O(\log_k n)$ dilated</td></tr>
</tbody>
</table>
<p>The $QK^\top$ product costs $O(n^2 d_k)$ per head and the attention matrix needs $O(n^2)$ memory, which dominates for long sequences. Remedies trade exactness or generality: sparse/local attention patterns, low-rank or kernelized (linear) attention, and FlashAttention (Dao et al., 2022), which is <em>exact</em> but tiles the computation so the $n\times n$ matrix never sits in slow memory.</p>
<div class="warn" data-name="Attention weights are not explanations">
<p>Attention maps are tempting to read as "what the model looked at", but different weight patterns can give the same output, and the values are transformed before and after (Jain &amp; Wallace, 2019). Use them as hypotheses, not evidence.</p>
</div>
<div class="warn" data-name="Length extrapolation">
<p>Learned absolute positions have no embedding beyond the training length, and even sinusoidal or rotary encodings meet unseen frequency combinations. Models often degrade sharply on longer inputs without specific adaptations.</p>
</div>
<div class="warn" data-name="Entropy collapse and rank collapse">
<p>If logits grow during training, attention becomes nearly one-hot and gradients through the softmax vanish (the saturation proposition again); normalizing queries and keys or capping logits helps. Conversely, pure attention without residuals and MLPs loses rank doubly-exponentially with depth (Dong, Cordonnier &amp; Loukas, 2021): repeated averaging drives all tokens to the same vector — the over-smoothing familiar from applying a kernel smoother again and again.</p>
</div>
<p>The broader lesson: the transformer replaces architectural priors (locality, recurrence) with a generic, learnable smoother plus position codes. With little data this is a weak prior and CNNs or RNNs often win; with very large data, the flexibility pays.</p>`,
      },
    ],
    quiz: [
      { q: R`Query and key coordinates are iid $\N(0,1)$ with $d_k=256$. What are the standard deviations of the logits with and without the $1/\sqrt{d_k}$ factor, and what happens to softmax over 100 keys in each case?`,
        a: R`Without: $\sqrt{256} = 16$. With: 1. Logits with sd 16 have a maximum many units above the runner-up, so softmax is essentially one-hot and its Jacobian $\diag(\mathbf p)-\mathbf p\mathbf p^\top$ is near zero; with sd 1 the weights are spread over many keys and gradients flow.` },
      { q: R`Show in one line that softmax attention with score $-\lVert\mathbf q-\mathbf k\rVert^2/(2h^2)$ equals dot-product attention with temperature $h^2$ when all keys have equal norm.`,
        a: R`$-\lVert\mathbf q-\mathbf k_j\rVert^2/(2h^2) = \mathbf q^\top\mathbf k_j/h^2 - (\lVert\mathbf q\rVert^2 + \lVert\mathbf k_j\rVert^2)/(2h^2)$; the subtracted term is constant in $j$ when $\lVert\mathbf k_j\rVert$ is constant, and softmax ignores additive constants.` },
      { q: R`Why can a transformer encoder without positional encodings not distinguish "the cat sat" from "sat the cat"? Name two things that would break this symmetry.`,
        a: R`Every component (self-attention, per-token FFN, layer norm, residuals) is permutation equivariant, so permuting the input tokens just permutes the outputs; any permutation-invariant readout (e.g. mean pooling) gives the same answer. Symmetry is broken by anything that treats positions differently: positional encodings (absolute, relative or rotary), or a causal mask, which depends on order.` },
      { q: R`Per layer, a transformer with $d = 1024$ processes $n = 8192$ tokens. Compare the cost of computing $QK^\top$ (all heads) with the cost of the FFN.`,
        a: R`$QK^\top$ over all heads: $n^2 d \approx 6.9\times10^{10}$ multiply-adds (and the same again for $AV$). FFN: two $d\times 4d$ matrices per token, $8nd^2 \approx 6.9\times10^{10}$. They are comparable at $n = 8d$; for longer sequences attention dominates quadratically.` },
      { q: R`What does the causal mask guarantee, and why does it allow parallel training?`,
        a: R`Output position $i$ depends only on inputs $1,\dots,i$ (proved by induction over layers). Therefore all $n$ conditional predictions $p(x_i\mid x_{\lt i})$ of a training sequence can be computed in a single forward pass and their losses summed, rather than running the model $n$ times.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  /*  Autoencoders & VAEs                                                */
  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'autoencoders-vae',
    domain: 'deep',
    order: 110,
    title: 'Autoencoders & Variational Autoencoders',
    subtitle: 'From compression to amortized variational inference in latent-variable models',
    level: 3,
    sources: { other: 'Kingma & Welling (2014); Rezende, Mohamed & Wierstra (2014); Baldi & Hornik (1989); Tipping & Bishop (1999); Goodfellow, Bengio & Courville (2016), Ch. 14 & §20.10.3; Prince, Understanding Deep Learning (2023), Ch. 17; Mohamed et al. (2020) on gradient estimators; Lucas et al. (2019) on posterior collapse' },
    prereqs: ['pca', 'variational-inference'],
    related: ['entropy-kl', 'em-algorithm', 'factor-analysis-ica', 'diffusion-models', 'gans'],
    tags: ['autoencoder', 'ELBO', 'reparameterization', 'latent variables', 'amortized inference', 'posterior collapse'],
    summary: R`An autoencoder learns an encoder $f$ and decoder $g$ so that $g(f(\mathbf x))\approx\mathbf x$ through a low-dimensional bottleneck; with linear maps and squared error it recovers exactly the principal subspace. The variational autoencoder turns this into a probabilistic generative model $p_\theta(\mathbf x) = \int p_\theta(\mathbf x\mid\mathbf z)p(\mathbf z)\d\mathbf z$, trained by maximizing the evidence lower bound $\E_{q_\phi}[\log p_\theta(\mathbf x\mid\mathbf z)] - \KL(q_\phi(\mathbf z\mid\mathbf x)\,\Vert\,p(\mathbf z))$, where the encoder $q_\phi$ is an amortized variational posterior. The reparameterization $\mathbf z = \boldsymbol\mu + \boldsymbol\sigma\odot\boldsymbol\epsilon$ moves the randomness out of the parameters so that low-variance pathwise gradients flow through the sampling step.`,
    survey: {
      task: 'Representation learning, compression, density modelling and generation',
      assumptions: R`Data lie near low-dimensional latent structure; the chosen likelihood $p_\theta(\mathbf x\mid\mathbf z)$ and variational family are adequate`,
      output: R`Encoder $q_\phi(\mathbf z\mid\mathbf x)$, decoder $p_\theta(\mathbf x\mid\mathbf z)$, a lower bound on $\log p_\theta(\mathbf x)$`,
      strengths: 'Principled likelihood bound; fast amortized inference; smooth, structured latent space; stable training',
      weaknesses: 'Blurry samples with factorized Gaussian decoders; ELBO gap; posterior collapse with strong decoders',
      complexity: R`One encoder + decoder pass per example per step; one sample of $\mathbf z$ usually suffices`,
    },
    sections: [
      {
        title: 'Autoencoders and what the bottleneck learns',
        html: R`
<p>An <strong>autoencoder</strong> consists of an encoder $f_\phi:\R^D\to\R^k$ and a decoder $g_\theta:\R^k\to\R^D$ trained to minimize reconstruction error, typically $\frac1n\sum_i\lVert\mathbf x_i - g_\theta(f_\phi(\mathbf x_i))\rVert^2$. If $k\lt D$ (<em>undercomplete</em>) the network cannot copy its input and must find a compressed code. Variants regularize instead of (or as well as) narrowing: <em>denoising</em> autoencoders reconstruct $\mathbf x$ from a corrupted $\tilde{\mathbf x}$ (Vincent et al., 2008), <em>sparse</em> autoencoders penalize code activity, <em>contractive</em> ones penalize $\lVert\partial f/\partial\mathbf x\rVert_F^2$.</p>
<p>The cleanest theory is for the linear case, which also anchors intuition for the nonlinear one.</p>
<div class="thm" data-kind="Theorem" data-name="Linear autoencoders find the principal subspace (Bourlard & Kamp, 1988; Baldi & Hornik, 1989)">
<p>Let $X\in\R^{n\times D}$ hold centred data with $\Sigma = X^\top X/n$ non-singular, eigenvalues $\lambda_1>\lambda_2>\cdots>\lambda_D>0$, eigenvectors $\mathbf u_1,\dots,\mathbf u_D$, and $U_k = [\mathbf u_1,\dots,\mathbf u_k]$. Consider
$$L(A,B) = \frac1n\sum_i\lVert\mathbf x_i - AB\,\mathbf x_i\rVert^2,\qquad A\in\R^{D\times k},\; B\in\R^{k\times D}.$$
Then $\min L = \sum_{j>k}\lambda_j$, attained exactly when $AB = U_kU_k^\top$; every minimizer has the form $A = U_kC$, $B = C^{-1}U_k^\top$ for an invertible $C\in\R^{k\times k}$. Every other critical point is a saddle.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>$L = \frac1n\lVert X - XM^\top\rVert_F^2$ with $M = AB$ of rank at most $k$; conversely any rank-$\le k$ matrix factors as $AB$. By Eckart–Young, the best rank-$\le k$ approximation of $X$ in Frobenius norm is its truncated SVD, $XV_kV_k^\top$, where $V_k$ holds the top right singular vectors of $X$ — the top eigenvectors of $\Sigma$, so $V_k = U_k$ — and the minimizer is unique because $\lambda_k>\lambda_{k+1}$. The error is $\frac1n\sum_{j>k}s_j^2 = \sum_{j>k}\lambda_j$. Since $XM^\top$ is a rank-$\le k$ matrix achieving this, $XM^\top = XU_kU_k^\top$, and full column rank of $X$ gives $M = U_kU_k^\top$. A rank-$k$ factorization $AB = U_kU_k^\top$ forces $\mathrm{col}(A) = \mathrm{col}(U_k)$, so $A = U_kC$ with $C$ invertible, and then $B = C^{-1}U_k^\top$. The saddle-point statement follows by examining the Hessian at critical points whose column space is spanned by any other set of $k$ eigenvectors (Baldi &amp; Hornik, 1989); we omit it. $\square$</p>
</details>
<p>Two lessons. First, the optimal linear autoencoder does exactly what <a href="#/t/pca">PCA</a> does — but it identifies only the <em>subspace</em>: the code $B\mathbf x = C^{-1}U_k^\top\mathbf x$ is an arbitrary invertible mixture of principal components, not the components themselves. Second, the loss landscape has no spurious local minima, so gradient descent reliably finds that subspace. The lab lets you watch it happen, and break it by making the data curved, where no linear subspace suffices.</p>
<div class="note">
<p>A nonlinear autoencoder generalizes PCA to a curved manifold, but nothing in the objective makes the code space well behaved: codes the encoder never produces may decode to garbage, so one cannot generate new data by sampling codes. Making the latent space a proper probability model is what the VAE adds.</p>
</div>`,
        lab: 'autoencoders-vae-pca',
      },
      {
        title: 'Latent-variable models and the evidence lower bound',
        html: R`
<p>Posit a generative model: draw $\mathbf z\sim p(\mathbf z) = \N(\mathbf 0, I_k)$, then $\mathbf x\sim p_\theta(\mathbf x\mid\mathbf z)$, where the likelihood's parameters are output by a neural network ("decoder"), e.g. $p_\theta(\mathbf x\mid\mathbf z) = \N(\mathbf x; g_\theta(\mathbf z), \sigma_x^2 I)$ or a product of Bernoullis. The marginal likelihood
$$p_\theta(\mathbf x) = \int p_\theta(\mathbf x\mid\mathbf z)\,p(\mathbf z)\d\mathbf z$$
is intractable for a nonlinear decoder, and so is the posterior $p_\theta(\mathbf z\mid\mathbf x)$. Maximum likelihood is out of reach directly; <a href="#/t/variational-inference">variational inference</a> provides a tractable surrogate.</p>
<div class="thm" data-kind="Theorem" data-name="The ELBO identity">
<p>For any density $q(\mathbf z)$ whose support is contained in that of $p_\theta(\cdot\mid\mathbf x)$,
$$\log p_\theta(\mathbf x) = \underbrace{\E_{q}\big[\log p_\theta(\mathbf x,\mathbf z) - \log q(\mathbf z)\big]}_{\mathrm{ELBO}(q,\theta)} + \KL\big(q(\mathbf z)\,\Vert\,p_\theta(\mathbf z\mid\mathbf x)\big),$$
and the ELBO can be rewritten as
$$\mathrm{ELBO} = \E_q[\log p_\theta(\mathbf x\mid\mathbf z)] - \KL\big(q(\mathbf z)\,\Vert\,p(\mathbf z)\big).$$
Since $\KL\ge0$, $\mathrm{ELBO}\le\log p_\theta(\mathbf x)$ with equality iff $q = p_\theta(\cdot\mid\mathbf x)$ a.e.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Using $p_\theta(\mathbf x,\mathbf z) = p_\theta(\mathbf z\mid\mathbf x)p_\theta(\mathbf x)$: $\E_q[\log p_\theta(\mathbf x,\mathbf z) - \log q(\mathbf z)] = \log p_\theta(\mathbf x) + \E_q[\log p_\theta(\mathbf z\mid\mathbf x) - \log q(\mathbf z)] = \log p_\theta(\mathbf x) - \KL(q\Vert p_\theta(\cdot\mid\mathbf x))$. For the second form, write $\log p_\theta(\mathbf x,\mathbf z) = \log p_\theta(\mathbf x\mid\mathbf z) + \log p(\mathbf z)$ and group $\E_q[\log p(\mathbf z) - \log q(\mathbf z)] = -\KL(q\Vert p)$. Non-negativity of KL, with equality iff the densities agree a.e., is Gibbs' inequality (<a href="#/t/entropy-kl">entropy &amp; KL</a>). $\square$</p>
</details>
<p>The two readings of the ELBO matter. As $\log p - \KL(q\Vert\text{posterior})$, maximizing over $q$ is posterior approximation, and maximizing over $\theta$ is approximate maximum likelihood — this is the <a href="#/t/em-algorithm">EM algorithm</a> when $q$ can be the exact posterior, and "variational EM" otherwise. As reconstruction minus a KL-to-prior, it is an autoencoder: the encoder proposes codes $\mathbf z\sim q$, the decoder must reconstruct $\mathbf x$ from them, and the KL term is a regularizer keeping the codes close to the prior so that sampling $\mathbf z\sim p(\mathbf z)$ later gives sensible outputs.</p>
<div class="def" data-name="Variational autoencoder (Kingma & Welling, 2014; Rezende et al., 2014)">
<p>Use an <em>amortized</em> variational family: a neural network ("encoder") maps each $\mathbf x$ to the parameters of $q_\phi(\mathbf z\mid\mathbf x) = \N\big(\boldsymbol\mu_\phi(\mathbf x),\diag\boldsymbol\sigma^2_\phi(\mathbf x)\big)$. Train $(\theta,\phi)$ jointly by stochastic gradient ascent on $\sum_i\mathrm{ELBO}(\mathbf x_i;\theta,\phi)$.</p>
</div>
<p>Amortization replaces one optimization per data point (classical VI) with one network shared by all of them. The price is the <em>amortization gap</em>: the encoder may not output the best Gaussian for each $\mathbf x$, so the bound is looser than per-point VI would give (Cremer, Li &amp; Duvenaud, 2018).</p>`,
      },
      {
        title: 'The reparameterization trick',
        html: R`
<p>Training needs $\nabla_\phi\,\E_{q_\phi(\mathbf z)}[F(\mathbf z)]$ where $F(\mathbf z) = \log p_\theta(\mathbf x\mid\mathbf z)$ (the KL term is analytic; see below). The difficulty is that $\phi$ appears in the <em>distribution</em> being averaged over. There are two standard unbiased estimators.</p>
<div class="cols">
<div><h4>Score function (REINFORCE)</h4>$$\nabla_\phi\E_{q_\phi}[F] = \E_{q_\phi}\big[F(\mathbf z)\,\nabla_\phi\log q_\phi(\mathbf z)\big]$$<p>From $\nabla q = q\nabla\log q$. Needs only evaluations of $F$; works for discrete $\mathbf z$.</p></div>
<div><h4>Pathwise (reparameterization)</h4>$$\nabla_\phi\E_{q_\phi}[F] = \E_{\boldsymbol\epsilon\sim\N(0,I)}\big[\nabla_\phi F(\boldsymbol\mu_\phi + \boldsymbol\sigma_\phi\odot\boldsymbol\epsilon)\big]$$<p>Write $\mathbf z = h_\phi(\boldsymbol\epsilon)$ with parameter-free noise; differentiate inside. Needs $F$ differentiable in $\mathbf z$.</p></div>
</div>
<p>The pathwise form is valid because the distribution of $\boldsymbol\epsilon$ does not depend on $\phi$, so $\E_{q_\phi}[F(\mathbf z)] = \E_{\boldsymbol\epsilon}[F(h_\phi(\boldsymbol\epsilon))]$ and the gradient passes inside the expectation (dominated convergence, given an integrable bound on $\nabla_\phi F(h_\phi(\boldsymbol\epsilon))$). With one sample of $\boldsymbol\epsilon$ per data point, the whole ELBO becomes an ordinary differentiable function of $(\theta,\phi)$ and <a href="#/t/backpropagation">backpropagation</a> runs through the sampler: $\partial\mathbf z/\partial\boldsymbol\mu = I$, $\partial\mathbf z/\partial\boldsymbol\sigma = \diag(\boldsymbol\epsilon)$.</p>
<div class="thm" data-kind="Proposition" data-name="Why the pathwise estimator usually has lower variance">
<p>Take $q = \N(\mu,\sigma^2)$ in one dimension and consider the gradient with respect to $\mu$, with $\epsilon\sim\N(0,1)$.</p>
<ol>
<li>Pathwise: $g_{\text{rep}} = F'(\mu+\sigma\epsilon)$. If $|F'|\le L$ then $\Var(g_{\text{rep}})\le L^2$, uniformly in $\mu$ and $\sigma$; if $F$ is linear, $\Var(g_{\text{rep}}) = 0$.</li>
<li>Score function: $g_{\text{sf}} = F(\mu+\sigma\epsilon)\,\epsilon/\sigma$, with $\E[g_{\text{sf}}^2] = \E[F(\mu+\sigma\epsilon)^2\epsilon^2]/\sigma^2$, which diverges like $F(\mu)^2/\sigma^2$ as $\sigma\to0$ whenever $F(\mu)\ne0$. For $F(z) = az$: $\Var(g_{\text{sf}}) = a^2(2 + \mu^2/\sigma^2)$, versus $0$.</li>
</ol>
</div>
<details class="proof"><summary>Proof</summary>
<p>(1) A random variable bounded in absolute value by $L$ has variance at most $\E[g^2]\le L^2$. (2) $\nabla_\mu\log q(z) = (z-\mu)/\sigma^2 = \epsilon/\sigma$. For $F(z)=az$: $g_{\text{sf}} = a(\mu+\sigma\epsilon)\epsilon/\sigma = a(\mu\epsilon/\sigma + \epsilon^2)$, so $\Var(g_{\text{sf}}) = a^2\big(\tfrac{\mu^2}{\sigma^2}\Var\epsilon + \Var(\epsilon^2) + 2\tfrac{\mu}{\sigma}\Cov(\epsilon,\epsilon^2)\big) = a^2(\mu^2/\sigma^2 + 2)$, using $\Var(\epsilon^2) = 2$ and $\Cov(\epsilon,\epsilon^2) = \E\epsilon^3 = 0$. For general continuous $F$ with a polynomial growth bound, dominated convergence gives $\E[F(\mu+\sigma\epsilon)^2\epsilon^2]\to F(\mu)^2\E\epsilon^2 = F(\mu)^2$ as $\sigma\to0$, while the mean $\E g_{\text{sf}} = \E F'(\mu+\sigma\epsilon)$ (Stein's lemma) stays bounded; so the variance grows like $F(\mu)^2/\sigma^2$. $\square$</p>
</details>
<p>The intuition: the score-function estimator learns about the gradient only through <em>which</em> samples happened to get high $F$ values — a correlation it must estimate from noise. The pathwise estimator uses the derivative $F'$ directly: it knows in which direction moving each sample would increase $F$. Control variates (subtracting a baseline $b$ from $F$, which leaves the expectation unchanged because $\E[\nabla\log q] = 0$) shrink the score-function variance but rarely close the gap. Neither estimator dominates universally — for some highly oscillatory $F$ the pathwise variance is larger (Mohamed et al., 2020) — but for smooth decoders the advantage is decisive. The lab measures both on an exactly solvable model.</p>`,
        lab: 'autoencoders-vae-elbo',
      },
      {
        title: 'The Gaussian KL term and the full objective',
        html: R`
<div class="thm" data-kind="Proposition" data-name="KL between a diagonal Gaussian and the standard normal">
<p>$$\KL\big(\N(\boldsymbol\mu,\diag\boldsymbol\sigma^2)\,\Vert\,\N(\mathbf 0,I_k)\big) = \frac12\sum_{j=1}^k\big(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\big).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Both densities factorize over coordinates, so the KL is a sum of one-dimensional terms. For one coordinate with $q = \N(\mu,\sigma^2)$, $p=\N(0,1)$:
$\E_q[\log q - \log p] = \E_q\big[-\tfrac12\log(2\pi\sigma^2) - \tfrac{(z-\mu)^2}{2\sigma^2} + \tfrac12\log 2\pi + \tfrac{z^2}{2}\big] = -\tfrac12\log\sigma^2 - \tfrac12 + \tfrac12(\mu^2+\sigma^2)$, using $\E_q(z-\mu)^2 = \sigma^2$ and $\E_qz^2 = \mu^2+\sigma^2$. $\square$</p>
</details>
<p>Each term is minimized at $\mu_j=0$, $\sigma_j=1$: the KL pulls every code toward the prior, penalizing both displacement ($\mu_j^2$) and over-confidence ($-\log\sigma_j^2$ blows up as $\sigma_j\to0$). The reconstruction term pulls the other way — it wants distinct, precise codes for distinct inputs. The ELBO's balance between them is the VAE.</p>
<div class="algo" data-name="VAE training step (Gaussian decoder)">
<ol>
<li>Sample a minibatch $\mathbf x_1,\dots,\mathbf x_B$. Encode: $(\boldsymbol\mu_i,\log\boldsymbol\sigma_i^2) = \mathrm{enc}_\phi(\mathbf x_i)$.</li>
<li>Reparameterize: $\boldsymbol\epsilon_i\sim\N(\mathbf 0,I)$, $\mathbf z_i = \boldsymbol\mu_i + \boldsymbol\sigma_i\odot\boldsymbol\epsilon_i$.</li>
<li>Decode $\hat{\mathbf x}_i = g_\theta(\mathbf z_i)$ and form $\widehat{\mathrm{ELBO}} = \frac1B\sum_i\Big[-\frac{\lVert\mathbf x_i - \hat{\mathbf x}_i\rVert^2}{2\sigma_x^2} - \frac D2\log(2\pi\sigma_x^2) - \frac12\sum_j(\mu_{ij}^2 + \sigma_{ij}^2 - 1 - \log\sigma_{ij}^2)\Big]$.</li>
<li>Backpropagate through decoder, the sampling step and encoder; take an optimizer step on $-\widehat{\mathrm{ELBO}}$.</li>
</ol>
</div>
<p>The decoder variance $\sigma_x^2$ sets the exchange rate between reconstruction and KL: small $\sigma_x^2$ makes the model an almost-deterministic autoencoder; large $\sigma_x^2$ lets the KL win and the code shrinks toward the prior. Weighting the KL by $\beta\ne1$ ($\beta$-VAE; Higgins et al., 2017) is, for a fixed Gaussian decoder variance, equivalent to rescaling $\sigma_x^2$, and trades reconstruction for "disentanglement". Only $\beta = 1$ gives a bound on $\log p_\theta(\mathbf x)$.</p>
<div class="ex" data-name="The linear-Gaussian VAE is probabilistic PCA">
<p>With a linear decoder $p(\mathbf x\mid\mathbf z) = \N(W\mathbf z + \mathbf b,\sigma^2I)$ the marginal is $\N(\mathbf b, WW^\top + \sigma^2 I)$ — probabilistic PCA (Tipping &amp; Bishop, 1999). The posterior is Gaussian, so a Gaussian encoder can be exact and the ELBO is tight. The maximum-likelihood $W$ is $U_k(\Lambda_k - \sigma^2I)_+^{1/2}R$ for an orthogonal $R$: columns aligned with eigenvalues $\lambda_j\le\sigma^2$ are <em>zero</em>. Those latent dimensions are switched off — the decoder ignores them and their posterior equals the prior — which is the simplest exact instance of posterior collapse (Lucas et al., 2019).</p>
</div>`,
      },
      {
        title: 'Posterior collapse and other pathologies',
        html: R`
<div class="def" data-name="Posterior collapse">
<p>Some or all latent coordinates satisfy $q_\phi(z_j\mid\mathbf x)\approx p(z_j)$ for all $\mathbf x$: they carry no information about the input, contribute zero KL, and are ignored by the decoder.</p>
</div>
<p>Why does the ELBO permit this? The dataset-averaged KL term decomposes as $\E_{\mathbf x}\KL(q(\mathbf z\mid\mathbf x)\Vert p(\mathbf z)) = I_q(\mathbf x;\mathbf z) + \KL(\bar q(\mathbf z)\Vert p(\mathbf z))$, where $\bar q$ is the aggregate posterior $\E_{\mathbf x}q(\mathbf z\mid\mathbf x)$ and $I_q$ the mutual information between data and code under the encoder (Hoffman &amp; Johnson, 2016). The ELBO thus <em>charges</em> for every nat of information the code carries. If the decoder can model $\mathbf x$ well without $\mathbf z$ — as a powerful autoregressive decoder (PixelCNN, an LSTM language model) can — then using $\mathbf z$ buys little reconstruction and costs KL, and the optimum is to ignore it. Collapse is then not an optimization failure but the correct answer to the stated objective; in the linear case above it is the exact maximum-likelihood solution.</p>
<div class="warn" data-name="Remedies change the objective">
<p>KL annealing (ramp the KL weight from 0 to 1; Bowman et al., 2016), "free bits" (do not penalize KL below a threshold per dimension), weakening the decoder, or constraining $q$ so its KL cannot be zero ($\delta$-VAE) all work by moving away from the plain ELBO. Decide first whether you need an informative latent code or a good density model; they are different goals.</p>
</div>
<div class="warn" data-name="Blurry samples">
<p>A factorized Gaussian decoder treats pixels as conditionally independent given $\mathbf z$; uncertainty about fine detail is expressed as averaging, which looks like blur. Maximum likelihood (and the ELBO) minimizes the forward KL $\KL(p_{\text{data}}\Vert p_\theta)$, which is mass-covering and punishes missing any data mode, whereas GANs, trained toward mode-seeking objectives, produce sharper but less diverse samples. Hierarchical VAEs (NVAE, VDVAE) and diffusion decoders largely close the quality gap.</p>
</div>
<div class="warn" data-name="Holes in the latent space">
<p>The aggregate posterior $\bar q(\mathbf z)$ need not match the prior. Regions of high prior but low aggregate-posterior density decode to atypical outputs. Learned or mixture priors (VampPrior) and two-stage VAEs address this.</p>
</div>
<p>Finally, the ELBO is a lower bound, so comparing models by ELBO compares bounds, not likelihoods. Importance-weighted bounds $\E\log\frac1K\sum_{k}p(\mathbf x,\mathbf z_k)/q(\mathbf z_k\mid\mathbf x)$ (Burda, Grosse &amp; Salakhutdinov, 2016) are non-decreasing in $K$ and converge to $\log p(\mathbf x)$ under mild conditions, which makes them the standard for evaluation.</p>`,
      },
      {
        title: 'Where VAEs sit among generative models',
        html: R`
<table>
<thead><tr><th>Model</th><th>Likelihood</th><th>Sampling</th><th>Latent code</th></tr></thead>
<tbody>
<tr><td>VAE</td><td>Lower bound (ELBO)</td><td>One decoder pass</td><td>Yes, with encoder</td></tr>
<tr><td><a href="#/t/gans">GAN</a></td><td>None</td><td>One generator pass</td><td>Yes, no encoder</td></tr>
<tr><td>Normalizing flow</td><td>Exact</td><td>One (invertible) pass</td><td>Same dimension as data</td></tr>
<tr><td>Autoregressive</td><td>Exact</td><td>Sequential, $D$ steps</td><td>No</td></tr>
<tr><td><a href="#/t/diffusion-models">Diffusion</a></td><td>Bound (or exact via ODE)</td><td>Many denoising steps</td><td>Noise of data dimension</td></tr>
</tbody>
</table>
<p>A diffusion model is, formally, a hierarchical VAE with a <em>fixed</em> encoder (adding Gaussian noise) and many latent layers; its training objective is an ELBO whose terms are Gaussian KLs of exactly the kind derived above. Latent diffusion models put a diffusion model inside the latent space of an autoencoder, combining the compression of the first with the sample quality of the second.</p>`,
      },
    ],
    quiz: [
      { q: R`A linear autoencoder with a 2-dimensional bottleneck is trained to optimality on centred data with covariance eigenvalues $5, 3, 1, 0.5$. What is its mean squared reconstruction error, and are its two code coordinates the first two principal component scores?`,
        a: R`$1 + 0.5 = 1.5$, the sum of discarded eigenvalues. Not necessarily: the optimum is $AB = U_2U_2^\top$, and $B = C^{-1}U_2^\top$ for any invertible $C$, so the code is an arbitrary invertible linear mixture of the first two PC scores. Only the subspace is identified.` },
      { q: R`Derive $\log p(\mathbf x) - \mathrm{ELBO}(q)$ and explain what maximizing the ELBO over $q$ alone accomplishes.`,
        a: R`$\log p(\mathbf x) - \mathrm{ELBO}(q) = \KL(q(\mathbf z)\Vert p(\mathbf z\mid\mathbf x))\ge 0$. Since $\log p(\mathbf x)$ does not depend on $q$, maximizing the ELBO over $q$ is the same as minimizing the KL from $q$ to the exact posterior — it is posterior approximation, and the bound becomes tight when $q$ equals the posterior.` },
      { q: R`Compute $\KL(\N(1, 0.25)\Vert\N(0,1))$, where $0.25$ is the variance.`,
        a: R`$\tfrac12(\mu^2+\sigma^2-1-\log\sigma^2) = \tfrac12(1 + 0.25 - 1 - \log 0.25) = \tfrac12(0.25 + 1.386) \approx 0.818$ nats.` },
      { q: R`Why does the reparameterization trick not work directly for a categorical latent variable, and what is used instead?`,
        a: R`Writing $z = h_\phi(\epsilon)$ for a discrete $z$ requires a step function of $\phi$, whose derivative is zero almost everywhere, so pathwise gradients carry no information. Options: the score-function estimator with baselines/control variates, or continuous relaxations such as Gumbel-softmax (Concrete), which give biased but low-variance pathwise gradients.` },
      { q: R`A VAE with an autoregressive LSTM decoder reaches a good ELBO but its KL term is 0.01 nats. What happened, and why is it consistent with the objective?`,
        a: R`Posterior collapse: the encoder outputs (nearly) the prior for every input and the decoder ignores $\mathbf z$. The averaged KL term equals the mutual information $I_q(\mathbf x;\mathbf z)$ plus a non-negative term, so the ELBO charges for information carried by $\mathbf z$. If the autoregressive decoder can model $\mathbf x$ without it, the optimum sets $I_q\approx 0$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  /*  GANs                                                               */
  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'gans',
    domain: 'deep',
    order: 120,
    title: 'Generative Adversarial Networks',
    subtitle: 'Learning a sampler by playing it against a classifier',
    level: 3,
    sources: { other: 'Goodfellow et al. (2014); Arjovsky & Bottou (2017); Arjovsky, Chintala & Bottou (2017); Gulrajani et al. (2017); Nowozin, Cseke & Tomioka (2016); Goodfellow, Bengio & Courville (2016), §20.10.4; Prince, Understanding Deep Learning (2023), Ch. 15; Villani, Optimal Transport (2009) for Kantorovich–Rubinstein' },
    prereqs: ['entropy-kl', 'perceptron-mlp'],
    related: ['autoencoders-vae', 'diffusion-models', 'logistic-regression', 'sgd-optimizers'],
    tags: ['implicit models', 'minimax', 'Jensen–Shannon', 'mode collapse', 'Wasserstein', 'optimal transport'],
    summary: R`A GAN trains a generator $G$ that pushes simple noise $\mathbf z$ forward to samples $G(\mathbf z)$, without ever writing down a density. A discriminator $D$ is trained to tell real data from generated samples, and the generator is trained to fool it. For a fixed generator the best discriminator is $D^* = p_{\text{data}}/(p_{\text{data}} + p_g)$ — a density-ratio estimator — and plugging it in shows that the generator is minimizing the Jensen–Shannon divergence to the data. That divergence saturates when the two distributions barely overlap, which, together with the instability of simultaneous gradient play, causes vanishing gradients and mode collapse. Wasserstein GANs replace JS by the earth-mover distance, whose Kantorovich–Rubinstein dual is a supremum over 1-Lipschitz critics and which varies smoothly even between disjoint distributions.`,
    survey: {
      task: 'Sampling from complex high-dimensional distributions (images, audio); domain translation',
      assumptions: 'Generator and discriminator classes rich enough; the two-player game can be approximately solved by alternating SGD',
      output: R`A generator $G:\mathbf z\mapsto\mathbf x$ (implicit model; no density)`,
      strengths: 'Very sharp samples; single-pass generation; no likelihood or posterior needed; flexible losses (f-divergences, IPMs)',
      weaknesses: 'Unstable minimax training; mode collapse; no likelihood for evaluation; sensitive to architecture and hyperparameters',
      complexity: R`One $D$ and one $G$ forward/backward pass per step (often $k$ critic steps per generator step)`,
    },
    sections: [
      {
        title: 'Implicit models and the classifier trick',
        html: R`
<p>Likelihood-based models (<a href="#/t/autoencoders-vae">VAEs</a>, flows, autoregressive models) need a tractable density or a bound on it. A GAN dispenses with densities entirely. It specifies a <strong>generator</strong> $G_\theta:\R^k\to\R^D$ and a noise distribution $p(\mathbf z)$ (Gaussian or uniform); the model distribution $p_g$ is the pushforward, the law of $G_\theta(\mathbf z)$. When $k\lt D$, $p_g$ is concentrated on a $k$-dimensional set and has no density on $\R^D$ at all. Sampling is trivial; evaluating $p_g(\mathbf x)$ is impossible.</p>
<p>How can one fit a model one cannot evaluate? By comparing samples. The key observation, older than GANs, is that <em>classification estimates density ratios</em>. Mix real data (label 1) and model samples (label 0) in equal proportion. The Bayes-optimal probability of label 1 at $\mathbf x$ is
$$\P(y=1\mid\mathbf x) = \frac{p_{\text{data}}(\mathbf x)}{p_{\text{data}}(\mathbf x) + p_g(\mathbf x)},$$
so a well-trained probabilistic classifier (e.g. <a href="#/t/logistic-regression">logistic regression</a> or a neural net trained with cross-entropy) estimates $p_{\text{data}}/p_g$ through its odds. If the classifier cannot do better than chance, the two distributions are indistinguishable to it. The GAN turns this into a training signal for the generator.</p>
<div class="note">
<p>A GAN is a learned two-sample test whose test statistic is fed back to the generator. The discriminator is the critic; the generator updates to make the critic's job harder; at the ideal end point the test has no power.</p>
</div>`,
      },
      {
        title: 'The minimax game and the optimal discriminator',
        html: R`
<div class="def" data-name="GAN objective (Goodfellow et al., 2014)">
<p>$$\min_G\max_D\; V(D,G) = \E_{\mathbf x\sim p_{\text{data}}}\big[\log D(\mathbf x)\big] + \E_{\mathbf z\sim p(\mathbf z)}\big[\log\big(1 - D(G(\mathbf z))\big)\big].$$
$D$ maximizes the log-likelihood of correctly labelling real (1) and fake (0) samples — the negative binary cross-entropy. $G$ minimizes the same quantity.</p>
</div>
<div class="thm" data-kind="Proposition" data-name="Optimal discriminator">
<p>For fixed $G$, assume $p_{\text{data}}$ and $p_g$ have densities. The discriminator maximizing $V(D,G)$ over all measurable $D:\R^D\to[0,1]$ is
$$D^*_G(\mathbf x) = \frac{p_{\text{data}}(\mathbf x)}{p_{\text{data}}(\mathbf x) + p_g(\mathbf x)}$$
on the union of the supports (and arbitrary elsewhere).</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By the change of variables $\mathbf x = G(\mathbf z)$ in the second expectation, $V(D,G) = \int\big[p_{\text{data}}(\mathbf x)\log D(\mathbf x) + p_g(\mathbf x)\log(1-D(\mathbf x))\big]\d\mathbf x$. The integrand can be maximized pointwise. For $a,b\ge0$ not both zero, $y\mapsto a\log y + b\log(1-y)$ on $[0,1]$ is strictly concave with derivative $a/y - b/(1-y)$, which vanishes at $y = a/(a+b)$. Take $a = p_{\text{data}}(\mathbf x)$, $b = p_g(\mathbf x)$. $\square$</p>
</details>
<div class="thm" data-kind="Theorem" data-name="The generator minimizes Jensen–Shannon divergence">
<p>Let $C(G) = \max_D V(D,G) = V(D^*_G,G)$. Then
$$C(G) = -\log 4 + 2\,\mathrm{JS}(p_{\text{data}}\,\Vert\,p_g),\qquad \mathrm{JS}(P\Vert Q) = \tfrac12\KL(P\Vert M) + \tfrac12\KL(Q\Vert M),\;\; M = \tfrac12(P+Q).$$
Hence the global minimum of $C$ is $-\log 4$, attained if and only if $p_g = p_{\text{data}}$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Substitute $D^*$: $C(G) = \E_{p_{\text{data}}}\log\frac{p_{\text{data}}}{p_{\text{data}}+p_g} + \E_{p_g}\log\frac{p_g}{p_{\text{data}}+p_g}$. Since $p_{\text{data}} + p_g = 2M$, each log equals $\log\frac{p_{\text{data}}}{M} - \log 2$ (resp. $\log\frac{p_g}{M}-\log 2$). Hence $C(G) = -2\log 2 + \KL(p_{\text{data}}\Vert M) + \KL(p_g\Vert M) = -\log 4 + 2\,\mathrm{JS}$. JS is non-negative and zero iff $P = Q$ (it is a sum of KLs, each zero iff its arguments agree, and $P = M$ iff $P = Q$). $\square$</p>
</details>
<p>The theorem is a statement about the <em>outer</em> problem with the inner one solved exactly, over all functions. Goodfellow et al. also showed that if $p_g$ itself is updated in function space with $D$ kept optimal, it converges to $p_{\text{data}}$, because $C$ is convex in $p_g$. Neither condition holds in practice: $D$ and $G$ are parametric networks, $D$ is only partially optimized between generator steps, and the game is non-convex in parameters. The rest of this topic is about what goes wrong as a result.</p>
<div class="ex" data-name="Reading the discriminator">
<p>At any point in training, $D(\mathbf x)/(1-D(\mathbf x))$ estimates $p_{\text{data}}(\mathbf x)/p_g(\mathbf x)$. Where $D\approx 1$, the generator is under-producing; where $D\approx0$, it is over-producing. The right panel of the training lab plots the current $D$ against the ideal $D^*$ computed from the true data density and a histogram of generator samples.</p>
</div>`,
      },
      {
        title: 'Saturation and the non-saturating loss',
        html: R`
<p>Early in training, generated samples are obviously fake and $D(G(\mathbf z))\approx 0$. Write $D = \sigma(a)$ with logit $a$. The generator's minimax loss $\log(1-\sigma(a))$ has
$$\frac{\partial}{\partial a}\log\big(1-\sigma(a)\big) = -\sigma(a)\;\xrightarrow{a\to-\infty}\;0,$$
so the more confidently $D$ rejects a sample, the <em>less</em> gradient the generator receives. Goodfellow et al. therefore proposed the <strong>non-saturating</strong> generator loss: maximize $\E\log D(G(\mathbf z))$ instead, i.e. minimize $-\log\sigma(a)$, whose derivative $-(1-\sigma(a))\to -1$ as $a\to-\infty$. Both losses have the same fixed point ($D = 1/2$ everywhere when $p_g = p_{\text{data}}$), but the non-saturating one gives strong gradients exactly when the generator is worst.</p>
<p>It is not, however, the same divergence. Arjovsky &amp; Bottou (2017) showed that with $D = D^*_{G_{\theta_0}}$ held fixed,
$$\nabla_\theta\,\E_{\mathbf z}\big[-\log D^*(G_\theta(\mathbf z))\big]\Big|_{\theta=\theta_0} = \nabla_\theta\Big[\KL(p_{g_\theta}\Vert p_{\text{data}}) - 2\,\mathrm{JS}(p_{g_\theta}\Vert p_{\text{data}})\Big]\Big|_{\theta=\theta_0}.$$
The $\KL(p_g\Vert p_{\text{data}})$ term is the <em>reverse</em> KL: it is infinite if $p_g$ puts mass where $p_{\text{data}}$ has none, but costs little if $p_g$ ignores some data modes. It is mode-seeking, and the negative JS term even rewards distance. This is one theoretical account of why GAN samples are sharp but can lack diversity.</p>
<div class="cols">
<div><h4>Forward KL $\KL(p_{\text{data}}\Vert p_g)$</h4><p>Maximum likelihood. Infinite penalty for missing data mass. Mass-covering: tends to over-disperse, giving blur.</p></div>
<div><h4>Reverse KL $\KL(p_g\Vert p_{\text{data}})$</h4><p>Infinite penalty for producing implausible samples. Mode-seeking: tends to drop modes, giving sharp but less diverse samples.</p></div>
</div>`,
      },
      {
        title: 'Mode collapse and non-convergence',
        html: R`
<p>In the minimax formulation the discriminator is optimized <em>first</em>, inside. Alternating gradient steps instead approximate a different game, closer to $\max_D\min_G$ in places. Consider what the generator does against a <em>fixed</em> discriminator: it moves all its samples toward the point(s) where $D$ is largest — a single mode, if $G$ is flexible enough to concentrate there. The discriminator then learns to reject that mode, the generator jumps to another, and the pair can cycle indefinitely. This is <strong>mode collapse</strong>. In its partial form — modes dropped and later recovered, small modes ignored — it is visible even in the one-dimensional lab below.</p>
<p>Even the simplest game shows that simultaneous gradient play need not converge.</p>
<div class="thm" data-kind="Proposition" data-name="Gradient play diverges on a bilinear game">
<p>Let $V(x,y) = xy$, with $x$ minimizing and $y$ maximizing; the unique equilibrium is $(0,0)$. Simultaneous gradient descent–ascent $x_{t+1} = x_t - \eta y_t$, $y_{t+1} = y_t + \eta x_t$ satisfies $x_{t+1}^2 + y_{t+1}^2 = (1+\eta^2)(x_t^2+y_t^2)$: the iterates spiral outward for every step size $\eta>0$. Alternating updates ($y$ uses the new $x$) stay on bounded orbits but do not converge.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$(x_t-\eta y_t)^2 + (y_t+\eta x_t)^2 = x_t^2 + y_t^2 + \eta^2(x_t^2+y_t^2)$; the cross terms $\mp2\eta x_ty_t$ cancel. For the alternating version, $y_{t+1} = y_t + \eta(x_t - \eta y_t)$, the update matrix $\begin{pmatrix}1&-\eta\\ \eta&1-\eta^2\end{pmatrix}$ has determinant 1 and trace $2-\eta^2$; for $0<\eta<2$ its eigenvalues are complex conjugates of modulus 1, so the map is conjugate to a rotation. $\square$</p>
</details>
<p>The gradient vector field of a game has a rotational component that pure descent cannot remove. Practical stabilizers, roughly from most to least principled: two time-scale learning rates (Heusel et al., 2017, with a convergence proof to local Nash equilibria under assumptions), optimistic/extragradient methods (which converge on bilinear games), regularizing the discriminator (gradient penalties, spectral normalization; Miyato et al., 2018), and architectural heuristics (DCGAN). Minibatch discrimination and unrolled GANs target collapse directly by letting $D$ see sample diversity or letting $G$ anticipate $D$'s response.</p>`,
        lab: 'gans-training',
      },
      {
        title: 'When Jensen–Shannon fails: disjoint supports',
        html: R`
<div class="thm" data-kind="Proposition" data-name="JS is constant on disjoint supports">
<p>If $P$ and $Q$ are concentrated on disjoint sets, then $\mathrm{JS}(P\Vert Q) = \log 2$, and the discriminator $D = \ind_{\mathrm{supp} P}$ achieves $V = 0$, the maximum possible.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>On the support of $P$, $M = P/2$ (as $Q$ puts no mass there), so $\log(\d P/\d M) = \log 2$ $P$-a.s. and $\KL(P\Vert M) = \log 2$. Likewise $\KL(Q\Vert M) = \log2$, so $\mathrm{JS} = \log 2$. With $D = \ind_{\mathrm{supp}P}$ both log terms are $\log 1 = 0$. $\square$</p>
</details>
<p>This is the generic situation, not a corner case. Natural images concentrate near a low-dimensional manifold, and $p_g$ is supported on the image of a $k$-dimensional noise space; two such sets in $\R^D$ typically intersect in a set of measure zero for both. Arjovsky &amp; Bottou (2017) show that then a perfect discriminator exists and, with the minimax loss, the generator's gradient tends to zero as $D$ approaches it. The divergence is "flat at $\log 2$" until the distributions overlap, so it provides no signal about <em>how far apart</em> they are.</p>
<div class="ex" data-name="Point masses">
<p>Let $p_{\text{data}} = \delta_0$ and $p_{g} = \delta_\theta$ on $\R$. Then $\mathrm{JS} = \log 2$ for every $\theta\ne0$ and $0$ at $\theta=0$ — discontinuous, with zero gradient everywhere it is defined. $\KL$ is $+\infty$ for $\theta\ne0$. The earth-mover distance is $W_1 = |\theta|$: continuous, with gradient $\pm1$ pointing at the target. The divergence lab shows the smoothed version of this with narrow Gaussians.</p>
</div>`,
        lab: 'gans-divergences',
      },
      {
        title: 'Wasserstein GANs and Kantorovich–Rubinstein duality',
        html: R`
<div class="def" data-name="Wasserstein-1 (earth-mover) distance">
<p>For distributions $P,Q$ on $\R^D$ with finite first moments,
$$W_1(P,Q) = \inf_{\gamma\in\Pi(P,Q)}\E_{(X,Y)\sim\gamma}\lVert X - Y\rVert,$$
where $\Pi(P,Q)$ is the set of couplings (joint distributions with marginals $P$ and $Q$). It is the minimum average distance mass must be moved to turn $P$ into $Q$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Kantorovich–Rubinstein duality">
<p>$$W_1(P,Q) = \sup_{\lVert f\rVert_{\mathrm{Lip}}\le 1}\Big(\E_P f(X) - \E_Q f(Y)\Big),$$
the supremum over all 1-Lipschitz $f:\R^D\to\R$. In one dimension, $W_1(P,Q) = \int_{\R}|F_P(t) - F_Q(t)|\d t = \int_0^1|F_P^{-1}(u) - F_Q^{-1}(u)|\d u$.</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p><em>Weak duality ($\sup\le\inf$).</em> For any coupling $\gamma$ and 1-Lipschitz $f$: $\E_Pf - \E_Qf = \E_\gamma[f(X) - f(Y)]\le\E_\gamma\lVert X-Y\rVert$. Take the sup over $f$ and the inf over $\gamma$.</p>
<p><em>Strong duality.</em> The transport problem is an infinite-dimensional linear program; its dual is $\sup\{\E_P\varphi + \E_Q\psi : \varphi(x)+\psi(y)\le\lVert x-y\rVert\}$, and equality of primal and dual values follows from Fenchel–Rockafellar duality (or, for discrete $P,Q$, finite LP duality plus approximation). For the cost $\lVert x-y\rVert$, given a feasible $\varphi$ one can replace it by its "$c$-transform" $\varphi^c(y) = \inf_x(\lVert x - y\rVert - \varphi(x))$, which is 1-Lipschitz, and show the optimum has $\psi = -\varphi$ with $\varphi$ 1-Lipschitz (Villani, 2009, Thm 5.10 and Remark 6.5).</p>
<p><em>One dimension.</em> The quantile coupling $(F_P^{-1}(U),F_Q^{-1}(U))$ with $U$ uniform is optimal for convex costs on $\R$, giving the quantile formula; Fubini converts it to the CDF formula. $\square$</p>
</details>
<p>WGAN (Arjovsky, Chintala &amp; Bottou, 2017) replaces the discriminator by a <strong>critic</strong> $f_w$ with real-valued output, trained to maximize $\E_{p_{\text{data}}}f_w - \E_{\mathbf z}f_w(G(\mathbf z))$ over a class of Lipschitz functions, and trains $G$ to minimize $-\E_{\mathbf z}f_w(G(\mathbf z))$. Their key theorem: if $G_\theta(\mathbf z)$ is continuous in $\theta$, then $W_1(p_{\text{data}},p_{g_\theta})$ is continuous in $\theta$; if $G$ is locally Lipschitz with integrable constants, it is differentiable almost everywhere. Neither JS nor KL has these properties.</p>
<table>
<thead><tr><th>Lipschitz enforcement</th><th>Mechanism</th><th>Caveat</th></tr></thead>
<tbody>
<tr><td>Weight clipping (original WGAN)</td><td>Clip every weight to $[-c,c]$; bounds the Lipschitz constant by some $K(c)$</td><td>Crude; critic capacity collapses toward simple functions, or gradients explode for large $c$</td></tr>
<tr><td>Gradient penalty (WGAN-GP; Gulrajani et al., 2017)</td><td>Add $\lambda\,\E(\lVert\nabla f(\hat{\mathbf x})\rVert - 1)^2$ on random interpolates $\hat{\mathbf x}$</td><td>Soft constraint only on sampled lines; needs double backpropagation</td></tr>
<tr><td>Spectral normalization (Miyato et al., 2018)</td><td>Divide each weight matrix by its largest singular value</td><td>Bounds the Lipschitz constant by 1 per layer; may over-constrain</td></tr>
</tbody>
</table>
<p>Because the dual uses a <em>restricted</em> critic class (a neural network, with a crude Lipschitz constraint), what a WGAN actually minimizes is a neural <em>integral probability metric</em>, not $W_1$ itself. That is not a defect: the restricted metric still metrizes weak convergence under conditions, and its smoothness is what matters for optimization.</p>`,
      },
      {
        title: 'Evaluation, variants and pitfalls',
        html: R`
<p><strong>f-GANs.</strong> Any $f$-divergence $D_f(P\Vert Q) = \E_Q f(\d P/\d Q)$ has a variational lower bound $\sup_T\{\E_P T - \E_Q f^*(T)\}$ via the convex conjugate $f^*$, and plugging a neural $T$ in gives a GAN for that divergence (Nowozin, Cseke &amp; Tomioka, 2016). The original GAN is the case $f(u) = u\log u - (u+1)\log(u+1)$, which yields $2\,\mathrm{JS} - \log 4$.</p>
<p><strong>Evaluation without likelihood.</strong> The Fréchet Inception Distance fits Gaussians to network features of real and generated samples and reports $\lVert\boldsymbol\mu_r-\boldsymbol\mu_g\rVert^2 + \tr\big(\bSigma_r+\bSigma_g-2(\bSigma_r\bSigma_g)^{1/2}\big)$, the squared 2-Wasserstein distance between those Gaussians (Heusel et al., 2017). It is biased in finite samples, depends on the feature network, and does not separate fidelity from diversity; precision/recall metrics try to do the latter.</p>
<div class="warn" data-name="Discriminator loss is not a progress meter">
<p>In the standard GAN, the discriminator's loss reflects the current balance of the game, not sample quality. A WGAN critic's value is a more meaningful (but still biased) estimate of distance, one of WGAN's practical advantages.</p>
</div>
<div class="warn" data-name="Memorization and evaluation leakage">
<p>A generator that reproduces training images exactly achieves excellent FID. Always check nearest training neighbours of generated samples.</p>
</div>
<div class="warn" data-name="Mode dropping is silent">
<p>Samples can look perfect while whole classes are missing. Count modes (as the lab does) or measure recall in feature space.</p>
</div>
<p>GANs dominated image synthesis from about 2016 to 2021 (StyleGAN set the standard for faces). Their training difficulties, and the absence of a likelihood, are the main reasons <a href="#/t/diffusion-models">diffusion models</a> — which have a stable regression objective and cover modes well — displaced them, although adversarial losses survive as components (e.g. in autoencoders used for latent diffusion, and in distillation of diffusion samplers).</p>`,
      },
    ],
    quiz: [
      { q: R`For a fixed generator, $p_{\text{data}}(\mathbf x) = 0.3$ and $p_g(\mathbf x) = 0.1$ at some point. What does the optimal discriminator output there, and what does it tell the generator?`,
        a: R`$D^* = 0.3/(0.3+0.1) = 0.75$. Odds $D^*/(1-D^*) = 3 = p_{\text{data}}/p_g$: the generator under-produces samples near this point by a factor of 3.` },
      { q: R`What is $\max_D V(D,G)$ when $p_g = p_{\text{data}}$? When their supports are disjoint?`,
        a: R`Equal: $D^* = 1/2$ everywhere and $V = 2\log\frac12 = -\log 4$ ($\mathrm{JS} = 0$). Disjoint: $\mathrm{JS} = \log 2$ so $V = -\log 4 + 2\log 2 = 0$, achieved by a perfect discriminator.` },
      { q: R`Why does the non-saturating generator loss give larger gradients early in training? Does it optimize the same divergence?`,
        a: R`With $D=\sigma(a)$ and $a\ll0$ for fake samples, $\partial_a\log(1-\sigma(a)) = -\sigma(a)\approx0$, while $\partial_a[-\log\sigma(a)] = -(1-\sigma(a))\approx -1$. It has the same fixed point but, with $D$ optimal, its gradient equals that of $\KL(p_g\Vert p_{\text{data}}) - 2\mathrm{JS}$ — a mode-seeking objective, not JS.` },
      { q: R`Compute $W_1$ between $\N(0,1)$ and $\N(3,1)$, and explain why the JS divergence between $\N(0,\epsilon^2)$ and $\N(3,\epsilon^2)$ gives the generator almost no gradient for small $\epsilon$.`,
        a: R`A pure translation: the quantile functions differ by 3 everywhere, so $W_1 = 3$ (and $\partial W_1/\partial\theta = \pm1$ for shift $\theta$). For small $\epsilon$ the two densities barely overlap, so JS is within a tiny amount of its maximum $\log2$, and it stays there for all shifts beyond a few $\epsilon$: its derivative with respect to the shift is essentially zero.` },
      { q: R`Show that simultaneous gradient descent–ascent on $V(x,y) = xy$ cannot converge from any non-zero start.`,
        a: R`The update $(x,y)\mapsto(x-\eta y, y+\eta x)$ multiplies the squared norm by exactly $1+\eta^2>1$ (cross terms cancel), so $\lVert(x_t,y_t)\rVert = (1+\eta^2)^{t/2}\lVert(x_0,y_0)\rVert\to\infty$.` },
    ],
  });

  /* ------------------------------------------------------------------ */
  /*  Diffusion & score-based models                                     */
  /* ------------------------------------------------------------------ */
  Atlas.topic({
    id: 'diffusion-models',
    domain: 'deep',
    order: 130,
    title: 'Diffusion & Score-Based Generative Models',
    subtitle: 'Destroy data with noise, learn the score, and run the process backwards',
    level: 3,
    sources: { other: 'Sohl-Dickstein et al. (2015); Ho, Jain & Abbeel (2020); Song et al. (2021) score SDEs; Song & Ermon (2019); Vincent (2011); Hyvärinen (2005); Anderson (1982); Efron (2011) on Tweedie; Song, Meng & Ermon (2021) DDIM; Prince, Understanding Deep Learning (2023), Ch. 18' },
    prereqs: ['autoencoders-vae', 'stochastic-calculus'],
    related: ['langevin-mala', 'gans', 'brownian-motion', 'entropy-kl', 'kernel-density'],
    tags: ['diffusion', 'score matching', 'Tweedie', 'SDE', 'probability flow', 'Langevin', 'DDPM'],
    summary: R`A diffusion model fixes a forward process that gradually turns data into Gaussian noise, $q(\mathbf x_t\mid\mathbf x_0) = \N(\sqrt{\bar\alpha_t}\,\mathbf x_0, (1-\bar\alpha_t)I)$, and learns to reverse it. The only thing the reverse process needs is the <em>score</em> $\nabla\log p_t$ of each noised marginal, and that can be learned by plain regression: predict the noise that was added (denoising score matching), which by Tweedie's formula is equivalent to estimating the posterior mean of the clean data. In continuous time the forward process is an SDE; Anderson's theorem gives the reverse-time SDE, and a deterministic probability-flow ODE shares its marginals. The reverse SDE is exactly the ODE plus Langevin dynamics on the current marginal, linking diffusion sampling to MCMC.`,
    survey: {
      task: 'High-fidelity generation (images, audio, video, molecules); conditional generation; likelihood estimation via the ODE',
      assumptions: R`The score $\nabla\log p_t$ can be learned accurately at all noise levels; enough reverse steps to control discretization error`,
      output: R`A noise-prediction (score) network $\boldsymbol\epsilon_\theta(\mathbf x_t,t)$; samplers (stochastic or deterministic)`,
      strengths: 'Stable regression objective; excellent mode coverage and sample quality; exact likelihood via ODE; flexible guidance',
      weaknesses: 'Slow sampling (tens to thousands of network evaluations); memorization if the score is fitted too exactly',
      complexity: R`Training: one network pass per example; sampling: $N$ passes, $N\approx 10$–$1000$`,
    },
    sections: [
      {
        title: 'The idea: learn to undo a known corruption',
        html: R`
<p>Generating data from scratch is hard; removing a little noise from a slightly noisy image is comparatively easy. Diffusion models (Sohl-Dickstein et al., 2015; Ho, Jain &amp; Abbeel, 2020) exploit this by chaining many easy denoising steps. A fixed <strong>forward process</strong> adds Gaussian noise in $T$ small steps,
$$q(\mathbf x_t\mid\mathbf x_{t-1}) = \N\big(\sqrt{1-\beta_t}\,\mathbf x_{t-1},\;\beta_t I\big),\qquad t = 1,\dots,T,$$
with a variance schedule $0<\beta_1<\cdots<\beta_T<1$ (Ho et al. use $\beta$ linear from $10^{-4}$ to $0.02$ with $T = 1000$). A learned <strong>reverse process</strong> $p_\theta(\mathbf x_{t-1}\mid\mathbf x_t) = \N(\boldsymbol\mu_\theta(\mathbf x_t,t),\sigma_t^2I)$ starts from $\mathbf x_T\sim\N(\mathbf 0,I)$ and walks back. The shrinkage factor $\sqrt{1-\beta_t}$ keeps the total variance bounded ("variance preserving"), so the forward chain converges to $\N(\mathbf 0,I)$ rather than blowing up.</p>
<div class="thm" data-kind="Proposition" data-name="Closed-form marginal of the forward process">
<p>With $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s=1}^t\alpha_s$,
$$q(\mathbf x_t\mid\mathbf x_0) = \N\big(\sqrt{\bar\alpha_t}\,\mathbf x_0,\;(1-\bar\alpha_t)I\big),\quad\text{i.e.}\quad \mathbf x_t = \sqrt{\bar\alpha_t}\,\mathbf x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\;\;\boldsymbol\epsilon\sim\N(\mathbf 0,I).$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Induction on $t$; $t=1$ is the definition. Suppose $\mathbf x_{t-1} = \sqrt{\bar\alpha_{t-1}}\mathbf x_0 + \sqrt{1-\bar\alpha_{t-1}}\,\boldsymbol\epsilon'$. Then $\mathbf x_t = \sqrt{\alpha_t}\mathbf x_{t-1} + \sqrt{1-\alpha_t}\,\boldsymbol\epsilon'' = \sqrt{\bar\alpha_t}\mathbf x_0 + \sqrt{\alpha_t(1-\bar\alpha_{t-1})}\,\boldsymbol\epsilon' + \sqrt{1-\alpha_t}\,\boldsymbol\epsilon''$ with independent standard normals $\boldsymbol\epsilon',\boldsymbol\epsilon''$. The noise is Gaussian with mean zero and covariance $\big(\alpha_t - \bar\alpha_t + 1 - \alpha_t\big)I = (1-\bar\alpha_t)I$. $\square$</p>
</details>
<p>Two consequences drive everything. Training can jump to any noise level in one shot — no need to simulate the chain. And since $\bar\alpha_T\approx 4\times10^{-5}$ for the standard schedule, $q(\mathbf x_T\mid\mathbf x_0)\approx\N(\mathbf 0,I)$ regardless of $\mathbf x_0$: the end of the forward process forgets the data, so the reverse process can start from pure noise. The signal-to-noise ratio $\mathrm{SNR}(t) = \bar\alpha_t/(1-\bar\alpha_t)$ decreases monotonically from $\infty$ to nearly 0.</p>
<div class="thm" data-kind="Proposition" data-name="The forward posterior is Gaussian">
<p>$q(\mathbf x_{t-1}\mid\mathbf x_t,\mathbf x_0) = \N(\tilde{\boldsymbol\mu}_t,\tilde\beta_tI)$ with
$$\tilde{\boldsymbol\mu}_t = \frac{\sqrt{\bar\alpha_{t-1}}\,\beta_t}{1-\bar\alpha_t}\mathbf x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf x_t,\qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>By Bayes and the Markov property, $q(\mathbf x_{t-1}\mid\mathbf x_t,\mathbf x_0)\propto q(\mathbf x_t\mid\mathbf x_{t-1})\,q(\mathbf x_{t-1}\mid\mathbf x_0)$, a product of Gaussians in $\mathbf x_{t-1}$: $\N(\mathbf x_t;\sqrt{\alpha_t}\mathbf x_{t-1},\beta_tI)$ and $\N(\mathbf x_{t-1};\sqrt{\bar\alpha_{t-1}}\mathbf x_0,(1-\bar\alpha_{t-1})I)$. Precisions add: $\alpha_t/\beta_t + 1/(1-\bar\alpha_{t-1}) = (1-\bar\alpha_t)/(\beta_t(1-\bar\alpha_{t-1}))$, using $\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = 1-\bar\alpha_t$. Its inverse is $\tilde\beta_t$, and the mean is $\tilde\beta_t\big(\sqrt{\alpha_t}\mathbf x_t/\beta_t + \sqrt{\bar\alpha_{t-1}}\mathbf x_0/(1-\bar\alpha_{t-1})\big)$, which simplifies to $\tilde{\boldsymbol\mu}_t$. $\square$</p>
</details>`,
        lab: 'diffusion-models-forward',
      },
      {
        title: 'The variational bound and the noise-prediction objective',
        html: R`
<p>The reverse chain is a latent-variable model with latents $\mathbf x_{1:T}$, and the forward chain is a (fixed) variational posterior — a hierarchical <a href="#/t/autoencoders-vae">VAE</a> whose encoder is not learned. The ELBO applies verbatim.</p>
<div class="thm" data-kind="Theorem" data-name="ELBO decomposition (Sohl-Dickstein et al., 2015; Ho et al., 2020)">
<p>$$-\log p_\theta(\mathbf x_0)\le\E_q\Big[\underbrace{\KL\big(q(\mathbf x_T\mid\mathbf x_0)\Vert p(\mathbf x_T)\big)}_{L_T} + \sum_{t=2}^T\underbrace{\KL\big(q(\mathbf x_{t-1}\mid\mathbf x_t,\mathbf x_0)\Vert p_\theta(\mathbf x_{t-1}\mid\mathbf x_t)\big)}_{L_{t-1}}\;\underbrace{-\log p_\theta(\mathbf x_0\mid\mathbf x_1)}_{L_0}\Big].$$</p>
</div>
<details class="proof"><summary>Proof sketch</summary>
<p>The ELBO with $q(\mathbf x_{1:T}\mid\mathbf x_0)$ as variational distribution gives $-\log p_\theta(\mathbf x_0)\le\E_q\big[-\log p(\mathbf x_T) - \sum_{t\ge1}\log\frac{p_\theta(\mathbf x_{t-1}\mid\mathbf x_t)}{q(\mathbf x_t\mid\mathbf x_{t-1})}\big]$. For $t\ge2$ rewrite, using the Markov property and Bayes, $q(\mathbf x_t\mid\mathbf x_{t-1}) = q(\mathbf x_{t-1}\mid\mathbf x_t,\mathbf x_0)\,q(\mathbf x_t\mid\mathbf x_0)/q(\mathbf x_{t-1}\mid\mathbf x_0)$. The ratios $q(\mathbf x_t\mid\mathbf x_0)/q(\mathbf x_{t-1}\mid\mathbf x_0)$ telescope to $q(\mathbf x_T\mid\mathbf x_0)/q(\mathbf x_1\mid\mathbf x_0)$, and the $q(\mathbf x_1\mid\mathbf x_0)$ cancels against the $t=1$ term. Grouping $\log p(\mathbf x_T)$ with $q(\mathbf x_T\mid\mathbf x_0)$ gives $L_T$, and each remaining pair gives an expected KL, $L_{t-1}$. $\square$</p>
</details>
<p>$L_T$ has no parameters. Each $L_{t-1}$ is a KL between Gaussians. If $p_\theta$ has fixed variance $\sigma_t^2$, $L_{t-1} = \E_q\lVert\tilde{\boldsymbol\mu}_t - \boldsymbol\mu_\theta(\mathbf x_t,t)\rVert^2/(2\sigma_t^2) + C$. Substituting $\mathbf x_0 = (\mathbf x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon)/\sqrt{\bar\alpha_t}$ into $\tilde{\boldsymbol\mu}_t$ gives
$$\tilde{\boldsymbol\mu}_t = \frac{1}{\sqrt{\alpha_t}}\Big(\mathbf x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon\Big),$$
which suggests parameterizing $\boldsymbol\mu_\theta$ the same way with a network $\boldsymbol\epsilon_\theta(\mathbf x_t,t)$ in place of $\boldsymbol\epsilon$. Then
$$L_{t-1} = \E_{\mathbf x_0,\boldsymbol\epsilon}\Big[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\big\lVert\boldsymbol\epsilon - \boldsymbol\epsilon_\theta\big(\sqrt{\bar\alpha_t}\mathbf x_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon,\;t\big)\big\rVert^2\Big] + C.$$
Ho et al. found that dropping the weight, $L_{\text{simple}} = \E_{t,\mathbf x_0,\boldsymbol\epsilon}\lVert\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\mathbf x_t,t)\rVert^2$ with $t$ uniform, gives better samples (it down-weights the very low-noise terms that matter for likelihood but not for perception).</p>
<div class="algo" data-name="DDPM training and sampling">
<ol>
<li><em>Train:</em> repeat — draw $\mathbf x_0$ from data, $t\sim\mathrm{Unif}\{1..T\}$, $\boldsymbol\epsilon\sim\N(0,I)$; take a gradient step on $\lVert\boldsymbol\epsilon - \boldsymbol\epsilon_\theta(\sqrt{\bar\alpha_t}\mathbf x_0 + \sqrt{1-\bar\alpha_t}\boldsymbol\epsilon, t)\rVert^2$.</li>
<li><em>Sample:</em> $\mathbf x_T\sim\N(0,I)$; for $t = T,\dots,1$: $\mathbf x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\big(\mathbf x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol\epsilon_\theta(\mathbf x_t,t)\big) + \sigma_t\mathbf z$, with $\mathbf z\sim\N(0,I)$ ($\mathbf z = 0$ at $t=1$) and $\sigma_t^2 = \beta_t$ or $\tilde\beta_t$.</li>
</ol>
</div>
<p>Training is just regression — no adversary, no posterior network, no sampling inside the loss — which is why diffusion training is so stable.</p>`,
      },
      {
        title: 'Scores, denoising and Tweedie\'s formula',
        html: R`
<p>What does the noise predictor actually estimate? The <strong>score</strong> of a density is $\nabla_{\mathbf x}\log p(\mathbf x)$: a vector field pointing toward higher density, which does not need the normalizing constant. Let $p_t$ denote the marginal of $\mathbf x_t$ (data pushed through the forward process).</p>
<div class="thm" data-kind="Theorem" data-name="Tweedie's formula (Robbins, 1956; Efron, 2011)">
<p>If $\mathbf y = a\mathbf x + \sigma\boldsymbol\epsilon$ with $\boldsymbol\epsilon\sim\N(\mathbf 0,I)$ independent of $\mathbf x$ and $a>0$, and $p$ is the density of $\mathbf y$, then
$$\E[\mathbf x\mid\mathbf y] = \frac{\mathbf y + \sigma^2\nabla\log p(\mathbf y)}{a}.$$</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>$p(\mathbf y) = \int\N(\mathbf y; a\mathbf x,\sigma^2I)\,\pi(\d\mathbf x)$. Differentiating under the integral, $\nabla_{\mathbf y}\N(\mathbf y; a\mathbf x,\sigma^2I) = -\sigma^{-2}(\mathbf y - a\mathbf x)\N(\mathbf y;a\mathbf x,\sigma^2I)$, so $\nabla p(\mathbf y) = -\sigma^{-2}\int(\mathbf y - a\mathbf x)\N(\mathbf y;a\mathbf x,\sigma^2 I)\,\pi(\d\mathbf x)$. Dividing by $p(\mathbf y)$ turns the integral into a posterior expectation: $\nabla\log p(\mathbf y) = -\sigma^{-2}(\mathbf y - a\E[\mathbf x\mid\mathbf y])$. Rearrange. $\square$</p>
</details>
<p>With $a = \sqrt{\bar\alpha_t}$, $\sigma^2 = 1-\bar\alpha_t$ and $\boldsymbol\epsilon = (\mathbf x_t - \sqrt{\bar\alpha_t}\mathbf x_0)/\sqrt{1-\bar\alpha_t}$, Tweedie gives
$$\E[\boldsymbol\epsilon\mid\mathbf x_t] = -\sqrt{1-\bar\alpha_t}\;\nabla\log p_t(\mathbf x_t).$$
The minimizer of the squared-error objective is the conditional mean, so the ideal noise predictor is $\boldsymbol\epsilon^*_\theta(\mathbf x_t,t) = -\sqrt{1-\bar\alpha_t}\nabla\log p_t(\mathbf x_t)$: <strong>a denoiser is a score estimator</strong>, and $\hat{\mathbf x}_0 = (\mathbf x_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol\epsilon_\theta)/\sqrt{\bar\alpha_t}$ is the posterior-mean (MMSE) denoised image. The forward lab lets you click a noisy point and see $\E[\mathbf x_0\mid\mathbf x_t]$.</p>
<div class="thm" data-kind="Proposition" data-name="Denoising score matching (Vincent, 2011)">
<p>For any $\mathbf s_\theta$ with finite second moments,
$$\E_{q(\mathbf x_0,\mathbf x_t)}\big\lVert\mathbf s_\theta(\mathbf x_t) - \nabla_{\mathbf x_t}\log q(\mathbf x_t\mid\mathbf x_0)\big\rVert^2 = \E_{p_t}\big\lVert\mathbf s_\theta(\mathbf x_t) - \nabla\log p_t(\mathbf x_t)\big\rVert^2 + C,$$
where $C$ does not depend on $\theta$.</p>
</div>
<details class="proof"><summary>Proof</summary>
<p>Expand both squares. The $\lVert\mathbf s_\theta\rVert^2$ terms agree; the remaining $\theta$-free terms go into $C$. For the cross terms, $\E_{q(\mathbf x_0,\mathbf x_t)}[\mathbf s_\theta(\mathbf x_t)^\top\nabla\log q(\mathbf x_t\mid\mathbf x_0)] = \int\!\!\int\mathbf s_\theta(\mathbf x_t)^\top\nabla_{\mathbf x_t}q(\mathbf x_t\mid\mathbf x_0)\,\pi(\mathbf x_0)\d\mathbf x_0\d\mathbf x_t = \int\mathbf s_\theta(\mathbf x_t)^\top\nabla p_t(\mathbf x_t)\d\mathbf x_t = \E_{p_t}[\mathbf s_\theta^\top\nabla\log p_t]$. $\square$</p>
</details>
<p>Since $\nabla_{\mathbf x_t}\log q(\mathbf x_t\mid\mathbf x_0) = -\boldsymbol\epsilon/\sqrt{1-\bar\alpha_t}$, the DDPM loss is denoising score matching with $\mathbf s_\theta = -\boldsymbol\epsilon_\theta/\sqrt{1-\bar\alpha_t}$, summed over noise levels. The unknown, intractable target $\nabla\log p_t$ has been replaced by a known one whose conditional mean it is. (Hyvärinen's original score matching (2005) avoids noise by integrating by parts, at the cost of a Hessian trace.)</p>`,
      },
      {
        title: 'Continuous time: reverse SDEs and the probability-flow ODE',
        html: R`
<p>Let the step sizes shrink. With $\beta_t = \beta(t)\Delta t$, the forward chain $\mathbf x_{t+\Delta t} = \sqrt{1-\beta(t)\Delta t}\,\mathbf x_t + \sqrt{\beta(t)\Delta t}\,\boldsymbol\epsilon\approx\mathbf x_t - \tfrac12\beta(t)\mathbf x_t\Delta t + \sqrt{\beta(t)}\,\Delta\mathbf W$ becomes the <strong>variance-preserving SDE</strong> (an Ornstein–Uhlenbeck process with time-varying rate; see <a href="#/t/stochastic-calculus">Itô calculus</a>)
$$\d\mathbf x = -\tfrac12\beta(t)\mathbf x\d t + \sqrt{\beta(t)}\d\mathbf W,$$
whose transition is $\mathbf x_t\mid\mathbf x_0\sim\N\big(e^{-B(t)/2}\mathbf x_0,(1-e^{-B(t)})I\big)$ with $B(t) = \int_0^t\beta(s)\d s$, i.e. $\bar\alpha(t) = e^{-B(t)}$. Song et al. (2021) cast all diffusion models as SDEs $\d\mathbf x = \mathbf f(\mathbf x,t)\d t + g(t)\d\mathbf W$ (the variance-exploding SDE of score-based models is $\mathbf f = 0$).</p>
<div class="thm" data-kind="Theorem" data-name="Reverse-time SDE (Anderson, 1982)">
<p>Under regularity conditions, the time reversal $\mathbf y_\tau = \mathbf x_{T-\tau}$ of $\d\mathbf x = \mathbf f\d t + g\d\mathbf W$ with marginals $p_t$ is itself a diffusion:
$$\d\mathbf x = \big[\mathbf f(\mathbf x,t) - g(t)^2\nabla\log p_t(\mathbf x)\big]\d t + g(t)\d\bar{\mathbf W},$$
run from $t = T$ down to $0$, where $\bar{\mathbf W}$ is a Brownian motion in reversed time and $\d t<0$.</p>
</div>
<div class="thm" data-kind="Theorem" data-name="Probability-flow ODE (Song et al., 2021)">
<p>The deterministic ODE
$$\frac{\d\mathbf x}{\d t} = \mathbf f(\mathbf x,t) - \tfrac12 g(t)^2\nabla\log p_t(\mathbf x)$$
has the same marginal densities $p_t$ as the forward SDE, when started from $p_T$ at time $T$ (or $p_0$ at time 0).</p>
</div>
<details class="proof"><summary>Proof sketch (both theorems, at the level of marginals)</summary>
<p>The forward marginals satisfy the Fokker–Planck equation $\partial_tp = -\nabla\cdot(\mathbf fp) + \tfrac12g^2\Delta p$. Since $\Delta p = \nabla\cdot(\nabla p) = \nabla\cdot(p\nabla\log p)$, this is $\partial_tp = -\nabla\cdot\big[(\mathbf f - \tfrac12g^2\nabla\log p)\,p\big]$ — the continuity equation of a deterministic flow with velocity $\mathbf f - \tfrac12g^2\nabla\log p_t$, which is the ODE. For the reverse SDE, write it in forward time $\tau = T-t$ as $\d\mathbf y = [-\mathbf f + g^2\nabla\log p_{T-\tau}]\d\tau + g\d\mathbf W$. Its Fokker–Planck equation for $r_\tau$ is $\partial_\tau r = -\nabla\cdot\big[(-\mathbf f + g^2\nabla\log p_{T-\tau})r\big] + \tfrac12g^2\Delta r$. Substituting $r_\tau = p_{T-\tau}$: the right side becomes $\nabla\cdot(\mathbf fp) - g^2\Delta p + \tfrac12g^2\Delta p = \nabla\cdot(\mathbf fp) - \tfrac12g^2\Delta p = -\partial_tp$, which is exactly $\partial_\tau p_{T-\tau}$. So $p_{T-\tau}$ solves the reverse Fokker–Planck equation; uniqueness gives the claim. (Anderson's theorem is stronger: the reversed <em>path</em> distribution matches.) $\square$</p>
</details>
<p>For the VP-SDE these read: reverse SDE $\d\mathbf x = -\beta(t)\big[\tfrac12\mathbf x + \nabla\log p_t(\mathbf x)\big]\d t + \sqrt{\beta(t)}\d\bar{\mathbf W}$; probability-flow ODE $\d\mathbf x/\d t = -\tfrac12\beta(t)\big[\mathbf x + \nabla\log p_t(\mathbf x)\big]$. DDPM ancestral sampling is a particular discretization of the first; DDIM (Song, Meng &amp; Ermon, 2021) with $\eta = 0$ is a discretization of the second. The ODE gives a deterministic, invertible map between noise and data, so it also yields <em>exact likelihoods</em> via the instantaneous change-of-variables formula $\log p_0(\mathbf x_0) = \log p_T(\mathbf x_T) + \int_0^T\nabla\cdot\mathbf v(\mathbf x_t,t)\d t$ for the ODE velocity $\mathbf v$.</p>`,
        lab: 'diffusion-models-reverse',
      },
      {
        title: 'The Langevin connection',
        html: R`
<p><a href="#/t/langevin-mala">Langevin dynamics</a> $\d\mathbf x = \nabla\log\pi(\mathbf x)\d t + \sqrt2\d\mathbf W$ has $\pi$ as a stationary density: its Fokker–Planck right side $-\nabla\cdot(p\nabla\log\pi) + \Delta p = \nabla\cdot\big(p\nabla\log(p/\pi)\big)$ vanishes at $p = \pi$. It needs only the score of $\pi$. So why not sample data by running Langevin with a learned score of $p_{\text{data}}$? Two reasons (Song &amp; Ermon, 2019): the score is poorly estimated in low-density regions, where there is no training data; and Langevin mixes very slowly between separated modes and does not recover their relative weights in reasonable time. Noise fixes both: $p_t$ for large $t$ is smooth, unimodal and well estimated everywhere, and annealing $t$ downward carries the right mode proportions along.</p>
<div class="thm" data-kind="Proposition" data-name="Reverse SDE = probability flow + Langevin">
<p>The reverse-SDE drift splits as
$$\underbrace{\mathbf f - g^2\nabla\log p_t}_{\text{reverse SDE}} = \underbrace{\big(\mathbf f - \tfrac12g^2\nabla\log p_t\big)}_{\text{probability-flow ODE}} \;-\;\underbrace{\tfrac12 g^2\nabla\log p_t}_{\text{Langevin drift}},$$
and in reversed time ($\d\tau = -\d t$) the second piece together with the noise $g\d\bar{\mathbf W}$ is a Langevin diffusion $\tfrac12g^2\nabla\log p_t\,\d\tau + g\d\mathbf W$ with step $\epsilon = \tfrac12g^2\d\tau$, which leaves the current marginal $p_t$ invariant.</p>
</div>
<p>So a stochastic reverse sampler is "transport the density along the probability flow, while simultaneously running Langevin MCMC on the current marginal". The Langevin part is self-correcting: if discretization or score errors push particles off $p_t$, it pulls them back, which is why stochastic samplers are more robust to score error than the ODE, at the cost of needing more steps. Predictor–corrector samplers (Song et al., 2021) make this explicit, alternating an ODE/SDE step with a few Langevin (or MALA-style) corrector steps at fixed $t$.</p>
<div class="note">
<p>The reverse-sampling lab makes the trade-off concrete. With the <em>exact</em> score, the deterministic ODE is at least as accurate as the Euler–Maruyama SDE even at 10–20 steps (the SDE pays for its injected noise with discretization variance). Multiply the score by a factor $c\ne1$ — a crude model of an imperfect network — and the ODE transports particles to the wrong distribution, while the SDE's Langevin component keeps pulling them back toward the current marginal, so its error grows far more slowly.</p>
</div>`,
      },
      {
        title: 'Pitfalls, guidance and what the theory does not say',
        html: R`
<div class="warn" data-name="The exact score memorizes">
<p>If $p_0$ is the <em>empirical</em> distribution of $n$ training points, $p_t$ is an $n$-component Gaussian mixture with an explicit score, and the reverse process driven by that exact score reproduces training points exactly as $t\to0$. A diffusion model that generalizes therefore does <em>not</em> learn the optimal score of its training objective; the inductive biases of the network (and early stopping) produce a smoother score — in effect a <a href="#/t/kernel-density">kernel density estimate</a>-like smoothing of the data — and memorization appears when models are large relative to the data.</p>
</div>
<div class="warn" data-name="Discretization error">
<p>Each sampling step costs a network evaluation. With few steps, the Euler discretization of the SDE adds excess variance, and that of the ODE drifts off the data manifold. Higher-order ODE solvers (DPM-Solver, Heun) and distillation (progressive distillation, consistency models) cut steps from 1000 to roughly 1–50.</p>
</div>
<div class="warn" data-name="Score errors near t = 0">
<p>The score of nearly clean data is huge and ill-conditioned (it scales like $1/\sqrt{1-\bar\alpha_t}$). Samplers stop at a small $t_{\min}>0$ and output the denoised mean $\hat{\mathbf x}_0$.</p>
</div>
<p><strong>Guidance.</strong> For conditional generation, Bayes' rule in score form gives $\nabla\log p_t(\mathbf x\mid c) = \nabla\log p_t(\mathbf x) + \nabla\log p_t(c\mid\mathbf x)$. <em>Classifier guidance</em> adds a scaled gradient of a noisy-image classifier; <em>classifier-free guidance</em> (Ho &amp; Salimans, 2022) trains one network with and without the condition and uses $\tilde{\boldsymbol\epsilon} = (1+w)\boldsymbol\epsilon_\theta(\mathbf x_t,c) - w\,\boldsymbol\epsilon_\theta(\mathbf x_t)$. For $w>0$ this samples (approximately) from a sharpened $p(\mathbf x)\,p(c\mid\mathbf x)^{1+w}$: better prompt fidelity, less diversity — a deliberate move from mass-covering toward mode-seeking.</p>
<table>
<thead><tr><th>Object</th><th>Discrete (DDPM)</th><th>Continuous (SDE)</th></tr></thead>
<tbody>
<tr><td>Forward</td><td>$q(\mathbf x_t\mid\mathbf x_{t-1}) = \N(\sqrt{\alpha_t}\mathbf x_{t-1},\beta_tI)$</td><td>$\d\mathbf x = -\tfrac12\beta\mathbf x\d t + \sqrt\beta\d\mathbf W$</td></tr>
<tr><td>Learned quantity</td><td>$\boldsymbol\epsilon_\theta(\mathbf x_t,t)$</td><td>$\mathbf s_\theta(\mathbf x,t) = -\boldsymbol\epsilon_\theta/\sqrt{1-\bar\alpha(t)}$</td></tr>
<tr><td>Stochastic sampler</td><td>Ancestral sampling</td><td>Reverse SDE (Euler–Maruyama)</td></tr>
<tr><td>Deterministic sampler</td><td>DDIM ($\eta=0$)</td><td>Probability-flow ODE</td></tr>
<tr><td>Objective</td><td>ELBO / $L_{\text{simple}}$</td><td>Weighted denoising score matching</td></tr>
</tbody>
</table>`,
      },
    ],
    quiz: [
      { q: R`Using the closed form, how would you sample $\mathbf x_{500}$ given $\mathbf x_0$ in one step? What is $\E[\mathbf x_{500}\mid\mathbf x_0]$ and its covariance?`,
        a: R`$\mathbf x_{500} = \sqrt{\bar\alpha_{500}}\,\mathbf x_0 + \sqrt{1-\bar\alpha_{500}}\,\boldsymbol\epsilon$ with $\boldsymbol\epsilon\sim\N(0,I)$. Mean $\sqrt{\bar\alpha_{500}}\,\mathbf x_0$, covariance $(1-\bar\alpha_{500})I$, where $\bar\alpha_{500} = \prod_{s\le500}(1-\beta_s)$. No need to simulate 500 steps.` },
      { q: R`State Tweedie's formula and use it to explain why a network trained to predict the added noise is learning the score.`,
        a: R`If $\mathbf y = a\mathbf x + \sigma\boldsymbol\epsilon$, then $\E[\mathbf x\mid\mathbf y] = (\mathbf y + \sigma^2\nabla\log p(\mathbf y))/a$, equivalently $\E[\boldsymbol\epsilon\mid\mathbf y] = -\sigma\nabla\log p(\mathbf y)$. The least-squares noise predictor converges to $\E[\boldsymbol\epsilon\mid\mathbf x_t]$, which is $-\sqrt{1-\bar\alpha_t}\nabla\log p_t(\mathbf x_t)$: the score up to a known scale.` },
      { q: R`The probability-flow ODE and the reverse SDE share marginals. Why might one still prefer the SDE when the score network is imperfect?`,
        a: R`The reverse SDE equals the ODE plus Langevin dynamics targeting the current marginal $p_t$. The Langevin component pulls particles back toward high-density regions of $p_t$, correcting accumulated errors from an imperfect score or coarse discretization; the ODE has no such contraction and propagates errors deterministically.` },
      { q: R`Show from the Fokker–Planck equation why the probability-flow drift has a factor $\tfrac12$ in front of $g^2\nabla\log p_t$.`,
        a: R`$\partial_tp = -\nabla\cdot(\mathbf fp) + \tfrac12g^2\Delta p$ and $\Delta p = \nabla\cdot(p\nabla\log p)$, so $\partial_tp = -\nabla\cdot[(\mathbf f - \tfrac12g^2\nabla\log p)p]$: a pure continuity equation with velocity $\mathbf f - \tfrac12g^2\nabla\log p$. The $\tfrac12$ is the $\tfrac12$ of the Itô diffusion term.` },
      { q: R`If you trained a diffusion model so well that it learned the exact score of the noised <em>empirical</em> distribution of the training set, what would sampling produce?`,
        a: R`Copies of training points. The noised empirical distribution is a Gaussian mixture centred at the (scaled) training points; its exact reverse process collapses onto those centres as $t\to0$. Generalization relies on the network learning a smoothed approximation, not the exact empirical score.` },
    ],
  });
})();
