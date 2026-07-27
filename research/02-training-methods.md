# Training and Sampling Energy-Based Models: Implementation Reference

Status: research spec for `ebm-py` v0.1. Sources: Song & Kingma, *How to Train Your Energy-Based Models* (arXiv:2101.03288); Du & Mordatch, *Implicit Generation and Modeling with EBMs* (arXiv:1903.08689, "IGEBM"); Nijkamp et al., *On the Anatomy of MCMC-Based Maximum Likelihood Learning of EBMs* (arXiv:1903.12370, AAAI 2020); Grathwohl et al., *JEM* (arXiv:1912.03263); Du et al., *Improved Contrastive Divergence* (arXiv:2012.01316); Vincent 2011; Song et al. 2019 (sliced SM); Gutmann & Hyvärinen 2010 (NCE); Gao et al. 2021 (diffusion recovery likelihood, arXiv:2012.08125); Schröder et al. 2023 (energy discrepancy, arXiv:2307.06431).

---

## 0. Setup, notation, sign conventions

An EBM defines an unnormalized density over $x \in \mathbb{R}^D$:

$$
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}, \qquad Z(\theta) = \int \exp(-E_\theta(x))\,dx .
$$

**Sign conventions (fix these once, library-wide):**

- `energy(x)` returns $E_\theta(x)$, shape `(B,)`. **Low energy = high probability.**
- Unnormalized log-density: `log_prob_unnorm(x) = -energy(x)`.
- Score: $s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$ (does not involve $Z$).
- Samplers **descend** energy: `x <- x - step * grad_E + noise`.
- The MLE surrogate loss is `E(pos) - E(neg)`: minimize it to push data energy *down* and sample energy *up*.

The network is any `nn.Module` mapping `(B, ...) -> (B,)` (or `(B,1)`, squeeze it). For class-conditional / JEM-style models, $E_\theta(x) = -\mathrm{logsumexp}_y f_\theta(x)[y]$, so a classifier is an EBM for free.

---

## 1. Maximum likelihood via MCMC

### 1.1 The gradient identity

$$
\nabla_\theta \, \mathbb{E}_{x\sim p_{\text{data}}}[-\log p_\theta(x)]
= \mathbb{E}_{x^+ \sim p_{\text{data}}}\!\left[\nabla_\theta E_\theta(x^+)\right]
- \mathbb{E}_{x^- \sim p_\theta}\!\left[\nabla_\theta E_\theta(x^-)\right].
$$

The second ("negative phase") expectation is intractable; approximate it with MCMC samples $x^-$. Given a batch of positives $x^+$ and negatives $x^-$, the **surrogate loss** whose $\theta$-gradient equals the estimator is

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_i E_\theta(x_i^+) \;-\; \frac{1}{M}\sum_j E_\theta(\mathrm{sg}[x_j^-]),
$$

where $\mathrm{sg}[\cdot]$ is stop-gradient. **Critical:** $x^-$ was produced by a sampler that itself differentiates $E_\theta$ w.r.t. $x$; you must NOT backprop through that sampling chain into $\theta$ (unless you deliberately add the Improved-CD KL term, §1.5). Detach $x^-$ before the loss.

Note this surrogate is *not* a real objective — its value is meaningless (it can go to $-\infty$); only its gradient matters. Log it, but never early-stop on it.

### 1.2 Contrastive Divergence (CD-k)

Initialize the chain **at the data batch**, run $k$ MCMC steps (classically $k=1$ for RBMs; $k\sim 5\text{–}100$ for ConvNet EBMs):

```python
x_neg = x_pos.clone().detach()
x_neg = sampler.run(energy, x_neg, n_steps=k)  # no grad to theta
loss = energy(x_pos).mean() - energy(x_neg.detach()).mean()
```

Bias: CD-k drops a term of the true gradient (the derivative of the k-step sampler kernel w.r.t. $\theta$); it is *not* the gradient of any fixed objective. Fine in practice for short-run training; data-init CD tends to learn energies accurate near the data manifold but poor far away (bad for generation from noise, fine for e.g. denoising/OOD near data).

### 1.3 Persistent CD (PCD)

Keep the chain state across parameter updates: maintain persistent particles $\{x^-_j\}$; each iteration, continue the chain for $k$ steps from where it stopped, use the result as negatives, store it back. Rationale: the chain approximately tracks the slowly-moving $p_\theta$, so few steps per iteration suffice. Failure mode: if the learning rate is too high the model "outruns" the chains; energies of persistent samples collapse and training diverges. The replay buffer (next) is PCD with a large particle reservoir + occasional reinitialization.

### 1.4 The Du & Mordatch (IGEBM) recipe — the default image recipe

Exact, verified details from arXiv:1903.08689:

- **Replay buffer** $\mathcal{B}$ of **10,000** samples. Each iteration, draw the negative batch from $\mathcal{B}$ with probability **95%** per sample; **5%** are re-initialized from uniform noise $\mathcal{U}[0,1]^D$ (data scaled to $[0,1]$). After sampling, write the resulting negatives back into the buffer (replace random slots / FIFO).
- **Langevin sampling**, $K = 60$ steps for CIFAR-10 / ImageNet-32:
  $$ \tilde{x}^k = \tilde{x}^{k-1} - \frac{\lambda}{2}\,\mathrm{clip}_{[-0.01,\,0.01]}\!\big(\nabla_x E_\theta(\tilde{x}^{k-1})\big) + \omega^k, $$
  with, **in practice, decoupled step and noise**: gradient step size $\lambda$ (10 for CIFAR-10; 100 for ImageNet-128) and *fixed* noise $\omega^k \sim \mathcal{N}(0, 0.005^2 I)$ — i.e. much colder than the theoretically correct $\mathcal{N}(0, \lambda I)$. This is standard: everyone runs a low-temperature/biased Langevin.
  - **Clip the Langevin gradient** elementwise to magnitude $\le 0.01$ (this is what makes step sizes like 10 sane).
  - **Clamp samples** to the data range (e.g. `x.clamp_(0, 1)`) after every step.
- **L2 energy regularization** in the loss, coefficient $\alpha = 1$:
  $$ \mathcal{L} = \alpha\big(E_\theta(x^+)^2 + E_\theta(x^-)^2\big) + E_\theta(x^+) - E_\theta(x^-). $$
  This is the main thing preventing the unbounded drift of the (arbitrary-offset) energy scale.
- **Spectral normalization** on all conv/linear layers (Miyato-style, power iteration).
- **Architecture**: ResNet with LeakyReLU (Improved-CD later switched to Swish/SiLU), average-pooling downsampling (no strided conv), 3×3 stride-1 first conv, zero-init of the second conv in each residual block. **No BatchNorm** (batch statistics interact catastrophically with MCMC negatives; if any norm, use GroupNorm/LayerNorm or nothing).
- **Optimizer**: Adam, lr $10^{-4}$, $\beta_1 = 0$, $\beta_2 = 0.999$; batch 128 pos + 128 neg. Additionally clip parameter gradients that are >3σ w.r.t. Adam's second-moment estimate (a per-parameter adaptive clip; a plain `clip_grad_norm_` is an acceptable substitute).
- **Data noise**: add small noise to data (e.g. uniform dequantization noise 1/256 for images; Gaussian $\sigma \approx 0.005\text{–}0.03$ is common in follow-ups) — smooths $p_{\text{data}}$ and stabilizes the positive phase.

### 1.5 Improved CD (Du et al. 2021) — optional extension

Adds the gradient term CD ignores, expressed as a KL/entropy correction: backprop through the *last few* Langevin steps into $\theta$ via a term $\mathbb{E}[E_{\mathrm{sg}[\theta]}(x^-_\theta)]$ (stop-grad on the energy's parameters, *keep* grad through the sample path). Plus data augmentation applied to buffer samples between chain restarts to aid mixing. Worth an extension hook (`loss.kl_correction: bool`), not a v0.1 default.

### 1.6 The Anatomy paper (Nijkamp et al., 1903.12370): two regimes

Verified findings — these should shape the library's presets:

- Two axes: sign of $\mathbb{E}[E(x^+)] - \mathbb{E}[E(x^-)]$, and whether the finite-step MCMC has converged to the model's steady state. **High-quality synthesis is easier in the *non-convergent* regime** where short-run MCMC ≠ steady state: the model + K-step sampler together act like a generator; the learned density itself may be poor.
- **Non-convergent / short-run recipe** (synthesis-oriented): noise (non-informative) init each iteration, $L \ge 100$ Langevin steps, step size $\epsilon = 1$ *with the noise term turned off* ($\tau=0$) or heavily damped; plain ConvNet, ReLU is fine, **no spectral norm, no norm layers, no regularization needed**; Adam, lr $10^{-4}$.
- **Convergent recipe** (density-oriented, valid steady state): persistent chains, $L = 500$, properly tuned $\epsilon \approx 0.015$ for 32×32 images in $[-1,1]$ (rule of thumb: match the local std of data along the most constrained direction), full noise ($\tau=1$), **SGD** lr $5\times10^{-4}$ (Adam interferes with learning a realistic steady state).
- Implication for the library: expose `temperature`/`noise_scale` and `init` (data / noise / persistent) as first-class sampler options; ship both a "short-run" preset and a "convergent" preset.

### 1.7 Training-step pseudocode (the core loop)

```python
def training_step(model, x_pos, buffer, cfg):
    # ---- negatives ----
    x_neg = buffer.sample(cfg.batch_size, reinit_prob=0.05)  # detached
    model.requires_grad_(False)  # freeze theta during sampling
    x_neg = langevin(
        model,
        x_neg,
        n_steps=cfg.k,
        step_size=cfg.step,
        noise=cfg.noise,
        grad_clip=0.01,
        clamp=(0.0, 1.0),
    )
    model.requires_grad_(True)
    x_neg = x_neg.detach()
    buffer.push(x_neg)

    # ---- loss ----
    x_pos = x_pos + cfg.data_noise * torch.randn_like(x_pos)
    e_pos, e_neg = model(x_pos), model(x_neg)
    loss_cd = e_pos.mean() - e_neg.mean()
    loss_reg = cfg.alpha * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
    loss = loss_cd + loss_reg

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)  # substitute for Adam-3σ clip
    opt.step()
    ema.update(model)
    # log: e_pos.mean(), e_neg.mean(), their gap, grad norms — collapse of the gap to
    # large negative values, or |E| exploding, are the standard divergence signatures.
```

`langevin` internals must get $\nabla_x E$ without building a graph on $\theta$:

```python
def langevin(model, x, n_steps, step_size, noise, grad_clip=None, clamp=None):
    for _ in range(n_steps):
        x = x.detach().requires_grad_(True)
        e = model(x).sum()
        g = torch.autograd.grad(e, x)[0]  # create_graph=False
        if grad_clip is not None:
            g = g.clamp(-grad_clip, grad_clip)
        x = x.detach() - step_size * g + noise * torch.randn_like(x)
        if clamp is not None:
            x = x.clamp(*clamp)
    return x.detach()
```

Freezing params (`requires_grad_(False)`) during sampling is both a large memory/speed win and a correctness guarantee that no sampler graph leaks into the loss.

### 1.8 Replay buffer mechanics

```python
class ReplayBuffer:
    def __init__(self, capacity, shape, init):  # init: callable -> uniform/gaussian noise
        self.data = init(capacity, *shape)  # keep on CPU for big buffers; pin memory

    def sample(self, n, reinit_prob):
        idx = torch.randint(len(self.data), (n,))
        x = self.data[idx].clone()
        mask = torch.rand(n) < reinit_prob  # per-sample, not per-batch
        x[mask] = init(mask.sum(), *shape)
        self._last_idx = idx
        return x

    def push(self, x):
        self.data[self._last_idx] = x.detach().cpu()
```

Details that matter: (i) reinit is *per sample*; (ii) store detached, and store back to the *same* slots (or random slots — both used; same-slot preserves chain identity = closer to PCD); (iii) buffer >> batch (10k vs 128) so each chain is updated rarely — this is the "many slow chains" trick; (iv) samples for FID/visualization should come from fresh long runs or the buffer, using the **EMA** model.

---

## 2. Samplers

### 2.1 ULA / SGLD (the workhorse)

Discretized Langevin diffusion targeting $p_\theta \propto e^{-E_\theta}$:

$$
x_{t+1} = x_t - \epsilon \, \nabla_x E_\theta(x_t) + \sqrt{2\epsilon}\,\eta_t,\qquad \eta_t \sim \mathcal{N}(0, I).
$$

(Equivalent parameterization: step $\lambda/2$ and noise $\mathcal{N}(0,\lambda)$.) Unadjusted ⇒ $O(\epsilon)$ bias in the stationary distribution; nobody Metropolis-corrects at image scale. **Practitioner reality:** step size and noise are decoupled; running noise $\sigma < \sqrt{2\epsilon}$ is sampling a *tempered* target $p^{1/T}$ with $T = \sigma^2 / (2\epsilon)$ — IGEBM's ($\lambda{=}10$ after grad-clip, $\sigma{=}0.005$) is extremely cold. The library API should be:

```python
LangevinSampler(step_size, noise_scale=None,   # None -> sqrt(2*step_size), "correct" ULA
                n_steps, grad_clip=None, clamp=None, noise_decay=1.0)
```

**Annealed Langevin** (Song & Ermon NCSN-style): geometric noise ladder $\sigma_1 > \dots > \sigma_L$, run a few steps at each level with $\epsilon_i = \epsilon \cdot \sigma_i^2/\sigma_L^2$ — relevant if the library supports multi-noise EBMs / DSM-trained models; provide as `AnnealedLangevin`.

### 2.2 MALA

ULA proposal + Metropolis–Hastings correction. Proposal $q(x'|x) = \mathcal{N}\big(x - \epsilon\nabla E(x),\, 2\epsilon I\big)$; accept with probability

$$
\min\Big(1,\; \exp\big(E(x) - E(x')\big)\,\frac{q(x\,|\,x')}{q(x'\,|\,x)}\Big),\qquad
\log q(x'|x) = -\frac{\|x' - x + \epsilon\nabla E(x)\|^2}{4\epsilon} + \text{const}.
$$

Exact stationary distribution. Tune $\epsilon$ for ~50–60% acceptance (optimal ≈ 0.574 in high-dim theory). **Use for low-dim data and evaluation** (2D toys, tabular, AIS transition kernels); at image dimension the step size required for nonzero acceptance is tiny, so practitioners don't use it for training negatives.

### 2.3 HMC

Augment with momentum $v \sim \mathcal{N}(0, M)$, Hamiltonian $H(x,v) = E(x) + \frac{1}{2}v^\top M^{-1} v$; leapfrog integrate $L$ steps of size $\epsilon$; MH-accept with $\min(1, e^{H_{\text{old}} - H_{\text{new}}})$. Target ~65–85% acceptance. Best-in-class for smooth low-dim targets and inside AIS. Same caveat as MALA for images. v0.1: implement standard leapfrog HMC with identity mass; leave NUTS out (point to Pyro/BlackJAX).

### 2.4 What practitioners actually use

| Setting | Sampler | Steps | Step size | Noise |
|---|---|---|---|---|
| CIFAR/ImageNet training negatives (IGEBM) | ULA + buffer | 60 | 10 (grad pre-clipped to ±0.01) | 0.005 fixed |
| JEM (CIFAR, classifier-EBM) | SGLD + buffer | 20 | 1 (on raw grad) | 0.01 fixed |
| Short-run synthesis (Nijkamp) | ULA, noise init | 100 | 1 | ~0 (τ=0) |
| Convergent EBM (Nijkamp) | ULA, persistent | 500 | ε=0.015, correct noise | $\sqrt{2\epsilon}$ |
| 2D / tabular | ULA or MALA/HMC | 100–500 | 1e-3…1e-1 | correct $\sqrt{2\epsilon}$ |

---

## 3. Score matching family

All avoid $Z$ by fitting the score $s_\theta(x) = -\nabla_x E_\theta(x)$. For an EBM library, the model is still $E_\theta$; the score is obtained by autograd (`create_graph=True` needed, since the loss differentiates through $\nabla_x E$).

### 3.1 Exact / implicit score matching (Hyvärinen 2005)

$$
J_{\text{ESM}}(\theta) = \mathbb{E}_{p_{\text{data}}}\Big[\tfrac{1}{2}\|s_\theta(x)\|^2 + \mathrm{tr}\big(\nabla_x s_\theta(x)\big)\Big] + \text{const}.
$$

The trace of the Jacobian (= $-\Delta_x E_\theta$) costs $D$ backward passes ⇒ only viable for small $D$ (2D toys, tabular ≤ ~50 dims). Implement with a loop of `autograd.grad` per dimension, or Hutchinson (which *is* sliced SM). Note: consistent only if $p_\theta$ is smooth and $p_{\text{data}}$ has full support — it famously cannot see the relative weights of well-separated modes ("blindness").

### 3.2 Denoising score matching (Vincent 2011)

Perturb data with $q_\sigma(\tilde{x}|x) = \mathcal{N}(x, \sigma^2 I)$:

$$
J_{\text{DSM}}(\theta) = \tfrac{1}{2}\,\mathbb{E}_{x \sim p_{\text{data}},\, \tilde{x} \sim q_\sigma(\cdot|x)}
\Big\| s_\theta(\tilde{x}) + \frac{\tilde{x} - x}{\sigma^2} \Big\|^2 .
$$

No second derivatives w.r.t. $x$ dimension-by-dimension (one extra backward for $\nabla_x E$, then backprop through it once). Estimates the score of the *smoothed* density $p_{\text{data}} * \mathcal{N}(0,\sigma^2)$ — bias grows with $\sigma$, variance blows up as $\sigma \to 0$. Practical form scales the residual by $\sigma$ for conditioning: $\mathbb{E}\|\sigma s_\theta(\tilde x) + \varepsilon\|^2$, $\tilde x = x + \sigma\varepsilon$. Multi-$\sigma$ DSM (a noise-conditional $E_\theta(x,\sigma)$, NCSN-style) is the bridge to diffusion — v0.1 should support a single fixed $\sigma$ and leave the ladder as an extension.

```python
def dsm_loss(model, x, sigma):
    eps = torch.randn_like(x)
    x_t = (x + sigma * eps).requires_grad_(True)
    e = model(x_t).sum()
    score = -torch.autograd.grad(e, x_t, create_graph=True)[0]
    return 0.5 * ((sigma * score + eps) ** 2).flatten(1).sum(-1).mean()
```

### 3.3 Sliced score matching (Song et al. 2019)

Random projections replace the Jacobian trace:

$$
J_{\text{SSM}}(\theta) = \mathbb{E}_{v}\,\mathbb{E}_{x}\Big[ v^\top \nabla_x s_\theta(x)\, v + \tfrac{1}{2}\big(v^\top s_\theta(x)\big)^2 \Big],
$$

$v \sim \mathcal{N}(0,I)$ or Rademacher. The first term is one Hessian-vector product: `hvp = grad((score * v).sum(), x)`, then `v·hvp`. **SSM-VR** (variance-reduced, use with $v$ Gaussian): replace $\frac12 (v^\top s)^2$ by $\frac12 \|s\|^2$ — usually the better default. Preferred when you want the *clean-data* score in medium/high dims without DSM's noise bias.

**When to prefer which:** ESM only for tiny $D$ / exactness tests; DSM for images or whenever slight smoothing is acceptable (fastest, most robust); SSM(-VR) when the exact score matters and $D$ is too large for ESM. All of them learn shape-of-density well but, like ESM, are weak on inter-mode weights unless noise is large (DSM with big $\sigma$) — which motivates the annealed/multiscale versions.

---

## 4. Noise-contrastive estimation (NCE)

Turn density estimation into classification against a known noise distribution $q(x)$ (must have tractable density and sampling; must cover $p_{\text{data}}$'s support). With $\nu = $ noise-to-data ratio and an **explicitly learnable normalizer** $c$ (so $\log p_\theta(x) = -E_\theta(x) - c$; NCE, unlike MLE, can estimate $Z$):

$$
h_\theta(x) = \log p_\theta(x) - \log q(x) - \log \nu, \qquad
J_{\text{NCE}} = -\,\mathbb{E}_{p_{\text{data}}}\big[\log \sigma(h_\theta(x))\big] \; - \; \nu\,\mathbb{E}_{q}\big[\log\big(1 - \sigma(h_\theta(x))\big)\big].
$$

This is a well-defined (strictly proper) objective; as $\nu \to \infty$ it approaches MLE. **Failure mode:** if $q$ and $p_{\text{data}}$ barely overlap, the classifier saturates and gradients vanish — NCE works great in low dims / with strong noise models (e.g. a fitted Gaussian/flow as $q$), poorly with naive noise on images.

**Conditional / ranking variants:** for conditional EBMs $E_\theta(x|c)$, RankingNCE (Ma & Collins 2018) / InfoNCE-style: given one positive and $M$ negatives from $q(\cdot|c)$,

$$
J = -\,\mathbb{E}\left[\log \frac{e^{-E_\theta(x^+|c)}/q(x^+|c)}{\sum_{j=0}^{M} e^{-E_\theta(x_j|c)}/q(x_j|c)}\right],
$$

which never needs $Z(c)$ (it cancels). This is the loss to expose for conditional EBMs over discrete or structured outputs. **Flow-contrastive estimation** (Gao et al. 2020): jointly train a normalizing flow as an adaptive $q$ — extension point, not v0.1.

```python
def nce_loss(model, log_c, x_data, noise_dist, nu):
    x_noise = noise_dist.sample((nu * len(x_data),))

    def logit(x):
        return (-model(x) - log_c) - noise_dist.log_prob(x) - math.log(nu)

    return -(F.logsigmoid(logit(x_data)).mean() + nu * F.logsigmoid(-logit(x_noise)).mean())
```

(`log_c` is a scalar `nn.Parameter` — the library's `Energy` wrapper should own an optional `log_z` parameter used only by NCE-type losses.)

---

## 5. Other estimators (extension-point survey)

- **Energy discrepancy** (Schröder et al. 2023): MCMC-free, score-free. For Gaussian perturbation scale $t$, $M$ contrastive draws, stabilizer $w$:
  $$ \mathcal{L}(\theta) = \mathbb{E}_{x\sim p_{\text{data}},\, y \sim \mathcal{N}(x, tI)}\Big[\log\Big(\tfrac{w}{M} + \tfrac{1}{M}\sum_{i=1}^{M} e^{\,E_\theta(x) - E_\theta(y + \sqrt{t}\,\xi_i)}\Big)\Big],\ \xi_i \sim \mathcal{N}(0,I). $$
  Cheap, stable, interpolates SM ($t\to0$) and MLE-like behavior ($t\to\infty$). Cheap to add later; the loss only needs batched energy evals — no new model API.
- **Diffusion recovery likelihood** (Gao et al. 2021): define noisy marginals $\tilde x = x + \sigma_t \varepsilon$; learn conditional EBMs by MLE of the *recovery* posterior $p_\theta(x \mid \tilde x) \propto \exp\big(-E_\theta(x, t) - \|x - \tilde x\|^2/2\sigma_t^2\big)$, which is nearly unimodal ⇒ short Langevin suffices. Sampling = ancestral chain over $t$. Requires a time/noise-conditional energy and a scheduler — this is the main reason the core `Energy` interface should allow optional conditioning arguments from day one.
- **Cooperative learning (CoopNets, Xie et al.)**: a generator $g_\phi$ proposes MCMC initializations; the EBM's Langevin refines them; $g_\phi$ regresses onto the refined samples ("MCMC teaching"). Amortizes sampling. Extension point: the sampler API should accept a pluggable `init_fn` (noise / buffer / generator), which makes CoopNets and VERA/amortized-negative-sampler methods drop-in.
- **Adversarial / variational (VERA, Grathwohl et al. 2021; f-EBM)**: negative phase via a trained sampler network with an entropy correction. Same `init_fn`/`negative_sampler` hook covers it.

Library conclusion: v0.1 does not implement these, but the API must have (a) optional conditioning on the energy, (b) pluggable negative-sample sources, (c) losses that are plain functions of batched energies.

---

## 6. Stabilization: what actually matters

Distilled from 2101.03288, 1903.08689, 1903.12370, and JEM/Improved-CD lore, roughly in order of importance:

1. **Stop-gradient on negatives + frozen params during sampling.** The #1 correctness bug in naive implementations.
2. **Energy L2 regularization** $\alpha(E(x^+)^2 + E(x^-)^2)$, $\alpha \approx 1$ (IGEBM) or 0.1–1. Pins the arbitrary energy offset/scale; without it $|E|$ drifts and Langevin step sizes go stale. (Alternative used in some code: regularize the *gap* $(E^+ - E^-)^2$.)
3. **Langevin gradient clipping** (±0.01 elementwise, IGEBM) and **sample clamping** to the data range each step.
4. **No BatchNorm, ever.** Either spectral norm on all layers (IGEBM — smooths the energy landscape, bounds Lipschitz constant, helps convergent-ish training) or *no normalization at all* (Nijkamp — sufficient in the short-run regime). Make `spectral_norm` a one-flag wrapper.
5. **Smooth activations**: Swish/SiLU (or LeakyReLU ≥ 0.05 slope). ReLU's zero second derivative degrades Langevin gradients and kills score-matching losses (which need $\nabla_x^2 E$); GELU/SiLU is the modern default.
6. **Small noise added to data** (dequantization 1/256, or Gaussian 0.005–0.03): prevents the positive phase from sharpening onto a measure-zero manifold; markedly stabilizes training.
7. **EMA of weights** (decay 0.999–0.9999) for evaluation/sampling — large FID improvement, standard since NCSN/Improved-CD.
8. **Optimizer**: Adam(0.0 or 0.9, 0.999), lr 1e-4, + global grad-norm clip (~0.1–1.0 or IGEBM's adaptive 3σ clip). For convergent training, plain SGD.
9. **Monitor** $\bar E^+ - \bar E^-$: healthy training hovers near 0 with small oscillation; sustained large positive gap ⇒ sampler too weak (more steps / bigger step); runaway negative ⇒ divergence (lower lr, more reg, more chain reinit).
10. **Low temperature sampling at train time is a feature, not a bug** (short-run regime), but document that the resulting $E_\theta$ is then not a calibrated log-density.

---

## 7. Evaluation

- **Log-likelihood via AIS** (Neal 2001): estimate $\log Z$ by annealing from a tractable $p_0$ (broad Gaussian / uniform on the data cube) through $p_\beta \propto p_0^{1-\beta} e^{-\beta E_\theta}$, with HMC/MALA transitions at each of ~1e3–1e5 temperatures; $\hat Z = \frac{1}{S}\sum_s \prod_t w_t^{(s)}$ in log-space via logsumexp. Report $-E(x) - \log \hat Z$ per dim in bits: divide by $D \ln 2$. AIS gives a stochastic lower bound on $\log Z$… hence an *upper* bound flavor on likelihood — pair with reverse AIS / RAISE for a bracket if rigor is needed. Ship `evaluate.ais_log_z(energy, prior, n_temps, n_chains, transition="hmc")`.
- **Sample quality**: FID (and IS) on 50k samples from the EMA model via long-run Langevin or the replay buffer. Reference points: IGEBM CIFAR-10 FID ≈ 38, Improved CD ≈ 25, recovery likelihood ≈ 9.6.
- **OOD detection**: score with $-E_\theta(x)$ (or JEM's logsumexp logits, or the gradient-norm score $\|\nabla_x E\|$); report AUROC vs SVHN/CIFAR-100/interp. Note the known caveat: likelihood-based OOD can invert (assign OOD *higher* density); energy-gradient scores are more robust.
- **Energy histograms**: overlay $E_\theta$ histograms of (train data, held-out data, buffer/model samples, OOD data). The cheapest, most informative diagnostic — should be a built-in logging utility. Train≪held-out gap ⇒ memorization; samples≪data ⇒ sampler found spurious minima.
- **2D toys**: exact visual density $e^{-E}$ heatmap vs. data, plus MMD between samples and data.

---

## 8. Recommended defaults

### 8.1 2D toy quickstart (two-moons, rings, checkerboard; data scaled to ~[-4,4]²)

- Model: MLP 2→128→128→128→1, SiLU, no norm.
- Loss: CD with replay buffer; $\alpha_{\text{reg}} = 0.1$. (Also ship SSM-VR and DSM($\sigma{=}0.1$) as one-line swaps — they "just work" here.)
- Sampler: **proper** ULA — step $\epsilon = 10^{-2}$, noise $\sqrt{2\epsilon} \approx 0.14$, $K = 100$; MALA optional for exactness.
- Buffer 8192, reinit 5%, init $\mathcal{U}[-4,4]^2$; batch 256; Adam(0.9, 0.999) lr $10^{-3}$; grad-norm clip 1.0; 5k–20k iters. No sample clamping needed; no Langevin grad clipping needed.

### 8.2 CIFAR-scale images (data in [0,1]; the IGEBM/JEM lineage)

- Model: 8–12-block ResNet, SiLU/LeakyReLU, avg-pool downsampling, no norm layers, `spectral_norm=True` by default.
- Loss: CD + buffer, $\alpha_{\text{reg}} = 1.0$ on $E^2$; data noise: dequantize + Gaussian $\sigma = 0.005$–0.03.
- Sampler (short-run preset, default): ULA, $K = 60$, grad clip ±0.01, step 10, noise 0.005, clamp [0,1]. (JEM-flavored cheaper preset: $K = 20$, step 1, noise 0.01.)
- Buffer 10,000, reinit 5%, init $\mathcal{U}[0,1]$; batch 128; Adam($\beta_1{=}0.0$, $\beta_2{=}0.999$) lr $10^{-4}$; grad-norm clip; EMA 0.999.
- Convergent preset (opt-in): persistent init, $K = 500$, $\epsilon = 0.015$ (data in [-1,1]), full noise, SGD lr $5\times10^{-4}$, no spectral norm.

---

## 9. Implementation checklist (the load-bearing subtleties)

1. `p ∝ exp(-E)`; sampler subtracts the gradient; loss is `E(pos).mean() - E(neg).mean()`. Getting any one sign wrong trains an anti-model that "works" for a few hundred steps then explodes — add a unit test on a quadratic energy where the stationary distribution is known in closed form.
2. Negatives detached; parameters frozen during sampling; sampler uses `torch.autograd.grad(E.sum(), x)` with `create_graph=False`. Score-matching losses are the opposite: `create_graph=True` and the loss backprops through $\nabla_x E$.
3. Buffer: per-sample reinit; write-back after sampling; store on CPU detached; sample with `.clone()`.
4. Decoupled `step_size` / `noise_scale` with `noise_scale=None` meaning the theoretically correct $\sqrt{2\epsilon}$; presets encode the practitioner (cold) values.
5. The CD loss value is not a training signal — surface energy-gap and gradient-norm diagnostics instead.
6. `model.eval()` vs `train()` must be irrelevant (no BN/dropout) — enforce or warn.
