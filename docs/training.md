# Training methods

Five loss families, one interface: every loss is an `nn.Module` called as
`loss_fn(energy, x)` (or `loss_fn(energy, x, y)` when supervised) and returns
`LossOutput(loss, metrics, x_neg)`.

## Choosing a loss

| You have | Use | Why |
|---|---|---|
| Low-dimensional / toy data | `ContrastiveDivergence` + `LangevinDynamics` | Simple, exact enough, fast to iterate |
| Images or other high-dim data | `ContrastiveDivergence` + `ReplayBuffer` (the IGEBM recipe) | Persistent chains amortize mixing |
| Unstable CD training | `DiffusionRecoveryLikelihood` | The tethered conditional is near-unimodal — short-run MCMC actually mixes |
| No MCMC budget at all | `MultiSigmaDenoisingScoreMatching` or `SlicedScoreMatching` | Score matching needs no negatives |
| A tractable noise distribution | `NoiseContrastiveEstimation` | Turns density estimation into classification; learns `log Z` as a parameter |
| Labels, and you want a classifier too | `JEMLoss` | One network that classifies and generates |

## Contrastive divergence

Maximum likelihood's gradient is `∇E(data) - ∇E(model samples)`; CD approximates
the model samples with a short MCMC run:

```python
loss_fn = ebm.ContrastiveDivergence(
    ebm.LangevinDynamics(step_size=1e-2, steps=60),
    buffer=ebm.ReplayBuffer(8192, (2,)),   # persistent CD
    energy_reg=0.1,                        # keeps energy magnitudes bounded
)
```

The image-scale ("IGEBM") recipe is the same object with practitioner
constants: `LangevinDynamics(step_size=10.0, noise_scale=0.005,
grad_clip=0.01, steps=60)`, `ReplayBuffer(10_000, shape, reinit_prob=0.05)`,
`energy_reg=1.0`, samples clamped to `(-1, 1)`.

!!! warning "The CD loss value is not a convergence signal"
    `E(x⁺) - E(x⁻)` hovers near zero (and can go negative) at equilibrium.
    Watch `metrics["energy_gap"]` and energy histograms instead; a training
    run where the gap diverges to large negative values means the sampler
    can't keep up with the energy landscape — regularize harder or tether
    (see below).

## Diffusion recovery likelihood

The most stable known EBM training (Gao et al. 2021). Instead of sampling the
multimodal marginal, sample the *recovery posterior* between adjacent levels
of a noise ladder \(\sigma_1 > \dots > \sigma_L\):

\[
p(x \mid \tilde{x}) \propto \exp\!\big({-E(x, \sigma_{t+1})}
  - \|\tilde{x} - x\|^2 / 2s_t^2\big),
\qquad s_t^2 = \sigma_t^2 - \sigma_{t+1}^2 .
\]

The quadratic tether makes this near-unimodal, so 30 Langevin steps genuinely
mix. Requires a noise-conditional energy:

```python
net = ebm.nets.NoiseConditionalMLPEnergy(dim=2)
sigmas = ebm.geometric_sigmas(3.0, 0.05, 10)
loss_fn = ebm.DiffusionRecoveryLikelihood(sigmas, mcmc_steps=30)
# ... train, then generate by walking the ladder down:
samples = ebm.drl_sample(net, sigmas, 2000, (2,))
```

## Score matching

No negatives, no MCMC — match \(\nabla_x \log p\) instead. `DenoisingScoreMatching(sigma)`
for a single noise scale; `MultiSigmaDenoisingScoreMatching(sigmas)` is the
NCSN objective across a ladder (pair it with `AnnealedLangevinDynamics` for
generation); `SlicedScoreMatching` avoids noise entirely via random
projections. Note the energy's *additive constant* is invisible to the score —
the output bias never receives gradient, which is expected.

## Noise-contrastive estimation

`NoiseContrastiveEstimation(noise_dist, noise_ratio)` classifies data against
noise samples; `log Z` is learned as an explicit parameter, so the energy
comes out (approximately) *normalized*. Works best when the noise distribution
overlaps the data well.

## JEM: classifier as EBM

Any K-class classifier is secretly an EBM: `E(x) = -logsumexp(logits)`.
`ClassifierEnergy` wraps the classifier; `JEMLoss` adds cross-entropy to a CD
loss on the marginal energy:

```python
clf = ebm.ClassifierEnergy(my_classifier)          # (B, *shape) -> (B, K)
loss_fn = ebm.JEMLoss(ebm.ContrastiveDivergence(sampler), cd_weight=1.0)
trainer.fit((X, Y), ...)                           # supervised batches
samples_of_class_3 = sampler.sample(clf.condition(3), x0)
```
