# Sampling

All samplers target $p(x) \propto e^{-E(x)}$ by *descending* the energy
gradient, share the interface `sampler.sample(energy, x_init, steps=...,
return_trajectory=...)`, freeze the energy network's parameters while running,
and return detached samples.

## Catalog

| Sampler | Exact? | Cost/step | Use for |
|---|---|---|---|
| `LangevinDynamics` (ULA/SGLD) | biased | 1 gradient | training negatives, toy sampling |
| `MALA` | exact | 2 gradients | final samples, AIS transitions |
| `HMC` | exact | L+1 gradients | smooth targets, AIS transitions |
| `GibbsWithGradients` | exact | 2 gradients | binary data `{0,1}^D` |
| `CategoricalGibbsWithGradients` | exact | 2 gradients | one-hot data `(B, *dims, K)` |
| `AnnealedLangevinDynamics` | biased | 1 gradient | NCSN generation down a noise ladder |

## The two Langevin regimes

One keyword apart:

```python
# Mathematically correct: noise = sqrt(2 * step_size); targets p ∝ exp(-E).
ebm.LangevinDynamics(step_size=1e-2, steps=100)

# Practitioner "short-run" image regime: huge step, cold decoupled noise.
ebm.LangevinDynamics(step_size=10.0, noise_scale=0.005, grad_clip=0.01, steps=60)
```

The first converges to the model distribution and is what you want for toy
data, MCMC diagnostics, and anything where correctness matters. The second is
an optimization-with-jitter that works empirically for image EBM training —
use it with a `ReplayBuffer` and `energy_reg`, and don't expect its samples to
be calibrated draws from $p$.

## Tuning

- **MALA / HMC** expose `last_accept_rate`. Aim for ~0.5–0.7 (MALA) or
  ~0.65–0.9 (HMC); adjust `step_size` accordingly. AIS does this for you
  automatically (`adapt_step_size=True`).
- **Gradient clipping** (`grad_clip`) is a stability tool for training
  negatives, not for correct sampling — it biases the chain.
- **Discrete samplers** propose one flip / one category change per step, so
  budget `steps` at a few times the number of positions; the first-order
  proposal is exact for linear energies (accept rate ≈ 1), and acceptance
  degrades gracefully with interaction strength. See
  [`examples/train_ising.py`](examples.md#discrete-ebms-a-2d-ising-lattice) for
  `GibbsWithGradients` on a 2D Ising lattice.
- **`AnnealedLangevinDynamics(sigmas, step_size, steps_per_sigma)`** follows
  NCSN Algorithm 1: per level $\alpha_i = \epsilon\,\sigma_i^2/\sigma_L^2$,
  update $x \leftarrow x - (\alpha_i/2)\nabla E(x,\sigma_i) +
  \sqrt{\alpha_i}\,\xi$. Pair with `MultiSigmaDenoisingScoreMatching`.

## Writing your own

Subclass `Sampler`, implement `step(energy, x) -> x'`, and use the
`Sampler._energy_grad(energy, x)` helper for gradients — `sample()` then
handles parameter freezing, `torch.enable_grad`, detaching, and trajectories
for free.
