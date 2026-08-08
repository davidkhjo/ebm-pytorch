# ebm-pytorch

A small, reliable PyTorch library for training and using **energy-based models** (EBMs).

**Docs: https://davidkhjo.github.io/ebm-pytorch/**

An EBM defines an unnormalized density `p(x) ∝ exp(-E(x))` through a neural network
`E: (B, *shape) -> (B,)`. This library provides the pieces that every EBM project
rebuilds from scratch — MCMC samplers, training losses, replay buffers, diagnostics —
as small composable objects with tested, correct defaults.

> Not to be confused with *Explainable Boosting Machines* (interpretml), which also
> go by "EBM". This is the deep-learning kind: LeCun et al. (2006), Du & Mordatch
> (2019), Song & Kingma (2021).

## Install

```bash
pip install -e ".[dev]"        # from a checkout
# runtime dependency is just torch>=2.0; add [viz] for matplotlib plotting
```

## Quickstart

```python
import torch, ebm

energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
sampler = ebm.LangevinDynamics(step_size=1e-2, steps=60)
loss_fn = ebm.ContrastiveDivergence(sampler, buffer=ebm.ReplayBuffer(8192, (2,)))

trainer = ebm.Trainer(energy, loss_fn, lr=1e-3)
trainer.fit(ebm.datasets.two_moons(8192), steps=3000, batch_size=256)

samples = sampler.sample(energy, torch.randn(2000, 2), steps=500)
```

The `Trainer` is optional sugar — the underlying loop is plain PyTorch:

```python
opt = torch.optim.Adam(energy.parameters(), lr=1e-3)
for step in range(3000):
    x = data[torch.randint(len(data), (256,))]
    out = loss_fn(energy, x)  # LossOutput(loss, metrics, x_neg)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
```

## What's in the box

| Piece | Contents |
|---|---|
| **Energies** | any callable `(B, *shape) -> (B,)`; `EnergyModel` wrapper, `ebm.score`, `nets.MLPEnergy` / `nets.ConvEnergy` (SiLU, optional spectral norm, no batch norm); noise-conditional variants for NCSN |
| **Samplers** | `LangevinDynamics` (ULA/SGLD with decoupled step/noise, gradient clipping, sample clamping), `MALA`, `HMC`, `GibbsWithGradients` (binary) + `CategoricalGibbsWithGradients` (one-hot), `AnnealedLangevinDynamics` (noise ladder) |
| **Losses** | `ContrastiveDivergence` (CD-k / persistent CD via `ReplayBuffer`), `DiffusionRecoveryLikelihood` + `drl_sample` (recovery likelihood on a noise ladder — the most stable EBM training known), `DenoisingScoreMatching` + `MultiSigmaDenoisingScoreMatching` (NCSN), `SlicedScoreMatching` (VR variant), `NoiseContrastiveEstimation` (learnable `log_z`), `JEMLoss` (classifier + EBM) |
| **JEM** | `ClassifierEnergy`: any K-class classifier as an EBM (`E(x) = -logsumexp logits`), class-conditional sampling via `.condition(y)` |
| **Composition** | `SumEnergy` (product of experts), `MixtureEnergy`, `TemperedEnergy` — energies compose like densities and nest freely |
| **Training** | thin `Trainer` (device, optimizer incl. loss params, EMA weights, supervised `(x, y)` batches, metric history), `EMA` |
| **Eval** | **`ais_log_z` / `reverse_ais_log_z`** — bracket `log Z` from below and above for honest log-likelihoods (nats or bits/dim), `eval.frechet_distance` (FID with any feature extractor), `eval.ood_auroc`, energy histograms |
| **Data & viz** | 2D toy datasets (`two_moons`, `eight_gaussians`, `checkerboard`, `rings`, `spirals`), `viz.energy_contour` / `plot_samples` / `energy_histogram` |

## Conventions that matter

- **Sign:** `p ∝ exp(-E)`. Samplers *descend* the energy gradient; training pushes
  data energy *down*.
- **Stop-gradients:** MCMC negatives are detached and the energy network's
  parameters are frozen during sampling; parameter gradients flow only through
  `E(x_data) - E(x_neg)`. Score-matching losses do the opposite and backprop
  through the score (`create_graph=True`).
- **The CD loss value is not a convergence signal.** It hovers near zero (or goes
  negative) at equilibrium — monitor `metrics["energy_gap"]` and the energy
  histograms instead.
- **When plain CD won't converge, tether the sampler.** Diffusion recovery
  likelihood trains the same noise-conditional energies as NCSN but samples
  only the *conditional* `p(x | x̃) ∝ exp(-E(x,σ) - ‖x̃-x‖²/2s²)` between
  adjacent noise levels — near-unimodal, so short-run MCMC genuinely mixes.
- **Two Langevin regimes, one keyword apart:** `noise_scale=None` gives the
  mathematically correct `sqrt(2*step_size)` noise (convergent training, toy data);
  practitioner "short-run" image training uses a cold, decoupled noise, e.g.
  `LangevinDynamics(step_size=10.0, noise_scale=0.005, grad_clip=0.01, steps=60)`
  with `ReplayBuffer(10_000, shape, reinit_prob=0.05)` and `energy_reg=1.0`
  (the IGEBM recipe).

## Development

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest            # includes distribution-correctness tests for the samplers
uv run ruff check .
```

Design notes and the research that informed this library live in `research/`.

## Evaluating log-likelihood

```python
lower = ebm.ais_log_z(energy, shape=(2,), n_temps=1000, n_chains=256)
upper = ebm.reverse_ais_log_z(energy, lower.samples, n_temps=1000)
bits = ebm.log_likelihood(energy, x_test, lower.log_z, dim=2)  # bits/dim
```

Forward AIS is a stochastic *lower* bound on `log Z`; reverse AIS (run from
model samples) is a stochastic *upper* bound. When the two agree, trust the
number; when they don't, increase `n_temps` and check `.ess`.

## Roadmap

PyPI release, docs site.

## License

MIT
