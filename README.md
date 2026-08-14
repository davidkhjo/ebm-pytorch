# ebmkit

[![CI](https://github.com/davidkhjo/ebmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/davidkhjo/ebmkit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/ebmkit.svg)](https://pypi.org/project/ebmkit/)
[![Python](https://img.shields.io/pypi/pyversions/ebmkit.svg)](https://pypi.org/project/ebmkit/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/davidkhjo/ebmkit/blob/main/LICENSE)

A small, reliable PyTorch toolkit for training and using **energy-based models** —
the MCMC samplers, training losses, replay buffers, and diagnostics every EBM
project otherwise rebuilds from scratch, as composable objects with tested defaults.

An EBM is an unnormalized density `p(x) ∝ exp(-E(x))` defined by a network
`E: (B, *shape) -> (B,)`. `torch` is the only runtime dependency.

> Not the *Explainable Boosting Machines* that also go by "EBM" — this is the
> deep-learning kind (LeCun et al. 2006; Du & Mordatch 2019; Song & Kingma 2021).

## Install

```bash
pip install ebmkit          # runtime dependency is just torch>=2.0
pip install "ebmkit[viz]"   # + matplotlib for the plotting helpers
```

The import name is `ebm`:

```python
import torch, ebm

energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
sampler = ebm.LangevinDynamics(step_size=1e-2, steps=60)
loss_fn = ebm.ContrastiveDivergence(sampler, buffer=ebm.ReplayBuffer(8192, (2,)))

trainer = ebm.Trainer(energy, loss_fn, lr=1e-3)
trainer.fit(ebm.datasets.two_moons(8192), steps=3000, batch_size=256)

samples = sampler.sample(energy, torch.randn(2000, 2), steps=500)
```

The `Trainer` is optional sugar — the loop underneath is plain PyTorch (each loss
returns `LossOutput(loss, metrics, x_neg)`; call `out.loss.backward()`).

## What's in the box

| Piece | Contents |
|---|---|
| **Energies** | any callable `(B, *shape) -> (B,)`; `nets.MLPEnergy` / `ConvEnergy` / `ConvClassifier` (SiLU, optional spectral norm, no batch norm), `nets.IsingEnergy` / `PottsEnergy` (discrete lattices), noise-conditional variants for NCSN; `EnergyModel`, `ebm.score` |
| **Samplers** | `LangevinDynamics` (ULA/SGLD), `MALA`, `HMC`, `GibbsWithGradients` + `CategoricalGibbsWithGradients`, `AnnealedLangevinDynamics` |
| **Losses** | `ContrastiveDivergence` (CD-k / persistent CD), `DiffusionRecoveryLikelihood` + `drl_sample`, `DenoisingScoreMatching` / `MultiSigmaDenoisingScoreMatching` (NCSN), `SlicedScoreMatching`, `NoiseContrastiveEstimation`, `JEMLoss` |
| **Composition** | `SumEnergy` (product of experts), `MixtureEnergy`, `TemperedEnergy` — energies compose like densities and nest |
| **Training** | thin `Trainer` (device, EMA, supervised batches, `save`/`load` checkpointing), `ReplayBuffer`, `EMA` |
| **Eval** | `ais_log_z` / `reverse_ais_log_z` (bracket `log Z`), `bits_per_dim`, `frechet_distance` (FID), `mmd`, `ood_auroc` |
| **Data & viz** | 2D toys (`two_moons`, `eight_gaussians`, `checkerboard`, `rings`, `spirals`) and torchvision-free image loaders (`mnist`, `fashion_mnist`, `cifar10`, `cifar100`); `viz.energy_contour` / `plot_samples` / `energy_histogram` / `show_images` |

## Docs

- [Training methods](https://github.com/davidkhjo/ebmkit/blob/main/docs/training.md) — choosing a loss; CD/PCD, NCSN, DRL, NCE, JEM
- [Samplers](https://github.com/davidkhjo/ebmkit/blob/main/docs/sampling.md) — Langevin/MALA/HMC, Gibbs-with-Gradients, annealed
- [Evaluation](https://github.com/davidkhjo/ebmkit/blob/main/docs/evaluation.md) — log-Z bracketing, FID, MMD, OOD, bits/dim
- [Composition](https://github.com/davidkhjo/ebmkit/blob/main/docs/composition.md) — products, mixtures, tempering
- [Benchmarks](https://github.com/davidkhjo/ebmkit/blob/main/docs/benchmarks.md) — every loss family scored on the eval stack

## Examples

Runnable scripts in [`examples/`](https://github.com/davidkhjo/ebmkit/tree/main/examples) (`python examples/<name>.py`):

- `train_two_moons.py` — the canonical 2D contrastive-divergence smoke test
- `train_jem.py` / `train_mnist_jem.py` — classify, generate, and detect OOD with one network
- `train_mnist.py` — the image-scale IGEBM short-run recipe
- `train_composition.py` — product of experts / mixture / tempering, without retraining
- `train_ising.py` / `train_potts.py` — discrete lattices via (categorical) Gibbs-with-Gradients
- `train_ncsn.py` — score-based generation: multi-sigma denoising + annealed Langevin
- `train_cifar_ood.py` — energy-based OOD at color scale (CIFAR-10 vs CIFAR-100)
- `checkpoint_resume.py` — save a run and resume it in a fresh process

## Conventions

- **Sign:** `p ∝ exp(-E)` — low energy is high probability. Samplers *descend*
  the energy gradient; training pushes data energy *down*. Never flip this.
- **Stop-gradients:** MCMC negatives are detached and the energy's parameters are
  frozen during sampling; score-matching losses instead keep the graph
  (`create_graph=True`).
- **The CD loss value is not a convergence signal** — it hovers near zero at
  equilibrium; watch `metrics["energy_gap"]` and energy histograms.

See [`docs/`](https://github.com/davidkhjo/ebmkit/tree/main/docs) and [CONTRIBUTING.md](https://github.com/davidkhjo/ebmkit/blob/main/CONTRIBUTING.md) for the rest.

## Development

```bash
uv run pytest        # tests (CPU-only, seeded)
uv run ruff check .  # lint
uv run mypy          # type-check
```

## Citation

If you use ebmkit in your research, please cite it — see [CITATION.cff](https://github.com/davidkhjo/ebmkit/blob/main/CITATION.cff).

## License

MIT — see [LICENSE](https://github.com/davidkhjo/ebmkit/blob/main/LICENSE).
