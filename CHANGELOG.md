# Changelog

## Unreleased

- `datasets.cifar10`: CIFAR-10 as `(N, 3, 32, 32)` float32 in `[-1, 1]` with no
  torchvision dependency (downloads the binary distribution and parses the
  `.bin` batches directly — no pickle; cached in `~/.cache/ebm-pytorch`).
  Native 32x32 for the default `ConvEnergy` / `ConvClassifier`.

## 0.7.0 — 2026-08-07

- `nets.ConvClassifier`: the `ConvEnergy` trunk with a position-sensitive
  (flattened, not pooled) K-logit head, so `ClassifierEnergy` (JEM) works on
  images out of the box. Pooled logits are translation-invariant "bags of
  class evidence", and conditional sampling on them tiles the canvas with
  class strokes instead of drawing one centered object.
- `JEMLoss(conditional_negatives=True)`: negative chains target a uniformly
  random class's joint energy `E(x, y)` (as in the JEM reference
  implementation) while the loss terms stay marginal. Without it,
  class-conditional generation at image scale produces adversarial textures —
  sampling a logit that CD never trained.
- `datasets.fashion_mnist`: Fashion-MNIST with the same format and scaling as
  `mnist` — the classic out-of-distribution counterpart for evaluating hybrid
  models with `eval.ood_auroc`.
- New example: `examples/train_mnist_jem.py` — one MNIST network that
  classifies, flags Fashion-MNIST as OOD by energy, and draws digits of a
  requested class, using the same IGEBM sampler constants as the
  unconditional example.

## 0.6.0 — 2026-08-05

- `datasets.mnist`: MNIST as `(N, 1, 28, 28)` float32 in `[-1, 1]` with no
  torchvision dependency (downloads and parses the raw IDX files; cached in
  `~/.cache/ebm-pytorch`).
- New example: `examples/train_mnist.py` — the IGEBM short-run recipe
  generating digits in ~5 minutes on Apple Silicon.
- `NoiseConditionalConvEnergy`: sigma now modulates every block via FiLM
  (scale + bias, zero-initialized) instead of a bias after the stem only, and
  `spectral_norm` defaults to `False` — bias-only conditioning and capped
  gradients were too weak for recovery-likelihood training. Docs now explain
  when DRL's generation path needs large budgets and when to prefer the
  short-run CD recipe.

## 0.5.2 — 2026-08-03

- `eval.mmd`: unbiased squared maximum mean discrepancy (RBF kernel, median
  heuristic or explicit bandwidth) — catches distribution mismatch that
  `frechet_distance`'s two moments miss.
- Benchmarks page (`docs/benchmarks.md` + `examples/benchmark_losses.py`):
  all six training methods on two-moons, scored by FD, MMD, AIS-bracketed
  test log-likelihood, and wall time.

## 0.5.1 — 2026-08-03

- Documentation site at https://dkjo8.github.io/ebm-pytorch/ (mkdocs-material +
  mkdocstrings API reference; deployed from main).
- PyPI trusted-publishing workflow: GitHub releases now publish `ebm-pytorch`
  to PyPI automatically.

## 0.5.0 — 2026-07-30

- Diffusion recovery likelihood (Gao et al. 2021): `DiffusionRecoveryLikelihood`
  loss (CD on the tethered recovery posterior between adjacent noise levels)
  and `drl_sample` progressive generation; reuses the noise-conditional nets
  and `geometric_sigmas`.
- `py.typed` marker (PEP 561) — the library ships its type annotations.

## 0.4.0 — 2026-07-28

- `reverse_ais_log_z`: RAISE-style reverse AIS from model samples — a
  stochastic *upper* bound on `log Z` that brackets the `ais_log_z` lower
  bound; shared annealing core, same schedules and diagnostics.
- `CategoricalGibbsWithGradients`: exact locally-informed MH sampler for
  one-hot categorical data `(B, *dims, K)`; verified against closed-form
  categorical and Potts distributions.
- `eval.frechet_distance`: pure-torch Fréchet distance (no scipy); pass an
  Inception embedding as `feature_fn` for standard FID.

## 0.3.0 — 2026-07-28

- Energy composition algebra: `SumEnergy` (product of experts), `MixtureEnergy`,
  `TemperedEnergy` — nestable, sampler/loss-compatible, module params registered.
- `datasets.two_moons(..., return_labels=True)` for supervised experiments.
- New example: `examples/train_jem.py` (classify + generate with one network).

## 0.2.0 — 2026-07-27

- AIS log-partition estimation: `ais_log_z` (HMC/MALA transitions, ESS/stderr
  diagnostics, linear/geometric/custom schedules) and `log_likelihood`
  (nats or bits/dim).
- JEM classifier-as-EBM: `ClassifierEnergy`, `JEMLoss`, class-conditional
  sampling via `.condition(y)`; `Trainer` supports supervised `(x, y)` batches.
- `GibbsWithGradients`: exact single-flip MH sampler for binary EBMs.
- NCSN stack: `ConditionalEnergyFn`, noise-conditional MLP/Conv energies,
  `MultiSigmaDenoisingScoreMatching`, `AnnealedLangevinDynamics`,
  `geometric_sigmas`.

## 0.1.0 — 2026-07-27

- Initial release: energy-function convention `p ∝ exp(-E)`; ULA/MALA/HMC
  samplers; CD/PCD with `ReplayBuffer`; denoising + sliced score matching; NCE;
  thin `Trainer` with EMA; 2D toy datasets; OOD eval; viz helpers.
