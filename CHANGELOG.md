# Changelog

## 0.13.0 — 2026-08-15

Research-feature expansion — every addition validated against a closed-form or
known distribution, torch-only at runtime.

- **Samplers.** `ParallelTempering` (replica exchange over any base sampler —
  escapes multimodal traps single-chain Langevin can't), `UnderdampedLangevin`
  (SGHMC; momentum for better mixing, exact `exp(-E)` position marginal),
  `PreconditionedLangevin` (fixed diagonal preconditioner for ill-conditioned
  targets), and `GibbsSampler` (block Gibbs for models with tractable
  conditionals, e.g. the RBM).
- **Losses.** `ExactScoreMatching` (Hyvärinen 2005 — the O(D) exact objective
  the sliced/denoising variants approximate) and `EnergyDiscrepancy` (Schröder
  et al. 2023 — a sampler-free, score-free objective).
- **Energies.** `RBM` (Bernoulli RBM as a free energy, with block-Gibbs and an
  exact `log_z`), `ResNetEnergy` (the IGEBM / Improved-CD residual image net),
  and the closed-form test targets `FunnelEnergy` (Neal's funnel) and
  `GaussianMixtureEnergy`.
- **Eval.** `effective_sample_size` and `split_rhat` (MCMC convergence
  diagnostics), `precision_recall` (Kynkäänniemi et al. 2019 — fidelity vs
  coverage), and `inception_score` (formula over user-supplied classifier probs).
- **MCMC-free discrete losses.** `PseudoLikelihood` (Besag 1975) and
  `RatioMatching` (Hyvärinen 2007) — train any binary energy (RBM, Ising) from
  single-bit-flip energy differences, no sampler.
- **Goodness-of-fit eval.** `kernel_stein_discrepancy` (Liu et al. 2016 — a
  score-only GoF test / model selector, no partition function) and
  `classifier_two_sample_test` (Lopez-Paz & Oquab 2017).
- **`nets.BananaEnergy`** — a curved twisted-Gaussian MCMC stress test with an
  exact sampler (`exact_sample`) for ground-truthed benchmarks.
- **Deterministic & score-SDE samplers.** `SVGD` (Stein variational gradient
  descent — deterministic interacting-particle transport), and `ProbabilityFlowODE`
  + `PredictorCorrector` (Song et al. 2021 VE score-SDE samplers over a
  noise-conditional energy; the ODE is reproducible).
- **Examples.** `sampling_hard_targets.py`, `train_rbm.py`,
  `train_energy_discrepancy.py`, `train_ising_pseudolikelihood.py`,
  `goodness_of_fit.py`, `benchmark_samplers.py`, `deterministic_sampling.py`.

## 0.12.0 — 2026-08-14

- Renamed the distribution to **`ebmkit`** (`pip install ebmkit`). The import
  name is unchanged (`import ebm`).
- Docs simplified: the mkdocs / GitHub Pages site is removed in favor of a
  cleaner README plus plain-markdown guides in `docs/`.
- Added a Claude PR-review workflow (`.github/workflows/claude-code-review.yml`).

## 0.11.4 — 2026-08-14

- Packaging: make the two relative README links (`examples/`, `CONTRIBUTING.md`)
  absolute GitHub URLs so the PyPI project page — which renders the README as
  the long description — shows working links instead of broken relative ones.
  `twine check` passes on the built sdist + wheel.

## 0.11.3 — 2026-08-12

- Static type checking: CI now runs `mypy` over `src/ebm`, so the types the
  library advertises via its `py.typed` marker are actually verified. Fixed the
  type imprecisions this surfaced (registered-buffer annotations, the Trainer's
  supervised-call signature, AIS step-size adaptation, the annealed sampler's
  intentional override) — no runtime behavior change.

## 0.11.2 — 2026-08-12

- Performance (GPU/MPS): defer the per-step `last_accept_rate` host sync in
  `MALA` / `HMC` / the discrete samplers — the accept fraction is kept on-device
  and only materialized to a Python float when read, turning one GPU→CPU sync
  *per MCMC step* into one *per `sample()` call*. Measured ~5× faster MALA
  sampling and ~20% faster HMC-based AIS on Apple MPS with a cheap energy; no
  effect on CPU or on the already-sync-free `LangevinDynamics`. Also coalesced
  `ContrastiveDivergence`'s four metric syncs into one. No API change — returned
  values are identical.

## 0.11.1 — 2026-08-12

- Repository polish: `CONTRIBUTING.md` (dev workflow + the load-bearing
  conventions), `CITATION.cff`, and richer `[project.urls]` (Repository,
  Issues, Changelog) for the package's PyPI sidebar.

## 0.11.0 — 2026-08-12

- `datasets.cifar100`: CIFAR-100 as `(N, 3, 32, 32)` float32 in `[-1, 1]`,
  torchvision-free (same binary `tar.gz` family as `cifar10`; fine labels
  0-99). The standard natural-image out-of-distribution counterpart to
  CIFAR-10 for `eval.ood_auroc`.
- New example: `examples/train_cifar_ood.py` — energy-based OOD at color scale.
  A `ConvClassifier` trained on CIFAR-10 (cross-entropy-dominant JEM) flags
  CIFAR-100 as out-of-distribution by its `-logsumexp(logits)` energy, framed
  honestly as a hard near-OOD test (~61% accuracy, ~0.57 AUROC).

## 0.10.1 — 2026-08-09

- `eval.mmd` / `eval.frechet_distance`: raise a clear `ValueError` on degenerate
  inputs that previously returned `NaN` — fewer than 2 samples per set (the
  unbiased MMD denominator and single-sample covariance are undefined) and a
  vanishing RBF bandwidth from the median heuristic on near-identical samples.
- Test hardening: direct unit tests for `nets.IsingEnergy` / `nets.PottsEnergy`
  (closed-form lattice energies), `_GaussianFourierFeatures`, `utils.EMA` (exact
  lerp + state-dict roundtrip) and `frozen_params`, and an AIS ESS closed-form
  check (equals `n_chains` when the target matches the base).

## 0.10.0 — 2026-08-09

- `nets.PottsEnergy`: the K-color generalization of `IsingEnergy` — a 2D
  nearest-neighbor Potts energy `E(x) = -J Σ 1[c_i = c_j]` for one-hot
  `(B, H, W, K)` data, with an optional learnable coupling.
- New example: `examples/train_potts.py` — samples a 5-state Potts lattice with
  `CategoricalGibbsWithGradients` at three coupling strengths, showing the
  disorder-to-domains transition.
- New example: `examples/train_ncsn.py` — a standalone tour of the NCSN stack
  (`NoiseConditionalMLPEnergy` + `MultiSigmaDenoisingScoreMatching` +
  `AnnealedLangevinDynamics`), showing the learned score field and multi-mode
  coverage on eight Gaussians.

## 0.9.0 — 2026-08-08

- `nets.IsingEnergy`: the library's first discrete/lattice energy — a 2D
  nearest-neighbor Ising energy `E(x) = -J Σ s_i s_j` for binary `(B, H, W)`
  data, with an optional learnable coupling.
- New example: `examples/train_composition.py` — trains two stripe experts and
  composes them into a product of experts (intersection), a mixture (union),
  and a tempered energy, all sampled by the same `LangevinDynamics`.
- New example: `examples/train_ising.py` — samples a 2D Ising lattice with
  `GibbsWithGradients` at three coupling strengths, showing the ferromagnetic
  ordering (neighbor agreement rising with `J`).
- New example: `examples/checkpoint_resume.py` — demonstrates `Trainer.save` /
  `Trainer.load` with periodic checkpointing from the `callback` hook, then a
  fresh trainer resuming the run (energy, EMA, and replay buffer restored).

## 0.8.0 — 2026-08-08

- `datasets.cifar10`: CIFAR-10 as `(N, 3, 32, 32)` float32 in `[-1, 1]` with no
  torchvision dependency (downloads the binary distribution and parses the
  `.bin` batches directly — no pickle; cached in `~/.cache/ebm-pytorch`).
  Native 32x32 for the default `ConvEnergy` / `ConvClassifier`.
- `viz.show_images`: tile a batch of `(N, C, H, W)` images into one grid —
  grayscale (`C=1`) or RGB (`C=3`), with the `[-1, 1] -> [0, 1]` rescale built
  in. The MNIST examples now use it instead of hand-rolled subplot loops.
- `Trainer.save` / `Trainer.load` (+ `state_dict` / `load_state_dict`):
  checkpoint and resume a run — energy weights, optimizer moments, EMA, loss
  parameters, the PCD replay buffer, and the step counter / metric history.
- `eval.bits_per_dim`: the conventionally reported density-model score
  (lower is better) — the negation of `log_likelihood(..., dim=D)`, with `D`
  defaulting to the per-sample element count.

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
