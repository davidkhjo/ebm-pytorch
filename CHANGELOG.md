# Changelog

## 0.16.0 — 2026-08-21

The No-U-Turn Sampler and a latent-variable EBM — the last two items on the
roadmap. Both closed-form / distributionally validated and torch-only.

- **`NUTS`** — the No-U-Turn Sampler (Hoffman & Gelman 2014, multinomial variant):
  HMC that tunes its own trajectory length (doubling until a whole-span U-turn) and
  step size (dual-averaging warmup to `target_accept`), so it needs neither
  `leapfrog_steps` nor a step-size sweep. All chains are evolved in lockstep with a
  per-chain freeze mask, so each chain's draw is identical to independent
  single-chain NUTS. Exposes `last_tree_depth` and a `divergences` count. Validated:
  recovers a standard and a correlated Gaussian's covariance, tunes to 0.8
  acceptance, enters Neal's funnel, and terminates by U-turn (small tree depths,
  zero divergences on the Gaussian). Example `nuts_sampling.py`.
- **`LatentEBM`** — a latent-variable EBM: a joint `E(x, z) = E_prior(z) +
  E(x | z)` coupling a prior over a latent `z` (default standard normal) with a
  decoder energy. The data marginal is intractable, so `sample_joint` runs block
  Gibbs — alternating an MCMC update of `z` under its posterior `E(z | x)` with an
  update of `x` under `E(x | z)`; `posterior_energy` / `conditional_energy` expose
  those blocks as plain `EnergyFn`s. Validated on the linear-Gaussian conjugate
  case: block Gibbs recovers the exact marginal `N(0, WWᵀ + σ²I)` and the Gaussian
  posterior `z | x`. Example `latent_ebm.py`.

## 0.15.0 — 2026-08-20

The largest release yet: a variance-preserving diffusion track and trainable
exact-likelihood flows (CNF + spline), classifier-free guidance and calibration,
a self-tuning MALA, and ensemble-uncertainty / MCMC-diagnostic tooling — plus an
internal restructure. Every feature is closed-form or distributionally validated
and torch-only.

- **Diagnostics & ensemble uncertainty.** `viz.autocorrelation_plot` /
  `rank_plot` / `trace_plot` and a numeric `eval.autocorrelation` for judging MCMC
  mixing; `EnsembleEnergy` (a deep-ensemble *mean* energy — the geometric-mean
  density, distinct from `MixtureEnergy`'s logsumexp) with `member_energies` /
  `disagreement`, plus `eval.ensemble_disagreement` as an epistemic OOD score.
  Validated: the ACF estimator matches an AR(1)'s `ρ^t`; the ensemble reproduces
  the closed-form combined Gaussian; disagreement separates in-distribution from
  OOD at AUROC ≈ 1. Example `ensemble_ood.py`.
- **`nets.NeuralSplineCouplingFlow`** — a rational-quadratic neural spline flow
  (Durkan et al. 2019): monotonic-spline coupling layers dropped into the
  affine-flow scaffold, strictly more expressive per layer, so it fits sharp
  multi-modal 2D densities at a lower NLL than the affine flow for the same
  depth. Same self-normalized contract (exact `log_prob`, one-pass sampling,
  `forward = -log_prob`). Validated: exact invertibility incl. the linear tails
  and analytic `log|det|` vs the autograd Jacobian (to 1e-8 in double precision);
  fits a Gaussian and two-moons. Example `train_spline_flow.py`.
- **Self-tuning sampler.** `AdaptiveMALA` — a MALA that tunes its own
  `step_size` by Nesterov dual averaging (Hoffman & Gelman 2014) to the
  MALA-optimal 0.574 acceptance during a warmup, then freezes and samples
  unbiasedly. `precondition=True` estimates a diagonal metric from a first warmup
  window and re-tunes under it, so ill-conditioned targets mix at a far larger
  step (`x ← x − εM∇E + √(2εM)ξ`, still exact). Validated: acceptance → 0.574 and
  covariance recovered on a correlated Gaussian; the metric recovers a 100:1
  condition number. Example `adaptive_mala.py`.
- **Classifier-free guidance & calibration.** `GuidedEnergy` (+
  `ClassifierEnergy.guide`) — `Ẽ_w(x|y) = (1+w)E(x|y) − w E(x)` to sharpen class
  selection; and `eval.expected_calibration_error` / `reliability_curve` /
  `temperature_scale` for trustworthy JEM classifiers. Example `jem_guidance.py`.

- **Housekeeping (no API change).** Deduplicated shared helpers into a private
  `ebm/_functional.py`, and split the two largest modules into packages
  (`nets/`, `eval/`) mirroring `losses/` and `samplers/`. Added input-validation
  guards, a coverage gate, and viz tests.
- **Variance-preserving diffusion (DDPM).** `VPSchedule` (linear/cosine),
  `VPDenoisingScoreMatching` (ε-prediction loss where the ε-net is derived from
  the energy, `ε_θ = √(1-ᾱ_t)·∇E`), and `DDPMAncestralSampler` — the
  variance-preserving counterpart to the VE/NCSN stack, reusing the
  noise-conditional energies. Validated: the ancestral sampler recovers a known
  Gaussian's variance; VP-DSM trains a model whose samples match it. Example
  `train_diffusion.py`.
- **`nets.ContinuousNormalizingFlow`** (FFJORD) — a *trainable* neural-ODE flow,
  the generalization of `eval.pf_ode_log_likelihood`: a velocity field defines an
  ODE from data to a Gaussian base and the exact log-density follows the
  instantaneous change of variables (Hutchinson trace), all torch-only (fixed-step
  RK4, direct backprop). Trains by maximum likelihood, samples in one reverse pass,
  and doubles as a self-normalized energy. Validated against a linear-field
  closed form and exact rotation-invariance. Example `train_cnf.py`.

## 0.14.0 — 2026-08-17

Exact likelihoods for score-based models, torch-only and closed-form validated.

- **`eval.pf_ode_log_likelihood`** — exact per-sample log-density of any
  noise-conditional / diffusion energy by integrating the probability-flow ODE
  with the FFJORD instantaneous change of variables (Hutchinson trace), a
  partition-function-free alternative to the AIS `log_likelihood`. Validated
  against the analytic Gaussian log-density.
- **`nets.AffineCouplingFlow`** — a RealNVP normalizing flow with an exact
  `log_prob` / `sample`, exposed as a self-normalized energy
  (`forward(x) = -log_prob(x)`, `log Z = 0`). Validated by its exact log-det
  identity and a Gaussian fit.
- **Examples.** `exact_likelihood_ode.py` (PF-ODE bits/dim, OOD separation, AIS
  cross-check), `train_flow.py` (RealNVP on two-moons).

## 0.13.0 — 2026-08-17

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
- **MCMC-free discrete losses.** `PseudoLikelihood` (Besag 1975),
  `RatioMatching` (Hyvärinen 2007) — train any binary energy (RBM, Ising) from
  single-bit-flip energy differences — and `ConcreteScoreMatching` (Meng et al.
  2022), the categorical analogue of score matching for one-hot data.
- **Goodness-of-fit & information eval.** `kernel_stein_discrepancy` (Liu et al.
  2016 — a score-only GoF test / model selector, no partition function),
  `classifier_two_sample_test` (Lopez-Paz & Oquab 2017), `fisher_divergence`
  (score-space distance between two EBMs), and `mutual_information` (MINE,
  Belghazi et al. 2018).
- **`nets.BananaEnergy`** — a curved twisted-Gaussian MCMC stress test with an
  exact sampler (`exact_sample`) for ground-truthed benchmarks.
- **Deterministic & score-SDE samplers.** `SVGD` (Stein variational gradient
  descent — deterministic interacting-particle transport), and `ProbabilityFlowODE`
  + `PredictorCorrector` (Song et al. 2021 VE score-SDE samplers over a
  noise-conditional energy; the ODE is reproducible).
- **`TemperedTransitions`** — Neal's (1996) single-chain annealed barrier
  crossing, a replica-free complement to `ParallelTempering`.
- **Examples.** `sampling_hard_targets.py`, `train_rbm.py`,
  `train_energy_discrepancy.py`, `train_ising_pseudolikelihood.py`,
  `goodness_of_fit.py`, `benchmark_samplers.py`, `deterministic_sampling.py`,
  `train_potts_concrete.py`, `mine_mutual_information.py`.

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
