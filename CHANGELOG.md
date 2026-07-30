# Changelog

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
