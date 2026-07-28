# Changelog

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
