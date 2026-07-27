# ebm-pytorch — design synthesis

Synthesized 2026-07-27 from three parallel research reports (see the other files
in this directory) plus the Nia-indexed sources listed in `nia.json`.

## Positioning

The deep-learning EBM ecosystem has exactly one maintained general-purpose
library — **torchebm** (broad scope, drifting into flow matching, API churn,
solo maintainer) — and a graveyard of frozen paper code (JEM, openai/ebm_code_release,
ebm-anatomy, igebm-pytorch). The field is paper-rich, tool-poor.

**This library's bet: narrow and reliable beats broad.** Ship the pieces every
EBM project rebuilds — samplers, losses, buffers, diagnostics — as small
composable objects with *tested, correct defaults*, and productize training
reliability (the ebm-anatomy insight) and evaluation, which no one ships.

Naming: PyPI `ebm` is taken (a Norwegian building-energy tool) and "EBM" search
mindshare belongs to interpretml's Explainable Boosting Machines. Distribution
is `ebm-pytorch`, import name `ebm`, README disambiguates.

## Core design decisions

1. **Energy is a callable, not a base class.** Anything `(B, *event_shape) -> (B,)`
   works everywhere; `EnergyModel` is optional sugar. (Pattern from
   torchdiffeq: pass the model *to* the algorithm.)
2. **Sign convention `p ∝ exp(-E)`** enforced and tested against closed-form
   Gaussians. Samplers descend the gradient; training pushes data energy down.
3. **Losses are `nn.Module`s** with one contract:
   `loss_fn(energy, x) -> LossOutput(loss, metrics, x_neg)`. Losses with
   parameters (NCE's learnable `log_z`) compose with optimizers uniformly.
   (Fixes torchebm's inconsistent loss return types.)
4. **Samplers own MCMC mechanics**: `sample()` freezes energy params
   (`frozen_params`), takes gradients w.r.t. samples only via
   `torch.autograd.grad(create_graph=False)`, returns detached samples.
   Score-matching losses invert this (`create_graph=True`).
5. **Both Langevin regimes one keyword apart**: `noise_scale=None` = correct
   `sqrt(2ε)` (convergent regime); decoupled cold noise + `grad_clip` + `clamp`
   = practitioner short-run regime (IGEBM: step 10.0, noise 0.005, clip 0.01,
   60 steps, buffer 10k @ 5% reinit, energy_reg 1.0, Adam(0, 0.999) 1e-4).
6. **ReplayBuffer semantics** (Du & Mordatch): per-sample 5% reinit, write-back
   to the same slots after sampling, detached CPU storage, capacity >> batch.
7. **Trainer is thin and optional** — reproducible in ~6 lines of vanilla
   PyTorch; exists for device/optimizer/EMA/history bundling. No Lightning-style
   inheritance lock-in.
8. **CD loss value is not a signal** — surface `energy_gap`, energy histograms.
   No BatchNorm in energy nets; SiLU; spectral norm as a flag.

## v0.1 surface (implemented)

- `energy.py` — `EnergyFn`, `score()`, `EnergyModel`
- `nets.py` — `MLPEnergy`, `ConvEnergy`
- `samplers/` — `LangevinDynamics` (ULA/SGLD), `MALA`, `HMC` (leapfrog)
- `losses/` — `ContrastiveDivergence` (CD/PCD + energy reg),
  `DenoisingScoreMatching`, `SlicedScoreMatching` (VR), `NoiseContrastiveEstimation`
- `buffer.py` — `ReplayBuffer`
- `training.py` — `Trainer`; `utils.py` — `EMA`, `frozen_params`
- `datasets.py` — 2D toys; `eval.py` — `ood_auroc`, batched `energies`;
  `viz.py` — contour/samples/energy-histogram plots
- Tests include distribution-correctness checks: samplers recover N(0, I) from
  `E = ||x||²/2`; DSM/SSM minimized at the true score scale; NCE optimum equals
  `2 log 2` when model == noise.

## Deliberately deferred (extension points exist)

AIS/RAISE log-Z (top roadmap item — nobody ships it), JEM-style conditional
EBMs (`logsumexp` energy from a classifier), discrete EBMs (Gibbs-with-gradients),
multi-sigma DSM + annealed Langevin, Improved-CD KL correction, energy
composition algebra (products of experts, tempering), diffusion/flow interop,
distributed training, Hub integration, CLI/Hydra.

The three affordances that keep these cheap to add later: energies are plain
callables (compose freely), samplers expose `step()` and pluggable init via the
buffer's `init_fn`, and losses are pure functions of batched energies.
