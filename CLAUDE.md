# ebmkit

Python library (`import ebm`, PyPI name `ebmkit`) for training and using
energy-based models in PyTorch. src/ layout, hatchling build, torch is the only
runtime dependency.

## Commands

- `uv run pytest` — full test suite (CPU-only, includes statistical sampler tests seeded in `tests/conftest.py`)
- `uv run ruff check .` and `uv run ruff format .`

## Conventions

- `p(x) ∝ exp(-E(x))`: low energy = high probability. Samplers subtract the
  energy gradient. Never flip this sign convention.
- Energy functions are callables `(B, *event_shape) -> (B,)`.
- MCMC negatives must be detached before the loss; the energy net's params are
  frozen during sampling (`ebm.utils.frozen_params`). Score-matching losses use
  `create_graph=True` instead.
- Losses are `nn.Module`s returning `LossOutput(loss, metrics, x_neg)`.
  Supervised losses (e.g. `JEMLoss`) set a class attribute `supervised = True`
  and are called as `loss_fn(energy, x, y)` by the Trainer.
- Noise-conditional energies follow `ConditionalEnergyFn`: `(x, sigma) -> (B,)`
  with sigma `(B,)`; adapt to plain-energy APIs via closures.
- No BatchNorm in energy networks; SiLU activations; spectral norm is a flag.

## Research context

`research/` holds the reports the design is based on (existing-library survey,
training-methods spec, API design). Findings are also stashed in Nia (contexts +
indexed repos/papers — see `nia.json`; query with `nia search query "..."`).
