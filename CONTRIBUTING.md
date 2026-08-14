# Contributing to ebmkit

Thanks for your interest! This library aims to be **narrow and reliable** — a
small, well-tested set of energy-based-model primitives — so contributions that
deepen correctness, clarity, and evaluation are especially welcome.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/). No manual venv needed:

```bash
git clone https://github.com/davidkhjo/ebmkit
cd ebmkit
uv run pytest            # full test suite (CPU-only, seeded)
uv run ruff check .      # lint
uv run ruff format .     # format
```

`torch` is the only runtime dependency; `matplotlib` is an optional `[viz]`
extra used by the examples and `ebm.viz`.

## Conventions that matter

These are load-bearing — please keep them intact:

- **Sign convention:** `p(x) ∝ exp(-E(x))` everywhere. Low energy = high
  probability. Samplers *descend* the energy gradient; training pushes data
  energy *down*. Never flip this.
- **Energy functions** are any callable `(B, *event_shape) -> (B,)` — a plain
  function, a lambda, or an `nn.Module`.
- **Stop-gradients:** MCMC negatives are detached before the loss, and the
  energy network's parameters are frozen during sampling
  (`ebm.utils.frozen_params`). Score-matching losses instead keep the graph
  with `create_graph=True`.
- **Losses** are `nn.Module`s returning `LossOutput(loss, metrics, x_neg)`.
  Supervised losses (e.g. `JEMLoss`) set a class attribute `supervised = True`.
- **Networks:** no BatchNorm in energy networks (it breaks per-sample energies
  and MCMC); SiLU activations; spectral norm is an opt-in flag.

## Testing

Every core module has a test file under `tests/`. Prefer **closed-form or
distributional checks** over smoke tests — the suite verifies samplers against
known Gaussians, AIS against analytic `log Z`, lattice energies against
hand-computed values, and so on. Dataset tests are **offline** (they build
synthetic archives in `tmp_path`); never add a test that hits the network.

## Pull requests

1. Branch from `main`.
2. Keep the change focused; add or update tests.
3. Ensure `uv run pytest`, `uv run ruff check .`, and `uv run ruff format
   --check .` are green. If you touch docstrings or the public API, also run
   `uv run mkdocs build --strict`.
4. Open a PR against `main`. CI runs the suite on Python 3.10–3.13.

## Adding a loss or sampler

- A **loss** subclasses `nn.Module`, lives under `src/ebm/losses/`, and returns
  a `LossOutput`. Add a distributional test (e.g. recover a known parameter).
- A **sampler** subclasses `ebm.samplers.base.Sampler` and implements `step`;
  the base class handles the detach/freeze loop. Add a test that it targets a
  known distribution and, for MH samplers, exposes `last_accept_rate`.
