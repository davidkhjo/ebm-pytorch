# 03 — API Design: Patterns from Best-Loved Research Libraries, and the Recommended Design for `ebm`

*Research date: 2026-07-27. Sources: each library's README/docs/source as of this date (links inline).*

---

## Part 1 — What the best small/mid research libraries actually do

### 1.1 normflows (VincentStimper/normalizing-flows)

**The pattern: one thin composition class over a list of layer modules, plus a base distribution.**
The whole library is "base + flows + optional target," and the model class is little more than
a container that knows how to run the list forward/backward and compute the two KLs.

```python
import normflows as nf

base = nf.distributions.base.DiagGaussian(2)

flows = []
for i in range(32):
    param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    flows.append(nf.flows.Permute(2, mode="swap"))

target = nf.distributions.target.TwoMoons()
model = nf.NormalizingFlow(base, flows, target)

loss = model.forward_kld(x)  # maximum likelihood
loss = model.reverse_kld(num_samples=512)  # variational fit to target
loss.backward()
```

Package layout (from the repo):

```
normflows/
  __init__.py   core.py   core_test.py   transforms.py   transforms_test.py
  distributions/   flows/   nets/   sampling/   utils/
```

**Worth copying**
- *The user builds a plain Python list of `nn.Module`s* — no registry, no config files, no
  string dispatch. Composition is literal.
- Loss computation lives as *methods that return a scalar tensor*; the user owns the
  optimizer and the loop. The README training loop is vanilla PyTorch.
- `nets.MLP(...)` convenience: researchers on toy problems shouldn't hand-write MLPs.
- Built-in toy *target distributions* (`TwoMoons`) so the quickstart is self-contained.
- Tests co-located next to the modules they test (`core.py` / `core_test.py`) — low
  ceremony, easy for contributors.

**Worth avoiding**
- Loss-as-model-method couples the model class to every training objective; it scales
  poorly once objectives multiply (normflows gets away with two KLs; EBMs have CD, PCD,
  score matching variants, NCE, ...). Losses should be their own objects/functions.

### 1.2 torchdiffeq / torchsde

**The pattern: one verb, functional, with the "model" passed in as an argument.**
torchdiffeq's entire public API is three functions; everything else hides in a private
`_impl/` package (`torchdiffeq/{__init__.py, _impl/}` — that's the whole tree):

```python
from torchdiffeq import odeint

odeint(func, y0, t)  # func: f(t, x); y0: Tensor; t: 1-D Tensor
from torchdiffeq import odeint_adjoint as odeint  # drop-in O(1)-memory variant
```

with `rtol`, `atol`, `method` ("dopri5" default, euler, rk4, ...), `options` as kwargs.
Documented gotcha: "`func` must be a `nn.Module` when using the adjoint method."

torchsde is the same shape, but the dynamics object is a small `nn.Module` with a
documented *method-name contract* plus class-attribute declarations:

```python
class SDE(torch.nn.Module):
    noise_type = "general"
    sde_type = "ito"

    def f(self, t, y):
        return self.mu(y)  # drift

    def g(self, t, y):
        return self.sigma(y).view(b, s, m)  # diffusion


ys = torchsde.sdeint(sde, y0, ts)
```

**Worth copying**
- **The thing being integrated/sampled is an argument, not a base class.** `odeint(func, ...)`
  works on any callable. For an EBM library: `sampler.sample(energy, x0)` should accept any
  `Callable[[Tensor], Tensor]`, so the same sampler works on a raw function, an `nn.Module`,
  a tempered/annealed closure, or a lambda over two models.
- Solver/sampler selection by string with a good default, tuned by a few kwargs.
- Drop-in aliasing (`odeint_adjoint as odeint`) as the upgrade path — API-compatible
  variants, not flags that change semantics.
- Private `_impl` keeps the import surface honest: if it's not in `__init__.py`, it's not API.

**Worth avoiding**
- Pure-functional-only surface makes stateful things awkward (adjoint needs the Module
  anyway). EBM samplers have knobs (step size, noise, clipping) and sometimes state
  (adaptive step size) — lightweight *classes with a functional call* fit better than bare
  functions with 10 kwargs.

### 1.3 huggingface/diffusers

**The pattern: three-layer split (models / schedulers / pipelines) + config capture.**
Stated philosophy, verbatim from the README: *"usability over performance," "simple over
easy," "customizability over abstractions."* The pipeline is a one-liner; but the README
*also* shows assembling the denoising loop yourself from the parts:

```python
from diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)

input = torch.randn((1, 3, size, size), device="cuda")
for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
        input = scheduler.step(noisy_residual, t, input).prev_sample
```

The pipeline (`DiffusionPipeline.from_pretrained(...)`) is *just orchestration* of parts
that remain independently usable and swappable (any scheduler works with any compatible
model). Config machinery (`configuration_utils.py`): `ConfigMixin` + an
`@register_to_config` decorator on `__init__` captures all init args into a frozen dict at
construction time; `save_config()` writes JSON (with class name + version metadata);
`from_config()` reconstructs by filtering the dict against the `__init__` signature.
Docstring: *"Base class for all configuration classes. All configuration parameters are
stored under `self.config`."*

**Worth copying**
- **Sampler ("scheduler") ⟂ model separation**, with `sampler.step(...)` exposed so the
  advanced user can write their own loop, and a convenience layer on top for everyone else.
  Returning a small result object (`.prev_sample`, `.sample`) instead of bare tensors
  leaves room to add fields without breaking callers.
- **Init-args-as-config**: capture `__init__` kwargs automatically → `save/load` and `repr`
  for free, no parallel dataclass to keep in sync. A 40-line version of
  `register_to_config` (no Hub, no compat shims) is enough for a small library.
- The README literally teaches the internals; the abstraction is transparent, not magic.

**Worth avoiding**
- Sheer size and Hub coupling; `from_pretrained` net-download semantics are overkill for a
  research library. Take the JSON-config idea, point it at a local directory:
  `save(dir)` / `load(dir)` writing `config.json` + `state_dict.pt`.
- Mixin stacking depth (ConfigMixin + ModelMixin + Loaders...) — one small mixin max.

### 1.4 gpytorch / pyro

**The pattern: "you subclass a Module and write `forward`; the library gives you parts and
a loss object."** GPyTorch's canonical example:

```python
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for i in range(training_iter):
    optimizer.zero_grad()
    loss = -mll(model(train_x), train_y)
    loss.backward()
    optimizer.step()
```

Pyro's stated principles (verbatim): **Universal**; **Scalable**; **Minimal** — *"implemented
with a small core of powerful, composable abstractions"*; **Flexible** — *"automation when
you want it, control when you need it."*

**Worth copying**
- The *loss is a first-class callable object* (`ExactMarginalLogLikelihood(likelihood, model)`)
  configured once, then called in a vanilla PyTorch loop the user writes and owns. This is
  exactly the right shape for CD/score-matching losses.
- Domain vocabulary as module namespaces (`gpytorch.kernels`, `.means`, `.mlls`,
  `.likelihoods`) — discoverable via autocomplete, reads like the math.
- "Automation when you want it, control when you need it" is the right one-line design goal.

**Worth avoiding**
- Pyro's effect-handler/poutine machinery is the opposite of transparent — powerful but
  famously hard to debug. An EBM library needs zero of that.
- GPyTorch's `model.train()/model.eval()` changing *mathematical* behavior (not just
  dropout/BN) is a recurring footgun; don't overload PyTorch mode switches with semantics.

### 1.5 Lightning

```python
class LitAutoEncoder(L.LightningModule):
    def training_step(self, batch, batch_idx):
        ...
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


trainer = L.Trainer()
trainer.fit(autoencoder, DataLoader(train), DataLoader(val))
```

**Take**: the *ergonomics* — `Trainer(...).fit(...)` as a two-line happy path; sane logging;
device placement handled for you. Also note **Fabric**'s existence: Lightning itself
concluded many users want *"expert-level control over [the] PyTorch training loop"* with only
device/precision handled — i.e., the inversion-of-control Trainer is not universally loved.

**Avoid**: framework lock-in. `LightningModule` inheritance contaminates model code —
models can't be used outside Lightning without carrying the base class; hook lifecycle
(20+ overridable methods) is a hidden control flow; version churn is a research-repo tax.
**Rule for us: the Trainer takes plain objects; nothing in the library inherits from or
requires the Trainer.** Every Trainer feature must be reproducible in ~10 explicit lines.

### 1.6 Prior art collision check: `torchebm` and the name "EBM"

- **`torchebm`** (PyPI, v0.7.5, July 2026, alpha) already exists and has expanded far beyond
  EBMs: six primitives (Energy/Field, Interpolant, Coupling, Objective, Sampler, Integrator)
  covering diffusion, flow matching, and Schrödinger bridges, with `LangevinDynamics`, `HMC`,
  `ContrastiveDivergence`, score-matching losses, integrators, OT couplings.
  *Differentiation strategy: stay narrow and boring — classic energy-based modeling done
  extremely well, tiny API, no simulation-free/flow scope.* Its 6-way primitive
  decomposition is also a cautionary tale: `EquilibriumMatchingLoss(model=field,
  interpolant="linear", energy_type="dot")` requires learning a private ontology before
  training a 2-layer MLP on two moons.
- **"EBM" name collision**: interpretml's **Explainable Boosting Machine** dominates search
  and pip-mindshare for "EBM"; PyPI **`ebm`** is taken (a Norwegian building-stock energy
  forecasting tool, v1.1.0). Distribution and import names can differ, so `import ebm` is
  achievable under a different PyPI name — but shadow-collision risk with the existing `ebm`
  dist (both would install an `ebm/` package) argues for care. See §2.0.

### 1.7 Cross-library pattern summary

| Pattern | Seen in | Adopt? |
|---|---|---|
| Model passed as argument to solver/sampler | torchdiffeq, torchsde | Yes — core decision |
| Loss as configured callable object, user owns loop | gpytorch (mll), normflows | Yes |
| Parts ⟂ orchestration; convenience layer is thin & optional | diffusers pipelines, Lightning | Yes (one small `Trainer`) |
| Init-args captured to JSON config; local save/load | diffusers `ConfigMixin` | Yes, ~40-line version |
| String selection with strong defaults (`method="dopri5"`) | torchdiffeq | Yes, for nets/init schemes |
| Private `_impl`/explicit `__init__` = the API contract | torchdiffeq | Yes |
| Toy targets/datasets in-library for self-contained quickstart | normflows, torchebm | Yes |
| Result namedtuple/dataclass instead of bare tensor returns | diffusers `.prev_sample` | Yes |
| Method-name contract on `nn.Module` (`f`/`g`) | torchsde | Partially (`forward` → energy; that's all) |
| Deep mixin stacks, Hub coupling | diffusers | No |
| Inheritance-based trainer coupling, hook lifecycle | Lightning | No |
| Loss as model method | normflows | No |
| Large private ontology of primitives | torchebm | No |

---

## Part 2 — Recommended design

### 2.0 Naming

- **Import name: `ebm`.** Short, exactly what researchers type in papers.
- **PyPI distribution name: `ebm-pytorch`** (verified free as of 2026-07-27; so are
  `pytorch-ebm`, `energy-models`, `ebmlib`). `ebm` itself is **taken** on PyPI.
- Risks to document in the README, first paragraph: (a) "EBM" ≠ interpretml's Explainable
  Boosting Machine — say so explicitly for SEO and human disambiguation ("energy-based
  models, not Explainable Boosting Machines"); (b) installing `ebm-pytorch` alongside the
  unrelated `ebm` dist would clash on the `ebm/` import package — unlikely audience overlap
  (Norwegian building-stock forecasting), but if it ever bites, the fallback is renaming the
  import package to `ebmx` or `nrgpt`-style in a 0.x release. Decide before 1.0.
- Do **not** ship a `torchebm`-adjacent name; that namespace is occupied and active.

### 2.1 Package / module layout

`src/` layout (import-from-source bugs caught at test time; the installed package is the
only importable one), tests co-located out-of-package in `tests/` (repo-level `pytest`
conventions beat normflows' in-package tests for wheel size and tooling).

```
ebm-pytorch/
├── pyproject.toml
├── README.md
├── LICENSE                  # MIT
├── src/ebm/
│   ├── __init__.py          # THE public API: ~20 re-exported names + __version__
│   ├── py.typed
│   ├── energy.py            # EnergyModel base, wrap(), score(), Tempered/Interpolated helpers
│   ├── nets.py              # MLPEnergy, ConvEnergy — batteries for common energy nets
│   ├── samplers/
│   │   ├── __init__.py
│   │   ├── base.py          # Sampler ABC, SampleResult
│   │   ├── langevin.py      # LangevinDynamics (ULA), MALA
│   │   └── hmc.py           # HMC
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── base.py          # LossOutput
│   │   ├── cd.py            # ContrastiveDivergence (covers CD-k and PCD via buffer arg)
│   │   └── score_matching.py# DenoisingScoreMatching, SlicedScoreMatching
│   ├── buffer.py            # ReplayBuffer
│   ├── training.py          # Trainer (thin), TrainState, callbacks (log/ckpt only)
│   ├── datasets.py          # two_moons, checkerboard, rings, gaussian_mixture → Tensor
│   ├── viz.py               # plot_energy, plot_samples (guarded matplotlib import)
│   └── _config.py           # ~40-line register_to_config + save/load (diffusers-lite)
├── tests/                   # mirrors src/ebm; pytest
├── examples/                # runnable scripts, each < 100 lines: 01_two_moons.py, 02_mnist_pcd.py, ...
└── docs/                    # mkdocs-material + mkdocstrings
```

Rationale: namespaces mirror the vocabulary of the field (`samplers`, `losses`), like
`gpytorch.kernels`/`.mlls`. Flat single files (`buffer.py`, `datasets.py`, `nets.py`)
until a second file is genuinely needed — subpackages only where a family exists
(samplers, losses). Everything importable from top level: `ebm.LangevinDynamics`,
`ebm.ContrastiveDivergence` — `__init__.py` is the API contract (torchdiffeq lesson).

### 2.2 Core abstractions — exact signatures

**Energy = any callable `(B, *shape) → (B,)`. No mandatory base class.** (torchdiffeq's
lesson.) The base class is optional sugar:

```python
# energy.py
EnergyFn = Callable[[Tensor], Tensor]  # (B, *event_shape) -> (B,)


class EnergyModel(nn.Module):
    """Optional convenience base. Subclasses implement forward(x) -> (B,) energies."""

    def forward(self, x: Tensor) -> Tensor: ...  # abstract
    def score(self, x: Tensor) -> Tensor:  # -∇x E(x), create_graph-aware
        ...
    def save(self, path: str | Path) -> None: ...  # config.json + state_dict.pt
    @classmethod
    def load(cls, path: str | Path, map_location=...) -> "EnergyModel": ...


def score(energy: EnergyFn, x: Tensor, create_graph: bool = False) -> Tensor:
    """Free function: -grad_x energy(x).sum(); works on any callable."""
```

**Samplers: small classes; energy is an argument to `sample`, never a constructor arg**
(so one sampler instance serves training negatives, eval, tempered variants):

```python
# samplers/base.py
@dataclass
class SampleResult:
    samples: Tensor  # final state, no grad
    trajectory: Tensor | None = None  # (steps+1, B, *shape) if requested
    acceptance_rate: Tensor | None = None  # MALA/HMC


class Sampler(ABC):
    @abstractmethod
    def step(self, energy: EnergyFn, x: Tensor, t: int) -> Tensor: ...
    def sample(
        self,
        energy: EnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,  # None -> self.default steps
        return_trajectory: bool = False,
    ) -> Tensor | SampleResult: ...


# samplers/langevin.py
class LangevinDynamics(Sampler):
    def __init__(
        self,
        step_size: float = 1e-2,
        steps: int = 100,
        noise_scale: float = 1.0,  # 0.0 -> pure gradient descent
        grad_clip: float | None = None,
        clamp: tuple[float, float] | None = None,
    ): ...
```

`step()` public (diffusers' `scheduler.step`) → users write custom loops (annealed step
sizes, intermediate visualization) without forking. `sample()` returns a bare detached
Tensor by default; `SampleResult` only when trajectory/diagnostics requested — the common
case stays one-liner clean.

**Losses: configured callables, gpytorch-mll style, returning a rich-but-simple output:**

```python
# losses/base.py
@dataclass
class LossOutput:
    loss: Tensor  # scalar; call .loss.backward()
    metrics: dict[str, float]  # {"energy/pos": ..., "energy/neg": ...}
    x_neg: Tensor | None = None  # negatives (detached), for viz/buffer inspection


# losses/cd.py
class ContrastiveDivergence:
    """CD-k; pass buffer=ReplayBuffer(...) for persistent CD (JEM-style)."""

    def __init__(
        self,
        sampler: Sampler,
        *,
        steps: int | None = None,
        buffer: ReplayBuffer | None = None,
        init: Callable[[int], Tensor] | None = None,  # fresh-negative init
        energy_reg: float = 0.0,
    ):  # alpha * (E+² + E-²)
        ...
    def __call__(self, energy: EnergyFn, x: Tensor) -> LossOutput: ...


# losses/score_matching.py  — no sampler needed
class DenoisingScoreMatching:
    def __init__(self, sigma: float | Sequence[float] = 0.1): ...
    def __call__(self, energy: EnergyFn, x: Tensor) -> LossOutput: ...


class SlicedScoreMatching:
    def __init__(self, n_projections: int = 1, variance_reduction: bool = True): ...
    def __call__(self, energy: EnergyFn, x: Tensor) -> LossOutput: ...
```

All losses share one calling convention `loss_fn(energy, x) -> LossOutput`, so they are
interchangeable in any loop and in the Trainer. Classes, not functions, because they carry
real configuration and state (sampler, buffer) — but they're plain callables, so
functional-minded users see no difference at the call site.

**Replay buffer — the one piece every EBM repo hand-rolls badly:**

```python
class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        shape: tuple[int, ...],
        *,
        reinit_prob: float = 0.05,  # fraction resampled from init_fn
        init_fn: Callable[[int], Tensor] | None = None,  # default: U(-1,1)/N(0,1)
        device: torch.device | str = "cpu",
    ): ...
    def sample(self, n: int) -> Tensor: ...
    def push(self, x: Tensor) -> None: ...
    def state_dict(self) -> dict: ...  # checkpointable
    def load_state_dict(self, state: dict) -> None: ...
    def __len__(self) -> int: ...
```

**Trainer: thin, optional, takes plain objects, no inheritance (anti-Lightning):**

```python
class Trainer:
    def __init__(
        self,
        energy: nn.Module,
        loss: Callable[[EnergyFn, Tensor], LossOutput],
        *,
        optimizer: torch.optim.Optimizer | None = None,  # default: Adam(1e-3)
        device: str | torch.device | None = None,  # default: auto
        grad_clip: float | None = None,
        log_every: int = 100,
        callbacks: Sequence[Callable[[TrainState], None]] = (),
    ): ...
    def fit(
        self,
        data: Tensor | DataLoader | Iterable[Tensor],
        *,
        steps: int = 10_000,
        batch_size: int = 256,
    ) -> TrainState: ...
```

Contract: *everything `Trainer.fit` does is expressible in the ~8-line manual loop shown in
the README right next to it.* No hooks beyond plain-callable callbacks receiving a
`TrainState` (step, loss_output, energy, optimizer). No `Trainer` types appear anywhere
else in the library.

**Config capture (`_config.py`)**: a diffusers-lite `@register_to_config` on `__init__` of
`EnergyModel` subclasses, samplers, and losses → `repr()` shows hyperparameters,
`.save(dir)`/`.load(dir)` round-trips `config.json` + weights locally. No Hub, no compat
tables, no FrozenDict — a plain dict and ~40 lines.

### 2.3 README quickstart (the block users should see first)

```python
import torch, ebm

energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))  # any nn.Module (B,2)->(B,) works
sampler = ebm.LangevinDynamics(step_size=1e-2, steps=100)
buffer = ebm.ReplayBuffer(capacity=8192, shape=(2,))
loss_fn = ebm.ContrastiveDivergence(sampler, buffer=buffer)  # persistent CD

data = ebm.datasets.two_moons(8192)
opt = torch.optim.Adam(energy.parameters(), lr=1e-3)
for step in range(3000):
    x = data[torch.randint(len(data), (256,))]
    out = loss_fn(energy, x)
    opt.zero_grad()
    out.loss.backward()
    opt.step()

samples = sampler.sample(energy, torch.randn(2000, 2), steps=500)
ebm.viz.plot_energy(energy, samples=samples)  # pip install ebm-pytorch[viz]
```

14 lines; shows all five nouns (net, sampler, buffer, loss, data), a fully-owned vanilla
loop, and reuse of the same sampler for generation. Immediately below it in the README, the
two-line equivalent: `ebm.Trainer(energy, loss_fn).fit(data, steps=3000)` — automation when
you want it, control when you need it.

### 2.4 Packaging, tooling, CI, docs

**pyproject.toml** (single source of truth; no setup.py/setup.cfg):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
# hatchling over uv_build: boring, universal, no ecosystem bet; uv still works as the
# frontend (uv build / uv pip install -e .). Revisit uv_build post-1.0 if desired.

[project]
name = "ebm-pytorch"
requires-python = ">=3.10"
dependencies = ["torch>=2.0"]            # the ONLY hard runtime dep (numpy comes with torch)
dynamic = ["version"]                     # hatch-vcs from git tags

[project.optional-dependencies]
viz  = ["matplotlib>=3.7"]
dev  = ["pytest>=8", "pytest-cov", "ruff>=0.6", "mypy>=1.10", "pre-commit"]
docs = ["mkdocs-material", "mkdocstrings[python]"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "NPY", "RUF"]  # + "D" for src/ only, google style
[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --tb=short"

[tool.mypy]
python_version = "3.10"
strict = false                            # pragmatic: full annotations, lenient checking
disallow_untyped_defs = true              # ...but every public def is annotated
```

- **Typing policy**: every public function/class fully annotated (PEP 604 unions, builtin
  generics — 3.10 floor makes this clean); `py.typed` marker shipped; mypy in CI on `src/`
  only, not tests. No `TypeVar` gymnastics — `Tensor` in/out is the honest research-library
  contract. `EnergyFn` alias documented as the central protocol.
- **Ruff is both linter and formatter** (`ruff format`); pre-commit with ruff + ruff-format
  only. No black, no isort (ruff's `I` covers it).
- **Tests**: pytest; every sampler tested against analytically-known distributions (sample
  a 2D Gaussian energy, check mean/cov); losses tested for gradient-flow and sign
  correctness; seeded, CPU, < 60 s total. One `slow` marker for a full two-moons
  convergence test run on schedule, not per-push.
- **CI (GitHub Actions)**: (1) `lint` — ruff check + format --check + mypy; (2) `test` —
  matrix {py3.10, py3.12, py3.13} × {torch 2.0 (floor pin), torch latest}, CPU wheels via
  the cpu index for speed; (3) `build` — `python -m build` + `twine check`; (4) `docs` —
  mkdocs build --strict; release job on tag → PyPI via trusted publishing (OIDC, no tokens).
- **Docs**: mkdocs-material + mkdocstrings (API pages generated from those Google-style
  docstrings), and the docs are mostly *examples*: each `examples/*.py` doubles as a docs
  page. A short "Math ↔ API" page mapping E(x), ∇E, CD-k, PCD to the class names — the page
  researchers actually want.

### 2.5 What NOT to build in v0.1

1. **No diffusion / flow matching / interpolants / OT couplings** — that's `torchebm`'s
   turf and the fastest route to an ontology users must learn. Narrow is the moat.
2. **No distributed / multi-GPU / mixed-precision machinery** — users bring `accelerate`
   or Fabric themselves; the Trainer stays single-device. Nothing in the API blocks it later.
3. **No config/CLI framework** (no Hydra, no YAML experiment configs, no entry points).
   Python files are the config format.
4. **No Hub / `from_pretrained` over network** — local `save/load` only.
5. **No callbacks ecosystem, loggers, or W&B/TensorBoard integrations** — `LossOutput.metrics`
   is a plain dict; users log it with whatever they use. One example script shows how.
6. **No exotic samplers** (no RMHMC, no parallel tempering, no NUTS) and **no NCE/flow-
   contrastive losses** in 0.1. ULA/MALA/HMC + CD/PCD/DSM/SSM covers ~90% of papers;
   the ABCs make the rest contributable.
7. **No custom CUDA/triton kernels, no torch.compile guarantees** (don't prevent it either).
8. **No image-scale training recipes** (JEM, IGEBM reproductions) in the package — one
   `examples/02_mnist_pcd.py` proves scale; reproductions live in a separate repo later.
9. **No conditional/joint EBMs API** (`E(x, y)`) — solvable today by closure
   (`lambda x: energy(x, y)`) precisely because samplers take callables; a real API can wait
   for evidence.
10. **No dataset registry beyond the 2D toys** — torchvision exists.

The 0.1 promise, in one sentence: *train and sample a classic EBM in 14 transparent lines,
with parts you can swap, read, and outgrow without leaving the library.*
