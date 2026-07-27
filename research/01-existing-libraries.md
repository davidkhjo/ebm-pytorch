# Survey of the Existing EBM Library / Codebase Ecosystem

*Research note 01 for a new open-source Python EBM library. Compiled 2026-07-27.*

Scope: deep-learning energy-based models — unnormalized density models `E_θ(x)`, trained via
contrastive divergence (CD), score matching, or other MCMC-in-the-loop / simulation-free
objectives, and sampled with Langevin/HMC/etc.

**Name-collision warning:** "EBM" on PyPI and in most search results means **Explainable
Boosting Machine** — the glassbox GAM from Microsoft's `interpret` (interpretml) package
(`interpret.glassbox.ExplainableBoostingClassifier`). That project completely dominates the
acronym in tabular-ML circles. Any new library needs a name that does not fight this collision
(`torchebm` sidesteps it with the framework prefix; a bare `ebm` / `pyebm` package name would
be a discoverability and SEO mistake).

---

## 1. Ecosystem at a glance

| Project | Type | Framework | Stars | Last push | Status |
|---|---|---|---|---|---|
| **soran-ghaderi/torchebm** | general EBM/generative library | PyTorch | 106 | 2026-07-21 | **active**, PyPI v0.7.5, 18 releases |
| wgrathwohl/JEM | paper code (JEM classifier-EBM) | PyTorch | 437 | 2022-09 | archived-in-practice |
| openai/ebm_code_release | paper code (Du & Mordatch 2019) | **TensorFlow 1** | 364 | 2023-04 | dead |
| point0bar1/ebm-anatomy | paper code (Anatomy of MCMC EBM) | PyTorch | 39 | 2024-07 | frozen, still exemplary |
| rosinality/igebm-pytorch | unofficial IGEBM reimpl. | PyTorch | 68 | 2019-03 | dead |
| yilundu/improved_contrastive_divergence | paper code (ICML'21) | PyTorch | 70 | 2022-04 | frozen |
| wgrathwohl/VERA | paper code (EBMs w/o MCMC) | PyTorch | 64 | 2024-02 | frozen |
| sndnyang/JEMPP | paper code (JEM++) | PyTorch | 13 | 2023-03 | frozen |
| wgrathwohl/GWG_release | discrete-EBM sampling (Gibbs-with-Gradients) | PyTorch | 61 | 2023-07 | frozen |
| ermongroup/ncsn | paper code (NCSN, score-based) | PyTorch | 790 | 2024-02 | frozen |
| yang-song/score_sde | paper code, library-quality | **JAX/Flax** | 1,838 | 2022-11 | frozen |
| yang-song/score_sde_pytorch | ditto | PyTorch | 2,124 | 2024-07 | frozen |
| google-research/discs | discrete-sampling benchmark | JAX | 64 | 2024-08 | frozen |
| blackjax-devs/blackjax | Bayesian MCMC library (adjacent) | JAX | 1,102 | 2026-07 | very active |
| gugarosa/learnergy | RBM/DBN library (classical EBMs) | PyTorch | 73 | 2026-05 | maintained |
| facebookresearch/eb_jepa | EB-JEPA library (SSL "energy" ≠ density) | PyTorch | 747 | 2026-07 | active, new (Feb 2026) |
| facebookresearch/flow_matching | flow-matching library (adjacent) | PyTorch | 4,640 | 2026-01 | active |
| atong01/conditional-flow-matching (TorchCFM) | flow-matching library (adjacent) | PyTorch | 2,550 | 2026-07 | active |
| yataobian/awesome-ebm | curated paper list | — | 395 | 2026-04 | maintained |

Headline: **there is exactly one general-purpose, maintained, deep-learning EBM *library***
(torchebm). Everything else is frozen paper code, a classical-RBM library, or an adjacent
diffusion/flow/MCMC library that covers part of the space.

---

## 2. torchebm (soran-ghaderi/torchebm) — the closest prior art

- GitHub 106 ★, pushed 2026-07-21 (days ago). PyPI `torchebm` v0.7.5, 18 releases.
  Solo-maintainer project with unusually serious engineering: CI, docs CI, benchmarks CI +
  public benchmarks dashboard, dependabot, CITATION.cff, examples gallery executed in CI,
  distributed-training tests (FSDP2), lazy-loaded subpackages for fast import.
- Tagline (2026): *"Simulation-free, GPU-first generative modeling in PyTorch — composable
  primitives for scalable, stable training of modern EBMs, diffusion, flow matching, and
  Schrödinger bridges."* It began as a classical EBM library (CD + Langevin/HMC) and has
  expanded into a unified generative-modeling framework.

### 2.1 Architecture: "component algebra"

From `docs/concepts/design.md` — methods factor into orthogonal components, one subpackage each:

| Component | Question | Package |
|---|---|---|
| Energy / field | what is the system? | `torchebm.core`, `torchebm.models` |
| Interpolant | along which path do noise and data connect? | `torchebm.interpolants` |
| Coupling | which noise sample pairs with which datum? | `torchebm.couplings` |
| Objective | how is the model fit? | `torchebm.losses` |
| Sampler | which dynamics produce samples? | `torchebm.samplers` |
| Integrator | how are the dynamics discretized? | `torchebm.integrators` |

Stated design principles: **composition over frameworks** (no `Trainer`, training loops are
user code over plain objects), **one contract per axis**, **vectorization first** (chains are a
batch dimension), **stateless math + `Schedulable` hyperparameters** (any step size / noise
scale / temperature accepts a float or a scheduler). Explicit scope boundaries: no pretrained
checkpoints, no dataset loaders beyond 2D synthetic, no orchestration; Schrödinger bridges and
discrete-state models named as roadmap gaps.

### 2.2 Defining an energy function

`BaseModel` (in `torchebm/core/base_model.py`) is the energy contract: any differentiable map
`(N, *dims) -> (N,)`, with autograd-derived `gradient()` overridable by analytic gradients:

```python
class BaseModel(TorchEBMModule, ABC):
    force_fp32_gradient: bool = False          # opt-in fp32 grads under AMP

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scalar energy per sample: input (batch, *dims) -> (batch,)."""

    def gradient(self, x, model_kwargs=None) -> torch.Tensor:
        # default: torch.autograd.grad of forward(x, **model_kwargs) w.r.t. x,
        # with dtype/device management, shape validation, and a clear error if
        # forward() didn't use x differentiably.
```

Built-in analytic energies with known statistics (so sampler behavior is measurable):
`GaussianModel`, `DoubleWellModel`, `HarmonicModel`, `RosenbrockModel`, `AckleyModel`,
`RastriginModel`. A user's neural EBM is just a `BaseModel` subclass whose `forward` returns
scalar energy. Conditioning flows through an explicit `model_kwargs` dict that samplers and
losses thread through every energy/gradient call.

### 2.3 Sampling

`BaseSampler` fixes a single `sample()` contract shared by all samplers:

```python
sampler = LangevinDynamics(
    energy, step_size=0.01, noise_scale=1.0, clamp=(-1, 1), integrator="euler_maruyama"
)
samples = sampler.sample(n_samples=100, dim=2, n_steps=500)
# or start from given x; options:
#   thin=..., return_trajectory=True  -> [n_samples, n_steps//thin, *dims]
#   return_diagnostics=True -> (x, {"mean": ..., "var": ..., "energy": ...,
#                                   "acceptance_rate": ...})   # per kept step
#   generator=torch.Generator(...)    -> full RNG reproducibility, per-rank seeds
#   model_kwargs={"y": labels}        -> conditional sampling
```

Samplers: `LangevinDynamics`, `HamiltonianMonteCarlo`, `RiemannianManifoldHMC`,
`GradientDescentSampler`, `NesterovSampler` (mode-seeking), and `FlowSampler`
(continuous-time generation; `mode="ode"` probability-flow or `mode="sde"` diffusion, with
`prediction=` velocity / score / noise). Notably, every sampler takes `integrator=` as a
registry string or instance — the numerics are swappable independently of the dynamics:
Heun/Bosh3/RK4/RK438, adaptive Dopri5/Dopri8/AdaptiveHeun, Euler–Maruyama (+ implicit
backward EM), Leapfrog / generalized Leapfrog. There is even a fused CUDA Langevin kernel
(`torchebm/cuda/fused_langevin.py`).

### 2.4 Losses

`torchebm.losses`:

| Objective | Inner sampling |
|---|---|
| `ContrastiveDivergence` (CD-k; `persistent=True` → PCD with replay buffer, `new_sample_ratio` reinit, `init_steps` warmup, energy-L2 regularizer, optional data noise) | yes |
| `ScoreMatching` (exact, Hessian-based) | no |
| `DenoisingScoreMatching` | no |
| `SlicedScoreMatching` (Hutchinson projections) | no |
| `EquilibriumMatchingLoss` (Wang & Du 2025 — time-invariant equilibrium field) | no |
| `EnergyMatchingLoss` (Balcerak et al. 2025 — OT flow warm-up, then contrastive sharpening) | phase 2 only |

Losses take the model *and* sampler as constructor args; training loops are plain PyTorch:

```python
energy = MyEnergyNet()  # BaseModel subclass
sampler = LangevinDynamics(energy, step_size=0.01)
cd = ContrastiveDivergence(
    model=energy, sampler=sampler, k_steps=10, persistent=True, buffer_size=10000
)
opt = torch.optim.Adam(energy.parameters(), lr=1e-3)
for x in loader:
    loss, neg_samples = cd(x)  # runs k MCMC steps internally
    opt.zero_grad()
    loss.backward()
    opt.step()
```

Internals of `ContrastiveDivergence.compute_loss` worth noting: loss is
`E[x_data] - E[x_model]` + optional `energy_reg_weight * (E²_pos + E²_neg)` stabilizer, and a
**sync-free NaN guard** (`torch.where(isfinite(loss), loss, 0.1)` instead of a CPU-syncing
`if isnan`). The file also contains skeleton stubs `PersistentContrastiveDivergence` and
`ParallelTemperingCD` whose bodies are commented out — visible work-in-progress.

Modern-methods snippets from the README (each "runs as-is"):

```python
# Equilibrium matching, then sample by ODE integration or energy descent
loss_fn = EquilibriumMatchingLoss(model=field, interpolant="linear", energy_type="dot")
flow = FlowSampler(model=field, interpolant="linear", negate_velocity=True, integrator="euler")
samples = flow.sample(x=torch.randn(1000, 2), n_steps=100)

# Energy matching: two-phase, then temperature-scheduled Langevin
loss_fn = EnergyMatchingLoss(
    model=potential,
    coupling=SinkhornCoupling(reg=0.01),
    lambda_cd=0.0,
    epsilon_max=0.15,
    tau_star=0.8,
)
...
loss_fn.lambda_cd = 2.0  # switch on contrastive phase
temperature = TemperatureScheduler(epsilon_max=0.15, tau_star=0.8, n_steps=200, t_end=1.0)
samples = LangevinDynamics(model=potential, step_size=0.01, noise_scale=temperature).sample(
    x=torch.randn(4000, 2), n_steps=200
)

# Score-SDE diffusion is a FlowSampler configuration, not a separate subsystem
diffusion = FlowSampler(model=field, mode="sde", interpolant="vp", prediction="noise")
```

Plus `torchebm.interpolants` (`LinearInterpolant`, `CosineInterpolant`,
`VariancePreservingInterpolant`), `torchebm.couplings` (`IndependentCoupling`,
`GreedyCoupling`, `SinkhornCoupling`, `ExactOTCoupling`, `UnbalancedSinkhornCoupling`,
`ReflowCoupling`), `torchebm.datasets` (8 synthetic 2D generators), `torchebm.models`
(`ConditionalTransformer2D`, CFG wrapper, transformer components), `torchebm.distributed`.

### 2.5 Strengths / weaknesses

**Strengths**
- The only *maintained, packaged* EBM library; very recent activity; real engineering
  discipline (CI-executed examples, benchmarks dashboard, FSDP2 tests, RNG `generator=`
  threading, AMP/dtype care, lazy imports).
- The component-algebra design is genuinely good prior art: energy/interpolant/coupling/
  objective/sampler/integrator as orthogonal axes, registries + class paths for each.
- Covers both classical (CD/PCD, SM/DSM/SSM, Langevin/HMC) and 2024–2025 methods
  (equilibrium matching, energy matching, OT couplings, flow/score sampling) in one API.
- Docs are substantive: concepts pages (design, objectives, sampling, transport), blog posts,
  graded examples gallery (`00-foundations` → `90-showcase`).

**Weaknesses / openings**
- Tiny community: 106 stars, essentially one maintainer; bus factor 1; API still churning
  (deprecation shims in CD; commented-out stubs like `ParallelTemperingCD` shipped in the
  package).
- Scope has drifted toward the crowded flow-matching/diffusion space, where
  facebookresearch/flow_matching (4.6k★) and TorchCFM (2.5k★) are entrenched — its EBM core
  is the differentiated part, but the branding now leads with "simulation-free."
- No discrete-EBM support (Gibbs-with-Gradients-style samplers), no RBM/classical lineage, no
  JEM-style classifier-EBM utilities, no AIS / log-Z estimation, no OOD-score utilities —
  i.e., the *evaluation* side of EBMs is thin (diagnostics are mean/var/energy only; no ESS,
  R-hat, FID hooks).
- Losses return inconsistent shapes (`ContrastiveDivergence` returns `(loss, neg_samples)`
  tuple; score-matching losses return scalars). Some legacy dtype/device plumbing
  (`device=torch.device("cpu")` defaults in loss constructors).
- PyTorch-only; nothing for JAX users.

---

## 3. Canonical research codebases (frozen, but the API patterns matter)

### 3.1 wgrathwohl/JEM — "Your classifier is secretly an EBM" (437★, last push 2022)

Flat research code: `train_wrn_ebm.py`, `eval_wrn_ebm.py`, `wideresnet.py`. The enduring API
idea is the **classifier-as-EBM wrapper**: a K-class classifier's logits define
`E(x, y) = -logits[y]` and `E(x) = -logsumexp_y logits[y]`:

```python
class CCF(F):  # F wraps a WideResNet
    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return logits.logsumexp(1)  # unnormalized log p(x)
        else:
            return t.gather(logits, 1, y[:, None])  # log p(x, y)
```

Training combines cross-entropy on `classify(x)` with SGLD-based maximum likelihood on
`logsumexp`; the SGLD sampler is a closure with a class-conditional replay buffer:

```python
def sample_q(f, replay_buffer, y=None, n_steps=args.n_steps):
    init_sample, buffer_inds = sample_p_0(
        replay_buffer, bs, y=y
    )  # buffer ∪ reinit_freq fresh noise
    x_k = t.autograd.Variable(init_sample, requires_grad=True)
    for k in range(n_steps):
        f_prime = t.autograd.grad(f(x_k, y=y).sum(), [x_k], retain_graph=True)[0]
        x_k.data += args.sgld_lr * f_prime + args.sgld_std * t.randn_like(x_k)
    replay_buffer[buffer_inds] = x_k.detach().cpu()
    return x_k.detach()
```

Everything is `argparse`-driven; nothing reusable without copy-paste. Famous for instability
(the README documents divergence/restart folklore). Downstream: **JEM++** (sndnyang/JEMPP,
13★) adds proximal SGLD, YOPO, informative init; **VERA** (wgrathwohl/VERA, 64★) trains EBMs
without MCMC via a variational generator.

### 3.2 openai/ebm_code_release — Du & Mordatch, Implicit Generation (364★, TF1, dead)

**TensorFlow 1 + horovod + MPI**; unusable today without archaeology (`scipy.misc.imsave`,
TF1 sessions). Historically important for: replay buffer at scale, spectral-norm ResNet
energies, conditional EBMs, compositionality demos, HMC experiments (`hmc.py`), and AIS
evaluation (`ais.py` — one of the few public log-Z estimation implementations). All
configuration through ~60 `absl` flags (`objective ∈ {cd, logsumexp, softplus}`,
`replay_batch`, `pcd`, `hmc`, `proj_norm`, ...). The PyTorch community substitute is
**rosinality/igebm-pytorch** (68★, dead since 2019) with a clean minimal `SampleBuffer` +
SGLD loop, and Du's later **yilundu/improved_contrastive_divergence** (ICML'21, 70★) which
adds the KL-divergence gradient term dropped by standard CD, data augmentation between MCMC
steps, and multiscale energies.

### 3.3 point0bar1/ebm-anatomy — "Anatomy of MCMC-based ML" (39★)

PyTorch, config-driven (JSON in `config_locker/`), ~6 files. The most *pedagogically*
valuable EBM code: explicit taxonomy of Langevin init strategies and the
convergent/non-convergent training distinction, with diagnostics built into the loop:

```python
def sample_s_t(batch_size, L, init_type, update_s_t_0=True):
    # init_type ∈ {"persistent", "data", "uniform", "gaussian"}
    x_s_t = t.autograd.Variable(x_s_t_0.clone(), requires_grad=True)
    r_s_t = t.zeros(1)  # avg gradient magnitude along the path
    for ell in range(L):
        f_prime = t.autograd.grad(f(x_s_t).sum(), [x_s_t])[0]
        x_s_t.data += -f_prime + config["epsilon"] * t.randn_like(x_s_t)
        r_s_t += f_prime.view(len(f_prime), -1).norm(dim=1).mean()
```

It records `d_s_t` (pos/neg energy gap) and `r_s_t` (gradient magnitude) every iteration —
exactly the training-health diagnostics a library should productize. Also: LR scaled by
`ε²/2` for noise-invariant tuning. Related: enijkamp/short_run (short-run MCMC as a learned
generator, 6★).

### 3.4 Score/diffusion-adjacent: yang-song/score_sde (JAX, 1.8k★) & score_sde_pytorch (2.1k★)

Not an EBM library per se, but the best-engineered sampler architecture in the family, and the
direct ancestor of torchebm's integrator/registry design. Key patterns in `sampling.py`:

```python
_PREDICTORS, _CORRECTORS = {}, {}

@register_predictor(name='euler_maruyama') ...
@register_corrector(name='langevin') ...        # annealed Langevin as a *corrector*

def get_pc_sampler(sde, model, shape, predictor, corrector, snr, n_steps_each, ...):
    # Predictor–Corrector: any reverse-SDE discretization (predictor) composed
    # with any score-MCMC (corrector); ODE sampler via black-box scipy integrate.
```

- SDEs are first-class objects (`sde_lib.VPSDE/VESDE/subVPSDE`) exposing marginal params,
  discretization, and reverse-SDE construction; samplers are *generated* (`get_pc_sampler`
  returns a jitted function) rather than instantiated — the idiomatic JAX style.
- Decorator-based registries keyed by config strings (`config.sampling.predictor='ancestral'`)
  — the pattern torchebm mirrors with `get_integrator` / string kwargs.
- Precursor: ermongroup/ncsn (790★) — annealed Langevin over a noise ladder.

---

## 4. JAX ecosystem: no dedicated EBM library exists

Searches (July 2026) find **no maintained JAX EBM library** — only scattered paper repos and
notebooks. What exists is infrastructure that an EBM stack would sit on:

- **blackjax** (1.1k★, very active): MALA, HMC/NUTS, SGLD/SGHMC in a functional
  `kernel(rng_key, state) -> state` API. Aimed at Bayesian posteriors (log-density given), not
  at learning `E_θ`; no CD-style training loop, no replay buffers — but its sampler-kernel
  contract is the reference for functional API design.
- **yang-song/score_sde** (above) — Flax, frozen.
- **google-research/discs** (64★): benchmark of *discrete*-space samplers (GWG, path
  auxiliary, DLMC...) in JAX — the discrete-EBM sampling state of the art lives here and in
  wgrathwohl/GWG_release, not in any library.
- **lollcat/fab-jax** (13★): Flow-Annealed Importance Sampling Bootstrap — Boltzmann-targets
  niche (relevant to log-Z / AIS features).

This is the largest structural gap in the ecosystem: a "blackjax for learned energies" does
not exist.

## 5. Other libraries in the neighborhood

- **learnergy** (73★, maintained, PyPI): PyTorch RBM/DBN library — Bernoulli/Gaussian/ReLU
  RBMs, Conv-DBNs, discriminative RBMs, with a sklearn-ish `model.fit(dataset, batch_size,
  epochs)` API returning `(mse, pseudo_likelihood)`. Covers the *classical* EBM lineage only;
  no deep energy networks, no Langevin-trained EBMs.
- **facebookresearch/eb_jepa** (747★, released Feb 2026 with arXiv:2602.03604): "energy-based"
  in the LeCun JEPA sense — energy as a learned compatibility score in representation space,
  trained with regularized objectives (no MCMC, no density). Different problem; relevant
  mainly because it will absorb the "energy-based library" search traffic and mindshare.
- **UvA Deep Learning tutorials** (phlippe/uvadlc_notebooks, 3.2k★): Tutorial 8 "Deep
  Energy-Based Generative Models" is the de-facto onboarding document for EBM training
  (`Sampler` class with replay buffer + SGLD, CD with energy regularization, PyTorch
  Lightning). Its ubiquity signals demand for a library that packages exactly this.
- **yataobian/awesome-ebm** (395★, updated 2026-04): almost entirely a *paper* list; the only
  library-like links it carries are learnergy, torchebm, ncsn, CoopNets, icebeem, and
  mini-ebm. Confirms that the tooling side of the field is thin.
- **Compositional/EBM-diffusion hybrids** (Yilun Du's orbit): yilundu/reduce_reuse_recycle
  (ICML'23, 151★ — annealed MCMC on diffusion-model energies, compositional operators),
  energy-based-model org repos (composable diffusion 489★, COMET, IRED). All paper code, all
  demonstrate demand for *composable energies* (`E1(x)+E2(x)`, products/mixtures of experts)
  that no library currently offers as an API.

---

## 6. Gap analysis — what a new, well-engineered EBM library should provide

**The competitive landscape in one line:** frozen paper code (JEM, Du & Mordatch, anatomy,
IGEBM), one young solo-maintainer library that is drifting toward flow matching (torchebm),
strong adjacent libraries that own their slices (blackjax → Bayesian MCMC, flow_matching /
TorchCFM → simulation-free transport, learnergy → RBMs), and nothing at all in JAX.

Concrete gaps, roughly ordered by value:

1. **Training reliability as a product.** Every EBM codebase reimplements the same folklore:
   replay buffers with reinit fraction, spectral norm, energy-L2 regularizers, data noise,
   step-size/noise tuning, divergence detection-and-restart. No library ships (a) opinionated,
   tested "stable CD" recipes with defaults known to converge on standard benchmarks, (b)
   ebm-anatomy-style *live diagnostics* (pos/neg energy gap, Langevin gradient magnitude,
   buffer statistics, acceptance rates, ESS/R-hat) surfaced as first-class training-health
   telemetry with actionable warnings ("your chains are not mixing; energy gap diverging").
   This is the single biggest unmet need.
2. **Evaluation and normalization tooling.** AIS / RAISE / SMC log-Z estimation, annealed
   bridge sampling, likelihood bounds, OOD scoring utilities, sample-quality hooks (FID
   plumbing). Only the dead TF1 OpenAI repo ever shipped AIS. Nothing packaged exists.
3. **A JAX (or framework-dual) EBM stack.** "blackjax for learned energies": pure-function
   energy `E(params, x)`, sampler kernels, CD/SM losses as `loss(params, rng, batch)`,
   vmap/pmap-native chains. Zero competition today; also the natural substrate for
   `jit`-compiled inner MCMC loops that PyTorch handles poorly (though `torch.compile` on
   sampler loops is itself an unexplored differentiator).
4. **Discrete EBMs.** Gibbs-with-Gradients and successors exist only as paper code
   (GWG_release, discs). No library offers discrete samplers + CD over
   binary/categorical/graph data. Large applied audience (proteins, molecules, text).
5. **Classifier-EBM (JEM) as a wrapper, done right.** `JEMWrapper(classifier)` giving
   `logsumexp` energy, hybrid CE+CD training recipe, calibration/OOD/adversarial-robustness
   evaluation — the JEM repo has 437★ and is frozen; JEM++ improvements are unpackaged.
6. **Compositionality API.** Energies as an algebra: `E1 + E2`, tempered `E/T`, products of
   experts, classifier guidance on energies, MCMC over composed diffusion energies (reduce-
   reuse-recycle). Demonstrably in demand, nowhere productized.
7. **Interoperability with the transport world rather than competition.** Accept a
   diffusion/flow model as an initialization or proposal (short Langevin refinement of
   flow-matching samples; EBM as reranker/refiner), convert score ↔ energy ↔ velocity.
   torchebm gestures at this; a focused EBM library can own the "calibrated energy on top of
   your generative model" niche instead of rebuilding flow matching.
8. **Table stakes it must match (torchebm sets the bar):** scalar-energy `forward` contract
   with autograd `gradient` fallback + analytic override; one `sample()` contract with
   trajectories/thinning/diagnostics/`generator` reproducibility; schedulable
   hyperparameters; string registries + class instances for samplers/integrators; losses
   composed from `(model, sampler)`; AMP/dtype/device care; CI-executed examples; docs with
   the math. Also learn from its miscues: consistent loss return types, no dead stub classes
   in shipped wheels, and a name that neither collides with Explainable Boosting Machines nor
   gets diluted into a general generative-modeling framework.

### Key file references (local copies of surveyed sources)

Scratch copies used for this report were fetched from GitHub master branches on 2026-07-27:
torchebm (`torchebm/core/base_model.py`, `core/base_sampler.py`, `losses/contrastive_divergence.py`,
`losses/score_matching.py`, `samplers/langevin_dynamics.py`, `docs/concepts/design.md`, README),
JEM `train_wrn_ebm.py`, openai `train.py`, ebm-anatomy `train_toy.py`, igebm-pytorch `train.py`,
score_sde `sampling.py`, learnergy README. Star counts / push dates via GitHub API, same day.
