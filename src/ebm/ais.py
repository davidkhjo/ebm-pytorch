"""Annealed importance sampling (Neal, 2001) for log-partition estimation.

Estimates ``log Z`` of ``p(x) ∝ exp(-E(x))`` by annealing from a tractable
base distribution through ``log γ_β(x) = (1-β) log p₀(x) - β E(x)``, running
an MCMC transition at each temperature and accumulating importance weights in
log space. With ``log Z`` in hand, ``log_likelihood`` turns raw energies into
(estimated) log densities.

Caveat: AIS is a stochastic *lower* bound on ``log Z`` in expectation
(Jensen), so derived likelihoods lean optimistic. Increase ``n_temps`` until
the estimate stabilizes; watch ``ess`` (effective sample size) — values near 1
mean a few chains dominate and the estimate is unreliable.

``reverse_ais_log_z`` runs the same path backwards from model samples and is
a stochastic *upper* bound (RAISE; Burda et al., 2015). Together the two
bracket the true ``log Z``: when the gap is small, both are trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor, nn

from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler
from ebm.samplers.hmc import HMC
from ebm.utils import frozen_params


class _AdaptiveSampler(Protocol):
    """A sampler whose ``step_size`` AIS may tune from the acceptance rate."""

    step_size: float


@dataclass
class AISResult:
    """Result of an AIS run.

    Attributes:
        log_z: Estimated log partition function.
        log_weights: Per-chain log importance weights ``(n_chains,)``.
        ess: Effective sample size in ``[1, n_chains]``.
        stderr: Delta-method standard error of ``log_z``.
        samples: Final chain states at β=1, approximately ``~ p``.
    """

    log_z: float
    log_weights: Tensor
    ess: float
    stderr: float
    samples: Tensor


def _make_betas(schedule: str | Tensor, n_temps: int) -> Tensor:
    if isinstance(schedule, Tensor):
        betas = schedule.double()
        if betas.dim() != 1 or len(betas) < 2 or betas[0] != 0 or betas[-1] != 1:
            raise ValueError("explicit betas must be 1D, start at 0, and end at 1")
        if (betas.diff() <= 0).any():
            raise ValueError("explicit betas must be strictly increasing")
        return betas
    if schedule == "linear":
        return torch.linspace(0, 1, n_temps + 1, dtype=torch.float64)
    if schedule == "geometric":
        # Dense near β=0, where the annealing path moves fastest from a broad base.
        return torch.cat(
            [
                torch.zeros(1, dtype=torch.float64),
                torch.logspace(-4, 0, n_temps, dtype=torch.float64),
            ]
        )
    raise ValueError(f"unknown schedule {schedule!r}")


def _default_base(
    shape: tuple[int, ...], base_scale: float, device: torch.device | str | None
) -> torch.distributions.Distribution:
    loc = torch.zeros(shape, device=device)
    return torch.distributions.Independent(
        torch.distributions.Normal(loc, base_scale * torch.ones_like(loc)), len(shape)
    )


def _anneal(
    energy: EnergyFn,
    base: torch.distributions.Distribution,
    x: Tensor,
    betas: Tensor,
    sampler: Sampler,
    transitions: int,
    adapt_step_size: bool,
) -> tuple[Tensor, Tensor]:
    """Run the annealing chain along ``betas`` (either direction).

    At each move β → β' the log-weight increment is
    ``(β' - β) (-E(x) - log p₀(x))``, followed by an MCMC transition targeting
    ``γ_{β'}``. Returns the final states and per-chain accumulated log-weights.
    """
    module = energy if isinstance(energy, nn.Module) else None
    original_step_size = getattr(sampler, "step_size", None)
    log_w = torch.zeros(x.shape[0], dtype=torch.float64, device=x.device)

    try:
        with frozen_params(module):
            for beta_prev, beta_next in zip(betas[:-1], betas[1:], strict=True):
                with torch.no_grad():
                    # log γ_{β'}(x) - log γ_β(x) = Δβ (-E(x) - log p₀(x)).
                    delta = float(beta_next - beta_prev)
                    log_w += delta * (-energy(x) - base.log_prob(x)).double()

                b = float(beta_next)

                def target(z: Tensor, b: float = b) -> Tensor:
                    return -(1 - b) * base.log_prob(z) + b * energy(z)

                x = sampler.sample(target, x, steps=transitions)

                if adapt_step_size and original_step_size is not None:
                    rate = getattr(sampler, "last_accept_rate", None)
                    if rate is not None:
                        adaptive = cast("_AdaptiveSampler", sampler)
                        if rate > 0.8:
                            adaptive.step_size *= 1.1
                        elif rate < 0.5:
                            adaptive.step_size *= 0.9
    finally:
        if original_step_size is not None:
            cast("_AdaptiveSampler", sampler).step_size = original_step_size
    return x, log_w


def ais_log_z(
    energy: EnergyFn,
    shape: tuple[int, ...],
    *,
    base: torch.distributions.Distribution | None = None,
    base_scale: float = 1.0,
    n_temps: int = 1000,
    n_chains: int = 64,
    transitions: int = 1,
    sampler: Sampler | None = None,
    schedule: str | Tensor = "linear",
    adapt_step_size: bool = True,
    device: torch.device | str | None = None,
) -> AISResult:
    """Estimate ``log Z = log ∫ exp(-E(x)) dx`` by annealed importance sampling.

    Args:
        energy: Energy function ``(B, *shape) -> (B,)``.
        shape: Event shape of a single sample.
        base: Normalized base distribution with ``sample`` and a differentiable
            ``log_prob`` over single events. Defaults to
            ``N(0, base_scale² I)`` over ``shape``; choose it broad enough to
            cover the target's support.
        n_temps: Number of intermediate temperatures. Many cheap temperatures
            beat few long MCMC runs for AIS variance.
        n_chains: Number of independent AIS chains.
        transitions: MCMC steps per temperature.
        sampler: Transition kernel; defaults to
            ``HMC(step_size=0.1, leapfrog_steps=5)``. MALA also works well.
        schedule: ``"linear"``, ``"geometric"``, or an explicit increasing
            tensor of betas from 0 to 1.
        adapt_step_size: Adapt the sampler's step size between temperatures
            from its ``last_accept_rate`` (uses only past temperatures, so the
            estimate stays valid). The original step size is restored.
    """
    if base is None:
        base = _default_base(shape, base_scale, device)
    if sampler is None:
        sampler = HMC(step_size=0.1, leapfrog_steps=5, steps=transitions)

    betas = _make_betas(schedule, n_temps)
    x = base.sample((n_chains,))
    if device is not None:
        x = x.to(device)

    x, log_w = _anneal(energy, base, x, betas, sampler, transitions, adapt_step_size)

    log_z = (torch.logsumexp(log_w, dim=0) - math.log(n_chains)).item()
    ess = torch.exp(2 * torch.logsumexp(log_w, 0) - torch.logsumexp(2 * log_w, 0)).item()
    norm_w = torch.exp(log_w - log_w.logsumexp(0) + math.log(n_chains))
    stderr = (norm_w.std() / math.sqrt(n_chains)).item()
    return AISResult(log_z=log_z, log_weights=log_w, ess=ess, stderr=stderr, samples=x)


def reverse_ais_log_z(
    energy: EnergyFn,
    x: Tensor,
    *,
    base: torch.distributions.Distribution | None = None,
    base_scale: float = 1.0,
    n_temps: int = 1000,
    transitions: int = 1,
    sampler: Sampler | None = None,
    schedule: str | Tensor = "linear",
    adapt_step_size: bool = True,
) -> AISResult:
    """Estimate ``log Z`` by reverse AIS from model samples (RAISE-style upper bound).

    Runs the AIS path of `ais_log_z` backwards: chains start at ``x`` and
    anneal β from 1 down to 0. The reverse weights satisfy
    ``E[w] = Z₀/Z = 1/Z``, so by Jensen the estimate is a stochastic *upper*
    bound on ``log Z`` in expectation — the mirror image of forward AIS.
    Report both to bracket the truth::

        lower = ebm.ais_log_z(energy, shape=(2,), n_temps=500)
        upper = ebm.reverse_ais_log_z(energy, samples, n_temps=500)

    The bound (and the bracket) is only as honest as ``x``: it holds exactly
    for exact samples from ``p(x) ∝ exp(-E(x))``, and approximately for
    long-run MCMC samples (e.g. ``ais_log_z(...).samples`` or a long
    Langevin/HMC run). Poor samples bias the estimate further *upward*, so the
    bracket stays conservative rather than falsely tight.

    Args:
        energy: Energy function ``(B, *shape) -> (B,)``.
        x: Approximate model samples ``(n_chains, *shape)``; one reverse chain
            starts from each.
        base: Same role and default (``N(0, base_scale² I)``) as in
            `ais_log_z`; use the *same* base in both directions when
            bracketing.
        schedule: Same options as `ais_log_z`, specified in forward
            orientation (0 → 1); it is reversed internally.

    Returns:
        `AISResult` whose ``samples`` are the final states at β=0
        (approximately base-distributed — a sanity check, not model samples).
    """
    if base is None:
        base = _default_base(tuple(x.shape[1:]), base_scale, x.device)
    if sampler is None:
        sampler = HMC(step_size=0.1, leapfrog_steps=5, steps=transitions)

    betas = _make_betas(schedule, n_temps).flip(0)
    n_chains = x.shape[0]

    x, log_w = _anneal(energy, base, x.detach(), betas, sampler, transitions, adapt_step_size)

    # (1/S) Σ exp(log_w) estimates Z₀/Z = 1/Z, so log Z ≈ log S - logsumexp(log_w).
    log_z = (math.log(n_chains) - torch.logsumexp(log_w, dim=0)).item()
    ess = torch.exp(2 * torch.logsumexp(log_w, 0) - torch.logsumexp(2 * log_w, 0)).item()
    norm_w = torch.exp(log_w - log_w.logsumexp(0) + math.log(n_chains))
    stderr = (norm_w.std() / math.sqrt(n_chains)).item()
    return AISResult(log_z=log_z, log_weights=log_w, ess=ess, stderr=stderr, samples=x)


@torch.no_grad()
def log_likelihood(energy: EnergyFn, x: Tensor, log_z: float, *, dim: int | None = None) -> Tensor:
    """Per-sample ``log p(x) = -E(x) - log Z`` in nats, or bits/dim with ``dim``.

    Higher is better in both units; the commonly reported "bits per dim" is
    the negation of the ``dim``-normalized value returned here.
    """
    ll = -energy(x) - log_z
    if dim is not None:
        ll = ll / (dim * math.log(2))
    return ll
