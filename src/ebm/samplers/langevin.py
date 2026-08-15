"""Langevin samplers: unadjusted (ULA/SGLD) and Metropolis-adjusted (MALA)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler


def _flat_sum(x: Tensor) -> Tensor:
    """Sum over all non-batch dimensions."""
    return x.reshape(x.shape[0], -1).sum(dim=1)


class LangevinDynamics(Sampler):
    """Unadjusted Langevin dynamics: ``x' = x - ε ∇E(x) + σ ξ``.

    With ``noise_scale=None`` the mathematically correct ``σ = sqrt(2ε)`` is
    used and the chain targets ``p ∝ exp(-E)`` (up to discretization bias).
    Practitioner "short-run" training for images instead uses a much colder,
    decoupled noise (e.g. ``step_size=10.0, noise_scale=0.005`` with
    ``grad_clip=0.01`` in IGEBM) — both regimes are one keyword apart.

    Args:
        step_size: ε, the gradient step size.
        steps: Default number of transitions per ``sample`` call.
        noise_scale: σ; ``None`` means ``sqrt(2 * step_size)``.
        grad_clip: If set, clamp each gradient element to ``[-grad_clip, grad_clip]``.
        clamp: If set, clamp samples to ``(lo, hi)`` after every step
            (e.g. ``(-1, 1)`` for normalized images).
    """

    def __init__(
        self,
        step_size: float = 1e-2,
        steps: int = 100,
        noise_scale: float | None = None,
        grad_clip: float | None = None,
        clamp: tuple[float, float] | None = None,
    ):
        super().__init__(steps)
        self.step_size = step_size
        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
        self.clamp = clamp

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        _, grad = self._energy_grad(energy, x)
        if self.grad_clip is not None:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        sigma = self.noise_scale if self.noise_scale is not None else math.sqrt(2 * self.step_size)
        x = x.detach() - self.step_size * grad + sigma * torch.randn_like(x)
        if self.clamp is not None:
            x = x.clamp(*self.clamp)
        return x


class UnderdampedLangevin(Sampler):
    """Underdamped (2nd-order) Langevin / SGHMC — momentum-augmented dynamics.

    Augments the state with a velocity ``v`` and integrates

    ``v ← (1 - εγ) v - ε ∇E(x) + sqrt(2γε) ξ`` ,  ``x ← x + ε v``

    (symplectic Euler of Chen et al., 2014). The stationary distribution is
    ``exp(-E(x) - ½‖v‖²)``, so the position marginal is exactly ``p ∝ exp(-E)``;
    the momentum smooths the trajectory and mixes better than overdamped ULA on
    correlated targets. The velocity is fresh state per ``sample`` call (reset to
    zero), not carried between runs.

    Args:
        step_size: ε, the integrator step.
        friction: γ, the damping (larger → closer to overdamped Langevin).
        steps: default number of transitions per ``sample`` call.
    """

    def __init__(self, step_size: float = 0.1, friction: float = 1.0, steps: int = 100):
        super().__init__(steps)
        self.step_size = step_size
        self.friction = friction
        self._v: Tensor | None = None

    def sample(
        self,
        energy: EnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        self._v = torch.zeros_like(x_init)  # momentum is per-run state
        return super().sample(energy, x_init, steps=steps, return_trajectory=return_trajectory)

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        eps, gamma = self.step_size, self.friction
        if self._v is None or self._v.shape != x.shape:
            self._v = torch.zeros_like(x)
        _, grad = self._energy_grad(energy, x)
        noise = math.sqrt(2 * gamma * eps) * torch.randn_like(x)
        v = (1 - eps * gamma) * self._v - eps * grad + noise
        self._v = v.detach()
        return x.detach() + eps * v


class PreconditionedLangevin(Sampler):
    """Unadjusted Langevin with a fixed diagonal preconditioner ``M``.

    ``x ← x - ε (M ⊙ ∇E(x)) + sqrt(2ε) sqrt(M) ⊙ ξ``. A constant ``M`` rescales
    each coordinate's step and noise together, so the chain still targets exactly
    ``p ∝ exp(-E)`` (no drift-correction term is needed, unlike adaptive
    preconditioners). Set ``M`` to roughly the target's per-coordinate variance
    to fix the slow mixing overdamped Langevin suffers on ill-conditioned
    targets. ``precond`` broadcasts against the data's event shape.
    """

    def __init__(self, precond: Tensor, step_size: float = 1e-2, steps: int = 100):
        super().__init__(steps)
        self.step_size = step_size
        precond = torch.as_tensor(precond, dtype=torch.float32)
        if (precond <= 0).any():
            raise ValueError("preconditioner entries must be positive")
        self._precond = precond

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        eps = self.step_size
        m = self._precond.to(device=x.device, dtype=x.dtype)
        _, grad = self._energy_grad(energy, x)
        noise = torch.sqrt(2 * eps * m) * torch.randn_like(x)
        return x.detach() - eps * (m * grad) + noise


class MALA(Sampler):
    """Metropolis-adjusted Langevin: ULA proposals with an accept/reject step.

    Unbiased (targets exactly ``p ∝ exp(-E)``), at roughly twice the cost per
    step of ULA. ``last_accept_rate`` reports the acceptance rate of the most
    recent step; tune ``step_size`` toward ~0.5–0.7.
    """

    def __init__(self, step_size: float = 1e-2, steps: int = 100):
        super().__init__(steps)
        self.step_size = step_size

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        eps = self.step_size
        e_x, grad_x = self._energy_grad(energy, x)
        mean_fwd = x.detach() - eps * grad_x
        proposal = mean_fwd + math.sqrt(2 * eps) * torch.randn_like(x)

        e_prop, grad_prop = self._energy_grad(energy, proposal)
        mean_bwd = proposal.detach() - eps * grad_prop

        log_q_fwd = -_flat_sum((proposal - mean_fwd).pow(2)) / (4 * eps)
        log_q_bwd = -_flat_sum((x.detach() - mean_bwd).pow(2)) / (4 * eps)
        log_alpha = (e_x - e_prop) + (log_q_bwd - log_q_fwd)

        accept = torch.log(torch.rand_like(log_alpha)) < log_alpha
        self._last_accept = accept.float().mean()
        accept = accept.reshape(-1, *([1] * (x.dim() - 1)))
        return torch.where(accept, proposal.detach(), x.detach())
