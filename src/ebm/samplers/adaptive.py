"""Self-tuning MALA: dual-averaging step-size warmup + optional preconditioner."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ebm._functional import flat_sum as _flat_sum
from ebm._functional import mh_accept
from ebm.energy import EnergyFn
from ebm.samplers.langevin import MALA
from ebm.utils import frozen_params


class AdaptiveMALA(MALA):
    """MALA that tunes its own ``step_size`` (and optionally a diagonal metric).

    A warmup phase adapts ``ε`` with Nesterov dual averaging (Hoffman & Gelman,
    2014) toward a target acceptance rate — ``0.574`` is optimal for MALA — then
    freezes the *averaged* ``ε`` and runs an ordinary (unbiased) MALA chain. No
    hand-tuning of ``step_size``; the passed value is only the starting guess.

    The recursion drives the smooth **acceptance probability**
    ``ᾱ = mean min(1, e^{logα})`` (lower variance than the 0/1 accept indicator)
    to ``target_accept``, updating ``logε̄`` as a running average so the frozen
    value is stable rather than the last noisy iterate.

    With ``precondition=True`` a diagonal metric ``M`` (per-coordinate, geometric
    mean 1) is estimated from a first warmup window's draws, then dual averaging
    is **restarted** and ``ε`` re-tuned under that fixed ``M`` — the step and its
    noise rescale together (``x ← x − ε M∇E + sqrt(2εM) ξ``), so the chain still
    targets exactly ``p ∝ exp(-E)``. This fixes the slow mixing plain MALA
    suffers on ill-conditioned targets.

    Args:
        step_size: initial ε (a starting guess; overwritten by warmup).
        steps: default number of *post-warmup* transitions per ``sample`` call.
        warmup: dual-averaging iterations (per window; two windows if preconditioning).
        target_accept: δ, the acceptance the warmup targets (0.574 is MALA-optimal).
        precondition: estimate and use a diagonal metric ``M``.
        gamma, t0, kappa: dual-averaging shrinkage / stabilization / decay constants.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        steps: int = 100,
        *,
        warmup: int = 1000,
        target_accept: float = 0.574,
        precondition: bool = False,
        gamma: float = 0.05,
        t0: float = 10.0,
        kappa: float = 0.75,
    ):
        super().__init__(step_size, steps)
        if not 0.0 < target_accept < 1.0:
            raise ValueError("target_accept must be in (0, 1)")
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        self.warmup = warmup
        self.target_accept = target_accept
        self.precondition = precondition
        self.gamma = gamma
        self.t0 = t0
        self.kappa = kappa
        self._precond: Tensor | None = None

    @property
    def preconditioner(self) -> Tensor | None:
        """The fitted diagonal metric ``M`` (``None`` until a preconditioned run)."""
        return self._precond

    def _transition(self, energy: EnergyFn, x: Tensor) -> tuple[Tensor, Tensor]:
        """One MALA transition; returns ``(x_next, mean acceptance probability)``.

        Honors ``self._precond`` (a diagonal metric ``M``; ``None`` = identity) in
        both the drift and the noise, keeping the proposal reversible.
        """
        eps = self.step_size
        m = self._precond
        e_x, grad_x = self._energy_grad(energy, x)
        mean_fwd = x.detach() - eps * (grad_x if m is None else m * grad_x)
        if m is None:
            noise = math.sqrt(2 * eps) * torch.randn_like(x)
        else:
            noise = torch.sqrt(2 * eps * m) * torch.randn_like(x)
        proposal = mean_fwd + noise

        e_prop, grad_prop = self._energy_grad(energy, proposal)
        mean_bwd = proposal.detach() - eps * (grad_prop if m is None else m * grad_prop)

        inv = 1.0 if m is None else 1.0 / m
        log_q_fwd = -_flat_sum((proposal - mean_fwd).pow(2) * inv) / (4 * eps)
        log_q_bwd = -_flat_sum((x.detach() - mean_bwd).pow(2) * inv) / (4 * eps)
        log_alpha = (e_x - e_prop) + (log_q_bwd - log_q_fwd)

        accept = torch.log(torch.rand_like(log_alpha)) < log_alpha
        self._last_accept = accept.float().mean()
        accept_prob = torch.exp(log_alpha.clamp(max=0.0)).mean()
        return mh_accept(x.detach(), proposal.detach(), accept), accept_prob

    def _dual_average(
        self, energy: EnergyFn, x: Tensor, n: int, eps0: float, *, collect: bool
    ) -> tuple[Tensor, float, list[Tensor]]:
        """Run ``n`` dual-averaging steps from ``eps0``; return ``(x, ε̄, draws)``."""
        if n == 0:
            return x, eps0, []
        mu = math.log(10 * eps0)  # bias adaptation toward larger steps
        log_eps = math.log(eps0)
        log_ebar = 0.0
        h_bar = 0.0
        draws: list[Tensor] = []
        for m in range(1, n + 1):
            self.step_size = math.exp(log_eps)
            x, accept_prob = self._transition(energy, x)
            x = x.detach()
            gap = self.target_accept - float(accept_prob)
            h_bar = (1 - 1 / (m + self.t0)) * h_bar + gap / (m + self.t0)
            log_eps = mu - math.sqrt(m) / self.gamma * h_bar
            eta = m**-self.kappa
            log_ebar = eta * log_eps + (1 - eta) * log_ebar
            if collect and m > n // 2:
                draws.append(x)
        return x, math.exp(log_ebar), draws

    def sample(
        self,
        energy: EnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        """Warm up (tuning ``ε``, optionally ``M``), freeze, then sample."""
        n_steps = self.steps if steps is None else steps
        eps0 = self.step_size
        x = x_init.detach().clone()
        module = energy if isinstance(energy, nn.Module) else None
        with frozen_params(module), torch.enable_grad():
            if self.precondition:
                self._precond = None  # window 1: identity metric, tune ε + collect
                x, _, draws = self._dual_average(energy, x, self.warmup, eps0, collect=True)
                var = torch.cat(draws, dim=0).var(dim=0).clamp_min(1e-8)
                self._precond = var / var.log().mean().exp()  # geometric mean 1
                # window 2: fix M, RESTART dual averaging, re-tune ε
                x, eps, _ = self._dual_average(energy, x, self.warmup, eps0, collect=False)
            else:
                self._precond = None
                x, eps, _ = self._dual_average(energy, x, self.warmup, eps0, collect=False)
            self.step_size = eps  # freeze the averaged step size

            trajectory = [x.clone()] if return_trajectory else None
            for _ in range(n_steps):
                x, _ = self._transition(energy, x)
                x = x.detach()
                if trajectory is not None:
                    trajectory.append(x.clone())
        if trajectory is not None:
            return torch.stack(trajectory)
        return x
