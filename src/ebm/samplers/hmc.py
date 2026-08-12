"""Hamiltonian Monte Carlo with leapfrog integration."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler


def _flat_sum(x: Tensor) -> Tensor:
    return x.reshape(x.shape[0], -1).sum(dim=1)


class HMC(Sampler):
    """Hamiltonian Monte Carlo over ``p ∝ exp(-E)``.

    One ``step`` = one full trajectory (momentum resample, ``leapfrog_steps``
    leapfrog updates, Metropolis accept/reject). Best suited to smooth,
    low-dimensional energies; ``last_accept_rate`` reports the latest
    acceptance rate — tune ``step_size`` toward ~0.6–0.9.
    """

    def __init__(
        self,
        step_size: float = 0.1,
        leapfrog_steps: int = 10,
        steps: int = 50,
    ):
        super().__init__(steps)
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        eps = self.step_size
        x0 = x.detach()
        p0 = torch.randn_like(x0)

        e0, grad = self._energy_grad(energy, x0)
        h0 = e0 + 0.5 * _flat_sum(p0.pow(2))

        # Leapfrog: half momentum step, full position steps, half momentum step.
        x_new = x0
        p = p0 - 0.5 * eps * grad
        for i in range(self.leapfrog_steps):
            x_new = x_new + eps * p
            e_new, grad = self._energy_grad(energy, x_new)
            p = p - (0.5 if i == self.leapfrog_steps - 1 else 1.0) * eps * grad

        h1 = e_new + 0.5 * _flat_sum(p.pow(2))
        log_alpha = h0 - h1
        accept = torch.log(torch.rand_like(log_alpha)) < log_alpha
        self._last_accept = accept.float().mean()
        accept = accept.reshape(-1, *([1] * (x.dim() - 1)))
        return torch.where(accept, x_new.detach(), x0)
