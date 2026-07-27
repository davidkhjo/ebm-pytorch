"""Sampler base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from ebm.energy import EnergyFn
from ebm.utils import frozen_params


class Sampler(ABC):
    """Base class for MCMC samplers over an energy function.

    Subclasses implement ``step``; ``sample`` runs the chain, freezing the
    energy network's parameters (gradients are taken w.r.t. samples only) and
    returning detached samples.
    """

    def __init__(self, steps: int):
        self.steps = steps

    @abstractmethod
    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        """Advance every chain in the batch by one transition."""

    def sample(
        self,
        energy: EnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        """Run the chain from ``x_init`` for ``steps`` transitions.

        Returns detached samples ``(B, *shape)``, or the full trajectory
        ``(steps + 1, B, *shape)`` if ``return_trajectory`` is set.
        """
        n_steps = self.steps if steps is None else steps
        x = x_init.detach().clone()
        trajectory = [x.clone()] if return_trajectory else None
        module = energy if isinstance(energy, nn.Module) else None
        with frozen_params(module), torch.enable_grad():
            for _ in range(n_steps):
                x = self.step(energy, x).detach()
                if trajectory is not None:
                    trajectory.append(x.clone())
        if trajectory is not None:
            return torch.stack(trajectory)
        return x

    @staticmethod
    def _energy_grad(energy: EnergyFn, x: Tensor) -> tuple[Tensor, Tensor]:
        """Per-sample energies and their gradient w.r.t. ``x`` (no graph kept)."""
        x = x.detach().requires_grad_(True)
        e = energy(x)
        (grad,) = torch.autograd.grad(e.sum(), x)
        return e.detach(), grad
