"""Annealed Langevin dynamics over a noise ladder (Song & Ermon, 2019)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ebm._functional import validate_descending_sigmas
from ebm.energy import ConditionalEnergyFn
from ebm.samplers.base import Sampler
from ebm.utils import frozen_params


class AnnealedLangevinDynamics(Sampler):
    """NCSN Algorithm 1 for a noise-conditional energy ``energy(x, sigma)``.

    Runs Langevin at each level of a descending ``sigmas`` ladder, warm-starting
    each level from the last. Per level ``i``: ``alpha_i = step_size *
    sigma_i**2 / sigma_L**2`` and

        ``x <- x - (alpha_i / 2) * grad E(x, sigma_i) + sqrt(alpha_i) * z``

    (noise ``sqrt(alpha)``, matching the ``alpha/2`` drift coefficient — this
    is the "step λ/2, noise N(0, λ)" ULA parameterization).

    ``step_size`` is the step at the *final* (smallest) sigma; NCSN uses 2e-5
    for images in [0, 1]. Keep it below ``2 * sigma_min**2`` for stability.
    """

    def __init__(self, sigmas: Tensor, step_size: float = 2e-5, steps_per_sigma: int = 100):
        super().__init__(steps_per_sigma)
        self.sigmas = validate_descending_sigmas(sigmas, min_len=1)
        self.step_size = step_size

    # This sampler intentionally extends the base signature: it targets a
    # noise-conditional energy and carries a per-level sigma index.
    def step(  # type: ignore[override]
        self, energy: ConditionalEnergyFn, x: Tensor, sigma_index: int = -1
    ) -> Tensor:
        sigma = self.sigmas[sigma_index].to(x.device)
        alpha = self.step_size * (sigma / self.sigmas[-1]) ** 2

        def energy_at_sigma(z: Tensor) -> Tensor:
            return energy(z, sigma.expand(z.shape[0]))

        _, grad = self._energy_grad(energy_at_sigma, x)
        return x.detach() - (alpha / 2) * grad + torch.sqrt(alpha) * torch.randn_like(x)

    def sample(  # type: ignore[override]
        self,
        energy: ConditionalEnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        """Anneal through the full ladder, ``steps`` (default ``steps_per_sigma``)
        Langevin steps per level. Trajectories have length ``len(sigmas) * steps + 1``.
        """
        n_steps = self.steps if steps is None else steps
        x = x_init.detach().clone()
        trajectory = [x.clone()] if return_trajectory else None
        module = energy if isinstance(energy, nn.Module) else None
        with frozen_params(module), torch.enable_grad():
            for i in range(len(self.sigmas)):
                for _ in range(n_steps):
                    x = self.step(energy, x, sigma_index=i).detach()
                    if trajectory is not None:
                        trajectory.append(x.clone())
        if trajectory is not None:
            return torch.stack(trajectory)
        return x
