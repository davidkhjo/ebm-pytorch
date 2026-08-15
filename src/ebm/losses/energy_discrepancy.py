"""Energy discrepancy: a sampler-free, score-free EBM loss (Schröder et al. 2023)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ebm.energy import EnergyFn
from ebm.losses.base import LossOutput


class EnergyDiscrepancy(nn.Module):
    """Energy discrepancy loss — no MCMC, no score (Schröder et al., NeurIPS 2023).

    Perturbs each data point with Gaussian noise and contrasts its energy against
    ``M`` twice-perturbed copies::

        L = mean_i logsumexp_j( [ E(x_i) - E(x_i + √t·ξ_i + √t·ξ'_ij) ]  ++  [log w] )

    where ``ξ_i`` is shared across the ``M`` contrastive draws for point ``i`` and
    ``ξ'_ij`` is drawn independently. The appended ``log w`` column is the
    ``w``-stabilization: it lower-bounds the log and keeps the estimator from
    diverging at small ``t``. Cost is ``M + 1`` forward energy evaluations — no
    sampler to tune, no ``create_graph``, no divergence signatures.

    Args:
        sigma: perturbation scale ``√t`` (the noise std). Scale to the data.
        m_particles: number of contrastive draws ``M`` per point.
        w_stable: stabilizer ``w`` (keep ``> 0``; ``w=0`` can diverge).
    """

    def __init__(self, sigma: float = 1.0, m_particles: int = 4, w_stable: float = 1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if m_particles < 1:
            raise ValueError("m_particles must be >= 1")
        if w_stable < 0:
            raise ValueError("w_stable must be >= 0")
        self.sigma = sigma
        self.m_particles = m_particles
        self.w_stable = w_stable

    def forward(self, energy: EnergyFn, x: Tensor) -> LossOutput:
        b = x.shape[0]
        event = x.shape[1:]
        xi = torch.randn_like(x) * self.sigma  # (B, *event), shared per point
        xi_prime = torch.randn(b, self.m_particles, *event, device=x.device, dtype=x.dtype)
        pert = x.unsqueeze(1) + xi.unsqueeze(1) + self.sigma * xi_prime  # (B, M, *event)

        pos = energy(x)  # (B,)
        neg = energy(pert.reshape(b * self.m_particles, *event)).reshape(b, self.m_particles)
        val = pos.unsqueeze(1) - neg  # (B, M)
        if self.w_stable > 0:
            logw = val.new_full((b, 1), math.log(self.w_stable))
            val = torch.cat([val, logw], dim=1)  # (B, M+1)
        loss = val.logsumexp(dim=1).mean()

        x_neg = pert.reshape(b * self.m_particles, *event).detach()
        return LossOutput(loss=loss, metrics={"loss": loss.item()}, x_neg=x_neg)
