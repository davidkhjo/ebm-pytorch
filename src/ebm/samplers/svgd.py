"""Stein variational gradient descent — deterministic particle transport."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler


class SVGD(Sampler):
    """Stein variational gradient descent (Liu & Wang, 2016).

    A deterministic sampler that transports a set of **interacting** particles
    toward ``p ∝ exp(-E)`` by steepest descent of the KL divergence in an RBF
    RKHS. Each update mixes a score-driven attraction with a kernel repulsion
    that keeps the particles spread out::

        φ(xᵢ) = (1/n) Σⱼ [ k(xⱼ, xᵢ)·s(xⱼ) + ∇_{xⱼ} k(xⱼ, xᵢ) ] ,   s = -∇E
        xᵢ ← xᵢ + ε·φ(xᵢ)

    with the RBF kernel ``k(x,y)=exp(-‖x-y‖²/2h²)`` and the median-heuristic
    bandwidth ``h² = median(pairwise ‖·‖²)/log n``.

    Unlike the MCMC samplers, the batch dimension is the **particle count** and
    the particles are *not* independent draws — they interact, and the transport
    is deterministic given ``x_init``. Use a few hundred particles.

    Args:
        step_size: ε, the transport step.
        steps: number of updates per ``sample`` call.
        bandwidth: fixed RBF bandwidth; ``None`` uses the per-step median heuristic.
    """

    def __init__(self, step_size: float = 0.3, steps: int = 1000, bandwidth: float | None = None):
        super().__init__(steps)
        self.step_size = step_size
        self.bandwidth = bandwidth

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        n = x.shape[0]
        _, grad = self._energy_grad(energy, x)
        s = (-grad).reshape(n, -1)  # score, (n, d)
        xf = x.reshape(n, -1)

        d2 = torch.cdist(xf, xf).pow(2)  # (n, n)
        if self.bandwidth is not None:
            h2 = self.bandwidth**2
        else:
            off = d2[~torch.eye(n, dtype=torch.bool, device=x.device)]
            h2 = float((off.median() / math.log(max(n, 2))).clamp_min(1e-8))
        k = torch.exp(-d2 / (2 * h2))  # (n, n), symmetric

        drive = k @ s / n
        repel = (k.sum(dim=1, keepdim=True) * xf - k @ xf) / h2 / n
        return (xf + self.step_size * (drive + repel)).reshape_as(x)
