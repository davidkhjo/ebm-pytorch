"""Analytic test-target energies (closed-form ground truth)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class FunnelEnergy(nn.Module):
    """Neal's funnel — the canonical MCMC stress test, as an energy ``(B, D) -> (B,)``.

    Coordinate 0 is the log-scale ``v ~ N(0, v_scale²)``; the remaining
    ``n = D - 1`` "neck" coordinates are ``x_i | v ~ N(0, e^v)``. The negative
    log density (up to a constant) is

    ``E(x) = ½[ v² / v_scale² + e^{-v} ‖x_neck‖² + n·v ]``.

    The geometry is deliberately vicious: at negative ``v`` the neck collapses to
    a needle no fixed step size can navigate, so single-scale samplers fail and
    the true marginals (``v ~ N(0, v_scale²)``, ``x_i | v ~ N(0, e^v)``) are the
    ground truth to check against. Differentiable, so it drives any gradient
    sampler. Reference: Neal (2003), "Slice Sampling".
    """

    def __init__(self, dim: int = 2, v_scale: float = 3.0):
        super().__init__()
        if dim < 2:
            raise ValueError("funnel needs dim >= 2 (one scale coordinate + a neck)")
        if v_scale <= 0:
            raise ValueError("v_scale must be positive")
        self.dim = dim
        self.v_scale = v_scale

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.dim:
            raise ValueError(f"expected (B, {self.dim}), got {tuple(x.shape)}")
        v = x[:, 0]
        neck = x[:, 1:]
        n = neck.shape[1]
        neck_sq = neck.pow(2).sum(dim=1)
        return 0.5 * (v.pow(2) / self.v_scale**2 + torch.exp(-v) * neck_sq + n * v)


class GaussianMixtureEnergy(nn.Module):
    """Isotropic Gaussian-mixture energy with known modes ``(B, D) -> (B,)``.

    ``E(x) = -logsumexp_k( log w_k - ‖x - μ_k‖² / (2 σ²) )`` — the (unnormalized)
    energy of ``p(x) ∝ Σ_k w_k N(x; μ_k, σ² I)``. A closed-form multimodal target:
    with well-separated means and equal weights, a correct sampler must visit the
    modes in proportion to ``w_k``, which single-chain gradient MCMC cannot do
    across high barriers — exactly what `ParallelTempering` is for.
    """

    means: Tensor
    log_weights: Tensor

    def __init__(self, means: Tensor, weights: Sequence[float] | None = None, std: float = 1.0):
        super().__init__()
        means = torch.as_tensor(means, dtype=torch.float32)
        if means.dim() != 2:
            raise ValueError("means must be (K, D)")
        if std <= 0:
            raise ValueError("std must be positive")
        k = means.shape[0]
        if weights is None:
            weights = [1.0] * k
        if len(weights) != k or any(w <= 0 for w in weights):
            raise ValueError("need one positive weight per mode")
        self.std = std
        self.register_buffer("means", means)
        self.register_buffer("log_weights", torch.log(torch.tensor([float(w) for w in weights])))

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != self.means.shape[1]:
            raise ValueError(f"expected (B, {self.means.shape[1]}), got {tuple(x.shape)}")
        sq = (x[:, None, :] - self.means[None, :, :]).pow(2).sum(dim=2)
        log_comp = self.log_weights[None, :] - sq / (2 * self.std**2)
        return -torch.logsumexp(log_comp, dim=1)


class BananaEnergy(nn.Module):
    """Curved "banana" (twisted-Gaussian) energy ``(B, 2) -> (B,)`` — Haario et al. 1999.

    A Gaussian bent along a parabola:
    ``E(x) = x₀²/(2σ₀²) + (x₁ - b(x₀² - σ₀²))²/(2σ₁²)``. The strong x₀–x₁ curvature
    defeats isotropic step sizes, so it is a standard MCMC stress test with a
    different failure mode from `FunnelEnergy` (varying scale) and
    `GaussianMixtureEnergy` (barriers). Its unique strength as a test target:
    exact i.i.d. samples are available in closed form via `exact_sample`, so a
    sampler's output can be checked directly.
    """

    def __init__(self, b: float = 0.5, sigma: tuple[float, float] = (1.0, 1.0)):
        super().__init__()
        if sigma[0] <= 0 or sigma[1] <= 0:
            raise ValueError("sigma entries must be positive")
        self.b = b
        self.sigma = sigma

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2 or x.shape[1] != 2:
            raise ValueError(f"expected (B, 2), got {tuple(x.shape)}")
        s0, s1 = self.sigma
        warp = x[:, 1] - self.b * (x[:, 0] ** 2 - s0**2)
        return x[:, 0] ** 2 / (2 * s0**2) + warp**2 / (2 * s1**2)

    def exact_sample(self, n: int, *, generator: torch.Generator | None = None) -> Tensor:
        """Exact i.i.d. draws: sample a Gaussian, then bend it along the parabola."""
        s0, s1 = self.sigma
        u = torch.randn(n, 2, generator=generator) * torch.tensor([s0, s1])
        x = u.clone()
        x[:, 1] = u[:, 1] + self.b * (u[:, 0] ** 2 - s0**2)
        return x
