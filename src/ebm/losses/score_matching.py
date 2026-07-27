"""Score matching losses: denoising (Vincent 2011) and sliced (Song et al. 2019).

Both train the model score ``s(x) = -∇_x E(x)`` without any MCMC sampling.
Unlike CD, these losses backpropagate *through* the score, so the inner
gradient is taken with ``create_graph=True``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ebm.energy import EnergyFn, score
from ebm.losses.base import LossOutput


def _flat_sum(x: Tensor) -> Tensor:
    return x.reshape(x.shape[0], -1).sum(dim=1)


def geometric_sigmas(sigma_max: float, sigma_min: float, n: int) -> Tensor:
    """Descending geometric noise ladder from ``sigma_max`` to ``sigma_min``.

    The standard NCSN schedule; pass to ``MultiSigmaDenoisingScoreMatching``
    and ``AnnealedLangevinDynamics``.
    """
    if not (sigma_max > sigma_min > 0):
        raise ValueError("need sigma_max > sigma_min > 0")
    return torch.logspace(math.log10(sigma_max), math.log10(sigma_min), n)


class DenoisingScoreMatching(nn.Module):
    """Single-noise-level DSM: match the score of the ``σ``-smoothed data.

    ``L = 1/2 E[ ‖s(x + σε) + ε/σ‖² ]`` with ``ε ~ N(0, I)``. The learned
    model approximates the data distribution convolved with ``N(0, σ² I)``, so
    choose ``σ`` small relative to the data scale.
    """

    def __init__(self, sigma: float = 0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, energy: EnergyFn, x: Tensor) -> LossOutput:
        noise = torch.randn_like(x)
        x_noisy = x + self.sigma * noise
        s = score(energy, x_noisy, create_graph=True)
        target = -noise / self.sigma
        loss = 0.5 * _flat_sum((s - target).pow(2)).mean()
        return LossOutput(loss=loss, metrics={"loss": loss.item()})


class MultiSigmaDenoisingScoreMatching(nn.Module):
    """NCSN-style DSM across a noise ladder (Song & Ermon, 2019).

    Expects a *noise-conditional* energy ``energy(x, sigma) -> (B,)`` (see
    ``ebm.nets.NoiseConditionalMLPEnergy``). Each sample gets a random level
    from ``sigmas``; the per-level losses are combined with the standard
    ``sigma**2`` weighting, written in its stable form
    ``L = 1/2 E‖sigma * s(x + sigma*eps, sigma) + eps‖²``.

    Sample from the trained model with ``AnnealedLangevinDynamics`` over the
    same ladder.
    """

    def __init__(self, sigmas: Tensor):
        super().__init__()
        sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
        if sigmas.dim() != 1 or (sigmas <= 0).any():
            raise ValueError("sigmas must be a 1D tensor of positive values")
        self.register_buffer("sigmas", sigmas)

    def forward(self, energy, x: Tensor) -> LossOutput:
        batch_size = x.shape[0]
        idx = torch.randint(len(self.sigmas), (batch_size,), device=x.device)
        sigma = self.sigmas[idx]
        sigma_b = sigma.reshape(batch_size, *([1] * (x.dim() - 1)))

        noise = torch.randn_like(x)
        x_noisy = (x + sigma_b * noise).detach().requires_grad_(True)
        e = energy(x_noisy, sigma)
        (grad_e,) = torch.autograd.grad(e.sum(), x_noisy, create_graph=True)
        s = -grad_e

        loss = 0.5 * _flat_sum((sigma_b * s + noise).pow(2)).mean()
        return LossOutput(loss=loss, metrics={"loss": loss.item()})


class SlicedScoreMatching(nn.Module):
    """Sliced score matching with random projections (Hutchinson-style).

    ``L = E_v[ vᵀ ∇_x s(x) v ] + (1/2) E[(vᵀ s)²]``; the variance-reduced
    variant (``vr=True``, recommended) replaces the second term with
    ``(1/2)‖s‖²``. Costs one extra backprop per projection but needs no noise
    scale and no sampler.
    """

    def __init__(self, n_projections: int = 1, vr: bool = True):
        super().__init__()
        self.n_projections = n_projections
        self.vr = vr

    def forward(self, energy: EnergyFn, x: Tensor) -> LossOutput:
        x = x.detach().requires_grad_(True)
        e = energy(x)
        (grad_e,) = torch.autograd.grad(e.sum(), x, create_graph=True)
        s = -grad_e

        total = x.new_zeros(x.shape[0])
        for _ in range(self.n_projections):
            v = torch.randn_like(x)
            sv = _flat_sum(s * v)
            (jac_v,) = torch.autograd.grad(sv.sum(), x, create_graph=True)
            trace_term = _flat_sum(jac_v * v)
            norm_term = 0.5 * _flat_sum(s.pow(2)) if self.vr else 0.5 * sv.pow(2)
            total = total + trace_term + norm_term

        loss = (total / self.n_projections).mean()
        return LossOutput(loss=loss, metrics={"loss": loss.item()})
