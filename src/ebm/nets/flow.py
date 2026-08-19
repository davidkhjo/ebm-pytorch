"""Normalizing-flow energies (RealNVP)."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ebm._functional import standard_normal_logprob


class _CouplingLayer(nn.Module):
    """One affine coupling: transforms the ``1-mask`` half conditioned on the ``mask`` half."""

    mask: Tensor

    def __init__(self, dim: int, hidden: int, mask: Tensor):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )

    def _scale_shift(self, conditioned: Tensor) -> tuple[Tensor, Tensor]:
        s, t = self.net(conditioned).chunk(2, dim=-1)
        keep = 1 - self.mask
        return torch.tanh(s) * keep, t * keep  # tanh-bounded log-scale for stability

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        xm = x * self.mask
        s, t = self._scale_shift(xm)
        y = xm + (1 - self.mask) * (x * torch.exp(s) + t)
        return y, s.sum(dim=-1)  # log|det J| = sum of the active log-scales

    def inverse(self, y: Tensor) -> Tensor:
        ym = y * self.mask
        s, t = self._scale_shift(ym)
        return ym + (1 - self.mask) * ((y - t) * torch.exp(-s))


class AffineCouplingFlow(nn.Module):
    """RealNVP affine-coupling normalizing flow, exposed as an exact-likelihood energy.

    A stack of affine coupling layers with alternating masks, giving a
    tractable-density model with an exact log-likelihood and exact sampling. The
    change of variables ``log p(x) = log N(f(x); 0, I) + log|det ∂f/∂x|`` is exact
    (no partition function), so this doubles as a *self-normalized energy*:
    ``forward(x) = -log_prob(x)`` is a valid `EnergyFn` with ``log Z = 0``,
    usable anywhere in the library, and it can serve as an exact base for an EBM.
    Operates on vector data ``(B, dim)``.
    """

    def __init__(self, dim: int, n_layers: int = 6, hidden: int = 64):
        super().__init__()
        if dim < 2:
            raise ValueError("AffineCouplingFlow needs dim >= 2 to split coordinates")
        self.dim = dim
        layers = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0  # alternate which half is conditioned
            layers.append(_CouplingLayer(dim, hidden, mask))
        self.layers = nn.ModuleList(layers)

    def transform(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Map data ``x`` to the base space, returning ``(z, log|det ∂z/∂x|)``."""
        logdet = x.new_zeros(x.shape[0])
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, z: Tensor) -> Tensor:
        """Map base samples ``z`` back to data space."""
        for layer in reversed(self.layers):
            assert isinstance(layer, _CouplingLayer)
            z = layer.inverse(z)
        return z

    def log_prob(self, x: Tensor) -> Tensor:
        z, logdet = self.transform(x)
        return standard_normal_logprob(z) + logdet

    def sample(self, n: int) -> Tensor:
        z = torch.randn(n, self.dim, device=next(self.parameters()).device)
        return self.inverse(z)

    def forward(self, x: Tensor) -> Tensor:
        return -self.log_prob(x)  # energy = -log p, so log Z = 0 exactly
