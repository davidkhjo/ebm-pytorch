"""Noise-conditional (sigma-conditioned) energy networks for score-based models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ebm.nets._common import _maybe_sn


def _as_batch_sigma(sigma, x: Tensor) -> Tensor:
    """Normalize float / 0-dim / ``(B,)`` sigma to a ``(B,)`` tensor on ``x``'s device."""
    if not isinstance(sigma, Tensor):
        sigma = torch.tensor(float(sigma))
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    if sigma.dim() == 0:
        sigma = sigma.expand(x.shape[0])
    return sigma


class _GaussianFourierFeatures(nn.Module):
    """Random Fourier embedding of ``log sigma`` (Song et al., score-SDE style)."""

    freqs: Tensor

    def __init__(self, embed_dim: int = 32, scale: float = 1.0):
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError("embed_dim must be even")
        self.register_buffer("freqs", torch.randn(embed_dim // 2) * scale)

    def forward(self, sigma: Tensor) -> Tensor:
        angles = 2 * torch.pi * torch.log(sigma)[:, None] * self.freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=1)


class NoiseConditionalMLPEnergy(nn.Module):
    """Noise-conditional MLP energy ``(x: (B, dim), sigma: (B,)) -> (B,)``.

    The sigma embedding is concatenated to ``x`` at the input layer. Pairs with
    ``MultiSigmaDenoisingScoreMatching`` and ``AnnealedLangevinDynamics``.
    """

    def __init__(
        self,
        dim: int,
        hidden: Sequence[int] = (128, 128),
        embed_dim: int = 32,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.embed = _GaussianFourierFeatures(embed_dim)
        layers: list[nn.Module] = []
        in_dim = dim + embed_dim
        for h in hidden:
            layers += [_maybe_sn(nn.Linear(in_dim, h), spectral_norm), nn.SiLU()]
            in_dim = h
        layers.append(_maybe_sn(nn.Linear(in_dim, 1), spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, sigma) -> Tensor:
        sigma = _as_batch_sigma(sigma, x)
        h = torch.cat([x, self.embed(sigma)], dim=1)
        return self.net(h).squeeze(-1)


class NoiseConditionalConvEnergy(nn.Module):
    """Noise-conditional CNN energy ``(x: (B, C, H, W), sigma: (B,)) -> (B,)``.

    The sigma embedding modulates every block with FiLM (per-channel scale and
    bias) — bias-only conditioning is too weak when the noise ladder spans a
    wide variance range, as in recovery-likelihood training.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 256),
        embed_dim: int = 32,
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.embed = _GaussianFourierFeatures(embed_dim)
        self.convs = nn.ModuleList()
        self.films = nn.ModuleList()
        c_in = in_channels
        for c_out in channels:
            conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
            self.convs.append(_maybe_sn(conv, spectral_norm))
            film = nn.Linear(embed_dim, 2 * c_out)
            nn.init.zeros_(film.weight)
            nn.init.zeros_(film.bias)
            self.films.append(film)
            c_in = c_out
        self.act = nn.SiLU()
        self.head = _maybe_sn(nn.Linear(c_in, 1), spectral_norm)

    def forward(self, x: Tensor, sigma) -> Tensor:
        sigma = _as_batch_sigma(sigma, x)
        emb = self.embed(sigma)
        h = x
        for conv, film in zip(self.convs, self.films, strict=True):
            h = conv(h)
            scale, bias = film(emb).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + bias[:, :, None, None]
            h = self.act(h)
        return self.head(h.mean(dim=(2, 3))).squeeze(-1)
