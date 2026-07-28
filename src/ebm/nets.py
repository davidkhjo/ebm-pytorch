"""Ready-made energy networks.

Design choices follow standard EBM practice: smooth activations (SiLU/Swish),
no batch normalization (it breaks per-sample energies and MCMC), and optional
spectral normalization for Lipschitz control (Du & Mordatch, 2019).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils.parametrizations import spectral_norm


def _maybe_sn(layer: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(layer) if enabled else layer


class MLPEnergy(nn.Module):
    """MLP energy function for vector data: ``(B, dim) -> (B,)``."""

    def __init__(
        self,
        dim: int,
        hidden: Sequence[int] = (128, 128),
        spectral_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim
        for h in hidden:
            layers += [_maybe_sn(nn.Linear(in_dim, h), spectral_norm), nn.SiLU()]
            in_dim = h
        layers.append(_maybe_sn(nn.Linear(in_dim, 1), spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


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

    The sigma embedding enters as a per-channel bias after the first
    convolution (FiLM-style, bias only).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 256),
        embed_dim: int = 32,
        spectral_norm: bool = True,
    ):
        super().__init__()
        self.embed = _GaussianFourierFeatures(embed_dim)
        self.embed_proj = nn.Linear(embed_dim, channels[0])
        self.stem = _maybe_sn(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=2, padding=1), spectral_norm
        )
        layers: list[nn.Module] = [nn.SiLU()]
        c_in = channels[0]
        for c_out in channels[1:]:
            conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
            layers += [_maybe_sn(conv, spectral_norm), nn.SiLU()]
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.head = _maybe_sn(nn.Linear(c_in, 1), spectral_norm)

    def forward(self, x: Tensor, sigma) -> Tensor:
        sigma = _as_batch_sigma(sigma, x)
        bias = self.embed_proj(self.embed(sigma))
        h = self.stem(x) + bias[:, :, None, None]
        h = self.body(h).mean(dim=(2, 3))
        return self.head(h).squeeze(-1)


class ConvEnergy(nn.Module):
    """Small strided CNN energy function for image data: ``(B, C, H, W) -> (B,)``.

    A reasonable baseline for 32x32 images; swap in your own architecture for
    serious image experiments.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256, 256),
        spectral_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
            layers += [_maybe_sn(conv, spectral_norm), nn.SiLU()]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = _maybe_sn(nn.Linear(c_in, 1), spectral_norm)

    def forward(self, x: Tensor) -> Tensor:
        h = self.features(x)
        h = h.mean(dim=(2, 3))
        return self.head(h).squeeze(-1)
