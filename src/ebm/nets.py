"""Ready-made energy networks.

Design choices follow standard EBM practice: smooth activations (SiLU/Swish),
no batch normalization (it breaks per-sample energies and MCMC), and optional
spectral normalization for Lipschitz control (Du & Mordatch, 2019).
"""

from __future__ import annotations

from collections.abc import Sequence

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
