"""MLP, conv, and residual energy networks."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
from torch.nn import functional as F

from ebm.nets._common import _maybe_sn


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


class ConvClassifier(nn.Module):
    """Small strided CNN classifier ``(B, C, H, W) -> (B, K)`` for image JEM.

    The same trunk as `ConvEnergy` with a flattened (position-sensitive)
    K-logit head — wrap it in ``ebm.ClassifierEnergy`` to get the marginal
    energy ``E(x) = -logsumexp(logits)`` and class-conditional energies. No
    batch normalization, so per-sample energies and MCMC stay well-defined.

    The head is deliberately *not* pooled: a mean-pooled logit is a
    translation-invariant "bag of class evidence", and minimizing that
    conditional energy prefers a canvas tiled with class-typical strokes over
    a single centered digit. ``image_size`` fixes the flattened dimension.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        image_size: int = 32,
        channels: Sequence[int] = (64, 128, 256, 256),
        spectral_norm: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        size = image_size
        for c_out in channels:
            conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1)
            layers += [_maybe_sn(conv, spectral_norm), nn.SiLU()]
            c_in = c_out
            size = (size + 1) // 2
        self.features = nn.Sequential(*layers)
        self.head = _maybe_sn(nn.Linear(c_in * size * size, num_classes), spectral_norm)

    def forward(self, x: Tensor) -> Tensor:
        h = self.features(x)
        return self.head(h.flatten(1))


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


class _ResBlock(nn.Module):
    """Pre-activation residual block; the second conv is zero-initialized so the
    block is the identity at init (unless spectral norm reparametrizes it)."""

    def __init__(
        self, c_in: int, c_out: int, *, spectral_norm: bool = False, downsample: bool = False
    ):
        super().__init__()
        self.act = nn.SiLU()
        self.conv1 = _maybe_sn(nn.Conv2d(c_in, c_out, 3, padding=1), spectral_norm)
        conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        if not spectral_norm:  # zero-init is incompatible with the SN parametrization
            nn.init.zeros_(conv2.weight)
            assert conv2.bias is not None
            nn.init.zeros_(conv2.bias)
        self.conv2 = _maybe_sn(conv2, spectral_norm)
        self.skip: nn.Module = (
            nn.Identity() if c_in == c_out else _maybe_sn(nn.Conv2d(c_in, c_out, 1), spectral_norm)
        )
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        h = self.conv1(self.act(x))
        h = self.conv2(self.act(h))
        out = self.skip(x) + h
        if self.downsample:
            out = F.avg_pool2d(out, 2)
        return out


class ResNetEnergy(nn.Module):
    """Residual CNN energy for images ``(B, C, H, W) -> (B,)`` (IGEBM / Improved-CD).

    The standard recipe for image EBMs: a stack of pre-activation residual blocks
    (SiLU, 3x3 convs, no normalization), average-pool downsampling between
    stages, global-pool then a linear head. The second conv of each block is
    zero-initialized so every block starts as the identity — a stable init for
    the long MCMC chains contrastive divergence needs. ``spectral_norm=True``
    wraps every conv and the head for Lipschitz control (Du & Mordatch, 2019).
    Works at any spatial size (the global pool removes the size dependence).
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256),
        spectral_norm: bool = False,
    ):
        super().__init__()
        self.stem = _maybe_sn(nn.Conv2d(in_channels, channels[0], 3, padding=1), spectral_norm)
        blocks: list[nn.Module] = []
        c_in = channels[0]
        for c_out in channels:
            blocks.append(_ResBlock(c_in, c_out, spectral_norm=spectral_norm, downsample=True))
            c_in = c_out
        self.blocks = nn.ModuleList(blocks)
        self.act = nn.SiLU()
        self.head = _maybe_sn(nn.Linear(c_in, 1), spectral_norm)

    def forward(self, x: Tensor) -> Tensor:
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        h = self.act(h).mean(dim=(2, 3))  # global average pool
        return self.head(h).squeeze(-1)
