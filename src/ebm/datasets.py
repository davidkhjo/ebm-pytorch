"""Datasets: 2D toy distributions plus a torchvision-free MNIST loader.

The 2D functions return a float32 tensor of shape ``(n, 2)``, roughly
centered and on a scale of a few units, with an optional ``generator`` for
determinism. `mnist` and `fashion_mnist` download and parse the raw IDX
files directly.
"""

from __future__ import annotations

import gzip
import math
import urllib.request
from pathlib import Path

import torch
from torch import Tensor


def two_moons(
    n: int,
    noise: float = 0.1,
    generator: torch.Generator | None = None,
    return_labels: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Two interleaved moons; with ``return_labels`` also returns 0/1 moon labels."""
    n_upper = n // 2
    n_lower = n - n_upper
    theta_u = math.pi * torch.rand(n_upper, generator=generator)
    theta_l = math.pi * torch.rand(n_lower, generator=generator)
    upper = torch.stack([torch.cos(theta_u), torch.sin(theta_u)], dim=1)
    lower = torch.stack([1 - torch.cos(theta_l), 0.5 - torch.sin(theta_l)], dim=1)
    x = torch.cat([upper, lower]) + noise * torch.randn(n, 2, generator=generator)
    x = x - torch.tensor([0.5, 0.25])
    y = torch.cat([torch.zeros(n_upper, dtype=torch.long), torch.ones(n_lower, dtype=torch.long)])
    perm = torch.randperm(n, generator=generator)
    x = x[perm].float()
    if return_labels:
        return x, y[perm]
    return x


_MNIST_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FASHION_MIRROR = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"


def _parse_idx(raw: bytes) -> Tensor:
    """Parse an IDX-format buffer (the raw MNIST file format) into a uint8 tensor."""
    if raw[:2] != b"\x00\x00" or raw[2] != 0x08:
        raise ValueError("not an unsigned-byte IDX buffer")
    n_dims = raw[3]
    shape = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(n_dims)]
    offset = 4 + 4 * n_dims
    data = torch.frombuffer(bytearray(raw[offset:]), dtype=torch.uint8)
    return data.reshape(shape)


def _load_mnist_like(
    mirror: str,
    cache_prefix: str,
    root: str | Path | None,
    train: bool,
    return_labels: bool,
) -> Tensor | tuple[Tensor, Tensor]:
    root = Path(root).expanduser() if root is not None else Path.home() / ".cache" / "ebm-pytorch"
    root.mkdir(parents=True, exist_ok=True)
    prefix = "train" if train else "t10k"

    tensors = []
    for kind in ("images-idx3", "labels-idx1"):
        name = f"{prefix}-{kind}-ubyte.gz"
        path = root / (cache_prefix + name)
        if not path.exists():
            urllib.request.urlretrieve(mirror + name, path)  # noqa: S310
        tensors.append(_parse_idx(gzip.decompress(path.read_bytes())))

    images, labels = tensors
    x = images.unsqueeze(1).float() / 127.5 - 1.0
    if return_labels:
        return x, labels.long()
    return x


def mnist(
    root: str | Path | None = None,
    train: bool = True,
    return_labels: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """MNIST digits as ``(N, 1, 28, 28)`` float32 in ``[-1, 1]`` — no torchvision.

    Downloads the raw IDX files (~11 MB) on first use into ``root`` (default
    ``~/.cache/ebm-pytorch``) and parses them directly. The ``[-1, 1]`` range
    pairs with ``LangevinDynamics(clamp=(-1, 1))`` and the image-EBM recipes.

    Args:
        root: Cache directory.
        train: Training split (60k) or test split (10k).
        return_labels: Also return ``(N,)`` int64 labels.
    """
    return _load_mnist_like(_MNIST_MIRROR, "", root, train, return_labels)


def fashion_mnist(
    root: str | Path | None = None,
    train: bool = True,
    return_labels: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Fashion-MNIST as ``(N, 1, 28, 28)`` float32 in ``[-1, 1]``.

    Same IDX format, shapes, and scaling as `mnist` (cached alongside it with a
    ``fashion-`` prefix). Its classic role in the EBM literature is as the
    out-of-distribution counterpart to MNIST when evaluating a hybrid model's
    ``eval.ood_auroc`` (Grathwohl et al., 2020).
    """
    return _load_mnist_like(_FASHION_MIRROR, "fashion-", root, train, return_labels)


def eight_gaussians(n: int, std: float = 0.15, generator: torch.Generator | None = None) -> Tensor:
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = 2.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    idx = torch.randint(8, (n,), generator=generator)
    return (centers[idx] + std * torch.randn(n, 2, generator=generator)).float()


def checkerboard(n: int, generator: torch.Generator | None = None) -> Tensor:
    x1 = 4 * torch.rand(n, generator=generator) - 2
    offset = 2 * torch.randint(2, (n,), generator=generator).float()
    x2 = torch.rand(n, generator=generator) + torch.floor(x1) % 2 + offset - 2
    return torch.stack([x1, x2], dim=1).float()


def rings(n: int, noise: float = 0.05, generator: torch.Generator | None = None) -> Tensor:
    radii = torch.tensor([0.7, 1.4, 2.1])
    idx = torch.randint(3, (n,), generator=generator)
    theta = 2 * math.pi * torch.rand(n, generator=generator)
    r = radii[idx] + noise * torch.randn(n, generator=generator)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1).float()


def spirals(n: int, noise: float = 0.08, generator: torch.Generator | None = None) -> Tensor:
    n_a = n // 2
    n_b = n - n_a
    t_a = torch.sqrt(torch.rand(n_a, generator=generator)) * 3 * math.pi
    t_b = torch.sqrt(torch.rand(n_b, generator=generator)) * 3 * math.pi
    arm_a = torch.stack([t_a * torch.cos(t_a), t_a * torch.sin(t_a)], dim=1) / (1.5 * math.pi)
    arm_b = -torch.stack([t_b * torch.cos(t_b), t_b * torch.sin(t_b)], dim=1) / (1.5 * math.pi)
    x = torch.cat([arm_a, arm_b]) + noise * torch.randn(n, 2, generator=generator)
    return x[torch.randperm(n, generator=generator)].float()
