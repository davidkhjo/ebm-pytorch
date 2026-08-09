"""Plotting helpers for 2D energies and samples (requires the ``viz`` extra)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ebm.energy import EnergyFn


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for ebm.viz — install with `pip install ebm-pytorch[viz]`"
        ) from err
    return plt


@torch.no_grad()
def energy_contour(
    energy: EnergyFn,
    bounds: tuple[float, float] = (-3.0, 3.0),
    resolution: int = 200,
    device: torch.device | str | None = None,
    ax=None,
    levels: int = 50,
):
    """Filled contour plot of a 2D energy landscape. Returns the axes."""
    plt = _require_matplotlib()
    lo, hi = bounds
    grid = torch.linspace(lo, hi, resolution)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    if device is not None:
        points = points.to(device)
    e = energy(points).detach().cpu().reshape(resolution, resolution)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.contourf(xx.numpy(), yy.numpy(), e.numpy(), levels=levels)
    ax.set_aspect("equal")
    return ax


def plot_samples(x: Tensor, ax=None, **scatter_kwargs):
    """Scatter plot of 2D samples. Returns the axes."""
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    x = x.detach().cpu()
    scatter_kwargs.setdefault("s", 3)
    scatter_kwargs.setdefault("alpha", 0.5)
    ax.scatter(x[:, 0], x[:, 1], **scatter_kwargs)
    ax.set_aspect("equal")
    return ax


@torch.no_grad()
def show_images(
    images: Tensor,
    *,
    nrow: int = 8,
    ax: Any = None,
    rescale: bool = True,
    title: str | None = None,
):
    """Tile a batch of images ``(N, C, H, W)`` into a single grid. Returns the axes.

    Handles the image-EBM convention directly: ``C == 1`` renders as inverted
    grayscale (dark strokes on white, like the MNIST examples), ``C == 3`` as
    RGB. With ``rescale`` (default) images are mapped from ``[-1, 1]`` to
    ``[0, 1]`` and clamped — matching `datasets.mnist` / `datasets.cifar10` and
    ``LangevinDynamics(clamp=(-1, 1))``. Pass ``rescale=False`` for images
    already in ``[0, 1]``.

    Args:
        images: ``(N, C, H, W)`` tensor with ``C`` in ``{1, 3}``.
        nrow: Number of images per row; rows are filled left-to-right.
        ax: Existing axes to draw on; a new figure is created if ``None``.
        rescale: Map ``[-1, 1] -> [0, 1]`` before display.
        title: Optional axes title.
    """
    plt = _require_matplotlib()
    x = images.detach().cpu().float()
    if x.dim() != 4 or x.shape[1] not in (1, 3):
        raise ValueError(f"expected (N, C, H, W) with C in {{1, 3}}, got {tuple(x.shape)}")
    if rescale:
        x = x * 0.5 + 0.5
    x = x.clamp(0, 1)

    n, c, h, w = x.shape
    ncol = nrow
    nrows = (n + ncol - 1) // ncol
    grid = torch.ones(c, nrows * h, ncol * w)  # white padding for the last row
    for i in range(n):
        r, col = divmod(i, ncol)
        grid[:, r * h : (r + 1) * h, col * w : (col + 1) * w] = x[i]

    if ax is None:
        _, ax = plt.subplots(figsize=(ncol, nrows))
    if c == 1:
        ax.imshow(grid[0], cmap="gray_r", vmin=0, vmax=1)
    else:
        ax.imshow(grid.permute(1, 2, 0))
    ax.axis("off")
    if title is not None:
        ax.set_title(title)
    return ax


def energy_histogram(energy: EnergyFn, batches: dict[str, Tensor], ax=None, bins: int = 60):
    """Overlaid energy histograms, e.g. ``{"data": x, "samples": x_neg}``.

    Overlapping data/sample histograms are the quickest sanity check that
    CD training is balanced.
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    with torch.no_grad():
        for label, x in batches.items():
            e = energy(x).detach().cpu().numpy()
            ax.hist(e, bins=bins, alpha=0.5, label=label, density=True)
    ax.set_xlabel("energy")
    ax.legend()
    return ax
