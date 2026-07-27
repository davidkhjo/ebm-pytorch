"""Plotting helpers for 2D energies and samples (requires the ``viz`` extra)."""

from __future__ import annotations

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
