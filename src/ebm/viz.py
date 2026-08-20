"""Plotting helpers for 2D energies and samples (requires the ``viz`` extra)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ebm.energy import EnergyFn

__all__ = [
    "autocorrelation_plot",
    "energy_contour",
    "energy_histogram",
    "plot_samples",
    "rank_plot",
    "show_images",
    "trace_plot",
]


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for ebm.viz — install with `pip install ebmkit[viz]`"
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


@torch.no_grad()
def _chain_ranks(samples: Tensor, dim: int = 0) -> Tensor:
    """Rank each draw among the pooled draws of all chains; returns ``(n_chains, n_samples)``."""
    from ebm.eval.diagnostics import _as_chains

    x = _as_chains(samples)[..., dim]  # (M, N)
    m, n = x.shape
    order = x.reshape(-1).argsort()
    ranks = torch.empty(m * n, dtype=torch.float64)
    ranks[order] = torch.arange(1, m * n + 1, dtype=torch.float64)
    return ranks.reshape(m, n)


@torch.no_grad()
def autocorrelation_plot(samples: Tensor, max_lag: int = 40, dim: int = 0, ax=None):
    """Autocorrelation ``ρ̂_t`` vs lag for MCMC output, with a ``±1.96/√N`` band. Returns the axes.

    Input is ``(n_chains, n_samples[, dim])``. Bars decaying to inside the band
    within a few lags indicate good mixing; a slow decay flags autocorrelation.
    """
    from ebm.eval.diagnostics import autocorrelation

    plt = _require_matplotlib()
    acf = autocorrelation(samples, max_lag)[:, dim]
    n = samples.shape[1]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(acf)), acf.numpy(), width=0.8, color="#5c50c9")
    band = 1.96 / (n**0.5)
    for y in (band, -band):
        ax.axhline(y, ls="--", color="gray", lw=1)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    return ax


@torch.no_grad()
def rank_plot(samples: Tensor, dim: int = 0, bins: int = 20, ax=None):
    """Vehtari rank histogram per chain (uniform under good mixing). Returns the axes.

    Ranks every draw among the pooled draws of all chains, then histograms each
    chain's ranks. Well-mixed chains give flat, overlapping histograms near the
    dashed uniform line; a chain offset from the others shows a sloped or skewed
    histogram. Reference: Vehtari et al. (2021).
    """
    plt = _require_matplotlib()
    ranks = _chain_ranks(samples, dim)
    m, n = ranks.shape
    edges = torch.linspace(1, m * n, bins + 1).numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for i in range(m):
        ax.hist(ranks[i].numpy(), bins=edges, histtype="step", alpha=0.7)
    ax.axhline(n / bins, ls="--", color="gray", lw=1)  # uniform expectation per bin
    ax.set_xlabel("rank")
    ax.set_ylabel("count")
    return ax


@torch.no_grad()
def trace_plot(samples: Tensor, dim: int = 0, max_chains: int = 8, ax=None):
    """Trace of each chain's value over iterations (mixing at a glance). Returns the axes.

    Input is ``(n_chains, n_samples[, dim])``. Chains that overlap and wander
    across the same range have mixed; a chain stuck at a different level has not.
    """
    from ebm.eval.diagnostics import _as_chains

    plt = _require_matplotlib()
    x = _as_chains(samples)[..., dim]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    for i in range(min(x.shape[0], max_chains)):
        ax.plot(x[i].numpy(), alpha=0.7, lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"dim {dim}")
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
