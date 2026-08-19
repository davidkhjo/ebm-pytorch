"""Shared numeric helpers (private).

Small tensor free-functions that were duplicated across samplers, losses, and
eval. Kept private (leading underscore module) so the public API is unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor


def flat_sum(x: Tensor) -> Tensor:
    """Sum over all non-batch dimensions, returning ``(B,)``."""
    return x.reshape(x.shape[0], -1).sum(dim=1)


def mh_accept(x: Tensor, proposal: Tensor, accept: Tensor) -> Tensor:
    """Metropolis select: keep ``proposal`` where ``accept`` (a ``(B,)`` bool), else ``x``.

    Callers set ``self._last_accept = accept.float().mean()`` themselves (kept as a
    device tensor, materialized lazily) before calling this.
    """
    mask = accept.reshape(-1, *([1] * (x.dim() - 1)))
    return torch.where(mask, proposal, x)


def validate_descending_sigmas(sigmas: Tensor, *, min_len: int = 2) -> Tensor:
    """Coerce to a float32 1D noise ladder, strictly decreasing (largest first)."""
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
    if sigmas.dim() != 1 or len(sigmas) < min_len:
        raise ValueError(f"sigmas must be a 1D tensor with at least {min_len} level(s)")
    if (sigmas <= 0).any() or (sigmas.diff() >= 0).any():
        raise ValueError("sigmas must be positive and strictly decreasing (largest first)")
    return sigmas


def validate_positive_sigmas(sigmas: Tensor) -> Tensor:
    """Coerce to a float32 1D tensor of positive noise levels (order-agnostic)."""
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
    if sigmas.dim() != 1 or (sigmas <= 0).any():
        raise ValueError("sigmas must be a 1D tensor of positive values")
    return sigmas


def median_sq_dist(d2: Tensor) -> Tensor:
    """Median of the off-diagonal entries of an ``(n, n)`` squared-distance matrix."""
    n = d2.shape[0]
    return d2[~torch.eye(n, dtype=torch.bool, device=d2.device)].median()


def rbf_bandwidth(d2: Tensor, bandwidth: float | None) -> float:
    """RBF bandwidth: the median heuristic (``sqrt`` of the median squared distance)
    when ``bandwidth`` is ``None``; raises if it (or an explicit value) is ``<= 0``."""
    if bandwidth is None:
        bandwidth = float(median_sq_dist(d2).sqrt())
    if bandwidth <= 0:
        raise ValueError(
            "RBF bandwidth is 0 (the median pairwise distance vanished — the "
            "samples are nearly identical); pass an explicit positive bandwidth"
        )
    return bandwidth
