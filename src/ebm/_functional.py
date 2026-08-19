"""Shared numeric helpers (private).

Small tensor free-functions that were duplicated across samplers, losses, and
eval. Kept private (leading underscore module) so the public API is unchanged.
"""

from __future__ import annotations

from torch import Tensor


def flat_sum(x: Tensor) -> Tensor:
    """Sum over all non-batch dimensions, returning ``(B,)``."""
    return x.reshape(x.shape[0], -1).sum(dim=1)
