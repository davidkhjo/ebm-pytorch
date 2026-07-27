"""Energy function protocol and helpers.

The library-wide convention is ``p(x) ∝ exp(-E(x))``: *low* energy means
*high* probability. An energy function is any callable mapping a batch
``(B, *event_shape)`` to a vector of energies ``(B,)`` — a plain function,
a lambda, or an ``nn.Module``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

EnergyFn = Callable[[Tensor], Tensor]
"""Any callable mapping ``(B, *event_shape)`` to per-sample energies ``(B,)``."""


def score(energy: EnergyFn, x: Tensor, *, create_graph: bool = False) -> Tensor:
    """Score function ``∇_x log p(x) = -∇_x E(x)``.

    Args:
        energy: Energy function following the ``(B, *shape) -> (B,)`` convention.
        x: Input batch. Does not need ``requires_grad``.
        create_graph: Set ``True`` when the score itself appears in a loss that
            is backpropagated (score matching); leave ``False`` inside samplers.
    """
    if not x.requires_grad:
        x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        e = energy(x)
        (grad,) = torch.autograd.grad(e.sum(), x, create_graph=create_graph)
    return -grad


class EnergyModel(nn.Module):
    """Optional convenience wrapper around an energy network.

    Guarantees the ``(B,) `` output shape and exposes ``score``. Any plain
    ``nn.Module`` (or function) with the right shape works everywhere in the
    library — this wrapper is sugar, not a requirement.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: Tensor) -> Tensor:
        e = self.net(x)
        if e.dim() > 1:
            e = e.reshape(x.shape[0])
        return e

    def score(self, x: Tensor, *, create_graph: bool = False) -> Tensor:
        return score(self, x, create_graph=create_graph)
