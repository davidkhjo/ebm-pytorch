"""Block-Gibbs sampler for models with tractable conditionals (e.g. an RBM)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from torch import Tensor

from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler


@runtime_checkable
class _BlockGibbs(Protocol):
    def gibbs_step(self, v: Tensor) -> Tensor: ...


class GibbsSampler(Sampler):
    """Block Gibbs sampling via a model's own conditionals.

    Unlike the gradient-based samplers, this takes no step size: each ``step``
    delegates to the energy's ``gibbs_step`` (e.g. `ebm.nets.RBM`'s
    ``v -> h -> v'`` block update), so it draws exact samples from the model's
    conditionals. Running it for ``k`` steps from a data batch gives the CD-k
    negatives for `ebm.ContrastiveDivergence`.
    """

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        if not isinstance(energy, _BlockGibbs):
            raise TypeError(
                "GibbsSampler needs an energy exposing gibbs_step(v) -> v "
                "(e.g. ebm.nets.RBM); got a plain energy function"
            )
        return energy.gibbs_step(x)
