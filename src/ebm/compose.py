"""Energy composition: products of experts, mixtures, and tempering.

Because ``p ∝ exp(-E)``, algebra on energies is algebra on densities:

- adding energies multiplies densities (product of experts),
- ``-logsumexp`` of negated energies mixes densities (mixture of experts),
- dividing an energy by a temperature flattens or sharpens the density.

All compositions are energy functions themselves, so they nest and plug into
every sampler, loss, and evaluation utility. Components may be ``nn.Module``s
(their parameters are registered, trained, and frozen during sampling) or
plain callables.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ebm.energy import EnergyFn


class _Composite(nn.Module):
    """Holds component energies, registering the nn.Module ones."""

    def __init__(self, energies: Sequence[EnergyFn]):
        super().__init__()
        if not energies:
            raise ValueError("need at least one component energy")
        self.energies = list(energies)
        self._modules_list = nn.ModuleList([e for e in energies if isinstance(e, nn.Module)])


class SumEnergy(_Composite):
    """Product of experts: ``E(x) = Σ w_i E_i(x)`` so ``p ∝ Π p_i^{w_i}``.

    Each expert can veto: the product is only high-density where every
    component is. Weights default to 1.
    """

    weights: Tensor

    def __init__(self, *energies: EnergyFn, weights: Sequence[float] | None = None):
        super().__init__(energies)
        if weights is None:
            weights = [1.0] * len(self.energies)
        if len(weights) != len(self.energies):
            raise ValueError("need one weight per energy")
        self.register_buffer("weights", torch.tensor([float(w) for w in weights]))

    def forward(self, x: Tensor) -> Tensor:
        total = self.weights[0] * self.energies[0](x)
        for w, e in zip(self.weights[1:], self.energies[1:], strict=True):
            total = total + w * e(x)
        return total


class MixtureEnergy(_Composite):
    """Mixture of experts: ``p ∝ Σ w_i exp(-E_i(x))``.

    ``E(x) = -logsumexp_i(log w_i - E_i(x))``. Note the mixture is over the
    components' *unnormalized* densities — with differently-scaled experts the
    effective mixture proportions absorb the unknown Z_i.
    """

    log_weights: Tensor

    def __init__(self, *energies: EnergyFn, weights: Sequence[float] | None = None):
        super().__init__(energies)
        if weights is None:
            weights = [1.0] * len(self.energies)
        if len(weights) != len(self.energies) or any(w <= 0 for w in weights):
            raise ValueError("need one positive weight per energy")
        self.register_buffer("log_weights", torch.log(torch.tensor([float(w) for w in weights])))

    def forward(self, x: Tensor) -> Tensor:
        stacked = torch.stack(
            [lw - e(x) for lw, e in zip(self.log_weights, self.energies, strict=True)]
        )
        return -torch.logsumexp(stacked, dim=0)


class TemperedEnergy(nn.Module):
    """``E(x) / T``: temperature ``T > 1`` flattens ``p``, ``T < 1`` sharpens it.

    Useful for annealing schedules and for the intermediate distributions of
    tempered transitions (a tempered Gaussian energy has std ``sqrt(T)``).
    """

    def __init__(self, energy: EnergyFn, temperature: float):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.energy = energy if isinstance(energy, nn.Module) else None
        self._energy_fn = energy
        self.temperature = temperature

    def forward(self, x: Tensor) -> Tensor:
        return self._energy_fn(x) / self.temperature
