"""MCMC-free losses for binary energy models (pseudo-likelihood, ratio matching).

Both train any binary energy ``(B, *shape) -> (B,)`` — e.g. `ebm.nets.RBM`'s free
energy or `ebm.nets.IsingEnergy` — using only single-bit-flip energy differences,
with no sampler and no ``create_graph``. Cost is ``O(D)`` energy evaluations per
step (one per bit), so they suit modest dimensions.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ebm.energy import EnergyFn
from ebm.losses.base import LossOutput


class PseudoLikelihood(nn.Module):
    """Maximum pseudo-likelihood (Besag 1975) for binary energies.

    Maximizes ``Σ_i log p(x_i | x_{-i})``. For a binary variable the full
    conditional is closed-form in the energy — flipping bit ``i`` and differencing
    gives ``p(x_i = 1 | x_{-i}) = σ(E(x_i=0) - E(x_i=1))`` — so this is exact and
    sampler-free. The loss is the mean per-bit binary cross-entropy of each bit
    against its own conditional. Consistent: its optimum is the data distribution,
    which for a tractable model can be checked against exact maximum likelihood.
    """

    def forward(self, energy: EnergyFn, x: Tensor) -> LossOutput:
        b = x.shape[0]
        event = x.shape[1:]
        flat = x.reshape(b, -1)
        d = flat.shape[1]

        nll = x.new_zeros(b)
        for i in range(d):
            off = flat.clone()
            off[:, i] = 0.0
            on = flat.clone()
            on[:, i] = 1.0
            # logit of p(x_i = 1 | x_{-i}) = σ(E(x_i=0) - E(x_i=1)).
            logit = energy(off.reshape(b, *event)) - energy(on.reshape(b, *event))
            nll = nll + F.binary_cross_entropy_with_logits(logit, flat[:, i], reduction="none")

        loss = (nll / d).mean()
        return LossOutput(loss=loss, metrics={"loss": loss.item()})


class RatioMatching(nn.Module):
    """Ratio matching (Hyvärinen 2007) — the binary analogue of score matching.

    With the single-bit-flip ratio ``g_i(x) = σ(E(x) - E(flip_i x))``, minimizes
    ``Σ_i g_i(x)²`` over the data. Like score matching it needs no partition
    function and no negatives; unlike it, it is built from finite differences
    (bit flips) rather than gradients, which is what discrete data requires.
    Consistent — its optimum is the data distribution.
    """

    def forward(self, energy: EnergyFn, x: Tensor) -> LossOutput:
        b = x.shape[0]
        event = x.shape[1:]
        flat = x.reshape(b, -1)
        d = flat.shape[1]

        e_x = energy(x)
        total = x.new_zeros(b)
        for i in range(d):
            flipped = flat.clone()
            flipped[:, i] = 1.0 - flipped[:, i]
            g = torch.sigmoid(e_x - energy(flipped.reshape(b, *event)))
            total = total + g.pow(2)

        loss = total.mean()
        return LossOutput(loss=loss, metrics={"loss": loss.item()})
