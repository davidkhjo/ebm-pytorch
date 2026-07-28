"""Evaluation helpers.

Relative diagnostics (OOD scores, batched energies) plus absolute
log-likelihood via annealed importance sampling (``ais_log_z`` /
``log_likelihood``, implemented in ``ebm.ais`` and re-exported here).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ebm.ais import AISResult, ais_log_z, log_likelihood
from ebm.energy import EnergyFn

__all__ = ["energies", "ood_auroc", "ais_log_z", "log_likelihood", "AISResult"]


@torch.no_grad()
def energies(energy: EnergyFn, x: Tensor, batch_size: int = 1024) -> Tensor:
    """Energies of ``x`` computed in batches, returned on CPU."""
    out = [energy(chunk).detach().cpu() for chunk in x.split(batch_size)]
    return torch.cat(out)


def ood_auroc(energy: EnergyFn, x_in: Tensor, x_out: Tensor) -> float:
    """AUROC for separating in-distribution from OOD data by energy.

    Uses ``-E(x)`` as the in-distribution score (low energy = in-distribution).
    1.0 is perfect separation, 0.5 is chance. Computed via the rank statistic
    (Mann-Whitney U), no sklearn required.
    """
    s_in = -energies(energy, x_in)
    s_out = -energies(energy, x_out)
    scores = torch.cat([s_in, s_out])
    n_in, n_out = len(s_in), len(s_out)

    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # Midranks for ties so AUROC is exact with discrete energies.
    for value in scores.unique():
        mask = scores == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()

    rank_sum_in = ranks[:n_in].sum().item()
    u = rank_sum_in - n_in * (n_in + 1) / 2
    return u / (n_in * n_out)
