"""Contrastive divergence (CD-k) and persistent CD with a replay buffer."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ebm.buffer import InitFn, ReplayBuffer
from ebm.energy import EnergyFn
from ebm.losses.base import LossOutput
from ebm.samplers.base import Sampler


class ContrastiveDivergence(nn.Module):
    """Maximum-likelihood surrogate loss via MCMC negatives.

    Computes ``E(x_data).mean() - E(x_neg).mean()`` where ``x_neg`` comes from
    running ``sampler`` (negatives are detached — the parameter gradient flows
    only through the two energy terms). With a ``ReplayBuffer`` this is
    persistent CD: chains start from the buffer and are written back after
    sampling, which is the standard recipe for image-scale EBMs.

    The optional ``energy_reg`` adds ``α (E(x)² + E(x_neg)²)`` to keep energy
    magnitudes bounded (Du & Mordatch use α=1 for images; for toy data small
    values like 0.001 suffice or it can be off).

    Note: the loss *value* is not a convergence signal (it can be negative and
    hovers near zero at equilibrium) — monitor ``metrics["energy_gap"]``.
    """

    def __init__(
        self,
        sampler: Sampler,
        buffer: ReplayBuffer | None = None,
        energy_reg: float = 0.0,
        init_fn: InitFn | None = None,
    ):
        super().__init__()
        self.sampler = sampler
        self.buffer = buffer
        self.energy_reg = energy_reg
        self.init_fn = init_fn if init_fn is not None else torch.randn

    def forward(self, energy: EnergyFn, x: Tensor) -> LossOutput:
        batch_size = x.shape[0]
        if self.buffer is not None:
            x0 = self.buffer.sample(batch_size, device=x.device)
        else:
            x0 = self.init_fn((batch_size, *x.shape[1:])).to(device=x.device, dtype=x.dtype)

        x_neg = self.sampler.sample(energy, x0).detach()
        if self.buffer is not None:
            self.buffer.push(x_neg)

        e_pos = energy(x)
        e_neg = energy(x_neg)
        ep, en = e_pos.mean(), e_neg.mean()
        loss = ep - en
        if self.energy_reg > 0:
            loss = loss + self.energy_reg * (e_pos.pow(2).mean() + e_neg.pow(2).mean())

        # One host sync for all four scalar metrics instead of four.
        loss_v, ep_v, en_v, gap_v = torch.stack([loss.detach(), ep, en, ep - en]).tolist()
        metrics = {
            "loss": loss_v,
            "energy_pos": ep_v,
            "energy_neg": en_v,
            "energy_gap": gap_v,
        }
        return LossOutput(loss=loss, metrics=metrics, x_neg=x_neg)
