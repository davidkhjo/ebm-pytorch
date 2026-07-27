"""Noise-contrastive estimation (Gutmann & Hyvärinen 2010)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution

from ebm.energy import EnergyFn
from ebm.losses.base import LossOutput


class NoiseContrastiveEstimation(nn.Module):
    """Binary NCE: classify data against samples from a known noise distribution.

    The model density is ``log p(x) = -E(x) - log_z`` with ``log_z`` learned as
    a parameter of this loss — include ``loss_fn.parameters()`` in the
    optimizer (the ``Trainer`` does this automatically). After training,
    ``log_z`` estimates the log partition function, so the model is
    (approximately) normalized.

    NCE works well when ``noise_dist`` overlaps the data substantially; the
    default standard normal is fine for standardized low-dimensional data but
    hopeless for images.

    Args:
        noise_dist: A ``torch.distributions.Distribution`` with ``sample`` and
            ``log_prob`` over single events; defaults to standard normal over
            the event shape encountered at first call.
        noise_ratio: Noise samples per data sample (ν in the paper).
    """

    def __init__(self, noise_dist: Distribution | None = None, noise_ratio: int = 1):
        super().__init__()
        self.noise_dist = noise_dist
        self.noise_ratio = noise_ratio
        self.log_z = nn.Parameter(torch.zeros(()))

    def _default_noise(self, x: Tensor) -> Distribution:
        loc = torch.zeros(x.shape[1:], device=x.device, dtype=x.dtype)
        return torch.distributions.Independent(
            torch.distributions.Normal(loc, torch.ones_like(loc)), len(x.shape[1:])
        )

    def forward(self, energy: EnergyFn, x: Tensor) -> LossOutput:
        noise_dist = self.noise_dist if self.noise_dist is not None else self._default_noise(x)
        n_noise = x.shape[0] * self.noise_ratio
        x_noise = noise_dist.sample((n_noise,)).to(device=x.device, dtype=x.dtype)

        log_ratio_pos = -energy(x) - self.log_z - noise_dist.log_prob(x)
        log_ratio_neg = -energy(x_noise) - self.log_z - noise_dist.log_prob(x_noise)
        log_nu = math.log(self.noise_ratio)

        loss = -(
            F.logsigmoid(log_ratio_pos - log_nu).mean()
            + self.noise_ratio * F.logsigmoid(-(log_ratio_neg - log_nu)).mean()
        )
        metrics = {
            "loss": loss.item(),
            "log_z": self.log_z.item(),
            "acc_pos": (log_ratio_pos > log_nu).float().mean().item(),
            "acc_neg": (log_ratio_neg < log_nu).float().mean().item(),
        }
        return LossOutput(loss=loss, metrics=metrics, x_neg=x_noise.detach())
