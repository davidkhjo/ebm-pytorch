"""JEM: treating a classifier as an energy-based model (Grathwohl et al., 2020).

A K-class classifier's logits define a joint EBM: ``E(x, y) = -logits[y]`` and
the marginal ``E(x) = -logsumexp_y logits[y]``, so ``p(y|x) = softmax(logits)``
falls out for free. Training combines cross-entropy on the logits with
contrastive divergence on the marginal energy.

For class-conditional persistent chains (one replay buffer per class — the
full JEM recipe), keep K separate ``ReplayBuffer``s and sample negatives with
``energy.condition(k)``; the marginal-CD setup here is the simple, stable core.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ebm.losses.base import LossOutput
from ebm.losses.cd import ContrastiveDivergence


class ClassifierEnergy(nn.Module):
    """Wrap a K-class classifier ``(B, *shape) -> (B, K)`` as an EBM.

    ``forward(x)`` is the marginal energy ``E(x) = -logsumexp(logits, 1)`` — a
    drop-in energy function for every sampler and loss in the library.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def logits(self, x: Tensor) -> Tensor:
        return self.net(x)

    def forward(self, x: Tensor) -> Tensor:
        return -self.logits(x).logsumexp(dim=1)

    def conditional(self, x: Tensor, y: Tensor | int) -> Tensor:
        """Joint energy ``E(x, y) = -logits[:, y]`` for fixed or per-sample labels."""
        logits = self.logits(x)
        if isinstance(y, int):
            return -logits[:, y]
        y = y.to(logits.device)
        return -logits.gather(1, y.reshape(-1, 1)).squeeze(1)

    def condition(self, y: Tensor | int) -> ConditionalEnergy:
        """Energy function of ``x`` with the class fixed, for conditional sampling."""
        return ConditionalEnergy(self, y)


class ConditionalEnergy(nn.Module):
    """``E(x, y)`` with ``y`` held fixed — an energy function of ``x`` alone.

    An ``nn.Module`` (not a closure) so samplers recognize it and freeze the
    underlying classifier's parameters during sampling.
    """

    def __init__(self, energy: ClassifierEnergy, y: Tensor | int):
        super().__init__()
        self.energy = energy
        self.y = y

    def forward(self, x: Tensor) -> Tensor:
        return self.energy.conditional(x, self.y)


class JEMLoss(nn.Module):
    """Joint classifier + EBM objective: ``CE(logits, y) + cd_weight * CD(E)``.

    Composes an existing ``ContrastiveDivergence`` (bring your own sampler,
    buffer, and energy regularization) with cross-entropy on the same batch.
    Costs one extra classifier forward on ``x`` (once for logits, once inside
    CD's marginal energy) — accepted for the sake of reusing CD wholesale.

    The ``supervised`` attribute tells ``Trainer`` to call this as
    ``loss_fn(energy, x, y)``.
    """

    supervised = True

    def __init__(self, cd: ContrastiveDivergence, cd_weight: float = 1.0):
        super().__init__()
        self.cd = cd
        self.cd_weight = cd_weight

    def forward(self, energy: ClassifierEnergy, x: Tensor, y: Tensor) -> LossOutput:
        logits = energy.logits(x)
        ce = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        cd_out = self.cd(energy, x)
        loss = ce + self.cd_weight * cd_out.loss

        metrics = {
            "loss": loss.item(),
            "ce": ce.item(),
            "acc": acc.item(),
            "cd_loss": cd_out.metrics["loss"],
            "energy_pos": cd_out.metrics["energy_pos"],
            "energy_neg": cd_out.metrics["energy_neg"],
            "energy_gap": cd_out.metrics["energy_gap"],
        }
        return LossOutput(loss=loss, metrics=metrics, x_neg=cd_out.x_neg)
