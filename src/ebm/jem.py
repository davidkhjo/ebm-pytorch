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

import torch
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

    def guide(self, y: Tensor | int, weight: float = 1.0) -> GuidedEnergy:
        """Classifier-free-guided energy of ``x`` for class ``y`` (weight ``w``)."""
        return GuidedEnergy(self, y, weight)


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


class GuidedEnergy(nn.Module):
    """Classifier-free-guided energy ``Ẽ_w(x|y) = (1+w)·E(x|y) − w·E(x)``.

    Sharpens class selection by amplifying the class-conditional energy relative to
    the marginal — the energy of ``p̃(x|y) ∝ p(x|y)^{1+w} / p(x)^w`` (Ho & Salimans,
    2022). ``weight=0`` is the plain conditional; larger ``weight`` pushes samples
    toward regions where class ``y`` dominates. Both branches come from the *same*
    `ClassifierEnergy` (no label-dropout needed), and ``logits`` is computed once.
    An ``nn.Module`` (not a closure) so samplers freeze the classifier during
    sampling. Note: for large ``weight`` the guided energy can become
    non-normalizable — keep ``weight`` modest and prefer MALA/HMC (which reject
    divergent proposals).
    """

    def __init__(self, energy: ClassifierEnergy, y: Tensor | int, weight: float = 1.0):
        super().__init__()
        if weight < 0:
            raise ValueError("guidance weight must be >= 0")
        self.energy = energy
        self.y = y
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        logits = self.energy.logits(x)
        if isinstance(self.y, int):
            e_cond = -logits[:, self.y]
        else:
            y = self.y.to(logits.device)
            e_cond = -logits.gather(1, y.reshape(-1, 1)).squeeze(1)
        e_marg = -logits.logsumexp(dim=1)
        return (1 + self.weight) * e_cond - self.weight * e_marg


class JEMLoss(nn.Module):
    """Joint classifier + EBM objective: ``CE(logits, y) + cd_weight * CD(E)``.

    Composes an existing ``ContrastiveDivergence`` (bring your own sampler,
    buffer, and energy regularization) with cross-entropy on the same batch.
    Costs one extra classifier forward on ``x`` (once for logits, once inside
    CD's marginal energy) — accepted for the sake of reusing CD wholesale.

    With ``conditional_negatives=True`` each negative chain is steered by a
    uniformly random class: the sampler runs on ``E(x, y) = -logits[y]``
    instead of the marginal (the loss terms stay marginal). This is how the
    JEM reference implementation samples, and it is what makes
    *class-conditional generation* via ``energy.condition(k)`` work — sampling
    a single logit that CD never trained just produces adversarial textures.

    The ``supervised`` attribute tells ``Trainer`` to call this as
    ``loss_fn(energy, x, y)``.
    """

    supervised = True

    def __init__(
        self,
        cd: ContrastiveDivergence,
        cd_weight: float = 1.0,
        conditional_negatives: bool = False,
    ):
        super().__init__()
        self.cd = cd
        self.cd_weight = cd_weight
        self.conditional_negatives = conditional_negatives

    def _conditional_cd(self, energy: ClassifierEnergy, x: Tensor) -> LossOutput:
        # ContrastiveDivergence with one change: chains target E(x, y) for a
        # random y each, so every class's sampling path gets trained. Loss
        # terms stay marginal (this is CD for p(x) = sum_y p(x, y) with
        # negatives drawn ancestrally: y ~ uniform, then x ~ p(x|y)).
        cd = self.cd
        batch_size = x.shape[0]
        if cd.buffer is not None:
            x0 = cd.buffer.sample(batch_size, device=x.device)
        else:
            x0 = cd.init_fn((batch_size, *x.shape[1:])).to(device=x.device, dtype=x.dtype)

        num_classes = energy.logits(x[:1]).shape[-1]
        y_neg = torch.randint(num_classes, (batch_size,), device=x.device)
        x_neg = cd.sampler.sample(energy.condition(y_neg), x0).detach()
        if cd.buffer is not None:
            cd.buffer.push(x_neg)

        e_pos = energy(x)
        e_neg = energy(x_neg)
        loss = e_pos.mean() - e_neg.mean()
        if cd.energy_reg > 0:
            loss = loss + cd.energy_reg * (e_pos.pow(2).mean() + e_neg.pow(2).mean())
        metrics = {
            "loss": loss.item(),
            "energy_pos": e_pos.mean().item(),
            "energy_neg": e_neg.mean().item(),
            "energy_gap": (e_pos.mean() - e_neg.mean()).item(),
        }
        return LossOutput(loss=loss, metrics=metrics, x_neg=x_neg)

    def forward(self, energy: ClassifierEnergy, x: Tensor, y: Tensor) -> LossOutput:
        logits = energy.logits(x)
        ce = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        if self.conditional_negatives:
            cd_out = self._conditional_cd(energy, x)
        else:
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
