"""Discrete-space sampling: Gibbs-with-Gradients (Grathwohl et al., 2021)."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler


class GibbsWithGradients(Sampler):
    """Locally-informed single-flip Metropolis-Hastings for binary data.

    Samples from ``p(x) ∝ exp(-E(x))`` over ``x ∈ {0, 1}^D``. The gradient of
    the energy at the current point scores every possible bit flip at the cost
    of one backward pass: ``d_i = (2 x_i - 1) ∂E/∂x_i`` estimates the energy
    decrease from flipping bit ``i``, a flip is proposed from
    ``Categorical(softmax(d / 2))``, and a Metropolis-Hastings correction makes
    the chain exact.

    The energy function must accept *float* tensors of 0/1 values and be
    differentiable with respect to that relaxed input. One ``step`` proposes
    one flip per chain and costs two energy-gradient evaluations (like MALA);
    plan on a few sweeps, i.e. ``steps`` of roughly 2-4x the number of bits.

    For training a discrete EBM, `ContrastiveDivergence` and `ReplayBuffer`
    work unchanged — pass a Bernoulli initializer:
    ``init_fn=lambda shape: torch.bernoulli(torch.full(shape, 0.5))``.
    """

    def __init__(self, steps: int = 100):
        super().__init__(steps)
        self.last_accept_rate: float | None = None

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        batch_size = x.shape[0]
        e_x, grad_x = self._energy_grad(energy, x)
        x_flat = x.detach().reshape(batch_size, -1)

        fwd_logits = (2 * x_flat - 1) * grad_x.reshape(batch_size, -1) / 2
        idx = torch.distributions.Categorical(logits=fwd_logits).sample()
        rows = torch.arange(batch_size, device=x.device)

        prop_flat = x_flat.clone()
        prop_flat[rows, idx] = 1 - prop_flat[rows, idx]
        proposal = prop_flat.reshape_as(x)

        e_prop, grad_prop = self._energy_grad(energy, proposal)
        bwd_logits = (2 * prop_flat - 1) * grad_prop.reshape(batch_size, -1) / 2

        log_q_fwd = torch.log_softmax(fwd_logits, dim=1)[rows, idx]
        log_q_bwd = torch.log_softmax(bwd_logits, dim=1)[rows, idx]
        log_alpha = (e_x - e_prop) + (log_q_bwd - log_q_fwd)

        accept = torch.log(torch.rand_like(log_alpha)) < log_alpha
        self.last_accept_rate = accept.float().mean().item()
        accept = accept.reshape(-1, *([1] * (x.dim() - 1)))
        return torch.where(accept, proposal, x.detach())
