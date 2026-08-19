"""Discrete-space sampling: Gibbs-with-Gradients (Grathwohl et al., 2021)."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm._functional import mh_accept
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
        self._last_accept = accept.float().mean()
        return mh_accept(x.detach(), proposal, accept)


class CategoricalGibbsWithGradients(Sampler):
    """Gibbs-with-Gradients for one-hot categorical data.

    Samples from ``p(x) ∝ exp(-E(x))`` over ``x ∈ {one-hot}^{D×K}``, given as
    float tensors ``(B, *dims, K)`` whose *last* dimension is the one-hot
    category axis. One energy gradient scores every single-position category
    change at once: with ``g = -∂E/∂x``, the estimated log-probability gain of
    switching position ``i`` to category ``k`` is ``d_{ik} = g_{ik} - g_{ic}``
    (``c`` the current category); a change is proposed from
    ``Categorical(softmax(d / 2))`` over all ``D·K`` options and corrected
    with Metropolis-Hastings, so the chain is exact.

    The energy must accept float one-hot tensors and be differentiable with
    respect to that relaxed input. Proposing the current category is allowed
    and is a no-op, so accept rates run high; as with the binary sampler, plan
    on ``steps`` of a few times the number of positions ``D``.

    For training, `ContrastiveDivergence` and `ReplayBuffer` work unchanged
    with a uniform one-hot initializer::

        init_fn = lambda shape: torch.nn.functional.one_hot(
            torch.randint(K, shape[:-1]), K
        ).float()
    """

    def __init__(self, steps: int = 100):
        super().__init__(steps)

    @staticmethod
    def _change_logits(x3: Tensor, grad3: Tensor) -> Tensor:
        # d_{ik} = g_{ik} - g_{i, current}; zero at each position's current category.
        g = -grad3
        current = (g * x3).sum(dim=-1, keepdim=True)
        return (g - current) / 2

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        batch_size, n_cats = x.shape[0], x.shape[-1]
        e_x, grad_x = self._energy_grad(energy, x)
        x3 = x.detach().reshape(batch_size, -1, n_cats)

        fwd_logits = self._change_logits(x3, grad_x.reshape_as(x3)).reshape(batch_size, -1)
        idx = torch.distributions.Categorical(logits=fwd_logits).sample()
        pos, cat = idx // n_cats, idx % n_cats
        rows = torch.arange(batch_size, device=x.device)

        old_cat = x3[rows, pos].argmax(dim=-1)
        prop3 = x3.clone()
        prop3[rows, pos] = torch.nn.functional.one_hot(cat, n_cats).to(x3.dtype)
        proposal = prop3.reshape_as(x)

        e_prop, grad_prop = self._energy_grad(energy, proposal)
        bwd_logits = self._change_logits(prop3, grad_prop.reshape_as(prop3)).reshape(batch_size, -1)

        bwd_idx = pos * n_cats + old_cat
        log_q_fwd = torch.log_softmax(fwd_logits, dim=1)[rows, idx]
        log_q_bwd = torch.log_softmax(bwd_logits, dim=1)[rows, bwd_idx]
        log_alpha = (e_x - e_prop) + (log_q_bwd - log_q_fwd)

        accept = torch.log(torch.rand_like(log_alpha)) < log_alpha
        self._last_accept = accept.float().mean()
        return mh_accept(x.detach(), proposal, accept)
