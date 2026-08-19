"""Continuous normalizing flow (FFJORD) — a trainable exact-likelihood density."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ebm._functional import standard_normal_logprob
from ebm._ode import integrate_logprob, integrate_reverse, sample_eps


class _TimeMLP(nn.Module):
    """Velocity field ``v(x, t) -> (B, dim)`` conditioned on time by concatenation."""

    def __init__(self, dim: int, hidden: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = dim + 1
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.SiLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, t: float) -> Tensor:
        tt = x.new_full((x.shape[0], 1), float(t))
        return self.net(torch.cat([x, tt], dim=1))


class ContinuousNormalizingFlow(nn.Module):
    """FFJORD continuous normalizing flow (Grathwohl et al. 2019), trainable in pure torch.

    The trainable generalization of `ebm.eval.pf_ode_log_likelihood`: a neural
    velocity field ``v_θ(x, t)`` defines the ODE ``ẋ = v_θ`` from data (t=0) to a
    ``N(0, I)`` base (t=1), and the exact log-density follows the instantaneous
    change of variables ``d log p/dt = -tr(∂v/∂x)`` (Hutchinson trace). Fixed-step
    RK4, direct backprop (no adjoint), so `log_prob` trains by maximum likelihood
    and `sample` draws in one reverse pass. Like the coupling flow, it is a
    self-normalized energy: ``forward(x) = -log_prob(x)`` with ``log Z = 0``.
    Operates on vector data ``(B, dim)``. Pass a custom ``field`` to override the
    default MLP (e.g. for the closed-form tests).
    """

    def __init__(
        self,
        dim: int,
        hidden: Sequence[int] = (64, 64),
        n_steps: int = 20,
        field: nn.Module | None = None,
        eps_dist: str = "rademacher",
    ):
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim
        self.n_steps = n_steps
        self.eps_dist = eps_dist
        self.field = field if field is not None else _TimeMLP(dim, hidden)

    def log_prob(self, x: Tensor) -> Tensor:
        eps = sample_eps(x, self.eps_dist)
        x0 = x if x.requires_grad else x.detach().requires_grad_(True)
        with torch.enable_grad():
            z, logdet = integrate_logprob(
                self.field, x0, eps, n_steps=self.n_steps, create_graph=self.training
            )
        return standard_normal_logprob(z) + logdet

    def sample(self, n: int) -> Tensor:
        z = torch.randn(n, self.dim, device=next(self.parameters()).device)
        return integrate_reverse(self.field, z, n_steps=self.n_steps)

    def forward(self, x: Tensor) -> Tensor:
        return -self.log_prob(x)  # energy = -log p, so log Z = 0 exactly
