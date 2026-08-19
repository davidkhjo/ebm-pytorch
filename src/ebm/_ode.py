"""Fixed-step augmented-ODE integration for continuous normalizing flows (private).

Torch-only RK4 over the augmented state ``(x, log-det)``, with the divergence
``tr(∂v/∂x)`` estimated by Hutchinson (one vector-Jacobian product per probe).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ebm._functional import flat_sum

Field = Callable[[Tensor, float], Tensor]


def sample_eps(x: Tensor, dist: str = "rademacher") -> Tensor:
    """A Hutchinson probe vector shaped like ``x`` (Rademacher or Gaussian)."""
    if dist == "rademacher":
        return (torch.randint(0, 2, x.shape, device=x.device) * 2 - 1).to(x.dtype)
    if dist == "gaussian":
        return torch.randn_like(x)
    raise ValueError("dist must be 'rademacher' or 'gaussian'")


def _vel_and_div(field: Field, x: Tensor, t: float, eps: Tensor, create_graph: bool):
    v = field(x, t)
    (grad_x,) = torch.autograd.grad(
        v, x, grad_outputs=eps, create_graph=create_graph, retain_graph=True
    )
    return v, flat_sum(grad_x * eps)


def integrate_logprob(
    field: Field, x0: Tensor, eps: Tensor, *, n_steps: int, create_graph: bool
) -> tuple[Tensor, Tensor]:
    """RK4 the flow from data (t=0) to base (t=1), accumulating ``∫ tr(∂v/∂x) dt``.

    Returns ``(z, logdet)`` where ``z = x(1)`` and ``logdet = ∫₀¹ tr(∂v/∂x) dt`` so
    that ``log p(x0) = log p_base(z) + logdet``. Requires ``x0`` to track grad.
    """
    h = 1.0 / n_steps
    x, logdet = x0, x0.new_zeros(x0.shape[0])
    for i in range(n_steps):
        t = i * h
        k1v, k1t = _vel_and_div(field, x, t, eps, create_graph)
        k2v, k2t = _vel_and_div(field, x + 0.5 * h * k1v, t + 0.5 * h, eps, create_graph)
        k3v, k3t = _vel_and_div(field, x + 0.5 * h * k2v, t + 0.5 * h, eps, create_graph)
        k4v, k4t = _vel_and_div(field, x + h * k3v, t + h, eps, create_graph)
        x = x + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        logdet = logdet + h / 6 * (k1t + 2 * k2t + 2 * k3t + k4t)
    return x, logdet


@torch.no_grad()
def integrate_reverse(field: Field, z: Tensor, *, n_steps: int) -> Tensor:
    """RK4 the flow from base (t=1) back to data (t=0); no log-det needed for sampling."""
    h = 1.0 / n_steps
    x = z
    for i in range(n_steps):
        t = 1.0 - i * h
        k1 = field(x, t)
        k2 = field(x - 0.5 * h * k1, t - 0.5 * h)
        k3 = field(x - 0.5 * h * k2, t - 0.5 * h)
        k4 = field(x - h * k3, t - h)
        x = x - h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x
