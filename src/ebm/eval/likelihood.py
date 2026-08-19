"""Likelihood / log-partition estimation (AIS + probability-flow ODE)."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm._functional import standard_normal_logprob
from ebm._ode import sample_eps as _hutchinson_eps
from ebm.ais import log_likelihood
from ebm.energy import ConditionalEnergyFn, EnergyFn


def pf_ode_log_likelihood(
    energy: ConditionalEnergyFn,
    x0: Tensor,
    sigmas: Tensor,
    *,
    n_hutchinson: int = 1,
    eps_dist: str = "rademacher",
) -> Tensor:
    """Exact log-density of a score model via the probability-flow ODE (FFJORD).

    Integrates the deterministic VE probability-flow ODE
    ``dx/dσ = -σ·s(x, σ)`` (``s = -∇_x E``) from data up to the base
    ``N(0, σ_max² I)`` while accumulating the log-determinant with the
    instantaneous change of variables ``d log p/dσ = -tr(∂f/∂x)``, the trace
    estimated by Hutchinson (one vector-Jacobian product per probe vector). This
    is a **partition-function-free** exact-likelihood estimator for any
    noise-conditional / diffusion energy — an independent alternative to the AIS
    `log_likelihood`. Returns per-sample ``log p(x0)`` in nats, shape ``(B,)``;
    ``bits/dim = -log p / (D·ln 2)``.

    Args:
        energy: noise-conditional energy ``(x, sigma) -> (B,)``.
        x0: data batch ``(B, *event)``.
        sigmas: **descending** noise ladder (largest first, as for
            `ProbabilityFlowODE`); more rungs = less discretization bias.
        n_hutchinson: number of trace probe vectors (raise for anisotropic models).
        eps_dist: ``"rademacher"`` (lower variance) or ``"gaussian"``.
    """
    sig = torch.as_tensor(sigmas, dtype=x0.dtype, device=x0.device)
    if sig.dim() != 1 or len(sig) < 2 or (sig <= 0).any():
        raise ValueError("sigmas must be a 1D tensor of >= 2 positive values")
    if (sig.diff() >= 0).any():
        raise ValueError("sigmas must be strictly decreasing (largest first)")
    sig = sig.flip(0)  # integrate data -> noise on the ascending ladder

    x = x0.detach()
    b = x.shape[0]
    logdet = x.new_zeros(b)
    epss = [_hutchinson_eps(x, eps_dist) for _ in range(n_hutchinson)]

    for i in range(len(sig) - 1):
        sa = sig[i]
        c = 0.5 * (sig[i + 1] ** 2 - sa**2)
        xr = x.detach().requires_grad_(True)
        with torch.enable_grad():
            e = energy(xr, sa.expand(b))
            (grad_e,) = torch.autograd.grad(e.sum(), xr, create_graph=True)
            s = -grad_e
            trace = x.new_zeros(b)
            for ep in epss:
                (jvp,) = torch.autograd.grad(s, xr, grad_outputs=ep, retain_graph=True)
                trace = trace + (jvp * ep).reshape(b, -1).sum(dim=1)
            trace = trace / len(epss)
        x = (x - c * s).detach()
        logdet = logdet + (-c) * trace.detach()

    return standard_normal_logprob(x, scale=float(sig[-1])) + logdet


@torch.no_grad()
def bits_per_dim(energy: EnergyFn, x: Tensor, log_z: float, *, dim: int | None = None) -> Tensor:
    """Per-sample bits-per-dimension, the standard density-model score (lower is better).

    ``BPD = (E(x) + log Z) / (D log 2)``, where ``D`` is the per-sample
    dimensionality (defaults to ``x[0].numel()``). This is exactly the negation
    of ``log_likelihood(..., dim=D)`` — flipped so the sign matches the
    convention in the literature, where a *lower* BPD means higher likelihood.
    Needs a ``log_z`` estimate from `ais_log_z` / `reverse_ais_log_z`.
    """
    if dim is None:
        dim = x[0].numel()
    return -log_likelihood(energy, x, log_z, dim=dim)
