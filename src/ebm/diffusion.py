"""Variance-preserving diffusion (DDPM): schedule, ε-prediction loss, ancestral sampler.

Energy-parameterized so the EBM stays primary: the ε-network is *derived* from the
energy rather than a separate head. With the forward process
``x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`` and the score ``s = -∇_x E``,

    ε_θ(x, t) = -√(1-ᾱ_t)·s = √(1-ᾱ_t)·∇_x E(x, σ_t)

where ``σ_t = √(1-ᾱ_t)`` is the per-step noise level. Conditioning the energy on
``σ_t`` (a positive scalar) reuses the noise-conditional energies directly
(``NoiseConditionalMLPEnergy`` / ``NoiseConditionalConvEnergy``, ``energy(x, σ)``).
This is the variance-preserving counterpart to the variance-exploding NCSN stack.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ebm._functional import flat_sum
from ebm.energy import ConditionalEnergyFn
from ebm.losses.base import LossOutput
from ebm.samplers.base import Sampler
from ebm.utils import frozen_params


class VPSchedule(nn.Module):
    """Variance-preserving noise schedule (DDPM), ``T`` discrete steps.

    Holds ``β_t``, ``α_t = 1-β_t``, ``ᾱ_t = ∏α``, and the per-step noise level
    ``σ_t = √(1-ᾱ_t)`` as buffers (so it moves with ``.to(device)``). ``"linear"``
    is the original DDPM schedule; ``"cosine"`` is Nichol & Dhariwal (2021).
    """

    betas: Tensor
    alphas: Tensor
    alpha_bar: Tensor
    sigma: Tensor

    def __init__(
        self,
        num_steps: int = 1000,
        schedule: str = "linear",
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
    ):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if schedule == "linear":
            betas = torch.linspace(beta_min, beta_max, num_steps)
        elif schedule == "cosine":
            steps = torch.arange(num_steps + 1) / num_steps
            f = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-8, 0.999)
        else:
            raise ValueError("schedule must be 'linear' or 'cosine'")
        alphas = 1 - betas
        self.num_steps = num_steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", torch.cumprod(alphas, dim=0))
        self.register_buffer("sigma", (1 - self.alpha_bar).sqrt())

    def q_sample(self, x0: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Forward diffusion ``x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·eps`` at integer steps ``t``."""
        ab = self.alpha_bar[t].reshape(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


class VPDenoisingScoreMatching(nn.Module):
    """DDPM ε-prediction loss (``L_simple``) for an energy model.

    Draws a random step ``t``, forms the noised ``x_t``, and matches the derived
    ``ε_θ = √(1-ᾱ_t)·∇E(x_t, σ_t)`` to the true noise ``ε``. Backpropagates through
    the score, so the inner gradient uses ``create_graph=True`` (no MCMC). Pair the
    trained energy with `DDPMAncestralSampler`.
    """

    def __init__(self, schedule: VPSchedule):
        super().__init__()
        self.schedule = schedule

    def forward(self, energy: ConditionalEnergyFn, x: Tensor) -> LossOutput:
        b = x.shape[0]
        sch = self.schedule
        t = torch.randint(0, sch.num_steps, (b,), device=x.device)
        ab = sch.alpha_bar[t].reshape(b, *([1] * (x.dim() - 1)))
        eps = torch.randn_like(x)
        x_t = (ab.sqrt() * x + (1 - ab).sqrt() * eps).detach().requires_grad_(True)
        e = energy(x_t, sch.sigma[t])
        (grad_e,) = torch.autograd.grad(e.sum(), x_t, create_graph=True)
        eps_pred = (1 - ab).sqrt() * grad_e  # ε_θ = √(1-ᾱ)·∇E
        loss = flat_sum((eps_pred - eps).pow(2)).mean()
        return LossOutput(loss=loss, metrics={"loss": loss.item()})


class DDPMAncestralSampler(Sampler):
    """DDPM ancestral (reverse-process) sampler for an energy model.

    Walks ``t = T-1 … 0`` with the posterior mean
    ``μ = (x_t - β_t/√(1-ᾱ_t)·ε_θ)/√α_t`` and variance ``β_t`` (no noise at ``t=0``),
    where ``ε_θ = √(1-ᾱ_t)·∇E(x_t, σ_t)``. Start from ``x_init ~ N(0, I)``.
    """

    def __init__(self, schedule: VPSchedule, steps: int = 1):
        super().__init__(steps)
        self.schedule = schedule

    def sample(  # type: ignore[override]
        self,
        energy: ConditionalEnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        module = energy if isinstance(energy, nn.Module) else None
        x = x_init.detach().clone()
        traj = [x.clone()] if return_trajectory else None
        with frozen_params(module), torch.enable_grad():
            for t in reversed(range(self.schedule.num_steps)):
                x = self.step(energy, x, t).detach()  # type: ignore[call-arg,arg-type]
                if traj is not None:
                    traj.append(x.clone())
        return torch.stack(traj) if traj is not None else x

    def step(self, energy: ConditionalEnergyFn, x: Tensor, t: int = 0) -> Tensor:  # type: ignore[override]
        sch = self.schedule
        ab, a, beta = sch.alpha_bar[t], sch.alphas[t], sch.betas[t]
        sigma_t = sch.sigma[t].expand(x.shape[0])
        _, grad = self._energy_grad(lambda z: energy(z, sigma_t), x)
        eps_pred = (1 - ab).sqrt() * grad
        mean = (x.detach() - beta / (1 - ab).sqrt() * eps_pred) / a.sqrt()
        if t > 0:
            return mean + beta.sqrt() * torch.randn_like(x)
        return mean
