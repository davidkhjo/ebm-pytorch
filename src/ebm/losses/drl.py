"""Diffusion recovery likelihood (Gao et al., 2021).

Trains a noise-conditional EBM by maximizing the *recovery* likelihood
``p(x | x̃) ∝ exp(-E(x, σ) - ||x̃ - x||² / (2 s²))`` between adjacent levels
of a noise ladder, instead of the marginal likelihood. The quadratic tether
makes the conditional close to unimodal, so short-run MCMC actually mixes —
this is the most stable known way to train EBMs on complex data.

Sign convention as everywhere in this library: ``p ∝ exp(-E)``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ebm._functional import validate_descending_sigmas as _validate_sigmas
from ebm.energy import ConditionalEnergyFn
from ebm.losses.base import LossOutput
from ebm.samplers.base import Sampler
from ebm.utils import frozen_params


def _expand(v: Tensor, like: Tensor) -> Tensor:
    """Reshape a per-sample scalar ``(B,)`` for broadcasting against ``like``."""
    return v.reshape(-1, *([1] * (like.dim() - 1)))


def _recovery_sample(
    energy: ConditionalEnergyFn,
    x_tilde: Tensor,
    sigma: Tensor,
    tether: Tensor,
    steps: int,
    step_scale: float,
) -> Tensor:
    """Langevin sampling of ``p(z | x̃) ∝ exp(-E(z, σ) - ||x̃ - z||²/(2 tether²))``.

    Starts at ``x_tilde`` and takes ``steps`` steps of size ``ε = step_scale
    tether²`` per sample: ``z ← z - (ε/2) ∇E_cond(z) + sqrt(ε) ξ`` (Gao et
    al., Alg. 1 with ``ε = b² σ²``). ``sigma`` and ``tether`` are per-sample
    ``(B,)``. Parameters are frozen and the result is detached.
    """
    x_tilde = x_tilde.detach()
    t2 = _expand(tether.pow(2), x_tilde)
    eps = step_scale * t2

    def cond_energy(z: Tensor) -> Tensor:
        quad = (z - x_tilde).pow(2).reshape(len(z), -1).sum(dim=1)
        return energy(z, sigma) + quad / (2 * tether.pow(2))

    module = energy if isinstance(energy, nn.Module) else None
    z = x_tilde
    with frozen_params(module), torch.enable_grad():
        for _ in range(steps):
            _, grad = Sampler._energy_grad(cond_energy, z)
            z = z.detach() - (eps / 2) * grad + eps.sqrt() * torch.randn_like(z)
    return z.detach()


class DiffusionRecoveryLikelihood(nn.Module):
    """Recovery-likelihood training for a noise-conditional energy.

    Expects a `ConditionalEnergyFn` ``energy(x, sigma) -> (B,)`` (e.g.
    ``nets.NoiseConditionalMLPEnergy``) and a strictly decreasing noise ladder
    ``σ₁ > … > σ_L`` (``geometric_sigmas`` reversed ordering, same convention
    as `AnnealedLangevinDynamics`; make σ₁ comparable to the data scale and
    σ_L small).

    Each training step draws a random adjacent pair ``(σ_t, σ_{t+1})`` per
    sample and does CD on the recovery posterior: the positive is data
    perturbed to the cleaner level, ``x⁺ = x + σ_{t+1} ε``; the tether point
    adds the incremental noise ``x̃ = x⁺ + s_t ε'`` with
    ``s_t = sqrt(σ_t² - σ_{t+1}²)``; the negative comes from ``mcmc_steps``
    Langevin steps on ``E(·, σ_{t+1}) + ||x̃ - ·||²/(2 s_t²)`` started at
    ``x̃``. The loss is ``E(x⁺, σ_{t+1}) - E(x⁻, σ_{t+1})`` (the tether has no
    parameters), so ``E(·, σ)`` learns each smoothed marginal. Generate with
    `drl_sample`. As with CD, monitor ``metrics["energy_gap"]``, not the loss
    value.

    Args:
        sigmas: Noise ladder ``(L,)``, strictly decreasing, registered as a
            buffer (moves with ``.to(device)``).
        mcmc_steps: Langevin steps per negative.
        step_scale: Step size as a fraction of the tether variance:
            ``ε = step_scale · s_t²`` (``b²`` in Gao et al.).
        energy_reg: Optional ``α (E(x⁺)² + E(x⁻)²)`` magnitude regularizer.
    """

    sigmas: Tensor

    def __init__(
        self,
        sigmas: Tensor,
        mcmc_steps: int = 30,
        step_scale: float = 0.1,
        energy_reg: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("sigmas", _validate_sigmas(sigmas))
        self.mcmc_steps = mcmc_steps
        self.step_scale = step_scale
        self.energy_reg = energy_reg

    def forward(self, energy: ConditionalEnergyFn, x: Tensor) -> LossOutput:
        sigmas = self.sigmas
        t = torch.randint(len(sigmas) - 1, (x.shape[0],), device=x.device)
        sigma_next = sigmas[t + 1]
        tether = (sigmas[t].pow(2) - sigmas[t + 1].pow(2)).sqrt()

        x_pos = x + _expand(sigma_next, x) * torch.randn_like(x)
        x_tilde = x_pos + _expand(tether, x) * torch.randn_like(x)
        x_neg = _recovery_sample(
            energy, x_tilde, sigma_next, tether, self.mcmc_steps, self.step_scale
        )

        e_pos = energy(x_pos, sigma_next)
        e_neg = energy(x_neg, sigma_next)
        loss = e_pos.mean() - e_neg.mean()
        if self.energy_reg > 0:
            loss = loss + self.energy_reg * (e_pos.pow(2).mean() + e_neg.pow(2).mean())

        metrics = {
            "loss": loss.item(),
            "energy_pos": e_pos.mean().item(),
            "energy_neg": e_neg.mean().item(),
            "energy_gap": (e_pos.mean() - e_neg.mean()).item(),
        }
        return LossOutput(loss=loss, metrics=metrics, x_neg=x_neg)


@torch.no_grad()
def drl_sample(
    energy: ConditionalEnergyFn,
    sigmas: Tensor,
    n: int,
    shape: tuple[int, ...],
    *,
    mcmc_steps: int = 30,
    step_scale: float = 0.1,
    device: torch.device | str | None = None,
) -> Tensor:
    """Progressive generation from a recovery-likelihood-trained energy.

    Initializes ``x ~ N(0, σ₁² I)`` (valid when σ₁ dominates the data scale,
    as in NCSN) and walks down the ladder: at each adjacent pair the current
    sample is the tether point and ``mcmc_steps`` Langevin steps sample
    ``p(x_{t+1} | x_t)``. Returns samples at the final level σ_L, ≈ clean data
    for small σ_L.
    """
    sigmas = _validate_sigmas(sigmas)
    x = float(sigmas[0]) * torch.randn(n, *shape, device=device)
    for t in range(len(sigmas) - 1):
        sigma_next = sigmas[t + 1].expand(n).to(x.device)
        tether = (sigmas[t].pow(2) - sigmas[t + 1].pow(2)).sqrt().expand(n).to(x.device)
        x = _recovery_sample(energy, x, sigma_next, tether, mcmc_steps, step_scale)
    return x
