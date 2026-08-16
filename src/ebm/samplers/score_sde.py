"""Score-based SDE samplers for noise-conditional models (Song et al. 2021).

Both walk a descending noise ladder ``σ_0 > σ_1 > … > σ_L`` over a
noise-conditional energy ``energy(x, σ) -> (B,)`` (VE-SDE parameterization,
matching `MultiSigmaDenoisingScoreMatching` / `AnnealedLangevinDynamics`).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ebm.energy import ConditionalEnergyFn
from ebm.samplers.base import Sampler
from ebm.utils import frozen_params


class _ScoreSDESampler(Sampler):
    """Shared ladder plumbing for the VE score-SDE samplers."""

    sigmas: Tensor

    def __init__(self, sigmas: Tensor, steps: int = 1):
        super().__init__(steps)
        sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
        if sigmas.dim() != 1 or len(sigmas) < 2 or (sigmas <= 0).any():
            raise ValueError("sigmas must be a 1D tensor of >= 2 positive values")
        if (sigmas.diff() >= 0).any():
            raise ValueError("sigmas must be strictly decreasing (largest first)")
        self.sigmas = sigmas

    def _score_at(self, energy: ConditionalEnergyFn, x: Tensor, sigma: Tensor) -> Tensor:
        """Model score ``-∇_x E(x, σ)`` at a fixed scalar ``σ``."""
        sigma_b = sigma.to(x.device).expand(x.shape[0])
        _, grad = self._energy_grad(lambda z: energy(z, sigma_b), x)
        return -grad

    def sample(  # type: ignore[override]
        self,
        energy: ConditionalEnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        """Integrate the ladder from ``x_init``.

        ``x_init`` should be drawn at the top of the ladder, ``~ N(0, σ_0² I)``.
        """
        module = energy if isinstance(energy, nn.Module) else None
        x = x_init.detach().clone()
        traj = [x.clone()] if return_trajectory else None
        with frozen_params(module), torch.enable_grad():
            for i in range(len(self.sigmas) - 1):
                x = self.step(energy, x, i).detach()  # type: ignore[call-arg,arg-type]
                if traj is not None:
                    traj.append(x.clone())
        return torch.stack(traj) if traj is not None else x


class ProbabilityFlowODE(_ScoreSDESampler):
    """Deterministic reverse probability-flow ODE (Song et al. 2021, VE).

    The noise-free counterpart of `AnnealedLangevinDynamics`: one Euler step per
    ladder rung, ``x ← x + ½(σ_i² - σ_{i+1}²)·s(x, σ_i)`` with ``s = -∇_x E``.
    Because there is no injected noise the map is **deterministic** — the same
    ``x_init`` always yields the same samples — which is what enables exact
    likelihoods and fast, reproducible generation. Accuracy grows with the number
    of ladder rungs (use a long ``geometric_sigmas``). Start from
    ``x_init ~ N(0, σ_0² I)``.
    """

    def step(self, energy: ConditionalEnergyFn, x: Tensor, sigma_index: int = 0) -> Tensor:  # type: ignore[override]
        si, sj = self.sigmas[sigma_index], self.sigmas[sigma_index + 1]
        coef = 0.5 * (si**2 - sj**2)
        return x.detach() + coef * self._score_at(energy, x, si)


class PredictorCorrector(_ScoreSDESampler):
    """Predictor–corrector sampler (Song et al. 2021, VE).

    Per rung: ``n_corrector`` Langevin **corrector** steps at the fixed level
    ``σ_i`` (with a signal-to-noise-adaptive step) followed by one reverse-SDE
    **predictor** step to ``σ_{i+1}``::

        corrector:  x ← x + ε·s + sqrt(2ε)·z ,   ε = 2(snr·‖z‖/‖s‖)²
        predictor:  x ← x + (σ_i² - σ_{i+1}²)·s + sqrt(σ_i² - σ_{i+1}²)·z'

    The corrector keeps the chain on the ``σ_i`` marginal while the predictor
    advances the ladder — the standard NCSN++ sampler. Start from
    ``x_init ~ N(0, σ_0² I)``.
    """

    def __init__(self, sigmas: Tensor, n_corrector: int = 1, snr: float = 0.16, steps: int = 1):
        super().__init__(sigmas, steps)
        if n_corrector < 0:
            raise ValueError("n_corrector must be >= 0")
        self.n_corrector = n_corrector
        self.snr = snr

    def _mean_norm(self, x: Tensor) -> Tensor:
        return x.reshape(x.shape[0], -1).norm(dim=1).mean()

    def step(self, energy: ConditionalEnergyFn, x: Tensor, sigma_index: int = 0) -> Tensor:  # type: ignore[override]
        si, sj = self.sigmas[sigma_index], self.sigmas[sigma_index + 1]
        x = x.detach()
        for _ in range(self.n_corrector):
            s = self._score_at(energy, x, si)
            # One scalar SNR-adaptive step for the whole batch (mean norms) — a
            # per-sample ratio blows up wherever the score momentarily vanishes.
            ratio = self._mean_norm(torch.randn_like(x)) / self._mean_norm(s).clamp_min(1e-8)
            eps = 2 * (self.snr * ratio) ** 2
            x = x + eps * s + torch.sqrt(2 * eps) * torch.randn_like(x)
        # Predictor: one reverse-SDE Euler–Maruyama step.
        coef = si**2 - sj**2
        s = self._score_at(energy, x, si)
        return x + coef * s + torch.sqrt(coef) * torch.randn_like(x)
