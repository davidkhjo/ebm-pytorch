"""MCMC diagnostics and batched energy/OOD helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm.energy import EnergyFn


def _as_chains(samples: Tensor) -> Tensor:
    """Coerce MCMC output to ``(n_chains, n_samples, dim)`` in float64."""
    x = samples.detach().cpu().double()
    if x.dim() == 2:
        x = x.unsqueeze(-1)  # (M, N) -> a single scalar quantity per draw
    if x.dim() != 3:
        raise ValueError("samples must be (n_chains, n_samples) or (n_chains, n_samples, dim)")
    if x.shape[1] < 4:
        raise ValueError("need at least 4 samples per chain for a meaningful diagnostic")
    return x


@torch.no_grad()
def split_rhat(samples: Tensor) -> Tensor:
    """Split-R̂ (Gelman–Rubin) convergence diagnostic, per dimension.

    Input is ``(n_chains, n_samples[, dim])``. Each chain is split in half (so a
    single long chain still exposes non-stationarity), then

    ``R̂ = sqrt(v̂⁺ / W)``,  ``v̂⁺ = (N-1)/N · W + B/N``

    with ``W`` the mean within-chain variance and ``B`` the between-chain
    variance of the half-chains. Values near ``1.0`` indicate the chains have
    mixed; ``> 1.01`` is the usual "not converged" flag. Returns a ``(dim,)``
    tensor. Reference: Vehtari et al. (2021), BDA3.
    """
    x = _as_chains(samples)
    _, n, _ = x.shape
    half = n // 2
    if half < 2:
        raise ValueError("need at least 4 samples per chain to split and estimate variance")
    split = torch.cat([x[:, :half], x[:, half : 2 * half]], dim=0)  # (2M, half, d)
    means = split.mean(dim=1)  # (2M, d)
    within = split.var(dim=1, unbiased=True).mean(dim=0)  # (d,)
    between = half * means.var(dim=0, unbiased=True)  # (d,)
    var_plus = (half - 1) / half * within + between / half
    return torch.sqrt(var_plus / within)


@torch.no_grad()
def effective_sample_size(samples: Tensor) -> Tensor:
    """Effective sample size of an MCMC run, per dimension.

    Input is ``(n_chains, n_samples[, dim])``. Combines the within-chain
    autocorrelation with the between-chain variance (Stan/BDA3 estimator) and
    truncates the autocorrelation sum with Geyer's initial-monotone rule:

    ``ESS = M·N / (1 + 2 Σ_t ρ̂_t)``.

    For independent draws ``ESS → M·N``; a stuck or slowly-mixing chain gives an
    ESS far below the raw draw count. Autocovariances are computed by FFT
    (zero-padded past ``2N`` to avoid circular wraparound), no scipy. Returns a
    ``(dim,)`` tensor.
    """
    x = _as_chains(samples)
    m, n, d = x.shape

    centered = x - x.mean(dim=1, keepdim=True)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft <<= 1
    spec = torch.fft.rfft(centered, n=n_fft, dim=1)
    acov = torch.fft.irfft(spec.abs().pow(2), n=n_fft, dim=1)[:, :n, :]  # (M, N, d)
    acov = acov / n * (n / (n - 1))  # unbiased so acov[:,0] == per-chain variance

    within = acov[:, 0, :].mean(dim=0)  # (d,)  == W
    if m > 1:
        between = n * x.mean(dim=1).var(dim=0, unbiased=True)  # (d,)
    else:
        between = torch.zeros(d, dtype=x.dtype)
    var_plus = (n - 1) / n * within + between / n
    mean_acov = acov.mean(dim=0)  # (N, d)
    rho = 1 - (within[None, :] - mean_acov) / var_plus[None, :]  # (N, d)

    ess = torch.empty(d, dtype=x.dtype)
    for j in range(d):
        r = rho[:, j]
        # Geyer initial-monotone sequence: pair Γ_k = ρ_{2k} + ρ_{2k+1}, sum
        # while positive, forcing the pairs to be non-increasing. Then
        # τ = -1 + 2 Σ Γ_k  (== 1 + 2 Σ_{t≥1} ρ_t untruncated).
        gamma_sum = 0.0
        prev_gamma = float("inf")
        k = 0
        while 2 * k + 1 < n:
            gamma = float(r[2 * k] + r[2 * k + 1])
            gamma = min(gamma, prev_gamma)  # enforce non-increasing pairs
            if gamma <= 0:
                break
            gamma_sum += gamma
            prev_gamma = gamma
            k += 1
        tau = -1.0 + 2.0 * gamma_sum
        ess[j] = m * n / max(tau, 1e-12)
    return ess


@torch.no_grad()
def energies(energy: EnergyFn, x: Tensor, batch_size: int = 1024) -> Tensor:
    """Energies of ``x`` computed in batches, returned on CPU."""
    out = [energy(chunk).detach().cpu() for chunk in x.split(batch_size)]
    return torch.cat(out)


@torch.no_grad()
def ood_auroc(energy: EnergyFn, x_in: Tensor, x_out: Tensor) -> float:
    """AUROC for separating in-distribution from OOD data by energy.

    Uses ``-E(x)`` as the in-distribution score (low energy = in-distribution).
    1.0 is perfect separation, 0.5 is chance. Computed via the rank statistic
    (Mann-Whitney U), no sklearn required.
    """
    s_in = -energies(energy, x_in)
    s_out = -energies(energy, x_out)
    scores = torch.cat([s_in, s_out])
    n_in, n_out = len(s_in), len(s_out)

    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # Midranks for ties so AUROC is exact with discrete energies.
    for value in scores.unique():
        mask = scores == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()

    rank_sum_in = ranks[:n_in].sum().item()
    u = rank_sum_in - n_in * (n_in + 1) / 2
    return u / (n_in * n_out)
