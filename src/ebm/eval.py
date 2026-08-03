"""Evaluation helpers.

Relative diagnostics (OOD scores, batched energies, Fréchet distance / FID)
plus absolute log-likelihood via annealed importance sampling (``ais_log_z`` /
``reverse_ais_log_z`` / ``log_likelihood``, implemented in ``ebm.ais`` and
re-exported here).
"""

from __future__ import annotations

import torch
from torch import Tensor

from ebm.ais import AISResult, ais_log_z, log_likelihood, reverse_ais_log_z
from ebm.energy import EnergyFn

__all__ = [
    "energies",
    "frechet_distance",
    "mmd",
    "ood_auroc",
    "ais_log_z",
    "reverse_ais_log_z",
    "log_likelihood",
    "AISResult",
]


@torch.no_grad()
def energies(energy: EnergyFn, x: Tensor, batch_size: int = 1024) -> Tensor:
    """Energies of ``x`` computed in batches, returned on CPU."""
    out = [energy(chunk).detach().cpu() for chunk in x.split(batch_size)]
    return torch.cat(out)


@torch.no_grad()
def frechet_distance(
    x: Tensor,
    y: Tensor,
    *,
    feature_fn=None,
    batch_size: int = 1024,
) -> float:
    """Fréchet distance between Gaussians fitted to two sample sets.

    ``FD = ||μ_x - μ_y||² + Tr(Σ_x + Σ_y - 2 (Σ_x Σ_y)^{1/2})`` — 0 for
    identical distributions, larger is worse. Computed in float64 with the
    matrix square root taken by eigendecomposition; no scipy needed.

    With ``feature_fn=None`` samples are compared directly (flattened) — the
    right thing for toy data. Passing an Inception feature extractor (e.g.
    torchvision's ``inception_v3`` up to the pool layer, or any
    ``(B, *shape) -> (B, D)`` embedding network) makes this the standard
    **FID**; features are extracted in ``batch_size`` chunks and gathered on
    CPU. As always with FID, use a few thousand samples per side — small
    sample sets bias the covariance term upward.
    """

    def _features(t: Tensor) -> Tensor:
        if feature_fn is None:
            return t.reshape(len(t), -1).cpu().double()
        chunks = [feature_fn(c).detach().cpu() for c in t.split(batch_size)]
        return torch.cat(chunks).reshape(len(t), -1).double()

    fx, fy = _features(x), _features(y)
    mu_x, mu_y = fx.mean(0), fy.mean(0)
    cov_x = torch.cov(fx.T).reshape(fx.shape[1], -1)
    cov_y = torch.cov(fy.T).reshape(fy.shape[1], -1)

    # Tr((Σx Σy)^{1/2}) = Tr((√Σx Σy √Σx)^{1/2}), the latter symmetric PSD.
    vals, vecs = torch.linalg.eigh(cov_x)
    sqrt_x = (vecs * vals.clamp_min(0).sqrt()) @ vecs.T
    inner = sqrt_x @ cov_y @ sqrt_x
    tr_sqrt = torch.linalg.eigvalsh((inner + inner.T) / 2).clamp_min(0).sqrt().sum()

    fd = ((mu_x - mu_y) ** 2).sum() + cov_x.trace() + cov_y.trace() - 2 * tr_sqrt
    return float(fd.clamp_min(0))


@torch.no_grad()
def mmd(x: Tensor, y: Tensor, *, bandwidth: float | None = None) -> float:
    """Unbiased squared maximum mean discrepancy with an RBF kernel.

    ``MMD²(p, q) = E k(x,x') + E k(y,y') - 2 E k(x,y)`` — 0 iff the
    distributions match (for a characteristic kernel), and unlike
    `frechet_distance` it sees *all* moments: a blurred version of the data
    matches mean and covariance almost perfectly but not MMD. The unbiased
    estimator can come out slightly negative near zero.

    Samples are flattened and compared in float64. ``bandwidth=None`` uses the
    median heuristic (median pairwise distance across the pooled sample).
    Cost is O((n+m)²) in memory and time — a couple thousand samples per side
    is plenty.
    """
    fx = x.reshape(len(x), -1).cpu().double()
    fy = y.reshape(len(y), -1).cpu().double()
    d2 = torch.cdist(torch.cat([fx, fy]), torch.cat([fx, fy])).pow(2)
    if bandwidth is None:
        off_diag = d2[~torch.eye(len(d2), dtype=torch.bool)]
        bandwidth = off_diag.sqrt().median().item()
    k = torch.exp(-d2 / (2 * bandwidth**2))

    n, m = len(fx), len(fy)
    k_xx = (k[:n, :n].sum() - k[:n, :n].diagonal().sum()) / (n * (n - 1))
    k_yy = (k[n:, n:].sum() - k[n:, n:].diagonal().sum()) / (m * (m - 1))
    k_xy = k[:n, n:].mean()
    return float(k_xx + k_yy - 2 * k_xy)


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
