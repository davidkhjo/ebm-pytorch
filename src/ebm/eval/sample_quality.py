"""Sample-quality distances: FID, MMD, precision/recall, inception score."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm._functional import rbf_bandwidth


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
    if len(fx) < 2 or len(fy) < 2:
        raise ValueError("frechet_distance needs at least 2 samples per set")
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
    n, m = len(x), len(y)
    if n < 2 or m < 2:
        raise ValueError("mmd needs at least 2 samples per set")
    fx = x.reshape(n, -1).cpu().double()
    fy = y.reshape(m, -1).cpu().double()
    d2 = torch.cdist(torch.cat([fx, fy]), torch.cat([fx, fy])).pow(2)
    bandwidth = rbf_bandwidth(d2, bandwidth)
    k = torch.exp(-d2 / (2 * bandwidth**2))

    k_xx = (k[:n, :n].sum() - k[:n, :n].diagonal().sum()) / (n * (n - 1))
    k_yy = (k[n:, n:].sum() - k[n:, n:].diagonal().sum()) / (m * (m - 1))
    k_xy = k[:n, n:].mean()
    return float(k_xx + k_yy - 2 * k_xy)


@torch.no_grad()
def precision_recall(
    real: Tensor,
    gen: Tensor,
    *,
    k: int = 3,
    feature_fn=None,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Improved precision & recall (Kynkäänniemi et al., 2019).

    Estimates each set's manifold as the union of per-point k-NN hyperspheres,
    then measures overlap: **precision** is the fraction of generated points that
    land inside the *real* manifold (fidelity), **recall** the fraction of real
    points inside the *generated* manifold (coverage/diversity). Unlike
    `frechet_distance`, this separates "are the samples realistic" from "do they
    cover the data" — a mode-dropping model scores high precision, low recall.

    ``feature_fn`` works exactly as in `frechet_distance` (``None`` compares the
    flattened samples directly; pass an embedding for image data). Returns
    ``(precision, recall)`` in ``[0, 1]``. Reference: arXiv:1904.06991.
    """

    def _features(t: Tensor) -> Tensor:
        if feature_fn is None:
            return t.reshape(len(t), -1).cpu().double()
        chunks = [feature_fn(c).detach().cpu() for c in t.split(batch_size)]
        return torch.cat(chunks).reshape(len(t), -1).double()

    fr, fg = _features(real), _features(gen)
    if len(fr) <= k or len(fg) <= k:
        raise ValueError(f"precision_recall needs more than k={k} samples per set")

    def _knn_radius(f: Tensor) -> Tensor:
        # Column 0 of the sorted distances is the zero self-distance, so the
        # k-th neighbor sits at column k.
        return torch.cdist(f, f).sort(dim=1).values[:, k]

    radius_real = _knn_radius(fr)
    radius_gen = _knn_radius(fg)
    # A point is "inside" a manifold if it lies within some reference sphere.
    precision = (torch.cdist(fg, fr) <= radius_real[None, :]).any(dim=1).double().mean()
    recall = (torch.cdist(fr, fg) <= radius_gen[None, :]).any(dim=1).double().mean()
    return float(precision), float(recall)


@torch.no_grad()
def inception_score(probs: Tensor) -> float:
    """Inception score ``exp(E_x KL(p(y|x) ‖ p(y)))`` from classifier probabilities.

    ``probs`` is an ``(N, K)`` tensor of per-sample class probabilities from *any*
    classifier you supply — this ships the score's formula only, not a bundled
    model, so it stays torch-only and works with whatever labels are meaningful
    for your data. Higher is better: it rewards confident per-sample predictions
    (sharp ``p(y|x)``) that are diverse in aggregate (near-uniform marginal
    ``p(y)``). ``IS = K`` for perfectly confident, perfectly balanced classes;
    ``IS = 1`` when every prediction equals the marginal.
    """
    p = probs.double()
    if p.dim() != 2:
        raise ValueError("probs must be (N, K) class probabilities")
    p = p.clamp_min(1e-12)
    marginal = p.mean(dim=0, keepdim=True)
    kl = (p * (p.log() - marginal.log())).sum(dim=1)
    return float(torch.exp(kl.mean()))
