"""Evaluation helpers.

Relative diagnostics (OOD scores, batched energies, Fréchet distance / FID)
plus absolute log-likelihood via annealed importance sampling (``ais_log_z`` /
``reverse_ais_log_z`` / ``log_likelihood``, implemented in ``ebm.ais`` and
re-exported here).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ebm.ais import AISResult, ais_log_z, log_likelihood, reverse_ais_log_z
from ebm.energy import EnergyFn, score

__all__ = [
    "energies",
    "frechet_distance",
    "mmd",
    "ood_auroc",
    "ais_log_z",
    "reverse_ais_log_z",
    "log_likelihood",
    "bits_per_dim",
    "effective_sample_size",
    "split_rhat",
    "precision_recall",
    "inception_score",
    "kernel_stein_discrepancy",
    "classifier_two_sample_test",
    "fisher_divergence",
    "mutual_information",
    "AISResult",
]


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
    m, n, d = x.shape
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
    if bandwidth is None:
        off_diag = d2[~torch.eye(len(d2), dtype=torch.bool)]
        bandwidth = off_diag.sqrt().median().item()
    if bandwidth <= 0:
        raise ValueError(
            "RBF bandwidth is 0 (the median pairwise distance vanished — the "
            "samples are nearly identical); pass an explicit positive bandwidth"
        )
    k = torch.exp(-d2 / (2 * bandwidth**2))

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


@torch.no_grad()
def kernel_stein_discrepancy(
    energy: EnergyFn, x: Tensor, *, bandwidth: float | None = None
) -> float:
    """Kernel Stein discrepancy between an EBM and a sample set (Liu et al. 2016).

    A goodness-of-fit test that asks "did these samples come from *this* energy?"
    using **only the score** ``∇log p = -∇E`` — no partition function and no
    samples from the model needed (unlike `frechet_distance` / `mmd`, which
    compare two sample sets). With ``s = score(energy, ·)`` and the RBF kernel
    ``k(x,y) = exp(-‖x-y‖²/2h²)``, returns the U-statistic estimate of

    ``KSD² = E[ s(x)ᵀs(y) k + (s(x)-s(y))ᵀ(x-y) k/h² + (d/h² - ‖x-y‖²/h⁴) k ]``

    over distinct pairs. ``≈ 0`` when the samples match the model, positive and
    growing as they diverge. ``bandwidth=None`` uses the median heuristic. Like
    `mmd`, the unbiased estimate can come out slightly negative near zero.
    Computed in float64.
    """
    n = len(x)
    if n < 2:
        raise ValueError("kernel_stein_discrepancy needs at least 2 samples")
    s = score(energy, x).detach().reshape(n, -1).double()
    xf = x.detach().reshape(n, -1).double()
    d = xf.shape[1]

    diff = xf[:, None, :] - xf[None, :, :]  # (n, n, d)
    d2 = diff.pow(2).sum(dim=-1)  # (n, n)
    if bandwidth is None:
        off = d2[~torch.eye(n, dtype=torch.bool)]
        bandwidth = off.sqrt().median().item()
    if bandwidth <= 0:
        raise ValueError(
            "RBF bandwidth is 0 (the median pairwise distance vanished — the "
            "samples are nearly identical); pass an explicit positive bandwidth"
        )
    h2 = bandwidth**2
    k = torch.exp(-d2 / (2 * h2))

    sdot = s @ s.t()  # s(x)ᵀs(y)
    sx = (s[:, None, :] * diff).sum(dim=-1)  # s(x_i)·(x_i - x_j)
    sy = (s[None, :, :] * diff).sum(dim=-1)  # s(x_j)·(x_i - x_j)
    u = k * (sdot + (sx - sy) / h2 + d / h2 - d2 / h2**2)
    return float(u[~torch.eye(n, dtype=torch.bool)].mean())


def classifier_two_sample_test(
    real: Tensor,
    gen: Tensor,
    *,
    hidden: int = 64,
    epochs: int = 200,
    test_frac: float = 0.5,
) -> float:
    """Classifier two-sample test (Lopez-Paz & Oquab 2017): can a net tell them apart?

    Trains a small MLP to distinguish ``real`` from ``gen`` on a random split and
    returns its **held-out accuracy**: ``≈ 0.5`` means the two sets are
    indistinguishable (a good generative fit), ``→ 1.0`` means they are easily
    separated. A single-number, interpretable complement to `frechet_distance` /
    `mmd`. Pure torch; samples are flattened and standardized.
    """
    rf = real.detach().reshape(len(real), -1).float()
    gf = gen.detach().reshape(len(gen), -1).float()
    x = torch.cat([rf, gf])
    y = torch.cat([x.new_zeros(len(rf)), x.new_ones(len(gf))])
    x = (x - x.mean(0)) / x.std(0).clamp_min(1e-8)

    perm = torch.randperm(len(x))
    x, y = x[perm], y[perm]
    n_test = int(len(x) * test_frac)
    x_tr, y_tr, x_te, y_te = x[n_test:], y[n_test:], x[:n_test], y[:n_test]

    net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(net(x_tr).squeeze(-1), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (net(x_te).squeeze(-1) > 0).float()
    return float((pred == y_te).float().mean())


def fisher_divergence(energy_a: EnergyFn, energy_b: EnergyFn, x: Tensor) -> float:
    """Fisher divergence ``E_x ‖s_a(x) - s_b(x)‖²`` between two EBMs (via scores).

    The expected squared gap between the two models' scores ``s = -∇E`` on the
    sample set ``x`` — a score-space distance that, unlike a partition-function
    comparison, is available directly from the energies. ``0`` iff the two scores
    agree on the support of ``x``. (Needs gradients, so not under ``no_grad``.)
    """
    s_a = score(energy_a, x).detach()
    s_b = score(energy_b, x).detach()
    return float((s_a - s_b).reshape(len(x), -1).pow(2).sum(dim=1).mean())


def mutual_information(
    x: Tensor, y: Tensor, *, hidden: int = 64, epochs: int = 400, lr: float = 1e-3
) -> float:
    """Mutual information between paired samples via MINE (Belghazi et al. 2018).

    Trains a small statistics network ``T(x, y)`` to maximize the Donsker–Varadhan
    lower bound ``I(X;Y) ≥ E_{p(x,y)}[T] - log E_{p(x)p(y)}[exp T]`` (marginal
    samples obtained by shuffling ``y`` within the batch) and returns the bound.
    A neural, distribution-free estimator; higher = more dependence. Pure torch.
    """
    xf = x.detach().reshape(len(x), -1).float()
    yf = y.detach().reshape(len(y), -1).float()
    net = nn.Sequential(
        nn.Linear(xf.shape[1] + yf.shape[1], hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bound = 0.0
    for _ in range(epochs):
        perm = torch.randperm(len(yf))
        joint = net(torch.cat([xf, yf], dim=1)).squeeze(-1)
        marginal = net(torch.cat([xf, yf[perm]], dim=1)).squeeze(-1)
        mi = joint.mean() - torch.logsumexp(marginal, dim=0) + math.log(len(marginal))
        opt.zero_grad()
        (-mi).backward()
        opt.step()
        bound = float(mi.detach())
    return bound


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
