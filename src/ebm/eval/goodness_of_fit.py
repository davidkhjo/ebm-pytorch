"""Goodness-of-fit and information metrics (KSD, C2ST, Fisher, MINE)."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ebm._functional import rbf_bandwidth
from ebm.energy import EnergyFn, score


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
    bandwidth = rbf_bandwidth(d2, bandwidth)
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
    if not 0 < test_frac < 1:
        raise ValueError("test_frac must be in (0, 1)")
    rf = real.detach().reshape(len(real), -1).float()
    gf = gen.detach().reshape(len(gen), -1).float()
    if len(rf) < 2 or len(gf) < 2:
        raise ValueError("classifier_two_sample_test needs at least 2 samples per set")
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
