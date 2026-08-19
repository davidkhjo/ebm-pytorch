"""Continuous normalizing flow (FFJORD): closed-form log-det, invariance, training."""

import math

import torch
from torch import nn

import ebm


class _LinearField(nn.Module):
    """Constant linear velocity ``v(x, t) = x Aᵀ`` (divergence = tr(A))."""

    def __init__(self, a):
        super().__init__()
        self.a = nn.Parameter(a, requires_grad=False)

    def forward(self, x, t):
        return x @ self.a.t()


def test_cnf_matches_linear_field_closed_form():
    # For v = xAᵀ the flow maps x -> e^A x with log|det| = tr(A), so
    # log p(x) = log N(e^A x; 0, I) + tr(A). Hutchinson is unbiased → the mean matches.
    torch.manual_seed(0)
    d = 3
    a = 0.2 * torch.randn(d, d)
    cnf = ebm.nets.ContinuousNormalizingFlow(dim=d, n_steps=40, field=_LinearField(a)).eval()
    x = torch.randn(4000, d)
    lp = cnf.log_prob(x).detach()

    z = x @ torch.matrix_exp(a).t()
    closed = -0.5 * z.pow(2).sum(1) - 0.5 * d * math.log(2 * math.pi) + torch.trace(a)
    assert abs(lp.mean().item() - closed.mean().item()) < 0.05


def test_cnf_rotation_is_measure_preserving():
    # A skew-symmetric field is a rotation: tr = 0 exactly (εᵀSε = 0) and the base
    # Gaussian is rotation-invariant, so log_prob(x) == log N(x; 0, I) exactly.
    torch.manual_seed(0)
    d = 3
    a = torch.randn(d, d)
    skew = a - a.t()
    cnf = ebm.nets.ContinuousNormalizingFlow(dim=d, n_steps=40, field=_LinearField(skew)).eval()
    x = torch.randn(500, d)
    log_n = -0.5 * x.pow(2).sum(1) - 0.5 * d * math.log(2 * math.pi)
    assert (cnf.log_prob(x).detach() - log_n).abs().max().item() < 1e-4


def test_cnf_trains_and_samples():
    torch.manual_seed(0)
    data = ebm.datasets.eight_gaussians(4000)
    cnf = ebm.nets.ContinuousNormalizingFlow(dim=2, hidden=(64, 64), n_steps=20)
    opt = torch.optim.Adam(cnf.parameters(), lr=3e-3)
    cnf.train()
    for _ in range(600):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = -cnf.log_prob(batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    cnf.eval()
    samples = cnf.sample(2000)
    assert samples.shape == (2000, 2)
    assert ebm.eval.mmd(samples, data[:2000], bandwidth=0.3) < 0.05
