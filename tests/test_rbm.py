"""RBM tests: the free-energy identity, exact log Z, Gibbs stationarity, CD training."""

import itertools

import pytest
import torch

import ebm


def _small_rbm(v=2, h=3, seed=0):
    torch.manual_seed(seed)
    rbm = ebm.nets.RBM(v, h)
    with torch.no_grad():
        rbm.W.copy_(torch.randn(h, v) * 0.8)
        rbm.b.copy_(torch.randn(v) * 0.5)
        rbm.c.copy_(torch.randn(h) * 0.5)
    return rbm


def _all_visibles(v):
    return torch.tensor(list(itertools.product([0.0, 1.0], repeat=v)))


def test_free_energy_matches_brute_force_marginal():
    v, h = 2, 3
    rbm = _small_rbm(v, h)
    vs = _all_visibles(v)
    model_p = torch.softmax(-rbm(vs), dim=0)

    # Enumerate the full joint and marginalize the hiddens by hand.
    z = 0.0
    unnorm = []
    for vv in vs:
        s = 0.0
        for hh in itertools.product([0.0, 1.0], repeat=h):
            hv = torch.tensor(hh)
            e = -(vv @ rbm.b) - (hv @ rbm.c) - (hv @ (rbm.W @ vv))
            s = s + torch.exp(-e)
        unnorm.append(s)
        z += s
    brute = torch.stack([u / z for u in unnorm])
    assert torch.allclose(model_p, brute.detach(), atol=1e-6)


def test_log_z_exact_both_enumeration_paths():
    # min(V,H) path (enumerate hiddens) must equal the visible-enumeration sum
    # and the brute-force Z, exactly.
    rbm = _small_rbm(2, 3)
    vs = _all_visibles(2)
    visible_enum = torch.logsumexp(-rbm(vs), dim=0)
    assert torch.allclose(rbm.log_z(), visible_enum, atol=1e-6)

    # A wider-visible model exercises the other branch (enumerate hiddens).
    rbm2 = _small_rbm(5, 3, seed=1)
    vs2 = _all_visibles(5)
    assert torch.allclose(rbm2.log_z(), torch.logsumexp(-rbm2(vs2), dim=0), atol=1e-5)

    with pytest.raises(ValueError):
        ebm.nets.RBM(25, 25).log_z()  # too large to enumerate


def test_block_gibbs_is_stationary_for_the_model():
    v, h = 2, 3
    rbm = _small_rbm(v, h)
    model_p = torch.softmax(-rbm(_all_visibles(v)), dim=0)

    torch.manual_seed(1)
    x = torch.bernoulli(torch.full((20000, v), 0.5))
    for _ in range(200):
        x = rbm.gibbs_step(x)
    codes = (x[:, 0] * 2 + x[:, 1]).long()
    emp = torch.bincount(codes, minlength=4).float()
    emp /= emp.sum()
    assert (emp - model_p).abs().max().item() < 0.02


def test_cd_recovers_independent_bernoulli():
    q = torch.tensor([0.2, 0.5, 0.8, 0.35])
    rbm = ebm.nets.RBM(4, 8)
    opt = torch.optim.Adam(rbm.parameters(), lr=0.05)
    for _ in range(1500):
        v_data = torch.bernoulli(q.expand(256, 4))
        v_neg = rbm.gibbs_step(v_data)  # CD-1: one block-Gibbs step from the data
        loss = rbm(v_data).mean() - rbm(v_neg).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    samples = torch.bernoulli(torch.full((20000, 4), 0.5))
    for _ in range(300):
        samples = rbm.gibbs_step(samples)
    assert (samples.mean(0) - q).abs().max().item() < 0.05


def test_gibbs_sampler_drives_the_rbm_and_rejects_plain_energies():
    rbm = ebm.nets.RBM(4, 6)
    sampler = ebm.GibbsSampler(steps=5)
    v0 = torch.bernoulli(torch.full((32, 4), 0.5))
    out = sampler.sample(rbm, v0)
    assert out.shape == (32, 4)
    assert torch.logical_or(out == 0, out == 1).all()  # stays binary

    with pytest.raises(TypeError):
        sampler.sample(lambda x: x.pow(2).sum(1), torch.randn(4, 4))
