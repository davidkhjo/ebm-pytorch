"""Diffusion recovery likelihood against the analytic Gaussian case.

For data ``N(0, I)`` the σ-smoothed marginal is ``N(0, (1+σ²) I)`` with true
conditional energy ``E*(z, σ) = ||z||² / (2 (1+σ²))``, and the recovery
posterior between adjacent levels is Gaussian in closed form:
``mean = x̃ v / (v + s²)``, ``var = v s² / (v + s²)`` with ``v = 1 + σ_next²``.
"""

import math

import pytest
import torch

import ebm
from ebm.losses.drl import _recovery_sample


def analytic_energy(z, sigma):
    v = 1 + sigma**2
    return z.pow(2).reshape(len(z), -1).sum(dim=1) / (2 * v)


def test_recovery_sampler_matches_analytic_posterior():
    n = 4000
    sigma_t, sigma_next = 1.0, 0.5
    s2 = sigma_t**2 - sigma_next**2
    v = 1 + sigma_next**2
    x_tilde = 2.0 * torch.ones(n, 2)

    z = _recovery_sample(
        analytic_energy,
        x_tilde,
        sigma=torch.full((n,), sigma_next),
        tether=torch.full((n,), math.sqrt(s2)),
        steps=100,
        step_scale=0.1,
    )
    true_mean = 2.0 * v / (v + s2)
    true_std = math.sqrt(v * s2 / (v + s2))
    assert (z.mean(0) - true_mean).abs().max().item() < 0.05
    assert (z.std(0) - true_std).abs().max().item() < 0.05
    assert not z.requires_grad


def test_drl_sample_recovers_gaussian_on_analytic_energy():
    sigmas = ebm.geometric_sigmas(3.0, 0.05, 8)
    x = ebm.drl_sample(analytic_energy, sigmas, 4000, (2,), mcmc_steps=40)
    assert x.shape == (4000, 2)
    assert x.mean(0).abs().max().item() < 0.1
    assert (x.std(0) - 1.0).abs().max().item() < 0.1


def test_drl_loss_grads_metrics_and_detached_negatives():
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=2, hidden=(32,))
    loss_fn = ebm.DiffusionRecoveryLikelihood(ebm.geometric_sigmas(2.0, 0.1, 5), mcmc_steps=5)
    out = loss_fn(net, torch.randn(16, 2))

    assert not out.x_neg.requires_grad
    assert out.x_neg.shape == (16, 2)
    for key in ("loss", "energy_pos", "energy_neg", "energy_gap"):
        assert key in out.metrics
    out.loss.backward()
    grads = [p.grad for name, p in net.named_parameters() if "weight" in name]
    assert all(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(p.requires_grad for p in net.parameters())


def test_drl_sigma_validation():
    for bad in (
        torch.tensor([1.0]),  # too short
        torch.tensor([0.1, 1.0]),  # increasing
        torch.tensor([1.0, -0.5]),  # non-positive
        torch.tensor([1.0, 1.0]),  # not strictly decreasing
    ):
        with pytest.raises(ValueError):
            ebm.DiffusionRecoveryLikelihood(bad)


def test_drl_trains_gaussian_end_to_end():
    torch.manual_seed(0)
    data = torch.randn(4096, 1)
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=1, hidden=(32, 32))
    sigmas = ebm.geometric_sigmas(2.0, 0.1, 5)
    loss_fn = ebm.DiffusionRecoveryLikelihood(sigmas, mcmc_steps=20)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(300):
        x = data[torch.randint(len(data), (128,))]
        out = loss_fn(net, x)
        opt.zero_grad()
        out.loss.backward()
        opt.step()

    samples = ebm.drl_sample(net, sigmas, 2000, (1,), mcmc_steps=30)
    assert samples.mean().abs().item() < 0.25
    assert (samples.std() - 1.0).abs().item() < 0.25
