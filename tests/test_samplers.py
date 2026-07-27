"""Distribution-correctness tests: samplers must recover a known Gaussian."""

import torch

import ebm
from tests.conftest import quadratic_energy


def _check_standard_normal(samples, mean_tol=0.1, std_tol=0.12):
    assert samples.mean().abs().item() < mean_tol
    assert (samples.std() - 1.0).abs().item() < std_tol


def test_ula_targets_standard_normal():
    sampler = ebm.LangevinDynamics(step_size=0.05, steps=300)
    x0 = 3 * torch.randn(4000, 2)
    samples = sampler.sample(quadratic_energy, x0)
    _check_standard_normal(samples)
    assert not samples.requires_grad


def test_mala_targets_standard_normal():
    sampler = ebm.MALA(step_size=0.2, steps=300)
    samples = sampler.sample(quadratic_energy, 3 * torch.randn(4000, 2))
    _check_standard_normal(samples)
    assert 0.3 < sampler.last_accept_rate <= 1.0


def test_hmc_targets_standard_normal():
    sampler = ebm.HMC(step_size=0.2, leapfrog_steps=10, steps=50)
    samples = sampler.sample(quadratic_energy, 3 * torch.randn(2000, 2))
    _check_standard_normal(samples)
    assert sampler.last_accept_rate > 0.6


def test_sample_works_under_no_grad_and_restores_requires_grad():
    net = ebm.nets.MLPEnergy(dim=2, hidden=(16,))
    sampler = ebm.LangevinDynamics(step_size=0.01, steps=5)
    with torch.no_grad():
        samples = sampler.sample(net, torch.randn(8, 2))
    assert samples.shape == (8, 2)
    assert all(p.requires_grad for p in net.parameters())


def test_return_trajectory_and_steps_override():
    sampler = ebm.LangevinDynamics(step_size=0.01, steps=100)
    traj = sampler.sample(quadratic_energy, torch.randn(4, 2), steps=7, return_trajectory=True)
    assert traj.shape == (8, 4, 2)


def test_clamp_and_grad_clip():
    sampler = ebm.LangevinDynamics(step_size=0.5, steps=20, grad_clip=0.01, clamp=(-1.0, 1.0))
    samples = sampler.sample(quadratic_energy, 5 * torch.randn(16, 2))
    assert samples.abs().max().item() <= 1.0
