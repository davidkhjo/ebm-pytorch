"""Distribution-correctness tests: samplers must recover a known Gaussian."""

import pytest
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


def test_accept_rate_is_none_before_sampling_then_float():
    for sampler in (ebm.MALA(step_size=0.2, steps=20), ebm.HMC(step_size=0.2, steps=5)):
        assert sampler.last_accept_rate is None  # nothing sampled yet
        sampler.sample(quadratic_energy, torch.randn(64, 2))
        rate = sampler.last_accept_rate
        assert isinstance(rate, float) and 0.0 < rate <= 1.0
    # Unadjusted Langevin has no accept step, so it stays None.
    ula = ebm.LangevinDynamics(step_size=0.01, steps=5)
    ula.sample(quadratic_energy, torch.randn(8, 2))
    assert ula.last_accept_rate is None


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


def test_parallel_tempering_reduces_to_base_on_gaussian():
    # With a unimodal target the cold chain must still be exactly N(0, I).
    base = ebm.LangevinDynamics(step_size=0.05, steps=1)
    pt = ebm.ParallelTempering(base, temperatures=(1.0, 2.0, 4.0), steps=300)
    samples = pt.sample(quadratic_energy, 3 * torch.randn(3000, 2))
    _check_standard_normal(samples)
    assert not samples.requires_grad
    assert all(0.0 < r <= 1.0 for r in pt.swap_rates())


def test_parallel_tempering_escapes_a_trapped_mode():
    means = torch.tensor([[-4.0], [4.0]])
    energy = ebm.nets.GaussianMixtureEnergy(means, std=1.0)
    x0 = -4.0 + 0.3 * torch.randn(1500, 1)  # every chain starts in the LEFT mode

    # A single cold Langevin chain stays trapped where it started.
    single = ebm.LangevinDynamics(step_size=0.08, steps=1000).sample(energy, x0.clone())
    assert (single > 0).float().mean().item() < 0.05

    # Parallel tempering crosses the barrier and recovers both modes ~50/50.
    base = ebm.LangevinDynamics(step_size=0.08, steps=1)
    pt = ebm.ParallelTempering(base, temperatures=(1.0, 4.0, 16.0, 64.0), steps=1000)
    pooled = pt.sample(energy, x0.clone())
    right_mass = (pooled > 0).float().mean().item()
    assert 0.3 < right_mass < 0.7
    assert all(0.1 < r < 0.95 for r in pt.swap_rates())


def test_parallel_tempering_validates_ladder():
    with pytest.raises(ValueError):
        ebm.ParallelTempering(ebm.LangevinDynamics(), temperatures=(4.0, 1.0))  # not ascending
    with pytest.raises(ValueError):
        ebm.ParallelTempering(ebm.LangevinDynamics(), temperatures=(1.0,))  # need >= 2
