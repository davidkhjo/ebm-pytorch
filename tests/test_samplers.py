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


def test_tempered_transitions_crosses_a_barrier_and_keeps_weights():
    # Asymmetric mixture (70/30). A single MALA chain stays trapped; tempered
    # transitions melt/refreeze across the barrier and recover the true split.
    energy = ebm.nets.GaussianMixtureEnergy(
        torch.tensor([[-4.0], [4.0]]), weights=[0.7, 0.3], std=1.0
    )
    x0 = -4.0 + 0.3 * torch.randn(4000, 1)  # trapped in the LEFT (heavier) mode

    trapped = ebm.MALA(step_size=0.15, steps=600).sample(energy, x0.clone())
    assert (trapped > 0).float().mean().item() < 0.05

    tt = ebm.TemperedTransitions(
        ebm.MALA(step_size=0.15), temperatures=(1.0, 2.0, 4.0, 8.0, 16.0), kernel_steps=5, steps=120
    )
    pooled = tt.sample(energy, x0.clone())
    assert 0.2 < (pooled > 0).float().mean().item() < 0.4  # recovers ~0.30, not 0.5
    assert tt.last_accept_rate is not None


def test_tempered_transitions_validates_ladder():
    with pytest.raises(ValueError):
        ebm.TemperedTransitions(ebm.MALA(), temperatures=(2.0, 4.0))  # first must be 1.0
    with pytest.raises(ValueError):
        ebm.TemperedTransitions(ebm.MALA(), temperatures=(1.0, 0.5))  # not ascending


def test_underdamped_targets_standard_normal_and_resets_momentum():
    sampler = ebm.UnderdampedLangevin(step_size=0.1, friction=1.0, steps=300)
    samples = sampler.sample(quadratic_energy, 3 * torch.randn(4000, 2))
    _check_standard_normal(samples)
    assert not samples.requires_grad
    # A second run must not inherit stale velocity — still exactly N(0, I).
    again = sampler.sample(quadratic_energy, 3 * torch.randn(4000, 2))
    _check_standard_normal(again)


def test_underdamped_explores_the_funnel():
    sampler = ebm.UnderdampedLangevin(step_size=0.1, friction=1.0, steps=300)
    samples = sampler.sample(ebm.nets.FunnelEnergy(dim=2, v_scale=3.0), torch.randn(4000, 2))
    assert torch.isfinite(samples).all()
    # The scale coordinate spreads out (a trapped chain would collapse it).
    assert samples[:, 0].std().item() > 1.0


def test_svgd_recovers_a_correlated_gaussian():
    mu = torch.tensor([1.0, -2.0])
    cov = torch.tensor([[2.0, 0.8], [0.8, 1.0]])
    precision = torch.linalg.inv(cov)

    def energy(x):
        c = x - mu
        return 0.5 * ((c @ precision) * c).sum(dim=1)

    particles = ebm.SVGD(step_size=0.3, steps=1000).sample(
        energy, torch.randn(300, 2) * 0.5 + torch.tensor([5.0, 5.0])
    )
    assert (particles.mean(0) - mu).abs().max().item() < 0.15
    assert (torch.cov(particles.T) - cov).abs().max().item() < 0.2
    assert not particles.requires_grad


def test_svgd_spreads_across_both_modes():
    energy = ebm.nets.GaussianMixtureEnergy(torch.tensor([[-3.0], [3.0]]), std=1.0)
    particles = ebm.SVGD(step_size=0.3, steps=800).sample(energy, torch.randn(400, 1) * 0.5)
    frac_neg = (particles < 0).float().mean().item()
    assert 0.3 < frac_neg < 0.7  # deterministic transport still covers both modes


def test_hmc_recovers_the_banana():
    # The banana has an exact sampler, so we can check the MCMC output directly.
    energy = ebm.nets.BananaEnergy(b=0.5, sigma=(1.0, 1.0))
    exact = energy.exact_sample(3000)
    hmc = ebm.HMC(step_size=0.15, leapfrog_steps=15, steps=200)
    samples = hmc.sample(energy, torch.randn(3000, 2))
    # Close to the exact draws, and much closer than an isotropic Gaussian is.
    to_exact = ebm.eval.mmd(samples, exact)
    to_normal = ebm.eval.mmd(torch.randn(3000, 2), exact)
    assert to_exact < 0.002
    assert to_exact < 0.3 * to_normal


def test_preconditioned_langevin_handles_anisotropy():
    # Target N(0, diag(1, 25)): per-dim std should be [1, 5].
    def aniso(x):
        return 0.5 * (x[:, 0] ** 2 / 1.0 + x[:, 1] ** 2 / 25.0)

    sampler = ebm.PreconditionedLangevin(
        precond=torch.tensor([1.0, 25.0]), step_size=0.05, steps=500
    )
    std = sampler.sample(aniso, torch.randn(6000, 2)).std(dim=0)
    assert abs(std[0].item() - 1.0) < 0.2
    assert abs(std[1].item() - 5.0) < 0.8
    with pytest.raises(ValueError):
        ebm.PreconditionedLangevin(precond=torch.tensor([1.0, -1.0]))


def _correlated_gaussian(cov):
    precision = torch.linalg.inv(cov)

    def energy(x):
        return 0.5 * ((x @ precision) * x).sum(dim=1)

    return energy


def test_adaptive_mala_tunes_step_size_to_target_accept():
    # A wildly wrong initial step is driven to the 0.574-optimal acceptance,
    # and the frozen chain recovers a correlated Gaussian's covariance.
    cov = torch.tensor([[2.0, 1.2], [1.2, 1.5]])
    sampler = ebm.AdaptiveMALA(step_size=0.1, steps=600, warmup=1000)
    samples = sampler.sample(_correlated_gaussian(cov), 3 * torch.randn(4000, 2))
    assert abs(sampler.last_accept_rate - 0.574) < 0.05
    assert (torch.cov(samples.T) - cov).abs().max().item() < 0.15
    assert not samples.requires_grad
    assert sampler.preconditioner is None


def test_adaptive_mala_preconditioner_learns_the_scales():
    # Target N(0, diag(25, 0.25)) — condition number 100. The estimated diagonal
    # metric should recover that ~100:1 ratio, and acceptance still hits target.
    cov = torch.diag(torch.tensor([25.0, 0.25]))
    sampler = ebm.AdaptiveMALA(step_size=0.1, steps=600, warmup=1000, precondition=True)
    samples = sampler.sample(_correlated_gaussian(cov), torch.randn(4000, 2))
    m = sampler.preconditioner
    assert m is not None
    ratio = (m[0] / m[1]).item()
    assert 40.0 < ratio < 250.0  # true condition number is 100
    assert abs(m.log().mean().exp().item() - 1.0) < 1e-4  # geometric mean 1
    assert abs(sampler.last_accept_rate - 0.574) < 0.06
    assert abs(samples.std(0)[0].item() - 5.0) < 0.7  # sqrt(25)
    assert abs(samples.std(0)[1].item() - 0.5) < 0.1  # sqrt(0.25)


def test_adaptive_mala_zero_warmup_keeps_step_size_and_validates():
    sampler = ebm.AdaptiveMALA(step_size=0.2, steps=8, warmup=0)
    traj = sampler.sample(quadratic_energy, torch.randn(256, 2), return_trajectory=True)
    assert traj.shape == (9, 256, 2)  # init + 8 transitions
    assert sampler.step_size == 0.2  # nothing to adapt
    assert isinstance(sampler.last_accept_rate, float)
    with pytest.raises(ValueError):
        ebm.AdaptiveMALA(target_accept=1.5)
    with pytest.raises(ValueError):
        ebm.AdaptiveMALA(warmup=-1)
