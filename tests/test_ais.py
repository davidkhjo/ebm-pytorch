"""AIS must recover the closed-form log Z of a Gaussian energy."""

import math

import torch

import ebm
from ebm.ais import ais_log_z, log_likelihood, reverse_ais_log_z
from tests.conftest import quadratic_energy

TRUE_LOG_Z = math.log(2 * math.pi)  # E = ||x||^2/2 in 2D: Z = (2*pi)^{d/2}


def test_ais_recovers_gaussian_log_z():
    result = ais_log_z(quadratic_energy, (2,), base_scale=2.0, n_temps=40, n_chains=256)
    assert abs(result.log_z - TRUE_LOG_Z) < 0.15
    assert 1.0 <= result.ess <= 256.0
    assert result.stderr > 0
    assert result.samples.shape == (256, 2)
    assert (result.samples.std() - 1.0).abs().item() < 0.15


def test_ais_ess_equals_n_chains_when_target_matches_base():
    # If the target energy equals the base's negative log-density, every
    # incremental weight is exactly 0, so all importance weights are equal and
    # the effective sample size hits its ceiling n_chains exactly (stderr 0).
    base = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    n_chains = 32
    result = ais_log_z(lambda x: -base.log_prob(x), (2,), base=base, n_temps=10, n_chains=n_chains)
    assert result.ess == n_chains
    assert result.stderr == 0.0
    assert abs(result.log_z) < 1e-6  # log Z of a normalized density is 0


def test_ais_error_decreases_with_more_temps():
    few = ais_log_z(quadratic_energy, (2,), base_scale=2.0, n_temps=3, n_chains=256)
    many = ais_log_z(quadratic_energy, (2,), base_scale=2.0, n_temps=40, n_chains=256)
    err_few = abs(few.log_z - TRUE_LOG_Z)
    err_many = abs(many.log_z - TRUE_LOG_Z)
    assert err_many < err_few
    assert err_many < 0.15


def test_ais_freezes_and_restores_module_params():
    net = ebm.nets.MLPEnergy(dim=2, hidden=(16,))
    result = ais_log_z(net, (2,), n_temps=5, n_chains=16)
    assert math.isfinite(result.log_z)
    assert all(p.requires_grad for p in net.parameters())
    assert all(p.grad is None for p in net.parameters())


def test_ais_custom_base_schedule_and_validation():
    loc = torch.zeros(2)
    base = torch.distributions.Independent(torch.distributions.Normal(loc, 3 * torch.ones(2)), 1)
    explicit = torch.tensor([0.0, 0.3, 0.7, 1.0])
    result = ais_log_z(quadratic_energy, (2,), base=base, schedule=explicit, n_chains=32)
    assert math.isfinite(result.log_z)

    geo = ais_log_z(quadratic_energy, (2,), schedule="geometric", n_temps=10, n_chains=32)
    assert math.isfinite(geo.log_z)

    for bad in (
        torch.tensor([0.1, 1.0]),
        torch.tensor([0.0, 0.5]),
        torch.tensor([0.0, 0.5, 0.4, 1.0]),
    ):
        try:
            ais_log_z(quadratic_energy, (2,), schedule=bad, n_chains=4)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for betas {bad}")


def test_reverse_ais_recovers_gaussian_log_z():
    # Exact model samples: E = ||x||^2/2 is N(0, I).
    x = torch.randn(256, 2)
    result = reverse_ais_log_z(quadratic_energy, x, base_scale=2.0, n_temps=40)
    assert abs(result.log_z - TRUE_LOG_Z) < 0.15
    assert 1.0 <= result.ess <= 256.0
    assert result.stderr > 0
    # Final states have annealed back to the base N(0, 4I).
    assert result.samples.shape == (256, 2)
    assert (result.samples.std() - 2.0).abs().item() < 0.3


def test_forward_and_reverse_bracket_log_z():
    # A target far from the base with a coarse schedule makes both estimators
    # visibly biased in opposite directions. Individual runs are heavy-tailed,
    # so assert the bracket on medians over seeded repetitions.
    mu = torch.tensor([3.0, 3.0])

    def shifted(z):
        return 0.5 * ((z - mu) ** 2).sum(dim=-1)  # still log Z = log 2*pi

    lowers, uppers = [], []
    for seed in range(9):
        torch.manual_seed(seed)
        lowers.append(ais_log_z(shifted, (2,), n_temps=5, n_chains=128).log_z)
        uppers.append(reverse_ais_log_z(shifted, mu + torch.randn(128, 2), n_temps=5).log_z)
    lower, upper = torch.tensor(lowers).median(), torch.tensor(uppers).median()
    assert lower < TRUE_LOG_Z < upper


def test_reverse_ais_freezes_and_restores_module_params():
    net = ebm.nets.MLPEnergy(dim=2, hidden=(16,))
    result = reverse_ais_log_z(net, torch.randn(16, 2), n_temps=5)
    assert math.isfinite(result.log_z)
    assert all(p.requires_grad for p in net.parameters())
    assert all(p.grad is None for p in net.parameters())


def test_log_likelihood_matches_normal():
    x = torch.randn(64, 2)
    ll = log_likelihood(quadratic_energy, x, TRUE_LOG_Z)
    normal = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )
    assert torch.allclose(ll, normal.log_prob(x), atol=1e-5)
    bits = log_likelihood(quadratic_energy, x, TRUE_LOG_Z, dim=2)
    assert torch.allclose(bits, ll / (2 * math.log(2)), atol=1e-6)
