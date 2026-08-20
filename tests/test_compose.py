"""Composition algebra checked against closed-form Gaussian identities."""

import math

import torch

import ebm
from tests.conftest import quadratic_energy


def gaussian_energy(mean, var):
    return lambda x: ((x - mean) ** 2 / (2 * var)).sum(dim=1)


def test_sum_energy_is_gaussian_product():
    # N(0, 1) * N(2, 1) ∝ N(1, 0.5): precision-weighted mean and summed precision.
    product = ebm.SumEnergy(gaussian_energy(0.0, 1.0), gaussian_energy(2.0, 1.0))
    sampler = ebm.MALA(step_size=0.1, steps=400)
    samples = sampler.sample(product, torch.randn(4000, 1))
    assert (samples.mean() - 1.0).abs().item() < 0.05
    assert (samples.std() - math.sqrt(0.5)).abs().item() < 0.05


def test_sum_energy_weights():
    # 2 * E for a standard Gaussian is N(0, 1/2).
    doubled = ebm.SumEnergy(quadratic_energy, weights=[2.0])
    x = torch.randn(16, 2)
    assert torch.allclose(doubled(x), 2 * quadratic_energy(x))


def test_tempered_energy_scales_std():
    tempered = ebm.TemperedEnergy(quadratic_energy, temperature=4.0)
    sampler = ebm.MALA(step_size=0.5, steps=400)
    samples = sampler.sample(tempered, torch.randn(4000, 2))
    assert (samples.std() - 2.0).abs().item() < 0.1

    try:
        ebm.TemperedEnergy(quadratic_energy, temperature=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for temperature 0")


def test_mixture_energy_matches_closed_form():
    e1 = gaussian_energy(-2.0, 0.25)
    e2 = gaussian_energy(2.0, 0.25)
    mix = ebm.MixtureEnergy(e1, e2, weights=[1.0, 3.0])
    x = torch.randn(64, 1)
    expected = -torch.logsumexp(torch.stack([math.log(1.0) - e1(x), math.log(3.0) - e2(x)]), dim=0)
    assert torch.allclose(mix(x), expected, atol=1e-5)


def test_mixture_sampling_is_bimodal():
    mix = ebm.MixtureEnergy(gaussian_energy(-2.0, 0.25), gaussian_energy(2.0, 0.25))
    # Chains initialized across both basins; Langevin keeps them balanced.
    x0 = torch.cat([torch.randn(1000, 1) - 2, torch.randn(1000, 1) + 2])
    samples = ebm.LangevinDynamics(step_size=0.05, steps=200).sample(mix, x0)
    assert samples.mean().abs().item() < 0.2
    assert (samples.abs().mean() - 2.0).abs().item() < 0.2


def test_composition_registers_and_trains_modules():
    net = ebm.nets.MLPEnergy(dim=2, hidden=(16,))
    composed = ebm.SumEnergy(net, quadratic_energy)
    assert len(list(composed.parameters())) == len(list(net.parameters()))

    loss_fn = ebm.ContrastiveDivergence(ebm.LangevinDynamics(step_size=0.05, steps=5))
    out = loss_fn(composed, torch.randn(32, 2))
    out.loss.backward()
    assert all(p.grad is not None for p in net.parameters())
    assert all(p.requires_grad for p in net.parameters())


def test_compositions_nest():
    nested = ebm.TemperedEnergy(
        ebm.SumEnergy(quadratic_energy, ebm.MixtureEnergy(quadratic_energy)), 2.0
    )
    assert nested(torch.randn(8, 2)).shape == (8,)


def test_ensemble_energy_is_geometric_mean_gaussian():
    # Members N(0, σ_i²) → mean energy is N(0, 1/precision), precision = mean(1/σ_i²).
    sigmas = [1.0, 0.5, 2.0]
    members = [gaussian_energy(0.0, s**2) for s in sigmas]
    ens = ebm.EnsembleEnergy(*members)
    precision = sum(1 / s**2 for s in sigmas) / len(sigmas)
    samples = ebm.MALA(step_size=0.1, steps=800).sample(ens, torch.randn(6000, 2))
    assert (samples.var(0) - 1 / precision).abs().max().item() < 0.05
    # member_energies stacks per-member; forward is their mean.
    x = torch.randn(16, 2)
    me = ens.member_energies(x)
    assert me.shape == (16, 3)
    assert torch.allclose(ens(x), me.mean(dim=1))


def test_ensemble_disagreement_flags_ood():
    sigmas = [1.0, 0.5, 2.0]
    ens = ebm.EnsembleEnergy(*[gaussian_energy(0.0, s**2) for s in sigmas])
    x_in = torch.randn(2000, 2)
    x_out = 6.0 + torch.randn(2000, 2)
    d_in = ebm.eval.ensemble_disagreement(ens, x_in)
    d_out = ebm.eval.ensemble_disagreement(ens, x_out)
    assert d_in.shape == (2000,)
    assert d_out.mean() > 50 * d_in.mean()  # members diverge off-distribution
    # As an OOD score it separates the two sets almost perfectly.
    auroc = ebm.eval.ood_auroc(lambda z: ebm.eval.ensemble_disagreement(ens, z), x_in, x_out)
    assert auroc > 0.99
