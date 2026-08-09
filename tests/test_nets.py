"""Direct unit tests for the energy networks (shapes + lattice closed forms)."""

import pytest
import torch

import ebm
from ebm.nets import _GaussianFourierFeatures


def test_energy_net_output_shapes():
    b = 4
    assert ebm.nets.MLPEnergy(dim=3)(torch.randn(b, 3)).shape == (b,)
    assert ebm.nets.ConvEnergy(in_channels=3)(torch.randn(b, 3, 32, 32)).shape == (b,)
    clf = ebm.nets.ConvClassifier(num_classes=7, in_channels=3, image_size=32)
    assert clf(torch.randn(b, 3, 32, 32)).shape == (b, 7)
    nc = ebm.nets.NoiseConditionalMLPEnergy(dim=2)
    assert nc(torch.randn(b, 2), torch.full((b,), 0.5)).shape == (b,)


def test_ising_energy_closed_form():
    # E(x) = -J * sum_<i,j> s_i s_j over right + down neighbors, s = 2x - 1.
    energy = ebm.nets.IsingEnergy(coupling=0.5)

    # All spins +1: 2 right + 2 down aligned pairs, each product +1 -> E = -4J.
    ones = torch.ones(1, 2, 2)
    assert torch.allclose(energy(ones), torch.tensor([-2.0]))

    # Checkerboard: all 4 neighbor pairs anti-aligned (product -1) -> E = +4J.
    checker = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    assert torch.allclose(energy(checker), torch.tensor([2.0]))

    with pytest.raises(ValueError):
        energy(torch.ones(1, 2, 2, 2))  # wrong rank


def test_potts_energy_closed_form():
    # E(x) = -J * sum_<i,j> 1[c_i == c_j]; one-hot dot product over right + down.
    energy = ebm.nets.PottsEnergy(coupling=0.5)
    k = 3

    def lattice(colors):
        idx = torch.tensor(colors).reshape(1, 2, 2)
        return torch.nn.functional.one_hot(idx, k).float()

    # Single color: all 4 neighbor pairs match -> E = -4J.
    assert torch.allclose(energy(lattice([[0, 0], [0, 0]])), torch.tensor([-2.0]))

    # Two-color checkerboard: no adjacent pair matches -> E = 0.
    assert torch.allclose(energy(lattice([[0, 1], [1, 0]])), torch.tensor([0.0]))

    with pytest.raises(ValueError):
        energy(torch.ones(1, 2, 2))  # missing category axis


def test_gaussian_fourier_features_shape_and_even_dim():
    embed = _GaussianFourierFeatures(embed_dim=16)
    out = embed(torch.rand(5).clamp_min(1e-3))
    assert out.shape == (5, 16)
    with pytest.raises(ValueError):
        _GaussianFourierFeatures(embed_dim=15)  # must be even
