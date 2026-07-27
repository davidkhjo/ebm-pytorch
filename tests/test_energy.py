import torch
from torch import nn

import ebm
from tests.conftest import quadratic_energy


def test_score_of_quadratic_is_negative_x():
    x = torch.randn(64, 3)
    s = ebm.score(quadratic_energy, x)
    assert torch.allclose(s, -x, atol=1e-5)


def test_score_create_graph_flows_to_params():
    net = ebm.nets.MLPEnergy(dim=2, hidden=(16,))
    x = torch.randn(8, 2)
    s = ebm.score(net, x, create_graph=True)
    s.pow(2).sum().backward()
    # The output bias correctly gets no gradient (energy is defined up to a
    # constant, which the score cannot see) — weights must all get gradients.
    grads = [p.grad for name, p in net.named_parameters() if "weight" in name]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_energy_model_squeezes_output():
    net = nn.Linear(4, 1)
    model = ebm.EnergyModel(net)
    e = model(torch.randn(10, 4))
    assert e.shape == (10,)


def test_mlp_and_conv_energy_shapes():
    mlp = ebm.nets.MLPEnergy(dim=5, spectral_norm=True)
    assert mlp(torch.randn(7, 5)).shape == (7,)
    conv = ebm.nets.ConvEnergy(in_channels=1, channels=(8, 16))
    assert conv(torch.randn(3, 1, 32, 32)).shape == (3,)
