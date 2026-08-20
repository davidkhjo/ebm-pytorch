"""Direct unit tests for the energy networks (shapes + lattice closed forms)."""

import pytest
import torch

import ebm
from ebm.nets import _GaussianFourierFeatures, _ResBlock


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


def test_funnel_energy_closed_form():
    # E(x) = 0.5 * (v^2 / v_scale^2 + e^{-v} ||neck||^2 + n * v).
    energy = ebm.nets.FunnelEnergy(dim=3, v_scale=3.0)
    x = torch.tensor([[0.0, 1.0, 2.0], [1.5, -1.0, 0.0]])
    v = x[:, 0]
    neck_sq = x[:, 1:].pow(2).sum(dim=1)
    expected = 0.5 * (v.pow(2) / 9.0 + torch.exp(-v) * neck_sq + 2 * v)
    assert torch.allclose(energy(x), expected)

    # At the origin only the v-quadratic survives (e^0 * 0 + 0); here E = 0.
    assert torch.allclose(energy(torch.zeros(1, 3)), torch.zeros(1))
    with pytest.raises(ValueError):
        ebm.nets.FunnelEnergy(dim=1)  # needs a neck
    with pytest.raises(ValueError):
        energy(torch.zeros(4, 2))  # wrong width


def test_gaussian_mixture_energy_closed_form():
    means = torch.tensor([[-4.0, 0.0], [4.0, 0.0]])
    energy = ebm.nets.GaussianMixtureEnergy(means, std=0.5)

    # Deep inside one well-separated mode the other component is negligible, so
    # E(x) ≈ ||x - μ||^2 / (2σ²) - log(w) with equal unit weights (log 1 = 0).
    x = torch.tensor([[-4.3, 0.2]])
    near = ((x - means[0]).pow(2).sum() / (2 * 0.5**2)).reshape(1)
    assert torch.allclose(
        energy(x),
        -torch.logsumexp(
            torch.tensor([-near.item(), -(((x - means[1]).pow(2).sum()) / (2 * 0.5**2)).item()]), 0
        ).reshape(1),
    )

    # Symmetric target: energy at mirrored points is equal.
    assert torch.allclose(energy(torch.tensor([[-4.0, 0.0]])), energy(torch.tensor([[4.0, 0.0]])))
    with pytest.raises(ValueError):
        ebm.nets.GaussianMixtureEnergy(means, weights=[1.0], std=1.0)  # weight count
    with pytest.raises(ValueError):
        ebm.nets.GaussianMixtureEnergy(means, std=0.0)  # bad std


def test_banana_energy_closed_form_and_exact_sampler():
    energy = ebm.nets.BananaEnergy(b=0.5, sigma=(1.0, 1.0))
    x = torch.tensor([[0.5, 0.3]])
    warp = 0.3 - 0.5 * (0.25 - 1.0)
    expected = 0.25 / 2 + warp**2 / 2
    assert torch.allclose(energy(x), torch.tensor([expected]), atol=1e-5)

    # Exact draws have closed-form variances: Var[x0]=σ0²=1, Var[x1]=b²·2σ0⁴+σ1²=1.5.
    samples = energy.exact_sample(50000)
    assert abs(samples[:, 0].var().item() - 1.0) < 0.05
    assert abs(samples[:, 1].var().item() - 1.5) < 0.06

    with pytest.raises(ValueError):
        ebm.nets.BananaEnergy(sigma=(0.0, 1.0))
    with pytest.raises(ValueError):
        energy(torch.zeros(4, 3))  # must be (B, 2)


def test_resnet_energy_shape_and_batch_independence():
    net = ebm.nets.ResNetEnergy(in_channels=1, channels=(16, 32))
    x = torch.randn(4, 1, 16, 16)
    out = net(x)
    assert out.shape == (4,)
    # No normalization across the batch: a sample's energy is independent of the rest.
    assert torch.allclose(out[2], net(x[2:3])[0], atol=1e-6)


def test_resnet_block_is_identity_at_init():
    # Zero-initialized second conv + identity skip → the block is the identity.
    block = _ResBlock(8, 8, spectral_norm=False, downsample=False)
    y = torch.randn(2, 8, 12, 12)
    assert torch.allclose(block(y), y, atol=1e-6)


def test_resnet_energy_spectral_norm_trains():
    net = ebm.nets.ResNetEnergy(in_channels=3, channels=(16, 32), spectral_norm=True)
    out = net(torch.randn(2, 3, 32, 32))
    assert out.shape == (2,)
    out.sum().backward()
    assert all(p.grad is not None for p in net.parameters() if p.requires_grad)


def test_affine_coupling_flow_logdet_identity_and_roundtrip():
    flow = ebm.nets.AffineCouplingFlow(dim=3, n_layers=6, hidden=32)
    x = torch.randn(5, 3)
    z, logdet = flow.transform(x)
    # Change-of-variables log|det| equals the brute-force Jacobian log-det.
    for i in range(len(x)):
        jac = torch.autograd.functional.jacobian(
            lambda v: flow.transform(v.unsqueeze(0))[0].squeeze(0), x[i]
        )
        assert abs(logdet[i].item() - torch.linalg.slogdet(jac)[1].item()) < 1e-4
    # The flow is exactly invertible, and forward is the energy -log p.
    assert torch.allclose(flow.inverse(z), x, atol=1e-4)
    assert torch.allclose(flow(x), -flow.log_prob(x))
    with pytest.raises(ValueError):
        ebm.nets.AffineCouplingFlow(dim=1)


def test_affine_coupling_flow_fits_a_gaussian():
    import math

    d = 2
    mu = torch.tensor([1.0, -0.5])
    cov = torch.tensor([[2.0, 0.6], [0.6, 1.0]])
    chol = torch.linalg.cholesky(cov)
    data = torch.randn(8000, d) @ chol.t() + mu

    flow = ebm.nets.AffineCouplingFlow(dim=d, n_layers=8, hidden=64)
    opt = torch.optim.Adam(flow.parameters(), lr=5e-3)
    for _ in range(3000):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = -flow.log_prob(batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    test = torch.randn(4000, d) @ chol.t() + mu
    prec = torch.linalg.inv(cov)
    c = test - mu
    analytic = (
        -0.5 * ((c @ prec) * c).sum(dim=1)
        - 0.5 * d * math.log(2 * math.pi)
        - 0.5 * torch.linalg.slogdet(cov)[1]
    )
    assert (flow.log_prob(test) - analytic).abs().mean().item() < 0.2
    samples = flow.sample(8000)
    assert (samples.mean(0) - mu).abs().max().item() < 0.2
    assert (torch.cov(samples.T) - cov).abs().max().item() < 0.3


def test_spline_flow_invertible_and_logdet_across_domain():
    # Run in double precision: the rational-quadratic map is *exactly* invertible,
    # so this pins the math (float32 loses a few digits per layer, as all RQ-spline
    # flows do). Perturb the zero-init net so the spline is genuinely nonlinear.
    flow = ebm.nets.NeuralSplineCouplingFlow(dim=3, n_layers=4, num_bins=8, bound=3.0).double()
    with torch.no_grad():
        for p in flow.parameters():
            p.add_(0.4 * torch.randn_like(p))
    x = (6 * torch.randn(6, 3)).double()  # spans well past the [-3, 3] bound into the tails
    z, logdet = flow.transform(x)
    assert torch.allclose(flow.inverse(z), x, atol=1e-8)  # exact roundtrip incl. tails
    for i in range(len(x)):
        jac = torch.autograd.functional.jacobian(
            lambda v: flow.transform(v.unsqueeze(0))[0].squeeze(0), x[i]
        )
        assert abs(logdet[i].item() - torch.linalg.slogdet(jac)[1].item()) < 1e-8
    assert torch.allclose(flow(x), -flow.log_prob(x))
    with pytest.raises(ValueError):
        ebm.nets.NeuralSplineCouplingFlow(dim=1)


def test_spline_flow_fits_a_gaussian():
    import math

    d = 2
    # The spline reshapes only within [-bound, bound] (identity tails), so the
    # target's mass must sit inside it — here a zero-centred Gaussian well within ±3.
    mu = torch.zeros(d)
    cov = torch.tensor([[0.7, 0.3], [0.3, 0.6]])
    chol = torch.linalg.cholesky(cov)
    data = torch.randn(8000, d) @ chol.t() + mu

    flow = ebm.nets.NeuralSplineCouplingFlow(dim=d, n_layers=6, num_bins=8, bound=3.0)
    opt = torch.optim.Adam(flow.parameters(), lr=5e-3)
    for _ in range(2000):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = -flow.log_prob(batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    test = torch.randn(4000, d) @ chol.t() + mu
    prec = torch.linalg.inv(cov)
    c = test - mu
    analytic = (
        -0.5 * ((c @ prec) * c).sum(dim=1)
        - 0.5 * d * math.log(2 * math.pi)
        - 0.5 * torch.linalg.slogdet(cov)[1]
    )
    assert (flow.log_prob(test) - analytic).abs().mean().item() < 0.2
    # It is practically invertible in float32 after training.
    z, _ = flow.transform(test)
    assert torch.allclose(flow.inverse(z), test, atol=1e-2)


def test_spline_flow_fits_two_moons():
    data = ebm.datasets.two_moons(8000)
    flow = ebm.nets.NeuralSplineCouplingFlow(dim=2, n_layers=6, num_bins=8, bound=3.0)
    opt = torch.optim.Adam(flow.parameters(), lr=5e-3)
    for _ in range(2000):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = -flow.log_prob(batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    samples = flow.sample(8000)
    # Samples land on the data manifold (both crescents), far closer than a Gaussian.
    to_data = ebm.eval.mmd(samples, data)
    to_normal = ebm.eval.mmd(torch.randn(8000, 2), data)
    assert to_data < 0.01
    assert to_data < 0.2 * to_normal
    # Both arcs are covered (the moons split above/below the x-axis near the centre).
    upper = ((samples[:, 0].abs() < 0.5) & (samples[:, 1] > 0.3)).float().mean()
    lower = ((samples[:, 0].abs() < 0.5) & (samples[:, 1] < -0.0)).float().mean()
    assert upper > 0.02 and lower > 0.02
