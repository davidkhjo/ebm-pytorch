"""Gibbs-with-Gradients: correctness against closed-form discrete distributions."""

import torch
from torch import nn

import ebm


def bernoulli_init(shape):
    return torch.bernoulli(torch.full(shape, 0.5))


def test_gwg_recovers_independent_bernoulli_marginals():
    # E(x) = -theta.x  =>  p(x_i = 1) = sigmoid(theta_i), independent bits.
    theta = torch.tensor([2.0, 0.0, -1.0, 0.5])
    energy = lambda x: -(x * theta).sum(dim=1)  # noqa: E731
    sampler = ebm.GibbsWithGradients(steps=60)
    samples = sampler.sample(energy, bernoulli_init((2000, 4)))
    assert torch.allclose(samples.mean(0), torch.sigmoid(theta), atol=0.05)


def test_gwg_pairwise_coupling():
    # Ising pair: E = -J s1 s2 with s = 2x - 1  =>  P(x1 == x2) = sigmoid(2J).
    j = 1.0
    energy = lambda x: -j * (2 * x[:, 0] - 1) * (2 * x[:, 1] - 1)  # noqa: E731
    sampler = ebm.GibbsWithGradients(steps=40)
    samples = sampler.sample(energy, bernoulli_init((4000, 2)))
    agree = (samples[:, 0] == samples[:, 1]).float().mean()
    expected = torch.sigmoid(torch.tensor(2 * j))
    assert (agree - expected).abs().item() < 0.05


def test_gwg_output_binary_and_accept_rate():
    theta = torch.tensor([1.0, -1.0, 0.5, 0.0])
    energy = lambda x: -(x * theta).sum(dim=1)  # noqa: E731
    sampler = ebm.GibbsWithGradients(steps=20)
    samples = sampler.sample(energy, bernoulli_init((500, 4)))
    assert set(samples.unique().tolist()) <= {0.0, 1.0}
    # For a linear energy the first-order flip estimate is exact, so the MH
    # correction accepts almost always.
    assert 0.5 < sampler.last_accept_rate <= 1.0


def test_gwg_event_shape_and_module_energy():
    net = ebm.nets.MLPEnergy(dim=6, hidden=(16,))
    energy = ebm.EnergyModel(nn.Sequential(nn.Flatten(), net.net))
    sampler = ebm.GibbsWithGradients(steps=5)
    with torch.no_grad():
        samples = sampler.sample(energy, bernoulli_init((8, 2, 3)))
    assert samples.shape == (8, 2, 3)
    assert set(samples.unique().tolist()) <= {0.0, 1.0}
    assert all(p.requires_grad for p in energy.parameters())


def onehot_init(n, d, k, generator=None):
    return torch.nn.functional.one_hot(torch.randint(k, (n, d), generator=generator), k).float()


def test_categorical_gwg_recovers_independent_marginals():
    # E(x) = -sum theta * x over one-hot x  =>  p(x_i = k) = softmax(theta_i)_k.
    theta = torch.tensor([[1.0, 0.0, -1.0], [0.5, 0.5, -0.5]])
    energy = lambda x: -(x * theta).sum(dim=(1, 2))  # noqa: E731
    sampler = ebm.CategoricalGibbsWithGradients(steps=60)
    samples = sampler.sample(energy, onehot_init(2000, 2, 3))
    marginals = samples.mean(0)
    assert torch.allclose(marginals, torch.softmax(theta, dim=1), atol=0.05)


def test_categorical_gwg_pairwise_coupling():
    # Potts pair: E = -J * 1[c1 == c2] with K categories  =>
    # P(c1 == c2) = e^J / (e^J + K - 1).
    j, k = 1.0, 3
    energy = lambda x: -j * (x[:, 0] * x[:, 1]).sum(dim=1)  # noqa: E731
    sampler = ebm.CategoricalGibbsWithGradients(steps=50)
    samples = sampler.sample(energy, onehot_init(4000, 2, k))
    agree = (samples[:, 0].argmax(1) == samples[:, 1].argmax(1)).float().mean()
    expected = torch.e**j / (torch.e**j + k - 1)
    assert (agree - expected).abs().item() < 0.05


def test_categorical_gwg_output_onehot_and_accept_rate():
    theta = torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.5, -0.5]])
    energy = lambda x: -(x * theta).sum(dim=(1, 2))  # noqa: E731
    sampler = ebm.CategoricalGibbsWithGradients(steps=20)
    samples = sampler.sample(energy, onehot_init(500, 2, 3))
    assert set(samples.unique().tolist()) <= {0.0, 1.0}
    assert torch.all(samples.sum(dim=-1) == 1.0)  # exactly one-hot per position
    # Linear energy: the first-order category-change estimate is exact.
    assert 0.5 < sampler.last_accept_rate <= 1.0


def test_categorical_gwg_module_energy_and_no_grad():
    net = ebm.nets.MLPEnergy(dim=12, hidden=(16,))
    energy = ebm.EnergyModel(nn.Sequential(nn.Flatten(), net.net))
    sampler = ebm.CategoricalGibbsWithGradients(steps=5)
    with torch.no_grad():
        samples = sampler.sample(energy, onehot_init(8, 3, 4))
    assert samples.shape == (8, 3, 4)
    assert torch.all(samples.sum(dim=-1) == 1.0)
    assert all(p.requires_grad for p in energy.parameters())


class LinearBernoulliEnergy(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return -(x * self.theta).sum(dim=1)


def test_cd_gwg_trains_bernoulli_ebm():
    true_theta = torch.tensor([1.5, -1.0, 0.0, 0.5])
    data = torch.bernoulli(torch.sigmoid(true_theta).expand(2000, 4))

    energy = LinearBernoulliEnergy(4)
    loss_fn = ebm.ContrastiveDivergence(
        ebm.GibbsWithGradients(steps=10),
        buffer=ebm.ReplayBuffer(512, (4,), init_fn=bernoulli_init),
    )
    opt = torch.optim.Adam(energy.parameters(), lr=0.05)
    for _ in range(200):
        x = data[torch.randint(len(data), (128,))]
        out = loss_fn(energy, x)
        opt.zero_grad()
        out.loss.backward()
        opt.step()

    learned = torch.sigmoid(energy.theta.detach())
    assert torch.allclose(learned, torch.sigmoid(true_theta), atol=0.1)
