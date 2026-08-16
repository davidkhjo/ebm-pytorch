import pytest
import torch
from torch import nn

import ebm


class ScaledQuadratic(nn.Module):
    """E(x) = a * ||x||^2 / 2 with learnable a; true N(0, I) score at a=1."""

    def __init__(self, a: float):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a))

    def forward(self, x):
        return 0.5 * self.a * x.pow(2).flatten(1).sum(dim=1)


def test_cd_negatives_detached_and_grads_flow():
    net = ebm.nets.MLPEnergy(dim=2, hidden=(16,))
    loss_fn = ebm.ContrastiveDivergence(ebm.LangevinDynamics(step_size=0.05, steps=10))
    out = loss_fn(net, torch.randn(32, 2))
    assert out.x_neg is not None and not out.x_neg.requires_grad
    out.loss.backward()
    assert all(p.grad is not None for p in net.parameters())
    assert all(p.requires_grad for p in net.parameters())
    assert {"energy_pos", "energy_neg", "energy_gap"} <= out.metrics.keys()


def test_cd_with_buffer_persists_chains():
    buf = ebm.ReplayBuffer(capacity=64, shape=(2,), reinit_prob=0.0)
    before = buf.data.clone()
    loss_fn = ebm.ContrastiveDivergence(ebm.LangevinDynamics(step_size=0.05, steps=5), buffer=buf)
    loss_fn(ebm.nets.MLPEnergy(dim=2, hidden=(8,)), torch.randn(16, 2))
    assert not torch.equal(before, buf.data)


def test_cd_energy_reg_increases_loss_magnitude():
    net = ScaledQuadratic(1.0)
    sampler = ebm.LangevinDynamics(step_size=0.05, steps=5)
    torch.manual_seed(1)
    plain = ebm.ContrastiveDivergence(sampler)(net, torch.randn(64, 2))
    torch.manual_seed(1)
    reg = ebm.ContrastiveDivergence(sampler, energy_reg=1.0)(net, torch.randn(64, 2))
    assert reg.loss.item() > plain.loss.item()


def test_dsm_minimized_near_true_smoothed_score():
    # Data N(0, I), sigma=0.5 => smoothed dist N(0, 1.25 I), true a = 1/1.25 = 0.8.
    x = torch.randn(4000, 2)
    loss_fn = ebm.DenoisingScoreMatching(sigma=0.5)
    losses = {}
    for a in (0.3, 0.8, 2.0):
        torch.manual_seed(7)
        losses[a] = loss_fn(ScaledQuadratic(a), x).loss.item()
    assert losses[0.8] < losses[0.3]
    assert losses[0.8] < losses[2.0]


def test_ssm_minimized_at_true_scale():
    # Data N(0, I): SSM objective for E = a||x||^2/2 is minimized at a=1.
    x = torch.randn(4000, 2)
    loss_fn = ebm.SlicedScoreMatching(n_projections=4)
    losses = {}
    for a in (0.5, 1.0, 2.0):
        torch.manual_seed(7)
        losses[a] = loss_fn(ScaledQuadratic(a), x).loss.item()
    assert losses[1.0] < losses[0.5]
    assert losses[1.0] < losses[2.0]


def test_ssm_backward_flows():
    net = ebm.nets.MLPEnergy(dim=2, hidden=(16,))
    out = ebm.SlicedScoreMatching()(net, torch.randn(32, 2))
    out.loss.backward()
    # Score losses cannot see the output bias (energy is defined up to a constant).
    grads = [p.grad for name, p in net.named_parameters() if "weight" in name]
    assert all(g is not None for g in grads)


class _FreeQuadratic(nn.Module):
    """E(x) = ½ xᵀ M x with a full symmetric learnable M (recovers A = Σ⁻¹)."""

    def __init__(self, d):
        super().__init__()
        self.P = nn.Parameter(0.1 * torch.randn(d, d) + torch.eye(d))

    def matrix(self):
        return (self.P + self.P.t()) / 2

    def forward(self, x):
        return 0.5 * ((x @ self.matrix()) * x).sum(dim=1)


def test_exact_score_matching_recovers_precision_matrix():
    # Data N(0, Σ): the exact-SM optimum for E = ½xᵀMx is M = Σ⁻¹ = A_true.
    d = 3
    a_true = torch.tensor([[2.0, 0.5, 0.0], [0.5, 1.5, 0.3], [0.0, 0.3, 1.0]])
    chol = torch.linalg.cholesky(torch.linalg.inv(a_true))
    data = torch.randn(4000, d) @ chol.t()

    energy = _FreeQuadratic(d)
    esm = ebm.ExactScoreMatching()
    opt = torch.optim.Adam(energy.parameters(), lr=0.02)
    for _ in range(1500):
        idx = torch.randint(0, len(data), (512,))
        loss = esm(energy, data[idx]).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert (energy.matrix().detach() - a_true).abs().max().item() < 0.1


def test_exact_score_matching_agrees_with_sliced_on_a_gaussian():
    # ESM is the exact objective SSM estimates; their values match in expectation.
    x = torch.randn(2000, 3)
    energy = ScaledQuadratic(1.3)
    exact = ebm.ExactScoreMatching()(energy, x).loss.item()
    sliced = ebm.SlicedScoreMatching(n_projections=50, vr=True)(energy, x).loss.item()
    assert abs(exact - sliced) < 0.1


class _Quadratic1D(nn.Module):
    """E(x) = a (x - b)² — its ED minimizer on N(μ, s²) is a=1/(2s²), b=μ."""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.6))
        self.b = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return (self.a * (x - self.b).pow(2)).sum(dim=1)


def test_energy_discrepancy_recovers_gaussian():
    mu, s = 1.5, 1.2
    data = mu + s * torch.randn(8000, 1)
    energy = _Quadratic1D()
    ed = ebm.EnergyDiscrepancy(sigma=1.0, m_particles=16, w_stable=1.0)
    opt = torch.optim.Adam(energy.parameters(), lr=0.01)
    for _ in range(3000):
        idx = torch.randint(0, len(data), (512,))
        loss = ed(energy, data[idx]).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(energy.a.item() - 1 / (2 * s**2)) < 0.05
    assert abs(energy.b.item() - mu) < 0.1


def test_energy_discrepancy_shapes_and_validation():
    net = ebm.nets.MLPEnergy(dim=2, hidden=(16,))
    out = ebm.EnergyDiscrepancy(m_particles=4)(net, torch.randn(8, 2))
    assert out.x_neg.shape == (8 * 4, 2)  # M contrastive draws per point
    out.loss.backward()
    assert all(p.grad is not None for n, p in net.named_parameters() if "weight" in n)

    # No sampler, so the loss is finite with w>0 even for a wild energy.
    assert torch.isfinite(ebm.EnergyDiscrepancy(w_stable=0.0)(net, torch.randn(8, 2)).loss)
    for bad in (dict(sigma=0.0), dict(m_particles=0), dict(w_stable=-1.0)):
        with pytest.raises(ValueError):
            ebm.EnergyDiscrepancy(**bad)


def _train_rbm_marginals(loss_fn, q, steps=1500):
    rbm = ebm.nets.RBM(len(q), 8)
    opt = torch.optim.Adam(rbm.parameters(), lr=0.05)
    for _ in range(steps):
        v = torch.bernoulli(q.expand(256, len(q)))
        loss = loss_fn(rbm, v).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    samples = torch.bernoulli(torch.full((20000, len(q)), 0.5))
    for _ in range(300):
        samples = rbm.gibbs_step(samples)
    return samples.mean(0)


def test_pseudolikelihood_recovers_independent_bernoulli():
    q = torch.tensor([0.2, 0.5, 0.8, 0.35])
    marginals = _train_rbm_marginals(ebm.PseudoLikelihood(), q)
    assert (marginals - q).abs().max().item() < 0.05


def test_ratio_matching_recovers_independent_bernoulli():
    q = torch.tensor([0.2, 0.5, 0.8, 0.35])
    marginals = _train_rbm_marginals(ebm.RatioMatching(), q)
    assert (marginals - q).abs().max().item() < 0.05


def test_pseudolikelihood_recovers_tiny_rbm_pmf():
    # Data from two noisy 3-bit prototypes; PL must recover the full pmf, which
    # we read off exactly via softmax(-F) (i.e. using the exact log Z).
    import itertools

    protos = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    idx = torch.randint(0, 2, (4000,))
    flip = (torch.rand(4000, 3) < 0.1).float()
    data = (protos[idx] - flip).abs()

    rbm = ebm.nets.RBM(3, 6)
    opt = torch.optim.Adam(rbm.parameters(), lr=0.05)
    for _ in range(2000):
        v = data[torch.randint(0, len(data), (256,))]
        loss = ebm.PseudoLikelihood()(rbm, v).loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    states = torch.tensor(list(itertools.product([0.0, 1.0], repeat=3)))
    model_pmf = torch.softmax(-rbm(states), dim=0)
    codes = (data @ torch.tensor([4.0, 2.0, 1.0])).long()
    data_pmf = torch.bincount(codes, minlength=8).float()
    data_pmf /= data_pmf.sum()
    assert (model_pmf - data_pmf).abs().max().item() < 0.05


def test_discrete_losses_are_mcmc_free_and_flow_gradients():
    rbm = ebm.nets.RBM(4, 6)
    v = torch.bernoulli(torch.full((16, 4), 0.5))
    for loss_fn in (ebm.PseudoLikelihood(), ebm.RatioMatching()):
        out = loss_fn(rbm, v)
        assert out.x_neg is None  # no negatives — nothing was sampled
        out.loss.backward()
        assert all(p.grad is not None for p in rbm.parameters())
        rbm.zero_grad()


class _CategoricalEnergy(nn.Module):
    """Small learnable energy on one-hot (B, D, K) data."""

    def __init__(self, d, k):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d * k, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1)).squeeze(-1)


def _categorical_states(d, k):
    import itertools

    return torch.stack(
        [
            torch.nn.functional.one_hot(torch.tensor(c), k).float()
            for c in itertools.product(range(k), repeat=d)
        ]
    )


def _train_csm_kl(loss_fn, d=2, k=3, steps=2000):
    states = _categorical_states(d, k)
    torch.manual_seed(3)
    data_pmf = torch.softmax(torch.randn(k**d) * 1.5, dim=0)
    data = states[torch.multinomial(data_pmf, 20000, replacement=True)]

    torch.manual_seed(0)
    energy = _CategoricalEnergy(d, k)
    opt = torch.optim.Adam(energy.parameters(), lr=0.01)
    for _ in range(steps):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = loss_fn(energy, batch).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    model_pmf = torch.softmax(-energy(states), dim=0)
    return (
        (data_pmf * (data_pmf.clamp_min(1e-9).log() - model_pmf.clamp_min(1e-9).log())).sum().item()
    )


def test_concrete_score_matching_recovers_categorical_pmf():
    # The two-term objective recovers the full pmf on an enumerable categorical model.
    assert _train_csm_kl(ebm.ConcreteScoreMatching()) < 0.05


def test_concrete_score_matching_two_term_beats_naive():
    # Dropping the cross term is inconsistent — it inverts the distribution.
    class _Naive(ebm.ConcreteScoreMatching):
        def forward(self, energy, x):
            b, k = x.shape[0], x.shape[-1]
            flat = x.reshape(b, -1, k)
            e_x = energy(x)
            total = x.new_zeros(b)
            for site in range(flat.shape[1]):
                for cat in range(k):
                    y = flat.clone()
                    y[:, site, :] = 0.0
                    y[:, site, cat] = 1.0
                    diff = (e_x - energy(y.reshape_as(x))).clamp(-15, 15)
                    total = total + 0.5 * (torch.exp(diff) - 1) ** 2  # no cross term
            return ebm.LossOutput(loss=total.mean())

    assert _train_csm_kl(ebm.ConcreteScoreMatching()) < 0.5 * _train_csm_kl(_Naive())


def test_nce_log_z_learns():
    net = ScaledQuadratic(1.0)
    loss_fn = ebm.NoiseContrastiveEstimation()
    out = loss_fn(net, torch.randn(64, 2))
    out.loss.backward()
    assert loss_fn.log_z.grad is not None
    assert net.a.grad is not None
    assert "log_z" in out.metrics


def test_nce_optimum_for_matching_model_and_noise():
    # Model N(0, I) with log_z = true log Z, noise N(0, I): the classifier is at
    # chance, and the loss equals 2*log 2.
    import math

    net = ScaledQuadratic(1.0)
    loss_fn = ebm.NoiseContrastiveEstimation()
    with torch.no_grad():
        loss_fn.log_z.fill_(math.log(2 * math.pi))  # log Z of N(0, I) in 2D
    out = loss_fn(net, torch.randn(2000, 2))
    assert abs(out.loss.item() - 2 * math.log(2)) < 0.05
