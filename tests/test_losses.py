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
