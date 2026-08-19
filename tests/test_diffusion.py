"""Variance-preserving diffusion: schedule, DDPM ancestral sampler, VP-DSM training.

Ground truth: for data N(0, σ_d²), the step-t marginal is N(0, v_t) with
v_t = ᾱ_t σ_d² + (1-ᾱ_t), so E*(x, σ_t) = ‖x‖²/(2 v_t) is the exact energy, and the
reverse process must return samples with std ≈ σ_d.
"""

import torch

import ebm


def _gaussian_energy(sd):
    def energy(x, sigma):
        v = (sd**2 + sigma**2 * (1 - sd**2)).reshape(-1, *([1] * (x.dim() - 1)))
        return (x.pow(2) / (2 * v)).flatten(1).sum(dim=1)

    return energy


def test_vp_schedule_properties():
    for schedule in ("linear", "cosine"):
        sch = ebm.VPSchedule(num_steps=200, schedule=schedule)
        assert (sch.alpha_bar.diff() < 0).all()  # ᾱ strictly decreasing
        assert (sch.sigma.diff() > 0).all()  # noise level increasing
        assert sch.alpha_bar[0] < 1.0 and sch.alpha_bar[-1] > 0
    # q_sample of N(0,I) at step t has variance ᾱ + (1-ᾱ) = 1 (variance-preserving).
    sch = ebm.VPSchedule(num_steps=1000)
    x0 = torch.randn(20000, 1)
    t = torch.full((20000,), 500)
    xt = sch.q_sample(x0, t, torch.randn_like(x0))
    assert abs(xt.var().item() - 1.0) < 0.05


def test_ddpm_ancestral_recovers_gaussian():
    sd = 1.7
    sch = ebm.VPSchedule(num_steps=1000)
    samples = ebm.DDPMAncestralSampler(sch).sample(_gaussian_energy(sd), torch.randn(6000, 2))
    assert abs(samples.std().item() - sd) < 0.1
    assert not samples.requires_grad


def test_vp_dsm_trains_and_samples():
    sd = 1.5
    data = sd * torch.randn(8000, 1)
    sch = ebm.VPSchedule(num_steps=1000)
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=1, hidden=(128, 128))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = ebm.VPDenoisingScoreMatching(sch)
    for _ in range(2500):
        batch = data[torch.randint(0, len(data), (256,))]
        loss = loss_fn(net, batch).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    samples = ebm.DDPMAncestralSampler(sch).sample(net, torch.randn(4000, 1))
    assert abs(samples.std().item() - sd) < 0.25
