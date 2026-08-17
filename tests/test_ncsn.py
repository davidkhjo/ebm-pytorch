"""Multi-sigma DSM and annealed Langevin, checked against analytic smoothed scores.

For data ~ N(0, I), the sigma-smoothed distribution is N(0, (1 + sigma^2) I),
with energy E*(x, sigma) = ||x||^2 / (2 (1 + sigma^2)) and score
s*(x, sigma) = -x / (1 + sigma^2).
"""

import torch

import ebm


def smoothed_energy(x, sigma):
    var = 1 + sigma.reshape(-1, *([1] * (x.dim() - 1))) ** 2
    return (x.pow(2) / (2 * var)).reshape(x.shape[0], -1).sum(dim=1)


def test_multisigma_dsm_minimized_at_true_smoothed_energy():
    x = torch.randn(4000, 2)
    loss_fn = ebm.MultiSigmaDenoisingScoreMatching(ebm.geometric_sigmas(2.0, 0.1, 5))

    candidates = {
        "true": smoothed_energy,
        "unsmoothed": lambda x, sigma: 0.5 * x.pow(2).sum(dim=1),
        "wrong_scale": lambda x, sigma: smoothed_energy(x, sigma) / 4,
    }
    losses = {}
    for name, fn in candidates.items():
        torch.manual_seed(7)
        losses[name] = loss_fn(fn, x).loss.item()
    assert losses["true"] < losses["unsmoothed"]
    assert losses["true"] < losses["wrong_scale"]


def test_multisigma_dsm_backward_and_validation():
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=2, hidden=(16,))
    loss_fn = ebm.MultiSigmaDenoisingScoreMatching(ebm.geometric_sigmas(1.0, 0.1, 4))
    out = loss_fn(net, torch.randn(32, 2))
    assert out.x_neg is None and "loss" in out.metrics
    out.loss.backward()
    grads = [p.grad for name, p in net.named_parameters() if "weight" in name]
    assert all(g is not None for g in grads)

    for bad in (torch.tensor([[1.0]]), torch.tensor([1.0, -0.5])):
        try:
            ebm.MultiSigmaDenoisingScoreMatching(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_noise_conditional_nets_shapes():
    mlp = ebm.nets.NoiseConditionalMLPEnergy(dim=3, hidden=(16,))
    x = torch.randn(8, 3)
    for sigma in (0.5, torch.tensor(0.5), torch.full((8,), 0.5)):
        assert mlp(x, sigma).shape == (8,)
    # Energy must actually depend on sigma.
    assert not torch.allclose(mlp(x, 0.1), mlp(x, 2.0))

    conv = ebm.nets.NoiseConditionalConvEnergy(in_channels=1, channels=(8, 16))
    assert conv(torch.randn(4, 1, 16, 16), 0.3).shape == (4,)


def test_annealed_langevin_targets_smoothed_gaussian():
    sigmas = ebm.geometric_sigmas(2.0, 0.1, 10)
    sampler = ebm.AnnealedLangevinDynamics(sigmas, step_size=5e-3, steps_per_sigma=100)
    samples = sampler.sample(smoothed_energy, 3 * torch.randn(2000, 2))
    # Final level targets N(0, (1 + 0.01) I).
    assert samples.mean().abs().item() < 0.1
    assert (samples.std() - 1.0).abs().item() < 0.15
    assert not samples.requires_grad

    traj = sampler.sample(smoothed_energy, torch.randn(4, 2), steps=3, return_trajectory=True)
    assert traj.shape == (31, 4, 2)

    for bad in (torch.tensor([0.1, 2.0]), torch.tensor([1.0, 1.0])):
        try:
            ebm.AnnealedLangevinDynamics(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for non-descending sigmas")


def test_ncsn_end_to_end():
    data = torch.randn(4096, 1)
    sigmas = ebm.geometric_sigmas(2.0, 0.1, 5)
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=1, hidden=(64, 64))
    loss_fn = ebm.MultiSigmaDenoisingScoreMatching(sigmas)

    trainer = ebm.Trainer(net, loss_fn, lr=1e-3, device="cpu")
    history = trainer.fit(data, steps=400, batch_size=128, verbose=False)
    first = sum(h["loss"] for h in history[:20]) / 20
    last = sum(h["loss"] for h in history[-20:]) / 20
    assert last < first

    # Trained score tracks the analytic smoothed score at both ends of the ladder.
    grid = torch.linspace(-3, 3, 200).reshape(-1, 1)
    for sigma in (sigmas[0], sigmas[-1]):
        s_model = ebm.score(lambda z, s=sigma: net(z, s.expand(len(z))), grid).squeeze(1)
        s_true = (-grid / (1 + sigma**2)).squeeze(1)
        cos = torch.nn.functional.cosine_similarity(s_model, s_true, dim=0)
        assert cos.item() > 0.9

    sampler = ebm.AnnealedLangevinDynamics(sigmas, step_size=5e-3, steps_per_sigma=60)
    samples = sampler.sample(net, torch.randn(1000, 1) * 3)
    assert samples.mean().abs().item() < 0.25
    assert (samples.std() - 1.0).abs().item() < 0.3


def test_probability_flow_ode_recovers_gaussian_and_is_deterministic():
    sigmas = ebm.geometric_sigmas(10.0, 0.01, 500)
    x0 = torch.randn(4000, 2) * float(sigmas[0])  # ~ N(0, sigma_max^2 I)
    ode = ebm.ProbabilityFlowODE(sigmas)
    out = ode.sample(smoothed_energy, x0)
    assert out.mean().abs().item() < 0.05
    assert (out.std() - 1.0).abs().item() < 0.05
    # No injected noise → the map is deterministic.
    assert torch.allclose(out, ode.sample(smoothed_energy, x0))


def test_predictor_corrector_recovers_gaussian():
    sigmas = ebm.geometric_sigmas(10.0, 0.01, 500)
    x0 = torch.randn(4000, 2) * float(sigmas[0])
    out = ebm.PredictorCorrector(sigmas, n_corrector=1, snr=0.16).sample(smoothed_energy, x0)
    assert out.mean().abs().item() < 0.05
    assert (out.std() - 1.0).abs().item() < 0.08


def test_score_sde_samplers_validate_ladder():
    import pytest

    with pytest.raises(ValueError):
        ebm.ProbabilityFlowODE(torch.tensor([0.1, 0.5]))  # not decreasing
    with pytest.raises(ValueError):
        ebm.PredictorCorrector(torch.tensor([1.0]))  # need >= 2 levels


def test_pf_ode_log_likelihood_matches_gaussian_density():
    import math

    d = 4
    x = torch.randn(2000, d)
    analytic = -0.5 * x.pow(2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)  # log N(x;0,I)
    sigmas = ebm.geometric_sigmas(10.0, 0.01, 512)
    ll = ebm.eval.pf_ode_log_likelihood(smoothed_energy, x, sigmas)
    assert (ll - analytic).abs().mean().item() < 0.15

    # A point at the mode is far more likely than a distant outlier.
    near = ebm.eval.pf_ode_log_likelihood(smoothed_energy, torch.zeros(1, d), sigmas)
    far = ebm.eval.pf_ode_log_likelihood(smoothed_energy, 4 * torch.ones(1, d), sigmas)
    assert near.item() > far.item() + 10

    import pytest

    with pytest.raises(ValueError):
        ebm.eval.pf_ode_log_likelihood(smoothed_energy, x, torch.tensor([0.1, 0.5]))  # ascending
