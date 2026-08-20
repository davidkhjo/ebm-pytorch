"""Latent-variable EBM checked against the linear-Gaussian conjugate identities."""

import pytest
import torch
from torch import nn

import ebm

# Linear-Gaussian model: z ~ N(0, I), x | z ~ N(Wz, σ²I).
_W = torch.tensor([[1.2, 0.4], [0.3, 0.9]])
_SIG = 0.5


def _decoder(x, z):
    return 0.5 * ((x - z @ _W.t()) ** 2).sum(dim=1) / _SIG**2


def test_latent_ebm_recovers_the_gaussian_marginal():
    # Block Gibbs on the joint → x marginal is N(0, WWᵀ + σ²I).
    model = ebm.LatentEBM(_decoder, latent_dim=2)
    x, z = model.sample_joint(ebm.MALA(step_size=0.05, steps=5), torch.randn(5000, 2), steps=250)
    assert x.shape == (5000, 2) and z.shape == (5000, 2)
    assert not x.requires_grad
    marginal_cov = _W @ _W.t() + _SIG**2 * torch.eye(2)
    assert (torch.cov(x.T) - marginal_cov).abs().max().item() < 0.1


def test_latent_ebm_posterior_is_conjugate_gaussian():
    model = ebm.LatentEBM(_decoder, latent_dim=2)
    x0 = torch.tensor([[1.0, -0.5]]).repeat(4000, 1)
    # posterior_energy(x0) is a valid EnergyFn in z; sample it directly.
    z = torch.randn(4000, 2)
    sampler = ebm.MALA(step_size=0.03, steps=3)
    for _ in range(400):
        z = sampler.sample(model.posterior_energy(x0), z)
    sig_post = torch.linalg.inv(torch.eye(2) + _W.t() @ _W / _SIG**2)
    mu_post = (x0[:1] @ (_W / _SIG**2)) @ sig_post.t()
    assert (torch.cov(z.T) - sig_post).abs().max().item() < 0.05
    assert (z.mean(0) - mu_post[0]).abs().max().item() < 0.05


def test_latent_ebm_joint_custom_prior_and_registration():
    # joint_energy = prior + decoder; a custom prior is honored.
    prior = ebm.nets.GaussianMixtureEnergy(torch.tensor([[-2.0, 0.0], [2.0, 0.0]]), std=0.5)
    model = ebm.LatentEBM(_decoder, latent_dim=2, prior=prior)
    x, z = torch.randn(8, 2), torch.randn(8, 2)
    assert torch.allclose(model.joint_energy(x, z), prior(z) + _decoder(x, z))
    assert torch.allclose(model.prior_energy(z), prior(z))
    assert torch.allclose(model.decoder_energy(x, z), _decoder(x, z))
    assert torch.allclose(model.conditional_energy(z)(x), _decoder(x, z))
    assert torch.allclose(model.posterior_energy(x)(z), prior(z) + _decoder(x, z))

    # An nn.Module decoder is registered so its parameters train/freeze.
    class LinearDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Linear(2, 2, bias=False)

        def forward(self, x, z):
            return 0.5 * ((x - self.w(z)) ** 2).sum(dim=1)

    dec = LinearDecoder()
    m2 = ebm.LatentEBM(dec, latent_dim=2)
    assert any(p is dec.w.weight for p in m2.parameters())  # registered

    with pytest.raises(ValueError):
        ebm.LatentEBM(_decoder, latent_dim=0)


def test_latent_ebm_sample_returns_marginal_and_takes_z_init():
    model = ebm.LatentEBM(_decoder, latent_dim=2)
    x = model.sample(
        ebm.MALA(step_size=0.05, steps=3),
        torch.randn(64, 2),
        z_init=torch.zeros(64, 2),
        steps=5,
    )
    assert x.shape == (64, 2)
