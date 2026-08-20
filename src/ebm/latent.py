"""Latent-variable energy-based models: a joint ``E(x, z)`` sampled by block Gibbs."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from ebm._functional import flat_sum
from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler

DecoderEnergyFn = Callable[[Tensor, Tensor], Tensor]
"""A conditional energy ``(x: (B, *shape), z: (B, latent_dim)) -> (B,)`` — ``E(x | z)``."""


def _standard_normal_energy(z: Tensor) -> Tensor:
    return 0.5 * flat_sum(z.pow(2))


class LatentEBM(nn.Module):
    """A latent-variable EBM: joint ``E(x, z) = E_prior(z) + E_dec(x, z)``.

    Couples a **prior** energy over a latent ``z`` (default the standard normal
    ``½‖z‖²``) with a **decoder** energy ``E(x | z)`` to define a joint density
    ``p(x, z) ∝ exp(-E(x, z))``. The data marginal ``p(x) ∝ ∫ exp(-E(x, z)) dz``
    is generally intractable — as with all latent EBMs, you don't evaluate it, you
    *sample the joint*. `sample_joint` runs block Gibbs, alternating an MCMC update
    of ``z`` under its posterior ``E(z | x)`` with an update of ``x`` under
    ``E(x | z)``; the ``x`` marginal of the joint chain is the model's ``p(x)``.

    A linear-Gaussian instance is the conjugate sanity check: prior ``z ~ N(0, I)``
    and decoder ``½‖x − Wz‖²/σ²`` (i.e. ``x | z ~ N(Wz, σ²I)``) give the exact
    marginal ``x ~ N(0, WWᵀ + σ²I)`` and Gaussian posterior ``z | x`` — both
    recovered by the block-Gibbs chain.

    Args:
        decoder: the conditional energy ``E(x | z)`` as a callable ``(x, z) -> (B,)``
            (an ``nn.Module`` is registered so its parameters train and freeze).
        latent_dim: dimensionality of ``z``.
        prior: energy over ``z`` ``(z) -> (B,)``; defaults to the standard normal.
    """

    def __init__(
        self,
        decoder: DecoderEnergyFn,
        latent_dim: int,
        prior: EnergyFn | None = None,
    ):
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        self.latent_dim = latent_dim
        self._decoder = decoder
        self._prior = prior if prior is not None else _standard_normal_energy
        # Register any nn.Module components so params train and freeze during sampling.
        self._components = nn.ModuleList([m for m in (decoder, prior) if isinstance(m, nn.Module)])

    def prior_energy(self, z: Tensor) -> Tensor:
        """Prior energy ``E_prior(z)``."""
        return self._prior(z)

    def decoder_energy(self, x: Tensor, z: Tensor) -> Tensor:
        """Decoder (conditional) energy ``E(x | z)``."""
        return self._decoder(x, z)

    def joint_energy(self, x: Tensor, z: Tensor) -> Tensor:
        """Joint energy ``E(x, z) = E_prior(z) + E(x | z)``."""
        return self._prior(z) + self._decoder(x, z)

    def conditional_energy(self, z: Tensor) -> EnergyFn:
        """The energy ``E(x | z)`` as an `EnergyFn` over ``x`` for fixed ``z``."""
        return lambda x: self._decoder(x, z)

    def posterior_energy(self, x: Tensor) -> EnergyFn:
        """The posterior ``E(z | x) = E_prior(z) + E(x | z)`` (up to a constant), over ``z``.

        This is a valid `EnergyFn` in ``z``: the missing ``x``-only normalizer is
        constant in ``z``, so any sampler targets the exact posterior ``p(z | x)``.
        """
        return lambda z: self._prior(z) + self._decoder(x, z)

    @torch.no_grad()
    def sample_joint(
        self,
        sampler: Sampler,
        x_init: Tensor,
        *,
        z_init: Tensor | None = None,
        steps: int = 100,
        inner_steps: int | None = None,
        latent_sampler: Sampler | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Block-Gibbs sample the joint; returns ``(x, z)``.

        Each outer step updates ``z`` under `posterior_energy` then ``x`` under
        `conditional_energy`, running ``inner_steps`` transitions of ``sampler``
        per block (``sampler.steps`` if ``None``). ``latent_sampler`` overrides the
        sampler used for the ``z`` block. ``z`` starts from the prior sample space
        (standard normal) unless ``z_init`` is given.
        """
        x = x_init.detach().clone()
        z = (
            z_init.detach().clone()
            if z_init is not None
            else torch.randn(x.shape[0], self.latent_dim, device=x.device, dtype=x.dtype)
        )
        z_sampler = latent_sampler if latent_sampler is not None else sampler
        for _ in range(steps):
            z = z_sampler.sample(self.posterior_energy(x), z, steps=inner_steps)
            x = sampler.sample(self.conditional_energy(z), x, steps=inner_steps)
        return x, z

    def sample(
        self,
        sampler: Sampler,
        x_init: Tensor,
        *,
        z_init: Tensor | None = None,
        steps: int = 100,
        inner_steps: int | None = None,
        latent_sampler: Sampler | None = None,
    ) -> Tensor:
        """Block-Gibbs sample and return the data marginal ``x`` (see `sample_joint`)."""
        x, _ = self.sample_joint(
            sampler,
            x_init,
            z_init=z_init,
            steps=steps,
            inner_steps=inner_steps,
            latent_sampler=latent_sampler,
        )
        return x
