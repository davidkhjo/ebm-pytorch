"""Parallel tempering (replica exchange) over a wrapped base sampler."""

from __future__ import annotations

import torch
from torch import Tensor

from ebm._functional import mh_accept
from ebm.energy import EnergyFn
from ebm.samplers.base import Sampler


class _Tempered:
    """``energy(x) / T`` as a plain callable, so any base sampler targets ``exp(-E/T)``."""

    def __init__(self, energy: EnergyFn, temperature: float):
        self.energy = energy
        self.temperature = temperature

    def __call__(self, x: Tensor) -> Tensor:
        return self.energy(x) / self.temperature


class ParallelTempering(Sampler):
    """Replica exchange: run a temperature ladder and swap adjacent replicas.

    Wraps a ``base_sampler`` and runs one chain per temperature
    ``1 = T_0 < T_1 < …``, each targeting ``p^{1/T} ∝ exp(-E/T)``. Hot replicas
    cross energy barriers a single cold chain cannot; periodic swaps carry that
    mixing down to the ``T=0`` chain, whose states are returned. Each sweep
    advances every replica with the base sampler, then proposes swaps of
    alternating adjacent pairs, accepting ``(k, k+1)`` with

    ``log α = (1/T_k - 1/T_{k+1}) · (E(x_k) - E(x_{k+1}))``

    (detailed balance for the joint tempered target). ``swap_rates()`` reports
    the per-pair acceptance — aim for ~0.2–0.6 by spacing the ladder
    geometrically; too low means insert a rung.

    Args:
        base_sampler: sampler used to advance each replica one step per sweep.
        temperatures: ascending ladder; ``temperatures[0]`` (the target) should
            be 1.0.
        swap_every: propose swaps every this many sweeps.
        steps: default number of sweeps per ``sample`` call.
    """

    def __init__(
        self,
        base_sampler: Sampler,
        temperatures: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
        swap_every: int = 1,
        steps: int = 100,
    ):
        super().__init__(steps)
        temps = [float(t) for t in temperatures]
        if len(temps) < 2:
            raise ValueError("need at least 2 temperatures")
        if any(t <= 0 for t in temps):
            raise ValueError("temperatures must be positive")
        if temps != sorted(temps):
            raise ValueError("temperatures must be ascending (T0 the target, then hotter)")
        if swap_every < 1:
            raise ValueError("swap_every must be >= 1")
        self.base = base_sampler
        self.temperatures = temps
        self.swap_every = swap_every
        self.n_replicas = len(temps)
        self._replicas: list[Tensor] | None = None
        self._sweep = 0
        self._swap_accept = torch.zeros(self.n_replicas - 1)
        self._swap_count = torch.zeros(self.n_replicas - 1)

    def sample(
        self,
        energy: EnergyFn,
        x_init: Tensor,
        *,
        steps: int | None = None,
        return_trajectory: bool = False,
    ) -> Tensor:
        self._replicas = [x_init.detach().clone() for _ in range(self.n_replicas)]
        self._sweep = 0
        self._swap_accept = torch.zeros(self.n_replicas - 1)
        self._swap_count = torch.zeros(self.n_replicas - 1)
        return super().sample(energy, x_init, steps=steps, return_trajectory=return_trajectory)

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        # One sweep. The passed ``x`` is the base class's copy of the cold chain;
        # the full ensemble lives in ``self._replicas`` and is what we advance.
        assert self._replicas is not None, "call sample(); ParallelTempering owns its replicas"
        for k in range(self.n_replicas):
            tempered = _Tempered(energy, self.temperatures[k])
            self._replicas[k] = self.base.step(tempered, self._replicas[k]).detach()
        self._sweep += 1
        if self._sweep % self.swap_every == 0:
            self._swap(energy)
        return self._replicas[0]

    def _swap(self, energy: EnergyFn) -> None:
        reps = self._replicas
        assert reps is not None
        with torch.no_grad():
            e = [energy(r) for r in reps]  # untempered energies, each (B,)
        parity = self._sweep % 2  # alternate the even / odd adjacent pair sets
        for k in range(parity, self.n_replicas - 1, 2):
            d_beta = 1.0 / self.temperatures[k] - 1.0 / self.temperatures[k + 1]
            log_alpha = d_beta * (e[k] - e[k + 1])
            accept = torch.log(torch.rand_like(log_alpha)) < log_alpha
            mask = accept.reshape(-1, *([1] * (reps[k].dim() - 1)))
            rk, rk1 = reps[k], reps[k + 1]
            reps[k], reps[k + 1] = torch.where(mask, rk1, rk), torch.where(mask, rk, rk1)
            e[k], e[k + 1] = (
                torch.where(accept, e[k + 1], e[k]),
                torch.where(accept, e[k], e[k + 1]),
            )
            self._swap_accept[k] += accept.float().mean().cpu()
            self._swap_count[k] += 1

    def swap_rates(self) -> list[float]:
        """Mean acceptance of each adjacent swap pair over the last ``sample`` call."""
        return (self._swap_accept / self._swap_count.clamp(min=1)).tolist()


class TemperedTransitions(Sampler):
    """Tempered transitions: single-chain barrier crossing (Neal 1996).

    A Metropolis move that melts and refreezes: it drives one chain up a
    temperature ladder to a hot, freely-mixing distribution and back down to the
    target, accumulating a log-weight along the way and doing a single
    accept/reject at the end. Unlike `ParallelTempering` it keeps no replicas —
    one chain crosses barriers by annealing. Each rung applies ``kernel_steps`` of
    the wrapped ``base_sampler`` at that temperature.

    Args:
        base_sampler: sampler used for the within-rung moves.
        temperatures: ascending ladder; ``temperatures[0]`` must be 1.0 (the
            target), the rest hotter.
        kernel_steps: base-sampler steps applied at each rung.
        steps: number of tempered-transition moves per ``sample`` call.
    """

    def __init__(
        self,
        base_sampler: Sampler,
        temperatures: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 16.0),
        kernel_steps: int = 5,
        steps: int = 100,
    ):
        super().__init__(steps)
        temps = [float(t) for t in temperatures]
        if len(temps) < 2:
            raise ValueError("need at least 2 temperatures")
        if any(t <= 0 for t in temps):
            raise ValueError("temperatures must be positive")
        if temps != sorted(temps):
            raise ValueError("temperatures must be ascending (T0 the target, then hotter)")
        if abs(temps[0] - 1.0) > 1e-6:
            raise ValueError("temperatures[0] must be 1.0 (the target distribution)")
        if kernel_steps < 1:
            raise ValueError("kernel_steps must be >= 1")
        self.base = base_sampler
        self.temperatures = temps
        self.betas = [1.0 / t for t in temps]  # descending inverse temperatures
        self.kernel_steps = kernel_steps

    def step(self, energy: EnergyFn, x: Tensor) -> Tensor:
        x_prop, log_w = self._transition(energy, x)
        accept = torch.log(torch.rand_like(log_w)) < log_w
        self._last_accept = accept.float().mean()
        return mh_accept(x.detach(), x_prop, accept)

    def _kernel(self, energy: EnergyFn, x: Tensor, temperature: float) -> Tensor:
        tempered = _Tempered(energy, temperature)
        for _ in range(self.kernel_steps):
            x = self.base.step(tempered, x).detach()
        return x

    def _transition(self, energy: EnergyFn, x: Tensor) -> tuple[Tensor, Tensor]:
        x = x.detach().clone()
        log_w = x.new_zeros(x.shape[0])
        betas, temps, n = self.betas, self.temperatures, len(self.betas)
        # Up the ladder (cold -> hot), then back down (Neal's telescoping weight).
        for i in range(n - 1):
            with torch.no_grad():
                log_w = log_w + (betas[i] - betas[i + 1]) * energy(x)
            x = self._kernel(energy, x, temps[i + 1])
        for i in range(n - 1, 0, -1):
            with torch.no_grad():
                log_w = log_w + (betas[i] - betas[i - 1]) * energy(x)
            x = self._kernel(energy, x, temps[i - 1])
        return x, log_w
