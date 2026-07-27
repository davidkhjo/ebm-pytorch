"""Shared utilities: parameter freezing and exponential moving averages."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import nn


@contextmanager
def frozen_params(module: nn.Module | None) -> Iterator[None]:
    """Temporarily set ``requires_grad=False`` on all parameters of ``module``.

    Used while running MCMC so that gradients are taken with respect to the
    samples only, avoiding building autograd graphs over the parameters.
    Accepts ``None`` (no-op) so samplers work with plain callables too.
    """
    if module is None:
        yield
        return
    states = [(p, p.requires_grad) for p in module.parameters()]
    for p, _ in states:
        p.requires_grad_(False)
    try:
        yield
    finally:
        for p, state in states:
            p.requires_grad_(state)


class EMA:
    """Exponential moving average of a module's parameters.

    EMA weights consistently produce better samples for MCMC-trained EBMs.

    Example:
        ema = EMA(energy, decay=0.999)
        for ...:
            ...train step...
            ema.update()
        sampler.sample(ema.module, x0)  # sample with averaged weights
    """

    def __init__(self, module: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self._source = module
        self.module = copy.deepcopy(module)
        self.module.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self) -> None:
        for ema_p, p in zip(self.module.parameters(), self._source.parameters(), strict=True):
            ema_p.lerp_(p.detach(), 1.0 - self.decay)
        for ema_b, b in zip(self.module.buffers(), self._source.buffers(), strict=True):
            ema_b.copy_(b)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "module": self.module.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.module.load_state_dict(state["module"])
