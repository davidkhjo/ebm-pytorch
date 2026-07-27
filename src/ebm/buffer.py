"""Replay buffer of persistent MCMC chains (persistent contrastive divergence)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

InitFn = Callable[[tuple[int, ...]], Tensor]
"""Maps a shape ``(B, *event_shape)`` to a fresh (CPU) initialization tensor."""


class ReplayBuffer:
    """Stores past negative samples so MCMC chains persist across training steps.

    Follows the Du & Mordatch (2019) recipe: sample chain starts uniformly from
    the buffer, reinitialize each with probability ``reinit_prob`` (default 5%),
    and after sampling write the results back to the same slots via ``push``.
    Contents are kept detached on CPU.

    Args:
        capacity: Number of stored chains. Should be much larger than the batch
            size (e.g. 10_000 vs. 128).
        shape: Event shape of a single sample, e.g. ``(2,)`` or ``(3, 32, 32)``.
        reinit_prob: Per-sample probability of restarting a chain from ``init_fn``.
        init_fn: Fresh-start distribution; defaults to standard normal.
    """

    def __init__(
        self,
        capacity: int,
        shape: tuple[int, ...],
        reinit_prob: float = 0.05,
        init_fn: InitFn | None = None,
    ):
        self.capacity = capacity
        self.shape = tuple(shape)
        self.reinit_prob = reinit_prob
        self.init_fn = init_fn if init_fn is not None else torch.randn
        self.data = self.init_fn((capacity, *self.shape)).detach().cpu()
        self._last_idx: Tensor | None = None

    def sample(self, batch_size: int, device: torch.device | str | None = None) -> Tensor:
        """Draw chain starts; remembers the drawn slots for the next ``push``."""
        idx = torch.randint(self.capacity, (batch_size,))
        x = self.data[idx].clone()
        reinit = torch.rand(batch_size) < self.reinit_prob
        if reinit.any():
            x[reinit] = self.init_fn((int(reinit.sum()), *self.shape)).to(x.dtype)
        self._last_idx = idx
        return x.to(device) if device is not None else x

    def push(self, x: Tensor, idx: Tensor | None = None) -> None:
        """Write samples back to the buffer (by default, the slots last drawn)."""
        if idx is None:
            idx = self._last_idx
        if idx is None:
            raise RuntimeError("push() called before sample() and without explicit idx")
        if x.shape[0] != idx.shape[0]:
            raise ValueError(f"got {x.shape[0]} samples for {idx.shape[0]} slots")
        self.data[idx] = x.detach().cpu()

    def __len__(self) -> int:
        return self.capacity

    def state_dict(self) -> dict:
        return {"data": self.data, "reinit_prob": self.reinit_prob}

    def load_state_dict(self, state: dict) -> None:
        self.data = state["data"].detach().cpu()
        self.reinit_prob = state["reinit_prob"]
