"""A thin convenience trainer.

Everything the trainer does is reproducible in a few lines of vanilla PyTorch
(see the README) — it exists to bundle the boring parts: batching, device
placement, optimizer setup including loss parameters, EMA, and metric history.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn

from ebm.losses.base import LossOutput
from ebm.utils import EMA


class Trainer:
    """Minimal training loop for an energy model and a loss.

    Args:
        energy: Energy network ``(B, *shape) -> (B,)``.
        loss_fn: Any ``(energy, x) -> LossOutput`` callable; if it is an
            ``nn.Module`` its parameters (e.g. NCE's ``log_z``) are optimized too.
        optimizer: Defaults to ``Adam(lr=1e-3)`` over energy + loss parameters.
        ema_decay: If set, maintain an EMA copy of the energy net at
            ``trainer.ema.module``.
        grad_clip_norm: If set, clip the global gradient norm each step.
        callback: Called as ``callback(step, out)`` after each optimizer step.
    """

    def __init__(
        self,
        energy: nn.Module,
        loss_fn: Callable[[nn.Module, Tensor], LossOutput],
        optimizer: torch.optim.Optimizer | None = None,
        lr: float = 1e-3,
        device: torch.device | str | None = None,
        ema_decay: float | None = None,
        grad_clip_norm: float | None = None,
        log_every: int = 100,
        callback: Callable[[int, LossOutput], None] | None = None,
    ):
        self.device = torch.device(device) if device is not None else _default_device()
        self.energy = energy.to(self.device)
        self.loss_fn = loss_fn
        if isinstance(loss_fn, nn.Module):
            loss_fn.to(self.device)

        if optimizer is None:
            params = list(self.energy.parameters())
            if isinstance(loss_fn, nn.Module):
                params += list(loss_fn.parameters())
            optimizer = torch.optim.Adam(params, lr=lr)
        self.optimizer = optimizer

        self._supervised = bool(getattr(loss_fn, "supervised", False))
        self.ema = EMA(self.energy, ema_decay) if ema_decay is not None else None
        self.grad_clip_norm = grad_clip_norm
        self.log_every = log_every
        self.callback = callback
        self.history: list[dict[str, float]] = []
        self.step_count = 0

    def fit(
        self,
        data: Tensor | tuple[Tensor, ...] | Iterable,
        steps: int = 10_000,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Train for ``steps`` optimizer steps; returns the metric history.

        ``data`` may be a single tensor ``(N, *shape)`` (random batches are
        drawn), an ``(X, Y)`` tensor pair for supervised losses (joint random
        batches), or any iterable of batches (e.g. a ``DataLoader``, cycled).
        Losses with a ``supervised = True`` attribute (e.g. ``JEMLoss``) are
        called as ``loss_fn(energy, x, y)``; labels are dropped otherwise.
        """
        batches = _batch_iterator(data, batch_size)
        self.energy.train()
        for _ in range(steps):
            batch = next(batches)
            if isinstance(batch, (tuple, list)):
                x = batch[0].to(self.device)
                y = batch[1].to(self.device) if len(batch) > 1 else None
            else:
                x, y = batch.to(self.device), None
            if self._supervised:
                if y is None:
                    raise ValueError("supervised loss requires (x, y) batches")
                out = self.loss_fn(self.energy, x, y)
            else:
                out = self.loss_fn(self.energy, x)

            self.optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            if self.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.energy.parameters(), self.grad_clip_norm)
            self.optimizer.step()
            if self.ema is not None:
                self.ema.update()

            self.step_count += 1
            self.history.append({"step": self.step_count, **out.metrics})
            if self.callback is not None:
                self.callback(self.step_count, out)
            if verbose and self.step_count % self.log_every == 0:
                shown = {k: v for k, v in out.metrics.items() if k != "step"}
                pretty = "  ".join(f"{k}={v:.4f}" for k, v in shown.items())
                print(f"step {self.step_count}: {pretty}")
        self.energy.eval()
        return self.history


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _batch_iterator(data, batch_size: int):
    if isinstance(data, Tensor):
        n = data.shape[0]
        while True:
            idx = torch.randint(n, (batch_size,))
            yield data[idx]
    elif isinstance(data, (tuple, list)) and all(isinstance(t, Tensor) for t in data):
        n = data[0].shape[0]
        while True:
            idx = torch.randint(n, (batch_size,))
            yield tuple(t[idx] for t in data)
    else:
        while True:
            yielded = False
            for batch in data:
                yielded = True
                yield batch
            if not yielded:
                raise ValueError("data iterable produced no batches")
