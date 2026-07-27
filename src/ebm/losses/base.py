"""Shared loss output container.

All losses follow one convention: ``loss_fn(energy, x) -> LossOutput`` where
``energy`` maps ``(B, *shape) -> (B,)`` and ``x`` is a data batch. Losses are
``nn.Module``s so that losses with learnable parameters (e.g. NCE's ``log_z``)
compose with optimizers uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class LossOutput:
    """Result of a loss evaluation.

    Attributes:
        loss: Scalar to call ``.backward()`` on.
        metrics: Detached diagnostics for logging. Note the CD loss *value* is
            not a meaningful training signal — monitor ``energy_gap`` instead.
        x_neg: Negative samples used (detached), when the loss produces them.
    """

    loss: Tensor
    metrics: dict[str, float] = field(default_factory=dict)
    x_neg: Tensor | None = None
