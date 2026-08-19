"""Shared building blocks for the energy networks."""

from __future__ import annotations

from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


def _maybe_sn(layer: nn.Module, enabled: bool) -> nn.Module:
    return spectral_norm(layer) if enabled else layer
