"""Standard 2D toy distributions for developing and testing EBMs.

All functions return a float32 tensor of shape ``(n, 2)``, roughly centered
and on a scale of a few units, with an optional ``generator`` for determinism.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def two_moons(
    n: int,
    noise: float = 0.1,
    generator: torch.Generator | None = None,
    return_labels: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Two interleaved moons; with ``return_labels`` also returns 0/1 moon labels."""
    n_upper = n // 2
    n_lower = n - n_upper
    theta_u = math.pi * torch.rand(n_upper, generator=generator)
    theta_l = math.pi * torch.rand(n_lower, generator=generator)
    upper = torch.stack([torch.cos(theta_u), torch.sin(theta_u)], dim=1)
    lower = torch.stack([1 - torch.cos(theta_l), 0.5 - torch.sin(theta_l)], dim=1)
    x = torch.cat([upper, lower]) + noise * torch.randn(n, 2, generator=generator)
    x = x - torch.tensor([0.5, 0.25])
    y = torch.cat([torch.zeros(n_upper, dtype=torch.long), torch.ones(n_lower, dtype=torch.long)])
    perm = torch.randperm(n, generator=generator)
    x = x[perm].float()
    if return_labels:
        return x, y[perm]
    return x


def eight_gaussians(n: int, std: float = 0.15, generator: torch.Generator | None = None) -> Tensor:
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = 2.0 * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    idx = torch.randint(8, (n,), generator=generator)
    return (centers[idx] + std * torch.randn(n, 2, generator=generator)).float()


def checkerboard(n: int, generator: torch.Generator | None = None) -> Tensor:
    x1 = 4 * torch.rand(n, generator=generator) - 2
    offset = 2 * torch.randint(2, (n,), generator=generator).float()
    x2 = torch.rand(n, generator=generator) + torch.floor(x1) % 2 + offset - 2
    return torch.stack([x1, x2], dim=1).float()


def rings(n: int, noise: float = 0.05, generator: torch.Generator | None = None) -> Tensor:
    radii = torch.tensor([0.7, 1.4, 2.1])
    idx = torch.randint(3, (n,), generator=generator)
    theta = 2 * math.pi * torch.rand(n, generator=generator)
    r = radii[idx] + noise * torch.randn(n, generator=generator)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1).float()


def spirals(n: int, noise: float = 0.08, generator: torch.Generator | None = None) -> Tensor:
    n_a = n // 2
    n_b = n - n_a
    t_a = torch.sqrt(torch.rand(n_a, generator=generator)) * 3 * math.pi
    t_b = torch.sqrt(torch.rand(n_b, generator=generator)) * 3 * math.pi
    arm_a = torch.stack([t_a * torch.cos(t_a), t_a * torch.sin(t_a)], dim=1) / (1.5 * math.pi)
    arm_b = -torch.stack([t_b * torch.cos(t_b), t_b * torch.sin(t_b)], dim=1) / (1.5 * math.pi)
    x = torch.cat([arm_a, arm_b]) + noise * torch.randn(n, 2, generator=generator)
    return x[torch.randperm(n, generator=generator)].float()
