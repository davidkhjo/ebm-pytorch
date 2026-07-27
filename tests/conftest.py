import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def quadratic_energy(x):
    """Standard normal energy: E(x) = ||x||^2 / 2, so p = N(0, I)."""
    return 0.5 * x.pow(2).flatten(1).sum(dim=1)
