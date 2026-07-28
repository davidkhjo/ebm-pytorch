import torch

import ebm
from tests.conftest import quadratic_energy


def test_dataset_shapes_and_dtypes():
    for fn in (
        ebm.datasets.two_moons,
        ebm.datasets.eight_gaussians,
        ebm.datasets.checkerboard,
        ebm.datasets.rings,
        ebm.datasets.spirals,
    ):
        x = fn(101)
        assert x.shape == (101, 2)
        assert x.dtype == torch.float32
        assert torch.isfinite(x).all()
        assert x.abs().max() < 10


def test_datasets_deterministic_with_generator():
    g1 = torch.Generator().manual_seed(3)
    g2 = torch.Generator().manual_seed(3)
    a = ebm.datasets.two_moons(50, generator=g1)
    b = ebm.datasets.two_moons(50, generator=g2)
    assert torch.equal(a, b)


def test_ood_auroc_separates_by_energy():
    x_in = 0.1 * torch.randn(200, 2)  # near the mode of N(0, I): low energy
    x_out = 5.0 + 0.1 * torch.randn(200, 2)  # far away: high energy
    auroc = ebm.eval.ood_auroc(quadratic_energy, x_in, x_out)
    assert auroc > 0.99


def test_ood_auroc_chance_on_identical_data():
    x = torch.randn(300, 2)
    auroc = ebm.eval.ood_auroc(quadratic_energy, x, x.clone())
    assert abs(auroc - 0.5) < 1e-6


def test_batched_energies():
    x = torch.randn(2500, 2)
    e = ebm.eval.energies(quadratic_energy, x, batch_size=1000)
    assert e.shape == (2500,)
    assert torch.allclose(e, quadratic_energy(x), atol=1e-5)


def test_two_moons_labels():
    x, y = ebm.datasets.two_moons(200, return_labels=True)
    assert x.shape == (200, 2) and y.shape == (200,)
    assert set(y.unique().tolist()) == {0, 1}
    # Upper moon (label 0) sits higher on average than the lower moon.
    assert x[y == 0][:, 1].mean() > x[y == 1][:, 1].mean()
