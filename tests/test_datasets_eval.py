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


def test_frechet_distance_zero_on_identical_sets():
    x = torch.randn(3000, 2)
    assert ebm.eval.frechet_distance(x, x.clone()) < 1e-8


def test_frechet_distance_matches_gaussian_closed_form():
    # Diagonal Gaussians: FD = ||mu1 - mu2||^2 + sum_i (s1_i - s2_i)^2.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(20000, 2, generator=g)  # N(0, I)
    shift = torch.tensor([3.0, 0.0])
    scale = torch.tensor([1.0, 2.0])
    y = scale * torch.randn(20000, 2, generator=g) + shift  # N(shift, diag(1, 4))
    expected = 3.0**2 + (2.0 - 1.0) ** 2
    fd = ebm.eval.frechet_distance(x, y)
    assert abs(fd - expected) < 0.3
    # Symmetric in its arguments.
    assert abs(ebm.eval.frechet_distance(y, x) - fd) < 1e-6


def test_frechet_distance_feature_fn_and_shapes():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1000, 3, 2, generator=g)  # non-flat event shape
    y = torch.randn(1000, 3, 2, generator=g) + 1.0
    plain = ebm.eval.frechet_distance(x, y)
    assert plain > 0
    # An identity feature_fn (with batching) must agree with the direct path.
    ident = ebm.eval.frechet_distance(x, y, feature_fn=lambda t: t, batch_size=256)
    assert abs(ident - plain) < 1e-8
    # A projection feature works and changes the value.
    w = torch.randn(6, 4, generator=g)
    proj = ebm.eval.frechet_distance(x, y, feature_fn=lambda t: t.reshape(len(t), -1) @ w)
    assert proj > 0


def test_idx_parser_roundtrip():
    from ebm.datasets import _parse_idx

    # Hand-built IDX: magic 0x00000803 (uint8, 3 dims), shape (2, 3, 4).
    payload = bytes(range(24))
    raw = b"\x00\x00\x08\x03" + (2).to_bytes(4, "big") + (3).to_bytes(4, "big")
    raw += (4).to_bytes(4, "big") + payload
    t = _parse_idx(raw)
    assert t.shape == (2, 3, 4)
    assert t.dtype == torch.uint8
    assert t.flatten().tolist() == list(range(24))

    # 1-D labels variant.
    labels = _parse_idx(b"\x00\x00\x08\x01" + (5).to_bytes(4, "big") + bytes([7, 2, 1, 0, 4]))
    assert labels.tolist() == [7, 2, 1, 0, 4]

    import pytest

    with pytest.raises(ValueError):
        _parse_idx(b"\x00\x00\x0d\x01" + (1).to_bytes(4, "big") + b"\x00")


def test_mmd_zero_for_same_distribution_positive_for_blur():
    g = torch.Generator().manual_seed(2)
    moons_a = ebm.datasets.two_moons(1500, generator=g)
    moons_b = ebm.datasets.two_moons(1500, generator=g)
    same = ebm.eval.mmd(moons_a, moons_b)
    assert abs(same) < 5e-3  # unbiased estimate can dip slightly below zero

    # A blurred copy keeps mean/cov (frechet_distance barely moves), but MMD at
    # a structure-scale bandwidth sees it clearly.
    blurred = moons_a + 0.3 * torch.randn(1500, 2, generator=g)
    same_03 = ebm.eval.mmd(moons_a, moons_b, bandwidth=0.3)
    blur_03 = ebm.eval.mmd(moons_a, blurred, bandwidth=0.3)
    assert blur_03 > 10 * abs(same_03)
    assert blur_03 > 5e-3
    assert ebm.eval.frechet_distance(moons_a, blurred) < 0.2  # FD is nearly blind here


def test_two_moons_labels():
    x, y = ebm.datasets.two_moons(200, return_labels=True)
    assert x.shape == (200, 2) and y.shape == (200,)
    assert set(y.unique().tolist()) == {0, 1}
    # Upper moon (label 0) sits higher on average than the lower moon.
    assert x[y == 0][:, 1].mean() > x[y == 1][:, 1].mean()
