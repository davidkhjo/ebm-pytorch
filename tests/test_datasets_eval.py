import math

import pytest
import torch

import ebm
from tests.conftest import quadratic_energy


def _ar1(m, n, rho):
    """M chains of a stationary AR(1): x_t = ρ x_{t-1} + sqrt(1-ρ²) ε, var ≈ 1."""
    eps = torch.randn(m, n)
    x = torch.empty(m, n)
    x[:, 0] = eps[:, 0]
    s = math.sqrt(1 - rho**2)
    for t in range(1, n):
        x[:, t] = rho * x[:, t - 1] + s * eps[:, t]
    return x


def test_split_rhat_iid_near_one_and_flags_nonmixing():
    iid = torch.randn(8, 2000, 3)
    rhat = ebm.eval.split_rhat(iid)
    assert rhat.shape == (3,)
    assert (rhat < 1.02).all()

    # Chains parked at different means never mixed → R̂ ≫ 1.
    offsets = torch.arange(8).reshape(8, 1, 1).double()
    stuck = 0.01 * torch.randn(8, 2000, 1) + offsets
    assert ebm.eval.split_rhat(stuck).item() > 5.0


def test_ess_iid_recovers_full_count():
    iid = torch.randn(8, 2000, 2)
    ess = ebm.eval.effective_sample_size(iid)
    assert ess.shape == (2,)
    # Independent draws: ESS ≈ M·N = 16000.
    assert (ess > 0.85 * 16000).all() and (ess < 1.15 * 16000).all()


@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_ess_matches_ar1_closed_form(rho):
    m, n = 8, 4000
    x = _ar1(m, n, rho)
    ess = ebm.eval.effective_sample_size(x).item()
    predicted = m * n * (1 - rho) / (1 + rho)  # N(1-ρ)/(1+ρ) per chain
    assert abs(ess - predicted) / predicted < 0.2


def test_diagnostics_reject_too_short():
    with pytest.raises(ValueError):
        ebm.eval.effective_sample_size(torch.randn(4, 3, 2))


def test_precision_recall_limits():
    # Two draws from the same distribution → both metrics near 1.
    p, r = ebm.eval.precision_recall(torch.randn(2000, 2), torch.randn(2000, 2))
    assert p > 0.9 and r > 0.9

    # Disjoint supports → both near 0.
    p, r = ebm.eval.precision_recall(torch.randn(1500, 2), torch.randn(1500, 2) + 20.0)
    assert p < 0.05 and r < 0.05

    with pytest.raises(ValueError):
        ebm.eval.precision_recall(torch.randn(3, 2), torch.randn(3, 2), k=3)


def test_precision_recall_detects_mode_drop():
    # Real data has two modes; the generator only covers one.
    left = torch.randn(1000, 2) - torch.tensor([6.0, 0.0])
    right = torch.randn(1000, 2) + torch.tensor([6.0, 0.0])
    real = torch.cat([left, right])
    gen = torch.randn(2000, 2) + torch.tensor([6.0, 0.0])
    p, r = ebm.eval.precision_recall(real, gen)
    assert p > 0.8  # generated samples are realistic (inside the real manifold)
    assert r < 0.65  # but cover only ~half the data (one dropped mode)


def test_fisher_divergence_matches_two_gaussian_closed_form():
    mu1 = torch.tensor([0.0, 0.0])
    s1 = torch.tensor([[1.0, 0.3], [0.3, 1.0]])
    mu2 = torch.tensor([0.5, -0.5])
    s2 = torch.tensor([[1.5, 0.0], [0.0, 0.8]])
    a, b = torch.linalg.inv(s1), torch.linalg.inv(s2)
    m = b - a
    closed = torch.trace(m @ s1 @ m.t()) + (b @ (mu1 - mu2)).pow(2).sum()

    x = torch.randn(20000, 2) @ torch.linalg.cholesky(s1).t() + mu1
    e_a = lambda z: 0.5 * (((z - mu1) @ a) * (z - mu1)).sum(dim=1)
    e_b = lambda z: 0.5 * (((z - mu2) @ b) * (z - mu2)).sum(dim=1)
    assert abs(ebm.eval.fisher_divergence(e_a, e_b, x) - closed.item()) < 0.05
    # Identical models → zero.
    assert ebm.eval.fisher_divergence(e_a, e_a, x) < 1e-6


@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_mutual_information_matches_gaussian_closed_form(rho):
    n = 4000
    x = torch.randn(n, 1)
    y = rho * x + (1 - rho**2) ** 0.5 * torch.randn(n, 1)
    true_mi = -0.5 * math.log(1 - rho**2)
    assert abs(ebm.eval.mutual_information(x, y, epochs=500) - true_mi) < 0.05


def test_inception_score_bounds():
    k = 5
    confident = torch.eye(k)[torch.arange(1000) % k]  # sharp and class-balanced
    assert abs(ebm.eval.inception_score(confident) - k) < 1e-3
    uniform = torch.full((1000, k), 1.0 / k)  # every prediction is the marginal
    assert abs(ebm.eval.inception_score(uniform) - 1.0) < 1e-3
    with pytest.raises(ValueError):
        ebm.eval.inception_score(torch.rand(10))  # not (N, K)


def test_kernel_stein_discrepancy_matches_and_detects_mismatch():
    x = torch.randn(500, 2)  # ~ N(0, I)

    def gaussian(a):
        return lambda z: 0.5 * a * z.pow(2).sum(dim=1)  # score of N(0, 1/a)

    # Correct model: KSD² ≈ 0. Wrong scale: positive and growing with mismatch.
    assert abs(ebm.eval.kernel_stein_discrepancy(gaussian(1.0), x)) < 0.01
    ksd_15 = ebm.eval.kernel_stein_discrepancy(gaussian(1.5), x)
    ksd_20 = ebm.eval.kernel_stein_discrepancy(gaussian(2.0), x)
    assert 0.02 < ksd_15 < ksd_20

    with pytest.raises(ValueError):
        ebm.eval.kernel_stein_discrepancy(gaussian(1.0), torch.randn(1, 2))


def test_classifier_two_sample_test_limits():
    same = ebm.eval.classifier_two_sample_test(torch.randn(1500, 2), torch.randn(1500, 2))
    assert 0.4 < same < 0.6  # indistinguishable → chance accuracy
    diff = ebm.eval.classifier_two_sample_test(torch.randn(1500, 2), torch.randn(1500, 2) + 5.0)
    assert diff > 0.95  # far apart → perfectly separable


def test_bits_per_dim_matches_gaussian_closed_form():
    # Standard normal: E(x)=||x||^2/2, log Z = (D/2) log(2 pi).
    d = 5
    x = torch.randn(32, d)
    log_z = 0.5 * d * math.log(2 * math.pi)
    bpd = ebm.eval.bits_per_dim(quadratic_energy, x, log_z)  # dim defaults to d
    # Closed form: BPD = (E(x) + log Z) / (D log 2).
    expected = (quadratic_energy(x) + log_z) / (d * math.log(2))
    assert bpd.shape == (32,)
    assert torch.allclose(bpd, expected, atol=1e-5)
    # Lower is better: a well-modeled point (near the mode) beats an outlier.
    near = torch.zeros(1, d)
    far = 5 * torch.ones(1, d)
    assert (
        ebm.eval.bits_per_dim(quadratic_energy, near, log_z).item()
        < ebm.eval.bits_per_dim(quadratic_energy, far, log_z).item()
    )
    # Explicit dim overrides the default.
    assert torch.allclose(
        ebm.eval.bits_per_dim(quadratic_energy, x, log_z, dim=d),
        -ebm.eval.log_likelihood(quadratic_energy, x, log_z, dim=d),
    )


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


def test_mnist_like_loaders_offline(tmp_path):
    # Pre-populate the cache with tiny hand-built IDX files: the loaders must
    # parse and scale them without touching the network.
    import gzip

    def idx_images(n):
        raw = b"\x00\x00\x08\x03" + b"".join(d.to_bytes(4, "big") for d in (n, 28, 28))
        return gzip.compress(raw + bytes((i * 37) % 256 for i in range(n * 28 * 28)))

    def idx_labels(values):
        raw = b"\x00\x00\x08\x01" + len(values).to_bytes(4, "big") + bytes(values)
        return gzip.compress(raw)

    for prefix in ("", "fashion-"):
        (tmp_path / f"{prefix}train-images-idx3-ubyte.gz").write_bytes(idx_images(3))
        (tmp_path / f"{prefix}train-labels-idx1-ubyte.gz").write_bytes(idx_labels([5, 0, 9]))

    for loader in (ebm.datasets.mnist, ebm.datasets.fashion_mnist):
        x, y = loader(root=tmp_path, return_labels=True)
        assert x.shape == (3, 1, 28, 28) and x.dtype == torch.float32
        assert x.min() >= -1 and x.max() <= 1
        assert y.tolist() == [5, 0, 9] and y.dtype == torch.int64
        assert torch.equal(loader(root=tmp_path), x)


def test_cifar_parser_roundtrip():
    from ebm.datasets import _parse_cifar_batch

    # Two records: label byte then 3072 image bytes (R, G, B planes).
    rec0 = bytes([3]) + bytes(i % 256 for i in range(3072))
    rec1 = bytes([7]) + bytes((i * 5) % 256 for i in range(3072))
    images, labels = _parse_cifar_batch(rec0 + rec1)
    assert images.shape == (2, 3, 32, 32) and images.dtype == torch.uint8
    assert labels.tolist() == [3, 7]
    # First red-plane pixel is byte 0 of the image payload.
    assert images[0, 0, 0, 0].item() == 0
    assert images[1, 0, 0, 1].item() == 5

    import pytest

    with pytest.raises(ValueError):
        _parse_cifar_batch(b"\x00" * 100)  # not a whole number of records


def test_cifar10_offline(tmp_path):
    # Build a tiny archive with the real member layout; the loader must parse
    # and scale it to [-1, 1] without hitting the network.
    import io
    import tarfile

    def batch_bytes(labels):
        out = bytearray()
        for i, lab in enumerate(labels):
            out += bytes([lab]) + bytes((j + i) % 256 for j in range(3072))
        return bytes(out)

    archive = tmp_path / "cifar-10-binary.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        members = [f"cifar-10-batches-bin/data_batch_{i}.bin" for i in range(1, 6)]
        members.append("cifar-10-batches-bin/test_batch.bin")
        for k, name in enumerate(members):
            payload = batch_bytes([k, k + 1])
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    x, y = ebm.datasets.cifar10(root=tmp_path, return_labels=True)
    assert x.shape == (10, 3, 32, 32) and x.dtype == torch.float32  # 5 train batches x 2
    assert x.min() >= -1 and x.max() <= 1
    assert y[:2].tolist() == [0, 1]
    test_x = ebm.datasets.cifar10(root=tmp_path, train=False)
    assert test_x.shape == (2, 3, 32, 32)


def test_cifar100_parser_roundtrip():
    from ebm.datasets import _parse_cifar100_batch

    # Record: coarse label, fine label, then 3072 image bytes (R, G, B planes).
    rec0 = bytes([11, 3]) + bytes(i % 256 for i in range(3072))
    rec1 = bytes([19, 7]) + bytes((i * 5) % 256 for i in range(3072))
    images, labels = _parse_cifar100_batch(rec0 + rec1)
    assert images.shape == (2, 3, 32, 32) and images.dtype == torch.uint8
    assert labels.tolist() == [3, 7]  # fine label (column 1), not the coarse column 0
    assert images[0, 0, 0, 0].item() == 0  # image payload starts at byte 2
    assert images[1, 0, 0, 1].item() == 5

    import pytest

    with pytest.raises(ValueError):
        _parse_cifar100_batch(b"\x00" * 100)  # not a whole number of records


def test_cifar100_offline(tmp_path):
    import io
    import tarfile

    def batch_bytes(fine_labels):
        out = bytearray()
        for i, lab in enumerate(fine_labels):
            out += bytes([0, lab]) + bytes((j + i) % 256 for j in range(3072))
        return bytes(out)

    archive = tmp_path / "cifar-100-binary.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, labels in (
            ("cifar-100-binary/train.bin", [5, 42, 99]),
            ("cifar-100-binary/test.bin", [7, 13]),
        ):
            payload = batch_bytes(labels)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))

    x, y = ebm.datasets.cifar100(root=tmp_path, return_labels=True)
    assert x.shape == (3, 3, 32, 32) and x.dtype == torch.float32
    assert x.min() >= -1 and x.max() <= 1
    assert y.tolist() == [5, 42, 99] and y.dtype == torch.int64
    assert ebm.datasets.cifar100(root=tmp_path, train=False).shape == (2, 3, 32, 32)


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


def test_mmd_and_frechet_reject_degenerate_inputs():
    import pytest

    g = torch.Generator().manual_seed(3)
    a = torch.randn(4, 2, generator=g)

    # Fewer than 2 samples: the unbiased MMD denominator n(n-1) and the
    # single-sample covariance are undefined — raise instead of returning NaN.
    with pytest.raises(ValueError):
        ebm.eval.mmd(a[:1], a)
    with pytest.raises(ValueError):
        ebm.eval.mmd(a, a[:1])
    with pytest.raises(ValueError):
        ebm.eval.frechet_distance(a[:1], a[:1])

    # Zero RBF bandwidth: the median pairwise distance vanishes when the pooled
    # samples are identical, which would divide by zero in the kernel.
    identical = torch.ones(5, 2)
    with pytest.raises(ValueError):
        ebm.eval.mmd(identical, identical.clone())  # median heuristic -> 0
    with pytest.raises(ValueError):
        ebm.eval.mmd(a, a.clone(), bandwidth=0.0)

    # The 2-sample minimum still computes a finite value (positive control).
    val = ebm.eval.mmd(a[:2], a[2:], bandwidth=1.0)
    assert math.isfinite(val)
    assert math.isfinite(ebm.eval.frechet_distance(a[:2], a[2:]))


def test_two_moons_labels():
    x, y = ebm.datasets.two_moons(200, return_labels=True)
    assert x.shape == (200, 2) and y.shape == (200,)
    assert set(y.unique().tolist()) == {0, 1}
    # Upper moon (label 0) sits higher on average than the lower moon.
    assert x[y == 0][:, 1].mean() > x[y == 1][:, 1].mean()
