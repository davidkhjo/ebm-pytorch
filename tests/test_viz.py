import pytest
import torch

import ebm

pytest.importorskip("matplotlib")


def test_show_images_grayscale_and_rgb():
    import matplotlib

    matplotlib.use("Agg")

    # 5 grayscale images into an nrow=3 grid => 2 rows, last row padded.
    gray = torch.rand(5, 1, 8, 8) * 2 - 1
    ax = ebm.viz.show_images(gray, nrow=3)
    img = ax.images[0].get_array()
    assert img.shape == (2 * 8, 3 * 8)  # (nrows*H, ncol*W)

    # RGB path: 6 images into nrow=3 => exactly 2 full rows, HxWx3.
    rgb = torch.rand(6, 3, 8, 8) * 2 - 1
    ax2 = ebm.viz.show_images(rgb, nrow=3, title="rgb")
    arr = ax2.images[0].get_array()
    assert arr.shape == (2 * 8, 3 * 8, 3)
    assert float(arr.min()) >= 0.0 and float(arr.max()) <= 1.0  # rescaled + clamped
    assert ax2.get_title() == "rgb"


def test_show_images_rejects_bad_shape():
    with pytest.raises(ValueError):
        ebm.viz.show_images(torch.rand(4, 2, 8, 8))  # C=2 unsupported
    with pytest.raises(ValueError):
        ebm.viz.show_images(torch.rand(4, 8, 8))  # not 4D


def test_energy_contour_plot_samples_and_histogram():
    import matplotlib

    matplotlib.use("Agg")
    from tests.conftest import quadratic_energy

    # energy_contour renders a filled contour of a 2D energy.
    ax = ebm.viz.energy_contour(quadratic_energy, bounds=(-2.0, 2.0), resolution=40)
    assert len(ax.collections) > 0

    # plot_samples scatters 2D points onto (optionally shared) axes.
    ax2 = ebm.viz.plot_samples(torch.randn(200, 2))
    assert len(ax2.collections) == 1
    assert ebm.viz.plot_samples(torch.randn(50, 2), ax=ax2) is ax2  # reuses the axes

    # energy_histogram overlays one histogram per labelled batch.
    ax3 = ebm.viz.energy_histogram(
        quadratic_energy, {"data": torch.randn(300, 2), "model": torch.randn(300, 2) + 1}
    )
    assert ax3.get_xlabel() == "energy"
    assert len(ax3.get_legend().get_texts()) == 2
