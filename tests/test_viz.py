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
