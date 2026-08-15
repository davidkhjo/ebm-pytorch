"""Train a Bernoulli RBM on binary "bars" images with CD-1, exactly.

Generates simple horizontal/vertical bar patterns on a small grid, fits an RBM
by contrastive divergence (the free-energy gradient *is* the RBM ML gradient),
then draws samples by block Gibbs. Because the model is small, we also print the
exact log-partition function — the closed-form check that makes RBMs the most
verifiable EBM.

Run:  python examples/train_rbm.py
Outputs train_rbm_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm

GRID = 6  # 6x6 images -> 36 visible units


def bars_dataset(n: int) -> torch.Tensor:
    """Random single horizontal or vertical bars on a GRID x GRID canvas."""
    imgs = torch.zeros(n, GRID, GRID)
    vertical = torch.rand(n) < 0.5
    idx = torch.randint(0, GRID, (n,))
    for i in range(n):
        if vertical[i]:
            imgs[i, :, idx[i]] = 1.0
        else:
            imgs[i, idx[i], :] = 1.0
    return imgs.reshape(n, GRID * GRID)


def main() -> None:
    torch.manual_seed(0)
    data = bars_dataset(4096)

    # 16 hidden units: enough for the 12 bar patterns, and small enough that the
    # exact log Z (which enumerates the smaller layer, 2^16 states) is tractable.
    rbm = ebm.nets.RBM(n_visible=GRID * GRID, n_hidden=16)
    opt = torch.optim.Adam(rbm.parameters(), lr=1e-2)
    for step in range(3000):
        batch = data[torch.randint(0, len(data), (128,))]
        v_neg = rbm.gibbs_step(batch)  # CD-1: one block-Gibbs step from the data
        loss = rbm(batch).mean() - rbm(v_neg).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step:4d}  energy_gap {loss.item():+.3f}")

    print(f"exact log Z = {rbm.log_z().item():.3f}")

    # Sample from the model by a longer Gibbs chain started from noise.
    v = torch.bernoulli(torch.full((64, GRID * GRID), 0.5))
    for _ in range(500):
        v = rbm.gibbs_step(v)
    samples = v.reshape(64, 1, GRID, GRID)

    import matplotlib

    matplotlib.use("Agg")
    ax = ebm.viz.show_images(samples, nrow=8, title="RBM samples (block Gibbs from noise)")
    out = Path(__file__).parent / "train_rbm_result.png"
    ax.figure.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
