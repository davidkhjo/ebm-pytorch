"""Score-based generation with NCSN: multi-sigma denoising + annealed Langevin.

A standalone tour of the noise-conditional (NCSN) stack, which the benchmark
only touches in passing. The recipe (Song & Ermon, 2019):

1. `NoiseConditionalMLPEnergy` — an energy `E(x, σ)` conditioned on a noise
   level, whose score `−∇_x E` is trained to denoise.
2. `MultiSigmaDenoisingScoreMatching` over a geometric σ ladder — no MCMC, no
   negatives; each level learns to point noised samples back toward the data.
3. `AnnealedLangevinDynamics` — sample by running Langevin down the ladder,
   from the largest σ (easy, near-Gaussian) to the smallest (sharp).

On the eight-Gaussians target the annealed sampler traverses all modes — the
thing plain single-σ Langevin struggles with. Runs on CPU in ~1 minute.

Run:  python examples/train_ncsn.py
Outputs ncsn_result.png next to this script (requires the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.eight_gaussians(8192)

    sigmas = ebm.geometric_sigmas(2.0, 0.02, 20)
    net = ebm.nets.NoiseConditionalMLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.MultiSigmaDenoisingScoreMatching(sigmas)
    ebm.Trainer(net, loss_fn, lr=1e-3, device="cpu").fit(data, steps=8000, batch_size=256)

    sampler = ebm.AnnealedLangevinDynamics(sigmas, step_size=4e-3, steps_per_sigma=200)
    samples = sampler.sample(net, torch.randn(2000, 2))

    # Learned score field s(x) = -grad_x E(x, sigma_min) on a grid.
    sigma_min = float(sigmas[-1])
    grid = torch.linspace(-3, 3, 20)
    xx, yy = torch.meshgrid(grid, grid, indexing="xy")
    pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).requires_grad_(True)
    e = net(pts, torch.full((len(pts),), sigma_min))
    score = -torch.autograd.grad(e.sum(), pts)[0].detach()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ebm.viz.plot_samples(data[:2000], ax=axes[0], color="tab:blue")
    axes[0].set_title("data (8 Gaussians)")
    axes[1].quiver(
        xx.numpy(),
        yy.numpy(),
        score[:, 0].reshape(20, 20).numpy(),
        score[:, 1].reshape(20, 20).numpy(),
        color="tab:green",
    )
    axes[1].set_aspect("equal")
    axes[1].set_title(f"learned score  −∇E(x, σ={sigma_min:.2f})")
    ebm.viz.plot_samples(samples, ax=axes[2], color="tab:purple")
    axes[2].set_title("annealed-Langevin samples")

    out = Path(__file__).parent / "ncsn_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
