"""Latent-variable EBM: a joint E(x, z) sampled by block Gibbs.

`LatentEBM` couples a prior over a latent `z` with a decoder energy `E(x | z)`
into a joint `p(x, z) ∝ exp(-E(x, z))`. The data marginal `p(x)` is intractable,
so you sample the joint: block Gibbs alternates an MCMC update of `z` under its
posterior with an update of `x` under the decoder. Here a fixed nonlinear decoder
bends a 1-D Gaussian latent into a curved 2-D manifold; because the decoder is
Gaussian we also have a cheap *ancestral* reference (draw z ~ N(0,1), then
x = g(z) + σε), and the block-Gibbs marginal should match it — an MCMC-vs-exact
check on a nontrivial manifold.

Run:  python examples/latent_ebm.py
Outputs latent_ebm_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import ebm

SIGMA = 0.15


class CurveDecoder(nn.Module):
    """Fixed 1-D → 2-D generator g(z); E(x | z) = ‖x − g(z)‖² / (2σ²)."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.g = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 2))
        for p in self.g.parameters():
            p.requires_grad_(False)

    def mean(self, z: torch.Tensor) -> torch.Tensor:
        return self.g(z)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((x - self.g(z)) ** 2).sum(dim=1) / SIGMA**2


def main() -> None:
    torch.manual_seed(0)
    decoder = CurveDecoder()
    model = ebm.LatentEBM(decoder, latent_dim=1)  # standard-normal prior over z

    # Ancestral reference: z ~ N(0, 1), x = g(z) + σε (exact for this Gaussian decoder).
    z_anc = torch.randn(5000, 1)
    x_anc = decoder.mean(z_anc) + SIGMA * torch.randn(5000, 2)

    # Block-Gibbs samples of the same joint (MCMC in both blocks).
    x_gibbs, _ = model.sample_joint(
        ebm.MALA(step_size=0.02, steps=5), torch.randn(5000, 2), steps=300
    )

    mmd = ebm.eval.mmd(x_gibbs, x_anc)
    mmd_ref = ebm.eval.mmd(torch.randn(5000, 2), x_anc)
    print(f"MMD(block-Gibbs, ancestral) = {mmd:.4f}   (vs a Gaussian: {mmd_ref:.4f})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, s, title in (
        (axes[0], x_anc, "ancestral (z→x)"),
        (axes[1], x_gibbs, "block-Gibbs on E(x, z)"),
    ):
        ax.scatter(s[:, 0], s[:, 1], s=4, alpha=0.3, color="#5c50c9")
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.suptitle("Latent EBM: block-Gibbs sampling matches the ancestral marginal")
    out = Path(__file__).parent / "latent_ebm_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
