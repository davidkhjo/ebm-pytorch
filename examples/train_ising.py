"""Sample a 2D Ising lattice with Gibbs-with-Gradients (discrete EBM).

A showcase of the library's *discrete* side: `GibbsWithGradients` is an exact
Metropolis-Hastings sampler for binary data ``x ∈ {0, 1}^D``, and
`nets.IsingEnergy` is a 2D nearest-neighbor lattice energy
``E(x) = -J Σ_⟨i,j⟩ s_i s_j`` with spins ``s = 2x - 1``.

We start from a random binary lattice and run the sampler at three coupling
strengths. Weak coupling stays disordered (salt-and-pepper); strong coupling
condenses into large aligned domains — the ferromagnetic ordering the energy
rewards. The printed neighbor-agreement fraction rises with J.

Self-contained (no dataset), runs on CPU in well under a minute.

Run:  python examples/train_ising.py
Outputs ising_result.png next to this script (requires the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def neighbor_agreement(x):
    """Fraction of right/down neighbor pairs whose spins align."""
    s = 2 * x - 1
    right = (s[:, :, :-1] == s[:, :, 1:]).float().mean()
    down = (s[:, :-1, :] == s[:, 1:, :]).float().mean()
    return 0.5 * (right + down).item()


def main() -> None:
    torch.manual_seed(0)
    size, n = 48, 4
    couplings = [0.1, 0.4, 1.0]

    sampler = ebm.GibbsWithGradients(steps=3000)
    panels = []
    for j in couplings:
        energy = ebm.nets.IsingEnergy(coupling=j)
        x0 = torch.bernoulli(torch.full((n, size, size), 0.5))
        x = sampler.sample(energy, x0)
        agree = neighbor_agreement(x)
        print(f"coupling J={j:>4}:  neighbor agreement {agree:.3f}")
        panels.append((j, agree, x))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (j, agree, x) in zip(axes, panels, strict=True):
        # One lattice per coupling, rendered as a black/white spin grid.
        ebm.viz.show_images(x[:1].unsqueeze(1), nrow=1, ax=ax, rescale=False)
        ax.set_title(f"J = {j}   (agreement {agree:.2f})")
    fig.suptitle("2D Ising lattices sampled with Gibbs-with-Gradients", fontsize=13)
    fig.tight_layout()
    out = Path(__file__).parent / "ising_result.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
