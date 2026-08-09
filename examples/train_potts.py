"""Sample a 2D Potts lattice with categorical Gibbs-with-Gradients.

The categorical counterpart of the Ising example. `nets.PottsEnergy` is the
K-color generalization of `IsingEnergy` — neighbors that share a color lower
the energy, ``E(x) = -J Σ_⟨i,j⟩ 1[c_i = c_j]`` — and
`CategoricalGibbsWithGradients` is an exact Metropolis-Hastings sampler for
one-hot data ``(B, H, W, K)``.

Starting from a random K-color lattice and running the sampler at three
coupling strengths reproduces the Potts phase behavior: a disordered speckle
(neighbor agreement near 1/K) condenses into large same-color domains. The
printed agreement fraction rises with the coupling J.

Self-contained (no dataset), runs on CPU in well under a minute.

Run:  python examples/train_potts.py
Outputs potts_result.png next to this script (requires the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm

K = 5  # number of colors


def onehot_lattice(n, size, k):
    idx = torch.randint(k, (n, size, size))
    return torch.nn.functional.one_hot(idx, k).float()


def neighbor_agreement(x):
    """Fraction of right/down neighbor pairs that share a color."""
    c = x.argmax(-1)
    right = (c[:, :, :-1] == c[:, :, 1:]).float().mean()
    down = (c[:, :-1, :] == c[:, 1:, :]).float().mean()
    return 0.5 * (right + down).item()


def main() -> None:
    torch.manual_seed(0)
    size = 40
    couplings = [0.2, 0.8, 2.0]

    sampler = ebm.CategoricalGibbsWithGradients(steps=4000)
    panels = []
    for j in couplings:
        energy = ebm.nets.PottsEnergy(coupling=j)
        x = sampler.sample(energy, onehot_lattice(1, size, K))
        agree = neighbor_agreement(x)
        print(f"coupling J={j:>4}:  neighbor agreement {agree:.3f}  (random ≈ {1 / K:.2f})")
        panels.append((j, agree, x.argmax(-1)[0]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (j, agree, colors) in zip(axes, panels, strict=True):
        ax.imshow(colors, cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
        ax.set_title(f"J = {j}   (agreement {agree:.2f})")
        ax.axis("off")
    fig.suptitle(f"2D {K}-state Potts lattices via categorical Gibbs-with-Gradients", fontsize=13)
    fig.tight_layout()
    out = Path(__file__).parent / "potts_result.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
