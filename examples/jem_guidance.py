"""Classifier-free guidance on an energy classifier — sharpening class selection.

`ebm.ClassifierEnergy` turns any classifier into an energy with a marginal `E(x)`
and per-class conditionals `E(x|y)`. Classifier-free guidance combines them,
`Ẽ_w(x|y) = (1+w)E(x|y) − w E(x)`, to concentrate samples on the target class.
The effect is only visible when the classes *overlap* (a confident, well-separated
classifier needs no guidance), so we use two overlapping Gaussian classes and
watch guidance suppress the wrong mode as `w` grows. The identical
`energy.guide(y, w)` call works on a trained JEM (`ebm.JEMLoss`).

Run:  python examples/jem_guidance.py
Outputs jem_guidance_result.png next to this script (needs the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import ebm

MU = torch.tensor([[-1.5, 0.0], [1.5, 0.0]])  # two overlapping class centres


class GaussianLogits(nn.Module):
    """Logits[:, k] = -½‖x - μ_k‖², so E(x|k) = ½‖x - μ_k‖² and the marginal is the mixture."""

    def forward(self, x):
        return torch.stack([-0.5 * ((x - MU[k]) ** 2).sum(1) for k in range(2)], dim=1)


def main() -> None:
    energy = ebm.ClassifierEnergy(GaussianLogits())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, w in zip(axes, (0.0, 2.0, 6.0), strict=True):
        torch.manual_seed(1)
        samples = ebm.MALA(step_size=0.05, steps=400).sample(
            energy.guide(0, w), 2 * torch.randn(3000, 2)
        )
        frac = (samples[:, 0] < 0).float().mean().item()  # class-0 mode sits at x0 < 0
        ax.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.35, color="#5c50c9")
        ax.scatter(*MU.t(), s=120, marker="x", color="#c95c50")
        ax.set_title(f"guidance w = {w:.0f}   (correct-mode frac {frac:.3f})")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-3, 3)
        print(f"w={w:.0f}  P(correct mode) = {frac:.3f}")
    fig.suptitle("Classifier-free guidance suppresses the wrong class as w grows")
    out = Path(__file__).parent / "jem_guidance_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
