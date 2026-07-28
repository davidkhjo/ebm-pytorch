"""JEM on labeled two-moons: one network that classifies AND generates.

Run:  python examples/train_jem.py
Outputs jem_result.png next to this script (requires the [viz] extra).
"""

from pathlib import Path

import torch
from torch import nn

import ebm


def main() -> None:
    torch.manual_seed(0)
    x, y = ebm.datasets.two_moons(8192, return_labels=True)

    energy = ebm.ClassifierEnergy(
        nn.Sequential(
            nn.Linear(2, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2)
        )
    )
    loss_fn = ebm.JEMLoss(
        ebm.ContrastiveDivergence(
            ebm.LangevinDynamics(step_size=0.01, steps=40),
            buffer=ebm.ReplayBuffer(8192, (2,)),
            energy_reg=0.1,
        )
    )
    trainer = ebm.Trainer(energy, loss_fn, lr=1e-3, ema_decay=0.999)
    trainer.fit((x, y), steps=4000, batch_size=256)

    with torch.no_grad():
        acc = (energy.logits(x.to(trainer.device)).argmax(1).cpu() == y).float().mean()
    print(f"classifier accuracy: {acc:.3f}")

    ema = trainer.ema.module
    sampler = ebm.LangevinDynamics(step_size=0.01, steps=400)
    noise = torch.randn(1000, 2, device=trainer.device)
    cond = {k: sampler.sample(ema.condition(k), noise) for k in (0, 1)}

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for k, color in ((0, "tab:blue"), (1, "tab:orange")):
        ebm.viz.plot_samples(x[y == k][:1000], ax=axes[0], color=color)
    axes[0].set_title("data (colored by class)")
    ebm.viz.energy_contour(ema, bounds=(-2.5, 2.5), device=trainer.device, ax=axes[1])
    axes[1].set_title("marginal energy E(x)")
    for k, color in ((0, "tab:blue"), (1, "tab:orange")):
        ebm.viz.plot_samples(cond[k], ax=axes[2], color=color)
    axes[2].set_title("class-conditional samples")

    out = Path(__file__).parent / "jem_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
