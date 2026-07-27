"""Train an EBM on the two-moons distribution and plot the learned energy.

Run:  python examples/train_two_moons.py
Outputs two_moons_result.png next to this script (requires the [viz] extra).
"""

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.two_moons(8192)

    energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    sampler = ebm.LangevinDynamics(step_size=0.01, steps=60)
    loss_fn = ebm.ContrastiveDivergence(
        sampler,
        buffer=ebm.ReplayBuffer(capacity=8192, shape=(2,)),
        energy_reg=0.1,
    )

    trainer = ebm.Trainer(energy, loss_fn, lr=1e-3, ema_decay=0.999)
    trainer.fit(data, steps=6000, batch_size=256)

    # Sample from scratch with more steps than used in training.
    long_sampler = ebm.LangevinDynamics(step_size=0.01, steps=500)
    samples = long_sampler.sample(trainer.ema.module, torch.randn(2000, 2, device=trainer.device))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ebm.viz.plot_samples(data[:2000], ax=axes[0])
    axes[0].set_title("data")
    ebm.viz.energy_contour(
        trainer.ema.module, bounds=(-2.5, 2.5), device=trainer.device, ax=axes[1]
    )
    axes[1].set_title("learned energy")
    ebm.viz.plot_samples(samples, ax=axes[2])
    axes[2].set_title("samples")

    out = Path(__file__).parent / "two_moons_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
