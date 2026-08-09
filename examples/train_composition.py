"""Compose trained energies: product of experts, mixture, and tempering.

Because ``p(x) ∝ exp(-E(x))``, arithmetic on energies is arithmetic on
densities. We train two experts on crossing stripes — one horizontal, one
vertical — then combine them *without retraining*:

- ``SumEnergy(A, B)``  → ``p_A · p_B``  → mass only where **both** agree
  (the intersection: a blob where the stripes cross).
- ``MixtureEnergy(A, B)`` → ``p_A + p_B`` → mass where **either** fires
  (the union: the full plus-shape).
- ``TemperedEnergy(A, T)`` → ``p_A^{1/T}`` → flatten (``T>1``) or sharpen.

Every composition is itself an energy function, so the same
``LangevinDynamics`` samples all of them. Runs on CPU in ~1 minute.

Run:  python examples/train_composition.py
Outputs composition_result.png next to this script (requires the [viz] extra).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def stripe(n, horizontal, generator):
    """A stripe of points: long axis uniform in [-2, 2], short axis N(0, 0.15)."""
    long = 4 * torch.rand(n, 1, generator=generator) - 2
    short = 0.15 * torch.randn(n, 1, generator=generator)
    return torch.cat([long, short] if horizontal else [short, long], dim=1)


def train_expert(data):
    energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.ContrastiveDivergence(
        ebm.LangevinDynamics(step_size=0.01, steps=60),
        buffer=ebm.ReplayBuffer(capacity=4096, shape=(2,)),
        energy_reg=0.1,
    )
    trainer = ebm.Trainer(energy, loss_fn, lr=1e-3, ema_decay=0.999)
    trainer.fit(data, steps=3000, batch_size=256, verbose=False)
    return trainer.ema.module, trainer.device


def main() -> None:
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    data_h = stripe(4096, horizontal=True, generator=g)
    data_v = stripe(4096, horizontal=False, generator=g)

    expert_h, device = train_expert(data_h)
    expert_v, _ = train_expert(data_v)

    product = ebm.SumEnergy(expert_h, expert_v)  # p_H * p_V  -> intersection
    mixture = ebm.MixtureEnergy(expert_h, expert_v)  # p_H + p_V  -> union
    sharp = ebm.TemperedEnergy(mixture, temperature=0.25)  # sharpen the union

    sampler = ebm.LangevinDynamics(step_size=0.01, steps=500)
    noise = torch.randn(2000, 2, device=device)
    samples = {
        "product (intersection)": sampler.sample(product, noise),
        "mixture (union)": sampler.sample(mixture, noise),
        "tempered mixture (T=0.25)": sampler.sample(sharp, noise),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    ebm.viz.plot_samples(data_h[:1500], ax=axes[0, 0], color="tab:blue")
    ebm.viz.plot_samples(data_v[:1500], ax=axes[0, 0], color="tab:orange")
    axes[0, 0].set_title("two experts' data (H + V stripes)")
    for ax, (energy, name) in zip(
        axes[0, 1:], [(product, "product energy"), (mixture, "mixture energy")], strict=True
    ):
        ebm.viz.energy_contour(energy, bounds=(-2.5, 2.5), device=device, ax=ax)
        ax.set_title(name)
    for ax, (name, s) in zip(axes[1], samples.items(), strict=True):
        ebm.viz.plot_samples(s, ax=ax, color="tab:purple")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_title(name)

    out = Path(__file__).parent / "composition_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
