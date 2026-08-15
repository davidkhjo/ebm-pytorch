"""MCMC-free EBM training on two-moons with the energy-discrepancy loss.

Energy discrepancy needs no sampler during training — just perturbed-energy
contrasts — so this whole training loop runs without a single MCMC chain. We
still sample at the end (with Langevin) purely to visualize the learned energy.

Run:  python examples/train_energy_discrepancy.py
Outputs energy_discrepancy_result.png next to this script (needs [viz]).
"""

from __future__ import annotations

from pathlib import Path

import torch

import ebm


def main() -> None:
    torch.manual_seed(0)
    data = ebm.datasets.two_moons(8192)

    energy = ebm.nets.MLPEnergy(dim=2, hidden=(128, 128))
    loss_fn = ebm.EnergyDiscrepancy(sigma=0.3, m_particles=16, w_stable=1.0)
    # No sampler, no buffer — the Trainer just backprops the loss each step.
    trainer = ebm.Trainer(energy, loss_fn, lr=1e-3, ema_decay=0.999)
    trainer.fit(data, steps=6000, batch_size=256)

    model = trainer.ema.module
    samples = ebm.LangevinDynamics(step_size=0.01, steps=500).sample(
        model, torch.randn(2000, 2, device=trainer.device)
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ebm.viz.plot_samples(data[:2000], ax=axes[0])
    axes[0].set_title("data")
    ebm.viz.energy_contour(model, bounds=(-2.5, 2.5), device=trainer.device, ax=axes[1])
    axes[1].set_title("learned energy (MCMC-free)")
    ebm.viz.plot_samples(samples, ax=axes[2])
    axes[2].set_title("samples")

    out = Path(__file__).parent / "energy_discrepancy_result.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
